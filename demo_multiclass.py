#!/usr/bin/env python3
"""
Demo: Multi-class inference with SAM3.

This script demonstrates how to use Sam3MultiClassPredictor to detect and
segment multiple object classes efficiently.  The backbone (80% of compute)
runs once, and the lightweight encoder+decoder runs per class.

Use --fast to enable the optimized predictor (batched forward, torch.compile,
FP16, presence-based early exit).

Use --single-pass for the fastest mode: one encoder+decoder+masks pass with
cosine-based class assignment (no per-class passes at all).

Usage:
    python demo_multiclass.py --image path/to/image.jpg \
        --classes "car" "pedestrian" "bicycle" \
        --confidence 0.3

    # Fast mode (batched + compiled + fp16 + early-exit):
    python demo_multiclass.py --image path/to/image.jpg \
        --classes "car" "pedestrian" "bicycle" --fast

    # Single-pass mode (fastest — one pass with cosine class assignment):
    python demo_multiclass.py --image path/to/image.jpg \
        --classes "car" "pedestrian" "bicycle" --single-pass

    # Benchmark mode (compares all five approaches):
    python demo_multiclass.py --benchmark --classes "car" "pedestrian" "bicycle"
"""

import argparse
import os
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from sam3.model_builder import build_sam3_image_model, load_pruned_config, build_pruned_sam3_image_model
from sam3.model.sam3_multiclass import Sam3MultiClassPredictor
from sam3.model.sam3_multiclass_fast import Sam3MultiClassPredictorFast
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.efficient_backbone import build_efficientsam3_model

# Distinct colours per class (RGB).  Cycles if more classes than colours.
CLASS_COLOURS = [
    (230,  25,  75),  # red
    ( 60, 180,  75),  # green
    (  0, 130, 200),  # blue
    (255, 225,  25),  # yellow
    (245, 130,  48),  # orange
    (145,  30, 180),  # purple
    ( 70, 240, 240),  # cyan
    (240,  50, 230),  # magenta
    (210, 245,  60),  # lime
    (250, 190, 212),  # pink
    (  0, 128, 128),  # teal
    (220, 190, 255),  # lavender
]


def annotate_image(
    image: Image.Image,
    results: Dict,
    class_names: List[str],
    mask_alpha: float = 0.45,
    box_width: int = 3,
    font_size: int = 0,
) -> Image.Image:
    """Draw masks, boxes and labels on a copy of the image.

    Args:
        image: Original PIL image.
        results: Dict from predictor.predict() with boxes, masks, scores, etc.
        class_names: Ordered list of all class names (for colour assignment).
        mask_alpha: Opacity for the mask overlay (0 = transparent, 1 = opaque).
        box_width: Bounding-box line width in pixels.
        font_size: Label font size.  0 = auto-scale from image height.

    Returns:
        Annotated PIL image.
    """
    img = image.convert("RGB").copy()
    w, h = img.size

    if font_size <= 0:
        font_size = max(12, int(h / 60))

    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except (OSError, IOError):
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
        except (OSError, IOError):
            font = ImageFont.load_default()

    n_colours = len(CLASS_COLOURS)
    # Map class name → stable colour index
    class_to_colour = {name: CLASS_COLOURS[i % n_colours] for i, name in enumerate(class_names)}

    num_dets = len(results["scores"])

    if results["masks"] is not None and mask_alpha > 0:
        overlay = np.array(img, dtype=np.float32)       # (H, W, 3)
        mask_layer = np.zeros_like(overlay)
        mask_weight = np.zeros((h, w, 1), dtype=np.float32)

        for i in range(num_dets):
            cls_name = results["class_names"][i]
            colour = class_to_colour.get(cls_name, CLASS_COLOURS[0])
            mask = results["masks"][i].cpu().numpy().astype(bool)

            c = np.array(colour, dtype=np.float32)
            mask_layer[mask] += c
            mask_weight[mask] += 1.0

        valid = mask_weight[..., 0] > 0
        mask_layer[valid] /= mask_weight[valid]
        blended = overlay.copy()
        blended[valid] = overlay[valid] * (1 - mask_alpha) + mask_layer[valid] * mask_alpha
        img = Image.fromarray(blended.clip(0, 255).astype(np.uint8))

    draw = ImageDraw.Draw(img)

    for i in range(num_dets):
        cls_name = results["class_names"][i]
        score = results["scores"][i].item()
        box = results["boxes"][i].cpu().tolist()
        colour = class_to_colour.get(cls_name, CLASS_COLOURS[0])

        x1, y1, x2, y2 = box
        draw.rectangle([x1, y1, x2, y2], outline=colour, width=box_width)

        label = f"{cls_name} {score:.2f}"
        bbox = draw.textbbox((0, 0), label, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        # label background
        lx, ly = x1, max(y1 - th - 4, 0)
        draw.rectangle([lx, ly, lx + tw + 6, ly + th + 4], fill=colour)
        draw.text((lx + 3, ly + 1), label, fill=(255, 255, 255), font=font)

    return img


def run_multiclass_inference(
    image_path: str,
    class_names: List[str],
    confidence_threshold: float = 0.3,
    nms_threshold: float = 0.7,
    device: str = "cuda",
    checkpoint_path: str = None,
    fast: bool = False,
    compile_mode: str = None,
    shared_encoder: bool = False,
    generic_prompt: str = "object",
    single_pass: bool = False,
    class_method: str = "cosine",
    prototype_path: str = None,
    detection_only: bool = False,
    output_path: str = None,
    warmup: int = 0,
    trt_engine_path: str = None,
    trt_enc_dec_engine_path: str = None,
    trt_max_classes: int = 16,
    imgsz: int = 1008,
    text_cache: str = None,
    efficient_backbone: str = None,
    efficient_model: str = None,
    skip_blocks: set = None,
    mask_blocks: list = None,
):
    """Run multi-class detection + segmentation on a single image."""

    if imgsz % 14 != 0:
        raise ValueError(
            f"--imgsz must be divisible by 14 (ViT patch size), got {imgsz}"
        )

    # Check if we can use TRT-only mode (no checkpoint needed)
    text_cache_exists = text_cache and os.path.exists(text_cache)
    use_trt_only = (
        text_cache_exists
        and trt_engine_path
        and trt_enc_dec_engine_path
        and detection_only
        and checkpoint_path is None
    )

    if use_trt_only:
        from sam3.model.sam3_multiclass_fast import _TRTModelStub
        print(f"Using TRT-only mode on {device} (no checkpoint — text from cache)")
        model = _TRTModelStub(device=device)
    elif efficient_backbone:
        print(f"Loading EfficientSAM3 ({efficient_backbone} {efficient_model}) on {device} (resolution={imgsz})...")
        model = build_efficientsam3_model(
            backbone_type=efficient_backbone,
            model_name=efficient_model,
            checkpoint_path=checkpoint_path,
            device=device,
            eval_mode=True,
        )
    else:
        skip_msg = f", skip_blocks={sorted(skip_blocks)}" if skip_blocks else ""
        print(f"Loading SAM3 model on {device} (resolution={imgsz}{skip_msg})...")
        pruned_config = load_pruned_config(checkpoint_path) if checkpoint_path else None
        if pruned_config is not None:
            print(f"  Detected pruned checkpoint: {pruned_config}")
            model = build_pruned_sam3_image_model(
                checkpoint_path=checkpoint_path,
                pruning_config=pruned_config,
                device=device,
                eval_mode=True,
                skip_blocks=skip_blocks,
            )
            # Distilled checkpoints don't train the presence prediction head,
            # so disable it to avoid multiplying scores by untrained garbage.
            if model.transformer.decoder.presence_token is not None:
                print("  Disabling untrained presence token for distilled checkpoint")
                model.transformer.decoder.presence_token = None
        else:
            model = build_sam3_image_model(
                device=device,
                checkpoint_path=checkpoint_path,
                eval_mode=True,
                skip_blocks=skip_blocks,
                mask_blocks=mask_blocks,
            )

    # Precompute position encoding buffers for target resolution
    # (must happen before torch.compile warmup for CUDAGraph safety)
    if imgsz != 1008:
        pos_enc = model.backbone.vision_backbone.position_encoding
        pos_enc.precompute_for_resolution(imgsz)

    # Create predictor
    if single_pass:
        print(f"Using SINGLE-PASS predictor (1x encoder+decoder + {class_method} class scoring)")
        predictor = Sam3MultiClassPredictorFast(
            model, device=device,
            resolution=imgsz,
            compile_mode=compile_mode if class_method != "attention" else None,
            use_fp16=True,
            single_pass=True,
            class_method=class_method,
            prototype_path=prototype_path,
            detection_only=detection_only,
            trt_engine_path=trt_engine_path,
        )
    elif fast or use_trt_only:
        mode_parts = ["batched", "fp16"]
        if compile_mode:
            mode_parts.append(f"compile={compile_mode}")
        if trt_engine_path:
            mode_parts.append("trt-backbone")
        if trt_enc_dec_engine_path:
            mode_parts.append("trt-enc-dec")
        if shared_encoder:
            mode_parts.append(f"shared-enc(\"{generic_prompt}\")")
        if detection_only:
            mode_parts.append("detection-only")
        if text_cache:
            mode_parts.append("text-cache")
        print(f"Using FAST predictor ({' + '.join(mode_parts)})")
        predictor = Sam3MultiClassPredictorFast(
            model, device=device,
            resolution=imgsz,
            compile_mode=compile_mode,
            use_fp16=True,
            presence_threshold=0.05,
            shared_encoder=shared_encoder,
            generic_prompt=generic_prompt,
            detection_only=detection_only,
            trt_engine_path=trt_engine_path,
            trt_enc_dec_engine_path=trt_enc_dec_engine_path,
            trt_max_classes=trt_max_classes,
        )
    else:
        print("Using standard predictor (per-class sequential)")
        predictor = Sam3MultiClassPredictor(model, device=device, resolution=imgsz, detection_only=detection_only)

    # Pre-compute class embeddings (done once, reusable across images)
    print(f"Setting {len(class_names)} classes: {class_names}")
    t0 = time.perf_counter()
    if text_cache and isinstance(predictor, Sam3MultiClassPredictorFast):
        predictor.set_classes(class_names, text_cache=text_cache)
    else:
        predictor.set_classes(class_names)
    t_classes = time.perf_counter() - t0
    print(f"  Text encoding took {t_classes*1000:.1f}ms")

    # Load image
    print(f"Processing image: {image_path}")
    image = Image.open(image_path).convert("RGB")
    print(f"  Image size: {image.size[0]}x{image.size[1]}")

    # Warmup pass (CUDA init, torch.compile, kernel caching)
    if warmup > 0:
        print(f"  Running {warmup} warmup pass(es)...")
        for _ in range(warmup):
            _state = predictor.set_image(image)
            predictor.predict(_state, confidence_threshold=confidence_threshold,
                              nms_threshold=nms_threshold)
        if device == "cuda":
            torch.cuda.synchronize()
        print(f"  Warmup done")

    # Timed backbone pass (with CUDA sync for accurate timing)
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    state = predictor.set_image(image)
    if device == "cuda":
        torch.cuda.synchronize()
    t_backbone = time.perf_counter() - t0
    print(f"  Backbone encoding took {t_backbone*1000:.1f}ms")

    # Fine-grained profiling of set_image components
    if device == "cuda" and hasattr(predictor, '_profile_set_image'):
        print("  --- set_image breakdown ---")
        for label, ms in predictor._profile_set_image(image):
            print(f"    {label}: {ms:.1f}ms")

    # Timed prediction pass
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    results = predictor.predict(
        state,
        confidence_threshold=confidence_threshold,
        nms_threshold=nms_threshold,
    )
    if device == "cuda":
        torch.cuda.synchronize()
    t_predict = time.perf_counter() - t0
    print(f"  Prediction took {t_predict*1000:.1f}ms")

    # Print results
    num_dets = len(results["scores"])
    print(f"\nDetected {num_dets} objects:")
    for i in range(num_dets):
        cls_name = results["class_names"][i]
        score = results["scores"][i].item()
        box = results["boxes"][i].tolist()
        suffix = ""
        if results["masks"] is not None:
            suffix = f"  mask_area={results['masks'][i].sum().item()}"
        print(
            f"  [{i}] {cls_name:20s}  score={score:.3f}  "
            f"box=[{box[0]:.0f},{box[1]:.0f},{box[2]:.0f},{box[3]:.0f}]"
            f"{suffix}"
        )

    print(f"\nTotal time: {(t_backbone + t_predict)*1000:.1f}ms "
          f"(backbone={t_backbone*1000:.1f}ms + predict={t_predict*1000:.1f}ms)")

    # Save annotated image
    if output_path is None:
        stem = Path(image_path).stem
        output_path = f"{stem}_annotated.jpg"
    if results["masks"] is not None:
        annotated = annotate_image(image, results, class_names)
    else:
        annotated = annotate_image(image, results, class_names, mask_alpha=0.0)
    annotated.save(output_path, quality=95)
    print(f"\nSaved annotated image to {output_path}")

    return results


def run_benchmark(
    class_names: List[str],
    device: str = "cuda",
    checkpoint_path: str = None,
    compile_mode: str = None,
    generic_prompt: str = "object",
    num_warmup: int = 3,
    num_runs: int = 10,
    efficient_backbone: str = None,
    efficient_model: str = None,
):
    """Benchmark: per-prompt vs sequential vs batched vs shared-encoder vs single-pass."""

    if efficient_backbone:
        print(f"Loading EfficientSAM3 ({efficient_backbone} {efficient_model}) on {device}...")
        model = build_efficientsam3_model(
            backbone_type=efficient_backbone,
            model_name=efficient_model,
            checkpoint_path=checkpoint_path,
            device=device,
            eval_mode=True,
        )
    else:
        print(f"Loading SAM3 model on {device}...")
        pruned_config = load_pruned_config(checkpoint_path) if checkpoint_path else None
        if pruned_config is not None:
            print(f"  Detected pruned checkpoint: {pruned_config}")
            model = build_pruned_sam3_image_model(
                checkpoint_path=checkpoint_path,
                pruning_config=pruned_config,
                device=device,
                eval_mode=True,
            )
            if model.transformer.decoder.presence_token is not None:
                print("  Disabling untrained presence token for distilled checkpoint")
                model.transformer.decoder.presence_token = None
        else:
            model = build_sam3_image_model(
                device=device,
                checkpoint_path=checkpoint_path,
                eval_mode=True,
            )

    N = len(class_names)
    print(f"\nBenchmarking {N} classes: {class_names}")
    print(f"  Warmup runs: {num_warmup}, Timed runs: {num_runs}")

    # Create a dummy image
    dummy_image = torch.randn(3, 1008, 1008, device=device) * 0.5

    def bench(name, setup_fn, run_fn):
        setup_fn()
        for _ in range(num_warmup):
            run_fn()
        if device == "cuda":
            torch.cuda.synchronize()
        times = []
        for _ in range(num_runs):
            t0 = time.perf_counter()
            run_fn()
            if device == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
        avg = sum(times) / len(times) * 1000
        print(f"  {name:45s} {avg:8.1f}ms avg")
        return avg

    # ---- 1. Per-prompt (N passes) ----
    processor = Sam3Processor(model, device=device, confidence_threshold=0.3)

    def pp_run():
        state = processor.set_image(dummy_image)
        for cls in class_names:
            processor.set_text_prompt(cls, state)

    pp_avg = bench(
        f"Per-prompt ({N} passes)",
        lambda: None,
        pp_run,
    )

    # ---- 2. Sequential multi-class (shared backbone) ----
    mc_predictor = Sam3MultiClassPredictor(model, device=device)
    mc_predictor.set_classes(class_names)

    def mc_run():
        state = mc_predictor.set_image(dummy_image)
        mc_predictor.predict(state, confidence_threshold=0.3)

    mc_avg = bench(
        "Multi-class sequential",
        lambda: None,
        mc_run,
    )

    # ---- 3. Fast batched (encoder bs=N, decoder bs=N) ----
    fast_predictor = Sam3MultiClassPredictorFast(
        model, device=device,
        compile_mode=compile_mode,
        use_fp16=True,
        presence_threshold=0.05,
        shared_encoder=False,
    )
    fast_predictor.set_classes(class_names)

    def fast_run():
        state = fast_predictor.set_image(dummy_image)
        fast_predictor.predict(state, confidence_threshold=0.3)

    compile_tag = f" + compile={compile_mode}" if compile_mode else ""
    fast_avg = bench(
        f"Fast batched{compile_tag} + fp16",
        lambda: None,
        fast_run,
    )

    # ---- 4. Shared encoder (encoder bs=1, decoder bs=N) ----
    shared_predictor = Sam3MultiClassPredictorFast(
        model, device=device,
        compile_mode=compile_mode,
        use_fp16=True,
        presence_threshold=0.05,
        shared_encoder=True,
        generic_prompt=generic_prompt,
    )
    shared_predictor.set_classes(class_names)

    def shared_run():
        state = shared_predictor.set_image(dummy_image)
        shared_predictor.predict(state, confidence_threshold=0.3)

    shared_avg = bench(
        f"Shared-enc(\"{generic_prompt}\"){compile_tag} + fp16",
        lambda: None,
        shared_run,
    )

    # ---- 5. Single-pass (encoder bs=1, decoder bs=1, cosine scoring) ----
    single_predictor = Sam3MultiClassPredictorFast(
        model, device=device,
        compile_mode=compile_mode,
        use_fp16=True,
        single_pass=True,
    )
    single_predictor.set_classes(class_names)

    def single_run():
        state = single_predictor.set_image(dummy_image)
        single_predictor.predict(state, confidence_threshold=0.3)

    single_avg = bench(
        f"Single-pass{compile_tag} + fp16 + cosine",
        lambda: None,
        single_run,
    )

    # ---- Results ----
    print(f"\n{'='*65}")
    print(f"BENCHMARK RESULTS ({N} classes)")
    print(f"{'='*65}")
    print(f"  Per-prompt ({N} passes):          {pp_avg:8.1f}ms")
    print(f"  Multi-class sequential:           {mc_avg:8.1f}ms  "
          f"({pp_avg/mc_avg:.2f}x vs per-prompt)")
    print(f"  Fast batched:                     {fast_avg:8.1f}ms  "
          f"({pp_avg/fast_avg:.2f}x vs per-prompt)")
    print(f"  Shared-enc + batched:             {shared_avg:8.1f}ms  "
          f"({pp_avg/shared_avg:.2f}x vs per-prompt, "
          f"{fast_avg/shared_avg:.2f}x vs batched)")
    print(f"  Single-pass + cosine:             {single_avg:8.1f}ms  "
          f"({pp_avg/single_avg:.2f}x vs per-prompt, "
          f"{fast_avg/single_avg:.2f}x vs batched)")
    print(f"{'='*65}")


def main():
    parser = argparse.ArgumentParser(
        description="SAM3 multi-class inference demo"
    )
    parser.add_argument(
        "--image", type=str, default=None,
        help="Path to input image"
    )
    parser.add_argument(
        "--classes", nargs="+", type=str,
        default=["car", "pedestrian", "bicycle"],
        help="Target class names"
    )
    parser.add_argument(
        "--coco", action="store_true",
        help="Use all 80 COCO classes (overrides --classes)"
    )
    parser.add_argument(
        "--confidence", type=float, default=0.3,
        help="Confidence threshold"
    )
    parser.add_argument(
        "--nms", type=float, default=0.7,
        help="NMS IoU threshold"
    )
    parser.add_argument(
        "--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use"
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to model checkpoint (default: download from HF)"
    )
    parser.add_argument(
        "--efficient-backbone", type=str, default=None,
        choices=["efficientvit", "repvit", "tinyvit"],
        help="Use lightweight EfficientSAM3 backbone instead of ViT-H",
    )
    parser.add_argument(
        "--efficient-model", type=str, default=None,
        help="Backbone variant (e.g. b0/b1/b2, m0_9/m1_1/m2_3, 5m/11m/21m)",
    )
    parser.add_argument(
        "--fast", action="store_true",
        help="Use fast predictor (batched + fp16 + early-exit)"
    )
    parser.add_argument(
        "--compile", type=str, default=None,
        choices=["default", "reduce-overhead", "max-autotune"],
        help="torch.compile mode (requires --fast)"
    )
    parser.add_argument(
        "--shared-encoder", action="store_true",
        help="Run encoder once with generic prompt (requires --fast)"
    )
    parser.add_argument(
        "--generic-prompt", type=str, default="object",
        help="Scene-level prompt for shared encoder (e.g. 'urban', 'indoor')"
    )
    parser.add_argument(
        "--single-pass", action="store_true",
        help="True single-pass: 1x encoder+decoder+masks with cosine class scoring"
    )
    parser.add_argument(
        "--class-method", type=str, default="cosine",
        choices=["cosine", "attention", "prototype"],
        help="Class assignment method for single-pass mode (default: cosine)"
    )
    parser.add_argument(
        "--prototype-path", type=str, default=None,
        help="Path to calibrated prototypes .pt file (for --class-method prototype)"
    )
    parser.add_argument(
        "--detection-only", action="store_true",
        help="Skip mask generation — return boxes + scores only (faster)"
    )
    parser.add_argument(
        "--imgsz", type=int, default=1008,
        help="Input image resolution (default: 1008). Must be divisible by 14 (ViT patch size)."
    )
    parser.add_argument(
        "--warmup", type=int, default=0,
        help="Number of warmup passes before timed inference (for accurate benchmarking)"
    )
    parser.add_argument(
        "--trt", type=str, default=None, metavar="ENGINE",
        help="Path to TensorRT engine for backbone (built via sam3.trt.build_engine)"
    )
    parser.add_argument(
        "--trt-enc-dec", type=str, default=None, metavar="ENGINE",
        help="Path to TensorRT engine for encoder+decoder+scoring "
             "(built via sam3.trt.build_engine --type enc-dec)"
    )
    parser.add_argument(
        "--trt-max-classes", type=int, default=4,
        help="Max classes for enc-dec TRT engine (must match --max-classes during export)"
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Output path for annotated image (default: <input>_annotated.jpg)"
    )
    parser.add_argument(
        "--text-cache", type=str, default=None, metavar="PATH",
        help="Path to cached text embeddings (.pt). If the file exists "
             "with matching classes, skips text encoder (and --checkpoint "
             "is not needed with full TRT). Otherwise, computes and saves.",
    )
    parser.add_argument(
        "--skip-blocks", type=str, default=None, metavar="INDICES",
        help="Comma-separated ViT block indices to skip for block pruning "
             "(e.g. '1,3,9,11,17,19,25,27' skips 8 window blocks). "
             "Cannot skip global attention blocks [7,15,23,31]."
    )
    parser.add_argument(
        "--mask-blocks", type=str, default=None, metavar="SPEC",
        help="Fine-grained sub-block pruning from BlockPruner search. "
             "Comma-separated 'idx:type' pairs (e.g. '0:attn,1:mlp,2:attn'). "
             "Masks attention or MLP sub-blocks independently. "
             "Run scripts/block_pruner_search.py to find optimal pruning order."
    )
    parser.add_argument(
        "--benchmark", action="store_true",
        help="Run benchmark comparing all five approaches"
    )
    args = parser.parse_args()

    if args.coco:
        from sam3.coco_classes import COCO_CLASSES
        args.classes = COCO_CLASSES

    if args.efficient_backbone and not args.efficient_model:
        parser.error("--efficient-backbone requires --efficient-model")

    # Parse skip_blocks
    skip_blocks = None
    if args.skip_blocks:
        skip_blocks = set(int(x.strip()) for x in args.skip_blocks.split(","))

    # Parse mask_blocks (fine-grained sub-block pruning)
    mask_blocks = None
    if args.mask_blocks:
        mask_blocks = [s.strip() for s in args.mask_blocks.split(",")]

    if args.benchmark:
        run_benchmark(
            class_names=args.classes,
            device=args.device,
            checkpoint_path=args.checkpoint,
            compile_mode=args.compile,
            generic_prompt=args.generic_prompt,
            efficient_backbone=args.efficient_backbone,
            efficient_model=args.efficient_model,
        )
    elif args.image:
        run_multiclass_inference(
            image_path=args.image,
            class_names=args.classes,
            confidence_threshold=args.confidence,
            nms_threshold=args.nms,
            device=args.device,
            checkpoint_path=args.checkpoint,
            fast=args.fast,
            compile_mode=args.compile,
            shared_encoder=args.shared_encoder,
            generic_prompt=args.generic_prompt,
            single_pass=args.single_pass,
            class_method=args.class_method,
            prototype_path=args.prototype_path,
            detection_only=args.detection_only,
            output_path=args.output,
            warmup=args.warmup,
            trt_engine_path=args.trt,
            trt_enc_dec_engine_path=args.trt_enc_dec,
            trt_max_classes=args.trt_max_classes,
            imgsz=args.imgsz,
            text_cache=args.text_cache,
            efficient_backbone=args.efficient_backbone,
            efficient_model=args.efficient_model,
            skip_blocks=skip_blocks,
            mask_blocks=mask_blocks,
        )
    else:
        parser.print_help()
        print("\nProvide --image for inference or --benchmark for timing comparison.")


if __name__ == "__main__":
    main()
