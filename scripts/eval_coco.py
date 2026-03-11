#!/usr/bin/env python3
"""Standard COCO val2017 evaluation with pycocotools.

Evaluates RT-SAM3 detection quality on the full COCO val2017 set (5000 images)
using the official COCO evaluation protocol: mAP@[.5:.95], mAP@50, mAP@75,
and per-size breakdowns (small/medium/large).

Uses the optimized TRT pipeline for fast inference across all 80 classes.
Each config can specify its own backbone TRT engine, enc-dec TRT engine,
resolution, and pruning mask — enabling multi-resolution comparison in a
single run.

Requirements:
    pip install pycocotools    # (pycocotools-windows on Windows if needed)

Config format:
    "name=key:val;key:val;..."

    Keys:
        trt:<path>      Backbone TRT engine (omit for PyTorch backbone)
        encdec:<path>   Enc-dec TRT engine (overrides --trt-enc-dec)
        mask:<spec>     Sub-block pruning mask (e.g. 25:attn,28:mlp,...)
        imgsz:<int>     Resolution (overrides --imgsz)

Usage:
    PYTHONIOENCODING=utf-8 python scripts/eval_coco.py \
        --images-dir D:/val2017 \
        --ann-file D:/coco2017labels/coco/annotations/instances_val2017.json \
        --checkpoint sam3.pt \
        --configs \
            "full_1008=trt:hf_backbone_1008_fp16.engine;encdec:enc_dec_coco_fp16_80.engine;imgsz:1008" \
            "prune16_1008=trt:hf_backbone_1008_masked_fp16.engine;encdec:enc_dec_coco_fp16_80.engine;mask:25:attn,28:mlp,27:attn,22:attn,28:attn,30:mlp,20:attn,27:mlp,26:attn,22:mlp,24:attn,18:attn,20:mlp,21:attn,25:mlp,18:mlp;imgsz:1008" \
            "full_644=trt:hf_backbone_644_fp16.engine;encdec:enc_dec_644_coco80_fp16.engine;imgsz:644" \
            "prune16_644=trt:hf_backbone_644_masked_fp16.engine;encdec:enc_dec_644_coco80_fp16.engine;mask:25:attn,28:mlp,27:attn,22:attn,28:attn,30:mlp,20:attn,27:mlp,26:attn,22:mlp,24:attn,18:attn,20:mlp,21:attn,25:mlp,18:mlp;imgsz:644"
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sam3.coco_classes import COCO_CLASSES
from sam3.model_builder import build_sam3_image_model
from sam3.efficient_backbone import build_efficientsam3_model
from sam3.model.sam3_multiclass_fast import Sam3MultiClassPredictorFast

def _build_student_model(checkpoint, backbone_config, adapter_checkpoint, device, lora_rank=0):
    """Build student model with distilled adapter weights."""
    from sam3.distillation.sam3_student import build_sam3_student_model
    model = build_sam3_student_model(
        backbone_config=backbone_config,
        teacher_checkpoint=checkpoint,
        load_from_HF=checkpoint is None,
        device=device,
        freeze_teacher=True,
        pretrained_student=True,
    )
    if lora_rank > 0:
        from sam3.distillation.lora import apply_lora
        n = apply_lora(model.backbone.student_backbone.backbone, rank=lora_rank, alpha=lora_rank)
        print(f"  Applied LoRA (rank={lora_rank}) to {n} backbone layers")
    if adapter_checkpoint:
        print(f"  Loading adapter weights from {adapter_checkpoint}")
        ckpt = torch.load(adapter_checkpoint, map_location=device)
        model.backbone.student_backbone.load_state_dict(
            ckpt["student_state_dict"]
        )
    model.eval()
    return model

try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
except ImportError:
    print("ERROR: pycocotools is required.")
    print("  Install with:  pip install pycocotools")
    print("  On Windows:    pip install pycocotools-windows")
    sys.exit(1)


MASK_16 = (
    "25:attn,28:mlp,27:attn,22:attn,28:attn,30:mlp,20:attn,27:mlp,"
    "26:attn,22:mlp,24:attn,18:attn,20:mlp,21:attn,25:mlp,18:mlp"
)


def parse_config(config_str: str) -> Dict:
    """Parse config string into a dict.

    Formats:
        "name"                                  — PyTorch backbone, defaults
        "name=trt:bb.engine;encdec:ed.engine"   — TRT backbone + enc-dec
        "name=trt:bb.engine;mask:25:attn,...;imgsz:644"
    """
    result = {
        "name": None, "trt": None, "encdec": None,
        "mask_blocks": None, "imgsz": None,
    }

    if "=" in config_str:
        name, rest = config_str.split("=", 1)
        result["name"] = name
        for part in rest.split(";"):
            if part.startswith("trt:"):
                result["trt"] = part[4:]
            elif part.startswith("encdec:"):
                result["encdec"] = part[7:]
            elif part.startswith("mask:"):
                result["mask_blocks"] = part[5:]
            elif part.startswith("imgsz:"):
                result["imgsz"] = int(part[6:])
            elif part == "pytorch":
                pass
    else:
        result["name"] = config_str

    return result


def apply_mask_config(trunk, mask_blocks_str: Optional[str]) -> int:
    """Toggle mask_attn/mask_mlp on trunk blocks. Returns count masked."""
    for blk in trunk.blocks:
        blk.mask_attn = False
        blk.mask_mlp = False

    if not mask_blocks_str:
        return 0

    n = 0
    for entry in mask_blocks_str.split(","):
        entry = entry.strip()
        if not entry:
            continue
        idx_str, sub_type = entry.split(":")
        blk = trunk.blocks[int(idx_str)]
        if sub_type == "attn":
            blk.mask_attn = True
        elif sub_type == "mlp":
            blk.mask_mlp = True
        n += 1
    return n


def evaluate_config(
    config: Dict,
    model,
    coco_gt: COCO,
    image_dir: str,
    img_ids: List[int],
    class_names: List[str],
    idx_to_cat_id: List[int],
    default_imgsz: int,
    default_encdec: Optional[str],
    trt_max_classes: int,
    confidence: float,
    nms_threshold: float,
    device: str,
    text_cache: Optional[str],
    max_dets_per_img: int,
    compile_mode: Optional[str] = None,
) -> Dict:
    """Run inference + COCO evaluation for one config."""
    name = config["name"]
    trt_engine = config["trt"]
    encdec = config["encdec"] or default_encdec
    mask_str = config["mask_blocks"]
    imgsz = config["imgsz"] or default_imgsz

    # Apply pruning mask to backbone trunk (skip for student/efficient models)
    trunk = getattr(getattr(model.backbone, 'vision_backbone', None), 'trunk', None)
    if trunk is not None and hasattr(trunk, 'blocks'):
        n_masked = apply_mask_config(trunk, mask_str)
    else:
        n_masked = 0

    is_trt = trt_engine is not None

    print(f"\n{'='*70}")
    print(f"Config: {name}")
    print(f"  Backbone:  {'TRT ' + os.path.basename(trt_engine) if is_trt else 'PyTorch'}")
    print(f"  Enc-dec:   {'TRT ' + os.path.basename(encdec) if encdec else 'PyTorch'}")
    print(f"  Compile:   {compile_mode or 'disabled'}")
    print(f"  Resolution: {imgsz}")
    print(f"  Masked:    {n_masked} sub-blocks")
    print(f"{'='*70}")

    # Create predictor
    predictor = Sam3MultiClassPredictorFast(
        model,
        device=device,
        resolution=imgsz,
        compile_mode=compile_mode,
        use_fp16=True,
        detection_only=True,
        presence_threshold=0.0,
        trt_engine_path=trt_engine,
        trt_enc_dec_engine_path=encdec,
        trt_max_classes=trt_max_classes,
    )
    predictor.set_classes(class_names, text_cache=text_cache)

    # Warmup (3 passes on first image)
    print("  Warming up ...")
    warmup_info = coco_gt.loadImgs(img_ids[0])[0]
    warmup_path = os.path.join(image_dir, warmup_info["file_name"])
    warmup_img = Image.open(warmup_path).convert("RGB")
    for _ in range(3):
        state = predictor.set_image(warmup_img)
        predictor.predict(
            state, confidence_threshold=confidence,
            nms_threshold=nms_threshold,
        )

    # Run inference on all images
    n_images = len(img_ids)
    coco_results = []
    total_ms = 0.0
    total_dets = 0

    pbar = tqdm(img_ids, desc=f"  {name}", unit="img",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}, {postfix}]")
    for img_id in pbar:
        img_info = coco_gt.loadImgs(img_id)[0]
        img_path = os.path.join(image_dir, img_info["file_name"])

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            tqdm.write(f"    SKIP {img_path}: {e}")
            continue

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        state = predictor.set_image(image)
        results = predictor.predict(
            state,
            confidence_threshold=confidence,
            nms_threshold=nms_threshold,
            per_class_nms=True,
        )
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000
        total_ms += elapsed_ms

        # Extract predictions
        if results["boxes"] is not None and len(results["boxes"]) > 0:
            boxes = results["boxes"].cpu()
            scores = results["scores"].cpu()
            class_ids = results["class_ids"].cpu()

            # Keep top-k by score
            if len(boxes) > max_dets_per_img:
                topk = scores.argsort(descending=True)[:max_dets_per_img]
                boxes = boxes[topk]
                scores = scores[topk]
                class_ids = class_ids[topk]

            for j in range(len(boxes)):
                x1, y1, x2, y2 = boxes[j].tolist()
                cid = class_ids[j].item()
                cat_id = idx_to_cat_id[cid]
                coco_results.append({
                    "image_id": img_id,
                    "category_id": cat_id,
                    "bbox": [
                        round(x1, 2),
                        round(y1, 2),
                        round(x2 - x1, 2),  # width
                        round(y2 - y1, 2),  # height
                    ],
                    "score": round(scores[j].item(), 5),
                })
                total_dets += 1

        n_done = pbar.n
        if n_done > 0:
            pbar.set_postfix_str(f"{total_ms/n_done:.0f}ms/img, {total_dets}dets")

    pbar.close()
    avg_ms = total_ms / max(n_images, 1)
    print(f"  Done: {total_dets} detections, {avg_ms:.0f}ms/img")

    # COCO evaluation
    if not coco_results:
        print("  WARNING: No detections produced!")
        stats = {
            "mAP": 0, "mAP50": 0, "mAP75": 0,
            "mAP_small": 0, "mAP_medium": 0, "mAP_large": 0,
        }
    else:
        results_file = f"_coco_results_{name}.json"
        with open(results_file, "w") as f:
            json.dump(coco_results, f)

        print(f"\n  Running pycocotools evaluation ...")
        coco_dt = coco_gt.loadRes(results_file)
        coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
        coco_eval.params.imgIds = img_ids  # Only evaluate on our subset
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        stats = {
            "mAP": coco_eval.stats[0],        # AP @ IoU=0.50:0.95
            "mAP50": coco_eval.stats[1],       # AP @ IoU=0.50
            "mAP75": coco_eval.stats[2],       # AP @ IoU=0.75
            "mAP_small": coco_eval.stats[3],   # AP small
            "mAP_medium": coco_eval.stats[4],  # AP medium
            "mAP_large": coco_eval.stats[5],   # AP large
            "AR1": coco_eval.stats[6],         # AR maxDets=1
            "AR10": coco_eval.stats[7],        # AR maxDets=10
            "AR100": coco_eval.stats[8],       # AR maxDets=100
        }

        os.remove(results_file)

    stats["name"] = name
    stats["imgsz"] = imgsz
    stats["n_masked"] = n_masked
    stats["total_dets"] = total_dets
    stats["avg_ms"] = avg_ms
    stats["n_images"] = n_images

    # Clean up TRT resources
    del predictor
    torch.cuda.empty_cache()

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Standard COCO val2017 evaluation with pycocotools",
    )
    parser.add_argument(
        "--images-dir", required=True,
        help="Directory containing val2017 images",
    )
    parser.add_argument(
        "--ann-file", required=True,
        help="Path to instances_val2017.json",
    )
    parser.add_argument("--checkpoint", default="sam3.pt")
    parser.add_argument(
        "--max-images", type=int, default=5000,
        help="Max images to evaluate (default: 5000 = full val set)",
    )
    parser.add_argument(
        "--imgsz", type=int, default=1008,
        help="Default resolution (configs can override)",
    )
    parser.add_argument(
        "--confidence", type=float, default=0.01,
        help="Confidence threshold (low for COCO eval, default: 0.01)",
    )
    parser.add_argument(
        "--nms", type=float, default=0.7,
        help="NMS IoU threshold (per-class)",
    )
    parser.add_argument(
        "--trt-enc-dec", type=str, default=None,
        help="Default TRT enc-dec engine (configs can override with encdec:)",
    )
    parser.add_argument("--trt-max-classes", type=int, default=80)
    parser.add_argument(
        "--text-cache", type=str, default=None,
        help="Text embedding cache file (.pt)",
    )
    parser.add_argument(
        "--configs", nargs="+", required=True,
        help='Config specs (see module docstring for format)',
    )
    parser.add_argument(
        "--max-dets", type=int, default=300,
        help="Max detections per image to submit (default: 300)",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--student-backbone", type=str, default=None,
        choices=["efficientvit_l1", "efficientvit_l2", "repvit_m2_3", "tiny_vit_21m"],
        help="Use distilled student backbone instead of ViT-H",
    )
    parser.add_argument(
        "--adapter-checkpoint", type=str, default=None,
        help="Path to adapter weights (.pt) from distillation",
    )
    parser.add_argument(
        "--lora-rank", type=int, default=0,
        help="LoRA rank used during training (must match checkpoint)",
    )
    parser.add_argument(
        "--compile-mode", type=str, default=None,
        choices=["default", "reduce-overhead", "max-autotune"],
        help="torch.compile mode for encoder/decoder/backbone (default: None = disabled)",
    )
    parser.add_argument(
        "--efficient-backbone", type=str, default=None,
        choices=["efficientvit", "repvit", "tinyvit"],
        help="Use EfficientSAM3 lightweight backbone",
    )
    parser.add_argument(
        "--efficient-model", type=str, default=None,
        help="Backbone variant (e.g. b0/b1/b2, m0_9/m1_1/m2_3, 5m/11m/21m)",
    )
    args = parser.parse_args()

    # Load COCO annotations
    print(f"Loading COCO annotations from {args.ann_file} ...")
    coco_gt = COCO(args.ann_file)

    # Build class index → COCO category ID mapping
    cats = coco_gt.loadCats(coco_gt.getCatIds())
    cat_name_to_id = {c["name"]: c["id"] for c in cats}
    idx_to_cat_id = []
    for name in COCO_CLASSES:
        if name not in cat_name_to_id:
            print(f"  WARNING: '{name}' not found in annotations, using -1")
            idx_to_cat_id.append(-1)
        else:
            idx_to_cat_id.append(cat_name_to_id[name])
    print(f"  Mapped {sum(1 for x in idx_to_cat_id if x > 0)}/80 classes")

    # Select images
    all_img_ids = sorted(coco_gt.getImgIds())
    img_ids = all_img_ids[:args.max_images]
    n_images = len(img_ids)

    # Parse configs
    configs = [parse_config(c) for c in args.configs]

    print(f"\nImages:     {n_images} / {len(all_img_ids)}")
    print(f"Classes:    80 (COCO)")
    print(f"Default res: {args.imgsz}")
    print(f"Confidence: {args.confidence}")
    print(f"NMS:        {args.nms}")
    print(f"Max dets:   {args.max_dets}")
    print(f"Configs:    {[c['name'] for c in configs]}")
    if args.trt_enc_dec:
        print(f"Default enc-dec: {args.trt_enc_dec}")

    # Load model
    if args.efficient_backbone:
        if not args.efficient_model:
            print("ERROR: --efficient-backbone requires --efficient-model")
            sys.exit(1)
        print(f"\nLoading EfficientSAM3 ({args.efficient_backbone} {args.efficient_model}) ...")
        model = build_efficientsam3_model(
            backbone_type=args.efficient_backbone,
            model_name=args.efficient_model,
            checkpoint_path=args.checkpoint,
            device=args.device,
            eval_mode=True,
        )
    elif args.student_backbone:
        print(f"\nBuilding student model ({args.student_backbone}) ...")
        model = _build_student_model(
            checkpoint=args.checkpoint,
            backbone_config=args.student_backbone,
            adapter_checkpoint=args.adapter_checkpoint,
            device=args.device,
            lora_rank=args.lora_rank,
        )
    else:
        print(f"\nLoading SAM3 from {args.checkpoint} ...")
        model = build_sam3_image_model(
            checkpoint_path=args.checkpoint,
            device=args.device,
            eval_mode=True,
        )

    # Evaluate each config
    all_results = []
    for config in configs:
        result = evaluate_config(
            config=config,
            model=model,
            coco_gt=coco_gt,
            image_dir=args.images_dir,
            img_ids=img_ids,
            class_names=list(COCO_CLASSES),
            idx_to_cat_id=idx_to_cat_id,
            default_imgsz=args.imgsz,
            default_encdec=args.trt_enc_dec,
            trt_max_classes=args.trt_max_classes,
            confidence=args.confidence,
            nms_threshold=args.nms,
            device=args.device,
            text_cache=args.text_cache,
            max_dets_per_img=args.max_dets,
            compile_mode=args.compile_mode,
        )
        all_results.append(result)

    # Summary table
    W = 105
    print(f"\n\n{'='*W}")
    print(f"COCO val2017 EVALUATION  ({n_images} images, 80 classes, pycocotools)")
    print(f"{'='*W}")
    header = (
        f"  {'Config':<18s}  {'Res':>4s}  {'Pruned':>6s}  "
        f"{'mAP':>6s}  {'mAP50':>6s}  {'mAP75':>6s}  "
        f"{'AP_S':>5s}  {'AP_M':>5s}  {'AP_L':>5s}  "
        f"{'Dets':>7s}  {'ms/img':>7s}"
    )
    print(header)
    print(f"  {'-'*(W-2)}")

    for r in all_results:
        print(
            f"  {r['name']:<18s}  {r['imgsz']:>4d}  {r['n_masked']:>6d}  "
            f"{r['mAP']:>6.3f}  {r['mAP50']:>6.3f}  {r['mAP75']:>6.3f}  "
            f"{r.get('mAP_small', 0):>5.3f}  {r.get('mAP_medium', 0):>5.3f}  "
            f"{r.get('mAP_large', 0):>5.3f}  "
            f"{r['total_dets']:>7d}  {r['avg_ms']:>6.0f}ms"
        )

    print(f"{'='*W}")
    print(f"  conf={args.confidence}, NMS={args.nms}, max_dets={args.max_dets}")

    # Relative comparison
    if len(all_results) > 1:
        baseline = all_results[0]
        print(f"\n{'='*70}")
        print(f"RELATIVE (vs {baseline['name']}):")
        print(f"{'='*70}")
        for r in all_results:
            rel_map = r["mAP"] / baseline["mAP"] * 100 if baseline["mAP"] > 0 else 0
            rel_50 = r["mAP50"] / baseline["mAP50"] * 100 if baseline["mAP50"] > 0 else 0
            speedup = baseline["avg_ms"] / r["avg_ms"] if r["avg_ms"] > 0 else 0
            print(
                f"  {r['name']:<18s}  "
                f"mAP={r['mAP']:.3f} ({rel_map:5.1f}%)  "
                f"mAP50={r['mAP50']:.3f} ({rel_50:5.1f}%)  "
                f"speed={speedup:.2f}x"
            )
        print(f"{'='*70}")

    # Save results JSON
    out_path = "coco_eval_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    return all_results


if __name__ == "__main__":
    main()
