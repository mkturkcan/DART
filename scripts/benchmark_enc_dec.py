#!/usr/bin/env python3
"""Benchmark encoder+decoder: TensorRT FP16 vs PyTorch (eager / compiled / FP16).

Compares latency and quality (cosine similarity, score/box diff) for all
configurations of the encoder+decoder+scoring pipeline.

Configurations tested:
  1. PyTorch FP32 eager        (reference)
  2. PyTorch FP16 autocast     (eager)
  3. torch.compile default     (FP16 autocast)
  4. TensorRT FP16             (existing enc_dec engine)
  5. TensorRT FP16 pure        (no mixed precision)

Usage:
    python scripts/benchmark_enc_dec.py --checkpoint sam3.pt --image x.jpg

    # With a specific engine (if not using default path):
    python scripts/benchmark_enc_dec.py --checkpoint sam3.pt --image x.jpg \
        --engine enc_dec_fp16.engine

    # Skip TRT tests (PyTorch-only):
    python scripts/benchmark_enc_dec.py --checkpoint sam3.pt --image x.jpg --no-trt

    # Rebuild ONNX + engines from scratch:
    python scripts/benchmark_enc_dec.py --checkpoint sam3.pt --image x.jpg --rebuild
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def cosine_similarity(a, b):
    a_flat = a.float().flatten()
    b_flat = b.float().flatten()
    return torch.nn.functional.cosine_similarity(
        a_flat.unsqueeze(0), b_flat.unsqueeze(0)
    ).item()


def benchmark_fn(fn, n_warmup=10, n_runs=50, label=""):
    """Benchmark a callable, return (last_output, stats_dict)."""
    print(f"\n--- {label} ---")
    print(f"  Warming up ({n_warmup} runs)...")
    with torch.inference_mode():
        for _ in range(n_warmup):
            out = fn()
    torch.cuda.synchronize()

    print(f"  Timing ({n_runs} runs)...")
    times = []
    with torch.inference_mode():
        for _ in range(n_runs):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = fn()
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)

    avg = np.mean(times)
    mn = np.min(times)
    p50 = np.percentile(times, 50)
    p95 = np.percentile(times, 95)
    print(f"  Avg: {avg:.2f}ms  Min: {mn:.2f}ms  P50: {p50:.2f}ms  P95: {p95:.2f}ms")
    return out, {"label": label, "avg_ms": avg, "min_ms": mn, "p50": p50, "p95": p95}


def print_quality(label, scores, boxes, ref_scores, ref_boxes):
    """Print quality metrics vs FP32 reference."""
    score_cos = cosine_similarity(scores, ref_scores)
    box_cos = cosine_similarity(boxes, ref_boxes)
    score_diff = (scores.float() - ref_scores.float()).abs().max().item()
    box_diff = (boxes.float() - ref_boxes.float()).abs().max().item()
    print(f"  Quality vs FP32 ref:")
    print(f"    scores: cos={score_cos:.6f}  max_diff={score_diff:.4e}")
    print(f"    boxes:  cos={box_cos:.6f}  max_diff={box_diff:.4e}")


def build_enc_dec_engine(
    onnx_path, engine_path, mixed_precision=None, opt_level=3
):
    """Build a TRT engine from enc_dec ONNX."""
    from sam3.trt.build_engine import build_engine

    build_engine(
        onnx_path=onnx_path,
        output_path=engine_path,
        engine_type="enc-dec",
        fp16=True,
        mixed_precision=mixed_precision if mixed_precision else "none",
        workspace_gb=4.0,
        opt_level=opt_level,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark enc-dec: TRT vs PyTorch"
    )
    parser.add_argument("--checkpoint", default="sam3.pt")
    parser.add_argument("--image", default="x.jpg")
    parser.add_argument(
        "--classes", nargs="+",
        default=["person", "car", "bicycle", "dog"],
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument(
        "--engine", default=None,
        help="Path to pre-built TRT FP16 engine (default: auto-build)",
    )
    parser.add_argument(
        "--onnx", default="enc_dec.onnx",
        help="Path to enc_dec ONNX (for building engines)",
    )
    parser.add_argument("--no-trt", action="store_true", help="Skip TRT tests")
    parser.add_argument(
        "--rebuild", action="store_true",
        help="Force re-export ONNX and rebuild engines",
    )
    parser.add_argument(
        "--max-classes", type=int, default=4,
        help="Max classes for TRT engine batch dim",
    )
    args = parser.parse_args()

    device = "cuda"
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Classes: {args.classes}")
    num_classes = len(args.classes)

    # ---------------------------------------------------------------
    # Load model and prepare inputs
    # ---------------------------------------------------------------
    from PIL import Image
    from torchvision.transforms import v2

    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_multiclass_fast import Sam3MultiClassPredictorFast

    print("\nLoading model...")
    model = build_sam3_image_model(
        device=device, checkpoint_path=args.checkpoint, eval_mode=True,
    )
    predictor = Sam3MultiClassPredictorFast(
        model, device=device, resolution=1008,
        use_fp16=False, detection_only=True,
    )

    # Set classes (pre-computes text embeddings)
    predictor.set_classes(args.classes)

    # Run backbone on real image
    image = Image.open(args.image).convert("RGB")
    state = predictor.set_image(image)
    backbone_out = state["backbone_out"]

    # Extract image features (same as predict() does)
    img_ids = torch.tensor([0], device=device, dtype=torch.long)
    backbone_out, img_feats, img_pos_embeds, vis_feat_sizes = (
        model._get_img_feats(backbone_out, img_ids)
    )

    # Text features
    text_feats = predictor._batched_text   # (seq, N, d) seq-first
    text_mask = predictor._batched_mask    # (N, seq)
    N = num_classes

    print(f"\nEncoder-decoder inputs:")
    print(f"  img_feats[-1]: {img_feats[-1].shape}")
    print(f"  img_pos[-1]:   {img_pos_embeds[-1].shape}")
    print(f"  text_feats:    {text_feats.shape}")
    print(f"  text_mask:     {text_mask.shape}")
    print(f"  num_classes:   {N}")

    # ---------------------------------------------------------------
    # Prepare PyTorch encoder+decoder forward functions
    # ---------------------------------------------------------------
    encoder = model.transformer.encoder
    decoder = model.transformer.decoder
    scoring = model.dot_prod_scoring
    from sam3.model.model_misc import inverse_sigmoid

    def pytorch_enc_dec():
        """Full encoder+decoder+scoring+box PyTorch forward at bs=N."""
        # Expand image features to N classes
        batched_img_feats = [f.expand(-1, N, -1).contiguous() for f in img_feats]
        batched_img_pos = [p.expand(-1, N, -1).contiguous() for p in img_pos_embeds]

        prompt = text_feats
        prompt_mask_bool = text_mask
        prompt_pos = torch.zeros_like(prompt)

        memory = encoder(
            src=batched_img_feats,
            src_key_padding_mask=None,
            src_pos=batched_img_pos,
            prompt=prompt,
            prompt_pos=prompt_pos,
            prompt_key_padding_mask=prompt_mask_bool,
            feat_sizes=vis_feat_sizes,
        )

        query_embed = decoder.query_embed.weight
        tgt = query_embed.unsqueeze(1).expand(-1, N, -1)

        hs, ref_boxes, _, _ = decoder(
            tgt=tgt,
            memory=memory["memory"],
            memory_key_padding_mask=memory["padding_mask"],
            pos=memory["pos_embed"],
            reference_boxes=None,
            level_start_index=memory["level_start_index"],
            spatial_shapes=memory["spatial_shapes"],
            valid_ratios=memory["valid_ratios"],
            tgt_mask=None,
            memory_text=prompt,
            text_attention_mask=prompt_mask_bool,
            apply_dac=False,
        )

        hs = hs.transpose(1, 2)
        ref_boxes = ref_boxes.transpose(1, 2)

        hs_last = hs[-1:]
        ref_last = ref_boxes[-1]

        scores = scoring(hs_last, prompt, prompt_mask_bool)[0]
        box_offsets = decoder.bbox_embed(hs_last)[0]
        boxes = (inverse_sigmoid(ref_last) + box_offsets).sigmoid()

        return scores, boxes

    # ---------------------------------------------------------------
    # 1. PyTorch FP32 eager (reference)
    # ---------------------------------------------------------------
    results = []

    ref_out, ref_stats = benchmark_fn(
        pytorch_enc_dec, args.warmup, args.runs,
        "PyTorch FP32 eager"
    )
    ref_scores, ref_boxes = ref_out
    results.append(ref_stats)
    print(f"  Output shapes: scores={ref_scores.shape}, boxes={ref_boxes.shape}")

    # ---------------------------------------------------------------
    # 2. PyTorch FP16 autocast eager
    # ---------------------------------------------------------------
    def pytorch_enc_dec_fp16():
        with torch.autocast("cuda", dtype=torch.float16):
            return pytorch_enc_dec()

    out, stats = benchmark_fn(
        pytorch_enc_dec_fp16, args.warmup, args.runs,
        "PyTorch FP16 autocast (eager)"
    )
    print_quality("FP16 eager", out[0], out[1], ref_scores, ref_boxes)
    results.append(stats)

    # ---------------------------------------------------------------
    # 3. torch.compile default + FP16
    # ---------------------------------------------------------------
    try:
        print("\nCompiling encoder+decoder (torch.compile default, dynamic=False)...")
        compiled_encoder = torch.compile(encoder.forward, mode="default", dynamic=False)
        compiled_decoder = torch.compile(decoder.forward, mode="default", dynamic=False)

        def compiled_enc_dec_fp16():
            with torch.autocast("cuda", dtype=torch.float16):
                batched_img_feats = [f.expand(-1, N, -1).contiguous() for f in img_feats]
                batched_img_pos = [p.expand(-1, N, -1).contiguous() for p in img_pos_embeds]

                prompt = text_feats
                prompt_mask_bool = text_mask
                prompt_pos = torch.zeros_like(prompt)

                memory = compiled_encoder(
                    src=batched_img_feats,
                    src_key_padding_mask=None,
                    src_pos=batched_img_pos,
                    prompt=prompt,
                    prompt_pos=prompt_pos,
                    prompt_key_padding_mask=prompt_mask_bool,
                    feat_sizes=vis_feat_sizes,
                )

                query_embed = decoder.query_embed.weight
                tgt = query_embed.unsqueeze(1).expand(-1, N, -1)

                hs, ref_boxes_out, _, _ = compiled_decoder(
                    tgt=tgt,
                    memory=memory["memory"],
                    memory_key_padding_mask=memory["padding_mask"],
                    pos=memory["pos_embed"],
                    reference_boxes=None,
                    level_start_index=memory["level_start_index"],
                    spatial_shapes=memory["spatial_shapes"],
                    valid_ratios=memory["valid_ratios"],
                    tgt_mask=None,
                    memory_text=prompt,
                    text_attention_mask=prompt_mask_bool,
                    apply_dac=False,
                )

                hs = hs.transpose(1, 2)
                ref_boxes_out = ref_boxes_out.transpose(1, 2)

                hs_last = hs[-1:]
                ref_last = ref_boxes_out[-1]

                scores = scoring(hs_last, prompt, prompt_mask_bool)[0]
                box_offsets = decoder.bbox_embed(hs_last)[0]
                boxes = (inverse_sigmoid(ref_last) + box_offsets).sigmoid()

                return scores, boxes

        out, stats = benchmark_fn(
            compiled_enc_dec_fp16, args.warmup, args.runs,
            "torch.compile default + FP16"
        )
        print_quality("compile default FP16", out[0], out[1], ref_scores, ref_boxes)
        results.append(stats)
    except Exception as e:
        print(f"\n  torch.compile FAILED: {e}")
        print("  Skipping compiled benchmark.")
    finally:
        torch._dynamo.reset()
        torch.cuda.empty_cache()

    # ---------------------------------------------------------------
    # 4 & 5. TensorRT FP16
    # ---------------------------------------------------------------
    if not args.no_trt:
        try:
            import tensorrt as trt
            print(f"\nTensorRT: {trt.__version__}")
        except ImportError:
            print("\nTensorRT not installed, skipping TRT tests.")
            args.no_trt = True

    if not args.no_trt:
        onnx_path = args.onnx
        max_classes = args.max_classes
        assert N <= max_classes, f"num_classes={N} > max_classes={max_classes}"

        # Export ONNX if needed
        if args.rebuild or not os.path.exists(onnx_path):
            print(f"\nExporting enc_dec ONNX -> {onnx_path} ...")
            from sam3.trt.export_enc_dec import export_onnx
            export_onnx(
                checkpoint_path=args.checkpoint,
                output_path=onnx_path,
                max_classes=max_classes,
            )

        from sam3.trt.trt_enc_dec import TRTEncoderDecoder

        # --- 4. TRT FP16 pure (no mixed precision) ---
        engine_pure = args.engine or "enc_dec_fp16_pure.engine"
        if args.rebuild or not os.path.exists(engine_pure):
            print(f"\nBuilding TRT pure FP16 engine -> {engine_pure} ...")
            build_enc_dec_engine(onnx_path, engine_pure, mixed_precision=None)

        trt_pure = TRTEncoderDecoder(
            engine_path=engine_pure, max_classes=max_classes, device=device,
        )

        def trt_pure_fn():
            return trt_pure.forward(
                img_feats=img_feats,
                img_pos_embeds=img_pos_embeds,
                text_feats=text_feats,
                text_mask=text_mask,
                num_classes=N,
            )

        out, stats = benchmark_fn(
            trt_pure_fn, args.warmup, args.runs,
            "TensorRT FP16 (pure)"
        )
        print_quality("TRT FP16 pure", out[0], out[1], ref_scores, ref_boxes)
        results.append(stats)

        del trt_pure
        torch.cuda.empty_cache()

        # --- 5. TRT FP16 with norm-only mixed precision ---
        engine_norm = "enc_dec_fp16_norm.engine"
        if args.rebuild or not os.path.exists(engine_norm):
            print(f"\nBuilding TRT FP16 norm-only engine -> {engine_norm} ...")
            try:
                build_enc_dec_engine(
                    onnx_path, engine_norm, mixed_precision="norm-only"
                )
            except RuntimeError as e:
                print(f"  Build failed: {e}")
                engine_norm = None

        if engine_norm and os.path.exists(engine_norm):
            trt_norm = TRTEncoderDecoder(
                engine_path=engine_norm, max_classes=max_classes, device=device,
            )

            def trt_norm_fn():
                return trt_norm.forward(
                    img_feats=img_feats,
                    img_pos_embeds=img_pos_embeds,
                    text_feats=text_feats,
                    text_mask=text_mask,
                    num_classes=N,
                )

            out, stats = benchmark_fn(
                trt_norm_fn, args.warmup, args.runs,
                "TensorRT FP16 (norm FP32)"
            )
            print_quality("TRT FP16 norm", out[0], out[1], ref_scores, ref_boxes)
            results.append(stats)

            del trt_norm
            torch.cuda.empty_cache()

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------
    print(f"\n\n{'='*75}")
    print("SUMMARY - Encoder+Decoder+Scoring Pipeline")
    print(f"{'='*75}")
    print(f"  Classes: {args.classes} (N={N})")
    print(f"  Image features: {img_feats[-1].shape}")
    print()
    print(f"{'Backend':<42s} {'Avg':>8s} {'Min':>8s} {'P50':>8s} {'P95':>8s}")
    print(f"{'-'*42} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for r in results:
        print(
            f"{r['label']:<42s} "
            f"{r['avg_ms']:>7.2f}ms "
            f"{r['min_ms']:>7.2f}ms "
            f"{r['p50']:>7.2f}ms "
            f"{r['p95']:>7.2f}ms"
        )

    # Speedup vs FP32 eager
    ref_avg = results[0]["avg_ms"]
    print()
    for r in results[1:]:
        speedup = ref_avg / r["avg_ms"]
        print(f"  {r['label']}: {speedup:.2f}x vs FP32 eager")


if __name__ == "__main__":
    main()
