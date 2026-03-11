#!/usr/bin/env python3
"""Benchmark TRT FP16 backbone with different mixed-precision strategies.

Builds multiple TRT engines with varying precision overrides and compares
each against PyTorch FP32 reference outputs for accuracy (cosine similarity)
and speed (ms/inference).

Usage:
    python scripts/benchmark_fp16_precision.py \
        --onnx backbone.onnx \
        --checkpoint sam3.pt \
        --image x.jpg

Strategies tested:
    1. pure-fp16:           All layers FP16 (broken, cos~0.07)
    2. norm-only:           Only NORMALIZATION+SOFTMAX in FP32
    3. norm-softmax-reduce: NORMALIZATION+SOFTMAX+REDUCE in FP32
    4. attention:           Attention MatMul+Softmax+Norm in FP32 (existing)
    5. all-matmul:          All MatMul+Softmax+Norm in FP32
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def cosine_similarity(a, b):
    """Compute cosine similarity between two flattened tensors."""
    a_flat = a.float().flatten()
    b_flat = b.float().flatten()
    return torch.nn.functional.cosine_similarity(
        a_flat.unsqueeze(0), b_flat.unsqueeze(0)
    ).item()


def get_pytorch_reference(checkpoint_path, image, device="cuda"):
    """Run PyTorch FP32 backbone and return reference outputs."""
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_multiclass_fast import Sam3MultiClassPredictorFast

    print("Loading PyTorch model for reference...")
    model = build_sam3_image_model(
        device=device, checkpoint_path=checkpoint_path, eval_mode=True,
    )
    predictor = Sam3MultiClassPredictorFast(
        model, device=device, resolution=1008,
        use_fp16=False, detection_only=True,
    )

    # Run backbone
    with torch.inference_mode():
        img_tensor = predictor.transform(image).unsqueeze(0).to(device)
        backbone_out = model.backbone(img_tensor)

    # Extract FPN features
    fpn_keys = sorted(backbone_out.keys())
    ref_outputs = {k: backbone_out[k].clone() for k in fpn_keys}
    print(f"  Reference outputs: {[f'{k}: {v.shape}' for k, v in ref_outputs.items()]}")

    # Cleanup
    del model, predictor
    torch.cuda.empty_cache()

    return ref_outputs, img_tensor


def build_and_benchmark_engine(
    onnx_path, strategy, img_tensor, ref_outputs,
    workspace_gb=4.0, opt_level=3, n_warmup=5, n_runs=20,
):
    """Build a TRT engine with the given strategy, benchmark speed and accuracy."""
    import tensorrt as trt
    from sam3.trt.trt_backbone import TRTBackbone

    engine_path = f"backbone_{strategy}.engine"
    print(f"\n{'='*60}")
    print(f"Strategy: {strategy}")
    print(f"{'='*60}")

    # Build engine
    from sam3.trt.build_engine import build_engine
    build_engine(
        onnx_path=onnx_path,
        output_path=engine_path,
        engine_type="backbone",
        fp16=True,
        mixed_precision=strategy if strategy != "pure-fp16" else "none",
        workspace_gb=workspace_gb,
        opt_level=opt_level,
    )

    # Load engine
    print(f"Loading engine: {engine_path}")
    trt_backbone = TRTBackbone(engine_path, device="cuda")

    # Warmup
    print(f"Warming up ({n_warmup} runs)...")
    with torch.inference_mode():
        for _ in range(n_warmup):
            trt_backbone.forward_image(img_tensor)
    torch.cuda.synchronize()

    # Benchmark speed
    print(f"Benchmarking ({n_runs} runs)...")
    times = []
    trt_out = None
    with torch.inference_mode():
        for i in range(n_runs):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = trt_backbone.forward_image(img_tensor)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)
            if i == n_runs - 1:
                trt_out = out

    avg_ms = np.mean(times)
    min_ms = np.min(times)
    std_ms = np.std(times)

    # Compute accuracy
    cos_scores = {}
    for key in ref_outputs:
        if key in trt_out:
            cos = cosine_similarity(ref_outputs[key], trt_out[key])
            cos_scores[key] = cos

    # Cleanup
    del trt_backbone
    torch.cuda.empty_cache()

    # Clean up engine file
    # (keep it for now so user can inspect)

    return {
        "strategy": strategy,
        "avg_ms": avg_ms,
        "min_ms": min_ms,
        "std_ms": std_ms,
        "cosine": cos_scores,
        "engine_path": engine_path,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark TRT FP16 backbone with different precision strategies"
    )
    parser.add_argument("--onnx", required=True, help="Backbone ONNX model")
    parser.add_argument("--checkpoint", required=True, help="SAM3 checkpoint")
    parser.add_argument("--image", required=True, help="Test image")
    parser.add_argument("--workspace", type=float, default=4.0)
    parser.add_argument("--opt-level", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument(
        "--strategies", nargs="+",
        default=["pure-fp16", "norm-only", "norm-softmax-reduce", "attention"],
        help="Strategies to test",
    )
    args = parser.parse_args()

    device = "cuda"
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load image
    image = Image.open(args.image).convert("RGB")
    print(f"Image: {args.image} ({image.size[0]}x{image.size[1]})")

    # Get PyTorch reference
    ref_outputs, img_tensor = get_pytorch_reference(
        args.checkpoint, image, device=device,
    )

    # Benchmark each strategy
    results = []
    for strategy in args.strategies:
        try:
            result = build_and_benchmark_engine(
                onnx_path=args.onnx,
                strategy=strategy,
                img_tensor=img_tensor,
                ref_outputs=ref_outputs,
                workspace_gb=args.workspace,
                opt_level=args.opt_level,
                n_warmup=args.warmup,
                n_runs=args.runs,
            )
            results.append(result)
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({
                "strategy": strategy,
                "avg_ms": float("inf"),
                "min_ms": float("inf"),
                "std_ms": 0,
                "cosine": {},
                "error": str(e),
            })

    # Print summary table
    print(f"\n\n{'='*80}")
    print("RESULTS SUMMARY")
    print(f"{'='*80}")
    print(f"{'Strategy':<25s} {'Avg ms':>8s} {'Min ms':>8s} {'Cos (avg)':>10s} {'Cos (min)':>10s}")
    print(f"{'-'*25} {'-'*8} {'-'*8} {'-'*10} {'-'*10}")
    for r in results:
        if "error" in r:
            print(f"{r['strategy']:<25s} {'ERROR':>8s} {'':>8s} {'':>10s} {r['error']}")
            continue
        cos_vals = list(r["cosine"].values())
        cos_avg = np.mean(cos_vals) if cos_vals else 0
        cos_min = np.min(cos_vals) if cos_vals else 0
        print(
            f"{r['strategy']:<25s} {r['avg_ms']:>8.1f} {r['min_ms']:>8.1f} "
            f"{cos_avg:>10.6f} {cos_min:>10.6f}"
        )

    # Detailed per-output cosines
    print(f"\nPer-output cosine similarities:")
    for r in results:
        if "error" in r:
            continue
        print(f"\n  {r['strategy']}:")
        for key, cos in r["cosine"].items():
            status = "OK" if cos > 0.99 else "WARN" if cos > 0.9 else "BAD"
            print(f"    {key}: {cos:.6f} [{status}]")

    print(f"\n{'='*80}")
    print("Engine files saved (for manual testing):")
    for r in results:
        if "error" not in r:
            print(f"  {r['engine_path']}")


if __name__ == "__main__":
    main()
