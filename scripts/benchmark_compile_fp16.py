#!/usr/bin/env python3
"""Benchmark torch.compile inductor with FP16 autocast on torch 2.10.

Quick focused test: eager FP32 vs torch.compile FP16 autocast (max-autotune).

Usage:
    python scripts/benchmark_compile_fp16.py --checkpoint sam3.pt --image x.jpg
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
    a_flat = a.float().flatten()
    b_flat = b.float().flatten()
    return torch.nn.functional.cosine_similarity(
        a_flat.unsqueeze(0), b_flat.unsqueeze(0)
    ).item()


def benchmark_fn(fn, img_tensor, n_warmup=10, n_runs=50, label=""):
    print(f"\n--- {label} ---")
    print(f"  Warming up ({n_warmup} runs)...")
    with torch.inference_mode():
        for _ in range(n_warmup):
            out = fn(img_tensor)
    torch.cuda.synchronize()

    print(f"  Timing ({n_runs} runs)...")
    times = []
    with torch.inference_mode():
        for _ in range(n_runs):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = fn(img_tensor)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)

    avg_ms = np.mean(times)
    min_ms = np.min(times)
    p50 = np.percentile(times, 50)
    p95 = np.percentile(times, 95)
    print(f"  Avg: {avg_ms:.1f}ms  Min: {min_ms:.1f}ms  P50: {p50:.1f}ms  P95: {p95:.1f}ms")
    return out, {"label": label, "avg_ms": avg_ms, "min_ms": min_ms, "p50": p50, "p95": p95}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="sam3.pt")
    parser.add_argument("--image", default="x.jpg")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--runs", type=int, default=50)
    args = parser.parse_args()

    device = "cuda"
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}")

    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_multiclass_fast import Sam3MultiClassPredictorFast
    from torchvision.transforms import v2

    model = build_sam3_image_model(
        device=device, checkpoint_path=args.checkpoint, eval_mode=True,
    )
    predictor = Sam3MultiClassPredictorFast(
        model, device=device, resolution=1008,
        use_fp16=False, detection_only=True,
    )

    image = Image.open(args.image).convert("RGB")
    with torch.inference_mode():
        resized = image.resize((1008, 1008), Image.BILINEAR)
        img_tensor = v2.functional.to_image(resized).to(device)
        img_tensor = predictor.transform(img_tensor).unsqueeze(0)

    backbone = model.backbone
    results = []

    # 1. Eager FP32
    def eager_fn(x):
        return backbone.forward_image(x)

    ref_out, ref_stats = benchmark_fn(
        eager_fn, img_tensor, args.warmup, args.runs, "PyTorch FP32 (eager)"
    )
    ref_fpn = ref_out["backbone_fpn"]
    ref_dict = {f"fpn_{i}": ref_fpn[i] for i in range(len(ref_fpn))}
    results.append(ref_stats)

    def print_cosines(out):
        fpn = out["backbone_fpn"]
        for i in range(len(fpn)):
            k = f"fpn_{i}"
            cos = cosine_similarity(ref_dict[k], fpn[i])
            status = "OK" if cos > 0.999 else "WARN" if cos > 0.99 else "BAD"
            print(f"    {k}: cos={cos:.6f} [{status}]")

    # 2. torch.compile max-autotune FP32
    print("\nCompiling inductor max-autotune FP32...")
    compiled_fp32 = torch.compile(backbone.forward_image, mode="max-autotune", fullgraph=False)
    out, stats = benchmark_fn(compiled_fp32, img_tensor, args.warmup, args.runs,
                              "torch.compile inductor FP32")
    print_cosines(out)
    results.append(stats)

    torch._dynamo.reset()
    torch.cuda.empty_cache()

    # 3. torch.compile max-autotune with FP16 autocast
    print("\nCompiling inductor max-autotune + FP16 autocast...")
    compiled_fp16 = torch.compile(backbone.forward_image, mode="max-autotune", fullgraph=False)

    def compiled_fp16_fn(x):
        with torch.autocast("cuda", dtype=torch.float16):
            return compiled_fp16(x)

    out, stats = benchmark_fn(compiled_fp16_fn, img_tensor, args.warmup, args.runs,
                              "torch.compile inductor FP16 autocast")
    print_cosines(out)
    results.append(stats)

    torch._dynamo.reset()
    torch.cuda.empty_cache()

    # 4. torch.compile reduce-overhead with FP16 autocast
    print("\nCompiling inductor reduce-overhead + FP16 autocast...")
    compiled_ro = torch.compile(backbone.forward_image, mode="reduce-overhead", fullgraph=False)

    def compiled_ro_fn(x):
        with torch.autocast("cuda", dtype=torch.float16):
            return compiled_ro(x)

    out, stats = benchmark_fn(compiled_ro_fn, img_tensor, args.warmup, args.runs,
                              "torch.compile reduce-overhead FP16")
    print_cosines(out)
    results.append(stats)

    # Summary
    print(f"\n\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"{'Backend':<45s} {'Avg':>7s} {'Min':>7s} {'P50':>7s}")
    print(f"{'-'*45} {'-'*7} {'-'*7} {'-'*7}")
    for r in results:
        print(f"{r['label']:<45s} {r['avg_ms']:>6.1f}ms {r['min_ms']:>6.1f}ms {r['p50']:>6.1f}ms")


if __name__ == "__main__":
    main()
