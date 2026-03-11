#!/usr/bin/env python3
"""Test FP16 reduction (accumulation) in PyTorch matmul.

By default, PyTorch uses FP32 accumulation for FP16 matmul (via cuBLAS).
TRT uses FP16 accumulation, which is faster but less precise.

The flag `torch.backends.cuda.matmul.allow_fp16_reduction` controls this.
When True, cuBLAS may use FP16 accumulation (like TRT).

If this flag causes the same degradation as TRT, it confirms the root cause
is FP16 accumulation (not the graph structure or other TRT issues).

Also test: what if we only allow FP16 reduction for MLP (fc1, fc2) but not
attention? This might give us most of the speed while keeping accuracy.
"""

import sys
import time
import torch
import torch.nn as nn
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DEVICE = "cuda"
DTYPE = torch.float16


def cosine_sim(a, b):
    return torch.nn.functional.cosine_similarity(
        a.float().flatten().unsqueeze(0),
        b.float().flatten().unsqueeze(0),
    ).item()


def benchmark(fn, dummy, warmup=20, iters=100):
    with torch.inference_mode():
        for _ in range(warmup):
            fn(dummy)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn(dummy)
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / iters * 1000


def main():
    from sam3.model_builder import build_sam3_image_model

    print("Loading SAM3 model...")
    model = build_sam3_image_model(
        device=DEVICE, checkpoint_path="sam3.pt", eval_mode=True,
    )
    dummy = torch.randn(1, 3, 1008, 1008, device=DEVICE)

    backbone = model.backbone

    # Reference with FP32 accumulation (default)
    print("\n=== PyTorch FP16 with FP32 accumulation (default) ===")
    torch.backends.cuda.matmul.allow_fp16_reduction = False
    with torch.inference_mode(), torch.autocast("cuda", dtype=DTYPE):
        out_fp32acc = backbone.forward_image(dummy)
    ref_fpn = out_fp32acc["backbone_fpn"]

    def run_fp32acc(x):
        with torch.autocast("cuda", dtype=DTYPE):
            return backbone.forward_image(x)
    ms_fp32 = benchmark(run_fp32acc, dummy)
    print(f"  Speed: {ms_fp32:.1f}ms")

    # FP16 accumulation
    print("\n=== PyTorch FP16 with FP16 accumulation (allow_fp16_reduction=True) ===")
    torch.backends.cuda.matmul.allow_fp16_reduction = True
    with torch.inference_mode(), torch.autocast("cuda", dtype=DTYPE):
        out_fp16acc = backbone.forward_image(dummy)
    fpn_fp16 = out_fp16acc["backbone_fpn"]

    cos = [cosine_sim(ref_fpn[i], fpn_fp16[i]) for i in range(3)]
    status = "OK" if cos[-1] > 0.99 else "BROKEN" if cos[-1] < 0.5 else "DEGRADED"
    print(f"  cos=[{cos[0]:.4f}, {cos[1]:.4f}, {cos[2]:.4f}] | {status}")

    def run_fp16acc(x):
        with torch.autocast("cuda", dtype=DTYPE):
            return backbone.forward_image(x)
    ms_fp16 = benchmark(run_fp16acc, dummy)
    print(f"  Speed: {ms_fp16:.1f}ms (vs {ms_fp32:.1f}ms, speedup: {ms_fp32/ms_fp16:.2f}x)")

    # Test with torch.compile + FP16 reduction
    print("\n=== torch.compile max-autotune + FP16 accumulation ===")
    torch.backends.cuda.matmul.allow_fp16_reduction = True
    compiled = torch.compile(backbone.forward_image, mode="max-autotune")

    with torch.inference_mode(), torch.autocast("cuda", dtype=DTYPE):
        out_compiled = compiled(dummy)
    fpn_compiled = out_compiled["backbone_fpn"]

    cos_c = [cosine_sim(ref_fpn[i], fpn_compiled[i]) for i in range(3)]
    status_c = "OK" if cos_c[-1] > 0.99 else "BROKEN" if cos_c[-1] < 0.5 else "DEGRADED"
    print(f"  cos=[{cos_c[0]:.4f}, {cos_c[1]:.4f}, {cos_c[2]:.4f}] | {status_c}")

    def run_compiled(x):
        with torch.autocast("cuda", dtype=DTYPE):
            return compiled(x)
    ms_compiled = benchmark(run_compiled, dummy)
    print(f"  Speed: {ms_compiled:.1f}ms")

    # Reset and test torch.compile with FP32 accumulation for comparison
    torch._dynamo.reset()
    torch.cuda.empty_cache()
    torch.backends.cuda.matmul.allow_fp16_reduction = False

    print("\n=== torch.compile max-autotune + FP32 accumulation (control) ===")
    compiled2 = torch.compile(backbone.forward_image, mode="max-autotune")

    with torch.inference_mode(), torch.autocast("cuda", dtype=DTYPE):
        out_compiled2 = compiled2(dummy)
    fpn_compiled2 = out_compiled2["backbone_fpn"]

    cos_c2 = [cosine_sim(ref_fpn[i], fpn_compiled2[i]) for i in range(3)]
    print(f"  cos=[{cos_c2[0]:.4f}, {cos_c2[1]:.4f}, {cos_c2[2]:.4f}]")

    def run_compiled2(x):
        with torch.autocast("cuda", dtype=DTYPE):
            return compiled2(x)
    ms_compiled2 = benchmark(run_compiled2, dummy)
    print(f"  Speed: {ms_compiled2:.1f}ms")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"  {'Approach':>45s} | {'cos[-1]':>8s} | {'Speed':>7s}")
    print("  " + "-" * 65)
    print(f"  {'Eager FP16 + FP32 acc (default)':>45s} | {'1.0000':>8s} | {ms_fp32:>5.1f}ms")
    print(f"  {'Eager FP16 + FP16 acc':>45s} | {cos[-1]:>8.4f} | {ms_fp16:>5.1f}ms")
    print(f"  {'Compiled FP16 + FP32 acc':>45s} | {cos_c2[-1]:>8.4f} | {ms_compiled2:>5.1f}ms")
    print(f"  {'Compiled FP16 + FP16 acc':>45s} | {cos_c[-1]:>8.4f} | {ms_compiled:>5.1f}ms")

    torch.backends.cuda.matmul.allow_fp16_reduction = False
    print("\nDone!")


if __name__ == "__main__":
    main()
