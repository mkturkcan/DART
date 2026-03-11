#!/usr/bin/env python3
"""Benchmark Torch-TensorRT compilation of the SAM3 ViT-H backbone.

Tests torch.compile with backend="torch_tensorrt" using use_fp32_acc=True
to get FP16 compute with FP32 accumulation — solving the matmul accuracy
issue that plagues vanilla TRT FP16.

Compares:
  1. PyTorch FP32 reference (eager)
  2. torch.compile inductor (max-autotune)
  3. Torch-TensorRT FP16 with FP32 accumulation
  4. Torch-TensorRT FP32 (no FP16)

Usage:
    python scripts/benchmark_torch_trt.py \
        --checkpoint sam3.pt \
        --image x.jpg
"""

import argparse
import os
import sys
import time

# Add DLL directories for TensorRT + torch (Windows)
_site = os.path.join(sys.prefix, "Lib", "site-packages")
for _pkg in ["tensorrt_cu12_libs", "tensorrt_cu13_libs", "torch/lib", "torch_tensorrt/lib"]:
    _p = os.path.join(_site, _pkg)
    if os.path.isdir(_p) and hasattr(os, "add_dll_directory"):
        os.add_dll_directory(_p)

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
    """Benchmark a forward function, return timing stats and output."""
    print(f"\n--- {label} ---")

    # Warmup
    print(f"  Warming up ({n_warmup} runs)...")
    with torch.inference_mode():
        for _ in range(n_warmup):
            out = fn(img_tensor)
    torch.cuda.synchronize()

    # Timed runs
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

    # Check torch_tensorrt availability
    try:
        import torch_tensorrt
        print(f"Torch-TensorRT: {torch_tensorrt.__version__}")
    except ImportError:
        print("ERROR: torch_tensorrt not installed. pip install torch-tensorrt")
        sys.exit(1)

    # Load model
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

    # Prepare input
    image = Image.open(args.image).convert("RGB")
    with torch.inference_mode():
        resized = image.resize((1008, 1008), Image.BILINEAR)
        img_tensor = v2.functional.to_image(resized).to(device)
        img_tensor = predictor.transform(img_tensor).unsqueeze(0)

    backbone = model.backbone
    results = []

    # 1. PyTorch FP32 eager reference
    def eager_fn(x):
        return backbone.forward_image(x)

    ref_out, ref_stats = benchmark_fn(
        eager_fn, img_tensor, args.warmup, args.runs,
        "PyTorch FP32 (eager)"
    )
    ref_fpn = ref_out["backbone_fpn"]
    ref_dict = {f"fpn_{i}": ref_fpn[i] for i in range(len(ref_fpn))}
    results.append(ref_stats)

    def print_cosines(out, ref_dict):
        fpn = out["backbone_fpn"]
        for i in range(len(fpn)):
            k = f"fpn_{i}"
            if k in ref_dict:
                cos = cosine_similarity(ref_dict[k], fpn[i])
                status = "OK" if cos > 0.999 else "WARN" if cos > 0.99 else "BAD"
                print(f"    {k}: cos={cos:.6f} [{status}]")

    # 2. torch.compile inductor (max-autotune)
    print("\nCompiling with inductor max-autotune...")
    try:
        backbone_inductor = torch.compile(
            backbone.forward_image,
            mode="max-autotune",
            fullgraph=False,
        )
        out, stats = benchmark_fn(
            backbone_inductor, img_tensor, args.warmup, args.runs,
            "torch.compile inductor (max-autotune)"
        )
        print_cosines(out, ref_dict)
        results.append(stats)
    except Exception as e:
        print(f"  ERROR: {e}")

    # Clear compile caches
    torch._dynamo.reset()
    torch.cuda.empty_cache()

    # Patch RoPE: replace complex64 ops with real arithmetic (same as ONNX export)
    # Torch-TensorRT doesn't support complex64 (from RoPE's view_as_complex).
    print("\nPatching RoPE (complex -> real arithmetic) for TRT compatibility...")
    from sam3.trt.rope_onnx import patch_rope_for_export
    patch_rope_for_export(backbone)

    # Verify patched backbone still produces correct output
    with torch.inference_mode():
        patched_out = backbone.forward_image(img_tensor)
    patched_fpn = patched_out["backbone_fpn"]
    for i in range(len(patched_fpn)):
        cos = cosine_similarity(ref_dict[f"fpn_{i}"], patched_fpn[i])
        print(f"  Patched RoPE sanity check fpn_{i}: cos={cos:.6f}")

    # 3. Torch-TensorRT FP16 with FP32 acc (use_explicit_typing=True)
    print("\nCompiling with Torch-TensorRT FP16 + fp32_acc + explicit_typing...")
    try:
        backbone_trt_fp16_et = torch.compile(
            backbone.forward_image,
            backend="torch_tensorrt",
            dynamic=False,
            options={
                "enabled_precisions": {torch.float16},
                "use_fp32_acc": True,
                "use_explicit_typing": True,
                "min_block_size": 3,
                "debug": False,
                "truncate_double": True,
            },
        )
        out, stats = benchmark_fn(
            backbone_trt_fp16_et, img_tensor, args.warmup, args.runs,
            "Torch-TRT FP16 fp32acc+explicit_typing"
        )
        print_cosines(out, ref_dict)
        results.append(stats)
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()

    # Clear compile caches
    torch._dynamo.reset()
    torch.cuda.empty_cache()

    # 4. Torch-TensorRT FP16 with FP32 acc (no explicit_typing)
    print("\nCompiling with Torch-TensorRT FP16 + fp32_acc (no explicit_typing)...")
    try:
        backbone_trt_fp16 = torch.compile(
            backbone.forward_image,
            backend="torch_tensorrt",
            dynamic=False,
            options={
                "enabled_precisions": {torch.float16},
                "use_fp32_acc": True,
                "use_explicit_typing": False,
                "min_block_size": 3,
                "debug": False,
                "truncate_double": True,
            },
        )
        out, stats = benchmark_fn(
            backbone_trt_fp16, img_tensor, args.warmup, args.runs,
            "Torch-TRT FP16 fp32acc (no explicit)"
        )
        print_cosines(out, ref_dict)
        results.append(stats)
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()

    # Clear compile caches
    torch._dynamo.reset()
    torch.cuda.empty_cache()

    # 5. Torch-TensorRT FP16 without FP32 acc (raw FP16, for comparison)
    print("\nCompiling with Torch-TensorRT FP16 (no fp32_acc)...")
    try:
        backbone_trt_fp16_noacc = torch.compile(
            backbone.forward_image,
            backend="torch_tensorrt",
            dynamic=False,
            options={
                "enabled_precisions": {torch.float16},
                "use_fp32_acc": False,
                "min_block_size": 3,
                "debug": False,
                "truncate_double": True,
            },
        )
        out, stats = benchmark_fn(
            backbone_trt_fp16_noacc, img_tensor, args.warmup, args.runs,
            "Torch-TRT FP16 (no fp32_acc)"
        )
        print_cosines(out, ref_dict)
        results.append(stats)
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()

    # Clear compile caches
    torch._dynamo.reset()
    torch.cuda.empty_cache()

    # 6. Torch-TensorRT FP32 only (no FP16)
    print("\nCompiling with Torch-TensorRT FP32...")
    try:
        backbone_trt_fp32 = torch.compile(
            backbone.forward_image,
            backend="torch_tensorrt",
            dynamic=False,
            options={
                "enabled_precisions": {torch.float32},
                "min_block_size": 3,
                "debug": False,
            },
        )
        out, stats = benchmark_fn(
            backbone_trt_fp32, img_tensor, args.warmup, args.runs,
            "Torch-TRT FP32"
        )
        print_cosines(out, ref_dict)
        results.append(stats)
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()

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
