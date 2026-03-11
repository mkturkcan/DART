#!/usr/bin/env python3
"""Test torch.compile with different modes for SAM3 backbone.

torch.compile with inductor backend uses cuBLAS for MatMul, which provides
FP32 accumulation even in FP16 mode. This is exactly what TRT FP16 lacks.

The 'reduce-overhead' mode captures CUDA graphs to eliminate kernel launch
overhead, which should make compiled code significantly faster than eager.

Previous results:
- PyTorch eager FP16: ~87ms (correct)
- torch.compile max-autotune: 174ms (correct but slow - autotuning overhead?)
- TRT FP16 (ONNX): 44ms (broken)
- TRT FP16 (torch_tensorrt): 54ms (broken)
- TRT mixed precision: 130ms (correct but slow)

Target: <66ms with correct results (cos >0.99)
"""

import sys
import time
import torch
import torch.nn as nn
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DEVICE = "cuda"
DTYPE = torch.float16


class BackboneWrapper(nn.Module):
    """Wraps backbone to return tuple instead of dict."""
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, images):
        out = self.backbone.forward_image(images)
        fpn = out["backbone_fpn"]
        return fpn[0], fpn[1], fpn[2]


def cosine_sim(a, b):
    return torch.nn.functional.cosine_similarity(
        a.float().flatten().unsqueeze(0),
        b.float().flatten().unsqueeze(0),
    ).item()


def benchmark(fn, dummy, warmup=20, iters=100):
    """Careful benchmark with generous warmup."""
    with torch.inference_mode():
        # Extended warmup
        for _ in range(warmup):
            fn(dummy)
        torch.cuda.synchronize()

        # Benchmark
        t0 = time.perf_counter()
        for _ in range(iters):
            fn(dummy)
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / iters * 1000


def test_mode(model, dummy, pt_ref, mode_name, compile_kwargs):
    """Test a specific torch.compile configuration."""
    print(f"\n{'=' * 80}")
    print(f"torch.compile: {mode_name}")
    print(f"  kwargs: {compile_kwargs}")
    print("=" * 80)

    backbone = model.backbone

    print("  Compiling (may take several minutes for max-autotune)...")
    t0 = time.time()
    try:
        compiled = torch.compile(backbone.forward_image, **compile_kwargs)

        # First call triggers compilation
        with torch.inference_mode(), torch.autocast("cuda", dtype=DTYPE):
            out = compiled(dummy)
        fpn = out["backbone_fpn"]
    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback
        traceback.print_exc()
        return None, None

    compile_time = time.time() - t0
    print(f"  Compilation took: {compile_time:.0f}s")

    cos = [cosine_sim(pt_ref[i], fpn[i]) for i in range(3)]

    # Extended warmup for CUDA graph modes
    print("  Warming up (20 iterations)...")
    with torch.inference_mode(), torch.autocast("cuda", dtype=DTYPE):
        for _ in range(20):
            compiled(dummy)

    # Benchmark
    print("  Benchmarking (100 iterations)...")
    def run_fn(x):
        with torch.autocast("cuda", dtype=DTYPE):
            return compiled(x)
    ms = benchmark(run_fn, dummy)

    status = "OK" if cos[-1] > 0.99 else "BROKEN" if cos[-1] < 0.5 else "DEGRADED"
    print(f"  cos=[{cos[0]:.4f}, {cos[1]:.4f}, {cos[2]:.4f}] | {ms:.1f}ms | {status}")

    return cos, ms


def test_cuda_graph_manual(model, dummy, pt_ref):
    """Test manual CUDA graph capture for maximum speed."""
    print(f"\n{'=' * 80}")
    print("Manual CUDA Graph capture")
    print("=" * 80)

    backbone = model.backbone

    # Warmup
    print("  Warming up...")
    with torch.inference_mode(), torch.autocast("cuda", dtype=DTYPE):
        for _ in range(3):
            out = backbone.forward_image(dummy)

    # Capture CUDA graph
    print("  Capturing CUDA graph...")
    static_input = dummy.clone()

    # Warmup for graph capture
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s), torch.inference_mode(), torch.autocast("cuda", dtype=DTYPE):
        for _ in range(3):
            out = backbone.forward_image(static_input)
    torch.cuda.current_stream().wait_stream(s)

    # Capture
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g), torch.inference_mode(), torch.autocast("cuda", dtype=DTYPE):
        static_out = backbone.forward_image(static_input)

    static_fpn = static_out["backbone_fpn"]

    # Test replay
    print("  Testing accuracy...")
    static_input.copy_(dummy)
    g.replay()
    torch.cuda.synchronize()

    cos = [cosine_sim(pt_ref[i], static_fpn[i]) for i in range(3)]

    # Benchmark
    print("  Benchmarking (100 iterations)...")
    with torch.inference_mode():
        for _ in range(20):
            g.replay()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(100):
            g.replay()
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) / 100 * 1000

    status = "OK" if cos[-1] > 0.99 else "BROKEN" if cos[-1] < 0.5 else "DEGRADED"
    print(f"  cos=[{cos[0]:.4f}, {cos[1]:.4f}, {cos[2]:.4f}] | {ms:.1f}ms | {status}")

    return cos, ms


def main():
    from sam3.model_builder import build_sam3_image_model

    print("Loading SAM3 model...")
    model = build_sam3_image_model(
        device=DEVICE, checkpoint_path="sam3.pt", eval_mode=True,
    )
    dummy = torch.randn(1, 3, 1008, 1008, device=DEVICE)

    print("Getting PyTorch FP16 reference...")
    backbone = model.backbone
    with torch.inference_mode(), torch.autocast("cuda", dtype=DTYPE):
        out = backbone.forward_image(dummy)
    pt_ref = out["backbone_fpn"]

    # Benchmark eager
    def pt_eager(x):
        with torch.autocast("cuda", dtype=DTYPE):
            return backbone.forward_image(x)
    ms_eager = benchmark(pt_eager, dummy)
    print(f"  PyTorch eager FP16: {ms_eager:.1f}ms")

    results = {"eager": ([1.0, 1.0, 1.0], ms_eager)}

    # ================================================================
    # Mode 1: default (basic optimization, no CUDA graphs)
    # ================================================================
    try:
        cos, ms = test_mode(model, dummy, pt_ref, "default",
                           {"mode": "default"})
        if cos: results["default"] = (cos, ms)
    except Exception as e:
        print(f"  FAILED: {e}")
    torch._dynamo.reset()
    torch.cuda.empty_cache()

    # ================================================================
    # Mode 2: reduce-overhead (CUDA graphs for reduced launch overhead)
    # ================================================================
    try:
        cos, ms = test_mode(model, dummy, pt_ref, "reduce-overhead",
                           {"mode": "reduce-overhead"})
        if cos: results["reduce-overhead"] = (cos, ms)
    except Exception as e:
        print(f"  FAILED: {e}")
    torch._dynamo.reset()
    torch.cuda.empty_cache()

    # ================================================================
    # Mode 3: max-autotune (autotuning + CUDA graphs)
    # ================================================================
    try:
        cos, ms = test_mode(model, dummy, pt_ref, "max-autotune",
                           {"mode": "max-autotune"})
        if cos: results["max-autotune"] = (cos, ms)
    except Exception as e:
        print(f"  FAILED: {e}")
    torch._dynamo.reset()
    torch.cuda.empty_cache()

    # ================================================================
    # Mode 4: max-autotune-no-cudagraphs
    # ================================================================
    try:
        cos, ms = test_mode(model, dummy, pt_ref, "max-autotune-no-cudagraphs",
                           {"mode": "max-autotune-no-cudagraphs"})
        if cos: results["max-autotune-no-cg"] = (cos, ms)
    except Exception as e:
        print(f"  FAILED: {e}")
    torch._dynamo.reset()
    torch.cuda.empty_cache()

    # ================================================================
    # Mode 5: Manual CUDA graph capture (raw, no torch.compile)
    # ================================================================
    try:
        cos, ms = test_cuda_graph_manual(model, dummy, pt_ref)
        if cos: results["manual_cuda_graph"] = (cos, ms)
    except Exception as e:
        print(f"  FAILED: {e}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"  {'Approach':>30s} | {'FPN[-1] cos':>12s} | {'Speed':>7s} | Status")
    print("  " + "-" * 65)
    for name, (cos, ms) in sorted(results.items(), key=lambda x: x[1][1]):
        status = "OK" if cos[-1] > 0.99 else "BROKEN" if cos[-1] < 0.5 else "DEGRADED"
        print(f"  {name:>30s} | {cos[-1]:>12.4f} | {ms:>5.1f}ms | {status}")

    print("\nDone!")


if __name__ == "__main__":
    main()
