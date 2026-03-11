#!/usr/bin/env python3
"""Benchmark PyTorch FP16 backbone with various optimizations.

Tests: baseline, cudnn.benchmark, channels_last, torch.compile, CUDA graphs.
"""

import gc
import time
import torch
import torch.nn.functional as F
from sam3.model_builder import build_sam3_image_model

device = "cuda"
N_WARMUP = 10
N_ITERS = 100


def benchmark_fn(fn, dummy, label, n_warmup=N_WARMUP, n_iters=N_ITERS):
    """Benchmark a function with warmup and timing."""
    # Warmup
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        for _ in range(n_warmup):
            fn(dummy)
        torch.cuda.synchronize()

    # Timed iterations
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_iters):
            fn(dummy)
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) / n_iters * 1000

    print(f"  {label}: {ms:.1f} ms/frame")
    return ms


def benchmark_cuda_graph(fn, dummy, label, n_warmup=N_WARMUP, n_iters=N_ITERS):
    """Benchmark with CUDA graph capture and replay."""
    # Pre-allocate static input
    static_input = dummy.clone()

    # Warmup the function first (important for lazy init)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        for _ in range(3):
            fn(static_input)
        torch.cuda.synchronize()

    # Capture CUDA graph
    graph = torch.cuda.CUDAGraph()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        with torch.cuda.graph(graph):
            out = fn(static_input)
    torch.cuda.synchronize()

    # Warmup replay
    for _ in range(n_warmup):
        static_input.copy_(dummy)
        graph.replay()
    torch.cuda.synchronize()

    # Timed replay
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iters):
        static_input.copy_(dummy)
        graph.replay()
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / n_iters * 1000

    print(f"  {label}: {ms:.1f} ms/frame")
    return ms


def main():
    dummy = torch.randn(1, 3, 1008, 1008, device=device)

    # =============================================
    # Test 1: Baseline FP16 autocast
    # =============================================
    print("Loading model...")
    model = build_sam3_image_model(
        device=device, checkpoint_path="sam3.pt", eval_mode=True,
    )
    backbone = model.backbone

    print("\n=== Test 1: Baseline FP16 autocast ===")
    benchmark_fn(backbone.forward_image, dummy, "Baseline")

    # =============================================
    # Test 2: cudnn.benchmark = True
    # =============================================
    print("\n=== Test 2: + cudnn.benchmark ===")
    torch.backends.cudnn.benchmark = True
    # Need a few extra warmup iters for cudnn to find best algorithms
    benchmark_fn(backbone.forward_image, dummy, "cudnn.benchmark", n_warmup=20)

    # =============================================
    # Test 3: channels_last memory format on Conv layers
    # =============================================
    print("\n=== Test 3: + channels_last ===")
    # Convert conv-heavy parts to channels_last
    # PatchEmbed conv and FPN neck convs
    trunk = backbone.vision_backbone.trunk
    if hasattr(trunk, 'patch_embed'):
        trunk.patch_embed = trunk.patch_embed.to(memory_format=torch.channels_last)

    neck = backbone.vision_backbone
    # Convert the whole neck module to channels_last
    for name, module in neck.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)):
            module.to(memory_format=torch.channels_last)

    dummy_cl = dummy.to(memory_format=torch.channels_last)
    benchmark_fn(backbone.forward_image, dummy_cl, "channels_last")
    # Also test with regular input format (channels_last conv still benefits)
    benchmark_fn(backbone.forward_image, dummy, "channels_last (NCHW input)")

    # =============================================
    # Test 4: torch.compile (reduce-overhead mode)
    # =============================================
    print("\n=== Test 4: + torch.compile (reduce-overhead) ===")
    try:
        compiled_fn = torch.compile(
            backbone.forward_image,
            mode="reduce-overhead",
            dynamic=False,
        )
        # Extra warmup for compilation
        print("  Compiling (first few runs will be slow)...")
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            for i in range(5):
                t0 = time.perf_counter()
                compiled_fn(dummy)
                torch.cuda.synchronize()
                print(f"    Warmup {i}: {(time.perf_counter()-t0)*1000:.0f}ms")
        benchmark_fn(compiled_fn, dummy, "compile(reduce-overhead)")
    except Exception as e:
        print(f"  FAILED: {e}")

    # =============================================
    # Test 5: torch.compile (max-autotune mode)
    # =============================================
    print("\n=== Test 5: + torch.compile (max-autotune) ===")
    try:
        compiled_fn2 = torch.compile(
            backbone.forward_image,
            mode="max-autotune",
            dynamic=False,
        )
        print("  Compiling (first few runs will be slow)...")
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            for i in range(5):
                t0 = time.perf_counter()
                compiled_fn2(dummy)
                torch.cuda.synchronize()
                print(f"    Warmup {i}: {(time.perf_counter()-t0)*1000:.0f}ms")
        benchmark_fn(compiled_fn2, dummy, "compile(max-autotune)")
    except Exception as e:
        print(f"  FAILED: {e}")

    # =============================================
    # Test 6: CUDA Graph
    # =============================================
    print("\n=== Test 6: CUDA Graph ===")
    try:
        benchmark_cuda_graph(backbone.forward_image, dummy, "CUDA Graph")
    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback
        traceback.print_exc()

    # =============================================
    # Test 7: torch.compile + CUDA Graph (via reduce-overhead)
    # =============================================
    print("\n=== Test 7: compile(reduce-overhead) should auto-use CUDA graphs ===")
    # reduce-overhead mode already uses CUDA graphs internally
    # Just re-benchmark the compiled version after full warmup
    try:
        benchmark_fn(compiled_fn, dummy, "compile(reduce-overhead) post-warmup",
                     n_warmup=20, n_iters=200)
    except Exception as e:
        print(f"  FAILED: {e}")

    print("\nDone!")


if __name__ == "__main__":
    main()
