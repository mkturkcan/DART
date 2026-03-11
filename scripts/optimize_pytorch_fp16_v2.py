#!/usr/bin/env python3
"""Benchmark PyTorch FP16 backbone optimizations (v2).

Focuses on approaches that work without triton:
- torch.compile with backend="eager" and "aot_eager"
- CUDA graphs (manual)
- Monkey-patching triton for inductor backend
"""

import gc
import time
import torch
from sam3.model_builder import build_sam3_image_model

device = "cuda"
N_WARMUP = 10
N_ITERS = 100


def bench(fn, dummy, label, n_warmup=N_WARMUP, n_iters=N_ITERS):
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        for _ in range(n_warmup):
            fn(dummy)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_iters):
            fn(dummy)
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) / n_iters * 1000
    print(f"  {label}: {ms:.1f} ms/frame")
    return ms


def main():
    dummy = torch.randn(1, 3, 1008, 1008, device=device)

    print("Loading model...")
    model = build_sam3_image_model(
        device=device, checkpoint_path="sam3.pt", eval_mode=True,
    )
    backbone = model.backbone
    torch.backends.cudnn.benchmark = True

    # Baseline
    print("\n=== Baseline FP16 autocast ===")
    bench(backbone.forward_image, dummy, "Baseline")

    # Try patching triton to fix inductor
    print("\n=== Patching triton for inductor ===")
    try:
        from triton.compiler import compiler as triton_compiler
        if not hasattr(triton_compiler, 'triton_key'):
            # Alias get_cache_key as triton_key
            triton_compiler.triton_key = triton_compiler.get_cache_key
            print("  Patched triton_key = get_cache_key")
    except Exception as e:
        print(f"  Patch failed: {e}")

    # torch.compile with inductor (after patch)
    print("\n=== torch.compile (inductor, reduce-overhead) ===")
    try:
        compiled_inductor = torch.compile(
            backbone.forward_image,
            mode="reduce-overhead",
            dynamic=False,
        )
        # Compile warmup
        print("  Compiling...")
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            for i in range(5):
                t0 = time.perf_counter()
                compiled_inductor(dummy)
                torch.cuda.synchronize()
                print(f"    Warmup {i}: {(time.perf_counter()-t0)*1000:.0f}ms")
        bench(compiled_inductor, dummy, "inductor(reduce-overhead)")
    except Exception as e:
        print(f"  FAILED: {e}")

    # torch.compile with inductor max-autotune
    print("\n=== torch.compile (inductor, max-autotune) ===")
    try:
        compiled_autotune = torch.compile(
            backbone.forward_image,
            mode="max-autotune",
            dynamic=False,
        )
        print("  Compiling...")
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            for i in range(5):
                t0 = time.perf_counter()
                compiled_autotune(dummy)
                torch.cuda.synchronize()
                print(f"    Warmup {i}: {(time.perf_counter()-t0)*1000:.0f}ms")
        bench(compiled_autotune, dummy, "inductor(max-autotune)")
    except Exception as e:
        print(f"  FAILED: {e}")

    # torch.compile default mode
    print("\n=== torch.compile (inductor, default) ===")
    try:
        compiled_default = torch.compile(
            backbone.forward_image,
            dynamic=False,
        )
        print("  Compiling...")
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            for i in range(5):
                t0 = time.perf_counter()
                compiled_default(dummy)
                torch.cuda.synchronize()
                print(f"    Warmup {i}: {(time.perf_counter()-t0)*1000:.0f}ms")
        bench(compiled_default, dummy, "inductor(default)")
    except Exception as e:
        print(f"  FAILED: {e}")

    # Final long benchmark of best
    print("\n=== Final long benchmark (200 iters) ===")
    bench(backbone.forward_image, dummy, "Baseline", n_warmup=20, n_iters=200)
    try:
        bench(compiled_inductor, dummy, "inductor(reduce-overhead)", n_warmup=20, n_iters=200)
    except:
        pass

    print("\nDone!")


if __name__ == "__main__":
    main()
