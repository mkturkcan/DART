#!/usr/bin/env python3
"""Benchmark torch.compile on PyTorch FP16 backbone."""

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
    torch.backends.cudnn.benchmark = True

    print("Loading model...")
    model = build_sam3_image_model(
        device=device, checkpoint_path="sam3.pt", eval_mode=True,
    )
    backbone = model.backbone

    print("\n=== Baseline FP16 ===")
    bench(backbone.forward_image, dummy, "Baseline")

    # Test torch.compile with different modes
    for mode in ["default", "reduce-overhead", "max-autotune"]:
        print(f"\n=== torch.compile (mode={mode}) ===")
        try:
            compiled = torch.compile(
                backbone.forward_image,
                mode=mode,
                dynamic=False,
            )
            print("  Compiling (first runs are slow)...")
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                for i in range(8):
                    t0 = time.perf_counter()
                    compiled(dummy)
                    torch.cuda.synchronize()
                    ms = (time.perf_counter() - t0) * 1000
                    print(f"    Run {i}: {ms:.0f}ms")
            bench(compiled, dummy, f"compile({mode})")

            # Verify correctness
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                ref = backbone.forward_image(dummy)
                comp = compiled(dummy)
            for i in range(3):
                cos = torch.nn.functional.cosine_similarity(
                    ref["backbone_fpn"][i].float().flatten().unsqueeze(0),
                    comp["backbone_fpn"][i].float().flatten().unsqueeze(0),
                ).item()
                print(f"    FPN[{i}] cosine: {cos:.6f}")
        except Exception as e:
            print(f"  FAILED: {e}")

    print("\nDone!")


if __name__ == "__main__":
    main()
