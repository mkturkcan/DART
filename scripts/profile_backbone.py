#!/usr/bin/env python3
"""Profile SAM3 backbone to find bottleneck operations."""

import sys
import time
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DEVICE = "cuda"
DTYPE = torch.float16


def benchmark(fn, dummy, warmup=10, iters=50):
    with torch.inference_mode():
        for _ in range(warmup):
            fn(dummy)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn(dummy)
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / iters * 1000


def profile_blocks(model, dummy):
    """Profile individual ViT blocks using hooks."""
    backbone = model.backbone
    trunk = backbone.vision_backbone.trunk

    global_blocks = set(trunk.full_attn_ids)
    print(f"Global attention block indices: {sorted(global_blocks)}")
    print(f"Window size: {trunk.blocks[0].attn.window_size if hasattr(trunk.blocks[0].attn, 'window_size') else 'N/A'}")

    # Use hooks to time blocks
    block_times = {}

    class Timer:
        def __init__(self, name):
            self.name = name
            self.start_event = None
            self.end_event = None

        def pre_hook(self, module, input):
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.end_event = torch.cuda.Event(enable_timing=True)
            self.start_event.record()

        def post_hook(self, module, input, output):
            self.end_event.record()

    timers = []
    hooks = []
    for i, block in enumerate(trunk.blocks):
        t = Timer(f"block_{i}")
        timers.append(t)
        hooks.append(block.register_forward_pre_hook(t.pre_hook))
        hooks.append(block.register_forward_hook(t.post_hook))

    # Run forward pass
    with torch.inference_mode(), torch.autocast("cuda", dtype=DTYPE):
        # Warmup
        for _ in range(3):
            backbone.forward_image(dummy)

        # Timed run
        backbone.forward_image(dummy)
        torch.cuda.synchronize()

    # Collect times
    for i, t in enumerate(timers):
        ms = t.start_event.elapsed_time(t.end_event)
        block_times[i] = ms

    # Remove hooks
    for h in hooks:
        h.remove()

    # Print results
    print(f"\n{'Block':>8s} | {'Type':>8s} | {'Time (ms)':>10s}")
    print("-" * 35)

    total_global = 0
    total_window = 0
    for i in range(len(trunk.blocks)):
        is_global = i in global_blocks
        label = "GLOBAL" if is_global else "window"
        ms = block_times[i]
        print(f"  {i:>5d} | {label:>8s} | {ms:>8.2f}ms")
        if is_global:
            total_global += ms
        else:
            total_window += ms

    total = total_global + total_window
    print(f"\nTotal blocks: {total:.1f}ms")
    print(f"  Global (4):  {total_global:.1f}ms ({total_global/total*100:.0f}%)")
    print(f"  Window (28): {total_window:.1f}ms ({total_window/total*100:.0f}%)")


def test_resolutions(model):
    """Test different resolutions for speed."""
    backbone = model.backbone

    # Resolutions must be divisible by patch_size * window_size
    # Patch size = 16, window_size = 14
    # So feature map = res/16, and windowed blocks use 14x14 windows
    resolutions = [1008, 896, 864, 784, 672, 560]

    print(f"\n{'=' * 60}")
    print("Resolution scaling (eager FP16)")
    print(f"{'=' * 60}")
    print(f"  {'Resolution':>10s} | {'Feature map':>12s} | {'Speed':>7s}")
    print("  " + "-" * 40)

    for res in resolutions:
        dummy = torch.randn(1, 3, res, res, device=DEVICE)
        try:
            def run_fn(x):
                with torch.autocast("cuda", dtype=DTYPE):
                    return backbone.forward_image(x)

            with torch.inference_mode(), torch.autocast("cuda", dtype=DTYPE):
                out = backbone.forward_image(dummy)
            fpn = out["backbone_fpn"]
            fm_h, fm_w = fpn[-1].shape[2], fpn[-1].shape[3]

            ms = benchmark(run_fn, dummy, warmup=5, iters=30)
            print(f"  {res:>10d} | {fm_h}x{fm_w:>3d} | {ms:>5.1f}ms")
        except Exception as e:
            print(f"  {res:>10d} | FAILED: {e}")
        del dummy
        torch.cuda.empty_cache()


def test_compiled_resolutions(model):
    """Test torch.compile speed at different resolutions."""
    backbone = model.backbone

    resolutions = [1008, 896, 864, 784, 672]

    print(f"\n{'=' * 60}")
    print("Resolution scaling (torch.compile max-autotune)")
    print(f"{'=' * 60}")
    print(f"  {'Resolution':>10s} | {'Speed':>7s}")
    print("  " + "-" * 25)

    for res in resolutions:
        torch._dynamo.reset()
        torch.cuda.empty_cache()

        dummy = torch.randn(1, 3, res, res, device=DEVICE)

        compiled = torch.compile(backbone.forward_image, mode="max-autotune")

        def run_fn(x):
            with torch.autocast("cuda", dtype=DTYPE):
                return compiled(x)

        try:
            with torch.inference_mode():
                run_fn(dummy)  # triggers compilation
            ms = benchmark(run_fn, dummy, warmup=10, iters=50)
            print(f"  {res:>10d} | {ms:>5.1f}ms")
        except Exception as e:
            print(f"  {res:>10d} | FAILED: {e}")
        del dummy
        torch.cuda.empty_cache()


def main():
    from sam3.model_builder import build_sam3_image_model

    print("Loading SAM3 model...")
    model = build_sam3_image_model(
        device=DEVICE, checkpoint_path="sam3.pt", eval_mode=True,
    )
    dummy = torch.randn(1, 3, 1008, 1008, device=DEVICE)

    # Profile per-block timing
    profile_blocks(model, dummy)

    # Test resolution scaling
    test_resolutions(model)

    # Test compiled resolution scaling
    test_compiled_resolutions(model)

    print("\nDone!")


if __name__ == "__main__":
    main()
