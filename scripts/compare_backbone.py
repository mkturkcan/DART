#!/usr/bin/env python3
"""Compare backbone implementations: speed + cosine similarity.

Benchmarks PyTorch eager, torch.compile, and TRT FP16 backbones on the same
input, reporting both per-call latency and cosine similarity vs the FP32
reference.

NOTE on TRT backbone engines:
  By default, ``build_engine --fp16`` auto-applies mixed-precision (attention
  layers forced to FP32) for backbone engines.  This makes the engine ~128ms
  — slower than torch.compile at ~75ms — but preserves cosine ~1.0.

  To build a PURE FP16 engine (fast ~26ms, but may be broken cos<0.1):
    python -m sam3.trt.build_engine --onnx backbone.onnx \\
        --output backbone_fp16_pure.engine --fp16 --mixed-precision none

  To build a MIXED PRECISION engine (slow ~128ms, correct cos~1.0):
    python -m sam3.trt.build_engine --onnx backbone.onnx \\
        --output backbone_fp16_mixed.engine --fp16

  Pass both to this script with --trt and --trt-mixed to compare all.

Usage:
    # Compare eager vs compile (no TRT):
    python scripts/compare_backbone.py --checkpoint sam3.pt --image x.jpg

    # Compare all (eager, compile, TRT pure FP16, TRT mixed):
    python scripts/compare_backbone.py \\
        --checkpoint sam3.pt --image x.jpg \\
        --trt backbone_fp16_pure.engine \\
        --trt-mixed backbone_fp16_mixed.engine

    # Compare multiple surgical precision engines:
    python scripts/compare_backbone.py \\
        --checkpoint sam3.pt --image x.jpg \\
        --trt-engines "attn_v:backbone_attn_v.engine,global_attn:backbone_global.engine,attn_core:backbone_attn_core.engine"

    # With block pruning:
    python scripts/compare_backbone.py \\
        --checkpoint sam3.pt --image x.jpg \\
        --mask-blocks "25:attn,28:mlp,27:attn,22:attn"

    # Custom resolution:
    python scripts/compare_backbone.py \\
        --checkpoint sam3.pt --image x.jpg --imgsz 644
"""

import argparse
import time

import torch
from PIL import Image
from torchvision.transforms import v2

from sam3.model_builder import (
    build_sam3_image_model,
    build_pruned_sam3_image_model,
    load_pruned_config,
)


def _cosine(a, b):
    """Cosine similarity between two flat tensors."""
    return torch.nn.functional.cosine_similarity(
        a.float().flatten().unsqueeze(0),
        b.float().flatten().unsqueeze(0),
    ).item()


def _benchmark(fn, warmup=5, repeats=20):
    """Time a function with warmup, return median ms."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)

    times.sort()
    return times[len(times) // 2]  # median


def main():
    parser = argparse.ArgumentParser(
        description="Compare backbone implementations: speed + cosine similarity"
    )
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint")
    parser.add_argument("--trt", default=None,
                        help="TRT backbone engine, pure FP16 (--mixed-precision none)")
    parser.add_argument("--trt-mixed", default=None,
                        help="TRT backbone engine, mixed precision (default --fp16 build)")
    parser.add_argument("--trt-engines", default=None,
                        help="Compare multiple TRT engines. Comma-separated label:path pairs, "
                             "e.g. 'attn_v:bb_attn_v.engine,global:bb_global.engine'")
    parser.add_argument("--image", default=None, help="Test image (uses random if omitted)")
    parser.add_argument("--imgsz", type=int, default=1008)
    parser.add_argument(
        "--compile", type=str, default="max-autotune",
        choices=["default", "reduce-overhead", "max-autotune"],
        help="torch.compile mode (default: max-autotune)",
    )
    parser.add_argument("--mask-blocks", type=str, default=None,
                        help="Comma-separated sub-block pruning spec")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations")
    parser.add_argument("--repeats", type=int, default=20, help="Timed iterations")
    args = parser.parse_args()

    device = "cuda"
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Resolution: {args.imgsz}px")

    # --- Load model ---
    mask_blocks = None
    if args.mask_blocks:
        mask_blocks = [s.strip() for s in args.mask_blocks.split(",")]

    print("Loading PyTorch model...")
    pruned_config = load_pruned_config(args.checkpoint)
    if pruned_config is not None:
        print(f"  Pruned checkpoint: {pruned_config}")
        model = build_pruned_sam3_image_model(
            checkpoint_path=args.checkpoint,
            pruning_config=pruned_config,
            device=device, eval_mode=True,
        )
    else:
        model = build_sam3_image_model(
            device=device, checkpoint_path=args.checkpoint, eval_mode=True,
            mask_blocks=mask_blocks,
        )

    if mask_blocks:
        print(f"  Mask blocks: {args.mask_blocks}")

    backbone = model.backbone

    # Precompute position encoding for non-default resolution
    if args.imgsz != 1008:
        pos_enc = backbone.vision_backbone.position_encoding
        pos_enc.precompute_for_resolution(args.imgsz)

    # --- Prepare input ---
    transform = v2.Compose([
        v2.ToDtype(torch.uint8, scale=True),
        v2.Resize(size=(args.imgsz, args.imgsz)),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    if args.image:
        print(f"Image: {args.image}")
        img = Image.open(args.image).convert("RGB")
        img = img.resize((args.imgsz, args.imgsz), Image.BILINEAR)
        tensor = v2.functional.to_image(img).to(device)
        tensor = transform(tensor).unsqueeze(0)
    else:
        print("Image: random tensor")
        tensor = torch.randn(1, 3, args.imgsz, args.imgsz, device=device)

    # ===================================================================
    # 1. PyTorch FP32 (reference)
    # ===================================================================
    print(f"\n--- PyTorch FP32 (reference) ---")
    with torch.inference_mode():
        ref_out = backbone.forward_image(tensor)
    ref_fpn = ref_out["backbone_fpn"][-1]  # last FPN level

    fp32_ms = _benchmark(
        lambda: backbone.forward_image(tensor),
        warmup=args.warmup, repeats=args.repeats,
    )
    print(f"  Median: {fp32_ms:.1f}ms")

    # ===================================================================
    # 2. PyTorch FP16 autocast (eager)
    # ===================================================================
    print(f"\n--- PyTorch FP16 autocast (eager) ---")
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        fp16_out = backbone.forward_image(tensor)
    fp16_fpn = fp16_out["backbone_fpn"][-1]
    cos_fp16 = _cosine(ref_fpn, fp16_fpn)

    def _run_fp16_eager():
        with torch.autocast("cuda", dtype=torch.float16):
            backbone.forward_image(tensor)

    fp16_ms = _benchmark(_run_fp16_eager, warmup=args.warmup, repeats=args.repeats)
    print(f"  Median: {fp16_ms:.1f}ms, cosine vs FP32: {cos_fp16:.6f}")

    # ===================================================================
    # 3. torch.compile FP16
    # ===================================================================
    print(f"\n--- torch.compile({args.compile}) FP16 ---")
    compiled_fn = torch.compile(
        backbone.forward_image, mode=args.compile, dynamic=False,
    )

    # Warmup compile (may take 60-120s for max-autotune)
    print(f"  Compiling (this may take 60-120s for max-autotune)...")
    t_comp = time.perf_counter()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        for _ in range(3):
            compiled_fn(tensor)
    torch.cuda.synchronize()
    print(f"  Compile done ({time.perf_counter() - t_comp:.0f}s)")

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        compile_out = compiled_fn(tensor)
    compile_fpn = compile_out["backbone_fpn"][-1]
    cos_compile = _cosine(ref_fpn, compile_fpn)

    def _run_compiled():
        with torch.autocast("cuda", dtype=torch.float16):
            compiled_fn(tensor)

    compile_ms = _benchmark(_run_compiled, warmup=args.warmup, repeats=args.repeats)
    print(f"  Median: {compile_ms:.1f}ms, cosine vs FP32: {cos_compile:.6f}")

    # ===================================================================
    # 4+. TRT engines (--trt, --trt-mixed, --trt-engines)
    # ===================================================================
    # Collect all TRT engines to test: list of (label, path)
    trt_engines = []
    if args.trt:
        trt_engines.append(("TRT FP16 pure", args.trt))
    if args.trt_mixed:
        trt_engines.append(("TRT FP16 mixed (attn→FP32)", args.trt_mixed))
    if args.trt_engines:
        for item in args.trt_engines.split(","):
            item = item.strip()
            if ":" in item:
                label, path = item.split(":", 1)
                trt_engines.append((f"TRT {label.strip()}", path.strip()))
            else:
                trt_engines.append((f"TRT {item}", item))

    trt_results = []  # list of (label, ms, cosine)
    if trt_engines:
        from sam3.trt.trt_backbone import TRTBackbone
        pos_module = backbone.vision_backbone.position_encoding

    for label, engine_path in trt_engines:
        print(f"\n--- {label} ({engine_path}) ---")
        trt_bb = TRTBackbone(
            engine_path=engine_path, device=device,
            pos_encoding_module=pos_module,
        )

        with torch.inference_mode():
            trt_out = trt_bb.forward_image(tensor)
        trt_fpn = trt_out["backbone_fpn"][-1]
        cos = _cosine(ref_fpn, trt_fpn)

        # Use a local var to avoid lambda capture issues
        _trt_bb = trt_bb
        ms = _benchmark(
            lambda: _trt_bb.forward_image(tensor),
            warmup=args.warmup, repeats=args.repeats,
        )
        print(f"  Median: {ms:.1f}ms, cosine vs FP32: {cos:.6f}")
        trt_results.append((label, ms, cos))

        # Free engine to avoid GPU memory buildup
        del trt_bb, _trt_bb

    # ===================================================================
    # Summary table
    # ===================================================================
    print(f"\n{'='*78}")
    print(f"SUMMARY ({args.imgsz}px, {args.warmup} warmup, {args.repeats} repeats)")
    if mask_blocks:
        print(f"  Mask blocks: {args.mask_blocks}")
    print(f"{'='*78}")
    print(f"  {'Backend':<40} {'ms':>8} {'Cosine':>10} {'Speedup':>10}")
    print(f"  {'-'*40} {'-'*8} {'-'*10} {'-'*10}")
    print(f"  {'PyTorch FP32 (reference)':<40} {fp32_ms:>8.1f} {'1.000000':>10} {'ref':>10}")
    print(f"  {'PyTorch FP16 (eager)':<40} {fp16_ms:>8.1f} {cos_fp16:>10.6f} "
          f"{fp32_ms/fp16_ms:>9.2f}x")
    compile_label = f"torch.compile({args.compile}) FP16"
    print(f"  {compile_label:<40} {compile_ms:>8.1f} {cos_compile:>10.6f} "
          f"{fp32_ms/compile_ms:>9.2f}x")
    for label, ms, cos in trt_results:
        status = "OK" if cos > 0.99 else "BROKEN" if cos < 0.5 else "DEGRADED"
        print(f"  {label:<40} {ms:>8.1f} {cos:>10.6f} "
              f"{fp32_ms/ms:>9.2f}x  {status}")
    print(f"{'='*78}")

    # Recommendations
    ok_engines = [(l, ms, c) for l, ms, c in trt_results if c > 0.99]
    broken_engines = [(l, ms, c) for l, ms, c in trt_results if c < 0.5]
    if broken_engines:
        print(f"\n  {len(broken_engines)} TRT engine(s) BROKEN (FP16 accumulation error).")
    if ok_engines:
        best = min(ok_engines, key=lambda x: x[1])
        if best[1] < compile_ms:
            print(f"  Best correct TRT: {best[0]} ({best[1]:.0f}ms, cos={best[2]:.4f})")
            print(f"  → {fp32_ms/best[1]:.1f}x faster than FP32, "
                  f"{compile_ms/best[1]:.1f}x faster than torch.compile")
        else:
            print(f"  Best correct TRT ({best[0]}, {best[1]:.0f}ms) is slower "
                  f"than torch.compile ({compile_ms:.0f}ms).")
            print(f"  → Use torch.compile for this GPU.")
    elif trt_results and not ok_engines:
        print(f"  No correct TRT engines found. → Use torch.compile ({compile_ms:.0f}ms).")


if __name__ == "__main__":
    main()
