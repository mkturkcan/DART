#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""Build a TRT FP16 enc-dec engine for the standard 80 COCO classes.

One-stop script that runs the full pipeline:
  1. Export encoder+decoder+scoring to ONNX
  2. Build TRT FP16 engine (with --mixed-precision none for enc-dec)
  3. Cache text embeddings for the 80 COCO classes

Outputs (default names):
  - enc_dec_coco.onnx       — ONNX model
  - enc_dec_coco_fp16.engine — TRT FP16 engine (GPU-specific)
  - text_cache_coco.pt      — Cached text embeddings for all 80 classes

Usage:
    # Default: 80 classes, 644px, FP16
    python scripts/build_coco_engine.py --checkpoint sam3.pt

    # Custom resolution and output directory
    python scripts/build_coco_engine.py --checkpoint sam3.pt --imgsz 1008 --outdir engines/

    # Skip steps (e.g., ONNX already exported)
    python scripts/build_coco_engine.py --checkpoint sam3.pt --skip-onnx

    # Custom subset of COCO classes
    python scripts/build_coco_engine.py --checkpoint sam3.pt --classes person car bicycle
"""

import argparse
import os
import sys
import time

from sam3.coco_classes import COCO_CLASSES


def cache_text_embeddings(checkpoint, class_names, output_path, device="cuda"):
    """Compute and save text embeddings for the given class names."""
    import torch
    from sam3.model_builder import build_sam3_image_model

    print(f"\n{'='*60}")
    print(f"Step 3: Caching text embeddings ({len(class_names)} classes)")
    print(f"{'='*60}")

    print(f"Loading model from {checkpoint} ...")
    model = build_sam3_image_model(
        checkpoint_path=checkpoint,
        device=device,
        eval_mode=True,
        load_from_HF=False,
        enable_segmentation=False,
    )

    print(f"Computing text embeddings for {len(class_names)} classes ...")
    with torch.inference_mode():
        text_outputs = model.backbone.forward_text(class_names, device=device)

    torch.save({
        "class_names": list(class_names),
        "text": text_outputs["language_features"],
        "mask": text_outputs["language_mask"],
    }, output_path)

    print(f"Saved text cache: {output_path}")
    print(f"  Classes: {', '.join(class_names[:5])}{'...' if len(class_names) > 5 else ''}")
    print(f"  Text features shape: {text_outputs['language_features'].shape}")


def main():
    parser = argparse.ArgumentParser(
        description="Build TRT FP16 enc-dec engine for COCO classes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default: all 80 COCO classes, 644px
  python scripts/build_coco_engine.py --checkpoint sam3.pt

  # Higher resolution
  python scripts/build_coco_engine.py --checkpoint sam3.pt --imgsz 1008

  # Custom classes only
  python scripts/build_coco_engine.py --checkpoint sam3.pt --classes person car bicycle

  # Resume from existing ONNX
  python scripts/build_coco_engine.py --checkpoint sam3.pt --skip-onnx
""",
    )
    parser.add_argument(
        "--checkpoint", required=True,
        help="Path to SAM3 checkpoint (.pt)",
    )
    parser.add_argument(
        "--classes", nargs="+", default=None,
        help="Custom class names (default: all 80 COCO classes)",
    )
    parser.add_argument(
        "--imgsz", type=int, default=644,
        help="Input image resolution, must be divisible by 14 "
             "(default: 644 → 46x46 spatial features)",
    )
    parser.add_argument(
        "--outdir", type=str, default=".",
        help="Output directory for all generated files (default: current dir)",
    )
    parser.add_argument(
        "--prefix", type=str, default="enc_dec_coco",
        help="Filename prefix for outputs (default: enc_dec_coco)",
    )
    parser.add_argument(
        "--skip-onnx", action="store_true",
        help="Skip ONNX export (use existing ONNX file)",
    )
    parser.add_argument(
        "--skip-engine", action="store_true",
        help="Skip TRT engine build (only export ONNX + cache text)",
    )
    parser.add_argument(
        "--skip-text-cache", action="store_true",
        help="Skip text embedding caching",
    )
    parser.add_argument(
        "--opt-level", type=int, default=3, choices=[0, 1, 2, 3, 4, 5],
        help="TRT builder optimization level (default: 3). "
             "Try 0 if build fails with OOM.",
    )
    parser.add_argument(
        "--workspace", type=float, default=4.0,
        help="TRT workspace size in GB (default: 4.0)",
    )
    args = parser.parse_args()

    class_names = args.classes if args.classes else COCO_CLASSES
    max_classes = len(class_names)

    # Validate imgsz
    if args.imgsz % 14 != 0:
        print(f"ERROR: --imgsz must be divisible by 14, got {args.imgsz}")
        sys.exit(1)

    spatial = args.imgsz // 14

    # Output paths
    os.makedirs(args.outdir, exist_ok=True)
    onnx_path = os.path.join(args.outdir, f"{args.prefix}.onnx")
    engine_path = os.path.join(
        args.outdir,
        f"{args.prefix}_fp16_{max_classes}.engine",
    )
    text_cache_path = os.path.join(args.outdir, f"text_cache_coco.pt")

    print(f"SAM3 COCO Engine Builder")
    print(f"========================")
    print(f"  Checkpoint:    {args.checkpoint}")
    print(f"  Classes:       {max_classes} ({'COCO-80' if not args.classes else 'custom'})")
    print(f"  Resolution:    {args.imgsz}px → {spatial}x{spatial} spatial")
    print(f"  Output dir:    {args.outdir}")
    print(f"  ONNX:          {onnx_path}")
    print(f"  Engine:        {engine_path}")
    print(f"  Text cache:    {text_cache_path}")

    t_total = time.time()

    # Step 1: ONNX export
    if not args.skip_onnx:
        from sam3.trt.export_enc_dec import export_onnx

        print(f"\n{'='*60}")
        print(f"Step 1: Exporting ONNX (max_classes={max_classes}, imgsz={args.imgsz})")
        print(f"{'='*60}")

        t0 = time.time()
        export_onnx(
            checkpoint_path=args.checkpoint,
            output_path=onnx_path,
            max_classes=max_classes,
            imgsz=args.imgsz,
        )
        print(f"  ONNX export took {time.time() - t0:.1f}s")
    else:
        if not os.path.exists(onnx_path):
            print(f"ERROR: --skip-onnx specified but {onnx_path} does not exist")
            sys.exit(1)
        print(f"\nSkipping ONNX export (using existing: {onnx_path})")

    # Step 2: TRT engine build
    if not args.skip_engine:
        from sam3.trt.build_engine import build_engine

        print(f"\n{'='*60}")
        print(f"Step 2: Building TRT FP16 engine")
        print(f"{'='*60}")

        t0 = time.time()
        build_engine(
            onnx_path=onnx_path,
            output_path=engine_path,
            engine_type="enc-dec",
            fp16=True,
            mixed_precision="none",  # pure FP16 is fine for enc-dec
            max_classes=max_classes,
            opt_level=args.opt_level,
            workspace_gb=args.workspace,
        )
        print(f"  Engine build took {time.time() - t0:.1f}s")
    else:
        print(f"\nSkipping TRT engine build")

    # Step 3: Text embedding cache
    if not args.skip_text_cache:
        cache_text_embeddings(
            checkpoint=args.checkpoint,
            class_names=class_names,
            output_path=text_cache_path,
        )
    else:
        print(f"\nSkipping text cache")

    elapsed = time.time() - t_total
    print(f"\n{'='*60}")
    print(f"Done in {elapsed:.1f}s")
    print(f"{'='*60}")
    print(f"\nGenerated files:")
    for path in [onnx_path, engine_path, text_cache_path]:
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"  {path} ({size_mb:.1f} MB)")

    print(f"\nUsage example:")
    print(f"  # Single image")
    print(f"  python demo_multiclass.py \\")
    print(f"      --image x.jpg \\")
    print(f"      --classes person car bicycle \\")
    print(f"      --fast --detection-only \\")
    print(f"      --compile max-autotune \\")
    print(f"      --trt-enc-dec {engine_path} \\")
    print(f"      --text-cache {text_cache_path} \\")
    print(f"      --imgsz {args.imgsz} --warmup 3")
    print()
    print(f"  # Video")
    print(f"  python demo_video.py \\")
    print(f"      --video input.mp4 \\")
    print(f"      --classes person car bicycle \\")
    print(f"      --checkpoint {args.checkpoint} \\")
    print(f"      --compile max-autotune \\")
    print(f"      --trt-enc-dec {engine_path} \\")
    print(f"      --text-cache {text_cache_path} \\")
    print(f"      --imgsz {args.imgsz} --track")


if __name__ == "__main__":
    main()
