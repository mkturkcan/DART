#!/usr/bin/env python3
"""Automatic dataset labeling with SAM3/DART.

Labels images with user-provided class names and saves results in YOLO
or COCO format. Uses TRT FP16 backbone + 16-class enc-dec with presence
tokens for fast inference. Automatically builds TRT engines if not found.

Options:
  --images-dir PATH       Directory of images or a single image file.
  --classes NAME [NAME..] Class names to detect. Order determines class IDs
                          (0-indexed). Any text prompt works (open-vocabulary).
  --checkpoint PATH       SAM3 checkpoint. Pass sam3.pt to auto-download.
  --output-dir, -o PATH   Where to write labels. Default: labels/ next to images.
  --format {yolo,coco}    Output format (default: yolo).
      yolo: One .txt per image. Each line: class_id x_center y_center w h
            (normalized to [0,1]). Plus classes.txt for the index mapping.
      coco: Single annotations.json with images, annotations, categories.
            Bounding boxes in absolute [x, y, w, h] pixels.
  --confidence FLOAT      Minimum detection confidence (default: 0.3).
  --nms FLOAT             Per-class NMS IoU threshold (default: 0.7).
  --merge-nms GROUPS      Cross-class NMS for overlapping similar classes.
                          Suppresses duplicate boxes between classes that
                          describe the same object (e.g. car vs truck).
                          Format: semicolon-separated groups of comma-separated
                          class names. Classes not in any group keep independent
                          per-class NMS.
                            "car,truck,bus"             one merge group
                            "car,truck,bus;cat,dog"     two merge groups
                            "all"                       merge everything
  --merge-nms-thresh FLOAT  IoU threshold for cross-class merge NMS
                          (default: 0.5). Typically stricter than --nms since
                          cross-class overlaps are more likely to be duplicates.
  --imgsz INT             Input resolution, must be divisible by 14
                          (default: 1008). Lower values (644, 868) are faster
                          but less accurate.
  --trt-backbone PATH     Pre-built backbone TRT engine. If omitted, the
                          engine is built automatically on first run (~5 min).
  --trt-enc-dec PATH      Pre-built encoder-decoder TRT engine. If omitted,
                          built automatically on first run (~3 min).

Usage:
    # Label a directory of images (YOLO format)
    PYTHONIOENCODING=utf-8 python scripts/auto_label.py \
        --images-dir /path/to/images \
        --classes person car bicycle dog \
        --checkpoint sam3.pt

    # COCO format output to a specific folder
    PYTHONIOENCODING=utf-8 python scripts/auto_label.py \
        --images-dir /path/to/images \
        --classes person car bicycle dog \
        --checkpoint sam3.pt --format coco -o /path/to/output

    # Merge NMS for vehicle classes, keep others independent
    PYTHONIOENCODING=utf-8 python scripts/auto_label.py \
        --images-dir /path/to/images \
        --classes person car truck bus bicycle \
        --checkpoint sam3.pt \
        --merge-nms "car,truck,bus"

    # Use existing TRT engines
    PYTHONIOENCODING=utf-8 python scripts/auto_label.py \
        --images-dir /path/to/images \
        --classes person car bicycle dog \
        --checkpoint sam3.pt \
        --trt-backbone hf_backbone_1008_fp16.engine \
        --trt-enc-dec enc_dec_1008_c16_presence_fp16_opt5.engine
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from torchvision.ops import nms as torchvision_nms

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# Default engine paths
DEFAULT_BACKBONE_ENGINE = "hf_backbone_1008_fp16.engine"
DEFAULT_ENC_DEC_ENGINE = "enc_dec_1008_c16_presence_fp16.engine"
DEFAULT_BACKBONE_ONNX_DIR = "onnx_hf_backbone_1008"
DEFAULT_ENC_DEC_ONNX = "enc_dec_1008_c16_presence.onnx"


def find_images(path):
    """Find all images in a directory or return a single image path."""
    p = Path(path)
    if p.is_file():
        if p.suffix.lower() in IMAGE_EXTENSIONS:
            return [p]
        else:
            print(f"ERROR: {p} is not a recognized image format.")
            print(f"  Supported: {', '.join(sorted(IMAGE_EXTENSIONS))}")
            sys.exit(1)
    elif p.is_dir():
        images = sorted(
            f for f in p.iterdir()
            if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not images:
            print(f"ERROR: No images found in {p}")
            print(f"  Supported formats: {', '.join(sorted(IMAGE_EXTENSIONS))}")
            sys.exit(1)
        return images
    else:
        print(f"ERROR: {p} does not exist.")
        sys.exit(1)


def check_checkpoint(checkpoint_path):
    """Verify checkpoint exists and give helpful error if not."""
    if checkpoint_path is None:
        print("ERROR: --checkpoint is required.")
        print("  The SAM3 checkpoint (sam3.pt) auto-downloads from HuggingFace")
        print("  on first use if you pass --checkpoint sam3.pt")
        sys.exit(1)
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Checkpoint not found: {checkpoint_path}")
        print("  Download from HuggingFace or pass a valid path.")
        print("  The checkpoint auto-downloads if you use --checkpoint sam3.pt")
        sys.exit(1)


def build_backbone_engine(checkpoint_path, imgsz):
    """Export and build the backbone TRT engine if missing."""
    onnx_dir = DEFAULT_BACKBONE_ONNX_DIR
    onnx_path = os.path.join(onnx_dir, "hf_backbone.onnx")
    engine_path = DEFAULT_BACKBONE_ENGINE

    if os.path.exists(engine_path):
        return engine_path

    print(f"\n{'='*60}")
    print(f"  Backbone TRT engine not found: {engine_path}")
    print(f"  Building automatically (one-time, takes ~5 min)...")
    print(f"{'='*60}\n")

    # Step 1: Export ONNX via HF path
    if not os.path.exists(onnx_path):
        print("Step 1/2: Exporting backbone to ONNX...")
        cmd = [
            sys.executable, "scripts/export_hf_backbone.py",
            "--image", "x.jpg",
            "--imgsz", str(imgsz),
            "--output-onnx", onnx_path,
            "--output-engine", engine_path,
        ]
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        result = subprocess.run(cmd, env=env)
        if result.returncode != 0:
            print("\nERROR: Backbone export failed.")
            print("  Try running manually:")
            print(f"  PYTHONIOENCODING=utf-8 python scripts/export_hf_backbone.py "
                  f"--image x.jpg --imgsz {imgsz}")
            sys.exit(1)
        if os.path.exists(engine_path):
            return engine_path

    # Step 2: Build TRT engine from ONNX
    print("Step 2/2: Building TRT FP16 engine from ONNX...")
    cmd = [
        sys.executable, "-m", "sam3.trt.build_engine",
        "--onnx", onnx_path,
        "--output", engine_path,
        "--fp16", "--mixed-precision", "none",
    ]
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    result = subprocess.run(cmd, env=env)
    if result.returncode != 0:
        print("\nERROR: TRT engine build failed.")
        print("  Possible causes:")
        print("  - TensorRT not installed (pip install tensorrt)")
        print("  - Insufficient GPU memory (need ~4GB free)")
        print("  Try running manually:")
        print(f"  python -m sam3.trt.build_engine --onnx {onnx_path} "
              f"--output {engine_path} --fp16 --mixed-precision none")
        sys.exit(1)

    if not os.path.exists(engine_path):
        print(f"\nERROR: Engine was not created at {engine_path}")
        sys.exit(1)

    print(f"  Backbone engine built: {engine_path}")
    return engine_path


def build_enc_dec_engine(checkpoint_path, imgsz, max_classes=16):
    """Export and build the encoder-decoder TRT engine if missing."""
    onnx_path = DEFAULT_ENC_DEC_ONNX
    engine_path = DEFAULT_ENC_DEC_ENGINE

    if os.path.exists(engine_path):
        return engine_path

    print(f"\n{'='*60}")
    print(f"  Enc-dec TRT engine not found: {engine_path}")
    print(f"  Building automatically (one-time, takes ~3 min)...")
    print(f"{'='*60}\n")

    # Step 1: Export ONNX
    if not os.path.exists(onnx_path):
        print("Step 1/2: Exporting encoder-decoder to ONNX...")
        cmd = [
            sys.executable, "-m", "sam3.trt.export_enc_dec",
            "--checkpoint", checkpoint_path,
            "--output", onnx_path,
            "--max-classes", str(max_classes),
            "--imgsz", str(imgsz),
            "--presence",
        ]
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        result = subprocess.run(cmd, env=env)
        if result.returncode != 0:
            print("\nERROR: Encoder-decoder ONNX export failed.")
            print("  Try running manually:")
            print(f"  PYTHONIOENCODING=utf-8 python -m sam3.trt.export_enc_dec "
                  f"--checkpoint {checkpoint_path} --output {onnx_path} "
                  f"--max-classes {max_classes} --imgsz {imgsz} --presence")
            sys.exit(1)

    # Step 2: Build TRT engine
    print("Step 2/2: Building TRT FP16 engine from ONNX...")
    cmd = [
        sys.executable, "-m", "sam3.trt.build_engine",
        "--onnx", onnx_path,
        "--output", engine_path,
        "--fp16", "--mixed-precision", "none",
    ]
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    result = subprocess.run(cmd, env=env)
    if result.returncode != 0:
        print("\nERROR: Enc-dec TRT engine build failed.")
        print("  Possible causes:")
        print("  - TensorRT not installed (pip install tensorrt)")
        print("  - Insufficient GPU memory (16-class engine needs ~8GB free)")
        print("  - Try reducing --max-classes to 4 or 8")
        print("  Manual command:")
        print(f"  python -m sam3.trt.build_engine --onnx {onnx_path} "
              f"--output {engine_path} --fp16 --mixed-precision none")
        sys.exit(1)

    if not os.path.exists(engine_path):
        print(f"\nERROR: Engine was not created at {engine_path}")
        sys.exit(1)

    print(f"  Enc-dec engine built: {engine_path}")
    return engine_path


def parse_merge_groups(merge_spec, class_names):
    """Parse --merge-nms spec into a list of class-id sets.

    Args:
        merge_spec: "all" or semicolon-separated groups like "car,truck,bus;cat,dog"
        class_names: Ordered list of class names.

    Returns:
        List of sets, each containing class indices that share NMS.
        Classes not in any group are left independent (no cross-class NMS).
    """
    name_to_id = {name: i for i, name in enumerate(class_names)}

    if merge_spec.strip().lower() == "all":
        return [set(range(len(class_names)))]

    groups = []
    for group_str in merge_spec.split(";"):
        group_str = group_str.strip()
        if not group_str:
            continue
        names = [n.strip() for n in group_str.split(",") if n.strip()]
        ids = set()
        for name in names:
            if name not in name_to_id:
                print(f"ERROR: Unknown class '{name}' in --merge-nms")
                print(f"  Available classes: {class_names}")
                sys.exit(1)
            ids.add(name_to_id[name])
        if len(ids) >= 2:
            groups.append(ids)
        elif len(ids) == 1:
            print(f"WARNING: Merge group '{group_str}' has only one class, skipping.")

    return groups


def apply_merge_nms(results, merge_groups, nms_threshold):
    """Apply cross-class NMS within each merge group.

    For each merge group, run NMS on the union of all detections from those
    classes. Detections from classes not in any group are kept as-is.
    """
    boxes = results["boxes"]       # (N, 4) xyxy
    scores = results["scores"]     # (N,)
    class_ids = results["class_ids"]  # (N,)
    class_names = results["class_names"]  # list of N strings

    if len(scores) == 0:
        return results

    # Track which detections to keep
    keep_mask = torch.ones(len(scores), dtype=torch.bool, device=scores.device)

    # Find which detections belong to any merge group
    merged_ids = set()
    for group in merge_groups:
        merged_ids.update(group)

    for group in merge_groups:
        # Select detections belonging to this group
        in_group = torch.tensor(
            [int(class_ids[i].item()) in group for i in range(len(class_ids))],
            dtype=torch.bool, device=scores.device,
        )
        if in_group.sum() < 2:
            continue

        group_indices = torch.where(in_group)[0]
        group_boxes = boxes[group_indices]
        group_scores = scores[group_indices]

        # Run class-agnostic NMS within this group
        nms_keep = torchvision_nms(group_boxes, group_scores, nms_threshold)
        suppressed = torch.ones(len(group_indices), dtype=torch.bool, device=scores.device)
        suppressed[nms_keep] = False
        # Mark suppressed detections
        keep_mask[group_indices[suppressed]] = False

    # Filter results
    keep_idx = torch.where(keep_mask)[0]
    return {
        "boxes": boxes[keep_idx],
        "scores": scores[keep_idx],
        "class_ids": class_ids[keep_idx],
        "class_names": [class_names[i] for i in keep_idx.cpu().tolist()],
    }


def results_to_yolo(results, img_width, img_height):
    """Convert detection results to YOLO format lines.

    YOLO format: class_id x_center y_center width height
    All values normalized to [0, 1].
    """
    lines = []
    boxes = results["boxes"]
    class_ids = results["class_ids"]
    scores = results["scores"]

    for i in range(len(scores)):
        x1, y1, x2, y2 = boxes[i].cpu().tolist()
        cls_id = int(class_ids[i].item())

        x_center = (x1 + x2) / 2.0 / img_width
        y_center = (y1 + y2) / 2.0 / img_height
        w = (x2 - x1) / img_width
        h = (y2 - y1) / img_height

        x_center = max(0.0, min(1.0, x_center))
        y_center = max(0.0, min(1.0, y_center))
        w = max(0.0, min(1.0, w))
        h = max(0.0, min(1.0, h))

        lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")

    return lines


def results_to_coco_anns(results, image_id, ann_id_start):
    """Convert detection results to COCO annotation dicts.

    Returns list of COCO annotation dicts and the next available ann_id.
    """
    anns = []
    boxes = results["boxes"]
    class_ids = results["class_ids"]
    scores = results["scores"]

    for i in range(len(scores)):
        x1, y1, x2, y2 = boxes[i].cpu().tolist()
        w = x2 - x1
        h = y2 - y1
        anns.append({
            "id": ann_id_start + i,
            "image_id": image_id,
            "category_id": int(class_ids[i].item()),
            "bbox": [round(x1, 2), round(y1, 2), round(w, 2), round(h, 2)],
            "area": round(w * h, 2),
            "score": round(float(scores[i].item()), 4),
            "iscrowd": 0,
        })

    return anns, ann_id_start + len(scores)


@torch.inference_mode()
def run_labeling(args):
    """Main labeling loop."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: No CUDA GPU detected. Inference will be very slow.")
        print("  TRT engines require a CUDA GPU. Falling back to PyTorch CPU.")

    images = find_images(args.images_dir)
    print(f"Found {len(images)} images to label")

    check_checkpoint(args.checkpoint)

    # Resolve TRT engines
    use_trt = device == "cuda"
    backbone_engine = args.trt_backbone
    enc_dec_engine = args.trt_enc_dec

    if use_trt:
        if backbone_engine is None:
            backbone_engine = build_backbone_engine(args.checkpoint, args.imgsz)
        elif not os.path.exists(backbone_engine):
            print(f"ERROR: Backbone engine not found: {backbone_engine}")
            print(f"  Remove --trt-backbone to auto-build, or provide a valid path.")
            sys.exit(1)

        if enc_dec_engine is None:
            enc_dec_engine = build_enc_dec_engine(args.checkpoint, args.imgsz)
        elif not os.path.exists(enc_dec_engine):
            print(f"ERROR: Enc-dec engine not found: {enc_dec_engine}")
            print(f"  Remove --trt-enc-dec to auto-build, or provide a valid path.")
            sys.exit(1)

    # Parse merge groups
    classes = args.classes
    merge_groups = []
    if args.merge_nms:
        merge_groups = parse_merge_groups(args.merge_nms, classes)
        group_strs = []
        for g in merge_groups:
            group_strs.append("{" + ", ".join(classes[i] for i in sorted(g)) + "}")
        print(f"Merge NMS groups: {', '.join(group_strs)}")

    # Load model
    print(f"\nLoading SAM3 model from {args.checkpoint} ...")
    from sam3.model_builder import build_sam3_image_model
    model = build_sam3_image_model(
        device=device, checkpoint_path=args.checkpoint, eval_mode=True,
    )

    from sam3.model.sam3_multiclass_fast import Sam3MultiClassPredictorFast
    predictor = Sam3MultiClassPredictorFast(
        model, device=device,
        resolution=args.imgsz,
        trt_engine_path=backbone_engine if use_trt else None,
        use_fp16=use_trt,
        detection_only=True,
        trt_enc_dec_engine_path=enc_dec_engine if use_trt else None,
        trt_max_classes=16,
    )

    print(f"Classes ({len(classes)}): {classes}")
    if len(classes) > 16:
        print(f"NOTE: {len(classes)} classes > 16. Enc-dec will run in batches of 16.")
    predictor.set_classes(classes)

    # Warmup
    print("Warming up...")
    dummy = Image.new("RGB", (args.imgsz, args.imgsz))
    state = predictor.set_image(dummy)
    predictor.predict(state, confidence_threshold=0.5)
    if device == "cuda":
        torch.cuda.synchronize()

    # Setup output directory
    output_dir = Path(args.output_dir) if args.output_dir else None
    if output_dir is None:
        if Path(args.images_dir).is_dir():
            output_dir = Path(args.images_dir).parent / "labels"
        else:
            output_dir = Path(args.images_dir).parent / "labels"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write classes.txt (useful for both formats)
    classes_file = output_dir / "classes.txt"
    classes_file.write_text("\n".join(classes) + "\n")
    print(f"Class list saved to {classes_file}")

    is_coco = args.format == "coco"

    # COCO format state
    coco_images = []
    coco_annotations = []
    ann_id = 1

    # Label images
    fmt_label = "COCO" if is_coco else "YOLO"
    print(f"\nLabeling {len(images)} images -> {output_dir}/ ({fmt_label} format)")
    print(f"  Confidence threshold: {args.confidence}")
    print(f"  NMS threshold: {args.nms}")
    if merge_groups:
        print(f"  Merge NMS threshold: {args.merge_nms_thresh}")
    print()

    total_dets = 0
    t_start = time.perf_counter()

    for idx, img_path in enumerate(images):
        img = Image.open(img_path).convert("RGB")
        img_w, img_h = img.size

        state = predictor.set_image(img)
        results = predictor.predict(
            state,
            confidence_threshold=args.confidence,
            nms_threshold=args.nms,
        )

        # Apply merge NMS if requested
        if merge_groups and len(results["scores"]) > 0:
            results = apply_merge_nms(results, merge_groups, args.merge_nms_thresh)

        n_dets = len(results["scores"])
        total_dets += n_dets

        if is_coco:
            image_id = idx + 1
            coco_images.append({
                "id": image_id,
                "file_name": img_path.name,
                "width": img_w,
                "height": img_h,
            })
            anns, ann_id = results_to_coco_anns(results, image_id, ann_id)
            coco_annotations.extend(anns)
        else:
            yolo_lines = results_to_yolo(results, img_w, img_h)
            label_path = output_dir / (img_path.stem + ".txt")
            label_path.write_text("\n".join(yolo_lines) + "\n" if yolo_lines else "")

        if (idx + 1) % 50 == 0 or idx == 0 or idx == len(images) - 1:
            elapsed = time.perf_counter() - t_start
            fps = (idx + 1) / elapsed if elapsed > 0 else 0
            print(f"  [{idx+1}/{len(images)}] {img_path.name}: "
                  f"{n_dets} dets, {fps:.1f} img/s")

    # Save COCO JSON
    if is_coco:
        coco_output = {
            "images": coco_images,
            "annotations": coco_annotations,
            "categories": [
                {"id": i, "name": name} for i, name in enumerate(classes)
            ],
        }
        coco_path = output_dir / "annotations.json"
        with open(coco_path, "w") as f:
            json.dump(coco_output, f, indent=2)
        print(f"\n  COCO annotations saved to {coco_path}")

    elapsed = time.perf_counter() - t_start
    fps = len(images) / elapsed if elapsed > 0 else 0

    print(f"\nDone. Labeled {len(images)} images in {elapsed:.1f}s ({fps:.1f} img/s)")
    print(f"  Total detections: {total_dets}")
    print(f"  Avg detections/image: {total_dets / len(images):.1f}")
    print(f"  Labels saved to: {output_dir}/")
    print(f"  Class mapping: {classes_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Auto-label images with SAM3/DART",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Label a folder (YOLO format, default)
  PYTHONIOENCODING=utf-8 python scripts/auto_label.py \\
      --images-dir /path/to/images --classes person car dog \\
      --checkpoint sam3.pt

  # COCO format
  PYTHONIOENCODING=utf-8 python scripts/auto_label.py \\
      --images-dir /path/to/images --classes person car dog \\
      --checkpoint sam3.pt --format coco --output-dir /path/to/output

  # Merge NMS for similar classes
  PYTHONIOENCODING=utf-8 python scripts/auto_label.py \\
      --images-dir /path/to/images \\
      --classes person car truck bus bicycle motorcycle \\
      --checkpoint sam3.pt \\
      --merge-nms "car,truck,bus;bicycle,motorcycle"

  # Merge all classes (fully class-agnostic NMS)
  PYTHONIOENCODING=utf-8 python scripts/auto_label.py \\
      --images-dir /path/to/images --classes person car dog \\
      --checkpoint sam3.pt --merge-nms all

Output (YOLO):
  One .txt per image:  class_id  x_center  y_center  width  height
  Plus classes.txt mapping indices to names.

Output (COCO):
  Single annotations.json with images, annotations, categories.
  Bounding boxes in absolute [x, y, w, h] format.
        """,
    )
    parser.add_argument(
        "--images-dir", required=True,
        help="Path to directory of images, or a single image file",
    )
    parser.add_argument(
        "--classes", nargs="+", required=True,
        help="Class names to detect (e.g. --classes person car bicycle)",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="SAM3 checkpoint path (default: auto-download sam3.pt)",
    )
    parser.add_argument(
        "--output-dir", "-o", type=str, default=None,
        help="Output directory for labels (default: labels/ next to images)",
    )
    parser.add_argument(
        "--format", type=str, default="yolo", choices=["yolo", "coco"],
        help="Label format: yolo (per-image .txt) or coco (single .json)",
    )
    parser.add_argument(
        "--confidence", type=float, default=0.3,
        help="Confidence threshold (default: 0.3)",
    )
    parser.add_argument(
        "--nms", type=float, default=0.7,
        help="Per-class NMS IoU threshold (default: 0.7)",
    )
    parser.add_argument(
        "--merge-nms", type=str, default=None, metavar="GROUPS",
        help='Cross-class NMS groups. Use "all" for fully class-agnostic, '
             'or semicolon-separated groups: "car,truck,bus;cat,dog"',
    )
    parser.add_argument(
        "--merge-nms-thresh", type=float, default=0.5,
        help="IoU threshold for cross-class merge NMS (default: 0.5)",
    )
    parser.add_argument(
        "--imgsz", type=int, default=1008,
        help="Input resolution, must be divisible by 14 (default: 1008)",
    )
    parser.add_argument(
        "--trt-backbone", type=str, default=None,
        help="Path to pre-built backbone TRT engine (auto-built if omitted)",
    )
    parser.add_argument(
        "--trt-enc-dec", type=str, default=None,
        help="Path to pre-built enc-dec TRT engine (auto-built if omitted)",
    )
    args = parser.parse_args()

    if args.imgsz % 14 != 0:
        print(f"ERROR: --imgsz must be divisible by 14, got {args.imgsz}")
        print(f"  Common values: 644, 868, 1008")
        sys.exit(1)

    if not args.classes:
        print("ERROR: --classes must specify at least one class name")
        sys.exit(1)

    run_labeling(args)


if __name__ == "__main__":
    main()
