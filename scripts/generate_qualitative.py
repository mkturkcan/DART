#!/usr/bin/env python3
"""Generate qualitative detection figures for the paper.

Randomly selects COCO val2017 images (seeded), runs DART multi-class detection,
and saves annotated images + a combined grid for LaTeX inclusion.

Usage:
    python scripts/generate_qualitative.py --seed 42
    python scripts/generate_qualitative.py --seed 7 --n-images 4
    python scripts/generate_qualitative.py --seed 42 --classes person car dog bicycle
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── Defaults ──────────────────────────────────────────────────────────────
IMAGES_DIR = "D:/val2017"
ANN_FILE = "D:/coco2017labels/coco/annotations/instances_val2017.json"
CHECKPOINT = "sam3.pt"
OUTPUT_DIR = "paper/figures"
RESOLUTION = 1008

# Classes to detect — diverse and visually interesting
DEFAULT_CLASSES = ["person", "car", "dog", "bicycle", "chair", "cat"]

# Minimum annotation criteria for "interesting" images
MIN_CATEGORIES = 3   # image must have objects from >= 3 of our target classes
MIN_OBJECTS = 5       # image must have >= 5 annotated objects total

# ── Color palette (matches paper style, high contrast on photos) ──────────
COLORS = [
    (230,  25,  75),  # red       — person
    (  0, 130, 200),  # blue      — car
    (245, 130,  48),  # orange    — dog
    ( 60, 180,  75),  # green     — bicycle
    (145,  30, 180),  # purple    — chair
    ( 70, 240, 240),  # cyan      — cat
    (255, 225,  25),  # yellow
    (240,  50, 230),  # magenta
]


def load_coco_annotations(ann_file: str) -> dict:
    """Load COCO annotations and build image→categories mapping."""
    with open(ann_file) as f:
        coco = json.load(f)

    # category id → name
    cat_id_to_name = {c["id"]: c["name"] for c in coco["categories"]}

    # image id → {filename, categories, n_objects}
    img_info = {}
    for img in coco["images"]:
        img_info[img["id"]] = {
            "file_name": img["file_name"],
            "width": img["width"],
            "height": img["height"],
            "categories": set(),
            "n_objects": 0,
        }

    for ann in coco["annotations"]:
        img_id = ann["image_id"]
        if img_id in img_info:
            cat_name = cat_id_to_name.get(ann["category_id"], "")
            img_info[img_id]["categories"].add(cat_name)
            img_info[img_id]["n_objects"] += 1

    return img_info


def find_interesting_images(
    img_info: dict,
    target_classes: List[str],
    min_categories: int = 3,
    min_objects: int = 5,
    landscape_only: bool = True,
) -> List[dict]:
    """Filter images that contain multiple target classes."""
    candidates = []
    target_set = set(target_classes)

    for img_id, info in img_info.items():
        overlap = info["categories"] & target_set
        if len(overlap) < min_categories or info["n_objects"] < min_objects:
            continue
        if landscape_only and info["height"] >= info["width"]:
            continue
        candidates.append({
            "img_id": img_id,
            "file_name": info["file_name"],
            "width": info["width"],
            "height": info["height"],
            "n_target_classes": len(overlap),
            "target_classes": sorted(overlap),
            "n_objects": info["n_objects"],
        })

    # Sort by diversity then object count (most diverse first)
    candidates.sort(key=lambda x: (-x["n_target_classes"], -x["n_objects"]))
    return candidates


def cross_class_nms(results: Dict, iou_threshold: float = 0.5) -> Dict:
    """Apply NMS across all classes to remove duplicate detections of the same object."""
    if len(results["scores"]) == 0:
        return results

    boxes = results["boxes"]  # (N, 4) tensor
    scores = results["scores"]  # (N,) tensor

    from torchvision.ops import nms
    keep = nms(boxes, scores, iou_threshold)
    keep = keep.sort().values  # preserve original order

    filtered = {
        "boxes": boxes[keep],
        "scores": scores[keep],
        "class_names": [results["class_names"][i] for i in keep.cpu().tolist()],
        "masks": results["masks"][keep] if results.get("masks") is not None else None,
    }
    return filtered


def annotate_detections(
    image: Image.Image,
    results: Dict,
    class_names: List[str],
    box_width: int = 4,
    font_size: int = 0,
    min_score: float = 0.3,
) -> Image.Image:
    """Draw bounding boxes and labels on image. No masks — detection only."""
    img = image.convert("RGB").copy()
    w, h = img.size

    if font_size <= 0:
        font_size = max(16, int(h / 40))

    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except (OSError, IOError):
        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size
            )
        except (OSError, IOError):
            font = ImageFont.load_default()

    # Map class → stable color
    color_map = {name: COLORS[i % len(COLORS)] for i, name in enumerate(class_names)}

    draw = ImageDraw.Draw(img)
    num_dets = len(results["scores"])

    for i in range(num_dets):
        score = results["scores"][i].item()
        if score < min_score:
            continue

        cls_name = results["class_names"][i]
        box = results["boxes"][i].cpu().tolist()
        color = color_map.get(cls_name, COLORS[0])

        x1, y1, x2, y2 = box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=box_width)

        label = f"{cls_name} {score:.2f}"
        bbox = draw.textbbox((0, 0), label, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]

        # Label background
        lx, ly = x1, max(y1 - th - 6, 0)
        draw.rectangle([lx, ly, lx + tw + 8, ly + th + 6], fill=color)
        draw.text((lx + 4, ly + 2), label, fill=(255, 255, 255), font=font)

    return img


def create_grid(images: List[Image.Image], n_cols: int = 3, pad: int = 8) -> Image.Image:
    """Combine images into a grid with uniform sizing and white padding."""
    n = len(images)
    n_rows = (n + n_cols - 1) // n_cols

    # Resize all to the same height, preserving aspect ratio
    target_h = min(img.size[1] for img in images)
    resized = []
    for img in images:
        w, h = img.size
        new_w = int(w * target_h / h)
        resized.append(img.resize((new_w, target_h), Image.LANCZOS))

    # Make all same width (max width after resize)
    max_w = max(img.size[0] for img in resized)
    cell_w = max_w
    cell_h = target_h

    grid_w = n_cols * cell_w + (n_cols + 1) * pad
    grid_h = n_rows * cell_h + (n_rows + 1) * pad
    grid = Image.new("RGB", (grid_w, grid_h), (255, 255, 255))

    for idx, img in enumerate(resized):
        row, col = divmod(idx, n_cols)
        # Center image in cell
        x_offset = col * (cell_w + pad) + pad + (cell_w - img.size[0]) // 2
        y_offset = row * (cell_h + pad) + pad
        grid.paste(img, (x_offset, y_offset))

    return grid


def main():
    parser = argparse.ArgumentParser(
        description="Generate qualitative detection figures for the paper"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for image selection")
    parser.add_argument("--n-images", type=int, default=3, help="Number of images to generate")
    parser.add_argument("--classes", nargs="+", default=DEFAULT_CLASSES, help="Target classes")
    parser.add_argument("--images-dir", default=IMAGES_DIR, help="COCO val2017 images dir")
    parser.add_argument("--ann-file", default=ANN_FILE, help="COCO annotations JSON")
    parser.add_argument("--checkpoint", default=CHECKPOINT, help="SAM3 checkpoint path")
    parser.add_argument("--output-dir", default=OUTPUT_DIR, help="Output directory")
    parser.add_argument("--imgsz", type=int, default=RESOLUTION, help="Inference resolution")
    parser.add_argument("--confidence", type=float, default=0.3, help="Confidence threshold")
    parser.add_argument("--trt", default=None, help="TRT backbone engine (optional)")
    parser.add_argument(
        "--trt-enc-dec", default=None, help="TRT enc-dec engine (optional)"
    )
    parser.add_argument("--trt-max-classes", type=int, default=16)
    parser.add_argument("--min-categories", type=int, default=MIN_CATEGORIES,
                        help="Min target classes per image")
    parser.add_argument("--min-objects", type=int, default=MIN_OBJECTS,
                        help="Min annotated objects per image")
    parser.add_argument("--grid-cols", type=int, default=3, help="Grid columns")
    parser.add_argument("--images", nargs="+", default=None,
                        help="Override: specific image filenames to use (skip random selection)")
    parser.add_argument("--list-candidates", action="store_true",
                        help="Just list candidate images without running inference")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── 1. Find interesting images ────────────────────────────────────────
    print(f"Loading COCO annotations from {args.ann_file}...")
    img_info = load_coco_annotations(args.ann_file)
    print(f"  {len(img_info)} images in annotations")

    candidates = find_interesting_images(
        img_info, args.classes, args.min_categories, args.min_objects
    )
    print(f"  {len(candidates)} images have >= {args.min_categories} target classes "
          f"and >= {args.min_objects} objects")

    if not candidates:
        print("No suitable images found. Try lowering --min-categories or --min-objects.")
        return

    # ── 2. Select images ─────────────────────────────────────────────────
    if args.images:
        # Manual override: use specified filenames
        selected = []
        all_by_name = {c["file_name"]: c for c in candidates}
        # Also allow images not in candidate list
        for fname in args.images:
            if fname in all_by_name:
                selected.append(all_by_name[fname])
            else:
                selected.append({
                    "file_name": fname,
                    "n_target_classes": 0,
                    "target_classes": [],
                    "n_objects": 0,
                })
        print(f"\nManual selection: {len(selected)} images")
    else:
        rng = random.Random(args.seed)
        selected = rng.sample(candidates, min(args.n_images, len(candidates)))
        print(f"\nSeed={args.seed}, selected {len(selected)} images:")

    for s in selected:
        print(f"  {s['file_name']}  ({s['n_target_classes']} classes: "
              f"{', '.join(s['target_classes'])}; {s['n_objects']} objects)")

    if args.list_candidates:
        print(f"\nTop 20 candidates:")
        for c in candidates[:20]:
            print(f"  {c['file_name']}  ({c['n_target_classes']} cls: "
                  f"{', '.join(c['target_classes'])}; {c['n_objects']} obj)")
        return

    # ── 3. Load model ─────────────────────────────────────────────────────
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_multiclass_fast import Sam3MultiClassPredictorFast

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nLoading SAM3 model on {device}...")

    model = build_sam3_image_model(
        device=device,
        checkpoint_path=args.checkpoint,
        eval_mode=True,
    )

    predictor = Sam3MultiClassPredictorFast(
        model,
        device=device,
        resolution=args.imgsz,
        use_fp16=True,
        detection_only=True,
        trt_engine_path=args.trt,
        trt_enc_dec_engine_path=args.trt_enc_dec,
        trt_max_classes=args.trt_max_classes,
    )

    print(f"Setting {len(args.classes)} classes: {args.classes}")
    predictor.set_classes(args.classes)

    # Warmup
    print("Warmup pass...")
    dummy_img = Image.new("RGB", (640, 480), (128, 128, 128))
    state = predictor.set_image(dummy_img)
    predictor.predict(state, confidence_threshold=0.5)
    if device == "cuda":
        torch.cuda.synchronize()

    # ── 4. Run inference and annotate ─────────────────────────────────────
    annotated_images = []
    for i, sel in enumerate(selected):
        img_path = os.path.join(args.images_dir, sel["file_name"])
        print(f"\n[{i+1}/{len(selected)}] Processing {sel['file_name']}...")

        image = Image.open(img_path).convert("RGB")

        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        state = predictor.set_image(image)
        results = predictor.predict(
            state,
            confidence_threshold=args.confidence,
            nms_threshold=0.7,
        )
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        # Cross-class NMS to remove duplicate detections of the same object
        n_before = len(results["scores"])
        results = cross_class_nms(results, iou_threshold=0.5)
        n_dets = len(results["scores"])
        if n_before != n_dets:
            print(f"  {n_before} detections → {n_dets} after cross-class NMS")
        else:
            print(f"  {n_dets} detections in {elapsed*1000:.1f}ms")

        # Show per-class counts
        from collections import Counter
        cls_counts = Counter(results["class_names"])
        for cls, cnt in cls_counts.most_common():
            print(f"    {cls}: {cnt}")

        annotated = annotate_detections(
            image, results, args.classes,
            min_score=args.confidence,
        )

        # Save individual image
        out_path = os.path.join(args.output_dir, f"qual_{i+1}.jpg")
        annotated.save(out_path, quality=95)
        print(f"  Saved {out_path}")
        annotated_images.append(annotated)

    # ── 5. Create combined grid ───────────────────────────────────────────
    if annotated_images:
        grid = create_grid(annotated_images, n_cols=args.grid_cols)
        grid_path = os.path.join(args.output_dir, "qualitative_grid.jpg")
        grid.save(grid_path, quality=95)
        print(f"\nCombined grid saved to {grid_path}")
        print(f"  Grid size: {grid.size[0]}x{grid.size[1]}")

    print(f"\nDone! Re-run with --seed <N> to get different images.")


if __name__ == "__main__":
    main()
