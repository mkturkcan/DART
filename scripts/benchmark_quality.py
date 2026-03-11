#!/usr/bin/env python3
"""
Quality + speed benchmark for SAM3 multi-class inference approaches.

Compares detection quality and speed of five approaches:
  1. Sequential per-class (ground truth baseline)
  2. Fast batched (encoder+decoder at bs=N)
  3. Shared-encoder + batched decoder
  4. Single-pass + cosine class scoring

Uses the sequential per-class predictor as ground truth.  All methods
use cross-class NMS to avoid overdetection of overlapping objects.

Metrics:
  - Precision, Recall, F1  (detection matching via mask IoU)
  - Class accuracy          (correct class among matched detections)
  - Average ms/image        (backbone + predict)

Usage:
    python scripts/benchmark_quality.py \
        --images-dir /path/to/images \
        --classes "car" "pedestrian" "bicycle" \
        --max-images 100

    # Custom resolution and thresholds:
    python scripts/benchmark_quality.py \
        --images-dir /path/to/coco/val2017 \
        --classes "person" "car" "dog" "cat" \
        --imgsz 1008 --confidence 0.3 --max-images 500
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
from PIL import Image

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_multiclass import Sam3MultiClassPredictor
from sam3.model.sam3_multiclass_fast import Sam3MultiClassPredictorFast


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def find_images(images_dir: str, max_images: int) -> List[str]:
    """Find image files in a directory, sorted alphabetically, up to max_images."""
    paths = []
    for fname in sorted(os.listdir(images_dir)):
        if Path(fname).suffix.lower() in IMAGE_EXTENSIONS:
            paths.append(os.path.join(images_dir, fname))
            if len(paths) >= max_images:
                break
    return paths


def mask_iou_matrix(masks_a: torch.Tensor, masks_b: torch.Tensor) -> torch.Tensor:
    """Compute pairwise mask IoU between two sets of binary masks.

    Args:
        masks_a: (M, H, W) binary masks.
        masks_b: (N, H, W) binary masks.

    Returns:
        (M, N) IoU matrix.
    """
    M, N = masks_a.shape[0], masks_b.shape[0]
    if M == 0 or N == 0:
        return torch.zeros(M, N, device=masks_a.device)

    a_flat = masks_a.reshape(M, -1).float()
    b_flat = masks_b.reshape(N, -1).float()

    intersection = a_flat @ b_flat.T
    area_a = a_flat.sum(dim=1, keepdim=True)  # (M, 1)
    area_b = b_flat.sum(dim=1, keepdim=True)  # (N, 1)
    union = area_a + area_b.T - intersection

    return intersection / union.clamp(min=1.0)


def match_detections(
    gt_results: Dict,
    pred_results: Dict,
    iou_threshold: float = 0.5,
) -> Dict:
    """Match predicted detections to ground truth via greedy mask IoU.

    Returns:
        tp:             matched detections (IoU >= threshold)
        fp:             unmatched predictions
        fn:             unmatched ground truth
        class_correct:  matched detections with correct class
        class_total:    total matched detections (= tp)
    """
    gt_masks = gt_results["masks"]
    pred_masks = pred_results["masks"]
    gt_cids = gt_results["class_ids"]
    pred_cids = pred_results["class_ids"]

    n_gt = len(gt_masks)
    n_pred = len(pred_masks)

    if n_gt == 0 and n_pred == 0:
        return dict(tp=0, fp=0, fn=0, class_correct=0, class_total=0)
    if n_gt == 0:
        return dict(tp=0, fp=n_pred, fn=0, class_correct=0, class_total=0)
    if n_pred == 0:
        return dict(tp=0, fp=0, fn=n_gt, class_correct=0, class_total=0)

    iou_mat = mask_iou_matrix(gt_masks, pred_masks)  # (n_gt, n_pred)

    # Greedy matching: pick highest IoU pair, mark both as used, repeat
    gt_matched = set()
    pred_matched = set()
    class_correct = 0

    flat_ious = iou_mat.reshape(-1)
    order = flat_ious.argsort(descending=True)

    for idx in order:
        iou_val = flat_ious[idx].item()
        if iou_val < iou_threshold:
            break
        gi = idx.item() // n_pred
        pi = idx.item() % n_pred
        if gi in gt_matched or pi in pred_matched:
            continue
        gt_matched.add(gi)
        pred_matched.add(pi)
        if gt_cids[gi] == pred_cids[pi]:
            class_correct += 1

    tp = len(gt_matched)
    return dict(
        tp=tp,
        fp=n_pred - len(pred_matched),
        fn=n_gt - tp,
        class_correct=class_correct,
        class_total=tp,
    )


def run_predictor(predictor, image, confidence_threshold, nms_threshold, device):
    """Run set_image + predict, return (results, elapsed_ms)."""
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    state = predictor.set_image(image)
    results = predictor.predict(
        state,
        confidence_threshold=confidence_threshold,
        nms_threshold=nms_threshold,
        per_class_nms=False,  # cross-class NMS to avoid overdetection
    )

    if device == "cuda":
        torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return results, elapsed_ms


def main():
    parser = argparse.ArgumentParser(
        description="SAM3 multi-class quality + speed benchmark"
    )
    parser.add_argument(
        "--images-dir", type=str, required=True,
        help="Directory containing input images",
    )
    parser.add_argument(
        "--classes", nargs="+", type=str,
        default=["car", "pedestrian", "bicycle"],
        help="Target class names",
    )
    parser.add_argument(
        "--max-images", type=int, default=1000,
        help="Maximum number of images to process (default: 1000)",
    )
    parser.add_argument(
        "--imgsz", type=int, default=1008,
        help="Model input resolution (default: 1008)",
    )
    parser.add_argument(
        "--confidence", type=float, default=0.3,
        help="Confidence threshold for all methods",
    )
    parser.add_argument(
        "--nms", type=float, default=0.7,
        help="Cross-class NMS IoU threshold",
    )
    parser.add_argument(
        "--match-iou", type=float, default=0.5,
        help="IoU threshold for matching predictions to ground truth",
    )
    parser.add_argument(
        "--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--compile", type=str, default=None,
        choices=["default", "reduce-overhead", "max-autotune"],
        help="torch.compile mode for fast predictors",
    )
    parser.add_argument(
        "--generic-prompt", type=str, default="object",
        help="Scene-level prompt for shared-encoder mode",
    )
    parser.add_argument(
        "--methods", nargs="+", type=str, default=None,
        choices=["sequential", "batched", "shared-encoder", "single-pass"],
        help="Subset of methods to benchmark (default: all)",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Discover images
    # ------------------------------------------------------------------
    image_paths = find_images(args.images_dir, args.max_images)
    if not image_paths:
        print(f"No images found in {args.images_dir}")
        sys.exit(1)

    n_images = len(image_paths)
    n_classes = len(args.classes)
    enabled = set(args.methods) if args.methods else {
        "sequential", "batched", "shared-encoder", "single-pass",
    }

    print(f"Images:      {n_images} (from {args.images_dir})")
    print(f"Classes:     {args.classes}")
    print(f"Resolution:  {args.imgsz}")
    print(f"Confidence:  {args.confidence}")
    print(f"NMS IoU:     {args.nms} (cross-class)")
    print(f"Match IoU:   {args.match_iou}")
    print(f"Methods:     {sorted(enabled)}")
    print()

    # ------------------------------------------------------------------
    # Load model and build predictors
    # ------------------------------------------------------------------
    print(f"Loading SAM3 model on {args.device}...")
    model = build_sam3_image_model(
        device=args.device,
        checkpoint_path=args.checkpoint,
        eval_mode=True,
    )

    # Ordered dict of (label, predictor)
    predictors: Dict[str, object] = {}

    # Ground truth is always the sequential predictor
    gt_predictor = Sam3MultiClassPredictor(
        model, device=args.device, resolution=args.imgsz,
    )
    gt_predictor.set_classes(args.classes)
    predictors["Sequential (GT)"] = gt_predictor

    if "batched" in enabled:
        p = Sam3MultiClassPredictorFast(
            model, device=args.device, resolution=args.imgsz,
            compile_mode=args.compile, use_fp16=True,
            presence_threshold=0.05,
        )
        p.set_classes(args.classes)
        predictors["Fast batched"] = p

    if "shared-encoder" in enabled:
        p = Sam3MultiClassPredictorFast(
            model, device=args.device, resolution=args.imgsz,
            compile_mode=args.compile, use_fp16=True,
            presence_threshold=0.05,
            shared_encoder=True, generic_prompt=args.generic_prompt,
        )
        p.set_classes(args.classes)
        predictors["Shared-encoder"] = p

    if "single-pass" in enabled:
        p = Sam3MultiClassPredictorFast(
            model, device=args.device, resolution=args.imgsz,
            compile_mode=args.compile, use_fp16=True,
            single_pass=True,
        )
        p.set_classes(args.classes)
        predictors["Single-pass"] = p

    # ------------------------------------------------------------------
    # Warmup (1 image, not counted)
    # ------------------------------------------------------------------
    print("Warming up...")
    warmup_img = Image.open(image_paths[0]).convert("RGB")
    for name, pred in predictors.items():
        run_predictor(pred, warmup_img, args.confidence, args.nms, args.device)
    print()

    # ------------------------------------------------------------------
    # Benchmark loop
    # ------------------------------------------------------------------
    # Per-method accumulators
    acc = {
        name: dict(tp=0, fp=0, fn=0, cls_ok=0, cls_tot=0, times=[], n_dets=0)
        for name in predictors
    }

    print(f"Processing {n_images} images...")
    for img_idx, img_path in enumerate(image_paths):
        image = Image.open(img_path).convert("RGB")

        if (img_idx + 1) % max(n_images // 20, 1) == 0 or img_idx == 0:
            print(f"  [{img_idx + 1:>{len(str(n_images))}}/{n_images}] "
                  f"{os.path.basename(img_path)}")

        gt_results = None

        for name, pred in predictors.items():
            results, elapsed_ms = run_predictor(
                pred, image, args.confidence, args.nms, args.device,
            )
            a = acc[name]
            a["times"].append(elapsed_ms)
            n_det = len(results["scores"])
            a["n_dets"] += n_det

            if name == "Sequential (GT)":
                gt_results = results
                # GT is always a perfect match against itself
                a["tp"] += n_det
                a["cls_ok"] += n_det
                a["cls_tot"] += n_det
            else:
                m = match_detections(
                    gt_results, results, iou_threshold=args.match_iou,
                )
                a["tp"] += m["tp"]
                a["fp"] += m["fp"]
                a["fn"] += m["fn"]
                a["cls_ok"] += m["class_correct"]
                a["cls_tot"] += m["class_total"]

    # ------------------------------------------------------------------
    # Results table
    # ------------------------------------------------------------------
    W = 85
    print(f"\n{'=' * W}")
    print(f"QUALITY BENCHMARK  "
          f"({n_images} images, {n_classes} classes, imgsz={args.imgsz})")
    print(f"{'=' * W}")
    header = (f"  {'Method':<28s}  {'Prec':>6s}  {'Rec':>6s}  {'F1':>6s}"
              f"  {'ClsAcc':>6s}  {'ms/img':>8s}  {'Dets':>6s}")
    print(header)
    print(f"  {'-' * (W - 2)}")

    gt_avg_ms = None
    for name in predictors:
        a = acc[name]
        tp, fp, fn = a["tp"], a["fp"], a["fn"]

        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-8)
        cls_acc = a["cls_ok"] / max(a["cls_tot"], 1)
        avg_ms = sum(a["times"]) / len(a["times"])

        tag = ""
        if name == "Sequential (GT)":
            gt_avg_ms = avg_ms
            tag = "  (baseline)"
        elif gt_avg_ms is not None and gt_avg_ms > 0:
            speedup = gt_avg_ms / avg_ms
            tag = f"  ({speedup:.1f}x)"

        print(f"  {name:<28s}  {prec:6.3f}  {rec:6.3f}  {f1:6.3f}"
              f"  {cls_acc:6.3f}  {avg_ms:8.1f}  {a['n_dets']:>6d}{tag}")

    print(f"{'=' * W}")
    print(f"  Cross-class NMS IoU={args.nms}, "
          f"match IoU={args.match_iou}, "
          f"conf={args.confidence}")
    print(f"  GT total detections: {acc['Sequential (GT)']['tp']}")
    print()


if __name__ == "__main__":
    main()
