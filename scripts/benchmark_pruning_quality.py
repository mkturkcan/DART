#!/usr/bin/env python3
"""Pruning quality benchmark using COCO val2017 ground truth.

Evaluates detection quality of different backbone configurations against
real COCO annotations (YOLO format).  Computes precision, recall, F1, and
mAP@50 across hundreds of images — much more rigorous than counting
detections on a single image.

Each "config" can use either a TRT backbone engine (with pruning baked in
at export time) or PyTorch backbone (with runtime mask toggling).  A shared
TRT encoder-decoder engine can be used across all configs.

Usage (TRT backbone + TRT enc-dec, 644, 80 COCO classes):
    python scripts/benchmark_pruning_quality.py \
        --images-dir D:/val2017 \
        --labels-dir D:/coco2017labels/coco/labels/val2017 \
        --checkpoint sam3.pt --imgsz 644 --coco \
        --trt-enc-dec enc_dec_644_coco80_fp16.engine --trt-max-classes 80 \
        --max-images 200 \
        --configs \
            "full_trt=trt:hf_backbone_644_fp16.engine" \
            "prune16_trt=trt:hf_backbone_644_masked_fp16.engine;mask:25:attn,28:mlp,27:attn,22:attn,28:attn,30:mlp,20:attn,27:mlp,26:attn,22:mlp,24:attn,18:attn,20:mlp,21:attn,25:mlp,18:mlp" \
            "full_fp32=pytorch"

Config format:
    "name"                         — PyTorch backbone, no pruning
    "name=trt:engine_path"         — TRT backbone
    "name=mask:spec"               — PyTorch backbone with mask-blocks
    "name=trt:engine;mask:spec"    — TRT backbone + mask-blocks for enc-dec model
"""

import argparse
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sam3.coco_classes import COCO_CLASSES
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_multiclass_fast import Sam3MultiClassPredictorFast


def load_yolo_labels(label_path: str) -> List[Tuple[int, float, float, float, float]]:
    """Load YOLO format labels: class_id cx cy w h (normalized 0-1)."""
    labels = []
    if not os.path.exists(label_path):
        return labels
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                cid = int(parts[0])
                cx, cy, w, h = (
                    float(parts[1]), float(parts[2]),
                    float(parts[3]), float(parts[4]),
                )
                labels.append((cid, cx, cy, w, h))
    return labels


def yolo_to_xyxy(
    labels: List[Tuple], img_w: int, img_h: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert YOLO labels to xyxy boxes + class ids."""
    if not labels:
        return torch.zeros(0, 4), torch.zeros(0, dtype=torch.long)

    boxes, cids = [], []
    for cid, cx, cy, w, h in labels:
        x1 = (cx - w / 2) * img_w
        y1 = (cy - h / 2) * img_h
        x2 = (cx + w / 2) * img_w
        y2 = (cy + h / 2) * img_h
        boxes.append([x1, y1, x2, y2])
        cids.append(cid)

    return (
        torch.tensor(boxes, dtype=torch.float32),
        torch.tensor(cids, dtype=torch.long),
    )


def box_iou_matrix(
    boxes_a: torch.Tensor, boxes_b: torch.Tensor,
) -> torch.Tensor:
    """Pairwise box IoU between two sets of xyxy boxes."""
    M, N = boxes_a.shape[0], boxes_b.shape[0]
    if M == 0 or N == 0:
        return torch.zeros(M, N)

    lt = torch.max(boxes_a[:, None, :2], boxes_b[None, :, :2])
    rb = torch.min(boxes_a[:, None, 2:], boxes_b[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]

    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter

    return inter / union.clamp(min=1e-6)


def match_predictions_to_gt(
    pred_boxes: torch.Tensor,
    pred_scores: torch.Tensor,
    pred_cids: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_cids: torch.Tensor,
    iou_threshold: float = 0.5,
) -> Dict:
    """Match predictions to GT via greedy box IoU."""
    n_pred = len(pred_boxes)
    n_gt = len(gt_boxes)

    if n_gt == 0 and n_pred == 0:
        return dict(tp=0, fp=0, fn=0, cls_correct=0)
    if n_gt == 0:
        return dict(tp=0, fp=n_pred, fn=0, cls_correct=0)
    if n_pred == 0:
        return dict(tp=0, fp=0, fn=n_gt, cls_correct=0)

    iou_mat = box_iou_matrix(pred_boxes, gt_boxes)

    pred_matched = set()
    gt_matched = set()
    cls_correct = 0

    flat_ious = iou_mat.reshape(-1)
    order = flat_ious.argsort(descending=True)

    for idx in order:
        iou_val = flat_ious[idx].item()
        if iou_val < iou_threshold:
            break
        pi = idx.item() // n_gt
        gi = idx.item() % n_gt
        if pi in pred_matched or gi in gt_matched:
            continue
        pred_matched.add(pi)
        gt_matched.add(gi)
        if pred_cids[pi] == gt_cids[gi]:
            cls_correct += 1

    tp = len(pred_matched)
    return dict(
        tp=tp,
        fp=n_pred - tp,
        fn=n_gt - len(gt_matched),
        cls_correct=cls_correct,
    )


def compute_ap_per_class(
    all_preds: List[Dict],
    iou_threshold: float = 0.5,
    num_classes: int = 80,
) -> Tuple[float, Dict[int, float]]:
    """Compute per-class AP@50 and mAP@50 using COCO-style 101-pt interp."""
    class_preds = defaultdict(list)
    class_n_gt = defaultdict(int)

    for img_data in all_preds:
        pred_boxes = img_data["pred_boxes"]
        pred_scores = img_data["pred_scores"]
        pred_cids = img_data["pred_cids"]
        gt_boxes = img_data["gt_boxes"]
        gt_cids = img_data["gt_cids"]

        for cid in gt_cids.tolist():
            class_n_gt[cid] += 1

        if len(pred_boxes) == 0 or len(gt_boxes) == 0:
            for i in range(len(pred_boxes)):
                cid = pred_cids[i].item()
                class_preds[cid].append((pred_scores[i].item(), False))
            continue

        iou_mat = box_iou_matrix(pred_boxes, gt_boxes)
        gt_used = set()
        score_order = pred_scores.argsort(descending=True)

        for pi in score_order.tolist():
            cid = pred_cids[pi].item()
            score = pred_scores[pi].item()

            best_iou, best_gi = 0, -1
            for gi in range(len(gt_boxes)):
                if gi in gt_used or gt_cids[gi].item() != cid:
                    continue
                iou_val = iou_mat[pi, gi].item()
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_gi = gi

            if best_iou >= iou_threshold and best_gi >= 0:
                gt_used.add(best_gi)
                class_preds[cid].append((score, True))
            else:
                class_preds[cid].append((score, False))

    per_class_ap = {}
    for cid in range(num_classes):
        n_gt = class_n_gt.get(cid, 0)
        if n_gt == 0:
            continue

        preds = class_preds.get(cid, [])
        if not preds:
            per_class_ap[cid] = 0.0
            continue

        preds.sort(key=lambda x: -x[0])
        is_tp = [p[1] for p in preds]

        tp_cum = np.cumsum(is_tp).astype(np.float64)
        fp_cum = np.cumsum([not t for t in is_tp]).astype(np.float64)

        precision = tp_cum / (tp_cum + fp_cum)
        recall = tp_cum / n_gt

        # 101-point interpolation
        ap = 0.0
        for t in np.linspace(0, 1, 101):
            prec_at_recall = precision[recall >= t]
            if len(prec_at_recall) > 0:
                ap += prec_at_recall.max()
        ap /= 101.0
        per_class_ap[cid] = ap

    if per_class_ap:
        mAP = np.mean(list(per_class_ap.values()))
    else:
        mAP = 0.0

    return mAP, per_class_ap


def parse_config(config_str: str) -> Dict:
    """Parse config string into a dict.

    Formats:
        "name"                         → PyTorch, no pruning
        "name=trt:engine_path"         → TRT backbone
        "name=mask:spec"               → PyTorch + mask-blocks
        "name=trt:engine;mask:spec"    → TRT backbone + mask-blocks
    """
    result = {"name": None, "trt": None, "mask_blocks": None, "imgsz": None}

    # Split name from rest
    if "=" in config_str:
        name, rest = config_str.split("=", 1)
        result["name"] = name

        # Parse semicolon-separated parts
        for part in rest.split(";"):
            if part.startswith("trt:"):
                result["trt"] = part[4:]
            elif part.startswith("mask:"):
                result["mask_blocks"] = part[5:].split(",")
            elif part.startswith("imgsz:"):
                result["imgsz"] = int(part[6:])
            elif part == "pytorch":
                result["trt"] = None
            else:
                if part.endswith(".engine"):
                    result["trt"] = part
                else:
                    result["trt"] = None
    else:
        result["name"] = config_str

    return result


def apply_mask_config(trunk, mask_blocks: Optional[List[str]]):
    """Toggle mask_attn/mask_mlp on trunk blocks."""
    for blk in trunk.blocks:
        blk.mask_attn = False
        blk.mask_mlp = False

    if mask_blocks is None:
        return 0

    n = 0
    for entry in mask_blocks:
        idx_str, sub_type = entry.split(":")
        blk = trunk.blocks[int(idx_str)]
        if sub_type == "attn":
            blk.mask_attn = True
        elif sub_type == "mlp":
            blk.mask_mlp = True
        n += 1
    return n


def find_image_label_pairs(
    images_dir: str, labels_dir: str, max_images: int,
    most_annotations: bool = False,
) -> List[Tuple[str, str]]:
    """Find matching image-label pairs.

    If most_annotations=True, sorts by annotation count descending and
    returns the top max_images most-annotated images.
    """
    exts = {".jpg", ".jpeg", ".png"}
    all_pairs = []
    for fname in sorted(os.listdir(images_dir)):
        stem = Path(fname).stem
        if Path(fname).suffix.lower() not in exts:
            continue
        label_path = os.path.join(labels_dir, stem + ".txt")
        if os.path.exists(label_path):
            all_pairs.append((os.path.join(images_dir, fname), label_path))

    if most_annotations and len(all_pairs) > max_images:
        # Sort by number of annotations (lines in label file), descending
        def count_lines(pair):
            with open(pair[1]) as f:
                return sum(1 for line in f if line.strip())
        all_pairs.sort(key=count_lines, reverse=True)

    return all_pairs[:max_images]


def evaluate_config(
    config: Dict,
    model,
    class_names: List[str],
    eval_cids: set,
    pairs: List[Tuple[str, str]],
    trt_enc_dec_path: Optional[str],
    trt_max_classes: int,
    imgsz: int,
    confidence: float,
    nms: float,
    iou_threshold: float,
    device: str,
    text_cache: Optional[str] = None,
) -> Dict:
    """Run evaluation for a single config."""
    config_name = config["name"]
    trt_engine = config["trt"]
    mask_blocks = config["mask_blocks"]
    config_imgsz = config.get("imgsz") or imgsz  # per-config override

    trunk = model.backbone.vision_backbone.trunk

    # Apply mask-blocks (for both PyTorch and TRT modes — TRT needs it
    # for the Meta enc/dec model dims; PyTorch needs it for backbone pruning)
    n_masked = apply_mask_config(trunk, mask_blocks)

    # Determine backbone mode
    is_trt = trt_engine is not None
    use_trt_enc_dec = trt_enc_dec_path is not None and is_trt
    mode_str = f"TRT ({os.path.basename(trt_engine)})" if is_trt else "PyTorch"

    print(f"\n{'='*60}")
    print(f"Config: {config_name}")
    print(f"  Backbone: {mode_str}")
    print(f"  Resolution: {config_imgsz}")
    print(f"  Masked sub-blocks: {n_masked}")
    if use_trt_enc_dec:
        print(f"  Enc-dec: TRT ({os.path.basename(trt_enc_dec_path)})")
    else:
        print(f"  Enc-dec: PyTorch")
    print(f"{'='*60}")

    # Create predictor for this config
    # PyTorch-only configs don't use TRT enc-dec (resolution may differ)
    predictor = Sam3MultiClassPredictorFast(
        model,
        device=device,
        resolution=config_imgsz,
        use_fp16=is_trt or use_trt_enc_dec,
        detection_only=True,
        presence_threshold=0.0,
        trt_engine_path=trt_engine,
        trt_enc_dec_engine_path=trt_enc_dec_path if use_trt_enc_dec else None,
        trt_max_classes=trt_max_classes,
    )
    predictor.set_classes(class_names, text_cache=text_cache)

    # Warmup (2 images)
    print("  Warming up ...")
    for i in range(min(2, len(pairs))):
        img = Image.open(pairs[i][0]).convert("RGB")
        state = predictor.set_image(img)
        predictor.predict(state, confidence_threshold=confidence, nms_threshold=nms)

    # Evaluate
    n_images = len(pairs)
    all_img_data = []
    total_preds = 0
    total_gt = 0
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_cls_correct = 0
    total_ms = 0

    n_skipped = 0
    print(f"  Evaluating {n_images} images ...")
    for img_idx, (img_path, label_path) in enumerate(pairs):
        if (img_idx + 1) % max(n_images // 10, 1) == 0:
            print(f"    [{img_idx + 1:>{len(str(n_images))}}/{n_images}]")

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            n_skipped += 1
            continue
        img_w, img_h = image.size
        labels = load_yolo_labels(label_path)
        labels = [
            (cid, cx, cy, w, h)
            for cid, cx, cy, w, h in labels
            if cid in eval_cids
        ]
        gt_boxes, gt_cids = yolo_to_xyxy(labels, img_w, img_h)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        state = predictor.set_image(image)
        results = predictor.predict(
            state,
            confidence_threshold=confidence,
            nms_threshold=nms,
            per_class_nms=False,
        )
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000
        total_ms += elapsed_ms

        # Extract predictions
        if results["boxes"] is not None and len(results["boxes"]) > 0:
            pred_boxes = results["boxes"].cpu()
            pred_scores = results["scores"].cpu()
            pred_cids = results["class_ids"].cpu()
        else:
            pred_boxes = torch.zeros(0, 4)
            pred_scores = torch.zeros(0)
            pred_cids = torch.zeros(0, dtype=torch.long)

        all_img_data.append({
            "pred_boxes": pred_boxes,
            "pred_scores": pred_scores,
            "pred_cids": pred_cids,
            "gt_boxes": gt_boxes,
            "gt_cids": gt_cids,
        })

        m = match_predictions_to_gt(
            pred_boxes, pred_scores, pred_cids,
            gt_boxes, gt_cids, iou_threshold=iou_threshold,
        )
        total_tp += m["tp"]
        total_fp += m["fp"]
        total_fn += m["fn"]
        total_cls_correct += m["cls_correct"]
        total_preds += len(pred_boxes)
        total_gt += len(gt_boxes)

    # Aggregate metrics
    n_evaluated = n_images - n_skipped
    if n_skipped:
        print(f"  Skipped {n_skipped} corrupt images")
    precision = total_tp / max(total_tp + total_fp, 1)
    recall = total_tp / max(total_tp + total_fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    cls_acc = total_cls_correct / max(total_tp, 1)
    avg_ms = total_ms / max(n_evaluated, 1)

    mAP, per_class_ap = compute_ap_per_class(
        all_img_data, iou_threshold=iou_threshold, num_classes=80,
    )

    result = {
        "name": config_name,
        "mode": "TRT" if is_trt else "PyTorch",
        "imgsz": config_imgsz,
        "n_masked": n_masked,
        "total_preds": total_preds,
        "total_gt": total_gt,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "cls_acc": cls_acc,
        "mAP50": mAP,
        "avg_ms": avg_ms,
        "per_class_ap": per_class_ap,
    }

    print(f"  Results: Dets={total_preds}, GT={total_gt}, "
          f"P={precision:.3f} R={recall:.3f} F1={f1:.3f} "
          f"mAP@50={mAP:.3f} ({avg_ms:.0f}ms/img)")

    # Clean up TRT resources
    del predictor
    torch.cuda.empty_cache()

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Pruning quality benchmark with COCO val2017 ground truth",
    )
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--labels-dir", required=True)
    parser.add_argument("--checkpoint", default="sam3.pt")
    parser.add_argument("--max-images", type=int, default=200)
    parser.add_argument("--imgsz", type=int, default=1008)
    parser.add_argument("--confidence", type=float, default=0.3)
    parser.add_argument("--nms", type=float, default=0.7)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument(
        "--trt-enc-dec", type=str, default=None,
        help="Shared TRT enc-dec engine (used for all configs)",
    )
    parser.add_argument("--trt-max-classes", type=int, default=80)
    parser.add_argument(
        "--text-cache", type=str, default=None,
        help="Path to cache text embeddings (.pt file)",
    )
    parser.add_argument(
        "--coco", action="store_true",
        help="Use all 80 COCO classes",
    )
    parser.add_argument(
        "--classes", nargs="+", type=str, default=None,
        help="Subset of class names (default: all 80 COCO if --coco)",
    )
    parser.add_argument(
        "--configs", nargs="+", required=True,
        help='Configs: "name", "name=trt:engine", "name=mask:spec", '
             '"name=trt:engine;mask:spec", "name=pytorch"',
    )
    parser.add_argument(
        "--most-annotations", action="store_true",
        help="Select images with the most GT annotations (more informative)",
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    # Classes
    if args.coco:
        class_names = list(COCO_CLASSES)
    elif args.classes:
        class_names = args.classes
    else:
        class_names = list(COCO_CLASSES)

    class_to_idx = {name: i for i, name in enumerate(COCO_CLASSES)}
    eval_cids = set()
    for name in class_names:
        if name in class_to_idx:
            eval_cids.add(class_to_idx[name])

    n_classes = len(class_names)

    # Parse configs
    configs = [parse_config(c) for c in args.configs]

    # Find image-label pairs
    pairs = find_image_label_pairs(
        args.images_dir, args.labels_dir, args.max_images,
        most_annotations=args.most_annotations,
    )
    n_images = len(pairs)
    if n_images == 0:
        print("No matching image-label pairs found!")
        sys.exit(1)

    selection = "most-annotated" if args.most_annotations else "first alphabetical"
    print(f"Images:        {n_images} ({selection})")
    print(f"Classes:       {n_classes}")
    print(f"Resolution:    {args.imgsz}")
    print(f"Confidence:    {args.confidence}")
    print(f"NMS:           {args.nms}")
    print(f"IoU threshold: {args.iou_threshold}")
    print(f"Configs:       {[c['name'] for c in configs]}")
    if args.trt_enc_dec:
        print(f"TRT enc-dec:   {args.trt_enc_dec}")

    # Load model
    print(f"\nLoading SAM3 from {args.checkpoint} ...")
    model = build_sam3_image_model(
        checkpoint_path=args.checkpoint,
        device=args.device,
        eval_mode=True,
    )

    # Run each config
    results_table = []
    for config in configs:
        result = evaluate_config(
            config=config,
            model=model,
            class_names=class_names,
            eval_cids=eval_cids,
            pairs=pairs,
            trt_enc_dec_path=args.trt_enc_dec,
            trt_max_classes=args.trt_max_classes,
            imgsz=args.imgsz,
            confidence=args.confidence,
            nms=args.nms,
            iou_threshold=args.iou_threshold,
            device=args.device,
            text_cache=args.text_cache,
        )
        results_table.append(result)

    # Summary table
    W = 110
    print(f"\n\n{'=' * W}")
    print(f"PRUNING QUALITY BENCHMARK  "
          f"({n_images} images, {n_classes} classes)")
    print(f"{'=' * W}")
    header = (
        f"  {'Config':<20s}  {'Mode':<8s}  {'Res':>4s}  {'Pruned':>6s}  {'Dets':>6s}  "
        f"{'Prec':>6s}  {'Rec':>6s}  {'F1':>6s}  "
        f"{'ClsAcc':>6s}  {'mAP@50':>7s}  {'ms/img':>7s}"
    )
    print(header)
    print(f"  {'-' * (W - 2)}")

    baseline_map = None
    baseline_ms = None
    for r in results_table:
        if baseline_map is None:
            baseline_map = r["mAP50"]
            baseline_ms = r["avg_ms"]

        print(
            f"  {r['name']:<20s}  {r['mode']:<8s}  {r['imgsz']:>4d}  "
            f"{r['n_masked']:>6d}  "
            f"{r['total_preds']:>6d}  "
            f"{r['precision']:>6.3f}  {r['recall']:>6.3f}  {r['f1']:>6.3f}  "
            f"{r['cls_acc']:>6.3f}  {r['mAP50']:>7.3f}  {r['avg_ms']:>6.0f}ms"
        )

    print(f"{'=' * W}")
    print(f"  GT annotations: {results_table[0]['total_gt']}  "
          f"(conf={args.confidence}, NMS={args.nms}, "
          f"IoU={args.iou_threshold})")

    # Relative quality table
    if len(results_table) > 1 and baseline_map and baseline_map > 0:
        print(f"\n{'=' * 65}")
        print("RELATIVE QUALITY (vs first config):")
        print(f"{'=' * 65}")
        for r in results_table:
            rel = r["mAP50"] / baseline_map * 100
            delta = r["mAP50"] - baseline_map
            speed = baseline_ms / r["avg_ms"] if r["avg_ms"] > 0 else 0
            print(f"  {r['name']:<20s}  mAP@50={r['mAP50']:.3f}  "
                  f"({rel:5.1f}%, {delta:+.3f})  "
                  f"speed={speed:.2f}x")
        print(f"{'=' * 65}")

    # Per-class top/bottom for first config
    if results_table:
        full_ap = results_table[0]["per_class_ap"]
        if full_ap:
            sorted_classes = sorted(full_ap.items(), key=lambda x: -x[1])
            print(f"\nPer-class AP@50 ({results_table[0]['name']}) — top 5:")
            for cid, ap in sorted_classes[:5]:
                print(f"  {COCO_CLASSES[cid]:<20s}  AP={ap:.3f}")
            print(f"Per-class AP@50 ({results_table[0]['name']}) — bottom 5:")
            for cid, ap in sorted_classes[-5:]:
                print(f"  {COCO_CLASSES[cid]:<20s}  AP={ap:.3f}")

    return results_table


if __name__ == "__main__":
    main()
