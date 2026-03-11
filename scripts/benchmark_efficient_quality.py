#!/usr/bin/env python3
"""Benchmark EfficientSAM3 models on COCO val2017.

Evaluates detection mAP@50 for each EfficientSAM3 lightweight backbone variant
against the full SAM3 model (ViT-H). Uses the same evaluation infrastructure
as benchmark_pruning_quality.py.

Usage:
    python scripts/benchmark_efficient_quality.py \
        --images-dir D:/val2017 \
        --labels-dir D:/coco2017labels/coco/labels/val2017 \
        --sam3-checkpoint sam3.pt \
        --efficient-dir stage1_all_converted \
        --max-images 200 --coco
"""

import argparse
import os
import sys
import time
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sam3.coco_classes import COCO_CLASSES
from sam3.efficient_backbone import build_efficientsam3_model
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_multiclass_fast import Sam3MultiClassPredictorFast
from scripts.benchmark_pruning_quality import (
    compute_ap_per_class,
    find_image_label_pairs,
    load_yolo_labels,
    match_predictions_to_gt,
    yolo_to_xyxy,
)

# Checkpoint filename -> (backbone_type, model_name, display_name)
EFFICIENT_MODELS = {
    "efficient_sam3_efficientvit_m_geo_ft.pt": ("efficientvit", "b1", "EfficientViT-B1"),
    "efficient_sam3_repvit_l.pt": ("repvit", "m2.3", "RepViT-M2.3"),
    "efficient_sam3_tinyvit_m_geo_ft.pt": ("tinyvit", "11m", "TinyViT-11M"),
}


def evaluate_model(
    model,
    display_name,
    class_names,
    eval_cids,
    pairs,
    imgsz,
    confidence,
    nms,
    iou_threshold,
    device,
):
    """Run COCO evaluation for a single model."""
    print(f"\n{'='*60}")
    print(f"Model: {display_name}")
    print(f"  Resolution: {imgsz}")
    print(f"{'='*60}")

    predictor = Sam3MultiClassPredictorFast(
        model,
        device=device,
        resolution=imgsz,
        use_fp16=True,
        detection_only=True,
        presence_threshold=0.0,
    )
    predictor.set_classes(class_names)

    # Warmup
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
        labels = [(cid, cx, cy, w, h) for cid, cx, cy, w, h in labels if cid in eval_cids]
        gt_boxes, gt_cids = yolo_to_xyxy(labels, img_w, img_h)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        state = predictor.set_image(image)
        results = predictor.predict(
            state, confidence_threshold=confidence, nms_threshold=nms, per_class_nms=False,
        )
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000
        total_ms += elapsed_ms

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
            pred_boxes, pred_scores, pred_cids, gt_boxes, gt_cids, iou_threshold=iou_threshold,
        )
        total_tp += m["tp"]
        total_fp += m["fp"]
        total_fn += m["fn"]
        total_cls_correct += m["cls_correct"]
        total_preds += len(pred_boxes)
        total_gt += len(gt_boxes)

    n_evaluated = n_images - n_skipped
    precision = total_tp / max(total_tp + total_fp, 1)
    recall = total_tp / max(total_tp + total_fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    cls_acc = total_cls_correct / max(total_tp, 1)
    avg_ms = total_ms / max(n_evaluated, 1)

    mAP, per_class_ap = compute_ap_per_class(all_img_data, iou_threshold=iou_threshold, num_classes=80)

    print(
        f"  Results: Dets={total_preds}, GT={total_gt}, "
        f"P={precision:.3f} R={recall:.3f} F1={f1:.3f} "
        f"mAP@50={mAP:.3f} ({avg_ms:.0f}ms/img)"
    )

    del predictor
    torch.cuda.empty_cache()

    return {
        "name": display_name,
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


def main():
    parser = argparse.ArgumentParser(description="EfficientSAM3 COCO mAP benchmark")
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--labels-dir", required=True)
    parser.add_argument("--sam3-checkpoint", default="sam3.pt")
    parser.add_argument("--efficient-dir", default="stage1_all_converted")
    parser.add_argument("--max-images", type=int, default=200)
    parser.add_argument("--imgsz", type=int, default=1008)
    parser.add_argument("--confidence", type=float, default=0.3)
    parser.add_argument("--nms", type=float, default=0.7)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--coco", action="store_true")
    parser.add_argument("--classes", nargs="+", type=str, default=None)
    parser.add_argument("--most-annotations", action="store_true")
    parser.add_argument("--skip-full", action="store_true", help="Skip full SAM3 baseline")
    parser.add_argument("--only", type=str, default=None, help="Only run one model (e.g. efficientvit)")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.coco:
        class_names = list(COCO_CLASSES)
    elif args.classes:
        class_names = args.classes
    else:
        class_names = list(COCO_CLASSES)

    class_to_idx = {name: i for i, name in enumerate(COCO_CLASSES)}
    eval_cids = {class_to_idx[name] for name in class_names if name in class_to_idx}

    pairs = find_image_label_pairs(
        args.images_dir, args.labels_dir, args.max_images,
        most_annotations=args.most_annotations,
    )
    if not pairs:
        print("No matching image-label pairs found!")
        sys.exit(1)

    print(f"Images:     {len(pairs)}")
    print(f"Classes:    {len(class_names)}")
    print(f"Resolution: {args.imgsz}")

    results_table = []

    # 1. Full SAM3 baseline (ViT-H)
    if not args.skip_full:
        print(f"\nLoading full SAM3 from {args.sam3_checkpoint} ...")
        full_model = build_sam3_image_model(
            checkpoint_path=args.sam3_checkpoint,
            device=args.device,
            eval_mode=True,
        )
        result = evaluate_model(
            full_model, "SAM3 (ViT-H)", class_names, eval_cids, pairs,
            args.imgsz, args.confidence, args.nms, args.iou_threshold, args.device,
        )
        results_table.append(result)
        del full_model
        torch.cuda.empty_cache()

    # 2. EfficientSAM3 variants
    for ckpt_name, (backbone_type, model_name, display_name) in EFFICIENT_MODELS.items():
        if args.only and args.only not in ckpt_name and args.only not in display_name.lower():
            continue

        ckpt_path = os.path.join(args.efficient_dir, ckpt_name)
        if not os.path.exists(ckpt_path):
            print(f"\nSkipping {display_name}: {ckpt_path} not found")
            continue

        print(f"\nLoading {display_name} from {ckpt_path} ...")
        try:
            model = build_efficientsam3_model(
                backbone_type=backbone_type,
                model_name=model_name,
                checkpoint_path=ckpt_path,
                device=args.device,
                eval_mode=True,
            )
        except Exception as e:
            print(f"  ERROR loading {display_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

        result = evaluate_model(
            model, display_name, class_names, eval_cids, pairs,
            args.imgsz, args.confidence, args.nms, args.iou_threshold, args.device,
        )
        results_table.append(result)
        del model
        torch.cuda.empty_cache()

    # Summary
    W = 100
    print(f"\n\n{'='*W}")
    print(f"EFFICIENTSAM3 QUALITY BENCHMARK ({len(pairs)} images, {len(class_names)} classes, {args.imgsz}px)")
    print(f"{'='*W}")
    header = (
        f"  {'Model':<22s}  {'Dets':>6s}  "
        f"{'Prec':>6s}  {'Rec':>6s}  {'F1':>6s}  "
        f"{'ClsAcc':>6s}  {'mAP@50':>7s}  {'ms/img':>7s}"
    )
    print(header)
    print(f"  {'-'*(W-2)}")

    baseline_map = results_table[0]["mAP50"] if results_table else None
    for r in results_table:
        rel = f"({r['mAP50']/baseline_map*100:.0f}%)" if baseline_map and baseline_map > 0 else ""
        print(
            f"  {r['name']:<22s}  {r['total_preds']:>6d}  "
            f"{r['precision']:>6.3f}  {r['recall']:>6.3f}  {r['f1']:>6.3f}  "
            f"{r['cls_acc']:>6.3f}  {r['mAP50']:>7.3f}  {r['avg_ms']:>6.0f}ms  {rel}"
        )

    print(f"{'='*W}")
    if results_table:
        print(f"  GT annotations: {results_table[0]['total_gt']}")


if __name__ == "__main__":
    main()
