#!/usr/bin/env python3
"""COCO val2017 instance segmentation evaluation (GT-box-prompted).

Replicates the EfficientSAM3 official evaluation protocol:
  - Given ground-truth bounding boxes, predict instance masks
  - Measure mean mask IoU across all annotations

This evaluates the SAM-style "segment anything given a prompt" capability,
NOT open-vocabulary detection.

Usage:
    PYTHONIOENCODING=utf-8 python scripts/eval_cocoseg.py \
        --images-dir D:/val2017 \
        --ann-file D:/coco2017labels/coco/annotations/instances_val2017.json \
        --checkpoint sam3.pt

    # EfficientSAM3 model:
    PYTHONIOENCODING=utf-8 python scripts/eval_cocoseg.py \
        --images-dir D:/val2017 \
        --ann-file D:/coco2017labels/coco/annotations/instances_val2017.json \
        --checkpoint stage1_all_converted/efficient_sam3_repvit_l.pt \
        --efficient-backbone repvit --efficient-model m2_3
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from pycocotools.coco import COCO
except ImportError:
    print("ERROR: pycocotools required. Install: pip install pycocotools-windows")
    sys.exit(1)

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


def calculate_iou(pred_mask, gt_mask):
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    if union == 0:
        return 0.0
    return intersection / union


def main():
    parser = argparse.ArgumentParser(
        description="COCO instance segmentation eval (GT-box-prompted, mask mIoU)"
    )
    parser.add_argument("--images-dir", required=True, help="val2017 images directory")
    parser.add_argument("--ann-file", required=True, help="instances_val2017.json")
    parser.add_argument("--checkpoint", default="sam3.pt")
    parser.add_argument("--max-images", type=int, default=5000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--efficient-backbone", type=str, default=None,
        choices=["efficientvit", "repvit", "tinyvit"],
    )
    parser.add_argument("--efficient-model", type=str, default=None)
    args = parser.parse_args()

    # Load COCO annotations
    print(f"Loading COCO annotations from {args.ann_file} ...")
    coco = COCO(args.ann_file)
    img_ids = sorted(coco.getImgIds())[:args.max_images]
    print(f"  {len(img_ids)} images")

    # Build model with inst_interactivity enabled
    if args.efficient_backbone:
        if not args.efficient_model:
            print("ERROR: --efficient-backbone requires --efficient-model")
            sys.exit(1)
        print(f"\nLoading EfficientSAM3 ({args.efficient_backbone} {args.efficient_model}) ...")
        from sam3.efficient_backbone import build_efficientsam3_model
        model = build_efficientsam3_model(
            backbone_type=args.efficient_backbone,
            model_name=args.efficient_model,
            checkpoint_path=args.checkpoint,
            device=args.device,
            eval_mode=True,
            enable_inst_interactivity=True,
        )
    else:
        print(f"\nLoading SAM3 from {args.checkpoint} ...")
        model = build_sam3_image_model(
            checkpoint_path=args.checkpoint,
            device=args.device,
            eval_mode=True,
            enable_inst_interactivity=True,
        )

    processor = Sam3Processor(model, device=args.device)

    # Evaluate
    ious = []
    n_skip = 0
    start_time = time.time()

    pbar = tqdm(img_ids, desc="Eval", unit="img")
    for img_id in pbar:
        img_info = coco.loadImgs(img_id)[0]
        img_path = os.path.join(args.images_dir, img_info["file_name"])
        if not os.path.exists(img_path):
            n_skip += 1
            continue

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            tqdm.write(f"  SKIP {img_path}: {e}")
            n_skip += 1
            continue

        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        if not anns:
            continue

        with torch.inference_mode():
            inference_state = processor.set_image(image)

        for ann in anns:
            if ann.get("iscrowd", 0):
                continue

            bbox = ann["bbox"]  # x, y, w, h
            box = np.array([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])

            with torch.inference_mode():
                masks, scores, _ = model.predict_inst(
                    inference_state,
                    point_coords=None,
                    point_labels=None,
                    box=box[None, :],
                    multimask_output=False,
                )

            if isinstance(masks, torch.Tensor):
                pred_mask = masks[0].cpu().numpy() > 0
            else:
                pred_mask = masks[0] > 0

            gt_mask = coco.annToMask(ann)
            iou = calculate_iou(pred_mask, gt_mask)
            ious.append(iou)

        if len(ious) > 0:
            pbar.set_postfix_str(f"mIoU={np.mean(ious):.4f}, n={len(ious)}")

    elapsed = time.time() - start_time

    if not ious:
        print("\nNo valid evaluations.")
        return

    miou = np.mean(ious)
    print(f"\n{'='*60}")
    print(f"COCO val2017 Instance Segmentation (GT-box-prompted)")
    print(f"{'='*60}")
    if args.efficient_backbone:
        print(f"  Model:      EfficientSAM3 {args.efficient_backbone} {args.efficient_model}")
    else:
        print(f"  Model:      SAM3 (ViT-H)")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Images:     {len(img_ids)} (skipped {n_skip})")
    print(f"  Annotations: {len(ious)}")
    print(f"  mIoU:       {miou:.4f}")
    print(f"  Time:       {elapsed:.1f}s ({elapsed/len(img_ids)*1000:.0f}ms/img)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
