#!/usr/bin/env python3
"""End-to-end detection test: TRT FP16-mixed backbone vs PyTorch backbone."""

import torch
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_multiclass_fast import Sam3MultiClassPredictorFast

device = "cuda"
classes = ["car", "person", "bicycle", "dog"]

print("Loading model...")
model = build_sam3_image_model(
    device=device, checkpoint_path="sam3.pt", eval_mode=True,
)

img = Image.open("x.jpg").convert("RGB")

# PyTorch predictor
print("\n--- PyTorch backbone ---")
pred_pt = Sam3MultiClassPredictorFast(
    model, device=device, use_fp16=True, detection_only=True,
)
pred_pt.set_classes(classes)

with torch.inference_mode():
    state_pt = pred_pt.set_image(img)
    res_pt = pred_pt.predict(state_pt, confidence_threshold=0.1)

print(f"  {len(res_pt['scores'])} detections")
for i in range(min(10, len(res_pt['scores']))):
    print(f"    [{i}] {res_pt['class_names'][i]:15s} score={res_pt['scores'][i].item():.3f} "
          f"box={[int(x) for x in res_pt['boxes'][i].tolist()]}")

# TRT backbone predictor
print("\n--- TRT FP16-mixed backbone ---")
pred_trt = Sam3MultiClassPredictorFast(
    model, device=device, use_fp16=True, detection_only=True,
    trt_engine_path="backbone_fp16_fixed.engine",
)
pred_trt.set_classes(classes)

with torch.inference_mode():
    state_trt = pred_trt.set_image(img)
    res_trt = pred_trt.predict(state_trt, confidence_threshold=0.1)

print(f"  {len(res_trt['scores'])} detections")
for i in range(min(10, len(res_trt['scores']))):
    print(f"    [{i}] {res_trt['class_names'][i]:15s} score={res_trt['scores'][i].item():.3f} "
          f"box={[int(x) for x in res_trt['boxes'][i].tolist()]}")

# Compare
print("\n--- Comparison ---")
n_pt = len(res_pt['scores'])
n_trt = len(res_trt['scores'])
print(f"  PyTorch: {n_pt} detections, TRT: {n_trt} detections")

if n_pt > 0 and n_trt > 0:
    # Match detections by class and approximate box overlap
    for i in range(min(5, n_pt)):
        pt_name = res_pt['class_names'][i]
        pt_score = res_pt['scores'][i].item()
        pt_box = res_pt['boxes'][i]

        # Find closest TRT detection with same class
        best_j = -1
        best_iou = 0
        for j in range(n_trt):
            if res_trt['class_names'][j] == pt_name:
                # Simple IoU
                trt_box = res_trt['boxes'][j]
                x1 = max(pt_box[0], trt_box[0]).item()
                y1 = max(pt_box[1], trt_box[1]).item()
                x2 = min(pt_box[2], trt_box[2]).item()
                y2 = min(pt_box[3], trt_box[3]).item()
                inter = max(0, x2 - x1) * max(0, y2 - y1)
                area_pt = (pt_box[2] - pt_box[0]).item() * (pt_box[3] - pt_box[1]).item()
                area_trt = (trt_box[2] - trt_box[0]).item() * (trt_box[3] - trt_box[1]).item()
                union = area_pt + area_trt - inter
                iou = inter / union if union > 0 else 0
                if iou > best_iou:
                    best_iou = iou
                    best_j = j

        if best_j >= 0:
            trt_score = res_trt['scores'][best_j].item()
            print(f"  PT[{i}] {pt_name:15s} score={pt_score:.3f} -> TRT[{best_j}] score={trt_score:.3f} IoU={best_iou:.3f}")
        else:
            print(f"  PT[{i}] {pt_name:15s} score={pt_score:.3f} -> NO MATCH in TRT")

print("\nDone!")
