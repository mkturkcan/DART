#!/usr/bin/env python3
"""Run COCO eval for all 4 distilled student models sequentially."""

import json
import subprocess
import sys
import time

PYTHON = r"C:\Users\mehme\anaconda3\envs\sam3\python.exe"

MODELS = [
    {
        "name": "efficientvit_l1",
        "backbone": "efficientvit_l1",
        "adapter": "distilled/efficientvit_l1_distilled.pt",
        "engine": "student_efficientvit_l1_fp16.engine",
    },
    {
        "name": "efficientvit_l2",
        "backbone": "efficientvit_l2",
        "adapter": "distilled/efficientvit_l2_distilled.pt",
        "engine": "student_efficientvit_l2_fp16.engine",
    },
    {
        "name": "repvit_m2_3",
        "backbone": "repvit_m2_3",
        "adapter": "distilled/repvit_m2_3_distilled.pt",
        "engine": "student_repvit_m2_3_fp16.engine",
    },
    {
        "name": "tiny_vit_21m",
        "backbone": "tiny_vit_21m",
        "adapter": "distilled/tiny_vit_21m_distilled.pt",
        "engine": "student_tiny_vit_21m_fp16.engine",
    },
]

all_results = []

for m in MODELS:
    print(f"\n{'='*70}")
    print(f"  EVALUATING: {m['name']}")
    print(f"{'='*70}")
    t0 = time.perf_counter()

    cmd = [
        PYTHON, "scripts/eval_coco.py",
        "--images-dir", "D:/val2017",
        "--ann-file", "D:/coco2017labels/coco/annotations/instances_val2017.json",
        "--checkpoint", "sam3.pt",
        "--student-backbone", m["backbone"],
        "--adapter-checkpoint", m["adapter"],
        "--trt-enc-dec", "enc_dec_1008_c16_fp16_16.engine",
        "--trt-max-classes", "16",
        "--configs", f"{m['name']}=trt:{m['engine']};imgsz:1008",
    ]

    import os
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    result = subprocess.run(cmd, capture_output=False, text=True, env=env)

    dt = time.perf_counter() - t0
    print(f"\n  {m['name']} completed in {dt/60:.1f} min (exit={result.returncode})")

    # Read the per-model results
    try:
        with open("coco_eval_results.json") as f:
            model_results = json.load(f)
        if model_results:
            all_results.extend(model_results)
    except Exception as e:
        print(f"  WARNING: Could not read results: {e}")

# Save combined results
with open("coco_eval_all_students.json", "w") as f:
    json.dump(all_results, f, indent=2)

# Print summary table
W = 105
print(f"\n\n{'='*W}")
print(f"COMBINED COCO val2017 RESULTS (5000 images, 80 classes)")
print(f"{'='*W}")
header = (
    f"  {'Model':<18s}  {'mAP':>6s}  {'mAP50':>6s}  {'mAP75':>6s}  "
    f"{'AP_S':>5s}  {'AP_M':>5s}  {'AP_L':>5s}  "
    f"{'AR@100':>6s}  {'ms/img':>7s}"
)
print(header)
print(f"  {'-'*(W-2)}")
for r in all_results:
    print(
        f"  {r['name']:<18s}  {r['mAP']:>6.3f}  {r['mAP50']:>6.3f}  "
        f"{r['mAP75']:>6.3f}  {r.get('mAP_small', 0):>5.3f}  "
        f"{r.get('mAP_medium', 0):>5.3f}  {r.get('mAP_large', 0):>5.3f}  "
        f"{r.get('AR100', 0):>6.3f}  {r['avg_ms']:>6.0f}ms"
    )
print(f"{'='*W}")
print(f"\nResults saved to coco_eval_all_students.json")
