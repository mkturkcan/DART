#!/usr/bin/env python3
"""Benchmark TRT backbone engines: FP16-fixed vs FP32 vs PyTorch."""

import time
import torch
from sam3.model_builder import build_sam3_image_model
from sam3.trt.trt_backbone import TRTBackbone

device = "cuda"
N_WARMUP = 5
N_ITERS = 50

print("Loading model...")
model = build_sam3_image_model(
    device=device, checkpoint_path="sam3.pt", eval_mode=True,
)
backbone = model.backbone
pos_module = backbone.vision_backbone.position_encoding

dummy = torch.randn(1, 3, 1008, 1008, device=device)

# Benchmark PyTorch
print("\n--- PyTorch FP32 ---")
with torch.inference_mode():
    for _ in range(N_WARMUP):
        backbone.forward_image(dummy)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N_ITERS):
        backbone.forward_image(dummy)
    torch.cuda.synchronize()
    pt_ms = (time.perf_counter() - t0) / N_ITERS * 1000
print(f"  {pt_ms:.1f} ms/frame")

# Benchmark PyTorch FP16
print("\n--- PyTorch FP16 (autocast) ---")
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
    for _ in range(N_WARMUP):
        backbone.forward_image(dummy)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N_ITERS):
        backbone.forward_image(dummy)
    torch.cuda.synchronize()
    pt16_ms = (time.perf_counter() - t0) / N_ITERS * 1000
print(f"  {pt16_ms:.1f} ms/frame")

# Benchmark engines
engines = {
    "TRT FP32": "backbone_fp32.engine",
    "TRT FP16 (fixed)": "backbone_fp16_fixed.engine",
}

import os
for label, path in engines.items():
    if not os.path.exists(path):
        print(f"\n--- {label} --- SKIPPED (not found: {path})")
        continue

    print(f"\n--- {label} ---")
    try:
        trt_bb = TRTBackbone(engine_path=path, device=device, pos_encoding_module=pos_module)
    except Exception as e:
        print(f"  SKIPPED: {e}")
        continue

    with torch.inference_mode():
        for _ in range(N_WARMUP):
            trt_bb.forward_image(dummy)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(N_ITERS):
            trt_bb.forward_image(dummy)
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) / N_ITERS * 1000
    print(f"  {ms:.1f} ms/frame")

print("\nDone!")
