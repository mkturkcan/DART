#!/usr/bin/env python3
"""Test surgical FP16 engine: accuracy + speed."""

import time
import torch
from sam3.model_builder import build_sam3_image_model
from sam3.trt.trt_backbone import TRTBackbone
from PIL import Image
from torchvision.transforms import v2

device = "cuda"
N_WARMUP = 10
N_ITERS = 100

print("Loading model...")
model = build_sam3_image_model(
    device=device, checkpoint_path="sam3.pt", eval_mode=True,
)
backbone = model.backbone
pos_module = backbone.vision_backbone.position_encoding

# Prepare image input
transform = v2.Compose([
    v2.ToDtype(torch.uint8, scale=True),
    v2.Resize(size=(1008, 1008)),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])
img = Image.open("x.jpg").convert("RGB")
img = img.resize((1008, 1008), Image.BILINEAR)
tensor = v2.functional.to_image(img).to(device)
tensor = transform(tensor).unsqueeze(0)

dummy = torch.randn(1, 3, 1008, 1008, device=device)

# PyTorch reference
with torch.inference_mode():
    pt_out = backbone.forward_image(tensor)
pt_fpn = pt_out["backbone_fpn"]

engines = {
    "FP16 surgical (attn-only FP32)": "backbone_fp16_surgical.engine",
    "FP16 all-MatMul FP32": "backbone_fp16_fixed.engine",
}

for label, path in engines.items():
    import os
    if not os.path.exists(path):
        print(f"\n--- {label} --- SKIPPED")
        continue

    print(f"\n--- {label} ---")
    try:
        trt_bb = TRTBackbone(engine_path=path, device=device, pos_encoding_module=pos_module)
    except Exception as e:
        print(f"  SKIPPED: {e}")
        continue

    # Accuracy
    with torch.inference_mode():
        trt_out = trt_bb.forward_image(tensor)
    trt_fpn = trt_out["backbone_fpn"]

    for i in range(3):
        cos = torch.nn.functional.cosine_similarity(
            pt_fpn[i].float().flatten().unsqueeze(0),
            trt_fpn[i].float().flatten().unsqueeze(0),
        ).item()
        diff = (pt_fpn[i].float() - trt_fpn[i].float()).abs().max().item()
        print(f"  FPN[{i}]: cosine={cos:.6f}, max_diff={diff:.4f}")

    # Speed
    with torch.inference_mode():
        for _ in range(N_WARMUP):
            trt_bb.forward_image(dummy)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(N_ITERS):
            trt_bb.forward_image(dummy)
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) / N_ITERS * 1000
    print(f"  Speed: {ms:.1f} ms/frame")

# PyTorch baselines
print("\n--- PyTorch FP32 ---")
with torch.inference_mode():
    for _ in range(N_WARMUP):
        backbone.forward_image(dummy)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N_ITERS):
        backbone.forward_image(dummy)
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / N_ITERS * 1000
print(f"  Speed: {ms:.1f} ms/frame")

print("\n--- PyTorch FP16 autocast ---")
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
    for _ in range(N_WARMUP):
        backbone.forward_image(dummy)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N_ITERS):
        backbone.forward_image(dummy)
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / N_ITERS * 1000
print(f"  Speed: {ms:.1f} ms/frame")

print("\nDone!")
