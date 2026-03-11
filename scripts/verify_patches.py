#!/usr/bin/env python3
"""Verify that RoPE + SDPA patches produce same output as original in PyTorch."""

import torch
from sam3.model_builder import build_sam3_image_model
from sam3.trt.rope_onnx import (
    patch_rope_for_export, unpatch_rope,
    patch_sdpa_for_export, unpatch_sdpa,
)

device = "cuda"

print("Loading model...")
model = build_sam3_image_model(
    device=device, checkpoint_path="sam3.pt", eval_mode=True,
)
backbone = model.backbone

dummy = torch.randn(1, 3, 1008, 1008, device=device)

# Run original
print("Running original backbone...")
with torch.inference_mode():
    orig_out = backbone.forward_image(dummy)
orig_fpn = orig_out["backbone_fpn"]

# Test 1: SDPA patch only (no RoPE patch)
print("\n=== Test 1: SDPA patch only ===")
patch_sdpa_for_export(backbone)
with torch.inference_mode():
    sdpa_out = backbone.forward_image(dummy)
sdpa_fpn = sdpa_out["backbone_fpn"]
unpatch_sdpa(backbone)

for i in range(3):
    diff = (orig_fpn[i].float() - sdpa_fpn[i].float()).abs()
    cos = torch.nn.functional.cosine_similarity(
        orig_fpn[i].float().flatten().unsqueeze(0),
        sdpa_fpn[i].float().flatten().unsqueeze(0),
    )
    print(f"  FPN[{i}]: max_diff={diff.max().item():.2e}, cosine={cos.item():.8f}")

# Test 2: RoPE patch only
print("\n=== Test 2: RoPE patch only ===")
patch_rope_for_export(backbone)
with torch.inference_mode():
    rope_out = backbone.forward_image(dummy)
rope_fpn = rope_out["backbone_fpn"]
unpatch_rope(backbone)

for i in range(3):
    diff = (orig_fpn[i].float() - rope_fpn[i].float()).abs()
    cos = torch.nn.functional.cosine_similarity(
        orig_fpn[i].float().flatten().unsqueeze(0),
        rope_fpn[i].float().flatten().unsqueeze(0),
    )
    print(f"  FPN[{i}]: max_diff={diff.max().item():.2e}, cosine={cos.item():.8f}")

# Test 3: Both patches
print("\n=== Test 3: RoPE + SDPA patches ===")
patch_rope_for_export(backbone)
patch_sdpa_for_export(backbone)
with torch.inference_mode():
    both_out = backbone.forward_image(dummy)
both_fpn = both_out["backbone_fpn"]
unpatch_sdpa(backbone)
unpatch_rope(backbone)

for i in range(3):
    diff = (orig_fpn[i].float() - both_fpn[i].float()).abs()
    cos = torch.nn.functional.cosine_similarity(
        orig_fpn[i].float().flatten().unsqueeze(0),
        both_fpn[i].float().flatten().unsqueeze(0),
    )
    print(f"  FPN[{i}]: max_diff={diff.max().item():.2e}, cosine={cos.item():.8f}")

# Verify original is restored
print("\n=== Verify original restored ===")
with torch.inference_mode():
    check_out = backbone.forward_image(dummy)
check_fpn = check_out["backbone_fpn"]
for i in range(3):
    diff = (orig_fpn[i] - check_fpn[i]).abs().max().item()
    print(f"  FPN[{i}]: max_diff={diff:.2e} (should be 0)")
