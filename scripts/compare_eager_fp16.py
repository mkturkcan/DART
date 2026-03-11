#!/usr/bin/env python3
"""Minimal TRT FP16 vs PyTorch backbone comparison (no ORT dependency)."""

import argparse
import torch
import numpy as np
from PIL import Image
from torchvision.transforms import v2

from sam3.model_builder import build_sam3_image_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--trt", required=True, help="TRT engine path")
    parser.add_argument("--image", default=None)
    parser.add_argument("--imgsz", type=int, default=1008)
    args = parser.parse_args()

    device = "cuda"

    # Load PyTorch model
    print("Loading PyTorch model...")
    model = build_sam3_image_model(
        device=device,
        checkpoint_path=args.checkpoint,
        eval_mode=True,
    )
    backbone = model.backbone

    # Load TRT engine
    print("Loading TRT engine...")
    from sam3.trt.trt_backbone import TRTBackbone

    pos_module = backbone.vision_backbone.position_encoding
    trt_bb = TRTBackbone(
        engine_path=args.trt,
        device=device,
        pos_encoding_module=pos_module,
    )

    # Prepare input
    transform = v2.Compose([
        v2.ToDtype(torch.uint8, scale=True),
        v2.Resize(size=(args.imgsz, args.imgsz)),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    if args.image:
        print(f"Loading image: {args.image}")
        img = Image.open(args.image).convert("RGB")
        img = img.resize((args.imgsz, args.imgsz), Image.BILINEAR)
        tensor = v2.functional.to_image(img).to(device)
        tensor = transform(tensor).unsqueeze(0)
    else:
        tensor = torch.randn(1, 3, args.imgsz, args.imgsz, device=device)

    print(f"Input: {tensor.shape}, range=[{tensor.min().item():.3f}, {tensor.max().item():.3f}]")

    # Run PyTorch
    print("\nRunning PyTorch backbone...")
    with torch.inference_mode():
        pt_out = backbone.forward_image(tensor)
    pt_fpn = pt_out["backbone_fpn"]

    # Run TRT
    print("Running TRT backbone...")
    with torch.inference_mode():
        trt_out = trt_bb.forward_image(tensor)
    trt_fpn = trt_out["backbone_fpn"]

    # Compare
    print("\n=== TRT FP16 vs PyTorch FP32 ===")
    for i in range(len(pt_fpn)):
        pt_f = pt_fpn[i].float()
        trt_f = trt_fpn[i].float()
        diff = (pt_f - trt_f).abs()
        cos = torch.nn.functional.cosine_similarity(
            pt_f.flatten().unsqueeze(0), trt_f.flatten().unsqueeze(0)
        )
        print(f"  FPN[{i}]: cosine={cos.item():.6f}, "
              f"max_diff={diff.max().item():.4f}, "
              f"mean_diff={diff.mean().item():.6f}, "
              f"pt_std={pt_f.std().item():.4f}, trt_std={trt_f.std().item():.4f}")

    # Final verdict
    cos_last = torch.nn.functional.cosine_similarity(
        pt_fpn[-1].float().flatten().unsqueeze(0),
        trt_fpn[-1].float().flatten().unsqueeze(0),
    )
    print(f"\nFINAL COSINE (FPN[-1]): {cos_last.item():.6f}")
    if cos_last.item() > 0.99:
        print("RESULT: FP16 TRT WORKS!")
    elif cos_last.item() > 0.9:
        print("RESULT: Marginal - needs investigation")
    else:
        print("RESULT: FP16 TRT BROKEN")


if __name__ == "__main__":
    main()
