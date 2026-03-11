#!/usr/bin/env python3
"""Test ONNX Runtime with TensorRT EP in FP16 vs PyTorch backbone.

This uses TRT under the hood but through ORT's wrapper, which may
handle FP16 precision conversion differently from standalone TRT engine builds.
"""

import argparse
import torch
import numpy as np
import onnxruntime as ort
from PIL import Image
from torchvision.transforms import v2

from sam3.model_builder import build_sam3_image_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--onnx", required=True, help="ONNX model path")
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

    # Prepare input
    if args.image:
        print(f"Loading image: {args.image}")
        img = Image.open(args.image).convert("RGB")
        transform = v2.Compose([
            v2.ToDtype(torch.uint8, scale=True),
            v2.Resize(size=(args.imgsz, args.imgsz)),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        tensor = v2.functional.to_image(img).to(device)
        tensor = transform(tensor).unsqueeze(0)
    else:
        tensor = torch.randn(1, 3, args.imgsz, args.imgsz, device=device)

    # PyTorch reference
    print("Running PyTorch backbone...")
    with torch.inference_mode():
        pt_out = backbone.forward_image(tensor)
    pt_fpn = pt_out["backbone_fpn"]

    input_np = tensor.cpu().numpy()

    # Test 1: ORT with CUDA EP (baseline - should match PyTorch)
    print("\n--- ORT + CUDA EP (FP32) ---")
    sess_cuda = ort.InferenceSession(
        args.onnx,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    ort_cuda_out = sess_cuda.run(None, {"images": input_np})
    for i, arr in enumerate(ort_cuda_out):
        t = torch.from_numpy(arr).to(device)
        cos = torch.nn.functional.cosine_similarity(
            pt_fpn[i].float().flatten().unsqueeze(0),
            t.float().flatten().unsqueeze(0),
        )
        print(f"  FPN[{i}]: cosine={cos.item():.6f}, "
              f"range=[{arr.min():.4f}, {arr.max():.4f}], "
              f"std={arr.std():.4f}")

    # Test 2: ORT with TRT EP in FP16
    print("\n--- ORT + TensorRT EP (FP16) ---")
    trt_opts = {
        "trt_fp16_enable": "True",
        "trt_engine_cache_enable": "True",
        "trt_engine_cache_path": "./trt_cache",
    }
    try:
        sess_trt = ort.InferenceSession(
            args.onnx,
            providers=[
                ("TensorrtExecutionProvider", trt_opts),
                "CUDAExecutionProvider",
            ],
        )
        ort_trt_out = sess_trt.run(None, {"images": input_np})
        for i, arr in enumerate(ort_trt_out):
            t = torch.from_numpy(arr).to(device)
            cos = torch.nn.functional.cosine_similarity(
                pt_fpn[i].float().flatten().unsqueeze(0),
                t.float().flatten().unsqueeze(0),
            )
            print(f"  FPN[{i}]: cosine={cos.item():.6f}, "
                  f"range=[{arr.min():.4f}, {arr.max():.4f}], "
                  f"std={arr.std():.4f}")
    except Exception as e:
        print(f"  Failed: {e}")

    # Test 3: ORT with TRT EP in FP32 (control)
    print("\n--- ORT + TensorRT EP (FP32) ---")
    trt_opts_fp32 = {
        "trt_fp16_enable": "False",
        "trt_engine_cache_enable": "True",
        "trt_engine_cache_path": "./trt_cache_fp32",
    }
    try:
        sess_trt32 = ort.InferenceSession(
            args.onnx,
            providers=[
                ("TensorrtExecutionProvider", trt_opts_fp32),
                "CUDAExecutionProvider",
            ],
        )
        ort_trt32_out = sess_trt32.run(None, {"images": input_np})
        for i, arr in enumerate(ort_trt32_out):
            t = torch.from_numpy(arr).to(device)
            cos = torch.nn.functional.cosine_similarity(
                pt_fpn[i].float().flatten().unsqueeze(0),
                t.float().flatten().unsqueeze(0),
            )
            print(f"  FPN[{i}]: cosine={cos.item():.6f}, "
                  f"range=[{arr.min():.4f}, {arr.max():.4f}], "
                  f"std={arr.std():.4f}")
    except Exception as e:
        print(f"  Failed: {e}")


if __name__ == "__main__":
    main()
