#!/usr/bin/env python3
"""Find the minimal set of layer types that need FP32 for TRT FP16 accuracy.

Tests different combinations of FP32/FP16 layer assignments on the original
backbone.onnx (with SDPA) to find which layer types cause numerical issues.
"""

import argparse
import sys
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from torchvision.transforms import v2

import tensorrt as trt

from sam3.model_builder import build_sam3_image_model
from sam3.trt.trt_backbone import TRTBackbone

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def build_with_precision_map(onnx_path, engine_path, fp32_type_names, fp16_type_names):
    """Build engine with specific layer types forced to FP32 or FP16."""
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            raise RuntimeError("Failed to parse ONNX")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
    config.set_flag(trt.BuilderFlag.FP16)

    if hasattr(config, "builder_optimization_level"):
        config.builder_optimization_level = 3

    # Set precision constraint flag
    for flag_name in ("OBEY_PRECISION_CONSTRAINTS", "PREFER_PRECISION_CONSTRAINTS", "STRICT_TYPES"):
        if hasattr(trt.BuilderFlag, flag_name):
            config.set_flag(getattr(trt.BuilderFlag, flag_name))
            break

    # Resolve type names
    fp32_types = set()
    for name in fp32_type_names:
        if hasattr(trt.LayerType, name):
            fp32_types.add(getattr(trt.LayerType, name))

    fp16_types = set()
    for name in fp16_type_names:
        if hasattr(trt.LayerType, name):
            fp16_types.add(getattr(trt.LayerType, name))

    # Skip types (non-compute)
    skip_types = set()
    for name in ("SHAPE", "CONSTANT", "IDENTITY", "SHUFFLE", "GATHER",
                 "SLICE", "SQUEEZE", "UNSQUEEZE", "CONCATENATION", "CONDITION",
                 "CAST", "ASSERTION", "FILL", "SCATTER", "RESIZE",
                 "NON_ZERO", "ONE_HOT", "GRID_SAMPLE"):
        if hasattr(trt.LayerType, name):
            skip_types.add(getattr(trt.LayerType, name))

    fp32_count = 0
    fp16_count = 0
    for i in range(network.num_layers):
        layer = network.get_layer(i)
        if layer.type in skip_types:
            continue
        elif layer.type in fp32_types:
            layer.precision = trt.float32
            for j in range(layer.num_outputs):
                layer.set_output_type(j, trt.float32)
            fp32_count += 1
        elif layer.type in fp16_types:
            layer.precision = trt.float16
            for j in range(layer.num_outputs):
                layer.set_output_type(j, trt.float16)
            fp16_count += 1
        # else: let TRT decide

    print(f"    Precision: {fp32_count} FP32, {fp16_count} FP16")

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("Engine build failed")

    with open(engine_path, "wb") as f:
        f.write(serialized)

    return Path(engine_path).stat().st_size / (1024 * 1024)


def test_engine(engine_path, backbone, tensor):
    """Test engine and return cosine similarity for FPN[-1]."""
    pos_module = backbone.vision_backbone.position_encoding
    trt_bb = TRTBackbone(
        engine_path=engine_path,
        device="cuda",
        pos_encoding_module=pos_module,
    )

    with torch.inference_mode():
        pt_out = backbone.forward_image(tensor)
        trt_out = trt_bb.forward_image(tensor)

    pt_fpn = pt_out["backbone_fpn"]
    trt_fpn = trt_out["backbone_fpn"]

    cosines = []
    for i in range(3):
        cos = torch.nn.functional.cosine_similarity(
            pt_fpn[i].float().flatten().unsqueeze(0),
            trt_fpn[i].float().flatten().unsqueeze(0),
        ).item()
        cosines.append(cos)

    return cosines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", default="backbone.onnx")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", default=None)
    parser.add_argument("--imgsz", type=int, default=1008)
    args = parser.parse_args()

    device = "cuda"

    # Load model
    print("Loading model...")
    model = build_sam3_image_model(
        device=device, checkpoint_path=args.checkpoint, eval_mode=True,
    )
    backbone = model.backbone

    # Prepare input
    transform = v2.Compose([
        v2.ToDtype(torch.uint8, scale=True),
        v2.Resize(size=(args.imgsz, args.imgsz)),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    if args.image:
        img = Image.open(args.image).convert("RGB")
        img = img.resize((args.imgsz, args.imgsz), Image.BILINEAR)
        tensor = v2.functional.to_image(img).to(device)
        tensor = transform(tensor).unsqueeze(0)
    else:
        tensor = torch.randn(1, 3, args.imgsz, args.imgsz, device=device)

    # All compute layer types we might want to set
    all_compute_types = [
        "MATRIX_MULTIPLY",
        "SOFTMAX",
        "NORMALIZATION",
        "ELEMENTWISE",
        "REDUCE",
        "UNARY",
        "CONVOLUTION",
        "DECONVOLUTION",
    ]

    engine_path = "_test_mixed.engine"

    # Test 1: Pure FP16 (baseline - expected broken)
    print("\n=== Test 0: Pure FP16 (no precision constraints) ===")
    # Skip this - we know it's broken (cosine ~0.088)

    # Test 2: MatMul only in FP32
    print("\n=== Test 1: MATRIX_MULTIPLY in FP32, rest FP16 ===")
    fp32 = ["MATRIX_MULTIPLY"]
    fp16 = ["CONVOLUTION", "SOFTMAX", "NORMALIZATION", "ELEMENTWISE", "REDUCE", "UNARY"]
    build_with_precision_map(args.onnx, engine_path, fp32, fp16)
    cosines = test_engine(engine_path, backbone, tensor)
    print(f"    Cosines: {[f'{c:.6f}' for c in cosines]}")

    # Test 3: MatMul + Softmax in FP32
    print("\n=== Test 2: MATRIX_MULTIPLY + SOFTMAX in FP32 ===")
    fp32 = ["MATRIX_MULTIPLY", "SOFTMAX"]
    fp16 = ["CONVOLUTION", "NORMALIZATION", "ELEMENTWISE", "REDUCE", "UNARY"]
    build_with_precision_map(args.onnx, engine_path, fp32, fp16)
    cosines = test_engine(engine_path, backbone, tensor)
    print(f"    Cosines: {[f'{c:.6f}' for c in cosines]}")

    # Test 4: MatMul + Softmax + Norm in FP32
    print("\n=== Test 3: MATRIX_MULTIPLY + SOFTMAX + NORMALIZATION in FP32 ===")
    fp32 = ["MATRIX_MULTIPLY", "SOFTMAX", "NORMALIZATION"]
    fp16 = ["CONVOLUTION", "ELEMENTWISE", "REDUCE", "UNARY"]
    build_with_precision_map(args.onnx, engine_path, fp32, fp16)
    cosines = test_engine(engine_path, backbone, tensor)
    print(f"    Cosines: {[f'{c:.6f}' for c in cosines]}")

    # Test 5: Everything except Conv in FP32 (known working)
    print("\n=== Test 4: ALL except Conv in FP32 (control) ===")
    fp32 = ["MATRIX_MULTIPLY", "SOFTMAX", "NORMALIZATION", "ELEMENTWISE", "REDUCE", "UNARY"]
    fp16 = ["CONVOLUTION"]
    build_with_precision_map(args.onnx, engine_path, fp32, fp16)
    cosines = test_engine(engine_path, backbone, tensor)
    print(f"    Cosines: {[f'{c:.6f}' for c in cosines]}")

    # Test 6: Everything except Conv+MatMul in FP32
    print("\n=== Test 5: Softmax+Norm+Elementwise+Reduce+Unary FP32, Conv+MatMul FP16 ===")
    fp32 = ["SOFTMAX", "NORMALIZATION", "ELEMENTWISE", "REDUCE", "UNARY"]
    fp16 = ["CONVOLUTION", "MATRIX_MULTIPLY"]
    build_with_precision_map(args.onnx, engine_path, fp32, fp16)
    cosines = test_engine(engine_path, backbone, tensor)
    print(f"    Cosines: {[f'{c:.6f}' for c in cosines]}")

    # Clean up
    Path(engine_path).unlink(missing_ok=True)
    print("\nDone!")


if __name__ == "__main__":
    main()
