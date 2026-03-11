#!/usr/bin/env python3
"""Test BF16 MatMul layers in TRT to fix FP16 overflow without FP32 penalty.

BF16 has same exponent range as FP32 (no overflow) but uses tensor cores.
Tests: pure BF16, FP16+BF16 mixed (MatMul in BF16, rest in FP16).
"""

import sys
import time
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.INFO)
ONNX_PATH = "backbone.onnx"
DEVICE = "cuda"


def build_engine_bf16_mixed(onnx_path, output_path):
    """Build engine: FP16 for most layers, BF16 for MatMul+Softmax."""
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, TRT_LOGGER)

    print(f"Parsing {onnx_path}...")
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"  Error: {parser.get_error(i)}")
            raise RuntimeError("ONNX parse failed")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)

    # Enable both FP16 and BF16
    config.set_flag(trt.BuilderFlag.FP16)
    config.set_flag(trt.BuilderFlag.BF16)
    config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)

    fp32_types = set()
    for name in ("MATRIX_MULTIPLY", "SOFTMAX"):
        if hasattr(trt.LayerType, name):
            fp32_types.add(getattr(trt.LayerType, name))

    skip_types = set()
    for name in (
        "SHAPE", "CONSTANT", "IDENTITY", "SHUFFLE", "GATHER",
        "SLICE", "SQUEEZE", "UNSQUEEZE", "CONCATENATION", "CONDITION",
        "CAST", "ASSERTION", "FILL", "SCATTER", "RESIZE",
        "NON_ZERO", "ONE_HOT", "GRID_SAMPLE",
    ):
        if hasattr(trt.LayerType, name):
            skip_types.add(getattr(trt.LayerType, name))

    bf16_count = 0
    fp16_count = 0
    skip_count = 0
    for i in range(network.num_layers):
        layer = network.get_layer(i)
        if layer.type in skip_types:
            skip_count += 1
            continue
        if layer.type in fp32_types:
            layer.precision = trt.bfloat16
            for j in range(layer.num_outputs):
                layer.set_output_type(j, trt.bfloat16)
            bf16_count += 1
        else:
            layer.precision = trt.float16
            for j in range(layer.num_outputs):
                layer.set_output_type(j, trt.float16)
            fp16_count += 1

    print(f"  Mixed: {bf16_count} BF16 (MatMul+Softmax) / {fp16_count} FP16 / {skip_count} skip")

    print("Building engine...")
    t0 = time.time()
    engine_bytes = builder.build_serialized_network(network, config)
    print(f"  Build time: {time.time()-t0:.0f}s")

    if engine_bytes is None:
        raise RuntimeError("Engine build failed")

    with open(output_path, "wb") as f:
        f.write(engine_bytes)
    print(f"  Saved: {output_path} ({Path(output_path).stat().st_size/1e6:.0f} MB)")
    return output_path


def build_engine_pure_bf16(onnx_path, output_path):
    """Build engine: BF16 only (no FP16)."""
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, TRT_LOGGER)

    print(f"Parsing {onnx_path}...")
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            raise RuntimeError("ONNX parse failed")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
    config.set_flag(trt.BuilderFlag.BF16)
    print("  Pure BF16 mode")

    print("Building engine...")
    t0 = time.time()
    engine_bytes = builder.build_serialized_network(network, config)
    print(f"  Build time: {time.time()-t0:.0f}s")

    if engine_bytes is None:
        raise RuntimeError("Engine build failed")

    with open(output_path, "wb") as f:
        f.write(engine_bytes)
    print(f"  Saved: {output_path} ({Path(output_path).stat().st_size/1e6:.0f} MB)")
    return output_path


def load_and_benchmark(engine_path, label, n_warmup=10, n_iters=100):
    """Load TRT engine and benchmark + compare vs PyTorch."""
    from sam3.model_builder import build_sam3_image_model
    from sam3.trt.trt_backbone import TRTBackbone

    # PyTorch reference
    print(f"\n=== {label} ===")
    print("Loading PyTorch model...")
    model = build_sam3_image_model(
        device=DEVICE, checkpoint_path="sam3.pt", eval_mode=True,
    )
    backbone = model.backbone

    dummy = torch.randn(1, 3, 1008, 1008, device=DEVICE)

    # PyTorch baseline
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        pt_out = backbone.forward_image(dummy)
    pt_fpn = pt_out["backbone_fpn"]

    # Load TRT
    print(f"Loading TRT engine: {engine_path}")
    pos_module = backbone.vision_backbone.position_encoding
    trt_bb = TRTBackbone(
        engine_path=engine_path,
        device=DEVICE,
        pos_encoding_module=pos_module,
    )

    # Accuracy
    with torch.inference_mode():
        trt_out = trt_bb.forward_image(dummy)
    trt_fpn = trt_out["backbone_fpn"]

    print("  Accuracy vs PyTorch:")
    for i in range(len(pt_fpn)):
        cos = torch.nn.functional.cosine_similarity(
            pt_fpn[i].float().flatten().unsqueeze(0),
            trt_fpn[i].float().flatten().unsqueeze(0),
        ).item()
        diff = (pt_fpn[i].float() - trt_fpn[i].float()).abs()
        print(f"    FPN[{i}]: cosine={cos:.6f}, max_diff={diff.max().item():.4f}")

    # Speed
    with torch.inference_mode():
        for _ in range(n_warmup):
            trt_bb.forward_image(dummy)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_iters):
            trt_bb.forward_image(dummy)
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) / n_iters * 1000
    print(f"  Speed: {ms:.1f} ms/frame")

    # Cleanup
    del trt_bb, model, backbone
    torch.cuda.empty_cache()
    return ms


def main():
    # 1. Pure BF16
    print("\n" + "=" * 60)
    print("Building PURE BF16 engine")
    print("=" * 60)
    try:
        build_engine_pure_bf16(ONNX_PATH, "backbone_bf16.engine")
        load_and_benchmark("backbone_bf16.engine", "Pure BF16")
    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback; traceback.print_exc()

    # 2. FP16 + BF16 mixed (MatMul/Softmax in BF16)
    print("\n" + "=" * 60)
    print("Building FP16 + BF16-mixed engine (MatMul+Softmax in BF16)")
    print("=" * 60)
    try:
        build_engine_bf16_mixed(ONNX_PATH, "backbone_fp16_bf16mix.engine")
        load_and_benchmark("backbone_fp16_bf16mix.engine", "FP16 + BF16 MatMul")
    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback; traceback.print_exc()

    print("\nDone!")


if __name__ == "__main__":
    main()
