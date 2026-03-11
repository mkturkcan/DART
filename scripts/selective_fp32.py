#!/usr/bin/env python3
"""Build TRT engine with selective FP32: only attention MatMuls + Softmax.

Identifies attention MatMuls (Q@K^T and attn@V) by checking if both inputs
are dynamic (not constant weights). Projection/MLP MatMuls have one constant
input (weight tensor) and stay in FP16.
"""

import time
import torch
import tensorrt as trt
from pathlib import Path
from PIL import Image
from torchvision.transforms import v2

from sam3.model_builder import build_sam3_image_model
from sam3.trt.trt_backbone import TRTBackbone

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def build_selective_fp32(onnx_path, engine_path, opt_level=3):
    """Build engine where only attention MatMuls and Softmax are FP32."""
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
        config.builder_optimization_level = opt_level

    for flag_name in ("OBEY_PRECISION_CONSTRAINTS", "PREFER_PRECISION_CONSTRAINTS", "STRICT_TYPES"):
        if hasattr(trt.BuilderFlag, flag_name):
            config.set_flag(getattr(trt.BuilderFlag, flag_name))
            break

    # Skip types
    skip_types = set()
    for name in ("SHAPE", "CONSTANT", "IDENTITY", "SHUFFLE", "GATHER",
                 "SLICE", "SQUEEZE", "UNSQUEEZE", "CONCATENATION", "CONDITION",
                 "CAST", "ASSERTION", "FILL", "SCATTER", "RESIZE",
                 "NON_ZERO", "ONE_HOT", "GRID_SAMPLE"):
        if hasattr(trt.LayerType, name):
            skip_types.add(getattr(trt.LayerType, name))

    # Softmax type
    softmax_type = getattr(trt.LayerType, "SOFTMAX", None)
    matmul_type = getattr(trt.LayerType, "MATRIX_MULTIPLY", None)

    # Collect all layer info
    type_counts = {}
    fp32_count = 0
    fp16_count = 0
    skip_count = 0
    attn_matmul = 0
    proj_matmul = 0

    for i in range(network.num_layers):
        layer = network.get_layer(i)
        t = str(layer.type)
        type_counts[t] = type_counts.get(t, 0) + 1

        if layer.type in skip_types:
            skip_count += 1
            continue

        # Force Softmax to FP32 (attention numerics)
        if layer.type == softmax_type:
            layer.precision = trt.float32
            for j in range(layer.num_outputs):
                layer.set_output_type(j, trt.float32)
            fp32_count += 1
            continue

        # For MatMul, check if it's attention (both inputs dynamic) or projection
        if layer.type == matmul_type:
            # Check if either input is a constant (weight)
            is_projection = False
            for inp_idx in range(layer.num_inputs):
                inp = layer.get_input(inp_idx)
                if inp is not None and inp.is_shape_tensor:
                    continue
                # Check if this input comes from a Constant layer
                # TRT doesn't give us easy access to check this.
                # Instead, check the tensor name for weight-like patterns.
                if inp is not None:
                    name = inp.name
                    # Weight tensors typically have names like "onnx::MatMul_XXX" (constants)
                    # or include "weight" in the name
                    if "onnx::MatMul" in name or "weight" in name.lower():
                        is_projection = True
                        break

            if not is_projection:
                # Attention MatMul - force FP32
                layer.precision = trt.float32
                for j in range(layer.num_outputs):
                    layer.set_output_type(j, trt.float32)
                fp32_count += 1
                attn_matmul += 1
            else:
                # Projection/MLP MatMul - keep FP16
                layer.precision = trt.float16
                for j in range(layer.num_outputs):
                    layer.set_output_type(j, trt.float16)
                fp16_count += 1
                proj_matmul += 1
            continue

        # Everything else: let FP16 flag handle it
        fp16_count += 1

    print(f"  Network layers: {network.num_layers}")
    print(f"  Selective FP32: {fp32_count} FP32, {fp16_count} FP16, {skip_count} skip")
    print(f"  Attention MatMuls: {attn_matmul} FP32, Projection MatMuls: {proj_matmul} FP16")

    print("  Building engine...")
    t0 = time.perf_counter()
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("Engine build failed")
    print(f"  Built in {time.perf_counter() - t0:.1f}s")

    with open(engine_path, "wb") as f:
        f.write(serialized)

    size_mb = Path(engine_path).stat().st_size / (1024 * 1024)
    print(f"  Saved: {engine_path} ({size_mb:.1f} MB)")
    return engine_path


def benchmark(engine_path, backbone, dummy, n_warmup=5, n_iters=50):
    pos_module = backbone.vision_backbone.position_encoding
    trt_bb = TRTBackbone(engine_path=engine_path, device="cuda", pos_encoding_module=pos_module)

    with torch.inference_mode():
        # Get PyTorch reference
        pt_out = backbone.forward_image(dummy)
        pt_fpn = pt_out["backbone_fpn"]

        # Get TRT output
        trt_out = trt_bb.forward_image(dummy)
        trt_fpn = trt_out["backbone_fpn"]

        # Accuracy
        for i in range(3):
            cos = torch.nn.functional.cosine_similarity(
                pt_fpn[i].float().flatten().unsqueeze(0),
                trt_fpn[i].float().flatten().unsqueeze(0),
            ).item()
            print(f"  FPN[{i}]: cosine={cos:.6f}")

        # Speed
        for _ in range(n_warmup):
            trt_bb.forward_image(dummy)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_iters):
            trt_bb.forward_image(dummy)
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) / n_iters * 1000
        print(f"  Speed: {ms:.1f} ms/frame")


def main():
    device = "cuda"

    print("Loading model...")
    model = build_sam3_image_model(
        device=device, checkpoint_path="sam3.pt", eval_mode=True,
    )
    backbone = model.backbone
    dummy = torch.randn(1, 3, 1008, 1008, device=device)

    engine_path = "backbone_selective_fp32.engine"
    build_selective_fp32("backbone.onnx", engine_path)

    print("\n--- Selective FP32 Engine ---")
    benchmark(engine_path, backbone, dummy)


if __name__ == "__main__":
    main()
