#!/usr/bin/env python3
"""Analyze the ONNX graph of a single exported ViT block.

Previous test showed 1 real block gives TRT FP16 cos=0.997 vs 1.000 for synthetic.
This script exports 1 block and analyzes the ONNX ops to find what's different.
Also tests stripping individual op types to identify the culprit.
"""

import sys
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import onnx
from onnx import TensorProto, helper, numpy_helper

from sam3.model_builder import build_sam3_image_model
from sam3.trt.rope_onnx import patch_rope_for_export, unpatch_rope
from sam3.model.vitdet import get_abs_pos

import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)  # VERBOSE to see kernel decisions
DEVICE = "cuda"


class SingleBlockWrapper(nn.Module):
    def __init__(self, trunk, block_idx=0):
        super().__init__()
        self.block = trunk.blocks[block_idx]

    def forward(self, x):
        return self.block(x)


def export_single_block(model, block_idx=0):
    """Export one block and return ONNX path."""
    trunk = model.backbone.vision_backbone.trunk

    # Get real input
    dummy = torch.randn(1, 3, 1008, 1008, device=DEVICE)
    with torch.inference_mode():
        x = trunk.patch_embed(dummy)
        h, w = x.shape[1], x.shape[2]
        if trunk.pos_embed is not None:
            x = x + get_abs_pos(trunk.pos_embed, trunk.pretrain_use_cls_token,
                                (h, w), trunk.retain_cls_token, tiling=trunk.tile_abs_pos)
        if hasattr(trunk, 'ln_pre') and trunk.ln_pre is not None:
            x = trunk.ln_pre(x)

        # Run preceding blocks
        for i in range(block_idx):
            x = trunk.blocks[i](x)

    wrapper = SingleBlockWrapper(trunk, block_idx)
    onnx_path = f"test_block{block_idx}.onnx"

    patch_rope_for_export(model.backbone)
    with torch.inference_mode():
        torch.onnx.export(
            wrapper, (x,), onnx_path,
            input_names=["tokens"],
            output_names=["output"],
            opset_version=17,
            do_constant_folding=True,
        )
    unpatch_rope(model.backbone)

    return onnx_path, x


def analyze_onnx(onnx_path):
    """Analyze ONNX graph structure."""
    model = onnx.load(onnx_path)
    graph = model.graph

    op_counts = Counter()
    op_details = {}
    for node in graph.node:
        op_counts[node.op_type] += 1
        if node.op_type not in op_details:
            op_details[node.op_type] = []
        op_details[node.op_type].append(node.name or f"{node.op_type}_{len(op_details[node.op_type])}")

    print(f"\n  ONNX graph for {onnx_path}:")
    print(f"  Total nodes: {len(graph.node)}")
    print(f"  Op counts:")
    for op, count in sorted(op_counts.items(), key=lambda x: -x[1]):
        print(f"    {op:30s}: {count}")

    # Print all nodes in order
    print(f"\n  Full graph (in execution order):")
    for i, node in enumerate(graph.node):
        inputs = [f"{inp}" for inp in node.input[:3]]
        outputs = [f"{out}" for out in node.output[:2]]
        attrs = ""
        for attr in node.attribute:
            if attr.name in ("perm", "axis", "axes"):
                if attr.ints:
                    attrs += f" {attr.name}={list(attr.ints)}"
                elif attr.i:
                    attrs += f" {attr.name}={attr.i}"
        print(f"  {i:4d}. {node.op_type:25s} | in={inputs} | out={outputs}{attrs}")

    return model


def build_and_test_trt(onnx_path, x_input, label="", verbose_log=False):
    """Build TRT FP16 engine and test."""
    log_level = trt.Logger.VERBOSE if verbose_log else trt.Logger.WARNING
    logger = trt.Logger(log_level)

    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"  Parse error: {parser.get_error(i)}")
            return None

    # Print TRT layer types
    layer_types = Counter()
    for i in range(network.num_layers):
        layer = network.get_layer(i)
        layer_types[str(layer.type).split(".")[-1]] += 1

    if not verbose_log:
        print(f"\n  TRT layers for {label}:")
        for lt, count in sorted(layer_types.items(), key=lambda x: -x[1]):
            print(f"    {lt:30s}: {count}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
    config.set_flag(trt.BuilderFlag.FP16)

    engine_bytes = builder.build_serialized_network(network, config)
    if engine_bytes is None:
        return None

    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_bytes)
    ctx = engine.create_execution_context()

    d_in = x_input.float().contiguous().cuda()
    out_name = engine.get_tensor_name(engine.num_io_tensors - 1)
    out_shape = ctx.get_tensor_shape(out_name)
    d_out = torch.empty(list(out_shape), dtype=torch.float32, device="cuda")

    ctx.set_tensor_address(engine.get_tensor_name(0), d_in.data_ptr())
    ctx.set_tensor_address(out_name, d_out.data_ptr())
    ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()

    return d_out


def test_trt_with_precision_control(onnx_path, x_input, fp32_layer_types):
    """Build TRT engine with specific layer types forced to FP32."""
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            return None

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
    config.set_flag(trt.BuilderFlag.FP16)
    config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)

    fp32_count = 0
    for i in range(network.num_layers):
        layer = network.get_layer(i)
        type_name = str(layer.type).split(".")[-1]
        if type_name in fp32_layer_types:
            layer.precision = trt.float32
            layer.set_output_type(0, trt.float32)
            fp32_count += 1

    engine_bytes = builder.build_serialized_network(network, config)
    if engine_bytes is None:
        return None

    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_bytes)
    ctx = engine.create_execution_context()

    d_in = x_input.float().contiguous().cuda()
    out_name = engine.get_tensor_name(engine.num_io_tensors - 1)
    out_shape = ctx.get_tensor_shape(out_name)
    d_out = torch.empty(list(out_shape), dtype=torch.float32, device="cuda")

    ctx.set_tensor_address(engine.get_tensor_name(0), d_in.data_ptr())
    ctx.set_tensor_address(out_name, d_out.data_ptr())
    ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()

    return d_out, fp32_count


def cosine_sim(a, b):
    return torch.nn.functional.cosine_similarity(
        a.float().flatten().unsqueeze(0),
        b.float().flatten().unsqueeze(0),
    ).item()


def main():
    print("Loading model...")
    model = build_sam3_image_model(
        device=DEVICE, checkpoint_path="sam3.pt", eval_mode=True,
    )

    # Export and analyze block 0
    print("\n" + "=" * 80)
    print("PART 1: ONNX graph analysis of block 0")
    print("=" * 80)
    onnx_path, x_input = export_single_block(model, block_idx=0)
    onnx_model = analyze_onnx(onnx_path)

    # Test in TRT
    print("\n" + "=" * 80)
    print("PART 2: TRT layer analysis")
    print("=" * 80)
    Y_fp32 = model.backbone.vision_backbone.trunk.blocks[0](x_input.float())
    Y_trt = build_and_test_trt(onnx_path, x_input, "block 0")

    if Y_trt is not None:
        cos = cosine_sim(Y_fp32, Y_trt)
        diff = (Y_fp32 - Y_trt).abs().max().item()
        print(f"\n  Block 0 TRT FP16: cos={cos:.6f}, max_diff={diff:.4f}")

    # Test forcing individual layer types to FP32
    print("\n" + "=" * 80)
    print("PART 3: Which layer types need FP32? (single block)")
    print("=" * 80)
    print(f"  {'FP32 layers':>35s} | {'Cosine':>10s} | {'FP32 count':>10s}")
    print("  " + "-" * 65)

    for fp32_types in [
        [],  # Pure FP16
        ["MATRIX_MULTIPLY"],
        ["SOFTMAX"],
        ["ELEMENTWISE"],
        ["NORMALIZATION"],
        ["MATRIX_MULTIPLY", "SOFTMAX"],
        ["MATRIX_MULTIPLY", "NORMALIZATION"],
        ["MATRIX_MULTIPLY", "SOFTMAX", "NORMALIZATION"],
        ["POINTWISE"],
        ["REDUCE"],
        ["SHUFFLE"],
    ]:
        label = "+".join(fp32_types) if fp32_types else "pure FP16"
        result = test_trt_with_precision_control(onnx_path, x_input, set(fp32_types))
        if result is not None:
            Y_mixed, count = result
            cos = cosine_sim(Y_fp32, Y_mixed)
            print(f"  {label:>35s} | {cos:>10.6f} | {count:>10d}")

    # Also test block 7 (first global attention)
    print("\n" + "=" * 80)
    print("PART 4: Block 7 (first global attention) analysis")
    print("=" * 80)
    onnx_path_7, x_input_7 = export_single_block(model, block_idx=7)

    # Get FP32 ref for block 7
    trunk = model.backbone.vision_backbone.trunk
    with torch.inference_mode():
        x_7 = x_input_7.float()
        Y_fp32_7 = trunk.blocks[7](x_7)

    Y_trt_7 = build_and_test_trt(onnx_path_7, x_input_7, "block 7")
    if Y_trt_7 is not None:
        cos = cosine_sim(Y_fp32_7, Y_trt_7)
        print(f"\n  Block 7 TRT FP16: cos={cos:.6f}")

    print(f"\n  {'FP32 layers':>35s} | {'Cosine':>10s} | {'FP32 count':>10s}")
    print("  " + "-" * 65)

    for fp32_types in [
        [],
        ["MATRIX_MULTIPLY"],
        ["SOFTMAX"],
        ["MATRIX_MULTIPLY", "SOFTMAX"],
        ["ELEMENTWISE"],
        ["POINTWISE"],
    ]:
        label = "+".join(fp32_types) if fp32_types else "pure FP16"
        result = test_trt_with_precision_control(onnx_path_7, x_input_7, set(fp32_types))
        if result is not None:
            Y_mixed, count = result
            cos = cosine_sim(Y_fp32_7, Y_mixed)
            print(f"  {label:>35s} | {cos:>10.6f} | {count:>10d}")

    # Cleanup
    Path(onnx_path).unlink(missing_ok=True)
    Path(onnx_path_7).unlink(missing_ok=True)

    print("\nDone!")


if __name__ == "__main__":
    main()
