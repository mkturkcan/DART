#!/usr/bin/env python3
"""Test which TRT layer types need FP32 for a single exported ViT block.

Previous finding: 1 real exported block gives TRT FP16 cos=0.997 vs 1.000 synthetic.
This tests forcing individual layer types to FP32 to find the culprit.
"""

import sys
import torch
import torch.nn as nn
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sam3.model_builder import build_sam3_image_model
from sam3.trt.rope_onnx import patch_rope_for_export, unpatch_rope
from sam3.model.vitdet import get_abs_pos

import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
DEVICE = "cuda"


class BlocksWrapper(nn.Module):
    def __init__(self, trunk, num_blocks):
        super().__init__()
        self.blocks = nn.ModuleList(trunk.blocks[:num_blocks])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


def get_trunk_input(model):
    """Get real trunk input (after patch_embed + pos)."""
    trunk = model.backbone.vision_backbone.trunk
    dummy = torch.randn(1, 3, 1008, 1008, device=DEVICE)
    with torch.inference_mode():
        x = trunk.patch_embed(dummy)
        h, w = x.shape[1], x.shape[2]
        if trunk.pos_embed is not None:
            x = x + get_abs_pos(trunk.pos_embed, trunk.pretrain_use_cls_token,
                                (h, w), trunk.retain_cls_token, tiling=trunk.tile_abs_pos)
        if hasattr(trunk, 'ln_pre') and trunk.ln_pre is not None:
            x = trunk.ln_pre(x)
    return x


def export_blocks(model, num_blocks, onnx_path):
    """Export N blocks."""
    trunk = model.backbone.vision_backbone.trunk
    x_input = get_trunk_input(model)

    wrapper = BlocksWrapper(trunk, num_blocks)
    patch_rope_for_export(model.backbone)
    with torch.inference_mode():
        torch.onnx.export(
            wrapper, (x_input,), onnx_path,
            input_names=["tokens"], output_names=["output"],
            opset_version=17, do_constant_folding=True,
        )
    unpatch_rope(model.backbone)
    return x_input


def build_trt(onnx_path, fp32_types=None, fp32_names=None):
    """Build TRT engine, optionally forcing layers to FP32."""
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            raise RuntimeError("Parse failed")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
    config.set_flag(trt.BuilderFlag.FP16)

    if fp32_types or fp32_names:
        config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
        fp32_count = 0
        for i in range(network.num_layers):
            layer = network.get_layer(i)
            type_name = str(layer.type).split(".")[-1]
            force = False
            if fp32_types and type_name in fp32_types:
                force = True
            if fp32_names:
                for name_pattern in fp32_names:
                    if name_pattern in layer.name:
                        force = True
            if force:
                layer.precision = trt.float32
                layer.set_output_type(0, trt.float32)
                fp32_count += 1
    else:
        fp32_count = 0

    engine_bytes = builder.build_serialized_network(network, config)
    if engine_bytes is None:
        raise RuntimeError("Build failed")

    return engine_bytes, fp32_count, network.num_layers


def run_trt(engine_bytes, x_input):
    """Run TRT engine."""
    runtime = trt.Runtime(TRT_LOGGER)
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


def get_layer_info(onnx_path):
    """Get TRT layer types and names."""
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    with open(onnx_path, "rb") as f:
        parser.parse(f.read())

    types = Counter()
    layers_by_type = {}
    for i in range(network.num_layers):
        layer = network.get_layer(i)
        t = str(layer.type).split(".")[-1]
        types[t] += 1
        if t not in layers_by_type:
            layers_by_type[t] = []
        layers_by_type[t].append(layer.name)

    return types, layers_by_type


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
    trunk = model.backbone.vision_backbone.trunk

    # ==== TEST SINGLE BLOCK ====
    print("\n" + "=" * 80)
    print("TEST 1: Single block (block 0) - which layer types need FP32?")
    print("=" * 80)

    onnx_1 = "test_1block.onnx"
    x_input = export_blocks(model, 1, onnx_1)

    # FP32 reference
    with torch.inference_mode():
        Y_fp32 = trunk.blocks[0](x_input.float())

    # Layer info
    types, layers = get_layer_info(onnx_1)
    print(f"\n  TRT layer types ({sum(types.values())} total):")
    for t, c in sorted(types.items(), key=lambda x: -x[1]):
        print(f"    {t:30s}: {c}")

    # Test each layer type forced to FP32
    print(f"\n  {'FP32 layer type(s)':>40s} | {'Cosine':>10s} | {'max_diff':>10s} | {'FP32/Total':>12s}")
    print("  " + "-" * 85)

    tests = [
        (set(), "pure FP16"),
        ({"MATRIX_MULTIPLY"}, "MATRIX_MULTIPLY"),
        ({"SOFTMAX"}, "SOFTMAX"),
        ({"ELEMENTWISE"}, "ELEMENTWISE"),
        ({"NORMALIZATION"}, "NORMALIZATION"),
        ({"POINTWISE"}, "POINTWISE"),
        ({"REDUCE"}, "REDUCE"),
        ({"SHUFFLE"}, "SHUFFLE"),
        ({"MATRIX_MULTIPLY", "SOFTMAX"}, "MATMUL+SOFTMAX"),
        ({"MATRIX_MULTIPLY", "POINTWISE"}, "MATMUL+POINTWISE"),
        ({"MATRIX_MULTIPLY", "ELEMENTWISE"}, "MATMUL+ELEMENTWISE"),
        ({"POINTWISE", "ELEMENTWISE"}, "POINTWISE+ELEMENTWISE"),
        ({"POINTWISE", "SOFTMAX"}, "POINTWISE+SOFTMAX"),
        ({"MATRIX_MULTIPLY", "SOFTMAX", "POINTWISE"}, "MATMUL+SOFTMAX+POINTWISE"),
    ]

    for fp32_set, label in tests:
        try:
            eng, n_fp32, n_total = build_trt(onnx_1, fp32_types=fp32_set)
            Y = run_trt(eng, x_input)
            cos = cosine_sim(Y_fp32, Y)
            diff = (Y_fp32 - Y).abs().max().item()
            print(f"  {label:>40s} | {cos:>10.6f} | {diff:>10.4f} | {n_fp32:>4d}/{n_total:<4d}")
        except Exception as e:
            print(f"  {label:>40s} | FAILED: {e}")

    # ==== TEST 8 BLOCKS ====
    print("\n" + "=" * 80)
    print("TEST 2: 8 blocks - which layer types need FP32?")
    print("=" * 80)

    onnx_8 = "test_8blocks.onnx"
    x_input = export_blocks(model, 8, onnx_8)

    with torch.inference_mode():
        Y_fp32 = x_input.float()
        for i in range(8):
            Y_fp32 = trunk.blocks[i](Y_fp32)

    types, _ = get_layer_info(onnx_8)
    print(f"\n  TRT layer types ({sum(types.values())} total):")
    for t, c in sorted(types.items(), key=lambda x: -x[1])[:10]:
        print(f"    {t:30s}: {c}")

    print(f"\n  {'FP32 layer type(s)':>40s} | {'Cosine':>10s} | {'max_diff':>10s} | {'FP32/Total':>12s}")
    print("  " + "-" * 85)

    for fp32_set, label in [
        (set(), "pure FP16"),
        ({"MATRIX_MULTIPLY"}, "MATRIX_MULTIPLY"),
        ({"POINTWISE"}, "POINTWISE"),
        ({"MATRIX_MULTIPLY", "SOFTMAX"}, "MATMUL+SOFTMAX"),
        ({"MATRIX_MULTIPLY", "POINTWISE"}, "MATMUL+POINTWISE"),
        ({"POINTWISE", "SOFTMAX"}, "POINTWISE+SOFTMAX"),
        ({"MATRIX_MULTIPLY", "SOFTMAX", "POINTWISE"}, "MATMUL+SOFTMAX+POINTWISE"),
    ]:
        try:
            eng, n_fp32, n_total = build_trt(onnx_8, fp32_types=fp32_set)
            Y = run_trt(eng, x_input)
            cos = cosine_sim(Y_fp32, Y)
            diff = (Y_fp32 - Y).abs().max().item()
            print(f"  {label:>40s} | {cos:>10.6f} | {diff:>10.4f} | {n_fp32:>4d}/{n_total:<4d}")
        except Exception as e:
            print(f"  {label:>40s} | FAILED: {e}")

    # ==== TEST: POINTWISE layer names ====
    print("\n" + "=" * 80)
    print("TEST 3: What are POINTWISE layers? (block 0)")
    print("=" * 80)

    _, layers = get_layer_info(onnx_1)
    if "POINTWISE" in layers:
        print(f"\n  POINTWISE layers ({len(layers['POINTWISE'])}):")
        for name in layers["POINTWISE"]:
            print(f"    {name}")
    if "ELEMENTWISE" in layers:
        print(f"\n  ELEMENTWISE layers ({len(layers['ELEMENTWISE'])}):")
        for name in layers["ELEMENTWISE"][:20]:
            print(f"    {name}")

    # Cleanup
    Path(onnx_1).unlink(missing_ok=True)
    Path(onnx_8).unlink(missing_ok=True)

    print("\nDone!")


if __name__ == "__main__":
    main()
