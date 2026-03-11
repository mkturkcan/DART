#!/usr/bin/env python3
"""Test selective FP32: only force attention MatMul to FP32, keep MLP in FP16.

Previous finding: MATMUL+SOFTMAX all FP32 gives cos=0.9998 at 128-148ms.
But MLP MatMul (fc1, fc2) is 2/6 of the FP32 MatMuls per block.
If we can keep MLP in FP16, we save ~40% of the FP32 overhead.

Also test: can we use TRT's FP16 with FP32 accumulation?
"""

import sys
import time
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


class BackboneWrapper(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, images):
        out = self.backbone.forward_image(images)
        fpn = out["backbone_fpn"]
        return fpn[0], fpn[1], fpn[2]


def export_backbone(model, onnx_path):
    """Export full backbone."""
    backbone = model.backbone
    patch_rope_for_export(backbone)
    wrapper = BackboneWrapper(backbone)
    dummy = torch.randn(1, 3, 1008, 1008, device=DEVICE)

    with torch.inference_mode():
        torch.onnx.export(
            wrapper, (dummy,), onnx_path,
            input_names=["image"],
            output_names=["fpn0", "fpn1", "fpn2"],
            opset_version=17, do_constant_folding=True,
        )

    unpatch_rope(backbone)
    print(f"  Exported: {Path(onnx_path).stat().st_size / 1e6:.0f} MB")


def build_with_selective_fp32(onnx_path, output_path, fp32_strategy, timing=True):
    """Build TRT engine with selective FP32 layers.

    Strategies:
    - "pure_fp16": no constraints
    - "all_matmul_softmax": all MATRIX_MULTIPLY + SOFTMAX to FP32
    - "attn_matmul_softmax": only attention MatMul + Softmax (skip MLP fc1, fc2)
    - "qkv_attn_softmax": only QKV projection + Q@K^T + attn@V + proj + Softmax
    - "qk_softmax": only Q@K^T + Softmax (minimal)
    """
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            raise RuntimeError("Parse failed")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
    config.set_flag(trt.BuilderFlag.FP16)

    # Classify layers by name
    layer_info = []
    for i in range(network.num_layers):
        layer = network.get_layer(i)
        type_name = str(layer.type).split(".")[-1]
        layer_info.append((i, type_name, layer.name, layer))

    fp32_count = 0
    total = network.num_layers

    if fp32_strategy == "pure_fp16":
        pass
    elif fp32_strategy == "all_matmul_softmax":
        config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
        for i, type_name, name, layer in layer_info:
            if type_name in ("MATRIX_MULTIPLY", "SOFTMAX"):
                layer.precision = trt.float32
                layer.set_output_type(0, trt.float32)
                fp32_count += 1
    elif fp32_strategy == "attn_matmul_softmax":
        # Force attention-related MatMul + Softmax, skip MLP fc1/fc2
        config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
        for i, type_name, name, layer in layer_info:
            if type_name == "SOFTMAX":
                layer.precision = trt.float32
                layer.set_output_type(0, trt.float32)
                fp32_count += 1
            elif type_name == "MATRIX_MULTIPLY":
                # Skip MLP layers
                is_mlp = "fc1" in name or "fc2" in name or "mlp" in name
                if not is_mlp:
                    layer.precision = trt.float32
                    layer.set_output_type(0, trt.float32)
                    fp32_count += 1
    elif fp32_strategy == "qkv_attn_softmax":
        # Only QKV + attention MatMul (Q@K^T, attn@V) + proj + Softmax
        # Skip MLP AND output projection
        config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
        for i, type_name, name, layer in layer_info:
            if type_name == "SOFTMAX":
                layer.precision = trt.float32
                layer.set_output_type(0, trt.float32)
                fp32_count += 1
            elif type_name == "MATRIX_MULTIPLY":
                is_mlp = "fc1" in name or "fc2" in name or "mlp" in name
                is_proj = "proj" in name and "qkv" not in name
                if not is_mlp and not is_proj:
                    layer.precision = trt.float32
                    layer.set_output_type(0, trt.float32)
                    fp32_count += 1
    elif fp32_strategy == "qk_softmax_only":
        # Minimal: only Q@K^T MatMul + Softmax
        config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
        for i, type_name, name, layer in layer_info:
            if type_name == "SOFTMAX":
                layer.precision = trt.float32
                layer.set_output_type(0, trt.float32)
                fp32_count += 1
            elif type_name == "MATRIX_MULTIPLY":
                # Only Q@K^T MatMul (not QKV proj, not output proj, not MLP)
                is_attn_score = ("MatMul" in name and
                                 "qkv" not in name and
                                 "proj" not in name and
                                 "fc1" not in name and
                                 "fc2" not in name and
                                 "mlp" not in name)
                if is_attn_score:
                    layer.precision = trt.float32
                    layer.set_output_type(0, trt.float32)
                    fp32_count += 1
    elif fp32_strategy == "softmax_only":
        config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
        for i, type_name, name, layer in layer_info:
            if type_name == "SOFTMAX":
                layer.precision = trt.float32
                layer.set_output_type(0, trt.float32)
                fp32_count += 1

    print(f"  Strategy: {fp32_strategy}, FP32 layers: {fp32_count}/{total}")

    t0 = time.time()
    engine_bytes = builder.build_serialized_network(network, config)
    build_time = time.time() - t0
    print(f"  Build time: {build_time:.0f}s")

    if engine_bytes is None:
        raise RuntimeError("Build failed")
    with open(output_path, "wb") as f:
        f.write(engine_bytes)
    return fp32_count


def test_engine(engine_path, model, dummy, label):
    """Test accuracy and speed."""
    from sam3.trt.trt_backbone import TRTBackbone

    backbone = model.backbone
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        pt_out = backbone.forward_image(dummy)
    pt_fpn = pt_out["backbone_fpn"]

    pos_module = backbone.vision_backbone.position_encoding
    trt_bb = TRTBackbone(engine_path=engine_path, device=DEVICE,
                         pos_encoding_module=pos_module)

    with torch.inference_mode():
        trt_out = trt_bb.forward_image(dummy)
    trt_fpn = trt_out["backbone_fpn"]

    cos = [
        torch.nn.functional.cosine_similarity(
            pt_fpn[i].float().flatten().unsqueeze(0),
            trt_fpn[i].float().flatten().unsqueeze(0),
        ).item()
        for i in range(len(pt_fpn))
    ]

    # Speed benchmark
    with torch.inference_mode():
        for _ in range(10):
            trt_bb.forward_image(dummy)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(50):
            trt_bb.forward_image(dummy)
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) / 50 * 1000

    status = "OK" if cos[-1] > 0.99 else "BROKEN" if cos[-1] < 0.5 else "DEGRADED"
    print(f"  {label:35s} | cos=[{cos[0]:.4f}, {cos[1]:.4f}, {cos[2]:.4f}] | {ms:.1f}ms | {status}")

    del trt_bb
    torch.cuda.empty_cache()
    return cos, ms


def main():
    print("Loading model...")
    model = build_sam3_image_model(
        device=DEVICE, checkpoint_path="sam3.pt", eval_mode=True,
    )
    dummy = torch.randn(1, 3, 1008, 1008, device=DEVICE)

    onnx_path = "backbone.onnx"
    if not Path(onnx_path).exists():
        print("\nExporting backbone...")
        export_backbone(model, onnx_path)

    # First, analyze MatMul layer names to understand which are attention vs MLP
    print("\n" + "=" * 80)
    print("ANALYSIS: MatMul layer names in TRT")
    print("=" * 80)

    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    with open(onnx_path, "rb") as f:
        parser.parse(f.read())

    matmul_names = []
    softmax_names = []
    for i in range(network.num_layers):
        layer = network.get_layer(i)
        type_name = str(layer.type).split(".")[-1]
        if type_name == "MATRIX_MULTIPLY":
            matmul_names.append(layer.name)
        elif type_name == "SOFTMAX":
            softmax_names.append(layer.name)

    print(f"\n  Total MATRIX_MULTIPLY: {len(matmul_names)}")
    print(f"  Total SOFTMAX: {len(softmax_names)}")

    # Categorize MatMul layers
    qkv_count = sum(1 for n in matmul_names if "qkv" in n)
    proj_count = sum(1 for n in matmul_names if "proj" in n and "qkv" not in n)
    fc1_count = sum(1 for n in matmul_names if "fc1" in n)
    fc2_count = sum(1 for n in matmul_names if "fc2" in n)
    attn_count = len(matmul_names) - qkv_count - proj_count - fc1_count - fc2_count

    print(f"\n  MatMul breakdown:")
    print(f"    QKV projection:  {qkv_count}")
    print(f"    Attention (Q@K^T, attn@V): {attn_count}")
    print(f"    Output projection: {proj_count}")
    print(f"    MLP fc1: {fc1_count}")
    print(f"    MLP fc2: {fc2_count}")

    # Show sample names
    print(f"\n  Sample MatMul names (first 10):")
    for name in matmul_names[:10]:
        cat = "qkv" if "qkv" in name else "proj" if "proj" in name else "fc1" if "fc1" in name else "fc2" if "fc2" in name else "attn"
        print(f"    [{cat:5s}] {name}")

    # Build and test engines
    print("\n" + "=" * 80)
    print("BENCHMARK: Selective FP32 strategies")
    print("=" * 80)

    strategies = [
        "pure_fp16",
        "softmax_only",
        "qk_softmax_only",
        "qkv_attn_softmax",
        "attn_matmul_softmax",
        "all_matmul_softmax",
    ]

    results = {}
    for strategy in strategies:
        engine_path = f"backbone_{strategy}.engine"
        print(f"\n--- {strategy} ---")
        try:
            fp32_n = build_with_selective_fp32(onnx_path, engine_path, strategy)
            cos, ms = test_engine(engine_path, model, dummy, strategy)
            results[strategy] = (cos, ms, fp32_n)
        except Exception as e:
            print(f"  FAILED: {e}")
        finally:
            Path(engine_path).unlink(missing_ok=True)

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"  {'Strategy':>30s} | {'FPN[-1] cos':>12s} | {'Speed (ms)':>10s} | {'FP32 layers':>12s}")
    print("  " + "-" * 75)
    for strategy, (cos, ms, n) in results.items():
        print(f"  {strategy:>30s} | {cos[-1]:>12.4f} | {ms:>10.1f} | {n:>12d}")

    print("\nDone!")


if __name__ == "__main__":
    main()
