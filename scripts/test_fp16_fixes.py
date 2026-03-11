#!/usr/bin/env python3
"""Test approaches to make TRT FP16 work at full speed (<66ms).

Approaches:
1. onnxsim to simplify the exported ONNX graph
2. _fp32_sdpa patch: explicit Cast(FP32) ops around attention in ONNX
3. Both combined
4. ONNX graph surgery with onnx-graphsurgeon to fold constants
"""

import sys
import time
import torch
import torch.nn as nn
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sam3.model_builder import build_sam3_image_model
from sam3.trt.rope_onnx import (
    patch_rope_for_export, unpatch_rope,
    patch_sdpa_for_export, unpatch_sdpa,
)

import tensorrt as trt
import onnx

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


def export_backbone(model, onnx_path, use_fp32_sdpa=False):
    """Export backbone with optional FP32 SDPA patch."""
    backbone = model.backbone
    patch_rope_for_export(backbone)
    if use_fp32_sdpa:
        patch_sdpa_for_export(backbone)

    wrapper = BackboneWrapper(backbone)
    dummy = torch.randn(1, 3, 1008, 1008, device=DEVICE)

    with torch.inference_mode():
        torch.onnx.export(
            wrapper, (dummy,), onnx_path,
            input_names=["image"],
            output_names=["fpn0", "fpn1", "fpn2"],
            opset_version=17, do_constant_folding=True,
        )

    if use_fp32_sdpa:
        unpatch_sdpa(backbone)
    unpatch_rope(backbone)

    size = Path(onnx_path).stat().st_size / 1e6
    model_onnx = onnx.load(onnx_path)
    n_nodes = len(model_onnx.graph.node)
    ops = Counter(n.op_type for n in model_onnx.graph.node)
    print(f"  Exported {onnx_path}: {size:.0f}MB, {n_nodes} nodes")
    print(f"    Top ops: {dict(ops.most_common(10))}")
    return onnx_path


def simplify_onnx(input_path, output_path):
    """Simplify ONNX with onnxsim."""
    import onnxsim

    print(f"  Simplifying {input_path} -> {output_path}...")
    model = onnx.load(input_path)
    n_before = len(model.graph.node)

    model_sim, check = onnxsim.simplify(model)
    if not check:
        print("  WARNING: onnxsim validation failed")

    onnx.save(model_sim, output_path)
    n_after = len(model_sim.graph.node)
    ops = Counter(n.op_type for n in model_sim.graph.node)
    print(f"  Simplified: {n_before} -> {n_after} nodes ({n_before - n_after} removed)")
    print(f"    Top ops: {dict(ops.most_common(10))}")
    return output_path


def fold_constants_gs(input_path, output_path):
    """Use onnx-graphsurgeon to fold constants and clean up."""
    import onnx_graphsurgeon as gs
    import numpy as np

    print(f"  Graph surgery on {input_path} -> {output_path}...")
    graph = gs.import_onnx(onnx.load(input_path))

    # Fold constants
    graph.fold_constants()
    graph.cleanup()
    graph.toposort()

    model = gs.export_onnx(graph)
    n_nodes = len(model.graph.node)
    ops = Counter(n.op_type for n in model.graph.node)
    onnx.save(model, output_path)
    print(f"  After GS: {n_nodes} nodes")
    print(f"    Top ops: {dict(ops.most_common(10))}")
    return output_path


def build_and_test(onnx_path, model, dummy, label, fp32_strategy=None):
    """Build TRT FP16 engine and test accuracy + speed."""
    from sam3.trt.trt_backbone import TRTBackbone

    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"  Parse error: {parser.get_error(i)}")
            return None

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
    config.set_flag(trt.BuilderFlag.FP16)

    # Layer stats
    layer_types = Counter()
    for i in range(network.num_layers):
        layer_types[str(network.get_layer(i).type).split(".")[-1]] += 1
    total = network.num_layers
    print(f"  TRT layers: {total} | MatMul={layer_types.get('MATRIX_MULTIPLY',0)} "
          f"Softmax={layer_types.get('SOFTMAX',0)} Cast={layer_types.get('CAST',0)}")

    fp32_count = 0
    if fp32_strategy == "attn_matmul_softmax":
        config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
        for i in range(network.num_layers):
            layer = network.get_layer(i)
            t = str(layer.type).split(".")[-1]
            if t == "SOFTMAX":
                layer.precision = trt.float32
                layer.set_output_type(0, trt.float32)
                fp32_count += 1
            elif t == "MATRIX_MULTIPLY":
                name = layer.name
                if not ("fc1" in name or "fc2" in name or "mlp" in name):
                    layer.precision = trt.float32
                    layer.set_output_type(0, trt.float32)
                    fp32_count += 1
        print(f"  FP32 constraints: {fp32_count} layers")

    t0 = time.time()
    engine_bytes = builder.build_serialized_network(network, config)
    build_s = time.time() - t0

    if engine_bytes is None:
        print(f"  {label}: BUILD FAILED")
        return None

    engine_path = onnx_path.replace(".onnx", ".engine")
    with open(engine_path, "wb") as f:
        f.write(engine_bytes)

    # Test accuracy
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
        for i in range(3)
    ]

    # Speed
    with torch.inference_mode():
        for _ in range(10):
            trt_bb.forward_image(dummy)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(100):
            trt_bb.forward_image(dummy)
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) / 100 * 1000

    status = "OK" if cos[-1] > 0.99 else "BROKEN" if cos[-1] < 0.5 else "DEGRADED"
    print(f"  {label:40s} | cos=[{cos[0]:.4f},{cos[1]:.4f},{cos[2]:.4f}] | "
          f"{ms:.1f}ms | build={build_s:.0f}s | {status}")

    del trt_bb
    torch.cuda.empty_cache()
    Path(engine_path).unlink(missing_ok=True)
    return cos, ms


def main():
    print("Loading model...")
    model = build_sam3_image_model(
        device=DEVICE, checkpoint_path="sam3.pt", eval_mode=True,
    )
    dummy = torch.randn(1, 3, 1008, 1008, device=DEVICE)

    results = {}

    # ============================================================
    # Approach 0: Baseline (original ONNX, pure FP16)
    # ============================================================
    print("\n" + "=" * 80)
    print("APPROACH 0: Baseline (original ONNX, pure FP16)")
    print("=" * 80)
    onnx_0 = "backbone.onnx"
    if not Path(onnx_0).exists():
        export_backbone(model, onnx_0)
    r = build_and_test(onnx_0, model, dummy, "baseline pure FP16")
    if r: results["baseline_fp16"] = r

    # ============================================================
    # Approach 0b: Baseline with attn FP32 (current best)
    # ============================================================
    print("\n" + "=" * 80)
    print("APPROACH 0b: Baseline + attention FP32 constraints")
    print("=" * 80)
    r = build_and_test(onnx_0, model, dummy, "baseline attn FP32",
                       fp32_strategy="attn_matmul_softmax")
    if r: results["baseline_attn_fp32"] = r

    # ============================================================
    # Approach 1: onnxsim simplified graph
    # ============================================================
    print("\n" + "=" * 80)
    print("APPROACH 1: onnxsim simplified ONNX, pure FP16")
    print("=" * 80)
    onnx_1 = "backbone_sim.onnx"
    try:
        simplify_onnx(onnx_0, onnx_1)
        r = build_and_test(onnx_1, model, dummy, "onnxsim pure FP16")
        if r: results["onnxsim_fp16"] = r
    except Exception as e:
        print(f"  FAILED: {e}")

    # ============================================================
    # Approach 2: FP32 SDPA patch (explicit Cast ops in ONNX)
    # ============================================================
    print("\n" + "=" * 80)
    print("APPROACH 2: FP32 SDPA patch, pure FP16 engine")
    print("=" * 80)
    onnx_2 = "backbone_fp32sdpa.onnx"
    export_backbone(model, onnx_2, use_fp32_sdpa=True)
    r = build_and_test(onnx_2, model, dummy, "fp32_sdpa pure FP16")
    if r: results["fp32sdpa_fp16"] = r

    # ============================================================
    # Approach 3: FP32 SDPA + onnxsim
    # ============================================================
    print("\n" + "=" * 80)
    print("APPROACH 3: FP32 SDPA + onnxsim, pure FP16 engine")
    print("=" * 80)
    onnx_3 = "backbone_fp32sdpa_sim.onnx"
    try:
        simplify_onnx(onnx_2, onnx_3)
        r = build_and_test(onnx_3, model, dummy, "fp32_sdpa + onnxsim FP16")
        if r: results["fp32sdpa_sim_fp16"] = r
    except Exception as e:
        print(f"  FAILED: {e}")

    # ============================================================
    # Approach 4: onnx-graphsurgeon constant folding
    # ============================================================
    print("\n" + "=" * 80)
    print("APPROACH 4: onnx-graphsurgeon constant fold, pure FP16")
    print("=" * 80)
    onnx_4 = "backbone_gs.onnx"
    try:
        fold_constants_gs(onnx_0, onnx_4)
        r = build_and_test(onnx_4, model, dummy, "graphsurgeon pure FP16")
        if r: results["gs_fp16"] = r
    except Exception as e:
        print(f"  FAILED: {e}")

    # ============================================================
    # Approach 5: FP32 SDPA + graphsurgeon
    # ============================================================
    print("\n" + "=" * 80)
    print("APPROACH 5: FP32 SDPA + graphsurgeon, pure FP16")
    print("=" * 80)
    onnx_5 = "backbone_fp32sdpa_gs.onnx"
    try:
        fold_constants_gs(onnx_2, onnx_5)
        r = build_and_test(onnx_5, model, dummy, "fp32_sdpa + GS pure FP16")
        if r: results["fp32sdpa_gs_fp16"] = r
    except Exception as e:
        print(f"  FAILED: {e}")

    # ============================================================
    # Approach 2b: FP32 SDPA patch + attn FP32 constraints (belt and suspenders)
    # ============================================================
    print("\n" + "=" * 80)
    print("APPROACH 2b: FP32 SDPA + attn FP32 constraints")
    print("=" * 80)
    r = build_and_test(onnx_2, model, dummy, "fp32_sdpa + attn FP32 constraints",
                       fp32_strategy="attn_matmul_softmax")
    if r: results["fp32sdpa_attn_fp32"] = r

    # ============================================================
    # SUMMARY
    # ============================================================
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"  {'Approach':>45s} | {'FPN[-1] cos':>12s} | {'Speed':>7s} | Status")
    print("  " + "-" * 80)
    for name, (cos, ms) in sorted(results.items(), key=lambda x: -x[1][0][-1]):
        status = "OK" if cos[-1] > 0.99 else "BROKEN" if cos[-1] < 0.5 else "DEGRADED"
        print(f"  {name:>45s} | {cos[-1]:>12.4f} | {ms:>5.1f}ms | {status}")

    # Cleanup temp files
    for f in [onnx_1, onnx_2, onnx_3, onnx_4, onnx_5]:
        Path(f).unlink(missing_ok=True)

    print("\nDone!")


if __name__ == "__main__":
    main()
