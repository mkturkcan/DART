#!/usr/bin/env python3
"""Export distilled student backbones to ONNX (dynamo) + TRT FP16.

Uses the dynamo ONNX exporter (opset 20, native Gelu ops), then inserts
Identity nodes between Conv and Gelu to prevent TRT from fusing them into
an unimplementable Conv+Gelu kernel at FP16.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import onnx
from onnx import helper
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sam3.distillation.student_backbone import build_student_backbone


class _Wrapper(nn.Module):
    def __init__(self, student_bb):
        super().__init__()
        self.backbone = student_bb

    def forward(self, images: torch.Tensor):
        out = self.backbone.forward_image(images)
        fpn = out["backbone_fpn"]
        return fpn[0], fpn[1], fpn[2]


def _fix_conv_gelu_fusion(onnx_path):
    """Insert Identity nodes between Conv and Gelu to prevent TRT fusion."""
    m = onnx.load(onnx_path)
    g = m.graph

    conv_outputs = set()
    for node in g.node:
        if node.op_type == "Conv":
            for out in node.output:
                conv_outputs.add(out)

    new_nodes = []
    n_fixed = 0
    for node in g.node:
        if node.op_type == "Gelu":
            for i, inp in enumerate(node.input):
                if inp in conv_outputs:
                    new_name = inp + "_identity"
                    identity_node = helper.make_node(
                        "Identity",
                        inputs=[inp],
                        outputs=[new_name],
                        name=f"identity_break_fusion_{n_fixed}",
                    )
                    new_nodes.append(identity_node)
                    node.input[i] = new_name
                    n_fixed += 1

    for id_node in new_nodes:
        g.node.append(id_node)

    onnx.save(m, onnx_path)
    print(f"  Fixed {n_fixed} Conv->Gelu fusions with Identity nodes")
    return n_fixed


def export_and_build(backbone_config, adapter_checkpoint, output_prefix, imgsz=1008):
    onnx_path = f"{output_prefix}.onnx"
    engine_path = f"{output_prefix}_fp16.engine"

    # Build backbone
    print(f"\n{'='*60}")
    print(f"  {backbone_config}")
    print(f"{'='*60}")
    student_bb = build_student_backbone(
        config_name=backbone_config, pretrained=True, freeze_backbone=True,
    )
    ckpt = torch.load(adapter_checkpoint, map_location="cpu")
    student_bb.load_state_dict(ckpt["student_state_dict"], strict=False)

    wrapper = _Wrapper(student_bb).cuda().eval()
    dummy = torch.randn(1, 3, imgsz, imgsz, device="cuda")

    with torch.no_grad():
        fpn0, fpn1, fpn2 = wrapper(dummy)
    print(f"  fpn_0: {list(fpn0.shape)}, fpn_1: {list(fpn1.shape)}, fpn_2: {list(fpn2.shape)}")

    # Dynamo ONNX export (keeps native Gelu op at opset 20)
    print(f"  Exporting ONNX (dynamo) -> {onnx_path}")
    t0 = time.perf_counter()
    export_output = torch.onnx.export(wrapper, (dummy,), dynamo=True)
    export_output.save(onnx_path)
    print(f"  ONNX export done ({time.perf_counter() - t0:.1f}s)")

    # Fix Conv+Gelu fusion issue for TRT
    _fix_conv_gelu_fusion(onnx_path)

    # Free GPU memory
    del wrapper, student_bb, dummy
    torch.cuda.empty_cache()

    # Build TRT FP16
    import tensorrt as trt

    TRT_LOGGER = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, TRT_LOGGER)

    onnx_abs = str(Path(onnx_path).resolve())
    print(f"  Parsing ONNX: {onnx_abs}")
    if hasattr(parser, "parse_from_file"):
        if not parser.parse_from_file(onnx_abs):
            for i in range(parser.num_errors):
                print(f"    Error: {parser.get_error(i)}")
            raise RuntimeError("ONNX parse failed")
    else:
        with open(onnx_abs, "rb") as f:
            if not parser.parse(f.read()):
                for i in range(parser.num_errors):
                    print(f"    Error: {parser.get_error(i)}")
                raise RuntimeError("ONNX parse failed")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
    config.set_flag(trt.BuilderFlag.FP16)
    if hasattr(config, "builder_optimization_level"):
        config.builder_optimization_level = 3

    print(f"  Building TRT FP16 engine -> {engine_path}")
    t0 = time.perf_counter()
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError(f"Engine build failed for {backbone_config}")

    with open(engine_path, "wb") as f:
        f.write(serialized)
    size_mb = Path(engine_path).stat().st_size / 1e6
    dt = time.perf_counter() - t0
    print(f"  Done ({dt:.0f}s), {size_mb:.0f} MB -> {engine_path}")

    del builder, network, parser, config, serialized
    torch.cuda.empty_cache()

    return onnx_path, engine_path


MODELS = {
    "efficientvit_l1": "distilled/efficientvit_l1_distilled.pt",
    "efficientvit_l2": "distilled/efficientvit_l2_distilled.pt",
    "repvit_m2_3": "distilled/repvit_m2_3_distilled.pt",
    "tiny_vit_21m": "distilled/tiny_vit_21m_distilled.pt",
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(MODELS.keys()),
        choices=list(MODELS.keys()),
    )
    parser.add_argument("--imgsz", type=int, default=1008)
    args = parser.parse_args()

    results = []
    for name in args.models:
        adapter_ckpt = MODELS[name]
        prefix = f"student_{name}"
        onnx_path, engine_path = export_and_build(
            backbone_config=name,
            adapter_checkpoint=adapter_ckpt,
            output_prefix=prefix,
            imgsz=args.imgsz,
        )
        results.append((name, onnx_path, engine_path))

    print(f"\n{'='*60}")
    print("ALL EXPORTS COMPLETE:")
    for name, onnx_p, eng_p in results:
        print(f"  {name}: {eng_p}")
    print(f"{'='*60}")
