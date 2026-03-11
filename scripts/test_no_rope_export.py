#!/usr/bin/env python3
"""Export backbone without RoPE and test TRT FP16 accuracy.

If TRT FP16 works without RoPE, the issue is in the ONNX RoPE decomposition.
If it still fails, the issue is in the compounding of MatMul errors.
"""

import sys
import time
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sam3.model_builder import build_sam3_image_model
from sam3.trt.rope_onnx import patch_rope_for_export, unpatch_rope

import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
DEVICE = "cuda"


class _BackboneForExport(torch.nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, images):
        out = self.backbone.forward_image(images)
        fpn = out["backbone_fpn"]
        return fpn[0], fpn[1], fpn[2]


def _disable_rope(model):
    """Disable RoPE by making _apply_rope a no-op."""
    trunk = model.backbone.vision_backbone.trunk
    for block in trunk.blocks:
        attn = block.attn
        attn._orig_apply_rope = attn._apply_rope
        attn._apply_rope = lambda q, k: (q, k)


def _restore_rope(model):
    trunk = model.backbone.vision_backbone.trunk
    for block in trunk.blocks:
        attn = block.attn
        if hasattr(attn, '_orig_apply_rope'):
            attn._apply_rope = attn._orig_apply_rope


def export_backbone(model, output_path, use_rope=True):
    """Export backbone to ONNX."""
    backbone = model.backbone

    if not use_rope:
        _disable_rope(model)

    # Always patch rope for export (handles complex number compat)
    patch_rope_for_export(backbone)

    wrapper = _BackboneForExport(backbone)
    dummy = torch.randn(1, 3, 1008, 1008, device=DEVICE)

    label = "with RoPE" if use_rope else "WITHOUT RoPE"
    print(f"Exporting backbone ({label}) to {output_path}...")
    with torch.inference_mode():
        torch.onnx.export(
            wrapper,
            (dummy,),
            output_path,
            input_names=["image"],
            output_names=["fpn0", "fpn1", "fpn2"],
            opset_version=17,
            do_constant_folding=True,
        )
    print(f"  Exported: {Path(output_path).stat().st_size / 1e6:.0f} MB")

    unpatch_rope(backbone)
    if not use_rope:
        _restore_rope(model)

    return output_path


def build_fp16_engine(onnx_path, output_path):
    """Build pure FP16 TRT engine."""
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, TRT_LOGGER)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"  Parse error: {parser.get_error(i)}")
            raise RuntimeError("Parse failed")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
    config.set_flag(trt.BuilderFlag.FP16)

    # Count layer types
    from collections import Counter
    types = Counter()
    for i in range(network.num_layers):
        layer = network.get_layer(i)
        types[str(layer.type).split(".")[-1]] += 1
    print(f"  Layer types: {dict(types.most_common(15))}")

    t0 = time.time()
    engine_bytes = builder.build_serialized_network(network, config)
    print(f"  Built in {time.time()-t0:.0f}s")

    if engine_bytes is None:
        raise RuntimeError("Build failed")
    with open(output_path, "wb") as f:
        f.write(engine_bytes)
    print(f"  Saved: {output_path}")


def test_accuracy(engine_path, label, model, dummy, use_rope=True):
    """Test TRT engine accuracy vs PyTorch."""
    from sam3.trt.trt_backbone import TRTBackbone

    backbone = model.backbone

    if not use_rope:
        _disable_rope(model)

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        pt_out = backbone.forward_image(dummy)
    pt_fpn = pt_out["backbone_fpn"]

    if not use_rope:
        _restore_rope(model)

    pos_module = backbone.vision_backbone.position_encoding
    trt_bb = TRTBackbone(
        engine_path=engine_path, device=DEVICE, pos_encoding_module=pos_module,
    )

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

    # Speed
    with torch.inference_mode():
        for _ in range(10):
            trt_bb.forward_image(dummy)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(50):
            trt_bb.forward_image(dummy)
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) / 50 * 1000

    status = "OK" if cos[-1] > 0.99 else "BROKEN"
    print(f"  {label:30s} | cos=[{cos[0]:.4f}, {cos[1]:.4f}, {cos[2]:.4f}] | {ms:.1f}ms | {status}")

    del trt_bb
    torch.cuda.empty_cache()


def main():
    print("Loading model...")
    model = build_sam3_image_model(
        device=DEVICE, checkpoint_path="sam3.pt", eval_mode=True,
    )
    dummy = torch.randn(1, 3, 1008, 1008, device=DEVICE)

    print("\n" + "=" * 80)
    print("Test: Does RoPE decomposition cause TRT FP16 failure?")
    print("=" * 80)

    # Export without RoPE
    export_backbone(model, "backbone_no_rope.onnx", use_rope=False)
    build_fp16_engine("backbone_no_rope.onnx", "backbone_no_rope_fp16.engine")
    test_accuracy("backbone_no_rope_fp16.engine", "FP16 WITHOUT RoPE", model, dummy, use_rope=False)

    # Compare with original backbone.onnx
    print()
    build_fp16_engine("backbone.onnx", "backbone_with_rope_fp16.engine")
    test_accuracy("backbone_with_rope_fp16.engine", "FP16 WITH RoPE (original)", model, dummy, use_rope=True)

    # Cleanup
    for f in ["backbone_no_rope.onnx", "backbone_no_rope_fp16.engine", "backbone_with_rope_fp16.engine"]:
        Path(f).unlink(missing_ok=True)

    print("\nDone!")


if __name__ == "__main__":
    main()
