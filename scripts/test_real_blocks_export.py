#!/usr/bin/env python3
"""Export subsets of the REAL SAM3 backbone via torch.onnx.export and test in TRT FP16.

Previous tests showed:
- Synthetic attention + real weights: TRT FP16 cosine 0.999993 (8 blocks)
- Full backbone ONNX: TRT FP16 cosine 0.07

Missing pieces: MLP, RoPE, window partitioning, actual ONNX export graph structure.
This script exports N real blocks through torch.onnx.export to reproduce the exact
graph structure TRT sees.
"""

import sys
import time
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sam3.model_builder import build_sam3_image_model
from sam3.trt.rope_onnx import patch_rope_for_export, unpatch_rope

import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
DEVICE = "cuda"


class NBlockWrapper(nn.Module):
    """Wraps N blocks of the ViT trunk for ONNX export."""
    def __init__(self, blocks):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class NBlockWithRoPEWrapper(nn.Module):
    """Wraps N blocks with RoPE support for ONNX export."""
    def __init__(self, trunk, num_blocks):
        super().__init__()
        self.blocks = nn.ModuleList(trunk.blocks[:num_blocks])
        # Store RoPE-related state
        self.trunk = trunk

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class FullTrunkWrapper(nn.Module):
    """Wraps the full backbone forward up to FPN."""
    def __init__(self, backbone, num_blocks=None):
        super().__init__()
        self.backbone = backbone
        self.num_blocks = num_blocks

    def forward(self, images):
        out = self.backbone.forward_image(images)
        # Return just the last FPN level
        return out["backbone_fpn"][-1]


class TrunkOnlyWrapper(nn.Module):
    """Wraps just the ViT trunk (patch_embed + pos + blocks)."""
    def __init__(self, trunk, num_blocks=None):
        super().__init__()
        self.trunk = trunk
        self.num_blocks = num_blocks

    def forward(self, x):
        """x: already embedded tokens [B, H, W, C]"""
        for i, block in enumerate(self.trunk.blocks):
            if self.num_blocks is not None and i >= self.num_blocks:
                break
            x = block(x)
        return x


def build_trt_engine(onnx_path, fp16=True):
    """Build TRT engine from ONNX."""
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"  Parse error: {parser.get_error(i)}")
            raise RuntimeError("Parse failed")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    engine_bytes = builder.build_serialized_network(network, config)
    if engine_bytes is None:
        raise RuntimeError("Engine build failed")
    return engine_bytes


def run_trt_engine(engine_bytes, input_tensor):
    """Run TRT engine on input."""
    runtime = trt.Runtime(TRT_LOGGER)
    engine = runtime.deserialize_cuda_engine(engine_bytes)
    ctx = engine.create_execution_context()

    d_in = input_tensor.float().contiguous().cuda()
    out_name = engine.get_tensor_name(engine.num_io_tensors - 1)
    out_shape = ctx.get_tensor_shape(out_name)
    d_out = torch.empty(list(out_shape), dtype=torch.float32, device="cuda")

    ctx.set_tensor_address(engine.get_tensor_name(0), d_in.data_ptr())
    ctx.set_tensor_address(out_name, d_out.data_ptr())
    ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    return d_out


def cosine_sim(a, b):
    return torch.nn.functional.cosine_similarity(
        a.float().flatten().unsqueeze(0),
        b.float().flatten().unsqueeze(0),
    ).item()


def test_blocks_exported(model, num_blocks_list):
    """Export N blocks via torch.onnx.export and test in TRT FP16."""
    trunk = model.backbone.vision_backbone.trunk

    # Get a real intermediate input (after patch_embed + pos)
    dummy_img = torch.randn(1, 3, 1008, 1008, device=DEVICE)
    with torch.inference_mode():
        from sam3.model.vitdet import get_abs_pos
        x_input = trunk.patch_embed(dummy_img)
        h, w = x_input.shape[1], x_input.shape[2]
        if trunk.pos_embed is not None:
            x_input = x_input + get_abs_pos(
                trunk.pos_embed, trunk.pretrain_use_cls_token,
                (h, w), trunk.retain_cls_token, tiling=trunk.tile_abs_pos,
            )
        if hasattr(trunk, 'ln_pre') and trunk.ln_pre is not None:
            x_input = trunk.ln_pre(x_input)

    print(f"  Trunk input shape: {list(x_input.shape)}, range: [{x_input.min():.2f}, {x_input.max():.2f}]")

    print(f"\n  {'Blocks':>7s} | {'TRT FP16 cos':>12s} | {'PT FP16 cos':>12s} | "
          f"{'TRT max_diff':>12s} | {'PT max_diff':>12s} | {'Out absmax':>11s}")
    print("  " + "-" * 80)

    for num_blocks in num_blocks_list:
        onnx_path = f"test_trunk_{num_blocks}blk.onnx"

        # Create wrapper
        wrapper = TrunkOnlyWrapper(trunk, num_blocks=num_blocks)

        # Patch RoPE for export
        patch_rope_for_export(model.backbone)

        try:
            with torch.inference_mode():
                torch.onnx.export(
                    wrapper,
                    (x_input,),
                    onnx_path,
                    input_names=["tokens"],
                    output_names=["output"],
                    opset_version=17,
                    do_constant_folding=True,
                )
        except Exception as e:
            print(f"  {num_blocks:>7d} | EXPORT FAILED: {e}")
            unpatch_rope(model.backbone)
            continue

        unpatch_rope(model.backbone)

        # FP32 reference
        with torch.inference_mode():
            Y_fp32 = wrapper(x_input.float())

        # PyTorch FP16 reference
        with torch.inference_mode():
            wrapper_half = TrunkOnlyWrapper(trunk, num_blocks=num_blocks)
            # Run blocks in half precision
            x_half = x_input.half()
            for i, block in enumerate(trunk.blocks):
                if i >= num_blocks:
                    break
                x_half = block.half()(x_half)
            Y_pt16 = x_half

        pt_cos = cosine_sim(Y_fp32, Y_pt16)
        pt_diff = (Y_fp32 - Y_pt16.float()).abs().max().item()

        # Restore blocks to float
        for block in trunk.blocks:
            block.float()

        # TRT FP16
        try:
            engine_bytes = build_trt_engine(onnx_path, fp16=True)
            Y_trt = run_trt_engine(engine_bytes, x_input)
            trt_cos = cosine_sim(Y_fp32, Y_trt)
            trt_diff = (Y_fp32 - Y_trt).abs().max().item()
        except Exception as e:
            print(f"  {num_blocks:>7d} | TRT FAILED: {e}")
            Path(onnx_path).unlink(missing_ok=True)
            continue

        out_max = Y_fp32.abs().max().item()
        print(f"  {num_blocks:>7d} | {trt_cos:>12.6f} | {pt_cos:>12.6f} | "
              f"{trt_diff:>12.4f} | {pt_diff:>12.4f} | {out_max:>11.2f}")

        Path(onnx_path).unlink(missing_ok=True)


def test_blocks_without_rope(model, num_blocks_list):
    """Same test but with RoPE disabled to isolate its effect."""
    trunk = model.backbone.vision_backbone.trunk

    # Disable RoPE
    for block in trunk.blocks:
        attn = block.attn
        attn._orig_apply_rope = attn._apply_rope
        attn._apply_rope = lambda q, k: (q, k)

    # Get input
    dummy_img = torch.randn(1, 3, 1008, 1008, device=DEVICE)
    with torch.inference_mode():
        from sam3.model.vitdet import get_abs_pos
        x_input = trunk.patch_embed(dummy_img)
        h, w = x_input.shape[1], x_input.shape[2]
        if trunk.pos_embed is not None:
            x_input = x_input + get_abs_pos(
                trunk.pos_embed, trunk.pretrain_use_cls_token,
                (h, w), trunk.retain_cls_token, tiling=trunk.tile_abs_pos,
            )
        if hasattr(trunk, 'ln_pre') and trunk.ln_pre is not None:
            x_input = trunk.ln_pre(x_input)

    print(f"\n  {'Blocks':>7s} | {'TRT FP16 cos':>12s} | {'PT FP16 cos':>12s} | "
          f"{'TRT max_diff':>12s} | {'PT max_diff':>12s}")
    print("  " + "-" * 65)

    for num_blocks in num_blocks_list:
        onnx_path = f"test_trunk_norope_{num_blocks}blk.onnx"

        wrapper = TrunkOnlyWrapper(trunk, num_blocks=num_blocks)

        # Patch for export (even though RoPE is disabled, the export path needs this)
        patch_rope_for_export(model.backbone)

        try:
            with torch.inference_mode():
                torch.onnx.export(
                    wrapper,
                    (x_input,),
                    onnx_path,
                    input_names=["tokens"],
                    output_names=["output"],
                    opset_version=17,
                    do_constant_folding=True,
                )
        except Exception as e:
            print(f"  {num_blocks:>7d} | EXPORT FAILED: {e}")
            unpatch_rope(model.backbone)
            continue

        unpatch_rope(model.backbone)

        # FP32 reference
        with torch.inference_mode():
            Y_fp32 = wrapper(x_input.float())

        # PT FP16
        with torch.inference_mode():
            x_half = x_input.half()
            for i, block in enumerate(trunk.blocks):
                if i >= num_blocks:
                    break
                x_half = block.half()(x_half)
            Y_pt16 = x_half

        pt_cos = cosine_sim(Y_fp32, Y_pt16)
        pt_diff = (Y_fp32 - Y_pt16.float()).abs().max().item()

        for block in trunk.blocks:
            block.float()

        # TRT FP16
        try:
            engine_bytes = build_trt_engine(onnx_path, fp16=True)
            Y_trt = run_trt_engine(engine_bytes, x_input)
            trt_cos = cosine_sim(Y_fp32, Y_trt)
            trt_diff = (Y_fp32 - Y_trt).abs().max().item()
        except Exception as e:
            print(f"  {num_blocks:>7d} | TRT FAILED: {e}")
            Path(onnx_path).unlink(missing_ok=True)
            continue

        print(f"  {num_blocks:>7d} | {trt_cos:>12.6f} | {pt_cos:>12.6f} | "
              f"{trt_diff:>12.4f} | {pt_diff:>12.4f}")

        Path(onnx_path).unlink(missing_ok=True)

    # Restore RoPE
    for block in trunk.blocks:
        attn = block.attn
        if hasattr(attn, '_orig_apply_rope'):
            attn._apply_rope = attn._orig_apply_rope


def main():
    print("Loading model...")
    model = build_sam3_image_model(
        device=DEVICE, checkpoint_path="sam3.pt", eval_mode=True,
    )

    # Test 1: Real blocks exported via torch.onnx.export
    print("\n" + "=" * 80)
    print("TEST 1: Real ViT blocks exported via torch.onnx.export (WITH RoPE)")
    print("=" * 80)
    test_blocks_exported(model, [1, 2, 4, 8, 16, 32])

    # Test 2: Without RoPE
    print("\n" + "=" * 80)
    print("TEST 2: Real ViT blocks exported (WITHOUT RoPE)")
    print("=" * 80)
    test_blocks_without_rope(model, [1, 2, 4, 8, 16, 32])

    print("\nDone!")


if __name__ == "__main__":
    main()
