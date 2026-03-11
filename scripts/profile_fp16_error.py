#!/usr/bin/env python3
"""Profile where FP16 errors accumulate vs FP32 reference.

Runs the backbone in both FP32 and pure FP16, comparing outputs at every
block to find exactly where the error diverges. This simulates what TRT
FP16 does (everything in FP16) vs what PyTorch autocast does (some ops FP32).
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sam3.model_builder import build_sam3_image_model

DEVICE = "cuda"


def cosine_sim(a, b):
    return F.cosine_similarity(
        a.float().flatten().unsqueeze(0),
        b.float().flatten().unsqueeze(0),
    ).item()


def max_abs_diff(a, b):
    return (a.float() - b.float()).abs().max().item()


def rel_error(a, b):
    """Mean relative error."""
    a_f, b_f = a.float(), b.float()
    denom = a_f.abs().clamp(min=1e-6)
    return ((a_f - b_f).abs() / denom).mean().item()


def get_trunk_input(trunk, dummy):
    """Reproduce trunk forward up to the first block (patch_embed + pos + ln_pre)."""
    from sam3.model.vitdet import get_abs_pos
    x = trunk.patch_embed(dummy)
    h, w = x.shape[1], x.shape[2]
    if trunk.pos_embed is not None:
        x = x + get_abs_pos(
            trunk.pos_embed,
            trunk.pretrain_use_cls_token,
            (h, w),
            trunk.retain_cls_token,
            tiling=trunk.tile_abs_pos,
        )
    if hasattr(trunk, 'ln_pre') and trunk.ln_pre is not None:
        x = trunk.ln_pre(x)
    return x


def profile_block_by_block(model, dummy):
    """Run each ViT block in FP32 and FP16, comparing outputs."""
    backbone = model.backbone
    trunk = backbone.vision_backbone.trunk

    print("Running FP32 reference...")
    with torch.inference_mode():
        ref_out = backbone.forward_image(dummy.float())
    ref_fpn = ref_out["backbone_fpn"]

    # Now run block-by-block: both paths start from the same FP32 trunk input,
    # then one continues in FP32 and the other in FP16
    print("\nRunning block-by-block FP16 analysis...\n")

    with torch.inference_mode():
        x_fp32 = get_trunk_input(trunk, dummy.float())
        x_fp16 = get_trunk_input(trunk.half(), dummy.half())

    cos = cosine_sim(x_fp32, x_fp16)
    print(f"  {'patch_embed+pos+ln_pre':<35s} | cos={cos:.6f} | max_diff={max_abs_diff(x_fp32, x_fp16):.4f} | "
          f"fp32_range=[{x_fp32.min().item():.2f}, {x_fp32.max().item():.2f}] | "
          f"fp16_range=[{x_fp16.float().min().item():.2f}, {x_fp16.float().max().item():.2f}]")

    # Run each block
    print(f"\n  {'Block':<35s} | {'Cosine':>10s} | {'MaxDiff':>10s} | {'RelErr':>10s} | "
          f"{'FP32 range':>20s} | {'FP16 absmax':>12s}")
    print("  " + "-" * 115)

    for i, block in enumerate(trunk.blocks):
        with torch.inference_mode():
            x_fp32 = block.float()(x_fp32.float())
            x_fp16 = block.half()(x_fp16.half())

        cos = cosine_sim(x_fp32, x_fp16)
        md = max_abs_diff(x_fp32, x_fp16)
        re = rel_error(x_fp32, x_fp16)
        fp32_min, fp32_max = x_fp32.min().item(), x_fp32.max().item()
        fp16_absmax = x_fp16.float().abs().max().item()

        status = ""
        if cos < 0.99:
            status = " << DIVERGED"
        elif cos < 0.999:
            status = " << degrading"

        print(f"  block.{i:<28d} | {cos:>10.6f} | {md:>10.4f} | {re:>10.6f} | "
              f"[{fp32_min:>8.2f}, {fp32_max:>8.2f}] | {fp16_absmax:>12.2f}{status}")

    # Final FPN comparison
    print("\n  FPN outputs (full model FP16 vs FP32 reference):")
    with torch.inference_mode():
        from sam3.model_builder import build_sam3_image_model
        model_half = build_sam3_image_model(
            device=DEVICE, checkpoint_path="sam3.pt", eval_mode=True,
        )
        model_half.backbone.half()
        fp16_out = model_half.backbone.forward_image(dummy.half())
    fp16_fpn = fp16_out["backbone_fpn"]

    for i in range(len(ref_fpn)):
        cos = cosine_sim(ref_fpn[i], fp16_fpn[i])
        md = max_abs_diff(ref_fpn[i], fp16_fpn[i])
        print(f"    FPN[{i}]: cosine={cos:.6f}, max_diff={md:.4f}")
    del model_half
    torch.cuda.empty_cache()


def profile_single_block_ops(model, dummy, block_idx=0):
    """Break down a single block into individual ops and measure error."""
    backbone = model.backbone
    trunk = backbone.vision_backbone.trunk

    # Get input to the target block
    with torch.inference_mode():
        x = get_trunk_input(trunk, dummy.float())
        for i in range(block_idx):
            x = trunk.blocks[i](x)

    block = trunk.blocks[block_idx]
    x_fp32 = x.float()
    x_fp16 = x.half()

    print(f"\n  Detailed ops for block {block_idx}:")
    print(f"  {'Operation':<40s} | {'Cosine':>10s} | {'MaxDiff':>10s} | {'AbsMax FP32':>12s} | {'AbsMax FP16':>12s}")
    print("  " + "-" * 100)

    with torch.inference_mode():
        # 1. Pre-attention LayerNorm
        if hasattr(block, 'norm1'):
            n1_fp32 = block.norm1(x_fp32.float())
            n1_fp16 = block.norm1.half()(x_fp16.half())
            cos = cosine_sim(n1_fp32, n1_fp16)
            md = max_abs_diff(n1_fp32, n1_fp16)
            print(f"  {'norm1 (LayerNorm)':<40s} | {cos:>10.6f} | {md:>10.4f} | "
                  f"{n1_fp32.abs().max().item():>12.4f} | {n1_fp16.float().abs().max().item():>12.4f}")
        else:
            n1_fp32 = x_fp32
            n1_fp16 = x_fp16

        # 2. QKV projection
        attn = block.attn
        qkv_fp32 = attn.qkv(n1_fp32.float())
        qkv_fp16 = attn.qkv.half()(n1_fp16.half())
        cos = cosine_sim(qkv_fp32, qkv_fp16)
        md = max_abs_diff(qkv_fp32, qkv_fp16)
        print(f"  {'attn.qkv (Linear 1024->3072)':<40s} | {cos:>10.6f} | {md:>10.4f} | "
              f"{qkv_fp32.abs().max().item():>12.4f} | {qkv_fp16.float().abs().max().item():>12.4f}")

        # 3. Split into Q, K, V and compute attention scores
        num_heads = attn.num_heads
        B = qkv_fp32.shape[0]
        if qkv_fp32.dim() == 4:
            H, W = qkv_fp32.shape[1], qkv_fp32.shape[2]
            N = H * W
            qkv_r32 = qkv_fp32.reshape(B, N, 3, num_heads, -1).permute(2, 0, 3, 1, 4)
            qkv_r16 = qkv_fp16.reshape(B, N, 3, num_heads, -1).permute(2, 0, 3, 1, 4)
        else:
            N = qkv_fp32.shape[1]
            qkv_r32 = qkv_fp32.reshape(B, N, 3, num_heads, -1).permute(2, 0, 3, 1, 4)
            qkv_r16 = qkv_fp16.reshape(B, N, 3, num_heads, -1).permute(2, 0, 3, 1, 4)

        q32, k32, v32 = qkv_r32[0], qkv_r32[1], qkv_r32[2]
        q16, k16, v16 = qkv_r16[0].float(), qkv_r16[1].float(), qkv_r16[2].float()
        head_dim = q32.shape[-1]

        # Q@K^T
        scores_fp32 = torch.matmul(q32.float(), k32.float().transpose(-2, -1)) * (head_dim ** -0.5)
        scores_fp16 = torch.matmul(q16, k16.transpose(-2, -1)) * (head_dim ** -0.5)
        cos = cosine_sim(scores_fp32, scores_fp16)
        md = max_abs_diff(scores_fp32, scores_fp16)
        print(f"  {'Q@K^T * scale (from FP16 QKV)':<40s} | {cos:>10.6f} | {md:>10.4f} | "
              f"{scores_fp32.abs().max().item():>12.4f} | {scores_fp16.abs().max().item():>12.4f}")

        # Now do Q@K^T in actual FP16 arithmetic (simulating TRT)
        scores_fp16_arith = torch.matmul(
            qkv_r16[0].half(), qkv_r16[1].half().transpose(-2, -1)
        ).float() * (head_dim ** -0.5)
        cos_arith = cosine_sim(scores_fp32, scores_fp16_arith)
        md_arith = max_abs_diff(scores_fp32, scores_fp16_arith)
        print(f"  {'Q@K^T * scale (FP16 matmul)':<40s} | {cos_arith:>10.6f} | {md_arith:>10.4f} | "
              f"{'':>12s} | {scores_fp16_arith.abs().max().item():>12.4f}")

        # Softmax in FP32 vs FP16
        softmax_fp32 = torch.softmax(scores_fp32.float(), dim=-1)
        softmax_fp16 = torch.softmax(scores_fp16_arith.half(), dim=-1)
        cos = cosine_sim(softmax_fp32, softmax_fp16)
        md = max_abs_diff(softmax_fp32, softmax_fp16)
        print(f"  {'Softmax (FP32 vs FP16 input)':<40s} | {cos:>10.6f} | {md:>10.4f} | "
              f"{softmax_fp32.abs().max().item():>12.4f} | {softmax_fp16.float().abs().max().item():>12.4f}")

        # attn @ V
        attn_out_fp32 = torch.matmul(softmax_fp32.float(), v32.float())
        attn_out_fp16 = torch.matmul(softmax_fp16.half(), qkv_r16[2].half()).float()
        cos = cosine_sim(attn_out_fp32, attn_out_fp16)
        md = max_abs_diff(attn_out_fp32, attn_out_fp16)
        print(f"  {'Attn @ V (accumulated error)':<40s} | {cos:>10.6f} | {md:>10.4f} | "
              f"{attn_out_fp32.abs().max().item():>12.4f} | {attn_out_fp16.abs().max().item():>12.4f}")


def profile_layernorm_precision(model, dummy):
    """Test if LayerNorm in FP16 causes issues."""
    backbone = model.backbone
    trunk = backbone.vision_backbone.trunk

    print("\n  LayerNorm FP16 precision test:")
    print(f"  {'Layer':<40s} | {'Cosine':>10s} | {'MaxDiff':>10s}")
    print("  " + "-" * 70)

    with torch.inference_mode():
        x = get_trunk_input(trunk, dummy.float())

        for i, block in enumerate(trunk.blocks):
            if hasattr(block, 'norm1'):
                ln_fp32 = block.norm1(x.float())
                ln_fp16 = block.norm1.half()(x.half())
                cos = cosine_sim(ln_fp32, ln_fp16)
                md = max_abs_diff(ln_fp32, ln_fp16)
                print(f"  {'block.' + str(i) + '.norm1':<40s} | {cos:>10.6f} | {md:>10.4f}")
            x = block(x)
            if i >= 7:
                break  # First 8 blocks is enough to see the trend


def main():
    print("Loading model...")
    model = build_sam3_image_model(
        device=DEVICE, checkpoint_path="sam3.pt", eval_mode=True,
    )
    dummy = torch.randn(1, 3, 1008, 1008, device=DEVICE)

    # Part 1: Block-by-block error accumulation
    print("\n" + "=" * 80)
    print("PART 1: Block-by-block FP32 vs FP16 comparison")
    print("=" * 80)
    profile_block_by_block(model, dummy)

    # Part 2: Detailed single block analysis
    print("\n" + "=" * 80)
    print("PART 2: Detailed op-by-op analysis (blocks 0 and 15)")
    print("=" * 80)

    # Reload to clear any state
    del model
    torch.cuda.empty_cache()
    model = build_sam3_image_model(
        device=DEVICE, checkpoint_path="sam3.pt", eval_mode=True,
    )

    profile_single_block_ops(model, dummy, block_idx=0)
    profile_single_block_ops(model, dummy, block_idx=15)

    # Part 3: LayerNorm precision
    print("\n" + "=" * 80)
    print("PART 3: LayerNorm FP16 precision")
    print("=" * 80)
    profile_layernorm_precision(model, dummy)

    print("\nDone!")


if __name__ == "__main__":
    main()
