#!/usr/bin/env python3
"""Profile activation ranges layer-by-layer in the ViT-H backbone.

Goal: find exactly which operations produce values outside FP16 range (>65504)
so we can fix the weights/graph to make pure FP16 TRT work at 26ms.

FP16 range: [-65504, 65504]
FP16 smallest subnormal: ~5.96e-8
"""

import sys
from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sam3.model_builder import build_sam3_image_model

DEVICE = "cuda"
FP16_MAX = 65504.0


def profile_backbone(model, images):
    """Run backbone with hooks to capture activation statistics."""
    backbone = model.backbone
    trunk = backbone.vision_backbone.trunk

    stats = OrderedDict()
    hooks = []

    def make_hook(name):
        def hook_fn(module, input, output):
            tensors = []
            if isinstance(output, torch.Tensor):
                tensors.append(("out", output))
            elif isinstance(output, (tuple, list)):
                for i, t in enumerate(output):
                    if isinstance(t, torch.Tensor):
                        tensors.append((f"out[{i}]", t))
            elif isinstance(output, dict):
                for k, v in output.items():
                    if isinstance(v, torch.Tensor):
                        tensors.append((f"out.{k}", v))

            # Also capture inputs
            if isinstance(input, (tuple, list)):
                for i, t in enumerate(input):
                    if isinstance(t, torch.Tensor):
                        tensors.append((f"in[{i}]", t))

            for suffix, t in tensors:
                key = f"{name}/{suffix}"
                t_flat = t.float().flatten()
                abs_max = t_flat.abs().max().item()
                stats[key] = {
                    "shape": list(t.shape),
                    "dtype": str(t.dtype),
                    "min": t_flat.min().item(),
                    "max": t_flat.max().item(),
                    "abs_max": abs_max,
                    "mean": t_flat.mean().item(),
                    "std": t_flat.std().item(),
                    "overflow_fp16": abs_max > FP16_MAX,
                    "near_overflow": abs_max > FP16_MAX * 0.5,
                    "num_overflow": (t_flat.abs() > FP16_MAX).sum().item(),
                    "pct_overflow": (t_flat.abs() > FP16_MAX).float().mean().item() * 100,
                }
        return hook_fn

    # Register hooks on every submodule
    for name, module in backbone.named_modules():
        if len(list(module.children())) == 0:  # leaf modules only
            hooks.append(module.register_forward_hook(make_hook(name)))

    # Also add special hooks for attention internals
    # We need to monkey-patch the attention forward to capture Q, K, V, and scores
    attn_internals = {}

    def patch_attention_block(block_idx, block):
        attn = block.attn
        orig_forward = attn.forward

        def patched_forward(x, *args, **kwargs):
            B, H, W, C = x.shape if x.dim() == 4 else (1, 1, x.shape[0], x.shape[1])

            # Get QKV
            qkv = attn.qkv(x)
            prefix = f"trunk.blocks.{block_idx}.attn"

            # Record QKV stats
            q_flat = qkv.float().flatten()
            attn_internals[f"{prefix}/qkv_output"] = {
                "abs_max": q_flat.abs().max().item(),
                "overflow_fp16": q_flat.abs().max().item() > FP16_MAX,
                "std": q_flat.std().item(),
            }

            # Call original forward
            return orig_forward(x, *args, **kwargs)

        attn.forward = patched_forward

    for i, block in enumerate(trunk.blocks):
        patch_attention_block(i, block)

    # Run inference
    print("Running inference with hooks...")
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        for img in images:
            backbone.forward_image(img)

    # Remove hooks
    for h in hooks:
        h.remove()

    return stats, attn_internals


def profile_attention_detailed(model, dummy):
    """Detailed profiling of attention Q@K^T scores."""
    backbone = model.backbone
    trunk = backbone.vision_backbone.trunk

    attention_stats = []

    def patch_block(block_idx, block):
        attn = block.attn
        orig_forward = attn.forward

        def detailed_forward(x, *args, **kwargs):
            B, N, C = x.shape if x.dim() == 3 else (x.shape[0], x.shape[1] * x.shape[2], x.shape[3])

            # Manually compute attention to capture intermediates
            qkv_out = attn.qkv(x)

            # Reshape into Q, K, V (the exact reshape depends on implementation)
            # SAM3 ViT-H: num_heads=16, head_dim=64
            num_heads = attn.num_heads
            head_dim = C // num_heads

            # qkv shape: (B, N, 3*C)
            if qkv_out.dim() == 4:
                B, H, W, _ = qkv_out.shape
                qkv_flat = qkv_out.reshape(B, H * W, -1)
            else:
                qkv_flat = qkv_out

            qkv_reshaped = qkv_flat.reshape(B, -1, 3, num_heads, head_dim)
            q, k, v = qkv_reshaped.unbind(dim=2)  # each: (B, N, num_heads, head_dim)
            q = q.transpose(1, 2)  # (B, num_heads, N, head_dim)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)

            # Compute Q@K^T (attention scores before scaling)
            scores_raw = torch.matmul(q.float(), k.float().transpose(-2, -1))
            scale = head_dim ** -0.5
            scores_scaled = scores_raw * scale

            attention_stats.append({
                "block": block_idx,
                "q_abs_max": q.float().abs().max().item(),
                "k_abs_max": k.float().abs().max().item(),
                "v_abs_max": v.float().abs().max().item(),
                "q_std": q.float().std().item(),
                "k_std": k.float().std().item(),
                "qkv_proj_abs_max": qkv_out.float().abs().max().item(),
                "scores_raw_abs_max": scores_raw.abs().max().item(),
                "scores_raw_std": scores_raw.std().item(),
                "scores_scaled_abs_max": scores_scaled.abs().max().item(),
                "scores_scaled_std": scores_scaled.std().item(),
                "scores_overflow_fp16": scores_scaled.abs().max().item() > FP16_MAX,
                "head_dim": head_dim,
                "num_tokens": q.shape[2],
            })

            # Run original forward
            return orig_forward(x, *args, **kwargs)

        attn.forward = detailed_forward

    for i, block in enumerate(trunk.blocks):
        patch_block(i, block)

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        backbone.forward_image(dummy)

    return attention_stats


def main():
    print("Loading model...")
    model = build_sam3_image_model(
        device=DEVICE, checkpoint_path="sam3.pt", eval_mode=True,
    )

    # Use a real image and a random tensor
    dummy = torch.randn(1, 3, 1008, 1008, device=DEVICE)

    # ==========================================
    # Part 1: Layer-by-layer activation profiling
    # ==========================================
    print("\n" + "=" * 80)
    print("PART 1: Layer-by-layer activation statistics")
    print("=" * 80)

    stats, attn_int = profile_backbone(model, [dummy])

    # Print overflow layers
    print(f"\n{'Layer':<70s} | {'AbsMax':>10s} | {'Std':>8s} | FP16?")
    print("-" * 110)

    overflow_layers = []
    near_overflow_layers = []
    for name, s in stats.items():
        if s["overflow_fp16"]:
            overflow_layers.append((name, s))
            print(f"  {name:<68s} | {s['abs_max']:>10.1f} | {s['std']:>8.3f} | OVERFLOW ({s['pct_overflow']:.1f}%)")
        elif s["near_overflow"]:
            near_overflow_layers.append((name, s))

    if not overflow_layers:
        print("  No FP16 overflow detected in any layer!")
    else:
        print(f"\n  Total layers with FP16 overflow: {len(overflow_layers)}")

    if near_overflow_layers:
        print(f"\n  Layers near FP16 overflow (abs_max > {FP16_MAX/2:.0f}):")
        for name, s in near_overflow_layers[:20]:
            print(f"    {name:<66s} | {s['abs_max']:>10.1f}")

    # ==========================================
    # Part 2: Detailed attention profiling
    # ==========================================
    print("\n" + "=" * 80)
    print("PART 2: Attention internals (Q, K, V, scores)")
    print("=" * 80)

    # Reload model to clear patches
    del model
    torch.cuda.empty_cache()
    model = build_sam3_image_model(
        device=DEVICE, checkpoint_path="sam3.pt", eval_mode=True,
    )

    attn_stats = profile_attention_detailed(model, dummy)

    print(f"\n{'Block':>5s} | {'Q abs_max':>10s} | {'K abs_max':>10s} | {'QKV proj':>10s} | "
          f"{'Score raw':>10s} | {'Score scl':>10s} | {'Q std':>8s} | Overflow?")
    print("-" * 105)

    for s in attn_stats:
        overflow_marker = "OVERFLOW!" if s["scores_overflow_fp16"] else ""
        raw_overflow = "raw>65k" if s["scores_raw_abs_max"] > FP16_MAX else ""
        print(f"  {s['block']:>3d}   | {s['q_abs_max']:>10.1f} | {s['k_abs_max']:>10.1f} | "
              f"{s['qkv_proj_abs_max']:>10.1f} | {s['scores_raw_abs_max']:>10.1f} | "
              f"{s['scores_scaled_abs_max']:>10.1f} | {s['q_std']:>8.3f} | {raw_overflow} {overflow_marker}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    max_q = max(s["q_abs_max"] for s in attn_stats)
    max_k = max(s["k_abs_max"] for s in attn_stats)
    max_score_raw = max(s["scores_raw_abs_max"] for s in attn_stats)
    max_score_scaled = max(s["scores_scaled_abs_max"] for s in attn_stats)
    max_qkv = max(s["qkv_proj_abs_max"] for s in attn_stats)
    n_overflow_raw = sum(1 for s in attn_stats if s["scores_raw_abs_max"] > FP16_MAX)
    n_overflow_scaled = sum(1 for s in attn_stats if s["scores_overflow_fp16"])

    print(f"  FP16 max value:           {FP16_MAX}")
    print(f"  Max |Q|:                  {max_q:.1f}")
    print(f"  Max |K|:                  {max_k:.1f}")
    print(f"  Max |QKV proj output|:    {max_qkv:.1f}")
    print(f"  Max |Q@K^T| (raw):        {max_score_raw:.1f}")
    print(f"  Max |Q@K^T| (scaled):     {max_score_scaled:.1f}")
    print(f"  Blocks with raw overflow: {n_overflow_raw}/32")
    print(f"  Blocks with scl overflow: {n_overflow_scaled}/32")
    print(f"  Head dim:                 {attn_stats[0]['head_dim']}")
    print(f"  Scale factor:             {attn_stats[0]['head_dim'] ** -0.5:.6f}")

    if max_score_scaled <= FP16_MAX:
        print("\n  FINDING: Scaled attention scores fit in FP16!")
        print("  The overflow must be in the QKV projection or MLP layers.")
        print(f"  Max QKV projection output: {max_qkv:.1f}")
        if max_qkv > FP16_MAX:
            print("  → QKV PROJECTION overflows FP16!")
        else:
            print("  → QKV projection fits in FP16 too. Check MLP layers.")
    else:
        print(f"\n  FINDING: Attention scores OVERFLOW FP16!")
        print(f"  Need to scale Q or K by ~{(max_score_scaled / FP16_MAX):.1f}x to fix.")


if __name__ == "__main__":
    main()
