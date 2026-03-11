#!/usr/bin/env python3
"""Diagnose where FP16 precision breaks in the ViT backbone.

Tests:
  1. FP32 intermediate activations: check for FP16-unsafe values (>65504)
  2. Block-by-block FP16 round-trip error
  3. Pure FP16 forward vs FP32 forward (block comparison)
  4. Attention internals: Q, K, V, attention logits range check

Usage:
    python scripts/diagnose_fp16.py
"""

import math
import torch
import torch.nn.functional as F


def main():
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.vitdet import Attention

    print("Loading model...")
    model = build_sam3_image_model(
        checkpoint_path="sam3.pt", device="cuda", eval_mode=True
    )
    backbone = model.backbone
    trunk = backbone.vision_backbone.trunk

    dummy = torch.randn(1, 3, 1008, 1008, device="cuda")

    # =========================================================================
    # TEST 1: FP32 forward — check intermediate activation ranges
    # =========================================================================
    print("\n" + "=" * 70)
    print("TEST 1: FP32 activation ranges (checking for FP16-unsafe values)")
    print("=" * 70)

    block_outputs_fp32 = {}

    def make_block_hook(name, storage):
        def hook(module, inp, out):
            storage[name] = out.detach().clone()
        return hook

    handles = []
    for i, blk in enumerate(trunk.blocks):
        h = blk.register_forward_hook(make_block_hook(f"block_{i}", block_outputs_fp32))
        handles.append(h)

    with torch.inference_mode():
        out_fp32 = backbone.forward_image(dummy)
    for h in handles:
        h.remove()

    print(f"{'Block':>6} {'WinSz':>5} {'MaxAbs':>10} {'>65504?':>8} "
          f"{'FP16 RT Err':>12} {'Std':>10} {'Mean':>10}")
    for i, blk in enumerate(trunk.blocks):
        name = f"block_{i}"
        act = block_outputs_fp32[name].float()
        max_abs = act.abs().max().item()
        overflow = max_abs > 65504
        rt_err = (act - act.half().float()).abs().max().item()
        std = act.std().item()
        mean = act.mean().item()
        ws = blk.window_size
        flag = " <<<" if overflow else ""
        print(f"{i:>6d} {ws:>5d} {max_abs:>10.1f} {'YES' if overflow else 'no':>8s} "
              f"{rt_err:>12.4e} {std:>10.4f} {mean:>10.4f}{flag}")

    # =========================================================================
    # TEST 2: Attention internals — check Q, K, attention logit ranges
    # =========================================================================
    print("\n" + "=" * 70)
    print("TEST 2: Attention internals (Q, K, logits ranges)")
    print("=" * 70)

    attn_stats = {}

    def make_attn_hook(block_idx, storage):
        """Hook that captures pre/post attention stats."""
        def hook(module, inp, out):
            x_in = inp[0]
            B = x_in.shape[0]
            if x_in.ndim == 4:
                L = x_in.shape[1] * x_in.shape[2]
            else:
                L = x_in.shape[1]

            # Recompute QKV to inspect intermediates
            with torch.inference_mode():
                qkv = module.qkv(x_in).reshape(B, L, 3, module.num_heads, -1)
                q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)

                q, k = module._apply_rope(q, k)

                # Check if concat_rel_pos is used
                if module.use_rel_pos and hasattr(module, 'rel_pos_h') and module.rel_pos_h is not None:
                    from sam3.model.vitdet import concat_rel_pos
                    s = 1 if module.cls_token else 0
                    if x_in.ndim == 4:
                        H, W = x_in.shape[1], x_in.shape[2]
                    else:
                        H = W = int(math.sqrt(L - s))
                    q, k = concat_rel_pos(
                        q, k, module.rel_pos_h, module.rel_pos_w,
                        q_size=(H, W), k_size=(H, W),
                        cls_token=module.cls_token,
                    )

                head_dim_for_scale = q.shape[-1]
                scale = head_dim_for_scale ** -0.5

                # Compute raw attention logits (before scale)
                raw_logits = q @ k.transpose(-2, -1)
                # Scaled logits
                scaled_logits = raw_logits * scale

            storage[block_idx] = {
                "q_max": q.abs().max().item(),
                "k_max": k.abs().max().item(),
                "v_max": v.abs().max().item(),
                "q_shape": tuple(q.shape),
                "raw_logits_max": raw_logits.abs().max().item(),
                "scaled_logits_max": scaled_logits.abs().max().item(),
                "scaled_logits_fp16_max": scaled_logits.half().float().abs().max().item(),
                "logit_fp16_err": (scaled_logits - scaled_logits.half().float()).abs().max().item(),
                "head_dim": head_dim_for_scale,
                "scale": scale,
            }
        return hook

    handles = []
    for i, blk in enumerate(trunk.blocks):
        for name, module in blk.named_modules():
            if isinstance(module, Attention):
                h = module.register_forward_hook(make_attn_hook(i, attn_stats))
                handles.append(h)

    with torch.inference_mode():
        _ = backbone.forward_image(dummy)
    for h in handles:
        h.remove()

    print(f"{'Block':>6} {'WinSz':>5} {'Q shape':>22} {'|Q|max':>8} {'|K|max':>8} "
          f"{'|V|max':>8} {'|logit|':>8} {'scaled':>8} {'FP16err':>10} {'hdim':>5}")
    for i in sorted(attn_stats.keys()):
        s = attn_stats[i]
        ws = trunk.blocks[i].window_size
        print(f"{i:>6d} {ws:>5d} {str(s['q_shape']):>22s} "
              f"{s['q_max']:>8.1f} {s['k_max']:>8.1f} {s['v_max']:>8.1f} "
              f"{s['raw_logits_max']:>8.1f} {s['scaled_logits_max']:>8.1f} "
              f"{s['logit_fp16_err']:>10.2e} {s['head_dim']:>5d}")

    # =========================================================================
    # TEST 3: Pure FP16 forward vs FP32 (does PyTorch FP16 also break?)
    # =========================================================================
    print("\n" + "=" * 70)
    print("TEST 3: Pure FP16 model (.half()) vs FP32 — block comparison")
    print("=" * 70)

    # Patch RoPE to real-valued so .half() doesn't break on complex buffers
    from sam3.trt.rope_onnx import patch_rope_for_export, unpatch_rope
    patch_rope_for_export(backbone)

    # Clone the model state to create FP16 version
    import copy
    backbone_fp16 = copy.deepcopy(backbone).half()

    block_outputs_fp16 = {}
    handles = []
    for i, blk in enumerate(backbone_fp16.vision_backbone.trunk.blocks):
        h = blk.register_forward_hook(make_block_hook(f"block_{i}", block_outputs_fp16))
        handles.append(h)

    with torch.inference_mode():
        out_fp16 = backbone_fp16.forward_image(dummy.half())
    for h in handles:
        h.remove()

    # Also get FP32 with patched RoPE for fair comparison
    block_outputs_fp32_patched = {}
    handles = []
    for i, blk in enumerate(trunk.blocks):
        h = blk.register_forward_hook(make_block_hook(f"block_{i}", block_outputs_fp32_patched))
        handles.append(h)

    with torch.inference_mode():
        out_fp32_patched = backbone.forward_image(dummy)
    for h in handles:
        h.remove()

    unpatch_rope(backbone)

    print(f"{'Block':>6} {'WinSz':>5} {'MaxDiff':>10} {'MeanDiff':>10} "
          f"{'Cosine':>10} {'FP16 NaN':>8} {'FP16 Inf':>8} {'FP16 MaxAbs':>11}")
    any_bad = False
    for i in range(len(trunk.blocks)):
        name = f"block_{i}"
        ws = trunk.blocks[i].window_size
        fp32_act = block_outputs_fp32_patched[name].float()
        fp16_act = block_outputs_fp16[name].float()

        diff = (fp32_act - fp16_act).abs()
        cos = F.cosine_similarity(
            fp32_act.flatten().unsqueeze(0),
            fp16_act.flatten().unsqueeze(0),
        )
        has_nan = fp16_act.isnan().any().item()
        has_inf = fp16_act.isinf().any().item()
        max_abs = fp16_act.abs().max().item()
        flag = ""
        if cos.item() < 0.99:
            flag = " <<< DIVERGED"
            any_bad = True
        if has_nan or has_inf:
            flag = " <<< NaN/Inf!"
            any_bad = True
        print(f"{i:>6d} {ws:>5d} {diff.max().item():>10.4e} {diff.mean().item():>10.4e} "
              f"{cos.item():>10.6f} {'YES' if has_nan else 'no':>8s} "
              f"{'YES' if has_inf else 'no':>8s} {max_abs:>11.1f}{flag}")

    # FPN comparison
    print("\nFPN output comparison (FP16 vs FP32):")
    fp32_fpn = out_fp32_patched["backbone_fpn"]
    fp16_fpn = out_fp16["backbone_fpn"]
    for i in range(len(fp32_fpn)):
        f32 = fp32_fpn[i].float()
        f16 = fp16_fpn[i].float()
        diff = (f32 - f16).abs()
        cos = F.cosine_similarity(f32.flatten().unsqueeze(0), f16.flatten().unsqueeze(0))
        print(f"  FPN[{i}]: max_diff={diff.max().item():.4e}, cosine={cos.item():.6f}")

    if not any_bad:
        print("\nPure FP16 in PyTorch works fine — the issue is TRT-specific.")
        print("The problem is likely in ONNX export (SDPA decomposition, CONDITION block,")
        print("or how TRT optimizes the graph in FP16 mode).")
    else:
        print("\nPure FP16 ALSO breaks in PyTorch — the model has FP16 incompatibility.")

    # Cleanup
    del backbone_fp16
    torch.cuda.empty_cache()

    # =========================================================================
    # TEST 4: Check ONNX export — does SDPA decompose correctly?
    # =========================================================================
    print("\n" + "=" * 70)
    print("TEST 4: ONNX model structure analysis")
    print("=" * 70)
    try:
        import onnx
        model_onnx = onnx.load("backbone.onnx")
        graph = model_onnx.graph

        # Count op types
        from collections import Counter
        op_counts = Counter(n.op_type for n in graph.node)
        print(f"ONNX graph: {len(graph.node)} nodes")
        print("Top ops:")
        for op, count in op_counts.most_common(15):
            print(f"  {op}: {count}")

        # Check for If/Loop nodes (control flow)
        if_nodes = [n for n in graph.node if n.op_type == "If"]
        print(f"\nConditional (If) nodes: {len(if_nodes)}")
        for n in if_nodes:
            print(f"  {n.name}: inputs={[i for i in n.input]}")

        # Check for any Concat with shape issues
        concat_nodes = [n for n in graph.node if n.op_type == "Concat"]
        print(f"Concat nodes: {len(concat_nodes)}")

    except ImportError:
        print("onnx not installed, skipping")


if __name__ == "__main__":
    main()
