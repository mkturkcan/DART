# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""ONNX-compatible RoPE 2D replacement for ViT backbone export.

The original RoPE in vitdet.py uses torch.polar, torch.view_as_complex, and
torch.view_as_real — none of which are supported in ONNX.  This module
provides a mathematically equivalent implementation using only real-valued
arithmetic (multiply, add, stack, reshape) that exports cleanly to ONNX.

Usage:
    from sam3.trt.rope_onnx import patch_rope_for_export, unpatch_rope

    patch_rope_for_export(backbone)   # before torch.onnx.export
    torch.onnx.export(...)
    unpatch_rope(backbone)            # restore original if needed
"""

from typing import Tuple

import torch
from torch import Tensor


def apply_rotary_enc_real(
    xq: Tensor,
    xk: Tensor,
    rope_cos: Tensor,
    rope_sin: Tensor,
    repeat_freqs_k: bool = False,
) -> Tuple[Tensor, Tensor]:
    """RoPE via real-valued arithmetic (ONNX-safe).

    Complex multiplication (a+bi)(c+di) = (ac-bd) + (ad+bc)i
    where freqs_cis = cos + i*sin, so c=cos, d=sin.

    Args:
        xq: Query tensor (B, num_heads, L, head_dim).
        xk: Key tensor (B, num_heads, L_k, head_dim).
        rope_cos: Cosine buffer (L, head_dim//2).
        rope_sin: Sine buffer (L, head_dim//2).
        repeat_freqs_k: Tile freqs along seq dim if k is longer than q.
    """
    # Split into even/odd pairs: (B, H, L, D) → (B, H, L, D//2, 2)
    xq_pairs = xq.float().reshape(*xq.shape[:-1], -1, 2)
    xq_r = xq_pairs[..., 0]  # (B, H, L, D//2)
    xq_i = xq_pairs[..., 1]

    # Broadcast cos/sin: (L, D//2) → (1, 1, L, D//2)
    cos = rope_cos.unsqueeze(0).unsqueeze(0)
    sin = rope_sin.unsqueeze(0).unsqueeze(0)

    # Complex multiply: out = x * (cos + i*sin)
    oq_r = xq_r * cos - xq_i * sin
    oq_i = xq_r * sin + xq_i * cos
    xq_out = torch.stack([oq_r, oq_i], dim=-1).flatten(-2)
    xq_out = xq_out.to(xq.dtype)

    if xk.shape[-2] == 0:
        return xq_out, xk

    xk_pairs = xk.float().reshape(*xk.shape[:-1], -1, 2)
    xk_r = xk_pairs[..., 0]
    xk_i = xk_pairs[..., 1]

    if repeat_freqs_k:
        r = xk.shape[-2] // xq.shape[-2]
        cos = cos.repeat(1, 1, r, 1)
        sin = sin.repeat(1, 1, r, 1)

    ok_r = xk_r * cos - xk_i * sin
    ok_i = xk_r * sin + xk_i * cos
    xk_out = torch.stack([ok_r, ok_i], dim=-1).flatten(-2)
    xk_out = xk_out.to(xk.dtype)

    return xq_out, xk_out


def _make_real_apply_rope(rope_cos: Tensor, rope_sin: Tensor):
    """Create a bound _apply_rope replacement for one Attention module."""

    def _apply_rope_real(self, q: Tensor, k: Tensor) -> Tuple[Tensor, Tensor]:
        if not self.use_rope:
            return q, k
        return apply_rotary_enc_real(q, k, rope_cos, rope_sin)

    return _apply_rope_real


def patch_rope_for_export(backbone) -> None:
    """Replace complex-valued RoPE with real arithmetic on all Attention modules.

    Walks the backbone's ViT trunk, finds every Attention layer that uses RoPE,
    decomposes its complex ``freqs_cis`` buffer into real ``rope_cos`` /
    ``rope_sin`` buffers, and monkey-patches ``_apply_rope``.

    Stores originals on each module so ``unpatch_rope`` can restore them.
    """
    from sam3.model.vitdet import Attention

    trunk = backbone.vision_backbone.trunk

    for name, module in trunk.named_modules():
        if not isinstance(module, Attention):
            continue
        if not module.use_rope or module.freqs_cis is None:
            continue

        # Decompose complex buffer to real cos/sin
        freqs_cis = module.freqs_cis  # (L, head_dim//2) complex64
        rope_cos = freqs_cis.real.contiguous()  # (L, head_dim//2) float32
        rope_sin = freqs_cis.imag.contiguous()

        # Store originals for unpatch
        module._orig_freqs_cis = freqs_cis
        module._orig_apply_rope = module._apply_rope

        # Register real buffers
        module.register_buffer("rope_cos", rope_cos)
        module.register_buffer("rope_sin", rope_sin)

        # Replace _apply_rope with real-valued version
        import types

        real_fn = _make_real_apply_rope(module.rope_cos, module.rope_sin)
        module._apply_rope = types.MethodType(real_fn, module)

        # Remove complex buffer (ONNX can't serialize it)
        del module.freqs_cis
        module.freqs_cis = None


def unpatch_rope(backbone) -> None:
    """Restore original complex-valued RoPE on all patched Attention modules."""
    from sam3.model.vitdet import Attention

    trunk = backbone.vision_backbone.trunk

    for name, module in trunk.named_modules():
        if not isinstance(module, Attention):
            continue
        if not hasattr(module, "_orig_freqs_cis"):
            continue

        # Restore original buffer and method
        module.freqs_cis = module._orig_freqs_cis
        module._apply_rope = module._orig_apply_rope

        # Clean up
        if hasattr(module, "rope_cos"):
            del module.rope_cos
        if hasattr(module, "rope_sin"):
            del module.rope_sin
        del module._orig_freqs_cis
        del module._orig_apply_rope


# ---------------------------------------------------------------------------
# SDPA -> eager attention patching for ONNX export
# ---------------------------------------------------------------------------
# F.scaled_dot_product_attention produces ONNX graphs that TRT miscompiles
# in FP16 (cosine ~0.09 vs PyTorch). The HuggingFace SAM3 implementation uses
# explicit matmul+scale+softmax+matmul ("eager attention") and TRT FP16 works
# perfectly (cosine ~0.9999). This patch replaces SDPA with eager attention
# during export.


def _eager_sdpa(q: Tensor, k: Tensor, v: Tensor) -> Tensor:
    """Drop-in replacement for F.scaled_dot_product_attention using explicit ops.

    Equivalent to: softmax(Q @ K^T / sqrt(d_k)) @ V
    """
    scale = q.shape[-1] ** -0.5
    attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
    attn_weights = torch.softmax(attn_weights, dim=-1)
    return torch.matmul(attn_weights, v)


def _fp32_sdpa(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attn_mask: Tensor = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float = None,
    enable_gqa: bool = False,
) -> Tensor:
    """SDPA replacement that computes attention in FP32.

    Casts Q/K/V to FP32 for the attention computation, then casts back.
    This inserts explicit Cast ops into the ONNX graph that TRT will respect,
    keeping the attention numerics safe in FP16 engines without needing
    per-layer precision constraints.
    """
    orig_dtype = query.dtype

    # Cast to FP32 for attention computation
    q = query.float()
    k = key.float()
    v = value.float()

    if scale is None:
        scale = q.shape[-1] ** -0.5
    attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
    if attn_mask is not None:
        attn_weights = attn_weights + attn_mask.float()
    attn_weights = torch.softmax(attn_weights, dim=-1)
    out = torch.matmul(attn_weights, v)

    return out.to(orig_dtype)


# Store original SDPA for unpatch
_orig_sdpa = None


def patch_sdpa_for_export(backbone) -> None:
    """Monkey-patch F.scaled_dot_product_attention globally with eager attention.

    This preserves the original Attention.forward() graph structure (identical
    TorchScript trace) but replaces the fused SDPA kernel with explicit
    matmul+scale+softmax+matmul that TRT FP16 handles correctly.

    The backbone arg is accepted for API compatibility but not used — the
    patch is global on torch.nn.functional.
    """
    import torch.nn.functional as F

    global _orig_sdpa
    _orig_sdpa = F.scaled_dot_product_attention
    F.scaled_dot_product_attention = _fp32_sdpa

    # Also patch the module-level reference in vitdet.py
    from sam3.model import vitdet
    vitdet.F.scaled_dot_product_attention = _fp32_sdpa

    print("  Patched F.scaled_dot_product_attention -> FP32 attention (global)")


def unpatch_sdpa(backbone=None) -> None:
    """Restore original F.scaled_dot_product_attention."""
    import torch.nn.functional as F

    global _orig_sdpa
    if _orig_sdpa is not None:
        F.scaled_dot_product_attention = _orig_sdpa
        from sam3.model import vitdet
        vitdet.F.scaled_dot_product_attention = _orig_sdpa
        _orig_sdpa = None
        print("  Restored original F.scaled_dot_product_attention")
