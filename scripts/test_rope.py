#!/usr/bin/env python3
"""Minimal test: compare original vs real-valued RoPE on identical inputs."""

import torch
from sam3.model.vitdet import apply_rotary_enc, compute_axial_cis, Attention
from sam3.trt.rope_onnx import apply_rotary_enc_real


def test_rope_math():
    """Test that apply_rotary_enc_real matches apply_rotary_enc exactly."""
    print("=== Test 1: Direct math comparison ===")

    # Create typical RoPE frequencies (2D axial, like the ViT uses)
    head_dim = 80  # typical for ViT-H with 16 heads, 1280 / 16 = 80
    L = 72 * 72    # sequence length for 1008/14 = 72 patches

    freqs_cis = compute_axial_cis(
        dim=head_dim, end_x=72, end_y=72
    )  # (5184, 40) complex
    print(f"freqs_cis: shape={freqs_cis.shape}, dtype={freqs_cis.dtype}")

    # Decompose to cos/sin
    rope_cos = freqs_cis.real.contiguous()
    rope_sin = freqs_cis.imag.contiguous()

    # Random q, k inputs (typical shape: B=1, num_heads=16, L=5184, head_dim=80)
    B, H = 1, 16
    q = torch.randn(B, H, L, head_dim)
    k = torch.randn(B, H, L, head_dim)

    # Original (complex-valued)
    q_orig, k_orig = apply_rotary_enc(q, k, freqs_cis=freqs_cis)

    # Replacement (real-valued)
    q_real, k_real = apply_rotary_enc_real(q, k, rope_cos, rope_sin)

    # Compare
    q_diff = (q_orig.float() - q_real.float()).abs()
    k_diff = (k_orig.float() - k_real.float()).abs()
    print(f"q: max_diff={q_diff.max().item():.2e}, mean_diff={q_diff.mean().item():.2e}")
    print(f"k: max_diff={k_diff.max().item():.2e}, mean_diff={k_diff.mean().item():.2e}")

    if q_diff.max() < 1e-5 and k_diff.max() < 1e-5:
        print("PASS: RoPE math is equivalent")
    else:
        print("FAIL: RoPE math differs!")
        # Find where the max diff is
        idx = q_diff.argmax()
        print(f"  Max diff at flat index {idx.item()}")
        print(f"  q_orig value: {q_orig.flatten()[idx].item():.6f}")
        print(f"  q_real value: {q_real.flatten()[idx].item():.6f}")


def test_rope_with_cls_token():
    """Test with cls token prepended (identity rotation)."""
    print("\n=== Test 2: With cls token ===")

    head_dim = 80
    L = 72 * 72

    freqs_cis = compute_axial_cis(dim=head_dim, end_x=72, end_y=72)
    # Prepend cls token row (identity rotation: cos=1, sin=0)
    t = torch.zeros(head_dim // 2, dtype=torch.float32)
    cls_freqs = torch.polar(torch.ones_like(t), t)[None, :]
    freqs_cis = torch.cat([cls_freqs, freqs_cis], dim=0)  # (L+1, 40)
    print(f"freqs_cis with cls: shape={freqs_cis.shape}")

    rope_cos = freqs_cis.real.contiguous()
    rope_sin = freqs_cis.imag.contiguous()

    B, H = 1, 16
    q = torch.randn(B, H, L + 1, head_dim)
    k = torch.randn(B, H, L + 1, head_dim)

    q_orig, k_orig = apply_rotary_enc(q, k, freqs_cis=freqs_cis)
    q_real, k_real = apply_rotary_enc_real(q, k, rope_cos, rope_sin)

    q_diff = (q_orig.float() - q_real.float()).abs()
    k_diff = (k_orig.float() - k_real.float()).abs()
    print(f"q: max_diff={q_diff.max().item():.2e}, mean_diff={q_diff.mean().item():.2e}")
    print(f"k: max_diff={k_diff.max().item():.2e}, mean_diff={k_diff.mean().item():.2e}")

    if q_diff.max() < 1e-5:
        print("PASS")
    else:
        print("FAIL")


def test_rope_windowed():
    """Test with windowed attention (shorter sequence)."""
    print("\n=== Test 3: Windowed attention (L=196) ===")

    head_dim = 80
    window_size = 14
    L = window_size * window_size  # 196

    freqs_cis = compute_axial_cis(
        dim=head_dim, end_x=window_size, end_y=window_size
    )  # (196, 40) complex
    print(f"freqs_cis: shape={freqs_cis.shape}")

    rope_cos = freqs_cis.real.contiguous()
    rope_sin = freqs_cis.imag.contiguous()

    B, H = 1, 16
    q = torch.randn(B, H, L, head_dim)
    k = torch.randn(B, H, L, head_dim)

    q_orig, k_orig = apply_rotary_enc(q, k, freqs_cis=freqs_cis)
    q_real, k_real = apply_rotary_enc_real(q, k, rope_cos, rope_sin)

    q_diff = (q_orig.float() - q_real.float()).abs()
    print(f"q: max_diff={q_diff.max().item():.2e}, mean_diff={q_diff.mean().item():.2e}")

    if q_diff.max() < 1e-5:
        print("PASS")
    else:
        print("FAIL")


def test_full_patching():
    """Test patching on actual model Attention modules."""
    print("\n=== Test 4: Full backbone patching ===")

    from sam3.model_builder import build_sam3_image_model
    from sam3.trt.rope_onnx import patch_rope_for_export, unpatch_rope

    print("Loading model...")
    model = build_sam3_image_model(
        checkpoint_path="sam3.pt", device="cuda", eval_mode=True
    )
    backbone = model.backbone

    # Run original
    dummy = torch.randn(1, 3, 1008, 1008, device="cuda")
    with torch.inference_mode():
        orig_out = backbone.forward_image(dummy)

    # Patch and run
    patch_rope_for_export(backbone)
    with torch.inference_mode():
        patched_out = backbone.forward_image(dummy)
    unpatch_rope(backbone)

    # Compare
    for i, (name, shape) in enumerate(
        [("FPN[0] 288x288", None), ("FPN[1] 144x144", None), ("FPN[2] 72x72", None)]
    ):
        orig_f = orig_out["backbone_fpn"][i].float()
        patch_f = patched_out["backbone_fpn"][i].float()
        diff = (orig_f - patch_f).abs()
        cos = torch.nn.functional.cosine_similarity(
            orig_f.flatten().unsqueeze(0), patch_f.flatten().unsqueeze(0)
        )
        print(f"  {name}: max_diff={diff.max().item():.2e}, "
              f"mean_diff={diff.mean().item():.2e}, cosine={cos.item():.6f}")

    # Also test per-module: hook into each attention module and compare
    print("\n  Per-module RoPE comparison (first 5 modules):")
    trunk = backbone.vision_backbone.trunk
    count = 0
    for name, module in trunk.named_modules():
        if not isinstance(module, Attention) or not module.use_rope:
            continue
        if count >= 5:
            break

        freqs_cis = module.freqs_cis
        rope_cos = freqs_cis.real.contiguous()
        rope_sin = freqs_cis.imag.contiguous()

        # Random input matching this module's expected shapes
        L = freqs_cis.shape[0]
        q = torch.randn(1, module.num_heads, L, module.head_dim, device="cuda")
        k = torch.randn(1, module.num_heads, L, module.head_dim, device="cuda")

        q_orig, k_orig = apply_rotary_enc(q, k, freqs_cis=freqs_cis)
        q_real, k_real = apply_rotary_enc_real(q, k, rope_cos, rope_sin)

        q_diff = (q_orig.float() - q_real.float()).abs()
        print(f"    {name}: L={L}, max_diff={q_diff.max().item():.2e}")
        count += 1


if __name__ == "__main__":
    # CPU tests first (fast)
    test_rope_math()
    test_rope_with_cls_token()
    test_rope_windowed()

    # GPU test with actual model
    import sys
    if "--full" in sys.argv:
        test_full_patching()
    else:
        print("\nRun with --full to test on the actual model backbone")
