# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""Export SAM3 encoder+decoder+scoring pipeline to ONNX.

Exports the full detection head (TransformerEncoderFusion + TransformerDecoder +
DotProductScoring + box prediction) as a single ONNX model with fixed shapes.
The backbone and text encoder remain outside the ONNX graph.

Designed for ``--fast --detection-only`` mode:
  - apply_dac = False
  - boxRPB = "log" (pre-warmed coord cache for spatial_size x spatial_size)
  - presence_token: disabled by default (use --presence to include)

Usage:
    python -m sam3.trt.export_enc_dec \\
        --checkpoint path/to/sam3.pt \\
        --output enc_dec.onnx \\
        --max-classes 4 \\
        --imgsz 1008
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn

from sam3.model.model_misc import inverse_sigmoid


class _EncDecForExport(nn.Module):
    """Wrap encoder+decoder+scoring+box-prediction for ONNX export.

    Fixed-shape inputs:
        img_feat:   (max_bs, 256, H, W)   — single FPN level
        img_pos:    (max_bs, 256, H, W)   — position encoding
        text_feats: (32, max_bs, 256)      — per-class text features
        text_mask:  (max_bs, 32)           — text padding mask (True=padding)

    Fixed-shape outputs:
        scores:   (max_bs, 200, 1)  — detection logits
        boxes:    (max_bs, 200, 4)  — cxcywh coordinates (sigmoid)
        presence: (max_bs, 1)       — presence logits (only if use_presence=True)
    """

    def __init__(self, encoder, decoder, scoring, spatial_size, use_presence=True):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.scoring = scoring
        self.spatial_size = spatial_size  # (H, W) tuple
        self.use_presence = use_presence

    def forward(self, img_feat, img_pos, text_feats, text_mask):
        bs = img_feat.shape[0]

        # Cast mask from float32 (TRT-friendly) to bool (PyTorch-native)
        text_mask = text_mask.to(torch.bool)

        # Convert NCHW → seq-first (H*W, bs, d) for encoder input
        img_seq = img_feat.flatten(2).permute(2, 0, 1)   # (H*W, bs, 256)
        pos_seq = img_pos.flatten(2).permute(2, 0, 1)    # (H*W, bs, 256)

        # --- Encoder ---
        memory = self.encoder(
            src=[img_seq],
            src_key_padding_mask=None,
            src_pos=[pos_seq],
            prompt=text_feats,
            prompt_key_padding_mask=text_mask,
            feat_sizes=[self.spatial_size],
        )

        enc_hs = memory["memory"]      # (5184, bs, 256) seq-first
        enc_pos = memory["pos_embed"]   # (5184, bs, 256) seq-first

        # --- Decoder ---
        query_embed = self.decoder.query_embed.weight
        tgt = query_embed.unsqueeze(1).expand(-1, bs, -1)  # (200, bs, 256)

        hs, ref_boxes, dec_presence_out, _ = self.decoder(
            tgt=tgt,
            memory=enc_hs,
            memory_key_padding_mask=memory["padding_mask"],
            pos=enc_pos,
            reference_boxes=None,
            level_start_index=memory["level_start_index"],
            spatial_shapes=memory["spatial_shapes"],
            valid_ratios=memory["valid_ratios"],
            tgt_mask=None,
            memory_text=text_feats,
            text_attention_mask=text_mask,
            apply_dac=False,
        )

        # hs: (6, 200, bs, 256) → (6, bs, 200, 256)
        hs = hs.transpose(1, 2)
        ref_boxes = ref_boxes.transpose(1, 2)

        # Last layer only
        hs_last = hs[-1:]         # (1, bs, 200, 256)
        ref_last = ref_boxes[-1]  # (bs, 200, 4)

        # --- Scoring ---
        scores = self.scoring(hs_last, text_feats, text_mask)  # (1, bs, 200, 1)
        scores = scores[0]  # (bs, 200, 1)

        # --- Box prediction ---
        box_offsets = self.decoder.bbox_embed(hs_last)  # (1, bs, 200, 4)
        box_offsets = box_offsets[0]  # (bs, 200, 4)
        ref_inv = inverse_sigmoid(ref_last)
        boxes = (ref_inv + box_offsets).sigmoid()  # (bs, 200, 4)

        # --- Presence logits ---
        if self.use_presence and dec_presence_out is not None:
            # dec_presence_out: (num_layers, 1, bs) → last layer → (1, bs)
            presence_logits = dec_presence_out[-1]  # (1, bs)
            presence_logits = presence_logits.permute(1, 0)  # (bs, 1)
            return scores, boxes, presence_logits

        return scores, boxes


def export_onnx(
    checkpoint_path: str,
    output_path: str,
    max_classes: int = 4,
    imgsz: int = 1008,
    opset_version: int = 14,
    validate: bool = True,
    efficient_backbone: str = None,
    efficient_model: str = None,
    presence: bool = False,
):
    """Export the encoder+decoder+scoring pipeline to ONNX."""
    spatial = imgsz // 14
    assert imgsz % 14 == 0, f"imgsz must be divisible by 14, got {imgsz}"
    print(f"Target resolution: imgsz={imgsz} → spatial={spatial}x{spatial}")

    if efficient_backbone:
        from sam3.efficient_backbone import build_efficientsam3_model
        print(f"Loading EfficientSAM3 ({efficient_backbone} {efficient_model}) from {checkpoint_path} ...")
        model = build_efficientsam3_model(
            backbone_type=efficient_backbone,
            model_name=efficient_model,
            checkpoint_path=checkpoint_path,
            device="cuda",
            eval_mode=True,
        )
    else:
        from sam3.model_builder import build_sam3_image_model
        print(f"Loading SAM3 model from {checkpoint_path} ...")
        model = build_sam3_image_model(
            checkpoint_path=checkpoint_path,
            device="cuda",
            eval_mode=True,
            load_from_HF=False,
            enable_segmentation=False,
        )

    encoder = model.transformer.encoder
    decoder = model.transformer.decoder
    scoring = model.dot_prod_scoring

    # Presence token handling
    use_presence = presence and decoder.presence_token is not None
    if not use_presence:
        decoder.presence_token = None
        print("Presence token disabled (detection-only mode)")
    else:
        print("Presence token enabled — will export presence logits as 3rd output")

    # Disable torch.compile in decoder (prevent self-compilation during tracing)
    decoder.compile_mode = None
    decoder.compiled = True

    # Re-warm boxRPB coord cache for the target spatial size
    coords_h, coords_w = decoder._get_coords(spatial, spatial, device="cuda")
    decoder.compilable_cord_cache = (coords_h, coords_w)
    decoder.compilable_stored_size = (spatial, spatial)
    print(f"boxRPB coord cache warmed for ({spatial}, {spatial})")

    spatial_size = (spatial, spatial)

    # Create export module
    export_module = _EncDecForExport(
        encoder, decoder, scoring, spatial_size, use_presence=use_presence
    ).cuda().eval()

    # Dummy inputs (fixed shapes)
    dummy_img = torch.randn(max_classes, 256, spatial, spatial, device="cuda")
    dummy_pos = torch.randn(max_classes, 256, spatial, spatial, device="cuda")
    dummy_text = torch.randn(32, max_classes, 256, device="cuda")
    # Use float32 for mask input (TRT handles float reliably; cast to bool in wrapper)
    dummy_mask = torch.zeros(max_classes, 32, dtype=torch.float32, device="cuda")
    # Mark some classes as padding (last few)
    dummy_mask[max_classes // 2:, :] = 1.0

    # Warm-up forward pass
    print("Running warm-up forward pass ...")
    with torch.no_grad():
        outputs = export_module(dummy_img, dummy_pos, dummy_text, dummy_mask)
    if use_presence:
        scores_pt, boxes_pt, presence_pt = outputs
        print(f"  PyTorch output shapes: scores={scores_pt.shape}, boxes={boxes_pt.shape}, presence={presence_pt.shape}")
    else:
        scores_pt, boxes_pt = outputs
        print(f"  PyTorch output shapes: scores={scores_pt.shape}, boxes={boxes_pt.shape}")

    # Validate against direct PyTorch forward
    if validate:
        print("Validating wrapper output matches direct PyTorch forward ...")
        with torch.no_grad():
            # Run the same pipeline manually (as in _forward_batched)
            img_seq = dummy_img.flatten(2).permute(2, 0, 1)
            pos_seq = dummy_pos.flatten(2).permute(2, 0, 1)

            # Cast mask to bool (same as wrapper does)
            dummy_mask_bool = dummy_mask.to(torch.bool)
            memory = encoder(
                src=[img_seq],
                src_key_padding_mask=None,
                src_pos=[pos_seq],
                prompt=dummy_text,
                prompt_key_padding_mask=dummy_mask_bool,
                feat_sizes=[spatial_size],
            )

            query_embed = decoder.query_embed.weight
            tgt = query_embed.unsqueeze(1).expand(-1, max_classes, -1)

            hs, ref_boxes, dec_pres_val, _ = decoder(
                tgt=tgt,
                memory=memory["memory"],
                memory_key_padding_mask=memory["padding_mask"],
                pos=memory["pos_embed"],
                reference_boxes=None,
                level_start_index=memory["level_start_index"],
                spatial_shapes=memory["spatial_shapes"],
                valid_ratios=memory["valid_ratios"],
                tgt_mask=None,
                memory_text=dummy_text,
                text_attention_mask=dummy_mask_bool,
                apply_dac=False,
            )

            hs = hs.transpose(1, 2)
            ref_boxes = ref_boxes.transpose(1, 2)

            hs_last = hs[-1:]
            ref_last = ref_boxes[-1]

            scores_direct = scoring(hs_last, dummy_text, dummy_mask_bool)[0]
            box_offsets = decoder.bbox_embed(hs_last)[0]
            boxes_direct = (inverse_sigmoid(ref_last) + box_offsets).sigmoid()

        score_diff = (scores_pt - scores_direct).abs().max().item()
        box_diff = (boxes_pt - boxes_direct).abs().max().item()
        print(f"  scores max diff: {score_diff:.2e}")
        print(f"  boxes max diff:  {box_diff:.2e}")
        if score_diff > 1e-4 or box_diff > 1e-4:
            print("  WARNING: large difference between wrapper and direct forward!")

    # ONNX export
    input_names = ["img_feat", "img_pos", "text_feats", "text_mask"]
    output_names = ["scores", "boxes"]
    if use_presence:
        output_names.append("presence")

    print(f"Exporting to ONNX (opset {opset_version}) -> {output_path} ...")
    with torch.no_grad():
        torch.onnx.export(
            export_module,
            (dummy_img, dummy_pos, dummy_text, dummy_mask),
            output_path,
            opset_version=opset_version,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=None,  # fully static shapes
        )

    # Simplify if onnxsim is available.
    # Run in a subprocess because onnxsim's C++ backend can crash (segfault /
    # OOM kill) on large models, which would kill the main process.
    import shutil
    import subprocess
    if shutil.which("python"):
        print("Running onnxsim in subprocess ...")
        result = subprocess.run(
            ["python", "-m", "onnxsim", output_path, output_path],
            capture_output=True, text=True, timeout=300,
        )
        if result.returncode == 0:
            print("  Simplified successfully.")
        else:
            stderr = result.stderr.strip().split("\n")[-1] if result.stderr else ""
            print(f"  onnxsim failed (rc={result.returncode}): {stderr}")
            print("  Keeping original ONNX (works fine with TRT).")
    else:
        print("  Skipping onnxsim (python not found in PATH).")

    # Print file size
    size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    print(f"Done. ONNX model saved: {output_path} ({size_mb:.1f} MB)")

    # Validate ONNX model loads
    try:
        import onnx

        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX model validation passed.")
    except ImportError:
        pass


def main():
    parser = argparse.ArgumentParser(
        description="Export SAM3 encoder+decoder to ONNX"
    )
    parser.add_argument(
        "--checkpoint", required=True, help="Path to SAM3 checkpoint (.pt)"
    )
    parser.add_argument(
        "--output", default="enc_dec.onnx", help="Output ONNX file path"
    )
    parser.add_argument(
        "--max-classes",
        type=int,
        default=4,
        help="Maximum number of classes (batch dimension)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=1008,
        help="Input image resolution (must be divisible by 14). "
             "Default 1008 → 72x72 spatial features. "
             "Use 672 for faster inference (48x48).",
    )
    parser.add_argument(
        "--opset", type=int, default=14, help="ONNX opset version (14 recommended for TRT compatibility)"
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip validation of wrapper vs direct outputs",
    )
    parser.add_argument(
        "--presence",
        action="store_true",
        help="Include presence token logits as 3rd output (default: disabled)",
    )
    parser.add_argument(
        "--efficient-backbone",
        type=str,
        default=None,
        choices=["efficientvit", "repvit", "tinyvit"],
        help="Use EfficientSAM3 lightweight backbone checkpoint",
    )
    parser.add_argument(
        "--efficient-model",
        type=str,
        default=None,
        help="Backbone variant (e.g. b0/b1/b2, m0_9/m1_1/m2_3, 5m/11m/21m)",
    )
    args = parser.parse_args()

    if args.efficient_backbone and not args.efficient_model:
        parser.error("--efficient-backbone requires --efficient-model")

    export_onnx(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        max_classes=args.max_classes,
        imgsz=args.imgsz,
        opset_version=args.opset,
        validate=not args.no_validate,
        efficient_backbone=args.efficient_backbone,
        efficient_model=args.efficient_model,
        presence=args.presence,
    )


if __name__ == "__main__":
    main()
