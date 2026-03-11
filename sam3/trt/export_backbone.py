# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""Export SAM3 vision backbone (ViT + FPN neck) to ONNX.

Position encodings are deterministic from spatial size and are excluded from
the ONNX graph — the TRT runtime wrapper pre-computes them separately.

Supports two export modes:
  - **TorchScript** (default): Classic ``torch.onnx.export`` via TorchScript tracing.
  - **Dynamo** (``--dynamo``): Uses ``torch.onnx.export(dynamo=True)`` for
    FX-graph-based export (PyTorch 2.5+). Produces a fundamentally different
    ONNX graph that may avoid TRT FP16 numerical issues caused by how
    TorchScript decomposes ``F.scaled_dot_product_attention``.

Usage:
    # TorchScript export (original):
    python -m sam3.trt.export_backbone \\
        --checkpoint path/to/sam3.pt \\
        --output backbone.onnx

    # Dynamo export (recommended for FP16 TRT):
    python -m sam3.trt.export_backbone \\
        --checkpoint path/to/sam3.pt \\
        --output backbone_dynamo.onnx --dynamo
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


class _BackboneForExport(nn.Module):
    """Thin wrapper that calls forward_image and returns a flat tuple of FPN tensors.

    ONNX export requires a module returning tensors (not dicts/lists), so we
    flatten the 3 FPN levels into positional outputs.
    """

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, images: torch.Tensor):
        out = self.backbone.forward_image(images)
        fpn = out["backbone_fpn"]  # list of 3 tensors after scalp=1
        # Return flat tuple: fpn_0, fpn_1, fpn_2
        return fpn[0], fpn[1], fpn[2]


def _export_torchscript(export_module, dummy, output_path, opset_version, output_names):
    """Export via classic TorchScript tracing (original path)."""
    print(f"Exporting to ONNX via TorchScript (opset {opset_version}) -> {output_path} ...")
    with torch.no_grad():
        torch.onnx.export(
            export_module,
            (dummy,),
            output_path,
            opset_version=opset_version,
            input_names=["images"],
            output_names=output_names,
            dynamic_axes=None,  # fully static shape
        )

    # Simplify if onnxsim is available
    try:
        import onnx
        from onnxsim import simplify

        print("Running onnxsim.simplify() ...")
        model_onnx = onnx.load(output_path)
        model_simp, ok = simplify(model_onnx)
        if ok:
            onnx.save(model_simp, output_path)
            print("  Simplified successfully.")
        else:
            print("  Simplification failed, keeping original.")
    except ImportError:
        print("  onnxsim not installed, skipping simplification.")


def _export_dynamo(export_module, dummy, output_path, opset_version, output_names):
    """Export via torch.export / dynamo FX graph (PyTorch 2.5+).

    This produces a fundamentally different ONNX graph from TorchScript.
    The FX-based decomposition handles F.scaled_dot_product_attention and
    control flow differently, which may avoid TRT FP16 numerical issues.
    """
    print(f"Exporting to ONNX via Dynamo (opset {opset_version}) -> {output_path} ...")

    # Check if dynamo export is available
    import inspect
    export_sig = inspect.signature(torch.onnx.export)
    has_dynamo_param = "dynamo" in export_sig.parameters

    if has_dynamo_param:
        # PyTorch >= 2.5: unified API with dynamo=True
        print("  Using torch.onnx.export(dynamo=True) ...")
        with torch.no_grad():
            torch.onnx.export(
                export_module,
                (dummy,),
                output_path,
                opset_version=opset_version,
                input_names=["images"],
                output_names=output_names,
                dynamo=True,
            )
    elif hasattr(torch.onnx, "dynamo_export"):
        # PyTorch 2.1-2.4: separate dynamo_export API
        print("  Using torch.onnx.dynamo_export() ...")
        with torch.no_grad():
            onnx_program = torch.onnx.dynamo_export(export_module, dummy)
            onnx_program.save(output_path)
        print("  Note: input/output names may differ from requested names")
        print("  with dynamo_export(). Check the model with onnx.load().")
    else:
        print("ERROR: Dynamo export requires PyTorch >= 2.1")
        print("  Your PyTorch version:", torch.__version__)
        print("  Falling back to TorchScript export.")
        _export_torchscript(export_module, dummy, output_path, opset_version, output_names)
        return

    # Note: onnxsim may not work on dynamo-exported graphs (different structure).
    # Try it but don't fail if it doesn't work.
    try:
        import onnx
        from onnxsim import simplify

        print("Attempting onnxsim.simplify() on dynamo graph ...")
        model_onnx = onnx.load(output_path)
        model_simp, ok = simplify(model_onnx)
        if ok:
            onnx.save(model_simp, output_path)
            print("  Simplified successfully.")
        else:
            print("  Simplification failed (normal for dynamo graphs), keeping original.")
    except ImportError:
        print("  onnxsim not installed, skipping simplification.")
    except Exception as e:
        print(f"  onnxsim failed on dynamo graph (expected): {e}")
        print("  Keeping original dynamo-exported graph.")


def export_onnx(
    checkpoint_path: str,
    output_path: str,
    opset_version: int = 17,
    validate: bool = True,
    use_dynamo: bool = False,
):
    """Export the SAM3 backbone to ONNX with real-valued RoPE."""
    from sam3.model_builder import build_sam3_image_model
    from sam3.trt.rope_onnx import patch_rope_for_export, patch_sdpa_for_export

    print(f"Loading SAM3 model from {checkpoint_path} ...")
    model = build_sam3_image_model(
        checkpoint_path=checkpoint_path,
        device="cuda",
        eval_mode=True,
        load_from_HF=False,
        enable_segmentation=False,  # not needed for backbone export
    )
    backbone = model.backbone

    # Patch RoPE for ONNX compatibility (needed for both TorchScript and dynamo)
    print("Patching RoPE for ONNX export (complex -> real arithmetic) ...")
    patch_rope_for_export(backbone)

    # Patch SDPA -> eager attention for TRT FP16 compatibility
    print("Patching SDPA for ONNX export (SDPA -> eager attention) ...")
    patch_sdpa_for_export(backbone)

    # Wrap for flat output
    export_module = _BackboneForExport(backbone).cuda().eval()

    # Dummy input
    dummy = torch.randn(1, 3, 1008, 1008, device="cuda")

    # Validate patched model produces same outputs as original
    if validate:
        print("Validating patched backbone outputs ...")
        with torch.no_grad():
            patched_out = export_module(dummy)

        # Unpatch, run original, re-patch
        from sam3.trt.rope_onnx import unpatch_rope

        unpatch_rope(backbone)
        with torch.no_grad():
            orig_out = backbone.forward_image(dummy)
        patch_rope_for_export(backbone)

        for i in range(3):
            diff = (patched_out[i] - orig_out["backbone_fpn"][i]).abs().max().item()
            print(f"  FPN level {i}: max abs diff = {diff:.2e}")
            if diff > 1e-4:
                print(f"  WARNING: large difference at level {i}!")

    # Define output names
    output_names = ["fpn_0", "fpn_1", "fpn_2"]

    # Use appropriate opset for dynamo (default 18)
    if use_dynamo and opset_version < 18:
        print(f"  Note: Bumping opset {opset_version} -> 18 (minimum for dynamo export)")
        opset_version = 18

    # Export
    if use_dynamo:
        _export_dynamo(export_module, dummy, output_path, opset_version, output_names)
    else:
        _export_torchscript(export_module, dummy, output_path, opset_version, output_names)

    # Print file size
    size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    print(f"Done. ONNX model saved: {output_path} ({size_mb:.1f} MB)")

    # Validate ONNX model loads correctly
    try:
        import onnx

        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX model validation passed.")
    except ImportError:
        pass
    except Exception as e:
        print(f"ONNX validation warning: {e}")
        print("  (Dynamo-exported models may use ops not in the checker's registry)")

    # Print ONNX graph summary
    try:
        import onnx
        from collections import Counter

        onnx_model = onnx.load(output_path)
        graph = onnx_model.graph
        op_counts = Counter(n.op_type for n in graph.node)
        print(f"\nONNX graph summary: {len(graph.node)} nodes")
        print(f"  Inputs: {len(graph.input)}, Outputs: {len(graph.output)}")
        for inp in graph.input:
            dims = [d.dim_value or d.dim_param for d in inp.type.tensor_type.shape.dim]
            print(f"    {inp.name}: {dims}")
        for out in graph.output:
            dims = [d.dim_value or d.dim_param for d in out.type.tensor_type.shape.dim]
            print(f"    {out.name}: {dims}")
        print(f"  Top ops:")
        for op, count in op_counts.most_common(15):
            print(f"    {op}: {count}")
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(description="Export SAM3 backbone to ONNX")
    parser.add_argument(
        "--checkpoint", required=True, help="Path to SAM3 checkpoint (.pt)"
    )
    parser.add_argument(
        "--output", default="backbone.onnx", help="Output ONNX file path"
    )
    parser.add_argument(
        "--opset", type=int, default=17, help="ONNX opset version"
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip validation of patched vs original outputs",
    )
    parser.add_argument(
        "--dynamo",
        action="store_true",
        help="Use dynamo-based export (PyTorch 2.5+). Produces a different "
             "ONNX graph that may fix TRT FP16 numerical issues.",
    )
    args = parser.parse_args()

    export_onnx(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        opset_version=args.opset,
        validate=not args.no_validate,
        use_dynamo=args.dynamo,
    )


if __name__ == "__main__":
    main()
