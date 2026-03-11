"""
ONNX export for SAM3 student backbone + FPN adapter.

Exports only the vision backbone (timm model + FPN adapter), NOT the
encoder/decoder/text encoder. The enc-dec uses its own TRT engine.

Output: 3 FPN feature maps matching the teacher backbone interface:
    fpn_0: (1, 256, 288, 288)
    fpn_1: (1, 256, 144, 144)
    fpn_2: (1, 256, 72, 72)

Usage:
    python -m sam3.distillation.export_onnx \
        --checkpoint sam3.pt \
        --adapter-checkpoint adapter_final.pt \
        --backbone repvit_m2_3 \
        --output student_backbone.onnx
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn


class _StudentBackboneForExport(nn.Module):
    """Wrapper returning flat FPN tuple for ONNX export."""

    def __init__(self, student_backbone):
        super().__init__()
        self.backbone = student_backbone

    def forward(self, images: torch.Tensor):
        out = self.backbone.forward_image(images)
        fpn = out["backbone_fpn"]
        return fpn[0], fpn[1], fpn[2]


def export_onnx(
    backbone_config: str,
    adapter_checkpoint: str,
    output_path: str,
    checkpoint: str = None,
    imgsz: int = 1008,
    opset_version: int = 17,
    device: str = "cuda",
):
    from sam3.distillation.student_backbone import build_student_backbone

    print(f"Building student backbone: {backbone_config}")
    student_bb = build_student_backbone(
        config_name=backbone_config,
        pretrained=True,
        freeze_backbone=True,
    )

    print(f"Loading adapter weights from {adapter_checkpoint}")
    ckpt = torch.load(adapter_checkpoint, map_location="cpu")
    student_bb.load_state_dict(ckpt["student_state_dict"], strict=False)

    export_module = _StudentBackboneForExport(student_bb).to(device).eval()

    dummy = torch.randn(1, 3, imgsz, imgsz, device=device)

    # Validate forward works
    print("Validating forward pass ...")
    with torch.no_grad():
        fpn0, fpn1, fpn2 = export_module(dummy)
    print(f"  fpn_0: {list(fpn0.shape)}")
    print(f"  fpn_1: {list(fpn1.shape)}")
    print(f"  fpn_2: {list(fpn2.shape)}")

    output_names = ["fpn_0", "fpn_1", "fpn_2"]

    print(f"Exporting to ONNX (opset {opset_version}) -> {output_path}")
    with torch.no_grad():
        torch.onnx.export(
            export_module,
            (dummy,),
            output_path,
            opset_version=opset_version,
            input_names=["images"],
            output_names=output_names,
            dynamic_axes=None,
        )

    # Simplify
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
        print("  onnxsim not installed, skipping.")

    size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    print(f"Done. {output_path} ({size_mb:.1f} MB)")

    # Validate
    try:
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX validation passed.")
    except ImportError:
        pass
    except Exception as e:
        print(f"ONNX validation warning: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export student backbone to ONNX")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Teacher checkpoint (unused, for compat)")
    parser.add_argument("--adapter-checkpoint", type=str, required=True)
    parser.add_argument("--backbone", type=str, default="efficientvit_l1")
    parser.add_argument("--output", type=str, default="student_backbone.onnx")
    parser.add_argument("--imgsz", type=int, default=1008)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    export_onnx(
        backbone_config=args.backbone,
        adapter_checkpoint=args.adapter_checkpoint,
        output_path=args.output,
        checkpoint=args.checkpoint,
        imgsz=args.imgsz,
        opset_version=args.opset,
        device=args.device,
    )
