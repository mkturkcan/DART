#!/usr/bin/env python3
"""Test whether TRT's native SDPA handling gives better FP16 accuracy.

Exports backbone in two ways:
  1. With _fp32_sdpa patch (explicit matmul — our current approach)
  2. Without SDPA patch (native F.scaled_dot_product_attention)

Then builds FP16 engines for each and compares accuracy vs PyTorch reference.

The hypothesis: TRT might recognize the native SDPA pattern and map it to
cuDNN's fused MHA kernel, which uses FP32 accumulation internally.

Usage:
    python scripts/test_sdpa_trt.py --checkpoint sam3.pt --image x.jpg
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def cosine_similarity(a, b):
    a_flat = a.float().flatten()
    b_flat = b.float().flatten()
    return torch.nn.functional.cosine_similarity(
        a_flat.unsqueeze(0), b_flat.unsqueeze(0)
    ).item()


class _BackboneForExport(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, images: torch.Tensor):
        out = self.backbone.forward_image(images)
        fpn = out["backbone_fpn"]
        return fpn[0], fpn[1], fpn[2]


def export_and_test(backbone, dummy, img_tensor, ref_dict, label, onnx_path,
                    engine_path, patch_sdpa=True):
    """Export backbone → ONNX → TRT FP16 engine, then test accuracy."""
    from sam3.trt.rope_onnx import patch_rope_for_export, unpatch_rope

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    # Patch RoPE (always needed for ONNX)
    patch_rope_for_export(backbone)

    if patch_sdpa:
        from sam3.trt.rope_onnx import patch_sdpa_for_export
        patch_sdpa_for_export(backbone)

    export_module = _BackboneForExport(backbone).cuda().eval()

    # Export to ONNX
    print(f"  Exporting to {onnx_path} ...")
    with torch.no_grad():
        torch.onnx.export(
            export_module,
            (dummy,),
            onnx_path,
            opset_version=17,
            input_names=["images"],
            output_names=["fpn_0", "fpn_1", "fpn_2"],
            dynamic_axes=None,
        )

    # Print ONNX op counts
    try:
        import onnx
        from collections import Counter
        model = onnx.load(onnx_path)
        ops = Counter(n.op_type for n in model.graph.node)
        print(f"  ONNX: {len(model.graph.node)} nodes")
        for op, count in ops.most_common(10):
            print(f"    {op}: {count}")
        del model
    except Exception as e:
        print(f"  ONNX analysis failed: {e}")

    # Unpatch
    unpatch_rope(backbone)
    if patch_sdpa:
        from sam3.trt.rope_onnx import unpatch_sdpa
        unpatch_sdpa(backbone)

    # Build TRT FP16 engine
    print(f"  Building TRT FP16 engine: {engine_path} ...")
    from sam3.trt.build_engine import build_engine
    build_engine(
        onnx_path=onnx_path,
        output_path=engine_path,
        fp16=True,
        mixed_precision="attention",  # Norm+Softmax+Attn MatMul FP32
        workspace_gb=4.0,
    )

    # Test accuracy
    print(f"  Testing accuracy...")
    from sam3.trt.trt_backbone import TRTBackbone
    trt_bb = TRTBackbone(engine_path, device="cuda")

    with torch.inference_mode():
        for _ in range(5):
            trt_bb.forward_image(img_tensor)
        torch.cuda.synchronize()

        trt_out = trt_bb.forward_image(img_tensor)
        torch.cuda.synchronize()

    trt_fpn = trt_out["backbone_fpn"]
    print(f"  Cosine similarity vs PyTorch FP32:")
    for i in range(len(trt_fpn)):
        k = f"fpn_{i}"
        cos = cosine_similarity(ref_dict[k], trt_fpn[i])
        status = "OK" if cos > 0.999 else "WARN" if cos > 0.99 else "BAD"
        print(f"    {k}: cos={cos:.6f} [{status}]")

    # Also test with pure FP16 (no mixed precision)
    print(f"\n  Building PURE FP16 engine (no mixed precision)...")
    engine_path_pure = engine_path.replace(".engine", "_pure.engine")
    build_engine(
        onnx_path=onnx_path,
        output_path=engine_path_pure,
        fp16=True,
        mixed_precision=None,
        workspace_gb=4.0,
    )
    trt_bb2 = TRTBackbone(engine_path_pure, device="cuda")
    with torch.inference_mode():
        for _ in range(5):
            trt_bb2.forward_image(img_tensor)
        torch.cuda.synchronize()
        trt_out2 = trt_bb2.forward_image(img_tensor)
        torch.cuda.synchronize()

    trt_fpn2 = trt_out2["backbone_fpn"]
    print(f"  Pure FP16 cosine similarity:")
    for i in range(len(trt_fpn2)):
        k = f"fpn_{i}"
        cos = cosine_similarity(ref_dict[k], trt_fpn2[i])
        status = "OK" if cos > 0.999 else "WARN" if cos > 0.99 else "BAD"
        print(f"    {k}: cos={cos:.6f} [{status}]")

    # Speed test on pure FP16
    times = []
    with torch.inference_mode():
        for _ in range(30):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            trt_bb2.forward_image(img_tensor)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)
    print(f"  Pure FP16 speed: {np.mean(times):.1f}ms avg, {np.min(times):.1f}ms min")

    del trt_bb, trt_bb2
    torch.cuda.empty_cache()

    # Cleanup
    for p in [onnx_path, engine_path, engine_path_pure]:
        if os.path.exists(p):
            os.remove(p)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="sam3.pt")
    parser.add_argument("--image", default="x.jpg")
    args = parser.parse_args()

    device = "cuda"
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")

    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_multiclass_fast import Sam3MultiClassPredictorFast
    from torchvision.transforms import v2

    model = build_sam3_image_model(
        device=device, checkpoint_path=args.checkpoint, eval_mode=True,
    )
    predictor = Sam3MultiClassPredictorFast(
        model, device=device, resolution=1008,
        use_fp16=False, detection_only=True,
    )

    image = Image.open(args.image).convert("RGB")
    with torch.inference_mode():
        resized = image.resize((1008, 1008), Image.BILINEAR)
        img_tensor = v2.functional.to_image(resized).to(device)
        img_tensor = predictor.transform(img_tensor).unsqueeze(0)
        ref_out = model.backbone.forward_image(img_tensor)

    ref_fpn = ref_out["backbone_fpn"]
    ref_dict = {f"fpn_{i}": ref_fpn[i] for i in range(len(ref_fpn))}
    dummy = torch.randn(1, 3, 1008, 1008, device=device)

    backbone = model.backbone
    del predictor
    torch.cuda.empty_cache()

    # Test 1: WITH SDPA patch (our current approach)
    export_and_test(
        backbone, dummy, img_tensor, ref_dict,
        "WITH _fp32_sdpa patch (explicit FP32 matmul)",
        "backbone_sdpa_patched.onnx",
        "backbone_sdpa_patched.engine",
        patch_sdpa=True,
    )

    # Test 2: WITHOUT SDPA patch (native SDPA)
    export_and_test(
        backbone, dummy, img_tensor, ref_dict,
        "WITHOUT SDPA patch (native F.scaled_dot_product_attention)",
        "backbone_native_sdpa.onnx",
        "backbone_native_sdpa.engine",
        patch_sdpa=False,
    )


if __name__ == "__main__":
    main()
