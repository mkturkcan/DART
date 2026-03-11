#!/usr/bin/env python3
"""Test Torch-TensorRT compilation of SAM3 backbone.

Instead of ONNX -> TRT (which creates complex graphs that break FP16),
use torch_tensorrt.compile with dynamo IR. This preserves PyTorch's SDPA op
and lets TRT fuse attention into its MHA kernel with FP32 accumulation.

Key: Must patch RoPE to real-valued arithmetic first (complex64 unsupported).
"""

import sys
import time
import torch
import torch.nn as nn
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DEVICE = "cuda"
DTYPE = torch.float16


class BackboneWrapper(nn.Module):
    """Wraps backbone to return tuple instead of dict."""
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, images):
        out = self.backbone.forward_image(images)
        fpn = out["backbone_fpn"]
        return fpn[0], fpn[1], fpn[2]


def get_pytorch_reference(model, dummy):
    """Get PyTorch FP16 reference output."""
    backbone = model.backbone
    with torch.inference_mode(), torch.autocast("cuda", dtype=DTYPE):
        out = backbone.forward_image(dummy)
    return out["backbone_fpn"]


def cosine_sim(a, b):
    return torch.nn.functional.cosine_similarity(
        a.float().flatten().unsqueeze(0),
        b.float().flatten().unsqueeze(0),
    ).item()


def benchmark(fn, dummy, warmup=10, iters=50):
    """Benchmark a function, return ms per call."""
    with torch.inference_mode():
        for _ in range(warmup):
            fn(dummy)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn(dummy)
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / iters * 1000


def test_torch_trt_dynamo(model, dummy, pt_ref, precision_label, enabled_precisions):
    """Test torch_tensorrt.compile with dynamo IR."""
    import torch_tensorrt
    from sam3.trt.rope_onnx import patch_rope_for_export, unpatch_rope

    print(f"\n{'=' * 80}")
    print(f"torch_tensorrt.compile(ir='dynamo') {precision_label}")
    print("=" * 80)

    backbone = model.backbone

    # Patch RoPE to remove complex64 tensors
    print("  Patching RoPE for real-valued arithmetic...")
    patch_rope_for_export(backbone)

    wrapper = BackboneWrapper(backbone).eval().cuda()

    print(f"  Compiling with Torch-TensorRT ({precision_label})...")
    try:
        compiled = torch_tensorrt.compile(
            wrapper,
            ir="dynamo",
            inputs=[torch_tensorrt.Input(
                shape=(1, 3, 1008, 1008),
                dtype=torch.float32,
            )],
            enabled_precisions=enabled_precisions,
            workspace_size=4 << 30,
            truncate_long_and_double=True,
            min_block_size=1,
        )
    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback
        traceback.print_exc()
        unpatch_rope(backbone)
        return None, None

    # Test accuracy
    print("  Testing accuracy...")
    with torch.inference_mode():
        out = compiled(dummy)

    cos = [cosine_sim(pt_ref[i], out[i]) for i in range(3)]

    # Benchmark
    print("  Benchmarking...")
    ms = benchmark(lambda x: compiled(x), dummy)

    status = "OK" if cos[-1] > 0.99 else "BROKEN" if cos[-1] < 0.5 else "DEGRADED"
    print(f"  cos=[{cos[0]:.4f}, {cos[1]:.4f}, {cos[2]:.4f}] | {ms:.1f}ms | {status}")

    unpatch_rope(backbone)
    return cos, ms


def test_torch_compile_trt_backend(model, dummy, pt_ref, precision_label, enabled_precisions):
    """Test torch.compile with tensorrt backend."""
    import torch_tensorrt  # noqa: registers backend
    from sam3.trt.rope_onnx import patch_rope_for_export, unpatch_rope

    print(f"\n{'=' * 80}")
    print(f"torch.compile(backend='torch_tensorrt') {precision_label}")
    print("=" * 80)

    backbone = model.backbone
    patch_rope_for_export(backbone)

    print(f"  Compiling with torch.compile TRT backend ({precision_label})...")
    try:
        compiled = torch.compile(
            backbone.forward_image,
            backend="torch_tensorrt",
            options={
                "enabled_precisions": enabled_precisions,
                "workspace_size": 4 << 30,
                "truncate_long_and_double": True,
                "min_block_size": 1,
            },
        )

        # Warmup (includes JIT compilation)
        with torch.inference_mode(), torch.autocast("cuda", dtype=DTYPE):
            out = compiled(dummy)
        fpn = out["backbone_fpn"]
    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback
        traceback.print_exc()
        unpatch_rope(backbone)
        return None, None

    cos = [cosine_sim(pt_ref[i], fpn[i]) for i in range(3)]

    # Benchmark
    print("  Benchmarking...")
    def run_fn(x):
        with torch.autocast("cuda", dtype=DTYPE):
            return compiled(x)
    ms = benchmark(run_fn, dummy)

    status = "OK" if cos[-1] > 0.99 else "BROKEN" if cos[-1] < 0.5 else "DEGRADED"
    print(f"  cos=[{cos[0]:.4f}, {cos[1]:.4f}, {cos[2]:.4f}] | {ms:.1f}ms | {status}")

    unpatch_rope(backbone)
    return cos, ms


def test_export_trt(model, dummy, pt_ref, precision_label, enabled_precisions):
    """Test torch.export -> torch_tensorrt.dynamo.compile."""
    import torch_tensorrt
    from sam3.trt.rope_onnx import patch_rope_for_export, unpatch_rope

    print(f"\n{'=' * 80}")
    print(f"torch.export + torch_tensorrt.dynamo.compile {precision_label}")
    print("=" * 80)

    backbone = model.backbone
    patch_rope_for_export(backbone)

    wrapper = BackboneWrapper(backbone).eval().cuda()

    print("  Exporting with torch.export...")
    try:
        exported = torch.export.export(wrapper, (dummy,))
        print(f"  Export successful, graph has {len(exported.graph.nodes)} nodes")

        print(f"  Compiling exported program with TRT ({precision_label})...")
        compiled = torch_tensorrt.dynamo.compile(
            exported,
            inputs=[dummy],
            enabled_precisions=enabled_precisions,
            workspace_size=4 << 30,
            truncate_long_and_double=True,
            min_block_size=1,
        )
    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback
        traceback.print_exc()
        unpatch_rope(backbone)
        return None, None

    # Test accuracy
    print("  Testing accuracy...")
    with torch.inference_mode():
        out = compiled(dummy)

    cos = [cosine_sim(pt_ref[i], out[i]) for i in range(3)]

    # Benchmark
    print("  Benchmarking...")
    ms = benchmark(lambda x: compiled(x), dummy)

    status = "OK" if cos[-1] > 0.99 else "BROKEN" if cos[-1] < 0.5 else "DEGRADED"
    print(f"  cos=[{cos[0]:.4f}, {cos[1]:.4f}, {cos[2]:.4f}] | {ms:.1f}ms | {status}")

    unpatch_rope(backbone)
    return cos, ms


def main():
    from sam3.model_builder import build_sam3_image_model

    print("Loading SAM3 model...")
    model = build_sam3_image_model(
        device=DEVICE, checkpoint_path="sam3.pt", eval_mode=True,
    )
    dummy = torch.randn(1, 3, 1008, 1008, device=DEVICE)

    print("Getting PyTorch FP16 reference...")
    pt_ref = get_pytorch_reference(model, dummy)

    # Benchmark PyTorch eager FP16
    backbone = model.backbone
    def pt_eager(x):
        with torch.autocast("cuda", dtype=DTYPE):
            return backbone.forward_image(x)
    ms_eager = benchmark(pt_eager, dummy)
    print(f"  PyTorch eager FP16: {ms_eager:.1f}ms")

    results = {}

    # ================================================================
    # Approach 1: torch_tensorrt.compile FP16 only
    # ================================================================
    try:
        cos, ms = test_torch_trt_dynamo(
            model, dummy, pt_ref, "FP16",
            {torch.float16},
        )
        if cos: results["trt_dynamo_fp16"] = (cos, ms)
    except Exception as e:
        print(f"  FAILED: {e}")

    torch._dynamo.reset()
    torch.cuda.empty_cache()

    # ================================================================
    # Approach 2: torch_tensorrt.compile FP16+FP32 (TRT auto-selects)
    # ================================================================
    try:
        cos, ms = test_torch_trt_dynamo(
            model, dummy, pt_ref, "FP16+FP32",
            {torch.float16, torch.float32},
        )
        if cos: results["trt_dynamo_fp16_fp32"] = (cos, ms)
    except Exception as e:
        print(f"  FAILED: {e}")

    torch._dynamo.reset()
    torch.cuda.empty_cache()

    # ================================================================
    # Approach 3: torch.compile TRT backend FP16
    # ================================================================
    try:
        cos, ms = test_torch_compile_trt_backend(
            model, dummy, pt_ref, "FP16",
            {torch.float16},
        )
        if cos: results["compile_trt_fp16"] = (cos, ms)
    except Exception as e:
        print(f"  FAILED: {e}")

    torch._dynamo.reset()
    torch.cuda.empty_cache()

    # ================================================================
    # Approach 4: torch.export + TRT FP16
    # ================================================================
    try:
        cos, ms = test_export_trt(
            model, dummy, pt_ref, "FP16",
            {torch.float16},
        )
        if cos: results["export_trt_fp16"] = (cos, ms)
    except Exception as e:
        print(f"  FAILED: {e}")

    torch._dynamo.reset()
    torch.cuda.empty_cache()

    # ================================================================
    # Approach 5: torch.export + TRT FP16+FP32
    # ================================================================
    try:
        cos, ms = test_export_trt(
            model, dummy, pt_ref, "FP16+FP32",
            {torch.float16, torch.float32},
        )
        if cos: results["export_trt_fp16_fp32"] = (cos, ms)
    except Exception as e:
        print(f"  FAILED: {e}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"  PyTorch eager FP16: {ms_eager:.1f}ms")
    if results:
        print(f"  {'Approach':>25s} | {'FPN[-1] cos':>12s} | {'Speed':>7s} | Status")
        print("  " + "-" * 60)
        for name, (cos, ms) in sorted(results.items(), key=lambda x: x[1][1]):
            status = "OK" if cos[-1] > 0.99 else "BROKEN" if cos[-1] < 0.5 else "DEGRADED"
            print(f"  {name:>25s} | {cos[-1]:>12.4f} | {ms:>5.1f}ms | {status}")
    else:
        print("  No successful approaches!")

    print("\nDone!")


if __name__ == "__main__":
    main()
