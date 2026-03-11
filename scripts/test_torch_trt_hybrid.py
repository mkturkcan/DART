#!/usr/bin/env python3
"""Test Torch-TensorRT hybrid mode: TRT FP16 for conv/MLP, PyTorch for attention.

Key findings so far:
- Pure TRT FP16 (all paths): 54ms, cos 0.09 (BROKEN)
- Pure TRT FP16+FP32: 54ms, cos 0.09 (TRT still uses FP16 for attention)
- ONNX TRT mixed precision: 130ms, cos 0.9998 (too slow)

Strategy: Use torch_tensorrt's `torch_executed_ops` to keep attention SDPA in
PyTorch (which guarantees FP32 accumulation) while letting TRT handle the rest
(conv, MLP, LayerNorm) in FP16. This should give:
- Conv/MLP/LN: TRT FP16 (fast)
- Attention SDPA: PyTorch FP32-accumulated (correct)
- Target: <66ms, cos >0.99
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


def analyze_graph_ops(model, dummy):
    """Analyze what aten ops the model uses via torch.export."""
    from sam3.trt.rope_onnx import patch_rope_for_export, unpatch_rope

    backbone = model.backbone
    patch_rope_for_export(backbone)
    wrapper = BackboneWrapper(backbone).eval().cuda()

    print("Analyzing graph ops via torch.export...")
    exported = torch.export.export(wrapper, (dummy,))

    op_counts = {}
    for node in exported.graph.nodes:
        if node.op == "call_function":
            op_name = str(node.target)
            op_counts[op_name] = op_counts.get(op_name, 0) + 1

    print(f"  Total ops: {sum(op_counts.values())}")
    print(f"  Unique ops: {len(op_counts)}")
    print("\n  All ops:")
    for op, count in sorted(op_counts.items(), key=lambda x: -x[1]):
        print(f"    {count:4d}x {op}")

    # Find attention-related ops
    attention_ops = [op for op in op_counts if "attention" in op.lower() or "sdpa" in op.lower() or "scaled_dot" in op.lower()]
    print(f"\n  Attention ops: {attention_ops}")

    unpatch_rope(backbone)
    return op_counts


def test_hybrid(model, dummy, pt_ref, label, torch_executed_ops):
    """Test torch_tensorrt with specific ops kept in PyTorch."""
    import torch_tensorrt
    from sam3.trt.rope_onnx import patch_rope_for_export, unpatch_rope

    print(f"\n{'=' * 80}")
    print(f"HYBRID: {label}")
    print(f"  torch_executed_ops: {torch_executed_ops}")
    print("=" * 80)

    backbone = model.backbone
    patch_rope_for_export(backbone)

    wrapper = BackboneWrapper(backbone).eval().cuda()

    print("  Compiling...")
    try:
        compiled = torch_tensorrt.compile(
            wrapper,
            ir="dynamo",
            inputs=[torch_tensorrt.Input(
                shape=(1, 3, 1008, 1008),
                dtype=torch.float32,
            )],
            enabled_precisions={torch.float16},
            workspace_size=4 << 30,
            truncate_long_and_double=True,
            min_block_size=1,
            torch_executed_ops=torch_executed_ops,
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


def test_compile_backend_hybrid(model, dummy, pt_ref, label, torch_executed_ops):
    """Test torch.compile with TRT backend, keeping specific ops in PyTorch."""
    import torch_tensorrt  # noqa

    from sam3.trt.rope_onnx import patch_rope_for_export, unpatch_rope

    print(f"\n{'=' * 80}")
    print(f"torch.compile HYBRID: {label}")
    print(f"  torch_executed_ops: {torch_executed_ops}")
    print("=" * 80)

    backbone = model.backbone
    patch_rope_for_export(backbone)

    print("  Compiling...")
    try:
        compiled = torch.compile(
            backbone.forward_image,
            backend="torch_tensorrt",
            options={
                "enabled_precisions": {torch.float16},
                "workspace_size": 4 << 30,
                "truncate_long_and_double": True,
                "min_block_size": 1,
                "torch_executed_ops": torch_executed_ops,
            },
        )

        # Warmup
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


def main():
    from sam3.model_builder import build_sam3_image_model

    print("Loading SAM3 model...")
    model = build_sam3_image_model(
        device=DEVICE, checkpoint_path="sam3.pt", eval_mode=True,
    )
    dummy = torch.randn(1, 3, 1008, 1008, device=DEVICE)

    print("Getting PyTorch FP16 reference...")
    pt_ref = get_pytorch_reference(model, dummy)

    # Benchmark PyTorch eager
    backbone = model.backbone
    def pt_eager(x):
        with torch.autocast("cuda", dtype=DTYPE):
            return backbone.forward_image(x)
    ms_eager = benchmark(pt_eager, dummy)
    print(f"  PyTorch eager FP16: {ms_eager:.1f}ms")

    # First, analyze what ops the graph uses
    op_counts = analyze_graph_ops(model, dummy)

    results = {}

    # ================================================================
    # Strategy 1: Keep only SDPA in PyTorch
    # ================================================================
    sdpa_ops = {
        "torch.ops.aten._scaled_dot_product_flash_attention.default",
        "torch.ops.aten._scaled_dot_product_efficient_attention.default",
        "torch.ops.aten.scaled_dot_product_attention.default",
    }
    # Filter to only ops that actually appear in the graph
    actual_sdpa = {op for op in sdpa_ops if op in [str(k) for k in op_counts.keys()]}
    if not actual_sdpa:
        # Try broader matching
        actual_sdpa = {str(k) for k in op_counts.keys() if "attention" in str(k).lower() or "sdpa" in str(k).lower()}
    print(f"\n  Detected SDPA ops: {actual_sdpa}")

    try:
        cos, ms = test_hybrid(model, dummy, pt_ref, "SDPA in PyTorch", actual_sdpa)
        if cos: results["hybrid_sdpa"] = (cos, ms)
    except Exception as e:
        print(f"  FAILED: {e}")
    torch._dynamo.reset()
    torch.cuda.empty_cache()

    # ================================================================
    # Strategy 2: Keep SDPA + matmul in PyTorch (attention matmuls)
    # ================================================================
    matmul_ops = {str(k) for k in op_counts.keys() if "matmul" in str(k).lower() or "mm" in str(k).lower() or "bmm" in str(k).lower()}
    sdpa_matmul = actual_sdpa | matmul_ops
    print(f"\n  MatMul ops: {matmul_ops}")

    try:
        cos, ms = test_hybrid(model, dummy, pt_ref, "SDPA+MatMul in PyTorch", sdpa_matmul)
        if cos: results["hybrid_sdpa_matmul"] = (cos, ms)
    except Exception as e:
        print(f"  FAILED: {e}")
    torch._dynamo.reset()
    torch.cuda.empty_cache()

    # ================================================================
    # Strategy 3: Keep SDPA + softmax in PyTorch
    # ================================================================
    softmax_ops = {str(k) for k in op_counts.keys() if "softmax" in str(k).lower()}
    sdpa_softmax = actual_sdpa | softmax_ops
    print(f"\n  Softmax ops: {softmax_ops}")

    try:
        cos, ms = test_hybrid(model, dummy, pt_ref, "SDPA+Softmax in PyTorch", sdpa_softmax)
        if cos: results["hybrid_sdpa_softmax"] = (cos, ms)
    except Exception as e:
        print(f"  FAILED: {e}")
    torch._dynamo.reset()
    torch.cuda.empty_cache()

    # ================================================================
    # Strategy 4: torch.compile backend with SDPA in PyTorch
    # ================================================================
    try:
        cos, ms = test_compile_backend_hybrid(
            model, dummy, pt_ref, "SDPA in PyTorch",
            actual_sdpa,
        )
        if cos: results["compile_hybrid_sdpa"] = (cos, ms)
    except Exception as e:
        print(f"  FAILED: {e}")
    torch._dynamo.reset()
    torch.cuda.empty_cache()

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"  PyTorch eager FP16: {ms_eager:.1f}ms")
    if results:
        print(f"  {'Approach':>30s} | {'FPN[-1] cos':>12s} | {'Speed':>7s} | Status")
        print("  " + "-" * 65)
        for name, (cos, ms) in sorted(results.items(), key=lambda x: x[1][1]):
            status = "OK" if cos[-1] > 0.99 else "BROKEN" if cos[-1] < 0.5 else "DEGRADED"
            print(f"  {name:>30s} | {cos[-1]:>12.4f} | {ms:>5.1f}ms | {status}")
    else:
        print("  No successful approaches!")

    print("\nDone!")


if __name__ == "__main__":
    main()
