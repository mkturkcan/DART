#!/usr/bin/env python3
"""Test if TRT FP16 MatMul uses FP32 accumulation.

Hypothesis: TRT's FP16 MatMul doesn't guarantee FP32 accumulation for
all matrix shapes. With 1024-element dot products, FP16 accumulation
gives massive precision loss.

Tests:
1. Build FP16 engine with opt_level=0 (no fusion) - isolates kernel precision
2. Build FP16 engine with opt_level=3 (normal) - normal comparison
3. Compare against PyTorch FP16 (which guarantees FP32 accumulation)
"""

import sys
import time
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
ONNX_PATH = "backbone.onnx"
DEVICE = "cuda"


def build_fp16(onnx_path, output_path, opt_level=3):
    """Build pure FP16 engine (no mixed precision constraints)."""
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, TRT_LOGGER)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            raise RuntimeError("ONNX parse failed")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
    config.set_flag(trt.BuilderFlag.FP16)
    if hasattr(config, "builder_optimization_level"):
        config.builder_optimization_level = opt_level
    print(f"  Building FP16 engine with opt_level={opt_level}...")

    t0 = time.time()
    engine_bytes = builder.build_serialized_network(network, config)
    print(f"  Build time: {time.time()-t0:.0f}s")

    if engine_bytes is None:
        raise RuntimeError("Engine build failed")
    with open(output_path, "wb") as f:
        f.write(engine_bytes)
    print(f"  Saved: {output_path} ({Path(output_path).stat().st_size/1e6:.0f} MB)")


def test_engine(engine_path, label, model, dummy):
    """Test engine accuracy and speed."""
    from sam3.trt.trt_backbone import TRTBackbone

    backbone = model.backbone
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        pt_out = backbone.forward_image(dummy)
    pt_fpn = pt_out["backbone_fpn"]

    pos_module = backbone.vision_backbone.position_encoding
    trt_bb = TRTBackbone(
        engine_path=engine_path, device=DEVICE, pos_encoding_module=pos_module,
    )

    with torch.inference_mode():
        trt_out = trt_bb.forward_image(dummy)
    trt_fpn = trt_out["backbone_fpn"]

    cos = [
        torch.nn.functional.cosine_similarity(
            pt_fpn[i].float().flatten().unsqueeze(0),
            trt_fpn[i].float().flatten().unsqueeze(0),
        ).item()
        for i in range(len(pt_fpn))
    ]

    # Speed
    with torch.inference_mode():
        for _ in range(10):
            trt_bb.forward_image(dummy)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(50):
            trt_bb.forward_image(dummy)
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) / 50 * 1000

    print(f"  {label:35s} | cos=[{cos[0]:.4f}, {cos[1]:.4f}, {cos[2]:.4f}] | {ms:.1f}ms")
    del trt_bb
    torch.cuda.empty_cache()


def test_matmul_accum():
    """Direct test: does CUDA FP16 MatMul use FP32 accumulation?"""
    print("\n  Direct MatMul accumulation test (1024-dim dot product):")

    # Create test matrices mimicking ViT-H QKV projection
    # A: (5184, 1024) - token features (5184 = 72*72 patches)
    # B: (1024, 3072) - QKV weight matrix
    torch.manual_seed(42)
    A_fp32 = torch.randn(5184, 1024, device=DEVICE)
    B_fp32 = torch.randn(1024, 3072, device=DEVICE)

    # FP32 reference
    C_fp32 = torch.matmul(A_fp32, B_fp32)

    # FP16 with tensor cores (PyTorch always uses FP32 accumulation)
    A_fp16 = A_fp32.half()
    B_fp16 = B_fp32.half()
    C_fp16 = torch.matmul(A_fp16, B_fp16)

    cos = torch.nn.functional.cosine_similarity(
        C_fp32.flatten().unsqueeze(0),
        C_fp16.float().flatten().unsqueeze(0),
    ).item()
    max_diff = (C_fp32 - C_fp16.float()).abs().max().item()
    rel = ((C_fp32 - C_fp16.float()).abs() / C_fp32.abs().clamp(min=1e-6)).mean().item()

    print(f"    PyTorch FP16 MatMul (5184x1024 @ 1024x3072):")
    print(f"      cosine={cos:.6f}, max_diff={max_diff:.4f}, mean_rel_err={rel:.6f}")
    print(f"      Result range: FP32=[{C_fp32.min().item():.1f}, {C_fp32.max().item():.1f}], "
          f"FP16=[{C_fp16.float().min().item():.1f}, {C_fp16.float().max().item():.1f}]")

    # Test with different reduction dimensions
    print("\n    MatMul precision vs reduction dimension:")
    print(f"    {'Dim':>6s} | {'Cosine':>10s} | {'MaxDiff':>10s} | {'MeanRelErr':>12s}")
    print("    " + "-" * 50)
    for K in [64, 128, 256, 512, 1024, 2048, 4096]:
        A = torch.randn(1024, K, device=DEVICE)
        B = torch.randn(K, 1024, device=DEVICE)
        C_ref = torch.matmul(A, B)
        C_f16 = torch.matmul(A.half(), B.half())
        cos = torch.nn.functional.cosine_similarity(
            C_ref.flatten().unsqueeze(0),
            C_f16.float().flatten().unsqueeze(0),
        ).item()
        md = (C_ref - C_f16.float()).abs().max().item()
        re = ((C_ref - C_f16.float()).abs() / C_ref.abs().clamp(min=1e-6)).mean().item()
        print(f"    {K:>6d} | {cos:>10.6f} | {md:>10.4f} | {re:>12.6f}")


def main():
    from sam3.model_builder import build_sam3_image_model

    print("Loading model...")
    model = build_sam3_image_model(
        device=DEVICE, checkpoint_path="sam3.pt", eval_mode=True,
    )
    dummy = torch.randn(1, 3, 1008, 1008, device=DEVICE)

    # Part 1: Direct MatMul accumulation test
    print("\n" + "=" * 80)
    print("PART 1: MatMul FP16 accumulation precision")
    print("=" * 80)
    test_matmul_accum()

    # Part 2: TRT engines with different opt levels
    print("\n" + "=" * 80)
    print("PART 2: TRT FP16 engines")
    print("=" * 80)

    build_fp16(ONNX_PATH, "backbone_fp16_opt0.engine", opt_level=0)
    build_fp16(ONNX_PATH, "backbone_fp16_opt3.engine", opt_level=3)

    test_engine("backbone_fp16_opt0.engine", "FP16 opt_level=0 (no fusion)", model, dummy)
    test_engine("backbone_fp16_opt3.engine", "FP16 opt_level=3 (normal)", model, dummy)

    print("\nDone!")


if __name__ == "__main__":
    main()
