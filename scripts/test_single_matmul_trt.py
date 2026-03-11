#!/usr/bin/env python3
"""Minimal test: single MatMul layer in TRT FP16 vs PyTorch FP16.

Creates tiny ONNX models with just one or two operations, builds TRT FP16
engines, and compares against PyTorch to isolate where precision is lost.
"""

import sys
import time
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import onnx
    from onnx import TensorProto, helper
except ImportError:
    print("pip install onnx")
    sys.exit(1)

import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
DEVICE = "cuda"


def make_matmul_onnx(M, K, N, output_path, weight_data=None):
    """Create ONNX with: Y = MatMul(X, W) where X:(M,K), W:(K,N), Y:(M,N)"""
    if weight_data is None:
        weight_data = np.random.randn(K, N).astype(np.float32)

    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [M, K])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [M, N])
    W = helper.make_tensor("W", TensorProto.FLOAT, [K, N], weight_data.flatten())

    matmul = helper.make_node("MatMul", ["X", "W"], ["Y"])
    graph = helper.make_graph([matmul], "matmul_test", [X], [Y], [W])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    onnx.save(model, output_path)
    return weight_data


def make_layernorm_matmul_onnx(N, C, out_C, output_path, weight_data=None, ln_weight=None, ln_bias=None):
    """Create ONNX with: Y = MatMul(LayerNorm(X), W)
    X:(1, N, C), W:(C, out_C), Y:(1, N, out_C)"""
    if weight_data is None:
        weight_data = np.random.randn(C, out_C).astype(np.float32)
    if ln_weight is None:
        ln_weight = np.ones(C, dtype=np.float32)
    if ln_bias is None:
        ln_bias = np.zeros(C, dtype=np.float32)

    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, N, C])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, N, out_C])

    W = helper.make_tensor("W", TensorProto.FLOAT, [C, out_C], weight_data.flatten())
    LN_W = helper.make_tensor("ln_weight", TensorProto.FLOAT, [C], ln_weight.flatten())
    LN_B = helper.make_tensor("ln_bias", TensorProto.FLOAT, [C], ln_bias.flatten())

    ln = helper.make_node("LayerNormalization", ["X", "ln_weight", "ln_bias"], ["ln_out"],
                          axis=-1, epsilon=1e-6)
    matmul = helper.make_node("MatMul", ["ln_out", "W"], ["Y"])
    graph = helper.make_graph([ln, matmul], "ln_matmul_test", [X], [Y], [W, LN_W, LN_B])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    onnx.save(model, output_path)
    return weight_data


def build_and_run_trt(onnx_path, input_tensor, fp16=True):
    """Build TRT engine and run inference."""
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, TRT_LOGGER)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"  Parse error: {parser.get_error(i)}")
            raise RuntimeError("ONNX parse failed")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    engine_bytes = builder.build_serialized_network(network, config)
    if engine_bytes is None:
        raise RuntimeError("Engine build failed")

    runtime = trt.Runtime(TRT_LOGGER)
    engine = runtime.deserialize_cuda_engine(engine_bytes)

    # Create execution context
    context = engine.create_execution_context()

    # Allocate device buffers
    input_name = engine.get_tensor_name(0)
    output_name = engine.get_tensor_name(1)

    d_input = input_tensor.float().contiguous().cuda()
    output_shape = context.get_tensor_shape(output_name)
    d_output = torch.empty(list(output_shape), dtype=torch.float32, device="cuda")

    context.set_tensor_address(input_name, d_input.data_ptr())
    context.set_tensor_address(output_name, d_output.data_ptr())

    stream = torch.cuda.current_stream().cuda_stream
    context.execute_async_v3(stream)
    torch.cuda.synchronize()

    return d_output


def test_single_matmul():
    """Test a single MatMul in TRT FP16 vs PyTorch FP16."""
    print("\n  Single MatMul tests (Y = X @ W):")
    print(f"  {'Shape':>25s} | {'TRT FP16 cos':>12s} | {'TRT FP32 cos':>12s} | {'PT FP16 cos':>12s} | {'TRT FP16 max_diff':>18s}")
    print("  " + "-" * 95)

    for M, K, N, label in [
        (576, 64, 576, "attn_qk (windowed)"),      # Windowed attention Q@K^T
        (5184, 64, 5184, "attn_qk (global)"),       # Global attention Q@K^T
        (5184, 1024, 3072, "qkv_projection"),        # QKV projection
        (5184, 1024, 1024, "out_projection"),        # Output projection
        (5184, 1024, 4736, "mlp_fc1"),               # MLP first layer
        (5184, 4736, 1024, "mlp_fc2"),               # MLP second layer
    ]:
        onnx_path = f"test_matmul_{label.replace(' ', '_').replace('(', '').replace(')', '')}.onnx"
        W = make_matmul_onnx(M, K, N, onnx_path)
        W_torch = torch.from_numpy(W).to(DEVICE)

        # Input mimicking real activations (std ~1.3 like the profiling showed)
        X = torch.randn(M, K, device=DEVICE) * 1.3

        # PyTorch FP32 reference
        Y_fp32 = torch.matmul(X, W_torch)

        # PyTorch FP16
        Y_pt_fp16 = torch.matmul(X.half(), W_torch.half())
        pt_cos = torch.nn.functional.cosine_similarity(
            Y_fp32.flatten().unsqueeze(0),
            Y_pt_fp16.float().flatten().unsqueeze(0),
        ).item()

        # TRT FP16
        Y_trt_fp16 = build_and_run_trt(onnx_path, X, fp16=True)
        trt16_cos = torch.nn.functional.cosine_similarity(
            Y_fp32.flatten().unsqueeze(0),
            Y_trt_fp16.flatten().unsqueeze(0),
        ).item()
        trt16_diff = (Y_fp32 - Y_trt_fp16).abs().max().item()

        # TRT FP32
        Y_trt_fp32 = build_and_run_trt(onnx_path, X, fp16=False)
        trt32_cos = torch.nn.functional.cosine_similarity(
            Y_fp32.flatten().unsqueeze(0),
            Y_trt_fp32.flatten().unsqueeze(0),
        ).item()

        print(f"  {f'{M}x{K} @ {K}x{N} ({label})':<25s} | {trt16_cos:>12.6f} | {trt32_cos:>12.6f} | {pt_cos:>12.6f} | {trt16_diff:>18.4f}")

        Path(onnx_path).unlink(missing_ok=True)


def test_layernorm_plus_matmul():
    """Test LayerNorm -> MatMul chain in TRT FP16."""
    print("\n  LayerNorm + MatMul chain tests:")
    print(f"  {'Shape':>25s} | {'TRT FP16 cos':>12s} | {'TRT FP32 cos':>12s} | {'PT FP16 cos':>12s}")
    print("  " + "-" * 75)

    for N, C, out_C, label in [
        (576, 1024, 3072, "windowed_qkv"),
        (5184, 1024, 3072, "global_qkv"),
        (5184, 1024, 1024, "out_proj"),
    ]:
        onnx_path = f"test_ln_mm_{label}.onnx"
        W = make_layernorm_matmul_onnx(N, C, out_C, onnx_path)
        W_torch = torch.from_numpy(W).to(DEVICE)

        X = torch.randn(1, N, C, device=DEVICE) * 5.0  # Typical activation magnitude

        # PyTorch FP32 reference
        ln = torch.nn.LayerNorm(C, eps=1e-6).to(DEVICE)
        Y_fp32 = torch.matmul(ln(X), W_torch)

        # PyTorch FP16
        Y_pt = torch.matmul(ln.half()(X.half()), W_torch.half())
        pt_cos = torch.nn.functional.cosine_similarity(
            Y_fp32.flatten().unsqueeze(0),
            Y_pt.float().flatten().unsqueeze(0),
        ).item()

        # TRT FP16
        Y_trt16 = build_and_run_trt(onnx_path, X, fp16=True)
        trt16_cos = torch.nn.functional.cosine_similarity(
            Y_fp32.flatten().unsqueeze(0),
            Y_trt16.flatten().unsqueeze(0),
        ).item()

        # TRT FP32
        Y_trt32 = build_and_run_trt(onnx_path, X, fp16=False)
        trt32_cos = torch.nn.functional.cosine_similarity(
            Y_fp32.flatten().unsqueeze(0),
            Y_trt32.flatten().unsqueeze(0),
        ).item()

        print(f"  {f'LN({N},{C})+MM({C},{out_C}) [{label}]':<25s} | {trt16_cos:>12.6f} | {trt32_cos:>12.6f} | {pt_cos:>12.6f}")

        Path(onnx_path).unlink(missing_ok=True)


def main():
    print("=" * 80)
    print("Minimal TRT FP16 MatMul precision test")
    print("=" * 80)

    test_single_matmul()
    test_layernorm_plus_matmul()

    print("\nDone!")


if __name__ == "__main__":
    main()
