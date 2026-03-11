#!/usr/bin/env python3
"""Test TRT FP16 MatMul precision with varying input magnitudes.

Hypothesis: Later ViT blocks have activation magnitudes ~300. At this scale,
FP16 precision is +/-0.25, which might cause TRT's accumulation to diverge
differently from PyTorch's.

Also tests multi-layer chained MatMul in TRT to find the depth where
error explodes.
"""

import sys
import time
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import onnx
    from onnx import TensorProto, helper, numpy_helper
except ImportError:
    print("pip install onnx")
    sys.exit(1)

import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
DEVICE = "cuda"


def make_matmul_onnx(M, K, N, output_path, weight_data=None):
    """Create ONNX: Y = MatMul(X, W)"""
    if weight_data is None:
        weight_data = np.random.randn(K, N).astype(np.float32)

    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [M, K])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [M, N])
    W = helper.make_tensor("W", TensorProto.FLOAT, [K, N], weight_data.flatten())
    matmul = helper.make_node("MatMul", ["X", "W"], ["Y"])
    graph = helper.make_graph([matmul], "test", [X], [Y], [W])
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
    context = engine.create_execution_context()

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


def make_chained_matmul_onnx(num_layers, dim, output_path):
    """Create ONNX with chained MatMul: x = MatMul(x, W_i) for i in range(num_layers).

    Each weight matrix is dim x dim with small values (like a normalized linear layer).
    """
    nodes = []
    initializers = []
    weight_tensors = []

    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, dim])

    prev_name = "X"
    for i in range(num_layers):
        # Use orthogonal-like weights to prevent explosion/vanishing
        # Initialize as identity + small perturbation (like a well-initialized linear layer)
        W_np = np.eye(dim, dtype=np.float32) + np.random.randn(dim, dim).astype(np.float32) * 0.01
        weight_tensors.append(W_np)

        w_name = f"W_{i}"
        out_name = f"mm_{i}" if i < num_layers - 1 else "Y"

        W = helper.make_tensor(w_name, TensorProto.FLOAT, [dim, dim], W_np.flatten())
        initializers.append(W)

        matmul = helper.make_node("MatMul", [prev_name, w_name], [out_name])
        nodes.append(matmul)
        prev_name = out_name

    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, dim])
    graph = helper.make_graph(nodes, "chained_matmul", [X], [Y], initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    onnx.save(model, output_path)
    return weight_tensors


def make_vit_block_onnx(num_blocks, dim, output_path):
    """Create ONNX mimicking ViT blocks: LN -> MatMul -> Add (residual).

    Each block:
      h = LayerNorm(x)
      h = MatMul(h, W)   # simplified linear projection
      x = x + h          # residual connection

    This tests if the residual connection pattern amplifies FP16 errors.
    """
    nodes = []
    initializers = []

    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 64, dim])
    prev_name = "X"

    for i in range(num_blocks):
        # LayerNorm weights
        ln_w = np.ones(dim, dtype=np.float32)
        ln_b = np.zeros(dim, dtype=np.float32)
        ln_w_name = f"ln_w_{i}"
        ln_b_name = f"ln_b_{i}"
        initializers.append(helper.make_tensor(ln_w_name, TensorProto.FLOAT, [dim], ln_w))
        initializers.append(helper.make_tensor(ln_b_name, TensorProto.FLOAT, [dim], ln_b))

        # LayerNorm
        ln_out = f"ln_{i}"
        nodes.append(helper.make_node(
            "LayerNormalization", [prev_name, ln_w_name, ln_b_name], [ln_out],
            axis=-1, epsilon=1e-6
        ))

        # MatMul weight (identity + perturbation)
        W_np = np.eye(dim, dtype=np.float32) + np.random.randn(dim, dim).astype(np.float32) * 0.02
        w_name = f"W_{i}"
        initializers.append(helper.make_tensor(w_name, TensorProto.FLOAT, [dim, dim], W_np.flatten()))

        mm_out = f"mm_{i}"
        nodes.append(helper.make_node("MatMul", [ln_out, w_name], [mm_out]))

        # Residual Add
        add_out = f"add_{i}" if i < num_blocks - 1 else "Y"
        nodes.append(helper.make_node("Add", [prev_name, mm_out], [add_out]))
        prev_name = add_out

    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 64, dim])
    graph = helper.make_graph(nodes, "vit_blocks", [X], [Y], initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    onnx.save(model, output_path)


def test_magnitude_effect():
    """Test if input magnitude affects TRT FP16 MatMul precision."""
    print("\n  Single MatMul precision vs input magnitude:")
    print(f"  {'Magnitude':>10s} | {'TRT FP16 cos':>12s} | {'PT FP16 cos':>12s} | "
          f"{'TRT max_diff':>12s} | {'PT max_diff':>12s} | {'Output range':>20s}")
    print("  " + "-" * 100)

    # Use realistic ViT dimensions: 5184 tokens, 1024 hidden, 3072 QKV output
    M, K, N = 5184, 1024, 3072
    onnx_path = "test_mag.onnx"

    for magnitude in [1.0, 5.0, 10.0, 50.0, 100.0, 200.0, 300.0, 500.0]:
        # Use same weights each time
        torch.manual_seed(42)
        W_np = np.random.randn(K, N).astype(np.float32) * 0.02  # Small weights like real init
        make_matmul_onnx(M, K, N, onnx_path, weight_data=W_np)
        W_torch = torch.from_numpy(W_np).to(DEVICE)

        # Input scaled to target magnitude
        X = torch.randn(M, K, device=DEVICE) * magnitude

        # FP32 reference
        Y_fp32 = torch.matmul(X, W_torch)

        # PyTorch FP16
        Y_pt16 = torch.matmul(X.half(), W_torch.half())
        pt_cos = torch.nn.functional.cosine_similarity(
            Y_fp32.flatten().unsqueeze(0),
            Y_pt16.float().flatten().unsqueeze(0),
        ).item()
        pt_diff = (Y_fp32 - Y_pt16.float()).abs().max().item()

        # TRT FP16
        Y_trt16 = build_and_run_trt(onnx_path, X, fp16=True)
        trt_cos = torch.nn.functional.cosine_similarity(
            Y_fp32.flatten().unsqueeze(0),
            Y_trt16.flatten().unsqueeze(0),
        ).item()
        trt_diff = (Y_fp32 - Y_trt16).abs().max().item()

        out_range = f"[{Y_fp32.min().item():.1f}, {Y_fp32.max().item():.1f}]"
        print(f"  {magnitude:>10.1f} | {trt_cos:>12.6f} | {pt_cos:>12.6f} | "
              f"{trt_diff:>12.4f} | {pt_diff:>12.4f} | {out_range:>20s}")

    Path(onnx_path).unlink(missing_ok=True)


def test_chained_matmul_depth():
    """Test how chained MatMul error compounds in TRT FP16 vs PyTorch FP16."""
    print("\n  Chained MatMul error vs depth (dim=1024):")
    print(f"  {'Depth':>6s} | {'TRT FP16 cos':>12s} | {'PT FP16 cos':>12s} | "
          f"{'TRT max_diff':>12s} | {'Output absmax':>14s}")
    print("  " + "-" * 75)

    dim = 1024
    onnx_path = "test_chain.onnx"

    for depth in [1, 2, 4, 8, 16, 32, 64, 128, 192]:
        torch.manual_seed(42)
        weight_tensors = make_chained_matmul_onnx(depth, dim, onnx_path)

        X = torch.randn(1, dim, device=DEVICE)

        # FP32 reference
        x_fp32 = X.clone()
        for W_np in weight_tensors:
            W = torch.from_numpy(W_np).to(DEVICE)
            x_fp32 = torch.matmul(x_fp32, W)

        # PyTorch FP16
        x_pt16 = X.half()
        for W_np in weight_tensors:
            W = torch.from_numpy(W_np).to(DEVICE).half()
            x_pt16 = torch.matmul(x_pt16, W)

        pt_cos = torch.nn.functional.cosine_similarity(
            x_fp32.flatten().unsqueeze(0),
            x_pt16.float().flatten().unsqueeze(0),
        ).item()

        # TRT FP16
        Y_trt16 = build_and_run_trt(onnx_path, X, fp16=True)
        trt_cos = torch.nn.functional.cosine_similarity(
            x_fp32.flatten().unsqueeze(0),
            Y_trt16.flatten().unsqueeze(0),
        ).item()
        trt_diff = (x_fp32 - Y_trt16).abs().max().item()

        print(f"  {depth:>6d} | {trt_cos:>12.6f} | {pt_cos:>12.6f} | "
              f"{trt_diff:>12.4f} | {x_fp32.abs().max().item():>14.4f}")

        Path(onnx_path).unlink(missing_ok=True)


def test_vit_block_depth():
    """Test ViT-like blocks (LN -> MatMul -> Add) to see if residual connections
    amplify FP16 error in TRT differently than in PyTorch."""
    print("\n  ViT-like block (LN->MatMul->Add) error vs depth (dim=256):")
    print(f"  {'Blocks':>7s} | {'TRT FP16 cos':>12s} | {'PT FP16 cos':>12s} | "
          f"{'TRT max_diff':>12s} | {'PT max_diff':>12s} | {'Output absmax':>14s}")
    print("  " + "-" * 85)

    dim = 256  # Smaller to keep ONNX manageable
    seq_len = 64
    onnx_path = "test_vit_blocks.onnx"

    for num_blocks in [1, 2, 4, 8, 16, 32]:
        torch.manual_seed(42)
        make_vit_block_onnx(num_blocks, dim, onnx_path)

        X = torch.randn(1, seq_len, dim, device=DEVICE)

        # FP32 reference (manual)
        x_fp32 = X.clone()
        for i in range(num_blocks):
            h = torch.nn.functional.layer_norm(x_fp32, [dim], eps=1e-6)
            # Same weight init as ONNX
            torch.manual_seed(42 + i)  # Deterministic per-block
            np.random.seed(42)  # Reset numpy seed to match ONNX
            _ = np.random.randn(dim, dim)  # Skip to match iteration
            W = torch.eye(dim, device=DEVICE) + torch.randn(dim, dim, device=DEVICE) * 0.02
            # Re-seed properly - we need to match exactly
            # Actually, let's just read weights from ONNX
            pass

        # Better approach: load weights from ONNX and run manually
        onnx_model = onnx.load(onnx_path)
        weights = {}
        for init in onnx_model.graph.initializer:
            weights[init.name] = torch.from_numpy(numpy_helper.to_array(init)).to(DEVICE)

        # FP32 reference
        x_fp32 = X.clone()
        for i in range(num_blocks):
            h = torch.nn.functional.layer_norm(x_fp32, [dim], eps=1e-6)
            W = weights[f"W_{i}"]
            h = torch.matmul(h, W)
            x_fp32 = x_fp32 + h

        # PyTorch FP16
        x_pt16 = X.half()
        for i in range(num_blocks):
            h = torch.nn.functional.layer_norm(x_pt16, [dim], eps=1e-6)
            W = weights[f"W_{i}"].half()
            h = torch.matmul(h, W)
            x_pt16 = x_pt16 + h

        pt_cos = torch.nn.functional.cosine_similarity(
            x_fp32.flatten().unsqueeze(0),
            x_pt16.float().flatten().unsqueeze(0),
        ).item()
        pt_diff = (x_fp32 - x_pt16.float()).abs().max().item()

        # TRT FP16
        Y_trt16 = build_and_run_trt(onnx_path, X, fp16=True)
        trt_cos = torch.nn.functional.cosine_similarity(
            x_fp32.flatten().unsqueeze(0),
            Y_trt16.flatten().unsqueeze(0),
        ).item()
        trt_diff = (x_fp32 - Y_trt16).abs().max().item()

        print(f"  {num_blocks:>7d} | {trt_cos:>12.6f} | {pt_cos:>12.6f} | "
              f"{trt_diff:>12.4f} | {pt_diff:>12.4f} | {x_fp32.abs().max().item():>14.4f}")

        Path(onnx_path).unlink(missing_ok=True)


def test_vit_block_1024():
    """Same as above but with dim=1024 (matching ViT-H) - fewer blocks to stay manageable."""
    print("\n  ViT-like block (LN->MatMul->Add) at dim=1024:")
    print(f"  {'Blocks':>7s} | {'TRT FP16 cos':>12s} | {'PT FP16 cos':>12s} | "
          f"{'TRT max_diff':>12s} | {'PT max_diff':>12s} | {'Output absmax':>14s}")
    print("  " + "-" * 85)

    dim = 1024
    seq_len = 64  # Small seq to keep model size reasonable
    onnx_path = "test_vit_blocks_1024.onnx"

    for num_blocks in [1, 2, 4, 8, 16, 32]:
        torch.manual_seed(42)
        make_vit_block_onnx(num_blocks, dim, onnx_path)

        X = torch.randn(1, seq_len, dim, device=DEVICE)

        # Load weights from ONNX
        onnx_model = onnx.load(onnx_path)
        weights = {}
        for init in onnx_model.graph.initializer:
            weights[init.name] = torch.from_numpy(numpy_helper.to_array(init)).to(DEVICE)

        # FP32 reference
        x_fp32 = X.clone()
        for i in range(num_blocks):
            h = torch.nn.functional.layer_norm(x_fp32, [dim], eps=1e-6)
            W = weights[f"W_{i}"]
            h = torch.matmul(h, W)
            x_fp32 = x_fp32 + h

        # PyTorch FP16
        x_pt16 = X.half()
        for i in range(num_blocks):
            h = torch.nn.functional.layer_norm(x_pt16, [dim], eps=1e-6)
            W = weights[f"W_{i}"].half()
            h = torch.matmul(h, W)
            x_pt16 = x_pt16 + h

        pt_cos = torch.nn.functional.cosine_similarity(
            x_fp32.flatten().unsqueeze(0),
            x_pt16.float().flatten().unsqueeze(0),
        ).item()
        pt_diff = (x_fp32 - x_pt16.float()).abs().max().item()

        # TRT FP16
        Y_trt16 = build_and_run_trt(onnx_path, X, fp16=True)
        trt_cos = torch.nn.functional.cosine_similarity(
            x_fp32.flatten().unsqueeze(0),
            Y_trt16.flatten().unsqueeze(0),
        ).item()
        trt_diff = (x_fp32 - Y_trt16).abs().max().item()

        print(f"  {num_blocks:>7d} | {trt_cos:>12.6f} | {pt_cos:>12.6f} | "
              f"{trt_diff:>12.4f} | {pt_diff:>12.4f} | {x_fp32.abs().max().item():>14.4f}")

        Path(onnx_path).unlink(missing_ok=True)


def main():
    print("=" * 80)
    print("TRT FP16 Precision: Magnitude & Depth Analysis")
    print("=" * 80)

    # Part 1: Does input magnitude affect single-MatMul TRT FP16 precision?
    print("\n" + "=" * 80)
    print("PART 1: Single MatMul precision vs input magnitude")
    print("=" * 80)
    test_magnitude_effect()

    # Part 2: Chained MatMul (no residual) - pure compounding
    print("\n" + "=" * 80)
    print("PART 2: Chained MatMul error compounding (no residual)")
    print("=" * 80)
    test_chained_matmul_depth()

    # Part 3: ViT-like blocks with residual (dim=256)
    print("\n" + "=" * 80)
    print("PART 3: ViT-like blocks with residual (dim=256)")
    print("=" * 80)
    test_vit_block_depth()

    # Part 4: ViT-like blocks with residual (dim=1024, matching ViT-H)
    print("\n" + "=" * 80)
    print("PART 4: ViT-like blocks with residual (dim=1024)")
    print("=" * 80)
    test_vit_block_1024()

    print("\nDone!")


if __name__ == "__main__":
    main()
