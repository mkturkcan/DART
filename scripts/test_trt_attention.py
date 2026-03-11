#!/usr/bin/env python3
"""Test TRT FP16 with full attention mechanism.

Previous tests showed:
- Chained MatMul alone: TRT FP16 cosine 1.0 through 192 layers
- LN->MatMul->Add blocks: TRT FP16 cosine 0.999998 through 32 blocks
- Real backbone: TRT FP16 cosine 0.07

The ONLY missing piece is the attention mechanism: QKV -> split -> Q@K^T -> softmax -> attn@V.
Softmax is highly non-linear and can amplify small input differences into large output differences.

This test builds ONNX models with increasingly realistic attention blocks.
"""

import sys
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
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    engine_bytes = builder.build_serialized_network(network, config)
    if engine_bytes is None:
        raise RuntimeError("Engine build failed")

    runtime = trt.Runtime(TRT_LOGGER)
    engine = runtime.deserialize_cuda_engine(engine_bytes)
    context = engine.create_execution_context()

    input_name = engine.get_tensor_name(0)
    output_name = engine.get_tensor_name(engine.num_io_tensors - 1)

    d_input = input_tensor.float().contiguous().cuda()
    output_shape = context.get_tensor_shape(output_name)
    d_output = torch.empty(list(output_shape), dtype=torch.float32, device="cuda")

    context.set_tensor_address(input_name, d_input.data_ptr())
    context.set_tensor_address(output_name, d_output.data_ptr())

    stream = torch.cuda.current_stream().cuda_stream
    context.execute_async_v3(stream)
    torch.cuda.synchronize()

    return d_output


def make_attention_onnx(seq_len, dim, num_heads, output_path, num_blocks=1,
                        use_residual=True, use_mlp=False, use_layernorm=True):
    """Create ONNX with attention blocks.

    Each block:
      h = LayerNorm(x)                             (optional)
      qkv = MatMul(h, W_qkv)                       [seq, dim] @ [dim, 3*dim] -> [seq, 3*dim]
      q, k, v = split(reshape(qkv))                -> 3 x [heads, seq, head_dim]
      scores = MatMul(q, k^T) * scale               -> [heads, seq, seq]
      attn = Softmax(scores)                         -> [heads, seq, seq]
      context = MatMul(attn, v)                      -> [heads, seq, head_dim]
      out = MatMul(reshape(context), W_out)          -> [seq, dim]
      x = x + out                                   (residual)
      If use_mlp:
        h2 = LayerNorm(x)
        mlp = MatMul(MatMul(h2, W1), W2) + x        (simplified MLP)
    """
    nodes = []
    initializers = []
    head_dim = dim // num_heads

    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, seq_len, dim])
    prev_name = "X"

    for b in range(num_blocks):
        prefix = f"b{b}_"

        # LayerNorm
        if use_layernorm:
            ln_w = np.ones(dim, dtype=np.float32)
            ln_b = np.zeros(dim, dtype=np.float32)
            initializers.append(helper.make_tensor(f"{prefix}ln_w", TensorProto.FLOAT, [dim], ln_w))
            initializers.append(helper.make_tensor(f"{prefix}ln_b", TensorProto.FLOAT, [dim], ln_b))
            ln_out = f"{prefix}ln"
            nodes.append(helper.make_node(
                "LayerNormalization", [prev_name, f"{prefix}ln_w", f"{prefix}ln_b"], [ln_out],
                axis=-1, epsilon=1e-6
            ))
            attn_input = ln_out
        else:
            attn_input = prev_name

        # QKV projection
        W_qkv = np.random.randn(dim, 3 * dim).astype(np.float32) * (dim ** -0.5)
        initializers.append(helper.make_tensor(f"{prefix}W_qkv", TensorProto.FLOAT,
                                               [dim, 3 * dim], W_qkv.flatten()))
        qkv_out = f"{prefix}qkv"
        nodes.append(helper.make_node("MatMul", [attn_input, f"{prefix}W_qkv"], [qkv_out]))

        # Reshape to [1, seq_len, 3, num_heads, head_dim]
        shape_3hd = np.array([1, seq_len, 3, num_heads, head_dim], dtype=np.int64)
        initializers.append(helper.make_tensor(f"{prefix}shape_3hd", TensorProto.INT64,
                                               [5], shape_3hd))
        qkv_reshaped = f"{prefix}qkv_reshaped"
        nodes.append(helper.make_node("Reshape", [qkv_out, f"{prefix}shape_3hd"], [qkv_reshaped]))

        # Transpose to [3, 1, num_heads, seq_len, head_dim]
        qkv_transposed = f"{prefix}qkv_t"
        nodes.append(helper.make_node("Transpose", [qkv_reshaped], [qkv_transposed],
                                      perm=[2, 0, 3, 1, 4]))

        # Split into Q, K, V along axis 0
        q_name = f"{prefix}q"
        k_name = f"{prefix}k"
        v_name = f"{prefix}v"
        split_sizes = np.array([1, 1, 1], dtype=np.int64)
        initializers.append(helper.make_tensor(f"{prefix}split_sizes", TensorProto.INT64,
                                               [3], split_sizes))
        nodes.append(helper.make_node("Split", [qkv_transposed, f"{prefix}split_sizes"],
                                      [q_name, k_name, v_name], axis=0))

        # Squeeze the split dimension: [1, 1, heads, seq, head_dim] -> [1, heads, seq, head_dim]
        squeeze_axes = np.array([0], dtype=np.int64)
        initializers.append(helper.make_tensor(f"{prefix}squeeze_axes", TensorProto.INT64,
                                               [1], squeeze_axes))
        q_sq = f"{prefix}q_sq"
        k_sq = f"{prefix}k_sq"
        v_sq = f"{prefix}v_sq"
        nodes.append(helper.make_node("Squeeze", [q_name, f"{prefix}squeeze_axes"], [q_sq]))
        nodes.append(helper.make_node("Squeeze", [k_name, f"{prefix}squeeze_axes"], [k_sq]))
        nodes.append(helper.make_node("Squeeze", [v_name, f"{prefix}squeeze_axes"], [v_sq]))

        # K^T: transpose last two dims
        k_t = f"{prefix}k_t"
        nodes.append(helper.make_node("Transpose", [k_sq], [k_t], perm=[0, 1, 3, 2]))

        # Q @ K^T
        scores_raw = f"{prefix}scores_raw"
        nodes.append(helper.make_node("MatMul", [q_sq, k_t], [scores_raw]))

        # Scale by 1/sqrt(head_dim)
        scale_val = np.array([head_dim ** -0.5], dtype=np.float32)
        initializers.append(helper.make_tensor(f"{prefix}scale", TensorProto.FLOAT, [1], scale_val))
        scores_scaled = f"{prefix}scores_scaled"
        nodes.append(helper.make_node("Mul", [scores_raw, f"{prefix}scale"], [scores_scaled]))

        # Softmax
        attn_weights = f"{prefix}attn_weights"
        nodes.append(helper.make_node("Softmax", [scores_scaled], [attn_weights], axis=-1))

        # Attn @ V
        context = f"{prefix}context"
        nodes.append(helper.make_node("MatMul", [attn_weights, v_sq], [context]))

        # Transpose back: [1, heads, seq, head_dim] -> [1, seq, heads, head_dim]
        context_t = f"{prefix}context_t"
        nodes.append(helper.make_node("Transpose", [context], [context_t], perm=[0, 2, 1, 3]))

        # Reshape to [1, seq_len, dim]
        shape_flat = np.array([1, seq_len, dim], dtype=np.int64)
        initializers.append(helper.make_tensor(f"{prefix}shape_flat", TensorProto.INT64,
                                               [3], shape_flat))
        context_flat = f"{prefix}context_flat"
        nodes.append(helper.make_node("Reshape", [context_t, f"{prefix}shape_flat"], [context_flat]))

        # Output projection
        W_out = np.random.randn(dim, dim).astype(np.float32) * (dim ** -0.5)
        initializers.append(helper.make_tensor(f"{prefix}W_out", TensorProto.FLOAT,
                                               [dim, dim], W_out.flatten()))
        proj_out = f"{prefix}proj_out"
        nodes.append(helper.make_node("MatMul", [context_flat, f"{prefix}W_out"], [proj_out]))

        # Residual
        if use_residual:
            add_out = f"{prefix}add_out"
            nodes.append(helper.make_node("Add", [prev_name, proj_out], [add_out]))
            block_out = add_out
        else:
            block_out = proj_out

        # MLP block
        if use_mlp:
            mlp_dim = dim * 4

            # LN2
            initializers.append(helper.make_tensor(f"{prefix}ln2_w", TensorProto.FLOAT, [dim],
                                                   np.ones(dim, dtype=np.float32)))
            initializers.append(helper.make_tensor(f"{prefix}ln2_b", TensorProto.FLOAT, [dim],
                                                   np.zeros(dim, dtype=np.float32)))
            ln2_out = f"{prefix}ln2"
            nodes.append(helper.make_node(
                "LayerNormalization", [block_out, f"{prefix}ln2_w", f"{prefix}ln2_b"], [ln2_out],
                axis=-1, epsilon=1e-6
            ))

            # FC1
            W_fc1 = np.random.randn(dim, mlp_dim).astype(np.float32) * (dim ** -0.5)
            initializers.append(helper.make_tensor(f"{prefix}W_fc1", TensorProto.FLOAT,
                                                   [dim, mlp_dim], W_fc1.flatten()))
            fc1_out = f"{prefix}fc1"
            nodes.append(helper.make_node("MatMul", [ln2_out, f"{prefix}W_fc1"], [fc1_out]))

            # GELU activation
            gelu_out = f"{prefix}gelu"
            nodes.append(helper.make_node("Gelu", [fc1_out], [gelu_out], domain="com.microsoft"))

            # FC2
            W_fc2 = np.random.randn(mlp_dim, dim).astype(np.float32) * (mlp_dim ** -0.5)
            initializers.append(helper.make_tensor(f"{prefix}W_fc2", TensorProto.FLOAT,
                                                   [mlp_dim, dim], W_fc2.flatten()))
            fc2_out = f"{prefix}fc2"
            nodes.append(helper.make_node("MatMul", [gelu_out, f"{prefix}W_fc2"], [fc2_out]))

            # Residual
            mlp_add = f"{prefix}mlp_add"
            nodes.append(helper.make_node("Add", [block_out, fc2_out], [mlp_add]))
            block_out = mlp_add

        # Rename last output for final block
        if b == num_blocks - 1:
            # Final rename to Y
            rename_shape = np.array([1, seq_len, dim], dtype=np.int64)
            initializers.append(helper.make_tensor(f"final_shape", TensorProto.INT64,
                                                   [3], rename_shape))
            nodes.append(helper.make_node("Reshape", [block_out, "final_shape"], ["Y"]))
        prev_name = block_out

    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, seq_len, dim])
    graph = helper.make_graph(nodes, "attention_test", [X], [Y], initializers)
    model = helper.make_model(graph, opset_imports=[
        helper.make_opsetid("", 17),
        helper.make_opsetid("com.microsoft", 1),
    ])
    onnx.save(model, output_path)


def run_attention_pytorch(X, onnx_path, num_blocks, dim, num_heads, dtype=torch.float32,
                          use_residual=True, use_mlp=False, use_layernorm=True):
    """Run the same attention computation in PyTorch for comparison."""
    onnx_model = onnx.load(onnx_path)
    weights = {}
    for init in onnx_model.graph.initializer:
        arr = numpy_helper.to_array(init)
        if arr.dtype in (np.float32, np.float64):
            weights[init.name] = torch.from_numpy(arr).to(device=DEVICE, dtype=dtype)
        else:
            weights[init.name] = torch.from_numpy(arr).to(DEVICE)

    head_dim = dim // num_heads
    x = X.to(dtype)

    for b in range(num_blocks):
        prefix = f"b{b}_"

        # LayerNorm
        if use_layernorm:
            h = torch.nn.functional.layer_norm(x, [dim], eps=1e-6)
        else:
            h = x

        # QKV
        qkv = torch.matmul(h, weights[f"{prefix}W_qkv"])

        # Reshape and split
        B, S, _ = qkv.shape
        qkv = qkv.reshape(B, S, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention
        scores = torch.matmul(q, k.transpose(-2, -1)) * (head_dim ** -0.5)
        attn = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn, v)

        # Reshape back
        context = context.transpose(1, 2).reshape(B, S, dim)

        # Output projection
        out = torch.matmul(context, weights[f"{prefix}W_out"])

        # Residual
        if use_residual:
            x = x + out
        else:
            x = out

        # MLP
        if use_mlp:
            h2 = torch.nn.functional.layer_norm(x, [dim], eps=1e-6)
            fc1 = torch.matmul(h2, weights[f"{prefix}W_fc1"])
            fc1 = torch.nn.functional.gelu(fc1)
            fc2 = torch.matmul(fc1, weights[f"{prefix}W_fc2"])
            x = x + fc2

    return x


def cosine_sim(a, b):
    return torch.nn.functional.cosine_similarity(
        a.float().flatten().unsqueeze(0),
        b.float().flatten().unsqueeze(0),
    ).item()


def test_attention_depth():
    """Test attention blocks at different depths."""
    print("\n  Attention blocks (QKV->softmax->attn@V->proj+residual):")
    print(f"  {'Config':>35s} | {'TRT FP16 cos':>12s} | {'PT FP16 cos':>12s} | "
          f"{'TRT max_diff':>12s} | {'PT max_diff':>12s}")
    print("  " + "-" * 95)

    dim = 256
    num_heads = 8
    seq_len = 64
    onnx_path = "test_attn.onnx"

    for num_blocks in [1, 2, 4, 8, 16, 32]:
        torch.manual_seed(42)
        np.random.seed(42)
        make_attention_onnx(seq_len, dim, num_heads, onnx_path, num_blocks=num_blocks)

        X = torch.randn(1, seq_len, dim, device=DEVICE)

        # FP32 reference
        with torch.inference_mode():
            Y_fp32 = run_attention_pytorch(X, onnx_path, num_blocks, dim, num_heads, torch.float32)

        # PyTorch FP16
        with torch.inference_mode():
            Y_pt16 = run_attention_pytorch(X, onnx_path, num_blocks, dim, num_heads, torch.float16)

        pt_cos = cosine_sim(Y_fp32, Y_pt16)
        pt_diff = (Y_fp32 - Y_pt16.float()).abs().max().item()

        # TRT FP16
        Y_trt16 = build_and_run_trt(onnx_path, X, fp16=True)
        trt_cos = cosine_sim(Y_fp32, Y_trt16)
        trt_diff = (Y_fp32 - Y_trt16).abs().max().item()

        label = f"attn dim={dim} heads={num_heads} x{num_blocks}"
        print(f"  {label:>35s} | {trt_cos:>12.6f} | {pt_cos:>12.6f} | "
              f"{trt_diff:>12.4f} | {pt_diff:>12.4f}")

        Path(onnx_path).unlink(missing_ok=True)


def test_attention_dim():
    """Test attention at different dimensions (256 vs 1024)."""
    print("\n  Attention blocks at dim=1024 (matching ViT-H), 16 heads:")
    print(f"  {'Config':>35s} | {'TRT FP16 cos':>12s} | {'PT FP16 cos':>12s} | "
          f"{'TRT max_diff':>12s} | {'PT max_diff':>12s}")
    print("  " + "-" * 95)

    dim = 1024
    num_heads = 16
    seq_len = 64  # Small seq for manageable ONNX
    onnx_path = "test_attn_1024.onnx"

    for num_blocks in [1, 2, 4, 8]:
        torch.manual_seed(42)
        np.random.seed(42)
        make_attention_onnx(seq_len, dim, num_heads, onnx_path, num_blocks=num_blocks)

        X = torch.randn(1, seq_len, dim, device=DEVICE)

        with torch.inference_mode():
            Y_fp32 = run_attention_pytorch(X, onnx_path, num_blocks, dim, num_heads, torch.float32)
            Y_pt16 = run_attention_pytorch(X, onnx_path, num_blocks, dim, num_heads, torch.float16)

        pt_cos = cosine_sim(Y_fp32, Y_pt16)
        pt_diff = (Y_fp32 - Y_pt16.float()).abs().max().item()

        Y_trt16 = build_and_run_trt(onnx_path, X, fp16=True)
        trt_cos = cosine_sim(Y_fp32, Y_trt16)
        trt_diff = (Y_fp32 - Y_trt16).abs().max().item()

        label = f"attn dim=1024 heads=16 x{num_blocks}"
        print(f"  {label:>35s} | {trt_cos:>12.6f} | {pt_cos:>12.6f} | "
              f"{trt_diff:>12.4f} | {pt_diff:>12.4f}")

        Path(onnx_path).unlink(missing_ok=True)


def test_attention_seq_len():
    """Test with larger sequence lengths (closer to ViT-H's 5184 tokens)."""
    print("\n  Attention at larger sequence lengths (dim=256, 8 heads, 4 blocks):")
    print(f"  {'Seq len':>10s} | {'TRT FP16 cos':>12s} | {'PT FP16 cos':>12s} | "
          f"{'TRT max_diff':>12s} | {'PT max_diff':>12s}")
    print("  " + "-" * 70)

    dim = 256
    num_heads = 8
    num_blocks = 4
    onnx_path = "test_attn_seq.onnx"

    for seq_len in [64, 256, 576, 1024]:
        torch.manual_seed(42)
        np.random.seed(42)
        make_attention_onnx(seq_len, dim, num_heads, onnx_path, num_blocks=num_blocks)

        X = torch.randn(1, seq_len, dim, device=DEVICE)

        with torch.inference_mode():
            Y_fp32 = run_attention_pytorch(X, onnx_path, num_blocks, dim, num_heads, torch.float32)
            Y_pt16 = run_attention_pytorch(X, onnx_path, num_blocks, dim, num_heads, torch.float16)

        pt_cos = cosine_sim(Y_fp32, Y_pt16)
        pt_diff = (Y_fp32 - Y_pt16.float()).abs().max().item()

        Y_trt16 = build_and_run_trt(onnx_path, X, fp16=True)
        trt_cos = cosine_sim(Y_fp32, Y_trt16)
        trt_diff = (Y_fp32 - Y_trt16).abs().max().item()

        print(f"  {seq_len:>10d} | {trt_cos:>12.6f} | {pt_cos:>12.6f} | "
              f"{trt_diff:>12.4f} | {pt_diff:>12.4f}")

        Path(onnx_path).unlink(missing_ok=True)


def test_attention_with_mlp():
    """Test full transformer blocks (attention + MLP) at dim=256."""
    print("\n  Full transformer blocks (attention + MLP), dim=256, 8 heads:")
    print(f"  {'Blocks':>7s} | {'TRT FP16 cos':>12s} | {'PT FP16 cos':>12s} | "
          f"{'TRT max_diff':>12s} | {'PT max_diff':>12s}")
    print("  " + "-" * 70)

    dim = 256
    num_heads = 8
    seq_len = 64
    onnx_path = "test_attn_mlp.onnx"

    for num_blocks in [1, 2, 4, 8, 16]:
        torch.manual_seed(42)
        np.random.seed(42)

        try:
            make_attention_onnx(seq_len, dim, num_heads, onnx_path,
                                num_blocks=num_blocks, use_mlp=True)
        except Exception as e:
            print(f"  {num_blocks:>7d} | ONNX build failed: {e}")
            continue

        X = torch.randn(1, seq_len, dim, device=DEVICE)

        with torch.inference_mode():
            Y_fp32 = run_attention_pytorch(X, onnx_path, num_blocks, dim, num_heads,
                                           torch.float32, use_mlp=True)
            Y_pt16 = run_attention_pytorch(X, onnx_path, num_blocks, dim, num_heads,
                                           torch.float16, use_mlp=True)

        pt_cos = cosine_sim(Y_fp32, Y_pt16)
        pt_diff = (Y_fp32 - Y_pt16.float()).abs().max().item()

        try:
            Y_trt16 = build_and_run_trt(onnx_path, X, fp16=True)
            trt_cos = cosine_sim(Y_fp32, Y_trt16)
            trt_diff = (Y_fp32 - Y_trt16).abs().max().item()
            print(f"  {num_blocks:>7d} | {trt_cos:>12.6f} | {pt_cos:>12.6f} | "
                  f"{trt_diff:>12.4f} | {pt_diff:>12.4f}")
        except Exception as e:
            print(f"  {num_blocks:>7d} | TRT build failed: {e}")

        Path(onnx_path).unlink(missing_ok=True)


def test_no_residual():
    """Test attention WITHOUT residual connection to see if residual amplifies error."""
    print("\n  Attention WITHOUT residual (dim=256, 8 heads):")
    print(f"  {'Blocks':>7s} | {'TRT FP16 cos':>12s} | {'PT FP16 cos':>12s} | "
          f"{'With residual TRT':>18s} | {'With residual PT':>16s}")
    print("  " + "-" * 80)

    dim = 256
    num_heads = 8
    seq_len = 64
    onnx_path_nr = "test_attn_nr.onnx"
    onnx_path_r = "test_attn_r.onnx"

    for num_blocks in [1, 2, 4, 8]:
        torch.manual_seed(42)
        np.random.seed(42)

        # Without residual
        make_attention_onnx(seq_len, dim, num_heads, onnx_path_nr,
                            num_blocks=num_blocks, use_residual=False)
        X = torch.randn(1, seq_len, dim, device=DEVICE)

        with torch.inference_mode():
            Y_fp32_nr = run_attention_pytorch(X, onnx_path_nr, num_blocks, dim, num_heads,
                                              torch.float32, use_residual=False)
            Y_pt16_nr = run_attention_pytorch(X, onnx_path_nr, num_blocks, dim, num_heads,
                                              torch.float16, use_residual=False)
        pt_cos_nr = cosine_sim(Y_fp32_nr, Y_pt16_nr)
        Y_trt_nr = build_and_run_trt(onnx_path_nr, X, fp16=True)
        trt_cos_nr = cosine_sim(Y_fp32_nr, Y_trt_nr)

        # With residual (same seed for same weights)
        torch.manual_seed(42)
        np.random.seed(42)
        make_attention_onnx(seq_len, dim, num_heads, onnx_path_r,
                            num_blocks=num_blocks, use_residual=True)

        with torch.inference_mode():
            Y_fp32_r = run_attention_pytorch(X, onnx_path_r, num_blocks, dim, num_heads,
                                             torch.float32, use_residual=True)
            Y_pt16_r = run_attention_pytorch(X, onnx_path_r, num_blocks, dim, num_heads,
                                             torch.float16, use_residual=True)
        pt_cos_r = cosine_sim(Y_fp32_r, Y_pt16_r)
        Y_trt_r = build_and_run_trt(onnx_path_r, X, fp16=True)
        trt_cos_r = cosine_sim(Y_fp32_r, Y_trt_r)

        print(f"  {num_blocks:>7d} | {trt_cos_nr:>12.6f} | {pt_cos_nr:>12.6f} | "
              f"{trt_cos_r:>18.6f} | {pt_cos_r:>16.6f}")

        Path(onnx_path_nr).unlink(missing_ok=True)
        Path(onnx_path_r).unlink(missing_ok=True)


def main():
    print("=" * 80)
    print("TRT FP16 Attention Mechanism Analysis")
    print("=" * 80)

    # Test 1: Attention blocks at dim=256
    print("\n" + "=" * 80)
    print("TEST 1: Attention blocks vs depth (dim=256)")
    print("=" * 80)
    test_attention_depth()

    # Test 2: Attention at dim=1024
    print("\n" + "=" * 80)
    print("TEST 2: Attention blocks at dim=1024 (ViT-H scale)")
    print("=" * 80)
    test_attention_dim()

    # Test 3: Varying sequence length
    print("\n" + "=" * 80)
    print("TEST 3: Attention with varying sequence length")
    print("=" * 80)
    test_attention_seq_len()

    # Test 4: With vs without residual
    print("\n" + "=" * 80)
    print("TEST 4: Effect of residual connection")
    print("=" * 80)
    test_no_residual()

    # Test 5: Full transformer blocks (attn + MLP)
    print("\n" + "=" * 80)
    print("TEST 5: Full transformer blocks (attention + MLP)")
    print("=" * 80)
    test_attention_with_mlp()

    print("\nDone!")


if __name__ == "__main__":
    main()
