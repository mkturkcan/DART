#!/usr/bin/env python3
"""Test real SAM3 weights vs random weights in TRT FP16.

Key question: Do the trained weights cause the TRT FP16 failure,
or is it something else about the ONNX export/graph structure?
"""

import sys
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import onnx
from onnx import TensorProto, helper, numpy_helper
import tensorrt as trt

from sam3.model_builder import build_sam3_image_model

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
DEVICE = "cuda"


def make_attn_block_onnx(output_path, qkv_w, proj_w, qkv_b=None, proj_b=None,
                         seq_len=576, dim=1024, num_heads=16, num_blocks=1):
    """Make ONNX with N attention blocks using provided weights."""
    head_dim = dim // num_heads
    nodes = []
    initializers = []

    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, seq_len, dim])
    prev_name = "X"

    for b in range(num_blocks):
        p = f"b{b}_"

        # LayerNorm
        initializers.append(helper.make_tensor(f"{p}ln_w", TensorProto.FLOAT, [dim],
                                               np.ones(dim, dtype=np.float32)))
        initializers.append(helper.make_tensor(f"{p}ln_b", TensorProto.FLOAT, [dim],
                                               np.zeros(dim, dtype=np.float32)))
        nodes.append(helper.make_node("LayerNormalization",
                                      [prev_name, f"{p}ln_w", f"{p}ln_b"], [f"{p}ln"],
                                      axis=-1, epsilon=1e-6))

        # QKV (weight needs transpose: PyTorch [out,in] -> ONNX MatMul [in,out])
        qkv_w_t = qkv_w[b].T if isinstance(qkv_w, list) else qkv_w.T
        initializers.append(helper.make_tensor(f"{p}W_qkv", TensorProto.FLOAT,
                                               list(qkv_w_t.shape), qkv_w_t.flatten()))
        nodes.append(helper.make_node("MatMul", [f"{p}ln", f"{p}W_qkv"], [f"{p}qkv"]))

        qkv_out = f"{p}qkv"
        if qkv_b is not None:
            b_data = qkv_b[b] if isinstance(qkv_b, list) else qkv_b
            initializers.append(helper.make_tensor(f"{p}qkv_b", TensorProto.FLOAT,
                                                   [3*dim], b_data.flatten()))
            nodes.append(helper.make_node("Add", [qkv_out, f"{p}qkv_b"], [f"{p}qkv_biased"]))
            qkv_out = f"{p}qkv_biased"

        # Reshape + transpose for multi-head attention
        shape_val = np.array([1, seq_len, 3, num_heads, head_dim], dtype=np.int64)
        initializers.append(helper.make_tensor(f"{p}shape_3hd", TensorProto.INT64, [5], shape_val))
        nodes.append(helper.make_node("Reshape", [qkv_out, f"{p}shape_3hd"], [f"{p}qkv_r"]))
        nodes.append(helper.make_node("Transpose", [f"{p}qkv_r"], [f"{p}qkv_t"], perm=[2,0,3,1,4]))

        # Split Q,K,V
        split_s = np.array([1,1,1], dtype=np.int64)
        initializers.append(helper.make_tensor(f"{p}split", TensorProto.INT64, [3], split_s))
        nodes.append(helper.make_node("Split", [f"{p}qkv_t", f"{p}split"],
                                      [f"{p}q", f"{p}k", f"{p}v"], axis=0))

        axes = np.array([0], dtype=np.int64)
        initializers.append(helper.make_tensor(f"{p}axes", TensorProto.INT64, [1], axes))
        for n in ["q", "k", "v"]:
            nodes.append(helper.make_node("Squeeze", [f"{p}{n}", f"{p}axes"], [f"{p}{n}s"]))

        # Q @ K^T -> softmax -> attn @ V
        nodes.append(helper.make_node("Transpose", [f"{p}ks"], [f"{p}kt"], perm=[0,1,3,2]))
        nodes.append(helper.make_node("MatMul", [f"{p}qs", f"{p}kt"], [f"{p}scores_raw"]))

        scale = np.array([head_dim ** -0.5], dtype=np.float32)
        initializers.append(helper.make_tensor(f"{p}scale", TensorProto.FLOAT, [1], scale))
        nodes.append(helper.make_node("Mul", [f"{p}scores_raw", f"{p}scale"], [f"{p}scores"]))
        nodes.append(helper.make_node("Softmax", [f"{p}scores"], [f"{p}attn_w"], axis=-1))
        nodes.append(helper.make_node("MatMul", [f"{p}attn_w", f"{p}vs"], [f"{p}ctx"]))

        # Reshape back
        nodes.append(helper.make_node("Transpose", [f"{p}ctx"], [f"{p}ctx_t"], perm=[0,2,1,3]))
        shape_flat = np.array([1, seq_len, dim], dtype=np.int64)
        initializers.append(helper.make_tensor(f"{p}shape_flat", TensorProto.INT64, [3], shape_flat))
        nodes.append(helper.make_node("Reshape", [f"{p}ctx_t", f"{p}shape_flat"], [f"{p}ctx_flat"]))

        # Output projection
        proj_w_t = proj_w[b].T if isinstance(proj_w, list) else proj_w.T
        initializers.append(helper.make_tensor(f"{p}W_proj", TensorProto.FLOAT,
                                               list(proj_w_t.shape), proj_w_t.flatten()))
        nodes.append(helper.make_node("MatMul", [f"{p}ctx_flat", f"{p}W_proj"], [f"{p}proj"]))

        proj_out = f"{p}proj"
        if proj_b is not None:
            b_data = proj_b[b] if isinstance(proj_b, list) else proj_b
            initializers.append(helper.make_tensor(f"{p}proj_b", TensorProto.FLOAT,
                                                   [dim], b_data.flatten()))
            nodes.append(helper.make_node("Add", [proj_out, f"{p}proj_b"], [f"{p}proj_biased"]))
            proj_out = f"{p}proj_biased"

        # Residual
        res_out = f"{p}res" if b < num_blocks - 1 else "Y"
        nodes.append(helper.make_node("Add", [prev_name, proj_out], [res_out]))
        prev_name = res_out

    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, seq_len, dim])
    graph = helper.make_graph(nodes, "attn_blocks", [X], [Y], initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    onnx.save(model, output_path)


def build_trt_and_run(onnx_path, X, fp16=True):
    """Build TRT engine and run."""
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"  Parse error: {parser.get_error(i)}")
            raise RuntimeError("Parse failed")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    engine_bytes = builder.build_serialized_network(network, config)
    runtime = trt.Runtime(TRT_LOGGER)
    engine = runtime.deserialize_cuda_engine(engine_bytes)
    ctx = engine.create_execution_context()

    d_in = X.float().contiguous().cuda()
    out_shape = ctx.get_tensor_shape(engine.get_tensor_name(engine.num_io_tensors - 1))
    d_out = torch.empty(list(out_shape), dtype=torch.float32, device="cuda")

    ctx.set_tensor_address(engine.get_tensor_name(0), d_in.data_ptr())
    ctx.set_tensor_address(engine.get_tensor_name(engine.num_io_tensors - 1), d_out.data_ptr())
    ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    return d_out


def run_pytorch_ref(X, weights_list, dim, num_heads, use_bias=True):
    """Run the same attention in PyTorch FP32."""
    head_dim = dim // num_heads
    x = X.float()

    for qkv_w, proj_w, qkv_b, proj_b in weights_list:
        h = torch.nn.functional.layer_norm(x, [dim], eps=1e-6)
        qkv = torch.matmul(h, torch.from_numpy(qkv_w.T).to(DEVICE))
        if use_bias and qkv_b is not None:
            qkv = qkv + torch.from_numpy(qkv_b).to(DEVICE)

        B, S, _ = qkv.shape
        qkv = qkv.reshape(B, S, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        scores = torch.matmul(q, k.transpose(-2, -1)) * (head_dim ** -0.5)
        attn = torch.softmax(scores, dim=-1)
        ctx = torch.matmul(attn, v).transpose(1, 2).reshape(B, S, dim)
        out = torch.matmul(ctx, torch.from_numpy(proj_w.T).to(DEVICE))
        if use_bias and proj_b is not None:
            out = out + torch.from_numpy(proj_b).to(DEVICE)
        x = x + out

    return x


def cosine_sim(a, b):
    return torch.nn.functional.cosine_similarity(
        a.float().flatten().unsqueeze(0),
        b.float().flatten().unsqueeze(0),
    ).item()


def main():
    print("Loading model...")
    model = build_sam3_image_model(
        device=DEVICE, checkpoint_path="sam3.pt", eval_mode=True,
    )
    trunk = model.backbone.vision_backbone.trunk

    dim = 1024
    num_heads = 16
    seq_len = 576  # Window size
    onnx_path = "test_weights.onnx"

    X = torch.randn(1, seq_len, dim, device=DEVICE)

    # ==========================================
    # TEST 1: Single block, real vs random weights
    # ==========================================
    print("\n" + "=" * 80)
    print("TEST 1: Single attention block - real vs random weights")
    print("=" * 80)
    print(f"  {'Label':<35s} | {'TRT FP16 cos':>12s} | {'max_diff':>10s}")
    print("  " + "-" * 65)

    for block_idx in [0, 7, 15, 23, 31]:
        attn = trunk.blocks[block_idx].attn
        qkv_w = attn.qkv.weight.data.float().cpu().numpy()
        proj_w = attn.proj.weight.data.float().cpu().numpy()
        qkv_b = attn.qkv.bias.data.float().cpu().numpy() if attn.qkv.bias is not None else None
        proj_b = attn.proj.bias.data.float().cpu().numpy() if attn.proj.bias is not None else None

        make_attn_block_onnx(onnx_path, qkv_w, proj_w, qkv_b, proj_b, seq_len, dim, num_heads)

        with torch.inference_mode():
            Y_fp32 = run_pytorch_ref(X, [(qkv_w, proj_w, qkv_b, proj_b)], dim, num_heads)
        Y_trt = build_trt_and_run(onnx_path, X)
        cos = cosine_sim(Y_fp32, Y_trt)
        diff = (Y_fp32 - Y_trt).abs().max().item()
        print(f"  {'Block ' + str(block_idx) + ' real weights':<35s} | {cos:>12.6f} | {diff:>10.4f}")

        Path(onnx_path).unlink(missing_ok=True)

    # Random weights at same scale
    np.random.seed(42)
    qkv_w_rand = np.random.randn(3072, 1024).astype(np.float32) * 0.025
    proj_w_rand = np.random.randn(1024, 1024).astype(np.float32) * 0.025
    make_attn_block_onnx(onnx_path, qkv_w_rand, proj_w_rand, None, None, seq_len, dim, num_heads)
    with torch.inference_mode():
        Y_fp32 = run_pytorch_ref(X, [(qkv_w_rand, proj_w_rand, None, None)], dim, num_heads, use_bias=False)
    Y_trt = build_trt_and_run(onnx_path, X)
    cos = cosine_sim(Y_fp32, Y_trt)
    diff = (Y_fp32 - Y_trt).abs().max().item()
    print(f"  {'Random weights (no bias)':<35s} | {cos:>12.6f} | {diff:>10.4f}")
    Path(onnx_path).unlink(missing_ok=True)

    # ==========================================
    # TEST 2: What about the BIAS?
    # ==========================================
    print("\n" + "=" * 80)
    print("TEST 2: Effect of QKV bias (block 0 has |bias|=11.3!)")
    print("=" * 80)
    print(f"  {'Label':<35s} | {'TRT FP16 cos':>12s} | {'max_diff':>10s}")
    print("  " + "-" * 65)

    attn = trunk.blocks[0].attn
    qkv_w = attn.qkv.weight.data.float().cpu().numpy()
    proj_w = attn.proj.weight.data.float().cpu().numpy()
    qkv_b = attn.qkv.bias.data.float().cpu().numpy()
    proj_b = attn.proj.bias.data.float().cpu().numpy()

    # Real weights WITH bias
    make_attn_block_onnx(onnx_path, qkv_w, proj_w, qkv_b, proj_b, seq_len, dim, num_heads)
    with torch.inference_mode():
        Y_fp32 = run_pytorch_ref(X, [(qkv_w, proj_w, qkv_b, proj_b)], dim, num_heads)
    Y_trt = build_trt_and_run(onnx_path, X)
    cos = cosine_sim(Y_fp32, Y_trt)
    diff = (Y_fp32 - Y_trt).abs().max().item()
    print(f"  {'Block 0 WITH bias':<35s} | {cos:>12.6f} | {diff:>10.4f}")
    Path(onnx_path).unlink(missing_ok=True)

    # Real weights WITHOUT bias
    make_attn_block_onnx(onnx_path, qkv_w, proj_w, None, None, seq_len, dim, num_heads)
    with torch.inference_mode():
        Y_fp32 = run_pytorch_ref(X, [(qkv_w, proj_w, None, None)], dim, num_heads, use_bias=False)
    Y_trt = build_trt_and_run(onnx_path, X)
    cos = cosine_sim(Y_fp32, Y_trt)
    diff = (Y_fp32 - Y_trt).abs().max().item()
    print(f"  {'Block 0 WITHOUT bias':<35s} | {cos:>12.6f} | {diff:>10.4f}")
    Path(onnx_path).unlink(missing_ok=True)

    # ==========================================
    # TEST 3: Multi-block with real weights
    # ==========================================
    print("\n" + "=" * 80)
    print("TEST 3: Multiple blocks with real weights (depth test)")
    print("=" * 80)
    print(f"  {'Blocks':>7s} | {'TRT FP16 cos':>12s} | {'max_diff':>10s}")
    print("  " + "-" * 40)

    for num_blocks in [1, 2, 4, 8]:
        weights_list = []
        qkv_ws, proj_ws, qkv_bs, proj_bs = [], [], [], []

        for i in range(num_blocks):
            attn = trunk.blocks[i].attn
            qw = attn.qkv.weight.data.float().cpu().numpy()
            pw = attn.proj.weight.data.float().cpu().numpy()
            qb = attn.qkv.bias.data.float().cpu().numpy() if attn.qkv.bias is not None else None
            pb = attn.proj.bias.data.float().cpu().numpy() if attn.proj.bias is not None else None

            qkv_ws.append(qw)
            proj_ws.append(pw)
            qkv_bs.append(qb)
            proj_bs.append(pb)
            weights_list.append((qw, pw, qb, pb))

        make_attn_block_onnx(onnx_path, qkv_ws, proj_ws, qkv_bs, proj_bs,
                             seq_len, dim, num_heads, num_blocks=num_blocks)

        with torch.inference_mode():
            Y_fp32 = run_pytorch_ref(X, weights_list, dim, num_heads)
        Y_trt = build_trt_and_run(onnx_path, X)
        cos = cosine_sim(Y_fp32, Y_trt)
        diff = (Y_fp32 - Y_trt).abs().max().item()
        print(f"  {num_blocks:>7d} | {cos:>12.6f} | {diff:>10.4f}")

        Path(onnx_path).unlink(missing_ok=True)

    # ==========================================
    # TEST 4: Full sequence length (5184 = 72*72 for global attention blocks)
    # ==========================================
    print("\n" + "=" * 80)
    print("TEST 4: Larger sequence lengths with real block 7 weights (global attention)")
    print("=" * 80)
    print(f"  {'Seq len':>8s} | {'TRT FP16 cos':>12s} | {'max_diff':>10s} | {'Score |max|':>12s}")
    print("  " + "-" * 55)

    attn = trunk.blocks[7].attn  # First global attention block
    qkv_w = attn.qkv.weight.data.float().cpu().numpy()
    proj_w = attn.proj.weight.data.float().cpu().numpy()
    qkv_b = attn.qkv.bias.data.float().cpu().numpy() if attn.qkv.bias is not None else None
    proj_b = attn.proj.bias.data.float().cpu().numpy() if attn.proj.bias is not None else None

    for seq_len_test in [64, 256, 576, 1024, 2048]:
        X_test = torch.randn(1, seq_len_test, dim, device=DEVICE)

        make_attn_block_onnx(onnx_path, qkv_w, proj_w, qkv_b, proj_b,
                             seq_len_test, dim, num_heads)

        with torch.inference_mode():
            Y_fp32 = run_pytorch_ref(X_test, [(qkv_w, proj_w, qkv_b, proj_b)], dim, num_heads)

            # Also compute attention score magnitude
            h = torch.nn.functional.layer_norm(X_test.float(), [dim], eps=1e-6)
            qkv = torch.matmul(h, torch.from_numpy(qkv_w.T).to(DEVICE))
            qkv = qkv + torch.from_numpy(qkv_b).to(DEVICE)
            B, S, _ = qkv.shape
            head_dim = dim // num_heads
            qkv = qkv.reshape(B, S, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
            scores = torch.matmul(qkv[0], qkv[1].transpose(-2, -1)) * (head_dim ** -0.5)
            score_max = scores.abs().max().item()

        Y_trt = build_trt_and_run(onnx_path, X_test)
        cos = cosine_sim(Y_fp32, Y_trt)
        diff = (Y_fp32 - Y_trt).abs().max().item()
        print(f"  {seq_len_test:>8d} | {cos:>12.6f} | {diff:>10.4f} | {score_max:>12.2f}")

        Path(onnx_path).unlink(missing_ok=True)

    # ==========================================
    # TEST 5: QKV bias magnitude effect
    # ==========================================
    print("\n" + "=" * 80)
    print("TEST 5: Effect of QKV bias magnitude")
    print("=" * 80)
    print(f"  {'Bias scale':>10s} | {'TRT FP16 cos':>12s} | {'max_diff':>10s} | {'Score |max|':>12s}")
    print("  " + "-" * 55)

    attn = trunk.blocks[0].attn
    qkv_w = attn.qkv.weight.data.float().cpu().numpy()
    proj_w = attn.proj.weight.data.float().cpu().numpy()
    qkv_b_orig = attn.qkv.bias.data.float().cpu().numpy()
    proj_b = attn.proj.bias.data.float().cpu().numpy()

    X_test = torch.randn(1, 576, dim, device=DEVICE)

    for bias_scale in [0.0, 0.1, 0.5, 1.0, 2.0, 5.0]:
        qkv_b_scaled = qkv_b_orig * bias_scale

        make_attn_block_onnx(onnx_path, qkv_w, proj_w,
                             qkv_b_scaled if bias_scale > 0 else None,
                             proj_b if bias_scale > 0 else None,
                             576, dim, num_heads)

        with torch.inference_mode():
            Y_fp32 = run_pytorch_ref(X_test,
                                     [(qkv_w, proj_w,
                                       qkv_b_scaled if bias_scale > 0 else None,
                                       proj_b if bias_scale > 0 else None)],
                                     dim, num_heads,
                                     use_bias=(bias_scale > 0))

            # Score magnitude
            h = torch.nn.functional.layer_norm(X_test.float(), [dim], eps=1e-6)
            qkv = torch.matmul(h, torch.from_numpy(qkv_w.T).to(DEVICE))
            if bias_scale > 0:
                qkv = qkv + torch.from_numpy(qkv_b_scaled).to(DEVICE)
            B, S, _ = qkv.shape
            head_dim = dim // num_heads
            qkv = qkv.reshape(B, S, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
            scores = torch.matmul(qkv[0], qkv[1].transpose(-2, -1)) * (head_dim ** -0.5)
            score_max = scores.abs().max().item()

        Y_trt = build_trt_and_run(onnx_path, X_test)
        cos = cosine_sim(Y_fp32, Y_trt)
        diff = (Y_fp32 - Y_trt).abs().max().item()
        print(f"  {bias_scale:>10.1f} | {cos:>12.6f} | {diff:>10.4f} | {score_max:>12.2f}")

        Path(onnx_path).unlink(missing_ok=True)

    print("\nDone!")


if __name__ == "__main__":
    main()
