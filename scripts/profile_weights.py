#!/usr/bin/env python3
"""Profile SAM3 weight statistics to find FP16-hostile patterns.

Synthetic attention models work perfectly in TRT FP16. The real SAM3 backbone
fails catastrophically. The difference must be in the trained weight properties.

This script profiles:
1. Weight magnitudes and distributions per layer
2. FP16 quantization error in weights themselves
3. Outlier analysis
4. Tests attention with REAL weights in a synthetic ONNX graph
"""

import sys
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sam3.model_builder import build_sam3_image_model

DEVICE = "cuda"


def profile_weight_stats(model):
    """Profile weight statistics of all ViT layers."""
    trunk = model.backbone.vision_backbone.trunk

    print("\n  Weight statistics for all ViT parameters:")
    print(f"  {'Parameter':<55s} | {'Shape':>20s} | {'|max|':>8s} | {'std':>8s} | "
          f"{'FP16 err':>10s} | {'Outliers':>8s}")
    print("  " + "-" * 130)

    total_params = 0
    total_outliers = 0
    problematic = []

    for name, param in trunk.named_parameters():
        p = param.data.float()
        p16 = p.half().float()
        fp16_err = (p - p16).abs()

        abs_max = p.abs().max().item()
        std = p.std().item()
        mean_fp16_err = fp16_err.mean().item()
        max_fp16_err = fp16_err.max().item()

        # Count "outliers" - values > 4*std
        outlier_count = (p.abs() > 4 * std).sum().item()
        outlier_pct = outlier_count / p.numel() * 100

        # Flag if FP16 quantization error is large relative to values
        rel_fp16_err = (fp16_err / p.abs().clamp(min=1e-6)).mean().item()

        shape_str = str(list(p.shape))
        total_params += p.numel()
        total_outliers += outlier_count

        # Only print noteworthy layers
        if abs_max > 1.0 or outlier_pct > 1.0 or 'qkv' in name or 'proj' in name:
            print(f"  {name:<55s} | {shape_str:>20s} | {abs_max:>8.3f} | {std:>8.4f} | "
                  f"{max_fp16_err:>10.6f} | {outlier_pct:>7.2f}%")

        if abs_max > 5.0:
            problematic.append((name, abs_max, std, p.shape))

    print(f"\n  Total params: {total_params:,}")
    print(f"  Total outliers (>4*std): {total_outliers:,} ({total_outliers/total_params*100:.3f}%)")

    if problematic:
        print(f"\n  Layers with |max| > 5:")
        for name, mx, std, shape in problematic:
            print(f"    {name}: |max|={mx:.3f}, std={std:.4f}, shape={list(shape)}")

    return trunk


def profile_attention_weight_properties(model):
    """Specifically analyze attention weight properties that could cause FP16 issues."""
    trunk = model.backbone.vision_backbone.trunk

    print("\n\n  Attention weight analysis (per block):")
    print(f"  {'Block':>6s} | {'QKV |max|':>10s} | {'QKV std':>8s} | {'Proj |max|':>10s} | "
          f"{'Proj std':>8s} | {'Q@K^T scale':>12s} | {'Softmax temp':>13s}")
    print("  " + "-" * 90)

    for i, block in enumerate(trunk.blocks):
        attn = block.attn
        qkv_w = attn.qkv.weight.data.float()
        proj_w = attn.proj.weight.data.float()

        # Estimate Q@K^T magnitude: if ||q|| ~ ||k|| ~ std(qkv_w) * sqrt(dim_in)
        # then Q@K^T ~ head_dim * std(q) * std(k)
        head_dim = qkv_w.shape[0] // 3 // attn.num_heads
        dim_in = qkv_w.shape[1]

        # Q, K weight norms
        qkv_out = qkv_w.shape[0]
        q_w = qkv_w[:qkv_out//3]
        k_w = qkv_w[qkv_out//3:2*qkv_out//3]

        # Frobenius norm per head
        q_norms = q_w.reshape(attn.num_heads, head_dim, -1).norm(dim=(1,2))
        k_norms = k_w.reshape(attn.num_heads, head_dim, -1).norm(dim=(1,2))

        # Expected Q@K^T magnitude (rough estimate)
        # score ~ (q_norm * k_norm / sqrt(dim_in)) * input_std * 1/sqrt(head_dim)
        est_score = (q_norms * k_norms).mean().item() / dim_in

        print(f"  {i:>6d} | {qkv_w.abs().max().item():>10.4f} | {qkv_w.std().item():>8.5f} | "
              f"{proj_w.abs().max().item():>10.4f} | {proj_w.std().item():>8.5f} | "
              f"{est_score:>12.4f} | {1.0/head_dim**0.5:>13.4f}")


def test_qkv_fp16_sensitivity(model):
    """Test if QKV weights are particularly sensitive to FP16 quantization.

    The FP16 quantization error in weights could be amplified by the
    large reduction dimension (1024 inputs per QKV projection).
    """
    trunk = model.backbone.vision_backbone.trunk
    dummy = torch.randn(1, 72, 72, 1024, device=DEVICE)  # Typical ViT input

    print("\n\n  QKV projection FP16 sensitivity (real weights, random input):")
    print(f"  {'Block':>6s} | {'QKV cos(fp32,fp16)':>20s} | {'QKV max_diff':>12s} | "
          f"{'Score cos':>10s} | {'Score max_diff':>14s} | {'Attn cos':>10s}")
    print("  " + "-" * 100)

    with torch.inference_mode():
        x = dummy.float()
        for i, block in enumerate(trunk.blocks):
            attn = block.attn

            # Get QKV in FP32
            qkv_fp32 = attn.qkv(x.float())

            # Get QKV in FP16 (simulating TRT)
            qkv_fp16 = attn.qkv.half()(x.half())

            qkv_cos = torch.nn.functional.cosine_similarity(
                qkv_fp32.flatten().unsqueeze(0),
                qkv_fp16.float().flatten().unsqueeze(0),
            ).item()
            qkv_diff = (qkv_fp32 - qkv_fp16.float()).abs().max().item()

            # Split into Q, K, V and compute scores
            B, H, W, C = qkv_fp32.shape
            num_heads = attn.num_heads
            head_dim = C // 3 // num_heads

            qkv_r32 = qkv_fp32.reshape(B, H*W, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
            qkv_r16 = qkv_fp16.reshape(B, H*W, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)

            # Compute attention scores
            q32, k32 = qkv_r32[0], qkv_r32[1]
            q16, k16 = qkv_r16[0], qkv_r16[1]

            # Only compute for first 576 tokens (windowed attention window)
            q32_w = q32[:, :, :576, :]
            k32_w = k32[:, :, :576, :]
            q16_w = q16[:, :, :576, :]
            k16_w = k16[:, :, :576, :]

            scores_fp32 = torch.matmul(q32_w.float(), k32_w.float().transpose(-2, -1)) * (head_dim ** -0.5)
            scores_fp16 = torch.matmul(q16_w.half(), k16_w.half().transpose(-2, -1)).float() * (head_dim ** -0.5)

            score_cos = torch.nn.functional.cosine_similarity(
                scores_fp32.flatten().unsqueeze(0),
                scores_fp16.flatten().unsqueeze(0),
            ).item()
            score_diff = (scores_fp32 - scores_fp16).abs().max().item()

            # Softmax
            attn_fp32 = torch.softmax(scores_fp32, dim=-1)
            attn_fp16 = torch.softmax(scores_fp16.half(), dim=-1)
            attn_cos = torch.nn.functional.cosine_similarity(
                attn_fp32.flatten().unsqueeze(0),
                attn_fp16.float().flatten().unsqueeze(0),
            ).item()

            print(f"  {i:>6d} | {qkv_cos:>20.6f} | {qkv_diff:>12.4f} | "
                  f"{score_cos:>10.6f} | {score_diff:>14.4f} | {attn_cos:>10.6f}")

            # Run block for next iteration
            x = block(x)

            if i >= 15:  # First 16 blocks
                break


def test_weight_condition_number(model):
    """Test the condition number of key weight matrices.

    High condition number = small FP16 rounding errors get amplified.
    """
    trunk = model.backbone.vision_backbone.trunk

    print("\n\n  Weight matrix condition numbers (higher = more FP16 sensitive):")
    print(f"  {'Layer':<45s} | {'Cond(F32)':>12s} | {'|max|/|min|':>12s} | "
          f"{'Spectral norm':>14s}")
    print("  " + "-" * 95)

    for i in [0, 7, 15, 23, 31]:  # Key blocks
        block = trunk.blocks[i]
        for name, param in [
            (f"block.{i}.attn.qkv", block.attn.qkv.weight),
            (f"block.{i}.attn.proj", block.attn.proj.weight),
            (f"block.{i}.mlp.fc1", block.mlp.layers[0][0].weight if hasattr(block.mlp, 'layers') else None),
        ]:
            if param is None:
                continue
            w = param.data.float()

            # Singular values
            try:
                if w.dim() == 2:
                    S = torch.linalg.svdvals(w)
                    cond = (S[0] / S[-1]).item()
                    spectral = S[0].item()
                    min_sv = S[-1].item()
                    print(f"  {name:<45s} | {cond:>12.1f} | {S[0].item()/S[-1].item():>12.1f} | "
                          f"{spectral:>14.4f}")
            except Exception as e:
                print(f"  {name:<45s} | ERROR: {e}")


def test_real_vs_random_weights():
    """Key test: take the real QKV weights from block 0, put them in a synthetic
    ONNX attention model, and test in TRT FP16. Compare with random weights.

    If real weights break TRT but random don't, we've isolated the issue to weight properties.
    """
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    trunk = model.backbone.vision_backbone.trunk
    block = trunk.blocks[0]
    attn = block.attn

    dim = 1024
    num_heads = 16
    head_dim = 64
    seq_len = 576  # Window size

    print("\n\n  Real vs random weights in synthetic ONNX (seq=576, dim=1024, 16 heads):")

    # Extract real weights
    qkv_w_real = attn.qkv.weight.data.float().cpu().numpy()  # [3072, 1024]
    qkv_b_real = attn.qkv.bias.data.float().cpu().numpy() if attn.qkv.bias is not None else None
    proj_w_real = attn.proj.weight.data.float().cpu().numpy()  # [1024, 1024]
    proj_b_real = attn.proj.bias.data.float().cpu().numpy() if attn.proj.bias is not None else None

    print(f"  QKV weight: shape={qkv_w_real.shape}, |max|={np.abs(qkv_w_real).max():.4f}, "
          f"std={qkv_w_real.std():.5f}")
    print(f"  Proj weight: shape={proj_w_real.shape}, |max|={np.abs(proj_w_real).max():.4f}, "
          f"std={proj_w_real.std():.5f}")

    def make_single_attn_onnx(output_path, qkv_w, proj_w, qkv_b=None, proj_b=None):
        """Make ONNX: LN -> QKV -> split -> attention -> proj -> residual."""
        nodes = []
        initializers = []

        # Note: QKV weight in PyTorch Linear is [out, in], needs transpose for MatMul
        qkv_w_t = qkv_w.T  # [in, out] = [1024, 3072]
        proj_w_t = proj_w.T  # [in, out] = [1024, 1024]

        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, seq_len, dim])

        # LayerNorm
        initializers.append(helper.make_tensor("ln_w", TensorProto.FLOAT, [dim],
                                               np.ones(dim, dtype=np.float32)))
        initializers.append(helper.make_tensor("ln_b", TensorProto.FLOAT, [dim],
                                               np.zeros(dim, dtype=np.float32)))
        nodes.append(helper.make_node("LayerNormalization", ["X", "ln_w", "ln_b"], ["ln_out"],
                                      axis=-1, epsilon=1e-6))

        # QKV MatMul
        initializers.append(helper.make_tensor("W_qkv", TensorProto.FLOAT,
                                               list(qkv_w_t.shape), qkv_w_t.flatten()))
        qkv_name = "qkv_mm"
        nodes.append(helper.make_node("MatMul", ["ln_out", "W_qkv"], [qkv_name]))

        # Add bias if present
        if qkv_b is not None:
            initializers.append(helper.make_tensor("qkv_bias", TensorProto.FLOAT,
                                                   [3 * dim], qkv_b.flatten()))
            nodes.append(helper.make_node("Add", [qkv_name, "qkv_bias"], ["qkv_biased"]))
            qkv_name = "qkv_biased"

        # Reshape to [1, seq_len, 3, num_heads, head_dim]
        shape_val = np.array([1, seq_len, 3, num_heads, head_dim], dtype=np.int64)
        initializers.append(helper.make_tensor("shape_3hd", TensorProto.INT64, [5], shape_val))
        nodes.append(helper.make_node("Reshape", [qkv_name, "shape_3hd"], ["qkv_reshaped"]))

        # Transpose to [3, 1, num_heads, seq_len, head_dim]
        nodes.append(helper.make_node("Transpose", ["qkv_reshaped"], ["qkv_t"], perm=[2, 0, 3, 1, 4]))

        # Split
        split_sizes = np.array([1, 1, 1], dtype=np.int64)
        initializers.append(helper.make_tensor("split_sizes", TensorProto.INT64, [3], split_sizes))
        nodes.append(helper.make_node("Split", ["qkv_t", "split_sizes"], ["q", "k", "v"], axis=0))

        # Squeeze
        axes = np.array([0], dtype=np.int64)
        initializers.append(helper.make_tensor("axes", TensorProto.INT64, [1], axes))
        nodes.append(helper.make_node("Squeeze", ["q", "axes"], ["q_sq"]))
        nodes.append(helper.make_node("Squeeze", ["k", "axes"], ["k_sq"]))
        nodes.append(helper.make_node("Squeeze", ["v", "axes"], ["v_sq"]))

        # K transpose
        nodes.append(helper.make_node("Transpose", ["k_sq"], ["k_t"], perm=[0, 1, 3, 2]))

        # Q @ K^T
        nodes.append(helper.make_node("MatMul", ["q_sq", "k_t"], ["scores_raw"]))

        # Scale
        scale = np.array([head_dim ** -0.5], dtype=np.float32)
        initializers.append(helper.make_tensor("scale", TensorProto.FLOAT, [1], scale))
        nodes.append(helper.make_node("Mul", ["scores_raw", "scale"], ["scores"]))

        # Softmax
        nodes.append(helper.make_node("Softmax", ["scores"], ["attn_w"], axis=-1))

        # Attn @ V
        nodes.append(helper.make_node("MatMul", ["attn_w", "v_sq"], ["context"]))

        # Transpose back and reshape
        nodes.append(helper.make_node("Transpose", ["context"], ["context_t"], perm=[0, 2, 1, 3]))
        shape_flat = np.array([1, seq_len, dim], dtype=np.int64)
        initializers.append(helper.make_tensor("shape_flat", TensorProto.INT64, [3], shape_flat))
        nodes.append(helper.make_node("Reshape", ["context_t", "shape_flat"], ["context_flat"]))

        # Output projection
        initializers.append(helper.make_tensor("W_proj", TensorProto.FLOAT,
                                               list(proj_w_t.shape), proj_w_t.flatten()))
        proj_name = "proj_mm"
        nodes.append(helper.make_node("MatMul", ["context_flat", "W_proj"], [proj_name]))

        if proj_b is not None:
            initializers.append(helper.make_tensor("proj_bias", TensorProto.FLOAT,
                                                   [dim], proj_b.flatten()))
            nodes.append(helper.make_node("Add", [proj_name, "proj_bias"], ["proj_biased"]))
            proj_name = "proj_biased"

        # Residual
        nodes.append(helper.make_node("Add", ["X", proj_name], ["Y"]))

        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, seq_len, dim])
        graph = helper.make_graph(nodes, "real_attn", [X], [Y], initializers)
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        onnx.save(model, output_path)

    def run_trt_and_compare(onnx_path, X, label):
        """Build TRT FP16 and compare."""
        import tensorrt as trt

        builder = trt.Builder(trt.Logger(trt.Logger.WARNING))
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, trt.Logger(trt.Logger.WARNING))
        with open(onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                print(f"  Parse failed for {label}")
                return

        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)
        config.set_flag(trt.BuilderFlag.FP16)

        engine_bytes = builder.build_serialized_network(network, config)
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        engine = runtime.deserialize_cuda_engine(engine_bytes)
        context = engine.create_execution_context()

        input_name = engine.get_tensor_name(0)
        output_name = engine.get_tensor_name(engine.num_io_tensors - 1)

        d_input = X.float().contiguous().cuda()
        output_shape = context.get_tensor_shape(output_name)
        d_output = torch.empty(list(output_shape), dtype=torch.float32, device="cuda")

        context.set_tensor_address(input_name, d_input.data_ptr())
        context.set_tensor_address(output_name, d_output.data_ptr())

        stream = torch.cuda.current_stream().cuda_stream
        context.execute_async_v3(stream)
        torch.cuda.synchronize()

        # FP32 reference using PyTorch
        qkv_w = onnx.load(onnx_path)
        weights = {}
        for init in qkv_w.graph.initializer:
            arr = numpy_helper.to_array(init)
            if arr.dtype in (np.float32, np.float64):
                weights[init.name] = torch.from_numpy(arr).to(DEVICE)

        with torch.inference_mode():
            h = torch.nn.functional.layer_norm(X.float(), [dim], eps=1e-6)
            qkv = torch.matmul(h, weights["W_qkv"])
            if "qkv_bias" in weights:
                qkv = qkv + weights["qkv_bias"]

            B, S, _ = qkv.shape
            qkv = qkv.reshape(B, S, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]

            scores = torch.matmul(q, k.transpose(-2, -1)) * (head_dim ** -0.5)
            attn_weights = torch.softmax(scores, dim=-1)
            ctx = torch.matmul(attn_weights, v)
            ctx = ctx.transpose(1, 2).reshape(B, S, dim)
            out = torch.matmul(ctx, weights["W_proj"])
            if "proj_bias" in weights:
                out = out + weights["proj_bias"]
            Y_fp32 = X.float() + out

        cos = torch.nn.functional.cosine_similarity(
            Y_fp32.flatten().unsqueeze(0),
            d_output.flatten().unsqueeze(0),
        ).item()
        diff = (Y_fp32 - d_output).abs().max().item()

        print(f"  {label:<30s} | cos={cos:.6f} | max_diff={diff:.4f}")
        return cos

    X = torch.randn(1, seq_len, dim, device=DEVICE)

    # Test with real weights
    make_single_attn_onnx("test_real_weights.onnx", qkv_w_real, proj_w_real,
                          qkv_b_real, proj_b_real)
    run_trt_and_compare("test_real_weights.onnx", X, "Real SAM3 block 0 weights")

    # Test with random weights (same scale)
    np.random.seed(42)
    qkv_w_rand = np.random.randn(*qkv_w_real.shape).astype(np.float32) * qkv_w_real.std()
    proj_w_rand = np.random.randn(*proj_w_real.shape).astype(np.float32) * proj_w_real.std()
    make_single_attn_onnx("test_rand_weights.onnx", qkv_w_rand, proj_w_rand)
    run_trt_and_compare("test_rand_weights.onnx", X, "Random weights (same scale)")

    # Test with scaled-down real weights
    make_single_attn_onnx("test_scaled_weights.onnx", qkv_w_real * 0.5, proj_w_real * 0.5,
                          qkv_b_real * 0.5 if qkv_b_real is not None else None,
                          proj_b_real * 0.5 if proj_b_real is not None else None)
    run_trt_and_compare("test_scaled_weights.onnx", X, "Real weights * 0.5")

    # Test blocks from different depths
    print("\n  Testing real weights from different blocks:")
    for block_idx in [0, 7, 15, 23, 31]:
        block = trunk.blocks[block_idx]
        attn = block.attn
        qkv_w = attn.qkv.weight.data.float().cpu().numpy()
        proj_w = attn.proj.weight.data.float().cpu().numpy()
        qkv_b = attn.qkv.bias.data.float().cpu().numpy() if attn.qkv.bias is not None else None
        proj_b = attn.proj.bias.data.float().cpu().numpy() if attn.proj.bias is not None else None

        make_single_attn_onnx(f"test_block{block_idx}.onnx", qkv_w, proj_w, qkv_b, proj_b)
        run_trt_and_compare(f"test_block{block_idx}.onnx", X, f"Block {block_idx} real weights")
        Path(f"test_block{block_idx}.onnx").unlink(missing_ok=True)

    # Cleanup
    for f in ["test_real_weights.onnx", "test_rand_weights.onnx", "test_scaled_weights.onnx"]:
        Path(f).unlink(missing_ok=True)


if __name__ == "__main__":
    print("Loading model...")
    model = build_sam3_image_model(
        device=DEVICE, checkpoint_path="sam3.pt", eval_mode=True,
    )

    print("\n" + "=" * 80)
    print("PART 1: Weight statistics")
    print("=" * 80)
    profile_weight_stats(model)

    print("\n" + "=" * 80)
    print("PART 2: Attention weight properties")
    print("=" * 80)
    profile_attention_weight_properties(model)

    print("\n" + "=" * 80)
    print("PART 3: Weight condition numbers")
    print("=" * 80)
    test_weight_condition_number(model)

    print("\n" + "=" * 80)
    print("PART 4: QKV FP16 sensitivity (block-by-block)")
    print("=" * 80)
    test_qkv_fp16_sensitivity(model)

    print("\n" + "=" * 80)
    print("PART 5: Real vs random weights in TRT FP16")
    print("=" * 80)
    test_real_vs_random_weights()

    print("\nDone!")
