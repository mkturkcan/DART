#!/usr/bin/env python3
"""Bisect which ViT blocks need FP32 attention for accurate output.

Strategy: Binary search on blocks. Force attention MatMul in blocks [0..N]
to FP32 and blocks [N+1..31] to FP16. Find the minimal N that gives
cos > 0.99.

This tells us whether the error accumulation is uniform across blocks
or concentrated in specific blocks (early vs late).

Usage:
    python scripts/bisect_blocks_fp32.py \
        --onnx backbone.onnx \
        --checkpoint sam3.pt \
        --image x.jpg
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import tensorrt as trt


def cosine_similarity(a, b):
    a_flat = a.float().flatten()
    b_flat = b.float().flatten()
    return torch.nn.functional.cosine_similarity(
        a_flat.unsqueeze(0), b_flat.unsqueeze(0)
    ).item()


def build_engine_with_block_range(onnx_path, fp32_blocks, engine_path, workspace_gb=4.0):
    """Build engine with specific blocks' attention in FP32."""
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            raise RuntimeError("Failed to parse ONNX")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE, int(workspace_gb * (1 << 30))
    )
    if hasattr(config, "builder_optimization_level"):
        config.builder_optimization_level = 3

    config.set_flag(trt.BuilderFlag.FP16)
    config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)

    # Skip types
    skip_types = set()
    for type_name in (
        "SHAPE", "CONSTANT", "IDENTITY", "SHUFFLE", "GATHER",
        "SLICE", "SQUEEZE", "UNSQUEEZE", "CONCATENATION", "CONDITION",
        "CAST", "ASSERTION", "FILL", "SCATTER", "RESIZE",
        "NON_ZERO", "ONE_HOT", "GRID_SAMPLE",
        "CONDITIONAL_INPUT", "CONDITIONAL_OUTPUT",
    ):
        if hasattr(trt.LayerType, type_name):
            skip_types.add(getattr(trt.LayerType, type_name))

    softmax_type = getattr(trt.LayerType, "SOFTMAX", None)
    matmul_type = getattr(trt.LayerType, "MATRIX_MULTIPLY", None)
    norm_type = getattr(trt.LayerType, "NORMALIZATION", None)

    # Build set of block indices that should have FP32 attention
    fp32_block_set = set(fp32_blocks)

    fp32_count = 0
    for i in range(network.num_layers):
        layer = network.get_layer(i)
        if layer.type in skip_types:
            continue

        force_fp32 = False
        name = layer.name

        # Always force all LayerNorm and Softmax to FP32 (baseline)
        if layer.type == norm_type or layer.type == softmax_type:
            force_fp32 = True
        elif layer.type == matmul_type:
            # Check if this MatMul is attention (not MLP) in a target block
            for block_idx in fp32_block_set:
                block_str = f"blocks.{block_idx}/"
                if block_str in name and "/attn/" in name:
                    # Only Q@K^T and attn@V (not qkv or proj)
                    if "/qkv/" not in name and "/proj/" not in name:
                        force_fp32 = True
                    break

        if force_fp32:
            layer.precision = trt.float32
            for j in range(layer.num_outputs):
                layer.set_output_type(j, trt.float32)
            fp32_count += 1

    print(f"  FP32 layers: {fp32_count} (blocks {fp32_blocks})")

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("Engine build failed")

    with open(engine_path, "wb") as f:
        f.write(serialized)

    return engine_path


def test_engine(engine_path, img_tensor, ref_dict, n_warmup=5, n_runs=20):
    """Test engine accuracy and speed."""
    from sam3.trt.trt_backbone import TRTBackbone

    trt_bb = TRTBackbone(engine_path, device="cuda")

    with torch.inference_mode():
        for _ in range(n_warmup):
            trt_bb.forward_image(img_tensor)
    torch.cuda.synchronize()

    times = []
    with torch.inference_mode():
        for _ in range(n_runs):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = trt_bb.forward_image(img_tensor)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)

    trt_fpn = out["backbone_fpn"]
    cos_scores = {}
    for i, (k, ref) in enumerate(ref_dict.items()):
        cos_scores[k] = cosine_similarity(ref, trt_fpn[i])

    del trt_bb
    torch.cuda.empty_cache()

    return {
        "avg_ms": np.mean(times),
        "min_ms": np.min(times),
        "cosines": cos_scores,
        "cos_min": min(cos_scores.values()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    args = parser.parse_args()

    device = "cuda"
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    # PyTorch reference
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
    del model, predictor
    torch.cuda.empty_cache()

    # Test configurations
    configs = [
        ("no-blocks (norm+softmax only)", []),
        ("blocks 0-7", list(range(0, 8))),
        ("blocks 0-15", list(range(0, 16))),
        ("blocks 0-23", list(range(0, 24))),
        ("blocks 0-31 (all)", list(range(0, 32))),
        ("blocks 24-31 (last 8)", list(range(24, 32))),
        ("blocks 16-31 (last 16)", list(range(16, 32))),
        ("blocks 8-31 (last 24)", list(range(8, 32))),
    ]

    results = []
    for label, blocks in configs:
        engine_path = f"backbone_bisect_{label.replace(' ', '_').replace('(', '').replace(')', '')}.engine"
        print(f"\n{'='*60}")
        print(f"Config: {label}")
        print(f"{'='*60}")
        try:
            build_engine_with_block_range(args.onnx, blocks, engine_path)
            result = test_engine(engine_path, img_tensor, ref_dict)
            result["label"] = label
            result["blocks"] = blocks
            results.append(result)
            print(f"  Speed: {result['avg_ms']:.1f}ms avg, {result['min_ms']:.1f}ms min")
            print(f"  Cosine: {result['cos_min']:.6f} (min across FPN)")
            for k, cos in result["cosines"].items():
                status = "OK" if cos > 0.999 else "WARN" if cos > 0.99 else "BAD"
                print(f"    {k}: {cos:.6f} [{status}]")
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

        # Clean up engine to save disk
        if os.path.exists(engine_path):
            os.remove(engine_path)

    # Summary
    print(f"\n\n{'='*80}")
    print("BLOCK BISECTION RESULTS")
    print(f"{'='*80}")
    print(f"{'Config':<35s} {'Avg ms':>8s} {'Min ms':>8s} {'Cos min':>10s} {'Status':>8s}")
    print(f"{'-'*35} {'-'*8} {'-'*8} {'-'*10} {'-'*8}")
    for r in results:
        status = "OK" if r["cos_min"] > 0.999 else "WARN" if r["cos_min"] > 0.99 else "BAD"
        print(
            f"{r['label']:<35s} {r['avg_ms']:>8.1f} {r['min_ms']:>8.1f} "
            f"{r['cos_min']:>10.6f} {status:>8s}"
        )


if __name__ == "__main__":
    main()
