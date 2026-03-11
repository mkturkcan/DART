#!/usr/bin/env python3
"""Binary search: which MatMul subsets need FP32 to maintain accuracy?

MatMul groups in ViT-H (32 blocks):
  - qkv:  /trunk/blocks.X/attn/qkv/MatMul   (32 layers) — QKV projection
  - attn:  /trunk/blocks.X/attn/MatMul[_1]   (64 layers) — Q@K^T and attn@V
  - proj: /trunk/blocks.X/attn/proj/MatMul   (32 layers) — output projection
  - mlp:  /trunk/blocks.X/mlp/fc[12]/MatMul  (64 layers) — MLP
  - softmax: all Softmax layers              (32 layers)
  - other: FPN neck and other MatMuls

Tests each subset individually and in combinations.
"""

import re
import sys
import time
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
ONNX_PATH = "backbone.onnx"
DEVICE = "cuda"


# Name patterns for MatMul groups
PATTERNS = {
    "qkv":     re.compile(r"/attn/qkv/MatMul$"),
    "attn_qk": re.compile(r"/attn/MatMul$"),       # Q@K^T only
    "attn_v":  re.compile(r"/attn/MatMul_1$"),      # attn@V only
    "proj":    re.compile(r"/attn/proj/MatMul$"),
    "mlp_fc1": re.compile(r"/mlp/fc1/MatMul$"),
    "mlp_fc2": re.compile(r"/mlp/fc2/MatMul$"),
}


def build_and_test(fp32_groups, label):
    """Build engine with specific MatMul groups in FP32, test accuracy."""
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, TRT_LOGGER)
    with open(ONNX_PATH, "rb") as f:
        if not parser.parse(f.read()):
            raise RuntimeError("ONNX parse failed")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
    config.set_flag(trt.BuilderFlag.FP16)
    config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)

    softmax_type = getattr(trt.LayerType, "SOFTMAX", None)
    matmul_type = getattr(trt.LayerType, "MATRIX_MULTIPLY", None)

    skip_types = set()
    for name in (
        "SHAPE", "CONSTANT", "IDENTITY", "SHUFFLE", "GATHER",
        "SLICE", "SQUEEZE", "UNSQUEEZE", "CONCATENATION", "CONDITION",
        "CAST", "ASSERTION", "FILL", "SCATTER", "RESIZE",
        "NON_ZERO", "ONE_HOT", "GRID_SAMPLE",
    ):
        if hasattr(trt.LayerType, name):
            skip_types.add(getattr(trt.LayerType, name))

    fp32_count = 0
    fp16_count = 0
    for i in range(network.num_layers):
        layer = network.get_layer(i)
        if layer.type in skip_types:
            continue

        force_fp32 = False

        # Always force Softmax to FP32 if "softmax" in groups
        if layer.type == softmax_type and "softmax" in fp32_groups:
            force_fp32 = True

        # Check MatMul groups
        if layer.type == matmul_type:
            if "all_matmul" in fp32_groups:
                force_fp32 = True
            else:
                for group_name in fp32_groups:
                    if group_name in PATTERNS and PATTERNS[group_name].search(layer.name):
                        force_fp32 = True
                        break

        if force_fp32:
            layer.precision = trt.float32
            for j in range(layer.num_outputs):
                layer.set_output_type(j, trt.float32)
            fp32_count += 1
        else:
            fp16_count += 1

    engine_path = f"backbone_test_{label}.engine"
    t0 = time.time()
    engine_bytes = builder.build_serialized_network(network, config)
    build_time = time.time() - t0

    if engine_bytes is None:
        print(f"  {label}: BUILD FAILED")
        return None

    with open(engine_path, "wb") as f:
        f.write(engine_bytes)

    # Load and test
    from sam3.model_builder import build_sam3_image_model
    from sam3.trt.trt_backbone import TRTBackbone

    model = build_sam3_image_model(
        device=DEVICE, checkpoint_path="sam3.pt", eval_mode=True,
    )
    backbone = model.backbone
    dummy = torch.randn(1, 3, 1008, 1008, device=DEVICE)

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

    cos_last = torch.nn.functional.cosine_similarity(
        pt_fpn[-1].float().flatten().unsqueeze(0),
        trt_fpn[-1].float().flatten().unsqueeze(0),
    ).item()

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

    status = "OK" if cos_last > 0.99 else "BROKEN"
    print(f"  {label:40s} | fp32={fp32_count:3d} | cos={cos_last:.4f} | {ms:.1f}ms | {status} | build={build_time:.0f}s")

    del trt_bb, model, backbone
    torch.cuda.empty_cache()

    # Cleanup engine file
    Path(engine_path).unlink(missing_ok=True)
    return cos_last, ms


def main():
    print(f"{'Config':42s} | {'FP32':>5s} | {'Cosine':>8s} | {'Speed':>6s} | Status | Build")
    print("-" * 100)

    # Test individual groups
    tests = [
        # Individual groups (with softmax always FP32)
        (["softmax"],                                           "softmax_only"),
        (["softmax", "attn_qk"],                                "softmax+attn_qk"),
        (["softmax", "attn_qk", "attn_v"],                     "softmax+attn"),
        (["softmax", "attn_qk", "attn_v", "qkv"],              "softmax+attn+qkv"),
        (["softmax", "attn_qk", "attn_v", "qkv", "proj"],      "softmax+attn+qkv+proj"),
        (["softmax", "mlp_fc1", "mlp_fc2"],                     "softmax+mlp"),
        (["softmax", "attn_qk", "attn_v", "mlp_fc1", "mlp_fc2"], "softmax+attn+mlp"),
        (["softmax", "all_matmul"],                              "softmax+all_matmul"),
    ]

    for groups, label in tests:
        try:
            build_and_test(groups, label)
        except Exception as e:
            print(f"  {label:40s} | FAILED: {e}")


if __name__ == "__main__":
    main()
