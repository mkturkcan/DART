#!/usr/bin/env python3
"""Test if forcing LayerNorm to FP32 fixes TRT FP16 accuracy.

Hypothesis: TRT's FP16 LayerNorm accumulates in FP16 (not FP32),
causing precision loss that compounds through 32 ViT blocks.
If forcing just NORMALIZATION layers to FP32 fixes accuracy while
keeping MATRIX_MULTIPLY in FP16, we get near-26ms speed!
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


def build_with_fp32_types(onnx_path, output_path, fp32_type_names, label):
    """Build FP16 engine with specific layer types forced to FP32."""
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
    config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)

    fp32_types = set()
    for name in fp32_type_names:
        if hasattr(trt.LayerType, name):
            fp32_types.add(getattr(trt.LayerType, name))
        else:
            print(f"  WARNING: LayerType.{name} not found")

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
    skip_count = 0
    type_counts = {}
    for i in range(network.num_layers):
        layer = network.get_layer(i)
        if layer.type in skip_types:
            skip_count += 1
            continue
        if layer.type in fp32_types:
            layer.precision = trt.float32
            for j in range(layer.num_outputs):
                layer.set_output_type(j, trt.float32)
            fp32_count += 1
            tn = str(layer.type).split(".")[-1]
            type_counts[tn] = type_counts.get(tn, 0) + 1
        else:
            fp16_count += 1

    type_str = " + ".join(f"{v} {k}" for k, v in type_counts.items())
    print(f"  {label}: {fp32_count} FP32 ({type_str}) / {fp16_count} FP16 / {skip_count} skip")

    t0 = time.time()
    engine_bytes = builder.build_serialized_network(network, config)
    build_s = time.time() - t0
    if engine_bytes is None:
        raise RuntimeError("Build failed")
    with open(output_path, "wb") as f:
        f.write(engine_bytes)
    print(f"  Built in {build_s:.0f}s, size={Path(output_path).stat().st_size/1e6:.0f} MB")
    return output_path


def test_engine(engine_path, label, model, dummy):
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

    cos_last = torch.nn.functional.cosine_similarity(
        pt_fpn[-1].float().flatten().unsqueeze(0),
        trt_fpn[-1].float().flatten().unsqueeze(0),
    ).item()

    with torch.inference_mode():
        for _ in range(10):
            trt_bb.forward_image(dummy)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(50):
            trt_bb.forward_image(dummy)
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) / 50 * 1000

    status = "OK" if cos_last > 0.99 else ("MARGINAL" if cos_last > 0.9 else "BROKEN")
    print(f"  {label:45s} | cos={cos_last:.4f} | {ms:6.1f}ms | {status}")

    del trt_bb
    torch.cuda.empty_cache()


def main():
    from sam3.model_builder import build_sam3_image_model

    print("Loading model...")
    model = build_sam3_image_model(
        device=DEVICE, checkpoint_path="sam3.pt", eval_mode=True,
    )
    dummy = torch.randn(1, 3, 1008, 1008, device=DEVICE)

    print(f"\n{'Config':47s} | {'Cosine':>8s} | {'Speed':>7s} | Status")
    print("-" * 85)

    # Test 1: Just NORMALIZATION (LayerNorm) FP32
    tests = [
        (["NORMALIZATION"],                                    "norm_only"),
        (["NORMALIZATION", "SOFTMAX"],                         "norm+softmax"),
        (["NORMALIZATION", "SOFTMAX", "REDUCE"],               "norm+softmax+reduce"),
        (["NORMALIZATION", "SOFTMAX", "ELEMENTWISE"],          "norm+softmax+elementwise"),
        (["NORMALIZATION", "SOFTMAX", "MATRIX_MULTIPLY"],      "norm+softmax+matmul (prev best)"),
    ]

    for fp32_types, label in tests:
        engine_path = f"backbone_test_{label.replace(' ', '_').replace('(', '').replace(')', '')}.engine"
        try:
            build_with_fp32_types(ONNX_PATH, engine_path, fp32_types, label)
            test_engine(engine_path, label, model, dummy)
        except Exception as e:
            print(f"  {label:45s} | FAILED: {e}")
        finally:
            Path(engine_path).unlink(missing_ok=True)

    print("\nDone!")


if __name__ == "__main__":
    main()
