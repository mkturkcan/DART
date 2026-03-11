#!/usr/bin/env python3
"""Test HuggingFace SAM3 backbone export and TRT FP16 accuracy.

Exports the HF Sam3VisionModel to ONNX, builds a TRT FP16 engine, and
compares the output against PyTorch to determine if the HF implementation
avoids the TRT FP16 numerical bug seen with the Meta SAM3 codebase.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


class HFBackboneWrapper(nn.Module):
    """Wraps HF Sam3VisionModel to return flat FPN tuple."""

    def __init__(self, vision_model):
        super().__init__()
        self.vision_model = vision_model

    def forward(self, pixel_values: torch.Tensor):
        outputs = self.vision_model(pixel_values)
        # HF Sam3VisionModel returns Sam3VisionEncoderOutput
        if hasattr(outputs, 'multi_scale_features'):
            fpn = outputs.multi_scale_features
        elif hasattr(outputs, 'fpn_hidden_states'):
            fpn = outputs.fpn_hidden_states
        else:
            raise RuntimeError(f"Unknown output attrs: {list(outputs.keys())}")
        return fpn[0], fpn[1], fpn[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-id",
        default="facebook/sam3-hiera-large",
        help="HuggingFace model ID",
    )
    parser.add_argument("--imgsz", type=int, default=1008)
    parser.add_argument("--skip-build", action="store_true")
    args = parser.parse_args()

    device = "cuda"

    # Step 1: Create HF vision model from config (random weights - we only
    # need the architecture to test if TRT FP16 works)
    print(f"Creating HF vision model from config: {args.model_id}")
    from transformers import Sam3Config
    from transformers import Sam3VisionModel

    config = Sam3Config.from_pretrained(args.model_id)
    vc = config.vision_config
    print(f"  Vision backbone: hidden_size={vc.backbone_config.hidden_size}, "
          f"layers={vc.backbone_config.num_hidden_layers}, "
          f"heads={vc.backbone_config.num_attention_heads}, "
          f"fpn_hidden={vc.fpn_hidden_size}")

    # Force eager attention (no SDPA) for clean ONNX tracing
    vc.backbone_config._attn_implementation = "eager"

    # Instantiate vision model with random weights (no download)
    print("  Instantiating Sam3VisionModel with random weights (eager attn)...")
    vision_model = Sam3VisionModel(vc).to(device).eval()
    print(f"  Vision model params: {sum(p.numel() for p in vision_model.parameters()) / 1e6:.1f}M")

    # Check what the vision model outputs
    print("\n--- Testing HF vision model ---")
    dummy = torch.randn(1, 3, args.imgsz, args.imgsz, device=device)
    with torch.no_grad():
        pt_out = vision_model(dummy)

    print(f"  Output type: {type(pt_out)}")
    # Try different attribute names across HF versions
    if hasattr(pt_out, 'multi_scale_features'):
        fpn = pt_out.multi_scale_features
    elif hasattr(pt_out, 'fpn_hidden_states'):
        fpn = pt_out.fpn_hidden_states
    else:
        print(f"  Available attrs: {[a for a in dir(pt_out) if not a.startswith('_')]}")
        print("Cannot proceed -- unknown output format")
        return

    print(f"  FPN levels: {len(fpn)}")
    for i, t in enumerate(fpn):
        print(f"    FPN[{i}]: shape={t.shape}, dtype={t.dtype}, "
              f"range=[{t.min().item():.4f}, {t.max().item():.4f}], "
              f"std={t.std().item():.4f}")

    # Step 2: Wrap and export to ONNX
    print("\n--- Exporting to ONNX ---")
    wrapper = HFBackboneWrapper(vision_model).to(device).eval()

    # Verify wrapper works
    with torch.no_grad():
        w_out = wrapper(dummy)
    print(f"  Wrapper outputs: {len(w_out)} tensors")
    for i, t in enumerate(w_out):
        print(f"    [{i}]: shape={t.shape}")

    onnx_path = "backbone_hf.onnx"
    output_names = ["fpn_0", "fpn_1", "fpn_2"]

    print(f"  Exporting to {onnx_path} ...")
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy,),
            onnx_path,
            opset_version=17,
            input_names=["images"],
            output_names=output_names,
            dynamic_axes=None,
        )

    size_mb = Path(onnx_path).stat().st_size / (1024 * 1024)
    print(f"  Exported: {onnx_path} ({size_mb:.1f} MB)")

    # Print ONNX summary
    try:
        import onnx
        from collections import Counter
        model_onnx = onnx.load(onnx_path)
        graph = model_onnx.graph
        op_counts = Counter(n.op_type for n in graph.node)
        print(f"  Nodes: {len(graph.node)}")
        print(f"  Top ops: {dict(op_counts.most_common(10))}")

        # Check for complex/problematic ops
        for op in ["If", "Loop", "Where"]:
            if op in op_counts:
                print(f"  WARNING: Found {op_counts[op]} {op} nodes")
    except Exception as e:
        print(f"  ONNX analysis failed: {e}")

    # Skip ONNX Runtime validation (causes segfault/OOM on large models)
    # Go directly to TRT build

    if args.skip_build:
        print("\nSkipping TRT engine build (--skip-build)")
        return

    # Step 4: Build TRT FP16 engine
    print("\n--- Building TRT FP16 engine ---")
    import tensorrt as trt

    engine_path = "backbone_hf_fp16.engine"
    TRT_LOGGER = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser_trt = trt.OnnxParser(network, TRT_LOGGER)

    with open(onnx_path, "rb") as f:
        if not parser_trt.parse(f.read()):
            for i in range(parser_trt.num_errors):
                print(f"  Parse error: {parser_trt.get_error(i)}")
            return

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
    config.set_flag(trt.BuilderFlag.FP16)
    if hasattr(config, "builder_optimization_level"):
        config.builder_optimization_level = 3

    print("  Building engine (this may take several minutes) ...")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        print("  Engine build FAILED")
        return

    with open(engine_path, "wb") as f:
        f.write(serialized)
    print(f"  Engine saved: {engine_path}")

    # Step 5: Run TRT engine and compare
    print("\n--- TRT FP16 vs PyTorch comparison ---")
    runtime = trt.Runtime(TRT_LOGGER)
    engine = runtime.deserialize_cuda_engine(serialized)
    context = engine.create_execution_context()

    # Allocate buffers
    input_buf = dummy.contiguous()
    output_bufs = []
    for i in range(3):
        name = output_names[i]
        shape = engine.get_tensor_shape(name)
        dtype_trt = engine.get_tensor_dtype(name)
        torch_dtype = torch.float32 if dtype_trt == trt.float32 else torch.float16
        buf = torch.empty(*shape, dtype=torch_dtype, device=device)
        output_bufs.append(buf)

    context.set_tensor_address("images", input_buf.data_ptr())
    for i, name in enumerate(output_names):
        context.set_tensor_address(name, output_bufs[i].data_ptr())

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        context.execute_async_v3(stream.cuda_stream)
    stream.synchronize()

    for i in range(3):
        trt_f = output_bufs[i].float()
        pt_f = fpn[i].float()
        diff = (pt_f - trt_f).abs()
        cos = torch.nn.functional.cosine_similarity(
            pt_f.flatten().unsqueeze(0),
            trt_f.flatten().unsqueeze(0),
        )
        print(f"  FPN[{i}]: cosine={cos.item():.6f}, "
              f"max_diff={diff.max().item():.4f}, "
              f"pt_std={pt_f.std().item():.4f}, trt_std={trt_f.std().item():.4f}")

    # Final verdict
    cos_last = torch.nn.functional.cosine_similarity(
        fpn[-1].float().flatten().unsqueeze(0),
        output_bufs[-1].float().flatten().unsqueeze(0),
    )
    print(f"\n  FINAL COSINE (FPN[-1]): {cos_last.item():.6f}")
    if cos_last.item() > 0.99:
        print("  RESULT: HF backbone FP16 TRT WORKS!")
    else:
        print("  RESULT: HF backbone FP16 TRT also broken")


if __name__ == "__main__":
    main()
