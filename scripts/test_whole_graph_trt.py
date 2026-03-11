#!/usr/bin/env python3
"""Test whole-graph TRT compilation approach from SAM3-TensorRT.

Exports the entire HuggingFace SAM3 model as a single ONNX graph with
baked-in text prompt, builds a TRT FP16 engine, and validates quality
against the PyTorch reference (cosine similarity).

Based on: https://github.com/dataplayer12/SAM3-TensorRT

Usage:
    python scripts/test_whole_graph_trt.py --prompt "dog" --image x.jpg
"""

import argparse
import time
from pathlib import Path

import torch
import numpy as np


def export_onnx(prompt: str, output_dir: str = "onnx_whole"):
    """Export whole SAM3 model to ONNX with baked-in prompt."""
    from transformers.models.sam3 import Sam3Processor, Sam3Model
    from PIL import Image

    print("Loading HuggingFace SAM3 model (eager attention for ONNX compat)...")
    # Export on CPU for compat; use eager attention — SDPA segfaults during
    # torch.onnx.export tracing in newer transformers versions
    device = "cpu"
    model = Sam3Model.from_pretrained(
        "facebook/sam3", attn_implementation="eager"
    ).to(device)
    processor = Sam3Processor.from_pretrained("facebook/sam3")
    model.eval()

    # Build sample input to trace
    # Use a dummy image — the spatial size matters, not content
    dummy_image = Image.new("RGB", (1008, 1008), color=(128, 128, 128))
    inputs = processor(images=dummy_image, text=prompt, return_tensors="pt").to(device)

    pixel_values = inputs["pixel_values"]
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    print(f"  pixel_values: {pixel_values.shape}")
    print(f"  input_ids: {input_ids.shape} = {input_ids}")
    print(f"  attention_mask: {attention_mask.shape}")

    # Wrapper that bakes in text tokens as constants
    class Sam3ONNXWrapper(torch.nn.Module):
        def __init__(self, sam3, input_ids, attention_mask):
            super().__init__()
            self.sam3 = sam3
            self.register_buffer("const_input_ids", input_ids.to(torch.int64))
            self.register_buffer("const_attention_mask", attention_mask.to(torch.int64))

        def forward(self, pixel_values):
            outputs = self.sam3(
                pixel_values=pixel_values,
                input_ids=self.const_input_ids,
                attention_mask=self.const_attention_mask,
            )
            return outputs.pred_masks, outputs.semantic_seg

    wrapper = Sam3ONNXWrapper(model, input_ids, attention_mask).eval()

    # Export
    out_path = Path(output_dir)
    out_path.mkdir(exist_ok=True)
    onnx_path = str(out_path / "sam3_whole.onnx")

    print(f"Exporting ONNX to {onnx_path} ...")
    t0 = time.perf_counter()
    # Use dynamo-based export — the TorchScript tracer segfaults on Windows
    # with transformers 5.x due to attention dispatch changes
    export_output = torch.onnx.export(
        wrapper,
        (pixel_values,),
        dynamo=True,
    )
    export_output.save(onnx_path)
    print(f"  Export done ({time.perf_counter() - t0:.1f}s)")

    # Check file size
    import glob
    total_size = 0
    for f in glob.glob(str(out_path / "*")):
        total_size += Path(f).stat().st_size
    print(f"  Total ONNX size: {total_size / 1e6:.0f} MB")

    return onnx_path


def build_engine(onnx_path: str, output_path: str):
    """Build TRT FP16 engine from ONNX using Python API."""
    import tensorrt as trt

    TRT_LOGGER = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, TRT_LOGGER)

    print(f"Parsing ONNX: {onnx_path}")
    # Use parse_from_file so TRT resolves external data files relative to
    # the ONNX file path (dynamo export creates .onnx.data alongside .onnx)
    onnx_abs = str(Path(onnx_path).resolve())
    if hasattr(parser, "parse_from_file"):
        if not parser.parse_from_file(onnx_abs):
            for i in range(parser.num_errors):
                print(f"  Error: {parser.get_error(i)}")
            raise RuntimeError("ONNX parse failed")
    else:
        with open(onnx_abs, "rb") as f:
            if not parser.parse(f.read()):
                for i in range(parser.num_errors):
                    print(f"  Error: {parser.get_error(i)}")
                raise RuntimeError("ONNX parse failed")

    # Print network info
    print(f"  Layers: {network.num_layers}")
    print(f"  Inputs: {network.num_inputs}")
    for i in range(network.num_inputs):
        inp = network.get_input(i)
        print(f"    {inp.name}: {inp.shape}")
    print(f"  Outputs: {network.num_outputs}")
    for i in range(network.num_outputs):
        out = network.get_output(i)
        print(f"    {out.name}: {out.shape}")

    # Build config — pure FP16, no mixed precision (matching trtexec --fp16)
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
    config.set_flag(trt.BuilderFlag.FP16)
    if hasattr(config, "builder_optimization_level"):
        config.builder_optimization_level = 3
    print("Building TRT engine (FP16, no precision constraints)...")

    t0 = time.perf_counter()
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("Engine build failed")

    with open(output_path, "wb") as f:
        f.write(serialized)
    size_mb = Path(output_path).stat().st_size / 1e6
    print(f"  Done ({time.perf_counter() - t0:.0f}s), {size_mb:.0f} MB -> {output_path}")
    return output_path


def run_pytorch_reference(image_path: str, prompt: str):
    """Run HF SAM3 in PyTorch and return outputs for comparison."""
    from transformers.models.sam3 import Sam3Processor, Sam3Model
    from PIL import Image

    print("Running PyTorch reference...")
    model = Sam3Model.from_pretrained("facebook/sam3").to("cuda")
    processor = Sam3Processor.from_pretrained("facebook/sam3")
    model.eval()

    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, text=prompt, return_tensors="pt").to("cuda")

    with torch.inference_mode():
        outputs = model(**inputs)

    pred_masks = outputs.pred_masks.clone()
    semantic_seg = outputs.semantic_seg.clone()

    print(f"  pred_masks: {pred_masks.shape}")
    print(f"  semantic_seg: {semantic_seg.shape}")

    # Also return pixel_values for TRT input
    pixel_values = inputs["pixel_values"].clone()

    del model
    torch.cuda.empty_cache()

    return pixel_values, pred_masks, semantic_seg


def run_trt_engine(engine_path: str, pixel_values: torch.Tensor):
    """Run TRT engine and return outputs."""
    import tensorrt as trt

    print(f"Loading TRT engine: {engine_path}")
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    with open(engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()

    # Map dtypes
    _trt_to_torch = {
        trt.float32: torch.float32,
        trt.float16: torch.float16,
        trt.int32: torch.int32,
    }

    # Allocate I/O buffers based on engine bindings
    io_names = []
    io_bufs = {}
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        shape = engine.get_tensor_shape(name)
        dtype = _trt_to_torch.get(engine.get_tensor_dtype(name), torch.float32)
        mode = engine.get_tensor_mode(name)
        is_input = (mode == trt.TensorIOMode.INPUT)
        io_names.append(name)
        print(f"  {'INPUT' if is_input else 'OUTPUT'}: {name} {list(shape)} {dtype}")

        if is_input:
            # Copy input to correct dtype
            buf = pixel_values.to(dtype=dtype, device="cuda").contiguous()
        else:
            buf = torch.empty(list(shape), dtype=dtype, device="cuda")

        io_bufs[name] = buf
        context.set_tensor_address(name, buf.data_ptr())

    # Execute
    stream = torch.cuda.Stream()
    context.execute_async_v3(stream.cuda_stream)
    stream.synchronize()

    # Get outputs (convert to float32)
    output_names = [n for n in io_names if engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]
    outputs = {name: io_bufs[name].float() for name in output_names}

    # Benchmark
    for _ in range(5):
        context.execute_async_v3(stream.cuda_stream)
    stream.synchronize()
    times = []
    for _ in range(20):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        context.execute_async_v3(stream.cuda_stream)
        stream.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    median_ms = times[len(times) // 2]
    print(f"  TRT median: {median_ms:.1f}ms")

    del context, engine
    return outputs, median_ms


def cosine(a, b):
    return torch.nn.functional.cosine_similarity(
        a.float().flatten().unsqueeze(0),
        b.float().flatten().unsqueeze(0),
    ).item()


def main():
    parser = argparse.ArgumentParser(description="Test whole-graph SAM3 TRT")
    parser.add_argument("--prompt", default="dog", help="Text prompt to bake in")
    parser.add_argument("--image", default="x.jpg", help="Test image")
    parser.add_argument("--skip-export", action="store_true",
                        help="Skip ONNX export (reuse existing)")
    parser.add_argument("--skip-build", action="store_true",
                        help="Skip engine build (reuse existing)")
    parser.add_argument("--onnx-dir", default="onnx_whole",
                        help="ONNX output directory")
    parser.add_argument("--engine", default="sam3_whole_fp16.engine",
                        help="TRT engine path")
    args = parser.parse_args()

    onnx_path = str(Path(args.onnx_dir) / "sam3_whole.onnx")

    # Step 1: Export ONNX
    if not args.skip_export:
        onnx_path = export_onnx(args.prompt, args.onnx_dir)
    else:
        print(f"Skipping export, using: {onnx_path}")

    # Step 2: Build TRT engine
    if not args.skip_build:
        build_engine(onnx_path, args.engine)
    else:
        print(f"Skipping build, using: {args.engine}")

    # Step 3: PyTorch reference
    pixel_values, ref_masks, ref_seg = run_pytorch_reference(args.image, args.prompt)

    # Step 4: TRT inference
    trt_outputs, trt_ms = run_trt_engine(args.engine, pixel_values)

    # Step 5: Compare — match TRT outputs to reference by shape (dynamo
    # export renames outputs to op names like "einsum", "conv2d_12")
    print(f"\n{'='*60}")
    print("QUALITY COMPARISON")
    print(f"{'='*60}")
    ref_pairs = [("pred_masks", ref_masks), ("semantic_seg", ref_seg)]
    for name, ref in ref_pairs:
        # Try exact name match first, then match by shape
        trt_out = trt_outputs.get(name)
        matched_name = name
        if trt_out is None:
            for trt_name, trt_val in trt_outputs.items():
                if list(trt_val.shape) == list(ref.shape):
                    trt_out = trt_val
                    matched_name = f"{name} (TRT: {trt_name})"
                    break
        if trt_out is not None:
            cos = cosine(ref, trt_out)
            max_diff = (ref - trt_out).abs().max().item()
            status = "OK" if cos > 0.99 else "BROKEN" if cos < 0.5 else "DEGRADED"
            print(f"  {matched_name}:")
            print(f"    Shape: ref={list(ref.shape)}, trt={list(trt_out.shape)}")
            print(f"    Cosine: {cos:.6f}  MaxDiff: {max_diff:.4f}  {status}")
        else:
            print(f"  {name}: not found in TRT outputs")
            print(f"  Available: {list(trt_outputs.keys())}")

    print(f"\n  TRT whole-graph FP16: {trt_ms:.1f}ms")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
