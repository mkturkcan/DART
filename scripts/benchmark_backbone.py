#!/usr/bin/env python3
"""Benchmark SAM3 backbone under different execution backends.

Isolates where time is spent: PyTorch vs ONNX Runtime vs TensorRT,
and within TRT: raw engine vs our Python wrapper.

Usage:
    # PyTorch only (no extra deps needed)
    python scripts/benchmark_backbone.py --checkpoint sam3.pt

    # + TensorRT engine
    python scripts/benchmark_backbone.py --checkpoint sam3.pt --trt backbone_int8.engine

    # + ONNX model
    python scripts/benchmark_backbone.py --checkpoint sam3.pt --trt backbone_int8.engine --onnx backbone.onnx

    # Adjust runs
    python scripts/benchmark_backbone.py --checkpoint sam3.pt --trt backbone_int8.engine --warmup 5 --runs 20
"""

import argparse
import time

import torch
import numpy as np


def benchmark(fn, warmup=5, runs=20, label=""):
    """Run fn repeatedly, return median and mean ms."""
    # Warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    times.sort()
    median = times[len(times) // 2]
    mean = sum(times) / len(times)
    best = times[0]
    worst = times[-1]
    print(f"  {label:40s}  median={median:7.1f}ms  mean={mean:7.1f}ms  best={best:7.1f}ms  worst={worst:7.1f}ms")
    return median


def _load_model(checkpoint_path):
    """Load SAM3 model, handling BPE tokenizer issues gracefully."""
    from sam3.model_builder import build_sam3_image_model
    try:
        model = build_sam3_image_model(checkpoint_path)
        return model.to("cuda").eval()
    except Exception as e:
        print(f"  (Full model load failed: {e})")
        print("  Building vision backbone only (no text encoder needed for benchmarking)...")

    # Build the full VL backbone with a minimal text encoder
    from sam3.model_builder import _create_vision_backbone, _create_vl_backbone
    from sam3.model.text_encoder_ve import VETextEncoder

    vision_encoder = _create_vision_backbone()

    class _DummyTokenizer:
        context_length = 32
        def __call__(self, *args, **kwargs):
            return torch.zeros(1, 32, dtype=torch.long)

    text_encoder = VETextEncoder(d_model=256, tokenizer=_DummyTokenizer())
    vl_backbone = _create_vl_backbone(vision_encoder, text_encoder)

    # Load matching weights from checkpoint
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt if isinstance(ckpt, dict) and "model" not in ckpt else ckpt.get("model", ckpt)
    bb_prefix = "backbone."
    bb_weights = {k[len(bb_prefix):]: v for k, v in state_dict.items() if k.startswith(bb_prefix)}
    if bb_weights:
        missing, unexpected = vl_backbone.load_state_dict(bb_weights, strict=False)
        loaded = len(bb_weights) - len(unexpected)
        print(f"  Loaded {loaded} backbone weights (missing={len(missing)}, unexpected={len(unexpected)})")
    else:
        print("  WARNING: No backbone weights found in checkpoint")

    vl_backbone = vl_backbone.to("cuda").eval()

    # Wrap so bench functions can access model.backbone
    class _Model:
        def __init__(self, bb):
            self.backbone = bb
    return _Model(vl_backbone)


def bench_pytorch(checkpoint_path, warmup, runs):
    """Benchmark PyTorch backbone in FP32 and FP16."""
    print("\n=== PyTorch backbone ===")

    model = _load_model(checkpoint_path)
    backbone = model.backbone

    dummy = torch.randn(1, 3, 1008, 1008, device="cuda")

    # FP32
    @torch.inference_mode()
    def run_fp32():
        backbone.forward_image(dummy)

    benchmark(run_fp32, warmup, runs, "PyTorch FP32")

    # FP16 autocast
    @torch.inference_mode()
    def run_fp16():
        with torch.autocast("cuda", dtype=torch.float16):
            backbone.forward_image(dummy)

    benchmark(run_fp16, warmup, runs, "PyTorch FP16 (autocast)")

    # FP16 model weights
    backbone_f16 = backbone.half()
    dummy_f16 = dummy.half()

    @torch.inference_mode()
    def run_fp16_native():
        backbone_f16.forward_image(dummy_f16)

    benchmark(run_fp16_native, warmup, runs, "PyTorch FP16 (native .half())")

    # torch.compile variants (Linux/Triton only)
    import platform
    if platform.system() == "Linux":
        for mode in ("default", "reduce-overhead", "max-autotune"):
            try:
                compiled_fn = torch.compile(
                    backbone.forward_image, mode=mode, dynamic=False
                )

                @torch.inference_mode()
                def run_compiled(fn=compiled_fn):
                    with torch.autocast("cuda", dtype=torch.float16):
                        fn(dummy)

                benchmark(run_compiled, warmup, runs,
                          f"torch.compile({mode}) + FP16")
            except Exception as e:
                print(f"  torch.compile({mode}) failed: {e}")

        # CUDA graphs (lowest possible kernel launch overhead)
        try:
            backbone_f16_cg = backbone.half()
            dummy_cg = dummy.half()

            # Warmup for graph capture
            with torch.inference_mode():
                for _ in range(3):
                    backbone_f16_cg.forward_image(dummy_cg)
            torch.cuda.synchronize()

            # Capture graph
            g = torch.cuda.CUDAGraph()
            with torch.inference_mode():
                with torch.cuda.graph(g):
                    backbone_f16_cg.forward_image(dummy_cg)

            @torch.inference_mode()
            def run_cuda_graph():
                g.replay()

            benchmark(run_cuda_graph, warmup, runs, "CUDA Graph + FP16")

            # Explicit cleanup to prevent state leaking into later benchmarks
            del g
            del backbone_f16_cg
            torch.cuda.synchronize()
        except Exception as e:
            print(f"  CUDA Graph failed: {e}")
    else:
        print("  (Skipping torch.compile/CUDA graphs — Linux only)")

    del model, backbone, backbone_f16
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


def bench_trt_raw(engine_path, warmup, runs):
    """Benchmark TRT engine directly — no PyTorch wrapper, just raw TRT API."""
    print("\n=== TensorRT raw (no PyTorch wrapper) ===")

    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    with open(engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()

    # Print engine info
    num_io = engine.num_io_tensors
    print(f"  Engine: {num_io} I/O tensors")
    for i in range(num_io):
        name = engine.get_tensor_name(i)
        shape = engine.get_tensor_shape(name)
        dtype = engine.get_tensor_dtype(name)
        mode = engine.get_tensor_mode(name)
        print(f"    [{i}] {name:10s}  shape={list(shape)}  dtype={dtype}  mode={mode}")

    # Allocate raw CUDA buffers via PyTorch (simplest way)
    import torch
    buffers = {}
    for i in range(num_io):
        name = engine.get_tensor_name(i)
        shape = engine.get_tensor_shape(name)
        dt = engine.get_tensor_dtype(name)
        if dt == trt.float32:
            torch_dt = torch.float32
        elif dt == trt.float16:
            torch_dt = torch.float16
        elif dt == trt.int8:
            torch_dt = torch.int8
        else:
            torch_dt = torch.float32
        buf = torch.empty(list(shape), dtype=torch_dt, device="cuda")
        buffers[name] = buf
        context.set_tensor_address(name, buf.data_ptr())

    # Use raw CUDA stream (not PyTorch stream)
    stream = torch.cuda.Stream()

    def run_trt():
        context.execute_async_v3(stream.cuda_stream)
        stream.synchronize()

    benchmark(run_trt, warmup, runs, "TRT raw (execute_async_v3 only)")

    # With input copy (simulates real usage)
    input_name = engine.get_tensor_name(0)
    inp_buf = buffers[input_name]
    dummy_input = torch.randn(
        inp_buf.shape, dtype=inp_buf.dtype, device=inp_buf.device
    )

    def run_trt_with_copy():
        buffers[input_name].copy_(dummy_input)
        context.execute_async_v3(stream.cuda_stream)
        stream.synchronize()

    benchmark(run_trt_with_copy, warmup, runs, "TRT raw + input copy")

    del context, engine
    torch.cuda.empty_cache()


def bench_trt_wrapper(checkpoint_path, engine_path, warmup, runs):
    """Benchmark TRT backbone via our Python wrapper."""
    print("\n=== TensorRT via Python wrapper (trt_backbone.py) ===")

    model = _load_model(checkpoint_path)

    from sam3.trt.trt_backbone import TRTBackbone
    pos_module = model.backbone.vision_backbone.position_encoding
    trt_bb = TRTBackbone(
        engine_path=engine_path,
        device="cuda",
        pos_encoding_module=pos_module,
    )

    dummy = torch.randn(1, 3, 1008, 1008, device="cuda")

    def run_wrapper():
        trt_bb.forward_image(dummy)

    benchmark(run_wrapper, warmup, runs, "TRT wrapper (forward_image)")

    del model, trt_bb
    torch.cuda.empty_cache()


def bench_onnx(onnx_path, warmup, runs):
    """Benchmark ONNX Runtime with CUDA EP."""
    print("\n=== ONNX Runtime (CUDAExecutionProvider) ===")

    try:
        import onnxruntime as ort
    except ImportError:
        print("  onnxruntime not installed — skipping")
        return

    providers = ort.get_available_providers()
    print(f"  Available providers: {providers}")

    if "CUDAExecutionProvider" not in providers:
        print("  CUDAExecutionProvider not available — skipping")
        return

    sess = ort.InferenceSession(
        onnx_path,
        providers=["CUDAExecutionProvider"],
    )

    # Get input info
    input_info = sess.get_inputs()[0]
    print(f"  Input: {input_info.name}  shape={input_info.shape}  type={input_info.type}")

    dummy = np.random.randn(*input_info.shape).astype(np.float32)

    def run_ort():
        sess.run(None, {input_info.name: dummy})

    benchmark(run_ort, warmup, runs, "ONNX Runtime CUDA EP")


def main():
    parser = argparse.ArgumentParser(description="Benchmark SAM3 backbone")
    parser.add_argument("--checkpoint", required=True, help="SAM3 checkpoint (.pt)")
    parser.add_argument("--trt", default=None, nargs="+", help="TRT engine file(s) for backbone (can pass multiple)")
    parser.add_argument("--onnx", default=None, help="ONNX model file for backbone")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations")
    parser.add_argument("--runs", type=int, default=20, help="Timed iterations")
    parser.add_argument("--skip-pytorch", action="store_true", help="Skip PyTorch benchmarks")
    args = parser.parse_args()

    print(f"Benchmarking SAM3 backbone (warmup={args.warmup}, runs={args.runs})")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"PyTorch: {torch.__version__}")

    try:
        import tensorrt as trt
        print(f"TensorRT: {trt.__version__}")
    except ImportError:
        print("TensorRT: not installed")

    # 1. PyTorch
    if not args.skip_pytorch:
        bench_pytorch(args.checkpoint, args.warmup, args.runs)

    # 2. TRT raw + wrapper (for each engine)
    if args.trt:
        for engine_path in args.trt:
            print(f"\n{'='*60}")
            print(f"  Engine: {engine_path}")
            print(f"{'='*60}")
            bench_trt_raw(engine_path, args.warmup, args.runs)
            bench_trt_wrapper(args.checkpoint, engine_path, args.warmup, args.runs)

    # 3. ONNX Runtime
    if args.onnx:
        bench_onnx(args.onnx, args.warmup, args.runs)

    print("\nDone.")


if __name__ == "__main__":
    main()
