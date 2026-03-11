#!/usr/bin/env python3
"""Benchmark ONNX Runtime backbone inference: accuracy vs PyTorch + speed.

Tests CUDAExecutionProvider (cuDNN/cuBLAS) and optionally
TensorrtExecutionProvider for the backbone ONNX model.

Usage:
    python scripts/benchmark_onnxrt.py \
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


def cosine_similarity(a, b):
    if isinstance(a, np.ndarray):
        a = torch.from_numpy(a)
    if isinstance(b, np.ndarray):
        b = torch.from_numpy(b)
    a_flat = a.float().flatten()
    b_flat = b.float().flatten()
    return torch.nn.functional.cosine_similarity(
        a_flat.unsqueeze(0), b_flat.unsqueeze(0)
    ).item()


def benchmark_provider(session, input_name, input_np, output_names, ref_dict, n_warmup, n_runs, label):
    """Benchmark an ORT session and compute accuracy."""
    print(f"\n--- {label} ---")

    # Warmup
    print(f"  Warming up ({n_warmup} runs)...")
    for _ in range(n_warmup):
        session.run(output_names, {input_name: input_np})

    # Timed runs
    print(f"  Timing ({n_runs} runs)...")
    times = []
    last_outputs = None
    for _ in range(n_runs):
        t0 = time.perf_counter()
        outputs = session.run(output_names, {input_name: input_np})
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
        last_outputs = outputs

    avg_ms = np.mean(times)
    min_ms = np.min(times)
    p50 = np.percentile(times, 50)
    p95 = np.percentile(times, 95)

    print(f"  Avg: {avg_ms:.1f}ms")
    print(f"  Min: {min_ms:.1f}ms")
    print(f"  P50: {p50:.1f}ms")
    print(f"  P95: {p95:.1f}ms")

    # Accuracy
    print(f"  Cosine similarity vs PyTorch FP32:")
    for i, name in enumerate(output_names):
        key = f"fpn_{i}"
        if key in ref_dict:
            cos = cosine_similarity(ref_dict[key], last_outputs[i])
            status = "OK" if cos > 0.999 else "WARN" if cos > 0.99 else "BAD"
            range_str = f"[{last_outputs[i].min():.4f}, {last_outputs[i].max():.4f}]"
            print(f"    {name} ({key}): cos={cos:.6f} [{status}]  range {range_str}")

    return {"label": label, "avg_ms": avg_ms, "min_ms": min_ms, "p50": p50, "p95": p95}


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark ONNX Runtime backbone inference"
    )
    parser.add_argument("--onnx", required=True, help="Backbone ONNX model")
    parser.add_argument("--checkpoint", required=True, help="SAM3 checkpoint")
    parser.add_argument("--image", required=True, help="Test image")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument(
        "--providers", nargs="+",
        default=["cuda", "cuda-fp16"],
        choices=["cuda", "cuda-fp16", "trt", "trt-fp16"],
        help="Providers to test",
    )
    args = parser.parse_args()

    import onnxruntime as ort
    print(f"ONNX Runtime version: {ort.__version__}")
    print(f"Available providers: {ort.get_available_providers()}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    # --- PyTorch reference ---
    print("\n--- PyTorch FP32 Reference ---")
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_multiclass_fast import Sam3MultiClassPredictorFast
    from torchvision.transforms import v2

    model = build_sam3_image_model(
        device="cuda", checkpoint_path=args.checkpoint, eval_mode=True,
    )
    predictor = Sam3MultiClassPredictorFast(
        model, device="cuda", resolution=1008,
        use_fp16=False, detection_only=True,
    )

    image = Image.open(args.image).convert("RGB")
    with torch.inference_mode():
        resized = image.resize((1008, 1008), Image.BILINEAR)
        img_tensor = v2.functional.to_image(resized).to("cuda")
        img_tensor = predictor.transform(img_tensor).unsqueeze(0)
        ref_out = model.backbone.forward_image(img_tensor)

    ref_fpn = ref_out["backbone_fpn"]
    ref_dict = {f"fpn_{i}": ref_fpn[i] for i in range(len(ref_fpn))}
    for k, v in ref_dict.items():
        print(f"  {k}: {v.shape}, range [{v.min():.4f}, {v.max():.4f}]")

    # Prepare input for ORT (CPU numpy, FP32)
    input_np = img_tensor.cpu().numpy()
    print(f"  Input shape: {input_np.shape}, dtype: {input_np.dtype}")

    del model, predictor
    torch.cuda.empty_cache()

    # --- Get output names from ONNX ---
    import onnx
    onnx_model = onnx.load(args.onnx)
    output_names = [o.name for o in onnx_model.graph.output]
    input_name = onnx_model.graph.input[0].name
    print(f"\n  ONNX input: {input_name}")
    print(f"  ONNX outputs: {output_names}")
    del onnx_model

    results = []
    for provider in args.providers:
        try:
            if provider == "cuda":
                sess_options = ort.SessionOptions()
                sess = ort.InferenceSession(
                    args.onnx,
                    sess_options=sess_options,
                    providers=[
                        ("CUDAExecutionProvider", {
                            "device_id": 0,
                            "arena_extend_strategy": "kSameAsRequested",
                        }),
                    ],
                )
                result = benchmark_provider(
                    sess, input_name, input_np, output_names,
                    ref_dict, args.warmup, args.runs,
                    "ORT CUDAExecutionProvider (FP32)",
                )
                results.append(result)
                del sess

            elif provider == "cuda-fp16":
                # ORT CUDA with graph optimization + FP16 conversion
                sess_options = ort.SessionOptions()
                sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                sess = ort.InferenceSession(
                    args.onnx,
                    sess_options=sess_options,
                    providers=[
                        ("CUDAExecutionProvider", {
                            "device_id": 0,
                            "arena_extend_strategy": "kSameAsRequested",
                        }),
                    ],
                )
                # Run with FP16 input
                input_fp16 = input_np.astype(np.float16).astype(np.float32)
                result = benchmark_provider(
                    sess, input_name, input_fp16, output_names,
                    ref_dict, args.warmup, args.runs,
                    "ORT CUDAExecutionProvider (FP32 model, FP16 input)",
                )
                results.append(result)
                del sess

            elif provider == "trt":
                sess_options = ort.SessionOptions()
                sess = ort.InferenceSession(
                    args.onnx,
                    sess_options=sess_options,
                    providers=[
                        ("TensorrtExecutionProvider", {
                            "device_id": 0,
                            "trt_fp16_enable": False,
                            "trt_engine_cache_enable": True,
                            "trt_engine_cache_path": "./ort_trt_cache",
                        }),
                        ("CUDAExecutionProvider", {"device_id": 0}),
                    ],
                )
                result = benchmark_provider(
                    sess, input_name, input_np, output_names,
                    ref_dict, args.warmup, args.runs,
                    "ORT TensorrtExecutionProvider (FP32)",
                )
                results.append(result)
                del sess

            elif provider == "trt-fp16":
                sess_options = ort.SessionOptions()
                sess = ort.InferenceSession(
                    args.onnx,
                    sess_options=sess_options,
                    providers=[
                        ("TensorrtExecutionProvider", {
                            "device_id": 0,
                            "trt_fp16_enable": True,
                            "trt_engine_cache_enable": True,
                            "trt_engine_cache_path": "./ort_trt_cache_fp16",
                        }),
                        ("CUDAExecutionProvider", {"device_id": 0}),
                    ],
                )
                result = benchmark_provider(
                    sess, input_name, input_np, output_names,
                    ref_dict, args.warmup, args.runs,
                    "ORT TensorrtExecutionProvider (FP16)",
                )
                results.append(result)
                del sess

        except Exception as e:
            print(f"\n  ERROR with {provider}: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print(f"\n\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Provider':<50s} {'Avg':>7s} {'Min':>7s} {'P50':>7s}")
    print(f"{'-'*50} {'-'*7} {'-'*7} {'-'*7}")
    for r in results:
        print(f"{r['label']:<50s} {r['avg_ms']:>6.1f}ms {r['min_ms']:>6.1f}ms {r['p50']:>6.1f}ms")


if __name__ == "__main__":
    main()
