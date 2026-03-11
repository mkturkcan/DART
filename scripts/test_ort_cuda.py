#!/usr/bin/env python3
"""Quick test of ONNX Runtime CUDAExecutionProvider on backbone."""

import os
import sys
import time

# Add torch lib to DLL search path for cuDNN 9
torch_lib = os.path.join(
    sys.prefix, "Lib", "site-packages", "torch", "lib"
)
if os.path.isdir(torch_lib):
    os.add_dll_directory(torch_lib)

import numpy as np
import onnxruntime as ort

print(f"ORT version: {ort.__version__}")
print(f"Providers: {ort.get_available_providers()}")

onnx_path = sys.argv[1] if len(sys.argv) > 1 else "backbone_folded.onnx"

# Try TRT provider first (doesn't need cuDNN), fall back to CUDA
try:
    sess = ort.InferenceSession(
        onnx_path,
        providers=[
            ("TensorrtExecutionProvider", {
                "device_id": 0,
                "trt_fp16_enable": False,
                "trt_engine_cache_enable": True,
                "trt_engine_cache_path": "./ort_trt_cache",
            }),
        ]
    )
except Exception as e:
    print(f"TRT provider failed: {e}")
    sess = ort.InferenceSession(
        onnx_path,
        providers=[("CUDAExecutionProvider", {"device_id": 0})]
    )
print(f"Session created with: {sess.get_providers()}")

inp = sess.get_inputs()[0]
print(f"Input: {inp.name} {inp.shape} {inp.type}")
for o in sess.get_outputs():
    print(f"Output: {o.name} {o.shape} {o.type}")

x = np.random.randn(1, 3, 1008, 1008).astype(np.float32)

print("\nWarmup (5 runs)...")
for i in range(5):
    out = sess.run(None, {inp.name: x})
    print(f"  Run {i}: shapes = {[o.shape for o in out]}")

print("\nTiming (20 runs)...")
times = []
for _ in range(20):
    t0 = time.perf_counter()
    out = sess.run(None, {inp.name: x})
    t1 = time.perf_counter()
    times.append((t1 - t0) * 1000)

print(f"  Avg: {np.mean(times):.1f}ms")
print(f"  Min: {np.min(times):.1f}ms")
print(f"  P50: {np.percentile(times, 50):.1f}ms")
print(f"  P95: {np.percentile(times, 95):.1f}ms")
