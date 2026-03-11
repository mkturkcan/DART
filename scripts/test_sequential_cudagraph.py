#!/usr/bin/env python3
"""Test sequential backbone execution with manual CUDA graph capture.

Tests the hypothesis that sequential (no streams) + CUDA graph for backbone
is faster than the pipelined approach due to avoiding SM contention.
"""

import os
import sys
import time

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import v2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_multiclass_fast import Sam3MultiClassPredictorFast


def benchmark(fn, n_warmup=3, n_runs=20, label=""):
    """Benchmark a callable, return avg_ms."""
    with torch.inference_mode():
        for _ in range(n_warmup):
            fn()
    torch.cuda.synchronize()

    times = []
    with torch.inference_mode():
        for _ in range(n_runs):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)

    avg = np.mean(times)
    mn = np.min(times)
    p50 = np.percentile(times, 50)
    print(f"  {label}: avg={avg:.1f}ms  min={mn:.1f}ms  p50={p50:.1f}ms")
    return p50


def main():
    device = "cuda"
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load model
    model = build_sam3_image_model(
        device=device, checkpoint_path="sam3.pt", eval_mode=True,
    )
    classes = ["person", "car", "bicycle", "dog"]

    # --- Predictor with TRT enc-dec ---
    trt_path = "enc_dec_fp16_pure.engine"
    has_trt = os.path.exists(trt_path)

    predictor = Sam3MultiClassPredictorFast(
        model, device=device, resolution=1008,
        use_fp16=True, detection_only=True,
        trt_enc_dec_engine_path=trt_path if has_trt else None,
        trt_max_classes=4,
    )
    predictor.set_classes(classes)
    predictor._ensure_compiled()

    backbone = model.backbone

    # Prepare input
    image = Image.open("x.jpg").convert("RGB")
    resized = image.resize((1008, 1008), Image.BILINEAR)
    img_tensor = v2.functional.to_image(resized).to(device)
    img_tensor = predictor.transform(img_tensor).unsqueeze(0)

    # ---------------------------------------------------------------
    # Approach 1: Full backbone with torch.compile default + FP16
    # ---------------------------------------------------------------
    print("\n=== Approach 1: torch.compile default + FP16 (sequential) ===")

    compiled_backbone = torch.compile(
        backbone.forward_image, mode="default", dynamic=False,
    )

    # Warmup
    with torch.inference_mode():
        for _ in range(3):
            with torch.autocast("cuda", dtype=torch.float16):
                bb_out = compiled_backbone(img_tensor)

    # Benchmark backbone alone
    def run_backbone():
        with torch.autocast("cuda", dtype=torch.float16):
            return compiled_backbone(img_tensor)

    p50_bb = benchmark(run_backbone, label="Backbone only")

    # Benchmark backbone + predict
    def run_full():
        with torch.autocast("cuda", dtype=torch.float16):
            bb_out = compiled_backbone(img_tensor)
        state = {
            "backbone_out": bb_out,
            "original_height": 1008,
            "original_width": 1008,
        }
        predictor.predict(state, confidence_threshold=0.3)

    p50_full = benchmark(run_full, label="Backbone + predict")

    torch._dynamo.reset()
    torch.cuda.empty_cache()

    # ---------------------------------------------------------------
    # Approach 2: Manual CUDA graph on full backbone
    # ---------------------------------------------------------------
    print("\n=== Approach 2: Manual CUDA graph on full backbone ===")

    compiled_backbone = torch.compile(
        backbone.forward_image, mode="default", dynamic=False,
    )

    # Static input buffer
    static_input = img_tensor.clone()

    # Warmup with cache_enabled=False
    with torch.inference_mode():
        with torch.amp.autocast("cuda", dtype=torch.float16,
                                cache_enabled=False):
            for _ in range(5):
                bb_out = compiled_backbone(static_input)
    torch.cuda.synchronize()

    # Capture CUDA graph
    g = torch.cuda.CUDAGraph()
    with torch.inference_mode():
        with torch.amp.autocast("cuda", dtype=torch.float16,
                                cache_enabled=False):
            with torch.cuda.graph(g):
                static_bb_out = compiled_backbone(static_input)

    print("  CUDA graph captured!")

    # Verify correctness
    with torch.inference_mode():
        with torch.autocast("cuda", dtype=torch.float16):
            ref_out = backbone.forward_image(img_tensor)
        static_input.copy_(img_tensor)
        g.replay()
        torch.cuda.synchronize()

        for key in ["vision_features"]:
            if key in ref_out and key in static_bb_out:
                cos = torch.nn.functional.cosine_similarity(
                    ref_out[key].flatten().float(),
                    static_bb_out[key].flatten().float(),
                    dim=0,
                ).item()
                diff = (ref_out[key] - static_bb_out[key]).abs().max().item()
                print(f"  {key}: cos={cos:.6f}  max_diff={diff:.2e}")

    # Benchmark CUDA graph backbone alone
    def run_graph_backbone():
        g.replay()

    p50_graph_bb = benchmark(run_graph_backbone, label="CUDA graph backbone")

    # Benchmark CUDA graph backbone + predict
    def run_graph_full():
        static_input.copy_(img_tensor)
        g.replay()
        torch.cuda.synchronize()  # ensure backbone done before predict reads
        state = {
            "backbone_out": static_bb_out,
            "original_height": 1008,
            "original_width": 1008,
        }
        predictor.predict(state, confidence_threshold=0.3)

    p50_graph_full = benchmark(run_graph_full, label="CUDA graph + predict")

    # ---------------------------------------------------------------
    # Approach 3: CUDA graph on split backbone (sequential, no streams)
    # ---------------------------------------------------------------
    print("\n=== Approach 3: CUDA graph on split backbone (sequential) ===")

    torch._dynamo.reset()
    torch.cuda.empty_cache()

    split = 20
    part1_fn = torch.compile(
        backbone.forward_image_part1, mode="default", dynamic=False,
    )
    part2_fn = torch.compile(
        backbone.forward_image_part2, mode="default", dynamic=False,
    )

    static_input2 = img_tensor.clone()

    # Warmup part1
    with torch.inference_mode():
        with torch.amp.autocast("cuda", dtype=torch.float16,
                                cache_enabled=False):
            for _ in range(5):
                inter = part1_fn(static_input2, split)
    torch.cuda.synchronize()

    # Capture part1 graph
    g1 = torch.cuda.CUDAGraph()
    with torch.inference_mode():
        with torch.amp.autocast("cuda", dtype=torch.float16,
                                cache_enabled=False):
            with torch.cuda.graph(g1):
                static_inter = part1_fn(static_input2, split)

    # Make a copy for part2 input
    static_inter_for_p2 = {"x": static_inter["x"].clone(), "s": static_inter["s"]}

    # Warmup part2
    with torch.inference_mode():
        with torch.amp.autocast("cuda", dtype=torch.float16,
                                cache_enabled=False):
            for _ in range(5):
                bb_out = part2_fn(static_inter_for_p2, split)
    torch.cuda.synchronize()

    # Capture part2 graph
    g2 = torch.cuda.CUDAGraph()
    with torch.inference_mode():
        with torch.amp.autocast("cuda", dtype=torch.float16,
                                cache_enabled=False):
            with torch.cuda.graph(g2):
                static_split_bb_out = part2_fn(static_inter_for_p2, split)

    print(f"  CUDA graphs captured (split={split})")

    # Benchmark split CUDA graph backbone only (sequential)
    def run_split_graph_backbone():
        g1.replay()
        static_inter_for_p2["x"].copy_(static_inter["x"])
        g2.replay()

    p50_split_bb = benchmark(run_split_graph_backbone, label="Split graph backbone")

    # Benchmark split CUDA graph backbone + predict
    def run_split_graph_full():
        static_input2.copy_(img_tensor)
        g1.replay()
        static_inter_for_p2["x"].copy_(static_inter["x"])
        g2.replay()
        torch.cuda.synchronize()
        state = {
            "backbone_out": static_split_bb_out,
            "original_height": 1008,
            "original_width": 1008,
        }
        predictor.predict(state, confidence_threshold=0.3)

    p50_split_full = benchmark(run_split_graph_full, label="Split graph + predict")

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  torch.compile backbone only:    {p50_bb:.1f}ms")
    print(f"  torch.compile + predict:        {p50_full:.1f}ms")
    print(f"  CUDA graph backbone only:       {p50_graph_bb:.1f}ms")
    print(f"  CUDA graph + predict:           {p50_graph_full:.1f}ms")
    print(f"  Split graph backbone only:      {p50_split_bb:.1f}ms")
    print(f"  Split graph + predict:          {p50_split_full:.1f}ms")
    print()
    print(f"  Enc-dec type: {'TRT FP16' if has_trt else 'PyTorch compiled'}")
    print(f"  Best full-pipeline FPS:         {1000 / min(p50_full, p50_graph_full, p50_split_full):.1f}")


if __name__ == "__main__":
    main()
