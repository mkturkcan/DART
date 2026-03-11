#!/usr/bin/env python3
"""Diagnose why the split backbone pipeline isn't overlapping.

Measures GPU timings with CUDA events to see if part1 and part2+enc-dec
actually run concurrently on different streams.
"""

import os
import sys
import time

import torch
from PIL import Image
from torchvision.transforms import v2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_multiclass_fast import Sam3MultiClassPredictorFast


def main():
    device = "cuda"
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load model
    model = build_sam3_image_model(
        device=device, checkpoint_path="sam3.pt", eval_mode=True,
    )
    predictor = Sam3MultiClassPredictorFast(
        model, device=device, resolution=1008,
        use_fp16=True, detection_only=True,
    )
    predictor.set_classes(["person", "car", "bicycle", "dog"])
    predictor._ensure_compiled()

    backbone = model.backbone
    split_block = 20

    # Compile part1 and part2
    part1_fn = torch.compile(backbone.forward_image_part1, mode="default", dynamic=False)
    part2_fn = torch.compile(backbone.forward_image_part2, mode="default", dynamic=False)

    # Prepare input
    image = Image.open("x.jpg").convert("RGB")
    resized = image.resize((1008, 1008), Image.BILINEAR)
    img_tensor = v2.functional.to_image(resized).to(device)
    img_tensor = predictor.transform(img_tensor).unsqueeze(0)

    backbone_stream = torch.cuda.Stream(device=device)

    # Warmup
    print("\nWarming up...")
    with torch.inference_mode():
        for _ in range(3):
            with torch.autocast("cuda", dtype=torch.float16):
                inter = part1_fn(img_tensor, split_block)
                inter_clone = {"x": inter["x"].clone(), "s": inter["s"]}
                bb_out = part2_fn(inter_clone, split_block)
            state = {
                "backbone_out": bb_out,
                "original_height": 1008,
                "original_width": 1008,
            }
            predictor.predict(state, confidence_threshold=0.5)
    torch.cuda.synchronize()
    print("  Warmup done")

    # Get a cached intermediate for part2
    with torch.inference_mode():
        with torch.autocast("cuda", dtype=torch.float16):
            inter_base = part1_fn(img_tensor, split_block)
            inter_cached = {"x": inter_base["x"].clone(), "s": inter_base["s"]}
    torch.cuda.synchronize()

    # ---------------------------------------------------------------
    # Test 1: Sequential execution (baseline)
    # ---------------------------------------------------------------
    print("\n--- Test 1: Sequential (all on default stream) ---")
    times = []
    with torch.inference_mode():
        for _ in range(5):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.autocast("cuda", dtype=torch.float16):
                inter = part1_fn(img_tensor, split_block)
                inter_c = {"x": inter["x"].clone(), "s": inter["s"]}
                bb_out = part2_fn(inter_c, split_block)
            state = {
                "backbone_out": bb_out,
                "original_height": 1008,
                "original_width": 1008,
            }
            predictor.predict(state, confidence_threshold=0.5)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)
    avg = sum(times) / len(times)
    print(f"  Sequential total: {avg:.1f}ms")

    # ---------------------------------------------------------------
    # Test 2: Part1 only
    # ---------------------------------------------------------------
    print("\n--- Test 2: Part1 alone ---")
    times = []
    with torch.inference_mode():
        for _ in range(10):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.autocast("cuda", dtype=torch.float16):
                inter = part1_fn(img_tensor, split_block)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)
    avg = sum(times) / len(times)
    print(f"  Part1 alone: {avg:.1f}ms")

    # ---------------------------------------------------------------
    # Test 3: Part2 + enc-dec only
    # ---------------------------------------------------------------
    print("\n--- Test 3: Part2 + enc-dec alone ---")
    times = []
    with torch.inference_mode():
        for _ in range(10):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            inter_c = {"x": inter_cached["x"].clone(), "s": inter_cached["s"]}
            with torch.autocast("cuda", dtype=torch.float16):
                bb_out = part2_fn(inter_c, split_block)
            state = {
                "backbone_out": bb_out,
                "original_height": 1008,
                "original_width": 1008,
            }
            predictor.predict(state, confidence_threshold=0.5)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)
    avg = sum(times) / len(times)
    print(f"  Part2+enc-dec alone: {avg:.1f}ms")

    # ---------------------------------------------------------------
    # Test 4: Overlap part1(stream) with part2+enc-dec(default)
    # ---------------------------------------------------------------
    print("\n--- Test 4: Pipelined (part1 on stream, part2+enc-dec on default) ---")
    times = []
    with torch.inference_mode():
        for _ in range(10):
            torch.cuda.synchronize()
            t0 = time.perf_counter()

            # Launch part1 on backbone_stream
            backbone_stream.wait_event(
                torch.cuda.current_stream(device).record_event()
            )
            with torch.cuda.stream(backbone_stream):
                with torch.autocast("cuda", dtype=torch.float16):
                    _inter_next = part1_fn(img_tensor, split_block)
            bb_event = backbone_stream.record_event()

            # Run part2+enc-dec on default stream
            inter_c = {"x": inter_cached["x"].clone(), "s": inter_cached["s"]}
            with torch.autocast("cuda", dtype=torch.float16):
                bb_out = part2_fn(inter_c, split_block)
            state = {
                "backbone_out": bb_out,
                "original_height": 1008,
                "original_width": 1008,
            }
            predictor.predict(state, confidence_threshold=0.5)

            # Wait for both
            torch.cuda.current_stream(device).wait_event(bb_event)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)

    avg = sum(times) / len(times)
    print(f"  Pipelined total: {avg:.1f}ms")

    # ---------------------------------------------------------------
    # Test 5: Overlap with CUDA events for detailed timing
    # ---------------------------------------------------------------
    print("\n--- Test 5: CUDA event detailed timing ---")
    e_start = torch.cuda.Event(enable_timing=True)
    e_p1_done = torch.cuda.Event(enable_timing=True)
    e_p2_done = torch.cuda.Event(enable_timing=True)
    e_enc_done = torch.cuda.Event(enable_timing=True)
    e_all_done = torch.cuda.Event(enable_timing=True)

    p1_times = []
    p2_enc_times = []
    total_times = []

    with torch.inference_mode():
        for _ in range(10):
            torch.cuda.synchronize()

            # Record start on default stream
            e_start.record()

            # Launch part1 on backbone_stream
            backbone_stream.wait_event(
                torch.cuda.current_stream(device).record_event()
            )
            with torch.cuda.stream(backbone_stream):
                with torch.autocast("cuda", dtype=torch.float16):
                    _inter_next = part1_fn(img_tensor, split_block)
                e_p1_done.record()

            # Run part2+enc-dec on default stream
            inter_c = {"x": inter_cached["x"].clone(), "s": inter_cached["s"]}
            with torch.autocast("cuda", dtype=torch.float16):
                bb_out = part2_fn(inter_c, split_block)
            e_p2_done.record()

            state = {
                "backbone_out": bb_out,
                "original_height": 1008,
                "original_width": 1008,
            }
            predictor.predict(state, confidence_threshold=0.5)
            e_enc_done.record()

            # Wait for backbone_stream too
            torch.cuda.current_stream(device).wait_event(e_p1_done)
            e_all_done.record()

            torch.cuda.synchronize()

            p1_ms = e_start.elapsed_time(e_p1_done)
            p2_ms = e_start.elapsed_time(e_p2_done)
            enc_ms = e_start.elapsed_time(e_enc_done)
            all_ms = e_start.elapsed_time(e_all_done)

            p1_times.append(p1_ms)
            p2_enc_times.append(enc_ms)
            total_times.append(all_ms)

    print(f"  Part1 (backbone_stream): {sum(p1_times)/len(p1_times):.1f}ms")
    print(f"  Part2+enc-dec (default): {sum(p2_enc_times)/len(p2_enc_times):.1f}ms")
    print(f"  Total (max of both):     {sum(total_times)/len(total_times):.1f}ms")
    expected = max(
        sum(p1_times) / len(p1_times),
        sum(p2_enc_times) / len(p2_enc_times),
    )
    print(f"  Expected if overlapping: {expected:.1f}ms")
    print(f"  Overlap efficiency:      {expected / (sum(total_times)/len(total_times)) * 100:.0f}%")


if __name__ == "__main__":
    main()
