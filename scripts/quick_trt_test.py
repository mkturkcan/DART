#!/usr/bin/env python3
"""Quick test: compare a TRT backbone engine against PyTorch reference.

Reports cosine similarity per FPN output and inference speed.

Usage:
    python scripts/quick_trt_test.py \
        --engine backbone_norm_only.engine \
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
    a_flat = a.float().flatten()
    b_flat = b.float().flatten()
    return torch.nn.functional.cosine_similarity(
        a_flat.unsqueeze(0), b_flat.unsqueeze(0)
    ).item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", required=True, help="TRT engine file")
    parser.add_argument("--checkpoint", default="sam3.pt")
    parser.add_argument("--image", default="x.jpg")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--runs", type=int, default=50)
    args = parser.parse_args()

    device = "cuda"
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load image
    image = Image.open(args.image).convert("RGB")

    # --- PyTorch reference ---
    print("\n--- PyTorch FP32 Reference ---")
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_multiclass_fast import Sam3MultiClassPredictorFast

    model = build_sam3_image_model(
        device=device, checkpoint_path=args.checkpoint, eval_mode=True,
    )
    predictor = Sam3MultiClassPredictorFast(
        model, device=device, resolution=1008,
        use_fp16=False, detection_only=True,
    )

    from torchvision.transforms import v2

    from torchvision.transforms import v2

    with torch.inference_mode():
        resized = image.resize((1008, 1008), Image.BILINEAR)
        img_tensor = v2.functional.to_image(resized).to(device)
        img_tensor = predictor.transform(img_tensor).unsqueeze(0)
        ref_out = model.backbone.forward_image(img_tensor)

    # Extract comparable outputs (FPN features)
    ref_fpn = ref_out["backbone_fpn"]
    ref_keys = [f"fpn_{i}" for i in range(len(ref_fpn))]
    ref_dict = {f"fpn_{i}": ref_fpn[i] for i in range(len(ref_fpn))}
    for k in ref_keys:
        v = ref_dict[k]
        print(f"  {k}: {v.shape}, range [{v.min():.4f}, {v.max():.4f}]")

    del model, predictor
    torch.cuda.empty_cache()

    # --- TRT engine ---
    print(f"\n--- TRT Engine: {args.engine} ---")
    from sam3.trt.trt_backbone import TRTBackbone

    trt_bb = TRTBackbone(args.engine, device=device)

    # Warmup
    print(f"Warming up ({args.warmup} runs)...")
    with torch.inference_mode():
        for _ in range(args.warmup):
            trt_bb.forward_image(img_tensor)
    torch.cuda.synchronize()

    # Timed runs
    print(f"Timing ({args.runs} runs)...")
    times = []
    with torch.inference_mode():
        for _ in range(args.runs):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            trt_out = trt_bb.forward_image(img_tensor)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)

    avg_ms = np.mean(times)
    min_ms = np.min(times)
    p50 = np.percentile(times, 50)
    p95 = np.percentile(times, 95)

    print(f"\n  Avg: {avg_ms:.1f}ms")
    print(f"  Min: {min_ms:.1f}ms")
    print(f"  P50: {p50:.1f}ms")
    print(f"  P95: {p95:.1f}ms")

    # Cosine similarity
    # TRT backbone returns dict with backbone_fpn list
    trt_fpn = trt_out["backbone_fpn"]
    trt_dict = {f"fpn_{i}": trt_fpn[i] for i in range(len(trt_fpn))}

    print(f"\n  Cosine similarity vs PyTorch FP32:")
    for k in ref_keys:
        if k in trt_dict:
            cos = cosine_similarity(ref_dict[k], trt_dict[k])
            status = "OK" if cos > 0.999 else "WARN" if cos > 0.99 else "BAD"
            range_str = f"[{trt_dict[k].min():.4f}, {trt_dict[k].max():.4f}]"
            print(f"    {k}: cos={cos:.6f} [{status}]  range {range_str}")

    del trt_bb
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
