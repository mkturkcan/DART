#!/usr/bin/env python3
"""Find the optimal backbone split point for the pipelined video processor.

Benchmarks every possible split point (1..31) and finds the one that
minimizes per-frame time = max(part1_time, part2_time + enc_dec_time).

For each split point, measures:
  - Part 1: patch_embed + pos_embed + ln_pre + blocks[0:split]
  - Part 2: blocks[split:32] + FPN neck
Both under torch.compile default + FP16 autocast.

Then computes the pipeline-optimal split assuming a given enc_dec cost.

Usage:
    python scripts/optimize_split_point.py --checkpoint sam3.pt --image x.jpg

    # With a specific enc-dec cost (default: auto-measured or 14ms):
    python scripts/optimize_split_point.py --checkpoint sam3.pt --image x.jpg \
        --enc-dec-ms 14.0

    # Also include TRT enc-dec measurement:
    python scripts/optimize_split_point.py --checkpoint sam3.pt --image x.jpg \
        --trt-engine enc_dec_fp16_pure.engine
"""

import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def benchmark_fn(fn, n_warmup=5, n_runs=30):
    """Benchmark a callable, return (avg_ms, min_ms, p50_ms)."""
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

    return np.mean(times), np.min(times), np.percentile(times, 50)


def main():
    parser = argparse.ArgumentParser(
        description="Find optimal backbone split point for pipelined video"
    )
    parser.add_argument("--checkpoint", default="sam3.pt")
    parser.add_argument("--image", default="x.jpg")
    parser.add_argument(
        "--enc-dec-ms", type=float, default=None,
        help="Enc-dec cost in ms (default: auto-measure or use --trt-engine)",
    )
    parser.add_argument(
        "--trt-engine", default=None,
        help="TRT enc-dec engine path for measuring enc-dec cost",
    )
    parser.add_argument(
        "--classes", nargs="+",
        default=["person", "car", "bicycle", "dog"],
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument(
        "--compile", action="store_true", default=True,
        help="Use torch.compile (default: True)",
    )
    parser.add_argument(
        "--no-compile", action="store_true",
        help="Disable torch.compile (use eager FP16)",
    )
    args = parser.parse_args()
    use_compile = args.compile and not args.no_compile

    device = "cuda"
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Classes: {args.classes} (N={len(args.classes)})")
    print(f"Mode: {'torch.compile default + FP16' if use_compile else 'eager FP16'}")

    # ---------------------------------------------------------------
    # Load model
    # ---------------------------------------------------------------
    from PIL import Image
    from torchvision.transforms import v2

    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_multiclass_fast import Sam3MultiClassPredictorFast

    print("\nLoading model...")
    model = build_sam3_image_model(
        device=device, checkpoint_path=args.checkpoint, eval_mode=True,
    )
    predictor = Sam3MultiClassPredictorFast(
        model, device=device, resolution=1008,
        use_fp16=False, detection_only=True,
    )
    predictor.set_classes(args.classes)

    # Prepare image tensor
    image = Image.open(args.image).convert("RGB")
    resized = image.resize((1008, 1008), Image.BILINEAR)
    img_tensor = v2.functional.to_image(resized).to(device)
    img_tensor = predictor.transform(img_tensor).unsqueeze(0)

    backbone = model.backbone
    trunk = backbone.vision_backbone.trunk
    num_blocks = len(trunk.blocks)
    global_attn_blocks = list(trunk.full_attn_ids)

    print(f"\nViT: {num_blocks} blocks, global attention at {global_attn_blocks}")
    print(f"Benchmarking {args.runs} runs per config, {args.warmup} warmup...\n")

    # ---------------------------------------------------------------
    # Measure enc-dec cost
    # ---------------------------------------------------------------
    enc_dec_costs = {}

    if args.trt_engine and os.path.exists(args.trt_engine):
        from sam3.trt.trt_enc_dec import TRTEncoderDecoder

        state = predictor.set_image(image)
        backbone_out = state["backbone_out"]
        img_ids = torch.tensor([0], device=device, dtype=torch.long)
        _, img_feats, img_pos_embeds, vis_feat_sizes = model._get_img_feats(
            backbone_out, img_ids
        )
        text_feats = predictor._batched_text
        text_mask = predictor._batched_mask
        N = len(args.classes)

        trt_enc_dec = TRTEncoderDecoder(
            engine_path=args.trt_engine, max_classes=4, device=device,
        )

        def trt_fn():
            return trt_enc_dec.forward(
                img_feats=img_feats,
                img_pos_embeds=img_pos_embeds,
                text_feats=text_feats,
                text_mask=text_mask,
                num_classes=N,
            )

        avg, mn, p50 = benchmark_fn(trt_fn, args.warmup, args.runs)
        enc_dec_costs["TRT FP16"] = p50
        print(f"  TRT enc-dec: avg={avg:.1f}ms  min={mn:.1f}ms  p50={p50:.1f}ms")
        del trt_enc_dec
        torch.cuda.empty_cache()

    # Measure PyTorch compiled enc-dec
    if use_compile:
        state = predictor.set_image(image)
        backbone_out = state["backbone_out"]
        img_ids = torch.tensor([0], device=device, dtype=torch.long)
        _, img_feats, img_pos_embeds, vis_feat_sizes = model._get_img_feats(
            backbone_out, img_ids
        )
        text_feats = predictor._batched_text
        text_mask = predictor._batched_mask
        N = len(args.classes)

        encoder = model.transformer.encoder
        decoder = model.transformer.decoder
        scoring = model.dot_prod_scoring
        from sam3.model.model_misc import inverse_sigmoid

        compiled_encoder = torch.compile(encoder.forward, mode="default", dynamic=False)
        compiled_decoder = torch.compile(decoder.forward, mode="default", dynamic=False)

        def pytorch_enc_dec():
            with torch.autocast("cuda", dtype=torch.float16):
                batched_img = [f.expand(-1, N, -1).contiguous() for f in img_feats]
                batched_pos = [p.expand(-1, N, -1).contiguous() for p in img_pos_embeds]
                prompt = text_feats
                prompt_mask = text_mask
                prompt_pos = torch.zeros_like(prompt)

                memory = compiled_encoder(
                    src=batched_img, src_key_padding_mask=None,
                    src_pos=batched_pos, prompt=prompt,
                    prompt_pos=prompt_pos,
                    prompt_key_padding_mask=prompt_mask,
                    feat_sizes=vis_feat_sizes,
                )
                query_embed = decoder.query_embed.weight
                tgt = query_embed.unsqueeze(1).expand(-1, N, -1)
                hs, ref_boxes, _, _ = compiled_decoder(
                    tgt=tgt, memory=memory["memory"],
                    memory_key_padding_mask=memory["padding_mask"],
                    pos=memory["pos_embed"], reference_boxes=None,
                    level_start_index=memory["level_start_index"],
                    spatial_shapes=memory["spatial_shapes"],
                    valid_ratios=memory["valid_ratios"],
                    tgt_mask=None, memory_text=prompt,
                    text_attention_mask=prompt_mask, apply_dac=False,
                )
                hs = hs.transpose(1, 2)
                ref_boxes = ref_boxes.transpose(1, 2)
                hs_last = hs[-1:]
                ref_last = ref_boxes[-1]
                scores = scoring(hs_last, prompt, prompt_mask)[0]
                box_offsets = decoder.bbox_embed(hs_last)[0]
                boxes = (inverse_sigmoid(ref_last) + box_offsets).sigmoid()
                return scores, boxes

        avg, mn, p50 = benchmark_fn(pytorch_enc_dec, args.warmup, args.runs)
        enc_dec_costs["compile+FP16"] = p50
        print(f"  Compiled enc-dec: avg={avg:.1f}ms  min={mn:.1f}ms  p50={p50:.1f}ms")

        torch._dynamo.reset()
        torch.cuda.empty_cache()

    if args.enc_dec_ms is not None:
        enc_dec_costs["user-specified"] = args.enc_dec_ms

    if not enc_dec_costs:
        enc_dec_costs["default-estimate"] = 14.0
        print(f"  Using default enc-dec estimate: 14.0ms")

    print(f"\n  Enc-dec costs: {enc_dec_costs}")

    # ---------------------------------------------------------------
    # Benchmark each split point
    # ---------------------------------------------------------------
    print(f"\n{'='*80}")
    print("Benchmarking backbone split points (split = # blocks in Part 1)")
    print(f"{'='*80}\n")

    # We need to test split points from 1 to 31 (0 = no blocks in part1, 32 = no blocks in part2)
    # Part2 must include block 31 (the last global attention block) for output collection
    split_results = []

    for split in range(1, num_blocks):
        # Compile fresh for each split point
        def make_part1_fn(s):
            def fn():
                with torch.autocast("cuda", dtype=torch.float16):
                    return backbone.forward_image_part1(img_tensor, split_block=s)
            return fn

        def make_part2_fn(s):
            def fn():
                with torch.autocast("cuda", dtype=torch.float16):
                    return backbone.forward_image_part2(inter, split_block=s)
            return fn

        if use_compile:
            # Compile part1 and part2 for this split
            compiled_part1 = torch.compile(
                backbone.forward_image_part1, mode="default", dynamic=False
            )
            compiled_part2 = torch.compile(
                backbone.forward_image_part2, mode="default", dynamic=False
            )

            def part1_fn():
                with torch.autocast("cuda", dtype=torch.float16):
                    return compiled_part1(img_tensor, split_block=split)

            def part2_fn():
                with torch.autocast("cuda", dtype=torch.float16):
                    return compiled_part2(inter, split_block=split)
        else:
            part1_fn = make_part1_fn(split)
            part2_fn = make_part2_fn(split)

        # Get intermediate tensor for part2
        with torch.inference_mode():
            with torch.autocast("cuda", dtype=torch.float16):
                inter = backbone.forward_image_part1(img_tensor, split_block=split)

        # Benchmark part1
        p1_avg, p1_min, p1_p50 = benchmark_fn(part1_fn, args.warmup, args.runs)

        # Benchmark part2 (using cached intermediate)
        p2_avg, p2_min, p2_p50 = benchmark_fn(part2_fn, args.warmup, args.runs)

        # Clean up compile state
        if use_compile:
            torch._dynamo.reset()
            torch.cuda.empty_cache()

        is_global = split in global_attn_blocks
        marker = " *" if is_global else ""

        split_results.append({
            "split": split,
            "p1_avg": p1_avg, "p1_min": p1_min, "p1_p50": p1_p50,
            "p2_avg": p2_avg, "p2_min": p2_min, "p2_p50": p2_p50,
            "is_global": is_global,
        })

        print(
            f"  split={split:2d}{marker:2s}  "
            f"part1: {p1_p50:6.1f}ms  "
            f"part2+FPN: {p2_p50:6.1f}ms  "
            f"sequential: {p1_p50 + p2_p50:6.1f}ms"
        )

    # ---------------------------------------------------------------
    # Find optimal split for each enc-dec cost
    # ---------------------------------------------------------------
    print(f"\n\n{'='*80}")
    print("OPTIMIZATION RESULTS")
    print(f"{'='*80}")

    # Also compute full backbone time
    def full_backbone_fn():
        with torch.autocast("cuda", dtype=torch.float16):
            return backbone.forward_image(img_tensor)

    if use_compile:
        compiled_full = torch.compile(
            backbone.forward_image, mode="default", dynamic=False
        )

        def full_fn():
            with torch.autocast("cuda", dtype=torch.float16):
                return compiled_full(img_tensor)

        full_avg, full_min, full_p50 = benchmark_fn(full_fn, args.warmup, args.runs)
        torch._dynamo.reset()
        torch.cuda.empty_cache()
    else:
        full_avg, full_min, full_p50 = benchmark_fn(
            full_backbone_fn, args.warmup, args.runs
        )

    print(f"\n  Full backbone: avg={full_avg:.1f}ms  min={full_min:.1f}ms  p50={full_p50:.1f}ms")

    for enc_label, enc_dec_ms in enc_dec_costs.items():
        print(f"\n  --- Enc-dec: {enc_label} = {enc_dec_ms:.1f}ms ---")
        print()

        # Current (no split): max(full_backbone, enc_dec)
        current_frame = max(full_p50, enc_dec_ms)
        current_fps = 1000.0 / current_frame
        print(f"  Current (no split): {current_frame:.1f}ms/frame = {current_fps:.1f} FPS")

        # For each split point, compute pipeline frame time
        best_split = None
        best_frame = float("inf")

        print()
        print(f"  {'split':>5s}  {'part1':>7s}  {'part2+enc_dec':>13s}  {'frame_time':>10s}  {'FPS':>6s}  {'speedup':>7s}")
        print(f"  {'-'*5}  {'-'*7}  {'-'*13}  {'-'*10}  {'-'*6}  {'-'*7}")

        for r in split_results:
            s = r["split"]
            p1 = r["p1_p50"]
            p2_plus_enc = r["p2_p50"] + enc_dec_ms
            frame_time = max(p1, p2_plus_enc)
            fps = 1000.0 / frame_time
            speedup = current_frame / frame_time
            marker = " *" if r["is_global"] else ""

            if frame_time < best_frame:
                best_frame = frame_time
                best_split = s

            indicator = " <-- BEST" if s == best_split and frame_time == best_frame else ""
            print(
                f"  {s:5d}{marker:2s} {p1:7.1f}ms  {p2_plus_enc:10.1f}ms  "
                f"{frame_time:7.1f}ms  {fps:6.1f}  {speedup:6.2f}x{indicator}"
            )

        best_r = next(r for r in split_results if r["split"] == best_split)
        best_fps = 1000.0 / best_frame
        improvement = (best_fps - current_fps) / current_fps * 100

        print(f"\n  OPTIMAL: split_block={best_split}")
        print(f"    Part 1 (blocks 0-{best_split-1}): {best_r['p1_p50']:.1f}ms")
        print(f"    Part 2 (blocks {best_split}-31) + FPN + enc-dec: "
              f"{best_r['p2_p50']:.1f} + {enc_dec_ms:.1f} = {best_r['p2_p50'] + enc_dec_ms:.1f}ms")
        print(f"    Frame time: {best_frame:.1f}ms = {best_fps:.1f} FPS "
              f"(+{improvement:.0f}% vs current {current_fps:.1f} FPS)")


if __name__ == "__main__":
    main()
