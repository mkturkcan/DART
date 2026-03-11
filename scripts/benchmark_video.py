#!/usr/bin/env python3
"""Benchmark SAM3 video: sequential vs pipelined execution.

Compares two execution modes:
  - Sequential: backbone(N) → enc-dec(N) → backbone(N+1) → enc-dec(N+1)
  - Pipelined:  backbone(N+1) || enc-dec(N) on separate CUDA streams

Supports both TRT backbone and torch.compile backbone.

Usage:
    # TRT backbone — compare sequential vs pipelined:
    python scripts/benchmark_video.py \
        --video input.mp4 \
        --classes person car \
        --checkpoint sam3.pt \
        --trt hf_backbone_1008_fp16.engine \
        --trt-enc-dec enc_dec_fp16.engine \
        --max-frames 100

    # torch.compile backbone — compare sequential vs pipelined:
    python scripts/benchmark_video.py \
        --video input.mp4 \
        --classes person car \
        --checkpoint sam3.pt \
        --compile max-autotune \
        --trt-enc-dec enc_dec_fp16.engine \
        --max-frames 100

    # Sequential only:
    python scripts/benchmark_video.py \
        --video input.mp4 --coco \
        --checkpoint sam3.pt \
        --trt hf_backbone_644_masked_fp16.engine \
        --trt-enc-dec enc_dec_644_coco80_fp16.engine --trt-max-classes 80 \
        --imgsz 644 --mode sequential --max-frames 100
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sam3.model_builder import (
    build_pruned_sam3_image_model,
    build_sam3_image_model,
    load_pruned_config,
)
from sam3.model.sam3_multiclass_fast import Sam3MultiClassPredictorFast


def draw_detections(frame_bgr, results, class_names, tracks=None):
    """Draw boxes and labels on an OpenCV BGR frame (in-place)."""
    COLOURS = [
        (75, 25, 230), (75, 180, 60), (200, 130, 0), (25, 225, 255),
        (48, 130, 245), (180, 30, 145), (240, 240, 70), (230, 50, 240),
    ]

    if tracks is not None:
        for track in tracks:
            cls_idx = track.class_id
            cls_name = class_names[cls_idx] if cls_idx < len(class_names) else "?"
            colour = COLOURS[cls_idx % len(COLOURS)]
            box = track.box_xyxy
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), colour, 2)
            label = f"#{track.track_id} {cls_name} {track.score:.2f}"
            (tw, th), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(
                frame_bgr, (x1, max(y1 - th - 6, 0)),
                (x1 + tw + 4, max(y1, th + 6)), colour, -1,
            )
            cv2.putText(
                frame_bgr, label, (x1 + 2, max(y1 - 4, th + 2)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
            )
    else:
        for i in range(len(results["scores"])):
            cls_name = results["class_names"][i]
            score = results["scores"][i].item()
            box = results["boxes"][i].cpu().tolist()
            cls_idx = class_names.index(cls_name) if cls_name in class_names else 0
            colour = COLOURS[cls_idx % len(COLOURS)]
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), colour, 2)
            label = f"{cls_name} {score:.2f}"
            (tw, th), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(
                frame_bgr, (x1, max(y1 - th - 6, 0)),
                (x1 + tw + 4, max(y1, th + 6)), colour, -1,
            )
            cv2.putText(
                frame_bgr, label, (x1 + 2, max(y1 - 4, th + 2)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
            )
    return frame_bgr


@torch.inference_mode()
def run_sequential(predictor, video_path, max_frames, confidence, nms,
                   tracker=None, writer=None, display=False, class_names=None):
    """Run video with sequential backbone → enc-dec (no stream overlap)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames > 0:
        total = min(total, max_frames)

    frame_times = []
    backbone_times = []
    predict_times = []

    frame_idx = 0
    while True:
        ret, frame_bgr = cap.read()
        if not ret or (max_frames > 0 and frame_idx >= max_frames):
            break

        torch.cuda.synchronize()
        t0 = time.perf_counter()

        # Backbone
        state = predictor.set_image(
            Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        )
        torch.cuda.synchronize()
        t_bb = time.perf_counter()

        # Enc-dec + postprocess
        results = predictor.predict(
            state,
            confidence_threshold=confidence,
            nms_threshold=nms,
        )
        torch.cuda.synchronize()
        t_end = time.perf_counter()

        bb_ms = (t_bb - t0) * 1000
        pred_ms = (t_end - t_bb) * 1000
        total_ms = (t_end - t0) * 1000

        frame_times.append(total_ms)
        backbone_times.append(bb_ms)
        predict_times.append(pred_ms)

        # Tracking
        n_dets = len(results["scores"])
        tracks = None
        if tracker is not None and n_dets > 0:
            boxes_np = results["boxes"].cpu().numpy()
            scores_np = results["scores"].cpu().numpy()
            class_ids_np = results["class_ids"].cpu().numpy()
            tracks = tracker.update(boxes_np, scores_np, class_ids_np)
        elif tracker is not None:
            tracks = tracker.update(
                np.empty((0, 4), dtype=np.float32),
                np.empty(0, dtype=np.float32),
                np.empty(0, dtype=np.int64),
            )

        if frame_idx % 30 == 0:
            print(f"  Frame {frame_idx}: {total_ms:.1f}ms "
                  f"(bb={bb_ms:.1f} + pred={pred_ms:.1f}), {n_dets} dets")

        if writer is not None or display:
            annotated = draw_detections(
                frame_bgr.copy(), results, class_names, tracks=tracks,
            )
            if writer is not None:
                writer.write(annotated)
            if display:
                cv2.imshow("SAM3 Sequential", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        frame_idx += 1

    cap.release()

    # Compute stats (skip frame 0 for steady-state)
    n = len(frame_times)
    if n <= 1:
        steady = frame_times
        steady_bb = backbone_times
        steady_pred = predict_times
    else:
        steady = frame_times[1:]
        steady_bb = backbone_times[1:]
        steady_pred = predict_times[1:]

    return {
        "total_frames": n,
        "first_frame_ms": frame_times[0] if n > 0 else 0,
        "steady_avg_ms": sum(steady) / len(steady) if steady else 0,
        "steady_bb_ms": sum(steady_bb) / len(steady_bb) if steady_bb else 0,
        "steady_pred_ms": sum(steady_pred) / len(steady_pred) if steady_pred else 0,
        "steady_state_fps": 1000.0 / (sum(steady) / len(steady)) if steady else 0,
    }


def run_pipelined(predictor, video_path, max_frames, confidence, nms,
                  backbone_engine_path=None, split_backbone=False,
                  split_block=20, cuda_graphs=False,
                  trt_split_backbone=None,
                  tracker=None, writer=None, display=False, class_names=None):
    """Run video with pipelined backbone || enc-dec (CUDA stream overlap).

    Uses TRTSplitPipelinedVideoProcessor for split TRT backbone,
    PipelinedVideoProcessor for full TRT backbone,
    SplitBackboneVideoProcessor for split compile backbone,
    or CompiledVideoProcessor for torch.compile backbone.
    """
    if trt_split_backbone is not None:
        from sam3.video_pipeline import TRTSplitPipelinedVideoProcessor
        pipeline = TRTSplitPipelinedVideoProcessor(
            predictor=predictor,
            split_backbone=trt_split_backbone,
        )
    elif split_backbone:
        from sam3.video_pipeline import SplitBackboneVideoProcessor
        pipeline = SplitBackboneVideoProcessor(
            predictor=predictor,
            split_block=split_block,
            cuda_graphs=cuda_graphs,
        )
        # Warmup split pipeline
        use_fp16 = predictor.use_fp16
        dummy_img = Image.new("RGB", (predictor.resolution, predictor.resolution))
        with torch.inference_mode():
            from torchvision.transforms import v2
            dummy_tensor = v2.functional.to_image(dummy_img).to(predictor.device)
            dummy_tensor = predictor.transform(dummy_tensor).unsqueeze(0)
            for _ in range(3):
                with torch.autocast("cuda", dtype=torch.float16, enabled=use_fp16):
                    inter = pipeline._part1_fn(dummy_tensor, split_block)
                    inter = {"x": inter["x"].clone(), "s": inter["s"]}
                    bb_out = pipeline._part2_fn(inter, split_block)
                state = {
                    "backbone_out": bb_out,
                    "original_height": predictor.resolution,
                    "original_width": predictor.resolution,
                }
                predictor.predict(state, confidence_threshold=0.5)
        torch.cuda.synchronize()
    elif backbone_engine_path is not None:
        from sam3.video_pipeline import PipelinedVideoProcessor
        pipeline = PipelinedVideoProcessor(
            predictor=predictor,
            backbone_engine_path=backbone_engine_path,
        )
    else:
        from sam3.video_pipeline import CompiledVideoProcessor
        pipeline = CompiledVideoProcessor(predictor=predictor)

    def on_frame(frame_idx, results, frame_bgr):
        n_dets = len(results["scores"])

        tracks = None
        if tracker is not None and n_dets > 0:
            boxes_np = results["boxes"].cpu().numpy()
            scores_np = results["scores"].cpu().numpy()
            class_ids_np = results["class_ids"].cpu().numpy()
            tracks = tracker.update(boxes_np, scores_np, class_ids_np)
        elif tracker is not None:
            tracks = tracker.update(
                np.empty((0, 4), dtype=np.float32),
                np.empty(0, dtype=np.float32),
                np.empty(0, dtype=np.int64),
            )

        if frame_idx % 30 == 0:
            print(f"  Frame {frame_idx}: {n_dets} dets")

        if writer is not None or display:
            annotated = draw_detections(
                frame_bgr.copy(), results, class_names, tracks=tracks,
            )
            if writer is not None:
                writer.write(annotated)
            if display:
                cv2.imshow("SAM3 Pipelined", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    raise KeyboardInterrupt

    stats = pipeline.process_video(
        video_path,
        confidence_threshold=confidence,
        nms_threshold=nms,
        callback=on_frame,
        max_frames=max_frames,
    )
    return stats


def print_stats(label, stats, enc_dec_type):
    """Print results for a single benchmark run."""
    n = stats.get("total_frames", 0)
    print(f"\n  [{label}]")
    print(f"  Frames:           {n}")
    if "first_frame_ms" in stats:
        print(f"  First frame:      {stats['first_frame_ms']:.1f}ms")
    if "steady_bb_ms" in stats:
        print(f"  Backbone avg:     {stats['steady_bb_ms']:.1f}ms")
    if "steady_pred_ms" in stats:
        print(f"  Enc-dec avg:      {stats['steady_pred_ms']:.1f}ms")
    if "steady_avg_ms" in stats:
        print(f"  Total avg:        {stats['steady_avg_ms']:.1f}ms/frame")
    if "steady_state_fps" in stats:
        print(f"  Steady-state FPS: {stats['steady_state_fps']:.1f}")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark SAM3 video: sequential vs pipelined",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--video", required=True, help="Input video file")
    parser.add_argument(
        "--classes", nargs="+", type=str,
        default=["car", "pedestrian", "bicycle"],
        help="Target class names",
    )
    parser.add_argument(
        "--coco", action="store_true",
        help="Use all 80 COCO classes (overrides --classes)",
    )
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument(
        "--trt", type=str, default=None, metavar="ENGINE",
        help="TRT backbone engine path (use instead of --compile)",
    )
    parser.add_argument(
        "--compile", type=str, default=None,
        choices=["default", "reduce-overhead", "max-autotune"],
        help="torch.compile mode for backbone (use instead of --trt)",
    )
    parser.add_argument("--trt-enc-dec", type=str, default=None, metavar="ENGINE")
    parser.add_argument("--trt-max-classes", type=int, default=4)
    parser.add_argument("--confidence", type=float, default=0.3)
    parser.add_argument("--nms", type=float, default=0.7)
    parser.add_argument("--output", "-o", type=str, default=None)
    parser.add_argument("--display", action="store_true")
    parser.add_argument("--imgsz", type=int, default=1008)
    parser.add_argument("--max-frames", type=int, default=100)
    parser.add_argument("--text-cache", type=str, default=None, metavar="PATH")
    parser.add_argument(
        "--mode", type=str, default="both",
        choices=["sequential", "pipelined", "both"],
        help="Benchmark mode: sequential, pipelined, or both (default: both)",
    )
    parser.add_argument(
        "--mask-blocks", type=str, default=None,
        help="Comma-separated sub-block pruning spec (e.g. '25:attn,28:mlp')",
    )
    parser.add_argument(
        "--split-backbone", action="store_true",
        help="Split ViT backbone across pipeline stages (requires --compile)",
    )
    parser.add_argument(
        "--split-block", type=int, default=20,
        help="Block index to split ViT at (default 20)",
    )
    parser.add_argument(
        "--cuda-graphs", action="store_true",
        help="Use manual CUDA graph capture with split backbone",
    )
    parser.add_argument(
        "--trt-part1", type=str, default=None, metavar="ENGINE",
        help="TRT Part1 engine for split backbone pipeline",
    )
    parser.add_argument(
        "--trt-part2", type=str, default=None, metavar="ENGINE",
        help="TRT Part2 engine for split backbone pipeline",
    )
    # Tracking
    parser.add_argument("--track", action="store_true", help="Enable ByteTrack")
    parser.add_argument("--track-thresh", type=float, default=0.5)
    parser.add_argument("--match-thresh", type=float, default=0.5)
    parser.add_argument("--max-time-lost", type=int, default=30)
    parser.add_argument("--class-agnostic-nms", type=float, default=None, metavar="THRESH",
                        help="Class-agnostic NMS threshold applied before tracking. Disabled by default.")
    args = parser.parse_args()

    if args.coco:
        from sam3.coco_classes import COCO_CLASSES
        args.classes = COCO_CLASSES

    if args.imgsz % 14 != 0:
        print(f"ERROR: --imgsz must be divisible by 14, got {args.imgsz}")
        sys.exit(1)

    if args.trt and args.compile:
        print("ERROR: --trt and --compile are mutually exclusive")
        sys.exit(1)

    if args.split_backbone and args.compile is None:
        print("ERROR: --split-backbone requires --compile MODE")
        sys.exit(1)

    if args.split_backbone and args.trt:
        print("ERROR: --split-backbone is not compatible with --trt")
        sys.exit(1)

    if args.trt_part1 and not args.trt_part2:
        print("ERROR: --trt-part1 requires --trt-part2")
        sys.exit(1)

    # Determine backbone mode
    if args.trt_part1 and args.trt_part2:
        backbone_mode = "trt-split"
        backbone_label = (f"TRT split ({os.path.basename(args.trt_part1)} + "
                          f"{os.path.basename(args.trt_part2)})")
    elif args.trt:
        backbone_mode = "trt"
        backbone_label = f"TRT ({os.path.basename(args.trt)})"
    elif args.compile:
        backbone_mode = "compile"
        backbone_label = f"torch.compile({args.compile})"
    else:
        backbone_mode = "pytorch"
        backbone_label = "PyTorch (eager)"

    run_seq = args.mode in ("sequential", "both")
    run_pipe = args.mode in ("pipelined", "both")

    device = "cuda"
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Backbone: {backbone_label}")
    print(f"Enc-dec:  {'TRT FP16' if args.trt_enc_dec else 'PyTorch'}")
    print(f"Classes:  {len(args.classes)}")
    print(f"Mode:     {args.mode}")
    print(f"Frames:   {args.max_frames}")
    print(f"Resolution: {args.imgsz}")

    # --- Load model ---
    text_cache_exists = args.text_cache and os.path.exists(args.text_cache)
    use_stub = (
        text_cache_exists
        and args.trt_enc_dec
        and args.checkpoint is None
    )

    # Parse mask_blocks
    mask_blocks = None
    if args.mask_blocks:
        mask_blocks = [s.strip() for s in args.mask_blocks.split(",")]

    if use_stub:
        from sam3.model.sam3_multiclass_fast import _TRTModelStub
        print("Using lightweight model stub (text from cache, no checkpoint)")
        model = _TRTModelStub(device=device)
    else:
        if args.checkpoint is None and not text_cache_exists:
            print("NOTE: No --checkpoint, will attempt HuggingFace download")
        skip_msg = f", mask_blocks={mask_blocks}" if mask_blocks else ""
        print(f"Loading SAM3 model...{skip_msg}")
        pruned_config = (
            load_pruned_config(args.checkpoint) if args.checkpoint else None
        )
        if pruned_config is not None:
            model = build_pruned_sam3_image_model(
                checkpoint_path=args.checkpoint,
                pruning_config=pruned_config,
                device=device, eval_mode=True,
            )
            if model.transformer.decoder.presence_token is not None:
                model.transformer.decoder.presence_token = None
        else:
            model = build_sam3_image_model(
                device=device, checkpoint_path=args.checkpoint, eval_mode=True,
                mask_blocks=mask_blocks,
            )

    # Precompute position encoding for non-default resolution
    if not use_stub and args.imgsz != 1008:
        pos_enc = model.backbone.vision_backbone.position_encoding
        pos_enc.precompute_for_resolution(args.imgsz)

    # --- Create predictor ---
    # For trt-split mode, don't pass --trt (split backbone is separate)
    predictor = Sam3MultiClassPredictorFast(
        model, device=device,
        resolution=args.imgsz,
        compile_mode=args.compile if backbone_mode == "compile" else None,
        trt_engine_path=args.trt if backbone_mode != "trt-split" else None,
        use_fp16=True,
        detection_only=True,
        trt_enc_dec_engine_path=args.trt_enc_dec,
        trt_max_classes=args.trt_max_classes,
    )

    # For trt-split: create split backbone and inject as backbone_fn
    trt_split_bb = None
    if backbone_mode == "trt-split":
        from sam3.trt.trt_backbone import TRTSplitBackbone
        pos_module = model.backbone.vision_backbone.position_encoding
        trt_split_bb = TRTSplitBackbone(
            part1_engine_path=args.trt_part1,
            part2_engine_path=args.trt_part2,
            device=device,
            pos_encoding_module=pos_module,
        )
        # Inject split backbone's forward_image as the predictor's backbone fn
        predictor._backbone_fn = trt_split_bb.forward_image
        predictor._trt_backbone = trt_split_bb

    print(f"Setting {len(args.classes)} classes ...")
    predictor.set_classes(args.classes, text_cache=args.text_cache)

    # --- Warmup ---
    if backbone_mode == "compile":
        print(f"\nWarming up torch.compile({args.compile})...")
        print("  This takes 60-120s on first run.")
    else:
        print(f"\nWarming up ({backbone_label})...")
    t_warmup = time.perf_counter()

    predictor._ensure_compiled()
    dummy_img = Image.new("RGB", (args.imgsz, args.imgsz))
    with torch.inference_mode():
        state = predictor.set_image(dummy_img)
        predictor.predict(state, confidence_threshold=0.5)
        for _ in range(2):
            state = predictor.set_image(dummy_img)
            predictor.predict(state, confidence_threshold=0.5)
    torch.cuda.synchronize()
    warmup_s = time.perf_counter() - t_warmup
    print(f"  Warmup done ({warmup_s:.0f}s)")

    # --- Create tracker factory ---
    def make_tracker():
        if not args.track:
            return None
        from sam3.tracking import BYTETracker
        ca_nms = args.class_agnostic_nms if args.class_agnostic_nms is not None else 1.0
        return BYTETracker(
            track_thresh=args.track_thresh,
            match_thresh=args.match_thresh,
            max_time_lost=args.max_time_lost,
            class_agnostic_nms_thresh=ca_nms,
        )

    enc_dec_type = "TRT FP16" if args.trt_enc_dec else "PyTorch"
    all_stats = {}

    # --- Sequential benchmark ---
    if run_seq:
        print(f"\n{'='*55}")
        print(f"SEQUENTIAL: backbone -> enc-dec (no overlap)")
        print(f"{'='*55}")
        tracker = make_tracker()

        seq_stats = run_sequential(
            predictor, args.video, args.max_frames,
            args.confidence, args.nms,
            tracker=tracker,
            class_names=args.classes,
        )
        all_stats["sequential"] = seq_stats
        print_stats("Sequential", seq_stats, enc_dec_type)

    # --- Pipelined benchmark ---
    if run_pipe:
        print(f"\n{'='*55}")
        if backbone_mode == "trt-split":
            print(f"TRT-SPLIT-PIPELINED: part1(N+1) || (part2(N) + enc-dec(N))")
        elif args.split_backbone:
            print(f"SPLIT-PIPELINED: part1(N+1) || (part2(N) + enc-dec(N))")
            print(f"  split_block={args.split_block}, cuda_graphs={args.cuda_graphs}")
        else:
            print(f"PIPELINED: backbone(N+1) || enc-dec(N) (CUDA streams)")
        print(f"{'='*55}")
        tracker = make_tracker()

        # Video writer only for pipelined (or if only running pipelined)
        writer = None
        if args.output:
            cap_tmp = cv2.VideoCapture(args.video)
            fps_out = cap_tmp.get(cv2.CAP_PROP_FPS)
            w_out = int(cap_tmp.get(cv2.CAP_PROP_FRAME_WIDTH))
            h_out = int(cap_tmp.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap_tmp.release()
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(args.output, fourcc, fps_out, (w_out, h_out))

        try:
            pipe_stats = run_pipelined(
                predictor, args.video, args.max_frames,
                args.confidence, args.nms,
                backbone_engine_path=args.trt,
                split_backbone=args.split_backbone,
                split_block=args.split_block,
                cuda_graphs=args.cuda_graphs,
                trt_split_backbone=trt_split_bb,
                tracker=tracker, writer=writer,
                display=args.display, class_names=args.classes,
            )
        except KeyboardInterrupt:
            print("\nStopped by user.")
            pipe_stats = {"total_frames": 0, "steady_avg_ms": 0, "steady_state_fps": 0}

        if writer is not None:
            writer.release()

        all_stats["pipelined"] = pipe_stats
        print_stats("Pipelined", pipe_stats, enc_dec_type)

    # --- Comparison table ---
    if run_seq and run_pipe:
        seq = all_stats["sequential"]
        pipe = all_stats["pipelined"]
        seq_ms = seq.get("steady_avg_ms", 0)
        pipe_ms = pipe.get("steady_avg_ms", 0)
        seq_fps = seq.get("steady_state_fps", 0)
        pipe_fps = pipe.get("steady_state_fps", 0)
        speedup = seq_ms / pipe_ms if pipe_ms > 0 else 0

        print(f"\n{'='*55}")
        print(f"COMPARISON")
        print(f"{'='*55}")
        print(f"  {'Mode':<15} {'ms/frame':>10} {'FPS':>8} {'Speedup':>10}")
        print(f"  {'-'*15} {'-'*10} {'-'*8} {'-'*10}")
        print(f"  {'Sequential':<15} {seq_ms:>10.1f} {seq_fps:>8.1f} {'1.00x':>10}")
        print(f"  {'Pipelined':<15} {pipe_ms:>10.1f} {pipe_fps:>8.1f} {speedup:>9.2f}x")
        if seq.get("steady_bb_ms"):
            print(f"\n  Sequential breakdown:")
            print(f"    Backbone:  {seq['steady_bb_ms']:.1f}ms")
            print(f"    Enc-dec:   {seq['steady_pred_ms']:.1f}ms")
            print(f"    Total:     {seq_ms:.1f}ms (sum)")
            print(f"  Pipelined:   {pipe_ms:.1f}ms (max of backbone, enc-dec)")
        print(f"{'='*55}")

    if args.display:
        cv2.destroyAllWindows()
    if args.output and run_pipe:
        print(f"\nOutput saved: {args.output}")


if __name__ == "__main__":
    main()
