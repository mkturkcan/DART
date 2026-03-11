#!/usr/bin/env python3
"""
Demo: Pipelined video processing with SAM3.

Overlaps backbone(frame N+1) with encoder-decoder(frame N) on separate
CUDA streams for reduced per-frame latency.

Three backbone modes:
  --trt ENGINE        TRT backbone (fastest, requires FP16 engine)
  --compile MODE      torch.compile backbone (correct accuracy, ~75ms)
  --split-backbone    Split ViT at block 24 for better pipeline balance (~58ms)

Optional tracking with ByteTrack (--track) for persistent object IDs.

Usage:
    # TRT backbone + TRT enc-dec (fastest, if accuracy is acceptable):
    python demo_video.py --video input.mp4 \
        --classes person car bicycle \
        --checkpoint sam3.pt --trt backbone_fp16.engine \
        --trt-enc-dec enc_dec_fp16.engine --output output.mp4

    # torch.compile backbone + TRT enc-dec (correct accuracy):
    python demo_video.py --video input.mp4 \
        --classes person car bicycle \
        --checkpoint sam3.pt --compile max-autotune \
        --trt-enc-dec enc_dec_fp16.engine --output output.mp4

    # With ByteTrack tracking:
    python demo_video.py --video input.mp4 \
        --classes person car bicycle \
        --checkpoint sam3.pt --compile max-autotune \
        --trt-enc-dec enc_dec_fp16.engine --output output.mp4 --track
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np
import torch
from PIL import Image

from sam3.model_builder import (
    build_pruned_sam3_image_model,
    build_sam3_image_model,
    load_pruned_config,
)
from sam3.efficient_backbone import build_efficientsam3_model
from sam3.model.sam3_multiclass_fast import Sam3MultiClassPredictorFast
from demo_multiclass import CLASS_COLOURS


def draw_detections_cv2(frame_bgr, results, class_names, tracks=None):
    """Draw boxes and labels on an OpenCV BGR frame (in-place, fast).

    Args:
        frame_bgr: BGR frame to annotate.
        results: Detection results dict from predictor.predict().
        class_names: Ordered list of class names for colour assignment.
        tracks: Optional list of STrack objects from ByteTrack. When
            provided, draws track IDs instead of raw detections.
    """
    n_colours = len(CLASS_COLOURS)
    class_to_colour = {
        name: CLASS_COLOURS[i % n_colours] for i, name in enumerate(class_names)
    }

    if tracks is not None:
        # Draw tracked objects with persistent IDs
        for track in tracks:
            cls_idx = track.class_id
            cls_name = class_names[cls_idx] if cls_idx < len(class_names) else "?"
            colour_rgb = class_to_colour.get(cls_name, CLASS_COLOURS[0])
            colour_bgr = (colour_rgb[2], colour_rgb[1], colour_rgb[0])

            box = track.box_xyxy
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), colour_bgr, 2)

            label = f"#{track.track_id} {cls_name} {track.score:.2f}"
            (tw, th), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(
                frame_bgr, (x1, max(y1 - th - 6, 0)),
                (x1 + tw + 4, max(y1, th + 6)), colour_bgr, -1,
            )
            cv2.putText(
                frame_bgr, label, (x1 + 2, max(y1 - 4, th + 2)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
            )
    else:
        # Draw raw detections (no tracking)
        for i in range(len(results["scores"])):
            cls_name = results["class_names"][i]
            score = results["scores"][i].item()
            box = results["boxes"][i].cpu().tolist()
            colour_rgb = class_to_colour.get(cls_name, CLASS_COLOURS[0])
            colour_bgr = (colour_rgb[2], colour_rgb[1], colour_rgb[0])

            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), colour_bgr, 2)

            label = f"{cls_name} {score:.2f}"
            (tw, th), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(
                frame_bgr, (x1, max(y1 - th - 6, 0)),
                (x1 + tw + 4, max(y1, th + 6)), colour_bgr, -1,
            )
            cv2.putText(
                frame_bgr, label, (x1 + 2, max(y1 - 4, th + 2)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
            )

    return frame_bgr


def main():
    parser = argparse.ArgumentParser(
        description="SAM3 pipelined video processing"
    )
    parser.add_argument(
        "--video", required=True, help="Input video file path"
    )
    parser.add_argument(
        "--classes", nargs="+", type=str,
        default=["car", "pedestrian", "bicycle"],
        help="Target class names",
    )
    parser.add_argument(
        "--coco", action="store_true",
        help="Use all 80 COCO classes (overrides --classes)",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Model checkpoint path (default: download from HF)",
    )
    parser.add_argument(
        "--efficient-backbone", type=str, default=None,
        choices=["efficientvit", "repvit", "tinyvit"],
        help="Use lightweight EfficientSAM3 backbone instead of ViT-H",
    )
    parser.add_argument(
        "--efficient-model", type=str, default=None,
        help="Backbone variant (e.g. b0/b1/b2, m0_9/m1_1/m2_3, 5m/11m/21m)",
    )
    # Backbone mode: TRT or torch.compile (at least one required)
    parser.add_argument(
        "--trt", type=str, default=None, metavar="ENGINE",
        help="TRT backbone engine (enables TRT pipelining)",
    )
    parser.add_argument(
        "--compile", type=str, default=None, metavar="MODE",
        choices=["default", "reduce-overhead", "max-autotune"],
        help="torch.compile mode for backbone (e.g. max-autotune)",
    )
    parser.add_argument(
        "--trt-enc-dec", type=str, default=None, metavar="ENGINE",
        help="TRT encoder-decoder engine (optional)",
    )
    parser.add_argument(
        "--trt-max-classes", type=int, default=4,
        help="Max classes for enc-dec TRT engine",
    )
    parser.add_argument(
        "--confidence", type=float, default=0.3,
        help="Detection confidence threshold",
    )
    parser.add_argument(
        "--nms", type=float, default=0.7,
        help="NMS IoU threshold",
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Output video file path (optional)",
    )
    parser.add_argument(
        "--display", action="store_true",
        help="Show live preview with cv2.imshow",
    )
    parser.add_argument(
        "--imgsz", type=int, default=1008,
        help="Input resolution (default 1008, must be divisible by 14)",
    )
    parser.add_argument(
        "--max-frames", type=int, default=0,
        help="Stop after N frames (0 = all)",
    )
    parser.add_argument(
        "--split-backbone", action="store_true",
        help="Split ViT backbone for better pipeline balance. "
             "Requires --compile (not --trt). Up to 67%% throughput improvement.",
    )
    parser.add_argument(
        "--split-block", type=int, default=20,
        help="Block index to split ViT at (default 20, optimal for TRT enc-dec). "
             "Use 24 if not using TRT enc-dec.",
    )
    parser.add_argument(
        "--cuda-graphs", action="store_true",
        help="Use manual CUDA graph capture with split backbone. "
             "Equivalent to max-autotune performance without cross-function conflicts.",
    )
    parser.add_argument(
        "--text-cache", type=str, default=None, metavar="PATH",
        help="Path to cached text embeddings (.pt).",
    )
    # Tracking
    parser.add_argument(
        "--track", action="store_true",
        help="Enable ByteTrack multi-object tracking",
    )
    parser.add_argument(
        "--track-thresh", type=float, default=0.5,
        help="ByteTrack: high/low score threshold (default 0.5)",
    )
    parser.add_argument(
        "--match-thresh", type=float, default=0.5,
        help="ByteTrack: IoU matching threshold (default 0.5)",
    )
    parser.add_argument(
        "--max-time-lost", type=int, default=30,
        help="ByteTrack: frames before removing lost track (default 30)",
    )
    parser.add_argument(
        "--class-agnostic-nms", type=float, default=None, metavar="THRESH",
        help="Class-agnostic NMS threshold applied before tracking. Suppresses "
             "overlapping detections of different classes (e.g. car/suv on same "
             "object). Disabled by default; pass a threshold (e.g. 0.7) to enable.",
    )
    parser.add_argument(
        "--skip-blocks", type=str, default=None, metavar="INDICES",
        help="Comma-separated ViT block indices to skip for block pruning "
             "(e.g. '1,3,9,11,17,19,25,27' skips 8 window blocks). "
             "Cannot skip global attention blocks [7,15,23,31]."
    )
    parser.add_argument(
        "--mask-blocks", type=str, default=None, metavar="SPEC",
        help="Fine-grained sub-block pruning from BlockPruner search. "
             "Comma-separated 'idx:type' pairs (e.g. '0:attn,1:mlp,2:attn'). "
             "Run scripts/block_pruner_search.py to find optimal pruning order."
    )
    args = parser.parse_args()

    if args.coco:
        from sam3.coco_classes import COCO_CLASSES
        args.classes = COCO_CLASSES

    if args.imgsz % 14 != 0:
        print(f"ERROR: --imgsz must be divisible by 14, got {args.imgsz}")
        sys.exit(1)

    if args.trt is None and args.compile is None and not args.efficient_backbone:
        print("ERROR: Specify --trt ENGINE or --compile MODE (or both)")
        sys.exit(1)

    if args.split_backbone and args.compile is None:
        print("ERROR: --split-backbone requires --compile MODE")
        sys.exit(1)

    if args.split_backbone and args.trt:
        print("ERROR: --split-backbone is not compatible with --trt")
        sys.exit(1)

    if args.efficient_backbone and not args.efficient_model:
        print("ERROR: --efficient-backbone requires --efficient-model")
        sys.exit(1)

    if args.efficient_backbone and args.split_backbone:
        print("ERROR: --split-backbone is not compatible with --efficient-backbone "
              "(student backbone has no ViT blocks to split)")
        sys.exit(1)

    # Parse skip_blocks
    skip_blocks = None
    if args.skip_blocks:
        skip_blocks = set(int(x.strip()) for x in args.skip_blocks.split(","))
        if args.efficient_backbone:
            print("ERROR: --skip-blocks is not compatible with --efficient-backbone")
            sys.exit(1)

    # Parse mask_blocks (fine-grained sub-block pruning)
    mask_blocks = None
    if args.mask_blocks:
        mask_blocks = [s.strip() for s in args.mask_blocks.split(",")]
        if args.efficient_backbone:
            print("ERROR: --mask-blocks is not compatible with --efficient-backbone")
            sys.exit(1)

    device = "cuda"
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    # --- Determine backbone mode ---
    if args.efficient_backbone:
        bb_tag = f"{args.efficient_backbone}/{args.efficient_model}"
        if args.trt:
            backbone_mode = "trt"
            print(f"Backbone: {bb_tag} + TRT ({args.trt})")
        elif args.compile:
            backbone_mode = "compile"
            print(f"Backbone: {bb_tag} + torch.compile({args.compile})")
        else:
            backbone_mode = "compile"
            print(f"Backbone: {bb_tag} (no compile)")
    elif args.trt:
        backbone_mode = "trt"
        print(f"Backbone: TRT ({args.trt})")
    elif args.split_backbone:
        backbone_mode = "split"
        cg_str = " + CUDA graphs" if args.cuda_graphs else ""
        print(f"Backbone: split @ block {args.split_block} + torch.compile({args.compile}){cg_str}")
    else:
        backbone_mode = "compile"
        print(f"Backbone: torch.compile({args.compile})")

    if args.trt_enc_dec:
        print(f"Enc-dec:  TRT ({args.trt_enc_dec})")
    else:
        print(f"Enc-dec:  PyTorch")

    # --- Load model ---
    text_cache_exists = args.text_cache and os.path.exists(args.text_cache)
    use_trt_only = (
        text_cache_exists
        and args.trt
        and args.trt_enc_dec
        and args.checkpoint is None
    )

    if use_trt_only:
        from sam3.model.sam3_multiclass_fast import _TRTModelStub
        print("Using TRT-only mode (no checkpoint — text from cache)")
        model = _TRTModelStub(device=device)
    elif args.efficient_backbone:
        print(f"Loading EfficientSAM3 ({args.efficient_backbone} {args.efficient_model})...")
        model = build_efficientsam3_model(
            backbone_type=args.efficient_backbone,
            model_name=args.efficient_model,
            checkpoint_path=args.checkpoint,
            device=device,
            eval_mode=True,
        )
    else:
        if args.checkpoint is None and not text_cache_exists:
            print("NOTE: No --checkpoint provided, will attempt HuggingFace download")

        skip_msg = f", skip_blocks={sorted(skip_blocks)}" if skip_blocks else ""
        if mask_blocks:
            skip_msg += f", mask_blocks={mask_blocks}"
        print(f"Loading SAM3 model...{skip_msg}")
        pruned_config = (
            load_pruned_config(args.checkpoint) if args.checkpoint else None
        )
        if pruned_config is not None:
            print(f"  Detected pruned checkpoint: {pruned_config}")
            model = build_pruned_sam3_image_model(
                checkpoint_path=args.checkpoint,
                pruning_config=pruned_config,
                device=device,
                eval_mode=True,
                skip_blocks=skip_blocks,
            )
            if model.transformer.decoder.presence_token is not None:
                model.transformer.decoder.presence_token = None
        else:
            model = build_sam3_image_model(
                device=device,
                checkpoint_path=args.checkpoint,
                eval_mode=True,
                skip_blocks=skip_blocks,
                mask_blocks=mask_blocks,
            )

    # Precompute position encoding buffers for target resolution
    # (must happen before torch.compile warmup for CUDAGraph safety)
    if args.imgsz != 1008:
        pos_enc = model.backbone.vision_backbone.position_encoding
        pos_enc.precompute_for_resolution(args.imgsz)

    # --- Create predictor ---
    predictor = Sam3MultiClassPredictorFast(
        model,
        device=device,
        resolution=args.imgsz,
        use_fp16=True,
        detection_only=True,
        trt_engine_path=args.trt,
        compile_mode=args.compile if backbone_mode == "compile" else None,
        trt_enc_dec_engine_path=args.trt_enc_dec,
        trt_max_classes=args.trt_max_classes,
    )

    print(f"Setting {len(args.classes)} classes: {args.classes}")
    predictor.set_classes(args.classes, text_cache=args.text_cache)

    # --- Create pipeline ---
    if backbone_mode == "trt":
        from sam3.video_pipeline import PipelinedVideoProcessor
        pipeline = PipelinedVideoProcessor(
            predictor=predictor,
            backbone_engine_path=args.trt,
        )
    elif backbone_mode == "split":
        from sam3.video_pipeline import SplitBackboneVideoProcessor

        cg_str = " + CUDA graphs" if args.cuda_graphs else ""
        print(f"\nWarming up split backbone + torch.compile({args.compile}){cg_str}...")
        print("  This takes 60-120s on first run.")
        t_warmup = time.perf_counter()

        split_block = args.split_block
        pipeline = SplitBackboneVideoProcessor(
            predictor=predictor,
            split_block=split_block,
            cuda_graphs=args.cuda_graphs,
        )

        # Warmup: run split pipeline on dummy input
        dummy_img = Image.new("RGB", (args.imgsz, args.imgsz))
        with torch.inference_mode():
            bb = predictor.model.backbone
            from torchvision.transforms import v2
            dummy_tensor = v2.functional.to_image(dummy_img).to(device)
            dummy_tensor = predictor.transform(dummy_tensor).unsqueeze(0)
            with torch.autocast("cuda", dtype=torch.float16):
                inter = pipeline._part1_fn(dummy_tensor, split_block)
                inter = {"x": inter["x"].clone(), "s": inter["s"]}
                bb_out = pipeline._part2_fn(inter, split_block)
            state = {
                "backbone_out": bb_out,
                "original_height": args.imgsz,
                "original_width": args.imgsz,
            }
            predictor.predict(state, confidence_threshold=0.5)
            # Two more warmup passes
            for _ in range(2):
                with torch.autocast("cuda", dtype=torch.float16):
                    inter = pipeline._part1_fn(dummy_tensor, split_block)
                    inter = {"x": inter["x"].clone(), "s": inter["s"]}
                    bb_out = pipeline._part2_fn(inter, split_block)
                state["backbone_out"] = bb_out
                predictor.predict(state, confidence_threshold=0.5)
        torch.cuda.synchronize()
        print(f"  Warmup done ({time.perf_counter() - t_warmup:.0f}s)")
    else:
        from sam3.video_pipeline import CompiledVideoProcessor

        # Warmup torch.compile (takes 60-120s for max-autotune)
        print(f"\nWarming up torch.compile({args.compile})...")
        print("  This takes 60-120s on first run.")
        t_warmup = time.perf_counter()
        predictor._ensure_compiled()
        dummy_img = Image.new("RGB", (args.imgsz, args.imgsz))
        with torch.inference_mode():
            state = predictor.set_image(dummy_img)
            predictor.predict(state, confidence_threshold=0.5)
            # Two more warmup passes
            for _ in range(2):
                state = predictor.set_image(dummy_img)
                predictor.predict(state, confidence_threshold=0.5)
        torch.cuda.synchronize()
        print(f"  Warmup done ({time.perf_counter() - t_warmup:.0f}s)")

        pipeline = CompiledVideoProcessor(predictor=predictor)

    # --- Create tracker ---
    tracker = None
    if args.track:
        from sam3.tracking import BYTETracker
        ca_nms = args.class_agnostic_nms if args.class_agnostic_nms is not None else 1.0
        tracker = BYTETracker(
            track_thresh=args.track_thresh,
            match_thresh=args.match_thresh,
            max_time_lost=args.max_time_lost,
            class_agnostic_nms_thresh=ca_nms,
        )
        ca_label = f", class-agnostic-nms={ca_nms}" if ca_nms < 1.0 else ""
        print(f"Tracking: ByteTrack (thresh={args.track_thresh}, "
              f"match={args.match_thresh}, max_lost={args.max_time_lost}"
              f"{ca_label})")

    # --- Video writer ---
    writer = None
    if args.output:
        cap_tmp = cv2.VideoCapture(args.video)
        fps_out = cap_tmp.get(cv2.CAP_PROP_FPS)
        w_out = int(cap_tmp.get(cv2.CAP_PROP_FRAME_WIDTH))
        h_out = int(cap_tmp.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap_tmp.release()
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output, fourcc, fps_out, (w_out, h_out))
        print(f"Output: {args.output} ({w_out}x{h_out} @ {fps_out:.1f} FPS)")

    # --- Frame callback ---
    frame_count = [0]
    det_count = [0]
    track_count = [0]

    def on_frame(frame_idx, results, frame_bgr):
        n_dets = len(results["scores"])
        frame_count[0] = frame_idx + 1
        det_count[0] += n_dets

        # Run tracker
        tracks = None
        if tracker is not None and n_dets > 0:
            boxes_np = results["boxes"].cpu().numpy()
            scores_np = results["scores"].cpu().numpy()
            class_ids_np = results["class_ids"].cpu().numpy()
            tracks = tracker.update(boxes_np, scores_np, class_ids_np)
            track_count[0] = max(
                track_count[0],
                max((t.track_id for t in tracks), default=0),
            )
        elif tracker is not None:
            tracks = tracker.update(
                np.empty((0, 4), dtype=np.float32),
                np.empty(0, dtype=np.float32),
                np.empty(0, dtype=np.int64),
            )

        # Progress
        if frame_idx % 30 == 0:
            n_tracks = len(tracks) if tracks else 0
            if tracker:
                print(f"  Frame {frame_idx}: {n_dets} dets, "
                      f"{n_tracks} tracks")
            else:
                print(f"  Frame {frame_idx}: {n_dets} detections")

        # Annotate and write/display
        if writer is not None or args.display:
            annotated = draw_detections_cv2(
                frame_bgr.copy(), results, args.classes, tracks=tracks,
            )
            if writer is not None:
                writer.write(annotated)
            if args.display:
                cv2.imshow("SAM3 Video", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    raise KeyboardInterrupt

    # --- Run pipeline ---
    print(f"\nProcessing video...")
    t_total = time.perf_counter()
    try:
        stats = pipeline.process_video(
            args.video,
            confidence_threshold=args.confidence,
            nms_threshold=args.nms,
            callback=on_frame,
            max_frames=args.max_frames,
        )
    except KeyboardInterrupt:
        print("\nStopped by user.")
        stats = {
            "total_frames": frame_count[0],
            "elapsed_s": time.perf_counter() - t_total,
        }
        stats["fps"] = (
            stats["total_frames"] / stats["elapsed_s"]
            if stats["elapsed_s"] > 0 else 0
        )

    if writer is not None:
        writer.release()
    if args.display:
        cv2.destroyAllWindows()

    # --- Print stats ---
    print(f"\n{'='*55}")
    print(f"RESULTS")
    print(f"{'='*55}")
    if args.efficient_backbone:
        bb_label = f"{args.efficient_backbone}/{args.efficient_model}"
        if args.compile:
            bb_label += f" + torch.compile({args.compile})"
    elif backbone_mode == "trt":
        bb_label = f"TRT ({args.trt})"
    elif backbone_mode == "split":
        bb_label = f"split @ block {args.split_block} + torch.compile({args.compile})"
    else:
        bb_label = f"torch.compile({args.compile})"
    ed_label = f"TRT" if args.trt_enc_dec else "PyTorch"
    print(f"  Backbone:            {bb_label}")
    print(f"  Enc-dec:             {ed_label}")
    print(f"  Frames processed:    {stats['total_frames']}")
    print(f"  Total elapsed:       {stats.get('elapsed_s', 0):.1f}s")
    print(f"  Overall FPS:         {stats.get('fps', 0):.1f}")
    if "steady_state_fps" in stats:
        print(f"  Steady-state FPS:    {stats['steady_state_fps']:.1f}")
        print(f"  First frame:         {stats['first_frame_ms']:.1f}ms")
        print(f"  Steady avg/frame:    {stats['steady_avg_ms']:.1f}ms")
    print(f"  Total detections:    {det_count[0]}")
    if tracker:
        print(f"  Unique tracks:       {track_count[0]}")
    if args.output:
        print(f"  Output saved:        {args.output}")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
