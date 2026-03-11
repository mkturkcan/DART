# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""Pipelined video processors for SAM3.

Three pipeline implementations:

1. ``PipelinedVideoProcessor`` — Uses two TRT backbone instances with separate
   CUDA streams for ping-pong double-buffering.  Requires TRT backbone engine.

2. ``CompiledVideoProcessor`` — Uses ``torch.compile`` backbone on a dedicated
   CUDA stream, overlapping with enc-dec on the default stream.  No TRT
   backbone needed; works with TRT or PyTorch enc-dec.

3. ``SplitBackboneVideoProcessor`` — Splits the ViT backbone at block 24 so
   that blocks 24-31 + FPN run alongside enc-dec on the default stream, while
   blocks 0-23 for the next frame run on a separate stream.  Rebalances the
   pipeline when backbone >> enc-dec.  Zero quality loss (all 32 blocks run).

Typical numbers on RTX 4080:

  TRT pipeline (PipelinedVideoProcessor):
    - Backbone: ~26ms (TRT FP16, broken accuracy)
    - Enc-dec:  ~20ms (TRT FP16)
    - Pipelined: ~26ms (38 FPS)

  Compiled pipeline (CompiledVideoProcessor):
    - Backbone: ~75ms (torch.compile max-autotune, correct)
    - Enc-dec:  ~20ms (TRT FP16)
    - Pipelined: ~75ms (13.3 FPS)

  Split pipeline (SplitBackboneVideoProcessor):
    With TRT enc-dec (split=20, cuda_graphs=True):
    - Part 1 (blocks 0-19): ~48ms
    - Part 2 (blocks 20-31) + FPN + TRT enc-dec: ~47ms
    - Pipelined: ~48ms (20.8 FPS)

    With compiled enc-dec (split=24):
    - Part 1 (blocks 0-23): ~58ms
    - Part 2 (blocks 24-31) + FPN + enc-dec: ~58ms
    - Pipelined: ~58ms (17.2 FPS)
"""

import time
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch import Tensor


def _preprocess_frame(
    frame_bgr: np.ndarray,
    resolution: int,
    transform,
    device,
) -> Tuple[Tensor, int, int]:
    """Convert an OpenCV BGR frame to a normalized GPU tensor.

    Skips CPU resize — the GPU transform pipeline already contains
    ``v2.Resize`` which handles resizing more efficiently on-device.
    Only BGR→RGB conversion is done on CPU (negligible cost).

    Args:
        frame_bgr: (H, W, 3) uint8 BGR array from cv2.
        resolution: Target square resolution (e.g. 1008).
        transform: Predictor's torchvision transform pipeline.
        device: Target CUDA device.

    Returns:
        (tensor, orig_h, orig_w) where tensor is (1, 3, res, res) float32.
    """
    orig_h, orig_w = frame_bgr.shape[:2]

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1)
    tensor = tensor.to(device)
    tensor = transform(tensor).unsqueeze(0)

    return tensor, orig_h, orig_w


def _clone_backbone_out(bb_out: Dict) -> Dict:
    """Deep-clone backbone output dict so buffers can be safely reused.

    Only clones tensor values; non-tensor values (like None) are passed through.
    This is needed when overlapping backbone(N+1) with enc-dec(N), since
    torch.compile with CUDA graphs may reuse static output buffers.
    """
    cloned = {}
    for k, v in bb_out.items():
        if isinstance(v, Tensor):
            cloned[k] = v.clone()
        elif isinstance(v, list):
            cloned[k] = [t.clone() if isinstance(t, Tensor) else t for t in v]
        else:
            cloned[k] = v
    return cloned


# ---------------------------------------------------------------------------
# PipelinedVideoProcessor (TRT backbone, double-buffered)
# ---------------------------------------------------------------------------

class PipelinedVideoProcessor:
    """Overlap backbone(frame N+1) with enc-dec(frame N) for near-real-time.

    Requires a TRT backbone engine.  The encoder-decoder can be either TRT
    or PyTorch (the pipeline benefits either way since backbone and enc-dec
    run on independent CUDA streams).

    Args:
        predictor: An initialized ``Sam3MultiClassPredictorFast`` with
            ``set_classes()`` already called and TRT backbone configured.
        backbone_engine_path: Path to the TRT backbone engine file.
            A second ``TRTBackbone`` instance is created from this file
            for ping-pong double-buffering.
    """

    def __init__(
        self,
        predictor,
        backbone_engine_path: str,
    ):
        from sam3.trt.trt_backbone import TRTBackbone

        self.predictor = predictor
        self.device = predictor.device
        self.resolution = predictor.resolution

        predictor._ensure_compiled()

        if predictor._trt_backbone is None:
            raise ValueError(
                "PipelinedVideoProcessor requires a TRT backbone. "
                "Pass trt_engine_path when creating the predictor."
            )

        pos_module = predictor.model.backbone.vision_backbone.position_encoding
        self._backbones = [
            predictor._trt_backbone,
            TRTBackbone(
                engine_path=backbone_engine_path,
                device=str(self.device),
                pos_encoding_module=pos_module,
            ),
        ]

    @torch.inference_mode()
    def process_video(
        self,
        video_path: str,
        confidence_threshold: float = 0.3,
        nms_threshold: float = 0.7,
        per_class_nms: bool = True,
        callback: Optional[Callable] = None,
        max_frames: int = 0,
    ) -> Dict:
        """Process a video with pipelined backbone/enc-dec execution.

        Args:
            video_path: Path to input video file.
            confidence_threshold: Minimum detection score.
            nms_threshold: NMS IoU threshold.
            per_class_nms: Per-class (True) or cross-class (False) NMS.
            callback: Optional ``fn(frame_idx, results, frame_bgr)``
                called for each processed frame.
            max_frames: Stop after this many frames (0 = process all).

        Returns:
            Stats dict with timing information.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        fps_in = cap.get(cv2.CAP_PROP_FPS)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w_in = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h_in = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if max_frames > 0:
            total = min(total, max_frames)
        print(
            f"Video: {video_path} ({w_in}x{h_in} @ {fps_in:.1f} FPS, "
            f"{total} frames)"
        )

        bb_a, bb_b = self._backbones
        frame_times = []

        ret, frame_bgr = cap.read()
        if not ret:
            cap.release()
            return {"total_frames": 0, "elapsed_s": 0, "fps": 0}

        # First frame: fully blocking
        torch.cuda.synchronize()
        t_start = time.perf_counter()
        t0 = t_start

        tensor_0, orig_h, orig_w = _preprocess_frame(
            frame_bgr, self.resolution, self.predictor.transform, self.device
        )
        bb_out_a = bb_a.forward_image(tensor_0)
        state_a = {
            "backbone_out": bb_out_a,
            "original_height": orig_h,
            "original_width": orig_w,
        }

        ret, next_frame = cap.read()
        has_next = ret
        bb_event_b = None
        bb_out_b = None
        next_orig_h, next_orig_w = 0, 0

        if has_next:
            tensor_1, next_orig_h, next_orig_w = _preprocess_frame(
                next_frame, self.resolution, self.predictor.transform, self.device
            )
            bb_out_b, bb_event_b = bb_b.forward_image_async(tensor_1)

        results = self.predictor.predict(
            state_a,
            confidence_threshold=confidence_threshold,
            nms_threshold=nms_threshold,
            per_class_nms=per_class_nms,
        )

        torch.cuda.synchronize()
        first_frame_ms = (time.perf_counter() - t0) * 1000
        frame_times.append(first_frame_ms)

        if callback is not None:
            callback(0, results, frame_bgr)

        frame_idx = 1

        # Pipeline loop
        while has_next and (max_frames <= 0 or frame_idx < max_frames):
            torch.cuda.synchronize()
            t0 = time.perf_counter()

            current_frame = next_frame
            current_bb_out = bb_out_b
            current_event = bb_event_b
            current_orig_h = next_orig_h
            current_orig_w = next_orig_w

            ret, next_frame = cap.read()
            has_next = ret

            if has_next and (max_frames <= 0 or frame_idx + 1 < max_frames):
                next_tensor, next_orig_h, next_orig_w = _preprocess_frame(
                    next_frame, self.resolution, self.predictor.transform,
                    self.device,
                )
                bb_out_next, bb_event_next = bb_a.forward_image_async(
                    next_tensor
                )
            else:
                bb_out_next = None
                bb_event_next = None
                has_next = False

            torch.cuda.current_stream(
                torch.device(self.device)
            ).wait_event(current_event)

            state = {
                "backbone_out": current_bb_out,
                "original_height": current_orig_h,
                "original_width": current_orig_w,
            }
            results = self.predictor.predict(
                state,
                confidence_threshold=confidence_threshold,
                nms_threshold=nms_threshold,
                per_class_nms=per_class_nms,
            )

            torch.cuda.synchronize()
            frame_ms = (time.perf_counter() - t0) * 1000
            frame_times.append(frame_ms)

            if callback is not None:
                callback(frame_idx, results, current_frame)

            frame_idx += 1
            bb_a, bb_b = bb_b, bb_a
            bb_out_b = bb_out_next
            bb_event_b = bb_event_next

        cap.release()
        return _compute_stats(frame_times, t_start)


# ---------------------------------------------------------------------------
# CompiledVideoProcessor (torch.compile backbone, stream-based overlap)
# ---------------------------------------------------------------------------

class CompiledVideoProcessor:
    """Overlap torch.compile backbone with enc-dec via CUDA streams.

    Runs backbone on a dedicated CUDA stream while enc-dec runs on the
    default stream (TRT enc-dec has its own internal stream too). Clones
    backbone outputs between frames to avoid buffer conflicts when
    torch.compile uses CUDA graphs with static output buffers.

    Effective per-frame time: ``max(backbone, enc_dec + postprocess)``
    instead of ``backbone + enc_dec + postprocess``.

    Args:
        predictor: An initialized ``Sam3MultiClassPredictorFast`` with
            ``set_classes()`` already called.  Must have ``compile_mode``
            set (e.g. ``"max-autotune"``).
    """

    def __init__(self, predictor):
        self.predictor = predictor
        self.device = predictor.device
        self.resolution = predictor.resolution

        # Ensure predictor is compiled (triggers torch.compile + TRT load)
        predictor._ensure_compiled()

        if predictor._backbone_fn is None:
            raise ValueError("Predictor backbone not initialized")

        # Dedicated CUDA stream for backbone execution
        self._backbone_stream = torch.cuda.Stream(device=self.device)

    @torch.inference_mode()
    def process_video(
        self,
        video_path: str,
        confidence_threshold: float = 0.3,
        nms_threshold: float = 0.7,
        per_class_nms: bool = True,
        callback: Optional[Callable] = None,
        max_frames: int = 0,
    ) -> Dict:
        """Process a video with pipelined backbone/enc-dec execution.

        Pipeline timeline (steady state):

        .. code-block:: text

            backbone_stream: [--- backbone(N+1) ~75ms ---]
            default_stream:  [-- enc-dec(N) ~20ms --][post]
                             |                            |
                             t=0                        t=75ms → output frame N

        Args:
            video_path: Path to input video file.
            confidence_threshold: Minimum detection score.
            nms_threshold: NMS IoU threshold.
            per_class_nms: Per-class (True) or cross-class (False) NMS.
            callback: Optional ``fn(frame_idx, results, frame_bgr)``
                called for each processed frame.
            max_frames: Stop after this many frames (0 = process all).

        Returns:
            Stats dict with timing information.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        fps_in = cap.get(cv2.CAP_PROP_FPS)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w_in = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h_in = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if max_frames > 0:
            total = min(total, max_frames)
        print(
            f"Video: {video_path} ({w_in}x{h_in} @ {fps_in:.1f} FPS, "
            f"{total} frames)"
        )

        backbone_fn = self.predictor._backbone_fn
        backbone_stream = self._backbone_stream
        frame_times = []

        # --- Read first frame ---
        ret, frame_bgr = cap.read()
        if not ret:
            cap.release()
            return {"total_frames": 0, "elapsed_s": 0, "fps": 0}

        # --- First frame: fully blocking (cold start) ---
        torch.cuda.synchronize()
        t_start = time.perf_counter()
        t0 = t_start

        tensor_0, orig_h, orig_w = _preprocess_frame(
            frame_bgr, self.resolution, self.predictor.transform, self.device
        )

        # Run backbone on default stream (first call, may trigger compilation)
        with torch.autocast("cuda", dtype=torch.float16):
            bb_out = backbone_fn(tensor_0)
        # Clone so we have a safe copy for enc-dec
        bb_out_safe = _clone_backbone_out(bb_out)

        state = {
            "backbone_out": bb_out_safe,
            "original_height": orig_h,
            "original_width": orig_w,
        }

        # Read next frame and preprocess (while we still need to enc-dec frame 0)
        ret, next_frame = cap.read()
        has_next = ret
        next_tensor = None
        next_orig_h, next_orig_w = 0, 0

        if has_next:
            next_tensor, next_orig_h, next_orig_w = _preprocess_frame(
                next_frame, self.resolution, self.predictor.transform,
                self.device,
            )
            # Launch backbone for frame 1 async on backbone_stream
            # Wait for default stream's clone to finish before overwriting
            backbone_stream.wait_event(
                torch.cuda.current_stream(self.device).record_event()
            )
            with torch.cuda.stream(backbone_stream):
                with torch.autocast("cuda", dtype=torch.float16):
                    bb_out_next = backbone_fn(next_tensor)
            bb_done_event = backbone_stream.record_event()
        else:
            bb_out_next = None
            bb_done_event = None

        # Run enc-dec on frame 0 (default stream)
        # While this runs, backbone(frame 1) executes on backbone_stream
        results = self.predictor.predict(
            state,
            confidence_threshold=confidence_threshold,
            nms_threshold=nms_threshold,
            per_class_nms=per_class_nms,
        )

        torch.cuda.synchronize()
        first_frame_ms = (time.perf_counter() - t0) * 1000
        frame_times.append(first_frame_ms)

        if callback is not None:
            callback(0, results, frame_bgr)

        frame_idx = 1

        # --- Pipeline loop ---
        while has_next and (max_frames <= 0 or frame_idx < max_frames):
            torch.cuda.synchronize()
            t0 = time.perf_counter()

            current_frame = next_frame

            # Wait for current frame's backbone to finish
            torch.cuda.current_stream(self.device).wait_event(bb_done_event)

            # Clone backbone output to free the buffers for next backbone call
            bb_out_safe = _clone_backbone_out(bb_out_next)

            current_state = {
                "backbone_out": bb_out_safe,
                "original_height": next_orig_h,
                "original_width": next_orig_w,
            }

            # Read and preprocess the NEXT frame
            ret, next_frame = cap.read()
            has_next = ret

            if has_next and (max_frames <= 0 or frame_idx + 1 < max_frames):
                next_tensor, next_orig_h, next_orig_w = _preprocess_frame(
                    next_frame, self.resolution, self.predictor.transform,
                    self.device,
                )
                # Launch backbone(N+2) async — wait for clone to finish first
                backbone_stream.wait_event(
                    torch.cuda.current_stream(self.device).record_event()
                )
                with torch.cuda.stream(backbone_stream):
                    with torch.autocast("cuda", dtype=torch.float16):
                        bb_out_next = backbone_fn(next_tensor)
                bb_done_event = backbone_stream.record_event()
            else:
                bb_out_next = None
                bb_done_event = None
                has_next = False

            # Run enc-dec + postprocess on current frame (default stream)
            # Overlaps with backbone(N+2) on backbone_stream
            results = self.predictor.predict(
                current_state,
                confidence_threshold=confidence_threshold,
                nms_threshold=nms_threshold,
                per_class_nms=per_class_nms,
            )

            torch.cuda.synchronize()
            frame_ms = (time.perf_counter() - t0) * 1000
            frame_times.append(frame_ms)

            if callback is not None:
                callback(frame_idx, results, current_frame)

            frame_idx += 1

        cap.release()
        return _compute_stats(frame_times, t_start)


# ---------------------------------------------------------------------------
# SplitBackboneVideoProcessor (split ViT blocks across pipeline stages)
# ---------------------------------------------------------------------------

def _clone_intermediate(intermediate: Dict) -> Dict:
    """Deep-clone ViT intermediate state dict for double-buffering.

    torch.compile with CUDA graphs may reuse static output buffers, so we
    clone the intermediate tensor to prevent the next frame's part1 from
    overwriting the current frame's data while part2 + enc-dec reads it.
    """
    return {
        "x": intermediate["x"].clone(),
        "s": intermediate["s"],
    }


class SplitBackboneVideoProcessor:
    """Split backbone across pipeline stages for higher throughput.

    Splits the ViT backbone at a configurable block index:

    - **Part 1** (blocks 0..split-1): runs on ``backbone_stream`` for the NEXT frame
    - **Part 2** (blocks split..31) + FPN + enc-dec: runs on default stream for
      the CURRENT frame

    All 32 blocks still run for every frame -- **zero quality loss**.

    Effective per-frame time: ``max(part1_time, part2_time + enc_dec_time)``
    instead of ``max(full_backbone_time, enc_dec_time)``.

    Optimal split points on RTX 4080 (torch.compile default + FP16):

    .. code-block:: text

        With TRT enc-dec (14ms):   split=20 -> 48ms/frame (20.8 FPS, +67%)
        With compiled enc-dec (36ms): split=24 -> 58ms/frame (17.2 FPS, +37%)

    When ``cuda_graphs=True``, manual CUDA graph capture wraps the compiled
    functions.  This gives ``torch.compile(mode="default")`` kernel fusion
    **plus** CUDA-graph-level launch overhead elimination -- equivalent to
    ``max-autotune`` performance without the cross-function CUDA graph tree
    conflicts.

    Args:
        predictor: An initialized ``Sam3MultiClassPredictorFast`` with
            ``set_classes()`` already called.
        split_block: Block index at which to split the ViT trunk. Blocks
            ``[0, split_block)`` run as part 1, ``[split_block, end)`` as
            part 2.  Default 20 (optimal for TRT enc-dec at ~14ms).
        compile_mode: torch.compile mode for part1 and part2. Defaults to
            ``"default"``.  Use ``"default"`` for kernel fusion without
            automatic CUDA graphs.  Pair with ``cuda_graphs=True`` for
            manual graph capture that avoids cross-function conflicts.
        cuda_graphs: If True, capture manual CUDA graphs around the compiled
            part1 and part2 functions after warmup.  Eliminates CPU launch
            overhead (~2-5ms savings).  Requires static input shapes.
    """

    def __init__(
        self,
        predictor,
        split_block: int = 20,
        compile_mode: Optional[str] = "default",
        cuda_graphs: bool = False,
    ):
        self.predictor = predictor
        self.device = predictor.device
        self.resolution = predictor.resolution
        self._split_block = split_block
        self._cuda_graphs = cuda_graphs

        predictor._ensure_compiled()

        backbone = predictor.model.backbone

        mode = compile_mode

        if mode is not None:
            self._part1_fn = torch.compile(
                backbone.forward_image_part1,
                mode=mode,
                dynamic=False,
            )
            self._part2_fn = torch.compile(
                backbone.forward_image_part2,
                mode=mode,
                dynamic=False,
            )
        else:
            self._part1_fn = backbone.forward_image_part1
            self._part2_fn = backbone.forward_image_part2

        self._backbone_stream = torch.cuda.Stream(device=self.device)

        # CUDA graph state (initialized lazily during warmup)
        self._graphs_captured = False
        self._g1 = None  # CUDA graph for part1
        self._g2 = None  # CUDA graph for part2
        self._static_input = None  # static input buffer for part1
        self._static_inter_in = None  # static intermediate input for part2
        self._static_inter_out = None  # static intermediate output from part1
        self._static_bb_out = None  # static backbone_out from part2

    def _capture_cuda_graphs(self, sample_input: Tensor):
        """Capture CUDA graphs for part1 and part2 after warmup.

        Must be called with a sample input tensor of the correct shape.
        Uses torch.amp.autocast with cache_enabled=False to prevent stale
        weight cast cache entries in the captured graphs.
        """
        split = self._split_block
        use_fp16 = self.predictor.use_fp16

        # --- Static input buffer for part1 ---
        self._static_input = sample_input.clone()

        # --- Warmup part1 on a side stream ---
        s1 = torch.cuda.Stream(device=self.device)
        s1.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(s1):
            with torch.amp.autocast("cuda", dtype=torch.float16,
                                    enabled=use_fp16, cache_enabled=False):
                for _ in range(3):
                    inter = self._part1_fn(self._static_input, split)
        torch.cuda.current_stream(self.device).wait_stream(s1)

        # --- Capture part1 graph ---
        self._g1 = torch.cuda.CUDAGraph()
        with torch.amp.autocast("cuda", dtype=torch.float16,
                                enabled=use_fp16, cache_enabled=False):
            with torch.cuda.graph(self._g1):
                static_inter = self._part1_fn(self._static_input, split)
        # static_inter["x"] is at a fixed address, updated on every g1.replay()
        self._static_inter_out = static_inter

        # --- Static intermediate buffer for part2 (separate from part1's output) ---
        self._static_inter_in = {
            "x": static_inter["x"].clone(),
            "s": static_inter["s"],
        }

        # --- Warmup part2 ---
        s2 = torch.cuda.Stream(device=self.device)
        s2.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(s2):
            with torch.amp.autocast("cuda", dtype=torch.float16,
                                    enabled=use_fp16, cache_enabled=False):
                for _ in range(3):
                    bb_out = self._part2_fn(self._static_inter_in, split)
        torch.cuda.current_stream(self.device).wait_stream(s2)

        # --- Capture part2 graph ---
        self._g2 = torch.cuda.CUDAGraph()
        with torch.amp.autocast("cuda", dtype=torch.float16,
                                enabled=use_fp16, cache_enabled=False):
            with torch.cuda.graph(self._g2):
                static_bb = self._part2_fn(self._static_inter_in, split)
        # static_bb dict tensors at fixed addresses, updated on every g2.replay()
        self._static_bb_out = static_bb
        self._graphs_captured = True

        print(
            f"  CUDA graphs captured for split backbone "
            f"(split={split}, input={list(sample_input.shape)})"
        )

    @torch.inference_mode()
    def process_video(
        self,
        video_path: str,
        confidence_threshold: float = 0.3,
        nms_threshold: float = 0.7,
        per_class_nms: bool = True,
        callback: Optional[Callable] = None,
        max_frames: int = 0,
    ) -> Dict:
        """Process a video with split-backbone pipelining.

        Pipeline timeline (steady state):

        .. code-block:: text

            backbone_stream: [-- part1(N+1) ~58ms --]
            default_stream:  [-- part2(N) + enc-dec(N) ~40ms --]
                             ^                                  ^
                             t=0                              t=58ms → output frame N

        Args:
            video_path: Path to input video file.
            confidence_threshold: Minimum detection score.
            nms_threshold: NMS IoU threshold.
            per_class_nms: Per-class (True) or cross-class (False) NMS.
            callback: Optional ``fn(frame_idx, results, frame_bgr)``
                called for each processed frame.
            max_frames: Stop after this many frames (0 = process all).

        Returns:
            Stats dict with timing information.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        fps_in = cap.get(cv2.CAP_PROP_FPS)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w_in = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h_in = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if max_frames > 0:
            total = min(total, max_frames)
        print(
            f"Video: {video_path} ({w_in}x{h_in} @ {fps_in:.1f} FPS, "
            f"{total} frames)"
        )

        split_block = self._split_block
        backbone_stream = self._backbone_stream
        use_fp16 = self.predictor.use_fp16
        use_graphs = self._cuda_graphs
        frame_times = []

        # --- Read first frame ---
        ret, frame_bgr = cap.read()
        if not ret:
            cap.release()
            return {"total_frames": 0, "elapsed_s": 0, "fps": 0}

        # --- Cold start: frame 0 (fully blocking) ---
        torch.cuda.synchronize()
        t_start = time.perf_counter()
        t0 = t_start

        tensor_0, orig_h, orig_w = _preprocess_frame(
            frame_bgr, self.resolution, self.predictor.transform, self.device
        )

        # Run full split pipeline blocking for frame 0
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_fp16):
            torch.compiler.cudagraph_mark_step_begin()
            intermediate_0 = self._part1_fn(tensor_0, split_block)
            intermediate_0 = _clone_intermediate(intermediate_0)
            torch.compiler.cudagraph_mark_step_begin()
            bb_out_0 = self._part2_fn(intermediate_0, split_block)

        # Capture CUDA graphs after first frame warmup
        if use_graphs and not self._graphs_captured:
            self._capture_cuda_graphs(tensor_0)

        state_0 = {
            "backbone_out": bb_out_0,
            "original_height": orig_h,
            "original_width": orig_w,
        }

        # Read next frame and start part1 async
        ret, next_frame = cap.read()
        has_next = ret
        next_intermediate = None
        bb_done_event = None
        next_orig_h, next_orig_w = 0, 0

        if has_next:
            next_tensor, next_orig_h, next_orig_w = _preprocess_frame(
                next_frame, self.resolution, self.predictor.transform,
                self.device,
            )
            # Launch part1 for frame 1 on backbone_stream
            backbone_stream.wait_event(
                torch.cuda.current_stream(self.device).record_event()
            )
            if use_graphs and self._graphs_captured:
                with torch.cuda.stream(backbone_stream):
                    self._static_input.copy_(next_tensor)
                    self._g1.replay()
                next_intermediate = self._static_inter_out
            else:
                with torch.cuda.stream(backbone_stream):
                    with torch.autocast("cuda", dtype=torch.float16, enabled=use_fp16):
                        torch.compiler.cudagraph_mark_step_begin()
                        next_intermediate = self._part1_fn(next_tensor, split_block)
            bb_done_event = backbone_stream.record_event()

        # Run enc-dec for frame 0 (default stream)
        results = self.predictor.predict(
            state_0,
            confidence_threshold=confidence_threshold,
            nms_threshold=nms_threshold,
            per_class_nms=per_class_nms,
        )

        torch.cuda.synchronize()
        first_frame_ms = (time.perf_counter() - t0) * 1000
        frame_times.append(first_frame_ms)

        if callback is not None:
            callback(0, results, frame_bgr)

        frame_idx = 1

        # --- Pipeline loop ---
        while has_next and (max_frames <= 0 or frame_idx < max_frames):
            torch.cuda.synchronize()
            t0 = time.perf_counter()

            current_frame = next_frame

            # Wait for current frame's part1 to finish
            torch.cuda.current_stream(self.device).wait_event(bb_done_event)

            # Read and preprocess NEXT frame (while GPU processes current)
            ret, next_frame = cap.read()
            has_next = ret
            current_orig_h = next_orig_h
            current_orig_w = next_orig_w

            if use_graphs and self._graphs_captured:
                # CUDA graph path: copy intermediate to part2's static buffer,
                # then launch part1(next) and part2(current) concurrently
                self._static_inter_in["x"].copy_(
                    next_intermediate["x"]
                )

                if has_next and (max_frames <= 0 or frame_idx + 1 < max_frames):
                    next_tensor, next_orig_h, next_orig_w = _preprocess_frame(
                        next_frame, self.resolution, self.predictor.transform,
                        self.device,
                    )
                    # Launch part1(N+2) on backbone_stream
                    backbone_stream.wait_event(
                        torch.cuda.current_stream(self.device).record_event()
                    )
                    with torch.cuda.stream(backbone_stream):
                        self._static_input.copy_(next_tensor)
                        self._g1.replay()
                    next_intermediate = self._static_inter_out
                    bb_done_event = backbone_stream.record_event()
                else:
                    has_next = False

                # Run part2(current) via CUDA graph on default stream
                self._g2.replay()
                bb_out = self._static_bb_out
            else:
                # Non-graph path: clone intermediate, run part1/part2 normally
                current_intermediate = _clone_intermediate(next_intermediate)

                if has_next and (max_frames <= 0 or frame_idx + 1 < max_frames):
                    next_tensor, next_orig_h, next_orig_w = _preprocess_frame(
                        next_frame, self.resolution, self.predictor.transform,
                        self.device,
                    )
                    backbone_stream.wait_event(
                        torch.cuda.current_stream(self.device).record_event()
                    )
                    with torch.cuda.stream(backbone_stream):
                        with torch.autocast("cuda", dtype=torch.float16, enabled=use_fp16):
                            torch.compiler.cudagraph_mark_step_begin()
                            next_intermediate = self._part1_fn(
                                next_tensor, split_block
                            )
                    bb_done_event = backbone_stream.record_event()
                else:
                    has_next = False

                # Run part2 for current frame (default stream)
                with torch.autocast("cuda", dtype=torch.float16, enabled=use_fp16):
                    torch.compiler.cudagraph_mark_step_begin()
                    bb_out = self._part2_fn(current_intermediate, split_block)

            state = {
                "backbone_out": bb_out,
                "original_height": current_orig_h,
                "original_width": current_orig_w,
            }
            results = self.predictor.predict(
                state,
                confidence_threshold=confidence_threshold,
                nms_threshold=nms_threshold,
                per_class_nms=per_class_nms,
            )

            torch.cuda.synchronize()
            frame_ms = (time.perf_counter() - t0) * 1000
            frame_times.append(frame_ms)

            if callback is not None:
                callback(frame_idx, results, current_frame)

            frame_idx += 1

        cap.release()
        return _compute_stats(frame_times, t_start)


# ---------------------------------------------------------------------------
# TRTSplitPipelinedVideoProcessor (split TRT backbone, balanced pipeline)
# ---------------------------------------------------------------------------

class TRTSplitPipelinedVideoProcessor:
    """Pipeline with split TRT backbone for balanced overlap.

    Uses two TRT engines (Part1 and Part2) to split the ViT backbone at a
    configurable block index.  This allows balancing the pipeline stages:

    - **Stream 1**: Part1(N+1) — embeddings + blocks[0..K-1]
    - **Default stream**: Part2(N) + enc-dec(N) — blocks[K..31] + FPN + enc-dec

    Effective per-frame time: ``max(part1_ms, part2_ms + enc_dec_ms)`` which
    can be significantly better than ``max(backbone_ms, enc_dec_ms)`` when the
    full backbone is much slower than enc-dec.

    Example (1008 resolution, 8 classes, split_block=27):

    .. code-block:: text

        Full TRT backbone:   57ms backbone, 39ms enc-dec → pipelined 82ms (SM contention)
        Split TRT backbone:  ~50ms Part1, ~7ms Part2 + 39ms enc-dec = ~50ms → 20 FPS

    Args:
        predictor: Initialized ``Sam3MultiClassPredictorFast``.
        split_backbone: ``TRTSplitBackbone`` instance with Part1 + Part2 engines.
    """

    def __init__(self, predictor, split_backbone):
        self.predictor = predictor
        self.device = predictor.device
        self.resolution = predictor.resolution
        self._split_bb = split_backbone

        predictor._ensure_compiled()

    @torch.inference_mode()
    def process_video(
        self,
        video_path: str,
        confidence_threshold: float = 0.3,
        nms_threshold: float = 0.7,
        per_class_nms: bool = True,
        callback: Optional[Callable] = None,
        max_frames: int = 0,
    ) -> Dict:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        fps_in = cap.get(cv2.CAP_PROP_FPS)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w_in = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h_in = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if max_frames > 0:
            total = min(total, max_frames)
        print(
            f"Video: {video_path} ({w_in}x{h_in} @ {fps_in:.1f} FPS, "
            f"{total} frames)"
        )

        split_bb = self._split_bb
        frame_times = []

        # --- Read first frame ---
        ret, frame_bgr = cap.read()
        if not ret:
            cap.release()
            return {"total_frames": 0, "elapsed_s": 0, "fps": 0}

        # --- Frame 0: fully blocking ---
        torch.cuda.synchronize()
        t_start = time.perf_counter()
        t0 = t_start

        tensor_0, orig_h, orig_w = _preprocess_frame(
            frame_bgr, self.resolution, self.predictor.transform, self.device
        )

        # Run full backbone sequentially for frame 0
        bb_out_0 = split_bb.forward_image(tensor_0)

        state_0 = {
            "backbone_out": bb_out_0,
            "original_height": orig_h,
            "original_width": orig_w,
        }

        # Read next frame
        ret, next_frame = cap.read()
        has_next = ret
        p1_buf = None
        p1_event = None
        next_orig_h, next_orig_w = 0, 0

        if has_next:
            next_tensor, next_orig_h, next_orig_w = _preprocess_frame(
                next_frame, self.resolution, self.predictor.transform,
                self.device,
            )
            # Launch Part1 for frame 1 async
            p1_buf, p1_event = split_bb.forward_part1_async(next_tensor)

        # Run enc-dec on frame 0 (default stream)
        results = self.predictor.predict(
            state_0,
            confidence_threshold=confidence_threshold,
            nms_threshold=nms_threshold,
            per_class_nms=per_class_nms,
        )

        torch.cuda.synchronize()
        first_frame_ms = (time.perf_counter() - t0) * 1000
        frame_times.append(first_frame_ms)

        if callback is not None:
            callback(0, results, frame_bgr)

        frame_idx = 1

        # --- Pipeline loop ---
        while has_next and (max_frames <= 0 or frame_idx < max_frames):
            torch.cuda.synchronize()
            t0 = time.perf_counter()

            current_frame = next_frame
            current_orig_h = next_orig_h
            current_orig_w = next_orig_w

            # Wait for Part1 of current frame to finish
            torch.cuda.current_stream(self.device).wait_event(p1_event)

            # Clone the intermediate so Part1 buffer is free for next frame
            current_intermediate = p1_buf.clone()

            # Read and preprocess NEXT frame
            ret, next_frame = cap.read()
            has_next = ret

            if has_next and (max_frames <= 0 or frame_idx + 1 < max_frames):
                next_tensor, next_orig_h, next_orig_w = _preprocess_frame(
                    next_frame, self.resolution, self.predictor.transform,
                    self.device,
                )
                # Launch Part1(N+2) async on stream1
                p1_buf, p1_event = split_bb.forward_part1_async(next_tensor)
            else:
                has_next = False

            # Run Part2(current) + enc-dec(current) on default stream
            # This overlaps with Part1(N+2) on stream1
            bb_out = split_bb.forward_part2(current_intermediate)

            state = {
                "backbone_out": bb_out,
                "original_height": current_orig_h,
                "original_width": current_orig_w,
            }
            results = self.predictor.predict(
                state,
                confidence_threshold=confidence_threshold,
                nms_threshold=nms_threshold,
                per_class_nms=per_class_nms,
            )

            torch.cuda.synchronize()
            frame_ms = (time.perf_counter() - t0) * 1000
            frame_times.append(frame_ms)

            if callback is not None:
                callback(frame_idx, results, current_frame)

            frame_idx += 1

        cap.release()
        return _compute_stats(frame_times, t_start)


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def _compute_stats(frame_times: List[float], t_start: float) -> Dict:
    """Compute timing stats from per-frame times."""
    elapsed = time.perf_counter() - t_start
    n = len(frame_times)
    if n == 0:
        return {"total_frames": 0, "elapsed_s": 0, "fps": 0}

    avg_ms = sum(frame_times) / n
    if n > 1:
        steady_times = frame_times[1:]
        steady_avg = sum(steady_times) / len(steady_times)
        steady_fps = 1000.0 / steady_avg if steady_avg > 0 else 0
    else:
        steady_avg = avg_ms
        steady_fps = 1000.0 / avg_ms if avg_ms > 0 else 0

    return {
        "total_frames": n,
        "elapsed_s": elapsed,
        "fps": n / elapsed if elapsed > 0 else 0,
        "steady_state_fps": steady_fps,
        "avg_frame_ms": avg_ms,
        "first_frame_ms": frame_times[0],
        "steady_avg_ms": steady_avg,
    }
