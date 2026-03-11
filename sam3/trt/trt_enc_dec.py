# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""TensorRT runtime wrapper for the SAM3 encoder+decoder+scoring pipeline.

Loads a serialized TRT engine (built from the ONNX export of
``_EncDecForExport``), allocates GPU buffers, and provides a ``forward()``
method that replaces the PyTorch encoder+decoder+scoring+box forward pass.

The engine has fixed shapes at ``max_classes`` batch size.  The wrapper
handles padding/slicing for the actual number of classes at runtime.
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor

try:
    import tensorrt as trt
except ImportError:
    trt = None


class TRTEncoderDecoder:
    """Drop-in TensorRT replacement for the encoder+decoder+scoring pipeline.

    Args:
        engine_path: Path to serialized TensorRT engine file.
        max_classes: Fixed batch dimension the engine was built with.
        device: CUDA device string (default "cuda").
    """

    def __init__(
        self,
        engine_path: str,
        max_classes: int = None,  # deprecated, read from engine
        device: str = "cuda",
    ):
        if trt is None:
            raise ImportError(
                "tensorrt is required. Install with: pip install tensorrt"
            )

        self.device = torch.device(device)

        # Load engine
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        with open(engine_path, "rb") as f:
            engine_data = f.read()
        self.engine = runtime.deserialize_cuda_engine(engine_data)
        self.context = self.engine.create_execution_context()

        # Query all dimensions from the engine's img_feat binding
        img_shape = self.engine.get_tensor_shape("img_feat")
        # img_shape is (max_classes, 256, H, W)
        self.max_classes = img_shape[0]
        self.spatial_h = img_shape[2]
        self.spatial_w = img_shape[3]

        # Allocate persistent GPU buffers — inputs
        mc = self.max_classes
        self._buf_img = torch.zeros(
            mc, 256, self.spatial_h, self.spatial_w,
            dtype=torch.float32, device=self.device,
        )
        self._buf_pos = torch.zeros(
            mc, 256, self.spatial_h, self.spatial_w,
            dtype=torch.float32, device=self.device,
        )
        self._buf_text = torch.zeros(
            32, mc, 256, dtype=torch.float32, device=self.device
        )
        self._buf_mask = torch.ones(
            mc, 32, dtype=torch.float32, device=self.device
        )

        # Allocate persistent GPU buffers — outputs
        self._buf_scores = torch.zeros(
            mc, 200, 1, dtype=torch.float32, device=self.device
        )
        self._buf_boxes = torch.zeros(
            mc, 200, 4, dtype=torch.float32, device=self.device
        )

        # Check if engine has presence output
        self.has_presence = any(
            self.engine.get_tensor_name(i) == "presence"
            for i in range(self.engine.num_io_tensors)
        )
        if self.has_presence:
            self._buf_presence = torch.zeros(
                mc, 1, dtype=torch.float32, device=self.device
            )

        # Set tensor addresses in execution context
        self.context.set_tensor_address("img_feat", self._buf_img.data_ptr())
        self.context.set_tensor_address("img_pos", self._buf_pos.data_ptr())
        self.context.set_tensor_address("text_feats", self._buf_text.data_ptr())
        self.context.set_tensor_address("text_mask", self._buf_mask.data_ptr())
        self.context.set_tensor_address("scores", self._buf_scores.data_ptr())
        self.context.set_tensor_address("boxes", self._buf_boxes.data_ptr())
        if self.has_presence:
            self.context.set_tensor_address("presence", self._buf_presence.data_ptr())

        # CUDA stream for TRT execution
        self._stream = torch.cuda.Stream(device=self.device)

        print(
            f"TRT encoder-decoder loaded: {engine_path} "
            f"(max_classes={self.max_classes}, "
            f"spatial={self.spatial_h}x{self.spatial_w}, "
            f"imgsz={self.spatial_h * 14})"
        )

    def forward(
        self,
        img_feats: List[Tensor],
        img_pos_embeds: List[Tensor],
        text_feats: Tensor,
        text_mask: Tensor,
        num_classes: int,
    ) -> Tuple[Tensor, Tensor]:
        """Run TRT encoder+decoder+scoring inference.

        Args:
            img_feats: List of FPN features [(H*W, 1, d)] — only last level used.
            img_pos_embeds: List of position encodings [(H*W, 1, d)].
            text_feats: Text features (seq, N, d) — seq-first.
            text_mask: Text padding mask (N, seq) — True=padding.
            num_classes: Actual number of classes (N <= max_classes).

        Returns:
            Tuple of:
              - scores: (N, 200, 1) detection logits
              - boxes: (N, 200, 4) cxcywh coordinates (sigmoid)
        """
        N = num_classes
        assert N <= self.max_classes, (
            f"num_classes={N} exceeds max_classes={self.max_classes}"
        )

        # --- Pack inputs into fixed-size buffers ---

        # Image features: last FPN level, (H*W, 1, d) → (1, d, H, W) → expand
        img_feat = img_feats[-1]  # (H*W, 1, 256)
        hw = img_feat.shape[0]
        h = w = int(math.isqrt(hw))
        assert h * w == hw, f"Non-square spatial features: {hw} elements"
        if h != self.spatial_h or w != self.spatial_w:
            raise RuntimeError(
                f"TRT engine expects {self.spatial_h}x{self.spatial_w} spatial "
                f"features (imgsz={self.spatial_h * 14}) but got {h}x{w} "
                f"(imgsz={h * 14}). Rebuild the TRT engine at the new "
                f"resolution, or use --imgsz {self.spatial_h * 14}."
            )
        img_nchw = img_feat.squeeze(1).permute(1, 0).reshape(1, 256, h, w)

        self._buf_img[:] = img_nchw.expand(self.max_classes, -1, -1, -1)

        # Position encoding: same format
        pos_feat = img_pos_embeds[-1]  # (H*W, 1, 256)
        pos_nchw = pos_feat.squeeze(1).permute(1, 0).reshape(1, 256, h, w)
        self._buf_pos[:] = pos_nchw.expand(self.max_classes, -1, -1, -1)

        # Text features: pad from N to max_classes
        self._buf_text.zero_()
        self._buf_text[:, :N, :] = text_feats[:, :N, :].float()

        # Text mask: pad with True (padding) — store as float for TRT
        self._buf_mask.fill_(1.0)
        self._buf_mask[:N, :] = text_mask[:N, :].float()

        # --- Execute TRT ---
        # Ensure buffer copies (on default stream) finish before TRT reads them
        copy_event = torch.cuda.current_stream(self.device).record_event()
        self._stream.wait_event(copy_event)

        self.context.execute_async_v3(self._stream.cuda_stream)
        self._stream.synchronize()

        # --- Slice outputs to actual N classes ---
        scores = self._buf_scores[:N].clone()  # (N, 200, 1)
        boxes = self._buf_boxes[:N].clone()    # (N, 200, 4)

        if self.has_presence:
            presence = self._buf_presence[:N].clone()  # (N, 1)
            return scores, boxes, presence

        return scores, boxes

    def __del__(self):
        """Clean up TRT resources."""
        if hasattr(self, "context"):
            del self.context
        if hasattr(self, "engine"):
            del self.engine
