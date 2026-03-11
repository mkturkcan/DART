# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""TensorRT runtime wrapper for the SAM3 vision backbone.

Loads a serialized TRT engine, allocates GPU buffers, and provides a
``forward_image()`` method that is a drop-in replacement for
``SAM3VLBackbone.forward_image()``.

Works with engines from both the Meta backbone export (``export_backbone.py``)
and HuggingFace backbone export (``export_hf_backbone.py``).  Tensor names
and shapes are auto-detected from the engine — no hardcoded assumptions.

Position encodings are deterministic from spatial size (PositionEmbeddingSine)
and are pre-computed once at init — they are NOT part of the TRT engine.
"""

from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor

try:
    import tensorrt as trt
except ImportError:
    trt = None


class TRTBackbone:
    """Drop-in TensorRT replacement for SAM3VLBackbone.forward_image().

    I/O tensor names and shapes are auto-detected from the engine, so this
    works with any backbone engine that has 1 image input and 3 FPN outputs.

    Args:
        engine_path: Path to serialized TensorRT engine file.
        device: CUDA device string (default "cuda").
        pos_encoding_module: Optional PositionEmbeddingSine instance for
            computing vision_pos_enc.  If None, position encodings are
            created from scratch.
    """

    def __init__(
        self,
        engine_path: str,
        device: str = "cuda",
        pos_encoding_module=None,
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

        # Map TRT dtypes → torch dtypes
        _trt_to_torch = {
            trt.float32: torch.float32,
            trt.float16: torch.float16,
            trt.int32: torch.int32,
        }
        if hasattr(trt, "bfloat16"):
            _trt_to_torch[trt.bfloat16] = torch.bfloat16

        # Auto-detect I/O tensor names, shapes, and dtypes from the engine.
        # This makes TRTBackbone work with any backbone engine regardless of
        # tensor naming convention (Meta: images/fpn_0/1/2, HF: pixel_values/conv2d_*).
        self._input_name: str = ""
        self._output_names: List[str] = []
        output_shapes: List[Tuple[int, ...]] = []

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = tuple(self.engine.get_tensor_shape(name))
            dtype = _trt_to_torch.get(
                self.engine.get_tensor_dtype(name), torch.float32
            )
            mode = self.engine.get_tensor_mode(name)

            if mode == trt.TensorIOMode.INPUT:
                self._input_name = name
                self._input_buf = torch.empty(
                    shape, dtype=dtype, device=self.device
                )
            else:
                self._output_names.append(name)
                output_shapes.append(shape)
                # Buffer allocated below after sorting

        # Sort outputs by spatial size descending (largest FPN level first)
        # so fpn[0] is always the highest-resolution level regardless of
        # engine naming. E.g. [288x288, 144x144, 72x72].
        sorted_indices = sorted(
            range(len(output_shapes)),
            key=lambda i: output_shapes[i][-1],  # sort by last dim (width)
            reverse=True,
        )
        self._output_names = [self._output_names[i] for i in sorted_indices]
        output_shapes = [output_shapes[i] for i in sorted_indices]

        # Allocate output buffers in sorted order
        self._output_bufs: List[Tensor] = []
        for name, shape in zip(self._output_names, output_shapes):
            out_dtype = _trt_to_torch.get(
                self.engine.get_tensor_dtype(name), torch.float32
            )
            self._output_bufs.append(
                torch.empty(shape, dtype=out_dtype, device=self.device)
            )

        # Log I/O for debugging
        print(f"  Engine input: {self._input_name} "
              f"{list(self._input_buf.shape)} {self._input_buf.dtype}")
        for name, buf in zip(self._output_names, self._output_bufs):
            print(f"  Engine output: {name} {list(buf.shape)} {buf.dtype}")

        # Set tensor addresses in the execution context
        self.context.set_tensor_address(
            self._input_name, self._input_buf.data_ptr()
        )
        for name, buf in zip(self._output_names, self._output_bufs):
            self.context.set_tensor_address(name, buf.data_ptr())

        # Pre-compute position encodings (deterministic from spatial size)
        self._vision_pos_enc = self._precompute_pos_enc(
            pos_encoding_module, output_shapes
        )

        # CUDA stream for TRT execution
        self._stream = torch.cuda.Stream(device=self.device)

        print(
            f"TRT backbone loaded: {engine_path} "
            f"({sum(b.nelement() * b.element_size() for b in self._output_bufs) / 1e6:.1f}MB output buffers)"
        )

    def _precompute_pos_enc(
        self, pos_module=None, output_shapes=None
    ) -> List[Tensor]:
        """Compute PositionEmbeddingSine for each FPN spatial size."""
        if pos_module is not None:
            pos_enc = []
            for shape in output_shapes:
                dummy = torch.zeros(shape, device=self.device)
                pe = pos_module(dummy).detach()
                pos_enc.append(pe)
            return pos_enc

        # Fallback: create a fresh PositionEmbeddingSine (d_model=256)
        from sam3.model.position_encoding import PositionEmbeddingSine

        pe_module = PositionEmbeddingSine(num_pos_feats=256, normalize=True)
        pos_enc = []
        for shape in output_shapes:
            dummy = torch.zeros(shape, device=self.device)
            pe = pe_module(dummy).detach().to(self.device)
            pos_enc.append(pe)
        return pos_enc

    def forward_image(self, samples: Tensor) -> Dict:
        """Run TRT backbone inference.

        Args:
            samples: Input image tensor (1, 3, H, W) on CUDA.

        Returns:
            Dict matching SAM3VLBackbone.forward_image() format:
              - backbone_fpn: List[Tensor] — 3 FPN levels (high-res first)
              - vision_pos_enc: List[Tensor] — 3 position encodings
              - vision_features: Tensor — last (lowest-res) FPN level
              - sam2_backbone_out: None
        """
        # Copy input to persistent buffer, converting dtype if needed
        self._input_buf.copy_(samples)

        # Record event on default stream so TRT stream waits for copy_
        event = torch.cuda.current_stream(self.device).record_event()
        self._stream.wait_event(event)

        # Execute TRT
        self.context.execute_async_v3(self._stream.cuda_stream)

        # Make default stream wait for TRT to finish
        event = self._stream.record_event()
        torch.cuda.current_stream(self.device).wait_event(event)

        # Convert outputs to FP32 if the engine produced FP16
        fpn = [buf.float() for buf in self._output_bufs]

        return {
            "backbone_fpn": fpn,
            "vision_pos_enc": self._vision_pos_enc,
            "vision_features": fpn[-1],
            "sam2_backbone_out": None,
        }

    def forward_image_async(
        self, samples: Tensor
    ) -> Tuple[Dict, torch.cuda.Event]:
        """Start TRT backbone inference without blocking the default stream.

        Same as ``forward_image()`` but returns immediately after launching
        the TRT kernel.  The caller must wait for the returned event before
        reading from the output tensors.

        Args:
            samples: Input image tensor (1, 3, H, W) on CUDA.

        Returns:
            Tuple of (output_dict, done_event).  ``done_event`` fires when
            the TRT execution completes and output buffers are safe to read.
        """
        self._input_buf.copy_(samples)

        copy_event = torch.cuda.current_stream(self.device).record_event()
        self._stream.wait_event(copy_event)

        self.context.execute_async_v3(self._stream.cuda_stream)

        done_event = self._stream.record_event()

        fpn = [buf.float() for buf in self._output_bufs]
        return {
            "backbone_fpn": fpn,
            "vision_pos_enc": self._vision_pos_enc,
            "vision_features": fpn[-1],
            "sam2_backbone_out": None,
        }, done_event

    def __del__(self):
        """Clean up TRT resources."""
        if hasattr(self, "context"):
            del self.context
        if hasattr(self, "engine"):
            del self.engine


class TRTSplitBackbone:
    """Split backbone with two TRT engines for pipelined video processing.

    Part 1 runs blocks 0..K-1 (embeddings + layer_norm + early blocks),
    outputting an intermediate hidden-state tensor [B, H, W, D] (BHWD).
    Part 2 runs blocks K..31 plus the FPN neck, producing the standard
    3-level FPN output.

    Pipeline usage:
      - Stream 1: part1(frame N+1) — produces intermediate
      - Stream 0: part2(frame N)   — produces FPN, then enc-dec runs

    Effective per-frame time: max(part1_ms, part2_ms + enc_dec_ms) instead of
    backbone_ms + enc_dec_ms.

    Args:
        part1_engine_path: TRT engine for Part 1 (pixel_values → hidden_states).
        part2_engine_path: TRT engine for Part 2 (hidden_states → 3 FPN levels).
        device: CUDA device string.
        pos_encoding_module: PositionEmbeddingSine for vision_pos_enc.
    """

    def __init__(
        self,
        part1_engine_path: str,
        part2_engine_path: str,
        device: str = "cuda",
        pos_encoding_module=None,
    ):
        if trt is None:
            raise ImportError("tensorrt is required")

        self.device = torch.device(device)

        _trt_to_torch = {
            trt.float32: torch.float32,
            trt.float16: torch.float16,
            trt.int32: torch.int32,
        }
        if hasattr(trt, "bfloat16"):
            _trt_to_torch[trt.bfloat16] = torch.bfloat16

        # --- Load Part 1 engine ---
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        with open(part1_engine_path, "rb") as f:
            self._engine1 = runtime.deserialize_cuda_engine(f.read())
        self._ctx1 = self._engine1.create_execution_context()

        self._p1_input_name = ""
        self._p1_output_name = ""
        for i in range(self._engine1.num_io_tensors):
            name = self._engine1.get_tensor_name(i)
            shape = tuple(self._engine1.get_tensor_shape(name))
            dtype = _trt_to_torch.get(
                self._engine1.get_tensor_dtype(name), torch.float32
            )
            mode = self._engine1.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                self._p1_input_name = name
                self._p1_input_buf = torch.empty(
                    shape, dtype=dtype, device=self.device
                )
                print(f"  Part1 input:  {name} {list(shape)} {dtype}")
            else:
                self._p1_output_name = name
                self._p1_output_buf = torch.empty(
                    shape, dtype=dtype, device=self.device
                )
                print(f"  Part1 output: {name} {list(shape)} {dtype}")

        self._ctx1.set_tensor_address(
            self._p1_input_name, self._p1_input_buf.data_ptr()
        )
        self._ctx1.set_tensor_address(
            self._p1_output_name, self._p1_output_buf.data_ptr()
        )

        # --- Load Part 2 engine ---
        with open(part2_engine_path, "rb") as f:
            self._engine2 = runtime.deserialize_cuda_engine(f.read())
        self._ctx2 = self._engine2.create_execution_context()

        self._p2_input_name = ""
        self._p2_output_names: List[str] = []
        p2_output_shapes: List[Tuple[int, ...]] = []

        for i in range(self._engine2.num_io_tensors):
            name = self._engine2.get_tensor_name(i)
            shape = tuple(self._engine2.get_tensor_shape(name))
            dtype = _trt_to_torch.get(
                self._engine2.get_tensor_dtype(name), torch.float32
            )
            mode = self._engine2.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                self._p2_input_name = name
                self._p2_input_buf = torch.empty(
                    shape, dtype=dtype, device=self.device
                )
                print(f"  Part2 input:  {name} {list(shape)} {dtype}")
            else:
                self._p2_output_names.append(name)
                p2_output_shapes.append(shape)

        # Sort Part2 outputs by spatial size descending (largest FPN first)
        sorted_idx = sorted(
            range(len(p2_output_shapes)),
            key=lambda i: p2_output_shapes[i][-1],
            reverse=True,
        )
        self._p2_output_names = [self._p2_output_names[i] for i in sorted_idx]
        p2_output_shapes = [p2_output_shapes[i] for i in sorted_idx]

        self._p2_output_bufs: List[Tensor] = []
        for name, shape in zip(self._p2_output_names, p2_output_shapes):
            out_dtype = _trt_to_torch.get(
                self._engine2.get_tensor_dtype(name), torch.float32
            )
            buf = torch.empty(shape, dtype=out_dtype, device=self.device)
            self._p2_output_bufs.append(buf)
            print(f"  Part2 output: {name} {list(shape)} {out_dtype}")

        self._ctx2.set_tensor_address(
            self._p2_input_name, self._p2_input_buf.data_ptr()
        )
        for name, buf in zip(self._p2_output_names, self._p2_output_bufs):
            self._ctx2.set_tensor_address(name, buf.data_ptr())

        # Position encodings (from Part2 FPN output shapes)
        self._vision_pos_enc = self._precompute_pos_enc(
            pos_encoding_module, p2_output_shapes
        )

        # Separate streams for Part1 and Part2
        self._stream1 = torch.cuda.Stream(device=self.device)
        self._stream2 = torch.cuda.Stream(device=self.device)

        print(
            f"TRT split backbone loaded: "
            f"Part1={part1_engine_path}, Part2={part2_engine_path}"
        )

    def _precompute_pos_enc(self, pos_module, output_shapes):
        if pos_module is not None:
            return [
                pos_module(torch.zeros(s, device=self.device)).detach()
                for s in output_shapes
            ]
        from sam3.model.position_encoding import PositionEmbeddingSine
        pe_module = PositionEmbeddingSine(num_pos_feats=256, normalize=True)
        return [
            pe_module(torch.zeros(s, device=self.device)).detach().to(self.device)
            for s in output_shapes
        ]

    def forward_part1_async(self, samples: Tensor) -> Tuple[Tensor, torch.cuda.Event]:
        """Run Part1 on stream1 (non-blocking). Returns (output_buf, done_event)."""
        self._p1_input_buf.copy_(samples)
        copy_event = torch.cuda.current_stream(self.device).record_event()
        self._stream1.wait_event(copy_event)
        self._ctx1.execute_async_v3(self._stream1.cuda_stream)
        done = self._stream1.record_event()
        return self._p1_output_buf, done

    def forward_part2(self, intermediate: Tensor) -> Dict:
        """Run Part2 on stream2 (blocking). Returns backbone_out dict.

        Uses a non-default stream to avoid TRT's extra cudaStreamSynchronize()
        overhead on the default stream.
        """
        self._p2_input_buf.copy_(intermediate)
        copy_event = torch.cuda.current_stream(self.device).record_event()
        self._stream2.wait_event(copy_event)
        self._ctx2.execute_async_v3(self._stream2.cuda_stream)
        done = self._stream2.record_event()
        torch.cuda.current_stream(self.device).wait_event(done)
        fpn = [buf.float() for buf in self._p2_output_bufs]
        return {
            "backbone_fpn": fpn,
            "vision_pos_enc": self._vision_pos_enc,
            "vision_features": fpn[-1],
            "sam2_backbone_out": None,
        }

    def forward_image(self, samples: Tensor) -> Dict:
        """Run full backbone sequentially (Part1 → Part2). For non-pipelined use.

        Both parts use non-default streams to avoid TRT default-stream overhead.
        """
        self._p1_input_buf.copy_(samples)

        # Part1 on stream1
        copy_event = torch.cuda.current_stream(self.device).record_event()
        self._stream1.wait_event(copy_event)
        self._ctx1.execute_async_v3(self._stream1.cuda_stream)
        p1_done = self._stream1.record_event()

        # Wait for Part1, copy intermediate to Part2 input
        torch.cuda.current_stream(self.device).wait_event(p1_done)
        self._p2_input_buf.copy_(self._p1_output_buf)

        # Part2 on stream2
        copy_event2 = torch.cuda.current_stream(self.device).record_event()
        self._stream2.wait_event(copy_event2)
        self._ctx2.execute_async_v3(self._stream2.cuda_stream)
        p2_done = self._stream2.record_event()

        # Wait for Part2 to finish
        torch.cuda.current_stream(self.device).wait_event(p2_done)
        fpn = [buf.float() for buf in self._p2_output_bufs]
        return {
            "backbone_fpn": fpn,
            "vision_pos_enc": self._vision_pos_enc,
            "vision_features": fpn[-1],
            "sam2_backbone_out": None,
        }

    def __del__(self):
        for attr in ("_ctx1", "_ctx2", "_engine1", "_engine2"):
            if hasattr(self, attr):
                delattr(self, attr)
