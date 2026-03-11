# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""
Optimized multi-class inference for SAM3.

Builds on Sam3MultiClassPredictor with five key optimizations:

  1. **Batched forward**: All N classes run through encoder+decoder+seg_head
     in a single call with bs=N, replacing the sequential per-class loop.
     GPU parallelism replaces kernel launch overhead.

  2. **torch.compile**: Optionally JIT-compiles the encoder and decoder
     for fused kernels and reduced Python overhead.

  3. **FP16 inference**: Runs encoder+decoder+seg_head under torch.autocast
     for halved memory bandwidth and doubled tensor-core throughput.

  4. **Presence early-exit**: After the decoder, checks per-class presence
     logits and only runs mask generation for classes that are actually
     present in the image.

  5. **Shared encoder**: Runs the encoder once with a generic scene-level
     prompt (e.g. "urban"), then fans out to N class-specific decoder
     passes.  Eliminates the N× encoder cost entirely.

  6. **Single-pass**: True single-pass inference — concatenates all class
     text tokens, runs encoder+decoder+masks once at bs=1.  Class
     assignment uses cosine similarity between query features and per-class
     embeddings (L2-normalized to remove magnitude bias).  Detection
     confidence uses standard DotProductScoring.  Fastest possible mode.

Compute comparison (N classes, K present):
  Per-prompt SAM3:     N × (backbone + encoder + decoder + masks)
  Sequential:          1 × backbone + N × (encoder + decoder + masks)
  Batched:             1 × backbone + 1 × encoder(bs=N) + 1 × decoder(bs=N)
                       + 1 × masks(bs=K)
  Shared-enc batched:  1 × backbone + 1 × encoder(bs=1) + 1 × decoder(bs=N)
                       + 1 × masks(bs=K)
  Single-pass:         1 × backbone + 1 × encoder(bs=1) + 1 × decoder(bs=1)
                       + 1 × masks(bs=1)

Usage:
    from sam3.model.sam3_multiclass_fast import Sam3MultiClassPredictorFast

    predictor = Sam3MultiClassPredictorFast(
        model, device="cuda",
        compile_mode="reduce-overhead",  # or None to disable
        use_fp16=True,
        presence_threshold=0.05,
        shared_encoder=True,             # enable shared-encoder mode
        generic_prompt="urban",          # scene-level encoder prompt
    )
    predictor.set_classes(["car", "pedestrian", "bicycle", ...])

    state = predictor.set_image(image)
    results = predictor.predict(state, confidence_threshold=0.3)
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import PIL
import torch
import torch.nn.functional as F
from torchvision.transforms import v2

from torchvision.ops import batched_nms as _batched_nms, nms as _nms

from sam3.model.box_ops import box_cxcywh_to_xyxy
from sam3.model.data_misc import interpolate
from sam3.model.model_misc import inverse_sigmoid


class _TRTModelStub:
    """Minimal model stub for full-TRT mode (no checkpoint needed).

    Provides just enough interface for ``Sam3MultiClassPredictorFast`` to
    work with TRT backbone + TRT enc-dec in detection-only mode.  The
    text encoder is bypassed via cached text embeddings.
    """

    def __init__(self, device: str = "cuda", num_feature_levels: int = 1):
        self.num_feature_levels = num_feature_levels

        class _DummyForward:
            """Provides a ``.forward`` attribute that raises on call."""
            def forward(self, *args, **kwargs):
                raise RuntimeError(
                    "Not available in TRT-only mode. "
                    "Use --text-cache to bypass the text encoder."
                )

        class _Decoder(_DummyForward):
            presence_token = None

        class _Transformer:
            encoder = _DummyForward()
            decoder = _Decoder()

        class _VisionBackbone:
            position_encoding = None

        class _Backbone:
            vision_backbone = _VisionBackbone()

            def forward_text(self, *args, **kwargs):
                raise RuntimeError(
                    "Text encoder not available in TRT-only mode. "
                    "Use --text-cache with cached embeddings."
                )

            def forward_image(self, *args, **kwargs):
                raise RuntimeError("Backbone not available in TRT-only mode.")

        self.backbone = _Backbone()
        self.transformer = _Transformer()

    def _get_img_feats(self, backbone_out, img_ids):
        """Extract and reshape image features from backbone output."""
        vis_feats = backbone_out["backbone_fpn"][-self.num_feature_levels:]
        vis_pos_enc = backbone_out["vision_pos_enc"][-self.num_feature_levels:]
        vis_feat_sizes = [x.shape[-2:] for x in vis_pos_enc]
        img_feats = [x[img_ids].flatten(2).permute(2, 0, 1) for x in vis_feats]
        img_pos_embeds = [
            x[img_ids].flatten(2).permute(2, 0, 1) for x in vis_pos_enc
        ]
        return backbone_out, img_feats, img_pos_embeds, vis_feat_sizes


class Sam3MultiClassPredictorFast:
    """Optimized multi-class inference for SAM3.

    Batches all N classes through encoder+decoder in a single GPU call,
    optionally compiles the model, runs in FP16, and skips mask generation
    for absent classes.
    """

    # Valid class_method options for single-pass mode
    CLASS_METHODS = ("cosine", "attention", "prototype")

    def __init__(
        self,
        model,
        resolution: int = 1008,
        device: str = "cuda",
        compile_mode: Optional[str] = None,
        use_fp16: bool = True,
        presence_threshold: float = 0.05,
        shared_encoder: bool = False,
        generic_prompt: str = "object",
        single_pass: bool = False,
        class_method: str = "cosine",
        prototype_path: Optional[str] = None,
        detection_only: bool = False,
        trt_engine_path: Optional[str] = None,
        trt_enc_dec_engine_path: Optional[str] = None,
        trt_max_classes: int = 4,
    ):
        """
        Args:
            model: A Sam3Image model instance (already loaded with weights).
            resolution: Input image resolution (default 1008 to match SAM3).
            device: Torch device for inference.
            compile_mode: torch.compile mode for encoder/decoder.
                Options: None (disabled), "default", "reduce-overhead",
                "max-autotune". None by default.
            use_fp16: If True, run encoder/decoder/seg_head under autocast.
            presence_threshold: Minimum presence probability to keep a class.
                Set to 0.0 to disable early-exit (process all classes).
            shared_encoder: If True, run the encoder once with generic_prompt
                and fan out to N class-specific decoder passes.  Trades a
                small quality delta for eliminating the N× encoder cost.
            generic_prompt: Text prompt for the shared encoder pass.  Should
                describe the general scene context (e.g. "urban", "indoor",
                "aerial").  Only used when shared_encoder=True.
            single_pass: If True, run a true single-pass: concatenate all
                class text tokens, run encoder+decoder+masks at bs=1, then
                assign classes via class_method.  Fastest mode.
                Mutually exclusive with shared_encoder.
            class_method: Class assignment strategy for single-pass mode.
                "cosine"    — cosine similarity with text-derived embeddings
                             (zero training, baseline).
                "attention" — extract decoder ca_text attention weights and
                             sum per class token group (zero training).
                "prototype" — cosine similarity with calibrated class
                             prototypes (requires prototype_path).
            prototype_path: Path to a .pt file with calibrated class
                prototypes (from scripts/calibrate_single_pass.py).
                Required when class_method="prototype".
            detection_only: If True, skip mask generation entirely and
                return only boxes + scores.  Uses box-based NMS instead
                of mask-based NMS.  ~13% faster per class.
            trt_engine_path: Path to a serialized TensorRT engine for
                the backbone.  When set, replaces the PyTorch backbone
                with a TRT backend (~2-4x faster).  Build one with
                ``python -m sam3.trt.build_engine``.
            trt_enc_dec_engine_path: Path to a serialized TensorRT engine
                for the encoder+decoder+scoring pipeline.  When set,
                replaces the PyTorch encoder+decoder with a TRT backend.
                Build with ``python -m sam3.trt.build_engine --type enc-dec``.
            trt_max_classes: Maximum number of classes the enc-dec TRT
                engine was built with.  Must match the ``--max-classes``
                used during export.
        """
        if single_pass and shared_encoder:
            raise ValueError("single_pass and shared_encoder are mutually exclusive")
        if class_method not in self.CLASS_METHODS:
            raise ValueError(f"class_method must be one of {self.CLASS_METHODS}")
        if class_method == "prototype" and prototype_path is None:
            raise ValueError("prototype_path is required when class_method='prototype'")

        self.model = model
        self.resolution = resolution
        self.device = device
        self.use_fp16 = use_fp16
        self.presence_threshold = presence_threshold
        self.shared_encoder = shared_encoder
        self.generic_prompt = generic_prompt
        self.single_pass = single_pass
        self._class_method = class_method
        self._prototype_path = prototype_path
        self.detection_only = detection_only
        self._trt_engine_path = trt_engine_path
        self._trt_backbone = None  # lazy-loaded TRTBackbone
        self._trt_enc_dec_engine_path = trt_enc_dec_engine_path
        self._trt_enc_dec = None  # lazy-loaded TRTEncoderDecoder
        self._trt_max_classes = trt_max_classes

        if trt_enc_dec_engine_path is not None and not detection_only:
            raise ValueError(
                "trt_enc_dec_engine_path requires detection_only=True "
                "(TRT enc-dec does not produce hidden states for mask generation)"
            )

        self.transform = v2.Compose(
            [
                v2.ToDtype(torch.uint8, scale=True),
                v2.Resize(size=(resolution, resolution)),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

        # Class embedding cache
        self._class_names: Optional[List[str]] = None
        self._num_classes: int = 0

        # Batched prompts: (seq, N, d) and (N, seq)
        self._batched_text: Optional[torch.Tensor] = None
        self._batched_mask: Optional[torch.Tensor] = None

        # Generic prompt embeddings for shared-encoder mode: (seq, 1, d), (1, seq)
        self._generic_text: Optional[torch.Tensor] = None
        self._generic_mask: Optional[torch.Tensor] = None

        # Single-pass mode: concatenated text and per-class cosine embeddings
        self._concat_text: Optional[torch.Tensor] = None   # (total_seq, 1, d)
        self._concat_mask: Optional[torch.Tensor] = None   # (1, total_seq)
        self._class_proj_norm: Optional[torch.Tensor] = None  # (N, d_proj) L2-normalized

        # Attention-based class assignment: token index ranges per class
        self._class_token_ranges: Optional[List[Tuple[int, int]]] = None

        # Prototype-based class assignment: calibrated per-class centroids
        self._calibrated_prototypes: Optional[torch.Tensor] = None  # (N, d_proj) L2-norm

        # torch.compile wrappers (lazy — compiled on first use)
        self._compile_mode = compile_mode
        self._encoder_fn = None
        self._decoder_fn = None
        self._backbone_fn = None

        # NOTE: Previously nulled presence_token here for detection_only mode,
        # but presence scores are valuable for detection quality (gating low-quality preds).
        # The _postprocess_detection path already handles presence_probs when available.

        # Cached zero tensors (avoid repeated allocation)
        self._prompt_pos_cache: Dict[Tuple[int, ...], torch.Tensor] = {}

    def _zeros_like_cached(self, tensor: torch.Tensor) -> torch.Tensor:
        """Return a cached zeros tensor matching the given shape/dtype/device."""
        key = (tensor.shape, tensor.dtype, tensor.device)
        if key not in self._prompt_pos_cache:
            self._prompt_pos_cache[key] = torch.zeros_like(tensor)
        return self._prompt_pos_cache[key]

    def _ensure_compiled(self) -> None:
        """Lazily compile encoder, decoder, and backbone on first use."""
        if self._encoder_fn is not None:
            return

        encoder = self.model.transformer.encoder
        decoder = self.model.transformer.decoder
        backbone = self.model.backbone

        if self._compile_mode is not None:
            self._encoder_fn = torch.compile(
                encoder.forward,
                mode=self._compile_mode,
                dynamic=True,
            )
            self._decoder_fn = torch.compile(
                decoder.forward,
                mode=self._compile_mode,
                dynamic=True,
            )
        else:
            self._encoder_fn = encoder.forward
            self._decoder_fn = decoder.forward

        # Encoder-decoder: TRT engine (lazy load)
        if self._trt_enc_dec_engine_path is not None and self._trt_enc_dec is None:
            from sam3.trt.trt_enc_dec import TRTEncoderDecoder

            self._trt_enc_dec = TRTEncoderDecoder(
                engine_path=self._trt_enc_dec_engine_path,
                max_classes=self._trt_max_classes,
                device=self.device,
            )

        # Backbone: TRT engine > torch.compile > raw forward
        if self._trt_engine_path is not None:
            from sam3.trt.trt_backbone import TRTBackbone

            # SAM3VLBackbone uses .vision_backbone, StudentVLBackbone uses .student_backbone
            if hasattr(backbone, 'vision_backbone'):
                pos_module = backbone.vision_backbone.position_encoding
            elif hasattr(backbone, 'student_backbone'):
                pos_module = backbone.student_backbone.position_encoding
            else:
                pos_module = None
            self._trt_backbone = TRTBackbone(
                engine_path=self._trt_engine_path,
                device=self.device,
                pos_encoding_module=pos_module,
            )
            self._backbone_fn = self._trt_backbone.forward_image
        elif self._compile_mode is not None:
            self._backbone_fn = torch.compile(
                backbone.forward_image,
                mode=self._compile_mode,
                dynamic=False,  # fixed input size (1, 3, 1008, 1008)
            )
        else:
            self._backbone_fn = backbone.forward_image

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def set_classes(
        self,
        class_names: List[str],
        text_cache: Optional[str] = None,
    ) -> None:
        """Pre-compute and cache text embeddings for all target classes.

        Stores prompts in batched form: (seq, N, d_model) ready for the
        decoder.  When shared_encoder=True, also encodes the generic prompt.

        Args:
            class_names: List of class names, e.g. ["car", "pedestrian"].
            text_cache: Optional path to a ``.pt`` file for caching text
                embeddings.  If the file exists and contains matching class
                names, embeddings are loaded from disk (no text encoder
                needed).  Otherwise, embeddings are computed and saved to
                this path for future reuse.
        """
        if not class_names:
            raise ValueError("class_names must be a non-empty list of strings")

        self._class_names = list(class_names)
        self._num_classes = len(class_names)

        # Try loading from cache
        if text_cache is not None:
            import os
            if os.path.exists(text_cache):
                data = torch.load(text_cache, map_location=self.device, weights_only=True)
                if data.get("class_names") == list(class_names):
                    self._batched_text = data["text"]
                    self._batched_mask = data["mask"]
                    print(f"  Loaded text embeddings from cache: {text_cache}")
                    return
                print(f"  Cache class mismatch, recomputing text embeddings")

        text_outputs = self.model.backbone.forward_text(
            class_names, device=self.device
        )
        # language_features: (seq, N, d) — already batched
        # language_mask:     (N, seq) — already batched
        self._batched_text = text_outputs["language_features"]
        self._batched_mask = text_outputs["language_mask"]

        # Save to cache
        if text_cache is not None:
            torch.save({
                "class_names": list(class_names),
                "text": self._batched_text,
                "mask": self._batched_mask,
            }, text_cache)
            print(f"  Saved text embeddings to cache: {text_cache}")

        # Encode generic prompt for shared-encoder mode
        if self.shared_encoder:
            generic_out = self.model.backbone.forward_text(
                [self.generic_prompt], device=self.device
            )
            self._generic_text = generic_out["language_features"]  # (seq, 1, d)
            self._generic_mask = generic_out["language_mask"]      # (1, seq)

        # Single-pass mode: concatenate valid tokens and pre-compute
        # per-class projected embeddings for cosine scoring
        if self.single_pass:
            N = self._num_classes
            # Concatenate valid (non-padding) tokens from all classes,
            # tracking per-class token boundaries for attention method
            all_tokens = []
            token_ranges = []
            offset = 0
            for i in range(N):
                valid = ~self._batched_mask[i]  # (seq,) True=valid
                tokens_i = self._batched_text[valid, i, :]  # (valid_i, d)
                n_valid = tokens_i.shape[0]
                all_tokens.append(tokens_i)
                token_ranges.append((offset, offset + n_valid))
                offset += n_valid
            self._class_token_ranges = token_ranges

            concat_tokens = torch.cat(all_tokens, dim=0)  # (total_seq, d)
            self._concat_text = concat_tokens.unsqueeze(1)  # (total_seq, 1, d)
            self._concat_mask = torch.zeros(
                1, concat_tokens.shape[0],
                dtype=torch.bool, device=self.device,
            )  # all valid (no padding)

            # Per-class projected embeddings for cosine/prototype class assignment
            scoring = self.model.dot_prod_scoring
            per_class_proj = []
            for i in range(N):
                class_text = self._batched_text[:, i:i+1, :]  # (seq, 1, d)
                class_mask = self._batched_mask[i:i+1, :]     # (1, seq)
                text_in = class_text
                if scoring.prompt_mlp is not None:
                    text_in = scoring.prompt_mlp(text_in)
                pooled = scoring.mean_pool_text(text_in, class_mask)  # (1, d_model)
                proj = scoring.prompt_proj(pooled)  # (1, d_proj)
                per_class_proj.append(proj)
            per_class_proj = torch.cat(per_class_proj, dim=0)  # (N, d_proj)
            self._class_proj_norm = F.normalize(per_class_proj, dim=-1)

            # Load calibrated prototypes if using prototype method
            if self._class_method == "prototype" and self._prototype_path is not None:
                proto_data = torch.load(self._prototype_path, map_location=self.device)
                # Expected format: {"class_names": [...], "prototypes": (N, d_proj)}
                proto_names = proto_data["class_names"]
                proto_tensor = proto_data["prototypes"]  # (N_proto, d_proj)
                # Reorder prototypes to match current class order
                reordered = []
                for name in class_names:
                    if name not in proto_names:
                        raise ValueError(
                            f"Class '{name}' not found in prototype file. "
                            f"Available: {proto_names}"
                        )
                    idx = proto_names.index(name)
                    reordered.append(proto_tensor[idx])
                self._calibrated_prototypes = F.normalize(
                    torch.stack(reordered).to(self.device), dim=-1
                )  # (N, d_proj)

    @torch.inference_mode()
    def set_image(
        self,
        image: Union[PIL.Image.Image, torch.Tensor, np.ndarray],
        state: Optional[Dict] = None,
    ) -> Dict:
        """Encode an image through the vision backbone.

        Runs the ViT-H backbone under FP16 autocast for ~1.5-2× speedup
        (halved memory bandwidth on tensor cores).  Uses compiled backbone
        if torch.compile was requested.

        Args:
            image: Input image (PIL, tensor, or numpy array).
            state: Optional state dict to update (creates new if None).

        Returns:
            State dict with encoded image features.
        """
        if state is None:
            state = {}

        if isinstance(image, PIL.Image.Image):
            width, height = image.size
        elif isinstance(image, (torch.Tensor, np.ndarray)):
            height, width = image.shape[-2:]
        else:
            raise ValueError("Image must be a PIL image, tensor, or ndarray")

        self._ensure_compiled()

        # Fast path: resize on CPU first so we transfer ~3MB (1008²×3)
        # instead of the full original image (e.g. 6000×4000 = 72MB).
        if isinstance(image, PIL.Image.Image):
            image = image.resize(
                (self.resolution, self.resolution), PIL.Image.BILINEAR
            )
        image_tensor = v2.functional.to_image(image).to(self.device)
        image_tensor = self.transform(image_tensor).unsqueeze(0)

        state["original_height"] = height
        state["original_width"] = width

        # TRT backbone handles its own precision — skip autocast overhead.
        # For PyTorch backbone, FP16 autocast gives ~1.5-2× speedup.
        if self._trt_engine_path is not None:
            state["backbone_out"] = self._backbone_fn(image_tensor)
        else:
            with torch.autocast("cuda", dtype=torch.float16, enabled=self.use_fp16):
                state["backbone_out"] = self._backbone_fn(image_tensor)

        return state

    @torch.inference_mode()
    def _profile_set_image(
        self,
        image: Union[PIL.Image.Image, torch.Tensor, np.ndarray],
    ) -> List[Tuple[str, float]]:
        """Profile set_image breakdown. Returns list of (label, ms) tuples."""
        import time

        results = []
        self._ensure_compiled()

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        if isinstance(image, PIL.Image.Image):
            image = image.resize(
                (self.resolution, self.resolution), PIL.Image.BILINEAR
            )
        results.append(("PIL resize (CPU)", (time.perf_counter() - t0) * 1000))

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        image_tensor = v2.functional.to_image(image)
        results.append(("PIL → tensor (CPU)", (time.perf_counter() - t0) * 1000))

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        image_tensor = image_tensor.to(self.device)
        torch.cuda.synchronize()
        results.append(("CPU → GPU transfer", (time.perf_counter() - t0) * 1000))

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        image_tensor = self.transform(image_tensor).unsqueeze(0)
        torch.cuda.synchronize()
        results.append(("Normalize (GPU)", (time.perf_counter() - t0) * 1000))

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        if self._trt_engine_path is not None:
            backbone_out = self._backbone_fn(image_tensor)
        else:
            with torch.autocast("cuda", dtype=torch.float16, enabled=self.use_fp16):
                backbone_out = self._backbone_fn(image_tensor)
        torch.cuda.synchronize()
        results.append(("Backbone forward", (time.perf_counter() - t0) * 1000))

        return results

    @torch.inference_mode()
    def predict(
        self,
        state: Dict,
        confidence_threshold: float = 0.3,
        nms_threshold: float = 0.7,
        per_class_nms: bool = True,
    ) -> Dict:
        """Run optimized multi-class detection + segmentation.

        All N classes run through encoder+decoder in one batched call.
        Mask generation runs only for classes passing the presence check.

        Args:
            state: State dict from set_image() with backbone features.
            confidence_threshold: Minimum score to keep a detection.
            nms_threshold: IoU threshold for mask-based NMS.
            per_class_nms: Per-class (True) or cross-class (False) NMS.

        Returns:
            Dict with "boxes", "masks", "masks_logits", "scores",
            "class_ids", "class_names".
        """
        if self._class_names is None:
            raise RuntimeError("Call set_classes() before predict()")
        if "backbone_out" not in state:
            raise RuntimeError("Call set_image() before predict()")

        self._ensure_compiled()

        backbone_out = state["backbone_out"]
        orig_h = state["original_height"]
        orig_w = state["original_width"]

        # Extract image features once (bs=1)
        img_ids = torch.tensor([0], device=self.device, dtype=torch.long)
        backbone_out, img_feats, img_pos_embeds, vis_feat_sizes = (
            self.model._get_img_feats(backbone_out, img_ids)
        )

        # Single-pass mode: one encoder+decoder+masks pass with cosine scoring
        if self.single_pass:
            with torch.autocast("cuda", dtype=torch.float16, enabled=self.use_fp16):
                return self._predict_single_pass(
                    backbone_out, img_feats, img_pos_embeds,
                    vis_feat_sizes, img_ids,
                    confidence_threshold=confidence_threshold,
                    nms_threshold=nms_threshold,
                    per_class_nms=per_class_nms,
                    orig_h=orig_h,
                    orig_w=orig_w,
                )

        # Forward: encoder + decoder + scoring + boxes for all N classes
        if self._trt_enc_dec is not None:
            forward_fn = self._forward_batched_trt
        elif self.shared_encoder:
            forward_fn = self._forward_shared_encoder
        else:
            forward_fn = self._forward_batched
        with torch.autocast("cuda", dtype=torch.float16, enabled=self.use_fp16):
            batched = forward_fn(
                backbone_out, img_feats, img_pos_embeds,
                vis_feat_sizes, img_ids,
            )

        if batched is None:
            return self._empty_result(orig_h, orig_w)

        return self._postprocess(
            batched=batched,
            backbone_out=backbone_out,
            img_ids=img_ids,
            orig_h=orig_h,
            orig_w=orig_w,
            confidence_threshold=confidence_threshold,
            nms_threshold=nms_threshold,
            per_class_nms=per_class_nms,
        )

    @torch.inference_mode()
    def predict_image(
        self,
        image: Union[PIL.Image.Image, torch.Tensor, np.ndarray],
        confidence_threshold: float = 0.3,
        nms_threshold: float = 0.7,
        per_class_nms: bool = True,
    ) -> Dict:
        """Convenience: set_image + predict in one call."""
        state = self.set_image(image)
        return self.predict(
            state,
            confidence_threshold=confidence_threshold,
            nms_threshold=nms_threshold,
            per_class_nms=per_class_nms,
        )

    # ------------------------------------------------------------------
    # Internal: batched forward
    # ------------------------------------------------------------------

    def _forward_batched(
        self,
        backbone_out: Dict,
        img_feats: list,
        img_pos_embeds: list,
        vis_feat_sizes: list,
        img_ids: torch.Tensor,
    ) -> Optional[Dict]:
        """Run encoder+decoder for all N classes in one batched call.

        Image features (bs=1) are repeated N times along the batch dim.
        Text prompts are already in (seq, N, d) form from set_classes().

        Returns dict with batched outputs, or None if no classes pass
        the presence check.
        """
        model = self.model
        N = self._num_classes

        # --- Expand image features: (H*W, 1, d) → (H*W, N, d) ---
        # No .clone() needed: encoder creates new tensors internally via
        # flatten/cat.  The expanded views (stride 0 on batch dim) are
        # read-only and handled correctly by downstream operations.
        batched_img_feats = [f.expand(-1, N, -1) for f in img_feats]
        batched_img_pos = [p.expand(-1, N, -1) for p in img_pos_embeds]

        prompt = self._batched_text        # (seq, N, d)
        prompt_mask = self._batched_mask   # (N, seq)

        # --- Encoder (bs=N) ---
        prompt_pos_embed = self._zeros_like_cached(prompt)
        memory = self._encoder_fn(
            src=batched_img_feats,
            src_key_padding_mask=None,
            src_pos=batched_img_pos,
            prompt=prompt,
            prompt_pos=prompt_pos_embed,
            prompt_key_padding_mask=prompt_mask,
            feat_sizes=vis_feat_sizes,
        )

        encoder_hidden_states = memory["memory"]  # (total_tokens, N, d)

        # --- Decoder (bs=N) ---
        # No .clone() needed: decoder creates new tensors via linear
        # projections and residual additions (never modifies tgt in-place).
        query_embed = model.transformer.decoder.query_embed.weight
        tgt = query_embed.unsqueeze(1).expand(-1, N, -1)

        hs, reference_boxes, dec_presence_out, _ = self._decoder_fn(
            tgt=tgt,
            memory=encoder_hidden_states,
            memory_key_padding_mask=memory["padding_mask"],
            pos=memory["pos_embed"],
            reference_boxes=None,
            level_start_index=memory["level_start_index"],
            spatial_shapes=memory["spatial_shapes"],
            valid_ratios=memory["valid_ratios"],
            tgt_mask=None,
            memory_text=prompt,
            text_attention_mask=prompt_mask,
            apply_dac=False,
        )
        # hs: (num_layers, num_queries, N, d) → (num_layers, N, num_queries, d)
        hs = hs.transpose(1, 2)
        reference_boxes = reference_boxes.transpose(1, 2)

        # --- Scoring (batched — DotProductScoring handles bs=N) ---
        scores = model.dot_prod_scoring(hs, prompt, prompt_mask)
        # scores: (num_layers, N, num_queries, 1)

        # --- Box prediction ---
        box_offsets = model.transformer.decoder.bbox_embed(hs)
        ref_inv = inverse_sigmoid(reference_boxes)
        outputs_coord = (ref_inv + box_offsets).sigmoid()

        # --- Presence check (early-exit: skip masks for absent classes) ---
        presence_probs = None
        if dec_presence_out is not None:
            # dec_presence_out: (num_layers, 1, N) → last layer → (1, N)
            presence_logits = dec_presence_out[-1]  # (1, N)
            presence_probs = presence_logits.sigmoid().squeeze(0)  # (N,)

        if presence_probs is not None and self.presence_threshold > 0.0:
            present_mask = presence_probs > self.presence_threshold
        else:
            present_mask = torch.ones(N, dtype=torch.bool, device=self.device)

        present_indices = present_mask.nonzero(as_tuple=True)[0]
        if len(present_indices) == 0:
            return None

        return {
            "scores_all": scores[-1],            # (N, Q, 1)
            "boxes_all": outputs_coord[-1],      # (N, Q, 4) cxcywh
            "hs_all": hs,                        # (layers, N, Q, d)
            "encoder_hidden_states": encoder_hidden_states,  # (tokens, N, d)
            "prompt": prompt,                    # (seq, N, d)
            "prompt_mask": prompt_mask,          # (N, seq)
            "presence_probs": presence_probs,    # (N,) or None
            "present_indices": present_indices,  # (K,) indices
        }

    def _forward_shared_encoder(
        self,
        backbone_out: Dict,
        img_feats: list,
        img_pos_embeds: list,
        vis_feat_sizes: list,
        img_ids: torch.Tensor,
    ) -> Optional[Dict]:
        """Shared-encoder mode: encoder(bs=1) + decoder(bs=N).

        Runs the encoder once with a generic scene-level prompt, then
        expands the encoded image features to N and runs the decoder
        with class-specific text prompts.

        Saves: N-1 encoder passes (encoder is ~6 transformer layers).
        Tradeoff: encoder image features are not class-conditioned.
        The decoder's text cross-attention compensates.
        """
        model = self.model
        N = self._num_classes

        # --- Encoder: bs=1 with generic prompt ---
        generic_text = self._generic_text   # (seq, 1, d)
        generic_mask = self._generic_mask   # (1, seq)

        generic_pos = self._zeros_like_cached(generic_text)
        memory = self._encoder_fn(
            src=img_feats,                         # (H*W, 1, d) — read-only
            src_key_padding_mask=None,
            src_pos=img_pos_embeds,                # read-only
            prompt=generic_text,
            prompt_pos=generic_pos,
            prompt_key_padding_mask=generic_mask,
            feat_sizes=vis_feat_sizes,
        )

        # --- Expand encoder output: bs=1 → bs=N ---
        # The decoder uses deformable attention with custom CUDA kernels that
        # may require contiguous memory, so we use .contiguous() here (which
        # creates a copy for expanded tensors).  The encoder inputs above
        # are at bs=1 and don't need expansion.
        enc_hs = memory["memory"]           # (total_tokens, 1, d)
        enc_pos = memory["pos_embed"]       # (total_tokens, 1, d)
        enc_pad = memory["padding_mask"]    # (1, total_tokens) or None
        enc_vr = memory["valid_ratios"]     # (1, num_levels, 2)

        enc_hs_n = enc_hs.expand(-1, N, -1).contiguous()
        enc_pos_n = enc_pos.expand(-1, N, -1).contiguous()
        enc_pad_n = enc_pad.expand(N, -1).contiguous() if enc_pad is not None else None
        enc_vr_n = enc_vr.expand(N, -1, -1).contiguous()

        # Class-specific text prompts for decoder
        prompt = self._batched_text        # (seq, N, d)
        prompt_mask = self._batched_mask   # (N, seq)

        # --- Decoder: bs=N with class-specific text ---
        query_embed = model.transformer.decoder.query_embed.weight
        tgt = query_embed.unsqueeze(1).expand(-1, N, -1)

        hs, reference_boxes, dec_presence_out, _ = self._decoder_fn(
            tgt=tgt,
            memory=enc_hs_n,
            memory_key_padding_mask=enc_pad_n,
            pos=enc_pos_n,
            reference_boxes=None,
            level_start_index=memory["level_start_index"],
            spatial_shapes=memory["spatial_shapes"],
            valid_ratios=enc_vr_n,
            tgt_mask=None,
            memory_text=prompt,          # class-specific
            text_attention_mask=prompt_mask,  # class-specific
            apply_dac=False,
        )
        # hs: (num_layers, Q, N, d) → (num_layers, N, Q, d)
        hs = hs.transpose(1, 2)
        reference_boxes = reference_boxes.transpose(1, 2)

        # --- Scoring (class-specific prompts) ---
        scores = model.dot_prod_scoring(hs, prompt, prompt_mask)

        # --- Box prediction ---
        box_offsets = model.transformer.decoder.bbox_embed(hs)
        ref_inv = inverse_sigmoid(reference_boxes)
        outputs_coord = (ref_inv + box_offsets).sigmoid()

        # --- Presence check ---
        presence_probs = None
        if dec_presence_out is not None:
            presence_logits = dec_presence_out[-1]  # (1, N)
            presence_probs = presence_logits.sigmoid().squeeze(0)  # (N,)

        if presence_probs is not None and self.presence_threshold > 0.0:
            present_mask = presence_probs > self.presence_threshold
        else:
            present_mask = torch.ones(N, dtype=torch.bool, device=self.device)

        present_indices = present_mask.nonzero(as_tuple=True)[0]
        if len(present_indices) == 0:
            return None

        # For the seg head, encoder_hidden_states need to be per-class.
        # In shared-encoder mode they are identical across classes but
        # the seg head's cross_attend_prompt (if present) will still
        # cross-attend to class-specific text, providing differentiation.
        return {
            "scores_all": scores[-1],
            "boxes_all": outputs_coord[-1],
            "hs_all": hs,
            "encoder_hidden_states": enc_hs_n,
            "prompt": prompt,
            "prompt_mask": prompt_mask,
            "presence_probs": presence_probs,
            "present_indices": present_indices,
        }

    def _forward_batched_trt(
        self,
        backbone_out: Dict,
        img_feats: list,
        img_pos_embeds: list,
        vis_feat_sizes: list,
        img_ids: torch.Tensor,
    ) -> Optional[Dict]:
        """Run encoder+decoder+scoring via TRT engine.

        Replaces _forward_batched when a TRT enc-dec engine is available.
        Pads to max_classes, runs TRT, slices back to actual N.
        If N > max_classes, processes classes in chunks automatically.
        No presence check (presence_token disabled in TRT mode).
        """
        N = self._num_classes
        max_c = self._trt_enc_dec.max_classes
        has_presence = self._trt_enc_dec.has_presence

        if N <= max_c:
            # Single pass — all classes fit in one TRT call
            result = self._trt_enc_dec.forward(
                img_feats=img_feats,
                img_pos_embeds=img_pos_embeds,
                text_feats=self._batched_text,
                text_mask=self._batched_mask,
                num_classes=N,
            )
            if has_presence:
                scores, boxes, presence_logits = result
            else:
                scores, boxes = result
                presence_logits = None
        else:
            # Chunked — process classes in groups of max_c
            all_scores, all_boxes, all_presence = [], [], []
            for start in range(0, N, max_c):
                end = min(start + max_c, N)
                result = self._trt_enc_dec.forward(
                    img_feats=img_feats,
                    img_pos_embeds=img_pos_embeds,
                    text_feats=self._batched_text[:, start:end, :],
                    text_mask=self._batched_mask[start:end, :],
                    num_classes=end - start,
                )
                if has_presence:
                    s, b, p = result
                    all_presence.append(p)
                else:
                    s, b = result
                all_scores.append(s)
                all_boxes.append(b)
            scores = torch.cat(all_scores, dim=0)  # (N, Q, 1)
            boxes = torch.cat(all_boxes, dim=0)    # (N, Q, 4)
            presence_logits = torch.cat(all_presence, dim=0) if has_presence else None

        # Presence probabilities from TRT engine (if available)
        presence_probs = None
        if presence_logits is not None:
            presence_probs = presence_logits.squeeze(-1).sigmoid()  # (N,)

        present_indices = torch.arange(N, device=self.device)

        return {
            "scores_all": scores,              # (N, Q, 1)
            "boxes_all": boxes,                # (N, Q, 4) cxcywh
            "hs_all": None,                    # not available from TRT
            "encoder_hidden_states": None,     # not available from TRT
            "prompt": self._batched_text,      # (seq, N, d)
            "prompt_mask": self._batched_mask, # (N, seq)
            "presence_probs": presence_probs,  # (N,) or None
            "present_indices": present_indices, # all classes
        }

    # ------------------------------------------------------------------
    # Internal: single-pass forward + post-processing
    # ------------------------------------------------------------------

    def _predict_single_pass(
        self,
        backbone_out: Dict,
        img_feats: list,
        img_pos_embeds: list,
        vis_feat_sizes: list,
        img_ids: torch.Tensor,
        confidence_threshold: float,
        nms_threshold: float,
        per_class_nms: bool,
        orig_h: int,
        orig_w: int,
    ) -> Dict:
        """Single-pass inference: one encoder+decoder+masks pass for all classes.

        Concatenates all class text tokens into (total_seq, 1, d), runs
        encoder+decoder+seg_head at bs=1.  Class assignment uses one of
        three methods (cosine, attention, or prototype).

        Total cost: 1 × (backbone + encoder + decoder + masks)
        """
        model = self.model

        prompt = self._concat_text       # (total_seq, 1, d)
        prompt_mask = self._concat_mask  # (1, total_seq)

        # --- Encoder at bs=1 ---
        prompt_pos = self._zeros_like_cached(prompt)
        memory = self._encoder_fn(
            src=img_feats,              # read-only, no clone needed
            src_key_padding_mask=None,
            src_pos=img_pos_embeds,     # read-only, no clone needed
            prompt=prompt,
            prompt_pos=prompt_pos,
            prompt_key_padding_mask=prompt_mask,
            feat_sizes=vis_feat_sizes,
        )

        enc_hs = memory["memory"]  # (total_tokens, 1, d)

        # --- Install attention hooks if using attention-based class assignment ---
        attn_container = {}
        hooks = []
        if self._class_method == "attention":
            last_ca_text = model.transformer.decoder.layers[-1].ca_text

            def pre_hook(module, args, kwargs):
                kwargs["need_weights"] = True
                kwargs["average_attn_weights"] = True
                return args, kwargs

            def post_hook(module, inp, output):
                # output is (attn_output, attn_weights)
                # attn_weights: (bs, Q, total_seq) with average_attn_weights=True
                attn_container["weights"] = output[1].detach()
                return output

            hooks.append(last_ca_text.register_forward_pre_hook(
                pre_hook, with_kwargs=True
            ))
            hooks.append(last_ca_text.register_forward_hook(post_hook))

        # --- Decoder at bs=1 ---
        query_embed = model.transformer.decoder.query_embed.weight
        tgt = query_embed.unsqueeze(1)  # (Q, 1, d)

        hs, reference_boxes, dec_presence_out, _ = self._decoder_fn(
            tgt=tgt,
            memory=enc_hs,
            memory_key_padding_mask=memory["padding_mask"],
            pos=memory["pos_embed"],
            reference_boxes=None,
            level_start_index=memory["level_start_index"],
            spatial_shapes=memory["spatial_shapes"],
            valid_ratios=memory["valid_ratios"],
            tgt_mask=None,
            memory_text=prompt,
            text_attention_mask=prompt_mask,
            apply_dac=False,
        )

        # Remove hooks immediately
        for h in hooks:
            h.remove()

        # hs: (num_layers, Q, 1, d) → (num_layers, 1, Q, d)
        hs = hs.transpose(1, 2)
        reference_boxes = reference_boxes.transpose(1, 2)

        # --- Box prediction ---
        box_offsets = model.transformer.decoder.bbox_embed(hs)
        ref_inv = inverse_sigmoid(reference_boxes)
        outputs_coord = (ref_inv + box_offsets).sigmoid()

        # --- Detection scoring (DotProductScoring with all text pooled) ---
        det_scores = model.dot_prod_scoring(hs, prompt, prompt_mask)
        # det_scores: (num_layers, 1, Q, 1) → last layer
        det_probs = det_scores[-1, 0, :, 0].float().sigmoid()  # (Q,)

        # --- Class assignment (depends on class_method) ---
        best_class_ids = self._assign_classes(hs, attn_container)

        # --- Filter by detection confidence ---
        keep = det_probs > confidence_threshold
        if not keep.any():
            return self._empty_result(orig_h, orig_w)

        scores_k = det_probs[keep]
        class_ids_k = best_class_ids[keep]
        boxes_k = outputs_coord[-1, 0, keep]  # (K, 4) cxcywh

        # --- Convert boxes to output format ---
        scale = torch.tensor(
            [orig_w, orig_h, orig_w, orig_h],
            device=self.device, dtype=torch.float32,
        )
        boxes_xyxy = box_cxcywh_to_xyxy(boxes_k.float()) * scale

        if self.detection_only:
            # Box-based NMS, skip mask generation entirely
            if nms_threshold < 1.0 and len(scores_k) > 0:
                if per_class_nms:
                    nms_keep = _batched_nms(boxes_xyxy, scores_k, class_ids_k, nms_threshold)
                else:
                    nms_keep = _nms(boxes_xyxy, scores_k, nms_threshold)
                scores_k = scores_k[nms_keep]
                class_ids_k = class_ids_k[nms_keep]
                boxes_xyxy = boxes_xyxy[nms_keep]

            sort_idx = scores_k.argsort(descending=True)
            return {
                "boxes": boxes_xyxy[sort_idx],
                "masks": None,
                "masks_logits": None,
                "scores": scores_k[sort_idx],
                "class_ids": class_ids_k[sort_idx],
                "class_names": [
                    self._class_names[c] for c in class_ids_k[sort_idx].tolist()
                ],
            }

        # --- Lazy mask generation: only for kept queries ---
        hs_kept = hs[:, :, keep]  # (layers, 1, K, d)
        seg_out = model.segmentation_head(
            backbone_feats=backbone_out["backbone_fpn"],
            obj_queries=hs_kept,
            image_ids=img_ids,
            encoder_hidden_states=enc_hs,
            prompt=prompt,
            prompt_mask=prompt_mask,
        )
        masks_k = seg_out["pred_masks"][0]  # (K, H, W)

        masks_logits = interpolate(
            masks_k.float().unsqueeze(1),
            (orig_h, orig_w),
            mode="bilinear",
            align_corners=False,
        ).sigmoid()  # (K, 1, H, W)
        masks_binary = (masks_logits > 0.5).squeeze(1)  # (K, H, W)

        # --- Mask-based NMS ---
        if nms_threshold < 1.0 and len(scores_k) > 0:
            nms_keep = self._nms(
                scores=scores_k,
                masks=masks_binary,
                class_ids=class_ids_k,
                iou_threshold=nms_threshold,
                per_class=per_class_nms,
            )
            scores_k = scores_k[nms_keep]
            class_ids_k = class_ids_k[nms_keep]
            boxes_xyxy = boxes_xyxy[nms_keep]
            masks_binary = masks_binary[nms_keep]
            masks_logits = masks_logits[nms_keep]

        # --- Sort by score ---
        sort_idx = scores_k.argsort(descending=True)

        return {
            "boxes": boxes_xyxy[sort_idx],
            "masks": masks_binary[sort_idx],
            "masks_logits": masks_logits[sort_idx],
            "scores": scores_k[sort_idx],
            "class_ids": class_ids_k[sort_idx],
            "class_names": [
                self._class_names[c] for c in class_ids_k[sort_idx].tolist()
            ],
        }

    def _assign_classes(
        self,
        hs: torch.Tensor,
        attn_container: Dict,
    ) -> torch.Tensor:
        """Assign class IDs to each query using the configured method.

        Args:
            hs: Decoder hidden states, (num_layers, 1, Q, d).
            attn_container: Dict with "weights" key if attention method was used.

        Returns:
            (Q,) tensor of class IDs.
        """
        Q = hs.shape[2]  # num queries
        if self._class_method == "attention":
            return self._assign_classes_attention(attn_container, num_queries=Q)
        elif self._class_method == "prototype":
            return self._assign_classes_prototype(hs)
        else:  # cosine
            return self._assign_classes_cosine(hs)

    def _assign_classes_cosine(self, hs: torch.Tensor) -> torch.Tensor:
        """Cosine similarity between hs_proj(hs) and per-class text embeddings."""
        scoring = self.model.dot_prod_scoring
        proj_hs = scoring.hs_proj(hs[-1, 0])  # (Q, d_proj)
        proj_hs_norm = F.normalize(proj_hs.float(), dim=-1)
        cosine_scores = proj_hs_norm @ self._class_proj_norm.T  # (Q, N)
        return cosine_scores.argmax(dim=-1)  # (Q,)

    def _assign_classes_attention(
        self, attn_container: Dict, num_queries: int,
    ) -> torch.Tensor:
        """Sum decoder ca_text attention weights per class token group.

        The attention hook captured weights of shape (bs, Q', total_seq)
        from the last decoder layer's text cross-attention.  Q' may be
        larger than Q if the decoder prepends a presence token — we strip
        that extra row.  We sum the attention each query pays to each
        class's token range and pick the class with highest total attention.
        """
        weights = attn_container.get("weights")
        if weights is None:
            raise RuntimeError(
                "Attention weights not captured. "
                "Ensure torch.compile is disabled when using class_method='attention'."
            )
        # weights: (1, Q', total_seq) → (Q', total_seq)
        attn = weights.squeeze(0).float()  # (Q', total_seq)

        # Strip presence token row(s) if Q' > num_queries
        if attn.shape[0] > num_queries:
            # Presence token is prepended, so take the last num_queries rows
            attn = attn[-num_queries:]

        N = self._num_classes
        Q = attn.shape[0]
        class_attn = torch.zeros(Q, N, device=attn.device)
        for i, (start, end) in enumerate(self._class_token_ranges):
            class_attn[:, i] = attn[:, start:end].sum(dim=-1)
        return class_attn.argmax(dim=-1)  # (Q,)

    def _assign_classes_prototype(self, hs: torch.Tensor) -> torch.Tensor:
        """Cosine similarity between hs_proj(hs) and calibrated class prototypes."""
        scoring = self.model.dot_prod_scoring
        proj_hs = scoring.hs_proj(hs[-1, 0])  # (Q, d_proj)
        proj_hs_norm = F.normalize(proj_hs.float(), dim=-1)
        cosine_scores = proj_hs_norm @ self._calibrated_prototypes.T  # (Q, N)
        return cosine_scores.argmax(dim=-1)  # (Q,)

    # ------------------------------------------------------------------
    # Internal: post-processing (for batched / shared-encoder modes)
    # ------------------------------------------------------------------

    def _postprocess(
        self,
        batched: Dict,
        backbone_out: Dict,
        img_ids: torch.Tensor,
        orig_h: int,
        orig_w: int,
        confidence_threshold: float,
        nms_threshold: float,
        per_class_nms: bool,
    ) -> Dict:
        """Post-process batched outputs into final predictions.

        For detection_only mode, uses fully vectorized tensor ops across all
        classes (no Python loop).  For mask mode, uses lazy per-class mask
        generation for the ~5-15 queries above the confidence threshold.
        """
        present_idx = batched["present_indices"]  # (K,)
        scores_all = batched["scores_all"]  # (N, Q, 1)
        boxes_all = batched["boxes_all"]    # (N, Q, 4) cxcywh
        presence_probs = batched["presence_probs"]  # (N,) or None

        scale = torch.tensor(
            [orig_w, orig_h, orig_w, orig_h],
            device=self.device, dtype=torch.float32,
        )

        if self.detection_only:
            return self._postprocess_detection(
                present_idx, scores_all, boxes_all, presence_probs,
                scale, orig_h, orig_w,
                confidence_threshold, nms_threshold, per_class_nms,
            )

        return self._postprocess_with_masks(
            batched, backbone_out, img_ids,
            present_idx, scores_all, boxes_all, presence_probs,
            scale, orig_h, orig_w,
            confidence_threshold, nms_threshold, per_class_nms,
        )

    def _postprocess_detection(
        self,
        present_idx: torch.Tensor,
        scores_all: torch.Tensor,
        boxes_all: torch.Tensor,
        presence_probs,
        scale: torch.Tensor,
        orig_h: int,
        orig_w: int,
        confidence_threshold: float,
        nms_threshold: float,
        per_class_nms: bool,
    ) -> Dict:
        """Vectorized detection-only postprocess (no Python per-class loop)."""
        # Sigmoid scores for all present classes at once: (K, Q)
        logits = scores_all[present_idx, :, 0].float()
        probs = logits.sigmoid()

        # Weight by presence probability
        if presence_probs is not None:
            probs = probs * presence_probs[present_idx].float().unsqueeze(1)

        # Confidence filter across all classes and queries
        mask = probs > confidence_threshold  # (K, Q)
        k_idx, q_idx = mask.nonzero(as_tuple=True)

        if k_idx.shape[0] == 0:
            return self._empty_result(orig_h, orig_w)

        # Gather kept detections
        scores = probs[k_idx, q_idx]
        class_ids = present_idx[k_idx]
        boxes_cxcy = boxes_all[class_ids, q_idx]  # (D, 4)
        boxes_xyxy = box_cxcywh_to_xyxy(boxes_cxcy.float()) * scale

        # Box-based NMS (torchvision, CUDA-accelerated)
        if nms_threshold < 1.0 and scores.shape[0] > 0:
            if per_class_nms:
                nms_keep = _batched_nms(boxes_xyxy, scores, class_ids, nms_threshold)
            else:
                nms_keep = _nms(boxes_xyxy, scores, nms_threshold)
            scores = scores[nms_keep]
            class_ids = class_ids[nms_keep]
            boxes_xyxy = boxes_xyxy[nms_keep]

        sort_idx = scores.argsort(descending=True)
        return {
            "boxes": boxes_xyxy[sort_idx],
            "masks": None,
            "masks_logits": None,
            "scores": scores[sort_idx],
            "class_ids": class_ids[sort_idx],
            "class_names": [
                self._class_names[c] for c in class_ids[sort_idx].tolist()
            ],
        }

    def _postprocess_with_masks(
        self,
        batched: Dict,
        backbone_out: Dict,
        img_ids: torch.Tensor,
        present_idx: torch.Tensor,
        scores_all: torch.Tensor,
        boxes_all: torch.Tensor,
        presence_probs,
        scale: torch.Tensor,
        orig_h: int,
        orig_w: int,
        confidence_threshold: float,
        nms_threshold: float,
        per_class_nms: bool,
    ) -> Dict:
        """Postprocess with lazy per-class mask generation."""
        model = self.model

        all_scores = []
        all_class_ids = []
        all_boxes = []
        all_masks_logits = []

        for class_idx in present_idx.tolist():
            logits = scores_all[class_idx, :, 0]
            probs = logits.float().sigmoid()

            if presence_probs is not None:
                probs = probs * presence_probs[class_idx].float()

            keep = probs > confidence_threshold
            if not keep.any():
                continue

            scores_k = probs[keep]
            boxes_k = boxes_all[class_idx][keep]
            boxes_xyxy = box_cxcywh_to_xyxy(boxes_k.float()) * scale

            hs_kept = batched["hs_all"][:, class_idx:class_idx+1, keep]
            with torch.autocast("cuda", dtype=torch.float16, enabled=self.use_fp16):
                seg_out = model.segmentation_head(
                    backbone_feats=backbone_out["backbone_fpn"],
                    obj_queries=hs_kept,
                    image_ids=img_ids,
                    encoder_hidden_states=batched["encoder_hidden_states"][:, class_idx:class_idx+1],
                    prompt=batched["prompt"][:, class_idx:class_idx+1],
                    prompt_mask=batched["prompt_mask"][class_idx:class_idx+1],
                )
            masks_k = seg_out["pred_masks"][0]
            masks_logits_k = interpolate(
                masks_k.float().unsqueeze(1),
                (orig_h, orig_w),
                mode="bilinear",
                align_corners=False,
            ).sigmoid()
            all_masks_logits.append(masks_logits_k)

            all_scores.append(scores_k)
            all_class_ids.append(
                torch.full_like(scores_k, class_idx, dtype=torch.long)
            )
            all_boxes.append(boxes_xyxy)

        if not all_scores:
            return self._empty_result(orig_h, orig_w)

        scores = torch.cat(all_scores)
        class_ids = torch.cat(all_class_ids)
        boxes_xyxy = torch.cat(all_boxes)
        masks_logits = torch.cat(all_masks_logits)
        masks_binary = (masks_logits > 0.5).squeeze(1)

        if nms_threshold < 1.0 and len(scores) > 0:
            nms_keep = self._nms(
                scores=scores,
                masks=masks_binary,
                class_ids=class_ids,
                iou_threshold=nms_threshold,
                per_class=per_class_nms,
            )
            scores = scores[nms_keep]
            class_ids = class_ids[nms_keep]
            boxes_xyxy = boxes_xyxy[nms_keep]
            masks_binary = masks_binary[nms_keep]
            masks_logits = masks_logits[nms_keep]

        sort_idx = scores.argsort(descending=True)

        return {
            "boxes": boxes_xyxy[sort_idx],
            "masks": masks_binary[sort_idx],
            "masks_logits": masks_logits[sort_idx],
            "scores": scores[sort_idx],
            "class_ids": class_ids[sort_idx],
            "class_names": [
                self._class_names[c] for c in class_ids[sort_idx].tolist()
            ],
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _empty_result(self, orig_h: int, orig_w: int) -> Dict:
        """Return an empty predictions dict."""
        if self.detection_only:
            return {
                "boxes": torch.zeros(0, 4, device=self.device),
                "masks": None,
                "masks_logits": None,
                "scores": torch.zeros(0, device=self.device),
                "class_ids": torch.zeros(0, device=self.device, dtype=torch.long),
                "class_names": [],
            }
        return {
            "boxes": torch.zeros(0, 4, device=self.device),
            "masks": torch.zeros(
                0, orig_h, orig_w, device=self.device, dtype=torch.bool
            ),
            "masks_logits": torch.zeros(0, 1, orig_h, orig_w, device=self.device),
            "scores": torch.zeros(0, device=self.device),
            "class_ids": torch.zeros(0, device=self.device, dtype=torch.long),
            "class_names": [],
        }

    @staticmethod
    def _mask_iou(mask_a: torch.Tensor, mask_b: torch.Tensor) -> torch.Tensor:
        """Compute IoU between two binary masks."""
        intersection = (mask_a & mask_b).sum().float()
        union = (mask_a | mask_b).sum().float()
        return intersection / union.clamp(min=1.0)

    def _nms(
        self,
        scores: torch.Tensor,
        masks: torch.Tensor,
        class_ids: torch.Tensor,
        iou_threshold: float,
        per_class: bool,
    ) -> torch.Tensor:
        """Greedy mask-based NMS."""
        order = scores.argsort(descending=True)
        keep = []
        for i in order.tolist():
            should_keep = True
            for j in keep:
                if per_class and class_ids[i] != class_ids[j]:
                    continue
                iou = self._mask_iou(masks[i], masks[j])
                if iou > iou_threshold:
                    should_keep = False
                    break
            if should_keep:
                keep.append(i)
        return torch.tensor(keep, device=scores.device, dtype=torch.long)
