# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""
Multi-class inference for SAM3.

Adapts SAM3's contrastive per-prompt architecture to open-vocabulary
multi-class detection + segmentation by sharing the expensive backbone
computation across classes while running the lightweight encoder+decoder
per class.

SAM3 was trained with single-class contrastive prompts, so the decoder's
text cross-attention produces class-specific query features only when a
single class prompt is provided.  This module runs the backbone once and
then loops over classes through the encoder+decoder, collecting per-class
detections and merging them with cross-class NMS.

Compute comparison (for N classes):
  Original SAM3:   N × (backbone + encoder + decoder + masks)
  This module:     1 × backbone + N × (encoder + decoder + masks)
  Savings:         backbone accounts for ~80% of total compute

Usage:
    from sam3.model.sam3_multiclass import Sam3MultiClassPredictor

    predictor = Sam3MultiClassPredictor(model, device="cuda")
    predictor.set_classes(["car", "pedestrian", "bicycle"])

    state = predictor.set_image(image)
    results = predictor.predict(state, confidence_threshold=0.3)

    # results["boxes"]       : (K, 4) xyxy pixel coordinates
    # results["masks"]       : (K, H, W) binary masks
    # results["scores"]      : (K,) confidence scores
    # results["class_ids"]   : (K,) integer class indices
    # results["class_names"] : list[str], class name per detection
"""

from typing import Dict, List, Optional, Union

import numpy as np
import PIL
import torch
from torchvision.ops import batched_nms as _batched_nms, nms as _nms
from torchvision.transforms import v2

from sam3.model.box_ops import box_cxcywh_to_xyxy
from sam3.model.data_misc import interpolate
from sam3.model.model_misc import inverse_sigmoid


class Sam3MultiClassPredictor:
    """Multi-class inference wrapper for SAM3.

    Runs the backbone once per image and loops the lightweight
    encoder+decoder per class to produce class-discriminative detections.
    """

    def __init__(
        self,
        model,
        resolution: int = 1008,
        device: str = "cuda",
        detection_only: bool = False,
    ):
        """
        Args:
            model: A Sam3Image model instance (already loaded with weights).
            resolution: Input image resolution (default 1008 to match SAM3).
            device: Torch device for inference.
            detection_only: If True, skip mask generation and return only
                boxes + scores.  Uses box-based NMS.
        """
        self.model = model
        self.resolution = resolution
        self.device = device
        self.detection_only = detection_only

        self.transform = v2.Compose(
            [
                v2.ToDtype(torch.uint8, scale=True),
                v2.Resize(size=(resolution, resolution)),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

        # Class embedding cache (populated by set_classes)
        self._class_names: Optional[List[str]] = None
        self._num_classes: int = 0

        # Per-class text features and masks
        # Each entry: text_feats (seq, 1, d), text_mask (1, seq)
        self._per_class_text: Optional[List[torch.Tensor]] = None
        self._per_class_mask: Optional[List[torch.Tensor]] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def set_classes(self, class_names: List[str]) -> None:
        """Pre-compute and cache text embeddings for all target classes.

        Runs the text encoder once for all classes and stores per-class
        text features separately (not concatenated) so each class can
        be processed independently through the encoder+decoder.

        Can be called once and reused across many images.

        Args:
            class_names: List of class names, e.g. ["car", "pedestrian"].
        """
        if not class_names:
            raise ValueError("class_names must be a non-empty list of strings")

        self._class_names = list(class_names)
        self._num_classes = len(class_names)

        # Run text encoder for all classes at once (efficient batching)
        text_outputs = self.model.backbone.forward_text(
            class_names, device=self.device
        )
        # language_features: (seq_len, N, d_model) — seq-first
        # language_mask:     (N, seq_len) — True = padding token
        text_feats = text_outputs["language_features"]
        text_masks = text_outputs["language_mask"]

        # Store per-class: each class gets (seq, 1, d) and (1, seq)
        self._per_class_text = []
        self._per_class_mask = []
        for i in range(self._num_classes):
            self._per_class_text.append(text_feats[:, i : i + 1, :])  # (seq, 1, d)
            self._per_class_mask.append(text_masks[i : i + 1, :])  # (1, seq)

    @torch.inference_mode()
    def set_image(
        self,
        image: Union[PIL.Image.Image, torch.Tensor, np.ndarray],
        state: Optional[Dict] = None,
    ) -> Dict:
        """Encode an image through the vision backbone.

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

        image_tensor = v2.functional.to_image(image).to(self.device)
        image_tensor = self.transform(image_tensor).unsqueeze(0)

        state["original_height"] = height
        state["original_width"] = width
        state["backbone_out"] = self.model.backbone.forward_image(image_tensor)

        return state

    @torch.inference_mode()
    def predict(
        self,
        state: Dict,
        confidence_threshold: float = 0.3,
        nms_threshold: float = 0.7,
        per_class_nms: bool = True,
    ) -> Dict:
        """Run multi-class detection + segmentation.

        Runs the encoder+decoder once per class (sharing cached backbone
        features), then merges all per-class detections and applies NMS.

        Args:
            state: State dict from set_image() containing backbone features.
            confidence_threshold: Minimum score to keep a detection.
            nms_threshold: IoU threshold for mask-based NMS.
            per_class_nms: If True, run NMS independently per class.
                If False, run cross-class NMS (no overlapping detections).

        Returns:
            Dict with keys:
              - "boxes": (K, 4) xyxy boxes in pixel coordinates
              - "masks": (K, H, W) binary masks at original resolution
              - "masks_logits": (K, 1, H, W) continuous mask logits
              - "scores": (K,) confidence scores
              - "class_ids": (K,) integer class indices
              - "class_names": list[str] of length K
        """
        if self._class_names is None:
            raise RuntimeError("Call set_classes() before predict()")
        if "backbone_out" not in state:
            raise RuntimeError("Call set_image() before predict()")

        backbone_out = state["backbone_out"]
        orig_h = state["original_height"]
        orig_w = state["original_width"]

        # Prepare image features once (shared across all classes)
        img_ids = torch.tensor([0], device=self.device, dtype=torch.long)
        backbone_out, img_feats, img_pos_embeds, vis_feat_sizes = (
            self.model._get_img_feats(backbone_out, img_ids)
        )

        # Collect per-class detections
        all_scores = []
        all_class_ids = []
        all_boxes = []
        all_masks_logits = []

        for class_idx in range(self._num_classes):
            class_out = self._forward_single_class(
                img_feats=img_feats,
                img_pos_embeds=img_pos_embeds,
                vis_feat_sizes=vis_feat_sizes,
                class_idx=class_idx,
            )

            # Extract single-class scores and apply confidence filter
            # pred_logits: (1, num_queries, 1) — single class score
            logits = class_out["pred_logits"].squeeze(0)  # (Q, 1)
            probs = logits.sigmoid().squeeze(-1)  # (Q,)

            # Apply presence score if available
            if class_out.get("presence_logit_dec") is not None:
                presence = class_out["presence_logit_dec"].sigmoid().squeeze()
                probs = probs * presence

            # Filter by confidence
            keep = probs > confidence_threshold
            if not keep.any():
                continue

            scores_k = probs[keep]
            boxes_k = class_out["pred_boxes"].squeeze(0)[keep]  # (K, 4) cxcywh

            # Convert boxes to xyxy at original resolution
            boxes_xyxy = box_cxcywh_to_xyxy(boxes_k)
            scale = torch.tensor(
                [orig_w, orig_h, orig_w, orig_h],
                device=self.device,
                dtype=boxes_xyxy.dtype,
            )
            boxes_xyxy = boxes_xyxy * scale

            if not self.detection_only:
                # --- Lazy mask generation: only for kept queries ---
                hs_kept = class_out["hs"][:, :, keep]  # (layers, 1, K, d)
                seg_out = self.model.segmentation_head(
                    backbone_feats=backbone_out["backbone_fpn"],
                    obj_queries=hs_kept,
                    image_ids=img_ids,
                    encoder_hidden_states=class_out["encoder_hidden_states"],
                    prompt=class_out["prompt"],
                    prompt_mask=class_out["prompt_mask"],
                )
                masks_k = seg_out["pred_masks"][0]  # (K, H, W)
                masks_logits_k = interpolate(
                    masks_k.float().unsqueeze(1),
                    (orig_h, orig_w),
                    mode="bilinear",
                    align_corners=False,
                ).sigmoid()  # (K, 1, H, W)
                all_masks_logits.append(masks_logits_k)

            all_scores.append(scores_k)
            all_class_ids.append(
                torch.full_like(scores_k, class_idx, dtype=torch.long)
            )
            all_boxes.append(boxes_xyxy)

        # Handle no detections
        if not all_scores:
            return self._empty_result(orig_h, orig_w)

        # Merge all classes
        scores = torch.cat(all_scores)
        class_ids = torch.cat(all_class_ids)
        boxes_xyxy = torch.cat(all_boxes)

        if self.detection_only:
            # Box-based NMS (torchvision, CUDA-accelerated)
            if nms_threshold < 1.0 and len(scores) > 0:
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
                "class_names": [self._class_names[c] for c in class_ids[sort_idx].tolist()],
            }

        masks_logits = torch.cat(all_masks_logits)
        masks_binary = (masks_logits > 0.5).squeeze(1)  # (K, H, W)

        # Apply mask-based NMS
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

        # Sort by score (descending)
        sort_idx = scores.argsort(descending=True)

        return {
            "boxes": boxes_xyxy[sort_idx],
            "masks": masks_binary[sort_idx],
            "masks_logits": masks_logits[sort_idx],
            "scores": scores[sort_idx],
            "class_ids": class_ids[sort_idx],
            "class_names": [self._class_names[c] for c in class_ids[sort_idx].tolist()],
        }

    @torch.inference_mode()
    def predict_image(
        self,
        image: Union[PIL.Image.Image, torch.Tensor, np.ndarray],
        confidence_threshold: float = 0.3,
        nms_threshold: float = 0.7,
        per_class_nms: bool = True,
    ) -> Dict:
        """Convenience: set_image + predict in one call.

        Args:
            image: Input image.
            confidence_threshold: Minimum detection score.
            nms_threshold: NMS IoU threshold.
            per_class_nms: Per-class vs cross-class NMS.

        Returns:
            Predictions dict (same as predict()).
        """
        state = self.set_image(image)
        return self.predict(
            state,
            confidence_threshold=confidence_threshold,
            nms_threshold=nms_threshold,
            per_class_nms=per_class_nms,
        )

    # ------------------------------------------------------------------
    # Internal: per-class forward
    # ------------------------------------------------------------------

    def _forward_single_class(
        self,
        img_feats: list,
        img_pos_embeds: list,
        vis_feat_sizes: list,
        class_idx: int,
    ) -> Dict:
        """Run encoder+decoder+scoring for a single class (no masks).

        Reuses cached backbone features (img_feats, img_pos_embeds) so the
        expensive backbone is NOT re-run.  Only the lightweight encoder
        (~1.9% of compute) and decoder (~3.4%) run per class.

        Mask generation is deferred to the caller, which runs the seg head
        only for queries above the confidence threshold (lazy mask gen).

        Args:
            img_feats: Pre-extracted image features from _get_img_feats.
            img_pos_embeds: Pre-extracted positional embeddings.
            vis_feat_sizes: Spatial sizes per FPN level.
            class_idx: Index into self._per_class_text/mask.

        Returns:
            Dict with scores, boxes, hs, encoder state (no masks).
        """
        model = self.model

        # --- Text prompt for this class (single class) ---
        prompt = self._per_class_text[class_idx]  # (seq, 1, d)
        prompt_mask = self._per_class_mask[class_idx]  # (1, seq)

        # --- Encoder (single class prompt) ---
        prompt_pos_embed = torch.zeros_like(prompt)
        memory = model.transformer.encoder(
            src=[f.clone() for f in img_feats],
            src_key_padding_mask=None,
            src_pos=[p.clone() for p in img_pos_embeds],
            prompt=prompt,
            prompt_pos=prompt_pos_embed,
            prompt_key_padding_mask=prompt_mask,
            feat_sizes=vis_feat_sizes,
        )

        # --- Decoder (single class prompt, bs=1) ---
        bs = 1
        query_embed = model.transformer.decoder.query_embed.weight
        tgt = query_embed.unsqueeze(1).repeat(1, bs, 1)

        hs, reference_boxes, dec_presence_out, _presence_feats = (
            model.transformer.decoder(
                tgt=tgt,
                memory=memory["memory"],
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
        )
        # hs: (num_layers, num_queries, bs, d_model)
        hs = hs.transpose(1, 2)  # (num_layers, bs=1, num_queries, d_model)
        reference_boxes = reference_boxes.transpose(1, 2)

        # --- Scoring (original single-class dot-product — correct by design) ---
        scores = model.dot_prod_scoring(hs, prompt, prompt_mask)
        # scores: (num_layers, 1, num_queries, 1)

        # --- Box prediction ---
        box_head = model.transformer.decoder.bbox_embed
        anchor_box_offsets = box_head(hs)
        ref_inv_sig = inverse_sigmoid(reference_boxes)
        outputs_coord = (ref_inv_sig + anchor_box_offsets).sigmoid()

        # --- Assemble output (last decoder layer, no masks yet) ---
        out = {
            "pred_logits": scores[-1],  # (1, num_queries, 1)
            "pred_boxes": outputs_coord[-1],  # (1, num_queries, 4) cxcywh
            "hs": hs,  # (num_layers, 1, num_queries, d)
            "encoder_hidden_states": memory["memory"],
            "prompt": prompt,
            "prompt_mask": prompt_mask,
        }

        if dec_presence_out is not None:
            out["presence_logit_dec"] = dec_presence_out[-1].transpose(0, 1)
        else:
            out["presence_logit_dec"] = None

        return out

    # ------------------------------------------------------------------
    # Internal: post-processing helpers
    # ------------------------------------------------------------------

    def _empty_result(self, orig_h: int, orig_w: int) -> Dict:
        """Return an empty predictions dict when nothing is detected."""
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
        """Greedy mask-based NMS.

        Args:
            scores: (K,) detection scores.
            masks: (K, H, W) binary masks.
            class_ids: (K,) class assignments.
            iou_threshold: Suppress detections with IoU above this.
            per_class: If True, only suppress within same class.

        Returns:
            Indices of kept detections.
        """
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
