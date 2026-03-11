#!/usr/bin/env python3
"""
Train SAM3 for native multi-class single-pass inference.

Fine-tunes the decoder text cross-attention layers and adds a multi-class
scoring head, enabling all N classes to be predicted in a single
encoder+decoder pass instead of N per-class passes.

Trainable components (~15M params):
  - Decoder ca_text layers (6 layers) -- class-selective text attention
  - MultiClassScoring head -- per-query, per-class dot-product scoring
  - bbox_embed -- box prediction head
  - Optionally: decoder FFN layers

Frozen components:
  - ViT-H backbone (439M)
  - Text encoder (353M)
  - Encoder (12M)
  - Decoder self-attention, image cross-attention
  - Segmentation head (13M)

Training data: COCO train2017 with GT annotations.
Matching: Hungarian algorithm (per-image).
Losses: focal loss (classification) + L1 + GIoU (box regression).

Usage:
    # Single GPU
    python scripts/train_multiclass.py --coco-dir /path/to/coco

    # Multi-GPU with mp.spawn (8xH100)
    python scripts/train_multiclass.py --coco-dir /path/to/coco --num-gpus 8

    # Multi-GPU with torchrun
    torchrun --nproc_per_node=8 scripts/train_multiclass.py \\
        --coco-dir /path/to/coco
"""

import argparse
import math
import os
import sys
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from PIL import Image
from scipy.optimize import linear_sum_assignment
from torchvision.transforms import v2

from sam3.model.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
from sam3.model.model_misc import inverse_sigmoid
from sam3.model.multiclass_head import (
    MultiClassScoring,
    precompute_class_embeddings,
    precompute_concat_text,
)
from sam3.model_builder import build_sam3_image_model


# ---------------------------------------------------------------------------
# COCO 80 class names (canonical order matching pycocotools category IDs)
# ---------------------------------------------------------------------------
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


class COCOMultiClassDataset(Dataset):
    """COCO dataset with GT boxes and class labels for multi-class training.

    Loads images from COCO train/val directory and annotations from the
    standard instances JSON file.  Returns normalized cxcywh boxes and
    contiguous class indices (0-79).
    """

    def __init__(self, coco_root, split="train2017", resolution=1008):
        from pycocotools.coco import COCO

        self.img_dir = os.path.join(coco_root, split)
        ann_file = os.path.join(
            coco_root, "annotations", f"instances_{split}.json"
        )
        if not os.path.exists(ann_file):
            raise FileNotFoundError(f"Annotation file not found: {ann_file}")

        print(f"Loading COCO annotations from {ann_file}...")
        self.coco = COCO(ann_file)

        # Filter images that actually exist and have annotations
        all_img_ids = sorted(self.coco.getImgIds())
        self.img_ids = []
        for img_id in all_img_ids:
            ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=False)
            if ann_ids:
                self.img_ids.append(img_id)
        print(f"  {len(self.img_ids)} images with annotations "
              f"(of {len(all_img_ids)} total)")

        # Build category mapping: COCO category IDs (1-90 with gaps) -> 0-79
        cat_ids = sorted(self.coco.getCatIds())
        self.cat_id_to_idx = {cid: idx for idx, cid in enumerate(cat_ids)}
        cats = self.coco.loadCats(cat_ids)
        self.class_names = [c["name"] for c in cats]
        self.num_classes = len(cat_ids)

        self.transform = v2.Compose([
            v2.ToDtype(torch.uint8, scale=True),
            v2.Resize(size=(resolution, resolution)),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]

        img_path = os.path.join(self.img_dir, img_info["file_name"])
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Warning: failed to load {img_path}: {e}")
            # Return a blank image with no annotations
            image = Image.new("RGB", (640, 480))
            image_tensor = v2.functional.to_image(image)
            image_tensor = self.transform(image_tensor)
            return image_tensor, torch.zeros(0, 4), torch.zeros(0, dtype=torch.long)

        orig_w, orig_h = image.size
        image_tensor = v2.functional.to_image(image)
        image_tensor = self.transform(image_tensor)

        # Load annotations (skip crowd)
        ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=False)
        anns = self.coco.loadAnns(ann_ids)

        boxes = []
        labels = []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            if w < 1 or h < 1:
                continue
            # Convert to normalized cxcywh
            cx = (x + w / 2) / orig_w
            cy = (y + h / 2) / orig_h
            nw = w / orig_w
            nh = h / orig_h
            # Clamp to [0, 1]
            cx = max(0, min(1, cx))
            cy = max(0, min(1, cy))
            nw = max(0.001, min(1, nw))
            nh = max(0.001, min(1, nh))
            boxes.append([cx, cy, nw, nh])
            labels.append(self.cat_id_to_idx[ann["category_id"]])

        if boxes:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.long)
        else:
            boxes = torch.zeros(0, 4, dtype=torch.float32)
            labels = torch.zeros(0, dtype=torch.long)

        return image_tensor, boxes, labels


def collate_fn(batch):
    """Custom collate: stack images, keep boxes/labels as lists."""
    images = torch.stack([item[0] for item in batch])
    boxes = [item[1] for item in batch]
    labels = [item[2] for item in batch]
    return images, boxes, labels


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def sigmoid_focal_loss(logits, targets, alpha=0.25, gamma=2.0):
    """Sigmoid focal loss for multi-label classification.

    Args:
        logits: (Q, N) per-class logits for each query.
        targets: (Q, N) one-hot target (or soft target).

    Returns:
        Scalar loss, normalized by number of positive targets.
    """
    p = logits.sigmoid()
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    focal_weight = (1 - p_t) ** gamma
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        focal_weight = alpha_t * focal_weight
    loss = focal_weight * ce
    return loss.sum()


def box_regression_loss(pred_boxes, gt_boxes):
    """Combined L1 + GIoU box regression loss.

    Args:
        pred_boxes: (M, 4) predicted boxes in cxcywh.
        gt_boxes: (M, 4) GT boxes in cxcywh.

    Returns:
        l1_loss, giou_loss (both scalars).
    """
    l1_loss = F.l1_loss(pred_boxes, gt_boxes, reduction="sum")

    pred_xyxy = box_cxcywh_to_xyxy(pred_boxes)
    gt_xyxy = box_cxcywh_to_xyxy(gt_boxes)
    # generalized_box_iou returns (M, M) — we want diagonal
    giou_matrix = generalized_box_iou(pred_xyxy, gt_xyxy)
    giou_diag = torch.diag(giou_matrix)
    giou_loss = (1 - giou_diag).sum()

    return l1_loss, giou_loss


# ---------------------------------------------------------------------------
# Hungarian matching
# ---------------------------------------------------------------------------

@torch.no_grad()
def hungarian_match(cls_logits, pred_boxes, gt_boxes, gt_labels):
    """Compute optimal bipartite matching between queries and GT objects.

    Args:
        cls_logits: (Q, N) per-class logits per query.
        pred_boxes: (Q, 4) predicted boxes in cxcywh.
        gt_boxes: (M, 4) GT boxes in cxcywh.
        gt_labels: (M,) GT class indices.

    Returns:
        pred_indices: (K,) matched query indices.
        gt_indices: (K,) matched GT indices.
    """
    Q = cls_logits.shape[0]
    M = gt_boxes.shape[0]

    if M == 0:
        return (
            torch.tensor([], dtype=torch.long, device=cls_logits.device),
            torch.tensor([], dtype=torch.long, device=cls_logits.device),
        )

    # Classification cost (focal-style)
    alpha, gamma = 0.25, 2.0
    out_prob = cls_logits.sigmoid()  # (Q, N)
    # Cost: lower when predicted prob for GT class is higher
    pos_cost = alpha * ((1 - out_prob[:, gt_labels]) ** gamma) * (
        -(out_prob[:, gt_labels] + 1e-8).log()
    )
    neg_cost = (1 - alpha) * (out_prob[:, gt_labels] ** gamma) * (
        -(1 - out_prob[:, gt_labels] + 1e-8).log()
    )
    cls_cost = pos_cost - neg_cost  # (Q, M)

    # L1 cost
    l1_cost = torch.cdist(pred_boxes.float(), gt_boxes.float(), p=1)  # (Q, M)

    # GIoU cost
    pred_xyxy = box_cxcywh_to_xyxy(pred_boxes.float())
    gt_xyxy = box_cxcywh_to_xyxy(gt_boxes.float())
    giou = generalized_box_iou(pred_xyxy, gt_xyxy)  # (Q, M)
    giou_cost = -giou

    # Total cost
    C = 2.0 * cls_cost + 5.0 * l1_cost + 2.0 * giou_cost

    # Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(C.cpu().numpy())

    return (
        torch.tensor(row_ind, dtype=torch.long, device=cls_logits.device),
        torch.tensor(col_ind, dtype=torch.long, device=cls_logits.device),
    )


# ---------------------------------------------------------------------------
# Training forward module (DDP-compatible)
# ---------------------------------------------------------------------------

class MultiClassForward(nn.Module):
    """Forward pass through decoder + multi-class scoring.

    This module wraps only the components that participate in gradient
    computation.  The backbone and encoder are run separately under
    torch.no_grad() and their outputs are passed as inputs.

    Contains:
      - The full decoder (most params frozen, only ca_text trainable)
      - MultiClassScoring head (fully trainable)
      - bbox_embed is part of the decoder
    """

    def __init__(self, decoder, multiclass_scoring):
        super().__init__()
        self.decoder = decoder
        self.multiclass_scoring = multiclass_scoring

    def forward(
        self,
        encoder_memory,
        memory_padding_mask,
        memory_pos,
        concat_text,
        concat_mask,
        per_class_pooled_text,
        level_start_index,
        spatial_shapes,
        valid_ratios,
    ):
        """
        Args:
            encoder_memory: (total_tokens, B, d) from encoder.
            memory_padding_mask: (B, total_tokens) or None.
            memory_pos: (total_tokens, B, d) positional embeddings.
            concat_text: (total_seq, B, d) concatenated class text tokens.
            concat_mask: (B, total_seq) text padding mask.
            per_class_pooled_text: (N, d) mean-pooled text per class.
            level_start_index: (num_levels,) from encoder.
            spatial_shapes: (num_levels, 2) from encoder.
            valid_ratios: (B, num_levels, 2) from encoder.

        Returns:
            cls_logits: (num_layers, B, Q, N) per-class logits.
            pred_boxes: (num_layers, B, Q, 4) predicted boxes (cxcywh, sigmoid).
        """
        B = encoder_memory.shape[1]
        query_embed = self.decoder.query_embed.weight
        tgt = query_embed.unsqueeze(1).expand(-1, B, -1)

        hs, ref_boxes, _, _ = self.decoder(
            tgt=tgt,
            memory=encoder_memory,
            memory_key_padding_mask=memory_padding_mask,
            pos=memory_pos,
            reference_boxes=None,
            level_start_index=level_start_index,
            spatial_shapes=spatial_shapes,
            valid_ratios=valid_ratios,
            tgt_mask=None,
            memory_text=concat_text,
            text_attention_mask=concat_mask,
            apply_dac=False,
        )

        # (num_layers, Q, B, d) -> (num_layers, B, Q, d)
        hs = hs.transpose(1, 2)
        ref_boxes = ref_boxes.transpose(1, 2)

        # Box prediction: offset + reference
        box_offsets = self.decoder.bbox_embed(hs)
        ref_inv = inverse_sigmoid(ref_boxes)
        pred_boxes = (ref_inv + box_offsets).sigmoid()

        # Multi-class scoring
        cls_logits = self.multiclass_scoring(hs, per_class_pooled_text)

        return cls_logits, pred_boxes


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class MultiClassTrainer:
    """Orchestrates multi-class single-pass training."""

    def __init__(
        self,
        model,
        multiclass_scoring,
        device,
        rank=0,
        world_size=1,
        lr=1e-4,
        weight_decay=0.05,
        warmup_steps=500,
        max_grad_norm=0.1,
        cls_weight=2.0,
        l1_weight=5.0,
        giou_weight=2.0,
        focal_alpha=0.25,
        focal_gamma=2.0,
        train_decoder_ffn=False,
    ):
        self.model = model
        self.device = device
        self.rank = rank
        self.world_size = world_size
        self.cls_weight = cls_weight
        self.l1_weight = l1_weight
        self.giou_weight = giou_weight
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.warmup_steps = warmup_steps
        self.max_grad_norm = max_grad_norm

        # Freeze everything
        for param in model.parameters():
            param.requires_grad = False

        # Unfreeze decoder ca_text layers + associated norm and dropout
        for layer in model.transformer.decoder.layers:
            if hasattr(layer, "ca_text"):
                for param in layer.ca_text.parameters():
                    param.requires_grad = True
            if hasattr(layer, "catext_norm"):
                for param in layer.catext_norm.parameters():
                    param.requires_grad = True

        # Optionally unfreeze decoder FFN (linear1, linear2, norm3)
        if train_decoder_ffn:
            for layer in model.transformer.decoder.layers:
                for attr in ["linear1", "linear2", "norm3"]:
                    if hasattr(layer, attr):
                        for param in getattr(layer, attr).parameters():
                            param.requires_grad = True

        # Unfreeze bbox_embed
        for param in model.transformer.decoder.bbox_embed.parameters():
            param.requires_grad = True

        # MultiClassScoring is fully trainable (already has requires_grad=True)
        self.multiclass_scoring = multiclass_scoring.to(device)

        # Build DDP-wrapped forward module
        self.forward_module = MultiClassForward(
            model.transformer.decoder, multiclass_scoring
        )

        if world_size > 1:
            self.forward_module = DDP(
                self.forward_module,
                device_ids=[rank] if torch.cuda.is_available() else None,
                find_unused_parameters=True,
            )

        # Collect trainable parameters
        trainable_params = [
            p for p in self.forward_module.parameters() if p.requires_grad
        ]
        self.num_trainable = sum(p.numel() for p in trainable_params)

        if rank == 0:
            total = sum(p.numel() for p in model.parameters()) + sum(
                p.numel() for p in multiclass_scoring.parameters()
            )
            print(
                f"Trainable: {self.num_trainable:,} / {total:,} "
                f"({100 * self.num_trainable / total:.2f}%)"
            )

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            trainable_params, lr=lr, weight_decay=weight_decay
        )
        self.scaler = GradScaler("cuda")
        self.base_lr = lr

        # Pre-computed text embeddings (set by precompute_text)
        self.concat_text = None
        self.concat_mask = None
        self.class_embeddings = None
        self.num_classes = 0

    def precompute_text(self, class_names):
        """Pre-compute and cache text embeddings for all classes."""
        self.num_classes = len(class_names)

        if self.rank == 0:
            print(f"Pre-computing text embeddings for {len(class_names)} classes...")

        # Concatenated text for encoder/decoder
        concat_text, concat_mask, _, _ = precompute_concat_text(
            self.model, class_names, device=self.device
        )
        self.concat_text = concat_text
        self.concat_mask = concat_mask

        # Per-class pooled embeddings for MultiClassScoring
        self.class_embeddings = precompute_class_embeddings(
            self.model, class_names, device=self.device
        )

        if self.rank == 0:
            total_seq = concat_text.shape[0]
            print(f"  Concat text: {total_seq} tokens, "
                  f"class embeddings: {self.class_embeddings.shape}")

    def _warmup_lr(self, step):
        """Linear warmup for learning rate."""
        if step < self.warmup_steps:
            lr = self.base_lr * (step + 1) / self.warmup_steps
            for pg in self.optimizer.param_groups:
                pg["lr"] = lr

    def _compute_loss(self, cls_logits, pred_boxes, gt_boxes_list, gt_labels_list):
        """Compute per-batch losses with Hungarian matching.

        Args:
            cls_logits: (num_layers, B, Q, N) per-class logits.
            pred_boxes: (num_layers, B, Q, 4) predicted boxes.
            gt_boxes_list: list of (Mi, 4) GT boxes per image.
            gt_labels_list: list of (Mi,) GT labels per image.

        Returns:
            loss_dict: {cls, l1, giou, total} scalar losses.
        """
        B = cls_logits.shape[1]
        Q = cls_logits.shape[2]
        N = cls_logits.shape[3]

        # Use last decoder layer for matching
        cls_last = cls_logits[-1]  # (B, Q, N)
        boxes_last = pred_boxes[-1]  # (B, Q, 4)

        total_cls = torch.tensor(0.0, device=cls_logits.device)
        total_l1 = torch.tensor(0.0, device=cls_logits.device)
        total_giou = torch.tensor(0.0, device=cls_logits.device)
        num_matched = 0

        for b in range(B):
            gt_boxes = gt_boxes_list[b].to(self.device)
            gt_labels = gt_labels_list[b].to(self.device)
            M = gt_boxes.shape[0]

            # Hungarian matching
            pred_idx, gt_idx = hungarian_match(
                cls_last[b], boxes_last[b], gt_boxes, gt_labels
            )

            # Classification loss: all queries
            # Target: matched queries get their GT class, unmatched get all zeros
            cls_target = torch.zeros(Q, N, device=cls_logits.device)
            if len(pred_idx) > 0:
                cls_target[pred_idx, gt_labels[gt_idx]] = 1.0
            total_cls += sigmoid_focal_loss(
                cls_last[b], cls_target,
                alpha=self.focal_alpha, gamma=self.focal_gamma,
            )

            # Box losses: only matched queries
            if len(pred_idx) > 0:
                l1, giou = box_regression_loss(
                    boxes_last[b][pred_idx], gt_boxes[gt_idx]
                )
                total_l1 += l1
                total_giou += giou
                num_matched += len(pred_idx)

        # Normalize
        num_matched = max(num_matched, 1)
        total_cls = total_cls / B
        total_l1 = total_l1 / num_matched
        total_giou = total_giou / num_matched

        total = (
            self.cls_weight * total_cls
            + self.l1_weight * total_l1
            + self.giou_weight * total_giou
        )

        return {
            "cls": total_cls.item(),
            "l1": total_l1.item(),
            "giou": total_giou.item(),
            "total": total.item(),
            "loss": total,
            "num_matched": num_matched,
        }

    def train_epoch(self, dataloader, epoch, total_epochs, global_step=0):
        """Train for one epoch.

        Returns:
            global_step: updated global step counter.
            avg_losses: dict of average losses for the epoch.
        """
        self.forward_module.train()
        # Keep backbone and encoder in eval mode
        self.model.backbone.eval()
        self.model.transformer.encoder.eval()

        accum_losses = {"cls": 0, "l1": 0, "giou": 0, "total": 0}
        num_batches = 0
        t0 = time.perf_counter()

        for batch_idx, (images, gt_boxes_list, gt_labels_list) in enumerate(dataloader):
            global_step += 1
            self._warmup_lr(global_step)

            images = images.to(self.device)
            B = images.shape[0]

            # --- Backbone + encoder (frozen, no_grad) ---
            with torch.no_grad():
                backbone_out = self.model.backbone.forward_image(images)
                img_ids = torch.arange(B, device=self.device, dtype=torch.long)
                backbone_out, img_feats, img_pos, vis_sizes = (
                    self.model._get_img_feats(backbone_out, img_ids)
                )

                # Expand cached text embeddings to batch size
                concat_text_B = self.concat_text.expand(-1, B, -1)
                concat_mask_B = self.concat_mask.expand(B, -1)

                # Encoder
                prompt_pos = torch.zeros_like(concat_text_B)
                encoder_out = self.model.transformer.encoder(
                    src=img_feats,
                    src_key_padding_mask=None,
                    src_pos=img_pos,
                    prompt=concat_text_B,
                    prompt_pos=prompt_pos,
                    prompt_key_padding_mask=concat_mask_B,
                    feat_sizes=vis_sizes,
                )

            # Detach encoder outputs to cut gradient flow
            enc_memory = encoder_out["memory"].detach()
            enc_pos = encoder_out["pos_embed"].detach()
            enc_pad = encoder_out.get("padding_mask")
            if enc_pad is not None:
                enc_pad = enc_pad.detach()
            valid_ratios = encoder_out["valid_ratios"].detach()

            # Also detach and expand text for decoder
            dec_text = concat_text_B.detach()
            dec_mask = concat_mask_B.detach()

            # --- Decoder + scoring (trainable, with grad) ---
            with autocast("cuda", dtype=torch.float16):
                cls_logits, pred_boxes = self.forward_module(
                    enc_memory, enc_pad, enc_pos,
                    dec_text, dec_mask, self.class_embeddings,
                    encoder_out["level_start_index"],
                    encoder_out["spatial_shapes"],
                    valid_ratios,
                )

                loss_dict = self._compute_loss(
                    cls_logits.float(), pred_boxes.float(),
                    gt_boxes_list, gt_labels_list,
                )

            # --- Backward ---
            self.optimizer.zero_grad()
            self.scaler.scale(loss_dict["loss"]).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(
                [p for p in self.forward_module.parameters() if p.requires_grad],
                self.max_grad_norm,
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Accumulate
            for k in accum_losses:
                accum_losses[k] += loss_dict[k]
            num_batches += 1

            # Log
            if self.rank == 0 and (batch_idx + 1) % 50 == 0:
                elapsed = time.perf_counter() - t0
                avg = {k: v / num_batches for k, v in accum_losses.items()}
                lr = self.optimizer.param_groups[0]["lr"]
                print(
                    f"  [{epoch}/{total_epochs}] "
                    f"step {batch_idx + 1}/{len(dataloader)} "
                    f"loss={avg['total']:.4f} "
                    f"(cls={avg['cls']:.4f} l1={avg['l1']:.4f} giou={avg['giou']:.4f}) "
                    f"lr={lr:.2e} "
                    f"matched={loss_dict['num_matched']} "
                    f"[{elapsed:.0f}s]"
                )

        avg_losses = {k: v / max(num_batches, 1) for k, v in accum_losses.items()}
        return global_step, avg_losses

    def save_checkpoint(self, path, epoch, class_names, global_step=0):
        """Save training checkpoint."""
        # Get the underlying module (unwrap DDP)
        fwd = self.forward_module
        if isinstance(fwd, DDP):
            fwd = fwd.module

        # Save only the trainable decoder weights (ca_text, catext_norm, bbox_embed)
        trainable_decoder_keys = [
            "ca_text", "catext_norm", "catext_dropout", "bbox_embed",
            "linear1", "linear2", "norm3",  # FFN (if trained)
        ]
        decoder_state = {
            k: v for k, v in fwd.decoder.state_dict().items()
            if any(t in k for t in trainable_decoder_keys)
        }

        checkpoint = {
            "epoch": epoch,
            "global_step": global_step,
            "multiclass_scoring": fwd.multiclass_scoring.state_dict(),
            "decoder_state_dict": decoder_state,
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "class_names": class_names,
            "num_classes": len(class_names),
        }
        torch.save(checkpoint, path)
        if self.rank == 0:
            print(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path):
        """Load a training checkpoint to resume."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)

        fwd = self.forward_module
        if isinstance(fwd, DDP):
            fwd = fwd.module

        # Load multiclass scoring
        fwd.multiclass_scoring.load_state_dict(ckpt["multiclass_scoring"])

        # Load decoder weights (partial — only trainable keys)
        decoder_state = fwd.decoder.state_dict()
        decoder_state.update(ckpt["decoder_state_dict"])
        fwd.decoder.load_state_dict(decoder_state)

        # Load optimizer and scaler
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scaler.load_state_dict(ckpt["scaler"])

        if self.rank == 0:
            print(f"Resumed from {path} (epoch {ckpt['epoch']})")

        return ckpt["epoch"], ckpt.get("global_step", 0)


# ---------------------------------------------------------------------------
# Worker (multi-GPU entry point)
# ---------------------------------------------------------------------------

def worker(rank, world_size, args):
    """DDP worker function."""
    # Setup DDP (skip if already initialized by torchrun)
    if not dist.is_initialized():
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(args.master_port)
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    device = f"cuda:{rank}"

    if rank == 0:
        print(f"\n{'=' * 60}")
        print(f"Multi-class single-pass training")
        print(f"GPUs: {world_size}, batch/GPU: {args.batch_size}")
        print(f"Effective batch size: {world_size * args.batch_size}")
        print(f"{'=' * 60}\n")

    # Load model
    if rank == 0:
        print("Loading SAM3 model...")
    model = build_sam3_image_model(
        device="cuda",
        checkpoint_path=args.checkpoint,
        eval_mode=False,
    )

    # Create multi-class scoring head (warm-start from DotProductScoring)
    multiclass_scoring = MultiClassScoring.from_dot_product_scoring(
        model.dot_prod_scoring
    )

    # Create trainer
    trainer = MultiClassTrainer(
        model=model,
        multiclass_scoring=multiclass_scoring,
        device=device,
        rank=rank,
        world_size=world_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        max_grad_norm=args.gradient_clip,
        cls_weight=args.cls_weight,
        l1_weight=args.l1_weight,
        giou_weight=args.giou_weight,
        train_decoder_ffn=args.train_decoder_ffn,
    )

    # Pre-compute text embeddings
    trainer.precompute_text(COCO_CLASSES)

    # Dataset + dataloader
    dataset = COCOMultiClassDataset(
        args.coco_dir, split=args.split, resolution=1008
    )
    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=True
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )

    if rank == 0:
        print(f"\nDataset: {len(dataset)} images, "
              f"{len(dataloader)} batches/epoch")
        print(f"Training for {args.epochs} epochs\n")

    # Resume or start fresh
    os.makedirs(args.output_dir, exist_ok=True)
    start_epoch = 1
    global_step = 0

    if args.resume:
        start_epoch, global_step = trainer.load_checkpoint(args.resume)
        start_epoch += 1  # resume from next epoch

    # Training loop
    for epoch in range(start_epoch, args.epochs + 1):
        sampler.set_epoch(epoch)

        if rank == 0:
            print(f"\n--- Epoch {epoch}/{args.epochs} ---")

        global_step, avg_losses = trainer.train_epoch(
            dataloader, epoch, args.epochs, global_step
        )

        if rank == 0:
            print(
                f"Epoch {epoch} complete: "
                f"loss={avg_losses['total']:.4f} "
                f"(cls={avg_losses['cls']:.4f} "
                f"l1={avg_losses['l1']:.4f} "
                f"giou={avg_losses['giou']:.4f})"
            )

        # Save checkpoint
        if rank == 0 and (epoch % args.save_every == 0 or epoch == args.epochs):
            ckpt_path = os.path.join(
                args.output_dir, f"multiclass_ep{epoch:03d}.pt"
            )
            trainer.save_checkpoint(ckpt_path, epoch, COCO_CLASSES, global_step)

    if world_size > 1:
        dist.destroy_process_group()


def worker_single(args):
    """Single-GPU worker (no DDP)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n{'=' * 60}")
    print(f"Multi-class single-pass training (single GPU)")
    print(f"Batch size: {args.batch_size}")
    print(f"{'=' * 60}\n")

    # Load model
    print("Loading SAM3 model...")
    model = build_sam3_image_model(
        device="cuda",
        checkpoint_path=args.checkpoint,
        eval_mode=False,
    )

    # Create multi-class scoring head
    multiclass_scoring = MultiClassScoring.from_dot_product_scoring(
        model.dot_prod_scoring
    )

    # Create trainer
    trainer = MultiClassTrainer(
        model=model,
        multiclass_scoring=multiclass_scoring,
        device=device,
        rank=0,
        world_size=1,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        max_grad_norm=args.gradient_clip,
        cls_weight=args.cls_weight,
        l1_weight=args.l1_weight,
        giou_weight=args.giou_weight,
        train_decoder_ffn=args.train_decoder_ffn,
    )

    # Pre-compute text embeddings
    trainer.precompute_text(COCO_CLASSES)

    # Dataset + dataloader
    dataset = COCOMultiClassDataset(
        args.coco_dir, split=args.split, resolution=1008
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )

    print(f"\nDataset: {len(dataset)} images, "
          f"{len(dataloader)} batches/epoch")
    print(f"Training for {args.epochs} epochs\n")

    # Resume or start fresh
    os.makedirs(args.output_dir, exist_ok=True)
    start_epoch = 1
    global_step = 0

    if args.resume:
        start_epoch, global_step = trainer.load_checkpoint(args.resume)
        start_epoch += 1

    # Training loop
    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\n--- Epoch {epoch}/{args.epochs} ---")

        global_step, avg_losses = trainer.train_epoch(
            dataloader, epoch, args.epochs, global_step
        )

        print(
            f"Epoch {epoch} complete: "
            f"loss={avg_losses['total']:.4f} "
            f"(cls={avg_losses['cls']:.4f} "
            f"l1={avg_losses['l1']:.4f} "
            f"giou={avg_losses['giou']:.4f})"
        )

        if epoch % args.save_every == 0 or epoch == args.epochs:
            ckpt_path = os.path.join(
                args.output_dir, f"multiclass_ep{epoch:03d}.pt"
            )
            trainer.save_checkpoint(ckpt_path, epoch, COCO_CLASSES, global_step)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train SAM3 for multi-class single-pass inference"
    )

    # Data
    parser.add_argument(
        "--coco-dir", type=str, required=True,
        help="COCO dataset root (contains train2017/, annotations/)",
    )
    parser.add_argument(
        "--split", type=str, default="train2017",
        help="Dataset split (default: train2017)",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="SAM3 checkpoint path (downloads from HF if not provided)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="./checkpoints/multiclass",
        help="Output directory for checkpoints",
    )

    # Training
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Per-GPU batch size")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--gradient-clip", type=float, default=0.1)

    # Loss weights
    parser.add_argument("--cls-weight", type=float, default=2.0)
    parser.add_argument("--l1-weight", type=float, default=5.0)
    parser.add_argument("--giou-weight", type=float, default=2.0)

    # Architecture
    parser.add_argument(
        "--train-decoder-ffn", action="store_true",
        help="Also fine-tune decoder FFN layers (more params, slower)",
    )

    # Infrastructure
    parser.add_argument("--num-gpus", type=int, default=0,
                        help="Number of GPUs (0 = single GPU, no DDP)")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--master-port", type=int, default=29500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume training from")

    args = parser.parse_args()

    # Set seed
    torch.manual_seed(args.seed)

    # Check for torchrun-launched distributed
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        # Launched via torchrun — worker() will detect dist is already
        # initialized via is_initialized() check
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = int(os.environ["WORLD_SIZE"])
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)
        worker(local_rank, world_size, args)

    elif args.num_gpus > 1:
        # Launch with mp.spawn
        mp.spawn(worker, args=(args.num_gpus, args), nprocs=args.num_gpus)

    else:
        # Single GPU
        worker_single(args)


if __name__ == "__main__":
    main()
