#!/usr/bin/env python3
"""
Fine-tune a structurally pruned SAM3 encoder+decoder via knowledge distillation.

The pruned model has fewer transformer layers, smaller FFN, and fewer queries,
so its weights were initialized from the unpruned model but the representations
are broken. This script recovers quality by distilling from the full (teacher)
model's encoder and decoder outputs.

Two modes:
  1. Shared ViT-H backbone (default): pruned encoder+decoder+scoring are
     trainable, backbone is shared with teacher (frozen).
  2. Student backbone (--student-backbone): also replaces the ViT-H backbone
     with a lightweight backbone (EfficientViT, RepViT, TinyViT) + trainable
     FPN adapter. This gives maximum compression for edge deployment.

Loss: weighted MSE on all module outputs:
  - Backbone FPN features (only with --student-backbone)
  - Encoder memory (fused image features)
  - Prompt-after-encoder (text features refined by encoder)
  - All decoder layer hidden states
  - Reference boxes (iteratively refined anchors)
  - Box predictions (bbox_embed offsets)
  - Scoring logits (dot-product confidence)

Data: COCO train2017 images (no labels needed) + COCO + LVIS text prompts.

Supports single-GPU and multi-GPU (auto-detected).

Usage (shared backbone):
    srun python scripts/finetune_pruned.py \
        --pruned-checkpoint pruned_sam3.pt \
        --coco-dir train2017 \
        --output-dir finetune_output \
        --epochs 10 --subset-size 20000 --batch-size 4

Usage (with lightweight backbone replacement):
    srun python scripts/finetune_pruned.py \
        --pruned-checkpoint pruned_sam3.pt \
        --coco-dir train2017 \
        --output-dir finetune_output \
        --student-backbone efficientvit_l1 \
        --epochs 15 --subset-size 20000 --batch-size 4

    The script auto-detects all visible GPUs and spawns one worker per GPU
    using torch.multiprocessing.spawn. No torchrun or --ntasks needed.
"""

import argparse
import os
import random
import sys
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data.distributed import DistributedSampler

from PIL import Image
from torchvision.transforms import v2


# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------

def is_dist_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_dist_initialized() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_dist_initialized() else 1


def is_main_process() -> bool:
    return get_rank() == 0


def dist_print(*args, **kwargs):
    """Print only on rank 0."""
    if is_main_process():
        print(*args, **kwargs)


def reduce_scalar(value: float, device: torch.device) -> float:
    """All-reduce a scalar value across ranks (mean)."""
    if not is_dist_initialized():
        return value
    t = torch.tensor(value, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t.item() / get_world_size()


def setup_for_spawn(rank, world_size):
    """Initialize distributed process group for mp.spawn launch."""
    os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR", "localhost")
    os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "29500")
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)


def cleanup_distributed():
    if is_dist_initialized():
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Text prompt classes: COCO 80 + LVIS (subset of ~300 diverse categories)
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

# LVIS categories (diverse subset covering rare/uncommon objects not in COCO)
LVIS_EXTRA_CLASSES = [
    # animals
    "alpaca", "antelope", "armadillo", "badger", "bat", "beaver", "bison",
    "buffalo", "bull", "camel", "canary", "cheetah", "chicken", "chimpanzee",
    "cobra", "crab", "crocodile", "crow", "deer", "dolphin", "donkey",
    "dragonfly", "duck", "eagle", "eel", "falcon", "flamingo", "fox",
    "frog", "gazelle", "goat", "goldfish", "goose", "gorilla", "hamster",
    "hawk", "hedgehog", "hippopotamus", "hornet", "hummingbird", "hyena",
    "iguana", "jaguar", "jellyfish", "kangaroo", "koala", "ladybug",
    "leopard", "lion", "lizard", "llama", "lobster", "lynx", "monkey",
    "moose", "moth", "octopus", "ostrich", "otter", "owl", "oyster",
    "panda", "parrot", "peacock", "pelican", "penguin", "pig", "pigeon",
    "porcupine", "rabbit", "raccoon", "rat", "raven", "rhinoceros",
    "rooster", "salamander", "salmon", "scorpion", "seahorse", "seal",
    "shark", "snail", "snake", "sparrow", "spider", "squid", "squirrel",
    "starfish", "stork", "swan", "tiger", "tortoise", "toucan", "trout",
    "turkey", "turtle", "vulture", "walrus", "whale", "wolf", "worm",
    # food & drink
    "avocado", "bagel", "baguette", "beer", "blueberry", "bread",
    "burrito", "butter", "cabbage", "candy", "cantaloupe", "celery",
    "cheese", "cherry", "chocolate", "coconut", "cookie", "corn",
    "cracker", "croissant", "cucumber", "cupcake", "egg", "fig",
    "garlic", "grape", "grapefruit", "hamburger", "honey", "ice cream",
    "jam", "ketchup", "lemon", "lettuce", "lime", "mango", "marshmallow",
    "melon", "milk", "muffin", "mushroom", "noodle", "olive", "onion",
    "pancake", "pasta", "peach", "peanut", "pear", "pepper", "pickle",
    "pie", "pineapple", "plum", "pomegranate", "popcorn", "potato",
    "pretzel", "pumpkin", "radish", "raspberry", "rice", "salad",
    "sausage", "spinach", "steak", "strawberry", "sushi", "taco",
    "tea", "tomato", "waffle", "watermelon", "yogurt", "zucchini",
    # household & objects
    "alarm clock", "anvil", "axe", "balloon", "bandage", "barrel",
    "basket", "bathrobe", "bathtub", "beacon", "belt", "binder",
    "blanket", "blender", "blinds", "bookshelf", "broom", "brush",
    "bucket", "bulletin board", "calculator", "calendar", "candle",
    "canteen", "cardboard", "carpet", "cart", "cassette", "chandelier",
    "clipboard", "comb", "compass", "cooler", "cork", "corkscrew",
    "crayon", "crib", "crown", "curtain", "cushion", "dartboard",
    "doorbell", "drawer", "drum", "dumbbell", "dustpan", "envelope",
    "eraser", "fan", "faucet", "fence", "fire extinguisher", "fishing rod",
    "flag", "flashlight", "flowerpot", "flute", "funnel", "globe",
    "guitar", "hammer", "hanger", "harmonica", "harp", "helmet",
    "horn", "hourglass", "iron", "jar", "kettle", "ladder",
    "lamp", "lantern", "lighter", "lock", "magazine", "mailbox",
    "map", "marker", "mat", "matchbox", "medal", "mirror",
    "mop", "napkin", "needle", "newspaper", "notebook", "paddle",
    "paintbrush", "palette", "pan", "paper towel", "pen", "pencil",
    "piano", "pillow", "pipe", "plate", "pliers", "poster",
    "radio", "razor", "ribbon", "rope", "rubber band", "rug",
    "ruler", "scale", "screwdriver", "shampoo", "shelf", "shield",
    "shoe", "shovel", "soap", "socket", "spatula", "sponge",
    "stapler", "stool", "stove", "straw", "suitcase", "sunglasses",
    "sword", "syringe", "tape", "teapot", "telephone", "tent",
    "thermometer", "tissue", "toilet paper", "toolbox", "tray",
    "trophy", "tweezers", "vacuum cleaner", "wallet", "washing machine",
    "watch", "wheelbarrow", "whistle", "wig", "window", "wrench",
    # vehicles & outdoor
    "ambulance", "anchor", "barge", "canoe", "crane", "excavator",
    "fire truck", "forklift", "golf cart", "helicopter", "jet ski",
    "kayak", "limousine", "minivan", "parachute", "pickup truck",
    "rickshaw", "rowboat", "sailboat", "scooter", "submarine",
    "taxi", "tractor", "tricycle", "van", "wheelchair",
    # structures & scenes
    "arch", "barn", "bridge", "cabin", "castle", "church",
    "dome", "fountain", "garage", "gazebo", "greenhouse", "lighthouse",
    "monument", "parking meter", "pier", "playground", "podium",
    "pyramid", "skyscraper", "staircase", "statue", "tower", "tunnel",
    "windmill",
]

# Deduplicated combined list
ALL_CLASSES = list(dict.fromkeys(COCO_CLASSES + LVIS_EXTRA_CLASSES))


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ImageFolderDataset(Dataset):
    """Load images from a directory (no labels needed)."""

    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

    def __init__(self, root: str, resolution: int = 1008):
        self.root = root
        self.resolution = resolution
        self.image_paths = []
        for entry in os.scandir(root):
            if entry.is_file() and os.path.splitext(entry.name)[1].lower() in self.IMAGE_EXTENSIONS:
                self.image_paths.append(entry.path)
        self.image_paths.sort()

        self.transform = v2.Compose([
            v2.ToDtype(torch.uint8, scale=True),
            v2.Resize(size=(resolution, resolution)),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        img = v2.functional.to_image(img)
        img = self.transform(img)
        return img


# ---------------------------------------------------------------------------
# DDP-compatible wrapper for trainable modules
# ---------------------------------------------------------------------------

class StudentForward(nn.Module):
    """Wraps all trainable student modules into one forward pass for DDP.

    Optionally includes a student backbone (lightweight + FPN adapter) for
    full backbone replacement. When student_backbone is provided, it runs the
    backbone inside the DDP wrapper so adapter gradients are synced.

    Returns all intermediate outputs needed for comprehensive distillation:
    - backbone_fpn (only when student_backbone is set)
    - encoder memory, prompt_after_enc
    - all decoder layer hidden states, reference boxes
    - box predictions (bbox_embed applied to hs)
    - scoring logits
    """

    def __init__(self, encoder, decoder, dot_prod_scoring, student_backbone=None):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.dot_prod_scoring = dot_prod_scoring
        self.student_backbone = student_backbone  # None = shared backbone mode

    def forward(self, img_feats, img_pos_embeds, vis_feat_sizes, prompt, prompt_mask,
                images=None):
        """Run [backbone ->] encoder -> decoder -> bbox_embed -> scoring.

        When self.student_backbone is set and images is provided, the backbone
        runs first to produce img_feats/pos_embeds (overriding the passed-in ones).
        """
        bs = prompt.shape[1]
        backbone_fpn = None

        if self.student_backbone is not None and images is not None:
            # Run student backbone (with grad for FPN adapter)
            backbone_out = self.student_backbone.forward_image(images)
            backbone_fpn = backbone_out["backbone_fpn"]
            # Extract features matching _get_img_feats format (last num_feature_levels)
            vis_feats = backbone_fpn[-1:]  # num_feature_levels=1
            vis_pos_enc = backbone_out["vision_pos_enc"][-1:]
            vis_feat_sizes = [x.shape[-2:] for x in vis_pos_enc]
            img_ids = torch.arange(images.shape[0], device=images.device, dtype=torch.long)
            img_feats = [x[img_ids].flatten(2).permute(2, 0, 1) for x in vis_feats]
            img_pos_embeds = [x[img_ids].flatten(2).permute(2, 0, 1) for x in vis_pos_enc]

        # --- Encoder ---
        prompt_pos = torch.zeros_like(prompt)
        memory_dict = self.encoder(
            src=[f.clone() for f in img_feats],
            src_key_padding_mask=None,
            src_pos=[p.clone() for p in img_pos_embeds],
            prompt=prompt,
            prompt_pos=prompt_pos,
            prompt_key_padding_mask=prompt_mask,
            feat_sizes=vis_feat_sizes,
        )

        prompt_after_enc = memory_dict.get("memory_text", prompt)

        # --- Decoder ---
        query_embed = self.decoder.query_embed.weight
        tgt = query_embed.unsqueeze(1).expand(-1, bs, -1)

        hs, reference_boxes, dec_presence_out, _ = self.decoder(
            tgt=tgt,
            memory=memory_dict["memory"],
            memory_key_padding_mask=memory_dict["padding_mask"],
            pos=memory_dict["pos_embed"],
            reference_boxes=None,
            level_start_index=memory_dict["level_start_index"],
            spatial_shapes=memory_dict["spatial_shapes"],
            valid_ratios=memory_dict["valid_ratios"],
            tgt_mask=None,
            memory_text=prompt,
            text_attention_mask=prompt_mask,
            apply_dac=False,
        )
        hs = hs.transpose(1, 2)  # (layers, bs, Q, d)
        reference_boxes = reference_boxes.transpose(1, 2)  # (layers, bs, Q, 4)

        # --- Box predictions (bbox_embed applied to hidden states) ---
        box_head = self.decoder.bbox_embed
        box_preds = box_head(hs)  # (layers, bs, Q, 4) - delta offsets

        # --- Scoring ---
        scores = self.dot_prod_scoring(hs, prompt, prompt_mask)

        return {
            "memory": memory_dict["memory"],
            "prompt_after_enc": prompt_after_enc,
            "hs": hs,
            "reference_boxes": reference_boxes,
            "box_preds": box_preds,
            "scores": scores,
            "backbone_fpn": backbone_fpn,  # None when using shared backbone
            "dec_presence_out": dec_presence_out,
        }


# ---------------------------------------------------------------------------
# Distillation trainer
# ---------------------------------------------------------------------------

class PrunedModelFinetuner:
    """Knowledge distillation from unpruned (teacher) to pruned (student) SAM3.

    Aligns all module outputs:
      - Backbone FPN features (when using student backbone)
      - Encoder memory (fused image features)
      - Prompt after encoder (text features refined by encoder fusion)
      - All decoder layer hidden states (not just last layer)
      - Reference boxes (iteratively refined anchors at each decoder layer)
      - Box predictions (bbox_embed offsets at each decoder layer)
      - Scoring logits (dot-product confidence scores)
    """

    def __init__(
        self,
        teacher_model,
        student_model,
        dataloader,
        sampler=None,
        device="cuda",
        lr=5e-4,
        weight_decay=1e-4,
        warmup_steps=200,
        encoder_loss_weight=1.0,
        prompt_loss_weight=0.5,
        decoder_loss_weight=1.0,
        refbox_loss_weight=1.0,
        boxpred_loss_weight=0.5,
        scoring_loss_weight=0.3,
        presence_loss_weight=1.0,
        backbone_loss_weight=2.0,
        max_grad_norm=1.0,
        num_prompts_per_batch=4,
        student_backbone=None,
    ):
        self.teacher = teacher_model
        self.student = student_model
        self.dataloader = dataloader
        self.sampler = sampler
        self.device = device
        self.encoder_loss_weight = encoder_loss_weight
        self.prompt_loss_weight = prompt_loss_weight
        self.decoder_loss_weight = decoder_loss_weight
        self.refbox_loss_weight = refbox_loss_weight
        self.boxpred_loss_weight = boxpred_loss_weight
        self.scoring_loss_weight = scoring_loss_weight
        self.presence_loss_weight = presence_loss_weight
        self.backbone_loss_weight = backbone_loss_weight
        self.max_grad_norm = max_grad_norm
        self.warmup_steps = warmup_steps
        self.num_prompts_per_batch = num_prompts_per_batch
        self.use_student_backbone = student_backbone is not None

        # Freeze everything except student encoder, decoder, scoring head
        self._freeze_shared_components()

        # Create DDP-compatible wrapper for trainable modules
        # When student_backbone is provided, it's included in the wrapper
        # so FPN adapter gradients are synced across GPUs.
        self.student_forward = StudentForward(
            encoder=self.student.transformer.encoder,
            decoder=self.student.transformer.decoder,
            dot_prod_scoring=self.student.dot_prod_scoring,
            student_backbone=student_backbone,
        ).to(device)

        # Wrap in DDP if distributed
        if is_dist_initialized():
            local_rank = torch.cuda.current_device()
            self.student_forward = DDP(
                self.student_forward,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=True,
            )
            dist_print("  Wrapped trainable modules in DDP")

        # Collect trainable parameters
        trainable_params = list(self._trainable_module().parameters())
        num_trainable = sum(p.numel() for p in trainable_params)
        dist_print(f"Trainable parameters: {num_trainable:,} ({num_trainable/1e6:.1f}M)")
        dist_print(f"Trainable tensors: {len(trainable_params)}")

        self.optimizer = torch.optim.AdamW(
            trainable_params, lr=lr, weight_decay=weight_decay,
        )
        self.scaler = GradScaler()

        # Pre-compute text embeddings for ALL classes (COCO + LVIS)
        self._precompute_text_embeddings()

    def _trainable_module(self):
        """Return the unwrapped trainable module (handles DDP)."""
        if isinstance(self.student_forward, DDP):
            return self.student_forward.module
        return self.student_forward

    def _freeze_shared_components(self):
        """Freeze backbone, text encoder, geometry encoder. Unfreeze encoder+decoder+scoring.

        When using student backbone, FPN adapter params are unfrozen via
        StudentForward (which owns the student backbone). The student_backbone's
        timm backbone is already frozen internally.
        """
        for param in self.student.parameters():
            param.requires_grad = False

        for param in self.student.transformer.encoder.parameters():
            param.requires_grad = True
        for param in self.student.transformer.decoder.parameters():
            param.requires_grad = True
        if self.student.dot_prod_scoring is not None:
            for param in self.student.dot_prod_scoring.parameters():
                param.requires_grad = True

        for param in self.teacher.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def _precompute_text_embeddings(self):
        """Pre-compute text embeddings for all classes (COCO + LVIS)."""
        dist_print(f"Pre-computing text embeddings for {len(ALL_CLASSES)} classes "
                   f"({len(COCO_CLASSES)} COCO + {len(LVIS_EXTRA_CLASSES)} LVIS)...")
        # Process in chunks to avoid OOM on the text encoder
        chunk_size = 100
        all_feats = []
        all_masks = []
        for i in range(0, len(ALL_CLASSES), chunk_size):
            chunk = ALL_CLASSES[i:i + chunk_size]
            text_outputs = self.teacher.backbone.forward_text(
                chunk, device=self.device,
            )
            all_feats.append(text_outputs["language_features"])
            all_masks.append(text_outputs["language_mask"])
        # Concatenate: feats (seq, N, d), masks (N, seq)
        self.all_text_feats = torch.cat(all_feats, dim=1)  # (seq, total_classes, d)
        self.all_text_masks = torch.cat(all_masks, dim=0)  # (total_classes, seq)
        self.num_classes = self.all_text_feats.shape[1]
        dist_print(f"  Text features shape: {self.all_text_feats.shape} "
                   f"({self.num_classes} classes)")

    def _sample_text_prompts(self, n):
        """Sample n random class text prompts. Returns (prompt, mask) lists."""
        indices = random.sample(range(self.num_classes), min(n, self.num_classes))
        prompts = []
        masks = []
        for idx in indices:
            prompts.append(self.all_text_feats[:, idx:idx+1, :])  # (seq, 1, d)
            masks.append(self.all_text_masks[idx:idx+1, :])  # (1, seq)
        return prompts, masks

    def _run_teacher(self, img_feats, img_pos_embeds, vis_feat_sizes, prompt, prompt_mask):
        """Run frozen teacher encoder -> decoder -> bbox_embed -> scoring (no grad).

        Returns dict with all intermediate outputs matching StudentForward output format.
        """
        bs = prompt.shape[1]
        prompt_pos = torch.zeros_like(prompt)
        memory = self.teacher.transformer.encoder(
            src=[f.clone() for f in img_feats],
            src_key_padding_mask=None,
            src_pos=[p.clone() for p in img_pos_embeds],
            prompt=prompt,
            prompt_pos=prompt_pos,
            prompt_key_padding_mask=prompt_mask,
            feat_sizes=vis_feat_sizes,
        )

        prompt_after_enc = memory.get("memory_text", prompt)

        query_embed = self.teacher.transformer.decoder.query_embed.weight
        tgt = query_embed.unsqueeze(1).expand(-1, bs, -1)
        hs, reference_boxes, dec_presence_out, _ = self.teacher.transformer.decoder(
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
        hs = hs.transpose(1, 2)
        reference_boxes = reference_boxes.transpose(1, 2)

        # Box predictions from bbox_embed
        box_head = self.teacher.transformer.decoder.bbox_embed
        box_preds = box_head(hs)

        # Scoring
        scores = self.teacher.dot_prod_scoring(hs, prompt, prompt_mask)

        return {
            "memory": memory["memory"],
            "prompt_after_enc": prompt_after_enc,
            "hs": hs,
            "reference_boxes": reference_boxes,
            "box_preds": box_preds,
            "scores": scores,
            "dec_presence_out": dec_presence_out,
        }

    def _compute_loss(self, teacher_out, student_out, teacher_backbone_fpn=None):
        """Compute comprehensive distillation loss across all aligned outputs."""
        losses = {}

        # 0. Backbone feature loss (only when using student backbone)
        if (student_out.get("backbone_fpn") is not None
                and teacher_backbone_fpn is not None):
            bb_loss = torch.tensor(0.0, device=self.device)
            s_fpn = student_out["backbone_fpn"]
            t_fpn = teacher_backbone_fpn
            # Align last N levels (min of student and teacher)
            n = min(len(s_fpn), len(t_fpn))
            for i in range(n):
                s_feat = s_fpn[-(i + 1)]
                t_feat = t_fpn[-(i + 1)]
                # Interpolate if spatial sizes differ
                if s_feat.shape[-2:] != t_feat.shape[-2:]:
                    s_feat = F.interpolate(
                        s_feat, size=t_feat.shape[-2:],
                        mode="bilinear", align_corners=False,
                    )
                bb_loss = bb_loss + F.mse_loss(s_feat, t_feat)
            losses["backbone"] = bb_loss / max(n, 1)
        else:
            losses["backbone"] = torch.tensor(0.0, device=self.device)

        # 1. Encoder memory loss: (seq, B, d) — the fused image features
        enc_loss = F.mse_loss(student_out["memory"], teacher_out["memory"])
        losses["encoder"] = enc_loss

        # 2. Prompt-after-encoder loss: (seq, B, d) — text features after encoder fusion
        prompt_loss = F.mse_loss(
            student_out["prompt_after_enc"], teacher_out["prompt_after_enc"]
        )
        losses["prompt"] = prompt_loss

        # 3. Decoder hidden states: align ALL layers, not just last
        #    teacher hs: (T_layers, bs, T_queries, d)
        #    student hs: (S_layers, bs, S_queries, d)
        t_hs = teacher_out["hs"]
        s_hs = student_out["hs"]
        t_layers, s_layers = t_hs.shape[0], s_hs.shape[0]
        min_queries = min(t_hs.shape[2], s_hs.shape[2])

        # Map student layers to teacher layers (evenly spaced)
        # e.g. if teacher has 6 layers, student has 3: map student [0,1,2] -> teacher [1,3,5]
        if s_layers < t_layers:
            layer_map = [
                int(round((i + 1) * t_layers / s_layers)) - 1
                for i in range(s_layers)
            ]
        else:
            layer_map = list(range(s_layers))

        dec_loss = torch.tensor(0.0, device=self.device)
        for s_idx, t_idx in enumerate(layer_map):
            if t_idx < t_layers:
                dec_loss = dec_loss + F.mse_loss(
                    s_hs[s_idx, :, :min_queries],
                    t_hs[t_idx, :, :min_queries],
                )
        dec_loss = dec_loss / max(len(layer_map), 1)
        losses["decoder_hs"] = dec_loss

        # 4. Reference boxes: align at each mapped decoder layer
        #    (layers, bs, Q, 4) in [0,1] cxcywh after sigmoid
        t_rb = teacher_out["reference_boxes"]
        s_rb = student_out["reference_boxes"]
        refbox_loss = torch.tensor(0.0, device=self.device)
        for s_idx, t_idx in enumerate(layer_map):
            if t_idx < t_rb.shape[0] and s_idx < s_rb.shape[0]:
                refbox_loss = refbox_loss + F.mse_loss(
                    s_rb[s_idx, :, :min_queries],
                    t_rb[t_idx, :, :min_queries],
                )
        refbox_loss = refbox_loss / max(len(layer_map), 1)
        losses["refbox"] = refbox_loss

        # 5. Box predictions: align bbox_embed outputs at each mapped layer
        #    (layers, bs, Q, 4) — the delta offsets
        t_bp = teacher_out["box_preds"]
        s_bp = student_out["box_preds"]
        boxpred_loss = torch.tensor(0.0, device=self.device)
        for s_idx, t_idx in enumerate(layer_map):
            if t_idx < t_bp.shape[0] and s_idx < s_bp.shape[0]:
                boxpred_loss = boxpred_loss + F.mse_loss(
                    s_bp[s_idx, :, :min_queries],
                    t_bp[t_idx, :, :min_queries],
                )
        boxpred_loss = boxpred_loss / max(len(layer_map), 1)
        losses["boxpred"] = boxpred_loss

        # 6. Scoring logits: align at each mapped decoder layer
        #    (layers, bs, Q, 1)
        t_sc = teacher_out["scores"]
        s_sc = student_out["scores"]
        score_loss = torch.tensor(0.0, device=self.device)
        for s_idx, t_idx in enumerate(layer_map):
            if t_idx < t_sc.shape[0] and s_idx < s_sc.shape[0]:
                min_q_sc = min(t_sc.shape[2], s_sc.shape[2])
                score_loss = score_loss + F.mse_loss(
                    s_sc[s_idx, :, :min_q_sc],
                    t_sc[t_idx, :, :min_q_sc],
                )
        score_loss = score_loss / max(len(layer_map), 1)
        losses["scoring"] = score_loss

        # 7. Presence logits: decoder presence prediction at each mapped layer
        #    Critical: Sam3MultiClassPredictor multiplies scores by presence,
        #    so if this head is untrained, all scores become ~0.
        t_pres = teacher_out.get("dec_presence_out")
        s_pres = student_out.get("dec_presence_out")
        if t_pres is not None and s_pres is not None:
            presence_loss = torch.tensor(0.0, device=self.device)
            for s_idx, t_idx in enumerate(layer_map):
                if t_idx < t_pres.shape[0] and s_idx < s_pres.shape[0]:
                    presence_loss = presence_loss + F.mse_loss(
                        s_pres[s_idx], t_pres[t_idx],
                    )
            presence_loss = presence_loss / max(len(layer_map), 1)
            losses["presence"] = presence_loss
        else:
            losses["presence"] = torch.tensor(0.0, device=self.device)

        total = (
            self.backbone_loss_weight * losses["backbone"]
            + self.encoder_loss_weight * losses["encoder"]
            + self.prompt_loss_weight * losses["prompt"]
            + self.decoder_loss_weight * losses["decoder_hs"]
            + self.refbox_loss_weight * losses["refbox"]
            + self.boxpred_loss_weight * losses["boxpred"]
            + self.scoring_loss_weight * losses["scoring"]
            + self.presence_loss_weight * losses["presence"]
        )
        losses["total"] = total
        return losses

    def _get_lr_scale(self, step):
        """Linear warmup schedule."""
        if step < self.warmup_steps:
            return (step + 1) / self.warmup_steps
        return 1.0

    LOSS_KEYS = ["total", "backbone", "encoder", "prompt", "decoder_hs", "refbox", "boxpred", "scoring", "presence"]

    def train_epoch(self, epoch, total_epochs):
        """Run one epoch of distillation training."""
        self.student_forward.train()

        # Set epoch on distributed sampler for proper shuffling
        if self.sampler is not None:
            self.sampler.set_epoch(epoch)

        epoch_losses = {k: 0.0 for k in self.LOSS_KEYS}
        num_batches = 0
        start_time = time.time()

        for batch_idx, images in enumerate(self.dataloader):
            global_step = epoch * len(self.dataloader) + batch_idx
            lr_scale = self._get_lr_scale(global_step)
            for pg in self.optimizer.param_groups:
                pg["lr"] = pg.get("initial_lr", pg["lr"]) * lr_scale

            if global_step == 0:
                for pg in self.optimizer.param_groups:
                    pg["initial_lr"] = pg["lr"]

            images = images.to(self.device, non_blocking=True)

            # --- Teacher backbone (always run, no grad) ---
            with torch.no_grad():
                teacher_backbone_out = self.teacher.backbone.forward_image(images)

            img_ids = torch.arange(images.shape[0], device=self.device, dtype=torch.long)
            _, teacher_img_feats, teacher_img_pos_embeds, teacher_vis_feat_sizes = (
                self.teacher._get_img_feats(teacher_backbone_out, img_ids)
            )

            # Teacher backbone FPN for feature-level distillation
            teacher_backbone_fpn = teacher_backbone_out.get("backbone_fpn")

            # For shared-backbone mode, student uses the same features
            if not self.use_student_backbone:
                student_img_feats = teacher_img_feats
                student_img_pos_embeds = teacher_img_pos_embeds
                student_vis_feat_sizes = teacher_vis_feat_sizes
            else:
                # Placeholder — student backbone runs inside StudentForward
                student_img_feats = None
                student_img_pos_embeds = None
                student_vis_feat_sizes = None

            # Sample multiple text prompts for diversity
            prompts, masks = self._sample_text_prompts(self.num_prompts_per_batch)
            bs = images.shape[0]

            # Accumulate losses over multiple prompts per batch
            batch_losses = {k: torch.tensor(0.0, device=self.device) for k in self.LOSS_KEYS}

            for prompt, prompt_mask in zip(prompts, masks):
                prompt = prompt.expand(-1, bs, -1).contiguous()
                prompt_mask = prompt_mask.expand(bs, -1).contiguous()

                # --- Teacher forward (no grad, uses teacher features) ---
                with torch.no_grad():
                    teacher_out = self._run_teacher(
                        teacher_img_feats, teacher_img_pos_embeds,
                        teacher_vis_feat_sizes, prompt, prompt_mask,
                    )

                # --- Student forward (with grad, through DDP wrapper) ---
                with autocast("cuda", dtype=torch.float16):
                    student_out = self.student_forward(
                        student_img_feats, student_img_pos_embeds,
                        student_vis_feat_sizes, prompt, prompt_mask,
                        images=images if self.use_student_backbone else None,
                    )

                    losses = self._compute_loss(
                        teacher_out, student_out,
                        teacher_backbone_fpn=teacher_backbone_fpn,
                    )

                for k in self.LOSS_KEYS:
                    batch_losses[k] = batch_losses[k] + losses[k]

            # Average over prompts
            for k in self.LOSS_KEYS:
                batch_losses[k] = batch_losses[k] / len(prompts)

            # --- Backward ---
            self.optimizer.zero_grad()
            self.scaler.scale(batch_losses["total"]).backward()

            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self._trainable_module().parameters(),
                self.max_grad_norm,
            )

            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Accumulate losses (reduce across ranks for logging)
            for k in self.LOSS_KEYS:
                epoch_losses[k] += reduce_scalar(batch_losses[k].item(), self.device)
            num_batches += 1

            if (batch_idx + 1) % 50 == 0 or batch_idx == 0:
                elapsed = time.time() - start_time
                avg_loss = epoch_losses["total"] / num_batches
                bb_str = ""
                if self.use_student_backbone:
                    bb_str = f"bb: {reduce_scalar(batch_losses['backbone'].item(), self.device):.4f} | "
                dist_print(
                    f"  [{epoch+1}/{total_epochs}] batch {batch_idx+1}/{len(self.dataloader)} | "
                    f"loss: {reduce_scalar(batch_losses['total'].item(), self.device):.4f} "
                    f"(avg: {avg_loss:.4f}) | "
                    f"{bb_str}"
                    f"enc: {reduce_scalar(batch_losses['encoder'].item(), self.device):.4f} | "
                    f"prompt: {reduce_scalar(batch_losses['prompt'].item(), self.device):.4f} | "
                    f"dec: {reduce_scalar(batch_losses['decoder_hs'].item(), self.device):.4f} | "
                    f"rbox: {reduce_scalar(batch_losses['refbox'].item(), self.device):.4f} | "
                    f"bpred: {reduce_scalar(batch_losses['boxpred'].item(), self.device):.4f} | "
                    f"score: {reduce_scalar(batch_losses['scoring'].item(), self.device):.4f} | "
                    f"pres: {reduce_scalar(batch_losses['presence'].item(), self.device):.4f} | "
                    f"lr: {self.optimizer.param_groups[0]['lr']:.2e} | "
                    f"{elapsed:.0f}s"
                )

        for k in epoch_losses:
            epoch_losses[k] /= max(num_batches, 1)
        elapsed = time.time() - start_time
        bb_str = ""
        if self.use_student_backbone:
            bb_str = f"bb: {epoch_losses['backbone']:.4f} | "
        dist_print(
            f"Epoch {epoch+1}/{total_epochs} done in {elapsed:.0f}s | "
            f"avg_loss: {epoch_losses['total']:.4f} | "
            f"{bb_str}"
            f"enc: {epoch_losses['encoder']:.4f} | "
            f"prompt: {epoch_losses['prompt']:.4f} | "
            f"dec: {epoch_losses['decoder_hs']:.4f} | "
            f"rbox: {epoch_losses['refbox']:.4f} | "
            f"bpred: {epoch_losses['boxpred']:.4f} | "
            f"score: {epoch_losses['scoring']:.4f} | "
            f"pres: {epoch_losses['presence']:.4f}"
        )
        return epoch_losses

    def save_checkpoint(self, path, pruning_config, epoch, losses,
                        backbone_config=None):
        """Save student checkpoint (rank 0 only)."""
        if not is_main_process():
            return
        state_dict = self.student.state_dict()
        prefixed = {"detector." + k: v for k, v in state_dict.items()}
        save_dict = {
            "model": prefixed,
            "pruning_config": pruning_config,
            "epoch": epoch,
            "losses": losses,
        }
        # Save student backbone adapter weights separately
        if self.use_student_backbone:
            module = self._trainable_module()
            bb_state = module.student_backbone.state_dict()
            save_dict["student_backbone"] = bb_state
            save_dict["backbone_config"] = backbone_config
        torch.save(save_dict, path)
        dist_print(f"Saved checkpoint to {path}")


# ---------------------------------------------------------------------------
# Worker function (one per GPU)
# ---------------------------------------------------------------------------

def worker(rank, world_size, args):
    """Per-GPU training worker, launched by mp.spawn."""
    # --- Distributed init ---
    setup_for_spawn(rank, world_size)
    device = f"cuda:{rank}"

    random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)

    if is_main_process():
        os.makedirs(args.output_dir, exist_ok=True)
    dist.barrier()

    use_student_bb = args.student_backbone is not None

    dist_print("=" * 60)
    dist_print("SAM3 Pruned Model Fine-tuning via Distillation")
    dist_print(f"  World size: {world_size} | Device: {device}")
    dist_print(f"  Text prompts: {len(ALL_CLASSES)} classes "
               f"({len(COCO_CLASSES)} COCO + {len(LVIS_EXTRA_CLASSES)} LVIS)")
    dist_print(f"  Prompts per batch: {args.num_prompts}")
    if use_student_bb:
        dist_print(f"  Student backbone: {args.student_backbone}")
    else:
        dist_print(f"  Backbone: shared ViT-H (teacher)")
    dist_print("=" * 60)

    # --- Load pruning config ---
    from sam3.model_builder import load_pruned_config
    pruning_config = load_pruned_config(args.pruned_checkpoint)
    if pruning_config is None:
        dist_print("ERROR: No pruning_config found in checkpoint.")
        dist_print("This script requires a pruned checkpoint from analyze_pruning.py --apply")
        cleanup_distributed()
        sys.exit(1)
    dist_print(f"Pruning config: {pruning_config}")

    # --- Build teacher model (unpruned) ---
    dist_print("\nBuilding teacher model (unpruned)...")
    from sam3.model_builder import build_sam3_image_model
    # Use device="cuda" (not f"cuda:{rank}") because _setup_device_and_mode
    # checks `device == "cuda"` exactly. torch.cuda.set_device(rank) already
    # ensures .cuda() routes to the correct GPU.
    teacher = build_sam3_image_model(
        checkpoint_path=args.teacher_checkpoint,
        device="cuda",
        eval_mode=True,
        enable_segmentation=False,
        load_from_HF=(args.teacher_checkpoint is None),
    )
    teacher.eval()
    dist_print(f"  Teacher params: {sum(p.numel() for p in teacher.parameters()):,}")

    # --- Build student model (pruned) ---
    dist_print("\nBuilding student model (pruned)...")
    from sam3.model_builder import build_pruned_sam3_image_model
    student = build_pruned_sam3_image_model(
        checkpoint_path=args.pruned_checkpoint,
        pruning_config=pruning_config,
        device="cuda",
        eval_mode=True,
        enable_segmentation=False,
    )
    dist_print(f"  Student params: {sum(p.numel() for p in student.parameters()):,}")

    enc_params = sum(p.numel() for p in student.transformer.encoder.parameters())
    dec_params = sum(p.numel() for p in student.transformer.decoder.parameters())
    scoring_params = sum(p.numel() for p in student.dot_prod_scoring.parameters())
    dist_print(f"  Encoder: {enc_params:,} | Decoder: {dec_params:,} | Scoring: {scoring_params:,}")

    # --- Backbone setup ---
    student_backbone = None
    if use_student_bb:
        dist_print(f"\nBuilding student backbone: {args.student_backbone}")
        from sam3.distillation.student_backbone import build_student_backbone
        student_backbone = build_student_backbone(
            config_name=args.student_backbone,
            pretrained=True,
            freeze_backbone=True,  # timm backbone frozen, only FPN adapter trains
        )
        student_backbone = student_backbone.to(device)
        adapter_params = student_backbone.trainable_params
        dist_print(f"  Student backbone adapter params: {adapter_params:,} ({adapter_params/1e6:.1f}M)")
        # Still share the text encoder from teacher for text embedding
        # (student backbone only replaces vision, not text)
    else:
        dist_print("\nSharing backbone weights between teacher and student...")
        student.backbone = teacher.backbone

    # --- Dataset ---
    dist_print(f"\nLoading images from {args.coco_dir}...")
    dataset = ImageFolderDataset(args.coco_dir)
    dist_print(f"  Found {len(dataset)} images")

    if args.subset_size > 0 and args.subset_size < len(dataset):
        rng = random.Random(args.seed)  # fixed seed so all ranks get same subset
        indices = rng.sample(range(len(dataset)), args.subset_size)
        dataset = Subset(dataset, indices)
        dist_print(f"  Using subset of {len(dataset)} images")

    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=True,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,  # sampler handles shuffling
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    dist_print(f"  Batches per epoch per GPU: {len(dataloader)}")
    dist_print(f"  Effective batch size: {args.batch_size * world_size}")

    # --- Trainer ---
    trainer = PrunedModelFinetuner(
        teacher_model=teacher,
        student_model=student,
        dataloader=dataloader,
        sampler=sampler,
        device=device,
        lr=args.lr,
        encoder_loss_weight=args.encoder_weight,
        prompt_loss_weight=args.prompt_weight,
        decoder_loss_weight=args.decoder_weight,
        refbox_loss_weight=args.refbox_weight,
        boxpred_loss_weight=args.boxpred_weight,
        scoring_loss_weight=args.scoring_weight,
        presence_loss_weight=args.presence_weight,
        backbone_loss_weight=args.backbone_weight,
        num_prompts_per_batch=args.num_prompts,
        student_backbone=student_backbone,
    )

    # --- Training loop ---
    dist_print(f"\nStarting training for {args.epochs} epochs...")
    best_loss = float("inf")
    best_path = os.path.join(args.output_dir, "finetuned_best.pt")

    for epoch in range(args.epochs):
        losses = trainer.train_epoch(epoch, args.epochs)

        if (epoch + 1) % args.save_every == 0:
            ckpt_path = os.path.join(args.output_dir, f"finetuned_epoch{epoch+1}.pt")
            trainer.save_checkpoint(ckpt_path, pruning_config, epoch + 1, losses,
                                    backbone_config=args.student_backbone)

        if losses["total"] < best_loss:
            best_loss = losses["total"]
            trainer.save_checkpoint(best_path, pruning_config, epoch + 1, losses,
                                    backbone_config=args.student_backbone)
            dist_print(f"  New best loss: {best_loss:.4f}")

    final_path = os.path.join(args.output_dir, "finetuned_final.pt")
    trainer.save_checkpoint(final_path, pruning_config, args.epochs, losses,
                            backbone_config=args.student_backbone)

    dist_print(f"\nDone! Best loss: {best_loss:.4f}")
    dist_print(f"Use the finetuned checkpoint with demo_multiclass.py:")
    dist_print(f"  python demo_multiclass.py --image x.jpg --classes person car --checkpoint {best_path}")

    cleanup_distributed()


# ---------------------------------------------------------------------------
# Single-GPU worker (no distributed)
# ---------------------------------------------------------------------------

def worker_single(args):
    """Single-GPU training (no distributed)."""
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    use_student_bb = args.student_backbone is not None
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("SAM3 Pruned Model Fine-tuning via Distillation")
    print(f"  Single GPU | Device: {device}")
    print(f"  Text prompts: {len(ALL_CLASSES)} classes "
          f"({len(COCO_CLASSES)} COCO + {len(LVIS_EXTRA_CLASSES)} LVIS)")
    print(f"  Prompts per batch: {args.num_prompts}")
    if use_student_bb:
        print(f"  Student backbone: {args.student_backbone}")
    else:
        print(f"  Backbone: shared ViT-H (teacher)")
    print("=" * 60)

    from sam3.model_builder import load_pruned_config
    pruning_config = load_pruned_config(args.pruned_checkpoint)
    if pruning_config is None:
        print("ERROR: No pruning_config found in checkpoint.")
        sys.exit(1)
    print(f"Pruning config: {pruning_config}")

    from sam3.model_builder import build_sam3_image_model
    print("\nBuilding teacher model (unpruned)...")
    build_device = "cuda" if torch.cuda.is_available() else "cpu"
    teacher = build_sam3_image_model(
        checkpoint_path=args.teacher_checkpoint,
        device=build_device,
        eval_mode=True,
        enable_segmentation=False,
        load_from_HF=(args.teacher_checkpoint is None),
    )
    teacher.eval()
    print(f"  Teacher params: {sum(p.numel() for p in teacher.parameters()):,}")

    from sam3.model_builder import build_pruned_sam3_image_model
    print("\nBuilding student model (pruned)...")
    student = build_pruned_sam3_image_model(
        checkpoint_path=args.pruned_checkpoint,
        pruning_config=pruning_config,
        device=build_device,
        eval_mode=True,
        enable_segmentation=False,
    )
    print(f"  Student params: {sum(p.numel() for p in student.parameters()):,}")

    student_backbone = None
    if use_student_bb:
        print(f"\nBuilding student backbone: {args.student_backbone}")
        from sam3.distillation.student_backbone import build_student_backbone
        student_backbone = build_student_backbone(
            config_name=args.student_backbone,
            pretrained=True,
            freeze_backbone=True,
        )
        student_backbone = student_backbone.to(device)
        adapter_params = student_backbone.trainable_params
        print(f"  Student backbone adapter params: {adapter_params:,} ({adapter_params/1e6:.1f}M)")
    else:
        student.backbone = teacher.backbone

    dataset = ImageFolderDataset(args.coco_dir)
    print(f"\nLoaded {len(dataset)} images from {args.coco_dir}")
    if args.subset_size > 0 and args.subset_size < len(dataset):
        indices = random.sample(range(len(dataset)), args.subset_size)
        dataset = Subset(dataset, indices)
        print(f"  Using subset of {len(dataset)} images")

    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )

    trainer = PrunedModelFinetuner(
        teacher_model=teacher, student_model=student,
        dataloader=dataloader, device=device, lr=args.lr,
        encoder_loss_weight=args.encoder_weight,
        prompt_loss_weight=args.prompt_weight,
        decoder_loss_weight=args.decoder_weight,
        refbox_loss_weight=args.refbox_weight,
        boxpred_loss_weight=args.boxpred_weight,
        scoring_loss_weight=args.scoring_weight,
        presence_loss_weight=args.presence_weight,
        backbone_loss_weight=args.backbone_weight,
        num_prompts_per_batch=args.num_prompts,
        student_backbone=student_backbone,
    )

    best_loss = float("inf")
    best_path = os.path.join(args.output_dir, "finetuned_best.pt")
    for epoch in range(args.epochs):
        losses = trainer.train_epoch(epoch, args.epochs)
        if (epoch + 1) % args.save_every == 0:
            trainer.save_checkpoint(
                os.path.join(args.output_dir, f"finetuned_epoch{epoch+1}.pt"),
                pruning_config, epoch + 1, losses,
                backbone_config=args.student_backbone,
            )
        if losses["total"] < best_loss:
            best_loss = losses["total"]
            trainer.save_checkpoint(best_path, pruning_config, epoch + 1, losses,
                                    backbone_config=args.student_backbone)
            print(f"  New best loss: {best_loss:.4f}")

    trainer.save_checkpoint(
        os.path.join(args.output_dir, "finetuned_final.pt"),
        pruning_config, args.epochs, losses,
        backbone_config=args.student_backbone,
    )
    print(f"\nDone! Best loss: {best_loss:.4f}")
    print(f"  python demo_multiclass.py --image x.jpg --classes person car --checkpoint {best_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Fine-tune pruned SAM3 via distillation")
    parser.add_argument(
        "--pruned-checkpoint", required=True,
        help="Path to pruned SAM3 checkpoint (from analyze_pruning.py --apply)",
    )
    parser.add_argument(
        "--teacher-checkpoint", default=None,
        help="Path to unpruned SAM3 checkpoint (defaults to HF download)",
    )
    parser.add_argument("--coco-dir", required=True, help="Path to COCO train2017 images")
    parser.add_argument("--output-dir", default="finetune_output", help="Output directory")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size per GPU")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--subset-size", type=int, default=5000,
                        help="Number of COCO images to use (0 = all)")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers per GPU")
    parser.add_argument("--save-every", type=int, default=2, help="Save checkpoint every N epochs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--gpus", type=int, default=None,
                        help="Number of GPUs to use (default: all visible)")
    parser.add_argument("--num-prompts", type=int, default=4,
                        help="Number of random text prompts per batch")

    # Student backbone (optional — replaces ViT-H with lightweight backbone)
    parser.add_argument("--student-backbone", type=str, default=None,
                        choices=["efficientvit_l1", "efficientvit_l2",
                                 "repvit_m2_3", "tiny_vit_21m"],
                        help="Lightweight backbone to replace ViT-H "
                             "(default: None = share teacher's ViT-H)")

    # Loss weights
    parser.add_argument("--encoder-weight", type=float, default=1.0,
                        help="Loss weight for encoder memory MSE")
    parser.add_argument("--prompt-weight", type=float, default=0.5,
                        help="Loss weight for prompt-after-encoder MSE")
    parser.add_argument("--decoder-weight", type=float, default=1.0,
                        help="Loss weight for decoder hidden states MSE (all layers)")
    parser.add_argument("--refbox-weight", type=float, default=1.0,
                        help="Loss weight for reference box MSE (all layers)")
    parser.add_argument("--boxpred-weight", type=float, default=0.5,
                        help="Loss weight for box prediction (bbox_embed) MSE")
    parser.add_argument("--scoring-weight", type=float, default=0.3,
                        help="Loss weight for scoring head MSE")
    parser.add_argument("--presence-weight", type=float, default=1.0,
                        help="Loss weight for decoder presence prediction MSE")
    parser.add_argument("--backbone-weight", type=float, default=2.0,
                        help="Loss weight for backbone FPN feature MSE "
                             "(only used with --student-backbone)")

    args = parser.parse_args()

    num_gpus = args.gpus or torch.cuda.device_count()

    if num_gpus > 1:
        print(f"Launching {num_gpus} GPU workers via mp.spawn...")
        mp.spawn(worker, args=(num_gpus, args), nprocs=num_gpus, join=True)
    else:
        worker_single(args)


if __name__ == "__main__":
    main()
