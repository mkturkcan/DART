"""
Lightweight student backbone with FPN adapter for SAM3 distillation.

Wraps a pre-trained lightweight backbone (e.g., EfficientViT-L1) and maps
its multi-scale outputs to match SAM3's backbone interface exactly.

SAM3 backbone output spec (after scalp=1, for 1008x1008 input):
  Level 0: (B, 256, 288, 288)  — segmentation head only
  Level 1: (B, 256, 144, 144)  — segmentation head only
  Level 2: (B, 256, 72, 72)    — encoder + segmentation head (most important)
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from sam3.model.position_encoding import PositionEmbeddingSine


# Target spatial sizes for each FPN level (SAM3 with 1008x1008 input, scalp=1)
TARGET_SIZES = [(288, 288), (144, 144), (72, 72)]
D_MODEL = 256


class FPNAdapterLevel(nn.Module):
    """Single FPN adapter level: project channels + interpolate + refine."""

    def __init__(self, in_channels: int, out_channels: int = D_MODEL):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)
        self.refine = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1, bias=True
        )

    def forward(self, x: torch.Tensor, target_size: Tuple[int, int]) -> torch.Tensor:
        x = self.proj(x)
        if x.shape[-2:] != target_size:
            x = F.interpolate(
                x, size=target_size, mode="bilinear", align_corners=False
            )
        x = self.refine(x)
        return x


class StudentBackbone(nn.Module):
    """Lightweight backbone + FPN adapter that matches SAM3's backbone interface.

    Uses a pre-trained backbone from timm with features_only=True, then maps
    each stage's output to (B, 256, H, W) at SAM3's expected spatial sizes.
    Only the FPN adapter layers are trainable; the backbone is frozen.

    Args:
        backbone_name: timm model name (e.g., 'efficientvit_l1.r224_in1k')
        pretrained: whether to load pre-trained weights
        num_levels: number of FPN levels to produce (3 for SAM3 with scalp=1)
        student_indices: which timm feature stages to use for each FPN level.
            Default (0, 1, 2) maps stage0→level0, stage1→level1, stage2→level2.
        target_sizes: spatial size for each FPN level output
        freeze_backbone: if True, freeze the backbone (only train adapter)
    """

    def __init__(
        self,
        backbone_name: str = "efficientvit_l1.r224_in1k",
        pretrained: bool = True,
        num_levels: int = 3,
        student_indices: Tuple[int, ...] = (0, 1, 2),
        target_sizes: Optional[List[Tuple[int, int]]] = None,
        freeze_backbone: bool = True,
    ):
        super().__init__()

        if target_sizes is None:
            target_sizes = TARGET_SIZES[:num_levels]

        assert len(student_indices) == num_levels
        assert len(target_sizes) == num_levels

        self.num_levels = num_levels
        self.student_indices = student_indices
        self.target_sizes = target_sizes

        # Create backbone from timm
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=list(student_indices),
        )

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()

        # Get channel dims from timm's feature_info
        channels = self.backbone.feature_info.channels()
        # channels corresponds to the selected out_indices
        selected_channels = [channels[i] for i in range(len(student_indices))]

        # FPN adapter: one level per selected stage
        self.adapters = nn.ModuleList(
            [FPNAdapterLevel(ch, D_MODEL) for ch in selected_channels]
        )

        # Position encoding (deterministic, no trainable params)
        self.position_encoding = PositionEmbeddingSine(
            num_pos_feats=D_MODEL, normalize=True
        )

        self._freeze_backbone = freeze_backbone

    def train(self, mode: bool = True):
        """Override to keep backbone frozen even when training."""
        super().train(mode)
        if self._freeze_backbone:
            self.backbone.eval()
        return self

    @property
    def trainable_params(self) -> int:
        """Number of trainable parameters (adapter only)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Extract multi-scale features from the backbone.

        Args:
            x: input image tensor (B, 3, H, W), already preprocessed

        Returns:
            List of FPN-adapted features, each (B, 256, target_H, target_W)
        """
        # Use no_grad only if backbone is fully frozen (no LoRA params)
        has_trainable = any(p.requires_grad for p in self.backbone.parameters())
        if self._freeze_backbone and not has_trainable:
            with torch.no_grad():
                raw_feats = self.backbone(x)
        else:
            raw_feats = self.backbone(x)

        adapted = []
        for i, (feat, adapter) in enumerate(zip(raw_feats, self.adapters)):
            adapted.append(adapter(feat, self.target_sizes[i]))

        return adapted

    def forward_image(self, samples: torch.Tensor) -> Dict[str, object]:
        """Match SAM3VLBackbone.forward_image interface.

        Args:
            samples: preprocessed image tensor (B, 3, 1008, 1008)

        Returns:
            Dict matching SAM3 backbone output format:
                - vision_features: last FPN level (B, 256, 72, 72)
                - vision_pos_enc: list of pos encodings per level
                - backbone_fpn: list of feature tensors per level
                - sam2_backbone_out: None (not supported)
        """
        fpn_features = self.forward_features(samples)

        # Generate position encodings for each level
        pos_encs = []
        for feat in fpn_features:
            pos = self.position_encoding(feat).to(feat.dtype)
            pos_encs.append(pos)

        return {
            "vision_features": fpn_features[-1],  # Last level = encoder input
            "vision_pos_enc": pos_encs,
            "backbone_fpn": fpn_features,
            "sam2_backbone_out": None,
        }


# Registry of supported backbone configs
BACKBONE_CONFIGS = {
    "efficientvit_l1": {
        "backbone_name": "efficientvit_l1.r224_in1k",
        "student_indices": (0, 1, 2),
        # channels: [64, 128, 256] at strides [4, 8, 16]
    },
    "efficientvit_l2": {
        "backbone_name": "efficientvit_l2.r384_in1k",
        "student_indices": (0, 1, 2),
        # channels: [64, 128, 256] at strides [4, 8, 16]
    },
    "repvit_m2_3": {
        "backbone_name": "repvit_m2_3.dist_450e_in1k",
        "student_indices": (0, 1, 2),
    },
    "tiny_vit_21m": {
        "backbone_name": "tiny_vit_21m_224.dist_in22k_ft_in1k",
        "student_indices": (0, 1, 2),
    },
}


def build_student_backbone(
    config_name: str = "efficientvit_l1",
    pretrained: bool = True,
    freeze_backbone: bool = True,
) -> StudentBackbone:
    """Build a student backbone by config name.

    Args:
        config_name: one of BACKBONE_CONFIGS keys
        pretrained: load pre-trained backbone weights
        freeze_backbone: freeze backbone params (train only adapter)

    Returns:
        StudentBackbone instance
    """
    if config_name not in BACKBONE_CONFIGS:
        raise ValueError(
            f"Unknown config '{config_name}'. "
            f"Available: {list(BACKBONE_CONFIGS.keys())}"
        )

    cfg = BACKBONE_CONFIGS[config_name]
    return StudentBackbone(
        backbone_name=cfg["backbone_name"],
        pretrained=pretrained,
        student_indices=cfg["student_indices"],
        freeze_backbone=freeze_backbone,
    )
