"""EfficientSAM3 lightweight backbone construction.

Provides `build_efficientsam3_model()` which assembles a full SAM3 model
with a lightweight student backbone (EfficientViT, RepViT, or TinyViT)
instead of the original ViT-H.

The student backbone was distilled from ViT-H features and produces matching
(B, 1024, 72, 72) feature maps via a learned projection head.
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from sam3.model_builder import (
    _create_dot_product_scoring,
    _create_geometry_encoder,
    _create_position_encoding,
    _create_sam3_model,
    _create_sam3_transformer,
    _create_text_encoder,
    _create_vit_neck,
    _create_vl_backbone,
    _load_checkpoint,
    _setup_device_and_mode,
)


class ImageStudentEncoder(nn.Module):
    """Projects lightweight backbone features to SAM3-compatible shape.

    Maps (B, in_channels, H, W) -> (B, embed_dim, embed_size, embed_size)
    via a 1x1 channel projection + 3x3 refinement + bilinear interpolation.
    """

    def __init__(self, backbone, in_channels, embed_dim, embed_size, img_size):
        super().__init__()
        self.backbone = backbone
        self.embed_size = embed_size
        self.img_size = img_size
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
        )

    def forward(self, x):
        feats = self.backbone(x)
        feats = self.head(feats)
        if feats.shape[-1] != self.embed_size or feats.shape[-2] != self.embed_size:
            feats = F.interpolate(
                feats,
                size=(self.embed_size, self.embed_size),
                mode="bilinear",
                align_corners=False,
            )
        return feats


def _build_efficient_backbone(backbone_type, model_name):
    """Build an EfficientSAM3 student backbone trunk.

    Returns an nn.Module that:
    - Takes (B, 3, H, W) input
    - Returns list of [(B, 1024, 72, 72)] feature maps
    - Has .channel_list = [1024] attribute

    This matches the interface expected by Sam3DualViTDetNeck.
    """
    if backbone_type == "efficientvit":
        from sam3.backbones.efficientvit.efficientvit.backbone import (
            efficientvit_backbone_b0,
            efficientvit_backbone_b1,
            efficientvit_backbone_b2,
        )
        factory = {
            "b0": efficientvit_backbone_b0,
            "b1": efficientvit_backbone_b1,
            "b2": efficientvit_backbone_b2,
        }
        if model_name not in factory:
            raise ValueError(f"Unknown EfficientViT model: {model_name}")
        backbone = factory[model_name]()

        class EfficientViTTrunkWrapper(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
                self.channel_list = [model.width_list[-1]]

            def forward(self, x):
                x = x[0] if isinstance(x, list) else x
                out = self.model(x)
                return out["stage_final"]

        wrapped = EfficientViTTrunkWrapper(backbone)
        in_channels = wrapped.channel_list[0]

    elif backbone_type == "repvit":
        from sam3.backbones.repvit import repvit_m0_9, repvit_m1_1, repvit_m2_3
        factory = {
            "m0.9": repvit_m0_9, "m0_9": repvit_m0_9,
            "m1.1": repvit_m1_1, "m1_1": repvit_m1_1,
            "m2.3": repvit_m2_3, "m2_3": repvit_m2_3,
        }
        if model_name not in factory:
            raise ValueError(f"Unknown RepViT model: {model_name}")
        backbone = factory[model_name](distillation=False, num_classes=0)

        class RepViTTrunkWrapper(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
                dummy = torch.zeros(1, 3, 224, 224)
                with torch.no_grad():
                    for f in model.features:
                        dummy = f(dummy)
                self.channel_list = [dummy.shape[1]]

            def forward(self, x):
                for f in self.model.features:
                    x = f(x)
                return x

        wrapped = RepViTTrunkWrapper(backbone)
        in_channels = wrapped.channel_list[0]

    elif backbone_type == "tinyvit":
        from sam3.backbones.tiny_vit import (
            tiny_vit_5m_224, tiny_vit_11m_224, tiny_vit_21m_224,
        )
        factory = {
            "5m": tiny_vit_5m_224,
            "11m": tiny_vit_11m_224,
            "21m": tiny_vit_21m_224,
        }
        if model_name not in factory:
            raise ValueError(f"Unknown TinyViT model: {model_name}")
        backbone = factory[model_name](img_size=1008, num_classes=0)

        class TinyViTTrunkWrapper(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
                self.channel_list = [model.layers[-1].dim]

            def forward(self, x):
                x = self.model.patch_embed(x)
                for layer in self.model.layers:
                    x = layer(x)
                B, L, C = x.shape
                side = int(L ** 0.5)
                x = x.view(B, side, side, C).permute(0, 3, 1, 2).contiguous()
                return x

        wrapped = TinyViTTrunkWrapper(backbone)
        in_channels = wrapped.channel_list[0]

    else:
        raise ValueError(f"Unknown backbone type: {backbone_type}")

    # Wrap with ImageStudentEncoder (projects to 1024 channels, interpolates to 72x72)
    student_encoder = ImageStudentEncoder(
        backbone=wrapped,
        in_channels=in_channels,
        embed_dim=1024,
        embed_size=72,
        img_size=1008,
    )
    student_encoder.channel_list = [1024]

    # Wrap to return list (Sam3DualViTDetNeck expects trunk() -> list)
    class ListWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            self.channel_list = model.channel_list

        def forward(self, x):
            # Sam3DualViTDetNeck passes tensor_list (list of tensors)
            if isinstance(x, (list, tuple)):
                x = x[0]
            return [self.model(x)]

    return ListWrapper(student_encoder)


def build_efficientsam3_model(
    backbone_type,
    model_name,
    checkpoint_path,
    device="cuda",
    eval_mode=True,
    enable_inst_interactivity=False,
):
    """Build an EfficientSAM3 model with a lightweight backbone.

    Returns a Sam3Image model with the same interface as the original SAM3,
    but with the ViT-H backbone replaced by a student backbone.
    """
    trunk = _build_efficient_backbone(backbone_type, model_name)

    position_encoding = _create_position_encoding(precompute_resolution=1008)
    vision_encoder = _create_vit_neck(
        position_encoding, trunk,
        enable_inst_interactivity=enable_inst_interactivity,
    )

    bpe_path = os.path.join(
        os.path.dirname(__file__), "assets", "bpe_simple_vocab_16e6.txt.gz"
    )
    text_encoder = _create_text_encoder(bpe_path)

    backbone = _create_vl_backbone(vision_encoder, text_encoder)
    transformer = _create_sam3_transformer()
    dot_prod_scoring = _create_dot_product_scoring()
    input_geometry_encoder = _create_geometry_encoder()

    if enable_inst_interactivity:
        from sam3.model_builder import build_tracker
        from sam3.model.sam1_task_predictor import SAM3InteractiveImagePredictor
        sam3_pvs_base = build_tracker(apply_temporal_disambiguation=False)
        inst_predictor = SAM3InteractiveImagePredictor(sam3_pvs_base)
    else:
        inst_predictor = None

    model = _create_sam3_model(
        backbone, transformer, input_geometry_encoder,
        segmentation_head=None, dot_prod_scoring=dot_prod_scoring,
        inst_interactive_predictor=inst_predictor, eval_mode=eval_mode,
    )

    if checkpoint_path:
        _load_checkpoint(model, checkpoint_path)

    model = _setup_device_and_mode(model, device, eval_mode)
    return model
