"""
Student SAM3 model assembly.

Builds a SAM3 model with the ViT backbone replaced by a lightweight student
backbone + FPN adapter. Everything else (encoder, decoder, segmentation head,
text encoder, scoring) is loaded from the teacher checkpoint and frozen.
"""

from typing import Dict, List, Optional

import pkg_resources
import torch
import torch.nn as nn
from iopath.common.file_io import g_pathmgr

from sam3.model.position_encoding import PositionEmbeddingSine
from sam3.model.vl_combiner import SAM3VLBackbone
from sam3.model_builder import (
    _create_dot_product_scoring,
    _create_geometry_encoder,
    _create_segmentation_head,
    _create_sam3_transformer,
    _create_text_encoder,
    _create_sam3_model,
    download_ckpt_from_hf,
)
from sam3.distillation.student_backbone import StudentBackbone, build_student_backbone


class StudentVLBackbone(nn.Module):
    """Vision-language backbone with student vision + teacher text encoder.

    Drop-in replacement for SAM3VLBackbone. Uses StudentBackbone for image
    features and the teacher's language_backbone for text encoding.
    """

    def __init__(self, student_backbone: StudentBackbone, text_encoder: nn.Module):
        super().__init__()
        self.student_backbone = student_backbone
        self.language_backbone = text_encoder
        # Match SAM3VLBackbone interface
        self.scalp = 0  # StudentBackbone already handles this

    def forward_image(self, samples: torch.Tensor) -> Dict:
        return self.student_backbone.forward_image(samples)

    def forward_text(
        self, captions, input_boxes=None, additional_text=None, device="cuda"
    ):
        """Forward text through teacher's language backbone.

        Reuses SAM3VLBackbone._forward_text_no_ack_ckpt logic.
        """
        from copy import copy
        from torch.nn.attention import sdpa_kernel, SDPBackend

        output = {}
        text_to_encode = copy(captions)
        if additional_text is not None:
            text_to_encode += additional_text

        sdpa_context = sdpa_kernel(
            [
                SDPBackend.MATH,
                SDPBackend.EFFICIENT_ATTENTION,
                SDPBackend.FLASH_ATTENTION,
            ]
        )

        with sdpa_context:
            text_attention_mask, text_memory, text_embeds = self.language_backbone(
                text_to_encode, input_boxes, device=device
            )

        if additional_text is not None:
            output["additional_text_features"] = text_memory[
                :, -len(additional_text) :
            ]
            output["additional_text_mask"] = text_attention_mask[
                -len(additional_text) :
            ]

        text_memory = text_memory[:, : len(captions)]
        text_attention_mask = text_attention_mask[: len(captions)]
        text_embeds = text_embeds[:, : len(captions)]
        output["language_features"] = text_memory
        output["language_mask"] = text_attention_mask
        output["language_embeds"] = text_embeds
        return output

    def forward(self, samples, captions, input_boxes=None, additional_text=None):
        output = self.forward_image(samples)
        device = output["vision_features"].device
        output.update(
            self.forward_text(captions, input_boxes, additional_text, device)
        )
        return output


def _load_teacher_weights(
    model: nn.Module,
    checkpoint_path: str,
    student_backbone: StudentBackbone,
):
    """Load teacher weights into the student model, skipping vision backbone.

    The teacher checkpoint has keys like:
        detector.backbone.vision_backbone.* → skip (replaced by student)
        detector.backbone.language_backbone.* → load into backbone.language_backbone
        detector.transformer.* → load
        detector.dot_prod_scoring.* → load
        detector.segmentation_head.* → load
        detector.geometry_encoder.* → load

    Args:
        model: the Sam3Image student model
        checkpoint_path: path to teacher checkpoint
        student_backbone: the student backbone (for reference, not loaded)
    """
    with g_pathmgr.open(checkpoint_path, "rb") as f:
        ckpt = torch.load(f, map_location="cpu", weights_only=True)
    if "model" in ckpt and isinstance(ckpt["model"], dict):
        ckpt = ckpt["model"]

    # Strip 'detector.' prefix
    teacher_state = {
        k.replace("detector.", ""): v for k, v in ckpt.items() if "detector" in k
    }

    # Filter out vision backbone keys (we have a different architecture)
    skip_prefixes = ("backbone.vision_backbone.",)
    filtered_state = {
        k: v
        for k, v in teacher_state.items()
        if not any(k.startswith(p) for p in skip_prefixes)
    }

    # Map teacher's backbone.language_backbone.* to our backbone.language_backbone.*
    # This should work directly since StudentVLBackbone uses the same attribute name.

    missing, unexpected = model.load_state_dict(filtered_state, strict=False)

    # Expected missing keys: student_backbone adapter weights
    student_prefix = "backbone.student_backbone."
    expected_missing = [k for k in missing if k.startswith(student_prefix)]
    truly_missing = [k for k in missing if not k.startswith(student_prefix)]

    if truly_missing:
        print(f"WARNING: Truly missing keys (not student backbone): {truly_missing}")
    if unexpected:
        print(f"WARNING: Unexpected keys: {unexpected}")

    loaded_count = len(filtered_state) - len(unexpected)
    print(
        f"Loaded {loaded_count} teacher weight tensors. "
        f"Skipped {len(teacher_state) - len(filtered_state)} vision backbone tensors. "
        f"Student adapter has {len(expected_missing)} uninitialized tensors."
    )


def freeze_teacher_components(model: nn.Module):
    """Freeze everything except the student backbone adapter.

    The student backbone's timm backbone is already frozen internally.
    We freeze: encoder, decoder, segmentation head, text encoder, scoring.
    We leave trainable: FPN adapter layers in StudentBackbone.
    """
    # Freeze all parameters first
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze only the student adapter
    student_bb = model.backbone.student_backbone
    for name, param in student_bb.named_parameters():
        # adapter layers are trainable; backbone is already frozen internally
        if "adapters" in name:
            param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} parameters ({100*trainable/total:.2f}%)")


def build_sam3_student_model(
    backbone_config: str = "efficientvit_l1",
    teacher_checkpoint: str = None,
    load_from_HF: bool = True,
    bpe_path: str = None,
    device: str = "cuda",
    freeze_teacher: bool = True,
    pretrained_student: bool = True,
):
    """Build a SAM3 student model with lightweight backbone.

    Args:
        backbone_config: student backbone config name (see student_backbone.py)
        teacher_checkpoint: path to teacher SAM3 checkpoint
        load_from_HF: if True and no checkpoint provided, download from HF
        bpe_path: path to BPE tokenizer vocab
        device: target device
        freeze_teacher: freeze all teacher components (train only adapter)
        pretrained_student: use pre-trained student backbone weights

    Returns:
        Sam3Image model with student backbone
    """
    if bpe_path is None:
        bpe_path = pkg_resources.resource_filename(
            "sam3", "assets/bpe_simple_vocab_16e6.txt.gz"
        )

    # 1. Build student backbone (with pre-trained weights + adapter)
    student_bb = build_student_backbone(
        config_name=backbone_config,
        pretrained=pretrained_student,
        freeze_backbone=True,
    )

    # 2. Build text encoder (will be loaded from teacher checkpoint)
    text_encoder = _create_text_encoder(bpe_path)

    # 3. Create StudentVLBackbone (combines student vision + teacher text)
    backbone = StudentVLBackbone(student_bb, text_encoder)

    # 4. Build remaining components (same architecture as teacher)
    transformer = _create_sam3_transformer()
    dot_prod_scoring = _create_dot_product_scoring()
    segmentation_head = _create_segmentation_head()
    geometry_encoder = _create_geometry_encoder()

    # 5. Assemble Sam3Image
    model = _create_sam3_model(
        backbone=backbone,
        transformer=transformer,
        input_geometry_encoder=geometry_encoder,
        segmentation_head=segmentation_head,
        dot_prod_scoring=dot_prod_scoring,
        inst_interactive_predictor=None,
        eval_mode=True,
    )

    # 6. Load teacher weights
    if load_from_HF and teacher_checkpoint is None:
        teacher_checkpoint = download_ckpt_from_hf()
    if teacher_checkpoint is not None:
        _load_teacher_weights(model, teacher_checkpoint, student_bb)

    # 7. Freeze teacher components
    if freeze_teacher:
        freeze_teacher_components(model)

    # 8. Move to device
    model = model.to(device)

    return model
