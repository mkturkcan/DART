"""
LoRA (Low-Rank Adaptation) for student backbone fine-tuning.

Applies low-rank decomposition to Conv2d 1x1 and nn.Linear layers in timm
backbones, keeping the pre-trained weights frozen while training only the
small A/B matrices. This prevents mode collapse during distillation.

Supports: efficientvit (Conv1x1 only), repvit (Conv1x1 only),
          tiny_vit (Linear + Conv1x1).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """LoRA wrapper for nn.Linear: frozen W + trainable A @ B."""

    def __init__(self, original: nn.Linear, rank: int = 4, alpha: float = 1.0):
        super().__init__()
        self.original = original
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # Freeze original
        original.weight.requires_grad = False
        if original.bias is not None:
            original.bias.requires_grad = False

        # Create LoRA params on same device/dtype as original weight
        w = original.weight
        self.lora_A = nn.Parameter(torch.empty(rank, original.in_features, device=w.device, dtype=w.dtype))
        self.lora_B = nn.Parameter(torch.zeros(original.out_features, rank, device=w.device, dtype=w.dtype))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.original(x)
        out = out + F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scaling
        return out


class LoRAConv1x1(nn.Module):
    """LoRA wrapper for Conv2d 1x1: frozen W + trainable A @ B."""

    def __init__(self, original: nn.Conv2d, rank: int = 4, alpha: float = 1.0):
        super().__init__()
        assert original.kernel_size == (1, 1) and original.groups == 1
        self.original = original
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # Freeze original
        original.weight.requires_grad = False
        if original.bias is not None:
            original.bias.requires_grad = False

        in_ch = original.in_channels
        out_ch = original.out_channels

        # Create LoRA params on same device/dtype as original weight
        w = original.weight
        self.lora_A = nn.Parameter(torch.empty(rank, in_ch, 1, 1, device=w.device, dtype=w.dtype))
        self.lora_B = nn.Parameter(torch.zeros(out_ch, rank, 1, 1, device=w.device, dtype=w.dtype))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.original(x)
        out = out + F.conv2d(F.conv2d(x, self.lora_A), self.lora_B) * self.scaling
        return out


def apply_lora(module: nn.Module, rank: int = 4, alpha: float = 1.0) -> int:
    """Apply LoRA to all Linear and Conv2d 1x1 layers in a module.

    Replaces layers in-place. Returns the number of LoRA-wrapped layers.

    Args:
        module: the nn.Module to modify (typically the timm backbone)
        rank: LoRA rank (lower = fewer params, more regularization)
        alpha: LoRA scaling factor (typically equal to rank)

    Returns:
        Number of layers wrapped with LoRA
    """
    count = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            setattr(module, name, LoRALinear(child, rank=rank, alpha=alpha))
            count += 1
        elif (
            isinstance(child, nn.Conv2d)
            and child.kernel_size == (1, 1)
            and child.groups == 1
        ):
            setattr(module, name, LoRAConv1x1(child, rank=rank, alpha=alpha))
            count += 1
        else:
            count += apply_lora(child, rank=rank, alpha=alpha)
    return count


def lora_param_count(module: nn.Module) -> int:
    """Count trainable LoRA parameters (lora_A and lora_B only)."""
    return sum(
        p.numel()
        for n, p in module.named_parameters()
        if p.requires_grad and "lora_" in n
    )
