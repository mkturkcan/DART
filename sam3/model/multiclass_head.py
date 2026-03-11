"""
Multi-class scoring head for single-pass SAM3 inference.

Replaces DotProductScoring (which pools all text into one embedding and
produces a single score per query) with per-class scoring.  Each query
gets a score against each class, enabling multi-class detection in a
single encoder+decoder pass.

Architecture:
    query_proj: Linear(d_model, d_proj) -- project decoder hidden states
    text_proj:  Linear(d_model, d_proj) -- project per-class mean-pooled text
    Score matrix: (query_proj(hs) @ text_proj(text).T) * scale

Can be warm-started from existing DotProductScoring weights.
"""

import math

import torch
import torch.nn as nn


class MultiClassScoring(nn.Module):
    """Per-query, per-class dot-product scoring.

    Given decoder hidden states (Q queries x d_model) and pre-computed
    per-class text embeddings (N classes x d_model), produces a score
    matrix of shape (Q x N).

    This is the multi-class generalization of DotProductScoring, which
    pools text into a single vector and produces (Q x 1) scores.
    """

    def __init__(self, d_model: int = 256, d_proj: int = 256):
        super().__init__()
        self.query_proj = nn.Linear(d_model, d_proj)
        self.text_proj = nn.Linear(d_model, d_proj)
        self.scale = 1.0 / math.sqrt(d_proj)

    def forward(
        self,
        hs: torch.Tensor,
        per_class_text: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            hs: Decoder hidden states, (num_layers, B, Q, d_model).
            per_class_text: Mean-pooled, prompt_mlp-processed text embeddings
                per class, (N, d_model).

        Returns:
            Per-class logits, (num_layers, B, Q, N).
        """
        q = self.query_proj(hs)  # (L, B, Q, d_proj)
        t = self.text_proj(per_class_text)  # (N, d_proj)
        logits = torch.einsum("lbqd,nd->lbqn", q, t) * self.scale
        return logits

    @classmethod
    def from_dot_product_scoring(cls, dot_prod_scoring) -> "MultiClassScoring":
        """Initialize from an existing DotProductScoring module.

        Copies hs_proj -> query_proj and prompt_proj -> text_proj for
        warm-start training.
        """
        d_model = dot_prod_scoring.hs_proj.in_features
        d_proj = dot_prod_scoring.hs_proj.out_features

        module = cls(d_model=d_model, d_proj=d_proj)
        module.query_proj.load_state_dict(dot_prod_scoring.hs_proj.state_dict())
        module.text_proj.load_state_dict(dot_prod_scoring.prompt_proj.state_dict())
        return module


@torch.inference_mode()
def precompute_class_embeddings(
    model,
    class_names: list,
    device: str = "cuda",
) -> torch.Tensor:
    """Pre-compute per-class mean-pooled text embeddings.

    Runs text through: text encoder -> prompt_mlp -> mean_pool.
    These embeddings are used as input to MultiClassScoring.text_proj.

    Args:
        model: Sam3Image model.
        class_names: List of class name strings.
        device: Device for computation.

    Returns:
        (N, d_model) tensor of per-class text embeddings.
    """
    # Process in chunks to avoid OOM with very large class lists
    chunk_size = 100
    all_pooled = []

    for start in range(0, len(class_names), chunk_size):
        chunk = class_names[start : start + chunk_size]
        text_out = model.backbone.forward_text(chunk, device=device)
        text_feats = text_out["language_features"]  # (seq, chunk_N, d)
        text_mask = text_out["language_mask"]  # (chunk_N, seq)

        scoring = model.dot_prod_scoring
        if scoring.prompt_mlp is not None:
            text_feats = scoring.prompt_mlp(text_feats)

        pooled = scoring.mean_pool_text(text_feats, text_mask)  # (chunk_N, d)
        all_pooled.append(pooled)

    return torch.cat(all_pooled, dim=0)  # (N, d_model)


@torch.inference_mode()
def precompute_concat_text(
    model,
    class_names: list,
    device: str = "cuda",
):
    """Pre-compute concatenated text for single-pass encoder/decoder.

    Concatenates valid (non-padding) text tokens from all classes into a
    single sequence.  Used as the prompt input to encoder and decoder.

    Args:
        model: Sam3Image model.
        class_names: List of class name strings.
        device: Device for computation.

    Returns:
        concat_text: (total_seq, 1, d) — concatenated valid text tokens.
        concat_mask: (1, total_seq) — all False (no padding).
        batched_text: (seq, N, d) — per-class text features (for reference).
        batched_mask: (N, seq) — per-class text mask.
    """
    text_out = model.backbone.forward_text(class_names, device=device)
    text_feats = text_out["language_features"]  # (seq, N, d)
    text_mask = text_out["language_mask"]  # (N, seq)

    N = len(class_names)
    all_tokens = []
    for i in range(N):
        valid = ~text_mask[i]  # (seq,) True = valid
        tokens = text_feats[valid, i, :]  # (valid_i, d)
        all_tokens.append(tokens)

    concat = torch.cat(all_tokens, dim=0)  # (total_seq, d)
    concat_text = concat.unsqueeze(1)  # (total_seq, 1, d)
    concat_mask = torch.zeros(
        1, concat.shape[0], dtype=torch.bool, device=device
    )  # all valid

    return concat_text, concat_mask, text_feats, text_mask
