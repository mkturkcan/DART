# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe

import math
from typing import Optional

import torch
from torch import nn


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(
        self,
        num_pos_feats,
        temperature: int = 10000,
        normalize: bool = True,
        scale: Optional[float] = None,
        precompute_resolution: Optional[int] = None,
    ):
        super().__init__()
        assert num_pos_feats % 2 == 0, "Expecting even model width"
        self.num_pos_feats = num_pos_feats // 2
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

        self.cache = {}
        # Precompute positional encodings to fill the cache and avoid symbolic
        # shape tracing errors in torch.compile.
        #
        # The precomputed sizes must match the actual FPN output sizes.  For
        # SAM3 with ViT patch_size=14 at 1008px input, the ViT produces 72x72
        # spatial features.  The FPN with scale_factors [4.0, 2.0, 1.0, 0.5]
        # then gives: 288, 144, 72, 36.
        #
        # Cached tensors are registered as non-persistent buffers so that
        # torch.compile(mode="max-autotune") with CUDA graph capture treats
        # them as proper module state (protected memory), avoiding the
        # "accessing tensor output of CUDAGraphs that has been overwritten"
        # error that occurs with plain dict-cached tensors.
        if precompute_resolution is not None:
            vit_size = precompute_resolution // 14  # 72 for 1008px
            precompute_sizes = [
                (vit_size * 4, vit_size * 4),    # FPN 4.0x -> 288
                (vit_size * 2, vit_size * 2),    # FPN 2.0x -> 144
                (vit_size, vit_size),            # FPN 1.0x -> 72
                (vit_size // 2, vit_size // 2),  # FPN 0.5x -> 36
            ]
            for size in precompute_sizes:
                tensors = torch.zeros((1, 1) + size, device="cuda")
                self.forward(tensors)
                buf = self.cache[size].clone().detach()
                self.register_buffer(
                    f"pos_{size[0]}x{size[1]}", buf, persistent=False
                )
            # Clear dict cache — lookups now go through registered buffers
            self.cache.clear()

    def precompute_for_resolution(self, resolution):
        """Precompute and register buffers for a given input resolution.

        Call this before torch.compile warmup when using a non-default
        resolution (e.g. --imgsz 672) to ensure CUDAGraph-safe buffers
        exist for all FPN output sizes.
        """
        vit_size = resolution // 14
        sizes = [
            (vit_size * 4, vit_size * 4),
            (vit_size * 2, vit_size * 2),
            (vit_size, vit_size),
            (vit_size // 2, vit_size // 2),
        ]
        for size in sizes:
            buf_name = f"pos_{size[0]}x{size[1]}"
            if getattr(self, buf_name, None) is not None:
                continue  # Already precomputed
            buf = next(self.buffers(), None)
            device = buf.device if buf is not None else "cuda"
            dummy = torch.zeros((1, 1) + size, device=device)
            self.forward(dummy)
            buf = self.cache[size].clone().detach()
            self.register_buffer(buf_name, buf, persistent=False)
        self.cache.clear()

    def _encode_xy(self, x, y):
        # The positions are expected to be normalized
        assert len(x) == len(y) and x.ndim == y.ndim == 1
        x_embed = x * self.scale
        y_embed = y * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, None] / dim_t
        pos_y = y_embed[:, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, 0::2].sin(), pos_x[:, 1::2].cos()), dim=2
        ).flatten(1)
        pos_y = torch.stack(
            (pos_y[:, 0::2].sin(), pos_y[:, 1::2].cos()), dim=2
        ).flatten(1)
        return pos_x, pos_y

    @torch.no_grad()
    def encode_boxes(self, x, y, w, h):
        pos_x, pos_y = self._encode_xy(x, y)
        pos = torch.cat((pos_y, pos_x, h[:, None], w[:, None]), dim=1)
        return pos

    encode = encode_boxes  # Backwards compatibility

    @torch.no_grad()
    def encode_points(self, x, y, labels):
        (bx, nx), (by, ny), (bl, nl) = x.shape, y.shape, labels.shape
        assert bx == by and nx == ny and bx == bl and nx == nl
        pos_x, pos_y = self._encode_xy(x.flatten(), y.flatten())
        pos_x, pos_y = pos_x.reshape(bx, nx, -1), pos_y.reshape(by, ny, -1)
        pos = torch.cat((pos_y, pos_x, labels[:, :, None]), dim=2)
        return pos

    @torch.no_grad()
    def forward(self, x):
        cache_key = (x.shape[-2], x.shape[-1])
        # Check registered buffer first (CUDA-graph safe for max-autotune)
        buf = getattr(self, f"pos_{cache_key[0]}x{cache_key[1]}", None)
        if buf is not None:
            return buf[None].repeat(x.shape[0], 1, 1, 1)
        # Dict cache fallback (used during precomputation and for
        # non-precomputed sizes in non-compiled paths)
        if cache_key in self.cache:
            return self.cache[cache_key][None].repeat(x.shape[0], 1, 1, 1)
        y_embed = (
            torch.arange(1, x.shape[-2] + 1, dtype=torch.float32, device=x.device)
            .view(1, -1, 1)
            .repeat(x.shape[0], 1, x.shape[-1])
        )
        x_embed = (
            torch.arange(1, x.shape[-1] + 1, dtype=torch.float32, device=x.device)
            .view(1, 1, -1)
            .repeat(x.shape[0], x.shape[-2], 1)
        )

        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        if cache_key is not None:
            self.cache[cache_key] = pos[0]
        return pos
