# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""INT8 calibration for the encoder+decoder TensorRT engine.

Unlike the backbone calibrator (which only needs raw images), this calibrator
must generate *intermediate features* for each calibration sample:
  1. Run the image through the SAM3 backbone → FPN features
  2. Run random COCO class names through the text encoder → text embeddings
  3. Feed (img_feat, img_pos, text_feats, text_mask) to the TRT calibrator

The backbone runs in PyTorch (or via a pre-built TRT engine) to produce
features.  Text embeddings come from the frozen text encoder.

Calibration results are cached to disk for reuse.
"""

import os
import random
from pathlib import Path

import numpy as np
import torch
from torchvision.transforms import v2

try:
    import tensorrt as trt
except ImportError:
    trt = None


from sam3.coco_classes import COCO_CLASSES


def _list_images(directory, extensions=(".jpg", ".jpeg", ".png", ".bmp")):
    """List image files in a directory (non-recursive)."""
    p = Path(directory)
    files = []
    for ext in extensions:
        files.extend(p.glob(f"*{ext}"))
        files.extend(p.glob(f"*{ext.upper()}"))
    return sorted(set(files))


_CalibratorBase = trt.IInt8EntropyCalibrator2 if trt is not None else object


class EncDecCalibrator(_CalibratorBase):
    """TensorRT INT8 calibrator for the encoder+decoder pipeline.

    For each calibration step:
      1. Load an image, run through backbone → FPN level (72x72)
      2. Sample random COCO class names, run through text encoder
      3. Pack (img_feat, img_pos, text_feats, text_mask) for TRT calibrator

    Args:
        model: SAM3 model instance (on CUDA, eval mode).
        image_dir: Directory with calibration images (e.g. COCO train2017/).
        max_classes: Fixed batch dimension (number of class slots).
        num_images: Number of calibration images to use.
        cache_file: Path to cache the calibration table.
        seed: Random seed for reproducibility.
        trt_backbone: Optional TRTBackbone instance to use instead of PyTorch.
    """

    def __init__(
        self,
        model,
        image_dir: str,
        max_classes: int = 4,
        num_images: int = 256,
        cache_file: str = "calibration_enc_dec.cache",
        seed: int = 42,
        trt_backbone=None,
    ):
        if trt is None:
            raise ImportError("tensorrt is required for INT8 calibration")

        super().__init__()

        self.model = model
        self.max_classes = max_classes
        self.cache_file = cache_file
        self.trt_backbone = trt_backbone

        # List and subsample images
        all_images = _list_images(image_dir)
        if len(all_images) == 0:
            raise FileNotFoundError(
                f"No images found in {image_dir}. "
                f"Expected .jpg/.png files (e.g., COCO train2017/)."
            )

        rng = random.Random(seed)
        if num_images < len(all_images):
            self.image_paths = rng.sample(all_images, num_images)
        else:
            self.image_paths = list(all_images)
            rng.shuffle(self.image_paths)

        self.rng = rng
        self.num_batches = len(self.image_paths)  # batch_size=1 (one image at a time)
        self.current_batch = 0

        # Same transform as inference
        self.transform = v2.Compose([
            v2.ToDtype(torch.uint8, scale=True),
            v2.Resize(size=(1008, 1008)),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        # Pre-compute position encoding for FPN level (72x72)
        pe_module = model.backbone.vision_backbone.position_encoding
        dummy = torch.zeros(1, 256, 72, 72, device="cuda")
        with torch.no_grad():
            self._pos_enc = pe_module(dummy).detach()  # (1, 256, 72, 72)

        # Pre-allocate device buffers for 4 inputs
        self._buf_img = torch.empty(
            max_classes, 256, 72, 72, dtype=torch.float32, device="cuda"
        )
        self._buf_pos = torch.empty(
            max_classes, 256, 72, 72, dtype=torch.float32, device="cuda"
        )
        self._buf_text = torch.empty(
            32, max_classes, 256, dtype=torch.float32, device="cuda"
        )
        self._buf_mask = torch.ones(
            max_classes, 32, dtype=torch.float32, device="cuda"
        )

        print(
            f"EncDec INT8 Calibrator: {len(self.image_paths)} images, "
            f"max_classes={max_classes}"
        )

    def get_batch_size(self):
        return 1

    def get_batch(self, names):
        """Generate one calibration sample: image features + random text."""
        if self.current_batch >= self.num_batches:
            return None

        img_path = self.image_paths[self.current_batch]

        from PIL import Image

        try:
            img = Image.open(img_path).convert("RGB")
            tensor = v2.functional.to_image(img)
            tensor = self.transform(tensor).unsqueeze(0).cuda()  # (1, 3, 1008, 1008)
        except Exception as e:
            print(f"  Warning: skipping {img_path}: {e}")
            tensor = torch.zeros(1, 3, 1008, 1008, device="cuda")

        # Run backbone
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            if self.trt_backbone is not None:
                backbone_out = self.trt_backbone.forward_image(tensor)
            else:
                backbone_out = self.model.backbone.forward_image(tensor)

        # Extract last FPN level (72x72) — matches num_feature_levels=1
        fpn_feat = backbone_out["backbone_fpn"][-1]  # (1, 256, 72, 72)

        # Random class names
        n_classes = self.rng.randint(1, min(self.max_classes, len(COCO_CLASSES)))
        class_names = self.rng.sample(COCO_CLASSES, n_classes)

        # Text encoder
        with torch.no_grad():
            text_out = self.model.backbone.forward_text(class_names, device="cuda")
        text_feats = text_out["language_features"]  # (32, n_classes, 256)
        text_mask = text_out["language_mask"]        # (n_classes, 32)

        # Pack into fixed-size buffers
        # Image: replicate FPN features to max_classes
        self._buf_img[:] = fpn_feat.expand(self.max_classes, -1, -1, -1)
        # Position encoding: replicate
        self._buf_pos[:] = self._pos_enc.expand(self.max_classes, -1, -1, -1)
        # Text: pad to max_classes with zeros
        self._buf_text.zero_()
        self._buf_text[:, :n_classes, :] = text_feats.float()
        # Mask: pad with True (padding) — cast bool to float for TRT
        self._buf_mask.fill_(1.0)
        self._buf_mask[:n_classes, :] = text_mask.float()

        self.current_batch += 1
        if self.current_batch % 50 == 0 or self.current_batch == self.num_batches:
            print(
                f"  Calibration sample {self.current_batch}/{self.num_batches}"
            )

        return [
            self._buf_img.data_ptr(),
            self._buf_pos.data_ptr(),
            self._buf_text.data_ptr(),
            self._buf_mask.data_ptr(),
        ]

    def read_calibration_cache(self):
        """Read cached calibration table if available."""
        if os.path.exists(self.cache_file):
            print(f"Reading calibration cache: {self.cache_file}")
            with open(self.cache_file, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        """Write calibration table to disk for reuse."""
        print(f"Writing calibration cache: {self.cache_file}")
        with open(self.cache_file, "wb") as f:
            f.write(cache)
