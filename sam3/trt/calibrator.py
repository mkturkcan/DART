# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""INT8 calibration data feeder for TensorRT engine building.

Loads a random subset of COCO images (no annotations needed), applies the
same pre-processing as SAM3 inference (resize to 1008, normalize to [-1, 1]),
and feeds them to the TensorRT INT8 calibrator one batch at a time.

Calibration results are cached to disk so subsequent engine builds skip
the calibration pass entirely.
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


def _list_images(directory: str, extensions=(".jpg", ".jpeg", ".png", ".bmp")):
    """List image files in a directory (non-recursive)."""
    p = Path(directory)
    files = []
    for ext in extensions:
        files.extend(p.glob(f"*{ext}"))
        files.extend(p.glob(f"*{ext.upper()}"))
    return sorted(set(files))


# Determine the base class at import time so the class definition is valid
# whether or not tensorrt is installed.
_CalibratorBase = trt.IInt8EntropyCalibrator2 if trt is not None else object


class CocoCalibrator(_CalibratorBase):
    """TensorRT INT8 entropy calibrator using COCO images.

    Inherits from ``trt.IInt8EntropyCalibrator2`` for best quantization quality.
    """

    def __init__(
        self,
        image_dir: str,
        num_images: int = 512,
        batch_size: int = 1,
        resolution: int = 1008,
        cache_file: str = "calibration.cache",
        seed: int = 42,
    ):
        if trt is None:
            raise ImportError("tensorrt is required for INT8 calibration")

        super().__init__()

        self.batch_size = batch_size
        self.resolution = resolution
        self.cache_file = cache_file

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

        self.num_batches = (len(self.image_paths) + batch_size - 1) // batch_size
        self.current_batch = 0

        # Same transform as Sam3MultiClassPredictorFast
        self.transform = v2.Compose([
            v2.ToDtype(torch.uint8, scale=True),
            v2.Resize(size=(resolution, resolution)),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        # Pre-allocate device buffer for one batch
        self.device_input = torch.empty(
            batch_size, 3, resolution, resolution,
            dtype=torch.float32, device="cuda",
        )

        print(
            f"INT8 Calibrator: {len(self.image_paths)} images, "
            f"{self.num_batches} batches (bs={batch_size})"
        )

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        """Load and preprocess the next batch of calibration images."""
        if self.current_batch >= self.num_batches:
            return None

        start = self.current_batch * self.batch_size
        end = min(start + self.batch_size, len(self.image_paths))
        batch_paths = self.image_paths[start:end]

        from PIL import Image

        images = []
        for p in batch_paths:
            try:
                img = Image.open(p).convert("RGB")
                tensor = v2.functional.to_image(img)
                tensor = self.transform(tensor)
                images.append(tensor)
            except Exception as e:
                print(f"  Warning: skipping {p}: {e}")
                # Use zeros as fallback
                images.append(torch.zeros(3, self.resolution, self.resolution))

        # Pad batch if needed (last batch may be smaller)
        while len(images) < self.batch_size:
            images.append(torch.zeros(3, self.resolution, self.resolution))

        batch = torch.stack(images).cuda()
        self.device_input.copy_(batch)

        self.current_batch += 1
        if self.current_batch % 50 == 0 or self.current_batch == self.num_batches:
            print(
                f"  Calibration batch {self.current_batch}/{self.num_batches}"
            )

        return [self.device_input.data_ptr()]

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
