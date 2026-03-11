"""
Feature distillation trainer for SAM3 student backbone.

Online distillation: runs teacher backbone and student backbone+adapter
side by side, minimizing MSE between their FPN outputs. Only the adapter
parameters receive gradients.

Supports:
  - Single GPU (RTX 4080 16GB, batch_size=1-2)
  - Multi-GPU DDP via torchrun (8xH100 80GB, batch_size=16+ per GPU)
"""

import os
import time
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision.transforms import v2

from PIL import Image


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


def setup_distributed():
    """Initialize distributed process group.

    Supports two launch modes:
      - torchrun:  sets RANK, LOCAL_RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT
      - srun:      sets SLURM_PROCID, SLURM_LOCALID, SLURM_NTASKS, SLURM_NODELIST
    Falls back to single-GPU if neither is detected.
    """
    if "RANK" in os.environ:
        # torchrun launch — env vars already set
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
    elif "SLURM_PROCID" in os.environ:
        # srun launch — map SLURM env vars to torch.distributed
        rank = int(os.environ["SLURM_PROCID"])
        local_rank = int(os.environ["SLURM_LOCALID"])
        world_size = int(os.environ["SLURM_NTASKS"])

        # Derive master address from SLURM_NODELIST if not already set
        if "MASTER_ADDR" not in os.environ:
            import subprocess
            nodelist = os.environ["SLURM_NODELIST"]
            result = subprocess.run(
                ["scontrol", "show", "hostnames", nodelist],
                capture_output=True, text=True,
            )
            os.environ["MASTER_ADDR"] = result.stdout.strip().split("\n")[0]
        if "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = "29500"

        os.environ["RANK"] = str(rank)
        os.environ["LOCAL_RANK"] = str(local_rank)
        os.environ["WORLD_SIZE"] = str(world_size)

        dist.init_process_group(
            backend="nccl",
            rank=rank,
            world_size=world_size,
        )
    else:
        return  # single-GPU, no distributed

    torch.cuda.set_device(local_rank)


def cleanup_distributed():
    if is_dist_initialized():
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ImageFolderDataset(Dataset):
    """Simple dataset that loads images from a directory (no labels needed).

    Supports COCO-style layout (flat directory of images) or any directory
    containing image files. No annotations required — we only need pixels.
    """

    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

    def __init__(self, root: str, resolution: int = 1008):
        self.root = root
        self.resolution = resolution
        self.image_paths = []

        for entry in os.scandir(root):
            if entry.is_file() and os.path.splitext(entry.name)[1].lower() in self.IMAGE_EXTENSIONS:
                self.image_paths.append(entry.path)

        self.image_paths.sort()

        self.transform = v2.Compose(
            [
                v2.ToDtype(torch.uint8, scale=True),
                v2.Resize(size=(resolution, resolution)),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        img = v2.functional.to_image(img)
        img = self.transform(img)
        return img


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

class FeatureDistillationLoss(nn.Module):
    """Weighted multi-scale MSE loss between teacher and student FPN features.

    The last level (encoder input) gets the highest weight since it's the
    most important for grounding performance.
    """

    def __init__(self, level_weights: Optional[list] = None):
        super().__init__()
        if level_weights is None:
            # 3 levels: level 0 (288x288), level 1 (144x144), level 2 (72x72)
            # Level 2 is most important (feeds encoder)
            level_weights = [0.15, 0.2, 0.65]
        self.level_weights = level_weights

    def forward(
        self,
        student_features: list,
        teacher_features: list,
    ) -> dict:
        assert len(student_features) == len(teacher_features) == len(self.level_weights)

        total_loss = torch.tensor(0.0, device=student_features[0].device)
        per_level_losses = []

        for i, (s_feat, t_feat, w) in enumerate(
            zip(student_features, teacher_features, self.level_weights)
        ):
            level_loss = F.mse_loss(s_feat, t_feat)
            total_loss = total_loss + w * level_loss
            per_level_losses.append(level_loss.item())

        return {
            "loss": total_loss,
            "per_level_losses": per_level_losses,
        }


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class DistillationTrainer:
    """Online feature distillation trainer.

    Runs the teacher backbone (frozen, FP16) and student backbone+adapter
    simultaneously, training only the adapter via MSE on FPN features.

    Supports single-GPU and multi-GPU (DDP) training. When launched via
    torchrun, automatically uses DistributedSampler and DDP.

    Args:
        teacher_model: full SAM3 model (only backbone.forward_image is used)
        student_backbone: StudentBackbone with trainable adapter
        data_dir: path to image directory (e.g., COCO train2017)
        output_dir: where to save checkpoints
        lr: learning rate for adapter (per-GPU; not scaled by world size)
        batch_size: images per GPU per step
        num_epochs: number of training epochs
        resolution: input resolution (should match SAM3's 1008)
        level_weights: per-level MSE weights
        num_workers: dataloader workers per GPU
        save_every: save checkpoint every N epochs
        log_every: print loss every N steps
    """

    def __init__(
        self,
        teacher_model,
        student_backbone,
        data_dir: str,
        output_dir: str = "distill_checkpoints",
        lr: float = 1e-3,
        batch_size: int = 1,
        num_epochs: int = 5,
        resolution: int = 1008,
        level_weights: Optional[list] = None,
        num_workers: int = 4,
        save_every: int = 1,
        log_every: int = 50,
        device: str = "cuda",
    ):
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.output_dir = output_dir
        self.save_every = save_every
        self.log_every = log_every

        # Distributed state
        self.distributed = is_dist_initialized()
        self.rank = get_rank()
        self.world_size = get_world_size()

        if self.distributed:
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            self.device = torch.device(f"cuda:{local_rank}")
        else:
            self.device = torch.device(device)

        if is_main_process():
            os.makedirs(output_dir, exist_ok=True)

        # Teacher: only need the vision backbone, frozen in FP16
        self.teacher_backbone = teacher_model.backbone.vision_backbone
        self.teacher_backbone.eval()
        self.teacher_backbone.to(self.device).half()
        for p in self.teacher_backbone.parameters():
            p.requires_grad = False

        # Scalp = 1 means drop last level, matching SAM3VLBackbone behavior
        self.scalp = getattr(teacher_model.backbone, "scalp", 1)

        # Student backbone with trainable adapter
        self.student = student_backbone.to(self.device)

        # Wrap adapter with DDP if distributed
        if self.distributed:
            # DDP wraps the whole student module but only adapter params have
            # requires_grad=True, so only those are synced.
            self.student = DDP(
                self.student,
                device_ids=[self.device.index],
                output_device=self.device.index,
                find_unused_parameters=False,
            )
            self._student_unwrapped = self.student.module
        else:
            self._student_unwrapped = self.student

        # Optimizer: only adapter params
        adapter_params = [p for p in self.student.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(adapter_params, lr=lr, weight_decay=0.01)

        # Loss
        self.criterion = FeatureDistillationLoss(level_weights)

        # Dataset + sampler
        self.dataset = ImageFolderDataset(data_dir, resolution=resolution)
        self.sampler = (
            DistributedSampler(self.dataset, shuffle=True) if self.distributed else None
        )
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=(self.sampler is None),
            sampler=self.sampler,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        )

        # Learning rate scheduler
        total_steps = len(self.dataloader) * num_epochs
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=total_steps, eta_min=lr * 0.01
        )

        # Mixed precision
        self.scaler = GradScaler()

        eff_batch = batch_size * self.world_size
        dist_print(f"Dataset: {len(self.dataset)} images from {data_dir}")
        dist_print(f"Trainable params: {self._student_unwrapped.trainable_params:,}")
        dist_print(
            f"Batch size: {batch_size}/GPU x {self.world_size} GPUs = "
            f"{eff_batch} effective"
        )
        dist_print(f"Epochs: {num_epochs}, Steps/epoch: {len(self.dataloader)}")
        dist_print(f"Total steps: {total_steps}")

    @torch.no_grad()
    def _get_teacher_features(self, images: torch.Tensor) -> list:
        """Run teacher backbone and return FPN features (after scalp)."""
        images_fp16 = images.half()
        sam3_out, _, _, _ = self.teacher_backbone(images_fp16)
        if self.scalp > 0:
            sam3_out = sam3_out[: -self.scalp]
        return [f.float() for f in sam3_out]

    def _get_student_features(self, images: torch.Tensor) -> list:
        """Run student backbone+adapter and return FPN features."""
        if self.distributed:
            return self.student.module.forward_features(images)
        return self.student.forward_features(images)

    def train_epoch(self, epoch: int) -> float:
        """Train one epoch. Returns average loss (reduced across ranks)."""
        self.student.train()

        if self.sampler is not None:
            self.sampler.set_epoch(epoch)

        total_loss = 0.0
        num_steps = 0
        t_epoch = time.perf_counter()

        for step, images in enumerate(self.dataloader):
            images = images.to(self.device, non_blocking=True)

            # Teacher forward (FP16, no grad)
            teacher_feats = self._get_teacher_features(images)

            # Student forward (mixed precision)
            self.optimizer.zero_grad()
            with autocast():
                student_feats = self._get_student_features(images)
                loss_dict = self.criterion(student_feats, teacher_feats)
                loss = loss_dict["loss"]

            # Backward (DDP auto-syncs adapter gradients)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            total_loss += loss.item()
            num_steps += 1

            if (step + 1) % self.log_every == 0 and is_main_process():
                avg = total_loss / num_steps
                lr = self.scheduler.get_last_lr()[0]
                per_level = loss_dict["per_level_losses"]
                elapsed = time.perf_counter() - t_epoch
                eff_imgs = (step + 1) * self.batch_size * self.world_size
                imgs_per_sec = eff_imgs / elapsed
                print(
                    f"  [Epoch {epoch+1}][{step+1}/{len(self.dataloader)}] "
                    f"loss={avg:.6f} (L0={per_level[0]:.4f} L1={per_level[1]:.4f} "
                    f"L2={per_level[2]:.4f}) lr={lr:.2e} {imgs_per_sec:.1f} img/s"
                )

        avg_loss = total_loss / max(num_steps, 1)
        return reduce_scalar(avg_loss, self.device)

    def save_checkpoint(self, epoch: int, loss: float):
        """Save student backbone adapter weights (rank 0 only)."""
        if not is_main_process():
            return

        path = os.path.join(self.output_dir, f"adapter_epoch{epoch+1}.pt")
        adapter_state = {}
        for name, param in self._student_unwrapped.named_parameters():
            if param.requires_grad:
                adapter_state[name] = param.data.cpu()
        torch.save(
            {
                "epoch": epoch + 1,
                "loss": loss,
                "adapter_state_dict": adapter_state,
                "student_state_dict": self._student_unwrapped.state_dict(),
            },
            path,
        )
        print(f"  Saved checkpoint: {path}")

    def train(self):
        """Run full training loop."""
        dist_print(f"\nStarting distillation training...")
        dist_print(f"{'='*60}")

        for epoch in range(self.num_epochs):
            t0 = time.perf_counter()
            avg_loss = self.train_epoch(epoch)
            elapsed = time.perf_counter() - t0

            dist_print(
                f"Epoch {epoch+1}/{self.num_epochs}: "
                f"avg_loss={avg_loss:.6f} time={elapsed:.1f}s"
            )

            if (epoch + 1) % self.save_every == 0 or (epoch + 1) == self.num_epochs:
                self.save_checkpoint(epoch, avg_loss)

            # Barrier so all ranks finish the epoch before next
            if self.distributed:
                dist.barrier()

        dist_print(f"\n{'='*60}")
        dist_print("Training complete!")

        if is_main_process():
            final_path = os.path.join(self.output_dir, "adapter_final.pt")
            torch.save(
                {
                    "epoch": self.num_epochs,
                    "student_state_dict": self._student_unwrapped.state_dict(),
                },
                final_path,
            )
            print(f"Final model saved: {final_path}")


# ---------------------------------------------------------------------------
# Phase 2: Encoder fine-tuning
# ---------------------------------------------------------------------------

class EncoderFineTuner(DistillationTrainer):
    """Phase 2: fine-tune the encoder to adapt to student features.

    After adapter-only distillation, we can optionally unfreeze the
    transformer encoder and fine-tune it with a lower learning rate.
    This helps close the accuracy gap by letting the encoder adapt
    to the student's feature distribution.
    """

    def __init__(
        self,
        student_model,
        data_dir: str,
        output_dir: str = "finetune_checkpoints",
        lr: float = 1e-5,
        batch_size: int = 1,
        num_epochs: int = 3,
        **kwargs,
    ):
        # We need the full student model (not just backbone)
        self.full_model = student_model

        self.distributed = is_dist_initialized()
        self.rank = get_rank()
        self.world_size = get_world_size()

        if self.distributed:
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            self.device = torch.device(f"cuda:{local_rank}")
        else:
            self.device = torch.device(kwargs.get("device", "cuda"))

        if is_main_process():
            os.makedirs(output_dir, exist_ok=True)

        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.output_dir = output_dir
        self.save_every = kwargs.get("save_every", 1)
        self.log_every = kwargs.get("log_every", 50)

        # Unfreeze encoder
        for param in self.full_model.transformer.encoder.parameters():
            param.requires_grad = True

        # Adapter params should already be trainable
        trainable_params = [
            p for p in self.full_model.parameters() if p.requires_grad
        ]

        self.optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=0.01)

        resolution = kwargs.get("resolution", 1008)
        num_workers = kwargs.get("num_workers", 4)
        self.dataset = ImageFolderDataset(data_dir, resolution=resolution)
        self.sampler = (
            DistributedSampler(self.dataset, shuffle=True) if self.distributed else None
        )
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=(self.sampler is None),
            sampler=self.sampler,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        )

        total_steps = len(self.dataloader) * num_epochs
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=total_steps, eta_min=lr * 0.01
        )
        self.scaler = GradScaler()

        trainable_count = sum(p.numel() for p in trainable_params)
        dist_print(
            f"Phase 2: Fine-tuning encoder + adapter ({trainable_count:,} params)"
        )
