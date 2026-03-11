#!/usr/bin/env python3
"""
SAM3 backbone distillation: train a lightweight FPN adapter to replace ViT-H.

Phase 1 (adapter-only): Frozen teacher backbone + frozen student backbone,
    train only ~5M adapter params via feature MSE.

Phase 2 (optional): Fine-tune student backbone with lower lr.
    Use --lora-rank to apply LoRA instead of full fine-tuning.

Usage (single GPU):
    python scripts/distill.py \
        --data-dir /path/to/coco/train2017 \
        --checkpoint /path/to/sam3.pt \
        --backbone efficientvit_l1 \
        --epochs 5 --batch-size 2 --lr 1e-3

Usage (8xH100 via SLURM srun):
    salloc --nodes=1 --ntasks-per-node=8 --gpus-per-node=8 --cpus-per-task=12
    srun python scripts/distill.py \
        --data-dir /path/to/coco/train2017 \
        --checkpoint /path/to/sam3.pt \
        --backbone efficientvit_l1 \
        --epochs 5 --batch-size 16 --lr 1e-3

Usage (8xH100 via torchrun, if preferred):
    torchrun --nproc_per_node=8 scripts/distill.py \
        --data-dir /path/to/coco/train2017 \
        --checkpoint /path/to/sam3.pt \
        --backbone efficientvit_l1 \
        --epochs 5 --batch-size 16 --lr 1e-3

Phase 2 (backbone fine-tuning with LoRA, after phase 1):
    python scripts/distill.py \
        --data-dir /path/to/coco/train2017 \
        --checkpoint /path/to/sam3.pt \
        --backbone efficientvit_l1 \
        --adapter-checkpoint distill_checkpoints/adapter_final.pt \
        --phase 2 --lora-rank 4 --epochs 3 --lr 1e-4

Test (single GPU, no srun needed):
    python scripts/distill.py \
        --checkpoint /path/to/sam3.pt \
        --adapter-checkpoint distill_checkpoints/adapter_final.pt \
        --test-image /path/to/image.jpg --test-prompt "car"
"""

import argparse
import os
import time

import torch


def run_phase1(args):
    """Phase 1: Adapter-only distillation."""
    from sam3.model_builder import build_sam3_image_model
    from sam3.distillation.student_backbone import build_student_backbone
    from sam3.distillation.distill_trainer import (
        DistillationTrainer,
        dist_print,
        is_main_process,
        setup_distributed,
    )

    setup_distributed()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = f"cuda:{local_rank}" if torch.distributed.is_initialized() else args.device

    dist_print("=" * 60)
    dist_print("Phase 1: Adapter-Only Feature Distillation")
    dist_print("=" * 60)

    # Build teacher model (full SAM3 — we only use its backbone)
    # Each rank loads independently; weights are identical across ranks.
    dist_print("\nLoading teacher SAM3 model...")
    teacher = build_sam3_image_model(
        device=device,
        checkpoint_path=args.checkpoint,
        eval_mode=True,
        load_from_HF=args.checkpoint is None,
        enable_inst_interactivity=False,
    )
    teacher.eval()

    # Build student backbone (same init on all ranks — timm pretrained)
    dist_print(f"\nBuilding student backbone: {args.backbone}")
    student_bb = build_student_backbone(
        config_name=args.backbone,
        pretrained=True,
        freeze_backbone=True,
    )
    student_bb = student_bb.to(device)
    dist_print(f"  Trainable adapter params: {student_bb.trainable_params:,}")

    # Train
    trainer = DistillationTrainer(
        teacher_model=teacher,
        student_backbone=student_bb,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        lr=args.lr,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        num_workers=args.num_workers,
        save_every=args.save_every,
        log_every=args.log_every,
        device=device,
    )

    # Free teacher model components we don't need (save VRAM)
    del teacher.transformer
    del teacher.dot_prod_scoring
    del teacher.segmentation_head
    del teacher.geometry_encoder
    del teacher.backbone.language_backbone
    torch.cuda.empty_cache()
    dist_print(f"\nFreed non-backbone teacher components to save VRAM")

    trainer.train()


def run_phase2(args):
    """Phase 2: Fine-tune student backbone (full or LoRA) + FPN adapter."""
    from sam3.model_builder import build_sam3_image_model
    from sam3.distillation.student_backbone import build_student_backbone
    from sam3.distillation.distill_trainer import (
        DistillationTrainer,
        dist_print,
        setup_distributed,
    )

    setup_distributed()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = f"cuda:{local_rank}" if torch.distributed.is_initialized() else args.device

    dist_print("=" * 60)
    dist_print("Phase 2: Student Backbone Fine-Tuning")
    dist_print("=" * 60)

    # Build teacher model (full SAM3 — we only use its backbone)
    dist_print("\nLoading teacher SAM3 model...")
    teacher = build_sam3_image_model(
        device=device,
        checkpoint_path=args.checkpoint,
        eval_mode=True,
        load_from_HF=args.checkpoint is None,
        enable_inst_interactivity=False,
    )
    teacher.eval()

    use_lora = args.lora_rank > 0

    # Build student backbone — frozen if using LoRA, unfrozen otherwise
    dist_print(f"\nBuilding student backbone: {args.backbone}")
    student_bb = build_student_backbone(
        config_name=args.backbone,
        pretrained=True,
        freeze_backbone=use_lora,
    )

    # Apply LoRA to the timm backbone (not the FPN adapters)
    if use_lora:
        from sam3.distillation.lora import apply_lora, lora_param_count
        n_wrapped = apply_lora(
            student_bb.backbone, rank=args.lora_rank, alpha=args.lora_rank
        )
        dist_print(
            f"  LoRA rank={args.lora_rank}: wrapped {n_wrapped} layers, "
            f"{lora_param_count(student_bb):,} LoRA params"
        )
    else:
        dist_print("  Full fine-tuning (no LoRA)")

    student_bb = student_bb.to(device)

    # Load adapter weights from phase 1
    if args.adapter_checkpoint:
        dist_print(f"Loading phase 1 weights from {args.adapter_checkpoint}")
        ckpt = torch.load(args.adapter_checkpoint, map_location=device)
        student_bb.load_state_dict(ckpt["student_state_dict"], strict=False)
    else:
        dist_print("WARNING: No --adapter-checkpoint provided. Starting from scratch.")

    total_params = sum(p.numel() for p in student_bb.parameters())
    trainable = sum(p.numel() for p in student_bb.parameters() if p.requires_grad)
    dist_print(f"  Total params: {total_params:,}")
    dist_print(f"  Trainable params: {trainable:,}")

    # Train with same DistillationTrainer
    trainer = DistillationTrainer(
        teacher_model=teacher,
        student_backbone=student_bb,
        data_dir=args.data_dir,
        output_dir=args.output_dir + "_phase2",
        lr=args.lr,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        num_workers=args.num_workers,
        save_every=args.save_every,
        log_every=args.log_every,
        device=device,
    )

    # Free teacher model components we don't need (save VRAM)
    del teacher.transformer
    del teacher.dot_prod_scoring
    del teacher.segmentation_head
    del teacher.geometry_encoder
    del teacher.backbone.language_backbone
    torch.cuda.empty_cache()
    dist_print(f"\nFreed non-backbone teacher components to save VRAM")

    trainer.train()


def run_test(args):
    """Test student model with a single image + text prompt (single GPU only)."""
    from sam3.distillation.sam3_student import build_sam3_student_model
    from sam3.model.sam3_image_processor import Sam3Processor
    from PIL import Image

    print("=" * 60)
    print("Testing Student Model")
    print("=" * 60)

    # Build student model
    print(f"\nBuilding student model with {args.backbone} backbone...")
    model = build_sam3_student_model(
        backbone_config=args.backbone,
        teacher_checkpoint=args.checkpoint,
        load_from_HF=args.checkpoint is None,
        device=args.device,
        freeze_teacher=True,
    )

    # Load adapter weights
    if args.adapter_checkpoint:
        print(f"Loading adapter weights from {args.adapter_checkpoint}")
        ckpt = torch.load(args.adapter_checkpoint, map_location=args.device)
        model.backbone.student_backbone.load_state_dict(
            ckpt["student_state_dict"]
        )

    model.eval()

    # Create processor
    processor = Sam3Processor(model, device=args.device)

    # Run inference
    image = Image.open(args.test_image)
    print(f"Image: {args.test_image} ({image.size[0]}x{image.size[1]})")
    print(f"Prompt: '{args.test_prompt}'")

    t0 = time.perf_counter()
    state = processor.set_image(image)
    t_backbone = time.perf_counter() - t0

    t0 = time.perf_counter()
    state = processor.set_text_prompt(args.test_prompt, state)
    t_head = time.perf_counter() - t0

    num_dets = len(state["scores"])
    print(f"\nResults: {num_dets} detections")
    print(f"  Backbone: {t_backbone*1000:.1f}ms")
    print(f"  Head: {t_head*1000:.1f}ms")
    print(f"  Total: {(t_backbone+t_head)*1000:.1f}ms")

    for i in range(min(num_dets, 10)):
        score = state["scores"][i].item()
        box = state["boxes"][i].tolist()
        print(
            f"  [{i}] score={score:.3f} "
            f"box=[{box[0]:.0f},{box[1]:.0f},{box[2]:.0f},{box[3]:.0f}]"
        )


def main():
    parser = argparse.ArgumentParser(description="SAM3 backbone distillation")

    parser.add_argument(
        "--phase", type=int, default=1, choices=[1, 2],
        help="Training phase: 1=adapter-only, 2=encoder fine-tuning"
    )
    parser.add_argument(
        "--data-dir", type=str, default=None,
        help="Path to image directory (e.g., COCO train2017)"
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to teacher SAM3 checkpoint (default: download from HF)"
    )
    parser.add_argument(
        "--adapter-checkpoint", type=str, default=None,
        help="Path to adapter checkpoint (for phase 2 or testing)"
    )
    parser.add_argument(
        "--backbone", type=str, default="efficientvit_l1",
        choices=["efficientvit_l1", "efficientvit_l2", "repvit_m2_3", "tiny_vit_21m"],
        help="Student backbone architecture"
    )
    parser.add_argument("--output-dir", type=str, default="distill_checkpoints")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument(
        "--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu"
    )

    # LoRA (phase 2 only)
    parser.add_argument(
        "--lora-rank", type=int, default=0,
        help="LoRA rank for phase 2 backbone fine-tuning (0=full fine-tune)"
    )

    # Test mode
    parser.add_argument("--test-image", type=str, default=None)
    parser.add_argument("--test-prompt", type=str, default="object")

    args = parser.parse_args()

    if args.test_image:
        run_test(args)
    elif args.data_dir is None:
        parser.print_help()
        print("\nProvide --data-dir for training or --test-image for testing.")
    elif args.phase == 1:
        run_phase1(args)
    elif args.phase == 2:
        run_phase2(args)


if __name__ == "__main__":
    main()
