#!/usr/bin/env python3
"""Training-free block pruning search for SAM3 ViT-H backbone.

Adapts the BlockPruner algorithm (arXiv:2406.10594) to vision transformers.
Each ViT block is decomposed into two sub-blocks (attention + MLP). A greedy
iterative search removes the sub-block whose removal causes the least feature
reconstruction loss, measured as L2 distance between original and pruned
backbone outputs on a calibration image set.

Global attention blocks (7, 15, 23, 31) can optionally be protected from
pruning since they provide the only cross-window information flow.

Usage:
    python scripts/block_pruner_search.py \\
        --checkpoint sam3.pt \\
        --calib-dir train2017 \\
        --num-images 32 \\
        --num-prune 16 \\
        --imgsz 1008

Output: Ordered pruning sequence with cumulative quality loss at each step.
"""

import argparse
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision.transforms import v2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def load_calibration_images(calib_dir, num_images, imgsz, device):
    """Load and preprocess calibration images."""
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    all_paths = sorted(
        p for p in Path(calib_dir).iterdir()
        if p.suffix.lower() in exts
    )
    if len(all_paths) < num_images:
        print(f"WARNING: Only {len(all_paths)} images found, using all")
        num_images = len(all_paths)

    # Sample evenly across dataset
    step = max(1, len(all_paths) // num_images)
    paths = all_paths[:num_images * step:step][:num_images]

    transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize(imgsz, max_size=imgsz + 1),
        v2.CenterCrop(imgsz),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    tensors = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        tensors.append(transform(img).to(device))

    return torch.stack(tensors)


def masked_block_forward(block, x, mask_attn=False, mask_mlp=False):
    """Run a single ViT block with optional attention/MLP masking.

    When mask_attn=True: skip attention, pass through via residual.
    When mask_mlp=True: skip MLP, pass through via residual.
    """
    from sam3.model.vitdet import window_partition, window_unpartition

    # Attention branch
    if not mask_attn:
        shortcut = x
        xn = block.norm1(x)
        if block.window_size > 0:
            H, W = xn.shape[1], xn.shape[2]
            xn, pad_hw = window_partition(xn, block.window_size)
        xn = block.ls1(block.attn(xn))
        if block.window_size > 0:
            xn = window_unpartition(xn, block.window_size, pad_hw, (H, W))
        x = shortcut + block.dropout(block.drop_path(xn))

    # MLP branch
    if not mask_mlp:
        x = x + block.dropout(
            block.drop_path(block.ls2(block.mlp(block.norm2(x))))
        )

    return x


@torch.no_grad()
def run_trunk_with_masks(trunk, images, masks, batch_size=4):
    """Run ViT trunk with per-block attention/MLP masks.

    Args:
        trunk: ViT module
        images: (N, 3, H, W) calibration images
        masks: dict mapping block_idx -> set of {"attn", "mlp"}
        batch_size: sub-batch size for memory efficiency

    Returns:
        (N, C, h, w) trunk output features
    """
    all_features = []
    for i in range(0, len(images), batch_size):
        batch = images[i:i + batch_size]
        with torch.autocast("cuda", dtype=torch.float16):
            x = trunk.patch_embed(batch)
            h, w = x.shape[1], x.shape[2]

            s = 0
            if trunk.retain_cls_token:
                x = torch.cat([trunk.class_embedding, x.flatten(1, 2)], dim=1)
                s = 1

            if trunk.pos_embed is not None:
                from sam3.model.vitdet import get_abs_pos  # noqa: E402
                x = x + get_abs_pos(
                    trunk.pos_embed,
                    trunk.pretrain_use_cls_token,
                    (h, w),
                    trunk.retain_cls_token,
                    tiling=trunk.tile_abs_pos,
                )

            x = trunk.ln_pre(x)

            for bi, blk in enumerate(trunk.blocks):
                block_masks = masks.get(bi, set())
                x = masked_block_forward(
                    blk, x,
                    mask_attn="attn" in block_masks,
                    mask_mlp="mlp" in block_masks,
                )

                # Extract features at last global attn block
                if bi == trunk.full_attn_ids[-1]:
                    x = trunk.ln_post(x)
                    feats = x[:, s:]
                    if feats.ndim == 4:
                        feats = feats.permute(0, 3, 1, 2)
                    else:
                        fh = fw = int(math.sqrt(feats.shape[1]))
                        feats = feats.reshape(
                            feats.shape[0], fh, fw, feats.shape[-1]
                        ).permute(0, 3, 1, 2)
                    break

            all_features.append(feats.float())

    return torch.cat(all_features, dim=0)


def compute_loss(ref, pruned):
    """Normalized L2 reconstruction loss."""
    return (ref - pruned).pow(2).mean().item()


def run_search(
    checkpoint_path,
    calib_dir,
    num_images=32,
    num_prune=16,
    imgsz=1008,
    batch_size=4,
    protect_global=True,
    block_type="mix",
    device="cuda",
):
    """Run greedy iterative block pruning search."""
    from sam3.model_builder import build_sam3_image_model

    print(f"Loading SAM3 model from {checkpoint_path} ...")
    model = build_sam3_image_model(
        checkpoint_path=checkpoint_path,
        device=device,
        eval_mode=True,
        load_from_HF=False,
        enable_segmentation=False,
    )

    trunk = model.backbone.vision_backbone.trunk
    num_blocks = len(trunk.blocks)
    global_ids = set(trunk.full_attn_ids)

    print(f"\nViT-H backbone: {num_blocks} blocks")
    print(f"  Global attention blocks: {sorted(global_ids)}")
    print(f"  Window attention blocks: "
          f"{sorted(set(range(num_blocks)) - global_ids)}")
    print(f"  Protect global blocks: {protect_global}")
    print(f"  Sub-block types: {block_type}")
    print(f"  Resolution: {imgsz}px")

    # Build candidate list
    candidates = []
    for i in range(num_blocks):
        if protect_global and i in global_ids:
            continue
        if block_type in ("attn", "mix"):
            candidates.append((i, "attn"))
        if block_type in ("mlp", "mix"):
            candidates.append((i, "mlp"))

    print(f"  Candidate sub-blocks: {len(candidates)}")
    print(f"  Sub-blocks to prune: {num_prune}")

    # Load calibration images
    print(f"\nLoading {num_images} calibration images from {calib_dir} ...")
    images = load_calibration_images(calib_dir, num_images, imgsz, device)
    print(f"  Loaded: {images.shape}")

    # Reference features (full model, no masks)
    print("Computing reference features ...")
    ref_features = run_trunk_with_masks(trunk, images, {}, batch_size)
    print(f"  Shape: {ref_features.shape}, "
          f"norm: {ref_features.norm().item():.1f}")

    # Greedy iterative search
    pruned_masks = {}  # block_idx -> set of pruned sub-types
    pruned_order = []  # list of (block_idx, sub_type, step_loss, cumul_loss)
    remaining = list(range(len(candidates)))

    header = (f"{'Step':>4}  {'Block':>5}  {'Type':>4}  {'Win/Glb':>7}  "
              f"{'StepLoss':>10}  {'CumulLoss':>10}  {'Time':>6}")
    print(f"\n{'='*len(header)}")
    print(header)
    print(f"{'='*len(header)}")

    for step in range(min(num_prune, len(remaining))):
        best_loss = float("inf")
        best_cand = None
        t0 = time.perf_counter()

        for cand_idx in remaining:
            block_idx, sub_type = candidates[cand_idx]

            # Build trial mask: existing pruned + this candidate
            trial_masks = {}
            for bi, types in pruned_masks.items():
                trial_masks[bi] = set(types)
            trial_masks.setdefault(block_idx, set()).add(sub_type)

            # Evaluate
            pruned_features = run_trunk_with_masks(
                trunk, images, trial_masks, batch_size
            )
            loss = compute_loss(ref_features, pruned_features)

            if loss < best_loss:
                best_loss = loss
                best_cand = cand_idx

        # Commit best candidate
        remaining.remove(best_cand)
        block_idx, sub_type = candidates[best_cand]
        pruned_masks.setdefault(block_idx, set()).add(sub_type)

        # Compute cumulative loss
        cumul_features = run_trunk_with_masks(
            trunk, images, pruned_masks, batch_size
        )
        cumul_loss = compute_loss(ref_features, cumul_features)

        is_global = "GLOBAL" if block_idx in global_ids else "window"
        elapsed = time.perf_counter() - t0
        pruned_order.append((block_idx, sub_type, best_loss, cumul_loss))

        print(f"{step+1:>4}  {block_idx:>5}  {sub_type:>4}  {is_global:>7}  "
              f"{best_loss:>10.6f}  {cumul_loss:>10.6f}  {elapsed:>5.1f}s")

    # --- Summary ---
    print(f"\n{'='*60}")
    print("PRUNING ORDER (least important first):")
    print(f"{'='*60}")

    attn_pruned = []
    mlp_pruned = []

    for rank, (block_idx, sub_type, _, cumul) in enumerate(pruned_order):
        is_global = "GLOBAL" if block_idx in global_ids else "window"
        print(f"  {rank+1:>2}. Block {block_idx:>2} {sub_type:>4} "
              f"({is_global})  cumul_loss={cumul:.6f}")
        if sub_type == "attn":
            attn_pruned.append(block_idx)
        else:
            mlp_pruned.append(block_idx)

    both_pruned = set(attn_pruned) & set(mlp_pruned)

    # --mask-blocks string (always output, most useful format)
    mask_parts = [f"{bi}:{st}" for bi, st, _, _ in pruned_order]
    print(f"\n--mask-blocks \"{','.join(mask_parts)}\"")

    print(f"\nAttention pruned: {sorted(attn_pruned)}")
    print(f"MLP pruned:       {sorted(mlp_pruned)}")
    if both_pruned:
        print(f"Full blocks (both attn+mlp): {sorted(both_pruned)}")
        print(f"  -> --skip-blocks {','.join(str(b) for b in sorted(both_pruned))}")

    # Speedup estimate (from profiling: MLP ~57%, attn ~20%, qkv+proj ~23%)
    # Per-block approximate times (compiled):
    #   Window attn: ~0.6ms, Window MLP: ~1.2ms
    #   Global attn: ~3.0ms, Global MLP: ~1.5ms
    saved_ms = 0
    for b in attn_pruned:
        saved_ms += 3.0 if b in global_ids else 0.6
    for b in mlp_pruned:
        saved_ms += 1.5 if b in global_ids else 1.2

    total_ms = 65  # approximate compiled backbone time
    print(f"\nEstimated savings: ~{saved_ms:.1f}ms of ~{total_ms}ms "
          f"({saved_ms/total_ms*100:.0f}%)")
    print(f"Estimated backbone: ~{total_ms - saved_ms:.1f}ms")

    # Quality comparison table
    print(f"\n{'='*60}")
    print("QUALITY vs SPEEDUP COMPARISON:")
    print(f"{'='*60}")
    print(f"{'Config':<35} {'Loss':>10}  {'Savings':>8}")
    print(f"{'-'*60}")
    for i, (bi, st, _, cl) in enumerate(pruned_order):
        n = i + 1
        # Compute savings for first n pruned sub-blocks
        s = 0
        for j in range(n):
            bj, tj = pruned_order[j][0], pruned_order[j][1]
            if tj == "attn":
                s += 3.0 if bj in global_ids else 0.6
            else:
                s += 1.5 if bj in global_ids else 1.2
        print(f"  Prune {n:>2} sub-blocks{'':<20} {cl:>10.6f}  {s:>6.1f}ms")

    return pruned_order


def main():
    parser = argparse.ArgumentParser(
        description="Training-free block pruning search for SAM3 ViT-H"
    )
    parser.add_argument(
        "--checkpoint", required=True,
        help="Path to SAM3 checkpoint (.pt)"
    )
    parser.add_argument(
        "--calib-dir", required=True,
        help="Directory with calibration images (e.g. train2017/)"
    )
    parser.add_argument(
        "--num-images", type=int, default=32,
        help="Number of calibration images (default 32)"
    )
    parser.add_argument(
        "--num-prune", type=int, default=16,
        help="Number of sub-blocks to prune (default 16)"
    )
    parser.add_argument(
        "--imgsz", type=int, default=1008,
        help="Input image resolution (default 1008)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=4,
        help="Batch size for calibration forward passes"
    )
    parser.add_argument(
        "--no-protect-global", action="store_true",
        help="Allow pruning global attention blocks (7, 15, 23, 31)"
    )
    parser.add_argument(
        "--block-type", choices=["attn", "mlp", "mix"], default="mix",
        help="Sub-block types to consider: attn, mlp, or mix (default)"
    )
    args = parser.parse_args()

    run_search(
        checkpoint_path=args.checkpoint,
        calib_dir=args.calib_dir,
        num_images=args.num_images,
        num_prune=args.num_prune,
        imgsz=args.imgsz,
        batch_size=args.batch_size,
        protect_global=not args.no_protect_global,
        block_type=args.block_type,
    )


if __name__ == "__main__":
    main()
