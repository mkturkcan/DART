#!/usr/bin/env python3
"""
Calibrate class prototypes and optionally fine-tune hs_proj for
single-pass multi-class inference.

Two modes:

  1. **prototype** — Run the GT sequential predictor on images, collect
     hs_proj features grouped by true class, compute L2-normalized
     centroids.  Saves a .pt file that can be loaded by
     Sam3MultiClassPredictorFast(class_method="prototype").

  2. **finetune** — Fine-tune only hs_proj (~65K params) with cross-entropy
     loss on cosine similarity scores against per-class text embeddings.
     Training data comes from the GT predictor's query-class associations.

Usage:
    # Collect prototypes (zero extra training):
    python scripts/calibrate_single_pass.py prototype \
        --images-dir /path/to/images \
        --classes "car" "pedestrian" "bicycle" \
        --output prototypes.pt \
        --max-images 200

    # Fine-tune hs_proj (lightweight training, ~5 min on GPU):
    python scripts/calibrate_single_pass.py finetune \
        --images-dir /path/to/images \
        --classes "car" "pedestrian" "bicycle" \
        --output hs_proj_finetuned.pt \
        --max-images 500 --epochs 20 --lr 1e-3
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F
from PIL import Image

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_multiclass import Sam3MultiClassPredictor
from sam3.model.sam3_multiclass_fast import Sam3MultiClassPredictorFast


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def find_images(images_dir: str, max_images: int) -> List[str]:
    paths = []
    for fname in sorted(os.listdir(images_dir)):
        if Path(fname).suffix.lower() in IMAGE_EXTENSIONS:
            paths.append(os.path.join(images_dir, fname))
            if len(paths) >= max_images:
                break
    return paths


# --------------------------------------------------------------------------
# Mode 1: Collect prototypes
# --------------------------------------------------------------------------


@torch.inference_mode()
def collect_prototypes(
    model,
    class_names: List[str],
    image_paths: List[str],
    device: str,
    confidence: float = 0.3,
    nms: float = 0.7,
) -> Dict:
    """Run GT predictor, project query features, group by class, compute centroids."""

    # GT predictor (sequential, per-class)
    gt_pred = Sam3MultiClassPredictor(model, device=device)
    gt_pred.set_classes(class_names)

    # Single-pass predictor (to get concatenated encoding + decoder output)
    sp_pred = Sam3MultiClassPredictorFast(
        model, device=device, use_fp16=True, single_pass=True,
    )
    sp_pred.set_classes(class_names)

    scoring = model.dot_prod_scoring
    N = len(class_names)

    # Accumulators: list of (d_proj,) tensors per class
    class_features = {i: [] for i in range(N)}
    total_dets = 0

    n_images = len(image_paths)
    print(f"Collecting prototypes from {n_images} images...")

    for img_idx, img_path in enumerate(image_paths):
        image = Image.open(img_path).convert("RGB")

        if (img_idx + 1) % max(n_images // 20, 1) == 0 or img_idx == 0:
            print(f"  [{img_idx + 1}/{n_images}] {os.path.basename(img_path)}")

        # --- GT predictions: get class labels ---
        gt_state = gt_pred.set_image(image)
        gt_results = gt_pred.predict(
            gt_state, confidence_threshold=confidence, nms_threshold=nms,
            per_class_nms=False,
        )

        n_det = len(gt_results["scores"])
        if n_det == 0:
            continue

        # --- Single-pass: get decoder hidden states ---
        # We need the hs_proj features from the single-pass decoder.
        # Run the single-pass encoder+decoder manually to get hs.
        sp_state = sp_pred.set_image(image)
        sp_pred._ensure_compiled()

        backbone_out = sp_state["backbone_out"]
        img_ids = torch.tensor([0], device=device, dtype=torch.long)
        backbone_out_proc, img_feats, img_pos_embeds, vis_feat_sizes = (
            model._get_img_feats(backbone_out, img_ids)
        )

        prompt = sp_pred._concat_text
        prompt_mask = sp_pred._concat_mask

        with torch.autocast("cuda", dtype=torch.float16, enabled=True):
            prompt_pos = torch.zeros_like(prompt)
            memory = sp_pred._encoder_fn(
                src=[f.clone() for f in img_feats],
                src_key_padding_mask=None,
                src_pos=[p.clone() for p in img_pos_embeds],
                prompt=prompt,
                prompt_pos=prompt_pos,
                prompt_key_padding_mask=prompt_mask,
                feat_sizes=vis_feat_sizes,
            )
            enc_hs = memory["memory"]

            query_embed = model.transformer.decoder.query_embed.weight
            tgt = query_embed.unsqueeze(1)

            hs, reference_boxes, _, _ = sp_pred._decoder_fn(
                tgt=tgt,
                memory=enc_hs,
                memory_key_padding_mask=memory["padding_mask"],
                pos=memory["pos_embed"],
                reference_boxes=None,
                level_start_index=memory["level_start_index"],
                spatial_shapes=memory["spatial_shapes"],
                valid_ratios=memory["valid_ratios"],
                tgt_mask=None,
                memory_text=prompt,
                text_attention_mask=prompt_mask,
                apply_dac=False,
            )
            # hs: (num_layers, Q, 1, d) → last layer: (Q, d)
            hs_last = hs[-1, :, 0, :]

        # Project to d_proj space
        proj_hs = scoring.hs_proj(hs_last.float())  # (Q, d_proj)

        # Match single-pass queries to GT detections via box IoU
        # Get single-pass boxes
        hs_t = hs.transpose(1, 2)  # (num_layers, 1, Q, d)
        reference_boxes_t = reference_boxes.transpose(1, 2)
        from sam3.model.model_misc import inverse_sigmoid
        box_offsets = model.transformer.decoder.bbox_embed(hs_t)
        ref_inv = inverse_sigmoid(reference_boxes_t)
        sp_boxes = (ref_inv + box_offsets).sigmoid()[-1, 0]  # (Q, 4) cxcywh

        det_scores = model.dot_prod_scoring(hs_t, prompt, prompt_mask)
        det_probs = det_scores[-1, 0, :, 0].float().sigmoid()  # (Q,)

        # Get GT boxes in cxcywh normalized form
        from sam3.model.box_ops import box_cxcywh_to_xyxy
        orig_h, orig_w = sp_state["original_height"], sp_state["original_width"]
        scale = torch.tensor([orig_w, orig_h, orig_w, orig_h],
                             device=device, dtype=torch.float32)

        # Convert GT boxes to normalized cxcywh
        gt_boxes_xyxy = gt_results["boxes"]  # (n_det, 4) in pixel coords
        gt_boxes_norm = gt_boxes_xyxy / scale  # normalize
        # xyxy to cxcywh
        gt_cx = (gt_boxes_norm[:, 0] + gt_boxes_norm[:, 2]) / 2
        gt_cy = (gt_boxes_norm[:, 1] + gt_boxes_norm[:, 3]) / 2
        gt_w = gt_boxes_norm[:, 2] - gt_boxes_norm[:, 0]
        gt_h = gt_boxes_norm[:, 3] - gt_boxes_norm[:, 1]
        gt_boxes_cxcywh = torch.stack([gt_cx, gt_cy, gt_w, gt_h], dim=-1)

        # Simple box IoU matching
        # For each GT det, find the best single-pass query
        for det_i in range(n_det):
            class_id = gt_results["class_ids"][det_i].item()
            gt_box = gt_boxes_cxcywh[det_i]

            # Box L1 distance to all queries
            dists = (sp_boxes - gt_box.unsqueeze(0)).abs().sum(dim=-1)  # (Q,)
            # Weight by detection confidence to prefer high-conf queries
            weighted_dists = dists / (det_probs + 1e-8)
            best_q = weighted_dists.argmin().item()

            feat = F.normalize(proj_hs[best_q].detach(), dim=-1)
            class_features[class_id].append(feat)
            total_dets += 1

    # Compute centroids
    print(f"\nTotal detections collected: {total_dets}")
    prototypes = []
    for i in range(N):
        feats = class_features[i]
        n = len(feats)
        print(f"  {class_names[i]}: {n} detections")
        if n == 0:
            print(f"    WARNING: No detections for class '{class_names[i]}'. "
                  f"Using text embedding as fallback.")
            prototypes.append(sp_pred._class_proj_norm[i])
        else:
            stacked = torch.stack(feats)  # (n, d_proj)
            centroid = F.normalize(stacked.mean(dim=0), dim=-1)
            prototypes.append(centroid)

    prototypes = torch.stack(prototypes)  # (N, d_proj)
    return {
        "class_names": class_names,
        "prototypes": prototypes.cpu(),
        "n_samples": {class_names[i]: len(class_features[i]) for i in range(N)},
    }


# --------------------------------------------------------------------------
# Mode 2: Fine-tune hs_proj
# --------------------------------------------------------------------------


def finetune_hs_proj(
    model,
    class_names: List[str],
    image_paths: List[str],
    device: str,
    epochs: int = 20,
    lr: float = 1e-3,
    confidence: float = 0.3,
    nms: float = 0.7,
    temperature: float = 0.07,
) -> Dict:
    """Fine-tune hs_proj with cross-entropy loss on per-class cosine similarity.

    Collects (query_feature, class_id) pairs from GT predictor + single-pass
    decoder, then trains hs_proj to maximize cosine similarity with the correct
    class text embedding.
    """

    # First collect training data: (hs_features, class_ids)
    print("Phase 1: Collecting training data from GT predictor...")

    gt_pred = Sam3MultiClassPredictor(model, device=device)
    gt_pred.set_classes(class_names)

    sp_pred = Sam3MultiClassPredictorFast(
        model, device=device, use_fp16=True, single_pass=True,
    )
    sp_pred.set_classes(class_names)

    scoring = model.dot_prod_scoring
    N = len(class_names)

    all_hs_feats = []   # raw hs (before hs_proj), (d_model,)
    all_class_ids = []   # int class id

    n_images = len(image_paths)
    with torch.inference_mode():
        for img_idx, img_path in enumerate(image_paths):
            image = Image.open(img_path).convert("RGB")

            if (img_idx + 1) % max(n_images // 20, 1) == 0 or img_idx == 0:
                print(f"  [{img_idx + 1}/{n_images}] {os.path.basename(img_path)}")

            # GT
            gt_state = gt_pred.set_image(image)
            gt_results = gt_pred.predict(
                gt_state, confidence_threshold=confidence, nms_threshold=nms,
                per_class_nms=False,
            )
            n_det = len(gt_results["scores"])
            if n_det == 0:
                continue

            # Single-pass decoder
            sp_state = sp_pred.set_image(image)
            sp_pred._ensure_compiled()

            backbone_out = sp_state["backbone_out"]
            img_ids = torch.tensor([0], device=device, dtype=torch.long)
            backbone_out_proc, img_feats, img_pos_embeds, vis_feat_sizes = (
                model._get_img_feats(backbone_out, img_ids)
            )

            prompt = sp_pred._concat_text
            prompt_mask = sp_pred._concat_mask

            with torch.autocast("cuda", dtype=torch.float16, enabled=True):
                prompt_pos = torch.zeros_like(prompt)
                memory = sp_pred._encoder_fn(
                    src=[f.clone() for f in img_feats],
                    src_key_padding_mask=None,
                    src_pos=[p.clone() for p in img_pos_embeds],
                    prompt=prompt,
                    prompt_pos=prompt_pos,
                    prompt_key_padding_mask=prompt_mask,
                    feat_sizes=vis_feat_sizes,
                )
                enc_hs = memory["memory"]
                query_embed = model.transformer.decoder.query_embed.weight
                tgt = query_embed.unsqueeze(1)

                hs, reference_boxes, _, _ = sp_pred._decoder_fn(
                    tgt=tgt,
                    memory=enc_hs,
                    memory_key_padding_mask=memory["padding_mask"],
                    pos=memory["pos_embed"],
                    reference_boxes=None,
                    level_start_index=memory["level_start_index"],
                    spatial_shapes=memory["spatial_shapes"],
                    valid_ratios=memory["valid_ratios"],
                    tgt_mask=None,
                    memory_text=prompt,
                    text_attention_mask=prompt_mask,
                    apply_dac=False,
                )
                hs_last = hs[-1, :, 0, :]  # (Q, d_model)

            # Match to GT (same as prototype collection)
            hs_t = hs.transpose(1, 2)
            reference_boxes_t = reference_boxes.transpose(1, 2)
            from sam3.model.model_misc import inverse_sigmoid
            box_offsets = model.transformer.decoder.bbox_embed(hs_t)
            ref_inv = inverse_sigmoid(reference_boxes_t)
            sp_boxes = (ref_inv + box_offsets).sigmoid()[-1, 0]

            det_scores = model.dot_prod_scoring(hs_t, prompt, prompt_mask)
            det_probs = det_scores[-1, 0, :, 0].float().sigmoid()

            from sam3.model.box_ops import box_cxcywh_to_xyxy
            orig_h, orig_w = sp_state["original_height"], sp_state["original_width"]
            scale = torch.tensor([orig_w, orig_h, orig_w, orig_h],
                                 device=device, dtype=torch.float32)
            gt_boxes_xyxy = gt_results["boxes"]
            gt_boxes_norm = gt_boxes_xyxy / scale
            gt_cx = (gt_boxes_norm[:, 0] + gt_boxes_norm[:, 2]) / 2
            gt_cy = (gt_boxes_norm[:, 1] + gt_boxes_norm[:, 3]) / 2
            gt_w = gt_boxes_norm[:, 2] - gt_boxes_norm[:, 0]
            gt_h = gt_boxes_norm[:, 3] - gt_boxes_norm[:, 1]
            gt_boxes_cxcywh = torch.stack([gt_cx, gt_cy, gt_w, gt_h], dim=-1)

            for det_i in range(n_det):
                class_id = gt_results["class_ids"][det_i].item()
                gt_box = gt_boxes_cxcywh[det_i]
                dists = (sp_boxes - gt_box.unsqueeze(0)).abs().sum(dim=-1)
                weighted_dists = dists / (det_probs + 1e-8)
                best_q = weighted_dists.argmin().item()

                all_hs_feats.append(hs_last[best_q].detach().float().cpu())
                all_class_ids.append(class_id)

    if not all_hs_feats:
        print("ERROR: No training data collected. Check images and class names.")
        sys.exit(1)

    X = torch.stack(all_hs_feats).to(device)  # (M, d_model)
    y = torch.tensor(all_class_ids, device=device, dtype=torch.long)  # (M,)
    print(f"\nCollected {len(X)} training samples across {N} classes")
    for i in range(N):
        print(f"  {class_names[i]}: {(y == i).sum().item()} samples")

    # Phase 2: Fine-tune hs_proj
    print(f"\nPhase 2: Fine-tuning hs_proj ({epochs} epochs, lr={lr})...")

    # Get per-class text embeddings (targets)
    with torch.inference_mode():
        per_class_proj = []
        for i in range(N):
            class_text = sp_pred._batched_text[:, i:i+1, :]
            class_mask = sp_pred._batched_mask[i:i+1, :]
            text_in = class_text
            if scoring.prompt_mlp is not None:
                text_in = scoring.prompt_mlp(text_in)
            pooled = scoring.mean_pool_text(text_in, class_mask)
            proj = scoring.prompt_proj(pooled)
            per_class_proj.append(proj)
        class_embeds = F.normalize(
            torch.cat(per_class_proj, dim=0), dim=-1
        )  # (N, d_proj)

    # Only train hs_proj
    hs_proj = scoring.hs_proj
    hs_proj.train()
    optimizer = torch.optim.AdamW(hs_proj.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Freeze everything else
    for param in model.parameters():
        param.requires_grad = False
    for param in hs_proj.parameters():
        param.requires_grad = True

    n_params = sum(p.numel() for p in hs_proj.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {n_params:,}")

    batch_size = min(512, len(X))
    best_acc = 0.0
    best_state = None

    for epoch in range(epochs):
        # Shuffle
        perm = torch.randperm(len(X), device=device)
        X_shuf = X[perm]
        y_shuf = y[perm]

        epoch_loss = 0.0
        epoch_correct = 0
        n_batches = 0

        for start in range(0, len(X), batch_size):
            end = min(start + batch_size, len(X))
            x_batch = X_shuf[start:end]
            y_batch = y_shuf[start:end]

            proj = scoring.hs_proj(x_batch)  # (B, d_proj)
            proj_norm = F.normalize(proj, dim=-1)

            # Cosine similarity logits
            logits = proj_norm @ class_embeds.T / temperature  # (B, N)

            loss = F.cross_entropy(logits, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_correct += (logits.argmax(dim=-1) == y_batch).sum().item()
            n_batches += 1

        scheduler.step()
        acc = epoch_correct / len(X)
        avg_loss = epoch_loss / n_batches
        lr_now = scheduler.get_last_lr()[0]

        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.cpu().clone() for k, v in hs_proj.state_dict().items()}

        print(f"  Epoch {epoch + 1:3d}/{epochs}  "
              f"loss={avg_loss:.4f}  acc={acc:.3f}  lr={lr_now:.2e}")

    # Restore best
    hs_proj.load_state_dict(best_state)
    hs_proj.eval()

    # Re-freeze
    for param in hs_proj.parameters():
        param.requires_grad = False

    print(f"\nBest accuracy: {best_acc:.3f}")

    return {
        "class_names": class_names,
        "hs_proj_state_dict": best_state,
        "best_accuracy": best_acc,
        "n_samples": len(X),
    }


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Calibrate single-pass class assignment for SAM3"
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    # Shared args
    def add_common_args(p):
        p.add_argument("--images-dir", type=str, required=True)
        p.add_argument("--classes", nargs="+", type=str,
                        default=["car", "pedestrian", "bicycle"])
        p.add_argument("--max-images", type=int, default=200)
        p.add_argument("--confidence", type=float, default=0.3)
        p.add_argument("--nms", type=float, default=0.7)
        p.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
        p.add_argument("--checkpoint", type=str, default=None)
        p.add_argument("--output", "-o", type=str, required=True,
                        help="Output .pt file path")

    # Prototype mode
    proto_parser = subparsers.add_parser("prototype",
                                          help="Collect calibrated class prototypes")
    add_common_args(proto_parser)

    # Finetune mode
    ft_parser = subparsers.add_parser("finetune",
                                       help="Fine-tune hs_proj scoring layer")
    add_common_args(ft_parser)
    ft_parser.add_argument("--epochs", type=int, default=20)
    ft_parser.add_argument("--lr", type=float, default=1e-3)
    ft_parser.add_argument("--temperature", type=float, default=0.07,
                            help="Temperature for contrastive loss")

    args = parser.parse_args()

    # Find images
    image_paths = find_images(args.images_dir, args.max_images)
    if not image_paths:
        print(f"No images found in {args.images_dir}")
        sys.exit(1)

    print(f"Images: {len(image_paths)} (from {args.images_dir})")
    print(f"Classes: {args.classes}")
    print()

    # Load model
    print(f"Loading SAM3 model on {args.device}...")
    model = build_sam3_image_model(
        device=args.device,
        checkpoint_path=args.checkpoint,
        eval_mode=True,
    )

    t0 = time.perf_counter()

    if args.mode == "prototype":
        result = collect_prototypes(
            model=model,
            class_names=args.classes,
            image_paths=image_paths,
            device=args.device,
            confidence=args.confidence,
            nms=args.nms,
        )
    elif args.mode == "finetune":
        result = finetune_hs_proj(
            model=model,
            class_names=args.classes,
            image_paths=image_paths,
            device=args.device,
            epochs=args.epochs,
            lr=args.lr,
            confidence=args.confidence,
            nms=args.nms,
            temperature=args.temperature,
        )

    elapsed = time.perf_counter() - t0
    print(f"\nCompleted in {elapsed:.1f}s")

    torch.save(result, args.output)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
