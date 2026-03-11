#!/usr/bin/env python3
"""
SAM3 Structural Pruning Analysis

Computes weight-based importance scores for all prunable components:
  1. Layer pruning   — Block Influence (BI) scores for encoder & decoder layers
  2. Head pruning    — Per-head importance in every attention module
  3. FFN pruning     — Per-neuron importance in every FFN
  4. Query pruning   — Redundancy analysis of 200 decoder query embeddings
  5. Seg head        — Component importance in segmentation head

Usage:
    python scripts/analyze_pruning.py --checkpoint /path/to/sam3.pt
    python scripts/analyze_pruning.py                # downloads from HF

Optional:
    --apply           Apply recommended pruning and save the pruned model
    --output pruned_sam3.pt
    --enc-layers 4    Keep top-K encoder layers  (default: auto-recommend)
    --dec-layers 4    Keep top-K decoder layers
    --heads 6         Keep top-K heads per module
    --ffn-dim 1536    Target FFN intermediate dim
    --queries 100     Keep top-K queries
"""

import argparse
import sys
from collections import defaultdict

import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────
#  Utilities
# ─────────────────────────────────────────────────────────────

def _extract_qkv(in_proj_weight, d_model):
    """Extract Q, K, V weight matrices from packed in_proj_weight."""
    assert in_proj_weight.shape == (3 * d_model, d_model)
    W_Q = in_proj_weight[:d_model]
    W_K = in_proj_weight[d_model : 2 * d_model]
    W_V = in_proj_weight[2 * d_model :]
    return W_Q, W_K, W_V


def _frobenius(tensor):
    return tensor.float().norm().item()


def _head_slices(d_model, n_heads):
    """Yield (start, end) index pairs for each head."""
    d_head = d_model // n_heads
    for h in range(n_heads):
        yield h * d_head, (h + 1) * d_head


# ─────────────────────────────────────────────────────────────
#  1. Layer Pruning — Block Influence Scores
# ─────────────────────────────────────────────────────────────

def compute_block_influence(attn_module, linear1, linear2, d_model):
    """
    BI_l = ||W_O @ W_V||_F  +  ||W_2 @ W_1||_F

    Measures how much each layer can change its input.
    Lower BI → more safely removable.
    """
    # Attention part
    W_O = attn_module.out_proj.weight.data  # (d, d)
    _, _, W_V = _extract_qkv(attn_module.in_proj_weight.data, d_model)
    bi_attn = _frobenius(W_O @ W_V)

    # FFN part
    W1 = linear1.weight.data  # (ffn_dim, d)
    W2 = linear2.weight.data  # (d, ffn_dim)
    bi_ffn = _frobenius(W2 @ W1)

    return bi_attn, bi_ffn, bi_attn + bi_ffn


def analyze_encoder_layers(encoder, d_model=256):
    """Compute BI scores for all encoder layers."""
    results = []
    for i, layer in enumerate(encoder.layers):
        # Self-attention BI
        sa_attn_bi, sa_ffn_bi, sa_total = compute_block_influence(
            layer.self_attn, layer.linear1, layer.linear2, d_model
        )
        # Cross-attention to text also contributes
        ca_W_O = layer.cross_attn_image.out_proj.weight.data
        _, _, ca_W_V = _extract_qkv(layer.cross_attn_image.in_proj_weight.data, d_model)
        ca_bi = _frobenius(ca_W_O @ ca_W_V)

        results.append({
            "layer": i,
            "self_attn_bi": sa_attn_bi,
            "cross_attn_bi": ca_bi,
            "ffn_bi": sa_ffn_bi,
            "total_bi": sa_total + ca_bi,
        })
    return results


def analyze_decoder_layers(decoder, d_model=256):
    """Compute BI scores for all decoder layers."""
    results = []
    for i, layer in enumerate(decoder.layers):
        # Self-attention
        sa_W_O = layer.self_attn.out_proj.weight.data
        _, _, sa_W_V = _extract_qkv(layer.self_attn.in_proj_weight.data, d_model)
        sa_bi = _frobenius(sa_W_O @ sa_W_V)

        # Cross-attention to image
        ca_W_O = layer.cross_attn.out_proj.weight.data
        _, _, ca_W_V = _extract_qkv(layer.cross_attn.in_proj_weight.data, d_model)
        ca_bi = _frobenius(ca_W_O @ ca_W_V)

        # Cross-attention to text
        cat_W_O = layer.ca_text.out_proj.weight.data
        _, _, cat_W_V = _extract_qkv(layer.ca_text.in_proj_weight.data, d_model)
        cat_bi = _frobenius(cat_W_O @ cat_W_V)

        # FFN
        W1 = layer.linear1.weight.data
        W2 = layer.linear2.weight.data
        ffn_bi = _frobenius(W2 @ W1)

        results.append({
            "layer": i,
            "self_attn_bi": sa_bi,
            "cross_attn_img_bi": ca_bi,
            "cross_attn_text_bi": cat_bi,
            "ffn_bi": ffn_bi,
            "total_bi": sa_bi + ca_bi + cat_bi + ffn_bi,
        })
    return results


# ─────────────────────────────────────────────────────────────
#  2. Head Pruning — Per-Head Importance
# ─────────────────────────────────────────────────────────────

def compute_head_importance(attn_module, d_model, n_heads):
    """
    I_h = ||W_O^h||_F * ||W_V^h||_F

    Higher I_h → more important head.
    """
    W_O = attn_module.out_proj.weight.data  # (d, d)
    _, _, W_V = _extract_qkv(attn_module.in_proj_weight.data, d_model)

    importances = []
    for h, (s, e) in enumerate(_head_slices(d_model, n_heads)):
        W_V_h = W_V[s:e, :]   # (d_head, d)
        W_O_h = W_O[:, s:e]   # (d, d_head)
        imp = _frobenius(W_O_h) * _frobenius(W_V_h)
        importances.append(imp)
    return importances


def analyze_all_heads(encoder, decoder, d_model=256, n_heads=8):
    """Compute head importance for every attention module."""
    results = {}

    # Encoder
    for i, layer in enumerate(encoder.layers):
        results[f"enc.{i}.self_attn"] = compute_head_importance(
            layer.self_attn, d_model, n_heads
        )
        results[f"enc.{i}.cross_attn"] = compute_head_importance(
            layer.cross_attn_image, d_model, n_heads
        )

    # Decoder
    for i, layer in enumerate(decoder.layers):
        results[f"dec.{i}.self_attn"] = compute_head_importance(
            layer.self_attn, d_model, n_heads
        )
        results[f"dec.{i}.cross_attn_img"] = compute_head_importance(
            layer.cross_attn, d_model, n_heads
        )
        results[f"dec.{i}.cross_attn_text"] = compute_head_importance(
            layer.ca_text, d_model, n_heads
        )

    return results


# ─────────────────────────────────────────────────────────────
#  3. FFN Width Pruning — Per-Neuron Importance
# ─────────────────────────────────────────────────────────────

def compute_neuron_importance(linear1, linear2):
    """
    I_i = ||W1[i, :]||_2  *  ||W2[:, i]||_2

    Ranks intermediate neurons by how much they affect output.
    """
    W1 = linear1.weight.data.float()  # (ffn_dim, d)
    W2 = linear2.weight.data.float()  # (d, ffn_dim)
    ffn_dim = W1.shape[0]

    row_norms = W1.norm(dim=1)        # (ffn_dim,)
    col_norms = W2.norm(dim=0)        # (ffn_dim,)
    importances = (row_norms * col_norms).tolist()
    return importances


def analyze_all_ffns(encoder, decoder):
    """Compute neuron importance for every FFN."""
    results = {}
    for i, layer in enumerate(encoder.layers):
        results[f"enc.{i}"] = compute_neuron_importance(layer.linear1, layer.linear2)
    for i, layer in enumerate(decoder.layers):
        results[f"dec.{i}"] = compute_neuron_importance(layer.linear1, layer.linear2)
    return results


# ─────────────────────────────────────────────────────────────
#  4. Query Pruning — Embedding Redundancy
# ─────────────────────────────────────────────────────────────

def analyze_query_redundancy(decoder, top_k_pairs=20):
    """
    Compute cosine similarity between all 200 query embeddings.
    High similarity → redundant queries that can be merged/pruned.
    """
    Q = decoder.query_embed.weight.data.float()  # (200, d)
    Q_norm = Q / Q.norm(dim=1, keepdim=True).clamp(min=1e-8)
    sim = Q_norm @ Q_norm.T  # (200, 200)

    # Zero out diagonal
    sim.fill_diagonal_(0)

    # Get top-k most similar pairs
    n = sim.shape[0]
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((i, j, sim[i, j].item()))
    pairs.sort(key=lambda x: -x[2])

    # Cluster analysis: how many queries have >0.9 cosine sim with another?
    high_sim_count = (sim > 0.9).sum().item() // 2  # each pair counted twice
    very_high_sim_count = (sim > 0.95).sum().item() // 2

    # Greedy clustering: group queries with >threshold similarity
    def greedy_cluster(threshold=0.85):
        assigned = set()
        clusters = []
        for i in range(n):
            if i in assigned:
                continue
            cluster = [i]
            assigned.add(i)
            for j in range(i + 1, n):
                if j in assigned:
                    continue
                if sim[i, j].item() > threshold:
                    cluster.append(j)
                    assigned.add(j)
            clusters.append(cluster)
        return clusters

    clusters = greedy_cluster(0.85)

    return {
        "top_pairs": pairs[:top_k_pairs],
        "high_sim_pairs_0.9": high_sim_count,
        "very_high_sim_pairs_0.95": very_high_sim_count,
        "mean_sim": sim.sum().item() / (n * (n - 1)),
        "max_sim": sim.max().item(),
        "clusters_at_0.85": len(clusters),
        "cluster_sizes": sorted([len(c) for c in clusters], reverse=True)[:10],
        "similarity_matrix": sim,
    }


# ─────────────────────────────────────────────────────────────
#  5. Segmentation Head Analysis
# ─────────────────────────────────────────────────────────────

def analyze_seg_head(seg_head):
    """Analyze segmentation head component importance."""
    results = {}

    # Cross-attend-prompt importance
    if hasattr(seg_head, "cross_attend_prompt") and seg_head.cross_attend_prompt is not None:
        ca = seg_head.cross_attend_prompt
        W_O = ca.out_proj.weight.data
        _, _, W_V = _extract_qkv(ca.in_proj_weight.data, 256)
        results["cross_attend_prompt_bi"] = _frobenius(W_O @ W_V)
        results["cross_attend_prompt_out_norm"] = _frobenius(W_O)
        total_params = sum(p.numel() for p in ca.parameters())
        results["cross_attend_prompt_params"] = total_params

    # Pixel decoder conv norms
    if hasattr(seg_head, "pixel_decoder"):
        pd = seg_head.pixel_decoder
        for i, conv in enumerate(pd.conv_layers):
            results[f"pixel_decoder.conv_{i}_weight_norm"] = _frobenius(conv.weight.data)
        total_params = sum(p.numel() for p in pd.parameters())
        results["pixel_decoder_params"] = total_params

    # Mask predictor
    if hasattr(seg_head, "mask_predictor"):
        mp = seg_head.mask_predictor
        if hasattr(mp, "mask_embed"):
            for i, layer in enumerate(mp.mask_embed.layers):
                results[f"mask_embed.layer_{i}_weight_norm"] = _frobenius(layer.weight.data)
        total_params = sum(p.numel() for p in mp.parameters())
        results["mask_predictor_params"] = total_params

    # Instance seg head
    if hasattr(seg_head, "instance_seg_head"):
        results["instance_seg_head_weight_norm"] = _frobenius(
            seg_head.instance_seg_head.weight.data
        )

    # Semantic seg head
    if hasattr(seg_head, "semantic_seg_head"):
        results["semantic_seg_head_weight_norm"] = _frobenius(
            seg_head.semantic_seg_head.weight.data
        )

    return results


# ─────────────────────────────────────────────────────────────
#  6. BoxRPB Analysis (significant decoder compute)
# ─────────────────────────────────────────────────────────────

def analyze_boxrpb(decoder):
    """Analyze boxRPB MLP weight norms — these are called every layer."""
    results = {}
    if hasattr(decoder, "boxRPB_embed_x"):
        for i, layer in enumerate(decoder.boxRPB_embed_x.layers):
            results[f"boxRPB_x.layer_{i}_norm"] = _frobenius(layer.weight.data)
        for i, layer in enumerate(decoder.boxRPB_embed_y.layers):
            results[f"boxRPB_y.layer_{i}_norm"] = _frobenius(layer.weight.data)
        total_params = sum(p.numel() for p in decoder.boxRPB_embed_x.parameters())
        total_params += sum(p.numel() for p in decoder.boxRPB_embed_y.parameters())
        results["boxRPB_total_params"] = total_params
    return results


# ─────────────────────────────────────────────────────────────
#  7. Component Parameter Count
# ─────────────────────────────────────────────────────────────

def count_parameters(model):
    """Detailed parameter breakdown."""
    counts = defaultdict(int)

    for name, param in model.named_parameters():
        parts = name.split(".")
        # Top-level component
        top = parts[0]
        counts[top] += param.numel()

        # More granular
        if top == "transformer":
            sub = parts[1] if len(parts) > 1 else "other"
            counts[f"transformer.{sub}"] += param.numel()

            if sub in ("encoder", "decoder") and len(parts) > 2:
                sub2 = parts[2]
                counts[f"transformer.{sub}.{sub2}"] += param.numel()

    return dict(counts)


# ─────────────────────────────────────────────────────────────
#  FLOPs Estimation
# ─────────────────────────────────────────────────────────────

def estimate_flops(d_model=256, ffn_dim=2048, n_heads=8,
                   enc_layers=6, dec_layers=6, n_queries=200,
                   img_tokens=5184, text_tokens=15):
    """
    Rough FLOPs estimate for encoder + decoder (excluding backbone).
    Self-attention: 2 * seq^2 * d  (Q@K^T + attn@V)
    Linear: 2 * in * out
    """
    flops = {}

    # --- Encoder (per layer) ---
    # Self-attention on image tokens: 2 * 2 * T^2 * d  (QK^T + AV)
    enc_sa = 2 * 2 * img_tokens ** 2 * d_model
    # Self-attention projections (Q, K, V, O): 4 * 2 * T * d^2
    enc_sa_proj = 4 * 2 * img_tokens * d_model ** 2
    # Cross-attention to text: 2 * 2 * T * text_tokens * d
    enc_ca = 2 * 2 * img_tokens * text_tokens * d_model
    enc_ca_proj = 4 * 2 * img_tokens * d_model ** 2  # Q/K/V/O (applied on img side)
    # FFN: 2 * T * d * ffn + 2 * T * ffn * d
    enc_ffn = 2 * 2 * img_tokens * d_model * ffn_dim

    enc_per_layer = enc_sa + enc_sa_proj + enc_ca + enc_ca_proj + enc_ffn
    enc_total = enc_per_layer * enc_layers

    flops["enc_self_attn_per_layer"] = enc_sa + enc_sa_proj
    flops["enc_cross_attn_per_layer"] = enc_ca + enc_ca_proj
    flops["enc_ffn_per_layer"] = enc_ffn
    flops["enc_per_layer"] = enc_per_layer
    flops["enc_total"] = enc_total

    # --- Decoder (per layer) ---
    # Self-attention on queries
    dec_sa = 2 * 2 * n_queries ** 2 * d_model
    dec_sa_proj = 4 * 2 * n_queries * d_model ** 2

    # Cross-attention to text
    dec_ca_text = 2 * 2 * n_queries * text_tokens * d_model
    dec_ca_text_proj = 4 * 2 * n_queries * d_model ** 2

    # Cross-attention to image (with boxRPB)
    dec_ca_img = 2 * 2 * n_queries * img_tokens * d_model
    dec_ca_img_proj = 4 * 2 * n_queries * d_model ** 2

    # boxRPB compute: MLP on (Q * H + Q * W) positions, 2-layer MLP(2, d, n_heads)
    H = W = int(img_tokens ** 0.5)  # ~72
    boxrpb = 2 * n_queries * (H + W) * 2 * d_model  # rough estimate

    # FFN
    dec_ffn = 2 * 2 * n_queries * d_model * ffn_dim

    dec_per_layer = (dec_sa + dec_sa_proj + dec_ca_text + dec_ca_text_proj +
                     dec_ca_img + dec_ca_img_proj + boxrpb + dec_ffn)
    dec_total = dec_per_layer * dec_layers

    flops["dec_self_attn_per_layer"] = dec_sa + dec_sa_proj
    flops["dec_cross_attn_text_per_layer"] = dec_ca_text + dec_ca_text_proj
    flops["dec_cross_attn_img_per_layer"] = dec_ca_img + dec_ca_img_proj
    flops["dec_boxrpb_per_layer"] = boxrpb
    flops["dec_ffn_per_layer"] = dec_ffn
    flops["dec_per_layer"] = dec_per_layer
    flops["dec_total"] = dec_total

    flops["total"] = enc_total + dec_total
    return flops


# ─────────────────────────────────────────────────────────────
#  Printing
# ─────────────────────────────────────────────────────────────

def print_section(title):
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


def print_layer_scores(title, scores, key="total_bi"):
    print(f"\n  {title}")
    print(f"  {'Layer':<8} {'Score':>10}  {'Rank':>6}  Details")
    print(f"  {'-' * 60}")

    ranked = sorted(scores, key=lambda x: x[key])
    rank_map = {s["layer"]: rank + 1 for rank, s in enumerate(ranked)}

    for s in scores:
        detail_parts = []
        for k, v in s.items():
            if k not in ("layer", key):
                detail_parts.append(f"{k}={v:.1f}")
        detail = "  ".join(detail_parts)
        print(
            f"  {s['layer']:<8} {s[key]:>10.1f}  "
            f"{'#' + str(rank_map[s['layer']]):>6}  {detail}"
        )

    # Recommendation
    n = len(scores)
    remove_candidates = [s["layer"] for s in ranked[:n // 2]]
    keep_candidates = [s["layer"] for s in ranked[n // 2:]]
    print(f"\n  Recommendation: keep layers {keep_candidates}, "
          f"remove {remove_candidates}")


def print_head_analysis(head_results, n_heads=8):
    print_section("HEAD IMPORTANCE (per attention module)")
    print(f"\n  Module{'':30}  ", end="")
    for h in range(n_heads):
        print(f"  H{h:d}", end="")
    print("   Min    Max   Ratio")
    print(f"  {'-' * (40 + n_heads * 6 + 20)}")

    for name, imps in sorted(head_results.items()):
        mn, mx = min(imps), max(imps)
        ratio = mx / mn if mn > 0 else float("inf")
        print(f"  {name:<38}", end="")
        for imp in imps:
            print(f"  {imp:4.1f}", end="")
        print(f"  {mn:5.1f}  {mx:5.1f}  {ratio:5.1f}x")

    # Aggregate across all modules: which head indices are weakest?
    head_totals = [0.0] * n_heads
    count = 0
    for imps in head_results.values():
        for h, imp in enumerate(imps):
            head_totals[h] += imp
        count += 1
    print(f"\n  Avg importance per head index:")
    for h in range(n_heads):
        print(f"    Head {h}: {head_totals[h] / count:.2f}")
    weakest = sorted(range(n_heads), key=lambda h: head_totals[h])
    print(f"  Weakest → strongest: {weakest}")


def print_ffn_analysis(ffn_results, target_dim=None):
    print_section("FFN NEURON IMPORTANCE")

    for name, imps in sorted(ffn_results.items()):
        imps_t = torch.tensor(imps)
        sorted_imps = imps_t.sort(descending=True).values
        dim = len(imps)

        # Cumulative importance
        total = sorted_imps.sum().item()
        cum = sorted_imps.cumsum(0)
        pct_90 = (cum >= 0.9 * total).nonzero(as_tuple=True)[0][0].item() + 1
        pct_95 = (cum >= 0.95 * total).nonzero(as_tuple=True)[0][0].item() + 1
        pct_99 = (cum >= 0.99 * total).nonzero(as_tuple=True)[0][0].item() + 1

        print(f"\n  {name} (dim={dim}):")
        print(f"    Top neuron: {sorted_imps[0].item():.3f}  "
              f"Bottom: {sorted_imps[-1].item():.3f}  "
              f"Ratio: {sorted_imps[0].item() / max(sorted_imps[-1].item(), 1e-8):.1f}x")
        print(f"    Neurons for 90% importance: {pct_90}/{dim} "
              f"({100 * pct_90 / dim:.0f}%)")
        print(f"    Neurons for 95% importance: {pct_95}/{dim} "
              f"({100 * pct_95 / dim:.0f}%)")
        print(f"    Neurons for 99% importance: {pct_99}/{dim} "
              f"({100 * pct_99 / dim:.0f}%)")

        if target_dim:
            kept_imp = sorted_imps[:target_dim].sum().item()
            print(f"    If pruned to {target_dim}: "
                  f"retain {100 * kept_imp / total:.1f}% of importance")


def print_query_analysis(query_results):
    print_section("QUERY EMBEDDING REDUNDANCY")

    print(f"\n  Mean pairwise cosine similarity: {query_results['mean_sim']:.4f}")
    print(f"  Max pairwise cosine similarity:  {query_results['max_sim']:.4f}")
    print(f"  Pairs with sim > 0.90: {query_results['high_sim_pairs_0.9']}")
    print(f"  Pairs with sim > 0.95: {query_results['very_high_sim_pairs_0.95']}")
    print(f"  Clusters at threshold 0.85: {query_results['clusters_at_0.85']} "
          f"(from 200 queries)")
    print(f"  Largest cluster sizes: {query_results['cluster_sizes']}")

    print(f"\n  Top-10 most similar query pairs:")
    for i, j, sim in query_results["top_pairs"][:10]:
        print(f"    Q{i:3d} <-> Q{j:3d}  sim={sim:.4f}")

    n_clusters = query_results["clusters_at_0.85"]
    print(f"\n  Recommendation: {n_clusters} clusters found at cosine 0.85 threshold.")
    if n_clusters < 150:
        print(f"  Could reduce 200 queries to ~{n_clusters} with minimal quality loss.")


def print_seg_head_analysis(seg_results):
    print_section("SEGMENTATION HEAD ANALYSIS")
    for k, v in sorted(seg_results.items()):
        if "params" in k:
            print(f"  {k}: {v:,}")
        else:
            print(f"  {k}: {v:.4f}")


def print_flops(flops):
    print_section("FLOPs ESTIMATE (encoder + decoder, excluding backbone)")

    def fmt(f):
        if f >= 1e12:
            return f"{f / 1e12:.2f}T"
        if f >= 1e9:
            return f"{f / 1e9:.2f}G"
        return f"{f / 1e6:.1f}M"

    print(f"\n  Encoder:")
    print(f"    Self-attention / layer: {fmt(flops['enc_self_attn_per_layer'])}")
    print(f"    Cross-attention / layer: {fmt(flops['enc_cross_attn_per_layer'])}")
    print(f"    FFN / layer:            {fmt(flops['enc_ffn_per_layer'])}")
    print(f"    Per layer total:        {fmt(flops['enc_per_layer'])}")
    print(f"    All {6} layers:          {fmt(flops['enc_total'])}")

    print(f"\n  Decoder:")
    print(f"    Self-attention / layer:        {fmt(flops['dec_self_attn_per_layer'])}")
    print(f"    Cross-attn text / layer:       {fmt(flops['dec_cross_attn_text_per_layer'])}")
    print(f"    Cross-attn image / layer:      {fmt(flops['dec_cross_attn_img_per_layer'])}")
    print(f"    BoxRPB / layer:                {fmt(flops['dec_boxrpb_per_layer'])}")
    print(f"    FFN / layer:                   {fmt(flops['dec_ffn_per_layer'])}")
    print(f"    Per layer total:               {fmt(flops['dec_per_layer'])}")
    print(f"    All {6} layers:                {fmt(flops['dec_total'])}")

    print(f"\n  TOTAL encoder+decoder: {fmt(flops['total'])}")

    # Encoder dominates because of O(T^2) self-attention on 5184 tokens
    enc_pct = 100 * flops["enc_total"] / flops["total"]
    dec_pct = 100 * flops["dec_total"] / flops["total"]
    print(f"  Encoder: {enc_pct:.1f}%  Decoder: {dec_pct:.1f}%")

    sa_pct = 100 * flops["enc_self_attn_per_layer"] / flops["enc_per_layer"]
    print(f"  Encoder self-attn is {sa_pct:.1f}% of encoder compute")


def print_pruning_recommendations(enc_scores, dec_scores, head_results, ffn_results,
                                  query_results, flops_baseline):
    print_section("PRUNING RECOMMENDATIONS SUMMARY")

    # Layer recommendations
    enc_ranked = sorted(enc_scores, key=lambda x: x["total_bi"])
    dec_ranked = sorted(dec_scores, key=lambda x: x["total_bi"])

    print("\n  Layer Pruning:")
    print(f"    Encoder: remove layers {[s['layer'] for s in enc_ranked[:2]]} "
          f"(lowest BI), keep {[s['layer'] for s in enc_ranked[2:]]}")
    print(f"    Decoder: remove layers {[s['layer'] for s in dec_ranked[:2]]} "
          f"(lowest BI), keep {[s['layer'] for s in dec_ranked[2:]]}")

    # Head recommendations
    total_heads = 0
    prunable_heads = 0
    for imps in head_results.values():
        total_heads += len(imps)
        mn, mx = min(imps), max(imps)
        if mn > 0:
            ratio = mx / mn
            # Heads with < 50% of max importance are prunable
            prunable_heads += sum(1 for imp in imps if imp < 0.5 * mx)

    print(f"\n  Head Pruning:")
    print(f"    {prunable_heads}/{total_heads} heads have <50% of max importance "
          f"in their module")
    print(f"    Recommendation: 8 -> 6 heads (prune 2 weakest per module)")

    # FFN recommendations
    print(f"\n  FFN Width Pruning:")
    for name, imps in sorted(ffn_results.items()):
        imps_t = torch.tensor(imps)
        sorted_imps = imps_t.sort(descending=True).values
        total = sorted_imps.sum().item()
        kept_1536 = sorted_imps[:1536].sum().item()
        kept_1024 = sorted_imps[:1024].sum().item()
        print(f"    {name}: 2048->1536 retains {100 * kept_1536 / total:.1f}%, "
              f"2048->1024 retains {100 * kept_1024 / total:.1f}%")

    # Query recommendations
    print(f"\n  Query Pruning:")
    n_clusters = query_results["clusters_at_0.85"]
    print(f"    {query_results['clusters_at_0.85']} distinct clusters at cosine 0.85")
    rec_queries = min(max(n_clusters, 64), 128)
    print(f"    Recommendation: 200 -> {rec_queries} queries")

    # Estimated speedup
    print(f"\n  Estimated Speedup:")

    # Pruned FLOPs estimate
    pruned_flops = estimate_flops(
        enc_layers=4, dec_layers=4, n_heads=6,
        ffn_dim=1536, n_queries=rec_queries
    )
    speedup = flops_baseline["total"] / pruned_flops["total"]
    print(f"    Baseline encoder+decoder: "
          f"{flops_baseline['total'] / 1e9:.1f}G FLOPs")
    print(f"    Pruned (4 enc, 4 dec, 6 heads, 1536 FFN, {rec_queries} queries): "
          f"{pruned_flops['total'] / 1e9:.1f}G FLOPs")
    print(f"    Theoretical speedup: {speedup:.2f}x")

    # With 3+3 layers
    aggressive_flops = estimate_flops(
        enc_layers=3, dec_layers=3, n_heads=4,
        ffn_dim=1024, n_queries=64
    )
    speedup_agg = flops_baseline["total"] / aggressive_flops["total"]
    print(f"    Aggressive (3 enc, 3 dec, 4 heads, 1024 FFN, 64 queries): "
          f"{aggressive_flops['total'] / 1e9:.1f}G FLOPs")
    print(f"    Theoretical speedup: {speedup_agg:.2f}x")

    # Note about backbone
    print(f"\n  Note: The ViT-H backbone (~1.93T FLOPs) dominates total model")
    print(f"  latency. For maximum speedup, combine structural pruning with")
    print(f"  backbone distillation (see scripts/distill.py).")
    print(f"  Encoder+decoder pruning primarily helps multi-class inference")
    print(f"  where the encoder/decoder run once per class.")


# ─────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="SAM3 Structural Pruning Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to SAM3 checkpoint (default: download from HF)")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--apply", action="store_true",
                        help="Apply recommended pruning and save")
    parser.add_argument("--output", type=str, default="pruned_sam3.pt",
                        help="Output path for pruned model")
    parser.add_argument("--enc-layers", type=int, default=None,
                        help="Keep top-K encoder layers")
    parser.add_argument("--dec-layers", type=int, default=None,
                        help="Keep top-K decoder layers")
    parser.add_argument("--heads", type=int, default=None,
                        help="Keep top-K heads per module")
    parser.add_argument("--ffn-dim", type=int, default=None,
                        help="Target FFN intermediate dimension")
    parser.add_argument("--queries", type=int, default=None,
                        help="Keep top-K queries")
    args = parser.parse_args()

    # ── Load model ──
    print("Loading SAM3 model...")
    from sam3.model_builder import build_sam3_image_model

    model = build_sam3_image_model(
        device=args.device,
        checkpoint_path=args.checkpoint,
        eval_mode=True,
        load_from_HF=args.checkpoint is None,
        enable_inst_interactivity=False,
    )
    model.eval()

    # Extract components
    encoder = model.transformer.encoder
    decoder = model.transformer.decoder
    seg_head = model.segmentation_head
    d_model = 256
    n_heads = 8

    # ── Parameter counts ──
    print_section("PARAMETER COUNTS")
    param_counts = count_parameters(model)
    total = sum(p.numel() for p in model.parameters())
    print(f"\n  Total parameters: {total:,}")
    for k, v in sorted(param_counts.items(), key=lambda x: -x[1]):
        pct = 100 * v / total
        if pct > 0.5:
            print(f"  {k:<45} {v:>12,}  ({pct:.1f}%)")

    # ── FLOPs baseline ──
    flops_baseline = estimate_flops()
    print_flops(flops_baseline)

    # ── Layer analysis ──
    print_section("LAYER PRUNING (Block Influence Scores)")
    enc_scores = analyze_encoder_layers(encoder, d_model)
    dec_scores = analyze_decoder_layers(decoder, d_model)
    print_layer_scores("Encoder Layers (lower BI = safer to remove)", enc_scores)
    print_layer_scores("Decoder Layers (lower BI = safer to remove)", dec_scores)

    # ── Head analysis ──
    head_results = analyze_all_heads(encoder, decoder, d_model, n_heads)
    print_head_analysis(head_results, n_heads)

    # ── FFN analysis ──
    ffn_results = analyze_all_ffns(encoder, decoder)
    print_ffn_analysis(ffn_results, target_dim=args.ffn_dim or 1536)

    # ── Query analysis ──
    query_results = analyze_query_redundancy(decoder)
    print_query_analysis(query_results)

    # ── Seg head analysis ──
    seg_results = analyze_seg_head(seg_head)
    print_seg_head_analysis(seg_results)

    # ── BoxRPB analysis ──
    boxrpb_results = analyze_boxrpb(decoder)
    if boxrpb_results:
        print_section("BoxRPB ANALYSIS")
        for k, v in sorted(boxrpb_results.items()):
            if "params" in k:
                print(f"  {k}: {v:,}")
            else:
                print(f"  {k}: {v:.4f}")

    # ── Summary recommendations ──
    print_pruning_recommendations(
        enc_scores, dec_scores, head_results, ffn_results,
        query_results, flops_baseline
    )

    # ── Apply pruning ──
    if args.apply:
        print_section("APPLYING PRUNING")
        apply_pruning(
            model, encoder, decoder, seg_head,
            enc_scores, dec_scores, head_results, ffn_results,
            query_results,
            enc_keep=args.enc_layers,
            dec_keep=args.dec_layers,
            head_keep=args.heads,
            ffn_target=args.ffn_dim,
            query_keep=args.queries,
            output_path=args.output,
            d_model=d_model,
            n_heads=n_heads,
        )


# ─────────────────────────────────────────────────────────────
#  Pruning Application
# ─────────────────────────────────────────────────────────────

def _prune_layers(module_list, scores, keep_n, key="total_bi"):
    """Remove lowest-scoring layers from a ModuleList."""
    ranked = sorted(scores, key=lambda x: x[key], reverse=True)
    keep_indices = sorted([s["layer"] for s in ranked[:keep_n]])
    new_layers = nn.ModuleList([module_list[i] for i in keep_indices])
    return new_layers, keep_indices


def _prune_heads_in_attn(attn, d_model, n_heads, keep_heads):
    """Prune attention heads by zeroing out pruned head weights in-place.

    Full structural head removal requires reshaping all Q/K/V/O matrices.
    For simplicity, we zero out pruned heads — this gives the same compute
    savings when combined with sparse kernels, or can be followed by
    physical removal in ONNX export.
    """
    d_head = d_model // n_heads
    prune_heads = set(range(n_heads)) - set(keep_heads)

    with torch.no_grad():
        for h in prune_heads:
            s, e = h * d_head, (h + 1) * d_head
            # Zero Q, K, V rows for this head
            attn.in_proj_weight[s:e, :] = 0
            attn.in_proj_weight[d_model + s : d_model + e, :] = 0
            attn.in_proj_weight[2 * d_model + s : 2 * d_model + e, :] = 0
            if attn.in_proj_bias is not None:
                attn.in_proj_bias[s:e] = 0
                attn.in_proj_bias[d_model + s : d_model + e] = 0
                attn.in_proj_bias[2 * d_model + s : 2 * d_model + e] = 0
            # Zero O columns for this head
            attn.out_proj.weight[:, s:e] = 0


def _prune_ffn(linear1, linear2, keep_n):
    """Structurally prune FFN by keeping only top-K neurons."""
    W1 = linear1.weight.data.float()
    W2 = linear2.weight.data.float()

    # Compute neuron importance
    row_norms = W1.norm(dim=1)
    col_norms = W2.norm(dim=0)
    importances = row_norms * col_norms

    # Keep top-K
    _, keep_idx = importances.topk(keep_n)
    keep_idx = keep_idx.sort().values

    # Create new layers
    new_linear1 = nn.Linear(linear1.in_features, keep_n, bias=linear1.bias is not None)
    new_linear2 = nn.Linear(keep_n, linear2.out_features, bias=linear2.bias is not None)

    new_linear1.weight.data = linear1.weight.data[keep_idx]
    if linear1.bias is not None:
        new_linear1.bias.data = linear1.bias.data[keep_idx]
    new_linear2.weight.data = linear2.weight.data[:, keep_idx]
    if linear2.bias is not None:
        new_linear2.bias.data = linear2.bias.data.clone()

    return new_linear1, new_linear2


def _prune_queries(decoder, keep_n, query_results):
    """Prune queries by keeping cluster representatives."""
    Q = decoder.query_embed.weight.data  # (200, d)

    # Use importance = distance from mean (diverse queries are more important)
    Q_float = Q.float()
    Q_norm = Q_float / Q_float.norm(dim=1, keepdim=True).clamp(min=1e-8)
    sim_matrix = query_results["similarity_matrix"]

    # Greedy selection: pick most dissimilar queries
    selected = [0]  # start with first query
    for _ in range(keep_n - 1):
        # For each unselected query, compute max similarity to any selected query
        min_sims = []
        for q in range(Q.shape[0]):
            if q in selected:
                min_sims.append(float("inf"))
            else:
                max_sim = max(sim_matrix[q, s].item() for s in selected)
                min_sims.append(max_sim)
        # Select query with lowest max-similarity (most different from selected set)
        next_q = min(range(len(min_sims)), key=lambda i: min_sims[i])
        selected.append(next_q)

    selected.sort()
    print(f"    Selected query indices: {selected[:20]}{'...' if len(selected) > 20 else ''}")

    # Update query embed
    new_embed = nn.Embedding(keep_n, Q.shape[1])
    new_embed.weight.data = Q[selected]
    decoder.query_embed = new_embed
    decoder.num_queries = keep_n

    # Update reference points
    if hasattr(decoder, "reference_points") and decoder.reference_points is not None:
        ref = decoder.reference_points.weight.data
        new_ref = nn.Embedding(keep_n, ref.shape[1])
        new_ref.weight.data = ref[selected]
        decoder.reference_points = new_ref

    return selected


def apply_pruning(model, encoder, decoder, seg_head,
                  enc_scores, dec_scores, head_results, ffn_results,
                  query_results,
                  enc_keep=None, dec_keep=None, head_keep=None,
                  ffn_target=None, query_keep=None,
                  output_path="pruned_sam3.pt",
                  d_model=256, n_heads=8):
    """Apply structural pruning based on analysis results."""

    # Defaults
    enc_keep = enc_keep or 4
    dec_keep = dec_keep or 4
    head_keep = head_keep or 6
    ffn_target = ffn_target or 1536
    query_keep = query_keep or 100

    # 1. Layer pruning
    print(f"\n  Pruning encoder: {len(encoder.layers)} -> {enc_keep} layers")
    encoder.layers, enc_kept = _prune_layers(encoder.layers, enc_scores, enc_keep)
    encoder.num_layers = enc_keep
    print(f"    Kept encoder layers: {enc_kept}")

    print(f"  Pruning decoder: {len(decoder.layers)} -> {dec_keep} layers")
    decoder.layers, dec_kept = _prune_layers(decoder.layers, dec_scores, dec_keep)
    # Also prune fine_layers if present
    if hasattr(decoder, "fine_layers") and decoder.fine_layers is not None:
        fine_layers = [decoder.fine_layers[i] for i in dec_kept]
        decoder.fine_layers = fine_layers
    decoder.num_layers = dec_keep
    print(f"    Kept decoder layers: {dec_kept}")

    # 2. Head pruning
    if head_keep < n_heads:
        print(f"\n  Pruning heads: {n_heads} -> {head_keep} per module")
        for name, imps in head_results.items():
            ranked_heads = sorted(range(n_heads), key=lambda h: imps[h], reverse=True)
            keep = sorted(ranked_heads[:head_keep])

            # Find the actual attention module
            parts = name.split(".")
            if parts[0] == "enc":
                layer_idx = int(parts[1])
                if layer_idx not in enc_kept:
                    continue
                mapped_idx = enc_kept.index(layer_idx)
                if "self_attn" in name:
                    _prune_heads_in_attn(encoder.layers[mapped_idx].self_attn,
                                         d_model, n_heads, keep)
                elif "cross_attn" in name:
                    _prune_heads_in_attn(encoder.layers[mapped_idx].cross_attn_image,
                                         d_model, n_heads, keep)
            elif parts[0] == "dec":
                layer_idx = int(parts[1])
                if layer_idx not in dec_kept:
                    continue
                mapped_idx = dec_kept.index(layer_idx)
                if "self_attn" in name:
                    _prune_heads_in_attn(decoder.layers[mapped_idx].self_attn,
                                         d_model, n_heads, keep)
                elif "cross_attn_img" in name:
                    _prune_heads_in_attn(decoder.layers[mapped_idx].cross_attn,
                                         d_model, n_heads, keep)
                elif "cross_attn_text" in name:
                    _prune_heads_in_attn(decoder.layers[mapped_idx].ca_text,
                                         d_model, n_heads, keep)

    # 3. FFN pruning
    if ffn_target < 2048:
        print(f"\n  Pruning FFN: 2048 -> {ffn_target}")
        for layer in encoder.layers:
            layer.linear1, layer.linear2 = _prune_ffn(
                layer.linear1, layer.linear2, ffn_target
            )
        for layer in decoder.layers:
            layer.linear1, layer.linear2 = _prune_ffn(
                layer.linear1, layer.linear2, ffn_target
            )

    # 4. Query pruning
    if query_keep < 200:
        print(f"\n  Pruning queries: 200 -> {query_keep}")
        _prune_queries(decoder, query_keep, query_results)

    # 5. Remove cross_attend_prompt from seg head (optional speedup)
    if hasattr(seg_head, "cross_attend_prompt") and seg_head.cross_attend_prompt is not None:
        print(f"\n  Removing seg_head.cross_attend_prompt")
        seg_head.cross_attend_prompt = None
        seg_head.cross_attn_norm = None

    # Count parameters after pruning
    total_after = sum(p.numel() for p in model.parameters())
    print(f"\n  Parameters after pruning: {total_after:,}")

    # Build checkpoint in the format expected by _load_checkpoint:
    #   {"model": {"detector.<key>": value, ...}, "pruning_config": {...}}
    pruning_config = {
        "enc_layers": enc_keep,
        "dec_layers": dec_keep,
        "ffn_dim": ffn_target,
        "num_queries": query_keep,
        "remove_cross_attend_prompt": (
            not hasattr(seg_head, "cross_attend_prompt")
            or seg_head.cross_attend_prompt is None
        ),
    }

    state_dict = model.state_dict()
    # Add "detector." prefix so _load_checkpoint can strip it back
    prefixed = {"detector." + k: v for k, v in state_dict.items()}

    print(f"\n  Saving pruned model to {output_path}")
    print(f"  Pruning config: {pruning_config}")
    torch.save({"model": prefixed, "pruning_config": pruning_config}, output_path)
    print("  Done!")


if __name__ == "__main__":
    main()
