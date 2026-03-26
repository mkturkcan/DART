# DART: Detect Anything in Real Time

Training-free framework that converts SAM3 into a real-time multi-class
open-vocabulary detector. Achieves **55.8 AP** on COCO val2017 (80 classes)
at **15.8 FPS** (4 classes, 1008px) on a single RTX 4080.

Distilled student backbones and pre-built weights are available on
[HuggingFace](https://huggingface.co/mehmetkeremturkcan/DART).

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Single-Image Detection](#single-image-detection)
- [Video Inference](#video-inference)
- [Tracking](#tracking)
- [TensorRT Export](#tensorrt-export)
  - [Backbone](#backbone-export)
  - [Encoder-Decoder](#encoder-decoder-export)
  - [Student Backbones](#student-backbone-export)
  - [EfficientSAM3 Backbones](#efficientsam3-backbone-export)
- [Block Pruning](#block-pruning)
- [Text Cache](#text-cache)
- [COCO Evaluation](#coco-evaluation)
- [Benchmarks](#benchmarks)
- [FP16 Precision Analysis](#fp16-precision-analysis)
- [Scripts Reference](#scripts-reference)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)

---

## Installation

Tested on Windows 11, RTX 4080 16 GB. All commands use bash syntax.

| Package | Version |
|---------|---------|
| Python | 3.11+ |
| torch | 2.7.0+ (CUDA 12.6+) |
| torchvision | 0.22.0+ |
| tensorrt | 10.9.0+ (optional, for TRT export) |
| numpy | < 2.0 |

```bash
# 1. Create a conda environment
conda create -n dartsam3 python=3.11 -y
conda activate dartsam3

# 2. Install PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# 3. Install TensorRT (optional — only needed for TRT engine export/inference)
pip install tensorrt

# 4. Install DART (installs all other dependencies automatically)
pip install -e .
```

> **Windows note:** Always prefix Python commands with `PYTHONIOENCODING=utf-8`
> on Windows. TRT, torch, and the detection scripts print Unicode characters
> that the default Windows console encoding (cp1252) cannot handle.

### Files needed

| File | Description | How to get |
|------|-------------|------------|
| `sam3.pt` | SAM3 checkpoint | Auto-downloads from HuggingFace on first run |
| Student weights | Distilled backbones (RepViT, TinyViT, EfficientViT) | [HuggingFace](https://huggingface.co/mehmetkeremturkcan/DART) |
| `x.jpg` | Test image | Any image |
| `input.mp4` | Test video | Any video |
| `train2017/` | Calibration images | [COCO](https://cocodataset.org) (only for pruning search) |
| `val2017/` | Validation images | [COCO](https://cocodataset.org) (only for evaluation) |

---

## Quick Start

Three backbone options from high quality to high speed. All start from
`sam3.pt` (auto-downloads on first run) and share the same encoder-decoder
engine.

### Step 1: Build the shared encoder-decoder engine (one-time)

```bash
python -m sam3.trt.export_enc_dec --checkpoint sam3.pt \
    --output enc_dec.onnx --max-classes 4 --imgsz 1008
python -m sam3.trt.build_engine --onnx enc_dec.onnx \
    --output enc_dec_fp16.engine --fp16 --mixed-precision none
```

### Step 2: Build a backbone engine

Pick one (or build all three — they coexist):

```bash
# ViT-H full (55.8 AP, 13.5 FPS) — highest quality
PYTHONIOENCODING=utf-8 python scripts/export_hf_backbone.py \
    --image x.jpg --imgsz 1008

# ViT-H Pruned-16 (53.6 AP, 19.1 FPS) — best quality/speed tradeoff
PYTHONIOENCODING=utf-8 python scripts/export_hf_backbone.py \
    --image x.jpg --imgsz 1008 \
    --pruned-checkpoint distilled/pruned_16blocks.pt \
    --output-engine hf_pruned16_fp16.engine

# RepViT-M2.3 (38.7 AP, 30.2 FPS) — fastest with usable quality
PYTHONIOENCODING=utf-8 python scripts/export_student_trt.py \
    --models repvit_m2_3
```

### Step 3: Run detection

```bash
# ViT-H full
python demo_multiclass.py --image x.jpg --classes person car bicycle dog \
    --trt hf_backbone_1008_fp16.engine --trt-enc-dec enc_dec_fp16.engine \
    --checkpoint sam3.pt --fast --detection-only -o x_annotated.jpg

# ViT-H Pruned-16
python demo_multiclass.py --image x.jpg --classes person car bicycle dog \
    --trt hf_pruned16_fp16.engine --trt-enc-dec enc_dec_fp16.engine \
    --checkpoint sam3.pt --fast --detection-only -o x_annotated.jpg

# RepViT-M2.3
python demo_multiclass.py --image x.jpg --classes person car bicycle dog \
    --trt student_repvit_m2_3_fp16.engine --trt-enc-dec enc_dec_fp16.engine \
    --checkpoint sam3.pt --fast --detection-only -o x_annotated.jpg
```

All three use the same `--trt-enc-dec` engine — only swap `--trt` to switch
backbones. No retraining or adapter flags needed; the TRT engine IS the
backbone.

### Without TRT (PyTorch only)

```bash
python demo_multiclass.py \
    --image x.jpg \
    --classes person car bicycle dog \
    --fast --detection-only
```

### 80-class COCO setup

```bash
# Build enc-dec engine + text cache for all 80 classes
python scripts/build_coco_engine.py --checkpoint sam3.pt

# Run video
python demo_video.py --video input.mp4 --coco \
    --checkpoint sam3.pt \
    --trt hf_backbone_1008_fp16.engine \
    --trt-enc-dec enc_dec_coco_fp16_80.engine \
    --text-cache text_cache_coco.pt \
    --imgsz 1008 -o output.mp4
```

---

## Single-Image Detection

### `demo_multiclass.py`

```bash
# Basic (batched FP16)
python demo_multiclass.py \
    --image x.jpg --classes person car bicycle dog \
    --checkpoint sam3.pt --fast --detection-only

# With torch.compile + TRT enc-dec + pruning
python demo_multiclass.py \
    --image x.jpg --classes person car bicycle dog \
    --checkpoint sam3.pt --fast --detection-only \
    --compile max-autotune \
    --trt-enc-dec enc_dec_fp16.engine \
    --imgsz 1008 --warmup 3

# Full TRT (backbone + enc-dec)
python demo_multiclass.py --image x.jpg --classes person car bicycle dog \
    --trt hf_backbone_1008_fp16.engine --trt-enc-dec enc_dec_fp16.engine \
    --checkpoint sam3.pt --fast --detection-only -o x_annotated.jpg

# Compare all inference modes
python demo_multiclass.py --benchmark --classes person car bicycle dog
```


### Key flags

| Flag | Default | Description |
|------|---------|-------------|
| `--image` | required | Input image |
| `--classes` | person car bicycle | Target class names |
| `--checkpoint` | None | Model checkpoint (auto-downloads if omitted) |
| `--fast` | off | Batched + FP16 + presence early-exit |
| `--detection-only` | off | Skip mask generation (boxes + scores only) |
| `--compile MODE` | None | `default`, `reduce-overhead`, or `max-autotune` |
| `--trt ENGINE` | None | TRT backbone engine |
| `--trt-enc-dec ENGINE` | None | TRT enc-dec engine |
| `--trt-max-classes` | 4 | Max classes the enc-dec engine was built for |
| `--imgsz` | 1008 | Input resolution (divisible by 14) |
| `--confidence` | 0.3 | Detection confidence threshold |
| `--nms` | 0.7 | Per-class NMS IoU threshold |
| `--warmup` | 0 | Warmup passes before timed inference |
| `--output` | auto | Output annotated image path |
| `--text-cache` | None | Text embedding cache file (.pt) |
| `--mask-blocks` | None | Sub-block pruning spec |
| `--skip-blocks` | None | Full block indices to skip |

---

## Video Inference

### `demo_video.py`

```bash
python demo_video.py \
    --video input.mp4 \
    --classes person car bicycle \
    --checkpoint sam3.pt \
    --trt hf_backbone_1008_fp16.engine \
    --trt-enc-dec enc_dec_fp16.engine \
    --imgsz 1008 -o output.mp4
```

The video pipeline automatically uses inter-frame pipelining: the backbone for
frame $t{+}1$ runs on a separate CUDA stream while the encoder-decoder processes
frame $t$, reducing per-frame latency to `max(backbone, enc-dec)`.

### Key flags

| Flag | Default | Description |
|------|---------|-------------|
| `--video` | required | Input video |
| `--classes` | car pedestrian bicycle | Target class names |
| `--coco` | off | Use all 80 COCO classes |
| `--compile MODE` | None | torch.compile mode |
| `--trt ENGINE` | None | TRT backbone engine |
| `--trt-enc-dec ENGINE` | None | TRT enc-dec engine |
| `--imgsz` | 1008 | Resolution (divisible by 14) |
| `--output` | None | Output video file |
| `--display` | off | Live preview window |
| `--max-frames` | 0 (all) | Stop after N frames |
| `--text-cache` | None | Text embedding cache |
| `--mask-blocks` | None | Sub-block pruning spec |
| `--track` | off | Enable ByteTrack |

---

## Tracking

ByteTrack multi-object tracking provides persistent IDs across video frames.

```bash
python demo_video.py --video input.mp4 \
    --classes person car bicycle \
    --checkpoint sam3.pt \
    --trt hf_backbone_1008_fp16.engine \
    --trt-enc-dec enc_dec_fp16.engine \
    --track --class-agnostic-nms 0.7 \
    -o output.mp4
```

Features: three-stage IoU association, vectorized Kalman filter (~0.1 ms/frame),
class label smoothing, score EMA, duplicate track suppression.

| Flag | Default | Description |
|------|---------|-------------|
| `--track` | off | Enable ByteTrack |
| `--track-thresh` | 0.5 | High/low score split |
| `--match-thresh` | 0.5 | Min IoU for association |
| `--max-time-lost` | 30 | Frames before dropping track |
| `--class-agnostic-nms THRESH` | disabled | Cross-class NMS (useful for similar classes like car/suv/van) |

---

## TensorRT Export

The pipeline uses two separate TRT FP16 engines that communicate via GPU tensors:

| Component | Script | Latency (1008px) |
|-----------|--------|-------------------|
| **Backbone** (ViT-H/14) | `scripts/export_hf_backbone.py` | 53 ms |
| **Encoder-decoder** (6+6 layers) | `sam3.trt.export_enc_dec` | 7–41 ms (1–8 cls) |

The text encoder stays in PyTorch and caches embeddings to GPU — changing
classes requires only recomputing text (milliseconds), not rebuilding engines.

### Backbone Export

The HuggingFace SAM3 backbone uses restructured attention (explicit Q·K^T,
real-valued RoPE) that enables correct TRT FP16 (cos > 0.999).

```bash
# Full backbone, 1008px
PYTHONIOENCODING=utf-8 python scripts/export_hf_backbone.py \
    --image x.jpg --imgsz 1008

# 644px (2.5x faster, -30% AP)
PYTHONIOENCODING=utf-8 python scripts/export_hf_backbone.py \
    --image x.jpg --imgsz 644

# With sub-block pruning
PYTHONIOENCODING=utf-8 python scripts/export_hf_backbone.py \
    --image x.jpg --imgsz 1008 \
    --mask-blocks "25:attn,28:mlp,27:attn,22:attn,28:attn,30:mlp,20:attn,27:mlp"
```

Outputs: `onnx_hf_backbone/hf_backbone.onnx` + `hf_backbone_fp16.engine`.
Customize with `--output-onnx` / `--output-engine`.

| Flag | Default | Description |
|------|---------|-------------|
| `--imgsz` | 1008 | Resolution (divisible by 14) |
| `--mask-blocks` | None | Sub-block pruning spec |
| `--skip-export` | off | Reuse existing ONNX |
| `--skip-build` | off | Reuse existing engine |
| `--benchmark-only` | off | Only benchmark existing engine |
| `--split-block K` | None | Split into two engines at block K |

#### I/O format

```
Input:  pixel_values  float32 [1, 3, H, W]
Output: fpn_0         float32 [1, 256, H/3.5, W/3.5]   # 4x upsample
        fpn_1         float32 [1, 256, H/7, W/7]        # 2x upsample
        fpn_2         float32 [1, 256, H/14, W/14]      # 1x identity
```

### Encoder-Decoder Export

```bash
# Export ONNX
python -m sam3.trt.export_enc_dec --checkpoint sam3.pt \
    --output enc_dec.onnx --max-classes 4 --imgsz 1008

# Build TRT FP16 engine
python -m sam3.trt.build_engine --onnx enc_dec.onnx \
    --output enc_dec_fp16.engine --fp16 --mixed-precision none
```

**Important:** Use `--mixed-precision none` for enc-dec engines. The auto-detect
heuristic applies backbone-specific rules that are wrong for the encoder-decoder.

| Flag | Default | Description |
|------|---------|-------------|
| `--max-classes` | 4 | Fixed batch dimension (set to max classes you'll detect) |
| `--imgsz` | 1008 | Must match inference resolution |

VRAM for engine build: ~4 GB per `max-classes` count. The engine is GPU-specific.

#### I/O format

```
Inputs:  img_feat    float32 [max_classes, 256, 72, 72]    FPN features
         img_pos     float32 [max_classes, 256, 72, 72]    position encoding
         text_feats  float32 [32, max_classes, 256]         per-class text
         text_mask   float32 [max_classes, 32]              text padding mask
Outputs: scores      float32 [max_classes, 200, 1]          detection logits
         boxes       float32 [max_classes, 200, 4]           cxcywh (sigmoid)
```

#### Common engine configurations

```bash
# 1-4 classes, 1008px (real-time at 15+ FPS)
python -m sam3.trt.export_enc_dec --checkpoint sam3.pt \
    --output enc_dec.onnx --max-classes 4 --imgsz 1008
python -m sam3.trt.build_engine --onnx enc_dec.onnx \
    --output enc_dec_fp16.engine --fp16 --mixed-precision none

# 80 COCO classes, 644px (single batch)
python -m sam3.trt.export_enc_dec --checkpoint sam3.pt \
    --output enc_dec_644_coco80.onnx --max-classes 80 --imgsz 644
python -m sam3.trt.build_engine --onnx enc_dec_644_coco80.onnx \
    --output enc_dec_644_coco80_fp16.engine --fp16 --mixed-precision none

# 80 COCO classes, 1008px — chunked (OOMs at max_classes=80 on 16 GB)
python -m sam3.trt.export_enc_dec --checkpoint sam3.pt \
    --output enc_dec_1008_c16.onnx --max-classes 16 --imgsz 1008
python -m sam3.trt.build_engine --onnx enc_dec_1008_c16.onnx \
    --output enc_dec_1008_c16_fp16.engine --fp16 --mixed-precision none
# The predictor automatically chunks 80 classes into 5 passes of 16.
```

### Student Backbone Export

Distilled student backbones replace ViT-H for 3–5x faster backbone inference.
Train adapters first with `scripts/distill.py`, then export:

```bash
# Export all 4 student backbones (ONNX + TRT FP16)
PYTHONIOENCODING=utf-8 python scripts/export_student_trt.py

# Or a subset
PYTHONIOENCODING=utf-8 python scripts/export_student_trt.py \
    --models repvit_m2_3 tiny_vit_21m
```

Student TRT engines are drop-in replacements — just swap `--trt`:

```bash
python demo_video.py --video input.mp4 --classes car person \
    --trt student_repvit_m2_3_fp16.engine \
    --trt-enc-dec enc_dec_fp16.engine \
    --checkpoint sam3.pt -o output.mp4
```

| Model | Backbone Params | COCO AP | Backbone Latency |
|-------|----------------|---------|------------------|
| ViT-H Pruned-16 | 220M | 53.6 | 26.6 ms |
| RepViT-M2.3 | 8.2M | 38.7 | 13.9 ms |
| TinyViT-21M | 21M | 30.1 | 12.2 ms |
| EfficientViT-L2 | 9.2M | 21.7 | 10.7 ms |
| EfficientViT-L1 | 5.3M | 16.3 | 10.4 ms |

### EfficientSAM3 Backbone Export

EfficientSAM3 checkpoints contain finetuned encoder-decoder weights — you must
build a **separate enc-dec engine per checkpoint**.

```bash
# Export backbone
PYTHONIOENCODING=utf-8 python scripts/export_efficient_backbone.py \
    --variant repvit

# Export enc-dec (must use matching checkpoint)
python -m sam3.trt.export_enc_dec \
    --checkpoint stage1_all_converted/efficient_sam3_repvit_l.pt \
    --efficient-backbone repvit --efficient-model m2_3 \
    --output enc_dec_repvit_c16.onnx --max-classes 16
python -m sam3.trt.build_engine --onnx enc_dec_repvit_c16.onnx \
    --output enc_dec_repvit_c16_fp16.engine --fp16 --mixed-precision none

# Run inference
python demo_multiclass.py --image x.jpg --classes person car dog \
    --checkpoint stage1_all_converted/efficient_sam3_repvit_l.pt \
    --efficient-backbone repvit --efficient-model m2_3 \
    --fast --detection-only \
    --trt-enc-dec enc_dec_repvit_c16_fp16.engine --trt-max-classes 16
```

Available checkpoints in `stage1_all_converted/`:

| Checkpoint | Backbone | Model flag |
|-----------|----------|------------|
| `efficient_sam3_efficientvit_m_geo_ft.pt` | EfficientViT-B1 | `--efficient-backbone efficientvit --efficient-model b1` |
| `efficient_sam3_tinyvit_m_geo_ft.pt` | TinyViT-11M | `--efficient-backbone tinyvit --efficient-model 11m` |
| `efficient_sam3_repvit_l.pt` | RepViT-M2.3 | `--efficient-backbone repvit --efficient-model m2_3` |

---

## Block Pruning

Two pruning granularities are supported: sub-block masking (skip individual
attention or MLP within a block) and full block removal (skip entire blocks).
Full block removal gives much better speed gains because TRT can eliminate the
blocks entirely from the engine.

### Analyze block importance

Measures each block's contribution by removing it and computing feature
reconstruction loss on calibration images:

```bash
python scripts/analyze_block_importance.py \
    --checkpoint sam3.pt --calib-dir train2017 \
    --num-images 20 --num-greedy 16 --imgsz 1008
```

Phase 1 ranks blocks individually. Phase 2 runs a greedy search that
iteratively removes the least-important block and reports cumulative loss,
cosine similarity, and estimated speedup.

### Sub-block pruning search

```bash
python scripts/block_pruner_search.py \
    --checkpoint sam3.pt --calib-dir train2017 \
    --num-images 16 --num-prune 16 --imgsz 1008
```

### Self-distillation for pruned backbone

After identifying blocks to remove, self-distillation recovers quality by
training the remaining blocks against the full backbone:

```bash
# Single GPU
python scripts/distill.py \
    --data-dir /path/to/coco/train2017 \
    --checkpoint sam3.pt \
    --phase prune \
    --skip-blocks "5,10,12,14,17,18,19,20,21,22,24,25,26,27,28,30" \
    --epochs 100 --batch-size 4 --lr 1e-4 \
    --output-dir skipblocks_distill

# 8xH100 via SLURM
srun --ntasks=1 torchrun --nproc_per_node=8 scripts/distill.py \
    --data-dir /path/to/coco/train2017 \
    --checkpoint sam3.pt \
    --phase prune \
    --skip-blocks "5,10,12,14,17,18,19,20,21,22,24,25,26,27,28,30" \
    --epochs 100 --batch-size 32 --lr 1e-4 \
    --output-dir skipblocks_distill
```

### Export pruned backbone to TRT

Use the HF export path for fused attention kernels:

```bash
PYTHONIOENCODING=utf-8 python scripts/export_hf_backbone.py \
    --image x.jpg \
    --output-onnx onnx_hf_backbone_1008_pruned/hf_backbone.onnx \
    --output-engine hf_backbone_1008_pruned_fp16.engine \
    --skip-blocks "5,10,12,14,17,18,19,20,21,22,24,25,26,27,28,30"
```

### Evaluate pruned backbone

```bash
PYTHONIOENCODING=utf-8 python scripts/eval_coco_official.py \
    --images-dir D:/val2017 \
    --ann-file D:/coco2017labels/coco/annotations/instances_val2017.json \
    --checkpoint sam3.pt \
    --pruned-checkpoint distilled/pruned_16blocks.pt \
    --configs "pruned16_1008=trt:hf_backbone_1008_pruned_fp16.engine;encdec:enc_dec_1008_c16_presence_fp16.engine;imgsz:1008"
```

### Recommended configs

```bash
# Full block removal, 16 blocks (1.8x backbone speedup, -2.2 AP)
--skip-blocks "5,10,12,14,17,18,19,20,21,22,24,25,26,27,28,30"

# Sub-block masking, 8 sub-blocks (minimal quality loss)
--mask-blocks "25:attn,28:mlp,27:attn,22:attn,28:attn,30:mlp,20:attn,27:mlp"

# Sub-block masking, 16 sub-blocks
--mask-blocks "25:attn,28:mlp,27:attn,22:attn,28:attn,30:mlp,20:attn,27:mlp,26:attn,22:mlp,24:attn,18:attn,20:mlp,21:attn,25:mlp,18:mlp"
```

The `--skip-blocks` or `--mask-blocks` flag must match between export
(`export_hf_backbone.py`) and inference (`demo_multiclass.py` / `demo_video.py`).

---

## Text Cache

Text embeddings can be cached to skip the text encoder on subsequent runs:

```bash
# First run: computes and saves embeddings
python demo_video.py --video input.mp4 \
    --classes person car bicycle \
    --checkpoint sam3.pt \
    --trt hf_backbone_1008_fp16.engine \
    --trt-enc-dec enc_dec_fp16.engine \
    --text-cache text_3classes.pt \
    --max-frames 1

# Subsequent runs: loads from cache (no checkpoint needed)
python demo_video.py --video input.mp4 \
    --classes person car bicycle \
    --trt hf_backbone_1008_fp16.engine \
    --trt-enc-dec enc_dec_fp16.engine \
    --text-cache text_3classes.pt \
    -o output.mp4
```

When both TRT engines and a text cache are provided, the full PyTorch model
is not loaded, eliminating ~20 s startup time.

---

## COCO Evaluation

### Detection AP (open-vocabulary)

Standard COCO AP (IoU 0.50–0.95) on val2017 (5,000 images, 80 classes):

```bash
PYTHONIOENCODING=utf-8 python scripts/eval_coco_official.py \
    --images-dir D:/val2017 \
    --ann-file D:/coco2017labels/coco/annotations/instances_val2017.json \
    --checkpoint sam3.pt \
    --configs "full_1008=trt:hf_backbone_1008_fp16.engine;encdec:enc_dec_1008_c16_fp16.engine;imgsz:1008"
```

| Configuration | Res. | AP | AP50 | AP_S | AP_L | ms/img |
|---------------|------|----|------|------|------|--------|
| **Full TRT FP16** | **1008** | **55.8** | **73.4** | **40.3** | **70.7** | 225 |
| SBP-16 TRT FP16 | 1008 | 47.6 | 63.5 | 32.5 | 62.0 | 220 |
| **Full TRT FP16** | **644** | **39.1** | **63.9** | 12.4 | 65.4 | 105 |
| SBP-16 TRT FP16 | 644 | 32.8 | 54.5 | 9.9 | 56.7 | 100 |

### Instance segmentation mIoU (GT-box-prompted)

Replicates the EfficientSAM3 official evaluation protocol (mask quality given
ground-truth boxes):

```bash
PYTHONIOENCODING=utf-8 python scripts/eval_cocoseg.py \
    --images-dir D:/val2017 \
    --ann-file D:/coco2017labels/coco/annotations/instances_val2017.json \
    --checkpoint sam3.pt
```

### Student backbone evaluation

```bash
PYTHONIOENCODING=utf-8 python scripts/eval_all_students.py
```

---

## Benchmarks

### FPS vs. class count (1008px, RTX 4080, TRT FP16)

| Classes | BB (ms) | E-D (ms) | Sequential FPS | Pipelined FPS |
|---------|---------|----------|----------------|---------------|
| 1 | 53 | 8 | 16.3 | 18.7 |
| 2 | 53 | 11 | 15.5 | 17.6 |
| 4 | 53 | 19 | 13.8 | 15.8 |
| 8 | 53 | 35 | 11.5 | 12.5 |

**15+ FPS** is achieved with up to 4 classes at 1008px. At 644px, all tested
class counts exceed 30 FPS.

### Student backbone speed (3 classes, 1008px, TRT FP16)

| Model | BB (ms) | Pipelined FPS | COCO AP |
|-------|---------|---------------|---------|
| EfficientViT-L1 | 10.4 | 64.2 | 16.3 |
| EfficientViT-L2 | 10.6 | 62.5 | 21.7 |
| TinyViT-21M | 12.0 | 57.8 | 30.1 |
| RepViT-M2.3 | 13.6 | 55.8 | 38.7 |
| ViT-H Pruned-16 | 26.6 | 37.6 | 53.6 |

### Reproduce benchmarks

```bash
# Video benchmark (sequential vs pipelined)
python scripts/benchmark_video.py --video input.mp4 --classes car person \
    --checkpoint sam3.pt \
    --trt hf_backbone_1008_fp16.engine --trt-enc-dec enc_dec_fp16.engine \
    --imgsz 1008 --max-frames 100

# Class-scaling benchmark (all student + teacher backbones)
PYTHONIOENCODING=utf-8 python scripts/benchmark_class_scaling.py

# All student backbones (sequential + pipelined)
PYTHONIOENCODING=utf-8 python scripts/benchmark_all_students.py
```

---

## FP16 Precision Analysis

The ViT-H backbone is vulnerable to FP16 accumulation error in TRT. Generic
FP16 MatMul rounding (~1e-4 per op) compounds through 32 residual blocks,
producing unusable features (cos = 0.058) unless attention dispatches to
accumulation-safe fused kernels.

The HuggingFace backbone export (`export_hf_backbone.py`) restructures
attention into canonical forms that TRT pattern-matches correctly → **cos >
0.999 at 53 ms**. This is the recommended path.

| Method | Latency | Cosine | Status |
|--------|---------|--------|--------|
| **Explicit-attn TRT FP16** | **53 ms** | **0.999** | recommended |
| Fused-SDPA TRT FP16 | 26 ms | 0.058 | broken |
| Fused-SDPA mixed (attn FP32) | 128 ms | 0.999 | correct but slow |
| torch.compile FP16 | 75 ms | 1.000 | correct |
| PyTorch eager FP16 | 87 ms | 1.000 | correct |

### Diagnose your GPU

```bash
python scripts/compare_backbone.py \
    --checkpoint sam3.pt --image x.jpg
```

### Detailed precision scripts

```bash
# Multi-strategy precision benchmark
python scripts/benchmark_fp16_precision.py --onnx backbone.onnx --checkpoint sam3.pt

# Block-level FP32 bisection
python scripts/bisect_blocks_fp32.py --onnx backbone.onnx --checkpoint sam3.pt
```

---

## Scripts Reference

### Demos

| Script | Description |
|--------|-------------|
| `demo_multiclass.py` | Single-image detection (all modes) |
| `demo_video.py` | Video inference with pipelining + tracking |
| `demo_efficientsam3.py` | EfficientSAM3 demo with lightweight backbones |

### Export and build

| Script | Description |
|--------|-------------|
| `scripts/export_hf_backbone.py` | Export HF backbone → ONNX → TRT FP16 |
| `python -m sam3.trt.export_enc_dec` | Export encoder-decoder → ONNX |
| `python -m sam3.trt.build_engine` | Build TRT engine from ONNX |
| `scripts/export_student_trt.py` | Export distilled student backbones |
| `scripts/export_efficient_backbone.py` | Export EfficientSAM3 backbones |
| `scripts/build_coco_engine.py` | One-command COCO 80-class setup |

### Evaluation

| Script | Description |
|--------|-------------|
| `scripts/eval_coco_official.py` | COCO val2017 detection AP (official protocol) |
| `scripts/eval_cocoseg.py` | Instance segmentation mIoU (GT-box-prompted) |
| `scripts/eval_all_students.py` | Evaluate all distilled student backbones |

### Benchmarks

| Script | Description |
|--------|-------------|
| `scripts/benchmark_video.py` | Sequential vs pipelined video benchmark |
| `scripts/benchmark_all_students.py` | All student backbones speed comparison |
| `scripts/benchmark_class_scaling.py` | FPS vs class count scaling |
| `scripts/compare_backbone.py` | Backbone speed + precision comparison |
| `scripts/benchmark_fp16_precision.py` | FP16 mixed-precision strategies |
| `scripts/bisect_blocks_fp32.py` | Block-level FP32 bisection |

### Training

| Script | Description |
|--------|-------------|
| `scripts/distill.py` | Train student backbone adapters |
| `scripts/block_pruner_search.py` | Calibrate sub-block pruning order |
| `scripts/analyze_block_importance.py` | Analyze full block importance and greedy removal |

---

## Troubleshooting

### `PYTHONIOENCODING=utf-8`

TRT, torch, and the detection scripts print Unicode characters (arrows, emoji)
that the default Windows console encoding (cp1252) can't encode. Always set
`PYTHONIOENCODING=utf-8` on Windows for **all** scripts, not just export scripts.

### torch.compile warmup takes 60–120 s

Expected for `max-autotune` (Triton autotuning). Use `--compile default` for
faster startup (~80 ms backbone, no autotuning). Warmup happens once per process.

### torch.compile fails with "Compiler: cl is not found"

`torch.compile` requires MSVC (`cl.exe`) on Windows. Install the
"Desktop development with C++" workload from Visual Studio Build Tools, or
use TRT engines instead (recommended for production speed).

### `AssertionError: num_classes=N exceeds max_classes=M`

The TRT enc-dec engine was built with fewer `--max-classes` than the number of
classes at runtime. Rebuild with higher `--max-classes`.

### CUDA out of memory during engine build

Reduce `--max-classes` (4 → ~4 GB, 8 → ~8 GB). Close other GPU processes.

### Wrong detections with TRT enc-dec

`--imgsz` at inference must match the `--imgsz` used during export and build.

### Backbone TRT gives cos < 0.5

FP16 accumulation issue. Use the HF export path (`export_hf_backbone.py`).
See [FP16 Precision Analysis](#fp16-precision-analysis).

### Engine not portable across GPUs or TRT versions

TRT engines are specific to both the GPU architecture and the TensorRT version
they were built with. ONNX files are portable — rebuild the engine on the
target GPU or after upgrading TensorRT.

### External ONNX data files

Dynamo export creates `model.onnx` + `model.onnx.data`. TRT's ONNX parser
must use `parse_from_file()` so it can find the external data relative to the
ONNX path. All provided scripts handle this automatically.

### Conv+Gelu TRT fusion bug (student backbones)

TRT 10.x has no FP16 implementation for fused `Conv+Gelu` kernels.
`export_student_trt.py` inserts Identity nodes to break the fusion pattern
automatically. If building manually, use FP32 or add Identity nodes.

## Citation

If you find DART useful in your research or applications, please cite it as:

```bibtex
@misc{turkcan2026detectrealtimesingleprompt,
      title={Detect Anything in Real Time: From Single-Prompt Segmentation to Multi-Class Detection},
      author={Mehmet Kerem Turkcan},
      year={2026},
      eprint={2603.11441},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2603.11441},
}
```
