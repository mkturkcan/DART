#!/usr/bin/env python3
"""Benchmark video latency vs number of classes for all backbones.

Runs benchmark_video.py in both sequential and pipelined modes for each
combination of {1, 2, 4, 8, 16} classes × {4 students + teacher ViT-H}
= 25 runs. Saves structured JSON results for plotting.
"""

import json
import os
import re
import subprocess
import sys
import time

PYTHON = r"C:\Users\mehme\anaconda3\envs\sam3\python.exe"

# ── Backbones ──────────────────────────────────────────────────────────
BACKBONES = [
    {"name": "efficientvit_l1", "engine": "student_efficientvit_l1_fp16.engine"},
    {"name": "efficientvit_l2", "engine": "student_efficientvit_l2_fp16.engine"},
    {"name": "repvit_m2_3",     "engine": "student_repvit_m2_3_fp16.engine"},
    {"name": "tiny_vit_21m",    "engine": "student_tiny_vit_21m_fp16.engine"},
    {"name": "vit_h_teacher",   "engine": "hf_backbone_1008_fp16.engine"},
]

# ── Class configurations ───────────────────────────────────────────────
# Pick diverse COCO classes so text embeddings are distinct
CLASS_LISTS = {
    1:  ["person"],
    2:  ["person", "car"],
    4:  ["person", "car", "dog", "bicycle"],
    8:  ["person", "car", "dog", "bicycle", "chair", "bottle", "laptop", "cat"],
    16: ["person", "car", "dog", "bicycle", "chair", "bottle", "laptop", "cat",
         "bus", "train", "airplane", "boat", "backpack", "umbrella", "handbag", "skateboard"],
}

# ── Enc-dec engines for each class count ───────────────────────────────
# max_classes must be >= num_classes; single pass when num_classes <= max_classes
ENC_DEC_CONFIG = {
    1:  {"engine": "enc_dec_1class_fp16.engine",        "max_classes": 1},
    2:  {"engine": "enc_dec_2class_fp16.engine",        "max_classes": 2},
    4:  {"engine": "enc_dec_4class_fp16.engine",        "max_classes": 4},
    8:  {"engine": "enc_dec_1008_c16_fp16_16.engine",   "max_classes": 16},
    16: {"engine": "enc_dec_1008_c16_fp16_16.engine",   "max_classes": 16},
}


def parse_section(lines: list[str]) -> dict:
    """Extract timing metrics from one section of benchmark_video.py output."""
    metrics = {}
    for line in lines:
        line = line.strip()
        if "Steady-state FPS:" in line:
            metrics["fps"] = float(line.split(":")[-1].strip())
        if "Total avg:" in line:
            m = re.search(r"([\d.]+)\s*ms", line)
            if m:
                metrics["total_ms"] = float(m.group(1))
        if "Backbone avg:" in line:
            m = re.search(r"([\d.]+)\s*ms", line)
            if m:
                metrics["backbone_ms"] = float(m.group(1))
        if "Enc-dec avg:" in line:
            m = re.search(r"([\d.]+)\s*ms", line)
            if m:
                metrics["encdec_ms"] = float(m.group(1))
    return metrics


def parse_output(stdout: str) -> dict:
    """Extract sequential and pipelined metrics from --mode both output."""
    lines = stdout.split("\n")

    # Find section boundaries
    seq_start = pipe_start = None
    for i, line in enumerate(lines):
        if "SEQUENTIAL:" in line:
            seq_start = i
        if "PIPELINED:" in line:
            pipe_start = i

    result = {}
    if seq_start is not None:
        end = pipe_start if pipe_start is not None else len(lines)
        seq = parse_section(lines[seq_start:end])
        result["seq_fps"] = seq.get("fps", 0)
        result["seq_total_ms"] = seq.get("total_ms", 0)
        result["backbone_ms"] = seq.get("backbone_ms", 0)
        result["seq_encdec_ms"] = seq.get("encdec_ms", 0)

    if pipe_start is not None:
        pipe = parse_section(lines[pipe_start:])
        result["pipe_fps"] = pipe.get("fps", 0)
        result["pipe_total_ms"] = pipe.get("total_ms", 0)
        if "backbone_ms" not in result:
            result["backbone_ms"] = pipe.get("backbone_ms", 0)
        result["pipe_encdec_ms"] = pipe.get("encdec_ms", 0)

    return result


def run_one(backbone: dict, n_classes: int) -> dict:
    """Run a single benchmark and return parsed results."""
    enc_dec = ENC_DEC_CONFIG[n_classes]
    classes = CLASS_LISTS[n_classes]

    cmd = [
        PYTHON, "scripts/benchmark_video.py",
        "--video", "input.mp4",
        "--classes", *classes,
        "--checkpoint", "sam3.pt",
        "--trt", backbone["engine"],
        "--trt-enc-dec", enc_dec["engine"],
        "--trt-max-classes", str(enc_dec["max_classes"]),
        "--imgsz", "1008",
        "--max-frames", "100",
        "--mode", "both",
    ]

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    t0 = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    wall_s = time.perf_counter() - t0

    metrics = parse_output(result.stdout)
    metrics["backbone"] = backbone["name"]
    metrics["n_classes"] = n_classes
    metrics["wall_s"] = wall_s

    if result.returncode != 0:
        metrics["error"] = True
        stderr_tail = "\n".join(result.stderr.strip().split("\n")[-10:])
        metrics["stderr_tail"] = stderr_tail

    return metrics


def warmup_gpu():
    """Run a short throwaway benchmark to warm up GPU and TRT context."""
    print("Warming up GPU (throwaway run)...")
    enc_dec = ENC_DEC_CONFIG[1]
    cmd = [
        PYTHON, "scripts/benchmark_video.py",
        "--video", "input.mp4",
        "--classes", "person",
        "--checkpoint", "sam3.pt",
        "--trt", BACKBONES[0]["engine"],
        "--trt-enc-dec", enc_dec["engine"],
        "--trt-max-classes", str(enc_dec["max_classes"]),
        "--imgsz", "1008",
        "--max-frames", "20",
        "--mode", "sequential",
    ]
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    subprocess.run(cmd, capture_output=True, text=True, env=env)
    print("Warmup done.\n")


def main():
    warmup_gpu()

    all_results = []
    total = len(BACKBONES) * len(CLASS_LISTS)
    idx = 0

    for bb in BACKBONES:
        for n_classes in sorted(CLASS_LISTS.keys()):
            idx += 1
            tag = f"[{idx}/{total}] {bb['name']} × {n_classes} classes"
            print(f"\n{'='*70}")
            print(f"  {tag}")
            print(f"{'='*70}")

            metrics = run_one(bb, n_classes)
            all_results.append(metrics)

            bb_ms = metrics.get("backbone_ms", 0)
            seq_ms = metrics.get("seq_total_ms", 0)
            seq_fps = metrics.get("seq_fps", 0)
            pipe_ms = metrics.get("pipe_total_ms", 0)
            pipe_fps = metrics.get("pipe_fps", 0)
            err = " [ERROR]" if metrics.get("error") else ""
            print(f"  BB={bb_ms:.1f}ms  Seq={seq_ms:.1f}ms({seq_fps:.1f}FPS)  "
                  f"Pipe={pipe_ms:.1f}ms({pipe_fps:.1f}FPS){err}")

            if metrics.get("error"):
                print(f"  stderr: {metrics.get('stderr_tail', '')[:300]}")

    # Save raw results
    out_path = "benchmark_class_scaling.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Group by backbone
    class_counts = sorted(CLASS_LISTS.keys())
    by_bb = {}
    for r in all_results:
        by_bb.setdefault(r["backbone"], {})[r["n_classes"]] = r

    bb_names = [b["name"] for b in BACKBONES]

    def print_table(title, key):
        print(f"\n{'='*90}")
        print(title)
        print(f"{'='*90}")
        header = f"  {'Backbone':<20s}"
        for nc in class_counts:
            header += f"  {nc:>2d}cls"
        print(header)
        print(f"  {'-'*80}")
        for bb_name in bb_names:
            row = f"  {bb_name:<20s}"
            for nc in class_counts:
                m = by_bb.get(bb_name, {}).get(nc, {})
                val = m.get(key, 0)
                row += f"  {val:>8.1f}"
            print(row)
        print(f"{'='*90}")

    print_table("SEQUENTIAL — ms/frame (1008px, 100 frames)", "seq_total_ms")
    print_table("SEQUENTIAL — FPS", "seq_fps")
    print_table("PIPELINED — ms/frame (1008px, 100 frames)", "pipe_total_ms")
    print_table("PIPELINED — FPS", "pipe_fps")

    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
