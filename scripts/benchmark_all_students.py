#!/usr/bin/env python3
"""Benchmark video (sequential vs pipelined) for all 4 distilled student models."""

import json
import subprocess
import sys
import time

PYTHON = r"C:\Users\mehme\anaconda3\envs\sam3\python.exe"

MODELS = [
    {
        "name": "efficientvit_l1",
        "engine": "student_efficientvit_l1_fp16.engine",
    },
    {
        "name": "efficientvit_l2",
        "engine": "student_efficientvit_l2_fp16.engine",
    },
    {
        "name": "repvit_m2_3",
        "engine": "student_repvit_m2_3_fp16.engine",
    },
    {
        "name": "tiny_vit_21m",
        "engine": "student_tiny_vit_21m_fp16.engine",
    },
]

all_results = {}

for m in MODELS:
    print(f"\n{'='*70}")
    print(f"  BENCHMARKING: {m['name']}")
    print(f"{'='*70}")
    t0 = time.perf_counter()

    cmd = [
        PYTHON, "scripts/benchmark_video.py",
        "--video", "input.mp4",
        "--classes", "person", "car", "dog",
        "--checkpoint", "sam3.pt",
        "--trt", m["engine"],
        "--trt-enc-dec", "enc_dec_3class_fp16.engine",
        "--trt-max-classes", "4",
        "--imgsz", "1008",
        "--max-frames", "100",
        "--mode", "both",
    ]

    import os
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    result = subprocess.run(
        cmd, capture_output=True, text=True, env=env,
    )

    dt = time.perf_counter() - t0
    print(result.stdout[-3000:] if len(result.stdout) > 3000 else result.stdout)
    if result.stderr:
        # Print only last few lines of stderr (warnings)
        stderr_lines = result.stderr.strip().split('\n')
        if len(stderr_lines) > 5:
            print(f"  ... ({len(stderr_lines)} stderr lines, showing last 5)")
            for line in stderr_lines[-5:]:
                print(f"  {line}")

    # Parse output for key metrics
    stats = {"name": m["name"], "time_s": dt}
    for line in result.stdout.split('\n'):
        line = line.strip()
        if 'Steady-state FPS:' in line:
            fps = float(line.split(':')[-1].strip())
            if 'sequential' not in all_results.get(m['name'], {}):
                stats['sequential_fps'] = fps
            else:
                stats['pipelined_fps'] = fps
        if 'Total avg:' in line:
            ms = float(line.split(':')[-1].strip().replace('ms/frame', '').strip())
            if 'sequential_ms' not in stats:
                stats['sequential_ms'] = ms
            else:
                stats['pipelined_ms'] = ms
        if 'Backbone avg:' in line:
            ms = float(line.split(':')[-1].strip().replace('ms', '').strip())
            stats['backbone_ms'] = ms
        if 'Enc-dec avg:' in line:
            ms = float(line.split(':')[-1].strip().replace('ms', '').strip())
            stats['encdec_ms'] = ms

    all_results[m['name']] = stats
    print(f"\n  {m['name']} completed in {dt:.0f}s")

# Save results
with open("benchmark_all_students.json", "w") as f:
    json.dump(all_results, f, indent=2)

# Print summary table
print(f"\n\n{'='*80}")
print(f"VIDEO BENCHMARK RESULTS (3 classes, 1008px, 100 frames)")
print(f"{'='*80}")
header = (
    f"  {'Model':<18s}  {'BB(ms)':>7s}  {'EncDec':>7s}  "
    f"{'Seq(ms)':>8s}  {'SeqFPS':>7s}  "
    f"{'Pipe(ms)':>8s}  {'PipeFPS':>8s}  {'Speedup':>8s}"
)
print(header)
print(f"  {'-'*76}")
for name, s in all_results.items():
    seq_ms = s.get('sequential_ms', 0)
    pipe_ms = s.get('pipelined_ms', 0)
    seq_fps = s.get('sequential_fps', 0)
    pipe_fps = s.get('pipelined_fps', 0)
    bb_ms = s.get('backbone_ms', 0)
    ed_ms = s.get('encdec_ms', 0)
    speedup = seq_ms / pipe_ms if pipe_ms > 0 else 0
    print(
        f"  {name:<18s}  {bb_ms:>7.1f}  {ed_ms:>7.1f}  "
        f"{seq_ms:>8.1f}  {seq_fps:>7.1f}  "
        f"{pipe_ms:>8.1f}  {pipe_fps:>8.1f}  {speedup:>7.2f}x"
    )
print(f"{'='*80}")
print(f"\nResults saved to benchmark_all_students.json")
