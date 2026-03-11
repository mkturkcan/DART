#!/usr/bin/env python3
"""Compare ONNX graph structures between SDPA and eager backbone exports."""

import onnx
from collections import Counter

for name in ["backbone.onnx", "backbone_eager2.onnx"]:
    model = onnx.load(name)
    graph = model.graph
    ops = Counter(n.op_type for n in graph.node)
    print(f"{name}: {len(graph.node)} nodes")
    for op, cnt in ops.most_common(20):
        print(f"  {op}: {cnt}")
    print()

# Compare
m1 = onnx.load("backbone.onnx")
m2 = onnx.load("backbone_eager2.onnx")
ops1 = Counter(n.op_type for n in m1.graph.node)
ops2 = Counter(n.op_type for n in m2.graph.node)

all_ops = set(ops1.keys()) | set(ops2.keys())
print("Differences:")
for op in sorted(all_ops):
    c1, c2 = ops1.get(op, 0), ops2.get(op, 0)
    if c1 != c2:
        print(f"  {op}: {c1} -> {c2} ({c2-c1:+d})")
