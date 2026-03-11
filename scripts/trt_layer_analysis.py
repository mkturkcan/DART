#!/usr/bin/env python3
"""Analyze TRT layer types in a backbone ONNX model.

Parses the ONNX model into a TRT network and reports all layer types,
their counts, and names (for targeted precision overrides).

Usage:
    python scripts/trt_layer_analysis.py --onnx backbone.onnx
    python scripts/trt_layer_analysis.py --onnx backbone.onnx --dump-names
"""

import argparse
import sys
from collections import Counter

try:
    import tensorrt as trt
except ImportError:
    print("ERROR: tensorrt package is required")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Analyze TRT layer types in ONNX model")
    parser.add_argument("--onnx", required=True, help="ONNX model path")
    parser.add_argument(
        "--dump-names", action="store_true",
        help="Print all layer names grouped by type",
    )
    parser.add_argument(
        "--filter-type", type=str, default=None,
        help="Only show layers of this type (e.g. REDUCE, ELEMENTWISE)",
    )
    args = parser.parse_args()

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    trt_parser = trt.OnnxParser(network, logger)

    print(f"Parsing: {args.onnx}")
    with open(args.onnx, "rb") as f:
        if not trt_parser.parse(f.read()):
            for i in range(trt_parser.num_errors):
                print(f"  Error: {trt_parser.get_error(i)}")
            sys.exit(1)

    print(f"Total layers: {network.num_layers}")
    print(f"Inputs: {network.num_inputs}, Outputs: {network.num_outputs}")
    print()

    # Collect layer info
    type_counts = Counter()
    type_to_layers = {}
    for i in range(network.num_layers):
        layer = network.get_layer(i)
        type_name = str(layer.type).split(".")[-1]
        type_counts[type_name] += 1
        if type_name not in type_to_layers:
            type_to_layers[type_name] = []
        type_to_layers[type_name].append(layer.name)

    # Summary
    print("=== Layer Type Summary ===")
    for type_name, count in type_counts.most_common():
        print(f"  {type_name:30s} {count:5d}")
    print(f"  {'TOTAL':30s} {network.num_layers:5d}")

    # Identify LayerNorm-related layers
    # LayerNorm in ONNX = ReduceMean + Sub + Pow + ReduceMean + Add + Sqrt + Div + Mul + Add
    # In TRT, these become REDUCE, ELEMENTWISE, UNARY layers
    print()
    print("=== LayerNorm-related ops ===")
    norm_keywords = ["norm", "layernorm", "layer_norm", "ln", "ReduceMean", "Pow"]
    norm_layers = []
    for i in range(network.num_layers):
        layer = network.get_layer(i)
        name_lower = layer.name.lower()
        if any(kw.lower() in name_lower for kw in norm_keywords):
            norm_layers.append((str(layer.type).split(".")[-1], layer.name))

    if norm_layers:
        for type_name, name in norm_layers[:50]:
            print(f"  [{type_name:20s}] {name}")
        if len(norm_layers) > 50:
            print(f"  ... and {len(norm_layers) - 50} more")
        print(f"  Total norm-related: {len(norm_layers)}")
    else:
        print("  No layers with 'norm' in name found.")
        print("  Checking REDUCE/UNARY layers (typical LayerNorm decomposition):")
        for type_name in ["REDUCE", "UNARY", "ELEMENTWISE"]:
            if type_name in type_to_layers:
                for name in type_to_layers[type_name][:10]:
                    print(f"    [{type_name:20s}] {name}")
                if len(type_to_layers[type_name]) > 10:
                    print(f"    ... and {len(type_to_layers[type_name]) - 10} more")

    if args.dump_names:
        print()
        print("=== All Layers by Type ===")
        for type_name in sorted(type_to_layers.keys()):
            if args.filter_type and args.filter_type.upper() != type_name:
                continue
            layers = type_to_layers[type_name]
            print(f"\n--- {type_name} ({len(layers)}) ---")
            for name in layers:
                print(f"  {name}")

    # Specific analysis: which layer types are "compute" (affected by FP16)
    print()
    print("=== Compute layers (affected by FP16 precision) ===")
    compute_types = [
        "MATRIX_MULTIPLY", "CONVOLUTION", "DECONVOLUTION",
        "ELEMENTWISE", "REDUCE", "UNARY", "SOFTMAX",
        "SCALE", "NORMALIZATION",
    ]
    for ct in compute_types:
        if ct in type_counts:
            print(f"  {ct:30s} {type_counts[ct]:5d}")


if __name__ == "__main__":
    main()
