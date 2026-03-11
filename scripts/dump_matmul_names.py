#!/usr/bin/env python3
"""Dump all MATRIX_MULTIPLY layer names from TRT network to identify attention vs projection."""

import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

builder = trt.Builder(TRT_LOGGER)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(network, TRT_LOGGER)

with open("backbone.onnx", "rb") as f:
    ok = parser.parse(f.read())
    if not ok:
        for i in range(parser.num_errors):
            print(f"  Error: {parser.get_error(i)}")
        raise RuntimeError("Failed to parse ONNX")

matmul_type = getattr(trt.LayerType, "MATRIX_MULTIPLY")
softmax_type = getattr(trt.LayerType, "SOFTMAX", None)

print(f"Total layers: {network.num_layers}")
print(f"\nMATRIX_MULTIPLY layers ({sum(1 for i in range(network.num_layers) if network.get_layer(i).type == matmul_type)}):")
for i in range(network.num_layers):
    layer = network.get_layer(i)
    if layer.type == matmul_type:
        inputs = []
        for j in range(layer.num_inputs):
            inp = layer.get_input(j)
            if inp is not None:
                inputs.append(f"{inp.name}:{list(inp.shape)}")
        outputs = []
        for j in range(layer.num_outputs):
            out = layer.get_output(j)
            if out is not None:
                outputs.append(f"{out.name}:{list(out.shape)}")
        print(f"  [{i:5d}] {layer.name}")
        print(f"         inputs:  {inputs}")
        print(f"         outputs: {outputs}")

if softmax_type:
    count = sum(1 for i in range(network.num_layers) if network.get_layer(i).type == softmax_type)
    print(f"\nSOFTMAX layers ({count}):")
    for i in range(network.num_layers):
        layer = network.get_layer(i)
        if layer.type == softmax_type:
            print(f"  [{i:5d}] {layer.name}")
