# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""Build a TensorRT engine from a SAM3 ONNX model.

Supports two engine types:
  - ``backbone`` (default): ViT-H backbone (1 input, 3 FPN outputs)
  - ``enc-dec``: encoder+decoder+scoring pipeline (4 inputs, 2 outputs)

Supports FP32, FP16, and INT8 precision.  INT8 requires a calibration image
directory (e.g., COCO train2017/).

Usage:
    # Backbone FP16 (recommended default):
    python -m sam3.trt.build_engine \\
        --onnx backbone.onnx --output backbone_fp16.engine --fp16

    # Backbone INT8 with COCO calibration:
    python -m sam3.trt.build_engine \\
        --onnx backbone.onnx --output backbone_int8.engine --int8 \\
        --calib-images ./train2017 --calib-count 512

    # Encoder-decoder INT8:
    python -m sam3.trt.build_engine \\
        --onnx enc_dec.onnx --output enc_dec_int8.engine --int8 \\
        --calib-images ./train2017 --type enc-dec \\
        --checkpoint path/to/sam3.pt
"""

import argparse
import sys
from pathlib import Path

try:
    import tensorrt as trt
except ImportError:
    print("ERROR: tensorrt package is required. Install with:")
    print("  pip install tensorrt")
    sys.exit(1)


TRT_LOGGER = trt.Logger(trt.Logger.INFO)


def _set_precision_constraints(config):
    """Set the strictest available precision constraint flag. Returns True if set."""
    for flag_name in (
        "OBEY_PRECISION_CONSTRAINTS",
        "PREFER_PRECISION_CONSTRAINTS",
        "STRICT_TYPES",
    ):
        if hasattr(trt.BuilderFlag, flag_name):
            config.set_flag(getattr(trt.BuilderFlag, flag_name))
            print(f"  Precision constraint flag: {flag_name}")
            return True
    print("  WARNING: No precision constraint flag available in this TRT version")
    return False


def _get_skip_types():
    """Return set of TRT layer types that should not have precision overrides."""
    skip_types = set()
    for type_name in (
        "SHAPE", "CONSTANT", "IDENTITY", "SHUFFLE", "GATHER",
        "SLICE", "SQUEEZE", "UNSQUEEZE", "CONCATENATION", "CONDITION",
        "CAST", "ASSERTION", "FILL", "SCATTER", "RESIZE",
        "NON_ZERO", "ONE_HOT", "GRID_SAMPLE",
        "CONDITIONAL_INPUT", "CONDITIONAL_OUTPUT",
    ):
        if hasattr(trt.LayerType, type_name):
            skip_types.add(getattr(trt.LayerType, type_name))
    return skip_types


def _apply_mixed_precision(network, config, mode="attention"):
    """Force attention-critical layers to FP32 for numerical stability.

    The ONNX graph exported by torch.onnx.export includes complex ops for RoPE
    and window partitioning that prevent TRT from fusing attention into its
    optimized MHA kernel with FP32 accumulation. Without this fusion, TRT's
    generic FP16 MatMul has slightly lower precision than PyTorch's (which
    guarantees FP32 accumulation). This tiny per-op difference compounds
    catastrophically through 32 ViT blocks (cosine 0.07 vs PyTorch).

    Modes (from most surgical to most conservative):
      - "attn-v-only": Forces only attn@V MatMul (the one with large inner dim
        = sequence length) + Softmax + Norm to FP32 across all 32 blocks.
        Q@K^T, QKV proj, output proj, and MLP all stay FP16. ~96 FP32 layers.
      - "global-attn": Forces all compute in the 4 global attention blocks
        (7, 15, 23, 31) to FP32. These blocks see 5184 tokens (vs 576 for
        windowed blocks) and suffer the worst FP16 accumulation error.
        The 28 windowed blocks stay entirely FP16.
      - "norm-only": Forces only NORMALIZATION (LayerNorm) + SOFTMAX to FP32.
        All MatMul stays FP16. ~97 FP32 layers. Fastest mixed-precision option.
      - "norm-softmax-reduce": Forces NORMALIZATION + SOFTMAX + REDUCE to FP32.
        Targets all statistical/normalization ops while keeping compute in FP16.
      - "attn-core": Forces Q@K^T + attn@V MatMul + Softmax + Norm to FP32.
        Keeps QKV/proj/MLP MatMul in FP16. ~161 FP32 layers.
      - "attention" (default): Forces QKV projection, attention MatMul, proj,
        and Softmax + Norm to FP32. MLP MatMuls stay FP16. ~225 FP32 layers.
      - "all": Forces ALL MatMul + Softmax + Norm to FP32 including MLP.
        ~289 FP32 layers, cosine 1.0000.
    """
    if not _set_precision_constraints(config):
        return

    skip_types = _get_skip_types()
    softmax_type = getattr(trt.LayerType, "SOFTMAX", None)
    matmul_type = getattr(trt.LayerType, "MATRIX_MULTIPLY", None)
    norm_type = getattr(trt.LayerType, "NORMALIZATION", None)
    reduce_type = getattr(trt.LayerType, "REDUCE", None)

    fp32_count = 0
    fp16_count = 0
    skip_count = 0
    for i in range(network.num_layers):
        layer = network.get_layer(i)
        if layer.type in skip_types:
            skip_count += 1
            continue

        force_fp32 = False

        if mode == "attn-v-only":
            # Only attn@V (MatMul_1) + Softmax + Norm to FP32
            # attn@V has inner dim = sequence length (5184 global, 576 windowed)
            # and is the primary source of FP16 accumulation error
            if layer.type == softmax_type or layer.type == norm_type:
                force_fp32 = True
            elif layer.type == matmul_type:
                name = layer.name
                # MatMul_1 in attention = attn_weights @ V
                # (MatMul without _1 = Q @ K^T, inner dim only 64)
                if "/attn/MatMul_1" in name:
                    force_fp32 = True
        elif mode == "global-attn":
            # Force all compute in global attention blocks to FP32
            # Global blocks (7, 15, 23, 31) see 5184 tokens vs 576 for windowed
            _GLOBAL_BLOCKS = ("blocks.7/", "blocks.15/", "blocks.23/", "blocks.31/")
            is_global = any(blk in layer.name for blk in _GLOBAL_BLOCKS)
            if is_global and layer.type in (matmul_type, softmax_type, norm_type, reduce_type):
                force_fp32 = True
        elif mode == "norm-only":
            # Only LayerNorm + Softmax to FP32
            if layer.type == norm_type or layer.type == softmax_type:
                force_fp32 = True
        elif mode == "norm-softmax-reduce":
            # LayerNorm + Softmax + Reduce to FP32
            if layer.type in (norm_type, softmax_type, reduce_type):
                force_fp32 = True
        elif mode == "attn-core":
            # Only Q@K^T and attn@V matmul (not QKV/proj/MLP) + Softmax + Norm
            if layer.type == softmax_type or layer.type == norm_type:
                force_fp32 = True
            elif layer.type == matmul_type:
                name = layer.name
                # attn/MatMul = Q@K^T, attn/MatMul_1 = attn@V
                # Exclude: qkv/MatMul, proj/MatMul, fc1/MatMul, fc2/MatMul
                is_attn_core = (
                    "/attn/MatMul" in name
                    and "/qkv/" not in name
                    and "/proj/" not in name
                )
                if is_attn_core:
                    force_fp32 = True
        elif mode in ("attention", "all"):
            if layer.type == softmax_type:
                force_fp32 = True
            elif layer.type == norm_type:
                force_fp32 = True
            elif layer.type == matmul_type:
                if mode == "all":
                    force_fp32 = True
                elif mode == "attention":
                    name = layer.name
                    is_mlp = "fc1" in name or "fc2" in name or "mlp" in name
                    if not is_mlp:
                        force_fp32 = True

        if force_fp32:
            layer.precision = trt.float32
            for j in range(layer.num_outputs):
                layer.set_output_type(j, trt.float32)
            fp32_count += 1
        else:
            fp16_count += 1

    print(
        f"  Mixed precision ({mode}): {fp32_count} FP32 / "
        f"{fp16_count} FP16 / {skip_count} skip "
        f"(of {network.num_layers} total)"
    )


def _apply_layer_precisions(network, config, layer_precisions_str):
    """Force specific named layers to a given precision.

    Format: "layer_name1:fp32,layer_name2:fp32,..." or use wildcards:
    "*/norm*:fp32" to match all layers containing "norm".

    This mimics trtexec's --layerPrecisions flag.
    """
    import fnmatch

    if not _set_precision_constraints(config):
        return

    # Parse layer_precisions string
    rules = []
    for item in layer_precisions_str.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            print(f"  WARNING: Invalid layer precision spec (no ':'): {item}")
            continue
        pattern, prec_str = item.rsplit(":", 1)
        prec_str = prec_str.strip().lower()
        if prec_str == "fp32":
            prec = trt.float32
        elif prec_str == "fp16":
            prec = trt.float16
        else:
            print(f"  WARNING: Unknown precision '{prec_str}', skipping: {item}")
            continue
        rules.append((pattern.strip(), prec))

    print(f"  Layer precision rules: {len(rules)}")
    for pattern, prec in rules:
        prec_name = "FP32" if prec == trt.float32 else "FP16"
        print(f"    {pattern} -> {prec_name}")

    matched_count = 0
    for i in range(network.num_layers):
        layer = network.get_layer(i)
        for pattern, prec in rules:
            if fnmatch.fnmatch(layer.name, pattern):
                layer.precision = prec
                for j in range(layer.num_outputs):
                    layer.set_output_type(j, prec)
                matched_count += 1
                break

    print(f"  Matched {matched_count} layers")


def _apply_all_fp32_except_conv(network, config):
    """Force ALL compute layers to FP32 except CONVOLUTION.

    This is a diagnostic mode to test if any non-Conv FP16 layer causes
    numerical issues. Only convolutions remain in FP16 for speed.
    """
    if not _set_precision_constraints(config):
        return

    conv_types = set()
    for type_name in ("CONVOLUTION", "DECONVOLUTION"):
        if hasattr(trt.LayerType, type_name):
            conv_types.add(getattr(trt.LayerType, type_name))

    skip_types = _get_skip_types()

    fp32_count = 0
    fp16_count = 0
    skip_count = 0
    for i in range(network.num_layers):
        layer = network.get_layer(i)
        if layer.type in skip_types:
            skip_count += 1
        elif layer.type in conv_types:
            layer.precision = trt.float16
            for j in range(layer.num_outputs):
                layer.set_output_type(j, trt.float16)
            fp16_count += 1
        else:
            layer.precision = trt.float32
            for j in range(layer.num_outputs):
                layer.set_output_type(j, trt.float32)
            fp32_count += 1

    print(
        f"  All-FP32 (except Conv): {fp32_count} FP32 / {fp16_count} FP16 / "
        f"{skip_count} skip (of {network.num_layers} total)"
    )


def _list_layers(onnx_path: str):
    """Parse ONNX and print all TRT layer names with types."""
    from collections import Counter

    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, TRT_LOGGER)

    print(f"Parsing ONNX model: {onnx_path}")
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"  ONNX parse error: {parser.get_error(i)}")
            raise RuntimeError("Failed to parse ONNX model")

    type_counts = Counter()
    print(f"\nTRT network: {network.num_layers} layers\n")
    for i in range(network.num_layers):
        layer = network.get_layer(i)
        type_name = layer.type.name
        type_counts[type_name] += 1
        print(f"  [{i:4d}] {type_name:<24s} {layer.name}")

    print(f"\nLayer type summary:")
    for type_name, count in type_counts.most_common():
        print(f"  {type_name:<24s} {count}")


def build_engine(
    onnx_path: str,
    output_path: str,
    engine_type: str = "backbone",
    fp16: bool = True,
    int8: bool = False,
    calib_images: str = None,
    calib_count: int = 512,
    calib_cache: str = "calibration.cache",
    workspace_gb: float = 4.0,
    checkpoint: str = None,
    max_classes: int = 4,
    opt_level: int = 3,
    mixed_precision: str = None,
    layer_precisions: str = None,
):
    """Parse ONNX and build a serialized TensorRT engine."""

    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # Parse ONNX
    print(f"Parsing ONNX model: {onnx_path}")
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"  ONNX parse error: {parser.get_error(i)}")
            raise RuntimeError("Failed to parse ONNX model")

    # Print network info
    print(f"  Inputs: {network.num_inputs}")
    for i in range(network.num_inputs):
        inp = network.get_input(i)
        print(f"    {inp.name}: {inp.shape} ({inp.dtype})")
    print(f"  Outputs: {network.num_outputs}")
    for i in range(network.num_outputs):
        out = network.get_output(i)
        print(f"    {out.name}: {out.shape} ({out.dtype})")

    # Build config
    config = builder.create_builder_config()
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE, int(workspace_gb * (1 << 30))
    )

    # Builder optimization level (0=fastest build/least optimization,
    # 5=slowest build/most optimization).  Lower levels prevent aggressive
    # kernel fusion that can cause OOM on large transformer graphs.
    if hasattr(config, "builder_optimization_level"):
        config.builder_optimization_level = opt_level
        print(f"Builder optimization level: {opt_level}")

    if fp16:
        if not builder.platform_has_fast_fp16:
            print("WARNING: GPU does not have fast FP16 support")
        config.set_flag(trt.BuilderFlag.FP16)
        print("Enabled FP16 precision")

        # Auto-apply mixed precision for backbone FP16 unless user specified
        # a mode.  Enc-dec is shallow enough that pure FP16 is fine.
        if mixed_precision == "none":
            mixed_precision = None
        elif mixed_precision is None and layer_precisions is None:
            if engine_type == "backbone":
                mixed_precision = "sensitive"
                print("  Auto-applying mixed precision (sensitive) for backbone. "
                      "Use --mixed-precision none for pure FP16.")
            # enc-dec: no auto-apply, pure FP16 is fine

    if int8:
        if not builder.platform_has_fast_int8:
            print("WARNING: GPU does not have fast INT8 support")
        config.set_flag(trt.BuilderFlag.INT8)

        if calib_images is None:
            raise ValueError(
                "--calib-images is required for INT8 calibration"
            )

        if engine_type == "enc-dec":
            # Encoder-decoder calibrator needs the full model
            if checkpoint is None:
                raise ValueError(
                    "--checkpoint is required for enc-dec INT8 calibration"
                )

            from sam3.model_builder import build_sam3_image_model
            from sam3.trt.calibrator_enc_dec import EncDecCalibrator

            print(f"Loading SAM3 model for calibration: {checkpoint}")
            calib_model = build_sam3_image_model(
                checkpoint_path=checkpoint,
                device="cuda",
                eval_mode=True,
                load_from_HF=False,
                enable_segmentation=False,
            )

            calibrator = EncDecCalibrator(
                model=calib_model,
                image_dir=calib_images,
                max_classes=max_classes,
                num_images=calib_count,
                cache_file=calib_cache,
            )
        else:
            from sam3.trt.calibrator import CocoCalibrator

            calibrator = CocoCalibrator(
                image_dir=calib_images,
                num_images=calib_count,
                cache_file=calib_cache,
            )

        config.int8_calibrator = calibrator
        print(f"Enabled INT8 precision with {calib_count} calibration images")

    # Apply layer-name-based precision overrides (highest priority)
    if layer_precisions:
        _apply_layer_precisions(network, config, layer_precisions)
    # Apply mixed precision mode
    elif mixed_precision == "all-fp32":
        _apply_all_fp32_except_conv(network, config)
    elif mixed_precision in ("sensitive", "attention"):
        _apply_mixed_precision(network, config, mode="attention")
    elif mixed_precision == "all-matmul":
        _apply_mixed_precision(network, config, mode="all")
    elif mixed_precision == "norm-only":
        _apply_mixed_precision(network, config, mode="norm-only")
    elif mixed_precision == "norm-softmax-reduce":
        _apply_mixed_precision(network, config, mode="norm-softmax-reduce")
    elif mixed_precision == "attn-core":
        _apply_mixed_precision(network, config, mode="attn-core")
    elif mixed_precision == "attn-v-only":
        _apply_mixed_precision(network, config, mode="attn-v-only")
    elif mixed_precision == "global-attn":
        _apply_mixed_precision(network, config, mode="global-attn")

    # Build engine
    print("Building TensorRT engine (this may take several minutes) ...")
    serialized_engine = builder.build_serialized_network(network, config)

    if serialized_engine is None:
        raise RuntimeError("Engine build failed")

    # Save
    with open(output_path, "wb") as f:
        f.write(serialized_engine)

    size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    print(f"Done. Engine saved: {output_path} ({size_mb:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(
        description="Build TensorRT engine from SAM3 ONNX model"
    )
    parser.add_argument(
        "--onnx", required=True, help="Input ONNX model path"
    )
    parser.add_argument(
        "--output", default="backbone.engine", help="Output engine file path"
    )
    parser.add_argument(
        "--type",
        choices=["backbone", "enc-dec"],
        default="backbone",
        help="Engine type: 'backbone' (ViT-H) or 'enc-dec' (encoder+decoder+scoring)",
    )
    parser.add_argument(
        "--fp16", action="store_true", help="Enable FP16 precision"
    )
    parser.add_argument(
        "--int8", action="store_true", help="Enable INT8 precision"
    )
    parser.add_argument(
        "--fp32", action="store_true", help="Force FP32 precision (no FP16/INT8)"
    )
    parser.add_argument(
        "--calib-images",
        help="Directory with calibration images (required for INT8)",
    )
    parser.add_argument(
        "--calib-count",
        type=int,
        default=512,
        help="Number of calibration images to use",
    )
    parser.add_argument(
        "--calib-cache",
        default="calibration.cache",
        help="Path to calibration cache file",
    )
    parser.add_argument(
        "--workspace",
        type=float,
        default=4.0,
        help="TensorRT workspace size in GB",
    )
    parser.add_argument(
        "--checkpoint",
        help="SAM3 checkpoint path (required for enc-dec INT8 calibration)",
    )
    parser.add_argument(
        "--max-classes",
        type=int,
        default=4,
        help="Max classes for enc-dec engine (batch dimension)",
    )
    parser.add_argument(
        "--opt-level",
        type=int,
        default=3,
        choices=[0, 1, 2, 3, 4, 5],
        help="Builder optimization level (0=fastest build, 5=max optimization). "
             "Try 0 if engine build fails with OOM on large transformer models.",
    )
    parser.add_argument(
        "--mixed-precision",
        choices=[
            "none", "attn-v-only", "global-attn",
            "norm-only", "norm-softmax-reduce", "attn-core",
            "all-fp32", "sensitive", "attention", "all-matmul",
        ],
        default=None,
        help="Mixed precision mode (from most surgical to most conservative): "
             "'none' disables (pure FP16); "
             "'attn-v-only' forces only attn@V MatMul+Softmax+Norm to FP32 (~96 layers); "
             "'global-attn' forces all compute in global attention blocks (7,15,23,31) to FP32; "
             "'norm-only' forces only LayerNorm+Softmax to FP32; "
             "'attn-core' forces attention MatMul+Softmax+Norm to FP32 (~161 layers); "
             "'attention' (default for FP16 backbone) forces all attention ops to FP32 (~225 layers); "
             "'all-matmul' forces ALL MatMul+Softmax to FP32; "
             "'all-fp32' forces everything except Conv to FP32.",
    )
    parser.add_argument(
        "--layer-precisions",
        type=str, default=None, metavar="SPEC",
        help="Force specific layers to a precision. Format: "
             "'pattern1:fp32,pattern2:fp32,...'. Supports wildcards: "
             "'*norm*:fp32,*Softmax*:fp32'. Overrides --mixed-precision. "
             "Mimics trtexec --layerPrecisions.",
    )
    parser.add_argument(
        "--list-layers",
        action="store_true",
        help="Parse ONNX and print all TRT layer names/types, then exit "
             "(no engine build). Useful for designing --layer-precisions specs.",
    )
    args = parser.parse_args()

    if args.list_layers:
        _list_layers(args.onnx)
        return

    if not args.fp16 and not args.int8 and not args.fp32:
        print("No precision flag set, defaulting to FP16")
        args.fp16 = True

    build_engine(
        onnx_path=args.onnx,
        output_path=args.output,
        engine_type=args.type,
        fp16=args.fp16,
        int8=args.int8,
        calib_images=args.calib_images,
        calib_count=args.calib_count,
        calib_cache=args.calib_cache,
        workspace_gb=args.workspace,
        checkpoint=args.checkpoint,
        max_classes=args.max_classes,
        opt_level=args.opt_level,
        mixed_precision=args.mixed_precision,
        layer_precisions=args.layer_precisions,
    )


if __name__ == "__main__":
    main()
