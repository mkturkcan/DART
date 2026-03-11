#!/usr/bin/env python3
"""Export HuggingFace SAM3 backbone (ViT-H + FPN neck) to ONNX and TRT.

Extracts just the vision_encoder from the HF Sam3Model, exports it as an
ONNX graph, builds a TRT FP16 engine, and benchmarks it against the PyTorch
reference with cosine similarity.

Supports block pruning (--mask-blocks) and custom resolution (--imgsz).

The backbone outputs 3 FPN levels matching the existing TRTBackbone contract:
    fpn_0: [1, 256, 4*P, 4*P]  (4x upsample)    e.g. 288x288 for imgsz=1008
    fpn_1: [1, 256, 2*P, 2*P]  (2x upsample)    e.g. 144x144
    fpn_2: [1, 256, P, P]      (1x identity)     e.g. 72x72
where P = imgsz // 14 (patch grid size).

Usage:
    # Full pipeline (default 1008px):
    PYTHONIOENCODING=utf-8 python scripts/export_hf_backbone.py --image x.jpg

    # With block pruning and custom resolution:
    PYTHONIOENCODING=utf-8 python scripts/export_hf_backbone.py --image x.jpg \\
        --imgsz 1008 \\
        --mask-blocks "25:attn,28:mlp,27:attn,22:attn"

    # Benchmark existing engine only:
    PYTHONIOENCODING=utf-8 python scripts/export_hf_backbone.py \\
        --image x.jpg --benchmark-only
"""

import argparse
import time
import types
from pathlib import Path

import torch
import torch.nn as nn


class HFBackboneForExport(nn.Module):
    """Wrapper that extracts 3 FPN levels from Sam3VisionModel.

    Drops the 4th FPN level (0.5x scale) to match the pipeline convention
    (equivalent to scalp=1 in the Meta codebase).
    """

    def __init__(self, vision_encoder):
        super().__init__()
        self.vision_encoder = vision_encoder

    def forward(self, pixel_values):
        outputs = self.vision_encoder(pixel_values)
        fpn = outputs.fpn_hidden_states  # tuple of 4 FPN levels
        # Drop the last level (0.5x) — downstream code uses fpn[:-1]
        return fpn[0], fpn[1], fpn[2]


class HFBackbonePart1ForExport(nn.Module):
    """Part 1: embeddings + reshape + layer_norm + blocks[0:split_block].

    Sam3ViTModel does: embeddings → [B,N,D] → view(B,H,W,D) → layer_norm → layers.
    Layer norm is applied BEFORE the layers, not after.

    Input:  pixel_values [B, 3, imgsz, imgsz]
    Output: hidden_states [B, H, W, D]  (D=1024 for ViT-H, H=W=imgsz//14)
    """

    def __init__(self, vision_encoder, split_block):
        super().__init__()
        backbone = vision_encoder.backbone  # Sam3ViTModel
        self.embeddings = backbone.embeddings
        self.layer_norm = backbone.layer_norm
        self.blocks = nn.ModuleList(list(backbone.layers[:split_block]))
        self.patch_size = backbone.config.patch_size

    def forward(self, pixel_values):
        x = self.embeddings(pixel_values)  # [B, N, D] (sequence format)
        B = x.shape[0]
        H = pixel_values.shape[-2] // self.patch_size
        W = pixel_values.shape[-1] // self.patch_size
        D = x.shape[-1]
        x = x.view(B, H, W, D)    # [B, H, W, D] (spatial BHWD)
        x = self.layer_norm(x)    # norm BEFORE layers
        for block in self.blocks:
            x = block(x)          # Sam3ViTLayer returns tensor directly
        return x  # [B, H, W, D]


class HFBackbonePart2ForExport(nn.Module):
    """Part 2: blocks[split_block:] + permute to BDHW + FPN neck.

    No layer_norm here — it was already applied in Part1 before the layers.
    After blocks, permute [B,H,W,D] → [B,D,H,W] for the FPN neck.

    Input:  hidden_states [B, H, W, D]
    Output: fpn_0 [B, 256, 4H, 4W], fpn_1 [B, 256, 2H, 2W], fpn_2 [B, 256, H, W]
    """

    def __init__(self, vision_encoder, split_block):
        super().__init__()
        backbone = vision_encoder.backbone  # Sam3ViTModel
        self.blocks = nn.ModuleList(list(backbone.layers[split_block:]))
        self.neck = vision_encoder.neck  # Sam3VisionNeck (FPN)

    def forward(self, hidden_states):
        for block in self.blocks:
            hidden_states = block(hidden_states)  # [B, H, W, D]
        # Permute to channels-first for FPN neck: [B, H, W, D] → [B, D, H, W]
        hidden_states = hidden_states.permute(0, 3, 1, 2)
        fpn_outputs, _ = self.neck(hidden_states)
        # Drop the 4th level (0.5x scale)
        return fpn_outputs[0], fpn_outputs[1], fpn_outputs[2]


def _parse_mask_blocks(mask_blocks_str):
    """Parse mask_blocks CLI string into a dict of {block_idx: set(sub_types)}.

    Args:
        mask_blocks_str: Comma-separated "idx:type" pairs, e.g.
            "25:attn,28:mlp,27:attn,22:attn"

    Returns:
        Dict mapping block index to set of sub-types to mask.
        E.g. {25: {"attn"}, 28: {"mlp"}, 27: {"attn"}, 22: {"attn"}}
    """
    parsed = {}
    for entry in mask_blocks_str.split(","):
        entry = entry.strip()
        if not entry:
            continue
        idx_str, sub_type = entry.split(":")
        parsed.setdefault(int(idx_str), set()).add(sub_type.strip())
    return parsed


def _apply_mask_blocks(vision_encoder, mask_blocks):
    """Monkey-patch HF Sam3ViTLayer.forward() to skip masked sub-blocks.

    The HF ViT layer forward is:
        residual = x
        x = norm1(x) -> window_partition -> rotary_emb -> attention -> unpartition
        x = residual + x
        residual = x
        x = norm2(x) -> mlp
        x = residual + dropout(x)

    When mask_attn=True, the attention branch is skipped (identity).
    When mask_mlp=True, the MLP branch is skipped (identity).
    """
    from transformers.models.sam3.modeling_sam3 import (
        window_partition,
        window_unpartition,
    )

    layers = vision_encoder.backbone.layers
    count = 0

    for block_idx, sub_types in mask_blocks.items():
        if block_idx >= len(layers):
            print(f"  WARNING: block {block_idx} out of range "
                  f"(model has {len(layers)} layers)")
            continue

        layer = layers[block_idx]
        do_mask_attn = "attn" in sub_types
        do_mask_mlp = "mlp" in sub_types

        def _make_masked_forward(orig_layer, skip_attn, skip_mlp):
            """Create a replacement forward that skips masked sub-blocks."""

            def masked_forward(self_unused, hidden_states, **kwargs):
                # Attention branch
                if not skip_attn:
                    residual = hidden_states
                    hidden_states = orig_layer.layer_norm1(hidden_states)
                    if orig_layer.window_size > 0:
                        height = hidden_states.shape[1]
                        width = hidden_states.shape[2]
                        hidden_states, pad_hw = window_partition(
                            hidden_states, orig_layer.window_size
                        )
                    position_embeddings = orig_layer.rotary_emb()
                    hidden_states, _ = orig_layer.attention(
                        hidden_states, position_embeddings, **kwargs
                    )
                    if orig_layer.window_size > 0:
                        hidden_states = window_unpartition(
                            hidden_states, orig_layer.window_size,
                            pad_hw, (height, width)
                        )
                    hidden_states = residual + hidden_states

                # MLP branch
                if not skip_mlp:
                    residual = hidden_states
                    hidden_states = orig_layer.layer_norm2(hidden_states)
                    hidden_states = orig_layer.mlp(hidden_states)
                    hidden_states = residual + orig_layer.dropout(hidden_states)

                return hidden_states

            return masked_forward

        layer.forward = types.MethodType(
            _make_masked_forward(layer, do_mask_attn, do_mask_mlp), layer
        )
        masked_parts = []
        if do_mask_attn:
            masked_parts.append("attn")
        if do_mask_mlp:
            masked_parts.append("mlp")
        count += 1

    print(f"  Applied mask_blocks to {count} layers")
    return count


def export_split_onnx(output_dir, split_block, imgsz=1008, mask_blocks=None):
    """Export HF SAM3 backbone as two ONNX models (Part1 + Part2) via dynamo."""
    from transformers.models.sam3 import Sam3Model

    print(f"Loading HuggingFace SAM3 model for split export (split_block={split_block})...")

    kwargs = {}
    if imgsz != 1008:
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained("facebook/sam3")
        config.image_size = imgsz
        config.detector_config.image_size = imgsz
        config.detector_config.vision_config.backbone_config.image_size = imgsz
        P = imgsz // 14
        config.detector_config.vision_config.backbone_feature_sizes = [
            [P * 4, P * 4], [P * 2, P * 2], [P, P]
        ]
        kwargs["config"] = config
        print(f"  Overriding image_size={imgsz} (spatial={P}x{P})")

    model = Sam3Model.from_pretrained(
        "facebook/sam3", attn_implementation="eager", **kwargs
    ).cpu()
    model.eval()

    if mask_blocks:
        print(f"Applying mask_blocks ({len(mask_blocks)} blocks)...")
        _apply_mask_blocks(model.vision_encoder, mask_blocks)

    # Create Part1 and Part2 wrappers
    part1 = HFBackbonePart1ForExport(model.vision_encoder, split_block).cpu().eval()
    part2 = HFBackbonePart2ForExport(model.vision_encoder, split_block).cpu().eval()

    P = imgsz // 14
    D = model.config.vision_config.backbone_config.hidden_size

    # Verify Part1
    dummy_pixels = torch.randn(1, 3, imgsz, imgsz)
    print(f"Running Part1 forward pass...")
    with torch.no_grad():
        intermediate = part1(dummy_pixels)
    print(f"  Part1 output: {list(intermediate.shape)}")  # [1, P, P, D]
    assert intermediate.shape == (1, P, P, D), \
        f"Expected [1, {P}, {P}, {D}], got {list(intermediate.shape)}"

    # Verify Part2
    print(f"Running Part2 forward pass...")
    with torch.no_grad():
        fpn0, fpn1, fpn2 = part2(intermediate)
    print(f"  fpn_0: {list(fpn0.shape)}")
    print(f"  fpn_1: {list(fpn1.shape)}")
    print(f"  fpn_2: {list(fpn2.shape)}")

    # Verify end-to-end matches full wrapper
    full_wrapper = HFBackboneForExport(model.vision_encoder).cpu().eval()
    with torch.no_grad():
        ref0, ref1, ref2 = full_wrapper(dummy_pixels)
    for i, (split_out, ref_out) in enumerate([(fpn0, ref0), (fpn1, ref1), (fpn2, ref2)]):
        diff = (split_out - ref_out).abs().max().item()
        print(f"  fpn_{i} max diff (split vs full): {diff:.6f}")
        assert diff < 1e-4, f"Split output diverges from full: max_diff={diff}"
    print("  Split outputs match full backbone!")

    out_path = Path(output_dir)
    out_path.mkdir(exist_ok=True)

    # Export Part1
    onnx_part1 = str(out_path / "hf_backbone_part1.onnx")
    print(f"\nExporting Part1 ONNX -> {onnx_part1} ...")
    t0 = time.perf_counter()
    export_output = torch.onnx.export(part1, (dummy_pixels,), dynamo=True)
    export_output.save(onnx_part1)
    print(f"  Part1 export done ({time.perf_counter() - t0:.1f}s)")

    # Export Part2
    onnx_part2 = str(out_path / "hf_backbone_part2.onnx")
    print(f"Exporting Part2 ONNX -> {onnx_part2} ...")
    t0 = time.perf_counter()
    export_output = torch.onnx.export(part2, (intermediate,), dynamo=True)
    export_output.save(onnx_part2)
    print(f"  Part2 export done ({time.perf_counter() - t0:.1f}s)")

    del model, part1, part2, full_wrapper
    return onnx_part1, onnx_part2


def export_onnx(output_dir, imgsz=1008, mask_blocks=None):
    """Export HF SAM3 backbone to ONNX via dynamo."""
    from transformers.models.sam3 import Sam3Model

    print("Loading HuggingFace SAM3 model (eager attention for ONNX compat)...")

    # For non-default resolutions, update the vision config so that
    # rotary embeddings are computed for the correct spatial grid.
    kwargs = {}
    if imgsz != 1008:
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained("facebook/sam3")
        config.image_size = imgsz
        config.detector_config.image_size = imgsz
        config.detector_config.vision_config.backbone_config.image_size = imgsz
        # Update backbone_feature_sizes for new spatial grid
        P = imgsz // 14
        config.detector_config.vision_config.backbone_feature_sizes = [
            [P * 4, P * 4], [P * 2, P * 2], [P, P]
        ]
        kwargs["config"] = config
        print(f"  Overriding image_size={imgsz} (spatial={P}x{P})")

    model = Sam3Model.from_pretrained(
        "facebook/sam3", attn_implementation="eager", **kwargs
    ).cpu()
    model.eval()

    # Apply block masking before export
    if mask_blocks:
        print(f"Applying mask_blocks ({len(mask_blocks)} blocks)...")
        _apply_mask_blocks(model.vision_encoder, mask_blocks)

    # Extract vision encoder and wrap it
    wrapper = HFBackboneForExport(model.vision_encoder).cpu().eval()

    # Dummy input at target resolution
    dummy = torch.randn(1, 3, imgsz, imgsz)

    # Verify forward pass works
    print(f"Running forward pass to verify wrapper (imgsz={imgsz})...")
    with torch.no_grad():
        fpn0, fpn1, fpn2 = wrapper(dummy)
    print(f"  fpn_0: {list(fpn0.shape)}")
    print(f"  fpn_1: {list(fpn1.shape)}")
    print(f"  fpn_2: {list(fpn2.shape)}")

    # Export
    out_path = Path(output_dir)
    out_path.mkdir(exist_ok=True)
    onnx_path = str(out_path / "hf_backbone.onnx")

    print(f"Exporting ONNX via dynamo -> {onnx_path} ...")
    t0 = time.perf_counter()
    export_output = torch.onnx.export(
        wrapper,
        (dummy,),
        dynamo=True,
    )
    export_output.save(onnx_path)
    dt = time.perf_counter() - t0
    print(f"  Export done ({dt:.1f}s)")

    # Check file sizes
    total_size = sum(f.stat().st_size for f in out_path.iterdir())
    print(f"  Total ONNX size: {total_size / 1e6:.0f} MB")

    del model, wrapper
    return onnx_path


def build_engine(onnx_path: str, output_path: str):
    """Build TRT FP16 engine from ONNX."""
    import tensorrt as trt

    TRT_LOGGER = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, TRT_LOGGER)

    onnx_abs = str(Path(onnx_path).resolve())
    print(f"Parsing ONNX: {onnx_abs}")
    if hasattr(parser, "parse_from_file"):
        if not parser.parse_from_file(onnx_abs):
            for i in range(parser.num_errors):
                print(f"  Error: {parser.get_error(i)}")
            raise RuntimeError("ONNX parse failed")
    else:
        with open(onnx_abs, "rb") as f:
            if not parser.parse(f.read()):
                for i in range(parser.num_errors):
                    print(f"  Error: {parser.get_error(i)}")
                raise RuntimeError("ONNX parse failed")

    print(f"  Layers: {network.num_layers}")
    print(f"  Inputs: {network.num_inputs}")
    for i in range(network.num_inputs):
        inp = network.get_input(i)
        print(f"    {inp.name}: {inp.shape}")
    print(f"  Outputs: {network.num_outputs}")
    for i in range(network.num_outputs):
        out = network.get_output(i)
        print(f"    {out.name}: {out.shape}")

    # Build FP16 engine — pure FP16, no mixed precision
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
    config.set_flag(trt.BuilderFlag.FP16)
    if hasattr(config, "builder_optimization_level"):
        config.builder_optimization_level = 3
    print("Building TRT engine (pure FP16)...")

    t0 = time.perf_counter()
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("Engine build failed")

    with open(output_path, "wb") as f:
        f.write(serialized)
    size_mb = Path(output_path).stat().st_size / 1e6
    print(f"  Done ({time.perf_counter() - t0:.0f}s), {size_mb:.0f} MB -> {output_path}")
    return output_path


def run_pytorch_reference(image_path, imgsz=1008, mask_blocks=None):
    """Run HF SAM3 vision encoder in PyTorch, return FPN outputs + pixel_values."""
    from transformers.models.sam3 import Sam3Processor, Sam3Model
    from PIL import Image

    print("Running PyTorch reference (HF backbone on CUDA)...")

    kwargs = {}
    if imgsz != 1008:
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained("facebook/sam3")
        config.image_size = imgsz
        config.detector_config.image_size = imgsz
        config.detector_config.vision_config.backbone_config.image_size = imgsz
        P = imgsz // 14
        config.detector_config.vision_config.backbone_feature_sizes = [
            [P * 4, P * 4], [P * 2, P * 2], [P, P]
        ]
        kwargs["config"] = config

    model = Sam3Model.from_pretrained("facebook/sam3", **kwargs).cuda()
    processor = Sam3Processor.from_pretrained("facebook/sam3")
    model.eval()

    # Apply same mask_blocks as the export for fair comparison
    if mask_blocks:
        _apply_mask_blocks(model.vision_encoder, mask_blocks)

    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, text="dummy", return_tensors="pt").to("cuda")
    pixel_values = inputs["pixel_values"]

    # Resize if non-default imgsz
    if imgsz != 1008:
        pixel_values = torch.nn.functional.interpolate(
            pixel_values, size=(imgsz, imgsz), mode="bilinear",
            align_corners=False,
        )

    with torch.inference_mode():
        vision_out = model.vision_encoder(pixel_values)

    # Get first 3 FPN levels (drop 4th)
    ref_fpn = [t.clone() for t in vision_out.fpn_hidden_states[:3]]
    print(f"  fpn_0: {list(ref_fpn[0].shape)}")
    print(f"  fpn_1: {list(ref_fpn[1].shape)}")
    print(f"  fpn_2: {list(ref_fpn[2].shape)}")

    # Benchmark PyTorch backbone
    for _ in range(3):
        with torch.inference_mode():
            model.vision_encoder(pixel_values)
    torch.cuda.synchronize()
    times = []
    for _ in range(10):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            model.vision_encoder(pixel_values)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    pytorch_ms = times[len(times) // 2]
    print(f"  PyTorch backbone median: {pytorch_ms:.1f}ms")

    pixel_values = pixel_values.clone()
    del model
    torch.cuda.empty_cache()

    return pixel_values, ref_fpn, pytorch_ms


def run_trt_engine(engine_path: str, pixel_values: torch.Tensor):
    """Run TRT backbone engine and return FPN outputs."""
    import tensorrt as trt

    print(f"Loading TRT engine: {engine_path}")
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    with open(engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()

    _trt_to_torch = {
        trt.float32: torch.float32,
        trt.float16: torch.float16,
        trt.int32: torch.int32,
    }

    # Allocate I/O buffers
    io_bufs = {}
    output_names = []
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        shape = engine.get_tensor_shape(name)
        dtype = _trt_to_torch.get(engine.get_tensor_dtype(name), torch.float32)
        mode = engine.get_tensor_mode(name)
        is_input = (mode == trt.TensorIOMode.INPUT)
        print(f"  {'INPUT' if is_input else 'OUTPUT'}: {name} {list(shape)} {dtype}")

        if is_input:
            buf = pixel_values.to(dtype=dtype, device="cuda").contiguous()
        else:
            buf = torch.empty(list(shape), dtype=dtype, device="cuda")
            output_names.append(name)

        io_bufs[name] = buf
        context.set_tensor_address(name, buf.data_ptr())

    # Execute
    stream = torch.cuda.Stream()
    context.execute_async_v3(stream.cuda_stream)
    stream.synchronize()

    # Get outputs as float32, sorted by spatial size descending (largest first)
    outputs = [io_bufs[name].float() for name in output_names]
    outputs.sort(key=lambda t: t.shape[-1], reverse=True)

    # Benchmark
    for _ in range(5):
        context.execute_async_v3(stream.cuda_stream)
    stream.synchronize()
    times = []
    for _ in range(20):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        context.execute_async_v3(stream.cuda_stream)
        stream.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    trt_ms = times[len(times) // 2]
    print(f"  TRT backbone median: {trt_ms:.1f}ms")

    del context, engine
    return outputs, trt_ms


def cosine(a, b):
    return torch.nn.functional.cosine_similarity(
        a.float().flatten().unsqueeze(0),
        b.float().flatten().unsqueeze(0),
    ).item()


def main():
    parser = argparse.ArgumentParser(
        description="Export HF SAM3 backbone to ONNX + TRT and benchmark"
    )
    parser.add_argument("--image", default="x.jpg", help="Test image for validation")
    parser.add_argument(
        "--output-onnx", default="onnx_hf_backbone/hf_backbone.onnx",
        help="ONNX output path"
    )
    parser.add_argument(
        "--output-engine", default="hf_backbone_fp16.engine",
        help="TRT engine output path"
    )
    parser.add_argument(
        "--imgsz", type=int, default=1008,
        help="Input resolution (must be divisible by 14)"
    )
    parser.add_argument(
        "--mask-blocks", type=str, default=None,
        help='Comma-separated "idx:type" pairs for sub-block pruning, '
             'e.g. "25:attn,28:mlp,27:attn"'
    )
    parser.add_argument(
        "--skip-export", action="store_true", help="Skip ONNX export"
    )
    parser.add_argument(
        "--skip-build", action="store_true", help="Skip TRT engine build"
    )
    parser.add_argument(
        "--benchmark-only", action="store_true",
        help="Skip export + build, just benchmark existing engine"
    )
    parser.add_argument(
        "--split-block", type=int, default=None,
        help="Split backbone at this block index. Exports two engines: "
             "Part1 (embeddings + blocks[0:K]) and Part2 (blocks[K:] + FPN). "
             "Use with --output-engine to set Part1 engine name (Part2 gets _part2 suffix).",
    )
    args = parser.parse_args()

    # Validate imgsz
    if args.imgsz % 14 != 0:
        parser.error(f"--imgsz must be divisible by 14, got {args.imgsz}")

    if args.benchmark_only:
        args.skip_export = True
        args.skip_build = True

    # Parse mask_blocks
    mask_blocks = None
    if args.mask_blocks:
        mask_blocks = _parse_mask_blocks(args.mask_blocks)
        print(f"Block masking: {len(mask_blocks)} blocks")
        for idx in sorted(mask_blocks):
            print(f"  block {idx}: mask {mask_blocks[idx]}")

    # --- Split export mode ---
    if args.split_block is not None:
        K = args.split_block
        onnx_dir = str(Path(args.output_onnx).parent)

        # Step 1: Export two ONNX models
        if not args.skip_export:
            onnx_part1, onnx_part2 = export_split_onnx(
                onnx_dir, K, imgsz=args.imgsz, mask_blocks=mask_blocks,
            )
        else:
            onnx_part1 = str(Path(onnx_dir) / "hf_backbone_part1.onnx")
            onnx_part2 = str(Path(onnx_dir) / "hf_backbone_part2.onnx")
            print(f"Skipping ONNX export, using: {onnx_part1}, {onnx_part2}")

        # Step 2: Build two TRT engines
        base = Path(args.output_engine)
        engine_part1 = str(base.parent / f"{base.stem}_part1{base.suffix}")
        engine_part2 = str(base.parent / f"{base.stem}_part2{base.suffix}")
        if not args.skip_build:
            build_engine(onnx_part1, engine_part1)
            build_engine(onnx_part2, engine_part2)
        else:
            print(f"Skipping engine build, using: {engine_part1}, {engine_part2}")

        # Step 3: Benchmark both parts
        pixel_values, ref_fpn, pytorch_ms = run_pytorch_reference(
            args.image, imgsz=args.imgsz, mask_blocks=mask_blocks,
        )

        print(f"\nBenchmarking Part1 engine: {engine_part1}")
        part1_out, part1_ms = run_trt_engine(engine_part1, pixel_values)

        # Part2 input is Part1 output (the intermediate hidden states)
        # Part1 output is a single tensor [B, H, W, D] (BHWD channels-last)
        intermediate = part1_out[0]  # run_trt_engine returns list sorted by spatial size
        print(f"\nBenchmarking Part2 engine: {engine_part2}")
        part2_out, part2_ms = run_trt_engine(engine_part2, intermediate)

        # Compare split pipeline vs PyTorch reference
        print(f"\n{'='*60}")
        print(f"SPLIT BACKBONE (split_block={K})")
        print(f"{'='*60}")
        for i in range(min(3, len(part2_out))):
            ref = ref_fpn[i]
            trt_out = part2_out[i]
            cos = cosine(ref, trt_out)
            max_diff = (ref.cuda() - trt_out.cuda()).abs().max().item()
            status = "OK" if cos > 0.99 else "BROKEN" if cos < 0.5 else "DEGRADED"
            print(f"  fpn_{i}: {list(ref.shape)}")
            print(f"    Cosine: {cos:.6f}  MaxDiff: {max_diff:.4f}  {status}")

        total_ms = part1_ms + part2_ms
        print(f"\n  Part1 (blocks 0..{K-1}):  {part1_ms:.1f}ms  -> {engine_part1}")
        print(f"  Part2 (blocks {K}..31+FPN): {part2_ms:.1f}ms  -> {engine_part2}")
        print(f"  Total (sequential):     {total_ms:.1f}ms")
        print(f"  PyTorch backbone:       {pytorch_ms:.1f}ms")
        if mask_blocks:
            print(f"  Block masking: {len(mask_blocks)} blocks pruned")
        print(f"{'='*60}")
        return

    # --- Full backbone export (original path) ---
    onnx_path = args.output_onnx
    onnx_dir = str(Path(onnx_path).parent)

    # Step 1: Export ONNX
    if not args.skip_export:
        onnx_path = export_onnx(onnx_dir, imgsz=args.imgsz,
                                mask_blocks=mask_blocks)
    else:
        print(f"Skipping ONNX export, using: {onnx_path}")

    # Step 2: Build TRT engine
    if not args.skip_build:
        build_engine(onnx_path, args.output_engine)
    else:
        print(f"Skipping engine build, using: {args.output_engine}")

    # Step 3: PyTorch reference
    pixel_values, ref_fpn, pytorch_ms = run_pytorch_reference(
        args.image, imgsz=args.imgsz, mask_blocks=mask_blocks
    )

    # Step 4: TRT inference
    trt_fpn, trt_ms = run_trt_engine(args.output_engine, pixel_values)

    # Step 5: Compare — match by index (both sorted largest-first)
    print(f"\n{'='*60}")
    print("QUALITY COMPARISON: HF Backbone FPN (TRT FP16 vs PyTorch)")
    print(f"{'='*60}")
    for i in range(min(3, len(trt_fpn))):
        ref = ref_fpn[i]
        trt_out = trt_fpn[i]
        cos = cosine(ref, trt_out)
        max_diff = (ref.cuda() - trt_out.cuda()).abs().max().item()
        status = "OK" if cos > 0.99 else "BROKEN" if cos < 0.5 else "DEGRADED"
        print(f"  fpn_{i}: {list(ref.shape)}")
        print(f"    Cosine: {cos:.6f}  MaxDiff: {max_diff:.4f}  {status}")

    if mask_blocks:
        print(f"\n  Block masking: {len(mask_blocks)} blocks pruned")
    print(f"  Resolution: {args.imgsz}x{args.imgsz}")
    print(f"  PyTorch backbone:  {pytorch_ms:.1f}ms")
    print(f"  TRT FP16 backbone: {trt_ms:.1f}ms")
    print(f"  Speedup: {pytorch_ms / trt_ms:.1f}x")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
