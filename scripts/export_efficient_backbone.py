#!/usr/bin/env python3
"""Export EfficientSAM3 lightweight backbone to ONNX + TRT FP16.

Exports the complete vision pipeline (CNN backbone + projection head + FPN neck)
as a single ONNX graph, builds a TRT FP16 engine, and validates cosine similarity
against the PyTorch reference.

EfficientSAM3 backbones are lightweight CNNs (4-23M params) that should work
perfectly with TRT FP16 (no ViT-H accumulation issues).

The output is 3 FPN levels matching the TRTBackbone contract:
    fpn_0: [1, 256, 4*P, 4*P]  (4x upsample)    e.g. 288x288 for 1008px
    fpn_1: [1, 256, 2*P, 2*P]  (2x upsample)    e.g. 144x144
    fpn_2: [1, 256, P, P]      (1x identity)     e.g. 72x72

Usage:
    # Export all three EfficientSAM3 variants:
    PYTHONIOENCODING=utf-8 python scripts/export_efficient_backbone.py \
        --image x.jpg --variant all

    # Export only EfficientViT-B1:
    PYTHONIOENCODING=utf-8 python scripts/export_efficient_backbone.py \
        --image x.jpg --variant efficientvit

    # Benchmark existing engine:
    PYTHONIOENCODING=utf-8 python scripts/export_efficient_backbone.py \
        --image x.jpg --variant efficientvit --benchmark-only
"""

import argparse
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sam3.efficient_backbone import build_efficientsam3_model


# Variant configs: (backbone_type, model_name, checkpoint_filename, display_name)
VARIANTS = {
    "efficientvit": ("efficientvit", "b1", "efficient_sam3_efficientvit_m_geo_ft.pt", "EfficientViT-B1"),
    "repvit": ("repvit", "m2.3", "efficient_sam3_repvit_l.pt", "RepViT-M2.3"),
    "tinyvit": ("tinyvit", "11m", "efficient_sam3_tinyvit_m_geo_ft.pt", "TinyViT-11M"),
}


class EfficientBackboneForExport(nn.Module):
    """Wrapper that extracts 3 FPN levels from EfficientSAM3 vision backbone.

    Takes pixel_values (B, 3, H, W) and returns 3 FPN levels:
        fpn_0: (B, 256, 4P, 4P), fpn_1: (B, 256, 2P, 2P), fpn_2: (B, 256, P, P)
    where P = embed_size (72 for 1008px).
    """

    def __init__(self, vision_backbone):
        super().__init__()
        self.vision_backbone = vision_backbone

    def forward(self, pixel_values):
        # trunk expects list of images, returns list of feature maps
        # Then FPN neck is applied
        # vision_backbone is Sam3DualViTDetNeck
        sam3_out, sam3_pos, _, _ = self.vision_backbone([pixel_values])
        # Drop the 4th level (0.5x) — downstream uses first 3 only
        return sam3_out[0], sam3_out[1], sam3_out[2]


def export_onnx(model, variant_name, imgsz, output_dir):
    """Export EfficientSAM3 backbone to ONNX via dynamo."""
    wrapper = EfficientBackboneForExport(model.backbone.vision_backbone).cpu().eval()

    dummy = torch.randn(1, 3, imgsz, imgsz)

    # Verify forward pass
    print(f"Running forward pass (imgsz={imgsz})...")
    with torch.no_grad():
        fpn0, fpn1, fpn2 = wrapper(dummy)
    print(f"  fpn_0: {list(fpn0.shape)}")
    print(f"  fpn_1: {list(fpn1.shape)}")
    print(f"  fpn_2: {list(fpn2.shape)}")

    out_path = Path(output_dir)
    out_path.mkdir(exist_ok=True)
    onnx_path = str(out_path / f"efficient_{variant_name}_backbone.onnx")

    print(f"Exporting ONNX via dynamo -> {onnx_path} ...")
    t0 = time.perf_counter()
    export_output = torch.onnx.export(wrapper, (dummy,), dynamo=True)
    export_output.save(onnx_path)
    dt = time.perf_counter() - t0
    total_size = sum(f.stat().st_size for f in out_path.iterdir() if f.name.startswith(f"efficient_{variant_name}"))
    print(f"  Export done ({dt:.1f}s), total ONNX size: {total_size / 1e6:.0f} MB")

    return onnx_path


def build_trt_engine(onnx_path, output_path):
    """Build TRT FP16 engine from ONNX (pure FP16, no mixed precision needed)."""
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
    for i in range(network.num_inputs):
        inp = network.get_input(i)
        print(f"  Input:  {inp.name}: {inp.shape}")
    for i in range(network.num_outputs):
        out = network.get_output(i)
        print(f"  Output: {out.name}: {out.shape}")

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


def run_trt_engine(engine_path, pixel_values):
    """Run TRT backbone engine and return FPN outputs + timing."""
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

    io_bufs = {}
    output_names = []
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        shape = engine.get_tensor_shape(name)
        dtype = _trt_to_torch.get(engine.get_tensor_dtype(name), torch.float32)
        mode = engine.get_tensor_mode(name)
        is_input = (mode == trt.TensorIOMode.INPUT)

        if is_input:
            buf = pixel_values.to(dtype=dtype, device="cuda").contiguous()
        else:
            buf = torch.empty(list(shape), dtype=dtype, device="cuda")
            output_names.append(name)

        io_bufs[name] = buf
        context.set_tensor_address(name, buf.data_ptr())

    stream = torch.cuda.Stream()
    context.execute_async_v3(stream.cuda_stream)
    stream.synchronize()

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


def process_variant(
    variant_key, ckpt_dir, image_path, imgsz, output_dir,
    skip_export, skip_build, benchmark_only,
):
    """Full pipeline for one EfficientSAM3 variant."""
    backbone_type, model_name, ckpt_file, display_name = VARIANTS[variant_key]
    ckpt_path = os.path.join(ckpt_dir, ckpt_file)

    if not os.path.exists(ckpt_path):
        print(f"\nSkipping {display_name}: {ckpt_path} not found")
        return None

    print(f"\n{'='*60}")
    print(f"Variant: {display_name} ({variant_key})")
    print(f"{'='*60}")

    engine_path = os.path.join(output_dir, f"efficient_{variant_key}_fp16.engine")
    onnx_dir = os.path.join(output_dir, f"onnx_efficient_{variant_key}")

    # Load model
    print(f"Loading {display_name} from {ckpt_path} ...")
    model = build_efficientsam3_model(
        backbone_type=backbone_type,
        model_name=model_name,
        checkpoint_path=ckpt_path,
        device="cpu",
        eval_mode=True,
    )

    backbone_params = sum(p.numel() for p in model.backbone.vision_backbone.trunk.parameters())
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Backbone params: {backbone_params/1e6:.1f}M, Total: {total_params/1e6:.1f}M")

    # Step 1: Export ONNX
    if not skip_export and not benchmark_only:
        onnx_path = export_onnx(model, variant_key, imgsz, onnx_dir)
    else:
        onnx_path = os.path.join(onnx_dir, f"efficient_{variant_key}_backbone.onnx")
        print(f"Skipping ONNX export, using: {onnx_path}")

    # Step 2: Build TRT engine
    if not skip_build and not benchmark_only:
        build_trt_engine(onnx_path, engine_path)
    else:
        print(f"Skipping engine build, using: {engine_path}")

    # Step 3: PyTorch reference
    from PIL import Image
    from torchvision.transforms import v2

    print(f"\nRunning PyTorch reference on CUDA...")
    model = model.cuda()
    image = Image.open(image_path).convert("RGB")
    transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize(size=(imgsz, imgsz)),
        v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    pixel_values = transform(image).unsqueeze(0).cuda()

    wrapper = EfficientBackboneForExport(model.backbone.vision_backbone).cuda().eval()
    with torch.inference_mode():
        ref_fpn0, ref_fpn1, ref_fpn2 = wrapper(pixel_values)
    ref_fpn = [ref_fpn0.clone(), ref_fpn1.clone(), ref_fpn2.clone()]
    print(f"  fpn_0: {list(ref_fpn[0].shape)}")
    print(f"  fpn_1: {list(ref_fpn[1].shape)}")
    print(f"  fpn_2: {list(ref_fpn[2].shape)}")

    # Benchmark PyTorch
    for _ in range(3):
        with torch.inference_mode():
            wrapper(pixel_values)
    torch.cuda.synchronize()
    times = []
    for _ in range(20):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            wrapper(pixel_values)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    pytorch_ms = times[len(times) // 2]
    print(f"  PyTorch backbone median: {pytorch_ms:.1f}ms")

    pixel_values_for_trt = pixel_values.clone()
    del model, wrapper
    torch.cuda.empty_cache()

    # Step 4: TRT inference + comparison
    if not os.path.exists(engine_path):
        print(f"  Engine not found: {engine_path}")
        return {"name": display_name, "pytorch_ms": pytorch_ms}

    trt_fpn, trt_ms = run_trt_engine(engine_path, pixel_values_for_trt)

    # Compare
    print(f"\n{'='*60}")
    print(f"QUALITY: {display_name} TRT FP16 vs PyTorch")
    print(f"{'='*60}")
    for i in range(min(3, len(trt_fpn))):
        ref = ref_fpn[i]
        trt_out = trt_fpn[i]
        cos = cosine(ref, trt_out)
        max_diff = (ref.cuda() - trt_out.cuda()).abs().max().item()
        status = "OK" if cos > 0.99 else "BROKEN" if cos < 0.5 else "DEGRADED"
        print(f"  fpn_{i}: cosine={cos:.6f}  max_diff={max_diff:.4f}  {status}")

    print(f"\n  PyTorch: {pytorch_ms:.1f}ms")
    print(f"  TRT FP16: {trt_ms:.1f}ms")
    print(f"  Speedup: {pytorch_ms / trt_ms:.1f}x")
    print(f"  Backbone params: {backbone_params/1e6:.1f}M")
    print(f"{'='*60}")

    return {
        "name": display_name,
        "pytorch_ms": pytorch_ms,
        "trt_ms": trt_ms,
        "cosines": [cosine(ref_fpn[i], trt_fpn[i]) for i in range(3)],
        "params_m": backbone_params / 1e6,
    }


def main():
    parser = argparse.ArgumentParser(description="Export EfficientSAM3 backbone to TRT FP16")
    parser.add_argument("--image", default="x.jpg", help="Test image")
    parser.add_argument("--imgsz", type=int, default=1008)
    parser.add_argument("--ckpt-dir", default="stage1_all_converted")
    parser.add_argument("--output-dir", default=".")
    parser.add_argument("--variant", default="all", choices=["all"] + list(VARIANTS.keys()))
    parser.add_argument("--skip-export", action="store_true")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--benchmark-only", action="store_true")
    args = parser.parse_args()

    variants = list(VARIANTS.keys()) if args.variant == "all" else [args.variant]

    results = []
    for v in variants:
        r = process_variant(
            v, args.ckpt_dir, args.image, args.imgsz, args.output_dir,
            args.skip_export, args.skip_build, args.benchmark_only,
        )
        if r:
            results.append(r)

    # Summary
    if results:
        print(f"\n\n{'='*80}")
        print("SUMMARY: EfficientSAM3 Backbone TRT FP16 Export")
        print(f"{'='*80}")
        print(f"  {'Model':<20s}  {'Params':>7s}  {'PyTorch':>9s}  {'TRT FP16':>9s}  {'Speedup':>7s}  {'Cos(fpn2)':>10s}")
        print(f"  {'-'*76}")
        for r in results:
            trt_ms = r.get("trt_ms", 0)
            speedup = r["pytorch_ms"] / trt_ms if trt_ms > 0 else 0
            cos2 = r.get("cosines", [0, 0, 0])[2] if "cosines" in r else 0
            print(
                f"  {r['name']:<20s}  {r.get('params_m', 0):>6.1f}M  "
                f"{r['pytorch_ms']:>7.1f}ms  {trt_ms:>7.1f}ms  "
                f"{speedup:>6.1f}x  {cos2:>10.6f}"
            )
        print(f"{'='*80}")


if __name__ == "__main__":
    main()
