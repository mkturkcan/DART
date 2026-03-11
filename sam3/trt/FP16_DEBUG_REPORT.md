# TRT FP16 Backbone Debug Report

## Problem
TRT FP16 backbone engine produces catastrophically wrong outputs (cosine similarity ~0.08 vs PyTorch).
FP32 TRT engine works perfectly (cosine = 0.999982).

## What We Tried

| Approach | Result |
|----------|--------|
| FP32 engine (backbone.onnx) | cosine=0.999982 (WORKS) |
| FP16 engine, plain | cosine=0.088 |
| FP16 engine + onnxsim simplified graph | cosine=0.088 |
| FP16 engine + OBEY_PRECISION_CONSTRAINTS (force Norm/Softmax/Elem/Unary to FP32) | cosine=0.088 |
| FP16 engine + opt-level 0 (minimal optimization) | cosine=0.078 |
| FP16 engine + opt-level 3 (default) from folded ONNX | cosine=0.078 |

All FP16 variants fail identically. Mixed precision constraints have zero effect.

## What We Ruled Out

1. **RoPE patching**: Bit-perfect in Python (per-module diff < 5e-7, full backbone cosine=0.999999)
2. **FP16 overflow**: No activation exceeds FP16 range (max=311, well under 65504)
3. **PyTorch FP16 incompatibility**: Pure FP16 PyTorch (.half()) works perfectly (all blocks cosine > 0.999)
4. **I/O dtype mismatch**: Engine I/O is FP32 for both FP16 and FP32 engines
5. **ONNX If/Condition nodes**: onnxsim removes them, still fails
6. **Graph complexity**: Simplified graph (8600 layers vs 12344) still fails
7. **Optimization level**: opt-level 0 through 3 all fail

## Root Cause Hypothesis

The issue is in how TRT's ONNX parser / FP16 compiler handles this specific graph pattern.
The ONNX model decomposes `F.scaled_dot_product_attention` into separate MatMul + Scale +
Softmax + MatMul ops. TRT's FP16 kernel selection or internal reformatting for these
decomposed ops produces systematically wrong output (~2x std inflation).

This does NOT reproduce in PyTorch because PyTorch uses fused FlashAttention/SDPA kernels
with FP32 internal accumulation, while TRT uses its own FP16 kernel implementations for
the decomposed ops.

## ONNX Model Stats (backbone_folded.onnx after onnxsim)
- 8030 ONNX nodes (8600 TRT layers)
- 192 MatMul, 32 Softmax, 65 LayerNorm, 2 If nodes (removed by onnxsim)
- 32 transformer blocks: 4 global attention (L=5184), 28 windowed (L=576, window=24)
- Head dim = 64, 16 heads, 4 global blocks use concat_rel_pos

## Recommended Next Steps

1. **Torch-TensorRT** (`torch_tensorrt`): Compile PyTorch model directly to TRT via
   `torch.compile(backend="tensorrt")`. Avoids ONNX entirely. Handles FP16 correctly
   because it understands PyTorch op semantics including fused SDPA.

2. **Export backbone-only without SDPA decomposition**: Use ONNX opset 21+ which has
   native `ScaledDotProductAttention` op, letting TRT use its own fused attention kernel.

3. **FP32 engine as fallback**: The FP32 engine works perfectly. It's larger (875 MB vs
   ~450 MB) and somewhat slower, but produces correct results. Viable for deployment
   where accuracy matters more than maximum throughput.

4. **HuggingFace SAM3 export**: The HuggingFace `Sam3Model` exports to ONNX cleanly
   and works with plain `trtexec --fp16`. Different implementation avoids the issue.
