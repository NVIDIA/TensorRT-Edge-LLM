# FP8 KV Cache

## Overview

FP8 KV cache reduces memory usage by quantizing the key-value cache from FP16 to FP8 (NVIDIA FP8 E4M3 format), achieving **~50% memory reduction** and **9--17% faster context attention** on Blackwell for d=128 models (3--7% for d=64) on Thor. On Blackwell (SM100+), the CuTe DSL FP8 FMHA kernel operates directly on FP8 Q/KV tensors, and Q quantization is fused into the RoPE kernel at zero additional cost — so the FP8 path delivers pure speedup with no overhead compared to FP16.

**Key Points**:
- Reduces KV cache memory usage by ~50% (FP8 vs FP16)
- Accelerates context attention by 9--17% on Blackwell (d=128 models) via FP8 CuTe DSL FMHA kernel
- Uses NVIDIA FP8 E4M3 format with per-tensor quantization scales
- Can be used independently or combined with weight quantization
- Automatically detected during engine build from ONNX model configuration
- Requires calibration data during quantization step

---

## Workflow

FP8 KV cache is checkpoint-driven in the `tensorrt-edgellm-quantize` ->
`tensorrt_edgellm` workflow. If checkpoint metadata marks KV cache quantization as
`fp8`, `tensorrt_edgellm` enables FP8 KV cache in the exported ONNX automatically.
There is no FP8 KV export flag in this path.

```bash
export PYTHONPATH=/path/to/TensorRT-Edge-LLM:$PYTHONPATH

tensorrt-edgellm-quantize llm \
  --model_dir Qwen/Qwen3-8B \
  --output_dir /tmp/qwen3_nvfp4_fp8kv \
  --quantization nvfp4 \
  --kv_cache_quantization fp8

tensorrt-edgellm-export \
  /tmp/qwen3_nvfp4_fp8kv \
  /tmp/qwen3_nvfp4_fp8kv_onnx
```

Build the TensorRT engine as usual. Engine build and runtime detect FP8 KV
cache from the exported ONNX model configuration:

```bash
./build/examples/llm/llm_build \
  --onnxDir /tmp/qwen3_nvfp4_fp8kv_onnx/llm \
  --engineDir engines/qwen3-8b-nvfp4-fp8kv \
  --maxBatchSize=1
```

Run inference with the built engine. No special runtime flags are needed:

```bash
./build/examples/llm/llm_inference \
  --engineDir engines/qwen3-8b-nvfp4-fp8kv \
  --inputFile tests/test_cases/llm_basic.json \
  --outputFile output.json
```

### EAGLE Speculative Decoding

For EAGLE, quantize the base checkpoint with `--kv_cache_quantization fp8`,
export it with `--eagle-base`, and export the draft checkpoint separately.

```bash
tensorrt-edgellm-quantize llm \
  --model_dir Qwen/Qwen3-8B \
  --output_dir Qwen3-8B/quantized-base-nvfp4-fp8kv \
  --quantization nvfp4 \
  --kv_cache_quantization fp8

tensorrt-edgellm-export \
  Qwen3-8B/quantized-base-nvfp4-fp8kv \
  Qwen3-8B/onnx/base \
  --eagle-base

tensorrt-edgellm-export \
  /path/to/eagle3_draft \
  Qwen3-8B/onnx/draft
```

Build the base with `--specBase`, build the draft with `--specDraft`, and run
inference with `--specDecode` as described in
[Speculative Decoding](../examples/speculative-decoding.md).

---

## Technical Details

### Quantization Format

- **Format**: NVIDIA FP8 E4M3 (4 exponent bits, 3 mantissa bits)
- **Scale**: Per-tensor dequantization scales `[q_scale, k_scale, v_scale]` for Q, K and V separately
- **Calibration**: Uses calibration dataset to determine optimal quantization scales
- **Memory Reduction**: ~50% reduction compared to FP16 (8 bits vs 16 bits per element)

### Quantization Process

During quantization:
1. Calibration data is passed through the model
2. Maximum absolute values (amax) are collected for K and V caches per layer
3. Quantization scales are computed: `scale = amax / FP8_E4M3_MAX` (where `FP8_E4M3_MAX = 448.0`)
4. Scales are stored in the model for use during inference

### Runtime Behavior

The main objective of FP8 KV cache is to optimize memory usage. During inference:
- KV cache is stored in FP8 format to reduce memory footprint
- QKV dequantization scales `[q_scale, k_scale, v_scale]` are provided to attention kernels
- Attention output is always FP16 — downstream Q/DQ for the output projection is handled by the TRT graph
- On Blackwell (SM100+), the CuTe DSL FP8 FMHA kernel reads FP8 Q/KV directly with dequant scales folded into softmax and output scaling
- Q is quantized to FP8 using the calibrated Q scale before the FP8 FMHA kernel
- No accuracy degradation expected for most use cases

---

## Performance: FP8 vs FP16 FMHA on Thor

When FP8 KV cache is enabled on Blackwell, context attention uses the CuTe DSL FP8 FMHA kernel, which operates directly on FP8 Q/KV tensors. Q quantization (FP16→FP8) is fused into the RoPE kernel at zero additional cost, so the kernel speedup translates directly to end-to-end savings. Attention output is always FP16.

Benchmarks were collected on a Blackwell (Thor / SM100) GPU: causal masking, persistent scheduling, cold L2 cache, FP32 accumulation, 50 warmup + 100 timed iterations.

### Key Findings

- **FP8 kernel is faster in all 54 tested configurations** (6 models × 3 batch sizes × 3 sequence lengths) — no case where FP16 is faster.
- d=128 models: **9--17% faster** — consistently strong FP8 benefit across all configurations.
- d=64 models: **3--7% faster** — lower benefit due to smaller head dimension.
- Speedup converges to ~9--10% at large `bs × seq_len` (compute-bound) and increases to 13--17% at smaller inputs (memory-bound, where reduced data movement matters more).

### Qwen3-0.6B (h_q=16, h_k=8, d=128)

| bs | seq_len | FP16 (us) | FP8 (us) | Saved |
|----|---------|-----------|----------|-------|
| 1 | 512 | 57.8 | 51.0 | +6.8 us (+11.8%) |
| 1 | 1024 | 131.2 | 115.5 | +15.7 us (+12.0%) |
| 1 | 4096 | 712.1 | 647.1 | +65.0 us (+9.1%) |
| 4 | 512 | 174.1 | 145.8 | +28.3 us (+16.3%) |
| 4 | 1024 | 393.5 | 350.5 | +43.0 us (+10.9%) |
| 4 | 4096 | 2713.5 | 2463.4 | +250.1 us (+9.2%) |
| 8 | 512 | 321.8 | 267.7 | +54.1 us (+16.8%) |
| 8 | 1024 | 769.6 | 687.3 | +82.3 us (+10.7%) |
| 8 | 4096 | 5413.8 | 4888.2 | +525.6 us (+9.7%) |

### Qwen3-8B (h_q=32, h_k=8, d=128)

| bs | seq_len | FP16 (us) | FP8 (us) | Saved |
|----|---------|-----------|----------|-------|
| 1 | 512 | 104.9 | 88.7 | +16.2 us (+15.4%) |
| 1 | 1024 | 217.2 | 193.7 | +23.5 us (+10.8%) |
| 1 | 4096 | 1385.7 | 1266.5 | +119.2 us (+8.6%) |
| 4 | 512 | 309.1 | 262.8 | +46.3 us (+15.0%) |
| 4 | 1024 | 765.1 | 686.0 | +79.1 us (+10.3%) |
| 4 | 4096 | 5399.7 | 4886.5 | +513.2 us (+9.5%) |
| 8 | 512 | 592.4 | 503.2 | +89.2 us (+15.1%) |
| 8 | 1024 | 1514.4 | 1342.5 | +171.9 us (+11.4%) |
| 8 | 4096 | 10688.1 | 9690.1 | +998.0 us (+9.3%) |

---

## Warning

**Model-Specific Accuracy Issues**: Not all models work well with FP8 KV cache. Some models may experience accuracy loss due to model-specific characteristics.

**Known Issues**:
- **Qwen2.5-7B-Instruct** and **Qwen2.5-VL-7B-Instruct** exhibit significant accuracy degradation with FP8 KV cache
- The root cause is that the KV cache of the last layer has very large values, resulting in substantial quantization loss
- This is a model-specific issue that also occurs in other frameworks, not a limitation of this implementation
- See [GitHub issue #4218](https://github.com/NVIDIA/TensorRT-LLM/issues/4218) for detailed discussion and examples of the accuracy issues with Qwen2.5 models

**Recommendation**: Test accuracy on your specific model before deploying FP8 KV cache in production. If significant accuracy loss is observed, consider using FP16 KV cache instead.

---

## Notes

- **Calibration Required**: FP8 KV cache quantization requires calibration data. Use `--dataset` to select a calibration dataset (default: `cnn_dailymail`).
- **Independent of Weight Quantization**: FP8 KV cache can be enabled independently of weight quantization. You can use NVFP4 weights (or FP8 weights) with FP8 KV cache.
- **Automatic Detection**: Engine build automatically detects FP8 KV cache from ONNX model configuration — no special build flags needed.
- **Compatibility**: Works with the supported LLM and EAGLE flows, but accuracy should be validated per checkpoint.
- **Memory Benefits**: Most beneficial for long-context models and high batch sizes where KV cache memory dominates.
- **Platform Requirements**: Requires CUDA 11.8+ for FP8 support (`cuda_fp8.h`). FP8 KV cache is supported on GPUs with compute capability SM89 and above (Ada Lovelace and newer).
