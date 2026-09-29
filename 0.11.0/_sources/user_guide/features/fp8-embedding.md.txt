# FP8 Embedding

## Overview

FP8 embedding reduces memory usage by quantizing the embedding table from FP16 to FP8 (NVIDIA FP8 E4M3 format), achieving **~50% memory reduction** for the embedding table. This is particularly beneficial for models with large vocabularies where the embedding table can be a significant portion of total memory usage.

**Key Points**:
- Reduces embedding table memory by ~50% (FP8 vs FP16)
- Uses NVIDIA FP8 E4M3 format with per-row block-wise quantization (block size 128)
- Can be used independently or combined with other quantization methods (NVFP4, INT4, FP8 weights)
- Automatically detected during engine build from safetensors metadata
- Output is always FP16 — dequantization happens on-the-fly during embedding lookup

---

## Workflow

Export with `--fp8-embedding` to write the runtime embedding sidecar in FP8:

```bash
export EDGE_LLM_PATH=/path/to/TensorRT-Edge-LLM
cd $EDGE_LLM_PATH
export PYTHONPATH=$EDGE_LLM_PATH:$PYTHONPATH

tensorrt-edgellm-export \
  /path/to/Qwen3-8B \
  /tmp/qwen3_onnx_fp8emb \
  --fp8-embedding
```

This flag only changes `embedding.safetensors`; it does not change the ONNX graph or the checkpoint weights. The sidecar contains the FP8 embedding table plus `embedding_scale`.

### Build Engine

Build the TensorRT engine as usual. The build process automatically detects FP8 embedding from the safetensors metadata:

```bash
./build/examples/llm/llm_build \
  --onnxDir /tmp/qwen3_onnx_fp8emb/llm \
  --engineDir engines/qwen3-8b-fp8emb \
  --maxBatchSize=1
```

No special build flags are required — FP8 embedding is automatically enabled based on the safetensors metadata.

### Run Inference

Run inference with the built engine. No special flags are needed:

```bash
./build/examples/llm/llm_inference \
  --engineDir engines/qwen3-8b-fp8emb \
  --inputFile tests/test_cases/llm_basic.json \
  --outputFile output.json
```

---

## Combining with Other Quantization

FP8 embedding can be combined with weight quantization for maximum memory savings:

```bash
export EDGE_LLM_PATH=/path/to/TensorRT-Edge-LLM
cd $EDGE_LLM_PATH
export PYTHONPATH=$EDGE_LLM_PATH:$PYTHONPATH

# Step 1: Quantize weights to a checkpoint
tensorrt-edgellm-quantize llm \
  --model_dir /path/to/Qwen3-8B \
  --output_dir /tmp/qwen3_nvfp4 \
  --quantization nvfp4

# Step 2: Export with both NVFP4 weights and FP8 embedding
tensorrt-edgellm-export \
  /tmp/qwen3_nvfp4 \
  /tmp/qwen3_nvfp4_fp8emb_onnx \
  --fp8-embedding
```

---

## Technical Details

### Quantization Format

- **Format**: NVIDIA FP8 E4M3 (4 exponent bits, 3 mantissa bits)
- **Scale Granularity**: Per-row block-wise quantization with block size 128
- **Scale Shape**: `[vocab_size, hidden_size / 128]` — one scale per 128 elements in each row
- **Memory Reduction**: ~50% reduction compared to FP16 (8 bits vs 16 bits per element)

### Quantization Process

During export with `--fp8-embedding` in `tensorrt_edgellm`:
1. The embedding table is divided into blocks of 128 elements along the hidden dimension
2. For each block, the maximum absolute value is computed
3. Quantization scale is computed: `scale = amax / FP8_E4M3_MAX` (where `FP8_E4M3_MAX = 448.0`)
4. Each block is quantized to FP8 using its scale
5. Both the FP8 embedding table and scales are saved to safetensors

### Runtime Behavior

During inference:
- The embedding kernel detects FP8 format from the tensor dtype
- For each token lookup, the corresponding row is dequantized on-the-fly
- Dequantization uses the per-block scales: `output = fp8_value * scale`
- Output is always FP16 for downstream computation

---

## Notes

- **No Calibration Required**: Unlike FP8 KV cache, FP8 embedding uses direct min-max quantization and does not require calibration data.
- **Independent of Weight Quantization**: FP8 embedding can be enabled independently of weight quantization. You can use FP16 weights with FP8 embedding, or combine with NVFP4/INT4/FP8 weight quantization.
- **Automatic Detection**: Engine build and runtime automatically detect FP8 embedding from safetensors metadata — no special flags needed after export.
- **Memory Benefits**: Most beneficial for models with large vocabularies (e.g., 150K+ tokens) where embedding tables can be 500MB+ in FP16.
- **Accuracy**: Block-wise quantization with block size 128 preserves accuracy well in practice. The embedding lookup output is identical to FP16 within quantization tolerance.

---

## Limitations

- **TTS Models Not Supported**: FP8 embedding is not supported for TTS, Qwen3-Omni talker, and code_predictor models due to specialized kernel requirements. When requested for these models, a warning is logged and FP16 embedding is used instead.
- **Platform Requirements**: Requires CUDA 11.8+ for FP8 support and GPUs with compute capability SM89+ (Ada/Hopper/Blackwell) for native FP8 hardware support.
