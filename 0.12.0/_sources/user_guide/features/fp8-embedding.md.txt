# FP8 Embedding

## Overview

FP8 embedding reduces memory usage by quantizing the embedding table from FP16 to FP8 (NVIDIA FP8 E4M3 format), achieving **~50% memory reduction** for the embedding table. This is particularly beneficial for models with large vocabularies where the embedding table can be a significant portion of total memory usage.

For newly quantized checkpoints, prefer
`trtedge quantize llm --token_embedding_quantization fp8` so ModelOpt
records the token-table tensor layout and metadata in the checkpoint. This
format uses one FP32 scale for the entire table, as does export-time FP8 and
Gemma4 PLE FP8. Quantized values stay compressed until lookup.
`trtedge export --fp8-embedding` remains the compatibility option for
older checkpoints with a dense token table.

**Key Points**:
- Reduces embedding table memory by ~50% (FP8 vs FP16)
- Uses NVIDIA FP8 E4M3 format with one per-tensor FP32 scale
- Can be used independently or combined with other quantization methods (NVFP4, INT4, FP8 weights)
- Detected by the runtime from `embedding.safetensors`; no build or runtime flags needed
- Output is always FP16 — dequantization happens on-the-fly during embedding lookup

---

## Workflow

Export with `--fp8-embedding` to write the runtime embedding sidecar in FP8:

```bash
export EDGE_LLM_PATH=/path/to/TensorRT-Edge-LLM
cd $EDGE_LLM_PATH
export PYTHONPATH=$EDGE_LLM_PATH:$PYTHONPATH

trtedge export \
  /path/to/Qwen3-8B \
  /tmp/qwen3_onnx_fp8emb \
  --fp8-embedding
```

This flag only changes `embedding.safetensors`; it does not change the ONNX graph or the checkpoint weights. The sidecar contains the FP8 embedding table plus `embedding_scale`.

### Build Engine

Build the TensorRT engine as usual. The builder copies `embedding.safetensors` to the engine directory:

```bash
./build/bin/llm_build \
  --onnx-dir /tmp/qwen3_onnx_fp8emb/llm \
  --engine-dir engines/qwen3-8b-fp8emb \
  --max-batch-size=1
```

No build flags are needed.

### Run Inference

Run inference with the built engine. No special flags are needed:

```bash
./build/bin/llm_inference \
  --engine-dir engines/qwen3-8b-fp8emb \
  --input-file tests/test_cases/llm_basic.json \
  --output-file output.json
```

---

## Combining with Other Quantization

FP8 embedding can be combined with weight quantization for maximum memory savings:

```bash
export EDGE_LLM_PATH=/path/to/TensorRT-Edge-LLM
cd $EDGE_LLM_PATH
export PYTHONPATH=$EDGE_LLM_PATH:$PYTHONPATH

# Step 1: Quantize weights to a checkpoint
trtedge quantize llm \
  --model-dir /path/to/Qwen3-8B \
  --output-dir /tmp/qwen3_nvfp4 \
  --quantization nvfp4

# Step 2: Export with both NVFP4 weights and FP8 embedding
trtedge export \
  /tmp/qwen3_nvfp4 \
  /tmp/qwen3_nvfp4_fp8emb_onnx \
  --fp8-embedding
```

---

## Technical Details

### Quantization Format

- **Format**: NVIDIA FP8 E4M3 (4 exponent bits, 3 mantissa bits)
- **Scale Granularity**: Per tensor
- **Scale Shape**: `[1, 1]` for token artifacts; one FP32 value for PLE
- **Memory Reduction**: ~50% reduction compared to FP16 (8 bits vs 16 bits per element)

### Quantization Process

During export with `--fp8-embedding` in `tensorrt_edgellm`:
1. The maximum absolute value of the complete table is computed
2. Rows are processed in bounded chunks, without a full FP32 table copy
3. Quantization scale is computed: `scale = amax / FP8_E4M3_MAX` (where `FP8_E4M3_MAX = 448.0`)
4. Each row is quantized to FP8 using the tensor scale
5. Both the FP8 embedding table and scales are saved to safetensors

### Runtime Behavior

During inference:
- The embedding kernel detects FP8 format from the tensor dtype
- For each token lookup, the corresponding row is dequantized on-the-fly
- Dequantization uses the tensor scale: `output = FP16(FP32(fp8_value) * scale)`
- Output is always FP16 for downstream computation

---

## Notes

- **No Calibration Required**: Unlike FP8 KV cache, FP8 embedding uses direct min-max quantization and does not require calibration data.
- **Independent of Weight Quantization**: FP8 embedding can be enabled independently of weight quantization. You can use FP16 weights with FP8 embedding, or combine with NVFP4/INT4/FP8 weight quantization.
- **Automatic Detection**: The runtime detects FP8 embedding from the table's data type; no flags are needed after export.
- **Memory Benefits**: Most beneficial for models with large vocabularies (e.g., 150K+ tokens) where embedding tables can be 500MB+ in FP16.
- **Accuracy**: Quantization changes values. Evaluate dataset accuracy against the original checkpoint for the intended workload; bitwise equality is not expected.

For host-backed token storage, pass `--offloadEmbedding` to `llm_inference`.
The table remains compressed in CUDA-pinned host memory and the same CUDA
lookup kernel reads it through unified virtual addressing. PLE additionally
supports file-backed caching through `--pleResidency cached --pleCacheSlots 4096`.

---

## Limitations

- **TTS Models Not Supported**: FP8 embedding is not supported for TTS, Qwen3-Omni talker, and code_predictor models due to specialized kernel requirements. When requested for these models, a warning is logged and FP16 embedding is used instead.
- **Platform Requirements**: Requires CUDA 11.8+ for FP8 support and GPUs with compute capability SM89+ (Ada/Hopper/Blackwell) for native FP8 hardware support.
