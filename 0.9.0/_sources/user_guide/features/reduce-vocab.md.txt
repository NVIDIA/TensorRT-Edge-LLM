# Vocabulary Reduction

## Overview

Vocabulary reduction restricts generation logits to a task-specific subset of token IDs. The transformer layers and input embedding table keep the original tokenizer vocabulary; only the exported logits and runtime sampling vocabulary are reduced.

**Important**: The user is responsible for creating the vocabulary mapping. Since the optimal reduced vocabulary is directly tied to your specific task and output distribution, we only provide a simple reference script (`tensorrt-edgellm-reduce-vocab`). You should customize the vocabulary selection based on your use case.

---

## Quick Start

### End-to-End Workflow (Standard Decoding)

Using Qwen3-0.6B as an example — smaller models benefit most from vocabulary reduction since LM head represents a larger fraction of total compute:

```bash
export PYTHONPATH=/path/to/TensorRT-Edge-LLM:$PYTHONPATH

# Optional: quantize the source checkpoint first
tensorrt-edgellm-quantize llm \
  --model_dir Qwen/Qwen3-0.6B \
  --output_dir qwen3_0_6b_nvfp4 \
  --quantization nvfp4 \
  --lm_head_quantization nvfp4

# Step 1: Generate vocabulary mapping
tensorrt-edgellm-reduce-vocab \
  --model_dir Qwen/Qwen3-0.6B \
  --output_dir reduced_vocab \
  --reduced_vocab_size 16384 \
  --method input_aware \
  --max_samples 50000

# Step 2: Export model with reduced vocabulary
tensorrt-edgellm-export \
  qwen3_0_6b_nvfp4 \
  qwen3_0_6b_onnx \
  --reduced-vocab-dir reduced_vocab/

# Step 3: Build TensorRT engine (unchanged)
./build/examples/llm/llm_build \
  --onnxDir qwen3_0_6b_onnx/llm \
  --engineDir engines/qwen3-0.6b \
  --maxBatchSize 1

# Step 4: Run inference (unchanged)
./build/examples/llm/llm_inference \
  --engineDir engines/qwen3-0.6b \
  --inputFile input.json \
  --outputFile output.json
```

The runtime automatically applies vocabulary reduction when `config.json` has `reduced_vocab_size` and `vocab_map.safetensors` is present. `tensorrt_edgellm` reduces the LM-head weights during export, including FP16, FP8, NVFP4, MXFP8, INT8 SmoothQuant, and packed INT4 LM heads. Packed INT4 LM heads require `group_size=128` and a `reduced_vocab_size` that is a multiple of 128.

---

## EAGLE Speculative Decoding Support

When using vocabulary reduction for EAGLE base models, you must include all tokens referenced in the draft's `d2t.safetensors` mapping.

**Prerequisite**: Export the draft model first to generate `d2t.safetensors`, then use the `--d2t_path` flag:

```bash
# Step 0: Export EAGLE draft model first (generates d2t.safetensors)
tensorrt-edgellm-export \
  AngelSlim/Qwen3-4B_eagle3 \
  draft_onnx

# Step 1: Generate vocabulary mapping with d2t constraint
tensorrt-edgellm-reduce-vocab \
  --model_dir Qwen/Qwen3-4B \
  --output_dir reduced_vocab \
  --reduced_vocab_size 16384 \
  --method input_aware \
  --d2t_path draft_onnx/llm/d2t.safetensors

# Step 2: Export base model with reduced vocabulary
tensorrt-edgellm-export \
  Qwen/Qwen3-4B \
  base_onnx \
  --eagle-base \
  --reduced-vocab-dir reduced_vocab/

# Step 3-4: Build and run as usual...
```

This ensures all draft-to-target token mappings remain valid after vocabulary reduction.

---

## Script Reference

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--model_dir` | Yes | - | Path to model directory (tokenizer + config) |
| `--output_dir` | Yes | - | Output directory for vocabulary files |
| `--reduced_vocab_size` | Yes | - | Target vocabulary size |
| `--method` | No | `input_aware` | `input_aware` or `frequency` |
| `--max_samples` | No | `50000` | Max samples from dataset |
| `--d2t_path` | No | - | EAGLE d2t.safetensors path |

**Methods**:
- `input_aware`: Analyzes CNN/DailyMail summaries + documents, applies input-aware filtering
- `frequency`: Simple token frequency analysis on input documents

---

## Custom Vocabulary Mapping

The provided script uses CNN/DailyMail as a reference dataset. For production use, create your own files with the following formats:

### `vocab_map.safetensors`

| Key | Type | Shape | Description |
|-----|------|-------|-------------|
| `vocab_map` | `int32` | `(reduced_vocab_size,)` | Sorted original token IDs to keep (must include EOS token) |

```python
import torch
from safetensors.torch import save_file

# Select original token IDs to keep (must be sorted, must include EOS)
selected_tokens = [0, 1, 2, 100, 101, 500, 1000, 1001, ...]
vocab_map = torch.tensor(sorted(selected_tokens), dtype=torch.int32)

save_file({"vocab_map": vocab_map}, "vocab_map.safetensors")
```

### `reduced_vocab.json`

```json
{
  "vocab_size": 151936,
  "reduced_vocab_size": 16384
}
```

---

## Notes

- Vocabulary reduction only affects output logits and sampling; other layers are unchanged
- The `vocab_map.safetensors` file is automatically copied during export and engine build
- Runtime transparently handles the mapping — no inference code changes required
- Ensure your reduced vocabulary covers all tokens your model needs to generate
