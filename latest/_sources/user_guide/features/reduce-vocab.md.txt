# Vocabulary Reduction

## Overview

Vocabulary reduction restricts generation logits to a task-specific subset of token IDs. The transformer layers and input embedding table keep the original tokenizer vocabulary; only the exported logits and runtime sampling vocabulary are reduced.

**Important**: The user is responsible for creating the vocabulary mapping. Since the optimal reduced vocabulary is directly tied to your specific task and output distribution, we only provide a simple reference script (`trtedge reduce-vocab`). You should customize the vocabulary selection based on your use case.

---

## Quick Start

### End-to-End Workflow (Standard Decoding)

Using Qwen3-0.6B as an example — smaller models benefit most from vocabulary reduction since LM head represents a larger fraction of total compute:

```bash
export PYTHONPATH=/path/to/TensorRT-Edge-LLM:$PYTHONPATH

# Optional: quantize the source checkpoint first
trtedge quantize llm \
  --model-dir Qwen/Qwen3-0.6B \
  --output-dir qwen3_0_6b_nvfp4 \
  --quantization nvfp4 \
  --lm-head-quantization nvfp4

# Step 1: Generate vocabulary mapping
trtedge reduce-vocab \
  --model-dir Qwen/Qwen3-0.6B \
  --output-dir reduced_vocab \
  --reduced-vocab-size 16384 \
  --method input_aware \
  --max-samples 50000

# Step 2: Export model with reduced vocabulary
trtedge export \
  qwen3_0_6b_nvfp4 \
  qwen3_0_6b_onnx \
  --reduced-vocab-dir reduced_vocab/

# Step 3: Build TensorRT engine (unchanged)
./build/examples/llm/llm_build \
  --onnx-dir qwen3_0_6b_onnx/llm \
  --engine-dir engines/qwen3-0.6b \
  --max-batch-size 1

# Step 4: Run inference (unchanged)
./build/examples/llm/llm_inference \
  --engine-dir engines/qwen3-0.6b \
  --input-file input.json \
  --output-file output.json
```

The runtime automatically applies vocabulary reduction when `config.json` has `reduced_vocab_size` and `vocab_map.safetensors` is present. `tensorrt_edgellm` reduces the LM-head weights during export, including FP16, FP8, NVFP4, MXFP8, INT8 SmoothQuant, and packed INT4 LM heads. Packed INT4 LM heads require `group_size=128` and a `reduced_vocab_size` that is a multiple of 128.

---

## EAGLE Speculative Decoding Support

When using vocabulary reduction for EAGLE base models, you must include all tokens referenced in the draft's `d2t.safetensors` mapping.

**Prerequisite**: Export the draft model first to generate `d2t.safetensors`, then use the `--d2t-path` flag:

```bash
# Step 0: Export EAGLE draft model first (generates d2t.safetensors)
trtedge export \
  AngelSlim/Qwen3-4B_eagle3 \
  draft_onnx

# Step 1: Generate vocabulary mapping with d2t constraint
trtedge reduce-vocab \
  --model-dir Qwen/Qwen3-4B \
  --output-dir reduced_vocab \
  --reduced-vocab-size 16384 \
  --method input_aware \
  --d2t-path draft_onnx/llm/d2t.safetensors

# Step 2: Export base model with reduced vocabulary
trtedge export \
  Qwen/Qwen3-4B \
  base_onnx \
  --eagle-base \
  --reduced-vocab-dir reduced_vocab/

# Step 3-4: Build and run as usual...
```

This ensures all draft-to-target token mappings remain valid after vocabulary reduction.

---

## DFlash Speculative Decoding Support

Vocabulary reduction can be applied independently to the DFlash **draft** model. Unlike EAGLE, the DFlash draft does not consume a `d2t` mapping; the draft's reduced LM-head IDs are remapped to the full base vocabulary inside `DFlashDecoder` via `mapReducedVocabToFullVocab`, so no draft↔base coordination is required.

Reducing the draft LM-head shrinks the per-step argmax over the draft vocabulary, the dominant cost of drafter forward passes for large vocabularies (e.g. Qwen3 family at 151936). The reduced vocabulary size is **an arbitrary value you choose** via `--reduced-vocab-size` on the `trtedge reduce-vocab` CLI; the table below reports the three sizes we happened to sweep on Qwen3-8B DFlash (Thor), not a hardcoded set of supported sizes.

| Draft vocab | Tok/sec | Accepted/iter | Speedup vs full |
|---|---|---|---|
| full (151936) | 18.5 | 1.45 | 1.00× |
| 128K | 18.7 | 1.45 | 1.01× |
| **64K** | **19.3** | **1.44** | **1.04×** |
| 32K | 18.1 | 1.35 | 0.97× |

Of the sizes we tried, **64K** was the sweet spot on this configuration: acceptance essentially unchanged (1.44 vs 1.45) while throughput improved ~4%. At 32K, acceptance dropped enough (1.35) that the drafter-latency win was undone. These numbers are for one model + one dataset — sweep your own sizes on your own workload before committing to a setting.

```bash
# Step 1: Generate vocabulary mapping for the DFlash draft
trtedge reduce-vocab \
  --model-dir Qwen/Qwen3.5-4B \
  --output-dir draft_reduced_vocab \
  --reduced-vocab-size 65536 \
  --method input_aware

# Step 2: Export DFlash draft with reduced vocabulary
trtedge export \
  Qwen/Qwen3.5-4B \
  qwen3_5_4b/onnx/draft_export \
  --dflash-draft \
  --dflash-draft-dir Qwen3.5-4B-DFlash \
  --draft-reduced-vocab-dir draft_reduced_vocab/

# The base model export (--dflash-base) is unchanged; reduction is draft-only.
```

The export writes `draft_vocab_map.safetensors` next to the draft engine. At runtime, `DFlashDecoder` gates on the draft engine config's `reduced_vocab_size > 0`: a missing map file when the config declares reduction is a hard error, and a stray map file when the config does not declare reduction is ignored. Base-model vocab reduction and draft-model vocab reduction are orthogonal and can be combined.

---

## MTP Speculative Decoding Support

Chain-MTP can reduce only the draft LM head while preserving the base model's full vocabulary. Use this path with `--spec-draft-top-k 1`; reduced-vocabulary tree-MTP is not supported because its tree path would require a direct map while the MTP decoder stores an offset map. Unlike EAGLE, chain-MTP does not require a `d2t` constraint. Base reduction through `--reduced-vocab-dir` and draft reduction through `--draft-reduced-vocab-dir` are independent and may be used separately or together.

```bash
# Step 1: Generate a draft vocabulary map
trtedge reduce-vocab \
  --model-dir Qwen/Qwen3.6-27B \
  --output-dir draft_reduced_vocab \
  --reduced-vocab-size 65536 \
  --method input_aware \
  --max-samples 50000

# Step 2: Export the MTP draft with reduced logits
trtedge export \
  Qwen/Qwen3.6-27B \
  qwen3_6_27b/onnx \
  --mtp \
  --draft-reduced-vocab-dir draft_reduced_vocab/
```

The draft engine config declares the reduced vocabulary size. When that value is positive, `draft_vocab_map.safetensors` is required in the engine directory; when it is zero, a stray sidecar is ignored. The sidecar stores direct reduced-to-full token IDs, and the MTP runtime converts them once at load time to the offset representation used by the proposal kernels.

---

## Script Reference

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--model-dir` | Yes | - | Path to model directory (tokenizer + config) |
| `--output-dir` | Yes | - | Output directory for vocabulary files |
| `--reduced-vocab-size` | Yes | - | Target vocabulary size |
| `--method` | No | `input_aware` | `input_aware` or `frequency` |
| `--max-samples` | No | `50000` | Max samples from dataset |
| `--d2t-path` | No | - | EAGLE d2t.safetensors path |

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
