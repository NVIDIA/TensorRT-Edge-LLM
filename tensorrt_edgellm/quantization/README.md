# Standalone Quantization for TensorRT Edge-LLM

Decoupled quantization pipeline that produces **unified HuggingFace
checkpoints** (safetensors + config) consumable by `tensorrt_edgellm`.
Install the quantization dependencies directly with
`pip install -e ".[quantize]"`.

## Design

### Problem

The old quantization path imported model-specific loaders, custom decoder
layers, ONNX plugins, and TensorRT native ops. This coupled quantization
(a GPU-only, PyTorch-only step) with the export pipeline, making the venv
heavy and brittle.

### Solution

```
┌──────────────────────────────────────────────┐
│           tensorrt_edgellm/quantization          │
│                                              │
│  quantization_configs.py  Recipe presets     │
│  quantize.py              Load → quant → save│
│  models/                                     │
│    eagle3_draft.py        Eagle3 draft model │
│    eagle4_draft.py        EAGLE-4 drafter    │
│    dflash_draft.py        DFlash draft model │
│  ../scripts/quantize.py    CLI entry point   │
└──────────────────┬───────────────────────────┘
                   │ unified checkpoint
                   ▼
┌──────────────────────────────────────────────┐
│          tensorrt_edgellm             │
│                                              │
│  Load checkpoint → build ONNX models        │
└──────────────────────────────────────────────┘
```

**Key decisions:**

1. **Model loading** uses only `AutoModelForCausalLM` /
   `AutoModelForImageTextToText`.  Required preprocessing, such as
   Phi-4-Multimodal LoRA merge, should be run before this quantization step.

2. **`models/` subfolder** holds standalone model implementations for
   architectures not available in HuggingFace `transformers`.  For
   example, EAGLE3 (speculative-decoding draft model) is reimplemented
   using standalone PyTorch building blocks (`RMSNorm`, `SwiGLUMLP`,
   `RotaryEmbedding`) with no dependency on `transformers` model
   classes.  Only the calibration `forward` is implemented; the full
   inference forward with KV-cache and GatherND belongs in the export
   layer.

3. **Calibration** uses explicit datasets by modality: text data for LLM,
   LM-head, KV-cache, CodePredictor, MTP, EAGLE3, and DFlash quantization;
   image-question data when `--visual_quantization fp8` is requested; and
   audio-transcript data for Qwen3-ASR/audio quantization.

4. **Output** is always a unified HuggingFace checkpoint produced by
   `modelopt.torch.export.export_hf_checkpoint`.  This is the format
   that `tensorrt_edgellm` reads.

### Supported quantization methods

| Backbone        | LM-head         | Token embedding | KV-cache | Visual tower | Audio tower | CodePredictor |
|-----------------|-----------------|-----------------|----------|--------------|-------------|---------------|
| `fp8`           | `fp8`           | `fp8`           | `fp8`    | `fp8`        | `fp8`       | `fp8`         |
| `int4_awq`      | `int4_awq`      | —               | —        | —            | —           | —             |
| `nvfp4`         | `nvfp4`         | `nvfp4`         | —        | —            | —           | —             |
| `mxfp8`         | `mxfp8`         | —               | —        | —            | —           | —             |
| `int8_sq`       | —               | —               | —        | —            | —           | —             |

Any backbone method can be combined with any lm_head method.  Examples:

- `nvfp4` backbone + `nvfp4` lm_head → full NVFP4
- `nvfp4` backbone + `fp8` lm_head → mixed precision
- `nvfp4` backbone + `int4_awq` lm_head → INT4 lm_head
- `fp8` backbone + `fp8` lm_head + `fp8` kv_cache → full FP8

The token embedding is independently selectable. For a checkpoint with a tied
token embedding and LM head, the workflow separates
their storage before exporting the embedding and linear-layer layouts. The
token embedding may be quantized while leaving the LM head in FP16.

Gemma4 E2B/E4B also have a separate packed per-layer embedding (PLE) table.
Use `--ple_embedding_quantization fp8` to store that table as per-tensor FP8;
the ONNX exporter, checkpoint-direct builder, and runtime gather dequantize it with its
ModelOpt scale and Gemma4 embedding scale. This option is independent of the
canonical token embedding. On a BF16 Qwen4-Exp checkpoint the same option
writes the n-gram table as FP8 shards with one scalar `weight_scale`, the layout
of the released FP8 checkpoints.

For export-time quantization of a dense checkpoint, use
`tensorrt-edgellm-export MODEL_DIR OUTPUT_DIR --fp8-embedding --fp8-ple`.
The token and PLE flags are independent. `--fp8-ple` writes the PLE table
under `OUTPUT_DIR/ple/`; keep this directory accessible at runtime.

## Usage

### LLM quantization

```bash
tensorrt-edgellm-quantize llm \
    --model_dir /path/to/Qwen3.5-0.8B \
    --output_dir /path/to/output \
    --quantization nvfp4 \
    --lm_head_quantization nvfp4 \
    --token_embedding_quantization nvfp4
```

For Gemma4 E2B/E4B, use `--token_embedding_quantization fp8` and
`--ple_embedding_quantization fp8` together to quantize both input tables.
Leave the E4B LM head in FP16: its separate FP8 compiled path currently
diverges from the provider's output.

Calibration uses the default `cnn_dailymail` text dataset; pick another with
`--text_dataset <name>`.

### Speculative draft quantization

```bash
tensorrt-edgellm-quantize draft \
    --base_model_dir /path/to/base_model \
    --draft_model_dir /path/to/eagle3_draft \
    --output_dir /path/to/output \
    --quantization fp8
```

The `draft` command auto-detects DFlash draft checkpoints from
`config.json::dflash_config`; otherwise it uses the Eagle3 draft flow.

DFlash draft quantization has been validated with NVFP4:

```bash
tensorrt-edgellm-quantize draft \
    --base_model_dir /path/to/base_model \
    --draft_model_dir /path/to/dflash_draft \
    --output_dir /path/to/output \
    --quantization nvfp4 \
    --lm_head_quantization nvfp4
```

DFlash uses the base model only for calibration hidden states and for the
LM-head fallback when the draft checkpoint does not carry its own LM head.
The DFlash target-hidden projector `fc` is intentionally excluded from PTQ so
the exporter can keep its full-FP32 accumulation path.

### Common options

| Flag | Default | Description |
|------|---------|-------------|
| `--quantization` | None | Backbone method |
| `--lm_head_quantization` | None | LM-head method |
| `--token_embedding_quantization` | None | Canonical runtime token table (`fp8` per-tensor or `nvfp4` block-16) |
| `--ple_embedding_quantization` | None | Gemma4 PLE or Qwen4-Exp n-gram table (`fp8` per-tensor) |
| `--kv_cache_quantization` | None | KV-cache method (fp8) |
| `--visual_quantization` | None | Visual tower method (fp8) |
| `--audio_quantization` | None | Audio tower method (fp8) |
| `--cp_quantization` | None | Qwen3-Omni / Qwen3-TTS CodePredictor method (fp8) |
| `--dtype` | fp16 | Loading dtype |
| `--device` | cuda | CUDA device |
| `--text_dataset` | cnn_dailymail | Registered text dataset name (`cnn_dailymail`, `wikitext`) |
| `--image_dataset` | mmmu | Registered image dataset name (used with `--visual_quantization`) |
| `--audio_dataset` | librispeech | Registered audio dataset name (Qwen3-ASR / audio tower) |
| `--num_samples` | 512 | Number of calibration samples |

Calibration datasets are selected **by name**. An unknown name fails out with
the list of registered datasets and a pointer to the customization guide. To
add your own dataset, or run a built-in offline, see
[Calibration Dataset Customization](../../docs/source/developer_guide/customization/calibration-datasets.md).

### Python API

```python
from tensorrt_edgellm.quantization import quantize_and_export

quantize_and_export(
    model_dir="/path/to/model",
    output_dir="/path/to/output",
    quantization="nvfp4",
    lm_head_quantization="nvfp4",
    text_dataset="cnn_dailymail",   # registered name, instance, or None
)
```

## End-to-end workflow

This package covers **step 1** (quantization) of the TensorRT Edge-LLM
export pipeline.  The full workflow is:

```bash
# 1. Quantize (this package — runs on any CUDA host)
tensorrt-edgellm-quantize llm \
    --model_dir /data/Qwen3.5-0.8B \
    --output_dir /tmp/qwen35-nvfp4 \
    --quantization nvfp4 --lm_head_quantization nvfp4

# 2. Export to ONNX (tensorrt_edgellm — runs on any CUDA host)
tensorrt-edgellm-export \
    /tmp/qwen35-nvfp4 /tmp/qwen35-onnx
```

Steps 3 (TRT engine build) and 4 (inference) run on the **edge device**
using the C++ runtime.  See the
[Quick Start Guide](https://nvidia.github.io/TensorRT-Edge-LLM/latest/user_guide/getting_started/quick-start-guide.html)
for engine build and inference instructions.
