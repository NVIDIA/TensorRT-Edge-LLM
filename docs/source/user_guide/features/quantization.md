# Quantization

Use `trtedge quantize` when you start from an FP16/BF16 checkpoint and need to create a unified quantized checkpoint for `tensorrt_edgellm`.

The quantization CLI writes a unified HuggingFace-style checkpoint directory that `tensorrt_edgellm` can export directly.

Skip this step when you already have a supported pre-quantized HuggingFace checkpoint.

## Select W4A4 or W4A16 for NVFP4 checkpoints

ModelOpt W4A4 checkpoints can run the same packed NVFP4 weights with NVFP4 or
FP16 activations. Export without an activation override for W4A4:

```bash
tensorrt-edgellm-export /path/to/modelopt_nvfp4 /tmp/model_w4a4 \
  --target-sm 110

tensorrt-edgellm-export /path/to/modelopt_nvfp4 /tmp/model_w4a16 \
  --no-quantize-activations --target-sm 110
```

The default uses the checkpoint's activation scales and the W4A4 GEMM/MoE
path. `--no-quantize-activations` keeps activations in FP16 and selects the
W4A16 GEMM/MoE path. Both modes reuse the checkpoint's packed weights.

The checkpoint-direct builder uses the same selection:

```bash
tensorrt-edgellm-build --model-dir /path/to/modelopt_nvfp4 \
  --engine-dir /tmp/engine_w4a16 --no-quantize-activations
```

## Setup

Complete [Installation](../getting_started/installation.md) and add the
`tools` extra described in
[Optional Python dependencies](../getting_started/installation.md#optional-python-dependencies).
This provides the dependencies for the quantization and ONNX export examples
below. Then verify the command:

```bash
trtedge quantize --help
```

## Quantize An LLM

```bash
trtedge quantize llm \
  --model-dir Qwen/Qwen3-0.6B \
  --output-dir /tmp/qwen3_0_6b_nvfp4 \
  --quantization nvfp4 \
  --lm-head-quantization nvfp4
```

Calibration uses the default `cnn_dailymail` text dataset. Pick another with
`--text-dataset <name>` (see [Calibration Datasets](#calibration-datasets)).

The output directory is a HuggingFace-style checkpoint that `tensorrt_edgellm` can export directly. See [Quick Start Guide](../getting_started/quick-start-guide.md) for the full export-build-inference workflow.

## Enable FP8 KV Cache

```bash
trtedge quantize llm \
  --model-dir Qwen/Qwen3-8B \
  --output-dir /tmp/qwen3_nvfp4_fp8kv \
  --quantization nvfp4 \
  --kv-cache-quantization fp8
```

When this checkpoint is exported with `tensorrt_edgellm`, FP8 KV cache is detected from the checkpoint metadata automatically. See [FP8 KV Cache](FP8KV.md) for details on FP8 KV cache behavior and platform requirements.

## Quantize A Visual Tower To FP8

For supported VLM checkpoints, the standalone quantizer can also calibrate the
visual tower to FP8. Use a multimodal calibration dataset so the visual path sees
real image activations:

```bash
trtedge quantize llm \
  --model-dir Qwen/Qwen3-VL-2B-Instruct \
  --output-dir /tmp/qwen3_vl_fp8_visual \
  --quantization nvfp4 \
  --lm-head-quantization nvfp4 \
  --visual-quantization fp8 \
  --image-dataset mmmu
```

`--image-dataset` selects the image-question calibration set (default `mmmu`);
the LLM path still uses `--text-dataset`.

## Quantize Qwen3-ASR Audio To FP8

Qwen3-ASR calibration streams audio-transcript pairs through the audio tower
and text decoder. Use an audio dataset with `audio` and `text` columns:

```bash
trtedge quantize llm \
  --model-dir /path/to/qwen3_asr \
  --output-dir /tmp/qwen3_asr_fp8_audio \
  --quantization fp8 \
  --audio-quantization fp8 \
  --audio-dataset librispeech
```

## Calibration Datasets

Calibration data is selected **by name**, per modality — the CLI passes only a
dataset name. Each name maps to a registered dataset implementation; an
unknown name fails out with the list of available datasets and a pointer to
the customization guide (it does not silently fall back).

| Modality | Flag (default) | Built-in names | Used for |
|---|---|---|---|
| Text  | `--text-dataset` (`cnn_dailymail`) | `cnn_dailymail`, `wikitext` | LLM weights, LM head, KV cache, CodePredictor, MTP, EAGLE3, DFlash |
| Image | `--image-dataset` (`mmmu`) | `mmmu` | Visual tower (`--visual-quantization`) |
| Audio | `--audio-dataset` (`librispeech`) | `librispeech` | Qwen3-ASR / audio tower |

To calibrate on your own data, write a generator for your modality and
register it under a name — see
[Calibration Dataset Customization](../../developer_guide/customization/calibration-datasets.md).
To run a built-in offline (e.g. CI), point it at a cached copy with the
`EDGELLM_QUANT_DATASET_<NAME>` environment variable; the CLI stays name-only.

## Quantize An EAGLE3 Draft

```bash
trtedge quantize draft \
  --base-model-dir /path/to/base_model \
  --draft-model-dir /path/to/eagle3_draft \
  --output-dir /tmp/eagle3_draft_fp8 \
  --quantization fp8
```

## Quantize A DFlash Draft

```bash
trtedge quantize draft \
  --base-model-dir /path/to/base_model \
  --draft-model-dir /path/to/dflash_draft \
  --output-dir /tmp/dflash_draft_nvfp4 \
  --quantization nvfp4 \
  --lm-head-quantization nvfp4
```

The `draft` command auto-detects DFlash from the draft checkpoint's
`dflash_config`. The base model is used during calibration to produce target
hidden states and as an LM-head fallback for draft checkpoints that do not
store `lm_head` weights. DFlash draft quantization has been validated with
NVFP4. The target-hidden projector stays in dense precision; export will
reject checkpoints where that `fc` module is quantized.

## Export NVFP4 MoE for SM12x

NVFP4 MoE weights use different export layouts on SM120/SM121 and
SM100/SM101/SM110. Select the SM12x layout before exporting for DGX Spark
(SM121) or an SM120 GPU:

```bash
export EDGELLM_NVFP4_MOE_TARGET=sm12x
trtedge export \
  /path/to/nvfp4_moe_checkpoint \
  /tmp/nvfp4_moe_sm12x_onnx
```

Without this setting, export uses the SM100/SM101/SM110 layout. The selected
weight layout is fixed in the exported artifacts and cannot be changed during
engine build or inference. Use a separate export directory for each layout.

## Quantize Embedding Table To FP8

FP8 embedding quantization is applied at export time via `tensorrt_edgellm`, not during the quantization step. Pass `--fp8-embedding` when exporting the quantized checkpoint. See [FP8 Embedding](fp8-embedding.md) for details and usage examples. Export the runtime embedding table in FP8:

```bash
trtedge export \
  /tmp/qwen35_nvfp4 \
  /tmp/qwen35_onnx \
  --fp8-embedding
```

## Supported Methods

| Component | Methods |
|---|---|
| Backbone | `fp8`, `int4_awq`, `nvfp4`, `mxfp8`, `int8_sq` |
| LM head | `fp8`, `int4_awq`, `nvfp4`, `mxfp8` |
| KV cache | `fp8` |
| Visual tower | `fp8` |
| Audio tower | `fp8` |
| CodePredictor | `fp8` |
| DFlash draft | validated with `nvfp4` backbone and optional `nvfp4` LM-head quantization |

## Notes

- The package writes unified checkpoints only. It does not export ONNX or build TensorRT engines.
- GPTQ checkpoints are loaded as pre-quantized checkpoints; this package does not create GPTQ models.
