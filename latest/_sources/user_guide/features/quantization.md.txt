# Quantization

Use `tensorrt-edgellm-quantize` when you start from an FP16/BF16 checkpoint and need to create a unified quantized checkpoint for `tensorrt_edgellm`.

The quantization CLI writes a unified HuggingFace-style checkpoint directory that `tensorrt_edgellm` can export directly.

Skip this step when you already have a supported pre-quantized HuggingFace checkpoint.

## Setup

Install `requirements.txt` and the `tools` extra from the Installation Guide before running the quantization CLI.

```bash
export EDGE_LLM_PATH=/path/to/TensorRT-Edge-LLM
cd $EDGE_LLM_PATH
export PYTHONPATH=$EDGE_LLM_PATH:$PYTHONPATH
tensorrt-edgellm-quantize --help
```

## Quantize An LLM

```bash
tensorrt-edgellm-quantize llm \
  --model_dir Qwen/Qwen3-0.6B \
  --output_dir /tmp/qwen3_0_6b_nvfp4 \
  --quantization nvfp4 \
  --lm_head_quantization nvfp4
```

Calibration uses the default `cnn_dailymail` text dataset. Pick another with
`--text_dataset <name>` (see [Calibration Datasets](#calibration-datasets)).

The output directory is a HuggingFace-style checkpoint that `tensorrt_edgellm` can export directly. See [Quick Start Guide](../getting_started/quick-start-guide.md) for the full export-build-inference workflow.

## Enable FP8 KV Cache

```bash
tensorrt-edgellm-quantize llm \
  --model_dir Qwen/Qwen3-8B \
  --output_dir /tmp/qwen3_nvfp4_fp8kv \
  --quantization nvfp4 \
  --kv_cache_quantization fp8
```

When this checkpoint is exported with `tensorrt_edgellm`, FP8 KV cache is detected from the checkpoint metadata automatically. See [FP8 KV Cache](FP8KV.md) for details on FP8 KV cache behavior and platform requirements.

## Quantize A Visual Tower To FP8

For supported VLM checkpoints, the standalone quantizer can also calibrate the
visual tower to FP8. Use a multimodal calibration dataset so the visual path sees
real image activations:

```bash
tensorrt-edgellm-quantize llm \
  --model_dir Qwen/Qwen3-VL-2B-Instruct \
  --output_dir /tmp/qwen3_vl_fp8_visual \
  --quantization nvfp4 \
  --lm_head_quantization nvfp4 \
  --visual_quantization fp8 \
  --image_dataset mmmu
```

`--image_dataset` selects the image-question calibration set (default `mmmu`);
the LLM path still uses `--text_dataset`.

## Quantize Qwen3-ASR Audio To FP8

Qwen3-ASR calibration streams audio-transcript pairs through the audio tower
and text decoder. Use an audio dataset with `audio` and `text` columns:

```bash
tensorrt-edgellm-quantize llm \
  --model_dir /path/to/qwen3_asr \
  --output_dir /tmp/qwen3_asr_fp8_audio \
  --quantization fp8 \
  --audio_quantization fp8 \
  --audio_dataset librispeech
```

## Calibration Datasets

Calibration data is selected **by name**, per modality — the CLI passes only a
dataset name. Each name maps to a registered dataset implementation; an
unknown name fails out with the list of available datasets and a pointer to
the customization guide (it does not silently fall back).

| Modality | Flag (default) | Built-in names | Used for |
|---|---|---|---|
| Text  | `--text_dataset` (`cnn_dailymail`) | `cnn_dailymail`, `wikitext` | LLM weights, LM head, KV cache, CodePredictor, MTP, EAGLE3, DFlash |
| Image | `--image_dataset` (`mmmu`) | `mmmu` | Visual tower (`--visual_quantization`) |
| Audio | `--audio_dataset` (`librispeech`) | `librispeech` | Qwen3-ASR / audio tower |

To calibrate on your own data, write a generator for your modality and
register it under a name — see
[Calibration Dataset Customization](../../developer_guide/customization/calibration-datasets.md).
To run a built-in offline (e.g. CI), point it at a cached copy with the
`EDGELLM_QUANT_DATASET_<NAME>` environment variable; the CLI stays name-only.

## Quantize An EAGLE3 Draft

```bash
tensorrt-edgellm-quantize draft \
  --base_model_dir /path/to/base_model \
  --draft_model_dir /path/to/eagle3_draft \
  --output_dir /tmp/eagle3_draft_fp8 \
  --quantization fp8
```

## Quantize A DFlash Draft

```bash
tensorrt-edgellm-quantize draft \
  --base_model_dir /path/to/base_model \
  --draft_model_dir /path/to/dflash_draft \
  --output_dir /tmp/dflash_draft_nvfp4 \
  --quantization nvfp4 \
  --lm_head_quantization nvfp4
```

The `draft` command auto-detects DFlash from the draft checkpoint's
`dflash_config`. The base model is used during calibration to produce target
hidden states and as an LM-head fallback for draft checkpoints that do not
store `lm_head` weights. DFlash draft quantization has been validated with
NVFP4. The target-hidden projector stays in dense precision; export will
reject checkpoints where that `fc` module is quantized.

## Quantize Embedding Table To FP8

FP8 embedding quantization is applied at export time via `tensorrt_edgellm`, not during the quantization step. Pass `--fp8-embedding` when exporting the quantized checkpoint. See [FP8 Embedding](fp8-embedding.md) for details and usage examples. Export the runtime embedding table in FP8:

```bash
tensorrt-edgellm-export \
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
