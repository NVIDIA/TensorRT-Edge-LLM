# Quantization Package Design

`tensorrt_edgellm.quantization` is a standalone checkpoint quantization package. It converts FP16 HuggingFace checkpoints into quantized HuggingFace-style checkpoints that either frontend can consume.

For command-line usage, see [Quantization](../../user_guide/features/quantization.md).

## Design Goals

- Keep quantization separate from both frontends.
- Preserve a HuggingFace-compatible checkpoint layout.
- Store quantization metadata in checkpoint config files so `tensorrt_edgellm` can infer export behavior from the checkpoint.
- Share the same output contract for locally quantized and downloaded pre-quantized checkpoints.

## Package Layout

| Path | Role |
|---|---|
| `tensorrt_edgellm/scripts/quantize.py` | User-facing `llm` and `draft` commands |
| `tensorrt_edgellm/quantization/quantize.py` | Model loading, ModelOpt configuration, calibration, and checkpoint writing |
| `tensorrt_edgellm/quantization/quantization_configs.py` | Shared ModelOpt quantization configurations |
| `tensorrt_edgellm/checkpoint/loader.py` | Loads quantized checkpoint tensors for consumers |
| `tensorrt_edgellm/quantization/models/eagle3_draft.py` | Standalone Eagle3 draft calibration model |
| `tensorrt_edgellm/quantization/models/eagle4_draft.py` | Standalone EAGLE-4 draft calibration model |
| `tensorrt_edgellm/quantization/models/dflash_draft.py` | Standalone DFlash draft calibration model |

## Flow

```text
FP16 checkpoint
  -> tensorrt_edgellm.quantization
  -> quantized safetensors + quantization metadata
  -> ONNX export or checkpoint-direct engine build
```

The quantization package stops after writing the checkpoint. The ONNX and
checkpoint-direct frontends independently read its metadata; neither owns the
quantization workflow.

## Artifact Contract

The output directory must contain:

- `config.json` with model architecture fields.
- Quantization metadata such as `quantization_config`, `hf_quant_config.json`, or equivalent fields understood by `tensorrt_edgellm`.
- One or more `.safetensors` checkpoint shards.
- Tokenizer and processor files needed by the runtime.

Each frontend uses this metadata to select quantized layers and repack checkpoint tensors. FP8 KV cache is inferred from checkpoints that mark KV cache quantization as `fp8`.

## Supported Methods

| Component | Methods |
|---|---|
| Backbone | `fp8`, `int4_awq`, `nvfp4`, `mxfp8`, `int8_sq` |
| LM head | `fp8`, `int4_awq`, `nvfp4`, `mxfp8` |
| KV cache | `fp8` |
| Visual tower | `fp8` |
| Audio tower | `fp8` |

Backbone and LM-head methods can be combined for mixed-precision checkpoints.
Calibration data is selected **by name**, per modality. Each name resolves to a
registered generator function under `tensorrt_edgellm/quantization/datasets/`
that yields raw samples for its modality: text strings, `(image, question)`
pairs for visual FP8, or `(audio_bytes, transcript)` pairs for Qwen3-ASR/audio
FP8. The CLI passes only the dataset name; an unknown name fails out with a
pointer to the customization guide. See
[Calibration Dataset Customization](../customization/calibration-datasets.md).

DFlash draft quantization follows the same checkpoint-only contract. The
standalone calibration model replays the draft's dense PyTorch modules using
base-model hidden states, then saves quantized draft weights plus
`hf_quant_config.json`. The DFlash `fc` target-hidden projector is excluded
from quantization because the ONNX exporter keeps that projection on the
full-FP32 accumulation path for accuracy.

## Limitations

- Model-specific export and build workarounds are intentionally not included.
- This package produces checkpoints only. It does not build ONNX or TensorRT engines.
