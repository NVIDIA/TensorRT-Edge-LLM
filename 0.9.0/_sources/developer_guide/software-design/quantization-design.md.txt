# Quantization Package Design

`tensorrt_edgellm.quantization` is a standalone checkpoint quantization package. It converts FP16 HuggingFace checkpoints into quantized HuggingFace-style checkpoints that the checkpoint exporter can export directly.

For command-line usage, see [Quantization](../../user_guide/features/quantization.md).

## Design Goals

- Keep quantization separate from ONNX export.
- Preserve a HuggingFace-compatible checkpoint layout.
- Store quantization metadata in checkpoint config files so `tensorrt_edgellm` can infer export behavior from the checkpoint.
- Share the same output contract for locally quantized and downloaded pre-quantized checkpoints.

## Package Layout

| Path | Role |
|---|---|
| `tensorrt_edgellm/scripts/quantize.py` | User-facing `llm` and `draft` commands |
| `tensorrt_edgellm/quantization/quantize.py` | Model loading, ModelOpt configuration, calibration, and checkpoint writing |
| `tensorrt_edgellm/quantization/quantization_configs.py` | Shared ModelOpt quantization configurations |
| `tensorrt_edgellm/config.py` | Reads quantization metadata during ONNX export |
| `tensorrt_edgellm/checkpoint/loader.py` | Loads and repacks quantized checkpoint tensors |
| `tensorrt_edgellm/quantization/models/eagle3_draft.py` | Standalone Eagle3 draft calibration model |
| `tensorrt_edgellm/quantization/models/dflash_draft.py` | Standalone DFlash draft calibration model |

## Flow

```text
FP16 checkpoint
  -> tensorrt_edgellm.quantization
  -> quantized safetensors + quantization metadata
  -> tensorrt_edgellm
  -> ONNX + runtime sidecars
```

The quantization package stops after writing the checkpoint. ONNX export, TensorRT engine build, and runtime execution remain owned by `tensorrt_edgellm`, `llm_build`, and `llm_inference`.

## Artifact Contract

The output directory must contain:

- `config.json` with model architecture fields.
- Quantization metadata such as `quantization_config`, `hf_quant_config.json`, or equivalent fields understood by `tensorrt_edgellm`.
- One or more `.safetensors` checkpoint shards.
- Tokenizer and processor files needed by the runtime.

`tensorrt_edgellm` uses this metadata to select quantized linear layers, repack checkpoint tensors, and enable FP8 KV cache when the checkpoint marks KV cache quantization as `fp8`. No separate FP8 KV export flag is required in the `tensorrt_edgellm` path.

## Supported Methods

| Component | Methods |
|---|---|
| Backbone | `fp8`, `int4_awq`, `nvfp4`, `mxfp8`, `int8_sq` |
| LM head | `fp8`, `int4_awq`, `nvfp4`, `mxfp8` |
| KV cache | `fp8` |
| Visual tower | `fp8` |

Backbone and LM-head methods can be combined for mixed-precision checkpoints.
Visual FP8 requires multimodal calibration data so visual quantizers observe
image activations.

DFlash draft quantization follows the same checkpoint-only contract. The
standalone calibration model replays the draft's dense PyTorch modules using
base-model hidden states, then saves quantized draft weights plus
`hf_quant_config.json`. The DFlash `fc` target-hidden projector is excluded
from quantization because the ONNX exporter keeps that projection on the
full-FP32 accumulation path for accuracy.

## Limitations

- Audio calibration is not implemented.
- Model-specific export workarounds that belong to ONNX export are intentionally not included.
- This package produces checkpoints only. It does not build ONNX or TensorRT engines.
