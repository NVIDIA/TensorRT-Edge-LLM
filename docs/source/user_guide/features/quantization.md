# Quantization

Use `trtedge quantize` when you start from an FP16/BF16 checkpoint and need to create a unified quantized checkpoint for `tensorrt_edgellm`.

The quantization CLI writes a unified HuggingFace-style checkpoint directory that `tensorrt_edgellm` can export directly.

Skip this step when you already have a supported pre-quantized HuggingFace checkpoint.

## Select W4A4 or W4A16 for NVFP4 checkpoints

ModelOpt W4A4 checkpoints can run the same packed NVFP4 weights with NVFP4 or
FP16 activations. Export without an activation override for W4A4:

```bash
trtedge export /path/to/modelopt_nvfp4 /tmp/model_w4a4 \
  --target-sm 110

trtedge export /path/to/modelopt_nvfp4 /tmp/model_w4a16 \
  --no-quantize-activations --target-sm 110
```

The default uses the checkpoint's activation scales and the W4A4 GEMM/MoE
path. `--no-quantize-activations` keeps activations in FP16 and selects the
W4A16 GEMM/MoE path. Both modes reuse the checkpoint's packed weights.

The checkpoint-direct builder uses the same selection:

```bash
trtedge build --model-dir /path/to/modelopt_nvfp4 \
  --engine-dir /tmp/engine_w4a16 --no-quantize-activations
```

## Select the TensorRT-native MoE layer for Qwen3-family MoE

Qwen3-family MoE experts run through the Edge-LLM MoE plugins by default.
`--trt-native-moe` selects TensorRT's native MoE layer instead, on both
frontends:

```bash
trtedge export /path/to/qwen3_moe /tmp/model_native_moe \
  --trt-native-moe --target-sm 100

trtedge build --model-dir /path/to/qwen3_moe \
  --engine-dir /tmp/engine_native_moe --trt-native-moe
```

The precision follows the checkpoint: FP16/BF16 experts use the FP16 layer,
and ModelOpt NVFP4 experts use W4A8 (the checkpoint's FP4 weights with static
FP8 activations from the calibrated scales). The switch is opt-in and covers
only these two forms: INT4, NVFP4-A16 (`--no-quantize-activations`) and
Nemotron-H experts are rejected rather than silently built with the plugins.
It needs TensorRT 11.3 or newer and an SM 10.x or 11.x device; performance
relative to the plugins is tracked separately.

`--trt-native-moe-w4a4` (export only, on top of `--trt-native-moe`) switches
ModelOpt NVFP4 experts from W4A8 to W4A4, TensorRT's MoE quantization mode 2:
the hidden states and the down-projection input are quantized dynamically to
NVFP4 (blocks of 16 with FP8 block scales; ModelOpt's calibrated `input_scale`
is the second-level scale, used as is) and the FP4 weights keep the
checkpoint's E4M3 block scales with a per-expert FP16 global. It needs
TensorRT 11.6 or newer (`IMoELayer::setQuantizationDynamicDblQ`) and is
rejected for unquantized experts and for `--no-quantize-activations`. It is
the OOTB counterpart of the plugins' W4A4 recipe for like-for-like
comparisons.

## Setup

Complete [Installation](../getting_started/installation.md) and add the
`quantize` and `onnx` extras described in
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

## Keep The Last Layer In FP16

A 4-bit GEMM writes its output in FP16. When the last decoder layer produces
activations beyond the FP16 range (65504), those outputs saturate and accuracy
collapses. `--keep-last-layer-fp16` leaves the last layer's `o_proj` and MLP
projections (`gate_proj`, `up_proj`, `down_proj`) unquantized. It applies to
`llm` and to DFlash-family `draft` quantization; EAGLE drafts reject it.

```bash
trtedge quantize draft \
  --base-model-dir /path/to/gemma4_target \
  --draft-model-dir /path/to/qwen3_dflash_draft \
  --output-dir /tmp/dflash_draft_nvfp4 \
  --quantization nvfp4 \
  --keep-last-layer-fp16
```

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
| Text  | `--text-dataset` (`cnn_dailymail`) | `cnn_dailymail`, `wikitext` | LLM weights, LM head, KV cache, CodePredictor, MTP, EAGLE3, EAGLE-4, DFlash |
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

## Quantize An EAGLE-4 Draft

```bash
trtedge quantize draft \
  --base-model-dir /path/to/bf16_base_model \
  --draft-model-dir /path/to/eagle4_draft \
  --output-dir /tmp/eagle4_draft_nvfp4 \
  --quantization nvfp4
```

The `draft` command treats a draft whose configuration sets `fc_norm`,
`norm_output`, or a low-rank candidate head as an EAGLE-4 drafter. It
quantizes the decoder-layer linears and keeps `fc` in FP16, as for DFlash
drafts, and records it in `exclude_modules`. The KV cache and the candidate
head factors also stay FP16, so `--kv-cache-quantization` is rejected; with a
candidate head, `lm_head` stays FP16 and `--lm-head-quantization` is rejected
too. For NVFP4-A16, quantize with `--quantization nvfp4` and export with
`--no-quantize-activations`; `--quantization nvfp4_a16_gptq` is rejected for
these drafts.
Calibration runs the base model only up to the deepest layer that the drafter
reads. For EAGLE drafts the command defaults to `--quantization fp8`
and for DFlash drafts to `nvfp4`, so name the precision explicitly.

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

## Quantize The Token Embedding Table

Quantize the canonical token table independently of the backbone:

```bash
trtedge quantize llm \
  --model_dir /path/to/model \
  --output_dir /tmp/model_nvfp4_embedding \
  --quantization nvfp4 \
  --token_embedding_quantization nvfp4
```

The supported methods are `fp8` and `nvfp4`. FP8 uses one FP32 scale for the
entire table, matching PLE and export-time FP8. NVFP4 uses a separate scale for
each token and 16-hidden block. Both layouts are consumed directly by the embedding lookup
kernel. Gemma4's scaled token embedding is supported; its output multiplier is
preserved in the runtime artifact. If token embeddings and the LM head share
weights, quantization separates their storage so each can use its own method.
The LM head can remain in FP16.

For export-time token quantization, use `trtedge export` with
`--fp8-embedding`. This path uses the same per-tensor FP8 semantics and
preserves the source checkpoint. A checkpoint with quantized token embeddings
already supplies its runtime format; omit `--fp8-embedding` for that checkpoint.

Token and PLE FP8 lookup use the same dequantization semantics:
`FP16(FP32(value) * scale)`. Model embedding multipliers are folded into the
FP32 scale. `llm_inference --offloadEmbedding` keeps the token artifact in
CUDA-pinned host memory; `--pleResidency cached --pleCacheSlots 4096` keeps
PLE file-backed with a bounded GPU row cache. PLE uses the existing gather
plugin followed by standard dequantization. Offloading does not change these
operations or the engine contract.
The scalar scale is independent of row placement, so moving compressed rows
between storage and the cache does not change dequantization.

Gemma4 E2B/E4B PLE and Qwen4-Exp n-gram tables support per-tensor FP8 through
either path:

- **Quantize time:** Add `--ple_embedding_quantization fp8` to
  `trtedge quantize llm`, then export the checkpoint normally.
- **Export time:** Add `--fp8-ple` to `trtedge export`. The exporter
  writes a quantized table under the output directory's `ple/` subdirectory
  and records that directory in `checkpoint_dir`. Keep this directory
  accessible during inference, including when using a cached PLE pool.

For example, quantize both tables while exporting a dense Gemma4 checkpoint:

```bash
trtedge export /path/to/gemma4 /tmp/gemma4_onnx \
  --skip-visual --skip-audio --fp8-embedding --fp8-ple
```

If you move the PLE artifact to another device, pass
`--pleFile /path/to/ple/model.safetensors` to `llm_inference`.

Both PLE paths preserve Gemma4's embedding multiplier and use an FP8 pool
input with dequantization inside the engine. Token and PLE options can be
selected independently.

## Supported Methods

| Component | Methods |
|---|---|
| Backbone | `fp8`, `int4_awq`, `nvfp4`, `mxfp8`, `int8_sq` |
| LM head | `fp8`, `int4_awq`, `nvfp4`, `mxfp8` |
| Token embedding | `fp8`, `nvfp4` |
| Gemma4 PLE embedding | `fp8` |
| KV cache | `fp8` |
| Visual tower | `fp8` |
| Audio tower | `fp8` |
| CodePredictor | `fp8` |
| DFlash draft | validated with `nvfp4` backbone and optional `nvfp4` LM-head quantization |

## Notes

- The package writes unified checkpoints only. It does not export ONNX or build TensorRT engines.
- GPTQ checkpoints are loaded as pre-quantized checkpoints; this package does not create GPTQ models.
