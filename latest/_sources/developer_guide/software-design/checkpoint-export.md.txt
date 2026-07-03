# Checkpoint Exporter Design

`tensorrt_edgellm` is the ONNX export frontend for TensorRT Edge-LLM. It constructs TensorRT Edge-LLM model implementations directly, loads HuggingFace checkpoint tensors from safetensors files, and emits ONNX plus runtime sidecar artifacts consumed by the existing engine builder and runtime.

This page describes the software design. For commands and workflows, see the [Quick Start Guide](../../user_guide/getting_started/quick-start-guide.md), [Quantization](../../user_guide/features/quantization.md), [LoRA](../../user_guide/features/lora.md), and [Experimental High-Level Python API and Server](../../user_guide/examples/experimental-server.md).

## Design Goals

- Use checkpoint metadata and safetensors files as the source of truth instead of tracing HuggingFace model execution.
- Keep model architecture code inside TensorRT Edge-LLM so export behavior is stable across `transformers` changes.
- Share the same export path for FP16/BF16 and pre-quantized checkpoints.
- Preserve compatibility with the existing `llm_build`, `llm_inference`, and multimodal runtime contracts.
- Make model support explicit through model implementations, component registries, and the [Supported Models](../../user_guide/getting_started/supported-models.md) matrix.

## Architecture

| Layer | Main files | Responsibility |
|---|---|---|
| Checkpoint config parsing | `tensorrt_edgellm/config.py`, `tensorrt_edgellm/checkpoint/checkpoint_utils.py` | Read root and promoted LLM configs, normalize architecture fields, parse quantization metadata, and infer feature flags. |
| Model dispatch | `tensorrt_edgellm/model.py`, `tensorrt_edgellm/__init__.py` | Dispatch `model_type` to a registered model class, with a default decoder implementation for compatible dense models. |
| Model implementations | `tensorrt_edgellm/models/` | Define TensorRT Edge-LLM-native text, visual, audio, TTS, MoE, hybrid, EAGLE, MTP, and DFlash modules. |
| Checkpoint loading | `tensorrt_edgellm/checkpoint/loader.py`, `tensorrt_edgellm/checkpoint/repacking.py` | Load safetensors, remap keys when needed, and repack quantized tensors into runtime/export formats. |
| Vocabulary reduction | `tensorrt_edgellm/vocab_reduction/selection.py`, `tensorrt_edgellm/vocab_reduction/onnx_export.py` | Generate `vocab_map.safetensors` separately from applying an existing map to supported LM-head tensors during ONNX export. |
| ONNX graph export | `tensorrt_edgellm/onnx/export.py`, `tensorrt_edgellm/onnx/export_encoder.py`, `tensorrt_edgellm/onnx/dynamo_translations.py` | Export text and encoder graphs, register custom ONNX schemas, and translate PyTorch custom ops to TensorRT Edge-LLM ONNX nodes. |
| Component orchestration | `tensorrt_edgellm/scripts/export.py` | Classify model components and coordinate text, visual, audio, TTS, code2wav, EAGLE, MTP, DFlash, and sidecar exports. |
| Runtime artifacts | `tensorrt_edgellm/checkpoint/checkpoint_utils.py`, `tensorrt_edgellm/chat_template.py` | Write tokenizer/config/chat-template/embedding sidecars expected by the C++ runtime and Python server. |

## Data Flow

The exporter follows a checkpoint-driven flow:

1. Read checkpoint configuration and safetensors index.
2. Promote nested LLM configs for multimodal checkpoints when the root config describes a wrapper model.
3. Build `ModelConfig`, including quantization, KV cache, layer-type, RoPE, and model-family fields.
4. Dispatch to the correct TensorRT Edge-LLM-native model implementation.
5. Load and repack weights from checkpoint tensors.
6. Optionally apply vocabulary reduction from `vocab_map.safetensors`.
7. Export ONNX graphs with TensorRT Edge-LLM custom-op schemas and dynamo translations.
8. Write runtime sidecars that the engine builder and runtime consume.

The design boundary is deliberate: `tensorrt_edgellm` owns checkpoint parsing, model construction, weight loading, and ONNX/runtime artifact export. TensorRT engine build and inference remain owned by the existing C++ tools and the Python HLAPI/server wrapper.

## Checkpoint Metadata Contract

`tensorrt_edgellm` relies on checkpoint metadata rather than command-line precision switches wherever possible.

- `config.json` provides architecture fields such as `model_type`, hidden size, attention heads, layer count, RoPE parameters, multimodal sub-configs, and hybrid layer types.
- `hf_quant_config.json` or embedded `quantization_config` provides quantization format, group size, excluded modules, per-layer overrides, and KV cache quantization.
- Safetensors index files provide tensor names and shapes used for loading, q/k normalization detection, and model-specific fallbacks.
- EAGLE3 draft checkpoints are detected from `draft_vocab_size`.
- DFlash draft checkpoints are detected from `dflash_config`.

FP8 KV cache is enabled automatically when checkpoint metadata marks KV cache quantization as `fp8`; the exporter does not require a separate FP8 KV export mode.

## Quantization Integration

The loader consumes both plain FP16/BF16 checkpoints and unified quantized checkpoints. The quantization package is a separate stage that writes checkpoint artifacts only; after that, `tensorrt_edgellm` reads the same checkpoint directory interface as it does for pre-quantized HuggingFace checkpoints.

Supported checkpoint formats are parsed through `QuantConfig` and implemented in shared linear/checkpoint utilities:

| Format | Exporter behavior |
|---|---|
| FP16/BF16 | Load tensors directly and export FP16/BF16 graph weights. |
| FP8 / MXFP8 | Load quantized weights and scales from checkpoint metadata. |
| NVFP4 / FP4 | Repack low-bit floating-point checkpoint tensors for TensorRT Edge-LLM custom ops. |
| INT4 AWQ / ModelOpt AWQ | Interpret AWQ layout variants and emit the expected packed tensor layout. |
| INT4 GPTQ | Load supported GPTQ INT4 checkpoints directly. |
| INT8 SmoothQuant | Preserve activation/weight scale metadata needed by quantized linear layers. |
| Mixed precision | Apply per-layer overrides from checkpoint metadata. |

For the quantization package design, see [Quantization Package Design](quantization-design.md). For usage from FP16/BF16 source checkpoints, see [Quantization](../../user_guide/features/quantization.md).

## Model and Component Dispatch

Text-only models usually enter through `AutoModel.from_pretrained`. The factory reads `ModelConfig.model_type` and selects either a registered implementation or the default `CausalLM` implementation for compatible decoder architectures.

Multimodal and audio checkpoints have additional component dispatch:

- Visual encoders are selected by `_VISUAL_REGISTRY` and family build functions in `tensorrt_edgellm/onnx/export_encoder.py`.
- Audio encoders are selected by audio model-type registries and component-specific key prefixes.
- TTS exports reuse the LLM export path for talker and code-predictor decoder components, with model-specific key remapping. Qwen3-TTS Code2Wav is exported from the checkpoint's `speech_tokenizer/` directory.
- Omni checkpoints may combine text, visual, audio, and vocoder/code2wav exports under one checkpoint-level orchestration.

This keeps component-specific model code separate while preserving one checkpoint-level export frontend.

## ONNX and Runtime Artifact Contract

The exported ONNX graphs use TensorRT Edge-LLM custom-op domains for operations that must lower to C++ runtime plugins or specialized kernels. The graph exporter registers ONNX schemas before export and provides custom dynamo translations so exported graphs match the engine builder's expected node signatures.

Runtime sidecars are written next to exported graphs and include normalized config files, tokenizer/chat-template assets, embedding tables, and optional sidecars such as FP8 embedding data. These artifacts are part of the contract with `llm_build`, `llm_inference`, and `experimental/server`.

## LoRA Integration

LoRA support is implemented as a graph and weight sidecar transformation after base ONNX export:

- Graph insertion augments supported linear projections with LoRA inputs and runtime hooks.
- Adapter processing converts LoRA checkpoint tensors into the layout expected by the runtime.
- Static merge uses `tensorrt-edgellm-merge-lora` before quantization for models such as Phi-4-Multimodal that ship required LoRA adapters.
- The base model export remains checkpoint-driven and does not depend on HuggingFace FX tracing.

Runtime usage is documented in [LoRA](../../user_guide/features/lora.md).

## Limitations

- FP8 embedding sidecars are not supported for TTS talker/code_predictor exports.
- TensorRT native-ops export mode is not supported at this time.
