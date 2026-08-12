# Experimental ONNX-less Engine Build

`tensorrt-edgellm-build` compiles every TensorRT engine required by a
checkpoint without writing or parsing ONNX. The command reads checkpoint
configuration and tensor metadata directly, selects model-owned Python
component definitions, and writes the same engine and runtime-artifact layout
consumed by the existing C++ executables. By default, the runtime reads the
original checkpoint and prepares final-layout weights once during
initialization.

This frontend is experimental. The supported workflow remains
`tensorrt-edgellm-export` followed by the component-specific C++ builders.

For implementation details and instructions for adding a model or operation,
see the [ONNX-less Builder Design](../../developer_guide/software-design/onnxless-builder.md).

## Prerequisites

Install the repository package in a virtual environment. The TensorRT Python
wheel must match the TensorRT headers and libraries used to build Edge-LLM:

```bash
cd /path/to/TensorRT-Edge-LLM
python3 -m venv --system-site-packages .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install \
  /path/to/TensorRT/python/tensorrt-<version>-cp312-none-linux_x86_64.whl
.venv/bin/python -m pip install .
```

Build Edge-LLM and its plugin library before compiling an engine:

```bash
cmake -S . -B build \
  -DTRT_PACKAGE_DIR=/path/to/TensorRT \
  -DENABLE_CUTE_DSL=ALL
cmake --build build -j16
```

`-DENABLE_CUTE_DSL=ALL` is required by this frontend because its compiled
extension includes `cutedsl_all.h`.

## Build All Components

The default `--components all` selection discovers the checkpoint family and
builds all of its components in runtime order:

```bash
.venv/bin/tensorrt-edgellm-build \
  --model-dir /path/to/checkpoint \
  --engine-dir /path/to/engines \
  --plugin-path /path/to/build/libNvInfer_edgellm_plugin.so \
  --max-input-len 2048 \
  --max-kv-cache-capacity 4096 \
  --max-batch-size 1
```

The input is a Hugging Face-style FP16/BF16 checkpoint or a supported
pre-quantized checkpoint. Precision is read from `hf_quant_config.json` or the
embedded `quantization_config`; FP8 and NVFP4 builds use the same command and
do not require a precision switch. An unrecognized quantization method,
algorithm, or declared NVFP4 layer without its required weight and scale
tensors is an error; it is never compiled silently as FP16. FP16 checkpoint
bindings reported by the build log can still be expected for embeddings and
explicitly excluded modules and do not describe the precision of the graph's
quantized projections.

Use `--components` only for an intentional partial rebuild:

```bash
.venv/bin/tensorrt-edgellm-build \
  --model-dir /path/to/checkpoint \
  --engine-dir /path/to/engines \
  --components visual,audio \
  --plugin-path /path/to/build/libNvInfer_edgellm_plugin.so
```

The component set and output layout are model-specific:

| Model kind | Components selected by `all` |
|---|---|
| Text LLM | LLM |
| VLM | LLM, visual |
| ASR | LLM, audio |
| Omni | LLM, visual, audio, and model-owned speech components |
| TTS | Talker, code predictor, Code2Wav, and checkpoint-owned clone encoders |
| Alpamayo | LLM, visual, action |
| DiffusionGemma | Diffusion backbone, visual |
| Cosmos3 policy | Understanding prefill, generation policy, VAE encoder |

| Component | Engine path under `--engine-dir` |
|---|---|
| LLM | `llm.engine` |
| Diffusion backbone | `dllm.engine` |
| Visual | `visual/visual.engine` |
| Audio | `audio/audio_encoder.engine` |
| Talker | `talker/llm.engine` |
| Code predictor | `code_predictor/llm.engine` |
| Code2Wav | `code2wav/code2wav.engine` |
| Speaker encoder | `clone_encoders/speaker_encoder.engine` |
| Speech tokenizer encoder | `clone_encoders/speech_tokenizer_encoder.engine` |
| Action | `action/action.engine` |
| Cosmos3 understanding prefill | `und_prefill/und_prefill.engine` |
| Cosmos3 generation policy | `gen/gen.engine` |
| Cosmos3 VAE encoder | `vae_encoder/vae_encoder.engine` |

Runtime configs, tokenizers, chat templates, preprocessing files, and
checkpoint binding recipes are written beside the component engines.
Those recipes include structural identities for every provider tensor and
bounded content samples spanning the component and target checkpoints used
during the build. A runtime checkpoint with matching tensor shapes but a
different recorded identity is rejected before CUDA page registration. Each
engine build reports checkpoint-identity, TensorRT, and remaining frontend
time separately.

## Run The Engines

Use the same C++ runtime and request JSON used by the ONNX workflow. A
text-only engine runs directly from the output directory:

```bash
build/examples/llm/llm_inference \
  --engineDir=/path/to/engines \
  --checkpointDir=/path/to/checkpoint \
  --inputFile=/path/to/input.json \
  --outputFile=/path/to/output.json
```

For VLM, ASR, and omni checkpoints, pass the same output directory as the
multimodal directory. The runtime discovers the model-specific `visual/` and
`audio/` components:

```bash
build/examples/llm/llm_inference \
  --engineDir=/path/to/engines \
  --multimodalEngineDir=/path/to/engines \
  --checkpointDir=/path/to/checkpoint \
  --inputFile=/path/to/input.json \
  --outputFile=/path/to/output.json
```

TTS uses the model-specific runtime:

```bash
build/examples/omni/qwen3_tts_inference \
  --talkerEngineDir=/path/to/engines/talker \
  --code2wavEngineDir=/path/to/engines/code2wav \
  --tokenizerDir=/path/to/engines/talker \
  --cloneEncoderDir=/path/to/engines/clone_encoders \
  --checkpointDir=/path/to/checkpoint \
  --inputFile=/path/to/input.json \
  --outputAudioDir=/path/to/audio-output
```

Alpamayo uses the action runtime:

```bash
build/examples/multimodal/action_inference \
  --engineDir=/path/to/engines \
  --multimodalEngineDir=/path/to/engines \
  --checkpointDir=/path/to/checkpoint \
  --inputFile=/path/to/input.json \
  --outputFile=/path/to/output.json
```

Cosmos3 policy checkpoints use their image-and-instruction runtime:

```bash
build/experimental_models/cosmos3/examples/cosmos3_policy_inference \
  --engineDir=/path/to/engines \
  --image=/path/to/observation.png \
  --prompt="Move the object into the tray." \
  --output=/path/to/action.json
```

See the [examples](../examples/index.md) for each runtime's request format.

`--checkpointDir` is required. During runtime initialization, Edge-LLM maps
provider safetensors or indexed contiguous tensor ranges from PyTorch ZIP
`.bin` checkpoints, performs all requested casts, transposes, packing, scale
conversion, and direct final-layout writes, synchronizes the preparation
stream, and releases the checkpoint mappings. The inference path sees only
immutable final-layout TensorRT inputs and performs no weight conversion or
allocation.

For tied FP16/BF16 token embeddings and output projections with identical
runtime conversion contracts, both runtime tensors share one weight-arena
allocation. Embedding scaling or any other recipe difference disables this
alias automatically.

The same checkpoint-backed engines can be profiled with `llm_bench`:

```bash
build/examples/llm/llm_bench \
  --engineDir=/path/to/engines \
  --checkpointDir=/path/to/checkpoint \
  --mode=prefill \
  --inputLen=128
```

## Speculative Decoding

One invocation builds both `spec_base.engine` and `spec_draft.engine`, plus the
target checkpoint's non-LLM components.

EAGLE3, DFlash, DSpark, and Gemma4 MTP use paired checkpoints:

```bash
.venv/bin/tensorrt-edgellm-build \
  --model-dir /path/to/target \
  --draft-model-dir /path/to/draft \
  --spec-type eagle3 \
  --engine-dir /path/to/engines \
  --plugin-path /path/to/build/libNvInfer_edgellm_plugin.so
```

Replace `eagle3` with `dflash`, `dspark`, or `gemma4_mtp` as appropriate.
Qwen3.5 native MTP reads draft layers from the target checkpoint:

```bash
.venv/bin/tensorrt-edgellm-build \
  --model-dir /path/to/qwen3.5-checkpoint \
  --spec-type mtp \
  --engine-dir /path/to/engines \
  --plugin-path /path/to/build/libNvInfer_edgellm_plugin.so
```

The integrated Qwen draft uses the target embedding and falls back to the
target's tied embedding or LM head when the `mtp.*` namespace has no output
projection. A Gemma4 assistant remains a separate checkpoint; it uses its own
projection when present and otherwise binds the compatible target projection.
Its speculative base is a verification engine and does not support
`disable_spec_decode=true`; build the target without `--spec-type` for a
target-only baseline.

Native Qwen MTP uses one runtime checkpoint:

```bash
build/examples/llm/llm_inference \
  --engineDir=/path/to/engines \
  --checkpointDir=/path/to/qwen3.5-checkpoint \
  --inputFile=/path/to/input.json \
  --outputFile=/path/to/output.json \
  --specDecode
```

Paired drafts use an explicit second checkpoint path:

```bash
build/examples/llm/llm_inference \
  --engineDir=/path/to/engines \
  --checkpointDir=/path/to/target \
  --draftCheckpointDir=/path/to/draft \
  --inputFile=/path/to/input.json \
  --outputFile=/path/to/output.json \
  --specDecode
```

`--draftCheckpointDir` applies to EAGLE3, DFlash, DSpark, and Gemma4 MTP. It is
rejected for native Qwen MTP because its draft layers are in
`--checkpointDir`.

## Optional Features

| Feature | Build option or behavior |
|---|---|
| Runtime LoRA inputs | `--max-lora-rank N` |
| Reduced vocabulary | `--reduced-vocab-dir DIR` |
| DFlash draft vocabulary | `--draft-reduced-vocab-dir DIR` |
| FP8 embedding sidecar | `--fp8-embedding` |
| Tensor parallel rank | `--tp-size N --tp-rank R` |
| Detailed TensorRT profiling names | `--profiling-detailed` |
| Partial component rebuild | `--components NAME[,NAME...]` |

Tensor parallelism currently builds one rank per invocation. Invoke the command
once for each `--tp-rank` and place the rank artifacts according to the normal
multi-GPU runtime layout.

## Support And Validation Status

The explicit registry includes Llama, Mistral, Qwen2, Qwen3, Qwen3-MoE,
Qwen2/2.5/3-VL, Qwen3.5 dense and MoE, Qwen3-ASR, Qwen3-Omni,
Qwen3-Omni-Next, Qwen3-TTS, InternVL3/3.5, Phi-4 Multimodal,
Nemotron-H/Omni, Gemma4 and Gemma4 Unified, DiffusionGemma, Cosmos3, and
Alpamayo. EAGLE3, MTP, DFlash, DSpark, and Gemma4 assistant drafts use
model-owned speculative definitions. Unsupported `model_type` values fail
before TensorRT network creation and list the registered choices.

The following table distinguishes implementation from automatic direct-builder
CI coverage. An implemented row can still have model-specific restrictions.

| Capability | Frontend status | Direct-builder L0 coverage |
|---|---|---|
| FP16/BF16 dense LLM | Implemented | Qwen2.5 on A30 |
| FP8 and block-wise FP8 dense LLM | Implemented | Qwen2.5 on RTX 5090 |
| NVFP4 dense LLM | Implemented | Qwen3 on RTX 5090 |
| MXFP8 and INT8 SmoothQuant | Implemented | Not yet |
| INT4 AWQ, ModelOpt AWQ, and GPTQ | Implemented | Not yet |
| FP8 KV cache | Implemented from checkpoint metadata | Not yet |
| FP8 embedding and reduced vocabulary | Implemented | Not yet |
| Runtime LoRA inputs | Implemented | Not yet |
| EAGLE3, Qwen3.5 MTP, DFlash, DSpark, and Gemma4 MTP | Implemented | EAGLE3 with Qwen3 on A30 |
| DiffusionGemma block diffusion | Implemented | Not yet |
| Visual, audio, TTS, omni, action, and Cosmos3 policy components | Implemented for registered families | Not yet |
| Tensor parallel graph generation | Implemented per rank | Not yet |

Known gaps relative to the supported ToT workflow are tracked explicitly:

- Quantization and calibration are checkpoint-producing steps outside this
  builder; run `tensorrt-edgellm-quantize` first.
- The CLI does not yet orchestrate every tensor-parallel rank in one process.
- Direct-builder CI does not yet cover every registered
  model-family/precision/feature combination.

These limitations apply to the experimental direct builder, not the supported
ONNX workflow.

## Build-Time Comparison

The following comparison used TensorRT 11.0.0.114, CUDA 13.2, an NVIDIA RTX
PRO 4000 Blackwell GPU, and `Qwen2.5-0.5B-Instruct`. Both paths used batch size
1, maximum input length 1024, and KV-cache capacity 2048.

| Workflow step | Wall time |
|---|---:|
| Original ONNX export | 26.60 s |
| Original ONNX parse and engine build | 30.96 s |
| Original total | 57.56 s |
| ONNX-less engine build | 43.16 s |

For this checkpoint and profile, the direct path reduced end-to-end build time
by 14.40 seconds (25.0%, or 1.33x). Both engine directories completed the same
C++ `llm_inference` smoke workload. Separately built TensorRT engines can
select different tactics, so this timing comparison does not assert
token-by-token identity between generated continuations.
