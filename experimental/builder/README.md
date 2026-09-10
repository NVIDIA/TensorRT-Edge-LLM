# Experimental ONNX-less TensorRT Builder

`tensorrt-edgellm-build` builds TensorRT engines directly from an Edge-LLM
checkpoint. It does not export, parse, or cache an ONNX graph. This is an
experimental alternative to the supported
`tensorrt-edgellm-export` plus component-builder workflow.

Use the
[ONNX-less engine build user guide](../../docs/source/user_guide/getting_started/direct-engine-builder.md)
for complete build and runtime commands. The
[developer design](../../docs/source/developer_guide/software-design/onnxless-builder.md)
traces model calls to TensorRT APIs and documents how to add an operation or
model family.

The input must be a Hugging Face-style checkpoint supported by the registry in
[`models/registry.py`](models/registry.py). Plain FP16/BF16 and supported
pre-quantized safetensors checkpoints use the same command. Contiguous tensors
in PyTorch ZIP `.bin` checkpoints are also supported.

## Install

Install the repository package and the TensorRT Python wheel that matches the
TensorRT libraries used to compile Edge-LLM:

```bash
python3 -m venv --system-site-packages .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install .
.venv/bin/python -m pip install \
  /path/to/TensorRT/python/tensorrt-<version>-cp312-none-linux_x86_64.whl
```

Build Edge-LLM before invoking the frontend. The default plugin path is
`build/libNvInfer_edgellm_plugin.so`.

The plugin must include the kernels required by the selected model. In
particular, Qwen3.5 hybrid models require Gated DeltaNet. Configure the needed
groups when those families are used:

```bash
cmake -S . -B build \
  -DTRT_PACKAGE_DIR=/path/to/TensorRT \
  -DENABLE_CUTE_DSL=ALL
cmake --build build -j16
```

## Build Every Component

The default component selection is `all`. One command discovers the checkpoint
family and builds each component in runtime order:

```bash
.venv/bin/tensorrt-edgellm-build \
  --model-dir /path/to/checkpoint \
  --engine-dir /path/to/engines \
  --plugin-path /path/to/build/libNvInfer_edgellm_plugin.so \
  --max-input-len 2048 \
  --max-kv-cache-capacity 4096 \
  --max-batch-size 1
```

`--components` is only needed for an intentional partial rebuild:

```bash
.venv/bin/tensorrt-edgellm-build \
  --model-dir /path/to/checkpoint \
  --engine-dir /path/to/engines \
  --components visual,audio \
  --plugin-path /path/to/build/libNvInfer_edgellm_plugin.so
```

The available components are `llm`, `dllm`, `visual`, `audio`, `talker`,
`code-predictor`, `code2wav`, `speaker-encoder`,
`speech-tokenizer-encoder`, `action`, `und-prefill`, `gen`, and
`vae-encoder`. A model can expose any subset:

| Model kind | Components built by `all` |
|---|---|
| Text LLM | LLM |
| VLM | LLM, visual |
| ASR | LLM, audio |
| Omni | LLM, visual, audio, and model-owned speech components |
| TTS | Talker, code-predictor, Code2Wav, and checkpoint-owned clone encoders |
| Alpamayo | LLM, visual, action |
| DiffusionGemma | Diffusion backbone, visual |
| Cosmos3 policy | Understanding prefill, generation policy, VAE encoder |

Each component is a separate TensorRT engine because components have different
shape profiles and runtime I/O contracts. The command still builds all of them
in one invocation and writes the layout consumed by the existing C++ runtime:

| Component | Engine path |
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

Runtime configs, tokenizers, chat templates, embeddings, and other model-owned
artifacts are emitted beside those engines.

## Weights Stay In The Checkpoint

Every supported weight that the runtime can rebuild from the original
checkpoint becomes either a TensorRT engine input or a plugin-owned resource.
`config.json` records the provider keys and conversion recipe for each external
weight. It also records dtype, shape, and byte count for every provider tensor
plus deterministic content samples from a bounded, model-spanning sentinel
set. The runtime validates that identity before registering checkpoint pages,
so a same-shape checkpoint with a different recorded identity is rejected
rather than silently producing unrelated output. `llm_inference` reads those
weights once at load time and performs layout conversion on the GPU:

```bash
llm_inference --engineDir /path/to/engines --checkpointDir /path/to/checkpoint ...
```

The runtime loads the original checkpoint during initialization. All checkpoint
reads, casts, transposes, and quantized layout conversions finish there, then
the preparation stream is synchronized and every checkpoint mapping and
temporary source tensor is released. Only immutable, final-layout `Tensor`
buffers remain bound to TensorRT; inference performs no weight allocation or
conversion.

For full-precision checkpoints, reproducible dense matrices, linear biases,
and ordinary normalization scales and biases follow this path. Checkpoint
source pages are registered in bounded windows and discarded as each final
parameter is written, so the loader does not retain a second model-sized CPU
copy beside the final GPU arena.

When an FP16/BF16 checkpoint ties the output projection to the token embedding,
the runtime materializes that provider tensor once. The LM-head input and
embedding table use separate typed `Tensor` views over the same arena range.
Scaled embeddings and otherwise different conversion recipes remain separate.

For PyTorch ZIP `.bin` checkpoints, the builder loads tensors on the `meta`
device and records their archive byte ranges without materializing payloads.
The C++ runtime maps those ranges directly from the original checkpoint.

**Serving an engine built this way requires `--checkpointDir`.** The
experimental builder always records original-checkpoint recipes; it does not
publish a second, transformed copy of externalized weights.

Some weights always stay inside the engine, because no checkpoint tensor
describes them:

| Weight | Why it stays |
|---|---|
| FP8 and MXFP8 | Their Q/DQ constants are folded into the graph by design |
| A reduced-vocabulary LM head | The head is rewritten at build time |
| Tensor-parallel model weights outside the fused NVFP4 plugin | Their TensorRT/Myelin layouts are selected at build time |
| The FP8 embedding table (`--fp8-embedding`) | Quantized at build time |
| Padded or fused MoE expert banks | No per-expert checkpoint layout to repack |
| A visual patch embedding | Folded into the graph as a small constant |
| Derived Code2Wav codebooks | Computed from checkpoint statistics at build time |

Visual, audio, action, and Code2Wav components participate too. Their FP16
projections and the audio encoders' supported convolution kernels become
engine inputs owned and bound by the corresponding component runner.
Code2Wav keeps convolution kernels in its engine and loads its linear weights
from the checkpoint. One `--checkpointDir` serves all components built from
the same checkpoint.

`--externalize-weights` narrows the default to specific kinds (`int4_ffn`,
`int4_moe`, `nvfp4_moe`, `nvfp4_tp`, `lm_head`, `fp16`, `embedding`, `all`)
and is repeatable. `nvfp4_tp` covers row-parallel packed weights and block
scales consumed by the fused NVFP4 GEMM/all-reduce plugin; the runtime slices
each rank's persistent plugin resource directly from the checkpoint. Requesting
a kind the build cannot support is an error, while the default silently keeps
that kind in the engine and logs it.

## Speculative Decoding

One invocation builds both `spec_base.engine` and `spec_draft.engine`, plus any
non-LLM components declared by the target checkpoint.

EAGLE3 uses a paired draft checkpoint:

```bash
.venv/bin/tensorrt-edgellm-build \
  --model-dir /path/to/target \
  --draft-model-dir /path/to/eagle3-draft \
  --spec-type eagle3 \
  --engine-dir /path/to/engines \
  --plugin-path /path/to/build/libNvInfer_edgellm_plugin.so
```

Qwen3.5 native MTP reads the draft layers from the target checkpoint:

```bash
.venv/bin/tensorrt-edgellm-build \
  --model-dir /path/to/qwen3.5-checkpoint \
  --spec-type mtp \
  --engine-dir /path/to/engines \
  --plugin-path /path/to/build/libNvInfer_edgellm_plugin.so
```

Do not pass `--draft-model-dir` for native Qwen MTP. The draft engine prefers
the checkpoint's `mtp.lm_head` when present and otherwise uses its compatible
tied/base projection. Add `--tree-base` when the runtime will use
`--specDraftTopK > 1`; hybrid Qwen tree and linear base engines are distinct
contracts.

At runtime, native Qwen MTP uses the same checkpoint for both engines:

```bash
llm_inference \
  --engineDir=/path/to/engines \
  --checkpointDir=/path/to/qwen3.5-checkpoint \
  --inputFile=/path/to/input.json \
  --outputFile=/path/to/output.json \
  --specDecode
```

DFlash uses its paired draft checkpoint and model-owned DFlash cache contract:

```bash
.venv/bin/tensorrt-edgellm-build \
  --model-dir /path/to/target \
  --draft-model-dir /path/to/dflash-draft \
  --spec-type dflash \
  --engine-dir /path/to/engines \
  --plugin-path /path/to/build/libNvInfer_edgellm_plugin.so
```

dSpark uses a paired Qwen3 target and draft checkpoint:

```bash
.venv/bin/tensorrt-edgellm-build \
  --model-dir /path/to/qwen3-target \
  --draft-model-dir /path/to/dspark-draft \
  --spec-type dspark \
  --engine-dir /path/to/engines \
  --plugin-path /path/to/build/libNvInfer_edgellm_plugin.so
```

Gemma4 MTP uses the matched assistant checkpoint:

```bash
.venv/bin/tensorrt-edgellm-build \
  --model-dir /path/to/gemma4-target \
  --draft-model-dir /path/to/gemma4-assistant \
  --spec-type gemma4_mtp \
  --engine-dir /path/to/engines \
  --plugin-path /path/to/build/libNvInfer_edgellm_plugin.so
```

The Gemma assistant owns its model layers and output projection. If a provider
assistant omits a compatible projection, only that projection is resolved from
the target checkpoint; the assistant model is never read as an integrated
target-side MTP layer.

The Gemma4 MTP base engine is a verification engine, so
`disable_spec_decode=true` is not a supported per-request fallback for this
bundle. Build the target checkpoint without `--spec-type` when target-only
inference or an accuracy baseline is required.

Pass both provider checkpoints when running a relocated engine bundle:

```bash
llm_inference \
  --engineDir=/path/to/engines \
  --checkpointDir=/path/to/gemma4-target \
  --draftCheckpointDir=/path/to/gemma4-assistant \
  --inputFile=/path/to/input.json \
  --outputFile=/path/to/output.json \
  --specDecode
```

The same `--draftCheckpointDir` contract applies to paired EAGLE3, DFlash, and
dSpark drafts. It is rejected for native Qwen MTP because those draft layers
belong to `--checkpointDir`.

## Loader And Modeling Contract

The builder intentionally owns its model definitions instead of importing the
ONNX exporter:

1. `core/config.py` reads checkpoint metadata and determines the exact model
   variant and component set.
2. `core/weights.py` and model-owned `weights.py` files resolve provider
   tensors and the target shapes needed to define the network. Externalized
   parameters are represented by shape and dtype metadata only and record a
   runtime conversion recipe.
3. `models/<family>/modeling_*.py` constructs a model-specific
   `NetworkModule` for each component.
4. `ops.functional` is the single PyTorch-like operation surface. TensorRT
   native layers and Edge-LLM extension layers are both called as `F.<op>`;
   model code does not select or expose the lowering kind.
5. `core/builder.py` creates one TensorRT `INetwork` and optimization profile
   per component, serializes the engine, and releases checkpoint storage.
6. Model-owned artifact writers emit the runtime contract consumed by the C++
   executables.

`Module` represents a nested layer and never owns network I/O.
`NetworkModule` represents exactly one TensorRT `INetwork` and is the only
module type allowed to declare component inputs and outputs. Model families
share common operations, but they do not reuse another family's model
definition.

Functional operations are grouped by semantic domain under
`ops/functional/`: core tensor operations, attention, distributed, MoE,
recurrent, and speculative. They all accept symbolic tensors and ordinary
Python attributes. TensorRT creator lookup and attribute encoding are private
lowering details in `ops/backend.py`.

For example, an FP16 `Linear` call reaches
`F.linear_from_weights -> Net.linear_from_weights -> Net.linear`, which emits
`add_constant`, `add_matrix_multiply`, and an optional `add_elementwise`.
RMSNorm is decomposed into native cast/reduce/elementwise/unary layers.
Attention and other Edge-LLM operations pass through the same functional API
and are lowered by `Net.operation` to `add_plugin_v3`. Model code never handles
these TensorRT details.

INT4 GEMM qweights/scales, INT4 MoE qweights/scales, NVFP4 MoE packed
weights/scales/alphas, and dense FP16 projections are static TensorRT network
inputs when their model family provides a checkpoint recipe. They remain
module parameters: model `forward()` signatures do not accept weight tensors.
Small generated constants, such as unit input scales, remain in the engine.

`Net.weight_input` records a checkpoint binding and writes no weight payload.
The TensorRT network sees only the final input shape, dtype, and operation, so
the builder does not load the externalized payload. A weight without a
reproducible transform recipe remains a constant rather than becoming an input
the runtime cannot fill.

## Supported Families

The explicit registry currently includes Llama, Mistral, Qwen2, Qwen3,
Qwen3-MoE, Qwen2/2.5/3-VL, Qwen3.5 dense and MoE, Qwen3-ASR, Qwen3-Omni,
Qwen3-Omni-Next, Qwen3-TTS, InternVL3/3.5, Phi-4 Multimodal,
Nemotron-H/Omni, Gemma4 and Gemma4 Unified, DiffusionGemma, Cosmos3, and
Alpamayo. EAGLE3, MTP, DFlash, dSpark, and Gemma4 assistant drafts use
model-owned speculative definitions. Unsupported `model_type` values fail
before a TensorRT network is created and report the registered choices.

The user guide contains an explicit
[implementation, CI, and known-gap matrix](../../docs/source/user_guide/getting_started/direct-engine-builder.md#support-and-validation-status).
This includes features such as FP8 embedding, reduced vocabulary, LoRA,
quantized formats, speculative modes, and current differences from the
supported ToT workflow.

## CI

The standard Edge-LLM L0 jobs remain enabled. Direct-builder coverage adds:

- A30 FP16 dense and EAGLE3 target/draft end-to-end cases
- RTX 5090 FP8 and NVFP4 dense end-to-end cases

Each case builds directly from a checkpoint, executes the matching C++ runtime,
checks the runtime artifact layout, and applies the existing accuracy check.
The RTX 5090 cases are part of the existing 50-series job, so they do not add a
new runner allocation.

See the
[ONNX-less builder user guide](../../docs/source/user_guide/getting_started/direct-engine-builder.md#build-time-comparison)
for the reproducible original-versus-direct build timing comparison.
