<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Experimental ONNX-less Engine Build

`trtedge build` compiles every TensorRT engine required by a
checkpoint without writing or parsing ONNX. The command reads checkpoint
configuration and tensor metadata directly, selects model-owned Python
component definitions, and writes the same engine and runtime-artifact layout
consumed by the existing C++ executables. Engines with external plugin weights
read the original checkpoint and prepare final-layout weights once during
runtime initialization.

This frontend is experimental. The supported workflow remains
`trtedge export` followed by the component-specific C++ builders.

For implementation details and instructions for adding a model or operation,
see the [ONNX-less Builder Design](../../developer_guide/software-design/onnxless-builder.md).

## Prerequisites

For a supported target, install `pip3 install 'tensorrt-edgellm[builder]'`
after the matching TensorRT Python wheel. The package selects its bundled
plugin automatically. Run commands in the activated wheel environment outside
a source checkout.

### From source

Install the repository package in a virtual environment. The TensorRT Python
wheel must match the TensorRT headers and libraries used to build Edge-LLM:

```bash
cd /path/to/TensorRT-Edge-LLM
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install \
  /path/to/TensorRT/python/tensorrt-<version>-cp312-none-linux_x86_64.whl
python -m pip install ".[builder]"
```

The standard Safetensors path uses NumPy and TensorRT directly; the `[builder]`
extra does not install PyTorch, ONNX, ModelOpt, or Optimum. Checkpoints should
include `tokenizer.json`; Transformers is available when that artifact needs
conversion.

Build Edge-LLM and its plugin library before compiling an engine:

```bash
cmake -S . -B build \
  -DTRT_PACKAGE_DIR=/path/to/TensorRT \
  -DENABLE_CUTE_DSL=ALL
cmake --build build -j16
```

Select the CuTeDSL groups and architecture artifacts required by the model.
`ALL` enables all available groups. Disabling CuTeDSL can leave a successfully
built engine without a usable prefill kernel.

## Experimental Weightless Refit Mode

The default flow supplies checkpoint weights as external engine inputs.
`--weightless-engine` instead builds named null-weight constants and refits
them from the checkpoint during runtime initialization, before creating the
execution context. Refit runs once per engine load, not per inference request;
the original checkpoint is still required.

Speculative base and draft engines are refit independently from their existing
checkpoint sources. Visual, audio, Code2Wav, and action engines also refit
before context creation. External plugin weights retain their existing loading
path; preprocessed constants already baked by the direct builder remain baked.

FP16 weights and supported native FP8, NVFP4, and INT4 layouts can use this
flow. Matrix payloads are deferred to runtime; global quantization scales
remain build-time constants. Quantized checkpoints must use a layout supported
by the target GPU. Weights that require build-time conversion or rewriting
remain embedded in the plan.

For tied FP16 embedding/LM-head weights with identical checkpoint conversion
recipes, the builder keeps the shared weight as an external input and refits
the other supported weights. This avoids duplicating the embedding inside
TensorRT and reduces peak build and runtime memory; the shared weight is still
absent from the serialized plan. Untied or incompatible weight layouts retain
their normal refit behavior.

Refit initialization temporarily holds source weights alongside the engine's
final weights. The source storage is released after refit completes, before
execution-context creation.

### TensorRT requirements

Use a TensorRT package with verified null-weight placeholder and refit support
on the target GPU. Use matching TensorRT Python bindings, headers, libraries,
and target-GPU builder resources.

NVIDIA's public [weight-placeholder documentation](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/refitting-engines.html#reducing-build-time-memory-usage-with-weight-placeholders)
describes this mechanism with `STRIP_PLAN` and `REFIT_INDIVIDUAL`, but does not
identify its first supported release. A public TensorRT package has not yet
been validated for this Edge-LLM flow.


### Build and run

Follow the [source installation](#from-source) above with the selected TRT
package, then enable the native refit path (default: `OFF`). Do not use a
prebuilt Edge-LLM wheel for this workflow.

```bash
export TRT_PACKAGE_DIR=/path/to/compatible-TensorRT
export LD_LIBRARY_PATH="$TRT_PACKAGE_DIR/lib:$LD_LIBRARY_PATH"
cmake -S . -B build-refit \
  -DTRT_PACKAGE_DIR="$TRT_PACKAGE_DIR" \
  -DENABLE_EXPERIMENTAL_WEIGHTLESS=ON \
  -DENABLE_CUTE_DSL=ALL \
  -DCUTE_DSL_ARTIFACT_TAG="<qualified-target-artifact-tag>"
cmake --build build-refit -j16

trtedge build \
  --model-dir /path/to/checkpoint \
  --engine-dir /path/to/refit-engines \
  --plugin-path build-refit/lib/libNvInfer_edgellm_plugin.so \
  --weightless-engine \
  --max-input-len 2048 --max-kv-cache-capacity 4096 --max-batch-size 1

export EDGELLM_PLUGIN_PATH="$PWD/build-refit/lib/libNvInfer_edgellm_plugin.so"
build-refit/bin/llm_inference \
  --engineDir=/path/to/refit-engines \
  --checkpointDir=/path/to/checkpoint \
  --inputFile=/path/to/input.json --outputFile=/path/to/output.json
```

Generate the target-specific CuTeDSL artifacts before configuring CMake and
include any additional TensorRT package dependencies in `LD_LIBRARY_PATH`.
Keep default and weightless engines in separate directories and rebuild for
each target GPU. A runtime built without the native opt-in rejects weightless
engines.

## Build All Components

The following commands work with either installation. A source install defaults
to `build/lib/libNvInfer_edgellm_plugin.so`; use `--plugin-path` only for a different
source-build location. A published wheel resolves its packaged plugin without
that option.

The default `--components all` selection discovers the checkpoint family and
builds all of its components in runtime order:

```bash
trtedge build \
  --model-dir /path/to/checkpoint \
  --engine-dir /path/to/engines \
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

### Pre-Quantized Checkpoints

Compatibility is determined from the registered model architecture,
quantization metadata, and tensor layout rather than the checkpoint publisher.
The direct builder recognizes these dense-weight contracts:

| Checkpoint contract | Lowering |
|---|---|
| AWQ, AutoAWQ, GPTQ, GPTQModel, or AutoRound W4 | INT4 W4A16 plugin with provider zero-point correction |
| GPTQ, GPTQModel, or compressed-tensors packed W8 | INT8 W8A16 dequantization |
| compressed-tensors packed W4 | INT4 W4A16 plugin |
| Per-channel or block FP8 | TensorRT FP8 lowering on a supported target |
| ModelOpt FP8, NVFP4, MXFP8, INT4 AWQ, or INT8 SmoothQuant | Native model-specific lowering |

The builder reads the provider's packed tensors directly; their quantization
packages are not runtime dependencies. Unsupported metadata or missing weight
and scale tensors fail the build instead of silently compiling the affected
projection as FP16.

### Target Precision Fallback

The direct builder retains the checkpoint weights and changes the execution
contract when their activation precision is unavailable on the target:

- NVFP4 W4A4 becomes NVFP4 W4A16 before native NVFP4 targets.
- FP8 W8A8 becomes INT8 W8A16 before native FP8 targets. Other unsupported
  FP8 or MXFP8 weight layouts use the same W8A16 path.
- Block FP8 becomes MXFP8 on targets with native MXFP8 execution.
- Packed compressed-tensors W4A4 and W8A8 metadata is consumed as W4A16 and
  W8A16, respectively, because the serialized integer weight layout is
  reusable without the unavailable activation contract.

Routed experts use their model-specific W4A16 or FP16 fallback because their
plugin contracts differ from dense GEMMs. The build log reports each
target-wide conversion that it applies.

Use `--components` only for an intentional partial rebuild:

```bash
trtedge build \
  --model-dir /path/to/checkpoint \
  --engine-dir /path/to/engines \
  --components visual,audio
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
| pi0.5 | Visual, prefix, action, hoisted AdaRMS condition |
| GR00T N1.7 | Visual, VL backbone, action |
| DiffusionGemma | Diffusion backbone, visual |
| Cosmos3 | LLM, visual; build the policy engines with `--components und-prefill,gen,vae-encoder` |
| MiniMax-H3 | Text encoder, generation, VAE decoder, audio decoder |

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
| pi0.5 prefix | `prefix/prefix.engine` |
| pi0.5 condition | `cond/cond.engine` |
| GR00T VL backbone | `vl_backbone/vl_backbone.engine` |
| Cosmos3 understanding prefill | `und_prefill/und_prefill.engine` |
| Cosmos3 generation policy | `gen/gen.engine` |
| Cosmos3 VAE encoder | `vae_encoder/vae_encoder.engine` |
| MiniMax-H3 text encoder | `text_encoder/text_encoder.engine` |
| MiniMax-H3 generation | `gen/gen.engine` |
| MiniMax-H3 VAE decoder | `vae_decoder/vae_decoder.engine` |
| MiniMax-H3 audio decoder | `audio_decoder/audio_decoder.engine` |

Runtime configs, tokenizers, chat templates, preprocessing files, and
checkpoint binding recipes are written beside the component engines.
Those recipes include structural identities for every provider tensor and
bounded content samples spanning the component and target checkpoints used
during the build. A runtime checkpoint with matching tensor shapes but a
different recorded identity is rejected before CUDA page registration. Each
engine build reports checkpoint-identity, TensorRT, and remaining frontend
time separately.

Tensor-parallel builds keep TensorRT-native dense weights, embeddings, and
eligible FP16 output heads in each rank engine by default. For
supported NVFP4 models, fused `o_proj` and `down_proj` plugins carry
rank-neutral recipes for packed weights and block scales. Each runtime rank
loads only its local input-axis shard and registers those immutable buffers as
plugin resources; they do not add TensorRT input bindings. Tensors required by
every rank, including the current Qwen embedding contract, remain replicated.
This preserves the TensorRT-native lowering used by the single-device NVFP4
path while avoiding full copies of the shardable fused-plugin tensors.

## Run The Engines

Wheel users can run the [Python runtime example](../examples/python-runtime.md#use-an-existing-engine)
with an existing compatible text engine, or use the
[Python server](../examples/server.md). The C++ commands below,
including speculative inference, require a source build; wheels do not install
these executables.

Use the same C++ runtime and request JSON used by the ONNX workflow. A
text-only engine runs directly from the output directory:

```bash
build/bin/llm_inference \
  --engine-dir=/path/to/engines \
  --checkpoint-dir=/path/to/checkpoint \
  --input-file=/path/to/input.json \
  --output-file=/path/to/output.json
```

For VLM, ASR, and omni checkpoints, pass the same output directory as the
multimodal directory. The runtime discovers the model-specific `visual/` and
`audio/` components:

```bash
build/bin/llm_inference \
  --engine-dir=/path/to/engines \
  --multimodal-engine-dir=/path/to/engines \
  --checkpoint-dir=/path/to/checkpoint \
  --input-file=/path/to/input.json \
  --output-file=/path/to/output.json
```

TTS uses the model-specific runtime:

```bash
build/bin/qwen3_tts_inference \
  --talker-engine-dir=/path/to/engines/talker \
  --code2wav-engine-dir=/path/to/engines/code2wav \
  --tokenizer-dir=/path/to/engines/talker \
  --clone-encoder-dir=/path/to/engines/clone_encoders \
  --checkpoint-dir=/path/to/checkpoint \
  --input-file=/path/to/input.json \
  --output-audio-dir=/path/to/audio-output
```

Alpamayo action builds default to `--reuse-llm-kv-cache=true`, which reuses
the paged prefix from an LLM using TP=1. For an LLM using TP>1, build the
action component separately with
`--components action --reuse-llm-kv-cache=false` to select dense KV inputs.
This option applies only to Alpamayo action builds.

Alpamayo uses the action runtime:

```bash
build/bin/action_inference \
  --engine-dir=/path/to/engines \
  --multimodal-engine-dir=/path/to/engines \
  --checkpoint-dir=/path/to/checkpoint \
  --input-file=/path/to/input.json \
  --output-file=/path/to/output.json
```

pi0.5 builds all four FP16 policy engines directly from the converted PyTorch
checkpoint. Quantized checkpoints and non-FP16 dense modes are rejected before
TensorRT network creation. The policy name is optional only when the checkpoint
feature schema unambiguously identifies `pi05_libero`:

```bash
.venv/bin/trtedge build \
  --model-dir /path/to/pi05-checkpoint \
  --engine-dir /path/to/pi05-engines \
  --plugin-path /path/to/build/lib/libNvInfer_edgellm_plugin.so \
  --pi05-policy-config pi05_libero \
  --max-batch-size 1

build/bin/pi05_policy_inference \
  --engineDir /path/to/pi05-engines \
  --inputFile observation.json \
  --output action.json
```

The command writes `visual/`, `prefix/`, `action/`, `cond/`, root-level
`embed_tokens.safetensors`, `text_tokenizer/`, and `policy.json` in the
layout consumed by the existing pi0.5 runtime. If the checkpoint carries no
`assets/norm_stats.json`, the builder warns and the bundle is not yet runnable
through the raw policy API; stage the matching file under
`<engine-dir>/assets/` before inference. The runtime rejects a missing file.

Use `--pi05-max-prefix-len` to override the 976-token default. It cannot be
smaller than the declared camera slots plus the 200-token prompt: 712 for
LIBERO/DROID and 968 for ALOHA. `--pi05-denoise-steps` overrides the
checkpoint/default denoise schedule.

Cosmos3 policy checkpoints use their image-and-instruction runtime:

```bash
build/bin/cosmos3_policy_inference \
  --engine-dir=/path/to/engines \
  --image=/path/to/observation.png \
  --prompt="Move the object into the tray." \
  --output=/path/to/action.json
```

See the [examples](../examples/index.md) for each runtime's request format.

`--checkpoint-dir` is required when the engines have external weights. During
runtime initialization, Edge-LLM maps
provider safetensors or indexed contiguous tensor ranges from PyTorch ZIP
`.bin` checkpoints, performs all requested casts, transposes, packing, scale
conversion, and direct final-layout writes. It then synchronizes the
preparation stream and releases the checkpoint mappings. Most prepared tensors
are immutable TensorRT inputs. Fused NVFP4 tensor-parallel `o_proj` and
`down_proj` weights are immutable plugin resources addressed by a serialized
resource id instead, so they do not enlarge the TensorRT binding table. The
inference path performs no weight conversion or allocation.

For tied FP16/BF16 token embeddings and output projections with identical
runtime conversion contracts, both runtime tensors share one weight-arena
allocation. Embedding scaling or any other recipe difference disables this
alias automatically.

The same checkpoint-backed engines can be profiled with `llm_bench`:

```bash
build/bin/llm_bench \
  --engine-dir=/path/to/engines \
  --checkpoint-dir=/path/to/checkpoint \
  --mode=fixed_round \
  --input-len=128 \
  --output-dir=/path/to/new-benchmark-report
```

## Speculative Decoding

One invocation builds both `spec_base.engine` and `spec_draft.engine`, plus the
target checkpoint's non-LLM components.

EAGLE3, DFlash, JetSpec, Domino, DSpark, and Gemma4 MTP use paired checkpoints:

```bash
trtedge build \
  --model-dir /path/to/target \
  --draft-model-dir /path/to/draft \
  --spec-type eagle3 \
  --engine-dir /path/to/engines
```

Replace `eagle3` with `dflash`, `jetspec`, `domino`, `dspark`, or `gemma4_mtp`
as appropriate. With `--spec-type eagle3`, a draft whose configuration sets
`fc_norm`, `norm_output`, or a low-rank candidate head builds as an EAGLE-4
draft; `--draft-dense-head` builds its full-vocabulary `lm_head` instead of the
candidate head. See [EAGLE-4](../examples/speculative-decoding.md#eagle-4) for
the command and its limitations.
DFlash2 checkpoints use `dflash`; the checkpoint architecture
selects DFlash2. Its base verify profile defaults to the checkpoint block size,
which serves chain verification; pass `--max-spec-verify-tokens N` (up to 128)
to also verify DDTrees of up to N nodes. LiLiCorr drafts also use `dflash`:
their `dflash_config.projector_type = "lilicorr"` contract builds the DFlash2
backbone with the candidate-lattice head instead of the selector. The LiLiCorr
draft engine is built at exactly the checkpoint block size, and LiLiCorr
verifies chains only. Direct LiLiCorr builds are validated on SM120 only; see
[LiLiCorr](../examples/speculative-decoding.md#lilicorr). JetSpec reads the
provider's `jetspec_config` or compatible `dflash_config` and requires its
causal proposal head. Domino reads the DFlash checkpoint's
`dflash_config.projector_type = "domino"` contract and writes
`domino_heads.safetensors` (GRU input table from the target `embed_tokens`, GRU
recurrent weight, state projection) next to the draft engine.

MTP, Gemma MTP, DFlash v1, JetSpec, Domino, and DSpark use the same target artifact
for chain and tree verification. Select the topology at runtime with
`--specDraftTopK`: 1 for a chain and greater values for supported branching.
Tree decoding is greedy-only. Recurrent target plugins always receive parent,
depth, and layout metadata; attention plugins do not receive the recurrent
layout marker. Build capacities bound runtime proposal and verification sizes;
no topology-specific build option is needed.

Qwen3.5 native MTP reads draft layers from the target checkpoint:

```bash
trtedge build \
  --model-dir /path/to/qwen3.5-checkpoint \
  --spec-type mtp \
  --engine-dir /path/to/engines
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
build/bin/llm_inference \
  --engine-dir=/path/to/engines \
  --checkpoint-dir=/path/to/qwen3.5-checkpoint \
  --input-file=/path/to/input.json \
  --output-file=/path/to/output.json \
  --spec-decode
```

Paired drafts use an explicit second checkpoint path:

```bash
build/bin/llm_inference \
  --engine-dir=/path/to/engines \
  --checkpoint-dir=/path/to/target \
  --draft-checkpoint-dir=/path/to/draft \
  --input-file=/path/to/input.json \
  --output-file=/path/to/output.json \
  --spec-decode
```

`--draft-checkpoint-dir` applies to EAGLE3, DFlash, DFlash2, LiLiCorr, JetSpec,
Domino, DSpark, and Gemma4 MTP. It is rejected for native Qwen MTP because its
draft layers are in `--checkpoint-dir`.

## Optional Features

| Feature | Build option or behavior |
|---|---|
| Runtime LoRA inputs | `--max-lora-rank N` |
| Reduced vocabulary | `--reduced-vocab-dir DIR` |
| DFlash V1 draft vocabulary | `--draft-reduced-vocab-dir DIR` |
| Jump-forward with tree drafting | `--max-spec-verify-tokens N` above the runtime `--spec-verify-tokens`; see [Jump-forward](../features/guided-decoding.md#jump-forward) |
| FP8 embedding sidecar | `--fp8-embedding` |
| Tensor parallel rank | `--tp-size N --tp-rank R` |
| Alpamayo action KV reuse | `--reuse-llm-kv-cache=true\|false` (default: `true`; use `false` for LLM TP>1) |
| Detailed TensorRT profiling names | `--profiling-detailed` |
| EAGLE-4 draft with its full-vocabulary `lm_head` | `--draft-dense-head` |
| Partial component rebuild | `--components NAME[,NAME...]` |
| First N decoder layers only | `--num-decoder-layer N` |
| TensorRT-native MoE layer for Qwen3-family MoE | `--trt-native-moe` |

JetSpec and chain-MTP draft vocabulary reduction use the ONNX export workflow described in [Vocabulary Reduction](../features/reduce-vocab.md); the direct engine builder currently accepts `--draft-reduced-vocab-dir` only for DFlash V1 drafts.

`--num-decoder-layer` truncates the LLM to its first N decoder layers, for the
few-layer numeric validation (`scripts/few-layer-validation.sh`). Weights are read
by name, so the dropped layers are simply never requested and the checkpoint needs
no preparation.

`--trt-native-moe` builds Qwen3-family MoE experts through TensorRT's native MoE
layer instead of the Edge-LLM MoE plugins, the same arm the ONNX exporter selects
with its flag of the same name: FP16/BF16 experts use the FP16 layer, ModelOpt NVFP4
experts run as W4A8, and INT4, NVFP4-A16 or Nemotron-H experts are rejected. It needs
TensorRT 11.3 or newer and an SM 10.x or 11.x active device; see
[Quantization](../features/quantization.md#select-the-tensorrt-native-moe-layer-for-qwen3-family-moe).

Tensor parallelism currently builds one rank per invocation. Invoke the command
once for each `--tp-rank` and place the rank artifacts according to the normal
multi-GPU runtime layout. Tensor-parallel direct builds currently support the
LLM component without speculative decoding; tensor-parallel speculative
decoding is not supported.

## Support And Validation Status

The explicit registry includes Llama, Mistral, Qwen2, Qwen3, Qwen3-MoE,
Qwen2/2.5/3-VL, Qwen3.5 dense and MoE, Qwen3-ASR, Qwen3-Omni, Qwen3-Omni-Next,
Qwen3-TTS, InternVL3/3.5, Phi-4 Multimodal, Nemotron-H/Omni, Gemma4 and Gemma4
Unified, DiffusionGemma, Cosmos3, Alpamayo, pi0.5, and GR00T N1.7. EAGLE3, EAGLE-4,
MTP, DFlash/DFlash2, LiLiCorr, JetSpec, DSpark, and Gemma4 assistant drafts use
model-owned speculative definitions. Unsupported `model_type` values fail before
TensorRT network creation and list the registered choices.

The following table distinguishes implementation from automatic direct-builder
CI coverage. An implemented row can still have model-specific restrictions.

| Capability | Frontend status | Direct-builder L0 coverage |
|---|---|---|
| FP16/BF16 dense LLM | Implemented | Qwen2.5 on A30 |
| FP8 and block-wise FP8 dense LLM | Implemented | Qwen2.5 on RTX 5090 |
| NVFP4 dense LLM | Implemented | Qwen3 on RTX 5090 |
| Qwen3-family MoE through TensorRT's native MoE layer (`--trt-native-moe`) | Implemented for FP16 and NVFP4 W4A8 experts | Qwen3-30B-A3B NVFP4 few-layer parity on B100 (TensorRT >= 11.3) |
| MXFP8 and INT8 SmoothQuant | Implemented | Not yet |
| INT4 AWQ, ModelOpt AWQ, and GPTQ | Implemented | Not yet |
| FP8 KV cache | Implemented from checkpoint metadata | Not yet |
| FP8 embedding and reduced vocabulary | Implemented | Not yet |
| Runtime LoRA inputs | Implemented | Not yet |
| EAGLE3, EAGLE-4, Qwen3.5 MTP, DFlash, DFlash2, LiLiCorr, JetSpec, Domino, DSpark, and Gemma4 MTP | Implemented | EAGLE3 with Qwen3 on A30 |
| DiffusionGemma block diffusion | Implemented | Not yet |
| Visual, audio, TTS, omni, action, Cosmos3 policy, and pi0.5 components | Implemented for registered families | Not yet |
| Tensor parallel graph generation | Implemented per rank | Not yet |

Known gaps relative to the supported ToT workflow are tracked explicitly:

- Quantization and calibration are checkpoint-producing steps outside this
  builder; run `trtedge quantize` first.
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
