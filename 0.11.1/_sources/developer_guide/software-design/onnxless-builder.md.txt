# Experimental ONNX-less Builder Design

The experimental builder compiles TensorRT engines directly from checkpoint
configuration and safetensors:

```text
checkpoint
  -> model-specific configuration and weight conversion
  -> model-specific NetworkModule
  -> symbolic Tensor and F.* operations
  -> TensorRT INetworkDefinition
  -> optimization profiles and serialized engine
  -> model-specific runtime artifacts
```

There is no ONNX graph, ONNX parser, or dependency on the existing ONNX
exporter in this path. The result uses the existing TensorRT Edge-LLM C++
runtime contract.

For installation, commands, runtime examples, component paths, supported
features, and current limitations, see
[Experimental ONNX-less Engine Build](../../user_guide/getting_started/direct-engine-builder.md).

## Design Goals

- Keep model definitions close to Hugging Face Transformers structure:
  modules create named child modules in `__init__` and implement `forward`.
- Give every engine component an explicit, model-specific input and output
  contract.
- Present TensorRT-native layers and Edge-LLM extension operations through one
  `ops.functional` API.
- Keep raw TensorRT network access out of model definitions.
- Load configuration and safetensors on CPU without constructing or tracing a
  provider `torch.nn.Module`.
- Build every component selected for a checkpoint in one command while
  preserving one TensorRT engine per runtime component.
- Reuse operations and genuinely architecture-neutral modules, but keep
  configuration, weight mapping, component graphs, and runtime artifacts owned
  by each model family.

## Package Boundaries

The implementation is under `experimental/builder`:

| Layer | Main files | Responsibility |
|---|---|---|
| CLI orchestration | `cli.py` | Resolve components, expand speculative base/draft builds, load the plugin once, and build components in runtime order. |
| Build driver | `core/builder.py` | Create a strongly typed TensorRT network, bind checkpoint metadata, configure profiles, serialize the engine, and publish runtime weight metadata. |
| Checkpoint contract | `core/config.py`, `core/quantization.py`, `core/weights.py` | Parse model/quantization metadata and provide streaming safetensors access. |
| Component contract | `core/contracts.py` | Define component names, output paths, profile limits, and deterministic build order. |
| Model registry | `models/registry.py` | Map an exact checkpoint `model_type` and component to a concrete `NetworkModule`, configuration module, weight converter, and artifact writer. |
| Model families | `models/<family>/` | Own configuration selection, model modules, checkpoint key conversion, preprocessing, and runtime artifacts. |
| Public modeling API | `ops/tensor.py`, `ops/module.py`, `ops/functional/` | Provide PyTorch-like symbolic tensors, `Module`/`NetworkModule`, and semantic operations. |
| TensorRT lowering | `ops/backend.py` | Own all `INetworkDefinition` calls, plugin creator lookup, plugin field encoding, and external weight inputs. |
| Runtime artifacts | `core/artifacts/` plus `models/<family>/artifacts.py` | Write configs, tokenizers, chat templates, embeddings, preprocessing assets, and external-weight manifests. |

The boundary is enforced by convention and imports: model files receive a
`BuildContext`, create `Module` objects, and call `Tensor` methods or
`ops.functional` functions. They do not receive an
`INetworkDefinition`.

## Build Lifecycle

One command follows this sequence:

1. `cli._resolve_build_selection` loads `BundleConfig` and asks
   `models/registry.py` for the exact component set associated with the root
   `model_type`.
2. `cli._build_plan` creates one entry per component. A speculative request
   expands the LLM entry into target/base and draft entries while keeping
   non-LLM components in the same plan.
3. `cli._build` loads `libNvInfer_edgellm_plugin.so` once, then invokes
   `_build_one` for each planned component.
4. `core.builder.build_engine` creates a strongly typed
   `trt.INetworkDefinition`, `trt.IBuilderConfig`, checkpoint `Weights`, and
   the internal `Net` lowering backend.
5. `models.build_model` asks the registry for one concrete `NetworkModule`,
   creates a `BuildContext`, and invokes `NetworkModule.build`.
6. `NetworkModule.build` declares component inputs with `input_tensors`, calls
   the model's `forward`, and marks only its explicitly returned outputs.
7. `core.builder` creates the component-specific optimization profiles and
   calls `builder.build_serialized_network`.
8. The model-owned artifact writer emits the runtime config. Checkpoint
   bindings or validated transformed-sidecar references are then added to that
   component's manifest.

Each loop iteration creates one independent TensorRT network. A VLM therefore
has one `NetworkModule` and `INetworkDefinition` for its LLM and another for its
visual encoder. The one-command behavior is orchestration, not a single engine
that mixes incompatible component I/O contracts.

## Module And Network Ownership

The frontend has two module types.

### `Module`

`Module` represents a nested model layer such as attention, MLP, RMSNorm,
convolution, or projector.

- Child modules are constructed in `__init__`.
- `__call__` enters the active build scope and delegates to `forward`.
- A module owns checkpoint names and ordinary configuration.
- A module cannot add network inputs or mark outputs.

This keeps model code structurally similar to Transformers:

```python
class Qwen3DecoderLayer(DecoderLayer):
    attention_class = Qwen3Attention


class Qwen3Model(DecoderModel):
    layer_class = Qwen3DecoderLayer
```

### `NetworkModule`

`NetworkModule` represents exactly one runtime component and one TensorRT
network.

- `input_tensors` declares the engine inputs and their dynamic dimensions.
- `forward` composes model-specific submodules and returns explicitly named
  tensors.
- `mark_outputs` is the only model-facing output-marking path.
- `from_config` is the extension point for components that need component
  metadata or build-profile limits during construction.

For example, `Qwen3ForCausalLM` owns paged KV-cache, RoPE, context-length,
page-table, token-selection, logits, and feedback-hidden-state bindings.
Visual, audio, action, TTS, and speculative classes define different contracts
instead of inheriting an assumed LLM interface.

`BuildContext` carries the active `Net`, normalized `DeviceConfig`, streaming
`Weights`, `BuildOptions`, checkpoint bundle, and build arguments. The
`ContextVar` in `ops/scope.py` makes that context available while a module is
executing, so every public operation can retain a tensor-first signature
without passing a network argument through every `forward` call.

## Operation API And TensorRT Lowering

Model code has one operation level:

- Tensor syntax for familiar expressions: `x + residual`, `x.reshape(...)`,
  `x @ weight`, `x.mean(...)`, and `x.silu()`.
- `from ...ops import functional as F` for free or multi-output operations:
  `F.linear_from_weights`, `F.rms_norm`, `F.attention`, `F.vit_attention`,
  `F.gated_delta_net`, and `F.nvfp4_moe`.

Neither API exposes whether an operation lowers to native TensorRT layers or an
Edge-LLM extension. That distinction exists only in `ops/backend.py`.

TensorRT API ownership is split at one explicit boundary:

- `core/builder.py` owns builder, network, builder-config, optimization-profile,
  serialization, and timing-cache lifecycle calls.
- `ops/backend.py` owns all layer-creation calls on
  `trt.INetworkDefinition`, including native TensorRT layers, constants,
  network weight inputs, and Edge-LLM extension layers.

Model modules and `ops/functional` never call TensorRT directly. Functional
operations define frontend semantics; the active `Net` in `ops/backend.py`
lowers those semantics into one or more TensorRT layers.

Concrete call paths are:

| Model expression | Public implementation | Backend method in `ops/backend.py` | TensorRT API |
|---|---|---|---|
| `x + residual` | `Tensor.__add__` | `Net.elementwise` | `add_elementwise` |
| `x.reshape(shape)` | `Tensor.reshape` | `Net.reshape` | `add_shuffle` |
| `x @ weight` | `Tensor.matmul` | `Net.matmul` | `add_matrix_multiply` |
| FP16 `Linear(x)` | `Linear.forward` -> `F.linear_from_weights` | `Net.linear_from_weights` -> `Net.linear` | `add_constant`, `add_matrix_multiply`, optional `add_elementwise` |
| `RMSNorm(x)` | `RMSNorm.forward` -> `F.rms_norm` | `Net.rmsnorm` | `add_cast`, `add_reduce`, `add_elementwise`, `add_unary` |
| Decoder attention | `F.attention` | `Net.operation("attention", ...)` | plugin creator plus `add_plugin_v3` |
| ViT attention | `F.vit_attention` | `Net.operation("vit_attention", ...)` | plugin creator plus `add_plugin_v3` |

### Native TensorRT Layer Example

Dense projection shows how a semantic frontend operation maps to a native
TensorRT layer composition. The builder creates a strongly typed network and
represents an FP16 projection as:

```text
checkpoint weight -> IConstantLayer
activation, weight -> IMatrixMultiplyLayer
optional bias -> IElementWiseLayer
```

The implementation is `Net.linear` in `ops/backend.py`; the strongly typed
network does not use the legacy fully-connected layer API. Quantized
projections enter through the same `Linear` module and
`F.linear_from_weights` call, after which `Net.linear_from_weights` dispatches
according to checkpoint metadata:

- FP8, MXFP8, NVFP4, and INT8 SmoothQuant use their TensorRT quantize,
  dequantize, cast, MatMul, or extension lowering.
- INT4 AWQ/GPTQ uses the INT4 groupwise GEMM extension.
- Excluded or mixed-precision modules fall back to the FP16 path.

The model does not select a backend layer directly.

RMSNorm is another native composition. It is decomposed instead of represented
by a model-owned custom operation:

```text
x
 -> cast FP32
 -> square
 -> reduce mean over hidden dimension
 -> add epsilon
 -> square root
 -> divide
 -> cast FP16
 -> multiply checkpoint scale
```

`ops/normalization.py` owns the checkpoint-facing `RMSNorm` module,
`ops/functional/core.py` owns the semantic function, and
`Net.rmsnorm` owns the TensorRT layer sequence.

### Edge-LLM Extension Layer Example

Semantic extension functions accept symbolic tensors and ordinary Python
attributes. For decoder attention:

```text
model
 -> F.attention(qkv, past_kv, ..., num_q_heads=..., head_size=...)
 -> functional._operation.operation("attention", tensors, attributes)
 -> Net.operation
 -> semantic name mapped to "AttentionPlugin"
 -> PluginField encoding and creator lookup
 -> INetworkDefinition.add_plugin_v3
```

`_OPERATION_CREATORS` in `ops/backend.py` is the only semantic-name to
creator-name map. `Net._operation_field` is the only generic attribute encoder.
Creator names, `PluginField` construction, and plugin layer creation must not
appear in model or semantic functional files.

## Adding An Operation

Add operations according to semantics, not provider:

1. Choose the appropriate file under `ops/functional/` and define a typed,
   tensor-first function. Add it to `ops/functional/__init__.py`.
2. For a native composition, add the minimal lowering method to `Net` and call
   it from the functional function or a `Tensor` method.
3. For an extension operation, add one stable semantic name to
   `_OPERATION_CREATORS` and call the private
   `functional._operation.operation` bridge.
4. Keep checkpoint lookup and model-specific defaults in a `Module`, not in
   `Net`.
5. Keep dynamic shape and output-count validation at the highest layer that
   owns that contract.

A caller should not need to know whether an implementation changed from a
plugin to a native TensorRT layer. That change stays behind the functional
operation boundary.

## Checkpoint And Weight Ownership

`BundleConfig` reads root and nested component configuration with plain JSON.
`DeviceConfig` normalizes the fields needed by text-like runtime components.
`core/quantization.py` reads `hf_quant_config.json` or an embedded
`quantization_config` and resolves a concrete precision per module.

`Weights` keeps safetensors files memory-mapped and requests tensors by
frontend name. A model family's `weights.py` owns aliases, prefix changes,
special tensor combinations, and quantized layout conversion. This prevents a
generic loader from accumulating model-name conditionals.

Large quantized extension weights can be static TensorRT network inputs:

- Dense INT4 GEMM qweights and scales
- INT4 MoE qweights
- NVFP4 MoE weight buffers

They remain parameters owned by `Linear` or the model-specific expert module;
they do not appear in model `forward` signatures. `Net.weight_input` records
only final tensor metadata and the provider-checkpoint conversion recipe.
`core/artifacts/external_weights.py` publishes those bindings in the component
config, and the C++ `ExternalWeightManager` maps, transforms, and binds the
original checkpoint once when loading the engine.

## Profiles And Runtime Artifacts

Text-like components receive separate context and generation optimization
profiles. Profile construction is centralized in `core/builder.py`, but the
shape set is driven by bindings declared by the concrete `NetworkModule`.
Visual, audio, Code2Wav, and action components receive component-specific
profiles derived from their own inputs and command limits.

Artifact ownership follows the model:

```text
models/<family>/artifacts.py
  -> optional family runtime_config.py / tokenizer.py / embeddings.py
  -> reusable serializers in core/artifacts/
```

The reusable serializers implement file formats. The family decides which
files and model-specific fields are required. This keeps sidecars aligned with
the component that consumes them.

## Adding A Model Family

Do not route a new architecture through another family's model definition
unless the provider architecture and runtime contract are genuinely identical.
Add a directory under `models/<family>/` with the applicable files:

| File | Required responsibility |
|---|---|
| `configuration.py` | Select and validate component configuration from the root checkpoint. |
| `modeling_<family>_<component>.py` | Define one `NetworkModule` per component and model-specific nested modules. |
| `weights.py` | Resolve checkpoint names and perform family-specific conversions. |
| `artifacts.py` | Select runtime artifact writers for every component. |
| `runtime_config.py` | Add model-specific runtime fields when generic fields are insufficient. |
| `tokenizer.py`, `embeddings.py`, `preprocessing.py` | Own optional family-specific assets. |
| `__init__.py` | Expose only the family's intended public definitions. |

Then:

1. Add one `ModelFamily` entry in `models/registry.py` with exact root
   `model_type` aliases and component classes.
2. Make each component's `NetworkModule.input_tensors` match the C++ runner
   bindings and profile requirements.
3. Return every runtime-visible output by name from `forward`.
4. Verify one-command `--components all` output paths and sidecars.
5. Add a focused end-to-end test that compiles the engine, executes the
   corresponding C++ runtime, and checks output accuracy.
6. Update the [supported model matrix](../../user_guide/getting_started/supported-models.md)
   only after the model's complete runtime contract is validated.

## Validation

`tests/defs/test_builder_pipeline.py` exercises the public contract rather than
individual helper classes:

1. Resolve a real checkpoint through the standard test configuration.
2. Invoke one all-component direct build.
3. Check every expected engine path and non-empty artifact.
4. Execute the matching C++ runtime once.
5. Validate modality-specific output contracts.
6. Run the existing dataset accuracy check.

Keep frontend tests focused on complete engine-build and runtime paths. Unit
tests for isolated helpers do not substitute for a model-level output check.
