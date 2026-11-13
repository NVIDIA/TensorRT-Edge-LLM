<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Extension wiring by route

Read the section for the route the user named. Each row is a place that must
agree with the others; missing one usually fails late, at engine build or
runtime binding.

## Maintained ONNX route

### Text model family

| Step | File | Contract |
|---|---|---|
| Model | `tensorrt_edgellm/models/<family>/modeling_<family>*.py` | `nn.Module` built from one `ModelConfig`; parameter names follow the provider checkpoint |
| Registration | `tensorrt_edgellm/_export_api.py` | `register_model(model_type, ModelClass, default_attention_scale)` for every root and promoted text `model_type` the checkpoint can present |
| Config fields | `tensorrt_edgellm/config.py` (`ModelConfig.from_pretrained`) | Parse model fields from checkpoint metadata, not from a display name |
| Weights | `tensorrt_edgellm/checkpoint/loader.py`, `repacking.py`, or a family key remap passed through `AutoModel.from_pretrained(key_remap=...)` | Only "Key not found in model" is logged (DEBUG); prefix and remap drops are silent, and the CLI logs totals. See Weight reconciliation in `SKILL.md` |
| Runtime config | `write_runtime_artifacts` in `tensorrt_edgellm/checkpoint/checkpoint_utils.py` | Every field `llm_build` and the runtime read must be in `llm/config.json` |
| LLM engine profiles | `cpp/builder/llmBuilder.cpp` | Profiles key on ONNX input names and `config.json` fields; an added graph input needs a profile and a runtime binding |
| Runtime config parsing | `cpp/runtime/config/llmEngineConfig.cpp` | Reads the exported `config.json` |

### Components

| Component | Python | C++ |
|---|---|---|
| Visual encoder | `build_<family>_visual(vcfg, weights, model_config=, dtype=)` in `models/<family>/modeling_<family>_visual.py`, returning a module with `get_onnx_export_args`; register in `_VISUAL_REGISTRY`, `_VISUAL_FAMILY_MODULE`, `_VISUAL_FAMILY_BUILD_FN`, and `_get_visual_config` (`tensorrt_edgellm/onnx/export_encoder.py`); add the root type to `_VLM_MODEL_TYPES` and write the visual `config.json` in `_export_visual` (`tensorrt_edgellm/onnx/cli.py`) | `ModelType` and `stringToModelType` in `cpp/multimodal/common/modelTypes.h`; profile in `cpp/builder/visualBuilder.cpp`; runner under `cpp/multimodal/<series>/` selected in `cpp/multimodal/common/multimodalRunner.cpp` (`cpp/multimodal/README.md`) |
| Audio encoder | `_AUDIO_MODEL_TYPES` and `_AUDIO_KEY_PREFIX` in `tensorrt_edgellm/onnx/export_encoder.py`; `_AUDIO_MODEL_TYPES` and `_export_audio` in `tensorrt_edgellm/onnx/cli.py` | Same `ModelType` path; profile in `cpp/builder/audioBuilder.cpp` |
| Action head | `_ACTION_MODEL_TYPES` and `_export_action` in `tensorrt_edgellm/onnx/cli.py`; `export_action_onnx` | `cpp/builder/actionBuilder.cpp`, `cpp/action/` |
| Talker, code predictor, Code2Wav | `_LLM_COMPONENTS`, `_CODE2WAV_MODEL_TYPES`, `_LAYOUT_OVERRIDES` in `tensorrt_edgellm/onnx/cli.py` | `llm_build`, `audio_build`, and the matching runtime |
| Speculative draft | Draft detection and variants in `tensorrt_edgellm/model.py` and `config.py`; `_export_*_draft` in `tensorrt_edgellm/onnx/cli.py` | `llm_build --specDraft`; base engine with `--specBase` |

Media-token embedding assembly runs in `cpp/kernels/embeddingKernels/` and
`cpp/runtime/preprocess/`; check them when a model places image or audio
embeddings differently.

### Custom operation (ONNX sidecar contract)

All four pieces are required and must use the same op name and signature:

1. `torch.library.custom_op` stub with a fake implementation in
   `tensorrt_edgellm/models/ops.py`.
2. Translation in `build_custom_translation_table()`
   (`tensorrt_edgellm/onnx/dynamo_translations.py`).
3. ONNX schema registered by `register_tensorrt_edgellm_onnx_custom_schemas()`
   (`tensorrt_edgellm/onnx/onnx_custom_schemas.py`).
4. TensorRT plugin and its creator under `cpp/plugins/`; use the
   `edge-llm-plugin-extension` skill (`edge-llm-cutedsl-kernel` for a CuTe DSL kernel)
   (`docs/source/developer_guide/customization/tensorrt-plugins.md`).

### Runtime behavior

| Behavior | Where |
|---|---|
| Chat template | `tensorrt_edgellm/chat_template.py` copies the provider Jinja; `cpp/chatTemplate/` renders it. A model without a provider template needs an entry in `_MANUAL_BY_MODEL_TYPE` and a C++ renderer. Contract: `docs/source/user_guide/format/chat-template-format.md` |
| Tokenizer | `cpp/tokenizer/`, reading the exported `tokenizer.json` |
| Image, video, audio preprocessing | `cpp/multimodal/<series>/`, `cpp/runtime/preprocess/` |
| Server tool-call and reasoning parsers | `tensorrt_edgellm/server/parsing/tool_calling.py`, `tensorrt_edgellm/server/parsing/reasoning.py` |

## Experimental checkpoint-direct route

Follow "Adding A Model Family" in
`docs/source/developer_guide/software-design/onnxless-builder.md` for the
per-family file list (`configuration.py`, `modeling_<family>_<component>.py`,
`weights.py`, `artifacts.py`, and the optional files it lists). The wiring it implies:

| Step | File | Contract |
|---|---|---|
| Registration | `ModelFamily` in `FAMILIES`, `tensorrt_edgellm/builder/models/registry.py` | `variants` maps every exact root `model_type` to its component set; `components` maps each `Component` to a `ComponentDefinition` naming its `NetworkModule`; duplicates fail at import |
| Configuration | `models/<family>/configuration.py` | Selects and validates each component's config from the root checkpoint |
| Network | `models/<family>/modeling_*.py` | One `NetworkModule` per component; `input_tensors` match the C++ runner bindings; outputs returned by name |
| Operations | `tensorrt_edgellm/builder/ops/functional/`, `Net` or `_OPERATION_CREATORS` in `tensorrt_edgellm/builder/ops/backend.py` | "Adding An Operation" in the design doc; model code uses `F.<op>`, `Tensor` methods, and shared `ops` modules, never `Net` or TensorRT directly |
| Weights | `models/<family>/weights.py` | Provider aliases, prefixes, fusions, quantized layout conversion |
| Externalized weights | `Net.weight_input`; `tensorrt_edgellm/builder/core/artifacts/external_weights.py`; C++ `cpp/runtime/state/externalWeightManager.h` and `cpp/runtime/weight/` | Records a provider-key recipe in `config.json`; the runtime rebuilds it from `--checkpointDir`. A weight without a reproducible recipe stays an engine constant |
| Artifacts | `models/<family>/artifacts.py` over `tensorrt_edgellm/builder/core/artifacts/` | Runtime config, tokenizer, chat template, embeddings beside each engine |
| Speculative drafts | `SPECULATIVE_*` and `FAMILY_SPECULATIVE_*` tables in `registry.py` | Draft class, weights, configuration, and artifact writer per spec type |
| Component kind absent from `Component` | `Component` and `ComponentSpec` in `tensorrt_edgellm/builder/core/contracts.py` | Only when no existing component fits |

Report as unsupported, rather than routing around, any capability listed as a
gap in "Support And Validation Status" of
`docs/source/user_guide/getting_started/direct-engine-builder.md`.

## Keep the contracts apart

| Contract | Lives in | Not interchangeable with |
|---|---|---|
| ONNX custom-op sidecar | `ops.py` stub, dynamo translation, ONNX schema, plugin | Direct-builder functional op lowering |
| ONNX runtime sidecars | `write_runtime_artifacts` output beside `model.onnx` | Direct-builder `artifacts.py` output |
| Direct-builder checkpoint binding | `weights.py`, `Net.weight_input`, provider-key recipes read from `--checkpointDir` | ONNX export's `--externalize-weights` output (`tensorrt_edgellm/external_weights.py`; not the builder flag of the same name), which writes its own safetensors weight files |
