<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Operation wiring map

Each row is one place the contract must agree; a skipped row fails at engine
build or, worse, silently at inference.

## Per-layer map

| Layer | Maintained ONNX route | Experimental checkpoint-direct route |
|---|---|---|
| Semantics and eager reference | `tensorrt_edgellm/models/ops.py`: `@torch.library.custom_op("trt_edgellm::<op>")` (older ops use `trt::`) plus `register_fake` for shape and dtype | `tensorrt_edgellm/builder/ops/functional/<domain>.py`: typed tensor-first function, exported from `tensorrt_edgellm/builder/ops/functional/__init__.py` |
| Lowering | `tensorrt_edgellm/onnx/dynamo_translations.py`: `@script()` translation that emits `_trt_edgellm.<CreatorName>(...)`, plus an entry in `build_custom_translation_table()` | Semantic name in `_OPERATION_CREATORS` (`tensorrt_edgellm/builder/ops/backend.py`); the functional calls `functional._operation.operation(name, inputs, **attributes)` |
| Schema | `tensorrt_edgellm/onnx/onnx_custom_schemas.py`: `OpSchema(domain="trt_edgellm")` appended to `_ALL_CUSTOM_SCHEMAS` | Creator `getFieldNames()`, read back through `Net.operation_attributes` |
| Model use | `tensorrt_edgellm/models/<family>/` calls `ops.<op>` | `tensorrt_edgellm/builder/models/<family>/` `Module` or `NetworkModule` calls `F.<op>` |
| Engine build | `cpp/builder/*Builder.cpp` calls `loadEdgellmPluginLib()`; the ONNX parser resolves the node by `op_type` | `tensorrt_edgellm/builder/core/builder.py` `load_plugin_library`; `Net.operation` calls `add_plugin_v3` |
| Plugin and kernel | `cpp/plugins/<plugin>/`, `cpp/kernels/<dir>/` (shared by both routes) | same |
| Weights | `tensorrt_edgellm/checkpoint/repacking.py`; externalized kinds in `tensorrt_edgellm/external_weights.py` | `tensorrt_edgellm/builder/weight_packing/`; `Net.weight_input` records a recipe; `tensorrt_edgellm/builder/core/artifacts/external_weights.py` publishes it |
| Runtime weight preparation | Engine constants, or safetensors files that `--externalize-weights` writes already in plugin layout | `cpp/runtime/state/externalWeightManager.cpp` with transforms in `cpp/kernels/weightsTransform/` (contract in its `README.md`) |

## Worked example: `PleGatherPlugin`

A small plugin with no creator fields (its ONNX node carries only
`plugin_version`), wired through both routes. Copy its shape.

| Layer | File and symbol |
|---|---|
| PyTorch stub | `tensorrt_edgellm/models/ops.py`: `trt::ple_pool_lookup` and `trt::ple_pool_gather` (older stubs use the `trt::` namespace; newer ones use `trt_edgellm::`) |
| ONNX translation | `tensorrt_edgellm/onnx/dynamo_translations.py`: `_ple_pool_lookup_translation`, `_ple_pool_gather_translation` (gather, then standard `DequantizeLinear`) |
| ONNX schema | `tensorrt_edgellm/onnx/onnx_custom_schemas.py`: `_ple_gather_schema` with required `plugin_version` |
| Direct functional | `tensorrt_edgellm/builder/ops/functional/core.py`: `ple_pool_lookup`, `ple_pool_gather` |
| Direct lowering | `tensorrt_edgellm/builder/ops/backend.py`: `"ple_pool_lookup": "PleGatherPlugin"` |
| C++ plugin | `cpp/plugins/pleGatherPlugin/pleGatherPlugin.{h,cpp}`: `validShape`, `validFormat`, `validContract` reused by `configurePlugin`, `onShapeChange`, and `enqueue` |
| Kernel | `cpp/kernels/embeddingKernels/pleGather.{h,cu}`: `launchPleGather` on the enqueue stream |
| CMake | none; picked up by the globs in `cpp/CMakeLists.txt` |
| Model use | `tensorrt_edgellm/models/gemma4/modeling_gemma4_text.py`, `tensorrt_edgellm/builder/models/gemma4/modeling_gemma4_text.py` |
| Export tests (patterns; not in the L0 list) | `tests/python-unittests/test_ple_pool_export.py`, `tests/python-unittests/test_gemma4_ple_pool.py` |

`PleGatherPlugin` has no `PluginRunner` execution test. When you extend it,
add one following `tests/python-unittests/test_qkv_concat_plugin.py`, which
also shows the rejection pattern.

## Weight layout alignment: `Int4GroupwiseGemmPluginV2`

A packed-weight plugin needs one layout selector to drive every producer and
the consumer. For INT4 groupwise GEMM, these must agree:

| Role | File |
|---|---|
| Selector | `int4_gemm_plugin_version()` in `tensorrt_edgellm/models/ops.py`; export flag `--int4-gemm-plugin-version`; direct `Net(int4_gemm_plugin_version=...)` |
| Emitted op | `tensorrt_edgellm/models/linear.py`; `Net.int4_linear` in `tensorrt_edgellm/builder/ops/backend.py` |
| Export repack | `tensorrt_edgellm/checkpoint/repacking.py` |
| Direct repack | `pack_cutedsl_fragment` in `tensorrt_edgellm/builder/weight_packing/int4.py` |
| Runtime repack | `cpp/kernels/weightsTransform/int4/groupwiseGemm/v2/int4CuteDslRepack.{h,cu}` |
| Consumer | `cpp/plugins/int4GroupwiseGemmPluginV2/` |
| Tests (pattern; not in the L0 list) | `tests/python-unittests/test_external_weights.py` |

The plugin also shows build-derived serialized state: `getFieldsToSerialize`
writes a field that is not among the creator's build fields.
