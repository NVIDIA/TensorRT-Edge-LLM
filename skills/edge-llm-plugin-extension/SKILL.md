---
name: edge-llm-plugin-extension
license: Apache-2.0
metadata:
  author: TensorRT Edge-LLM Team <noreply@nvidia.com>
  tags: [edge-llm, plugin, tensorrt, onnx, custom-op]
description: Add or change a TensorRT Edge-LLM operation (TensorRT IPluginV3 plugin, native TensorRT layer composition, or runtime-only operation) so one tensor, dtype, shape, layout, serialization, and weight-layout contract holds across the ONNX export route, the experimental checkpoint-direct builder, the C++ engine builders, libNvInfer_edgellm_plugin, and the runtime. Use when adding a custom op, plugin creator field, plugin input or output, dtype, weight repacking, or external-weight recipe; when a plugin is not found or rejects a configuration at engine build; or when choosing between a plugin and native TensorRT layers. Do not use for writing CuTe DSL kernels (use edge-llm-cutedsl-kernel), C++ test discipline (edge-llm-cpp-test), model CI cases (edge-llm-e2e-test), or diagnosing wrong model output (edge-llm-few-layer-test, then edge-llm-accuracy-debugging).
---

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Edge-LLM operation and plugin extension

An operation is one contract consumed by up to five places: the PyTorch
stub, the ONNX translation, the direct-builder lowering, the C++ plugin, and
the runtime that owns its buffers and weights. Read the canonical documents
first; this skill adds the cross-file wiring and the evidence gates.

- `docs/source/developer_guide/customization/tensorrt-plugins.md`: tensor
  contracts and kernel sources for four plugins only; `cpp/plugins/` is the full set.
- `docs/source/developer_guide/customization/customization-guide.md`, section
  "Custom Operators": the ONNX export extension points.
- `docs/source/developer_guide/software-design/onnxless-builder.md`, sections
  "Operation API And TensorRT Lowering" and "Adding An Operation".
- `AGENTS.md` (CUDA / GPU anti-patterns) and `CODING_GUIDELINES.md`.

Before choosing the wiring, read one recent merged plugin addition as a template
(`git log --oneline --no-merges -n 10 origin/main -- cpp/plugins`) and note where
this operation differs. Do not use a pre-PluginV3 implementation as a template.

## Instructions

Write the operation contract first, choose the smallest surface, wire both routes where required, implement the plugin, test, and gather end-to-end evidence in steps 1 to 5. Report with Completion evidence.

## 1. Write the contract, then choose

Before code, write down for every input and output: name, rank, which dims are
dynamic, dtype, format, accumulation precision, supported SMs and platforms,
and who owns the memory (engine binding, per-enqueue workspace, runtime pool,
or prepared weight). Then choose:

| Choose | When | Reason |
|---|---|---|
| Native TensorRT layers | Standard ONNX ops, `trt`-domain ops, or `Net` methods express it within the accuracy and latency budget on every claimed platform | No plugin ABI to version; TensorRT can fuse it; both routes get it from one lowering |
| Extend an existing plugin | An existing creator already owns the semantics; only an attribute, dtype, or dispatch case is missing | A second creator for the same semantics duplicates the whole contract |
| Edge-LLM IPluginV3 plugin | Needs a custom kernel, a packed weight layout, persistent state addressed by explicit indices, or a buffer layout native layers do not preserve (`QkvConcatPlugin`) | Only a plugin fixes dtype, layout, and kernel choice across both builders |
| Runtime-only | Work between engine executions or on host-scheduled state: sampling, embedding and PLE cache I/O, KV and state bookkeeping, weight preparation | Keeps it out of the engine; lives in `cpp/runtime/` with kernels in `cpp/kernels/` |

Refuse a plugin that re-implements a native layer without a measured reason,
and refuse a contract change hidden inside an unrelated change.

## 2. Wire both routes

The file-by-file map for both routes, the worked example, and the weight-layout
alignment points are in `references/wiring-map.md`. Read it before editing.
Invariants that are not obvious from any single file:

- ONNX node `op_type` equals the creator's `getPluginName()`; the domain is
  `trt_edgellm`; attribute names equal creator field names; `plugin_version`,
  when emitted, equals `getPluginVersion()`.
- The direct route resolves creators through `_OPERATION_CREATORS` in
  `tensorrt_edgellm/builder/ops/backend.py`, always with version `"1"` and an empty
  namespace. A version bump breaks that route unless `Net._creator` changes too.
- `Net._operation_field` encodes only INT32 and FLOAT32 fields. Keep creator
  fields to those types so both routes encode them identically.
- `torch.export` drops keyword arguments that equal their defaults, so an
  optional attribute needs the same default in the stub, the translation, and
  the creator.
- When a change reaches both frontends, update both. Model-specific code stays
  in that family's folder under `tensorrt_edgellm/models/` and
  `tensorrt_edgellm/builder/models/`.

## 3. Implement the C++ plugin

`cpp/CMakeLists.txt` globs `plugins/*.cpp` into `NvInfer_edgellm_plugin` and
recursively globs `kernels/*.cpp` plus `kernels/*.cu` into `edgellmKernels`
(`PLUGIN_JIT_*` sources go to `edgellmPluginJit` instead). A `.cu` file under
`cpp/plugins/` is silently not built: put CUDA code in `cpp/kernels/`. Kernel
artifacts and CuTe DSL groups belong to `edge-llm-cutedsl-kernel`.

1. Derive from `IPluginV3`, `IPluginV3OneCore`, `IPluginV3OneBuild`, and
   `IPluginV3OneRuntime`; return each from `getCapabilityInterface`. Register
   an `IPluginCreatorV3One` with `REGISTER_TENSORRT_PLUGIN` in namespace
   `trt_edgellm::plugins`, which is the C++ namespace only: the TensorRT plugin
   namespace stays empty (`""`). Keep name and version in named constants.
2. Both builders create strongly typed networks (`cpp/builder/builderUtils.cpp`,
   `tensorrt_edgellm/builder/core/builder.py`). `getOutputDataTypes` must derive
   every output dtype from input dtypes and attributes, and
   `supportsFormatCombination` must accept exactly the contract.
3. Build output shapes with `IExprBuilder`; require constant dims the kernel
   depends on. Validate min, opt, and max in `configurePlugin`, and revalidate
   in `onShapeChange` and `enqueue`. On violation, log which tensor, dtype, or
   shape is wrong and return `-1`; never throw across the library boundary.
4. `getFieldsToSerialize` writes every attribute plus any build-derived state
   the runtime needs. The creator accepts that set in the runtime phase and
   returns `nullptr` for missing or unknown fields, or, with a log naming the
   missing group, when an optional kernel group is not compiled in.
5. Per-enqueue scratch goes through `getWorkspaceSize`. Persistent state is an
   engine binding or a runtime-owned pool. Weights are bindings prepared by the
   runtime. Do not allocate memory or repack weights in `enqueue`.
6. Launch only on the `stream` passed to `enqueue`; no default stream and no
   synchronous CUDA API. Check every CUDA return value.
7. Reuse `cpp/plugins/utils/pluginUtils.h` (`parsePluginScalarField`,
   `Serializer`, `kDEVICE_ALIGNMENT`) and `rt::Tensor` instead of adding helpers
   or raw owning pointers.
8. An output that aliases an input also needs builder gating, as
   `usesAliasedPluginIO` in `cpp/builder/llmBuilder.cpp` does for
   `DFlashTargetKVCacheUpdatePlugin` (plugin name `DFlashTargetKVCacheUpdate`).

C++ builders and the example apps load the library through
`loadEdgellmPluginLib()` in `cpp/common/trtUtils.h`. Without `--plugin-path`, the direct builder uses the
installed package's plugin and only then `build/lib/libNvInfer_edgellm_plugin.so`,
so a stale installed library shadows a source build: pass `--plugin-path`.

## 4. Tests

Extend the nearest existing test; do not add a harness.

| Contract | Extend |
|---|---|
| Stub semantics, emitted node, attributes, schema | the pattern in `tests/python-unittests/test_ple_pool_export.py` (not itself in the L0 list): eager output vs reference, `onnx.checker` with `check_custom_domain=True` |
| Creator, shape and type negotiation, execution, rejection | `PluginRunner` in `tests/python-unittests/test_plugin_base.py`, as `test_qkv_concat_plugin.py` uses it; `build` serializes and deserializes the engine, so execution also covers the runtime-phase creator; assert rejection with `expect_unsupported=True` |
| Direct lowering: name, input order, attributes | Fake-`Net` recording tests, patterned on `tests/python-unittests/test_experimental_attention_ragged_contract.py` (not in the L0 list) |
| Plugin or kernel internals in C++ | `unittests/cpp/plugins/<plugin>/` or `unittests/cpp/kernels/<dir>/`, registered in an `edgellm_add_unit_test` call in `unittests/CMakeLists.txt` (configure fails on unregistered sources); follow `edge-llm-cpp-test` |

CI runs only tests named in `tests/test_lists/l0_python_ut.yml`, so list an added
Python test there; a GPU plugin test must also match `*plugin*.py` to run in the
device job. Cover every
dtype and rank the contract claims and at least one rejection path that
returns an actionable message. Mutation-test each added assertion as
`edge-llm-cpp-test` and `edge-llm-e2e-test` describe.

## 5. End-to-end evidence

For every claimed route, platform, and precision: export or direct-build a
representative model that enqueues the operation, build the engine, run
inference, and compare deterministically against the PyTorch reference.
Use `edge-llm-convert-model` to build and run it, `edge-llm-e2e-test` for the
model case, and `edge-llm-few-layer-test` when the change adds a layer, kernel,
or quantization recipe. Route accuracy failures to
`edge-llm-accuracy-debugging`.

Not evidence: an ONNX file without an engine; a `PluginRunner` pass alone; a
collection-only run; a skipped platform; an engine built from a stale plugin
library.

## Anti-patterns

| Do not | Reason |
|---|---|
| Add, remove, or reorder plugin inputs when an attribute or existing input suffices | Every input change breaks both translations, the bindings, and all serialized engines |
| Reinterpret a serialized field or input without rebuilding engines | Old engines load and silently compute wrong values |
| Select output dtype from builder precision flags | Networks are strongly typed; the plugin owns its dtypes |
| Allocate, repack, or synchronize in `enqueue` | Hot-path rule in `AGENTS.md`; breaks CUDA graph capture and latency |
| Put creator names or `PluginField` construction in model files | Lowering is private to `tensorrt_edgellm/builder/ops/backend.py` and the translation table |
| Fall back silently when an optional kernel group is missing | The creator must fail with the missing group named; any fallback is explicit and logged |

## Completion evidence

Report the revision; the decision row from section 1 with its reason; files changed per layer; focused test commands with
pass, fail, and skip counts; and, for each route, platform, and precision, the
build and inference commands, GPU and SM, engine artifacts, reference
comparison, and verdict. State plainly what was not run and why.

## Examples

- "Add a fused operation used by the ONNX and checkpoint-direct builders": write the contract, wire both routes, implement the C++ plugin, then run unit and end-to-end tests.
- "Change an existing plugin's tensor layout": update the contract, every consumer, and the tests, then rebuild engines, because old engines load and silently compute wrong values.
