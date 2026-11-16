<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# CuTe DSL wiring map

Read this before your first edit in a CuTe DSL kernel change. It lists the
files that must agree with each other, in the order to edit them.

## Name flow

One string, the variant `name`, connects every layer:

```text
KERNEL_VARIANTS[name]            kernelSrcs/build_cutedsl.py
  -> script args                 --file_name <name> --function_prefix <name>
                                 (appended last by _compile_command)
  -> <stem>.o, <stem>.h          export_to_c output; the stem is <name> unless the
                                 script suffixes it (`gdn_decode_mtp` emits
                                 `gdn_decode_mtp_cache`); multi-SM adds __arch<NN>
  -> metadata.json "variants"    object stems, canonical (marker removed)
  -> cutedsl_<group>_all.h       per-group umbrella header
  -> CUTE_DSL_<GROUP>_ENABLED    generic, from metadata "groups"
  -> optional per-variant define list(FIND _variants "<stem>") in cmake/CuteDsl.cmake
  -> C++ symbols                 <stem>_Kernel_Module_t, _Kernel_Module_Load,
                                 _Kernel_Module_Unload, cute_dsl_<stem>_wrapper
```

Metadata `variants`, the CMake `list(FIND ...)`, and the C++ symbols all key on
the emitted object stem. It equals the registry name unless the script suffixes
the export name. The build passes `--file_name`/`--function_prefix` last, so they
override any in `script_args`.

`cute_dsl_setup` applies defines and includes to the library targets in
`cpp/CMakeLists.txt` and to every `edgellm_add_unit_test` group.

## Worked example: `f16_moe`

Small, three SM families, every layer, and borrowed-code provenance.

| Layer | File |
|---|---|
| Kernel source | `kernelSrcs/f16_moe_cutedsl/grouped_gemm_{ampere,blackwell,blackwell_geforce}.py` |
| AOT exporter | `kernelSrcs/f16_moe_cutedsl/export_grouped_gemm.py`, plus `export_common.py` |
| Registry | Three `KernelVariant` entries named `f16_moe_*_grouped_fp16` in `kernelSrcs/build_cutedsl.py`, one per SM family, each with its own `supported_sms` |
| Group README | `kernelSrcs/f16_moe_cutedsl/README.md` (variant table, device ABI, plugin contract) |
| CMake defines | `cmake/CuteDsl.cmake`: `CUTE_DSL_F16_MOE_{AMPERE,BLACKWELL,BLACKWELL_GEFORCE}_ENABLED` per variant; `CUTE_DSL_F16_MOE_ARTIFACT_SM` or `CUTE_DSL_F16_MOE_MULTI_ARCH_ENABLED` per artifact |
| Runner | `cpp/kernels/moe/f16_cutedsl/cuteDslF16MoeRunner.{h,cpp}`: `canImplement`, `selectVariant`, `ensureKernelModules`, `isArtifactSm` |
| Module loading | `cpp/kernels/cuteDslModuleLoader.h`: `detail::LazyKernelModule` and `detail::ensureModuleLoaded` |
| Plugin consumer | `cpp/plugins/fp16MoePlugin/fp16MoePlugin.cpp` |
| C++ numeric test | `unittests/cpp/kernels/moe/f16MoeCuteDslTests.cu` (target `unitTestKernelsMoe`) |
| Plugin test | `tests/python-unittests/test_fp16_moe_plugin.py` |
| Registry test | `test_build_allows_f16_moe_for_foreign_target_sm` in `tests/python-unittests/test_build_cutedsl_helpers.py` |

Two script patterns exist. `f16_moe` uses an export-only script
(`export_grouped_gemm.py`) that always exports; its registry entries pass only
`--family`. `kernelSrcs/int4_fp16_gemm_cutedsl/` keeps a JIT correctness path in
the same script and exports only when the registry passes `--export_only` in
`script_args`.

## Edit order: a variant in an existing group

1. Kernel script. Accept `--output_dir`, `--file_name`, and `--function_prefix`
   and emit one module named by them. If the script also has a JIT path, gate
   export behind `--export_only`. Trace with `cutedsl_utils.aot_placeholders`,
   never with device tensors. Read `--target_sm` only when codegen depends on
   the SM; the registry entry then sets `wants_target_sm=True`.
2. `KERNEL_VARIANTS` entry: a unique `name`, the existing `group`, `script`
   relative to `kernelSrcs/`, `script_args`, and `supported_sms`.
3. CMake define, only if the C++ code must know about this variant separately
   from its group. That is the case when its `supported_sms` differ from its
   siblings, or when runner code references its symbols. Follow the GDN block
   and require the group as well:
   `list(FIND _variants "<stem>" ...)` plus `AND "<group>" IN_LIST _active_groups`
   (the GDN block does this; the `f16_moe` define block does not check the
   group), with a `STATUS` message so the configure log shows the define.
4. Runner: declare `LazyKernelModule<<stem>_Kernel_Module_t>`, load it through
   `ensureModuleLoaded<..._Load, ..._Unload>`, and call `cute_dsl_<stem>_wrapper`. Guard
   the declaration, the selection, and the admission check with the define
   from step 3.
5. Tests: add the variant to the group's C++ parameter table and to its
   registry test.
6. After validation, add the variant and its verified SMs to the group README.

## Additional edits: a group

- `kernelSrcs/build_cutedsl.py`: add the group to the module docstring, the
  `KernelVariant.group` docstring, the registry comment block, and the
  `--kernels` help text. `_ALL_GROUPS` derives from the registry.
- `cmake/CuteDsl.cmake` and `cpp/CMakeLists.txt`: add the group to the
  `ENABLE_CUTE_DSL` comment lists and cache help string, and to the per-group
  define comment block at the top of `cmake/CuteDsl.cmake` (the defines come from a generic loop). `CUTE_DSL_<GROUP>_ENABLED` needs no edit. If
  the runner needs the artifact SM, add the group to the condition that lists
  `f16_moe`, `w2a16_moe`, `lora`, `layernorm`, and `rmsnorm`. Then emit
  `CUTE_DSL_<GROUP>_ARTIFACT_SM` or the multi-arch defines the way `f16_moe` does.
- Runner header: wrap everything in `#if defined(CUTE_DSL_<GROUP>_ENABLED)`.
  Redefine `CUTE_DSL_CUDA_ERROR_CHECK` to `detail::recordCuteDslCudaError`
  around `#include "cutedsl_<group>_all.h"`, as the existing runners do. Sources
  under `cpp/kernels/` are collected by `GLOB_RECURSE`, so no list edit is
  needed.
- Consumers: build with the group absent. Admission returns false and the
  failure names the missing group; never fall back silently. A fallback must be
  a documented, explicitly selected path. Plugin rules: `edge-llm-plugin-extension`.
- Tests: a source in an already registered `unittests/` directory is picked up
  by the glob. An unregistered directory needs an `edgellm_add_unit_test` entry;
  otherwise configure fails with "Unit test sources are not in any group".
- Docs: add `kernelSrcs/<group>/README.md`, list it under "Group-Specific
  Details", and add the group name to the `--kernels` row of `kernelSrcs/README.md`.

## Multi-SM artifacts

`--gpu_arch sm_110,sm_120` compiles each variant per SM with an `__arch<NN>`
suffix, plus canonical headers and a host dispatcher keyed on the current device.
Generation fails if the header ABI differs across SMs. Runners keyed on the
artifact SM must handle the multi-arch define as well as the single-SM one.

## Symptom lookup

| Symptom | Likely cause |
|---|---|
| Configure: `ENABLE_CUTE_DSL='...' matched no groups` | The group is absent from `metadata.json`; regenerate with that group |
| Define missing, so code is compiled out | The variant name in the CMake `list(FIND ...)` differs from the emitted object stem (usually the registry name), or the target SM is outside `supported_sms` |
| Undefined `cute_dsl_<name>_wrapper` at link time | Stale artifact, or header and archive come from different builds; regenerate with `--clean` |
| Module load fails at runtime | The artifact SM differs from the device SM; check `isArtifactSm`-style admission |
| Results are correct in JIT but wrong in C++ | A baked trace dimension, alignment, or stride assumption differs between the AOT placeholders and the runner |
| `metadata.json` lists a removed variant | An incremental merge kept it; regenerate with `--clean` |
