---
name: edge-llm-cutedsl-kernel
license: Apache-2.0
metadata:
  author: TensorRT Edge-LLM Team <noreply@nvidia.com>
  tags: [edge-llm, cutedsl, kernel, artifacts]
description: Add or change a TensorRT Edge-LLM CuTe DSL kernel end to end, from the Python kernel source through the KERNEL_VARIANTS registry in kernelSrcs/build_cutedsl.py, GPU-free AOT artifact generation (native or AArch64 cross build), the metadata-driven CUTE_DSL_*_ENABLED defines in cmake/CuteDsl.cmake, and the C++ runner and SM dispatch, to numerical tests on every claimed SM and prebuilt tarball packaging. Use when adding a CuTe DSL kernel group or variant, changing supported_sms, regenerating cpp/kernels/cuteDSLArtifact, debugging a missing CuTe DSL define, symbol, or module-load failure, or preparing cutedsl tarballs. Do not use it to create or register a TensorRT plugin (use edge-llm-plugin-extension), for GTest authoring rules (use edge-llm-cpp-test), or for model deployment tests (use edge-llm-e2e-test).
---

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Edge-LLM CuTe DSL kernels

A CuTe DSL kernel reaches the runtime through a chain of files that must agree
on variant names, supported SMs, and the exported ABI. CMake never runs Python.
It links whatever is in `cpp/kernels/cuteDSLArtifact/<arch>/<tag>/`, and that
can be an extracted prebuilt that does not contain your change.

Read these before proposing an edit. They are canonical, so do not restate them:

- `kernelSrcs/README.md`: dependencies, Docker and venv generation, script
  flags, CMake integration, cross-compilation, and the tarball matrix.
- `kernelSrcs/<group>/README.md`, when present: variants, ABI, and standalone tests.
- `kernelSrcs/build_cutedsl.py`: `KernelVariant`, `KERNEL_VARIANTS`, `_compile_command`.
- `cmake/CuteDsl.cmake`: `cute_dsl_setup` and every group or variant define.
- `CODING_GUIDELINES.md` and the CUDA rules in `AGENTS.md`, for runner code.

The cross-file wiring map, with the edit order for an added variant or group, is in
[references/wiring-map.md](references/wiring-map.md). Read it before your first
edit.

Before designing a kernel, read one recent merged kernel or variant addition
(`git log --oneline --no-merges -n 10 origin/main -- kernelSrcs cmake/CuteDsl.cmake`)
instead of copying the oldest similar file. Repository history is authoritative
for the current registration and test shape; this skill is the invariant checklist.

## Instructions

Follow steps 1 to 5 in order: pin the target and baseline, develop the kernel, regenerate artifacts and reconfigure, test against the regenerated artifact, and package only after validation. Report with Completion evidence.

## 1. Pin the target

Record these values first. Each one selects a different artifact.

| Input | Where it is consumed |
|---|---|
| Target SM or SM set | `--gpu_arch sm_NN[,sm_MM]` sets the artifact tag `sm_NN` or `sm_NN_sm_MM` |
| CUDA major and compiler flavor | `--cuda-version`; one venv per `nvidia-cutlass-dsl[cu12]` or `[cu13]`; tarball suffix `_cudaNN` |
| Target CPU arch | `--arch x86_64` or `aarch64`; an arch that differs from the host is a cross build |
| Runtime platform | CMake `EMBEDDED_TARGET` or `CUTE_DSL_ARTIFACT_TAG`; see `docs/source/user_guide/getting_started/support-matrix.md` |
| Validation device | A GPU of every SM you claim. Without one, report the run as blocked. |

Claim only the SMs you can validate. `supported_sms` is the build whitelist,
and the group README's variant table is the user-facing claim.

## 2. Develop the kernel

| Loop | What it proves | What it does not prove |
|---|---|---|
| JIT standalone script on a GPU | Kernel math on that GPU's SM | The AOT trace, the exported ABI, or other SMs |
| AOT export on a CPU host, through `build_cutedsl.py` | The variant traces and exports for `--gpu_arch` with storage-free placeholders | Any numerical result |
| C++ test linked against the fresh artifact | ABI, runner dispatch, and numerics on the device's SM | Model integration |

- Keep JIT and AOT on the same kernel class; see the script patterns in the wiring map.
- AOT traces use `kernelSrcs/cutedsl_utils/aot_placeholders.py`. Each trace
  dimension is either baked into the cubin or marked dynamic. Name each baked
  value as a constant and enforce the same limit in the runner's admission check.
- Never query the device in a kernel script; the build host's GPU is not the
  target. A script that needs the SM reads `--target_sm`; its registry entry
  sets `wants_target_sm=True`.
- Selection is per group, not per variant. When the consumer is the attention
  stack, follow the group README and include its required FMHA group in a fresh
  artifact. Do not add `fmha` automatically for an unrelated standalone group.

## 3. Generate fresh artifacts and configure

1. Generate with the Docker or venv workflow in `kernelSrcs/README.md`. Use
   `--clean` for a first build, after you add, rename, or remove a variant, after
   you change `supported_sms`, and before packaging. A single-SM build without
   `--clean` merges into `metadata.json` (union of variant lists), so stale
   members survive. Multi-SM and Windows artifacts refuse to rebuild into an existing artifact without `--clean`.
2. Cross build: `--arch aarch64` on an x86_64 host emits target host objects and
   embeds the target-arch runtime shim. This proves the build only; run the
   tests on the target device. A native build on the device skips the cross step.
3. Reconfigure CMake. In the configure log, confirm that it did not print
   `CuTe DSL: extracting prebuilt` and that it printed the `CuTe DSL: ... <DEFINE> set`
   status line for each variant-specific define you added.
4. Inspect `metadata.json`: `variants` lists your object stems (equal to the variant names unless the script suffixes the export), `gpu_arch` or
   `gpu_archs` matches the target, and `build_date` is later than your last source edit.

A prebuilt tarball under `kernelSrcs/cuteDSLPrebuilt/`, or an artifact
directory extracted from one, is never evidence for modified kernel source.

## 4. Test

Extend the existing tests next to the group. Do not add a test framework.

| Layer | Existing home | Requirement |
|---|---|---|
| Registry and command contract (CPU) | `tests/python-unittests/test_build_cutedsl_helpers.py` (in the L0 list; `test_cutedsl_aot.py` is not, and needs `cutlass`) | Variant names, the SM whitelist, and rejected SMs, following the per-group tests already there |
| Kernel numerics (GPU) | The group's GTest under `unittests/`, for example `unittests/cpp/kernels/moe/f16MoeCuteDslTests.cu` | A human-readable reference, every exported variant, each supported dtype, boundary and residual shapes, and empty or zero-length inputs where the ABI allows them |
| Plugin or runtime integration | The consumer's test, for example `tests/python-unittests/test_fp16_moe_plugin.py` | Required whenever the kernel is not standalone; ownership moves to `edge-llm-plugin-extension` |
| Model path | `edge-llm-e2e-test`, plus `edge-llm-few-layer-test` for an added kernel | Only when the change affects model output |

- Follow `edge-llm-cpp-test` for GTest structure and mutation checks.
- Run the numeric test on each claimed SM. Name failures by variant
  (`<variant>` at `sm_NN`, with the dtype and shape) rather than by group.
- Record the workload (shapes, dtype, batch), the GPU and SM, and the CUDA
  flavor with every performance claim. Do not write numbers into this skill or
  into a README.

## 5. Package after validation

Package only after sections 3 and 4 pass on every claimed SM.

1. Update the group README variant table and its verified SMs. For an added group,
   also update the group lists in `kernelSrcs/README.md`.
2. Add or change a matrix target in `kernelSrcs/Dockerfile.cutedsl`
   (`CUTE_DSL_MATRIX`) and in `CUTE_DSL_TARGETS` in `kernelSrcs/build_cutedsl_tarballs.sh` (its per-CUDA-major default, used when `CUTE_DSL_MATRIX` is unset).
   Mirror the target in the tarball list in `kernelSrcs/README.md` and in the
   matrix test in `tests/python-unittests/test_build_cutedsl_helpers.py`.
3. Produce tarballs and `.sha256` files with the Docker builder, which validates
   metadata, ELF arch, layout, and checksum. Tarballs are git-ignored; never
   commit them.
4. Borrowed kernel code keeps its upstream license header and adds an
   `Adapted from <project> <version> <path>` line, as
   `kernelSrcs/f16_moe_cutedsl/grouped_gemm_blackwell.py` does. The group README
   states the origin. `CONTRIBUTING.md` requires DCO sign-off that certifies the
   right to submit, and `.pre-commit-config.yaml` excludes `kernelSrcs/`, so
   no hook adds or checks a header there.

## Anti-patterns

| Do not | Reason |
|---|---|
| Gate a variant with a family-wide define such as `CUTE_DSL_FMHA_BLACKWELL_ENABLED` | That define keys on `fmha_d64`, which only SM100/101/103/110 build. Code under it silently disappears on SM120/121. Key a define on your own variant name. |
| Dispatch on SM family alone | A single-SM artifact holds one SM's cubins. Also check the artifact SM, as `isArtifactSm` does in `cuteDslF16MoeRunner.cpp`. |
| Rename the export in the script or ignore `--file_name`/`--function_prefix` | CMake detection, the metadata `variants`, and multi-SM `__arch<NN>` symbol dispatch all key on the emitted object stem. Suffix it only deliberately, as `gdn_decode_mtp` does, and use the stem in CMake and C++. |
| Widen `supported_sms` without a run on that SM | The README and plugin admission advertise a kernel that nobody executed. |
| Validate with a JIT run or a prebuilt only | Neither exercises the AOT artifact your source produces. |
| Allocate, use the default stream, or skip error checks in `run()` | Follow `CODING_GUIDELINES.md` and the CUDA rules in `AGENTS.md`: workspace from the caller, an explicit `cudaStream_t`, and checked CUDA calls. |
| Add an environment variable to pick a variant | Selection belongs in the registry, the defines, and runner arguments. |

## Completion evidence

Report:

- The revision and the exact `build_cutedsl.py` and CMake commands.
- The `metadata.json` fields `gpu_arch`/`gpu_archs`, `variants`,
  `cutlass_dsl_version`, `cuda_version`, and `build_date`.
- The configure lines that show each define.
- Test commands with pass, fail, and skip counts per SM, dtype, and variant.
- The GPU used for each SM, and whether the artifact was a native or cross build.

Name each claimed SM that was not executed and the reason. Prebuilt artifacts,
JIT-only runs, export-only runs, and collection-only runs are not evidence for
changed kernel source.

## Examples

- "Change a kernel tile size and verify it": develop the change, regenerate the artifact for the target SM, rebuild, and run the kernel tests plus an end-to-end check.
- "Add a new variant for another SM": register the variant and every dependent define, regenerate, and show the runtime links the new artifact.
