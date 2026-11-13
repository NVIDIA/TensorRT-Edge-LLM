---
name: edge-llm-install
license: Apache-2.0
metadata:
  author: TensorRT Edge-LLM Team <noreply@nvidia.com>
  tags: [edge-llm, install, setup, wheel, build]
description: Select, install, verify, update, or remove TensorRT Edge-LLM, and install this repository's agent skills into Claude Code or Codex. Inspects revision, CPU architecture, GPU or SoC, JetPack/DriveOS, CUDA, TensorRT, and Python, rejects unsupported combinations before changing anything, then picks exactly one route - published wheel, local wheel, C++ source for the ONNX workflow, or Python source for the experimental checkpoint-direct builder and server - and proves it with a per-route gate. Use when asked to install, set up, build from source, cross-compile, pip install tensorrt-edgellm, upgrade the installed package or build in place, uninstall, fix a CMake or wheel-load failure, or install, update, or remove the edge-llm skills. Do not use for running a model (use edge-llm-serve or edge-llm-convert-model), writing or selecting tests (edge-llm-e2e-test, edge-llm-cpp-test), accuracy debugging (edge-llm-few-layer-test, edge-llm-accuracy-debugging), or release-to-release migration (edge-llm-version-migration).
---

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Install TensorRT Edge-LLM

Complete the selected installation and its gate; do not stop after recommending
a route. `docs/source/user_guide/getting_started/installation.md` owns every
install command and `support-matrix.md` owns platforms and the Wheel Packaging
Matrix. Read both from the revision being installed and run the documented
commands for the chosen route verbatim; this skill adds only the decisions and
traps around them. Also see `AGENTS.md` "Common Commands", `packaging/README.md`,
and `kernelSrcs/README.md`.

Installing this repository's skills into an agent host is in
`references/skills-onboarding.md`. Read it only for that task.

The repository's `main` branch and a released wheel can differ. For a
wheel, read the matching release tag's docs; for a checkout, its branch-local docs.

## Instructions

Inspect the host, pick exactly one route, reject unsupported combinations before changing anything, run that route's documented steps, and finish with its verification gate. Report the commands, versions, and gate output.

## Rules

- Inspect first; reject an unsupported combination by its missing support-matrix row.
- Pick one route; never switch between ONNX/C++ and checkpoint-direct unasked.
- Preserve a dirty worktree: no `git pull`, `checkout`, `stash`, `clean`, or `reset`.
- Ask before `sudo`, `apt`, system `pip`, `docker pull`, CUDA, TensorRT, or driver
  installs, `cmake --install` outside the workspace, and writes to shell profiles
  or agent configuration.
- Never print or store tokens; let the user run `hf auth login`.

## Inspect

Collect: `git rev-parse HEAD` and `git status --porcelain`; `uname -m` and
`/etc/os-release`; the SoC (`/proc/device-tree/model` or DMI product name) and
JetPack/DriveOS release files; GPU `compute_cap` from `nvidia-smi`; CUDA
(`nvcc --version`), TensorRT, and Python/glibc versions; and the workflow the
user needs (ONNX export plus C++ inference, checkpoint-direct build, OpenAI
server, or kernel/plugin development).

Look the platform up in `support-matrix.md` ("Platforms" and "Wheel Packaging
Matrix", including its Python and glibc requirements). Stop when there is no row.
If several visible GPUs have different SMs, the wheel loader rejects them: use
`CUDA_VISIBLE_DEVICES=<GPU-UUID>`; a source build picks the first GPU
`nvidia-smi` reports, so pass `-DCMAKE_CUDA_ARCHITECTURES` explicitly.

## Choose a route

| Need | Route (details in installation.md) |
|---|---|
| ONNX export plus C++ inference (maintained) | C++ source build; export host also needs the export extras |
| Checkpoint-direct build or server (experimental) | Published wheel when the matrix has a row, otherwise Python source |
| Wheel of the current checkout | Local wheel build |
| Kernel or plugin development | C++ source with `-DBUILD_UNIT_TESTS=ON` |

## Traps that cost time

- **Published wheel:** use a fresh venv as installation.md prescribes, never
  `pip install -e .` into it, and run the gate outside any checkout.
- **Local wheel:** follow "Build a local wheel from source" in full, including the
  toolchain requirements and the CuTe DSL archive step. It refuses tracked
  changes; `--allow-dirty-source` is development-only, so report it.
- **C++ source:** use the platform's configure command from installation.md Step 4
  as written. Edge devices and cross builds need `-DCMAKE_TOOLCHAIN_FILE` and
  `-DEMBEDDED_TARGET`; do not rely on auto-detection when the doc shows them.
  Set `-DCMAKE_BUILD_TYPE=Release`; the repository has no default.
- **Python bindings are needed only for Python inference or `trtedge serve`.**
  C++ inference, ONNX export, and checkpoint-direct builds do not need them.
  When enabled they use nanobind: install the pins and pass the `nanobind_DIR`
  argument exactly as "Optional Python frontend" shows, and pass `-DTRT_PACKAGE_DIR`
  (`/usr` for system TensorRT), which the nanobind CMake requires.
- **Python source:** use `build/` (or set `BUILD_DIR`), install editable, and run
  the source builder from the checkout root or pass `--plugin-path`.
- **CuTe DSL is mandatory** except on QNX. If no prebuilt artifact matches,
  configure prints the `kernelSrcs/build_cutedsl.py` command; run it in its own
  venv per `kernelSrcs/README.md`. `ENABLE_CUTE_DSL` is a cache variable: after
  an `OFF` run, reconfigure with `-DENABLE_CUTE_DSL=ALL` or a fresh build dir.
- A configure that cached a wrong `EMBEDDED_TARGET`, CUDA, or TensorRT needs a fresh
  `build-<target>` directory.

## Verify

Package creation alone is not success. Run the gate that installation.md gives
for the route: for a wheel, `runtime.load()` and `trtedge --version`/`--help`
outside any checkout (the minimal base wheel uses
`examples/python/installed_wheel_build_and_infer.py` as `python-runtime.md`
describes); for C++ source, `bin/llm_build --help`, `bin/llm_inference --help`,
and `lib/libNvInfer_edgellm_plugin.so`; for a cross build, `file` reports
AArch64 and the `--help` gate passes on the target; for the Python source route,
also `build/lib/_edgellm_runtime*.so` and `trtedge --version` showing the
checkout revision.

`--help` proves installation, not a model. Use `edge-llm-serve` for an HTTP
smoke test or `edge-llm-convert-model` for engine build and C++ inference.

## Update and remove

Wheel: upgrade the pinned version in the same venv and rerun the gate. Source:
only on a clean tree and with consent, update the revision and submodules,
reconfigure, rebuild, rerun the gate; moving between releases is
`edge-llm-version-migration`. Remove only what this task created (the venv, the
build directory, install prefixes; `pip uninstall tensorrt-edgellm` in its venv).

## Failures

Start from installation.md's troubleshooting section. Frequent ones:
`TensorRT was not found in the system paths` (pass the SDK root as
`TRT_PACKAGE_DIR`), a missing CuTe DSL artifact (run the command configure
prints), `runtime.load()` rejecting the payload (the stack is not a wheel row),
and `Could not import _edgellm_runtime` (rebuild `build/` with
`BUILD_PYTHON_BINDINGS=ON` and install editable).

## Report

Revision and dirty state, platform facts, the route and why, commands run, artifact
locations, gate output, and anything skipped or blocked.

## Examples

- "Install the published wheel with the runtime extra on this x86_64 GPU host": inspect, find the matrix row, install into a fresh venv, run the wheel gate, report versions.
- "Build Edge-LLM from source for the ONNX workflow": inspect CUDA, TensorRT, and SM, use the C++ source route, configure and build, run the build gate.
- "Install these agent skills for Claude Code outside a clone": read `references/skills-onboarding.md`, choose the user-level route, and link each skill with the user's approval.
