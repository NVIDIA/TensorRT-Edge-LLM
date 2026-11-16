---
name: edge-llm-version-migration
license: Apache-2.0
metadata:
  author: TensorRT Edge-LLM Team <noreply@nvidia.com>
  tags: [edge-llm, migration, upgrade, backport]
description: Migrate an existing TensorRT Edge-LLM deployment, integration, or downstream patch between releases, branches, TensorRT/CUDA/JetPack/DriveOS stacks, or checkpoint-family versions; also backport fixes to a release branch. Use when commands, package extras, CMake options, plugins, ONNX, engines, server requests, or downstream code must be mapped old-to-new. Inventories both revisions, generates a source-backed surface diff, rebuilds derived artifacts, and proves the migrated workload. Do not use for a first install (edge-llm-install), a supported-model conversion (edge-llm-convert-model), or new architecture support (edge-llm-add-model).
---

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Migrate between versions

A migration is a tested old-to-new mapping for a real workload. It is not a
changelog summary and not a blind dependency upgrade.

## Instructions

Capture the old baseline, compare source-backed surfaces with the helper script, write the old-to-new mapping before editing, then apply and validate incrementally in Phases 1 to 4. Report with Completion.

## Inputs

Require:

- exact from/to revisions or package versions;
- from/to platform stack (OS/SDK, CUDA, TensorRT, Python, GPU/SM);
- commands, scripts, imports, CMake options, plugins, model checkpoints, and
  request formats the deployment uses;
- one deterministic workload that passes before migration;
- whether this is an upgrade, downgrade, downstream port, or backport.

Preserve the old environment and artifacts. Work in a fresh checkout,
environment, build directory, cache, ONNX directory, and engine directory.

## Phase 1: capture the old baseline

Run the user's workload before editing. Record revisions, stack, commands,
input, output, logs, and accuracy/performance acceptance criteria. If the old
workload cannot run, mark the baseline unavailable and agree on a provider
reference; do not manufacture expected output.

## Phase 2: compare source-backed surfaces

Fetch both refs, then generate the first-pass inventory:

```bash
python3 skills/edge-llm-version-migration/scripts/compare_surfaces.py \
  <from-ref> <to-ref> --output <work-dir>/surface-diff.json
```

It needs Python 3.11 or newer (3.10 works with `tomli` installed). It reports
`requires-python`, dependencies, console scripts, extras, CMake options and cache
variables (every `CMakeLists.txt` and `*.cmake`), and which user-facing files
changed, including the `trtedge`, export, builder, server, and chat CLI modules
and `examples/cpp`. It lists changed files but does not diff individual flags,
and it does not see `-D` variables that CMake only reads (for example
`TRT_PACKAGE_DIR`); read those CLI parsers and installation.md at both refs. It
does not decide semantic compatibility.

Read at both refs with `git show <ref>:<path>`:

- `CHANGELOG.md`, support matrix, installation, limitations, quick start;
- `pyproject.toml` scripts, extras, and dependency pins;
- CLI parsers used by the deployment;
- relevant builders, plugins, runtime config, and server routes;
- model support and model-specific example page.

Also inspect the actual commits that introduced each breaking change:

```bash
git log --oneline <from-ref>..<to-ref> -- <affected-path>
git show <commit> -- <affected-path>
```

Release notes tell what changed; source and the introducing tests tell how it
works now.

## Phase 3: write the mapping before editing

For each inventoried dependency, produce:

| Surface | Old | New | Proving source/test | Required edit |
|---|---|---|---|---|
| command/flag/import/config/artifact | ... | ... | `<to-ref>:<path>` | ... |

An item without a proving source is unresolved. Probe the target command's
`--help` or stop. Do not opportunistically adopt unrelated features.

Treat these as rebuild boundaries:

- Edge-LLM, TensorRT, CUDA, platform SDK, target SM, exporter/plugin contract;
- model checkpoint or quantization recipe changes;
- component profile or engine binding changes.

Unless target-branch documentation explicitly guarantees compatibility,
re-export ONNX and rebuild all engines. Keep exporter, builders, plugin library,
and runtime from one revision.

## Phase 4: apply and validate incrementally

1. Recreate the target environment with `edge-llm-install`.
2. Update one mapping-table row at a time and run its focused test.
3. Re-convert and run the real checkpoint with `edge-llm-convert-model`, or use
   a fresh server cache and `edge-llm-serve`.
4. Compare deterministic output with the old baseline/provider reference.
5. Run the affected end-to-end and platform coverage.
6. Only then remove deprecated compatibility code or old artifacts, if asked.

Divergence unexplained by the target changelog is an accuracy issue:
`edge-llm-few-layer-test`, then `edge-llm-accuracy-debugging`.

## Special cases

### Platform-stack-only migration

Confirm the target row in the target revision's support matrix. Reconfigure
from an empty build directory, regenerate/select CuTe DSL artifacts for the
target SM/CUDA, and rebuild engines. “The libraries load” is not enough.

### Checkpoint family revision

Compare provider configs and tensor inventories with
`skills/edge-llm-add-model/scripts/inspect_checkpoint.py`. If architecture or tensor
contracts changed, this is model onboarding, not merely artifact migration.

### Release backport

Read [references/backport.md](references/backport.md). Check patch equivalence,
dependencies, and conflicts before touching the worktree; cherry-pick with
`-x -s`, run pre-commit and amend; adapt to release-branch APIs; prove the bug and fix on that branch.

## Completion

Report from/to revisions and stacks, old baseline, generated surface diff,
completed mapping table, edits, fresh artifact locations, exact validation
commands, output/reference comparison, and unresolved rows. A clean rebase,
successful install, passing `--help`, reused engine, or unrun backport is not a
successful migration.

## Available scripts

| Script | Purpose | Run |
|---|---|---|
| `scripts/compare_surfaces.py` | List dependencies, console scripts and extras, CMake options and cache variables, and changed user-facing paths between two refs; it does not judge compatibility | `python3 skills/edge-llm-version-migration/scripts/compare_surfaces.py <from-ref> <to-ref> --output <work-dir>/surface-diff.json` |

Run scripts with the shell tool, with Python 3.11 or newer (3.10 with `tomli`), and keep the
JSON output as evidence.

## Examples

- "Move my deployment from the previous release to current main": capture a working baseline, run `compare_surfaces.py` for the two refs, map each changed flag and extra, apply, and re-run the workload.
- "Backport this fix to a release branch": use `references/backport.md`, cherry-pick with a conflict inventory, and validate on the release branch.
