---
name: edge-llm-e2e-test
license: Apache-2.0
metadata:
  author: TensorRT Edge-LLM Team <noreply@nvidia.com>
description: Write, select, run, and verify TensorRT Edge-LLM end-to-end model tests. Use when adding or changing a model case, running one model deployment instead of an entire CI job, interpreting model accuracy or baseline failures, or selecting sufficient tests for Python, C++, CUDA, CuTe DSL, plugin, export, builder, runtime, or server changes.
---

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Edge-LLM end-to-end tests

Model deployments are declared in YAML and expanded into the existing pytest
pipelines. Before changing or running a case, read these canonical documents:

- `tests/model_configs/README.md`: case schema, authoring rules, commands,
  baseline workflow, and custom MR cases.
- `tests/README.md`: environment variables, dependencies, and build setup.

Do not duplicate those definitions in another document. Update the canonical
README when the workflow changes.

## Workflow

1. Identify the behavior and the narrowest test that can fail for it.
2. For model execution, add or update one deployment case under
   `tests/model_configs/l0/<model-family>/<base-model>.yml`.
3. Collect the owning priority and inspect the generated node IDs and phases.
4. Run only the case. Include the project-build node after C++, CUDA, or build
   system changes.
5. Verify output quality, required metrics, and baseline comparison separately.
6. Run the surrounding unit suite and pre-commit checks for changed source.
7. Report the command, hardware, phases, metrics, baseline, skips, and artifacts.

A collection-only run proves schema and routing, not execution. An export-only
run proves neither engine build nor inference. A hardware skip is not a pass.

## Authoring rules

- Use one file per base model. A case describes a checkpoint, frontend, target
  platform, engine limits, and one or more inference scenarios.
- Use a full Hugging Face `organization/model` ID or an absolute checkpoint
  path. Do not invent a path or infer a repository from a display name.
- Select `mode` from the implementation being tested. Cover both ONNX and
  checkpoint-direct paths when a change reaches both.
- Match `requirements` to the checkpoint and platform. A case routed to
  unsupported hardware has not validated the model.
- Reuse a registered input from `tests/test_cases/` unless the change also adds
  the corresponding metric implementation.
- Declare every accuracy and performance metric that constitutes success.
- Keep numeric references in the baseline CSV, not in model YAML.

Baseline comparison is required by default. Use `BASELINE_REQUIRED=0` only to
make a first measurement or seed a trusted golden. Such a run does not prove
regression parity. Follow the two-step seeding and comparison procedure in
`tests/model_configs/README.md`.

## Selecting coverage

Start with the nearest contract test, then add each execution layer affected by
the change. When a change reaches several precisions, drafts, frontends, or
platforms, list the combinations with `edge-llm-feature-combinations` first,
then choose evidence for each one here.

| Change | Inspect | Minimum evidence |
|---|---|---|
| Docs or metadata | `.pre-commit-config.yaml` | Pre-commit on changed files |
| Pure Python behavior | neighboring `tests/python-unittests/` | Focused pytest plus owning suite |
| Model schema or routing | `tests/defs/model_test_config.py` | Collection of every affected priority and one executing case |
| Metric or baseline behavior | `tests/defs/utils/command_execution.py` | `test_declared_metrics.py` and one model case |
| C++ behavior | `unittests/CMakeLists.txt` and owning directory | Registered GTest target and focused filter |
| CUDA or CuTe DSL kernel | owning kernel tests and CMake registration | Contract test and device execution on the required SM |
| TensorRT plugin | plugin source, registration, and tests | Plugin unit test and a model path that enqueues it |
| ONNX export | exporter tests and `test_checkpoint_export.py` | Unit contract, export, engine build, and inference |
| Checkpoint-direct builder | builder tests and `test_builder_pipeline.py` | Unit contract and a `direct_builder` case |
| OpenAI-compatible server | server unit tests and `test_server_pipeline.py` | Protocol/runtime test; server model case when model execution changes |

Do not invent a test target from a source filename. Inspect the owning CMake or
pytest registration first.

## Test quality

Put Python tests under `tests/python-unittests/`; mirror C++ and CUDA ownership
under `unittests/` and register the source in the owning target. Give each new
test a concise docstring that states the behavioral contract rather than
narrating implementation details.

Mutation-test new assertions: make one plausible temporary production change,
confirm the focused test fails for the intended reason, then revert the
mutation. Clear stale Python bytecode when necessary. Never leave the mutation
in the diff. A detected mutation shows sensitivity; it is insufficient on its
own to justify keeping a test.

Write a test only when it gives evidence about a behavior that users or
downstream jobs depend on. A test that guards a choice the project can change
at will costs more to maintain than it protects.

Keep:

- A real input-to-output decision, parse, metric, or pass/fail oracle, with a
  failing and a passing case.
- The production code path run end to end with a transport-level fake, such as
  `httpx.MockTransport` or a temporary file.
- A real effect: files removed on disk, a child process that ran, a failure
  that reaches the exit status or the emitted report.
- A regression for a fixed defect that fails on the old code.
- Safety behavior: refusing unsafe mounts or symlink targets, keeping another
  pipeline's output, cleanup scope and retention.

Remove:

- Configuration-validation tables and fixed policy values: thresholds,
  timeouts, retry counts and delays, quotas, chmod tolerance, selected
  malformed-input checks.
- Exact diagnostic text, private fields, call or command order, argument or
  path forwarding, copied environment values.
- Mocks of processes, signals, remote commands, or cleanup that stop short of
  showing the real effect.
- Text scans of YAML, logs, manifests, or file layout that stand in for running
  the code.
- Coverage that another suite or a CI step already provides.
- Cases that no plausible mutation can distinguish from the correct code.

For a threshold, assert one clear pass and one clear fail away from the
boundary. Match an error with `match=` only when the message is part of the
contract. Parametrize only when each case exercises a different decision.

## Failure interpretation

| Failure | Action |
|---|---|
| Case is absent or on the wrong job | Fix model YAML routing before using GPU time |
| ONNX mode has no export lane | Select a platform with an ONNX producer |
| Required metric was not produced | Check the input-file stem and metric registration |
| Output score is below its absolute threshold | Debug model output; do not widen the threshold |
| Baseline file or entry is missing | Correct the baseline or explicitly opt out for a seeding run |
| Metric regressed beyond tolerance | Fix or justify the regression; do not hide it with a wider limit |
| Checkpoint, TensorRT package, dataset, or device is missing | Report the run as blocked with the missing prerequisite |

Regression prompts for this skill are in `references/regression-cases.md`.
