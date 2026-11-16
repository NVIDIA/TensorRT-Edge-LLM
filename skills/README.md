<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# TensorRT Edge-LLM agent skills

This directory contains the public agent skills. `.agents/skills` and
`.claude/skills` point here for agent discovery.

| Public skill | Purpose |
|---|---|
| [`edge-llm-documentation`](edge-llm-documentation/SKILL.md) | Documentation authoring and review using Google style and repository conventions. |
| [`edge-llm-install`](edge-llm-install/SKILL.md) | Install, verify, update, or remove Edge-LLM and these skills. |
| [`edge-llm-serve`](edge-llm-serve/SKILL.md) | Launch and prove an OpenAI-compatible checkpoint server. |
| [`edge-llm-convert-model`](edge-llm-convert-model/SKILL.md) | Quantize/export, build engines, and run C++ inference. |
| [`edge-llm-add-model`](edge-llm-add-model/SKILL.md) | Add or customize a model on the ONNX and checkpoint-direct routes. |
| [`edge-llm-benchmark-accuracy`](edge-llm-benchmark-accuracy/SKILL.md) | Score dataset accuracy through the server with NeMo Evaluator. |
| [`edge-llm-benchmark-performance`](edge-llm-benchmark-performance/SKILL.md) | Measure TTFT, ITL, and throughput through the server with AIPerf. |
| [`edge-llm-cutedsl-kernel`](edge-llm-cutedsl-kernel/SKILL.md) | Add or change a CuTe DSL kernel and its artifacts. |
| [`edge-llm-plugin-extension`](edge-llm-plugin-extension/SKILL.md) | Add or change a TensorRT plugin across both frontends. |
| [`edge-llm-version-migration`](edge-llm-version-migration/SKILL.md) | Upgrade across releases or TensorRT versions, or backport a fix. |
| [`edge-llm-cpp-test`](edge-llm-cpp-test/SKILL.md) | C++ and CUDA contract tests. |
| [`edge-llm-e2e-test`](edge-llm-e2e-test/SKILL.md) | Complete model deployment tests. |
| [`edge-llm-feature-combinations`](edge-llm-feature-combinations/SKILL.md) | Precision, draft, frontend, and platform combinations a change reaches, checked before submission. |
| [`edge-llm-few-layer-test`](edge-llm-few-layer-test/SKILL.md) | First numeric check for model accuracy mismatches. |
| [`edge-llm-accuracy-debugging`](edge-llm-accuracy-debugging/SKILL.md) | Deeper tensor-level debugging after a mismatch. |

## How these skills are organized

- `AGENTS.md` contains repository-wide invariants only.
- Each `SKILL.md` is a focused executable workflow with explicit inputs,
  stop conditions, phases, validation gates, and completion evidence.
- Detailed maps and rare branches live under `references/` and are loaded only
  when needed. Reusable deterministic inspection belongs under `scripts/`.
- Branch-local docs, source, `--help`, and recent merged examples are the source
  of truth. A skill must not duplicate an exhaustive model/flag/file inventory
  that will become stale.
- Public skills must work in the public GitHub repository.


When editing a skill, test its commands against the current tree, run its
scripts on a minimal fixture, and check all relative links. Static review alone
is not sufficient.
