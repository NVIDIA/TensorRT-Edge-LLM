---
name: edge-llm-add-model
license: Apache-2.0
metadata:
  author: TensorRT Edge-LLM Team <noreply@nvidia.com>
  tags: [edge-llm, model-support, onboarding, checkpoint]
description: Implement TensorRT Edge-LLM support for a new Hugging Face checkpoint or model family. Use for a new `model_type`, architecture, text/vision/audio/action component, checkpoint tensor layout, speculative draft, model-specific custom operation, or runtime artifact. Starts from checkpoint inspection and the provider reference, finds the closest existing family, chooses the smallest supported frontend surface, and requires reference-to-export-to-engine-to-inference evidence. Do not use merely to convert an already supported checkpoint (edge-llm-convert-model), serve it (edge-llm-serve), or develop a standalone plugin/kernel (edge-llm-plugin-extension, edge-llm-cutedsl-kernel).
---

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Add model support

Implement from the exact checkpoint and its reference model. Do not port from a
model name, marketing family, or memory of another runtime.

## Instructions

Inspect the exact checkpoint and its provider reference, find the closest existing family, classify the smallest supported change, implement one component at a time, and validate each against the reference before the next. Finish with the Review pass and a short report.

## Read first

1. `AGENTS.md` and `CODING_GUIDELINES.md`.
2. [references/reconnaissance.md](references/reconnaissance.md), including the
   checkpoint inspector and finding the closest family.
3. Branch-local model docs and the selected frontend design:
   - ONNX: `checkpoint-export.md`, `engine-builder.md`, customization guide.
   - checkpoint-direct: `onnxless-builder.md`, direct-builder guide, and
     `tensorrt_edgellm/builder/README.md`.
4. [references/file-map.md](references/file-map.md) after the implementation
   class is known. Search current code for every symbol it names; paths can
   move.

## Phase 1: classify the change

Edge-LLM has independent model implementations for its supported ONNX frontend
and experimental checkpoint-direct frontend. Shared C++ runtimes do not make a
Python model family portable between them.

Select only the route the user requested:

| Reconnaissance result | Smallest change |
|---|---|
| Existing implementation, new equivalent `model_type` | Registry alias plus a proof that config and tensor contracts match |
| Standard dense decoder compatible with ONNX default model | No family fork; register the type explicitly (an unregistered ONNX `model_type` silently uses the default model, while the direct builder raises) and prove all required weights load and outputs match |
| Different block/math/tensor layout | New or extended family in that frontend |
| Missing vision/audio/action/draft component | Component implementation plus builder/runtime wiring |
| Missing operation expressible with existing layers | Add frontend lowering; avoid a plugin |
| Operation needs a custom kernel/layout/state contract | Use `edge-llm-plugin-extension`; use `edge-llm-cutedsl-kernel` for its CuTe DSL kernel |
| Model requires a model-local native runtime | Use `experimental_models/` only when the design requires it and the user accepted experimental scope |

State a short plan: the closest family, the differences, the route, and the files
expected to change. Get user agreement before a model-local native runtime or a
broad shared-core change.

## Phase 2: establish a reference before implementation

Create the smallest deterministic provider run that exercises the difference:
fixed seed, greedy decoding, known inputs, and captured intermediate output for
the first changed layer/component. Store model/config revision and input data.

For a new architecture, add a few-layer reference case before full deployment.
Use `edge-llm-few-layer-test`; this catches weight mapping and math errors faster
than a full generated sentence.

## Phase 3: implement vertically

Work one component at a time:

1. configuration and registry dispatch;
2. weight mapping/loading: every model parameter must map to a checkpoint key,
   and every checkpoint tensor must be used, transformed, tied, or knowingly
   ignored (see Weight reconciliation);
3. model/component graph;
4. operation lowering and plugin only when required;
5. runtime artifacts (tokenizer, template, processor, config);
6. C++ profile/binding/runtime changes only when the exported contract changes;
7. focused unit tests and negative contract tests.

Keep model-specific branches in the family directory. Shared code changes need
at least two existing families in tests to prove no regression.

### Weight reconciliation

The ONNX loader cannot be trusted to report skipped keys: prefix-filter and
`key_remap` drops are counted but never logged, "Key not found in model" is logged
only at DEBUG, the export CLI has no verbose flag, and parameters the checkpoint
never fills are not reported at all. So reconcile by name instead of by log:

1. Take the checkpoint tensor names from `inspect_checkpoint.py`.
2. Take the model's `state_dict()` keys (parameters and buffers) for the family
   class, and apply the family's `key_remap` to the checkpoint names.
3. Diff both ways. Every checkpoint-only name must be explained (tied, fused,
   quantization scale, or knowingly ignored), and every model-only name must be
   a buffer computed at construction or a defect.

The loader's "Loaded N, skipped M" line is only a cross-check. The direct builder
raises `KeyError` for a missing required tensor but does not report unused
checkpoint tensors, so run the same diff there.

For multimodal work, bring up the text model first, then each encoder/head, then
the composed runtime. For speculative decoding, prove base inference before the
draft path.

## Phase 4: validation ladder

Each level gates the next:

1. checkpoint inventory and provider reference;
2. config/registry/weight unit tests;
3. few-layer or component numeric comparison;
4. selected frontend build:
   - ONNX export and ONNX validation, or
   - checkpoint-direct network build;
5. TensorRT engine build on each claimed target;
6. deterministic inference versus the provider reference;
7. an end-to-end test for the model (see `edge-llm-e2e-test`);
8. documentation and support-table update.

Use `edge-llm-convert-model` for levels 4–6 and `edge-llm-e2e-test` for level 7.
Do not add a checkpoint to `supported-models.md` before inference and its
baseline pass on the stated route and platform; mark rows that work only on the
direct builder `(checkpoint-direct only)`.

## Review pass

Before completion: search for every registry or switch use of the added enum or
`model_type`; make sure no private paths, internal hostnames, checkpoint
locations, or credentials are in the change; run pre-commit and the narrowest
applicable tests; and state which route, precision, platform, and component
remain unverified.

## Report

State the checkpoint and revision, the closest family and route, what changed per
layer, the validation commands and hardware, the reference comparison, the weight
reconciliation result, and the remaining gaps. Export-only, compile-only,
collection-only, or "the output looks plausible" is not model-support evidence.

## Available scripts

| Script | Purpose | Run |
|---|---|---|
| `scripts/inspect_checkpoint.py` | Inventory `config.json`, safetensors headers (top-level shards only), shard completeness, tensors absent from the index, and tensor patterns without importing provider code; exits 2 when there are no shards or the index names missing shards or tensors | `python3.10 skills/edge-llm-add-model/scripts/inspect_checkpoint.py <checkpoint-dir> --output <work-dir>/checkpoint-inventory.json` |

Run scripts with the shell tool, with `python3.10` or newer (the default `python3` can be older),
and keep the JSON output as evidence.

## Examples

- "Add support for a new Hugging Face model_type": run the checkpoint inspector, find the closest existing family, state the plan, then implement and validate vertically.
- "This checkpoint is a renamed copy of an existing family": register the alias and prove the config and tensor contracts match; add no new family code.
