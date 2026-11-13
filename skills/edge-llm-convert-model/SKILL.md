---
name: edge-llm-convert-model
license: Apache-2.0
metadata:
  author: TensorRT Edge-LLM Team <noreply@nvidia.com>
  tags: [edge-llm, convert, export, engine-build, llm_inference]
description: Convert and run one supported Hugging Face checkpoint with TensorRT Edge-LLM. Use for quantization, checkpoint-to-ONNX export, `llm_build`/`visual_build`/`audio_build`, C++ `llm_inference`, the experimental checkpoint-direct `trtedge build`, engine profile sizing, or copying ONNX artifacts to an Edge target. Do not use for HTTP serving (edge-llm-serve), implementing a new architecture (edge-llm-add-model), installation (edge-llm-install), or accuracy triage (edge-llm-few-layer-test).
---

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Convert, build, and run a checkpoint

Finish one explicit route end to end. Export alone and engine build alone are
intermediate results, not success.

## Instructions

Select one route (ONNX with C++ builders, or checkpoint-direct), then run its steps through inference and compare the output with a reference. Report the route, commands, artifact directories, the result, and anything not run.

## Select the route

| Route | Flow |
|---|---|
| Supported ONNX/C++ | optional `trtedge quantize` -> `trtedge export` -> C++ component builders -> `llm_inference` |
| Experimental checkpoint-direct | `trtedge build` -> `llm_inference` with checkpoint-backed weights |

Read the selected revision's `docs/source/user_guide/getting_started/`
(`quick-start-guide.md`, `supported-models.md`, `support-matrix.md`, and
`direct-engine-builder.md` for the direct route), the model's example page under
`docs/source/user_guide/examples/` if it has one, and `--help` of each command
before running it. Docs and `--help` own flag names, defaults, and value lists;
do not copy them from memory.

Artifacts are not interchangeable between routes, and server cache bundles are
not builder inputs. Do not switch routes because one fails.

## Preflight

Note the checkpoint and revision, `model_type` and `architectures`, target
GPU/TensorRT, and the workload limits (batch, prompt tokens, generated tokens,
media limits). Reject a checkpoint/route combination that is not listed. An
unregistered ONNX model falls back to the default `CausalLM` and can skip
weights, so a successful export does not show support; send architecture work to
`edge-llm-add-model`. Use a fresh output directory per build, and build engines
on the machine that runs them.

## ONNX/C++ route

1. Optional quantization: `trtedge quantize llm|draft ...` on an x86 GPU host
   (`docs/source/user_guide/features/quantization.md`). Keep source and quantized
   checkpoints separate.
2. `trtedge export <checkpoint> <onnx-dir>` writes `llm/`, `visual/`, `audio/`
   component directories as the model needs. Check the export log's
   `Loaded N tensors, skipped M` line: the loader logs counts, not key names, so
   a nonzero skip must match a known exclusion (for example tied embeddings or a
   draft's `t2d`) or be investigated with the checkpoint inspector in
   `edge-llm-add-model`.
3. Copy the complete ONNX directory to the target if export and deployment hosts differ.
4. Build each component with its binary under `build/bin` (`llm_build`,
   `visual_build`, `audio_build`), sizing the profile from the workload.
5. Write a request in the format of `docs/source/user_guide/format/input-format.md`
   and run the matching inference binary.

```bash
trtedge export <checkpoint> <onnx-dir>
./build/bin/llm_build --onnx-dir <onnx-dir>/llm --engine-dir <engine-dir>/llm \
  --max-batch-size <n> --max-input-len <input> --max-kv-cache-capacity <input-plus-output>
./build/bin/llm_inference --engine-dir <engine-dir>/llm \
  --input-file <request.json> --output-file <response.json>
```

For multimodal and speculative components use the docs and `--help`; do not
guess component directory names.

## Checkpoint-direct route

1. Confirm the family in `tensorrt_edgellm/builder/models/registry.py` and the
   limits in `direct-engine-builder.md`.
2. Build, then run inference with the original checkpoint:

```bash
trtedge build --model-dir <checkpoint> --engine-dir <engine-dir> \
  --max-input-len <input> --max-kv-cache-capacity <input-plus-output> --max-batch-size <n>
./build/bin/llm_inference --engine-dir <engine-dir> --checkpoint-dir <checkpoint> \
  --input-file <request.json> --output-file <response.json>
```

Rules that are easy to miss (details in `direct-engine-builder.md`):

- Size the profile explicitly. The `trtedge build` defaults are smoke-test sized
  (see `--help`), so an unsized build rejects ordinary prompts.
- Tensor parallel builds one rank per run: repeat the build for each `--tp-rank`
  with the same `--tp-size`; it covers the LLM component only, without
  speculative decoding.
- Speculative bundles need extra flags at build and at inference
  (`--spec-type`, `--spec-decode`, a paired draft's `--draft-model-dir` and
  `--draft-checkpoint-dir`; native MTP must not pass the draft flags). Follow its
  "Speculative Decoding" section.
- Plugin resolution: from a source install the builder looks for
  `build/lib/libNvInfer_edgellm_plugin.so` relative to the current directory, so
  run from the repository root or pass `--plugin-path`. `llm_inference` uses
  `EDGELLM_PLUGIN_PATH` first, then `<bin>/../lib`; unset it or point it at the
  same build. A wheel user runs outside the source checkout.
- Weight externalization and weightless engines are experimental; use the doc's
  "Optional Features" and "Experimental Weightless Refit Mode" sections.

## Validate

Every required component builds, inference exits 0, the output is non-empty and
well-formed, and greedy output matches the provider reference or an accepted
baseline. For a new precision, architecture, layer, or kernel run
`edge-llm-few-layer-test` first; for CI coverage use `edge-llm-e2e-test`. Do not
claim support from `--help`, export, or a build without inference.

## Examples

- "Export this checkpoint to ONNX, build engines, and run llm_inference": ONNX/C++ route, size the profile from the workload, build every component, run inference, compare with the reference.
- "Build engines directly from the checkpoint": checkpoint-direct route, `trtedge build` with explicit `--max-input-len` and `--max-kv-cache-capacity`, then C++ inference on the bundle.
