---
name: edge-llm-few-layer-test
license: Apache-2.0
metadata:
  author: TensorRT Edge-LLM Team <noreply@nvidia.com>
description: Add or repair a few-layer numeric validation case, which compares the first N decoder layers of a checkpoint against a PyTorch golden. Covers when a change needs one, choosing the checkpoint, setting a cosine gate that is measured rather than tuned, and triaging a failure. Apply when adding a model architecture, layer type, quantization recipe, or kernel, or when a few-layer case fails.
---

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Edge-LLM few-layer numeric validation

Few-layer validation runs the first N decoder layers of a checkpoint through a PyTorch
golden and through the EdgeLLM engine, and compares logits, KV cache, and recurrent /
conv state per round. It answers one question: **does a layer compute the right
numbers.**

Canonical documents — read rather than restate:

- `docs/source/developer_guide/testing/few-layer-validation.md`: what the test does, why
  it gates on two quantities, where each threshold comes from, and the
  **Architecture coverage** table of what has been made to work and with which flags.
- `tests/model_configs/README.md`: the case schema and the `few_layer` scenario fields.

## When a change needs a few-layer case

**Most changes do not.** Add one for:

- a new model architecture,
- a new layer type,
- a new quantization recipe,
- a new kernel,
- a change to few-layer's own truncation or golden-loading path — `--num-decoder-layer`
  handling, `tests/golden_layer_dump.py` model-class resolution, weight loading. This
  code runs nowhere else, so a few-layer case is its only coverage.

Do not add one for scheduling, cache, batching, or decoding-mode work. Those are
orchestration; this fixture is a handful of rounds over one batch of requests issued
together, and cannot see a scheduling or eviction error. Use `edge-llm-cpp-test` (substitute engines) for those, and
`edge-llm-e2e-test` for deployment-level behavior.

**Removing an existing case is not a cleanup.** Several cases are the only coverage of
a decoding mode, a model family, or a kernel — dropping one takes a dimension to zero
while the remaining list still looks plentiful. A case may only be removed by replacing
it with a test at least as strong, and only with the user's explicit approval. Propose;
do not enact.

## Non-negotiable

1. **Ask which checkpoint to use. Never infer it from the model name.** The right
   variant is frequently not the obvious one, and the wrong one fails in ways that look
   like product bugs. See `references/checkpoints-and-golden.md`.
2. **If HuggingFace has no matching implementation, stop and ask the user what they
   want.** There is no fallback and no proxy golden. This is exception-level: surface it
   rather than working around it.
3. **Prove the golden loaded real weights before trusting a single number.** HF does not
   error on a checkpoint whose keys it does not recognise — it random-initialises and
   runs. A fake golden produces confident, meaningless numbers.
4. **The cosine gate is measured, not tuned until the run passes.** A failing run is
   triaged, not accommodated. See `references/thresholds.md`.
5. **Prove the run exercised what you think it did.** The script's exit code is
   accounting; that the intended path executed is the work. See "Assert the work, not
   the accounting" in the discipline reference.

## Workflow

1. State what the case protects, in plain language, before running anything: which
   architecture, layer type, recipe or kernel would have to break for it to go red. If
   the answer is already covered by an existing row in the coverage table, you may not
   need a new case.
2. Ask the user for the checkpoint. Confirm what the builder can actually build from it.
3. Run the driver directly first, not through pytest:
   ```
   bash scripts/few-layer-validation.sh --model <ckpt> --num-layers 4 \
     --build-dir $PWD/build --python $PWD/venv/bin/python --cos 0.80
   ```
   A deliberately loose `--cos` here measures rather than gates. Add
   `--direct-builder` and any modality or draft flags the architecture needs.
4. Confirm the golden is real and the intended path ran, before reading the cosine.
5. Set the gate from the measurement, following `references/thresholds.md`. Triage any
   failure there before touching a threshold.
6. Register the case under `tests/model_configs/l0/<family>/<base-model>.yml` with
   `scenario: few_layer`, taking `input_file` from "The request fixture" below rather
   than from the case next to it. Compute `requirements.min_gpu_memory_gb` from the **truncated**
   resident size plus headroom, not the checkpoint's full size.
7. Mutation-test the gate: perturb the numerics the case is supposed to protect and
   confirm it goes red, then revert. A gate well below 0.99 is exactly where a phantom
   assertion hides.
8. Add a row to the coverage table in the canonical document, including the flags the
   case needs and any caveat expensive to rediscover.

## The request fixture

Use `tests/test_cases/ragged_batch.json`: four requests of unequal length, which is also
what the driver picks when `--input-file` is omitted. The case YAML has to name a fixture
explicitly, and the pipeline cases sitting beside yours in the same file name
`llm_basic.json`, so copying the neighbour is the easy mistake and nothing will fail to
tell you about it.

**A `batch_size: 1` fixture is not a smaller version of the same test.** The engine dumps
each tensor full-length over the active batch and the comparison slices per sequence, so
with one request "reads the right slot" and "always reads slot 0" produce identical
numbers, and no call ever prefills two lengths at once. Four unequal prefills cost no
extra build and no extra checkpoint load.

**A long `max_generate_length` is not a stronger version either.** Both gates are a
minimum over every tensor and every round, so extra decode rounds exercise no new code
and only draw again from the distribution described under "The known limitation" in
`references/thresholds.md`. `llm_basic.json` runs 128 rounds; one MoE case was gated at
0.90 on the strength of a single one of them.

Reach for `tests/test_cases/llm_few_layer.json` (one request, 8 rounds) only when the
checkpoint is too large to push four sequences through the PyTorch golden inside the
case's timeout — and say so in the case comment, so the next reader does not treat the
narrower fixture as the norm.

## When a case fails

A failing few-layer case is ambiguous in a way most tests are not: it may be a product
defect, or the expected floor of a quantization recipe. Decide which before doing
anything else — `references/thresholds.md` gives the triage. Lowering the threshold to
make the run pass, while the case's comment still claims the original contract, is the
silent removal of coverage described in the discipline reference.

If the mismatch remains after validating the golden and threshold, use the
[`edge-llm-accuracy-debugging`](../edge-llm-accuracy-debugging/SKILL.md)
skill to capture intermediate tensors at the failing layer.

## References

| File | Read when |
|------|-----------|
| `../edge-llm-cpp-test/references/test-discipline.md` | Before writing or changing any test |
| `references/checkpoints-and-golden.md` | Choosing a checkpoint, or the golden looks wrong |
| `references/thresholds.md` | Setting a gate, or a case fails |
