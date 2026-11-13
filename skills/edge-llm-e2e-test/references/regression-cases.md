<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Regression cases

Use these prompts against the current checkout. Expected responses must follow
`tests/model_configs/README.md` for model-case commands and
`tests/README.md` for environment setup instead of reproducing either contract.

## Add a model case

**Prompt:** Add my checkpoint to the end-to-end tests and show me how to run
only that deployment.

**Expected:** Read both canonical READMEs, add one complete case under
`tests/model_configs/l0/<family>/`, collect the owning priority, inspect its
node IDs, and use the documented single-case selection. Choose a real full
Hugging Face ID or absolute path, supported hardware, registered input file,
and declared success metrics. Do not invent a pytest function or checkpoint.

## Verify correctness

**Prompt:** My model case passed. Is the model correct?

**Expected:** Report separately the absolute output-quality result, required
metric production, and comparison with the recorded baseline. State the
hardware and executed phases. Do not call a collection-only run, skipped case,
or `BASELINE_REQUIRED=0` seeding run a regression pass.

## Seed a new baseline

**Prompt:** My new case has no baseline. How do I establish one?

**Expected:** Follow the canonical two-step workflow: explicitly disable the
required comparison while autosaving a reviewed successful result, then rerun
with comparison enabled against the saved CSV. A missing baseline must fail by
default, and the first measurement alone is not parity evidence.

## Preflight without a GPU

**Prompt:** I wrote a model case but have no GPU. What can I verify?

**Expected:** Use the canonical collection command for the owning priority and
inspect the generated phases and IDs. State that this proves only schema and
routing; it does not build an engine or measure correctness.

## Diagnose a missing metric

**Prompt:** My case fails with `required metric rouge1 was not produced`.

**Expected:** Inspect the input-file stem and metric registration in
`tests/defs/utils/accuracy.py`. Do not delete the declared check or widen a
regression tolerance. Distinguish the failure from low ROUGE and from a missing
baseline.

## Change the schema

**Prompt:** Add a new engine dimension to model cases.

**Expected:** Change `tests/defs/model_test_config.py`, update the canonical
model-case README, collect every affected priority, and execute a case using
the field. Do not add an unused field or a unit test that merely retests YAML
expansion.

## Python behavior

**Prompt:** I changed validation in the OpenAI-compatible server. Add the right
test and verify it.

**Expected:** Add a focused test in the nearest server unit suite with a concise
contract docstring, mutation-test it, run the owning suite, and run pre-commit.
Require a server model case only if the behavior reaches model execution.

## C++ behavior

**Prompt:** I changed request handling in `cpp/runtime/llmInferenceRuntime.cpp`.
Add and run the smallest sufficient C++ test.

**Expected:** Inspect the owning test registration, build and run the registered
target with a focused GTest filter, and include required TensorRT/runtime setup
from the canonical testing docs. Do not infer a target name from the source.

## Kernel or plugin behavior

**Prompt:** I changed a CuTe DSL kernel and its TensorRT plugin wrapper. What
must pass?

**Expected:** Run the nearest contract coverage and device execution on a
compatible SM, then execute the smallest model case that enqueues the changed
path when needed. A collection-only run or hardware skip is incomplete.
