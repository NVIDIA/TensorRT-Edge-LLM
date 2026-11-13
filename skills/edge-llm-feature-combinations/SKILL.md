---
name: edge-llm-feature-combinations
license: Apache-2.0
metadata:
  author: TensorRT Edge-LLM Team <noreply@nvidia.com>
description: List the supported combinations a change reaches and verify them before the change is submitted. Use before coding a change to quantization or checkpoint parsing, checkpoint loading, the lm_head, KV cache, a speculative decoding method or draft, shared linear or attention code, either engine frontend (ONNX exporter or checkpoint-direct builder), a kernel or TensorRT plugin, tensor parallelism, or a C++ feature guard. Skip it for documentation, CI, test-only, and server-protocol changes, and for a fix that stays inside one model family on one frontend and does not touch quantized weight loading.
---

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Edge-LLM feature combinations

Edge-LLM supports combinations of model architecture, weight precision, lm_head
precision, KV cache precision, speculative decoding method, engine frontend,
engine limits, and platform. A change can pass on the combination its author
ran and fail on another supported combination. This skill finds those
combinations while the code is written.

This skill decides **which** combinations need evidence. The
`edge-llm-e2e-test` coverage table decides **how** to test each one, and the
change as a whole still needs the minimum evidence in that table.

Read `references/axes.md` and `references/interaction-checks.md`. Look up
values in their sources instead of copying them:

- Weight and lm_head precisions: `tests/valid_precisions.py`, the CLI choices
  in `tensorrt_edgellm/scripts/quantize.py`, and the method table in
  `tensorrt_edgellm/quantization/README.md`.
- Platforms and precision limits: `docs/source/user_guide/getting_started/support-matrix.md`
  and the Precision Notes in `supported-models.md`.
- Existing coverage: `tests/python-unittests/`, the `edgellm_add_unit_test`
  registrations in `unittests/CMakeLists.txt`, and the model cases described in
  `tests/model_configs/README.md`.

When the CLI choices and the documentation disagree, list the extra values as
rows (supported or rejected) and correct the source that is wrong.

## Precision comes from metadata

People choose checkpoint names, directory suffixes, and test case names. A name
such as `-NVFP4` or `-FP8-KV` can disagree with what the checkpoint contains.
Determine the precision of each checkpoint from its parsed quantization (see
[Checkpoint parse diff](#checkpoint-parse-diff)) or from its tensor dtypes.

## Before coding

These steps only read code. If you can run a subagent, give it the task and the
files you plan to change, let it build the table in the background while you
write the code, and reconcile the table with the diff before committing.

1. **Check what already exists.** Search the precision sources and
   `tests/python-unittests/` for the precision, key, or feature. Part of the
   combination may already work; start from the nearest existing test.
2. **List the touched axes.** For each symbol the change defines or edits,
   follow its callers through factories and registries (for example
   `make_linear` or the model registry) until you reach a model class, a draft,
   or a CLI entry point. For a tensor key, config field, quantization name,
   `model_type`, or draft variant, search the string instead of the function.
   When adding a method or variant, search the name of an existing sibling (for
   example `"dspark"`) and each `SpecDecodeMode::k*` switch; every hit is a
   branch the new one must join or reject. When the sibling has many hits,
   compare the two file lists (for example with `comm -23`) and read the files
   that mention the sibling but not the new name. Search `tensorrt_edgellm/`,
   `cpp/`, `kernelSrcs/`, `unittests/`, `tests/`, and `examples/`. Map each hit
   to an axis with `references/axes.md`, and check whether the other frontend
   has a counterpart. Check also whether an existing entry point on either
   frontend (for example `--spec-type dflash` for a DFlash-based draft) accepts
   the new checkpoint without detecting it.
3. **Enumerate values.** For each touched axis, take the supported values from
   the sources above. Group values only when they reach the same branch; name
   that file and function, and list the group members next to the
   representative. Base, draft, and lm_head precision are separate axes: cover
   each value once, and cross them only where their code meets (for example a
   draft that reuses the base lm_head).
4. **Add interaction rows.** Add each check in `references/interaction-checks.md`
   whose axes the change touches.
5. **Mark unsupported rows.** For a combination that the sources exclude and
   the change makes reachable, check that the code fails with an error that
   names the unsupported option. A rejection inherited from another method must
   name the new method or the option the user passed. A message that names
   another method, or that names a checkpoint field when the user passed a
   runtime option, is a defect row. Check that each limit a kernel or plugin
   enforces is also checked when the config is parsed or exported, so that an
   invalid input fails before the first decode step.
6. **Choose evidence.** Pick each row's test with the `edge-llm-e2e-test`
   coverage table, and follow these rules:
   - Every touched axis needs at least one row that runs. A row runs when it
     has a log from a local or CI run.
   - Numerics rows need a numeric comparison: an end-to-end case, or a local
     run of `scripts/few-layer-validation.sh` for per-layer checks. Register a
     new few-layer case only when `edge-llm-few-layer-test` calls for one.
   - Write model cases in the schema of `tests/model_configs/README.md` and run
     them with its "Run a case" commands. For a row you cannot run, give the
     exact command and the hardware it needs. For a new speculative decoding
     method, precision, or model architecture, add a case for each frontend.
   - For a row that an existing test already covers, cite the test and confirm
     the checkpoint precision with the parse diff. A test counts as coverage
     only when the test suite you run collects it.

If `pytest --collect-only` fails while importing a third-party plugin, set
`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`; add `--continue-on-collection-errors` when
some test modules need TensorRT. Set `LLM_SDK_DIR` for every Python test.

## Before committing

Compare `git diff --name-only <base>` (which includes uncommitted changes) with
the files in the table, and add rows for new files. Use
`git merge-base HEAD origin/main` as the base when the work is already
committed on a branch. Run the rows and fill in
the results.

## Checkpoint parse diff

Run it for any change to `tensorrt_edgellm/config.py`,
`tensorrt_edgellm/builder/core/quantization.py`, or quantization detection.
Parse every checkpoint you have access to (for example under `$LLM_MODELS_DIR`,
see `tests/README.md`) with the base commit and with the change, through both
frontends, and compare the results:

- ONNX frontend: `ModelConfig.from_pretrained(model_dir, default_attention_scale).quant`
  in `tensorrt_edgellm/config.py`.
- Checkpoint-direct frontend: `DeviceConfig.from_pretrained(model_dir).quant` in
  `tensorrt_edgellm/builder/core/config.py`.
- Draft checkpoints: the matching `make_<variant>_draft_config` in
  `tensorrt_edgellm/config.py`, and on the checkpoint-direct side
  `DeviceConfig.from_pretrained` followed by the variant's `configure_draft`,
  which applies the draft contract.

Parsing reads `config.json`, quantization metadata, and safetensors headers, so
it runs on CPU. Check out the base commit in a separate worktree, and write the
outputs outside the repository. Explain every difference in the table. For a
new `model_type` or draft variant, parse with the change only and confirm that
each module gets the intended precision. Without checkpoints, write unit tests
on minimal synthetic directories that hold `config.json`, the quantization
metadata, and a safetensors file; parsing those directories through the same
entry points counts as the parse evidence.

The parsed result covers the LLM component. It omits the per-module FP16
fallback for modules without quantization tensors and the lm_head that a draft
inherits from its base. A change to weight loading also needs a load or build
run.

## Output

Start the merge request section with the base commit. Then add the table:

| Axis values | Shared branch | Failure to look for | Evidence and command | Result |
|---|---|---|---|---|
| `nvfp4` weights, `fp8` lm_head, ONNX | `models/linear.py:FP8Linear` | lm_head input scale dropped | few-layer: `<command>` | pass, `<log>` |
| `--tp-size 2`, EAGLE3, checkpoint-direct | `builder/cli.py` | unsupported, must be rejected | `trtedge build <args> --tp-size 2 --spec-type eagle3` | error names speculative decoding |
| `fp8` weights, `fp8` lm_head, checkpoint-direct | `builder/core/weights.py:<function>` | scale lost on the other frontend | `<command>` on an SM89 or newer GPU | planned |

Write each Result as `pass <log>`, `fail <log>`,
`defect <file:line>`, `rejects as expected`, or `planned`.

List each untouched axis with the search that supports the claim, for example
`grep -rn <symbol> tensorrt_edgellm/builder` with no matches.

Regression prompts for this skill are in `references/regression-cases.md`.
