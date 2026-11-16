<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Interaction checks

Each check names axes that meet in shared code, the failure to look for, and
the evidence that finds it. Add the matching rows to the combination table
whenever a change touches one of the listed axes. Add a check only for axes
that meet in shared code.

## Checkpoint parsing

- **Weight precision and lm_head precision.** Mixed-precision checkpoints
  store per-module quantization groups. Check that every group shape parses to
  the intended per-module precision on both frontends, and that other mixed
  checkpoints keep their KV cache and visual precision. Evidence: checkpoint
  parse diff for the LLM; export or build of the visual component for visual
  precision.
- **Module name normalization and per-module overrides.**
  `_normalize_module_name` (`config.py`) and `normalize_module_name`
  (`builder/core/quantization.py`) strip different prefixes. A prefix left on
  a name can make an lm_head override match the whole body, or miss its module.
  Evidence: unit test on the parsed per-module precision for a checkpoint
  whose body and lm_head differ.
- **Checkpoint name and contents.** Names are chosen by people. Read the parsed
  quantization or the tensor dtypes; the builder falls back to FP16 for a module
  without quantization tensors.

## lm_head

- **Quantized lm_head and input scales.** Only the `int4_awq_modelopt` and
  `int8_sq` linears read `pre_quant_scale`, and the lm_head loaders treat it as
  optional. Check that a checkpoint with smoothing scales on a quantized
  lm_head either applies the scale or fails at load, on both frontends and in
  every draft that reuses the base lm_head.
- **lm_head and vocabulary size.** Some checkpoints set `vocab_size` larger
  than the tokenizer vocabulary. Check that sampling cannot return IDs at or
  above the tokenizer vocabulary size, with a quantized or reduced lm_head.

## KV cache

- **FP8 KV and scale keys.** Each checkpoint format and model family names its
  KV scales. A missing key falls back to a scale of 1.0 or downgrades the cache
  to FP16. Check the scale values that each frontend loads.
- **FP8 KV and cache reuse.** System-prompt cache capture and restore in
  `HybridCacheManager` accept only FP16 KV; generalized context reuse accepts
  FP16 and FP8. Check which path the change reaches, and that an unsupported
  path fails with an error that names FP8 KV.

## Numerics

- **Low precision and batch size.** TensorRT can select different kernels for
  batch size 1 and larger batches. Run an end-to-end decode at batch size 1
  and at a larger batch size.
- **Low precision and recurrent state.** Check value ranges of recurrent state
  in hybrid models (for example Mamba `dt`) against the storage precision.
- **ONNX export and equal weights.** Export merges equal initializers. Check
  per-layer outputs with a few-layer run after changes to export-time weight
  handling.

## Model architecture

- **Speculative decoding and hybrid recurrent bases.** After a partial accept,
  a hybrid base must roll back its recurrent state. Add a hybrid-base row for a
  change to speculative decoding.

## Frontends, targets, and builds

- **ONNX exporter and checkpoint-direct builder.** Each frontend parses
  quantization and loads weights separately. Make the same decision in both, or
  state why one frontend is out of scope.
- **Runtime sidecar files.** Each frontend writes method-specific tensors and
  configuration for the runtime on its own. Compare the files both frontends
  produce for the same checkpoint.
- **Native target and fallback target.** The checkpoint-direct builder runs
  the same checkpoint natively or as W8A16 or W4A16, depending on SM. Add one
  row for a native target and one for a fallback target.
- **Tensor parallelism and other axes.** Both frontends reject speculative
  decoding with tensor parallelism, and each limits the components and
  externalized weights it accepts. Check the error for each combination the
  change makes reachable.
- **Build variants and C++ feature guards.** Apply a feature `#if` to every
  overload and member it affects. Build every variant that compiles the changed
  file (TensorRT and TensorRT-RTX, with and without the feature).
