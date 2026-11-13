<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Combination axes

Each axis names where its values are defined and where the code branches on
them. The symbols are entry points, not every branch. If a named symbol is
missing, find its replacement and update this file in the same change.

## Weight precision

- **Values:** quantization constants at the top of `tensorrt_edgellm/config.py`
  and `tensorrt_edgellm/builder/core/quantization.py`. The builder adds
  `fp8_a16` and `fp8_block`. `mixed_precision` is a checkpoint algorithm that
  resolves to per-module overrides.
- **Checkpoint formats:** ModelOpt (`hf_quant_config.json` or
  `quantization_config`), compressed-tensors, GPTQ, and AWQ. Each format
  parses through its own function.
- **ONNX frontend:** `config.py` (`_grouped_quant_type` maps compressed-tensors
  groups; `_parse_mixed_precision`, `module_quant_type`,
  `_normalize_module_name`) and the linear classes in `models/linear.py`
  (`FP16Linear`, `FP8Linear`, `MXFP8Linear`, `NVFP4LinearMethod`, the AWQ,
  GPTQ, and INT8 SmoothQuant classes). Block-scaled FP8 has no ONNX linear; the
  loader dequantizes it.
- **Checkpoint-direct frontend:** `builder/core/quantization.py`
  (`parse_quantization`, `normalize_module_name`, `_parse_compressed_tensors`,
  `_compressed_group_contract`, `_parse_modelopt_mixed_precision`,
  `_parse_mixed_precision`) and `builder/core/weights.py`.
- **Target fallback (checkpoint-direct only):** `_target_precision_conversions`
  in `builder/core/builder.py` runs FP8 as W8A16 below SM89, runs NVFP4 as
  W4A16 and block FP8 and MXFP8 as W8A16 below SM100, and converts block FP8
  to MXFP8 on SM100 and newer.
- **Activation mode:** `--no-quantize-activations` runs NVFP4 as A16.

## lm_head precision

- **Values:** `VALID_LM_HEAD_PRECISIONS` in `tests/valid_precisions.py` and
  `_LM_HEAD_METHODS` in `quantization/quantization_configs.py`. Any backbone
  method combines with any lm_head method (`quantization/README.md`).
- **Branches:** `module_quant_type` in `config.py` and in
  `builder/core/weights.py`; the lm_head loaders in `model.py`;
  `baking_lm_head` in `builder/core/weight_policy.py` when the vocabulary is
  reduced.
- **Input scales:** only the `int4_awq_modelopt` and `int8_sq` linears read
  `pre_quant_scale` on either frontend.
- **Drafts:** DFlash, DFlash2, Domino, JetSpec, and DSpark drafts whose
  checkpoint has no `lm_head.*` reuse the base lm_head and its precision
  (`_inherit_dflash_lm_head_quant` in `model.py`). EAGLE3 and MTP drafts do
  not.
- **Tied embeddings:** a tied lm_head has no `lm_head.weight` tensor.

## KV cache precision

- **Values:** `fp16`, `fp8`, read from checkpoint metadata.
- **Detection:** `_compressed_kv_cache_quant` and `_detect_llm_kv_cache_fp8` in
  `config.py`; `_compressed_kv_cache_quant`, `_has_llm_kv_scales`, and
  `_resolve_kv_cache_quant` in `builder/core/quantization.py`.
- **Scale loading:** lookups of `.k_proj.k_scale` and `.v_proj.v_scale` in both
  frontends (`qkv_scales` in `builder/core/weights.py`) and in model families
  with their own attention. A missing key falls back to a scale of 1.0.
- **Runtime:** `cpp/runtime/kvCacheManager.{h,cpp}` and
  `hybridCacheManager.{h,cpp}`. System-prompt cache capture and restore in
  `HybridCacheManager` accept only FP16 KV; generalized context reuse
  (`cpp/runtime/state/contextCache/`) accepts FP16 and FP8.
- **Platform:** on SM80, SM86, and SM87 targets, attention runs in FP16 with an
  FP8 KV cache; the ONNX export needs `--target-sm`.

## Speculative decoding

- **Values:** none, EAGLE3, MTP, Gemma4 MTP, DFlash, DFlash2 (the `dflash` spec
  type with version 2), DSpark, Domino, and JetSpec. Chain and tree drafting
  (`--specDraftTopK` greater than 1) is a separate sub-axis.
- **Draft precision** is a separate axis from the base precision.
- **Code:** draft packages under `tensorrt_edgellm/models/` and
  `tensorrt_edgellm/builder/models/`; MTP heads in per-family `modeling_*_mtp.py`;
  `tensorrt_edgellm/quantization/models/*_draft.py`; `cpp/runtime/decoding/`.
  JetSpec exports with the DFlash draft model.
- Drafts call the base model's linear, attention, and lm_head code.
- **Per-method branches:** `SpecDecodeMode` in
  `cpp/runtime/config/llmEngineConfig.h` and its switches, method lists in the
  CLIs and the server config, and the speculative methods in
  `tests/defs/model_test_config.py`. A new method joins or rejects each one.

## Model architecture

- **Values:** dense, hybrid recurrent (Mamba, gated delta net), and MoE.
- **Branches:** recurrent state handling in `cpp/runtime/state/sharedResources.cpp`
  (`needsBaseVerifyIntermediateStates`) and `hybridCacheManager.{h,cpp}`;
  MoE plugins in `cpp/plugins/`.

## Engine frontend

- **Values:** ONNX exporter (`trtedge export`, then the C++ builder)
  and checkpoint-direct builder (`trtedge build`).
- Each frontend has its own model classes, quantization parser, and weight
  loader.

## Platform

- **Values:** `docs/source/user_guide/getting_started/support-matrix.md`.
- **Branches:** `compute_capability`, `sm110`, and `sm12x` in
  `builder/core/builder.py`; FMHA SM lists in `cpp/CMakeLists.txt`; CuTe DSL
  artifacts selected in `cmake/CuteDsl.cmake`; `IS_TRT_RTX` guards for
  TensorRT-RTX builds.

## Engine limits

- Max batch size, max sequence length, max input length, and image token
  limits. TensorRT can choose different kernels for batch size 1.

## Tensor parallelism

- Split linears (`ColumnParallelLinear`, `RowParallelLinear`) and their scales.
- For `--tp-size` greater than 1, `builder/cli.py` accepts only the LLM
  component and rejects speculative decoding, and `builder/core/builder.py`
  narrows the externalized weight kinds. `onnx/cli.py` exports per-rank ONNX
  and rejects speculative decoding and `--externalize-weights`.

## Other axes

- **MoE backend:** Edge-LLM MoE plugins or `--trt-native-moe`, each with its
  own expert precision limits (`builder/core/builder.py`, `onnx/cli.py`).
- **Weight externalization:** `--externalize-weights` kinds in
  `builder/core/weight_policy.py`.
- **Embedding precision:** token embedding quantization and `--fp8-embedding`.
- **LoRA:** `tensorrt_edgellm/lora/`.
- **Vocabulary reduction:** `tensorrt_edgellm/vocab_reduction/`.
- **Modality:** visual, audio, and TTS components export and build separately,
  each with its own precision options (`tensorrt_edgellm/scripts/quantize.py`).
- **Runtime features:** in-flight batching, KV cache reuse and system-prompt
  cache, guided decoding, and visual token pruning.
