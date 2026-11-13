---
name: edge-llm-accuracy-debugging
license: Apache-2.0
metadata:
  author: TensorRT Edge-LLM Team <noreply@nvidia.com>
description: Localize a TensorRT Edge-LLM accuracy mismatch against the model provider's PyTorch reference after few-layer validation identifies a failing component or layer.
---

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Edge-LLM accuracy debugging

Start an accuracy investigation with
[`edge-llm-few-layer-test`](../edge-llm-few-layer-test/SKILL.md). It compares
the reduced PyTorch and Edge-LLM executions and identifies the failing layer,
logits, cache, or recurrent state. Do not weaken its gate to make a mismatch
pass.

If the few-layer test fails and the first incorrect operation is still
unknown, use TensorRT debug-tensor instrumentation to capture intermediate
values from that layer. The [full-intermediate guide](references/full-intermediate-tensors.md)
and `assets/debugTensorDumper.{h,cpp}` describe the diagnostic path; the
standalone [comparator](scripts/compare_hf_trt_tensors.py) can compare a
compatible dump with model-provider PyTorch captures. Match checkpoint bytes,
inputs, preprocessing, dtype, and prefill/decode step before comparing values.
An exported ONNX graph is not the golden reference.

The [source-injection guide](references/source-injection.md) provides a way to
instrument the builder and runtime with `IDebugListener`. Check the bundled
patch against the current code before applying it in an isolated worktree;
the technique is active, but the patch is revision-specific. Do not reset
unrelated changes. After fixing the first incorrect operation, rerun the
few-layer test and the original end-to-end workload.
