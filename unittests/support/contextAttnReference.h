/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "common/tensor.h"
#include <cuda_runtime.h>
#include <stdint.h>

namespace trt_edgellm
{
namespace rt
{

/*!
 * Launch reference FMHA kernel using Tensor objects (BSHD layout).
 * Inputs: Q [B, Sq, Hq, D], K [B, Sk, Hkv, D], V [B, Sk, Hkv, D].
 * Output: O [B, Sq, Hq, D] (attention result only). Dimensions are taken from tensor shapes.
 *
 * @param Q        Query tensor [B, Sq, Hq, D]
 * @param K        Key tensor [B, Sk, Hkv, D]
 * @param V        Value tensor [B, Sk, Hkv, D]
 * @param O        Output tensor [B, Sq, Hq, D]
 * @param causal   If true, apply causal mask (k > q disallowed).
 * @param attentionScale Absolute multiplier applied to QK^T before softmax.
 * @param stream   CUDA stream for kernel launch
 */
void launchFmhaReferenceBshd(Tensor const& Q, Tensor const& K, Tensor const& V, Tensor& O, bool causal,
    float attentionScale, cudaStream_t stream = nullptr);

/*!
 * Launch reference FMHA kernel for compact layout.
 * Inputs: Q/K/V/O [total_tokens, H, D], cuSeqlens [B+1].
 * maxSeqLen controls grid/shared-memory extent and must cover every per-batch sequence length.
 *
 * @param Q        Query tensor [total_tokens, H, D]
 * @param K        Key tensor [total_tokens, H, D]
 * @param V        Value tensor [total_tokens, H, D]
 * @param O        Output tensor [total_tokens, H, D]
 * @param cuSeqlens Prefix-sum lengths [B+1]
 * @param maxSeqLen Maximum sequence length
 * @param causal   If true, apply causal mask (k > q disallowed).
 * @param attentionScale Absolute multiplier applied to QK^T before softmax.
 * @param stream   CUDA stream for kernel launch
 */
void launchFmhaReferenceCompact(Tensor const& Q, Tensor const& K, Tensor const& V, Tensor& O, Tensor const& cuSeqlens,
    int32_t maxSeqLen, bool causal, float attentionScale, cudaStream_t stream = nullptr);

} // namespace rt
} // namespace trt_edgellm
