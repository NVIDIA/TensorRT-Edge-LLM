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

namespace trt_edgellm
{
namespace kernel
{

//! Exact rejection sampling over sparse target/proposal supports.
//!
//! Each support row must contain unique token ids. The residual distribution
//! is evaluated on support(p) union support(q), so
//! callers such as DFlash2 never allocate [B, P, vocab] proposal probabilities.
void speculativeSparseAccept(rt::Tensor const& targetSupportProbabilities, rt::Tensor const& targetSupportIds,
    rt::Tensor const& proposalSupportProbabilities, rt::Tensor const& proposalSupportIds,
    rt::Tensor const& proposalTokenIds, rt::Tensor const& proposalLengths, rt::Tensor const& acceptUniforms,
    rt::Tensor& acceptedTokenIds, rt::Tensor& acceptLength, rt::Tensor* acceptedTokenIndices, cudaStream_t stream,
    rt::Tensor const* maxAcceptLengths = nullptr);

//! Verify sparse draft proposals against a dense target distribution without materializing a dense proposal or
//! residual distribution. Proposal support IDs must be unique within each row.
void speculativeDenseTargetAccept(rt::Tensor const& targetProbabilities, rt::Tensor const& proposalSupportProbabilities,
    rt::Tensor const& proposalSupportIds, rt::Tensor const& proposalTokenIds, rt::Tensor const& proposalLengths,
    rt::Tensor const& acceptUniforms, rt::Tensor& acceptedTokenIds, rt::Tensor& acceptLength,
    rt::Tensor* acceptedTokenIndices, cudaStream_t stream, rt::Tensor const* maxAcceptLengths = nullptr);

//! Convert descending top-k logits to the exact temperature/top-p sampling
//! distribution on the fixed sparse support. Entries outside the nucleus are
//! zero and the retained prefix is renormalized.
void speculativeNormalizeTopKTopP(
    rt::Tensor const& topKValues, rt::Tensor& probabilities, float temperature, float topP, cudaStream_t stream);

} // namespace kernel
} // namespace trt_edgellm
