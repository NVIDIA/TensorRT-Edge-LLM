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

#include <cstdint>

namespace trt_edgellm
{
namespace kernel
{

constexpr int32_t kDFlash2MaxSelectorTopK = 16;
constexpr int32_t kDFlash2MaxSelectorRank = 256;

void launchDFlash2CandidateSelector(rt::Tensor const& candidateIds, rt::Tensor const& unaryLogits,
    rt::Tensor const& projectedHidden, rt::Tensor const& anchorTokenIds, rt::Tensor const& predecessorCodebook,
    rt::Tensor const& successorCodebook, rt::Tensor const& uniforms, rt::Tensor const& temperatures,
    rt::Tensor const& greedyMask, rt::Tensor& proposalTokenIds, rt::Tensor& proposalProbabilities, cudaStream_t stream);

//! DDTree Stage A for DFlash2: selector-scored child lists for every (proposal row, parent state).
//!
//! Row d's parent states are the support indices of row d - 1, so a parent's state is the predecessor token
//! supportIds[b, d - 1, state]; row 0 has the single state 0 whose predecessor is the anchor. A child c of
//! state i scores s = unary + <P[pred] * z, Q[c]>, bitwise equal to launchDFlash2CandidateSelector's score, and
//! carries log(softmax(s)) over the row's valid support at temperature 1. Out-of-vocabulary and repeated support
//! ids score -inf. Each list keeps the best candidateTopK by (score desc, token asc); childStates holds the
//! candidate's support index, the child's state in row d + 1. Unused lists and slots are -inf.
//!
//! Inputs: supportIds [B, P, S] INT32, unaryValues [B, P, S] FP32, projectedHidden [B, P, R] FP16,
//! anchorTokenIds [B] INT32, predecessor/successor codebooks [V, R] FP16.
//! Outputs: candidateTokenIds / candidateLogProbs / candidateChildStates [B, P, S, K], K <= S.
void dflash2SelectorTreeCandidates(rt::Tensor const& supportIds, rt::Tensor const& unaryValues,
    rt::Tensor const& projectedHidden, rt::Tensor const& anchorTokenIds, rt::Tensor const& predecessorCodebook,
    rt::Tensor const& successorCodebook, rt::Tensor& candidateTokenIds, rt::Tensor& candidateLogProbs,
    rt::Tensor& candidateChildStates, cudaStream_t stream);

} // namespace kernel
} // namespace trt_edgellm
