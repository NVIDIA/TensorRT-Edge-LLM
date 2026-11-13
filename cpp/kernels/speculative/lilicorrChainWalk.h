/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "common/bindingNames.h"
#include "common/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace trt_edgellm
{
namespace kernel
{

using binding_names::kLiLiCorrMaxTopK;

//! Walks the LiLiCorr candidate lattice one slot at a time, starting from candidate index 0.
//! transitionScores[b,s,prev,next] scores candidate `next` of slot s given the index chosen at slot s-1.
//! Greedy rows (greedyMask[b] != 0 or temperatures[b] < 1e-3) take the argmax with ties to the lowest index and
//! emit a one-hot q; other rows sample softmax(row / temperature) by inverse CDF on uniforms[b,s] and emit it as q.
//! \param candidateIds [B,S,K] INT32, K <= kLiLiCorrMaxTopK
//! \param transitionScores [B,S,K,K] FP32
//! \param uniforms [B,S] FP32
//! \param temperatures [B] FP32
//! \param greedyMask [B] INT32
//! \param proposalTokenIds [B,S] INT32 output
//! \param proposalProbabilities [B,S,K] FP32 output
void launchLiLiCorrChainWalk(rt::Tensor const& candidateIds, rt::Tensor const& transitionScores,
    rt::Tensor const& uniforms, rt::Tensor const& temperatures, rt::Tensor const& greedyMask,
    rt::Tensor& proposalTokenIds, rt::Tensor& proposalProbabilities, cudaStream_t stream);

} // namespace kernel
} // namespace trt_edgellm
