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

#include <cstdint>
#include <cuda_runtime.h>

namespace trt_edgellm::kernel::detail
{

//! Forced acceptance length includes the target bonus token; one accepts no draft tokens.
//! Overrides run after natural acceptance. Missing paths return a valid root-only result with -1 in
//! unused final token padding; the existing acceptance D2H boundary reports the failed experiment.
void launchForceTreeAcceptance(int32_t const* top1Tokens, int32_t const* tokenIds, int8_t const* attentionMask,
    int32_t* acceptedTokenIds, int32_t* acceptedIndices, int32_t* acceptLength, int32_t batchSize, int32_t verifySize,
    int32_t outputStride, int32_t forcedAcceptanceLength, cudaStream_t stream);

//! Greedy linear override; proposalOffset is one when proposal IDs include the root.
void launchForceGreedyAcceptance(int32_t const* top1Tokens, int32_t const* proposalTokenIds,
    int32_t const* proposalLengths, int32_t* acceptedTokenIds, int32_t* acceptedIndices, int32_t* acceptLength,
    int32_t batchSize, int32_t proposalStride, int32_t proposalOffset, int32_t verifySize,
    int32_t forcedAcceptanceLength, cudaStream_t stream);

//! Sample the synthetic bonus from the target row after the forced prefix. Null targetIds denotes dense vocabulary.
void launchForceSamplingAcceptance(float const* targetProbabilities, int32_t const* targetIds,
    int32_t const* proposalTokenIds, int32_t const* proposalLengths, float const* uniforms, int32_t* acceptedTokenIds,
    int32_t* acceptedIndices, int32_t* acceptLength, int32_t batchSize, int32_t proposalStride, int32_t verifySize,
    int32_t targetSupportSize, int32_t const* maxAcceptLengths, int32_t forcedAcceptanceLength, cudaStream_t stream);

} // namespace trt_edgellm::kernel::detail
