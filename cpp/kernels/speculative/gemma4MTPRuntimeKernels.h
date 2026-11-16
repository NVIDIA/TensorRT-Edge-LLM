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

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

namespace trt_edgellm
{
namespace kernel
{

//! Build linear target-verification token IDs: [seed, draft_0, ..., draft_{N-1}].
void launchGemma4MTPBuildVerifyTokens(int32_t const* seedTokenIds, int32_t const* draftTokenIds,
    int32_t* verifyTokenIds, int32_t batchSize, int32_t draftSteps, cudaStream_t stream);

//! Gather one target hidden-state row per batch into the assistant seed-hidden input.
void launchGemma4MTPGatherSeedHidden(void const* sourceHiddenStates, void* seedHiddenStates,
    int32_t const* sourceTokenIndices, int32_t batchSize, int64_t sourceSeqLen, int64_t hiddenSize, size_t elementBytes,
    cudaStream_t stream);

//! Store sampled assistant token IDs into the draft-token matrix at the current draft step.
void launchGemma4MTPStoreDraftToken(int32_t const* selectedTokenIds, int32_t* draftTokenIds, int32_t batchSize,
    int32_t draftSteps, int32_t step, cudaStream_t stream);

} // namespace kernel
} // namespace trt_edgellm
