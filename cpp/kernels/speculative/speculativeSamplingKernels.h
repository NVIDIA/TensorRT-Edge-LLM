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

#include <cstdint>
#include <cuda_runtime.h>

namespace trt_edgellm::kernel::detail
{

void launchSparseAccept(float const* targetProbabilities, int32_t const* targetIds, float const* proposalProbabilities,
    int32_t const* proposalIds, int32_t const* proposalTokenIds, int32_t const* proposalLengths, float const* uniforms,
    int32_t* acceptedTokenIds, int32_t* acceptLength, int32_t* acceptedTokenIndices, int32_t batchSize,
    int32_t proposalStride, int32_t verifyProposalLen, int32_t targetSupportSize, int32_t proposalSupportSize,
    cudaStream_t stream, int32_t const* maxAcceptLengths);

void launchDenseTargetAccept(float const* targetProbabilities, float const* proposalProbabilities,
    int32_t const* proposalIds, int32_t const* proposalTokenIds, int32_t const* proposalLengths, float const* uniforms,
    int32_t* acceptedTokenIds, int32_t* acceptLength, int32_t* acceptedTokenIndices, int32_t batchSize,
    int32_t proposalStride, int32_t verifyProposalLen, int32_t vocabSize, int32_t proposalSupportSize,
    cudaStream_t stream, int32_t const* maxAcceptLengths);

void launchNormalizeTopKTopP(float const* topKValues, float* probabilities, int32_t rows, int32_t topK,
    float temperature, float topP, cudaStream_t stream);

} // namespace trt_edgellm::kernel::detail
