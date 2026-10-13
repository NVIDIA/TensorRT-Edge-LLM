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

#include "kernels/speculative/requestStableRngKernels.h"

#include "common/cudaUtils.h"
#include "runtime/decoding/requestStableRng.h"

namespace trt_edgellm::kernel
{
namespace
{

__global__ void requestStableSpecUniformsKernel(uint64_t const* requestSeeds, uint64_t const* nextAbsolutePositions,
    float* proposalUniforms, float* acceptUniforms, int32_t proposalLen)
{
    int32_t const batch = static_cast<int32_t>(blockIdx.x);
    int32_t const lane = static_cast<int32_t>(threadIdx.x);
    uint64_t const seed = requestSeeds[batch];
    uint64_t const nextPosition = nextAbsolutePositions[batch];
    int32_t const acceptStride = 2 * proposalLen + 1;
    if (lane < proposalLen)
    {
        uint64_t const position = nextPosition + static_cast<uint64_t>(lane);
        proposalUniforms[batch * proposalLen + lane]
            = rt::requestStableUniform(seed, position, rt::SpecRandomPurpose::kProposal, 0);
        acceptUniforms[batch * acceptStride + lane]
            = rt::requestStableUniform(seed, position, rt::SpecRandomPurpose::kAccept, 0);
        acceptUniforms[batch * acceptStride + proposalLen + lane]
            = rt::requestStableUniform(seed, position, rt::SpecRandomPurpose::kResidual, 0);
    }
    if (lane == 0)
    {
        acceptUniforms[batch * acceptStride + 2 * proposalLen] = rt::requestStableUniform(
            seed, nextPosition + static_cast<uint64_t>(proposalLen), rt::SpecRandomPurpose::kBonus, 0);
    }
}

} // namespace

void launchRequestStableSpecUniforms(uint64_t const* requestSeeds, uint64_t const* nextAbsolutePositions,
    float* proposalUniforms, float* acceptUniforms, int32_t batchSize, int32_t proposalLen, cudaStream_t stream)
{
    requestStableSpecUniformsKernel<<<batchSize, 32, 0, stream>>>(
        requestSeeds, nextAbsolutePositions, proposalUniforms, acceptUniforms, proposalLen);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace trt_edgellm::kernel
