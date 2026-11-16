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

#include "gemma4MTPRuntimeKernels.h"

#include "common/checkMacros.h"

#include <limits>

namespace trt_edgellm
{
namespace kernel
{
namespace
{

__global__ void buildVerifyTokensKernel(int32_t const* __restrict__ seedTokenIds,
    int32_t const* __restrict__ draftTokenIds, int32_t* __restrict__ verifyTokenIds, int32_t draftSteps,
    int32_t verifyTokenCount, int32_t totalElements)
{
    int32_t const idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= totalElements)
    {
        return;
    }

    int32_t const batchIdx = idx / verifyTokenCount;
    int32_t const posIdx = idx % verifyTokenCount;
    if (posIdx == 0)
    {
        verifyTokenIds[batchIdx * verifyTokenCount] = seedTokenIds[batchIdx];
    }
    else
    {
        verifyTokenIds[batchIdx * verifyTokenCount + posIdx] = draftTokenIds[batchIdx * draftSteps + posIdx - 1];
    }
}

__global__ void gatherSeedHiddenKernel(uint8_t const* __restrict__ sourceHiddenStates,
    uint8_t* __restrict__ seedHiddenStates, int32_t const* __restrict__ sourceTokenIndices, int64_t sourceSeqLen,
    int64_t rowBytes, int64_t totalBytes)
{
    int64_t const idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= totalBytes)
    {
        return;
    }

    int32_t const batchIdx = static_cast<int32_t>(idx / rowBytes);
    int64_t const byteIdx = idx % rowBytes;
    int64_t const sourceTokenIdx = sourceTokenIndices[batchIdx];
    int64_t const srcOffset = (static_cast<int64_t>(batchIdx) * sourceSeqLen + sourceTokenIdx) * rowBytes + byteIdx;
    seedHiddenStates[idx] = sourceHiddenStates[srcOffset];
}

__global__ void storeDraftTokenKernel(int32_t const* __restrict__ selectedTokenIds, int32_t* __restrict__ draftTokenIds,
    int32_t draftSteps, int32_t step, int32_t batchSize)
{
    int32_t const batchIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (batchIdx >= batchSize)
    {
        return;
    }

    draftTokenIds[batchIdx * draftSteps + step] = selectedTokenIds[batchIdx];
}

} // namespace

void launchGemma4MTPBuildVerifyTokens(int32_t const* seedTokenIds, int32_t const* draftTokenIds,
    int32_t* verifyTokenIds, int32_t batchSize, int32_t draftSteps, cudaStream_t stream)
{
    if (batchSize == 0 || draftSteps < 0)
    {
        return;
    }

    int32_t const verifyTokenCount = draftSteps + 1;
    int32_t const totalElements = batchSize * verifyTokenCount;
    constexpr int32_t kThreads = 256;
    int32_t const blocks = (totalElements + kThreads - 1) / kThreads;
    buildVerifyTokensKernel<<<blocks, kThreads, 0, stream>>>(
        seedTokenIds, draftTokenIds, verifyTokenIds, draftSteps, verifyTokenCount, totalElements);
    CUDA_CHECK(cudaGetLastError());
}

void launchGemma4MTPGatherSeedHidden(void const* sourceHiddenStates, void* seedHiddenStates,
    int32_t const* sourceTokenIndices, int32_t batchSize, int64_t sourceSeqLen, int64_t hiddenSize, size_t elementBytes,
    cudaStream_t stream)
{
    if (batchSize == 0 || hiddenSize == 0 || elementBytes == 0)
    {
        return;
    }

    int64_t const rowBytes = hiddenSize * static_cast<int64_t>(elementBytes);
    int64_t const totalBytes = static_cast<int64_t>(batchSize) * rowBytes;
    constexpr int32_t kThreads = 256;
    int64_t const blockCount = (totalBytes + kThreads - 1) / kThreads;
    check::check(blockCount <= std::numeric_limits<int32_t>::max(),
        "Gemma4 MTP seed-hidden gather exceeds CUDA grid dimension limit.");
    int32_t const blocks = static_cast<int32_t>(blockCount);
    gatherSeedHiddenKernel<<<blocks, kThreads, 0, stream>>>(static_cast<uint8_t const*>(sourceHiddenStates),
        static_cast<uint8_t*>(seedHiddenStates), sourceTokenIndices, sourceSeqLen, rowBytes, totalBytes);
    CUDA_CHECK(cudaGetLastError());
}

void launchGemma4MTPStoreDraftToken(int32_t const* selectedTokenIds, int32_t* draftTokenIds, int32_t batchSize,
    int32_t draftSteps, int32_t step, cudaStream_t stream)
{
    if (batchSize == 0)
    {
        return;
    }

    check::check(step >= 0 && step < draftSteps, "Gemma4 MTP draft-token store step is out of range.");
    constexpr int32_t kThreads = 256;
    int32_t const blocks = (batchSize + kThreads - 1) / kThreads;
    storeDraftTokenKernel<<<blocks, kThreads, 0, stream>>>(
        selectedTokenIds, draftTokenIds, draftSteps, step, batchSize);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace kernel
} // namespace trt_edgellm
