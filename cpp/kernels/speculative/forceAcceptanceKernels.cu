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

#include "common/checkMacros.h"
#include "forceAcceptanceKernels.h"
#include <cub/cub.cuh>

namespace trt_edgellm::kernel::detail
{
namespace
{

constexpr int32_t kBlockSize = 256;

__device__ int32_t treeDepth(int8_t const* mask, int32_t node, int32_t size)
{
    int32_t depth = 0;
    for (int32_t i = 0; i < size; ++i)
    {
        depth += mask[node * size + i] == 1;
    }
    return depth;
}

__global__ void forceTreeAcceptance(int32_t const* top1, int32_t const* tokens, int8_t const* mask, int32_t* output,
    int32_t* indices, int32_t* lengths, int32_t size, int32_t stride, int32_t drafts)
{
    int32_t const batch = blockIdx.x;
    top1 += batch * size;
    tokens += batch * size;
    mask += batch * size * size;
    output += batch * stride;
    indices += batch * stride;
    for (int32_t i = threadIdx.x; i < stride; i += blockDim.x)
    {
        output[i] = 0;
        indices[i] = -1;
    }
    extern __shared__ int32_t depths[];
    for (int32_t node = threadIdx.x; node < size; node += blockDim.x)
    {
        depths[node] = treeDepth(mask, node, size);
    }
    __syncthreads();
    if (threadIdx.x != 0)
    {
        return;
    }
    int32_t endpoint = -1;
    int32_t const rootDepth = depths[0];
    for (int32_t node = 0; node < size; ++node)
    {
        if (mask[node * size] == 1 && depths[node] == rootDepth + drafts)
        {
            endpoint = node;
            break;
        }
    }
    int32_t previous = 0;
    bool valid = endpoint >= 0;
    for (int32_t depth = 1; valid && depth <= drafts; ++depth)
    {
        int32_t next = -1;
        for (int32_t node = 1; node < size; ++node)
        {
            if (mask[endpoint * size + node] == 1 && mask[node * size + previous] == 1
                && depths[node] == rootDepth + depth)
            {
                next = node;
                break;
            }
        }
        valid = next >= 0;
        if (valid)
        {
            output[depth - 1] = tokens[next];
            indices[depth - 1] = previous;
            previous = next;
        }
    }
    if (valid)
    {
        output[drafts] = top1[previous];
        indices[drafts] = previous;
        lengths[batch] = drafts + 1;
    }
    else
    {
        // Persistent state commits a valid root only; failure is carried outside the accepted span.
        output[0] = top1[0];
        indices[0] = 0;
        lengths[batch] = 1;
        output[stride - 1] = -1;
    }
}

__global__ void forceGreedyAcceptance(int32_t const* top1, int32_t const* proposals, int32_t const* proposalLengths,
    int32_t* output, int32_t* indices, int32_t* lengths, int32_t proposalStride, int32_t proposalOffset,
    int32_t verifySize, int32_t drafts)
{
    int32_t const batch = blockIdx.x;
    int32_t const available = proposalLengths == nullptr ? verifySize - 1 : proposalLengths[batch];
    bool const valid = drafts <= available;
    int32_t const count = valid ? drafts : 0;
    for (int32_t i = 0; i < verifySize; ++i)
    {
        output[batch * verifySize + i] = i < count ? proposals[batch * proposalStride + proposalOffset + i] : 0;
        if (indices != nullptr)
        {
            indices[batch * verifySize + i] = i <= count ? i : -1;
        }
    }
    output[batch * verifySize + count] = top1[batch * verifySize + count];
    lengths[batch] = count + 1;
    if (!valid)
    {
        output[(batch + 1) * verifySize - 1] = -1;
    }
}

__global__ void forceSamplingAcceptance(float const* probabilities, int32_t const* targetIds, int32_t const* proposals,
    int32_t const* proposalLengths, float const* uniforms, int32_t* output, int32_t* indices, int32_t* lengths,
    int32_t proposalStride, int32_t verifySize, int32_t supportSize, int32_t const* budgets, int32_t drafts)
{
    int32_t const batch = blockIdx.x;
    bool const valid = drafts <= proposalLengths[batch];
    int32_t const count = valid ? drafts : 0;
    for (int32_t i = threadIdx.x; i < verifySize; i += blockDim.x)
    {
        output[batch * verifySize + i] = i < count ? proposals[batch * proposalStride + i] : 0;
        if (indices != nullptr)
        {
            indices[batch * verifySize + i] = i <= count ? i : -1;
        }
    }
    int64_t const offset = (static_cast<int64_t>(batch) * verifySize + count) * supportSize;
    float const threshold = uniforms[batch * (2 * proposalStride + 1) + 2 * proposalStride];
    using BlockScan = cub::BlockScan<float, kBlockSize>;
    __shared__ typename BlockScan::TempStorage scanStorage;
    __shared__ int32_t sampled;
    if (threadIdx.x == 0)
    {
        sampled = supportSize - 1;
    }
    __syncthreads();
    float accumulated = 0.0F;
    for (int32_t begin = 0; begin < supportSize; begin += kBlockSize)
    {
        int32_t const i = begin + threadIdx.x;
        float const probability = i < supportSize ? probabilities[offset + i] : 0.0F;
        float prefix;
        float aggregate;
        BlockScan(scanStorage).InclusiveSum(probability, prefix, aggregate);
        if (i < supportSize
            && (targetIds == nullptr ? accumulated + prefix > threshold : accumulated + prefix >= threshold))
        {
            atomicMin(&sampled, i);
        }
        accumulated += aggregate;
        __syncthreads();
        if (targetIds == nullptr ? accumulated > threshold : accumulated >= threshold)
        {
            break;
        }
    }
    if (threadIdx.x != 0)
    {
        return;
    }
    output[batch * verifySize + count] = targetIds == nullptr ? sampled : targetIds[offset + sampled];
    lengths[batch] = budgets == nullptr ? count + 1 : max(0, min(count + 1, budgets[batch]));
    if (!valid)
    {
        output[(batch + 1) * verifySize - 1] = -1;
    }
}

} // namespace

void launchForceTreeAcceptance(int32_t const* top1Tokens, int32_t const* tokenIds, int8_t const* attentionMask,
    int32_t* acceptedTokenIds, int32_t* acceptedIndices, int32_t* acceptLength, int32_t batchSize, int32_t verifySize,
    int32_t outputStride, int32_t forcedAcceptanceLength, cudaStream_t stream)
{
    forceTreeAcceptance<<<batchSize, kBlockSize, verifySize * sizeof(int32_t), stream>>>(top1Tokens, tokenIds,
        attentionMask, acceptedTokenIds, acceptedIndices, acceptLength, verifySize, outputStride,
        forcedAcceptanceLength - 1);
    CUDA_CHECK(cudaGetLastError());
}

void launchForceGreedyAcceptance(int32_t const* top1Tokens, int32_t const* proposalTokenIds,
    int32_t const* proposalLengths, int32_t* acceptedTokenIds, int32_t* acceptedIndices, int32_t* acceptLength,
    int32_t batchSize, int32_t proposalStride, int32_t proposalOffset, int32_t verifySize,
    int32_t forcedAcceptanceLength, cudaStream_t stream)
{
    forceGreedyAcceptance<<<batchSize, 1, 0, stream>>>(top1Tokens, proposalTokenIds, proposalLengths, acceptedTokenIds,
        acceptedIndices, acceptLength, proposalStride, proposalOffset, verifySize, forcedAcceptanceLength - 1);
    CUDA_CHECK(cudaGetLastError());
}

void launchForceSamplingAcceptance(float const* targetProbabilities, int32_t const* targetIds,
    int32_t const* proposalTokenIds, int32_t const* proposalLengths, float const* uniforms, int32_t* acceptedTokenIds,
    int32_t* acceptedIndices, int32_t* acceptLength, int32_t batchSize, int32_t proposalStride, int32_t verifySize,
    int32_t targetSupportSize, int32_t const* maxAcceptLengths, int32_t forcedAcceptanceLength, cudaStream_t stream)
{
    forceSamplingAcceptance<<<batchSize, kBlockSize, 0, stream>>>(targetProbabilities, targetIds, proposalTokenIds,
        proposalLengths, uniforms, acceptedTokenIds, acceptedIndices, acceptLength, proposalStride, verifySize,
        targetSupportSize, maxAcceptLengths, forcedAcceptanceLength - 1);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace trt_edgellm::kernel::detail
