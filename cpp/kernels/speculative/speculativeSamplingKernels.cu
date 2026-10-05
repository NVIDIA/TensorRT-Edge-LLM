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

#include "kernels/speculative/speculativeSamplingKernels.h"

#include "common/cudaUtils.h"

#include <cfloat>
#include <cmath>
#include <cub/cub.cuh>

namespace trt_edgellm::kernel::detail
{
namespace
{

constexpr int32_t kWarpSize{32};
constexpr int32_t kMaxSparseSupport{128};

struct BlockPrefixCallbackOp
{
    float runningTotal;

    __device__ explicit BlockPrefixCallbackOp(float initial)
        : runningTotal(initial)
    {
    }

    __device__ float operator()(float aggregate)
    {
        float const prefix = runningTotal;
        runningTotal += aggregate;
        return prefix;
    }
};

__device__ float warpSum(float value)
{
    for (int32_t offset = 16; offset > 0; offset >>= 1)
    {
        value += __shfl_down_sync(0xFFFFFFFFU, value, offset);
    }
    return __shfl_sync(0xFFFFFFFFU, value, 0);
}

__global__ void normalizeTopKTopPKernel(
    float const* values, float* probabilities, int32_t topK, float invTemperature, float topP)
{
    int32_t const row = static_cast<int32_t>(blockIdx.x);
    int32_t const lane = static_cast<int32_t>(threadIdx.x);
    __shared__ float probs[kMaxSparseSupport];
    __shared__ int32_t retained;
    __shared__ float retainedSum;

    float localMax = -FLT_MAX;
    for (int32_t k = lane; k < topK; k += kWarpSize)
    {
        localMax = fmaxf(localMax, values[row * topK + k] * invTemperature);
    }
    for (int32_t offset = 16; offset > 0; offset >>= 1)
    {
        localMax = fmaxf(localMax, __shfl_down_sync(0xFFFFFFFFU, localMax, offset));
    }
    float const maxValue = __shfl_sync(0xFFFFFFFFU, localMax, 0);
    float localSum = 0.0F;
    for (int32_t k = lane; k < topK; k += kWarpSize)
    {
        float const value = expf(values[row * topK + k] * invTemperature - maxValue);
        probs[k] = value;
        localSum += value;
    }
    float const total = fmaxf(warpSum(localSum), 1.0e-20F);
    __syncwarp();
    if (lane == 0)
    {
        float cumulative = 0.0F;
        retained = topK;
        for (int32_t k = 0; k < topK; ++k)
        {
            probs[k] /= total;
            cumulative += probs[k];
            if (cumulative >= topP)
            {
                retained = k + 1;
                break;
            }
        }
        retainedSum = 0.0F;
        for (int32_t k = 0; k < retained; ++k)
        {
            retainedSum += probs[k];
        }
        retainedSum = fmaxf(retainedSum, 1.0e-20F);
    }
    __syncwarp();
    for (int32_t k = lane; k < topK; k += kWarpSize)
    {
        probabilities[row * topK + k] = k < retained ? probs[k] / retainedSum : 0.0F;
    }
}

__global__ void sparseAcceptKernel(float const* targetProbabilities, int32_t const* targetIds,
    float const* proposalProbabilities, int32_t const* proposalIds, int32_t const* proposalTokenIds,
    int32_t const* proposalLengths, float const* uniforms, int32_t* acceptedTokenIds, int32_t* acceptLength,
    int32_t* acceptedTokenIndices, int32_t proposalStride, int32_t verifyProposalLen, int32_t targetSupportSize,
    int32_t proposalSupportSize, int32_t const* maxAcceptLengths)
{
    int32_t const batch = static_cast<int32_t>(blockIdx.x);
    int32_t const lane = static_cast<int32_t>(threadIdx.x);
    int32_t const verifyLen = verifyProposalLen + 1;
    int32_t const uniformStride = 2 * proposalStride + 1;
    __shared__ float residualWeights[kMaxSparseSupport];
    __shared__ int32_t residualIds[kMaxSparseSupport];
    __shared__ int32_t acceptedCount;
    __shared__ int32_t stop;

    if (lane == 0)
    {
        acceptedCount = 0;
        stop = 0;
    }
    for (int32_t pos = lane; pos < verifyLen; pos += kWarpSize)
    {
        acceptedTokenIds[batch * verifyLen + pos] = 0;
        if (acceptedTokenIndices != nullptr)
        {
            acceptedTokenIndices[batch * verifyLen + pos] = pos;
        }
    }
    __syncwarp();

    int32_t const rowProposalLen = max(0, min(verifyProposalLen, proposalLengths[batch]));
    for (int32_t step = 0; step < rowProposalLen; ++step)
    {
        int32_t const proposalToken = proposalTokenIds[batch * proposalStride + step];
        int64_t const targetOffset = (static_cast<int64_t>(batch) * verifyLen + step) * targetSupportSize;
        int64_t const proposalOffset = (static_cast<int64_t>(batch) * proposalStride + step) * proposalSupportSize;
        float pSelected = 0.0F;
        float qSelected = 0.0F;
        for (int32_t k = lane; k < targetSupportSize; k += kWarpSize)
        {
            pSelected += targetIds[targetOffset + k] == proposalToken ? targetProbabilities[targetOffset + k] : 0.0F;
        }
        for (int32_t k = lane; k < proposalSupportSize; k += kWarpSize)
        {
            qSelected
                += proposalIds[proposalOffset + k] == proposalToken ? proposalProbabilities[proposalOffset + k] : 0.0F;
        }
        pSelected = warpSum(pSelected);
        qSelected = warpSum(qSelected);
        if (lane == 0)
        {
            float const ratio = isfinite(pSelected) && isfinite(qSelected) && qSelected > 1.0e-20F
                ? fminf(1.0F, fmaxf(0.0F, pSelected / qSelected))
                : 0.0F;
            if (uniforms[batch * uniformStride + step] <= ratio)
            {
                acceptedTokenIds[batch * verifyLen + acceptedCount] = proposalToken;
                ++acceptedCount;
            }
            else
            {
                stop = 1;
            }
        }
        __syncwarp();
        if (stop == 0)
        {
            continue;
        }

        float localResidualSum = 0.0F;
        for (int32_t k = lane; k < targetSupportSize; k += kWarpSize)
        {
            int32_t const token = targetIds[targetOffset + k];
            float q = 0.0F;
            for (int32_t j = 0; j < proposalSupportSize; ++j)
            {
                q += proposalIds[proposalOffset + j] == token ? proposalProbabilities[proposalOffset + j] : 0.0F;
            }
            q = isfinite(q) ? q : 0.0F;
            float const weight = fmaxf(targetProbabilities[targetOffset + k] - q, 0.0F);
            residualWeights[k] = weight;
            residualIds[k] = token;
            localResidualSum += weight;
        }
        float const residualSum = warpSum(localResidualSum);
        __syncwarp();
        if (lane == 0)
        {
            float threshold = uniforms[batch * uniformStride + proposalStride + step] * residualSum;
            int32_t sampled = residualIds[0];
            if (residualSum <= 1.0e-20F)
            {
                float best = -1.0F;
                for (int32_t k = 0; k < targetSupportSize; ++k)
                {
                    if (targetProbabilities[targetOffset + k] > best)
                    {
                        best = targetProbabilities[targetOffset + k];
                        sampled = targetIds[targetOffset + k];
                    }
                }
            }
            else
            {
                for (int32_t k = 0; k < targetSupportSize; ++k)
                {
                    threshold -= residualWeights[k];
                    if (threshold <= 0.0F)
                    {
                        sampled = residualIds[k];
                        break;
                    }
                }
            }
            acceptedTokenIds[batch * verifyLen + acceptedCount] = sampled;
            int32_t const accepted = acceptedCount + 1;
            acceptLength[batch]
                = maxAcceptLengths == nullptr ? accepted : max(0, min(accepted, maxAcceptLengths[batch]));
        }
        return;
    }

    int64_t const bonusOffset = (static_cast<int64_t>(batch) * verifyLen + rowProposalLen) * targetSupportSize;
    if (lane == 0)
    {
        float threshold = uniforms[batch * uniformStride + 2 * proposalStride];
        int32_t sampled = targetIds[bonusOffset];
        for (int32_t k = 0; k < targetSupportSize; ++k)
        {
            threshold -= targetProbabilities[bonusOffset + k];
            if (threshold <= 0.0F)
            {
                sampled = targetIds[bonusOffset + k];
                break;
            }
        }
        acceptedTokenIds[batch * verifyLen + acceptedCount] = sampled;
        int32_t const accepted = acceptedCount + 1;
        acceptLength[batch] = maxAcceptLengths == nullptr ? accepted : max(0, min(accepted, maxAcceptLengths[batch]));
    }
}

template <int32_t BLOCK_SIZE>
__global__ void denseTargetAcceptKernel(float const* targetProbabilities, float const* proposalProbabilities,
    int32_t const* proposalIds, int32_t const* proposalTokenIds, int32_t const* proposalLengths, float const* uniforms,
    int32_t* acceptedTokenIds, int32_t* acceptLength, int32_t* acceptedTokenIndices, int32_t proposalStride,
    int32_t verifyProposalLen, int32_t vocabSize, int32_t proposalSupportSize, int32_t const* maxAcceptLengths)
{
    int32_t const batch = static_cast<int32_t>(blockIdx.x);
    int32_t const lane = static_cast<int32_t>(threadIdx.x);
    int32_t const verifyLen = verifyProposalLen + 1;
    int32_t const uniformStride = 2 * proposalStride + 1;
    using BlockScan = cub::BlockScan<float, BLOCK_SIZE>;
    __shared__ typename BlockScan::TempStorage scanStorage;
    __shared__ int32_t supportIds[kMaxSparseSupport];
    __shared__ float supportProbabilities[kMaxSparseSupport];
    __shared__ int32_t acceptedCount;
    __shared__ int32_t stop;
    __shared__ int32_t sampledToken;
    __shared__ float threshold;
    __shared__ float residualMass;

    if (lane == 0)
    {
        acceptedCount = 0;
        stop = 0;
    }
    for (int32_t pos = lane; pos < verifyLen; pos += BLOCK_SIZE)
    {
        acceptedTokenIds[batch * verifyLen + pos] = 0;
        if (acceptedTokenIndices != nullptr)
        {
            acceptedTokenIndices[batch * verifyLen + pos] = pos;
        }
    }
    __syncthreads();

    int32_t const rowProposalLen = max(0, min(verifyProposalLen, proposalLengths[batch]));
    for (int32_t step = 0; step < rowProposalLen; ++step)
    {
        int64_t const targetOffset = (static_cast<int64_t>(batch) * verifyLen + step) * vocabSize;
        int64_t const proposalOffset = (static_cast<int64_t>(batch) * proposalStride + step) * proposalSupportSize;
        for (int32_t k = lane; k < proposalSupportSize; k += BLOCK_SIZE)
        {
            supportIds[k] = proposalIds[proposalOffset + k];
            supportProbabilities[k] = proposalProbabilities[proposalOffset + k];
        }
        __syncthreads();
        if (lane == 0)
        {
            int32_t const proposalToken = proposalTokenIds[batch * proposalStride + step];
            float qSelected = 0.0F;
            for (int32_t k = 0; k < proposalSupportSize; ++k)
            {
                qSelected += supportIds[k] == proposalToken ? supportProbabilities[k] : 0.0F;
            }
            float const pSelected = proposalToken >= 0 && proposalToken < vocabSize
                ? targetProbabilities[targetOffset + proposalToken]
                : 0.0F;
            float const ratio = isfinite(pSelected) && isfinite(qSelected) && qSelected > 1.0e-20F
                ? fminf(1.0F, fmaxf(0.0F, pSelected / qSelected))
                : 0.0F;
            if (uniforms[batch * uniformStride + step] <= ratio)
            {
                acceptedTokenIds[batch * verifyLen + acceptedCount] = proposalToken;
                ++acceptedCount;
            }
            else
            {
                stop = 1;
            }
        }
        __syncthreads();
        if (stop == 0)
        {
            continue;
        }

        float overlap = 0.0F;
        if (lane < kWarpSize)
        {
            for (int32_t k = lane; k < proposalSupportSize; k += kWarpSize)
            {
                int32_t const token = supportIds[k];
                float const q = isfinite(supportProbabilities[k]) ? fmaxf(supportProbabilities[k], 0.0F) : 0.0F;
                float const p = token >= 0 && token < vocabSize ? targetProbabilities[targetOffset + token] : 0.0F;
                overlap += fminf(fmaxf(p, 0.0F), q);
            }
            overlap = warpSum(overlap);
        }
        if (lane == 0)
        {
            residualMass = fmaxf(1.0F - overlap, 0.0F);
            sampledToken = vocabSize;
            threshold = uniforms[batch * uniformStride + proposalStride + step]
                * (residualMass > 1.0e-20F ? residualMass : 1.0F);
        }
        __syncthreads();

        BlockPrefixCallbackOp prefix(0.0F);
        for (int32_t begin = 0; begin < vocabSize; begin += BLOCK_SIZE)
        {
            int32_t const token = begin + lane;
            float weight = token < vocabSize ? targetProbabilities[targetOffset + token] : 0.0F;
            if (residualMass > 1.0e-20F && token < vocabSize)
            {
                float q = 0.0F;
                for (int32_t k = 0; k < proposalSupportSize; ++k)
                {
                    q += supportIds[k] == token ? supportProbabilities[k] : 0.0F;
                }
                weight = fmaxf(weight - (isfinite(q) ? q : 0.0F), 0.0F);
            }
            float cumulative{0.0F};
            BlockScan(scanStorage).InclusiveSum(weight, cumulative, prefix);
            int32_t const validCount = min(BLOCK_SIZE, vocabSize - begin);
            int32_t const crossingCount = __syncthreads_count(token < vocabSize && cumulative > threshold);
            if (crossingCount > 0)
            {
                int32_t const crossingLane = validCount - crossingCount;
                if (lane == crossingLane)
                {
                    sampledToken = token;
                }
                __syncthreads();
                break;
            }
        }
        if (lane == 0)
        {
            sampledToken = min(sampledToken, vocabSize - 1);
            acceptedTokenIds[batch * verifyLen + acceptedCount] = sampledToken;
            int32_t const accepted = acceptedCount + 1;
            acceptLength[batch]
                = maxAcceptLengths == nullptr ? accepted : max(0, min(accepted, maxAcceptLengths[batch]));
        }
        return;
    }

    int64_t const bonusOffset = (static_cast<int64_t>(batch) * verifyLen + rowProposalLen) * vocabSize;
    if (lane == 0)
    {
        sampledToken = vocabSize;
        threshold = uniforms[batch * uniformStride + 2 * proposalStride];
    }
    __syncthreads();
    BlockPrefixCallbackOp prefix(0.0F);
    for (int32_t begin = 0; begin < vocabSize; begin += BLOCK_SIZE)
    {
        int32_t const token = begin + lane;
        float const weight = token < vocabSize ? targetProbabilities[bonusOffset + token] : 0.0F;
        float cumulative{0.0F};
        BlockScan(scanStorage).InclusiveSum(weight, cumulative, prefix);
        int32_t const validCount = min(BLOCK_SIZE, vocabSize - begin);
        int32_t const crossingCount = __syncthreads_count(token < vocabSize && cumulative > threshold);
        if (crossingCount > 0)
        {
            int32_t const crossingLane = validCount - crossingCount;
            if (lane == crossingLane)
            {
                sampledToken = token;
            }
            __syncthreads();
            break;
        }
    }
    if (lane == 0)
    {
        sampledToken = min(sampledToken, vocabSize - 1);
        acceptedTokenIds[batch * verifyLen + acceptedCount] = sampledToken;
        int32_t const accepted = acceptedCount + 1;
        acceptLength[batch] = maxAcceptLengths == nullptr ? accepted : max(0, min(accepted, maxAcceptLengths[batch]));
    }
}

} // namespace

void launchSparseAccept(float const* targetProbabilities, int32_t const* targetIds, float const* proposalProbabilities,
    int32_t const* proposalIds, int32_t const* proposalTokenIds, int32_t const* proposalLengths, float const* uniforms,
    int32_t* acceptedTokenIds, int32_t* acceptLength, int32_t* acceptedTokenIndices, int32_t batchSize,
    int32_t proposalStride, int32_t verifyProposalLen, int32_t targetSupportSize, int32_t proposalSupportSize,
    cudaStream_t stream, int32_t const* maxAcceptLengths)
{
    sparseAcceptKernel<<<batchSize, kWarpSize, 0, stream>>>(targetProbabilities, targetIds, proposalProbabilities,
        proposalIds, proposalTokenIds, proposalLengths, uniforms, acceptedTokenIds, acceptLength, acceptedTokenIndices,
        proposalStride, verifyProposalLen, targetSupportSize, proposalSupportSize, maxAcceptLengths);
    CUDA_CHECK(cudaGetLastError());
}

void launchDenseTargetAccept(float const* targetProbabilities, float const* proposalProbabilities,
    int32_t const* proposalIds, int32_t const* proposalTokenIds, int32_t const* proposalLengths, float const* uniforms,
    int32_t* acceptedTokenIds, int32_t* acceptLength, int32_t* acceptedTokenIndices, int32_t batchSize,
    int32_t proposalStride, int32_t verifyProposalLen, int32_t vocabSize, int32_t proposalSupportSize,
    cudaStream_t stream, int32_t const* maxAcceptLengths)
{
    constexpr int32_t kBlockSize{256};
    denseTargetAcceptKernel<kBlockSize><<<batchSize, kBlockSize, 0, stream>>>(targetProbabilities,
        proposalProbabilities, proposalIds, proposalTokenIds, proposalLengths, uniforms, acceptedTokenIds, acceptLength,
        acceptedTokenIndices, proposalStride, verifyProposalLen, vocabSize, proposalSupportSize, maxAcceptLengths);
    CUDA_CHECK(cudaGetLastError());
}

void launchNormalizeTopKTopP(float const* topKValues, float* probabilities, int32_t rows, int32_t topK,
    float temperature, float topP, cudaStream_t stream)
{
    normalizeTopKTopPKernel<<<rows, kWarpSize, 0, stream>>>(
        topKValues, probabilities, topK, 1.0F / fmaxf(temperature, 1.0e-6F), fminf(fmaxf(topP, 0.0F), 1.0F));
    CUDA_CHECK(cudaGetLastError());
}

} // namespace trt_edgellm::kernel::detail
