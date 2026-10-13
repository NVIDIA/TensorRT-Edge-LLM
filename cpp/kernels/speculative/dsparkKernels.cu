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
#include "common/cudaMacros.h"
#include "kernels/speculative/dsparkKernels.h"
#include "kernels/speculative/speculativeKernelsUtils.h"

#include <algorithm>
#include <cfloat>
#include <cub/cub.cuh>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <string>

namespace trt_edgellm
{
namespace kernel
{
namespace
{

static constexpr int32_t kProbabilityBlockSize = 256;
static constexpr int32_t kMaxParallelTopK = 128;
static constexpr int32_t kMarkovWarpsPerBlock = 16;
static constexpr int32_t kMarkovBlockSize = kMarkovWarpsPerBlock * 32;
static constexpr float kSPSDraftCost = 0.50F;
static constexpr float kSPSVerifyBaseCost = 1.00F;
static constexpr float kSPSVerifyTokenCost = 0.45F;
static constexpr float kSPSAcceptCost = 0.02F;

// Keep greedy verifier top-1 tie behavior aligned with EAGLE/DFlash.
struct DSparkTop1Helper
{
    float value;
    int32_t index;

    __device__ __forceinline__ DSparkTop1Helper()
        : value(-FLT_MAX)
        , index(-1)
    {
    }

    __device__ __forceinline__ void update(float elem, int32_t elemId)
    {
        if (elem > value || (elem == value && (index < 0 || elemId < index)))
        {
            value = elem;
            index = elemId;
        }
    }
};

struct DSparkTop1MaxOp
{
    __device__ __forceinline__ DSparkTop1Helper operator()(DSparkTop1Helper const& a, DSparkTop1Helper const& b) const
    {
        if (a.index < 0)
        {
            return b;
        }
        if (b.index < 0)
        {
            return a;
        }
        return (b.value > a.value || (b.value == a.value && b.index < a.index)) ? b : a;
    }
};

__device__ __forceinline__ float dsparkInvTemperature(float temperature)
{
    return (temperature < 1e-3F) ? 1.0F : 1.0F / temperature;
}

__device__ __forceinline__ uint64_t dsparkSplitMix64(uint64_t value)
{
    value += 0x9E3779B97F4A7C15ULL;
    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ULL;
    value = (value ^ (value >> 27)) * 0x94D049BB133111EBULL;
    return value ^ (value >> 31);
}

__device__ __forceinline__ float dsparkUniformFromCounter(uint64_t seed, uint64_t offset, uint64_t counter)
{
    uint64_t const mixed = dsparkSplitMix64(seed ^ (offset + 0xD1B54A32D192ED03ULL * (counter + 1ULL)));
    uint32_t const mantissa = static_cast<uint32_t>(mixed >> 40);
    return clampUniform(static_cast<float>(mantissa) * (1.0F / 16777216.0F));
}

__device__ int32_t dsparkSampleFromProbs(float const* probs, int32_t vocabSize, float uniform)
{
    float const target = clampUniform(uniform);
    float cumulative = 0.0F;
    int32_t fallback = 0;
    for (int32_t vocabIdx = 0; vocabIdx < vocabSize; ++vocabIdx)
    {
        float const prob = probs[vocabIdx];
        if (prob <= 0.0F)
        {
            continue;
        }
        fallback = vocabIdx;
        cumulative += prob;
        if (target < cumulative)
        {
            return vocabIdx;
        }
    }
    return fallback;
}

__device__ int32_t dsparkSampleFromResidual(
    float const* targetProbs, float const* draftProbs, int32_t vocabSize, float uniform)
{
    float residualSum = 0.0F;
    for (int32_t vocabIdx = 0; vocabIdx < vocabSize; ++vocabIdx)
    {
        residualSum += fmaxf(targetProbs[vocabIdx] - draftProbs[vocabIdx], 0.0F);
    }
    if (residualSum <= 1e-20F)
    {
        return dsparkSampleFromProbs(targetProbs, vocabSize, uniform);
    }

    float const target = clampUniform(uniform) * residualSum;
    float cumulative = 0.0F;
    int32_t fallback = 0;
    for (int32_t vocabIdx = 0; vocabIdx < vocabSize; ++vocabIdx)
    {
        float const residual = fmaxf(targetProbs[vocabIdx] - draftProbs[vocabIdx], 0.0F);
        if (residual <= 0.0F)
        {
            continue;
        }
        fallback = vocabIdx;
        cumulative += residual;
        if (target < cumulative)
        {
            return vocabIdx;
        }
    }
    return fallback;
}

__device__ void dsparkApplyTopKTopP(float* probs, int32_t vocabSize, int32_t topK, float topP)
{
    int32_t effectiveTopK = topK;
    if (effectiveTopK < 0 || effectiveTopK > vocabSize)
    {
        effectiveTopK = vocabSize;
    }
    bool const useTopK = effectiveTopK > 0 && effectiveTopK < vocabSize;
    bool const useTopP = topP < 1.0F - 1e-6F;
    if (!useTopK && !useTopP)
    {
        return;
    }

    if (useTopK)
    {
        float selectedSum = 0.0F;
        for (int32_t selected = 0; selected < effectiveTopK; ++selected)
        {
            float bestProb = -1.0F;
            int32_t bestIdx = -1;
            for (int32_t vocabIdx = 0; vocabIdx < vocabSize; ++vocabIdx)
            {
                float const prob = probs[vocabIdx];
                if (prob > bestProb)
                {
                    bestProb = prob;
                    bestIdx = vocabIdx;
                }
            }
            if (bestIdx < 0 || bestProb <= 0.0F)
            {
                break;
            }
            selectedSum += bestProb;
            probs[bestIdx] = -bestProb;
        }

        for (int32_t vocabIdx = 0; vocabIdx < vocabSize; ++vocabIdx)
        {
            if (probs[vocabIdx] >= 0.0F)
            {
                probs[vocabIdx] = 0.0F;
            }
        }

        if (selectedSum <= 1e-20F)
        {
            return;
        }

        if (!useTopP)
        {
            for (int32_t vocabIdx = 0; vocabIdx < vocabSize; ++vocabIdx)
            {
                if (probs[vocabIdx] < 0.0F)
                {
                    probs[vocabIdx] = -probs[vocabIdx] / selectedSum;
                }
            }
            return;
        }

        float const thresholdMass = fmaxf(topP, 1e-20F) * selectedSum;
        float remaining = thresholdMass;
        for (int32_t selected = 0; selected < effectiveTopK && remaining > 1e-20F; ++selected)
        {
            float bestProb = -1.0F;
            int32_t bestIdx = -1;
            for (int32_t vocabIdx = 0; vocabIdx < vocabSize; ++vocabIdx)
            {
                float const prob = -probs[vocabIdx];
                if (probs[vocabIdx] < 0.0F && prob > bestProb)
                {
                    bestProb = prob;
                    bestIdx = vocabIdx;
                }
            }
            if (bestIdx < 0 || bestProb <= 0.0F)
            {
                break;
            }
            float const assigned = fminf(bestProb, remaining);
            probs[bestIdx] = assigned / thresholdMass;
            remaining -= assigned;
        }
        for (int32_t vocabIdx = 0; vocabIdx < vocabSize; ++vocabIdx)
        {
            if (probs[vocabIdx] < 0.0F)
            {
                probs[vocabIdx] = 0.0F;
            }
        }
        return;
    }

    float const thresholdMass = fmaxf(topP, 1e-20F);
    float remaining = thresholdMass;
    while (remaining > 1e-20F)
    {
        float bestProb = -1.0F;
        int32_t bestIdx = -1;
        for (int32_t vocabIdx = 0; vocabIdx < vocabSize; ++vocabIdx)
        {
            float const prob = probs[vocabIdx];
            if (prob > bestProb)
            {
                bestProb = prob;
                bestIdx = vocabIdx;
            }
        }
        if (bestIdx < 0 || bestProb <= 0.0F)
        {
            break;
        }
        float const assigned = fminf(bestProb, remaining);
        probs[bestIdx] = -assigned / thresholdMass;
        remaining -= assigned;
    }
    for (int32_t vocabIdx = 0; vocabIdx < vocabSize; ++vocabIdx)
    {
        probs[vocabIdx] = probs[vocabIdx] < 0.0F ? -probs[vocabIdx] : 0.0F;
    }
}

__device__ void dsparkNormalizeLogitsRow(
    float const* logits, float* probs, int32_t vocabSize, float temperature, int32_t topK, float topP)
{
    int32_t const effectiveTopK = temperature < 1e-3F ? 1 : topK;
    float const effectiveTopP = temperature < 1e-3F ? 1.0F : topP;
    float const invTemp = dsparkInvTemperature(temperature);
    float maxLogit = -FLT_MAX;
    for (int32_t vocabIdx = 0; vocabIdx < vocabSize; ++vocabIdx)
    {
        maxLogit = fmaxf(maxLogit, logits[vocabIdx] * invTemp);
    }

    float sumExp = 0.0F;
    for (int32_t vocabIdx = 0; vocabIdx < vocabSize; ++vocabIdx)
    {
        float const expValue = expf(logits[vocabIdx] * invTemp - maxLogit);
        probs[vocabIdx] = expValue;
        sumExp += expValue;
    }

    if (sumExp <= 0.0F || !isfinite(sumExp))
    {
        float const uniform = 1.0F / static_cast<float>(vocabSize);
        for (int32_t vocabIdx = 0; vocabIdx < vocabSize; ++vocabIdx)
        {
            probs[vocabIdx] = uniform;
        }
        return;
    }

    for (int32_t vocabIdx = 0; vocabIdx < vocabSize; ++vocabIdx)
    {
        probs[vocabIdx] /= sumExp;
    }
    dsparkApplyTopKTopP(probs, vocabSize, effectiveTopK, effectiveTopP);
}

__device__ __forceinline__ void reduceMaxPair(float& localMax, int32_t& localIdx)
{
    for (int32_t offset = 16; offset > 0; offset >>= 1)
    {
        float const otherMax = __shfl_down_sync(0xFFFFFFFF, localMax, offset);
        int32_t const otherIdx = __shfl_down_sync(0xFFFFFFFF, localIdx, offset);
        if (otherMax > localMax || (otherMax == localMax && otherIdx < localIdx))
        {
            localMax = otherMax;
            localIdx = otherIdx;
        }
    }

    __shared__ float sMaxVal[32];
    __shared__ int32_t sMaxIdx[32];

    int32_t const warpId = threadIdx.x / 32;
    int32_t const laneId = threadIdx.x % 32;
    int32_t const numWarps = (blockDim.x + 31) / 32;

    if (laneId == 0)
    {
        sMaxVal[warpId] = localMax;
        sMaxIdx[warpId] = localIdx;
    }
    __syncthreads();

    if (warpId == 0)
    {
        float warpMax = (laneId < numWarps) ? sMaxVal[laneId] : -FLT_MAX;
        int32_t warpIdx = (laneId < numWarps) ? sMaxIdx[laneId] : 0;
        for (int32_t offset = 16; offset > 0; offset >>= 1)
        {
            float const otherMax = __shfl_down_sync(0xFFFFFFFF, warpMax, offset);
            int32_t const otherIdx = __shfl_down_sync(0xFFFFFFFF, warpIdx, offset);
            if (otherMax > warpMax || (otherMax == warpMax && otherIdx < warpIdx))
            {
                warpMax = otherMax;
                warpIdx = otherIdx;
            }
        }
        if (laneId == 0)
        {
            sMaxVal[0] = warpMax;
            sMaxIdx[0] = warpIdx;
        }
    }
    __syncthreads();

    localMax = sMaxVal[0];
    localIdx = sMaxIdx[0];
}

__device__ float dsparkReduceSum(float localSum)
{
    for (int32_t offset = 16; offset > 0; offset >>= 1)
    {
        localSum += __shfl_down_sync(0xFFFFFFFF, localSum, offset);
    }

    __shared__ float sSum[32];
    int32_t const warpId = threadIdx.x / 32;
    int32_t const laneId = threadIdx.x % 32;
    int32_t const numWarps = (blockDim.x + 31) / 32;

    if (laneId == 0)
    {
        sSum[warpId] = localSum;
    }
    __syncthreads();

    if (warpId == 0)
    {
        float warpSum = (laneId < numWarps) ? sSum[laneId] : 0.0F;
        for (int32_t offset = 16; offset > 0; offset >>= 1)
        {
            warpSum += __shfl_down_sync(0xFFFFFFFF, warpSum, offset);
        }
        if (laneId == 0)
        {
            sSum[0] = warpSum;
        }
    }
    __syncthreads();
    return sSum[0];
}

__global__ void dsparkBuildVerifyTokensKernel(int32_t const* __restrict__ lastAcceptedTokens, // [B]
    int32_t const* __restrict__ draftTokenIds,                                                // [B, draftStride]
    int32_t* __restrict__ verifyTokenIds, // [B, verifyProposalLen + 1]
    int32_t draftStride, int32_t verifyProposalLen, int32_t totalElements)
{
    int32_t const idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= totalElements)
    {
        return;
    }

    int32_t const verifyLen = verifyProposalLen + 1;
    int32_t const batchIdx = idx / verifyLen;
    int32_t const posIdx = idx % verifyLen;
    if (posIdx == 0)
    {
        verifyTokenIds[batchIdx * verifyLen] = lastAcceptedTokens[batchIdx];
    }
    else
    {
        verifyTokenIds[batchIdx * verifyLen + posIdx] = draftTokenIds[batchIdx * draftStride + posIdx - 1];
    }
}

template <int32_t BLOCK_SIZE>
__global__ void dsparkParallelArgmaxKernel(float const* __restrict__ baseLogits, // [B, VFY, V]
    int32_t* __restrict__ argmaxResults,                                         // [B, VFY]
    int32_t totalPositions, int32_t vocabSize)
{
    using BlockReduce = cub::BlockReduce<DSparkTop1Helper, BLOCK_SIZE>;
    __shared__ typename BlockReduce::TempStorage tempStorage;

    int32_t const posIdx = blockIdx.x;
    if (posIdx >= totalPositions)
    {
        return;
    }

    float const* posLogits = baseLogits + static_cast<int64_t>(posIdx) * vocabSize;
    DSparkTop1Helper partial;

    for (int32_t vocabIdx = threadIdx.x; vocabIdx < vocabSize; vocabIdx += BLOCK_SIZE)
    {
        partial.update(posLogits[vocabIdx], vocabIdx);
    }

    DSparkTop1Helper const blockMax = BlockReduce(tempStorage).Reduce(partial, DSparkTop1MaxOp());
    if (threadIdx.x == 0)
    {
        argmaxResults[posIdx] = blockMax.index != -1 ? blockMax.index : 0;
    }
}

__global__ void dsparkGreedyAcceptWalkKernel(int32_t const* __restrict__ argmaxResults, // [B, VFY]
    int32_t const* __restrict__ draftTokenIds,                                          // [B, draftStride]
    int32_t const* __restrict__ proposalLengths,                                        // [B]
    int32_t* __restrict__ acceptedTokenIds,                                             // [B, VFY]
    int32_t* __restrict__ acceptLength,                                                 // [B]
    int32_t draftStride, int32_t verifyProposalLen)
{
    int32_t const batchIdx = blockIdx.x;
    int32_t const verifyLen = verifyProposalLen + 1;
    int32_t const* batchArgmax = argmaxResults + batchIdx * verifyLen;
    int32_t const* batchDraft = draftTokenIds + batchIdx * draftStride;
    int32_t* batchAccepted = acceptedTokenIds + batchIdx * verifyLen;

    int32_t const rowProposalLen = max(1, min(verifyProposalLen, proposalLengths[batchIdx]));
    int32_t acceptedDraft = 0;
    for (int32_t i = 0; i < rowProposalLen; ++i)
    {
        if (batchArgmax[i] != batchDraft[i])
        {
            break;
        }
        batchAccepted[acceptedDraft] = batchDraft[i];
        ++acceptedDraft;
    }

    // Base bonus token at the first rejected position, or after the selected prefix.
    batchAccepted[acceptedDraft] = batchArgmax[acceptedDraft];
    acceptLength[batchIdx] = acceptedDraft + 1;
}

__global__ void dsparkLogitsToProbabilitiesKernel(float const* __restrict__ logits, float* __restrict__ probabilities,
    int32_t rows, int32_t vocabSize, float temperature, int32_t topK, float topP)
{
    int32_t const rowIdx = blockIdx.x;
    if (rowIdx >= rows || vocabSize <= 0)
    {
        return;
    }

    float const* rowLogits = logits + static_cast<int64_t>(rowIdx) * vocabSize;
    float* rowProbs = probabilities + static_cast<int64_t>(rowIdx) * vocabSize;
    int32_t effectiveTopK = temperature < 1e-3F ? 1 : topK;
    float const effectiveTopP = temperature < 1e-3F ? 1.0F : topP;
    if (effectiveTopK < 0 || effectiveTopK > vocabSize)
    {
        effectiveTopK = vocabSize;
    }
    bool const useTopK = effectiveTopK > 0 && effectiveTopK < vocabSize;
    bool const useTopP = effectiveTopP < 1.0F - 1e-6F;

    // Top-p without a bounded top-k still uses the exact scalar fallback. The
    // production DSpark path exercises top-k/top-k+top-p, which is parallelized below.
    if ((useTopP && !useTopK) || (useTopK && effectiveTopK > kMaxParallelTopK))
    {
        if (threadIdx.x == 0)
        {
            dsparkNormalizeLogitsRow(rowLogits, rowProbs, vocabSize, temperature, topK, topP);
        }
        return;
    }

    float const invTemp = dsparkInvTemperature(temperature);
    if (useTopK)
    {
        for (int32_t vocabIdx = threadIdx.x; vocabIdx < vocabSize; vocabIdx += blockDim.x)
        {
            rowProbs[vocabIdx] = 0.0F;
        }
        __syncthreads();

        __shared__ float selectedLogits[kMaxParallelTopK];
        __shared__ int32_t selectedIndices[kMaxParallelTopK];
        __shared__ float selectedExp[kMaxParallelTopK];
        for (int32_t selected = 0; selected < effectiveTopK; ++selected)
        {
            float localMax = -FLT_MAX;
            int32_t localIdx = 0;
            for (int32_t vocabIdx = threadIdx.x; vocabIdx < vocabSize; vocabIdx += blockDim.x)
            {
                if (rowProbs[vocabIdx] != 0.0F)
                {
                    continue;
                }
                float const val = rowLogits[vocabIdx] * invTemp;
                if (val > localMax || (val == localMax && vocabIdx < localIdx))
                {
                    localMax = val;
                    localIdx = vocabIdx;
                }
            }
            reduceMaxPair(localMax, localIdx);
            if (threadIdx.x == 0)
            {
                selectedLogits[selected] = localMax;
                selectedIndices[selected] = localIdx;
                rowProbs[localIdx] = -1.0F;
            }
            __syncthreads();
        }

        float const maxLogit = selectedLogits[0];
        float localSum = 0.0F;
        for (int32_t selected = threadIdx.x; selected < effectiveTopK; selected += blockDim.x)
        {
            float const expValue = expf(selectedLogits[selected] - maxLogit);
            selectedExp[selected] = expValue;
            localSum += expValue;
        }
        float const sumExp = dsparkReduceSum(localSum);
        if (threadIdx.x == 0)
        {
            for (int32_t selected = 0; selected < effectiveTopK; ++selected)
            {
                rowProbs[selectedIndices[selected]] = 0.0F;
            }
            float const denom = useTopP ? fmaxf(effectiveTopP * sumExp, 1e-20F) : fmaxf(sumExp, 1e-20F);
            float remaining = denom;
            for (int32_t selected = 0; selected < effectiveTopK; ++selected)
            {
                float assigned = selectedExp[selected];
                if (useTopP)
                {
                    assigned = fminf(assigned, remaining);
                    remaining -= assigned;
                }
                if (assigned > 0.0F)
                {
                    rowProbs[selectedIndices[selected]] = assigned / denom;
                }
                if (useTopP && remaining <= 1e-20F)
                {
                    break;
                }
            }
        }
        return;
    }

    float localMax = -FLT_MAX;
    int32_t localIdx = 0;
    for (int32_t vocabIdx = threadIdx.x; vocabIdx < vocabSize; vocabIdx += blockDim.x)
    {
        float const val = rowLogits[vocabIdx] * invTemp;
        if (val > localMax || (val == localMax && vocabIdx < localIdx))
        {
            localMax = val;
            localIdx = vocabIdx;
        }
    }
    reduceMaxPair(localMax, localIdx);
    float const maxLogit = localMax;

    float localSum = 0.0F;
    for (int32_t vocabIdx = threadIdx.x; vocabIdx < vocabSize; vocabIdx += blockDim.x)
    {
        localSum += expf(rowLogits[vocabIdx] * invTemp - maxLogit);
    }
    float const sumExp = dsparkReduceSum(localSum);
    if (sumExp <= 0.0F || !isfinite(sumExp))
    {
        float const uniform = 1.0F / static_cast<float>(vocabSize);
        for (int32_t vocabIdx = threadIdx.x; vocabIdx < vocabSize; vocabIdx += blockDim.x)
        {
            rowProbs[vocabIdx] = uniform;
        }
        return;
    }

    for (int32_t vocabIdx = threadIdx.x; vocabIdx < vocabSize; vocabIdx += blockDim.x)
    {
        rowProbs[vocabIdx] = expf(rowLogits[vocabIdx] * invTemp - maxLogit) / sumExp;
    }
}

__global__ void dsparkFillUniformsKernel(
    float* __restrict__ uniforms, int32_t totalElements, uint64_t philoxSeed, uint64_t philoxOffset)
{
    int32_t const idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= totalElements)
    {
        return;
    }
    uniforms[idx] = dsparkUniformFromCounter(philoxSeed, philoxOffset, static_cast<uint64_t>(idx));
}

__global__ void dsparkFillProposalLengthsKernel(
    int32_t* __restrict__ proposalLengths, int32_t batchSize, int32_t proposalLen)
{
    int32_t const idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batchSize)
    {
        proposalLengths[idx] = proposalLen;
    }
}

__global__ void dsparkConfidenceKernel(half const* __restrict__ draftHiddenStates, // [B, P, H]
    half const* __restrict__ markovW1,                                             // [V, R]
    half const* __restrict__ confidenceWeight,                                     // [H] or [H + R]
    half const* __restrict__ confidenceBias,                                       // [1]
    int32_t const* __restrict__ firstPrevTokens,                                   // [B]
    int32_t const* __restrict__ draftTokenIds,                                     // [B, P]
    float* __restrict__ confidenceScores,                                          // [B, P]
    int32_t proposalLen, int32_t hiddenSize, int32_t markovRank, bool confidenceWithMarkov, int32_t hiddenStride,
    int32_t hiddenOffset)
{
    int32_t const step = blockIdx.x;
    int32_t const batchIdx = blockIdx.y;
    float localSum = 0.0F;
    half const* hidden
        = draftHiddenStates + (static_cast<int64_t>(batchIdx) * hiddenStride + step + hiddenOffset) * hiddenSize;
    for (int32_t idx = threadIdx.x; idx < hiddenSize; idx += blockDim.x)
    {
        localSum += __half2float(hidden[idx]) * __half2float(confidenceWeight[idx]);
    }

    if (confidenceWithMarkov)
    {
        int32_t const prevToken
            = (step == 0) ? firstPrevTokens[batchIdx] : draftTokenIds[batchIdx * proposalLen + step - 1];
        half const* prevMarkov = markovW1 + static_cast<int64_t>(prevToken) * markovRank;
        half const* markovWeight = confidenceWeight + hiddenSize;
        for (int32_t idx = threadIdx.x; idx < markovRank; idx += blockDim.x)
        {
            localSum += __half2float(prevMarkov[idx]) * __half2float(markovWeight[idx]);
        }
    }

    float const dot = dsparkReduceSum(localSum);
    if (threadIdx.x == 0)
    {
        float const logit = dot + __half2float(confidenceBias[0]);
        confidenceScores[batchIdx * proposalLen + step] = 1.0F / (1.0F + expf(-logit));
    }
}

__global__ void dsparkThresholdProposalLengthsKernel(float const* __restrict__ confidenceScores, // [B, P]
    int32_t* __restrict__ proposalLengths, int32_t batchSize, int32_t proposalLen, float threshold,
    int32_t minProposalLen, int32_t maxProposalLen)
{
    int32_t const batchIdx = blockIdx.x;
    if (batchIdx >= batchSize || threadIdx.x != 0)
    {
        return;
    }

    int32_t const clampedMin = max(1, min(proposalLen, minProposalLen));
    int32_t const clampedMax = max(clampedMin, min(proposalLen, maxProposalLen <= 0 ? proposalLen : maxProposalLen));
    if (threshold <= 0.0F)
    {
        proposalLengths[batchIdx] = clampedMax;
        return;
    }

    float survival = 1.0F;
    int32_t selected = clampedMin;
    for (int32_t step = 0; step < clampedMax; ++step)
    {
        survival *= confidenceScores[batchIdx * proposalLen + step];
        if (survival >= threshold)
        {
            selected = step + 1;
        }
        else
        {
            break;
        }
    }
    proposalLengths[batchIdx] = max(clampedMin, min(clampedMax, selected));
}

__global__ void dsparkSPSProposalLengthsKernel(float const* __restrict__ confidenceScores, // [B, P]
    int32_t* __restrict__ proposalLengths, int32_t batchSize, int32_t proposalLen, float survivalFloor,
    int32_t minProposalLen, int32_t maxProposalLen)
{
    int32_t const batchIdx = blockIdx.x;
    if (batchIdx >= batchSize || threadIdx.x != 0)
    {
        return;
    }

    int32_t const clampedMin = max(1, min(proposalLen, minProposalLen));
    int32_t const clampedMax = max(clampedMin, min(proposalLen, maxProposalLen <= 0 ? proposalLen : maxProposalLen));
    float const floor = fmaxf(0.0F, fminf(1.0F, survivalFloor));

    float survival = 1.0F;
    float expectedTokens = 1.0F;
    float bestScore = -FLT_MAX;
    int32_t selected = clampedMin;

    for (int32_t step = 0; step < clampedMax; ++step)
    {
        int32_t const length = step + 1;
        survival *= confidenceScores[batchIdx * proposalLen + step];
        expectedTokens += survival;

        if (length >= clampedMin)
        {
            float const verifyLen = static_cast<float>(length + 1);
            float const cost = kSPSDraftCost + kSPSVerifyBaseCost + kSPSVerifyTokenCost * verifyLen
                + kSPSAcceptCost * static_cast<float>(length);
            float const score = expectedTokens / fmaxf(cost, 1e-6F);
            if (score > bestScore)
            {
                bestScore = score;
                selected = length;
            }
        }

        if (floor > 0.0F && survival < floor && length >= clampedMin)
        {
            break;
        }
    }

    proposalLengths[batchIdx] = max(clampedMin, min(clampedMax, selected));
}

__global__ void dsparkBuildMarkovLogitsKernel(float const* __restrict__ backboneLogits, // [B, P, V]
    half const* __restrict__ markovW1,                                                  // [V, R]
    half const* __restrict__ markovW2,                                                  // [V, R]
    int32_t const* __restrict__ firstPrevTokens,                                        // [B]
    int32_t const* __restrict__ draftTokenIds,                                          // [B, P]
    float* __restrict__ correctedLogits,                                                // [B, V]
    int32_t step, int32_t proposalLen, int32_t vocabSize, int32_t markovRank, int32_t logitsStride,
    int32_t logitsOffset)
{
    int32_t const vocabBlockIdx = blockIdx.x;
    int32_t const batchIdx = blockIdx.y;
    int32_t const warpId = threadIdx.x / 32;
    int32_t const laneId = threadIdx.x % 32;
    int32_t const vocabIdx = vocabBlockIdx * kMarkovWarpsPerBlock + warpId;
    if (vocabIdx >= vocabSize)
    {
        return;
    }

    int32_t const prevToken
        = (step == 0) ? firstPrevTokens[batchIdx] : draftTokenIds[batchIdx * proposalLen + step - 1];
    half const* prevMarkov = markovW1 + static_cast<int64_t>(prevToken) * markovRank;
    float const* stepLogits
        = backboneLogits + (static_cast<int64_t>(batchIdx) * logitsStride + step + logitsOffset) * vocabSize;
    half const* vocabMarkov = markovW2 + static_cast<int64_t>(vocabIdx) * markovRank;

    float bias = 0.0F;
    for (int32_t rankIdx = laneId; rankIdx < markovRank; rankIdx += 32)
    {
        bias += __half2float(prevMarkov[rankIdx]) * __half2float(vocabMarkov[rankIdx]);
    }
    for (int32_t offset = 16; offset > 0; offset >>= 1)
    {
        bias += __shfl_down_sync(0xFFFFFFFF, bias, offset);
    }
    if (laneId == 0)
    {
        correctedLogits[static_cast<int64_t>(batchIdx) * vocabSize + vocabIdx] = stepLogits[vocabIdx] + bias;
    }
}

namespace
{
constexpr int32_t kMarkovFusedWarpsPerBlock = 8;
constexpr int32_t kMarkovFusedTokensPerWarp = 8;
constexpr int32_t kMarkovFusedBlockTokens = kMarkovFusedWarpsPerBlock * kMarkovFusedTokensPerWarp;
constexpr int32_t kMarkovFusedMaxRank = 512;

//! Orderable-float key in the high bits, (0xFFFFFFFF - vocabIdx) in the low bits so
//! exact ties resolve to the lowest vocab index under unsigned max. Zero is a valid
//! floor: even -inf packs to a nonzero key.
__device__ inline unsigned long long dsparkPackGreedyCandidate(float value, int32_t vocabIdx)
{
    uint32_t bits = __float_as_uint(value);
    bits = (bits & 0x80000000U) ? ~bits : (bits | 0x80000000U);
    return (static_cast<unsigned long long>(bits) << 32)
        | static_cast<unsigned long long>(0xFFFFFFFFU - static_cast<uint32_t>(vocabIdx));
}

__device__ inline int32_t dsparkUnpackGreedyIndex(unsigned long long packed)
{
    return static_cast<int32_t>(0xFFFFFFFFU - static_cast<uint32_t>(packed & 0xFFFFFFFFULL));
}
} // namespace

__global__ void dsparkMarkovGreedyFusedKernel(float const* __restrict__ backboneLogits, // [B, P, V]
    half const* __restrict__ markovW1,                                                  // [V, R]
    half const* __restrict__ markovW2,                                                  // [V, R]
    int32_t const* __restrict__ firstPrevTokens,                                        // [B]
    unsigned long long* __restrict__ greedySlots,                                       // [B, P], zeroed per round
    float* __restrict__ stackedOut,                                                     // depth row base or nullptr
    int64_t outBatchStride, int32_t step, int32_t proposalLen, int32_t vocabSize, int32_t markovRank,
    int32_t logitsStride, int32_t logitsOffset)
{
    __shared__ half w1Shared[kMarkovFusedMaxRank];
    __shared__ unsigned long long warpBest[kMarkovFusedWarpsPerBlock];

    int32_t const batchIdx = blockIdx.y;
    int32_t const prevToken = (step == 0) ? firstPrevTokens[batchIdx]
                                          : dsparkUnpackGreedyIndex(greedySlots[batchIdx * proposalLen + step - 1]);
    half const* w1Row = markovW1 + static_cast<int64_t>(prevToken) * markovRank;
    for (int32_t i = threadIdx.x; i < markovRank; i += blockDim.x)
    {
        w1Shared[i] = w1Row[i];
    }
    __syncthreads();

    int32_t const warpId = threadIdx.x / 32;
    int32_t const laneId = threadIdx.x % 32;
    float const* stepLogits
        = backboneLogits + (static_cast<int64_t>(batchIdx) * logitsStride + step + logitsOffset) * vocabSize;
    int32_t const wordsPerRow = markovRank / 8;

    unsigned long long best = 0ULL;
    int32_t const tokenBase = blockIdx.x * kMarkovFusedBlockTokens + warpId * kMarkovFusedTokensPerWarp;
    for (int32_t t = 0; t < kMarkovFusedTokensPerWarp; ++t)
    {
        int32_t const vocabIdx = tokenBase + t;
        if (vocabIdx >= vocabSize)
        {
            break;
        }
        float partial = 0.0F;
        for (int32_t rankBase = laneId * 8; rankBase < markovRank; rankBase += 32 * 8)
        {
            uint4 const packed
                = reinterpret_cast<uint4 const*>(markovW2)[static_cast<int64_t>(vocabIdx) * wordsPerRow + rankBase / 8];
            half2 const* w2h2 = reinterpret_cast<half2 const*>(&packed);
            half2 const* w1h2 = reinterpret_cast<half2 const*>(w1Shared + rankBase);
#pragma unroll
            for (int32_t j = 0; j < 4; ++j)
            {
                float2 const a = __half22float2(w2h2[j]);
                float2 const b = __half22float2(w1h2[j]);
                partial += a.x * b.x + a.y * b.y;
            }
        }
        for (int32_t offset = 16; offset > 0; offset >>= 1)
        {
            partial += __shfl_down_sync(0xFFFFFFFF, partial, offset);
        }
        if (laneId == 0)
        {
            float const corrected = stepLogits[vocabIdx] + partial;
            if (stackedOut != nullptr)
            {
                stackedOut[static_cast<int64_t>(batchIdx) * outBatchStride + vocabIdx] = corrected;
            }
            unsigned long long const candidate = dsparkPackGreedyCandidate(corrected, vocabIdx);
            best = (candidate > best) ? candidate : best;
        }
    }
    if (laneId == 0)
    {
        warpBest[warpId] = best;
    }
    __syncthreads();
    if (threadIdx.x == 0)
    {
        unsigned long long blockBest = warpBest[0];
        for (int32_t w = 1; w < kMarkovFusedWarpsPerBlock; ++w)
        {
            blockBest = max(blockBest, warpBest[w]);
        }
        if (blockBest != 0ULL)
        {
            atomicMax(&greedySlots[batchIdx * proposalLen + step], blockBest);
        }
    }
}

#if SUPPORTS_FP8
__global__ void dsparkQuantizeMarkovW2Fp8Kernel(half const* __restrict__ markovW2, // [V, R]
    uint8_t* __restrict__ w2Fp8,                                                   // [V, R] E4M3 bytes
    half* __restrict__ w2RowScales,                                                // [V]
    int32_t vocabSize, int32_t markovRank)
{
    int32_t const rowIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (rowIdx >= vocabSize)
    {
        return;
    }
    half const* row = markovW2 + static_cast<int64_t>(rowIdx) * markovRank;
    float absMax = 0.0F;
    for (int32_t r = 0; r < markovRank; ++r)
    {
        absMax = fmaxf(absMax, fabsf(__half2float(row[r])));
    }
    // Rows with absMax < ~0.027 yield a subnormal half scale; the precision loss
    // is proportional to the (already tiny) row magnitude, so no clamp is needed.
    float const scale = (absMax > 0.0F) ? absMax / 448.0F : 1.0F;
    float const invScale = 1.0F / scale;
    w2RowScales[rowIdx] = __float2half(scale);
    uint8_t* outRow = w2Fp8 + static_cast<int64_t>(rowIdx) * markovRank;
    for (int32_t r = 0; r < markovRank; ++r)
    {
        __nv_fp8_e4m3 const q(__half2float(row[r]) * invScale);
        outRow[r] = *reinterpret_cast<uint8_t const*>(&q);
    }
}

//! FP8-W2 fused greedy step. Layout: 16 lanes per token (each lane one 16-byte load of
//! 16 E4M3 values), two consecutive tokens per warp so the 32 x 16 B loads are linear in
//! laneId. Hardware fp8x2 -> half2 conversion keeps the unpack cost near the FP16 path's.
//! Requires markovRank == 16 * 2^k <= kMarkovFusedMaxRank: lanesPerToken must divide the
//! warp exactly or the tail lane group would alias the next iteration's token and the
//! partial-width shuffle reduce would span an incomplete subwarp.
__global__ void dsparkMarkovGreedyFusedFp8Kernel(float const* __restrict__ backboneLogits, // [B, P, V]
    half const* __restrict__ markovW1,                                                     // [V, R]
    uint4 const* __restrict__ w2Fp8,      // [V, R/16] uint4 rows of E4M3 bytes
    half const* __restrict__ w2RowScales, // [V]
    int32_t const* __restrict__ firstPrevTokens, unsigned long long* __restrict__ greedySlots,
    float* __restrict__ stackedOut, int64_t outBatchStride, int32_t step, int32_t proposalLen, int32_t vocabSize,
    int32_t markovRank, int32_t logitsStride, int32_t logitsOffset)
{
    __shared__ half w1Shared[kMarkovFusedMaxRank];
    __shared__ unsigned long long warpBest[kMarkovFusedWarpsPerBlock];

    int32_t const batchIdx = blockIdx.y;
    int32_t const prevToken = (step == 0) ? firstPrevTokens[batchIdx]
                                          : dsparkUnpackGreedyIndex(greedySlots[batchIdx * proposalLen + step - 1]);
    half const* w1Row = markovW1 + static_cast<int64_t>(prevToken) * markovRank;
    for (int32_t i = threadIdx.x; i < markovRank; i += blockDim.x)
    {
        w1Shared[i] = w1Row[i];
    }
    __syncthreads();

    int32_t const warpId = threadIdx.x / 32;
    int32_t const laneId = threadIdx.x % 32;
    int32_t const lanesPerToken = markovRank / 16;
    int32_t const tokensPerWarpIter = 32 / lanesPerToken;
    int32_t const laneInGroup = laneId % lanesPerToken;
    int32_t const groupIdx = laneId / lanesPerToken;
    int32_t const rankBase = laneInGroup * 16;
    int32_t const vecsPerRow = lanesPerToken;
    float const* stepLogits
        = backboneLogits + (static_cast<int64_t>(batchIdx) * logitsStride + step + logitsOffset) * vocabSize;

    int32_t const warpTokens = tokensPerWarpIter * kMarkovFusedTokensPerWarp;
    int32_t const warpTokenBase = blockIdx.x * (kMarkovFusedWarpsPerBlock * warpTokens) + warpId * warpTokens;

    unsigned long long best = 0ULL;
    for (int32_t t = 0; t < kMarkovFusedTokensPerWarp; ++t)
    {
        int32_t const vocabIdx = warpTokenBase + t * tokensPerWarpIter + groupIdx;
        float partial = 0.0F;
        if (vocabIdx < vocabSize)
        {
            uint4 const packed = w2Fp8[static_cast<int64_t>(vocabIdx) * vecsPerRow + laneInGroup];
            __nv_fp8x2_storage_t const* pairs = reinterpret_cast<__nv_fp8x2_storage_t const*>(&packed);
            half2 const* w1h2 = reinterpret_cast<half2 const*>(w1Shared + rankBase);
            float sum = 0.0F;
#pragma unroll
            for (int32_t j = 0; j < 8; ++j)
            {
                __half2_raw const hr = __nv_cvt_fp8x2_to_halfraw2(pairs[j], __NV_E4M3);
                float2 const a = __half22float2(*reinterpret_cast<half2 const*>(&hr));
                float2 const b = __half22float2(w1h2[j]);
                sum += a.x * b.x + a.y * b.y;
            }
            partial = sum * __half2float(w2RowScales[vocabIdx]);
        }
        for (int32_t offset = lanesPerToken / 2; offset > 0; offset >>= 1)
        {
            partial += __shfl_down_sync(0xFFFFFFFF, partial, offset, lanesPerToken);
        }
        if (laneInGroup == 0 && vocabIdx < vocabSize)
        {
            float const corrected = stepLogits[vocabIdx] + partial;
            if (stackedOut != nullptr)
            {
                stackedOut[static_cast<int64_t>(batchIdx) * outBatchStride + vocabIdx] = corrected;
            }
            unsigned long long const candidate = dsparkPackGreedyCandidate(corrected, vocabIdx);
            best = (candidate > best) ? candidate : best;
        }
    }
    for (int32_t offset = 16; offset > 0; offset >>= 1)
    {
        unsigned long long const other = __shfl_down_sync(0xFFFFFFFF, best, offset);
        best = (other > best) ? other : best;
    }
    if (laneId == 0)
    {
        warpBest[warpId] = best;
    }
    __syncthreads();
    if (threadIdx.x == 0)
    {
        unsigned long long blockBest = warpBest[0];
        for (int32_t w = 1; w < kMarkovFusedWarpsPerBlock; ++w)
        {
            blockBest = max(blockBest, warpBest[w]);
        }
        if (blockBest != 0ULL)
        {
            atomicMax(&greedySlots[batchIdx * proposalLen + step], blockBest);
        }
    }
}

#endif // SUPPORTS_FP8

__global__ void dsparkFinalizeGreedyTokensKernel(
    unsigned long long const* __restrict__ greedySlots, int32_t* __restrict__ draftTokenIds, int32_t totalSteps)
{
    int32_t const idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= totalSteps)
    {
        return;
    }
    draftTokenIds[idx] = dsparkUnpackGreedyIndex(greedySlots[idx]);
}

__global__ void dsparkSampleProbabilityRowsKernel(float const* __restrict__ probabilities, // [B, V]
    float const* __restrict__ proposalUniforms,                                            // [B, P]
    int32_t* __restrict__ draftTokenIds,                                                   // [B, P]
    int32_t step, int32_t proposalLen, int32_t vocabSize)
{
    int32_t const batchIdx = blockIdx.x;
    if (threadIdx.x != 0)
    {
        return;
    }
    float const uniform = proposalUniforms[batchIdx * proposalLen + step];
    float const* rowProbs = probabilities + static_cast<int64_t>(batchIdx) * vocabSize;
    draftTokenIds[batchIdx * proposalLen + step] = dsparkSampleFromProbs(rowProbs, vocabSize, uniform);
}

__global__ void dsparkStoreDraftStepProbabilitiesKernel(float const* __restrict__ stepProbabilities, // [B, V]
    float* __restrict__ draftProbabilities,                                                          // [B, P, V]
    int32_t step, int32_t proposalLen, int32_t vocabSize, int32_t totalElements)
{
    int32_t const idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= totalElements)
    {
        return;
    }
    int32_t const batchIdx = idx / vocabSize;
    int32_t const vocabIdx = idx % vocabSize;
    draftProbabilities[(static_cast<int64_t>(batchIdx) * proposalLen + step) * vocabSize + vocabIdx]
        = stepProbabilities[static_cast<int64_t>(batchIdx) * vocabSize + vocabIdx];
}

__global__ void dsparkProbabilisticAcceptKernel(float const* __restrict__ targetProbabilities, // [B, VFY, V]
    float const* __restrict__ draftProbabilities,                                              // [B, draftStride, V]
    int32_t const* __restrict__ draftTokenIds,                                                 // [B, draftStride]
    int32_t const* __restrict__ proposalLengths,                                               // [B]
    float const* __restrict__ acceptUniforms,                                                  // [B, 2*draftStride + 1]
    int32_t* __restrict__ acceptedTokenIds,                                                    // [B, VFY]
    int32_t* __restrict__ acceptLength,                                                        // [B]
    int32_t draftStride, int32_t verifyProposalLen, int32_t vocabSize)
{
    int32_t const batchIdx = blockIdx.x;
    if (threadIdx.x != 0)
    {
        return;
    }

    int32_t const verifyLen = verifyProposalLen + 1;
    int32_t const uniformStride = 2 * draftStride + 1;
    int32_t* batchAccepted = acceptedTokenIds + batchIdx * verifyLen;
    for (int32_t pos = 0; pos < verifyLen; ++pos)
    {
        batchAccepted[pos] = 0;
    }

    int32_t const rowProposalLen = max(1, min(verifyProposalLen, proposalLengths[batchIdx]));
    int32_t acceptedDraft = 0;
    for (int32_t step = 0; step < rowProposalLen; ++step)
    {
        int32_t const draftToken = draftTokenIds[batchIdx * draftStride + step];
        float const* targetRow = targetProbabilities + (static_cast<int64_t>(batchIdx) * verifyLen + step) * vocabSize;
        float const* draftRow = draftProbabilities + (static_cast<int64_t>(batchIdx) * draftStride + step) * vocabSize;
        float const targetProb = targetRow[draftToken];
        float const draftProb = draftRow[draftToken];
        float const acceptProb = draftProb <= 1e-20F ? 1.0F : fminf(1.0F, targetProb / draftProb);
        float const acceptUniform = acceptUniforms[batchIdx * uniformStride + step];
        // p == 0 must reject: the guard above reads a vanishing draft probability as p/q -> inf,
        // the right limit only while p > 0.
        if (targetProb > 0.0F && acceptUniform <= acceptProb)
        {
            batchAccepted[acceptedDraft] = draftToken;
            ++acceptedDraft;
            continue;
        }

        float const residualUniform = acceptUniforms[batchIdx * uniformStride + draftStride + step];
        batchAccepted[acceptedDraft] = dsparkSampleFromResidual(targetRow, draftRow, vocabSize, residualUniform);
        acceptLength[batchIdx] = acceptedDraft + 1;
        return;
    }

    float const* bonusRow
        = targetProbabilities + (static_cast<int64_t>(batchIdx) * verifyLen + rowProposalLen) * vocabSize;
    float const bonusUniform = acceptUniforms[batchIdx * uniformStride + 2 * draftStride];
    batchAccepted[acceptedDraft] = dsparkSampleFromProbs(bonusRow, vocabSize, bonusUniform);
    acceptLength[batchIdx] = acceptedDraft + 1;
}

__device__ int32_t dsparkSampleFromSparseProbs(float const* probs, int32_t const* indices, int32_t topK, float uniform)
{
    float const target = clampUniform(uniform);
    float cumulative = 0.0F;
    int32_t fallback = topK > 0 ? indices[0] : 0;
    for (int32_t k = 0; k < topK; ++k)
    {
        float const prob = probs[k];
        if (prob <= 0.0F)
        {
            continue;
        }
        fallback = indices[k];
        cumulative += prob;
        if (target < cumulative)
        {
            return indices[k];
        }
    }
    return fallback;
}

__global__ void dsparkNormalizeTopKRowsKernel(float const* __restrict__ topKValues, // [rows, K]
    float* __restrict__ topKProbabilities,                                          // [rows, K]
    int32_t rows, int32_t topK, float temperature)
{
    int32_t const rowIdx = blockIdx.x;
    if (rowIdx >= rows || topK <= 0)
    {
        return;
    }

    float const invTemp = dsparkInvTemperature(temperature);
    float const* rowValues = topKValues + static_cast<int64_t>(rowIdx) * topK;
    float* rowProbs = topKProbabilities + static_cast<int64_t>(rowIdx) * topK;

    float localMax = -FLT_MAX;
    int32_t localIdx = 0;
    for (int32_t k = threadIdx.x; k < topK; k += blockDim.x)
    {
        float const val = rowValues[k] * invTemp;
        if (val > localMax || (val == localMax && k < localIdx))
        {
            localMax = val;
            localIdx = k;
        }
    }
    reduceMaxPair(localMax, localIdx);
    float const maxValue = localMax;

    float localSum = 0.0F;
    for (int32_t k = threadIdx.x; k < topK; k += blockDim.x)
    {
        localSum += expf(rowValues[k] * invTemp - maxValue);
    }
    float const sumValue = dsparkReduceSum(localSum);
    float const denom = fmaxf(sumValue, 1e-20F);
    for (int32_t k = threadIdx.x; k < topK; k += blockDim.x)
    {
        rowProbs[k] = sumValue > 0.0F && isfinite(sumValue) ? expf(rowValues[k] * invTemp - maxValue) / denom
                                                            : 1.0F / static_cast<float>(topK);
    }
}

__global__ void dsparkStoreDraftStepTop1Kernel(int32_t const* __restrict__ top1Indices, // [B, 1]
    int32_t* __restrict__ draftTokenIds, int32_t batchSize, int32_t step, int32_t proposalLen)
{
    int32_t const batchIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (batchIdx < batchSize)
    {
        draftTokenIds[batchIdx * proposalLen + step] = top1Indices[batchIdx];
    }
}

__global__ void dsparkSampleTopKRowsAndStoreKernel(float const* __restrict__ topKValues, // [B, K]
    int32_t const* __restrict__ topKIndices,                                             // [B, K]
    float const* __restrict__ proposalUniforms,                                          // [B, P]
    int32_t* __restrict__ draftTokenIds,                                                 // [B, P]
    float* __restrict__ draftTopKProbabilities,                                          // [B, P, K]
    int32_t* __restrict__ draftTopKIndices,                                              // [B, P, K]
    int32_t step, int32_t proposalLen, int32_t topK, float temperature)
{
    int32_t const batchIdx = blockIdx.x;
    float const invTemp = dsparkInvTemperature(temperature);
    float const* rowValues = topKValues + static_cast<int64_t>(batchIdx) * topK;
    int32_t const* rowIndices = topKIndices + static_cast<int64_t>(batchIdx) * topK;
    float* outProbs = draftTopKProbabilities + (static_cast<int64_t>(batchIdx) * proposalLen + step) * topK;
    int32_t* outIndices = draftTopKIndices + (static_cast<int64_t>(batchIdx) * proposalLen + step) * topK;

    float localMax = -FLT_MAX;
    int32_t localIdx = 0;
    for (int32_t k = threadIdx.x; k < topK; k += blockDim.x)
    {
        float const val = rowValues[k] * invTemp;
        if (val > localMax || (val == localMax && k < localIdx))
        {
            localMax = val;
            localIdx = k;
        }
    }
    reduceMaxPair(localMax, localIdx);
    float const maxValue = localMax;

    float localSum = 0.0F;
    for (int32_t k = threadIdx.x; k < topK; k += blockDim.x)
    {
        localSum += expf(rowValues[k] * invTemp - maxValue);
    }
    float const sumValue = dsparkReduceSum(localSum);
    float const denom = fmaxf(sumValue, 1e-20F);
    for (int32_t k = threadIdx.x; k < topK; k += blockDim.x)
    {
        outIndices[k] = rowIndices[k];
        outProbs[k] = sumValue > 0.0F && isfinite(sumValue) ? expf(rowValues[k] * invTemp - maxValue) / denom
                                                            : 1.0F / static_cast<float>(topK);
    }
    __syncthreads();

    if (threadIdx.x == 0)
    {
        float const uniform = proposalUniforms[batchIdx * proposalLen + step];
        draftTokenIds[batchIdx * proposalLen + step] = dsparkSampleFromSparseProbs(outProbs, outIndices, topK, uniform);
    }
}

} // anonymous namespace

int32_t dsparkMarkovPartialCount(int32_t vocabSize)
{
    return (vocabSize + kMarkovWarpsPerBlock - 1) / kMarkovWarpsPerBlock;
}

void dsparkBuildVerifyTokens(rt::Tensor const& lastAcceptedTokens, rt::Tensor const& draftTokenIds,
    rt::Tensor& verifyTokenIds, int32_t batchSize, int32_t draftStride, int32_t verifyProposalLen, cudaStream_t stream)
{
    int32_t const verifyLen = verifyProposalLen + 1;
    int32_t const totalThreads = batchSize * verifyLen;
    int32_t const numThreads = 256;
    int32_t const numBlocks = (totalThreads + numThreads - 1) / numThreads;

    dsparkBuildVerifyTokensKernel<<<numBlocks, numThreads, 0, stream>>>(
        static_cast<int32_t const*>(lastAcceptedTokens.rawPointer()),
        static_cast<int32_t const*>(draftTokenIds.rawPointer()), static_cast<int32_t*>(verifyTokenIds.rawPointer()),
        draftStride, verifyProposalLen, totalThreads);
    CUDA_CHECK(cudaGetLastError());
}

void dsparkGreedyAccept(rt::Tensor const& baseLogits, rt::Tensor const& draftTokenIds,
    rt::Tensor const& proposalLengths, rt::Tensor& acceptedTokenIds, rt::Tensor& acceptLength,
    rt::Tensor& argmaxScratch, int32_t batchSize, int32_t draftStride, int32_t verifyProposalLen, int32_t vocabSize,
    cudaStream_t stream)
{
    int32_t const verifyLen = verifyProposalLen + 1;
    int32_t const totalPositions = batchSize * verifyLen;
    int32_t* argmaxResults = static_cast<int32_t*>(argmaxScratch.rawPointer());

    static constexpr int32_t kGreedyAcceptArgmaxBlockSize = 256;
    dsparkParallelArgmaxKernel<kGreedyAcceptArgmaxBlockSize>
        <<<totalPositions, kGreedyAcceptArgmaxBlockSize, 0, stream>>>(
            static_cast<float const*>(baseLogits.rawPointer()), argmaxResults, totalPositions, vocabSize);
    CUDA_CHECK(cudaGetLastError());

    dsparkGreedyAcceptWalkKernel<<<batchSize, 1, 0, stream>>>(argmaxResults,
        static_cast<int32_t const*>(draftTokenIds.rawPointer()),
        static_cast<int32_t const*>(proposalLengths.rawPointer()), static_cast<int32_t*>(acceptedTokenIds.rawPointer()),
        static_cast<int32_t*>(acceptLength.rawPointer()), draftStride, verifyProposalLen);
    CUDA_CHECK(cudaGetLastError());
}

void dsparkFillProposalLengths(rt::Tensor& proposalLengths, int32_t batchSize, int32_t proposalLen, cudaStream_t stream)
{
    dim3 const block(kProbabilityBlockSize);
    dim3 const grid((batchSize + block.x - 1) / block.x);
    dsparkFillProposalLengthsKernel<<<grid, block, 0, stream>>>(
        static_cast<int32_t*>(proposalLengths.rawPointer()), batchSize, proposalLen);
    CUDA_CHECK(cudaGetLastError());
}

void dsparkComputeConfidenceAndProposalLengths(rt::Tensor const& draftHiddenStates, rt::Tensor const& markovW1,
    rt::Tensor const& confidenceWeight, rt::Tensor const& confidenceBias, rt::Tensor const& firstPrevTokens,
    rt::Tensor const& draftTokenIds, rt::Tensor& confidenceScores, rt::Tensor& proposalLengths, int32_t batchSize,
    int32_t proposalLen, int32_t hiddenSize, int32_t markovRank, bool confidenceWithMarkov, float threshold,
    int32_t minProposalLen, int32_t maxProposalLen, cudaStream_t stream, int32_t hiddenStride, int32_t hiddenOffset)
{
    dim3 const confidenceGrid(proposalLen, batchSize);
    dsparkConfidenceKernel<<<confidenceGrid, kProbabilityBlockSize, 0, stream>>>(
        static_cast<half const*>(draftHiddenStates.rawPointer()), static_cast<half const*>(markovW1.rawPointer()),
        static_cast<half const*>(confidenceWeight.rawPointer()), static_cast<half const*>(confidenceBias.rawPointer()),
        static_cast<int32_t const*>(firstPrevTokens.rawPointer()),
        static_cast<int32_t const*>(draftTokenIds.rawPointer()), static_cast<float*>(confidenceScores.rawPointer()),
        proposalLen, hiddenSize, markovRank, confidenceWithMarkov, hiddenStride > 0 ? hiddenStride : proposalLen,
        hiddenOffset);
    CUDA_CHECK(cudaGetLastError());
    dsparkThresholdProposalLengthsKernel<<<batchSize, 1, 0, stream>>>(
        static_cast<float const*>(confidenceScores.rawPointer()), static_cast<int32_t*>(proposalLengths.rawPointer()),
        batchSize, proposalLen, threshold, minProposalLen, maxProposalLen);
    CUDA_CHECK(cudaGetLastError());
}

void dsparkComputeConfidenceAndSPSProposalLengths(rt::Tensor const& draftHiddenStates, rt::Tensor const& markovW1,
    rt::Tensor const& confidenceWeight, rt::Tensor const& confidenceBias, rt::Tensor const& firstPrevTokens,
    rt::Tensor const& draftTokenIds, rt::Tensor& confidenceScores, rt::Tensor& proposalLengths, int32_t batchSize,
    int32_t proposalLen, int32_t hiddenSize, int32_t markovRank, bool confidenceWithMarkov, float survivalFloor,
    int32_t minProposalLen, int32_t maxProposalLen, cudaStream_t stream, int32_t hiddenStride, int32_t hiddenOffset)
{
    dim3 const confidenceGrid(proposalLen, batchSize);
    dsparkConfidenceKernel<<<confidenceGrid, kProbabilityBlockSize, 0, stream>>>(
        static_cast<half const*>(draftHiddenStates.rawPointer()), static_cast<half const*>(markovW1.rawPointer()),
        static_cast<half const*>(confidenceWeight.rawPointer()), static_cast<half const*>(confidenceBias.rawPointer()),
        static_cast<int32_t const*>(firstPrevTokens.rawPointer()),
        static_cast<int32_t const*>(draftTokenIds.rawPointer()), static_cast<float*>(confidenceScores.rawPointer()),
        proposalLen, hiddenSize, markovRank, confidenceWithMarkov, hiddenStride > 0 ? hiddenStride : proposalLen,
        hiddenOffset);
    CUDA_CHECK(cudaGetLastError());
    dsparkSPSProposalLengthsKernel<<<batchSize, 1, 0, stream>>>(
        static_cast<float const*>(confidenceScores.rawPointer()), static_cast<int32_t*>(proposalLengths.rawPointer()),
        batchSize, proposalLen, survivalFloor, minProposalLen, maxProposalLen);
    CUDA_CHECK(cudaGetLastError());
}

void dsparkComputeConfidenceScores(rt::Tensor const& draftHiddenStates, rt::Tensor const& markovW1,
    rt::Tensor const& confidenceWeight, rt::Tensor const& confidenceBias, rt::Tensor const& firstPrevTokens,
    rt::Tensor const& draftTokenIds, rt::Tensor& confidenceScores, int32_t batchSize, int32_t proposalLen,
    int32_t hiddenSize, int32_t markovRank, bool confidenceWithMarkov, cudaStream_t stream, int32_t hiddenStride,
    int32_t hiddenOffset)
{
    dim3 const confidenceGrid(proposalLen, batchSize);
    dsparkConfidenceKernel<<<confidenceGrid, kProbabilityBlockSize, 0, stream>>>(
        static_cast<half const*>(draftHiddenStates.rawPointer()), static_cast<half const*>(markovW1.rawPointer()),
        static_cast<half const*>(confidenceWeight.rawPointer()), static_cast<half const*>(confidenceBias.rawPointer()),
        static_cast<int32_t const*>(firstPrevTokens.rawPointer()),
        static_cast<int32_t const*>(draftTokenIds.rawPointer()), static_cast<float*>(confidenceScores.rawPointer()),
        proposalLen, hiddenSize, markovRank, confidenceWithMarkov, hiddenStride > 0 ? hiddenStride : proposalLen,
        hiddenOffset);
    CUDA_CHECK(cudaGetLastError());
}

void dsparkLogitsToProbabilities(rt::Tensor const& logits, rt::Tensor& probabilities, int32_t rows, int32_t vocabSize,
    float temperature, int32_t topK, float topP, cudaStream_t stream)
{
    dsparkLogitsToProbabilitiesKernel<<<rows, kProbabilityBlockSize, 0, stream>>>(
        static_cast<float const*>(logits.rawPointer()), static_cast<float*>(probabilities.rawPointer()), rows,
        vocabSize, temperature, topK, topP);
    CUDA_CHECK(cudaGetLastError());
}

void dsparkFillUniforms(
    rt::Tensor& uniforms, int32_t totalElements, uint64_t philoxSeed, uint64_t philoxOffset, cudaStream_t stream)
{
    int32_t const numThreads = 256;
    int32_t const numBlocks = (totalElements + numThreads - 1) / numThreads;
    dsparkFillUniformsKernel<<<numBlocks, numThreads, 0, stream>>>(
        static_cast<float*>(uniforms.rawPointer()), totalElements, philoxSeed, philoxOffset);
    CUDA_CHECK(cudaGetLastError());
}

void dsparkVanillaMarkovSample(rt::Tensor const& backboneLogits, rt::Tensor const& markovW1, rt::Tensor const& markovW2,
    rt::Tensor const& firstPrevTokens, rt::Tensor const& proposalUniforms, rt::Tensor& draftTokenIds,
    rt::Tensor& draftProbabilities, rt::Tensor& correctedLogitsScratch, rt::Tensor& probabilityScratch,
    int32_t batchSize, int32_t proposalLen, int32_t vocabSize, int32_t markovRank, float temperature, int32_t topK,
    float topP, cudaStream_t stream, int32_t logitsStride, int32_t logitsOffset)
{
    check::check(probabilityScratch.reshape({batchSize, vocabSize}), "Tensor reshape failed");
    int32_t const numVocabBlocks = dsparkMarkovPartialCount(vocabSize);
    dim3 const markovGrid(numVocabBlocks, batchSize);
    int32_t const totalProbabilityElements = batchSize * vocabSize;
    int32_t const copyThreads = 256;
    int32_t const copyBlocks = (totalProbabilityElements + copyThreads - 1) / copyThreads;

    for (int32_t step = 0; step < proposalLen; ++step)
    {
        dsparkBuildMarkovLogitsKernel<<<markovGrid, kMarkovBlockSize, 0, stream>>>(
            static_cast<float const*>(backboneLogits.rawPointer()), static_cast<half const*>(markovW1.rawPointer()),
            static_cast<half const*>(markovW2.rawPointer()), static_cast<int32_t const*>(firstPrevTokens.rawPointer()),
            static_cast<int32_t const*>(draftTokenIds.rawPointer()),
            static_cast<float*>(correctedLogitsScratch.rawPointer()), step, proposalLen, vocabSize, markovRank,
            logitsStride > 0 ? logitsStride : proposalLen, logitsOffset);
        CUDA_CHECK(cudaGetLastError());

        dsparkLogitsToProbabilities(
            correctedLogitsScratch, probabilityScratch, batchSize, vocabSize, temperature, topK, topP, stream);

        dsparkSampleProbabilityRowsKernel<<<batchSize, 1, 0, stream>>>(
            static_cast<float const*>(probabilityScratch.rawPointer()),
            static_cast<float const*>(proposalUniforms.rawPointer()), static_cast<int32_t*>(draftTokenIds.rawPointer()),
            step, proposalLen, vocabSize);
        CUDA_CHECK(cudaGetLastError());

        dsparkStoreDraftStepProbabilitiesKernel<<<copyBlocks, copyThreads, 0, stream>>>(
            static_cast<float const*>(probabilityScratch.rawPointer()),
            static_cast<float*>(draftProbabilities.rawPointer()), step, proposalLen, vocabSize,
            totalProbabilityElements);
        CUDA_CHECK(cudaGetLastError());
    }
}

void dsparkBuildMarkovLogits(rt::Tensor const& backboneLogits, rt::Tensor const& markovW1, rt::Tensor const& markovW2,
    rt::Tensor const& firstPrevTokens, rt::Tensor const& draftTokenIds, rt::Tensor& correctedLogitsScratch,
    int32_t batchSize, int32_t step, int32_t proposalLen, int32_t vocabSize, int32_t markovRank, cudaStream_t stream,
    int32_t logitsStride, int32_t logitsOffset)
{
    int32_t const numVocabBlocks = dsparkMarkovPartialCount(vocabSize);
    dim3 const markovGrid(numVocabBlocks, batchSize);
    dsparkBuildMarkovLogitsKernel<<<markovGrid, kMarkovBlockSize, 0, stream>>>(
        static_cast<float const*>(backboneLogits.rawPointer()), static_cast<half const*>(markovW1.rawPointer()),
        static_cast<half const*>(markovW2.rawPointer()), static_cast<int32_t const*>(firstPrevTokens.rawPointer()),
        static_cast<int32_t const*>(draftTokenIds.rawPointer()),
        static_cast<float*>(correctedLogitsScratch.rawPointer()), step, proposalLen, vocabSize, markovRank,
        logitsStride > 0 ? logitsStride : proposalLen, logitsOffset);
    CUDA_CHECK(cudaGetLastError());
}

bool dsparkFusedGreedySupported(int32_t markovRank)
{
    return markovRank > 0 && markovRank % 8 == 0 && markovRank <= kMarkovFusedMaxRank;
}

void dsparkMarkovGreedyFusedStep(rt::Tensor const& backboneLogits, rt::Tensor const& markovW1,
    rt::Tensor const& markovW2, rt::Tensor const& firstPrevTokens, rt::Tensor& greedySlots, rt::Tensor* stackedLogits,
    int32_t stackedDepthRow, int32_t batchSize, int32_t step, int32_t proposalLen, int32_t vocabSize,
    int32_t markovRank, cudaStream_t stream, int32_t logitsStride, int32_t logitsOffset)
{
    check::check(dsparkFusedGreedySupported(markovRank), "fused greedy Markov step: unsupported markov rank");
    int32_t const numTokenBlocks = (vocabSize + kMarkovFusedBlockTokens - 1) / kMarkovFusedBlockTokens;
    dim3 const grid(numTokenBlocks, batchSize);
    int32_t const blockSize = kMarkovFusedWarpsPerBlock * 32;

    float* stackedOut = nullptr;
    int64_t outBatchStride = 0;
    if (stackedLogits != nullptr)
    {
        // Stacked layout is [batch, proposalLen + 1, vocab]; point at this step's depth row.
        outBatchStride = static_cast<int64_t>(proposalLen + 1) * vocabSize;
        stackedOut
            = static_cast<float*>(stackedLogits->rawPointer()) + static_cast<int64_t>(stackedDepthRow) * vocabSize;
    }

    dsparkMarkovGreedyFusedKernel<<<grid, blockSize, 0, stream>>>(
        static_cast<float const*>(backboneLogits.rawPointer()), static_cast<half const*>(markovW1.rawPointer()),
        static_cast<half const*>(markovW2.rawPointer()), static_cast<int32_t const*>(firstPrevTokens.rawPointer()),
        static_cast<unsigned long long*>(greedySlots.rawPointer()), stackedOut, outBatchStride, step, proposalLen,
        vocabSize, markovRank, logitsStride > 0 ? logitsStride : proposalLen, logitsOffset);
    CUDA_CHECK(cudaGetLastError());
}

void dsparkQuantizeMarkovW2Fp8(rt::Tensor const& markovW2, rt::Tensor& w2Fp8, rt::Tensor& w2RowScales,
    int32_t vocabSize, int32_t markovRank, cudaStream_t stream)
{
    check::check(markovRank % 16 == 0, "FP8 markov_w2 requires markovRank divisible by 16");
    int32_t const numThreads = 128;
    int32_t const numBlocks = (vocabSize + numThreads - 1) / numThreads;
#if SUPPORTS_FP8
    dsparkQuantizeMarkovW2Fp8Kernel<<<numBlocks, numThreads, 0, stream>>>(
        static_cast<half const*>(markovW2.rawPointer()), static_cast<uint8_t*>(w2Fp8.rawPointer()),
        static_cast<half*>(w2RowScales.rawPointer()), vocabSize, markovRank);
    CUDA_CHECK(cudaGetLastError());
#else
    check::check(false, "FP8 markov_w2 requires CUDA >= 11.8 (cuda_fp8.h unavailable).");
#endif
}

void dsparkMarkovGreedyFusedStepFp8(rt::Tensor const& backboneLogits, rt::Tensor const& markovW1,
    rt::Tensor const& w2Fp8, rt::Tensor const& w2RowScales, rt::Tensor const& firstPrevTokens, rt::Tensor& greedySlots,
    rt::Tensor* stackedLogits, int32_t stackedDepthRow, int32_t batchSize, int32_t step, int32_t proposalLen,
    int32_t vocabSize, int32_t markovRank, cudaStream_t stream, int32_t logitsStride, int32_t logitsOffset)
{
    // lanesPerToken = markovRank / 16 must be a power of two: the kernel tiles a warp into
    // 32 / lanesPerToken lane groups and uses lanesPerToken-wide shuffle reductions, both of
    // which require the groups to cover the warp exactly.
    int32_t const lanesPerToken = markovRank / 16;
    check::check(
        markovRank % 16 == 0 && markovRank <= kMarkovFusedMaxRank && (lanesPerToken & (lanesPerToken - 1)) == 0,
        "FP8 fused greedy Markov step: unsupported markov rank");
    int32_t const warpTokens = (32 / lanesPerToken) * kMarkovFusedTokensPerWarp;
    int32_t const blockTokens = kMarkovFusedWarpsPerBlock * warpTokens;
    dim3 const grid((vocabSize + blockTokens - 1) / blockTokens, batchSize);
    int32_t const blockSize = kMarkovFusedWarpsPerBlock * 32;

    float* stackedOut = nullptr;
    int64_t outBatchStride = 0;
    if (stackedLogits != nullptr)
    {
        outBatchStride = static_cast<int64_t>(proposalLen + 1) * vocabSize;
        stackedOut
            = static_cast<float*>(stackedLogits->rawPointer()) + static_cast<int64_t>(stackedDepthRow) * vocabSize;
    }

#if SUPPORTS_FP8
    dsparkMarkovGreedyFusedFp8Kernel<<<grid, blockSize, 0, stream>>>(
        static_cast<float const*>(backboneLogits.rawPointer()), static_cast<half const*>(markovW1.rawPointer()),
        static_cast<uint4 const*>(w2Fp8.rawPointer()), static_cast<half const*>(w2RowScales.rawPointer()),
        static_cast<int32_t const*>(firstPrevTokens.rawPointer()),
        static_cast<unsigned long long*>(greedySlots.rawPointer()), stackedOut, outBatchStride, step, proposalLen,
        vocabSize, markovRank, logitsStride > 0 ? logitsStride : proposalLen, logitsOffset);
    CUDA_CHECK(cudaGetLastError());
#else
    check::check(false, "EDGELLM_DSPARK_W2_FP8 requires CUDA >= 11.8 (cuda_fp8.h unavailable).");
#endif
}

void dsparkFinalizeGreedyDraftTokens(
    rt::Tensor const& greedySlots, rt::Tensor& draftTokenIds, int32_t totalSteps, cudaStream_t stream)
{
    int32_t const numThreads = 128;
    int32_t const numBlocks = (totalSteps + numThreads - 1) / numThreads;
    dsparkFinalizeGreedyTokensKernel<<<numBlocks, numThreads, 0, stream>>>(
        static_cast<unsigned long long const*>(greedySlots.rawPointer()),
        static_cast<int32_t*>(draftTokenIds.rawPointer()), totalSteps);
    CUDA_CHECK(cudaGetLastError());
}

void dsparkSampleProbabilityRows(rt::Tensor const& probabilityScratch, rt::Tensor const& proposalUniforms,
    rt::Tensor& draftTokenIds, int32_t batchSize, int32_t step, int32_t proposalLen, int32_t vocabSize,
    cudaStream_t stream)
{
    dsparkSampleProbabilityRowsKernel<<<batchSize, 1, 0, stream>>>(
        static_cast<float const*>(probabilityScratch.rawPointer()),
        static_cast<float const*>(proposalUniforms.rawPointer()), static_cast<int32_t*>(draftTokenIds.rawPointer()),
        step, proposalLen, vocabSize);
    CUDA_CHECK(cudaGetLastError());
}

void dsparkStoreDraftStepProbabilities(rt::Tensor const& probabilityScratch, rt::Tensor& draftProbabilities,
    int32_t batchSize, int32_t step, int32_t proposalLen, int32_t vocabSize, cudaStream_t stream)
{
    int32_t const totalProbabilityElements = batchSize * vocabSize;
    int32_t const copyThreads = 256;
    int32_t const copyBlocks = (totalProbabilityElements + copyThreads - 1) / copyThreads;
    dsparkStoreDraftStepProbabilitiesKernel<<<copyBlocks, copyThreads, 0, stream>>>(
        static_cast<float const*>(probabilityScratch.rawPointer()),
        static_cast<float*>(draftProbabilities.rawPointer()), step, proposalLen, vocabSize, totalProbabilityElements);
    CUDA_CHECK(cudaGetLastError());
}

void dsparkNormalizeTopKRows(rt::Tensor const& topKValues, rt::Tensor& topKProbabilities, int32_t rows, int32_t topK,
    float temperature, cudaStream_t stream)
{
    dsparkNormalizeTopKRowsKernel<<<rows, kProbabilityBlockSize, 0, stream>>>(
        static_cast<float const*>(topKValues.rawPointer()), static_cast<float*>(topKProbabilities.rawPointer()), rows,
        topK, temperature);
    CUDA_CHECK(cudaGetLastError());
}

void dsparkStoreDraftStepTop1(rt::Tensor const& top1Indices, rt::Tensor& draftTokenIds, int32_t batchSize, int32_t step,
    int32_t proposalLen, cudaStream_t stream)
{
    int32_t constexpr threads = 256;
    int32_t const blocks = (batchSize + threads - 1) / threads;
    dsparkStoreDraftStepTop1Kernel<<<blocks, threads, 0, stream>>>(
        static_cast<int32_t const*>(top1Indices.rawPointer()), static_cast<int32_t*>(draftTokenIds.rawPointer()),
        batchSize, step, proposalLen);
    CUDA_CHECK(cudaGetLastError());
}

void dsparkSampleTopKRowsAndStore(rt::Tensor const& topKValues, rt::Tensor const& topKIndices,
    rt::Tensor const& proposalUniforms, rt::Tensor& draftTokenIds, rt::Tensor& draftTopKProbabilities,
    rt::Tensor& draftTopKIndices, int32_t batchSize, int32_t step, int32_t proposalLen, int32_t topK, float temperature,
    cudaStream_t stream)
{
    dsparkSampleTopKRowsAndStoreKernel<<<batchSize, kProbabilityBlockSize, 0, stream>>>(
        static_cast<float const*>(topKValues.rawPointer()), static_cast<int32_t const*>(topKIndices.rawPointer()),
        static_cast<float const*>(proposalUniforms.rawPointer()), static_cast<int32_t*>(draftTokenIds.rawPointer()),
        static_cast<float*>(draftTopKProbabilities.rawPointer()), static_cast<int32_t*>(draftTopKIndices.rawPointer()),
        step, proposalLen, topK, temperature);
    CUDA_CHECK(cudaGetLastError());
}

void dsparkProbabilisticAccept(rt::Tensor const& targetProbabilities, rt::Tensor const& draftProbabilities,
    rt::Tensor const& draftTokenIds, rt::Tensor const& proposalLengths, rt::Tensor const& acceptUniforms,
    rt::Tensor& acceptedTokenIds, rt::Tensor& acceptLength, int32_t batchSize, int32_t draftStride,
    int32_t verifyProposalLen, int32_t vocabSize, cudaStream_t stream)
{
    dsparkProbabilisticAcceptKernel<<<batchSize, 1, 0, stream>>>(
        static_cast<float const*>(targetProbabilities.rawPointer()),
        static_cast<float const*>(draftProbabilities.rawPointer()),
        static_cast<int32_t const*>(draftTokenIds.rawPointer()),
        static_cast<int32_t const*>(proposalLengths.rawPointer()),
        static_cast<float const*>(acceptUniforms.rawPointer()), static_cast<int32_t*>(acceptedTokenIds.rawPointer()),
        static_cast<int32_t*>(acceptLength.rawPointer()), draftStride, verifyProposalLen, vocabSize);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace kernel
} // namespace trt_edgellm
