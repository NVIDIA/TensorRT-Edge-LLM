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
#include "kernels/speculative/dsparkKernels.h"

#include <cfloat>
#include <cub/cub.cuh>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

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

__device__ __forceinline__ float dsparkClampUniform(float uniform)
{
    return fminf(fmaxf(uniform, 0.0F), 0.99999994F);
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
    return dsparkClampUniform(static_cast<float>(mantissa) * (1.0F / 16777216.0F));
}

__device__ int32_t dsparkSampleFromProbs(float const* probs, int32_t vocabSize, float uniform)
{
    float const target = dsparkClampUniform(uniform);
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

    float const target = dsparkClampUniform(uniform) * residualSum;
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
    int32_t proposalLen, int32_t hiddenSize, int32_t markovRank, bool confidenceWithMarkov)
{
    int32_t const step = blockIdx.x;
    int32_t const batchIdx = blockIdx.y;
    float localSum = 0.0F;
    half const* hidden = draftHiddenStates + (static_cast<int64_t>(batchIdx) * proposalLen + step) * hiddenSize;
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
    int32_t step, int32_t proposalLen, int32_t vocabSize, int32_t markovRank)
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
    float const* stepLogits = backboneLogits + (static_cast<int64_t>(batchIdx) * proposalLen + step) * vocabSize;
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
        if (acceptUniform <= acceptProb)
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

__device__ float dsparkSparseProbabilityForToken(
    float const* probs, int32_t const* indices, int32_t topK, int32_t token)
{
    for (int32_t k = 0; k < topK; ++k)
    {
        if (indices[k] == token)
        {
            return probs[k];
        }
    }
    return 0.0F;
}

__device__ int32_t dsparkSampleFromSparseProbs(float const* probs, int32_t const* indices, int32_t topK, float uniform)
{
    float const target = dsparkClampUniform(uniform);
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

__device__ int32_t dsparkSampleFromSparseResidual(float const* targetProbs, int32_t const* targetIndices,
    int32_t targetTopK, float const* draftProbs, int32_t const* draftIndices, int32_t draftTopK, float uniform)
{
    float residualSum = 0.0F;
    for (int32_t k = 0; k < targetTopK; ++k)
    {
        int32_t const token = targetIndices[k];
        float const draftProb = dsparkSparseProbabilityForToken(draftProbs, draftIndices, draftTopK, token);
        residualSum += fmaxf(targetProbs[k] - draftProb, 0.0F);
    }
    if (residualSum <= 1e-20F)
    {
        return dsparkSampleFromSparseProbs(targetProbs, targetIndices, targetTopK, uniform);
    }

    float const target = dsparkClampUniform(uniform) * residualSum;
    float cumulative = 0.0F;
    int32_t fallback = targetTopK > 0 ? targetIndices[0] : 0;
    for (int32_t k = 0; k < targetTopK; ++k)
    {
        int32_t const token = targetIndices[k];
        float const draftProb = dsparkSparseProbabilityForToken(draftProbs, draftIndices, draftTopK, token);
        float const residual = fmaxf(targetProbs[k] - draftProb, 0.0F);
        if (residual <= 0.0F)
        {
            continue;
        }
        fallback = token;
        cumulative += residual;
        if (target < cumulative)
        {
            return token;
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

__global__ void dsparkSparseTopKAcceptKernel(float const* __restrict__ targetTopKProbabilities, // [B, VFY, Kt]
    int32_t const* __restrict__ targetTopKIndices,                                              // [B, VFY, Kt]
    float const* __restrict__ draftTopKProbabilities,                                           // [B, draftStride, Kd]
    int32_t const* __restrict__ draftTopKIndices,                                               // [B, draftStride, Kd]
    int32_t const* __restrict__ draftTokenIds,                                                  // [B, draftStride]
    int32_t const* __restrict__ proposalLengths,                                                // [B]
    float const* __restrict__ acceptUniforms, // [B, 2*draftStride + 1]
    int32_t* __restrict__ acceptedTokenIds,   // [B, VFY]
    int32_t* __restrict__ acceptLength,       // [B]
    int32_t draftStride, int32_t verifyProposalLen, int32_t targetTopK, int32_t draftTopK)
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
        float const* targetRow
            = targetTopKProbabilities + (static_cast<int64_t>(batchIdx) * verifyLen + step) * targetTopK;
        int32_t const* targetIndexRow
            = targetTopKIndices + (static_cast<int64_t>(batchIdx) * verifyLen + step) * targetTopK;
        float const* draftRow
            = draftTopKProbabilities + (static_cast<int64_t>(batchIdx) * draftStride + step) * draftTopK;
        int32_t const* draftIndexRow
            = draftTopKIndices + (static_cast<int64_t>(batchIdx) * draftStride + step) * draftTopK;
        float const targetProb = dsparkSparseProbabilityForToken(targetRow, targetIndexRow, targetTopK, draftToken);
        float const draftProb = dsparkSparseProbabilityForToken(draftRow, draftIndexRow, draftTopK, draftToken);
        float const acceptProb = draftProb <= 1e-20F ? 1.0F : fminf(1.0F, targetProb / draftProb);
        float const acceptUniform = acceptUniforms[batchIdx * uniformStride + step];
        if (acceptUniform <= acceptProb)
        {
            batchAccepted[acceptedDraft] = draftToken;
            ++acceptedDraft;
            continue;
        }

        float const residualUniform = acceptUniforms[batchIdx * uniformStride + draftStride + step];
        batchAccepted[acceptedDraft] = dsparkSampleFromSparseResidual(
            targetRow, targetIndexRow, targetTopK, draftRow, draftIndexRow, draftTopK, residualUniform);
        acceptLength[batchIdx] = acceptedDraft + 1;
        return;
    }

    float const* bonusRow
        = targetTopKProbabilities + (static_cast<int64_t>(batchIdx) * verifyLen + rowProposalLen) * targetTopK;
    int32_t const* bonusIndexRow
        = targetTopKIndices + (static_cast<int64_t>(batchIdx) * verifyLen + rowProposalLen) * targetTopK;
    float const bonusUniform = acceptUniforms[batchIdx * uniformStride + 2 * draftStride];
    batchAccepted[acceptedDraft] = dsparkSampleFromSparseProbs(bonusRow, bonusIndexRow, targetTopK, bonusUniform);
    acceptLength[batchIdx] = acceptedDraft + 1;
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
    int32_t minProposalLen, int32_t maxProposalLen, cudaStream_t stream)
{
    dim3 const confidenceGrid(proposalLen, batchSize);
    dsparkConfidenceKernel<<<confidenceGrid, kProbabilityBlockSize, 0, stream>>>(
        static_cast<half const*>(draftHiddenStates.rawPointer()), static_cast<half const*>(markovW1.rawPointer()),
        static_cast<half const*>(confidenceWeight.rawPointer()), static_cast<half const*>(confidenceBias.rawPointer()),
        static_cast<int32_t const*>(firstPrevTokens.rawPointer()),
        static_cast<int32_t const*>(draftTokenIds.rawPointer()), static_cast<float*>(confidenceScores.rawPointer()),
        proposalLen, hiddenSize, markovRank, confidenceWithMarkov);
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
    int32_t minProposalLen, int32_t maxProposalLen, cudaStream_t stream)
{
    dim3 const confidenceGrid(proposalLen, batchSize);
    dsparkConfidenceKernel<<<confidenceGrid, kProbabilityBlockSize, 0, stream>>>(
        static_cast<half const*>(draftHiddenStates.rawPointer()), static_cast<half const*>(markovW1.rawPointer()),
        static_cast<half const*>(confidenceWeight.rawPointer()), static_cast<half const*>(confidenceBias.rawPointer()),
        static_cast<int32_t const*>(firstPrevTokens.rawPointer()),
        static_cast<int32_t const*>(draftTokenIds.rawPointer()), static_cast<float*>(confidenceScores.rawPointer()),
        proposalLen, hiddenSize, markovRank, confidenceWithMarkov);
    CUDA_CHECK(cudaGetLastError());
    dsparkSPSProposalLengthsKernel<<<batchSize, 1, 0, stream>>>(
        static_cast<float const*>(confidenceScores.rawPointer()), static_cast<int32_t*>(proposalLengths.rawPointer()),
        batchSize, proposalLen, survivalFloor, minProposalLen, maxProposalLen);
    CUDA_CHECK(cudaGetLastError());
}

void dsparkComputeConfidenceScores(rt::Tensor const& draftHiddenStates, rt::Tensor const& markovW1,
    rt::Tensor const& confidenceWeight, rt::Tensor const& confidenceBias, rt::Tensor const& firstPrevTokens,
    rt::Tensor const& draftTokenIds, rt::Tensor& confidenceScores, int32_t batchSize, int32_t proposalLen,
    int32_t hiddenSize, int32_t markovRank, bool confidenceWithMarkov, cudaStream_t stream)
{
    dim3 const confidenceGrid(proposalLen, batchSize);
    dsparkConfidenceKernel<<<confidenceGrid, kProbabilityBlockSize, 0, stream>>>(
        static_cast<half const*>(draftHiddenStates.rawPointer()), static_cast<half const*>(markovW1.rawPointer()),
        static_cast<half const*>(confidenceWeight.rawPointer()), static_cast<half const*>(confidenceBias.rawPointer()),
        static_cast<int32_t const*>(firstPrevTokens.rawPointer()),
        static_cast<int32_t const*>(draftTokenIds.rawPointer()), static_cast<float*>(confidenceScores.rawPointer()),
        proposalLen, hiddenSize, markovRank, confidenceWithMarkov);
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
    float topP, cudaStream_t stream)
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
            static_cast<float*>(correctedLogitsScratch.rawPointer()), step, proposalLen, vocabSize, markovRank);
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
    int32_t batchSize, int32_t step, int32_t proposalLen, int32_t vocabSize, int32_t markovRank, cudaStream_t stream)
{
    int32_t const numVocabBlocks = dsparkMarkovPartialCount(vocabSize);
    dim3 const markovGrid(numVocabBlocks, batchSize);
    dsparkBuildMarkovLogitsKernel<<<markovGrid, kMarkovBlockSize, 0, stream>>>(
        static_cast<float const*>(backboneLogits.rawPointer()), static_cast<half const*>(markovW1.rawPointer()),
        static_cast<half const*>(markovW2.rawPointer()), static_cast<int32_t const*>(firstPrevTokens.rawPointer()),
        static_cast<int32_t const*>(draftTokenIds.rawPointer()),
        static_cast<float*>(correctedLogitsScratch.rawPointer()), step, proposalLen, vocabSize, markovRank);
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

void dsparkSparseTopKAccept(rt::Tensor const& targetTopKProbabilities, rt::Tensor const& targetTopKIndices,
    rt::Tensor const& draftTopKProbabilities, rt::Tensor const& draftTopKIndices, rt::Tensor const& draftTokenIds,
    rt::Tensor const& proposalLengths, rt::Tensor const& acceptUniforms, rt::Tensor& acceptedTokenIds,
    rt::Tensor& acceptLength, int32_t batchSize, int32_t draftStride, int32_t verifyProposalLen, int32_t targetTopK,
    int32_t draftTopK, cudaStream_t stream)
{
    dsparkSparseTopKAcceptKernel<<<batchSize, 1, 0, stream>>>(
        static_cast<float const*>(targetTopKProbabilities.rawPointer()),
        static_cast<int32_t const*>(targetTopKIndices.rawPointer()),
        static_cast<float const*>(draftTopKProbabilities.rawPointer()),
        static_cast<int32_t const*>(draftTopKIndices.rawPointer()),
        static_cast<int32_t const*>(draftTokenIds.rawPointer()),
        static_cast<int32_t const*>(proposalLengths.rawPointer()),
        static_cast<float const*>(acceptUniforms.rawPointer()), static_cast<int32_t*>(acceptedTokenIds.rawPointer()),
        static_cast<int32_t*>(acceptLength.rawPointer()), draftStride, verifyProposalLen, targetTopK, draftTopK);
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
