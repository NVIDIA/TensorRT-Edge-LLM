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

#include "common/checkMacros.h"
#include "moeSigmoidGroupTopkKernels.h"
#include <cfloat>
#include <cub/cub.cuh>

namespace trt_edgellm
{
namespace kernel
{

static constexpr int32_t SGT_TPB = 256;

/**
 * @brief One-CTA-per-token kernel implementing sigmoid + grouped top-k routing.
 *
 * Dynamic shared memory layout (all float-sized):
 *   float sigmoidScores[numExperts]   — unbiased sigmoid for final weight gather
 *   float biasedScores[numExperts]    — sigmoid + bias for selection
 *   float groupScores[nGroup]         — top-2 sum per group
 *   int32_t groupSelected[nGroup]     — 1/0 mask for selected groups
 */
__launch_bounds__(SGT_TPB) __global__
    void sigmoidGroupTopkKernel(float const* __restrict__ gatingOutput, float* __restrict__ topkWeights,
        int32_t* __restrict__ topkIndices, int32_t numExperts, int32_t topK, int32_t nGroup, int32_t topkGroup,
        bool normTopkProb, float routedScalingFactor, float const* __restrict__ correctionBias)
{
    using cub_kvp = cub::KeyValuePair<int, float>;
    using BlockReduce = cub::BlockReduce<cub_kvp, SGT_TPB>;
    __shared__ typename BlockReduce::TempStorage reduceStorage;

    // Dynamic shared memory pointers
    extern __shared__ char sharedMem[];
    float* sigmoidScores = reinterpret_cast<float*>(sharedMem);
    float* biasedScores = sigmoidScores + numExperts;
    float* groupScores = biasedScores + numExperts;
    int32_t* groupSelected = reinterpret_cast<int32_t*>(groupScores + nGroup);

    int32_t const token = blockIdx.x;
    int32_t const tokenOffset = token * numExperts;
    int32_t const expertsPerGroup = numExperts / nGroup;

    // ==================== Step 1: Sigmoid + bias ====================
    for (int32_t e = threadIdx.x; e < numExperts; e += SGT_TPB)
    {
        float logit = gatingOutput[tokenOffset + e];
        float sig = 1.0f / (1.0f + expf(-logit));
        sigmoidScores[e] = sig;

        float biased = sig;
        if (correctionBias != nullptr)
        {
            biased += correctionBias[e];
        }
        biasedScores[e] = biased;
    }
    __syncthreads();

    // ==================== Step 2: Top-2 per group → groupScores ====================
    for (int32_t g = threadIdx.x; g < nGroup; g += SGT_TPB)
    {
        int32_t const groupStart = g * expertsPerGroup;
        float top1 = -FLT_MAX;
        float top2 = -FLT_MAX;
        for (int32_t i = 0; i < expertsPerGroup; i++)
        {
            float val = biasedScores[groupStart + i];
            if (val > top1)
            {
                top2 = top1;
                top1 = val;
            }
            else if (val > top2)
            {
                top2 = val;
            }
        }
        groupScores[g] = top1 + top2;
        groupSelected[g] = 0;
    }
    __syncthreads();

    // ==================== Step 3: Thread 0 picks topkGroup groups ====================
    if (threadIdx.x == 0)
    {
        for (int32_t i = 0; i < topkGroup; i++)
        {
            int32_t bestGroup = -1;
            float bestScore = -FLT_MAX;
            for (int32_t g = 0; g < nGroup; g++)
            {
                if (groupSelected[g] == 0 && groupScores[g] > bestScore)
                {
                    bestScore = groupScores[g];
                    bestGroup = g;
                }
            }
            if (bestGroup >= 0)
            {
                groupSelected[bestGroup] = 1;
            }
        }
    }
    __syncthreads();

    // ==================== Step 4: Mask unselected groups ====================
    for (int32_t e = threadIdx.x; e < numExperts; e += SGT_TPB)
    {
        int32_t const group = e / expertsPerGroup;
        if (groupSelected[group] == 0)
        {
            biasedScores[e] = -FLT_MAX;
        }
    }
    __syncthreads();

    // ==================== Step 5: Top-K from masked biased scores ====================
    cub::ArgMax argMax;
    float renormSum = 0.0f;
    int32_t const outBase = token * topK;

    for (int32_t kIdx = 0; kIdx < topK; kIdx++)
    {
        cub_kvp threadKvp;
        threadKvp.key = 0;
        threadKvp.value = -FLT_MAX;

        for (int32_t e = threadIdx.x; e < numExperts; e += SGT_TPB)
        {
            cub_kvp candidate;
            candidate.key = e;
            candidate.value = biasedScores[e];
            threadKvp = argMax(candidate, threadKvp);
        }

        cub_kvp const resultKvp = BlockReduce(reduceStorage).Reduce(threadKvp, argMax);

        if (threadIdx.x == 0)
        {
            int32_t const winnerIdx = resultKvp.key;
            topkIndices[outBase + kIdx] = winnerIdx;
            // Gather weight from ORIGINAL sigmoid (not biased)
            topkWeights[outBase + kIdx] = sigmoidScores[winnerIdx];
            renormSum += sigmoidScores[winnerIdx];
            // Mask out this expert for next iteration
            biasedScores[winnerIdx] = -FLT_MAX;
        }
        __syncthreads();
    }

    // ==================== Step 6: Renormalize + scale ====================
    if (threadIdx.x == 0)
    {
        if (normTopkProb && renormSum > 0.0f)
        {
            float invSum = 1.0f / renormSum;
            for (int32_t k = 0; k < topK; k++)
            {
                topkWeights[outBase + k] *= invSum;
            }
        }
        for (int32_t k = 0; k < topK; k++)
        {
            topkWeights[outBase + k] *= routedScalingFactor;
        }
    }
}

// ====================== Public API Implementation ======================

void moeSigmoidGroupTopk(rt::Tensor const& gatingOutput, rt::Tensor& topkWeights, rt::Tensor& topkIndices, int32_t topK,
    int32_t nGroup, int32_t topkGroup, bool normTopkProb, float routedScalingFactor, cudaStream_t stream,
    rt::OptionalInputTensor correctionBias)
{
    // Validate input shapes
    auto const gatingShape = gatingOutput.getShape();
    auto const weightsShape = topkWeights.getShape();
    auto const indicesShape = topkIndices.getShape();

    check::check(gatingShape.getNumDims() == 2, "gatingOutput must be 2D tensor [numTokens, numExperts]");
    check::check(weightsShape.getNumDims() == 2, "topkWeights must be 2D tensor [numTokens, topK]");
    check::check(indicesShape.getNumDims() == 2, "topkIndices must be 2D tensor [numTokens, topK]");

    int32_t const numTokens = static_cast<int32_t>(gatingShape[0]);
    int32_t const numExperts = static_cast<int32_t>(gatingShape[1]);

    check::check(weightsShape[0] == numTokens, "topkWeights first dimension must match numTokens");
    check::check(indicesShape[0] == numTokens, "topkIndices first dimension must match numTokens");
    check::check(weightsShape[1] == topK, "topkWeights second dimension must match topK");
    check::check(indicesShape[1] == topK, "topkIndices second dimension must match topK");
    check::check(topK <= numExperts, "topK must be less than or equal to numExperts");

    // Validate group parameters
    check::check(nGroup > 0, "nGroup must be positive");
    check::check(numExperts % nGroup == 0, "numExperts must be divisible by nGroup");
    check::check(topkGroup > 0 && topkGroup <= nGroup, "topkGroup must be in [1, nGroup]");

    // Validate data types — FP32 only
    check::check(gatingOutput.getDataType() == trt_edgellm::DataType::kFLOAT, "gatingOutput must be float32");
    check::check(topkWeights.getDataType() == trt_edgellm::DataType::kFLOAT, "topkWeights must be float32");
    check::check(topkIndices.getDataType() == trt_edgellm::DataType::kINT32, "topkIndices must be int32");

    // Validate device types
    check::check(gatingOutput.getDeviceType() == rt::DeviceType::kGPU, "gatingOutput must be on GPU");
    check::check(topkWeights.getDeviceType() == rt::DeviceType::kGPU, "topkWeights must be on GPU");
    check::check(topkIndices.getDeviceType() == rt::DeviceType::kGPU, "topkIndices must be on GPU");

    // Validate correction bias if provided
    float const* biasPtr = nullptr;
    if (correctionBias.has_value())
    {
        auto const& biasTensor = correctionBias.value().get();
        auto const biasShape = biasTensor.getShape();
        check::check(biasShape.getNumDims() == 1, "correctionBias must be 1D tensor [numExperts]");
        check::check(biasShape[0] == numExperts, "correctionBias size must match numExperts");
        check::check(biasTensor.getDataType() == trt_edgellm::DataType::kFLOAT, "correctionBias must be float32");
        check::check(biasTensor.getDeviceType() == rt::DeviceType::kGPU, "correctionBias must be on GPU");
        biasPtr = biasTensor.dataPointer<float>();
    }

    if (numTokens == 0)
    {
        return;
    }

    // Compute dynamic shared memory size
    size_t const sharedMemSize = static_cast<size_t>(numExperts) * sizeof(float) * 2 // sigmoidScores + biasedScores
        + static_cast<size_t>(nGroup) * sizeof(float)                                // groupScores
        + static_cast<size_t>(nGroup) * sizeof(int32_t);                             // groupSelected

    float const* inputPtr = gatingOutput.dataPointer<float>();
    float* topkWeightsPtr = topkWeights.dataPointer<float>();
    int32_t* topkIndicesPtr = topkIndices.dataPointer<int32_t>();

    sigmoidGroupTopkKernel<<<numTokens, SGT_TPB, sharedMemSize, stream>>>(inputPtr, topkWeightsPtr, topkIndicesPtr,
        numExperts, topK, nGroup, topkGroup, normTopkProb, routedScalingFactor, biasPtr);
}

} // namespace kernel
} // namespace trt_edgellm
