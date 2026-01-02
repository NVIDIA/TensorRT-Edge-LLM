/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

// clang-format off
#include "sampling.h"
// clang-format on
#include "common/checkMacros.h"
#include <cassert>
#include <cfloat>
#include <cstdint>
#include <cub/cub.cuh>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <curand_kernel.h>
#include <stdexcept>

namespace trt_edgellm
{

// Define missing constants for half precision
#ifndef HALF_FLT_MAX
#define HALF_FLT_MAX 65504.0f
#endif

// Internal workspace structure for memory management (FP32 only)
struct SamplingWorkspace
{
    void* ptr;
    size_t size;

    // Buffer pointers and sizes for different sampling methods
    float* topkTempLogits;
    int32_t* topkIndices;
    float* topkValues;

    void* toppTempStorage; // Keep as void* for CUB temp storage
    float* toppProbs;
    float* toppSortedProbs;
    int32_t* toppSortedIdVals;
    int32_t* toppIdVals;
    int32_t* toppOffsetBuf;
    int32_t* toppBeginOffsetBuf;
    int32_t* toppTopKIndices;
    float* toppTopKValues;
    int32_t* toppEarlyExitFlags;

    SamplingWorkspace()
        : ptr(nullptr)
        , size(0)
        , topkTempLogits(nullptr)
        , topkIndices(nullptr)
        , topkValues(nullptr)
        , toppTempStorage(nullptr)
        , toppProbs(nullptr)
        , toppSortedProbs(nullptr)
        , toppSortedIdVals(nullptr)
        , toppIdVals(nullptr)
        , toppOffsetBuf(nullptr)
        , toppBeginOffsetBuf(nullptr)
        , toppTopKIndices(nullptr)
        , toppTopKValues(nullptr)
        , toppEarlyExitFlags(nullptr)
    {
    }

    // Calculate workspace partitioning for given parameters
    void setupWorkspace(void* workspace, size_t workspaceSize, SamplingParams const& params)
    {
        ptr = workspace;
        size = workspaceSize;

        // Add alignment function (same as in size calculation)
        auto alignSize = [](size_t size) -> size_t {
            size_t const alignment = 256; // 256-byte alignment for optimal GPU memory access
            return (size + alignment - 1) & ~(alignment - 1);
        };

        // Calculate buffer sizes and offsets
        size_t offset = 0;

        if (params.useTopK)
        {
            // Top-K workspace layout
            size_t tempLogitsSize = alignSize(params.batchSize * params.vocabSize * sizeof(float));
            size_t indicesSize = alignSize(params.batchSize * 8 * params.topK * sizeof(int32_t)); // BLOCKS_PER_BEAM = 8
            size_t valuesSize = alignSize(params.batchSize * 8 * params.topK * sizeof(float));

            topkTempLogits = reinterpret_cast<float*>(static_cast<char*>(ptr) + offset);
            offset += tempLogitsSize;

            topkIndices = reinterpret_cast<int32_t*>(static_cast<char*>(ptr) + offset);
            offset += indicesSize;

            topkValues = reinterpret_cast<float*>(static_cast<char*>(ptr) + offset);
            offset += valuesSize;
        }
        else if (params.useTopP)
        {
            // Top-P workspace layout
            // Calculate CUB temp storage size
            size_t cubTempStorageSize;
            cub::DeviceSegmentedRadixSort::SortPairsDescending(nullptr, cubTempStorageSize,
                static_cast<float*>(nullptr), static_cast<float*>(nullptr), static_cast<int32_t*>(nullptr),
                static_cast<int32_t*>(nullptr), static_cast<int32_t>(params.vocabSize * params.batchSize),
                params.batchSize, static_cast<int32_t*>(nullptr), static_cast<int32_t*>(nullptr));

            toppTempStorage = static_cast<char*>(ptr) + offset;
            offset += alignSize(cubTempStorageSize);

            toppProbs = reinterpret_cast<float*>(static_cast<char*>(ptr) + offset);
            offset += alignSize(params.batchSize * params.vocabSize * sizeof(float));

            toppSortedProbs = reinterpret_cast<float*>(static_cast<char*>(ptr) + offset);
            offset += alignSize(params.batchSize * params.vocabSize * sizeof(float));

            toppSortedIdVals = reinterpret_cast<int32_t*>(static_cast<char*>(ptr) + offset);
            offset += alignSize(params.batchSize * params.vocabSize * sizeof(int32_t));

            toppIdVals = reinterpret_cast<int32_t*>(static_cast<char*>(ptr) + offset);
            offset += alignSize(params.batchSize * params.vocabSize * sizeof(int32_t));

            toppOffsetBuf = reinterpret_cast<int32_t*>(static_cast<char*>(ptr) + offset);
            offset += alignSize((params.batchSize + 1) * sizeof(int32_t));

            toppBeginOffsetBuf = reinterpret_cast<int32_t*>(static_cast<char*>(ptr) + offset);
            offset += alignSize((params.batchSize + 1) * sizeof(int32_t));

            toppTopKIndices = reinterpret_cast<int32_t*>(static_cast<char*>(ptr) + offset);
            offset += alignSize(params.batchSize * params.vocabSize * sizeof(int32_t));

            toppTopKValues = reinterpret_cast<float*>(static_cast<char*>(ptr) + offset);
            offset += alignSize(params.batchSize * params.vocabSize * sizeof(float));

            toppEarlyExitFlags = reinterpret_cast<int32_t*>(static_cast<char*>(ptr) + offset);
            offset += alignSize(params.batchSize * sizeof(int32_t));
        }

        // Validate workspace size
        if (offset > workspaceSize)
        {
            throw std::runtime_error("Workspace size too small. Required: " + std::to_string(offset)
                + ", provided: " + std::to_string(workspaceSize));
        }
    }

    // Setup workspace for selectAllTopK
    void setupWorkspaceForTopK(
        void* workspace, size_t workspaceSize, int32_t batchSize, int32_t vocabSize, int32_t topK)
    {
        ptr = workspace;
        size = workspaceSize;

        // Add alignment function (same as in size calculation)
        auto alignSize = [](size_t size) -> size_t {
            size_t const alignment = 256; // 256-byte alignment for optimal GPU memory access
            return (size + alignment - 1) & ~(alignment - 1);
        };

        size_t offset = 0;

        // Same layout as top-K sampling
        size_t tempLogitsSize = alignSize(batchSize * vocabSize * sizeof(float));
        size_t indicesSize = alignSize(batchSize * 8 * topK * sizeof(int32_t)); // BLOCKS_PER_BEAM = 8
        size_t valuesSize = alignSize(batchSize * 8 * topK * sizeof(float));

        topkTempLogits = reinterpret_cast<float*>(static_cast<char*>(ptr) + offset);
        offset += tempLogitsSize;

        topkIndices = reinterpret_cast<int32_t*>(static_cast<char*>(ptr) + offset);
        offset += indicesSize;

        topkValues = reinterpret_cast<float*>(static_cast<char*>(ptr) + offset);
        offset += valuesSize;

        // Validate workspace size
        if (offset > workspaceSize)
        {
            throw std::runtime_error("Workspace size too small for topK. Required: " + std::to_string(offset)
                + ", provided: " + std::to_string(workspaceSize));
        }
    }
};

// ========================================================================
// WORKSPACE SIZE CALCULATION
// ========================================================================

// Enhanced workspace size calculation that handles alignment (FP32 only)
size_t getTopKtopPSamplingWorkspaceSize(int32_t batchSize, int32_t vocabSize, SamplingParams const& params)
{
    size_t workspaceSize = 0;

    // Add alignment padding between buffers
    auto alignSize = [](size_t size) -> size_t {
        size_t const alignment = 256; // 256-byte alignment for optimal GPU memory access
        return (size + alignment - 1) & ~(alignment - 1);
    };

    if (params.useTopK)
    {
        workspaceSize += alignSize(batchSize * vocabSize * sizeof(float));         // temp logits
        workspaceSize += alignSize(batchSize * 8 * params.topK * sizeof(int32_t)); // top-k indices
        workspaceSize += alignSize(batchSize * 8 * params.topK * sizeof(float));   // top-k values
    }
    else if (params.useTopP)
    {
        // Calculate CUB temp storage size
        size_t cubTempStorageSize;
        cub::DeviceSegmentedRadixSort::SortPairsDescending(nullptr, cubTempStorageSize, static_cast<float*>(nullptr),
            static_cast<float*>(nullptr), static_cast<int32_t*>(nullptr), static_cast<int32_t*>(nullptr),
            static_cast<int32_t>(vocabSize * batchSize), batchSize, static_cast<int32_t*>(nullptr),
            static_cast<int32_t*>(nullptr));

        workspaceSize += alignSize(cubTempStorageSize);                      // CUB temp storage
        workspaceSize += alignSize(batchSize * vocabSize * sizeof(float));   // probs
        workspaceSize += alignSize(batchSize * vocabSize * sizeof(float));   // sorted probs
        workspaceSize += alignSize(batchSize * vocabSize * sizeof(int32_t)); // sorted id vals
        workspaceSize += alignSize(batchSize * vocabSize * sizeof(int32_t)); // id vals
        workspaceSize += alignSize((batchSize + 1) * sizeof(int32_t));       // offset buf
        workspaceSize += alignSize((batchSize + 1) * sizeof(int32_t));       // begin offset buf
        workspaceSize += alignSize(batchSize * vocabSize * sizeof(int32_t)); // top-k indices
        workspaceSize += alignSize(batchSize * vocabSize * sizeof(float));   // top-k values
        workspaceSize += alignSize(batchSize * sizeof(int32_t));             // early exit flags
    }
    else
    {
        // No filtering - error out
        throw std::runtime_error("Either topK or topP must be set");
    }

    return workspaceSize;
}

// Calculate workspace size for selectAllTopK (FP32 only)
size_t getSelectAllTopKWorkspaceSize(int32_t batchSize, int32_t vocabSize, int32_t topK)
{
    auto alignSize = [](size_t size) -> size_t {
        size_t const alignment = 256;
        return (size + alignment - 1) & ~(alignment - 1);
    };

    size_t workspaceSize = 0;
    workspaceSize += alignSize(batchSize * vocabSize * sizeof(float));  // temp logits
    workspaceSize += alignSize(batchSize * 8 * topK * sizeof(int32_t)); // top-k indices
    workspaceSize += alignSize(batchSize * 8 * topK * sizeof(float));   // top-k values

    return workspaceSize;
}

// Device functions for math operations (FP32 only)
__device__ float exp_device(float x)
{
    return expf(x);
}

__device__ float max_device(float a, float b)
{
    return fmaxf(a, b);
}

__device__ float invTemp_device(float temperature)
{
    return (temperature < 1e-3f) ? 1000.0f : 1.0f / temperature;
}

// Helper structures for top-k reduction operations (FP32 only)
struct TopK_2
{
    float value;
    int32_t index;

    __device__ __forceinline__ void init()
    {
        value = -FLT_MAX;
        index = -1;
    }

    __device__ __forceinline__ void insert(float elem, int32_t elemId)
    {
        if (elem > value)
        {
            value = elem;
            index = elemId;
        }
    }
};

struct maxOpFunctor
{
    __device__ __forceinline__ float operator()(float const& a, float const& b) const
    {
        return a > b ? a : b;
    }
};

struct sumOpFunctor
{
    __device__ __forceinline__ float operator()(float const& a, float const& b) const
    {
        return a + b;
    }
};

// Reduction operator for top-k (FP32 only)
struct topk2MaxOpFunctor
{
    __device__ __forceinline__ TopK_2 operator()(TopK_2 const& a, TopK_2 const& b) const
    {
        return a.value > b.value ? a : b;
    }
};

// Stage 2 kernel for returnAllTopK - returns indices and raw values only (no transformations)
template <int BLOCK_SIZE>
__global__ void topKStage2ReturnAllTopK(int32_t const* __restrict topKTmpIdBuf, float* topKTmpValBuf,
    int32_t* outputIndices, float* outputValues, int32_t batchSize, int32_t vocabSize, int32_t topK,
    int32_t blocksPerBeam)
{
    auto const tid = static_cast<int32_t>(threadIdx.x);
    auto const batchIdx = static_cast<int32_t>(blockIdx.x);

    if (batchIdx >= batchSize)
        return;

    auto const size = topK * blocksPerBeam;
    auto const stride = topK * blocksPerBeam;

    typedef cub::BlockReduce<TopK_2, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage tempStorage;
    extern __shared__ char array[];
    float* sVal = topKTmpValBuf + batchIdx * stride;
    auto* sId = reinterpret_cast<int32_t*>(array);
    auto sValSelected = reinterpret_cast<float*>(sId + topK);

    TopK_2 partial;

    // Iteratively find top-K elements by max value
    for (int32_t ite = 0; ite < topK; ite++)
    {
        partial.init();
#pragma unroll
        for (int32_t i = tid; i < size; i += BLOCK_SIZE)
        {
            partial.insert(sVal[i], i);
        }

        TopK_2 total = BlockReduce(tempStorage).Reduce(partial, topk2MaxOpFunctor());

        if (tid == 0)
        {
            sId[ite] = total.index;
            sValSelected[ite] = total.value; // Store original value
            sVal[total.index] = -FLT_MAX;    // Mask out selected element for next iteration
        }
        __syncthreads();
    }

    // Write outputs - return indices and original values only
    if (tid == 0)
    {
        for (int32_t ki = 0; ki < topK; ki++)
        {
            auto originalValue = sValSelected[ki];
            auto idx = sId[ki];

            if (outputIndices != nullptr)
            {
                auto outputId = idx != -1 ? topKTmpIdBuf[batchIdx * stride + idx] % vocabSize : vocabSize - 1;
                outputId = outputId == -1 ? vocabSize - 1 : outputId;
                outputIndices[batchIdx * topK + ki] = outputId;
            }

            if (outputValues != nullptr)
            {
                // Return original raw value from input (no transformation)
                outputValues[batchIdx * topK + ki] = originalValue;
            }
        }
    }
}

// Block prefix callback for cumulative sum computation
struct BlockPrefixCallbackOp
{
    float runningTotal;

    __device__ BlockPrefixCallbackOp(float runningTotal)
        : runningTotal(runningTotal)
    {
    }

    __device__ float operator()(float blockAggregate)
    {
        float oldPrefix = runningTotal;
        runningTotal += blockAggregate;
        return oldPrefix;
    }
};

// =======================================================================================
// TOP-K SAMPLING KERNELS (Based on TensorRT-LLM two-stage approach)
// =======================================================================================

// Stage 1: Find top-K elements using iterative block-level reduction (FP32 only)
template <int32_t BLOCK_SIZE_, int32_t BLOCKS_PER_BEAM_>
__global__ void topKStage1(float const* __restrict__ logits, float* tmpLogits, int32_t* topKTmpIdBuf,
    float* topKTmpValBuf, SamplingParams const params)
{
    typedef cub::BlockReduce<TopK_2, BLOCK_SIZE_> BlockReduce;
    __shared__ typename BlockReduce::TempStorage tempStorage;

    auto const tid = static_cast<int32_t>(threadIdx.x);
    auto const bid = static_cast<int32_t>(blockIdx.x);

    auto const batchId = bid / BLOCKS_PER_BEAM_;
    auto const blockLane = bid % BLOCKS_PER_BEAM_;

    if (batchId >= params.batchSize)
        return;

    auto const vocabSize = params.vocabSize;
    auto const k = params.topK;
    auto const temperature = params.temperature;
    auto const invTemp = invTemp_device(temperature);

    auto const tmpLogBufIndex = batchId * vocabSize;
    auto const tmpTopKBufIndex = batchId * BLOCKS_PER_BEAM_ * k + blockLane * k;

    TopK_2 partial;
    float const MAX_T_VAL = FLT_MAX;

    // Copy logits to temporary buffer and apply temperature
    for (auto elemId = tid + blockLane * BLOCK_SIZE_; elemId < vocabSize; elemId += BLOCK_SIZE_ * BLOCKS_PER_BEAM_)
    {
        auto localIndex = elemId + tmpLogBufIndex;
        float logit = logits[localIndex];
        tmpLogits[localIndex] = logit * invTemp;
    }

    // Find top-K elements iteratively
    for (int32_t ite = 0; ite < k; ite++)
    {
        partial.init();
#pragma unroll
        for (auto elemId = tid + blockLane * BLOCK_SIZE_; elemId < vocabSize; elemId += BLOCK_SIZE_ * BLOCKS_PER_BEAM_)
        {
            auto index = elemId + tmpLogBufIndex;
            partial.insert(tmpLogits[index], index);
        }

        TopK_2 total = BlockReduce(tempStorage).Reduce(partial, topk2MaxOpFunctor());

        if (tid == 0)
        {
            auto const index = tmpTopKBufIndex + ite;
            topKTmpIdBuf[index] = total.index;
            topKTmpValBuf[index] = total.value;

            if (total.index >= 0)
            {
                tmpLogits[total.index] = -MAX_T_VAL;
            }
        }
        __syncthreads();
    }
}

// Stage 2: Sample from top-K elements using softmax (FP32 only)
template <int BLOCK_SIZE_>
__global__ void topKStage2Sampling(int32_t const* __restrict__ topKTmpIdBuf, float* topKTmpValBuf,
    int32_t* __restrict__ selectedIndices, SamplingParams const params, uint64_t philoxSeed, uint64_t philoxOffset)
{
    float const MAX_T_VAL = FLT_MAX;

    auto const tid = static_cast<int32_t>(threadIdx.x);
    auto const batchIdx = static_cast<int32_t>(blockIdx.x);

    if (batchIdx >= params.batchSize)
    {
        return;
    }

    auto const k = params.topK;
    auto const topP = params.topP;
    auto const size = k * 8; // BLOCKS_PER_BEAM = 8
    auto const stride = k * 8;

    typedef cub::BlockReduce<TopK_2, BLOCK_SIZE_> BlockReduce;
    __shared__ typename BlockReduce::TempStorage tempStorage;
    extern __shared__ char array[];
    __shared__ float sSum;
    float* sVal = topKTmpValBuf + batchIdx * stride;
    auto* sId = reinterpret_cast<int32_t*>(array);

    if (tid == 0)
    {
        sSum = 0.0f;
    }
    TopK_2 partial;

    auto sVal2 = reinterpret_cast<float*>(sId + k);
    float maxLogit;

    // Sort top-K elements and compute softmax
    for (int32_t ite = 0; ite < k; ite++)
    {
        partial.init();
#pragma unroll
        for (int32_t i = tid; i < size; i += BLOCK_SIZE_)
        {
            partial.insert(sVal[i], i);
        }

        TopK_2 total = BlockReduce(tempStorage).Reduce(partial, topk2MaxOpFunctor());

        if (tid == 0)
        {
            if (ite == 0)
            {
                maxLogit = total.value;
            }

            if (total.index >= 0 && total.index < size)
            {
                sId[ite] = total.index;
                sVal[total.index] = -MAX_T_VAL;

                total.value = __expf(total.value - maxLogit);
                sVal2[ite] = total.value;
                sSum += total.value;
            }
            else
            {
                sId[ite] = -1;
                sVal2[ite] = 0.0f;
            }
        }
        __syncthreads();
    }

    // Sample from the distribution
    if (tid == 0)
    {
        // Initialize curand state for this batch
        curandState_t localState;
        curand_init(philoxSeed, batchIdx, philoxOffset, &localState);

        // When generating random number, topP filtering is applied to ensure the sum of the probabilities is less than
        // topP
        auto randNum = static_cast<float>(curand_uniform(&localState) * topP * sSum);

        for (int32_t ki = 0; ki < k; ki++)
        {
            auto expLogit = sVal2[ki];
            randNum = randNum - expLogit;

            if (randNum <= 0.0f || ki == k - 1)
            {
                auto idx = sId[ki];

                if (idx >= 0 && idx < stride)
                {
                    auto globalIdx = topKTmpIdBuf[batchIdx * stride + idx];
                    auto outputId = (globalIdx != -1) ? (globalIdx % params.vocabSize) : (params.vocabSize - 1);
                    outputId = (outputId == -1) ? (params.vocabSize - 1) : outputId;

                    if (outputId >= 0 && outputId < params.vocabSize)
                    {
                        selectedIndices[batchIdx] = outputId;
                    }
                    else
                    {
                        selectedIndices[batchIdx] = params.vocabSize - 1; // Safe fallback
                    }
                }
                else
                {
                    selectedIndices[batchIdx] = params.vocabSize - 1; // Safe fallback
                }
                break;
            }
        }
    }
}

template <int BLOCK_SIZE>
__global__ void softmaxKernel(
    float const* logits, float* probs, int32_t batchSize, int32_t vocabSize, float temperature)
{
    auto const batchId = static_cast<int32_t>(blockIdx.x);
    auto const tid = static_cast<int32_t>(threadIdx.x);

    if (batchId >= batchSize)
        return;

    auto const offset = batchId * vocabSize;
    auto const invTemp = invTemp_device(temperature);

    typedef cub::BlockReduce<float, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage tempStorage;
    __shared__ float maxLogit;
    __shared__ float sumExp;

    // Find max logit for numerical stability
    float threadMax = -FLT_MAX;
    for (int32_t i = tid; i < vocabSize; i += BLOCK_SIZE)
    {
        auto logit = logits[offset + i] * invTemp;
        threadMax = fmaxf(threadMax, logit);
    }

    // Use customed reductionOp to WAR CUDA12/13 compatibility issue
    float blockMax = BlockReduce(tempStorage).Reduce(threadMax, maxOpFunctor());
    if (tid == 0)
    {
        maxLogit = blockMax;
        sumExp = 0.0f;
    }
    __syncthreads();

    // Compute exp(logit - maxLogit) and sum
    float threadSum = 0.0f;
    for (int32_t i = tid; i < vocabSize; i += BLOCK_SIZE)
    {
        auto logit = logits[offset + i] * invTemp;
        auto expLogit = expf(logit - maxLogit);
        probs[offset + i] = expLogit;
        threadSum += expLogit;
    }

    float blockSum = BlockReduce(tempStorage).Reduce(threadSum, sumOpFunctor());
    if (tid == 0)
    {
        sumExp = blockSum;
    }
    __syncthreads();

    // Normalize to get probabilities
    for (int32_t i = tid; i < vocabSize; i += BLOCK_SIZE)
    {
        auto prob = probs[offset + i] / sumExp;
        probs[offset + i] = prob;
    }
}

// Initialize ID values and offsets for top-p sampling
__global__ void topPInitialize(
    int32_t* topPIdValBuf, int32_t* topPOffsetBuf, int32_t* beginTopPOffsetBuf, int32_t batchSize, int32_t vocabSize)
{
    auto const tid = static_cast<int32_t>(threadIdx.x);
    auto const bid = static_cast<int32_t>(blockIdx.x);

    if (bid == 0)
    {
        for (auto i = tid; i < batchSize + 1; i += static_cast<int32_t>(blockDim.x))
        {
            int32_t expectedOffset = i * vocabSize;
            topPOffsetBuf[i] = expectedOffset;
            beginTopPOffsetBuf[i] = expectedOffset;
        }
    }

    auto index = tid + bid * static_cast<int32_t>(blockDim.x);

    while (index < batchSize * vocabSize)
    {
        topPIdValBuf[index] = index % vocabSize;
        index += static_cast<int32_t>(blockDim.x * gridDim.x);
    }
}

// Early exit optimization: check if highest probability token exceeds threshold (FP32 only)
template <int THREADBLOCK_SIZE>
__launch_bounds__(THREADBLOCK_SIZE) __global__ void topPBeamTopKKernel(float const* probs, int32_t* topKTmpIdBuf,
    float* topKTmpValBuf, int32_t* earlyExitFlags, int32_t vocabSize, float topP, int32_t batchSize)
{
    auto const threadId = static_cast<int32_t>(threadIdx.x);
    auto const batchId = static_cast<int32_t>(blockIdx.x);

    if (batchId >= batchSize)
        return;

    float pThreshold = topP;

    typedef cub::BlockReduce<TopK_2, THREADBLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    TopK_2 partial;

    float const MAX_T_VAL = FLT_MAX;

    partial.value = -MAX_T_VAL;
    partial.index = -1;

#pragma unroll
    for (int32_t elemId = static_cast<int32_t>(threadId); elemId < vocabSize; elemId += THREADBLOCK_SIZE)
    {
        auto index = elemId + batchId * vocabSize;
        partial.insert(probs[index], elemId);
    }

    TopK_2 total = BlockReduce(temp_storage).Reduce(partial, topk2MaxOpFunctor());

    if (threadId == 0)
    {
        float sumProb = total.value;

        if (sumProb >= pThreshold)
        {
            // Early exit: set flag and store the selected token
            earlyExitFlags[batchId] = 1;
            auto index = batchId * vocabSize;
            topKTmpIdBuf[index] = total.index; // Store the actual token ID
            topKTmpValBuf[index] = total.value;
        }
        else
        {
            // No early exit
            earlyExitFlags[batchId] = 0;
        }
    }
}

// Final sampling stage using block-level prefix sum
template <int blockSize>
__global__ void topPSampling(float const* sortedProbs, int32_t const* sortedIdVals, int32_t* selectedIndices,
    int32_t const* topKTmpIdBuf, int32_t const* earlyExitFlags, int32_t vocabSize, uint64_t philoxSeed,
    uint64_t philoxOffset, float topP, int32_t batchSize)
{
    __shared__ float randNumS;

    auto const tid = static_cast<int32_t>(threadIdx.x);
    auto const batchId = static_cast<int32_t>(blockIdx.x);

    if (batchId >= batchSize)
        return;

    auto const probThreshold = topP;

    if (threadIdx.x == 0)
    {
        // Initialize curand state for this batch
        curandState_t localState;
        curand_init(philoxSeed, batchId, philoxOffset, &localState);

        auto const randomNumber = curand_uniform(&localState);
        randNumS = randomNumber * probThreshold;
    }

    typedef cub::BlockScan<float, blockSize> BlockScan;
    __shared__ typename BlockScan::TempStorage tempStorage;
    BlockPrefixCallbackOp prefixOp(0);

    __syncthreads();

    auto offset = batchId * vocabSize;
    auto end = ((vocabSize + blockSize - 1) / blockSize) * blockSize;
    int32_t selectedTokenId = 0;
    float threadOffset = 0;
    int32_t count = 0;

    for (int vi = tid; vi < end; vi += blockSize)
    {
        auto threadProb = (vi < vocabSize) ? (sortedProbs[offset + vi]) : 0.f;
        BlockScan(tempStorage).InclusiveSum(threadProb, threadOffset, prefixOp);
        count = __syncthreads_count(randNumS <= threadOffset);
        selectedTokenId = vi;

        if (count != 0)
        {
            break;
        }
    }

    selectedTokenId = min(selectedTokenId, vocabSize - 1);

    if (threadIdx.x == min(blockDim.x - count, blockDim.x - 1))
    {
        int32_t finalToken = sortedIdVals[offset + selectedTokenId];
        selectedIndices[batchId] = finalToken;
    }
}

// Updated sampling function with tensor-based interface (FP32 only)
void topKtopPSamplingFromLogits(rt::Tensor const& logits, rt::Tensor& selectedIndices, SamplingParams const& params,
    rt::Tensor& workspace, cudaStream_t stream, uint64_t philoxSeed, uint64_t philoxOffset)
{
    // Validate input tensors
    check::check(logits.getDeviceType() == rt::DeviceType::kGPU
            && selectedIndices.getDeviceType() == rt::DeviceType::kGPU
            && workspace.getDeviceType() == rt::DeviceType::kGPU,
        "All tensors must be on GPU");
    check::check(logits.getDataType() == nvinfer1::DataType::kFLOAT
            && selectedIndices.getDataType() == nvinfer1::DataType::kINT32
            && workspace.getDataType() == nvinfer1::DataType::kINT8,
        "Invalid tensor data types");

    auto logitsShape = logits.getShape();
    auto selectedIndicesShape = selectedIndices.getShape();
    check::check(logitsShape.getNumDims() == 2 && selectedIndicesShape.getNumDims() == 2, "Invalid tensor dimensions");
    check::check(logitsShape[0] == params.batchSize && logitsShape[1] == params.vocabSize,
        "Logits tensor shape mismatch with parameters");
    check::check(selectedIndicesShape[0] == params.batchSize && selectedIndicesShape[1] == 1,
        "Selected indices tensor shape mismatch with parameters");

    int const BLOCK_SIZE = 256;
    int const BLOCKS_PER_BEAM = 8;

    // Setup workspace partitioning
    SamplingWorkspace ws;
    ws.setupWorkspace(workspace.rawPointer(), workspace.getMemoryCapacity(), params);

    // Validate workspace buffers
    if (params.useTopK)
    {
        assert(ws.topkTempLogits != nullptr && ws.topkIndices != nullptr && ws.topkValues != nullptr);
    }

    if (params.useTopK)
    {

        // Stage 1: Find top-K elements
        dim3 grid1(params.batchSize * BLOCKS_PER_BEAM);
        dim3 block1(BLOCK_SIZE);

        topKStage1<BLOCK_SIZE, BLOCKS_PER_BEAM><<<grid1, block1, 0, stream>>>(
            logits.dataPointer<float>(), ws.topkTempLogits, ws.topkIndices, ws.topkValues, params);

        // Stage 2: Sample from top-K elements
        dim3 grid2(params.batchSize);
        dim3 block2(BLOCK_SIZE);
        size_t sharedMemSize = params.topK * sizeof(int32_t) + params.topK * sizeof(float);

        topKStage2Sampling<BLOCK_SIZE><<<grid2, block2, sharedMemSize, stream>>>(
            ws.topkIndices, ws.topkValues, selectedIndices.dataPointer<int32_t>(), params, philoxSeed, philoxOffset);
    }
    else if (params.useTopP)
    {
        // Top-P only sampling using workspace

        // Stage 0: Convert logits to probabilities using softmax
        int const SOFTMAX_BLOCK_SIZE = 256;

        softmaxKernel<SOFTMAX_BLOCK_SIZE><<<params.batchSize, SOFTMAX_BLOCK_SIZE, 0, stream>>>(
            logits.dataPointer<float>(), ws.toppProbs, params.batchSize, params.vocabSize, params.temperature);

        // Stage 1: Initialize
        topPInitialize<<<32, 512, 0, stream>>>(
            ws.toppIdVals, ws.toppOffsetBuf, ws.toppBeginOffsetBuf, params.batchSize, params.vocabSize);

        // Stage 2: Early exit optimization
        int const BLOCK_SIZE_TOPK = 256;

        topPBeamTopKKernel<BLOCK_SIZE_TOPK><<<params.batchSize, BLOCK_SIZE_TOPK, 0, stream>>>(ws.toppProbs,
            ws.toppTopKIndices, ws.toppTopKValues, ws.toppEarlyExitFlags, params.vocabSize, params.topP,
            params.batchSize);

        // Stage 3: Sort probabilities in descending order
        size_t cubTempStorageSize;
        cub::DeviceSegmentedRadixSort::SortPairsDescending(nullptr, cubTempStorageSize, static_cast<float*>(nullptr),
            static_cast<float*>(nullptr), static_cast<int32_t*>(nullptr), static_cast<int32_t*>(nullptr),
            static_cast<int32_t>(params.vocabSize * params.batchSize), params.batchSize, static_cast<int32_t*>(nullptr),
            static_cast<int32_t*>(nullptr));

        cub::DeviceSegmentedRadixSort::SortPairsDescending(ws.toppTempStorage, cubTempStorageSize, ws.toppProbs,
            ws.toppSortedProbs, ws.toppIdVals, ws.toppSortedIdVals, params.vocabSize * params.batchSize,
            params.batchSize, ws.toppBeginOffsetBuf, ws.toppOffsetBuf + 1, 0, sizeof(float) * 8, stream);

        // Stage 4: Sample using block-level prefix sum
        int const SAMPLING_BLOCK_SIZE = 256;

        topPSampling<SAMPLING_BLOCK_SIZE><<<params.batchSize, SAMPLING_BLOCK_SIZE, 0, stream>>>(ws.toppSortedProbs,
            ws.toppSortedIdVals, selectedIndices.dataPointer<int32_t>(), ws.toppTopKIndices, ws.toppEarlyExitFlags,
            params.vocabSize, philoxSeed, philoxOffset, params.topP, params.batchSize);
    }
}

void selectAllTopK(rt::Tensor const& input, rt::OptionalOutputTensor topKValues, rt::Tensor& topKIndices, int32_t topK,
    rt::Tensor& workspace, cudaStream_t stream)
{
    // Validate input tensors
    check::check(input.getDeviceType() == rt::DeviceType::kGPU && topKIndices.getDeviceType() == rt::DeviceType::kGPU
            && workspace.getDeviceType() == rt::DeviceType::kGPU,
        "All tensors must be on GPU");
    check::check(input.getDataType() == nvinfer1::DataType::kFLOAT
            && topKIndices.getDataType() == nvinfer1::DataType::kINT32
            && workspace.getDataType() == nvinfer1::DataType::kINT8,
        "Invalid tensor data types");

    auto inputShape = input.getShape();
    auto topKIndicesShape = topKIndices.getShape();
    check::check(inputShape.getNumDims() == 2 && topKIndicesShape.getNumDims() == 2, "Invalid tensor dimensions");

    int32_t batchSize = inputShape[0];
    int32_t vocabSize = inputShape[1];

    check::check(topKIndicesShape[0] == batchSize && topKIndicesShape[1] == topK, "TopK indices tensor shape mismatch");

    if (topKValues.has_value())
    {
        rt::Tensor& topKValuesTensor = topKValues.value().get();
        check::check(topKValuesTensor.getDeviceType() == rt::DeviceType::kGPU
                && topKValuesTensor.getDataType() == nvinfer1::DataType::kFLOAT,
            "TopK values tensor must be on GPU with float data type");
        auto topKValuesShape = topKValuesTensor.getShape();
        check::check(topKValuesShape.getNumDims() == 2 && topKValuesShape[0] == batchSize && topKValuesShape[1] == topK,
            "TopK values tensor shape mismatch");
    }

    if (topK <= 0 || topK > vocabSize)
    {
        return;
    }

    constexpr int32_t BLOCK_SIZE = 256;
    constexpr int32_t BLOCKS_PER_BEAM = 8;

    // Setup workspace partitioning
    SamplingWorkspace ws;
    ws.setupWorkspaceForTopK(workspace.rawPointer(), workspace.getMemoryCapacity(), batchSize, vocabSize, topK);

    // Create sampling parameters with temperature = 1.0 (no scaling applied to values)
    SamplingParams params(batchSize, vocabSize, 1.0f, topK);

    // Stage 1: Find top-K elements (temperature=1.0 means no value transformation)
    dim3 grid1(batchSize * BLOCKS_PER_BEAM);
    dim3 block1(BLOCK_SIZE);

    topKStage1<BLOCK_SIZE, BLOCKS_PER_BEAM><<<grid1, block1, 0, stream>>>(
        input.dataPointer<float>(), ws.topkTempLogits, ws.topkIndices, ws.topkValues, params);

    // Stage 2: Select final top-K from 8*K candidates and return indices + raw values
    dim3 grid2(batchSize);
    dim3 block2(BLOCK_SIZE);
    size_t sharedMemSize = topK * sizeof(int32_t) + topK * sizeof(float);

    float* topKValuesPtr = topKValues.has_value() ? topKValues.value().get().dataPointer<float>() : nullptr;
    topKStage2ReturnAllTopK<BLOCK_SIZE><<<grid2, block2, sharedMemSize, stream>>>(ws.topkIndices, ws.topkValues,
        topKIndices.dataPointer<int32_t>(), topKValuesPtr, batchSize, vocabSize, topK, BLOCKS_PER_BEAM);
}

// =======================================================================================
// VOCABULARY MAPPING KERNEL
// =======================================================================================

/*!
 * \brief CUDA kernel for mapping reduced vocabulary IDs to full vocabulary IDs (in-place)
 *
 * Each thread processes one element, performing a simple lookup operation:
 * vocabIds[idx] = vocabMappingTable[vocabIds[idx]]
 *
 * The operation is performed in-place, reading and writing to the same memory location.
 */
__global__ void mapReducedVocabToFullVocabKernel(
    int32_t* vocabIds, int32_t const* __restrict__ vocabMappingTable, int32_t totalElements)
{
    int32_t idx = static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);

    if (idx < totalElements)
    {
        // Read the reduced vocab ID, then overwrite with mapped full vocab ID
        int32_t reducedId = vocabIds[idx];
        vocabIds[idx] = vocabMappingTable[reducedId];
    }
}

void mapReducedVocabToFullVocab(rt::Tensor& vocabIds, rt::Tensor const& vocabMappingTable, cudaStream_t stream)
{
    // Validate device types
    check::check(
        vocabIds.getDeviceType() == rt::DeviceType::kGPU && vocabMappingTable.getDeviceType() == rt::DeviceType::kGPU,
        "All tensors must be on GPU");

    // Validate data types
    check::check(vocabIds.getDataType() == nvinfer1::DataType::kINT32
            && vocabMappingTable.getDataType() == nvinfer1::DataType::kINT32,
        "All tensors must have INT32 data type");

    // Calculate total number of elements
    int32_t totalElements = static_cast<int32_t>(vocabIds.getShape().volume());

    if (totalElements == 0)
    {
        return; // Nothing to map
    }

    // Launch kernel
    constexpr int32_t BLOCK_SIZE = 256;
    int32_t numBlocks = (totalElements + BLOCK_SIZE - 1) / BLOCK_SIZE;

    mapReducedVocabToFullVocabKernel<<<numBlocks, BLOCK_SIZE, 0, stream>>>(
        vocabIds.dataPointer<int32_t>(), vocabMappingTable.dataPointer<int32_t>(), totalElements);
}

} // namespace trt_edgellm
