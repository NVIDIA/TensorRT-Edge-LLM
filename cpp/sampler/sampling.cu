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
        ELLM_CHECK(offset <= workspaceSize,
            "Workspace size too small. Required: " + std::to_string(offset)
                + ", provided: " + std::to_string(workspaceSize));
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
        ELLM_CHECK(offset <= workspaceSize,
            "Workspace size too small for topK. Required: " + std::to_string(offset)
                + ", provided: " + std::to_string(workspaceSize));
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
size_t getSelectAllTopKWorkspaceSize(int32_t batchSize, int32_t vocabSize, int32_t topK) noexcept
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

template <typename T>
__device__ float logitToFloat(T value)
{
    return static_cast<float>(value);
}

template <>
__device__ float logitToFloat<half>(half value)
{
    return __half2float(value);
}

template <>
__device__ float logitToFloat<__nv_bfloat16>(__nv_bfloat16 value)
{
    return __bfloat162float(value);
}

template <typename T, int32_t BLOCK_SIZE_>
__global__ void argmaxEntropyKernel(T const* __restrict__ logits, int32_t* __restrict__ topIndices,
    float* __restrict__ entropy, int32_t rows, int32_t vocabSize, float temperature)
{
    using TopKReduce = cub::BlockReduce<TopK_2, BLOCK_SIZE_>;
    using FloatReduce = cub::BlockReduce<float, BLOCK_SIZE_>;

    __shared__ typename TopKReduce::TempStorage topKStorage;
    __shared__ typename FloatReduce::TempStorage floatStorage;
    __shared__ float rowMax;
    __shared__ float rowSumExp;

    auto const row = static_cast<int32_t>(blockIdx.x);
    auto const tid = static_cast<int32_t>(threadIdx.x);
    if (row >= rows)
    {
        return;
    }

    int64_t const rowOffset = static_cast<int64_t>(row) * vocabSize;
    auto const invTemp = invTemp_device(temperature);

    TopK_2 partial;
    partial.init();
    for (int32_t v = tid; v < vocabSize; v += BLOCK_SIZE_)
    {
        float const logit = logitToFloat(logits[rowOffset + v]) * invTemp;
        partial.insert(logit, v);
    }

    TopK_2 const best = TopKReduce(topKStorage).Reduce(partial, topk2MaxOpFunctor());
    if (tid == 0)
    {
        rowMax = best.value;
        topIndices[row] = best.index >= 0 ? best.index : vocabSize - 1;
    }
    __syncthreads();

    float localSumExp = 0.0F;
    float localWeightedShift = 0.0F;
    for (int32_t v = tid; v < vocabSize; v += BLOCK_SIZE_)
    {
        float const shifted = logitToFloat(logits[rowOffset + v]) * invTemp - rowMax;
        float const expShifted = expf(shifted);
        localSumExp += expShifted;
        localWeightedShift += expShifted * shifted;
    }

    float const sumExp = FloatReduce(floatStorage).Reduce(localSumExp, sumOpFunctor());
    if (tid == 0)
    {
        rowSumExp = sumExp;
    }
    __syncthreads();

    float const weightedShift = FloatReduce(floatStorage).Reduce(localWeightedShift, sumOpFunctor());
    if (tid == 0)
    {
        entropy[row] = rowSumExp > 0.0F ? logf(rowSumExp) - weightedShift / rowSumExp : 0.0F;
    }
}

// Stage 1: Find top-K elements using iterative block-level reduction.
template <typename T, int32_t BLOCK_SIZE_, int32_t BLOCKS_PER_BEAM_>
__global__ void topKStage1(T const* __restrict__ logits, float* tmpLogits, int32_t* topKTmpIdBuf, float* topKTmpValBuf,
    SamplingParams const params)
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
        float logit = logitToFloat(logits[localIndex]);
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

// Unified softmax / log-softmax kernel.
// LOG_OUTPUT=false: output[i] = exp(x[i]*invT - max) / sum_exp  (probabilities for sampling)
// LOG_OUTPUT=true:  output[i] = x[i]*invT - max - log(sum_exp)  (log-probabilities, T=1 for logprobs)
// Use customed reductionOp to WAR CUDA12/13 compatibility issue
template <int BLOCK_SIZE, bool LOG_OUTPUT>
__global__ void softmaxKernelImpl(
    float const* logits, float* output, int32_t batchSize, int32_t vocabSize, float temperature)
{
    auto const batchId = static_cast<int32_t>(blockIdx.x);
    auto const tid = static_cast<int32_t>(threadIdx.x);

    if (batchId >= batchSize)
        return;

    auto const offset = batchId * vocabSize;
    auto const invTemp = invTemp_device(temperature);

    typedef cub::BlockReduce<float, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage tempStorage;
    __shared__ float sharedMax;
    __shared__ float sharedAgg; // sum_exp (softmax) or log(sum_exp) (log-softmax)

    // Pass 1: find max for numerical stability
    float threadMax = -FLT_MAX;
    for (int32_t i = tid; i < vocabSize; i += BLOCK_SIZE)
        threadMax = fmaxf(threadMax, logits[offset + i] * invTemp);
    float blockMax = BlockReduce(tempStorage).Reduce(threadMax, maxOpFunctor());
    if (tid == 0)
    {
        sharedMax = blockMax;
        sharedAgg = 0.0f;
    }
    __syncthreads();

    // Pass 2: compute exp(x*invT - max), store intermediates for softmax, accumulate sum
    float threadSum = 0.0f;
    for (int32_t i = tid; i < vocabSize; i += BLOCK_SIZE)
    {
        float expVal = expf(logits[offset + i] * invTemp - sharedMax);
        if constexpr (!LOG_OUTPUT)
            output[offset + i] = expVal; // store for in-place normalization in pass 3
        threadSum += expVal;
    }
    float blockSum = BlockReduce(tempStorage).Reduce(threadSum, sumOpFunctor());
    if (tid == 0)
        sharedAgg = LOG_OUTPUT ? logf(blockSum) : blockSum;
    __syncthreads();

    // Pass 3: write final output
    for (int32_t i = tid; i < vocabSize; i += BLOCK_SIZE)
    {
        if constexpr (LOG_OUTPUT)
            output[offset + i] = logits[offset + i] * invTemp - sharedMax - sharedAgg;
        else
            output[offset + i] /= sharedAgg;
    }
}

__global__ void applyLogitBiasKernel(
    float* logits, int32_t const* tokenIds, float const* biasValues, int32_t const* offsets, int32_t vocabSize)
{
    int32_t const batchId = static_cast<int32_t>(blockIdx.x);
    int32_t const begin = offsets[batchId];
    int32_t const end = offsets[batchId + 1];
    float* rowLogits = logits + static_cast<int64_t>(batchId) * vocabSize;

    for (int32_t entryIdx = begin + static_cast<int32_t>(threadIdx.x); entryIdx < end;
        entryIdx += static_cast<int32_t>(blockDim.x))
    {
        int32_t const tokenId = tokenIds[entryIdx];
        if (tokenId >= 0 && tokenId < vocabSize)
        {
            rowLogits[tokenId] += biasValues[entryIdx];
        }
    }
}

__global__ void applyLogitBiasRepeatedRowsKernel(float* logits, int32_t const* tokenIds, float const* biasValues,
    int32_t const* offsets, int32_t vocabSize, int32_t rowsPerSlot)
{
    int32_t const rowId = static_cast<int32_t>(blockIdx.x);
    int32_t const slotId = rowId / rowsPerSlot;
    int32_t const begin = offsets[slotId];
    int32_t const end = offsets[slotId + 1];
    float* rowLogits = logits + static_cast<int64_t>(rowId) * vocabSize;

    for (int32_t entryIdx = begin + static_cast<int32_t>(threadIdx.x); entryIdx < end;
        entryIdx += static_cast<int32_t>(blockDim.x))
    {
        int32_t const tokenId = tokenIds[entryIdx];
        if (tokenId >= 0 && tokenId < vocabSize)
        {
            rowLogits[tokenId] += biasValues[entryIdx];
        }
    }
}

void validateLogitBiasInputs(rt::Tensor const& logits, rt::Tensor const& tokenIds, rt::Tensor const& biasValues,
    rt::Tensor const& offsets, int32_t rowsPerSlot)
{
    check::check(logits.getDeviceType() == rt::DeviceType::kGPU && tokenIds.getDeviceType() == rt::DeviceType::kGPU
            && biasValues.getDeviceType() == rt::DeviceType::kGPU && offsets.getDeviceType() == rt::DeviceType::kGPU,
        "All tensors must be on GPU");
    check::check(logits.getDataType() == nvinfer1::DataType::kFLOAT
            && tokenIds.getDataType() == nvinfer1::DataType::kINT32
            && biasValues.getDataType() == nvinfer1::DataType::kFLOAT
            && offsets.getDataType() == nvinfer1::DataType::kINT32,
        "Invalid tensor data types");

    auto const logitsShape = logits.getShape();
    auto const tokenIdsShape = tokenIds.getShape();
    auto const biasValuesShape = biasValues.getShape();
    auto const offsetsShape = offsets.getShape();

    check::check(logitsShape.getNumDims() == 2 && tokenIdsShape.getNumDims() == 1 && biasValuesShape.getNumDims() == 1
            && offsetsShape.getNumDims() == 1,
        "Invalid tensor dimensions");
    check::check(tokenIdsShape[0] == biasValuesShape[0], "Logit bias token/value shape mismatch");
    check::check(rowsPerSlot > 0, "Logit bias rowsPerSlot must be positive");
    check::check(logitsShape[0] % rowsPerSlot == 0, "Logit bias repeated-row logits shape mismatch");
    check::check(offsetsShape[0] == logitsShape[0] / rowsPerSlot + 1, "Logit bias offsets shape mismatch");
}

void applyLogitBias(rt::Tensor& logits, rt::Tensor const& tokenIds, rt::Tensor const& biasValues,
    rt::Tensor const& offsets, cudaStream_t stream)
{
    validateLogitBiasInputs(logits, tokenIds, biasValues, offsets, 1);

    auto const logitsShape = logits.getShape();
    auto const tokenIdsShape = tokenIds.getShape();

    int32_t const batchSize = static_cast<int32_t>(logitsShape[0]);
    int32_t const vocabSize = static_cast<int32_t>(logitsShape[1]);
    int64_t const numBiasedTokens = tokenIdsShape[0];
    if (batchSize == 0 || vocabSize == 0 || numBiasedTokens == 0)
    {
        return;
    }

    constexpr int32_t kBLOCK_SIZE = 256;
    applyLogitBiasKernel<<<batchSize, kBLOCK_SIZE, 0, stream>>>(logits.dataPointer<float>(),
        tokenIds.dataPointer<int32_t>(), biasValues.dataPointer<float>(), offsets.dataPointer<int32_t>(), vocabSize);
}

void applyLogitBiasRepeatedRows(rt::Tensor& logits, rt::Tensor const& tokenIds, rt::Tensor const& biasValues,
    rt::Tensor const& offsets, int32_t rowsPerSlot, cudaStream_t stream)
{
    validateLogitBiasInputs(logits, tokenIds, biasValues, offsets, rowsPerSlot);

    auto const logitsShape = logits.getShape();
    auto const tokenIdsShape = tokenIds.getShape();

    int32_t const totalRows = static_cast<int32_t>(logitsShape[0]);
    int32_t const vocabSize = static_cast<int32_t>(logitsShape[1]);
    int64_t const numBiasedTokens = tokenIdsShape[0];
    if (totalRows == 0 || vocabSize == 0 || numBiasedTokens == 0)
    {
        return;
    }

    constexpr int32_t kBLOCK_SIZE = 256;
    applyLogitBiasRepeatedRowsKernel<<<totalRows, kBLOCK_SIZE, 0, stream>>>(logits.dataPointer<float>(),
        tokenIds.dataPointer<int32_t>(), biasValues.dataPointer<float>(), offsets.dataPointer<int32_t>(), vocabSize,
        rowsPerSlot);
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

        topKStage1<float, BLOCK_SIZE, BLOCKS_PER_BEAM><<<grid1, block1, 0, stream>>>(
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

        softmaxKernelImpl<SOFTMAX_BLOCK_SIZE, false><<<params.batchSize, SOFTMAX_BLOCK_SIZE, 0, stream>>>(
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

template <int BLOCK_SIZE>
__global__ void topKLogitsToDenseProbabilitiesKernel(float const* __restrict__ topKValues,
    int32_t const* __restrict__ topKIndices, float* __restrict__ probabilities, int32_t batchSize, int32_t vocabSize,
    int32_t topK, float temperature)
{
    int32_t const batchIdx = static_cast<int32_t>(blockIdx.x);
    int32_t const tid = static_cast<int32_t>(threadIdx.x);
    if (batchIdx >= batchSize || vocabSize <= 0 || topK <= 0)
    {
        return;
    }

    float* rowProbs = probabilities + static_cast<int64_t>(batchIdx) * vocabSize;
    for (int32_t vocabIdx = tid; vocabIdx < vocabSize; vocabIdx += BLOCK_SIZE)
    {
        rowProbs[vocabIdx] = 0.0F;
    }
    __syncthreads();

    int32_t const effectiveTopK = topK < vocabSize ? topK : vocabSize;
    float const invTemp = invTemp_device(temperature);
    float const* rowValues = topKValues + static_cast<int64_t>(batchIdx) * topK;
    int32_t const* rowIndices = topKIndices + static_cast<int64_t>(batchIdx) * topK;

    typedef cub::BlockReduce<float, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage tempStorage;
    __shared__ float sharedMax;
    __shared__ float sharedSum;

    float localMax = -FLT_MAX;
    for (int32_t k = tid; k < effectiveTopK; k += BLOCK_SIZE)
    {
        int32_t const token = rowIndices[k];
        if (token >= 0 && token < vocabSize)
        {
            localMax = fmaxf(localMax, rowValues[k] * invTemp);
        }
    }
    float const maxValue = BlockReduce(tempStorage).Reduce(localMax, maxOpFunctor());
    if (tid == 0)
    {
        sharedMax = maxValue;
    }
    __syncthreads();

    float localSum = 0.0F;
    for (int32_t k = tid; k < effectiveTopK; k += BLOCK_SIZE)
    {
        int32_t const token = rowIndices[k];
        if (token >= 0 && token < vocabSize)
        {
            localSum += expf(rowValues[k] * invTemp - sharedMax);
        }
    }
    float const sumValue = BlockReduce(tempStorage).Reduce(localSum, sumOpFunctor());
    if (tid == 0)
    {
        sharedSum = sumValue;
    }
    __syncthreads();

    if (sharedSum <= 0.0F || !isfinite(sharedSum))
    {
        for (int32_t k = tid; k < effectiveTopK; k += BLOCK_SIZE)
        {
            int32_t const token = rowIndices[k];
            if (token >= 0 && token < vocabSize)
            {
                rowProbs[token] = 1.0F / static_cast<float>(effectiveTopK);
            }
        }
        return;
    }

    for (int32_t k = tid; k < effectiveTopK; k += BLOCK_SIZE)
    {
        int32_t const token = rowIndices[k];
        if (token >= 0 && token < vocabSize)
        {
            rowProbs[token] = expf(rowValues[k] * invTemp - sharedMax) / sharedSum;
        }
    }
}

void topKLogitsToDenseProbabilities(rt::Tensor const& topKValues, rt::Tensor const& topKIndices,
    rt::Tensor& probabilities, int32_t vocabSize, float temperature, cudaStream_t stream)
{
    check::check(topKValues.getDeviceType() == rt::DeviceType::kGPU
            && topKIndices.getDeviceType() == rt::DeviceType::kGPU
            && probabilities.getDeviceType() == rt::DeviceType::kGPU,
        "All tensors must be on GPU");
    check::check(topKValues.getDataType() == nvinfer1::DataType::kFLOAT
            && topKIndices.getDataType() == nvinfer1::DataType::kINT32
            && probabilities.getDataType() == nvinfer1::DataType::kFLOAT,
        "Invalid tensor data types");

    auto const topKValuesShape = topKValues.getShape();
    auto const topKIndicesShape = topKIndices.getShape();
    auto const probabilitiesShape = probabilities.getShape();
    check::check(
        topKValuesShape.getNumDims() == 2 && topKIndicesShape.getNumDims() == 2 && probabilitiesShape.getNumDims() == 2,
        "Invalid tensor dimensions");
    int32_t const batchSize = topKValuesShape[0];
    int32_t const topK = topKValuesShape[1];
    check::check(topKIndicesShape[0] == batchSize && topKIndicesShape[1] == topK, "TopK tensor shape mismatch");
    check::check(
        probabilitiesShape[0] == batchSize && probabilitiesShape[1] == vocabSize, "Probability tensor shape mismatch");

    constexpr int32_t BLOCK_SIZE = 256;
    topKLogitsToDenseProbabilitiesKernel<BLOCK_SIZE>
        <<<batchSize, BLOCK_SIZE, 0, stream>>>(topKValues.dataPointer<float>(), topKIndices.dataPointer<int32_t>(),
            probabilities.dataPointer<float>(), batchSize, vocabSize, topK, temperature);
    CUDA_CHECK(cudaGetLastError());
}

void selectAllTopK(rt::Tensor const& input, rt::OptionalOutputTensor topKValues, rt::Tensor& topKIndices, int32_t topK,
    rt::Tensor& workspace, cudaStream_t stream)
{
    // Validate input tensors
    check::check(input.getDeviceType() == rt::DeviceType::kGPU && topKIndices.getDeviceType() == rt::DeviceType::kGPU
            && workspace.getDeviceType() == rt::DeviceType::kGPU,
        "All tensors must be on GPU");
    check::check((input.getDataType() == nvinfer1::DataType::kFLOAT || input.getDataType() == nvinfer1::DataType::kHALF)
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

    if (input.getDataType() == nvinfer1::DataType::kHALF)
    {
        topKStage1<half, BLOCK_SIZE, BLOCKS_PER_BEAM><<<grid1, block1, 0, stream>>>(
            input.dataPointer<half>(), ws.topkTempLogits, ws.topkIndices, ws.topkValues, params);
    }
    else
    {
        topKStage1<float, BLOCK_SIZE, BLOCKS_PER_BEAM><<<grid1, block1, 0, stream>>>(
            input.dataPointer<float>(), ws.topkTempLogits, ws.topkIndices, ws.topkValues, params);
    }

    // Stage 2: Select final top-K from 8*K candidates and return indices + raw values
    dim3 grid2(batchSize);
    dim3 block2(BLOCK_SIZE);
    size_t sharedMemSize = topK * sizeof(int32_t) + topK * sizeof(float);

    float* topKValuesPtr = topKValues.has_value() ? topKValues.value().get().dataPointer<float>() : nullptr;
    topKStage2ReturnAllTopK<BLOCK_SIZE><<<grid2, block2, sharedMemSize, stream>>>(ws.topkIndices, ws.topkValues,
        topKIndices.dataPointer<int32_t>(), topKValuesPtr, batchSize, vocabSize, topK, BLOCKS_PER_BEAM);
}

void selectArgmaxAndComputeEntropy(
    rt::Tensor const& input, rt::Tensor& topIndices, rt::Tensor& entropy, float temperature, cudaStream_t stream)
{
    check::check(input.getDeviceType() == rt::DeviceType::kGPU && topIndices.getDeviceType() == rt::DeviceType::kGPU
            && entropy.getDeviceType() == rt::DeviceType::kGPU,
        "All tensors must be on GPU");
    check::check((input.getDataType() == nvinfer1::DataType::kFLOAT || input.getDataType() == nvinfer1::DataType::kHALF
                     || input.getDataType() == nvinfer1::DataType::kBF16)
            && topIndices.getDataType() == nvinfer1::DataType::kINT32
            && entropy.getDataType() == nvinfer1::DataType::kFLOAT,
        "Invalid tensor data types");

    auto const inputShape = input.getShape();
    auto const indexShape = topIndices.getShape();
    auto const entropyShape = entropy.getShape();
    check::check(inputShape.getNumDims() == 2 && indexShape.getNumDims() == 2 && entropyShape.getNumDims() == 1,
        "Invalid tensor dimensions");

    int32_t const rows = inputShape[0];
    int32_t const vocabSize = inputShape[1];
    check::check(indexShape[0] == rows && indexShape[1] == 1, "Top index tensor shape mismatch");
    check::check(entropyShape[0] == rows, "Entropy tensor shape mismatch");
    if (rows <= 0 || vocabSize <= 0)
    {
        return;
    }

    constexpr int32_t kBlockSize = 256;
    dim3 const grid(rows);
    dim3 const block(kBlockSize);
    if (input.getDataType() == nvinfer1::DataType::kHALF)
    {
        argmaxEntropyKernel<half, kBlockSize><<<grid, block, 0, stream>>>(input.dataPointer<half>(),
            topIndices.dataPointer<int32_t>(), entropy.dataPointer<float>(), rows, vocabSize, temperature);
    }
    else if (input.getDataType() == nvinfer1::DataType::kBF16)
    {
        argmaxEntropyKernel<__nv_bfloat16, kBlockSize><<<grid, block, 0, stream>>>(input.dataPointer<__nv_bfloat16>(),
            topIndices.dataPointer<int32_t>(), entropy.dataPointer<float>(), rows, vocabSize, temperature);
    }
    else
    {
        argmaxEntropyKernel<float, kBlockSize><<<grid, block, 0, stream>>>(input.dataPointer<float>(),
            topIndices.dataPointer<int32_t>(), entropy.dataPointer<float>(), rows, vocabSize, temperature);
    }
    CUDA_CHECK(cudaGetLastError());
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

namespace
{
//! 256-byte alignment keeps the top-K temp storage that follows the log-softmax
//! buffer inside the logprobs workspace CUB-friendly.
constexpr size_t kLogprobsBufferAlignment = 256;

size_t alignedLogSoftmaxBufferBytes(int32_t batchSize, int32_t vocabSize) noexcept
{
    size_t const bytes = static_cast<size_t>(batchSize) * vocabSize * sizeof(float);
    return (bytes + kLogprobsBufferAlignment - 1) / kLogprobsBufferAlignment * kLogprobsBufferAlignment;
}

//! Tensor-level launcher for the numerically-stable log-softmax kernel.
void logSoftmax(rt::Tensor const& logits, rt::Tensor& out, cudaStream_t stream)
{
    auto const shape = logits.getShape();
    int32_t const batchSize = static_cast<int32_t>(shape[0]);
    int32_t const vocabSize = static_cast<int32_t>(shape[1]);
    constexpr int32_t LOG_SOFTMAX_BLOCK = 256;
    softmaxKernelImpl<LOG_SOFTMAX_BLOCK, true><<<batchSize, LOG_SOFTMAX_BLOCK, 0, stream>>>(
        logits.dataPointer<float>(), out.dataPointer<float>(), batchSize, vocabSize, 1.0f);
}
} // namespace

size_t getExtractTopKLogprobsWorkspaceSize(int32_t batchSize, int32_t vocabSize, int32_t topK) noexcept
{
    // Log-softmax buffer up front (256B-aligned), selectAllTopK temp storage after it.
    return alignedLogSoftmaxBufferBytes(batchSize, vocabSize)
        + getSelectAllTopKWorkspaceSize(batchSize, vocabSize, topK);
}

void extractTopKLogprobs(rt::Tensor const& logits, rt::Tensor& logprobsValues, rt::Tensor& logprobsIndices,
    int32_t topK, rt::Tensor& workspace, cudaStream_t stream)
{
    check::check(logits.getDeviceType() == rt::DeviceType::kGPU
            && logprobsValues.getDeviceType() == rt::DeviceType::kGPU
            && logprobsIndices.getDeviceType() == rt::DeviceType::kGPU
            && workspace.getDeviceType() == rt::DeviceType::kGPU,
        "All tensors must be on GPU");

    auto const inputShape = logits.getShape();
    int32_t const batchSize = static_cast<int32_t>(inputShape[0]);
    int32_t const vocabSize = static_cast<int32_t>(inputShape[1]);

    // Carve the workspace: log-softmax buffer up front, top-K temp storage after it.
    size_t const bufferBytes = alignedLogSoftmaxBufferBytes(batchSize, vocabSize);
    check::check(static_cast<size_t>(workspace.getMemoryCapacity())
            >= getExtractTopKLogprobsWorkspaceSize(batchSize, vocabSize, topK),
        "Logprobs workspace too small; size it with getExtractTopKLogprobsWorkspaceSize()");
    int8_t* const wsBase = workspace.dataPointer<int8_t>();
    rt::Tensor logSoftmaxOut(wsBase, {batchSize, vocabSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT);
    rt::Tensor topKWorkspace(wsBase + bufferBytes,
        {static_cast<int64_t>(workspace.getMemoryCapacity() - static_cast<int64_t>(bufferBytes))}, rt::DeviceType::kGPU,
        nvinfer1::DataType::kINT8);

    // Compute numerically-stable log-softmax, then select top-K (values are already log-probs).
    logSoftmax(logits, logSoftmaxOut, stream);
    selectAllTopK(logSoftmaxOut, rt::OptionalOutputTensor{std::ref(logprobsValues)}, logprobsIndices, topK,
        topKWorkspace, stream);
}

namespace
{
// One block per output row; each thread copies a stride of vocab elements.
__global__ void gatherSpecVerifyAcceptedLogitRowsKernel(float const* __restrict__ logits,
    int32_t const* __restrict__ acceptedIndices, float* __restrict__ gathered, int32_t verifyTreeSize,
    int32_t maxAcceptDepth, int32_t vocabSize)
{
    int32_t const outRow = blockIdx.x; // 0 .. batchSize*maxAcceptDepth-1
    int32_t const batch = outRow / maxAcceptDepth;
    int32_t const treePos = acceptedIndices[outRow];
    int32_t const safeTreePos = min(max(treePos, 0), verifyTreeSize - 1);
    int32_t const inRow = batch * verifyTreeSize + safeTreePos;

    float const* src = logits + static_cast<ptrdiff_t>(inRow) * vocabSize;
    float* dst = gathered + static_cast<ptrdiff_t>(outRow) * vocabSize;
    for (int32_t v = threadIdx.x; v < vocabSize; v += blockDim.x)
        dst[v] = src[v];
}
} // namespace

void gatherSpecVerifyAcceptedLogitRows(rt::Tensor const& logits, rt::Tensor const& acceptedIndices,
    rt::Tensor& gathered, int32_t batchSize, int32_t verifyTreeSize, int32_t maxAcceptDepth, int32_t vocabSize,
    cudaStream_t stream)
{
    check::check(logits.getDeviceType() == rt::DeviceType::kGPU
            && acceptedIndices.getDeviceType() == rt::DeviceType::kGPU
            && gathered.getDeviceType() == rt::DeviceType::kGPU,
        "All tensors must be on GPU");
    int32_t const totalRows = batchSize * maxAcceptDepth;
    constexpr int32_t kBlock = 256;
    gatherSpecVerifyAcceptedLogitRowsKernel<<<totalRows, kBlock, 0, stream>>>(logits.dataPointer<float>(),
        acceptedIndices.dataPointer<int32_t>(), gathered.dataPointer<float>(), verifyTreeSize, maxAcceptDepth,
        vocabSize);
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
