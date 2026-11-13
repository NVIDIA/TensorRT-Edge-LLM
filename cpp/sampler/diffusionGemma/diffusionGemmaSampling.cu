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

#include "sampler/diffusionGemma/diffusionGemmaSampling.h"

#include "common/checkMacros.h"
#include "common/counterRng.h"

#include <cfloat>
#include <cstdint>
#include <cub/cub.cuh>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace trt_edgellm
{
namespace
{

struct DiffusionTopK
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

struct DiffusionTopKMax
{
    __device__ __forceinline__ DiffusionTopK operator()(DiffusionTopK const& a, DiffusionTopK const& b) const
    {
        return a.value > b.value ? a : b;
    }
};

struct DiffusionEntropyState
{
    float maxValue;
    float sumExp;
    float weightedShift;

    __device__ __forceinline__ void init()
    {
        maxValue = -FLT_MAX;
        sumExp = 0.0F;
        weightedShift = 0.0F;
    }

    __device__ __forceinline__ void insert(float value)
    {
        if (sumExp <= 0.0F)
        {
            maxValue = value;
            sumExp = 1.0F;
            weightedShift = 0.0F;
            return;
        }

        if (value > maxValue)
        {
            float const delta = maxValue - value;
            float const scale = expf(delta);
            weightedShift = weightedShift * scale + sumExp * scale * delta;
            sumExp = sumExp * scale + 1.0F;
            maxValue = value;
        }
        else
        {
            float const shifted = value - maxValue;
            float const expShifted = expf(shifted);
            sumExp += expShifted;
            weightedShift += expShifted * shifted;
        }
    }
};

struct DiffusionEntropyStateReduce
{
    __device__ __forceinline__ DiffusionEntropyState operator()(
        DiffusionEntropyState const& a, DiffusionEntropyState const& b) const
    {
        if (a.sumExp <= 0.0F)
        {
            return b;
        }
        if (b.sumExp <= 0.0F)
        {
            return a;
        }

        DiffusionEntropyState out;
        out.maxValue = fmaxf(a.maxValue, b.maxValue);
        float const aDelta = a.maxValue - out.maxValue;
        float const bDelta = b.maxValue - out.maxValue;
        float const aScale = expf(aDelta);
        float const bScale = expf(bDelta);
        float const scaledASum = a.sumExp * aScale;
        float const scaledBSum = b.sumExp * bScale;
        out.sumExp = scaledASum + scaledBSum;
        out.weightedShift
            = a.weightedShift * aScale + scaledASum * aDelta + b.weightedShift * bScale + scaledBSum * bDelta;
        return out;
    }
};

template <typename T>
__device__ float diffusionLogitToFloat(T value)
{
    return static_cast<float>(value);
}

template <>
__device__ float diffusionLogitToFloat<half>(half value)
{
    return __half2float(value);
}

template <>
__device__ float diffusionLogitToFloat<__nv_bfloat16>(__nv_bfloat16 value)
{
    return __bfloat162float(value);
}

__device__ __forceinline__ float diffusionInvTemp(float temperature)
{
    return (temperature < 1e-3F) ? 1000.0F : 1.0F / temperature;
}

__device__ __forceinline__ float diffusionUniform01(
    uint64_t randomSeed, uint64_t row, uint64_t token, uint64_t randomOffset)
{
    uint64_t const mixed = counter_rng::splitMix64(
        randomSeed ^ (row * 0xD1B54A32D192ED03ULL) ^ (token * 0xABC98388FB8FAC03ULL) ^ randomOffset);
    uint32_t const value = static_cast<uint32_t>((mixed >> 40U) & 0x00FFFFFFU) + 1U;
    // Open interval: clamp only the largest 24-bit sample so Gumbel never sees
    // u == 1.0F, while keeping this hot path free of double-precision math.
    return value == 0x01000000U ? 0x1.fffffep-1F : static_cast<float>(value) * 0x1.0p-24F;
}

__device__ __forceinline__ int32_t diffusionUniformToken(
    uint64_t randomSeed, uint64_t row, uint64_t randomOffset, int32_t vocabSize)
{
    uint64_t const mixed = counter_rng::splitMix64(randomSeed ^ (row * 0xD1B54A32D192ED03ULL) ^ randomOffset);
    return static_cast<int32_t>(mixed % static_cast<uint64_t>(vocabSize));
}

template <typename T, int32_t BLOCK_SIZE_>
__global__ void diffusionGumbelSampleKernel(T const* __restrict__ logits, int32_t* __restrict__ sampledIds,
    int32_t rows, int32_t vocabSize, float temperature, uint64_t randomSeed, uint64_t randomOffset)
{
    using TopKReduce = cub::BlockReduce<DiffusionTopK, BLOCK_SIZE_>;
    __shared__ typename TopKReduce::TempStorage topKStorage;

    int32_t const row = static_cast<int32_t>(blockIdx.x);
    int32_t const tid = static_cast<int32_t>(threadIdx.x);
    if (row >= rows)
    {
        return;
    }

    int64_t const rowOffset = static_cast<int64_t>(row) * vocabSize;
    float const invTemp = diffusionInvTemp(temperature);
    bool const greedy = temperature < 1e-6F;

    DiffusionTopK partial;
    partial.init();
    for (int32_t v = tid; v < vocabSize; v += BLOCK_SIZE_)
    {
        float score = diffusionLogitToFloat(logits[rowOffset + v]) * invTemp;
        if (!greedy)
        {
            float const u
                = diffusionUniform01(randomSeed, static_cast<uint64_t>(row), static_cast<uint64_t>(v), randomOffset);
            score += -logf(-logf(u));
        }
        partial.insert(score, v);
    }

    DiffusionTopK const sampled = TopKReduce(topKStorage).Reduce(partial, DiffusionTopKMax());
    if (tid == 0)
    {
        sampledIds[row] = sampled.index >= 0 ? sampled.index : vocabSize - 1;
    }
}

template <typename T, int32_t BLOCK_SIZE_>
__global__ void diffusionArgmaxKernel(
    T const* __restrict__ logits, int32_t* __restrict__ topIndices, int32_t rows, int32_t vocabSize, float temperature)
{
    using TopKReduce = cub::BlockReduce<DiffusionTopK, BLOCK_SIZE_>;
    __shared__ typename TopKReduce::TempStorage topKStorage;

    int32_t const row = static_cast<int32_t>(blockIdx.x);
    int32_t const tid = static_cast<int32_t>(threadIdx.x);
    if (row >= rows)
    {
        return;
    }

    int64_t const rowOffset = static_cast<int64_t>(row) * vocabSize;
    float const invTemp = diffusionInvTemp(temperature);

    DiffusionTopK partial;
    partial.init();
    for (int32_t v = tid; v < vocabSize; v += BLOCK_SIZE_)
    {
        float const scaledLogit = diffusionLogitToFloat(logits[rowOffset + v]) * invTemp;
        partial.insert(scaledLogit, v);
    }

    DiffusionTopK const best = TopKReduce(topKStorage).Reduce(partial, DiffusionTopKMax());
    if (tid == 0)
    {
        topIndices[row] = best.index >= 0 ? best.index : vocabSize - 1;
    }
}

template <typename T, int32_t BLOCK_SIZE_>
__global__ void diffusionSampleAndEntropyKernel(T const* __restrict__ logits, int32_t* __restrict__ sampledIds,
    int32_t* __restrict__ topIndices, float* __restrict__ entropy, int32_t rows, int32_t vocabSize, float temperature,
    uint64_t randomSeed, uint64_t randomOffset)
{
    using TopKReduce = cub::BlockReduce<DiffusionTopK, BLOCK_SIZE_>;
    using EntropyReduce = cub::BlockReduce<DiffusionEntropyState, BLOCK_SIZE_>;

    union ReduceStorage
    {
        typename TopKReduce::TempStorage topK;
        typename EntropyReduce::TempStorage entropy;
    };
    __shared__ ReduceStorage reduceStorage;

    int32_t const row = static_cast<int32_t>(blockIdx.x);
    int32_t const tid = static_cast<int32_t>(threadIdx.x);
    if (row >= rows)
    {
        return;
    }

    int64_t const rowOffset = static_cast<int64_t>(row) * vocabSize;
    float const invTemp = diffusionInvTemp(temperature);
    bool const greedy = temperature < 1e-6F;

    DiffusionTopK argmaxPartial;
    DiffusionTopK samplePartial;
    DiffusionEntropyState entropyPartial;
    argmaxPartial.init();
    samplePartial.init();
    entropyPartial.init();
    for (int32_t v = tid; v < vocabSize; v += BLOCK_SIZE_)
    {
        float const scaledLogit = diffusionLogitToFloat(logits[rowOffset + v]) * invTemp;
        argmaxPartial.insert(scaledLogit, v);
        entropyPartial.insert(scaledLogit);

        float sampleScore = scaledLogit;
        if (!greedy)
        {
            float const u
                = diffusionUniform01(randomSeed, static_cast<uint64_t>(row), static_cast<uint64_t>(v), randomOffset);
            sampleScore += -logf(-logf(u));
        }
        samplePartial.insert(sampleScore, v);
    }

    DiffusionTopK const best = TopKReduce(reduceStorage.topK).Reduce(argmaxPartial, DiffusionTopKMax());
    __syncthreads();
    DiffusionTopK const sampled = TopKReduce(reduceStorage.topK).Reduce(samplePartial, DiffusionTopKMax());
    __syncthreads();
    DiffusionEntropyState const entropyState
        = EntropyReduce(reduceStorage.entropy).Reduce(entropyPartial, DiffusionEntropyStateReduce());
    if (tid == 0)
    {
        topIndices[row] = best.index >= 0 ? best.index : vocabSize - 1;
        sampledIds[row] = sampled.index >= 0 ? sampled.index : vocabSize - 1;
        entropy[row] = entropyState.sumExp > 0.0F
            ? logf(entropyState.sumExp) - entropyState.weightedShift / entropyState.sumExp
            : 0.0F;
    }
}

constexpr int32_t kDiffusionCanvasSamplerBlockSize{256};

__device__ __forceinline__ bool diffusionEntropyCandidateBetter(
    float candidateEntropy, int32_t candidatePos, float bestEntropy, int32_t bestPos)
{
    if (candidatePos < 0)
    {
        return false;
    }
    if (bestPos < 0)
    {
        return true;
    }
    return candidateEntropy < bestEntropy || (candidateEntropy == bestEntropy && candidatePos < bestPos);
}

__global__ void initializeDiffusionCanvasKernel(int32_t* canvasIds, int32_t* previousArgmaxIds, int32_t* stableCounts,
    int8_t* acceptedMask, int32_t* prefixLengths, int32_t rows, int32_t batchSize, int32_t vocabSize,
    uint64_t randomSeed, uint64_t randomOffset)
{
    int32_t const idx = static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx < rows)
    {
        canvasIds[idx] = diffusionUniformToken(randomSeed, static_cast<uint64_t>(idx), randomOffset, vocabSize);
        previousArgmaxIds[idx] = -1;
        stableCounts[idx] = 0;
        acceptedMask[idx] = 0;
    }
    if (idx < batchSize)
    {
        prefixLengths[idx] = 0;
    }
}

template <int32_t BLOCK_SIZE_>
__global__ void diffusionSampleAndUpdateCanvasKernel(int32_t const* sampledIds, int32_t const* argmaxIds,
    float const* entropy, int32_t* canvasIds, int32_t* argmaxCanvasIds, int32_t* previousArgmaxIds,
    int32_t* stableCounts, int8_t* acceptedMask, int32_t* prefixLengths, int32_t const* validCanvasLengths,
    int32_t canvasLen, float entropyThreshold, float entropyBound, int32_t stabilityWindow, bool forceAccept,
    int32_t vocabSize, uint64_t randomSeed, uint64_t randomOffset)
{
    using FloatScan = cub::BlockScan<float, BLOCK_SIZE_>;

    extern __shared__ int8_t dynamicBudgetAccepted[];
    __shared__ int8_t staticBudgetAccepted[BLOCK_SIZE_];
    __shared__ float sortedEntropy[BLOCK_SIZE_];
    __shared__ float sortedBudgetEntropy[BLOCK_SIZE_];
    __shared__ int32_t sortedPosition[BLOCK_SIZE_];
    __shared__ float reductionEntropy[BLOCK_SIZE_];
    __shared__ int32_t reductionPosition[BLOCK_SIZE_];
    __shared__ int32_t reductionStable[BLOCK_SIZE_];
    __shared__ typename FloatScan::TempStorage scanStorage;
    __shared__ int32_t clampedValidLen;
    __shared__ int32_t skipRow;
    __shared__ int32_t keepSelecting;
    __shared__ float cumulativeEntropy;
    __shared__ float maxSelectedEntropy;
    __shared__ int32_t converged;

    int32_t const batch = static_cast<int32_t>(blockIdx.x);
    int32_t const tid = static_cast<int32_t>(threadIdx.x);
    int32_t const base = batch * canvasLen;

    if (tid == 0)
    {
        int32_t const validLen = validCanvasLengths != nullptr ? validCanvasLengths[batch] : canvasLen;
        clampedValidLen = validLen < 1 ? 1 : (validLen > canvasLen ? canvasLen : validLen);
        skipRow = prefixLengths[batch] >= clampedValidLen ? 1 : 0;
    }
    __syncthreads();

    if (skipRow != 0)
    {
        return;
    }

    if (forceAccept)
    {
        for (int32_t i = tid; i < clampedValidLen; i += BLOCK_SIZE_)
        {
            int32_t const row = base + i;
            int32_t const argmaxToken = argmaxIds[row];
            previousArgmaxIds[row] = argmaxToken;
            stableCounts[row] = stabilityWindow + 1;
            argmaxCanvasIds[row] = argmaxToken;
            canvasIds[row] = argmaxToken;
            acceptedMask[row] = 1;
        }
        for (int32_t i = clampedValidLen + tid; i < canvasLen; i += BLOCK_SIZE_)
        {
            acceptedMask[base + i] = 0;
        }
        if (tid == 0)
        {
            prefixLengths[batch] = clampedValidLen;
        }
        return;
    }

    int8_t* budgetAccepted = clampedValidLen <= BLOCK_SIZE_ ? staticBudgetAccepted : dynamicBudgetAccepted;
    for (int32_t i = tid; i < clampedValidLen; i += BLOCK_SIZE_)
    {
        budgetAccepted[i] = 0;
    }
    __syncthreads();

    if (entropyBound >= 0.0F && clampedValidLen <= BLOCK_SIZE_)
    {
        if (tid < BLOCK_SIZE_)
        {
            float entropyKey = FLT_MAX;
            float budgetEntropy = 0.0F;
            int32_t position = INT32_MAX;
            if (tid < clampedValidLen)
            {
                float const rowEntropy = entropy[base + tid];
                if (isfinite(rowEntropy))
                {
                    entropyKey = rowEntropy;
                    budgetEntropy = fmaxf(0.0F, rowEntropy);
                    position = tid;
                }
            }
            sortedEntropy[tid] = entropyKey;
            sortedBudgetEntropy[tid] = budgetEntropy;
            sortedPosition[tid] = position;
        }
        __syncthreads();

        for (int32_t sortSize = 2; sortSize <= BLOCK_SIZE_; sortSize <<= 1)
        {
            for (int32_t stride = sortSize >> 1; stride > 0; stride >>= 1)
            {
                int32_t const other = tid ^ stride;
                if (other > tid)
                {
                    bool const ascending = (tid & sortSize) == 0;
                    bool const otherFirst = diffusionEntropyCandidateBetter(
                        sortedEntropy[other], sortedPosition[other], sortedEntropy[tid], sortedPosition[tid]);
                    if ((ascending && otherFirst) || (!ascending && !otherFirst))
                    {
                        float const entropyTmp = sortedEntropy[tid];
                        float const budgetEntropyTmp = sortedBudgetEntropy[tid];
                        int32_t const positionTmp = sortedPosition[tid];
                        sortedEntropy[tid] = sortedEntropy[other];
                        sortedBudgetEntropy[tid] = sortedBudgetEntropy[other];
                        sortedPosition[tid] = sortedPosition[other];
                        sortedEntropy[other] = entropyTmp;
                        sortedBudgetEntropy[other] = budgetEntropyTmp;
                        sortedPosition[other] = positionTmp;
                    }
                }
                __syncthreads();
            }
        }

        float scanInput = 0.0F;
        if (tid < clampedValidLen && sortedPosition[tid] != INT32_MAX)
        {
            scanInput = sortedBudgetEntropy[tid];
        }
        float inclusiveEntropy = 0.0F;
        FloatScan(scanStorage).InclusiveSum(scanInput, inclusiveEntropy);
        __syncthreads();

        if (tid < clampedValidLen && sortedPosition[tid] != INT32_MAX)
        {
            bool const exceedsBudget = inclusiveEntropy - sortedBudgetEntropy[tid] > entropyBound;
            if (!exceedsBudget)
            {
                budgetAccepted[sortedPosition[tid]] = 1;
            }
        }
        __syncthreads();
    }
    else if (entropyBound >= 0.0F)
    {
        if (tid == 0)
        {
            cumulativeEntropy = 0.0F;
            maxSelectedEntropy = 0.0F;
            keepSelecting = 1;
        }
        __syncthreads();

        for (int32_t selectedCount = 0; selectedCount < clampedValidLen; ++selectedCount)
        {
            float bestEntropy = FLT_MAX;
            int32_t bestPos = -1;
            for (int32_t i = tid; i < clampedValidLen; i += BLOCK_SIZE_)
            {
                if (budgetAccepted[i] != 0)
                {
                    continue;
                }
                float const candidateEntropy = entropy[base + i];
                if (isfinite(candidateEntropy)
                    && diffusionEntropyCandidateBetter(candidateEntropy, i, bestEntropy, bestPos))
                {
                    bestEntropy = candidateEntropy;
                    bestPos = i;
                }
            }

            reductionEntropy[tid] = bestEntropy;
            reductionPosition[tid] = bestPos;
            __syncthreads();

            for (int32_t stride = BLOCK_SIZE_ / 2; stride > 0; stride >>= 1)
            {
                if (tid < stride)
                {
                    float const candidateEntropy = reductionEntropy[tid + stride];
                    int32_t const candidatePos = reductionPosition[tid + stride];
                    if (diffusionEntropyCandidateBetter(
                            candidateEntropy, candidatePos, reductionEntropy[tid], reductionPosition[tid]))
                    {
                        reductionEntropy[tid] = candidateEntropy;
                        reductionPosition[tid] = candidatePos;
                    }
                }
                __syncthreads();
            }

            if (tid == 0)
            {
                int32_t const bestPosition = reductionPosition[0];
                if (bestPosition < 0)
                {
                    keepSelecting = 0;
                }
                else
                {
                    float const clampedEntropy = fmaxf(0.0F, reductionEntropy[0]);
                    cumulativeEntropy += clampedEntropy;
                    maxSelectedEntropy = fmaxf(maxSelectedEntropy, clampedEntropy);
                    if (cumulativeEntropy - maxSelectedEntropy > entropyBound)
                    {
                        keepSelecting = 0;
                    }
                    else
                    {
                        budgetAccepted[bestPosition] = 1;
                        keepSelecting = 1;
                    }
                }
            }
            __syncthreads();

            if (keepSelecting == 0)
            {
                break;
            }
        }
    }

    float entropySum = 0.0F;
    int32_t allStable = 1;
    for (int32_t i = tid; i < clampedValidLen; i += BLOCK_SIZE_)
    {
        int32_t const row = base + i;
        int32_t const argmaxToken = argmaxIds[row];
        int32_t const sampledToken = sampledIds[row];
        bool const sameAsPrevious = argmaxToken == previousArgmaxIds[row];
        int32_t const nextStable = sameAsPrevious ? (stableCounts[row] + 1) : 1;
        bool const tokenStable = (stabilityWindow <= 0) || (nextStable > stabilityWindow);
        previousArgmaxIds[row] = argmaxToken;
        stableCounts[row] = nextStable;
        argmaxCanvasIds[row] = argmaxToken;

        float const rowEntropy = entropy[row];
        entropySum += isfinite(rowEntropy) ? fmaxf(0.0F, rowEntropy) : FLT_MAX / 1024.0F;
        allStable = allStable && tokenStable;

        bool const accepted = budgetAccepted[i] != 0;
        acceptedMask[row] = accepted ? 1 : 0;
        if (accepted)
        {
            canvasIds[row] = sampledToken;
        }
        else
        {
            canvasIds[row] = diffusionUniformToken(randomSeed, static_cast<uint64_t>(row), randomOffset, vocabSize);
        }
    }

    for (int32_t i = clampedValidLen + tid; i < canvasLen; i += BLOCK_SIZE_)
    {
        acceptedMask[base + i] = 0;
    }

    reductionEntropy[tid] = entropySum;
    reductionStable[tid] = allStable;
    __syncthreads();

    for (int32_t stride = BLOCK_SIZE_ / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            reductionEntropy[tid] += reductionEntropy[tid + stride];
            reductionStable[tid] = reductionStable[tid] && reductionStable[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        float const meanEntropy = reductionEntropy[0] / static_cast<float>(clampedValidLen);
        converged = forceAccept || (reductionStable[0] != 0 && meanEntropy < entropyThreshold);
        prefixLengths[batch] = converged != 0 ? clampedValidLen : 0;
    }
    __syncthreads();

    if (converged != 0)
    {
        for (int32_t i = tid; i < clampedValidLen; i += BLOCK_SIZE_)
        {
            canvasIds[base + i] = argmaxCanvasIds[base + i];
        }
    }
}

__global__ void compactDiffusionCanvasKernel(
    int32_t const* canvasIds, int32_t* commitCanvasIds, int32_t canvasLen, int32_t blockLen, int32_t totalElements)
{
    int32_t const idx = static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= totalElements)
    {
        return;
    }

    int32_t const batch = idx / blockLen;
    int32_t const pos = idx % blockLen;
    commitCanvasIds[idx] = canvasIds[batch * canvasLen + pos];
}

__global__ void compactDiffusionCanvasVarLenKernel(int32_t const* canvasIds, int32_t const* commitLengths,
    int32_t* commitCanvasIds, int32_t canvasLen, int32_t maxBlockLen, int32_t padTokenId, int32_t totalElements)
{
    int32_t const idx = static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= totalElements)
    {
        return;
    }

    int32_t const batch = idx / maxBlockLen;
    int32_t const pos = idx % maxBlockLen;
    int32_t const length = commitLengths[batch];
    commitCanvasIds[idx] = pos < length ? canvasIds[batch * canvasLen + pos] : padTokenId;
}

} // namespace

void sampleDiffusionTokensFromLogits(rt::Tensor const& input, rt::Tensor& sampledIds, float temperature,
    cudaStream_t stream, DiffusionRandomParams const& random)
{
    check::check(input.getDeviceType() == rt::DeviceType::kGPU && sampledIds.getDeviceType() == rt::DeviceType::kGPU,
        "All tensors must be on GPU");
    check::check(
        (input.getDataType() == trt_edgellm::DataType::kFLOAT || input.getDataType() == trt_edgellm::DataType::kHALF
            || input.getDataType() == trt_edgellm::DataType::kBF16)
            && sampledIds.getDataType() == trt_edgellm::DataType::kINT32,
        "Invalid tensor data types");

    auto const inputShape = input.getShape();
    auto const sampledShape = sampledIds.getShape();
    check::check(inputShape.getNumDims() == 2, "Invalid logits tensor dimensions");
    int32_t const rows = inputShape[0];
    int32_t const vocabSize = inputShape[1];
    check::check(sampledShape.volume() == rows, "Sampled token tensor shape mismatch");
    if (rows <= 0 || vocabSize <= 0)
    {
        return;
    }

    constexpr int32_t kBlockSize = 256;
    dim3 const grid(rows);
    dim3 const block(kBlockSize);
    if (input.getDataType() == trt_edgellm::DataType::kHALF)
    {
        diffusionGumbelSampleKernel<half, kBlockSize><<<grid, block, 0, stream>>>(input.dataPointer<half>(),
            sampledIds.dataPointer<int32_t>(), rows, vocabSize, temperature, random.seed, random.offset);
    }
    else if (input.getDataType() == trt_edgellm::DataType::kBF16)
    {
        diffusionGumbelSampleKernel<__nv_bfloat16, kBlockSize>
            <<<grid, block, 0, stream>>>(input.dataPointer<__nv_bfloat16>(), sampledIds.dataPointer<int32_t>(), rows,
                vocabSize, temperature, random.seed, random.offset);
    }
    else
    {
        diffusionGumbelSampleKernel<float, kBlockSize><<<grid, block, 0, stream>>>(input.dataPointer<float>(),
            sampledIds.dataPointer<int32_t>(), rows, vocabSize, temperature, random.seed, random.offset);
    }
    CUDA_CHECK(cudaGetLastError());
}

void sampleDiffusionTokensAndComputeEntropy(rt::Tensor const& input, rt::Tensor& sampledIds, rt::Tensor& topIndices,
    rt::Tensor& entropy, float temperature, cudaStream_t stream, DiffusionRandomParams const& random)
{
    check::check(input.getDeviceType() == rt::DeviceType::kGPU && sampledIds.getDeviceType() == rt::DeviceType::kGPU
            && topIndices.getDeviceType() == rt::DeviceType::kGPU && entropy.getDeviceType() == rt::DeviceType::kGPU,
        "All tensors must be on GPU");
    check::check(
        (input.getDataType() == trt_edgellm::DataType::kFLOAT || input.getDataType() == trt_edgellm::DataType::kHALF
            || input.getDataType() == trt_edgellm::DataType::kBF16)
            && sampledIds.getDataType() == trt_edgellm::DataType::kINT32
            && topIndices.getDataType() == trt_edgellm::DataType::kINT32
            && entropy.getDataType() == trt_edgellm::DataType::kFLOAT,
        "Invalid tensor data types");

    auto const inputShape = input.getShape();
    auto const sampledShape = sampledIds.getShape();
    auto const indexShape = topIndices.getShape();
    auto const entropyShape = entropy.getShape();
    check::check(inputShape.getNumDims() == 2 && indexShape.getNumDims() == 2 && entropyShape.getNumDims() == 1,
        "Invalid tensor dimensions");

    int32_t const rows = inputShape[0];
    int32_t const vocabSize = inputShape[1];
    check::check(sampledShape.volume() == rows, "Sampled token tensor shape mismatch");
    check::check(indexShape[0] == rows && indexShape[1] == 1, "Top index tensor shape mismatch");
    check::check(entropyShape[0] == rows, "Entropy tensor shape mismatch");
    if (rows <= 0 || vocabSize <= 0)
    {
        return;
    }

    constexpr int32_t kBlockSize = 256;
    dim3 const grid(rows);
    dim3 const block(kBlockSize);
    if (input.getDataType() == trt_edgellm::DataType::kHALF)
    {
        diffusionSampleAndEntropyKernel<half, kBlockSize><<<grid, block, 0, stream>>>(input.dataPointer<half>(),
            sampledIds.dataPointer<int32_t>(), topIndices.dataPointer<int32_t>(), entropy.dataPointer<float>(), rows,
            vocabSize, temperature, random.seed, random.offset);
    }
    else if (input.getDataType() == trt_edgellm::DataType::kBF16)
    {
        diffusionSampleAndEntropyKernel<__nv_bfloat16, kBlockSize><<<grid, block, 0, stream>>>(
            input.dataPointer<__nv_bfloat16>(), sampledIds.dataPointer<int32_t>(), topIndices.dataPointer<int32_t>(),
            entropy.dataPointer<float>(), rows, vocabSize, temperature, random.seed, random.offset);
    }
    else
    {
        diffusionSampleAndEntropyKernel<float, kBlockSize><<<grid, block, 0, stream>>>(input.dataPointer<float>(),
            sampledIds.dataPointer<int32_t>(), topIndices.dataPointer<int32_t>(), entropy.dataPointer<float>(), rows,
            vocabSize, temperature, random.seed, random.offset);
    }
    CUDA_CHECK(cudaGetLastError());
}

void selectDiffusionArgmaxFromLogits(
    rt::Tensor const& input, rt::Tensor& topIndices, float temperature, cudaStream_t stream)
{
    check::check(input.getDeviceType() == rt::DeviceType::kGPU && topIndices.getDeviceType() == rt::DeviceType::kGPU,
        "All tensors must be on GPU");
    check::check(
        (input.getDataType() == trt_edgellm::DataType::kFLOAT || input.getDataType() == trt_edgellm::DataType::kHALF
            || input.getDataType() == trt_edgellm::DataType::kBF16)
            && topIndices.getDataType() == trt_edgellm::DataType::kINT32,
        "Invalid tensor data types");

    auto const inputShape = input.getShape();
    auto const indexShape = topIndices.getShape();
    check::check(inputShape.getNumDims() == 2 && indexShape.getNumDims() == 2, "Invalid tensor dimensions");

    int32_t const rows = inputShape[0];
    int32_t const vocabSize = inputShape[1];
    check::check(indexShape[0] == rows && indexShape[1] == 1, "Top index tensor shape mismatch");
    if (rows <= 0 || vocabSize <= 0)
    {
        return;
    }

    constexpr int32_t kBlockSize = 256;
    dim3 const grid(rows);
    dim3 const block(kBlockSize);
    if (input.getDataType() == trt_edgellm::DataType::kHALF)
    {
        diffusionArgmaxKernel<half, kBlockSize><<<grid, block, 0, stream>>>(
            input.dataPointer<half>(), topIndices.dataPointer<int32_t>(), rows, vocabSize, temperature);
    }
    else if (input.getDataType() == trt_edgellm::DataType::kBF16)
    {
        diffusionArgmaxKernel<__nv_bfloat16, kBlockSize><<<grid, block, 0, stream>>>(
            input.dataPointer<__nv_bfloat16>(), topIndices.dataPointer<int32_t>(), rows, vocabSize, temperature);
    }
    else
    {
        diffusionArgmaxKernel<float, kBlockSize><<<grid, block, 0, stream>>>(
            input.dataPointer<float>(), topIndices.dataPointer<int32_t>(), rows, vocabSize, temperature);
    }
    CUDA_CHECK(cudaGetLastError());
}

void initializeDiffusionCanvas(rt::Tensor& canvasIds, rt::Tensor& previousArgmaxIds, rt::Tensor& stableCounts,
    rt::Tensor& acceptedMask, rt::Tensor& prefixLengths, DiffusionCanvasInitParams const& params, cudaStream_t stream)
{
    check::check(canvasIds.getDeviceType() == rt::DeviceType::kGPU
            && previousArgmaxIds.getDeviceType() == rt::DeviceType::kGPU
            && stableCounts.getDeviceType() == rt::DeviceType::kGPU
            && acceptedMask.getDeviceType() == rt::DeviceType::kGPU
            && prefixLengths.getDeviceType() == rt::DeviceType::kGPU,
        "All diffusion canvas sampler tensors must be on GPU");
    check::check(canvasIds.getDataType() == trt_edgellm::DataType::kINT32
            && previousArgmaxIds.getDataType() == trt_edgellm::DataType::kINT32
            && stableCounts.getDataType() == trt_edgellm::DataType::kINT32
            && acceptedMask.getDataType() == trt_edgellm::DataType::kINT8
            && prefixLengths.getDataType() == trt_edgellm::DataType::kINT32,
        "Invalid diffusion canvas sampler tensor data types");

    auto const canvasShape = canvasIds.getShape();
    auto const previousShape = previousArgmaxIds.getShape();
    auto const stableShape = stableCounts.getShape();
    auto const acceptedShape = acceptedMask.getShape();
    auto const prefixShape = prefixLengths.getShape();
    check::check(canvasShape.getNumDims() == 2 && previousShape == canvasShape && stableShape == canvasShape
            && acceptedShape == canvasShape && prefixShape.getNumDims() == 1,
        "Invalid diffusion canvas sampler tensor shapes");

    int32_t const batchSize = canvasShape[0];
    int32_t const canvasLen = canvasShape[1];
    check::check(prefixShape[0] == batchSize, "Diffusion prefix length tensor shape mismatch");
    if (batchSize <= 0 || canvasLen <= 0)
    {
        return;
    }
    check::check(params.vocabSize > 0, "Diffusion canvas sampler vocab size must be positive");

    int32_t const rows = batchSize * canvasLen;
    constexpr int32_t kBlockSize = 256;
    int32_t const totalElements = rows > batchSize ? rows : batchSize;
    int32_t const blocks = (totalElements + kBlockSize - 1) / kBlockSize;
    initializeDiffusionCanvasKernel<<<blocks, kBlockSize, 0, stream>>>(canvasIds.dataPointer<int32_t>(),
        previousArgmaxIds.dataPointer<int32_t>(), stableCounts.dataPointer<int32_t>(),
        acceptedMask.dataPointer<int8_t>(), prefixLengths.dataPointer<int32_t>(), rows, batchSize, params.vocabSize,
        params.random.seed, params.random.offset);
    CUDA_CHECK(cudaGetLastError());
}

void diffusionSampleAndUpdateCanvas(rt::Tensor const& sampledIds, rt::Tensor const& argmaxIds,
    rt::Tensor const& entropy, rt::Tensor& canvasIds, rt::Tensor& argmaxCanvasIds, rt::Tensor& previousArgmaxIds,
    rt::Tensor& stableCounts, rt::Tensor& acceptedMask, rt::Tensor& prefixLengths,
    DiffusionCanvasUpdateParams const& params, cudaStream_t stream, rt::OptionalInputTensor validCanvasLengths)
{
    check::check(sampledIds.getDeviceType() == rt::DeviceType::kGPU && argmaxIds.getDeviceType() == rt::DeviceType::kGPU
            && entropy.getDeviceType() == rt::DeviceType::kGPU && canvasIds.getDeviceType() == rt::DeviceType::kGPU
            && argmaxCanvasIds.getDeviceType() == rt::DeviceType::kGPU
            && previousArgmaxIds.getDeviceType() == rt::DeviceType::kGPU
            && stableCounts.getDeviceType() == rt::DeviceType::kGPU
            && acceptedMask.getDeviceType() == rt::DeviceType::kGPU
            && prefixLengths.getDeviceType() == rt::DeviceType::kGPU
            && (!validCanvasLengths.has_value()
                || validCanvasLengths.value().get().getDeviceType() == rt::DeviceType::kGPU),
        "All diffusion sampler tensors must be on GPU");
    check::check(sampledIds.getDataType() == trt_edgellm::DataType::kINT32
            && argmaxIds.getDataType() == trt_edgellm::DataType::kINT32
            && entropy.getDataType() == trt_edgellm::DataType::kFLOAT
            && canvasIds.getDataType() == trt_edgellm::DataType::kINT32
            && argmaxCanvasIds.getDataType() == trt_edgellm::DataType::kINT32
            && previousArgmaxIds.getDataType() == trt_edgellm::DataType::kINT32
            && stableCounts.getDataType() == trt_edgellm::DataType::kINT32
            && acceptedMask.getDataType() == trt_edgellm::DataType::kINT8
            && prefixLengths.getDataType() == trt_edgellm::DataType::kINT32
            && (!validCanvasLengths.has_value()
                || validCanvasLengths.value().get().getDataType() == trt_edgellm::DataType::kINT32),
        "Invalid diffusion sampler tensor data types");

    auto const sampledShape = sampledIds.getShape();
    auto const argmaxShape = argmaxIds.getShape();
    auto const entropyShape = entropy.getShape();
    auto const canvasShape = canvasIds.getShape();
    auto const argmaxCanvasShape = argmaxCanvasIds.getShape();
    auto const previousShape = previousArgmaxIds.getShape();
    auto const stableShape = stableCounts.getShape();
    auto const acceptedShape = acceptedMask.getShape();
    auto const prefixShape = prefixLengths.getShape();
    rt::Coords const validShape
        = validCanvasLengths.has_value() ? validCanvasLengths.value().get().getShape() : rt::Coords{};
    check::check(entropyShape.getNumDims() == 1 && canvasShape.getNumDims() == 2 && argmaxCanvasShape == canvasShape
            && previousShape == canvasShape && stableShape == canvasShape && acceptedShape == canvasShape
            && prefixShape.getNumDims() == 1 && (!validCanvasLengths.has_value() || validShape.getNumDims() == 1),
        "Invalid diffusion sampler tensor shapes");

    int32_t const batchSize = canvasShape[0];
    int32_t const canvasLen = canvasShape[1];
    int32_t const rows = batchSize * canvasLen;
    check::check(sampledShape.volume() == rows && argmaxShape.volume() == rows && entropyShape[0] == rows
            && prefixShape[0] == batchSize && (!validCanvasLengths.has_value() || validShape[0] == batchSize),
        "Diffusion sampler tensor shape mismatch");
    if (batchSize <= 0 || canvasLen <= 0)
    {
        return;
    }
    check::check(params.stabilityWindow > 0, "Diffusion sampler stability window must be positive");
    check::check(params.vocabSize > 0, "Diffusion sampler vocab size must be positive");

    int32_t const* validCanvasLengthsPtr
        = validCanvasLengths.has_value() ? validCanvasLengths.value().get().dataPointer<int32_t>() : nullptr;
    dim3 const grid(batchSize);
    dim3 const block(kDiffusionCanvasSamplerBlockSize);
    size_t const sharedMemSize = static_cast<size_t>(canvasLen) * sizeof(int8_t);
    diffusionSampleAndUpdateCanvasKernel<kDiffusionCanvasSamplerBlockSize>
        <<<grid, block, sharedMemSize, stream>>>(sampledIds.dataPointer<int32_t>(), argmaxIds.dataPointer<int32_t>(),
            entropy.dataPointer<float>(), canvasIds.dataPointer<int32_t>(), argmaxCanvasIds.dataPointer<int32_t>(),
            previousArgmaxIds.dataPointer<int32_t>(), stableCounts.dataPointer<int32_t>(),
            acceptedMask.dataPointer<int8_t>(), prefixLengths.dataPointer<int32_t>(), validCanvasLengthsPtr, canvasLen,
            params.entropyThreshold, params.entropyBound, params.stabilityWindow, params.forceAccept, params.vocabSize,
            params.random.seed, params.random.offset);
    CUDA_CHECK(cudaGetLastError());
}

void compactDiffusionCanvas(rt::Tensor const& canvasIds, rt::Tensor& commitCanvasIds, int32_t batchSize,
    int32_t canvasLen, int32_t blockLen, cudaStream_t stream)
{
    check::check(
        canvasIds.getDeviceType() == rt::DeviceType::kGPU && commitCanvasIds.getDeviceType() == rt::DeviceType::kGPU,
        "Diffusion commit canvas tensors must be on GPU");
    check::check(canvasIds.getDataType() == trt_edgellm::DataType::kINT32
            && commitCanvasIds.getDataType() == trt_edgellm::DataType::kINT32,
        "Diffusion commit canvas tensors must have INT32 data type");
    auto const canvasShape = canvasIds.getShape();
    auto const commitShape = commitCanvasIds.getShape();
    check::check(canvasShape.getNumDims() == 2 && commitShape.getNumDims() == 2,
        "Invalid diffusion commit canvas tensor shapes");
    check::check(canvasShape[0] == batchSize && canvasShape[1] == canvasLen && commitShape[0] == batchSize
            && commitShape[1] == blockLen,
        "Diffusion commit canvas tensor shape mismatch");
    check::check(
        batchSize > 0 && blockLen > 0 && blockLen <= canvasLen, "Diffusion commit canvas dimensions are out of range");

    int32_t const totalElements = batchSize * blockLen;
    constexpr int32_t kBlockSize = 256;
    int32_t const blocks = (totalElements + kBlockSize - 1) / kBlockSize;
    compactDiffusionCanvasKernel<<<blocks, kBlockSize, 0, stream>>>(
        canvasIds.dataPointer<int32_t>(), commitCanvasIds.dataPointer<int32_t>(), canvasLen, blockLen, totalElements);
    CUDA_CHECK(cudaGetLastError());
}

void compactDiffusionCanvas(rt::Tensor const& canvasIds, rt::Tensor const& commitLengths, rt::Tensor& commitCanvasIds,
    int32_t batchSize, int32_t canvasLen, int32_t maxBlockLen, int32_t padTokenId, cudaStream_t stream)
{
    check::check(canvasIds.getDeviceType() == rt::DeviceType::kGPU
            && commitLengths.getDeviceType() == rt::DeviceType::kGPU
            && commitCanvasIds.getDeviceType() == rt::DeviceType::kGPU,
        "Diffusion variable commit canvas tensors must be on GPU");
    check::check(canvasIds.getDataType() == trt_edgellm::DataType::kINT32
            && commitLengths.getDataType() == trt_edgellm::DataType::kINT32
            && commitCanvasIds.getDataType() == trt_edgellm::DataType::kINT32,
        "Diffusion variable commit canvas tensors must have INT32 data type");

    auto const canvasShape = canvasIds.getShape();
    auto const lengthsShape = commitLengths.getShape();
    auto const commitShape = commitCanvasIds.getShape();
    check::check(canvasShape.getNumDims() == 2 && lengthsShape.getNumDims() == 1 && commitShape.getNumDims() == 2,
        "Invalid diffusion variable commit canvas tensor shapes");
    check::check(canvasShape[0] == batchSize && canvasShape[1] == canvasLen && lengthsShape[0] == batchSize
            && commitShape[0] == batchSize && commitShape[1] == maxBlockLen,
        "Diffusion variable commit canvas tensor shape mismatch");
    check::check(batchSize > 0 && maxBlockLen > 0 && maxBlockLen <= canvasLen,
        "Diffusion variable commit canvas dimensions are out of range");
    check::check(padTokenId >= 0, "Diffusion variable commit pad token must be non-negative");

    int32_t const totalElements = batchSize * maxBlockLen;
    constexpr int32_t kBlockSize = 256;
    int32_t const blocks = (totalElements + kBlockSize - 1) / kBlockSize;
    compactDiffusionCanvasVarLenKernel<<<blocks, kBlockSize, 0, stream>>>(canvasIds.dataPointer<int32_t>(),
        commitLengths.dataPointer<int32_t>(), commitCanvasIds.dataPointer<int32_t>(), canvasLen, maxBlockLen,
        padTokenId, totalElements);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace trt_edgellm
