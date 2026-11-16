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

#include "kernels/speculative/dflash2CandidateSelector.h"

#include "common/checkMacros.h"
#include "common/cudaUtils.h"
#include "kernels/speculative/speculativeKernelsUtils.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cfloat>
#include <cmath>
#include <limits>

namespace trt_edgellm
{
namespace kernel
{
namespace
{

constexpr int32_t kWarpSize{32};
constexpr int32_t kTreeCandidateThreads{256};

//! One element of P[p] ⊙ z, the predecessor-conditioned selector query.
__device__ __forceinline__ float selectorQueryElement(half predecessor, half projected)
{
    return static_cast<float>(predecessor) * static_cast<float>(projected);
}

//! Selector bilinear term <query, Q[c]> reduced across one warp; the sum is valid in lane 0. Chain and tree
//! scoring share this lane split and fmaf order, so their scores are bitwise equal.
__device__ __forceinline__ float selectorWarpDot(
    float const* query, half const* successorRow, int32_t rank, int32_t lane)
{
    float dot = 0.0F;
    for (int32_t r = lane; r < rank; r += kWarpSize)
    {
        dot = fmaf(query[r], static_cast<float>(successorRow[r]), dot);
    }
    for (int32_t offset = kWarpSize / 2; offset > 0; offset >>= 1)
    {
        dot += __shfl_down_sync(0xFFFFFFFFU, dot, offset);
    }
    return dot;
}

__device__ __forceinline__ bool isVocabToken(int32_t token, int32_t vocabSize)
{
    return token >= 0 && token < vocabSize;
}

__global__ void candidateSelectorKernel(int32_t const* __restrict__ candidateIds, float const* __restrict__ unaryLogits,
    half const* __restrict__ projectedHidden, int32_t const* __restrict__ anchorTokenIds,
    half const* __restrict__ predecessorCodebook, half const* __restrict__ successorCodebook,
    float const* __restrict__ uniforms, float const* __restrict__ temperatures, int32_t const* __restrict__ greedyMask,
    int32_t* __restrict__ proposalTokenIds, float* __restrict__ proposalProbabilities, int32_t steps, int32_t topK,
    int32_t rank, int32_t vocabSize)
{
    int32_t const batch = static_cast<int32_t>(blockIdx.x);
    int32_t const warp = static_cast<int32_t>(threadIdx.x / 32);
    int32_t const lane = static_cast<int32_t>(threadIdx.x % 32);
    __shared__ float scores[kDFlash2MaxSelectorTopK];
    __shared__ float probabilities[kDFlash2MaxSelectorTopK];
    __shared__ float predecessorHidden[kDFlash2MaxSelectorRank];
    __shared__ int32_t previousToken;

    if (threadIdx.x == 0)
    {
        previousToken = anchorTokenIds[batch];
    }
    __syncthreads();

    for (int32_t step = 0; step < steps; ++step)
    {
        int32_t const row = batch * steps + step;
        for (int32_t r = static_cast<int32_t>(threadIdx.x); r < rank; r += static_cast<int32_t>(blockDim.x))
        {
            predecessorHidden[r] = isVocabToken(previousToken, vocabSize)
                ? selectorQueryElement(predecessorCodebook[previousToken * rank + r], projectedHidden[row * rank + r])
                : 0.0F;
        }
        __syncthreads();

        if (warp < topK)
        {
            int32_t const candidate = candidateIds[row * topK + warp];
            float const dot = isVocabToken(candidate, vocabSize)
                ? selectorWarpDot(predecessorHidden, successorCodebook + candidate * rank, rank, lane)
                : 0.0F;
            if (lane == 0)
            {
                scores[warp] = unaryLogits[row * topK + warp] + dot;
            }
        }
        __syncthreads();

        if (threadIdx.x == 0)
        {
            int32_t selected{0};
            if (greedyMask[batch] != 0 || temperatures[batch] < 1.0e-3F)
            {
                float best = -FLT_MAX;
                int32_t bestToken = INT_MAX;
                for (int32_t k = 0; k < topK; ++k)
                {
                    int32_t const token = candidateIds[row * topK + k];
                    if (scores[k] > best || (scores[k] == best && token < bestToken))
                    {
                        best = scores[k];
                        bestToken = token;
                        selected = k;
                    }
                }
                for (int32_t k = 0; k < topK; ++k)
                {
                    probabilities[k] = k == selected ? 1.0F : 0.0F;
                }
            }
            else
            {
                float const invTemperature = 1.0F / temperatures[batch];
                float maxScore = -FLT_MAX;
                for (int32_t k = 0; k < topK; ++k)
                {
                    maxScore = fmaxf(maxScore, scores[k] * invTemperature);
                }
                float sum = 0.0F;
                for (int32_t k = 0; k < topK; ++k)
                {
                    probabilities[k] = expf(scores[k] * invTemperature - maxScore);
                    sum += probabilities[k];
                }
                float cumulative = 0.0F;
                float const uniform = fminf(fmaxf(uniforms[row], 0.0F), 0.99999994F);
                selected = topK - 1;
                for (int32_t k = 0; k < topK; ++k)
                {
                    probabilities[k] /= sum;
                }
                for (int32_t k = 0; k < topK; ++k)
                {
                    cumulative += probabilities[k];
                    if (uniform < cumulative && selected == topK - 1)
                    {
                        selected = k;
                    }
                }
            }
            previousToken = candidateIds[row * topK + selected];
            proposalTokenIds[row] = previousToken;
        }
        __syncthreads();
        if (threadIdx.x < topK)
        {
            proposalProbabilities[(batch * steps + step) * topK + threadIdx.x] = probabilities[threadIdx.x];
        }
        __syncthreads();
    }
}

//! isBetterCandidate made strict over support entries: invalid entries share a score and may share a token, and each
//! entry's rank must be unique.
__device__ __forceinline__ bool isBetterSupportEntry(
    float lhsScore, int32_t lhsToken, int32_t lhsIndex, float rhsScore, int32_t rhsToken, int32_t rhsIndex)
{
    return isBetterCandidate(lhsScore, lhsToken, rhsScore, rhsToken)
        || (lhsScore == rhsScore && lhsToken == rhsToken && lhsIndex < rhsIndex);
}

// One CTA per (batch, proposal row). Row d's parent states are the support indices of row d - 1; row 0 has the
// single anchor state. Each warp scores (state, support) pairs; then one thread per pair ranks it in its state's
// list and writes the slot it lands in when the rank is below candidateTopK.
__global__ void selectorTreeCandidatesKernel(int32_t const* __restrict__ supportIds,
    float const* __restrict__ unaryValues, half const* __restrict__ projectedHidden,
    int32_t const* __restrict__ anchorTokenIds, half const* __restrict__ predecessorCodebook,
    half const* __restrict__ successorCodebook, int32_t* __restrict__ candidateTokenIds,
    float* __restrict__ candidateLogProbs, int32_t* __restrict__ candidateChildStates, int32_t proposalRows,
    int32_t supportSize, int32_t rank, int32_t vocabSize, int32_t candidateTopK)
{
    __shared__ float sQuery[kDFlash2MaxSelectorTopK][kDFlash2MaxSelectorRank];
    __shared__ float sScores[kDFlash2MaxSelectorTopK][kDFlash2MaxSelectorTopK];
    __shared__ int32_t sSupport[kDFlash2MaxSelectorTopK];
    __shared__ int32_t sSupportValid[kDFlash2MaxSelectorTopK];

    int32_t const row = static_cast<int32_t>(blockIdx.x);
    int32_t const batch = row / proposalRows;
    int32_t const depthRow = row % proposalRows;
    int32_t const tid = static_cast<int32_t>(threadIdx.x);
    int32_t const warp = tid / kWarpSize;
    int32_t const lane = tid % kWarpSize;
    int32_t const warps = static_cast<int32_t>(blockDim.x) / kWarpSize;
    int32_t const stateCount = depthRow == 0 ? 1 : supportSize;

    if (tid < supportSize)
    {
        int32_t const token = supportIds[row * supportSize + tid];
        bool valid = isVocabToken(token, vocabSize);
        for (int32_t earlier = 0; earlier < tid && valid; ++earlier)
        {
            valid = supportIds[row * supportSize + earlier] != token;
        }
        sSupport[tid] = token;
        sSupportValid[tid] = valid ? 1 : 0;
    }
    for (int32_t idx = tid; idx < stateCount * rank; idx += static_cast<int32_t>(blockDim.x))
    {
        int32_t const state = idx / rank;
        int32_t const r = idx % rank;
        int32_t const predecessor = depthRow == 0 ? anchorTokenIds[batch] : supportIds[(row - 1) * supportSize + state];
        sQuery[state][r] = isVocabToken(predecessor, vocabSize)
            ? selectorQueryElement(predecessorCodebook[predecessor * rank + r], projectedHidden[row * rank + r])
            : 0.0F;
    }
    __syncthreads();

    for (int32_t pair = warp; pair < stateCount * supportSize; pair += warps)
    {
        int32_t const state = pair / supportSize;
        int32_t const index = pair % supportSize;
        if (sSupportValid[index] == 0)
        {
            if (lane == 0)
            {
                sScores[state][index] = -INFINITY;
            }
            continue;
        }
        float const dot = selectorWarpDot(sQuery[state], successorCodebook + sSupport[index] * rank, rank, lane);
        if (lane == 0)
        {
            sScores[state][index] = unaryValues[row * supportSize + index] + dot;
        }
    }
    __syncthreads();

    int32_t const listBase = row * supportSize;
    for (int32_t pair = tid; pair < supportSize * supportSize; pair += static_cast<int32_t>(blockDim.x))
    {
        int32_t const state = pair / supportSize;
        int32_t const index = pair % supportSize;
        int32_t const out = (listBase + state) * candidateTopK;
        if (state >= stateCount)
        {
            if (index < candidateTopK)
            {
                candidateTokenIds[out + index] = 0;
                candidateLogProbs[out + index] = -INFINITY;
                candidateChildStates[out + index] = 0;
            }
            continue;
        }
        float maxScore = -INFINITY;
        for (int32_t k = 0; k < supportSize; ++k)
        {
            maxScore = fmaxf(maxScore, sScores[state][k]);
        }
        float sum = 0.0F;
        if (maxScore > -INFINITY)
        {
            for (int32_t k = 0; k < supportSize; ++k)
            {
                sum += expf(sScores[state][k] - maxScore);
            }
        }
        float const score = sScores[state][index];
        int32_t const token = sSupport[index];
        int32_t rankInList = 0;
        for (int32_t k = 0; k < supportSize; ++k)
        {
            rankInList += isBetterSupportEntry(sScores[state][k], sSupport[k], k, score, token, index) ? 1 : 0;
        }
        if (rankInList < candidateTopK)
        {
            bool const finite = score > -INFINITY;
            candidateTokenIds[out + rankInList] = finite ? token : 0;
            candidateLogProbs[out + rankInList] = finite ? score - (maxScore + logf(sum)) : -INFINITY;
            candidateChildStates[out + rankInList] = finite ? index : 0;
        }
    }
}

void launchKernel(int32_t const* candidateIds, float const* unaryLogits, half const* projectedHidden,
    int32_t const* anchorTokenIds, half const* predecessorCodebook, half const* successorCodebook,
    float const* uniforms, float const* temperatures, int32_t const* greedyMask, int32_t* proposalTokenIds,
    float* proposalProbabilities, int32_t batchSize, int32_t steps, int32_t topK, int32_t rank, int32_t vocabSize,
    cudaStream_t stream)
{
    candidateSelectorKernel<<<batchSize, 512, 0, stream>>>(candidateIds, unaryLogits, projectedHidden, anchorTokenIds,
        predecessorCodebook, successorCodebook, uniforms, temperatures, greedyMask, proposalTokenIds,
        proposalProbabilities, steps, topK, rank, vocabSize);
    CUDA_CHECK(cudaGetLastError());
}

bool isGpuTensor(rt::Tensor const& tensor) noexcept
{
    return tensor.getDeviceType() == rt::DeviceType::kGPU && tensor.rawPointer() != nullptr;
}

//! Inputs the chain and tree selectors share: candidate ids and unary logits [batch, rows, topK <= 16], projected
//! hidden [batch, rows, rank], anchors [batch] and both codebooks [vocab, rank], with rank a multiple of 32 up to 256.
bool hasSelectorInputContract(rt::Tensor const& candidateIds, rt::Tensor const& unaryLogits,
    rt::Tensor const& projectedHidden, rt::Tensor const& anchorTokenIds, rt::Tensor const& predecessorCodebook,
    rt::Tensor const& successorCodebook) noexcept
{
    rt::Coords const candidateShape = candidateIds.getShape();
    if (candidateShape.getNumDims() != 3)
    {
        return false;
    }
    rt::Coords const projectedShape = projectedHidden.getShape();
    rt::Coords const codebookShape = predecessorCodebook.getShape();
    if (projectedShape.getNumDims() != 3 || codebookShape.getNumDims() != 2)
    {
        return false;
    }
    constexpr int64_t kInt32Max = std::numeric_limits<int32_t>::max();
    if (candidateShape[0] <= 0 || candidateShape[0] > kInt32Max || candidateShape[1] <= 0
        || candidateShape[1] > kInt32Max || candidateShape[2] <= 0 || candidateShape[2] > kInt32Max
        || projectedShape[2] <= 0 || projectedShape[2] > kInt32Max || codebookShape[0] <= 0
        || codebookShape[0] > kInt32Max)
    {
        return false;
    }
    int32_t const batchSize = static_cast<int32_t>(candidateShape[0]);
    int32_t const steps = static_cast<int32_t>(candidateShape[1]);
    int32_t const topK = static_cast<int32_t>(candidateShape[2]);
    int32_t const rank = static_cast<int32_t>(projectedShape[2]);
    int32_t const vocabSize = static_cast<int32_t>(codebookShape[0]);
    trt_edgellm::DataType const activationType = projectedHidden.getDataType();
    if (topK > kDFlash2MaxSelectorTopK || rank > kDFlash2MaxSelectorRank || rank % 32 != 0
        || candidateIds.getDataType() != trt_edgellm::DataType::kINT32
        || unaryLogits.getDataType() != trt_edgellm::DataType::kFLOAT || activationType != trt_edgellm::DataType::kHALF
        || anchorTokenIds.getDataType() != trt_edgellm::DataType::kINT32
        || predecessorCodebook.getDataType() != activationType || successorCodebook.getDataType() != activationType)
    {
        return false;
    }
    if (unaryLogits.getShape() != candidateShape || projectedShape != rt::Coords{batchSize, steps, rank}
        || anchorTokenIds.getShape() != rt::Coords{batchSize} || codebookShape != rt::Coords{vocabSize, rank}
        || successorCodebook.getShape() != codebookShape)
    {
        return false;
    }
    return isGpuTensor(candidateIds) && isGpuTensor(unaryLogits) && isGpuTensor(projectedHidden)
        && isGpuTensor(anchorTokenIds) && isGpuTensor(predecessorCodebook) && isGpuTensor(successorCodebook);
}

bool hasCandidateSelectorContract(rt::Tensor const& candidateIds, rt::Tensor const& unaryLogits,
    rt::Tensor const& projectedHidden, rt::Tensor const& anchorTokenIds, rt::Tensor const& predecessorCodebook,
    rt::Tensor const& successorCodebook, rt::Tensor const& uniforms, rt::Tensor const& temperatures,
    rt::Tensor const& greedyMask, rt::Tensor const& proposalTokenIds, rt::Tensor const& proposalProbabilities) noexcept
{
    if (!hasSelectorInputContract(
            candidateIds, unaryLogits, projectedHidden, anchorTokenIds, predecessorCodebook, successorCodebook))
    {
        return false;
    }
    rt::Coords const candidateShape = candidateIds.getShape();
    rt::Coords const rowShape{candidateShape[0], candidateShape[1]};
    rt::Coords const batchShape{candidateShape[0]};
    return uniforms.getDataType() == trt_edgellm::DataType::kFLOAT
        && temperatures.getDataType() == trt_edgellm::DataType::kFLOAT
        && greedyMask.getDataType() == trt_edgellm::DataType::kINT32
        && proposalTokenIds.getDataType() == trt_edgellm::DataType::kINT32
        && proposalProbabilities.getDataType() == trt_edgellm::DataType::kFLOAT && uniforms.getShape() == rowShape
        && temperatures.getShape() == batchShape && greedyMask.getShape() == batchShape
        && proposalTokenIds.getShape() == rowShape && proposalProbabilities.getShape() == candidateShape
        && isGpuTensor(uniforms) && isGpuTensor(temperatures) && isGpuTensor(greedyMask)
        && isGpuTensor(proposalTokenIds) && isGpuTensor(proposalProbabilities);
}

} // namespace

void launchDFlash2CandidateSelector(rt::Tensor const& candidateIds, rt::Tensor const& unaryLogits,
    rt::Tensor const& projectedHidden, rt::Tensor const& anchorTokenIds, rt::Tensor const& predecessorCodebook,
    rt::Tensor const& successorCodebook, rt::Tensor const& uniforms, rt::Tensor const& temperatures,
    rt::Tensor const& greedyMask, rt::Tensor& proposalTokenIds, rt::Tensor& proposalProbabilities, cudaStream_t stream)
{
    check::check(
        hasCandidateSelectorContract(candidateIds, unaryLogits, projectedHidden, anchorTokenIds, predecessorCodebook,
            successorCodebook, uniforms, temperatures, greedyMask, proposalTokenIds, proposalProbabilities),
        "DFlash2 candidate selector tensor contract does not match");
    rt::Coords const candidateShape = candidateIds.getShape();
    rt::Coords const projectedShape = projectedHidden.getShape();
    rt::Coords const codebookShape = predecessorCodebook.getShape();
    int32_t const batchSize = static_cast<int32_t>(candidateShape[0]);
    int32_t const steps = static_cast<int32_t>(candidateShape[1]);
    int32_t const topK = static_cast<int32_t>(candidateShape[2]);
    int32_t const rank = static_cast<int32_t>(projectedShape[2]);
    int32_t const vocabSize = static_cast<int32_t>(codebookShape[0]);
    launchKernel(candidateIds.dataPointer<int32_t>(), unaryLogits.dataPointer<float>(),
        projectedHidden.dataPointer<half>(), anchorTokenIds.dataPointer<int32_t>(),
        predecessorCodebook.dataPointer<half>(), successorCodebook.dataPointer<half>(), uniforms.dataPointer<float>(),
        temperatures.dataPointer<float>(), greedyMask.dataPointer<int32_t>(), proposalTokenIds.dataPointer<int32_t>(),
        proposalProbabilities.dataPointer<float>(), batchSize, steps, topK, rank, vocabSize, stream);
}

void dflash2SelectorTreeCandidates(rt::Tensor const& supportIds, rt::Tensor const& unaryValues,
    rt::Tensor const& projectedHidden, rt::Tensor const& anchorTokenIds, rt::Tensor const& predecessorCodebook,
    rt::Tensor const& successorCodebook, rt::Tensor& candidateTokenIds, rt::Tensor& candidateLogProbs,
    rt::Tensor& candidateChildStates, cudaStream_t stream)
{
    check::check(hasSelectorInputContract(
                     supportIds, unaryValues, projectedHidden, anchorTokenIds, predecessorCodebook, successorCodebook),
        "DFlash2 tree selector inputs do not match the selector contract.");
    rt::Coords const supportShape = supportIds.getShape();
    rt::Coords const tableShape = candidateTokenIds.getShape();
    int32_t const batchSize = static_cast<int32_t>(supportShape[0]);
    int32_t const proposalRows = static_cast<int32_t>(supportShape[1]);
    int32_t const supportSize = static_cast<int32_t>(supportShape[2]);
    int32_t const rank = static_cast<int32_t>(projectedHidden.getShape()[2]);
    int32_t const vocabSize = static_cast<int32_t>(predecessorCodebook.getShape()[0]);
    check::check(tableShape.getNumDims() == 4 && tableShape[0] == batchSize && tableShape[1] == proposalRows
            && tableShape[2] == supportSize && tableShape[3] > 0 && tableShape[3] <= supportSize
            && candidateLogProbs.getShape() == tableShape && candidateChildStates.getShape() == tableShape
            && candidateTokenIds.getDataType() == trt_edgellm::DataType::kINT32
            && candidateLogProbs.getDataType() == trt_edgellm::DataType::kFLOAT
            && candidateChildStates.getDataType() == trt_edgellm::DataType::kINT32 && isGpuTensor(candidateTokenIds)
            && isGpuTensor(candidateLogProbs) && isGpuTensor(candidateChildStates),
        "DFlash2 tree candidate table must be GPU INT32 / FP32 / INT32 [batch, proposalRows, selectorTopK, "
        "candidateTopK <= selectorTopK].");
    int32_t const candidateTopK = static_cast<int32_t>(tableShape[3]);

    selectorTreeCandidatesKernel<<<batchSize * proposalRows, kTreeCandidateThreads, 0, stream>>>(
        supportIds.dataPointer<int32_t>(), unaryValues.dataPointer<float>(), projectedHidden.dataPointer<half>(),
        anchorTokenIds.dataPointer<int32_t>(), predecessorCodebook.dataPointer<half>(),
        successorCodebook.dataPointer<half>(), candidateTokenIds.dataPointer<int32_t>(),
        candidateLogProbs.dataPointer<float>(), candidateChildStates.dataPointer<int32_t>(), proposalRows, supportSize,
        rank, vocabSize, candidateTopK);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace kernel
} // namespace trt_edgellm
