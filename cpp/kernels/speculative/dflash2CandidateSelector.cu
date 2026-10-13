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

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cfloat>
#include <limits>

namespace trt_edgellm
{
namespace kernel
{
namespace
{

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
            predecessorHidden[r] = previousToken >= 0 && previousToken < vocabSize
                ? static_cast<float>(predecessorCodebook[previousToken * rank + r])
                    * static_cast<float>(projectedHidden[row * rank + r])
                : 0.0F;
        }
        __syncthreads();

        if (warp < topK)
        {
            int32_t const candidate = candidateIds[row * topK + warp];
            float dot = 0.0F;
            if (candidate >= 0 && candidate < vocabSize)
            {
                for (int32_t r = lane; r < rank; r += 32)
                {
                    float const succ = static_cast<float>(successorCodebook[candidate * rank + r]);
                    dot = fmaf(predecessorHidden[r], succ, dot);
                }
            }
            for (int32_t offset = 16; offset > 0; offset >>= 1)
            {
                dot += __shfl_down_sync(0xFFFFFFFFU, dot, offset);
            }
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

bool hasCandidateSelectorContract(rt::Tensor const& candidateIds, rt::Tensor const& unaryLogits,
    rt::Tensor const& projectedHidden, rt::Tensor const& anchorTokenIds, rt::Tensor const& predecessorCodebook,
    rt::Tensor const& successorCodebook, rt::Tensor const& uniforms, rt::Tensor const& temperatures,
    rt::Tensor const& greedyMask, rt::Tensor const& proposalTokenIds, rt::Tensor const& proposalProbabilities) noexcept
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
    nvinfer1::DataType const activationType = projectedHidden.getDataType();
    if (topK > kDFlash2MaxSelectorTopK || rank > kDFlash2MaxSelectorRank || rank % 32 != 0
        || candidateIds.getDataType() != nvinfer1::DataType::kINT32
        || unaryLogits.getDataType() != nvinfer1::DataType::kFLOAT || activationType != nvinfer1::DataType::kHALF
        || anchorTokenIds.getDataType() != nvinfer1::DataType::kINT32
        || predecessorCodebook.getDataType() != activationType || successorCodebook.getDataType() != activationType
        || uniforms.getDataType() != nvinfer1::DataType::kFLOAT
        || temperatures.getDataType() != nvinfer1::DataType::kFLOAT
        || greedyMask.getDataType() != nvinfer1::DataType::kINT32
        || proposalTokenIds.getDataType() != nvinfer1::DataType::kINT32
        || proposalProbabilities.getDataType() != nvinfer1::DataType::kFLOAT)
    {
        return false;
    }
    if (unaryLogits.getShape() != candidateShape || projectedShape != rt::Coords{batchSize, steps, rank}
        || anchorTokenIds.getShape() != rt::Coords{batchSize} || codebookShape != rt::Coords{vocabSize, rank}
        || successorCodebook.getShape() != codebookShape || uniforms.getShape() != rt::Coords{batchSize, steps}
        || temperatures.getShape() != rt::Coords{batchSize} || greedyMask.getShape() != rt::Coords{batchSize}
        || proposalTokenIds.getShape() != rt::Coords{batchSize, steps}
        || proposalProbabilities.getShape() != candidateShape)
    {
        return false;
    }
    return isGpuTensor(candidateIds) && isGpuTensor(unaryLogits) && isGpuTensor(projectedHidden)
        && isGpuTensor(anchorTokenIds) && isGpuTensor(predecessorCodebook) && isGpuTensor(successorCodebook)
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

} // namespace kernel
} // namespace trt_edgellm
