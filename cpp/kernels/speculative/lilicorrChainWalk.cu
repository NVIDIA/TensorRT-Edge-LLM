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

#include "kernels/speculative/lilicorrChainWalk.h"

#include "common/checkMacros.h"
#include "common/cudaUtils.h"

#include <cuda_runtime.h>

#include <limits>
#include <string>

namespace trt_edgellm
{
namespace kernel
{
namespace
{

constexpr int32_t kWARP_SIZE = 32;
constexpr uint32_t kFULL_WARP_MASK = 0xFFFFFFFFU;
constexpr float kGREEDY_TEMPERATURE_THRESHOLD = 1.0e-3F;
// Largest float below 1. If rounding leaves the total cumulative mass at or below the uniform, the walk picks K-1.
constexpr float kMAX_UNIFORM = 0.99999994F;

static_assert(kLiLiCorrMaxTopK <= kWARP_SIZE, "one lane per candidate");

//! One warp per request; lane k owns candidate k of the current slot.
__global__ void lilicorrChainWalkKernel(int32_t const* __restrict__ candidateIds,
    float const* __restrict__ transitionScores, float const* __restrict__ uniforms,
    float const* __restrict__ temperatures, int32_t const* __restrict__ greedyMask,
    int32_t* __restrict__ proposalTokenIds, float* __restrict__ proposalProbabilities, int32_t steps, int32_t topK)
{
    int64_t const batch = static_cast<int64_t>(blockIdx.x);
    int32_t const lane = static_cast<int32_t>(threadIdx.x);
    bool const active = lane < topK;
    float const temperature = temperatures[batch];
    bool const greedy = greedyMask[batch] != 0 || temperature < kGREEDY_TEMPERATURE_THRESHOLD;

    int32_t previous = 0;
    for (int32_t step = 0; step < steps; ++step)
    {
        int64_t const row = batch * steps + step;
        float const score = active ? transitionScores[(row * topK + previous) * topK + lane]
                                   : -std::numeric_limits<float>::infinity();
        int32_t selected = 0;
        float probability = 0.0F;
        if (greedy)
        {
            // Inactive lanes carry -inf and an index >= topK, so they never beat a real candidate on a tie.
            float bestScore = score;
            int32_t bestIndex = lane;
            for (int32_t offset = kWARP_SIZE / 2; offset > 0; offset >>= 1)
            {
                float const otherScore = __shfl_down_sync(kFULL_WARP_MASK, bestScore, offset);
                int32_t const otherIndex = __shfl_down_sync(kFULL_WARP_MASK, bestIndex, offset);
                if (otherScore > bestScore || (otherScore == bestScore && otherIndex < bestIndex))
                {
                    bestScore = otherScore;
                    bestIndex = otherIndex;
                }
            }
            selected = __shfl_sync(kFULL_WARP_MASK, bestIndex, 0);
            probability = lane == selected ? 1.0F : 0.0F;
        }
        else
        {
            float const scaled = active ? score / temperature : -std::numeric_limits<float>::infinity();
            float maxScaled = scaled;
            for (int32_t offset = kWARP_SIZE / 2; offset > 0; offset >>= 1)
            {
                maxScaled = fmaxf(maxScaled, __shfl_xor_sync(kFULL_WARP_MASK, maxScaled, offset));
            }
            float const weight = active ? expf(scaled - maxScaled) : 0.0F;
            float sum = weight;
            for (int32_t offset = kWARP_SIZE / 2; offset > 0; offset >>= 1)
            {
                sum += __shfl_xor_sync(kFULL_WARP_MASK, sum, offset);
            }
            probability = weight / sum;

            float cumulative = probability;
            for (int32_t offset = 1; offset < kWARP_SIZE; offset <<= 1)
            {
                float const lower = __shfl_up_sync(kFULL_WARP_MASK, cumulative, offset);
                if (lane >= offset)
                {
                    cumulative += lower;
                }
            }
            float const uniform = fminf(fmaxf(uniforms[row], 0.0F), kMAX_UNIFORM);
            uint32_t const hits = __ballot_sync(kFULL_WARP_MASK, active && uniform < cumulative);
            selected = hits != 0U ? __ffs(static_cast<int32_t>(hits)) - 1 : topK - 1;
        }

        if (active)
        {
            proposalProbabilities[row * topK + lane] = probability;
        }
        if (lane == 0)
        {
            proposalTokenIds[row] = candidateIds[row * topK + selected];
        }
        previous = selected;
    }
}

bool isGpuTensor(rt::Tensor const& tensor) noexcept
{
    return tensor.getDeviceType() == rt::DeviceType::kGPU && tensor.rawPointer() != nullptr;
}

bool hasChainWalkContract(rt::Tensor const& candidateIds, rt::Tensor const& transitionScores,
    rt::Tensor const& uniforms, rt::Tensor const& temperatures, rt::Tensor const& greedyMask,
    rt::Tensor const& proposalTokenIds, rt::Tensor const& proposalProbabilities) noexcept
{
    rt::Coords const candidateShape = candidateIds.getShape();
    if (candidateShape.getNumDims() != 3)
    {
        return false;
    }
    constexpr int64_t int32Max = std::numeric_limits<int32_t>::max();
    int64_t const batchSize = candidateShape[0];
    int64_t const steps = candidateShape[1];
    int64_t const topK = candidateShape[2];
    if (batchSize <= 0 || batchSize > int32Max || steps <= 0 || steps > int32Max || topK <= 0
        || topK > kLiLiCorrMaxTopK)
    {
        return false;
    }
    if (candidateIds.getDataType() != trt_edgellm::DataType::kINT32
        || transitionScores.getDataType() != trt_edgellm::DataType::kFLOAT
        || uniforms.getDataType() != trt_edgellm::DataType::kFLOAT
        || temperatures.getDataType() != trt_edgellm::DataType::kFLOAT
        || greedyMask.getDataType() != trt_edgellm::DataType::kINT32
        || proposalTokenIds.getDataType() != trt_edgellm::DataType::kINT32
        || proposalProbabilities.getDataType() != trt_edgellm::DataType::kFLOAT)
    {
        return false;
    }
    if (transitionScores.getShape() != rt::Coords{batchSize, steps, topK, topK}
        || uniforms.getShape() != rt::Coords{batchSize, steps} || temperatures.getShape() != rt::Coords{batchSize}
        || greedyMask.getShape() != rt::Coords{batchSize} || proposalTokenIds.getShape() != rt::Coords{batchSize, steps}
        || proposalProbabilities.getShape() != candidateShape)
    {
        return false;
    }
    return isGpuTensor(candidateIds) && isGpuTensor(transitionScores) && isGpuTensor(uniforms)
        && isGpuTensor(temperatures) && isGpuTensor(greedyMask) && isGpuTensor(proposalTokenIds)
        && isGpuTensor(proposalProbabilities);
}

} // namespace

void launchLiLiCorrChainWalk(rt::Tensor const& candidateIds, rt::Tensor const& transitionScores,
    rt::Tensor const& uniforms, rt::Tensor const& temperatures, rt::Tensor const& greedyMask,
    rt::Tensor& proposalTokenIds, rt::Tensor& proposalProbabilities, cudaStream_t stream)
{
    check::check(hasChainWalkContract(candidateIds, transitionScores, uniforms, temperatures, greedyMask,
                     proposalTokenIds, proposalProbabilities),
        "LiLiCorr chain walk expects GPU tensors ids [B,S,K] INT32 (1 <= K <= " + std::to_string(kLiLiCorrMaxTopK)
            + "), scores [B,S,K,K] FP32, uniforms [B,S] FP32, temperatures [B] FP32, greedyMask [B] INT32, "
              "tokens [B,S] INT32 and q [B,S,K] FP32");
    rt::Coords const candidateShape = candidateIds.getShape();
    int32_t const batchSize = static_cast<int32_t>(candidateShape[0]);
    int32_t const steps = static_cast<int32_t>(candidateShape[1]);
    int32_t const topK = static_cast<int32_t>(candidateShape[2]);
    lilicorrChainWalkKernel<<<batchSize, kWARP_SIZE, 0, stream>>>(candidateIds.dataPointer<int32_t>(),
        transitionScores.dataPointer<float>(), uniforms.dataPointer<float>(), temperatures.dataPointer<float>(),
        greedyMask.dataPointer<int32_t>(), proposalTokenIds.dataPointer<int32_t>(),
        proposalProbabilities.dataPointer<float>(), steps, topK);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace kernel
} // namespace trt_edgellm
