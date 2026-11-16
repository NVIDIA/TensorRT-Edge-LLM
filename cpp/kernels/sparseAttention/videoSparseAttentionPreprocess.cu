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

#include "videoSparseAttentionRunner.h"

#include <cuda_bf16.h>

namespace trt_edgellm
{
namespace kernel
{

namespace
{

constexpr int32_t kHEAD_SIZE{VideoSparseAttentionRunner::kHEAD_SIZE};
constexpr int32_t kBLOCK_SIZE{VideoSparseAttentionRunner::kBLOCK_SIZE};
constexpr float kSOFTMAX_SCALE{0x1.6A09E6P-4F}; // float32(1 / sqrt(128))
constexpr float kLOG2E{1.4426950408889634F};
constexpr float kVARIANCE_EPSILON{1.0E-6F};

__device__ __forceinline__ size_t rowOffset(int32_t row, int32_t head, int32_t numHeads)
{
    return (static_cast<size_t>(row) * numHeads + head) * kHEAD_SIZE + threadIdx.x;
}

//! One CTA per (block, head), one thread per channel.
__global__ void reduceKvKernel(__nv_bfloat16 const* __restrict__ k, __nv_bfloat16 const* __restrict__ v,
    __nv_bfloat16* __restrict__ kc, __nv_bfloat16* __restrict__ vc, int32_t sequenceLength, int32_t numHeads)
{
    int32_t const block = blockIdx.x;
    int32_t const head = blockIdx.y;
    int32_t const firstRow = block * kBLOCK_SIZE;
    int32_t const blockLength = min(kBLOCK_SIZE, sequenceLength - firstRow);
    float kSum{0.0F};
    float vSum{0.0F};
    for (int32_t row = 0; row < blockLength; ++row)
    {
        size_t const offset = rowOffset(firstRow + row, head, numHeads);
        kSum += __bfloat162float(k[offset]);
        vSum += __bfloat162float(v[offset]);
    }
    size_t const offset = rowOffset(block, head, numHeads);
    kc[offset] = __float2bfloat16_rn(kSum / static_cast<float>(blockLength));
    vc[offset] = __float2bfloat16_rn(vSum);
}

//! One CTA per head: mean and diagonal variance of the BF16-rounded KC rows.
__global__ void reduceKcStatsKernel(__nv_bfloat16 const* __restrict__ kc, float* __restrict__ kcMean,
    float* __restrict__ kcVarDiag, int32_t numBlocks, int32_t numHeads)
{
    int32_t const head = blockIdx.x;
    float total{0.0F};
    float totalSquare{0.0F};
    for (int32_t block = 0; block < numBlocks; ++block)
    {
        float const value = __bfloat162float(kc[rowOffset(block, head, numHeads)]);
        total += value;
        totalSquare += value * value;
    }
    float const mean = total / static_cast<float>(numBlocks);
    size_t const offset = static_cast<size_t>(head) * kHEAD_SIZE + threadIdx.x;
    kcMean[offset] = mean;
    kcVarDiag[offset] = fmaxf(totalSquare / static_cast<float>(numBlocks) - mean * mean, 0.0F);
}

//! One CTA per (block, head): ``mean + std`` of the query-centroid score against all KC rows, in the
//! log2-scaled softmax domain of the attention kernel.
__global__ void routingThresholdKernel(__nv_bfloat16 const* __restrict__ q, float const* __restrict__ kcMean,
    float const* __restrict__ kcVarDiag, float* __restrict__ threshold, int32_t sequenceLength, int32_t numHeads)
{
    __shared__ float meanTerms[kHEAD_SIZE];
    __shared__ float varianceTerms[kHEAD_SIZE];

    int32_t const block = blockIdx.x;
    int32_t const head = blockIdx.y;
    int32_t const dimension = threadIdx.x;
    int32_t const firstRow = block * kBLOCK_SIZE;
    int32_t const blockLength = min(kBLOCK_SIZE, sequenceLength - firstRow);
    float qSum{0.0F};
    for (int32_t row = 0; row < blockLength; ++row)
    {
        qSum += __bfloat162float(q[rowOffset(firstRow + row, head, numHeads)]);
    }
    float const centroid = qSum / static_cast<float>(blockLength);
    size_t const statistic = static_cast<size_t>(head) * kHEAD_SIZE + dimension;
    meanTerms[dimension] = centroid * kcMean[statistic];
    varianceTerms[dimension] = centroid * centroid * kcVarDiag[statistic];
    __syncthreads();
    for (int32_t stride = kHEAD_SIZE / 2; stride > 0; stride /= 2)
    {
        if (dimension < stride)
        {
            meanTerms[dimension] += meanTerms[dimension + stride];
            varianceTerms[dimension] += varianceTerms[dimension + stride];
        }
        __syncthreads();
    }
    if (dimension == 0)
    {
        float const scale = kSOFTMAX_SCALE * kLOG2E;
        float const variance = fmaxf(varianceTerms[0], 0.0F) * scale * scale;
        threshold[static_cast<size_t>(block) * numHeads + head]
            = meanTerms[0] * scale + sqrtf(variance + kVARIANCE_EPSILON);
    }
}

} // namespace

cudaError_t launchVideoSparseAttentionPreprocess(void const* q, void const* k, void const* v, void* kc, void* vc,
    float* kcMean, float* kcVarDiag, float* threshold, int32_t sequenceLength, int32_t numHeads,
    cudaStream_t stream) noexcept
{
    int32_t const numBlocks = (sequenceLength + kBLOCK_SIZE - 1) / kBLOCK_SIZE;
    dim3 const blockGrid(numBlocks, numHeads);
    reduceKvKernel<<<blockGrid, kHEAD_SIZE, 0, stream>>>(static_cast<__nv_bfloat16 const*>(k),
        static_cast<__nv_bfloat16 const*>(v), static_cast<__nv_bfloat16*>(kc), static_cast<__nv_bfloat16*>(vc),
        sequenceLength, numHeads);
    reduceKcStatsKernel<<<numHeads, kHEAD_SIZE, 0, stream>>>(
        static_cast<__nv_bfloat16 const*>(kc), kcMean, kcVarDiag, numBlocks, numHeads);
    routingThresholdKernel<<<blockGrid, kHEAD_SIZE, 0, stream>>>(
        static_cast<__nv_bfloat16 const*>(q), kcMean, kcVarDiag, threshold, sequenceLength, numHeads);
    return cudaGetLastError();
}

} // namespace kernel
} // namespace trt_edgellm
