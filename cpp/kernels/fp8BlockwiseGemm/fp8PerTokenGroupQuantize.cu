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

#include "fp8PerTokenGroupQuantize.h"

#include <cmath>
#include <limits>

namespace trt_edgellm::kernels
{
namespace
{

constexpr int32_t kWarpSize{32};
constexpr int32_t kWarpsPerBlock{kFp8BlockwiseGroupSize / kWarpSize};
constexpr float kFp8E4m3Max{448.0F};

__device__ __forceinline__ float warpReduceMax(float value)
{
#pragma unroll
    for (int32_t offset = kWarpSize / 2; offset > 0; offset /= 2)
    {
        value = fmaxf(value, __shfl_down_sync(0xFFFFFFFFU, value, offset));
    }
    return value;
}

__global__ void fp8PerTokenGroupQuantizeKernel(half const* input, __nv_fp8_e4m3* output, float* scales)
{
    int32_t const lane = static_cast<int32_t>(threadIdx.x) % kWarpSize;
    int32_t const warp = static_cast<int32_t>(threadIdx.x) / kWarpSize;
    int64_t const group = static_cast<int64_t>(blockIdx.x);
    int64_t const element = group * kFp8BlockwiseGroupSize + threadIdx.x;

    float const value = __half2float(input[element]);
    float localMax = warpReduceMax(fabsf(value));

    __shared__ float warpMaxima[kWarpsPerBlock];
    __shared__ float groupScale;
    if (lane == 0)
    {
        warpMaxima[warp] = localMax;
    }
    __syncthreads();

    if (warp == 0)
    {
        localMax = lane < kWarpsPerBlock ? warpMaxima[lane] : 0.0F;
        localMax = warpReduceMax(localMax);
        if (lane == 0)
        {
            groupScale = localMax > 0.0F ? localMax / kFp8E4m3Max : 1.0F;
            scales[group] = groupScale;
        }
    }
    __syncthreads();

    float const quantized = fmaxf(-kFp8E4m3Max, fminf(kFp8E4m3Max, value / groupScale));
    output[element] = __nv_fp8_e4m3(quantized);
}

} // namespace

cudaError_t launchFp8PerTokenGroupQuantize(half const* input, __nv_fp8_e4m3* output, float* scales, int32_t numTokens,
    int32_t hiddenSize, cudaStream_t stream) noexcept
{
    if (input == nullptr || output == nullptr || scales == nullptr || stream == nullptr || numTokens <= 0
        || hiddenSize <= 0 || hiddenSize % kFp8BlockwiseGroupSize != 0)
    {
        return cudaErrorInvalidValue;
    }

    int64_t const groups = static_cast<int64_t>(numTokens) * (hiddenSize / kFp8BlockwiseGroupSize);
    if (groups <= 0 || groups > std::numeric_limits<int32_t>::max())
    {
        return cudaErrorInvalidValue;
    }

    fp8PerTokenGroupQuantizeKernel<<<static_cast<uint32_t>(groups), kFp8BlockwiseGroupSize, 0, stream>>>(
        input, output, scales);
    return cudaPeekAtLastError();
}

} // namespace trt_edgellm::kernels
