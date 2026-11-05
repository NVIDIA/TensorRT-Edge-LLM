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
#include "awqZeroCorrection.h"

#include <algorithm>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace trt_edgellm
{
namespace kernel
{
namespace
{

__global__ void awqZeroCorrectionKernel(
    int32_t const* qzeros, void const* scales, float* output, int32_t groups, int32_t N, AwqScaleType scaleType)
{
    int64_t const count = static_cast<int64_t>(groups) * N;
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; index < count;
        index += static_cast<int64_t>(blockDim.x) * gridDim.x)
    {
        constexpr int32_t reverseOrder[8] = {0, 4, 1, 5, 2, 6, 3, 7};
        int32_t const channel = static_cast<int32_t>(index % N);
        int32_t const group = static_cast<int32_t>(index / N);
        uint32_t const word = static_cast<uint32_t>(qzeros[static_cast<int64_t>(group) * (N / 8) + channel / 8]);
        int32_t const zero = static_cast<int32_t>((word >> (4 * reverseOrder[channel % 8])) & 0xF);
        float scale;
        if (scaleType == AwqScaleType::kFp16)
        {
            scale = __half2float(static_cast<__half const*>(scales)[index]);
        }
        else if (scaleType == AwqScaleType::kBf16)
        {
            scale = __half2float(__float2half_rn(__bfloat162float(static_cast<__nv_bfloat16 const*>(scales)[index])));
        }
        else
        {
            scale = __half2float(__float2half_rn(static_cast<float const*>(scales)[index]));
        }
        output[index] = static_cast<float>(8 - zero) * scale;
    }
}

} // namespace

cudaError_t launchAwqZeroCorrection(int32_t const* qzeros, void const* scales, float* output, int32_t groups, int32_t N,
    AwqScaleType scaleType, cudaStream_t stream)
{
    if (qzeros == nullptr || scales == nullptr || output == nullptr || groups <= 0 || N <= 0 || N % 8 != 0)
    {
        return cudaErrorInvalidValue;
    }
    constexpr int32_t threads = 256;
    int64_t const count = static_cast<int64_t>(groups) * N;
    int32_t const blocks = static_cast<int32_t>(std::min<int64_t>((count + threads - 1) / threads, 65535));
    awqZeroCorrectionKernel<<<blocks, threads, 0, stream>>>(qzeros, scales, output, groups, N, scaleType);
    return cudaGetLastError();
}

} // namespace kernel
} // namespace trt_edgellm
