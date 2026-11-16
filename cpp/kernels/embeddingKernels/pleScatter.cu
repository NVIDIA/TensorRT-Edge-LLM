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

#include "pleScatter.h"

#include <algorithm>
#include <cmath>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <limits>

namespace trt_edgellm
{
namespace kernel
{
namespace
{
__global__ void pleScatterKernel(
    uint8_t const* rows, int32_t const* slots, uint8_t* pool, int64_t count, int64_t rowBytes)
{
    for (int64_t row = blockIdx.x; row < count; row += gridDim.x)
    {
        int64_t const destination = static_cast<int64_t>(slots[row]) * rowBytes;
        int64_t const source = row * rowBytes;
        for (int64_t column = threadIdx.x; column < rowBytes; column += blockDim.x)
        {
            pool[destination + column] = rows[source + column];
        }
    }
}

template <typename T, bool kContiguous>
__global__ void pleScatterConvertKernel(
    T const* rows, int32_t const* slots, __half* pool, int64_t count, int64_t width, float scale)
{
    for (int64_t row = blockIdx.x; row < count; row += gridDim.x)
    {
        int64_t const destination = (kContiguous ? row : static_cast<int64_t>(slots[row])) * width;
        for (int64_t column = threadIdx.x; column < width; column += blockDim.x)
        {
            // The FP16 rounding must precede scaling, matching checkpoint weight assembly.
            __half const value = __float2half_rn(static_cast<float>(rows[row * width + column]));
            pool[destination + column] = __float2half_rn(__half2float(value) * scale);
        }
    }
}
} // namespace

cudaError_t launchPleScatter(void const* rows, int32_t const* slots, void* pool, int64_t rowCount, int64_t rowBytes,
    cudaStream_t stream) noexcept
{
    if (rows == nullptr || slots == nullptr || pool == nullptr || stream == nullptr || rowCount <= 0 || rowBytes <= 0
        || rowCount > std::numeric_limits<int64_t>::max() / rowBytes)
    {
        return cudaErrorInvalidValue;
    }
    int32_t const blocks = static_cast<int32_t>(std::min<int64_t>(rowCount, 65535));
    pleScatterKernel<<<blocks, 128, 0, stream>>>(
        static_cast<uint8_t const*>(rows), slots, static_cast<uint8_t*>(pool), rowCount, rowBytes);
    return cudaGetLastError();
}

template <bool kContiguous>
cudaError_t convertRows(void const* rows, int32_t const* slots, void* pool, int64_t rowCount, int64_t rowWidth,
    trt_edgellm::DataType sourceType, float scale, cudaStream_t stream) noexcept
{
    if (rows == nullptr || (!kContiguous && slots == nullptr) || pool == nullptr || stream == nullptr || rowCount <= 0
        || rowWidth <= 0 || rowCount > std::numeric_limits<int64_t>::max() / rowWidth || !std::isfinite(scale)
        || scale <= 0)
    {
        return cudaErrorInvalidValue;
    }
    int32_t const blocks = static_cast<int32_t>(std::min<int64_t>(rowCount, 65535));
    auto* output = static_cast<__half*>(pool);
    switch (sourceType)
    {
    case trt_edgellm::DataType::kHALF:
        pleScatterConvertKernel<__half, kContiguous>
            <<<blocks, 128, 0, stream>>>(static_cast<__half const*>(rows), slots, output, rowCount, rowWidth, scale);
        break;
    case trt_edgellm::DataType::kBF16:
        pleScatterConvertKernel<__nv_bfloat16, kContiguous><<<blocks, 128, 0, stream>>>(
            static_cast<__nv_bfloat16 const*>(rows), slots, output, rowCount, rowWidth, scale);
        break;
    case trt_edgellm::DataType::kFLOAT:
        pleScatterConvertKernel<float, kContiguous>
            <<<blocks, 128, 0, stream>>>(static_cast<float const*>(rows), slots, output, rowCount, rowWidth, scale);
        break;
    default: return cudaErrorInvalidValue;
    }
    return cudaGetLastError();
}

cudaError_t launchPleScatterConvert(void const* rows, int32_t const* slots, void* pool, int64_t rowCount,
    int64_t rowWidth, trt_edgellm::DataType sourceType, float scale, cudaStream_t stream) noexcept
{
    return convertRows<false>(rows, slots, pool, rowCount, rowWidth, sourceType, scale, stream);
}

cudaError_t launchPleConvert(void const* rows, void* output, int64_t rowCount, int64_t rowWidth,
    trt_edgellm::DataType sourceType, float scale, cudaStream_t stream) noexcept
{
    return convertRows<true>(rows, nullptr, output, rowCount, rowWidth, sourceType, scale, stream);
}

} // namespace kernel
} // namespace trt_edgellm
