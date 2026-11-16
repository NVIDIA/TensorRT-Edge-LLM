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

#include "fp16MoeAssemble.h"
#include "kernels/weightsTransform/common/checkpointSourceBatch.h"

#include <algorithm>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace trt_edgellm
{
namespace kernel
{
namespace
{

constexpr int32_t kSwigluInterleaveRows = 64;

__device__ __forceinline__ __half toHalf(__half value)
{
    return value;
}

__device__ __forceinline__ __half toHalf(__nv_bfloat16 value)
{
    return __float2half(__bfloat162float(value));
}

__device__ __forceinline__ __half toHalf(float value)
{
    return __float2half(value);
}

template <typename Source>
__global__ void fp16MoeSourceBatchKernel(CheckpointSourceBatch<uint8_t> firstSources,
    CheckpointSourceBatch<uint8_t> secondSources, __half* output, int32_t count, int32_t sourceRows,
    int32_t sourceColumns, int32_t outputRows, int32_t outputColumns, int32_t paired)
{
    int64_t const elementsPerExpert = static_cast<int64_t>(outputRows) * outputColumns;
    int64_t const totalElements = static_cast<int64_t>(count) * elementsPerExpert;
    int64_t outputIndex = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t const stride = static_cast<int64_t>(blockDim.x) * gridDim.x;

    for (; outputIndex < totalElements; outputIndex += stride)
    {
        int32_t const expert = static_cast<int32_t>(outputIndex / elementsPerExpert);
        int64_t const localIndex = outputIndex - static_cast<int64_t>(expert) * elementsPerExpert;
        int32_t const outputRow = static_cast<int32_t>(localIndex / outputColumns);
        int32_t const column = static_cast<int32_t>(localIndex - static_cast<int64_t>(outputRow) * outputColumns);

        if (column >= sourceColumns || (!paired && outputRow >= sourceRows))
        {
            output[outputIndex] = __float2half(0.0f);
            continue;
        }

        int32_t sourceRow = outputRow;
        uint8_t const* sourceBytes = firstSources.get(expert);
        if (paired)
        {
            int32_t const localRow = outputRow % (2 * kSwigluInterleaveRows);
            bool const second = localRow >= kSwigluInterleaveRows;
            sourceRow = (outputRow / (2 * kSwigluInterleaveRows)) * kSwigluInterleaveRows
                + (second ? localRow - kSwigluInterleaveRows : localRow);
            sourceBytes = second ? secondSources.get(expert) : sourceBytes;
        }
        auto const* source = reinterpret_cast<Source const*>(sourceBytes);
        output[outputIndex] = toHalf(source[static_cast<int64_t>(sourceRow) * sourceColumns + column]);
    }
}

template <typename Source>
void launch(CheckpointSourceBatch<uint8_t> const& firstPointers, CheckpointSourceBatch<uint8_t> const& secondPointers,
    void* output, int32_t count, int32_t sourceRows, int32_t sourceColumns, int32_t outputRows, int32_t outputColumns,
    bool paired, cudaStream_t stream)
{
    int64_t const totalElements = static_cast<int64_t>(count) * outputRows * outputColumns;
    int32_t constexpr threads = 256;
    int32_t const blocks
        = static_cast<int32_t>(std::min<int64_t>((totalElements + threads - 1) / threads, static_cast<int64_t>(65535)));
    fp16MoeSourceBatchKernel<Source><<<blocks, threads, 0, stream>>>(firstPointers, secondPointers,
        static_cast<__half*>(output), count, sourceRows, sourceColumns, outputRows, outputColumns, paired ? 1 : 0);
}

} // namespace

cudaError_t launchFp16MoeSourceBatch(uint8_t const* const* firstSources, uint8_t const* const* secondSources,
    int32_t count, void* output, int32_t sourceRows, int32_t sourceColumns, int32_t outputRows, int32_t outputColumns,
    Fp16MoeSourceType sourceType, cudaStream_t stream)
{
    bool const paired = secondSources != nullptr;
    if (firstSources == nullptr || output == nullptr || count <= 0 || count > kCheckpointSourcesPerLaunch
        || sourceRows <= 0 || sourceColumns <= 0 || outputRows < sourceRows || outputColumns < sourceColumns
        || (paired
            && (sourceRows % kSwigluInterleaveRows != 0 || outputRows != 2 * sourceRows
                || outputColumns != sourceColumns)))
    {
        return cudaErrorInvalidValue;
    }
    for (int32_t index = 0; index < count; ++index)
    {
        if (firstSources[index] == nullptr || (paired && secondSources[index] == nullptr))
        {
            return cudaErrorInvalidValue;
        }
    }

    auto const firstPointers = makeCheckpointSourceBatch(firstSources, count);
    auto const secondPointers
        = paired ? makeCheckpointSourceBatch(secondSources, count) : CheckpointSourceBatch<uint8_t>{};
    switch (sourceType)
    {
    case Fp16MoeSourceType::kFp16:
        launch<__half>(firstPointers, secondPointers, output, count, sourceRows, sourceColumns, outputRows,
            outputColumns, paired, stream);
        break;
    case Fp16MoeSourceType::kBf16:
        launch<__nv_bfloat16>(firstPointers, secondPointers, output, count, sourceRows, sourceColumns, outputRows,
            outputColumns, paired, stream);
        break;
    case Fp16MoeSourceType::kFp32:
        launch<float>(firstPointers, secondPointers, output, count, sourceRows, sourceColumns, outputRows,
            outputColumns, paired, stream);
        break;
    default: return cudaErrorInvalidValue;
    }
    return cudaGetLastError();
}

} // namespace kernel
} // namespace trt_edgellm
