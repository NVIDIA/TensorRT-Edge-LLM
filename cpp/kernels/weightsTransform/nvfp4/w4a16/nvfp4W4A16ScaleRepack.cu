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

#include "nvfp4W4A16MarlinRepack.h"

#include "kernels/weightsTransform/common/checkpointSourceBatch.h"

namespace trt_edgellm
{
namespace kernel
{
namespace
{

__global__ void nvfp4A16MarlinScaleRepackSourceBatchKernel(CheckpointSourceBatch<uint8_t> firstSources,
    CheckpointSourceBatch<uint8_t> secondSources, int8_t* marlin, int32_t count, int32_t sourceN, int32_t sourceGroups,
    int32_t outputN, int32_t outputGroups, int32_t paired)
{
    int32_t const N = paired ? 2 * outputN : outputN;
    int64_t const index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t const elements = static_cast<int64_t>(count) * outputGroups * N;
    if (index >= elements)
    {
        return;
    }
    int32_t const n = static_cast<int32_t>(index % N);
    int64_t const row = index / N;
    int32_t const expert = static_cast<int32_t>(row / outputGroups);
    int32_t const group = static_cast<int32_t>(row % outputGroups);
    bool const useSecond = paired && n >= outputN;
    int32_t const sourceRow = useSecond ? n - outputN : n;
    uint8_t value = 0;
    if (sourceRow < sourceN && group < sourceGroups)
    {
        uint8_t const* source = useSecond ? secondSources.get(expert) : firstSources.get(expert);
        value = source[static_cast<size_t>(sourceRow) * sourceGroups + group];
    }
    int32_t const offset = n % 64;
    int32_t const permutationRow = offset / 8;
    int32_t const column = offset % 8;
    int32_t const swappedRow = (permutationRow & 4) | ((permutationRow & 1) << 1) | ((permutationRow >> 1) & 1);
    int32_t const outputColumn = (n / 64) * 64 + 8 * column + swappedRow;
    marlin[row * N + outputColumn] = static_cast<int8_t>(value);
}

} // namespace

cudaError_t launchNvfp4W4A16ScaleRepack(uint8_t const* const* firstSources, uint8_t const* const* secondSources,
    int8_t* marlinOutput, int32_t count, int32_t sourceN, int32_t sourceGroups, int32_t outputN, int32_t outputGroups,
    cudaStream_t stream)
{
    bool const paired = secondSources != nullptr;
    if (firstSources == nullptr || marlinOutput == nullptr || count <= 0 || count > kCheckpointSourcesPerLaunch
        || sourceN <= 0 || sourceGroups <= 0 || outputN < sourceN || outputGroups < sourceGroups || (outputN % 64) != 0)
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
    int32_t const N = paired ? 2 * outputN : outputN;
    int64_t const elements = static_cast<int64_t>(count) * outputGroups * N;
    int32_t constexpr threads = 256;
    int32_t const blocks = static_cast<int32_t>((elements + threads - 1) / threads);
    nvfp4A16MarlinScaleRepackSourceBatchKernel<<<blocks, threads, 0, static_cast<cudaStream_t>(stream)>>>(
        makeCheckpointSourceBatch(firstSources, count),
        makeCheckpointSourceBatch(paired ? secondSources : nullptr, paired ? count : 0), marlinOutput, count, sourceN,
        sourceGroups, outputN, outputGroups, paired ? 1 : 0);
    return cudaGetLastError();
}

} // namespace kernel
} // namespace trt_edgellm
