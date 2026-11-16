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
#include "kernels/weightsTransform/common/marlinRepackIndex.cuh"

namespace trt_edgellm
{
namespace kernel
{
namespace
{

__global__ void nvfp4W4A16MarlinRepackSourceBatchKernel(CheckpointSourceBatch<uint8_t> firstWeights,
    CheckpointSourceBatch<uint8_t> secondWeights, int32_t* marlinOut, int32_t sourceN, int32_t sourceK, int32_t outputN,
    int32_t outputK, int32_t paired)
{
    int32_t const expert = static_cast<int32_t>(blockIdx.z);
    int32_t const kTile = static_cast<int32_t>(blockIdx.y);
    int32_t const nTile = static_cast<int32_t>(blockIdx.x);
    int32_t const slot = static_cast<int32_t>(threadIdx.x);
    int32_t const N = paired ? 2 * outputN : outputN;

    uint32_t packed = 0;
#pragma unroll
    for (int32_t p = 0; p < 8; ++p)
    {
        int32_t const packSrc = marlin_repack::packIndex(p);
        int32_t const n = nTile * 64 + marlin_repack::columnIndex(slot, packSrc);
        int32_t const k = kTile * 16 + marlin_repack::rowIndex(slot, packSrc);
        bool const useSecond = paired && n >= outputN;
        int32_t const sourceRow = useSecond ? n - outputN : n;
        uint32_t value = 0;
        if (sourceRow < sourceN && k < sourceK)
        {
            uint8_t const* source = useSecond ? secondWeights.get(expert) : firstWeights.get(expert);
            uint8_t const byte = source[static_cast<size_t>(sourceRow) * (sourceK / 2) + k / 2];
            value = (byte >> (4 * (k % 2))) & 0xF;
        }
        packed |= value << (4 * p);
    }

    int32_t const outCol = nTile * 128 + marlin_repack::outputIndex(slot);
    marlinOut[(static_cast<size_t>(expert) * (outputK / 16) + kTile) * static_cast<size_t>((N / 64) * 128) + outCol]
        = static_cast<int32_t>(packed);
}

} // namespace

cudaError_t launchNvfp4W4A16MarlinRepackSourceBatch(uint8_t const* const* firstWeights,
    uint8_t const* const* secondWeights, int32_t* marlinOutput, int32_t count, int32_t sourceN, int32_t sourceK,
    int32_t outputN, int32_t outputK, cudaStream_t stream)
{
    bool const paired = secondWeights != nullptr;
    if (firstWeights == nullptr || marlinOutput == nullptr || count <= 0 || count > kCheckpointSourcesPerLaunch
        || sourceN <= 0 || sourceK <= 0 || outputN < sourceN || outputK < sourceK || (sourceK % 2) != 0
        || (outputK % 16) != 0 || (outputN % 64) != 0)
    {
        return cudaErrorInvalidValue;
    }
    for (int32_t index = 0; index < count; ++index)
    {
        if (firstWeights[index] == nullptr || (paired && secondWeights[index] == nullptr))
        {
            return cudaErrorInvalidValue;
        }
    }
    int32_t const N = paired ? 2 * outputN : outputN;
    dim3 const grid(N / 64, outputK / 16, count);
    nvfp4W4A16MarlinRepackSourceBatchKernel<<<grid, 128, 0, stream>>>(makeCheckpointSourceBatch(firstWeights, count),
        makeCheckpointSourceBatch(paired ? secondWeights : nullptr, paired ? count : 0), marlinOutput, sourceN, sourceK,
        outputN, outputK, paired ? 1 : 0);
    return cudaGetLastError();
}

} // namespace kernel
} // namespace trt_edgellm
