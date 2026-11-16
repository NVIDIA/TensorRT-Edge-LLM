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

#include "int4PluginV1Repack.h"
#include "kernels/weightsTransform/int4/groupwiseGemm/common/int4SourceAccessors.cuh"

namespace trt_edgellm
{
namespace kernel
{
namespace
{

__device__ int32_t permuteKForward(int32_t k)
{
    int32_t const block = k & ~31;
    int32_t const offset = k & 31;
    int32_t const a = offset >> 3;
    int32_t const b = (offset >> 1) & 3;
    int32_t const c = offset & 1;
    int32_t const permuted = block + ((((b << 2) | a) << 1) | c);
    constexpr int32_t reorder[8] = {0, 4, 1, 5, 2, 6, 3, 7};
    return (permuted & ~7) + reorder[permuted & 7];
}

__device__ int32_t permuteKInverse(int32_t destination)
{
    int32_t const block = destination & ~31;
#pragma unroll
    for (int32_t offset = 0; offset < 32; ++offset)
    {
        if (permuteKForward(block + offset) == destination)
        {
            return block + offset;
        }
    }
    return destination;
}

template <typename Accessor>
__global__ void directInt4PluginV1RepackKernel(Accessor accessor, uint16_t* packedWords, int32_t K, int64_t wordCount)
{
    int64_t const index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= wordCount)
    {
        return;
    }

    int32_t const n4 = static_cast<int32_t>(index / K);
    int32_t const k = static_cast<int32_t>(index % K);
    int32_t const kTile = k / 64;
    int32_t const kOffset = k % 64;
    int32_t const n = n4 * 4 + kOffset / 16;
    int32_t const sourceK0 = kTile * 64 + (kOffset % 16) * 4;

    uint16_t word = 0;
#pragma unroll
    for (int32_t nibble = 0; nibble < 4; ++nibble)
    {
        int32_t const sourceK = permuteKInverse(sourceK0 + nibble);
        word |= static_cast<uint16_t>(accessor(n, sourceK) & 0xF) << (4 * nibble);
    }
    packedWords[index] = word;
}

template <typename Accessor>
cudaError_t launchDirectInt4PluginV1Repack(
    Accessor accessor, int8_t* packedNhalfK, int32_t N, int32_t K, cudaStream_t stream)
{
    if (packedNhalfK == nullptr || N <= 0 || K <= 0 || (N % 4) != 0 || (K % 64) != 0)
    {
        return cudaErrorInvalidValue;
    }
    int64_t const words = static_cast<int64_t>(N / 4) * K;
    int32_t constexpr threads = 256;
    int32_t const blocks = static_cast<int32_t>((words + threads - 1) / threads);
    directInt4PluginV1RepackKernel<<<blocks, threads, 0, stream>>>(
        accessor, reinterpret_cast<uint16_t*>(packedNhalfK), K, words);
    return cudaGetLastError();
}

} // namespace

cudaError_t launchModelOptInt4PluginV1Repack(
    uint8_t const* weightNhalfK, int8_t* packedNhalfK, int32_t N, int32_t K, cudaStream_t stream)
{
    if (weightNhalfK == nullptr)
    {
        return cudaErrorInvalidValue;
    }
    return launchDirectInt4PluginV1Repack(ModelOptInt4Accessor{weightNhalfK, K}, packedNhalfK, N, K, stream);
}

cudaError_t launchGptqInt4PluginV1Repack(int32_t const* qweightK8N, int32_t const* qzerosGN8,
    int32_t const* activationPermutation, int8_t* packedNhalfK, int32_t N, int32_t K, int32_t numGroups,
    int32_t groupSize, int32_t zeroPointOffset, bool outputMajor, cudaStream_t stream)
{
    if (qweightK8N == nullptr || (qzerosGN8 != nullptr && (numGroups <= 0 || numGroups * groupSize != K)))
    {
        return cudaErrorInvalidValue;
    }
    return launchDirectInt4PluginV1Repack(
        GptqInt4Accessor{qweightK8N, qzerosGN8, activationPermutation, N, K, groupSize, zeroPointOffset, outputMajor},
        packedNhalfK, N, K, stream);
}

cudaError_t launchAwqInt4PluginV1Repack(
    int32_t const* qweightKN8, int8_t* packedNhalfK, int32_t N, int32_t K, cudaStream_t stream)
{
    if (qweightKN8 == nullptr)
    {
        return cudaErrorInvalidValue;
    }
    return launchDirectInt4PluginV1Repack(AwqInt4Accessor{qweightKN8, N}, packedNhalfK, N, K, stream);
}

} // namespace kernel
} // namespace trt_edgellm
