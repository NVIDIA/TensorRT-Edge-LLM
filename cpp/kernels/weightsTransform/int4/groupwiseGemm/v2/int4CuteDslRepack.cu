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

#include "int4CuteDslRepack.h"
#include "kernels/weightsTransform/int4/groupwiseGemm/common/int4SourceAccessors.cuh"

namespace trt_edgellm
{
namespace kernel
{
namespace
{

template <typename Accessor>
__device__ uint32_t directFragmentWord(
    Accessor const& accessor, int32_t N, int32_t K, int32_t outputRow, int32_t thread)
{
    int32_t const kTiles = K / 64;
    int32_t const fragment = outputRow % 8;
    int32_t const tile = outputRow / 8;
    int32_t const kTile = tile % kTiles;
    int32_t const nBlock = tile / kTiles;
    int32_t const kBlock = fragment / 2;
    int32_t const nPair = fragment % 2;

    int32_t const nLo = nBlock * 128 + thread / 4 + nPair * 64;
    int32_t const nHi = nLo + 32;
    int32_t const k0 = kTile * 64 + kBlock * 16 + 2 * (thread % 4);
    int32_t const ns[8] = {nLo, nLo, nHi, nHi, nLo, nLo, nHi, nHi};
    int32_t const ks[8] = {k0, k0 + 8, k0, k0 + 8, k0 + 1, k0 + 9, k0 + 1, k0 + 9};

    uint32_t word = 0;
#pragma unroll
    for (int32_t index = 0; index < 8; ++index)
    {
        uint8_t const value = ns[index] < N ? accessor(ns[index], ks[index]) : 8;
        word |= static_cast<uint32_t>(value) << (4 * index);
    }
    return word;
}

template <typename Accessor>
__global__ void directInt4CuteDslRepackKernel(
    Accessor accessor, uint32_t* fragmentWords, int32_t N, int32_t K, int32_t rows)
{
    int32_t const row = static_cast<int32_t>(blockIdx.x);
    int32_t const thread = static_cast<int32_t>(threadIdx.x);
    if (row < rows && thread < 128)
    {
        fragmentWords[static_cast<size_t>(row) * 128 + thread] = directFragmentWord(accessor, N, K, row, thread);
    }
}

} // namespace

int32_t int4CuteDslFragmentRows(int32_t N, int32_t K)
{
    return ((N + 127) / 128) * ((K + 63) / 64) * 8;
}

cudaError_t launchModelOptInt4CuteDslRepack(
    uint8_t const* weightNhalfK, int8_t* fragmentRows512, int32_t N, int32_t K, cudaStream_t stream)
{
    if (weightNhalfK == nullptr || fragmentRows512 == nullptr || N <= 0 || K <= 0 || (K % 64) != 0)
    {
        return cudaErrorInvalidValue;
    }
    int32_t const rows = int4CuteDslFragmentRows(N, K);
    directInt4CuteDslRepackKernel<<<rows, 128, 0, stream>>>(
        ModelOptInt4Accessor{weightNhalfK, K}, reinterpret_cast<uint32_t*>(fragmentRows512), N, K, rows);
    return cudaGetLastError();
}

cudaError_t launchGptqInt4CuteDslRepack(int32_t const* qweightK8N, int32_t const* qzerosGN8,
    int32_t const* activationPermutation, int8_t* fragmentRows512, int32_t N, int32_t K, int32_t numGroups,
    int32_t groupSize, int32_t zeroPointOffset, cudaStream_t stream)
{
    if (qweightK8N == nullptr || fragmentRows512 == nullptr || N <= 0 || K <= 0 || (K % 64) != 0
        || (qzerosGN8 != nullptr && (numGroups <= 0 || numGroups * groupSize != K)))
    {
        return cudaErrorInvalidValue;
    }
    int32_t const rows = int4CuteDslFragmentRows(N, K);
    directInt4CuteDslRepackKernel<<<rows, 128, 0, stream>>>(
        GptqInt4Accessor{qweightK8N, qzerosGN8, activationPermutation, N, groupSize, zeroPointOffset},
        reinterpret_cast<uint32_t*>(fragmentRows512), N, K, rows);
    return cudaGetLastError();
}

cudaError_t launchAwqInt4CuteDslRepack(int32_t const* qweightKN8, int32_t const* qzerosGN8, int8_t* fragmentRows512,
    int32_t N, int32_t K, int32_t numGroups, int32_t groupSize, cudaStream_t stream)
{
    if (qweightKN8 == nullptr || qzerosGN8 == nullptr || fragmentRows512 == nullptr || N <= 0 || K <= 0 || (K % 64) != 0
        || numGroups <= 0 || numGroups * groupSize != K)
    {
        return cudaErrorInvalidValue;
    }
    int32_t const rows = int4CuteDslFragmentRows(N, K);
    directInt4CuteDslRepackKernel<<<rows, 128, 0, stream>>>(
        AwqInt4Accessor{qweightKN8, qzerosGN8, N, groupSize}, reinterpret_cast<uint32_t*>(fragmentRows512), N, K, rows);
    return cudaGetLastError();
}

cudaError_t launchAwqCenteredInt4CuteDslRepack(
    int32_t const* qweightKN8, int8_t* fragmentRows512, int32_t N, int32_t K, cudaStream_t stream)
{
    if (qweightKN8 == nullptr || fragmentRows512 == nullptr || N <= 0 || K <= 0 || (K % 64) != 0)
    {
        return cudaErrorInvalidValue;
    }
    int32_t const rows = int4CuteDslFragmentRows(N, K);
    directInt4CuteDslRepackKernel<<<rows, 128, 0, stream>>>(
        AwqInt4Accessor{qweightKN8, nullptr, N, 0}, reinterpret_cast<uint32_t*>(fragmentRows512), N, K, rows);
    return cudaGetLastError();
}

} // namespace kernel
} // namespace trt_edgellm
