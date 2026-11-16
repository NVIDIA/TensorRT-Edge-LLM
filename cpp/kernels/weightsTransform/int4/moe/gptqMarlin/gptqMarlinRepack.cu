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

// GPU GPTQ to Marlin MoE repack. Layout and pack indices match Edge-LLM
// Marlin. The algorithm follows vLLM gptq_marlin_repack without host nibble
// materialization and remaps zero points to Marlin's (q - 8) convention.

#include "gptqMarlinRepack.h"
#include "kernels/weightsTransform/common/checkpointSourceBatch.h"
#include "kernels/weightsTransform/common/marlinRepackIndex.cuh"

namespace trt_edgellm
{
namespace kernel
{
namespace
{

__device__ inline int32_t loadGptqNibble(
    int32_t const* qweight /*[E,K/8,N]*/, int32_t Estride, int32_t K8, int32_t N, int32_t e, int32_t k, int32_t n)
{
    int32_t const row = k / 8;
    int32_t const pack = k % 8;
    int32_t const word = qweight[static_cast<size_t>(e) * Estride + static_cast<size_t>(row) * N + n];
    return (word >> (4 * pack)) & 0xF;
}

__device__ inline int32_t loadQzeroNibble(
    int32_t const* qzeros /*[E,G,N/8]*/, int32_t EstrideZ, int32_t G, int32_t Ndiv8, int32_t e, int32_t g, int32_t n)
{
    int32_t const od = n / 8;
    int32_t const pack = n % 8;
    int32_t const word = qzeros[static_cast<size_t>(e) * EstrideZ + static_cast<size_t>(g) * Ndiv8 + od];
    return (word >> (4 * pack)) & 0xF;
}

__global__ void gptqMarlinRepackKernel(int32_t const* qweight, int32_t const* qzeros, int32_t* marlinOut, int32_t E,
    int32_t N, int32_t K, int32_t numGroups, int32_t groupSize, int32_t zeroPointOffset)
{
    int32_t const expert = static_cast<int32_t>(blockIdx.z);
    int32_t const kTile = static_cast<int32_t>(blockIdx.y);
    int32_t const nTile = static_cast<int32_t>(blockIdx.x);
    int32_t const slot = static_cast<int32_t>(threadIdx.x);
    if (expert >= E || slot >= 128)
    {
        return;
    }
    int32_t const kTiles = K / 16;
    int32_t const nTiles = N / 64;
    if (kTile >= kTiles || nTile >= nTiles)
    {
        return;
    }

    int32_t const inDiv8 = K / 8;
    int32_t const Ndiv8 = N / 8;
    size_t const Estride = static_cast<size_t>(inDiv8) * static_cast<size_t>(N);
    size_t const EstrideZ = static_cast<size_t>(numGroups) * static_cast<size_t>(Ndiv8);
    bool const hasZeros = (qzeros != nullptr) && (numGroups > 0);

    uint32_t packed = 0;
#pragma unroll
    for (int32_t p = 0; p < 8; ++p)
    {
        int32_t const packSrc = marlin_repack::packIndex(p);
        int32_t const rIdx = marlin_repack::rowIndex(slot, packSrc);
        int32_t const cIdx = marlin_repack::columnIndex(slot, packSrc);
        int32_t const n = nTile * 64 + cIdx;
        int32_t const k = kTile * 16 + rIdx;
        int32_t q = loadGptqNibble(qweight, static_cast<int32_t>(Estride), inDiv8, N, expert, k, n);
        if (hasZeros)
        {
            int32_t const g = min(k / groupSize, numGroups - 1);
            int32_t const z = loadQzeroNibble(qzeros, static_cast<int32_t>(EstrideZ), numGroups, Ndiv8, expert, g, n);
            q = q - z - zeroPointOffset + 8;
            q = max(0, min(15, q));
        }
        packed |= static_cast<uint32_t>(q & 0xF) << (4 * p);
    }

    int32_t const outCol = nTile * 128 + marlin_repack::outputIndex(slot);
    marlinOut[(static_cast<size_t>(expert) * kTiles + kTile) * static_cast<size_t>(nTiles * 128) + outCol]
        = static_cast<int32_t>(packed);
}

__global__ void gptqMarlinRepackSourceBatchKernel(CheckpointSourceBatch<int32_t> firstQweights,
    CheckpointSourceBatch<int32_t> secondQweights, CheckpointSourceBatch<int32_t> firstQzeros,
    CheckpointSourceBatch<int32_t> secondQzeros, int32_t* marlinOut, int32_t E, int32_t projectionN, int32_t K,
    int32_t numGroups, int32_t groupSize, int32_t zeroPointOffset, int32_t paired)
{
    int32_t const expert = static_cast<int32_t>(blockIdx.z);
    int32_t const kTile = static_cast<int32_t>(blockIdx.y);
    int32_t const nTile = static_cast<int32_t>(blockIdx.x);
    int32_t const slot = static_cast<int32_t>(threadIdx.x);
    int32_t const N = paired ? 2 * projectionN : projectionN;
    int32_t const kTiles = K / 16;
    int32_t const nTiles = N / 64;
    if (expert >= E || slot >= 128 || kTile >= kTiles || nTile >= nTiles)
    {
        return;
    }

    uint32_t packed = 0;
#pragma unroll
    for (int32_t p = 0; p < 8; ++p)
    {
        int32_t const packSrc = marlin_repack::packIndex(p);
        int32_t const rIdx = marlin_repack::rowIndex(slot, packSrc);
        int32_t const cIdx = marlin_repack::columnIndex(slot, packSrc);
        int32_t const n = nTile * 64 + cIdx;
        int32_t const k = kTile * 16 + rIdx;
        bool const useSecond = paired && n >= projectionN;
        int32_t const projectionColumn = useSecond ? n - projectionN : n;
        int32_t const* qweight = useSecond ? secondQweights.get(expert) : firstQweights.get(expert);
        int32_t const word = qweight[static_cast<size_t>(k / 8) * projectionN + projectionColumn];
        int32_t value = (word >> (4 * (k % 8))) & 0xF;

        int32_t const* qzeros = useSecond ? secondQzeros.get(expert) : firstQzeros.get(expert);
        if (qzeros != nullptr && numGroups > 0)
        {
            int32_t const zeroWord
                = qzeros[static_cast<size_t>(k / groupSize) * (projectionN / 8) + projectionColumn / 8];
            int32_t const zero = (zeroWord >> (4 * (projectionColumn % 8))) & 0xF;
            value = max(0, min(15, value - zero - zeroPointOffset + 8));
        }
        packed |= static_cast<uint32_t>(value) << (4 * p);
    }

    int32_t const outCol = nTile * 128 + marlin_repack::outputIndex(slot);
    marlinOut[(static_cast<size_t>(expert) * kTiles + kTile) * static_cast<size_t>(nTiles * 128) + outCol]
        = static_cast<int32_t>(packed);
}

} // namespace

cudaError_t launchGptqMarlinRepack(int32_t const* dQweightE_K8_N, int32_t const* dQzerosE_G_N8_orNull, int32_t* dMarlin,
    int32_t E, int32_t N, int32_t K, int32_t numGroups, int32_t groupSize, int32_t zeroPointOffset, cudaStream_t stream)
{
    if (E <= 0 || N <= 0 || K <= 0 || (K % 16) != 0 || (N % 64) != 0)
    {
        return cudaErrorInvalidValue;
    }
    if (dQzerosE_G_N8_orNull != nullptr)
    {
        if (numGroups <= 0 || groupSize <= 0 || (numGroups * groupSize) != K)
        {
            return cudaErrorInvalidValue;
        }
    }
    int32_t const kTiles = K / 16;
    int32_t const nTiles = N / 64;
    dim3 grid(nTiles, kTiles, E);
    dim3 block(128);
    gptqMarlinRepackKernel<<<grid, block, 0, stream>>>(
        dQweightE_K8_N, dQzerosE_G_N8_orNull, dMarlin, E, N, K, numGroups, groupSize, zeroPointOffset);
    return cudaGetLastError();
}

cudaError_t launchGptqMarlinRepackSourceBatch(int32_t const* const* firstQweights, int32_t const* const* secondQweights,
    int32_t const* const* firstQzeros, int32_t const* const* secondQzeros, int32_t* marlinOutput, int32_t count,
    int32_t projectionN, int32_t K, int32_t numGroups, int32_t groupSize, int32_t zeroPointOffset, cudaStream_t stream)
{
    bool const paired = secondQweights != nullptr;
    bool const hasZeros = firstQzeros != nullptr;
    if (firstQweights == nullptr || marlinOutput == nullptr || count <= 0 || count > kCheckpointSourcesPerLaunch
        || projectionN <= 0 || K <= 0 || (projectionN % 64) != 0 || (K % 16) != 0
        || (hasZeros && (numGroups <= 0 || numGroups * groupSize != K))
        || (paired && hasZeros != (secondQzeros != nullptr)))
    {
        return cudaErrorInvalidValue;
    }

    for (int32_t index = 0; index < count; ++index)
    {
        if (firstQweights[index] == nullptr || (paired && secondQweights[index] == nullptr)
            || (hasZeros && firstQzeros[index] == nullptr) || (paired && hasZeros && secondQzeros[index] == nullptr))
        {
            return cudaErrorInvalidValue;
        }
    }
    int32_t const N = paired ? 2 * projectionN : projectionN;
    dim3 const grid(N / 64, K / 16, count);
    gptqMarlinRepackSourceBatchKernel<<<grid, 128, 0, stream>>>(makeCheckpointSourceBatch(firstQweights, count),
        makeCheckpointSourceBatch(paired ? secondQweights : nullptr, paired ? count : 0),
        makeCheckpointSourceBatch(hasZeros ? firstQzeros : nullptr, hasZeros ? count : 0),
        makeCheckpointSourceBatch(paired && hasZeros ? secondQzeros : nullptr, paired && hasZeros ? count : 0),
        marlinOutput, count, projectionN, K, numGroups, groupSize, zeroPointOffset, paired ? 1 : 0);
    return cudaGetLastError();
}

} // namespace kernel
} // namespace trt_edgellm
