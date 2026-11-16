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

#include "iq2XxsRepack.h"
#include "kernels/weightsTransform/common/checkpointSourceBatch.h"

#include <algorithm>

namespace trt_edgellm
{
namespace kernel
{
namespace
{

constexpr int32_t kThreads = 256;
constexpr int32_t kMaxBlocks = 4096;

// Checkpoint blocks are 66 bytes, so payload words and bases are only byte aligned in the mapped source.
__device__ __forceinline__ uint16_t loadU16(uint8_t const* p)
{
    return static_cast<uint16_t>(p[0] | (p[1] << 8));
}

__device__ __forceinline__ int32_t loadI32(uint8_t const* p)
{
    return static_cast<int32_t>(static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8)
        | (static_cast<uint32_t>(p[2]) << 16) | (static_cast<uint32_t>(p[3]) << 24));
}

__global__ void gemmValuesKernel(uint8_t const* source, int8_t* values, int32_t N, int32_t kBlocks, int32_t packing)
{
    int64_t const total = static_cast<int64_t>(N) * kBlocks * 64;
    for (int64_t d = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; d < total;
        d += static_cast<int64_t>(gridDim.x) * blockDim.x)
    {
        int64_t t = d / 64;
        int32_t const byte = static_cast<int32_t>(d % 64);
        int32_t const p = static_cast<int32_t>(t % packing);
        t /= packing;
        int32_t const k = static_cast<int32_t>(t % kBlocks);
        int64_t const group = t / kBlocks;
        int64_t const row = group * packing + p;
        values[d] = static_cast<int8_t>(source[(row * kBlocks + k) * kIq2XxsBlockBytes + 2 + byte]);
    }
}

__global__ void gemmBasesKernel(uint8_t const* source, uint16_t* bases, int32_t N, int32_t kBlocks)
{
    int64_t const total = static_cast<int64_t>(N) * kBlocks;
    for (int64_t d = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; d < total;
        d += static_cast<int64_t>(gridDim.x) * blockDim.x)
    {
        int64_t const k = d / N;
        int64_t const row = d % N;
        bases[d] = loadU16(source + (row * kBlocks + k) * kIq2XxsBlockBytes);
    }
}

// Destination [N/64, K/128, 8, 4, 8, 2] words; word (wo * 8 + wi) of row (n64 * 64 + rq * 16 + h * 8 + j).
__global__ void moeWordsKernel(
    CheckpointSourceBatch<uint8_t> sources, int32_t* words, int32_t N, int32_t kBlocks, int64_t wordsPerExpert)
{
    uint8_t const* source = sources.get(blockIdx.y);
    int32_t* destination = words + blockIdx.y * wordsPerExpert;
    int32_t const wordGroups = kBlocks * 2;
    for (int64_t d = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; d < wordsPerExpert;
        d += static_cast<int64_t>(gridDim.x) * blockDim.x)
    {
        int64_t t = d;
        int32_t const h = static_cast<int32_t>(t % 2);
        t /= 2;
        int32_t const j = static_cast<int32_t>(t % 8);
        t /= 8;
        int32_t const rq = static_cast<int32_t>(t % 4);
        t /= 4;
        int32_t const wi = static_cast<int32_t>(t % 8);
        t /= 8;
        int32_t const wo = static_cast<int32_t>(t % wordGroups);
        int64_t const n64 = t / wordGroups;
        int64_t const row = n64 * 64 + rq * 16 + h * 8 + j;
        int32_t const word = wo * 8 + wi;
        destination[d] = loadI32(source + (row * kBlocks + word / 16) * kIq2XxsBlockBytes + 2 + (word % 16) * 4);
    }
}

// Destination [K/256, N/16, 8, 2] FP16 bases; base of row (g * 16 + h * 8 + j).
__global__ void moeBasesKernel(
    CheckpointSourceBatch<uint8_t> sources, uint16_t* bases, int32_t N, int32_t kBlocks, int64_t basesPerExpert)
{
    uint8_t const* source = sources.get(blockIdx.y);
    uint16_t* destination = bases + blockIdx.y * basesPerExpert;
    int32_t const rowGroups = N / 16;
    for (int64_t d = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; d < basesPerExpert;
        d += static_cast<int64_t>(gridDim.x) * blockDim.x)
    {
        int64_t t = d;
        int32_t const h = static_cast<int32_t>(t % 2);
        t /= 2;
        int32_t const j = static_cast<int32_t>(t % 8);
        t /= 8;
        int32_t const g = static_cast<int32_t>(t % rowGroups);
        int64_t const k = t / rowGroups;
        int64_t const row = g * 16 + h * 8 + j;
        destination[d] = loadU16(source + (row * kBlocks + k) * kIq2XxsBlockBytes);
    }
}

uint32_t gridFor(int64_t total)
{
    return static_cast<uint32_t>(std::min<int64_t>(kMaxBlocks, (total + kThreads - 1) / kThreads));
}

} // namespace

cudaError_t launchIq2XxsGemmRepack(
    uint8_t const* source, int8_t* values, uint16_t* bases, int32_t N, int32_t kBlocks, cudaStream_t stream)
{
    if (source == nullptr || N <= 0 || kBlocks <= 0 || N % 8 != 0)
    {
        return cudaErrorInvalidValue;
    }
    if (values != nullptr)
    {
        int32_t const packing = N % 128 == 0 ? 128 : 8;
        gemmValuesKernel<<<gridFor(static_cast<int64_t>(N) * kBlocks * 64), kThreads, 0, stream>>>(
            source, values, N, kBlocks, packing);
    }
    if (bases != nullptr)
    {
        gemmBasesKernel<<<gridFor(static_cast<int64_t>(N) * kBlocks), kThreads, 0, stream>>>(source, bases, N, kBlocks);
    }
    return cudaGetLastError();
}

cudaError_t launchIq2XxsMoeRepackSourceBatch(uint8_t const* const* sources, int32_t count, int32_t* words,
    uint16_t* bases, int32_t N, int32_t kBlocks, cudaStream_t stream)
{
    if (sources == nullptr || count <= 0 || count > kCheckpointSourcesPerLaunch || N <= 0 || N % 128 != 0
        || kBlocks <= 0)
    {
        return cudaErrorInvalidValue;
    }
    auto const batch = makeCheckpointSourceBatch(sources, count);
    int64_t const perExpert = static_cast<int64_t>(N) * kBlocks * 16;
    if (words != nullptr)
    {
        moeWordsKernel<<<dim3(gridFor(perExpert), count), kThreads, 0, stream>>>(batch, words, N, kBlocks, perExpert);
    }
    if (bases != nullptr)
    {
        int64_t const basesPerExpert = static_cast<int64_t>(N) * kBlocks;
        moeBasesKernel<<<dim3(gridFor(basesPerExpert), count), kThreads, 0, stream>>>(
            batch, bases, N, kBlocks, basesPerExpert);
    }
    return cudaGetLastError();
}

} // namespace kernel
} // namespace trt_edgellm
