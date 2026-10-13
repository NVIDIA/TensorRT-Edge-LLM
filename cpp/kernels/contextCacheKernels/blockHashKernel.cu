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

#include "kernels/contextCacheKernels/blockHashKernel.h"

#include <cstdint>

namespace trt_edgellm
{
namespace rt
{
namespace
{

// FNV-1a 128-bit constants — identical to blockHash.cpp.
// CUDA lacks __uint128_t on device; emulate with hi/lo uint64_t pair.
struct Fnv128
{
    uint64_t hi;
    uint64_t lo;
};

// FNV-1a 128-bit prime: 2^88 + 2^8 + 0x3B
__device__ constexpr Fnv128 kPRIME{0x0000000001000000ULL, 0x000000000000013BULL};
__device__ constexpr Fnv128 kOFFSET{0x6C62272E07BB0142ULL, 0x62B821756295C58DULL};

__device__ __forceinline__ Fnv128 fnvMultiply(Fnv128 state, Fnv128 prime)
{
    // 128-bit multiply mod 2^128: only keep the low 128 bits.
    unsigned long long lo_lo = state.lo * prime.lo;
    unsigned long long hi_part = __umul64hi(state.lo, prime.lo);
    unsigned long long result_hi = state.hi * prime.lo + state.lo * prime.hi + hi_part;
    return Fnv128{result_hi, lo_lo};
}

__device__ __forceinline__ Fnv128 fnvByte(Fnv128 state, uint8_t value)
{
    state.lo ^= static_cast<uint64_t>(value);
    return fnvMultiply(state, kPRIME);
}

// Process 16 bytes loaded as a uint4 through FNV-1a in little-endian byte order.
__device__ __forceinline__ Fnv128 fnvUint4(Fnv128 state, uint4 v)
{
    // uint4 = {x, y, z, w} each uint32_t, process in memory order (LE bytes within each word).
    state = fnvByte(state, static_cast<uint8_t>(v.x));
    state = fnvByte(state, static_cast<uint8_t>(v.x >> 8));
    state = fnvByte(state, static_cast<uint8_t>(v.x >> 16));
    state = fnvByte(state, static_cast<uint8_t>(v.x >> 24));
    state = fnvByte(state, static_cast<uint8_t>(v.y));
    state = fnvByte(state, static_cast<uint8_t>(v.y >> 8));
    state = fnvByte(state, static_cast<uint8_t>(v.y >> 16));
    state = fnvByte(state, static_cast<uint8_t>(v.y >> 24));
    state = fnvByte(state, static_cast<uint8_t>(v.z));
    state = fnvByte(state, static_cast<uint8_t>(v.z >> 8));
    state = fnvByte(state, static_cast<uint8_t>(v.z >> 16));
    state = fnvByte(state, static_cast<uint8_t>(v.z >> 24));
    state = fnvByte(state, static_cast<uint8_t>(v.w));
    state = fnvByte(state, static_cast<uint8_t>(v.w >> 8));
    state = fnvByte(state, static_cast<uint8_t>(v.w >> 16));
    state = fnvByte(state, static_cast<uint8_t>(v.w >> 24));
    return state;
}

// Multi-CTA Phase 1 kernel: each thread hashes one chunk, writes partial to global output.
__global__ void fnv1aChunksKernel(
    uint8_t const* __restrict__ data, size_t totalSize, int32_t numChunks, uint64_t* partialsOut)
{
    int32_t const chunkIdx = static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (chunkIdx >= numChunks)
    {
        return;
    }

    size_t const chunkSize = (totalSize + static_cast<size_t>(numChunks) - 1) / static_cast<size_t>(numChunks);
    size_t const chunkStart = static_cast<size_t>(chunkIdx) * chunkSize;
    size_t const chunkEnd = min(chunkStart + chunkSize, totalSize);

    Fnv128 partial = kOFFSET;
    size_t pos = chunkStart;

    // Vectorized bulk: read 16 bytes at a time via uint4.
    size_t const alignedStart = (chunkStart + 15ULL) & ~15ULL;
    size_t const alignedEnd = chunkEnd & ~15ULL;

    if (alignedStart < alignedEnd)
    {
        for (; pos < alignedStart; ++pos)
        {
            partial = fnvByte(partial, data[pos]);
        }

        uint4 const* vecPtr = reinterpret_cast<uint4 const*>(data + alignedStart);
        size_t const numVecIters = (alignedEnd - alignedStart) >> 4;
        for (size_t v = 0; v < numVecIters; ++v)
        {
            partial = fnvUint4(partial, vecPtr[v]);
        }
        pos = alignedEnd;
    }

    for (; pos < chunkEnd; ++pos)
    {
        partial = fnvByte(partial, data[pos]);
    }

    partialsOut[chunkIdx * 2] = partial.hi;
    partialsOut[chunkIdx * 2 + 1] = partial.lo;
}

} // namespace

void launchFnv1aHashChunksKernel(uint8_t const* deviceData, size_t totalSize, int32_t numChunks, int32_t numCtas,
    uint64_t* partialsOut, cudaStream_t stream)
{
    int32_t const threadsPerCta = (numChunks + numCtas - 1) / numCtas;
    fnv1aChunksKernel<<<numCtas, threadsPerCta, 0, stream>>>(deviceData, totalSize, numChunks, partialsOut);
}

} // namespace rt
} // namespace trt_edgellm
