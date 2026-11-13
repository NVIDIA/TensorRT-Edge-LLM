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

#pragma once

#include "safetyAttentionKernels.h"

#include <cmath>
#include <cuda_runtime.h>

namespace trt_edgellm
{
namespace safety
{
namespace rope_detail
{
constexpr uint32_t kVECTOR_SIZE{8};
constexpr uint32_t kTHREADS_PER_BLOCK{128};
constexpr uint64_t kNONFINITE_ERROR{(1ULL << 0) | (1ULL << 1)};

// Copied from ade55abc28de3c75ac3a96f4a1e4f8c7c8df3878, cpp/kernels/common/vectorizedTypes.cuh,
// DVec<half>. Safety adaptation: retain only the FP16 vector operations; omit other precision dependencies.
struct HalfVector
{
    uint4 data;

    __device__ __forceinline__ half& operator[](uint32_t idx)
    {
        return reinterpret_cast<half*>(&data)[idx];
    }

    __device__ __forceinline__ half const& operator[](uint32_t idx) const
    {
        return reinterpret_cast<half const*>(&data)[idx];
    }

    __device__ __forceinline__ void load(half const* ptr)
    {
        data = *reinterpret_cast<uint4 const*>(ptr);
    }

    __device__ __forceinline__ void store(half* ptr) const
    {
        *reinterpret_cast<uint4*>(ptr) = data;
    }
};

// Copied from ade55abc28de3c75ac3a96f4a1e4f8c7c8df3878, cpp/kernels/common/vectorizedTypes.cuh,
// DVec<float>. Safety adaptation: retain only coefficient loading and access; use explicit C++ casts.
struct FloatVector
{
    float4 data[2];

    __device__ __forceinline__ float const& operator[](uint32_t idx) const
    {
        return reinterpret_cast<float const*>(data)[idx];
    }

    __device__ __forceinline__ void load(float const* ptr)
    {
        data[0] = *reinterpret_cast<float4 const*>(ptr);
        data[1] = *reinterpret_cast<float4 const*>(ptr + 4);
    }
};

// Copied from ade55abc28de3c75ac3a96f4a1e4f8c7c8df3878, cpp/kernels/posEncoding/applyRopeWriteKV.cu,
// applyRope<half>. Safety adaptation: report nonfinite arithmetic and FP16 conversion through the graph error mask.
__device__ __forceinline__ half applyRope(half x, half y, float cos, float sin, bool isLeft, uint64_t* error)
{
    float const val
        = isLeft ? (__half2float(x) * cos - __half2float(y) * sin) : (__half2float(x) * cos + __half2float(y) * sin);
    half const result = __float2half(val);
    if (!isfinite(val) || !isfinite(__half2float(result)))
    {
        atomicOr(reinterpret_cast<unsigned long long*>(error), static_cast<unsigned long long>(kNONFINITE_ERROR));
    }
    return result;
}

// Copied from ade55abc28de3c75ac3a96f4a1e4f8c7c8df3878, cpp/kernels/posEncoding/applyRopeWriteKV.cu,
// vecApplyRopeNonInterleave<half>. Safety adaptations: FP16-only vectors, bounded scalar rotation when a rotary
// half is not eight-element aligned, and nonfinite reporting. Aligned rotations preserve vector loads and math.
__device__ __forceinline__ HalfVector vecApplyRopeNonInterleave(
    half const* dataPtr, float const* coefficients, uint32_t rotaryDim, uint64_t* error)
{
    HalfVector input;
    uint32_t const vecOffset = threadIdx.x * kVECTOR_SIZE;
    input.load(dataPtr + vecOffset);
    if (vecOffset >= rotaryDim)
    {
        return input;
    }

    uint32_t const halfRotary = rotaryDim / 2;
    HalfVector result;
    if (halfRotary % kVECTOR_SIZE == 0)
    {
        // Non-interleaved RoPE pairs component j with j + rotaryDim/2 across the two rotary halves.
        uint32_t const permuteOffset = vecOffset < halfRotary ? vecOffset + halfRotary : vecOffset - halfRotary;
        HalfVector permuteInput;
        permuteInput.load(dataPtr + permuteOffset);
        // Each coefficient row stores [cos[halfRotary], sin[halfRotary]]; paired components share a frequency.
        uint32_t const cosOffset = vecOffset % halfRotary;
        FloatVector cosVec;
        FloatVector sinVec;
        cosVec.load(coefficients + cosOffset);
        sinVec.load(coefficients + cosOffset + halfRotary);
#pragma unroll
        for (uint32_t i = 0; i < kVECTOR_SIZE; ++i)
        {
            result[i] = applyRope(input[i], permuteInput[i], cosVec[i], sinVec[i], vecOffset < halfRotary, error);
        }
    }
    else
    {
#pragma unroll
        for (uint32_t i = 0; i < kVECTOR_SIZE; ++i)
        {
            uint32_t const component = vecOffset + i;
            if (component < rotaryDim)
            {
                bool const isLeft = component < halfRotary;
                uint32_t const partner = isLeft ? component + halfRotary : component - halfRotary;
                uint32_t const frequency = component % halfRotary;
                result[i] = applyRope(input[i], dataPtr[partner], coefficients[frequency],
                    coefficients[halfRotary + frequency], isLeft, error);
            }
            else
            {
                result[i] = input[i];
            }
        }
    }
    return result;
}

__device__ __forceinline__ void reportNonfinite(HalfVector const& vector, uint64_t* error)
{
    bool finite = true;
#pragma unroll
    for (uint32_t i = 0; i < kVECTOR_SIZE; ++i)
    {
        finite = finite && isfinite(__half2float(vector[i]));
    }
    if (!finite)
    {
        atomicOr(reinterpret_cast<unsigned long long*>(error), static_cast<unsigned long long>(kNONFINITE_ERROR));
    }
}
} // namespace rope_detail

//! Requires validated D64/D128; grid is (ceil(tokens / block.y), qHeads + kvHeads).
//! Packed QKV, query workspace, and cache bases must be 16-byte aligned.
inline dim3 safetyRopeBlock(AttentionLaunch const& shape) noexcept
{
    uint32_t const lanes = static_cast<uint32_t>(shape.headSize) / rope_detail::kVECTOR_SIZE;
    return dim3(lanes, rope_detail::kTHREADS_PER_BLOCK / lanes);
}

// Copied from ade55abc28de3c75ac3a96f4a1e4f8c7c8df3878, cpp/kernels/posEncoding/applyRopeWriteKV.cu,
// applyRopeFromPackedToSplitKernel<half, half, false>. Safety adaptations: batch-one validated start/length metadata;
// guard all data accesses with the validation flag; keep only FP16 contiguous-cache/Q-scratch outputs; remove QK norm,
// FP8, tree positions, PDL, and scratch K/V; report nonfinite data. Preserve eight-element vectorization;
// address the cache as [1, 2, KV heads, capacity, head size].
__global__ void safetyApplyRopeFromPackedToSplit(AttentionLaunch shape, half const* packed, int32_t const* lengths,
    int32_t const* starts, float const* rope, half* queries, half* cache, int32_t const* valid, uint64_t* error)
{
    using namespace rope_detail;
    // block.x lanes cover eight head components each; block.y rows cover separate tokens.
    uint32_t const tokenIdx = blockIdx.x * blockDim.y + threadIdx.y;
    if (*valid == 0)
    {
        return;
    }
    if (tokenIdx >= static_cast<uint32_t>(shape.tokens) || tokenIdx >= static_cast<uint32_t>(lengths[0]))
    {
        return;
    }

    int32_t const position = (starts == nullptr ? 0 : starts[0]) + static_cast<int32_t>(tokenIdx);
    // Packed input is [token, Q heads | K heads | V heads, head component]. Offsets count FP16 elements.
    int64_t const combinedHeads = static_cast<int64_t>(shape.qHeads) + 2 * static_cast<int64_t>(shape.kvHeads);
    int64_t const tokenBaseInPacked = static_cast<int64_t>(tokenIdx) * combinedHeads * shape.headSize;
    uint32_t const headIdx = blockIdx.y;
    uint32_t const vecBase = kVECTOR_SIZE * threadIdx.x;
    float const* const coefficients = rope + static_cast<int64_t>(position) * shape.rotaryDim;

    if (headIdx < static_cast<uint32_t>(shape.qHeads))
    {
        int64_t const packedQOffset = tokenBaseInPacked + static_cast<int64_t>(headIdx) * shape.headSize;
        // Q scratch omits the K/V head groups but retains [token, Q head, component] order.
        int64_t const scratchQOffset = (static_cast<int64_t>(tokenIdx) * shape.qHeads + headIdx) * shape.headSize;
        HalfVector const qRoped
            = vecApplyRopeNonInterleave(packed + packedQOffset, coefficients, shape.rotaryDim, error);
        reportNonfinite(qRoped, error);
        qRoped.store(queries + scratchQOffset + vecBase);
    }
    else
    {
        uint32_t const kvHeadIdx = headIdx - shape.qHeads;
        int64_t const packedKOffset
            = tokenBaseInPacked + static_cast<int64_t>(shape.qHeads + kvHeadIdx) * shape.headSize;
        int64_t const packedVOffset
            = tokenBaseInPacked + static_cast<int64_t>(shape.qHeads + shape.kvHeads + kvHeadIdx) * shape.headSize;
        HalfVector const kRoped
            = vecApplyRopeNonInterleave(packed + packedKOffset, coefficients, shape.rotaryDim, error);
        HalfVector vSrc;
        vSrc.load(packed + packedVOffset + vecBase);
        reportNonfinite(kRoped, error);
        reportNonfinite(vSrc, error);
        // Each K/V plane is head-major: [KV head, cache position, head component].
        int64_t const kOffset
            = (static_cast<int64_t>(kvHeadIdx) * shape.capacity + position) * shape.headSize + vecBase;
        int64_t const vOffset = kOffset + static_cast<int64_t>(shape.kvHeads) * shape.capacity * shape.headSize;
        kRoped.store(cache + kOffset);
        vSrc.store(cache + vOffset);
    }
}

} // namespace safety
} // namespace trt_edgellm
