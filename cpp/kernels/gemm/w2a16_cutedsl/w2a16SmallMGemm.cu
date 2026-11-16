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

#include "w2a16SmallMGemm.h"

#if defined(CUTE_DSL_W2A16_GEMM_ENABLED)

#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace trt_edgellm
{
namespace kernel
{
namespace
{

constexpr int32_t kCHANNELS_PER_GROUP = 16;
constexpr int32_t kWARPS = 8;
constexpr int32_t kTHREADS = kWARPS * 32;
constexpr int32_t kQUANT_BLOCK = 256;
constexpr int32_t kRECORD_BYTES = 64;
constexpr int32_t kSUB_BLOCKS = 8;
constexpr int32_t kTOKENS = 8;
constexpr int32_t kLOOKUP_ENTRIES = 256;
// Magnitude-table copies, interleaved per entry and selected by grid lane, so the 16-byte gathers of an 8-lane
// shared-memory phase spread over more banks.
constexpr int32_t kLOOKUP_COPIES = 4;
constexpr int32_t kSTAGES = 3;

//! A CTA's eight warps cover four 16-channel tiles times two K groups; warp (channel group, K group) takes every
//! second 256-weight block of its tile. Measured fastest on SM120 for the Bolt shared-expert shapes.
constexpr int32_t kCHANNEL_GROUPS = 4;
constexpr int32_t kK_GROUPS = kWARPS / kCHANNEL_GROUPS;
constexpr int32_t kCHANNELS = kCHANNEL_GROUPS * kCHANNELS_PER_GROUP;

//! One pipeline stage: the packed records and bases of kK_GROUPS consecutive 256-weight blocks.
struct alignas(16) SmallMStage
{
    uint4 records[kK_GROUPS][kCHANNELS][kRECORD_BYTES / 16];
    __half bases[kK_GROUPS][kCHANNELS];
};

union alignas(16) SmallMScratch
{
    uint4 magnitudes[kLOOKUP_ENTRIES * kLOOKUP_COPIES];
    float partial[kK_GROUPS][kTOKENS][kCHANNELS];
};

__device__ __forceinline__ void copyAsync16(void* shared, void const* global, bool valid)
{
    uint32_t const address = static_cast<uint32_t>(__cvta_generic_to_shared(shared));
    int32_t const bytes = valid ? 16 : 0;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(address), "l"(global), "r"(bytes));
}

__device__ __forceinline__ void commitAsync()
{
    asm volatile("cp.async.commit_group;");
}

template <int32_t kPENDING>
__device__ __forceinline__ void waitAsync()
{
    asm volatile("cp.async.wait_group %0;" ::"n"(kPENDING));
}

__device__ __forceinline__ void mmaBf16(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
    uint32_t b1, float c0, float c1, float c2, float c3)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
        "{%10, %11, %12, %13};"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "f"(c0), "f"(c1), "f"(c2), "f"(c3));
#endif
}

//! Signed magnitudes of one grid entry: seven stored sign bits plus an even-parity eighth; word w holds elements 2w
//! (bits 0-15) and 2w+1 (bits 16-31).
__device__ __forceinline__ void signedMagnitudes(uint4 const& magnitude, uint32_t signs7, uint32_t (&w)[4])
{
    uint32_t const signs = signs7 | ((__popc(signs7) & 1U) << 7);
    // Each multiplier moves sign bits 2w and 2w+1 to bits 15 and 31 without overlapping partial products.
    w[0] = magnitude.x ^ ((signs * 0x40008000U) & 0x80008000U);
    w[1] = magnitude.y ^ ((signs * 0x10002000U) & 0x80008000U);
    w[2] = magnitude.z ^ ((signs * 0x04000800U) & 0x80008000U);
    w[3] = magnitude.w ^ ((signs * 0x01000200U) & 0x80008000U);
}

__device__ __forceinline__ void loadStage(SmallMStage& stage, uint8_t const* values, __half const* bases,
    int32_t firstBlock, int32_t channel0, int32_t n, int32_t numKBlocks, int32_t packing, int32_t tid)
{
    constexpr int32_t kPARTS = kRECORD_BYTES / 16;
    for (int32_t i = tid; i < kK_GROUPS * kCHANNELS * kPARTS; i += kTHREADS)
    {
        int32_t const part = i % kPARTS;
        int32_t const local = i / kPARTS % kCHANNELS;
        int32_t const group = i / (kPARTS * kCHANNELS);
        int32_t const block = firstBlock + group;
        int32_t const channel = channel0 + local;
        bool const valid = channel < n && block < numKBlocks;
        size_t const offset = valid
            ? ((static_cast<size_t>(channel / packing) * numKBlocks + block) * packing + channel % packing)
                * kRECORD_BYTES
            : 0;
        copyAsync16(&stage.records[group][local][part], values + offset + 16 * part, valid);
    }
    if (tid < kK_GROUPS * kCHANNELS / 8)
    {
        int32_t const group = tid / (kCHANNELS / 8);
        int32_t const local = tid % (kCHANNELS / 8) * 8;
        int32_t const block = firstBlock + group;
        bool const valid = channel0 + local < n && block < numKBlocks;
        copyAsync16(
            &stage.bases[group][local], bases + (valid ? static_cast<size_t>(block) * n + channel0 + local : 0), valid);
    }
}

//! The packed weights of a CTA's channel tiles stream through a cp.async pipeline; the K groups are reduced in shared
//! memory. Within a 32-weight sub-block, lane (g, c) holds grid entry c of channels g and g + 8 and the matching eight
//! activations of token g, so the K order inside each mma is permuted identically for both operands. Each mma pair
//! covers one sub-block, whose FP32 result is then scaled by that sub-block's scale.
__global__ void __launch_bounds__(kTHREADS) w2a16SmallMGemmKernel(__nv_bfloat16 const* __restrict__ input,
    uint8_t const* __restrict__ values, __half const* __restrict__ bases, uint4 const* __restrict__ lookup,
    __nv_bfloat16* __restrict__ output, int32_t m, int32_t n, int32_t k)
{
    __shared__ SmallMStage stages[kSTAGES];
    __shared__ SmallMScratch scratch;

    int32_t const tid = static_cast<int32_t>(threadIdx.x);
    int32_t const channel0 = static_cast<int32_t>(blockIdx.x) * kCHANNELS;
    int32_t const numKBlocks = k / kQUANT_BLOCK;
    int32_t const stageCount = (numKBlocks + kK_GROUPS - 1) / kK_GROUPS;
    int32_t const packing = n % 128 == 0 ? 128 : 8;

#pragma unroll
    for (int32_t s = 0; s < kSTAGES - 1; ++s)
    {
        if (s < stageCount)
        {
            loadStage(stages[s], values, bases, s * kK_GROUPS, channel0, n, numKBlocks, packing, tid);
        }
        commitAsync();
    }
    for (int32_t i = tid; i < kLOOKUP_ENTRIES * kLOOKUP_COPIES; i += kTHREADS)
    {
        scratch.magnitudes[i] = lookup[i / kLOOKUP_COPIES];
    }

    int32_t const warp = tid / 32;
    int32_t const lane = tid % 32;
    int32_t const channelGroup = warp % kCHANNEL_GROUPS;
    int32_t const groupK = warp / kCHANNEL_GROUPS;
    int32_t const g = lane / 4;
    int32_t const c = lane % 4;
    int32_t const copy = c % kLOOKUP_COPIES;
    int32_t const rows[2] = {channelGroup * kCHANNELS_PER_GROUP + g, channelGroup * kCHANNELS_PER_GROUP + g + 8};

    float acc[4] = {0.F, 0.F, 0.F, 0.F};

    for (int32_t s = 0; s < stageCount; ++s)
    {
        int32_t const block = s * kK_GROUPS + groupK;
        uint4 activations[kSUB_BLOCKS];
        bool const validRow = g < m && block < numKBlocks;
        uint4 const* row = reinterpret_cast<uint4 const*>(
            input + (validRow ? static_cast<size_t>(g) * k + block * kQUANT_BLOCK + c * 8 : 0));
#pragma unroll
        for (int32_t q = 0; q < kSUB_BLOCKS; ++q)
        {
            activations[q] = validRow ? __ldg(row + q * 4) : make_uint4(0, 0, 0, 0);
        }
        waitAsync<kSTAGES - 2>();
        __syncthreads();
        int32_t const prefetch = s + kSTAGES - 1;
        if (prefetch < stageCount)
        {
            loadStage(
                stages[prefetch % kSTAGES], values, bases, prefetch * kK_GROUPS, channel0, n, numKBlocks, packing, tid);
        }
        commitAsync();
        if (block >= numKBlocks)
        {
            continue;
        }

        auto const& stage = stages[s % kSTAGES];
        float baseScale[2];
#pragma unroll
        for (int32_t r = 0; r < 2; ++r)
        {
            baseScale[r] = __half2float(stage.bases[groupK][rows[r]]) * 0.25F;
        }
#pragma unroll
        for (int32_t q = 0; q < kSUB_BLOCKS; ++q)
        {
            uint32_t w[2][4];
            float scale[2];
#pragma unroll
            for (int32_t r = 0; r < 2; ++r)
            {
                uint2 const record = reinterpret_cast<uint2 const*>(stage.records[groupK][rows[r]])[q];
                uint4 const magnitude = scratch.magnitudes[((record.x >> (8 * c)) & 0xFFU) * kLOOKUP_COPIES + copy];
                signedMagnitudes(magnitude, (record.y >> (7 * c)) & 0x7FU, w[r]);
                scale[r] = baseScale[r] * (static_cast<float>(record.y >> 28) + 0.5F);
            }
            uint4 const& x = activations[q];
            float d[4];
            mmaBf16(d, w[0][0], w[1][0], w[0][1], w[1][1], x.x, x.y, 0.F, 0.F, 0.F, 0.F);
            mmaBf16(d, w[0][2], w[1][2], w[0][3], w[1][3], x.z, x.w, d[0], d[1], d[2], d[3]);
            acc[0] = fmaf(scale[0], d[0], acc[0]);
            acc[1] = fmaf(scale[0], d[1], acc[1]);
            acc[2] = fmaf(scale[1], d[2], acc[2]);
            acc[3] = fmaf(scale[1], d[3], acc[3]);
        }
    }
    waitAsync<0>();
    __syncthreads();

    scratch.partial[groupK][2 * c][rows[0]] = acc[0];
    scratch.partial[groupK][2 * c + 1][rows[0]] = acc[1];
    scratch.partial[groupK][2 * c][rows[1]] = acc[2];
    scratch.partial[groupK][2 * c + 1][rows[1]] = acc[3];
    __syncthreads();
    for (int32_t i = tid; i < kTOKENS * kCHANNELS; i += kTHREADS)
    {
        int32_t const token = i / kCHANNELS;
        int32_t const local = i % kCHANNELS;
        int32_t const channel = channel0 + local;
        if (token < m && channel < n)
        {
            float sum = 0.F;
#pragma unroll
            for (int32_t group = 0; group < kK_GROUPS; ++group)
            {
                sum += scratch.partial[group][token][local];
            }
            output[static_cast<size_t>(token) * n + channel] = __float2bfloat16_rn(sum);
        }
    }
}

} // namespace

cudaError_t launchW2A16SmallMGemm(void const* input, void const* values, void const* bases, void const* lookup,
    void* output, int32_t m, int32_t n, int32_t k, cudaStream_t stream) noexcept
{
    if (m < 1 || m > kW2A16_SMALL_M_MAX_TOKENS || n <= 0 || n % 8 != 0 || k <= 0 || k % kQUANT_BLOCK != 0)
    {
        return cudaErrorInvalidValue;
    }
    dim3 const grid((n + kCHANNELS - 1) / kCHANNELS);
    w2a16SmallMGemmKernel<<<grid, kTHREADS, 0, stream>>>(static_cast<__nv_bfloat16 const*>(input),
        static_cast<uint8_t const*>(values), static_cast<__half const*>(bases), static_cast<uint4 const*>(lookup),
        static_cast<__nv_bfloat16*>(output), m, n, k);
    return cudaGetLastError();
}

} // namespace kernel
} // namespace trt_edgellm

#endif // defined(CUTE_DSL_W2A16_GEMM_ENABLED)
