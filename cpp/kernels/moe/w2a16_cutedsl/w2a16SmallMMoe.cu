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

#include "w2a16SmallMMoe.h"

// Thor runs every token count on the UMMA kernels; only the fused AOT targets use the small-M kernels.
#if defined(CUTE_DSL_W2A16_MOE_ENABLED) && defined(CUTE_DSL_W2A16_MOE_FUSED_ENABLED)

#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace trt_edgellm
{
namespace kernel
{
namespace
{

constexpr int32_t kHIDDEN = 1024;
constexpr int32_t kTOP_K = 9;
constexpr int32_t kTOKENS_PER_TILE = 8;
constexpr int32_t kMAX_TOKENS = 16;
constexpr int32_t kCHANNEL_GROUPS = 4;
constexpr int32_t kK_GROUPS = 2;
constexpr int32_t kCHANNELS_PER_CTA = kCHANNEL_GROUPS * 16;
constexpr int32_t kTHREADS = kCHANNEL_GROUPS * kK_GROUPS * 32;
constexpr int32_t kQUANT_BLOCK = 256;
constexpr int32_t kLOOKUP_ENTRIES = 256;
// Magnitude-table copies, interleaved per entry and selected by grid lane, so the 16-byte gathers of an 8-lane
// shared-memory phase spread over more banks.
constexpr int32_t kLOOKUP_COPIES = 4;
constexpr size_t kALIGNMENT = 256;
// Channel tiles per CTA: one token touches 9 experts and keeps more CTAs; more tokens amortize each CTA's table and
// its route scan over more channels.
constexpr int32_t kSINGLE_TOKEN_TILES_PER_CTA = 2;
constexpr int32_t kTILES_PER_CTA = 4;
constexpr int32_t kTWO_TILE_TILES_PER_CTA = 16;

size_t alignUp(size_t value) noexcept
{
    return (value + kALIGNMENT - 1) / kALIGNMENT * kALIGNMENT;
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

__device__ __forceinline__ void signedMagnitudes(uint4 const& magnitude, uint32_t signs7, uint32_t (&w)[4])
{
    uint32_t const signs = signs7 | ((__popc(signs7) & 1U) << 7);
    // Each multiplier moves sign bits 2w and 2w+1 to bits 15 and 31 without overlapping partial products.
    w[0] = magnitude.x ^ ((signs * 0x40008000U) & 0x80008000U);
    w[1] = magnitude.y ^ ((signs * 0x10002000U) & 0x80008000U);
    w[2] = magnitude.z ^ ((signs * 0x04000800U) & 0x80008000U);
    w[3] = magnitude.w ^ ((signs * 0x01000200U) & 0x80008000U);
}

//! BF16 grid magnitudes from the packed layout's pair selectors: entry e holds four 16-bit selectors whose low and
//! high bytes store 2 * code (code 0, 1, 2 -> 8, 25, 43) for elements 2p and 2p + 1.
__device__ __forceinline__ void buildMagnitudes(uint4* magnitudes, uint16_t const* selectors, int32_t tid)
{
    constexpr uint32_t kVALUES[3] = {0x4100U, 0x41C8U, 0x422CU};
    for (int32_t i = tid; i < kLOOKUP_ENTRIES; i += kTHREADS)
    {
        uint32_t words[4];
#pragma unroll
        for (int32_t p = 0; p < 4; ++p)
        {
            uint32_t const selector = selectors[i * 4 + p];
            words[p] = kVALUES[(selector & 0xFU) >> 1] | (kVALUES[((selector >> 8) & 0xFU) >> 1] << 16);
        }
        uint4 const value = make_uint4(words[0], words[1], words[2], words[3]);
#pragma unroll
        for (int32_t c = 0; c < kLOOKUP_COPIES; ++c)
        {
            magnitudes[i * kLOOKUP_COPIES + c] = value;
        }
    }
}

struct alignas(16) RouteTable
{
    int32_t count;
    int32_t routes[kMAX_TOKENS]; //!< token * 9 + slot of every token routed to this CTA's expert, in token order.
};

//! The CTA of route slot `route` owns its expert when no earlier slot has the same expert; it then lists the routes
//! of every token that selected that expert.
__device__ __forceinline__ bool claimExpert(
    RouteTable& table, int32_t const* topkIds, int32_t route, int32_t m, int32_t numExperts, int32_t& expert)
{
    expert = topkIds[route];
    if (expert < 0 || expert >= numExperts)
    {
        return false;
    }
    for (int32_t r = 0; r < route; ++r)
    {
        if (topkIds[r] == expert)
        {
            return false;
        }
    }
    if (threadIdx.x == 0)
    {
        int32_t count = 0;
        for (int32_t r = route; r < m * kTOP_K && count < kMAX_TOKENS; ++r)
        {
            if (topkIds[r] == expert)
            {
                table.routes[count++] = r;
            }
        }
        table.count = count;
    }
    return true;
}

//! Contracts this CTA's 64 output channels of `expert` against up to 8 * kTOKEN_TILES activation rows. The packed
//! payload is int32 [E, N/64, K/128, 8 words, 4 row groups, 8, 2] with row = 64 * tile + 16 * group + 8 * j + i, so the
//! two rows (i, j = 0/1) a lane needs are adjacent words; bases are FP16 [E, K/256, N/16, 8, 2] in the same row order.
//! Within a 32-weight sub-block, lane (g, c) holds grid entry c of rows g and g + 8 and the matching eight activations
//! of row g, so the K order inside each mma is permuted identically for both operands; each mma pair covers one
//! sub-block and its FP32 result is scaled by that sub-block's scale. Returns this warp's [16 x 8] partial per tile.
template <int32_t kTOKEN_TILES>
__device__ __forceinline__ void expertTile(float (&acc)[kTOKEN_TILES][4], int32_t const* __restrict__ payload,
    __half const* __restrict__ bases, uint4 const* magnitudes, __nv_bfloat16 const* const (&rows)[kMAX_TOKENS],
    int32_t expert, int32_t n, int32_t k, int32_t tile, int32_t lane, int32_t channelGroup, int32_t groupK)
{
    int32_t const g = lane / 4;
    int32_t const c = lane % 4;
    int32_t const copy = c % kLOOKUP_COPIES;
    int32_t const numKTiles = k / 128;
    int32_t const numKBlocks = k / kQUANT_BLOCK;
#pragma unroll
    for (int32_t t = 0; t < kTOKEN_TILES; ++t)
    {
        acc[t][0] = acc[t][1] = acc[t][2] = acc[t][3] = 0.F;
    }
    for (int32_t block = groupK; block < numKBlocks; block += kK_GROUPS)
    {
        uint2 grids[8], signs[8];
#pragma unroll
        for (int32_t half = 0; half < 2; ++half)
        {
            int32_t const* tileWords = payload
                + (((static_cast<size_t>(expert) * (n / 64) + tile) * numKTiles + block * 2 + half) * 8) * 64
                + channelGroup * 16 + g * 2;
#pragma unroll
            for (int32_t s = 0; s < 4; ++s)
            {
                grids[half * 4 + s] = __ldg(reinterpret_cast<uint2 const*>(tileWords + (2 * s) * 64));
                signs[half * 4 + s] = __ldg(reinterpret_cast<uint2 const*>(tileWords + (2 * s + 1) * 64));
            }
        }
        __half2 const base = __ldg(reinterpret_cast<__half2 const*>(bases
            + ((static_cast<size_t>(expert) * numKBlocks + block) * (n / 16) + tile * 4 + channelGroup) * 16 + g * 2));
        float const baseScale[2] = {__low2float(base) * 0.25F, __high2float(base) * 0.25F};
        uint4 activations[kTOKEN_TILES][8];
#pragma unroll
        for (int32_t t = 0; t < kTOKEN_TILES; ++t)
        {
            __nv_bfloat16 const* const row = rows[t * kTOKENS_PER_TILE + g];
#pragma unroll
            for (int32_t q = 0; q < 8; ++q)
            {
                activations[t][q] = row != nullptr
                    ? __ldg(reinterpret_cast<uint4 const*>(row + block * kQUANT_BLOCK + q * 32 + c * 8))
                    : make_uint4(0, 0, 0, 0);
            }
        }
#pragma unroll
        for (int32_t q = 0; q < 8; ++q)
        {
            uint32_t w[2][4];
            float scale[2];
            uint32_t const rowGrids[2] = {grids[q].x, grids[q].y};
            uint32_t const rowSigns[2] = {signs[q].x, signs[q].y};
#pragma unroll
            for (int32_t r = 0; r < 2; ++r)
            {
                uint4 const magnitude = magnitudes[((rowGrids[r] >> (8 * c)) & 0xFFU) * kLOOKUP_COPIES + copy];
                signedMagnitudes(magnitude, (rowSigns[r] >> (7 * c)) & 0x7FU, w[r]);
                scale[r] = baseScale[r] * (static_cast<float>(rowSigns[r] >> 28) + 0.5F);
            }
#pragma unroll
            for (int32_t t = 0; t < kTOKEN_TILES; ++t)
            {
                uint4 const& x = activations[t][q];
                float d[4];
                mmaBf16(d, w[0][0], w[1][0], w[0][1], w[1][1], x.x, x.y, 0.F, 0.F, 0.F, 0.F);
                mmaBf16(d, w[0][2], w[1][2], w[0][3], w[1][3], x.z, x.w, d[0], d[1], d[2], d[3]);
                acc[t][0] = fmaf(scale[0], d[0], acc[t][0]);
                acc[t][1] = fmaf(scale[0], d[1], acc[t][1]);
                acc[t][2] = fmaf(scale[1], d[2], acc[t][2]);
                acc[t][3] = fmaf(scale[1], d[3], acc[t][3]);
            }
        }
    }
}

struct alignas(16) MoeShared
{
    uint4 magnitudes[kLOOKUP_ENTRIES * kLOOKUP_COPIES];
    float partial[kK_GROUPS][kMAX_TOKENS][kCHANNELS_PER_CTA];
    RouteTable table;
};

//! Writes the CTA's [tokens x 64 channels] sums (K groups added in order) through `store(q, channel, value)`.
template <int32_t kTOKEN_TILES, typename Store>
__device__ __forceinline__ void reduceTile(MoeShared& shared, float const (&acc)[kTOKEN_TILES][4], int32_t lane,
    int32_t channelGroup, int32_t groupK, Store store)
{
    int32_t const g = lane / 4;
    int32_t const c = lane % 4;
    int32_t const rows[2] = {channelGroup * 16 + g, channelGroup * 16 + g + 8};
#pragma unroll
    for (int32_t t = 0; t < kTOKEN_TILES; ++t)
    {
        int32_t const token = t * kTOKENS_PER_TILE + 2 * c;
        shared.partial[groupK][token][rows[0]] = acc[t][0];
        shared.partial[groupK][token + 1][rows[0]] = acc[t][1];
        shared.partial[groupK][token][rows[1]] = acc[t][2];
        shared.partial[groupK][token + 1][rows[1]] = acc[t][3];
    }
    __syncthreads();
    for (int32_t i = static_cast<int32_t>(threadIdx.x); i < shared.table.count * kCHANNELS_PER_CTA; i += kTHREADS)
    {
        int32_t const q = i / kCHANNELS_PER_CTA;
        int32_t const local = i % kCHANNELS_PER_CTA;
        float sum = 0.F;
#pragma unroll
        for (int32_t group = 0; group < kK_GROUPS; ++group)
        {
            sum += shared.partial[group][q][local];
        }
        store(q, local, sum);
    }
    __syncthreads();
}

//! FC1 + ReLU2 for one expert and 64 intermediate channels: act[route] = bf16(relu(bf16(x @ W1^T))^2).
template <int32_t kTOKEN_TILES>
__global__ void __launch_bounds__(kTHREADS)
    smallMMoeFc1Kernel(W2A16SmallMMoeParams const params, __nv_bfloat16* __restrict__ activated, int32_t tilesPerCta)
{
    __shared__ MoeShared shared;
    int32_t expert{};
    if (!claimExpert(shared.table, params.topkIds, static_cast<int32_t>(blockIdx.y), params.numTokens,
            params.numExperts, expert))
    {
        return;
    }
    int32_t const tid = static_cast<int32_t>(threadIdx.x);
    buildMagnitudes(shared.magnitudes, static_cast<uint16_t const*>(params.selectorTable), tid);
    __syncthreads();
    int32_t const warp = tid / 32;
    int32_t const lane = tid % 32;
    auto const* hidden = static_cast<__nv_bfloat16 const*>(params.hiddenStates);
    __nv_bfloat16 const* rows[kMAX_TOKENS];
#pragma unroll
    for (int32_t q = 0; q < kMAX_TOKENS; ++q)
    {
        rows[q] = q < shared.table.count ? hidden + static_cast<size_t>(shared.table.routes[q] / kTOP_K) * kHIDDEN
                                         : nullptr;
    }
    float const expertScale = params.expertScales[expert];
    int32_t const inter = params.moeInterSize;
    int32_t const tiles = inter / kCHANNELS_PER_CTA;
    for (int32_t tile = static_cast<int32_t>(blockIdx.x) * tilesPerCta;
        tile < tiles && tile < (static_cast<int32_t>(blockIdx.x) + 1) * tilesPerCta; ++tile)
    {
        float acc[kTOKEN_TILES][4];
        expertTile<kTOKEN_TILES>(acc, static_cast<int32_t const*>(params.fc1Weights),
            static_cast<__half const*>(params.fc1Bases), shared.magnitudes, rows, expert, inter, kHIDDEN, tile, lane,
            warp % kCHANNEL_GROUPS, warp / kCHANNEL_GROUPS);
        reduceTile(shared, acc, lane, warp % kCHANNEL_GROUPS, warp / kCHANNEL_GROUPS,
            [&](int32_t q, int32_t local, float sum) {
                float const projected = __bfloat162float(__float2bfloat16_rn(sum * expertScale));
                float const relu = fmaxf(projected, 0.F);
                activated[static_cast<size_t>(shared.table.routes[q]) * inter + tile * kCHANNELS_PER_CTA + local]
                    = __float2bfloat16_rn(relu * relu);
            });
    }
}

//! FC2 for one expert and 64 hidden channels: contribution[route] = bf16(act[route] @ W2^T * topk weight).
template <int32_t kTOKEN_TILES>
__global__ void __launch_bounds__(kTHREADS) smallMMoeFc2Kernel(W2A16SmallMMoeParams const params,
    __nv_bfloat16 const* __restrict__ activated, __nv_bfloat16* __restrict__ contributions, int32_t tilesPerCta)
{
    __shared__ MoeShared shared;
    int32_t expert{};
    if (!claimExpert(shared.table, params.topkIds, static_cast<int32_t>(blockIdx.y), params.numTokens,
            params.numExperts, expert))
    {
        return;
    }
    int32_t const tid = static_cast<int32_t>(threadIdx.x);
    buildMagnitudes(shared.magnitudes, static_cast<uint16_t const*>(params.selectorTable), tid);
    __syncthreads();
    int32_t const warp = tid / 32;
    int32_t const lane = tid % 32;
    int32_t const inter = params.moeInterSize;
    __nv_bfloat16 const* rows[kMAX_TOKENS];
#pragma unroll
    for (int32_t q = 0; q < kMAX_TOKENS; ++q)
    {
        rows[q] = q < shared.table.count ? activated + static_cast<size_t>(shared.table.routes[q]) * inter : nullptr;
    }
    float const expertScale = params.expertScales[expert];
    constexpr int32_t kTILES = kHIDDEN / kCHANNELS_PER_CTA;
    for (int32_t tile = static_cast<int32_t>(blockIdx.x) * tilesPerCta;
        tile < kTILES && tile < (static_cast<int32_t>(blockIdx.x) + 1) * tilesPerCta; ++tile)
    {
        float acc[kTOKEN_TILES][4];
        expertTile<kTOKEN_TILES>(acc, static_cast<int32_t const*>(params.fc2Weights),
            static_cast<__half const*>(params.fc2Bases), shared.magnitudes, rows, expert, kHIDDEN, inter, tile, lane,
            warp % kCHANNEL_GROUPS, warp / kCHANNEL_GROUPS);
        reduceTile(shared, acc, lane, warp % kCHANNEL_GROUPS, warp / kCHANNEL_GROUPS,
            [&](int32_t q, int32_t local, float sum) {
                int32_t const route = shared.table.routes[q];
                contributions[static_cast<size_t>(route) * kHIDDEN + tile * kCHANNELS_PER_CTA + local]
                    = __float2bfloat16_rn(sum * expertScale * params.topkWeights[route]);
            });
    }
}

//! output[token] = bf16(sum over the nine slots, in slot order, of the FP32 value of each BF16 contribution).
__global__ void smallMMoeSumKernel(W2A16SmallMMoeParams const params, __nv_bfloat16 const* __restrict__ contributions)
{
    int32_t const token = static_cast<int32_t>(blockIdx.x);
    auto* output = static_cast<__nv_bfloat16*>(params.output);
    for (int32_t ch = static_cast<int32_t>(threadIdx.x); ch < kHIDDEN; ch += static_cast<int32_t>(blockDim.x))
    {
        float sum = 0.F;
        for (int32_t slot = 0; slot < kTOP_K; ++slot)
        {
            int32_t const route = token * kTOP_K + slot;
            int32_t const expert = params.topkIds[route];
            if (expert >= 0 && expert < params.numExperts)
            {
                sum += __bfloat162float(contributions[static_cast<size_t>(route) * kHIDDEN + ch]);
            }
        }
        output[static_cast<size_t>(token) * kHIDDEN + ch] = __float2bfloat16_rn(sum);
    }
}

} // namespace

size_t w2a16SmallMMoeWorkspaceSize(int32_t numTokens, int32_t moeInterSize) noexcept
{
    size_t const routes = static_cast<size_t>(numTokens) * kTOP_K;
    return alignUp(routes * moeInterSize * sizeof(__nv_bfloat16)) + alignUp(routes * kHIDDEN * sizeof(__nv_bfloat16));
}

cudaError_t launchW2A16SmallMMoe(W2A16SmallMMoeParams const& params, void* workspace, cudaStream_t stream) noexcept
{
    if (params.numTokens < 1 || params.numTokens > kW2A16_SMALL_M_MOE_MAX_TOKENS || params.numExperts <= 0
        || params.moeInterSize <= 0 || params.moeInterSize % kQUANT_BLOCK != 0 || workspace == nullptr)
    {
        return cudaErrorInvalidValue;
    }
    size_t const routes = static_cast<size_t>(params.numTokens) * kTOP_K;
    auto* activated = static_cast<__nv_bfloat16*>(workspace);
    auto* contributions = reinterpret_cast<__nv_bfloat16*>(
        static_cast<char*>(workspace) + alignUp(routes * params.moeInterSize * sizeof(__nv_bfloat16)));
    int32_t const tilesPerCta = params.numTokens == 1 ? kSINGLE_TOKEN_TILES_PER_CTA
        : params.numTokens <= kTOKENS_PER_TILE        ? kTILES_PER_CTA
                                                      : kTWO_TILE_TILES_PER_CTA;
    int32_t const fc1Tiles = params.moeInterSize / kCHANNELS_PER_CTA;
    dim3 const fc1Grid((fc1Tiles + tilesPerCta - 1) / tilesPerCta, static_cast<uint32_t>(routes));
    if (params.numTokens <= kTOKENS_PER_TILE)
    {
        smallMMoeFc1Kernel<1><<<fc1Grid, kTHREADS, 0, stream>>>(params, activated, tilesPerCta);
    }
    else
    {
        smallMMoeFc1Kernel<2><<<fc1Grid, kTHREADS, 0, stream>>>(params, activated, tilesPerCta);
    }
    int32_t const fc2Tiles = kHIDDEN / kCHANNELS_PER_CTA;
    dim3 const fc2Grid((fc2Tiles + tilesPerCta - 1) / tilesPerCta, static_cast<uint32_t>(routes));
    if (params.numTokens <= kTOKENS_PER_TILE)
    {
        smallMMoeFc2Kernel<1><<<fc2Grid, kTHREADS, 0, stream>>>(params, activated, contributions, tilesPerCta);
    }
    else
    {
        smallMMoeFc2Kernel<2><<<fc2Grid, kTHREADS, 0, stream>>>(params, activated, contributions, tilesPerCta);
    }
    smallMMoeSumKernel<<<params.numTokens, 256, 0, stream>>>(params, contributions);
    return cudaGetLastError();
}

} // namespace kernel
} // namespace trt_edgellm

#endif // defined(CUTE_DSL_W2A16_MOE_ENABLED) && defined(CUTE_DSL_W2A16_MOE_FUSED_ENABLED)
