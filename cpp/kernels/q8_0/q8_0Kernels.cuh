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

// Q8_0 x Q8_1 linear kernels. Q8_0 weights stay packed as checkpoint blocks [N, K/32] of {FP16 scale, 32 x int8}.
// Activations are FP16 or BF16; outputs are FP16, BF16 or FP32, so the graph needs no casts around the linear.
// One launch treats up to kMaxWeights weights of one activation (for example Q, K and V) as a single matrix whose
// output columns follow in weight order, so they share one quantization and fill one packed output.
// The Q8_1 activation quantization and the DP4A block dot product follow llama.cpp (MIT):
// https://github.com/ggml-org/llama.cpp/blob/7339054744f109c4cd89b75689dbb8a2c154d60e/ggml/src/ggml-cuda/quantize.cu#L54-L99
// https://github.com/ggml-org/llama.cpp/blob/7339054744f109c4cd89b75689dbb8a2c154d60e/ggml/src/ggml-cuda/vecdotq.cuh#L246-L259

#include <cstddef>
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace trt_edgellm::kernel::q8_0
{
constexpr int32_t kBlockSize = 32;
constexpr int32_t kWordsPerBlock = kBlockSize / 4;
constexpr int32_t kWarpSize = 32;
constexpr int32_t kWarpsPerCta = 4;
//! Activation rows fill the N=8 side of mma.m16n8k32.
constexpr int32_t kTokensPerTile = 8;
//! Weight channels fill the M=16 side of mma.m16n8k32.
constexpr int32_t kChannelsPerTile = 16;
constexpr int32_t kTileCodeBytes = kTokensPerTile * kBlockSize;
constexpr uint32_t kFullMask = 0xffffffffU;
constexpr int32_t kMaxWeights = 4;

struct alignas(2) PackedWeightBlock
{
    half scale;
    int8_t codes[kBlockSize];
};
static_assert(sizeof(PackedWeightBlock) == 34);

// Weight w holds output columns [columnEnd[w - 1], columnEnd[w]), starting at column 0 for w = 0.
struct WeightSet
{
    PackedWeightBlock const* weights[kMaxWeights];
    int32_t columnEnd[kMaxWeights];
    int32_t count;
};

static __device__ __forceinline__ PackedWeightBlock const* weightRow(
    WeightSet const& set, int32_t channel, int32_t blocksPerRow)
{
    PackedWeightBlock const* base = set.weights[0];
    int32_t first = 0;
#pragma unroll
    for (int32_t w = 1; w < kMaxWeights; ++w)
    {
        if (w < set.count && channel >= set.columnEnd[w - 1])
        {
            base = set.weights[w];
            first = set.columnEnd[w - 1];
        }
    }
    return base + static_cast<size_t>(channel - first) * blocksPerRow;
}

static __device__ __forceinline__ float toFloat(half value)
{
    return __half2float(value);
}

static __device__ __forceinline__ float toFloat(__nv_bfloat16 value)
{
    return __bfloat162float(value);
}

template <typename T>
static __device__ __forceinline__ T fromFloat(float value);

template <>
__device__ __forceinline__ float fromFloat<float>(float value)
{
    return value;
}

template <>
__device__ __forceinline__ half fromFloat<half>(float value)
{
    return __float2half_rn(value);
}

template <>
__device__ __forceinline__ __nv_bfloat16 fromFloat<__nv_bfloat16>(float value)
{
    return __float2bfloat16_rn(value);
}

// Q8_1 activations are stored per (8-row token tile, K block) in mma B-fragment
// order: lane l's two B registers are the 8 bytes at l * 8, so token t owns bytes
// [32 t, 32 t + 32) and word 2q + r holds k = 16 r + 4 q .. 16 r + 4 q + 3. FP32
// scales follow as [tile][block][token]. Rows past the real row count are never
// written: they only feed mma output columns that are not stored.
constexpr size_t kActivationBytesPerBlock = kBlockSize + sizeof(float);

__host__ __device__ inline int32_t tokenTiles(int32_t rows)
{
    return (rows + kTokensPerTile - 1) / kTokensPerTile;
}

__host__ __device__ inline size_t activationCodeBytes(int32_t rows, int32_t blocksPerRow)
{
    return static_cast<size_t>(tokenTiles(rows)) * blocksPerRow * kTileCodeBytes;
}

__host__ __device__ inline int32_t fragmentOffset(int32_t tokenInTile, int32_t k)
{
    int32_t const lane = tokenInTile * 4 + (k % 16) / 4;
    return lane * 8 + (k / 16) * 4 + k % 4;
}

// This is the Q8_1 per-block scale/rounding from llama.cpp's MMQ quantizer, writing
// the fragment layout above. Each warp quantizes one (row, block).
template <typename TIn>
static __global__ void quantizeActivationFragments(
    TIn const* input, int8_t* codes, float* scales, int32_t rows, int32_t blocksPerRow)
{
    int32_t const row = blockIdx.y;
    int32_t const block = blockIdx.x * kWarpsPerCta + threadIdx.y;
    if (row >= rows || block >= blocksPerRow)
    {
        return;
    }
    int32_t const lane = threadIdx.x;
    float const value = toFloat(input[(static_cast<size_t>(row) * blocksPerRow + block) * kBlockSize + lane]);
    float peak = fabsf(value);
#pragma unroll
    for (int32_t offset = 16; offset > 0; offset >>= 1)
    {
        peak = fmaxf(peak, __shfl_xor_sync(kFullMask, peak, offset));
    }
    float const inverse = peak > 0.0f ? 127.0f / peak : 0.0f;
    size_t const tileBlock = static_cast<size_t>(row / kTokensPerTile) * blocksPerRow + block;
    int32_t const tokenInTile = row % kTokensPerTile;
    codes[tileBlock * kTileCodeBytes + fragmentOffset(tokenInTile, lane)]
        = static_cast<int8_t>(roundf(value * inverse));
    if (lane == 0)
    {
        scales[tileBlock * kTokensPerTile + tokenInTile] = peak > 0.0f ? 1.0f / inverse : 0.0f;
    }
}

static __device__ __forceinline__ float halfBitsToFloat(uint32_t bits)
{
    return __half2float(__ushort_as_half(static_cast<unsigned short>(bits)));
}

// A block's codes start 2 bytes after its 2-byte scale, so they are 4-byte aligned
// only every other block. Read aligned words and realign with a funnel shift; the
// trailing halfword read stays inside the 34-byte block.
static __device__ __forceinline__ float loadWeightBlock(
    PackedWeightBlock const* block, uint32_t (&codes)[kWordsPerBlock])
{
    uintptr_t const codeAddress = reinterpret_cast<uintptr_t>(block->codes);
    uint32_t const shift = static_cast<uint32_t>(codeAddress & 3U) * 8U;
    uint32_t const* aligned = reinterpret_cast<uint32_t const*>(codeAddress & ~static_cast<uintptr_t>(3U));
    uint32_t words[kWordsPerBlock + 1];
#pragma unroll
    for (int32_t word = 0; word < kWordsPerBlock; ++word)
    {
        words[word] = aligned[word];
    }
    words[kWordsPerBlock] = shift != 0U ? *reinterpret_cast<uint16_t const*>(aligned + kWordsPerBlock) : 0U;
#pragma unroll
    for (int32_t word = 0; word < kWordsPerBlock; ++word)
    {
        codes[word] = __funnelshift_r(words[word], words[word + 1], shift);
    }
    uint32_t const scaleBits
        = shift != 0U ? words[0] : static_cast<uint32_t>(*reinterpret_cast<uint16_t const*>(&block->scale));
    return halfBitsToFloat(scaleBits);
}

// Halfword loads of one block: more load instructions than loadWeightBlock, but faster when a warp loads only one
// channel's block per K step (measured on SM120).
static __device__ __forceinline__ float loadWeightBlockHalfwords(
    PackedWeightBlock const* block, uint32_t (&codes)[kWordsPerBlock])
{
    uint16_t const* halves = reinterpret_cast<uint16_t const*>(block->codes);
#pragma unroll
    for (int32_t word = 0; word < kWordsPerBlock; ++word)
    {
        codes[word] = static_cast<uint32_t>(halves[2 * word]) | (static_cast<uint32_t>(halves[2 * word + 1]) << 16);
    }
    return __half2float(block->scale);
}

// DP4A GEMV for a few activation rows. kWarpsPerChannel warps split K for each
// group of kChannels output channels; lane l of warp w takes blocks
// l + 32 * (w + kWarpsPerChannel * i), and every token keeps that order for any
// kTokens or kChannels.
template <int32_t kTokens, int32_t kChannels, int32_t kWarpsPerChannel, typename TOut>
__global__ void __launch_bounds__(kWarpSize* kWarpsPerCta)
    packedGemv(int8_t const* __restrict__ activationCodes, float const* __restrict__ activationScales,
        WeightSet const weights, TOut* __restrict__ output, int32_t rows, int32_t outFeatures, int32_t blocksPerRow)
{
    static_assert(kWarpsPerCta % kWarpsPerChannel == 0);
    constexpr int32_t kGroupsPerCta = kWarpsPerCta / kWarpsPerChannel;
    int32_t const lane = threadIdx.x;
    int32_t const warp = threadIdx.y;
    int32_t const warpInGroup = warp % kWarpsPerChannel;
    int32_t const firstChannel = (blockIdx.x * kGroupsPerCta + warp / kWarpsPerChannel) * kChannels;
    int32_t const firstToken = blockIdx.y * kTokens;
    float sums[kChannels][kTokens] = {};
    if (firstChannel < outFeatures)
    {
        PackedWeightBlock const* channelRows[kChannels];
#pragma unroll
        for (int32_t c = 0; c < kChannels; ++c)
        {
            channelRows[c] = weightRow(weights, min(firstChannel + c, outFeatures - 1), blocksPerRow);
        }
        for (int32_t block = warpInGroup * kWarpSize + lane; block < blocksPerRow;
            block += kWarpSize * kWarpsPerChannel)
        {
            uint32_t codes[kChannels][kWordsPerBlock];
            float weightScales[kChannels];
#pragma unroll
            for (int32_t c = 0; c < kChannels; ++c)
            {
                weightScales[c] = kChannels == 1 ? loadWeightBlockHalfwords(channelRows[c] + block, codes[c])
                                                 : loadWeightBlock(channelRows[c] + block, codes[c]);
            }
#pragma unroll
            for (int32_t t = 0; t < kTokens; ++t)
            {
                int32_t const token = firstToken + t;
                if (token < rows)
                {
                    size_t const tileBlock = static_cast<size_t>(token / kTokensPerTile) * blocksPerRow + block;
                    int32_t const tokenInTile = token % kTokensPerTile;
                    int4 const* a = reinterpret_cast<int4 const*>(
                        activationCodes + tileBlock * kTileCodeBytes + tokenInTile * kBlockSize);
                    int4 const lo = a[0];
                    int4 const hi = a[1];
                    int32_t const act[kWordsPerBlock] = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
                    float const activationScale = activationScales[tileBlock * kTokensPerTile + tokenInTile];
#pragma unroll
                    for (int32_t c = 0; c < kChannels; ++c)
                    {
                        int32_t dot = 0;
#pragma unroll
                        for (int32_t q = 0; q < 4; ++q)
                        {
                            dot = __dp4a(static_cast<int32_t>(codes[c][q]), act[2 * q], dot);
                            dot = __dp4a(static_cast<int32_t>(codes[c][q + 4]), act[2 * q + 1], dot);
                        }
                        sums[c][t] += static_cast<float>(dot) * weightScales[c] * activationScale;
                    }
                }
            }
        }
    }
#pragma unroll
    for (int32_t offset = 16; offset > 0; offset >>= 1)
    {
#pragma unroll
        for (int32_t c = 0; c < kChannels; ++c)
        {
#pragma unroll
            for (int32_t t = 0; t < kTokens; ++t)
            {
                sums[c][t] += __shfl_down_sync(kFullMask, sums[c][t], offset);
            }
        }
    }
    if constexpr (kWarpsPerChannel > 1)
    {
        __shared__ float partials[kWarpsPerCta][kChannels][kTokens];
        if (lane == 0)
        {
#pragma unroll
            for (int32_t c = 0; c < kChannels; ++c)
            {
#pragma unroll
                for (int32_t t = 0; t < kTokens; ++t)
                {
                    partials[warp][c][t] = sums[c][t];
                }
            }
        }
        __syncthreads();
        if (warpInGroup != 0)
        {
            return;
        }
#pragma unroll
        for (int32_t c = 0; c < kChannels; ++c)
        {
#pragma unroll
            for (int32_t t = 0; t < kTokens; ++t)
            {
#pragma unroll
                for (int32_t other = 1; other < kWarpsPerChannel; ++other)
                {
                    sums[c][t] += partials[warp + other][c][t];
                }
            }
        }
    }
    if (lane == 0)
    {
#pragma unroll
        for (int32_t c = 0; c < kChannels; ++c)
        {
#pragma unroll
            for (int32_t t = 0; t < kTokens; ++t)
            {
                if (firstChannel + c < outFeatures && firstToken + t < rows)
                {
                    output[static_cast<size_t>(firstToken + t) * outFeatures + firstChannel + c]
                        = fromFloat<TOut>(sums[c][t]);
                }
            }
        }
    }
}

static __device__ __forceinline__ void mmaS8(int32_t (&d)[4], uint32_t const (&a)[4], uint2 const& b)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
        "{%10, %10, %10, %10};\n"
        : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b.x), "r"(b.y), "r"(0));
#else
    __trap();
#endif
}

// Tensor-core packed GEMM. A warp owns 16 output channels (mma M) and kTokenTiles
// 8-row activation tiles (mma N), and walks K in pairs of Q8_0 blocks: a pair is
// 68 bytes, so it is 4-byte aligned when the row has an even block count, and the
// even block's codes are realigned with a funnel shift. kWarpsPerChannelTile warps
// split the pairs of one channel tile. The FP32 accumulation order depends only on
// the K geometry and kWarpsPerChannelTile, never on the row count.
template <int32_t kTokenTiles, int32_t kWarpsPerChannelTile, typename TOut>
__global__ void __launch_bounds__(kWarpSize*(kWarpsPerChannelTile > kWarpsPerCta ? kWarpsPerChannelTile : kWarpsPerCta))
    packedGemmMma(int8_t const* __restrict__ activationCodes, float const* __restrict__ activationScales,
        WeightSet const weights, TOut* __restrict__ output, int32_t rows, int32_t outFeatures, int32_t blocksPerRow)
{
    constexpr int32_t kCtaWarps = kWarpsPerChannelTile > kWarpsPerCta ? kWarpsPerChannelTile : kWarpsPerCta;
    constexpr int32_t kChannelTilesPerCta = kCtaWarps / kWarpsPerChannelTile;
    constexpr int32_t kWordsPerPair = 2 * sizeof(PackedWeightBlock) / sizeof(uint32_t);
    // In a pair, even-block codes start at byte 2 and odd-block codes at byte 36 (word 9);
    // the odd-block scale is the high half of word 8, which quad 3 loads as w5.
    constexpr int32_t kOddCodeWord = 9;
    constexpr int32_t kHalfShift = 16;
    constexpr int32_t kOddScaleQuad = 3;
    // With few K-split warps each warp walks many pairs; two running sums over alternate pairs halve its FP32
    // accumulation chain.
    constexpr int32_t kMaxWarpsForTwoSums = 4;

    int32_t const lane = threadIdx.x;
    int32_t const warp = threadIdx.y;
    int32_t const group = lane / 4;
    int32_t const quad = lane % 4;
    int32_t const warpInTile = warp % kWarpsPerChannelTile;
    int32_t const firstChannel = (blockIdx.x * kChannelTilesPerCta + warp / kWarpsPerChannelTile) * kChannelsPerTile;
    int32_t const firstTile = blockIdx.y * kTokenTiles;
    int32_t const validTiles = min(kTokenTiles, tokenTiles(rows) - firstTile);
    int32_t const pairs = blocksPerRow / 2;
    int32_t const evenScaleLane = lane & ~3;
    int32_t const oddScaleLane = evenScaleLane + kOddScaleQuad;

    float acc[kTokenTiles][4] = {};
    if (firstChannel < outFeatures)
    {
        uint32_t const* rowWords[2];
#pragma unroll
        for (int32_t half = 0; half < 2; ++half)
        {
            int32_t const channel = min(firstChannel + group + 8 * half, outFeatures - 1);
            rowWords[half] = reinterpret_cast<uint32_t const*>(weightRow(weights, channel, blocksPerRow));
        }
        auto accumulatePair = [&](int32_t pair, float (&sum)[kTokenTiles][4]) {
            // a[parity] follows the m16n8k32 A layout: registers 0/2 hold channel `group` at
            // k = 4 quad / 16 + 4 quad, registers 1/3 hold channel `group + 8`.
            uint32_t a[2][4];
            float weightScales[2][2];
#pragma unroll
            for (int32_t half = 0; half < 2; ++half)
            {
                uint32_t const* w = rowWords[half] + pair * kWordsPerPair;
                uint32_t const w0 = w[quad];
                uint32_t const w1 = w[quad + 1];
                uint32_t const w4 = w[quad + 4];
                uint32_t const w5 = w[quad + 5];
                a[0][half] = __funnelshift_r(w0, w1, kHalfShift);
                a[0][half + 2] = __funnelshift_r(w4, w5, kHalfShift);
                a[1][half] = w[quad + kOddCodeWord];
                a[1][half + 2] = w[quad + kOddCodeWord + 4];
                weightScales[0][half] = halfBitsToFloat(__shfl_sync(kFullMask, w0, evenScaleLane));
                weightScales[1][half] = halfBitsToFloat(__shfl_sync(kFullMask, w5, oddScaleLane) >> kHalfShift);
            }
            // The pair's two block terms are summed before they join the running sum, which halves its length.
            float pairSum[kTokenTiles][4];
#pragma unroll
            for (int32_t parity = 0; parity < 2; ++parity)
            {
                size_t const block = 2 * pair + parity;
#pragma unroll
                for (int32_t t = 0; t < kTokenTiles; ++t)
                {
                    if (t < validTiles)
                    {
                        size_t const tileBlock = static_cast<size_t>(firstTile + t) * blocksPerRow + block;
                        uint2 const b
                            = reinterpret_cast<uint2 const*>(activationCodes + tileBlock * kTileCodeBytes)[lane];
                        float2 const tokenScales
                            = reinterpret_cast<float2 const*>(activationScales + tileBlock * kTokensPerTile)[quad];
                        int32_t d[4];
                        mmaS8(d, a[parity], b);
                        float const scaled[4] = {static_cast<float>(d[0]) * weightScales[parity][0],
                            static_cast<float>(d[1]) * weightScales[parity][0],
                            static_cast<float>(d[2]) * weightScales[parity][1],
                            static_cast<float>(d[3]) * weightScales[parity][1]};
                        float const tokenScale[4] = {tokenScales.x, tokenScales.y, tokenScales.x, tokenScales.y};
#pragma unroll
                        for (int32_t i = 0; i < 4; ++i)
                        {
                            pairSum[t][i] = parity == 0 ? scaled[i] * tokenScale[i]
                                                        : fmaf(scaled[i], tokenScale[i], pairSum[t][i]);
                        }
                    }
                }
            }
#pragma unroll
            for (int32_t t = 0; t < kTokenTiles; ++t)
            {
                if (t < validTiles)
                {
#pragma unroll
                    for (int32_t i = 0; i < 4; ++i)
                    {
                        sum[t][i] += pairSum[t][i];
                    }
                }
            }
        };
        if constexpr (kWarpsPerChannelTile > kMaxWarpsForTwoSums)
        {
#pragma unroll 2
            for (int32_t pair = warpInTile; pair < pairs; pair += kWarpsPerChannelTile)
            {
                accumulatePair(pair, acc);
            }
        }
        else
        {
            float accAlt[kTokenTiles][4] = {};
            int32_t pair = warpInTile;
            for (; pair + kWarpsPerChannelTile < pairs; pair += 2 * kWarpsPerChannelTile)
            {
                accumulatePair(pair, acc);
                accumulatePair(pair + kWarpsPerChannelTile, accAlt);
            }
            if (pair < pairs)
            {
                accumulatePair(pair, acc);
            }
#pragma unroll
            for (int32_t t = 0; t < kTokenTiles; ++t)
            {
#pragma unroll
                for (int32_t i = 0; i < 4; ++i)
                {
                    acc[t][i] += accAlt[t][i];
                }
            }
        }
    }
    if constexpr (kWarpsPerChannelTile > 1)
    {
        __shared__ float partials[kCtaWarps][kTokenTiles][4][kWarpSize];
#pragma unroll
        for (int32_t t = 0; t < kTokenTiles; ++t)
        {
#pragma unroll
            for (int32_t i = 0; i < 4; ++i)
            {
                partials[warp][t][i][lane] = acc[t][i];
            }
        }
        __syncthreads();
        if (warpInTile != 0)
        {
            return;
        }
        // Pairwise tree over the K-split warps' partial sums.
#pragma unroll
        for (int32_t t = 0; t < kTokenTiles; ++t)
        {
#pragma unroll
            for (int32_t i = 0; i < 4; ++i)
            {
                float partial[kWarpsPerChannelTile];
#pragma unroll
                for (int32_t other = 0; other < kWarpsPerChannelTile; ++other)
                {
                    partial[other] = partials[warp + other][t][i][lane];
                }
#pragma unroll
                for (int32_t stride = kWarpsPerChannelTile / 2; stride > 0; stride /= 2)
                {
#pragma unroll
                    for (int32_t other = 0; other < stride; ++other)
                    {
                        partial[other] += partial[other + stride];
                    }
                }
                acc[t][i] = partial[0];
            }
        }
    }
    // C layout: registers 0/1 are channel `group`, tokens 2 quad / 2 quad + 1; 2/3 are channel `group + 8`.
#pragma unroll
    for (int32_t t = 0; t < kTokenTiles; ++t)
    {
#pragma unroll
        for (int32_t i = 0; i < 4; ++i)
        {
            int32_t const channel = firstChannel + group + (i >> 1) * 8;
            int32_t const token = (firstTile + t) * kTokensPerTile + 2 * quad + (i & 1);
            if (channel < outFeatures && token < rows)
            {
                output[static_cast<size_t>(token) * outFeatures + channel] = fromFloat<TOut>(acc[t][i]);
            }
        }
    }
}
// Prefill tiling: a CTA stages a 64-token x 128-channel tile through shared memory, so each weight block is read
// once per 64 tokens instead of once per 8- or 16-row token tile. One m16n8k32 INT8 MMA step covers exactly one Q8_0
// block, and its integer dot is scaled by the activation and weight block scales, as in the DP4A path.
constexpr int32_t kTiledMinRows = 32;
constexpr int32_t kTiledTokens = 64;
constexpr int32_t kTiledChannels = 128;
constexpr int32_t kTiledBlocks = 4;
constexpr int32_t kTiledStages = 3;
constexpr int32_t kTiledWarpsN = 4;
constexpr int32_t kTiledThreads = 256;
constexpr int32_t kWarpTokens = 32;
constexpr int32_t kWarpChannels = 32;
constexpr int32_t kTiledTokenTiles = kTiledTokens / kTokensPerTile;
// Per stage, the fragment-layout activations of kTiledTokenTiles token tiles: codes, then scales.
constexpr int32_t kTileCodeStageBytes = kTiledBlocks * kTileCodeBytes;
constexpr int32_t kTileScaleStageBytes = kTiledBlocks * kTokensPerTile * sizeof(float);
constexpr int32_t kActivationScaleOffset = kTiledTokenTiles * kTileCodeStageBytes;
constexpr int32_t kActivationStageBytes = kActivationScaleOffset + kTiledTokenTiles * kTileScaleStageBytes;
constexpr int32_t kWeightTileBytes = kTiledBlocks * sizeof(PackedWeightBlock);
constexpr int32_t kStageBytes = kActivationStageBytes + kTiledChannels * kWeightTileBytes;
constexpr int32_t kTiledSmemBytes = kTiledStages * kStageBytes;
static_assert(kTileCodeStageBytes % 16 == 0 && kTileScaleStageBytes % 16 == 0 && kWeightTileBytes % 8 == 0);

template <int32_t kBytes>
static __device__ __forceinline__ void cpAsync(void* dst, void const* src, bool valid)
{
    auto const smem = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
    asm volatile(
        "cp.async.ca.shared.global [%0], [%1], %2, %3;\n" ::"r"(smem), "l"(src), "n"(kBytes), "r"(valid ? kBytes : 0));
}

static __device__ __forceinline__ uint32_t loadUnaligned32(uint8_t const* p)
{
    auto const* h = reinterpret_cast<uint16_t const*>(p);
    return static_cast<uint32_t>(h[0]) | (static_cast<uint32_t>(h[1]) << 16);
}

// Activations use the Q8_1 fragment layout above, shared with the GEMV and small-M MMA paths.
template <typename TOut>
static __global__ void __launch_bounds__(kTiledThreads)
    packedGemmTiled(int8_t const* activationCodes, float const* activationScales, WeightSet const weights, TOut* output,
        int32_t rows, int32_t outFeatures, int32_t blocksPerRow)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    extern __shared__ __align__(16) uint8_t smem[];
    int32_t const firstToken = blockIdx.y * kTiledTokens;
    int32_t const firstTokenTile = firstToken / kTokensPerTile;
    int32_t const rowTiles = tokenTiles(rows);
    int32_t const firstChannel = blockIdx.x * kTiledChannels;
    int32_t const tiles = blocksPerRow / kTiledBlocks;
    auto const* codeBytes = reinterpret_cast<uint8_t const*>(activationCodes);
    auto const* scaleBytes = reinterpret_cast<uint8_t const*>(activationScales);

    auto loadTile = [&](int32_t tile) {
        uint8_t* stage = smem + (tile % kTiledStages) * kStageBytes;
        constexpr int32_t kCodeChunks = kTileCodeStageBytes / 16;
        for (int32_t i = threadIdx.x; i < kTiledTokenTiles * kCodeChunks; i += kTiledThreads)
        {
            int32_t const tokenTile = i / kCodeChunks;
            int32_t const chunk = i % kCodeChunks;
            bool const valid = firstTokenTile + tokenTile < rowTiles;
            size_t const tileBlock
                = static_cast<size_t>(firstTokenTile + tokenTile) * blocksPerRow + tile * kTiledBlocks;
            uint8_t const* src = valid ? codeBytes + tileBlock * kTileCodeBytes + chunk * 16 : codeBytes;
            cpAsync<16>(stage + tokenTile * kTileCodeStageBytes + chunk * 16, src, valid);
        }
        constexpr int32_t kScaleChunks = kTileScaleStageBytes / 16;
        for (int32_t i = threadIdx.x; i < kTiledTokenTiles * kScaleChunks; i += kTiledThreads)
        {
            int32_t const tokenTile = i / kScaleChunks;
            int32_t const chunk = i % kScaleChunks;
            bool const valid = firstTokenTile + tokenTile < rowTiles;
            size_t const tileBlock
                = static_cast<size_t>(firstTokenTile + tokenTile) * blocksPerRow + tile * kTiledBlocks;
            uint8_t const* src
                = valid ? scaleBytes + tileBlock * kTokensPerTile * sizeof(float) + chunk * 16 : scaleBytes;
            cpAsync<16>(stage + kActivationScaleOffset + tokenTile * kTileScaleStageBytes + chunk * 16, src, valid);
        }
        uint8_t* weightStage = stage + kActivationStageBytes;
        constexpr int32_t kWeightChunks = kWeightTileBytes / 8;
        for (int32_t i = threadIdx.x; i < kTiledChannels * kWeightChunks; i += kTiledThreads)
        {
            int32_t const channel = i / kWeightChunks;
            int32_t const chunk = i % kWeightChunks;
            bool const valid = firstChannel + channel < outFeatures;
            auto const* src = reinterpret_cast<uint8_t const*>(weights.weights[0]);
            if (valid)
            {
                src = reinterpret_cast<uint8_t const*>(weightRow(weights, firstChannel + channel, blocksPerRow))
                    + tile * kWeightTileBytes + chunk * 8;
            }
            cpAsync<8>(weightStage + channel * kWeightTileBytes + chunk * 8, src, valid);
        }
    };

    int32_t const warp = threadIdx.x / 32;
    int32_t const lane = threadIdx.x % 32;
    int32_t const group = lane / 4;
    int32_t const quad = lane % 4;
    int32_t const warpToken = (warp / kTiledWarpsN) * kWarpTokens;
    int32_t const warpChannel = (warp % kTiledWarpsN) * kWarpChannels;

    float acc[2][4][4] = {};
#pragma unroll
    for (int32_t s = 0; s < kTiledStages - 1; ++s)
    {
        if (s < tiles)
        {
            loadTile(s);
        }
        asm volatile("cp.async.commit_group;\n" ::);
    }
    for (int32_t tile = 0; tile < tiles; ++tile)
    {
        asm volatile("cp.async.wait_group %0;\n" ::"n"(kTiledStages - 2));
        __syncthreads();
        if (tile + kTiledStages - 1 < tiles)
        {
            loadTile(tile + kTiledStages - 1);
        }
        asm volatile("cp.async.commit_group;\n" ::);

        uint8_t const* sA = smem + (tile % kTiledStages) * kStageBytes;
        uint8_t const* sW = sA + kActivationStageBytes;
#pragma unroll
        for (int32_t block = 0; block < kTiledBlocks; ++block)
        {
            int32_t const wOffset = block * static_cast<int32_t>(sizeof(PackedWeightBlock));
            uint32_t a[2][4];
            float aScale[2][2];
#pragma unroll
            for (int32_t mt = 0; mt < 2; ++mt)
            {
                // Rows group and group + 8 of this m16 tile are token `group` of two consecutive token tiles; each
                // token's 8 fragment bytes at quad hold k = 4 quad.. (A reg 0/1) and k = 16 + 4 quad.. (A reg 2/3).
                int32_t const tokenTile = (warpToken + mt * 16) / kTokensPerTile;
                uint8_t const* c0
                    = sA + tokenTile * kTileCodeStageBytes + block * kTileCodeBytes + group * kBlockSize + quad * 8;
                uint8_t const* c1 = c0 + kTileCodeStageBytes;
                uint2 const lo = *reinterpret_cast<uint2 const*>(c0);
                uint2 const hi = *reinterpret_cast<uint2 const*>(c1);
                a[mt][0] = lo.x;
                a[mt][1] = hi.x;
                a[mt][2] = lo.y;
                a[mt][3] = hi.y;
                float const* s0
                    = reinterpret_cast<float const*>(sA + kActivationScaleOffset + tokenTile * kTileScaleStageBytes)
                    + block * kTokensPerTile + group;
                aScale[mt][0] = s0[0];
                aScale[mt][1] = s0[kTileScaleStageBytes / sizeof(float)];
            }
#pragma unroll
            for (int32_t nt = 0; nt < 4; ++nt)
            {
                uint8_t const* w = sW + (warpChannel + nt * 8 + group) * kWeightTileBytes + wOffset;
                uint32_t const b0 = loadUnaligned32(w + 2 + quad * 4);
                uint32_t const b1 = loadUnaligned32(w + 18 + quad * 4);
                uint8_t const* ws = sW + (warpChannel + nt * 8 + quad * 2) * kWeightTileBytes + wOffset;
                float const wScale0 = __half2float(*reinterpret_cast<half const*>(ws));
                float const wScale1 = __half2float(*reinterpret_cast<half const*>(ws + kWeightTileBytes));
#pragma unroll
                for (int32_t mt = 0; mt < 2; ++mt)
                {
                    int32_t c[4];
                    asm volatile(
                        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                        "{%10,%10,%10,%10};\n"
                        : "=r"(c[0]), "=r"(c[1]), "=r"(c[2]), "=r"(c[3])
                        : "r"(a[mt][0]), "r"(a[mt][1]), "r"(a[mt][2]), "r"(a[mt][3]), "r"(b0), "r"(b1), "r"(0));
                    acc[mt][nt][0] += static_cast<float>(c[0]) * aScale[mt][0] * wScale0;
                    acc[mt][nt][1] += static_cast<float>(c[1]) * aScale[mt][0] * wScale1;
                    acc[mt][nt][2] += static_cast<float>(c[2]) * aScale[mt][1] * wScale0;
                    acc[mt][nt][3] += static_cast<float>(c[3]) * aScale[mt][1] * wScale1;
                }
            }
        }
    }
    asm volatile("cp.async.wait_group 0;\n" ::);

#pragma unroll
    for (int32_t mt = 0; mt < 2; ++mt)
    {
#pragma unroll
        for (int32_t rowHalf = 0; rowHalf < 2; ++rowHalf)
        {
            int32_t const row = firstToken + warpToken + mt * 16 + group + rowHalf * 8;
            if (row >= rows)
            {
                continue;
            }
#pragma unroll
            for (int32_t nt = 0; nt < 4; ++nt)
            {
                int32_t const channel = firstChannel + warpChannel + nt * 8 + quad * 2;
                TOut* dst = output + static_cast<size_t>(row) * outFeatures + channel;
                if (channel < outFeatures)
                {
                    dst[0] = fromFloat<TOut>(acc[mt][nt][rowHalf * 2]);
                }
                if (channel + 1 < outFeatures)
                {
                    dst[1] = fromFloat<TOut>(acc[mt][nt][rowHalf * 2 + 1]);
                }
            }
        }
    }
#endif
}
} // namespace trt_edgellm::kernel::q8_0
