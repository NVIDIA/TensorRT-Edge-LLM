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

// CUDA-core kernels of Nvfp4A16BlackwellMoePlugin (Thor SM110 W4A16 routed MoE),
// compiled with NVRTC per layer shape (nvfp4A16BlackwellMoeJitCompiler.cpp bakes
// the MOE_* macros below) and launched through the driver API
// (nvfp4A16BlackwellMoeJitRunner.cpp).  The tcgen05 grouped GEMMs of the
// prefill path are CuTe DSL AOT kernels and live in
// kernelSrcs/nvfp4_a16_blackwell_moe/.
//
// Decode chain (T <= kDecodeMaxTokens): route -> fc1 [-> fc1_reduce] -> fc2 [-> fc2_reduce]
// Prefill chain (T >= 2):               route -> layout -> gather -> grouped GEMMs
//
// Every kernel is a Programmatic Dependent Launch secondary: it issues
// griddepcontrol.wait before its first read of anything its immediate
// predecessor produced and before its first global write (TensorRT may still
// hand workspace/output memory to the running predecessor), immediately followed
// by griddepcontrol.launch_dependents (a dependent grid is scheduled once every
// CTA of this grid has triggered, i.e. during its last wave, and its own wait
// orders the data).  Inputs produced two or more launches earlier are complete
// and visible when a kernel starts (its predecessor passed its own wait before
// triggering), so those are read before the wait.  Both instructions are no-ops
// when the grid was launched without the PDL attribute.
//
// Weight layout BLACKWELL_MOE_N128_K64_V1 (one buffer shared with the GEMMs):
//   qweight      int8 [E, N_pad/128, K/64, 128, 32]   32 B = 64 E2M1 codes per row tile
//   block_scales int8 [E, N_pad/128, K/64, 128, 4]    raw E4M3, one per 16 K
//   global_scale fp32 [E]
// Each 32-byte code row carries the TMA SWIZZLE_32B image: rows with bit 2 of
// n%128 set store their two 16-byte halves swapped.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>

#if !defined(MOE_NUM_EXPERTS) || !defined(MOE_TOP_K) || !defined(MOE_HIDDEN) || !defined(MOE_INTER)                    \
    || !defined(MOE_INTER_PAD) || !defined(MOE_FC1_SPLIT_K) || !defined(MOE_FC2_SPLIT_K)                               \
    || !defined(MOE_FC2_PREFETCH_SLOTS) || !defined(MOE_DATA_TYPE) || !defined(MOE_LAYOUT_ABI)                         \
    || !defined(MOE_SOURCE_ABI)
#error "NVFP4-A16 Blackwell MoE JIT configuration is incomplete"
#endif
#if MOE_LAYOUT_ABI != 1
#error "Unsupported NVFP4-A16 Blackwell MoE layout ABI"
#endif
#if MOE_SOURCE_ABI != 1
#error "Unsupported NVFP4-A16 Blackwell MoE source ABI"
#endif

namespace
{

// ---------------------------------------------------------------------------
// Baked contract
// ---------------------------------------------------------------------------
constexpr int kNumExperts{MOE_NUM_EXPERTS};
constexpr int kTopK{MOE_TOP_K};
constexpr int kHidden{MOE_HIDDEN};       // H: FC1 K, FC2 N
constexpr int kInter{MOE_INTER};         // I: FC2 K
constexpr int kInterPad{MOE_INTER_PAD};  // FC1 N (I rounded up to 128)
constexpr int kFc1SplitK{MOE_FC1_SPLIT_K};
constexpr int kFc2SplitK{MOE_FC2_SPLIT_K};
constexpr int kFc2PrefetchSlots{MOE_FC2_PREFETCH_SLOTS};

#if MOE_DATA_TYPE == 0
using ActT = half;
#elif MOE_DATA_TYPE == 1
using ActT = __nv_bfloat16;
#else
#error "Unsupported NVFP4-A16 Blackwell MoE data type"
#endif

constexpr int kWarpSize{32};
constexpr int kWarpsPerBlock{8};
constexpr int kThreadsPerBlock{kWarpSize * kWarpsPerBlock};
constexpr int kLayoutThreads{1024};
constexpr int kNTile{128};
constexpr int kKTile{64};
constexpr int kRowsPerWarp{16};
constexpr int kPackedBytesPerRowTile{kKTile / 2};
constexpr int kScalesPerRowTile{kKTile / 16};
constexpr int kPackedBytesPerThread{kPackedBytesPerRowTile / 2};
constexpr int kSmemHalfStride{kKTile / 2 + 4};
constexpr int kSmemRowStride{kKTile + 4};
constexpr int kCodeBytesPerTile{kNTile * kPackedBytesPerRowTile};
constexpr int kScaleBytesPerTile{kNTile * kScalesPerRowTile};
constexpr float kNegInf{-3.402823466e+38f};

constexpr int kHiddenKBlocks{kHidden / kKTile};
constexpr int kInterKBlocks{kInter / kKTile};
constexpr int kFc1TilesPerSplit{(kHiddenKBlocks + kFc1SplitK - 1) / kFc1SplitK};
constexpr int kFc2TilesPerSplit{(kInterKBlocks + kFc2SplitK - 1) / kFc2SplitK};
constexpr long long kFc1ExpertPlaneCodes{static_cast<long long>(kInterPad) * kHidden / 2};
constexpr long long kFc2ExpertPlaneCodes{static_cast<long long>(kHidden) * kInter / 2};
//! Experts per lane of the warp routing (32 lanes cover kNumExperts).
constexpr int kRouteExpertsPerLane{(kNumExperts + kWarpSize - 1) / kWarpSize};
constexpr int kRouteWordsPerWarp{kNumExperts + 2 * kTopK};

//! Weight row tiles each lane keeps in flight before consuming them.  Thor's
//! LPDDR5X latency is high; with one 16-byte load per lane per tile the kernel
//! is latency bound, with four it streams.
constexpr int kPrefetchTiles{4};
//! CTAs per SM the FC1/FC2 kernels are register-bounded to.  With 8 warps and
//! kPrefetchTiles 16-byte loads per lane each CTA keeps 16 KB of weight traffic
//! in flight; Thor needs >= 32 KB per SM to stream near its ~235 GB/s ceiling
//! with only 20 SMs.  FC1 compiles to ~72 registers, FC2 (top-k slot loop) is
//! capped at 80 by the (256, 3) bound and lands at ~79 without spills.
constexpr int kFc1CtasPerSm{2};
constexpr int kFc2CtasPerSm{3};

static_assert(kNumExperts >= 1 && kNumExperts <= 512, "numExperts out of range");
static_assert(kTopK >= 1 && kTopK <= 32 && kTopK <= kNumExperts, "topK out of range");
static_assert(kHidden % kNTile == 0 && kInter % kKTile == 0 && kInterPad % kNTile == 0, "misaligned shape");
static_assert(kInterPad >= kInter && kInterPad - kInter < kNTile, "interSizePadded must be I rounded up to 128");
static_assert(kFc1SplitK >= 1 && kFc1SplitK <= kHiddenKBlocks, "fc1SplitK out of range");
static_assert(kFc2SplitK >= 1 && kFc2SplitK <= kInterKBlocks, "fc2SplitK out of range");
static_assert(kFc2PrefetchSlots >= 0 && kFc2PrefetchSlots <= kTopK, "fc2PrefetchSlots out of range");

// ---------------------------------------------------------------------------
// PDL / async-copy / load helpers
// ---------------------------------------------------------------------------
__device__ __forceinline__ void pdlWait()
{
    asm volatile("griddepcontrol.wait;\n" ::: "memory");
}

__device__ __forceinline__ void pdlTrigger()
{
    asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory");
}

__device__ __forceinline__ void cpAsync16(void* const smemDst, void const* const gmemSrc)
{
    unsigned int const dst = static_cast<unsigned int>(__cvta_generic_to_shared(smemDst));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(dst), "l"(gmemSrc) : "memory");
}

__device__ __forceinline__ void cpAsyncCommit()
{
    asm volatile("cp.async.commit_group;" ::: "memory");
}

__device__ __forceinline__ void cpAsyncWaitAll()
{
    asm volatile("cp.async.wait_group 0;" ::: "memory");
}

//! Prefetch [pointer, pointer + bytes) into L2, one 128-byte line per lane per step.
__device__ __forceinline__ void prefetchL2(void const* const pointer, int const bytes, int const lane)
{
    for (int offset = lane * 128; offset < bytes; offset += kWarpSize * 128)
    {
        asm volatile("prefetch.global.L2 [%0];" ::"l"(static_cast<unsigned char const*>(pointer) + offset));
    }
}

struct __align__(16) Uint4
{
    unsigned int x;
    unsigned int y;
    unsigned int z;
    unsigned int w;
};

__device__ __forceinline__ Uint4 loadGlobal128NoAllocate(void const* const pointer)
{
    Uint4 value{};
    asm("ld.global.L1::no_allocate.v4.u32 {%0, %1, %2, %3}, [%4];"
        : "=r"(value.x), "=r"(value.y), "=r"(value.z), "=r"(value.w)
        : "l"(pointer)
        : "memory");
    return value;
}

__device__ __forceinline__ Uint4 loadGlobal128Retain(void const* const pointer)
{
    Uint4 value{};
    asm("ld.global.L1::evict_last.v4.u32 {%0, %1, %2, %3}, [%4];"
        : "=r"(value.x), "=r"(value.y), "=r"(value.z), "=r"(value.w)
        : "l"(pointer)
        : "memory");
    return value;
}

__device__ __forceinline__ float2 fp4PairToFloat2(unsigned int const packedByte)
{
    __half2_raw const raw = __nv_cvt_fp4x2_to_halfraw2(static_cast<__nv_fp4x2_storage_t>(packedByte), __NV_E2M1);
    return __half22float2(static_cast<half2>(raw));
}

__device__ __forceinline__ float2 fp8PairToFloat2(unsigned short const packedBytes)
{
    __half2_raw const raw = __nv_cvt_fp8x2_to_halfraw2(static_cast<__nv_fp8x2_storage_t>(packedBytes), __NV_E4M3);
    return __half22float2(static_cast<half2>(raw));
}

template <typename T>
struct ActTraits;

template <>
struct ActTraits<half>
{
    static __device__ __forceinline__ float2 pairToFloat2(half const* const p)
    {
        return __half22float2(*reinterpret_cast<half2 const*>(p));
    }
    static __device__ __forceinline__ half fromFloat(float const v)
    {
        return __float2half_rn(v);
    }
    static __device__ __forceinline__ void storePair(half* const p, float2 const v)
    {
        *reinterpret_cast<half2*>(p) = __floats2half2_rn(v.x, v.y);
    }
};

template <>
struct ActTraits<__nv_bfloat16>
{
    static __device__ __forceinline__ float2 pairToFloat2(__nv_bfloat16 const* const p)
    {
        return __bfloat1622float2(*reinterpret_cast<__nv_bfloat162 const*>(p));
    }
    static __device__ __forceinline__ __nv_bfloat16 fromFloat(float const v)
    {
        return __float2bfloat16_rn(v);
    }
    static __device__ __forceinline__ void storePair(__nv_bfloat16* const p, float2 const v)
    {
        *reinterpret_cast<__nv_bfloat162*>(p) = __floats2bfloat162_rn(v.x, v.y);
    }
};

// ---------------------------------------------------------------------------
// Dequant-GEMV core (mirrors kernelSrcs/nvfp4A16BlackwellGemv): 8 warps per CTA,
// one CTA per 128-row N tile, 16 rows per warp, two lanes per output row (each
// lane owns 32 of the 64 K values of a tile), 128-bit streaming weight loads,
// cvt.rn.f16x2.e2m1x2 / .e4m3x2 dequant, fp32 FMA.
// ---------------------------------------------------------------------------

//! Stage K tiles [kBlockBegin, kBlockEnd) of one activation row into shared memory
//! as fp32, one kSmemRowStride-float slot per tile with the 4-float skew between the
//! two K halves (bank-conflict free reads).  All threads of the CTA cooperate; the
//! caller synchronizes.
__device__ __forceinline__ void stageActivationRange(
    ActT const* const row, int const kBlockBegin, int const kBlockEnd, float* const sharedActivation)
{
    constexpr int kElementsPerVector{static_cast<int>(sizeof(Uint4) / sizeof(ActT))};
    constexpr int kVectorsPerTile{kKTile / kElementsPerVector};
    int const numVectors = (kBlockEnd - kBlockBegin) * kVectorsPerTile;
    for (int vector = static_cast<int>(threadIdx.x); vector < numVectors; vector += kThreadsPerBlock)
    {
        int const tile = vector / kVectorsPerTile;
        int const logicalK = (vector - tile * kVectorsPerTile) * kElementsPerVector;
        Uint4 const packed = loadGlobal128Retain(row + (kBlockBegin + tile) * kKTile + logicalK);
        ActT const* const elements = reinterpret_cast<ActT const*>(&packed);
        int const stagedK = tile * kSmemRowStride + logicalK + (logicalK >= kKTile / 2 ? 4 : 0);
        float2* const staged = reinterpret_cast<float2*>(sharedActivation + stagedK);
#pragma unroll
        for (int pair = 0; pair < kElementsPerVector / 2; ++pair)
        {
            staged[pair] = ActTraits<ActT>::pairToFloat2(elements + pair * 2);
        }
    }
}

//! One lane's half of a weight row tile: 16 packed bytes (32 E2M1 codes) and the
//! two E4M3 block scales covering them.
struct RowTileHalf
{
    Uint4 codes;
    unsigned short scales;
};

//! ``rowTileIndex`` addresses BLACKWELL_MOE_N128_K64_V1:
//! ((n/128)*kBlocks + kBlock)*128 + n%128 inside the expert plane.  Bit 2 of the
//! row tile index is the SWIZZLE_32B half-swap bit.
__device__ __forceinline__ RowTileHalf loadRowTileHalf(unsigned char const* __restrict__ const qweights,
    unsigned char const* __restrict__ const blockScales, long long const rowTileIndex, int const kHalf)
{
    RowTileHalf tile{};
    int const storedHalf = kHalf ^ static_cast<int>((rowTileIndex >> 2) & 1);
    tile.codes = loadGlobal128NoAllocate(
        qweights + rowTileIndex * kPackedBytesPerRowTile + storedHalf * kPackedBytesPerThread);
    tile.scales = *reinterpret_cast<unsigned short const*>(blockScales + rowTileIndex * kScalesPerRowTile + kHalf * 2);
    return tile;
}

//! Same row tile half read from the shared-memory staging of one slot's K range
//! (kCodeBytesPerTile / kScaleBytesPerTile per tile, tiles consecutive).
__device__ __forceinline__ RowTileHalf loadRowTileHalfSmem(
    unsigned char const* const sCodes, unsigned char const* const sScales, int const tile, int const rowInTile, int const kHalf)
{
    RowTileHalf value{};
    int const storedHalf = kHalf ^ ((rowInTile >> 2) & 1);
    value.codes = *reinterpret_cast<Uint4 const*>(
        sCodes + tile * kCodeBytesPerTile + rowInTile * kPackedBytesPerRowTile + storedHalf * kPackedBytesPerThread);
    value.scales = *reinterpret_cast<unsigned short const*>(
        sScales + tile * kScaleBytesPerTile + rowInTile * kScalesPerRowTile + kHalf * 2);
    return value;
}

//! Dot product of a lane's 32 dequantized weights with the matching staged
//! activation half (``stagedHalf`` already offset by tile and K half).
__device__ __forceinline__ float dotRowTileHalf(RowTileHalf const& tile, float const* const stagedHalf)
{
    float2 const scales = fp8PairToFloat2(tile.scales);
    unsigned int const words[4]{tile.codes.x, tile.codes.y, tile.codes.z, tile.codes.w};
    float acc = 0.0f;
#pragma unroll
    for (int word = 0; word < 4; ++word)
    {
#pragma unroll
        for (int byte = 0; byte < 4; ++byte)
        {
            int const pair = word * 4 + byte;
            unsigned int const packedByte = (words[word] >> (byte * 8)) & 0xFFU;
            float2 const weights = fp4PairToFloat2(packedByte);
            float const scale = pair < 8 ? scales.x : scales.y;
            float2 const values = *reinterpret_cast<float2 const*>(stagedHalf + pair * 2);
            float const dot = fmaf(values.x, weights.x, values.y * weights.y);
            acc = fmaf(dot, scale, acc);
        }
    }
    return acc;
}

//! Dot of one weight row (this lane's half) over the numTiles tiles staged in
//! shared memory by the pre-wait cp.async prefetch.
__device__ __forceinline__ float dotStagedRowRange(unsigned char const* const sCodes, unsigned char const* const sScales,
    int const numTiles, int const rowInTile, int const kHalf, float const* const stagedRange)
{
    float acc = 0.0f;
    float const* const stagedHalf = stagedRange + kHalf * kSmemHalfStride;
    for (int tile = 0; tile < numTiles; ++tile)
    {
        acc += dotRowTileHalf(
            loadRowTileHalfSmem(sCodes, sScales, tile, rowInTile, kHalf), stagedHalf + tile * kSmemRowStride);
    }
    return acc;
}

//! Streams K tiles [kBlockBegin, kBlockEnd) of one weight row (this lane's half)
//! against the activation staged for the same range, kPrefetchTiles loads in
//! flight.  One predicated loop keeps the live tile array at kPrefetchTiles
//! entries, which is what bounds the register count.
__device__ __forceinline__ float streamRowRange(unsigned char const* __restrict__ const qweights,
    unsigned char const* __restrict__ const blockScales, long long const rowTileBase, int const kBlockBegin,
    int const kBlockEnd, int const kHalf, float const* const stagedRange)
{
    float acc = 0.0f;
    float const* const stagedHalf = stagedRange + kHalf * kSmemHalfStride;
    for (int kBlock = kBlockBegin; kBlock < kBlockEnd; kBlock += kPrefetchTiles)
    {
        RowTileHalf tiles[kPrefetchTiles];
#pragma unroll
        for (int j = 0; j < kPrefetchTiles; ++j)
        {
            if (kBlock + j < kBlockEnd)
            {
                tiles[j] = loadRowTileHalf(
                    qweights, blockScales, rowTileBase + static_cast<long long>(kBlock + j) * kNTile, kHalf);
            }
        }
#pragma unroll
        for (int j = 0; j < kPrefetchTiles; ++j)
        {
            if (kBlock + j < kBlockEnd)
            {
                acc += dotRowTileHalf(tiles[j], stagedHalf + (kBlock + j - kBlockBegin) * kSmemRowStride);
            }
        }
    }
    return acc;
}

// ---------------------------------------------------------------------------
// Routing: single-warp sigmoid top-k for the ungrouped contract (n_group == 1),
// same selection order and weights as moeSigmoidGroupTopk: sigmoid -> +bias ->
// iterative arg-max (ties resolve to the lower expert id) -> weights from the
// UNbiased sigmoid -> optional renorm -> scale.  Grouped contracts use the
// shared moeSigmoidGroupTopk kernel from the runner instead.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void sigmoidTopkWarp(float const* __restrict__ logits,
    float const* __restrict__ correctionBias, bool const normTopkProb, float const routedScalingFactor,
    float* sSigmoid, int* sTopkIdx, float* sTopkWeight)
{
    int const lane = static_cast<int>(threadIdx.x) & (kWarpSize - 1);
    float biased[kRouteExpertsPerLane];
#pragma unroll
    for (int i = 0; i < kRouteExpertsPerLane; ++i)
    {
        int const e = lane + kWarpSize * i;
        biased[i] = kNegInf;
        if (e < kNumExperts)
        {
            float const sig = 1.0f / (1.0f + expf(-logits[e]));
            sSigmoid[e] = sig;
            biased[i] = correctionBias != nullptr ? sig + correctionBias[e] : sig;
        }
    }
    __syncwarp();
    float sum = 0.0f;
    for (int k = 0; k < kTopK; ++k)
    {
        float bestVal = kNegInf;
        int bestIdx = 0x7FFFFFFF;
#pragma unroll
        for (int i = 0; i < kRouteExpertsPerLane; ++i)
        {
            // Lane-local candidates are in increasing expert order, so the first
            // strictly greater value wins and ties keep the lower expert id.
            if (biased[i] > bestVal)
            {
                bestVal = biased[i];
                bestIdx = lane + kWarpSize * i;
            }
        }
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
        {
            float const otherVal = __shfl_xor_sync(0xFFFFFFFFU, bestVal, offset);
            int const otherIdx = __shfl_xor_sync(0xFFFFFFFFU, bestIdx, offset);
            if (otherVal > bestVal || (otherVal == bestVal && otherIdx < bestIdx))
            {
                bestVal = otherVal;
                bestIdx = otherIdx;
            }
        }
        // bestIdx is warp-uniform here; the owning lane retires it.
        if ((bestIdx & (kWarpSize - 1)) == lane)
        {
#pragma unroll
            for (int i = 0; i < kRouteExpertsPerLane; ++i)
            {
                if (i == (bestIdx >> 5))
                {
                    biased[i] = kNegInf;
                }
            }
        }
        float const weight = sSigmoid[bestIdx];
        sum += weight;
        if (lane == 0)
        {
            sTopkIdx[k] = bestIdx;
            sTopkWeight[k] = weight;
        }
    }
    __syncwarp();
    if (lane < kTopK)
    {
        float const inv = (normTopkProb && sum > 0.0f) ? 1.0f / sum : 1.0f;
        sTopkWeight[lane] = sTopkWeight[lane] * inv * routedScalingFactor;
    }
    __syncwarp();
}

} // namespace

// ===========================================================================
// Entry points.  Grids (host, nvfp4A16BlackwellMoeJitRunner.cpp):
//   route:      ceil(T / 8) x 256                 layout: 1 x 1024
//   gather:     (maxRowsPadded + T) x 256
//   fc1:        (I_pad/128, T*topK, fc1SplitK) x 256   fc1_reduce: ceil(T*topK*I_pad/2 / 256) x 256
//   fc2:        (H/128, T, fc2SplitK) x 256           fc2_reduce: ceil(T*H/2 / 256) x 256
// All shared memory is static (sized by the baked contract).
// ===========================================================================

//! Warp-per-token sigmoid top-k routing; writes topkIndices/topkWeights [T, topK].
extern "C" __global__ __launch_bounds__(kThreadsPerBlock) void nvfp4_a16_blackwell_moe_route(
    float const* __restrict__ logits, float const* __restrict__ correctionBias, int const numTokens,
    int const normTopkProb, float const routedScalingFactor, int* __restrict__ topkIndices,
    float* __restrict__ topkWeights)
{
    __shared__ __align__(16) float routeSmem[kWarpsPerBlock * kRouteWordsPerWarp];
    int const warp = static_cast<int>(threadIdx.x) / kWarpSize;
    int const lane = static_cast<int>(threadIdx.x) & (kWarpSize - 1);
    int const token = static_cast<int>(blockIdx.x) * kWarpsPerBlock + warp;
    // The router logits are written by the preceding layer; the correction bias
    // is a constant weight and the only input that may be touched before the wait.
    if (correctionBias != nullptr)
    {
        prefetchL2(correctionBias, kNumExperts * 4, lane);
    }
    pdlWait();
    pdlTrigger();
    if (token < numTokens)
    {
        float* const sSigmoid = routeSmem + warp * kRouteWordsPerWarp;
        float* const sWeight = sSigmoid + kNumExperts;
        int* const sIdx = reinterpret_cast<int*>(sWeight + kTopK);
        sigmoidTopkWarp(logits + static_cast<long long>(token) * kNumExperts, correctionBias, normTopkProb != 0,
            routedScalingFactor, sSigmoid, sIdx, sWeight);
        if (lane < kTopK)
        {
            topkIndices[static_cast<long long>(token) * kTopK + lane] = sIdx[lane];
            topkWeights[static_cast<long long>(token) * kTopK + lane] = sWeight[lane];
        }
    }
}

//! Single-CTA expert-contiguous, tile-padded layout (same contract as buildLayoutGpu
//! minus tile_mn_limit, which the grouped GEMM does not read): permuted_idx[r] =
//! expanded slot (token * topK + k) or -1 for a pad row, tile_group_idx[t] = expert
//! of token tile t, num_valid_tiles[0] = tile count.  Rows of one expert are
//! contiguous and start at a tile boundary; experts with no rows get no tiles.
extern "C" __global__ __launch_bounds__(kLayoutThreads) void nvfp4_a16_blackwell_moe_layout(
    int const* __restrict__ topkIndices, int const numSlots, int const tokenTile, int* __restrict__ permutedIdx,
    int* __restrict__ tileGroupIdx, int* __restrict__ numValidTiles)
{
    __shared__ int count[kNumExperts];
    __shared__ int rowOffset[kNumExperts];
    __shared__ int tileOffset[kNumExperts];
    __shared__ int cursor[kNumExperts];
    int const tid = static_cast<int>(threadIdx.x);
    for (int e = tid; e < kNumExperts; e += kLayoutThreads)
    {
        count[e] = 0;
        cursor[e] = 0;
    }
    __syncthreads();
    // topkIndices is written by the routing kernel; the shared-memory counters
    // above are the only work that can precede the wait.
    pdlWait();
    pdlTrigger();
    for (int i = tid; i < numSlots; i += kLayoutThreads)
    {
        int const expert = topkIndices[i];
        if (expert >= 0 && expert < kNumExperts)
        {
            atomicAdd_block(&count[expert], 1);
        }
    }
    __syncthreads();
    // Warp 0: exclusive scan of padded rows and tiles over experts (<= 16 experts per lane).
    if (tid < kWarpSize)
    {
        constexpr int kPerLane{(kNumExperts + kWarpSize - 1) / kWarpSize};
        int rows = 0;
        int tiles = 0;
        for (int j = 0; j < kPerLane; ++j)
        {
            int const e = tid * kPerLane + j;
            if (e < kNumExperts)
            {
                int const t = (count[e] + tokenTile - 1) / tokenTile;
                rows += t * tokenTile;
                tiles += t;
            }
        }
        int rowsIncl = rows;
        int tilesIncl = tiles;
#pragma unroll
        for (int offset = 1; offset < kWarpSize; offset <<= 1)
        {
            int const r = __shfl_up_sync(0xFFFFFFFFU, rowsIncl, offset);
            int const t = __shfl_up_sync(0xFFFFFFFFU, tilesIncl, offset);
            if (tid >= offset)
            {
                rowsIncl += r;
                tilesIncl += t;
            }
        }
        int rowBase = rowsIncl - rows;
        int tileBase = tilesIncl - tiles;
        for (int j = 0; j < kPerLane; ++j)
        {
            int const e = tid * kPerLane + j;
            if (e < kNumExperts)
            {
                rowOffset[e] = rowBase;
                tileOffset[e] = tileBase;
                int const t = (count[e] + tokenTile - 1) / tokenTile;
                rowBase += t * tokenTile;
                tileBase += t;
            }
        }
        if (tid == kWarpSize - 1)
        {
            numValidTiles[0] = tilesIncl;
        }
    }
    __syncthreads();
    // Tile owners and pad rows, one expert per thread.
    for (int e = tid; e < kNumExperts; e += kLayoutThreads)
    {
        int const c = count[e];
        int const t = (c + tokenTile - 1) / tokenTile;
        for (int i = 0; i < t; ++i)
        {
            tileGroupIdx[tileOffset[e] + i] = e;
        }
        for (int r = c; r < t * tokenTile; ++r)
        {
            permutedIdx[rowOffset[e] + r] = -1;
        }
    }
    // Scatter slots into their expert's row range.
    for (int i = tid; i < numSlots; i += kLayoutThreads)
    {
        int const expert = topkIndices[i];
        if (expert >= 0 && expert < kNumExperts)
        {
            int const pos = atomicAdd_block(&cursor[expert], 1);
            permutedIdx[rowOffset[expert] + pos] = i;
        }
    }
}

//! Gathers the routed activation rows into the expert-contiguous permuted buffer
//! (blocks [0, maxRowsPadded)) and zeroes the layer output (blocks
//! [maxRowsPadded, maxRowsPadded + T)) for the FC2 scatter-add.
extern "C" __global__ __launch_bounds__(kThreadsPerBlock) void nvfp4_a16_blackwell_moe_gather(
    ActT const* __restrict__ hidden, int const* __restrict__ permutedIdx, int const* __restrict__ numValidTiles,
    int const tokenTile, int const numTokens, long long const maxRowsPadded, ActT* __restrict__ permuted,
    ActT* __restrict__ output)
{
    constexpr int kElementsPerVector{static_cast<int>(16 / sizeof(ActT))};
    constexpr int kVectorsPerRow{kHidden / kElementsPerVector};
    long long const block = static_cast<long long>(blockIdx.x);
    uint4 const zero{0U, 0U, 0U, 0U};
    // permutedIdx / numValidTiles come from the layout kernel and the output rows
    // may still alias memory the predecessor uses: nothing may be read or written
    // before the grid dependency resolves.
    pdlWait();
    pdlTrigger();
    if (block < maxRowsPadded)
    {
        long long const validRows = static_cast<long long>(numValidTiles[0]) * tokenTile;
        // Rows past the valid tiles and padding rows need no fill: FC1 produces a
        // discarded row from whatever is there and the FC2 epilogue skips it
        // through permuted_idx.
        int const expanded = block < validRows ? permutedIdx[block] : -1;
        if (expanded >= 0)
        {
            uint4* const dst = reinterpret_cast<uint4*>(permuted + block * kHidden);
            int const token = expanded / kTopK;
            uint4 const* const src = reinterpret_cast<uint4 const*>(hidden + static_cast<long long>(token) * kHidden);
            for (int v = static_cast<int>(threadIdx.x); v < kVectorsPerRow; v += kThreadsPerBlock)
            {
                dst[v] = src[v];
            }
        }
    }
    else
    {
        long long const token = block - maxRowsPadded;
        if (token < numTokens)
        {
            uint4* const dst = reinterpret_cast<uint4*>(output + token * kHidden);
            for (int v = static_cast<int>(threadIdx.x); v < kVectorsPerRow; v += kThreadsPerBlock)
            {
                dst[v] = zero;
            }
        }
    }
}

//! FC1: one 128-row tile of relu(alpha * x[token] . W1[expert])^2 for one
//! (token, slot) row over this CTA's K range, or an fp32 partial when split-K > 1.
//! The activation row belongs to the previous layer, which had completed before
//! the routing kernel passed its own wait, so it is staged before this kernel's
//! wait; only the expert id (routing output) has to wait.
extern "C" __global__ __launch_bounds__(kThreadsPerBlock, kFc1CtasPerSm) void nvfp4_a16_blackwell_moe_fc1(
    ActT const* __restrict__ hidden, int const* __restrict__ topkIndices, unsigned char const* __restrict__ qweights,
    unsigned char const* __restrict__ blockScales, float const* __restrict__ globalScales,
    ActT* __restrict__ fc1Output, float* __restrict__ partials, int const numTokens)
{
    __shared__ __align__(16) float sharedActivation[kFc1TilesPerSplit * kSmemRowStride];
    int const slot = static_cast<int>(blockIdx.y);
    int const token = slot / kTopK;
    int const split = static_cast<int>(blockIdx.z);
    int const kBlockBegin = kHiddenKBlocks * split / kFc1SplitK;
    int const kBlockEnd = kHiddenKBlocks * (split + 1) / kFc1SplitK;
    ActT const* const activationRow = hidden + static_cast<long long>(token) * kHidden;
    stageActivationRange(activationRow, kBlockBegin, kBlockEnd, sharedActivation);
    pdlWait();
    pdlTrigger();
    int const expert = topkIndices[slot];
    __syncthreads();
    int const lane = static_cast<int>(threadIdx.x) & (kWarpSize - 1);
    int const warp = static_cast<int>(threadIdx.x) / kWarpSize;
    int const rowInTile = warp * kRowsPerWarp + lane / 2;
    int const kHalf = lane & 1;
    int const nBlock = static_cast<int>(blockIdx.x);
    unsigned char const* const expertCodes = qweights + static_cast<long long>(expert) * kFc1ExpertPlaneCodes;
    unsigned char const* const expertScales = blockScales + static_cast<long long>(expert) * (kFc1ExpertPlaneCodes / 8);
    long long const rowTileBase = static_cast<long long>(nBlock) * kHiddenKBlocks * kNTile + rowInTile;
    float acc = streamRowRange(expertCodes, expertScales, rowTileBase, kBlockBegin, kBlockEnd, kHalf, sharedActivation);
    acc += __shfl_xor_sync(0xFFFFFFFFU, acc, 1, kWarpSize);
    if (kHalf == 0)
    {
        int const n = nBlock * kNTile + rowInTile;
        if (kFc1SplitK == 1)
        {
            float const scaled = fmaxf(acc * globalScales[expert], 0.0f);
            fc1Output[static_cast<long long>(slot) * kInterPad + n] = ActTraits<ActT>::fromFloat(scaled * scaled);
        }
        else
        {
            partials[(static_cast<long long>(split) * (static_cast<long long>(numTokens) * kTopK) + slot) * kInterPad
                + n]
                = acc;
        }
    }
}

//! FC1 split-K finalize: sum partials, apply alpha[expert], relu^2, narrow.
extern "C" __global__ __launch_bounds__(kThreadsPerBlock) void nvfp4_a16_blackwell_moe_fc1_reduce(
    int const* __restrict__ topkIndices, float const* __restrict__ globalScales, float const* __restrict__ partials,
    ActT* __restrict__ fc1Output, int const numTokens)
{
    long long const pairIndex = static_cast<long long>(blockIdx.x) * kThreadsPerBlock + threadIdx.x;
    long long const numSlots = static_cast<long long>(numTokens) * kTopK;
    long long const numPairs = numSlots * kInterPad / 2;
    bool const active = pairIndex < numPairs;
    long long const elementIndex = pairIndex * 2;
    // Routing results are complete and visible (FC1 waited before it triggered);
    // only the partials need this kernel's wait.
    float alpha = 0.0f;
    if (active)
    {
        int const slot = static_cast<int>(elementIndex / kInterPad);
        alpha = globalScales[topkIndices[slot]];
    }
    pdlWait();
    pdlTrigger();
    if (active)
    {
        float2 sum{0.0f, 0.0f};
#pragma unroll
        for (int split = 0; split < kFc1SplitK; ++split)
        {
            float2 const v = *reinterpret_cast<float2 const*>(
                partials + static_cast<long long>(split) * numSlots * kInterPad + elementIndex);
            sum.x += v.x;
            sum.y += v.y;
        }
        float const a = fmaxf(sum.x * alpha, 0.0f);
        float const b = fmaxf(sum.y * alpha, 0.0f);
        ActTraits<ActT>::storePair(fc1Output + elementIndex, make_float2(a * a, b * b));
    }
}

//! FC2: each CTA loops over the token's top-k slots and accumulates
//! alpha[e] * w[token, slot] * a_slot . W2[e] for its 128 hidden features in fp32,
//! then writes the token output (or an fp32 partial when split-K > 1).  No atomics.
//! Before the wait the first kFc2PrefetchSlots slots' weight tiles (this CTA's K
//! range, contiguous in the layout) are copied into shared memory with cp.async:
//! their expert ids are routing results, complete and visible because FC1 (and its
//! reduce) waited before they triggered; FC1's output must wait.
extern "C" __global__ __launch_bounds__(kThreadsPerBlock, kFc2CtasPerSm) void nvfp4_a16_blackwell_moe_fc2(
    ActT const* __restrict__ fc1Output, int const* __restrict__ topkIndices, float const* __restrict__ topkWeights,
    unsigned char const* __restrict__ qweights, unsigned char const* __restrict__ blockScales,
    float const* __restrict__ globalScales, ActT* __restrict__ output, float* __restrict__ partials,
    int const numTokens)
{
    __shared__ __align__(16) float sharedActivation[kTopK * kFc2TilesPerSplit * kSmemRowStride];
    __shared__ __align__(16) unsigned char sCodes[kFc2PrefetchSlots * kFc2TilesPerSplit * kCodeBytesPerTile + 16];
    __shared__ __align__(16) unsigned char sScales[kFc2PrefetchSlots * kFc2TilesPerSplit * kScaleBytesPerTile + 16];
    int const token = static_cast<int>(blockIdx.y);
    int const nBlock = static_cast<int>(blockIdx.x);
    int const split = static_cast<int>(blockIdx.z);
    int const lane = static_cast<int>(threadIdx.x) & (kWarpSize - 1);
    int const warp = static_cast<int>(threadIdx.x) / kWarpSize;
    int const rowInTile = warp * kRowsPerWarp + lane / 2;
    int const kHalf = lane & 1;
    int const kBlockBegin = kInterKBlocks * split / kFc2SplitK;
    int const kBlockEnd = kInterKBlocks * (split + 1) / kFc2SplitK;
    int const numTiles = kBlockEnd - kBlockBegin;
    int const tid = static_cast<int>(threadIdx.x);
    // This CTA's K range of one expert is contiguous in the layout: row tiles
    // (nBlock * kBlocks + kBlock) * 128 .. for kBlock in [begin, end).
    long long const rangeTileBase = (static_cast<long long>(nBlock) * kInterKBlocks + kBlockBegin) * kNTile;
#pragma unroll
    for (int slotInToken = 0; slotInToken < kFc2PrefetchSlots; ++slotInToken)
    {
        int const expert = topkIndices[static_cast<long long>(token) * kTopK + slotInToken];
        unsigned char const* const codes
            = qweights + static_cast<long long>(expert) * kFc2ExpertPlaneCodes + rangeTileBase * kPackedBytesPerRowTile;
        unsigned char const* const scales
            = blockScales + static_cast<long long>(expert) * (kFc2ExpertPlaneCodes / 8) + rangeTileBase * kScalesPerRowTile;
        unsigned char* const dstCodes = sCodes + slotInToken * kFc2TilesPerSplit * kCodeBytesPerTile;
        unsigned char* const dstScales = sScales + slotInToken * kFc2TilesPerSplit * kScaleBytesPerTile;
        for (int v = tid; v < numTiles * (kCodeBytesPerTile / 16); v += kThreadsPerBlock)
        {
            cpAsync16(dstCodes + v * 16, codes + v * 16);
        }
        for (int v = tid; v < numTiles * (kScaleBytesPerTile / 16); v += kThreadsPerBlock)
        {
            cpAsync16(dstScales + v * 16, scales + v * 16);
        }
    }
    if (kFc2PrefetchSlots > 0)
    {
        cpAsyncCommit();
    }
    pdlWait();
    pdlTrigger();
    for (int slotInToken = 0; slotInToken < kTopK; ++slotInToken)
    {
        long long const slot = static_cast<long long>(token) * kTopK + slotInToken;
        stageActivationRange(fc1Output + slot * kInterPad, kBlockBegin, kBlockEnd,
            sharedActivation + slotInToken * kFc2TilesPerSplit * kSmemRowStride);
    }
    if (kFc2PrefetchSlots > 0)
    {
        cpAsyncWaitAll();
    }
    __syncthreads();
    long long const rowTileBase = static_cast<long long>(nBlock) * kInterKBlocks * kNTile + rowInTile;
    float total = 0.0f;
    for (int slotInToken = 0; slotInToken < kTopK; ++slotInToken)
    {
        long long const slot = static_cast<long long>(token) * kTopK + slotInToken;
        int const expert = topkIndices[slot];
        float const weight = topkWeights[slot] * globalScales[expert];
        float const* const stagedRange = sharedActivation + slotInToken * kFc2TilesPerSplit * kSmemRowStride;
        float acc = 0.0f;
        if (slotInToken < kFc2PrefetchSlots)
        {
            acc = dotStagedRowRange(sCodes + slotInToken * kFc2TilesPerSplit * kCodeBytesPerTile,
                sScales + slotInToken * kFc2TilesPerSplit * kScaleBytesPerTile, numTiles, rowInTile, kHalf, stagedRange);
        }
        else
        {
            unsigned char const* const expertCodes = qweights + static_cast<long long>(expert) * kFc2ExpertPlaneCodes;
            unsigned char const* const expertScales
                = blockScales + static_cast<long long>(expert) * (kFc2ExpertPlaneCodes / 8);
            acc = streamRowRange(expertCodes, expertScales, rowTileBase, kBlockBegin, kBlockEnd, kHalf, stagedRange);
        }
        total = fmaf(weight, acc, total);
    }
    total += __shfl_xor_sync(0xFFFFFFFFU, total, 1, kWarpSize);
    if (kHalf == 0)
    {
        int const n = nBlock * kNTile + rowInTile;
        if (kFc2SplitK == 1)
        {
            output[static_cast<long long>(token) * kHidden + n] = ActTraits<ActT>::fromFloat(total);
        }
        else
        {
            partials[(static_cast<long long>(split) * numTokens + token) * kHidden + n] = total;
        }
    }
}

//! FC2 split-K finalize: sum partials, narrow.
extern "C" __global__ __launch_bounds__(kThreadsPerBlock) void nvfp4_a16_blackwell_moe_fc2_reduce(
    float const* __restrict__ partials, ActT* __restrict__ output, int const numTokens)
{
    long long const pairIndex = static_cast<long long>(blockIdx.x) * kThreadsPerBlock + threadIdx.x;
    long long const numElements = static_cast<long long>(numTokens) * kHidden;
    pdlWait(); // partials from FC2 (nothing of this kernel's inputs predates them)
    pdlTrigger();
    if (pairIndex * 2 < numElements)
    {
        long long const elementIndex = pairIndex * 2;
        float2 sum{0.0f, 0.0f};
#pragma unroll
        for (int split = 0; split < kFc2SplitK; ++split)
        {
            float2 const v
                = *reinterpret_cast<float2 const*>(partials + static_cast<long long>(split) * numElements + elementIndex);
            sum.x += v.x;
            sum.y += v.y;
        }
        ActTraits<ActT>::storePair(output + elementIndex, sum);
    }
}
