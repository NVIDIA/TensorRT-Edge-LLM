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

#include "common/checkMacros.h"
#include "common/cudaMacros.h"
#include "common/cudaUtils.h"
#include "common/tensor.h"
#include "kernels/speculative/lowRankCandidateHeadKernels.h"
#include "kernels/speculative/speculativeKernelsUtils.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#if SUPPORTS_CLUSTER_LAUNCH
#include <cooperative_groups.h>
#endif

#include <algorithm>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace trt_edgellm
{
namespace kernel
{

//! Steps from the device's preferred configuration towards smaller workspaces (see shrinkingOverrides).
struct LowRankCandidateHeadOverrides
{
    int32_t grid{0};            //!< P for every bucket, at least ceil(V / 65536); 0 keeps the device choice.
    int32_t capacity{0};        //!< CAP for every bucket, a multiple of 32; 0 keeps the device choice.
    bool pilot{true};           //!< false runs no pilot bound in any bucket.
    bool fewerWitnesses{false}; //!< 2 pilot witnesses per candidate instead of the device choice (2 or 4).
};

namespace
{

constexpr int32_t kLrhWarpSize = 32;
constexpr uint32_t kLrhFullMask = 0xFFFFFFFFU;
constexpr int32_t kLrhMaxRank = 256;
constexpr int32_t kLrhMaxCandidates = 256;
constexpr int32_t kLrhProjectionWarps = 4;
constexpr int32_t kLrhProjectionThreads = kLrhProjectionWarps * kLrhWarpSize;
constexpr int32_t kLrhReduceThreads = 256;
constexpr int32_t kMergeThreads = 256;
constexpr int32_t kMergeWarps = kMergeThreads / kLrhWarpSize;
//! Rows per projection launch (eight MMA n-tiles).
constexpr int32_t kLrhProjectionMaxRows = LowRankCandidateHeadRunner::kMaxRowsPerPass;
constexpr int32_t kLrhMmaM = 16;
constexpr int32_t kLrhMmaN = 8;
constexpr int32_t kLrhMmaK = 16;
constexpr int32_t kLrhHalvesPerVector = 8;
//! Halves covered by one 16-byte load per lane quad: two MMA k-steps under the fixed k-permutation below.
constexpr int32_t kLrhKPair = 2 * kLrhMmaK;
//! Orderable value of -inf. Non-finite exact logits sort with this value, above the zero padding key.
constexpr uint32_t kLrhNegInfOrderable = 0x007FFFFFU;

//! Unsigned image of an fp32 score whose order equals the float order. NaN maps to 0 (below -inf) and -0 to +0.
__device__ __forceinline__ uint32_t lrhOrderableScore(float score)
{
    uint32_t bits = __float_as_uint(score);
    if ((bits & 0x7FFFFFFFU) > 0x7F800000U)
    {
        return 0U;
    }
    if (bits == 0x80000000U)
    {
        bits = 0U;
    }
    return (bits & 0x80000000U) != 0U ? ~bits : (bits | 0x80000000U);
}

//! Exact-logit sort value: every non-finite logit counts as -inf.
__device__ __forceinline__ uint32_t lrhOrderableLogit(float logit)
{
    bool const finite = (__float_as_uint(logit) & 0x7F800000U) != 0x7F800000U;
    return finite ? lrhOrderableScore(logit) : kLrhNegInfOrderable;
}

__device__ __forceinline__ float lrhScoreFromOrderable(uint32_t orderable)
{
    uint32_t const bits = (orderable & 0x80000000U) != 0U ? (orderable & 0x7FFFFFFFU) : ~orderable;
    return __uint_as_float(bits);
}

//! (score, id) key: unsigned order is score descending, then id ascending. Real keys are never 0.
__device__ __forceinline__ uint64_t lrhPackKey(uint32_t orderable, uint32_t id)
{
    return (static_cast<uint64_t>(orderable) << 32) | static_cast<uint64_t>(0xFFFFFFFFU - id);
}

__device__ __forceinline__ int32_t lrhKeyId(uint64_t key)
{
    return static_cast<int32_t>(0xFFFFFFFFU - static_cast<uint32_t>(key & 0xFFFFFFFFULL));
}

//! Xor-butterfly maximum over the lanes whose indices differ only in bits at or above minOffset (the whole warp by
//! default); each of those lanes gets the result.
__device__ __forceinline__ uint64_t lrhWarpMax(uint64_t value, int32_t minOffset = 1)
{
#pragma unroll
    for (int32_t offset = kLrhWarpSize / 2; offset >= minOffset; offset >>= 1)
    {
        uint64_t const other = __shfl_xor_sync(kLrhFullMask, static_cast<unsigned long long>(value), offset);
        value = other > value ? other : value;
    }
    return value;
}

__device__ __forceinline__ uint64_t lrhWarpMin(uint64_t value)
{
#pragma unroll
    for (int32_t offset = kLrhWarpSize / 2; offset > 0; offset >>= 1)
    {
        uint64_t const other = __shfl_xor_sync(kLrhFullMask, static_cast<unsigned long long>(value), offset);
        value = other < value ? other : value;
    }
    return value;
}

//! Xor-butterfly warp sum: every lane gets the same result, summed in the same order.
template <typename T>
__device__ __forceinline__ T lrhWarpSum(T value)
{
#pragma unroll
    for (int32_t offset = kLrhWarpSize / 2; offset > 0; offset >>= 1)
    {
        value += __shfl_xor_sync(kLrhFullMask, value, offset);
    }
    return value;
}

//! Sorts one value per lane in descending lane order (bitonic network over the warp).
template <typename T>
__device__ __forceinline__ T lrhWarpSortDescending(T value)
{
    int32_t const lane = static_cast<int32_t>(threadIdx.x) % kLrhWarpSize;
#pragma unroll
    for (int32_t k = 2; k <= kLrhWarpSize; k <<= 1)
    {
#pragma unroll
        for (int32_t j = k >> 1; j > 0; j >>= 1)
        {
            T const other = __shfl_xor_sync(kLrhFullMask, value, j);
            bool const keepMax = ((lane & j) == 0) == ((lane & k) == 0);
            value = keepMax ? max(value, other) : min(value, other);
        }
    }
    return value;
}

//! D += A x B for one m16n8k16 tile (f16 inputs, f32 accumulation).
__device__ __forceinline__ void lrhMma16816(float (&acc)[4], uint32_t const (&a)[4], uint32_t const (&b)[2])
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
        "{%0,%1,%2,%3};\n"
        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
#endif
}

//! Two m16n8k16 steps from 16-byte loads under the fixed k-permutation used by K0 and K1: lane (g, t) holds halves
//! [8t, 8t + 8) of a 32-wide k block for rows g (rowLo) and g + 8 (rowHi) and for its batch column (b). Step 0 uses
//! halves 8t + {0..3}, step 1 halves 8t + {4..7}; A and B use the same mapping, so the pair accumulates the full
//! 32-term dot product in a fixed order that does not depend on R.
__device__ __forceinline__ void lrhMmaKPair(float (&acc)[4], uint4 const& rowLo, uint4 const& rowHi, uint4 const& b)
{
    uint32_t const a0[4] = {rowLo.x, rowHi.x, rowLo.y, rowHi.y};
    uint32_t const b0[2] = {b.x, b.y};
    lrhMma16816(acc, a0, b0);
    uint32_t const a1[4] = {rowLo.z, rowHi.z, rowLo.w, rowHi.w};
    uint32_t const b1[2] = {b.z, b.w};
    lrhMma16816(acc, a1, b1);
}

//! Whether lane quad t's 8-half chunk of k-pair kp (halves [32 kp + 8t, + 8)) lies inside the rank. rank is a
//! multiple of 16, so a chunk is either fully inside or fully outside.
__device__ __forceinline__ bool lrhKPairChunkInRank(int32_t kp, int32_t pairId, int32_t rank)
{
    return kp * kLrhKPair + kLrhHalvesPerVector * pairId < rank;
}

//! s fragments for lrhAccumulateScoreTile: sFrag[nt][kp] = loadS(nt, column, inRank), where inRank says whether
//! the chunk lies inside the rank; loadS returns zeros past the rank and for batch rows that do not exist.
template <int32_t kTiles, int32_t kMaxKPairs, bool kFullRank, class LoadS>
__device__ __forceinline__ void lrhLoadScoreFragments(
    uint4 (&sFrag)[kTiles][kMaxKPairs], int32_t rank, int32_t pairId, LoadS const& loadS)
{
#pragma unroll
    for (int32_t nt = 0; nt < kTiles; ++nt)
    {
#pragma unroll
        for (int32_t kp = 0; kp < kMaxKPairs; ++kp)
        {
            sFrag[nt][kp] = loadS(
                nt, kp * kLrhKPair + kLrhHalvesPerVector * pairId, kFullRank || lrhKPairChunkInRank(kp, pairId, rank));
        }
    }
}

//! Adds the low-rank scores of 16 vocab rows against kTiles x 8 batch rows to acc: acc[nt] is the m16n8 accumulator
//! fragment of A rows g and g + 8 against the s rows of sFrag[nt], accumulated with lrhMmaKPair over the k-pairs in
//! order. loadA(kp, inRank, lo, hi) loads the two A chunks of k-pair kp, zeros when !inRank. The selection kernel
//! and the pilot both start acc at zero and score with this function: a pilot witness is a valid lower bound only
//! because it is bitwise equal to the selection kernel's score of the same key.
template <int32_t kTiles, int32_t kMaxKPairs, bool kFullRank, class LoadA>
__device__ __forceinline__ void lrhAccumulateScoreTile(
    float (&acc)[kTiles][4], uint4 const (&sFrag)[kTiles][kMaxKPairs], int32_t rank, int32_t pairId, LoadA const& loadA)
{
    int32_t const kPairs = kFullRank ? kMaxKPairs : (rank + kLrhKPair - 1) / kLrhKPair;
#pragma unroll
    for (int32_t kp = 0; kp < kMaxKPairs; ++kp)
    {
        if (kFullRank || kp < kPairs)
        {
            uint4 rowLo;
            uint4 rowHi;
            loadA(kp, kFullRank || lrhKPairChunkInRank(kp, pairId, rank), rowLo, rowHi);
#pragma unroll
            for (int32_t nt = 0; nt < kTiles; ++nt)
            {
                lrhMmaKPair(acc[nt], rowLo, rowHi, sFrag[nt][kp]);
            }
        }
    }
}

__device__ __forceinline__ void lrhTriggerDependents()
{
#if SUPPORTS_PROGRAMMATIC_DEPENDENT_LAUNCH && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory");
#endif
}

__device__ __forceinline__ void lrhWaitPrerequisites()
{
#if SUPPORTS_PROGRAMMATIC_DEPENDENT_LAUNCH && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.wait;\n" ::: "memory");
#endif
}

//! 16-byte load of a kernel input that no grid writes during the call (read-only path).
__device__ __forceinline__ uint4 lrhLoad16(half const* ptr, bool valid)
{
    return valid ? __ldg(reinterpret_cast<uint4 const*>(ptr)) : make_uint4(0U, 0U, 0U, 0U);
}

//! 16-byte load of data written by a prerequisite grid. Under PDL the dependent grid starts before the producer
//! writes, so the read-only (non-coherent) path could return stale lines; this load goes through L2.
__device__ __forceinline__ uint4 lrhLoad16Produced(half const* ptr, bool valid)
{
    return valid ? __ldcg(reinterpret_cast<uint4 const*>(ptr)) : make_uint4(0U, 0U, 0U, 0U);
}

//! K0a: partial s for one 16-row block of downProj and one contiguous k range (split blockIdx.y of gridDim.y). The
//! k partition depends only on hiddenSize and the split count, and the four warp partials are summed in warp order.
__global__ void __launch_bounds__(kLrhProjectionThreads) lrhProjectionPartialKernel(half const* __restrict__ hidden,
    half const* __restrict__ downProj, float* __restrict__ partials, int32_t rows, int32_t hiddenSize, int32_t rank)
{
    constexpr int32_t kMaxTiles = kLrhProjectionMaxRows / kLrhMmaN;
    __shared__ float warpPartial[kLrhProjectionWarps][kLrhMmaM][kLrhProjectionMaxRows];

    if (threadIdx.x == 0)
    {
        lrhTriggerDependents();
    }
    int32_t const warp = static_cast<int32_t>(threadIdx.x) / kLrhWarpSize;
    int32_t const lane = static_cast<int32_t>(threadIdx.x) % kLrhWarpSize;
    int32_t const groupId = lane >> 2;
    int32_t const pairId = lane & 3;
    int32_t const mBase = static_cast<int32_t>(blockIdx.x) * kLrhMmaM;
    int32_t const split = static_cast<int32_t>(blockIdx.y);
    int32_t const splits = static_cast<int32_t>(gridDim.y);
    int32_t const nTiles = (rows + kLrhMmaN - 1) / kLrhMmaN;
    int32_t const kPairs = (hiddenSize + kLrhKPair - 1) / kLrhKPair;
    int32_t const kpBegin = split * kPairs / splits;
    int32_t const kpEnd = (split + 1) * kPairs / splits;

    float acc[kMaxTiles][4] = {};
    half const* const bRowLo = downProj + static_cast<int64_t>(mBase + groupId) * hiddenSize;
    half const* const bRowHi = bRowLo + static_cast<int64_t>(kLrhMmaM / 2) * hiddenSize;
#pragma unroll 2
    for (int32_t kp = kpBegin + warp; kp < kpEnd; kp += kLrhProjectionWarps)
    {
        // hiddenSize % 8 == 0, so a lane's 8-half chunk is either fully in range or fully out of range (zeros).
        int32_t const col = kp * kLrhKPair + kLrhHalvesPerVector * pairId;
        bool const colValid = col < hiddenSize;
        uint4 const rowLo = lrhLoad16(bRowLo + col, colValid);
        uint4 const rowHi = lrhLoad16(bRowHi + col, colValid);
        uint4 b[kMaxTiles];
#pragma unroll
        for (int32_t nt = 0; nt < kMaxTiles; ++nt)
        {
            int32_t const n = nt * kLrhMmaN + groupId;
            b[nt] = lrhLoad16(hidden + static_cast<int64_t>(n) * hiddenSize + col, colValid && nt < nTiles && n < rows);
        }
#pragma unroll
        for (int32_t nt = 0; nt < kMaxTiles; ++nt)
        {
            if (nt < nTiles)
            {
                lrhMmaKPair(acc[nt], rowLo, rowHi, b[nt]);
            }
        }
    }
#pragma unroll
    for (int32_t nt = 0; nt < kMaxTiles; ++nt)
    {
        if (nt < nTiles)
        {
            int32_t const n = nt * kLrhMmaN + 2 * pairId;
            warpPartial[warp][groupId][n] = acc[nt][0];
            warpPartial[warp][groupId][n + 1] = acc[nt][1];
            warpPartial[warp][groupId + kLrhMmaM / 2][n] = acc[nt][2];
            warpPartial[warp][groupId + kLrhMmaM / 2][n + 1] = acc[nt][3];
        }
    }
    __syncthreads();
    float* const out = partials + static_cast<int64_t>(split) * rows * rank;
    for (int32_t o = static_cast<int32_t>(threadIdx.x); o < kLrhMmaM * rows; o += kLrhProjectionThreads)
    {
        int32_t const n = o / kLrhMmaM;
        int32_t const m = o % kLrhMmaM;
        float sum = 0.0F;
#pragma unroll
        for (int32_t w = 0; w < kLrhProjectionWarps; ++w)
        {
            sum += warpPartial[w][m][n];
        }
        out[static_cast<int64_t>(n) * rank + mBase + m] = sum;
    }
}

//! K0b: s = fp16(sum of the split partials in split order). Also zeroes the per-row thresholds and tickets, so the
//! head keeps no state between calls.
__global__ void __launch_bounds__(kLrhReduceThreads)
    lrhProjectionReduceKernel(float const* __restrict__ partials, half* __restrict__ projected, int32_t rows,
        int32_t rank, int32_t splits, unsigned long long* __restrict__ thresholds, uint32_t* __restrict__ tickets)
{
    if (threadIdx.x == 0)
    {
        lrhTriggerDependents();
    }
    lrhWaitPrerequisites();
    int32_t const count = rows * rank;
    int32_t const o = static_cast<int32_t>(blockIdx.x) * kLrhReduceThreads + static_cast<int32_t>(threadIdx.x);
    if (o < count)
    {
        float sum = 0.0F;
        for (int32_t split = 0; split < splits; ++split)
        {
            sum += __ldcg(partials + static_cast<int64_t>(split) * count + o);
        }
        projected[o] = __float2half_rn(sum);
    }
    if (blockIdx.x == 0)
    {
        for (int32_t i = static_cast<int32_t>(threadIdx.x); i < rows; i += kLrhReduceThreads)
        {
            thresholds[i] = 0ULL;
            tickets[i] = 0U;
        }
    }
}

template <typename Kernel, typename... Args>
void lrhLaunch(Kernel kernel, dim3 grid, dim3 block, size_t sharedBytes, cudaStream_t stream, bool usePdl,
    int32_t clusterY, Args... args)
{
#if SUPPORTS_PROGRAMMATIC_DEPENDENT_LAUNCH || SUPPORTS_CLUSTER_LAUNCH
    cudaLaunchAttribute attributes[2]{};
    uint32_t count = 0;
    if (usePdl)
    {
        attributes[count].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attributes[count].val.programmaticStreamSerializationAllowed = 1;
        ++count;
    }
    if (clusterY > 1)
    {
        attributes[count].id = cudaLaunchAttributeClusterDimension;
        attributes[count].val.clusterDim.x = 1;
        attributes[count].val.clusterDim.y = static_cast<uint32_t>(clusterY);
        attributes[count].val.clusterDim.z = 1;
        ++count;
    }
    cudaLaunchConfig_t launch{};
    launch.gridDim = grid;
    launch.blockDim = block;
    launch.dynamicSmemBytes = sharedBytes;
    launch.stream = stream;
    launch.attrs = count > 0 ? attributes : nullptr;
    launch.numAttrs = count;
    CUDA_CHECK(cudaLaunchKernelEx(&launch, kernel, args...));
#else
    (void) usePdl;
    (void) clusterY;
    kernel<<<grid, block, sharedBytes, stream>>>(args...);
    CUDA_CHECK(cudaPeekAtLastError());
#endif
}

__device__ __forceinline__ float lrhDot8(uint4 const& w, uint4 const& h, float acc)
{
    uint32_t const wWords[4] = {w.x, w.y, w.z, w.w};
    uint32_t const hWords[4] = {h.x, h.y, h.z, h.w};
#pragma unroll
    for (int32_t i = 0; i < 4; ++i)
    {
        float2 const wf = __half22float2(*reinterpret_cast<half2 const*>(&wWords[i]));
        float2 const hf = __half22float2(*reinterpret_cast<half2 const*>(&hWords[i]));
        acc = __fmaf_rn(wf.x, hf.x, acc);
        acc = __fmaf_rn(wf.y, hf.y, acc);
    }
    return acc;
}

//! Number of fixed hidden-size segments per exact logit. Each segment is one warp work item, so a candidate's row
//! of lmHead is read by four warps in parallel.
constexpr int32_t kLrhGatherSegments = 4;

//! Exact logits lmHead[ids[i]] . hiddenRow for numIds candidates of one row, computed by every warp of the block
//! (all threads call it; it ends with a block barrier). A logit is the sum, in segment order, of four segment dot
//! products; each segment accumulates lane-strided 8-element chunks in fp32 and ends with an xor butterfly. The
//! result depends only on the lmHead row and hiddenRow, never on kInflight, the block or the other candidates.
//! ids: numIds entries; partials: kLrhGatherSegments * numIds floats of shared memory; logits: numIds outputs.
__device__ void lrhBlockGatherDot(half const* __restrict__ lmHead, int32_t hiddenSize, int32_t const* ids,
    int32_t numIds, half const* hiddenRow, float* partials, float* logits)
{
    // Work items in flight per warp.
    constexpr int32_t kInflight = 2;
    // Chunks per lane loaded together for each item; larger hidden sizes take several rounds.
    constexpr int32_t kChunksPerRound = 6;
    int32_t const warp = static_cast<int32_t>(threadIdx.x) / kLrhWarpSize;
    int32_t const lane = static_cast<int32_t>(threadIdx.x) % kLrhWarpSize;
    int32_t const warps = static_cast<int32_t>(blockDim.x) / kLrhWarpSize;
    int32_t const chunks = hiddenSize / kLrhHalvesPerVector;
    int32_t const items = numIds * kLrhGatherSegments;
    int32_t const maxLen = (chunks + kLrhGatherSegments - 1) / kLrhGatherSegments;
    for (int32_t base = warp * kInflight; base < items; base += warps * kInflight)
    {
        float acc[kInflight];
        int64_t rowOffset[kInflight];
        int32_t segBegin[kInflight];
        int32_t segEnd[kInflight];
#pragma unroll
        for (int32_t i = 0; i < kInflight; ++i)
        {
            int32_t const item = min(base + i, items - 1);
            int32_t const seg = item % kLrhGatherSegments;
            acc[i] = 0.0F;
            rowOffset[i] = static_cast<int64_t>(ids[item / kLrhGatherSegments]) * hiddenSize;
            segBegin[i] = seg * chunks / kLrhGatherSegments;
            segEnd[i] = base + i < items ? (seg + 1) * chunks / kLrhGatherSegments : segBegin[i];
        }
        for (int32_t round = 0; round < maxLen; round += kChunksPerRound * kLrhWarpSize)
        {
            uint4 w[kInflight][kChunksPerRound];
#pragma unroll
            for (int32_t i = 0; i < kInflight; ++i)
            {
#pragma unroll
                for (int32_t r = 0; r < kChunksPerRound; ++r)
                {
                    int32_t const c = segBegin[i] + round + lane + kLrhWarpSize * r;
                    w[i][r] = c < segEnd[i] ? __ldg(reinterpret_cast<uint4 const*>(lmHead + rowOffset[i]
                                                  + static_cast<int64_t>(c) * kLrhHalvesPerVector))
                                            : make_uint4(0U, 0U, 0U, 0U);
                }
            }
#pragma unroll
            for (int32_t i = 0; i < kInflight; ++i)
            {
#pragma unroll
                for (int32_t r = 0; r < kChunksPerRound; ++r)
                {
                    int32_t const c = segBegin[i] + round + lane + kLrhWarpSize * r;
                    if (c < segEnd[i])
                    {
                        uint4 const hv = *reinterpret_cast<uint4 const*>(
                            hiddenRow + static_cast<int64_t>(c) * kLrhHalvesPerVector);
                        acc[i] = lrhDot8(w[i][r], hv, acc[i]);
                    }
                }
            }
        }
#pragma unroll
        for (int32_t i = 0; i < kInflight; ++i)
        {
            acc[i] = lrhWarpSum(acc[i]);
            if (lane == 0 && base + i < items)
            {
                partials[base + i] = acc[i];
            }
        }
    }
    __syncthreads();
    for (int32_t i = static_cast<int32_t>(threadIdx.x); i < numIds; i += static_cast<int32_t>(blockDim.x))
    {
        float sum = partials[i * kLrhGatherSegments];
#pragma unroll
        for (int32_t seg = 1; seg < kLrhGatherSegments; ++seg)
        {
            sum += partials[i * kLrhGatherSegments + seg];
        }
        logits[i] = sum;
    }
    __syncthreads();
}

//! sorted[0, n) = keys[0, n) in descending order; called by all kMergeThreads threads, ends with a block barrier.
//! Equal keys (zero padding or a repeated id) are ranked by position, so the ranks are always a permutation.
__device__ void lrhBlockSortDescending(uint64_t const* keys, uint64_t* sorted, int32_t n)
{
    for (int32_t i = static_cast<int32_t>(threadIdx.x); i < n; i += kMergeThreads)
    {
        uint64_t const key = keys[i];
        int32_t rank = 0;
        for (int32_t j = 0; j < n; ++j)
        {
            uint64_t const other = keys[j];
            rank += (other > key || (other == key && j < i)) ? 1 : 0;
        }
        sorted[rank] = key;
    }
    __syncthreads();
}

static_assert(kLrhMaxCandidates <= kMergeThreads, "the finalize builds one key per thread");

//! Sort-then-reduce finalize of one row, called by all kMergeThreads threads of the block. Sorts the candidates by
//! (exact logit desc, id asc) with non-finite logits as -inf, then warp 0 computes max = first entry and sum-exp over
//! the sorted positions (lane-strided, then an xor butterfly) and writes ids and log_softmax. logits and ids are
//! shared-memory arrays of numCandidates entries; keys and sorted are shared scratch of numCandidates entries each.
__device__ void lrhBlockFinalize(float const* logits, int32_t const* ids, int32_t numCandidates, uint64_t* keys,
    uint64_t* sorted, int32_t* __restrict__ outIds, float* __restrict__ outLogProbs)
{
    int32_t const tid = static_cast<int32_t>(threadIdx.x);
    if (tid < numCandidates)
    {
        keys[tid] = lrhPackKey(lrhOrderableLogit(logits[tid]), static_cast<uint32_t>(ids[tid]));
    }
    __syncthreads();
    lrhBlockSortDescending(keys, sorted, numCandidates);
    if (tid < kLrhWarpSize)
    {
        uint32_t const topOrderable = static_cast<uint32_t>(sorted[0] >> 32);
        bool const anyFinite = topOrderable != kLrhNegInfOrderable;
        float const maxLogit = anyFinite ? lrhScoreFromOrderable(topOrderable) : 0.0F;
        float sum = 0.0F;
        for (int32_t i = tid; i < numCandidates; i += kLrhWarpSize)
        {
            uint32_t const orderable = static_cast<uint32_t>(sorted[i] >> 32);
            if (orderable != kLrhNegInfOrderable)
            {
                sum += expf(lrhScoreFromOrderable(orderable) - maxLogit);
            }
        }
        float const logSum = logf(lrhWarpSum(sum));
        for (int32_t i = tid; i < numCandidates; i += kLrhWarpSize)
        {
            uint64_t const key = sorted[i];
            uint32_t const orderable = static_cast<uint32_t>(key >> 32);
            outIds[i] = lrhKeyId(key);
            outLogProbs[i] = orderable != kLrhNegInfOrderable ? (lrhScoreFromOrderable(orderable) - maxLogit) - logSum
                                                              : __int_as_float(static_cast<int32_t>(0xFF800000U));
        }
    }
}

//! Shape and tensor checks of the fused runner. Returns R.
int32_t lrhCheckTensors(char const* caller, LowRankCandidateHeadShape const& shape, rt::Tensor const& hidden,
    rt::Tensor const& downProj, rt::Tensor const& upProj, rt::Tensor const& lmHead, rt::Tensor const& candidateIds,
    rt::Tensor const& candidateLogProbs)
{
    // Messages are built only on failure, so a passing check does not allocate.
    auto const checkMatrix = [&](rt::Tensor const& t, DataType type, int64_t rows, int64_t cols, char const* name) {
        auto const s = t.getShape();
        ELLM_CHECK(
            t.getDeviceType() == rt::DeviceType::kGPU, std::string(caller) + ": " + name + " must be on the GPU");
        ELLM_CHECK(t.getDataType() == type, std::string(caller) + ": " + name + " has the wrong data type");
        ELLM_CHECK(s.getNumDims() == 2 && (rows < 0 || s[0] == rows) && s[1] == cols,
            std::string(caller) + ": " + name + " has the wrong shape");
    };
    int64_t const rows = hidden.getShape().getNumDims() == 2 ? hidden.getShape()[0] : -1;
    ELLM_CHECK(rows >= 0, std::string(caller) + ": hidden must be [R, hiddenSize]");
    checkMatrix(hidden, DataType::kHALF, -1, shape.hiddenSize, "hidden");
    checkMatrix(downProj, DataType::kHALF, shape.rank, shape.hiddenSize, "downProj");
    checkMatrix(upProj, DataType::kHALF, shape.vocabSize, shape.rank, "upProj");
    checkMatrix(lmHead, DataType::kHALF, shape.vocabSize, shape.hiddenSize, "lmHead");
    checkMatrix(candidateIds, DataType::kINT32, rows, shape.numCandidates, "candidateIds");
    checkMatrix(candidateLogProbs, DataType::kFLOAT, rows, shape.numCandidates, "candidateLogProbs");
    constexpr uintptr_t kAlignMask = 15U;
    auto const checkAligned = [&](rt::Tensor const& t, char const* name) {
        ELLM_CHECK((reinterpret_cast<uintptr_t>(t.rawPointer()) & kAlignMask) == 0U,
            std::string(caller) + ": " + name + " must be 16-byte aligned");
    };
    checkAligned(hidden, "hidden");
    checkAligned(downProj, "downProj");
    checkAligned(upProj, "upProj");
    checkAligned(lmHead, "lmHead");
    return static_cast<int32_t>(rows);
}

void lrhCheckShape(char const* caller, LowRankCandidateHeadShape const& shape)
{
    ELLM_CHECK(shape.rank >= kLrhMmaK && shape.rank <= kLrhMaxRank && shape.rank % kLrhMmaK == 0,
        std::string(caller) + ": rank must be a multiple of 16 in [16, 256], got " + std::to_string(shape.rank));
    ELLM_CHECK(shape.hiddenSize > 0 && shape.hiddenSize % kLrhHalvesPerVector == 0,
        std::string(caller) + ": hiddenSize must be a positive multiple of 8, got " + std::to_string(shape.hiddenSize));
    ELLM_CHECK(shape.numCandidates >= 1 && shape.numCandidates <= kLrhMaxCandidates,
        std::string(caller) + ": numCandidates must be in [1, 256], got " + std::to_string(shape.numCandidates));
    ELLM_CHECK(shape.vocabSize >= shape.numCandidates,
        std::string(caller) + ": vocabSize must be at least numCandidates, got " + std::to_string(shape.vocabSize));
}

constexpr int32_t kK1WarpsPerGroup = 4;
constexpr int32_t kK1ThreadsPerGroup = kK1WarpsPerGroup * kLrhWarpSize;
constexpr int32_t kK1MaxCapacity = 512;
constexpr int32_t kK1EntriesPerLane = kK1MaxCapacity / kLrhWarpSize;
constexpr int32_t kK1CapacityHeadroom = 64;
constexpr int32_t kK1CapacityStep = 32;
constexpr int32_t kRadixBins = 256;
constexpr int32_t kRadixDigitBits = 8;
constexpr int32_t kSlabAlign = 16;
constexpr int32_t kMaxSlabRows = 65536;
constexpr int32_t kMinListHead = 4;
constexpr int32_t kMergeCandidateCapacity = 2048;
constexpr int32_t kMaxMergeGroups = LowRankCandidateHeadConfig::kMaxK2Groups;
constexpr int32_t kNumBucketRows = LowRankCandidateHeadConfig::kMaxBuckets;
constexpr int32_t kBucketRows[kNumBucketRows] = {8, 16, 32, 64};
constexpr int32_t kBucketTilesPerWarp[kNumBucketRows] = {1, 2, 2, 4};
constexpr int32_t kBucketGroups[kNumBucketRows] = {1, 1, 2, 2};
static_assert(kBucketRows[kNumBucketRows - 1] == LowRankCandidateHeadRunner::kMaxRowsPerPass,
    "workspaceBytes stops growing at the largest bucket");
constexpr uint64_t kInvalidKey = 0ULL;
//! When upProj streams from DRAM, row buckets above this size run the pilot bound by default (at or below it, K1 is
//! bandwidth bound and the pilot's re-read of upProj costs more than its cheaper inserts save).
constexpr int32_t kPilotMinRows = 8;
//! Orderable value of +inf: keys at or above it have a non-finite score.
constexpr uint32_t kPosInfOrderable = 0xFF800000U;
//! Upper bound on select passes (at most about 10 halving score passes plus 8 bit passes are ever needed); it only
//! guards against malformed input.
constexpr int32_t kMaxSelectPasses = 32;

// ---------------------------------------------------------------------------------------------------------------
// PTX wrappers. SM90+ streams upProj with cp.async.bulk and uses PDL and clusters; SM80-SM89 use cp.async 16 B.
// ---------------------------------------------------------------------------------------------------------------

__device__ __forceinline__ uint32_t lrhSmemAddr(void const* ptr)
{
    return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

__device__ __forceinline__ void lrhMbarInit(uint64_t* bar, uint32_t count)
{
    asm volatile("mbarrier.init.shared.b64 [%0], %1;\n" ::"r"(lrhSmemAddr(bar)), "r"(count) : "memory");
}

__device__ __forceinline__ void lrhMbarInitFence()
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
#endif
}

__device__ __forceinline__ void lrhMbarArrive(uint64_t* bar)
{
    asm volatile("{\n.reg .b64 state;\nmbarrier.arrive.shared.b64 state, [%0];\n}\n" ::"r"(lrhSmemAddr(bar))
        : "memory");
}

__device__ __forceinline__ void lrhMbarWait(uint64_t* bar, uint32_t parity)
{
    uint32_t done = 0;
    do
    {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
        asm volatile(
            "{\n.reg .pred p;\nmbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\nselp.u32 %0, 1, 0, p;\n}\n"
            : "=r"(done)
            : "r"(lrhSmemAddr(bar)), "r"(parity)
            : "memory");
#else
        // test_wait does not suspend the thread; back off so waiting warps leave issue slots to the others.
        asm volatile("{\n.reg .pred p;\nmbarrier.test_wait.parity.shared.b64 p, [%1], %2;\nselp.u32 %0, 1, 0, p;\n}\n"
            : "=r"(done)
            : "r"(lrhSmemAddr(bar)), "r"(parity)
            : "memory");
        if (done == 0U)
        {
            __nanosleep(32);
        }
#endif
    } while (done == 0U);
}

__device__ __forceinline__ uint64_t lrhEvictFirstPolicy()
{
    uint64_t policy = 0;
    asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;\n" : "=l"(policy));
    return policy;
}

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
__device__ __forceinline__ void lrhMbarArriveExpectTx(uint64_t* bar, uint32_t bytes)
{
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(lrhSmemAddr(bar)), "r"(bytes)
        : "memory");
}

__device__ __forceinline__ void lrhBulkCopy(
    void* dst, void const* src, uint32_t bytes, uint64_t* bar, uint64_t policy, bool evictFirst)
{
    if (evictFirst)
    {
        asm volatile(
            "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], "
            "%4;\n" ::"r"(lrhSmemAddr(dst)),
            "l"(src), "r"(bytes), "r"(lrhSmemAddr(bar)), "l"(policy)
            : "memory");
    }
    else
    {
        asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n" ::"r"(
                         lrhSmemAddr(dst)),
            "l"(src), "r"(bytes), "r"(lrhSmemAddr(bar))
            : "memory");
    }
}
#else
__device__ __forceinline__ void lrhCpAsync16(void* dst, void const* src, uint64_t policy, bool evictFirst)
{
    if (evictFirst)
    {
        asm volatile("cp.async.cg.shared.global.L2::cache_hint [%0], [%1], 16, %2;\n" ::"r"(lrhSmemAddr(dst)), "l"(src),
            "l"(policy)
            : "memory");
    }
    else
    {
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(lrhSmemAddr(dst)), "l"(src) : "memory");
    }
}

__device__ __forceinline__ void lrhCpAsyncArriveNoInc(uint64_t* bar)
{
    asm volatile("cp.async.mbarrier.arrive.noinc.shared.b64 [%0];\n" ::"r"(lrhSmemAddr(bar)) : "memory");
}
#endif

__device__ __forceinline__ unsigned long long lrhLoadRelaxed(unsigned long long const* ptr)
{
    unsigned long long value = 0;
    asm volatile("ld.relaxed.gpu.global.u64 %0, [%1];\n" : "=l"(value) : "l"(ptr) : "memory");
    return value;
}

__device__ __forceinline__ void lrhNamedBarrier(int32_t id, int32_t threads)
{
    asm volatile("bar.sync %0, %1;\n" ::"r"(id), "r"(threads) : "memory");
}

//! Named barrier that also returns whether any participating thread passed a true flag.
__device__ __forceinline__ bool lrhNamedBarrierAny(int32_t id, int32_t threads, bool flag)
{
    uint32_t any = 0;
    asm volatile("{\n.reg .pred p, q;\nsetp.ne.u32 p, %1, 0;\nbar.red.or.pred q, %2, %3, p;\nselp.u32 %0, 1, 0, q;\n}\n"
        : "=r"(any)
        : "r"(flag ? 1U : 0U), "r"(id), "r"(threads)
        : "memory");
    return any != 0U;
}

// ---------------------------------------------------------------------------------------------------------------
// Exact radix select. The candidate set is always a key interval [lo, hi]; each pass splits it into 256 bins that
// are non-decreasing in the key, keeps the bin holding the k-th largest key, and stops when that bin is taken whole.
// The first passes bin linearly in the fp32 score between the interval's end scores, which spreads score-like
// keys evenly; when such a pass keeps more than half of the interval (or the scores are equal or not finite), the
// pass takes 8 bits of (key - lo) instead, which shrinks the interval by 8 bits per pass.
// ---------------------------------------------------------------------------------------------------------------

struct LrhBinning
{
    uint64_t lo;
    float scoreLo;
    float scale;
    int32_t shift;
    bool floatMode;
};

__device__ __forceinline__ LrhBinning lrhMakeBinning(uint64_t lo, uint64_t hi, bool allowFloat)
{
    LrhBinning binning{};
    binning.lo = lo;
    uint32_t const ordLo = static_cast<uint32_t>(lo >> 32);
    uint32_t const ordHi = static_cast<uint32_t>(hi >> 32);
    if (allowFloat && ordLo != ordHi && ordLo > kLrhNegInfOrderable && ordHi < kPosInfOrderable)
    {
        float const scoreLo = lrhScoreFromOrderable(ordLo);
        float const range = lrhScoreFromOrderable(ordHi) - scoreLo;
        float const scale = static_cast<float>(kRadixBins) / range;
        if (range > 0.0F && (__float_as_uint(scale) & 0x7F800000U) != 0x7F800000U)
        {
            binning.floatMode = true;
            binning.scoreLo = scoreLo;
            binning.scale = scale;
            return binning;
        }
    }
    int32_t const bits = 64 - __clzll(static_cast<long long>(hi - lo));
    binning.shift = max(0, bits - kRadixDigitBits);
    return binning;
}

__device__ __forceinline__ uint32_t lrhBin(LrhBinning const& binning, uint64_t key)
{
    if (binning.floatMode)
    {
        float const offset
            = (lrhScoreFromOrderable(static_cast<uint32_t>(key >> 32)) - binning.scoreLo) * binning.scale;
        return static_cast<uint32_t>(min(max(__float2int_rd(offset), 0), kRadixBins - 1));
    }
    return static_cast<uint32_t>((key - binning.lo) >> binning.shift);
}

//! One warp scans a 256-bin histogram from the highest bin down and returns the bin holding the need-th largest key
//! and the number of keys in higher bins.
__device__ __forceinline__ void lrhFindBin(
    uint32_t const* hist, uint32_t need, uint32_t& bin, uint32_t& above, uint32_t& binCount)
{
    int32_t const lane = static_cast<int32_t>(threadIdx.x) % kLrhWarpSize;
    constexpr int32_t kBinsPerLane = kRadixBins / kLrhWarpSize;
    uint32_t counts[kBinsPerLane];
    uint32_t local = 0;
#pragma unroll
    for (int32_t i = 0; i < kBinsPerLane; ++i)
    {
        counts[i] = hist[kRadixBins - 1 - kBinsPerLane * lane - i];
        local += counts[i];
    }
    uint32_t inclusive = local;
#pragma unroll
    for (int32_t offset = 1; offset < kLrhWarpSize; offset <<= 1)
    {
        uint32_t const other = __shfl_up_sync(kLrhFullMask, inclusive, offset);
        if (lane >= offset)
        {
            inclusive += other;
        }
    }
    uint32_t const owners = __ballot_sync(kLrhFullMask, inclusive >= need);
    int32_t const owner = __ffs(static_cast<int32_t>(owners)) - 1;
    uint32_t cumulative = inclusive - local;
    uint32_t myBin = 0;
    uint32_t myAbove = 0;
    uint32_t myCount = 0;
    bool found = false;
#pragma unroll
    for (int32_t i = 0; i < kBinsPerLane; ++i)
    {
        bool const hit = !found && cumulative + counts[i] >= need;
        if (hit)
        {
            myBin = static_cast<uint32_t>(kRadixBins - 1 - kBinsPerLane * lane - i);
            myAbove = cumulative;
            myCount = counts[i];
            found = true;
        }
        cumulative += counts[i];
    }
    bin = __shfl_sync(kLrhFullMask, myBin, owner);
    above = __shfl_sync(kLrhFullMask, myAbove, owner);
    binCount = __shfl_sync(kLrhFullMask, myCount, owner);
}

struct LrhBlockScratch
{
    unsigned long long warpHi[kMergeWarps];
    unsigned long long warpLo[kMergeWarps];
    int32_t warpCount[kMergeWarps];
    unsigned long long hi;
    unsigned long long lo;
    int32_t count;
    uint32_t bin;
    uint32_t above;
    uint32_t binCount;
    int32_t candidateCount;
    int32_t mergedCount;
    int32_t tailCount;
    int32_t isLast;
};

//! Block-wide min, max and count; every thread gets the results. Called by all kMergeThreads threads.
__device__ void lrhBlockRange(uint64_t& lo, uint64_t& hi, int32_t& count, LrhBlockScratch& scratch)
{
    int32_t const warp = static_cast<int32_t>(threadIdx.x) / kLrhWarpSize;
    int32_t const lane = static_cast<int32_t>(threadIdx.x) % kLrhWarpSize;
    hi = lrhWarpMax(hi);
    lo = lrhWarpMin(lo);
    count = lrhWarpSum(count);
    if (lane == 0)
    {
        scratch.warpHi[warp] = hi;
        scratch.warpLo[warp] = lo;
        scratch.warpCount[warp] = count;
    }
    __syncthreads();
    if (warp == 0)
    {
        uint64_t h = lane < kMergeWarps ? scratch.warpHi[lane] : 0ULL;
        uint64_t l = lane < kMergeWarps ? scratch.warpLo[lane] : ~0ULL;
        int32_t c = lane < kMergeWarps ? scratch.warpCount[lane] : 0;
        h = lrhWarpMax(h);
        l = lrhWarpMin(l);
        c = lrhWarpSum(c);
        if (lane == 0)
        {
            scratch.hi = h;
            scratch.lo = l;
            scratch.count = c;
        }
    }
    __syncthreads();
    hi = scratch.hi;
    lo = scratch.lo;
    count = scratch.count;
    __syncthreads();
}

//! Threshold t such that exactly k of the nonzero keys (unique) are >= t, or the smallest of them when there are at
//! most k (~0 when there are none). visit(f) calls f(key) once per key, each key in exactly one calling thread.
//! kBlock: all kMergeThreads threads of the block call it, visit passes only nonzero keys and scratch holds the block
//! reductions. Otherwise one warp calls it, every lane passes the same number of keys and scratch is unused. hist is
//! the callers' 256-word shared histogram. Every caller gets the result.
template <bool kBlock, class Visit>
__device__ uint64_t lrhRadixSelect(Visit const& visit, int32_t k, uint32_t* hist, LrhBlockScratch* scratch)
{
    int32_t const tid = static_cast<int32_t>(threadIdx.x);
    int32_t const thread = kBlock ? tid : tid % kLrhWarpSize;
    auto const sync = []() {
        if constexpr (kBlock)
        {
            __syncthreads();
        }
        else
        {
            __syncwarp();
        }
    };
    uint64_t hi = 0;
    uint64_t lo = ~0ULL;
    int32_t members = 0;
    visit([&](uint64_t key) {
        bool const valid = kBlock || key != kInvalidKey;
        if (valid)
        {
            hi = key > hi ? key : hi;
            lo = key < lo ? key : lo;
        }
        members += kBlock ? 1 : __popc(__ballot_sync(kLrhFullMask, valid));
    });
    if constexpr (kBlock)
    {
        lrhBlockRange(lo, hi, members, *scratch);
    }
    else
    {
        hi = lrhWarpMax(hi);
        lo = lrhWarpMin(lo);
    }
    if (members <= k)
    {
        return lo;
    }
    // Here members > k >= 1, so lo is a nonzero key and the range checks below also exclude zero keys.
    uint32_t need = static_cast<uint32_t>(k);
    bool allowFloat = true;
    for (int32_t passIndex = 0; passIndex < kMaxSelectPasses && hi != lo; ++passIndex)
    {
        LrhBinning const binning = lrhMakeBinning(lo, hi, allowFloat);
        for (int32_t b = thread; b < kRadixBins; b += kBlock ? kMergeThreads : kLrhWarpSize)
        {
            hist[b] = 0U;
        }
        sync();
        visit([&](uint64_t key) {
            if (key >= lo && key <= hi)
            {
                atomicAdd(&hist[lrhBin(binning, key)], 1U);
            }
        });
        sync();
        uint32_t bin = 0;
        uint32_t above = 0;
        uint32_t binCount = 0;
        if (!kBlock || thread < kLrhWarpSize)
        {
            lrhFindBin(hist, need, bin, above, binCount);
            if (kBlock && thread == 0)
            {
                scratch->bin = bin;
                scratch->above = above;
                scratch->binCount = binCount;
            }
        }
        sync();
        if constexpr (kBlock)
        {
            bin = scratch->bin;
            above = scratch->above;
            binCount = scratch->binCount;
        }
        need -= above;
        uint64_t binLo = ~0ULL;
        uint64_t binHi = 0;
        visit([&](uint64_t key) {
            if (key >= lo && key <= hi && lrhBin(binning, key) == bin)
            {
                binLo = key < binLo ? key : binLo;
                binHi = key > binHi ? key : binHi;
            }
        });
        if constexpr (kBlock)
        {
            int32_t unused = 0;
            lrhBlockRange(binLo, binHi, unused, *scratch);
        }
        else
        {
            binLo = lrhWarpMin(binLo);
        }
        if (binCount == need)
        {
            return binLo;
        }
        if (binning.floatMode && 2 * static_cast<int32_t>(binCount) > members)
        {
            allowFloat = false;
        }
        members = static_cast<int32_t>(binCount);
        lo = binLo;
        hi = kBlock ? binHi : lrhWarpMax(binHi);
    }
    return lo;
}

// ---------------------------------------------------------------------------------------------------------------
// K1: persistent low-rank scoring with a per-CTA streaming top-Kc.
// ---------------------------------------------------------------------------------------------------------------

struct LrhSelectParams
{
    half const* upProj;
    half const* projected;
    unsigned long long* thresholds;
    unsigned long long* listKeys;
    int32_t* listCounts;
    int32_t vocabSize;
    int32_t rank;
    int32_t rows;
    int32_t slabRows;
    int32_t capacity;
    int32_t tileRows;
    int32_t stages;
    int32_t numCandidates;
    int32_t listHead;
    int32_t evictFirst;
};

struct LrhSelectLayout
{
    size_t bars;
    size_t tau;
    size_t counts;
    size_t hist;
    size_t scores;
    size_t idxs;
    size_t stages;
    size_t total;
};

__host__ __device__ inline size_t lrhAlignUp(size_t value, size_t alignment)
{
    return (value + alignment - 1) / alignment * alignment;
}

//! Survivor row strides (elements): one extra 32-bit word per row so the rows of a tile column fall on different
//! shared-memory banks.
__host__ __device__ inline int32_t lrhScoreStride(int32_t capacity)
{
    return capacity + 1;
}

__host__ __device__ inline int32_t lrhIdxStride(int32_t capacity)
{
    return capacity + 2;
}

__host__ __device__ inline LrhSelectLayout lrhSelectLayout(
    int32_t rowsCap, int32_t consumerWarps, int32_t stages, int32_t tileRows, int32_t capacity, int32_t rank)
{
    LrhSelectLayout layout{};
    size_t offset = 0;
    layout.bars = offset;
    offset += static_cast<size_t>(2 * stages) * sizeof(uint64_t);
    layout.tau = lrhAlignUp(offset, sizeof(uint64_t));
    offset = layout.tau + static_cast<size_t>(rowsCap) * sizeof(uint64_t);
    layout.counts = offset;
    offset += static_cast<size_t>(rowsCap) * sizeof(int32_t);
    layout.hist = lrhAlignUp(offset, 16);
    offset = layout.hist + static_cast<size_t>(consumerWarps) * kRadixBins * sizeof(uint32_t);
    layout.scores = lrhAlignUp(offset, 16);
    offset = layout.scores + static_cast<size_t>(rowsCap) * lrhScoreStride(capacity) * sizeof(uint32_t);
    layout.idxs = lrhAlignUp(offset, 16);
    offset = layout.idxs + static_cast<size_t>(rowsCap) * lrhIdxStride(capacity) * sizeof(uint16_t);
    layout.stages = lrhAlignUp(offset, 128);
    offset = layout.stages + static_cast<size_t>(stages) * tileRows * rank * sizeof(half);
    layout.total = offset;
    return layout;
}

struct LrhSelectShared
{
    unsigned long long* tau;
    int32_t* counts;
    uint32_t* scores; //!< fp32 score bits; the orderable value is formed when a buffer is read.
    uint16_t* idxs;
};

//! Shrinks row c's survivor buffer (warp-local; the group barriers order it). The pivot is the orderable score of
//! one of 32 sampled buffer entries, chosen by a binary search over the sorted sample so that between
//! numCandidates and capacity - tileRows keys have a score of at least the pivot (counted exactly). Those keys are
//! kept, and pivot << 32 is a lower bound of the row's numCandidates-th key in this slab. An exact select is the
//! fallback when no sampled pivot qualifies (for example, when most scores are equal).
__device__ void lrhCompactRow(LrhSelectParams const& p, LrhSelectShared const& s, int32_t c, int32_t slabBegin,
    uint32_t* hist, uint32_t lowerMask)
{
    int32_t const lane = static_cast<int32_t>(threadIdx.x) % kLrhWarpSize;
    int32_t const n = s.counts[c];
    uint32_t* const scores = s.scores + static_cast<int64_t>(c) * lrhScoreStride(p.capacity);
    uint16_t* const idxs = s.idxs + static_cast<int64_t>(c) * lrhIdxStride(p.capacity);
    uint32_t ord[kK1EntriesPerLane];
    uint16_t idx[kK1EntriesPerLane];
#pragma unroll
    for (int32_t e = 0; e < kK1EntriesPerLane; ++e)
    {
        int32_t const i = lane + kLrhWarpSize * e;
        bool const valid = i < n;
        ord[e] = valid ? lrhOrderableScore(__uint_as_float(scores[i])) : 0U;
        idx[e] = valid ? idxs[i] : static_cast<uint16_t>(0);
    }
    uint32_t const sample
        = lrhWarpSortDescending(lrhOrderableScore(__uint_as_float(scores[(lane * n) / kLrhWarpSize])));
    int32_t const limit = p.capacity - p.tileRows;
    int32_t low = 0;
    int32_t high = kLrhWarpSize - 1;
    int32_t guess = min(high, (p.numCandidates + p.numCandidates / 4) * kLrhWarpSize / max(n, 1));
    uint64_t threshold = 0ULL;
    bool found = false;
    while (low <= high)
    {
        uint32_t const pivot = __shfl_sync(kLrhFullMask, sample, guess);
        int32_t atLeast = 0;
#pragma unroll
        for (int32_t e = 0; e < kK1EntriesPerLane; ++e)
        {
            atLeast += (lane + kLrhWarpSize * e < n && ord[e] >= pivot) ? 1 : 0;
        }
        atLeast = lrhWarpSum(atLeast);
        if (atLeast < p.numCandidates)
        {
            low = guess + 1;
        }
        else if (atLeast > limit)
        {
            high = guess - 1;
        }
        else
        {
            threshold = static_cast<uint64_t>(pivot) << 32;
            found = true;
            break;
        }
        guess = (low + high) / 2;
    }
    if (!found)
    {
        uint64_t keys[kK1EntriesPerLane];
#pragma unroll
        for (int32_t e = 0; e < kK1EntriesPerLane; ++e)
        {
            keys[e] = lane + kLrhWarpSize * e < n ? lrhPackKey(ord[e], static_cast<uint32_t>(slabBegin + idx[e]))
                                                  : kInvalidKey;
        }
        auto const visit = [&](auto const& f) {
#pragma unroll
            for (int32_t e = 0; e < kK1EntriesPerLane; ++e)
            {
                f(keys[e]);
            }
        };
        threshold = lrhRadixSelect<false>(visit, p.numCandidates, hist, nullptr);
    }
    // Every lane has read its entries before any lane overwrites the buffer.
    __syncwarp();
    int32_t written = 0;
#pragma unroll
    for (int32_t e = 0; e < kK1EntriesPerLane; ++e)
    {
        bool const keep
            = lane + kLrhWarpSize * e < n && lrhPackKey(ord[e], static_cast<uint32_t>(slabBegin + idx[e])) >= threshold;
        uint32_t const mask = __ballot_sync(kLrhFullMask, keep);
        if (keep)
        {
            int32_t const pos = written + __popc(mask & lowerMask);
            // Exact round trip: ord is the orderable image of a real score (NaN images are 0).
            scores[pos] = __float_as_uint(lrhScoreFromOrderable(ord[e]));
            idxs[pos] = idx[e];
        }
        written += __popc(mask);
    }
    if (lane == 0)
    {
        s.counts[c] = written;
        unsigned long long tau = threshold > s.tau[c] ? threshold : s.tau[c];
        // Publish this slab's bound and adopt the best bound any CTA has reached (both are lower bounds of the row's
        // numCandidates-th key).
        unsigned long long const global = atomicMax(&p.thresholds[c], tau);
        tau = global > tau ? global : tau;
        s.tau[c] = tau;
    }
}

//! Writes row c's final list from its first n buffer entries, kEntries per lane: drops keys below the shared
//! threshold (they cannot be among the row's numCandidates largest) and stores the listHead largest keys first, in
//! descending order, so the merge can bound the list from its head alone. The list holds at most `capacity` keys.
template <int32_t kEntries>
__device__ void lrhWriteListEntries(
    LrhSelectParams const& p, LrhSelectShared const& s, int32_t c, int32_t slabBegin, int32_t n, uint64_t shared)
{
    int32_t const lane = static_cast<int32_t>(threadIdx.x) % kLrhWarpSize;
    uint32_t const* const scores = s.scores + static_cast<int64_t>(c) * lrhScoreStride(p.capacity);
    uint16_t const* const idxs = s.idxs + static_cast<int64_t>(c) * lrhIdxStride(p.capacity);
    uint64_t keys[kEntries];
    int32_t valid = 0;
#pragma unroll
    for (int32_t e = 0; e < kEntries; ++e)
    {
        int32_t const i = lane + kLrhWarpSize * e;
        uint64_t const key = i < n
            ? lrhPackKey(lrhOrderableScore(__uint_as_float(scores[i])), static_cast<uint32_t>(slabBegin + idxs[i]))
            : kInvalidKey;
        keys[e] = key >= shared ? key : kInvalidKey;
        valid += __popc(__ballot_sync(kLrhFullMask, keys[e] != kInvalidKey));
    }
    int64_t const list = static_cast<int64_t>(blockIdx.x) * p.rows + c;
    unsigned long long* const out = p.listKeys + list * p.capacity;
    int32_t const headCount = min(p.listHead, valid);
    for (int32_t h = 0; h < headCount; ++h)
    {
        uint64_t best = 0ULL;
#pragma unroll
        for (int32_t e = 0; e < kEntries; ++e)
        {
            best = keys[e] > best ? keys[e] : best;
        }
        uint64_t const winner = lrhWarpMax(best);
#pragma unroll
        for (int32_t e = 0; e < kEntries; ++e)
        {
            if (keys[e] == winner)
            {
                keys[e] = kInvalidKey;
            }
        }
        if (lane == 0)
        {
            out[h] = winner;
        }
    }
    int32_t written = headCount;
#pragma unroll
    for (int32_t e = 0; e < kEntries; ++e)
    {
        bool const take = keys[e] != kInvalidKey;
        uint32_t const mask = __ballot_sync(kLrhFullMask, take);
        if (take)
        {
            out[written + __popc(mask & ((1U << lane) - 1U))] = keys[e];
        }
        written += __popc(mask);
    }
    if (lane == 0)
    {
        p.listCounts[list] = valid;
    }
}

//! Row c's final list (see lrhWriteListEntries). Buffers usually hold far fewer than capacity keys, so the
//! per-lane entry count is chosen from the fill.
__device__ void lrhWriteList(LrhSelectParams const& p, LrhSelectShared const& s, int32_t c, int32_t slabBegin)
{
    int32_t const n = s.counts[c];
    uint64_t const shared = lrhLoadRelaxed(&p.thresholds[c]);
    if (n <= kLrhWarpSize)
    {
        lrhWriteListEntries<1>(p, s, c, slabBegin, n, shared);
    }
    else if (n <= 4 * kLrhWarpSize)
    {
        lrhWriteListEntries<4>(p, s, c, slabBegin, n, shared);
    }
    else
    {
        lrhWriteListEntries<kK1EntriesPerLane>(p, s, c, slabBegin, n, shared);
    }
}

//! K1 producer warp: streams the slab's upProj tiles through the stage ring (SM90+: one bulk copy per tile, SM80-89:
//! 16-byte cp.async). upProj is an engine weight, so it is streamed before the prerequisite grid completes. A tile of
//! T rows is one contiguous block of upProj.
__device__ __forceinline__ void lrhProduceTiles(LrhSelectParams const& p, half* stageBuf, uint64_t* fullBar,
    uint64_t* emptyBar, int32_t slabBegin, int32_t slabEnd, int32_t numTiles, int32_t lane)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    int32_t const stageElems = p.tileRows * p.rank;
    bool const evictFirst = p.evictFirst != 0;
    uint64_t const policy = evictFirst ? lrhEvictFirstPolicy() : 0ULL;
    int32_t stage = 0;
    uint32_t phase = 0;
    for (int32_t tile = 0; tile < numTiles; ++tile)
    {
        if (tile >= p.stages)
        {
            // The consumers released this stage's previous use in the phase before the current one.
            lrhMbarWait(&emptyBar[stage], phase ^ 1U);
        }
        int32_t const tileBegin = slabBegin + tile * p.tileRows;
        int32_t const bytes = min(p.tileRows, slabEnd - tileBegin) * p.rank * static_cast<int32_t>(sizeof(half));
        half* const dst = stageBuf + static_cast<int64_t>(stage) * stageElems;
        half const* const src = p.upProj + static_cast<int64_t>(tileBegin) * p.rank;
#if __CUDA_ARCH__ >= 900
        if (lane == 0)
        {
            lrhMbarArriveExpectTx(&fullBar[stage], static_cast<uint32_t>(bytes));
            lrhBulkCopy(dst, src, static_cast<uint32_t>(bytes), &fullBar[stage], policy, evictFirst);
        }
#else
        for (int32_t q = lane; q < bytes / 16; q += kLrhWarpSize)
        {
            lrhCpAsync16(dst + q * kLrhHalvesPerVector, src + q * kLrhHalvesPerVector, policy, evictFirst);
        }
        lrhCpAsyncArriveNoInc(&fullBar[stage]);
#endif
        if (++stage == p.stages)
        {
            stage = 0;
            phase ^= 1U;
        }
    }
#if __CUDA_ARCH__ < 900
    asm volatile("cp.async.wait_all;\n" ::: "memory");
#endif
    if (lane == 0)
    {
        lrhTriggerDependents();
    }
#endif
}

//! One-group variants keep two CTAs per SM (the small-R grid is twice the SM count). kFullRank: rank is exactly
//! kMaxKPairs x 32, so every fragment load is in range and needs no mask.
template <int32_t kTilesPerWarp, int32_t kGroups, int32_t kMaxKPairs, bool kFullRank>
__global__ void __launch_bounds__((kGroups * kK1WarpsPerGroup + 1) * kLrhWarpSize, kGroups == 1 ? 2 : 1)
    lrhSelectKernel(LrhSelectParams const p)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    constexpr int32_t kConsumerWarps = kGroups * kK1WarpsPerGroup;
    constexpr int32_t kGroupRows = kTilesPerWarp * kLrhMmaN;
    constexpr int32_t kRowsCap = kGroups * kGroupRows;
    static_assert(kGroupRows <= kLrhWarpSize, "the compaction scan gives each row of a group one lane");
#if __CUDA_ARCH__ >= 900
    constexpr uint32_t kFullArrivals = 1U;
#else
    constexpr uint32_t kFullArrivals = kLrhWarpSize;
#endif
    extern __shared__ __align__(16) unsigned char smem[];
    LrhSelectLayout const layout = lrhSelectLayout(kRowsCap, kConsumerWarps, p.stages, p.tileRows, p.capacity, p.rank);
    auto* const fullBar = reinterpret_cast<uint64_t*>(smem + layout.bars);
    uint64_t* const emptyBar = fullBar + p.stages;
    LrhSelectShared s{};
    s.tau = reinterpret_cast<unsigned long long*>(smem + layout.tau);
    s.counts = reinterpret_cast<int32_t*>(smem + layout.counts);
    s.scores = reinterpret_cast<uint32_t*>(smem + layout.scores);
    s.idxs = reinterpret_cast<uint16_t*>(smem + layout.idxs);
    auto* const histAll = reinterpret_cast<uint32_t*>(smem + layout.hist);
    auto* const stageBuf = reinterpret_cast<half*>(smem + layout.stages);

    int32_t const tid = static_cast<int32_t>(threadIdx.x);
    int32_t const warp = tid / kLrhWarpSize;
    int32_t const lane = tid % kLrhWarpSize;
    int32_t const slabBegin = min(static_cast<int32_t>(blockIdx.x) * p.slabRows, p.vocabSize);
    int32_t const slabEnd = min(slabBegin + p.slabRows, p.vocabSize);
    int32_t const numTiles = (slabEnd - slabBegin + p.tileRows - 1) / p.tileRows;
    int32_t const stageElems = p.tileRows * p.rank;

    if (tid == 0)
    {
        for (int32_t st = 0; st < p.stages; ++st)
        {
            lrhMbarInit(&fullBar[st], kFullArrivals);
            lrhMbarInit(&emptyBar[st], kConsumerWarps);
        }
        lrhMbarInitFence();
    }
    for (int32_t i = tid; i < kRowsCap; i += static_cast<int32_t>(blockDim.x))
    {
        s.tau[i] = 0ULL;
        s.counts[i] = 0;
    }
    __syncthreads();

    if (warp == kConsumerWarps)
    {
        lrhProduceTiles(p, stageBuf, fullBar, emptyBar, slabBegin, slabEnd, numTiles, lane);
        return;
    }

    int32_t const group = warp / kK1WarpsPerGroup;
    int32_t const warpInGroup = warp % kK1WarpsPerGroup;
    int32_t const groupRowBase = group * kGroupRows;
    int32_t const barrierId = 1 + group;
    int32_t const groupId = lane >> 2;
    int32_t const pairId = lane & 3;
    uint32_t const lowerMask = (1U << lane) - 1U;
    uint32_t* const hist = histAll + warp * kRadixBins;

    // s and the shared thresholds are K0 outputs.
    lrhWaitPrerequisites();
    {
        int32_t const groupThread = warpInGroup * kLrhWarpSize + lane;
        int32_t const c = groupRowBase + groupThread;
        if (groupThread < kGroupRows && c < p.rows)
        {
            s.tau[c] = lrhLoadRelaxed(&p.thresholds[c]);
        }
        lrhNamedBarrier(barrierId, kK1ThreadsPerGroup);
    }

    uint4 sFrag[kTilesPerWarp][kMaxKPairs];
    lrhLoadScoreFragments<kTilesPerWarp, kMaxKPairs, kFullRank>(
        sFrag, p.rank, pairId, [&](int32_t nt, int32_t col, bool inRank) {
            int32_t const n = groupRowBase + nt * kLrhMmaN + groupId;
            return lrhLoad16Produced(p.projected + static_cast<int64_t>(n) * p.rank + col, n < p.rows && inRank);
        });

    bool const mmaWarp = warpInGroup * kLrhMmaM < p.tileRows;
    int32_t const fragmentRow = warpInGroup * kLrhMmaM + groupId;
    uint32_t const columnMask = 0x11111111U << pairId;
    int32_t const compactAbove = p.capacity - p.tileRows;

    // Prefilter of each (n-tile, j) row of this lane: the score of the row's threshold key, -inf when the threshold
    // is 0 (every key passes) and +inf past p.rows.
    float prefilter[kTilesPerWarp][2];
    auto const loadPrefilter = [&]() {
#pragma unroll
        for (int32_t nt = 0; nt < kTilesPerWarp; ++nt)
        {
#pragma unroll
            for (int32_t j = 0; j < 2; ++j)
            {
                int32_t const c = groupRowBase + nt * kLrhMmaN + 2 * pairId + j;
                unsigned long long const tau = c < p.rows ? s.tau[c] : 0ULL;
                prefilter[nt][j] = c >= p.rows ? __int_as_float(0x7F800000)
                    : tau == 0ULL              ? __int_as_float(static_cast<int32_t>(0xFF800000U))
                                               : lrhScoreFromOrderable(static_cast<uint32_t>(tau >> 32));
            }
        }
    };
    loadPrefilter();

    int32_t stage = 0;
    uint32_t phase = 0;
    for (int32_t tile = 0; tile < numTiles; ++tile)
    {
        int32_t const tileBegin = slabBegin + tile * p.tileRows;
        lrhMbarWait(&fullBar[stage], phase);

        // Bitwise equal to the pilot's witness scores (lrhAccumulateScoreTile from zero).
        float acc[kTilesPerWarp][4] = {};
        if (mmaWarp)
        {
            half const* const rowLoPtr = stageBuf + static_cast<int64_t>(stage) * stageElems
                + static_cast<int64_t>(fragmentRow) * p.rank + kLrhHalvesPerVector * pairId;
            half const* const rowHiPtr = rowLoPtr + static_cast<int64_t>(kLrhMmaM / 2) * p.rank;
            lrhAccumulateScoreTile<kTilesPerWarp, kMaxKPairs, kFullRank>(
                acc, sFrag, p.rank, pairId, [&](int32_t kp, bool inRank, uint4& rowLo, uint4& rowHi) {
                    uint4 const zero = make_uint4(0U, 0U, 0U, 0U);
                    rowLo = inRank ? *reinterpret_cast<uint4 const*>(rowLoPtr + kp * kLrhKPair) : zero;
                    rowHi = inRank ? *reinterpret_cast<uint4 const*>(rowHiPtr + kp * kLrhKPair) : zero;
                });
        }
        __syncwarp();
        if (lane == 0)
        {
            lrhMbarArrive(&emptyBar[stage]);
        }
        if (++stage == p.stages)
        {
            stage = 0;
            phase ^= 1U;
        }

        // overflow: an append of this tile took a row past capacity - tileRows. A row only grows by appends, and
        // the last append of a tile sees its final count, so the barrier's OR tells every warp of the group
        // whether any row needs a compaction.
        bool overflow = false;
        if (mmaWarp)
        {
            bool any = false;
#pragma unroll
            for (int32_t nt = 0; nt < kTilesPerWarp; ++nt)
            {
#pragma unroll
                for (int32_t i = 0; i < 4; ++i)
                {
                    any = any || !(acc[nt][i] < prefilter[nt][i & 1]);
                }
            }
            if (__any_sync(kLrhFullMask, any))
            {
                uint32_t const vocab0 = static_cast<uint32_t>(tileBegin + fragmentRow);
                bool const inSlab[2]
                    = {static_cast<int32_t>(vocab0) < slabEnd, static_cast<int32_t>(vocab0) + kLrhMmaM / 2 < slabEnd};
#pragma unroll
                for (int32_t nt = 0; nt < kTilesPerWarp; ++nt)
                {
                    // Lane (g, t) holds rows c0 + j (j = 0, 1) for vocab rows vocab0 + 8h (h = 0, 1). A key passes
                    // when it is at or above its row's threshold (a lower bound of the row's Kc-th key that can equal
                    // it); a CTA scores each vocab row once, so no key is appended twice. score > prefilter decides
                    // pass and score < prefilter decides fail (both hold for the real values under flush-to-zero
                    // compares); equal or NaN scores, and rows without a threshold, compare the 64-bit keys.
                    int32_t const c0 = groupRowBase + nt * kLrhMmaN + 2 * pairId;
                    bool pass[2][2];
                    bool exact = false;
                    bool candidate = false;
#pragma unroll
                    for (int32_t j = 0; j < 2; ++j)
                    {
#pragma unroll
                        for (int32_t h = 0; h < 2; ++h)
                        {
                            float const score = acc[nt][2 * h + j];
                            bool const above = score > prefilter[nt][j];
                            bool const unsure = !above && !(score < prefilter[nt][j]);
                            pass[j][h] = above && inSlab[h];
                            exact = exact || unsure;
                            candidate = candidate || above || unsure;
                        }
                    }
                    if (!__any_sync(kLrhFullMask, candidate))
                    {
                        continue;
                    }
                    if (__any_sync(kLrhFullMask, exact))
                    {
#pragma unroll
                        for (int32_t j = 0; j < 2; ++j)
                        {
                            bool const rowValid = c0 + j < p.rows;
                            unsigned long long const threshold = rowValid ? s.tau[c0 + j] : 0ULL;
#pragma unroll
                            for (int32_t h = 0; h < 2; ++h)
                            {
                                float const score = acc[nt][2 * h + j];
                                if (!(score > prefilter[nt][j]) && !(score < prefilter[nt][j]))
                                {
                                    uint32_t const vocab = vocab0 + static_cast<uint32_t>(h * kLrhMmaM / 2);
                                    pass[j][h] = rowValid && inSlab[h]
                                        && lrhPackKey(lrhOrderableScore(score), vocab) >= threshold;
                                }
                            }
                        }
                    }
                    uint32_t mask[2][2];
#pragma unroll
                    for (int32_t j = 0; j < 2; ++j)
                    {
#pragma unroll
                        for (int32_t h = 0; h < 2; ++h)
                        {
                            mask[j][h] = __ballot_sync(kLrhFullMask, pass[j][h]);
                        }
                    }
                    // Lanes of one column (same t) own the same two rows and count their appends identically.
                    uint32_t const lo0 = __popc(mask[0][0] & columnMask);
                    uint32_t const lo1 = __popc(mask[1][0] & columnMask);
                    uint32_t const added0 = lo0 + __popc(mask[0][1] & columnMask);
                    uint32_t const added1 = lo1 + __popc(mask[1][1] & columnMask);
                    // Lane (0, t) reserves row c0's slots and lane (1, t) row c0 + 1's, in one atomic instruction.
                    uint32_t const added = groupId == 0 ? added0 : added1;
                    uint32_t previous = 0U;
                    if (groupId < 2 && added != 0U)
                    {
                        previous = atomicAdd(reinterpret_cast<uint32_t*>(s.counts + c0 + groupId), added);
                    }
                    // Lane (g, t) is lane 4g + t.
                    uint32_t const base[2] = {
                        __shfl_sync(kLrhFullMask, previous, pairId), __shfl_sync(kLrhFullMask, previous, 4 + pairId)};
                    overflow = overflow || static_cast<int32_t>(base[0] + added0) > compactAbove
                        || static_cast<int32_t>(base[1] + added1) > compactAbove;
#pragma unroll
                    for (int32_t j = 0; j < 2; ++j)
                    {
#pragma unroll
                        for (int32_t h = 0; h < 2; ++h)
                        {
                            if (pass[j][h])
                            {
                                int32_t const c = c0 + j;
                                uint32_t const pos = base[j] + (h == 0 ? 0U : (j == 0 ? lo0 : lo1))
                                    + __popc(mask[j][h] & columnMask & lowerMask);
                                s.scores[static_cast<int64_t>(c) * lrhScoreStride(p.capacity) + pos]
                                    = __float_as_uint(acc[nt][2 * h + j]);
                                s.idxs[static_cast<int64_t>(c) * lrhIdxStride(p.capacity) + pos]
                                    = static_cast<uint16_t>(vocab0 + static_cast<uint32_t>(h * kLrhMmaM / 2)
                                        - static_cast<uint32_t>(slabBegin));
                            }
                        }
                    }
                }
            }
        }
        // Every append of this tile is visible before a row is compacted, and every compaction finishes before the
        // next tile filters (count <= capacity - tileRows holds at each tile start).
        if (lrhNamedBarrierAny(barrierId, kK1ThreadsPerGroup, overflow))
        {
            // Warp w of the group compacts rows w, w + 4, ... and reads only their counts (another warp may already
            // be compacting, and rewriting, its own rows).
            int32_t const c = groupRowBase + lane;
            bool const over = lane < kGroupRows && lane % kK1WarpsPerGroup == warpInGroup && c < p.rows
                && s.counts[c] > compactAbove;
            uint32_t need = __ballot_sync(kLrhFullMask, over);
            while (need != 0U)
            {
                int32_t const rr = __ffs(static_cast<int32_t>(need)) - 1;
                need &= need - 1U;
                lrhCompactRow(p, s, groupRowBase + rr, slabBegin, hist, lowerMask);
            }
            lrhNamedBarrier(barrierId, kK1ThreadsPerGroup);
            loadPrefilter();
        }
    }

    for (int32_t rr = warpInGroup; rr < kGroupRows; rr += kK1WarpsPerGroup)
    {
        int32_t const c = groupRowBase + rr;
        if (c < p.rows)
        {
            lrhWriteList(p, s, c, slabBegin);
        }
    }
#endif
}

using LrhSelectKernelFn = void (*)(LrhSelectParams);

template <int32_t kBucket>
LrhSelectKernelFn lrhSelectVariant(int32_t rank)
{
    constexpr int32_t kTilesPerWarp = kBucketTilesPerWarp[kBucket];
    constexpr int32_t kGroups = kBucketGroups[kBucket];
    if (rank == 4 * kLrhKPair)
    {
        return lrhSelectKernel<kTilesPerWarp, kGroups, 4, true>;
    }
    return rank > 4 * kLrhKPair ? lrhSelectKernel<kTilesPerWarp, kGroups, 8, false>
                                : lrhSelectKernel<kTilesPerWarp, kGroups, 4, false>;
}

LrhSelectKernelFn lrhSelectKernelFor(int32_t bucket, int32_t rank)
{
    switch (bucket)
    {
    case 0: return lrhSelectVariant<0>(rank);
    case 1: return lrhSelectVariant<1>(rank);
    case 2: return lrhSelectVariant<2>(rank);
    default: return lrhSelectVariant<3>(rank);
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Pilot bound: the largest key of each sampled tile is a real key (a witness), so the Kc-th largest witness of a row
// is at most the row's Kc-th key and can seed every CTA's threshold from the first tile.
// ---------------------------------------------------------------------------------------------------------------

constexpr int32_t kPilotWarps = 4;
constexpr int32_t kPilotThreads = kPilotWarps * kLrhWarpSize;
constexpr int32_t kPilotTileRows = kPilotWarps * kLrhMmaM;
constexpr int32_t kPilotMaxWitnesses = 1024;
constexpr int32_t kPilotEntriesPerLane = kPilotMaxWitnesses / kLrhWarpSize;
static_assert(kLrhProjectionMaxRows <= kPilotThreads, "the pilot writes each row's witness from one thread");

struct LrhPilotParams
{
    half const* upProj;
    half const* projected;
    unsigned long long* witnesses;
    unsigned long long* thresholds;
    int32_t vocabSize;
    int32_t rank;
    int32_t rows;
    int32_t tileStride;
    int32_t numWitnesses;
    int32_t numCandidates;
};

//! K0c: witness [w, row] = the largest key of pilot tile w (vocab rows [w * tileStride * 64, + 64)) for every row of
//! the pass. CTA w scores tile w, with s staged in shared memory.
template <int32_t kMaxKPairs>
__global__ void __launch_bounds__(kPilotThreads) lrhPilotKernel(LrhPilotParams const p)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    constexpr int32_t kMaxTiles = kLrhProjectionMaxRows / kLrhMmaN;
    __shared__ __align__(16) half sShared[kLrhProjectionMaxRows * kMaxKPairs * kLrhKPair];
    __shared__ unsigned long long warpMax[kPilotWarps][kLrhProjectionMaxRows];
    int32_t const tid = static_cast<int32_t>(threadIdx.x);
    if (tid == 0)
    {
        lrhTriggerDependents();
    }
    int32_t const warp = tid / kLrhWarpSize;
    int32_t const lane = tid % kLrhWarpSize;
    int32_t const groupId = lane >> 2;
    int32_t const pairId = lane & 3;
    int32_t const nTiles = (p.rows + kLrhMmaN - 1) / kLrhMmaN;
    int32_t const w = static_cast<int32_t>(blockIdx.x);
    int32_t const vocabLo = w * p.tileStride * kPilotTileRows + warp * kLrhMmaM + groupId;
    int32_t const vocabHi = vocabLo + kLrhMmaM / 2;
    // upProj is a weight: its fragments are loaded before the wait on the projection.
    uint4 fragLo[kMaxKPairs];
    uint4 fragHi[kMaxKPairs];
#pragma unroll
    for (int32_t kp = 0; kp < kMaxKPairs; ++kp)
    {
        int32_t const col = kp * kLrhKPair + kLrhHalvesPerVector * pairId;
        bool const colValid = lrhKPairChunkInRank(kp, pairId, p.rank);
        fragLo[kp]
            = lrhLoad16(p.upProj + static_cast<int64_t>(vocabLo) * p.rank + col, colValid && vocabLo < p.vocabSize);
        fragHi[kp]
            = lrhLoad16(p.upProj + static_cast<int64_t>(vocabHi) * p.rank + col, colValid && vocabHi < p.vocabSize);
    }
    lrhWaitPrerequisites();
    int32_t const vectorsPerRow = p.rank / kLrhHalvesPerVector;
    for (int32_t i = tid; i < p.rows * vectorsPerRow; i += kPilotThreads)
    {
        int32_t const n = i / vectorsPerRow;
        int32_t const v = i % vectorsPerRow;
        reinterpret_cast<uint4*>(sShared)[n * kMaxKPairs * (kLrhKPair / kLrhHalvesPerVector) + v]
            = __ldcg(reinterpret_cast<uint4 const*>(p.projected) + static_cast<int64_t>(n) * vectorsPerRow + v);
    }
    __syncthreads();
#pragma unroll
    for (int32_t nt = 0; nt < kMaxTiles; ++nt)
    {
        if (nt < nTiles)
        {
            int32_t const n = nt * kLrhMmaN + groupId;
            uint4 b[1][kMaxKPairs];
            lrhLoadScoreFragments<1, kMaxKPairs, false>(b, p.rank, pairId, [&](int32_t, int32_t col, bool inRank) {
                return n < p.rows && inRank
                    ? *reinterpret_cast<uint4 const*>(sShared + n * kMaxKPairs * kLrhKPair + col)
                    : make_uint4(0U, 0U, 0U, 0U);
            });
            // Bitwise equal to K1's scores of the same keys (lrhAccumulateScoreTile from zero).
            float acc[1][4] = {};
            lrhAccumulateScoreTile<1, kMaxKPairs, false>(
                acc, b, p.rank, pairId, [&](int32_t kp, bool, uint4& rowLo, uint4& rowHi) {
                    // The fragments were loaded with zeros past the rank.
                    rowLo = fragLo[kp];
                    rowHi = fragHi[kp];
                });
#pragma unroll
            for (int32_t j = 0; j < 2; ++j)
            {
                uint64_t const keyLo = vocabLo < p.vocabSize
                    ? lrhPackKey(lrhOrderableScore(acc[0][j]), static_cast<uint32_t>(vocabLo))
                    : kInvalidKey;
                uint64_t const keyHi = vocabHi < p.vocabSize
                    ? lrhPackKey(lrhOrderableScore(acc[0][2 + j]), static_cast<uint32_t>(vocabHi))
                    : kInvalidKey;
                // Maximum over the lanes 4 g + pairId (g = 0..7), which hold the same batch row c.
                uint64_t const best = lrhWarpMax(keyLo > keyHi ? keyLo : keyHi, 4);
                int32_t const c = nt * kLrhMmaN + 2 * pairId + j;
                if (groupId == 0 && c < p.rows)
                {
                    warpMax[warp][c] = best;
                }
            }
        }
    }
    __syncthreads();
    if (tid < p.rows)
    {
        unsigned long long best = 0ULL;
#pragma unroll
        for (int32_t wp = 0; wp < kPilotWarps; ++wp)
        {
            best = warpMax[wp][tid] > best ? warpMax[wp][tid] : best;
        }
        p.witnesses[static_cast<int64_t>(w) * p.rows + tid] = best;
    }
#endif
}

//! K0d: thresholds[row] = a value with at least numCandidates witnesses at or above it (one warp per row). The
//! value is the largest of 32 sorted sampled witnesses that keeps at least numCandidates witnesses, counted exactly.
__global__ void __launch_bounds__(kLrhWarpSize) lrhPilotBoundKernel(LrhPilotParams const p)
{
    int32_t const lane = static_cast<int32_t>(threadIdx.x);
    int32_t const row = static_cast<int32_t>(blockIdx.x);
    if (lane == 0)
    {
        lrhTriggerDependents();
    }
    lrhWaitPrerequisites();
    uint64_t keys[kPilotEntriesPerLane];
#pragma unroll
    for (int32_t e = 0; e < kPilotEntriesPerLane; ++e)
    {
        int32_t const w = lane + kLrhWarpSize * e;
        keys[e] = w < p.numWitnesses ? __ldcg(p.witnesses + static_cast<int64_t>(w) * p.rows + row) : kInvalidKey;
    }
    // A sample of 32 witnesses spread over all of them, sorted descending.
    unsigned long long const sample = lrhWarpSortDescending(
        __ldcg(p.witnesses + static_cast<int64_t>(lane * p.numWitnesses / kLrhWarpSize) * p.rows + row));
    // The count of witnesses at or above a value grows as the value falls, so a binary search over the sorted sample
    // finds the largest sampled value that keeps numCandidates witnesses.
    uint64_t bound = 0ULL;
    int32_t low = 0;
    int32_t high = kLrhWarpSize - 1;
    while (low <= high)
    {
        int32_t const mid = (low + high) / 2;
        uint64_t const value = __shfl_sync(kLrhFullMask, sample, mid);
        int32_t atLeast = 0;
#pragma unroll
        for (int32_t e = 0; e < kPilotEntriesPerLane; ++e)
        {
            atLeast += keys[e] != kInvalidKey && keys[e] >= value ? 1 : 0;
        }
        atLeast = lrhWarpSum(atLeast);
        if (atLeast >= p.numCandidates && value != kInvalidKey)
        {
            bound = value;
            high = mid - 1;
        }
        else
        {
            low = mid + 1;
        }
    }
    if (lane == 0)
    {
        p.thresholds[row] = bound;
    }
}

using LrhPilotKernelFn = void (*)(LrhPilotParams);

LrhPilotKernelFn lrhPilotKernelFor(int32_t rank)
{
    return rank > 4 * kLrhKPair ? lrhPilotKernel<8> : lrhPilotKernel<4>;
}

// ---------------------------------------------------------------------------------------------------------------
// K2: merge of the per-CTA lists, gather-dot, ticket finalize.
// ---------------------------------------------------------------------------------------------------------------

struct LrhMergeParams
{
    unsigned long long const* listKeys;
    int32_t const* listCounts;
    //! Final shared thresholds, each a lower bound of its row's Kc-th key.
    unsigned long long const* thresholds;
    half const* hidden;
    half const* lmHead;
    float* logits;
    uint32_t* tickets;
    int32_t* candidateIds;
    float* candidateLogProbs;
    int32_t rows;
    int32_t numLists;
    int32_t listHead;
    int32_t listCapacity;
    int32_t numCandidates;
    int32_t hiddenSize;
    int32_t vocabSize;
    int32_t useCluster;
};

struct LrhMergeLayout
{
    size_t hidden;
    size_t counts;
    size_t head;
    size_t candidates;
    size_t merged;
    size_t sorted;
    size_t sortKeys;
    size_t logits;
    size_t ids;
    size_t hist;
    size_t partials;
    size_t tails;
    size_t scratch;
    size_t total;
};

__host__ __device__ inline LrhMergeLayout lrhMergeLayout(int32_t hiddenSize, int32_t headCapacity)
{
    LrhMergeLayout layout{};
    size_t offset = 0;
    layout.hidden = offset;
    offset = lrhAlignUp(static_cast<size_t>(hiddenSize) * sizeof(half), 16);
    layout.counts = offset;
    offset += lrhAlignUp(static_cast<size_t>(headCapacity) * sizeof(int32_t), 16);
    layout.head = offset;
    offset += static_cast<size_t>(headCapacity) * sizeof(uint64_t);
    layout.candidates = offset;
    offset += static_cast<size_t>(kMergeCandidateCapacity) * sizeof(uint64_t);
    layout.merged = offset;
    offset += static_cast<size_t>(kLrhMaxCandidates) * sizeof(uint64_t);
    layout.sorted = offset;
    offset += static_cast<size_t>(kLrhMaxCandidates) * sizeof(uint64_t);
    layout.sortKeys = offset;
    offset += static_cast<size_t>(kLrhMaxCandidates) * sizeof(uint64_t);
    layout.logits = offset;
    offset += static_cast<size_t>(kLrhMaxCandidates) * sizeof(float);
    layout.ids = offset;
    offset += static_cast<size_t>(kLrhMaxCandidates) * sizeof(int32_t);
    layout.hist = offset;
    offset += static_cast<size_t>(kRadixBins) * sizeof(uint32_t);
    layout.partials = offset;
    offset += static_cast<size_t>(kLrhGatherSegments) * kLrhMaxCandidates * sizeof(float);
    layout.tails = offset;
    offset += lrhAlignUp(static_cast<size_t>(headCapacity) * sizeof(int32_t), 16);
    layout.scratch = lrhAlignUp(offset, 16);
    layout.total = layout.scratch + lrhAlignUp(sizeof(LrhBlockScratch), 16);
    return layout;
}

//! Appends keys of the calling warp to a shared list with one atomic per warp.
__device__ __forceinline__ void lrhWarpAppend(bool take, uint64_t key, uint64_t* list, int32_t capacity, int32_t* count)
{
    int32_t const lane = static_cast<int32_t>(threadIdx.x) % kLrhWarpSize;
    uint32_t const mask = __ballot_sync(kLrhFullMask, take);
    if (mask == 0U)
    {
        return;
    }
    int32_t base = 0;
    if (lane == __ffs(static_cast<int32_t>(mask)) - 1)
    {
        base = atomicAdd(count, __popc(mask));
    }
    base = __shfl_sync(kLrhFullMask, base, __ffs(static_cast<int32_t>(mask)) - 1);
    int32_t const pos = base + __popc(mask & ((1U << lane) - 1U));
    if (take && pos < capacity)
    {
        list[pos] = key;
    }
}

//! Selects the numCandidates largest keys of row `row` over all lists into merged[] (in no particular order).
__device__ void lrhMergeRow(LrhMergeParams const& p, int32_t row, int32_t* counts, uint64_t* head, int32_t* tails,
    uint64_t* candidates, uint64_t* merged, uint32_t* hist, LrhBlockScratch& scratch)
{
    int32_t const tid = static_cast<int32_t>(threadIdx.x);
    int32_t const warp = tid / kLrhWarpSize;
    int32_t const lane = tid % kLrhWarpSize;
    int32_t const kc = p.numCandidates;
    int32_t const listHead = p.listHead;
    int32_t const headCount = p.numLists * listHead;
    auto const listBase
        = [&](int32_t list) { return p.listKeys + (static_cast<int64_t>(list) * p.rows + row) * p.listCapacity; };
    // Lists, counts and thresholds are written by K0 and K1 (PDL prerequisites): read them through L2.
    uint64_t const tau = __ldcg(p.thresholds + row);
    auto const listCount
        = [&](int32_t list) { return __ldcg(p.listCounts + static_cast<int64_t>(list) * p.rows + row); };
    for (int32_t list = tid; list < p.numLists; list += kMergeThreads)
    {
        counts[list] = listCount(list);
    }
    // The lists hold at least Kc keys at or above the shared threshold. If that ever failed, the unfilled slots stay
    // invalid keys and become clamped ids instead of uninitialized indices.
    for (int32_t i = tid; i < kc; i += kMergeThreads)
    {
        merged[i] = kInvalidKey;
    }

    if (tid == 0)
    {
        scratch.candidateCount = 0;
        scratch.mergedCount = 0;
        scratch.tailCount = 0;
    }
    int32_t localHeads = 0;
    for (int32_t i = tid; i < headCount; i += kMergeThreads)
    {
        // The count and the key are loaded together (the list region always holds listHead slots); slots past the
        // count are masked afterwards.
        int32_t const list = i / listHead;
        int32_t const slot = i % listHead;
        int32_t const count = listCount(list);
        uint64_t const key = __ldcg(listBase(list) + slot);
        head[i] = slot < count ? key : kInvalidKey;
        localHeads += slot < count ? 1 : 0;
    }
    uint64_t unusedLo = ~0ULL;
    uint64_t unusedHi = 0ULL;
    lrhBlockRange(unusedLo, unusedHi, localHeads, scratch);

    // Lower bounds of the row's Kc-th key: the final shared threshold, and the Kc-th largest list head key (a subset of
    // the union).
    uint64_t bound = tau > 1ULL ? tau : 1ULL;
    if (localHeads >= kc)
    {
        uint64_t const headBound = lrhRadixSelect<true>(
            [&](auto const& f) {
                for (int32_t i = tid; i < headCount; i += kMergeThreads)
                {
                    if (head[i] != kInvalidKey)
                    {
                        f(head[i]);
                    }
                }
            },
            kc, hist, &scratch);
        bound = headBound > bound ? headBound : bound;
    }
    for (int32_t base = warp * kLrhWarpSize; base < headCount; base += kMergeThreads)
    {
        int32_t const i = base + lane;
        uint64_t const key = i < headCount ? head[i] : kInvalidKey;
        lrhWarpAppend(
            key != kInvalidKey && key >= bound, key, candidates, kMergeCandidateCapacity, &scratch.candidateCount);
    }
    // A list's tail keys are below its smallest head key, so only lists whose head stays above the bound are read.
    for (int32_t list = tid; list < p.numLists; list += kMergeThreads)
    {
        if (counts[list] > listHead)
        {
            uint64_t smallest = ~0ULL;
            for (int32_t i = 0; i < listHead; ++i)
            {
                uint64_t const key = head[static_cast<int64_t>(list) * listHead + i];
                smallest = key < smallest ? key : smallest;
            }
            if (smallest >= bound)
            {
                tails[atomicAdd(&scratch.tailCount, 1)] = list;
            }
        }
    }
    __syncthreads();
    for (int32_t t = warp; t < scratch.tailCount; t += kMergeWarps)
    {
        int32_t const list = tails[t];
        int32_t const count = counts[list];
        unsigned long long const* const keys = listBase(list);
        for (int32_t i0 = listHead; i0 < count; i0 += kLrhWarpSize)
        {
            int32_t const i = i0 + lane;
            uint64_t const key = i < count ? __ldcg(keys + i) : kInvalidKey;
            lrhWarpAppend(
                key != kInvalidKey && key >= bound, key, candidates, kMergeCandidateCapacity, &scratch.candidateCount);
        }
    }
    __syncthreads();
    int32_t const candidateCount = scratch.candidateCount;
    // Writes the Kc largest of the keys visit(f) enumerates to merged.
    auto const collect = [&](auto const& visit) {
        uint64_t const threshold = lrhRadixSelect<true>(visit, kc, hist, &scratch);
        visit([&](uint64_t key) {
            if (key >= threshold)
            {
                int32_t const pos = atomicAdd(&scratch.mergedCount, 1);
                if (pos < kc)
                {
                    merged[pos] = key;
                }
            }
        });
    };
    if (candidateCount <= kMergeCandidateCapacity)
    {
        collect([&](auto const& f) {
            for (int32_t i = tid; i < candidateCount; i += kMergeThreads)
            {
                f(candidates[i]);
            }
        });
    }
    else
    {
        collect([&](auto const& f) {
            for (int32_t list = warp; list < p.numLists; list += kMergeWarps)
            {
                int32_t const count = counts[list];
                unsigned long long const* const keys = listBase(list);
                for (int32_t i = lane; i < count; i += kLrhWarpSize)
                {
                    uint64_t const key = __ldcg(keys + i);
                    if (key >= bound)
                    {
                        f(key);
                    }
                }
            }
        });
    }
    __syncthreads();
}

//! Candidate id of a merged key, clamped to the vocabulary so a corrupted list can never index lmHead out of range.
__device__ __forceinline__ int32_t lrhMergedId(uint64_t key, int32_t vocabSize)
{
    return min(max(lrhKeyId(key), 0), vocabSize - 1);
}

//! Two CTAs per SM (at most 128 registers), so a pass of up to 64 rows fits in one wave.
__global__ void __launch_bounds__(kMergeThreads, 2) lrhMergeKernel(LrhMergeParams const p)
{
    extern __shared__ __align__(16) unsigned char smem[];
    LrhMergeLayout const layout = lrhMergeLayout(p.hiddenSize, p.numLists * p.listHead);
    auto* const hiddenRow = reinterpret_cast<half*>(smem + layout.hidden);
    auto* const counts = reinterpret_cast<int32_t*>(smem + layout.counts);
    auto* const head = reinterpret_cast<uint64_t*>(smem + layout.head);
    auto* const candidates = reinterpret_cast<uint64_t*>(smem + layout.candidates);
    auto* const merged = reinterpret_cast<uint64_t*>(smem + layout.merged);
    auto* const sorted = reinterpret_cast<uint64_t*>(smem + layout.sorted);
    auto* const sortKeys = reinterpret_cast<uint64_t*>(smem + layout.sortKeys);
    auto* const logits = reinterpret_cast<float*>(smem + layout.logits);
    auto* const ids = reinterpret_cast<int32_t*>(smem + layout.ids);
    auto* const hist = reinterpret_cast<uint32_t*>(smem + layout.hist);
    auto* const partials = reinterpret_cast<float*>(smem + layout.partials);
    auto* const tails = reinterpret_cast<int32_t*>(smem + layout.tails);
    auto& scratch = *reinterpret_cast<LrhBlockScratch*>(smem + layout.scratch);

    int32_t const row = static_cast<int32_t>(blockIdx.x);
    int32_t const part = static_cast<int32_t>(blockIdx.y);
    int32_t const parts = static_cast<int32_t>(gridDim.y);
    int32_t const tid = static_cast<int32_t>(threadIdx.x);
    int32_t const kc = p.numCandidates;

    // hidden is the plugin input and was complete before K0 started (K0 is not a PDL dependent), so it can be read
    // before the wait; the lists, counts and tickets are read only after it.
    auto const* const hiddenSrc = reinterpret_cast<uint4 const*>(p.hidden + static_cast<int64_t>(row) * p.hiddenSize);
    for (int32_t i = tid; i < p.hiddenSize / kLrhHalvesPerVector; i += kMergeThreads)
    {
        reinterpret_cast<uint4*>(hiddenRow)[i] = hiddenSrc[i];
    }
    lrhWaitPrerequisites();

    bool merger = true;
    // Whether this code shares the leader's merge through the cluster; device code compiled without cluster support
    // merges in every CTA even if the host asked for a cluster.
    bool clustered = false;
#if SUPPORTS_CLUSTER_LAUNCH && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    namespace cg = cooperative_groups;
    if (p.useCluster != 0)
    {
        clustered = true;
        merger = cg::this_cluster().block_rank() == 0U;
    }
#endif
    // Slot i of the row is gathered by the CTA that owns it and finalized from p.logits by the last CTA, so every CTA
    // must see the same slot order: a cluster shares its leader's merge, and redundant merges are sorted. The
    // finalize sorts the candidates by exact logit, so the slot order never reaches the output.
    bool const sharedOrder = parts == 1 || clustered;
    uint64_t* const slots = sharedOrder ? merged : sorted;
    if (merger)
    {
        lrhMergeRow(p, row, counts, head, tails, candidates, merged, hist, scratch);
        if (!sharedOrder)
        {
            lrhBlockSortDescending(merged, sorted, kc);
        }
    }
#if SUPPORTS_CLUSTER_LAUNCH && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    if (p.useCluster != 0)
    {
        cg::cluster_group cluster = cg::this_cluster();
        cluster.sync();
        if (!merger)
        {
            uint64_t const* const leader = cluster.map_shared_rank(slots, 0);
            for (int32_t i = tid; i < kc; i += kMergeThreads)
            {
                slots[i] = leader[i];
            }
        }
        cluster.sync();
    }
#endif
    __syncthreads();

    int32_t const slotBegin = part * kc / parts;
    int32_t const slotEnd = (part + 1) * kc / parts;
    for (int32_t i = tid; i < slotEnd - slotBegin; i += kMergeThreads)
    {
        ids[i] = lrhMergedId(slots[slotBegin + i], p.vocabSize);
    }
    __syncthreads();
    lrhBlockGatherDot(p.lmHead, p.hiddenSize, ids, slotEnd - slotBegin, hiddenRow, partials, logits);
    for (int32_t i = tid; i < slotEnd - slotBegin; i += kMergeThreads)
    {
        p.logits[static_cast<int64_t>(row) * kc + slotBegin + i] = logits[i];
    }
    __threadfence();
    __syncthreads();
    if (tid == 0)
    {
        // K0 zeroed the ticket in this call; atomicInc wraps it back to 0 at the last arrival.
        uint32_t const previous = atomicInc(&p.tickets[row], static_cast<uint32_t>(parts - 1));
        scratch.isLast = previous == static_cast<uint32_t>(parts - 1) ? 1 : 0;
    }
    __syncthreads();
    if (scratch.isLast == 0)
    {
        return;
    }
    __threadfence();
    for (int32_t i = tid; i < kc; i += kMergeThreads)
    {
        logits[i] = __ldcg(p.logits + static_cast<int64_t>(row) * kc + i);
        ids[i] = lrhMergedId(slots[i], p.vocabSize);
    }
    __syncthreads();
    lrhBlockFinalize(logits, ids, kc, sortKeys, sorted, p.candidateIds + static_cast<int64_t>(row) * kc,
        p.candidateLogProbs + static_cast<int64_t>(row) * kc);
}

// ---------------------------------------------------------------------------------------------------------------
// Host side.
// ---------------------------------------------------------------------------------------------------------------

size_t selectSharedBytes(int32_t bucket, int32_t stages, int32_t tileRows, int32_t capacity, int32_t rank)
{
    return lrhSelectLayout(
        kBucketRows[bucket], kBucketGroups[bucket] * kK1WarpsPerGroup, stages, tileRows, capacity, rank)
        .total;
}

int32_t selectThreads(int32_t bucket)
{
    return (kBucketGroups[bucket] * kK1WarpsPerGroup + 1) * kLrhWarpSize;
}

int32_t currentDeviceAttribute(cudaDeviceAttr attribute)
{
    int32_t device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    int32_t value = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&value, attribute, device));
    return value;
}

//! Raises a kernel's dynamic shared-memory limit to the current device's maximum, so runners built in one process
//! (from any prop) never lower each other's limit.
void allowFullSharedMemory(void const* kernel)
{
    CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
        currentDeviceAttribute(cudaDevAttrMaxSharedMemoryPerBlockOptin)));
}

int32_t blocksPerSm(void const* kernel, int32_t threads, size_t sharedBytes)
{
    int32_t blocks = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, kernel, threads, sharedBytes));
    return blocks;
}

//! Co-resident (1, G, 1) clusters of the merge kernel for G = 2..kMaxMergeGroups (0 where G cannot be placed).
void queryMergeClusters(LowRankCandidateHeadConfig& config)
{
#if SUPPORTS_CLUSTER_LAUNCH
    if (config.smVersion < 90 || currentDeviceAttribute(cudaDevAttrClusterLaunch) == 0)
    {
        return;
    }
    for (int32_t groups = 2; groups <= kMaxMergeGroups; ++groups)
    {
        cudaLaunchAttribute attribute{};
        attribute.id = cudaLaunchAttributeClusterDimension;
        attribute.val.clusterDim.x = 1;
        attribute.val.clusterDim.y = static_cast<uint32_t>(groups);
        attribute.val.clusterDim.z = 1;
        cudaLaunchConfig_t launch{};
        launch.gridDim = dim3(1, groups, 1);
        launch.blockDim = dim3(kMergeThreads);
        launch.dynamicSmemBytes = config.k2SharedBytes;
        launch.attrs = &attribute;
        launch.numAttrs = 1;
        int32_t clusters = 0;
        if (cudaOccupancyMaxActiveClusters(&clusters, lrhMergeKernel, &launch) != cudaSuccess)
        {
            // An unplaceable cluster size is reported as an error; clear it and treat it as zero clusters.
            (void) cudaGetLastError();
            clusters = 0;
        }
        config.k2ClustersPerWave[groups] = clusters;
    }
    config.useCluster = config.k2ClustersPerWave[2] > 0;
#else
    (void) config;
#endif
}

//! The projection runs one k split per kLrhProjectionKPairsPerSplit k-pairs (256 hidden columns). The split count
//! depends on hiddenSize only, so s (and every result) is the same on devices with different SM counts.
constexpr int32_t kLrhProjectionKPairsPerSplit = 8;

LowRankCandidateHeadConfig computeConfig(LowRankCandidateHeadShape const& shape, int32_t maxRows,
    cudaDeviceProp const& prop, LowRankCandidateHeadOverrides const& overrides)
{
    char const* const caller = "LowRankCandidateHeadRunner";
    lrhCheckShape(caller, shape);
    ELLM_CHECK(maxRows >= 1, std::string(caller) + ": maxRows must be positive");
    LowRankCandidateHeadConfig config;
    config.smVersion = prop.major * 10 + prop.minor;
    ELLM_CHECK(config.smVersion >= 80,
        std::string(caller) + ": requires SM80 or newer, got SM" + std::to_string(config.smVersion));
    config.numSMs = prop.multiProcessorCount;
    size_t const optInBytes = prop.sharedMemPerBlockOptin;

    if (overrides.capacity != 0)
    {
        ELLM_CHECK(overrides.capacity % kK1CapacityStep == 0 && overrides.capacity >= shape.numCandidates + 16
                && overrides.capacity <= kK1MaxCapacity,
            std::string(caller) + ": forced capacity must be a multiple of 32 in [Kc + 16, 512]");
    }

    // Tile, rows per pass and stages: the first (T, Rp, S) in (64, 32, 16) x (64, 32, 16) x (4, 3) whose selection
    // kernel fits. Smaller tiles only matter for large ranks on devices with little shared memory.
    int32_t tileRows = 0;
    int32_t baseCapacity = 0;
    int32_t rowsPerPass = 0;
    int32_t stages = 0;
    for (int32_t const t : {64, 32, 16})
    {
        if (stages != 0 || (overrides.capacity != 0 && overrides.capacity < shape.numCandidates + t))
        {
            continue;
        }
        // At most 256 + 64 + 64 rounded up to 32, below kK1MaxCapacity.
        int32_t const capacity = overrides.capacity != 0
            ? overrides.capacity
            : static_cast<int32_t>(divUp(shape.numCandidates + t + kK1CapacityHeadroom, kK1CapacityStep))
                * kK1CapacityStep;
        for (int32_t b = kNumBucketRows - 1; b >= 1 && stages == 0; --b)
        {
            for (int32_t const s : {4, 3})
            {
                if (stages == 0 && selectSharedBytes(b, s, t, capacity, shape.rank) <= optInBytes)
                {
                    tileRows = t;
                    baseCapacity = capacity;
                    rowsPerPass = kBucketRows[b];
                    stages = s;
                }
            }
        }
    }
    ELLM_CHECK(stages != 0,
        std::string(caller) + ": no configuration fits " + std::to_string(optInBytes) + " bytes of shared memory");
    config.rowsPerPass = rowsPerPass;
    config.tileRows = tileRows;
    config.stages = stages;
    config.projectionSplits
        = static_cast<int32_t>(divUp(divUp(shape.hiddenSize, kLrhKPair), kLrhProjectionKPairsPerSplit));

    int32_t const passRows = std::min(rowsPerPass, maxRows);
    int32_t const minGrid = static_cast<int32_t>(divUp(shape.vocabSize, kMaxSlabRows));
    int32_t const maxGrid = static_cast<int32_t>(divUp(shape.vocabSize, kSlabAlign));
    if (overrides.grid != 0)
    {
        ELLM_CHECK(overrides.grid >= minGrid, std::string(caller) + ": forced grid must be at least ceil(V / 65536)");
    }
    constexpr double kL2ResidentFraction = 0.75;
    double const upProjBytes = static_cast<double>(shape.vocabSize) * shape.rank * sizeof(half);
    config.evictFirstA = upProjBytes > kL2ResidentFraction * static_cast<double>(prop.l2CacheSize);
    bool const upProjInL2 = !config.evictFirstA;
    size_t maxHead = 0;
    config.numBuckets = 0;
    for (int32_t b = 0; b < kNumBucketRows && kBucketRows[b] <= rowsPerPass; ++b)
    {
        if (b > 0 && kBucketRows[b - 1] >= passRows)
        {
            break;
        }
        LowRankCandidateHeadBucket bucket;
        bucket.maxRows = kBucketRows[b];
        auto const* const kernel = reinterpret_cast<void const*>(lrhSelectKernelFor(b, shape.rank));
        allowFullSharedMemory(kernel);
        int32_t const threads = selectThreads(b);
        int32_t capacity = baseCapacity;
        size_t bytes = selectSharedBytes(b, stages, tileRows, capacity, shape.rank);
        int32_t const occupancy = blocksPerSm(kernel, threads, bytes);
        ELLM_CHECK(occupancy >= 1, std::string(caller) + ": selection kernel cannot launch");
        if (overrides.capacity == 0)
        {
            // Grow the survivor buffer while the occupancy stays the same: fewer compactions at no cost.
            while (capacity + kK1CapacityStep <= kK1MaxCapacity)
            {
                size_t const larger = selectSharedBytes(b, stages, tileRows, capacity + kK1CapacityStep, shape.rank);
                if (larger > optInBytes || blocksPerSm(kernel, threads, larger) < occupancy)
                {
                    break;
                }
                capacity += kK1CapacityStep;
                bytes = larger;
            }
        }
        bucket.capacity = capacity;
        bucket.sharedBytes = bytes;
        int32_t grid = overrides.grid;
        if (grid == 0)
        {
            grid = std::min(std::max(config.numSMs * occupancy, minGrid), maxGrid);
        }
        bucket.grid = grid;
        bucket.slabRows = static_cast<int32_t>(divUp(divUp(shape.vocabSize, grid), kSlabAlign)) * kSlabAlign;
        ELLM_CHECK(bucket.slabRows <= kMaxSlabRows, std::string(caller) + ": slab exceeds the 16-bit slab index");
        bucket.listHead = std::min(
            shape.numCandidates, std::max(kMinListHead, static_cast<int32_t>(divUp(2 * shape.numCandidates, grid))));
        // Pilot witnesses: one per sampled 64-row tile (at least one per SM, at most one per tile), 2 x Kc of them,
        // or 4 x Kc when upProj stays in L2: re-reading the sampled tiles is then nearly free, and the tighter bound
        // lets fewer scores into the survivor buffers of the latency-bound selection kernel.
        int32_t const vocabTiles = shape.vocabSize / kPilotTileRows;
        int32_t const witnessesPerCandidate = upProjInL2 && !overrides.fewerWitnesses ? 4 : 2;
        int32_t const witnesses = std::min(
            {vocabTiles, kPilotMaxWitnesses, std::max(witnessesPerCandidate * shape.numCandidates, config.numSMs)});
        bool const pilotDefault = upProjInL2 || bucket.maxRows > kPilotMinRows;
        bool const pilot = overrides.pilot && pilotDefault;
        bucket.pilotTiles = pilot && witnesses >= shape.numCandidates ? witnesses : 0;
        maxHead = std::max(maxHead, static_cast<size_t>(grid) * bucket.listHead);
        config.buckets[config.numBuckets++] = bucket;
    }
    config.k2SharedBytes = lrhMergeLayout(shape.hiddenSize, static_cast<int32_t>(maxHead)).total;
    ELLM_CHECK(config.k2SharedBytes <= optInBytes,
        std::string(caller) + ": merge kernel shared memory exceeds the device limit");
    auto const* const mergeKernel = reinterpret_cast<void const*>(lrhMergeKernel);
    allowFullSharedMemory(mergeKernel);
    int32_t const mergeBlocks = blocksPerSm(mergeKernel, kMergeThreads, config.k2SharedBytes);
    ELLM_CHECK(mergeBlocks >= 1, std::string(caller) + ": merge kernel cannot launch");
    config.k2CtasPerWave = mergeBlocks * config.numSMs;
    queryMergeClusters(config);
#if SUPPORTS_PROGRAMMATIC_DEPENDENT_LAUNCH
    config.usePdl = config.smVersion >= 90;
#endif
    return config;
}

struct WorkspaceLayout
{
    size_t witnesses;
    size_t partials;
    size_t projected;
    size_t thresholds;
    size_t tickets;
    size_t logits;
    size_t listCounts;
    size_t listKeys;
    size_t total;
};

WorkspaceLayout workspaceLayout(
    LowRankCandidateHeadShape const& shape, int32_t maxRows, LowRankCandidateHeadConfig const& config)
{
    int32_t const passRows = std::min(maxRows, config.rowsPerPass);
    size_t maxLists = 0;
    size_t maxListKeys = 0;
    size_t maxWitnesses = 0;
    for (int32_t b = 0; b < config.numBuckets; ++b)
    {
        LowRankCandidateHeadBucket const& bucket = config.buckets[b];
        size_t const lists = static_cast<size_t>(bucket.grid) * std::min(bucket.maxRows, passRows);
        maxLists = std::max(maxLists, lists);
        maxListKeys = std::max(maxListKeys, lists * bucket.capacity);
        maxWitnesses
            = std::max(maxWitnesses, static_cast<size_t>(bucket.pilotTiles) * std::min(bucket.maxRows, passRows));
    }
    WorkspaceLayout layout{};
    size_t offset = 0;
    auto const take = [&](size_t bytes) {
        size_t const at = offset;
        offset += alignSpeculativeWorkspaceSize(bytes);
        return at;
    };
    layout.witnesses = take(maxWitnesses * sizeof(uint64_t));
    layout.partials = take(static_cast<size_t>(config.projectionSplits) * passRows * shape.rank * sizeof(float));
    layout.projected = take(static_cast<size_t>(passRows) * shape.rank * sizeof(half));
    layout.thresholds = take(static_cast<size_t>(passRows) * sizeof(uint64_t));
    layout.tickets = take(static_cast<size_t>(passRows) * sizeof(uint32_t));
    layout.logits = take(static_cast<size_t>(passRows) * shape.numCandidates * sizeof(float));
    layout.listCounts = take(maxLists * sizeof(int32_t));
    layout.listKeys = take(maxListKeys * sizeof(uint64_t));
    layout.total = offset;
    return layout;
}

//! Overrides from the device's preferred configuration towards smaller workspaces, each step cumulative on the last:
//! half the pilot witnesses, no pilot, survivor buffers (CAP) down to the smallest, then fewer K1 CTAs per pass (P)
//! down to ceil(V / 65536). The sizes are not monotone along the ladder (a smaller CAP can raise K1 occupancy and with
//! it the default P), so callers take the first step that fits or the minimum over all steps.
std::vector<LowRankCandidateHeadOverrides> shrinkingOverrides(
    LowRankCandidateHeadShape const& shape, int32_t maxRows, cudaDeviceProp const& prop)
{
    LowRankCandidateHeadOverrides step{};
    LowRankCandidateHeadConfig const preferred = computeConfig(shape, maxRows, prop, step);
    LowRankCandidateHeadBucket const& top = preferred.buckets[preferred.numBuckets - 1];
    std::vector<LowRankCandidateHeadOverrides> ladder{step};
    step.fewerWitnesses = true;
    ladder.push_back(step);
    step.pilot = false;
    ladder.push_back(step);
    // The smallest CAP holds Kc plus one tile of the smallest tile height (16 rows).
    int32_t const minCapacity
        = static_cast<int32_t>(divUp(shape.numCandidates + 16, kK1CapacityStep)) * kK1CapacityStep;
    for (int32_t capacity = top.capacity - kK1CapacityStep; capacity >= minCapacity; capacity -= kK1CapacityStep)
    {
        step.capacity = capacity;
        ladder.push_back(step);
    }
    int32_t const minGrid = static_cast<int32_t>(divUp(shape.vocabSize, kMaxSlabRows));
    for (int32_t grid = top.grid; grid > minGrid;)
    {
        grid = std::max(minGrid, grid / 2);
        step.grid = grid;
        ladder.push_back(step);
    }
    return ladder;
}

int32_t bucketIndexForRows(LowRankCandidateHeadConfig const& config, int32_t rows)
{
    int32_t b = 0;
    while (b + 1 < config.numBuckets && rows > config.buckets[b].maxRows)
    {
        ++b;
    }
    return b;
}

//! Pilot tile w covers vocab rows [w x stride x 64, + 64): the witnesses spread evenly over the vocabulary.
int32_t pilotTileStride(LowRankCandidateHeadShape const& shape, LowRankCandidateHeadBucket const& bucket)
{
    return std::max(1, (shape.vocabSize / kPilotTileRows) / bucket.pilotTiles);
}

//! G = min(k2CtasPerWave / rows, kMaxK2Groups), at least 1, lowered until rows clusters of G CTAs are co-resident.
int32_t mergeGroupsForRows(LowRankCandidateHeadConfig const& config, int32_t rows)
{
    int32_t groups = std::max(1, std::min(config.k2CtasPerWave / std::max(rows, 1), kMaxMergeGroups));
    if (config.useCluster)
    {
        while (groups > 1 && config.k2ClustersPerWave[groups] < rows)
        {
            --groups;
        }
    }
    return groups;
}

struct HeadPointers
{
    half const* hidden;
    half const* downProj;
    half const* upProj;
    half const* lmHead;
    int32_t* candidateIds;
    float* candidateLogProbs;
};

//! Launches K0, the pilot bound when the pass's bucket runs one, K1 and K2 over rows [passBegin, passBegin + passRows).
void launchPass(LowRankCandidateHeadShape const& shape, LowRankCandidateHeadConfig const& config,
    WorkspaceLayout const& layout, HeadPointers const& ptrs, int32_t passBegin, int32_t passRows, void* workspace,
    cudaStream_t stream)
{
    auto* const base = static_cast<char*>(workspace);
    auto* const witnesses = reinterpret_cast<unsigned long long*>(base + layout.witnesses);
    auto* const partials = reinterpret_cast<float*>(base + layout.partials);
    auto* const projected = reinterpret_cast<half*>(base + layout.projected);
    auto* const thresholds = reinterpret_cast<unsigned long long*>(base + layout.thresholds);
    auto* const tickets = reinterpret_cast<uint32_t*>(base + layout.tickets);
    auto* const logits = reinterpret_cast<float*>(base + layout.logits);
    auto* const listCounts = reinterpret_cast<int32_t*>(base + layout.listCounts);
    auto* const listKeys = reinterpret_cast<unsigned long long*>(base + layout.listKeys);
    int32_t const b = bucketIndexForRows(config, passRows);
    LowRankCandidateHeadBucket const& bucket = config.buckets[b];
    half const* const hidden = ptrs.hidden + static_cast<int64_t>(passBegin) * shape.hiddenSize;

    // K0a is the chain head after the previous layer, so it is not a PDL dependent.
    lrhLaunch(lrhProjectionPartialKernel, dim3(shape.rank / kLrhMmaM, config.projectionSplits),
        dim3(kLrhProjectionThreads), 0, stream, false, 1, hidden, ptrs.downProj, partials, passRows, shape.hiddenSize,
        shape.rank);
    lrhLaunch(lrhProjectionReduceKernel, dim3(divUp(passRows * shape.rank, kLrhReduceThreads)), dim3(kLrhReduceThreads),
        0, stream, config.usePdl, 1, static_cast<float const*>(partials), projected, passRows, shape.rank,
        config.projectionSplits, thresholds, tickets);
    if (bucket.pilotTiles > 0)
    {
        LrhPilotParams pilot{};
        pilot.upProj = ptrs.upProj;
        pilot.projected = projected;
        pilot.witnesses = witnesses;
        pilot.thresholds = thresholds;
        pilot.vocabSize = shape.vocabSize;
        pilot.rank = shape.rank;
        pilot.rows = passRows;
        pilot.tileStride = pilotTileStride(shape, bucket);
        pilot.numWitnesses = bucket.pilotTiles;
        pilot.numCandidates = shape.numCandidates;
        lrhLaunch(lrhPilotKernelFor(shape.rank), dim3(bucket.pilotTiles), dim3(kPilotThreads), 0, stream, config.usePdl,
            1, pilot);
        lrhLaunch(lrhPilotBoundKernel, dim3(passRows), dim3(kLrhWarpSize), 0, stream, config.usePdl, 1, pilot);
    }

    LrhSelectParams select{};
    select.upProj = ptrs.upProj;
    select.projected = projected;
    select.thresholds = thresholds;
    select.listKeys = listKeys;
    select.listCounts = listCounts;
    select.vocabSize = shape.vocabSize;
    select.rank = shape.rank;
    select.rows = passRows;
    select.slabRows = bucket.slabRows;
    select.capacity = bucket.capacity;
    select.tileRows = config.tileRows;
    select.stages = config.stages;
    select.numCandidates = shape.numCandidates;
    select.listHead = bucket.listHead;
    select.evictFirst = config.evictFirstA ? 1 : 0;
    lrhLaunch(lrhSelectKernelFor(b, shape.rank), dim3(bucket.grid), dim3(selectThreads(b)), bucket.sharedBytes, stream,
        config.usePdl, 1, select);

    LrhMergeParams merge{};
    merge.listKeys = listKeys;
    merge.listCounts = listCounts;
    merge.thresholds = thresholds;
    merge.hidden = hidden;
    merge.lmHead = ptrs.lmHead;
    merge.logits = logits;
    merge.tickets = tickets;
    merge.candidateIds = ptrs.candidateIds + static_cast<int64_t>(passBegin) * shape.numCandidates;
    merge.candidateLogProbs = ptrs.candidateLogProbs + static_cast<int64_t>(passBegin) * shape.numCandidates;
    merge.rows = passRows;
    merge.numLists = bucket.grid;
    merge.listHead = bucket.listHead;
    merge.listCapacity = bucket.capacity;
    merge.numCandidates = shape.numCandidates;
    merge.hiddenSize = shape.hiddenSize;
    merge.vocabSize = shape.vocabSize;
    int32_t const groups = mergeGroupsForRows(config, passRows);
    merge.useCluster = config.useCluster && groups > 1 ? 1 : 0;
    lrhLaunch(lrhMergeKernel, dim3(passRows, groups), dim3(kMergeThreads), config.k2SharedBytes, stream, config.usePdl,
        merge.useCluster != 0 ? groups : 1, merge);
}

} // namespace

LowRankCandidateHeadRunner::LowRankCandidateHeadRunner(
    LowRankCandidateHeadShape shape, int32_t maxRows, cudaDeviceProp const& prop)
    : LowRankCandidateHeadRunner(shape, maxRows, prop, LowRankCandidateHeadOverrides{})
{
}

LowRankCandidateHeadRunner::LowRankCandidateHeadRunner(LowRankCandidateHeadShape shape, int32_t maxRows,
    cudaDeviceProp const& prop, LowRankCandidateHeadOverrides const& overrides)
    : mShape(shape)
    , mMaxRows(maxRows)
    , mConfig(computeConfig(shape, maxRows, prop, overrides))
{
    mWorkspaceBytes = workspaceLayout(mShape, mMaxRows, mConfig).total;
}

size_t LowRankCandidateHeadRunner::workspaceBytes() const noexcept
{
    return mWorkspaceBytes;
}

size_t LowRankCandidateHeadRunner::workspaceBytes(
    LowRankCandidateHeadShape shape, int32_t maxRows, cudaDeviceProp const& prop)
{
    return workspaceLayout(shape, maxRows, computeConfig(shape, maxRows, prop, LowRankCandidateHeadOverrides{})).total;
}

size_t LowRankCandidateHeadRunner::minimumWorkspaceBytes(
    LowRankCandidateHeadShape shape, int32_t maxRows, cudaDeviceProp const& prop)
{
    size_t minimum = std::numeric_limits<size_t>::max();
    for (LowRankCandidateHeadOverrides const& overrides : shrinkingOverrides(shape, maxRows, prop))
    {
        minimum
            = std::min(minimum, workspaceLayout(shape, maxRows, computeConfig(shape, maxRows, prop, overrides)).total);
    }
    return minimum;
}

LowRankCandidateHeadRunner LowRankCandidateHeadRunner::withinWorkspace(
    LowRankCandidateHeadShape shape, int32_t maxRows, cudaDeviceProp const& prop, size_t workspaceLimit)
{
    for (LowRankCandidateHeadOverrides const& overrides : shrinkingOverrides(shape, maxRows, prop))
    {
        if (workspaceLayout(shape, maxRows, computeConfig(shape, maxRows, prop, overrides)).total <= workspaceLimit)
        {
            return LowRankCandidateHeadRunner(shape, maxRows, prop, overrides);
        }
    }
    throw std::runtime_error("LowRankCandidateHeadRunner: no configuration fits " + std::to_string(workspaceLimit)
        + " workspace bytes; the smallest needs " + std::to_string(minimumWorkspaceBytes(shape, maxRows, prop)));
}

void LowRankCandidateHeadRunner::run(rt::Tensor const& hidden, rt::Tensor const& downProj, rt::Tensor const& upProj,
    rt::Tensor const& lmHead, rt::Tensor& candidateIds, rt::Tensor& candidateLogProbs, void* workspace,
    cudaStream_t stream) const
{
    char const* const caller = "LowRankCandidateHeadRunner::run";
    int32_t const rows
        = lrhCheckTensors(caller, mShape, hidden, downProj, upProj, lmHead, candidateIds, candidateLogProbs);
    ELLM_CHECK(rows <= mMaxRows,
        std::string(caller) + ": " + std::to_string(rows) + " rows exceed maxRows " + std::to_string(mMaxRows));
    ELLM_CHECK(rows == 0 || (workspace != nullptr && (reinterpret_cast<uintptr_t>(workspace) & 15U) == 0U),
        std::string(caller) + ": workspace must be non-null and 16-byte aligned");
    if (rows == 0)
    {
        return;
    }
    HeadPointers const ptrs{hidden.dataPointer<half>(), downProj.dataPointer<half>(), upProj.dataPointer<half>(),
        lmHead.dataPointer<half>(), candidateIds.dataPointer<int32_t>(), candidateLogProbs.dataPointer<float>()};
    WorkspaceLayout const layout = workspaceLayout(mShape, mMaxRows, mConfig);
    for (int32_t passBegin = 0; passBegin < rows; passBegin += mConfig.rowsPerPass)
    {
        int32_t const passRows = std::min(mConfig.rowsPerPass, rows - passBegin);
        launchPass(mShape, mConfig, layout, ptrs, passBegin, passRows, workspace, stream);
    }
}

} // namespace kernel
} // namespace trt_edgellm
