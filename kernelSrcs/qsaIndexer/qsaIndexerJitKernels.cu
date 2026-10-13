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

// QSA indexer device kernels, NVRTC dialect. Compiled at runtime by
// cpp/kernels/qsaIndexer/qsaIndexerJitCompiler.cpp; the host launchers live in
// cpp/kernels/qsaIndexer/qsaIndexerKernels.cpp. The exact numerics documented in
// cpp/kernels/qsaIndexer/qsaIndexerKernels.h are pinned by unit tests: keep the
// device code bit-identical when editing either side.

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#if !defined(QSA_INDEXER_DATA_TYPE) || !defined(QSA_INDEXER_NUM_HEADS) || !defined(QSA_INDEXER_HEAD_DIM)               \
    || !defined(QSA_INDEXER_COMPRESS_RATIO) || !defined(QSA_INDEXER_INDEX_BUDGET) || !defined(QSA_INDEXER_INDEX_WIDTH) \
    || !defined(QSA_INDEXER_ROTARY_DIM) || !defined(QSA_INDEXER_TOPK_THREADS) || !defined(QSA_INDEXER_TOKENS_PER_PAGE) \
    || !defined(QSA_INDEXER_UNUSED_PAGE_ENTRY) || !defined(QSA_INDEXER_SCORES_COLUMNS_PER_CTA)                          \
    || !defined(QSA_INDEXER_SCORES_COLUMNS_PER_WARP)
#error "QSA indexer JIT configuration is incomplete"
#endif

// NVRTC has no <cstdint>/<cfloat>; spell out the fixed-width types and FLT_MAX.
using int32_t = int;
using int64_t = long long;
using uint32_t = unsigned int;
using uintptr_t = unsigned long long;
static_assert(sizeof(int32_t) == 4, "int32_t must be 4 bytes");
static_assert(sizeof(int64_t) == 8, "int64_t must be 8 bytes");
static_assert(sizeof(uint32_t) == 4, "uint32_t must be 4 bytes");
static_assert(sizeof(uintptr_t) == sizeof(void*), "uintptr_t must hold a pointer");

namespace
{

#if QSA_INDEXER_DATA_TYPE == 0
using QsaType = half;
#elif QSA_INDEXER_DATA_TYPE == 1
using QsaType = __nv_bfloat16;
#else
#error "Unsupported QSA indexer data type"
#endif

//! FLT_MAX; masked logits must compare equal to the host's -FLT_MAX bit for bit.
constexpr float kFloatMax = 3.402823466e+38F;

//! QSA indexer geometry. The values arrive from the host kQSA_* constants in
//! cpp/kernels/qsaIndexer/qsaIndexerKernels.h through the -D defines built by
//! cpp/kernels/qsaIndexer/qsaIndexerJitCompiler.cpp, so there is one definition.
constexpr int32_t kQSA_INDEXER_NUM_HEADS = QSA_INDEXER_NUM_HEADS; //!< Index-Q heads.
constexpr int32_t kQSA_INDEXER_HEAD_DIM = QSA_INDEXER_HEAD_DIM;   //!< Index-Q / index-K head dimension.
//! Raw index-K tokens averaged per compressed block.
constexpr int32_t kQSA_COMPRESS_RATIO = QSA_INDEXER_COMPRESS_RATIO;
//! Blocks selected per query row.
constexpr int32_t kQSA_BLOCK_TOPK = QSA_INDEXER_INDEX_BUDGET / QSA_INDEXER_COMPRESS_RATIO;
static_assert(kQSA_BLOCK_TOPK * kQSA_COMPRESS_RATIO == QSA_INDEXER_INDEX_BUDGET,
    "QSA_INDEXER_INDEX_BUDGET must be a whole number of compressed blocks");
constexpr int32_t kQSA_INDEX_WIDTH = QSA_INDEXER_INDEX_WIDTH; //!< Expanded token indices per query row.
static_assert(kQSA_INDEX_WIDTH == QSA_INDEXER_INDEX_BUDGET + kQSA_COMPRESS_RATIO - 1,
    "QSA_INDEXER_INDEX_WIDTH must be the budget plus the (ratio - 1) causal tail slots");
constexpr int32_t kQSA_INDEXER_ROTARY_DIM = QSA_INDEXER_ROTARY_DIM;
static_assert(kQSA_INDEXER_ROTARY_DIM % 2 == 0 && kQSA_INDEXER_ROTARY_DIM <= kQSA_INDEXER_HEAD_DIM,
    "neox rope pairs (d, d + rotary / 2) must stay inside the head");

//! Paged KV pool geometry: rt::kTOKENS_PER_PAGE and rt::kUNUSED_PAGE_ENTRY
//! (cpp/common/pagedKvTypes.h), supplied through the same -D defines.
constexpr int32_t kTOKENS_PER_PAGE = QSA_INDEXER_TOKENS_PER_PAGE;
constexpr int32_t kUNUSED_PAGE_ENTRY = QSA_INDEXER_UNUSED_PAGE_ENTRY;

constexpr int32_t kWarpSize = 32;
constexpr int32_t kDimsPerLane = kQSA_INDEXER_HEAD_DIM / kWarpSize;
constexpr int32_t kIndexQkRowWidth = (kQSA_INDEXER_NUM_HEADS + 1) * kQSA_INDEXER_HEAD_DIM;
constexpr int32_t kQNormedRowWidth = kQSA_INDEXER_NUM_HEADS * kQSA_INDEXER_HEAD_DIM;

//! K2/B2 launch geometry (host kQSA_INDEXER_SCORES_*): each warp scores kScoresColumnsPerWarp
//! block-columns of a kScoresColumnsPerCta tile, one warp per column group.
constexpr int32_t kScoresColumnsPerCta = QSA_INDEXER_SCORES_COLUMNS_PER_CTA;
constexpr int32_t kScoresColumnsPerWarp = QSA_INDEXER_SCORES_COLUMNS_PER_WARP;

// The kernel bodies unroll 4 heads, 4 tokens per block and kDimsPerLane dims per lane by
// hand, and B2 serves a warp's block-columns from one page-table lookup: refuse any
// geometry they do not implement instead of compiling to silently wrong numerics.
static_assert(kQSA_INDEXER_NUM_HEADS == 4 && kQSA_COMPRESS_RATIO == 4,
    "the indexer kernels are written for 4 index-Q heads and 4-token blocks");
static_assert(kQSA_INDEXER_HEAD_DIM % kWarpSize == 0, "each lane must own a whole slice of the head");
static_assert(kScoresColumnsPerCta % kScoresColumnsPerWarp == 0, "K2/B2 CTA tile must be whole warps");
static_assert(kTOKENS_PER_PAGE % (kQSA_COMPRESS_RATIO * kScoresColumnsPerWarp) == 0,
    "a warp's block-columns must live in one page");
constexpr int32_t kRawKColumnOffset = kQSA_INDEXER_NUM_HEADS * kQSA_INDEXER_HEAD_DIM;      // 512
constexpr int32_t kHalfRotaryDim = kQSA_INDEXER_ROTARY_DIM / 2;                            // 32
constexpr uint32_t kFullWarpMask = 0xffffffffU;

template <typename T>
__device__ __forceinline__ float toFloat(T const& v);

template <>
__device__ __forceinline__ float toFloat<half>(half const& v)
{
    return __half2float(v);
}

template <>
__device__ __forceinline__ float toFloat<__nv_bfloat16>(__nv_bfloat16 const& v)
{
    return __bfloat162float(v);
}

template <typename T>
__device__ __forceinline__ T fromFloat(float v);

template <>
__device__ __forceinline__ half fromFloat<half>(float v)
{
    return __float2half(v);
}

template <>
__device__ __forceinline__ __nv_bfloat16 fromFloat<__nv_bfloat16>(float v)
{
    return __float2bfloat16(v);
}

//! Butterfly warp reduction; returns the 32-lane sum replicated to every lane.
__device__ __forceinline__ float warpReduceSum(float v)
{
#pragma unroll
    for (int32_t offset = kWarpSize / 2; offset > 0; offset >>= 1)
    {
        v += __shfl_xor_sync(kFullWarpMask, v, offset);
    }
    return v;
}

//! Device-side view of the host QsaIndexerPoolState (cpp/kernels/qsaIndexer/
//! qsaIndexerKernels.h). The host launchers pass the fields individually through the launch
//! parameter pack; each kernel reassembles this view at entry. Field meaning and the
//! addressing contract are documented on the host struct — keep the two in sync.
struct QsaIndexerPoolView
{
    void* poolPtr;            //!< [2*numPages, 128, numKVHeads, poolHeadDim] QsaType base
    int32_t const* pageTable; //!< [B, 2, maxPagesPerSeq]; V ids pre-offset +numPages
    int32_t maxPagesPerSeq;
    int32_t numPages;    //!< K-plane page count
    int32_t numKVHeads;  //!< 2
    int32_t poolHeadDim; //!< 384
    int32_t headSize;    //!< 256 == tail column offset
};

//! Page id for `token` on `plane` (0 = K, 1 = V) of sequence `batchIdx`, or
//! kUNUSED_PAGE_ENTRY when the slot is out of range, unallocated, or out of pool bounds.
//! V-plane ids are stored pre-offset by +numPages — never re-add the offset.
__device__ __forceinline__ int32_t qsaPoolPage(
    QsaIndexerPoolView const& pool, int32_t batchIdx, int32_t plane, int32_t token)
{
    int32_t const slot = token / kTOKENS_PER_PAGE;
    if (slot >= pool.maxPagesPerSeq)
    {
        return kUNUSED_PAGE_ENTRY;
    }
    int32_t const page = pool.pageTable[(static_cast<int64_t>(batchIdx) * 2 + plane) * pool.maxPagesPerSeq + slot];
    return (page >= 0 && page < 2 * pool.numPages) ? page : kUNUSED_PAGE_ENTRY;
}

//! Pointer to `token`'s tail column headSize on `page` (head 0) — the first of the 128
//! tail elements. See the host QsaIndexerPoolState for the element-offset contract.
template <typename T>
__device__ __forceinline__ T* qsaPoolTailPtr(QsaIndexerPoolView const& pool, T* poolBase, int32_t page, int32_t token)
{
    int64_t const rowStride = static_cast<int64_t>(pool.numKVHeads) * pool.poolHeadDim;
    return poolBase + (static_cast<int64_t>(page) * kTOKENS_PER_PAGE + token % kTOKENS_PER_PAGE) * rowStride
        + pool.headSize;
}

//! Neox (half-rotation) partial rope on one FP32 head staged in shared memory.
//! Dim d < 32: y = x[d]*cos[d] - x[d+32]*sin[d]; 32 <= d < 64: y = x[d]*cos[d-32] + x[d-32]*sin[d-32];
//! d >= 64: passthrough. cosSinRow points at cosSin[pos][0] (cos [0:32], sin [32:64]).
__device__ __forceinline__ float applyNeoxRopeFromSmem(float const* headSmem, float const* cosSinRow, int32_t d)
{
    if (d < kHalfRotaryDim)
    {
        return headSmem[d] * cosSinRow[d] - headSmem[d + kHalfRotaryDim] * cosSinRow[kHalfRotaryDim + d];
    }
    if (d < kQSA_INDEXER_ROTARY_DIM)
    {
        int32_t const j = d - kHalfRotaryDim;
        return headSmem[d] * cosSinRow[j] + headSmem[j] * cosSinRow[kHalfRotaryDim + j];
    }
    return headSmem[d];
}

//! K1b body shared by qsa_indexer_k_compress and qsa_indexer_k_compress_paged.
//! With kPagedOut each valid kbar row is ALSO written to the pool V-tail of the block's
//! first token; the dense output and numerics are unchanged.
template <bool kPagedOut>
__device__ __forceinline__ void qsaIndexerKCompressImpl(QsaType* kbar, QsaType const* indexQk, float const* cosSin,
    int32_t const* contextLengths, QsaType const* wK, float rmsEps, int32_t seqLen, int32_t numBlocks,
    int32_t numGroups, int32_t blockBegin, int32_t pastLen, QsaIndexerPoolView const& pool)
{
    int32_t const batchIdx = static_cast<int32_t>(blockIdx.x) / numGroups;
    int32_t const blockId = blockBegin + static_cast<int32_t>(blockIdx.x) % numGroups;
    int32_t const lane = static_cast<int32_t>(threadIdx.x);

    QsaType* out = kbar + (static_cast<int64_t>(batchIdx) * numBlocks + blockId) * kQSA_INDEXER_HEAD_DIM;

    int32_t const firstToken = blockId * kQSA_COMPRESS_RATIO;
    bool const valid = (firstToken + kQSA_COMPRESS_RATIO - 1 < contextLengths[batchIdx]) && (firstToken >= pastLen);
    if (!valid)
    {
        // Incomplete/out-of-range block: exact zeros, inputs never read.
#pragma unroll
        for (int32_t i = 0; i < kDimsPerLane; ++i)
        {
            out[lane * kDimsPerLane + i] = fromFloat<QsaType>(0.0f);
        }
        return;
    }

    __shared__ float sK[kQSA_INDEXER_HEAD_DIM];

    // Raw index-K rows of the 4 block tokens (columns [512, 640) of indexQk).
    QsaType const* k0 = indexQk + static_cast<int64_t>(batchIdx) * seqLen * kIndexQkRowWidth
        + static_cast<int64_t>(firstToken - pastLen) * kIndexQkRowWidth + kRawKColumnOffset;

    float m[kDimsPerLane];
    float sumSq = 0.0f;
#pragma unroll
    for (int32_t i = 0; i < kDimsPerLane; ++i)
    {
        int32_t const d = lane * kDimsPerLane + i;
        // FP32 mean in FIXED sequential order, then the bit-compatibility-critical cast to QsaType.
        float acc = toFloat(k0[d]);
        acc = acc + toFloat(k0[kIndexQkRowWidth + d]);
        acc = acc + toFloat(k0[2 * kIndexQkRowWidth + d]);
        acc = acc + toFloat(k0[3 * kIndexQkRowWidth + d]);
        acc = acc * 0.25f;
        m[i] = toFloat(fromFloat<QsaType>(acc));
        sumSq = fmaf(m[i], m[i], sumSq);
    }
    sumSq = warpReduceSum(sumSq);
    float const invRms = rsqrtf(sumSq / static_cast<float>(kQSA_INDEXER_HEAD_DIM) + rmsEps);

#pragma unroll
    for (int32_t i = 0; i < kDimsPerLane; ++i)
    {
        int32_t const d = lane * kDimsPerLane + i;
        sK[d] = m[i] * invRms * (1.0f + toFloat(wK[d]));
    }
    __syncwarp();

    QsaType* vTail = nullptr;
    if constexpr (kPagedOut)
    {
        int32_t const vPage = qsaPoolPage(pool, batchIdx, /* plane = */ 1, firstToken);
        if (vPage != kUNUSED_PAGE_ENTRY)
        {
            vTail = qsaPoolTailPtr(pool, static_cast<QsaType*>(pool.poolPtr), vPage, firstToken);
        }
    }

    // Rope at the position of the block's first token; single cast to QsaType at the very end.
    float const* cosSinRow = cosSin + static_cast<int64_t>(firstToken) * kQSA_INDEXER_ROTARY_DIM;
#pragma unroll
    for (int32_t i = 0; i < kDimsPerLane; ++i)
    {
        int32_t const d = lane * kDimsPerLane + i;
        QsaType const value = fromFloat<QsaType>(applyNeoxRopeFromSmem(sK, cosSinRow, d));
        out[d] = value;
        if constexpr (kPagedOut)
        {
            if (vTail != nullptr)
            {
                vTail[d] = value;
            }
        }
    }
}

constexpr int32_t kTopKThreads = QSA_INDEXER_TOPK_THREADS; //!< B3 CTA size.
constexpr int32_t kTopKWarps = kTopKThreads / kWarpSize;
constexpr int32_t kRadixBins = 256; //!< 8-bit digits, MSB first.
static_assert(kTopKThreads % kWarpSize == 0 && kTopKThreads <= 1024,
    "B3 ballots need whole warps, within the 1024-thread CTA limit");

//! Monotone orderable map of FP32 bits: u(a) > u(b) iff a sorts after b in cub's float
//! radix order (which orders -0.0f < +0.0f and NaNs by payload — ties stay bit-exact).
__device__ __forceinline__ uint32_t logitOrderKey(float v)
{
    uint32_t const bits = __float_as_uint(v);
    return (bits & 0x80000000U) != 0U ? ~bits : (bits | 0x80000000U);
}

} // namespace

//! K1a: see launchQsaIndexQPrep. grid(B*S), block(128); warp h owns q head h, 4 dims/lane.
extern "C" __global__ void qsa_indexer_q_prep(QsaType* qNormed, QsaType const* indexQk, float const* cosSin,
    int32_t const* contextLengths, QsaType const* wQ, float rmsEps, int32_t seqLen)
{
    int32_t const row = static_cast<int32_t>(blockIdx.x); // b * seqLen + t
    int32_t const batchIdx = row / seqLen;
    int32_t const tokenIdx = row % seqLen;
    int32_t const head = static_cast<int32_t>(threadIdx.x) / kWarpSize;
    int32_t const lane = static_cast<int32_t>(threadIdx.x) % kWarpSize;

    QsaType* out = qNormed + static_cast<int64_t>(row) * kQNormedRowWidth + head * kQSA_INDEXER_HEAD_DIM;

    if (tokenIdx >= contextLengths[batchIdx])
    {
        // Padding row: exact zeros, inputs never read.
#pragma unroll
        for (int32_t i = 0; i < kDimsPerLane; ++i)
        {
            out[lane * kDimsPerLane + i] = fromFloat<QsaType>(0.0f);
        }
        return;
    }

    __shared__ float sQ[kQSA_INDEXER_NUM_HEADS][kQSA_INDEXER_HEAD_DIM];

    QsaType const* in = indexQk + static_cast<int64_t>(row) * kIndexQkRowWidth + head * kQSA_INDEXER_HEAD_DIM;

    // FP32 sum-of-squares over the 128 head dims (4 elements per lane + warp reduction).
    float x[kDimsPerLane];
    float sumSq = 0.0f;
#pragma unroll
    for (int32_t i = 0; i < kDimsPerLane; ++i)
    {
        x[i] = toFloat(in[lane * kDimsPerLane + i]);
        sumSq = fmaf(x[i], x[i], sumSq);
    }
    sumSq = warpReduceSum(sumSq);
    float const invRms = rsqrtf(sumSq / static_cast<float>(kQSA_INDEXER_HEAD_DIM) + rmsEps);

    // Gemma norm with raw checkpoint gamma: y = x * invRms * (1 + w), all FP32, staged in smem.
#pragma unroll
    for (int32_t i = 0; i < kDimsPerLane; ++i)
    {
        int32_t const d = lane * kDimsPerLane + i;
        sQ[head][d] = x[i] * invRms * (1.0f + toFloat(wQ[d]));
    }
    __syncwarp();

    // Partial neox rope at position tokenIdx; single cast to QsaType at the very end.
    float const* cosSinRow = cosSin + static_cast<int64_t>(tokenIdx) * kQSA_INDEXER_ROTARY_DIM;
#pragma unroll
    for (int32_t i = 0; i < kDimsPerLane; ++i)
    {
        int32_t const d = lane * kDimsPerLane + i;
        out[d] = fromFloat<QsaType>(applyNeoxRopeFromSmem(sQ[head], cosSinRow, d));
    }
}

//! K1b: see launchQsaIndexKCompress. grid(B * numGroups), block(32); one warp per (b, g).
extern "C" __global__ void qsa_indexer_k_compress(QsaType* kbar, QsaType const* indexQk, float const* cosSin,
    int32_t const* contextLengths, QsaType const* wK, float rmsEps, int32_t seqLen, int32_t numBlocks,
    int32_t numGroups, int32_t blockBegin, int32_t pastLen)
{
    qsaIndexerKCompressImpl</* kPagedOut = */ false>(kbar, indexQk, cosSin, contextLengths, wK, rmsEps, seqLen,
        numBlocks, numGroups, blockBegin, pastLen, QsaIndexerPoolView{});
}

//! K1b paged variant: see launchQsaIndexKCompressPaged. Same grid/block and dense numerics
//! as qsa_indexer_k_compress; each valid kbar row is ALSO written to the pool V-tail of the
//! block's first token.
extern "C" __global__ void qsa_indexer_k_compress_paged(QsaType* kbar, QsaType const* indexQk, float const* cosSin,
    int32_t const* contextLengths, QsaType const* wK, float rmsEps, int32_t seqLen, int32_t numBlocks,
    int32_t numGroups, int32_t blockBegin, int32_t pastLen, void* poolPtr, int32_t const* pageTable,
    int32_t maxPagesPerSeq, int32_t numPages, int32_t numKVHeads, int32_t poolHeadDim, int32_t headSize)
{
    QsaIndexerPoolView const pool{poolPtr, pageTable, maxPagesPerSeq, numPages, numKVHeads, poolHeadDim, headSize};
    qsaIndexerKCompressImpl</* kPagedOut = */ true>(
        kbar, indexQk, cosSin, contextLengths, wK, rmsEps, seqLen, numBlocks, numGroups, blockBegin, pastLen, pool);
}

//! K2: see launchQsaIndexScores. grid(numRows, ceilDiv(numBlocks, 64)), block(256).
//! The CTA stages the row's 4x128 q in FP32 smem; each of the 8 warps owns 8 consecutive
//! block-columns of the 64-column tile and evaluates them one at a time.
extern "C" __global__ void qsa_indexer_scores(float* logits, QsaType const* qNormed, QsaType const* kbar,
    int32_t const* contextLengths, int32_t seqLen, int32_t numBlocks, int32_t rowStart)
{
    constexpr int32_t kColumnsPerCta = kScoresColumnsPerCta;
    constexpr int32_t kColumnsPerWarp = kScoresColumnsPerWarp;

    int32_t const chunkRow = static_cast<int32_t>(blockIdx.x);
    int32_t const row = rowStart + chunkRow;
    int32_t const batchIdx = row / seqLen;
    int32_t const tokenIdx = row % seqLen;
    int32_t const warp = static_cast<int32_t>(threadIdx.x) / kWarpSize;
    int32_t const lane = static_cast<int32_t>(threadIdx.x) % kWarpSize;
    int32_t const columnBase = static_cast<int32_t>(blockIdx.y) * kColumnsPerCta + warp * kColumnsPerWarp;

    int32_t const contextLen = contextLengths[batchIdx];
    bool const isPadding = (tokenIdx >= contextLen) || (contextLen == 0);
    int32_t const numVisible = isPadding ? 0 : (tokenIdx + 1) / kQSA_COMPRESS_RATIO;

    float* outRow = logits + static_cast<int64_t>(chunkRow) * numBlocks;

    __shared__ float sQ[kQNormedRowWidth];
    if (!isPadding)
    {
        QsaType const* q = qNormed + static_cast<int64_t>(row) * kQNormedRowWidth;
        for (int32_t i = static_cast<int32_t>(threadIdx.x); i < kQNormedRowWidth; i += static_cast<int32_t>(blockDim.x))
        {
            sQ[i] = toFloat(q[i]);
        }
    }
    __syncthreads();

    for (int32_t it = 0; it < kColumnsPerWarp; ++it)
    {
        int32_t const blockId = columnBase + it;
        if (blockId >= numBlocks)
        {
            break;
        }

        float score = -kFloatMax;
        if (blockId < numVisible)
        {
            QsaType const* kb = kbar + (static_cast<int64_t>(batchIdx) * numBlocks + blockId) * kQSA_INDEXER_HEAD_DIM;
            float partial[kQSA_INDEXER_NUM_HEADS] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
            for (int32_t i = 0; i < kDimsPerLane; ++i)
            {
                int32_t const d = lane + i * kWarpSize;
                float const kf = toFloat(kb[d]);
#pragma unroll
                for (int32_t h = 0; h < kQSA_INDEXER_NUM_HEADS; ++h)
                {
                    partial[h] = fmaf(sQ[h * kQSA_INDEXER_HEAD_DIM + d], kf, partial[h]);
                }
            }
#pragma unroll
            for (int32_t h = 0; h < kQSA_INDEXER_NUM_HEADS; ++h)
            {
                partial[h] = warpReduceSum(partial[h]);
            }
            // Relu PER HEAD, sum over heads (ascending h), scale AFTER the sum.
            float sum = fmaxf(0.0f, partial[0]);
            sum += fmaxf(0.0f, partial[1]);
            sum += fmaxf(0.0f, partial[2]);
            sum += fmaxf(0.0f, partial[3]);
            score = sum * (1.0f / sqrtf(static_cast<float>(kQSA_INDEXER_HEAD_DIM)));
        }
        if (lane == 0)
        {
            outRow[blockId] = score;
        }
    }
}

//! ids[r * numBlocks + c] = c for the segmented sort values.
extern "C" __global__ void qsa_indexer_ids_fill(int32_t* ids, int64_t numItems, int32_t numBlocks)
{
    int64_t const stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < numItems; i += stride)
    {
        ids[i] = static_cast<int32_t>(i % numBlocks);
    }
}

//! K4: see launchQsaIndexExpand. grid(numRows), block(256), threads stride the 2051 outputs.
extern "C" __global__ void qsa_indexer_expand(int32_t* outIdx, int32_t const* sortedIds, int32_t const* contextLengths,
    int32_t seqLen, int32_t numBlocks, int32_t rowStart)
{
    int32_t const chunkRow = static_cast<int32_t>(blockIdx.x);
    int32_t const row = rowStart + chunkRow;
    int32_t const batchIdx = row / seqLen;
    int32_t const tokenIdx = row % seqLen;
    int32_t const contextLen = contextLengths[batchIdx];
    bool const isPadding = (tokenIdx >= contextLen) || (contextLen == 0);

    int32_t numVisible = 0;
    int32_t numSelected = 0;
    int32_t tailCount = 0;
    if (!isPadding)
    {
        numVisible = (tokenIdx + 1) / kQSA_COMPRESS_RATIO;
        numSelected = min(kQSA_BLOCK_TOPK, numVisible);
        tailCount = (tokenIdx + 1) % kQSA_COMPRESS_RATIO;
    }

    int32_t const* rowIds = sortedIds + static_cast<int64_t>(chunkRow) * numBlocks;
    int32_t* out = outIdx + static_cast<int64_t>(row) * kQSA_INDEX_WIDTH;

    int32_t const expandedCount = numSelected * kQSA_COMPRESS_RATIO;
    for (int32_t j = static_cast<int32_t>(threadIdx.x); j < kQSA_INDEX_WIDTH; j += static_cast<int32_t>(blockDim.x))
    {
        int32_t value = -1;
        if (!isPadding)
        {
            if (j < expandedCount)
            {
                // Padding rows never reach this read: sortedIds may be garbage there.
                value = rowIds[j / kQSA_COMPRESS_RATIO] * kQSA_COMPRESS_RATIO + j % kQSA_COMPRESS_RATIO;
            }
            else if (j < expandedCount + tailCount)
            {
                value = numVisible * kQSA_COMPRESS_RATIO + (j - expandedCount);
            }
        }
        out[j] = value;
    }
}

//! Prefill tail scatter: see launchQsaRawKTailWritePrefill. grid(B, kQSA_COMPRESS_RATIO - 1),
//! block(32): CTA (b, j) persists token ctx_b - ctx_b % 4 + j when j < ctx_b % 4.
extern "C" __global__ void qsa_indexer_raw_k_tail_write_prefill(QsaType const* indexQk, int32_t const* contextLengths,
    void* poolPtr, int32_t const* pageTable, int32_t maxPagesPerSeq, int32_t numPages, int32_t numKVHeads,
    int32_t poolHeadDim, int32_t headSize, int32_t seqLen)
{
    QsaIndexerPoolView const pool{poolPtr, pageTable, maxPagesPerSeq, numPages, numKVHeads, poolHeadDim, headSize};

    int32_t const batchIdx = static_cast<int32_t>(blockIdx.x);
    int32_t const tailSlot = static_cast<int32_t>(blockIdx.y);
    int32_t const lane = static_cast<int32_t>(threadIdx.x);

    int32_t const contextLen = contextLengths[batchIdx];
    int32_t const tailCount = contextLen % kQSA_COMPRESS_RATIO; // <= 0 for padding sequences
    if (tailSlot >= tailCount)
    {
        return;
    }
    int32_t const tokenIdx = contextLen - tailCount + tailSlot;
    if (tokenIdx >= seqLen)
    {
        return;
    }
    int32_t const kPage = qsaPoolPage(pool, batchIdx, /* plane = */ 0, tokenIdx);
    if (kPage == kUNUSED_PAGE_ENTRY)
    {
        return;
    }
    int64_t const row = static_cast<int64_t>(batchIdx) * seqLen + tokenIdx;
    QsaType const* src = indexQk + row * kIndexQkRowWidth + kRawKColumnOffset;
    QsaType* dst = qsaPoolTailPtr(pool, static_cast<QsaType*>(pool.poolPtr), kPage, tokenIdx);
#pragma unroll
    for (int32_t i = 0; i < kDimsPerLane; ++i)
    {
        dst[lane * kDimsPerLane + i] = src[lane * kDimsPerLane + i]; // bit-unmodified
    }
}

//! B1: see launchQsaIndexerPreDecode. grid(B), block(128) = 4 warps; warp h owns q head h
//! in phase 1, warp 0 alone runs phase 2 (raw-K tail write) or phase 3 (compress).
extern "C" __global__ void qsa_indexer_pre_decode(QsaType* qNormed, QsaType const* indexQk, float const* cosSin,
    int32_t const* contextLengths, QsaType const* wQ, QsaType const* wK, float rmsEps, void* poolPtr,
    int32_t const* pageTable, int32_t maxPagesPerSeq, int32_t numPages, int32_t numKVHeads, int32_t poolHeadDim,
    int32_t headSize)
{
    QsaIndexerPoolView const pool{poolPtr, pageTable, maxPagesPerSeq, numPages, numKVHeads, poolHeadDim, headSize};

    int32_t const batchIdx = static_cast<int32_t>(blockIdx.x);
    int32_t const head = static_cast<int32_t>(threadIdx.x) / kWarpSize;
    int32_t const lane = static_cast<int32_t>(threadIdx.x) % kWarpSize;
    int32_t const contextLen = contextLengths[batchIdx];

    QsaType* out = qNormed + static_cast<int64_t>(batchIdx) * kQNormedRowWidth + head * kQSA_INDEXER_HEAD_DIM;
    if (contextLen <= 0)
    {
        // Decode rows are never padded; this guard only keeps an empty slot total.
#pragma unroll
        for (int32_t i = 0; i < kDimsPerLane; ++i)
        {
            out[lane * kDimsPerLane + i] = fromFloat<QsaType>(0.0f);
        }
        return;
    }
    int32_t const pos = contextLen - 1; // the new token's absolute position

    __shared__ float sQ[kQSA_INDEXER_NUM_HEADS][kQSA_INDEXER_HEAD_DIM];
    __shared__ float sK[kQSA_INDEXER_HEAD_DIM];

    // Phase 1: K1a numerics verbatim at S = 1. The rope position is pos, NOT the indexQk
    // row index (K1a's `row % seqLen` position only coincides with it in prefill).
    QsaType const* in = indexQk + static_cast<int64_t>(batchIdx) * kIndexQkRowWidth + head * kQSA_INDEXER_HEAD_DIM;
    float x[kDimsPerLane];
    float sumSq = 0.0f;
#pragma unroll
    for (int32_t i = 0; i < kDimsPerLane; ++i)
    {
        x[i] = toFloat(in[lane * kDimsPerLane + i]);
        sumSq = fmaf(x[i], x[i], sumSq);
    }
    sumSq = warpReduceSum(sumSq);
    float const invRms = rsqrtf(sumSq / static_cast<float>(kQSA_INDEXER_HEAD_DIM) + rmsEps);
#pragma unroll
    for (int32_t i = 0; i < kDimsPerLane; ++i)
    {
        int32_t const d = lane * kDimsPerLane + i;
        sQ[head][d] = x[i] * invRms * (1.0f + toFloat(wQ[d]));
    }
    __syncwarp();
    float const* cosSinRow = cosSin + static_cast<int64_t>(pos) * kQSA_INDEXER_ROTARY_DIM;
#pragma unroll
    for (int32_t i = 0; i < kDimsPerLane; ++i)
    {
        int32_t const d = lane * kDimsPerLane + i;
        out[d] = fromFloat<QsaType>(applyNeoxRopeFromSmem(sQ[head], cosSinRow, d));
    }

    if (head != 0)
    {
        return;
    }

    // Phase 2 (warp 0): the new token's raw index-K, persisted bit-unmodified only while its
    // block stays incomplete; otherwise phase 3 consumes it from registers.
    QsaType* const poolBase = static_cast<QsaType*>(pool.poolPtr);
    QsaType const* rawIn = indexQk + static_cast<int64_t>(batchIdx) * kIndexQkRowWidth + kRawKColumnOffset;
    QsaType rawK[kDimsPerLane];
#pragma unroll
    for (int32_t i = 0; i < kDimsPerLane; ++i)
    {
        rawK[i] = rawIn[lane * kDimsPerLane + i];
    }
    int32_t const kPage = qsaPoolPage(pool, batchIdx, /* plane = */ 0, pos);
    if (kPage == kUNUSED_PAGE_ENTRY)
    {
        return; // unallocated page: nothing persistable this step
    }
    if (contextLen % kQSA_COMPRESS_RATIO != 0)
    {
        QsaType* const kTail = qsaPoolTailPtr(pool, poolBase, kPage, pos);
#pragma unroll
        for (int32_t i = 0; i < kDimsPerLane; ++i)
        {
            kTail[lane * kDimsPerLane + i] = rawK[i];
        }
        return;
    }

    // Phase 3 (warp 0): this token completes block 4g..4g+3 — compress it.
    int32_t const firstToken = contextLen - kQSA_COMPRESS_RATIO; // == 4 * (contextLen/4 - 1)
    // Tokens 4g..4g+3 share pos's K page: a 4-token block never straddles a 128-token page.
    QsaType const* const tail0 = qsaPoolTailPtr(pool, poolBase, kPage, firstToken);
    QsaType const* const tail1 = qsaPoolTailPtr(pool, poolBase, kPage, firstToken + 1);
    QsaType const* const tail2 = qsaPoolTailPtr(pool, poolBase, kPage, firstToken + 2);
    float m[kDimsPerLane];
    float sumSqK = 0.0f;
#pragma unroll
    for (int32_t i = 0; i < kDimsPerLane; ++i)
    {
        int32_t const d = lane * kDimsPerLane + i;
        // K1b numerics verbatim: FP32 mean in FIXED sequential order, then the
        // bit-compatibility-critical cast to QsaType. Token 4g+3 comes from phase 2's registers.
        float acc = toFloat(tail0[d]);
        acc = acc + toFloat(tail1[d]);
        acc = acc + toFloat(tail2[d]);
        acc = acc + toFloat(rawK[i]);
        acc = acc * 0.25f;
        m[i] = toFloat(fromFloat<QsaType>(acc));
        sumSqK = fmaf(m[i], m[i], sumSqK);
    }
    sumSqK = warpReduceSum(sumSqK);
    float const invRmsK = rsqrtf(sumSqK / static_cast<float>(kQSA_INDEXER_HEAD_DIM) + rmsEps);
#pragma unroll
    for (int32_t i = 0; i < kDimsPerLane; ++i)
    {
        int32_t const d = lane * kDimsPerLane + i;
        sK[d] = m[i] * invRmsK * (1.0f + toFloat(wK[d]));
    }
    __syncwarp();

    int32_t const vPage = qsaPoolPage(pool, batchIdx, /* plane = */ 1, firstToken);
    if (vPage == kUNUSED_PAGE_ENTRY)
    {
        return;
    }
    QsaType* const vTail = qsaPoolTailPtr(pool, poolBase, vPage, firstToken);
    float const* cosSinBlock = cosSin + static_cast<int64_t>(firstToken) * kQSA_INDEXER_ROTARY_DIM;
#pragma unroll
    for (int32_t i = 0; i < kDimsPerLane; ++i)
    {
        int32_t const d = lane * kDimsPerLane + i;
        vTail[d] = fromFloat<QsaType>(applyNeoxRopeFromSmem(sK, cosSinBlock, d));
    }
}

//! B2: see launchQsaIndexScoresDecode. grid(B, ceilDiv(maxBlocks, 64)), block(256).
//! K2's smem-staged q / warp-per-column / lane-strided FMA structure at S = 1, with kbar
//! gathered from the pool V-tails; only the valid prefix [0, nVis) is written.
extern "C" __global__ void qsa_indexer_scores_decode(float* logits, QsaType const* qNormed,
    int32_t const* contextLengths, void* poolPtr, int32_t const* pageTable, int32_t maxPagesPerSeq, int32_t numPages,
    int32_t numKVHeads, int32_t poolHeadDim, int32_t headSize, int32_t maxBlocks)
{
    QsaIndexerPoolView const pool{poolPtr, pageTable, maxPagesPerSeq, numPages, numKVHeads, poolHeadDim, headSize};

    constexpr int32_t kColumnsPerCta = kScoresColumnsPerCta;
    constexpr int32_t kColumnsPerWarp = kScoresColumnsPerWarp;

    int32_t const batchIdx = static_cast<int32_t>(blockIdx.x);
    // The min() is out-of-bounds protection only; maxBlocks must cover ceilDiv(max ctx, 4).
    int32_t const numVisible = min(contextLengths[batchIdx] / kQSA_COMPRESS_RATIO, maxBlocks);
    int32_t const ctaColumnBase = static_cast<int32_t>(blockIdx.y) * kColumnsPerCta;
    if (ctaColumnBase >= numVisible)
    {
        return; // whole CTA past the visible prefix
    }

    int32_t const warp = static_cast<int32_t>(threadIdx.x) / kWarpSize;
    int32_t const lane = static_cast<int32_t>(threadIdx.x) % kWarpSize;

    __shared__ float sQ[kQNormedRowWidth];
    QsaType const* q = qNormed + static_cast<int64_t>(batchIdx) * kQNormedRowWidth;
    for (int32_t i = static_cast<int32_t>(threadIdx.x); i < kQNormedRowWidth; i += static_cast<int32_t>(blockDim.x))
    {
        sQ[i] = toFloat(q[i]);
    }
    __syncthreads();

    int32_t const columnBase = ctaColumnBase + warp * kColumnsPerWarp;
    if (columnBase >= numVisible)
    {
        return;
    }
    // The warp's 8 block-columns span tokens [4*columnBase, 4*columnBase + 32), which live
    // in ONE 128-token page: a single page-table lookup serves all 8 gathers.
    int32_t const vPage = qsaPoolPage(pool, batchIdx, /* plane = */ 1, columnBase * kQSA_COMPRESS_RATIO);
    if (vPage == kUNUSED_PAGE_ENTRY)
    {
        return;
    }
    QsaType const* const poolBase = static_cast<QsaType const*>(pool.poolPtr);
    float* outRow = logits + static_cast<int64_t>(batchIdx) * maxBlocks;

    for (int32_t it = 0; it < kColumnsPerWarp; ++it)
    {
        int32_t const blockId = columnBase + it;
        if (blockId >= numVisible)
        {
            break;
        }
        QsaType const* kb = qsaPoolTailPtr(pool, poolBase, vPage, blockId * kQSA_COMPRESS_RATIO);
        float partial[kQSA_INDEXER_NUM_HEADS] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
        for (int32_t i = 0; i < kDimsPerLane; ++i)
        {
            int32_t const d = lane + i * kWarpSize;
            float const kf = toFloat(kb[d]);
#pragma unroll
            for (int32_t h = 0; h < kQSA_INDEXER_NUM_HEADS; ++h)
            {
                partial[h] = fmaf(sQ[h * kQSA_INDEXER_HEAD_DIM + d], kf, partial[h]);
            }
        }
#pragma unroll
        for (int32_t h = 0; h < kQSA_INDEXER_NUM_HEADS; ++h)
        {
            partial[h] = warpReduceSum(partial[h]);
        }
        // Relu PER HEAD, sum over heads (ascending h), scale AFTER the sum.
        float sum = fmaxf(0.0f, partial[0]);
        sum += fmaxf(0.0f, partial[1]);
        sum += fmaxf(0.0f, partial[2]);
        sum += fmaxf(0.0f, partial[3]);
        if (lane == 0)
        {
            outRow[blockId] = sum * (1.0f / sqrtf(static_cast<float>(kQSA_INDEXER_HEAD_DIM)));
        }
    }
}

//! B3: see launchQsaTopKExpandDecode. grid(B), block(512).
extern "C" __global__ void qsa_indexer_topk_expand_decode(int32_t* outIdx, float const* logits,
    int32_t const* contextLengths, int32_t* splitCounters, int32_t numKVHeads, int32_t maxBlocks)
{
    int32_t const batchIdx = static_cast<int32_t>(blockIdx.x);
    int32_t const tid = static_cast<int32_t>(threadIdx.x);

    // (a) Zero the decode attention's split-K merge counters. Workspace memory is not
    // persistent across enqueues, so this in-enqueue zeroing is load-bearing (CUDA graphs).
    if (splitCounters != nullptr && tid < numKVHeads)
    {
        splitCounters[batchIdx * numKVHeads + tid] = 0;
    }

    int32_t const contextLen = contextLengths[batchIdx];
    int32_t const numVisible = contextLen / kQSA_COMPRESS_RATIO;
    int32_t const tailCount = contextLen - numVisible * kQSA_COMPRESS_RATIO;
    int32_t* out = outIdx + static_cast<int64_t>(batchIdx) * kQSA_INDEX_WIDTH;

    // (b) Fast path: every complete block is selected (ascending), so together with the
    // causal tail the output is exactly the prefix [0, contextLen). Logits are never read.
    if (numVisible <= kQSA_BLOCK_TOPK)
    {
        for (int32_t j = tid; j < kQSA_INDEX_WIDTH; j += kTopKThreads)
        {
            out[j] = j < contextLen ? j : -1;
        }
        return;
    }

    float const* rowLogits = logits + static_cast<int64_t>(batchIdx) * maxBlocks;
    // The min() is out-of-bounds protection only; maxBlocks must cover ceilDiv(max ctx, 4).
    int32_t const scanCount = min(numVisible, maxBlocks);

    // (c) MSB-first 8-bit radix select of the top kQSA_BLOCK_TOPK orderable keys.
    __shared__ uint32_t sHist[kRadixBins];
    __shared__ uint32_t sPrefix;    // resolved key bits above the current digit
    __shared__ int32_t sRemaining;  // quota left among keys matching sPrefix
    __shared__ uint32_t sThreshold; // final threshold key U
    __shared__ int32_t sTieQuota;   // u == U keys to take, ascending block id
    __shared__ int32_t sResolved;

    if (tid == 0)
    {
        sPrefix = 0U;
        sRemaining = kQSA_BLOCK_TOPK;
        sResolved = 0;
    }
    __syncthreads();

    bool const vecAligned = (reinterpret_cast<uintptr_t>(rowLogits) & 0xFU) == 0U;
    for (int32_t shift = 24; shift >= 0; shift -= 8)
    {
        if (sResolved != 0)
        {
            break; // uniform: sResolved was published before the loop-tail __syncthreads
        }
        for (int32_t i = tid; i < kRadixBins; i += kTopKThreads)
        {
            sHist[i] = 0U;
        }
        __syncthreads();

        uint32_t const prefix = sPrefix;
        bool const hasFilter = shift != 24;
        auto histogramOne = [&](float v) {
            uint32_t const u = logitOrderKey(v);
            if (!hasFilter || (u >> (shift + 8)) == prefix)
            {
                atomicAdd(&sHist[(u >> shift) & 0xFFU], 1U);
            }
        };
        if (vecAligned)
        {
            int32_t const numVec = scanCount / 4;
            float4 const* rowVec = reinterpret_cast<float4 const*>(rowLogits);
            for (int32_t i = tid; i < numVec; i += kTopKThreads)
            {
                float4 const v = rowVec[i];
                histogramOne(v.x);
                histogramOne(v.y);
                histogramOne(v.z);
                histogramOne(v.w);
            }
            for (int32_t g = numVec * 4 + tid; g < scanCount; g += kTopKThreads)
            {
                histogramOne(rowLogits[g]);
            }
        }
        else
        {
            for (int32_t g = tid; g < scanCount; g += kTopKThreads)
            {
                histogramOne(rowLogits[g]);
            }
        }
        __syncthreads();

        if (tid == 0)
        {
            // Descending cumulative pick of the digit holding the k-th largest key.
            int32_t const remaining = sRemaining;
            int32_t cumAbove = 0;
            int32_t digit = kRadixBins - 1;
            for (; digit > 0; --digit)
            {
                int32_t const binCount = static_cast<int32_t>(sHist[digit]);
                if (cumAbove + binCount >= remaining)
                {
                    break;
                }
                cumAbove += binCount;
            }
            int32_t const need = remaining - cumAbove; // in [1, sHist[digit]]
            uint32_t const resolvedPrefix = (prefix << 8) | static_cast<uint32_t>(digit);
            if (need == static_cast<int32_t>(sHist[digit]))
            {
                // The bin exactly fills the quota: every key >= the bin's lower bound is
                // selected (exactly kQSA_BLOCK_TOPK of them), so a take-all tie quota works.
                sThreshold = resolvedPrefix << shift;
                sTieQuota = kQSA_BLOCK_TOPK;
                sResolved = 1;
            }
            else if (shift == 0)
            {
                sThreshold = resolvedPrefix;
                sTieQuota = need;
                sResolved = 1;
            }
            else
            {
                sPrefix = resolvedPrefix;
                sRemaining = need;
            }
        }
        __syncthreads();
    }

    // (d) Deterministic fixed-order emit: u > U always selected; u == U fills the remaining
    // quota in ASCENDING block-id order. This is exactly the stable descending
    // cub::DeviceSegmentedRadixSort tie behavior of the prefill path, so prefill and decode
    // select IDENTICAL block sets for identical logits.
    uint32_t const threshold = sThreshold;
    int32_t const tieQuota = sTieQuota;
    int32_t const warp = tid / kWarpSize;
    int32_t const lane = tid % kWarpSize;
    uint32_t const laneMaskLt = (1U << lane) - 1U;

    __shared__ int32_t sWarpTie[kTopKWarps];
    __shared__ int32_t sWarpSel[kTopKWarps];
    __shared__ int32_t sTieBase;
    __shared__ int32_t sSelBase;
    if (tid == 0)
    {
        sTieBase = 0;
        sSelBase = 0;
    }
    __syncthreads();

    for (int32_t base = 0; base < scanCount; base += kTopKThreads)
    {
        int32_t const g = base + tid;
        bool const inRange = g < scanCount;
        uint32_t const u = inRange ? logitOrderKey(rowLogits[g]) : 0U;
        bool const isTie = inRange && u == threshold;

        uint32_t const tieBallot = __ballot_sync(kFullWarpMask, isTie);
        if (lane == 0)
        {
            sWarpTie[warp] = __popc(tieBallot);
        }
        __syncthreads();
        int32_t tieRank = sTieBase + __popc(tieBallot & laneMaskLt);
        for (int32_t w = 0; w < warp; ++w)
        {
            tieRank += sWarpTie[w];
        }

        bool const selected = (inRange && u > threshold) || (isTie && tieRank < tieQuota);
        uint32_t const selBallot = __ballot_sync(kFullWarpMask, selected);
        if (lane == 0)
        {
            sWarpSel[warp] = __popc(selBallot);
        }
        __syncthreads();
        int32_t outPos = sSelBase + __popc(selBallot & laneMaskLt);
        for (int32_t w = 0; w < warp; ++w)
        {
            outPos += sWarpSel[w];
        }

        if (selected)
        {
#pragma unroll
            for (int32_t j = 0; j < kQSA_COMPRESS_RATIO; ++j)
            {
                out[outPos * kQSA_COMPRESS_RATIO + j] = g * kQSA_COMPRESS_RATIO + j;
            }
        }
        __syncthreads();
        if (tid == 0)
        {
            int32_t tieTotal = 0;
            int32_t selTotal = 0;
            for (int32_t w = 0; w < kTopKWarps; ++w)
            {
                tieTotal += sWarpTie[w];
                selTotal += sWarpSel[w];
            }
            sTieBase += tieTotal;
            sSelBase += selTotal;
        }
        __syncthreads();
    }

    // (e) The emit filled slots [0, 4*kQSA_BLOCK_TOPK) with exactly kQSA_BLOCK_TOPK blocks;
    // append the causal tail and -1 pad the rest.
    constexpr int32_t kExpandedCount = kQSA_BLOCK_TOPK * kQSA_COMPRESS_RATIO;
    for (int32_t j = kExpandedCount + tid; j < kQSA_INDEX_WIDTH; j += kTopKThreads)
    {
        out[j] = j < kExpandedCount + tailCount ? numVisible * kQSA_COMPRESS_RATIO + (j - kExpandedCount) : -1;
    }
}
