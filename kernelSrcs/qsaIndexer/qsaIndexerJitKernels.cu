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
//! Block rope row: the 64 FP32 cos/sin of a block's first token 4g, kept as raw bytes in that token's V-tail; kbar
//! of the block sits in the V-tail of token 4g + 1. The layout is in qsaIndexerKernels.h. One warp moves the row
//! with 2 floats per lane; the 128-element tail must hold it exactly.
constexpr int32_t kBlockRopeRowFloats = kQSA_INDEXER_ROTARY_DIM;                // 64
constexpr int32_t kBlockRopeRowFloatsPerLane = kBlockRopeRowFloats / kWarpSize; // 2
static_assert(kQSA_INDEXER_HEAD_DIM * sizeof(QsaType) == kBlockRopeRowFloats * sizeof(float),
    "the block rope row must fill the 128-element tail exactly");
static_assert(kBlockRopeRowFloats % kWarpSize == 0, "one warp moves a whole rope row");
constexpr int32_t kKbarTokenOffset = 1;
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

//! Warp-cooperative raw copy of one FP32 rope row (64 floats) into a V-tail viewed as FP32: the
//! block rope row. Byte-exact: the cell is never interpreted as QsaType. Alignment: the tail
//! starts at element offset (page * 128 + token) * rowStride + headSize; the host requires
//! rowStride and headSize to be even, so the byte offset is a multiple of 4 and scalar FP32
//! access is well-defined.
__device__ __forceinline__ void qsaWriteBlockRopeRow(QsaType* vTail, float const* cosSinRow, int32_t lane)
{
    float* blockRopeRow = reinterpret_cast<float*>(vTail);
#pragma unroll
    for (int32_t i = 0; i < kBlockRopeRowFloatsPerLane; ++i)
    {
        blockRopeRow[lane * kBlockRopeRowFloatsPerLane + i] = cosSinRow[lane * kBlockRopeRowFloatsPerLane + i];
    }
}

//! Inverse of qsaWriteBlockRopeRow: stage a block rope row into shared memory. The caller must
//! __syncwarp() before reading sCosSin, whose entries [d] and [32 + d] were written by other lanes.
__device__ __forceinline__ void qsaLoadBlockRopeRow(float* sCosSin, QsaType const* vTail, int32_t lane)
{
    float const* blockRopeRow = reinterpret_cast<float const*>(vTail);
#pragma unroll
    for (int32_t i = 0; i < kBlockRopeRowFloatsPerLane; ++i)
    {
        sCosSin[lane * kBlockRopeRowFloatsPerLane + i] = blockRopeRow[lane * kBlockRopeRowFloatsPerLane + i];
    }
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

//! The one block compress every path runs, so all of them agree bit for bit. One warp; out is
//! written in full. cosSinRow may point at shared memory the warp wrote before the call.
__device__ __forceinline__ void qsaCompressBlock(QsaType* out, QsaType const* const* rawKeys, float const* cosSinRow,
    QsaType const* wK, float rmsEps, float* sK, int32_t lane)
{
    float mean[kDimsPerLane];
    float sumSq = 0.0f;
#pragma unroll
    for (int32_t i = 0; i < kDimsPerLane; ++i)
    {
        int32_t const d = lane * kDimsPerLane + i;
        // FP32 mean in FIXED sequential order, then the bit-compatibility-critical cast to QsaType.
        float acc = toFloat(rawKeys[0][d]);
        acc = acc + toFloat(rawKeys[1][d]);
        acc = acc + toFloat(rawKeys[2][d]);
        acc = acc + toFloat(rawKeys[3][d]);
        mean[i] = toFloat(fromFloat<QsaType>(acc * 0.25f));
        sumSq = fmaf(mean[i], mean[i], sumSq);
    }
    sumSq = warpReduceSum(sumSq);
    float const invRms = rsqrtf(sumSq / static_cast<float>(kQSA_INDEXER_HEAD_DIM) + rmsEps);
#pragma unroll
    for (int32_t i = 0; i < kDimsPerLane; ++i)
    {
        int32_t const d = lane * kDimsPerLane + i;
        sK[d] = mean[i] * invRms * (1.0f + toFloat(wK[d]));
    }
    __syncwarp();
    // Rope at the block's first position; single cast to QsaType at the very end.
#pragma unroll
    for (int32_t i = 0; i < kDimsPerLane; ++i)
    {
        int32_t const d = lane * kDimsPerLane + i;
        out[d] = fromFloat<QsaType>(applyNeoxRopeFromSmem(sK, cosSinRow, d));
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

    // Partial neox rope with this token's token-aligned row (activation row b*S + t); single
    // cast to QsaType at the very end.
    float const* cosSinRow = cosSin + static_cast<int64_t>(row) * kQSA_INDEXER_ROTARY_DIM;
#pragma unroll
    for (int32_t i = 0; i < kDimsPerLane; ++i)
    {
        int32_t const d = lane * kDimsPerLane + i;
        out[d] = fromFloat<QsaType>(applyNeoxRopeFromSmem(sQ[head], cosSinRow, d));
    }
}

//! K1b: see launchQsaIndexKCompress. grid(B * numBlocks), block(32); one warp per (b, g).
extern "C" __global__ void qsa_indexer_k_compress(QsaType* kbar, QsaType const* indexQk, float const* cosSin,
    int32_t const* contextLengths, QsaType const* wK, float rmsEps, int32_t seqLen, int32_t numBlocks)
{
    int32_t const batchIdx = static_cast<int32_t>(blockIdx.x) / numBlocks;
    int32_t const blockId = static_cast<int32_t>(blockIdx.x) % numBlocks;
    int32_t const lane = static_cast<int32_t>(threadIdx.x);

    QsaType* out = kbar + (static_cast<int64_t>(batchIdx) * numBlocks + blockId) * kQSA_INDEXER_HEAD_DIM;

    int32_t const firstToken = blockId * kQSA_COMPRESS_RATIO;
    bool const valid = firstToken + kQSA_COMPRESS_RATIO - 1 < contextLengths[batchIdx];
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
        + static_cast<int64_t>(firstToken) * kIndexQkRowWidth + kRawKColumnOffset;
    QsaType const* const rawKeys[kQSA_COMPRESS_RATIO]
        = {k0, k0 + kIndexQkRowWidth, k0 + 2 * kIndexQkRowWidth, k0 + 3 * kIndexQkRowWidth};
    // Rope at the block's first token, from that token's token-aligned row (the row k0 came from).
    float const* cosSinRow = cosSin + (static_cast<int64_t>(batchIdx) * seqLen + firstToken) * kQSA_INDEXER_ROTARY_DIM;
    qsaCompressBlock(out, rawKeys, cosSinRow, wK, rmsEps, sK, lane);
}

//! K2: see launchQsaIndexScores. grid(numRows, ceilDiv(numBlocks, 64)), block(256).
//! The CTA stages the row's 4x128 q in FP32 smem; each of the 8 warps owns 8 consecutive
//! block-columns of the 64-column tile and evaluates them one at a time.
template <bool kPaged>
__device__ __forceinline__ void qsaIndexerScoresImpl(float* logits, int32_t* visibleCounts, QsaType const* qNormed,
    QsaType const* kbar, int32_t const* contextLengths, int32_t const* pastLengths, int32_t seqLen, int32_t numBlocks,
    int32_t rowStart, QsaIndexerPoolView const& pool)
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
    int32_t const pastLen = kPaged && pastLengths != nullptr ? pastLengths[batchIdx] : 0;
    int32_t const numVisible = isPadding ? 0 : (pastLen + tokenIdx + 1) / kQSA_COMPRESS_RATIO;

    if (blockIdx.y == 0 && threadIdx.x == 0)
    {
        visibleCounts[chunkRow] = numVisible;
    }
    int32_t const ctaColumnBase = static_cast<int32_t>(blockIdx.y) * kColumnsPerCta;
    if (ctaColumnBase >= numVisible)
    {
        return;
    }

    float* outRow = logits + static_cast<int64_t>(chunkRow) * numBlocks;

    __shared__ float sQ[kQNormedRowWidth];
    QsaType const* q = qNormed + static_cast<int64_t>(row) * kQNormedRowWidth;
    for (int32_t i = static_cast<int32_t>(threadIdx.x); i < kQNormedRowWidth; i += static_cast<int32_t>(blockDim.x))
    {
        sQ[i] = toFloat(q[i]);
    }
    __syncthreads();

    for (int32_t it = 0; it < kColumnsPerWarp; ++it)
    {
        int32_t const blockId = columnBase + it;
        if (blockId >= numBlocks || blockId >= numVisible)
        {
            break;
        }

        float score = -kFloatMax;
        QsaType const* kb = nullptr;
        if constexpr (kPaged)
        {
            int32_t const firstToken = blockId * kQSA_COMPRESS_RATIO;
            int32_t const page = qsaPoolPage(pool, batchIdx, /* plane = */ 1, firstToken);
            if (page == kUNUSED_PAGE_ENTRY)
            {
                if (lane == 0)
                {
                    outRow[blockId] = score;
                }
                continue;
            }
            kb = qsaPoolTailPtr(pool, static_cast<QsaType const*>(pool.poolPtr), page, firstToken + kKbarTokenOffset);
        }
        else
        {
            kb = kbar + (static_cast<int64_t>(batchIdx) * numBlocks + blockId) * kQSA_INDEXER_HEAD_DIM;
        }
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
        if (lane == 0)
        {
            outRow[blockId] = score;
        }
    }
}

extern "C" __global__ void qsa_indexer_scores(float* logits, int32_t* visibleCounts, QsaType const* qNormed,
    QsaType const* kbar, int32_t const* contextLengths, int32_t seqLen, int32_t numBlocks, int32_t rowStart)
{
    qsaIndexerScoresImpl<false>(
        logits, visibleCounts, qNormed, kbar, contextLengths, nullptr, seqLen, numBlocks, rowStart, {});
}

extern "C" __global__ void qsa_indexer_scores_paged(float* logits, int32_t* visibleCounts, QsaType const* qNormed,
    int32_t const* queryLengths, int32_t const* pastLengths, int32_t seqLen, int32_t maxBlocks, int32_t rowStart,
    void* poolPtr, int32_t const* pageTable, int32_t maxPagesPerSeq, int32_t numPages, int32_t numKVHeads,
    int32_t poolHeadDim, int32_t headSize)
{
    QsaIndexerPoolView const pool{poolPtr, pageTable, maxPagesPerSeq, numPages, numKVHeads, poolHeadDim, headSize};
    qsaIndexerScoresImpl<true>(
        logits, visibleCounts, qNormed, nullptr, queryLengths, pastLengths, seqLen, maxBlocks, rowStart, pool);
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
    int32_t seqLen, int32_t numBlocks, int32_t rowStart, int32_t const* pastLengths)
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
        int32_t const totalLen = (pastLengths == nullptr ? 0 : pastLengths[batchIdx]) + tokenIdx + 1;
        numVisible = totalLen / kQSA_COMPRESS_RATIO;
        numSelected = min(kQSA_BLOCK_TOPK, numVisible);
        tailCount = totalLen % kQSA_COMPRESS_RATIO;
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

//! Prefill tail write: see launchQsaRawKTailWritePrefill. grid(B * S), block(32): one warp per
//! query row writes that token's raw index-K and, for a block's first token, the block rope row.
extern "C" __global__ void qsa_indexer_raw_k_tail_write_prefill(QsaType const* indexQk, float const* cosSin,
    int32_t const* contextLengths, void* poolPtr, int32_t const* pageTable, int32_t maxPagesPerSeq, int32_t numPages,
    int32_t numKVHeads, int32_t poolHeadDim, int32_t headSize, int32_t seqLen, int32_t const* pastLengths)
{
    QsaIndexerPoolView const pool{poolPtr, pageTable, maxPagesPerSeq, numPages, numKVHeads, poolHeadDim, headSize};

    int32_t const row = static_cast<int32_t>(blockIdx.x); // b * seqLen + t
    int32_t const batchIdx = row / seqLen;
    int32_t const lane = static_cast<int32_t>(threadIdx.x);
    if (row % seqLen >= contextLengths[batchIdx])
    {
        return;
    }
    int32_t const tokenIdx = (pastLengths == nullptr ? 0 : pastLengths[batchIdx]) + row % seqLen;
    int32_t const kPage = qsaPoolPage(pool, batchIdx, /* plane = */ 0, tokenIdx);
    if (kPage == kUNUSED_PAGE_ENTRY)
    {
        return;
    }
    QsaType const* src = indexQk + static_cast<int64_t>(row) * kIndexQkRowWidth + kRawKColumnOffset;
    QsaType* dst = qsaPoolTailPtr(pool, static_cast<QsaType*>(pool.poolPtr), kPage, tokenIdx);
#pragma unroll
    for (int32_t i = 0; i < kDimsPerLane; ++i)
    {
        dst[lane * kDimsPerLane + i] = src[lane * kDimsPerLane + i]; // bit-unmodified
    }
    if (tokenIdx % kQSA_COMPRESS_RATIO != 0)
    {
        return;
    }
    int32_t const vPage = qsaPoolPage(pool, batchIdx, /* plane = */ 1, tokenIdx);
    if (vPage == kUNUSED_PAGE_ENTRY)
    {
        return;
    }
    qsaWriteBlockRopeRow(qsaPoolTailPtr(pool, static_cast<QsaType*>(pool.poolPtr), vPage, tokenIdx),
        cosSin + static_cast<int64_t>(row) * kQSA_INDEXER_ROTARY_DIM, lane);
}

//! One warp per newly completed group; a group crossing the chunk boundary reads the cached raw
//! keys and block rope row from the pool.
extern "C" __global__ void qsa_indexer_k_compress_append(QsaType const* indexQk, float const* cosSin,
    int32_t const* queryLengths, int32_t const* pastLengths, QsaType const* wK, float rmsEps, int32_t seqLen,
    void* poolPtr, int32_t const* pageTable, int32_t maxPagesPerSeq, int32_t numPages, int32_t numKVHeads,
    int32_t poolHeadDim, int32_t headSize)
{
    int32_t const batchIdx = static_cast<int32_t>(blockIdx.y);
    int32_t const lane = static_cast<int32_t>(threadIdx.x);
    int32_t const pastLen = pastLengths == nullptr ? 0 : pastLengths[batchIdx];
    int32_t const queryLen = queryLengths[batchIdx];
    int32_t const blockId = pastLen / kQSA_COMPRESS_RATIO + static_cast<int32_t>(blockIdx.x);
    int32_t const firstToken = blockId * kQSA_COMPRESS_RATIO;
    if (queryLen <= 0 || firstToken + kQSA_COMPRESS_RATIO > pastLen + queryLen)
    {
        return;
    }

    QsaIndexerPoolView const pool{poolPtr, pageTable, maxPagesPerSeq, numPages, numKVHeads, poolHeadDim, headSize};
    int32_t const vPage = qsaPoolPage(pool, batchIdx, /* plane = */ 1, firstToken);
    if (vPage == kUNUSED_PAGE_ENTRY)
    {
        return;
    }
    QsaType const* rawKeys[kQSA_COMPRESS_RATIO];
#pragma unroll
    for (int32_t j = 0; j < kQSA_COMPRESS_RATIO; ++j)
    {
        int32_t const token = firstToken + j;
        if (token < pastLen)
        {
            int32_t const kPage = qsaPoolPage(pool, batchIdx, /* plane = */ 0, token);
            if (kPage == kUNUSED_PAGE_ENTRY)
            {
                return;
            }
            rawKeys[j] = qsaPoolTailPtr(pool, static_cast<QsaType const*>(poolPtr), kPage, token);
        }
        else
        {
            rawKeys[j] = indexQk + (static_cast<int64_t>(batchIdx) * seqLen + token - pastLen) * kIndexQkRowWidth
                + kRawKColumnOffset;
        }
    }

    __shared__ float sK[kQSA_INDEXER_HEAD_DIM];
    __shared__ float sCosSin[kBlockRopeRowFloats];
    float const* cosSinRow = nullptr;
    if (firstToken < pastLen)
    {
        qsaLoadBlockRopeRow(sCosSin, qsaPoolTailPtr(pool, static_cast<QsaType const*>(poolPtr), vPage, firstToken), lane);
        cosSinRow = sCosSin;
    }
    else
    {
        cosSinRow = cosSin
            + (static_cast<int64_t>(batchIdx) * seqLen + firstToken - pastLen) * kQSA_INDEXER_ROTARY_DIM;
    }
    qsaCompressBlock(qsaPoolTailPtr(pool, static_cast<QsaType*>(poolPtr), vPage, firstToken + kKbarTokenOffset),
        rawKeys, cosSinRow, wK, rmsEps, sK, lane);
}

//! B1: see launchQsaIndexerPreDecode. grid(B), block(128) = 4 warps; warp h owns q head h
//! in phase 1, warp 0 alone runs phase 2 (tail write) and phase 3 (compress).
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
    __shared__ float sCosSin[kBlockRopeRowFloats];

    // Phase 1: K1a numerics verbatim at S = 1. Decode is one row per sequence, so the new
    // token's token-aligned rope row is row batchIdx (position pos).
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
    float const* cosSinRow = cosSin + static_cast<int64_t>(batchIdx) * kQSA_INDEXER_ROTARY_DIM;
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

    // Phase 2 (warp 0): the new token's raw index-K, bit-unmodified, and the block rope row
    // when it opens a block.
    QsaType* const poolBase = static_cast<QsaType*>(pool.poolPtr);
    int32_t const kPage = qsaPoolPage(pool, batchIdx, /* plane = */ 0, pos);
    if (kPage == kUNUSED_PAGE_ENTRY)
    {
        return; // unallocated page: nothing persistable this step
    }
    QsaType const* rawIn = indexQk + static_cast<int64_t>(batchIdx) * kIndexQkRowWidth + kRawKColumnOffset;
    QsaType* const kTail = qsaPoolTailPtr(pool, poolBase, kPage, pos);
#pragma unroll
    for (int32_t i = 0; i < kDimsPerLane; ++i)
    {
        kTail[lane * kDimsPerLane + i] = rawIn[lane * kDimsPerLane + i];
    }
    if (pos % kQSA_COMPRESS_RATIO == 0)
    {
        int32_t const vPage = qsaPoolPage(pool, batchIdx, /* plane = */ 1, pos);
        if (vPage != kUNUSED_PAGE_ENTRY)
        {
            qsaWriteBlockRopeRow(qsaPoolTailPtr(pool, poolBase, vPage, pos), cosSinRow, lane);
        }
    }
    if (contextLen % kQSA_COMPRESS_RATIO != 0)
    {
        return;
    }

    // Phase 3 (warp 0): this token completes block 4g..4g+3. Tokens 4g..4g+2 share pos's K
    // page (a 4-token block never straddles a 128-token page); token 4g+3 is read from indexQk.
    int32_t const firstToken = contextLen - kQSA_COMPRESS_RATIO;
    int32_t const vPage = qsaPoolPage(pool, batchIdx, /* plane = */ 1, firstToken);
    if (vPage == kUNUSED_PAGE_ENTRY)
    {
        return;
    }
    QsaType const* const rawKeys[kQSA_COMPRESS_RATIO] = {qsaPoolTailPtr(pool, poolBase, kPage, firstToken),
        qsaPoolTailPtr(pool, poolBase, kPage, firstToken + 1), qsaPoolTailPtr(pool, poolBase, kPage, firstToken + 2),
        rawIn};
    qsaLoadBlockRopeRow(sCosSin, qsaPoolTailPtr(pool, poolBase, vPage, firstToken), lane);
    qsaCompressBlock(qsaPoolTailPtr(pool, poolBase, vPage, firstToken + kKbarTokenOffset), rawKeys, sCosSin, wK,
        rmsEps, sK, lane);
}

//! Seal: see launchQsaIndexerSeal. grid(kQSA_SEAL_BLOCKS, B), block(32): CTA (i, b) recompresses
//! block lengths[b] / 4 - 1 - i from the pool tails.
extern "C" __global__ void qsa_indexer_seal(int32_t const* lengths, QsaType const* wK, float rmsEps, void* poolPtr,
    int32_t const* pageTable, int32_t maxPagesPerSeq, int32_t numPages, int32_t numKVHeads, int32_t poolHeadDim,
    int32_t headSize)
{
    int32_t const batchIdx = static_cast<int32_t>(blockIdx.y);
    int32_t const lane = static_cast<int32_t>(threadIdx.x);
    int32_t const blockId = lengths[batchIdx] / kQSA_COMPRESS_RATIO - 1 - static_cast<int32_t>(blockIdx.x);
    if (blockId < 0)
    {
        return;
    }
    QsaIndexerPoolView const pool{poolPtr, pageTable, maxPagesPerSeq, numPages, numKVHeads, poolHeadDim, headSize};
    int32_t const firstToken = blockId * kQSA_COMPRESS_RATIO;
    // A 4-token block never straddles a 128-token page.
    int32_t const kPage = qsaPoolPage(pool, batchIdx, /* plane = */ 0, firstToken);
    int32_t const vPage = qsaPoolPage(pool, batchIdx, /* plane = */ 1, firstToken);
    if (kPage == kUNUSED_PAGE_ENTRY || vPage == kUNUSED_PAGE_ENTRY)
    {
        return;
    }
    QsaType* const poolBase = static_cast<QsaType*>(poolPtr);
    QsaType const* rawKeys[kQSA_COMPRESS_RATIO];
#pragma unroll
    for (int32_t j = 0; j < kQSA_COMPRESS_RATIO; ++j)
    {
        rawKeys[j] = qsaPoolTailPtr(pool, poolBase, kPage, firstToken + j);
    }
    __shared__ float sK[kQSA_INDEXER_HEAD_DIM];
    __shared__ float sCosSin[kBlockRopeRowFloats];
    qsaLoadBlockRopeRow(sCosSin, qsaPoolTailPtr(pool, poolBase, vPage, firstToken), lane);
    qsaCompressBlock(qsaPoolTailPtr(pool, poolBase, vPage, firstToken + kKbarTokenOffset), rawKeys, sCosSin, wK,
        rmsEps, sK, lane);
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
        QsaType const* kb = qsaPoolTailPtr(pool, poolBase, vPage, blockId * kQSA_COMPRESS_RATIO + kKbarTokenOffset);
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

//! One warp per query. Only lane zero walks the bounded parent chain.
extern "C" __global__ void qsa_indexer_tree_paths(int32_t* paths, int32_t* logicalLengths, int32_t const* parents, int32_t const* depths, int32_t const* queryLengths, int32_t const* pastLengths,
    int32_t const* pageTable, int32_t seqLen, int32_t maxBlocks, int32_t maxPagesPerSeq, int32_t numPages)
{
    int32_t const row = static_cast<int32_t>(blockIdx.x);
    int32_t const b = row / seqLen;
    int32_t const node = row % seqLen;
    int32_t* const path = paths + static_cast<int64_t>(row) * seqLen;
    for (int32_t i = static_cast<int32_t>(threadIdx.x); i < seqLen; i += kWarpSize)
    {
        path[i] = -1;
    }
    __syncwarp();
    if (threadIdx.x != 0)
    {
        return;
    }
    // Padding and invalid rows keep logicalLengths zero, so they select no blocks.
    logicalLengths[row] = 0;
    int32_t const count = queryLengths[b];
    int32_t const past = pastLengths[b];
    if (count < 0 || count > seqLen || past < 0 || node >= count)
    {
        return;
    }
    int32_t const depth = depths[row];
    int64_t const length = static_cast<int64_t>(past) + depth + 1;
    int64_t const capacity = static_cast<int64_t>(maxPagesPerSeq) * kTOKENS_PER_PAGE;
    if (depth < 0 || depth >= seqLen || length > static_cast<int64_t>(maxBlocks) * kQSA_COMPRESS_RATIO
        || static_cast<int64_t>(past) + count > capacity)
    {
        return;
    }
    int32_t current = node;
    for (int32_t d = depth; d >= 0; --d)
    {
        if (current < 0 || current >= count || depths[b * seqLen + current] != d)
        {
            return;
        }
        path[d] = current;
        int32_t const parent = parents[b * seqLen + current];
        if ((d == 0 && (current != 0 || parent != -1)) || (d > 0 && (parent < 0 || parent >= current)))
        {
            return;
        }
        current = parent;
    }
    // Validate K/V pages covering [0, past + count): KV slots follow node IDs, not branch depths.
    int32_t const pages
        = static_cast<int32_t>((static_cast<int64_t>(past) + count + kTOKENS_PER_PAGE - 1) / kTOKENS_PER_PAGE);
    for (int32_t slot = 0; slot < pages; ++slot)
    {
        int64_t const offset = static_cast<int64_t>(b) * 2 * maxPagesPerSeq + slot;
        int32_t const kPage = pageTable[offset];
        int32_t const vPage = pageTable[offset + maxPagesPerSeq];
        if (kPage < 0 || kPage >= numPages || vPage < numPages || static_cast<int64_t>(vPage) >= 2LL * numPages)
        {
            return;
        }
    }
    logicalLengths[row] = static_cast<int32_t>(length);
}

//! One warp per node. Writes the node's tails into its own slot past + node (the slot its K/V use): raw index-K, and
//! the block rope row when its position past + depth opens a block. A completed block belongs to its final node and
//! is shared by that node's descendants.
extern "C" __global__ void qsa_indexer_tree_k_compress(QsaType* newKbar, QsaType const* indexQk, float const* cosSin,
    int32_t const* paths, int32_t const* logicalLengths, int32_t const* pastLengths, QsaType const* wK, float rmsEps,
    int32_t seqLen, void* poolPtr, int32_t const* pageTable, int32_t maxPagesPerSeq, int32_t numPages,
    int32_t numKVHeads, int32_t poolHeadDim, int32_t headSize)
{
    int32_t const row = static_cast<int32_t>(blockIdx.x);
    int32_t const batchIdx = row / seqLen;
    int32_t const lane = static_cast<int32_t>(threadIdx.x);
    int32_t const length = logicalLengths[row];
    QsaType* const out = newKbar + static_cast<int64_t>(row) * kQSA_INDEXER_HEAD_DIM;
    for (int32_t d = lane; d < kQSA_INDEXER_HEAD_DIM; d += kWarpSize)
    {
        out[d] = fromFloat<QsaType>(0.0f);
    }
    if (length == 0)
    {
        return;
    }
    int32_t const pastLen = pastLengths[batchIdx];
    QsaIndexerPoolView const pool{poolPtr, pageTable, maxPagesPerSeq, numPages, numKVHeads, poolHeadDim, headSize};
    QsaType* const poolBase = static_cast<QsaType*>(poolPtr);
    int32_t const slot = pastLen + row % seqLen;
    QsaType* const kTail = qsaPoolTailPtr(pool, poolBase, qsaPoolPage(pool, batchIdx, 0, slot), slot);
    for (int32_t d = lane; d < kQSA_INDEXER_HEAD_DIM; d += kWarpSize)
    {
        kTail[d] = indexQk[static_cast<int64_t>(row) * kIndexQkRowWidth + kRawKColumnOffset + d];
    }
    if ((length - 1) % kQSA_COMPRESS_RATIO == 0)
    {
        qsaWriteBlockRopeRow(qsaPoolTailPtr(pool, poolBase, qsaPoolPage(pool, batchIdx, 1, slot), slot),
            cosSin + static_cast<int64_t>(row) * kQSA_INDEXER_ROTARY_DIM, lane);
    }
    if (length % kQSA_COMPRESS_RATIO != 0)
    {
        return;
    }
    int32_t const firstToken = length - kQSA_COMPRESS_RATIO;
    int32_t const* const path = paths + static_cast<int64_t>(row) * seqLen;
    QsaType const* rawKeys[kQSA_COMPRESS_RATIO];
#pragma unroll
    for (int32_t j = 0; j < kQSA_COMPRESS_RATIO; ++j)
    {
        int32_t const token = firstToken + j;
        if (token < pastLen)
        {
            int32_t const page = qsaPoolPage(pool, batchIdx, 0, token);
            rawKeys[j] = qsaPoolTailPtr(pool, static_cast<QsaType const*>(poolPtr), page, token);
        }
        else
        {
            rawKeys[j] = indexQk + (static_cast<int64_t>(batchIdx) * seqLen + path[token - pastLen]) * kIndexQkRowWidth
                + kRawKColumnOffset;
        }
    }
    __shared__ float sK[kQSA_INDEXER_HEAD_DIM];
    __shared__ float sCosSin[kBlockRopeRowFloats];
    float const* cosSinRow = nullptr;
    if (firstToken < pastLen)
    {
        int32_t const page = qsaPoolPage(pool, batchIdx, 1, firstToken);
        qsaLoadBlockRopeRow(
            sCosSin, qsaPoolTailPtr(pool, static_cast<QsaType const*>(poolPtr), page, firstToken), lane);
        cosSinRow = sCosSin;
    }
    else
    {
        cosSinRow
            = cosSin + (static_cast<int64_t>(batchIdx) * seqLen + path[firstToken - pastLen]) * kQSA_INDEXER_ROTARY_DIM;
    }
    qsaCompressBlock(out, rawKeys, cosSinRow, wK, rmsEps, sK, lane);
}

//! Same reduction order as qsaIndexerScoresImpl; only the compressed-key address depends on the tree.
extern "C" __global__ void qsa_indexer_tree_scores(float* logits, QsaType const* qNormed, QsaType const* newKbar,
    int32_t const* paths, int32_t const* logicalLengths, int32_t const* pastLengths, int32_t seqLen, int32_t maxBlocks,
    void* poolPtr, int32_t const* pageTable, int32_t maxPagesPerSeq, int32_t numPages, int32_t numKVHeads,
    int32_t poolHeadDim, int32_t headSize)
{
    int32_t const row = static_cast<int32_t>(blockIdx.x);
    int32_t const batchIdx = row / seqLen;
    int32_t const warp = static_cast<int32_t>(threadIdx.x) / kWarpSize;
    int32_t const lane = static_cast<int32_t>(threadIdx.x) % kWarpSize;
    int32_t const numVisible = logicalLengths[row] / kQSA_COMPRESS_RATIO;
    int32_t const columnBase = static_cast<int32_t>(blockIdx.y) * kScoresColumnsPerCta + warp * kScoresColumnsPerWarp;
    if (static_cast<int32_t>(blockIdx.y) * kScoresColumnsPerCta >= numVisible)
    {
        return;
    }
    __shared__ float sQ[kQNormedRowWidth];
    QsaType const* const q = qNormed + static_cast<int64_t>(row) * kQNormedRowWidth;
    for (int32_t i = static_cast<int32_t>(threadIdx.x); i < kQNormedRowWidth; i += static_cast<int32_t>(blockDim.x))
    {
        sQ[i] = toFloat(q[i]);
    }
    __syncthreads();
    int32_t const past = pastLengths[batchIdx];
    int32_t const* const path = paths + static_cast<int64_t>(row) * seqLen;
    float* const outRow = logits + static_cast<int64_t>(row) * maxBlocks;
    QsaIndexerPoolView const pool{poolPtr, pageTable, maxPagesPerSeq, numPages, numKVHeads, poolHeadDim, headSize};
    for (int32_t it = 0; it < kScoresColumnsPerWarp; ++it)
    {
        int32_t const blockId = columnBase + it;
        if (blockId >= numVisible)
        {
            break;
        }
        QsaType const* kb = nullptr;
        if (blockId < past / kQSA_COMPRESS_RATIO)
        {
            int32_t const firstToken = blockId * kQSA_COMPRESS_RATIO;
            int32_t const page = qsaPoolPage(pool, batchIdx, 1, firstToken);
            kb = qsaPoolTailPtr(pool, static_cast<QsaType const*>(poolPtr), page, firstToken + kKbarTokenOffset);
        }
        else
        {
            int32_t const endDepth = (blockId + 1) * kQSA_COMPRESS_RATIO - 1 - past;
            kb = newKbar + (static_cast<int64_t>(batchIdx) * seqLen + path[endDepth]) * kQSA_INDEXER_HEAD_DIM;
        }
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
        float const score = sum * (1.0f / sqrtf(static_cast<float>(kQSA_INDEXER_HEAD_DIM)));
        if (lane == 0)
        {
            outRow[blockId] = score;
        }
    }
}

extern "C" __global__ void qsa_indexer_tree_map_indices(
    int32_t* outIdx, int32_t const* paths, int32_t const* logicalLengths, int32_t const* pastLengths, int32_t seqLen)
{
    int32_t const row = static_cast<int32_t>(blockIdx.x);
    int32_t const past = pastLengths[row / seqLen];
    int32_t const length = logicalLengths[row];
    int32_t const* const path = paths + static_cast<int64_t>(row) * seqLen;
    for (int32_t i = static_cast<int32_t>(threadIdx.x); i < kQSA_INDEX_WIDTH; i += static_cast<int32_t>(blockDim.x))
    {
        int64_t const offset = static_cast<int64_t>(row) * kQSA_INDEX_WIDTH + i;
        int32_t const logical = outIdx[offset];
        outIdx[offset] = length == 0 || logical < 0 ? -1 : (logical < past ? logical : past + path[logical - past]);
    }
}
