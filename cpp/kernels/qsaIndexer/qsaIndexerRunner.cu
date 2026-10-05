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
#include "qsaIndexerKernels.h"
#include "qsaIndexerRunner.h"

#include <algorithm>
#include <cstdint>
#include <cub/cub.cuh>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
// The cub:: iterator facades were removed from CCCL 3.0 (CUDA 13); the Thrust iterators are
// the stable spelling on both CUDA 12 and 13.
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

namespace trt_edgellm
{
namespace kernel
{

namespace
{

//! Device workspace slot alignment; matches plugins::kDEVICE_ALIGNMENT without depending on
//! the plugin layer from a kernel-layer translation unit.
constexpr size_t kWorkspaceAlignment = 128;

//! Sort keys buffer cap: 32 MiB of FP32 keys = 8388608 elements. Bounds the segmented-sort
//! working set (keys + values + alternates ~ 128 MiB) independently of batchSize * seqLen.
constexpr int64_t kMaxSortKeyElements = 8388608;

//! Row-chunk floor so tiny numBlocks still batch enough segments per sort call.
constexpr int64_t kMinRowChunk = 256;

constexpr int32_t ceilDiv(int32_t a, int32_t b)
{
    return (a + b - 1) / b;
}

size_t alignUp(size_t size)
{
    return (size + kWorkspaceAlignment - 1) / kWorkspaceAlignment * kWorkspaceAlignment;
}

//! Carve `bytes` from the front of `workspace` (advancing it by the aligned size).
template <typename T>
T* assignFromWorkspace(std::byte*& workspace, size_t bytes)
{
    T* const ptr = reinterpret_cast<T*>(workspace);
    workspace += alignUp(bytes);
    return ptr;
}

//! Rows per sort chunk: Rc = min(B * S, max(kMaxSortKeyElements / MB, kMinRowChunk)).
int32_t getRowChunk(int32_t batchSize, int32_t seqLen, int32_t numBlocks)
{
    int64_t const totalRows = static_cast<int64_t>(batchSize) * seqLen;
    int64_t const cappedRows = std::max(kMaxSortKeyElements / numBlocks, kMinRowChunk);
    return static_cast<int32_t>(std::min(totalRows, cappedRows));
}

//! Fixed-width segment offsets for cub: segment i begins at i * numBlocks.
struct SegmentOffsetOp
{
    int32_t numBlocks;

    __host__ __device__ __forceinline__ int32_t operator()(int32_t i) const
    {
        return i * numBlocks;
    }
};

using SegmentOffsetIterator = thrust::transform_iterator<SegmentOffsetOp, thrust::counting_iterator<int32_t>>;

//! cub temp storage bytes for one descending segmented sort of numRows segments x numBlocks keys.
size_t getCubSortTempBytes(int32_t numRows, int32_t numBlocks)
{
    size_t tempBytes = 0;
    cub::DoubleBuffer<float> keys(nullptr, nullptr);
    cub::DoubleBuffer<int32_t> values(nullptr, nullptr);
    SegmentOffsetIterator const beginOffsets(thrust::counting_iterator<int32_t>(0), SegmentOffsetOp{numBlocks});
    SegmentOffsetIterator const endOffsets(thrust::counting_iterator<int32_t>(1), SegmentOffsetOp{numBlocks});
    CUDA_CHECK(cub::DeviceSegmentedRadixSort::SortPairsDescending(nullptr, tempBytes, keys, values, numRows * numBlocks,
        numRows, beginOffsets, endOffsets, 0, static_cast<int>(sizeof(float) * 8)));
    return tempBytes;
}

} // namespace

size_t getQsaIndexerWorkspaceSize(int32_t batchSize, int32_t maxSeqLen)
{
    ELLM_CHECK(batchSize > 0 && maxSeqLen > 0, "getQsaIndexerWorkspaceSize: batchSize and maxSeqLen must be positive");

    constexpr size_t kElemSize = 2; // half / bfloat16
    int32_t const numBlocks = ceilDiv(maxSeqLen, kQSA_COMPRESS_RATIO);
    int32_t const rowChunk = getRowChunk(batchSize, maxSeqLen, numBlocks);
    int64_t const totalRows = static_cast<int64_t>(batchSize) * maxSeqLen;
    // Runtime chunkItems = rowChunk(S) * MB(S) = min(totalRows * MB, floor(K / MB) * MB or
    // kMinRowChunk * MB) is NOT monotone in seqLen (floor(K / MB) * MB peaks when MB divides
    // K exactly). Size the sort slots with the monotone upper bound
    //   boundItems(S) = min(totalRows * MB, max(K, kMinRowChunk * MB))
    // which dominates the runtime value at every seqLen <= maxSeqLen, so a workspace sized
    // at the profile max always covers the runtime carve.
    int64_t const boundItems = std::min(totalRows * numBlocks, std::max(kMaxSortKeyElements, kMinRowChunk * numBlocks));

    size_t size = 0;
    // Slot 0: qNormed [B, S, 4, 128] T.
    size += alignUp(static_cast<size_t>(totalRows) * kQSA_INDEXER_NUM_HEADS * kQSA_INDEXER_HEAD_DIM * kElemSize);
    // Slot 1: kbar [B, MB, 128] T.
    size += alignUp(static_cast<size_t>(batchSize) * numBlocks * kQSA_INDEXER_HEAD_DIM * kElemSize);
    // Slots 2-3: sort keys (logits) + alternate, FP32, boundItems elements.
    size += 2 * alignUp(static_cast<size_t>(boundItems) * sizeof(float));
    // Slots 4-5: sort values (block ids) + alternate, INT32, boundItems elements.
    size += 2 * alignUp(static_cast<size_t>(boundItems) * sizeof(int32_t));
    // Slot 6: cub segmented radix sort temp storage. The DoubleBuffer overload needs only
    // bookkeeping-scale temp storage; budget the probe at the max shape plus a fixed 1 MiB
    // floor to stay safe against smaller-shape probes and CCCL version drift.
    size += alignUp(std::max(getCubSortTempBytes(rowChunk, numBlocks), static_cast<size_t>(1) << 20));
    return size;
}

template <typename T>
void runQsaIndexerPrefill(int32_t* outIdx, T const* indexQk, float const* cosSin, int32_t const* contextLengths,
    T const* wQ, T const* wK, float rmsEps, void* workspace, size_t workspaceBytes, int32_t batchSize, int32_t seqLen,
    QsaIndexerPoolState const* poolState, cudaStream_t stream)
{
    ELLM_CHECK(outIdx != nullptr && indexQk != nullptr && cosSin != nullptr && contextLengths != nullptr
            && wQ != nullptr && wK != nullptr && workspace != nullptr,
        "runQsaIndexerPrefill: null pointer argument");
    ELLM_CHECK(batchSize > 0 && seqLen > 0, "runQsaIndexerPrefill: batchSize and seqLen must be positive");
    ELLM_CHECK(reinterpret_cast<uintptr_t>(workspace) % kWorkspaceAlignment == 0,
        "runQsaIndexerPrefill: workspace must be 128-byte aligned");
    ELLM_CHECK(workspaceBytes >= getQsaIndexerWorkspaceSize(batchSize, seqLen),
        "runQsaIndexerPrefill: workspace too small; need "
            + std::to_string(getQsaIndexerWorkspaceSize(batchSize, seqLen)) + " bytes, got "
            + std::to_string(workspaceBytes));

    int32_t const numBlocks = ceilDiv(seqLen, kQSA_COMPRESS_RATIO);
    int32_t const rowChunk = getRowChunk(batchSize, seqLen, numBlocks);
    int64_t const totalRows = static_cast<int64_t>(batchSize) * seqLen;
    int64_t const chunkItems = static_cast<int64_t>(rowChunk) * numBlocks;

    // Carve the workspace; layout documented at getQsaIndexerWorkspaceSize.
    std::byte* cursor = static_cast<std::byte*>(workspace);
    T* const qNormed = assignFromWorkspace<T>(
        cursor, static_cast<size_t>(totalRows) * kQSA_INDEXER_NUM_HEADS * kQSA_INDEXER_HEAD_DIM * sizeof(T));
    T* const kbar = assignFromWorkspace<T>(
        cursor, static_cast<size_t>(batchSize) * numBlocks * kQSA_INDEXER_HEAD_DIM * sizeof(T));
    float* const sortKeys = assignFromWorkspace<float>(cursor, static_cast<size_t>(chunkItems) * sizeof(float));
    float* const sortKeysAlt = assignFromWorkspace<float>(cursor, static_cast<size_t>(chunkItems) * sizeof(float));
    int32_t* const sortValues = assignFromWorkspace<int32_t>(cursor, static_cast<size_t>(chunkItems) * sizeof(int32_t));
    int32_t* const sortValuesAlt
        = assignFromWorkspace<int32_t>(cursor, static_cast<size_t>(chunkItems) * sizeof(int32_t));
    size_t const cubTempBytes = getCubSortTempBytes(rowChunk, numBlocks);
    void* const cubTemp = assignFromWorkspace<std::byte>(cursor, cubTempBytes);

    // K1a + K1b: prep all query heads and compressed keys once. With a pool state the K1b
    // paged variant additionally persists each kbar to the pool V-tails (dense output
    // bit-identical), and the tail scatter persists the raw index-K of the trailing
    // incomplete block — the decode pipeline's B1/B2 read exactly this state.
    launchQsaIndexQPrep<T>(qNormed, indexQk, cosSin, contextLengths, wQ, rmsEps, batchSize, seqLen, stream);
    if (poolState != nullptr)
    {
        launchQsaRawKTailWritePrefill<T>(indexQk, contextLengths, *poolState, batchSize, seqLen, stream);
        launchQsaIndexKCompressPaged<T>(kbar, indexQk, cosSin, contextLengths, wK, rmsEps, *poolState, batchSize,
            seqLen, numBlocks, /* blockBegin = */ 0, /* blockEnd = */ numBlocks, /* pastLen = */ 0, stream);
    }
    else
    {
        launchQsaIndexKCompress<T>(kbar, indexQk, cosSin, contextLengths, wK, rmsEps, batchSize, seqLen, numBlocks,
            /* blockBegin = */ 0, /* blockEnd = */ numBlocks, /* pastLen = */ 0, stream);
    }

    SegmentOffsetIterator const beginOffsets(thrust::counting_iterator<int32_t>(0), SegmentOffsetOp{numBlocks});
    SegmentOffsetIterator const endOffsets(thrust::counting_iterator<int32_t>(1), SegmentOffsetOp{numBlocks});

    for (int64_t rowStart = 0; rowStart < totalRows; rowStart += rowChunk)
    {
        int32_t const numRows = static_cast<int32_t>(std::min<int64_t>(rowChunk, totalRows - rowStart));

        launchQsaIndexIdsFill(sortValues, numRows, numBlocks, stream);
        launchQsaIndexScores<T>(sortKeys, qNormed, kbar, contextLengths, batchSize, seqLen, numBlocks,
            static_cast<int32_t>(rowStart), numRows, stream);

        // Per-row descending sort of (logit, blockId) over fixed-width segments. DoubleBuffer
        // halves the temp storage; Current() tells which buffer holds the sorted values.
        cub::DoubleBuffer<float> keys(sortKeys, sortKeysAlt);
        cub::DoubleBuffer<int32_t> values(sortValues, sortValuesAlt);
        size_t tempBytes = cubTempBytes;
        CUDA_CHECK(cub::DeviceSegmentedRadixSort::SortPairsDescending(cubTemp, tempBytes, keys, values,
            numRows * numBlocks, numRows, beginOffsets, endOffsets, 0, static_cast<int>(sizeof(float) * 8), stream));

        launchQsaIndexExpand(outIdx, values.Current(), contextLengths, batchSize, seqLen, numBlocks,
            static_cast<int32_t>(rowStart), numRows, stream);
    }
}

size_t getQsaIndexerDecodeWorkspaceSize(int32_t maxBatchSize, int32_t maxBlocks)
{
    ELLM_CHECK(maxBatchSize > 0 && maxBlocks > 0,
        "getQsaIndexerDecodeWorkspaceSize: maxBatchSize and maxBlocks must be positive");

    constexpr size_t kElemSize = 2; // half / bfloat16
    size_t size = 0;
    // Slot 0: qNormed [B, 1, 4, 128] T.
    size += alignUp(static_cast<size_t>(maxBatchSize) * kQSA_INDEXER_NUM_HEADS * kQSA_INDEXER_HEAD_DIM * kElemSize);
    // Slot 1: logits [B, maxBlocks] FP32.
    size += alignUp(static_cast<size_t>(maxBatchSize) * maxBlocks * sizeof(float));
    return size;
}

template <typename T>
void runQsaIndexerDecode(int32_t* outIdx, T const* indexQk, float const* cosSin, int32_t const* contextLengths,
    T const* wQ, T const* wK, float rmsEps, QsaIndexerPoolState const& poolState, int32_t* splitCounters,
    void* workspace, size_t workspaceBytes, int32_t batchSize, int32_t maxBlocks, cudaStream_t stream)
{
    ELLM_CHECK(outIdx != nullptr && indexQk != nullptr && cosSin != nullptr && contextLengths != nullptr
            && wQ != nullptr && wK != nullptr && workspace != nullptr,
        "runQsaIndexerDecode: null pointer argument");
    ELLM_CHECK(batchSize > 0 && maxBlocks > 0, "runQsaIndexerDecode: batchSize and maxBlocks must be positive");
    ELLM_CHECK(reinterpret_cast<uintptr_t>(workspace) % kWorkspaceAlignment == 0,
        "runQsaIndexerDecode: workspace must be 128-byte aligned");
    ELLM_CHECK(workspaceBytes >= getQsaIndexerDecodeWorkspaceSize(batchSize, maxBlocks),
        "runQsaIndexerDecode: workspace too small; need "
            + std::to_string(getQsaIndexerDecodeWorkspaceSize(batchSize, maxBlocks)) + " bytes, got "
            + std::to_string(workspaceBytes));

    // Carve the workspace; layout documented at getQsaIndexerDecodeWorkspaceSize.
    std::byte* cursor = static_cast<std::byte*>(workspace);
    T* const qNormed = assignFromWorkspace<T>(
        cursor, static_cast<size_t>(batchSize) * kQSA_INDEXER_NUM_HEADS * kQSA_INDEXER_HEAD_DIM * sizeof(T));
    float* const logits
        = assignFromWorkspace<float>(cursor, static_cast<size_t>(batchSize) * maxBlocks * sizeof(float));

    launchQsaIndexerPreDecode<T>(
        qNormed, indexQk, cosSin, contextLengths, wQ, wK, rmsEps, poolState, batchSize, stream);
    launchQsaIndexScoresDecode<T>(logits, qNormed, contextLengths, poolState, batchSize, maxBlocks, stream);
    launchQsaTopKExpandDecode(
        outIdx, logits, contextLengths, splitCounters, poolState.numKVHeads, batchSize, maxBlocks, stream);
}

// Explicit instantiations for the supported activation types.
template void runQsaIndexerPrefill<half>(int32_t*, half const*, float const*, int32_t const*, half const*, half const*,
    float, void*, size_t, int32_t, int32_t, QsaIndexerPoolState const*, cudaStream_t);
template void runQsaIndexerPrefill<__nv_bfloat16>(int32_t*, __nv_bfloat16 const*, float const*, int32_t const*,
    __nv_bfloat16 const*, __nv_bfloat16 const*, float, void*, size_t, int32_t, int32_t, QsaIndexerPoolState const*,
    cudaStream_t);

template void runQsaIndexerDecode<half>(int32_t*, half const*, float const*, int32_t const*, half const*, half const*,
    float, QsaIndexerPoolState const&, int32_t*, void*, size_t, int32_t, int32_t, cudaStream_t);
template void runQsaIndexerDecode<__nv_bfloat16>(int32_t*, __nv_bfloat16 const*, float const*, int32_t const*,
    __nv_bfloat16 const*, __nv_bfloat16 const*, float, QsaIndexerPoolState const&, int32_t*, void*, size_t, int32_t,
    int32_t, cudaStream_t);

} // namespace kernel
} // namespace trt_edgellm
