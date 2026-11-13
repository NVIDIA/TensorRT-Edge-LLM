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
#include <limits>
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

// CUDA 11.x CUB requires the begin and end offsets to have the same iterator type.
struct SegmentOffsetOp
{
    int32_t numBlocks;
    int32_t numRows;
    int32_t const* visibleCounts;
    bool isEnd;

    __host__ __device__ __forceinline__ int32_t operator()(int32_t i) const
    {
        // An empty trailing segment keeps max(endOffsets) equal to the capacity-sized num_items.
        int32_t const count = !isEnd || i == numRows ? 0 : (visibleCounts == nullptr ? numBlocks : visibleCounts[i]);
        return i * numBlocks + count;
    }
};

using SegmentOffsetIterator = thrust::transform_iterator<SegmentOffsetOp, thrust::counting_iterator<int32_t>>;

//! cub temp storage bytes for one descending segmented sort of numRows segments x numBlocks keys.
size_t getCubSortTempBytes(int32_t numRows, int32_t numBlocks)
{
    size_t tempBytes = 0;
    cub::DoubleBuffer<float> keys(nullptr, nullptr);
    cub::DoubleBuffer<int32_t> values(nullptr, nullptr);
    SegmentOffsetIterator const beginOffsets(
        thrust::counting_iterator<int32_t>(0), SegmentOffsetOp{numBlocks, numRows, nullptr, false});
    SegmentOffsetIterator const endOffsets(
        thrust::counting_iterator<int32_t>(0), SegmentOffsetOp{numBlocks, numRows, nullptr, true});
    CUDA_CHECK(cub::DeviceSegmentedRadixSort::SortPairsDescending(nullptr, tempBytes, keys, values, numRows * numBlocks,
        numRows + 1, beginOffsets, endOffsets, 0, static_cast<int>(sizeof(float) * 8)));
    return tempBytes;
}

} // namespace

size_t getQsaIndexerWorkspaceSize(int32_t batchSize, int32_t seqLen, int32_t maxBlocks, bool paged)
{
    ELLM_CHECK(batchSize > 0 && seqLen > 0 && maxBlocks > 0, "getQsaIndexerWorkspaceSize: dimensions must be positive");
    int64_t const totalRows = static_cast<int64_t>(batchSize) * seqLen;
    ELLM_CHECK(totalRows <= std::numeric_limits<int32_t>::max(),
        "getQsaIndexerWorkspaceSize: query row count exceeds INT32 range");
    int32_t const rowChunk = getRowChunk(batchSize, seqLen, maxBlocks);
    ELLM_CHECK(static_cast<int64_t>(rowChunk) * maxBlocks <= std::numeric_limits<int32_t>::max(),
        "getQsaIndexerWorkspaceSize: sort item count exceeds INT32 range");
    int64_t const boundItems = std::min(totalRows * maxBlocks, std::max(kMaxSortKeyElements, kMinRowChunk * maxBlocks));
    constexpr size_t kElemSize = 2;
    size_t size = alignUp(static_cast<size_t>(totalRows) * kQSA_INDEXER_NUM_HEADS * kQSA_INDEXER_HEAD_DIM * kElemSize);
    if (!paged)
    {
        size += alignUp(static_cast<size_t>(batchSize) * maxBlocks * kQSA_INDEXER_HEAD_DIM * kElemSize);
    }
    // This monotone bound covers the row-chunk rounding at every smaller profile shape.
    size += 2 * alignUp(static_cast<size_t>(boundItems) * sizeof(float));
    size += 2 * alignUp(static_cast<size_t>(boundItems) * sizeof(int32_t));
    // Reserve counts for all rows so the bound stays monotone when a smaller capacity enlarges rowChunk.
    size += alignUp(static_cast<size_t>(totalRows) * sizeof(int32_t));
    // DoubleBuffer uses bookkeeping-scale CUB scratch; the 1 MiB floor adds headroom for
    // smaller-profile probe variation while retaining any larger queried requirement.
    size += alignUp(std::max(getCubSortTempBytes(rowChunk, maxBlocks), static_cast<size_t>(1) << 20));
    return size;
}

template <typename T>
void runQsaIndexerPrefill(int32_t* outIdx, T const* indexQk, float const* cosSin, int32_t const* queryLengths,
    int32_t const* pastLengths, T const* wQ, T const* wK, float rmsEps, QsaIndexerPoolState const* poolState,
    void* workspace, size_t workspaceBytes, int32_t batchSize, int32_t seqLen, int32_t maxBlocks, cudaStream_t stream)
{
    ELLM_CHECK(outIdx != nullptr && indexQk != nullptr && cosSin != nullptr && queryLengths != nullptr && wQ != nullptr
            && wK != nullptr && workspace != nullptr,
        "runQsaIndexerPrefill: null pointer argument");
    ELLM_CHECK(poolState != nullptr || pastLengths == nullptr,
        "runQsaIndexerPrefill: a cached prefix requires paged pool state");
    ELLM_CHECK(pastLengths != nullptr
            || maxBlocks >= (static_cast<int64_t>(seqLen) + kQSA_COMPRESS_RATIO - 1) / kQSA_COMPRESS_RATIO,
        "runQsaIndexerPrefill: maxBlocks must cover ceilDiv(seqLen, 4) when pastLengths is null");
    ELLM_CHECK(reinterpret_cast<uintptr_t>(workspace) % kWorkspaceAlignment == 0,
        "runQsaIndexerPrefill: workspace must be 128-byte aligned");
    size_t const requiredBytes = getQsaIndexerWorkspaceSize(batchSize, seqLen, maxBlocks, poolState != nullptr);
    ELLM_CHECK(workspaceBytes >= requiredBytes,
        "runQsaIndexerPrefill: workspace too small; need " + std::to_string(requiredBytes) + " bytes, got "
            + std::to_string(workspaceBytes));

    int32_t const rowChunk = getRowChunk(batchSize, seqLen, maxBlocks);
    int64_t const totalRows = static_cast<int64_t>(batchSize) * seqLen;
    int64_t const chunkItems = static_cast<int64_t>(rowChunk) * maxBlocks;
    std::byte* cursor = static_cast<std::byte*>(workspace);
    T* const qNormed = assignFromWorkspace<T>(
        cursor, static_cast<size_t>(totalRows) * kQSA_INDEXER_NUM_HEADS * kQSA_INDEXER_HEAD_DIM * sizeof(T));
    T* const kbar = poolState == nullptr
        ? assignFromWorkspace<T>(cursor, static_cast<size_t>(batchSize) * maxBlocks * kQSA_INDEXER_HEAD_DIM * sizeof(T))
        : nullptr;
    float* const sortKeys = assignFromWorkspace<float>(cursor, static_cast<size_t>(chunkItems) * sizeof(float));
    float* const sortKeysAlt = assignFromWorkspace<float>(cursor, static_cast<size_t>(chunkItems) * sizeof(float));
    int32_t* const sortValues = assignFromWorkspace<int32_t>(cursor, static_cast<size_t>(chunkItems) * sizeof(int32_t));
    int32_t* const sortValuesAlt
        = assignFromWorkspace<int32_t>(cursor, static_cast<size_t>(chunkItems) * sizeof(int32_t));
    int32_t* const visibleCounts
        = assignFromWorkspace<int32_t>(cursor, static_cast<size_t>(rowChunk) * sizeof(int32_t));
    size_t const cubTempBytes = getCubSortTempBytes(rowChunk, maxBlocks);
    void* const cubTemp = assignFromWorkspace<std::byte>(cursor, cubTempBytes);

    if (pastLengths != nullptr)
    {
        launchQsaIndexerSeal<T>(pastLengths, wK, rmsEps, *poolState, batchSize, stream);
    }
    launchQsaIndexQPrep<T>(qNormed, indexQk, cosSin, queryLengths, wQ, rmsEps, batchSize, seqLen, stream);
    if (poolState != nullptr)
    {
        launchQsaIndexKCompressAppend<T>(
            indexQk, cosSin, queryLengths, pastLengths, wK, rmsEps, *poolState, batchSize, seqLen, stream);
        launchQsaRawKTailWritePrefill<T>(
            indexQk, cosSin, queryLengths, pastLengths, *poolState, batchSize, seqLen, stream);
    }
    else
    {
        launchQsaIndexKCompress<T>(
            kbar, indexQk, cosSin, queryLengths, wK, rmsEps, batchSize, seqLen, maxBlocks, stream);
    }

    SegmentOffsetIterator const beginOffsets(
        thrust::counting_iterator<int32_t>(0), SegmentOffsetOp{maxBlocks, rowChunk, nullptr, false});
    for (int64_t rowStart = 0; rowStart < totalRows; rowStart += rowChunk)
    {
        int32_t const numRows = static_cast<int32_t>(std::min<int64_t>(rowChunk, totalRows - rowStart));
        launchQsaIndexIdsFill(sortValues, numRows, maxBlocks, stream);
        if (poolState != nullptr)
        {
            launchQsaIndexScoresPaged<T>(sortKeys, visibleCounts, qNormed, queryLengths, pastLengths, *poolState,
                batchSize, seqLen, maxBlocks, static_cast<int32_t>(rowStart), numRows, stream);
        }
        else
        {
            launchQsaIndexScores<T>(sortKeys, visibleCounts, qNormed, kbar, queryLengths, batchSize, seqLen, maxBlocks,
                static_cast<int32_t>(rowStart), numRows, stream);
        }
        cub::DoubleBuffer<float> keys(sortKeys, sortKeysAlt);
        cub::DoubleBuffer<int32_t> values(sortValues, sortValuesAlt);
        size_t tempBytes = cubTempBytes;
        SegmentOffsetIterator const endOffsets(
            thrust::counting_iterator<int32_t>(0), SegmentOffsetOp{maxBlocks, numRows, visibleCounts, true});
        CUDA_CHECK(
            cub::DeviceSegmentedRadixSort::SortPairsDescending(cubTemp, tempBytes, keys, values, numRows * maxBlocks,
                numRows + 1, beginOffsets, endOffsets, 0, static_cast<int>(sizeof(float) * 8), stream));
        launchQsaIndexExpand(outIdx, values.Current(), queryLengths, pastLengths, batchSize, seqLen, maxBlocks,
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
template void runQsaIndexerPrefill<half>(int32_t*, half const*, float const*, int32_t const*, int32_t const*,
    half const*, half const*, float, QsaIndexerPoolState const*, void*, size_t, int32_t, int32_t, int32_t,
    cudaStream_t);
template void runQsaIndexerPrefill<__nv_bfloat16>(int32_t*, __nv_bfloat16 const*, float const*, int32_t const*,
    int32_t const*, __nv_bfloat16 const*, __nv_bfloat16 const*, float, QsaIndexerPoolState const*, void*, size_t,
    int32_t, int32_t, int32_t, cudaStream_t);

template void runQsaIndexerDecode<half>(int32_t*, half const*, float const*, int32_t const*, half const*, half const*,
    float, QsaIndexerPoolState const&, int32_t*, void*, size_t, int32_t, int32_t, cudaStream_t);
template void runQsaIndexerDecode<__nv_bfloat16>(int32_t*, __nv_bfloat16 const*, float const*, int32_t const*,
    __nv_bfloat16 const*, __nv_bfloat16 const*, float, QsaIndexerPoolState const&, int32_t*, void*, size_t, int32_t,
    int32_t, cudaStream_t);

size_t getQsaIndexerVerifyWorkspaceSize(int32_t batchSize, int32_t seqLen, int32_t maxBlocks)
{
    ELLM_CHECK(batchSize > 0 && seqLen > 0 && seqLen <= kQSA_VERIFY_MAX_NODES && maxBlocks > 0
            && maxBlocks <= std::numeric_limits<int32_t>::max() / kQSA_COMPRESS_RATIO,
        "getQsaIndexerVerifyWorkspaceSize: invalid dimensions");
    int64_t const rows = static_cast<int64_t>(batchSize) * seqLen;
    ELLM_CHECK(rows <= std::numeric_limits<int32_t>::max(), "getQsaIndexerVerifyWorkspaceSize: too many rows");
    constexpr size_t kElemBytes = 2;
    size_t size = alignUp(rows * kQSA_INDEXER_NUM_HEADS * kQSA_INDEXER_HEAD_DIM * kElemBytes);
    size += alignUp(rows * kQSA_INDEXER_HEAD_DIM * kElemBytes);
    size += alignUp(rows * seqLen * sizeof(int32_t));
    size += alignUp(rows * sizeof(int32_t));
    size += alignUp(rows * maxBlocks * sizeof(float));
    return size;
}

template <typename T>
void runQsaIndexerVerify(int32_t* outIdx, T const* indexQk, float const* cosSin, int32_t const* parents,
    int32_t const* depths, int32_t const* queryLengths, int32_t const* pastLengths, T const* wQ, T const* wK,
    float rmsEps, QsaIndexerPoolState const& poolState, void* workspace, size_t workspaceBytes, int32_t batchSize,
    int32_t seqLen, int32_t maxBlocks, cudaStream_t stream)
{
    ELLM_CHECK(outIdx != nullptr && indexQk != nullptr && cosSin != nullptr && parents != nullptr && depths != nullptr
            && queryLengths != nullptr && pastLengths != nullptr && wQ != nullptr && wK != nullptr
            && workspace != nullptr && stream != nullptr,
        "runQsaIndexerVerify: null pointer or default stream");
    ELLM_CHECK(reinterpret_cast<uintptr_t>(workspace) % kWorkspaceAlignment == 0,
        "runQsaIndexerVerify: workspace must be 128-byte aligned");
    ELLM_CHECK(workspaceBytes >= getQsaIndexerVerifyWorkspaceSize(batchSize, seqLen, maxBlocks),
        "runQsaIndexerVerify: insufficient workspace");
    size_t const rows = static_cast<size_t>(batchSize) * seqLen;
    std::byte* cursor = static_cast<std::byte*>(workspace);
    QsaIndexerVerifyScratch scratch{};
    scratch.qNormed = assignFromWorkspace<T>(cursor, rows * kQSA_INDEXER_NUM_HEADS * kQSA_INDEXER_HEAD_DIM * sizeof(T));
    scratch.newKbar = assignFromWorkspace<T>(cursor, rows * kQSA_INDEXER_HEAD_DIM * sizeof(T));
    scratch.paths = assignFromWorkspace<int32_t>(cursor, rows * seqLen * sizeof(int32_t));
    scratch.logicalLengths = assignFromWorkspace<int32_t>(cursor, rows * sizeof(int32_t));
    scratch.logits = assignFromWorkspace<float>(cursor, rows * maxBlocks * sizeof(float));
    launchQsaIndexerSeal<T>(pastLengths, wK, rmsEps, poolState, batchSize, stream);
    launchQsaIndexerVerify(outIdx, indexQk, cosSin, parents, depths, queryLengths, pastLengths, wQ, wK, rmsEps,
        poolState, scratch, batchSize, seqLen, maxBlocks, stream);
}

template void runQsaIndexerVerify<half>(int32_t*, half const*, float const*, int32_t const*, int32_t const*,
    int32_t const*, int32_t const*, half const*, half const*, float, QsaIndexerPoolState const&, void*, size_t, int32_t,
    int32_t, int32_t, cudaStream_t);
template void runQsaIndexerVerify<__nv_bfloat16>(int32_t*, __nv_bfloat16 const*, float const*, int32_t const*,
    int32_t const*, int32_t const*, int32_t const*, __nv_bfloat16 const*, __nv_bfloat16 const*, float,
    QsaIndexerPoolState const&, void*, size_t, int32_t, int32_t, int32_t, cudaStream_t);

} // namespace kernel
} // namespace trt_edgellm
