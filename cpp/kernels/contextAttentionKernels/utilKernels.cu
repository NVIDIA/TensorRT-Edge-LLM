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

#include "utilKernels.h"

#include "common/checkMacros.h"
#include "common/pagedKvTypes.h"
#include "kernels/common/slidingWindowUtils.cuh"

#include <algorithm>

namespace trt_edgellm
{
namespace kernel
{

__global__ void calCuQCuKVSeqLensAndKVEndIdxsKernel(int32_t const* inputSeqLen, int32_t const* kvCacheStartIndices,
    int32_t* cuQSeqlen, int32_t* cuKVSeqLens, int32_t* kvCacheEndIndices, int32_t* paddedCuKVSeqLens,
    int32_t runtimeSeqLen, int32_t batchSize)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        cuQSeqlen[0] = 0;
        cuKVSeqLens[0] = 0;
        if (paddedCuKVSeqLens != nullptr)
        {
            paddedCuKVSeqLens[0] = 0;
        }

        int32_t runningCuSeqLen = 0;
        int32_t runningCuKvCacheLen = 0;
        int32_t runningPaddedCuKvLen = 0;
        for (int32_t i = 0; i < batchSize; ++i)
        {
            runningCuSeqLen += inputSeqLen[i];
            cuQSeqlen[i + 1] = runningCuSeqLen;

            int32_t kvCacheStartIdx = 0;
            if (kvCacheStartIndices != nullptr)
            {
                kvCacheStartIdx = kvCacheStartIndices[i];
            }

            runningCuKvCacheLen += (kvCacheStartIdx + inputSeqLen[i]);
            cuKVSeqLens[i + 1] = runningCuKvCacheLen;
            // To keep semantic consistency with the packed QKV layout for RoPE, use runtimeSeqLen here.
            int32_t const kvEndIdx = kvCacheStartIdx + runtimeSeqLen;
            if (kvCacheEndIndices != nullptr)
            {
                kvCacheEndIndices[i] = kvEndIdx;
            }

            if (paddedCuKVSeqLens != nullptr)
            {
                runningPaddedCuKvLen += kvEndIdx;
                paddedCuKVSeqLens[i + 1] = runningPaddedCuKvLen;
            }
        }
    }
}

void calCuQCuKVSeqLens(rt::Tensor const& inputSeqLen, rt::Tensor const& kvCacheStartIndices, rt::Tensor& cuQSeqLens,
    rt::Tensor& cuKVSeqLens, cudaStream_t stream)
{
    int32_t const runtimeBatchSize = static_cast<int32_t>(inputSeqLen.getShape()[0]);
    check::check(cuQSeqLens.getShape()[0] == runtimeBatchSize + 1, "cuQSeqLens shall have shape [B+1].");
    check::check(cuKVSeqLens.getShape()[0] == runtimeBatchSize + 1, "cuKVSeqLens shall have shape [B+1].");
    check::check(kvCacheStartIndices.isEmpty() || kvCacheStartIndices.getShape()[0] == runtimeBatchSize,
        "KVCacheStartIndices tensor shall have shape [B].");

    calCuQCuKVSeqLensAndKVEndIdxsKernel<<<1, 1, 0, stream>>>(inputSeqLen.dataPointer<int32_t>(),
        kvCacheStartIndices.dataPointer<int32_t>(), cuQSeqLens.dataPointer<int32_t>(),
        cuKVSeqLens.dataPointer<int32_t>(), nullptr, nullptr, 0, runtimeBatchSize);
}

void calCuQCuKVSeqLensAndKVEndIdxs(rt::Tensor const& inputSeqLen, rt::Tensor const& kvCacheStartIndices,
    rt::Tensor& cuQSeqLens, rt::Tensor& cuKVSeqLens, rt::Tensor& kvCacheEndIdxs,
    rt::OptionalOutputTensor paddedCuKVSeqLens, int32_t const runtimeSeqLen, cudaStream_t stream)
{
    int32_t const runtimeBatchSize = static_cast<int32_t>(inputSeqLen.getShape()[0]);

    // Perform necessary shape checks.
    check::check(cuQSeqLens.getShape()[0] == (runtimeBatchSize + 1), "cuQSeqLens shall have shape [B+1].");
    check::check(cuKVSeqLens.getShape()[0] == (runtimeBatchSize + 1), "cuKVSeqLens shall have shape [B+1].");
    check::check(kvCacheEndIdxs.getShape()[0] == runtimeBatchSize, "kvCacheEndIdxs shall have shape [B].");

    if (!kvCacheStartIndices.isEmpty())
    {
        check::check(
            kvCacheStartIndices.getShape()[0] == runtimeBatchSize, "KVCacheStartIndices tensor shall have shape [B].");
    }
    else
    {
        // We rely on this nullptr behavior to indicate whether kvCacheStartIndices is available in the kernel.
        check::check(kvCacheStartIndices.rawPointer() == nullptr,
            "KVCacheStartIndices tensor shall be nullptr when it is empty.");
    }

    int32_t* paddedPtr = nullptr;
    if (paddedCuKVSeqLens.has_value())
    {
        rt::Tensor& paddedTensor = paddedCuKVSeqLens.value().get();
        check::check(paddedTensor.getShape()[0] == (runtimeBatchSize + 1), "paddedCuKVSeqLens shall have shape [B+1].");
        paddedPtr = paddedTensor.dataPointer<int32_t>();
    }

    calCuQCuKVSeqLensAndKVEndIdxsKernel<<<1, 1, 0, stream>>>(inputSeqLen.dataPointer<int32_t>(),
        kvCacheStartIndices.dataPointer<int32_t>(), cuQSeqLens.dataPointer<int32_t>(),
        cuKVSeqLens.dataPointer<int32_t>(), kvCacheEndIdxs.dataPointer<int32_t>(), paddedPtr, runtimeSeqLen,
        runtimeBatchSize);
}

namespace
{

__global__ void calSWAChunkedPrefillMetadataKernel(int32_t const* inputSeqLen, int32_t const* kvCacheStartIndices,
    int32_t* cuQSeqLens, int32_t* cuKVSeqLens, int32_t* kvCacheEndIdxs, int32_t* paddedCuKVSeqLens,
    int32_t runtimeSeqLen, int32_t slidingWindowSize, int32_t batchSize)
{
    if (threadIdx.x != 0 || blockIdx.x != 0)
    {
        return;
    }

    cuQSeqLens[0] = 0;
    cuKVSeqLens[0] = 0;
    paddedCuKVSeqLens[0] = 0;
    int32_t runningQSeqLen = 0;
    int32_t runningKVSeqLen = 0;
    int32_t runningPaddedKVSeqLen = 0;
    for (int32_t batchIdx = 0; batchIdx < batchSize; ++batchIdx)
    {
        int32_t const inputLen = inputSeqLen[batchIdx];
        int32_t const globalStart = kvCacheStartIndices[batchIdx];
        int32_t const residentLength = clampSWAResidentLength(globalStart, slidingWindowSize);
        runningQSeqLen += inputLen;
        runningKVSeqLen += residentLength + inputLen;
        runningPaddedKVSeqLen += residentLength + runtimeSeqLen;
        cuQSeqLens[batchIdx + 1] = runningQSeqLen;
        cuKVSeqLens[batchIdx + 1] = runningKVSeqLen;
        paddedCuKVSeqLens[batchIdx + 1] = runningPaddedKVSeqLen;
        kvCacheEndIdxs[batchIdx] = globalStart + runtimeSeqLen;
    }
}

__global__ void assemblePagedSWAChunkedPrefillFMHAKVKernel(half const* __restrict__ pool,
    int32_t const* __restrict__ pageTable, half const* __restrict__ k, half const* __restrict__ v,
    int32_t const* __restrict__ inputSeqLen, int32_t const* __restrict__ kvCacheStartIndices,
    half* __restrict__ kWorkspace, half* __restrict__ vWorkspace, int64_t totalElements, int32_t qSeqLen,
    int32_t workspaceSeqLen, int32_t maxPagesPerSeq, int32_t numPages, int32_t numKVHeads, int32_t headDim,
    int32_t slidingWindowSize)
{
    int64_t const linearIdx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (linearIdx >= totalElements)
    {
        return;
    }

    int64_t tmp = linearIdx;
    int32_t const dimIdx = static_cast<int32_t>(tmp % headDim);
    tmp /= headDim;
    int32_t const kvHeadIdx = static_cast<int32_t>(tmp % numKVHeads);
    tmp /= numKVHeads;
    int32_t const workspaceTokenIdx = static_cast<int32_t>(tmp % workspaceSeqLen);
    int32_t const batchIdx = static_cast<int32_t>(tmp / workspaceSeqLen);

    half kValue = __float2half(0.0F);
    half vValue = __float2half(0.0F);
    int32_t const globalStart = kvCacheStartIndices[batchIdx];
    int32_t const oldResidentLen = clampSWAResidentLength(globalStart, slidingWindowSize);
    if (workspaceTokenIdx < oldResidentLen)
    {
        int32_t const logicalToken = globalStart - oldResidentLen + workspaceTokenIdx;
        int32_t const logicalPage = logicalToken / rt::kTOKENS_PER_PAGE;
        int32_t const inPage = logicalToken % rt::kTOKENS_PER_PAGE;
        if (logicalPage >= 0 && logicalPage < maxPagesPerSeq)
        {
            int32_t const kPage = pageTable[(batchIdx * 2) * maxPagesPerSeq + logicalPage];
            int32_t const vPage = pageTable[(batchIdx * 2 + 1) * maxPagesPerSeq + logicalPage];
            int64_t const elementInPage = (static_cast<int64_t>(inPage) * numKVHeads + kvHeadIdx) * headDim + dimIdx;
            int64_t const pageStride = static_cast<int64_t>(rt::kTOKENS_PER_PAGE) * numKVHeads * headDim;
            if (kPage >= 0 && kPage < numPages)
            {
                kValue = pool[static_cast<int64_t>(kPage) * pageStride + elementInPage];
            }
            if (vPage >= numPages && vPage < 2 * numPages)
            {
                vValue = pool[static_cast<int64_t>(vPage) * pageStride + elementInPage];
            }
        }
    }
    else
    {
        int32_t const newTokenIdx = workspaceTokenIdx - oldResidentLen;
        int32_t const inputLen = inputSeqLen[batchIdx];
        if (newTokenIdx >= 0 && newTokenIdx < inputLen && newTokenIdx < qSeqLen)
        {
            int64_t const newOffset
                = ((static_cast<int64_t>(batchIdx) * qSeqLen + newTokenIdx) * numKVHeads + kvHeadIdx) * headDim
                + dimIdx;
            kValue = k[newOffset];
            vValue = v[newOffset];
        }
    }
    kWorkspace[linearIdx] = kValue;
    vWorkspace[linearIdx] = vValue;
}

} // namespace

void calSWAChunkedPrefillMetadata(rt::Tensor const& inputSeqLen, rt::Tensor const& kvCacheStartIndices,
    rt::Tensor& cuQSeqLens, rt::Tensor& cuKVSeqLens, rt::Tensor& kvCacheEndIdxs, rt::Tensor& paddedCuKVSeqLens,
    int32_t runtimeSeqLen, int32_t slidingWindowSize, cudaStream_t stream)
{
    int32_t const runtimeBatchSize = static_cast<int32_t>(inputSeqLen.getShape()[0]);
    check::check(runtimeSeqLen > 1, "SWA chunked prefill requires runtimeSeqLen > 1.");
    check::check(slidingWindowSize > 0, "Sliding window size must be positive.");
    check::check(inputSeqLen.getDataType() == nvinfer1::DataType::kINT32, "inputSeqLen must be INT32.");
    check::check(kvCacheStartIndices.getDataType() == nvinfer1::DataType::kINT32,
        "kvCacheStartIndices must be INT32 for SWA chunked prefill.");
    check::check(cuQSeqLens.getDataType() == nvinfer1::DataType::kINT32, "cuQSeqLens must be INT32.");
    check::check(cuKVSeqLens.getDataType() == nvinfer1::DataType::kINT32, "cuKVSeqLens must be INT32.");
    check::check(kvCacheEndIdxs.getDataType() == nvinfer1::DataType::kINT32, "kvCacheEndIdxs must be INT32.");
    check::check(paddedCuKVSeqLens.getDataType() == nvinfer1::DataType::kINT32, "paddedCuKVSeqLens must be INT32.");
    check::check(inputSeqLen.getShape().getNumDims() == 1, "inputSeqLen shall have shape [B].");
    check::check(
        kvCacheStartIndices.getShape().getNumDims() == 1 && kvCacheStartIndices.getShape()[0] == runtimeBatchSize,
        "kvCacheStartIndices shall have shape [B] for SWA chunked prefill.");
    check::check(cuQSeqLens.getShape()[0] == runtimeBatchSize + 1, "cuQSeqLens shall have shape [B+1].");
    check::check(cuKVSeqLens.getShape()[0] == runtimeBatchSize + 1, "cuKVSeqLens shall have shape [B+1].");
    check::check(kvCacheEndIdxs.getShape()[0] == runtimeBatchSize, "kvCacheEndIdxs shall have shape [B].");
    check::check(paddedCuKVSeqLens.getShape()[0] == runtimeBatchSize + 1, "paddedCuKVSeqLens shall have shape [B+1].");

    calSWAChunkedPrefillMetadataKernel<<<1, 1, 0, stream>>>(inputSeqLen.dataPointer<int32_t>(),
        kvCacheStartIndices.dataPointer<int32_t>(), cuQSeqLens.dataPointer<int32_t>(),
        cuKVSeqLens.dataPointer<int32_t>(), kvCacheEndIdxs.dataPointer<int32_t>(),
        paddedCuKVSeqLens.dataPointer<int32_t>(), runtimeSeqLen, slidingWindowSize, runtimeBatchSize);
}

void assemblePagedSWAChunkedPrefillFMHAKV(rt::Tensor const& swaPool, rt::Tensor const& swaPageTable,
    rt::Tensor const& k, rt::Tensor const& v, rt::Tensor const& inputSeqLen, rt::Tensor const& kvCacheStartIndices,
    rt::Tensor& kWorkspace, rt::Tensor& vWorkspace, int32_t slidingWindowSize, cudaStream_t stream)
{
    check::check(swaPool.getDataType() == nvinfer1::DataType::kHALF, "SWA pool must be FP16.");
    check::check(swaPageTable.getDataType() == nvinfer1::DataType::kINT32, "SWA page table must be INT32.");
    check::check(k.getDataType() == nvinfer1::DataType::kHALF && v.getDataType() == nvinfer1::DataType::kHALF,
        "SWA chunked-prefill K/V inputs must be FP16.");
    check::check(
        kWorkspace.getDataType() == nvinfer1::DataType::kHALF && vWorkspace.getDataType() == nvinfer1::DataType::kHALF,
        "SWA chunked-prefill workspaces must be FP16.");
    check::check(slidingWindowSize > 0, "Sliding window size must be positive.");

    rt::Coords const poolShape = swaPool.getShape();
    rt::Coords const pageTableShape = swaPageTable.getShape();
    rt::Coords const kShape = k.getShape();
    rt::Coords const vShape = v.getShape();
    rt::Coords const kWorkspaceShape = kWorkspace.getShape();
    rt::Coords const vWorkspaceShape = vWorkspace.getShape();
    check::check(poolShape.getNumDims() == 5 && poolShape[0] == 2 && poolShape[2] == rt::kTOKENS_PER_PAGE,
        "SWA pool shall have shape [2, numPages, 128, Hkv, D].");
    check::check(pageTableShape.getNumDims() == 3 && pageTableShape[1] == 2,
        "SWA page table shall have shape [B, 2, maxPagesPerSeq].");
    check::check(kShape.getNumDims() == 4 && vShape.getNumDims() == 4,
        "SWA chunked-prefill K/V shall have shape [B, S, Hkv, D].");
    check::check(kWorkspaceShape.getNumDims() == 4 && vWorkspaceShape.getNumDims() == 4,
        "SWA chunked-prefill K/V workspaces shall have shape [B, W+S, Hkv, D].");

    int32_t const batchSize = static_cast<int32_t>(kShape[0]);
    int32_t const qSeqLen = static_cast<int32_t>(kShape[1]);
    int32_t const numKVHeads = static_cast<int32_t>(kShape[2]);
    int32_t const headDim = static_cast<int32_t>(kShape[3]);
    int32_t const workspaceSeqLen = static_cast<int32_t>(kWorkspaceShape[1]);
    int32_t const maxPagesPerSeq = static_cast<int32_t>(pageTableShape[2]);
    int32_t const numPages = static_cast<int32_t>(poolShape[1]);
    check::check(vShape[0] == batchSize && vShape[1] == qSeqLen && vShape[2] == numKVHeads && vShape[3] == headDim,
        "SWA chunked-prefill V shape must match K.");
    check::check(poolShape[3] == numKVHeads && poolShape[4] == headDim, "SWA pool head geometry must match K/V.");
    check::check(pageTableShape[0] == batchSize, "SWA page table batch must match K/V.");
    check::check(kWorkspaceShape[0] == batchSize && kWorkspaceShape[2] == numKVHeads && kWorkspaceShape[3] == headDim
            && workspaceSeqLen == slidingWindowSize + qSeqLen,
        "SWA K workspace shall have shape [B, W+S, Hkv, D].");
    check::check(vWorkspaceShape[0] == batchSize && vWorkspaceShape[1] == workspaceSeqLen
            && vWorkspaceShape[2] == numKVHeads && vWorkspaceShape[3] == headDim,
        "SWA V workspace shape must match K workspace.");
    check::check(inputSeqLen.getShape().getNumDims() == 1 && inputSeqLen.getShape()[0] == batchSize,
        "inputSeqLen shall have shape [B].");
    check::check(kvCacheStartIndices.getShape().getNumDims() == 1 && kvCacheStartIndices.getShape()[0] == batchSize,
        "kvCacheStartIndices shall have shape [B].");

    int64_t const totalElements
        = static_cast<int64_t>(batchSize) * workspaceSeqLen * numKVHeads * static_cast<int64_t>(headDim);
    constexpr int32_t kBLOCK_SIZE = 256;
    int32_t const gridSize = static_cast<int32_t>((totalElements + kBLOCK_SIZE - 1) / kBLOCK_SIZE);
    assemblePagedSWAChunkedPrefillFMHAKVKernel<<<gridSize, kBLOCK_SIZE, 0, stream>>>(swaPool.dataPointer<half>(),
        swaPageTable.dataPointer<int32_t>(), k.dataPointer<half>(), v.dataPointer<half>(),
        inputSeqLen.dataPointer<int32_t>(), kvCacheStartIndices.dataPointer<int32_t>(), kWorkspace.dataPointer<half>(),
        vWorkspace.dataPointer<half>(), totalElements, qSeqLen, workspaceSeqLen, maxPagesPerSeq, numPages, numKVHeads,
        headDim, slidingWindowSize);
}

namespace
{
//! One thread per (batch, position): expand vision-block IDs into per-position
//! [blockBegin, blockEnd] intervals for the vision-block overlay prefill.
__global__ void buildVisionBlockRangesKernel(int32_t const* visionBlockIds, int32_t const* contextLengths,
    int32_t* blockBegin, int32_t* blockEnd, int32_t seqLen)
{
    int32_t const pos
        = static_cast<int32_t>(blockIdx.x) * static_cast<int32_t>(blockDim.x) + static_cast<int32_t>(threadIdx.x);
    int32_t const batch = static_cast<int32_t>(blockIdx.y);
    if (pos >= seqLen)
    {
        return;
    }
    int64_t const base = static_cast<int64_t>(batch) * seqLen;
    int32_t const contextLen = min(contextLengths[batch], seqLen);

    int32_t begin = -1;
    int32_t end = -1;
    if (pos < contextLen)
    {
        int32_t const blockId = visionBlockIds[base + pos];
        if (blockId >= 0)
        {
            // Contiguous-run expansion: blocks are short (a few hundred
            // tokens), so the linear scans are cheap.
            begin = pos;
            end = pos;
            while (begin > 0 && visionBlockIds[base + begin - 1] == blockId)
            {
                --begin;
            }
            while (end + 1 < contextLen && visionBlockIds[base + end + 1] == blockId)
            {
                ++end;
            }
        }
    }
    blockBegin[base + pos] = begin;
    blockEnd[base + pos] = end;
}

__global__ void gatherTokenAlignedRopeKernel(float const* source, float* output, int32_t const* positions,
    int32_t const* queryStartOffsets, int32_t const* queryLengths, int32_t const* stateIndices, int32_t numTokens,
    int32_t sourceRows, int32_t cacheCapacity, int32_t rotaryDim)
{
    int32_t const sequence = static_cast<int32_t>(blockIdx.x);
    int32_t const start = queryStartOffsets[sequence];
    int32_t const end = queryStartOffsets[sequence + 1];
    int32_t const length = queryLengths[sequence];
    if (start < 0 || end < start || end > numTokens || length < 0 || length > end - start)
    {
        return;
    }

    int32_t const sourceRow = sourceRows == 1 ? 0 : stateIndices[sequence];
    bool const validSourceRow = sourceRow >= 0 && sourceRow < sourceRows;
    int32_t const elements = (end - start) * rotaryDim;
    int32_t const tileElement
        = static_cast<int32_t>(blockIdx.y) * static_cast<int32_t>(blockDim.x) + static_cast<int32_t>(threadIdx.x);
    int32_t const tileStride = static_cast<int32_t>(gridDim.y) * static_cast<int32_t>(blockDim.x);
    for (int32_t localElement = tileElement; localElement < elements; localElement += tileStride)
    {
        int32_t const localToken = localElement / rotaryDim;
        int32_t const channel = localElement % rotaryDim;
        int32_t const token = start + localToken;
        int32_t const position = positions[token];
        bool const validToken = localToken < length && position >= 0 && position < cacheCapacity && validSourceRow;
        float value = 0.0F;
        if (validToken)
        {
            int64_t const sourceIndex
                = (static_cast<int64_t>(sourceRow) * cacheCapacity + position) * rotaryDim + channel;
            value = source[sourceIndex];
        }
        output[static_cast<int64_t>(token) * rotaryDim + channel] = value;
    }
}

__global__ void scatterActiveRowsKernel(
    uint8_t const* source, uint8_t* destination, int32_t const* stateIndices, int32_t residentRows, size_t rowBytes)
{
    int32_t const activeRow = static_cast<int32_t>(blockIdx.x);
    int32_t const residentRow = stateIndices[activeRow];
    if (residentRow < 0 || residentRow >= residentRows)
    {
        return;
    }
    uint8_t const* sourceRow = source + static_cast<size_t>(activeRow) * rowBytes;
    uint8_t* destinationRow = destination + static_cast<size_t>(residentRow) * rowBytes;
    if (rowBytes % sizeof(uint4) == 0)
    {
        auto const* sourceVectors = reinterpret_cast<uint4 const*>(sourceRow);
        auto* destinationVectors = reinterpret_cast<uint4*>(destinationRow);
        size_t const vectorCount = rowBytes / sizeof(uint4);
        for (size_t vector = threadIdx.x; vector < vectorCount; vector += blockDim.x)
        {
            destinationVectors[vector] = sourceVectors[vector];
        }
    }
    else
    {
        for (size_t byte = threadIdx.x; byte < rowBytes; byte += blockDim.x)
        {
            destinationRow[byte] = sourceRow[byte];
        }
    }
}

} // namespace

void launchBuildVisionBlockRanges(int32_t const* visionBlockIds, int32_t const* contextLengths, int32_t* blockBegin,
    int32_t* blockEnd, int32_t batchSize, int32_t seqLen, cudaStream_t stream)
{
    check::check(visionBlockIds != nullptr && contextLengths != nullptr && blockBegin != nullptr && blockEnd != nullptr,
        "Vision block range expansion received a null pointer");
    check::check(batchSize > 0 && seqLen > 0, "Vision block range expansion received invalid dimensions");

    constexpr int32_t kRANGE_THREADS = 256;
    dim3 const grid(
        static_cast<uint32_t>((seqLen + kRANGE_THREADS - 1) / kRANGE_THREADS), static_cast<uint32_t>(batchSize));
    buildVisionBlockRangesKernel<<<grid, kRANGE_THREADS, 0, stream>>>(
        visionBlockIds, contextLengths, blockBegin, blockEnd, seqLen);
    CUDA_CHECK(cudaGetLastError());
}

void launchScatterActiveRows(void const* source, void* destination, int32_t const* stateIndices, int32_t activeRows,
    int32_t residentRows, size_t rowBytes, cudaStream_t stream)
{
    check::check(source != nullptr && destination != nullptr && stateIndices != nullptr,
        "Indexed row scatter received a null pointer");
    check::check(activeRows > 0 && residentRows > 0 && rowBytes > 0, "Indexed row scatter received invalid extents");
    scatterActiveRowsKernel<<<activeRows, 256, 0, stream>>>(
        static_cast<uint8_t const*>(source), static_cast<uint8_t*>(destination), stateIndices, residentRows, rowBytes);
    CUDA_CHECK(cudaGetLastError());
}

void launchGatherTokenAlignedRope(float const* source, float* output, int32_t const* positions,
    int32_t const* queryStartOffsets, int32_t const* queryLengths, int32_t const* stateIndices, int32_t numTokens,
    int32_t numSequences, int32_t sourceRows, int32_t cacheCapacity, int32_t rotaryDim, cudaStream_t stream)
{
    check::check(source != nullptr && output != nullptr && positions != nullptr && queryStartOffsets != nullptr
            && queryLengths != nullptr,
        "Token-aligned RoPE gather received a null pointer");
    check::check(
        sourceRows == 1 || stateIndices != nullptr, "Resident token-aligned RoPE gather requires state indices");
    check::check(numTokens > 0 && numSequences > 0 && sourceRows > 0 && cacheCapacity > 0 && rotaryDim > 0,
        "Token-aligned RoPE gather received invalid dimensions");

    constexpr int32_t kGATHER_THREADS = 256;
    constexpr int32_t kMAX_TILES_PER_SEQUENCE = 32;
    int64_t const averageElements = (static_cast<int64_t>(numTokens) * rotaryDim + numSequences - 1) / numSequences;
    int32_t const tilesPerSequence = std::min<int64_t>(
        kMAX_TILES_PER_SEQUENCE, std::max<int64_t>(1, (averageElements + kGATHER_THREADS - 1) / kGATHER_THREADS));
    dim3 const grid(static_cast<uint32_t>(numSequences), static_cast<uint32_t>(tilesPerSequence));
    gatherTokenAlignedRopeKernel<<<grid, kGATHER_THREADS, 0, stream>>>(source, output, positions, queryStartOffsets,
        queryLengths, stateIndices, numTokens, sourceRows, cacheCapacity, rotaryDim);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace kernel
} // namespace trt_edgellm
