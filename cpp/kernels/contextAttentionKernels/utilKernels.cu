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
            kvCacheEndIndices[i] = kvEndIdx;

            if (paddedCuKVSeqLens != nullptr)
            {
                runningPaddedCuKvLen += kvEndIdx;
                paddedCuKVSeqLens[i + 1] = runningPaddedCuKvLen;
            }
        }
    }
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

} // namespace kernel
} // namespace trt_edgellm
