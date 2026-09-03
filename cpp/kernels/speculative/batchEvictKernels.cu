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

#include "batchEvictKernels.h"
#include "common/checkMacros.h"
#include "common/cudaUtils.h"
#include "common/pagedKvTypes.h"
#include "common/stringUtils.h"
#include "kernels/common/vectorizedTypes.cuh"
#include <cstdint>
#include <cuda_fp16.h>

namespace trt_edgellm
{
namespace kernel
{

//=============================================================================
// Generic Tensor Compaction Kernel
//=============================================================================

template <typename T>
__global__ void compactTensorBatchKernel(
    T const* src, int32_t const* batchMapping, T* dst, int32_t oldActiveBatch, int32_t batchStride)
{
    // Each CTA handles all elements (no batch-specific assignment)
    int32_t const elemIdx = blockIdx.x * blockDim.x + threadIdx.x;

    if (elemIdx >= batchStride)
    {
        return;
    }

    for (int32_t oldBatchIdx = 0; oldBatchIdx < oldActiveBatch; ++oldBatchIdx)
    {
        int32_t const newBatchIdx = batchMapping[oldBatchIdx];

        if (newBatchIdx < 0 || newBatchIdx >= oldActiveBatch)
        {
            continue;
        }

        if (oldBatchIdx == newBatchIdx)
        {
            continue;
        }

        int64_t const srcIdx = static_cast<int64_t>(oldBatchIdx) * batchStride + elemIdx;
        int64_t const dstIdx = static_cast<int64_t>(newBatchIdx) * batchStride + elemIdx;
        dst[dstIdx] = src[srcIdx];
    }
}

void compactTensorBatch(rt::Tensor const& src, rt::Tensor const& batchMapping, rt::Tensor& dst, int32_t oldActiveBatch,
    int32_t newActiveBatch, cudaStream_t stream)
{
    check::check(dst.getDeviceType() == rt::DeviceType::kGPU, "Destination tensor must be on GPU");
    check::check(src.getDeviceType() == rt::DeviceType::kGPU, "Source tensor must be on GPU");
    check::check(batchMapping.getDeviceType() == rt::DeviceType::kGPU, "Batch mapping must be on GPU");

    auto const& srcShape = src.getShape();
    check::check(srcShape.getNumDims() >= 1, "Tensor must have at least 1 dimension");
    check::check(srcShape[0] == oldActiveBatch, "First dimension must match oldActiveBatch");

    int64_t batchStride = 1;
    for (int32_t i = 1; i < srcShape.getNumDims(); ++i)
    {
        batchStride *= srcShape[i];
    }

    check::check(batchStride <= std::numeric_limits<int32_t>::max(), "Batch stride too large for int32_t");

    auto const batchStrideInt = static_cast<int32_t>(batchStride);

    if (batchStrideInt == 0)
    {
        return;
    }

    int32_t const threadsPerBlock = 512;
    int32_t const numBlocks = (batchStrideInt + threadsPerBlock - 1) / threadsPerBlock;

    dim3 gridDim(numBlocks);
    dim3 blockDim(threadsPerBlock);

    int32_t const* batchMappingPtr = batchMapping.dataPointer<int32_t>();

    // Get data type and dispatch to appropriate kernel
    nvinfer1::DataType const dataType = src.getDataType();

    switch (dataType)
    {
    case nvinfer1::DataType::kHALF:
        compactTensorBatchKernel<half><<<gridDim, blockDim, 0, stream>>>(
            src.dataPointer<half>(), batchMappingPtr, dst.dataPointer<half>(), oldActiveBatch, batchStrideInt);
        break;
    case nvinfer1::DataType::kFLOAT:
        compactTensorBatchKernel<float><<<gridDim, blockDim, 0, stream>>>(
            src.dataPointer<float>(), batchMappingPtr, dst.dataPointer<float>(), oldActiveBatch, batchStrideInt);
        break;
    case nvinfer1::DataType::kINT32:
        compactTensorBatchKernel<int32_t><<<gridDim, blockDim, 0, stream>>>(
            src.dataPointer<int32_t>(), batchMappingPtr, dst.dataPointer<int32_t>(), oldActiveBatch, batchStrideInt);
        break;
    case nvinfer1::DataType::kINT64:
        compactTensorBatchKernel<int64_t><<<gridDim, blockDim, 0, stream>>>(
            src.dataPointer<int64_t>(), batchMappingPtr, dst.dataPointer<int64_t>(), oldActiveBatch, batchStrideInt);
        break;
    case nvinfer1::DataType::kINT8:
        compactTensorBatchKernel<int8_t><<<gridDim, blockDim, 0, stream>>>(
            src.dataPointer<int8_t>(), batchMappingPtr, dst.dataPointer<int8_t>(), oldActiveBatch, batchStrideInt);
        break;
    // FP8 is 1-byte POD storage; copy it byte-wise via uint8_t.
    case nvinfer1::DataType::kFP8:
        compactTensorBatchKernel<uint8_t>
            <<<gridDim, blockDim, 0, stream>>>(static_cast<uint8_t const*>(src.rawPointer()), batchMappingPtr,
                static_cast<uint8_t*>(dst.rawPointer()), oldActiveBatch, batchStrideInt);
        break;
    default:
        throw std::invalid_argument(format::fmtstr(
            "compactTensorBatch: Unsupported data type=%d. Only HALF, FLOAT, INT32, INT64, INT8, and FP8 are "
            "supported.",
            static_cast<int>(dataType)));
    }

    CUDA_CHECK(cudaGetLastError());
}

//=============================================================================
// Batched KV pool compaction — grouped, vectorized, live-prefix
//=============================================================================

// One launch covers every layer of a headDim group; blockIdx.y selects (layer, K/V half). In the
// NHD pool a row's live data is its contiguous [0, liveLen*H*D) prefix, so the copy is a flat
// vectorized move. gridDim.x is a small fixed CTA count per (layer, half): scheduled work stays
// proportional to layers, while per-thread loops scale with the LIVE bytes actually moved (not the
// allocated capacity).
//
// In-place overlap safety: compaction only moves a survivor to a strictly lower row
// (newIdx < oldIdx). Every element column is owned by one fixed thread across all rows (ownership
// depends only on the in-row offset), each thread walks oldBatchIdx in ascending order, and it
// reads row r's element during iteration r — before any later iteration (old > r) can overwrite
// row r. Different (layer, half) planes are disjoint, so cross-CTA order does not matter.
template <typename T>
__global__ void compactKVCacheBatchedKernel(KVLayerInfo const* __restrict__ layerInfos,
    int32_t const* __restrict__ batchMapping, int32_t const* __restrict__ liveLengths, int32_t headDim,
    int32_t kvPoolPages, int32_t oldActiveBatch)
{
    KVLayerInfo const info = layerInfos[blockIdx.y >> 1];
    int64_t const elemsPerToken = static_cast<int64_t>(info.numKVHeads) * headDim;
    int64_t const rowElems = static_cast<int64_t>(info.maxSeqLen) * elemsPerToken;
    T* base = static_cast<T*>(info.data);
    if ((blockIdx.y & 1) != 0)
    {
        base += static_cast<int64_t>(kvPoolPages) * rt::kTOKENS_PER_PAGE * elemsPerToken; // V half
    }

    using Vec = DVec<T>;
    constexpr int32_t kVEC = static_cast<int32_t>(Vec::vec_size);
    int64_t const laneStart = (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) * kVEC;
    int64_t const stride = static_cast<int64_t>(gridDim.x) * blockDim.x * kVEC;

    for (int32_t oldBatchIdx = 0; oldBatchIdx < oldActiveBatch; ++oldBatchIdx)
    {
        int32_t const newBatchIdx = batchMapping[oldBatchIdx];
        if (newBatchIdx < 0 || newBatchIdx >= oldActiveBatch || newBatchIdx == oldBatchIdx)
        {
            continue;
        }

        // Only the live prefix of the row is moved; padding beyond it is left untouched.
        int64_t const liveElems = static_cast<int64_t>(liveLengths[oldBatchIdx]) * elemsPerToken;
        T const* src = base + static_cast<int64_t>(oldBatchIdx) * rowElems;
        T* dst = base + static_cast<int64_t>(newBatchIdx) * rowElems;

        for (int64_t e = laneStart; e < liveElems; e += stride)
        {
            if (e + kVEC <= liveElems)
            {
                Vec v;
                v.load(src + e);
                v.store(dst + e);
            }
            else
            {
                // Ragged tail of the live prefix: finish element-wise from the same owning thread
                // (chunk ownership is offset-only, so this stays overlap-safe).
                for (int64_t t = e; t < liveElems; ++t)
                {
                    dst[t] = src[t];
                }
            }
        }
    }
}

void compactKVCacheBatched(KVLayerInfo const* layerInfos, rt::Tensor const& batchMapping, rt::Tensor const& liveLengths,
    int32_t numLayers, int32_t headDim, int32_t kvPoolPages, nvinfer1::DataType kvCacheType, int32_t oldActiveBatch,
    int32_t newActiveBatch, cudaStream_t stream)
{
    check::check(batchMapping.getDeviceType() == rt::DeviceType::kGPU, "Batch mapping must be on GPU");
    check::check(liveLengths.getDeviceType() == rt::DeviceType::kGPU, "Live lengths must be on GPU");
    check::check(kvPoolPages > 0, "KV pool page count must be positive");

    // Identity (nothing finished) and all-evicted (nothing survives) both move zero rows.
    if (numLayers == 0 || oldActiveBatch == newActiveBatch || newActiveBatch <= 0)
    {
        return;
    }

    // Small fixed CTA count per (layer, half): each CTA grid-strides over the live prefix.
    constexpr int32_t kCTAS_PER_ROW = 32;
    dim3 const gridDim(kCTAS_PER_ROW, 2 * numLayers);
    dim3 const blockDim(256);

    int32_t const* batchMappingPtr = batchMapping.dataPointer<int32_t>();
    int32_t const* liveLengthsPtr = liveLengths.dataPointer<int32_t>();

    switch (kvCacheType)
    {
    case nvinfer1::DataType::kHALF:
        compactKVCacheBatchedKernel<half><<<gridDim, blockDim, 0, stream>>>(
            layerInfos, batchMappingPtr, liveLengthsPtr, headDim, kvPoolPages, oldActiveBatch);
        break;
    // FP8 is 1-byte POD storage; copy it byte-wise via uint8_t.
    case nvinfer1::DataType::kFP8:
        compactKVCacheBatchedKernel<uint8_t><<<gridDim, blockDim, 0, stream>>>(
            layerInfos, batchMappingPtr, liveLengthsPtr, headDim, kvPoolPages, oldActiveBatch);
        break;
    default:
        throw std::invalid_argument(
            format::fmtstr("compactKVCacheBatched: Unsupported KV cache data type=%d. Only HALF and FP8 are "
                           "supported.",
                static_cast<int>(kvCacheType)));
    }

    CUDA_CHECK(cudaGetLastError());
}

} // namespace kernel
} // namespace trt_edgellm
