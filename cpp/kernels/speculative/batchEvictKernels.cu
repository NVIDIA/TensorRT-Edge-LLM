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
__global__ void compactTensorBatchKernel(T const* src, int32_t const* batchMapping, T* dst, int32_t oldActiveBatch,
    int32_t newActiveBatch, int32_t batchStride)
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

        if (newBatchIdx < 0 || newBatchIdx >= newActiveBatch)
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
    check::check(oldActiveBatch > 0, "Old active batch must be positive");
    check::check(newActiveBatch >= 0 && newActiveBatch <= oldActiveBatch,
        "New active batch must be non-negative and no larger than the old active batch");
    check::check(batchMapping.getDataType() == nvinfer1::DataType::kINT32 && batchMapping.getShape().getNumDims() == 1
            && batchMapping.getShape()[0] == oldActiveBatch,
        "Batch mapping must be an INT32 vector matching oldActiveBatch");
    check::check(src.getDataType() == dst.getDataType(), "Source and destination data types must match");

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
        compactTensorBatchKernel<half><<<gridDim, blockDim, 0, stream>>>(src.dataPointer<half>(), batchMappingPtr,
            dst.dataPointer<half>(), oldActiveBatch, newActiveBatch, batchStrideInt);
        break;
    case nvinfer1::DataType::kFLOAT:
        compactTensorBatchKernel<float><<<gridDim, blockDim, 0, stream>>>(src.dataPointer<float>(), batchMappingPtr,
            dst.dataPointer<float>(), oldActiveBatch, newActiveBatch, batchStrideInt);
        break;
    case nvinfer1::DataType::kINT32:
        compactTensorBatchKernel<int32_t><<<gridDim, blockDim, 0, stream>>>(src.dataPointer<int32_t>(), batchMappingPtr,
            dst.dataPointer<int32_t>(), oldActiveBatch, newActiveBatch, batchStrideInt);
        break;
    case nvinfer1::DataType::kINT64:
        compactTensorBatchKernel<int64_t><<<gridDim, blockDim, 0, stream>>>(src.dataPointer<int64_t>(), batchMappingPtr,
            dst.dataPointer<int64_t>(), oldActiveBatch, newActiveBatch, batchStrideInt);
        break;
    case nvinfer1::DataType::kINT8:
        compactTensorBatchKernel<int8_t><<<gridDim, blockDim, 0, stream>>>(src.dataPointer<int8_t>(), batchMappingPtr,
            dst.dataPointer<int8_t>(), oldActiveBatch, newActiveBatch, batchStrideInt);
        break;
    // FP8 is 1-byte POD storage; copy it byte-wise via uint8_t.
    case nvinfer1::DataType::kFP8:
        compactTensorBatchKernel<uint8_t>
            <<<gridDim, blockDim, 0, stream>>>(static_cast<uint8_t const*>(src.rawPointer()), batchMappingPtr,
                static_cast<uint8_t*>(dst.rawPointer()), oldActiveBatch, newActiveBatch, batchStrideInt);
        break;
    default:
        throw std::invalid_argument(format::fmtstr(
            "compactTensorBatch: Unsupported data type=%d. Only HALF, FLOAT, INT32, INT64, INT8, and FP8 are "
            "supported.",
            static_cast<int>(dataType)));
    }

    CUDA_CHECK(cudaGetLastError());
}

void compactExecutionTensorBatch(rt::Tensor& tensor, rt::Tensor const& batchMapping, int32_t oldActiveBatch,
    int32_t newActiveBatch, cudaStream_t stream)
{
    check::check(oldActiveBatch > 0, "Old active batch must be positive");
    check::check(newActiveBatch > 0 && newActiveBatch <= oldActiveBatch,
        "New active batch must be positive and no larger than the old active batch");
    rt::Coords const originalShape = tensor.getShape();
    check::check(originalShape.getNumDims() > 0, "Execution tensor must have at least one dimension");
    check::check(originalShape[0] % oldActiveBatch == 0,
        "Execution tensor leading dimension must contain equally sized sequence blocks");

    int64_t const rowsPerSequence = originalShape[0] / oldActiveBatch;
    std::vector<int64_t> batchedShape;
    batchedShape.reserve(static_cast<size_t>(originalShape.getNumDims()) + 1);
    batchedShape.push_back(oldActiveBatch);
    batchedShape.push_back(rowsPerSequence);
    for (int32_t dim = 1; dim < originalShape.getNumDims(); ++dim)
    {
        batchedShape.push_back(originalShape[dim]);
    }
    check::check(tensor.reshape(batchedShape), "Failed to expose execution sequence blocks for compaction");
    compactTensorBatch(tensor, batchMapping, tensor, oldActiveBatch, newActiveBatch, stream);

    std::vector<int64_t> compactedShape;
    compactedShape.reserve(originalShape.getNumDims());
    compactedShape.push_back(newActiveBatch * rowsPerSequence);
    for (int32_t dim = 1; dim < originalShape.getNumDims(); ++dim)
    {
        compactedShape.push_back(originalShape[dim]);
    }
    check::check(tensor.reshape(compactedShape), "Failed to restore compacted execution tensor shape");
}

__global__ void compactExecutionRowIndicesKernel(int64_t* indices, int32_t const* batchMapping, int32_t oldActiveBatch,
    int32_t newActiveBatch, int32_t indicesPerSequence, int32_t executionRowsPerSequence)
{
    int32_t const indexInSequence = blockIdx.x * blockDim.x + threadIdx.x;
    if (indexInSequence >= indicesPerSequence)
    {
        return;
    }

    for (int32_t oldSlot = 0; oldSlot < oldActiveBatch; ++oldSlot)
    {
        int32_t const newSlot = batchMapping[oldSlot];
        if (newSlot < 0 || newSlot >= newActiveBatch)
        {
            continue;
        }
        int64_t const sourceIndex = static_cast<int64_t>(oldSlot) * indicesPerSequence + indexInSequence;
        int64_t const destinationIndex = static_cast<int64_t>(newSlot) * indicesPerSequence + indexInSequence;
        int64_t const value = indices[sourceIndex];
        indices[destinationIndex] = value - static_cast<int64_t>(oldSlot) * executionRowsPerSequence
            + static_cast<int64_t>(newSlot) * executionRowsPerSequence;
    }
}

void compactExecutionRowIndices(rt::Tensor& indices, rt::Tensor const& batchMapping, int32_t oldActiveBatch,
    int32_t newActiveBatch, int32_t executionRowsPerSequence, cudaStream_t stream)
{
    check::check(indices.getDeviceType() == rt::DeviceType::kGPU, "Execution-row indices must be on GPU");
    check::check(indices.getDataType() == nvinfer1::DataType::kINT64, "Execution-row indices must be INT64");
    check::check(batchMapping.getDeviceType() == rt::DeviceType::kGPU
            && batchMapping.getDataType() == nvinfer1::DataType::kINT32 && batchMapping.getShape().getNumDims() == 1
            && batchMapping.getShape()[0] == oldActiveBatch,
        "Batch mapping must be a GPU INT32 vector matching oldActiveBatch");
    check::check(oldActiveBatch > 0, "Old active batch must be positive");
    check::check(newActiveBatch > 0 && newActiveBatch <= oldActiveBatch,
        "New active batch must be positive and no larger than the old active batch");
    check::check(executionRowsPerSequence > 0, "Execution rows per sequence must be positive");
    check::check(indices.getShape().getNumDims() > 0 && indices.getShape()[0] == oldActiveBatch,
        "Execution-row indices must be batch-major");

    int64_t const indicesPerSequence64 = indices.getShape().volume() / oldActiveBatch;
    check::check(indicesPerSequence64 > 0 && indicesPerSequence64 <= std::numeric_limits<int32_t>::max(),
        "Execution-row indices per sequence are outside the supported range");
    int32_t const indicesPerSequence = static_cast<int32_t>(indicesPerSequence64);
    constexpr int32_t kThreadsPerBlock = 256;
    int32_t const blocks = (indicesPerSequence + kThreadsPerBlock - 1) / kThreadsPerBlock;
    compactExecutionRowIndicesKernel<<<blocks, kThreadsPerBlock, 0, stream>>>(indices.dataPointer<int64_t>(),
        batchMapping.dataPointer<int32_t>(), oldActiveBatch, newActiveBatch, indicesPerSequence,
        executionRowsPerSequence);
    CUDA_CHECK(cudaGetLastError());

    rt::Coords compactedShape = indices.getShape();
    compactedShape[0] = newActiveBatch;
    check::check(indices.reshape(compactedShape), "Failed to reshape compacted execution-row indices");
}

} // namespace kernel
} // namespace trt_edgellm
