/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "kernels/common/vectorizedTypes.cuh"
#include "kvCacheUtilsKernels.h"
#include <cuda_fp16.h>
#include <stdexcept>

namespace trt_edgellm
{
namespace kernel
{

__global__ void incrementLengthTensorKernel(
    int32_t* lengthTensor, int32_t const* incrementLength, int32_t increment, int32_t activeBatchSize)
{
    int32_t tIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t gridSize = blockDim.x * gridDim.x;
    for (int32_t i = tIdx; i < activeBatchSize; i += gridSize)
    {
        if (incrementLength == nullptr)
        {
            lengthTensor[i] += increment;
        }
        else
        {
            lengthTensor[i] += incrementLength[i];
        }
    }
}
void incrementLengthTensor(rt::Tensor& lengthTensor, int32_t increment, cudaStream_t stream)
{
    check::check(lengthTensor.getDeviceType() == rt::DeviceType::kGPU, "The lengthTensor shall reside on GPU.");
    check::check(
        lengthTensor.getDataType() == nvinfer1::DataType::kINT32, "The lengthTensor shall have data type of int32_t.");

    constexpr int32_t kBLOCK_SIZE = 32;
    constexpr int32_t kGRID_SIZE = 1;
    int32_t const activeBatchSize = lengthTensor.getShape()[0];

    incrementLengthTensorKernel<<<kGRID_SIZE, kBLOCK_SIZE, 0, stream>>>(
        lengthTensor.dataPointer<int32_t>(), nullptr, increment, activeBatchSize);
}

void incrementLengthTensor(rt::Tensor& lengthTensor, rt::Tensor const& newIncrementTensor, cudaStream_t stream)
{
    check::check(lengthTensor.getShape()[0] == newIncrementTensor.getShape()[0],
        "The lengthTensor and newIncrementTensor shall have the same batch size.");
    check::check(lengthTensor.getDeviceType() == rt::DeviceType::kGPU
            && newIncrementTensor.getDeviceType() == rt::DeviceType::kGPU,
        "Both input tensors shall reside on GPU.");
    check::check(lengthTensor.getDataType() == nvinfer1::DataType::kINT32
            && newIncrementTensor.getDataType() == nvinfer1::DataType::kINT32,
        "Both input tensors shall have data type of int32_t.");

    constexpr int32_t kBLOCK_SIZE = 32;
    constexpr int32_t kGRID_SIZE = 1;
    int32_t const activeBatchSize = lengthTensor.getShape()[0];
    incrementLengthTensorKernel<<<kGRID_SIZE, kBLOCK_SIZE, 0, stream>>>(
        lengthTensor.dataPointer<int32_t>(), newIncrementTensor.dataPointer<int32_t>(), 0, activeBatchSize);
}

// TODO: Check if CUTE can improve the peroformance or clarity of this kernel.
template <typename T, int32_t HEAD_DIM, bool TENSOR_TO_CACHE>
__global__ void instantiateKVCacheKernel(T* KVCacheBuffer, T* KVCacheTensor, int64_t kvCacheMaxBatch,
    int64_t kvCacheMaxSequenceLength, int64_t batchIdx, int64_t numDecoderLayers, int64_t numKVHeads,
    int64_t sequenceLength, int64_t headDim)
{
    static_assert(HEAD_DIM == 64 || HEAD_DIM == 128, "Only HEAD_DIM = 64 or 128 is supported now.");
    // The kernel will perform data movement between preComputedKVCache and KVCacheBuffer, both are BSHD layout.
    // The kernel have assumptions that:
    //     1. Each CTA will take over the work for one sequence [SxD] for one decoder-layer/KV-Head.
    //     2. Each thread will copy 16 bytes of data (half[8]), each warp will copy 512 bytes (half[256]) data per
    //     iteration.
    //     3. Each CTA will contains 128 threads (4 warps)
    //     4. TENSOR_TO_CACHE = true means the data movement is from preComputedKVCache to KVCacheBuffer,
    //        otherwise from KVCacheBuffer to preComputedKVCache.

    int32_t CTAIdx = blockIdx.x;
    int64_t const decoderLayerIdx = CTAIdx / (2 * numKVHeads);
    int64_t const kvHeadIdx = CTAIdx % (2 * numKVHeads);

    int32_t const tIdx = threadIdx.x;
    int32_t const warpIdx = threadIdx.y;

    // preComputedKVCache layout: [numDecoderLayers, 2, numKVHeads, sequenceLength, headDim], dst KVCache buffer has
    // layout [decoderLayerIdx, maxBatchSize, 2, kvHeadIdx, maxSequenceLength, headDim]. Treat 2xnumKVHeads as one
    // dimension for simplicity.
    int64_t const ctaTensorOffset
        = decoderLayerIdx * 2 * numKVHeads * sequenceLength * HEAD_DIM + kvHeadIdx * sequenceLength * HEAD_DIM;
    int64_t const ctaCacheOffset
        = decoderLayerIdx * (kvCacheMaxBatch * 2 * numKVHeads * kvCacheMaxSequenceLength * HEAD_DIM)
        + batchIdx * 2 * numKVHeads * kvCacheMaxSequenceLength * HEAD_DIM
        + kvHeadIdx * kvCacheMaxSequenceLength * HEAD_DIM;

    // Compute the warp start offset in the CTA and stride among iterations.
    constexpr int32_t kELEM_PER_WARP = DVec<T>::vec_size * 32;
    constexpr int32_t kSLEN_PER_WARP = kELEM_PER_WARP / HEAD_DIM;
    constexpr int32_t kWARP_SLEN_STRIDE = kSLEN_PER_WARP * 4;

    // sequenceLength is dynamic, so there can be leftovers that full warp load/store will exceed the range.
    // Compute the SLen that can be handled by full warp load/store.
    int64_t const sLenFullWarp = (sequenceLength / kSLEN_PER_WARP) * kSLEN_PER_WARP;

    DVec<T> vecData;
    int64_t warpSeqStartIdx = warpIdx * kSLEN_PER_WARP;
    while (warpSeqStartIdx < sLenFullWarp)
    {
        int64_t threadStartTensorOffset = ctaTensorOffset + warpSeqStartIdx * HEAD_DIM + tIdx * DVec<T>::vec_size;
        int64_t threadStartCacheOffset = ctaCacheOffset + warpSeqStartIdx * HEAD_DIM + tIdx * DVec<T>::vec_size;

        if constexpr (TENSOR_TO_CACHE)
        {
            vecData.load(KVCacheTensor + threadStartTensorOffset);
            vecData.store(KVCacheBuffer + threadStartCacheOffset);
        }
        else
        {
            vecData.load(KVCacheBuffer + threadStartCacheOffset);
            vecData.store(KVCacheTensor + threadStartTensorOffset);
        }
        warpSeqStartIdx += kWARP_SLEN_STRIDE;
    }

    // There could be leftovers, not all warp are relvent for this iterations.
    // Given the current supported HEAD_DIM = 64 or 128, each thread will load/store full vector of data.
    if (warpSeqStartIdx < sequenceLength)
    {
        int32_t leftOverElems = (sequenceLength - warpSeqStartIdx) * HEAD_DIM;
        if (tIdx * DVec<T>::vec_size < leftOverElems)
        {
            int64_t threadStartTensorOffset = ctaTensorOffset + warpSeqStartIdx * HEAD_DIM + tIdx * DVec<T>::vec_size;
            int64_t threadStartCacheOffset = ctaCacheOffset + warpSeqStartIdx * HEAD_DIM + tIdx * DVec<T>::vec_size;

            if constexpr (TENSOR_TO_CACHE)
            {
                vecData.load(KVCacheTensor + threadStartTensorOffset);
                vecData.store(KVCacheBuffer + threadStartCacheOffset);
            }
            else
            {
                vecData.load(KVCacheBuffer + threadStartCacheOffset);
                vecData.store(KVCacheTensor + threadStartTensorOffset);
            }
        }
    }
}

void instantiateKVCacheFromTensor(
    rt::Tensor& dstKVCacheBuffer, rt::Tensor const& srcKVCacheTensor, int32_t batchIdx, cudaStream_t stream)
{
    int32_t const numDecoderLayers = srcKVCacheTensor.getShape()[0];
    int32_t const numKVHeads = srcKVCacheTensor.getShape()[2];
    int32_t const sequenceLength = srcKVCacheTensor.getShape()[3];
    int32_t const headDim = srcKVCacheTensor.getShape()[4];

    int32_t const kvCacheMaxBatch = dstKVCacheBuffer.getShape()[1];
    int32_t const kvCacheMaxSequenceLength = dstKVCacheBuffer.getShape()[4];

    if (batchIdx >= kvCacheMaxBatch)
    {
        throw std::runtime_error(
            "instantiateKVCacheFromTensor(): batchIdx is out of range for the KVCache buffer. MaxSupportedBatch = "
            + std::to_string(kvCacheMaxBatch) + ", batchIdx = " + std::to_string(batchIdx));
    }

    if (sequenceLength > kvCacheMaxSequenceLength)
    {
        throw std::runtime_error(
            "instantiateKVCacheFromTensor(): sequenceLength is out of range for the KVCache buffer. "
            "MaxSupportedSequenceLength = "
            + std::to_string(kvCacheMaxSequenceLength) + ", sequenceLength = " + std::to_string(sequenceLength));
    }

    if (dstKVCacheBuffer.getDataType() != srcKVCacheTensor.getDataType()
        && dstKVCacheBuffer.getDataType() != nvinfer1::DataType::kHALF)
    {
        throw std::runtime_error(
            "instantiateKVCacheFromTensor(): KVCacheBuffer and preComputedKVCache shall both be half type now.");
    }

    // Kernel launch parameters that closely match the kernel implementation.
    dim3 gridDim(numDecoderLayers * 2 * numKVHeads);
    dim3 blockDim(32, 4);

    // We implemented bi-directional data movement kernel so we need to perform const_cast here.
    half* srcKVCacheTensorPtr = const_cast<half*>(srcKVCacheTensor.dataPointer<half>());
    switch (headDim)
    {
    case 64:
        instantiateKVCacheKernel<half, 64, true><<<gridDim, blockDim, 0, stream>>>(dstKVCacheBuffer.dataPointer<half>(),
            srcKVCacheTensorPtr, kvCacheMaxBatch, kvCacheMaxSequenceLength, batchIdx, numDecoderLayers, numKVHeads,
            sequenceLength, headDim);
        break;
    case 128:
        instantiateKVCacheKernel<half, 128, true><<<gridDim, blockDim, 0, stream>>>(
            dstKVCacheBuffer.dataPointer<half>(), srcKVCacheTensorPtr, kvCacheMaxBatch, kvCacheMaxSequenceLength,
            batchIdx, numDecoderLayers, numKVHeads, sequenceLength, headDim);
        break;
    default:
        throw std::runtime_error(
            "instantiateKVCacheFromTensor(): Only headDim = 64 or 128 are supported by the kernel, current headDim = "
            + std::to_string(headDim));
    }
}

void saveKVCacheIntoTensor(
    rt::Tensor& dstKVCacheTensor, rt::Tensor const& srcKVCacheBuffer, int32_t batchIdx, cudaStream_t stream)
{
    int32_t const numDecoderLayers = dstKVCacheTensor.getShape()[0];
    int32_t const numKVHeads = dstKVCacheTensor.getShape()[2];
    int32_t const sequenceLength = dstKVCacheTensor.getShape()[3];
    int32_t const headDim = dstKVCacheTensor.getShape()[4];

    int32_t const kvCacheMaxBatch = srcKVCacheBuffer.getShape()[1];
    int32_t const kvCacheMaxSequenceLength = srcKVCacheBuffer.getShape()[4];

    if (batchIdx >= kvCacheMaxBatch)
    {
        throw std::runtime_error(
            "saveKVCacheIntoTensor(): batchIdx is out of range for the KVCache buffer. MaxSupportedBatch = "
            + std::to_string(kvCacheMaxBatch) + ", batchIdx = " + std::to_string(batchIdx));
    }

    if (sequenceLength > kvCacheMaxSequenceLength)
    {
        throw std::runtime_error(
            "saveKVCacheIntoTensor(): sequenceLength is out of range for the KVCache buffer. "
            "MaxSupportedSequenceLength = "
            + std::to_string(kvCacheMaxSequenceLength) + ", sequenceLength = " + std::to_string(sequenceLength));
    }

    if (dstKVCacheTensor.getDataType() != srcKVCacheBuffer.getDataType()
        && dstKVCacheTensor.getDataType() != nvinfer1::DataType::kHALF)
    {
        throw std::runtime_error(
            "saveKVCacheIntoTensor(): KVCacheBuffer and preComputedKVCache shall both be half type now.");
    }

    // Kernel launch parameters that closely match the kernel implementation.
    dim3 gridDim(numDecoderLayers * 2 * numKVHeads);
    dim3 blockDim(32, 4);
    // We implemented bi-directional data movement kernel so we need to perform const_cast here.
    half* srcKVCacheBufferPtr = const_cast<half*>(srcKVCacheBuffer.dataPointer<half>());
    switch (headDim)
    {
    case 64:
        instantiateKVCacheKernel<half, 64, false><<<gridDim, blockDim, 0, stream>>>(srcKVCacheBufferPtr,
            dstKVCacheTensor.dataPointer<half>(), kvCacheMaxBatch, kvCacheMaxSequenceLength, batchIdx, numDecoderLayers,
            numKVHeads, sequenceLength, headDim);
        break;
    case 128:
        instantiateKVCacheKernel<half, 128, false><<<gridDim, blockDim, 0, stream>>>(srcKVCacheBufferPtr,
            dstKVCacheTensor.dataPointer<half>(), kvCacheMaxBatch, kvCacheMaxSequenceLength, batchIdx, numDecoderLayers,
            numKVHeads, sequenceLength, headDim);
        break;
    default:
        throw std::runtime_error(
            "saveKVCacheIntoTensor(): Only headDim = 64 or 128 are supported by the kernel, current headDim = "
            + std::to_string(headDim));
    }
}
} // namespace kernel
} // namespace trt_edgellm