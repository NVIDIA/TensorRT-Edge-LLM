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

#include "common/checkMacros.h"
#include "common/stringUtils.h"
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

// Single-layer tensor<->cache copy. Linear-vectorized per-(kv, head) scheme: one CTA handles one
// (kv*head) slice for the configured decoder layer, threads iterate linearly over seqLen*HEAD_DIM
// in VEC_SIZE chunks. HEAD_DIM-agnostic as long as it is a positive multiple of VEC_SIZE.
template <typename T, int32_t HEAD_DIM, bool TENSOR_TO_CACHE>
__global__ void instantiateKVCacheKernel(T* KVCacheBuffer, T* KVCacheTensor, int64_t kvCacheMaxBatch,
    int64_t kvCacheMaxSequenceLength, int64_t batchIdx, int64_t numDecoderLayers, int64_t numKVHeads,
    int64_t sequenceLength, int64_t headDim)
{
    static_assert(HEAD_DIM == 64 || HEAD_DIM == 128 || HEAD_DIM == 256 || HEAD_DIM == 512,
        "Only HEAD_DIM = 64, 128, 256, or 512 is supported.");
    using Vec = DVec<T>;
    constexpr int32_t VEC_SIZE = Vec::vec_size;
    static_assert(HEAD_DIM % VEC_SIZE == 0, "HEAD_DIM must be a multiple of vector size.");

    // Grid x: decoderLayerIdx * (2 * numKVHeads) + kvHeadIdx. Treat 2*numKVHeads as flat dim.
    int32_t const CTAIdx = blockIdx.x;
    int64_t const decoderLayerIdx = CTAIdx / (2 * numKVHeads);
    int64_t const kvHeadIdx = CTAIdx % (2 * numKVHeads);

    // Tensor layout: [numDecoderLayers, 2, numKVHeads, sequenceLength, headDim]
    // Cache  layout: [numDecoderLayers, maxBatch, 2, numKVHeads, maxSeqLen, headDim]
    int64_t const ctaTensorOffset
        = decoderLayerIdx * 2 * numKVHeads * sequenceLength * HEAD_DIM + kvHeadIdx * sequenceLength * HEAD_DIM;
    int64_t const ctaCacheOffset
        = decoderLayerIdx * (kvCacheMaxBatch * 2 * numKVHeads * kvCacheMaxSequenceLength * HEAD_DIM)
        + batchIdx * 2 * numKVHeads * kvCacheMaxSequenceLength * HEAD_DIM
        + kvHeadIdx * kvCacheMaxSequenceLength * HEAD_DIM;

    int32_t const numVecs = (sequenceLength * HEAD_DIM) / VEC_SIZE;
    int32_t const tid = threadIdx.y * blockDim.x + threadIdx.x;
    int32_t const threadsPerBlock = blockDim.x * blockDim.y;

    T* tensorPtr = KVCacheTensor + ctaTensorOffset;
    T* cachePtr = KVCacheBuffer + ctaCacheOffset;

    Vec vec;
    for (int32_t vecIdx = tid; vecIdx < numVecs; vecIdx += threadsPerBlock)
    {
        int64_t const elemOff = static_cast<int64_t>(vecIdx) * VEC_SIZE;
        if constexpr (TENSOR_TO_CACHE)
        {
            vec.load(tensorPtr + elemOff);
            vec.store(cachePtr + elemOff);
        }
        else
        {
            vec.load(cachePtr + elemOff);
            vec.store(tensorPtr + elemOff);
        }
    }
}

void instantiateKVCacheLayerFromTensor(
    rt::Tensor& dstKVCacheLayer, rt::Tensor const& srcKVCacheTensor, int32_t batchIdx, cudaStream_t stream)
{
    // srcKVCacheTensor shape: [2, numKVHeads, sequenceLength, headDim]
    auto const& srcShape = srcKVCacheTensor.getShape();
    check::check(srcShape.getNumDims() == 4 && srcShape[0] == 2,
        "Source tensor must be 4D: [2, numKVHeads, sequenceLength, headDim]");

    int32_t const numKVHeads = srcShape[1];
    int32_t const sequenceLength = srcShape[2];
    int32_t const headDim = srcShape[3];

    // dstKVCacheLayer shape: [maxBatchSize, 2, numKVHeads, maxSequenceLength, headDim]
    auto const& dstShape = dstKVCacheLayer.getShape();
    int32_t const kvCacheMaxBatch = dstShape[0];
    int32_t const kvCacheMaxSequenceLength = dstShape[3];

    ELLM_CHECK(batchIdx < kvCacheMaxBatch,
        "batchIdx out of range. Max=" + std::to_string(kvCacheMaxBatch) + ", got=" + std::to_string(batchIdx));
    ELLM_CHECK(sequenceLength <= kvCacheMaxSequenceLength,
        "sequenceLength exceeds max. Max=" + std::to_string(kvCacheMaxSequenceLength)
            + ", got=" + std::to_string(sequenceLength));
    ELLM_CHECK(dstKVCacheLayer.getDataType() == nvinfer1::DataType::kHALF, "Only half type is supported.");

    // Single layer: grid = 2 * numKVHeads CTAs
    dim3 gridDim(2 * numKVHeads);
    dim3 blockDim(32, 4);

    half* srcPtr = const_cast<half*>(srcKVCacheTensor.dataPointer<half>());

    switch (headDim)
    {
    case 64:
        instantiateKVCacheKernel<half, 64, true><<<gridDim, blockDim, 0, stream>>>(dstKVCacheLayer.dataPointer<half>(),
            srcPtr, kvCacheMaxBatch, kvCacheMaxSequenceLength, batchIdx, 1, numKVHeads, sequenceLength, headDim);
        break;
    case 128:
        instantiateKVCacheKernel<half, 128, true><<<gridDim, blockDim, 0, stream>>>(dstKVCacheLayer.dataPointer<half>(),
            srcPtr, kvCacheMaxBatch, kvCacheMaxSequenceLength, batchIdx, 1, numKVHeads, sequenceLength, headDim);
        break;
    case 256:
        instantiateKVCacheKernel<half, 256, true><<<gridDim, blockDim, 0, stream>>>(dstKVCacheLayer.dataPointer<half>(),
            srcPtr, kvCacheMaxBatch, kvCacheMaxSequenceLength, batchIdx, 1, numKVHeads, sequenceLength, headDim);
        break;
    case 512:
        instantiateKVCacheKernel<half, 512, true><<<gridDim, blockDim, 0, stream>>>(dstKVCacheLayer.dataPointer<half>(),
            srcPtr, kvCacheMaxBatch, kvCacheMaxSequenceLength, batchIdx, 1, numKVHeads, sequenceLength, headDim);
        break;
    default:
        throw std::runtime_error("instantiateKVCacheLayerFromTensor(): Unsupported headDim=" + std::to_string(headDim));
    }
}

void saveKVCacheLayerIntoTensor(
    rt::Tensor& dstKVCacheTensor, rt::Tensor const& srcKVCacheLayer, int32_t batchIdx, cudaStream_t stream)
{
    // dstKVCacheTensor shape: [2, numKVHeads, sequenceLength, headDim]
    auto const& dstShape = dstKVCacheTensor.getShape();
    check::check(dstShape.getNumDims() == 4 && dstShape[0] == 2,
        "Destination tensor must be 4D: [2, numKVHeads, sequenceLength, headDim]");

    int32_t const numKVHeads = dstShape[1];
    int32_t const sequenceLength = dstShape[2];
    int32_t const headDim = dstShape[3];

    // srcKVCacheLayer shape: [maxBatchSize, 2, numKVHeads, maxSequenceLength, headDim]
    auto const& srcShape = srcKVCacheLayer.getShape();
    int32_t const kvCacheMaxBatch = srcShape[0];
    int32_t const kvCacheMaxSequenceLength = srcShape[3];

    ELLM_CHECK(batchIdx < kvCacheMaxBatch,
        "batchIdx out of range. Max=" + std::to_string(kvCacheMaxBatch) + ", got=" + std::to_string(batchIdx));
    ELLM_CHECK(sequenceLength <= kvCacheMaxSequenceLength,
        "sequenceLength exceeds max. Max=" + std::to_string(kvCacheMaxSequenceLength)
            + ", got=" + std::to_string(sequenceLength));
    ELLM_CHECK(dstKVCacheTensor.getDataType() == nvinfer1::DataType::kHALF, "Only half type is supported.");

    dim3 gridDim(2 * numKVHeads);
    dim3 blockDim(32, 4);

    half* srcPtr = const_cast<half*>(srcKVCacheLayer.dataPointer<half>());

    switch (headDim)
    {
    case 64:
        instantiateKVCacheKernel<half, 64, false><<<gridDim, blockDim, 0, stream>>>(srcPtr,
            dstKVCacheTensor.dataPointer<half>(), kvCacheMaxBatch, kvCacheMaxSequenceLength, batchIdx, 1, numKVHeads,
            sequenceLength, headDim);
        break;
    case 128:
        instantiateKVCacheKernel<half, 128, false><<<gridDim, blockDim, 0, stream>>>(srcPtr,
            dstKVCacheTensor.dataPointer<half>(), kvCacheMaxBatch, kvCacheMaxSequenceLength, batchIdx, 1, numKVHeads,
            sequenceLength, headDim);
        break;
    case 256:
        instantiateKVCacheKernel<half, 256, false><<<gridDim, blockDim, 0, stream>>>(srcPtr,
            dstKVCacheTensor.dataPointer<half>(), kvCacheMaxBatch, kvCacheMaxSequenceLength, batchIdx, 1, numKVHeads,
            sequenceLength, headDim);
        break;
    case 512:
        instantiateKVCacheKernel<half, 512, false><<<gridDim, blockDim, 0, stream>>>(srcPtr,
            dstKVCacheTensor.dataPointer<half>(), kvCacheMaxBatch, kvCacheMaxSequenceLength, batchIdx, 1, numKVHeads,
            sequenceLength, headDim);
        break;
    default: throw std::runtime_error("saveKVCacheLayerIntoTensor(): Unsupported headDim=" + std::to_string(headDim));
    }
}

//=============================================================================
// Batched KV Cache Copy Kernel (save / restore across layers)
//=============================================================================

/// TENSOR_TO_CACHE == true  => copy from tensor (saved) to cache buffer (restore)
/// TENSOR_TO_CACHE == false => copy from cache buffer to tensor (save)
///
/// Linear-vectorized scheme (same as compactKVCacheBatchedKernel):
///   - Grid: (maxKVHeads * 2, numLayers). CTAs with blockIdx.x >= 2*layer.numKVHeads early-exit,
///     so layers in a HeadDimGroup may have different numKVHeads as long as maxKVHeads bounds them.
///   - Each CTA owns one (layer, kvIdx*head) slice and copies seqLen*HEAD_DIM elements.
///   - Threads iterate linearly in VEC_SIZE chunks — HEAD_DIM-agnostic, unlike the old
///     warp-per-seq-stride scheme which dies when kELEM_PER_WARP < HEAD_DIM (HEAD_DIM=512).
template <typename T, int32_t HEAD_DIM, bool TENSOR_TO_CACHE>
__global__ void batchedKVCacheCopyKernel(KVLayerInfo const* __restrict__ cacheLayerInfos,
    KVLayerInfo const* __restrict__ tensorLayerInfos, int64_t kvCacheMaxBatch, int64_t batchIdx, int64_t sequenceLength)
{
    static_assert(HEAD_DIM == 64 || HEAD_DIM == 128 || HEAD_DIM == 256 || HEAD_DIM == 512,
        "Only HEAD_DIM = 64, 128, 256, or 512 is supported.");
    using Vec = DVec<T>;
    constexpr int32_t VEC_SIZE = Vec::vec_size;
    static_assert(HEAD_DIM % VEC_SIZE == 0, "HEAD_DIM must be a multiple of vector size.");

    int32_t const layerIdx = blockIdx.y;
    int32_t const kvHeadIdx = blockIdx.x; // combined kv * head index

    KVLayerInfo const cacheInfo = cacheLayerInfos[layerIdx];
    KVLayerInfo const tensorInfo = tensorLayerInfos[layerIdx];
    int32_t const numKVHeads = cacheInfo.numKVHeads;
    int32_t const kvCacheMaxSeqLen = cacheInfo.maxSeqLen;

    if (kvHeadIdx >= numKVHeads * 2)
    {
        return;
    }

    T* cacheBuffer = static_cast<T*>(cacheInfo.data);
    T* tensorBuffer = static_cast<T*>(tensorInfo.data);

    // Tensor layout: [2, numKVHeads, sequenceLength, headDim] — NO batch dimension
    int64_t const tensorOffset = static_cast<int64_t>(kvHeadIdx) * sequenceLength * HEAD_DIM;

    // Cache layout: [maxBatch, 2, numKVHeads, maxSeqLen, headDim]
    int64_t const cacheOffset = batchIdx * 2 * numKVHeads * kvCacheMaxSeqLen * HEAD_DIM
        + static_cast<int64_t>(kvHeadIdx) * kvCacheMaxSeqLen * HEAD_DIM;

    int32_t const numVecs = static_cast<int32_t>((sequenceLength * HEAD_DIM) / VEC_SIZE);
    int32_t const tid = threadIdx.y * blockDim.x + threadIdx.x;
    int32_t const threadsPerBlock = blockDim.x * blockDim.y;

    T* tensorPtr = tensorBuffer + tensorOffset;
    T* cachePtr = cacheBuffer + cacheOffset;

    Vec vec;
    for (int32_t vecIdx = tid; vecIdx < numVecs; vecIdx += threadsPerBlock)
    {
        int64_t const elemOff = static_cast<int64_t>(vecIdx) * VEC_SIZE;
        if constexpr (TENSOR_TO_CACHE)
        {
            vec.load(tensorPtr + elemOff);
            vec.store(cachePtr + elemOff);
        }
        else
        {
            vec.load(cachePtr + elemOff);
            vec.store(tensorPtr + elemOff);
        }
    }
}

void saveKVCacheBatched(KVLayerInfo const* srcLayerInfos, KVLayerInfo const* dstLayerInfos, int32_t numLayers,
    int32_t headDim, int32_t maxKVHeads, int32_t maxBatchSize, int32_t batchIdx, int32_t sequenceLength,
    cudaStream_t stream)
{
    if (numLayers == 0 || sequenceLength == 0)
    {
        return;
    }

    dim3 grid(maxKVHeads * 2, numLayers);
    dim3 block(32, 4);

    switch (headDim)
    {
    case 64:
        batchedKVCacheCopyKernel<half, 64, false>
            <<<grid, block, 0, stream>>>(srcLayerInfos, dstLayerInfos, maxBatchSize, batchIdx, sequenceLength);
        break;
    case 128:
        batchedKVCacheCopyKernel<half, 128, false>
            <<<grid, block, 0, stream>>>(srcLayerInfos, dstLayerInfos, maxBatchSize, batchIdx, sequenceLength);
        break;
    case 256:
        batchedKVCacheCopyKernel<half, 256, false>
            <<<grid, block, 0, stream>>>(srcLayerInfos, dstLayerInfos, maxBatchSize, batchIdx, sequenceLength);
        break;
    case 512:
        batchedKVCacheCopyKernel<half, 512, false>
            <<<grid, block, 0, stream>>>(srcLayerInfos, dstLayerInfos, maxBatchSize, batchIdx, sequenceLength);
        break;
    default:
        throw std::invalid_argument(
            format::fmtstr("saveKVCacheBatched: Unsupported headDim=%d. Only 64, 128, 256, or 512.", headDim));
    }
    CUDA_CHECK(cudaGetLastError());
}

void instantiateKVCacheBatched(KVLayerInfo const* dstLayerInfos, KVLayerInfo const* srcLayerInfos, int32_t numLayers,
    int32_t headDim, int32_t maxKVHeads, int32_t maxBatchSize, int32_t batchIdx, int32_t sequenceLength,
    cudaStream_t stream)
{
    if (numLayers == 0 || sequenceLength == 0)
    {
        return;
    }

    dim3 grid(maxKVHeads * 2, numLayers);
    dim3 block(32, 4);

    switch (headDim)
    {
    case 64:
        batchedKVCacheCopyKernel<half, 64, true>
            <<<grid, block, 0, stream>>>(dstLayerInfos, srcLayerInfos, maxBatchSize, batchIdx, sequenceLength);
        break;
    case 128:
        batchedKVCacheCopyKernel<half, 128, true>
            <<<grid, block, 0, stream>>>(dstLayerInfos, srcLayerInfos, maxBatchSize, batchIdx, sequenceLength);
        break;
    case 256:
        batchedKVCacheCopyKernel<half, 256, true>
            <<<grid, block, 0, stream>>>(dstLayerInfos, srcLayerInfos, maxBatchSize, batchIdx, sequenceLength);
        break;
    case 512:
        batchedKVCacheCopyKernel<half, 512, true>
            <<<grid, block, 0, stream>>>(dstLayerInfos, srcLayerInfos, maxBatchSize, batchIdx, sequenceLength);
        break;
    default:
        throw std::invalid_argument(
            format::fmtstr("instantiateKVCacheBatched: Unsupported headDim=%d. Only 64, 128, 256, or 512.", headDim));
    }
    CUDA_CHECK(cudaGetLastError());
}

void instantiateKVCacheFromTensor(
    rt::Tensor& dstKVCacheBuffer, rt::Tensor const& srcKVCacheTensor, int32_t batchIdx, cudaStream_t stream)
{
    // srcKVCacheTensor shape: [numDecoderLayers, 2, numKVHeads, sequenceLength, headDim]
    int32_t const numDecoderLayers = srcKVCacheTensor.getShape()[0];
    int32_t const numKVHeads = srcKVCacheTensor.getShape()[2];
    int32_t const sequenceLength = srcKVCacheTensor.getShape()[3];
    int32_t const headDim = srcKVCacheTensor.getShape()[4];

    // dstKVCacheBuffer shape: [numDecoderLayers, maxBatchSize, 2, numKVHeads, maxSequenceLength, headDim]
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

    dim3 gridDim(numDecoderLayers * 2 * numKVHeads);
    dim3 blockDim(32, 4);
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
    CUDA_CHECK(cudaGetLastError());
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

    dim3 gridDim(numDecoderLayers * 2 * numKVHeads);
    dim3 blockDim(32, 4);
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
    CUDA_CHECK(cudaGetLastError());
}

} // namespace kernel
} // namespace trt_edgellm