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
#include "common/pagedKvTypes.h"
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
/// NHD pool layout. The cache for one layer is [2, maxBatch, capPadded, H, D] with the K/V
/// split OUTERMOST, so the K-half and V-half are each a contiguous [maxBatch, capPadded, H, D] pool.
/// Within a half, batch row `b`'s first `sequenceLength` tokens form one contiguous span of
/// sequenceLength*H*D elements starting at b*capPadded*H*D — token-major, then head, then dim.
///
/// Saved tensor layout (per layer): [2, sequenceLength, H, D] (K plane then V plane), each plane a
/// contiguous sequenceLength*H*D span. So the whole copy is two contiguous block copies per layer
/// (one for K, one for V) — no per-head striding needed.
///
/// Scheme:
///   - Grid: (2, numLayers). blockIdx.x selects the K-half (0) or V-half (1).
///   - Each CTA copies sequenceLength*H*D contiguous elements between cache row `batchIdx` of the
///     selected half and the matching plane of the saved tensor; threads iterate in VEC_SIZE chunks.
///   - cacheInfo.maxSeqLen carries the cache row's token capacity (capPadded).
template <typename T, int32_t HEAD_DIM, bool TENSOR_TO_CACHE>
__global__ void batchedKVCacheCopyKernel(KVLayerInfo const* __restrict__ cacheLayerInfos,
    KVLayerInfo const* __restrict__ tensorLayerInfos, int64_t kvPoolPages, int64_t batchIdx, int64_t sequenceLength)
{
    static_assert(HEAD_DIM == 64 || HEAD_DIM == 128 || HEAD_DIM == 256 || HEAD_DIM == 512,
        "Only HEAD_DIM = 64, 128, 256, or 512 is supported.");
    using Vec = DVec<T>;
    constexpr int32_t VEC_SIZE = Vec::vec_size;
    static_assert(HEAD_DIM % VEC_SIZE == 0, "HEAD_DIM must be a multiple of vector size.");

    int32_t const layerIdx = blockIdx.y;
    int32_t const kvIdx = blockIdx.x; // 0 = K-half, 1 = V-half

    KVLayerInfo const cacheInfo = cacheLayerInfos[layerIdx];
    KVLayerInfo const tensorInfo = tensorLayerInfos[layerIdx];
    int32_t const numKVHeads = cacheInfo.numKVHeads;
    int32_t const capPadded = cacheInfo.maxSeqLen; // per-row token capacity of the NHD pool

    T* cacheBuffer = static_cast<T*>(cacheInfo.data);
    T* tensorBuffer = static_cast<T*>(tensorInfo.data);

    // Element count of one contiguous (token-major) block for `sequenceLength` tokens, all heads.
    int64_t const blockElems = sequenceLength * static_cast<int64_t>(numKVHeads) * HEAD_DIM;

    // Cache half base: K-half starts at element 0, V-half one full half-pool later.
    int64_t const halfStride = kvPoolPages * static_cast<int64_t>(rt::kTOKENS_PER_PAGE) * numKVHeads * HEAD_DIM;
    int64_t const cacheOffset
        = static_cast<int64_t>(kvIdx) * halfStride + batchIdx * static_cast<int64_t>(capPadded) * numKVHeads * HEAD_DIM;

    // Saved tensor plane base: K plane then V plane, each blockElems long.
    int64_t const tensorOffset = static_cast<int64_t>(kvIdx) * blockElems;

    int32_t const numVecs = static_cast<int32_t>(blockElems / VEC_SIZE);
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
    int32_t headDim, int32_t kvPoolPages, int32_t batchIdx, int32_t sequenceLength, cudaStream_t stream)
{
    if (numLayers == 0 || sequenceLength == 0)
    {
        return;
    }

    // NHD: each layer is two contiguous block copies (K-half, V-half). Grid.x selects the half.
    dim3 grid(2, numLayers);
    dim3 block(32, 4);

    switch (headDim)
    {
    case 64:
        batchedKVCacheCopyKernel<half, 64, false>
            <<<grid, block, 0, stream>>>(srcLayerInfos, dstLayerInfos, kvPoolPages, batchIdx, sequenceLength);
        break;
    case 128:
        batchedKVCacheCopyKernel<half, 128, false>
            <<<grid, block, 0, stream>>>(srcLayerInfos, dstLayerInfos, kvPoolPages, batchIdx, sequenceLength);
        break;
    case 256:
        batchedKVCacheCopyKernel<half, 256, false>
            <<<grid, block, 0, stream>>>(srcLayerInfos, dstLayerInfos, kvPoolPages, batchIdx, sequenceLength);
        break;
    case 512:
        batchedKVCacheCopyKernel<half, 512, false>
            <<<grid, block, 0, stream>>>(srcLayerInfos, dstLayerInfos, kvPoolPages, batchIdx, sequenceLength);
        break;
    default:
        throw std::invalid_argument(
            format::fmtstr("saveKVCacheBatched: Unsupported headDim=%d. Only 64, 128, 256, or 512.", headDim));
    }
    CUDA_CHECK(cudaGetLastError());
}

void instantiateKVCacheBatched(KVLayerInfo const* dstLayerInfos, KVLayerInfo const* srcLayerInfos, int32_t numLayers,
    int32_t headDim, int32_t kvPoolPages, int32_t batchIdx, int32_t sequenceLength, cudaStream_t stream)
{
    if (numLayers == 0 || sequenceLength == 0)
    {
        return;
    }

    // NHD: each layer is two contiguous block copies (K-half, V-half). Grid.x selects the half.
    dim3 grid(2, numLayers);
    dim3 block(32, 4);

    switch (headDim)
    {
    case 64:
        batchedKVCacheCopyKernel<half, 64, true>
            <<<grid, block, 0, stream>>>(dstLayerInfos, srcLayerInfos, kvPoolPages, batchIdx, sequenceLength);
        break;
    case 128:
        batchedKVCacheCopyKernel<half, 128, true>
            <<<grid, block, 0, stream>>>(dstLayerInfos, srcLayerInfos, kvPoolPages, batchIdx, sequenceLength);
        break;
    case 256:
        batchedKVCacheCopyKernel<half, 256, true>
            <<<grid, block, 0, stream>>>(dstLayerInfos, srcLayerInfos, kvPoolPages, batchIdx, sequenceLength);
        break;
    case 512:
        batchedKVCacheCopyKernel<half, 512, true>
            <<<grid, block, 0, stream>>>(dstLayerInfos, srcLayerInfos, kvPoolPages, batchIdx, sequenceLength);
        break;
    default:
        throw std::invalid_argument(
            format::fmtstr("instantiateKVCacheBatched: Unsupported headDim=%d. Only 64, 128, 256, or 512.", headDim));
    }
    CUDA_CHECK(cudaGetLastError());
}

//=============================================================================
// Paged KV gather: page pools -> dense split K/V (for FMHA_v2-style consumers)
//=============================================================================

//! Resolves the K/V page ids for slot `batch`'s logical page and classifies it against the slot's live
//! range. Returns false (skip) when the logical page is entirely beyond `seqLen`. Sets @p beyondLive
//! when the page lies past the slot's live length (kvSeqLens[batch]) -- a legal padding tail that must
//! be zero-filled, never flagged. Otherwise the page is IN-RANGE and its page ids must be valid.
__device__ __forceinline__ bool resolveLivePage(int32_t const* __restrict__ pageTable,
    int32_t const* __restrict__ kvSeqLens, int32_t batch, int32_t logicalPage, int32_t maxPagesPerSeq, int32_t seqLen,
    int32_t& kPage, int32_t& vPage, int32_t& tokenStart, int32_t& tokensInPage, bool& beyondLive)
{
    tokenStart = logicalPage * rt::kTOKENS_PER_PAGE;
    tokensInPage = min(rt::kTOKENS_PER_PAGE, seqLen - tokenStart);
    if (tokensInPage <= 0)
    {
        return false;
    }
    beyondLive = tokenStart >= kvSeqLens[batch];
    kPage = pageTable[(static_cast<size_t>(batch) * 2 + 0) * maxPagesPerSeq + logicalPage];
    vPage = pageTable[(static_cast<size_t>(batch) * 2 + 1) * maxPagesPerSeq + logicalPage];
    return true;
}

//! Byte-copy gather (FP16/uint8 pools). Grid: (batchSize, numLogicalPages). Each block gathers one
//! slot's one logical page of K and V. Both halves index the SAME flat `pool` base (KVPageTable encodes
//! V page ids as K page id + numPages). An unmapped page zero-fills its destination span; a beyond-live page is
//! zero-filled.
__global__ void gatherPagedKVToSplitKernel(uint8_t const* __restrict__ pool, uint8_t* __restrict__ kDst,
    uint8_t* __restrict__ vDst, int32_t const* __restrict__ pageTable, int32_t const* __restrict__ kvSeqLens,
    int32_t maxPagesPerSeq, int32_t seqLen, size_t pageBytes, size_t tokenBytes)
{
    int32_t const batch = blockIdx.x;
    int32_t kPage, vPage, tokenStart, tokensInPage;
    bool beyondLive;
    if (!resolveLivePage(pageTable, kvSeqLens, batch, blockIdx.y, maxPagesPerSeq, seqLen, kPage, vPage, tokenStart,
            tokensInPage, beyondLive))
    {
        return;
    }
    size_t const bytesToCopy = static_cast<size_t>(tokensInPage) * tokenBytes;
    size_t const dstOffset = (static_cast<size_t>(batch) * seqLen + tokenStart) * tokenBytes;

    if (beyondLive)
    {
        for (size_t i = threadIdx.x; i < bytesToCopy; i += blockDim.x)
        {
            kDst[dstOffset + i] = 0;
            vDst[dstOffset + i] = 0;
        }
        return;
    }
    if (kPage < 0 || vPage < 0)
    {
        // Unmapped page: zero-fill the destination span (same as the padding tail) so downstream
        // consumers never read uninitialized workspace. The runtime guarantees mapped coverage for
        // live positions, so this only fires on an upstream table bug -- and fails soft.
        for (size_t i = threadIdx.x; i < bytesToCopy; i += blockDim.x)
        {
            kDst[dstOffset + i] = 0;
            vDst[dstOffset + i] = 0;
        }
        return;
    }
    uint8_t const* kSrc = pool + static_cast<size_t>(kPage) * pageBytes;
    uint8_t const* vSrc = pool + static_cast<size_t>(vPage) * pageBytes;
    for (size_t i = threadIdx.x; i < bytesToCopy; i += blockDim.x)
    {
        kDst[dstOffset + i] = kSrc[i];
        vDst[dstOffset + i] = vSrc[i];
    }
}

#if SUPPORTS_FP8
//! FP8 -> FP16 dequant gather. Same page addressing as the byte-copy kernel, but the pool is FP8 e4m3
//! and each element is dequantized (value * scale) into the FP16 destination, so the downstream
//! `dataPointer<half>()` consumers see real FP16 rather than reinterpreted FP8 bytes.
__global__ void gatherDequantFp8PagedKVToSplitKernel(__nv_fp8_e4m3 const* __restrict__ pool, half* __restrict__ kDst,
    half* __restrict__ vDst, int32_t const* __restrict__ pageTable, int32_t const* __restrict__ kvSeqLens,
    int32_t maxPagesPerSeq, int32_t seqLen, int32_t tokenElems, float kScale, float vScale)
{
    int32_t const batch = blockIdx.x;
    int32_t kPage, vPage, tokenStart, tokensInPage;
    bool beyondLive;
    if (!resolveLivePage(pageTable, kvSeqLens, batch, blockIdx.y, maxPagesPerSeq, seqLen, kPage, vPage, tokenStart,
            tokensInPage, beyondLive))
    {
        return;
    }
    size_t const elemsToCopy = static_cast<size_t>(tokensInPage) * tokenElems;
    size_t const dstOffset = (static_cast<size_t>(batch) * seqLen + tokenStart) * static_cast<size_t>(tokenElems);
    size_t const pageElems = static_cast<size_t>(rt::kTOKENS_PER_PAGE) * tokenElems;

    if (beyondLive)
    {
        for (size_t i = threadIdx.x; i < elemsToCopy; i += blockDim.x)
        {
            kDst[dstOffset + i] = __float2half(0.f);
            vDst[dstOffset + i] = __float2half(0.f);
        }
        return;
    }
    if (kPage < 0 || vPage < 0)
    {
        // Unmapped page: zero-fill (see the byte-copy kernel above).
        for (size_t i = threadIdx.x; i < elemsToCopy; i += blockDim.x)
        {
            kDst[dstOffset + i] = __float2half(0.f);
            vDst[dstOffset + i] = __float2half(0.f);
        }
        return;
    }
    __nv_fp8_e4m3 const* kSrc = pool + static_cast<size_t>(kPage) * pageElems;
    __nv_fp8_e4m3 const* vSrc = pool + static_cast<size_t>(vPage) * pageElems;
    for (size_t i = threadIdx.x; i < elemsToCopy; i += blockDim.x)
    {
        kDst[dstOffset + i] = __float2half(static_cast<float>(kSrc[i]) * kScale);
        vDst[dstOffset + i] = __float2half(static_cast<float>(vSrc[i]) * vScale);
    }
}
#endif // SUPPORTS_FP8

void gatherPagedKVToSplit(void const* pool, void* kDst, void* vDst, int32_t const* pageTable, int32_t const* kvSeqLens,
    int32_t maxPagesPerSeq, int32_t batchSize, int32_t seqLen, int32_t numKVHeads, int32_t headDim, size_t elemSize,
    bool dequantFp8, float kScale, float vScale, cudaStream_t stream)
{
    check::check(batchSize > 0 && seqLen > 0 && numKVHeads > 0 && headDim > 0 && elemSize > 0,
        "gatherPagedKVToSplit: batchSize, seqLen, numKVHeads, headDim, and elemSize must all be positive.");
    check::check(
        kvSeqLens != nullptr, "gatherPagedKVToSplit: kvSeqLens is required (drives the live-range zero-fill).");

    int32_t const numLogicalPages = (seqLen + rt::kTOKENS_PER_PAGE - 1) / rt::kTOKENS_PER_PAGE;
    check::check(numLogicalPages <= maxPagesPerSeq,
        "gatherPagedKVToSplit: seqLen requires more logical pages than the page table provides.");

    int32_t const tokenElems = numKVHeads * headDim;
    size_t const tokenBytes = static_cast<size_t>(tokenElems) * elemSize;
    size_t const pageBytes = tokenBytes * rt::kTOKENS_PER_PAGE;

    constexpr int32_t kTHREADS_PER_BLOCK = 256;
    dim3 grid(static_cast<uint32_t>(batchSize), static_cast<uint32_t>(numLogicalPages));
    if (dequantFp8)
    {
#if SUPPORTS_FP8
        gatherDequantFp8PagedKVToSplitKernel<<<grid, kTHREADS_PER_BLOCK, 0, stream>>>(
            static_cast<__nv_fp8_e4m3 const*>(pool), static_cast<half*>(kDst), static_cast<half*>(vDst), pageTable,
            kvSeqLens, maxPagesPerSeq, seqLen, tokenElems, kScale, vScale);
#else
        throw std::runtime_error("FP8 KV cache requested but CUDA_VERSION < 11080 (cuda_fp8.h unavailable).");
#endif
    }
    else
    {
        gatherPagedKVToSplitKernel<<<grid, kTHREADS_PER_BLOCK, 0, stream>>>(static_cast<uint8_t const*>(pool),
            static_cast<uint8_t*>(kDst), static_cast<uint8_t*>(vDst), pageTable, kvSeqLens, maxPagesPerSeq, seqLen,
            pageBytes, tokenBytes);
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace kernel
} // namespace trt_edgellm
