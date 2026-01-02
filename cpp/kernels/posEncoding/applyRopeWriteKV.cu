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

#include "applyRopeWriteKV.h"
#include "common/checkMacros.h"
#include "kernels/common/vectorizedTypes.cuh"

#include <cstdint>
#include <cuda_fp16.h>

namespace trt_edgellm
{
namespace kernel
{

template <typename T>
__device__ __forceinline__ T applyRope(T const& x, T const& y, float const& cos, float const& sin, bool const isLeft);

template <>
__device__ __forceinline__ half applyRope<half>(
    half const& x, half const& y, float const& cos, float const& sin, bool const isLeft)
{
    float val
        = isLeft ? (__half2float(x) * cos - __half2float(y) * sin) : (__half2float(x) * cos + __half2float(y) * sin);
    return __float2half(val);
}

template <typename T>
__device__ __forceinline__ DVec<T> vecApplyRopeNonInterleave(
    T* dataPtr, DVec<float> const& cosVec, DVec<float> const& sinVec, uint32_t const rotaryDim)
{
    DVec<T> result;
    DVec<T> input;
    DVec<T> permuteInput;

    uint32_t const vecOffset = threadIdx.x * DVec<T>::vec_size;
    input.load(dataPtr + vecOffset);

    if (vecOffset < rotaryDim)
    {
        uint32_t const permuteOffset
            = (vecOffset < rotaryDim / 2) ? vecOffset + rotaryDim / 2 : vecOffset - rotaryDim / 2;
        permuteInput.load(dataPtr + permuteOffset);

#pragma unroll
        for (uint32_t i = 0; i < DVec<T>::vec_size; ++i)
        {
            result[i] = applyRope(input[i], permuteInput[i], cosVec[i], sinVec[i], (vecOffset < rotaryDim / 2));
        }
        return result;
    }
    else
    {
        return input;
    }
}

template <typename T>
__global__ void applyRopeWriteKV(T* qkv, T* kvCache, T* qOut, float const* cosSinCache, int32_t const* kvCacheEndLens,
    int32_t const* tokenPosIds, int32_t qSeqLen, int32_t totalNumTokens, int32_t kvCacheCapacity, uint32_t numQHead,
    uint32_t numKVHead, uint32_t headDim, uint32_t rotaryDim, int32_t cosSinCacheBatchSize, int32_t cosSinCacheSeqLen)
{
    // Each CTA will process multiple tokens of a single head which each thread handles 16 / sizeof(T) elements.
    // blockDim.x: number of threads to process each token, blockDim.y: number of tokens processed by each CTA.
    // In this kernel we assume:
    //     1. The input tokens are batched with [B, qSeqLen], we use batchIdx info to write KVCache.
    //     2. Always write KVCache with layout of [B, Hk + Hv, S, headDim] where S = kvCacheCapacityLen.
    //     3. The QKV tensor has layout of [B, S, Hq+Hk+Hv, headDim] where S = qSeqLen.
    //     4. The cosSinCache has layout of [cosSinCacheBatchSize, cosSinCacheSeqLen, rotaryDim] where cosSinCacheSeqLen
    //     >= kvCacheCapacityLen.
    //        cosSinCacheBatchSize can be 1 (all batches share the same cache) or equal to input batch size.
    //     5. Write to qOut if qOut is provided, otherwise overwrite QKV.
    //     6. kvCacheEndLens: Length of KVCache after insertion the entries by this kernel.

    // TODO (fans): Unify and improve the logic of computing token positions/ kvcache insertion positions.

    uint32_t const bIdx = blockIdx.x;
    uint32_t const bIdy = blockIdx.y;
    uint32_t const tIdx = threadIdx.x;
    uint32_t const tIdy = threadIdx.y;

    uint32_t const bDimY = blockDim.y;
    uint32_t const tokenIdx = bIdx * bDimY + tIdy;
    if (tokenIdx >= totalNumTokens)
    {
        return;
    }

    // We assume all the batches have the same qSeqLen (non-ragged)
    int32_t const batchIdx = tokenIdx / qSeqLen;

    // Determine the position of CosSin Cache to read from.
    // Need to handle three scenarios: Context, vanllia decode, and tree attention.
    // Workaround: For vanllia decode use kvCacheEndLens to compute token positions.
    int32_t sinCosCachePos{};
    bool const isPaddingToken = (tokenPosIds != nullptr && tokenPosIds[tokenIdx] == -1);
    if (tokenPosIds != nullptr)
    {
        sinCosCachePos = tokenPosIds[tokenIdx];
        // For padding tokens (position = -1), use position 0 to avoid out-of-bounds access
        // The actual computation for padding tokens will be skipped below
        if (sinCosCachePos < 0)
        {
            sinCosCachePos = 0;
        }
    }
    else
    {
        int32_t const posStartId = kvCacheEndLens != nullptr ? kvCacheEndLens[batchIdx] - qSeqLen : 0;
        sinCosCachePos = posStartId + tokenIdx % qSeqLen;
    }

    // Vectorized load sin/cos cache from global memory.
    // If pos ids are not provided, use token idx in the sequence as cos/sinc cache posId.
    // non-interleaved rope:
    //      - cosVec = cosSinCache[cosSinCacheBatchIdx][sinCosCachePos][(tx * vec_size) % (rotaryDim / 2)]
    //      - sinVec = cosSinCache[cosSinCacheBatchIdx][sinCosCachePos][(tx * vec_size) % (rotaryDim / 2) + rotaryDim /
    //      2]
    // where cosSinCacheBatchIdx = (cosSinCacheBatchSize == 1) ? 0 : batchIdx
    uint32_t const sinOffset = rotaryDim / 2;
    uint32_t cosOffset;
    DVec<float> cosVec;
    DVec<float> sinVec;
    cosOffset = (tIdx * DVec<float>::vec_size) % (rotaryDim / 2);
    int32_t const cosSinCacheBatchIdx = (cosSinCacheBatchSize == 1) ? 0 : batchIdx;
    int32_t const cosSinCacheOffset = cosSinCacheBatchIdx * cosSinCacheSeqLen * rotaryDim + sinCosCachePos * rotaryDim;
    cosVec.load(cosSinCache + cosSinCacheOffset + cosOffset);
    sinVec.load(cosSinCache + cosSinCacheOffset + (cosOffset + sinOffset));

    // tokenIdx is the index of the token in the "flattened" BxS sequence
    int32_t const eleOffsetToken = tokenIdx * (numQHead + numKVHead * 2) * headDim;

    if (bIdy < numQHead)
    {
        int32_t const qHeadIdx = bIdy;
        T* qPTr = qkv + eleOffsetToken + qHeadIdx * headDim;
        DVec<T> qRoped;

        // For padding tokens, output zeros instead of RoPE-transformed values
        if (isPaddingToken)
        {
            // Zero out the Q vector for padding tokens
#pragma unroll
            for (uint32_t i = 0; i < DVec<T>::vec_size; ++i)
            {
                qRoped[i] = T(0);
            }
        }
        else
        {
            qRoped = vecApplyRopeNonInterleave(qPTr, cosVec, sinVec, rotaryDim);
        }

        if (qOut != nullptr)
        {
            int32_t qOutOffset;
            qOutOffset = tokenIdx * numQHead * headDim + qHeadIdx * headDim + DVec<T>::vec_size * tIdx;
            qRoped.store(qOut + qOutOffset);
        }
        else
        {
            int32_t const qOutOffset = eleOffsetToken + qHeadIdx * headDim + DVec<T>::vec_size * tIdx;
            qRoped.store(qkv + qOutOffset);
        }
    }
    else
    {
        // KV write is same for packed QKV and contiguous Q_KV.
        int32_t const kvHeadIdx = bIdy - numQHead;
        int32_t const kvCacheStartIdx = kvCacheEndLens != nullptr ? kvCacheEndLens[batchIdx] - qSeqLen : 0;
        int32_t const tokenIdxInCache = kvCacheStartIdx + tokenIdx % qSeqLen;
        int32_t const cacheOffsetSequence = batchIdx * 2 * numKVHead * kvCacheCapacity * headDim;

        int32_t const srcVOffset = eleOffsetToken + (numQHead + numKVHead) * headDim + kvHeadIdx * headDim;
        DVec<T> vSrc;
        vSrc.load(qkv + srcVOffset + DVec<T>::vec_size * tIdx);

        int32_t const srcKOffset = eleOffsetToken + numQHead * headDim + kvHeadIdx * headDim;
        DVec<T> kRoped;
        kRoped = vecApplyRopeNonInterleave(qkv + srcKOffset, cosVec, sinVec, rotaryDim);

        // This is an unique semantics that we only write kRoped back to QKV when qOut is not provided.
        // When qOut is supplied, the decoding attention kernel will directly read KVCache.
        if (qOut == nullptr)
        {
            kRoped.store(qkv + srcKOffset + DVec<T>::vec_size * tIdx);
        }

        // Skip writing K/V to cache for padding tokens (position = -1)
        // This ensures padding tokens don't corrupt valid cache entries
        if (!isPaddingToken)
        {
            // Save to KVCache which assume to have layout of [B, Hk + Hv, S, D]
            int32_t cacheOffsetK = cacheOffsetSequence + kvHeadIdx * kvCacheCapacity * headDim
                + tokenIdxInCache * headDim + DVec<T>::vec_size * tIdx;
            int32_t cacheOffsetV = cacheOffsetSequence + (numKVHead + kvHeadIdx) * kvCacheCapacity * headDim
                + tokenIdxInCache * headDim + DVec<T>::vec_size * tIdx;
            kRoped.store(kvCache + cacheOffsetK);
            vSrc.store(kvCache + cacheOffsetV);
        }
    }
}

void launchApplyRopeWriteKV(rt::Tensor& qkv, rt::Tensor& kvCache, rt::Tensor const& cosSinCache,
    rt::OptionalInputTensor kvCacheEndLens, rt::OptionalInputTensor tokenPosIds, rt::OptionalOutputTensor qOut,
    cudaStream_t stream)
{
    // QKV has layout of [B, S, H_q + H_k + H_v, D]
    // CosSinCache always in layout of [cosSinCacheBatchSize, cosSinCacheSeqLen, rotaryDim]
    // KVCache has layout of [B, 2, H_kv, S_cache_capacity, D]
    constexpr uint32_t kVEC_SIZE = DVec<half>::vec_size;
    constexpr uint32_t kTHREADS_PER_CTA = 128;

    // Collect runtime and shape information from the input / output tensors.
    // Static cast to uint32_t since we know the shape is always positive and within the range of uint32_t.
    // We use uint32_t in CUDA kernel to save register usage.
    uint32_t const runtimeBatchSize = static_cast<uint32_t>(qkv.getShape()[0]);
    uint32_t const runtimeSeqLen = static_cast<uint32_t>(qkv.getShape()[1]);
    uint32_t const numTotalHeads = static_cast<uint32_t>(qkv.getShape()[2]);
    uint32_t const headDim = static_cast<uint32_t>(qkv.getShape()[3]);
    uint32_t const numKVHeads = static_cast<uint32_t>(kvCache.getShape()[2]);
    uint32_t const kvCacheCapacity = static_cast<uint32_t>(kvCache.getShape()[3]);
    uint32_t const numQHeads = numTotalHeads - 2 * numKVHeads;
    uint32_t const totalNumTokens = runtimeBatchSize * runtimeSeqLen;

    uint32_t const cosSinCacheBatchSize = static_cast<uint32_t>(cosSinCache.getShape()[0]);
    uint32_t const cosSinCacheSeqLen = static_cast<uint32_t>(cosSinCache.getShape()[1]);
    uint32_t const rotaryDim = static_cast<uint32_t>(cosSinCache.getShape()[2]);

    // Device pointers for required input / output tensors.
    half* qkvPtr = qkv.dataPointer<half>();
    half* kvCachePtr = kvCache.dataPointer<half>();
    float const* cosSinCachePtr = cosSinCache.dataPointer<float>();

    // Device pointers for optional input / output tensors.
    int32_t const* kvCacheEndLensPtr
        = kvCacheEndLens.has_value() ? kvCacheEndLens.value().get().dataPointer<int32_t>() : nullptr;
    int32_t const* tokenPosIdsPtr
        = tokenPosIds.has_value() ? tokenPosIds.value().get().dataPointer<int32_t>() : nullptr;
    half* qOutPtr = qOut.has_value() ? qOut.value().get().dataPointer<half>() : nullptr;

    // Collect kernel launch parameters and invoke the kernel.
    // Each CTA will process either Q or KV (together) head of multiple tokens.
    uint32_t const tokenPerCTA = kTHREADS_PER_CTA * kVEC_SIZE / headDim;
    uint32_t const bDimX = headDim / kVEC_SIZE;
    uint32_t const bDimY = tokenPerCTA;
    uint32_t const gDimX = (totalNumTokens + tokenPerCTA - 1) / tokenPerCTA;
    uint32_t const gDimY = numQHeads + numKVHeads;

    dim3 grid(gDimX, gDimY);
    dim3 block(bDimX, bDimY);
    applyRopeWriteKV<half><<<grid, block, 0, stream>>>(qkvPtr, kvCachePtr, qOutPtr, cosSinCachePtr, kvCacheEndLensPtr,
        tokenPosIdsPtr, runtimeSeqLen, totalNumTokens, kvCacheCapacity, numQHeads, numKVHeads, headDim, rotaryDim,
        cosSinCacheBatchSize, cosSinCacheSeqLen);
}

void launchApplyRopeWriteKVPackedQKV(
    rt::Tensor const& cosSinCache, rt::Tensor& qkv, rt::Tensor& kvCache, cudaStream_t stream)
{
    // Handle case where the QKV can be packed into a single tensor. This happens when there are no existing KVCache
    // values. We will overwrite QKV directly and instantiate the KVCache. Later attention kernel can directly perform
    // on the packed QKV tensor.
    // We don't need QOut, kvCacheEndLens, and tokenPosIds.
    rt::OptionalInputTensor kvCacheEndLens{std::nullopt};
    rt::OptionalInputTensor tokenPosIds{std::nullopt};
    rt::OptionalOutputTensor qOut{std::nullopt};

    // Perform necessary consistent checks to ensure the kernel is launched correctly.
    check::check(qkv.getShape()[0] == kvCache.getShape()[0], "QKV and KVCache shall have the same batch size");
    check::check(qkv.getShape()[3] == kvCache.getShape()[4], "QKV and KVCache shall have the same head dimension");
    check::check(cosSinCache.getShape()[0] == 1 || cosSinCache.getShape()[0] == qkv.getShape()[0],
        "CosSinCache shall have batch size 1 or equal to runtime batch size");

    launchApplyRopeWriteKV(qkv, kvCache, cosSinCache, kvCacheEndLens, tokenPosIds, qOut, stream);
}

void launchApplyRopeWriteKVContinuousQAndKVCache(rt::Tensor const& cosSinCache, rt::Tensor const& kvCacheEndLens,
    rt::Tensor& qkv, rt::Tensor& kvCache, rt::Tensor& qOut, cudaStream_t stream)
{
    // Handle case where there are existing KVCache values. Thus, we are unable to use packed QKV tensor for attention
    // computation. Here we will write write to a dedicated Q tensor and KVCache. Since there are existing KVCache
    // values, we need the kvCacheEndLens to indicate where to write in the KVCache. From the end position, we will
    // write runtimeSeqLen tokens "forward".

    // Position to compute rope is same as position to write KVCache.
    rt::OptionalInputTensor tokenPosIds{std::nullopt};

    // Perform necessary consistent checks to ensure the kernel is launched correctly.
    int64_t const batchSize = qkv.getShape()[0];
    int64_t const headDim = qkv.getShape()[3];
    int64_t const numQHeads = qOut.getShape()[2];
    int64_t const numKVHeads = kvCache.getShape()[2];

    check::check(kvCacheEndLens.getShape()[0] == batchSize && kvCache.getShape()[0] == batchSize
            && qOut.getShape()[0] == batchSize,
        "All Input tensors shall have consistent batch size.");
    check::check(kvCache.getShape()[4] == headDim && qOut.getShape()[3] == headDim,
        "Head dimension shall be consistent between QKV/KVCache/QOut.");
    check::check(qkv.getShape()[2] == numQHeads + numKVHeads * 2, "QKV shall have consistent number of Q/K/V heads.");
    check::check(
        qkv.getShape()[1] == qOut.getShape()[1], "Runtime sequence length shall be consistent between QKV/QOut.");
    check::check(cosSinCache.getShape()[0] == 1 || cosSinCache.getShape()[0] == batchSize,
        "CosSinCache shall have batch size 1 or equal to runtime batch size");

    launchApplyRopeWriteKV(qkv, kvCache, cosSinCache, kvCacheEndLens, tokenPosIds, qOut, stream);
}

void launchApplyRopeWriteKVTreeDecoding(rt::Tensor const& cosSinCache, rt::Tensor const& kvCacheEndLens,
    rt::Tensor const& tokenPosIds, rt::Tensor& qkv, rt::Tensor& kvCache, rt::Tensor& qOut, cudaStream_t stream)
{
    // Special case where we need to perform tree attention for speculative decoding. The mapping between rope positions
    // are no longer consistent with the position to write KVCache. Thus we need the tokenPosIds to indicate the
    // position of token within sequence. Perform necessary consistent checks to ensure kernel launch correctly.
    int64_t const batchSize = qkv.getShape()[0];
    int64_t const headDim = qkv.getShape()[3];
    int64_t const numQHeads = qOut.getShape()[2];
    int64_t const numKVHeads = kvCache.getShape()[2];
    int64_t const runtimeSeqLen = qkv.getShape()[1];

    check::check(kvCacheEndLens.getShape()[0] == batchSize && kvCache.getShape()[0] == batchSize
            && qOut.getShape()[0] == batchSize && tokenPosIds.getShape()[0] == batchSize,
        "All Input tensors shall have consistent batch size.");
    check::check(kvCache.getShape()[4] == headDim && qOut.getShape()[3] == headDim,
        "Head dimension shall be consistent between QKV/KVCache/QOut.");
    check::check(qkv.getShape()[2] == numQHeads + numKVHeads * 2, "QKV shall have consistent number of Q/K/V heads.");
    check::check(qOut.getShape()[1] == runtimeSeqLen && tokenPosIds.getShape()[1] == runtimeSeqLen,
        "QKV/QOut/tokenPosIds shall have consistent sequence length.");
    check::check(cosSinCache.getShape()[0] == 1 || cosSinCache.getShape()[0] == qkv.getShape()[0],
        "CosSinCache shall have batch size 1 or equal to runtime batch size");

    launchApplyRopeWriteKV(qkv, kvCache, cosSinCache, kvCacheEndLens, tokenPosIds, qOut, stream);
}

} // namespace kernel
} // namespace trt_edgellm