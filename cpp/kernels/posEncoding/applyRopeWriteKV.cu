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

#include "applyRopeWriteKV.h"
#include "common/checkMacros.h"
#include "common/cudaMacros.h"
#include "common/pagedKvTypes.h"
#include "kernels/common/vectorizedTypes.cuh"

#include <cstdint>
#include <cuda_fp16.h>
#include <type_traits>

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
    T const* dataPtr, DVec<float> const& cosVec, DVec<float> const& sinVec, uint32_t const rotaryDim)
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

template <typename TCache>
__device__ __forceinline__ void storeVec(TCache* dst, int64_t base, DVec<half> const& vec, float scaleQuantOrig)
{
    if constexpr (std::is_same_v<TCache, half>)
    {
        vec.store(dst + base);
    }
#if SUPPORTS_FP8
    else if constexpr (std::is_same_v<TCache, __nv_fp8_e4m3>)
    {
        DVec<__nv_fp8_e4m3> out;
        assert(scaleQuantOrig > 0.0f);
        float const invScale = 1.0f / scaleQuantOrig;
#pragma unroll
        for (uint32_t i = 0; i < DVec<half>::vec_size; ++i)
        {
            float const scaled = __half2float(vec[i]) * invScale;
            out[i] = __nv_fp8_e4m3(scaled);
        }
        out.store(dst + base);
    }
#endif
}

template <typename T, typename TCache>
__global__ void applyRopeWriteKV(T* q, T* k, T const* v, TCache* kvCache, float const* cosSinCache,
    int32_t const* kvCacheEndLens, int32_t const* tokenPosIds, float kScaleQuantOrig, float vScaleQuantOrig,
    int32_t qSeqLen, int32_t totalNumTokens, int32_t kvCacheCapacity, uint32_t numQHead, uint32_t numKVHead,
    uint32_t headDim, uint32_t rotaryDim, int32_t cosSinCacheBatchSize, int32_t cosSinCacheSeqLen, bool writeKInPlace,
    int32_t const* pageTable, int32_t maxPagesPerSeq)
{
    // Each CTA will process multiple tokens of a single head which each thread handles 16 / sizeof(T) elements.
    // blockDim.x: number of threads to process each token, blockDim.y: number of tokens processed by each CTA.
    // In this kernel we assume:
    //     1. The input tokens are batched with [B, qSeqLen], we use batchIdx info to write KVCache.
    //     2. q, k, v have layout of [B, S, Hq, headDim], [B, S, Hkv, headDim], [B, S, Hv, headDim] where S = qSeqLen.
    //     3. Always write roped q to q in place.
    //     4. Always write KVCache with layout of [B, Hk + Hv, S, headDim] where S = kvCacheCapacityLen.
    //     5. The cosSinCache has layout of [cosSinCacheBatchSize, cosSinCacheSeqLen, rotaryDim] where cosSinCacheSeqLen
    //     >= kvCacheCapacityLen.
    //        cosSinCacheBatchSize can be 1 (all batches share the same cache) or equal to input batch size.
    //     6. kvCacheEndLens: Length of KVCache after insertion the entries by this kernel.
    //     7. writeKInPlace: To handle a special case where we want to run fmha without kv cache directly after this
    //     kernel. Without this, k will only be available through kvcache, which we'll need an additional transpose
    //     step. Note that even if this is false, k will still be written to kvcache.

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
    if (bIdy < numQHead)
    {
        int32_t const qHeadIdx = bIdy;
        int32_t const qOffset = tokenIdx * numQHead * headDim + qHeadIdx * headDim;
        T* qPtr = q + qOffset;
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
            qRoped = vecApplyRopeNonInterleave(qPtr, cosVec, sinVec, rotaryDim);
        }
        qRoped.store(qPtr + DVec<T>::vec_size * tIdx);
    }
    else
    {
        int32_t const kvHeadIdx = bIdy - numQHead;
        int32_t const kvOffset = tokenIdx * numKVHead * headDim + kvHeadIdx * headDim;
        T* kPtr = k + kvOffset;
        T const* vPtr = v + kvOffset;

        int32_t const kvCacheStartIdx = kvCacheEndLens != nullptr ? kvCacheEndLens[batchIdx] - qSeqLen : 0;
        int32_t const tokenIdxInCache = kvCacheStartIdx + tokenIdx % qSeqLen;

        // Load V before writing roped K in-place: when K and V share the same
        // buffer (e.g. Gemma4 global layers where K=V projection), the in-place
        // K write would corrupt V data if read afterwards.
        DVec<T> vSrc;
        vSrc.load(vPtr + DVec<T>::vec_size * tIdx);

        DVec<T> kRoped;
        kRoped = vecApplyRopeNonInterleave(kPtr, cosVec, sinVec, rotaryDim);

        if (writeKInPlace)
        {
            kRoped.store(kPtr + DVec<T>::vec_size * tIdx);
        }

        // Skip writing K/V to cache for padding tokens (position = -1)
        // This ensures padding tokens don't corrupt valid cache entries
        if (!isPaddingToken)
        {
            int32_t const vecBase = DVec<T>::vec_size * tIdx;
            if (pageTable != nullptr)
            {
                // Paged addressing: pageTable[b][0|1][j] carries K/V page ids into the SAME flat
                // pool, reshaped as [nPages, kTOKENS_PER_PAGE, Hkv, D]. A negative entry skips the
                // write for that half.
                int32_t const pageRow = tokenIdxInCache / rt::kTOKENS_PER_PAGE;
                int32_t const inPage = tokenIdxInCache % rt::kTOKENS_PER_PAGE;
                int32_t const kPage = pageTable[(batchIdx * 2 + 0) * maxPagesPerSeq + pageRow];
                int32_t const vPage = pageTable[(batchIdx * 2 + 1) * maxPagesPerSeq + pageRow];
                if (kPage >= 0)
                {
                    int64_t const kOffset
                        = (static_cast<int64_t>(kPage) * rt::kTOKENS_PER_PAGE + inPage) * numKVHead * headDim
                        + static_cast<int64_t>(kvHeadIdx) * headDim + vecBase;
                    storeVec(kvCache, kOffset, kRoped, kScaleQuantOrig);
                }
                // else: unmapped page for a non-padding position -- skip the write (the runtime
                // guarantees mapped coverage for live positions; see the header's bad-page note).
                if (vPage >= 0)
                {
                    int64_t const vOffset
                        = (static_cast<int64_t>(vPage) * rt::kTOKENS_PER_PAGE + inPage) * numKVHead * headDim
                        + static_cast<int64_t>(kvHeadIdx) * headDim + vecBase;
                    storeVec(kvCache, vOffset, vSrc, vScaleQuantOrig);
                }
            }
            else
            {
                // Legacy KVCache with layout [B, Hk + Hv, S, D].
                int64_t const cacheOffsetSequence
                    = static_cast<int64_t>(batchIdx) * 2 * numKVHead * kvCacheCapacity * headDim;
                int64_t const cacheOffsetK = cacheOffsetSequence
                    + static_cast<int64_t>(kvHeadIdx) * kvCacheCapacity * headDim + tokenIdxInCache * headDim + vecBase;
                int64_t const cacheOffsetV = cacheOffsetSequence
                    + static_cast<int64_t>(numKVHead + kvHeadIdx) * kvCacheCapacity * headDim
                    + tokenIdxInCache * headDim + vecBase;
                storeVec(kvCache, cacheOffsetK, kRoped, kScaleQuantOrig);
                storeVec(kvCache, cacheOffsetV, vSrc, vScaleQuantOrig);
            }
        }
    }
}

static void launchApplyRopeWriteKVKernel(rt::Tensor& q, rt::Tensor& k, rt::Tensor const& v, rt::Tensor& kvCache,
    rt::Tensor const& cosSinCache, rt::OptionalInputTensor kvCacheEndLens, rt::OptionalInputTensor tokenPosIds,
    float kScale, float vScale, cudaStream_t stream, bool writeKInPlace, int32_t const* pageTable,
    int32_t maxPagesPerSeq)
{
    auto const dt = kvCache.getDataType();
    constexpr uint32_t kVEC_SIZE = DVec<half>::vec_size;
    constexpr uint32_t kTHREADS_PER_CTA = 128;

    uint32_t const runtimeBatchSize = static_cast<uint32_t>(q.getShape()[0]);
    uint32_t const runtimeSeqLen = static_cast<uint32_t>(q.getShape()[1]);
    uint32_t const numQHeads = static_cast<uint32_t>(q.getShape()[2]);
    uint32_t const headDim = static_cast<uint32_t>(q.getShape()[3]);
    uint32_t const numKVHeads = static_cast<uint32_t>(kvCache.getShape()[2]);
    uint32_t const kvCacheCapacity = static_cast<uint32_t>(kvCache.getShape()[3]);
    uint32_t const totalNumTokens = runtimeBatchSize * runtimeSeqLen;

    uint32_t const cosSinCacheBatchSize = static_cast<uint32_t>(cosSinCache.getShape()[0]);
    uint32_t const cosSinCacheSeqLen = static_cast<uint32_t>(cosSinCache.getShape()[1]);
    uint32_t const rotaryDim = static_cast<uint32_t>(cosSinCache.getShape()[2]);

    half* qPtr = q.dataPointer<half>();
    half* kPtr = k.dataPointer<half>();
    half const* vPtr = v.dataPointer<half>();
    float const* cosSinCachePtr = cosSinCache.dataPointer<float>();

    int32_t const* kvCacheEndLensPtr
        = kvCacheEndLens.has_value() ? kvCacheEndLens.value().get().dataPointer<int32_t>() : nullptr;
    int32_t const* tokenPosIdsPtr
        = tokenPosIds.has_value() ? tokenPosIds.value().get().dataPointer<int32_t>() : nullptr;

    uint32_t const tokenPerCTA = kTHREADS_PER_CTA * kVEC_SIZE / headDim;
    uint32_t const bDimX = headDim / kVEC_SIZE;
    uint32_t const bDimY = tokenPerCTA;
    uint32_t const gDimX = (totalNumTokens + tokenPerCTA - 1) / tokenPerCTA;
    uint32_t const gDimY = numQHeads + numKVHeads;

    dim3 grid(gDimX, gDimY);
    dim3 block(bDimX, bDimY);

    if (dt == nvinfer1::DataType::kHALF)
    {
        half* kvCachePtr = kvCache.dataPointer<half>();
        applyRopeWriteKV<half, half><<<grid, block, 0, stream>>>(qPtr, kPtr, vPtr, kvCachePtr, cosSinCachePtr,
            kvCacheEndLensPtr, tokenPosIdsPtr, kScale, vScale, runtimeSeqLen, totalNumTokens, kvCacheCapacity,
            numQHeads, numKVHeads, headDim, rotaryDim, cosSinCacheBatchSize, cosSinCacheSeqLen, writeKInPlace,
            pageTable, maxPagesPerSeq);
    }
#if SUPPORTS_FP8
    else if (dt == nvinfer1::DataType::kFP8)
    {
        __nv_fp8_e4m3* kvCachePtr = kvCache.dataPointer<__nv_fp8_e4m3>();
        applyRopeWriteKV<half, __nv_fp8_e4m3><<<grid, block, 0, stream>>>(qPtr, kPtr, vPtr, kvCachePtr, cosSinCachePtr,
            kvCacheEndLensPtr, tokenPosIdsPtr, kScale, vScale, runtimeSeqLen, totalNumTokens, kvCacheCapacity,
            numQHeads, numKVHeads, headDim, rotaryDim, cosSinCacheBatchSize, cosSinCacheSeqLen, writeKInPlace,
            pageTable, maxPagesPerSeq);
    }
#endif
    else
    {
        check::check(false, "Unsupported KV cache dtype");
    }
}

void launchApplyRopeWriteKV(rt::Tensor const& cosSinCache, rt::OptionalInputTensor kvCacheEndLens, rt::Tensor& q,
    rt::Tensor& k, rt::Tensor const& v, rt::Tensor& kvCache, float kScale, float vScale, cudaStream_t stream,
    bool writeKInPlace, int32_t const* pageTable, int32_t maxPagesPerSeq)
{
    rt::OptionalInputTensor tokenPosIds{std::nullopt};

    int64_t const batchSize = q.getShape()[0];
    int64_t const headDim = q.getShape()[3];
    int64_t const numKVHeads = k.getShape()[2];

    check::check(k.getShape()[0] == batchSize && v.getShape()[0] == batchSize && kvCache.getShape()[0] == batchSize,
        "Q/K/V and KVCache shall have the same batch size");
    check::check(k.getShape()[3] == headDim && v.getShape()[3] == headDim && kvCache.getShape()[4] == headDim,
        "Head dimension shall be consistent between Q/K/V/KVCache.");
    check::check(cosSinCache.getShape()[0] == 1 || cosSinCache.getShape()[0] == batchSize,
        "CosSinCache shall have batch size 1 or equal to runtime batch size");

    if (kvCacheEndLens.has_value())
    {
        check::check(kvCacheEndLens.value().get().getShape()[0] == batchSize,
            "kvCacheEndLens shall have consistent batch size.");
        check::check(kvCache.getShape()[2] == numKVHeads, "KVCache shall have consistent number of K/V heads.");
    }

    launchApplyRopeWriteKVKernel(q, k, v, kvCache, cosSinCache, kvCacheEndLens, tokenPosIds, kScale, vScale, stream,
        writeKInPlace, pageTable, maxPagesPerSeq);
}

void launchApplyRopeWriteKVTreeDecoding(rt::Tensor const& cosSinCache, rt::Tensor const& kvCacheEndLens,
    rt::Tensor const& tokenPosIds, rt::Tensor& q, rt::Tensor& k, rt::Tensor const& v, rt::Tensor& kvCache, float kScale,
    float vScale, cudaStream_t stream, int32_t const* pageTable, int32_t maxPagesPerSeq)
{
    int64_t const batchSize = q.getShape()[0];
    int64_t const headDim = q.getShape()[3];
    int64_t const numQHeads = q.getShape()[2];
    int64_t const numKVHeads = k.getShape()[2];
    int64_t const runtimeSeqLen = q.getShape()[1];

    check::check(k.getShape()[0] == batchSize && v.getShape()[0] == batchSize
            && kvCacheEndLens.getShape()[0] == batchSize && kvCache.getShape()[0] == batchSize
            && tokenPosIds.getShape()[0] == batchSize,
        "All Input tensors shall have consistent batch size.");
    check::check(k.getShape()[3] == headDim && v.getShape()[3] == headDim && kvCache.getShape()[4] == headDim,
        "Head dimension shall be consistent between Q/K/V/KVCache.");
    check::check(v.getShape()[2] == numKVHeads && kvCache.getShape()[2] == numKVHeads,
        "K/V/KVCache shall have consistent number of heads.");
    check::check(tokenPosIds.getShape()[1] == runtimeSeqLen, "Q/tokenPosIds shall have consistent sequence length.");
    check::check(cosSinCache.getShape()[0] == 1 || cosSinCache.getShape()[0] == batchSize,
        "CosSinCache shall have batch size 1 or equal to runtime batch size");

    launchApplyRopeWriteKVKernel(q, k, v, kvCache, cosSinCache, kvCacheEndLens, tokenPosIds, kScale, vScale, stream,
        false, pageTable, maxPagesPerSeq);
}

// =============================================================================
// Optimized kernel for CuTe DSL FMHA path: RoPE Q in-place + write K/V to cache
// =============================================================================

template <typename T, typename TCache>
__global__ void applyRopeWriteKVSplitQKVKernel(T* __restrict__ q, T const* __restrict__ k, T const* __restrict__ v,
    TCache* __restrict__ kvCache, void* __restrict__ fp8QOut, float const* __restrict__ cosSinCache,
    int32_t const* __restrict__ kvCacheEndLens, float qScaleQuantOrig, float kScaleQuantOrig, float vScaleQuantOrig,
    int32_t qSeqLen, int32_t totalNumTokens, int32_t kvCacheCapacity, uint32_t numQHead, uint32_t numKVHead,
    uint32_t headDim, uint32_t rotaryDim, int32_t cosSinCacheBatchSize, int32_t cosSinCacheSeqLen,
    int32_t const* __restrict__ pageTable, int32_t maxPagesPerSeq)
{
    // Thread mapping (same as existing kernel for proven memory coalescing):
    //   blockDim.x = headDim / vec_size  (threads per token)
    //   blockDim.y = tokens per CTA
    //   gridDim.x  = ceil(totalNumTokens / blockDim.y)
    //   gridDim.y  = numQHead + numKVHead

    uint32_t const tIdx = threadIdx.x;
    uint32_t const tIdy = threadIdx.y;
    uint32_t const tokenIdx = blockIdx.x * blockDim.y + tIdy;

    if (tokenIdx >= totalNumTokens)
    {
        return;
    }

    int32_t const batchIdx = tokenIdx / qSeqLen;

    // Compute RoPE position: kvCacheEndLens[b] - qSeqLen + token_offset_in_seq
    int32_t const posStartId = kvCacheEndLens[batchIdx] - qSeqLen;
    int32_t const sinCosCachePos = posStartId + tokenIdx % qSeqLen;

    // Vectorized load cos/sin (non-interleaved RoPE)
    uint32_t const sinOffset = rotaryDim / 2;
    uint32_t const cosOffset = (tIdx * DVec<float>::vec_size) % (rotaryDim / 2);
    int32_t const cosSinCacheBatchIdx = (cosSinCacheBatchSize == 1) ? 0 : batchIdx;
    int32_t const cosSinCacheOffset = cosSinCacheBatchIdx * cosSinCacheSeqLen * rotaryDim + sinCosCachePos * rotaryDim;
    DVec<float> cosVec;
    DVec<float> sinVec;
    cosVec.load(cosSinCache + cosSinCacheOffset + cosOffset);
    sinVec.load(cosSinCache + cosSinCacheOffset + cosOffset + sinOffset);

    uint32_t const headIdx = blockIdx.y;

    if (headIdx < numQHead)
    {
        // --- Q head: apply RoPE, write FP16 in-place or FP8 to separate buffer ---
        int32_t const qOffset = tokenIdx * numQHead * headDim + headIdx * headDim;
        T* qPtr = q + qOffset;
        DVec<T> qRoped = vecApplyRopeNonInterleave(qPtr, cosVec, sinVec, rotaryDim);
#if SUPPORTS_FP8
        if (fp8QOut != nullptr)
        {
            storeVec(reinterpret_cast<__nv_fp8_e4m3*>(fp8QOut), qOffset + static_cast<int>(DVec<T>::vec_size * tIdx),
                qRoped, qScaleQuantOrig);
        }
        else
#endif
        {
            qRoped.store(qPtr + DVec<T>::vec_size * tIdx);
        }
    }
    else
    {
        // --- KV head: apply RoPE to K, write K and V to cache ---
        uint32_t const kvHeadIdx = headIdx - numQHead;
        int32_t const kvInputOffset = tokenIdx * numKVHead * headDim + kvHeadIdx * headDim;
        T const* kPtr = k + kvInputOffset;
        T const* vPtr = v + kvInputOffset;

        // Apply RoPE to K
        DVec<T> kRoped = vecApplyRopeNonInterleave(kPtr, cosVec, sinVec, rotaryDim);

        // Load V
        DVec<T> vSrc;
        vSrc.load(vPtr + DVec<T>::vec_size * tIdx);

        int32_t const tokenIdxInCache = kvCacheEndLens[batchIdx] - qSeqLen + tokenIdx % qSeqLen;
        int32_t const vecBase = DVec<T>::vec_size * tIdx;
        if (pageTable != nullptr)
        {
            // Paged addressing: pageTable[b][0|1][j] carries K/V page ids into the SAME flat pool,
            // reshaped as [nPages, kTOKENS_PER_PAGE, Hkv, D]. A negative entry skips the write for
            // that half.
            int32_t const pageRow = tokenIdxInCache / rt::kTOKENS_PER_PAGE;
            int32_t const inPage = tokenIdxInCache % rt::kTOKENS_PER_PAGE;
            int32_t const kPage = pageTable[(batchIdx * 2 + 0) * maxPagesPerSeq + pageRow];
            int32_t const vPage = pageTable[(batchIdx * 2 + 1) * maxPagesPerSeq + pageRow];
            if (kPage >= 0)
            {
                int64_t const kOffset
                    = (static_cast<int64_t>(kPage) * rt::kTOKENS_PER_PAGE + inPage) * numKVHead * headDim
                    + static_cast<int64_t>(kvHeadIdx) * headDim + vecBase;
                storeVec(kvCache, kOffset, kRoped, kScaleQuantOrig);
            }
            // else: unmapped page -- skip the write (runtime guarantees mapped coverage for
            // live positions; see the header's bad-page note).
            if (vPage >= 0)
            {
                int64_t const vOffset
                    = (static_cast<int64_t>(vPage) * rt::kTOKENS_PER_PAGE + inPage) * numKVHead * headDim
                    + static_cast<int64_t>(kvHeadIdx) * headDim + vecBase;
                storeVec(kvCache, vOffset, vSrc, vScaleQuantOrig);
            }
        }
        else
        {
            // Legacy KV cache layout: [B, 2, H_kv, S, D]
            //   K at [b, 0, h, s, :] = b*2*H*S*D + 0*H*S*D + h*S*D + s*D
            //   V at [b, 1, h, s, :] = b*2*H*S*D + 1*H*S*D + h*S*D + s*D
            int64_t const cacheBase = static_cast<int64_t>(batchIdx) * 2 * numKVHead * kvCacheCapacity * headDim;
            int64_t const cacheOffsetK = cacheBase + static_cast<int64_t>(kvHeadIdx) * kvCacheCapacity * headDim
                + tokenIdxInCache * headDim + vecBase;
            int64_t const cacheOffsetV = cacheBase
                + static_cast<int64_t>(numKVHead + kvHeadIdx) * kvCacheCapacity * headDim + tokenIdxInCache * headDim
                + vecBase;
            storeVec(kvCache, cacheOffsetK, kRoped, kScaleQuantOrig);
            storeVec(kvCache, cacheOffsetV, vSrc, vScaleQuantOrig);
        }
    }
}

void launchApplyRopeWriteKVSplitQKV(rt::Tensor const& cosSinCache, rt::Tensor const& kvCacheEndLens, rt::Tensor& q,
    rt::Tensor const& k, rt::Tensor const& v, rt::Tensor& kvCache, float kScale, float vScale, cudaStream_t stream,
    int32_t const* pageTable, int32_t maxPagesPerSeq, void* fp8QOut, float qScale)
{
    auto const dt = kvCache.getDataType();

    constexpr uint32_t kVEC_SIZE = DVec<half>::vec_size;
    constexpr uint32_t kTHREADS_PER_CTA = 128;

    uint32_t const runtimeBatchSize = static_cast<uint32_t>(q.getShape()[0]);
    uint32_t const runtimeSeqLen = static_cast<uint32_t>(q.getShape()[1]);
    uint32_t const numQHeads = static_cast<uint32_t>(q.getShape()[2]);
    uint32_t const headDim = static_cast<uint32_t>(q.getShape()[3]);
    uint32_t const numKVHeads = static_cast<uint32_t>(kvCache.getShape()[2]);
    uint32_t const kvCacheCapacity = static_cast<uint32_t>(kvCache.getShape()[3]);
    uint32_t const totalNumTokens = runtimeBatchSize * runtimeSeqLen;

    uint32_t const cosSinCacheBatchSize = static_cast<uint32_t>(cosSinCache.getShape()[0]);
    uint32_t const cosSinCacheSeqLen = static_cast<uint32_t>(cosSinCache.getShape()[1]);
    uint32_t const rotaryDim = static_cast<uint32_t>(cosSinCache.getShape()[2]);

    half* qPtr = q.dataPointer<half>();
    half const* kPtr = k.dataPointer<half>();
    half const* vPtr = v.dataPointer<half>();
    float const* cosSinCachePtr = cosSinCache.dataPointer<float>();
    int32_t const* kvCacheEndLensPtr = kvCacheEndLens.dataPointer<int32_t>();

    uint32_t const tokenPerCTA = kTHREADS_PER_CTA * kVEC_SIZE / headDim;
    uint32_t const bDimX = headDim / kVEC_SIZE;
    uint32_t const bDimY = tokenPerCTA;
    uint32_t const gDimX = (totalNumTokens + tokenPerCTA - 1) / tokenPerCTA;
    uint32_t const gDimY = numQHeads + numKVHeads;

    dim3 grid(gDimX, gDimY);
    dim3 block(bDimX, bDimY);

    if (dt == nvinfer1::DataType::kHALF)
    {
        half* kvCachePtr = kvCache.dataPointer<half>();
        applyRopeWriteKVSplitQKVKernel<half, half><<<grid, block, 0, stream>>>(qPtr, kPtr, vPtr, kvCachePtr, nullptr,
            cosSinCachePtr, kvCacheEndLensPtr, 1.0f, kScale, vScale, runtimeSeqLen, totalNumTokens, kvCacheCapacity,
            numQHeads, numKVHeads, headDim, rotaryDim, cosSinCacheBatchSize, cosSinCacheSeqLen, pageTable,
            maxPagesPerSeq);
    }
#if SUPPORTS_FP8
    else if (dt == nvinfer1::DataType::kFP8)
    {
        __nv_fp8_e4m3* kvCachePtr = kvCache.dataPointer<__nv_fp8_e4m3>();
        applyRopeWriteKVSplitQKVKernel<half, __nv_fp8_e4m3><<<grid, block, 0, stream>>>(qPtr, kPtr, vPtr, kvCachePtr,
            fp8QOut, cosSinCachePtr, kvCacheEndLensPtr, qScale, kScale, vScale, runtimeSeqLen, totalNumTokens,
            kvCacheCapacity, numQHeads, numKVHeads, headDim, rotaryDim, cosSinCacheBatchSize, cosSinCacheSeqLen,
            pageTable, maxPagesPerSeq);
    }
#endif
    else
    {
        check::check(false, "Unsupported KV cache dtype");
    }
}

// =============================================================================
// Q-only RoPE kernel for shared-KV layers (no KV cache write)
// =============================================================================

template <typename T>
__global__ void applyRopeQOnlyKernel(T* __restrict__ q, float const* __restrict__ cosSinCache,
    int32_t const* __restrict__ kvCacheEndLens, int32_t qSeqLen, int32_t totalNumTokens, uint32_t numQHead,
    uint32_t headDim, uint32_t rotaryDim, int32_t cosSinCacheBatchSize, int32_t cosSinCacheSeqLen)
{
    // Grid: (ceil(totalTokens / tokensPerCTA), numQHead)
    // Block: (headDim / vec_size, tokensPerCTA)
    uint32_t const tIdx = threadIdx.x;
    uint32_t const tIdy = threadIdx.y;
    uint32_t const tokenIdx = blockIdx.x * blockDim.y + tIdy;

    if (tokenIdx >= totalNumTokens)
    {
        return;
    }

    int32_t const batchIdx = tokenIdx / qSeqLen;
    int32_t const posStartId = kvCacheEndLens[batchIdx] - qSeqLen;
    int32_t const sinCosCachePos = posStartId + tokenIdx % qSeqLen;

    // Load cos/sin
    uint32_t const sinOffset = rotaryDim / 2;
    uint32_t const cosOffset = (tIdx * DVec<float>::vec_size) % (rotaryDim / 2);
    int32_t const cosSinCacheBatchIdx = (cosSinCacheBatchSize == 1) ? 0 : batchIdx;
    int32_t const cosSinCacheOffset = cosSinCacheBatchIdx * cosSinCacheSeqLen * rotaryDim + sinCosCachePos * rotaryDim;
    DVec<float> cosVec;
    DVec<float> sinVec;
    cosVec.load(cosSinCache + cosSinCacheOffset + cosOffset);
    sinVec.load(cosSinCache + cosSinCacheOffset + cosOffset + sinOffset);

    // Apply RoPE to Q head
    uint32_t const qHeadIdx = blockIdx.y;
    int32_t const qOffset = tokenIdx * numQHead * headDim + qHeadIdx * headDim;
    T* qPtr = q + qOffset;
    DVec<T> qRoped = vecApplyRopeNonInterleave(qPtr, cosVec, sinVec, rotaryDim);
    qRoped.store(qPtr + DVec<T>::vec_size * tIdx);
}

void launchApplyRopeQOnly(
    rt::Tensor const& cosSinCache, rt::Tensor const& kvCacheEndLens, rt::Tensor& q, cudaStream_t stream)
{
    constexpr uint32_t kVEC_SIZE = DVec<half>::vec_size;
    constexpr uint32_t kTHREADS_PER_CTA = 128;

    uint32_t const runtimeBatchSize = static_cast<uint32_t>(q.getShape()[0]);
    uint32_t const runtimeSeqLen = static_cast<uint32_t>(q.getShape()[1]);
    uint32_t const numQHeads = static_cast<uint32_t>(q.getShape()[2]);
    uint32_t const headDim = static_cast<uint32_t>(q.getShape()[3]);
    uint32_t const totalNumTokens = runtimeBatchSize * runtimeSeqLen;

    uint32_t const cosSinCacheBatchSize = static_cast<uint32_t>(cosSinCache.getShape()[0]);
    uint32_t const cosSinCacheSeqLen = static_cast<uint32_t>(cosSinCache.getShape()[1]);
    uint32_t const rotaryDim = static_cast<uint32_t>(cosSinCache.getShape()[2]);

    half* qPtr = q.dataPointer<half>();
    float const* cosSinCachePtr = cosSinCache.dataPointer<float>();
    int32_t const* kvCacheEndLensPtr = kvCacheEndLens.dataPointer<int32_t>();

    uint32_t const tokenPerCTA = kTHREADS_PER_CTA * kVEC_SIZE / headDim;
    uint32_t const bDimX = headDim / kVEC_SIZE;
    uint32_t const bDimY = tokenPerCTA;
    uint32_t const gDimX = (totalNumTokens + tokenPerCTA - 1) / tokenPerCTA;
    uint32_t const gDimY = numQHeads; // Q heads only — no KV threads

    dim3 grid(gDimX, gDimY);
    dim3 block(bDimX, bDimY);

    applyRopeQOnlyKernel<half><<<grid, block, 0, stream>>>(qPtr, cosSinCachePtr, kvCacheEndLensPtr, runtimeSeqLen,
        totalNumTokens, numQHeads, headDim, rotaryDim, cosSinCacheBatchSize, cosSinCacheSeqLen);
}

// =============================================================================
// Q-only RoPE kernel for shared-KV layers with tree decoding (per-token position IDs)
// =============================================================================

template <typename T>
__global__ void applyRopeQOnlyTreeDecodingKernel(T* __restrict__ q, float const* __restrict__ cosSinCache,
    int32_t const* __restrict__ tokenPosIds, int32_t qSeqLen, int32_t totalNumTokens, uint32_t numQHead,
    uint32_t headDim, uint32_t rotaryDim, int32_t cosSinCacheBatchSize, int32_t cosSinCacheSeqLen)
{
    // Grid: (ceil(totalTokens / tokensPerCTA), numQHead)
    // Block: (headDim / vec_size, tokensPerCTA)
    uint32_t const tIdx = threadIdx.x;
    uint32_t const tIdy = threadIdx.y;
    uint32_t const tokenIdx = blockIdx.x * blockDim.y + tIdy;

    if (tokenIdx >= totalNumTokens)
    {
        return;
    }

    int32_t const batchIdx = tokenIdx / qSeqLen;
    int32_t const sinCosCachePos = tokenPosIds[tokenIdx];

    // Load cos/sin
    uint32_t const sinOffset = rotaryDim / 2;
    uint32_t const cosOffset = (tIdx * DVec<float>::vec_size) % (rotaryDim / 2);
    int32_t const cosSinCacheBatchIdx = (cosSinCacheBatchSize == 1) ? 0 : batchIdx;
    int32_t const cosSinCacheOffset = cosSinCacheBatchIdx * cosSinCacheSeqLen * rotaryDim + sinCosCachePos * rotaryDim;
    DVec<float> cosVec;
    DVec<float> sinVec;
    cosVec.load(cosSinCache + cosSinCacheOffset + cosOffset);
    sinVec.load(cosSinCache + cosSinCacheOffset + cosOffset + sinOffset);

    // Apply RoPE to Q head
    uint32_t const qHeadIdx = blockIdx.y;
    int32_t const qOffset = tokenIdx * numQHead * headDim + qHeadIdx * headDim;
    T* qPtr = q + qOffset;
    DVec<T> qRoped = vecApplyRopeNonInterleave(qPtr, cosVec, sinVec, rotaryDim);
    qRoped.store(qPtr + DVec<T>::vec_size * tIdx);
}

void launchApplyRopeQOnlyTreeDecoding(
    rt::Tensor const& cosSinCache, rt::Tensor const& tokenPosIds, rt::Tensor& q, cudaStream_t stream)
{
    constexpr uint32_t kVEC_SIZE = DVec<half>::vec_size;
    constexpr uint32_t kTHREADS_PER_CTA = 128;

    uint32_t const runtimeBatchSize = static_cast<uint32_t>(q.getShape()[0]);
    uint32_t const runtimeSeqLen = static_cast<uint32_t>(q.getShape()[1]);
    uint32_t const numQHeads = static_cast<uint32_t>(q.getShape()[2]);
    uint32_t const headDim = static_cast<uint32_t>(q.getShape()[3]);
    uint32_t const totalNumTokens = runtimeBatchSize * runtimeSeqLen;

    uint32_t const cosSinCacheBatchSize = static_cast<uint32_t>(cosSinCache.getShape()[0]);
    uint32_t const cosSinCacheSeqLen = static_cast<uint32_t>(cosSinCache.getShape()[1]);
    uint32_t const rotaryDim = static_cast<uint32_t>(cosSinCache.getShape()[2]);

    check::check(tokenPosIds.getShape()[0] == runtimeBatchSize && tokenPosIds.getShape()[1] == runtimeSeqLen,
        "tokenPosIds shall have shape [B, S] matching Q.");

    half* qPtr = q.dataPointer<half>();
    float const* cosSinCachePtr = cosSinCache.dataPointer<float>();
    int32_t const* tokenPosIdsPtr = tokenPosIds.dataPointer<int32_t>();

    uint32_t const tokenPerCTA = kTHREADS_PER_CTA * kVEC_SIZE / headDim;
    uint32_t const bDimX = headDim / kVEC_SIZE;
    uint32_t const bDimY = tokenPerCTA;
    uint32_t const gDimX = (totalNumTokens + tokenPerCTA - 1) / tokenPerCTA;
    uint32_t const gDimY = numQHeads; // Q heads only — no KV threads

    dim3 grid(gDimX, gDimY);
    dim3 block(bDimX, bDimY);

    applyRopeQOnlyTreeDecodingKernel<half><<<grid, block, 0, stream>>>(qPtr, cosSinCachePtr, tokenPosIdsPtr,
        runtimeSeqLen, totalNumTokens, numQHeads, headDim, rotaryDim, cosSinCacheBatchSize, cosSinCacheSeqLen);
}

} // namespace kernel
} // namespace trt_edgellm
