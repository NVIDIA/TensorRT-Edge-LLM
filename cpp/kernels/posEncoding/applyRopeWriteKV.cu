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

//! Round @p x up to the next power of two — pads lanesPerHead so the warp-shuffle
//! butterfly reduction is well-defined for any headDim.
__host__ __device__ __forceinline__ constexpr uint32_t nextPowerOf2(uint32_t x)
{
    if (x <= 1)
    {
        return 1;
    }
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return x + 1;
}

//! Warp-shuffle butterfly reduction over @p paddedLanesPerHead lanes (power of two <= 32)
//! cooperating on one head. Ghost lanes must pass partial = 0 but still participate so the
//! warp converges. Returns the headDim sum-of-squares replicated to every lane.
__device__ __forceinline__ float warpReduceHeadSumOfSquares(float partial, uint32_t const paddedLanesPerHead)
{
    // Defensive check on debug builds: butterfly requires a power-of-two participating-lane count.
    assert((paddedLanesPerHead & (paddedLanesPerHead - 1)) == 0 && paddedLanesPerHead <= 32u);
    // Explicit __syncwarp() around the shfl sequence for robustness under Volta+
    // independent thread scheduling.
    __syncwarp();
    for (uint32_t offset = paddedLanesPerHead / 2; offset > 0; offset >>= 1)
    {
        partial += __shfl_xor_sync(0xffffffff, partial, offset);
    }
    __syncwarp();
    return partial;
}

//! Load this thread's slice of one head, compute the head-wide RMSNorm scale via warp
//! shuffle, multiply by gamma, and return the scaled slice (caller follows up with RoPE).
//! invRmsOut returns 1/sqrt(mean(x^2)+eps). Ghost lanes (isActiveLane == false) contribute
//! partial = 0, skip the (OOB) gamma/input loads, and return an uninitialized DVec — the
//! caller must gate stores on isActiveLane.
template <typename T>
__device__ __forceinline__ DVec<T> loadAndApplyRmsNorm(T const* dataPtr, T const* gammaPtr, uint32_t const headDim,
    uint32_t const paddedLanesPerHead, float const rmsEps, float& invRmsOut, bool const isActiveLane)
{
    uint32_t const vecOffset = threadIdx.x * DVec<T>::vec_size;
    DVec<T> input;
    DVec<T> gamma;
    float partial = 0.f;

    // Replicates the compiler backend's standalone-RMSNorm arithmetic so the fused path
    // stays bit-identical to the unfused graph; the _rn intrinsics keep each op correctly
    // rounded under --use_fast_math. Do not fold 1/sqrt back into rsqrtf (approximate).
    // The FMA accumulation is exact: a half squared fits fp32, so only the add rounds.
    if (isActiveLane)
    {
        input.load(dataPtr + vecOffset);
        gamma.load(gammaPtr + vecOffset);

#pragma unroll
        for (uint32_t i = 0; i < DVec<T>::vec_size; ++i)
        {
            float const v = __half2float(input[i]);
            partial = __fmaf_rn(v, v, partial);
        }
    }
    // Ghost lanes: partial stays 0; still participate in the warp-wide reduction below.

    float const headSumSq = warpReduceHeadSumOfSquares(partial, paddedLanesPerHead);
    float const meanSq = __fdiv_rn(headSumSq, static_cast<float>(headDim));
    invRmsOut = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(meanSq, rmsEps)));

    if (isActiveLane)
    {
#pragma unroll
        for (uint32_t i = 0; i < DVec<T>::vec_size; ++i)
        {
            input[i] = __hmul(__float2half(__fmul_rn(__half2float(input[i]), invRmsOut)), gamma[i]);
        }
    }
    return input;
}

//! Fused RMSNorm + non-interleaved RoPE on one head. The local and the RoPE-permuted slices
//! are normalized by the same invRms, each with its own gamma slice. Ghost lanes participate
//! only in the reduction and return data the caller must NOT store.
template <typename T>
__device__ __forceinline__ DVec<T> vecApplyRmsNormAndRopeNonInterleave(T const* dataPtr, T const* gammaPtr,
    DVec<float> const& cosVec, DVec<float> const& sinVec, uint32_t const rotaryDim, uint32_t const headDim,
    uint32_t const paddedLanesPerHead, float const rmsEps)
{
    uint32_t const vecOffset = threadIdx.x * DVec<T>::vec_size;
    bool const isActiveLane = (vecOffset < headDim);

    // Compute invRms (shared by both halves of the rotary permute), get the normed local slice.
    // All lanes (including ghosts) must call this — it contains the warp-wide butterfly shfl.
    float invRms;
    DVec<T> input = loadAndApplyRmsNorm(dataPtr, gammaPtr, headDim, paddedLanesPerHead, rmsEps, invRms, isActiveLane);

    // Fetch the RoPE permute-partner's already-normed slice:
    //   (A) shfl-partner (fast): full rotation + power-of-2 lanes — the partner is lane
    //       (X XOR actualLanes/2); grab its register via __shfl_xor_sync.
    //   (B) gmem reload (fallback): otherwise re-load the partner slice and re-norm.
    uint32_t const actualLanesPerHead = headDim / DVec<T>::vec_size;
    bool const canShflPartner = (rotaryDim == headDim) && ((actualLanesPerHead & (actualLanesPerHead - 1)) == 0);
    DVec<T> permuteInput;

    if (canShflPartner)
    {
        // All lanes participate uniformly here (canShflPartner forbids ghost lanes).
        // Pack two 16-bit elements per uint32 shfl word to halve the shfl count.
        static_assert(sizeof(T) == 2, "shfl-partner pack-into-uint32 assumes 16-bit element type");
        int const pairOffset = static_cast<int>(actualLanesPerHead / 2);
        __syncwarp();
#pragma unroll
        for (uint32_t i = 0; i < DVec<T>::vec_size; i += 2)
        {
            uint32_t packed;
            T* const packedAsT = reinterpret_cast<T*>(&packed);
            packedAsT[0] = input[i];
            packedAsT[1] = input[i + 1];
            uint32_t const shuffled = __shfl_xor_sync(0xffffffff, packed, pairOffset);
            T const* const shuffledAsT = reinterpret_cast<T const*>(&shuffled);
            permuteInput[i] = shuffledAsT[0];
            permuteInput[i + 1] = shuffledAsT[1];
        }
        __syncwarp();
    }
    else
    {
        // Strategy (B): gmem reload fallback. Early return for ghost/tail lanes — they
        // don't own real elements to RoPE.
        if (!isActiveLane || vecOffset >= rotaryDim)
        {
            return input;
        }
        uint32_t const permuteOffset
            = (vecOffset < rotaryDim / 2) ? vecOffset + rotaryDim / 2 : vecOffset - rotaryDim / 2;
        DVec<T> permuteGamma;
        permuteInput.load(dataPtr + permuteOffset);
        permuteGamma.load(gammaPtr + permuteOffset);
#pragma unroll
        for (uint32_t i = 0; i < DVec<T>::vec_size; ++i)
        {
            // Same backend-matching order as loadAndApplyRmsNorm: fp32 scale, cast to
            // half, then half-precision gamma multiply.
            permuteInput[i] = __hmul(__float2half(__fmul_rn(__half2float(permuteInput[i]), invRms)), permuteGamma[i]);
        }
    }

    DVec<T> result;
#pragma unroll
    for (uint32_t i = 0; i < DVec<T>::vec_size; ++i)
    {
        result[i] = applyRope(input[i], permuteInput[i], cosVec[i], sinVec[i], (vecOffset < rotaryDim / 2));
    }
    return result;
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
    int32_t qSeqLen, int32_t totalNumTokens, int32_t numPages, uint32_t numQHead, uint32_t numKVHead, uint32_t headDim,
    uint32_t rotaryDim, int32_t cosSinCacheBatchSize, int32_t cosSinCacheSeqLen, bool writeKInPlace,
    int32_t const* pageTable, int32_t maxPagesPerSeq)
{
    // Each CTA will process multiple tokens of a single head which each thread handles 16 / sizeof(T) elements.
    // blockDim.x: number of threads to process each token, blockDim.y: number of tokens processed by each CTA.
    // In this kernel we assume:
    //     1. The input tokens are batched with [B, qSeqLen], we use batchIdx info to write KVCache.
    //     2. q, k, v have layout of [B, S, Hq, headDim], [B, S, Hkv, headDim], [B, S, Hv, headDim] where S = qSeqLen.
    //     3. Always write roped q to q in place.
    //     4. Always write KVCache with layout of [2, numPages, kTOKENS_PER_PAGE, Hkv, headDim].
    //     5. The cosSinCache has layout of [cosSinCacheBatchSize, cosSinCacheSeqLen, rotaryDim] and covers all
    //     token positions addressed by this launch.
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
            int32_t const pageRow = tokenIdxInCache / rt::kTOKENS_PER_PAGE;
            int32_t const inPage = tokenIdxInCache % rt::kTOKENS_PER_PAGE;
            int32_t const kPage = pageTable[(batchIdx * 2 + 0) * maxPagesPerSeq + pageRow];
            int32_t const vPage = pageTable[(batchIdx * 2 + 1) * maxPagesPerSeq + pageRow];
            if (kPage >= 0 && kPage < numPages)
            {
                int64_t const kOffset
                    = (static_cast<int64_t>(kPage) * rt::kTOKENS_PER_PAGE + inPage) * numKVHead * headDim
                    + static_cast<int64_t>(kvHeadIdx) * headDim + vecBase;
                storeVec(kvCache, kOffset, kRoped, kScaleQuantOrig);
            }
            if (vPage >= numPages && static_cast<int64_t>(vPage) < 2 * static_cast<int64_t>(numPages))
            {
                int64_t const vOffset
                    = (static_cast<int64_t>(vPage) * rt::kTOKENS_PER_PAGE + inPage) * numKVHead * headDim
                    + static_cast<int64_t>(kvHeadIdx) * headDim + vecBase;
                storeVec(kvCache, vOffset, vSrc, vScaleQuantOrig);
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
    uint32_t const headDim = static_cast<uint32_t>(kvCache.getShape()[4]);
    uint32_t const numKVHeads = static_cast<uint32_t>(kvCache.getShape()[3]);
    int32_t const numPages = static_cast<int32_t>(kvCache.getShape()[1]);
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
            kvCacheEndLensPtr, tokenPosIdsPtr, kScale, vScale, runtimeSeqLen, totalNumTokens, numPages, numQHeads,
            numKVHeads, headDim, rotaryDim, cosSinCacheBatchSize, cosSinCacheSeqLen, writeKInPlace, pageTable,
            maxPagesPerSeq);
    }
#if SUPPORTS_FP8
    else if (dt == nvinfer1::DataType::kFP8)
    {
        __nv_fp8_e4m3* kvCachePtr = kvCache.dataPointer<__nv_fp8_e4m3>();
        applyRopeWriteKV<half, __nv_fp8_e4m3><<<grid, block, 0, stream>>>(qPtr, kPtr, vPtr, kvCachePtr, cosSinCachePtr,
            kvCacheEndLensPtr, tokenPosIdsPtr, kScale, vScale, runtimeSeqLen, totalNumTokens, numPages, numQHeads,
            numKVHeads, headDim, rotaryDim, cosSinCacheBatchSize, cosSinCacheSeqLen, writeKInPlace, pageTable,
            maxPagesPerSeq);
    }
#endif
    else
    {
        check::check(false, "Unsupported KV cache dtype");
    }
}

static void validatePagedKvPool(rt::Tensor const& kvCache, int64_t const numKVHeads, int64_t const headDim,
    int32_t const* pageTable, int32_t const maxPagesPerSeq)
{
    rt::Coords const poolShape = kvCache.getShape();
    constexpr int32_t kPoolNumDims = 5;
    constexpr int64_t kKvPlanes = 2;
    check::check(poolShape.getNumDims() == kPoolNumDims, "KV cache pool shape shall be [2, numPages, 128, Hkv, D].");
    check::check(poolShape[0] == kKvPlanes && poolShape[1] > 0 && poolShape[2] == rt::kTOKENS_PER_PAGE,
        "KV cache pool shape shall be [2, numPages, 128, Hkv, D] with positive numPages.");
    check::check(poolShape[3] == numKVHeads && poolShape[4] == headDim,
        "KV cache pool Hkv and D shall match the input K/V tensors.");
    check::check(pageTable != nullptr, "Paged KV writes require a page table.");
    check::check(maxPagesPerSeq > 0, "Paged KV writes require positive maxPagesPerSeq.");
    check::check(poolShape[1] >= maxPagesPerSeq, "Paged KV writes require maxPagesPerSeq <= numPages.");
}

void launchApplyRopeWriteKV(rt::Tensor const& cosSinCache, rt::OptionalInputTensor kvCacheEndLens, rt::Tensor& q,
    rt::Tensor& k, rt::Tensor const& v, rt::Tensor& kvCache, float kScale, float vScale, cudaStream_t stream,
    bool writeKInPlace, int32_t const* pageTable, int32_t maxPagesPerSeq)
{
    rt::OptionalInputTensor tokenPosIds{std::nullopt};

    int64_t const batchSize = q.getShape()[0];
    int64_t const headDim = q.getShape()[3];
    int64_t const numKVHeads = k.getShape()[2];

    check::check(k.getShape()[0] == batchSize && v.getShape()[0] == batchSize, "Q/K/V shall have the same batch size.");
    check::check(k.getShape()[2] == numKVHeads && v.getShape()[2] == numKVHeads && k.getShape()[3] == headDim
            && v.getShape()[3] == headDim,
        "K/V shall have consistent head counts and head dimensions.");
    check::check(cosSinCache.getShape()[0] == 1 || cosSinCache.getShape()[0] == batchSize,
        "CosSinCache shall have batch size 1 or equal to runtime batch size");

    validatePagedKvPool(kvCache, numKVHeads, headDim, pageTable, maxPagesPerSeq);

    if (kvCacheEndLens.has_value())
    {
        check::check(kvCacheEndLens.value().get().getShape()[0] == batchSize,
            "kvCacheEndLens shall have consistent batch size.");
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
    int64_t const numKVHeads = k.getShape()[2];
    int64_t const runtimeSeqLen = q.getShape()[1];

    check::check(k.getShape()[0] == batchSize && v.getShape()[0] == batchSize
            && kvCacheEndLens.getShape()[0] == batchSize && tokenPosIds.getShape()[0] == batchSize,
        "All Input tensors shall have consistent batch size.");
    check::check(
        k.getShape()[3] == headDim && v.getShape()[3] == headDim, "Head dimension shall be consistent between Q/K/V.");
    check::check(v.getShape()[2] == numKVHeads, "K/V shall have consistent number of heads.");
    check::check(tokenPosIds.getShape()[1] == runtimeSeqLen, "Q/tokenPosIds shall have consistent sequence length.");
    check::check(cosSinCache.getShape()[0] == 1 || cosSinCache.getShape()[0] == batchSize,
        "CosSinCache shall have batch size 1 or equal to runtime batch size");
    validatePagedKvPool(kvCache, numKVHeads, headDim, pageTable, maxPagesPerSeq);

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
    int32_t qSeqLen, int32_t totalNumTokens, int32_t numPages, uint32_t numQHead, uint32_t numKVHead, uint32_t headDim,
    uint32_t rotaryDim, int32_t cosSinCacheBatchSize, int32_t cosSinCacheSeqLen, int32_t const* __restrict__ pageTable,
    int32_t maxPagesPerSeq)
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
        int32_t const pageRow = tokenIdxInCache / rt::kTOKENS_PER_PAGE;
        int32_t const inPage = tokenIdxInCache % rt::kTOKENS_PER_PAGE;
        int32_t const kPage = pageTable[(batchIdx * 2 + 0) * maxPagesPerSeq + pageRow];
        int32_t const vPage = pageTable[(batchIdx * 2 + 1) * maxPagesPerSeq + pageRow];
        if (kPage >= 0 && kPage < numPages)
        {
            int64_t const kOffset = (static_cast<int64_t>(kPage) * rt::kTOKENS_PER_PAGE + inPage) * numKVHead * headDim
                + static_cast<int64_t>(kvHeadIdx) * headDim + vecBase;
            storeVec(kvCache, kOffset, kRoped, kScaleQuantOrig);
        }
        if (vPage >= numPages && static_cast<int64_t>(vPage) < 2 * static_cast<int64_t>(numPages))
        {
            int64_t const vOffset = (static_cast<int64_t>(vPage) * rt::kTOKENS_PER_PAGE + inPage) * numKVHead * headDim
                + static_cast<int64_t>(kvHeadIdx) * headDim + vecBase;
            storeVec(kvCache, vOffset, vSrc, vScaleQuantOrig);
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
    uint32_t const headDim = static_cast<uint32_t>(kvCache.getShape()[4]);
    uint32_t const numKVHeads = static_cast<uint32_t>(kvCache.getShape()[3]);
    int32_t const numPages = static_cast<int32_t>(kvCache.getShape()[1]);
    uint32_t const totalNumTokens = runtimeBatchSize * runtimeSeqLen;

    check::check(k.getShape()[0] == runtimeBatchSize && v.getShape()[0] == runtimeBatchSize
            && kvCacheEndLens.getShape()[0] == runtimeBatchSize,
        "Q/K/V and kvCacheEndLens shall have the same batch size.");
    check::check(q.getShape()[3] == headDim && k.getShape()[1] == runtimeSeqLen && v.getShape()[1] == runtimeSeqLen
            && k.getShape()[2] == numKVHeads && v.getShape()[2] == numKVHeads && k.getShape()[3] == headDim
            && v.getShape()[3] == headDim,
        "Q/K/V shall have consistent sequence length, heads, and head dimension.");
    validatePagedKvPool(kvCache, numKVHeads, headDim, pageTable, maxPagesPerSeq);

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
            cosSinCachePtr, kvCacheEndLensPtr, 1.0f, kScale, vScale, runtimeSeqLen, totalNumTokens, numPages, numQHeads,
            numKVHeads, headDim, rotaryDim, cosSinCacheBatchSize, cosSinCacheSeqLen, pageTable, maxPagesPerSeq);
    }
#if SUPPORTS_FP8
    else if (dt == nvinfer1::DataType::kFP8)
    {
        __nv_fp8_e4m3* kvCachePtr = kvCache.dataPointer<__nv_fp8_e4m3>();
        applyRopeWriteKVSplitQKVKernel<half, __nv_fp8_e4m3><<<grid, block, 0, stream>>>(qPtr, kPtr, vPtr, kvCachePtr,
            fp8QOut, cosSinCachePtr, kvCacheEndLensPtr, qScale, kScale, vScale, runtimeSeqLen, totalNumTokens, numPages,
            numQHeads, numKVHeads, headDim, rotaryDim, cosSinCacheBatchSize, cosSinCacheSeqLen, pageTable,
            maxPagesPerSeq);
    }
#endif
    else
    {
        check::check(false, "Unsupported KV cache dtype");
    }
}

// =============================================================================
// Packed QKV [B, S, Hq+2*Hkv, D] → RoPE Q/K → split outputs:
//   - roped Q to qScratch (or FP8 Q to fp8QOut), always
//   - roped K + V to kvCache, always
//   - roped K + V to kScratch/vScratch when SEPARATE_Q_K_V FMHA needs them
// Tree decoding via optional tokenPosIds (-1 = padding token).
// =============================================================================

template <typename T, typename TCache>
__global__ void applyRopeFromPackedToSplitKernel(T const* __restrict__ packedQKV, T* __restrict__ qScratch,
    T* __restrict__ kScratch, T* __restrict__ vScratch, TCache* __restrict__ kvCache, void* __restrict__ fp8QOut,
    float const* __restrict__ cosSinCache, int32_t const* __restrict__ kvCacheEndLens,
    int32_t const* __restrict__ tokenPosIds, int32_t const* __restrict__ cuQSeqLens, T const* __restrict__ qNormGamma,
    T const* __restrict__ kNormGamma, float rmsNormEps, float qScaleQuantOrig, float kScaleQuantOrig,
    float vScaleQuantOrig, int32_t qSeqLen, int32_t totalNumTokens, int32_t numPages, uint32_t numQHead,
    uint32_t numKVHead, uint32_t headDim, uint32_t rotaryDim, int32_t cosSinCacheBatchSize, int32_t cosSinCacheSeqLen,
    int32_t const* __restrict__ pageTable, int32_t maxPagesPerSeq)
{
    // Thread mapping (same as existing kernels for proven memory coalescing):
    //   blockDim.x = headDim / vec_size  (threads per token, cover head vector)
    //   blockDim.y = tokens per CTA
    //   gridDim.x  = ceil(totalNumTokens / blockDim.y)
    //   gridDim.y  = numQHead + numKVHead  (each CTA processes one head-type)
    //
    // Packed QKV stride per token = (numQHead + 2*numKVHead) * headDim.
    // Within a token: Q at offset [0, Hq), K at [Hq, Hq+Hkv), V at [Hq+Hkv, Hq+2*Hkv).

    uint32_t const tIdx = threadIdx.x;
    uint32_t const tIdy = threadIdx.y;
    uint32_t const tokenIdx = blockIdx.x * blockDim.y + tIdy;

    // Tail-token lanes must not early-return: a warp can span multiple token rows and the
    // fused-norm path runs full-mask warp collectives. They run on a clamped valid token
    // instead, and every global store below is gated on !isTailToken.
    bool const isTailToken = (tokenIdx >= static_cast<uint32_t>(totalNumTokens));
    uint32_t const clampedTokenIdx = isTailToken ? static_cast<uint32_t>(totalNumTokens - 1) : tokenIdx;

    int32_t const batchIdx = clampedTokenIdx / qSeqLen;
    int64_t const combinedHeads = static_cast<int64_t>(numQHead) + 2 * static_cast<int64_t>(numKVHead);

    // RoPE position: prefill (kvCacheEndLens - qSeqLen + offset), decode
    // (kvCacheEndLens[b] - 1), or tree (tokenPosIds; -1 = padding token, zeroed).
    // Ragged prefill padding must be identified before page-table or RoPE-cache
    // indexing. It cannot early-return because fused qk_norm uses warp collectives.
    int32_t const rowInBatch = static_cast<int32_t>(clampedTokenIdx % qSeqLen);
    int32_t actualQSeqLen = qSeqLen;
    if (cuQSeqLens != nullptr)
    {
        actualQSeqLen = cuQSeqLens[batchIdx + 1] - cuQSeqLens[batchIdx];
    }
    int32_t sinCosCachePos{};
    bool const isPaddingToken = (tokenPosIds != nullptr && tokenPosIds[clampedTokenIdx] == -1)
        || (cuQSeqLens != nullptr && rowInBatch >= actualQSeqLen);
    if (tokenPosIds != nullptr)
    {
        sinCosCachePos = tokenPosIds[clampedTokenIdx];
        if (sinCosCachePos < 0)
        {
            sinCosCachePos = 0;
        }
    }
    else
    {
        int32_t const posStartId = kvCacheEndLens != nullptr ? kvCacheEndLens[batchIdx] - qSeqLen : 0;
        int32_t const maxActualRow = actualQSeqLen > 0 ? actualQSeqLen - 1 : 0;
        int32_t const ropeRow = isPaddingToken && rowInBatch > maxActualRow ? maxActualRow : rowInBatch;
        sinCosCachePos = posStartId + ropeRow;
    }

    // Vectorized load cos/sin for this token's RoPE position.
    uint32_t const sinOffset = rotaryDim / 2;
    uint32_t const cosOffset = (tIdx * DVec<float>::vec_size) % (rotaryDim / 2);
    int32_t const cosSinCacheBatchIdx = (cosSinCacheBatchSize == 1) ? 0 : batchIdx;
    int32_t const cosSinCacheOffset = cosSinCacheBatchIdx * cosSinCacheSeqLen * rotaryDim + sinCosCachePos * rotaryDim;
    DVec<float> cosVec;
    DVec<float> sinVec;
    cosVec.load(cosSinCache + cosSinCacheOffset + cosOffset);
    sinVec.load(cosSinCache + cosSinCacheOffset + cosOffset + sinOffset);

    uint32_t const headIdx = blockIdx.y;
    int64_t const tokenBaseInPacked = static_cast<int64_t>(clampedTokenIdx) * combinedHeads * headDim;
    int32_t const vecBase = DVec<T>::vec_size * tIdx;

    // Ghost lanes (tIdx >= actualLanesPerHead) exist only to make blockDim.x a power of 2
    // for the warp-shuffle reduction (e.g. headDim=96: 12 -> 16).
    uint32_t const paddedLanesPerHead = blockDim.x;
    uint32_t const actualLanesPerHead = static_cast<uint32_t>(headDim) / DVec<T>::vec_size;
    bool const isActiveLane = (tIdx < actualLanesPerHead);

    if (headIdx < numQHead)
    {
        // --- Q head: read from packed QKV, optionally apply RMSNorm, apply RoPE,
        //     write to qScratch (or fp8QOut) ---
        uint32_t const qHeadIdx = headIdx;
        int64_t const packedQOffset = tokenBaseInPacked + static_cast<int64_t>(qHeadIdx) * headDim;
        int32_t const scratchQOffset = clampedTokenIdx * numQHead * headDim + qHeadIdx * headDim;

        DVec<T> qRoped;
        if (qNormGamma != nullptr)
        {
            // ALL lanes (ghosts and padding tokens included) must enter uniformly — the
            // call contains a warp-wide shfl. Padding tokens' qRoped is zeroed below.
            qRoped = vecApplyRmsNormAndRopeNonInterleave(packedQKV + packedQOffset, qNormGamma, cosVec, sinVec,
                rotaryDim, headDim, paddedLanesPerHead, rmsNormEps);
            if (isPaddingToken && isActiveLane)
            {
#pragma unroll
                for (uint32_t i = 0; i < DVec<T>::vec_size; ++i)
                {
                    qRoped[i] = T(0);
                }
            }
        }
        else if (isActiveLane)
        {
            // No norm path: ghost lanes skip (would OOB at vecBase >= headDim).
            if (isPaddingToken)
            {
                // Zero out Q for padding tokens (tree decoding).
#pragma unroll
                for (uint32_t i = 0; i < DVec<T>::vec_size; ++i)
                {
                    qRoped[i] = T(0);
                }
            }
            else
            {
                qRoped = vecApplyRopeNonInterleave(packedQKV + packedQOffset, cosVec, sinVec, rotaryDim);
            }
        }

        if (isActiveLane && !isTailToken)
        {
#if SUPPORTS_FP8
            if (fp8QOut != nullptr)
            {
                // CuTeDSL FP8 path: quantize roped Q to FP8 and store in separate buffer.
                storeVec(reinterpret_cast<__nv_fp8_e4m3*>(fp8QOut), scratchQOffset + vecBase, qRoped, qScaleQuantOrig);
            }
            else
#endif
            {
                qRoped.store(qScratch + scratchQOffset + vecBase);
            }
        }
    }
    else
    {
        // --- KV head: read K and V from packed, optionally RMSNorm(K), RoPE(K), write to scratch + cache ---
        uint32_t const kvHeadIdx = headIdx - numQHead;
        int64_t const packedKOffset = tokenBaseInPacked + static_cast<int64_t>(numQHead + kvHeadIdx) * headDim;
        int64_t const packedVOffset
            = tokenBaseInPacked + static_cast<int64_t>(numQHead + numKVHead + kvHeadIdx) * headDim;

        DVec<T> kRoped;
        if (kNormGamma != nullptr)
        {
            // All lanes (including ghosts) participate in the fused-norm shfl reduction.
            kRoped = vecApplyRmsNormAndRopeNonInterleave(packedQKV + packedKOffset, kNormGamma, cosVec, sinVec,
                rotaryDim, headDim, paddedLanesPerHead, rmsNormEps);
        }
        else if (isActiveLane)
        {
            // RoPE-only path: ghost lanes skip — vecBase would be OOB.
            kRoped = vecApplyRopeNonInterleave(packedQKV + packedKOffset, cosVec, sinVec, rotaryDim);
        }
        DVec<T> vSrc;
        if (isActiveLane)
        {
            vSrc.load(packedQKV + packedVOffset + vecBase);
        }

        // Skip all K/V writes for padding tokens (tree decoding), tail tokens, and ghost lanes.
        if (!isPaddingToken && isActiveLane && !isTailToken)
        {
            int32_t const scratchKVOffset = clampedTokenIdx * numKVHead * headDim + kvHeadIdx * headDim;

            // Optionally write to scratch K/V (for SEPARATE_Q_K_V FMHA to read directly).
            if (kScratch != nullptr)
            {
                kRoped.store(kScratch + scratchKVOffset + vecBase);
            }
            if (vScratch != nullptr)
            {
                vSrc.store(vScratch + scratchKVOffset + vecBase);
            }

            // Always write K and V to the KV cache.
            int32_t const kvCacheStartIdx = kvCacheEndLens != nullptr ? kvCacheEndLens[batchIdx] - qSeqLen : 0;
            int32_t const tokenIdxInCache = kvCacheStartIdx + clampedTokenIdx % qSeqLen;
            int32_t const pageRow = tokenIdxInCache / rt::kTOKENS_PER_PAGE;
            int32_t const inPage = tokenIdxInCache % rt::kTOKENS_PER_PAGE;
            int32_t const kPage = pageTable[(batchIdx * 2 + 0) * maxPagesPerSeq + pageRow];
            int32_t const vPage = pageTable[(batchIdx * 2 + 1) * maxPagesPerSeq + pageRow];
            if (kPage >= 0 && kPage < numPages)
            {
                int64_t const kOffset
                    = (static_cast<int64_t>(kPage) * rt::kTOKENS_PER_PAGE + inPage) * numKVHead * headDim
                    + static_cast<int64_t>(kvHeadIdx) * headDim + vecBase;
                storeVec(kvCache, kOffset, kRoped, kScaleQuantOrig);
            }
            if (vPage >= numPages && static_cast<int64_t>(vPage) < 2 * static_cast<int64_t>(numPages))
            {
                int64_t const vOffset
                    = (static_cast<int64_t>(vPage) * rt::kTOKENS_PER_PAGE + inPage) * numKVHead * headDim
                    + static_cast<int64_t>(kvHeadIdx) * headDim + vecBase;
                storeVec(kvCache, vOffset, vSrc, vScaleQuantOrig);
            }
        }
    }
}

void launchApplyRopeFromPackedToSplit(rt::Tensor const& cosSinCache, rt::OptionalInputTensor kvCacheEndLens,
    rt::OptionalInputTensor tokenPosIds, rt::Tensor const& packedQKV, rt::Tensor& qScratch, rt::Tensor& kvCache,
    float kScale, float vScale, cudaStream_t stream, int32_t const* pageTable, int32_t maxPagesPerSeq,
    void* kScratchOut, void* vScratchOut, void* fp8QOut, float qScale, half const* qNormGamma, half const* kNormGamma,
    float rmsNormEps, rt::OptionalInputTensor cuQSeqLens)
{
    auto const dt = kvCache.getDataType();
    constexpr uint32_t kVEC_SIZE = DVec<half>::vec_size;
    constexpr uint32_t kTHREADS_PER_CTA = 128;

    // packedQKV: [B, S, Hq+2*Hkv, D]
    int64_t const batchSize = packedQKV.getShape()[0];
    int64_t const runtimeSeqLen = packedQKV.getShape()[1];
    int64_t const combinedHeads = packedQKV.getShape()[2];
    int64_t const headDim = kvCache.getShape()[4];
    int64_t const numKVHeads = kvCache.getShape()[3];
    int64_t const numQHeads = combinedHeads - 2 * numKVHeads;
    int64_t const numPages = kvCache.getShape()[1];
    int64_t const totalNumTokens = batchSize * runtimeSeqLen;

    check::check(numQHeads > 0, "Packed QKV combined heads must exceed 2x KV heads.");
    check::check(qScratch.getShape()[0] == batchSize && qScratch.getShape()[1] == runtimeSeqLen
            && qScratch.getShape()[2] == numQHeads && qScratch.getShape()[3] == headDim,
        "qScratch shape shall be [B, S, Hq, D].");
    check::check(
        packedQKV.getShape()[3] == headDim, "Head dimension shall be consistent between packed QKV and KV pool.");
    validatePagedKvPool(kvCache, numKVHeads, headDim, pageTable, maxPagesPerSeq);

    int64_t const cosSinCacheBatchSize = cosSinCache.getShape()[0];
    int64_t const cosSinCacheSeqLen = cosSinCache.getShape()[1];
    int64_t const rotaryDim = cosSinCache.getShape()[2];
    check::check(cosSinCacheBatchSize == 1 || cosSinCacheBatchSize == batchSize,
        "CosSinCache shall have batch size 1 or equal to runtime batch size");

    if (kvCacheEndLens.has_value())
    {
        check::check(kvCacheEndLens.value().get().getShape()[0] == batchSize,
            "kvCacheEndLens shall have consistent batch size.");
    }
    if (tokenPosIds.has_value())
    {
        check::check(tokenPosIds.value().get().getShape()[0] == batchSize
                && tokenPosIds.value().get().getShape()[1] == runtimeSeqLen,
            "tokenPosIds shape shall be [B, S].");
    }
    if (cuQSeqLens.has_value())
    {
        check::check(cuQSeqLens.value().get().getShape()[0] == batchSize + 1, "cuQSeqLens shape shall be [B + 1].");
        check::check(cuQSeqLens.value().get().getDataType() == nvinfer1::DataType::kINT32,
            "cuQSeqLens shall have INT32 data type.");
    }

    half const* packedPtr = packedQKV.dataPointer<half>();
    half* qScratchPtr = qScratch.dataPointer<half>();
    half* kScratchPtr = reinterpret_cast<half*>(kScratchOut);
    half* vScratchPtr = reinterpret_cast<half*>(vScratchOut);
    float const* cosSinCachePtr = cosSinCache.dataPointer<float>();
    int32_t const* kvCacheEndLensPtr
        = kvCacheEndLens.has_value() ? kvCacheEndLens.value().get().dataPointer<int32_t>() : nullptr;
    int32_t const* tokenPosIdsPtr
        = tokenPosIds.has_value() ? tokenPosIds.value().get().dataPointer<int32_t>() : nullptr;
    int32_t const* cuQSeqLensPtr = cuQSeqLens.has_value() ? cuQSeqLens.value().get().dataPointer<int32_t>() : nullptr;

    // Fused RMSNorm needs a power-of-2 lane count for the warp-shuffle butterfly: pad
    // blockDim.x to nextPowerOf2(headDim / vec_size); ghost lanes only join the shfl.
    // Power-of-2 lane counts (e.g. headDim=128 -> 16) get no padding and no overhead.
    uint32_t const actualBDimX = static_cast<uint32_t>(headDim) / kVEC_SIZE;
    bool const fusedNormEnabled = (qNormGamma != nullptr || kNormGamma != nullptr);
    uint32_t const bDimX = fusedNormEnabled ? nextPowerOf2(actualBDimX) : actualBDimX;
    // The <= 32 lane constraint only applies with fused norm; without qk_norm larger heads
    // (e.g. headDim=512 => 64 lanes) take the plain RoPE path (no cross-lane reduction).
    check::check(!fusedNormEnabled || bDimX <= 32u,
        "Fused qk_norm in attention plugin requires paddedLanesPerHead <= 32 (i.e. headDim/vec_size <= 32). "
        "Disable qk_norm fusion for this model (or extend the warp-shuffle reduction to span multiple warps).");
    // Tokens per CTA computed from the (possibly padded) bDimX so that total threads ≤ kTHREADS_PER_CTA.
    // Floor-divide to match the original formula (kTHREADS_PER_CTA * vec_size / headDim) when no padding.
    uint32_t const tokenPerCTA = kTHREADS_PER_CTA / bDimX;
    uint32_t const bDimY = tokenPerCTA;
    uint32_t const gDimX = (static_cast<uint32_t>(totalNumTokens) + tokenPerCTA - 1) / tokenPerCTA;
    uint32_t const gDimY = static_cast<uint32_t>(numQHeads + numKVHeads);

    dim3 grid(gDimX, gDimY);
    dim3 block(bDimX, bDimY);

    if (dt == nvinfer1::DataType::kHALF)
    {
        half* kvCachePtr = kvCache.dataPointer<half>();
        applyRopeFromPackedToSplitKernel<half, half><<<grid, block, 0, stream>>>(packedPtr, qScratchPtr, kScratchPtr,
            vScratchPtr, kvCachePtr, nullptr /* fp8QOut */, cosSinCachePtr, kvCacheEndLensPtr, tokenPosIdsPtr,
            cuQSeqLensPtr, qNormGamma, kNormGamma, rmsNormEps, 1.0f /* qScaleQuantOrig */, kScale, vScale,
            static_cast<int32_t>(runtimeSeqLen), static_cast<int32_t>(totalNumTokens), static_cast<int32_t>(numPages),
            static_cast<uint32_t>(numQHeads), static_cast<uint32_t>(numKVHeads), static_cast<uint32_t>(headDim),
            static_cast<uint32_t>(rotaryDim), static_cast<int32_t>(cosSinCacheBatchSize),
            static_cast<int32_t>(cosSinCacheSeqLen), pageTable, maxPagesPerSeq);
    }
#if SUPPORTS_FP8
    else if (dt == nvinfer1::DataType::kFP8)
    {
        __nv_fp8_e4m3* kvCachePtr = kvCache.dataPointer<__nv_fp8_e4m3>();
        applyRopeFromPackedToSplitKernel<half, __nv_fp8_e4m3><<<grid, block, 0, stream>>>(packedPtr, qScratchPtr,
            kScratchPtr, vScratchPtr, kvCachePtr, fp8QOut, cosSinCachePtr, kvCacheEndLensPtr, tokenPosIdsPtr,
            cuQSeqLensPtr, qNormGamma, kNormGamma, rmsNormEps, qScale, kScale, vScale,
            static_cast<int32_t>(runtimeSeqLen), static_cast<int32_t>(totalNumTokens), static_cast<int32_t>(numPages),
            static_cast<uint32_t>(numQHeads), static_cast<uint32_t>(numKVHeads), static_cast<uint32_t>(headDim),
            static_cast<uint32_t>(rotaryDim), static_cast<int32_t>(cosSinCacheBatchSize),
            static_cast<int32_t>(cosSinCacheSeqLen), pageTable, maxPagesPerSeq);
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
