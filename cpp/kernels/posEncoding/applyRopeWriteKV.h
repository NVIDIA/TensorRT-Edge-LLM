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

#pragma once

#include "common/tensor.h"

#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace trt_edgellm
{
namespace kernel
{

//! Paged-KV bad-page semantics: the write kernels resolve each token's physical page through
//! `pageTable`. A negative page id (`kUNUSED_PAGE_ENTRY`) means the position is unmapped. A K id
//! outside `[0, numPages)` or a V id outside `[numPages, 2 * numPages)` is invalid. Unmapped and
//! invalid ids skip the write for that cache plane.
//! The runtime is responsible for never presenting an unmapped page for a live position: identity
//! tables cover every slot by construction, and reuse-mode table builders enforce coverage on the
//! host at build time.

//! @brief Launch kernel to apply RoPE positional encoding to Q/K and write K/V to KVCache.
//! @param[in] cosSinCache FP32 type tensor with layout of [cosSinCacheBatchSize, cosSinCacheSeqLen, rotaryDim]
//! @param[in] kvCacheEndLens Optional INT32 type tensor with layout of [batchSize], the end position of KVCache after
//! writing. When nullopt, KVCache is written from the start (prefill without prior cache).
//! @param[in,out] q FP16 type tensor with layout of [batchSize, runtimeSeqLen, Hq, headDim]
//! @param[in,out] k FP16 type tensor with layout of [batchSize, runtimeSeqLen, Hkv, headDim]
//! @param[in] v FP16 type tensor with layout of [batchSize, runtimeSeqLen, Hkv, headDim]
//! @param[out] kvCache FP16/FP8 paged pool with layout of [2, numPages, kTOKENS_PER_PAGE, Hkv, headDim]
//! @param[in] kScale K dequant scale (quant→orig). Use 1.0f for FP16 KV cache.
//! @param[in] vScale V dequant scale (quant→orig). Use 1.0f for FP16 KV cache.
//! @param[in] stream CUDA stream to launch the kernel
//! @param[in] writeKInPlace Controls whether roped K is additionally written back to the K tensor in-place,
//!     on top of always being written to kvCache.
//!     Set to true for the initial prefill path (SEPARATE_Q_K_V) where the downstream FMHA kernel reads Q, K, V
//!     as separate contiguous tensors rather than from the KV cache. In this case K must contain the roped result.
//!     Set to false (default) for chunked prefill with KV cache reuse, where FMHA reads KV from the transposed
//!     KV cache, and for all decoding paths (vanilla / tree), where the XQA kernel reads KV from the cache.
//! @param[in] pageTable Required INT32 device page table `[batchSize, 2, maxPagesPerSeq]`. K row ids are in
//!     `[0, numPages)` and V row ids are flattened as `K + numPages` in `[numPages, 2 * numPages)`.
//!     An unmapped or out-of-plane entry skips the write for that cache plane.
//! @param[in] maxPagesPerSeq Positive page-table inner dimension (number of page slots per sequence).
//! @throws std::runtime_error if tensor shape or data type is incorrect
void launchApplyRopeWriteKV(rt::Tensor const& cosSinCache, rt::OptionalInputTensor kvCacheEndLens, rt::Tensor& q,
    rt::Tensor& k, rt::Tensor const& v, rt::Tensor& kvCache, float kScale, float vScale, cudaStream_t stream,
    bool writeKInPlace, int32_t const* pageTable, int32_t maxPagesPerSeq);

//! @brief Launch the kernel when we are performing tree attention for speculative decoding.
//! @param[in] cosSinCache FP32 type tensor with layout of [cosSinCacheBatchSize, cosSinCacheSeqLen, rotaryDim]
//! @param[in] kvCacheEndLens INT32 type tensor with layout of [batchSize], the end position of KVCache after writing.
//! @param[in] tokenPosIds INT32 type tensor with layout of [batchSize, runtimeSeqLen], the position of token within
//! sequence.
//! @param[in,out] q FP16 type tensor with layout of [batchSize, runtimeSeqLen, Hq, headDim]
//! @param[in] k FP16 type tensor with layout of [batchSize, runtimeSeqLen, Hkv, headDim]
//! @param[in] v FP16 type tensor with layout of [batchSize, runtimeSeqLen, Hkv, headDim]
//! @param[out] kvCache FP16/FP8 paged pool with layout of [2, numPages, kTOKENS_PER_PAGE, Hkv, headDim].
//! @param[in] kScale K dequant scale (quant→orig). Use 1.0f for FP16 KV cache.
//! @param[in] vScale V dequant scale (quant→orig). Use 1.0f for FP16 KV cache.
//! @param[in] stream CUDA stream to launch the kernel
//! @note We won't overwrite K/V tensor in this case but we use Tensor& signature to reduce duplicate code.
//! @param[in] pageTable Required INT32 device page table `[batchSize, 2, maxPagesPerSeq]`. See launchApplyRopeWriteKV.
//! @param[in] maxPagesPerSeq Positive page-table inner dimension.
//! @throws std::runtime_error if tensor shape or data type is incorrect
void launchApplyRopeWriteKVTreeDecoding(rt::Tensor const& cosSinCache, rt::Tensor const& kvCacheEndLens,
    rt::Tensor const& tokenPosIds, rt::Tensor& q, rt::Tensor& k, rt::Tensor const& v, rt::Tensor& kvCache, float kScale,
    float vScale, cudaStream_t stream, int32_t const* pageTable, int32_t maxPagesPerSeq);

//! @brief Launch kernel to apply RoPE to Q, apply RoPE to K and write K/V to KVCache.
//!
//! Optimized for the CuTe DSL FMHA path: applies RoPE to Q, writes roped K and V into
//! KV pool [2, numPages, kTOKENS_PER_PAGE, H_kv, D]. Does NOT write roped K back to the K input tensor.
//!
//! When @p fp8QOut is non-null (FP8 KV cache path), the roped Q is quantized to FP8 and written
//! to the provided output buffer. The original FP16 Q tensor is NOT modified. The downstream
//! FP8 FMHA kernel reads Q from fp8QOut and K/V from the KV cache directly.
//!
//! When @p fp8QOut is null (FP16 path), RoPE is applied to Q in-place in the FP16 Q tensor.
//!
//! @param[in] cosSinCache FP32 type tensor with layout of [cosSinCacheBatchSize, cosSinCacheSeqLen, rotaryDim]
//! @param[in] kvCacheEndLens INT32 type tensor with layout of [batchSize], the end position of KVCache after writing.
//! @param[in,out] q FP16 type tensor with layout of [batchSize, runtimeSeqLen, Hq, headDim].
//!     RoPE applied in-place when fp8QOut is null.
//! @param[in] k FP16 type tensor with layout of [batchSize, runtimeSeqLen, Hkv, headDim]
//! @param[in] v FP16 type tensor with layout of [batchSize, runtimeSeqLen, Hkv, headDim]
//! @param[out] kvCache FP16/FP8 paged pool with layout of [2, numPages, kTOKENS_PER_PAGE, Hkv, headDim]
//! @param[in] kScale K dequant scale (quant→orig). Use 1.0f for FP16 KV cache.
//! @param[in] vScale V dequant scale (quant→orig). Use 1.0f for FP16 KV cache.
//! @param[in] stream CUDA stream to launch the kernel
//! @param[in] pageTable Required INT32 device page table `[batchSize, 2, maxPagesPerSeq]`. See launchApplyRopeWriteKV.
//! @param[in] maxPagesPerSeq Positive page-table inner dimension.
//! @param[out] fp8QOut Optional FP8 output buffer for roped Q [batchSize, runtimeSeqLen, Hq, headDim].
//!     When non-null, roped Q is quantized to FP8 E4M3 and stored here. Pass nullptr for FP16 in-place RoPE.
//! @param[in] qScale Q dequant scale (quant→orig). Only used when fp8QOut is non-null.
void launchApplyRopeWriteKVSplitQKV(rt::Tensor const& cosSinCache, rt::Tensor const& kvCacheEndLens, rt::Tensor& q,
    rt::Tensor const& k, rt::Tensor const& v, rt::Tensor& kvCache, float kScale, float vScale, cudaStream_t stream,
    int32_t const* pageTable, int32_t maxPagesPerSeq, void* fp8QOut = nullptr, float qScale = 1.0f);

//! @brief Launch kernel to apply RoPE to Q only (no KV write).
//!
//! Used for shared-KV layers where Q still needs positional encoding but the
//! KV cache belongs to a donor layer and must not be modified.
//!
//! @param[in] cosSinCache FP32 type tensor with layout of [cosSinCacheBatchSize, cosSinCacheSeqLen, rotaryDim]
//! @param[in] kvCacheEndLens INT32 type tensor with layout of [batchSize], used to compute RoPE position.
//! @param[in,out] q FP16 type tensor with layout of [batchSize, runtimeSeqLen, Hq, headDim]. RoPE applied in-place.
//! @param[in] stream CUDA stream to launch the kernel
void launchApplyRopeQOnly(
    rt::Tensor const& cosSinCache, rt::Tensor const& kvCacheEndLens, rt::Tensor& q, cudaStream_t stream);

//! @brief Launch kernel to apply RoPE to Q only, using per-token position IDs (tree decoding).
//!
//! For shared-KV layers during tree/speculative decoding, each candidate token has its own
//! position in the tree. RoPE is applied to Q using these explicit position IDs.
//! No KV cache write is performed (the donor layer's cache is already populated).
//!
//! @param[in] cosSinCache FP32 type tensor with layout of [cosSinCacheBatchSize, cosSinCacheSeqLen, rotaryDim]
//! @param[in] tokenPosIds INT32 type tensor with layout of [batchSize, runtimeSeqLen], per-token position IDs.
//! @param[in,out] q FP16 type tensor with layout of [batchSize, runtimeSeqLen, Hq, headDim]. RoPE applied in-place.
//! @param[in] stream CUDA stream to launch the kernel
void launchApplyRopeQOnlyTreeDecoding(
    rt::Tensor const& cosSinCache, rt::Tensor const& tokenPosIds, rt::Tensor& q, cudaStream_t stream);

//! @brief Launch kernel to read a packed QKV tensor, apply RoPE to Q and K, write roped Q to
//!        a split scratch tensor, and always write roped K and V to KVCache. Optionally also
//!        mirrors roped K and V to separate scratch tensors for the SEPARATE_Q_K_V FMHA path.
//!
//! Packed-input variant of @ref launchApplyRopeWriteKV — one fused QKV tensor in:
//!   - NORMAL_PREFILL (SEPARATE_Q_K_V FMHA): pass non-null @p kScratchOut / @p vScratchOut.
//!   - CHUNKED_PREFILL / decode: pass nullptr — K/V are read back from the cache.
//!   - Tree decoding: pass @p tokenPosIds (-1 = padding token, no cache write).
//!   - CuTeDSL + FP8: pass @p fp8QOut for FP8 roped Q; otherwise qScratch gets FP16 Q.
//!
//! @param[in]  cosSinCache  FP32 tensor [cosSinCacheBatchSize, cosSinCacheSeqLen, rotaryDim]
//! @param[in]  kvCacheEndLens Optional INT32 tensor [batchSize] — KV cache end position after insertion.
//!             Pass nullopt for prefill without prior cache (starts at position 0).
//! @param[in]  tokenPosIds  Optional INT32 tensor [batchSize, runtimeSeqLen] for tree decoding.
//!             Position -1 marks padding tokens whose Q is zeroed and K/V writes are skipped.
//! @param[in]  packedQKV    FP16 tensor [batchSize, runtimeSeqLen, Hq+2*Hkv, headDim], read-only.
//! @param[out] qScratch     FP16 tensor [batchSize, runtimeSeqLen, Hq, headDim] — roped Q output
//!             (unless @p fp8QOut is non-null, in which case this is unused).
//! @param[out] kvCache      FP16/FP8 paged pool [2, numPages, kTOKENS_PER_PAGE, Hkv, headDim] — K/V written here.
//! @param[in]  kScale       K dequant scale (quant→orig). Use 1.0f for FP16 KV cache.
//! @param[in]  vScale       V dequant scale (quant→orig). Use 1.0f for FP16 KV cache.
//! @param[in]  stream       CUDA stream.
//! @param[in]  pageTable    Required INT32 device page table `[batchSize, 2, maxPagesPerSeq]`.
//!             See launchApplyRopeWriteKV.
//! @param[in]  maxPagesPerSeq Positive page-table inner dimension.
//! @param[out] kScratchOut  Optional FP16 buffer [batchSize, runtimeSeqLen, Hkv, headDim] — mirrored roped K.
//!             Pass nullptr if downstream does not need scratch K.
//! @param[out] vScratchOut  Optional FP16 buffer [batchSize, runtimeSeqLen, Hkv, headDim] — mirrored V.
//!             Pass nullptr if downstream does not need scratch V.
//! @param[out] fp8QOut      Optional FP8 buffer [batchSize, runtimeSeqLen, Hq, headDim] — FP8-quantized roped Q.
//!             Pass nullptr for FP16 Q via qScratch.
//! @param[in]  qScale       Q dequant scale (quant→orig). Only used when @p fp8QOut is non-null.
//! @param[in]  qNormGamma   Optional FP16 device pointer [headDim] for per-head RMSNorm gamma applied to Q
//!             BEFORE RoPE. When non-null, qk_norm is computed inside this kernel via warp-shuffle
//!             reduction across the headDim/vec_size threads of blockDim.x.
//! @param[in]  kNormGamma   Optional FP16 device pointer [headDim] for per-head RMSNorm gamma applied to K
//!             BEFORE RoPE. Same conventions as @p qNormGamma. V is never RMSNormed.
//! @param[in]  rmsNormEps   Epsilon for the RMSNorm formula. Ignored when both gamma pointers are null.
//! @param[in]  cuQSeqLens   Optional INT32 tensor [batchSize + 1] carrying actual cumulative Q lengths for
//!             ragged prefill. Rows at or beyond the actual per-batch length have Q zeroed and skip all K/V writes.
//! @throws std::runtime_error if tensor shape or data type is incorrect.
void launchApplyRopeFromPackedToSplit(rt::Tensor const& cosSinCache, rt::OptionalInputTensor kvCacheEndLens,
    rt::OptionalInputTensor tokenPosIds, rt::Tensor const& packedQKV, rt::Tensor& qScratch, rt::Tensor& kvCache,
    float kScale, float vScale, cudaStream_t stream, int32_t const* pageTable, int32_t maxPagesPerSeq,
    void* kScratchOut = nullptr, void* vScratchOut = nullptr, void* fp8QOut = nullptr, float qScale = 1.0f,
    half const* qNormGamma = nullptr, half const* kNormGamma = nullptr, float rmsNormEps = 1e-6f,
    rt::OptionalInputTensor cuQSeqLens = std::nullopt);

} // namespace kernel
} // namespace trt_edgellm
