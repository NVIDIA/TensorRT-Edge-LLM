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
//! `pageTable`; a negative page id (kBAD_PAGE_INDEX) means the position is unmapped and the write
//! is silently skipped (same contract as other paged-KV runtimes, e.g. vLLM's slot_mapping < 0).
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
//! @param[out] kvCache FP16/FP8 type tensor with layout of [batchSize, 2, Hkv, kvCacheCapacity, headDim]
//! @param[in] kScale K dequant scale (quant→orig). Use 1.0f for FP16 KV cache.
//! @param[in] vScale V dequant scale (quant→orig). Use 1.0f for FP16 KV cache.
//! @param[in] stream CUDA stream to launch the kernel
//! @param[in] writeKInPlace Controls whether roped K is additionally written back to the K tensor in-place,
//!     on top of always being written to kvCache.
//!     Set to true for the initial prefill path (SEPARATE_Q_K_V) where the downstream FMHA kernel reads Q, K, V
//!     as separate contiguous tensors rather than from the KV cache. In this case K must contain the roped result.
//!     Set to false (default) for chunked prefill with KV cache reuse, where FMHA reads KV from the transposed
//!     KV cache, and for all decoding paths (vanilla / tree), where the XQA kernel reads KV from the cache.
//! @param[in] pageTable INT32 device page table `[batchSize, 2, maxPagesPerSeq]`; row `b*2+0` carries K page ids,
//!     row `b*2+1` the derived V page ids. When non-null, token `t` of slot `b` writes K to
//!     `kvCache[pageTable[(b*2+0)*maxPagesPerSeq + t/128]][t%128][h][d]` and V via the row's V half (both addressing
//!     the same flat `kvCache` buffer, reinterpreted as `[nPages, 128, Hkv, D]`). A negative entry skips the write
//!     for that half. Pass nullptr to use the legacy `[B, 2, Hkv, S, D]` addressing instead.
//! @param[in] maxPagesPerSeq Page-table inner dimension (number of page slots per sequence). Only used when
//!     pageTable != nullptr.
//!     the check (e.g. in unit tests that don't care).
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
//! @param[out] kvCache FP16/FP8 type tensor with layout of [batchSize, 2, Hkv, kvCacheCapacity, headDim], write KVCache
//! from the end position.
//! @param[in] kScale K dequant scale (quant→orig). Use 1.0f for FP16 KV cache.
//! @param[in] vScale V dequant scale (quant→orig). Use 1.0f for FP16 KV cache.
//! @param[in] stream CUDA stream to launch the kernel
//! @note We won't overwrite K/V tensor in this case but we use Tensor& signature to reduce duplicate code.
//! @param[in] pageTable Optional INT32 device page table `[batchSize, 2, maxPagesPerSeq]`. See launchApplyRopeWriteKV.
//! @param[in] maxPagesPerSeq Page-table inner dimension. Only used when pageTable != nullptr.
//! @throws std::runtime_error if tensor shape or data type is incorrect
void launchApplyRopeWriteKVTreeDecoding(rt::Tensor const& cosSinCache, rt::Tensor const& kvCacheEndLens,
    rt::Tensor const& tokenPosIds, rt::Tensor& q, rt::Tensor& k, rt::Tensor const& v, rt::Tensor& kvCache, float kScale,
    float vScale, cudaStream_t stream, int32_t const* pageTable, int32_t maxPagesPerSeq);

//! @brief Launch kernel to apply RoPE to Q, apply RoPE to K and write K/V to KVCache.
//!
//! Optimized for the CuTe DSL FMHA path: applies RoPE to Q, writes roped K and V into
//! KV cache [B, 2, H_kv, S, D]. Does NOT write roped K back to the K input tensor.
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
//! @param[out] kvCache FP16/FP8 type tensor with layout of [batchSize, 2, Hkv, kvCacheCapacity, headDim]
//! @param[in] kScale K dequant scale (quant→orig). Use 1.0f for FP16 KV cache.
//! @param[in] vScale V dequant scale (quant→orig). Use 1.0f for FP16 KV cache.
//! @param[in] stream CUDA stream to launch the kernel
//! @param[in] pageTable Optional INT32 device page table `[batchSize, 2, maxPagesPerSeq]`. See launchApplyRopeWriteKV.
//! @param[in] maxPagesPerSeq Page-table inner dimension. Only used when pageTable != nullptr.
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

} // namespace kernel
} // namespace trt_edgellm
