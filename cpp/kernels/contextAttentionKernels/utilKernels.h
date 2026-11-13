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

#include "common/cudaMacros.h"
#include "common/pagedKvTypes.h"
#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>

namespace trt_edgellm
{
namespace kernel
{

//! Expand [B, S] vision-block IDs into per-position [blockBegin, blockEnd]
//! interval tensors for the vision-block overlay prefill kernels.
//!
//! Each contiguous run of an identical non-negative ID inside the per-batch
//! valid prefix (contextLengths[b], clamped to seqLen) yields
//! blockBegin = run start and blockEnd = run end for every position in the
//! run.  Text/audio positions (ID < 0) and padding positions receive the
//! -1/-2 sentinel (empty interval).  All tensors are [B, S] int32 device
//! buffers.
void launchBuildVisionBlockRanges(int32_t const* visionBlockIds, int32_t const* contextLengths, int32_t* blockBegin,
    int32_t* blockEnd, int32_t batchSize, int32_t seqLen, cudaStream_t stream);

//! Fill [B, S] intervals that let every query row attend all logical KV
//! positions [0, cuKVSeqLens[b+1] - cuKVSeqLens[b]) through the vision-block overlay.
void launchBuildPaddingBlockRanges(int32_t const* cuKVSeqLens, int32_t* blockBegin, int32_t* blockEnd,
    int32_t batchSize, int32_t seqLen, cudaStream_t stream);

//! Gather token-aligned RoPE rows from a shared cache or a resident-slot cache.
//!
//! `sourceRows == 1` broadcasts the standard RoPE cache. Otherwise sequence
//! `i` resolves its source row through `stateIndices[i]`. Only rows in the
//! query intervals are gathered; padding remains zero.
void launchGatherTokenAlignedRope(float const* source, float* output, int32_t const* positions,
    int32_t const* queryStartOffsets, int32_t const* queryLengths, int32_t const* stateIndices, int32_t numTokens,
    int32_t numSequences, int32_t sourceRows, int32_t cacheCapacity, int32_t rotaryDim, cudaStream_t stream);

//! Scatter contiguous active rows into a resident-slot tensor using a device-side row map.
//! Invalid resident rows are ignored.
void launchScatterActiveRows(void const* source, void* destination, int32_t const* stateIndices, int32_t activeRows,
    int32_t residentRows, size_t rowBytes, cudaStream_t stream);

//! Fill a [numSequences, 2, maxPagesPerSeq] page table that maps sequence b's logical page p to K page
//! b * maxPagesPerSeq + p and V page vPageOffset + b * maxPagesPerSeq + p, i.e. a dense [B, cap, H, D] K buffer
//! followed by a V buffer viewed as one page pool.
void launchBuildDensePageTable(
    int32_t* pageTable, int32_t numSequences, int32_t maxPagesPerSeq, int32_t vPageOffset, cudaStream_t stream);

//! \brief Host-side wrapper that launches a lightweight CUDA kernel to compute prefix-sum of sequence lengths
//! and KV cache end indices.
//!
//! \param[in]  inputSeqLen       int32_t tensor with shape [B].  Actual token length of each request.
//! \param[in]  kvCacheStartIndices int32_t tensor with shape [B].  Start index of KV cache for each request.
//!                                (optional, pass in empty tensor to indicate zero start indices)
//! \param[out] cuQSeqLens        int32_t tensor with shape [B+1]. Exclusive prefix-sum of inputSeqLen.
//! \param[out] cuKVSeqLens       int32_t tensor with shape [B+1]. Exclusive prefix-sum of (kvCacheStartIndices[i] +
//!                                inputSeqLen[i]). If kvCacheStartIndices is empty, this will be exclusive prefix-sum
//!                                of inputSeqLen.
//! \param[out] kvCacheEndIdxs    int32_t tensor with shape [B].  Each element equals
//!                                kvCacheStartIndices[i] + runtimeSeqLen (Here we use padding to ease later kernel
//!                                launch).
//! \param[out] paddedCuKVSeqLens (optional) int32_t tensor with shape [B+1]. Exclusive prefix-sum of kvCacheEndIdxs
//!                                (= kvCacheStartIdx + runtimeSeqLen per batch). Pass std::nullopt to skip.
//!                                Background: CuTe DSL FMHA kernel uses bottom_right_align with offset = s_k - s_q.
//!                                Q is padded to runtimeSeqLen for all batches, so we must use padded KV lengths
//!                                (s_k = kvCacheEndIdx per batch) to keep offset non-negative. Using actual s_k
//!                                (< runtimeSeqLen for shorter batches) would produce a negative offset that masks
//!                                out valid KV positions, breaking attention.
//! \param[in]  runtimeSeqLen     Runtime sequence length (equals to the maximum of inputSeqLen).
//! \param[in]  stream            CUDA stream used to launch the kernel.
//! \note kvCacheStartIndices is optional. If it is not provided, kvStartIndices will be assumed to be 0.
//! \throws std::runtime_error if tensor shapes are invalid
void calCuQCuKVSeqLensAndKVEndIdxs(rt::Tensor const& inputSeqLen, rt::Tensor const& kvCacheStartIndices,
    rt::Tensor& cuQSeqLens, rt::Tensor& cuKVSeqLens, rt::Tensor& kvCacheEndIdxs,
    rt::OptionalOutputTensor paddedCuKVSeqLens, int32_t const runtimeSeqLen, cudaStream_t stream);

//! Build backend-neutral sequence metadata for ragged paged context attention.
//!
//! cuQSeqLens is the exclusive prefix sum of the active input lengths. cuKVSeqLens is the exclusive prefix sum of
//! kvCacheStartIndices[b] + inputSeqLen[b], so it includes the complete logical KV history for chunked prefill. The
//! outputs can be passed unchanged to either the optimized Blackwell or FMHA-v2 ragged paged backend.
void calCuQCuKVSeqLens(rt::Tensor const& inputSeqLen, rt::Tensor const& kvCacheStartIndices, rt::Tensor& cuQSeqLens,
    rt::Tensor& cuKVSeqLens, cudaStream_t stream);

//! \brief Compute sequence metadata for paged SWA chunked prefill.
//!
//! The temporary KV source contains the previous resident window followed by the current chunk. The KV prefix sums
//! therefore use `min(kvCacheStartIndices[b], slidingWindowSize) + inputSeqLen[b]`. `kvCacheEndIdxs` uses the padded
//! runtime sequence length so the generic RoPE/write kernel assigns position `start + tokenOffset` to every row.
//! `paddedCuKVSeqLens` uses the same resident prefix plus `runtimeSeqLen`, which preserves the causal offset when
//! ragged chunks are padded to the maximum query length.
void calSWAChunkedPrefillMetadata(rt::Tensor const& inputSeqLen, rt::Tensor const& kvCacheStartIndices,
    rt::Tensor& cuQSeqLens, rt::Tensor& cuKVSeqLens, rt::Tensor& kvCacheEndIdxs, rt::Tensor& paddedCuKVSeqLens,
    int32_t runtimeSeqLen, int32_t slidingWindowSize, cudaStream_t stream);

//! \brief Assemble split FP16 K/V for SWA chunked prefill from a paged resident window plus the current chunk.
//!
//! For each batch row, logical tokens `[start - min(start, W), start)` are gathered through `swaPageTable`; newly
//! roped K/V from the current chunk are appended directly. The outputs have shape `[B, W + S, Hkv, D]`, are padded
//! with zeros, and are consumed with cu-seqlens from calSWAChunkedPrefillMetadata(). FP8 resident pages are
//! dequantized with their K/V scales; current K/V and the workspaces remain FP16. The persistent pool remains bounded
//! independently of the maximum sequence length.
void assemblePagedSWAChunkedPrefillFMHAKV(rt::Tensor const& swaPool, rt::Tensor const& swaPageTable,
    rt::Tensor const& k, rt::Tensor const& v, rt::Tensor const& inputSeqLen, rt::Tensor const& kvCacheStartIndices,
    rt::Tensor& kWorkspace, rt::Tensor& vWorkspace, int32_t slidingWindowSize, float kScale, float vScale,
    cudaStream_t stream);

//! \brief Assemble the K/V of the prefill sequences of a compact phase 8 step for the paged ragged FMHA.
//!
//! Prefill sequence b owns query rows `[queryStartOffsets[b], queryStartOffsets[b + 1])` and KV extent
//! `kvEndLengths[b]`. Its in-window history `[max(0, past - W), past)` is read through `swaPageTable` (FP8 pages
//! dequantized, unmapped pages read as zero) and its current rows from the token-major `k`/`v` scratch `[T_p, Hkv, D]`.
//! Tokens keep their absolute in-page positions: pages `max(0, past - W) / 128` up to the page holding the last token
//! are copied to `assembledPool` `[2, P, 128, Hkv, D]` after the pages of earlier sequences, and
//! `assembledPageTable` `[Np, 2, maxPagesPerSeq]` maps every logical page into that range (clamped at both ends; V
//! entries add P). The FMHA therefore reads the assembly with the original KV extents and window. P must cover the
//! sum of the spans, e.g. `min(Np * maxPagesPerSeq, swaCompactAssembledPages(Np, T_p, W))`.
void assemblePagedSWACompactPrefillFMHAKV(rt::Tensor const& swaPool, int32_t const* swaPageTable,
    int32_t maxPagesPerSeq, rt::Tensor const& k, rt::Tensor const& v, int32_t const* queryStartOffsets,
    int32_t const* kvEndLengths, int32_t numSequences, int32_t slidingWindowSize, float kScale, float vScale,
    rt::Tensor& assembledPool, int32_t* assembledPageTable, cudaStream_t stream);

//! Upper bound on the assembled pages of assemblePagedSWACompactPrefillFMHAKV for Np sequences with T_p query rows:
//! each sequence spans at most (W + q_b) / 128 + 2 pages.
constexpr int64_t swaCompactAssembledPages(int64_t numSequences, int64_t numQueryRows, int64_t slidingWindowSize)
{
    return (numSequences * (slidingWindowSize + 2 * rt::kTOKENS_PER_PAGE) + numQueryRows + rt::kTOKENS_PER_PAGE - 1)
        / rt::kTOKENS_PER_PAGE;
}
} // namespace kernel
} // namespace trt_edgellm
