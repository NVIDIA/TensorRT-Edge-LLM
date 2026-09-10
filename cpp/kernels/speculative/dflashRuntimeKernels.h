/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace trt_edgellm
{
namespace kernel
{

/// Launch the DFlash target KV cache update kernel.
///
/// Applies RoPE to k_delta and writes k_rope + v_delta into the paged KV pool
/// at positions [deltaStart, deltaStart + deltaLen) for each batch element.
///
/// @param kDelta       [B, deltaLen, numKVHeads, headDim] FP16, k_normed, no RoPE
/// @param vDelta       [B, deltaLen, numKVHeads, headDim] FP16
/// @param kvCache      Paged KV pool [2, numPages, kTOKENS_PER_PAGE, numKVHeads, headDim] FP16 (in/out).
/// @param cosSinCache  [cosSinBatch, cosSinSeqLen, rotaryDim] FP32
/// @param deltaStartPositions [B] INT32
/// @param pageTable    [B, 2, maxPagesPerSeq] canonical page ids: K in [0, numPages), V in
///                     [numPages, 2 * numPages). Unmapped or out-of-plane ids skip that cache plane.
/// @param batchSize    ACTIVE batch size
/// @param deltaLen     number of delta tokens per batch
/// @param numKVHeads   number of KV heads
/// @param headDim      head dimension
/// @param rotaryDim    rotary embedding dimension
/// @param cosSinBatch  cos/sin cache batch size (1 or B)
/// @param cosSinSeqLen cos/sin cache sequence length
/// @param numPages     Number of physical pages in each KV plane
/// @param maxPagesPerSeq Logical pages per sequence; positions outside this capacity are skipped
/// @param stream       CUDA stream
/// @param deltaLengths  [B] INT32, per-batch delta lengths (skip t >= deltaLengths[b])
void launchDFlashTargetKVCacheUpdate(half const* kDelta, half const* vDelta, half* kvCache, float const* cosSinCache,
    int32_t const* deltaStartPositions, int32_t const* deltaLengths, int32_t const* pageTable, int32_t batchSize,
    int32_t deltaLen, int32_t numKVHeads, int32_t headDim, int32_t rotaryDim, int32_t cosSinBatch, int32_t cosSinSeqLen,
    int32_t numPages, int32_t maxPagesPerSeq, cudaStream_t stream);

/// Validate that a RoPE cos/sin cache covers every position DFlash's target-KV update can write.
///
/// `kvCapacity` is the page-aligned upper bound on every writable position. The RoPE cache tracks the
/// unpadded configured maximum and may therefore be smaller; reject only `cosSinSeqLen > kvCapacity`.
///
/// @param cosSinSeqLen Sequence length of the bound rope_cos_sin cache
/// @param kvCapacity   KV pool's padded per-slot capacity (capPadded)
/// @throws std::runtime_error if cosSinSeqLen > kvCapacity
void checkDFlashRopeCapacity(int32_t cosSinSeqLen, int32_t kvCapacity);

/// Launch kernel to prepare DFlash proposal attention inputs.
///
/// Computes target_len_after_delta = oldDraftCacheLengths[b] + deltaLen, then sets:
///   attention_pos_id[b, i] = target_len_after_delta + i
///   context_lengths[b] = target_len_after_delta + blockSize
///   packed_attention_mask: full non-causal within proposal block, or causal rows when requested
///
/// @param oldDraftCacheLengths [B] INT32 — draft cache lengths BEFORE delta (GPU)
/// @param deltaLengths [B] INT32 — per-batch delta token count (GPU)
/// @param blockSize   DFlash block size (BS)
/// @param packedAttentionMask [B, BS, divUp(BS,32)] INT32 — output
/// @param attentionPosId      [B, BS] INT32 — output
/// @param contextLengths      [B] INT32 — output
/// @param causalProposalMask  true => row i attends only to proposal positions [0, i]
/// @param batchSize    batch size
/// @param stream       CUDA stream
void launchDFlashPrepareProposalInputs(int32_t const* oldDraftCacheLengths, int32_t const* deltaLengths,
    int32_t blockSize, int32_t* packedAttentionMask, int32_t* attentionPosId, int32_t* contextLengths,
    bool causalProposalMask, int32_t batchSize, cudaStream_t stream);

/// Launch kernel to prepare DFlash base verification attention inputs.
///
/// DFlash verifies a linear block, so the base tree mask is always causal:
/// token i attends to proposal tokens [0, i]. This writes the packed INT32 mask
/// consumed by AttentionPlugin directly, without materializing an intermediate
/// unpacked [B, BS, BS] INT8 mask.
///
/// @param baseKVCacheLengths [B] INT32 — committed base cache lengths (GPU)
/// @param verifySize DFlash verify block size (BS)
/// @param packedAttentionMask [B, BS, divUp(BS,32)] INT32 — output
/// @param attentionPosId [B, BS] INT32 — output
/// @param selectTokenIndices [B, BS] INT64 — output
/// @param contextLengths [B] INT32 — output
/// @param batchSize batch size
/// @param stream CUDA stream
void launchDFlashPrepareBaseVerifyInputs(int32_t const* baseKVCacheLengths, int32_t verifySize,
    int32_t* packedAttentionMask, int32_t* attentionPosId, int64_t* selectTokenIndices, int32_t* contextLengths,
    int32_t batchSize, cudaStream_t stream);

/// Launch kernel to build DFlash linear verification inputs for EAGLE accept.
///
/// verifyTokenIds[b, 0] = lastAcceptedTokens[b];
/// verifyTokenIds[b, j + 1] = draftTokenIds[b, j], j in [0, proposalLen).
/// verifyTreeMask is an unpacked causal tree mask where row i attends to [0, i].
///
/// @param lastAcceptedTokens [B] INT32 — last committed token per batch
/// @param draftTokenIds [B, draftTokenStride] INT32 — DFlash draft argmax token IDs
/// @param verifyTokenIds [B, verifySize] INT32 — output base verify token IDs
/// @param verifyTreeMask [B, verifySize, verifySize] INT8 — output EAGLE-style causal tree mask
/// @param batchSize batch size
/// @param proposalLen number of proposal tokens copied after the anchor
/// @param draftTokenStride row stride of draftTokenIds, usually dflashBlockSize
/// @param verifySize base verification input size, must equal proposalLen + 1
/// @param stream CUDA stream
void launchDFlashBuildLinearVerifyInputs(int32_t const* lastAcceptedTokens, int32_t const* draftTokenIds,
    int32_t* verifyTokenIds, int8_t* verifyTreeMask, int32_t batchSize, int32_t proposalLen, int32_t draftTokenStride,
    int32_t verifySize, cudaStream_t stream);

} // namespace kernel
} // namespace trt_edgellm
