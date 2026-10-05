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

#include "qsaIndexerKernels.h"

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

namespace trt_edgellm
{
namespace kernel
{

//! \brief Workspace bytes required by runQsaIndexerPrefill.
//!
//! Workspace layout (each slot 128-byte aligned; MB = ceilDiv(maxSeqLen, 4),
//! Rc = min(batchSize * maxSeqLen, max(8388608 / MB, 256)) row-chunk size — the sort
//! keys buffer is capped at 32 MiB so the sort working set stays bounded for any maxSeqLen):
//!
//!   Slot  | Shape                  | Type  | Used by
//!   ------+------------------------+-------+------------------------------------------
//!   0     | [B, S, 4, 128]         | T     | qNormed        (K1a out, K2 in)
//!   1     | [B, MB, 128]           | T     | kbar           (K1b out, K2 in)
//!   2     | [Rc, MB]               | FP32  | logits         (K2 out, sort keys current)
//!   3     | [Rc, MB]               | FP32  | sort keys alternate (cub::DoubleBuffer)
//!   4     | [Rc, MB]               | INT32 | block ids      (idsFill out, sort values current)
//!   5     | [Rc, MB]               | INT32 | sort values alternate (cub::DoubleBuffer)
//!   6     | [cubTempBytes]         | BYTE  | cub::DeviceSegmentedRadixSort temp storage
//!
//! T is 2 bytes (half / bfloat16 share the size, so the layout is dtype-agnostic).
//! Slots 2-5 are sized with the monotone bound min(B*S*MB, max(8388608, 256*MB)) rather
//! than the runtime Rc*MB (which is not monotone in seqLen because floor(8388608/MB)*MB
//! peaks when MB divides 2^23); the bound dominates the runtime carve at every
//! seqLen <= maxSeqLen, so a workspace sized for (batchSize, maxSeqLen) covers every run
//! with the same batchSize and seqLen <= maxSeqLen.
//!
//! The expanded index output outIdx [B, S, 2051] INT32 is NOT part of this workspace; the
//! caller provides it separately (the plugin carves it from its own workspace).
//!
//! \param[in] batchSize Maximum batch size B
//! \param[in] maxSeqLen Maximum padded sequence length S
//! \return Required workspace bytes (128-byte-aligned slots)
//! \throws std::runtime_error on invalid arguments
size_t getQsaIndexerWorkspaceSize(int32_t batchSize, int32_t maxSeqLen);

//! \brief Run the full QSA indexer prefill pipeline.
//!
//! Stages: K1a q-prep; K1b k-compress; then per row chunk of Rc rows
//! {idsFill; K2 scores; cub::DeviceSegmentedRadixSort::SortPairsDescending; K4 expand}.
//! On return (asynchronous on `stream`), outIdx[b][t] holds the selected token indices for
//! query row (b, t): min(512, (t+1)/4) blocks expanded to 4 tokens each, packed to the
//! front in descending-logit block order, followed by the (t+1)%4 causal tail tokens
//! [4*((t+1)/4), t], -1 padded to width 2051. Padding rows (t >= contextLengths[b]) are
//! all -1. See qsaIndexerKernels.h for the per-kernel numerics contracts.
//!
//! \tparam T Activation type: half or __nv_bfloat16.
//! \param[out] outIdx Device INT32 [batchSize, seqLen, 2051] (kQSA_INDEX_WIDTH)
//! \param[in] indexQk Device T [batchSize, seqLen, 640]: 4 q-heads x 128 then raw k x 128 per row
//! \param[in] cosSin Device FP32 [maxPositions, 64] rope table, maxPositions >= seqLen;
//!            cos in columns [0, 32), sin in [32, 64) (rotaryDim 64 of the 128 head dims)
//! \param[in] contextLengths Device INT32 [batchSize], 0 <= contextLengths[b] <= seqLen
//! \param[in] wQ Device T [128] indexer q-norm gamma AS STORED IN THE CHECKPOINT (raw w;
//!            the kernels compute (1.0f + w) — do NOT pre-fold)
//! \param[in] wK Device T [128] indexer k-norm gamma, same raw-w convention
//! \param[in] rmsEps RMS norm epsilon (1e-6f for qwen4_exp)
//! \param[in] workspace Device scratch, >= getQsaIndexerWorkspaceSize(batchSize, seqLen)
//!            bytes and 128-byte aligned
//! \param[in] workspaceBytes Size of `workspace` in bytes
//! \param[in] batchSize Batch size B
//! \param[in] seqLen Padded sequence length S
//! \param[in] poolState Optional paged indexer state (see QsaIndexerPoolState). When
//!            non-null the pipeline ALSO persists decode state into the pool tails —
//!            every complete block's kbar (paged K1b variant) and the raw index-K of the
//!            trailing incomplete block's tokens (launchQsaRawKTailWritePrefill) — without
//!            changing outIdx or any dense numerics. Pass nullptr for the index lists alone.
//! \param[in] stream CUDA stream
//! \throws std::runtime_error on invalid arguments, insufficient workspace, or launch failure
template <typename T>
void runQsaIndexerPrefill(int32_t* outIdx, T const* indexQk, float const* cosSin, int32_t const* contextLengths,
    T const* wQ, T const* wK, float rmsEps, void* workspace, size_t workspaceBytes, int32_t batchSize, int32_t seqLen,
    QsaIndexerPoolState const* poolState, cudaStream_t stream);

//! \brief Workspace bytes required by runQsaIndexerDecode.
//!
//! Workspace layout (each slot 128-byte aligned; T is 2 bytes — half / bfloat16 share the
//! size, so the layout is dtype-agnostic):
//!
//!   Slot  | Shape             | Type  | Used by
//!   ------+-------------------+-------+--------------------------------------------
//!   0     | [B, 1, 4, 128]    | T     | qNormed  (B1 out, B2 in)
//!   1     | [B, maxBlocks]    | FP32  | logits   (B2 out, B3 in; valid prefix only)
//!
//! \param[in] maxBatchSize Maximum batch size B
//! \param[in] maxBlocks Maximum compressed-block count: ceilDiv(max KV capacity, 4)
//! \return Required workspace bytes (128-byte-aligned slots)
//! \throws std::runtime_error on invalid arguments
size_t getQsaIndexerDecodeWorkspaceSize(int32_t maxBatchSize, int32_t maxBlocks);

//! \brief Run the full QSA indexer decode pipeline (one token per sequence).
//!
//! Stages (all batchSize-wide, graph-safe, no cub, no allocation):
//!   B1 launchQsaIndexerPreDecode: q-prep at position contextLengths[b] - 1, then either the
//!      raw-K tail write for the new token (block still incomplete) or the compress of the
//!      block it completes;
//!   B2 launchQsaIndexScoresDecode: block logits from the paged kbar V-tails;
//!   B3 launchQsaTopKExpandDecode: split-counter zeroing + deterministic top-512 radix
//!      select + K4 expand.
//! On return (asynchronous on `stream`), outIdx[b][0] holds the selected token indices for
//! the decode row of sequence b: min(512, ctx/4) blocks expanded to 4 tokens each, packed
//! to the front, followed by the ctx%4 causal tail tokens [4*(ctx/4), ctx), -1 padded to
//! width 2051 (ctx = contextLengths[b] INCLUDING the new token). Two identical runs are
//! bitwise equal, and the selected block set matches runQsaIndexerPrefill for the same
//! logits (identical tie behavior). See qsaIndexerKernels.h for per-kernel contracts.
//!
//! \tparam T Activation type: half or __nv_bfloat16.
//! \param[out] outIdx Device INT32 [batchSize, 1, 2051] (kQSA_INDEX_WIDTH)
//! \param[in] indexQk Device T [batchSize, 1, 640]: 4 q-heads x 128 then raw k x 128 per row
//! \param[in] cosSin Device FP32 [maxPositions, 64] rope table, maxPositions >= max context
//! \param[in] contextLengths Device INT32 [batchSize] TOTAL lengths including the new token;
//!            every decode row is active (contextLengths[b] >= 1)
//! \param[in] wQ Device T [128] indexer q-norm gamma (raw checkpoint w, not 1+w)
//! \param[in] wK Device T [128] indexer k-norm gamma, same raw-w convention
//! \param[in] rmsEps RMS norm epsilon (1e-6f for qwen4_exp)
//! \param[in] poolState Paged indexer state; see QsaIndexerPoolState
//! \param[in,out] splitCounters Device INT32 [batchSize, poolState.numKVHeads] split-K
//!            merge counters of the decode attention, zeroed by B3 every step; may be
//!            nullptr when the attention path does not need them
//! \param[in] workspace Device scratch, >= getQsaIndexerDecodeWorkspaceSize(batchSize,
//!            maxBlocks) bytes and 128-byte aligned
//! \param[in] workspaceBytes Size of `workspace` in bytes
//! \param[in] batchSize Batch size B
//! \param[in] maxBlocks Logits row stride; must cover ceilDiv(max context, 4)
//! \param[in] stream CUDA stream
//! \throws std::runtime_error on invalid arguments, insufficient workspace, or launch failure
template <typename T>
void runQsaIndexerDecode(int32_t* outIdx, T const* indexQk, float const* cosSin, int32_t const* contextLengths,
    T const* wQ, T const* wK, float rmsEps, QsaIndexerPoolState const& poolState, int32_t* splitCounters,
    void* workspace, size_t workspaceBytes, int32_t batchSize, int32_t maxBlocks, cudaStream_t stream);

} // namespace kernel
} // namespace trt_edgellm
