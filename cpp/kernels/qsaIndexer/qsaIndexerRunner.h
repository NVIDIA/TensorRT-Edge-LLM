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

//! \brief Workspace bytes for the unified initial/chunk prefill indexer.
//! Each slot is 128-byte aligned: normalized queries [B,S,4,128] T; dense kbar [B,MB,128] T
//! only when !paged; two FP32 logits buffers and two INT32 id buffers [Rc,MB]; visible block
//! counts [Rc] INT32; CUB temp. Each sort segment covers only its device-computed visible prefix.
//! The counts allocation bound reserves B*S entries to cover larger Rc at smaller capacities.
//! Rc = min(B*S, max(8388608/MB, 256)); sort slots use the monotone allocation bound
//! min(B*S*MB, max(8388608, 256*MB)) to cover smaller profile shapes. The expanded output
//! [B,S,2051] INT32 is supplied separately. T is 2 bytes (FP16/BF16).
//! \param[in] batchSize Maximum batch size B
//! \param[in] seqLen Maximum padded query length S
//! \param[in] maxBlocks Score row width MB: ceilDiv(S,4) for a known empty prefix, or
//!            ceilDiv(max total KV length,4) when cached prefixes are present
//! \param[in] paged Whether the runner receives non-null poolState; paged scoring gathers
//!            kbar directly from the cache and needs no dense kbar workspace
//! \return Required workspace bytes
//! \throws std::runtime_error on invalid dimensions
size_t getQsaIndexerWorkspaceSize(int32_t batchSize, int32_t seqLen, int32_t maxBlocks, bool paged);

//! \brief Run initial or chunk prefill through one query-prep, sort and expansion pipeline.
//! Each query selects complete blocks visible at its own absolute position, appends the
//! causal incomplete tail and writes absolute token indices. With pastLengths, the cached
//! prefix's last blocks are resealed first; newly completed groups combine cached raw K with
//! current input. Every query row's tails are written to the pool (qsaIndexerKernels.h,
//! "Pool tails").
//! \tparam T Activation type: half or __nv_bfloat16
//! \param[out] outIdx Device INT32 [batchSize,seqLen,2051]; padded queries are all -1
//! \param[in] indexQk Device T [batchSize,seqLen,640], current query rows only
//! \param[in] cosSin Device FP32 [batchSize,seqLen,64], token-aligned rope rows at absolute
//!            positions pastLengths[b]+t; padded input rows are never read
//! \param[in] queryLengths Device INT32 [batchSize], valid current rows in [0,seqLen]
//! \param[in] pastLengths Optional device INT32 [batchSize] cached prefix lengths;
//!            nullptr means every prefix is empty, without reading any lengths buffer
//! \param[in] wQ Device T [128], raw checkpoint index-Q norm gamma (kernel adds 1)
//! \param[in] wK Device T [128], raw checkpoint index-K norm gamma (kernel adds 1)
//! \param[in] rmsEps RMS norm epsilon
//! \param[in] poolState Optional paged cache. Null supports standalone index selection with
//!            an empty prefix only, and uses dense kbar internally without persisting state
//! \param[in] workspace Device scratch, 128-byte aligned and at least the size returned by
//!            getQsaIndexerWorkspaceSize(batchSize,seqLen,maxBlocks,poolState!=nullptr)
//! \param[in] workspaceBytes Size of workspace in bytes
//! \param[in] batchSize Batch size B
//! \param[in] seqLen Padded current query length S
//! \param[in] maxBlocks With null pastLengths, must be >= ceilDiv(seqLen,4), including ragged inputs.
//!            Otherwise must cover ceilDiv(pastLengths[b]+queryLengths[b],4) for every entry;
//!            the caller supplies this device-length bound without a host synchronization.
//! \param[in] stream CUDA stream ordering all launches, with no allocation or host token loop
//! \throws std::runtime_error on invalid host arguments, insufficient workspace or launch failure
template <typename T>
void runQsaIndexerPrefill(int32_t* outIdx, T const* indexQk, float const* cosSin, int32_t const* queryLengths,
    int32_t const* pastLengths, T const* wQ, T const* wK, float rmsEps, QsaIndexerPoolState const* poolState,
    void* workspace, size_t workspaceBytes, int32_t batchSize, int32_t seqLen, int32_t maxBlocks, cudaStream_t stream);

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
//!   B1 launchQsaIndexerPreDecode: q-prep at position contextLengths[b] - 1, the new
//!      token's tail write, and the compress of the block it completes;
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
//! \param[in] cosSin Device FP32 [batchSize, 64] token-aligned rope rows: row b is the
//!            cos/sin of sequence b's new token (position contextLengths[b] - 1)
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

//! Workspace size for tree verification, with 128-byte aligned slots. T must be two bytes.
//! Dimensions must be positive, S <= 128 and B*S must fit INT32.
size_t getQsaIndexerVerifyWorkspaceSize(int32_t batchSize, int32_t seqLen, int32_t maxBlocks);

//! Verify each root-to-node path with ordinary QSA selection. First reseals the last blocks of [0, pastLengths[b]);
//! no other committed cell changes. Valid nodes precede padding and are topologically ordered from root 0 (parent -1,
//! depth 0). Candidate KV slots are pastLengths[b] + nodeId; cosSin uses pastLengths[b] + depths[b,s].
//! Each valid node's tails go to its slot pastLengths[b] + nodeId as if it sat at position pastLengths[b] + depth, so
//! moving an accepted node's whole row to that position commits them. outIdx [B,S,2051] holds physical slots (-1
//! padded); workspace may be reused after verify. A node with invalid metadata, capacity or pages selects no blocks and
//! writes nothing.
//! Buffers must not overlap; use a 128-byte-aligned workspace sized by getQsaIndexerVerifyWorkspaceSize.
//! maxBlocks must cover ceilDiv(pastLengths[b] + depths[b,s] + 1, 4).
//! Warm up ensureQsaIndexerKernelsLoaded<T>() before CUDA graph capture.
template <typename T>
void runQsaIndexerVerify(int32_t* outIdx, T const* indexQk, float const* cosSin, int32_t const* parents,
    int32_t const* depths, int32_t const* queryLengths, int32_t const* pastLengths, T const* wQ, T const* wK,
    float rmsEps, QsaIndexerPoolState const& poolState, void* workspace, size_t workspaceBytes, int32_t batchSize,
    int32_t seqLen, int32_t maxBlocks, cudaStream_t stream);

} // namespace kernel
} // namespace trt_edgellm
