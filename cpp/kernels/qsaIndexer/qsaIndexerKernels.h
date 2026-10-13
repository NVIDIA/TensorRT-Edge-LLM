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
#include <cuda_runtime.h>

namespace trt_edgellm
{
namespace kernel
{

//! QSA indexer geometry (Qwen sparse attention; fixed by the qwen4_exp architecture).
//! The indexer scores compressed KV blocks with a small weight-free attention head and
//! selects the top kQSA_BLOCK_TOPK blocks per query token.
constexpr int32_t kQSA_INDEXER_NUM_HEADS = 4;  //!< Index-Q heads.
constexpr int32_t kQSA_INDEXER_HEAD_DIM = 128; //!< Index-Q / index-K head dimension.
constexpr int32_t kQSA_COMPRESS_RATIO = 4;     //!< Raw index-K tokens averaged per compressed block.
constexpr int32_t kQSA_BLOCK_TOPK = 512;       //!< Blocks selected per query row.
//! Expanded tokens of the selected blocks.
constexpr int32_t kQSA_INDEX_BUDGET = kQSA_BLOCK_TOPK * kQSA_COMPRESS_RATIO;
//! kQSA_INDEX_BUDGET plus up to (kQSA_COMPRESS_RATIO - 1) causal tail tokens.
constexpr int32_t kQSA_INDEX_WIDTH = kQSA_INDEX_BUDGET + kQSA_COMPRESS_RATIO - 1;

//! Launch geometry shared with the NVRTC source (handed over as -D defines by
//! qsaIndexerJitCompiler.cpp): the K2/B2 block-score kernels give each warp
//! kQSA_INDEXER_SCORES_COLUMNS_PER_WARP block-columns of a kQSA_INDEXER_SCORES_COLUMNS_PER_CTA
//! tile; the B3 radix top-k select + expand kernel runs kQSA_INDEXER_TOPK_THREADS per CTA.
constexpr int32_t kQSA_INDEXER_SCORES_COLUMNS_PER_CTA = 64;
constexpr int32_t kQSA_INDEXER_SCORES_COLUMNS_PER_WARP = 8;
constexpr int32_t kQSA_INDEXER_SCORES_THREADS
    = kQSA_INDEXER_SCORES_COLUMNS_PER_CTA / kQSA_INDEXER_SCORES_COLUMNS_PER_WARP * 32; // one warp per column group
constexpr int32_t kQSA_INDEXER_TOPK_THREADS = 512;

//! Rotary geometry of the indexer heads: partial neox (non-interleaved) rope over the first
//! 64 of 128 dims; dims [64, 128) are passed through unrotated. The cos/sin table row layout
//! is [maxPositions, 64] with cos in columns [0, 32) and sin in columns [32, 64), matching
//! initializeNormalRopeCosSin (rotaryDim = 64).
constexpr int32_t kQSA_INDEXER_ROTARY_DIM = 64;

//! Where the QSA indexer's persistent state lives. Kernels are state-location-agnostic:
//! they receive this struct and compute addresses from it; only plugin launch sites know
//! the state is in the widened-pool tails.
//!
//! Addressing contract (pool laid out [2 * numPages, 128, numKVHeads, poolHeadDim] T):
//! element (page, row, head, col) lives at element offset
//!   page * 128 * numKVHeads * poolHeadDim + row * numKVHeads * poolHeadDim
//!       + head * poolHeadDim + col.
//! - Raw index-K of token t, valid only while t's 4-token block is incomplete (token 4g+3 is
//!   never written; rows of completed blocks are stale and never read): K plane, page
//!   pageTable[b][0][t >> 7], row t & 127, head 0, columns [headSize, headSize + 128) — the
//!   token's indexQk columns [512, 640) stored BIT-UNMODIFIED (keeps the decode compress
//!   bit-identical to prefill K1b).
//! - kbar of block g: V plane, page pageTable[b][1][(4 * g) >> 7] (the id already includes
//!   the +numPages V-plane offset — do NOT re-add it), row (4 * g) & 127, head 0, same
//!   columns.
//! Page-table entries equal to kUNUSED_PAGE_ENTRY (-1, cpp/common/pagedKvTypes.h) are
//! guarded BEFORE dereferencing; kernels treat them as "nothing persistable here".
struct QsaIndexerPoolState
{
    void* poolPtr;            //!< [2*numPages, 128, numKVHeads, poolHeadDim] T base
    int32_t const* pageTable; //!< [B, 2, maxPagesPerSeq]; V ids pre-offset +numPages
    int32_t maxPagesPerSeq;
    int32_t numPages;    //!< K-plane page count
    int32_t numKVHeads;  //!< 2
    int32_t poolHeadDim; //!< 384
    int32_t headSize;    //!< 256 == tail column offset
};

//! \brief Compile (NVRTC) and driver-load the QSA indexer kernels for activation type T.
//!
//! The device code lives in kernelSrcs/qsaIndexer/qsaIndexerJitKernels.cu and every launcher
//! below JIT-compiles and cuModuleLoadData's it lazily on first use, cached process-wide per
//! (CUDA context, SM version, dtype). Those one-time host-side steps must not happen inside a
//! CUDA Graph capture, so callers that capture the launchers (or enqueue them from a TRT
//! plugin whose stream may be captured) must call this warmup first from a non-captured
//! context, e.g. onShapeChange; after it returns, the launchers only issue cuLaunchKernel.
//!
//! \tparam T Activation type: half or __nv_bfloat16.
//! \throws std::runtime_error on NVRTC or CUDA driver failure
template <typename T>
void ensureQsaIndexerKernelsLoaded();

//! \brief K1a: Gemma-style qk-norm + partial neox rope on the index-Q heads.
//!
//! Reads the packed indexer projection `indexQk` [batchSize, seqLen, 640] where each row is
//! 4 q-heads x 128 dims followed by 1 raw k x 128 dims, and writes the normalized, roped
//! q heads to `qNormed` [batchSize, seqLen, 4, 128].
//!
//! Exact numerics (FP32 unless stated): per (b, t, h) head, meanSq = sum(x_f32^2) / 128 via
//! warp reduction; y = x_f32 * rsqrtf(meanSq + rmsEps) * (1.0f + w_f32) with w the RAW
//! checkpoint gamma (the kernel adds the +1, do NOT pre-fold); then neox rope on dims
//! [0, 64): pair (i, i+32) -> (x0*cos - x1*sin, x1*cos + x0*sin) with cos = cosSin[t][i],
//! sin = cosSin[t][32 + i]; single cast to T at the very end.
//!
//! Padding semantics: rows with t >= contextLengths[b] are written as exact zeros; their
//! `indexQk` contents are never read (NaN-safe).
//!
//! Launch: grid(batchSize * seqLen), block(128) = 4 warps, warp h owns head h, 4 dims/lane.
//!
//! \tparam T Activation type: half or __nv_bfloat16.
//! \param[out] qNormed [batchSize, seqLen, 4, 128] T, contiguous, naturally aligned
//! \param[in] indexQk [batchSize, seqLen, 640] T (4 q-heads x 128 then raw k x 128 per row)
//! \param[in] cosSin FP32 [maxPositions, 64] with maxPositions >= seqLen; cos [0:32], sin [32:64]
//! \param[in] contextLengths Device INT32 [batchSize], 0 <= contextLengths[b] <= seqLen
//! \param[in] wQ T [128] indexer q-norm gamma as stored in the checkpoint (raw w, not 1+w)
//! \param[in] rmsEps RMS norm epsilon (1e-6f for qwen4_exp)
//! \param[in] batchSize Batch size B
//! \param[in] seqLen Padded sequence length S
//! \param[in] stream CUDA stream
//! \throws std::runtime_error on invalid arguments or launch failure
template <typename T>
void launchQsaIndexQPrep(T* qNormed, T const* indexQk, float const* cosSin, int32_t const* contextLengths, T const* wQ,
    float rmsEps, int32_t batchSize, int32_t seqLen, cudaStream_t stream);

//! \brief K1b: compress raw index-K into per-block keys, then Gemma-norm + partial neox rope.
//!
//! For each (b, g) with g in [blockBegin, blockEnd): the block is valid iff it is complete
//! and fully inside the provided rows, i.e. 4*g + 3 < contextLengths[b] and 4*g >= pastLen.
//! Valid blocks compute, per dim d:
//!   mean = (((k0 + k1) + k2) + k3) * 0.25f      (FP32, FIXED sequential order)
//!   cast mean to T, re-load as FP32              (bit-compatibility-critical cast)
//!   y = m_f32 * rsqrtf(meanSq + rmsEps) * (1.0f + w_f32)   (meanSq over the 128 cast values)
//!   neox rope at position 4*g (same pairing/table layout as launchQsaIndexQPrep)
//!   cast to T at the very end.
//! kj is the raw index-K row of token 4*g + j, read from indexQk row (4*g + j - pastLen),
//! columns [512, 640). Invalid blocks are written as exact zeros and never read (NaN-safe).
//!
//! The prefill pipeline passes blockBegin = 0, blockEnd = numBlocks = ceilDiv(seqLen, 4), pastLen = 0.
//! The (blockBegin, blockEnd, pastLen) triplet compresses a sub-range of blocks of a longer
//! sequence (the decode bit-parity gtest relies on it); the decode pipeline itself compresses
//! the newly completed block in-kernel (B1, launchQsaIndexerPreDecode).
//!
//! Launch: grid(batchSize * (blockEnd - blockBegin)), block(32) = one warp per block, 4 dims/lane.
//!
//! \tparam T Activation type: half or __nv_bfloat16.
//! \param[out] kbar [batchSize, numBlocks, 128] T; only rows [blockBegin, blockEnd) are written
//! \param[in] indexQk [batchSize, seqLen, 640] T (raw k occupies columns [512, 640))
//! \param[in] cosSin FP32 [maxPositions, 64] with maxPositions >= 4 * blockEnd - 3
//! \param[in] contextLengths Device INT32 [batchSize] TOTAL context length (past + new)
//! \param[in] wK T [128] indexer k-norm gamma as stored in the checkpoint (raw w, not 1+w)
//! \param[in] rmsEps RMS norm epsilon
//! \param[in] batchSize Batch size B
//! \param[in] seqLen Number of token rows present in indexQk
//! \param[in] numBlocks Row stride of kbar: ceilDiv(maxTotalLen, 4) (== ceilDiv(seqLen, 4) in prefill)
//! \param[in] blockBegin First block index to compute (inclusive)
//! \param[in] blockEnd Last block index to compute (exclusive), blockEnd <= numBlocks
//! \param[in] pastLen Tokens already in the cache before indexQk row 0 (0 for prefill)
//! \param[in] stream CUDA stream
//! \throws std::runtime_error on invalid arguments or launch failure
template <typename T>
void launchQsaIndexKCompress(T* kbar, T const* indexQk, float const* cosSin, int32_t const* contextLengths, T const* wK,
    float rmsEps, int32_t batchSize, int32_t seqLen, int32_t numBlocks, int32_t blockBegin, int32_t blockEnd,
    int32_t pastLen, cudaStream_t stream);

//! \brief K2: block-relevance logits for a chunk of query rows.
//!
//! For global rows r = b * seqLen + t in [rowStart, rowStart + numRows) writes
//! logits[r - rowStart][g] for all g in [0, numBlocks):
//!   logits = (1 / sqrtf(128)) * sum_{h < 4} fmaxf(0.f, sum_d q[r][h][d] * kbar[b][g][d])
//! FP32 dot per head, relu PER HEAD, sum over heads (ascending h), scale AFTER the sum.
//! Causal visibility: g >= nVis(t) = (t + 1) / 4 is masked to -FLT_MAX. Padded rows
//! (t >= contextLengths[b] or contextLengths[b] == 0) write -FLT_MAX for the whole row and
//! never read q or kbar (NaN-safe).
//!
//! Launch: grid(numRows, ceilDiv(numBlocks, 64)), block(256); the CTA stages the row's
//! 4 x 128 q values in FP32 shared memory (2 KB); each warp evaluates one block-column at
//! a time via lane-strided FP32 FMA + warp shuffle reduction.
//!
//! \tparam T Activation type: half or __nv_bfloat16.
//! \param[out] logits FP32 [numRows, numBlocks] chunk-local rows
//! \param[in] qNormed [batchSize, seqLen, 4, 128] T from launchQsaIndexQPrep
//! \param[in] kbar [batchSize, numBlocks, 128] T from launchQsaIndexKCompress
//! \param[in] contextLengths Device INT32 [batchSize]
//! \param[in] batchSize Batch size B
//! \param[in] seqLen Padded sequence length S
//! \param[in] numBlocks ceilDiv(seqLen, 4)
//! \param[in] rowStart First global row of the chunk
//! \param[in] numRows Rows in the chunk; rowStart + numRows <= batchSize * seqLen
//! \param[in] stream CUDA stream
//! \throws std::runtime_error on invalid arguments or launch failure
template <typename T>
void launchQsaIndexScores(float* logits, T const* qNormed, T const* kbar, int32_t const* contextLengths,
    int32_t batchSize, int32_t seqLen, int32_t numBlocks, int32_t rowStart, int32_t numRows, cudaStream_t stream);

//! \brief Fill per-segment block ids for the segmented sort: ids[r * numBlocks + c] = c.
//!
//! \param[out] ids INT32 [numRows, numBlocks]
//! \param[in] numRows Rows (segments) in the chunk
//! \param[in] numBlocks Segment width
//! \param[in] stream CUDA stream
//! \throws std::runtime_error on invalid arguments or launch failure
void launchQsaIndexIdsFill(int32_t* ids, int32_t numRows, int32_t numBlocks, cudaStream_t stream);

//! \brief K4: expand top blocks into token indices and append the causal tail.
//!
//! For global rows r = b * seqLen + t in [rowStart, rowStart + numRows), with
//! nVis = (t + 1) / 4, kSel = min(512, nVis), tail = (t + 1) % 4:
//!   out[r][4*i + j] = sortedIds[r - rowStart][i] * 4 + j   for i < kSel, j in {0,1,2,3}
//!   out[r][4*kSel + j] = 4 * nVis + j                      for j < tail (ALWAYS appended)
//!   out[r][j] = -1                                          for all remaining slots.
//! sortedIds holds block ids sorted by descending logit (only the first kSel entries are
//! read). Padded rows (t >= contextLengths[b] or contextLengths[b] == 0) are written as
//! all -1 and never read sortedIds. Output token indices are distinct, in [0, t], unsorted.
//!
//! Launch: grid(numRows), block(256), threads stride the 2051 output slots.
//!
//! \param[out] outIdx INT32 [batchSize * seqLen, 2051]; rows [rowStart, rowStart + numRows) written
//! \param[in] sortedIds INT32 [numRows, numBlocks] chunk-local descending-logit block ids
//! \param[in] contextLengths Device INT32 [batchSize]
//! \param[in] batchSize Batch size B
//! \param[in] seqLen Padded sequence length S
//! \param[in] numBlocks ceilDiv(seqLen, 4)
//! \param[in] rowStart First global row of the chunk
//! \param[in] numRows Rows in the chunk
//! \param[in] stream CUDA stream
//! \throws std::runtime_error on invalid arguments or launch failure
void launchQsaIndexExpand(int32_t* outIdx, int32_t const* sortedIds, int32_t const* contextLengths, int32_t batchSize,
    int32_t seqLen, int32_t numBlocks, int32_t rowStart, int32_t numRows, cudaStream_t stream);

//! \brief Prefill raw-K tail scatter: persist the raw index-K of the trailing incomplete
//! block's tokens, t in [ctx_b - ctx_b % 4, ctx_b), into their pool K-tails (see
//! QsaIndexerPoolState for the addressing contract).
//!
//! Each such token copies indexQk row (b, t) columns [512, 640) BIT-UNMODIFIED to K plane
//! page pageTable[b][0][t >> 7], row t & 127, head 0, columns [headSize, headSize + 128).
//! Sequences with ctx_b % 4 == 0 and tokens whose page-table slot is out of range or
//! kUNUSED_PAGE_ENTRY write nothing; no other pool cell is touched.
//!
//! Launch: grid(batchSize, kQSA_COMPRESS_RATIO - 1), block(32) = one warp per (sequence,
//! trailing-block slot), 4 columns/lane.
//!
//! \tparam T Activation type: half or __nv_bfloat16 (must match the pool element type).
//! \param[in] indexQk [batchSize, seqLen, 640] T (raw k occupies columns [512, 640))
//! \param[in] contextLengths Device INT32 [batchSize], 0 <= contextLengths[b] <= seqLen
//! \param[in] poolState Paged indexer state; see QsaIndexerPoolState
//! \param[in] batchSize Batch size B
//! \param[in] seqLen Padded sequence length S
//! \param[in] stream CUDA stream
//! \throws std::runtime_error on invalid arguments or launch failure
template <typename T>
void launchQsaRawKTailWritePrefill(T const* indexQk, int32_t const* contextLengths,
    QsaIndexerPoolState const& poolState, int32_t batchSize, int32_t seqLen, cudaStream_t stream);

//! \brief K1b variant that ALSO persists each valid kbar row to the pool V-tail of the
//! block's first token (page pageTable[b][1][(4g) >> 7], row (4g) & 127, head 0, columns
//! [headSize, headSize + 128)).
//!
//! The dense `kbar` output and its numerics are bit-identical to launchQsaIndexKCompress
//! (the prefill scorer still consumes the dense workspace); the pool write is additive.
//! Invalid blocks write dense zeros and never touch the pool; out-of-range/unused page
//! entries skip the pool write only. Parameters as launchQsaIndexKCompress plus poolState.
//!
//! \throws std::runtime_error on invalid arguments or launch failure
template <typename T>
void launchQsaIndexKCompressPaged(T* kbar, T const* indexQk, float const* cosSin, int32_t const* contextLengths,
    T const* wK, float rmsEps, QsaIndexerPoolState const& poolState, int32_t batchSize, int32_t seqLen,
    int32_t numBlocks, int32_t blockBegin, int32_t blockEnd, int32_t pastLen, cudaStream_t stream);

//! \brief B1: fused decode pre-pass — q-prep + predicated raw-K tail write / block compress.
//!
//! One CTA per sequence; every sequence carries exactly one new token whose TOTAL context
//! length (including it) is contextLengths[b], so its position is pos = contextLengths[b]-1.
//! Three phases:
//!   1. (all 4 warps) K1a numerics VERBATIM at S = 1 with rope position pos — decode rows
//!      are never padded, so there is no padding branch (contextLengths[b] <= 0 writes
//!      exact zeros and skips phases 2-3). Output qNormed [batchSize, 1, 4, 128].
//!   2. (warp 0) only when contextLengths[b] % 4 != 0 (pos's block stays incomplete):
//!      indexQk columns [512, 640) stored BIT-UNMODIFIED to the pool K-tail of token pos.
//!   3. (warp 0) only when contextLengths[b] % 4 == 0: block g = contextLengths[b]/4 - 1
//!      just completed; raw keys of tokens 4g..4g+2 are read from their pool K-tails (same
//!      page as pos — a 4-token block never straddles a 128-token page) and token 4g+3's
//!      key is taken from registers; then K1b numerics
//!      VERBATIM (fixed-order FP32 mean x0.25 -> cast to T -> reload FP32 -> gemma norm
//!      (1+wK) -> rope at 4g -> cast) and the kbar row is written to the V-tail of token 4g.
//! The launch is unconditional and device-masked: safe to capture in a CUDA graph.
//!
//! Launch: grid(batchSize), block(128) = 4 warps.
//!
//! \tparam T Activation type: half or __nv_bfloat16 (must match the pool element type).
//! \param[out] qNormed [batchSize, 1, 4, 128] T
//! \param[in] indexQk [batchSize, 1, 640] T (4 q-heads x 128 then raw k x 128 per row)
//! \param[in] cosSin FP32 [maxPositions, 64] with maxPositions >= max contextLengths[b]
//! \param[in] contextLengths Device INT32 [batchSize] TOTAL lengths including the new token
//! \param[in] wQ T [128] indexer q-norm gamma (raw checkpoint w, not 1+w)
//! \param[in] wK T [128] indexer k-norm gamma, same raw-w convention
//! \param[in] rmsEps RMS norm epsilon
//! \param[in] poolState Paged indexer state; see QsaIndexerPoolState
//! \param[in] batchSize Batch size B
//! \param[in] stream CUDA stream
//! \throws std::runtime_error on invalid arguments or launch failure
template <typename T>
void launchQsaIndexerPreDecode(T* qNormed, T const* indexQk, float const* cosSin, int32_t const* contextLengths,
    T const* wQ, T const* wK, float rmsEps, QsaIndexerPoolState const& poolState, int32_t batchSize,
    cudaStream_t stream);

//! \brief B2: decode block-relevance logits from the paged kbar V-tails.
//!
//! K2 numerics at S = 1: logits[b][g] = (1/sqrtf(128)) * sum_h relu(q[b][h] . kbar[b][g])
//! (FP32 dot per head, relu PER HEAD, sum ascending h, scale AFTER the sum) for every
//! g < nVis(b) = contextLengths[b] / 4, with kbar gathered from the V-tail of token 4g.
//! ONLY the valid prefix [0, nVis) of each row is written (no -FLT_MAX fill; downstream
//! reads only the valid prefix). The grid is fixed from the host-static maxBlocks; CTAs
//! (and warps) past nVis early-out, and each warp's 8 block-columns share one page-table
//! lookup (32 consecutive blocks live in one 128-token page).
//!
//! Launch: grid(batchSize, ceilDiv(maxBlocks, 64)), block(256).
//!
//! \tparam T Activation type: half or __nv_bfloat16 (must match the pool element type).
//! \param[out] logits FP32 [batchSize, maxBlocks]; only [0, contextLengths[b]/4) written
//! \param[in] qNormed [batchSize, 1, 4, 128] T from launchQsaIndexerPreDecode
//! \param[in] contextLengths Device INT32 [batchSize] TOTAL lengths including the new token
//! \param[in] poolState Paged indexer state; see QsaIndexerPoolState
//! \param[in] batchSize Batch size B
//! \param[in] maxBlocks Logits row stride; must cover ceilDiv(max context, 4)
//! \param[in] stream CUDA stream
//! \throws std::runtime_error on invalid arguments or launch failure
template <typename T>
void launchQsaIndexScoresDecode(float* logits, T const* qNormed, int32_t const* contextLengths,
    QsaIndexerPoolState const& poolState, int32_t batchSize, int32_t maxBlocks, cudaStream_t stream);

//! \brief B3: decode top-512 block select + K4 expand (deterministic, no cub, no alloc).
//!
//! Per sequence b with ctx = contextLengths[b], nVis = ctx/4, kSel = min(512, nVis),
//! tail = ctx % 4:
//!   a. splitCounters[b*numKVHeads, b*numKVHeads + numKVHeads) are zeroed when the pointer
//!      is non-null (the decode attention's split-K merge counters; workspace memory is not
//!      persistent across enqueues, so in-enqueue zeroing is load-bearing for CUDA graphs).
//!   b. nVis <= 512 selects ALL blocks (fast path; logits are never read).
//!   c. Otherwise an MSB-first 8-bit radix select over the monotone orderable map of the
//!      FP32 logits (u = bits >= 0 ? bits | 0x80000000 : ~bits) resolves the threshold in
//!      <= 4 shared-memory histogram passes (float4-vectorized where aligned, early exit
//!      when a bin exactly fills the remaining quota).
//!   d. Emit is a fixed-order scan: u > threshold always selected; u == threshold fills the
//!      remaining quota in ASCENDING block-id order. This matches the stable descending
//!      cub::DeviceSegmentedRadixSort over ascending-id values used by prefill, so prefill
//!      and decode select IDENTICAL block sets for identical logits.
//!   e. K4 expand: out[b][4i+j] = sel[i]*4 + j packed front, then the tail tokens
//!      [4*nVis, ctx) (ALWAYS appended), then -1 padding to width 2051.
//! Two identical runs produce bitwise-identical outIdx.
//!
//! Launch: grid(batchSize), block(512); ~1.3 KB shared memory.
//!
//! \param[out] outIdx INT32 [batchSize, 1, 2051] (kQSA_INDEX_WIDTH)
//! \param[in] logits FP32 [batchSize, maxBlocks] from launchQsaIndexScoresDecode; only the
//!            valid prefix [0, nVis) of each row is read
//! \param[in] contextLengths Device INT32 [batchSize] TOTAL lengths including the new token
//! \param[in,out] splitCounters Device INT32 [batchSize, numKVHeads] or nullptr
//! \param[in] numKVHeads Counter row width; ignored when splitCounters is nullptr
//! \param[in] batchSize Batch size B
//! \param[in] maxBlocks Logits row stride; must cover ceilDiv(max context, 4)
//! \param[in] stream CUDA stream
//! \throws std::runtime_error on invalid arguments or launch failure
void launchQsaTopKExpandDecode(int32_t* outIdx, float const* logits, int32_t const* contextLengths,
    int32_t* splitCounters, int32_t numKVHeads, int32_t batchSize, int32_t maxBlocks, cudaStream_t stream);

} // namespace kernel
} // namespace trt_edgellm
