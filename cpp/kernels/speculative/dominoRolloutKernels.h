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

#include "common/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace trt_edgellm
{
namespace kernel
{

constexpr int32_t kDominoMaxProjDim = 512;
constexpr int32_t kDominoMaxGruHiddenDim = 4096;

/*!
 * @brief One bias-free GRU cell update over an embedded token input.
 *
 * Reads the input token per batch row as tokenIds[b * tokenStride + tokenOffset],
 * gathers its embedding row, and computes hOut = GRUCell(embed(token), hIn) with
 * the PyTorch r/z/n gate layout of weight_ih [3G, H] and weight_hh [3G, G].
 * hIn and hOut must be distinct buffers; the caller ping-pongs them across the
 * sequential rollout so every launch has capture-stable pointers.
 *
 * embeddingTable: [vocab, H] FP16; gruWeightIh: [3G, H] FP16; gruWeightHh: [3G, G] FP16;
 * hIn/hOut: [batch, G] FP16. All device work; no host synchronization.
 */
//! GRU cell h' = GRU(x, h) where the input projection x . W_ih^T is read from
//! gruInputTable[tokenId] ([vocab, 3G] FP16, precomputed at export) and the hidden
//! projection streams gruWeightHh ([3G, G] FP16). PyTorch gate order r, z, n; no bias.
void dominoGruUpdate(rt::Tensor const& gruInputTable, rt::Tensor const& tokenIds, int32_t tokenStride,
    int32_t tokenOffset, rt::Tensor const& gruWeightHh, rt::Tensor const& hIn, rt::Tensor& hOut, int32_t batchSize,
    int32_t gruHiddenDim, cudaStream_t stream);

//! Row layout of one correction step over the draft engine outputs.
//!
//! zProj holds one row per corrected position ([B, zRows, R]); poolBaseLogits holds one row
//! per proposal position ([B, poolRows, P], row 0 is the uncorrected first proposal). The
//! corrected scores of the step are optionally written to scoresOut[b, scoreRow, :]
//! ([B, scoreRows, P] FP32), which the DDTree builder consumes as per-depth logits.
struct DominoStepLayout
{
    int32_t zRow;
    int32_t zRows;
    int32_t logitsRow;
    int32_t poolRows;
    int32_t scoreRow;
    int32_t scoreRows;
};

//! Per-step scratch for the correction kernels, allocated once for the max batch.
//! correction: [B, projDim] FP32; winnerSlots: [B] INT64 packed (score, pool index) keys.
struct DominoCorrectScratch
{
    rt::Tensor& correction;
    rt::Tensor& winnerSlots;
};

/*!
 * @brief One Domino correction step: bias the pooled base logits and take top-1.
 *
 * Per batch row: c = SiLU(zProj[b, zRow, :] + projStateWeight @ h[b]),
 * score_p = poolBaseLogits[b, logitsRow, p] + dot(poolCodebook[b, p, :], c),
 * winner = argmax_p(score) with exact ties resolved to the lowest pool index,
 * draftTokenIds[b * draftTokenStride + tokenOffset] = poolCandidateIds[b, winner].
 * When scoresOut is non-null, score_p is also stored to scoresOut[b, scoreRow, p].
 *
 * poolCodebook: [batch, P, R] FP16; poolCandidateIds: [batch, P] INT32;
 * projStateWeight: [R, G] FP16; gruHidden: [batch, G] FP16.
 */
void dominoCorrectStep(rt::Tensor const& zProj, rt::Tensor const& poolBaseLogits, rt::Tensor const& poolCodebook,
    rt::Tensor const& poolCandidateIds, rt::Tensor const& projStateWeight, rt::Tensor const& gruHidden,
    rt::Tensor& draftTokenIds, rt::Tensor* scoresOut, DominoCorrectScratch scratch, DominoStepLayout layout,
    int32_t draftTokenStride, int32_t tokenOffset, int32_t batchSize, int32_t poolSize, int32_t projDim,
    int32_t gruHiddenDim, cudaStream_t stream);

/*!
 * @brief Uncorrected first proposal: argmax of poolBaseLogits row 0 per batch.
 *
 * draftTokenIds[b * draftTokenStride + tokenOffset] = poolCandidateIds[b, argmax_p(poolBaseLogits[b, 0, p])],
 * ties resolved to the lowest pool index. When scoresOut is non-null, row 0 is copied to
 * scoresOut[b, 0, :] so the DDTree builder sees the same scores the chain used.
 */
void dominoSelectFirstProposal(rt::Tensor const& poolBaseLogits, int32_t poolRows, rt::Tensor const& poolCandidateIds,
    rt::Tensor& draftTokenIds, rt::Tensor* scoresOut, int32_t scoreRows, rt::Tensor& winnerSlots,
    int32_t draftTokenStride, int32_t tokenOffset, int32_t batchSize, int32_t poolSize, cudaStream_t stream);

/*!
 * @brief Map DDTree node tokens from candidate-pool slots back to vocabulary ids.
 *
 * ddtreeBuild selects candidates in pool-slot space when fed Domino's stacked scores. For every
 * valid node except the root (node 0, already a vocabulary id), treeTokenIds[b, n] and
 * verifyTokenIds[b, n] are replaced by poolCandidateIds[b, slot].
 */
void dominoRemapPoolSlots(rt::Tensor& treeTokenIds, rt::Tensor& verifyTokenIds, rt::Tensor const& validCounts,
    rt::Tensor const& poolCandidateIds, int32_t batchSize, int32_t verifySize, int32_t poolSize, cudaStream_t stream);

} // namespace kernel
} // namespace trt_edgellm
