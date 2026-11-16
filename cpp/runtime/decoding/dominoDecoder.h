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

#include "runtime/decoding/dflashDecoder.h"

#include <filesystem>
#include <memory>

namespace trt_edgellm
{
namespace rt
{

//! Domino: the DFlash V1 cached-block draft plus a serial GRU correction rollout.
//!
//! The draft engine emits a shared candidate pool with its base logits for every proposal
//! position, the pool's codebook rows, and the projected draft hidden states. This decoder
//! replays the correction head (GRU + state projection from the `domino_heads.safetensors`
//! sidecar) along the greedy chain and records each position's corrected pool scores. The
//! DDTree builder inherited from DFlashDecoder then selects the verify tree from those scores
//! exactly as it does from DFlash logits, so topK=1 is the chain and topK>1 fans out around it.
class DominoDecoder final : public DFlashDecoder
{
public:
    DominoDecoder(DecodingRuntimeContext& runtime, std::filesystem::path const& engineDir,
        dflash_utils::CachedBlockDraftRuntimeConfig blockDraftConfig, std::unique_ptr<EngineExecutor> draftExecutor,
        ExternalWeightManager draftWeights, cudaStream_t stream);

protected:
    void reshapeDraftOutputs(int32_t batchSize) override;
    bool produceProposal(DecodingInferenceContext& context) override;
    //! The rollout already wrote the chain into mDraftTokenIds; there are no draft logits to argmax.
    void selectLinearProposalTokens(DecodingInferenceContext& /* context */) override {}
    //! The rollout materialized per-position pool scores; the tree is built from them.
    Tensor& pendingProposalTensor() override
    {
        return mStackedScores;
    }
    TreeCandidateLogits treeCandidateLogits() const override
    {
        return {mStackedScores, /*firstCandidateLogitsRow=*/0, nullptr};
    }
    void finalizeTreeTokens(DecodingInferenceContext& context) override;

private:
    void loadCorrectionHead(std::filesystem::path const& engineDir, cudaStream_t stream);
    bool runRollout(DecodingInferenceContext& context);

    Tensor mPoolIds;         //!< [B, poolSize] INT32
    Tensor mPoolLogits;      //!< [B, blockSize - 1, poolSize] FP32, row 0 is the uncorrected first proposal
    Tensor mPoolCodebook;    //!< [B, poolSize, projDim] FP16
    Tensor mZProj;           //!< [B, blockSize - 2, projDim] FP16
    Tensor mStackedScores;   //!< [B, blockSize, poolSize] FP32 corrected pool scores per depth row (last row unused)
    Tensor mGruInputTable;   //!< [vocab, 3G] FP16 sidecar: embed_tokens . W_ih^T per token
    Tensor mGruWeightHh;     //!< [3G, G] FP16 sidecar
    Tensor mProjStateWeight; //!< [projDim, G] FP16 sidecar
    Tensor mGruState[2];     //!< [B, G] FP16 ping-pong buffers
    Tensor mCorrection;      //!< [B, projDim] FP32 per-step correction vector
    Tensor mWinnerSlots;     //!< [B] INT64 packed (score, pool index) argmax slots
};

} // namespace rt
} // namespace trt_edgellm
