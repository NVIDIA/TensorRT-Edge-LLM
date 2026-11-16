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

#include "runtime/decoding/decodingStrategy.h"
#include "runtime/decoding/jumpForward.h"
#include "runtime/state/externalWeightManager.h"

#include <filesystem>
#include <memory>
#include <optional>

namespace trt_edgellm
{
namespace rt
{

//! Gemma4 assistant MTP decoder.
//!
//! This class owns the Gemma4-specific draft-engine binding path. The assistant
//! has no draft KV cache: its `past_key_values_*` inputs are zero-copy aliases
//! to the base target KV cache selected by `kv_sharing_map`.
class Gemma4MTPDecoder final : public DecodingStrategy
{
public:
    Gemma4MTPDecoder(DecodingRuntimeContext& runtime, std::unique_ptr<EngineExecutor> draftExecutor,
        ExternalWeightManager draftWeights, cudaStream_t stream);

    DecodingStrategyKind kind() const noexcept override
    {
        return DecodingStrategyKind::kGemma4MTP;
    }

    char const* name() const noexcept override
    {
        return "gemma4_mtp";
    }

    bool isSpeculative() const noexcept override
    {
        return true;
    }

    DecodingStrategyCapabilities capabilities() const noexcept override
    {
        // The paired speculative base is not a standalone target engine, so VanillaDecoder fallback is invalid.
        return {/*.ownsBaseVerificationCudaGraphs=*/false, /*.supportsLosslessSampling=*/false,
            /*.maxSamplingSupport=*/0, /*.fallbackToVanillaForNonGreedySampling=*/false};
    }

    DecodingKvHeadroomPolicy kvHeadroom() const override;

    bool decodeStep(DecodingInferenceContext& context) override;
    bool captureCudaGraphs(cudaStream_t stream) override;
    void beginBatch() noexcept override;
    void appendRow(int32_t executionRow) noexcept override;
    bool seatRow(DecodingInferenceContext& context, int32_t executionRow) override;

    int64_t getRequiredContextMemorySize() const noexcept override;
    void setContextMemory(Tensor& memory) override;

    void resetForNewSequences(Tensor& reuseLengths, cudaStream_t stream) override;
    void onBatchEvict(std::vector<int32_t> const& batchMapping, int32_t oldActiveBatch, int32_t newActiveBatch,
        Tensor& deviceBatchMapping, cudaStream_t stream) override;

private:
    //! Seed preparation + assistant draft chain, timed as one draft-proposal stage.
    bool runDraftProposal(DecodingInferenceContext& context);
    bool prepareSeed(DecodingInferenceContext& context);
    bool runAssistantDraftChain(DecodingInferenceContext& context);
    bool buildTreeVerifyInputs(int32_t activeBatchSize, cudaStream_t stream);
    bool runBaseVerification(DecodingInferenceContext& context);
    //! Grafts the forced chains of the slots that may jump into the drafted verify input, and captures the round's
    //! verify input for the grammar mask.
    void finishVerifyInput(DecodingInferenceContext& context);
    //! The verify input of this round: the grafted one in a grafting round, the drafted one otherwise.
    SpecVerifyInputs roundVerifyInputs(int32_t activeBatchSize) const;
    bool acceptAndCommit(DecodingInferenceContext& context);
    bool updateNextSeed(DecodingInferenceContext& context);

    DecodingRuntimeContext& mRuntime;
    std::unique_ptr<EngineExecutor> mDraftExecutor;
    TensorMap mDraftTensorMap;
    ExternalWeightManager mDraftExternalWeightManager;

    Tensor mSeedTokenIds;                     //!< [B] INT32 current root/seed token.
    Tensor mHostSeedTokenIds;                 //!< [B] INT32 host staging for current root/seed token.
    Tensor mSeedHiddenSourceTokenIndices;     //!< [B] INT32 source token index for seed-hidden gather.
    Tensor mHostSeedHiddenSourceTokenIndices; //!< [B] INT32 host staging for seed-hidden gather indices.
    Tensor mDraftTokenIds;                    //!< [B, P] INT32 assistant draft chain.
    Tensor mVerifyTokenIds;                   //!< [B, max(P + 1, J)] INT32 root plus assistant draft chain.
    Tensor mAcceptedTokenIds;                 //!< [B, max(P + 1, W)] INT32 append-only output tokens, root excluded.
    Tensor mAcceptedTokenIndices;             //!< [B, max(P + 1, W)] INT32 accepted predictor rows in verification.
    Tensor mAcceptLength;                     //!< [B] INT32 per-batch accepted output length.
    Tensor mHostAcceptLengths;                //!< [B] INT32 host staging.
    Tensor mHostSeatStaging;                  //!< [B, 2] INT32 pinned; seatRow() only.
    Tensor mHostAcceptedTokenIds;             //!< [B, max(P + 1, W)] INT32 host staging.
    Tensor mArgmaxScratch;                    //!< [B * max(P + 1, J)] INT32 chain target token IDs.
    //! Target hidden states of each row's last accepted span, [maxBatch * max(P + 1, W), baseOutputHiddenDim]; a
    //! seated prefill overwrites the pipeline's baseHiddenStates between decode steps.
    Tensor mAcceptedHiddenStates;
    //! Every row's seed comes from its last accepted span rather than the prefill.
    bool mDraftPrimed{false};
    int32_t mAcceptCapacity{0}; //!< Rows per slot of the carried accepted span: max(P + 1, W).
    RaggedExecutionBatch mRaggedMetadataScratch;

    bool mUseTree{false};
    Tensor mStackedDraftLogits; //!< [B, P + 1, draftVocab] per-depth proposal logits.
    Tensor mTreeTokenIds;       //!< [B, verifyTokenCount] flattened tree token ids.
    Tensor mTreeNodeScores;     //!< [B, verifyTokenCount] prefix log-prob scores.
    Tensor mValidCounts;        //!< [B] valid tree node counts.
    Tensor mTreeBuildWorkspace; //!< DDTree build workspace.
    JumpGraftSession mJump;     //!< The grafter of a deployment that can jump, and whether this round grafted.
};

} // namespace rt
} // namespace trt_edgellm
