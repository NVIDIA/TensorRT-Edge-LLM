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
#include "runtime/decoding/dflashDecodeUtils.h"
#include "runtime/decoding/jumpForward.h"
#include "runtime/decoding/specCommonStateTracker.h"
#include "runtime/state/externalWeightManager.h"

#include <filesystem>
#include <memory>
#include <optional>

namespace trt_edgellm
{
namespace rt
{

class DFlashDecoder : public DecodingStrategy
{
public:
    DFlashDecoder(DecodingRuntimeContext& runtime, std::filesystem::path const& engineDir,
        dflash_utils::CachedBlockDraftRuntimeConfig blockDraftConfig, std::unique_ptr<EngineExecutor> draftExecutor,
        ExternalWeightManager draftWeights, cudaStream_t stream);

    DecodingStrategyKind kind() const noexcept override
    {
        return DecodingStrategyKind::kDFlash;
    }

    char const* name() const noexcept override
    {
        return mVersion == DFlashVersion::kV2 ? "dflash" : userModeName();
    }

    bool isSpeculative() const noexcept override
    {
        return true;
    }

    DecodingStrategyCapabilities capabilities() const noexcept override
    {
        bool const greedyOnly = greedyOnlyAcceptance();
        return {/*.ownsBaseVerificationCudaGraphs=*/true,
            /*.supportsLosslessSampling=*/!greedyOnly,
            /*.maxSamplingSupport=*/greedyOnly ? 0 : dflash_utils::kDFlash2MaxSamplingSupport,
            /*.fallbackToVanillaForNonGreedySampling=*/greedyOnly,
            /*.requiresDefaultDecoderCudaGraphs=*/greedyOnly};
    }

    DecodingKvHeadroomPolicy kvHeadroom() const override;

    bool decodeStep(DecodingInferenceContext& context) override;
    bool captureCudaGraphs(cudaStream_t stream) override;
    bool initializeForGeneration(DecodingInferenceContext& context) override;
    void beginBatch() noexcept override;
    void appendRow(int32_t executionRow) noexcept override;
    bool seatRow(DecodingInferenceContext& context, int32_t executionRow) override;
    bool prefillSeatedChunk(DecodingInferenceContext& context) override;
    std::vector<int32_t> const& commonMaterializedStateLengths() const noexcept override;

    int64_t getRequiredContextMemorySize() const noexcept override;
    void setContextMemory(Tensor& memory) override;

    void resetForNewSequences(Tensor& reuseLengths, cudaStream_t stream) override;
    void onBatchEvict(std::vector<int32_t> const& batchMapping, int32_t oldActiveBatch, int32_t newActiveBatch,
        Tensor& deviceBatchMapping, cudaStream_t stream) override;

private:
    bool isV2() const noexcept
    {
        return mVersion == DFlashVersion::kV2;
    }
    bool isLiLiCorr() const noexcept
    {
        return isV2() && mRuntime.deployment.draft->dflashProjector == DFlashProjector::kLiLiCorr;
    }
    //! Candidate count per proposal slot shared by the draft outputs, proposal q and sparse acceptance.
    int32_t proposalSupportK() const noexcept
    {
        LLMEngineConfig const& draftCfg = *mRuntime.deployment.draft;
        return isLiLiCorr() ? draftCfg.specLiLiCorrTopK : draftCfg.specSelectorTopK;
    }
    //! Reshapes the projector's draft-engine outputs: support ids plus transition scores or selector inputs.
    void reshapeDFlash2DraftOutputs(int32_t batchSize);
    void initializeDFlash2(
        std::filesystem::path const& engineDir, ExternalWeightManager draftWeights, cudaStream_t stream);
    void initializeJumpGrafter();
    //! @param prefillDelta The target-hidden delta is the prefill's (a request's first forward, a context-cache
    //!        prefill chunk, a seated row) rather than the last round's accepted spans.
    //! @param heldBackTokens Prompt positions at each row's end a prefill delta leaves for the next step (seatRow
    //!        holds one back).
    bool runDraftForward(DecodingInferenceContext& context, bool prefillDelta, int32_t heldBackTokens = 0);
    bool prepareBlockDraftVerifyInputs(DecodingInferenceContext& context);
    bool prepareV2Proposal(DecodingInferenceContext& context);
    bool captureDraftCudaGraphs(cudaStream_t stream);
    bool buildTreeVerifyInputs(DecodingInferenceContext& context);
    //! Grafts the forced chains of the slots that may jump into the drafted verify input, and captures the round's
    //! verify input for the grammar mask.
    void finishVerifyInput(DecodingInferenceContext& context);
    bool runBaseVerification(DecodingInferenceContext& context);
    bool runV2Acceptance(DecodingInferenceContext& context, int32_t verifyTokenCount, int32_t maxAcceptLength);
    bool executeBaseVerification(DecodingInferenceContext& context, int32_t verifyTokenCount);
    bool capturePreparedBaseVerification(int32_t batchSize, int32_t verifyTokenCount, cudaStream_t stream);
    //! @param source The target hidden states the delta is taken from: the prefill's, or the carried accepted spans.
    void bindTargetHiddenDelta(Tensor& source, int32_t activeBatchSize, int64_t maxDeltaLen, int64_t sourceSeqLen,
        bool allowLargeDelta, cudaStream_t stream);
    void initializeDraftRaggedBindings(LLMEngineConfig const& draftCfg);
    void prepareDraftRaggedBindings(int32_t activeBatchSize, int32_t proposalLen, int32_t deltaWidth,
        cudaStream_t stream, std::vector<ResidentRef> const* residentRefs = nullptr);
    //! draftTopK > 1 verifies a DDTree; draftTopK = 1 verifies the linear block.
    bool useTreeVerification() const noexcept
    {
        return mBlockDraft.treePolicy == dflash_utils::BlockDraftTreePolicy::kDDTree;
    }
    //! Only the DFlash2 linear chain verifies with the lossless sparse-probability acceptor.
    bool greedyOnlyAcceptance() const noexcept
    {
        return !isV2() || useTreeVerification();
    }
    void selectDFlash2TreeCandidates(int32_t activeBatchSize, cudaStream_t stream);
    bool causalProposalMask() const noexcept
    {
        return mBlockDraft.proposalAttention == dflash_utils::ProposalAttentionPolicy::kCausal;
    }
    char const* userModeName() const noexcept
    {
        return specDecodeModeName(mBlockDraft.userMode);
    }

protected:
    // Hooks for cached-block-draft variants that post-process the draft engine outputs
    // (see DominoDecoder). The DFlash V1 defaults read argmax tokens from mDraftOutputLogits.
    //! Reshape the variant-owned draft engine output tensors for `batchSize` rows before a draft forward.
    virtual void reshapeDraftOutputs(int32_t batchSize);
    //! Post-process the draft engine outputs right after the draft forward completes.
    virtual bool produceProposal(DecodingInferenceContext& /* context */)
    {
        return true;
    }
    //! Fill mDraftTokenIds for the linear chain when produceProposal has not already done so.
    virtual void selectLinearProposalTokens(DecodingInferenceContext& context);
    //! The V1 proposal waiting for base verification: raw draft logits, argmax'd at verify time.
    virtual Tensor& pendingProposalTensor()
    {
        return mDraftOutputLogits;
    }
    //! Per-depth candidate logits the DDTree builder selects from. Row `firstCandidateLogitsRow + d - 1`
    //! feeds depth d; vocabMap optionally maps the selected ids to the full vocabulary.
    struct TreeCandidateLogits
    {
        Tensor const& logits;
        int32_t firstCandidateLogitsRow;
        Tensor const* vocabMap;
    };
    virtual TreeCandidateLogits treeCandidateLogits() const
    {
        return {
            mDraftOutputLogits, /*firstCandidateLogitsRow=*/1, mHasDraftVocabMap ? &mDraftVocabMappingTable : nullptr};
    }
    //! Post-process the built tree tokens (e.g. map candidate-pool slots to vocabulary ids).
    virtual void finalizeTreeTokens(DecodingInferenceContext& /* context */) {}

    bool checkCudaLastError(char const* stage) const;

    DecodingRuntimeContext& mRuntime;
    HybridCacheManager& mDraftCacheManager;
    DFlashVersion mVersion;
    dflash_utils::CachedBlockDraftRuntimeConfig mBlockDraft;
    std::unique_ptr<EngineExecutor> mDraftExecutor;
    TensorMap mDraftTensorMap;
    ExternalWeightManager mDraftExternalWeightManager;

    Tensor mDraftInputsEmbeds;        //!< [B, blockSize, draftHiddenSize] FP16
    Tensor mDraftTargetHidden;        //!< Compact scratch for [B, <= blockSize, baseOutputHiddenDim] FP16
    Tensor mDraftPrefillTargetHidden; //!< Lazy scratch for non-compact round-0 target hidden FP16, max batch reserve
    Tensor mDraftOutputLogits;        //!< [B, blockSize, vocabSize] FP32; empty when the draft engine emits no logits

    Tensor mDraftPackedAttentionMask;  //!< [B, blockSize, divUp(blockSize,32)] INT32
    Tensor mDraftAttentionPosId;       //!< [B, blockSize] INT32
    Tensor mDraftContextLengths;       //!< [B] INT32
    Tensor mDraftDeltaLenCommit;       //!< [B] INT32
    Tensor mDraftDeltaLens;            //!< [B] INT32
    Tensor mDraftDeltaRopeCosSin;      //!< [T_delta, rotaryDim] FLOAT
    Tensor mDraftDeltaPositions;       //!< [T_delta] INT32
    Tensor mDraftDeltaTokenToSequence; //!< [T_delta] INT32

    Tensor mDraftTokenIds;          //!< [B, blockSize] INT32
    Tensor mHostDraftInputIds;      //!< [B, blockSize] INT32 CPU
    Tensor mHostLastAcceptedTokens; //!< [B] INT32 CPU
    Tensor mHostDeltaLens;          //!< [B] INT32 CPU
    Tensor mLastAcceptedTokens;     //!< [B] INT32 GPU
    //! Fences the host staging the draft forward uploads from: a seated prefill returns with its draft-length
    //! commit still queued, and the next decode step rewrites the staging on the host.
    AsyncHostStagingFence mHostStagingFence;

    Tensor mTreeTokenIds;         //!< [B, jumpVerifyTokenCount] INT32, DDTree node tokens
    Tensor mTreeNodeScores;       //!< [B, verifyTokenCount] FP32, DDTree prefix scores
    Tensor mValidCounts;          //!< [B] INT32, DDTree valid node counts
    Tensor mVerifyTokenIds;       //!< [B, jumpVerifyTokenCount] INT32
    Tensor mAcceptedTokenIds;     //!< [B, maxAcceptBufferSize] INT32
    Tensor mAcceptedTokenIndices; //!< [B, maxAcceptBufferSize] INT32, verify logits/KV indices
    Tensor mAcceptLength;         //!< [B] INT32
    Tensor mHostAcceptLengths;    //!< [B] INT32 (CPU)
    Tensor mHostSeatStaging;      //!< [B, 2] INT32 (CPU); seatRow() only
    Tensor mHostAcceptedTokenIds; //!< [B, maxAcceptBufferSize] INT32 (CPU)

    Tensor mProposalSupportIds;             //!< [B, proposalLen, proposalSupportK] INT32
    Tensor mProposalSupportProbs;           //!< [B, proposalLen, proposalSupportK] FP32
    Tensor mProposalUnaryValues;            //!< [B, proposalLen, selectorTopK] FP32; selector only
    Tensor mProposalProjectedHidden;        //!< [B, proposalLen, selectorRank] FP16; selector only
    Tensor mSelectorPredecessorCodebook;    //!< [vocabSize, selectorRank] FP16; selector only
    Tensor mSelectorSuccessorCodebook;      //!< [vocabSize, selectorRank] FP16; selector only
    Tensor mLiLiCorrTransitionScores;       //!< [B, proposalLen, K, K] FP32 [b, s, prev, next]; LiLiCorr only
    Tensor mTargetTopKValues;               //!< [B * verifyTokenCount, maxTargetTopK] FP32
    Tensor mTargetTopKIds;                  //!< [B * verifyTokenCount, maxTargetTopK] INT32
    Tensor mTargetTopKProbs;                //!< [B * verifyTokenCount, maxTargetTopK] FP32
    Tensor mTargetProbabilities;            //!< [B * verifyTokenCount, vocabSize] FP32
    Tensor mProposalLengths;                //!< [B] INT32
    Tensor mProposalUniforms;               //!< [B, proposalLen] FP32
    Tensor mAcceptUniforms;                 //!< [B, 2 * proposalLen + 1] FP32
    Tensor mSamplingTemperatures;           //!< [B] FP32
    Tensor mProposalGreedyMask;             //!< [B] INT32
    Tensor mRemainingGenerationLengths;     //!< [B] INT32
    Tensor mHostRemainingGenerationLengths; //!< [B] INT32 pinned CPU
    Tensor mRequestSeeds;                   //!< [B] UINT64
    Tensor mNextAbsolutePositions;          //!< [B] UINT64
    Tensor mHostRequestSeeds;               //!< [B] INT64 pinned CPU
    Tensor mHostNextPositions;              //!< [B] INT64 pinned CPU
    Tensor mHostTemperatures;               //!< [B] FP32 pinned CPU
    Tensor mHostGreedyMask;                 //!< [B] INT32 pinned CPU

    //! DFlash2 DDTree candidate table [B, proposalLen, selectorTopK, draftTopK] (ddtreeAssemble Stage A output).
    Tensor mTreeCandidateTokenIds;    //!< INT32 token ids
    Tensor mTreeCandidateLogProbs;    //!< FP32 selector log-probabilities over the support
    Tensor mTreeCandidateChildStates; //!< INT32 support index of each candidate

    SpecCommonStateTracker mCommonStateTracker;
    Tensor mBuildWorkspace; //!< DDTree build workspace bytes

    //! Draft vocab map [reducedVocabSize] INT32 (GPU). Active when draft
    //! lm_head uses a reduced vocabulary. Sized to zero otherwise.
    Tensor mDraftVocabMappingTable;
    bool mHasDraftVocabMap{false};
    //! Target hidden states of each row's last accepted span, [maxBatch * maxAcceptBufferSize, baseOutputHiddenDim];
    //! a seated prefill overwrites the pipeline's baseHiddenStates between decode steps.
    Tensor mAcceptedHiddenStates;
    //! Every row's draft has consumed its prompt, so its next delta is its last accepted span.
    bool mDraftPrimed{false};

private:
    JumpGraftSession mJump; //!< The grafter of a deployment that can jump, and whether this round grafted.
};

} // namespace rt
} // namespace trt_edgellm
