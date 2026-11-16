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
#include "runtime/decoding/specCommonStateTracker.h"
#include "runtime/state/externalWeightManager.h"

#include <filesystem>
#include <memory>
#include <optional>

namespace trt_edgellm
{
namespace rt
{

class DSparkDecoder final : public DecodingStrategy
{
public:
    DSparkDecoder(DecodingRuntimeContext& runtime, std::filesystem::path const& engineDir,
        std::unique_ptr<EngineExecutor> draftExecutor, ExternalWeightManager draftWeights, cudaStream_t stream);

    DecodingStrategyKind kind() const noexcept override
    {
        return DecodingStrategyKind::kDSpark;
    }

    char const* name() const noexcept override
    {
        return "dspark";
    }

    bool isSpeculative() const noexcept override
    {
        return true;
    }

    DecodingStrategyCapabilities capabilities() const noexcept override
    {
        return {/*.ownsBaseVerificationCudaGraphs=*/false, /*.supportsLosslessSampling=*/!mHasDraftVocabMap,
            /*.maxSamplingSupport=*/0,
            /*.fallbackToVanillaForNonGreedySampling=*/mUseTree || mHasDraftVocabMap};
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
    //! @param prefillDelta The target-hidden delta is the prefill's (a request's first forward, a context-cache
    //!        prefill chunk, a seated row) rather than the last round's accepted spans.
    //! @param heldBackTokens Prompt positions at each row's end a prefill delta leaves for the next step (seatRow
    //!        holds one back).
    bool runDraftForward(DecodingInferenceContext& context, bool prefillDelta, int32_t heldBackTokens = 0);
    //! Grafts the forced chains of the slots that may jump into the round's drafted verify input, and captures the
    //! verify input for the grammar mask.
    void finishVerifyInput(DecodingInferenceContext& context);
    bool runBaseVerification(DecodingInferenceContext& context);
    bool buildTreeVerifyInputs(int32_t activeBatchSize, cudaStream_t stream, bool useConfidence);
    void prepareDraftRaggedBindings(int32_t activeBatchSize, int32_t executionWidth, int32_t deltaWidth,
        cudaStream_t stream, std::vector<ResidentRef> const* residentRefs = nullptr);
    void commitAcceptedTreePath(DecodingInferenceContext& context, int32_t verifyTokenCount, int32_t maxAcceptLength);
    void dsparkBiasMarkovGreedy(DecodingInferenceContext& context, int32_t activeBatchSize, int32_t proposalLen);
    void dsparkBiasMarkovSample(DecodingInferenceContext& context, int32_t activeBatchSize, int32_t proposalLen);
    void loadHeadSidecars(std::filesystem::path const& engineDir, cudaStream_t stream);
    void loadW2(std::filesystem::path const& engineDir, std::string const& headsFile, std::vector<Tensor>& tensors,
        cudaStream_t stream);
    void prepareW4A16(Tensor const& packedWeight, Tensor const& blockScales, cudaStream_t stream);
    void loadDraftVocabMap(std::filesystem::path const& engineDir, cudaStream_t stream);

    DecodingRuntimeContext& mRuntime;

    //! Draft KV cache manager (shared resource index 1)
    HybridCacheManager& mDraftCacheManager;

    std::unique_ptr<EngineExecutor> mDraftExecutor;
    TensorMap mDraftTensorMap;
    ExternalWeightManager mDraftExternalWeightManager;

    //! Draft engine I/O tensors
    Tensor mDraftInputsEmbeds; //!< [B, proposalLen, draftHiddenSize] FP16
    Tensor mDraftTargetHidden; //!< [B, deltaLen, baseOutputHiddenDim] FP16
    Tensor mDraftOutputLogits; //!< [B, proposalLen, vocabSize] FP32
    Tensor mDraftHiddenStates; //!< [B, proposalLen, draftHiddenSize] FP16

    //! Proposal attention inputs
    Tensor mDraftPackedAttentionMask;  //!< [B, proposalLen, divUp(proposalLen,32)] INT32
    Tensor mDraftAttentionPosId;       //!< [B, proposalLen] INT32
    Tensor mDraftContextLengths;       //!< [B] INT32
    Tensor mDraftDeltaLenCommit;       //!< [B] INT32
    Tensor mDraftDeltaLens;            //!< [B] INT32
    Tensor mDraftDeltaRopeCosSin;      //!< [T_delta, rotaryDim] FLOAT
    Tensor mDraftDeltaPositions;       //!< [T_delta] INT32
    Tensor mDraftDeltaTokenToSequence; //!< [T_delta] INT32

    //! Draft/verify/accepted tokens
    Tensor mDraftTokenIds;        //!< [B, proposalLen] INT32
    Tensor mVerifyTokenIds;       //!< [B, verifyLen] INT32, verifyLen = proposalLen + 1
    Tensor mAcceptedTokenIds;     //!< [B, max(verifyLen, jumpMaxPathLength)] INT32
    Tensor mAcceptLength;         //!< [B] INT32
    Tensor mHostAcceptLengths;    //!< [B] INT32 (CPU)
    Tensor mHostSeatStaging;      //!< [B, 2] INT32 (CPU); seatRow() only
    Tensor mHostAcceptedTokenIds; //!< [B, max(verifyLen, jumpMaxPathLength)] INT32 (CPU)

    //! Proposal self-attention inside the draft block is causal rather than
    //! bidirectional. Sourced from dspark_config.causal_head.
    bool mCausalProposalMask{false};

    SpecCommonStateTracker mCommonStateTracker;
    Tensor mHostDraftInputIds;      //!< [B, proposalLen] INT32 (CPU)
    Tensor mHostLastAcceptedTokens; //!< [B] INT32 (CPU)
    Tensor mHostDeltaLens;          //!< [B] INT32 (CPU)
    Tensor mHostProposalLengths;    //!< [B] INT32 (CPU)
    //! Fences the host staging the draft forward uploads from: a seated prefill returns with its draft-length
    //! commit still queued, and the next decode step rewrites the staging on the host.
    AsyncHostStagingFence mHostStagingFence;

    //! Stochastic DSpark sampling buffers
    Tensor mDraftProbabilities;      //!< [B, proposalLen, vocabSize] FP32
    Tensor mDraftStepLogits;         //!< [B, vocabSize] FP32
    Tensor mDraftStepProbabilities;  //!< [B, vocabSize] FP32
    Tensor mDraftStepTopKValues;     //!< [B, maxSparseTopK] FP32
    Tensor mDraftStepTopKIndices;    //!< [B, maxSparseTopK] INT32
    Tensor mDraftTopKProbabilities;  //!< [B, proposalLen, maxSparseTopK] FP32
    Tensor mDraftTopKIndices;        //!< [B, proposalLen, maxSparseTopK] INT32
    Tensor mTargetProbabilities;     //!< [B, verifyLen, vocabSize] FP32
    Tensor mTargetTopKValues;        //!< [B * verifyLen, maxSparseTopK] FP32
    Tensor mTargetTopKIndices;       //!< [B * verifyLen, maxSparseTopK] INT32
    Tensor mTargetTopKProbabilities; //!< [B * verifyLen, maxSparseTopK] FP32
    Tensor mDraftUniforms;           //!< [B, proposalLen] FP32
    Tensor mAcceptUniforms;          //!< [B, 2 * proposalLen + 1] FP32
    Tensor mConfidenceScores;        //!< [B, proposalLen] FP32
    Tensor mProposalLengths;         //!< [B] INT32 logical prefix selected for verification/accept

    Tensor mAcceptValidTokenCounts; //!< Root-inclusive bounds retained with pending chain proposals.

    //! Pre-allocated target selection scratch buffer [maxBatch * verifyLen] INT32
    Tensor mArgmaxScratch;

    //! Last accepted token per batch [maxBatch] INT32 (GPU)
    Tensor mLastAcceptedTokens;

    //! Tree drafting state.
    Tensor mStackedMarkovLogits;  //!< [maxBatch, blockSize+1, vocabSize] FP32, row 0 is a root placeholder
    Tensor mTreeTokenIds;         //!< [maxBatch, jumpVerifyTokenCount] INT32 flattened tree token ids
    Tensor mTreeNodeScores;       //!< [maxBatch, verifyTokenCount] FP32 prefix log-prob scores
    Tensor mValidCounts;          //!< [maxBatch] INT32 valid node counts
    Tensor mTreeBuildWorkspace;   //!< ddtreeBuild scratch for scheduler and biased requests
    Tensor mAcceptedTokenIndices; //!< [maxBatch, max(verifyLen, jumpMaxPathLength)] INT32 accepted verify-node indices

    //! DSpark Markov/confidence sidecars
    Tensor mMarkovW1;               //!< [targetVocabSize, markovRank] FP16
    Tensor mMarkovW2;               //!< [draftVocabSize, markovRank] FP16
    Tensor mMarkovW2Fp8;            //!< [draftVocabSize, markovRank] E4M3 bytes
    Tensor mMarkovW2RowScales;      //!< [draftVocabSize] FP16 per-row scales
    Tensor mMarkovGreedySlots;      //!< [maxBatch, proposalLen] packed greedy winners
    Tensor mConfidenceWeight;       //!< [hiddenSize + optional markovRank] FP16
    Tensor mConfidenceBias;         //!< [1] FP16
    Tensor mDraftVocabMappingTable; //!< [draftVocabSize] reduced-to-full IDs
    Tensor mMarkovW2GlobalScale;
    DSparkW2Backend mW2Backend{DSparkW2Backend::kFp16};
    Tensor mMarkovW2Opaque;
    Tensor mMarkovW2OpaqueScales;
    Tensor mMarkovTcgen5Activation;
    Tensor mMarkovTcgen5Correction;
    bool mHasConfidenceHead{false};
    bool mConfidenceHeadWithMarkov{false};
    bool mHasDraftVocabMap{false};

    //! DSpark-specific parameters
    bool mUseTree{false};          //!< draftTopK > 1 selects tree drafting
    bool mUseTreeScheduler{false}; //!< threshold scheduling uses confidence-guided ddtreeBuild
    int32_t mProposalLen{7};
    int32_t mVerifyLen{8};
    int32_t mAcceptCapacity{0}; //!< Rows per slot of the carried accepted span: max(verifyLen, jumpMaxPathLength)
    int32_t mDraftBlockLen{0};
    //! Index of the first draft query slot that carries a proposal. 0 when the checkpoint
    //! samples from the anchor, 1 when slot 0 is the bonus token (the validated layout).
    int32_t mDraftSlotOffset{1};
    int32_t mCurrentProposalLen{7};
    int32_t mCurrentVerifyLen{8};
    int32_t mMaskTokenId{151669};
    int32_t mDraftHiddenSize{0};
    int32_t mBaseOutputHiddenDim{0};
    int32_t mDraftVocabSize{0};
    int32_t mTargetVocabSize{0};
    int32_t mMarkovRank{0};
    DSparkSchedulerMode mSchedulerMode{DSparkSchedulerMode::kOff};
    float mConfidenceThreshold{0.0F};
    int32_t mMinScheduledProposalLen{1};
    int32_t mMaxScheduledProposalLen{0};
    //! Per-slot scheduled proposal lengths for the current round (scheduler on); empty otherwise.
    std::vector<int32_t> mScheduledProposalLengths;
    //! Draft distribution the latest proposal stored for stochastic acceptance.
    enum class DraftDistribution : int8_t
    {
        kNone,  //!< greedy proposal
        kDense, //!< mDraftProbabilities
        kTopK   //!< mDraftTopKProbabilities and mDraftTopKIndices
    };
    DraftDistribution mDraftDistribution{DraftDistribution::kNone};
    //! Target hidden states of each row's last accepted span, [maxBatch * max(verifyLen, jumpMaxPathLength),
    //! baseOutputHiddenDim]; a seated prefill overwrites the pipeline's baseHiddenStates between decode steps.
    Tensor mAcceptedHiddenStates;
    //! Every row's draft has consumed its prompt, so its next delta is its last accepted span.
    bool mDraftPrimed{false};
    JumpGraftSession mJump; //!< The grafter of a deployment that can jump, and whether this round grafted.
    //! The round's draft was built before eviction with its guided capture deferred to the walk; the capture is
    //! owed even if no slot may jump any more.
    bool mCaptureDeferred{false};
};

} // namespace rt
} // namespace trt_edgellm
