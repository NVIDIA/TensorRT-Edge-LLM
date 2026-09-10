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

#include "common/hashUtils.h"
#include "runtime/decoding/decodingStrategy.h"
#include "runtime/decoding/specCommonStateTracker.h"
#include "runtime/state/externalWeightManager.h"

#include <filesystem>
#include <memory>

namespace trt_edgellm
{
namespace rt
{

class DSparkDecoder final : public DecodingStrategy
{
public:
    DSparkDecoder(DecodingRuntimeContext& runtime, std::filesystem::path const& engineDir,
        SpecDecodeDraftingConfig const& draftingConfig, std::unique_ptr<EngineExecutor> draftExecutor,
        ExternalWeightManager draftWeights, cudaStream_t stream);

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

    DecodingKvHeadroom requiredKvHeadroom() const override;

    bool decodeStep(DecodingInferenceContext& context) override;
    bool captureCudaGraphs(cudaStream_t stream) override;
    bool initializeForGeneration(DecodingInferenceContext& context) override;
    std::vector<int32_t> const& commonMaterializedStateLengths() const noexcept override;

    int64_t getRequiredContextMemorySize() const noexcept override;
    void setContextMemory(Tensor& memory) override;

    bool hasSystemPromptKVCache(SystemPromptCacheKey const& key) const override;
    void restoreSystemPromptKVCache(SystemPromptCacheKey const& key, int32_t batchIdx, cudaStream_t stream) override;
    bool runSystemPromptPrefill(DecodingInferenceContext& context) override;
    void saveSystemPromptKVCache(SystemPromptCacheKey const& key, std::string const& prompt,
        std::vector<tokenizer::Rank> const& tokenizedPrompt, int32_t promptIdsLength, cudaStream_t stream) override;

    void resetForNewSequences(Tensor& reuseLengths, cudaStream_t stream) override;
    void onBatchEvict(std::vector<int32_t> const& batchMapping, int32_t oldActiveBatch, int32_t newActiveBatch,
        Tensor& deviceBatchMapping, cudaStream_t stream) override;

private:
    bool runDraftForward(DecodingInferenceContext& context);
    bool runBaseVerification(DecodingInferenceContext& context);
    bool buildTreeVerifyInputs(int32_t activeBatchSize, cudaStream_t stream, bool useConfidence);
    void commitAcceptedTreePath(DecodingInferenceContext& context, int32_t verifySize, int32_t maxAcceptLength);
    void dsparkBiasMarkovGreedy(DecodingInferenceContext& context, int32_t activeBatchSize, int32_t proposalLen);
    void dsparkBiasMarkovSample(DecodingInferenceContext& context, int32_t activeBatchSize, int32_t proposalLen);
    void loadHeadSidecars(std::filesystem::path const& engineDir, cudaStream_t stream);

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
    Tensor mDraftPackedAttentionMask; //!< [B, proposalLen, divUp(proposalLen,32)] INT32
    Tensor mDraftAttentionPosId;      //!< [B, proposalLen] INT32
    Tensor mDraftContextLengths;      //!< [B] INT32
    Tensor mDraftDeltaLenCommit;      //!< [B] INT32
    Tensor mDraftDeltaLens;           //!< [B] INT32

    //! Draft/verify/accepted tokens
    Tensor mDraftTokenIds;        //!< [B, proposalLen] INT32
    Tensor mVerifyTokenIds;       //!< [B, verifyLen] INT32, verifyLen = proposalLen + 1
    Tensor mAcceptedTokenIds;     //!< [B, verifyLen] INT32
    Tensor mAcceptLength;         //!< [B] INT32
    Tensor mHostAcceptLengths;    //!< [B] INT32 (CPU)
    Tensor mHostAcceptedTokenIds; //!< [B, verifyLen] INT32 (CPU)

    SpecCommonStateTracker mCommonStateTracker;
    Tensor mHostDraftInputIds;      //!< [B, proposalLen] INT32 (CPU)
    Tensor mHostLastAcceptedTokens; //!< [B] INT32 (CPU)
    Tensor mHostDeltaLens;          //!< [B] INT32 (CPU)
    Tensor mHostProposalLengths;    //!< [B] INT32 (CPU)

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

    //! Pre-allocated argmax scratch buffer for dsparkGreedyAccept [maxBatch * verifyLen] INT32
    Tensor mArgmaxScratch;

    //! Last accepted token per batch [maxBatch] INT32 (GPU)
    Tensor mLastAcceptedTokens;

    //! DDTree drafting state (draftingTopK > 1): the fanout happens in ddtreeBuild
    //! after drafting, on the stacked per-depth Markov-corrected logits.
    Tensor mStackedMarkovLogits;  //!< [maxBatch, blockSize+1, vocabSize] FP32, row 0 = root placeholder
    Tensor mTreeTokenIds;         //!< [maxBatch, verifySize] INT32 flattened tree token ids
    Tensor mTreeNodeDepths;       //!< [maxBatch, verifySize] INT32 node depths (root = 0)
    Tensor mTreeParentIds;        //!< [maxBatch, verifySize] INT32 parent node indices
    Tensor mTreeNodeScores;       //!< [maxBatch, verifySize] FP32 prefix log-prob scores
    Tensor mValidCounts;          //!< [maxBatch] INT32 valid node counts
    Tensor mVerifyTreeMask;       //!< [maxBatch, verifySize, verifySize] INT8 unpacked accept mask
    Tensor mTreeBuildWorkspace;   //!< ddtreeBuild scratch
    Tensor mAcceptedTokenIndices; //!< [maxBatch, verifySize] INT32 accepted verify-node indices

    //! DSpark Markov/confidence sidecars
    Tensor mMarkovW1;         //!< [vocabSize, markovRank] FP16
    Tensor mMarkovW2;         //!< [vocabSize, markovRank] FP16
    Tensor mConfidenceWeight; //!< [hiddenSize + optional markovRank] FP16
    Tensor mConfidenceBias;   //!< [1] FP16
    bool mHasConfidenceHead{false};
    bool mConfidenceHeadWithMarkov{false};

    //! System prompt KV cache for draft target KV
    hash_utils::HashMap<SystemPromptCacheKey, SystemPromptKVCache> mSystemPromptKVCacheDraft;

    //! DSpark-specific parameters
    bool mUseTree{false};          //!< draftingTopK > 1 selects DDTree drafting
    bool mUseTreeScheduler{false}; //!< scheduler!=off in tree mode: log(conf) bias on ddtree growth scores
    int32_t mProposalLen{7};
    int32_t mVerifyLen{8};
    int32_t mCurrentProposalLen{7};
    int32_t mCurrentVerifyLen{8};
    int32_t mMaskTokenId{151669};
    int32_t mDraftHiddenSize{0};
    int32_t mBaseOutputHiddenDim{0};
    int32_t mDraftVocabSize{0};
    int32_t mMarkovRank{0};
    DSparkSchedulerMode mSchedulerMode{DSparkSchedulerMode::kOff};
    float mConfidenceThreshold{0.0F};
    int32_t mLastBaseVerifyHiddenStride{0};
    int32_t mMinScheduledProposalLen{1};
    int32_t mMaxScheduledProposalLen{0};
};

} // namespace rt
} // namespace trt_edgellm
