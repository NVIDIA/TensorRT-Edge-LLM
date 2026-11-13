/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

// TODO(follow-up): Simplify MTPDecoder to deliver the simplest chain-based
// speculative drafting + verification. AR + tree-based proposal logic should
// live exclusively in EagleDecoder.
class MTPDecoder final : public DecodingStrategy
{
public:
    MTPDecoder(DecodingRuntimeContext& runtime, std::filesystem::path const& engineDir,
        std::unique_ptr<EngineExecutor> draftExecutor, ExternalWeightManager draftWeights, cudaStream_t stream);

    DecodingStrategyKind kind() const noexcept override
    {
        return DecodingStrategyKind::kMTP;
    }

    char const* name() const noexcept override
    {
        return "mtp";
    }

    bool isSpeculative() const noexcept override
    {
        return true;
    }

    DecodingStrategyCapabilities capabilities() const noexcept override
    {
        return {/*.ownsBaseVerificationCudaGraphs=*/false, /*.supportsLosslessSampling=*/false,
            /*.maxSamplingSupport=*/0, /*.fallbackToVanillaForNonGreedySampling=*/mUseTree};
    }

    DecodingKvHeadroomPolicy kvHeadroom() const override;

    bool decodeStep(DecodingInferenceContext& context) override;
    bool captureCudaGraphs(cudaStream_t stream) override;

    //! Hybrid+spec endpoint reuse runs the draft prefill pre-publication (mirrors EagleDecoder). Default MTP keeps its
    //! decode-round-0 draft prefill: this override is a no-op unless context.hybridSpecEndpointReuse is set.
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
    //! @param heldBackTokens Prompt positions at each row's end left for the next accept pass (seatRow holds one).
    bool runDraftModelPrefill(DecodingInferenceContext& context, int32_t heldBackTokens = 0);
    bool constructDraftProposal(DecodingInferenceContext& context);
    bool buildTreeVerifyInputs(int32_t activeBatchSize, cudaStream_t stream);
    //! Grafts the forced chains of the slots that may jump into the drafted verify input, and captures the round's
    //! verify input for the grammar mask.
    void finishVerifyInput(DecodingInferenceContext& context);
    bool runBaseModelVerification(DecodingInferenceContext& context);
    bool runDraftModelAcceptToken(DecodingInferenceContext& context);

    DecodingRuntimeContext& mRuntime;
    HybridCacheManager& mDraftCacheManager;

    std::unique_ptr<EngineExecutor> mDraftExecutor;
    TensorMap mDraftTensorMap;
    //! Externalized draft-engine weights, published into mDraftTensorMap.
    ExternalWeightManager mDraftExternalWeightManager;

    Tensor mDraftProposalSize;
    Tensor mDraftAttentionMask;
    Tensor mDraftTokenIdsFullTable;
    Tensor mDraftTokenScoreFullTable;
    Tensor mDraftTokenPredecessorFullTable;
    Tensor mDraftVocabMappingTable;
    Tensor mDraftRootTokenId;
    Tensor mDraftTokenIdsTable;
    Tensor mDraftTokenScoresTable;
    Tensor mDraftTokenIntermediateScores;
    Tensor mDraftTokenIntermediateParents;
    Tensor mAcceptedTokenIds;
    Tensor mAcceptedTokenIndices;
    Tensor mAcceptLength;
    Tensor mHostAcceptLengths;
    Tensor mHostSeatStaging; //!< [B, 2] INT32 pinned; seatRow() only
    Tensor mHostAcceptedTokenIds;
    //! Target hidden states of each row's last accepted span, [maxBatch * maxAcceptDepth, baseOutputHiddenDim]; a
    //! seated prefill overwrites the pipeline's baseHiddenStates between decode steps.
    Tensor mAcceptedHiddenStates;
    //! Every row's draft has consumed its prompt, so its next draft input is its last accepted span.
    bool mDraftPrimed{false};

    //! Tree drafting (draftTopK > 1): the chain drafter keeps one full logits row
    //! per proposal depth and the greedy tree builder grows a prefix-closed,
    //! score-prioritized verify tree from them.
    bool mUseTree{false};
    Tensor mStackedDraftLogits; //!< [maxBatch, draftSteps+1, draftVocab] per-depth chain logits
    Tensor mTreeTokenIds;       //!< [maxBatch, verifyTokenCount] flattened tree token ids
    Tensor mTreeNodeScores;     //!< [maxBatch, verifyTokenCount] prefix log-prob scores
    Tensor mValidCounts;        //!< [maxBatch] valid node counts
    Tensor mTreeBuildWorkspace; //!< ddtreeBuild temporary workspace

    SpecCommonStateTracker mCommonStateTracker;
    JumpGraftSession mJump; //!< The grafter of a deployment that can jump, and whether this round grafted.
};

} // namespace rt
} // namespace trt_edgellm
