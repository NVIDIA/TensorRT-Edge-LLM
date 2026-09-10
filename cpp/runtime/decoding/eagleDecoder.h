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

#include "common/hashUtils.h"
#include "runtime/decoding/decodingStrategy.h"
#include "runtime/decoding/specCommonStateTracker.h"
#include "runtime/state/externalWeightManager.h"

#include <filesystem>
#include <memory>
#include <unordered_map>

namespace trt_edgellm
{
namespace rt
{

class EagleDecoder final : public DecodingStrategy
{
public:
    EagleDecoder(DecodingRuntimeContext& runtime, std::filesystem::path const& engineDir,
        SpecDecodeDraftingConfig const& draftingConfig, std::unique_ptr<EngineExecutor> draftExecutor,
        ExternalWeightManager draftWeights, cudaStream_t stream);

    DecodingStrategyKind kind() const noexcept override
    {
        return DecodingStrategyKind::kEAGLE;
    }

    char const* name() const noexcept override
    {
        return "eagle";
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
    bool runDraftModelPrefill(DecodingInferenceContext& context);
    bool constructDraftProposal(DecodingInferenceContext& context);
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
    Tensor mHostAcceptedTokenIds;
    SpecCommonStateTracker mCommonStateTracker;

    hash_utils::HashMap<SystemPromptCacheKey, SystemPromptKVCache> mSystemPromptKVCacheDraft;
};

} // namespace rt
} // namespace trt_edgellm
