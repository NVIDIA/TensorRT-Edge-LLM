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
#include "runtime/state/externalWeightManager.h"

#include <filesystem>
#include <memory>
#include <unordered_map>

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
    Gemma4MTPDecoder(DecodingRuntimeContext& runtime, SpecDecodeDraftingConfig const& draftingConfig,
        std::unique_ptr<EngineExecutor> draftExecutor, ExternalWeightManager draftWeights, cudaStream_t stream);

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

    DecodingKvHeadroom requiredKvHeadroom() const override;

    bool decodeStep(DecodingInferenceContext& context) override;
    bool captureCudaGraphs(cudaStream_t stream) override;

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
    //! Seed preparation + assistant draft chain, timed as one draft-proposal stage.
    bool runDraftProposal(DecodingInferenceContext& context);
    bool prepareSeed(DecodingInferenceContext& context);
    bool runAssistantDraftChain(DecodingInferenceContext& context);
    bool runBaseVerification(DecodingInferenceContext& context);
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
    Tensor mDraftTokenIds;                    //!< [B, specDraftStep] INT32 assistant draft chain.
    Tensor mVerifyTokenIds;                   //!< [B, specDraftStep + 1] INT32 root plus assistant draft chain.
    Tensor mAcceptedTokenIds;     //!< [B, specDraftStep + 1] INT32 append-only output tokens, root excluded.
    Tensor mAcceptedTokenIndices; //!< [B, specDraftStep + 1] INT32 target verify KV indices.
    Tensor mAcceptLength;         //!< [B] INT32 per-batch accepted output length.
    Tensor mHostAcceptLengths;    //!< [B] INT32 host staging.
    Tensor mHostAcceptedTokenIds; //!< [B, specDraftStep + 1] INT32 host staging.
    Tensor mArgmaxScratch;        //!< [B * (specDraftStep + 1)] INT32 accept argmax scratch.

    hash_utils::HashMap<SystemPromptCacheKey, bool> mSystemPromptCacheKeys;
};

} // namespace rt
} // namespace trt_edgellm
