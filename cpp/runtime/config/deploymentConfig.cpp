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

#include "runtime/config/deploymentConfig.h"

#include "common/checkMacros.h"

#include "common/logger.h"
#include "common/pagedKvTypes.h"
#include "common/trtUtils.h"
#include "kernels/gdnKernels/gdnTreeChunkKernels.h"

#include <algorithm>

namespace trt_edgellm
{
namespace rt
{
namespace
{
bool isAttentionLayer(LLMEngineConfig const& cfg, int32_t absLayerIdx)
{
    return absLayerIdx >= 0 && absLayerIdx < static_cast<int32_t>(cfg.layerTypes.size())
        && cfg.layerTypes[absLayerIdx] == HybridCacheManager::LayerType::kAttention;
}

int32_t attentionLocalToAbsolute(LLMEngineConfig const& cfg, int32_t localLayerIdx)
{
    int32_t localIdx = 0;
    for (int32_t absIdx = 0; absIdx < static_cast<int32_t>(cfg.layerTypes.size()); ++absIdx)
    {
        if (cfg.layerTypes[absIdx] != HybridCacheManager::LayerType::kAttention)
        {
            continue;
        }
        if (localIdx == localLayerIdx)
        {
            return absIdx;
        }
        ++localIdx;
    }
    return -1;
}

int32_t attentionAbsoluteToLocal(LLMEngineConfig const& cfg, int32_t absLayerIdx)
{
    int32_t localIdx = 0;
    for (int32_t i = 0; i < static_cast<int32_t>(cfg.layerTypes.size()); ++i)
    {
        if (cfg.layerTypes[i] != HybridCacheManager::LayerType::kAttention)
        {
            continue;
        }
        if (i == absLayerIdx)
        {
            return localIdx;
        }
        ++localIdx;
    }
    return -1;
}

void validateDFlashDraftTargetLayerIds(LLMEngineConfig const& base, LLMEngineConfig const& draft)
{
    for (int32_t layerId : draft.dflashTargetLayerIds)
    {
        ELLM_CHECK(layerId >= 0 && layerId < base.numDecoderLayers,
            "DFlash draft target layer id " + std::to_string(layerId) + " is outside [0, base.num_hidden_layers).");
    }
}

void validateDFlashDraftKVPool(LLMEngineConfig const& draft)
{
    int32_t const floorPages = computeKvPoolFloorPages(draft.maxSupportedBatchSize, draft.maxKVCacheCapacity);
    ELLM_CHECK(draft.kvPoolPages == floorPages,
        "DFlash draft KV is identity-only and requires kvPoolPages=" + std::to_string(floorPages) + "; got "
            + std::to_string(draft.kvPoolPages) + ".");
}

void validateGemma4MTPConfig(LLMEngineConfig const& base, LLMEngineConfig& draft)
{
    ELLM_CHECK(
        base.specDecodeType == SpecDecodeMode::kGemma4MTP, "Gemma4 MTP validation requires a gemma4_mtp base config.");
    ELLM_CHECK(base.isSpecDecodeBase, "Gemma4 MTP base config must be exported with engine_role=base.");
    ELLM_CHECK(draft.specDecodeType == SpecDecodeMode::kGemma4MTP,
        "Gemma4 MTP draft config must set spec_decode_type=gemma4_mtp.");
    ELLM_CHECK(draft.modelType == "gemma4_assistant", "Gemma4 MTP draft config must set model=gemma4_assistant.");
    ELLM_CHECK(draft.baseModelHiddenSize == base.hiddenSize,
        "Gemma4 MTP draft base_model_hidden_size (" + std::to_string(draft.baseModelHiddenSize)
            + ") must match base hidden_size (" + std::to_string(base.hiddenSize) + ").");
    ELLM_CHECK(
        draft.vocabSize == base.vocabSize, "Gemma4 MTP draft vocab_size/draft_vocab_size must match base vocab_size.");
    ELLM_CHECK(draft.sharesTargetKV && !draft.hasOwnKVCache,
        "Gemma4 MTP assistant must share target KV and must not own a draft KV cache.");
    ELLM_CHECK(draft.returnsFeedbackHidden, "Gemma4 MTP assistant must return backbone-space hidden_states feedback.");
    ELLM_CHECK(draft.constantDraftPositions, "Gemma4 MTP assistant must set constant_draft_positions=true.");
    ELLM_CHECK(base.kvCacheDtype == draft.kvCacheDtype,
        std::string("Gemma4 MTP base/draft KV dtype mismatch: base=") + getDataTypeString(base.kvCacheDtype)
            + ", draft=" + getDataTypeString(draft.kvCacheDtype) + ".");
    ELLM_CHECK(base.kvPoolPages == draft.kvPoolPages,
        "Gemma4 MTP assistant shares base KV and requires matching kvPoolPages: base="
            + std::to_string(base.kvPoolPages) + ", draft=" + std::to_string(draft.kvPoolPages) + ".");
    ELLM_CHECK(static_cast<int32_t>(draft.gemma4MTPKVSharingMap.size()) == draft.numAttentionLayers,
        "Gemma4 MTP kv_sharing_map size (" + std::to_string(draft.gemma4MTPKVSharingMap.size())
            + ") must equal draft attention layer count (" + std::to_string(draft.numAttentionLayers) + ").");
    ELLM_CHECK(static_cast<int32_t>(draft.kvLayerConfigs.size()) == draft.numAttentionLayers,
        "Gemma4 MTP draft KV layer config size (" + std::to_string(draft.kvLayerConfigs.size())
            + ") must equal draft attention layer count (" + std::to_string(draft.numAttentionLayers) + ").");

    std::vector<bool> seenAssistantLayers(draft.numAttentionLayers, false);
    for (auto& entry : draft.gemma4MTPKVSharingMap)
    {
        ELLM_CHECK(entry.assistantLayerIdx >= 0 && entry.assistantLayerIdx < draft.numAttentionLayers,
            "Gemma4 MTP kv_sharing_map assistant layer " + std::to_string(entry.assistantLayerIdx)
                + " is outside draft attention-layer range.");
        ELLM_CHECK(!seenAssistantLayers[entry.assistantLayerIdx],
            "Gemma4 MTP kv_sharing_map has duplicate assistant layer " + std::to_string(entry.assistantLayerIdx) + ".");
        seenAssistantLayers[entry.assistantLayerIdx] = true;

        ELLM_CHECK(entry.targetAttentionLayerIdx >= 0
                && entry.targetAttentionLayerIdx < static_cast<int32_t>(base.kvLayerConfigs.size()),
            "Gemma4 MTP kv_sharing_map target attention layer " + std::to_string(entry.targetAttentionLayerIdx)
                + " is outside base attention-layer range.");

        entry.targetAbsoluteLayerIdx = attentionLocalToAbsolute(base, entry.targetAttentionLayerIdx);

        ELLM_CHECK(isAttentionLayer(base, entry.targetAbsoluteLayerIdx),
            "Gemma4 MTP kv_sharing_map target absolute layer " + std::to_string(entry.targetAbsoluteLayerIdx)
                + " is not a valid target attention layer.");

        int32_t const expectedLocal = attentionAbsoluteToLocal(base, entry.targetAbsoluteLayerIdx);
        ELLM_CHECK(entry.targetAttentionLayerIdx == expectedLocal,
            "Gemma4 MTP kv_sharing_map target local layer " + std::to_string(entry.targetAttentionLayerIdx)
                + " does not match target absolute layer " + std::to_string(entry.targetAbsoluteLayerIdx) + ".");

        auto const& assistantKV = draft.kvLayerConfigs[entry.assistantLayerIdx];
        auto const& targetKV = base.kvLayerConfigs[entry.targetAttentionLayerIdx];
        ELLM_CHECK(assistantKV.numKVHeads == targetKV.numKVHeads,
            "Gemma4 MTP shared KV num heads mismatch for assistant layer " + std::to_string(entry.assistantLayerIdx)
                + " -> target layer " + std::to_string(entry.targetAttentionLayerIdx) + ": assistant="
                + std::to_string(assistantKV.numKVHeads) + ", target=" + std::to_string(targetKV.numKVHeads) + ".");
        ELLM_CHECK(assistantKV.headDim == targetKV.headDim,
            "Gemma4 MTP shared KV head dim mismatch for assistant layer " + std::to_string(entry.assistantLayerIdx)
                + " -> target layer " + std::to_string(entry.targetAttentionLayerIdx) + ": assistant="
                + std::to_string(assistantKV.headDim) + ", target=" + std::to_string(targetKV.headDim) + ".");
    }
    for (int32_t assistantLayerIdx = 0; assistantLayerIdx < draft.numAttentionLayers; ++assistantLayerIdx)
    {
        ELLM_CHECK(seenAssistantLayers[assistantLayerIdx],
            "Gemma4 MTP kv_sharing_map is missing assistant layer " + std::to_string(assistantLayerIdx) + ".");
    }
}

int32_t resolveDFlashBlockSize(
    LLMEngineConfig const& base, LLMEngineConfig const& draft, SpecDecodeDraftingConfig const& draftingConfig)
{
    if (draftingConfig.dflashBlockSize > 0)
    {
        return draftingConfig.dflashBlockSize;
    }
    if (draft.dflashBlockSize > 0)
    {
        return draft.dflashBlockSize;
    }
    return base.dflashBlockSize;
}
} // namespace

int32_t DeploymentConfig::maxRuntimeBatchSize() const
{
    // When base and draft engines were built with different max batch sizes, fall
    // back to the smaller — the current runtime cannot drive either engine beyond
    // its engine-declared capacity, so the common ceiling is the safe choice. A
    // stricter "must match exactly" policy belongs to a follow-up that pairs with
    // an export-side guarantee; today we degrade gracefully and warn.
    int32_t const baseMax = base.maxSupportedBatchSize;
    if (!draft.has_value())
    {
        return baseMax;
    }
    int32_t const draftMax = draft->maxSupportedBatchSize;
    if (draftMax != baseMax)
    {
        LOG_WARNING(
            "base.maxSupportedBatchSize=%d vs draft.maxSupportedBatchSize=%d; "
            "using the smaller (%d). Re-export both engines against the same config to silence this warning.",
            baseMax, draftMax, std::min(baseMax, draftMax));
    }
    return std::min(baseMax, draftMax);
}

int32_t DeploymentConfig::effectiveMaxDraftProposalSize() const
{
    ELLM_CHECK(specConfig.has_value(),
        "effectiveMaxDraftProposalSize: speculative decoding configuration is not set. "
        "Guard the call with specConfig.has_value().");
    return std::max(specConfig->maxDraftProposalSize, specConfig->verifySize);
}

SpecDecodeMode DeploymentConfig::specDecodeMode() const noexcept
{
    if (!draft.has_value() || !specConfig.has_value())
    {
        return SpecDecodeMode::kNONE;
    }
    return base.specDecodeType;
}

int32_t DeploymentConfig::maxAcceptedTokensPerRound() const
{
    switch (specDecodeMode())
    {
    case SpecDecodeMode::kNONE: return 1;
    case SpecDecodeMode::kEAGLE:
    case SpecDecodeMode::kMTP:
    case SpecDecodeMode::kGemma4MTP: return specConfig->draftingStep + 1;
    case SpecDecodeMode::kDFlash: return std::min(specConfig->verifySize, specConfig->dflashBlockSize);
    }
    ELLM_CHECK(false, "maxAcceptedTokensPerRound: unhandled SpecDecodeMode");
    return 1;
}

DeploymentConfig createDeploymentConfig(std::filesystem::path const& baseConfigPath,
    std::optional<std::filesystem::path> const& draftConfigPath,
    std::optional<SpecDecodeDraftingConfig> const& draftingConfig)
{
    DeploymentConfig cfg;

    // --- Structural precondition: drafting cannot be set without draft ---
    ELLM_CHECK(!draftingConfig.has_value() || draftConfigPath.has_value(),
        "drafting configuration was provided but no draftConfigPath was set. "
        "SpecDecode drafting requires a draft engine config.");

    // --- Parse base ---
    cfg.base = parseEngineConfig(baseConfigPath);

    // --- Parse draft (if present) ---
    if (draftConfigPath.has_value())
    {
        cfg.draft = parseDraftEngineConfig(*draftConfigPath);
    }

    if (cfg.base.specDecodeType == SpecDecodeMode::kDFlash && cfg.draft.has_value())
    {
        validateDFlashDraftKVPool(*cfg.draft);
        validateDFlashDraftTargetLayerIds(cfg.base, *cfg.draft);
    }
    if (cfg.base.specDecodeType == SpecDecodeMode::kGemma4MTP && cfg.draft.has_value())
    {
        validateGemma4MTPConfig(cfg.base, *cfg.draft);
    }

    // Each engine's builder_config carries its own sequence budget. The base emits
    // `max_verify_tree_size` (its verification budget); the draft emits
    // `max_draft_tree_size` (its proposal budget), so consumers read each field
    // from the owning side. Shared physical resources are validated above by the
    // mode-specific checks.

    // --- Build consolidated SpecDecodeConfig and validate drafting limits ---
    if (draftingConfig.has_value())
    {
        ELLM_CHECK(cfg.base.specDecodeType != SpecDecodeMode::kNONE,
            "drafting configuration was provided but base config is not a speculative decoding base engine.");
        ELLM_CHECK(cfg.draft.has_value() && cfg.draft->specDecodeType == cfg.base.specDecodeType,
            "base and draft speculative decoding modes must match.");

        // Positivity: each drafting field must be >= 1. Rejecting zero/negative
        // up front gives downstream shape arithmetic a clean invariant and
        // produces a clearer error than a far-away bind-time mismatch.
        auto const requirePositiveField = [](int32_t value, char const* name) {
            ELLM_CHECK(
                value > 0, std::string("drafting.") + name + "=" + std::to_string(value) + " must be positive (>= 1).");
        };
        requirePositiveField(draftingConfig->draftingTopK, "draftingTopK");
        requirePositiveField(draftingConfig->draftingStep, "draftingStep");
        requirePositiveField(draftingConfig->verifySize, "verifySize");
        ELLM_CHECK(draftingConfig->dflashBlockSize >= 0,
            "drafting.dflashBlockSize=" + std::to_string(draftingConfig->dflashBlockSize)
                + " must be non-negative; use 0 to infer from DFlash engine config.");

        SpecDecodeConfig specConfig;
        // baseOutputHiddenDim comes from the draft config's `base_model_hidden_size`
        // (= base.hiddenSize * 3 for EAGLE-3, = base.hiddenSize for MTP). Don't
        // derive from base.hiddenSize directly — that's correct only for EAGLE-3.
        specConfig.baseOutputHiddenDim = cfg.draft->baseModelHiddenSize;
        specConfig.draftHiddenSize = cfg.draft->hiddenSize;
        specConfig.maxVerifySize = cfg.base.maxVerifyTreeSize;
        specConfig.maxDraftProposalSize = cfg.draft->maxDraftTreeSize;
        specConfig.draftingTopK = draftingConfig->draftingTopK;
        specConfig.draftingStep = draftingConfig->draftingStep;
        specConfig.verifySize = draftingConfig->verifySize;
        specConfig.dflashBlockSize = draftingConfig->dflashBlockSize;

        if (cfg.base.specDecodeType == SpecDecodeMode::kDFlash)
        {
            static constexpr int32_t kDFlashDDTreeMaxVerifySize = 128;
            static constexpr int32_t kDFlashDDTreeMaxCandidateTopK = 8;
            static constexpr int32_t kDFlashDDTreeMaxAcceptedPathLength = 16;
            static constexpr int32_t kDFlashHybridMaxBlockSize = 16;

            specConfig.dflashBlockSize = resolveDFlashBlockSize(cfg.base, *cfg.draft, *draftingConfig);
            ELLM_CHECK(specConfig.dflashBlockSize > 0,
                "DFlash requires resolved dflashBlockSize > 0. Set dflashBlockSize or export a draft "
                "dflash_config.block_size.");

            ELLM_CHECK(specConfig.draftingStep == 1,
                "DFlash supports draftingStep=1 only because one DFlash draft forward emits the full block. Use "
                "dflashBlockSize to control the DFlash draft horizon.");
            ELLM_CHECK(specConfig.dflashBlockSize <= specConfig.maxDraftProposalSize,
                "DFlash dflashBlockSize=" + std::to_string(specConfig.dflashBlockSize)
                    + " exceeds draft.maxDraftTreeSize=" + std::to_string(specConfig.maxDraftProposalSize)
                    + ". DFlash drafts one full block per iteration.");

            bool const useBranchingTree = specConfig.draftingTopK > 1;
            if (!useBranchingTree)
            {
                specConfig.verifySize = specConfig.dflashBlockSize;
            }
            else
            {
                ELLM_CHECK(specConfig.dflashBlockSize >= 2,
                    "DFlash branching DDTree requires dflashBlockSize >= 2 because node 0 is the root and the "
                    "remaining draft positions provide child candidates.");
                ELLM_CHECK(specConfig.draftingTopK < specConfig.verifySize,
                    "DFlash DDTree candidateTopK=" + std::to_string(specConfig.draftingTopK)
                        + " must be less than verifySize=" + std::to_string(specConfig.verifySize)
                        + " because the root consumes one verification node.");
                ELLM_CHECK(specConfig.draftingTopK <= kDFlashDDTreeMaxCandidateTopK,
                    "DFlash DDTree candidateTopK=" + std::to_string(specConfig.draftingTopK)
                        + " exceeds the current DDTree candidateTopK limit of "
                        + std::to_string(kDFlashDDTreeMaxCandidateTopK) + ".");
                ELLM_CHECK(specConfig.draftingTopK <= cfg.draft->outputVocabSize,
                    "DFlash draftingTopK=" + std::to_string(specConfig.draftingTopK)
                        + " exceeds draft output vocabulary size=" + std::to_string(cfg.draft->outputVocabSize) + ".");
                ELLM_CHECK(specConfig.verifySize <= kDFlashDDTreeMaxVerifySize,
                    "DFlash DDTree verifySize=" + std::to_string(specConfig.verifySize)
                        + " exceeds node budget limit of " + std::to_string(kDFlashDDTreeMaxVerifySize) + ".");
                int32_t const maxAcceptedPathLength = std::min(specConfig.dflashBlockSize, specConfig.verifySize);
                ELLM_CHECK(maxAcceptedPathLength <= kDFlashDDTreeMaxAcceptedPathLength,
                    "DFlash DDTree max accepted path length=" + std::to_string(maxAcceptedPathLength)
                        + " exceeds indexed commit path limit of " + std::to_string(kDFlashDDTreeMaxAcceptedPathLength)
                        + ".");
            }
            bool const hasLinearAttnLayers = (cfg.base.numLinearAttnLayers > 0);
            if (hasLinearAttnLayers)
            {
                ELLM_CHECK(specConfig.dflashBlockSize <= kDFlashHybridMaxBlockSize,
                    "DFlash dflashBlockSize=" + std::to_string(specConfig.dflashBlockSize)
                        + " exceeds Qwen3.5 GDN/causal-conv intermediate-state depth limit of "
                        + std::to_string(kDFlashHybridMaxBlockSize) + ".");
            }
        }
        else
        {
            // In practice both `draftingStep` and `draftingTopK` are <= ~64 (bounded
            // downstream by `maxDraftProposalSize`, which is tens, not millions), so
            // int32 multiplication is overflow-safe; keeping it in int32 avoids
            // widening noise.
            // Qwen-style MTP always drafts one greedy chain and uses draftingTopK only as the verification-tree
            // candidate fanout. Other speculative modes still expand draftingTopK rows in the draft engine.
            bool const mtpUsesChainDraftEngine = cfg.base.specDecodeType == SpecDecodeMode::kMTP;
            int32_t const requiredDraftInputSize
                = mtpUsesChainDraftEngine ? specConfig.draftingStep : specConfig.draftingStep * specConfig.draftingTopK;
            std::string const requiredDraftInputDescription = mtpUsesChainDraftEngine
                ? "MTP drafting.draftingStep=" + std::to_string(specConfig.draftingStep)
                : "drafting.draftingStep=" + std::to_string(specConfig.draftingStep) + " * drafting.draftingTopK="
                    + std::to_string(specConfig.draftingTopK) + " = " + std::to_string(requiredDraftInputSize);

            ELLM_CHECK(requiredDraftInputSize <= specConfig.maxDraftProposalSize,
                requiredDraftInputDescription
                    + " exceeds draft.maxDraftTreeSize=" + std::to_string(specConfig.maxDraftProposalSize)
                    + ". Drafting configuration exceeds engine proposal size capability.");
            ELLM_CHECK(
                specConfig.dflashBlockSize == 0, "dflashBlockSize can only be set when spec_decode_type=dflash.");

            if (cfg.base.specDecodeType == SpecDecodeMode::kMTP)
            {
                // MTP base verification currently reuses EAGLE utility kernels for accept, KV commit,
                // and hidden-state compaction. Those kernels support maxDepth <= 9. Each round
                // accepts at most draftingStep matched proposals plus one bonus token.
                static constexpr int32_t kMTPMaxAcceptDepthForCurrentEagleUtilityKernels = 9;
                int32_t const maxAcceptDepth = specConfig.draftingStep + 1;
                ELLM_CHECK(maxAcceptDepth <= kMTPMaxAcceptDepthForCurrentEagleUtilityKernels,
                    "MTP max accept depth (draftingStep+1)=" + std::to_string(maxAcceptDepth)
                        + " exceeds the current MTP EAGLE utility kernel max depth of "
                        + std::to_string(kMTPMaxAcceptDepthForCurrentEagleUtilityKernels)
                        + ". Extend eagleUtilKernels before using larger MTP draft steps.");

                // The chain drafter keeps one full logits row per depth and ddtreeBuild grows a prefix-closed,
                // score-prioritized tree of verifySize nodes with candidateFanout=draftingTopK. A fanout of one
                // produces the same token chain as linear MTP while preserving one engine/runtime contract.
                static constexpr int32_t kMTPTreeMaxVerifySize = 128;
                static constexpr int32_t kMTPTreeMaxCandidateFanout = 8;
                ELLM_CHECK(specConfig.draftingTopK < specConfig.verifySize,
                    "MTP tree draftingTopK=" + std::to_string(specConfig.draftingTopK)
                        + " must be less than verifySize=" + std::to_string(specConfig.verifySize)
                        + " because the root consumes one verification node.");
                ELLM_CHECK(specConfig.draftingTopK <= kMTPTreeMaxCandidateFanout,
                    "MTP tree draftingTopK=" + std::to_string(specConfig.draftingTopK)
                        + " exceeds the tree-build kernel candidate fanout limit of "
                        + std::to_string(kMTPTreeMaxCandidateFanout) + ".");
                ELLM_CHECK(specConfig.draftingTopK <= cfg.draft->outputVocabSize,
                    "MTP tree draftingTopK=" + std::to_string(specConfig.draftingTopK)
                        + " exceeds draft output vocabulary size=" + std::to_string(cfg.draft->outputVocabSize) + ".");
                ELLM_CHECK(specConfig.verifySize <= kMTPTreeMaxVerifySize,
                    "MTP tree verifySize=" + std::to_string(specConfig.verifySize)
                        + " exceeds the tree-build kernel node budget of " + std::to_string(kMTPTreeMaxVerifySize)
                        + ".");
                if (specConfig.draftingTopK == 1)
                {
                    ELLM_CHECK(specConfig.verifySize == maxAcceptDepth,
                        "MTP fanout-one tree requires verifySize=draftingStep+1 because it contains exactly one node "
                        "per proposal depth. Got verifySize="
                            + std::to_string(specConfig.verifySize)
                            + ", draftingStep=" + std::to_string(specConfig.draftingStep)
                            + ", expected verifySize=" + std::to_string(maxAcceptDepth) + ".");
                }
                // The accepted path is bounded by the replay-commit kernel.
                static constexpr int32_t kMTPHybridMaxProposalDepth = 16;
                bool const hasLinearAttnLayers = (cfg.base.numLinearAttnLayers > 0);
                if (hasLinearAttnLayers)
                {
                    ELLM_CHECK(maxAcceptDepth <= kMTPHybridMaxProposalDepth,
                        "MTP max accept depth (draftingStep+1)=" + std::to_string(maxAcceptDepth)
                            + " exceeds Qwen3.5 GDN/causal-conv intermediate-state depth limit of "
                            + std::to_string(kMTPHybridMaxProposalDepth) + ".");
                }
            }
        }

        bool const usesCompactGdnReplay = cfg.base.numLinearAttnLayers > 0
            && (cfg.base.specDecodeType == SpecDecodeMode::kMTP || cfg.base.specDecodeType == SpecDecodeMode::kDFlash);
        if (usesCompactGdnReplay)
        {
            ELLM_CHECK(specConfig.maxVerifySize <= kernel::kGDN_TREE_CHUNK_MAX_NODES,
                "Hybrid base.maxVerifyTreeSize=" + std::to_string(specConfig.maxVerifySize)
                    + " exceeds compact GDN verification limit " + std::to_string(kernel::kGDN_TREE_CHUNK_MAX_NODES)
                    + ".");
            ELLM_CHECK(specConfig.verifySize <= kernel::kGDN_TREE_CHUNK_MAX_NODES,
                "Hybrid drafting.verifySize=" + std::to_string(specConfig.verifySize)
                    + " exceeds compact GDN verification limit " + std::to_string(kernel::kGDN_TREE_CHUNK_MAX_NODES)
                    + ".");
        }

        ELLM_CHECK(specConfig.verifySize <= specConfig.maxVerifySize,
            "drafting.verifySize=" + std::to_string(specConfig.verifySize)
                + " exceeds base.maxVerifyTreeSize=" + std::to_string(specConfig.maxVerifySize)
                + ". Verification size exceeds base engine maximum verification size.");

        if (cfg.base.specDecodeType == SpecDecodeMode::kGemma4MTP)
        {
            ELLM_CHECK(specConfig.draftingTopK == 1,
                "Gemma4 MTP currently supports greedy chain drafting only; draftingTopK must be 1.");
            ELLM_CHECK(specConfig.verifySize == specConfig.draftingStep + 1,
                "Gemma4 MTP verifySize must equal draftingStep + 1 to include the root token and all draft tokens.");
        }

        cfg.specConfig = specConfig;
    }

    return cfg;
}

} // namespace rt
} // namespace trt_edgellm
