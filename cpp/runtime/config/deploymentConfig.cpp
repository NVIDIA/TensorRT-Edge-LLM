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
#include "common/pagedKvTypes.h"
#include "kernels/speculative/ddtreeKernels.h"

#include "common/logger.h"
#include "common/trtUtils.h"

#include <algorithm>
#include <limits>
#include <string>

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

void validateCachedDraftTargetLayerIds(LLMEngineConfig const& base, LLMEngineConfig const& draft, char const* modeName)
{
    for (int32_t layerId : draft.specTargetLayerIds)
    {
        ELLM_CHECK(layerId >= 0 && layerId < base.numDecoderLayers,
            std::string(modeName) + " draft target layer id " + std::to_string(layerId)
                + " is outside [0, base.num_hidden_layers).");
    }
}

void validateEagleConfig(LLMEngineConfig const& base, LLMEngineConfig const& draft)
{
    ELLM_CHECK(!base.specTargetLayerIds.empty(), "EAGLE requires an explicit base conditioning-layer contract.");
    std::vector<int32_t> sortedTargetLayers = base.specTargetLayerIds;
    std::sort(sortedTargetLayers.begin(), sortedTargetLayers.end());
    ELLM_CHECK(sortedTargetLayers.front() >= 0 && sortedTargetLayers.back() < base.numDecoderLayers
            && std::adjacent_find(sortedTargetLayers.begin(), sortedTargetLayers.end()) == sortedTargetLayers.end(),
        "EAGLE conditioning layer IDs must be unique and within the base decoder-layer range.");
    int64_t const expectedConditioningSize
        = static_cast<int64_t>(base.hiddenSize) * static_cast<int64_t>(base.specTargetLayerIds.size());
    ELLM_CHECK(draft.baseModelHiddenSize == expectedConditioningSize,
        "EAGLE base_model_hidden_size does not match the base hidden size and conditioning-layer count.");
}

void requireMinimumActiveKVPool(LLMEngineConfig const& config, char const* engineLabel)
{
    int64_t const computedMinimumActivePages
        = computeMinimumKvPoolPages(config.maxSupportedBatchSize, config.maxKVCacheCapacity);
    ELLM_CHECK(computedMinimumActivePages <= kMAX_KV_POOL_PAGES,
        std::string(engineLabel) + " minimum active pages exceed the largest int32-addressable paged-KV pool.");
    int32_t const minimumActivePages = static_cast<int32_t>(computedMinimumActivePages);
    ELLM_CHECK(config.kvPoolPages == minimumActivePages,
        std::string(engineLabel)
            + " does not support cross-request retention and requires max_kv_pool_pages to equal the "
              "minimum active pages ("
            + std::to_string(minimumActivePages) + "); got " + std::to_string(config.kvPoolPages) + ".");
}

void validateKVPoolMode(DeploymentConfig const& deployment)
{
    SpecDecodeMode const mode = deployment.base.specDecodeType;
    bool const nonHybridEagle = mode == SpecDecodeMode::kEAGLE && deployment.base.numLinearAttnLayers == 0;
    bool const attentionOnlyManagedSpec = deployment.base.numLinearAttnLayers == 0
        && (mode == SpecDecodeMode::kGemma4MTP || mode == SpecDecodeMode::kDFlash || mode == SpecDecodeMode::kJetSpec
            || mode == SpecDecodeMode::kDSpark);
    bool const baseSupportsCrossRequestRetention = !deployment.base.kvLayerConfigs.empty()
        && (mode == SpecDecodeMode::kNONE || nonHybridEagle || attentionOnlyManagedSpec);
    if (!baseSupportsCrossRequestRetention)
    {
        requireMinimumActiveKVPool(deployment.base, "base engine");
    }

    if (deployment.draft.has_value())
    {
        bool const draftSupportsCrossRequestRetention
            = !deployment.draft->kvLayerConfigs.empty() && (nonHybridEagle || attentionOnlyManagedSpec);
        if (!draftSupportsCrossRequestRetention)
        {
            requireMinimumActiveKVPool(*deployment.draft, "draft engine");
        }
    }
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

    // Paged shared-KV geometry must be IDENTICAL: the assistant's engine profiles fix the pool page
    // count (min=opt=max) and the kv_page_table width from ITS OWN build limits, while the runtime
    // binds the TARGET's pool and page table to those bindings. Unequal build limits therefore
    // cannot be reconciled at runtime (there is no min() escape hatch as there is for batch size),
    // so reject them here with the numbers instead of failing deep inside TensorRT shape checks.
    int32_t const basePagesPerSeq = rt::computeMaxPagesPerSeq(base.maxKVCacheCapacity);
    int32_t const draftPagesPerSeq = rt::computeMaxPagesPerSeq(draft.maxKVCacheCapacity);
    ELLM_CHECK(base.kvPoolPages == draft.kvPoolPages && basePagesPerSeq == draftPagesPerSeq,
        "Gemma4 MTP shared-KV pool geometry mismatch: base engine (maxBatchSize="
            + std::to_string(base.maxSupportedBatchSize) + ", maxKVCacheCapacity="
            + std::to_string(base.maxKVCacheCapacity) + ") has numPages=" + std::to_string(base.kvPoolPages)
            + "/maxPagesPerSeq=" + std::to_string(basePagesPerSeq)
            + ", assistant engine (maxBatchSize=" + std::to_string(draft.maxSupportedBatchSize)
            + ", maxKVCacheCapacity=" + std::to_string(draft.maxKVCacheCapacity) + ") has numPages="
            + std::to_string(draft.kvPoolPages) + "/maxPagesPerSeq=" + std::to_string(draftPagesPerSeq)
            + ". Rebuild the assistant engine with matching pool pages and page-table width.");
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
    if (draft.specDraftBlockSize > 0)
    {
        return draft.specDraftBlockSize;
    }
    return base.specDraftBlockSize;
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
    if (base.isDiffusionBackbone)
    {
        return std::max(1, base.diffusionCanvasLength);
    }
    switch (specDecodeMode())
    {
    case SpecDecodeMode::kNONE: return 1;
    case SpecDecodeMode::kEAGLE:
    case SpecDecodeMode::kMTP:
    case SpecDecodeMode::kGemma4MTP: return specConfig->draftingStep + 1;
    case SpecDecodeMode::kDFlash:
    case SpecDecodeMode::kJetSpec: return std::min(specConfig->verifySize, specConfig->dflashBlockSize);
    case SpecDecodeMode::kDSpark: return specConfig->verifySize;
    }
    ELLM_CHECK(false, "maxAcceptedTokensPerRound: unhandled SpecDecodeMode");
    return 1;
}

DeploymentConfig createDeploymentConfig(std::filesystem::path const& baseConfigPath,
    std::optional<std::filesystem::path> const& draftConfigPath,
    std::optional<SpecDecodeDraftingConfig> const& draftingConfig, std::optional<int32_t> rank,
    std::optional<int32_t> expectedWorldSize)
{
    DeploymentConfig cfg;

    // --- Structural precondition: drafting cannot be set without draft ---
    ELLM_CHECK(!draftingConfig.has_value() || draftConfigPath.has_value(),
        "drafting configuration was provided but no draftConfigPath was set. "
        "SpecDecode drafting requires a draft engine config.");

    // --- Parse base ---
    cfg.base = parseEngineConfig(baseConfigPath, rank, expectedWorldSize);

    ELLM_CHECK(!cfg.base.isDiffusionBackbone || !draftingConfig.has_value(),
        "DiffusionGemma block diffusion engines do not support speculative decoding drafting.");

    // --- Parse draft (if present) ---
    if (draftConfigPath.has_value())
    {
        cfg.draft = parseDraftEngineConfig(*draftConfigPath);
    }

    validateKVPoolMode(cfg);

    if (cfg.base.specDecodeType == SpecDecodeMode::kEAGLE && cfg.draft.has_value())
    {
        validateEagleConfig(cfg.base, *cfg.draft);
    }

    if (isCachedBlockDraftMode(cfg.base.specDecodeType) && cfg.draft.has_value())
    {
        validateCachedDraftTargetLayerIds(
            cfg.base, *cfg.draft, cfg.base.specDecodeType == SpecDecodeMode::kJetSpec ? "JetSpec" : "DFlash");
    }
    if (cfg.base.specDecodeType == SpecDecodeMode::kGemma4MTP && cfg.draft.has_value())
    {
        validateGemma4MTPConfig(cfg.base, *cfg.draft);
    }
    if (cfg.base.specDecodeType == SpecDecodeMode::kDSpark && cfg.draft.has_value())
    {
        validateCachedDraftTargetLayerIds(cfg.base, *cfg.draft, "DSpark");
    }

    // No cross-engine consistency check is needed for speculative tree budgets: the base emits
    // `max_verify_tree_size` (its verification budget); the draft emits
    // `max_draft_tree_size` (its proposal budget). Consumers read each field from the owning side.

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
        ELLM_CHECK(static_cast<int64_t>(draftingConfig->draftingStep) + 1 <= std::numeric_limits<int32_t>::max(),
            "drafting.draftingStep is too large to represent the accepted-token depth draftingStep+1.");
        bool const isCachedBlockDraft = isCachedBlockDraftMode(cfg.base.specDecodeType);
        bool const isLinearCachedBlockDraft = isCachedBlockDraft && draftingConfig->draftingTopK == 1;
        if (!isLinearCachedBlockDraft || draftingConfig->verifySize < 0)
        {
            requirePositiveField(draftingConfig->verifySize, "verifySize");
        }
        ELLM_CHECK(draftingConfig->dflashBlockSize >= 0,
            "drafting.dflashBlockSize=" + std::to_string(draftingConfig->dflashBlockSize)
                + " must be non-negative; use 0 to infer from DFlash/JetSpec engine config.");

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
        specConfig.dsparkSchedulerMode = draftingConfig->dsparkSchedulerMode;
        specConfig.dsparkConfidenceThreshold = draftingConfig->dsparkConfidenceThreshold;
        specConfig.dsparkMinProposalLen = draftingConfig->dsparkMinProposalLen;
        specConfig.dsparkMaxProposalLen = draftingConfig->dsparkMaxProposalLen;

        if (isCachedBlockDraft)
        {
            static constexpr int32_t kDFlashJetSpecDDTreeMaxAcceptedPathLength = 32;
            static constexpr int32_t kDFlashJetSpecHybridMaxBlockSize = 16;
            char const* modeName = cfg.base.specDecodeType == SpecDecodeMode::kJetSpec ? "JetSpec" : "DFlash";
            char const* configName
                = cfg.base.specDecodeType == SpecDecodeMode::kJetSpec ? "jetspec_config" : "dflash_config";

            specConfig.dflashBlockSize = resolveDFlashBlockSize(cfg.base, *cfg.draft, *draftingConfig);
            ELLM_CHECK(specConfig.dflashBlockSize > 0,
                std::string(modeName) + " requires resolved dflashBlockSize > 0. Set dflashBlockSize or export a draft "
                    + configName + ".block_size.");

            ELLM_CHECK(specConfig.draftingStep == 1,
                std::string(modeName)
                    + " supports draftingStep=1 only because one draft forward emits the full block. Use "
                      "dflashBlockSize to control the draft horizon.");
            ELLM_CHECK(specConfig.dflashBlockSize <= specConfig.maxDraftProposalSize,
                std::string(modeName) + " dflashBlockSize=" + std::to_string(specConfig.dflashBlockSize)
                    + " exceeds draft.maxDraftTreeSize=" + std::to_string(specConfig.maxDraftProposalSize)
                    + ". The draft emits one full block per iteration.");

            bool const useBranchingTree = specConfig.draftingTopK > 1;
            if (!useBranchingTree)
            {
                ELLM_CHECK(specConfig.dflashBlockSize >= 2,
                    std::string(modeName)
                        + " linear requires dflashBlockSize >= 2 because the base verify window is [anchor] + "
                          "draft tokens.");
                if (specConfig.verifySize > 0 && specConfig.verifySize != specConfig.dflashBlockSize)
                {
                    LOG_WARNING(
                        "%s linear uses dflashBlockSize as the base verify window; overriding verifySize=%d to "
                        "dflashBlockSize=%d for compatibility.",
                        modeName, specConfig.verifySize, specConfig.dflashBlockSize);
                }
                specConfig.verifySize = specConfig.dflashBlockSize;
            }
            else
            {
                ELLM_CHECK(specConfig.dflashBlockSize >= 2,
                    std::string(modeName)
                        + " branching DDTree requires dflashBlockSize >= 2 because node 0 is the root and the "
                          "remaining draft positions provide child candidates.");
                ELLM_CHECK(specConfig.draftingTopK < specConfig.verifySize,
                    std::string(modeName) + " DDTree candidateTopK=" + std::to_string(specConfig.draftingTopK)
                        + " must be less than verifySize=" + std::to_string(specConfig.verifySize)
                        + " because the root consumes one verification node.");
                ELLM_CHECK(specConfig.draftingTopK <= kernel::kDDTreeMaxCandidateTopK,
                    std::string(modeName) + " DDTree candidateTopK=" + std::to_string(specConfig.draftingTopK)
                        + " exceeds the current DDTree candidateTopK limit of "
                        + std::to_string(kernel::kDDTreeMaxCandidateTopK) + ".");
                ELLM_CHECK(specConfig.draftingTopK <= cfg.draft->outputVocabSize,
                    std::string(modeName) + " draftingTopK=" + std::to_string(specConfig.draftingTopK)
                        + " exceeds draft output vocabulary size=" + std::to_string(cfg.draft->outputVocabSize) + ".");
                ELLM_CHECK(specConfig.verifySize <= kernel::kDDTreeMaxVerifySize,
                    std::string(modeName) + " DDTree verifySize=" + std::to_string(specConfig.verifySize)
                        + " exceeds node budget limit of " + std::to_string(kernel::kDDTreeMaxVerifySize) + ".");
                int32_t const maxAcceptedPathLength = std::min(specConfig.dflashBlockSize, specConfig.verifySize);
                ELLM_CHECK(maxAcceptedPathLength <= kDFlashJetSpecDDTreeMaxAcceptedPathLength,
                    std::string(modeName) + " DDTree max accepted path length=" + std::to_string(maxAcceptedPathLength)
                        + " exceeds indexed commit path limit of "
                        + std::to_string(kDFlashJetSpecDDTreeMaxAcceptedPathLength) + ".");
            }
            bool const hasLinearAttnLayers = (cfg.base.numLinearAttnLayers > 0);
            if (hasLinearAttnLayers)
            {
                ELLM_CHECK(specConfig.dflashBlockSize <= kDFlashJetSpecHybridMaxBlockSize,
                    std::string(modeName) + " dflashBlockSize=" + std::to_string(specConfig.dflashBlockSize)
                        + " exceeds Qwen3.5 GDN/causal-conv intermediate-state depth limit of "
                        + std::to_string(kDFlashJetSpecHybridMaxBlockSize) + ".");
            }
        }
        else
        {
            // For MTP/DSpark tree drafting, draftingTopK is the DDTree candidate fanout
            // applied after drafting, not a draft-input multiplier.
            bool const fanoutTree = (cfg.base.specDecodeType == SpecDecodeMode::kMTP
                                        || cfg.base.specDecodeType == SpecDecodeMode::kDSpark)
                && specConfig.draftingTopK > 1;
            int64_t const requiredDraftInputSize = fanoutTree
                ? static_cast<int64_t>(specConfig.draftingStep)
                : static_cast<int64_t>(specConfig.draftingStep) * static_cast<int64_t>(specConfig.draftingTopK);

            ELLM_CHECK(requiredDraftInputSize <= specConfig.maxDraftProposalSize,
                "drafting.draftingStep=" + std::to_string(specConfig.draftingStep) + " * drafting.draftingTopK="
                    + std::to_string(specConfig.draftingTopK) + " = " + std::to_string(requiredDraftInputSize)
                    + " exceeds draft.maxDraftTreeSize=" + std::to_string(specConfig.maxDraftProposalSize)
                    + ". Drafting configuration exceeds engine proposal size capability.");
            ELLM_CHECK(specConfig.dflashBlockSize == 0,
                "dflashBlockSize can only be set when spec_decode_type=dflash or jetspec.");

            if (cfg.base.specDecodeType == SpecDecodeMode::kMTP)
            {
                // MTP base verification currently reuses EAGLE utility kernels for accept, KV commit,
                // and hidden-state compaction. Those kernels support maxDepth <= 16. Each round
                // accepts at most draftingStep matched proposals plus one bonus token, for both
                // the linear chain and tree drafting, so the same depth bound applies to either mode.
                static constexpr int32_t kMTPMaxAcceptDepthForCurrentEagleUtilityKernels = 16;
                int32_t const maxAcceptDepth = specConfig.draftingStep + 1;
                ELLM_CHECK(maxAcceptDepth <= kMTPMaxAcceptDepthForCurrentEagleUtilityKernels,
                    "MTP max accept depth (draftingStep+1)=" + std::to_string(maxAcceptDepth)
                        + " exceeds the current MTP EAGLE utility kernel max depth of "
                        + std::to_string(kMTPMaxAcceptDepthForCurrentEagleUtilityKernels)
                        + ". Extend eagleUtilKernels before using larger MTP draft steps.");

                bool const useTree = specConfig.draftingTopK > 1;
                if (!useTree)
                {
                    // Linear chain: the verification input covers the root plus the draftingStep
                    // proposed tokens, so verifySize is fully determined by draftingStep (and
                    // happens to equal the max accept depth).
                    ELLM_CHECK(specConfig.verifySize == maxAcceptDepth,
                        "MTP linear-chain speculative decoding (draftingTopK=1) requires "
                        "verifySize=draftingStep+1. Got verifySize="
                            + std::to_string(specConfig.verifySize)
                            + ", draftingStep=" + std::to_string(specConfig.draftingStep)
                            + ", expected verifySize=" + std::to_string(maxAcceptDepth) + ".");
                }
                else
                {
                    // Tree drafting: the chain drafter keeps one full logits row per depth and
                    // ddtreeBuild grows a prefix-closed, score-prioritized tree of verifySize
                    // nodes with candidateFanout=draftingTopK. The limits below come from the
                    // tree-build kernel (ddtreeKernels.h).
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
                            + " exceeds draft output vocabulary size=" + std::to_string(cfg.draft->outputVocabSize)
                            + ".");
                    ELLM_CHECK(specConfig.verifySize <= kMTPTreeMaxVerifySize,
                        "MTP tree verifySize=" + std::to_string(specConfig.verifySize)
                            + " exceeds the tree-build kernel node budget of " + std::to_string(kMTPTreeMaxVerifySize)
                            + ".");
                }
                // Hybrid base models materialize one GDN/causal-conv state checkpoint per
                // verify token; the intermediate-state kernels support at most 16 per pass.
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

        ELLM_CHECK(specConfig.verifySize <= specConfig.maxVerifySize,
            "drafting.verifySize=" + std::to_string(specConfig.verifySize)
                + " exceeds base.maxVerifyTreeSize=" + std::to_string(specConfig.maxVerifySize)
                + ". Verification size exceeds base engine maximum verification size.");

        if (cfg.base.specDecodeType != SpecDecodeMode::kDSpark)
        {
            ELLM_CHECK(specConfig.dsparkSchedulerMode == DSparkSchedulerMode::kOff,
                "DSpark scheduler options are only valid for spec_decode_type=dspark.");
        }

        if (cfg.base.specDecodeType == SpecDecodeMode::kGemma4MTP)
        {
            ELLM_CHECK(specConfig.draftingTopK == 1,
                "Gemma4 MTP currently supports greedy chain drafting only; draftingTopK must be 1.");
            ELLM_CHECK(specConfig.verifySize == specConfig.draftingStep + 1,
                "Gemma4 MTP verifySize must equal draftingStep + 1 to include the root token and all draft tokens.");
        }

        if (cfg.base.specDecodeType == SpecDecodeMode::kDSpark)
        {
            static constexpr int32_t kDSparkMaxVerifySizeForCurrentUtilityKernels = 17;
            // Tree-mode limits come from the shared DDTree build kernel (ddtreeKernels.h).
            static constexpr int32_t kDSparkTreeMaxVerifySize = 128;
            static constexpr int32_t kDSparkTreeMaxCandidateFanout = 8;
            static constexpr int32_t kDSparkTreeMaxAcceptedPathLength = 16;

            ELLM_CHECK(
                specConfig.draftingStep == 1, "DSpark drafts one full block per iteration; draftingStep must be 1.");
            ELLM_CHECK(cfg.draft.has_value(), "DSpark requires a draft engine.");

            bool const useTree = specConfig.draftingTopK > 1;
            // Tree decouples the proposal from the verify window: the draft always
            // emits the full block and verifySize is the tree node budget.
            int32_t const proposalLen = useTree ? cfg.draft->specDraftBlockSize : specConfig.verifySize - 1;

            if (!useTree)
            {
                ELLM_CHECK(proposalLen > 0,
                    "DSpark verifySize must be at least 2 because base verification is [anchor] + draft tokens.");
                ELLM_CHECK(proposalLen <= cfg.draft->specDraftBlockSize,
                    "DSpark proposalLen=" + std::to_string(proposalLen)
                        + " exceeds dspark_config.block_size=" + std::to_string(cfg.draft->specDraftBlockSize) + ".");
                ELLM_CHECK(specConfig.verifySize <= kDSparkMaxVerifySizeForCurrentUtilityKernels,
                    "DSpark verifySize=" + std::to_string(specConfig.verifySize)
                        + " exceeds current DSpark utility kernel max depth of "
                        + std::to_string(kDSparkMaxVerifySizeForCurrentUtilityKernels) + ".");
            }
            else
            {
                ELLM_CHECK(specConfig.draftingTopK < specConfig.verifySize,
                    "DSpark DDTree candidateTopK=" + std::to_string(specConfig.draftingTopK)
                        + " must be less than verifySize=" + std::to_string(specConfig.verifySize)
                        + " because the root consumes one verification node.");
                ELLM_CHECK(specConfig.draftingTopK <= kDSparkTreeMaxCandidateFanout,
                    "DSpark DDTree candidateTopK=" + std::to_string(specConfig.draftingTopK)
                        + " exceeds the current DDTree candidateTopK limit of "
                        + std::to_string(kDSparkTreeMaxCandidateFanout) + ".");
                ELLM_CHECK(specConfig.draftingTopK <= cfg.draft->outputVocabSize,
                    "DSpark DDTree candidateTopK=" + std::to_string(specConfig.draftingTopK)
                        + " exceeds draft output vocabulary size=" + std::to_string(cfg.draft->outputVocabSize) + ".");
                ELLM_CHECK(specConfig.verifySize <= kDSparkTreeMaxVerifySize,
                    "DSpark DDTree verifySize=" + std::to_string(specConfig.verifySize)
                        + " exceeds node budget limit of " + std::to_string(kDSparkTreeMaxVerifySize) + ".");
                int32_t const maxAcceptedPathLength = std::min(proposalLen + 1, specConfig.verifySize);
                ELLM_CHECK(maxAcceptedPathLength <= kDSparkTreeMaxAcceptedPathLength,
                    "DSpark DDTree max accepted path length=" + std::to_string(maxAcceptedPathLength)
                        + " exceeds indexed commit path limit of " + std::to_string(kDSparkTreeMaxAcceptedPathLength)
                        + ".");
                // threshold enables confidence-guided growth; SPS schedules verify
                // cost, which a fixed node budget cannot trade.
                ELLM_CHECK(specConfig.dsparkSchedulerMode != DSparkSchedulerMode::kSPS,
                    "DSpark DDTree has a fixed verify budget; SPS does not apply. Use threshold to enable "
                    "confidence-guided growth, or off.");
                ELLM_CHECK(specConfig.dsparkConfidenceThreshold < 1.0F,
                    "DSpark DDTree survival threshold must be in [0, 1): the root always survives, so a "
                    "floor of 1 would forbid all growth.");
            }
            ELLM_CHECK(proposalLen <= specConfig.maxDraftProposalSize,
                "DSpark proposalLen=" + std::to_string(proposalLen)
                    + " exceeds draft.maxDraftTreeSize=" + std::to_string(specConfig.maxDraftProposalSize)
                    + ". The draft engine profile must cover the drafted block.");

            if (specConfig.dsparkSchedulerMode != DSparkSchedulerMode::kOff)
            {
                ELLM_CHECK(cfg.draft->dsparkEnableConfidenceHead,
                    "DSpark scheduler requires draft dspark_config.enable_confidence_head=true.");
                ELLM_CHECK(specConfig.dsparkConfidenceThreshold >= 0.0F && specConfig.dsparkConfidenceThreshold <= 1.0F,
                    "DSpark confidence threshold must be in [0, 1].");
                ELLM_CHECK(specConfig.dsparkMinProposalLen >= 1, "DSpark min proposal length must be >= 1.");
                if (specConfig.dsparkMaxProposalLen <= 0)
                {
                    specConfig.dsparkMaxProposalLen = proposalLen;
                }
                ELLM_CHECK(specConfig.dsparkMaxProposalLen <= proposalLen,
                    "DSpark max proposal length=" + std::to_string(specConfig.dsparkMaxProposalLen)
                        + " exceeds proposalLen=" + std::to_string(proposalLen) + ".");
                ELLM_CHECK(specConfig.dsparkMinProposalLen <= specConfig.dsparkMaxProposalLen,
                    "DSpark min proposal length=" + std::to_string(specConfig.dsparkMinProposalLen)
                        + " exceeds max proposal length=" + std::to_string(specConfig.dsparkMaxProposalLen) + ".");
            }
        }

        cfg.specConfig = specConfig;
    }

    return cfg;
}

} // namespace rt
} // namespace trt_edgellm
