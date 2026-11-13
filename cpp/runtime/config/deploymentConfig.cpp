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
#include "common/specDecodeLimits.h"
#include "kernels/gdnKernels/gdnTreeChunkKernels.h"
#include "kernels/speculative/ddtreeKernels.h"
#include "kernels/speculative/speculativeKernels.h"

#include "common/logger.h"
#include "common/trtUtils.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
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

//! A candidate-head draft emits full-vocabulary token IDs, so it has no reduced vocabulary.
void validateEagleCandidateDraft(LLMEngineConfig const& base, LLMEngineConfig const& draft)
{
    if (draft.draftCandidateCount <= 0)
    {
        return;
    }
    ELLM_CHECK(draft.reducedVocabSize == 0,
        "EAGLE candidate-head drafts emit full-vocabulary IDs and do not support a reduced draft vocabulary (draft "
        "reduced_vocab_size="
            + std::to_string(draft.reducedVocabSize) + "). Re-export the draft without --reduced-vocab-dir.");
    ELLM_CHECK(draft.vocabSize == base.vocabSize,
        "EAGLE candidate-head draft vocab size (" + std::to_string(draft.vocabSize) + ") must match base vocab_size ("
            + std::to_string(base.vocabSize) + ").");
}

//! Longest accepted path (draftSteps + 1) of an EAGLE tree. buildSpecVerificationTree sizes its per-row
//! path arrays for this many nodes plus one level of headroom (kMAX_DEPTH = 16 + 1 in speculativeKernels.cu).
constexpr int32_t kEAGLE_TREE_MAX_ACCEPT_DEPTH{16};

//! Node limit of the padded EAGLE draft tree (draftSteps x draftTopK) and of the verify tree, below the shared
//! kMAX_SPEC_VERIFY_TREE_TOKENS: hybrid bases verify through the GDN tree-chunk kernels, and the draft-tree bound keeps
//! the table selectVerifyTreeNodes ranks within its 2048 entries. Candidate drafts on attention-only bases use the same
//! bound.
constexpr int32_t kEAGLE_MAX_TREE_NODES{64};
static_assert(kEAGLE_TREE_MAX_ACCEPT_DEPTH <= kernel::kGDN_TREE_CHUNK_MAX_ACCEPT
        && kEAGLE_MAX_TREE_NODES <= kernel::kGDN_TREE_CHUNK_MAX_NODES
        && kEAGLE_MAX_TREE_NODES <= kMAX_SPEC_VERIFY_TREE_TOKENS,
    "EAGLE tree limits must stay within the GDN tree-chunk limits and the speculative-decoding tree budget.");

//! Tree limits of candidate-head drafts and hybrid bases: the verification path arrays, the verify-node selection
//! table and the GDN tree-chunk kernels.
void validateEagleTreeLimits(SpecDecodeConfig const& spec)
{
    int64_t const draftSteps = spec.draftSteps;
    int64_t const draftTopK = spec.draftTopK;
    ELLM_CHECK(draftSteps + 1 <= kEAGLE_TREE_MAX_ACCEPT_DEPTH,
        "EAGLE draftSteps+1=" + std::to_string(draftSteps + 1) + " exceeds the maximum verification tree depth of "
            + std::to_string(kEAGLE_TREE_MAX_ACCEPT_DEPTH) + ".");
    ELLM_CHECK(draftSteps * draftTopK <= kEAGLE_MAX_TREE_NODES,
        "EAGLE draftSteps*draftTopK=" + std::to_string(draftSteps * draftTopK) + " exceeds the EAGLE tree limit of "
            + std::to_string(kEAGLE_MAX_TREE_NODES) + " nodes.");
    ELLM_CHECK(spec.verifyTokenCount <= kEAGLE_MAX_TREE_NODES,
        "EAGLE verifyTokenCount=" + std::to_string(spec.verifyTokenCount) + " exceeds the EAGLE tree limit of "
            + std::to_string(kEAGLE_MAX_TREE_NODES) + " nodes.");
    int64_t const treeCapacity = 1 + draftTopK + (draftSteps - 1) * draftTopK * draftTopK;
    ELLM_CHECK(spec.verifyTokenCount <= treeCapacity,
        "EAGLE verifyTokenCount=" + std::to_string(spec.verifyTokenCount)
            + " exceeds the draft tree capacity 1 + draftTopK + (draftSteps-1)*draftTopK^2 = "
            + std::to_string(treeCapacity) + ".");
}

//! Longest accepted path of an MTP round: the EAGLE utility kernels MTP runs on hold at most 16 accepted tokens.
constexpr int32_t kMTP_MAX_ACCEPT_DEPTH{16};
static_assert(kMTP_MAX_ACCEPT_DEPTH <= kernel::kSpecMaxAcceptedPathLength,
    "The MTP accepted-path limit must stay within the speculative commit kernels' path limit.");

//! Longest accepted path of a DSpark tree round, the indexed commit limit of the shared DDTree path.
constexpr int32_t kDSPARK_TREE_MAX_ACCEPT_DEPTH{16};

//! Jump-forward limits per mode and drafting geometry, with V = verifyTokenCount and S = draftSteps. A grafting round
//! verifies J = jumpVerifyTokenCount nodes and accepts up to W = jumpMaxPathLength tokens; maxJumpTokens == 0 means
//! the deployment cannot jump, and then J and W keep the plain round's values, as in every mode without a row.
//! Trees widen by the grafted nodes: J = min(maxVerifyTokens, node limit, V + C - 1) with C the longest path the
//! accept kernels and the next draft forward take back, maxJumpTokens = min(J - V, C - 1), W = max(plain path,
//! maxJumpTokens + 1). Chains replace draft nodes with forced tokens: J = W = chain width >= V, maxJumpTokens = J - 1.
//!   - EAGLE: C = min(path limit, maxDraftTokens), the draft re-reading the path in one proposal; 128 nodes and 32
//!     path entries, or the EAGLE tree limits (64, 16) the plain round obeys with a candidate-head draft or a hybrid
//!     base.
//!   - MTP: C = min(16, maxDraftTokens). Tree: C >= S + 1 and W keeps the S + 1 chain depth. Chain: width
//!     min(maxVerifyTokens, C).
//!   - Gemma4 MTP: the assistant drafts from the last accepted token alone, so maxDraftTokens bounds no path; C = 16.
//!     Chain: width min(maxVerifyTokens, 16).
//!   - DFlash V1, JetSpec and Domino DDTree: the next draft forward takes the accepted path as its delta, so
//!     C = min(draftBlockSize, 32 or 16 on a hybrid base). Linear, DFlash V2 too: the drafted block is the window,
//!     width V.
//!   - DSpark: the next draft forward takes the accepted path as its target-hidden delta, whose profile holds P + 1
//!     rows (P = proposalTokenCount); C = min(P + 1, 16). Tree: dense bases only, a hybrid base's tree does not
//!     widen. Chain: the drafted proposal is the window, width V.
void deriveJumpForwardLimits(DeploymentConfig const& cfg, SpecDecodeConfig& spec)
{
    SpecDecodeMode const mode = cfg.base.specDecodeType;
    bool const hybrid = cfg.base.numLinearAttnLayers > 0;
    int32_t const verifyWidth = spec.verifyTokenCount;
    spec.jumpVerifyTokenCount = verifyWidth;
    spec.jumpMaxPathLength = spec.maxPathLength;
    spec.maxJumpTokens = 0;
    int32_t pathLimit = kernel::kSpecMaxAcceptedPathLength;
    // Longest accepted path the draft takes back in one forward.
    int32_t draftPathLimit = spec.maxDraftTokens;
    int32_t const treeNodeLimit = hybrid ? kernel::kGDN_TREE_CHUNK_MAX_NODES : kMAX_SPEC_VERIFY_TREE_TOKENS;

    auto const tree = [&](int32_t nodeLimit, int32_t pathCapacity, int32_t plainPath) {
        int32_t const jumpWidth = std::min({spec.maxVerifyTokens, nodeLimit, verifyWidth + pathCapacity - 1});
        int32_t const maxJumpTokens = std::min(jumpWidth - verifyWidth, pathCapacity - 1);
        if (maxJumpTokens > 0)
        {
            spec.maxJumpTokens = maxJumpTokens;
            spec.jumpVerifyTokenCount = verifyWidth + maxJumpTokens;
            spec.jumpMaxPathLength = std::max(plainPath, maxJumpTokens + 1);
        }
    };
    auto const chain = [&](int32_t width) {
        if (width >= verifyWidth)
        {
            spec.maxJumpTokens = width - 1;
            spec.jumpVerifyTokenCount = width;
            spec.jumpMaxPathLength = width;
        }
    };

    switch (mode)
    {
    case SpecDecodeMode::kEAGLE:
    {
        bool const eagleTreeLimits = cfg.draft->draftCandidateCount > 0 || hybrid;
        pathLimit = eagleTreeLimits ? kEAGLE_TREE_MAX_ACCEPT_DEPTH : kernel::kSpecMaxAcceptedPathLength;
        tree(eagleTreeLimits ? kEAGLE_MAX_TREE_NODES : kMAX_SPEC_VERIFY_TREE_TOKENS,
            std::min(pathLimit, spec.maxDraftTokens), spec.maxPathLength);
        break;
    }
    case SpecDecodeMode::kMTP:
    {
        pathLimit = kMTP_MAX_ACCEPT_DEPTH;
        int32_t const pathCapacity = std::min(kMTP_MAX_ACCEPT_DEPTH, spec.maxDraftTokens);
        if (spec.draftTopK == 1)
        {
            chain(std::min(spec.maxVerifyTokens, pathCapacity));
        }
        else if (pathCapacity >= spec.draftSteps + 1)
        {
            tree(treeNodeLimit, pathCapacity, spec.draftSteps + 1);
        }
        break;
    }
    case SpecDecodeMode::kDFlash:
    case SpecDecodeMode::kJetSpec:
    case SpecDecodeMode::kDomino:
    {
        if (spec.draftTopK == 1)
        {
            pathLimit = verifyWidth;
            chain(verifyWidth);
        }
        else
        {
            pathLimit = hybrid ? kernel::kGDN_TREE_CHUNK_MAX_ACCEPT : kernel::kSpecMaxAcceptedPathLength;
            tree(treeNodeLimit, std::min(spec.draftBlockSize, pathLimit), spec.maxPathLength);
        }
        break;
    }
    case SpecDecodeMode::kGemma4MTP:
    {
        pathLimit = kMTP_MAX_ACCEPT_DEPTH;
        draftPathLimit = pathLimit;
        if (spec.draftTopK == 1)
        {
            chain(std::min(spec.maxVerifyTokens, pathLimit));
        }
        else
        {
            tree(kMAX_SPEC_VERIFY_TREE_TOKENS, pathLimit, spec.maxPathLength);
        }
        break;
    }
    case SpecDecodeMode::kDSpark:
    {
        draftPathLimit = spec.proposalTokenCount + 1;
        if (spec.draftTopK == 1)
        {
            pathLimit = verifyWidth;
            chain(verifyWidth);
            break;
        }
        pathLimit = kDSPARK_TREE_MAX_ACCEPT_DEPTH;
        int32_t const pathCapacity = std::min(draftPathLimit, pathLimit);
        // A plain path over the limit is rejected after this.
        if (!hybrid && spec.maxPathLength <= pathCapacity)
        {
            tree(kMAX_SPEC_VERIFY_TREE_TOKENS, pathCapacity, spec.maxPathLength);
        }
        break;
    }
    case SpecDecodeMode::kNONE: break;
    }
    ELLM_CHECK(spec.maxJumpTokens == 0
            || (spec.jumpVerifyTokenCount >= verifyWidth && spec.jumpVerifyTokenCount <= spec.maxVerifyTokens
                && spec.jumpMaxPathLength <= pathLimit && spec.jumpMaxPathLength <= draftPathLimit),
        std::string(specDecodeModeName(mode)) + " jump-forward limits (jumpVerifyTokenCount="
            + std::to_string(spec.jumpVerifyTokenCount) + ", jumpMaxPathLength="
            + std::to_string(spec.jumpMaxPathLength) + ") exceed the engine or kernel capacity.");
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
            || mode == SpecDecodeMode::kDomino || mode == SpecDecodeMode::kDSpark);
    bool const hybridManagedSpec = deployment.base.numLinearAttnLayers > 0 && deployment.base.numAttentionLayers > 0
        && (mode == SpecDecodeMode::kMTP || mode == SpecDecodeMode::kEAGLE || mode == SpecDecodeMode::kDFlash
            || mode == SpecDecodeMode::kJetSpec || mode == SpecDecodeMode::kDSpark || mode == SpecDecodeMode::kDomino);
    bool const baseSupportsCrossRequestRetention = !deployment.base.kvLayerConfigs.empty()
        && (mode == SpecDecodeMode::kNONE || nonHybridEagle || attentionOnlyManagedSpec || hybridManagedSpec);
    if (!baseSupportsCrossRequestRetention)
    {
        requireMinimumActiveKVPool(deployment.base, "base engine");
    }

    if (deployment.draft.has_value())
    {
        bool const draftSupportsCrossRequestRetention = !deployment.draft->kvLayerConfigs.empty()
            && (nonHybridEagle || attentionOnlyManagedSpec || hybridManagedSpec);
        if (!draftSupportsCrossRequestRetention)
        {
            requireMinimumActiveKVPool(*deployment.draft, "draft engine");
        }
    }
}

void validateMRopeGeometry(DeploymentConfig const& deployment)
{
    if (!deployment.draft.has_value())
    {
        return;
    }
    LLMEngineConfig const& base = deployment.base;
    LLMEngineConfig const& draft = *deployment.draft;
    bool const baseUsesMRope = base.ropeConfig.type == RopeType::kMRope;
    bool const draftUsesMRope = draft.ropeConfig.type == RopeType::kMRope;
    if (!draftUsesMRope)
    {
        return;
    }
    ELLM_CHECK(baseUsesMRope, "SpecDecode MRoPE draft requires the base engine to use MRoPE.");
    ELLM_CHECK(base.rotaryDim == draft.rotaryDim && base.maxKVCacheCapacity == draft.maxKVCacheCapacity
            && base.recurrentPoolRows == draft.recurrentPoolRows
            && base.ropeConfig.mropeSection == draft.ropeConfig.mropeSection
            && base.ropeConfig.rotaryTheta == draft.ropeConfig.rotaryTheta
            && base.ropeConfig.rotaryScale == draft.ropeConfig.rotaryScale
            && base.ropeConfig.partialRotaryFactor == draft.ropeConfig.partialRotaryFactor
            && base.ropeConfig.yarn == draft.ropeConfig.yarn,
        "SpecDecode base/draft MRoPE geometry is incompatible; rotary dimension, resident rows, KV capacity, "
        "frequency partition, theta, scale, partial rotary factor, and YaRN scaling must match.");
}

void validateDraftRopeBindings(DeploymentConfig const& deployment)
{
    if (!deployment.draft.has_value() || !deployment.draft->useDualRope)
    {
        return;
    }
    SpecDecodeMode const mode = deployment.base.specDecodeType;
    ELLM_CHECK(!isCachedBlockDraftMode(mode) && mode != SpecDecodeMode::kDSpark,
        std::string(specDecodeModeName(mode)) + " draft engines require a single RoPE binding.");
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

int32_t resolveDFlashBlockSize(LLMEngineConfig const& base, LLMEngineConfig const& draft)
{
    if (draft.specDraftBlockSize > 0)
    {
        return draft.specDraftBlockSize;
    }
    return base.specDraftBlockSize;
}

} // namespace

char const* dsparkW2BackendName(DSparkW2Backend backend)
{
    switch (backend)
    {
    case DSparkW2Backend::kAuto: return "auto";
    case DSparkW2Backend::kFp16: return "fp16";
    case DSparkW2Backend::kFp8: return "fp8";
    case DSparkW2Backend::kW4A16: return "w4a16";
    }
    throw std::invalid_argument("Invalid DSpark W2 backend");
}

DSparkW2Backend parseDSparkW2Backend(std::string const& name)
{
    for (auto backend :
        {DSparkW2Backend::kAuto, DSparkW2Backend::kFp16, DSparkW2Backend::kFp8, DSparkW2Backend::kW4A16})
    {
        if (name == dsparkW2BackendName(backend))
            return backend;
    }
    throw std::invalid_argument("DSpark W2 backend must be auto, fp16, fp8, or w4a16");
}

DSparkW2Backend resolveDSparkW2Backend(DSparkW2Backend requested, bool hasPackedW2, bool supportsW4A16)
{
    switch (requested)
    {
    case DSparkW2Backend::kAuto: return hasPackedW2 && supportsW4A16 ? DSparkW2Backend::kW4A16 : DSparkW2Backend::kFp16;
    case DSparkW2Backend::kFp16:
    case DSparkW2Backend::kFp8: return requested;
    case DSparkW2Backend::kW4A16:
        ELLM_CHECK(hasPackedW2, "W4A16 requires packed NVFP4 W2; FP16-to-NVFP4 quantization is not supported");
        ELLM_CHECK(supportsW4A16, "Requested W4A16 W2 backend is unsupported for this device, shape, scale, or AOT");
        return requested;
    }
    throw std::invalid_argument("Invalid DSpark W2 backend");
}

SpecDecodeDraftingConfig resolveSpecDecodeDraftingConfig(
    LLMEngineConfig const& base, LLMEngineConfig const& draft, SpecDecodeDraftingConfig const& requested)
{
    auto const requirePositiveField = [](int32_t value, char const* name) {
        ELLM_CHECK(
            value > 0, std::string("drafting.") + name + "=" + std::to_string(value) + " must be positive (>= 1).");
    };
    bool const isCachedBlockDraft = isCachedBlockDraftMode(base.specDecodeType);
    bool const isDSpark = base.specDecodeType == SpecDecodeMode::kDSpark;
    int32_t const draftTopK = requested.draftTopK.value_or(isCachedBlockDraft || isDSpark ? 1 : 0);
    requirePositiveField(draftTopK, "draftTopK");
    SpecVerifyLayout const verifyLayout = draftTopK == 1 ? SpecVerifyLayout::kChain : SpecVerifyLayout::kTree;
    bool const isBlockDraft = isCachedBlockDraft || isDSpark;

    int32_t draftingStep{0};
    if (isBlockDraft)
    {
        ELLM_CHECK(!requested.draftingStep.has_value() || *requested.draftingStep == 1,
            "Block decoders only support --specDraftStep 1.");
        draftingStep = 1;
    }
    else
    {
        ELLM_CHECK(!requested.proposalTokens.has_value(),
            "--specProposalTokens is not supported for iterative decoders; use --specDraftStep.");
        ELLM_CHECK(requested.draftingStep.has_value(),
            "drafting.draftingStep is required for iterative speculative decoding.");
        draftingStep = *requested.draftingStep;
        requirePositiveField(draftingStep, "draftingStep");
    }

    int32_t proposalTokenCount{0};
    if (isCachedBlockDraft)
    {
        int32_t const checkpointBlockSize = resolveDFlashBlockSize(base, draft);
        ELLM_CHECK(checkpointBlockSize >= 2, "Cached-block speculative decoding requires checkpoint block_size >= 2.");
        int32_t const draftTokenCapacity
            = draft.maxSpecDraftTokens > 0 ? draft.maxSpecDraftTokens : checkpointBlockSize;
        proposalTokenCount = requested.proposalTokens.value_or(std::min(checkpointBlockSize, draftTokenCapacity) - 1);
    }
    else if (isDSpark)
    {
        int32_t const checkpointProposalCount = draft.specDraftBlockSize;
        int32_t const anchorOffset = draft.dsparkSampleFromAnchor ? 0 : 1;
        ELLM_CHECK(checkpointProposalCount > 0, "DSpark requires a positive checkpoint block_size.");
        if (requested.proposalTokens.has_value())
        {
            proposalTokenCount = *requested.proposalTokens;
        }
        else if (verifyLayout == SpecVerifyLayout::kTree)
        {
            proposalTokenCount = checkpointProposalCount;
        }
        else
        {
            proposalTokenCount = std::min({checkpointProposalCount,
                std::max(0, draft.maxSpecDraftTokens - anchorOffset), base.maxSpecVerifyTokens - 1});
        }
    }
    else
    {
        proposalTokenCount = draftingStep;
    }
    requirePositiveField(proposalTokenCount, "proposalTokens");
    ELLM_CHECK(proposalTokenCount < std::numeric_limits<int32_t>::max(),
        "drafting proposal length is too large to add the verification root.");
    int32_t verifyTokenCount{0};
    if (verifyLayout == SpecVerifyLayout::kChain)
    {
        verifyTokenCount = proposalTokenCount + 1;
        ELLM_CHECK(!requested.verifyTokens.has_value() || *requested.verifyTokens == verifyTokenCount,
            isBlockDraft ? "Chain block decoding requires specVerifyTokens=specProposalTokens+1."
                         : "Chain iterative decoding requires specVerifyTokens=specDraftStep+1.");
    }
    else
    {
        verifyTokenCount = requested.verifyTokens.value_or(isBlockDraft ? base.maxSpecVerifyTokens : 0);
        requirePositiveField(verifyTokenCount, "verifyTokens");
    }

    auto resolved = requested;
    resolved.draftTopK = draftTopK;
    resolved.draftingStep = draftingStep;
    resolved.proposalTokens = isBlockDraft ? std::optional<int32_t>{proposalTokenCount} : std::nullopt;
    resolved.verifyTokens = verifyTokenCount;
    if (isDSpark && resolved.dsparkSchedulerMode != DSparkSchedulerMode::kOff && resolved.dsparkMaxProposalLen <= 0)
    {
        resolved.dsparkMaxProposalLen = proposalTokenCount;
    }
    return resolved;
}

std::optional<SpecDecodeDraftingConfig> DeploymentConfig::draftingConfig() const
{
    if (!specConfig)
    {
        return std::nullopt;
    }
    auto const& spec = *specConfig;
    SpecDecodeDraftingConfig result;
    result.draftTopK = spec.draftTopK;
    if (isCachedBlockDraftMode(specDecodeMode()) || specDecodeMode() == SpecDecodeMode::kDSpark)
    {
        result.draftingStep = 1;
        result.proposalTokens = spec.proposalTokenCount;
    }
    else
    {
        result.draftingStep = spec.draftSteps;
    }
    result.verifyTokens = spec.verifyTokenCount;
    result.dsparkSchedulerMode = spec.dsparkSchedulerMode;
    result.dsparkConfidenceThreshold = spec.dsparkConfidenceThreshold;
    result.dsparkMinProposalLen = spec.dsparkMinProposalLen;
    result.dsparkMaxProposalLen = spec.dsparkMaxProposalLen;
    result.dsparkW2Backend = spec.dsparkW2Backend;
    return result;
}

void DeploymentConfig::selectSwaKVCacheMode(bool contextReuseEnabled) noexcept
{
    auto selectMode = [contextReuseEnabled](LLMEngineConfig& config) {
        bool const boundedRuntimeSupported = config.specDecodeType == SpecDecodeMode::kNONE && !contextReuseEnabled;
        bool const boundedSavesMemory
            = boundedRuntimeSupported && config.supportsBoundedSwaKVCache() && config.numSwaPages < config.kvPoolPages;
        config.setSwaKVCacheMode(boundedSavesMemory ? SwaKVCacheMode::kBounded : SwaKVCacheMode::kFull);
    };

    selectMode(base);
    if (draft.has_value())
    {
        selectMode(*draft);
    }
}

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
    return std::max(specConfig->maxDraftTokens, specConfig->verifyTokenCount);
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
    if (specDecodeMode() == SpecDecodeMode::kNONE)
    {
        return 1;
    }
    // A grafting round's path capacity is derived from the plain one and never below it; a config built without the
    // derivation keeps the plain capacity.
    return std::max(specConfig->maxPathLength, specConfig->jumpMaxPathLength);
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

    validateMRopeGeometry(cfg);
    validateDraftRopeBindings(cfg);
    validateKVPoolMode(cfg);

    if (cfg.base.specDecodeType == SpecDecodeMode::kEAGLE && cfg.draft.has_value())
    {
        validateEagleConfig(cfg.base, *cfg.draft);
        validateEagleCandidateDraft(cfg.base, *cfg.draft);
    }

    if (cfg.base.specDecodeType == SpecDecodeMode::kDFlash && cfg.draft.has_value())
    {
        ELLM_CHECK(cfg.draft->specDecodeType == SpecDecodeMode::kDFlash,
            "DFlash base and draft speculative decoding modes must match.");
        ELLM_CHECK(cfg.base.dflashVersion == cfg.draft->dflashVersion,
            "DFlash base and draft dflash_config.version values must match.");
        ELLM_CHECK(cfg.base.dflashProjector == cfg.draft->dflashProjector,
            "DFlash base and draft dflash_config.projector_type values must match.");
    }

    if (isCachedBlockDraftMode(cfg.base.specDecodeType) && cfg.draft.has_value())
    {
        char const* modeName = cfg.base.specDecodeType == SpecDecodeMode::kJetSpec
            ? "JetSpec"
            : (cfg.base.specDecodeType == SpecDecodeMode::kDomino
                      ? "Domino"
                      : (cfg.base.dflashVersion == DFlashVersion::kV2 ? "DFlash V2" : "DFlash"));
        validateCachedDraftTargetLayerIds(cfg.base, *cfg.draft, modeName);
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
    // `max_spec_verify_tokens` (its verification budget); the draft emits
    // `max_spec_draft_tokens` (its proposal budget). Consumers read each field from the owning side.

    // --- Build consolidated SpecDecodeConfig and validate drafting limits ---
    if (draftingConfig.has_value())
    {
        ELLM_CHECK(cfg.base.specDecodeType != SpecDecodeMode::kNONE,
            "drafting configuration was provided but base config is not a speculative decoding base engine.");
        ELLM_CHECK(cfg.draft.has_value() && cfg.draft->specDecodeType == cfg.base.specDecodeType,
            "base and draft speculative decoding modes must match.");
        auto const resolved = resolveSpecDecodeDraftingConfig(cfg.base, *cfg.draft, *draftingConfig);
        bool const isCachedBlockDraft = isCachedBlockDraftMode(cfg.base.specDecodeType);
        bool const isDSpark = cfg.base.specDecodeType == SpecDecodeMode::kDSpark;
        int32_t const draftTopK = *resolved.draftTopK;
        int32_t const draftingStep = *resolved.draftingStep;
        int32_t const proposalTokenCount = isCachedBlockDraft || isDSpark ? *resolved.proposalTokens : draftingStep;
        int32_t const verifyTokenCount = *resolved.verifyTokens;
        SpecVerifyLayout const verifyLayout = draftTopK == 1 ? SpecVerifyLayout::kChain : SpecVerifyLayout::kTree;
        int32_t const draftBlockSize = isCachedBlockDraft ? proposalTokenCount + 1
            : isDSpark ? proposalTokenCount + (cfg.draft->dsparkSampleFromAnchor ? 0 : 1)
                       : 0;

        SpecDecodeConfig specConfig;
        // baseOutputHiddenDim comes from the draft config's `base_model_hidden_size`
        // (= base.hiddenSize * 3 for EAGLE-3, = base.hiddenSize for MTP). Don't
        // derive from base.hiddenSize directly — that's correct only for EAGLE-3.
        specConfig.baseOutputHiddenDim = cfg.draft->baseModelHiddenSize;
        specConfig.draftHiddenSize = cfg.draft->hiddenSize;
        specConfig.maxVerifyTokens = cfg.base.maxSpecVerifyTokens;
        specConfig.maxDraftTokens = cfg.draft->maxSpecDraftTokens;
        specConfig.draftTopK = draftTopK;
        specConfig.draftSteps = draftingStep;
        specConfig.verifyTokenCount = verifyTokenCount;
        specConfig.proposalTokenCount = proposalTokenCount;
        specConfig.maxPathLength = std::min(proposalTokenCount + 1, verifyTokenCount);
        specConfig.verifyLayout = verifyLayout;
        specConfig.draftBlockSize = draftBlockSize;
        specConfig.dsparkSchedulerMode = draftingConfig->dsparkSchedulerMode;
        specConfig.dsparkConfidenceThreshold = draftingConfig->dsparkConfidenceThreshold;
        specConfig.dsparkMinProposalLen = draftingConfig->dsparkMinProposalLen;
        specConfig.dsparkMaxProposalLen = draftingConfig->dsparkMaxProposalLen;
        specConfig.dsparkW2Backend = draftingConfig->dsparkW2Backend;
        (void) dsparkW2BackendName(specConfig.dsparkW2Backend);
        ELLM_CHECK(
            cfg.base.specDecodeType == SpecDecodeMode::kDSpark || specConfig.dsparkW2Backend == DSparkW2Backend::kAuto,
            "dsparkW2Backend applies only to DSpark deployments");

        if (isCachedBlockDraft)
        {
            bool const isDFlash2
                = cfg.base.specDecodeType == SpecDecodeMode::kDFlash && cfg.base.dflashVersion == DFlashVersion::kV2;
            static constexpr int32_t kDFlashJetSpecDDTreeMaxAcceptedPathLength = 32;
            static constexpr int32_t kDFlashJetSpecHybridMaxBlockSize = 16;
            char const* modeName = cfg.base.specDecodeType == SpecDecodeMode::kJetSpec
                ? "JetSpec"
                : (cfg.base.specDecodeType == SpecDecodeMode::kDomino ? "Domino" : (isDFlash2 ? "DFlash2" : "DFlash"));
            char const* configName
                = cfg.base.specDecodeType == SpecDecodeMode::kJetSpec ? "jetspec_config" : "dflash_config";

            if (cfg.base.specDecodeType == SpecDecodeMode::kDomino)
            {
                ELLM_CHECK(
                    cfg.draft->reducedVocabSize == 0, "Domino does not support reduced-vocabulary draft engines.");
                ELLM_CHECK(
                    cfg.base.vocabSize == cfg.draft->vocabSize, "Domino base and draft vocabulary sizes must match.");
                ELLM_CHECK(cfg.base.dominoGruHiddenDim == cfg.draft->dominoGruHiddenDim
                        && cfg.base.dominoProjDim == cfg.draft->dominoProjDim
                        && cfg.base.dominoPoolSize == cfg.draft->dominoPoolSize,
                    "Domino base and draft correction-head contracts must match.");
                ELLM_CHECK(cfg.base.specDraftBlockSize == cfg.draft->specDraftBlockSize,
                    "Domino base and draft must declare the same checkpoint block_size.");
            }

            if (isDFlash2)
            {
                ELLM_CHECK(
                    cfg.draft->reducedVocabSize == 0, "DFlash V2 does not support reduced-vocabulary draft engines.");
                bool const isLiLiCorr = cfg.base.dflashProjector == DFlashProjector::kLiLiCorr;
                if (isLiLiCorr)
                {
                    ELLM_CHECK(specConfig.draftTopK == 1,
                        "DFlash2 lilicorr_candidate_topk is checkpoint-owned and does not use DDTree; draftTopK "
                        "must be 1.");
                }
                else
                {
                    ELLM_CHECK(specConfig.draftTopK <= cfg.draft->specSelectorTopK,
                        "DFlash2 DDTree draftTopK=" + std::to_string(specConfig.draftTopK)
                            + " exceeds the checkpoint selector_top_k=" + std::to_string(cfg.draft->specSelectorTopK)
                            + "; every tree child comes from the selector support.");
                }
                ELLM_CHECK(
                    cfg.base.dflashVersion == DFlashVersion::kV2 && cfg.draft->dflashVersion == DFlashVersion::kV2,
                    "DFlash V2 base and draft must both use dflash_config.version=2.");
                constexpr int32_t kDFlash2ProductionLayers{5};
                ELLM_CHECK(cfg.base.specTargetLayerIds == cfg.draft->specTargetLayerIds,
                    "DFlash2 base and draft target layer IDs must match exactly.");
                ELLM_CHECK(cfg.draft->numDecoderLayers == kDFlash2ProductionLayers,
                    "DFlash2 production draft must contain exactly five decoder layers.");
                ELLM_CHECK(
                    cfg.base.hiddenSize == cfg.draft->hiddenSize, "DFlash2 base and draft hidden sizes must match.");
                int64_t const expectedConditioningSize = static_cast<int64_t>(cfg.base.hiddenSize)
                    * static_cast<int64_t>(cfg.base.specTargetLayerIds.size());
                ELLM_CHECK(static_cast<int64_t>(cfg.draft->baseModelHiddenSize) == expectedConditioningSize,
                    std::string("DFlash2 draft base_model_hidden_size must equal base hidden_size times ")
                        + (isLiLiCorr ? "the target layer count." : "five target layers."));
                ELLM_CHECK(
                    cfg.base.vocabSize == cfg.draft->vocabSize, "DFlash2 base and draft vocabulary sizes must match.");
                ELLM_CHECK(cfg.base.specDraftBlockSize == cfg.draft->specDraftBlockSize,
                    "DFlash2 base and draft must declare the same checkpoint block_size.");
                if (isLiLiCorr)
                {
                    ELLM_CHECK(cfg.base.specLiLiCorrTopK == cfg.draft->specLiLiCorrTopK,
                        "DFlash2 LiLiCorr base and draft lilicorr_candidate_topk values must match.");
                }
                else
                {
                    ELLM_CHECK(cfg.base.specSelectorTopK == cfg.draft->specSelectorTopK
                            && cfg.base.specSelectorRank == cfg.draft->specSelectorRank
                            && cfg.base.specConvKernelSize == cfg.draft->specConvKernelSize
                            && cfg.base.specConvGroupSize == cfg.draft->specConvGroupSize,
                        "DFlash2 base/draft selector and dynamic-convolution contracts must match.");
                }
            }

            int32_t const checkpointBlockSize = resolveDFlashBlockSize(cfg.base, *cfg.draft);
            ELLM_CHECK(specConfig.draftBlockSize > 0,
                std::string(modeName) + " requires resolved draftBlockSize > 0. Export a draft " + configName
                    + ".block_size.");
            if (isDFlash2)
            {
                ELLM_CHECK(specConfig.draftBlockSize == checkpointBlockSize,
                    "DFlash V2 requires specProposalTokens to match checkpoint block_size-1.");
            }
            else
            {
                ELLM_CHECK(specConfig.draftBlockSize <= checkpointBlockSize,
                    std::string(modeName) + " specProposalTokens exceeds checkpoint block_size-1.");
            }

            ELLM_CHECK(specConfig.draftSteps == 1,
                std::string(modeName)
                    + " supports draftSteps=1 only because one draft forward emits the full block. Use "
                      "--specProposalTokens to control the draft horizon.");
            ELLM_CHECK(specConfig.draftBlockSize <= specConfig.maxDraftTokens,
                std::string(modeName) + " draftBlockSize=" + std::to_string(specConfig.draftBlockSize)
                    + " exceeds draft.maxSpecDraftTokens=" + std::to_string(specConfig.maxDraftTokens)
                    + ". The draft emits one full block per iteration.");

            bool const useBranchingTree = specConfig.draftTopK > 1;
            if (!useBranchingTree)
            {
                ELLM_CHECK(specConfig.draftBlockSize >= 2,
                    std::string(modeName)
                        + " linear requires draftBlockSize >= 2 because the base verify window is [anchor] + "
                          "draft tokens.");
                ELLM_CHECK(specConfig.verifyTokenCount == specConfig.draftBlockSize,
                    std::string(modeName) + " chain requires specVerifyTokens=specProposalTokens+1.");
            }
            else
            {
                ELLM_CHECK(specConfig.draftBlockSize >= 2,
                    std::string(modeName)
                        + " branching DDTree requires draftBlockSize >= 2 because node 0 is the root and the "
                          "remaining draft positions provide child candidates.");
                ELLM_CHECK(specConfig.draftTopK < specConfig.verifyTokenCount,
                    std::string(modeName) + " DDTree candidateTopK=" + std::to_string(specConfig.draftTopK)
                        + " must be less than verifyTokenCount=" + std::to_string(specConfig.verifyTokenCount)
                        + " because the root consumes one verification node.");
                // DFlash2 lists children from its selector support (bounded above); the logit-row top-K kernel
                // bounds every other block draft.
                ELLM_CHECK(isDFlash2 || specConfig.draftTopK <= kernel::kDDTreeMaxCandidateTopK,
                    std::string(modeName) + " DDTree candidateTopK=" + std::to_string(specConfig.draftTopK)
                        + " exceeds the current DDTree candidateTopK limit of "
                        + std::to_string(kernel::kDDTreeMaxCandidateTopK) + ".");
                ELLM_CHECK(specConfig.draftTopK <= cfg.draft->outputVocabSize,
                    std::string(modeName) + " draftTopK=" + std::to_string(specConfig.draftTopK)
                        + " exceeds draft output vocabulary size=" + std::to_string(cfg.draft->outputVocabSize) + ".");
                ELLM_CHECK(specConfig.verifyTokenCount <= kernel::kDDTreeMaxVerifySize,
                    std::string(modeName) + " DDTree verifyTokenCount=" + std::to_string(specConfig.verifyTokenCount)
                        + " exceeds node budget limit of " + std::to_string(kernel::kDDTreeMaxVerifySize) + ".");
                int32_t const maxAcceptedPathLength = std::min(specConfig.draftBlockSize, specConfig.verifyTokenCount);
                ELLM_CHECK(maxAcceptedPathLength <= kDFlashJetSpecDDTreeMaxAcceptedPathLength,
                    std::string(modeName) + " DDTree max accepted path length=" + std::to_string(maxAcceptedPathLength)
                        + " exceeds indexed commit path limit of "
                        + std::to_string(kDFlashJetSpecDDTreeMaxAcceptedPathLength) + ".");
            }
            bool const hasLinearAttnLayers = (cfg.base.numLinearAttnLayers > 0);
            if (hasLinearAttnLayers)
            {
                ELLM_CHECK(specConfig.draftBlockSize <= kDFlashJetSpecHybridMaxBlockSize,
                    std::string(modeName) + " draftBlockSize=" + std::to_string(specConfig.draftBlockSize)
                        + " exceeds Qwen3.5 GDN/causal-conv intermediate-state depth limit of "
                        + std::to_string(kDFlashJetSpecHybridMaxBlockSize) + ".");
                if (useBranchingTree)
                {
                    ELLM_CHECK(specConfig.verifyTokenCount <= kernel::kGDN_TREE_CHUNK_MAX_NODES,
                        std::string(modeName)
                            + " hybrid DDTree verifyTokenCount=" + std::to_string(specConfig.verifyTokenCount)
                            + " exceeds the GDN/causal-conv transactional verify node limit of "
                            + std::to_string(kernel::kGDN_TREE_CHUNK_MAX_NODES) + ".");
                }
            }
        }
        else
        {
            // For MTP-family/DSpark tree drafting, draftTopK is the DDTree candidate fanout
            // applied after drafting, not a draft-input multiplier.
            bool const fanoutTree = (cfg.base.specDecodeType == SpecDecodeMode::kMTP
                                        || cfg.base.specDecodeType == SpecDecodeMode::kGemma4MTP
                                        || cfg.base.specDecodeType == SpecDecodeMode::kDSpark)
                && specConfig.draftTopK > 1;
            int64_t const requiredDraftInputSize = fanoutTree
                ? static_cast<int64_t>(specConfig.draftSteps)
                : static_cast<int64_t>(specConfig.draftSteps) * static_cast<int64_t>(specConfig.draftTopK);

            ELLM_CHECK(requiredDraftInputSize <= specConfig.maxDraftTokens,
                "Resolved draft input width=" + std::to_string(requiredDraftInputSize)
                    + " exceeds draft.maxSpecDraftTokens=" + std::to_string(specConfig.maxDraftTokens)
                    + ". Drafting configuration exceeds engine proposal size capability.");
            ELLM_CHECK(isDSpark || specConfig.draftBlockSize == 0,
                "draftBlockSize is only defined for block speculative modes.");

            if (cfg.base.specDecodeType == SpecDecodeMode::kEAGLE)
            {
                int64_t const maxAcceptDepth = static_cast<int64_t>(specConfig.draftSteps) + 1;
                ELLM_CHECK(maxAcceptDepth <= specConfig.verifyTokenCount,
                    "EAGLE verifyTokenCount=" + std::to_string(specConfig.verifyTokenCount)
                        + " is smaller than draftSteps+1=" + std::to_string(maxAcceptDepth) + ".");
                // No EAGLE tree commit has been validated on a base that verifies recurrent state by replay.
                ELLM_CHECK(cfg.base.numLinearAttnLayers == 0 || !cfg.base.recurrentSpecVerifyUsesReplay,
                    "EAGLE does not support bases that verify recurrent state by replay "
                    "(recurrent_spec_verify_mode=replay); run this base without EAGLE speculative decoding.");
                bool const candidateDraft = cfg.draft->draftCandidateCount > 0;
                if (candidateDraft || cfg.base.numLinearAttnLayers > 0)
                {
                    validateEagleTreeLimits(specConfig);
                }
                if (candidateDraft)
                {
                    ELLM_CHECK(specConfig.draftTopK <= cfg.draft->draftCandidateCount,
                        "EAGLE draftTopK=" + std::to_string(specConfig.draftTopK)
                            + " exceeds the draft candidate head's num_candidates="
                            + std::to_string(cfg.draft->draftCandidateCount) + ".");
                }
            }
            else if (cfg.base.specDecodeType == SpecDecodeMode::kMTP
                || cfg.base.specDecodeType == SpecDecodeMode::kGemma4MTP)
            {
                bool const useTree = specConfig.draftTopK > 1;
                int32_t const maxAcceptDepth = specConfig.draftSteps + 1;
                if (cfg.base.specDecodeType == SpecDecodeMode::kMTP || useTree)
                {
                    ELLM_CHECK(maxAcceptDepth <= kMTP_MAX_ACCEPT_DEPTH,
                        "MTP-family tree max accept depth (draftSteps+1)=" + std::to_string(maxAcceptDepth)
                            + " exceeds the current EAGLE utility kernel max depth of "
                            + std::to_string(kMTP_MAX_ACCEPT_DEPTH)
                            + ". Extend speculativeKernels before using larger tree draft steps.");
                }
                if (!useTree)
                {
                    // Linear chain: the verification input covers the root plus the draftSteps
                    // proposed tokens, so verifyTokenCount is fully determined by draftSteps (and
                    // happens to equal the max accept depth).
                    ELLM_CHECK(specConfig.verifyTokenCount == maxAcceptDepth,
                        "MTP linear-chain speculative decoding (draftTopK=1) requires "
                        "verifyTokenCount=draftSteps+1. Got verifyTokenCount="
                            + std::to_string(specConfig.verifyTokenCount)
                            + ", draftSteps=" + std::to_string(specConfig.draftSteps)
                            + ", expected verifyTokenCount=" + std::to_string(maxAcceptDepth) + ".");
                }
                else
                {
                    ELLM_CHECK(cfg.draft->reducedVocabSize == 0,
                        "MTP tree drafting (draftTopK > 1) does not support a reduced draft vocabulary; use "
                        "--specDraftTopK 1 or re-export the draft without --draft-reduced-vocab-dir.");
                    // Tree drafting: the chain drafter keeps one full logits row per depth and
                    // ddtreeBuild grows a prefix-closed, score-prioritized tree of verifyTokenCount
                    // nodes with candidateFanout=draftTopK. The fanout limit comes from the
                    // tree-build kernel (ddtreeKernels.h).
                    static constexpr int32_t kMTPTreeMaxCandidateFanout = 8;
                    ELLM_CHECK(specConfig.draftTopK < specConfig.verifyTokenCount,
                        "MTP tree draftTopK=" + std::to_string(specConfig.draftTopK)
                            + " must be less than verifyTokenCount=" + std::to_string(specConfig.verifyTokenCount)
                            + " because the root consumes one verification node.");
                    ELLM_CHECK(specConfig.draftTopK <= kMTPTreeMaxCandidateFanout,
                        "MTP tree draftTopK=" + std::to_string(specConfig.draftTopK)
                            + " exceeds the tree-build kernel candidate fanout limit of "
                            + std::to_string(kMTPTreeMaxCandidateFanout) + ".");
                    ELLM_CHECK(specConfig.draftTopK <= cfg.draft->outputVocabSize,
                        "MTP tree draftTopK=" + std::to_string(specConfig.draftTopK)
                            + " exceeds draft output vocabulary size=" + std::to_string(cfg.draft->outputVocabSize)
                            + ".");
                    ELLM_CHECK(specConfig.verifyTokenCount <= kMAX_SPEC_VERIFY_TREE_TOKENS,
                        "MTP tree verifyTokenCount=" + std::to_string(specConfig.verifyTokenCount)
                            + " exceeds the tree node budget of " + std::to_string(kMAX_SPEC_VERIFY_TREE_TOKENS) + ".");
                }
                // Hybrid base models materialize one GDN/causal-conv state checkpoint per
                // verify token; the intermediate-state kernels support at most 16 per pass.
                static constexpr int32_t kMTPHybridMaxProposalDepth = 16;
                bool const hasLinearAttnLayers = (cfg.base.numLinearAttnLayers > 0);
                if (hasLinearAttnLayers)
                {
                    ELLM_CHECK(maxAcceptDepth <= kMTPHybridMaxProposalDepth,
                        "MTP max accept depth (draftSteps+1)=" + std::to_string(maxAcceptDepth)
                            + " exceeds Qwen3.5 GDN/causal-conv intermediate-state depth limit of "
                            + std::to_string(kMTPHybridMaxProposalDepth) + ".");
                    if (useTree)
                    {
                        ELLM_CHECK(specConfig.verifyTokenCount <= kernel::kGDN_TREE_CHUNK_MAX_NODES,
                            "MTP hybrid DDTree verifyTokenCount=" + std::to_string(specConfig.verifyTokenCount)
                                + " exceeds the GDN/causal-conv transactional verify node limit of "
                                + std::to_string(kernel::kGDN_TREE_CHUNK_MAX_NODES) + ".");
                    }
                }
            }
        }

        ELLM_CHECK(specConfig.verifyTokenCount <= specConfig.maxVerifyTokens,
            "drafting.verifyTokenCount=" + std::to_string(specConfig.verifyTokenCount)
                + " exceeds base.maxSpecVerifyTokens=" + std::to_string(specConfig.maxVerifyTokens)
                + ". Verification size exceeds base engine maximum verification size.");

        deriveJumpForwardLimits(cfg, specConfig);

        if (cfg.base.specDecodeType != SpecDecodeMode::kDSpark)
        {
            ELLM_CHECK(specConfig.dsparkSchedulerMode == DSparkSchedulerMode::kOff,
                "DSpark scheduler options are only valid for spec_decode_type=dspark.");
        }

        if (cfg.base.specDecodeType == SpecDecodeMode::kDSpark)
        {
            static constexpr int32_t kDSparkMaxVerifySizeForCurrentUtilityKernels = 17;
            // The tree-mode fanout limit comes from the shared DDTree build kernel (ddtreeKernels.h).
            static constexpr int32_t kDSparkTreeMaxCandidateFanout = 8;

            ELLM_CHECK(specConfig.draftSteps == 1,
                "DSpark executes one proposal forward per iteration; draftSteps must be 1.");
            ELLM_CHECK(cfg.draft.has_value(), "DSpark requires a draft engine.");

            bool const useTree = specConfig.draftTopK > 1;
            if (useTree && cfg.base.numLinearAttnLayers > 0 && cfg.base.recurrentSpecVerifyUsesReplay)
            {
                ELLM_CHECK(specConfig.verifyTokenCount == specConfig.maxVerifyTokens,
                    "DSpark tree recurrent-state replay requires verifyTokenCount to match base.maxSpecVerifyTokens "
                    "exactly. "
                    "Build the base engine with maxSpecVerifyTokens="
                        + std::to_string(specConfig.verifyTokenCount) + " for this decoding topology.");
            }
            int32_t const proposalLen = specConfig.proposalTokenCount;
            int32_t const draftInputLen = proposalLen + (cfg.draft->dsparkSampleFromAnchor ? 0 : 1);

            if (!useTree)
            {
                ELLM_CHECK(proposalLen > 0,
                    "DSpark verifyTokenCount must be at least 2 because base verification is [anchor] + draft tokens.");
                ELLM_CHECK(proposalLen <= cfg.draft->specDraftBlockSize,
                    "DSpark proposalLen=" + std::to_string(proposalLen)
                        + " exceeds dspark_config.block_size=" + std::to_string(cfg.draft->specDraftBlockSize) + ".");
                if (cfg.base.numLinearAttnLayers > 0)
                {
                    ELLM_CHECK(specConfig.verifyTokenCount <= kernel::kGDN_TREE_CHUNK_MAX_ACCEPT,
                        "DSpark hybrid chain exceeds the causal-conv accepted-path limit.");
                }
                ELLM_CHECK(specConfig.verifyTokenCount <= kDSparkMaxVerifySizeForCurrentUtilityKernels,
                    "DSpark verifyTokenCount=" + std::to_string(specConfig.verifyTokenCount)
                        + " exceeds current DSpark utility kernel max depth of "
                        + std::to_string(kDSparkMaxVerifySizeForCurrentUtilityKernels) + ".");
            }
            else
            {
                if (cfg.base.numLinearAttnLayers > 0)
                {
                    ELLM_CHECK(specConfig.verifyTokenCount <= kernel::kGDN_TREE_CHUNK_MAX_NODES,
                        "DSpark hybrid tree exceeds the GDN chunk-kernel node limit.");
                }
                ELLM_CHECK(proposalLen == cfg.draft->specDraftBlockSize,
                    "DSpark tree requires specProposalTokens to match the checkpoint block_size.");
                ELLM_CHECK(specConfig.draftTopK < specConfig.verifyTokenCount,
                    "DSpark DDTree candidateTopK=" + std::to_string(specConfig.draftTopK)
                        + " must be less than verifyTokenCount=" + std::to_string(specConfig.verifyTokenCount)
                        + " because the root consumes one verification node.");
                ELLM_CHECK(specConfig.draftTopK <= kDSparkTreeMaxCandidateFanout,
                    "DSpark DDTree candidateTopK=" + std::to_string(specConfig.draftTopK)
                        + " exceeds the current DDTree candidateTopK limit of "
                        + std::to_string(kDSparkTreeMaxCandidateFanout) + ".");
                ELLM_CHECK(specConfig.draftTopK <= cfg.draft->outputVocabSize,
                    "DSpark DDTree candidateTopK=" + std::to_string(specConfig.draftTopK)
                        + " exceeds draft output vocabulary size=" + std::to_string(cfg.draft->outputVocabSize) + ".");
                ELLM_CHECK(specConfig.verifyTokenCount <= kMAX_SPEC_VERIFY_TREE_TOKENS,
                    "DSpark DDTree verifyTokenCount=" + std::to_string(specConfig.verifyTokenCount)
                        + " exceeds the tree node budget of " + std::to_string(kMAX_SPEC_VERIFY_TREE_TOKENS) + ".");
                int32_t const maxAcceptedPathLength = std::min(proposalLen + 1, specConfig.verifyTokenCount);
                ELLM_CHECK(maxAcceptedPathLength <= kDSPARK_TREE_MAX_ACCEPT_DEPTH,
                    "DSpark DDTree max accepted path length=" + std::to_string(maxAcceptedPathLength)
                        + " exceeds indexed commit path limit of " + std::to_string(kDSPARK_TREE_MAX_ACCEPT_DEPTH)
                        + ".");
                ELLM_CHECK(specConfig.dsparkSchedulerMode != DSparkSchedulerMode::kSPS,
                    "DSpark DDTree has a fixed verify budget; SPS does not apply. Use threshold to enable "
                    "confidence-guided growth, or off.");
                ELLM_CHECK(specConfig.dsparkConfidenceThreshold < 1.0F,
                    "DSpark DDTree survival threshold must be in [0, 1): the root always survives, so a "
                    "floor of 1 would forbid all growth.");
            }
            int64_t const maxDraftInputLen = specConfig.maxDraftTokens;
            ELLM_CHECK(draftInputLen <= maxDraftInputLen,
                "DSpark draft input length=" + std::to_string(draftInputLen)
                    + " exceeds the profiled input length=" + std::to_string(maxDraftInputLen)
                    + ". The draft engine profile must cover the executed proposal prefix.");

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
