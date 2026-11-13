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

#include "runtime/decoding/dflashDecodeUtils.h"

#include "common/checkMacros.h"
#include "kernels/speculative/ddtreeKernels.h"

#include <string>

namespace trt_edgellm
{
namespace rt
{
namespace dflash_utils
{

char const* proposalAttentionPolicyName(ProposalAttentionPolicy policy) noexcept
{
    switch (policy)
    {
    case ProposalAttentionPolicy::kBidirectional: return "bidirectional";
    case ProposalAttentionPolicy::kCausal: return "causal";
    }
    return "unknown";
}

char const* blockDraftTreePolicyName(BlockDraftTreePolicy policy) noexcept
{
    switch (policy)
    {
    case BlockDraftTreePolicy::kLinear: return "linear";
    case BlockDraftTreePolicy::kDDTree: return "ddtree";
    }
    return "unknown";
}

int32_t runtimeBlockSize(DeploymentConfig const& deployment)
{
    ELLM_CHECK(deployment.specConfig.has_value(), "cached block draft runtime block size requires specConfig.");
    ELLM_CHECK(deployment.specConfig->draftBlockSize > 0,
        "cached block draft runtime block size requires resolved draftBlockSize.");
    return deployment.specConfig->draftBlockSize;
}

bool shouldUseDDTree(DeploymentConfig const& deployment)
{
    if (!isCachedBlockDraftMode(deployment.specDecodeMode()) || !deployment.specConfig.has_value())
    {
        return false;
    }
    return deployment.specConfig->draftTopK > 1;
}

CachedBlockDraftRuntimeConfig makeCachedBlockDraftRuntimeConfig(DeploymentConfig const& deployment)
{
    ELLM_CHECK(deployment.specConfig.has_value(), "cached block draft runtime config requires specConfig.");
    ELLM_CHECK(deployment.draft.has_value(), "cached block draft runtime config requires a draft engine config.");

    SpecDecodeMode const userMode = deployment.specDecodeMode();
    ELLM_CHECK(isCachedBlockDraftMode(userMode),
        "cached block draft runtime config requires spec_decode_type=dflash, jetspec, or domino; got "
            + std::string(specDecodeModeName(userMode)) + ".");
    ELLM_CHECK(deployment.draft->specDecodeType == userMode,
        "cached block draft base/draft modes must match: base=" + std::string(specDecodeModeName(userMode))
            + ", draft=" + specDecodeModeName(deployment.draft->specDecodeType) + ".");
    ELLM_CHECK(deployment.base.reducedVocabSize == 0,
        std::string(specDecodeModeName(userMode)) + " does not support reduced-vocabulary base engines.");

    auto const& spec = *deployment.specConfig;
    CachedBlockDraftRuntimeConfig cfg;
    cfg.userMode = userMode;
    cfg.proposalAttention = userMode == SpecDecodeMode::kJetSpec ? ProposalAttentionPolicy::kCausal
                                                                 : ProposalAttentionPolicy::kBidirectional;
    cfg.treePolicy = spec.draftTopK > 1 ? BlockDraftTreePolicy::kDDTree : BlockDraftTreePolicy::kLinear;
    cfg.blockSize = runtimeBlockSize(deployment);
    cfg.verifyTokenCount = spec.verifyTokenCount;
    bool const isDFlash2 = userMode == SpecDecodeMode::kDFlash && deployment.base.dflashVersion == DFlashVersion::kV2;
    // DFlash V1/JetSpec/Domino DDTree reads logits rows [anchor, proposals...] of the full block; a DFlash2 draft
    // emits only the blockSize - 1 proposal rows.
    cfg.proposalLen = cfg.treePolicy == BlockDraftTreePolicy::kDDTree && !isDFlash2 ? cfg.blockSize : cfg.blockSize - 1;
    cfg.candidateTopK = spec.draftTopK;
    cfg.maskTokenId = deployment.draft->specDraftMaskTokenId > 0 ? deployment.draft->specDraftMaskTokenId
                                                                 : deployment.base.specDraftMaskTokenId;
    cfg.draftHiddenSize = spec.draftHiddenSize;
    cfg.baseOutputHiddenDim = spec.baseOutputHiddenDim;
    cfg.draftVocabSize = deployment.draft->outputVocabSize;
    if (userMode == SpecDecodeMode::kDomino)
    {
        cfg.correctionHead = BlockDraftCorrectionHead::kDomino;
        cfg.dominoGruHiddenDim = deployment.draft->dominoGruHiddenDim;
        cfg.dominoProjDim = deployment.draft->dominoProjDim;
        cfg.dominoPoolSize = deployment.draft->dominoPoolSize;
    }

    ELLM_CHECK(spec.draftSteps == 1,
        std::string(specDecodeModeName(userMode))
            + " cached block draft supports draftSteps=1 only; one draft forward emits the full block.");
    ELLM_CHECK(cfg.candidateTopK >= 1, "cached block draft runtime config requires candidateTopK >= 1.");
    if (cfg.treePolicy == BlockDraftTreePolicy::kLinear)
    {
        ELLM_CHECK(cfg.verifyTokenCount == cfg.blockSize,
            "cached block draft linear verifyTokenCount must equal blockSize: verifyTokenCount="
                + std::to_string(cfg.verifyTokenCount) + ", blockSize=" + std::to_string(cfg.blockSize) + ".");
        ELLM_CHECK(cfg.proposalLen > 0,
            "cached block draft linear requires verifyTokenCount >= 2 because node 0 is the root.");
    }
    else
    {
        int32_t const maxCandidateTopK
            = isDFlash2 ? deployment.draft->specSelectorTopK : kernel::kDDTreeMaxCandidateTopK;
        ELLM_CHECK(cfg.candidateTopK <= maxCandidateTopK,
            "cached block draft DDTree candidateTopK=" + std::to_string(cfg.candidateTopK) + " exceeds the limit of "
                + std::to_string(maxCandidateTopK) + ".");
    }
    if (userMode == SpecDecodeMode::kJetSpec)
    {
        ELLM_CHECK(deployment.base.specDraftCausalHead && deployment.draft->specDraftCausalHead,
            "JetSpec requires causal proposal attention in both base and draft configs.");
    }
    if (userMode == SpecDecodeMode::kDomino)
    {
        ELLM_CHECK(cfg.dominoGruHiddenDim > 0 && cfg.dominoProjDim > 0 && cfg.dominoPoolSize > 0,
            "Domino requires positive gru_hidden_dim, proj_dim, and pool_size in the draft engine config.");
        ELLM_CHECK(cfg.dominoPoolSize <= cfg.draftVocabSize,
            "Domino pool_size (" + std::to_string(cfg.dominoPoolSize) + ") exceeds the draft vocabulary ("
                + std::to_string(cfg.draftVocabSize) + ").");
        ELLM_CHECK(
            deployment.draft->reducedVocabSize == 0, "Domino does not support reduced-vocabulary draft engines.");
    }
    ELLM_CHECK(cfg.maskTokenId >= 0 && cfg.maskTokenId < deployment.base.vocabSize,
        std::string(specDecodeModeName(userMode)) + " cached block draft mask_token_id ("
            + std::to_string(cfg.maskTokenId) + ") is outside base embedding vocab range [0, "
            + std::to_string(deployment.base.vocabSize) + ").");

    return cfg;
}

} // namespace dflash_utils
} // namespace rt
} // namespace trt_edgellm
