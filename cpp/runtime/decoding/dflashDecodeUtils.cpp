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
    ELLM_CHECK(deployment.specConfig->dflashBlockSize > 0,
        "cached block draft runtime block size requires resolved dflashBlockSize.");
    return deployment.specConfig->dflashBlockSize;
}

bool shouldUseDDTree(DeploymentConfig const& deployment)
{
    if (!isCachedBlockDraftMode(deployment.specDecodeMode()) || !deployment.specConfig.has_value())
    {
        return false;
    }
    return deployment.specConfig->draftingTopK > 1;
}

CachedBlockDraftRuntimeConfig makeCachedBlockDraftRuntimeConfig(DeploymentConfig const& deployment)
{
    ELLM_CHECK(deployment.specConfig.has_value(), "cached block draft runtime config requires specConfig.");
    ELLM_CHECK(deployment.draft.has_value(), "cached block draft runtime config requires a draft engine config.");

    SpecDecodeMode const userMode = deployment.specDecodeMode();
    ELLM_CHECK(isCachedBlockDraftMode(userMode),
        "cached block draft runtime config requires spec_decode_type=dflash or jetspec; got "
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
    cfg.treePolicy = spec.draftingTopK > 1 ? BlockDraftTreePolicy::kDDTree : BlockDraftTreePolicy::kLinear;
    cfg.blockSize = runtimeBlockSize(deployment);
    cfg.verifySize = spec.verifySize;
    cfg.proposalLen = cfg.treePolicy == BlockDraftTreePolicy::kDDTree ? cfg.blockSize : cfg.verifySize - 1;
    cfg.candidateTopK = spec.draftingTopK;
    cfg.maskTokenId = deployment.draft->specDraftMaskTokenId > 0 ? deployment.draft->specDraftMaskTokenId
                                                                 : deployment.base.specDraftMaskTokenId;
    cfg.draftHiddenSize = spec.draftHiddenSize;
    cfg.baseOutputHiddenDim = spec.baseOutputHiddenDim;
    cfg.draftVocabSize = deployment.draft->outputVocabSize;

    ELLM_CHECK(spec.draftingStep == 1,
        std::string(specDecodeModeName(userMode))
            + " cached block draft supports draftingStep=1 only; one draft forward emits the full block.");
    ELLM_CHECK(cfg.candidateTopK >= 1, "cached block draft runtime config requires candidateTopK >= 1.");
    if (cfg.treePolicy == BlockDraftTreePolicy::kLinear)
    {
        ELLM_CHECK(cfg.verifySize == cfg.blockSize,
            "cached block draft linear verifySize must equal blockSize: verifySize=" + std::to_string(cfg.verifySize)
                + ", blockSize=" + std::to_string(cfg.blockSize) + ".");
        ELLM_CHECK(
            cfg.proposalLen > 0, "cached block draft linear requires verifySize >= 2 because node 0 is the root.");
    }
    else
    {
        ELLM_CHECK(cfg.candidateTopK <= kernel::kDDTreeMaxCandidateTopK,
            "cached block draft DDTree candidateTopK exceeds the current kernel limit.");
    }
    if (userMode == SpecDecodeMode::kJetSpec)
    {
        ELLM_CHECK(deployment.base.specDraftCausalHead && deployment.draft->specDraftCausalHead,
            "JetSpec requires causal proposal attention in both base and draft configs.");
    }
    ELLM_CHECK(cfg.maskTokenId >= 0 && cfg.maskTokenId < cfg.draftVocabSize,
        std::string(specDecodeModeName(userMode)) + " cached block draft mask_token_id ("
            + std::to_string(cfg.maskTokenId) + ") is outside draft vocab range [0, "
            + std::to_string(cfg.draftVocabSize) + ").");

    return cfg;
}

} // namespace dflash_utils
} // namespace rt
} // namespace trt_edgellm
