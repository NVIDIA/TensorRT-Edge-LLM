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

#include "runtime/state/contextCache/contextCacheDeployment.h"

#include "common/checkMacros.h"
#include "common/pagedKvTypes.h"

#include <algorithm>
#include <string>
#include <vector>

namespace trt_edgellm::rt
{
namespace
{

void validateStateContract(LLMEngineConfig const& config, char const* label)
{
    int32_t const attentionCount = static_cast<int32_t>(
        std::count(config.layerTypes.begin(), config.layerTypes.end(), HybridCacheManager::LayerType::kAttention));
    int32_t const recurrentCount = static_cast<int32_t>(config.layerTypes.size()) - attentionCount;
    ELLM_CHECK(static_cast<int32_t>(config.layerTypes.size()) == config.numAttentionLayers + config.numLinearAttnLayers,
        std::string(label) + " layer_types must describe every stateful attention/recurrent layer.");
    ELLM_CHECK(attentionCount == config.numAttentionLayers,
        std::string(label) + " attention layer count does not match layer_types.");
    ELLM_CHECK(recurrentCount == config.numLinearAttnLayers,
        std::string(label) + " recurrent layer count does not match layer_types.");
    ELLM_CHECK(static_cast<int32_t>(config.kvLayerConfigs.size()) == config.numAttentionLayers,
        std::string(label) + " kv_layer_configs must describe every attention layer.");
    ELLM_CHECK(config.kvSharingDonors.empty()
            || static_cast<int32_t>(config.kvSharingDonors.size()) == config.numAttentionLayers,
        std::string(label) + " kv_sharing_donors must be empty or match the attention layer count.");

    if (config.numAttentionLayers > 0)
    {
        // Snapshot byte copies and page-table rebinds preserve both supported KV storage layouts.
        ELLM_CHECK(config.kvCacheDtype == nvinfer1::DataType::kHALF || config.kvCacheDtype == nvinfer1::DataType::kFP8,
            std::string(label) + " uses a KV dtype outside the supported context-reuse boundary (FP16 or FP8 KV).");
        // SWA changes the kernel read mask, not physical retention: context reuse requires a full logical allocation
        // for every attention layer so cached pages remain valid when rebound across requests.
        int64_t const minimumActivePages
            = computeMinimumKvPoolPages(config.maxSupportedBatchSize, config.maxKVCacheCapacity);
        ELLM_CHECK(config.kvPoolPages >= minimumActivePages && config.kvPoolPages <= kMAX_KV_POOL_PAGES,
            std::string(label) + " has invalid context-cache pool geometry.");
    }

    if (!config.kvSharingDonors.empty())
    {
        for (int32_t recipient = 0; recipient < config.numAttentionLayers; ++recipient)
        {
            int32_t const donor = config.kvSharingDonors[recipient];
            if (donor < 0)
            {
                continue;
            }
            ELLM_CHECK(donor < config.numAttentionLayers && donor != recipient,
                std::string(label) + " has an invalid KV donor index.");
            ELLM_CHECK(
                config.kvSharingDonors[donor] == -1, std::string(label) + " KV donor chains/cycles are not supported.");
            ELLM_CHECK(config.kvLayerConfigs[donor].numKVHeads == config.kvLayerConfigs[recipient].numKVHeads
                    && config.kvLayerConfigs[donor].headDim == config.kvLayerConfigs[recipient].headDim,
                std::string(label) + " KV donor and recipient layouts must match.");
        }
    }

    if (config.numLinearAttnLayers > 0)
    {
        ELLM_CHECK(config.recurrentStateNumHeads > 0 && config.recurrentStateHeadDim > 0
                && config.recurrentStateSize > 0 && config.convDim > 0 && config.convKernel > 0,
            std::string(label) + " has an incomplete recurrent-state schema.");
    }
    ELLM_CHECK(!config.useVisionBidirectionalAttention,
        std::string(label) + " uses vision-bidirectional attention, which is not managed by the context cache.");
}

ContextCacheModelStateKind classifyBaseState(LLMEngineConfig const& config)
{
    if (config.numLinearAttnLayers == 0)
    {
        return ContextCacheModelStateKind::kAttentionOnly;
    }
    if (config.numAttentionLayers == 0)
    {
        return ContextCacheModelStateKind::kPureRecurrent;
    }
    return ContextCacheModelStateKind::kHybrid;
}

void validateSpecDeploymentTuple(DeploymentConfig const& deployment)
{
    ELLM_CHECK(deployment.base.isSpecDecodeBase && deployment.draft.has_value() && deployment.specConfig.has_value()
            && deployment.draft->specDecodeType == deployment.base.specDecodeType
            && !deployment.draft->isSpecDecodeBase,
        "Speculative context-cache deployment requires matching base-role/draft-role engines and configuration.");
    validateStateContract(*deployment.draft, "draft engine");
    ELLM_CHECK(deployment.specConfig->verifySize > 0 && deployment.specConfig->draftingStep > 0
            && deployment.specConfig->draftingTopK > 0,
        "Speculative context reuse has invalid execution geometry.");
}

} // namespace

ContextCacheDeploymentProfile validateContextCacheDeployment(DeploymentConfig const& deployment)
{
    ELLM_CHECK(!deployment.base.isDiffusionBackbone,
        "Block Diffusion context reuse is outside the current context-reuse support boundary.");
    validateStateContract(deployment.base, "base engine");

    if (deployment.base.specDecodeType == SpecDecodeMode::kNONE)
    {
        ELLM_CHECK(
            !deployment.base.isSpecDecodeBase && !deployment.draft.has_value() && !deployment.specConfig.has_value(),
            "Vanilla context-cache deployment requires an llm-role base and no draft/speculative configuration.");
        ELLM_CHECK(deployment.base.numAttentionLayers > 0 || deployment.base.numLinearAttnLayers > 0,
            "Context-cache deployment has no stateful layers.");
        return ContextCacheDeploymentProfile{classifyBaseState(deployment.base), std::nullopt};
    }

    validateSpecDeploymentTuple(deployment);
    std::optional<SpecReuseContract> const contract = resolveSpecReuseContract(deployment);
    ELLM_CHECK(contract.has_value(), "Speculative method has no context-reuse contract.");

    if (deployment.base.numLinearAttnLayers > 0)
    {
        ELLM_CHECK(deployment.base.specDecodeType == SpecDecodeMode::kMTP && deployment.base.numAttentionLayers > 0,
            "MTP context reuse requires a hybrid base with both attention and recurrent layers.");
        ELLM_CHECK(deployment.draft->hasOwnKVCache && !deployment.draft->sharesTargetKV,
            "MTP context reuse requires a draft engine with its own independent KV cache.");
        return ContextCacheDeploymentProfile{ContextCacheModelStateKind::kHybrid, *contract};
    }

    ELLM_CHECK(
        deployment.draft->numLinearAttnLayers == 0, "Hybrid speculative context reuse is supported only for MTP.");
    ELLM_CHECK(deployment.base.specDecodeType != SpecDecodeMode::kMTP,
        "MTP context reuse requires a hybrid base with both attention and recurrent layers.");
    ELLM_CHECK(
        deployment.base.numAttentionLayers > 0, "Speculative context reuse requires a page-backed base KV cache.");
    if (contract->ownsPagedSpecState)
    {
        ELLM_CHECK(deployment.draft->numAttentionLayers > 0 && deployment.draft->hasOwnKVCache
                && !deployment.draft->sharesTargetKV,
            "Paged speculative context reuse requires a draft engine with its own independent KV cache.");
    }
    else
    {
        ELLM_CHECK(deployment.draft->sharesTargetKV && !deployment.draft->hasOwnKVCache,
            "Shared-target-KV speculative context reuse requires a read-only assistant with no independent KV cache.");
    }

    return ContextCacheDeploymentProfile{ContextCacheModelStateKind::kAttentionOnly, *contract};
}

} // namespace trt_edgellm::rt
