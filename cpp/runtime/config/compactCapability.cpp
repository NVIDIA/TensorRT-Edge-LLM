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

#include "runtime/config/compactCapability.h"

#include "kernels/compactPhase/compactPhaseSupport.h"

#include <array>
#include <cstddef>
#include <string>
#include <utility>

namespace trt_edgellm
{
namespace rt
{

namespace
{

//! SMs a gap reason may name as supporting the refused shape.
constexpr std::array<int32_t, 11> kREPORTED_SMS{80, 86, 87, 89, 90, 100, 101, 103, 110, 120, 121};

//! "@p shape: why[; supported on SMa/SMb]", or empty when @p support(sm) holds.
template <typename Support>
std::string gapOf(std::string const& shape, Support const& support, int32_t sm)
{
    CompactPhaseSupport const here = support(sm);
    if (here.supported)
    {
        return {};
    }
    std::string sms;
    for (int32_t const other : kREPORTED_SMS)
    {
        if (support(other).supported)
        {
            sms += (sms.empty() ? "SM" : "/SM") + std::to_string(other);
        }
    }
    return shape + ": " + here.reason + (sms.empty() ? std::string{} : "; supported on " + sms);
}

AttentionLayerTraits attentionTraits(LLMEngineConfig const& config, size_t layer, int32_t sm)
{
    KVLayerConfig const& kv = config.kvLayerConfigs[layer];
    bool const boundedCapable
        = config.maxKVCacheCapacity > 0 && kv.kvCacheCapacity > 0 && kv.kvCacheCapacity < config.maxKVCacheCapacity;
    AttentionLayerTraits traits;
    traits.numQHeads = config.numAttentionHeads;
    traits.numKVHeads = kv.numKVHeads;
    traits.headSize = kv.headDim;
    traits.smVersion = sm;
    traits.tpSize = config.tpSize;
    traits.fp8KVCache = config.kvCacheDtype == trt_edgellm::DataType::kFP8;
    traits.slidingWindow = boundedCapable;
    traits.supportsBoundedKVCache = boundedCapable;
    traits.kvShared = layer < config.kvSharingDonors.size() && config.kvSharingDonors[layer] >= 0;
    traits.contextMaskSelector = config.contextMaskSelectorEnabled;
    return traits;
}

std::string attentionShape(size_t layer, AttentionLayerTraits const& traits)
{
    return "layer " + std::to_string(layer) + ": " + attentionLayerShape(traits);
}

//! The first attention layer without a phase-8 path.
std::string attentionGap(LLMEngineConfig const& config, int32_t sm)
{
    if (config.numAttentionHeads <= 0)
    {
        return "the engine config has no num_attention_heads; re-export the model";
    }
    for (size_t layer = 0; layer < config.kvLayerConfigs.size(); ++layer)
    {
        AttentionLayerTraits const traits = attentionTraits(config, layer, sm);
        if (traits.numKVHeads <= 0)
        {
            return "layer " + std::to_string(layer) + " has no KV heads in the engine config";
        }
        std::string gap = gapOf(
            attentionShape(layer, traits),
            [&traits](int32_t candidate) {
                AttentionLayerTraits onCandidate = traits;
                onCandidate.smVersion = candidate;
                return attentionCompactSupport(onCandidate);
            },
            sm);
        if (!gap.empty())
        {
            return gap;
        }
    }
    return {};
}

char const* kindName(CompactLayerKind kind)
{
    switch (kind)
    {
    case CompactLayerKind::kAttention: return "attention";
    case CompactLayerKind::kQsaAttention: return "QSA attention";
    case CompactLayerKind::kGatedDeltaNet: return "gated_delta_net";
    case CompactLayerKind::kMamba2Ssd: return "Mamba2 SSD";
    case CompactLayerKind::kCausalConv1d: return "causal_conv1d";
    case CompactLayerKind::kNgramConv: return "n-gram causal_conv1d";
    }
    return "unknown";
}

} // namespace

std::vector<CompactLayerGap> compactLayerGaps(LLMEngineConfig const& config, int32_t sm)
{
    std::vector<CompactLayerGap> gaps;
    auto const add = [&gaps](CompactLayerKind kind, std::string reason) {
        if (!reason.empty())
        {
            gaps.push_back({kind, std::move(reason)});
        }
    };

    if (!config.kvLayerConfigs.empty() && config.indexerHeadDim > 0)
    {
        // QsaAttentionPlugin decodes its ragged metadata with RaggedLayoutSupport::kPaddedOnly, which throws on
        // phase 8 whatever the SM.
        add(CompactLayerKind::kQsaAttention,
            "indexer head dim " + std::to_string(config.indexerHeadDim)
                + ": QSA attention has no compact execution path on any SM");
    }
    else if (!config.kvLayerConfigs.empty())
    {
        add(CompactLayerKind::kAttention, attentionGap(config, sm));
    }

    if (config.numLinearAttnLayers > 0)
    {
        int32_t const headDim = config.recurrentStateHeadDim;
        int32_t const stateSize = config.recurrentStateSize;
        // Only Mamba2 exports B/C groups.
        if (config.recurrentStateNumGroups > 0)
        {
            add(CompactLayerKind::kMamba2Ssd,
                gapOf(
                    "head dim " + std::to_string(headDim) + ", state size " + std::to_string(stateSize),
                    [&](int32_t candidate) {
                        return mamba2SsdCompactSupport(headDim, stateSize, candidate, /*specVerifyState=*/false);
                    },
                    sm));
        }
        else
        {
            add(CompactLayerKind::kGatedDeltaNet,
                gapOf(
                    "K head dim " + std::to_string(headDim) + ", V head dim " + std::to_string(stateSize),
                    [&](int32_t candidate) {
                        return gatedDeltaNetCompactSupport(
                            headDim, stateSize, candidate, /*specVerifyState=*/false, /*diffusionState=*/false);
                    },
                    sm));
        }
        if (config.convDim > 0 && config.convKernel > 0)
        {
            add(CompactLayerKind::kCausalConv1d,
                gapOf(
                    "width " + std::to_string(config.convKernel),
                    [&config](int32_t candidate) {
                        return causalConv1dCompactSupport(candidate, config.convDim, config.convKernel,
                            /*dilation=*/1, /*groups=*/config.convDim, /*specVerifyState=*/false);
                    },
                    sm));
        }
    }

    if (config.ngramEnabled())
    {
        // The n-gram injection is a depthwise causal_conv1d dilated by the n-gram size.
        add(CompactLayerKind::kNgramConv,
            gapOf(
                "width " + std::to_string(config.ngramConvKernel) + ", dilation " + std::to_string(config.ngramSize),
                [&config](int32_t candidate) {
                    return causalConv1dCompactSupport(candidate, config.ngramConvDim, config.ngramConvKernel,
                        config.ngramSize, /*groups=*/config.ngramConvDim, /*specVerifyState=*/false);
                },
                sm));
    }
    return gaps;
}

std::string compactExecutionGap(LLMEngineConfig const& config, int32_t sm)
{
    std::string blocking;
    for (CompactLayerGap const& gap : compactLayerGaps(config, sm))
    {
        blocking += (blocking.empty() ? "" : "; ") + std::string(kindName(gap.kind)) + " layers (" + gap.reason + ")";
    }
    if (blocking.empty())
    {
        return {};
    }
    return "no compact-execution kernels on SM" + std::to_string(sm) + " for " + blocking;
}

} // namespace rt
} // namespace trt_edgellm
