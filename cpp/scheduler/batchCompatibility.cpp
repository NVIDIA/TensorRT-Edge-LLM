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

#include "scheduler/batchCompatibility.h"

#include <algorithm>

namespace trt_edgellm
{
namespace rt
{
namespace scheduler
{
namespace
{

//! One field two requests must agree on before sharing a step, and how to compare it.
struct ComparedField
{
    char const* name;
    bool (*equal)(LLMGenerationRequest const&, LLMGenerationRequest const&);
};

constexpr ComparedField kComparedFields[] = {
    // Sampling is applied to the batch as a whole, so a mismatch would not fail loudly: one request
    // would simply generate with the other's settings.
    {"temperature", [](auto const& a, auto const& b) { return a.temperature == b.temperature; }},
    {"topP", [](auto const& a, auto const& b) { return a.topP == b.topP; }},
    {"topK", [](auto const& a, auto const& b) { return a.topK == b.topK; }},

    // The seed values themselves live per slot, but an explicit seed anywhere in a request flips
    // the batch-wide stable-sampling mode; a request that asked for determinism must not share a
    // step with one sampled the default way.
    {"hasExplicitSamplingSeed",
        [](auto const& a, auto const& b) {
            auto const hasSeed = [](LLMGenerationRequest const& r) {
                return r.samplingSeed.has_value()
                    || std::any_of(r.requests.begin(), r.requests.end(),
                        [](auto const& item) { return item.samplingSeed.has_value(); });
            };
            return hasSeed(a) == hasSeed(b);
        }},

    // Selects the draft proposal path for the whole speculative step.
    {"proposalSampling", [](auto const& a, auto const& b) { return a.proposalSampling == b.proposalSampling; }},

    // Bound once per step: the engine holds one set of LoRA weights.
    {"loraWeightsName", [](auto const& a, auto const& b) { return a.loraWeightsName == b.loraWeightsName; }},

    // Selects which decoder runs, so it cannot vary within a step.
    {"disableSpecDecode", [](auto const& a, auto const& b) { return a.disableSpecDecode == b.disableSpecDecode; }},

    // Shapes the logits output for the whole batch.
    {"numLogprobs", [](auto const& a, auto const& b) { return a.numLogprobs == b.numLogprobs; }},

    // Final detokenization uses the founding request's policy for every resident.
    {"skipSpecialTokens", [](auto const& a, auto const& b) { return a.skipSpecialTokens == b.skipSpecialTokens; }},

    // The context holds one batch-wide token callback, so a callback-bearing request never shares
    // a batch: the founder's callback would receive the joiner's tokens. Founder-only, like guided
    // decoding -- a callback on either side stalls the head until the batch turns over.
    {"onTokenGenerated",
        [](auto const& a, auto const& b) {
            return !a.onTokenGenerated.has_value() && !b.onTokenGenerated.has_value();
        }},

    // Step budget: mixing these lets the shorter request drive the batch's stopping point.
    {"maxGenerateLength", [](auto const& a, auto const& b) { return a.maxGenerateLength == b.maxGenerateLength; }},

    // Model-level overrides that reconfigure the step rather than one sequence.
    {"diffusionMaxDenoisingSteps",
        [](auto const& a, auto const& b) { return a.diffusionMaxDenoisingSteps == b.diffusionMaxDenoisingSteps; }},
    {"recurrentCaptureInterval",
        [](auto const& a, auto const& b) { return a.recurrentCaptureInterval == b.recurrentCaptureInterval; }},

    // Context-cache behaviour drives shared cache bookkeeping during the step.
    {"contextCacheLookupPolicy",
        [](auto const& a, auto const& b) { return a.contextCacheLookupPolicy == b.contextCacheLookupPolicy; }},
    {"contextCacheCommitPolicy",
        [](auto const& a, auto const& b) { return a.contextCacheCommitPolicy == b.contextCacheCommitPolicy; }},
    {"contextCacheReplayTailLength",
        [](auto const& a, auto const& b) { return a.contextCacheReplayTailLength == b.contextCacheReplayTailLength; }},

    // Writes into the shared system-prompt cache during the step.
    {"saveSystemPromptKVCache",
        [](auto const& a, auto const& b) { return a.saveSystemPromptKVCache == b.saveSystemPromptKVCache; }},

    // Changes what the tokenizer emits, and so what the step is asked to continue.
    {"enableThinking", [](auto const& a, auto const& b) { return a.enableThinking == b.enableThinking; }},

    // Outside the scope of batched execution; compared rather than ignored so that enabling one of
    // them can never silently share a step with a request that has not.
    {"generateAudio", [](auto const& a, auto const& b) { return a.generateAudio == b.generateAudio; }},
    {"acceptHiddenLayer", [](auto const& a, auto const& b) { return a.acceptHiddenLayer == b.acceptHiddenLayer; }},
};

} // namespace

bool BatchCompatibility::compatible(
    LLMGenerationRequest const& resident, LLMGenerationRequest const& candidate) noexcept
{
    for (auto const& field : kComparedFields)
    {
        if (!field.equal(resident, candidate))
        {
            return false;
        }
    }
    return true;
}

std::string BatchCompatibility::firstDifference(
    LLMGenerationRequest const& resident, LLMGenerationRequest const& candidate) noexcept
{
    for (auto const& field : kComparedFields)
    {
        if (!field.equal(resident, candidate))
        {
            return field.name;
        }
    }
    return {};
}

} // namespace scheduler
} // namespace rt
} // namespace trt_edgellm
