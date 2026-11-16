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

#include "sampler/sampling.h"

#include <gtest/gtest.h>

using namespace trt_edgellm::rt;
using namespace trt_edgellm::rt::scheduler;

namespace
{

//! Two requests that agree on everything, so each test can change exactly one field.
LLMGenerationRequest baseline()
{
    LLMGenerationRequest request;
    request.requests.resize(1);
    request.temperature = 0.7F;
    request.topP = 0.9F;
    request.topK = 40;
    request.maxGenerateLength = 128;
    request.loraWeightsName = "adapter-a";
    request.numLogprobs = 0;
    return request;
}

} // namespace

TEST(BatchCompatibilityTests, IdenticalRequestsMayShareAStep)
{
    EXPECT_TRUE(BatchCompatibility::compatible(baseline(), baseline()));
    EXPECT_TRUE(BatchCompatibility::firstDifference(baseline(), baseline()).empty());
}

TEST(BatchCompatibilityTests, EveryComparedFieldBlocksBatching)
{
    auto const firstDifference = [](auto mutate) {
        LLMGenerationRequest other = baseline();
        mutate(other);
        EXPECT_FALSE(BatchCompatibility::compatible(baseline(), other));
        return BatchCompatibility::firstDifference(baseline(), other);
    };

    // The draft proposal path is chosen once per speculative step.
    EXPECT_EQ(firstDifference([](auto& r) { r.proposalSampling = SpecProposalSampling::kGreedy; }), "proposalSampling");

    // Engine-wide settings select code paths and buffer shapes shared by every slot.
    EXPECT_EQ(firstDifference([](auto& r) { r.loraWeightsName = "adapter-b"; }), "loraWeightsName");
    EXPECT_EQ(firstDifference([](auto& r) { r.disableSpecDecode = true; }), "disableSpecDecode");
    EXPECT_EQ(firstDifference([](auto& r) { r.skipSpecialTokens = false; }), "skipSpecialTokens");

    // Context-cache behaviour travels with the coordinator request, so it is batch-wide too.
    EXPECT_EQ(firstDifference([](auto& r) { r.contextCacheLookupPolicy = ContextCacheLookupPolicy::kBypass; }),
        "contextCacheLookupPolicy");
    EXPECT_EQ(firstDifference([](auto& r) { r.contextCacheReplayTailLength = 8; }), "contextCacheReplayTailLength");
    EXPECT_EQ(
        firstDifference([](auto& r) { r.contextCacheCommitPolicy = ContextCacheCommitPolicy::kPrefillStateOnly; }),
        "contextCacheCommitPolicy");
    EXPECT_EQ(firstDifference([](auto& r) { r.enableThinking = !r.enableThinking; }), "enableThinking");
    EXPECT_EQ(firstDifference([](auto& r) { r.generateAudio = true; }), "generateAudio");
    EXPECT_EQ(firstDifference([](auto& r) { r.acceptHiddenLayer = 3; }), "acceptHiddenLayer");
    EXPECT_EQ(firstDifference([](auto& r) { r.diffusionMaxDenoisingSteps = 4; }), "diffusionMaxDenoisingSteps");
    EXPECT_EQ(firstDifference([](auto& r) { r.recurrentCaptureInterval = 16; }), "recurrentCaptureInterval");
    EXPECT_EQ(
        firstDifference([](auto& r) { r.onTokenGenerated = [](TokenCallbackInfo const&) {}; }), "onTokenGenerated");
}

TEST(BatchCompatibilityTests, PerSequenceSettingsDoNotBlockBatching)
{
    // The runtime applies these row by row: each sequence samples with its own parameters and
    // seed, stops at its own budget (or runs through EOS to it), and keeps its own logprobs width.
    auto const compatibleWith = [](auto mutate) {
        LLMGenerationRequest other = baseline();
        mutate(other);
        EXPECT_TRUE(BatchCompatibility::compatible(baseline(), other));
        EXPECT_TRUE(BatchCompatibility::firstDifference(baseline(), other).empty());
    };
    compatibleWith([](auto& r) { r.temperature = 0.1F; });
    compatibleWith([](auto& r) { r.topP = 0.5F; });
    compatibleWith([](auto& r) { r.topK = 1; });
    compatibleWith([](auto& r) { r.samplingSeed = 42; });
    compatibleWith([](auto& r) { r.requests.front().samplingSeed = 42; });
    compatibleWith([](auto& r) { r.numLogprobs = 5; });
    compatibleWith([](auto& r) { r.maxGenerateLength = 256; });
    compatibleWith([](auto& r) { r.ignoreEos = true; });
}

TEST(BatchCompatibilityTests, SamplingAgreesComparesTheRowParameters)
{
    // Asked only of requests on a decoder that takes one set of sampling parameters per batch.
    auto const agreesWith = [](auto mutate) {
        LLMGenerationRequest other = baseline();
        mutate(other);
        return BatchCompatibility::samplingAgrees(baseline(), other);
    };
    EXPECT_TRUE(agreesWith([](auto&) {}));
    EXPECT_FALSE(agreesWith([](auto& r) { r.temperature = 0.1F; }));
    EXPECT_FALSE(agreesWith([](auto& r) { r.topP = 0.5F; }));
    EXPECT_FALSE(agreesWith([](auto& r) { r.topK = 1; }));
    // Settings the decoder applies per row, or that are not sampling, do not count.
    EXPECT_TRUE(agreesWith([](auto& r) { r.samplingSeed = 42; }));
    EXPECT_TRUE(agreesWith([](auto& r) { r.numLogprobs = 5; }));
    EXPECT_TRUE(agreesWith([](auto& r) { r.maxGenerateLength = 256; }));

    // Top-k as the runtime stores it: a negative value disables it like zero, and values above the sampler's cap
    // are clamped to it.
    LLMGenerationRequest disabled = baseline();
    disabled.topK = 0;
    LLMGenerationRequest negative = baseline();
    negative.topK = -1;
    EXPECT_TRUE(BatchCompatibility::samplingAgrees(disabled, negative));
    LLMGenerationRequest capped = baseline();
    capped.topK = trt_edgellm::kMaxSamplingTopK;
    LLMGenerationRequest above = baseline();
    above.topK = trt_edgellm::kMaxSamplingTopK + 1;
    EXPECT_TRUE(BatchCompatibility::samplingAgrees(capped, above));
}

TEST(BatchCompatibilityTests, PerSequencePayloadDoesNotBlockBatching)
{
    // The whole point of batching is that the prompts differ. If any of these started blocking a
    // batch, concurrency would silently collapse to one request at a time.
    LLMGenerationRequest other = baseline();
    other.requests.resize(1);
    Message::MessageContent content;
    content.type = "text";
    content.content = "a completely different prompt";
    Message message;
    message.contents.push_back(content);
    other.requests[0].messages.push_back(message);
    EXPECT_TRUE(BatchCompatibility::compatible(baseline(), other));

    other = baseline();
    other.preTokenizedInputIds = {{1, 2, 3}};
    EXPECT_TRUE(BatchCompatibility::compatible(baseline(), other));

    other = baseline();
    other.streamChannels.resize(1);
    EXPECT_TRUE(BatchCompatibility::compatible(baseline(), other));

    // Per-item fields are per sequence by construction.
    other = baseline();
    other.requests.front().stopStrings = {"STOP"};
    EXPECT_TRUE(BatchCompatibility::compatible(baseline(), other));

    other = baseline();
    other.requests.front().logitBias = {{7, 1.5F}};
    EXPECT_TRUE(BatchCompatibility::compatible(baseline(), other));

    other = baseline();
    other.requests.front().contextCacheLookupPolicy = ContextCacheLookupPolicy::kBypass;
    EXPECT_TRUE(BatchCompatibility::compatible(baseline(), other));

    other = baseline();
    other.requests.front().guidedDecoding = GuidedDecodingParams{};
    EXPECT_TRUE(BatchCompatibility::compatible(baseline(), other));

    // Given a resident request without a grammar constraint, when a candidate carries a
    // guided-decoding constraint with jump-forward on, then the two may share a step.
    other = baseline();
    other.requests.front().guidedDecoding = GuidedDecodingParams{GuideType::kRegex, "[a-z]+", /*jumpForward=*/true};
    EXPECT_TRUE(BatchCompatibility::compatible(baseline(), other));
}

TEST(BatchCompatibilityTests, PreprocessingChoicesDoNotBlockBatching)
{
    // Chat templating and the generation prompt are applied before a request is admitted, so by
    // the time two requests share a step they cannot affect it.
    LLMGenerationRequest other = baseline();
    other.applyChatTemplate = false;
    EXPECT_TRUE(BatchCompatibility::compatible(baseline(), other));

    other = baseline();
    other.addGenerationPrompt = false;
    EXPECT_TRUE(BatchCompatibility::compatible(baseline(), other));

    // The tool and reasoning fields are consumed by the chat template alone.
    other = baseline();
    other.reasoningEffort = "high";
    EXPECT_TRUE(BatchCompatibility::compatible(baseline(), other));

    other = baseline();
    ToolDefinition tool;
    tool.name = "lookup";
    other.tools.push_back(tool);
    EXPECT_TRUE(BatchCompatibility::compatible(baseline(), other));

    other = baseline();
    other.toolChoice.mode = ToolChoice::Mode::kRequired;
    EXPECT_TRUE(BatchCompatibility::compatible(baseline(), other));

    other = baseline();
    other.parallelToolCalls = false;
    EXPECT_TRUE(BatchCompatibility::compatible(baseline(), other));
}

TEST(BatchCompatibilityTests, SchedulingFieldsDoNotBlockBatching)
{
    // Given two requests that differ only in their scheduling priority, or only in their generate budget
    // When the engine asks whether they may share a step
    // Then they may: priority orders admission and never reaches a step, and each resident stops at its own
    //      budget, so neither drives the other's stopping point
    LLMGenerationRequest priority = baseline();
    priority.priority = -5;
    LLMGenerationRequest budget = baseline();
    budget.maxGenerateLength = 256;
    for (LLMGenerationRequest const* other : {&priority, &budget})
    {
        EXPECT_TRUE(BatchCompatibility::compatible(baseline(), *other));
        EXPECT_TRUE(BatchCompatibility::firstDifference(baseline(), *other).empty());
    }
}

TEST(BatchCompatibilityTests, ComparisonIsSymmetricAndReflexive)
{
    LLMGenerationRequest a = baseline();
    LLMGenerationRequest b = baseline();
    b.loraWeightsName = "adapter-b";

    EXPECT_TRUE(BatchCompatibility::compatible(a, a));
    EXPECT_EQ(BatchCompatibility::compatible(a, b), BatchCompatibility::compatible(b, a))
        << "an asymmetric rule would make admission depend on arrival order";
}
