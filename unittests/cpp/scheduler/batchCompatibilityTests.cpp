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

    // Sampling parameters are applied to the batch as a whole. Batching a mismatch would not fail
    // loudly -- one request would simply generate with the other's settings.
    EXPECT_EQ(firstDifference([](auto& r) { r.temperature = 0.1F; }), "temperature");
    EXPECT_EQ(firstDifference([](auto& r) { r.topP = 0.5F; }), "topP");
    EXPECT_EQ(firstDifference([](auto& r) { r.topK = 1; }), "topK");
    // An explicit seed anywhere flips the batch-wide stable-sampling mode; the values themselves
    // are per-slot and may differ.
    EXPECT_EQ(firstDifference([](auto& r) { r.samplingSeed = 42; }), "hasExplicitSamplingSeed");
    EXPECT_EQ(firstDifference([](auto& r) { r.requests.front().samplingSeed = 42; }), "hasExplicitSamplingSeed");
    EXPECT_EQ(firstDifference([](auto& r) { r.proposalSampling = SpecProposalSampling::kGreedy; }), "proposalSampling");

    // Engine-wide settings select code paths and buffer shapes shared by every slot.
    EXPECT_EQ(firstDifference([](auto& r) { r.loraWeightsName = "adapter-b"; }), "loraWeightsName");
    EXPECT_EQ(firstDifference([](auto& r) { r.disableSpecDecode = true; }), "disableSpecDecode");
    EXPECT_EQ(firstDifference([](auto& r) { r.numLogprobs = 5; }), "numLogprobs");
    EXPECT_EQ(firstDifference([](auto& r) { r.skipSpecialTokens = false; }), "skipSpecialTokens");
    EXPECT_EQ(firstDifference([](auto& r) { r.maxGenerateLength = 256; }), "maxGenerateLength");

    // Context-cache behaviour travels with the coordinator request, so it is batch-wide too.
    EXPECT_EQ(firstDifference([](auto& r) { r.contextCacheLookupPolicy = ContextCacheLookupPolicy::kBypass; }),
        "contextCacheLookupPolicy");
    EXPECT_EQ(firstDifference([](auto& r) { r.contextCacheReplayTailLength = 8; }), "contextCacheReplayTailLength");
    EXPECT_EQ(
        firstDifference([](auto& r) { r.contextCacheCommitPolicy = ContextCacheCommitPolicy::kPrefillStateOnly; }),
        "contextCacheCommitPolicy");
    EXPECT_EQ(firstDifference([](auto& r) { r.saveSystemPromptKVCache = true; }), "saveSystemPromptKVCache");
    EXPECT_EQ(firstDifference([](auto& r) { r.enableThinking = !r.enableThinking; }), "enableThinking");
    EXPECT_EQ(firstDifference([](auto& r) { r.generateAudio = true; }), "generateAudio");
    EXPECT_EQ(firstDifference([](auto& r) { r.acceptHiddenLayer = 3; }), "acceptHiddenLayer");
    EXPECT_EQ(firstDifference([](auto& r) { r.diffusionMaxDenoisingSteps = 4; }), "diffusionMaxDenoisingSteps");
    EXPECT_EQ(firstDifference([](auto& r) { r.recurrentCaptureInterval = 16; }), "recurrentCaptureInterval");
    EXPECT_EQ(
        firstDifference([](auto& r) { r.onTokenGenerated = [](TokenCallbackInfo const&) {}; }), "onTokenGenerated");
}

TEST(BatchCompatibilityTests, DifferentExplicitSeedValuesMayShareAStep)
{
    LLMGenerationRequest a = baseline();
    LLMGenerationRequest b = baseline();
    a.requests.front().samplingSeed = 1;
    b.requests.front().samplingSeed = 2;
    EXPECT_TRUE(BatchCompatibility::compatible(a, b));
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
}

TEST(BatchCompatibilityTests, ComparisonIsSymmetricAndReflexive)
{
    LLMGenerationRequest a = baseline();
    LLMGenerationRequest b = baseline();
    b.temperature = 0.2F;

    EXPECT_TRUE(BatchCompatibility::compatible(a, a));
    EXPECT_EQ(BatchCompatibility::compatible(a, b), BatchCompatibility::compatible(b, a))
        << "an asymmetric rule would make admission depend on arrival order";
}
