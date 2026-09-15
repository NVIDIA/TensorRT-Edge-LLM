/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "runtime/state/sequencePolicy.h"
#include "runtime/state/sequenceChannel.h"
#include <cmath>
#include <gtest/gtest.h>
#include <limits>
using namespace trt_edgellm::rt;
namespace
{
SequenceSample token(int id)
{
    SequenceSample s;
    s.token = id;
    return s;
}
} // namespace
TEST(SequencePolicy, BiasedUnscaledLogprobsAndTopK)
{
    SequenceSampler sampler(4);
    float values[]{0, 1, 2, 3};
    SequenceOptions o;
    o.temperature = 0;
    o.numLogprobs = 4;
    o.logitBias[0] = 5;
    auto s = sampler.sample(values, o, 0);
    EXPECT_EQ(s.token, 0);
    EXPECT_EQ(s.randomCounter, 0U);
    EXPECT_EQ(s.top[0].token, 0);
    EXPECT_NEAR(s.logprob, 5 - std::log(std::exp(5) + std::exp(1) + std::exp(2) + std::exp(3)), 1e-12);
    o.temperature = 0.7;
    o.topK = 2;
    o.topP = 1;
    for (uint64_t i = 0; i < 100; ++i)
    {
        auto sampled = sampler.sample(values, o, i);
        EXPECT_TRUE(sampled.token == 0 || sampled.token == 3);
    }
}
TEST(SequencePolicy, RngIndependentOfOtherRowsAndSeed)
{
    SequenceSampler sampler(4);
    float values[]{0, 0, 0, 0};
    SequenceOptions a, b;
    a.temperature = 0.7;
    a.topK = 4;
    a.seed = 7;
    b = a;
    b.seed = 9;
    bool different = false;
    for (uint64_t counter = 0; counter < 100; ++counter)
    {
        auto reference = sampler.sample(values, a, counter);
        auto other = sampler.sample(values, b, counter);
        auto replay = sampler.sample(values, a, counter);
        EXPECT_EQ(reference.token, replay.token);
        EXPECT_EQ(reference.randomCounter, counter + 1);
        different = different || reference.token != other.token;
    }
    EXPECT_TRUE(different);
    EXPECT_THROW(sampler.sample(values, a, std::numeric_limits<uint64_t>::max()), std::overflow_error);
}
TEST(SequencePolicy, NucleusRetainsMinimalProbabilityPrefix)
{
    SequenceSampler sampler(4);
    float values[]{3, 2, 1, 0};
    SequenceOptions o;
    o.temperature = 0.8;
    o.topK = 4;
    o.topP = 0.1;
    for (uint64_t i = 0; i < 100; ++i)
    {
        EXPECT_EQ(sampler.sample(values, o, i).token, 0);
    }
    o.topP = 1;
    o.topK = 1;
    EXPECT_EQ(sampler.sample(values, o, 0).randomCounter, 0U);
}
TEST(SequencePolicy, StopAcrossTokensDoesNotLeakPrefix)
{
    SequencePolicy policy;
    SequenceOptions o;
    o.stopStrings = {"STOP", "TOP"};
    o.maxOutputTokens = 10;
    policy.reset(o, 16);
    policy.accept(token(1), "hello S");
    std::string emitted(policy.text());
    policy.accept(token(2), "TO");
    emitted = std::string(policy.text());
    EXPECT_EQ(emitted.find('S'), std::string::npos);
    policy.accept(token(3), "P ignored");
    EXPECT_EQ(policy.finish(), SequenceFinish::kStop);
    EXPECT_EQ(policy.text(), "hello ");
    EXPECT_THROW(policy.accept(token(4), "x"), std::logic_error);
}
TEST(SequencePolicy, UnicodeBoundariesAndReset)
{
    SequencePolicy policy;
    SequenceOptions o;
    o.maxOutputTokens = 4;
    policy.reset(o, 8);
    policy.accept(token(1), "\xe2");
    EXPECT_TRUE(policy.text().empty());
    policy.accept(token(2), "\x82\xac");
    EXPECT_EQ(policy.text(), "\xe2\x82\xac");
    policy.accept(token(3), "\xff");
    EXPECT_EQ(policy.text(), "\xe2\x82\xac\xef\xbf\xbd");
    policy.accept(token(4), "\xf0");
    EXPECT_EQ(policy.finish(), SequenceFinish::kLength);
    EXPECT_EQ(policy.text(), "\xe2\x82\xac\xef\xbf\xbd\xef\xbf\xbd");
    o.stopStrings = {"\xe2\x82\xac"};
    policy.reset(o, 8);
    policy.accept(token(1), "a\xe2");
    EXPECT_TRUE(policy.text().empty());
    policy.accept(token(2), "\x82\xac");
    EXPECT_EQ(policy.text(), "a");
    EXPECT_EQ(policy.finish(), SequenceFinish::kStop);
}
TEST(SequencePolicy, IndependentThinkingAndEos)
{
    SequenceOptions o;
    o.enableThinking = true;
    o.thinkingStartTokenId = 1;
    o.thinkingEndTokenId = 2;
    o.primaryEosTokenId = 3;
    o.eosTokenIds = {3, 4};
    o.maxOutputTokens = 10;
    SequencePolicy a, b;
    a.reset(o, 10);
    o.enableThinking = false;
    b.reset(o, 10);
    a.accept(token(1), "");
    EXPECT_TRUE(a.last().thinking);
    a.accept(token(4), "");
    EXPECT_EQ(a.finish(), SequenceFinish::kNone);
    b.accept(token(4), "");
    EXPECT_EQ(b.finish(), SequenceFinish::kEos);
    a.accept(token(2), "");
    EXPECT_FALSE(a.last().thinking);
    a.accept(token(4), "");
    EXPECT_EQ(a.finish(), SequenceFinish::kEos);
    o.enableThinking = true;
    a.reset(o, 10);
    a.accept(token(1), "");
    a.accept(token(3), "");
    EXPECT_EQ(a.finish(), SequenceFinish::kEos);
    a.reset(o, 10);
    a.accept(token(9), "answer");
    EXPECT_FALSE(a.last().thinking);
}
TEST(SequencePolicy, ValidatesMetadataBeforeAdmission)
{
    SequenceOptions o;
    o.numLogprobs = 51;
    EXPECT_THROW(validateSequenceOptions(o, 8), std::invalid_argument);
    o.numLogprobs = 0;
    o.logitBias[1] = std::numeric_limits<float>::quiet_NaN();
    EXPECT_THROW(validateSequenceOptions(o, 8), std::invalid_argument);
    o.logitBias.clear();
    o.stopStrings = {""};
    EXPECT_THROW(validateSequenceOptions(o, 8), std::invalid_argument);
    o.stopStrings.clear();
    o.topK = 9;
    EXPECT_THROW(validateSequenceOptions(o, 8), std::invalid_argument);
    SequenceSampler sampler(4);
    float values[]{0, 1, 2, std::numeric_limits<float>::infinity()};
    EXPECT_THROW(sampler.sample(values, SequenceOptions{}, 0), std::runtime_error);
}
TEST(SequenceChannel, WraparoundByteLimitAndTerminalOnFull)
{
    SequenceChannel channel(2, 5);
    EXPECT_TRUE(channel.push(token(1), "abc"));
    EXPECT_FALSE(channel.push(token(2), "def"));
    auto first = channel.read(std::chrono::milliseconds(0));
    ASSERT_TRUE(first.update);
    EXPECT_EQ(first.update->text, "abc");
    EXPECT_TRUE(channel.push(token(2), "def"));
    EXPECT_TRUE(channel.push(token(3), "gh"));
    EXPECT_FALSE(channel.push(token(4), ""));
    channel.close();
    auto second = channel.read(std::chrono::milliseconds(0));
    ASSERT_TRUE(second.update);
    EXPECT_EQ(second.update->text, "def");
    auto third = channel.read(std::chrono::milliseconds(0));
    ASSERT_TRUE(third.update);
    EXPECT_EQ(third.update->text, "gh");
    EXPECT_TRUE(third.closed);
    EXPECT_TRUE(channel.read(std::chrono::milliseconds(0)).closed);
    EXPECT_FALSE(channel.push(token(5), ""));
}

TEST(SequencePolicy, FinalUtf8ReplacementParticipatesInStopMatch)
{
    SequencePolicy policy;
    SequenceOptions o;
    o.maxOutputTokens = 1;
    o.stopStrings = {"\xef\xbf\xbd"};
    policy.reset(o, 4);
    policy.accept(token(1), "\xe2");
    EXPECT_EQ(policy.finish(), SequenceFinish::kStop);
    EXPECT_TRUE(policy.text().empty());
}
