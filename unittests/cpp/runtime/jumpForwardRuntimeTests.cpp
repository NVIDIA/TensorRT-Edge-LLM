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

// Jump-forward through LLMInferenceRuntime with a substitute engine: the vanilla decoder appends the tokens a
// grammar forces without a forward pass and catches their KV up in one context-chunk step.

#include "fakeEngineExecutor.h"
#include "guidedTokenizerFixture.h"
#include "substituteEngine.h"

#include "common/bindingNames.h"
#include "common/executionPhase.h"
#include "runtime/llmInferenceRuntime.h"
#include "runtime/streaming.h"

#include <fstream>
#include <string>
#include <vector>

using namespace trt_edgellm;
using namespace substitute_engine;

namespace
{

//! The tiny tokenizer spends one token per letter: "a" is 0, the other letters start at 3.
constexpr int32_t kEos{1};
constexpr int32_t kH{'h' - 'b' + 3};
constexpr int32_t kE{'e' - 'b' + 3};
constexpr int32_t kL{'l' - 'b' + 3};
constexpr int32_t kO{'o' - 'b' + 3};

class JumpForwardRuntimeTest : public ModelDirTest
{
protected:
    std::filesystem::path stageModelDir() override
    {
        return writeModelDir("jumpForwardRuntimeTests", makeTinyVanillaConfig());
    }

    //! A greedy request whose grammar forces the literal `hello`, with jump-forward on or off.
    static rt::LLMGenerationRequest makeHelloRequest(bool jumpForward, int64_t maxGenerateLength = 8)
    {
        rt::LLMGenerationRequest request = makeGreedyRequest("a", maxGenerateLength);
        request.requests.front().guidedDecoding = rt::GuidedDecodingParams{rt::GuideType::kRegex, "hello", jumpForward};
        return request;
    }

    struct Run
    {
        rt::LLMGenerationResponse response;
        std::vector<FakeEngineExecutor::PreparedRound> rounds;
        std::vector<std::vector<int32_t>> pastLengths;
        std::vector<std::vector<int32_t>> queryLengths;
        size_t forwards{0};
    };

    Run run(rt::LLMGenerationRequest const& request)
    {
        auto engine = makeFakeEngine();
        FakeEngineExecutor* const fake = engine.get();
        fake->recordBinding(binding_names::kPastLengths);
        fake->recordBinding(binding_names::kQueryLengths);
        auto runtime = makeRuntime(makeVanillaArtifacts(mModelDir, std::move(engine), mStream));
        Run result;
        EXPECT_TRUE(runtime.handleRequest(request, result.response, mStream));
        result.rounds = fake->preparedRounds();
        result.pastLengths = fake->preparedBinding(binding_names::kPastLengths);
        result.queryLengths = fake->preparedBinding(binding_names::kQueryLengths);
        result.forwards = fake->forwardsRun();
        return result;
    }
};

// Given a request whose grammar leaves only "hello" and the end of text, with jump-forward on
// When it is generated
// Then the text is the forced word ending at EOS, the engine ran one prefill, one decode step and one catch-up
//      step instead of a forward per letter, and the catch-up step fed the sampled letter plus the three forced
//      letters as one context chunk behind the two committed positions
TEST_F(JumpForwardRuntimeTest, ForcedLettersAreAppendedAndCaughtUpInOneContextChunk)
{
    Run const on = run(makeHelloRequest(/*jumpForward=*/true));

    ASSERT_EQ(on.response.outputIds.size(), 1U);
    EXPECT_EQ(on.response.outputIds[0], (std::vector<int32_t>{kH, kE, kL, kL, kO, kEos}));
    EXPECT_EQ(on.response.outputTexts[0], "hello");
    EXPECT_EQ(on.response.finishReasons[0], rt::FinishReason::kEndId);

    // Prefill samples 'h'; the decode step samples 'e' and appends the forced "llo"; the catch-up step feeds all four
    // and samples EOS.
    EXPECT_EQ(on.forwards, 3U);
    ASSERT_EQ(on.rounds.size(), 3U);
    EXPECT_EQ(on.rounds[0].profileIndex, kPrefillProfile);
    EXPECT_EQ(on.rounds[1].profileIndex, kDecodeProfile);
    EXPECT_EQ(on.rounds[1].dims.seqLen, 1);
    EXPECT_EQ(on.rounds[2].profileIndex, kPrefillProfile) << "the catch-up step is a context chunk";
    EXPECT_EQ(on.rounds[2].dims.executionPhaseLen, static_cast<int64_t>(rt::ExecutionPhase::kContextChunk));
    EXPECT_EQ(on.rounds[2].dims.seqLen, 4) << "one row per pending token: e, l, l, o";
    EXPECT_EQ(on.rounds[2].dims.selectLen, 1);
    ASSERT_EQ(on.pastLengths.size(), 3U);
    EXPECT_EQ(on.pastLengths[2], (std::vector<int32_t>{2})) << "the prompt 'a' and the sampled 'h' have KV";
    EXPECT_EQ(on.queryLengths[2], (std::vector<int32_t>{4}));
}

// Given a jump-forward request that asks for log-probabilities
// When it is generated
// Then every token has an entry list, a sampled token's list holds the requested K entries, and a forced token's
//      list holds that token alone at log-probability zero
TEST_F(JumpForwardRuntimeTest, ForcedTokensGetASyntheticLogprobRow)
{
    constexpr int32_t kTopK{3};
    rt::LLMGenerationRequest request = makeHelloRequest(/*jumpForward=*/true);
    request.numLogprobs = kTopK;
    Run const on = run(request);

    ASSERT_EQ(on.response.outputIds[0], (std::vector<int32_t>{kH, kE, kL, kL, kO, kEos}));
    ASSERT_EQ(on.response.logprobs.size(), 1U);
    auto const& rows = on.response.logprobs[0];
    ASSERT_EQ(rows.size(), 6U) << "one row per token, forced ones included";
    for (size_t const sampled : {0U, 1U, 5U})
    {
        EXPECT_EQ(rows[sampled].size(), static_cast<size_t>(kTopK)) << "row " << sampled;
    }
    for (size_t const forced : {2U, 3U, 4U})
    {
        ASSERT_EQ(rows[forced].size(), 1U) << "row " << forced;
        EXPECT_EQ(rows[forced][0].tokenId, on.response.outputIds[0][forced]);
        EXPECT_EQ(rows[forced][0].logprob, 0.0F);
    }
    EXPECT_EQ(rows[2][0].piece, "l");
    EXPECT_EQ(rows[3][0].piece, "l");
    EXPECT_EQ(rows[4][0].piece, "o");
}

// Given a grammar that forces text around two free positions (h, e, l, l, any letter, o, any letter), so the
//      tokens the engine chooses there depend on the KV state behind them
// When the request runs with and without jump-forward
// Then both runs choose the same tokens at the free positions, i.e. a catch-up step leaves the same KV state as
//      one forward per token, while the jump run spends fewer forwards
TEST_F(JumpForwardRuntimeTest, ChunkStepLeavesTheSameStateAsOneForwardPerToken)
{
    auto const makeRequest = [](bool jumpForward) {
        rt::LLMGenerationRequest request = makeGreedyRequest("a", /*maxGenerateLength=*/10);
        request.requests.front().guidedDecoding
            = rt::GuidedDecodingParams{rt::GuideType::kRegex, "hell[a-z]o[a-z]", jumpForward};
        return request;
    };
    Run const off = run(makeRequest(false));
    Run const on = run(makeRequest(true));

    ASSERT_EQ(off.response.outputIds.size(), 1U);
    ASSERT_EQ(off.response.outputIds[0].size(), 8U) << "four forced letters, a free one, o, a free one, EOS";
    std::vector<int32_t> const& offIds = off.response.outputIds[0];
    EXPECT_EQ(offIds[0], kH);
    EXPECT_EQ(offIds[1], kE);
    EXPECT_EQ(offIds[2], kL);
    EXPECT_EQ(offIds[3], kL);
    EXPECT_EQ(offIds[5], kO);
    EXPECT_EQ(offIds[7], kEos);
    EXPECT_NE(offIds[4], kEos);
    EXPECT_NE(offIds[6], kEos);
    EXPECT_EQ(on.response.outputIds[0], offIds) << "the free positions after a catch-up step chose differently";
    EXPECT_EQ(off.forwards, 8U);
    // Prefill; e (+ forced l, l); the chunk e,l,l samples the first free letter (+ forced o); the chunk ?,o samples
    // the second free letter; one decode step samples EOS.
    EXPECT_EQ(on.forwards, 5U);
}

// Given a batch of a jump-forward sequence and a free one, where the jump sequence is cancelled as soon as its
//      forced letters are appended, so the next step finds it finished with its pending run still without KV
// When that step runs
// Then the catch-up step still feeds the cancelled slot's run beside the free slot's token, the cancelled slot
//      leaves with what it had, and the free slot decodes exactly what it decodes alone
TEST_F(JumpForwardRuntimeTest, ACancelledSlotWithAPendingRunIsCaughtUpAndEvictedWithoutDisturbingItsNeighbour)
{
    constexpr int64_t kGenerateLength{6};
    rt::LLMGenerationResponse alone;
    {
        auto runtime = makeRuntime(makeVanillaArtifacts(mModelDir, makeFakeEngine(), mStream));
        ASSERT_TRUE(runtime.handleRequest(makeGreedyRequest("bcd", kGenerateLength), alone, mStream));
    }

    rt::LLMGenerationRequest batch = makeGreedyBatch({"a", "bcd"}, kGenerateLength);
    batch.requests[0].guidedDecoding = rt::GuidedDecodingParams{rt::GuideType::kRegex, "hello", /*jumpForward=*/true};
    std::shared_ptr<rt::StreamChannel> const channel = rt::StreamChannel::create();
    batch.streamChannels = {channel, nullptr};
    batch.onTokenGenerated = [&channel](rt::TokenCallbackInfo const& info) {
        if (info.batchIdx == 0 && info.tokenId == kO)
        {
            channel->cancel();
        }
    };
    auto engine = makeFakeEngine();
    FakeEngineExecutor* const fake = engine.get();
    fake->recordBinding(binding_names::kQueryLengths);
    auto runtime = makeRuntime(makeVanillaArtifacts(mModelDir, std::move(engine), mStream));
    rt::LLMGenerationResponse batched;
    ASSERT_TRUE(runtime.handleRequest(batch, batched, mStream));

    ASSERT_EQ(batched.outputIds.size(), 2U);
    EXPECT_EQ(batched.finishReasons[0], rt::FinishReason::kCancelled);
    EXPECT_EQ(batched.outputIds[0], (std::vector<int32_t>{kH, kE, kL, kL, kO}));
    EXPECT_EQ(batched.outputIds[1], alone.outputIds[0]) << "the cancelled row's catch-up disturbed the free slot";
    auto const& queryLengths = fake->preparedBinding(binding_names::kQueryLengths);
    ASSERT_GE(queryLengths.size(), 3U);
    EXPECT_EQ(queryLengths[2], (std::vector<int32_t>{4, 1})) << "the cancelled slot's pending run was still fed";
}

// Given a grammar forcing "hello", a generate budget of three tokens and a context cache
// When the jump after the sampled "e" can add one letter before the budget is spent
// Then the request ends by length with the first three letters, its two tokens without KV close the sequence without
//      tripping the cache ledger, and the cache still reports the sequence
TEST_F(JumpForwardRuntimeTest, AJumpThatSpendsTheLengthBudgetEndsTheRequestUnderAContextCache)
{
    rt::DeploymentOptions options{};
    options.engineDir = mModelDir.string();
    options.contextCacheConfig = {/*enabled=*/true, /*maxRecords=*/4};
    rt::LLMInferenceRuntime runtime{
        makeVanillaArtifacts(mModelDir, makeFakeEngine(), mStream), std::move(options), mStream};
    ASSERT_TRUE(runtime.getContextCacheMetrics().has_value());
    rt::LLMGenerationResponse response;
    ASSERT_TRUE(
        runtime.handleRequest(makeHelloRequest(/*jumpForward=*/true, /*maxGenerateLength=*/3), response, mStream));

    ASSERT_EQ(response.outputIds.size(), 1U);
    EXPECT_EQ(response.outputIds[0], (std::vector<int32_t>{kH, kE, kL}));
    EXPECT_EQ(response.finishReasons[0], rt::FinishReason::kLength);
    EXPECT_EQ(runtime.getContextCacheMetrics()->admittedSequences, 1U);
}

// Given a grammar forcing "hello" and the stop string "ll", with jump-forward on
// When the jump appends the forced letters that spell the stop string
// Then the request ends on that step with the stop-words reason and its text cut before the stop string, its token
//      list intact, and no further forward
TEST_F(JumpForwardRuntimeTest, AStopStringSpelledByForcedTextEndsTheRequestOnTheJumpStep)
{
    rt::LLMGenerationRequest request = makeHelloRequest(/*jumpForward=*/true);
    request.requests.front().stopStrings = {"ll"};
    Run const on = run(request);

    ASSERT_EQ(on.response.outputIds.size(), 1U);
    EXPECT_EQ(on.response.finishReasons[0], rt::FinishReason::kStopWords);
    EXPECT_EQ(on.response.outputTexts[0], "he");
    EXPECT_EQ(on.response.outputIds[0], (std::vector<int32_t>{kH, kE, kL, kL, kO})) << "outputIds keep every token";
    EXPECT_EQ(on.forwards, 2U) << "the stop was found in the forced text, before any catch-up step";
}

//! An engine whose context chunk holds three tokens: a catch-up row carries the sampled token plus at most two.
class JumpForwardNarrowChunkTest : public ModelDirTest
{
protected:
    static constexpr int32_t kChunkTokens{3};

    std::filesystem::path stageModelDir() override
    {
        Json config = makeTinyVanillaConfig();
        config["builder_config"]["max_input_len"] = kChunkTokens;
        return writeModelDir("jumpForwardNarrowChunkTests", config);
    }
};

// Given an engine whose context chunk holds three tokens and a grammar forcing "hello"
// When the request runs with jump-forward
// Then it spells the whole word, no step feeds more tokens than the chunk holds, and it costs fewer forwards than the
//      six of a run without jump-forward
TEST_F(JumpForwardNarrowChunkTest, AJumpIsCappedSoTheCatchUpRowFitsTheContextChunk)
{
    auto engine = makeFakeEngine();
    FakeEngineExecutor* const fake = engine.get();
    fake->recordBinding(binding_names::kQueryLengths);
    auto runtime = makeRuntime(makeVanillaArtifacts(mModelDir, std::move(engine), mStream));
    rt::LLMGenerationRequest request = makeGreedyRequest("a", /*maxGenerateLength=*/8);
    request.requests.front().guidedDecoding = rt::GuidedDecodingParams{rt::GuideType::kRegex, "hello", true};
    rt::LLMGenerationResponse response;
    ASSERT_TRUE(runtime.handleRequest(request, response, mStream));

    ASSERT_EQ(response.outputIds.size(), 1U);
    EXPECT_EQ(response.outputIds[0], (std::vector<int32_t>{kH, kE, kL, kL, kO, kEos}));
    EXPECT_LT(fake->forwardsRun(), 6U) << "no letter was jumped";
    auto const& queryLengths = fake->preparedBinding(binding_names::kQueryLengths);
    ASSERT_FALSE(queryLengths.empty());
    for (size_t round = 0; round < queryLengths.size(); ++round)
    {
        for (int32_t const queryLength : queryLengths[round])
        {
            EXPECT_LE(queryLength, kChunkTokens) << "round " << round;
        }
    }
}

//! The byte-level BPE fixture of the guided-decoding tests, staged as a deployment: its vocabulary has the merged
//! pieces (`",`) a rollback needs.
class JumpForwardBpeRuntimeTest : public ModelDirTest
{
protected:
    std::filesystem::path stageModelDir() override
    {
        auto const dir = makeScratchDir("jumpForwardBpeRuntimeTests");
        std::ofstream(dir / "config.json") << makeTinyVanillaConfig().dump(2);
        guidedFixture::writeBpeTokenizer(dir);
        std::ofstream(dir / "chat_template.jinja") << "{% for message in messages %}{{ message.content }}{% endfor %}";
        return dir;
    }

    //! `"a+","b"`: after the model closes `"a` with a quote, the grammar forces `,"b"`, which re-encodes together
    //! with that quote as `",` and rolls it back.
    static rt::LLMGenerationRequest makeRollbackRequest()
    {
        rt::LLMGenerationRequest request = makeGreedyRequest("a", /*maxGenerateLength=*/8);
        request.requests.front().guidedDecoding
            = rt::GuidedDecodingParams{rt::GuideType::kRegex, "\"a+\",\"b\"", /*jumpForward=*/true};
        request.numLogprobs = 2;
        return request;
    }

    std::unique_ptr<FakeEngineExecutor> makeRollbackEngine()
    {
        auto engine = makeFakeEngine();
        // Prefill samples the opening quote; the decode steps sample `a` and the closing quote.
        engine->forceOutputTokensPerRound({{guidedFixture::kQUOTE}, {guidedFixture::kA}, {guidedFixture::kQUOTE}});
        return engine;
    }

    static std::vector<int32_t> expectedIds()
    {
        using namespace guidedFixture;
        return {kQUOTE, kA, kQUOTE_COMMA, kQUOTE, kB, kQUOTE, kEOS};
    }
};

// Given a sampled closing quote the forced `,"b"` merges with, under a context cache
// When the jump is applied and the request runs to its end
// Then the quote is replaced by the merged piece in the token list and the generate count, its logprob row is
//      discarded and the replacement pieces get synthetic rows, the catch-up step feeds the whole run, and the
//      cache's ledger accepts the replaced token's progress
TEST_F(JumpForwardBpeRuntimeTest, ARollbackKeepsTheContextCacheLedgerConsistent)
{
    auto engine = makeRollbackEngine();
    FakeEngineExecutor* const fake = engine.get();
    fake->recordBinding(binding_names::kQueryLengths);
    rt::DeploymentOptions options{};
    options.engineDir = mModelDir.string();
    options.contextCacheConfig = {/*enabled=*/true, /*maxRecords=*/4};
    rt::LLMInferenceRuntime runtime{
        makeVanillaArtifacts(mModelDir, std::move(engine), mStream), std::move(options), mStream};
    ASSERT_TRUE(runtime.getContextCacheMetrics().has_value());
    rt::LLMGenerationResponse response;
    ASSERT_TRUE(runtime.handleRequest(makeRollbackRequest(), response, mStream));

    ASSERT_EQ(response.outputIds.size(), 1U);
    EXPECT_EQ(response.outputIds[0], expectedIds());
    EXPECT_EQ(response.outputTexts[0], "\"a\",\"b\"");
    EXPECT_EQ(response.finishReasons[0], rt::FinishReason::kEndId);
    EXPECT_EQ(runtime.getContextCacheMetrics()->admittedSequences, 1U);
    // Prefill, `a`, the quote (+ the jump), then one catch-up step over `",` `"` `b` `"`.
    EXPECT_EQ(fake->forwardsRun(), 4U);
    auto const& queryLengths = fake->preparedBinding(binding_names::kQueryLengths);
    ASSERT_EQ(queryLengths.size(), 4U);
    EXPECT_EQ(queryLengths[3], (std::vector<int32_t>{4}));

    ASSERT_EQ(response.logprobs.size(), 1U);
    auto const& rows = response.logprobs[0];
    ASSERT_EQ(rows.size(), 7U) << "one row per token; the rolled-back quote's row is gone";
    for (size_t const sampled : {0U, 1U, 6U})
    {
        EXPECT_EQ(rows[sampled].size(), 2U) << "row " << sampled;
    }
    for (size_t const forced : {2U, 3U, 4U, 5U})
    {
        ASSERT_EQ(rows[forced].size(), 1U) << "row " << forced;
        EXPECT_EQ(rows[forced][0].tokenId, expectedIds()[forced]);
        EXPECT_EQ(rows[forced][0].logprob, 0.0F);
    }
}

//! The SentencePiece-style fixture, staged as a deployment: encode() prepends a word marker, so a jump window has
//! to reach back to a word boundary.
class JumpForwardMetaspaceRuntimeTest : public ModelDirTest
{
protected:
    std::filesystem::path stageModelDir() override
    {
        auto const dir = makeScratchDir("jumpForwardMetaspaceRuntimeTests");
        std::ofstream(dir / "config.json") << makeTinyVanillaConfig().dump(2);
        guidedFixture::writeMetaspaceTokenizer(dir);
        std::ofstream(dir / "chat_template.jinja") << "{% for message in messages %}{{ message.content }}{% endfor %}";
        return dir;
    }
};

// Given a Metaspace tokenizer and a grammar that forces `,` after the generated word ` aa`
// When the jump window is built behind the mid-word `a`
// Then it widens back to the ` a` the word starts with, so the jump is taken: the comma is appended without a
//      forward and caught up with the `a`, while the same request without jump-forward spends a forward per token
TEST_F(JumpForwardMetaspaceRuntimeTest, TheJumpWindowWidensBackToAWordBoundary)
{
    using namespace guidedFixture;
    auto const run = [&](bool jumpForward) {
        auto engine = makeFakeEngine();
        FakeEngineExecutor* const fake = engine.get();
        engine->forceOutputTokensPerRound({{kMS_MARKER_A}, {kMS_A}});
        auto runtime = makeRuntime(makeVanillaArtifacts(mModelDir, std::move(engine), mStream));
        rt::LLMGenerationRequest request = makeGreedyRequest("a", /*maxGenerateLength=*/6);
        request.requests.front().guidedDecoding = rt::GuidedDecodingParams{rt::GuideType::kRegex, " aa,", jumpForward};
        rt::LLMGenerationResponse response;
        EXPECT_TRUE(runtime.handleRequest(request, response, mStream));
        return std::make_pair(response, fake->forwardsRun());
    };
    auto const [off, offForwards] = run(false);
    auto const [on, onForwards] = run(true);
    ASSERT_EQ(off.outputIds.size(), 1U);
    EXPECT_EQ(off.outputIds[0], (std::vector<int32_t>{kMS_MARKER_A, kMS_A, kMS_COMMA, kMS_EOS}));
    EXPECT_EQ(on.outputIds[0], off.outputIds[0]);
    EXPECT_EQ(offForwards, 4U);
    EXPECT_EQ(onForwards, 3U) << "the comma was not jumped: the window did not reach the word boundary";
}

} // namespace
