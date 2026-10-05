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

// Assembling LLMInferenceRuntime around injected artifacts, with no serialized engine anywhere, and driving
// handleRequest() through it.
//
// The per-decoder suites live beside the decoders they exercise, under cpp/runtime/decoding. What stays here is
// what belongs to the runtime itself rather than to any one decoding strategy: assembly, the vanilla decode loop,
// request validation, and log-probability reporting.

#include "substituteEngine.h"

#include "runtime/llmInferenceRuntime.h"
#include "runtime/llmRankRuntime.h"
#include "runtime/runtimeStepper.h"
#include "scheduler/requestEngine.h"

using namespace trt_edgellm;
using namespace substitute_engine;

namespace
{

class RuntimeAssemblyTest : public ModelDirTest
{
protected:
    std::filesystem::path stageModelDir() override
    {
        return writeModelDir("runtimeAssemblyTests", makeTinyVanillaConfig());
    }
};

// Given a model directory holding a config and a tokenizer but no serialized engine
// When the runtime is assembled around an injected executor
// Then it comes up as a working single-engine deployment
TEST_F(RuntimeAssemblyTest, AssemblesWithoutAnyEngineFileOnDisk)
{
    auto artifacts = makeVanillaArtifacts(mModelDir, makeEngine(), mStream);

    auto runtime = makeRuntime(std::move(artifacts));

    ASSERT_FALSE(std::filesystem::exists(mModelDir / "llm.engine"));
    EXPECT_FALSE(runtime.hasDraftModel());
    EXPECT_STREQ(runtime.getSpeculativeDecodingStrategyName(), "vanilla");
}

TEST_F(RuntimeAssemblyTest, CountsThePromptThroughInferencePreparationWithoutMutatingTheRequest)
{
    auto artifacts = makeVanillaArtifacts(mModelDir, makeEngine(), mStream);
    auto runtime = makeRuntime(std::move(artifacts));

    auto const request = makeGreedyRequest("aa", /*maxGenerateLength=*/1);
    EXPECT_EQ(runtime.countPromptTokens(request), (std::vector<int32_t>{2}));
    EXPECT_TRUE(request.formattedRequests.empty());
}

// Given an executor that reports how much scratch memory it needs
// When the runtime is assembled
// Then it asks that executor, and hands back a buffer at least that large
TEST_F(RuntimeAssemblyTest, SizesSharedContextMemoryFromTheExecutorItWasGiven)
{
    auto engine = makeEngine();
    expectContextMemorySizedFromTheExecutor(*engine, kContextMemoryBytes);

    auto runtime = makeRuntime(makeVanillaArtifacts(mModelDir, std::move(engine), mStream));
}

// Given a request for N tokens against a vanilla deployment
// When it is generated
// Then the engine runs one prefill forward and N-1 decode forwards, each on the profile its shapes were built for
TEST_F(RuntimeAssemblyTest, DrivesPrefillAndOneDecodeRoundPerGeneratedToken)
{
    using ::testing::_;
    using ::testing::AllOf;
    using ::testing::Each;
    using ::testing::SizeIs;

    constexpr int64_t kMaxGenerateLength{4};

    auto engine = makeEngine();
    auto& mock = *engine;

    // Prefill emits the first token, so N tokens cost one prefill forward plus N-1 decode forwards. Each round
    // switches the engine to the profile its shapes were built for.
    EXPECT_CALL(mock, prepare(kPrefillProfile, _, _, _)).Times(1);
    EXPECT_CALL(mock, prepare(kDecodeProfile, _, _, _)).Times(kMaxGenerateLength - 1);
    EXPECT_CALL(mock, execute(_)).Times(kMaxGenerateLength).WillRepeatedly(emit({}));

    auto artifacts = makeVanillaArtifacts(mModelDir, std::move(engine), mStream);
    auto runtime = makeRuntime(std::move(artifacts));

    auto const request = makeGreedyRequest("a", kMaxGenerateLength);
    rt::LLMGenerationResponse response;
    ASSERT_TRUE(runtime.handleRequest(request, response, mStream));

    expectResponseCoversEverySlot(response, 1);
    ASSERT_EQ(response.outputIds.size(), 1U);
    // The prompt is one character, and this tokenizer spends one token per character. Asserting the count the
    // runtime reports back makes that a checked fact rather than a comment the other tests lean on.
    EXPECT_EQ(response.inputTokenCounts[0], 1);
    // Every forward left the logits zeroed, so the whole completion is the tie-break token.
    EXPECT_THAT(response.outputIds[0], AllOf(SizeIs(kMaxGenerateLength), Each(kZeroLogitsToken)));
    EXPECT_EQ(response.finishReasons[0], rt::FinishReason::kLength);
}

TEST_F(RuntimeAssemblyTest, UsesPreTokenizedInputWithoutMessages)
{
    using ::testing::_;

    auto engine = makeEngine();
    auto& mock = *engine;
    EXPECT_CALL(mock, execute(_)).WillOnce(emit({}));

    auto artifacts = makeVanillaArtifacts(mModelDir, std::move(engine), mStream);
    rt::LLMInferenceRuntime runtime{
        std::move(artifacts), mModelDir.string(), /*multimodalEngineDir=*/"", {}, std::nullopt, mStream};

    rt::LLMGenerationRequest request{};
    request.requests.emplace_back();
    request.preTokenizedInputIds.push_back({0});
    request.temperature = 0.0F;
    request.topK = 1;
    request.topP = 1.0F;
    request.maxGenerateLength = 1;

    rt::LLMGenerationResponse response;
    ASSERT_TRUE(runtime.handleRequest(request, response, mStream));

    expectResponseCoversEverySlot(response, 1);
    EXPECT_EQ(response.inputTokenCounts[0], 1);
    EXPECT_THAT(response.outputIds[0], ::testing::ElementsAre(kZeroLogitsToken));
}

TEST_F(RuntimeAssemblyTest, PreservesPreTokenizedInputWhileFormattingMessages)
{
    using ::testing::_;

    auto engine = makeEngine();
    auto& mock = *engine;
    EXPECT_CALL(mock, execute(_)).WillOnce(emit({}));

    auto artifacts = makeVanillaArtifacts(mModelDir, std::move(engine), mStream);
    rt::LLMInferenceRuntime runtime{
        std::move(artifacts), mModelDir.string(), /*multimodalEngineDir=*/"", {}, std::nullopt, mStream};

    auto request = makeGreedyRequest("a", /*maxGenerateLength=*/1);
    request.preTokenizedInputIds.push_back({0, 0});

    rt::LLMGenerationResponse response;
    ASSERT_TRUE(runtime.handleRequest(request, response, mStream));

    expectResponseCoversEverySlot(response, 1);
    EXPECT_EQ(response.inputTokenCounts[0], 2);
    ASSERT_EQ(request.formattedRequests.size(), 1U);
    EXPECT_EQ(request.formattedRequests[0].formattedCompleteRequest, "a");
}

// Given a length cap far above what the engine is going to emit
// When a forward pass samples the tokenizer's EOS id
// Then generation stops on that token and reports kEndId rather than kLength
TEST_F(RuntimeAssemblyTest, StopsAtTheEosTokenTheEngineProduces)
{
    using ::testing::_;
    using ::testing::InSequence;

    auto engine = makeEngine();
    auto& mock = *engine;

    auto artifacts = makeVanillaArtifacts(mModelDir, std::move(engine), mStream);
    auto const eosId = static_cast<int32_t>(artifacts.tokenizer->getEosId());

    // Prefill and the first decode round emit ordinary tokens; the second decode round emits EOS. Stating this as a
    // sequence means a fourth forward fails on its own — no counter, and the failure names the call that broke it.
    {
        InSequence seq;
        EXPECT_CALL(mock, execute(_)).Times(2).WillRepeatedly(emit({}));
        EXPECT_CALL(mock, execute(_)).WillOnce(emit({eosId}));
    }

    auto runtime = makeRuntime(std::move(artifacts));

    auto const request = makeGreedyRequest("a", /*maxGenerateLength=*/8);
    rt::LLMGenerationResponse response;
    ASSERT_TRUE(runtime.handleRequest(request, response, mStream));

    expectResponseCoversEverySlot(response, 1);
    ASSERT_EQ(response.outputIds.size(), 1U);
    EXPECT_EQ(response.outputIds[0].size(), 3U);
    EXPECT_EQ(response.outputIds[0].back(), eosId);
    // The distinction the test is named for: stopping on the sampled token rather than on the length cap, which is
    // still 8 rounds away.
    EXPECT_EQ(response.finishReasons[0], rt::FinishReason::kEndId);
}

// Given a batch of two whose first slot samples EOS while the second keeps going
// When the remaining rounds run
// Then the survivor decodes on from engine row 0, and both slots still report against their own index
TEST_F(RuntimeAssemblyTest, CompactsTheBatchWhenOneSlotFinishesAheadOfTheOther)
{
    using ::testing::_;
    using ::testing::InSequence;

    constexpr int64_t kMaxGenerateLength{5};

    auto engine = makeEngine();
    auto& mock = *engine;

    auto artifacts = makeVanillaArtifacts(mModelDir, std::move(engine), mStream);
    auto const eosId = static_cast<int32_t>(artifacts.tokenizer->getEosId());

    // On the second forward slot 0 hits EOS while slot 1 keeps going. Slot 1 then occupies engine row 0 for the
    // remaining rounds, which is the batch compaction under test: the later expectations name row 0 and would not
    // match if the survivor had stayed in row 1.
    {
        InSequence seq;
        EXPECT_CALL(mock, execute(_)).WillOnce(emit({}));
        EXPECT_CALL(mock, execute(_)).WillOnce(emit({eosId, -1}));
        EXPECT_CALL(mock, execute(_)).Times(kMaxGenerateLength - 2).WillRepeatedly(emit({}));
    }

    auto runtime = makeRuntime(std::move(artifacts));

    auto request = makeGreedyRequest("a", kMaxGenerateLength);
    request.requests.push_back(request.requests.front());
    rt::LLMGenerationResponse response;
    ASSERT_TRUE(runtime.handleRequest(request, response, mStream));

    expectResponseCoversEverySlot(response, 2);
    ASSERT_EQ(response.outputIds.size(), 2U);
    EXPECT_EQ(response.outputIds[0].size(), 2U);
    EXPECT_EQ(response.outputIds[0].back(), eosId);
    EXPECT_EQ(response.outputIds[1].size(), static_cast<size_t>(kMaxGenerateLength));
    EXPECT_NE(response.outputIds[1].back(), eosId);
    // The two slots left for different reasons, and the compacted slot still reports against its own index.
    EXPECT_EQ(response.finishReasons[0], rt::FinishReason::kEndId);
    EXPECT_EQ(response.finishReasons[1], rt::FinishReason::kLength);
}

TEST_F(RuntimeAssemblyTest, LogicalEvictionRemapsTheSurvivorAndTheNextRequestRestoresIdentity)
{
    using ::testing::_;
    using ::testing::InSequence;

    constexpr int64_t kMaxGenerateLength{4};
    auto engine = makeEngine();
    auto& mock = *engine;

    auto artifacts = makeVanillaArtifacts(mModelDir, std::move(engine), mStream);
    auto const eosId = static_cast<int32_t>(artifacts.tokenizer->getEosId());

    {
        InSequence seq;
        EXPECT_CALL(mock, execute(_)).WillOnce(emit({}));
        EXPECT_CALL(mock, execute(_)).WillOnce(emit({eosId, -1}));
        EXPECT_CALL(mock, execute(_)).Times(kMaxGenerateLength - 2).WillRepeatedly(emit({}));
        EXPECT_CALL(mock, execute(_)).WillOnce(emit({eosId}));
    }

    rt::LLMInferenceRuntime runtime{
        std::move(artifacts), mModelDir.string(), /*multimodalEngineDir=*/"", {}, std::nullopt, mStream};

    auto firstRequest = makeGreedyRequest("a", kMaxGenerateLength);
    firstRequest.requests.push_back(firstRequest.requests.front());
    rt::LLMGenerationResponse firstResponse;
    ASSERT_TRUE(runtime.handleRequest(firstRequest, firstResponse, mStream));

    ASSERT_GE(mFirstPagePerPrepare.size(), 3U);
    EXPECT_NE(mFirstPagePerPrepare.back(), 0) << "logical compaction must preserve the old slot-1 physical page";

    size_t const beforeSecondRequest = mFirstPagePerPrepare.size();
    auto const secondRequest = makeGreedyRequest("a", /*maxGenerateLength=*/2);
    rt::LLMGenerationResponse secondResponse;
    ASSERT_TRUE(runtime.handleRequest(secondRequest, secondResponse, mStream));

    ASSERT_GT(mFirstPagePerPrepare.size(), beforeSecondRequest);
    EXPECT_EQ(mFirstPagePerPrepare[beforeSecondRequest], 0)
        << "a new unmanaged request must restore the base page table to identity";
}

TEST_F(RuntimeAssemblyTest, RejectsEngineWithoutContextSequenceCountCarrier)
{
    using ::testing::Return;
    using ::testing::StrEq;

    rt::LLMEngineConfig config;
    auto engine = makeEngine();
    ON_CALL(*engine, hasIOTensor(StrEq(binding_names::kContextSequenceCountCarrier))).WillByDefault(Return(false));

    EXPECT_THROW(validateAgainstEngine(config, *engine, "base"), std::runtime_error);
}

TEST_F(RuntimeAssemblyTest, AcceptsInt32ContextSequenceCountCarrier)
{
    using ::testing::Return;
    using ::testing::StrEq;

    rt::LLMEngineConfig config;
    auto engine = makeEngine();
    ON_CALL(*engine, hasIOTensor(StrEq(binding_names::kContextSequenceCountCarrier))).WillByDefault(Return(true));
    ON_CALL(*engine, getBindingDataType(StrEq(binding_names::kContextSequenceCountCarrier)))
        .WillByDefault(Return(nvinfer1::DataType::kINT32));

    EXPECT_NO_THROW(validateAgainstEngine(config, *engine, "base"));
}

TEST_F(RuntimeAssemblyTest, RejectsObsoleteGenericTokenOwnerBinding)
{
    using ::testing::Return;
    using ::testing::StrEq;

    rt::LLMEngineConfig config;
    auto engine = makeEngine();
    ON_CALL(*engine, hasIOTensor(StrEq("token_to_sequence"))).WillByDefault(Return(true));

    EXPECT_THROW(validateAgainstEngine(config, *engine, "base"), std::runtime_error);
}

// --------------------------------------------------------------------------
// Request validation.
//
// handleRequest() screens the request before it touches runtime state, so every
// rejection below has two halves: the call reports failure, and it did so
// without running a forward pass or leaving anything in the response. Callers
// that ignore the return value must not be able to read a previous request's
// output as if it belonged to this one.
//
// The same vanilla deployment as above; only the requests differ.
// --------------------------------------------------------------------------

class RequestValidationTest : public RuntimeAssemblyTest
{
protected:
    //! A request the runtime would accept, so each test below differs from a
    //! working request by exactly the field it is about.
    rt::LLMGenerationRequest validRequest(size_t slots = 1)
    {
        auto request = makeGreedyRequest("a", 2);
        while (request.requests.size() < slots)
        {
            request.requests.push_back(request.requests.front());
        }
        return request;
    }

    //! Assert the shared half of every rejection: no forward pass, and nothing
    //! left behind in the response.
    void expectRejectedWithoutRunning(rt::LLMGenerationRequest const& request)
    {
        auto engine = makeEngine();
        EXPECT_CALL(*engine, execute(::testing::_)).Times(0);

        auto artifacts = makeVanillaArtifacts(mModelDir, std::move(engine), mStream);
        auto runtime = makeRuntime(std::move(artifacts));

        rt::LLMGenerationResponse response;
        EXPECT_FALSE(runtime.handleRequest(request, response, mStream));
        EXPECT_TRUE(response.outputIds.empty());
        EXPECT_TRUE(response.outputTexts.empty());
        EXPECT_TRUE(response.finishReasons.empty());
    }
};

// A batch of zero has no work in it, and the decode loop's per-slot buffers are
// sized from the batch size, so an empty one would index into nothing.
TEST_F(RequestValidationTest, RejectsARequestCarryingNoSlots)
{
    expectRejectedWithoutRunning(rt::LLMGenerationRequest{});
}

// The engine's optimization profiles are built for a maximum batch, and the KV
// pool is sized to match. A larger batch has nowhere to run.
TEST_F(RequestValidationTest, RejectsABatchLargerThanTheEngineWasBuiltFor)
{
    expectRejectedWithoutRunning(validRequest(static_cast<size_t>(kMaxBatchSize) + 1));

    // The boundary itself is accepted, which is what makes the rejection above
    // attributable to the limit rather than to batching at all.
    auto engine = makeEngine();
    EXPECT_CALL(*engine, execute(::testing::_)).WillRepeatedly(emit({}));
    auto artifacts = makeVanillaArtifacts(mModelDir, std::move(engine), mStream);
    auto runtime = makeRuntime(std::move(artifacts));

    rt::LLMGenerationResponse response;
    EXPECT_TRUE(runtime.handleRequest(validRequest(static_cast<size_t>(kMaxBatchSize)), response, mStream));
}

// One empty slot fails the whole request: the batch shares a forward pass, so
// there is no partial acceptance to fall back to, and a slot with no messages
// would otherwise generate from an empty prompt.
//
// Defended in more than one place -- removing the explicit guard leaves the
// chat template rejecting it just as early -- so this pins the outcome (nothing
// runs, nothing is returned) rather than any single check.
TEST_F(RequestValidationTest, RejectsTheWholeBatchWhenOneSlotHasNoMessages)
{
    auto request = validRequest(2);
    request.requests[1].messages.clear();

    expectRejectedWithoutRunning(request);
}

// Logit bias is applied by scattering into a vocabulary-sized row. A token id
// outside the vocabulary writes past that row, which is why the bound is the
// full vocabulary and why both ends are checked.
TEST_F(RequestValidationTest, RejectsALogitBiasTokenOutsideTheVocabulary)
{
    auto negative = validRequest();
    negative.requests[0].logitBias[-1] = 1.0F;
    expectRejectedWithoutRunning(negative);

    auto past = validRequest();
    past.requests[0].logitBias[static_cast<int32_t>(kVocabSize)] = 1.0F;
    expectRejectedWithoutRunning(past);

    // The last valid id is accepted, so the rejection above is the bound and not
    // an off-by-one in the other direction.
    auto last = validRequest();
    last.requests[0].logitBias[static_cast<int32_t>(kVocabSize) - 1] = 1.0F;

    auto engine = makeEngine();
    EXPECT_CALL(*engine, execute(::testing::_)).WillRepeatedly(emit({}));
    auto artifacts = makeVanillaArtifacts(mModelDir, std::move(engine), mStream);
    auto runtime = makeRuntime(std::move(artifacts));

    rt::LLMGenerationResponse response;
    EXPECT_TRUE(runtime.handleRequest(last, response, mStream));
}

// A bias is added to a logit before sampling. NaN poisons the row's comparisons
// so no token can win, and an unbounded magnitude makes the bias the only thing
// that decides the output.
TEST_F(RequestValidationTest, RejectsALogitBiasValueThatIsNotFiniteAndBounded)
{
    for (float const bias : {std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity(),
             limits::security::kMinLogitBias - 1.0F, limits::security::kMaxLogitBias + 1.0F})
    {
        auto request = validRequest();
        request.requests[0].logitBias[0] = bias;
        SCOPED_TRACE(bias);
        expectRejectedWithoutRunning(request);
    }
}

// The kMaxLogitBiasTokens cap is deliberately not covered here. The bias map is
// keyed by token id, so this deployment's 128-token vocabulary cannot hold
// enough valid entries to reach it, and a map padded with out-of-range ids is
// rejected by the size check before the range check runs -- leaving the
// rejection unattributable. Covering it would mean a fixture whose only purpose
// is a vocabulary larger than the cap.

// A text-only deployment has no vision or audio runner. Accepting the buffers
// anyway would drop them silently and answer the prompt without its attachment.
TEST_F(RequestValidationTest, RejectsMediaThisDeploymentHasNoRunnerFor)
{
    auto withImage = validRequest();
    withImage.requests[0].imageBuffers.emplace_back();
    expectRejectedWithoutRunning(withImage);

    auto withAudio = validRequest();
    withAudio.requests[0].audioBuffers.emplace_back();
    expectRejectedWithoutRunning(withAudio);

    auto withTrajectory = validRequest();
    withTrajectory.requests[0].pastTrajectory.emplace();
    expectRejectedWithoutRunning(withTrajectory);
}

// The response is cleared before validation, not after it. Without that, a
// caller that reuses one response object and ignores the return value reads the
// previous request's tokens as this request's answer.
//
// Given a response object still holding a previous request's output
// When a request that fails validation is handled into that same object
// Then it is left empty
TEST_F(RequestValidationTest, ARejectedRequestClearsWhateverThePreviousOneLeftBehind)
{
    auto engine = makeEngine();
    EXPECT_CALL(*engine, execute(::testing::_)).WillRepeatedly(emit({}));

    auto artifacts = makeVanillaArtifacts(mModelDir, std::move(engine), mStream);
    auto runtime = makeRuntime(std::move(artifacts));

    rt::LLMGenerationResponse response;
    ASSERT_TRUE(runtime.handleRequest(validRequest(), response, mStream));
    ASSERT_FALSE(response.outputIds.empty());

    EXPECT_FALSE(runtime.handleRequest(rt::LLMGenerationRequest{}, response, mStream));
    EXPECT_TRUE(response.outputIds.empty());
    EXPECT_TRUE(response.outputTexts.empty());
    EXPECT_TRUE(response.finishReasons.empty());
}

// One runtime drives one request at a time; its decode buffers are members. The
// guard is documented as rejecting the second caller rather than corrupting
// both, so the reentrant call is made from inside a forward pass -- the only
// point at which a real overlap could happen.
//
// Given a request already in flight, reentered from inside its own forward pass
// When a second request is made on the same runtime
// Then the second is refused and the first still completes normally
TEST_F(RequestValidationTest, RejectsAReentrantRequestWithoutDisturbingTheOneInFlight)
{
    rt::LLMInferenceRuntime* runtimeUnderTest{nullptr};
    // Latched before the nested call, not after it: were the guard removed, the
    // nested request would run its own forward pass and reenter again, and a
    // flag set from the result would recurse until the stack ran out instead of
    // reporting a failure.
    bool reentryAttempted{false};
    bool reentryWasRejected{false};

    auto engine = makeEngine();
    EXPECT_CALL(*engine, execute(::testing::_)).WillRepeatedly([&](cudaStream_t stream) {
        if (runtimeUnderTest != nullptr && !reentryAttempted)
        {
            reentryAttempted = true;
            rt::LLMGenerationResponse nested;
            reentryWasRejected = !runtimeUnderTest->handleRequest(validRequest(), nested, stream);
            EXPECT_TRUE(nested.outputIds.empty());
        }
        writeLogits(*mLogits, kVocabSize, {}, stream);
        return true;
    });

    auto artifacts = makeVanillaArtifacts(mModelDir, std::move(engine), mStream);
    auto runtime = makeRuntime(std::move(artifacts));
    runtimeUnderTest = &runtime;

    rt::LLMGenerationResponse response;
    ASSERT_TRUE(runtime.handleRequest(makeGreedyRequest("a", 2), response, mStream));

    ASSERT_TRUE(reentryAttempted);
    EXPECT_TRUE(reentryWasRejected);
    // The outer request completed normally despite the rejected reentry.
    expectResponseCoversEverySlot(response, 1);
    EXPECT_EQ(response.outputIds[0].size(), 2U);
}

// countPromptTokens() answers "how long is this prompt" without generating.
// It has to agree with the count a real generation reports, or the two ways a
// caller can ask the same question disagree.
//
// Given a batch whose two slots carry prompts of different lengths
// When the prompt is counted without generating, and then generated
// Then the two answers agree slot for slot
TEST_F(RequestValidationTest, CountingPromptTokensAgreesWithWhatGenerationReports)
{
    auto engine = makeEngine();
    EXPECT_CALL(*engine, execute(::testing::_)).WillRepeatedly(emit({}));

    auto artifacts = makeVanillaArtifacts(mModelDir, std::move(engine), mStream);
    auto runtime = makeRuntime(std::move(artifacts));

    auto request = makeGreedyRequest("aaa", 2);
    request.requests.push_back(makeGreedyRequest("a", 2).requests.front());

    auto const counted = runtime.countPromptTokens(request);

    rt::LLMGenerationResponse response;
    ASSERT_TRUE(runtime.handleRequest(request, response, mStream));

    ASSERT_EQ(counted.size(), request.requests.size());
    EXPECT_EQ(counted, response.inputTokenCounts);
    // Distinct prompts, so agreement is not two identical constants meeting.
    EXPECT_NE(counted[0], counted[1]);
}

// The count comes from tokenizing text. Media expands into placeholder tokens
// the runtime only knows how to lay out during a real request, so answering
// with the text-only count would understate the prompt.
TEST_F(RequestValidationTest, CountingPromptTokensRefusesMediaRequests)
{
    auto artifacts = makeVanillaArtifacts(mModelDir, makeEngine(), mStream);
    auto runtime = makeRuntime(std::move(artifacts));

    auto request = makeGreedyRequest("a", 2);
    request.requests[0].imageBuffers.emplace_back();

    // Asserted on the reason, not just on throwing: a media request fails later
    // anyway when the template or the encoder rejects it, so a bare EXPECT_THROW
    // would still pass with this precondition removed.
    std::string reason;
    try
    {
        static_cast<void>(runtime.countPromptTokens(request));
    }
    catch (std::exception const& error)
    {
        reason = error.what();
    }
    EXPECT_THAT(reason, ::testing::HasSubstr("only available for text requests"));
}

// --------------------------------------------------------------------------
// Log-probability reporting.
//
// The substitute engine writes the logits, so the distribution they describe is
// known exactly and the expected log-probabilities are computed here from the
// definition rather than read out of the runtime. That makes these assertions
// an independent check of "logprobs are log(softmax(logits))" instead of a
// record of whatever the current implementation returns.
// --------------------------------------------------------------------------

class LogprobsTest : public RuntimeAssemblyTest
{
protected:
    //! writeLogits() marks one entry per row and leaves the rest zeroed, so a row
    //! is `kPeakLogit` in one place and 0 in kVocabSize-1 others. log(softmax) of
    //! that row is the peak (or 0) minus the log-sum-exp over the whole row.
    static constexpr float kPeakLogit{10.0F};

    static double logSumExpOfOneMarkedRow()
    {
        return std::log(std::exp(static_cast<double>(kPeakLogit)) + static_cast<double>(kVocabSize - 1));
    }

    static double expectedPeakLogprob()
    {
        return static_cast<double>(kPeakLogit) - logSumExpOfOneMarkedRow();
    }

    static double expectedOtherLogprob()
    {
        return -logSumExpOfOneMarkedRow();
    }

    //! A runtime over an engine whose forward passes emit `peaks` in order: the
    //! first is the token prefill produces, the rest are one decode round each.
    template <typename Body>
    void withEngineEmitting(std::vector<int32_t> const& peaks, Body&& body)
    {
        auto engine = makeEngine();
        {
            ::testing::InSequence const ordered;
            for (int32_t const peak : peaks)
            {
                EXPECT_CALL(*engine, execute(::testing::_)).WillOnce(emit({peak}));
            }
        }

        auto artifacts = makeVanillaArtifacts(mModelDir, std::move(engine), mStream);
        auto runtime = makeRuntime(std::move(artifacts));
        body(runtime);
    }
};

// Logprobs cost a log-softmax over the vocabulary and a device-to-host copy per
// step. A request that did not ask for them must not pay for them, and the
// absence has to be visible rather than reported as a list of empty steps.
//
// Given a request that did not ask for log-probabilities
// When it is generated
// Then none are reported at all
TEST_F(LogprobsTest, AreAbsentUnlessTheRequestAsksForThem)
{
    withEngineEmitting({5, 7}, [&](rt::LLMInferenceRuntime& runtime) {
        auto request = makeGreedyRequest("a", 2);
        ASSERT_EQ(request.numLogprobs, 0);

        rt::LLMGenerationResponse response;
        ASSERT_TRUE(runtime.handleRequest(request, response, mStream));

        ASSERT_EQ(response.logprobs.size(), 1U);
        EXPECT_TRUE(response.logprobs[0].empty());
        EXPECT_EQ(response.outputIds[0].size(), 2U);
    });
}

// The caller aligns logprobs to tokens by index, so there has to be exactly one
// entry list per generated token, each holding the requested number of entries.
// Reporting one list per decode round instead would drift by one at prefill.
//
// Given a request asking for the top K log-probabilities
// When N tokens are generated
// Then N entry lists come back, each holding K entries
TEST_F(LogprobsTest, ReportOneEntryListPerGeneratedToken)
{
    constexpr int32_t kTopK{4};

    withEngineEmitting({5, 7, 9}, [&](rt::LLMInferenceRuntime& runtime) {
        auto request = makeGreedyRequest("a", 3);
        request.numLogprobs = kTopK;

        rt::LLMGenerationResponse response;
        ASSERT_TRUE(runtime.handleRequest(request, response, mStream));

        ASSERT_EQ(response.logprobs.size(), 1U);
        ASSERT_EQ(response.logprobs[0].size(), response.outputIds[0].size());
        EXPECT_THAT(response.logprobs[0], ::testing::Each(::testing::SizeIs(kTopK)));
    });
}

// The values are log-probabilities of the engine's own logits. Both the peak and
// a runner-up are checked: the peak alone would also match an implementation
// that reported raw logits, since a near-certain token's log-probability is
// small, while the runner-ups are nowhere near their logit of zero.
//
// Given forward passes whose logits are known exactly
// When log-probabilities are reported for them
// Then every entry equals log(softmax(those logits)), computed here from the definition
TEST_F(LogprobsTest, ValuesAreTheLogSoftmaxOfTheLogitsTheEngineProduced)
{
    constexpr int32_t kTopK{3};

    withEngineEmitting({5, 7}, [&](rt::LLMInferenceRuntime& runtime) {
        auto request = makeGreedyRequest("a", 2);
        request.numLogprobs = kTopK;

        rt::LLMGenerationResponse response;
        ASSERT_TRUE(runtime.handleRequest(request, response, mStream));

        ASSERT_EQ(response.logprobs[0].size(), 2U);
        for (auto const& step : response.logprobs[0])
        {
            ASSERT_EQ(step.size(), static_cast<size_t>(kTopK));
            // Guard: an empty step list would make the loop below vacuous.
            EXPECT_NEAR(step[0].logprob, expectedPeakLogprob(), 1e-3);
            for (size_t k = 1; k < step.size(); ++k)
            {
                EXPECT_NEAR(step[k].logprob, expectedOtherLogprob(), 1e-3);
            }
        }
    });
}

// Two properties every log-probability list has by construction: it is ordered
// most-probable first, and no entry is positive, because a probability cannot
// exceed one. A caller reading only entry 0 depends on the first.
TEST_F(LogprobsTest, EntriesAreOrderedMostProbableFirstAndNeverPositive)
{
    constexpr int32_t kTopK{5};

    withEngineEmitting({5, 7}, [&](rt::LLMInferenceRuntime& runtime) {
        auto request = makeGreedyRequest("a", 2);
        request.numLogprobs = kTopK;

        rt::LLMGenerationResponse response;
        ASSERT_TRUE(runtime.handleRequest(request, response, mStream));

        // Without this the loop body never runs when no steps were reported, and
        // the ordering claim would hold vacuously.
        ASSERT_EQ(response.logprobs[0].size(), response.outputIds[0].size());
        ASSERT_FALSE(response.logprobs[0].empty());
        for (auto const& step : response.logprobs[0])
        {
            ASSERT_FALSE(step.empty());
            for (size_t k = 0; k < step.size(); ++k)
            {
                EXPECT_LE(step[k].logprob, 0.0F) << "entry " << k;
                if (k > 0)
                {
                    EXPECT_LE(step[k].logprob, step[k - 1].logprob) << "entry " << k;
                }
            }
        }
    });
}

// Sampling and log-probability extraction read the same logits by two different
// routes. Under greedy sampling both reduce to the argmax, so the token reported
// for a step and the token at the head of that step's list have to be the same
// one. A drift in either route -- a step offset, a stale row -- separates them.
//
// Given greedy sampling and a different peak token in each round
// When the sampled tokens and the log-probability lists are both read back
// Then step i's most probable entry is the token reported for step i
TEST_F(LogprobsTest, TheMostProbableEntryIsTheTokenGreedySamplingReturned)
{
    constexpr int32_t kTopK{2};
    // Distinct per round, so an off-by-one step alignment cannot pass by
    // matching a neighbouring step's token.
    std::vector<int32_t> const peaks{5, 7, 9};

    withEngineEmitting(peaks, [&](rt::LLMInferenceRuntime& runtime) {
        auto request = makeGreedyRequest("a", static_cast<int64_t>(peaks.size()));
        request.numLogprobs = kTopK;

        rt::LLMGenerationResponse response;
        ASSERT_TRUE(runtime.handleRequest(request, response, mStream));

        EXPECT_EQ(response.outputIds[0], peaks);
        ASSERT_EQ(response.logprobs[0].size(), peaks.size());
        for (size_t step = 0; step < peaks.size(); ++step)
        {
            EXPECT_EQ(response.logprobs[0][step][0].tokenId, peaks[step]) << "step " << step;
        }
    });
}

// The top-K width is bounded by the buffers assembly sized. An over-wide request
// is clamped rather than rejected, so a client that asks for more than the
// runtime supports still gets an answer.
// The other end of the range the header documents. A negative width has no
// meaningful reading, and the runtime treats it as disabled rather than
// rejecting the request or indexing a buffer with it. Pinned because it is the
// boundary a caller reaches by arithmetic -- computing a width and getting -1 --
// rather than by typing one.
//
// Given a request asking for a negative number of log-probabilities
// When it is generated
// Then it succeeds with none reported, the same as asking for zero
TEST_F(LogprobsTest, TreatANegativeWidthAsDisabledRatherThanRejectingTheRequest)
{
    withEngineEmitting({5, 7}, [&](rt::LLMInferenceRuntime& runtime) {
        auto request = makeGreedyRequest("a", 2);
        request.numLogprobs = -1;

        rt::LLMGenerationResponse response;
        ASSERT_TRUE(runtime.handleRequest(request, response, mStream));

        ASSERT_EQ(response.logprobs.size(), 1U);
        EXPECT_TRUE(response.logprobs[0].empty());
        // The request is still served; only the log-probabilities are withheld.
        EXPECT_EQ(response.outputIds[0].size(), 2U);
    });
}

TEST_F(LogprobsTest, ClampAnOverWideRequestToTheSupportedWidth)
{
    withEngineEmitting({5, 7}, [&](rt::LLMInferenceRuntime& runtime) {
        auto request = makeGreedyRequest("a", 2);
        request.numLogprobs = static_cast<int32_t>(kMaxLogprobsK) + 10;

        rt::LLMGenerationResponse response;
        ASSERT_TRUE(runtime.handleRequest(request, response, mStream));

        ASSERT_FALSE(response.logprobs[0].empty());
        EXPECT_THAT(response.logprobs[0], ::testing::Each(::testing::SizeIs(kMaxLogprobsK)));
    });
}

// Each slot reads its own row out of the staged top-K block. The two slots are
// given different peaks so a collector that read one slot's rows for the whole
// batch is visible: matching list lengths alone would not show it.
//
// Given a batch of two whose slots peak at different tokens
// When log-probabilities are reported
// Then each slot's lists are headed by its own peak
TEST_F(LogprobsTest, ReportEverySlotOfABatchFromItsOwnRow)
{
    constexpr int32_t kTopK{2};
    constexpr int32_t kFirstSlotPeak{5};
    constexpr int32_t kSecondSlotPeak{9};

    auto engine = makeEngine();
    EXPECT_CALL(*engine, execute(::testing::_)).WillRepeatedly(emit({kFirstSlotPeak, kSecondSlotPeak}));

    auto artifacts = makeVanillaArtifacts(mModelDir, std::move(engine), mStream);
    auto runtime = makeRuntime(std::move(artifacts));

    auto request = makeGreedyRequest("a", 2);
    request.requests.push_back(request.requests.front());
    request.numLogprobs = kTopK;

    rt::LLMGenerationResponse response;
    ASSERT_TRUE(runtime.handleRequest(request, response, mStream));

    ASSERT_EQ(response.logprobs.size(), 2U);
    std::array<int32_t, 2> const peaks{kFirstSlotPeak, kSecondSlotPeak};
    for (size_t slot = 0; slot < peaks.size(); ++slot)
    {
        ASSERT_FALSE(response.logprobs[slot].empty()) << "slot " << slot;
        EXPECT_EQ(response.logprobs[slot].size(), response.outputIds[slot].size()) << "slot " << slot;
        EXPECT_THAT(response.logprobs[slot], ::testing::Each(::testing::SizeIs(kTopK))) << "slot " << slot;
        for (auto const& step : response.logprobs[slot])
        {
            EXPECT_EQ(step[0].tokenId, peaks[slot]) << "slot " << slot;
        }
    }
}
} // namespace

// ---------------------------------------------------------------------------
// In-flight admission: a second sequence joins through the boundary hook while
// the first decodes, and each sequence's tokens are exactly what a serial run
// with the same engine outputs would have produced. The mock names every row's
// argmax per forward pass, so a token landing in the wrong slot -- through the
// seated prefill's swaps or the eviction that follows -- changes an assertion,
// not a probability.
// ---------------------------------------------------------------------------
TEST_F(RuntimeAssemblyTest, AdmitsASecondSequenceMidFlightAndKeepsBothTokenStreamsIntact)
{
    using ::testing::_;

    constexpr int64_t kMaxGenerateLength{4};
    constexpr int32_t kA1 = 3, kA2 = 4, kA3 = 5, kA4 = 6;
    constexpr int32_t kB1 = 7, kB2 = 8, kB3 = 9, kB4 = 10;
    constexpr int32_t kAdmittedIndexBase = 1; // A's request holds index 0

    auto engine = makeEngine();
    auto& mock = *engine;

    // Two prefill passes: A's founding one, and B's seated batch-1 pass mid-flight.
    EXPECT_CALL(mock, prepare(kPrefillProfile, _, _, _)).Times(2);
    EXPECT_CALL(mock, prepare(kDecodeProfile, _, _, _)).Times(4);
    {
        ::testing::InSequence forwardOrder;
        EXPECT_CALL(mock, execute(_)).WillOnce(emit({kA1}));      // A prefill
        EXPECT_CALL(mock, execute(_)).WillOnce(emit({kA2}));      // decode, A alone
        EXPECT_CALL(mock, execute(_)).WillOnce(emit({kB1}));      // B's seated prefill (batch 1)
        EXPECT_CALL(mock, execute(_)).WillOnce(emit({kA3, kB2})); // decode, both resident
        EXPECT_CALL(mock, execute(_)).WillOnce(emit({kA4, kB3})); // A finishes and is evicted
        EXPECT_CALL(mock, execute(_)).WillOnce(emit({kB4}));      // B, compacted to row 0, finishes
    }

    auto artifacts = makeVanillaArtifacts(mModelDir, std::move(engine), mStream);
    rt::LLMInferenceRuntime runtime{
        std::move(artifacts), mModelDir.string(), /*multimodalEngineDir=*/"", {}, std::nullopt, mStream};

    int32_t boundaryCalls = 0;
    bool admitted = false;
    std::unordered_map<int32_t, rt::BatchResult> harvested;
    auto hook = [&](rt::GenerationBoundary& batch) {
        ++boundaryCalls;
        // The second boundary sits after A's first decode step -- mid-generation by construction.
        if (boundaryCalls == 2)
        {
            rt::SlotSeed seed;
            seed.promptTokenIds = {42};
            seed.originalIndex = kAdmittedIndexBase;
            ASSERT_FALSE(admitted);
            EXPECT_EQ(batch.admitSequence(std::move(seed)), rt::AdmitDecision::kAdmitted);
            EXPECT_EQ(batch.residentCount(), 2);
            admitted = true;
        }
        for (auto& [index, result] : batch.takeCompletedAtOrAbove(kAdmittedIndexBase))
        {
            harvested.emplace(index, std::move(result));
        }
    };

    auto const request = makeGreedyRequest("a", kMaxGenerateLength);
    rt::LLMGenerationResponse response;
    ASSERT_TRUE(runtime.handleRequest(request, response, mStream, /*outputThinkerEmbeddings=*/false, hook));
    ASSERT_TRUE(admitted);

    // A's response is exactly its serial token stream, and covers only A: B's result must have
    // left through the harvest, not through the founding caller's response.
    expectResponseCoversEverySlot(response, 1);
    ASSERT_EQ(response.outputIds.size(), 1U);
    EXPECT_EQ(response.outputIds[0], (std::vector<int32_t>{kA1, kA2, kA3, kA4}));
    EXPECT_EQ(response.finishReasons[0], rt::FinishReason::kLength);

    // B's stream, token for token, through admission, two shared steps, A's eviction and its own.
    ASSERT_EQ(harvested.size(), 1U);
    auto const& resultB = harvested.at(kAdmittedIndexBase);
    EXPECT_EQ(resultB.generateLength, static_cast<int32_t>(kMaxGenerateLength));
    ASSERT_GE(resultB.tokenIds.size(), 4U);
    EXPECT_EQ(std::vector<int32_t>(resultB.tokenIds.end() - 4, resultB.tokenIds.end()),
        (std::vector<int32_t>{kB1, kB2, kB3, kB4}));
    EXPECT_EQ(resultB.terminalReason, rt::FinishReason::kLength);
}

// ---------------------------------------------------------------------------
// Shutdown ordering under leases: the one path whose failure mode is
// std::terminate. An engine owning a context-cache runtime shuts down with a
// founder decoding, a mid-flight admission holding a lease, and a request
// still queued. Every caller must get a terminal outcome, and the coordinator
// must reach its destructor quiescent -- it terminates the process otherwise,
// so this test passing IS the proof that every lease came back.
// ---------------------------------------------------------------------------
TEST_F(RuntimeAssemblyTest, ShutdownWithResidentLeasesReturnsEveryOutcomeAndEveryLease)
{
    using ::testing::_;

    auto engine = makeEngine();
    // Every forward pass emits token 3 for row 0 and lets other rows tie-break to 0; the pace
    // gives shutdown a wide mid-generation window to land in.
    ON_CALL(*engine, execute(_)).WillByDefault([this](cudaStream_t stream) {
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
        writeLogits(*mLogits, kVocabSize, {3}, stream);
        return true;
    });

    auto artifacts = makeVanillaArtifacts(mModelDir, std::move(engine), mStream);
    auto runtime = std::make_unique<rt::LLMInferenceRuntime>(std::move(artifacts), mModelDir.string(),
        /*multimodalEngineDir=*/"", std::unordered_map<std::string, std::string>{}, std::nullopt, mStream,
        rt::ContextCacheConfig{/*enabled=*/true, /*maxRecords=*/16});

    rt::scheduler::EngineConfig config;
    config.maxBatchSize = 2;
    rt::scheduler::RequestEngine requestEngine(std::move(runtime), mStream, config);

    // Generation lengths are sized to the page pool: 4 pages of 16 tokens serve two sequences of
    // (prompt + 12 + headroom) each, so the joiner's lease is grantable while the founder runs.
    // A larger ask never fails loudly -- the joiner waits on kNoCapacity forever, by design.
    // The assembly tokenizer is an empty shell, so every request carries pre-tokenized input --
    // the path admitRequest prefers anyway.
    auto const makeTokenizedRequest = [](std::vector<int32_t> tokenIds, int64_t maxGenerateLength) {
        rt::LLMGenerationRequest request = makeGreedyRequest("x", maxGenerateLength);
        request.preTokenizedInputIds = {std::move(tokenIds)};
        return request;
    };

    auto const deadline = std::chrono::steady_clock::now() + std::chrono::seconds(20);
    rt::scheduler::RequestHandle founder = requestEngine.submit(makeTokenizedRequest({41}, 12));
    while (requestEngine.resident() == 0)
    {
        ASSERT_LT(std::chrono::steady_clock::now(), deadline) << "the founder never became resident";
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    rt::scheduler::RequestHandle joiner = requestEngine.submit(makeTokenizedRequest({42}, 12));
    while (requestEngine.metrics().admittedMidFlight == 0)
    {
        if (joiner.ready())
        {
            try
            {
                joiner.get();
                FAIL() << "the joiner completed without ever being admitted mid-flight";
            }
            catch (std::exception const& error)
            {
                FAIL() << "the joiner was rejected: " << error.what();
            }
        }
        ASSERT_LT(std::chrono::steady_clock::now(), deadline)
            << "the joiner was never admitted (stallsNoCapacity=" << requestEngine.metrics().stallsNoCapacity
            << ", stallsIncompatible=" << requestEngine.metrics().stallsIncompatible
            << ", resident=" << requestEngine.resident() << ", queued=" << requestEngine.queued() << ")";
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    // The batch is full (maxBatchSize 2), so this one stays queued until shutdown abandons it --
    // unless the batch drains first, in which case it founds and is cancelled mid-run. Both are
    // legitimate shutdown paths; what matters below is that every caller gets an outcome.
    rt::scheduler::RequestHandle queued = requestEngine.submit(makeTokenizedRequest({43}, 12));

    requestEngine.shutdown(rt::scheduler::ShutdownMode::kCancel);

    // Nobody is left parked: every handle has a terminal outcome the moment shutdown returns.
    EXPECT_TRUE(founder.ready());
    EXPECT_TRUE(joiner.ready());
    EXPECT_TRUE(queued.ready());

    // Scope exit destroys the engine, then the runtime, then the coordinator, whose destructor
    // proves quiescence or terminates. Reaching the end of this test is the assertion.
}

TEST_F(RuntimeAssemblyTest, SteppedAdmissionMatchesTheFusedPathTokenForToken)
{
    using ::testing::_;

    // The same script as the fused mid-flight admission test; only the admission travels through
    // the stepper's typed admit + prefill split. Both streams must come out identical.
    constexpr int64_t kMaxGenerateLength{4};
    constexpr int32_t kA1 = 3, kA2 = 4, kA3 = 5, kA4 = 6;
    constexpr int32_t kB1 = 7, kB2 = 8, kB3 = 9, kB4 = 10;
    constexpr int32_t kAdmittedIndexBase = 1;

    auto engine = makeEngine();
    auto& mock = *engine;
    EXPECT_CALL(mock, prepare(kPrefillProfile, _, _, _)).Times(2);
    EXPECT_CALL(mock, prepare(kDecodeProfile, _, _, _)).Times(4);
    {
        ::testing::InSequence forwardOrder;
        EXPECT_CALL(mock, execute(_)).WillOnce(emit({kA1}));
        EXPECT_CALL(mock, execute(_)).WillOnce(emit({kA2}));
        EXPECT_CALL(mock, execute(_)).WillOnce(emit({kB1}));
        EXPECT_CALL(mock, execute(_)).WillOnce(emit({kA3, kB2}));
        EXPECT_CALL(mock, execute(_)).WillOnce(emit({kA4, kB3}));
        EXPECT_CALL(mock, execute(_)).WillOnce(emit({kB4}));
    }

    auto artifacts = makeVanillaArtifacts(mModelDir, std::move(engine), mStream);
    rt::LLMInferenceRuntime runtime{
        std::move(artifacts), mModelDir.string(), /*multimodalEngineDir=*/"", {}, std::nullopt, mStream};

    int32_t boundaryCalls = 0;
    std::optional<rt::RuntimeStepper> stepper;
    std::unordered_map<int32_t, rt::BatchResult> harvested;
    auto hook = [&](rt::GenerationBoundary& batch) {
        ++boundaryCalls;
        auto& session = static_cast<rt::LLMRankRuntime::GenerationSession&>(batch);
        if (!stepper.has_value())
        {
            stepper.emplace(session);
        }
        if (boundaryCalls == 2)
        {
            rt::AdmissionIntent intent;
            intent.requestId = 102;
            intent.seed.promptTokenIds = {42};
            intent.seed.originalIndex = kAdmittedIndexBase;
            rt::AdmissionResult const admitted = stepper->admit(std::move(intent));
            ASSERT_EQ(admitted.status, rt::AdmissionResult::Status::kAdmitted);
            EXPECT_EQ(admitted.ref.slot, 1);
            EXPECT_EQ(admitted.ref.epoch, 1U);

            rt::StepResult const seated = stepper->prefill({admitted.ref});
            ASSERT_TRUE(seated.ok);
            // B's delta is its lookahead token. A also reports here: the stepper was constructed
            // mid-request, so A's tokens outstanding at construction ride the first operation --
            // watermark semantics, exercised on purpose by this transitional construction point.
            ASSERT_EQ(seated.deltas.size(), 2U);
            std::unordered_map<int32_t, std::vector<int32_t>> deltasBySlot;
            for (auto const& [ref, delta] : seated.deltas)
            {
                deltasBySlot.emplace(ref.slot, delta.tokenIds);
            }
            EXPECT_EQ(deltasBySlot.at(0), (std::vector<int32_t>{kA1, kA2}));
            EXPECT_EQ(deltasBySlot.at(admitted.ref.slot), (std::vector<int32_t>{kB1}));
            EXPECT_TRUE(seated.finished.empty());
            EXPECT_FALSE(seated.publishedPrefix); // no context cache in this deployment
            EXPECT_EQ(batch.residentCount(), 2);
        }
        for (auto& [index, result] : batch.takeCompletedAtOrAbove(kAdmittedIndexBase))
        {
            harvested.emplace(index, std::move(result));
        }
    };

    auto const request = makeGreedyRequest("a", kMaxGenerateLength);
    rt::LLMGenerationResponse response;
    ASSERT_TRUE(runtime.handleRequest(request, response, mStream, /*outputThinkerEmbeddings=*/false, hook));

    expectResponseCoversEverySlot(response, 1);
    ASSERT_EQ(response.outputIds.size(), 1U);
    EXPECT_EQ(response.outputIds[0], (std::vector<int32_t>{kA1, kA2, kA3, kA4}));

    ASSERT_EQ(harvested.size(), 1U);
    auto const& resultB = harvested.at(kAdmittedIndexBase);
    ASSERT_GE(resultB.tokenIds.size(), 4U);
    EXPECT_EQ(std::vector<int32_t>(resultB.tokenIds.end() - 4, resultB.tokenIds.end()),
        (std::vector<int32_t>{kB1, kB2, kB3, kB4}));
    EXPECT_EQ(resultB.terminalReason, rt::FinishReason::kLength);
}

TEST_F(RuntimeAssemblyTest, StepperKeepsRefsStableAcrossEviction)
{
    using ::testing::_;

    // A stepper-driven decode carries the batch across A's eviction: the StepResult must file A
    // under finished and credit B's token to its ref -- which stays identical across the eviction,
    // because refs are stable resident identities and execution-row compaction is runtime-private.
    constexpr int64_t kMaxGenerateLength{3};
    constexpr int32_t kA1 = 3, kA2 = 4, kA3 = 5;
    constexpr int32_t kB1 = 7, kB2 = 8, kB3 = 9;
    constexpr int32_t kAdmittedIndexBase = 1;

    auto engine = makeEngine();
    auto& mock = *engine;
    EXPECT_CALL(mock, prepare(kPrefillProfile, _, _, _)).Times(2);
    EXPECT_CALL(mock, prepare(kDecodeProfile, _, _, _)).Times(3);
    {
        ::testing::InSequence forwardOrder;
        EXPECT_CALL(mock, execute(_)).WillOnce(emit({kA1}));      // A prefill
        EXPECT_CALL(mock, execute(_)).WillOnce(emit({kA2}));      // decode, A alone
        EXPECT_CALL(mock, execute(_)).WillOnce(emit({kB1}));      // B's seated prefill
        EXPECT_CALL(mock, execute(_)).WillOnce(emit({kA3, kB2})); // stepper decode: A finishes
        EXPECT_CALL(mock, execute(_)).WillOnce(emit({kB3}));      // loop decode: B finishes
    }

    auto artifacts = makeVanillaArtifacts(mModelDir, std::move(engine), mStream);
    rt::LLMInferenceRuntime runtime{
        std::move(artifacts), mModelDir.string(), /*multimodalEngineDir=*/"", {}, std::nullopt, mStream};

    int32_t boundaryCalls = 0;
    std::optional<rt::RuntimeStepper> stepper;
    bool stepped = false;
    std::unordered_map<int32_t, rt::BatchResult> harvested;
    auto hook = [&](rt::GenerationBoundary& batch) {
        ++boundaryCalls;
        auto& session = static_cast<rt::LLMRankRuntime::GenerationSession&>(batch);
        if (!stepper.has_value())
        {
            stepper.emplace(session);
        }
        if (boundaryCalls == 2)
        {
            rt::AdmissionIntent intent;
            intent.requestId = 102;
            intent.seed.promptTokenIds = {42};
            intent.seed.originalIndex = kAdmittedIndexBase;
            rt::AdmissionResult const admitted = stepper->admit(std::move(intent));
            ASSERT_EQ(admitted.status, rt::AdmissionResult::Status::kAdmitted);
            rt::ResidentRef const refB = admitted.ref;
            ASSERT_TRUE(stepper->prefill({refB}).ok);

            // One typed decode across A's finish: this is the extra forward pass the mock script
            // accounts for at position four.
            rt::StepResult const step = stepper->decode({stepper->residents()});
            ASSERT_TRUE(step.ok);

            // Deltas: A's final token is inside its finished snapshot, not a delta; B's token is
            // credited to the ref the caller held when it built the view.
            ASSERT_EQ(step.deltas.size(), 1U);
            EXPECT_EQ(step.deltas.front().first, refB);
            EXPECT_EQ(step.deltas.front().second.tokenIds, (std::vector<int32_t>{kB2}));

            ASSERT_EQ(step.finished.size(), 1U);
            EXPECT_EQ(step.finished.front().first, (rt::ResidentRef{0, 1}));
            auto const& resultA = step.finished.front().second;
            ASSERT_GE(resultA.tokenIds.size(), 3U);
            EXPECT_EQ(std::vector<int32_t>(resultA.tokenIds.end() - 3, resultA.tokenIds.end()),
                (std::vector<int32_t>{kA1, kA2, kA3}));
            EXPECT_EQ(resultA.terminalReason, rt::FinishReason::kLength);

            // Middle/head retirement: the survivor's execution row changed underneath, but its
            // resident identity -- the only identity the scheduler ever saw -- did not.
            ASSERT_EQ(stepper->residents().size(), 1U);
            EXPECT_EQ(stepper->residents().front(), refB);
            stepped = true;
        }
        for (auto& [index, result] : batch.takeCompletedAtOrAbove(kAdmittedIndexBase))
        {
            harvested.emplace(index, std::move(result));
        }
    };

    auto const request = makeGreedyRequest("a", kMaxGenerateLength);
    rt::LLMGenerationResponse response;
    ASSERT_TRUE(runtime.handleRequest(request, response, mStream, /*outputThinkerEmbeddings=*/false, hook));
    ASSERT_TRUE(stepped);

    // A's result left through the stepper's StepResult, so the founding response is empty of it;
    // B still finishes through the loop's own advance and leaves through the harvest.
    ASSERT_EQ(harvested.size(), 1U);
    EXPECT_EQ(harvested.at(kAdmittedIndexBase).terminalReason, rt::FinishReason::kLength);
}

TEST_F(RuntimeAssemblyTest, ASteppedRequestRunsEndToEndWithoutTheHook)
{
    using ::testing::_;

    // The canonical A/B interleave, driven entirely through the stepped control plane: no
    // handleRequest, no GenerationBoundaryHook. Every tick is an explicit typed operation and
    // every outcome leaves through a StepResult.
    constexpr int64_t kMaxGenerateLength{4};
    constexpr int32_t kA1 = 3, kA2 = 4, kA3 = 5, kA4 = 6;
    constexpr int32_t kB1 = 7, kB2 = 8, kB3 = 9, kB4 = 10;
    constexpr int32_t kC1 = 11, kC2 = 12, kC3 = 13, kC4 = 14;
    constexpr int32_t kIndexB = 1;
    constexpr int32_t kIndexC = 2;

    auto engine = makeEngine();
    auto& mock = *engine;
    EXPECT_CALL(mock, prepare(kPrefillProfile, _, _, _)).Times(3);
    EXPECT_CALL(mock, prepare(kDecodeProfile, _, _, _)).Times(6);
    {
        ::testing::InSequence forwardOrder;
        EXPECT_CALL(mock, execute(_)).WillOnce(emit({kA1}));      // founding prefill (inside begin)
        EXPECT_CALL(mock, execute(_)).WillOnce(emit({kA2}));      // decode, A alone
        EXPECT_CALL(mock, execute(_)).WillOnce(emit({kB1}));      // B's seated prefill tick
        EXPECT_CALL(mock, execute(_)).WillOnce(emit({kA3, kB2})); // decode, both resident
        EXPECT_CALL(mock, execute(_)).WillOnce(emit({kA4, kB3})); // A finishes and is evicted
        EXPECT_CALL(mock, execute(_)).WillOnce(emit({kC1}));      // C's seated prefill tick
        EXPECT_CALL(mock, execute(_)).WillOnce(emit({kB4, kC2})); // B finishes and is evicted
        EXPECT_CALL(mock, execute(_)).WillOnce(emit({kC3}));
        EXPECT_CALL(mock, execute(_)).WillOnce(emit({kC4})); // C finishes
    }

    auto artifacts = makeVanillaArtifacts(mModelDir, std::move(engine), mStream);
    rt::LLMInferenceRuntime runtime{
        std::move(artifacts), mModelDir.string(), /*multimodalEngineDir=*/"", {}, std::nullopt, mStream};
    ASSERT_TRUE(runtime.supportsSteppedExecution());

    EXPECT_THROW(
        runtime.beginStepped(makeGreedyRequest("a", kMaxGenerateLength), /*requestId=*/0, mStream), std::runtime_error);
    auto stepped = runtime.beginStepped(makeGreedyRequest("a", kMaxGenerateLength), /*requestId=*/101, mStream);
    ASSERT_NE(stepped, nullptr);
    rt::SteppedExecution& stepper = *stepped;

    std::unordered_map<int32_t, rt::LLMGenerationResponse> outcomes; // keyed by founding order
    auto commit = [&](rt::StepResult const& result, std::vector<std::string> const& stops) {
        ASSERT_TRUE(result.ok);
        for (auto const& [ref, snapshot] : result.finished)
        {
            outcomes.emplace(static_cast<int32_t>(outcomes.size()), stepped->materialize(snapshot, stops));
            (void) ref;
        }
    };

    // Tick 1: the founding prefill's post-pass primes A's first token.
    rt::StepResult const founding = stepper.prefill({stepper.residents().front()});
    ASSERT_TRUE(founding.ok);
    ASSERT_EQ(founding.deltas.size(), 1U);
    EXPECT_EQ(founding.deltas.front().second.tokenIds, (std::vector<int32_t>{kA1}));

    // Tick 2: decode, A alone.
    commit(stepper.decode({stepper.residents()}), {});

    // Tick 3: admit B (logical only), then its seated prefill tick.
    rt::LLMGenerationRequest requestB = makeGreedyRequest("b", kMaxGenerateLength);
    requestB.preTokenizedInputIds = {{43}};
    EXPECT_THROW(stepper.admit(requestB, kIndexB, /*requestId=*/0), std::runtime_error);
    rt::AdmissionResult const admitted = stepper.admit(requestB, kIndexB, /*requestId=*/102);
    ASSERT_EQ(admitted.status, rt::AdmissionResult::Status::kAdmitted);
    rt::StepResult const seated = stepper.prefill({admitted.ref});
    ASSERT_TRUE(seated.ok);
    ASSERT_EQ(seated.deltas.size(), 1U);
    EXPECT_EQ(seated.deltas.front().second.tokenIds, (std::vector<int32_t>{kB1}));

    // Decode until A retires; its finished ref releases resident slot 0.
    while (outcomes.empty())
    {
        commit(stepper.decode({stepper.residents()}), {});
    }

    // Tail-retirement reuse, the aliasing case from review: C is admitted after A released its
    // resident slot, so C reuses slot 0 with a bumped epoch and A's stale ref cannot name C.
    rt::LLMGenerationRequest requestC = makeGreedyRequest("c", kMaxGenerateLength);
    requestC.preTokenizedInputIds = {{44}};
    rt::AdmissionResult const admittedC = stepper.admit(requestC, kIndexC, /*requestId=*/103);
    ASSERT_EQ(admittedC.status, rt::AdmissionResult::Status::kAdmitted);
    EXPECT_EQ(admittedC.ref, (rt::ResidentRef{0, 2}));
    EXPECT_FALSE(admittedC.ref == (rt::ResidentRef{0, 1}));
    rt::StepResult const seatedC = stepper.prefill({admittedC.ref});
    ASSERT_TRUE(seatedC.ok);
    ASSERT_EQ(seatedC.deltas.size(), 1U);
    EXPECT_EQ(seatedC.deltas.front().second.tokenIds, (std::vector<int32_t>{kC1}));

    // Drain the rest; every terminal leaves through a StepResult.
    while (!stepper.residents().empty())
    {
        commit(stepper.decode({stepper.residents()}), {});
    }
    ASSERT_EQ(outcomes.size(), 3U);
    EXPECT_EQ(outcomes.at(0).outputIds.front(), (std::vector<int32_t>{kA1, kA2, kA3, kA4}));
    EXPECT_EQ(outcomes.at(0).finishReasons.front(), rt::FinishReason::kLength);
    EXPECT_EQ(outcomes.at(1).outputIds.front(), (std::vector<int32_t>{kB1, kB2, kB3, kB4}));
    EXPECT_EQ(outcomes.at(1).finishReasons.front(), rt::FinishReason::kLength);
    EXPECT_EQ(outcomes.at(2).outputIds.front(), (std::vector<int32_t>{kC1, kC2, kC3, kC4}));
    EXPECT_EQ(outcomes.at(2).finishReasons.front(), rt::FinishReason::kLength);

    rt::LLMGenerationResponse response;
    EXPECT_TRUE(stepped->finish(response));
}

TEST_F(RuntimeAssemblyTest, TheEngineDrivesTheSteppedPlaneEndToEnd)
{
    using ::testing::_;

    // The full stack, no hook anywhere: RequestEngine's actor owns the loop, the runtime executes
    // typed steps, and both callers get their serial token streams back through handles.
    constexpr int64_t kMaxGenerateLength{4};
    constexpr int32_t kA1 = 3, kA2 = 4, kA3 = 5, kA4 = 6;
    constexpr int32_t kB1 = 7, kB2 = 8, kB3 = 9, kB4 = 10;

    auto engine = makeEngine();
    auto& mock = *engine;
    std::atomic<bool> aSecondDecodeStarted{false};
    std::atomic<bool> bQueued{false};
    EXPECT_CALL(mock, prepare(kPrefillProfile, _, _, _)).Times(2);
    EXPECT_CALL(mock, prepare(kDecodeProfile, _, _, _)).Times(4);
    {
        ::testing::InSequence forwardOrder;
        EXPECT_CALL(mock, execute(_)).WillOnce(emit({kA1})); // founding prefill
        // Two-way handshake for a deterministic interleave: B is submitted only once A's first
        // decode is already executing (so no earlier tick can admit it), and that decode holds
        // until B is queued (so the very next tick does).
        EXPECT_CALL(mock, execute(_))
            .WillOnce([this, &aSecondDecodeStarted, &bQueued, action = emit({kA2})](cudaStream_t stream) {
                aSecondDecodeStarted.store(true, std::memory_order_release);
                while (!bQueued.load(std::memory_order_acquire))
                {
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                }
                return action(stream);
            });
        EXPECT_CALL(mock, execute(_)).WillOnce(emit({kB1})); // B's seated prefill tick
        EXPECT_CALL(mock, execute(_)).WillOnce(emit({kA3, kB2}));
        EXPECT_CALL(mock, execute(_)).WillOnce(emit({kA4, kB3})); // A finishes, evicted
        EXPECT_CALL(mock, execute(_)).WillOnce(emit({kB4}));      // B finishes
    }

    auto artifacts = makeVanillaArtifacts(mModelDir, std::move(engine), mStream);
    auto runtime = std::make_unique<rt::LLMInferenceRuntime>(std::move(artifacts), mModelDir.string(),
        /*multimodalEngineDir=*/"", std::unordered_map<std::string, std::string>{}, std::nullopt, mStream);
    ASSERT_TRUE(runtime->supportsSteppedExecution());

    rt::scheduler::EngineConfig config;
    config.maxBatchSize = 2;
    rt::scheduler::RequestEngine requestEngine(std::move(runtime), mStream, config);

    rt::scheduler::RequestHandle handleA = requestEngine.submit(makeGreedyRequest("a", kMaxGenerateLength));
    while (!aSecondDecodeStarted.load(std::memory_order_acquire))
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    rt::LLMGenerationRequest requestB = makeGreedyRequest("b", kMaxGenerateLength);
    requestB.preTokenizedInputIds = {{43}};
    rt::scheduler::RequestHandle handleB = requestEngine.submit(std::move(requestB));
    bQueued.store(true, std::memory_order_release);

    rt::LLMGenerationResponse const responseA = handleA.get();
    rt::LLMGenerationResponse const responseB = handleB.get();
    ASSERT_EQ(responseA.outputIds.size(), 1U);
    EXPECT_EQ(responseA.outputIds.front(), (std::vector<int32_t>{kA1, kA2, kA3, kA4}));
    EXPECT_EQ(responseA.finishReasons.front(), rt::FinishReason::kLength);
    ASSERT_EQ(responseB.outputIds.size(), 1U);
    EXPECT_EQ(responseB.outputIds.front(), (std::vector<int32_t>{kB1, kB2, kB3, kB4}));
    EXPECT_EQ(responseB.finishReasons.front(), rt::FinishReason::kLength);

    requestEngine.shutdown(rt::scheduler::ShutdownMode::kDrain);
    auto const metrics = requestEngine.metrics();
    EXPECT_EQ(metrics.completed, 2U);
    EXPECT_EQ(metrics.admittedMidFlight, 1U) << "B must have joined mid-flight through the stepped plane";
    EXPECT_EQ(metrics.failed, 0U);
}

TEST_F(RuntimeAssemblyTest, ASteppedDecodeThrowFailsTheBatchAndTheActorSurvives)
{
    using ::testing::_;

    // A CUDA failure inside a stepped decode leaves the runtime as an exception. The engine must
    // turn it into a terminal outcome for every resident and stay alive for the next request; an
    // unwind out of the actor thread would be std::terminate.
    constexpr int64_t kMaxGenerateLength{3};
    constexpr int32_t kA1 = 3, kC1 = 11, kC2 = 12, kC3 = 13;

    auto engine = makeEngine();
    auto& mock = *engine;
    EXPECT_CALL(mock, prepare(_, _, _, _)).Times(::testing::AnyNumber());
    {
        ::testing::InSequence forwardOrder;
        EXPECT_CALL(mock, execute(_)).WillOnce(emit({kA1})); // A's founding prefill
        EXPECT_CALL(mock, execute(_)).WillOnce([action = emit({kA1})](cudaStream_t stream) {
            throw std::runtime_error("device lost mid-decode");
            return action(stream);
        });
        EXPECT_CALL(mock, execute(_)).WillOnce(emit({kC1})); // C founds the next batch
        EXPECT_CALL(mock, execute(_)).WillOnce(emit({kC2}));
        EXPECT_CALL(mock, execute(_)).WillOnce(emit({kC3}));
    }

    auto artifacts = makeVanillaArtifacts(mModelDir, std::move(engine), mStream);
    auto runtime = std::make_unique<rt::LLMInferenceRuntime>(std::move(artifacts), mModelDir.string(),
        /*multimodalEngineDir=*/"", std::unordered_map<std::string, std::string>{}, std::nullopt, mStream);
    ASSERT_TRUE(runtime->supportsSteppedExecution());

    rt::scheduler::EngineConfig config;
    config.maxBatchSize = 2;
    rt::scheduler::RequestEngine requestEngine(std::move(runtime), mStream, config);

    rt::scheduler::RequestHandle handleA = requestEngine.submit(makeGreedyRequest("a", kMaxGenerateLength));
    EXPECT_THROW(handleA.get(), std::runtime_error);

    rt::LLMGenerationRequest requestC = makeGreedyRequest("b", kMaxGenerateLength);
    requestC.preTokenizedInputIds = {{43}};
    rt::scheduler::RequestHandle handleC = requestEngine.submit(std::move(requestC));
    rt::LLMGenerationResponse const responseC = handleC.get();
    ASSERT_EQ(responseC.outputIds.size(), 1U);
    EXPECT_EQ(responseC.outputIds.front(), (std::vector<int32_t>{kC1, kC2, kC3}));

    requestEngine.shutdown(rt::scheduler::ShutdownMode::kDrain);
    auto const metrics = requestEngine.metrics();
    EXPECT_EQ(metrics.failed, 1U);
    EXPECT_EQ(metrics.completed, 1U);
}
