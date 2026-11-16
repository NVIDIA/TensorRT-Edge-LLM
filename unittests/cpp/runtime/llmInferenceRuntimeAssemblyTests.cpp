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

#include "fakeEngineExecutor.h"
#include "substituteEngine.h"

#include "chatTemplate/chatTemplate.h"

#include "runtime/decoding/decodingStrategy.h"
#include "runtime/decoding/requestStableRng.h"
#include "runtime/llmInferenceRuntime.h"
#include "runtime/llmRankRuntime.h"
#include "runtime/runtimeSession.h"
#include "sampler/sampling.h"
#include "scheduler/requestEngine.h"
#include <cmath>
#include <optional>
#include <set>

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

// Given an ordinary executor refusal with a healthy CUDA stream
// When the request fails and a valid request follows
// Then the runtime remains available and executes the later request.
TEST_F(RuntimeAssemblyTest, RecoverableFailureLeavesRuntimeReusable)
{
    auto engine = makeEngine();
    EXPECT_CALL(*engine, execute(::testing::_)).WillOnce(::testing::Return(false)).WillOnce(emit({3}));
    auto runtime = makeRuntime(makeVanillaArtifacts(mModelDir, std::move(engine), mStream));
    auto request = makeGreedyRequest("a", 1);
    rt::LLMGenerationResponse response;
    EXPECT_FALSE(runtime.handleRequest(request, response, mStream));
    EXPECT_TRUE(runtime.unavailableReason().empty());
    ASSERT_TRUE(runtime.handleRequest(request, response, mStream));
    EXPECT_THAT(response.outputIds, ::testing::ElementsAre(::testing::ElementsAre(3)));
}

class RuntimeFailureDeathTest : public RuntimeAssemblyTest, public ::testing::WithParamInterface<bool>
{
};

// Given a request or graph capture whose executor launches an illegal GPU access in an isolated process
// When failed-request cleanup drains that stream
// Then the original CUDA fault is retained and another request is rejected without entering CUDA execution.
TEST_P(RuntimeFailureDeathTest, FatalCudaFailurePermanentlyRejectsRequests)
{
    ::testing::GTEST_FLAG(death_test_style) = "threadsafe";
    EXPECT_EXIT(
        {
            CUmodule module{};
            CUfunction kernel{};
            char const* ptx
                = ".version 7.0\n.target sm_80\n.address_size 64\n"
                  ".visible .entry fault() { .reg .b64 addr; mov.u64 addr, 0; "
                  "st.global.u32 [addr], 1; ret; }";
            CUDA_DRIVER_CHECK(cuModuleLoadData(&module, ptx));
            CUDA_DRIVER_CHECK(cuModuleGetFunction(&kernel, module, "fault"));
            auto engine = makeEngine();
            auto fault = [&](cudaStream_t stream) {
                CUDA_DRIVER_CHECK(cuLaunchKernel(kernel, 1, 1, 1, 1, 1, 1, 0, stream, nullptr, nullptr));
                return false;
            };
            if (GetParam())
            {
                EXPECT_CALL(*engine, captureGraph(::testing::_)).WillOnce(fault);
            }
            else
            {
                EXPECT_CALL(*engine, execute(::testing::_)).WillOnce(fault);
            }
            auto runtime = makeRuntime(makeVanillaArtifacts(mModelDir, std::move(engine), mStream));
            auto request = makeGreedyRequest("a", 1);
            rt::LLMGenerationResponse response;
            EXPECT_FALSE(GetParam() ? runtime.captureDecodingCUDAGraph(mStream)
                                    : runtime.handleRequest(request, response, mStream));
            auto const reason = runtime.unavailableReason();
            EXPECT_THAT(reason, ::testing::HasSubstr("cudaErrorIllegalAddress (700)"));
            EXPECT_FALSE(runtime.handleRequest(request, response, mStream));
            EXPECT_FALSE(runtime.captureDecodingCUDAGraph(mStream));
            EXPECT_EQ(runtime.unavailableReason(), reason);
            std::_Exit(::testing::Test::HasFailure() ? 1 : 0);
        },
        ::testing::ExitedWithCode(0), "");
}

INSTANTIATE_TEST_SUITE_P(RequestAndCapture, RuntimeFailureDeathTest, ::testing::Bool());

// Given an executor that refuses graph capture without a CUDA fault
// When capture fails and a request follows
// Then synchronous inference still succeeds and the runtime remains available.
TEST_F(RuntimeAssemblyTest, RecoverableCaptureFailureLeavesRuntimeReusable)
{
    auto engine = makeEngine();
    EXPECT_CALL(*engine, captureGraph(::testing::_)).WillRepeatedly(::testing::Return(false));
    auto runtime = makeRuntime(makeVanillaArtifacts(mModelDir, std::move(engine), mStream));
    EXPECT_FALSE(runtime.captureDecodingCUDAGraph(mStream));
    EXPECT_TRUE(runtime.unavailableReason().empty());
    rt::LLMGenerationResponse response;
    ASSERT_TRUE(runtime.handleRequest(makeGreedyRequest("a", 1), response, mStream));
    EXPECT_EQ(response.outputIds[0].size(), 1U);
}

TEST_F(RuntimeAssemblyTest, InjectedArtifactsValidateDeploymentPlePolicy)
{
    auto artifacts = makeVanillaArtifacts(mModelDir, makeEngine(), mStream);
    rt::DeploymentOptions options{};
    options.engineDir = mModelDir.string();
    options.pleConfig = {rt::PleResidency::kCACHED, 16};

    std::string reason;
    try
    {
        rt::LLMInferenceRuntime runtime{std::move(artifacts), std::move(options), mStream};
    }
    catch (std::exception const& error)
    {
        reason = error.what();
    }
    EXPECT_THAT(reason, ::testing::HasSubstr("Cached PLE currently requires a PLE-enabled engine"));
}

TEST_F(RuntimeAssemblyTest, RejectsConflictingInjectedSourcesBeforeConsumingArtifacts)
{
    auto artifacts = makeVanillaArtifacts(mModelDir, makeEngine(), mStream);
    artifacts.checkpointDir = mModelDir / "loaded-base";
    artifacts.draftCheckpointDir = mModelDir / "loaded-draft";
    artifacts.deployment.runtimeCachePath = mModelDir / "loaded-cache";
    auto const* executor = artifacts.baseExecutor.get();
    for (char const* field : {"checkpointDir", "draftCheckpointDir", "runtimeCachePath"})
    {
        rt::DeploymentOptions options{};
        options.engineDir = mModelDir.string();
        if (std::string(field) == "checkpointDir")
        {
            options.checkpointDir = (mModelDir / "different-base").string();
        }
        else if (std::string(field) == "draftCheckpointDir")
        {
            options.draftCheckpointDir = (mModelDir / "different-draft").string();
        }
        else
        {
            options.runtimeCachePath = mModelDir / "different-cache";
        }
        std::string reason;
        try
        {
            rt::LLMInferenceRuntime runtime{std::move(artifacts), std::move(options), mStream};
        }
        catch (std::exception const& error)
        {
            reason = error.what();
        }
        EXPECT_THAT(reason, ::testing::HasSubstr(std::string("Injected ") + field + " conflicts"));
        EXPECT_EQ(artifacts.baseExecutor.get(), executor);
        EXPECT_NE(artifacts.tokenizer, nullptr);
    }
}

TEST_F(RuntimeAssemblyTest, AcceptsMatchingInjectedSourcesAndChangedRuntimePolicy)
{
    auto artifacts = makeVanillaArtifacts(mModelDir, makeEngine(), mStream);
    artifacts.checkpointDir = mModelDir;
    rt::DeploymentOptions options{};
    options.engineDir = mModelDir.string();
    options.checkpointDir = (mModelDir / ".").string();
    options.contextCacheConfig = {/*enabled=*/true, /*maxRecords=*/4};
    rt::LLMInferenceRuntime runtime{std::move(artifacts), std::move(options), mStream};
    EXPECT_TRUE(runtime.getContextCacheMetrics().has_value());
    rt::LLMGenerationResponse response;
    ASSERT_TRUE(runtime.handleRequest(makeGreedyRequest("a", 2), response, mStream));
    EXPECT_EQ(response.outputIds[0].size(), 2U);
}

TEST_F(RuntimeAssemblyTest, RejectsDraftingOptionsForInjectedVanillaArtifacts)
{
    auto artifacts = makeVanillaArtifacts(mModelDir, makeEngine(), mStream);
    auto const* executor = artifacts.baseExecutor.get();
    rt::DeploymentOptions options{};
    options.engineDir = mModelDir.string();
    options.draftingConfig.emplace();
    options.draftingConfig->draftTopK = 1;
    options.draftingConfig->draftingStep = 1;
    options.draftingConfig->verifyTokens = 2;
    EXPECT_THROW((rt::LLMInferenceRuntime{std::move(artifacts), std::move(options), mStream}), std::runtime_error);
    EXPECT_EQ(artifacts.baseExecutor.get(), executor);
}

TEST(ModelArtifactsOptionsTest, PreservesDocumentedBlockDraftingDefaults)
{
    rt::ModelArtifacts artifacts;
    artifacts.deployment.base.specDecodeType = rt::SpecDecodeMode::kDFlash;
    artifacts.deployment.draft.emplace();
    artifacts.deployment.draft->specDraftBlockSize = 8;
    artifacts.deployment.specConfig.emplace();
    auto& spec = *artifacts.deployment.specConfig;
    spec.draftTopK = 1;
    spec.proposalTokenCount = 7;
    spec.verifyTokenCount = 8;
    rt::DeploymentOptions options{};
    options.draftingConfig.emplace();
    options.draftingConfig->draftTopK = 1;
    EXPECT_NO_THROW(artifacts.validateDeploymentOptions(options));
    options.draftingConfig->proposalTokens = 3;
    EXPECT_THROW(artifacts.validateDeploymentOptions(options), std::runtime_error);
}

TEST(ModelArtifactsOptionsTest, PreservesDsparkDefaultProposalLength)
{
    rt::ModelArtifacts artifacts;
    artifacts.deployment.base.specDecodeType = rt::SpecDecodeMode::kDSpark;
    artifacts.deployment.base.maxSpecVerifyTokens = 8;
    artifacts.deployment.draft.emplace();
    artifacts.deployment.specConfig.emplace();
    auto& spec = *artifacts.deployment.specConfig;
    spec.draftTopK = 1;
    spec.proposalTokenCount = 7;
    spec.verifyTokenCount = 8;
    artifacts.deployment.draft->specDraftBlockSize = 7;
    spec.dsparkSchedulerMode = rt::DSparkSchedulerMode::kThreshold;
    spec.dsparkMaxProposalLen = 7;
    spec.dsparkW2Backend = rt::DSparkW2Backend::kFp16;
    rt::DeploymentOptions options{};
    options.draftingConfig.emplace();
    options.draftingConfig->draftTopK = 1;
    options.draftingConfig->proposalTokens = 7;
    options.draftingConfig->verifyTokens = 8;
    options.draftingConfig->dsparkSchedulerMode = rt::DSparkSchedulerMode::kThreshold;
    options.draftingConfig->dsparkW2Backend = rt::DSparkW2Backend::kFp16;
    options.contextCacheConfig.enabled = true;
    EXPECT_NO_THROW(artifacts.validateDeploymentOptions(options));
    options.draftingConfig->dsparkMaxProposalLen = 6;
    EXPECT_THROW(artifacts.validateDeploymentOptions(options), std::runtime_error);
    options.draftingConfig->dsparkMaxProposalLen = 7;
    options.draftingConfig->dsparkW2Backend = rt::DSparkW2Backend::kFp8;
    EXPECT_THROW(artifacts.validateDeploymentOptions(options), std::runtime_error);
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
    // How many tokens came back, not which: this test is about the shape of the forward schedule, and naming the
    // token would tie it to whatever a row nobody configured happens to decode to.
    EXPECT_THAT(response.outputIds[0], SizeIs(kMaxGenerateLength));
    EXPECT_EQ(response.finishReasons[0], rt::FinishReason::kLength);
}

TEST_F(RuntimeAssemblyTest, FlushesTerminalTokensBelowTheStreamInterval)
{
    // Given a final token below the stream interval, when prefill or decode reaches the length limit,
    // then the stream includes every generated token before it closes.
    constexpr int32_t kTOKEN_ID = 3;
    for (int64_t const maxGenerateLength : {1, 3})
    {
        SCOPED_TRACE(maxGenerateLength);
        auto engine = makeEngine();
        EXPECT_CALL(*engine, execute(::testing::_)).WillRepeatedly(emit({kTOKEN_ID}));
        auto runtime = makeRuntime(makeVanillaArtifacts(mModelDir, std::move(engine), mStream));
        auto request = makeGreedyRequest("a", maxGenerateLength);
        auto channel = rt::StreamChannel::create();
        channel->setStreamInterval(2);
        request.streamChannels.push_back(channel);

        rt::LLMGenerationResponse response;
        ASSERT_TRUE(runtime.handleRequest(request, response, mStream));
        std::vector<int32_t> streamedIds;
        channel->consume([&](rt::StreamChunk&& chunk) {
            streamedIds.insert(streamedIds.end(), chunk.tokenIds.begin(), chunk.tokenIds.end());
        });
        EXPECT_EQ(streamedIds, std::vector<int32_t>(maxGenerateLength, kTOKEN_ID));
    }
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
    // The claim is that the pre-tokenized ids were the prompt, which the reported count carries. One token came
    // back; which one is whatever an unconfigured row decodes to and is no part of it.
    EXPECT_EQ(response.inputTokenCounts[0], 1);
    EXPECT_THAT(response.outputIds[0], ::testing::SizeIs(1));
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

TEST_F(RuntimeAssemblyTest, IgnoresEosUntilTheRequestLengthLimit)
{
    using ::testing::_;

    constexpr int64_t kMaxGenerateLength{3};
    auto engine = makeEngine();
    auto& mock = *engine;

    auto artifacts = makeVanillaArtifacts(mModelDir, std::move(engine), mStream);
    auto const eosId = static_cast<int32_t>(artifacts.tokenizer->getEosId());
    EXPECT_CALL(mock, execute(_)).Times(kMaxGenerateLength).WillRepeatedly(emit({eosId}));

    rt::LLMInferenceRuntime runtime{
        std::move(artifacts), mModelDir.string(), /*multimodalEngineDir=*/"", {}, std::nullopt, mStream};

    auto request = makeGreedyRequest("a", kMaxGenerateLength);
    request.ignoreEos = true;
    rt::LLMGenerationResponse response;
    ASSERT_TRUE(runtime.handleRequest(request, response, mStream));

    expectResponseCoversEverySlot(response, 1);
    EXPECT_THAT(response.outputIds[0], ::testing::AllOf(::testing::SizeIs(kMaxGenerateLength), ::testing::Each(eosId)));
    EXPECT_EQ(response.finishReasons[0], rt::FinishReason::kLength);
}

// Batching is a scheduling decision, so nothing about it may reach the answer. The failure this guards against --
// a sequence reading the page table row, KV pages or state row belonging to its neighbour -- is invisible in a
// batch whose slots are interchangeable, which is why the two prompts here have different lengths.
//
// Both slots are checked, not just one. Two sequences that end up on the same row do not corrupt each other
// symmetrically -- a batch collapsed onto a single row leaves one of the two slots decoding exactly what it would
// have decoded alone -- so asserting on whichever slot came to mind proves nothing about the other half.
//
// Given two prompts of different lengths, decoded alone and then together as one batch
// When neither run forces its output tokens
// Then each slot decodes exactly what it decoded alone
TEST_F(RuntimeAssemblyTest, EachSlotDecodesTheSameTokensAloneAsItDoesBesideAnotherRequest)
{
    constexpr int64_t kGenerateLength{4};
    auto const shortPrompt = makePrompt(3, /*variant=*/0);
    auto const longPrompt = makePrompt(9, /*variant=*/1);

    auto const alone = [this](std::string const& prompt) {
        auto runtime = makeRuntime(makeVanillaArtifacts(mModelDir, makeFakeEngine(), mStream));
        rt::LLMGenerationResponse response;
        EXPECT_TRUE(runtime.handleRequest(makeGreedyRequest(prompt, kGenerateLength), response, mStream));
        return response.outputIds.empty() ? std::vector<int32_t>{} : response.outputIds[0];
    };
    auto const shortAlone = alone(shortPrompt);
    auto const longAlone = alone(longPrompt);
    ASSERT_EQ(shortAlone.size(), static_cast<size_t>(kGenerateLength));
    ASSERT_NE(shortAlone, longAlone)
        << "the two prompts decode alike, so this fixture cannot tell one slot's state from the other's";

    rt::LLMGenerationResponse batched;
    {
        auto runtime = makeRuntime(makeVanillaArtifacts(mModelDir, makeFakeEngine(), mStream));
        ASSERT_TRUE(
            runtime.handleRequest(makeGreedyBatch({shortPrompt, longPrompt}, kGenerateLength), batched, mStream));
        ASSERT_EQ(batched.outputIds.size(), 2U);
    }

    EXPECT_EQ(batched.outputIds[0], shortAlone) << "slot 0's answer moved when a second request joined the batch";
    EXPECT_EQ(batched.outputIds[1], longAlone) << "slot 1's answer moved when it was batched beside another request";
}

// Given a vanilla runtime whose logits are the same at every step
// When a seeded sampling request generates, and an unseeded one generates twice
// Then the seeded request replays the sampler's draws for its request-stable uniforms exactly, while the
//      unseeded runs draw the shared counter stream keyed on the step and row: both runs are identical, follow
//      that stream, and differ from the seeded run
TEST_F(RuntimeAssemblyTest, SeededReplaysUnseededSharesCounterStream)
{
    constexpr int64_t kGenerateLength{40}; // peak probability ~0.42: 40 equal draws is below 1e-15
    constexpr int32_t kPeak{5};
    constexpr uint64_t kSeed{1234U};
    std::vector<int32_t> const prompt{11, 12, 13};
    auto engine = makeFakeEngine();
    engine->forceOutputTokens({kPeak});
    auto runtime = makeRuntime(makeVanillaArtifacts(mModelDir, std::move(engine), mStream));

    auto request = makeGreedyRequest("a", kGenerateLength);
    request.preTokenizedInputIds = {prompt};
    request.temperature = 1.0F;
    request.topK = 0;
    request.topP = 0.95F;
    request.ignoreEos = true;
    auto const generate = [&](std::optional<uint64_t> seed) {
        request.samplingSeed = seed;
        rt::LLMGenerationResponse response;
        EXPECT_TRUE(runtime.handleRequest(request, response, mStream));
        EXPECT_EQ(response.outputIds.size(), 1U);
        return response.outputIds.empty() ? std::vector<int32_t>{} : response.outputIds.front();
    };

    // The engine's row, sampled with the uniform the runtime derives for the seed at each position.
    std::vector<float> row(static_cast<size_t>(kVocabSize));
    for (int64_t token = 0; token < kVocabSize; ++token)
    {
        row[static_cast<size_t>(token)] = fakeOutputValue(token, kPeak, kVocabSize, FakeOutputScale::kLogits);
    }
    rt::Tensor logits({1, kVocabSize}, rt::DeviceType::kGPU, trt_edgellm::DataType::kFLOAT);
    rt::Tensor selected({1, 1}, rt::DeviceType::kGPU, trt_edgellm::DataType::kINT32);
    rt::Tensor uniform({1}, rt::DeviceType::kGPU, trt_edgellm::DataType::kFLOAT);
    CUDA_CHECK(cudaMemcpy(logits.rawPointer(), row.data(), row.size() * sizeof(float), cudaMemcpyHostToDevice));
    SamplingParams const params(
        1, static_cast<int32_t>(kVocabSize), request.temperature, static_cast<int32_t>(request.topK), request.topP);
    rt::Tensor workspace(
        {static_cast<int64_t>(getTopKtopPSamplingWorkspaceSize(1, static_cast<int32_t>(kVocabSize), params))},
        rt::DeviceType::kGPU, trt_edgellm::DataType::kINT8);
    auto const sampleWith = [&](auto const& uniformAt) {
        std::vector<int32_t> tokens;
        for (int64_t step = 0; step < kGenerateLength; ++step)
        {
            float const draw = uniformAt(static_cast<uint64_t>(step));
            CUDA_CHECK(cudaMemcpyAsync(uniform.rawPointer(), &draw, sizeof(float), cudaMemcpyHostToDevice, mStream));
            topKtopPSamplingFromLogits(
                logits, selected, params, workspace, mStream, /*randomSeed=*/0U, /*randomOffset=*/0U, &uniform);
            int32_t token{};
            CUDA_CHECK(
                cudaMemcpyAsync(&token, selected.rawPointer(), sizeof(int32_t), cudaMemcpyDeviceToHost, mStream));
            CUDA_CHECK(cudaStreamSynchronize(mStream));
            tokens.push_back(token);
        }
        return tokens;
    };
    std::vector<int32_t> const expectedSeeded = sampleWith([&](uint64_t step) {
        return rt::requestStableUniform(kSeed, rt::requestStableNextAbsolutePosition(prompt.size(), step),
            rt::SpecRandomPurpose::kTarget, /*lane=*/0U);
    });
    // The unseeded stream: the default seed, the tokens generated so far and row 0
    // (decoder_utils::samplingUniformForSlot).
    std::vector<int32_t> const expectedUnseeded = sampleWith(
        [&](uint64_t step) { return counter_rng::counterUniform(rt::kDefaultSamplingSeed, step, /*lane=*/0U); });
    ASSERT_GT(std::set<int32_t>(expectedSeeded.begin(), expectedSeeded.end()).size(), 1U)
        << "every position draws the same token, so this fixture cannot tell one random number from another";
    ASSERT_NE(expectedSeeded, expectedUnseeded) << "the two streams agree on every draw, so they cannot be told apart";

    EXPECT_EQ(generate(kSeed), expectedSeeded) << "a seeded request must replay its request-stable draws exactly";

    std::vector<int32_t> const first = generate(std::nullopt);
    std::vector<int32_t> const second = generate(std::nullopt);
    EXPECT_EQ(first, expectedUnseeded) << "an unseeded request must draw the shared counter stream";
    EXPECT_EQ(first, second) << "two unseeded runs must repeat exactly";
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
        .WillByDefault(Return(trt_edgellm::DataType::kINT32));

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

// Given an EAGLE draft config whose candidate head emits 8 candidates per row, and a draft engine whose support outputs
// spec_proposal_support_ids (INT32) and spec_proposal_support_log_probs (FP32) are [rows, 8]
// When the engine is validated as is, with a logits output as well, or with a log-prob output of width 4
// Then only the first passes: a candidate-head draft engine has no logits output and exactly Kc columns per row
TEST_F(RuntimeAssemblyTest, ValidatesCandidateDraftSupportOutputs)
{
    using ::testing::Return;
    using ::testing::StrEq;

    rt::LLMEngineConfig config;
    config.specDecodeType = rt::SpecDecodeMode::kEAGLE;
    config.draftCandidateCount = 8;
    auto engine = makeCandidateDraftEngine(config.draftCandidateCount);
    EXPECT_NO_THROW(validateAgainstEngine(config, *engine, "draft"));

    ON_CALL(*engine, hasIOTensor(StrEq(binding_names::kLogits))).WillByDefault(Return(true));
    EXPECT_THROW(validateAgainstEngine(config, *engine, "draft"), std::runtime_error);

    ON_CALL(*engine, hasIOTensor(StrEq(binding_names::kLogits))).WillByDefault(Return(false));
    nvinfer1::Dims narrow{};
    narrow.nbDims = 2;
    narrow.d[0] = -1;
    narrow.d[1] = 4;
    ON_CALL(*engine, getTensorShape(StrEq(binding_names::kSpecProposalSupportLogProbs))).WillByDefault(Return(narrow));
    EXPECT_THROW(validateAgainstEngine(config, *engine, "draft"), std::runtime_error);
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

// Sampling parameters outside the sampler's domain are rejected before anything runs, instead of
// the sampler throwing from inside a step.
TEST_F(RequestValidationTest, RejectsOutOfDomainSampling)
{
    auto const withSampling = [this](float temperature, int64_t topK, float topP) {
        auto request = validRequest();
        request.temperature = temperature;
        request.topK = topK;
        request.topP = topP;
        return request;
    };
    expectRejectedWithoutRunning(withSampling(std::nanf(""), 0, 1.0F));
    expectRejectedWithoutRunning(withSampling(-0.5F, 0, 1.0F));
    expectRejectedWithoutRunning(withSampling(1.0F, 0, 0.0F));
    expectRejectedWithoutRunning(withSampling(1.0F, 0, 1.5F));
    expectRejectedWithoutRunning(withSampling(1.0F, 0, std::nanf("")));
}

// A top-k above the sampler's cap is served at the cap (with a warning), as before this runtime
// sized its workspace for the whole batch: the API accepts any top-k and the request still runs.
TEST_F(RequestValidationTest, ClampsTopKAboveTheSamplerCap)
{
    auto engine = makeEngine();
    EXPECT_CALL(*engine, execute(::testing::_)).WillRepeatedly(emit({}));
    auto artifacts = makeVanillaArtifacts(mModelDir, std::move(engine), mStream);
    auto runtime = makeRuntime(std::move(artifacts));

    auto request = validRequest();
    request.temperature = 0.7F;
    request.topK = int64_t{kMaxSamplingTopK} + 1;
    request.topP = 1.0F;
    rt::LLMGenerationResponse response;
    ASSERT_TRUE(runtime.handleRequest(request, response, mStream));
    ASSERT_EQ(response.outputIds.size(), 1U);
    EXPECT_EQ(response.outputIds.front().size(), 2U);
}

// A temperature alone (top-k disabled by 0 or by the API's -1, top-p 1.0) generates: the row samples from its
// whole distribution.
TEST_F(RequestValidationTest, AcceptsTemperatureOnlySampling)
{
    auto engine = makeEngine();
    EXPECT_CALL(*engine, execute(::testing::_)).WillRepeatedly(emit({}));
    auto artifacts = makeVanillaArtifacts(mModelDir, std::move(engine), mStream);
    auto runtime = makeRuntime(std::move(artifacts));

    for (int64_t const topK : {int64_t{0}, int64_t{-1}})
    {
        auto request = validRequest();
        request.temperature = 0.7F;
        request.topK = topK;
        request.topP = 1.0F;
        rt::LLMGenerationResponse response;
        ASSERT_TRUE(runtime.handleRequest(request, response, mStream)) << "topK " << topK;
        ASSERT_EQ(response.outputIds.size(), 1U);
        EXPECT_EQ(response.outputIds.front().size(), 2U);
    }
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

    auto const request = makeGreedyBatch({"aaa", "a"}, 2);

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
    //! log(softmax) of the row a substitute engine writes for `winner`: that entry's logit minus the log-sum-exp
    //! over the whole row, both taken from fakeLogit() rather than from anything the runtime returned.
    //!
    //! Sharing fakeLogit() with the engine is not the "expectation moves with the mutation" trap: what is under
    //! test is the runtime's log-softmax, and mutating that moves only one side of the comparison. The engine's
    //! distribution is the fixture, and a test of log(softmax(x)) has to know x.
    static double logSumExpOfRow(int32_t winner)
    {
        double sum{0.0};
        for (int64_t token = 0; token < kVocabSize; ++token)
        {
            sum += std::exp(static_cast<double>(fakeLogit(token, winner)));
        }
        return std::log(sum);
    }

    static double expectedLogprob(int32_t winner, int32_t token)
    {
        return static_cast<double>(fakeLogit(token, winner)) - logSumExpOfRow(winner);
    }

    //! A runtime over an engine whose forward passes emit `peaks` in order: the
    //! first is the token prefill produces, the rest are one decode round each.
    //!
    //! Stated as engine state rather than as expected calls, because what these
    //! tests read back is the distribution the logits describe. How many times
    //! the runtime calls the engine to produce it is not their claim.
    template <typename Body>
    void withEngineEmitting(std::vector<int32_t> const& peaks, Body&& body)
    {
        std::vector<std::vector<int32_t>> schedule;
        schedule.reserve(peaks.size());
        for (int32_t const peak : peaks)
        {
            schedule.push_back({peak});
        }

        auto engine = makeFakeEngine();
        engine->forceOutputTokensPerRound(std::move(schedule));

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

    std::vector<int32_t> const peaks{5, 7};
    withEngineEmitting(peaks, [&](rt::LLMInferenceRuntime& runtime) {
        auto request = makeGreedyRequest("a", 2);
        request.numLogprobs = kTopK;

        rt::LLMGenerationResponse response;
        ASSERT_TRUE(runtime.handleRequest(request, response, mStream));

        ASSERT_EQ(response.logprobs[0].size(), 2U);
        for (size_t stepIndex = 0; stepIndex < response.logprobs[0].size(); ++stepIndex)
        {
            auto const& step = response.logprobs[0][stepIndex];
            // Guard: an empty step list would make the loop below vacuous.
            ASSERT_EQ(step.size(), static_cast<size_t>(kTopK));
            for (size_t k = 0; k < step.size(); ++k)
            {
                EXPECT_NEAR(step[k].logprob, expectedLogprob(peaks[stepIndex], step[k].tokenId), 1e-3)
                    << "step " << stepIndex << " entry " << k << " (token " << step[k].tokenId << ")";
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

    auto engine = makeFakeEngine();
    engine->forceOutputTokens({kFirstSlotPeak, kSecondSlotPeak});

    auto artifacts = makeVanillaArtifacts(mModelDir, std::move(engine), mStream);
    auto runtime = makeRuntime(std::move(artifacts));

    auto request = makeGreedyBatch({"a", "a"}, 2);
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

// --------------------------------------------------------------------------
// MTP per-request fallback: request.disableSpecDecode routes a single request through the vanilla path on an
// otherwise-speculative runtime, without touching the draft engine at all. This is the dynamic per-request
// dispatch an in-flight-batching rewrite (heterogeneous requests sharing one runtime) is most likely to break
// silently, so it gets its own call-recording harness rather than piggybacking on MtpAssemblyTest's mock.
// --------------------------------------------------------------------------

Json makeMtpDraftConfig()
{
    Json config = makeSpecDraftConfig("mtp", kVerifySize);
    // MTP's draft consumes the base hidden state unchanged, so this equals the base hidden size.
    config["base_model_hidden_size"] = config["hidden_size"];
    return config;
}

std::filesystem::path writeMtpModelDir(std::string const& suite)
{
    return writeSpecModelDir(suite, makeSpecBaseConfig("mtp", kVerifySize), makeMtpDraftConfig());
}

rt::ModelArtifacts makeMtpArtifacts(std::filesystem::path const& modelDir,
    std::unique_ptr<rt::EngineExecutor> baseEngine, std::unique_ptr<rt::EngineExecutor> draftEngine,
    cudaStream_t stream)
{
    return makeSpecArtifacts(modelDir, makeMtpDrafting(), std::move(baseEngine), std::move(draftEngine), stream);
}

class MtpFallbackTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        CUDA_CHECK(cudaStreamCreate(&mStream));
        mModelDir = writeMtpModelDir("mtpFallbackTests");
    }

    void TearDown() override
    {
        CUDA_CHECK(cudaStreamDestroy(mStream));
        std::filesystem::remove_all(mModelDir);
    }

    //! Each engine counts its own forwards, which is all this suite asks of them: whether the draft ran at all,
    //! and how many times the base did.
    std::unique_ptr<FakeEngineExecutor> makeObservingEngine(FakeEngineExecutor*& recordHere)
    {
        auto engine = makeFakeEngine(makeSpecBaseConfig("mtp", kVerifySize));
        engine->forceOutputTokens(agreedLogitsRows(kAgreedToken));
        recordHere = engine.get();
        return engine;
    }

    std::unique_ptr<FakeEngineExecutor> makeBaseEngine()
    {
        return makeObservingEngine(mBase);
    }

    //! A draft engine's output binding carries log-probabilities, not logits: every draft model ends in
    //! `F.log_softmax`. Drafting accumulates those values down the chain, so an engine on the wrong scale
    //! collapses the verify tree.
    std::unique_ptr<FakeEngineExecutor> makeDraftEngine()
    {
        auto engine = makeObservingEngine(mDraft);
        engine->actAsDraftEngine();
        return engine;
    }

    //! Engines left to derive their own output, for the test that compares speculative decoding against the
    //! vanilla path rather than counting forwards. Forcing tokens is not an option there: a forced row is written
    //! the same way on every round, which is not a draft distribution a tree can be built from.
    std::unique_ptr<FakeEngineExecutor> makeComputingEngine(FakeEngineExecutor*& recordHere, bool isDraft)
    {
        auto engine = makeFakeEngine(makeSpecBaseConfig("mtp", kVerifySize));
        if (isDraft)
        {
            engine->actAsDraftEngine();
        }
        recordHere = engine.get();
        return engine;
    }

    cudaStream_t mStream{};
    std::filesystem::path mModelDir;
    //! Non-owning: the runtime takes ownership of both engines and outlives every read of them here.
    FakeEngineExecutor* mBase{nullptr};
    FakeEngineExecutor* mDraft{nullptr};
};

//! Given an MTP runtime, when one request sets disableSpecDecode and a second (same runtime) does not,
//! then the first never calls the draft engine at all (vanilla-shaped: one base prefill, one base decode
//! per generated token) and the second still runs the ordinary draft-chain-then-verify round -- proving
//! the fallback is decided per request, not latched onto the runtime for its lifetime.
TEST_F(MtpFallbackTest, DisableSpecDecodeRoutesThatRequestThroughVanillaWithoutTouchingTheDraftEngine)
{
    auto artifacts = makeMtpArtifacts(mModelDir, makeBaseEngine(), makeDraftEngine(), mStream);
    rt::LLMInferenceRuntime runtime{
        std::move(artifacts), mModelDir.string(), /*multimodalEngineDir=*/"", {}, makeMtpDrafting(), mStream};

    constexpr int64_t kFallbackGenerateLength{3};
    auto fallbackRequest = makeGreedyRequest("a", kFallbackGenerateLength);
    fallbackRequest.disableSpecDecode = true;
    rt::LLMGenerationResponse fallbackResponse;
    ASSERT_TRUE(runtime.handleRequest(fallbackRequest, fallbackResponse, mStream));

    ASSERT_EQ(fallbackResponse.outputIds.size(), 1U);
    EXPECT_THAT(fallbackResponse.outputIds[0], ::testing::SizeIs(kFallbackGenerateLength));
    EXPECT_EQ(mDraft->forwardsRun(), 0U) << "draft engine executed while disableSpecDecode was set";
    // Forward count only (not the exact prepare/decode profile sequence, a scheduling-order assumption separate
    // from the fallback behavior this test protects): nothing beyond one forward per generated token ran on the
    // base engine.
    EXPECT_EQ(mBase->forwardsRun(), static_cast<size_t>(kFallbackGenerateLength));

    size_t const draftForwardsBeforeSpecDecode = mDraft->forwardsRun();
    auto const specDecodeRequest = makeGreedyRequest("a", kVerifySize);
    rt::LLMGenerationResponse specDecodeResponse;
    ASSERT_TRUE(runtime.handleRequest(specDecodeRequest, specDecodeResponse, mStream));

    ASSERT_EQ(specDecodeResponse.outputIds.size(), 1U);
    EXPECT_THAT(specDecodeResponse.outputIds[0], ::testing::SizeIs(kVerifySize));
    // The same runtime that just ran vanilla-only now uses the draft engine: the fallback did not stick.
    EXPECT_GT(mDraft->forwardsRun(), draftForwardsBeforeSpecDecode)
        << "second request should still use the draft engine";
}

// Speculative decoding is defined by the fact that it changes nothing: a draft is a guess the base engine either
// confirms or throws away, so greedy decoding returns the same tokens whether or not drafting ran. That is also
// the property nothing here could check. A draft proposing from a collapsed verify tree, or a verification
// committing one token too many, still produces output of the right length -- and the other tests in this file
// count forwards and read the shape of a round, which a wrong-but-well-formed draft satisfies exactly. This is
// the test that went red on the ordering defect the drafting fix addresses.
//
// Given one MTP runtime and one prompt
// When the request runs once with disableSpecDecode and once with drafting on, neither forcing its output tokens
// Then both return the same tokens, and drafting really did run on the speculative pass
TEST_F(MtpFallbackTest, SpeculativeDecodingReturnsWhatTheVanillaPathReturns)
{
    auto artifacts = makeMtpArtifacts(mModelDir, makeComputingEngine(mBase, /*isDraft=*/false),
        makeComputingEngine(mDraft, /*isDraft=*/true), mStream);
    rt::LLMInferenceRuntime runtime{
        std::move(artifacts), mModelDir.string(), /*multimodalEngineDir=*/"", {}, makeMtpDrafting(), mStream};

    constexpr int64_t kGenerateLength{6};
    auto const prompt = makePrompt(5);

    auto fallbackRequest = makeGreedyRequest(prompt, kGenerateLength);
    fallbackRequest.disableSpecDecode = true;
    rt::LLMGenerationResponse fallback;
    ASSERT_TRUE(runtime.handleRequest(fallbackRequest, fallback, mStream));
    ASSERT_EQ(fallback.outputIds.size(), 1U);
    ASSERT_THAT(fallback.outputIds[0], ::testing::SizeIs(kGenerateLength));
    ASSERT_EQ(mDraft->forwardsRun(), 0U) << "the fallback pass drafted, so it is not a vanilla reference";

    rt::LLMGenerationResponse speculative;
    ASSERT_TRUE(runtime.handleRequest(makeGreedyRequest(prompt, kGenerateLength), speculative, mStream));
    ASSERT_EQ(speculative.outputIds.size(), 1U);
    ASSERT_GT(mDraft->forwardsRun(), 0U) << "the speculative pass never drafted, so this proves nothing";

    EXPECT_EQ(speculative.outputIds[0], fallback.outputIds[0])
        << "drafting changed the answer, which is the one thing it is not allowed to do";
}

TEST(ShouldUseHybridSpecEndpointReuseTest, RequiresSpecModeAndAHybridBaseAndEitherCacheDirection)
{
    using rt::DecodingStrategyKind;
    using rt::shouldUseHybridSpecEndpointReuse;

    EXPECT_TRUE(shouldUseHybridSpecEndpointReuse(DecodingStrategyKind::kMTP, /*hybridBase=*/true,
        /*lookup=*/true, /*publish=*/false));
    EXPECT_TRUE(shouldUseHybridSpecEndpointReuse(DecodingStrategyKind::kMTP, /*hybridBase=*/true,
        /*lookup=*/false, /*publish=*/true));
    EXPECT_FALSE(shouldUseHybridSpecEndpointReuse(DecodingStrategyKind::kMTP, /*hybridBase=*/true,
        /*lookup=*/false, /*publish=*/false))
        << "neither looking up nor publishing state means there is nothing for this path to do";
    EXPECT_FALSE(shouldUseHybridSpecEndpointReuse(DecodingStrategyKind::kMTP, /*hybridBase=*/false,
        /*lookup=*/true, /*publish=*/true))
        << "a non-hybrid base has no recurrent state for the endpoint checkpoint to carry";
    EXPECT_TRUE(shouldUseHybridSpecEndpointReuse(DecodingStrategyKind::kEAGLE, /*hybridBase=*/true,
        /*lookup=*/true, /*publish=*/true))
        << "EAGLE on a hybrid base qualifies for hybrid spec endpoint reuse";
    EXPECT_TRUE(shouldUseHybridSpecEndpointReuse(DecodingStrategyKind::kDFlash, /*hybridBase=*/true,
        /*lookup=*/true, /*publish=*/true))
        << "DFlash on a hybrid base qualifies for hybrid spec endpoint reuse";
    EXPECT_FALSE(shouldUseHybridSpecEndpointReuse(DecodingStrategyKind::kVanilla, /*hybridBase=*/true,
        /*lookup=*/true, /*publish=*/true))
        << "Vanilla decoding does not use the hybrid spec endpoint path";
}

// --------------------------------------------------------------------------
// Hybrid+MTP context-cache endpoint reuse: same MTP assembly as above, but over a hybrid (attention+mamba) base
// with ContextCacheConfig enabled. No prior unit test constructed a runtime with context caching enabled at all --
// runHybridMtpPrefill (llmRankRuntime.cpp) had zero coverage. Regression fence ahead of the in-flight-batching
// rewrite to handleRequest().
// --------------------------------------------------------------------------

//! Same base as MTP, plus the hybrid (attention+mamba) layer split validateContextCacheDeployment requires for
//! MTP context reuse: attention layers first then mamba (llmEngineConfig.cpp's fallback layer-type ordering).
Json makeHybridMtpBaseConfig()
{
    Json config = makeSpecBaseConfig("mtp", kVerifySize);
    config["num_attention_layers"] = 1;
    config["num_linear_attn_layers"] = 1;
    config["recurrent_state_num_heads"] = 2;
    config["recurrent_state_head_dim"] = 128;
    config["recurrent_state_size"] = 128;
    // GDN chunk replay uses 128-wide K/V heads and one K head for two V heads.
    config["conv_dim"] = 512;
    config["conv_kernel"] = 4;
    config["recurrent_state_dtype"] = "fp32";
    config["conv_state_dtype"] = "fp16";
    // Endpoint reuse publishes a paged spec-decode record, which needs >=1 full page (128 tokens) of resident
    // content -- sizeSpeculativeKvPool() doesn't widen max_input_len, so this fixture must do it separately.
    config["builder_config"]["max_input_len"] = 256;
    return config;
}

std::filesystem::path writeHybridMtpModelDir()
{
    Json const baseConfig = makeHybridMtpBaseConfig();
    // Draft stays non-hybrid: validateContextCacheDeployment requires an independent (non-shared) draft KV cache,
    // which is exactly makeMtpDraftConfig()'s contract (hasOwnKVCache defaults true, sharesTargetKV defaults false).
    Json draftConfig = makeMtpDraftConfig();
    draftConfig["builder_config"]["max_input_len"] = 256;
    return writeSpecModelDir("hybridMtpCtxReuseTests", baseConfig, draftConfig);
}

class HybridMtpContextReuseTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        CUDA_CHECK(cudaStreamCreate(&mStream));
        mModelDir = writeHybridMtpModelDir();
    }

    void TearDown() override
    {
        CUDA_CHECK(cudaStreamDestroy(mStream));
        std::filesystem::remove_all(mModelDir);
    }

    //! Ties every logits row to token 0 ("a") -- the only non-special character this tokenizer can round-trip,
    //! which lets a continuation prompt be built by literal string concatenation instead of decoding ids back
    //! through the tokenizer.
    //!
    //! The two engines record separately, so a test picks the one it means rather than filtering a shared list by
    //! a flag: "the draft's first prefill" is a question about the draft engine.
    std::unique_ptr<FakeEngineExecutor> makeEngine(FakeEngineExecutor*& recordHere)
    {
        auto engine = makeFakeEngine(makeHybridMtpBaseConfig());
        // Every verify row, not only the first: while drafting could not accept past the tree root
        // the rest never reached the output, so forcing one row looked sufficient.
        engine->forceOutputTokens(agreedLogitsRows(0, kVerifySize));
        engine->recordBinding(binding_names::kKVPageTable);
        recordHere = engine.get();
        return engine;
    }

    std::unique_ptr<FakeEngineExecutor> makeBaseEngine()
    {
        auto engine = makeEngine(mBase);
        engine->exposeInt32Binding(binding_names::kTreeParentIds);
        engine->exposeInt32Binding(binding_names::kTreeDepths);
        engine->exposeInt32Binding(binding_names::kSpecVerifyLayout);
        return engine;
    }

    //! A draft engine's output binding carries log-probabilities, not logits: every draft model ends in
    //! `F.log_softmax`. Drafting accumulates those values down the chain, so an engine on the wrong scale
    //! collapses the verify tree.
    std::unique_ptr<FakeEngineExecutor> makeDraftEngine()
    {
        auto engine = makeEngine(mDraft);
        engine->actAsDraftEngine();
        return engine;
    }

    static bool isPrefill(FakeEngineExecutor::PreparedRound const& round)
    {
        return round.profileIndex == kPrefillProfile;
    }

    //! The first page `engine`'s page table named on its `round`-th forward, or kUNUSED_PAGE_ENTRY if that forward
    //! bound no page table.
    static int32_t firstPageOn(FakeEngineExecutor const& engine, size_t round)
    {
        auto const& table = engine.preparedBinding(binding_names::kKVPageTable)[round];
        return table.empty() ? rt::kUNUSED_PAGE_ENTRY : table.front();
    }

    //! Index of `engine`'s first prefill forward at or after `from`.
    static size_t findPrefill(FakeEngineExecutor const& engine, size_t from)
    {
        auto const& rounds = engine.preparedRounds();
        for (size_t i = from; i < rounds.size(); ++i)
        {
            if (isPrefill(rounds[i]))
            {
                return i;
            }
        }
        return rounds.size();
    }

    cudaStream_t mStream{};
    std::filesystem::path mModelDir;
    //! Non-owning: the runtime takes ownership of both engines and outlives every read of them here.
    FakeEngineExecutor* mBase{nullptr};
    FakeEngineExecutor* mDraft{nullptr};
};

//! Hybrid+MTP endpoint reuse publishes a paged spec-decode record (validateSpecState in cacheRecord.cpp), which
//! -- like plain context-cache reuse (kPagedPromptLength below) -- requires at least one full page
//! (kTOKENS_PER_PAGE == 128) of resident content.
constexpr int64_t kHybridMtpPagedPromptLength{130};

TEST_F(HybridMtpContextReuseTest, SecondTurnPrefillShrinksToTheBoundaryOnACacheHit)
{
    auto artifacts = makeMtpArtifacts(mModelDir, makeBaseEngine(), makeDraftEngine(), mStream);

    rt::ContextCacheConfig cacheConfig;
    cacheConfig.enabled = true;
    cacheConfig.maxRecords = 16;
    cacheConfig.maxSnapshots = 64;
    cacheConfig.encoderEmbeddingCacheBudgetBytes = 0;

    rt::LLMInferenceRuntime runtime{std::move(artifacts), mModelDir.string(), /*multimodalEngineDir=*/"", {},
        makeMtpDrafting(), mStream, cacheConfig};

    // maxGenerateLength must reach a full MTP round (kVerifySize): a request satisfied entirely out of prefill's
    // own sampled token never runs decodeStep(), so baseHiddenStates is never reshaped back to the 3D
    // [batch, seq, hiddenDim] triple boundary-hidden capture requires.
    auto firstRequest = makeGreedyRequest(makePrompt(kHybridMtpPagedPromptLength), /*maxGenerateLength=*/kVerifySize);
    firstRequest.contextCacheCommitPolicy = rt::ContextCacheCommitPolicy::kPrefillStateOnly;
    rt::LLMGenerationResponse firstResponse;
    ASSERT_TRUE(runtime.handleRequest(firstRequest, firstResponse, mStream));
    ASSERT_EQ(firstResponse.outputIds[0].size(), static_cast<size_t>(kVerifySize));
    EXPECT_THAT(firstResponse.outputIds[0], ::testing::Each(0)) << "engine ties every row to token 0 (\"a\")";

    size_t const baseRoundsBeforeSecond = mBase->preparedRounds().size();
    // Literal continuation of turn 1's exact token sequence (prompt + generated), which this single-character/
    // no-merge tokenizer round-trips without re-tokenization drift.
    constexpr int64_t kSecondPromptLength{kHybridMtpPagedPromptLength + kVerifySize};
    auto secondRequest = makeGreedyRequest(
        makePrompt(kHybridMtpPagedPromptLength) + std::string(kVerifySize, 'a'), /*maxGenerateLength=*/1);
    secondRequest.contextCacheCommitPolicy = rt::ContextCacheCommitPolicy::kPrefillStateOnly;
    rt::LLMGenerationResponse secondResponse;
    ASSERT_TRUE(runtime.handleRequest(secondRequest, secondResponse, mStream));

    size_t const secondBasePrefill = findPrefill(*mBase, baseRoundsBeforeSecond);
    ASSERT_LT(secondBasePrefill, mBase->preparedRounds().size())
        << "second request never reached a base prefill forward";
    EXPECT_LT(mBase->preparedRounds()[secondBasePrefill].dims.seqLen, kSecondPromptLength)
        << "endpoint reuse should fold the matched prefix into a shorter prefill; a full-length prefill means the "
           "checkpoint was not hit";
}

// Given a Hybrid+MTP endpoint whose recurrent/partial-KV state was restored on a cache hit
// When the second turn's draft prefill runs
// Then the draft engine's own kv_page_table binding names the lease's real physical page, not the table's
//   never-written sentinel -- and it is the same page slot 0's first turn was uploaded to, since this is a reuse
//   of that turn's own record, not a fresh allocation
//
// Regression lock for !1247 (note_63812832, fixed in ddebce082): the paired-restore path writes page *contents*
// but that alone does not make the draft engine reach them -- MTPDecoder::initializeForGeneration resolves the
// draft's KV purely through kvPageTables[1], which the coordinator must separately publish via
// preparePrefill()'s runsPairedDraftWorkingSet()-gated setRows()/upload(). That predicate used to be EAGLE-only,
// so Hybrid+MTP prefills silently kept whatever page ids were last uploaded (or the construction-time
// kUNUSED_PAGE_ENTRY sentinel, since a table that has never been uploaded never leaves that value), and the
// draft engine read KV through those stale bindings instead of the lease's own pages -- corrupting nothing that
// changes the sampled token, since the base always re-verifies, but silently costing acceptance rate.
TEST_F(HybridMtpContextReuseTest, SecondTurnDraftPrefillPublishesTheLeasedPageToTheDraftTable)
{
    auto artifacts = makeMtpArtifacts(mModelDir, makeBaseEngine(), makeDraftEngine(), mStream);

    rt::ContextCacheConfig cacheConfig;
    cacheConfig.enabled = true;
    cacheConfig.maxRecords = 16;
    cacheConfig.maxSnapshots = 64;
    cacheConfig.encoderEmbeddingCacheBudgetBytes = 0;

    rt::LLMInferenceRuntime runtime{std::move(artifacts), mModelDir.string(), /*multimodalEngineDir=*/"", {},
        makeMtpDrafting(), mStream, cacheConfig};

    auto firstRequest = makeGreedyRequest(makePrompt(kHybridMtpPagedPromptLength), /*maxGenerateLength=*/kVerifySize);
    firstRequest.contextCacheCommitPolicy = rt::ContextCacheCommitPolicy::kPrefillStateOnly;
    rt::LLMGenerationResponse firstResponse;
    ASSERT_TRUE(runtime.handleRequest(firstRequest, firstResponse, mStream));

    size_t const firstDraftPrefill = findPrefill(*mDraft, 0);
    ASSERT_LT(firstDraftPrefill, mDraft->preparedRounds().size())
        << "first request never reached a draft prefill forward";
    ASSERT_NE(firstPageOn(*mDraft, firstDraftPrefill), rt::kUNUSED_PAGE_ENTRY)
        << "the draft page table was never uploaded even on the producing turn";
    int32_t const producerDraftPage = firstPageOn(*mDraft, firstDraftPrefill);

    size_t const draftRoundsBeforeSecond = mDraft->preparedRounds().size();
    auto secondRequest = makeGreedyRequest(
        makePrompt(kHybridMtpPagedPromptLength) + std::string(kVerifySize, 'a'), /*maxGenerateLength=*/1);
    secondRequest.contextCacheCommitPolicy = rt::ContextCacheCommitPolicy::kPrefillStateOnly;
    rt::LLMGenerationResponse secondResponse;
    ASSERT_TRUE(runtime.handleRequest(secondRequest, secondResponse, mStream));

    size_t const secondDraftPrefill = findPrefill(*mDraft, draftRoundsBeforeSecond);
    ASSERT_LT(secondDraftPrefill, mDraft->preparedRounds().size())
        << "second (cache-hit) request never reached a draft prefill forward";
    EXPECT_NE(firstPageOn(*mDraft, secondDraftPrefill), rt::kUNUSED_PAGE_ENTRY)
        << "cache hit left the draft page table at its never-written sentinel -- the restore path skipped "
           "publishing the mapping, same as !1247";
    EXPECT_EQ(firstPageOn(*mDraft, secondDraftPrefill), producerDraftPage)
        << "the cache-hit draft prefill should resolve to the same physical page the producing turn recorded";
}

TEST_F(HybridMtpContextReuseTest, RecurrentCaptureIntervalViolationIsRejectedCleanly)
{
    auto artifacts = makeMtpArtifacts(mModelDir, makeBaseEngine(), makeDraftEngine(), mStream);

    rt::ContextCacheConfig cacheConfig;
    cacheConfig.enabled = true;
    cacheConfig.maxRecords = 16;
    cacheConfig.maxSnapshots = 64;
    cacheConfig.encoderEmbeddingCacheBudgetBytes = 0;

    rt::LLMInferenceRuntime runtime{std::move(artifacts), mModelDir.string(), /*multimodalEngineDir=*/"", {},
        makeMtpDrafting(), mStream, cacheConfig};

    // Hybrid+MTP endpoint reuse requires recurrentCaptureInterval==0 (periodic capture would land the recurrent
    // snapshot away from the stable predecessor boundary). A request that asks for both must be rejected outright,
    // not silently downgraded to periodic capture or to no reuse at all.
    auto request = makeGreedyRequest(makePrompt(kHybridMtpPagedPromptLength), /*maxGenerateLength=*/1);
    request.contextCacheCommitPolicy = rt::ContextCacheCommitPolicy::kPrefillStateOnly;
    request.recurrentCaptureInterval = 1;
    rt::LLMGenerationResponse response;
    EXPECT_FALSE(runtime.handleRequest(request, response, mStream));
    EXPECT_TRUE(response.outputIds.empty());
}

// --------------------------------------------------------------------------
// Plain (non-MTP) context-cache reuse: ContextCacheCoordinator's ordinary prefix lookup/publish path, not the
// MTP-specific boundary-hidden fold above. Unlike Hybrid+MTP reuse, validateContextCacheDeployment only requires
// numAttentionLayers>0 || numLinearAttnLayers>0 for a vanilla (SpecDecodeMode::kNONE) deployment --
// makeTinyVanillaConfig()'s plain attention-only base already qualifies, no hybrid (mamba) layers needed.
// --------------------------------------------------------------------------

//! Context-cache reuse operates at page granularity (kTOKENS_PER_PAGE == 128, common/pagedKvTypes.h): a sequence
//! shorter than one full page never has anything to publish, no matter how exact its prefix match is. Widen
//! makeTinyVanillaConfig()'s KV geometry so a >=1-page sequence fits.
Json makeContextReusePagedConfig()
{
    Json config = makeTinyVanillaConfig();
    config["builder_config"]["max_input_len"] = 256;
    config["builder_config"]["max_kv_cache_capacity"] = 256;
    config["builder_config"]["max_kv_pool_pages"] = rt::computeMinimumKvPoolPages(kMaxBatchSize, 256);
    return config;
}

//! One full page plus a few extra tokens, so the published record covers exactly one page and the tail past it
//! stays as new work on a cache hit -- long enough to source both requests' prompts (single-char tokenizer, 1
//! token per "a").
constexpr int64_t kPagedPromptLength{130};

class PlainContextReuseTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        CUDA_CHECK(cudaStreamCreate(&mStream));
        mModelDir = writeModelDir("plainContextReuseTests", makeContextReusePagedConfig());
    }

    void TearDown() override
    {
        CUDA_CHECK(cudaStreamDestroy(mStream));
        std::filesystem::remove_all(mModelDir);
    }

    //! Ties every logits row to token 0 ("a"), the only non-special character this tokenizer round-trips, so a
    //! continuation prompt can be built by literal string concatenation without risking re-tokenization drift.
    //!
    //! Whether a prefix was reused is invisible in the output -- the reused and recomputed paths decode the same
    //! text -- so what these tests read is the sequence length the runtime asked each forward for, which the
    //! engine records as it is prepared.
    std::unique_ptr<FakeEngineExecutor> makeEngine()
    {
        auto engine = makeFakeEngine(makeContextReusePagedConfig());
        engine->forceOutputTokens({0});
        mEngine = engine.get();
        return engine;
    }

    //! An engine left to derive its own output, for the tests that compare two runs rather than read a forward's
    //! shape. Cannot serve the continuation-prompt tests above: those need the generated text to be literal "a"s.
    std::unique_ptr<FakeEngineExecutor> makeComputingEngine()
    {
        auto engine = makeFakeEngine(makeContextReusePagedConfig());
        mEngine = engine.get();
        return engine;
    }

    //! The forwards recorded so far, owned by the engine the runtime took over.
    std::vector<FakeEngineExecutor::PreparedRound> const& calls() const
    {
        return mEngine->preparedRounds();
    }

    static bool isPrefill(FakeEngineExecutor::PreparedRound const& round)
    {
        return round.profileIndex == kPrefillProfile;
    }

    static rt::ContextCacheConfig makeCacheConfig()
    {
        rt::ContextCacheConfig cacheConfig;
        cacheConfig.enabled = true;
        cacheConfig.maxRecords = 16;
        cacheConfig.maxSnapshots = 64;
        cacheConfig.encoderEmbeddingCacheBudgetBytes = 0;
        return cacheConfig;
    }

    //! A request engine over a reuse-enabled runtime that seats up to two requests at once.
    std::unique_ptr<rt::scheduler::RequestEngine> makeRequestEngine(std::unique_ptr<FakeEngineExecutor> engine)
    {
        auto artifacts = makeVanillaArtifacts(mModelDir, std::move(engine), mStream);
        auto runtime = std::make_unique<rt::LLMInferenceRuntime>(std::move(artifacts), mModelDir.string(),
            /*multimodalEngineDir=*/"", std::unordered_map<std::string, std::string>{}, std::nullopt, mStream,
            makeCacheConfig());
        rt::scheduler::EngineConfig config;
        config.maxBatchSize = 2;
        return std::make_unique<rt::scheduler::RequestEngine>(std::move(runtime), mStream, config);
    }

    cudaStream_t mStream{};
    std::filesystem::path mModelDir;
    //! Non-owning: the runtime takes ownership of the engine, and outlives every read of it here.
    FakeEngineExecutor* mEngine{nullptr};
};

TEST_F(PlainContextReuseTest, SecondRequestReusesTheFirstRequestsPrefix)
{
    auto artifacts = makeVanillaArtifacts(mModelDir, makeEngine(), mStream);
    rt::LLMInferenceRuntime runtime{std::move(artifacts), mModelDir.string(), /*multimodalEngineDir=*/"", {},
        std::nullopt, mStream, makeCacheConfig()};

    constexpr int64_t kFirstGenerateLength{4};
    constexpr int64_t kTotalLength{kPagedPromptLength + kFirstGenerateLength};
    auto const firstPrompt = makePrompt(kPagedPromptLength);
    auto firstRequest = makeGreedyRequest(firstPrompt, kFirstGenerateLength);
    rt::LLMGenerationResponse firstResponse;
    ASSERT_TRUE(runtime.handleRequest(firstRequest, firstResponse, mStream));
    ASSERT_EQ(firstResponse.outputIds[0].size(), static_cast<size_t>(kFirstGenerateLength));

    auto const coldMetrics = runtime.getContextCacheMetrics();
    ASSERT_TRUE(coldMetrics.has_value());
    EXPECT_EQ(coldMetrics->hitSequences, 0U) << "the first request has nothing to reuse yet";

    size_t const callsBeforeSecond = calls().size();
    // Literal continuation of request 1's exact tokens (prompt + the kFirstGenerateLength generated "a"s) -- the
    // toy tokenizer round-trips this without drift.
    auto secondRequest = makeGreedyRequest(firstPrompt + std::string(kFirstGenerateLength, 'a'),
        /*maxGenerateLength=*/1);
    rt::LLMGenerationResponse secondResponse;
    ASSERT_TRUE(runtime.handleRequest(secondRequest, secondResponse, mStream));

    auto const warmMetrics = runtime.getContextCacheMetrics();
    ASSERT_TRUE(warmMetrics.has_value());
    EXPECT_EQ(warmMetrics->hitSequences, 1U)
        << "the second request's prefix should hit the first request's published state";
    EXPECT_GT(warmMetrics->reusedTokens, 0U);

    auto const secondPrefill
        = std::find_if(calls().begin() + static_cast<std::ptrdiff_t>(callsBeforeSecond), calls().end(), isPrefill);
    ASSERT_NE(secondPrefill, calls().end()) << "second request never reached a base prefill forward";
    EXPECT_LT(secondPrefill->dims.seqLen, kTotalLength)
        << "a cache hit should fold the matched prefix out of the prefill forward's sequence length";
}

// Every other reuse test here reads the sequence length a prefill asked for. That shows work was skipped, not that
// the right work was skipped: a reuse that restored a page belonging to something else, or that claimed a prefix
// one token longer than it actually put back, shortens the prefill just as convincingly. The property that defines
// a correct cache hit is that it is invisible -- a prompt served partly from cache has to decode exactly what the
// same prompt decodes cold.
//
// Given one prompt extending another, and neither run forcing its output tokens, so what comes back is derived
// from the KV entries behind it
// When the long prompt is served once on a cold runtime and once behind a published prefix of the short one
// Then both decode the same tokens, and the second one really did reuse
TEST_F(PlainContextReuseTest, APartlyReusedPromptDecodesWhatAColdPrefillWouldHave)
{
    constexpr int64_t kGenerateLength{4};
    constexpr int64_t kContinuationLength{kPagedPromptLength + 20};
    auto const prefix = std::string(kPagedPromptLength, 'a');
    auto const continuation = std::string(kContinuationLength, 'a');

    auto runtimeOn = [this](std::filesystem::path const& dir) {
        auto artifacts = makeVanillaArtifacts(dir, makeComputingEngine(), mStream);
        return rt::LLMInferenceRuntime{std::move(artifacts), dir.string(), /*multimodalEngineDir=*/"", {}, std::nullopt,
            mStream, makeCacheConfig()};
    };

    rt::LLMGenerationResponse cold;
    {
        auto runtime = runtimeOn(mModelDir);
        ASSERT_TRUE(runtime.handleRequest(makeGreedyRequest(continuation, kGenerateLength), cold, mStream));
        ASSERT_EQ(cold.outputIds[0].size(), static_cast<size_t>(kGenerateLength));
    }

    auto runtime = runtimeOn(mModelDir);
    rt::LLMGenerationResponse published;
    ASSERT_TRUE(runtime.handleRequest(makeGreedyRequest(prefix, kGenerateLength), published, mStream));

    rt::LLMGenerationResponse warm;
    ASSERT_TRUE(runtime.handleRequest(makeGreedyRequest(continuation, kGenerateLength), warm, mStream));

    auto const metrics = runtime.getContextCacheMetrics();
    ASSERT_TRUE(metrics.has_value());
    ASSERT_EQ(metrics->hitSequences, 1U) << "the continuation never reused anything, so this proves nothing";
    ASSERT_GT(metrics->reusedTokens, 0U);

    EXPECT_EQ(warm.outputIds[0], cold.outputIds[0])
        << "the reused prefix put different entries behind the continuation than a cold prefill computes";
}

// Given a request engine over a runtime with context reuse on
// When it opens, and then serves a prompt followed by a continuation of it
// Then its context-cache counters show an empty cache before the first submit, the prompt's publication by the
// time the prompt's response is released, and the continuation's hit by the time the continuation's response is
// released, reusing exactly the prompt tokens the continuation's prefill forward skipped
TEST_F(PlainContextReuseTest, RequestEngineReportsTheContextCacheItsRuntimeHolds)
{
    auto requestEngine = makeRequestEngine(makeEngine());

    auto const opened = requestEngine->contextCacheMetrics();
    ASSERT_TRUE(opened.has_value()) << "the engine hid the counters of a runtime that has a context cache";
    EXPECT_EQ(opened->admittedSequences, 0U);
    EXPECT_EQ(opened->currentRecords, 0U);
    EXPECT_GT(opened->baseKvPages.capacity, 0);
    EXPECT_EQ(opened->baseKvPages.free, opened->baseKvPages.capacity);

    constexpr int64_t kFirstGenerateLength{4};
    auto const firstPrompt = makePrompt(kPagedPromptLength);
    requestEngine->submit(makeGreedyRequest(firstPrompt, kFirstGenerateLength)).get();
    auto const published = requestEngine->contextCacheMetrics();
    ASSERT_TRUE(published.has_value());
    EXPECT_GT(published->committedPublications, 0U)
        << "the prompt's response was released before the counters saw the step that published its prefix";
    auto const continuation = firstPrompt + std::string(kFirstGenerateLength, 'a');
    rt::LLMGenerationResponse const second
        = requestEngine->submit(makeGreedyRequest(continuation, /*maxGenerateLength=*/1)).get();
    auto const served = requestEngine->contextCacheMetrics();

    requestEngine->shutdown(rt::scheduler::ShutdownMode::kDrain);
    auto const lastPrefill = std::find_if(calls().rbegin(), calls().rend(), isPrefill);
    ASSERT_NE(lastPrefill, calls().rend()) << "the continuation never reached a base prefill forward";
    int64_t const skipped = second.inputTokenCounts.at(0) - lastPrefill->dims.seqLen;
    ASSERT_GT(skipped, 0) << "the continuation never reused anything, so this proves nothing";

    ASSERT_TRUE(served.has_value());
    EXPECT_EQ(served->admittedSequences, 2U);
    EXPECT_EQ(served->hitSequences, 1U);
    EXPECT_EQ(served->reusedTokens, static_cast<uint64_t>(skipped));
}

// Given a continuation of a prompt whose state the context cache already holds
// When the continuation is served through the request engine with a stream attached
// Then its response and its stream both report as reused exactly the prompt tokens its prefill forward skipped
TEST_F(PlainContextReuseTest, ReusedPromptCountIsThePrefillTheEngineSkipped)
{
    auto requestEngine = makeRequestEngine(makeEngine());

    constexpr int64_t kFirstGenerateLength{4};
    auto const firstPrompt = makePrompt(kPagedPromptLength);
    requestEngine->submit(makeGreedyRequest(firstPrompt, kFirstGenerateLength)).get();
    auto request = makeGreedyRequest(firstPrompt + std::string(kFirstGenerateLength, 'a'), /*maxGenerateLength=*/1);
    auto const channel = rt::StreamChannel::create();
    request.streamChannels = {channel};
    rt::LLMGenerationResponse const response = requestEngine->submit(std::move(request)).get();

    requestEngine->shutdown(rt::scheduler::ShutdownMode::kDrain);
    auto const lastPrefill = std::find_if(calls().rbegin(), calls().rend(), isPrefill);
    ASSERT_NE(lastPrefill, calls().rend()) << "the continuation never reached a base prefill forward";
    int64_t const skipped = response.inputTokenCounts.at(0) - lastPrefill->dims.seqLen;
    ASSERT_GT(skipped, 0) << "the continuation never reused anything, so this proves nothing";

    ASSERT_EQ(response.reusedInputTokenCounts.size(), 1U);
    EXPECT_EQ(response.reusedInputTokenCounts[0], skipped);
    int32_t streamed{-1};
    while (auto chunk = channel->tryPop())
    {
        if (chunk->reusedPromptTokenCount >= 0)
        {
            streamed = chunk->reusedPromptTokenCount;
        }
    }
    EXPECT_EQ(streamed, skipped);
}

// Given a prompt served cold, then a continuation of it whose state the context cache now holds
// When both are served by the runtime's blocking path
// Then the cold prompt reports no reused tokens, and the continuation reports as reused exactly the prompt tokens its
// prefill forward skipped
TEST_F(PlainContextReuseTest, BlockingResponseReportsTheReusedPromptTokens)
{
    auto artifacts = makeVanillaArtifacts(mModelDir, makeEngine(), mStream);
    rt::LLMInferenceRuntime runtime{std::move(artifacts), mModelDir.string(), /*multimodalEngineDir=*/"", {},
        std::nullopt, mStream, makeCacheConfig()};

    constexpr int64_t kFirstGenerateLength{4};
    auto const firstPrompt = makePrompt(kPagedPromptLength);
    rt::LLMGenerationResponse cold;
    ASSERT_TRUE(runtime.handleRequest(makeGreedyRequest(firstPrompt, kFirstGenerateLength), cold, mStream));
    rt::LLMGenerationResponse warm;
    ASSERT_TRUE(runtime.handleRequest(
        makeGreedyRequest(firstPrompt + std::string(kFirstGenerateLength, 'a'), /*maxGenerateLength=*/1), warm,
        mStream));

    auto const lastPrefill = std::find_if(calls().rbegin(), calls().rend(), isPrefill);
    ASSERT_NE(lastPrefill, calls().rend()) << "the continuation never reached a base prefill forward";
    int64_t const skipped = warm.inputTokenCounts.at(0) - lastPrefill->dims.seqLen;
    ASSERT_GT(skipped, 0) << "the continuation never reused anything, so this proves nothing";

    EXPECT_EQ(cold.reusedInputTokenCounts, std::vector<int32_t>{0});
    ASSERT_EQ(warm.reusedInputTokenCounts.size(), 1U);
    EXPECT_EQ(warm.reusedInputTokenCounts[0], skipped);
}

// Given a stream that collects metrics
// When a request is served through the request engine
// Then the stream's metrics place the start of its prefill on the caller's steady clock, after the submit and
// before the response is released
TEST_F(PlainContextReuseTest, StreamMetricsPlaceThePrefillStartOnTheCallersSteadyClock)
{
    auto requestEngine = makeRequestEngine(makeEngine());
    auto const steadyNanoseconds = [] {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
            .count();
    };

    auto request = makeGreedyRequest(makePrompt(kPagedPromptLength), /*maxGenerateLength=*/2);
    auto const channel = rt::StreamChannel::create(/*collectMetrics=*/true);
    request.streamChannels = {channel};
    int64_t const submittedAt = steadyNanoseconds();
    requestEngine->submit(std::move(request)).get();
    int64_t const releasedAt = steadyNanoseconds();

    int64_t prefillStart{-1};
    while (auto chunk = channel->tryPop())
    {
        prefillStart = std::max(prefillStart, chunk->metrics.prefillStartNs);
    }
    EXPECT_GE(prefillStart, submittedAt);
    EXPECT_LE(prefillStart, releasedAt);
}

TEST_F(PlainContextReuseTest, DisablingReuseKeepsEveryPrefillAtFullLength)
{
    auto artifacts = makeVanillaArtifacts(mModelDir, makeEngine(), mStream);
    rt::LLMInferenceRuntime runtime{std::move(artifacts), mModelDir.string(), /*multimodalEngineDir=*/"", {},
        std::nullopt, mStream, rt::ContextCacheConfig{}}; // disabled (default)

    constexpr int64_t kFirstGenerateLength{4};
    constexpr int64_t kTotalLength{kPagedPromptLength + kFirstGenerateLength};
    auto const firstPrompt = makePrompt(kPagedPromptLength);
    auto firstRequest = makeGreedyRequest(firstPrompt, kFirstGenerateLength);
    rt::LLMGenerationResponse firstResponse;
    ASSERT_TRUE(runtime.handleRequest(firstRequest, firstResponse, mStream));

    EXPECT_FALSE(runtime.getContextCacheMetrics().has_value()) << "metrics should not exist for a disabled cache";

    auto secondRequest = makeGreedyRequest(firstPrompt + std::string(kFirstGenerateLength, 'a'),
        /*maxGenerateLength=*/1);
    rt::LLMGenerationResponse secondResponse;
    ASSERT_TRUE(runtime.handleRequest(secondRequest, secondResponse, mStream));

    // Both requests' prefills should be full-length with no cache to shrink against -- find the *last* prefill
    // (request 2's), since request 1's own prefill also matches the predicate.
    auto const lastPrefill = std::find_if(calls().rbegin(), calls().rend(), isPrefill);
    ASSERT_NE(lastPrefill, calls().rend());
    EXPECT_EQ(lastPrefill->dims.seqLen, kTotalLength);
}

// --------------------------------------------------------------------------
// ContextCacheCommitPolicy: kIncludingGeneratedTokens (the default -- SecondRequestReusesTheFirstRequestsPrefix
// above never overrides it) vs kPrefillStateOnly. The two tests above already exercise the default policy, but
// their prompt alone already spans a full page, so a hit there is explainable purely by the *prefill* publish --
// it doesn't distinguish "generated tokens got published too" from "only the prefill did". These two tests use a
// sub-page prompt plus a generation long enough to cross the page boundary, so only a decode-end publish (gated by
// ContextCacheCommitPolicy) can explain a hit; kPrefillStateOnly has nothing else to fall back on.
// --------------------------------------------------------------------------

//! Deliberately under one page (kTOKENS_PER_PAGE == 128) on its own -- only generation can push a sequence across
//! the first page boundary, so a hit downstream isolates decode-end publication specifically.
constexpr int64_t kSubPagePromptLength{100};

TEST_F(PlainContextReuseTest, DecodeEndPublicationLetsGeneratedTokensBeReusedAcrossThreeTurns)
{
    auto artifacts = makeVanillaArtifacts(mModelDir, makeEngine(), mStream);
    rt::LLMInferenceRuntime runtime{std::move(artifacts), mModelDir.string(), /*multimodalEngineDir=*/"", {},
        std::nullopt, mStream, makeCacheConfig()};

    // Turn 1: prompt (100) + generated (40) = 140, crossing the 128-token page boundary. contextCacheCommitPolicy
    // defaults to kIncludingGeneratedTokens (llmRuntimeUtils.h) -- not set explicitly, so this is what a real
    // client gets without opting into the narrower policy.
    constexpr int64_t kGen1{40};
    auto const turn1Prompt = makePrompt(kSubPagePromptLength);
    auto turn1 = makeGreedyRequest(turn1Prompt, kGen1);
    rt::LLMGenerationResponse resp1;
    ASSERT_TRUE(runtime.handleRequest(turn1, resp1, mStream));
    auto const afterTurn1 = runtime.getContextCacheMetrics();
    ASSERT_TRUE(afterTurn1.has_value());
    EXPECT_EQ(afterTurn1->hitSequences, 0U) << "nothing existed yet for turn 1 to reuse";

    // Turn 2: literal continuation of turn 1's exact tokens, + 20 more generated.
    constexpr int64_t kLen2{kSubPagePromptLength + kGen1};
    constexpr int64_t kGen2{20};
    auto turn2 = makeGreedyRequest(turn1Prompt + std::string(kGen1, 'a'), kGen2);
    rt::LLMGenerationResponse resp2;
    size_t const callsBeforeTurn2 = calls().size();
    ASSERT_TRUE(runtime.handleRequest(turn2, resp2, mStream));
    auto const afterTurn2 = runtime.getContextCacheMetrics();
    ASSERT_TRUE(afterTurn2.has_value());
    EXPECT_GT(afterTurn2->hitSequences, afterTurn1->hitSequences)
        << "turn 2's prefix, which only exists because turn 1's *generated* tail was published at decode-end, "
           "should hit";
    EXPECT_GT(afterTurn2->reusedTokens, 0U);
    auto const turn2Prefill
        = std::find_if(calls().begin() + static_cast<std::ptrdiff_t>(callsBeforeTurn2), calls().end(), isPrefill);
    ASSERT_NE(turn2Prefill, calls().end());
    EXPECT_LT(turn2Prefill->dims.seqLen, kLen2);

    // Turn 3: literal continuation of turn 2's exact tokens, + 20 more. `reusedTokens` is a lifetime-cumulative
    // counter (contextCacheMetrics.h), not a per-request figure, so this only shows turn 3 *also* achieved reuse
    // (the counter keeps climbing) -- it does not by itself prove turn 3 reused a larger span than turn 2 did. The
    // shrunk prefill length below is the unambiguous per-request signal for that.
    constexpr int64_t kLen3{kLen2 + kGen2};
    auto turn3 = makeGreedyRequest(turn1Prompt + std::string(kGen1 + kGen2, 'a'), /*maxGenerateLength=*/1);
    rt::LLMGenerationResponse resp3;
    size_t const callsBeforeTurn3 = calls().size();
    ASSERT_TRUE(runtime.handleRequest(turn3, resp3, mStream));
    auto const afterTurn3 = runtime.getContextCacheMetrics();
    ASSERT_TRUE(afterTurn3.has_value());
    EXPECT_GT(afterTurn3->reusedTokens, afterTurn2->reusedTokens) << "turn 3 should also have reused something";
    auto const turn3Prefill
        = std::find_if(calls().begin() + static_cast<std::ptrdiff_t>(callsBeforeTurn3), calls().end(), isPrefill);
    ASSERT_NE(turn3Prefill, calls().end());
    EXPECT_LT(turn3Prefill->dims.seqLen, kLen3)
        << "turn 3's continuation of turn 2 should also fold out of the prefill";
}

TEST_F(PlainContextReuseTest, PrefillStateOnlyPublishesNothingWhenThePromptAloneIsUnderOnePage)
{
    auto artifacts = makeVanillaArtifacts(mModelDir, makeEngine(), mStream);
    rt::LLMInferenceRuntime runtime{std::move(artifacts), mModelDir.string(), /*multimodalEngineDir=*/"", {},
        std::nullopt, mStream, makeCacheConfig()};

    // Same shape as the decode-end test above (100-token prompt, 40 generated, crossing the page boundary), but
    // with decode-end publication suppressed. The prefill alone is sub-page, so kPrefillEnd has nothing to publish
    // either -- with this policy, nothing is ever published for this request.
    constexpr int64_t kGen1{40};
    auto const turn1Prompt = makePrompt(kSubPagePromptLength);
    auto turn1 = makeGreedyRequest(turn1Prompt, kGen1);
    turn1.contextCacheCommitPolicy = rt::ContextCacheCommitPolicy::kPrefillStateOnly;
    rt::LLMGenerationResponse resp1;
    ASSERT_TRUE(runtime.handleRequest(turn1, resp1, mStream));
    auto const afterTurn1 = runtime.getContextCacheMetrics();
    ASSERT_TRUE(afterTurn1.has_value());
    EXPECT_EQ(afterTurn1->publishedEndpoints, 0U)
        << "kPrefillStateOnly suppresses the decode-end publish that would otherwise cover the generated tail, and "
           "the prefill alone is under one page -- nothing should have been published";

    constexpr int64_t kLen2{kSubPagePromptLength + kGen1};
    auto turn2 = makeGreedyRequest(turn1Prompt + std::string(kGen1, 'a'), /*maxGenerateLength=*/1);
    rt::LLMGenerationResponse resp2;
    size_t const callsBeforeTurn2 = calls().size();
    ASSERT_TRUE(runtime.handleRequest(turn2, resp2, mStream));
    auto const afterTurn2 = runtime.getContextCacheMetrics();
    ASSERT_TRUE(afterTurn2.has_value());
    EXPECT_EQ(afterTurn2->hitSequences, 0U) << "there was nothing published for turn 2 to reuse";
    auto const turn2Prefill
        = std::find_if(calls().begin() + static_cast<std::ptrdiff_t>(callsBeforeTurn2), calls().end(), isPrefill);
    ASSERT_NE(turn2Prefill, calls().end());
    EXPECT_EQ(turn2Prefill->dims.seqLen, kLen2) << "no reuse means the full prompt is re-prefilled";
}

//! One `handleRequest()` call, two slots with different reuse status: slot 0 is fresh and sub-page (never has
//! anything to reuse, by construction -- see kSubPagePromptLength's comment above), slot 1 is a literal
//! continuation of an earlier, separately-published request. `ContextCacheBatchAdmission` admits the whole batch
//! together (see contextCacheCoordinator.h), so this is the one place that actually exercises per-sequence
//! admission instead of the single-slot path every other test in this file uses.
TEST_F(PlainContextReuseTest, MixedBatchReusesOnlyTheSlotWithAPublishedPrefix)
{
    auto artifacts = makeVanillaArtifacts(mModelDir, makeEngine(), mStream);
    rt::LLMInferenceRuntime runtime{std::move(artifacts), mModelDir.string(), /*multimodalEngineDir=*/"", {},
        std::nullopt, mStream, makeCacheConfig()};

    constexpr int64_t kFirstGenerateLength{4};
    constexpr int64_t kTotalLength{kPagedPromptLength + kFirstGenerateLength};
    auto const firstPrompt = makePrompt(kPagedPromptLength);
    auto firstRequest = makeGreedyRequest(firstPrompt, kFirstGenerateLength);
    rt::LLMGenerationResponse firstResponse;
    ASSERT_TRUE(runtime.handleRequest(firstRequest, firstResponse, mStream));
    auto const afterFirst = runtime.getContextCacheMetrics();
    ASSERT_TRUE(afterFirst.has_value());

    // Slot 0: a fresh, sub-page prompt -- structurally unable to reuse a whole 128-token page no matter what its
    // content is. Slot 1: the exact continuation of firstRequest's tokens, published above.
    constexpr int64_t kColdSlotLength{50};
    size_t const callsBeforeMixed = calls().size();

    rt::LLMGenerationRequest mixedRequest{};
    rt::LLMGenerationRequest::Request coldSlot;
    coldSlot.messages.push_back(
        rt::Message{"user", {rt::Message::MessageContent{"text", makePrompt(kColdSlotLength, /*variant=*/1)}}});
    rt::LLMGenerationRequest::Request warmSlot;
    warmSlot.messages.push_back(rt::Message{
        "user", {rt::Message::MessageContent{"text", firstPrompt + std::string(kFirstGenerateLength, 'a')}}});
    mixedRequest.requests.push_back(std::move(coldSlot));
    mixedRequest.requests.push_back(std::move(warmSlot));
    mixedRequest.temperature = 0.0F;
    mixedRequest.topK = 1;
    mixedRequest.topP = 1.0F;
    mixedRequest.maxGenerateLength = 1;

    rt::LLMGenerationResponse mixedResponse;
    ASSERT_TRUE(runtime.handleRequest(mixedRequest, mixedResponse, mStream));
    ASSERT_EQ(mixedResponse.outputIds.size(), 2U);
    EXPECT_EQ(mixedResponse.outputIds[0].size(), 1U);
    EXPECT_EQ(mixedResponse.outputIds[1].size(), 1U);

    auto const afterMixed = runtime.getContextCacheMetrics();
    ASSERT_TRUE(afterMixed.has_value());
    EXPECT_EQ(afterMixed->admittedSequences - afterFirst->admittedSequences, 2U)
        << "both slots of the batch must be admitted, not just one";
    EXPECT_EQ(afterMixed->hitSequences - afterFirst->hitSequences, 1U)
        << "exactly one of the two slots should reuse -- a batch-wide hit/miss would mask per-sequence admission";
    EXPECT_GT(afterMixed->reusedTokens - afterFirst->reusedTokens, 0U);
    EXPECT_GE(afterMixed->noReusablePrefixPlans - afterFirst->noReusablePrefixPlans, 1U)
        << "the cold slot should be evaluated and found to have no reusable prefix, not silently skipped";

    // The metrics above are computed from the reuse plan, while the work is driven by what reaches the prefill
    // path -- two separate routes. Accounting a hit while still prefilling the matched tokens would satisfy every
    // assertion above, so the forward itself has to be checked. Under this deployment's entry padding the width is
    // `batch * longest entry`, which is what both bounds are expressed against.
    auto const mixedPrefill
        = std::find_if(calls().begin() + static_cast<std::ptrdiff_t>(callsBeforeMixed), calls().end(), isPrefill);
    ASSERT_NE(mixedPrefill, calls().end()) << "the mixed batch never reached a prefill forward";
    EXPECT_LT(mixedPrefill->dims.seqLen, 2 * kTotalLength)
        << "the warm slot still drove the width, so its published page was not folded out of the shared forward";
    EXPECT_GE(mixedPrefill->dims.seqLen, 2 * kColdSlotLength)
        << "the cold slot published nothing, so all of its tokens must still be in the shared forward";
}

//! An input limit of one page plus a little under a KV capacity of four pages, over a pool of eight pages: a prompt
//! can fit the KV cache yet be too long to prefill without a reused prefix.
Json makeShortInputContextReuseConfig()
{
    Json config = makeContextReusePagedConfig();
    config["builder_config"]["max_input_len"] = rt::kTOKENS_PER_PAGE + 8;
    config["builder_config"]["max_kv_cache_capacity"] = 4 * rt::kTOKENS_PER_PAGE;
    config["builder_config"]["max_kv_pool_pages"]
        = rt::computeMinimumKvPoolPages(kMaxBatchSize, 4 * rt::kTOKENS_PER_PAGE);
    return config;
}

class ShortInputContextReuseTest : public PlainContextReuseTest
{
protected:
    void SetUp() override
    {
        CUDA_CHECK(cudaStreamCreate(&mStream));
        mModelDir = writeModelDir("shortInputContextReuseTests", makeShortInputContextReuseConfig());
    }
};

// Given a context cache with three of its eight pages free, the least recently used conversation's two published
//      pages next in line for eviction
// When a cold prompt that fits the KV capacity but whose uncached part exceeds the input limit is served on the
//      blocking path
// Then it is refused with the marker the server maps to 413 before anything is evicted, so that conversation's next
//      turn still skips both pages
TEST_F(ShortInputContextReuseTest, AnOverlongPromptIsRefusedBeforeEvictingAnotherConversation)
{
    auto artifacts = makeVanillaArtifacts(mModelDir, makeEngine(), mStream);
    rt::LLMInferenceRuntime runtime{std::move(artifacts), mModelDir.string(), /*multimodalEngineDir=*/"", {},
        std::nullopt, mStream, makeCacheConfig()};
    auto const serve = [&](int64_t length, int32_t variant) {
        rt::LLMGenerationResponse response;
        return runtime.handleRequest(
            makeGreedyRequest(makePrompt(length, variant), /*maxGenerateLength=*/1), response, mStream);
    };

    constexpr int64_t kPage{rt::kTOKENS_PER_PAGE};
    // Every turn's uncached part fits the input limit: conversation 1 publishes two pages, then conversation 2 three.
    ASSERT_TRUE(serve(kPage + 2, 1));
    ASSERT_TRUE(serve(2 * kPage + 2, 1));
    for (int64_t pages = 1; pages <= 3; ++pages)
    {
        ASSERT_TRUE(serve(pages * kPage + 2, 2));
    }
    auto const before = runtime.getContextCacheMetrics();
    ASSERT_TRUE(before.has_value());

    try
    {
        serve(3 * kPage + 16, 3);
        ADD_FAILURE() << "a prompt whose uncached part exceeds the input limit was served";
    }
    catch (std::runtime_error const& error)
    {
        EXPECT_NE(std::string(error.what()).find("EDGELLM_INPUT_TOO_LONG"), std::string::npos) << error.what();
    }
    EXPECT_EQ(runtime.getContextCacheMetrics()->evictedRecords, before->evictedRecords);

    size_t const callsBeforeReturn = calls().size();
    ASSERT_TRUE(serve(2 * kPage + 7, 1));
    auto const returnPrefill
        = std::find_if(calls().begin() + static_cast<std::ptrdiff_t>(callsBeforeReturn), calls().end(), isPrefill);
    ASSERT_NE(returnPrefill, calls().end());
    EXPECT_EQ(returnPrefill->dims.seqLen, 7) << "conversation 1's published pages were evicted";
}

} // namespace

// ---------------------------------------------------------------------------
// Shutdown ordering under leases: the one path whose failure mode is
// std::terminate. An engine owning a context-cache runtime shuts down with a
// resident decoding, a second admission holding a lease, and a request
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
    // (prompt + 12 + headroom) each, so the second lease is grantable while the first runs.
    // A larger ask never fails loudly -- the joiner waits on kNoCapacity forever, by design.
    // The assembly tokenizer is an empty shell, so every request carries pre-tokenized input --
    // the path admitRequest prefers anyway.
    auto const makeTokenizedRequest = [](std::vector<int32_t> tokenIds, int64_t maxGenerateLength) {
        rt::LLMGenerationRequest request = makeGreedyRequest("x", maxGenerateLength);
        request.preTokenizedInputIds = {std::move(tokenIds)};
        return request;
    };

    auto const deadline = std::chrono::steady_clock::now() + std::chrono::seconds(20);
    rt::scheduler::RequestHandle first = requestEngine.submit(makeTokenizedRequest({41}, 12));
    while (requestEngine.resident() == 0)
    {
        ASSERT_LT(std::chrono::steady_clock::now(), deadline) << "the first request never became resident";
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
    EXPECT_TRUE(first.ready());
    EXPECT_TRUE(joiner.ready());
    EXPECT_TRUE(queued.ready());

    // Scope exit destroys the engine, then the runtime, then the coordinator, whose destructor
    // proves quiescence or terminates. Reaching the end of this test is the assertion.
}

TEST_F(RuntimeAssemblyTest, EngineDrivesSessionEndToEnd)
{
    using ::testing::_;

    // The full stack, no hook anywhere: RequestEngine's actor owns the loop, the runtime session
    // executes the steps it schedules, and both callers get their serial token streams back
    // through handles.
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
        EXPECT_CALL(mock, execute(_)).WillOnce(emit({kA1})); // A's context step
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
        EXPECT_CALL(mock, execute(_)).WillOnce(emit({kB1})); // B's context step
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
    EXPECT_EQ(metrics.admittedMidFlight, 1U) << "B must have joined A in the running session";
    EXPECT_EQ(metrics.failed, 0U);
}

// Given a two-sequence request with ample token budget and one-round benchmark control
// When the same runtime handles that request twice
// Then each response contains the prefill token plus one decode token, and both requests finish normally.
TEST_F(RuntimeAssemblyTest, BenchmarkRoundBudgetFinishesAndReleasesEverySequence)
{
    rt::DeploymentOptions options{};
    options.engineDir = mModelDir.string();
    options.benchmark.maxDecodeRounds = 1;
    auto artifacts = makeVanillaArtifacts(mModelDir, makeEngine(), mStream);
    rt::LLMInferenceRuntime runtime{std::move(artifacts), options, mStream};
    auto request = makeGreedyRequest("aa", 8);
    request.requests.push_back(request.requests.front());
    request.ignoreEos = true;
    for (int32_t iteration = 0; iteration < 2; ++iteration)
    {
        rt::LLMGenerationResponse response;
        ASSERT_TRUE(runtime.handleRequest(request, response, mStream));
        ASSERT_EQ(response.outputIds.size(), 2U);
        EXPECT_EQ(response.outputIds[0].size(), 2U);
        EXPECT_EQ(response.outputIds[1].size(), 2U);
        EXPECT_EQ(response.decodeRoundCounts, (std::vector<int32_t>{1, 1}));
        EXPECT_EQ(response.finishReasons,
            (std::vector<rt::FinishReason>{rt::FinishReason::kLength, rt::FinishReason::kLength}));
    }
}

// Given default experiment controls and a request ignoring EOS
// When a normal runtime handles the request
// Then the full token budget is generated, rather than stopping after one round.
TEST_F(RuntimeAssemblyTest, DefaultBenchmarkOptionsPreserveGenerationBudget)
{
    auto runtime = makeRuntime(makeVanillaArtifacts(mModelDir, makeEngine(), mStream));
    auto request = makeGreedyRequest("aa", 5);
    request.ignoreEos = true;
    rt::LLMGenerationResponse response;
    ASSERT_TRUE(runtime.handleRequest(request, response, mStream));
    EXPECT_EQ(response.outputIds[0].size(), 5U);
    EXPECT_EQ(response.decodeRoundCounts, std::vector<int32_t>{4});
}

namespace
{
struct ScopedAssemblyProfiling
{
    bool previous{getProfilingEnabled()};
    ScopedAssemblyProfiling()
    {
        setProfilingEnabled(true);
    }
    ~ScopedAssemblyProfiling()
    {
        setProfilingEnabled(previous);
    }
};

int64_t selectedStageCount(std::string const& stage)
{
    auto const data = gTimer.getTimingData(stage);
    return data ? data->getTotalRuns() : 0;
}
} // namespace

TEST_F(RuntimeAssemblyTest, ProfilingStorageFollowsKvClampedGenerationLength)
{
    // Given a requested output far beyond KV capacity, when a profiled request runs, then timing storage is sized
    // from the KV-clamped generation length rather than the requested one.
    auto runtime = makeRuntime(makeVanillaArtifacts(mModelDir, makeEngine(), mStream));
    ScopedAssemblyProfiling profiling;
    auto request = makeGreedyRequest("a", 4096);
    request.ignoreEos = true;
    rt::LLMGenerationResponse response;
    ASSERT_TRUE(runtime.handleRequest(request, response, mStream));
    int64_t constexpr kMaxKvCacheCapacity = 64;
    int64_t constexpr kVanillaSamplesPerRound = 8;
    EXPECT_LE(gTimer.capacity(), 256 + kMaxKvCacheCapacity * kVanillaSamplesPerRound);
}

TEST_F(RuntimeAssemblyTest, DisabledGpuTimingProfilesWithoutTimingEvents)
{
    // Given a runtime configured without GPU timing, when a profiled request runs, then it records no stage timings
    // and creates no timing events.
    rt::DeploymentOptions options{};
    options.engineDir = mModelDir.string();
    options.benchmark.gpuTiming = false;
    rt::LLMInferenceRuntime runtime{makeVanillaArtifacts(mModelDir, makeEngine(), mStream), options, mStream};
    ScopedAssemblyProfiling profiling;
    auto request = makeGreedyRequest("a", 4);
    request.ignoreEos = true;
    rt::LLMGenerationResponse response;
    ASSERT_TRUE(runtime.handleRequest(request, response, mStream));
    EXPECT_EQ(response.outputIds[0].size(), 4U);
    EXPECT_TRUE(gTimer.getSamples().empty());
    EXPECT_EQ(gTimer.capacity(), 0U);
}

TEST_F(RuntimeAssemblyTest, ProfilesRemainOwnedByAlternatingRuntimesAndCapture)
{
    // Given two runtimes on one thread, when requests and capture calls alternate, then each profile retains only its
    // own work.
    auto runtimeA = makeRuntime(makeVanillaArtifacts(mModelDir, makeEngine(), mStream));
    auto runtimeB = makeRuntime(makeVanillaArtifacts(mModelDir, makeEngine(), mStream));
    ScopedAssemblyProfiling profiling;
    auto requestA = makeGreedyRequest("a", 2);
    auto requestB = makeGreedyRequest("aa", 4);
    requestA.ignoreEos = true;
    requestB.ignoreEos = true;
    rt::LLMGenerationResponse response;
    ASSERT_TRUE(runtimeA.handleRequest(requestA, response, mStream));
    EXPECT_EQ(selectedStageCount("llm_prefill"), 1);
    EXPECT_EQ(selectedStageCount("decode_round"), 1);
    ASSERT_TRUE(runtimeB.handleRequest(requestB, response, mStream));
    EXPECT_EQ(selectedStageCount("decode_round"), 3);
    static_cast<void>(runtimeA.captureDecodingCUDAGraph(mStream));
    EXPECT_EQ(selectedStageCount("decode_round"), 1);
    requestA.maxGenerateLength = 4;
    ASSERT_TRUE(runtimeA.handleRequest(requestA, response, mStream));
    EXPECT_EQ(selectedStageCount("decode_round"), 4);
    static_cast<void>(runtimeB.captureDecodingCUDAGraph(mStream));
    EXPECT_EQ(selectedStageCount("decode_round"), 3);
}

// Given a one-round experiment whose requested prompt leaves no room for a full round
// When runtime capacity would clamp the output budget
// Then the experiment fails rather than reporting a shortened round, and a later valid request succeeds.
TEST_F(RuntimeAssemblyTest, BenchmarkRoundRejectsCapacityTruncation)
{
    auto config = makeTinyVanillaConfig();
    config["builder_config"]["max_input_len"] = 64;
    std::ofstream(mModelDir / "config.json") << config.dump();
    rt::DeploymentOptions options{};
    options.engineDir = mModelDir.string();
    options.benchmark.maxDecodeRounds = 1;
    auto artifacts = makeVanillaArtifacts(mModelDir, makeEngine(), mStream);
    rt::LLMInferenceRuntime runtime{std::move(artifacts), options, mStream};
    auto request = makeGreedyRequest("a", 8);
    request.preTokenizedInputIds = {std::vector<int32_t>(62, 0)};
    request.ignoreEos = true;
    rt::LLMGenerationResponse response;
    EXPECT_FALSE(runtime.handleRequest(request, response, mStream));
    request = makeGreedyRequest("a", 8);
    request.ignoreEos = true;
    ASSERT_TRUE(runtime.handleRequest(request, response, mStream));
    EXPECT_EQ(response.decodeRoundCounts, std::vector<int32_t>{1});
    EXPECT_EQ(response.outputIds[0].size(), 2U);
}

TEST_F(RuntimeAssemblyTest, RuntimeForwardTimingIncludesPreparationAndExcludesSampling)
{
    // Given an executor without internal timing, when runtime generates one decode token,
    // then prepare/execute belong to each shaped forward while sampled-token handling stays outside it.
    using ::testing::_;
    auto engine = makeEngine();
    ON_CALL(*engine, prepare(_, _, _, _))
        .WillByDefault([this](int32_t, rt::InferenceDims const&, rt::TensorMap const& map, cudaStream_t stream) {
            auto observation = gTimer.startStage("observed_prepare", stream);
            mLogits = map.get(binding_names::kLogits);
            return true;
        });
    ON_CALL(*engine, execute(_)).WillByDefault([this](cudaStream_t stream) {
        auto observation = gTimer.startStage("observed_execute", stream);
        writeLogits(*mLogits, kVocabSize, {}, stream);
        return true;
    });
    auto artifacts = makeVanillaArtifacts(mModelDir, std::move(engine), mStream);
    auto tokenizer = std::move(artifacts.tokenizer);
    chat_template::ChatTemplate chatTemplate;
    rt::DeploymentOptions options{};
    options.engineDir = mModelDir.string();
    rt::LLMRankRuntime runtime{std::move(artifacts), options, mStream, rt::ParallelMapping{}, *tokenizer, chatTemplate};
    auto request = makeGreedyRequest("a", 2);
    request.applyChatTemplate = false;
    request.preTokenizedInputIds = {{42, 43, 44}};
    request.requests.front().messages.clear();
    request.formattedRequests.resize(request.requests.size());
    request.ignoreEos = true;
    ScopedAssemblyProfiling profiling;
    rt::LLMGenerationResponse response;
    ASSERT_TRUE(runtime.handleRequest(
        request, response, mStream, false,
        [](void*, int32_t, cudaStream_t stream) {
            auto observation = gTimer.startStage("observed_after_sampling", stream);
            return true;
        },
        0));
    using Forward = std::tuple<std::string, std::string, int32_t, int32_t, int64_t, int64_t, int64_t>;
    std::vector<Forward> forwards;
    std::vector<std::string> preparedParents;
    std::vector<std::string> executedParents;
    std::vector<std::string> sampledParents;
    for (auto const& sample : gTimer.getSamples())
    {
        if (sample.forward)
        {
            forwards.emplace_back(sample.stageId, sample.parentStage, sample.round, sample.forwardIndex,
                sample.batchSize, sample.queryTokens, sample.selectedTokens);
        }
        if (sample.stageId == "observed_prepare")
        {
            preparedParents.push_back(sample.parentStage);
        }
        if (sample.stageId == "observed_execute")
        {
            executedParents.push_back(sample.parentStage);
        }
        if (sample.stageId == "observed_after_sampling" && sample.round == 0)
        {
            sampledParents.push_back(sample.parentStage);
        }
    }
    EXPECT_EQ(forwards,
        (std::vector<Forward>{
            {"forward/base", "llm_prefill", -1, 0, 1, 3, 1}, {"forward/base", "llm_generation", 0, 0, 1, 1, 1}}));
    EXPECT_EQ(preparedParents, (std::vector<std::string>{"forward/base", "forward/base"}));
    EXPECT_EQ(executedParents, (std::vector<std::string>{"forward/base", "forward/base"}));
    EXPECT_EQ(sampledParents, (std::vector<std::string>{"llm_generation"}));
}
