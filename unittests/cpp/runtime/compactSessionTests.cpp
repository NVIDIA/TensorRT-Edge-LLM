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

// Compact (phase-8) execution in the runtime session, driven against substitute engines. The fake engine computes
// every token from the sequence's history, so an output does not depend on how its prompt was split into chunks or
// which other sequences shared its steps.

#include "compactSessionDriver.h"
#include "fakeEngineExecutor.h"
#include "substituteEngine.h"

#include "common/cudaUtils.h"
#include "common/pagedKvTypes.h"
#include "common/trtUtils.h"
#include "runtime/config/compactCapability.h"
#include "runtime/llmInferenceRuntime.h"
#include "runtime/runtimeSession.h"
#include "scheduler/requestEngine.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <future>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace trt_edgellm;
using namespace substitute_engine;
using rt::CompactDriver;

namespace
{

constexpr int64_t kCompactBatchSize{4};
constexpr int64_t kMaxGenerateLength{4};
constexpr int64_t kPhase8{static_cast<int64_t>(rt::ExecutionPhase::kMixedPrefillDecode)};
constexpr int64_t kPhase3{static_cast<int64_t>(rt::ExecutionPhase::kAutoregressiveDecode)};
rt::SessionOptions const kCompactSession{/*allowCompactExecution=*/true};

//! @p config with layer shapes the phase-8 kernels cover: attention head dim 64 at GQA ratio 2 and, on a hybrid
//! config, K = V = 128 gated_delta_net layers; and built for compact execution (builder_config.compact_execution).
Json withCompactShapes(Json config)
{
    config["builder_config"]["compact_execution"] = true;
    config["head_dim"] = 64;
    config["num_attention_heads"] = 2 * config["num_key_value_heads"].get<int32_t>();
    if (config.value("num_linear_attn_layers", 0) > 0)
    {
        config["recurrent_state_head_dim"] = 128;
        config["recurrent_state_size"] = 128;
    }
    return config;
}

Json makeCompactVanillaConfig()
{
    Json config = makeTinyVanillaConfig();
    config["builder_config"]["max_batch_size"] = kCompactBatchSize;
    return withCompactShapes(std::move(config));
}

Json makeCompactHybridConfig()
{
    Json config = makeTinyHybridConfig();
    config["builder_config"]["max_batch_size"] = kCompactBatchSize;
    return withCompactShapes(std::move(config));
}

//! A fake engine for @p config whose prefill and decode profiles admit phase 8.
template <typename Engine = FakeEngineExecutor>
std::unique_ptr<Engine> makeCompactFake(Json const& config)
{
    auto engine = std::make_unique<Engine>(FakeLayerComposition::fromConfigJson(config));
    engine->advertisePhase8Profile();
    return engine;
}

rt::LLMGenerationRequest makeTokenizedRequest(
    std::vector<int32_t> tokenIds, int64_t maxGenerateLength = kMaxGenerateLength)
{
    rt::LLMGenerationRequest request = makeGreedyRequest("x", maxGenerateLength);
    request.preTokenizedInputIds = {std::move(tokenIds)};
    return request;
}

//! @p prompt with a generate budget of @p maxGenerateLength, generating through EOS so its length is the budget.
rt::LLMGenerationRequest budgetedRequest(std::vector<int32_t> prompt, int64_t maxGenerateLength)
{
    rt::LLMGenerationRequest request = makeTokenizedRequest(std::move(prompt), maxGenerateLength);
    request.ignoreEos = true;
    return request;
}

//! (token, logprob) of every top-logprob entry of every generated token.
std::vector<std::pair<int32_t, float>> logprobsOf(rt::LLMGenerationResponse const& response)
{
    std::vector<std::pair<int32_t, float>> entries;
    for (auto const& step : response.logprobs.at(0))
    {
        for (rt::LogprobEntry const& entry : step)
        {
            entries.emplace_back(entry.tokenId, entry.logprob);
        }
    }
    return entries;
}

std::vector<rt::ResidentRef> producers(rt::StepResult const& result)
{
    std::vector<rt::ResidentRef> refs;
    for (auto const& [ref, delta] : result.deltas)
    {
        EXPECT_EQ(delta.tokenIds.size(), 1U);
        refs.push_back(ref);
    }
    return refs;
}

std::vector<int32_t> promptOf(int32_t length, int32_t variant)
{
    return promptTokenIds(makePrompt(length, variant));
}

//! Put @p stream in error without a sticky device error, which would outlive the test: cudaStreamQuery reports an
//! error on a capturing stream. recoverStream() undoes it.
void putStreamInError(cudaStream_t stream)
{
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeRelaxed));
}

//! Undo what a failure left on @p stream: end the capture of putStreamInError(), clear a pending CUDA error.
void recoverStream(cudaStream_t stream)
{
    cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(stream, &capture) == cudaSuccess && capture != cudaStreamCaptureStatusNone)
    {
        cudaGraph_t graph = nullptr;
        static_cast<void>(cudaStreamEndCapture(stream, &graph));
        if (graph != nullptr)
        {
            static_cast<void>(cudaGraphDestroy(graph));
        }
    }
    static_cast<void>(cudaGetLastError());
}

//! A fake engine whose phase-8 forwards fail as phaseEightFailure says, and that runs afterForward after each forward:
//! between the step start, where a cancellation is first observed, and the step's commit.
class HookedFakeEngine : public FakeEngineExecutor
{
public:
    using FakeEngineExecutor::FakeEngineExecutor;

    enum class Failure : uint8_t
    {
        kNone,
        kPrepare,          //!< prepare() fails to bind the step.
        kRefusal,          //!< execute() fails with the device healthy, as when a plugin refuses phase 8.
        kPendingCudaError, //!< execute() fails leaving a non-sticky CUDA error pending.
        kStreamInError,    //!< execute() fails leaving the stream in error.
    };

    std::function<void()> afterForward;
    Failure phaseEightFailure{Failure::kNone};
    //! dedicateCompactContext() runs out of device memory, leaving a non-sticky error.
    bool failDedicatedContext{false};
    std::vector<int32_t> dedicatedProfiles;

    bool dedicateCompactContext(int32_t profileIndex, cudaStream_t stream) override
    {
        dedicatedProfiles.push_back(profileIndex);
        if (failDedicatedContext)
        {
            // Far beyond any device: refused without touching device memory.
            void* memory = nullptr;
            CUDA_CHECK(cudaMalloc(&memory, size_t{1} << 50));
        }
        return FakeEngineExecutor::dedicateCompactContext(profileIndex, stream);
    }

    bool prepare(
        int32_t profileIndex, rt::InferenceDims const& dims, rt::TensorMap const& map, cudaStream_t stream) override
    {
        if (phaseEightFailure == Failure::kPrepare && dims.executionPhaseLen == kPhase8)
        {
            return false;
        }
        return FakeEngineExecutor::prepare(profileIndex, dims, map, stream);
    }

    bool execute(cudaStream_t stream) override
    {
        bool const phaseEight = preparedRounds().back().dims.executionPhaseLen == kPhase8;
        Failure const failure = phaseEight ? phaseEightFailure : Failure::kNone;
        if (failure == Failure::kPendingCudaError)
        {
            // Far beyond any device: refused without touching device memory.
            void* memory = nullptr;
            static_cast<void>(cudaMalloc(&memory, size_t{1} << 50));
        }
        else if (failure == Failure::kStreamInError)
        {
            putStreamInError(stream);
        }
        bool const ok
            = (failure == Failure::kNone || failure == Failure::kPrepare) && FakeEngineExecutor::execute(stream);
        if (afterForward)
        {
            afterForward();
        }
        return ok;
    }
};

class CompactSessionTest : public ModelDirTest
{
protected:
    void SetUp() override
    {
        ModelDirTest::SetUp();
#if IS_TRT_RTX
        // LLMRankRuntime::compactPreconditionFailure refuses compact sessions on TensorRT-RTX.
        GTEST_SKIP() << "TensorRT-RTX engines may run on a GPU other than the one they were built for";
#endif
        std::string const gap
            = rt::compactExecutionGap(rt::parseEngineConfig(mModelDir / "config.json"), getSMVersion());
        if (!gap.empty())
        {
            GTEST_SKIP() << gap;
        }
    }

    std::filesystem::path stageModelDir() override
    {
        return writeModelDir("compactSessionTests", makeCompactVanillaConfig());
    }

    rt::LLMInferenceRuntime makeRuntimeOver(std::unique_ptr<rt::EngineExecutor> engine)
    {
        return makeRuntimeOver(std::move(engine), mModelDir);
    }

    rt::LLMInferenceRuntime makeRuntimeOver(
        std::unique_ptr<rt::EngineExecutor> engine, std::filesystem::path const& dir)
    {
        return rt::LLMInferenceRuntime{makeVanillaArtifacts(dir, std::move(engine), mStream), dir.string(),
            /*multimodalEngineDir=*/"", {}, std::nullopt, mStream};
    }

    //! A session allowing compact execution; throws, failing the test, unless it reports compact execution.
    std::unique_ptr<rt::RuntimeSession> openCompactSession(rt::LLMInferenceRuntime& runtime)
    {
        std::unique_ptr<rt::RuntimeSession> session = runtime.openSession(mStream, kCompactSession);
        if (session == nullptr || !session->capabilities().compact)
        {
            throw std::runtime_error(
                "no compact session: " + (session == nullptr ? "open failed" : session->capabilities().reason));
        }
        return session;
    }

    //! What the blocking path generates for @p request alone, on an engine that never runs phase 8.
    rt::LLMGenerationResponse blockingResponse(Json const& config, rt::LLMGenerationRequest const& request)
    {
        auto runtime = makeRuntimeOver(makeFakeEngine(config));
        rt::LLMGenerationResponse response;
        EXPECT_TRUE(runtime.handleRequest(request, response, mStream));
        return response;
    }

    std::vector<int32_t> blockingOutput(Json const& config, rt::LLMGenerationRequest const& request)
    {
        rt::LLMGenerationResponse const response = blockingResponse(config, request);
        return response.outputIds.empty() ? std::vector<int32_t>{} : response.outputIds.front();
    }

    std::vector<int32_t> blockingOutput(Json const& config, std::vector<int32_t> const& prompt)
    {
        return blockingOutput(config, makeTokenizedRequest(prompt));
    }

    //! Serves @p config, whose engine admits phase 8, by a RequestEngine under auto and under mixed: the auto engine
    //! must resolve to prefillFirst without compact execution, with @p reason in its policy reason, and serve A (three
    //! tokens) as the blocking path does without a phase-8 forward; the mixed engine must fail to start with @p reason.
    void expectAutoServesPrefillFirstAndMixedFails(Json const& config, std::string const& reason)
    {
        std::vector<int32_t> const promptA{40, 41, 42};
        std::vector<int32_t> const referenceA = blockingOutput(makeCompactVanillaConfig(), promptA);
        auto const dir = writeModelDir("compactSessionRejectedEngine", config);
        for (auto const mode : {rt::scheduler::IfbSchedulingMode::kAuto, rt::scheduler::IfbSchedulingMode::kMixed})
        {
            SCOPED_TRACE(rt::scheduler::toString(mode));
            auto engine = makeCompactFake(config);
            auto* const raw = engine.get();
            auto runtime = std::make_unique<rt::LLMInferenceRuntime>(
                makeVanillaArtifacts(dir, std::move(engine), mStream), dir.string(), /*multimodalEngineDir=*/"",
                std::unordered_map<std::string, std::string>{}, std::nullopt, mStream);
            rt::scheduler::EngineConfig engineConfig;
            engineConfig.maxBatchSize = kCompactBatchSize;
            engineConfig.scheduling.ifbSchedulingMode = mode;
            if (mode == rt::scheduler::IfbSchedulingMode::kMixed)
            {
                try
                {
                    rt::scheduler::RequestEngine requestEngine(std::move(runtime), mStream, engineConfig);
                    ADD_FAILURE() << "the mixed engine started";
                }
                catch (std::invalid_argument const& error)
                {
                    EXPECT_NE(std::string(error.what()).find(reason), std::string::npos) << error.what();
                }
                continue;
            }
            rt::scheduler::RequestEngine requestEngine(std::move(runtime), mStream, engineConfig);
            rt::scheduler::ResolvedSchedulingPolicy const policy = requestEngine.resolvedPolicy();
            EXPECT_EQ(policy.mode, rt::scheduler::IfbSchedulingMode::kPrefillFirst);
            EXPECT_FALSE(policy.compactExecution);
            EXPECT_NE(policy.reason.find(reason), std::string::npos) << policy.reason;
            EXPECT_EQ(requestEngine.submit(makeTokenizedRequest(promptA, kMaxGenerateLength)).get().outputIds.at(0),
                referenceA);
            for (auto const& round : raw->preparedRounds())
            {
                EXPECT_NE(round.dims.executionPhaseLen, kPhase8) << "a compact step ran on a refused deployment";
            }
        }
        std::filesystem::remove_all(dir);
    }
};

class CompactHybridSessionTest : public CompactSessionTest
{
protected:
    std::filesystem::path stageModelDir() override
    {
        return writeModelDir("compactHybridSessionTests", makeCompactHybridConfig());
    }
};

} // namespace

// Given a compact deployment with A decoding and two prompts P1 (7 tokens) and P2 (3 tokens) admitted, the three
//       with generate budgets of 4, 7 and 2 and top-2 logprobs, P2 with a logit bias that makes token 5 its best
// When one step runs A's decode row, P1's first 4 prompt tokens and P2's whole prompt, the next step
//      P1's last 3 tokens beside A's and P2's decode rows, and decode steps follow until all finish
// Then the first step is one forward of 8 tokens over 3 sequences on the prefill profile, prefill first, with one
//      logits row per sequence (the engine binds logits_indices by num_sequences) of which only A's and P2's are
//      sampled: A and P2 produce a token, P1 none; P1 produces its first token at its final chunk; and each sequence
//      stops at its own budget with FinishReason::kLength and the blocking path's tokens and logprobs for its request
//      alone
TEST_F(CompactSessionTest, MixedStepSamplesOnlyFinalChunksAndDecodeRows)
{
    std::vector<rt::LLMGenerationRequest> requests{budgetedRequest({40, 41}, 4),
        budgetedRequest({50, 51, 52, 53, 54, 55, 56}, 7), budgetedRequest({60, 61, 62}, 2)};
    for (rt::LLMGenerationRequest& request : requests)
    {
        request.numLogprobs = 2;
    }
    requests[2].requests.front().logitBias = {{5, 100.0F}};
    std::vector<rt::LLMGenerationResponse> references;
    for (rt::LLMGenerationRequest const& request : requests)
    {
        references.push_back(blockingResponse(makeCompactVanillaConfig(), request));
    }
    ASSERT_EQ(references[2].outputIds.front(), std::vector<int32_t>(2, 5)) << "the bias did not decide P2's tokens";
    auto engine = makeCompactFake(makeCompactVanillaConfig());
    auto* const raw = engine.get();
    auto runtime = makeRuntimeOver(std::move(engine));
    std::unique_ptr<rt::RuntimeSession> session = runtime.openSession(mStream, kCompactSession);
    ASSERT_NE(session, nullptr);
    CompactDriver driver(*session);
    rt::ResidentRef const a = driver.admit(requests[0], 1);
    ASSERT_TRUE(driver.run({driver.rest(a)}).ok);
    ASSERT_TRUE(driver.run({driver.decode(a)}).ok);
    rt::ResidentRef const p1 = driver.admit(requests[1], 2);
    rt::ResidentRef const p2 = driver.admit(requests[2], 3);

    size_t const forwardsBefore = raw->forwardsRun();
    rt::StepResult const mixed = driver.run({driver.decode(a), driver.chunk(p1, 4), driver.rest(p2)});
    ASSERT_TRUE(mixed.ok);
    EXPECT_EQ(raw->forwardsRun(), forwardsBefore + 1U) << "one forward for the whole step";
    rt::InferenceDims const& dims = raw->preparedRounds().back().dims;
    EXPECT_EQ(raw->preparedRounds().back().profileIndex, kPrefillProfile);
    EXPECT_EQ(dims.executionPhaseLen, kPhase8);
    EXPECT_EQ(dims.batch, 3);
    EXPECT_EQ(dims.seqLen, 8);
    EXPECT_EQ(dims.contextSequenceCount, 2);
    EXPECT_EQ(dims.selectLen, 3) << "one logits row per sequence, P1's a discarded placeholder";
    EXPECT_EQ(mixed.work, rt::SequenceWork::kContext);
    EXPECT_EQ(mixed.execution,
        (std::vector<rt::ScheduledSequenceDescriptor>{
            {rt::SequenceIdentity{2, p1}, rt::SequenceWork::kContext, 4, 0, false},
            {rt::SequenceIdentity{3, p2}, rt::SequenceWork::kContext, 3, 0, true},
            {rt::SequenceIdentity{1, a}, rt::SequenceWork::kDecode, 1, 3, true}}))
        << "prefill rows first, in plan order, then decode rows";
    EXPECT_THAT(producers(mixed), ::testing::UnorderedElementsAre(a, p2)) << "a non-final chunk produces no token";

    rt::StepResult const finalChunk = driver.run({driver.rest(p1), driver.decode(a), driver.decode(p2)});
    ASSERT_TRUE(finalChunk.ok);
    // A and P2 spend their budgets in this step, so their last tokens leave in their finished snapshots.
    EXPECT_THAT(producers(finalChunk), ::testing::ElementsAre(p1)) << "P1's final chunk produces its first token";
    driver.drain();
    for (size_t index = 0; index < requests.size(); ++index)
    {
        rt::RequestId const requestId = static_cast<rt::RequestId>(index + 1);
        SCOPED_TRACE("request " + std::to_string(requestId));
        EXPECT_EQ(driver.outcomes().at(requestId), rt::FinishReason::kLength);
        EXPECT_EQ(driver.output(requestId).size(), static_cast<size_t>(requests[index].maxGenerateLength));
        EXPECT_EQ(driver.output(requestId), references[index].outputIds.front());
        EXPECT_EQ(logprobsOf(driver.responses().at(requestId)), logprobsOf(references[index]));
        EXPECT_FALSE(logprobsOf(references[index]).empty());
    }
}

// Given a hybrid compact deployment with A decoding and P admitted
// When a step runs only a non-final chunk of P, the next only A's decode row, and C is admitted after
// Then the first step produces no token from its single placeholder logits row; the second runs as a phase-8 step on
//      the decode profile; neither step reads or writes the state of the resident it does not name; and C's admission
//      after these subset steps succeeds
TEST_F(CompactHybridSessionTest, SubsetStepsLeaveOtherResidentsUntouched)
{
    auto engine = makeCompactFake(makeCompactHybridConfig());
    auto* const raw = engine.get();
    raw->forceStateValues([](FakeStateSlot const& slot) { return 1.0 + static_cast<double>(slot.position); });
    std::vector<std::string> const pools{binding_names::formatRecurrentStateName(0, /*isPast=*/false),
        binding_names::formatConvStateName(0, /*isPast=*/false)};
    for (auto const& pool : pools)
    {
        raw->recordStatePool(pool);
    }
    auto runtime = makeRuntimeOver(std::move(engine));
    auto session = openCompactSession(runtime);
    CompactDriver driver(*session);
    rt::ResidentRef const a = driver.admit(makeTokenizedRequest({40, 41}), 1);
    ASSERT_TRUE(driver.run({driver.rest(a)}).ok);
    ASSERT_TRUE(driver.run({driver.decode(a)}).ok);
    rt::ResidentRef const p = driver.admit(makeTokenizedRequest({50, 51, 52, 53, 54, 55}), 2);
    ASSERT_EQ(a.slot, 0);
    ASSERT_EQ(p.slot, 1);

    rt::StepResult const chunkOnly = driver.run({driver.chunk(p, 2)});
    ASSERT_TRUE(chunkOnly.ok);
    EXPECT_EQ(raw->preparedRounds().back().dims.selectLen, 1) << "the placeholder logits row";
    EXPECT_TRUE(chunkOnly.deltas.empty());
    ASSERT_TRUE(driver.run({driver.decode(a)}).ok);
    EXPECT_EQ(raw->preparedRounds().back().profileIndex, kDecodeProfile);
    EXPECT_EQ(raw->preparedRounds().back().dims.executionPhaseLen, kPhase8);
    rt::ResidentRef const c = driver.admit(makeTokenizedRequest({60}), 3);
    EXPECT_EQ(c.slot, 2);
    // Each forward snapshots the pools as the steps before it left them.
    ASSERT_TRUE(driver.run({driver.chunk(p, 4), driver.rest(c), driver.decode(a)}).ok);
    for (auto const& pool : pools)
    {
        auto const& rounds = raw->preparedStatePool(pool);
        ASSERT_GE(rounds.size(), 3U);
        auto const& beforeChunk = rounds[rounds.size() - 3];
        auto const& afterChunk = rounds[rounds.size() - 2];
        auto const& afterDecode = rounds.back();
        EXPECT_EQ(afterChunk.at(0), beforeChunk.at(0)) << pool << ": P's chunk wrote A's state";
        EXPECT_NE(afterChunk.at(1), beforeChunk.at(1)) << pool << ": P's chunk did not advance P's state";
        EXPECT_EQ(afterDecode.at(1), afterChunk.at(1)) << pool << ": A's decode step wrote P's state";
        EXPECT_NE(afterDecode.at(0), afterChunk.at(0)) << pool << ": A's decode step did not advance A's state";
    }
    driver.drain();
    EXPECT_EQ(driver.outcomes().size(), 3U);
}

// Given a compact deployment with A decoding, and P (4 tokens) and Q admitted, nothing run, and cancelled
// When the caller schedules plans that break a compact-step rule: an unknown request, a sequence named twice, a past
//      length off the ledger, an empty or overlong chunk, a wrong sampling flag, a decode of the prompt-incomplete P
//      or of two tokens, a prompt chunk of the prompt-complete A; then a valid plan naming P's chunk and A's decode row
// Then each invalid plan is refused before anything moves, the cancellations included; the valid plan retires P and Q
//      as cancelled without running P's chunk, and A decodes alone in a phase-3 step on the decode profile, reported
//      as not compact
TEST_F(CompactSessionTest, RefusesPlansBreakingCompactRules)
{
    auto engine = makeCompactFake(makeCompactVanillaConfig());
    auto* const raw = engine.get();
    auto runtime = makeRuntimeOver(std::move(engine));
    auto session = openCompactSession(runtime);
    CompactDriver driver(*session);
    rt::ResidentRef const a = driver.admit(makeTokenizedRequest({40, 41}), 1);
    ASSERT_TRUE(driver.run({driver.rest(a)}).ok);
    auto cancelP = std::make_shared<std::atomic<bool>>(false);
    rt::ResidentRef const p = driver.admit(makeTokenizedRequest({50, 51, 52, 53}), 2, cancelP);
    rt::ResidentRef const q = driver.admit(makeTokenizedRequest({60, 61}), 3, cancelP);
    cancelP->store(true);
    size_t const forwardsBefore = raw->forwardsRun();

    auto unknown = driver.decode(a);
    unknown.identity.requestId = 999;
    auto pastOff = driver.decode(a);
    pastOff.pastLength += 1;
    auto samplesEarly = driver.chunk(p, 2);
    samplesEarly.selectLastTokenLogits = true;
    auto finalWithoutSample = driver.chunk(p, 4);
    finalWithoutSample.selectLastTokenLogits = false;
    auto twoTokenDecode = driver.decode(a);
    twoTokenDecode.queryLength = 2;
    auto zeroChunk = driver.chunk(p, 1);
    zeroChunk.queryLength = 0;
    std::vector<std::vector<rt::ScheduledSequenceDescriptor>> const refused{
        {unknown},
        {driver.decode(a), driver.decode(a)},
        {pastOff},
        {zeroChunk},
        {driver.chunk(p, 5)},
        {samplesEarly},
        {finalWithoutSample},
        {driver.decode(p)},
        {twoTokenDecode},
        {driver.chunk(a, 1)},
    };
    for (size_t index = 0; index < refused.size(); ++index)
    {
        EXPECT_THROW(session->execute(refused[index]), std::runtime_error) << "plan " << index;
    }
    EXPECT_EQ(raw->forwardsRun(), forwardsBefore) << "a refused plan runs nothing";
    EXPECT_EQ(driver.committed(a), 2);
    EXPECT_EQ(session->residents(), (std::vector<rt::ResidentRef>{a, p, q})) << "a refused plan retires nobody";

    rt::StepResult const step = driver.run({driver.decode(a), driver.chunk(p, 4)});
    ASSERT_TRUE(step.ok);
    EXPECT_EQ(raw->forwardsRun(), forwardsBefore + 1U);
    EXPECT_FALSE(step.compact);
    EXPECT_EQ(raw->preparedRounds().back().profileIndex, kDecodeProfile);
    EXPECT_EQ(raw->preparedRounds().back().dims.executionPhaseLen, kPhase3);
    EXPECT_EQ(step.execution.size(), 1U) << "the cancelled chunk does not run";
    EXPECT_THAT(producers(step), ::testing::ElementsAre(a));
    EXPECT_EQ(driver.outcomes().at(2), rt::FinishReason::kCancelled);
    EXPECT_EQ(driver.outcomes().at(3), rt::FinishReason::kCancelled);
    EXPECT_EQ(session->residents(), (std::vector<rt::ResidentRef>{a}));
}

namespace
{

constexpr int32_t kPage{rt::kTOKENS_PER_PAGE};

//! B = 4, an input limit of one page plus a little under a KV capacity of four pages.
Json withCompactContextReuseLimits(Json config)
{
    config["builder_config"]["max_batch_size"] = kCompactBatchSize;
    config["builder_config"]["max_input_len"] = kPage + 8;
    config["builder_config"]["max_kv_cache_capacity"] = 4 * kPage;
    config["builder_config"]["max_kv_pool_pages"] = rt::computeMinimumKvPoolPages(kCompactBatchSize, 4 * kPage);
    return withCompactShapes(std::move(config));
}

//! A context cache without the encoder embedding cache.
rt::ContextCacheConfig compactCacheConfig()
{
    rt::ContextCacheConfig cacheConfig;
    cacheConfig.enabled = true;
    cacheConfig.maxRecords = 16;
    cacheConfig.maxSnapshots = 64;
    cacheConfig.encoderEmbeddingCacheBudgetBytes = 0;
    return cacheConfig;
}

class CompactContextReuseSessionTest : public CompactSessionTest
{
protected:
    std::filesystem::path stageModelDir() override
    {
        return writeModelDir("compactContextReuseSessionTests", config());
    }

    virtual Json config() const
    {
        return withCompactContextReuseLimits(makeTinyVanillaConfig());
    }

    rt::LLMInferenceRuntime makeCachedRuntimeOver(std::unique_ptr<rt::EngineExecutor> engine)
    {
        return rt::LLMInferenceRuntime{makeVanillaArtifacts(mModelDir, std::move(engine), mStream), mModelDir.string(),
            /*multimodalEngineDir=*/"", {}, std::nullopt, mStream, compactCacheConfig()};
    }

    static rt::ContextCacheMetrics cacheMetrics(rt::LLMInferenceRuntime const& runtime)
    {
        std::optional<rt::ContextCacheMetrics> metrics = runtime.getContextCacheMetrics();
        EXPECT_TRUE(metrics.has_value());
        return metrics.value_or(rt::ContextCacheMetrics{});
    }

    void cancelledOrFailedSequencesPublishNothing();
};

class CompactHybridContextReuseSessionTest : public CompactContextReuseSessionTest
{
protected:
    std::filesystem::path stageModelDir() override
    {
        return writeModelDir("compactHybridContextReuseSessionTests", config());
    }

    Json config() const override
    {
        return withCompactContextReuseLimits(makeTinyHybridConfig());
    }
};

} // namespace

// Given a compact deployment under a context cache, X (126 tokens, no full page) decoding and P (132 tokens) admitted
// When P runs a non-final chunk, then its final chunk, each beside X's decode row; X finishes beside R's prompt; then Y
//      (X's prompt, its output and two more tokens) and Z (P's prompt) are admitted
// Then only P's final chunk publishes P's full page; X's decode rows advance the cache so that its finish publishes
//      its history's first page; Y and Z start at one page and run only their uncached tails, in one step; and every
//      output equals the blocking path's
TEST_F(CompactContextReuseSessionTest, ChunksPublishAtFinalChunkAndDecodeRowsAdvanceCache)
{
    std::vector<int32_t> const promptX = promptOf(kPage - 2, 0);
    std::vector<int32_t> const promptP = promptOf(kPage + 4, 1);
    std::vector<int32_t> const promptR = promptOf(10, 2);
    auto engine = makeCompactFake(config());
    auto* const raw = engine.get();
    auto runtime = makeCachedRuntimeOver(std::move(engine));
    auto const published = [&runtime]() { return cacheMetrics(runtime).committedPublications; };
    std::unordered_map<rt::RequestId, rt::LLMGenerationResponse> responses;
    std::vector<int32_t> promptY = promptX;
    {
        auto session = openCompactSession(runtime);
        CompactDriver driver(*session);
        rt::ResidentRef const x = driver.admit(makeTokenizedRequest(promptX), 1);
        ASSERT_TRUE(driver.run({driver.rest(x)}).ok);
        ASSERT_EQ(published(), 0U) << "X's prompt holds no full page";

        rt::ResidentRef const p = driver.admit(makeTokenizedRequest(promptP), 2);
        EXPECT_EQ(driver.committed(p), 0);
        rt::StepResult const nonFinal = driver.run({driver.decode(x), driver.chunk(p, 100)});
        ASSERT_TRUE(nonFinal.ok);
        EXPECT_FALSE(nonFinal.publishedPrefix);
        EXPECT_EQ(published(), 0U) << "a non-final chunk publishes nothing";

        rt::StepResult const finalChunk = driver.run({driver.rest(p), driver.decode(x)});
        ASSERT_TRUE(finalChunk.ok);
        EXPECT_TRUE(finalChunk.publishedPrefix);
        EXPECT_EQ(published(), 1U) << "P's final chunk publishes its full page";

        rt::ResidentRef const r = driver.admit(makeTokenizedRequest(promptR), 3);
        ASSERT_TRUE(driver.run({driver.rest(r), driver.decode(x)}).ok);
        EXPECT_EQ(raw->preparedRounds().back().dims.executionPhaseLen, kPhase8);
        ASSERT_EQ(driver.outcomes().count(1), 1U) << "X spent its budget in the mixed step";
        EXPECT_EQ(published(), 2U) << "X's history reached a full page through its decode rows";
        driver.drain();

        std::vector<int32_t> const& outputX = driver.output(1);
        promptY.insert(promptY.end(), outputX.begin(), outputX.end());
        promptY.insert(promptY.end(), {5, 6});
        rt::ResidentRef const y = driver.admit(makeTokenizedRequest(promptY), 4);
        rt::ResidentRef const z = driver.admit(makeTokenizedRequest(promptP), 5);
        EXPECT_EQ(driver.committed(y), kPage) << "X's decode-end page is reused";
        EXPECT_EQ(driver.committed(z), kPage) << "P's prompt-end page is reused";
        rt::StepResult const reused = driver.run({driver.chunk(y, 4), driver.chunk(z, 4)});
        ASSERT_TRUE(reused.ok);
        ASSERT_EQ(reused.execution.size(), 2U);
        EXPECT_EQ(reused.execution[0].pastLength, kPage);
        EXPECT_EQ(reused.execution[1].pastLength, kPage);
        EXPECT_EQ(raw->preparedRounds().back().dims.seqLen, 8) << "only the uncached tails run";
        driver.drain();
        responses = driver.responses();
    }
    EXPECT_EQ(responses.at(1).outputIds.front(), blockingOutput(config(), promptX));
    EXPECT_EQ(responses.at(2).outputIds.front(), blockingOutput(config(), promptP));
    EXPECT_EQ(responses.at(3).outputIds.front(), blockingOutput(config(), promptR));
    EXPECT_EQ(responses.at(4).outputIds.front(), blockingOutput(config(), promptY));
    EXPECT_EQ(responses.at(5).outputIds, responses.at(2).outputIds);
}

// Given a compact deployment under a context cache, attention-only or hybrid
// When P is cancelled between its chunks; Q is cancelled while its final chunk runs beside S's chunk and S finishes;
//      R (P's prompt) and T (S's prompt and one more token) run a chunk each, the next step fails to bind and the
//      session closes; and a new session serves P's and T's prompts
// Then the cancelled P and Q publish nothing and return their leases; S publishes at its final chunk; R finds nothing
//      cached and T resumes where S's publication ends (its full page, or the hybrid's prompt-end snapshot); the failed
//      step executes, produces and publishes nothing, and closing the session returns every lease; and P's, S's and
//      T's outputs equal the blocking path's
void CompactContextReuseSessionTest::cancelledOrFailedSequencesPublishNothing()
{
    bool const hybrid = config().value("num_linear_attn_layers", 0) > 0;
    std::vector<int32_t> const promptP = promptOf(kPage + 4, 1);
    std::vector<int32_t> const promptS = promptOf(kPage + 7, 3);
    std::vector<int32_t> promptT = promptS;
    promptT.push_back(5);
    auto engine = makeCompactFake<HookedFakeEngine>(config());
    auto* const raw = engine.get();
    auto runtime = makeCachedRuntimeOver(std::move(engine));
    auto const freePages = [&runtime]() { return cacheMetrics(runtime).baseKvPages.free; };
    auto const publications = [&runtime]() { return cacheMetrics(runtime).committedPublications; };
    std::vector<int32_t> outputS;
    int32_t freeBeforeRAndT = 0;
    {
        auto session = openCompactSession(runtime);
        CompactDriver driver(*session);
        int32_t const freeIdle = freePages();

        auto cancelP = std::make_shared<std::atomic<bool>>(false);
        rt::ResidentRef const p = driver.admit(makeTokenizedRequest(promptP), 1, cancelP);
        EXPECT_LT(freePages(), freeIdle) << "P holds a lease";
        ASSERT_TRUE(driver.run({driver.chunk(p, 100)}).ok);
        cancelP->store(true);
        rt::StepResult const retired = driver.run({driver.rest(p)});
        ASSERT_TRUE(retired.ok);
        EXPECT_TRUE(retired.execution.empty()) << "the cancelled chunk does not run";
        EXPECT_FALSE(retired.publishedPrefix);
        EXPECT_EQ(driver.outcomes().at(1), rt::FinishReason::kCancelled);
        EXPECT_EQ(publications(), 0U);
        EXPECT_EQ(freePages(), freeIdle) << "P's lease is released";

        auto cancelQ = std::make_shared<std::atomic<bool>>(false);
        rt::ResidentRef const q = driver.admit(makeTokenizedRequest(promptP), 2, cancelQ);
        EXPECT_EQ(driver.committed(q), 0) << "P published nothing";
        int32_t const pagesOfQ = freeIdle - freePages();
        rt::ResidentRef const s = driver.admit(makeTokenizedRequest(promptS), 3);
        int32_t const freeBesideQAndS = freePages();
        ASSERT_TRUE(driver.run({driver.chunk(q, 100), driver.chunk(s, 100)}).ok);
        raw->afterForward = [cancelQ]() { cancelQ->store(true); };
        rt::StepResult const cancelledAtCommit = driver.run({driver.rest(q), driver.chunk(s, 30)});
        raw->afterForward = nullptr;
        ASSERT_TRUE(cancelledAtCommit.ok);
        EXPECT_EQ(cancelledAtCommit.execution.size(), 2U) << "Q's final chunk ran";
        EXPECT_FALSE(cancelledAtCommit.publishedPrefix);
        EXPECT_EQ(driver.outcomes().at(2), rt::FinishReason::kCancelled);
        EXPECT_EQ(publications(), 0U) << "cancellation wins over publication";
        EXPECT_EQ(freePages(), freeBesideQAndS + pagesOfQ) << "Q's lease is released";
        rt::StepResult const finalS = driver.run({driver.rest(s)});
        ASSERT_TRUE(finalS.ok);
        EXPECT_TRUE(finalS.publishedPrefix);
        EXPECT_EQ(publications(), 1U) << "S publishes at its final chunk";
        driver.drain();
        outputS = driver.output(3);

        freeBeforeRAndT = freePages();
        uint64_t const restores = cacheMetrics(runtime).hybridRestores;
        rt::ResidentRef const r = driver.admit(makeTokenizedRequest(promptP), 4);
        EXPECT_EQ(driver.committed(r), 0) << "neither cancelled sequence published";
        rt::ResidentRef const t = driver.admit(makeTokenizedRequest(promptT), 5);
        int32_t const reused = hybrid ? static_cast<int32_t>(promptS.size()) : kPage;
        EXPECT_EQ(driver.committed(t), reused) << "T starts where S's publication ends";
        ASSERT_TRUE(driver.run({driver.chunk(r, 100), driver.rest(t)}).ok);
        EXPECT_EQ(raw->preparedRounds().back().dims.seqLen, 100 + static_cast<int64_t>(promptT.size()) - reused);
        EXPECT_EQ(cacheMetrics(runtime).hybridRestores, restores + (hybrid ? 1U : 0U));

        uint64_t const publishedBeforeFailure = publications();
        raw->phaseEightFailure = HookedFakeEngine::Failure::kPrepare;
        rt::StepResult const failed = session->execute({driver.rest(r), driver.decode(t)});
        raw->phaseEightFailure = HookedFakeEngine::Failure::kNone;
        EXPECT_FALSE(failed.ok);
        EXPECT_TRUE(failed.execution.empty());
        EXPECT_TRUE(failed.deltas.empty());
        EXPECT_FALSE(failed.publishedPrefix);
        EXPECT_EQ(publications(), publishedBeforeFailure);
    }
    EXPECT_EQ(freePages(), freeBeforeRAndT) << "a lease outlived the failed session";

    auto session = openCompactSession(runtime);
    CompactDriver driver(*session);
    rt::ResidentRef const p = driver.admit(makeTokenizedRequest(promptP), 6);
    rt::ResidentRef const t = driver.admit(makeTokenizedRequest(promptT), 7);
    ASSERT_TRUE(driver.run({driver.chunk(p, 100), driver.rest(t)}).ok);
    ASSERT_TRUE(driver.run({driver.rest(p), driver.decode(t)}).ok);
    driver.drain();
    EXPECT_EQ(driver.output(6), blockingOutput(config(), promptP));
    EXPECT_EQ(outputS, blockingOutput(config(), promptS));
    EXPECT_EQ(driver.output(7), blockingOutput(config(), promptT));
}

TEST_F(CompactContextReuseSessionTest, CancelledOrFailedSequencesPublishNothing)
{
    cancelledOrFailedSequencesPublishNothing();
}

TEST_F(CompactHybridContextReuseSessionTest, CancelledOrFailedSequencesPublishNothing)
{
    cancelledOrFailedSequencesPublishNothing();
}

// Given a compact deployment whose phase-8 forward fails in one of four ways: refused with the device healthy (as by a
//       plugin refusing phase 8), failing to bind the step, failing with a CUDA error pending, or failing with the
//       stream in error
// When a step runs P's prompt beside A's decode row, and then a new session serves B
// Then the step fails in every case; only the refusal reports compactRefused, says the request can be retried and
//      leaves the new session without compact execution, naming the refusal; B's output equals the blocking path's,
//      in a phase-8 step exactly when the session is still compact
TEST_F(CompactSessionTest, OnlyARefusedPhase8ForwardDisablesCompactExecution)
{
    using Failure = HookedFakeEngine::Failure;
    std::vector<int32_t> const promptB{70, 71, 72};
    std::vector<int32_t> const referenceB = blockingOutput(makeCompactVanillaConfig(), promptB);
    for (Failure const failure :
        {Failure::kRefusal, Failure::kPrepare, Failure::kPendingCudaError, Failure::kStreamInError})
    {
        bool const refused = failure == Failure::kRefusal;
        SCOPED_TRACE("failure " + std::to_string(static_cast<int32_t>(failure)));
        auto engine = makeCompactFake<HookedFakeEngine>(makeCompactVanillaConfig());
        auto* const raw = engine.get();
        auto runtime = makeRuntimeOver(std::move(engine));
        {
            auto session = openCompactSession(runtime);
            CompactDriver driver(*session);
            rt::ResidentRef const a = driver.admit(makeTokenizedRequest({40, 41}), 1);
            ASSERT_TRUE(driver.run({driver.rest(a)}).ok);
            rt::ResidentRef const p = driver.admit(makeTokenizedRequest({50, 51, 52}), 2);
            raw->phaseEightFailure = failure;
            rt::StepResult const failed = session->execute({driver.rest(p), driver.decode(a)});
            recoverStream(mStream);
            EXPECT_FALSE(failed.ok);
            EXPECT_EQ(failed.compactRefused, refused);
            EXPECT_EQ(failed.failure.find("can be retried") != std::string::npos, refused) << failed.failure;
        }
        if (!refused)
        {
            raw->phaseEightFailure = Failure::kNone;
        }

        size_t const roundsBefore = raw->preparedRounds().size();
        std::unique_ptr<rt::RuntimeSession> session = runtime.openSession(mStream, kCompactSession);
        ASSERT_NE(session, nullptr);
        rt::SessionCapabilities const capabilities = session->capabilities();
        EXPECT_EQ(capabilities.compact, !refused) << capabilities.reason;
        EXPECT_EQ(capabilities.reason.find("refused") != std::string::npos, refused) << capabilities.reason;
        CompactDriver driver(*session);
        rt::ResidentRef const b = driver.admit(makeTokenizedRequest(promptB), 3);
        ASSERT_TRUE(driver.run({driver.rest(b)}).ok);
        driver.drain();
        EXPECT_EQ(driver.output(3), referenceB);
        auto const& rounds = raw->preparedRounds();
        bool const ranPhase8 = std::any_of(rounds.begin() + static_cast<std::ptrdiff_t>(roundsBefore), rounds.end(),
            [](auto const& round) { return round.dims.executionPhaseLen == kPhase8; });
        EXPECT_EQ(ranPhase8, !refused);
    }
}

// Given a compact deployment under a context cache, served by a RequestEngine under auto, whose phase-8 forwards are
//       refused once A's prompt has run
// When P, streaming on its own channel, is admitted beside A's decode row and that step is refused
// Then A, which sampled a token, fails with a retryable reason; P, which sampled none, is served by the reopened
//      session on the channel it was submitted with: the channel delivers the blocking path's tokens for P alone and
//      ends with FinishReason::kLength, not with the error the closed session gives its residents
TEST_F(CompactContextReuseSessionTest, ARefusedStepReadmitsAPromptThatSampledNothingOnItsOwnChannel)
{
    rt::LLMGenerationRequest const requestA = budgetedRequest({40, 41}, 8);
    rt::LLMGenerationRequest requestP = budgetedRequest({50, 51, 52}, 4);
    std::vector<int32_t> const referenceP = blockingOutput(config(), requestP);
    auto engine = makeCompactFake<HookedFakeEngine>(config());
    auto* const raw = engine.get();
    std::promise<void> promptRan;
    std::promise<void> submitted;
    std::shared_future<void> const submittedP = submitted.get_future().share();
    int32_t forwards = 0;
    raw->afterForward = [&] {
        if (++forwards == 1)
        {
            promptRan.set_value();
            submittedP.wait();
            raw->phaseEightFailure = HookedFakeEngine::Failure::kRefusal;
        }
    };
    rt::scheduler::EngineConfig engineConfig;
    engineConfig.maxBatchSize = kCompactBatchSize;
    rt::scheduler::RequestEngine requestEngine(
        std::make_unique<rt::LLMInferenceRuntime>(makeVanillaArtifacts(mModelDir, std::move(engine), mStream),
            mModelDir.string(), /*multimodalEngineDir=*/"", std::unordered_map<std::string, std::string>{},
            std::nullopt, mStream, compactCacheConfig()),
        mStream, engineConfig);
    ASSERT_TRUE(requestEngine.resolvedPolicy().compactExecution);

    rt::scheduler::RequestHandle a = requestEngine.submit(requestA);
    bool const ran = promptRan.get_future().wait_for(std::chrono::seconds(5)) == std::future_status::ready;
    std::shared_ptr<rt::StreamChannel> const channelP = rt::StreamChannel::create();
    requestP.streamChannels = {channelP};
    rt::scheduler::RequestHandle p = requestEngine.submit(requestP);
    submitted.set_value();
    ASSERT_TRUE(ran);
    std::vector<int32_t> streamed;
    rt::FinishReason ended = rt::FinishReason::kNotFinished;
    channelP->consume([&](rt::StreamChunk&& chunk) {
        streamed.insert(streamed.end(), chunk.tokenIds.begin(), chunk.tokenIds.end());
        ended = chunk.finished ? chunk.reason : ended;
    });
    std::string failureA;
    try
    {
        a.get();
    }
    catch (std::exception const& error)
    {
        failureA = error.what();
    }

    EXPECT_NE(failureA.find("can be retried"), std::string::npos) << failureA;
    EXPECT_EQ(p.get().outputIds.at(0), referenceP);
    EXPECT_EQ(streamed, referenceP);
    EXPECT_EQ(ended, rt::FinishReason::kLength);
    EXPECT_EQ(requestEngine.metrics().compactRefusals, 1U);
}

// Given a compact deployment under a context cache, served by a RequestEngine under auto, with A decoding and P and Q
//       (132 tokens each) admitted beside it, whose first 128-token chunks run in one step
// When P is cancelled after that step, and the next step, which runs Q's final chunk beside A's decode row, is refused
// Then P ends as cancelled and the engine counts it as cancelled, not failed; A fails as a retryable compact refusal
//      (FailureKind kCompactStepRefused, the reason saying it can be retried); Q,
//      which sampled nothing, is served by the reopened session with the blocking path's tokens; and no request is
//      logged as unable to be queued again
TEST_F(CompactContextReuseSessionTest, AResidentCancelledInARefusedStepEndsCancelled)
{
    rt::LLMGenerationRequest const requestA = budgetedRequest({40, 41}, 8);
    rt::LLMGenerationRequest const requestP = budgetedRequest(promptOf(kPage + 4, 1), 4);
    rt::LLMGenerationRequest const requestQ = budgetedRequest(promptOf(kPage + 4, 2), 4);
    std::vector<int32_t> const referenceQ = blockingOutput(config(), requestQ);
    auto engine = makeCompactFake<HookedFakeEngine>(config());
    auto* const raw = engine.get();
    std::promise<void> promptRan;
    std::promise<void> submitted;
    std::shared_future<void> const submittedPAndQ = submitted.get_future().share();
    std::optional<rt::scheduler::RequestHandle> p;
    int32_t forwards = 0;
    raw->afterForward = [&] {
        ++forwards;
        if (forwards == 1)
        {
            promptRan.set_value();
            submittedPAndQ.wait();
        }
        else if (forwards == 2)
        {
            p->cancel();
            raw->phaseEightFailure = HookedFakeEngine::Failure::kRefusal;
        }
    };
    rt::scheduler::EngineConfig engineConfig;
    engineConfig.maxBatchSize = kCompactBatchSize;
    testing::internal::CaptureStderr();
    std::string failureA;
    std::optional<rt::scheduler::FailureKind> kindA;
    std::string failureP;
    std::vector<int32_t> outputQ;
    rt::scheduler::EngineMetrics metrics;
    bool compact = false;
    bool ran = false;
    {
        rt::scheduler::RequestEngine requestEngine(
            std::make_unique<rt::LLMInferenceRuntime>(makeVanillaArtifacts(mModelDir, std::move(engine), mStream),
                mModelDir.string(), /*multimodalEngineDir=*/"", std::unordered_map<std::string, std::string>{},
                std::nullopt, mStream, compactCacheConfig()),
            mStream, engineConfig);
        compact = requestEngine.resolvedPolicy().compactExecution;
        rt::scheduler::RequestHandle a = requestEngine.submit(requestA);
        ran = promptRan.get_future().wait_for(std::chrono::seconds(5)) == std::future_status::ready;
        p.emplace(requestEngine.submit(requestP));
        rt::scheduler::RequestHandle q = requestEngine.submit(requestQ);
        submitted.set_value();
        try
        {
            a.get();
        }
        catch (rt::scheduler::RequestFailedError const& error)
        {
            failureA = error.what();
            kindA = error.kind();
        }
        try
        {
            p->get();
        }
        catch (std::exception const& error)
        {
            failureP = error.what();
        }
        try
        {
            outputQ = q.get().outputIds.at(0);
        }
        catch (std::exception const& error)
        {
            ADD_FAILURE() << "Q failed: " << error.what();
        }
        metrics = requestEngine.metrics();
    }
    std::string const log = testing::internal::GetCapturedStderr();
    ASSERT_TRUE(compact);
    ASSERT_TRUE(ran);

    EXPECT_NE(failureP.find("was cancelled"), std::string::npos) << failureP;
    EXPECT_EQ(metrics.cancelled, 1U);
    EXPECT_EQ(metrics.failed, 1U);
    EXPECT_EQ(kindA, rt::scheduler::FailureKind::kCompactStepRefused) << failureA;
    EXPECT_NE(failureA.find("can be retried"), std::string::npos) << failureA;
    EXPECT_EQ(outputQ, referenceQ);
    EXPECT_EQ(metrics.compactRefusals, 1U);
    EXPECT_EQ(log.find("cannot be queued again"), std::string::npos) << log;
}

// Given a deployment whose engine admits phase 8 but whose attention head dim (80) no ragged paged prefill kernel
//       covers, served by a RequestEngine under auto, and the same deployment under mixed
// When the engines start, and the auto engine serves A
// Then the auto engine resolves to prefillFirst without compact execution, with a reason naming the attention layers
//      and their head dim, and serves A with the blocking path's output and no phase-8 forward; the mixed engine fails
//      to start with a message carrying the same reason
TEST_F(CompactSessionTest, LayersWithoutKernelsResolveAutoToPrefillFirstAndFailMixed)
{
    Json config = makeCompactVanillaConfig();
    config["head_dim"] = 80;
    expectAutoServesPrefillFirstAndMixedFails(config,
        "no compact-execution kernels on SM" + std::to_string(getSMVersion())
            + " for attention layers (layer 0: 4 query heads, 2 KV heads, head dim 80: ");
}

// Given a deployment whose layers all have compact-execution kernels on this GPU and whose engine admits phase 8, but
//       whose builder_config does not record compact_execution: true, served by a RequestEngine under auto and under
//       mixed
// When the engines start, and the auto engine serves A
// Then the auto engine resolves to prefillFirst without compact execution, with a reason saying the engine was not
//      built for compact execution, and serves A with the blocking path's output and no phase-8 forward; the mixed
//      engine fails to start with the same reason
TEST_F(CompactSessionTest, AnEngineNotBuiltForCompactExecutionResolvesAutoToPrefillFirstAndFailsMixed)
{
    Json config = makeCompactVanillaConfig();
    config["builder_config"].erase("compact_execution");
    expectAutoServesPrefillFirstAndMixedFails(config, "the engine was not built for compact execution");
}

// Given a compact-capable deployment served by a RequestEngine with ifbSchedulingMode = singleCohort, and the same
//       deployment under the default configuration
// When each engine starts and serves A and B together, two requests whose generate budgets differ (3 and 9)
// Then the singleCohort engine resolves to singleCohort without compact execution and runs no phase-8 forward, the
//      default one executes compact steps; and each engine stalls no request on batch compatibility and answers A
//      and B each with its own budget of the blocking path's tokens for it alone
TEST_F(CompactSessionTest, RequestEnginesServeEveryGenerateBudgetWithAndWithoutCompactSteps)
{
    std::vector<rt::LLMGenerationRequest> const requests{
        budgetedRequest({40, 41, 42}, 3), budgetedRequest({50, 51}, 9)};
    std::vector<std::vector<int32_t>> references;
    for (rt::LLMGenerationRequest const& request : requests)
    {
        references.push_back(blockingOutput(makeCompactVanillaConfig(), request));
    }
    for (bool const singleCohort : {true, false})
    {
        SCOPED_TRACE(singleCohort ? "singleCohort" : "default");
        auto engine = makeCompactFake(makeCompactVanillaConfig());
        auto* const raw = engine.get();
        auto runtime = std::make_unique<rt::LLMInferenceRuntime>(
            makeVanillaArtifacts(mModelDir, std::move(engine), mStream), mModelDir.string(),
            /*multimodalEngineDir=*/"", std::unordered_map<std::string, std::string>{}, std::nullopt, mStream);
        rt::scheduler::EngineConfig engineConfig;
        engineConfig.maxBatchSize = kCompactBatchSize;
        if (singleCohort)
        {
            engineConfig.scheduling.ifbSchedulingMode = rt::scheduler::IfbSchedulingMode::kSingleCohort;
        }
        rt::scheduler::RequestEngine requestEngine(std::move(runtime), mStream, engineConfig);
        rt::scheduler::ResolvedSchedulingPolicy const policy = requestEngine.resolvedPolicy();
        EXPECT_EQ(policy.mode == rt::scheduler::IfbSchedulingMode::kSingleCohort, singleCohort);
        EXPECT_EQ(policy.compactExecution, !singleCohort) << policy.reason;
        std::vector<rt::scheduler::RequestHandle> handles;
        for (rt::LLMGenerationRequest const& request : requests)
        {
            handles.push_back(requestEngine.submit(request));
        }
        for (size_t index = 0; index < requests.size(); ++index)
        {
            rt::LLMGenerationResponse const response = handles[index].get();
            EXPECT_EQ(response.outputIds.at(0), references[index]) << "request " << index;
            EXPECT_EQ(response.outputIds.at(0).size(), static_cast<size_t>(requests[index].maxGenerateLength));
        }
        EXPECT_EQ(requestEngine.metrics().stallsIncompatible, 0U);
        bool const ranPhase8 = std::any_of(raw->preparedRounds().begin(), raw->preparedRounds().end(),
            [](auto const& round) { return round.dims.executionPhaseLen == kPhase8; });
        EXPECT_EQ(ranPhase8, !singleCohort);
    }
}

namespace
{

constexpr int32_t kFakeImageToken{200};
constexpr int64_t kFakeRowsPerImage{6};
constexpr int64_t kFakeMaxRows{64};
constexpr float kFakeMropeBase{500.0F};

void CUDART_CB sleepOnStream(void* delay)
{
    std::this_thread::sleep_for(*static_cast<std::chrono::milliseconds const*>(delay));
}

//! A vision encoder without an engine. Each 'z' of the formatted prompt stands for one image of the request and
//! expands to kFakeRowsPerImage image tokens or, with layoutFromEncoder, to layoutRows(pixel) tokens, a count that (as
//! for a Nemotron-Omni video) only a preprocess with encoder work computes. Output row r of an image whose pixels all
//! hold v is imageRowValue(v, r, channel). infer() uploads its output from a pinned staging buffer that the next
//! infer() rewrites, queued behind stagedUploadDelay of device-side work.
class FakeVisionRunner : public rt::MultimodalRunner
{
public:
    enum class Failure : uint8_t
    {
        kNone,
        kPreprocess,         //!< preprocess() fails with the stream healthy.
        kPreprocessOnDevice, //!< preprocess() fails leaving the stream in error.
        kInfer,              //!< infer() fails without writing any output.
        kInferCudaCall,      //!< infer() throws from a failed CUDA call.
    };

    explicit FakeVisionRunner(int64_t hiddenSize, std::chrono::milliseconds stagedUploadDelay = {})
        : mHiddenSize(hiddenSize)
        , mStagedUploadDelay(stagedUploadDelay)
        , mStaging(
              {kFakeMaxRows, hiddenSize}, rt::DeviceType::kCPU, trt_edgellm::DataType::kHALF, "FakeVision::staging")
    {
        mModelType = multimodal::ModelType::QWEN2_5_VL;
        mOutputEmbedding = rt::Tensor(
            {kFakeMaxRows, hiddenSize}, rt::DeviceType::kGPU, trt_edgellm::DataType::kHALF, "FakeVision::output");
    }

    static float imageRowValue(int32_t pixel, int64_t row, int64_t channel)
    {
        return static_cast<float>((pixel * 7 + row * 13 + channel * 3 + 1) % kFakeModulus);
    }

    static int64_t layoutRows(int32_t pixel)
    {
        return 2 + pixel % 5;
    }

    bool tokenLayoutNeedsEncoder(rt::LLMGenerationRequest const& request) const noexcept override
    {
        return layoutFromEncoder && std::any_of(request.requests.begin(), request.requests.end(), [](auto const& item) {
            return !item.imageBuffers.empty();
        });
    }

    bool preprocess(rt::LLMGenerationRequest const& request, std::vector<std::vector<int32_t>>& batchedInputIds,
        tokenizer::Tokenizer const* tokenizer, rt::OptionalOutputTensor mropeCosSinOut, cudaStream_t stream,
        bool imageOnly, bool skipEncoderWork) override
    {
        if (failure == Failure::kPreprocessOnDevice)
        {
            putStreamInError(stream);
        }
        if (failure == Failure::kPreprocess || failure == Failure::kPreprocessOnDevice)
        {
            return false;
        }
        std::vector<int32_t> pixels;
        for (auto const& item : request.requests)
        {
            for (auto const& image : item.imageBuffers)
            {
                pixels.push_back(image.buffer->dataPointer<uint8_t>()[0]);
            }
        }
        if (!skipEncoderWork)
        {
            ++pixelPasses;
            mPixels = pixels;
            mRows.clear();
            for (int32_t const pixel : pixels)
            {
                mRows.push_back(layoutFromEncoder ? layoutRows(pixel) : kFakeRowsPerImage);
            }
        }
        std::vector<int64_t> const rows
            = layoutFromEncoder ? mRows : std::vector<int64_t>(pixels.size(), kFakeRowsPerImage);
        if (!imageOnly)
        {
            int32_t const placeholder = promptTokenIds("z").front();
            batchedInputIds.resize(std::max(batchedInputIds.size(), request.requests.size()));
            size_t image = 0;
            for (size_t index = 0; index < request.requests.size(); ++index)
            {
                std::vector<int32_t>& expanded = batchedInputIds[index];
                expanded.clear();
                for (int32_t const token :
                    tokenizer->encode(request.formattedRequests[index].formattedCompleteRequest, false))
                {
                    if (token != placeholder)
                    {
                        expanded.push_back(token);
                        continue;
                    }
                    int64_t const count = image < rows.size() ? rows[image] : 0;
                    ++image;
                    expanded.insert(expanded.end(), count, kFakeImageToken);
                }
            }
            if (mropeCosSinOut.has_value())
            {
                writeMrope(mropeCosSinOut->get(), static_cast<int64_t>(request.requests.size()), stream);
            }
        }
        mLastMediaTokenLengths = rows;
        int64_t const totalRows = std::accumulate(rows.begin(), rows.end(), int64_t{0});
        check::check(mOutputEmbedding.reshape({totalRows, mHiddenSize}), "FakeVisionRunner output reshape failed");
        return true;
    }

    bool infer(cudaStream_t stream) override
    {
        ++encoderRuns;
        if (failure == Failure::kInfer)
        {
            return false;
        }
        if (failure == Failure::kInferCudaCall)
        {
            // A non-sticky failure, cleared so that later CUDA calls in this process do not report it.
            cudaError_t const status = cudaMemcpyAsync(
                mOutputEmbedding.rawPointer(), mStaging.rawPointer(), 1, static_cast<cudaMemcpyKind>(-1), stream);
            static_cast<void>(cudaGetLastError());
            CUDA_CHECK(status);
        }
        half* const staging = mStaging.dataPointer<half>();
        size_t values = 0;
        for (size_t image = 0; image < mPixels.size(); ++image)
        {
            for (int64_t row = 0; row < mRows[image]; ++row)
            {
                for (int64_t channel = 0; channel < mHiddenSize; ++channel)
                {
                    staging[values++] = __float2half(imageRowValue(mPixels[image], row, channel));
                }
            }
        }
        if (mStagedUploadDelay.count() > 0)
        {
            CUDA_CHECK(cudaLaunchHostFunc(stream, sleepOnStream, &mStagedUploadDelay));
        }
        CUDA_CHECK(cudaMemcpyAsync(
            mOutputEmbedding.rawPointer(), staging, values * sizeof(half), cudaMemcpyHostToDevice, stream));
        return true;
    }

    bool validateAndFillConfig(std::string const& /*engineDir*/) override
    {
        return true;
    }

    bool allocateBuffer(cudaStream_t /*stream*/) override
    {
        return true;
    }

    std::vector<std::reference_wrapper<rt::Tensor>> getDeepstackFeatures() override
    {
        return {deepstack.begin(), deepstack.end()};
    }

    //! Deepstack feature buffers the encoder reports; never written.
    std::vector<rt::Tensor> deepstack;
    int32_t pixelPasses{0};
    int32_t encoderRuns{0};
    Failure failure{Failure::kNone};
    bool layoutFromEncoder{false};

private:
    //! Every rotary entry of position p holds kFakeMropeBase + p, far outside the [-1, 1] of a text-only table.
    static void writeMrope(rt::Tensor& table, int64_t rows, cudaStream_t stream)
    {
        auto const shape = table.getShape();
        int64_t const positions = shape[shape.getNumDims() - 2];
        int64_t const rotaryDim = shape[shape.getNumDims() - 1];
        check::check(table.reshape({rows, positions, rotaryDim}), "FakeVisionRunner MRope reshape failed");
        std::vector<float> host(static_cast<size_t>(rows * positions * rotaryDim));
        for (size_t index = 0; index < host.size(); ++index)
        {
            host[index] = kFakeMropeBase + static_cast<float>((static_cast<int64_t>(index) / rotaryDim) % positions);
        }
        CUDA_CHECK(cudaMemcpyAsync(
            table.rawPointer(), host.data(), host.size() * sizeof(float), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    int64_t mHiddenSize;
    std::chrono::milliseconds mStagedUploadDelay;
    rt::Tensor mStaging;
    std::vector<int32_t> mPixels;
    std::vector<int64_t> mRows;
};

//! A greedy request for @p prompt, one 'z' per image, with one image of uniform pixel per entry of @p pixels.
rt::LLMGenerationRequest makeImageRequest(std::string const& prompt, std::vector<uint8_t> const& pixels)
{
    rt::LLMGenerationRequest request = makeGreedyRequest(prompt, kMaxGenerateLength);
    for (uint8_t const pixel : pixels)
    {
        rt::Tensor image({1, 4, 4, 3}, rt::DeviceType::kCPU, trt_edgellm::DataType::kUINT8, "test::image");
        std::fill_n(image.dataPointer<uint8_t>(), 4 * 4 * 3, pixel);
        request.requests.front().imageBuffers.emplace_back(std::move(image));
    }
    return request;
}

Json makeCompactMediaConfig(bool mrope)
{
    Json config = makeCompactVanillaConfig();
    config["image_token_id"] = kFakeImageToken;
    if (mrope)
    {
        config["rope_scaling"] = Json{{"type", "mrope"}, {"mrope_section", {8, 12, 12}}};
    }
    return config;
}

class CompactMultimodalSessionTest : public CompactSessionTest
{
protected:
    std::filesystem::path stageModelDir() override
    {
        return writeModelDir(
            mrope() ? "compactMropeMediaSessionTests" : "compactMediaSessionTests", makeCompactMediaConfig(mrope()));
    }

    virtual bool mrope() const
    {
        return false;
    }

    //! A runtime over @p engine with a FakeVisionRunner; @p runner is pointed at it.
    rt::LLMInferenceRuntime makeMediaRuntime(std::unique_ptr<rt::EngineExecutor> engine, FakeVisionRunner*& runner,
        rt::ContextCacheConfig const& cacheConfig = {}, std::chrono::milliseconds stagedUploadDelay = {})
    {
        rt::ModelArtifacts artifacts = makeVanillaArtifacts(mModelDir, std::move(engine), mStream);
        auto vision = std::make_unique<FakeVisionRunner>(artifacts.deployment.base.hiddenSize, stagedUploadDelay);
        runner = vision.get();
        artifacts.visionRunner = std::move(vision);
        return rt::LLMInferenceRuntime{std::move(artifacts), mModelDir.string(), /*multimodalEngineDir=*/"", {},
            std::nullopt, mStream, cacheConfig};
    }

    //! What the blocking path generates for @p request alone, on an engine that never runs phase 8.
    std::vector<int32_t> blockingMediaOutput(rt::LLMGenerationRequest const& request, bool layoutFromEncoder = false)
    {
        FakeVisionRunner* runner = nullptr;
        auto runtime = makeMediaRuntime(makeFakeEngine(makeCompactMediaConfig(mrope())), runner);
        runner->layoutFromEncoder = layoutFromEncoder;
        rt::LLMGenerationResponse response;
        EXPECT_TRUE(runtime.handleRequest(request, response, mStream));
        return response.outputIds.empty() ? std::vector<int32_t>{} : response.outputIds.front();
    }
};

class CompactMropeMultimodalSessionTest : public CompactMultimodalSessionTest
{
protected:
    bool mrope() const override
    {
        return true;
    }
};

} // namespace

// Given a compact multimodal deployment with the encoder embedding cache, A (3 text, a 6-token image, 4 text) and C
//       (two images) admitted, and B a text request
// When A runs a chunk ending inside its image, then its final chunk beside C's chunk ending inside C's second image;
//      then C's final chunk, B's whole prompt and A's decode row share a step, and all decode to the end; then D, a
//      different prompt with A's image, is admitted and runs to the end
// Then every output equals the blocking path's, which depends on the image's pixels; A's and C's admissions each run
//      the encoder, and D's runs neither the encoder nor its pixel work
TEST_F(CompactMultimodalSessionTest, ImagesSplitAcrossChunksDecodeAsTheBlockingPath)
{
    rt::LLMGenerationRequest const requestA = makeImageRequest("abczdefg", {11});
    rt::LLMGenerationRequest const requestB = makeGreedyRequest("hijklmn", kMaxGenerateLength);
    rt::LLMGenerationRequest const requestC = makeImageRequest("qzrzs", {23, 37});
    rt::LLMGenerationRequest const imageAgain = makeImageRequest("tuzv", {11});
    std::vector<int32_t> const referenceA = blockingMediaOutput(requestA);
    std::vector<int32_t> const referenceB = blockingMediaOutput(requestB);
    std::vector<int32_t> const referenceC = blockingMediaOutput(requestC);
    std::vector<int32_t> const referenceImageAgain = blockingMediaOutput(imageAgain);
    ASSERT_NE(referenceA, blockingMediaOutput(makeImageRequest("abczdefg", {12})))
        << "the fixture cannot tell one image from another";

    FakeVisionRunner* runner = nullptr;
    auto runtime = makeMediaRuntime(makeCompactFake(makeCompactMediaConfig(false)), runner);
    auto session = openCompactSession(runtime);
    CompactDriver driver(*session);
    rt::ResidentRef const a = driver.admit(requestA, 1);
    ASSERT_EQ(driver.promptLength(a), 3 + kFakeRowsPerImage + 4);
    ASSERT_TRUE(driver.run({driver.chunk(a, 5)}).ok);
    rt::ResidentRef const c = driver.admit(requestC, 3);
    ASSERT_EQ(driver.promptLength(c), 3 + 2 * kFakeRowsPerImage);
    ASSERT_TRUE(driver.run({driver.rest(a), driver.chunk(c, 9)}).ok);
    rt::ResidentRef const b = driver.admit(requestB, 2);
    ASSERT_TRUE(driver.run({driver.rest(c), driver.chunk(b, 7), driver.decode(a)}).ok);
    driver.drain();
    EXPECT_EQ(driver.output(1), referenceA);
    EXPECT_EQ(driver.output(2), referenceB);
    EXPECT_EQ(driver.output(3), referenceC);
    EXPECT_EQ(runner->encoderRuns, 2) << "one encoder run per media admission";

    rt::ResidentRef const d = driver.admit(imageAgain, 4);
    EXPECT_EQ(runner->encoderRuns, 2) << "D's image is held by the encoder embedding cache";
    EXPECT_EQ(runner->pixelPasses, 2);
    ASSERT_TRUE(driver.run({driver.rest(d)}).ok);
    driver.drain();
    EXPECT_EQ(driver.output(4), referenceImageAgain);
}

// Given a compact multimodal deployment without the encoder embedding cache, whose encoder uploads its output from a
//       pinned staging buffer it rewrites at every run, the upload queued behind slower device work
// When image requests A and B are admitted back to back at one step boundary, and their prompts run in one step and
//      both decode to the end
// Then each output equals the blocking path's for the same request alone
TEST_F(CompactMultimodalSessionTest, BackToBackMediaAdmissionsKeepTheirOwnEncoderOutputs)
{
    rt::LLMGenerationRequest const requestA = makeImageRequest("abzc", {11});
    rt::LLMGenerationRequest const requestB = makeImageRequest("dezf", {23});
    std::vector<int32_t> const referenceA = blockingMediaOutput(requestA);
    std::vector<int32_t> const referenceB = blockingMediaOutput(requestB);
    ASSERT_NE(referenceA, blockingMediaOutput(makeImageRequest("abzc", {23})))
        << "the fixture cannot tell A's image from B's";
    rt::ContextCacheConfig noEncoderCache;
    noEncoderCache.encoderEmbeddingCacheBudgetBytes = 0;
    FakeVisionRunner* runner = nullptr;
    auto runtime = makeMediaRuntime(makeCompactFake(makeCompactMediaConfig(false)), runner, noEncoderCache,
        /*stagedUploadDelay=*/std::chrono::milliseconds{200});
    auto session = openCompactSession(runtime);
    CompactDriver driver(*session);
    rt::ResidentRef const a = driver.admit(requestA, 1);
    rt::ResidentRef const b = driver.admit(requestB, 2);
    ASSERT_TRUE(driver.run({driver.rest(a), driver.rest(b)}).ok);
    driver.drain();
    EXPECT_EQ(driver.output(1), referenceA);
    EXPECT_EQ(driver.output(2), referenceB);
}

// Given a compact multimodal deployment with the encoder embedding cache, whose vision runner computes an image's
//       token count only when its encoder runs (as Nemotron-Omni does for a video), and image requests A and B whose
//       images expand to 3 and 6 tokens
// When A, B, and then C, a new prompt with A's image (held by the encoder embedding cache), are each admitted to the
//      idle session and run to the end
// Then every such request needs the session to itself; each prompt carries its own image's token count; every output
//      equals the blocking path's for the same request alone; and every admission ran the encoder
TEST_F(CompactMultimodalSessionTest, MediaLaidOutByItsEncoderRunsAloneWithItsOwnLayout)
{
    std::vector<rt::LLMGenerationRequest> const requests{
        makeImageRequest("abzc", {11}), makeImageRequest("dezf", {24}), makeImageRequest("ghzij", {11})};
    std::vector<int32_t> const promptLengths{3 + 3, 3 + 6, 4 + 3};
    std::vector<std::vector<int32_t>> references;
    for (rt::LLMGenerationRequest const& request : requests)
    {
        references.push_back(blockingMediaOutput(request, /*layoutFromEncoder=*/true));
    }
    FakeVisionRunner* runner = nullptr;
    auto runtime = makeMediaRuntime(makeCompactFake(makeCompactMediaConfig(false)), runner);
    runner->layoutFromEncoder = true;
    auto session = openCompactSession(runtime);
    CompactDriver driver(*session);
    for (size_t index = 0; index < requests.size(); ++index)
    {
        SCOPED_TRACE("request " + std::to_string(index));
        EXPECT_TRUE(session->requiresExclusiveAdmission(requests[index]));
        rt::RequestId const requestId = static_cast<rt::RequestId>(index + 1);
        rt::ResidentRef const ref = driver.admit(requests[index], requestId);
        EXPECT_EQ(driver.promptLength(ref), promptLengths[index]);
        EXPECT_TRUE(session->exclusiveResident());
        ASSERT_TRUE(driver.run({driver.rest(ref)}).ok);
        driver.drain();
        EXPECT_EQ(driver.output(requestId), references[index]);
        EXPECT_EQ(runner->encoderRuns, static_cast<int32_t>(index + 1));
    }
}

// Given a compact M-RoPE deployment whose encoder writes a media position table (every entry of position p is 500 + p)
//       and M, an image request (9 tokens)
// When M's prompt runs in two chunks, the second beside T's prompt; M decodes beside U's prompt and T's decode row and
//      once more beside both; and once all have finished, text requests V fill every slot, M's former one included
// Then every token of M, prompt and decode alike, reads its position from M's media table, and every token of T, U
//      and V reads text-only values
TEST_F(CompactMropeMultimodalSessionTest, MediaTableWrittenOnceAndDecodeRowsKeepIt)
{
    auto engine = makeCompactFake(makeCompactMediaConfig(true));
    engine->recordFloatBinding(binding_names::kRopeCosSin);
    auto* const raw = engine.get();
    FakeVisionRunner* runner = nullptr;
    auto runtime = makeMediaRuntime(std::move(engine), runner);
    auto session = openCompactSession(runtime);
    CompactDriver driver(*session);
    rt::ResidentRef const m = driver.admit(makeImageRequest("abzc", {11}), 1);
    ASSERT_EQ(driver.promptLength(m), 9);
    ASSERT_TRUE(driver.run({driver.chunk(m, 4)}).ok);
    rt::ResidentRef const t = driver.admit(makeTokenizedRequest(promptOf(5, 1)), 2);
    ASSERT_TRUE(driver.run({driver.chunk(m, 5), driver.chunk(t, 5)}).ok);
    rt::ResidentRef const u = driver.admit(makeTokenizedRequest(promptOf(3, 2)), 3);
    ASSERT_TRUE(driver.run({driver.chunk(u, 3), driver.decode(m), driver.decode(t)}).ok);
    ASSERT_TRUE(driver.run(driver.decodeAll()).ok);
    driver.drain();
    std::vector<std::vector<double>> rounds = raw->preparedFloatBinding(binding_names::kRopeCosSin);
    constexpr int32_t kRefillLength{12};
    std::vector<rt::ScheduledSequenceDescriptor> refill;
    bool reusesMediaSlot = false;
    for (int32_t index = 0; index < kCompactBatchSize; ++index)
    {
        rt::ResidentRef const v = driver.admit(makeTokenizedRequest(promptOf(kRefillLength, 3 + index)), 4 + index);
        reusesMediaSlot = reusesMediaSlot || v.slot == m.slot;
        refill.push_back(driver.chunk(v, kRefillLength));
    }
    ASSERT_TRUE(reusesMediaSlot);
    ASSERT_TRUE(driver.run(refill).ok);

    // The M position of each row of the first four forwards and the refill, -1 for a text row.
    std::vector<std::vector<int32_t>> const layouts{{0, 1, 2, 3}, {4, 5, 6, 7, 8, -1, -1, -1, -1, -1},
        {-1, -1, -1, 9, -1}, {10, -1, -1}, std::vector<int32_t>(kCompactBatchSize * kRefillLength, -1)};
    ASSERT_GE(rounds.size(), layouts.size() - 1);
    rounds.resize(layouts.size() - 1);
    rounds.push_back(raw->preparedFloatBinding(binding_names::kRopeCosSin).back());
    for (size_t step = 0; step < layouts.size(); ++step)
    {
        std::vector<double> const& table = rounds[step];
        size_t const rotaryDim = table.size() / layouts[step].size();
        ASSERT_GT(rotaryDim, 0U);
        for (size_t entry = 0; entry < table.size(); ++entry)
        {
            int32_t const position = layouts[step][entry / rotaryDim];
            if (position >= 0)
            {
                EXPECT_EQ(table[entry], kFakeMropeBase + position) << "step " << step << " entry " << entry;
            }
            else
            {
                EXPECT_LE(std::abs(table[entry]), 1.0) << "step " << step << " entry " << entry;
            }
        }
    }
}

// Given deployments whose layers all have phase-8 kernels, each breaking one compact precondition (B = 1, vision
//       encoder rows not of the hidden size, vision-bidirectional attention, audio encoder rows wider than the hidden
//       size, deepstack layers the decoder does not take, a decode profile admitting phases up to 3 only, a dedicated
//       compact context that runs out of device memory after the media reservation was allocated), and one whose audio
//       encoder rows are of the hidden size
// When a session with the default SessionOptions opens on each, then a session allowing compact execution, and a
//      second one once any allocation would succeed
// Then no forward runs and the default session reports compact execution disabled for the session; each breaking
//      deployment reports compact execution unavailable with its precondition's reason, the last one reports compact
//      execution, no CUDA error is left pending, the second session reports the same, and the engine is asked to
//      dedicate a context to the prefill profile's compact steps only
TEST_F(CompactMultimodalSessionTest, StaticPreconditionsRefuseCompactWithTheirReason)
{
    using Adjust = std::function<void(rt::ModelArtifacts&, HookedFakeEngine&)>;
    auto const vision = [](int64_t widthDelta, int32_t deepstackLayers, bool failContext = false) -> Adjust {
        return [=](rt::ModelArtifacts& artifacts, HookedFakeEngine& engine) {
            auto runner = std::make_unique<FakeVisionRunner>(artifacts.deployment.base.hiddenSize + widthDelta);
            for (int32_t layer = 0; layer < deepstackLayers; ++layer)
            {
                runner->deepstack.emplace_back(rt::Coords{8, 8}, rt::DeviceType::kGPU, trt_edgellm::DataType::kHALF);
            }
            artifacts.visionRunner = std::move(runner);
            engine.failDedicatedContext = failContext;
        };
    };
    auto const audio = [](int64_t widthDelta) -> Adjust {
        return [widthDelta](rt::ModelArtifacts& artifacts, HookedFakeEngine&) {
            artifacts.audioRunner
                = std::make_unique<FakeVisionRunner>(artifacts.deployment.base.hiddenSize + widthDelta);
        };
    };
    Adjust const decodeProfileUpToPhase3
        = [](rt::ModelArtifacts&, HookedFakeEngine& engine) { engine.advertisePhase8Profile(kPhase3); };
    Json const media = makeCompactMediaConfig(false);
    Json singleRow = media;
    singleRow["builder_config"]["max_batch_size"] = 1;
    Json bidirectional = media;
    bidirectional["use_vision_bidirectional_attention"] = true;
    struct Case
    {
        Json config;
        Adjust adjust;
        //! Empty when the deployment keeps compact execution.
        std::string reason;
    };
    std::vector<Case> const cases{
        {singleRow, nullptr, "maxBatchSize is 1"},
        {media, vision(8, 0), "vision encoder's output rows are not FP16 rows"},
        {bidirectional, nullptr, "image tokens attend bidirectionally"},
        {media, audio(8), "audio encoder's output rows are not FP16 rows"},
        {media, audio(0), ""},
        {media, vision(0, 2), "vision encoder produces 2 deepstack feature layers but the decoder engine takes 0"},
        {media, decodeProfileUpToPhase3, "the engine's decode profile does not admit compact execution"},
        {media, vision(0, 0, /*failContext=*/true),
            "the resources of compact execution could not be allocated: CUDA runtime error in cudaMalloc"},
    };
    for (Case const& expected : cases)
    {
        SCOPED_TRACE(expected.reason);
        auto const dir = writeModelDir("compactPrecondition", expected.config);
        {
            auto engine = makeCompactFake<HookedFakeEngine>(expected.config);
            auto* const raw = engine.get();
            rt::ModelArtifacts artifacts = makeVanillaArtifacts(dir, std::move(engine), mStream);
            if (expected.adjust)
            {
                expected.adjust(artifacts, *raw);
            }
            rt::DeploymentOptions options{};
            options.engineDir = dir.string();
            rt::LLMInferenceRuntime runtime{std::move(artifacts), std::move(options), mStream};
            for (int32_t open = 0; open < 3; ++open)
            {
                std::unique_ptr<rt::RuntimeSession> session
                    = open == 0 ? runtime.openSession(mStream) : runtime.openSession(mStream, kCompactSession);
                ASSERT_NE(session, nullptr);
                rt::SessionCapabilities const capabilities = session->capabilities();
                std::string const reason = open == 0 ? "compact execution disabled for this session" : expected.reason;
                EXPECT_EQ(capabilities.compact, reason.empty()) << "open " << open << ": " << capabilities.reason;
                EXPECT_NE(capabilities.reason.find(reason), std::string::npos) << capabilities.reason;
                EXPECT_EQ(cudaGetLastError(), cudaSuccess) << "a failed allocation's error is still pending";
                if (open == 1)
                {
                    raw->failDedicatedContext = false;
                }
            }
            EXPECT_TRUE(raw->preparedRounds().empty()) << "a session open ran a forward";
            if (expected.reason.empty())
            {
                EXPECT_EQ(raw->dedicatedProfiles, (std::vector<int32_t>{kPrefillProfile, kPrefillProfile}));
            }
        }
        std::filesystem::remove_all(dir);
    }
}

// Given a compact multimodal deployment under a context cache, without the encoder embedding cache, with A resident
// When an image request M is admitted while the encoder or its preprocess fails with the stream healthy, then once it
//      recovers, and runs to the end; a too-long image request is admitted and M is offered to a full session; and M
//      is admitted while the encoder fails on the device (a failed CUDA call, or a preprocess leaving the stream in
//      error)
// Then each healthy failure throws an error other than SessionLostError and keeps no lease or slot, so the recovered
//      M takes the slot after A's and decodes as the blocking path does; the too-long request and the full-session
//      offer are refused before any pixel work or encoder run; and the device failure throws SessionLostError
TEST_F(CompactMultimodalSessionTest, RefusedOrFailedMediaAdmissionsKeepNothingAndADeviceFailureLosesTheSession)
{
    using Failure = FakeVisionRunner::Failure;
    rt::LLMGenerationRequest const media = makeImageRequest("abzcd", {17});
    std::vector<int32_t> const reference = blockingMediaOutput(media);
    for (Failure const deviceFailure : {Failure::kInferCudaCall, Failure::kPreprocessOnDevice})
    {
        SCOPED_TRACE(deviceFailure == Failure::kInferCudaCall ? "encoder CUDA call" : "preprocess stream in error");
        FakeVisionRunner* runner = nullptr;
        auto runtime = makeMediaRuntime(makeCompactFake(makeCompactMediaConfig(false)), runner, compactCacheConfig());
        auto session = openCompactSession(runtime);
        CompactDriver driver(*session);
        rt::ResidentRef const a = driver.admit(makeTokenizedRequest(promptOf(3, 0)), 1);
        ASSERT_TRUE(driver.run({driver.rest(a)}).ok);
        auto const freePages = [&runtime]() { return runtime.getContextCacheMetrics().value().baseKvPages.free; };
        int32_t const freeBeside = freePages();

        // 27 text + 6 image tokens = 33 > max_input_len 32, within the KV capacity of 64.
        EXPECT_THROW(session->admit(makeImageRequest(std::string(27, 'b') + "z", {11}), 2), std::runtime_error);
        EXPECT_EQ(runner->pixelPasses, 0);
        EXPECT_EQ(runner->encoderRuns, 0);

        for (Failure const healthyFailure : {Failure::kInfer, Failure::kPreprocess})
        {
            runner->failure = healthyFailure;
            try
            {
                session->admit(media, 3);
                ADD_FAILURE() << "the refused admission did not throw";
            }
            catch (rt::SessionLostError const& error)
            {
                ADD_FAILURE() << "a failure with the stream healthy lost the session: " << error.what();
            }
            catch (std::runtime_error const&)
            {
            }
            EXPECT_EQ(session->residents(), (std::vector<rt::ResidentRef>{a}));
            EXPECT_EQ(freePages(), freeBeside) << "the failed admission kept its lease";
        }

        runner->failure = Failure::kNone;
        rt::ResidentRef const m = driver.admit(media, 4);
        EXPECT_EQ(m.slot, 1) << "a failed admission kept its slot";
        ASSERT_TRUE(driver.run({driver.rest(m), driver.decode(a)}).ok);
        driver.drain();
        EXPECT_EQ(driver.output(4), reference);

        std::vector<rt::ScheduledSequenceDescriptor> prompts;
        for (int32_t index = 0; index < kCompactBatchSize; ++index)
        {
            prompts.push_back(driver.rest(driver.admit(makeTokenizedRequest(promptOf(3, index)), 10 + index)));
        }
        int32_t const pixelPasses = runner->pixelPasses;
        int32_t const encoderRuns = runner->encoderRuns;
        EXPECT_EQ(session->admit(media, 20).status, rt::AdmissionResult::Status::kNoCapacity);
        EXPECT_EQ(runner->pixelPasses, pixelPasses);
        EXPECT_EQ(runner->encoderRuns, encoderRuns);
        ASSERT_TRUE(driver.run(prompts).ok);
        driver.drain();

        runner->failure = deviceFailure;
        EXPECT_THROW(session->admit(media, 21), rt::SessionLostError);
        recoverStream(mStream);
    }
}

// Given compact deployments gathering Gemma4 per-layer embeddings from a resident PLE pool (row t + 1 holds token t)
//       or through a PLE row cache (128 + 1 slots over a 256-row table), each with A decoding and P1 (7 tokens, its
//       first and fourth equal) and P2 (3 tokens) admitted; and PLE row caches one usable slot short of what a compact
//       step can need (the step's 128 tokens over a 256-row table, or a 100-row table)
// When a session allowing compact execution opens on each; and on the compact ones a step runs A's decode row, P1's
//      first 4 tokens and P2's prompt, and the next P1's last 3 tokens beside the decode rows of A and P2
// Then the short caches report compact execution unavailable, naming the slots a compact step needs; and each step's
//      forward is handed one PLE row per token in execution order (prefill chunks in plan order, then each decode
//      row's last sampled token): the token's pool row, or a non-null cache row shared exactly by equal tokens
TEST_F(CompactSessionTest, CompactStepsGatherPleRowsInExecutionOrder)
{
    constexpr int64_t kPleLayers{2};
    struct Case
    {
        rt::PleResidency residency;
        int64_t tableRows;
        int32_t cacheSlots;
        //! Empty when the session is compact.
        std::string reason;
    };
    std::vector<Case> const cases{{rt::PleResidency::kRESIDENT, kVocabSize, 129, ""},
        {rt::PleResidency::kCACHED, 256, 129, ""},
        {rt::PleResidency::kCACHED, 256, 128, "at least 129 PLE cache slots"},
        {rt::PleResidency::kCACHED, 100, 100, "at least 101 PLE cache slots"}};
    for (Case const& expected : cases)
    {
        bool const cached = expected.residency == rt::PleResidency::kCACHED;
        SCOPED_TRACE("table rows " + std::to_string(expected.tableRows) + ", cache slots "
            + std::to_string(expected.cacheSlots));
        auto engine = makeCompactFake(makeCompactVanillaConfig());
        engine->recordBinding("ple_row_map");
        auto* const raw = engine.get();
        rt::ModelArtifacts artifacts = makeVanillaArtifacts(mModelDir, std::move(engine), mStream);
        rt::LLMEngineConfig& config = artifacts.deployment.base;
        config.gemmaPleEnabled = true;
        config.gemmaPleLayerCount = kPleLayers;
        config.gemmaPleHiddenSize = 1;
        config.plePool = rt::PlePoolConfig{expected.tableRows, kPleLayers, rt::PleTable::DType::kFP16, 1.0};
        config.plePool.source = {"weight", rt::PleTable::DType::kFP16, expected.tableRows * kPleLayers * 2, 1.0,
            {{0, "003c"}}, "ple_table.bin", 0};
        config.plePool.checkpointDir = mModelDir.string();
        std::vector<uint16_t> const table(static_cast<size_t>(expected.tableRows * kPleLayers), 0x3C00);
        std::ofstream(mModelDir / "ple_table.bin", std::ios::binary)
            .write(reinterpret_cast<char const*>(table.data()),
                static_cast<std::streamsize>(table.size() * sizeof(uint16_t)));
        rt::DeploymentOptions options{};
        options.engineDir = mModelDir.string();
        options.pleConfig.residency = expected.residency;
        options.pleConfig.cacheSlots = expected.cacheSlots;
        rt::LLMInferenceRuntime runtime{std::move(artifacts), std::move(options), mStream};
        if (!expected.reason.empty())
        {
            rt::SessionCapabilities const capabilities = runtime.openSession(mStream, kCompactSession)->capabilities();
            EXPECT_FALSE(capabilities.compact);
            EXPECT_NE(capabilities.reason.find(expected.reason), std::string::npos) << capabilities.reason;
            continue;
        }
        auto session = openCompactSession(runtime);
        CompactDriver driver(*session);
        auto const tokenOf = [](rt::StepResult const& result, rt::ResidentRef ref) {
            auto const it = std::find_if(
                result.deltas.begin(), result.deltas.end(), [ref](auto const& delta) { return delta.first == ref; });
            return it == result.deltas.end() ? -1 : it->second.tokenIds.at(0);
        };
        auto const expectRows = [&](std::vector<int32_t> const& tokens) {
            std::vector<int32_t> const& rows = raw->preparedBinding("ple_row_map").back();
            ASSERT_EQ(rows.size(), tokens.size());
            for (size_t i = 0; i < tokens.size(); ++i)
            {
                EXPECT_TRUE(cached ? rows[i] != 0 : rows[i] == tokens[i] + 1) << "token " << i << " row " << rows[i];
                for (size_t j = 0; cached && j < i; ++j)
                {
                    EXPECT_EQ(rows[i] == rows[j], tokens[i] == tokens[j]) << "tokens " << j << " and " << i;
                }
            }
        };
        rt::ResidentRef const a = driver.admit(makeTokenizedRequest({40, 41}), 1);
        rt::StepResult const prefillA = driver.run({driver.rest(a)});
        ASSERT_TRUE(prefillA.ok);
        rt::ResidentRef const p1 = driver.admit(makeTokenizedRequest({50, 51, 52, 50, 54, 55, 56}), 2);
        rt::ResidentRef const p2 = driver.admit(makeTokenizedRequest({60, 61, 62}), 3);
        rt::StepResult const mixed = driver.run({driver.decode(a), driver.chunk(p1, 4), driver.rest(p2)});
        ASSERT_TRUE(mixed.ok);
        expectRows({50, 51, 52, 50, 60, 61, 62, tokenOf(prefillA, a)});
        ASSERT_TRUE(driver.run({driver.rest(p1), driver.decode(a), driver.decode(p2)}).ok);
        expectRows({54, 55, 56, tokenOf(mixed, a), tokenOf(mixed, p2)});
    }
}
