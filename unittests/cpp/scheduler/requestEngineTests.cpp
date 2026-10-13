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

// The actor's admission, cancellation and shutdown paths, exercised through the stepped seam so
// none of this needs a device. What is under test is the actor's tick loop and its bookkeeping --
// drain, admit, commit, publish, FCFS blocking -- not the runtime's generation, which has its own
// tests (llmInferenceRuntimeAssemblyTests drives the real stepper).

#include "runtime/runtimeStepper.h"
#include "runtime/streaming.h"
#include "scheduler/engineChannel.h"
#include "scheduler/requestEngine.h"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

using namespace trt_edgellm::rt;
using namespace trt_edgellm::rt::scheduler;
using namespace std::chrono_literals;

namespace
{

//! One user message carrying @p prompt: the smallest request the runtime API accepts.
LLMGenerationRequest makeRequest(std::string prompt)
{
    Message::MessageContent content;
    content.type = "text";
    content.content = std::move(prompt);

    Message message;
    message.contents.push_back(std::move(content));

    LLMGenerationRequest request;
    // The sampling fields have no default initializer; left indeterminate, two requests built the
    // same way can compare batch-incompatible on stack garbage.
    request.temperature = 0.7F;
    request.topP = 0.9F;
    request.topK = 40;
    request.maxGenerateLength = 128;
    request.requests.resize(1);
    request.requests[0].messages.push_back(std::move(message));
    return request;
}

//! Reads back what makeRequest() put in, so a test can tell requests apart by their payload.
std::string promptOf(LLMGenerationRequest const& request)
{
    if (request.requests.empty() || request.requests[0].messages.empty()
        || request.requests[0].messages[0].contents.empty())
    {
        return "empty";
    }
    return request.requests[0].messages[0].contents[0].content;
}

//! Blocks the actor inside a step until released, so a test can observe the resident state.
class Gate
{
public:
    void wait()
    {
        std::unique_lock<std::mutex> lock(mMutex);
        mCv.wait(lock, [this] { return mOpen; });
    }

    void open()
    {
        {
            std::lock_guard<std::mutex> lock(mMutex);
            mOpen = true;
        }
        mCv.notify_all();
    }

private:
    std::mutex mMutex;
    std::condition_variable mCv;
    bool mOpen{false};
};

//! Polls @p done until it holds or @p budget runs out; a bounded wait so a regression fails
//! instead of hanging the suite.
template <typename Predicate>
bool waitUntil(Predicate&& done, std::chrono::milliseconds budget = 5s)
{
    auto const deadline = std::chrono::steady_clock::now() + budget;
    while (!done())
    {
        if (std::chrono::steady_clock::now() >= deadline)
        {
            return false;
        }
        std::this_thread::sleep_for(1ms);
    }
    return true;
}

//! What the fake batches do. Written by the test before it submits, read on the actor thread;
//! the counters are the test's window into what the actor did.
struct FakeScript
{
    //! Tokens each resident produces before it finishes, one per decode tick.
    int32_t tokensPerRequest{1};
    //! Extra tokens a joiner produces on top of tokensPerRequest, so a test can make a joiner
    //! outlive its founder whatever tick it was seated on.
    int32_t joinerExtraTokens{0};
    //! Runs at the top of every decode tick on the actor thread: a test parks the actor here.
    std::function<void()> beforeDecode;
    //! Refuse this many admissions with kNoCapacity before seating any.
    int32_t refusalsLeft{0};
    //! Reject every admission outright.
    bool rejectAdmissions{false};
    //! Every decode tick reports ok = false.
    bool failDecode{false};
    //! The first decode of this founding (1-based) throws; 0 never throws.
    int32_t throwOnDecodeOfBatch{0};
    //! Poll the runtime channel's cancel flag at the top of each step, as the runtime does.
    bool honorCancel{true};
    //! Drop finished joiners without reporting them: the batch loses them.
    bool abandonJoiners{false};
    //! beginStepped refuses (null) or throws.
    bool refuseFounding{false};
    bool throwOnFounding{false};

    std::atomic<int32_t> batches{0};
    std::atomic<int32_t> admissions{0};
    std::atomic<RequestId> lastAdmittedRequestId{kInvalidRequestId};
    std::atomic<int32_t> completed{0};
    std::atomic<int32_t> decodes{0};
};

//! A stand-in for the runtime's stepped batch: seats whatever the engine admits, emits one token
//! per resident per decode, finishes a resident when its budget is spent, and materializes the
//! prompt back as the output text so a test can tell results apart.
class FakeStepped final : public SteppedExecution
{
public:
    FakeStepped(FakeScript& script, LLMGenerationRequest const& founder)
        : mScript(script)
        , mBatchIndex(script.batches.fetch_add(1) + 1)
    {
        seat(founder, /*joiner=*/false);
    }

    std::vector<ResidentRef> residents() const override
    {
        std::vector<ResidentRef> refs;
        for (auto const& resident : mResidents)
        {
            refs.push_back(resident.ref);
        }
        return refs;
    }

    AdmissionResult admit(LLMGenerationRequest const& request, int32_t, RequestId requestId) override
    {
        if (mScript.refusalsLeft > 0)
        {
            --mScript.refusalsLeft;
            return {AdmissionResult::Status::kNoCapacity, {}, {}};
        }
        if (mScript.rejectAdmissions)
        {
            return {AdmissionResult::Status::kRejected, {}, "the fake batch rejects joiners"};
        }
        mScript.admissions.fetch_add(1);
        mScript.lastAdmittedRequestId.store(requestId);
        return {AdmissionResult::Status::kAdmitted, seat(request, /*joiner=*/true), {}};
    }

    StepResult prefill(ImmutablePrefillBatch const& batch) override
    {
        for (auto& resident : mResidents)
        {
            if (resident.ref == batch.target)
            {
                resident.prefilled = true;
            }
        }
        StepResult result;
        result.ok = true;
        return result;
    }

    StepResult decode(ImmutableDecodeBatch const&) override
    {
        mScript.decodes.fetch_add(1);
        if (mScript.beforeDecode)
        {
            mScript.beforeDecode();
        }
        if (mScript.throwOnDecodeOfBatch == mBatchIndex)
        {
            mScript.throwOnDecodeOfBatch = 0;
            throw std::runtime_error("device lost mid-decode");
        }
        StepResult result;
        if (mScript.failDecode)
        {
            return result; // ok stays false and nothing advanced: a failed step produced nothing
        }
        result.ok = true;
        for (auto it = mResidents.begin(); it != mResidents.end();)
        {
            if (mScript.honorCancel && it->channel != nullptr && it->channel->isCancelled())
            {
                result.finished.emplace_back(it->ref, snapshot(*it, FinishReason::kCancelled));
                it = mResidents.erase(it);
                continue;
            }
            it->tokens.push_back(++it->generated);
            if (it->generated >= mScript.tokensPerRequest + (it->joiner ? mScript.joinerExtraTokens : 0))
            {
                if (mScript.abandonJoiners && it->joiner)
                {
                    it = mResidents.erase(it); // gone, and nobody is told
                    continue;
                }
                result.finished.emplace_back(it->ref, snapshot(*it, FinishReason::kLength));
                mScript.completed.fetch_add(1);
                it = mResidents.erase(it);
                continue;
            }
            ++it;
        }
        return result;
    }

    LLMGenerationResponse materialize(BatchResult const& result, std::vector<std::string> const&) const override
    {
        LLMGenerationResponse response;
        response.outputIds.push_back(result.tokenIds);
        // The fake tags each snapshot with its handle (see snapshot()) so the prompt comes back.
        auto const prompt = mPrompts.find(result.rawBatchedInputIds.empty() ? -1 : result.rawBatchedInputIds.front());
        response.outputTexts.push_back(prompt == mPrompts.end() ? "materialized" : prompt->second);
        response.finishReasons.push_back(result.terminalReason);
        return response;
    }

    void abort() noexcept override
    {
        mResidents.clear();
    }

    bool finish(LLMGenerationResponse&) override
    {
        return true;
    }

private:
    struct Resident
    {
        ResidentRef ref;
        std::shared_ptr<StreamChannel> channel;
        std::vector<int32_t> tokens;
        int32_t generated{0};
        bool prefilled{false};
        bool joiner{false};
    };

    ResidentRef seat(LLMGenerationRequest const& request, bool joiner)
    {
        Resident resident;
        resident.ref = ResidentRef{mNextHandle++, 0};
        resident.channel = request.streamChannels.empty() ? nullptr : request.streamChannels.front();
        resident.joiner = joiner;
        mPrompts.emplace(resident.ref.slot, promptOf(request));
        mResidents.push_back(std::move(resident));
        return mResidents.back().ref;
    }

    static BatchResult snapshot(Resident const& resident, FinishReason reason)
    {
        BatchResult result;
        result.tokenIds = resident.tokens;
        result.generateLength = resident.generated;
        result.terminalReason = reason;
        result.rawBatchedInputIds = {resident.ref.slot}; // identity tag for materialize()
        return result;
    }

    FakeScript& mScript;
    int32_t const mBatchIndex;
    int32_t mNextHandle{0};
    std::vector<Resident> mResidents;
    std::unordered_map<int32_t, std::string> mPrompts;
};

//! The engine's stepped seam bound to a script.
RequestEngine::SteppedFactory fakeBatches(FakeScript& script)
{
    return [&script](LLMGenerationRequest& request, cudaStream_t) -> std::unique_ptr<SteppedExecution> {
        if (script.throwOnFounding)
        {
            throw std::runtime_error("beginStepped exploded");
        }
        if (script.refuseFounding)
        {
            return nullptr;
        }
        return std::make_unique<FakeStepped>(script, request);
    };
}

//! Parks the actor at its first decode tick until the gate opens; later ticks run through.
void holdFirstDecode(FakeScript& script, Gate& gate)
{
    auto held = std::make_shared<std::atomic<bool>>(false);
    script.beforeDecode = [&gate, held] {
        if (!held->exchange(true))
        {
            gate.wait();
        }
    };
}

} // namespace

TEST(RequestEngineTests, SubmitReturnsEarlyAndGetDeliversTheResponse)
{
    FakeScript script;
    Gate gate;
    holdFirstDecode(script, gate);
    RequestEngine engine(fakeBatches(script), EngineConfig{});

    auto const start = std::chrono::steady_clock::now();
    RequestHandle handle = engine.submit(makeRequest("a"));
    auto const elapsed = std::chrono::steady_clock::now() - start;
    // The whole point of the actor: submit posts and returns, it does not wait for the GPU.
    EXPECT_LT(elapsed, 200ms);
    EXPECT_NE(handle.id(), kInvalidRequestId);
    gate.open();
    EXPECT_NO_THROW(handle.get());

    // And the other half of the round trip: get() hands back what the batch materialized.
    FakeScript echo;
    RequestEngine echoEngine(fakeBatches(echo), EngineConfig{});
    LLMGenerationResponse response = echoEngine.submit(makeRequest("hello")).get();
    ASSERT_EQ(response.outputTexts.size(), 1U);
    EXPECT_EQ(response.outputTexts[0], "hello");
}

TEST(RequestEngineTests, RequestIdsAreUniqueAndNeverZero)
{
    FakeScript script;
    RequestEngine engine(fakeBatches(script), EngineConfig{});
    std::vector<RequestId> ids;
    for (int32_t i = 0; i < 8; ++i)
    {
        RequestHandle handle = engine.submit(makeRequest("p"));
        EXPECT_NE(handle.id(), kInvalidRequestId);
        ids.push_back(handle.id());
        handle.get();
    }
    for (size_t i = 1; i < ids.size(); ++i)
    {
        EXPECT_GT(ids[i], ids[i - 1]) << "ids must be monotonic so a late cancel cannot match a reused id";
    }
}

TEST(RequestEngineTests, RequestsRunOneAtATimeAndInOrder)
{
    FakeScript script;
    RequestEngine engine(fakeBatches(script), EngineConfig{}); // maxBatchSize 1: one founding at a time

    std::vector<RequestHandle> handles;
    for (int32_t i = 0; i < 5; ++i)
    {
        handles.push_back(engine.submit(makeRequest(std::to_string(i))));
    }
    for (int32_t i = 0; i < 5; ++i)
    {
        LLMGenerationResponse const response = handles[static_cast<size_t>(i)].get();
        ASSERT_EQ(response.outputTexts.size(), 1U);
        EXPECT_EQ(response.outputTexts[0], std::to_string(i)) << "admission is FCFS";
    }
    EXPECT_EQ(script.batches.load(), 5) << "maxBatchSize is 1, so every request founds its own batch";
    EXPECT_EQ(script.admissions.load(), 0);
}

TEST(RequestEngineTests, SubmitRejectsPerRequestLoraOutright)
{
    // This version of in-flight batching does not support per-request LoRA, and the refusal
    // happens at submit. Deliberately not a SubmitError -- that means transient
    // backpressure and invites a retry, while this request would be refused forever.
    FakeScript script;
    RequestEngine engine(fakeBatches(script), EngineConfig{});
    LLMGenerationRequest withLora = makeRequest("hello");
    withLora.loraWeightsName = "adapter-a";
    EXPECT_THROW(engine.submit(std::move(withLora)), std::invalid_argument);

    // The refusal reserves nothing and the engine keeps serving.
    EXPECT_NO_THROW(engine.submit(makeRequest("after")).get());
    EngineMetrics const m = engine.metrics();
    EXPECT_EQ(m.submitted, 1U);
    EXPECT_EQ(m.refused, 0U) << "an unsupported request is not backpressure";
}

TEST(RequestEngineTests, SubmitRejectsWhatTheStepPlaneDoesNotServe)
{
    // Speech output, the legacy system-prompt capture and trajectory execution carry response
    // state only the runtime's drain-time assembly fills. Same permanent-refusal semantics as
    // LoRA: fail fast at submit, with the alternative named, rather than a second control plane.
    FakeScript script;
    RequestEngine engine(fakeBatches(script), EngineConfig{});

    LLMGenerationRequest audio = makeRequest("speak");
    audio.generateAudio = true;
    EXPECT_THROW(engine.submit(std::move(audio)), std::invalid_argument);

    LLMGenerationRequest capture = makeRequest("warmup");
    capture.saveSystemPromptKVCache = true;
    EXPECT_THROW(engine.submit(std::move(capture)), std::invalid_argument);

    LLMGenerationRequest trajectory = makeRequest("act");
    trajectory.requests.front().pastTrajectory = std::vector<PastTrajectoryPoint>{{0.0F, 0.0F, 0.0F}};
    EXPECT_THROW(engine.submit(std::move(trajectory)), std::invalid_argument);

    EXPECT_NO_THROW(engine.submit(makeRequest("after")).get());
    EngineMetrics const m = engine.metrics();
    EXPECT_EQ(m.submitted, 1U);
    EXPECT_EQ(m.refused, 0U);
    EXPECT_EQ(script.batches.load(), 1) << "a rejected request never reaches a batch";
}

TEST(RequestEngineTests, SubmitFailsOnceTheInboxIsFull)
{
    FakeScript script;
    Gate gate;
    holdFirstDecode(script, gate);
    EngineConfig config;
    config.commandQueueCapacity = 2;
    RequestEngine engine(fakeBatches(script), config);

    // A bounded inbox is the backpressure signal; it refuses rather than growing or blocking.
    bool refused = false;
    std::vector<RequestHandle> handles;
    for (int32_t i = 0; i < 64 && !refused; ++i)
    {
        try
        {
            handles.push_back(engine.submit(makeRequest("p")));
        }
        catch (SubmitError const&)
        {
            refused = true;
        }
    }
    EXPECT_TRUE(refused) << "an unbounded inbox would swallow every request instead of pushing back";
    gate.open();
}

TEST(RequestEngineTests, CancelBeforeAdmissionTerminatesWithoutExecuting)
{
    FakeScript script;
    Gate gate;
    holdFirstDecode(script, gate);
    RequestEngine engine(fakeBatches(script), EngineConfig{});

    RequestHandle blocker = engine.submit(makeRequest("blocker"));
    // Wait for the blocker to occupy the actor, so the next submit is guaranteed to stay queued.
    ASSERT_TRUE(waitUntil([&] { return engine.resident() > 0; }));

    // Deliberately not waiting on queued(): the actor is inside the blocker's step and will not
    // drain its inbox until that returns, so this submission is not visible to the counters yet.
    RequestHandle queued = engine.submit(makeRequest("queued"));
    queued.cancel();
    // cancel() writes the stream flag on this thread, so the caller is released immediately rather
    // than waiting out the blocker.
    EXPECT_THROW(queued.get(), std::runtime_error);

    // An id nothing matches -- stale or fabricated -- is ignored rather than an error.
    engine.cancel(queued.id() + 1000);

    gate.open();
    blocker.get();
    engine.shutdown(ShutdownMode::kDrain);
    EXPECT_EQ(script.batches.load(), 1) << "a request cancelled while queued must never found a batch";
}

TEST(RequestEngineTests, CancelByIdIsLosslessWhenTheQueueIsFull)
{
    // cancel(id) used to ride the bounded command queue and was dropped, with a warning, when the
    // queue was full -- exactly the overload moment cancels matter most. It now plants the same
    // flag the handle path does, through the live-record registry, and nothing can drop it.
    FakeScript script;
    Gate gate;
    holdFirstDecode(script, gate);
    EngineConfig config;
    config.commandQueueCapacity = 2;
    RequestEngine engine(fakeBatches(script), config);

    RequestHandle blocker = engine.submit(makeRequest("blocker"));
    ASSERT_TRUE(waitUntil([&] { return engine.resident() > 0; }));
    std::vector<RequestHandle> fillers;
    while (true)
    {
        try
        {
            fillers.push_back(engine.submit(makeRequest("filler")));
        }
        catch (SubmitError const&)
        {
            break; // the queue is full: a command posted now would be dropped
        }
    }
    ASSERT_GE(fillers.size(), 1U);
    engine.cancel(fillers.front().id());

    gate.open();
    EXPECT_THROW(fillers.front().get(), std::runtime_error) << "the cancel was lost";
    for (size_t i = 1; i < fillers.size(); ++i)
    {
        EXPECT_NO_THROW(fillers[i].get());
    }
    EXPECT_NO_THROW(blocker.get());
    EXPECT_EQ(script.batches.load(), static_cast<int32_t>(fillers.size()))
        << "the cancelled filler must never have founded a batch";
    EXPECT_EQ(engine.metrics().cancelled, 1U);
}

TEST(RequestEngineTests, ACancelledHeadIsRetiredInsteadOfBeingAdmitted)
{
    // Cancelled while queued, the head must not pay a seated prefill (stalling every resident for
    // that tick) only to be evicted at the next poll; it retires before admission.
    FakeScript script;
    script.tokensPerRequest = 3;
    Gate gate;
    holdFirstDecode(script, gate);
    EngineConfig config;
    config.maxBatchSize = 2;
    RequestEngine engine(fakeBatches(script), config);

    RequestHandle founder = engine.submit(makeRequest("a"));
    ASSERT_TRUE(waitUntil([&] { return engine.resident() > 0; }));
    RequestHandle head = engine.submit(makeRequest("b"));
    head.cancel();
    gate.open();

    EXPECT_THROW(head.get(), std::runtime_error);
    EXPECT_NO_THROW(founder.get());
    EXPECT_EQ(script.admissions.load(), 0) << "a cancelled head was seated";
    EXPECT_EQ(script.batches.load(), 1);
    EXPECT_EQ(engine.metrics().cancelled, 1U);
}

TEST(RequestEngineTests, CancelOfAResidentRequestMarksItsStreamAndEndsItAtTheNextStep)
{
    FakeScript script;
    script.tokensPerRequest = 1000;
    Gate gate;
    holdFirstDecode(script, gate);
    RequestEngine engine(fakeBatches(script), EngineConfig{});

    RequestHandle handle = engine.submit(makeRequest("resident"));
    ASSERT_TRUE(waitUntil([&] { return engine.resident() > 0; }));

    handle.cancel();
    // The flag is written on the caller's thread: the caller learns immediately, and the runtime
    // sees it at the top of its next step -- here, once the gate releases that step.
    EXPECT_TRUE(handle.stream().isCancelled());
    gate.open();
    EXPECT_THROW(handle.get(), std::runtime_error);
    EXPECT_EQ(script.completed.load(), 0) << "a cancelled resident finishes as cancelled, not by length";
}

TEST(RequestEngineTests, ACallerSuppliedChannelIsTheOneTheHandleHandsOut)
{
    FakeScript script;
    RequestEngine engine(fakeBatches(script), EngineConfig{});
    auto supplied = StreamChannel::create();
    LLMGenerationRequest request = makeRequest("streamed");
    request.streamChannels = {supplied};
    RequestHandle handle = engine.submit(std::move(request));

    // One canonical channel: what the runtime writes and finishes is what stream() hands out, so a
    // consumer of either sees the same terminal state.
    EXPECT_EQ(handle.streamShared().get(), supplied.get());
    EXPECT_NO_THROW(handle.get());
}

TEST(RequestEngineTests, CancelWhileQueuedReachesACallerSuppliedChannel)
{
    FakeScript script;
    Gate gate;
    holdFirstDecode(script, gate);
    RequestEngine engine(fakeBatches(script), EngineConfig{});
    RequestHandle resident = engine.submit(makeRequest("resident"));
    ASSERT_TRUE(waitUntil([&] { return engine.resident() > 0; }));

    auto supplied = StreamChannel::create();
    LLMGenerationRequest queued = makeRequest("queued");
    queued.streamChannels = {supplied};
    RequestHandle handle = engine.submit(std::move(queued));
    handle.cancel();
    // The channel the caller is consuming must go terminal -- not a private one it never sees.
    EXPECT_TRUE(waitUntil([&] { return supplied->isCancelled(); }, 500ms));
    EXPECT_THROW(handle.get(), std::runtime_error);

    gate.open();
    EXPECT_NO_THROW(resident.get());
}

TEST(RequestEngineTests, AFailedStepBecomesAnErrorRatherThanAHang)
{
    FakeScript script;
    script.failDecode = true;
    RequestEngine engine(fakeBatches(script), EngineConfig{});
    RequestHandle handle = engine.submit(makeRequest("doomed"));
    EXPECT_THROW(handle.get(), std::runtime_error);
    // The channel reaches a terminal state too: a streaming consumer polls the channel rather
    // than the outcome flag, and a failed request whose channel stays silent parks that consumer
    // until its transport timeout (chaos-found on a too-long streaming founder).
    EXPECT_TRUE(handle.stream().isCancelled() || handle.stream().isFinished())
        << "a failed request left its channel non-terminal; streaming consumers would hang";
}

TEST(RequestEngineTests, AThrowingStepDoesNotKillTheActor)
{
    FakeScript script;
    script.throwOnDecodeOfBatch = 1;
    RequestEngine engine(fakeBatches(script), EngineConfig{});

    RequestHandle first = engine.submit(makeRequest("boom"));
    EXPECT_THROW(first.get(), std::runtime_error);

    // The actor must still be alive: an unwind through the tick loop would strand every later
    // caller with no terminal state at all.
    RequestHandle second = engine.submit(makeRequest("next"));
    LLMGenerationResponse response = second.get();
    ASSERT_EQ(response.outputTexts.size(), 1U);
    EXPECT_EQ(response.outputTexts[0], "next");
}

TEST(RequestEngineTests, ARefusedOrThrowingFoundingBecomesAnErrorAndTheActorSurvives)
{
    FakeScript script;
    script.refuseFounding = true;
    RequestEngine engine(fakeBatches(script), EngineConfig{});
    RequestHandle refused = engine.submit(makeRequest("refused"));
    EXPECT_THROW(refused.get(), std::runtime_error);

    script.refuseFounding = false;
    script.throwOnFounding = true;
    RequestHandle thrown = engine.submit(makeRequest("thrown"));
    EXPECT_THROW(thrown.get(), std::runtime_error);

    script.throwOnFounding = false;
    EXPECT_EQ(engine.submit(makeRequest("fine")).get().outputTexts.front(), "fine");
    EngineMetrics const m = engine.metrics();
    EXPECT_EQ(m.failed, 2U);
    EXPECT_EQ(m.completed, 1U);
}

TEST(RequestEngineTests, ShutdownDrainLetsQueuedRequestsFinish)
{
    FakeScript script;
    RequestEngine engine(fakeBatches(script), EngineConfig{});
    std::vector<RequestHandle> handles;
    for (int32_t i = 0; i < 4; ++i)
    {
        handles.push_back(engine.submit(makeRequest("p")));
    }
    engine.shutdown(ShutdownMode::kDrain);
    EXPECT_EQ(script.completed.load(), 4) << "kDrain owes every admitted request its result";
    for (auto& handle : handles)
    {
        EXPECT_NO_THROW(handle.get());
    }
}

TEST(RequestEngineTests, ShutdownCancelWakesEveryParkedCaller)
{
    FakeScript script;
    Gate gate;
    holdFirstDecode(script, gate);
    RequestEngine engine(fakeBatches(script), EngineConfig{});

    RequestHandle resident = engine.submit(makeRequest("resident"));
    ASSERT_TRUE(waitUntil([&] { return engine.resident() > 0; }));
    RequestHandle queued = engine.submit(makeRequest("queued"));

    // A caller parked in get() must not be left waiting forever because the engine went away.
    std::atomic<bool> queuedReturned{false};
    std::thread waiter([&queued, &queuedReturned] {
        try
        {
            queued.get();
        }
        catch (std::exception const&)
        {
        }
        queuedReturned.store(true);
    });

    gate.open();
    engine.shutdown(ShutdownMode::kCancel);
    waiter.join();
    EXPECT_TRUE(queuedReturned.load());
    EXPECT_TRUE(queued.ready() || queued.stream().isCancelled());
}

TEST(RequestEngineTests, SubmitAfterShutdownIsRefused)
{
    FakeScript script;
    RequestEngine engine(fakeBatches(script), EngineConfig{});
    engine.shutdown(ShutdownMode::kDrain);
    EXPECT_THROW(engine.submit(makeRequest("late")), SubmitError);
    EXPECT_FALSE(engine.running());
}

TEST(RequestEngineTests, DestructorShutsDownWithoutAnExplicitCall)
{
    // Nothing to assert beyond not hanging and not terminating: a joinable thread left in the
    // destructor would abort the process, which is exactly what the implied shutdown prevents.
    FakeScript script;
    {
        RequestEngine engine(fakeBatches(script), EngineConfig{});
        engine.submit(makeRequest("p"));
    }
    SUCCEED();
}

TEST(RequestEngineTests, ManyThreadsMaySubmitConcurrently)
{
    constexpr int32_t kThreads = 8;
    constexpr int32_t kPerThread = 25;
    FakeScript script;
    EngineConfig config;
    config.commandQueueCapacity = 256;
    RequestEngine engine(fakeBatches(script), config);

    std::atomic<int32_t> accepted{0};
    std::vector<std::thread> producers;
    for (int32_t t = 0; t < kThreads; ++t)
    {
        producers.emplace_back([&engine, &accepted] {
            for (int32_t i = 0; i < kPerThread;)
            {
                try
                {
                    RequestHandle handle = engine.submit(makeRequest("p"));
                    handle.get();
                    accepted.fetch_add(1, std::memory_order_relaxed);
                    ++i;
                }
                catch (SubmitError const&)
                {
                    std::this_thread::yield();
                }
            }
        });
    }
    for (auto& producer : producers)
    {
        producer.join();
    }
    EXPECT_EQ(accepted.load(), kThreads * kPerThread);
    EXPECT_EQ(script.completed.load(), kThreads * kPerThread);
    EXPECT_EQ(engine.queued(), 0);
    EXPECT_EQ(engine.resident(), 0);
}

// ---------------------------------------------------------------------------
// Regression tests for defects found by review. Each one fails against the
// implementation as it stood before the corresponding fix, and each bounds its
// own waiting so a regression reports as a failure rather than a hung suite.
// ---------------------------------------------------------------------------

TEST(RequestEngineTests, ClosingAdmissionIsAtomicWithRespectToSubmission)
{
    // The property the engine depends on, stated directly rather than left to a race to expose:
    // once closeAdmission() returns, no postSubmit() can still succeed. That is what makes the
    // actor's final sweep of the queue final, and therefore what stops an accepted request from
    // being abandoned with its caller parked on a stream nobody will finish.
    //
    // Checking a flag and then pushing cannot give this guarantee no matter how the two are
    // ordered, because a submit descheduled between them publishes after the sweep.
    EngineChannel channel(8);
    auto submit = [] {
        auto command = std::make_unique<EngineCommand>();
        command->type = CommandType::kSubmit;
        return command;
    };
    EXPECT_TRUE(channel.postSubmit(submit()));
    channel.closeAdmission();
    EXPECT_FALSE(channel.accepting());
    EXPECT_FALSE(channel.postSubmit(submit())) << "a submit slipped in after admission closed";

    // Stopping work is still deliverable: a cancel or shutdown arriving after the gate closed is
    // exactly the case that must get through.
    auto stop = std::make_unique<EngineCommand>();
    stop->type = CommandType::kShutdown;
    EXPECT_TRUE(channel.post(std::move(stop)));
}

TEST(RequestEngineTests, SubmitRacingShutdownNeverStrandsOrCancelsAnAcceptedRequest)
{
    // Stress cover for the property above and for kDrain's contract: a submit accepted before
    // shutdown() closed the gate must run to completion, not merely reach a terminal state. The
    // race window is a few instructions wide, so this guards against gross regressions;
    // ClosingAdmissionIsAtomicWithRespectToSubmission pins the invariant itself.
    for (int32_t round = 0; round < 200; ++round)
    {
        FakeScript script;
        auto engine = std::make_unique<RequestEngine>(fakeBatches(script), EngineConfig{});
        std::vector<RequestHandle> accepted;
        std::mutex mutex;

        std::thread submitter([&] {
            for (int32_t i = 0; i < 8; ++i)
            {
                try
                {
                    RequestHandle handle = engine->submit(makeRequest("p"));
                    std::lock_guard<std::mutex> lock(mutex);
                    accepted.push_back(std::move(handle));
                }
                catch (SubmitError const&)
                {
                }
            }
        });
        engine->shutdown(ShutdownMode::kDrain);
        submitter.join();

        std::lock_guard<std::mutex> lock(mutex);
        for (auto& handle : accepted)
        {
            ASSERT_TRUE(waitUntil([&] { return handle.ready(); }))
                << "round " << round << ": accepted request never terminated";
            EXPECT_NO_THROW(handle.get()) << "round " << round << ": kDrain cancelled an accepted request";
        }
        engine.reset();
    }
}

TEST(RequestEngineTests, HandleOutlivingItsEngineIsStillSafeToUse)
{
    FakeScript script;
    RequestHandle handle;
    {
        RequestEngine engine(fakeBatches(script), EngineConfig{});
        handle = engine.submit(makeRequest("p"));
    }
    // The handle shares the command channel rather than pointing at the engine, so these are
    // defined operations rather than a use-after-free. Whether the actor got to the request before
    // the destructor cancelled the backlog is a race the test does not fix; either way the request
    // has a terminal outcome and get() returns or throws without touching the dead engine.
    EXPECT_NE(handle.id(), kInvalidRequestId);
    EXPECT_NO_THROW(handle.cancel());
    EXPECT_TRUE(handle.stream().isFinished() || handle.stream().isCancelled());
    EXPECT_TRUE(handle.ready());
    try
    {
        LLMGenerationResponse const response = handle.get();
        EXPECT_EQ(response.outputTexts.front(), "p"); // the actor ran it before the engine died
    }
    catch (std::runtime_error const&)
    {
        // The destructor cancelled it first.
    }
}

TEST(RequestEngineTests, CancelWinsOverAStepThatFinishedTheSequenceAnyway)
{
    // The runtime polls the cancel flag at the top of a step; a cancel landing after that poll,
    // in the same step the sequence finishes naturally, reaches the engine as a kLength result.
    // Reporting completion would make the outcome depend on whether the caller read before or
    // after that step -- the caller had already asked to stop.
    FakeScript script;
    script.honorCancel = false; // the step never sees the flag; the engine must
    Gate gate;
    holdFirstDecode(script, gate);
    RequestEngine engine(fakeBatches(script), EngineConfig{});

    RequestHandle handle = engine.submit(makeRequest("p"));
    ASSERT_TRUE(waitUntil([&] { return engine.resident() > 0; }));
    handle.cancel();
    gate.open();
    engine.shutdown(ShutdownMode::kDrain);

    EXPECT_TRUE(handle.stream().isCancelled());
    EXPECT_THROW(handle.get(), std::runtime_error);
    EXPECT_EQ(engine.metrics().cancelled, 1U);
    EXPECT_EQ(engine.metrics().completed, 0U);
}

TEST(RequestEngineTests, OutcomeIsSeparateFromTheTokenStream)
{
    // The channel reports that the token stream ended; the handle reports that the request has an
    // outcome. They are set by different parties, and a request that never reaches the runtime has
    // a channel nobody finishes -- so the handle, not the channel, is what get() waits on.
    FakeScript script;
    RequestEngine engine(fakeBatches(script), EngineConfig{});
    RequestHandle handle = engine.submit(makeRequest("p"));
    ASSERT_NO_THROW(handle.get());
    EXPECT_TRUE(handle.ready());
}

TEST(RequestEngineTests, ShutdownCancelIsNotDowngradedWhenTheQueueIsFull)
{
    FakeScript script;
    Gate gate;
    holdFirstDecode(script, gate);
    EngineConfig config;
    config.commandQueueCapacity = 2;
    RequestEngine engine(fakeBatches(script), config);

    RequestHandle blocker = engine.submit(makeRequest("blocker"));
    ASSERT_TRUE(waitUntil([&] { return engine.resident() > 0; }));
    // Fill the queue so the shutdown command itself cannot be posted.
    while (true)
    {
        try
        {
            engine.submit(makeRequest("filler"));
        }
        catch (SubmitError const&)
        {
            break;
        }
    }

    gate.open();
    engine.shutdown(ShutdownMode::kCancel);
    // With the mode carried only by the command, a full queue left it at its kDrain default and the
    // whole backlog ran anyway.
    EXPECT_EQ(script.batches.load(), 1) << "kCancel degraded into a drain when the queue was full";
}

TEST(RequestEngineTests, ShutdownCancelReachesTheRunningBatchEvenWhenTheQueueIsFull)
{
    // The cancel of the running residents is driven by the stop flag at every tick, not by the
    // kShutdown command a full queue drops. Before that, shutdown(kCancel) waited out the whole
    // generation of the founder and its joiners and reported them completed.
    FakeScript script;
    script.tokensPerRequest = 1000000; // never finishes on its own within the test
    script.beforeDecode = [] { std::this_thread::sleep_for(1ms); };
    EngineConfig config;
    config.maxBatchSize = 2;
    config.commandQueueCapacity = 2;
    RequestEngine engine(fakeBatches(script), config);

    RequestHandle founder = engine.submit(makeRequest("founder"));
    ASSERT_TRUE(waitUntil([&] { return engine.resident() > 0; }));
    RequestHandle joiner = engine.submit(makeRequest("joiner"));
    ASSERT_TRUE(waitUntil([&] { return script.admissions.load() == 1; }))
        << "queued=" << engine.queued() << " resident=" << engine.resident() << " batches=" << script.batches.load()
        << " decodes=" << script.decodes.load() << " incompatible=" << engine.metrics().stallsIncompatible
        << " guided=" << engine.metrics().stallsGuided << " founderOnly=" << engine.metrics().stallsFounderOnly
        << " noCapacity=" << engine.metrics().stallsNoCapacity;
    while (true)
    {
        try
        {
            engine.submit(makeRequest("filler"));
        }
        catch (SubmitError const&)
        {
            break;
        }
    }

    auto const start = std::chrono::steady_clock::now();
    engine.shutdown(ShutdownMode::kCancel);
    EXPECT_LT(std::chrono::steady_clock::now() - start, 5s) << "shutdown waited out the running batch";
    EXPECT_THROW(founder.get(), std::runtime_error);
    EXPECT_THROW(joiner.get(), std::runtime_error);
    EXPECT_EQ(engine.metrics().completed, 0U) << "cancelled residents must not read back as completed";
}

TEST(RequestEngineTests, BacklogIsBoundedNotJustTheQueue)
{
    FakeScript script;
    Gate gate;
    holdFirstDecode(script, gate);
    EngineConfig config;
    config.commandQueueCapacity = 8;
    config.maxPendingRequests = 4;
    RequestEngine engine(fakeBatches(script), config);

    // A bounded command queue alone does not bound accepted work: the actor moves a queue's worth
    // into its own backlog at every tick. The pending limit is what actually pushes back.
    int32_t accepted = 0;
    for (int32_t i = 0; i < 64; ++i)
    {
        try
        {
            engine.submit(makeRequest("p"));
            ++accepted;
        }
        catch (SubmitError const&)
        {
            break;
        }
    }
    EXPECT_EQ(accepted, config.maxPendingRequests);
    gate.open();
    engine.shutdown(ShutdownMode::kDrain);
}

TEST(RequestEngineTests, RejectsAConfigurationThatWouldSpinTheActor)
{
    FakeScript script;
    EngineConfig config;
    config.idleWait = 0ms;
    EXPECT_THROW(RequestEngine(fakeBatches(script), config), std::runtime_error);

    config = EngineConfig{};
    config.maxPendingRequests = 0;
    EXPECT_THROW(RequestEngine(fakeBatches(script), config), std::runtime_error);

    config = EngineConfig{};
    config.maxBatchSize = 0;
    EXPECT_THROW(RequestEngine(fakeBatches(script), config), std::runtime_error);
}

TEST(RequestEngineTests, ShutdownJoinsOnceUnderContentionAndIsIdempotentAfter)
{
    FakeScript script;
    RequestEngine engine(fakeBatches(script), EngineConfig{});
    engine.submit(makeRequest("p"));

    std::vector<std::thread> stoppers;
    for (int32_t i = 0; i < 4; ++i)
    {
        stoppers.emplace_back(
            [&engine, i] { engine.shutdown(i % 2 == 0 ? ShutdownMode::kDrain : ShutdownMode::kCancel); });
    }
    for (auto& stopper : stoppers)
    {
        stopper.join();
    }
    EXPECT_FALSE(engine.running());

    // Idempotent afterwards too: late serial calls, whatever their mode, are no-ops.
    EXPECT_NO_THROW(engine.shutdown(ShutdownMode::kDrain));
    EXPECT_NO_THROW(engine.shutdown(ShutdownMode::kCancel));
}

TEST(RequestEngineTests, CompatibilityHandleRequestKeepsTheOldBlockingContract)
{
    // Success looks exactly like the old call: bool result, response filled, and the request taken
    // by const reference so callers written against the old signature can reuse their object.
    FakeScript script;
    RequestEngine engine(fakeBatches(script), EngineConfig{});
    LLMGenerationRequest reused = makeRequest("hello");
    LLMGenerationResponse response;
    EXPECT_TRUE(engine.handleRequest(reused, response));
    EXPECT_TRUE(engine.handleRequest(reused, response));
    ASSERT_EQ(response.outputTexts.size(), 1U);
    EXPECT_EQ(response.outputTexts[0], "hello");

    // Failure is a bool, never an exception, and must not scribble on the caller's response.
    FakeScript failing;
    failing.failDecode = true;
    RequestEngine failingEngine(fakeBatches(failing), EngineConfig{});
    LLMGenerationResponse untouched;
    untouched.outputTexts = {"untouched"};
    EXPECT_FALSE(failingEngine.handleRequest(makeRequest("doomed"), untouched));
    ASSERT_EQ(untouched.outputTexts.size(), 1U);
    EXPECT_EQ(untouched.outputTexts[0], "untouched");

    // After shutdown submit() throws; the wrapper converts that into false rather than letting an
    // exception escape to a caller that has never seen one from this call.
    engine.shutdown(ShutdownMode::kDrain);
    EXPECT_FALSE(engine.handleRequest(makeRequest("late"), response));
}

// ---------------------------------------------------------------------------
// In-flight admission at the engine level, driven through the fake batch. What
// is under test is the actor's tick work -- drain, admit, commit, publish, FCFS
// blocking -- not the runtime's generation, which has its own test.
// ---------------------------------------------------------------------------

TEST(RequestEngineTests, AdmitsAQueuedRequestMidFlightAndPublishesItsResult)
{
    FakeScript script;
    script.tokensPerRequest = 3;
    Gate gate;
    holdFirstDecode(script, gate); // hold the founder until the joiner is queued
    EngineConfig config;
    config.maxBatchSize = 2;
    RequestEngine engine(fakeBatches(script), config);

    RequestHandle founder = engine.submit(makeRequest("a"));
    ASSERT_TRUE(waitUntil([&] { return engine.resident() > 0; }));
    RequestHandle joiner = engine.submit(makeRequest("b"));
    gate.open();

    // The joiner's outcome comes from the commit, materialized by the batch.
    LLMGenerationResponse response = joiner.get();
    ASSERT_EQ(response.outputIds.size(), 1U);
    EXPECT_EQ(response.outputIds[0], (std::vector<int32_t>{1, 2, 3}));
    EXPECT_EQ(response.outputTexts[0], "b");
    EXPECT_EQ(founder.get().outputTexts[0], "a");

    EXPECT_EQ(script.batches.load(), 1) << "the joiner shared the founder's batch";
    EXPECT_EQ(script.admissions.load(), 1);
    EXPECT_EQ(engine.metrics().admittedMidFlight, 1U);
    EXPECT_EQ(script.lastAdmittedRequestId.load(), joiner.id());
}

TEST(RequestEngineTests, AnEvictedFounderIsPublishedBeforeTheBatchDrains)
{
    // The founder finishes while its joiner keeps the batch alive; its caller must get the outcome
    // then, not when the batch finally drains.
    FakeScript script;
    script.tokensPerRequest = 3;
    // The joiner outlives the founder whatever tick it is seated on; the test must not depend on
    // where the admission lands relative to the founder's decodes.
    script.joinerExtraTokens = 2;
    Gate gate;
    holdFirstDecode(script, gate);
    EngineConfig config;
    config.maxBatchSize = 2;
    RequestEngine engine(fakeBatches(script), config);

    RequestHandle founder = engine.submit(makeRequest("founder"));
    ASSERT_TRUE(waitUntil([&] { return engine.resident() > 0; }));
    RequestHandle joiner = engine.submit(makeRequest("joiner")); // queued while the actor is parked
    // Once the founder has retired, hold the batch at its next decode: the joiner then provably
    // still has work left at the moment the founder's caller is released. A timed sleep here made
    // the check below a race that a slow, busy machine lost.
    Gate afterFounder;
    script.beforeDecode = [&] {
        if (script.completed.load() >= 1)
        {
            afterFounder.wait();
        }
    };
    gate.open();

    EXPECT_NO_THROW(founder.get());
    EXPECT_FALSE(joiner.ready()) << "the founder's outcome must not wait for the batch to drain";
    afterFounder.open();
    EXPECT_NO_THROW(joiner.get());
    EXPECT_EQ(script.batches.load(), 1);
    EXPECT_EQ(engine.metrics().completed, 2U) << "one outcome each, published exactly once";
}

TEST(RequestEngineTests, AHeadTheBatchCannotTakeWaitsForItsOwnBatchInsteadOfJoining)
{
    // Two ways a head can be unjoinable -- batch-wide sampling mismatch and guided decoding (the
    // GuidedDecoder's matchers are slot-numbered and untracked by admission) -- one required
    // behaviour: wait, never join, then found the next batch. Rejecting would kill a request that
    // is perfectly able to run alone; joining would corrupt the resident batch.
    auto const unjoinableWaits = [](LLMGenerationRequest founderRequest, LLMGenerationRequest headRequest) {
        FakeScript script;
        script.tokensPerRequest = 20;
        script.beforeDecode = [] { std::this_thread::sleep_for(1ms); };
        EngineConfig config;
        config.maxBatchSize = 2;
        RequestEngine engine(fakeBatches(script), config);

        RequestHandle first = engine.submit(std::move(founderRequest));
        ASSERT_TRUE(waitUntil([&] { return engine.resident() > 0; }));
        RequestHandle second = engine.submit(std::move(headRequest));

        EXPECT_NO_THROW(first.get());
        EXPECT_NO_THROW(second.get());
        EXPECT_EQ(script.admissions.load(), 0) << "an unjoinable request was admitted into a shared batch";
        EXPECT_EQ(script.batches.load(), 2) << "the unjoinable head must run as its own batch afterwards";
    };

    LLMGenerationRequest incompatible = makeRequest("b");
    incompatible.temperature = 0.11F; // sampling is batch-wide; a mismatch must not share a step
    unjoinableWaits(makeRequest("a"), std::move(incompatible));

    LLMGenerationRequest guided = makeRequest("b");
    GuidedDecodingParams params;
    params.type = GuideType::kJsonObject;
    guided.requests.front().guidedDecoding = params;
    unjoinableWaits(makeRequest("a"), std::move(guided));
}

TEST(RequestEngineTests, ANoCapacityRefusalLeavesTheHeadQueuedForALaterTick)
{
    FakeScript script;
    script.tokensPerRequest = 40;
    script.refusalsLeft = 3; // transient pressure clears after a few ticks
    script.beforeDecode = [] { std::this_thread::sleep_for(1ms); };
    EngineConfig config;
    config.maxBatchSize = 2;
    RequestEngine engine(fakeBatches(script), config);

    RequestHandle founder = engine.submit(makeRequest("a"));
    ASSERT_TRUE(waitUntil([&] { return engine.resident() > 0; }));
    RequestHandle joiner = engine.submit(makeRequest("b"));

    EXPECT_NO_THROW(founder.get());
    EXPECT_NO_THROW(joiner.get()) << "a transient refusal must not kill the queued request";
    EXPECT_EQ(script.admissions.load(), 1) << "the head joined once the pressure cleared";
    EXPECT_EQ(engine.metrics().stallsNoCapacity, 3U) << "the head waited out the refused ticks";
}

TEST(RequestEngineTests, ARejectedAdmissionFailsTheHeadAndTheBatchGoesOn)
{
    FakeScript script;
    script.tokensPerRequest = 20;
    script.rejectAdmissions = true;
    script.beforeDecode = [] { std::this_thread::sleep_for(1ms); };
    EngineConfig config;
    config.maxBatchSize = 2;
    RequestEngine engine(fakeBatches(script), config);

    RequestHandle founder = engine.submit(makeRequest("a"));
    ASSERT_TRUE(waitUntil([&] { return engine.resident() > 0; }));
    RequestHandle rejected = engine.submit(makeRequest("b"));

    EXPECT_THROW(rejected.get(), std::runtime_error) << "a rejection is an explicit error, not a hang";
    EXPECT_NO_THROW(founder.get());
    EXPECT_EQ(script.batches.load(), 1) << "a rejected head does not found its own batch";
}

TEST(RequestEngineTests, ADrainedBatchTakesNoAdmissionAfterItsLastEviction)
{
    // The founder finishes on the tick that also sees a queued head. Admitting into the emptied
    // batch would kill the request -- there is no live founder to inherit sizing from; the head
    // becomes the next founder instead. Chaos-found on the earlier plane as a steady 0.75%
    // request-kill rate at exactly this boundary.
    FakeScript script;
    Gate gate;
    holdFirstDecode(script, gate); // the founder's one and only decode waits here
    EngineConfig config;
    config.maxBatchSize = 2;
    RequestEngine engine(fakeBatches(script), config);

    RequestHandle first = engine.submit(makeRequest("a"));
    ASSERT_TRUE(waitUntil([&] { return engine.resident() > 0; }));
    RequestHandle second = engine.submit(makeRequest("b"));
    gate.open();

    EXPECT_NO_THROW(first.get());
    EXPECT_NO_THROW(second.get()) << "the queued head must found the next batch, not die in the drained one";
    EXPECT_EQ(script.admissions.load(), 0) << "an emptied batch accepted an admission";
    EXPECT_EQ(script.batches.load(), 2);
}

TEST(RequestEngineTests, RetiredAdmissionsReleaseTheirBacklogReservation)
{
    // Every accepted request reserves a pending-count slot at submit(); an admission that skipped
    // the release would wedge the engine after maxPendingRequests cumulative admissions --
    // refusing all new work while sitting idle.
    FakeScript script;
    script.tokensPerRequest = 4;
    script.beforeDecode = [] { std::this_thread::sleep_for(1ms); };
    EngineConfig config;
    config.maxBatchSize = 2;
    config.maxPendingRequests = 3;
    RequestEngine engine(fakeBatches(script), config);

    // Twice the backlog limit in cumulative (founder, joiner) pairs, drained pair by pair.
    for (int32_t round = 0; round < config.maxPendingRequests * 2; ++round)
    {
        RequestHandle founder = engine.submit(makeRequest("a"));
        ASSERT_TRUE(waitUntil([&] { return engine.resident() > 0; }));
        RequestHandle joiner = engine.submit(makeRequest("b"));
        EXPECT_NO_THROW(founder.get()) << "round " << round;
        EXPECT_NO_THROW(joiner.get()) << "round " << round;
    }
}

TEST(RequestEngineTests, MetricsAccountForEveryRequestTheEngineSaw)
{
    FakeScript script;
    script.tokensPerRequest = 40;
    script.refusalsLeft = 2; // two countable no-capacity stalls before the admission lands
    script.beforeDecode = [] { std::this_thread::sleep_for(1ms); };
    EngineConfig config;
    config.maxBatchSize = 2;
    config.maxPendingRequests = 2; // the third concurrent submission is a countable refusal
    RequestEngine engine(fakeBatches(script), config);

    RequestHandle founder = engine.submit(makeRequest("a"));
    ASSERT_TRUE(waitUntil([&] { return engine.resident() > 0; }));
    RequestHandle joiner = engine.submit(makeRequest("b"));
    EXPECT_THROW(engine.submit(makeRequest("c")), SubmitError); // backlog is full
    EXPECT_NO_THROW(founder.get());
    EXPECT_NO_THROW(joiner.get());

    EngineMetrics const m = engine.metrics();
    EXPECT_EQ(m.submitted, 2U);
    EXPECT_EQ(m.refused, 1U);
    EXPECT_EQ(m.stallsNoCapacity, 2U);
    EXPECT_EQ(m.admittedMidFlight, 1U);
    EXPECT_EQ(m.completed, 2U);
    EXPECT_EQ(m.cancelled, 0U);
    EXPECT_EQ(m.failed, 0U);
    EXPECT_EQ(m.queueLatencyCount, 2U) << "one founding, one admission";
    EXPECT_GE(m.queueLatencyTotalUs, m.queueLatencyMaxUs);
    EXPECT_GT(m.queueLatencyMaxUs, 0U) << "the joiner waited out two refused ticks";
}

TEST(RequestEngineTests, ResidentCountFollowsAdmissionsAndEvictions)
{
    FakeScript script;
    script.tokensPerRequest = 200;
    script.beforeDecode = [] { std::this_thread::sleep_for(1ms); };
    EngineConfig config;
    config.maxBatchSize = 2;
    RequestEngine engine(fakeBatches(script), config);

    RequestHandle founder = engine.submit(makeRequest("a"));
    ASSERT_TRUE(waitUntil([&] { return engine.resident() == 1; }));
    RequestHandle joiner = engine.submit(makeRequest("b"));
    EXPECT_TRUE(waitUntil([&] { return engine.resident() == 2; })) << "the admission must show in resident()";
    EXPECT_NO_THROW(founder.get());
    EXPECT_NO_THROW(joiner.get());
    EXPECT_TRUE(waitUntil([&] { return engine.resident() == 0; }));
}

TEST(RequestEngineTests, AnAdmittedRequestAbandonedByTheBatchStillGetsAnOutcome)
{
    FakeScript script;
    script.tokensPerRequest = 3;
    script.abandonJoiners = true; // admitted -- and then the batch loses it without a result
    Gate gate;
    holdFirstDecode(script, gate);
    EngineConfig config;
    config.maxBatchSize = 2;
    RequestEngine engine(fakeBatches(script), config);

    RequestHandle founder = engine.submit(makeRequest("a"));
    ASSERT_TRUE(waitUntil([&] { return engine.resident() > 0; }));
    RequestHandle abandoned = engine.submit(makeRequest("b"));
    gate.open();

    EXPECT_NO_THROW(founder.get());
    // Not a hang, not silence: an explicit error reaches the caller.
    EXPECT_THROW(abandoned.get(), std::runtime_error);
    EXPECT_EQ(engine.metrics().failed, 1U);
}

TEST(RequestEngineTests, ShutdownCancelReachesAJoinerMidFlight)
{
    // The joiner is in the batch ledger, not in any second list; a cancelling shutdown must reach
    // it through the same path as the founder and report both as cancelled.
    FakeScript script;
    script.tokensPerRequest = 1000000;
    script.beforeDecode = [] { std::this_thread::sleep_for(1ms); };
    EngineConfig config;
    config.maxBatchSize = 2;
    RequestEngine engine(fakeBatches(script), config);

    RequestHandle founder = engine.submit(makeRequest("a"));
    ASSERT_TRUE(waitUntil([&] { return engine.resident() > 0; }));
    RequestHandle joiner = engine.submit(makeRequest("b"));
    ASSERT_TRUE(waitUntil([&] { return script.admissions.load() == 1; }));

    engine.shutdown(ShutdownMode::kCancel);
    EXPECT_THROW(founder.get(), std::runtime_error);
    EXPECT_THROW(joiner.get(), std::runtime_error);
    EXPECT_EQ(engine.metrics().cancelled, 2U);
    EXPECT_EQ(engine.metrics().completed, 0U);
}
