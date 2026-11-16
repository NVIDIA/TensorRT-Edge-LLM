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

// The request engine's actor over a scripted session: admission, scheduling, cancellation and
// shutdown, without a device. What is under test is the actor's tick work -- drain, admit, step,
// commit, publish, FCFS blocking, exclusivity -- not the runtime's generation, which has its own
// tests over a real session.

#include "runtime/llmRuntimeUtils.h"
#include "runtime/runtimeSession.h"
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
#include <optional>
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

bool isGuided(LLMGenerationRequest const& request)
{
    return !request.requests.empty() && request.requests.front().guidedDecoding.has_value();
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

//! What the fake session does. Written by the test before it submits, read on the actor thread;
//! the counters are the test's window into what the actor did.
struct FakeScript
{
    //! Tokens each resident produces before it finishes, one per decode step.
    int32_t tokensPerRequest{1};
    //! Extra tokens a sequence admitted into a running session produces on top of
    //! tokensPerRequest, so a test can make it outlive the residents it joined whatever step it
    //! was seated on.
    int32_t joinerExtraTokens{0};
    //! Runs at the top of every decode step on the actor thread, told how many sequences are
    //! resident: a test parks the actor here, at the step whose state it wants to observe.
    std::function<void(int32_t residents)> beforeDecode;
    //! Refuse this many admissions with kNoCapacity before seating any.
    int32_t refusalsLeft{0};
    //! Reject every admission outright.
    bool rejectAdmissions{false};
    //! The next admission throws SessionLostError: the reservation failed inside the runtime and
    //! the session's state is unknown. One-shot.
    bool loseSessionOnAdmit{false};
    //! Every decode step reports ok = false.
    bool failDecode{false};
    //! The first decode of this session (1-based, counting sessions opened) throws; 0 never throws.
    int32_t throwOnDecodeOfSession{0};
    //! Poll the request's cancel token and the channel's flag at the top of each step, as the
    //! runtime does.
    bool honorCancel{true};
    //! Drop finished sequences that were admitted into a running session without reporting them:
    //! the session loses them.
    bool abandonJoiners{false};
    //! The factory refuses (null) or throws when asked to open this session (1-based); 0 never.
    int32_t refuseOpenOfSession{0};
    int32_t throwOnOpenOfSession{0};
    //! capabilities() throws std::runtime_error.
    bool throwOnCapabilities{false};
    //! Every request runs on a decoder that takes one set of sampling parameters per batch.
    bool uniformSampling{false};
    //! Report context-cache counters, with one committed publication per request completed so far.
    bool reportsContextCache{false};
    //! How long reading those counters takes once a request has completed.
    std::chrono::milliseconds contextCacheReadDelay{0};

    std::atomic<int32_t> sessions{0};
    std::atomic<int32_t> closed{0};
    std::atomic<int32_t> admissions{0};
    std::atomic<int32_t> admissionsIntoRunning{0};
    std::atomic<int32_t> contextSteps{0};
    std::atomic<RequestId> lastAdmittedRequestId{kInvalidRequestId};
    std::atomic<int32_t> completed{0};
    std::atomic<int32_t> decodes{0};
};

//! A stand-in for the runtime session: seats whatever the engine admits, emits one token per
//! resident per decode step, finishes a resident when its budget is spent, and materializes the
//! prompt back as the output text so a test can tell results apart. A grammar-constrained request
//! is exclusive, as it is on the real session.
class FakeSession final : public RuntimeSession
{
public:
    FakeSession(FakeScript& script, int32_t index)
        : mScript(script)
        , mIndex(index)
    {
    }

    ~FakeSession() override
    {
        mScript.closed.fetch_add(1);
    }

    AdmissionResult admit(LLMGenerationRequest const& request, RequestId requestId,
        std::shared_ptr<std::atomic<bool> const> cancelToken) override
    {
        if (mScript.refusalsLeft > 0)
        {
            --mScript.refusalsLeft;
            return {AdmissionResult::Status::kNoCapacity, {}, {}, 1, 0};
        }
        if (mScript.rejectAdmissions)
        {
            return {AdmissionResult::Status::kRejected, {}, "the fake session rejects this request", 1, 0};
        }
        if (mScript.loseSessionOnAdmit)
        {
            mScript.loseSessionOnAdmit = false;
            throw SessionLostError("the lease path lost the device");
        }
        if (mPending.has_value())
        {
            throw std::runtime_error("admit: the previous admission has not run its context step yet");
        }
        bool const intoRunning = !mResidents.empty();
        Resident resident;
        resident.ref = ResidentRef{freeSlot(), ++mEpoch};
        resident.requestId = requestId;
        resident.channel = request.streamChannels.empty() ? nullptr : request.streamChannels.front();
        resident.cancelToken = std::move(cancelToken);
        resident.joiner = intoRunning;
        resident.exclusive = isGuided(request);
        mPrompts[resident.ref.slot] = promptOf(request);
        mResidents.push_back(std::move(resident));
        mPending = mResidents.back().ref;
        mScript.admissions.fetch_add(1);
        if (intoRunning)
        {
            mScript.admissionsIntoRunning.fetch_add(1);
        }
        mScript.lastAdmittedRequestId.store(requestId);
        return {AdmissionResult::Status::kAdmitted, mResidents.back().ref, {}, 1, 0};
    }

    StepResult execute(std::vector<ScheduledSequenceDescriptor> const& plan) override
    {
        StepResult result;
        if (plan.empty())
        {
            throw std::runtime_error("execute: empty plan");
        }
        result.work = plan.front().work;
        for (auto const& sequence : plan)
        {
            result.participants.push_back(sequence.identity);
        }
        if (result.work == SequenceWork::kContext)
        {
            if (!mPending.has_value() || plan.size() != 1 || plan.front().identity.resident != *mPending)
            {
                throw std::runtime_error("execute: the context step does not name the pending admission");
            }
            mPending.reset();
            mScript.contextSteps.fetch_add(1);
            result.ok = true;
            result.stepId = ++mStepId;
            result.execution = plan; // ran exactly as planned
            return result;
        }

        mScript.decodes.fetch_add(1);
        if (mScript.beforeDecode)
        {
            mScript.beforeDecode(static_cast<int32_t>(mResidents.size()));
        }
        if (mScript.throwOnDecodeOfSession == mIndex)
        {
            mScript.throwOnDecodeOfSession = 0;
            throw std::runtime_error("device lost mid-decode");
        }
        if (mScript.failDecode)
        {
            return result; // ok stays false and nothing advanced: a failed step produced nothing
        }
        result.ok = true;
        result.stepId = ++mStepId;
        result.execution = plan; // one token per resident, from its committed length
        for (auto it = mResidents.begin(); it != mResidents.end();)
        {
            bool const cancelled = mScript.honorCancel
                && ((it->channel != nullptr && it->channel->isCancelled())
                    || (it->cancelToken != nullptr && it->cancelToken->load()));
            if (cancelled)
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
        // The fake tags each snapshot with its slot (see snapshot()) so the prompt comes back.
        auto const prompt = mPrompts.find(result.rawBatchedInputIds.empty() ? -1 : result.rawBatchedInputIds.front());
        response.outputTexts.push_back(prompt == mPrompts.end() ? "materialized" : prompt->second);
        response.finishReasons.push_back(result.terminalReason);
        return response;
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

    bool idle() const noexcept override
    {
        return mResidents.empty();
    }

    bool exclusiveResident() const noexcept override
    {
        for (auto const& resident : mResidents)
        {
            if (resident.exclusive)
            {
                return true;
            }
        }
        return false;
    }

    bool requiresExclusiveAdmission(LLMGenerationRequest const& request) const noexcept override
    {
        return isGuided(request);
    }

    bool requiresUniformSampling(LLMGenerationRequest const&) const noexcept override
    {
        return mScript.uniformSampling;
    }

    std::optional<ContextCacheMetrics> contextCacheMetrics() const noexcept override
    {
        if (!mScript.reportsContextCache)
        {
            return std::nullopt;
        }
        auto const completed = static_cast<uint64_t>(mScript.completed.load());
        if (completed > 0)
        {
            std::this_thread::sleep_for(mScript.contextCacheReadDelay);
        }
        ContextCacheMetrics metrics;
        metrics.committedPublications = completed;
        return metrics;
    }

    //! No compact execution, over an engine built for B = 8, S_in = 1024.
    SessionCapabilities capabilities() const override
    {
        if (mScript.throwOnCapabilities)
        {
            throw std::runtime_error("the session's capabilities are unavailable");
        }
        return {false, "the fake session has no compact execution", 8, 1024};
    }

private:
    struct Resident
    {
        ResidentRef ref;
        RequestId requestId{};
        std::shared_ptr<StreamChannel> channel;
        std::shared_ptr<std::atomic<bool> const> cancelToken;
        std::vector<int32_t> tokens;
        int32_t generated{0};
        bool joiner{false};
        bool exclusive{false};
    };

    //! The lowest slot no resident holds, as a pool would hand out.
    int32_t freeSlot() const
    {
        for (int32_t slot = 0;; ++slot)
        {
            bool taken = false;
            for (auto const& resident : mResidents)
            {
                taken = taken || resident.ref.slot == slot;
            }
            if (!taken)
            {
                return slot;
            }
        }
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
    int32_t const mIndex;
    uint64_t mEpoch{0};
    StepId mStepId{0};
    std::vector<Resident> mResidents;
    std::optional<ResidentRef> mPending;
    std::unordered_map<int32_t, std::string> mPrompts;
};

//! The engine's session seam bound to a script.
RequestEngine::SessionFactory fakeSessions(FakeScript& script)
{
    return [&script](cudaStream_t) -> std::unique_ptr<RuntimeSession> {
        int32_t const index = script.sessions.fetch_add(1) + 1;
        if (script.throwOnOpenOfSession == index)
        {
            throw std::runtime_error("openSession exploded");
        }
        if (script.refuseOpenOfSession == index)
        {
            return nullptr;
        }
        return std::make_unique<FakeSession>(script, index);
    };
}

//! Parks the actor at its first decode step until the gate opens; later steps run through. A test
//! submits more requests while the actor is parked, so they land while the first is resident, for
//! certain -- an explicit hold rather than a decode slowed down enough to make it likely.
void holdFirstDecode(FakeScript& script, Gate& gate)
{
    auto held = std::make_shared<std::atomic<bool>>(false);
    script.beforeDecode = [&gate, held](int32_t) {
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
    RequestEngine engine(fakeSessions(script), EngineConfig{});

    auto const start = std::chrono::steady_clock::now();
    RequestHandle handle = engine.submit(makeRequest("a"));
    auto const elapsed = std::chrono::steady_clock::now() - start;
    // The whole point of the actor: submit posts and returns, it does not wait for the GPU.
    EXPECT_LT(elapsed, 200ms);
    EXPECT_NE(handle.id(), kInvalidRequestId);
    gate.open();
    EXPECT_NO_THROW(handle.get());

    // And the other half of the round trip: get() hands back what the session materialized.
    FakeScript echo;
    RequestEngine echoEngine(fakeSessions(echo), EngineConfig{});
    LLMGenerationResponse response = echoEngine.submit(makeRequest("hello")).get();
    ASSERT_EQ(response.outputTexts.size(), 1U);
    EXPECT_EQ(response.outputTexts[0], "hello");
}

TEST(RequestEngineTests, RequestIdsAreUniqueAndNeverZero)
{
    FakeScript script;
    RequestEngine engine(fakeSessions(script), EngineConfig{});
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
    RequestEngine engine(fakeSessions(script), EngineConfig{}); // maxBatchSize 1: one resident at a time

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
    // Every head here is admitted at the boundary where the previous resident was just evicted.
    // When batches were opened around their first request this was the boundary that killed
    // requests (a steady 0.75% kill rate, chaos-found); now an empty session is the ordinary case.
    EXPECT_EQ(script.sessions.load(), 1) << "one session serves every request; nothing is reopened between them";
    EXPECT_EQ(script.admissions.load(), 5);
    EXPECT_EQ(script.admissionsIntoRunning.load(), 0) << "maxBatchSize is 1, so every admission finds an empty session";
    EXPECT_EQ(script.contextSteps.load(), 5) << "every sequence's first token comes out of its own context step";
    EXPECT_EQ(script.closed.load(), 0) << "an idle session is not closed";
    engine.shutdown(ShutdownMode::kDrain);
    EXPECT_EQ(script.closed.load(), 1) << "shutdown closes the one session, once";
}

TEST(RequestEngineTests, SubmitRejectsWhatIsNotServed)
{
    // Per-request LoRA, speech output and trajectory execution carry state only the runtime's
    // blocking-path assembly fills. The refusal is permanent, so it happens at submit with the
    // alternative named -- deliberately not a SubmitError, which means transient backpressure and
    // invites a retry -- rather than through a second control plane.
    FakeScript script;
    RequestEngine engine(fakeSessions(script), EngineConfig{});

    LLMGenerationRequest withLora = makeRequest("hello");
    withLora.loraWeightsName = "adapter-a";
    EXPECT_THROW(engine.submit(std::move(withLora)), std::invalid_argument);

    LLMGenerationRequest audio = makeRequest("speak");
    audio.generateAudio = true;
    EXPECT_THROW(engine.submit(std::move(audio)), std::invalid_argument);

    LLMGenerationRequest trajectory = makeRequest("act");
    trajectory.requests.front().pastTrajectory = std::vector<PastTrajectoryPoint>{{0.0F, 0.0F, 0.0F}};
    EXPECT_THROW(engine.submit(std::move(trajectory)), std::invalid_argument);

    // The refusals reserve nothing and the engine keeps serving.
    EXPECT_NO_THROW(engine.submit(makeRequest("after")).get());
    EngineMetrics const m = engine.metrics();
    EXPECT_EQ(m.submitted, 1U);
    EXPECT_EQ(m.refused, 0U) << "an unsupported request is not backpressure";
    EXPECT_EQ(script.admissions.load(), 1) << "a rejected request never reaches the session";
}

TEST(RequestEngineTests, SubmitFailsOnceTheInboxIsFull)
{
    FakeScript script;
    Gate gate;
    holdFirstDecode(script, gate);
    EngineConfig config;
    config.commandQueueCapacity = 2;
    RequestEngine engine(fakeSessions(script), config);

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
    RequestEngine engine(fakeSessions(script), EngineConfig{});

    RequestHandle blocker = engine.submit(makeRequest("blocker"));
    // Wait for the blocker to occupy the actor, so the next submit is guaranteed to stay queued.
    ASSERT_TRUE(waitUntil([&] { return engine.resident() > 0; }));

    // Deliberately not waiting on queued(): the actor is inside the blocker's step and will not
    // drain its inbox until that returns, so this submission is not visible to the counters yet.
    RequestHandle queued = engine.submit(makeRequest("queued"));
    queued.cancel();
    // cancel() sets the token on this thread, so the caller is released immediately rather than
    // waiting out the blocker.
    EXPECT_THROW(queued.get(), std::runtime_error);

    // An id nothing matches -- stale or fabricated -- is ignored rather than an error.
    engine.cancel(queued.id() + 1000);

    // A consumer may cancel the channel it holds without going through the handle; the actor
    // treats that like a cancel too, so the request neither runs nor parks its caller.
    RequestHandle viaChannel = engine.submit(makeRequest("via channel"));
    viaChannel.stream().cancel();

    gate.open();
    EXPECT_THROW(viaChannel.get(), std::runtime_error);
    blocker.get();
    engine.shutdown(ShutdownMode::kDrain);
    EXPECT_EQ(script.admissions.load(), 1) << "a request cancelled while queued must never be admitted";
}

TEST(RequestEngineTests, CancelByIdIsLosslessWhenTheQueueIsFull)
{
    // cancel(id) used to ride the bounded command queue and was dropped, with a warning, when the
    // queue was full -- exactly the overload moment cancels matter most. It sets the same token the
    // handle path does, through the live-record registry, and nothing can drop it.
    FakeScript script;
    Gate gate;
    holdFirstDecode(script, gate);
    EngineConfig config;
    config.commandQueueCapacity = 2;
    RequestEngine engine(fakeSessions(script), config);

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
    ASSERT_TRUE(waitUntil([&] { return fillers.front().ready(); }));
    EXPECT_EQ(script.admissions.load(), static_cast<int32_t>(fillers.size()))
        << "the cancelled filler must never have been admitted";
    EXPECT_EQ(engine.metrics().cancelled, 1U);
}

TEST(RequestEngineTests, ACancelledHeadIsRetiredInsteadOfBeingAdmitted)
{
    // Cancelled while queued, the head must not pay a context step (stalling every resident for
    // that step) only to be evicted at the next poll; it retires before admission.
    FakeScript script;
    script.tokensPerRequest = 3;
    Gate gate;
    holdFirstDecode(script, gate);
    EngineConfig config;
    config.maxBatchSize = 2;
    RequestEngine engine(fakeSessions(script), config);

    RequestHandle first = engine.submit(makeRequest("a"));
    ASSERT_TRUE(waitUntil([&] { return engine.resident() > 0; }));
    RequestHandle head = engine.submit(makeRequest("b"));
    head.cancel();
    gate.open();

    EXPECT_THROW(head.get(), std::runtime_error);
    EXPECT_NO_THROW(first.get());
    ASSERT_TRUE(waitUntil([&] { return head.ready(); }));
    EXPECT_EQ(script.admissions.load(), 1) << "a cancelled head was seated";
    EXPECT_EQ(engine.metrics().cancelled, 1U);
}

TEST(RequestEngineTests, CancelEndsResidentAtNextStep)
{
    FakeScript script;
    script.tokensPerRequest = 1000;
    Gate gate;
    holdFirstDecode(script, gate);
    RequestEngine engine(fakeSessions(script), EngineConfig{});

    RequestHandle handle = engine.submit(makeRequest("resident"));
    ASSERT_TRUE(waitUntil([&] { return engine.resident() > 0; }));

    handle.cancel();
    // The token is written on the caller's thread: the caller learns immediately, and the session
    // sees it at the top of its next step -- here, once the gate releases that step. The channel
    // is cancelled too, so a consumer polling only the channel is not left parked.
    EXPECT_TRUE(handle.stream().isCancelled());
    gate.open();
    EXPECT_THROW(handle.get(), std::runtime_error);
    // get() released the caller on the token, before the actor published the outcome; wait for
    // that publication before reading what the actor recorded.
    ASSERT_TRUE(waitUntil([&] { return handle.ready(); }));
    EXPECT_EQ(script.completed.load(), 0) << "a cancelled resident finishes as cancelled, not by length";
    EXPECT_EQ(engine.metrics().cancelled, 1U);
}

TEST(RequestEngineTests, ACallerSuppliedChannelIsTheOneTheHandleHandsOut)
{
    FakeScript script;
    RequestEngine engine(fakeSessions(script), EngineConfig{});
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
    RequestEngine engine(fakeSessions(script), EngineConfig{});
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
    // And once the actor retired it, the channel is finished with the reason as well.
    EXPECT_TRUE(waitUntil([&] { return supplied->isFinished(); }, 500ms));
    EXPECT_EQ(supplied->getReason(), FinishReason::kCancelled);
}

TEST(RequestEngineTests, SessionFailureFailsResidentsAndReopens)
{
    // A step that reports failure, a step that throws and a reservation that fails inside the
    // runtime at admission end the same way: the session's state is unknown, so every resident --
    // and, at admission, the head -- fails with the reason, each channel is finished, the session
    // is replaced, and the next request runs in the fresh one. An unwind through the tick loop
    // would instead strand every later caller with no terminal state at all; treating the lost
    // admission as a refusal and scheduling on would drive a possibly poisoned session (an
    // executor whose forward threw is not driven further in TensorRT-LLM or vLLM either).
    enum class Cause
    {
        kStepReportsFailure,
        kStepThrows,
        kLostAtAdmission
    };
    for (Cause const cause : {Cause::kStepReportsFailure, Cause::kStepThrows, Cause::kLostAtAdmission})
    {
        SCOPED_TRACE(cause == Cause::kStepReportsFailure ? "the step reports failure"
                : cause == Cause::kStepThrows            ? "the step throws"
                                                         : "the session is lost at admission");
        FakeScript script;
        script.tokensPerRequest = 20;
        script.failDecode = cause == Cause::kStepReportsFailure;
        script.throwOnDecodeOfSession = cause == Cause::kStepThrows ? 1 : 0;
        Gate gate;
        holdFirstDecode(script, gate);
        EngineConfig config;
        config.maxBatchSize = 2;
        RequestEngine engine(fakeSessions(script), config);

        RequestHandle resident = engine.submit(makeRequest("a"));
        ASSERT_TRUE(waitUntil([&] { return engine.resident() > 0; }));
        std::optional<RequestHandle> head;
        if (cause == Cause::kLostAtAdmission)
        {
            script.loseSessionOnAdmit = true;
            head.emplace(engine.submit(makeRequest("b")));
        }
        gate.open();

        auto const failsNaming = [](RequestHandle& handle, char const* who, char const* needle) {
            try
            {
                handle.get();
                FAIL() << who << " read back as completed";
            }
            catch (std::runtime_error const& error)
            {
                // The failure reaches the caller by name, not as a generic "failed during generation".
                EXPECT_NE(std::string(error.what()).find(needle), std::string::npos) << who << ": " << error.what();
            }
            // The channel reaches a terminal state too, with the reason: a streaming consumer polls
            // the channel rather than the outcome flag, and a failed request whose channel stays
            // silent parks that consumer until its transport timeout (chaos-found on a too-long
            // streaming request).
            EXPECT_TRUE(handle.stream().isFinished()) << who << " left its channel non-terminal";
            EXPECT_EQ(handle.stream().getReason(), FinishReason::kError) << who;
        };
        failsNaming(resident, "the resident", "the runtime session failed");
        if (head.has_value())
        {
            failsNaming(*head, "the head", "lost the device");
        }
        EXPECT_EQ(engine.metrics().failed, head.has_value() ? 2U : 1U);

        script.failDecode = false;
        EXPECT_EQ(engine.submit(makeRequest("next")).get().outputTexts.front(), "next");
        EXPECT_EQ(script.sessions.load(), 2) << "the session that failed was replaced";
        EXPECT_EQ(script.closed.load(), 1);
    }
}

TEST(RequestEngineTests, FailedOpenIsConstructionError)
{
    // A runtime that cannot open its session cannot serve anything: that is the engine's
    // construction failing, with the actor joined, not the first request's error.
    FakeScript refusing;
    refusing.refuseOpenOfSession = 1;
    EXPECT_THROW(RequestEngine(fakeSessions(refusing), EngineConfig{}), std::runtime_error);

    FakeScript throwing;
    throwing.throwOnOpenOfSession = 1;
    EXPECT_THROW(RequestEngine(fakeSessions(throwing), EngineConfig{}), std::runtime_error);
}

TEST(RequestEngineTests, AStartupFailureOutsideTheSchedulingConfigurationIsNotAConfigurationError)
{
    // Given a runtime whose session opens but throws std::runtime_error when asked for its capabilities
    // When a RequestEngine is constructed over it
    // Then construction fails with a std::runtime_error that carries the session's error, not with the
    //      std::invalid_argument that reports a rejected scheduling configuration
    FakeScript script;
    script.throwOnCapabilities = true;
    try
    {
        RequestEngine engine(fakeSessions(script), EngineConfig{});
        FAIL() << "the engine was constructed";
    }
    catch (std::invalid_argument const& error)
    {
        FAIL() << "reported as a scheduling configuration error: " << error.what();
    }
    catch (std::runtime_error const& error)
    {
        EXPECT_NE(std::string(error.what()).find("the session's capabilities are unavailable"), std::string::npos)
            << error.what();
    }
}

TEST(RequestEngineTests, NoSessionFailsEveryRequestWithReason)
{
    // Two ways the engine ends up without a session: the runtime refuses to open a fresh one after
    // a failure, or failures repeat with no step completed between them (opening succeeds even
    // when the cause is sticky, so without a bound every later request would pay one failed step
    // and a reopen). Either way the engine keeps answering -- every request gets an explicit error
    // naming the reason -- rather than a busy signal or a caller parked forever.
    auto const failsUnavailable = [](RequestEngine& engine, char const* needle) {
        try
        {
            engine.submit(makeRequest("after")).get();
            FAIL() << "a request offered to an engine without a session must fail";
        }
        catch (std::runtime_error const& error)
        {
            EXPECT_NE(std::string(error.what()).find("session is unavailable"), std::string::npos) << error.what();
            EXPECT_NE(std::string(error.what()).find(needle), std::string::npos) << error.what();
        }
        EXPECT_TRUE(engine.running());
    };
    {
        SCOPED_TRACE("the runtime refuses to reopen");
        FakeScript script;
        script.throwOnDecodeOfSession = 1;
        script.refuseOpenOfSession = 2;
        RequestEngine engine(fakeSessions(script), EngineConfig{});
        EXPECT_THROW(engine.submit(makeRequest("boom")).get(), std::runtime_error);
        failsUnavailable(engine, "refused to open");
        EXPECT_EQ(engine.metrics().failed, 2U);
    }
    {
        SCOPED_TRACE("the session keeps failing");
        FakeScript script;
        EngineConfig config;
        config.maxConsecutiveSessionFailures = 2;
        RequestEngine engine(fakeSessions(script), config);

        // A failed request is published before its session is replaced, so the session count is
        // read only after the actor has served a later request (it is single-threaded).
        script.loseSessionOnAdmit = true;
        EXPECT_THROW(engine.submit(makeRequest("first")).get(), std::runtime_error);
        EXPECT_EQ(engine.submit(makeRequest("fine")).get().outputTexts.front(), "fine");
        EXPECT_EQ(script.sessions.load(), 2) << "one failure: reopened";

        script.loseSessionOnAdmit = true;
        EXPECT_THROW(engine.submit(makeRequest("second")).get(), std::runtime_error);
        script.loseSessionOnAdmit = true;
        EXPECT_THROW(engine.submit(makeRequest("third")).get(), std::runtime_error);
        EXPECT_EQ(script.sessions.load(), 3) << "the completed step reset the count: the second failure reopened";
        failsUnavailable(engine, "in a row");
        EXPECT_EQ(script.sessions.load(), 3) << "two in a row: not reopened";
    }
}

TEST(RequestEngineTests, ShutdownDrainLetsQueuedRequestsFinish)
{
    FakeScript script;
    RequestEngine engine(fakeSessions(script), EngineConfig{});
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
    EXPECT_EQ(script.closed.load(), 1);
}

TEST(RequestEngineTests, ShutdownCancelWakesEveryParkedCaller)
{
    FakeScript script;
    Gate gate;
    holdFirstDecode(script, gate);
    RequestEngine engine(fakeSessions(script), EngineConfig{});

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
    EXPECT_TRUE(queued.ready());
}

TEST(RequestEngineTests, SubmitAfterShutdownIsRefused)
{
    FakeScript script;
    RequestEngine engine(fakeSessions(script), EngineConfig{});
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
        RequestEngine engine(fakeSessions(script), EngineConfig{});
        engine.submit(makeRequest("p"));
    }
    EXPECT_EQ(script.closed.load(), 1);
}

TEST(RequestEngineTests, ManyThreadsMaySubmitConcurrently)
{
    constexpr int32_t kThreads = 8;
    static constexpr int32_t kPerThread = 25;
    FakeScript script;
    EngineConfig config;
    config.commandQueueCapacity = 256;
    RequestEngine engine(fakeSessions(script), config);

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
        auto engine = std::make_unique<RequestEngine>(fakeSessions(script), EngineConfig{});
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
        RequestEngine engine(fakeSessions(script), EngineConfig{});
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
    // The runtime polls the cancel token at the top of a step; a cancel landing after that poll,
    // in the same step the sequence finishes naturally, reaches the engine as a kLength result.
    // Reporting completion would make the outcome depend on whether the caller read before or
    // after that step -- the caller had already asked to stop.
    FakeScript script;
    script.honorCancel = false; // the step never sees the token; the engine must
    Gate gate;
    holdFirstDecode(script, gate);
    RequestEngine engine(fakeSessions(script), EngineConfig{});

    RequestHandle handle = engine.submit(makeRequest("p"));
    ASSERT_TRUE(waitUntil([&] { return engine.resident() > 0; }));
    handle.cancel();
    gate.open();
    engine.shutdown(ShutdownMode::kDrain);

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
    RequestEngine engine(fakeSessions(script), EngineConfig{});
    RequestHandle handle = engine.submit(makeRequest("p"));
    ASSERT_NO_THROW(handle.get());
    EXPECT_TRUE(handle.ready());
}

TEST(RequestEngineTests, ShutdownCancelReachesResidentsWithFullQueue)
{
    // The cancel of the residents is driven by the stop flag at every boundary, not by the
    // kShutdown command a full queue drops: shutdown(kCancel) must not wait out the residents'
    // whole generation, must not report them completed, and must not degrade into a drain that
    // admits the backlog (with the mode carried only by the command, a full queue left it at its
    // kDrain default and the whole backlog ran anyway).
    FakeScript script;
    script.tokensPerRequest = 1000000; // never finishes on its own within the test
    script.beforeDecode = [](int32_t) { std::this_thread::sleep_for(1ms); };
    EngineConfig config;
    config.maxBatchSize = 2;
    config.commandQueueCapacity = 2;
    RequestEngine engine(fakeSessions(script), config);

    RequestHandle first = engine.submit(makeRequest("first"));
    ASSERT_TRUE(waitUntil([&] { return engine.resident() > 0; }));
    RequestHandle second = engine.submit(makeRequest("second"));
    ASSERT_TRUE(waitUntil([&] { return script.admissionsIntoRunning.load() == 1; }))
        << "queued=" << engine.queued() << " resident=" << engine.resident() << " decodes=" << script.decodes.load()
        << " incompatible=" << engine.metrics().stallsIncompatible << " guided=" << engine.metrics().stallsGuided
        << " exclusive=" << engine.metrics().stallsExclusive << " noCapacity=" << engine.metrics().stallsNoCapacity;
    uint64_t fillers = 0;
    while (true)
    {
        try
        {
            engine.submit(makeRequest("filler"));
            ++fillers;
        }
        catch (SubmitError const&)
        {
            break;
        }
    }

    auto const start = std::chrono::steady_clock::now();
    engine.shutdown(ShutdownMode::kCancel);
    EXPECT_LT(std::chrono::steady_clock::now() - start, 5s) << "shutdown waited out the residents";
    EXPECT_THROW(first.get(), std::runtime_error);
    EXPECT_THROW(second.get(), std::runtime_error);
    EXPECT_EQ(engine.metrics().cancelled, 2U + fillers) << "both residents and every queued filler are cancelled";
    EXPECT_EQ(engine.metrics().completed, 0U) << "cancelled residents must not read back as completed";
    EXPECT_EQ(script.admissions.load(), 2) << "kCancel degraded into a drain when the queue was full";
    EXPECT_EQ(script.closed.load(), 1) << "the session is closed with its residents cancelled";
}

TEST(RequestEngineTests, BacklogIsBoundedNotJustTheQueue)
{
    FakeScript script;
    Gate gate;
    holdFirstDecode(script, gate);
    EngineConfig config;
    config.commandQueueCapacity = 8;
    config.maxPendingRequests = 4;
    RequestEngine engine(fakeSessions(script), config);

    // A bounded command queue alone does not bound accepted work: the actor moves a queue's worth
    // into its own backlog at every boundary. The pending limit is what actually pushes back.
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
    EXPECT_THROW(RequestEngine(fakeSessions(script), config), std::runtime_error);

    config = EngineConfig{};
    config.maxPendingRequests = 0;
    EXPECT_THROW(RequestEngine(fakeSessions(script), config), std::runtime_error);

    config = EngineConfig{};
    config.maxBatchSize = 0;
    EXPECT_THROW(RequestEngine(fakeSessions(script), config), std::runtime_error);
}

TEST(RequestEngineTests, ShutdownJoinsOnceUnderContentionAndIsIdempotentAfter)
{
    FakeScript script;
    RequestEngine engine(fakeSessions(script), EngineConfig{});
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
    EXPECT_EQ(script.closed.load(), 1);
}

TEST(RequestEngineTests, CompatibilityHandleRequestKeepsTheOldBlockingContract)
{
    // Success looks exactly like the old call: bool result, response filled, and the request taken
    // by const reference so callers written against the old signature can reuse their object.
    FakeScript script;
    RequestEngine engine(fakeSessions(script), EngineConfig{});
    LLMGenerationRequest reused = makeRequest("hello");
    LLMGenerationResponse response;
    EXPECT_TRUE(engine.handleRequest(reused, response));
    EXPECT_TRUE(engine.handleRequest(reused, response));
    ASSERT_EQ(response.outputTexts.size(), 1U);
    EXPECT_EQ(response.outputTexts[0], "hello");

    // Failure is a bool, never an exception, and must not scribble on the caller's response.
    FakeScript failing;
    failing.failDecode = true;
    RequestEngine failingEngine(fakeSessions(failing), EngineConfig{});
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
// In-flight admission at the engine level, driven through the fake session.
// ---------------------------------------------------------------------------

TEST(RequestEngineTests, AdmitsMidFlightAndPublishesEachAsItRetires)
{
    // The second request joins the first's running session and the first finishes while the
    // second keeps the session busy: resident() follows the admission and the eviction, the
    // first's caller gets its outcome then, not when the session finally drains, and each outcome
    // is the one the session materialized for that request.
    FakeScript script;
    script.tokensPerRequest = 3;
    // The second outlives the first whatever step it is seated on; the test must not depend on
    // where the admission lands relative to the first's decodes.
    script.joinerExtraTokens = 2;
    // Three holds, installed once (the hook is never reassigned while the actor may be inside it):
    // at the first decode step, so the second request lands while the first is resident; at the
    // first decode step both are resident for, so the count of two is observable; and once the
    // first has retired, at the next decode, so the second provably still has work left at the
    // moment the first's caller is released.
    Gate firstDecode;
    Gate bothResident;
    Gate afterFirst;
    auto heldFirst = std::make_shared<std::atomic<bool>>(false);
    auto heldBoth = std::make_shared<std::atomic<bool>>(false);
    script.beforeDecode = [&](int32_t residents) {
        if (!heldFirst->exchange(true))
        {
            firstDecode.wait();
        }
        else if (residents >= 2 && !heldBoth->exchange(true))
        {
            bothResident.wait();
        }
        else if (script.completed.load() >= 1)
        {
            afterFirst.wait();
        }
    };
    EngineConfig config;
    config.maxBatchSize = 2;
    RequestEngine engine(fakeSessions(script), config);

    RequestHandle first = engine.submit(makeRequest("first"));
    ASSERT_TRUE(waitUntil([&] { return engine.resident() == 1; }));
    RequestHandle second = engine.submit(makeRequest("second")); // queued while the actor is parked
    firstDecode.open();
    EXPECT_TRUE(waitUntil([&] { return engine.resident() == 2; })) << "the admission must show in resident()";
    bothResident.open();

    EXPECT_EQ(first.get().outputTexts.front(), "first");
    EXPECT_FALSE(second.ready()) << "the first's outcome must not wait for the session to drain";
    EXPECT_TRUE(waitUntil([&] { return engine.resident() == 1; })) << "the eviction must show in resident()";
    afterFirst.open();
    // The second's outcome comes from the commit, materialized by the session.
    LLMGenerationResponse const response = second.get();
    ASSERT_EQ(response.outputIds.size(), 1U);
    EXPECT_EQ(response.outputIds.front(), (std::vector<int32_t>{1, 2, 3, 4, 5}));
    EXPECT_EQ(response.outputTexts.front(), "second");
    EXPECT_TRUE(waitUntil([&] { return engine.resident() == 0; }));
    EXPECT_EQ(script.sessions.load(), 1);
    EXPECT_EQ(script.admissionsIntoRunning.load(), 1) << "the second request joined the first's running session";
    EXPECT_EQ(script.lastAdmittedRequestId.load(), second.id());
    EXPECT_EQ(engine.metrics().admittedMidFlight, 1U);
    EXPECT_EQ(engine.metrics().completed, 2U) << "one outcome each, published exactly once";
}

TEST(RequestEngineTests, IncompatibleHeadWaitsForEmptySession)
{
    // Batch-wide parameters must agree with the residents', and a grammar-constrained request runs
    // alone. Either head waits, never joins, and is admitted into the same session once it has
    // emptied; the head behind it waits in turn (FCFS throughout). Rejecting would kill a request
    // that is perfectly able to run alone; joining would corrupt the residents.
    struct Case
    {
        char const* name;
        std::function<void(LLMGenerationRequest&)> shape;
        uint64_t EngineMetrics::* stalls;
        bool uniformSampling{false};
    };
    Case const cases[] = {
        {"incompatible thinking mode", [](LLMGenerationRequest& head) { head.enableThinking = true; },
            &EngineMetrics::stallsIncompatible},
        // An opt-out runs on the default decoder, never inside a speculative step.
        {"speculation opt-out", [](LLMGenerationRequest& head) { head.disableSpecDecode = true; },
            &EngineMetrics::stallsIncompatible},
        {"grammar-constrained",
            [](LLMGenerationRequest& head) {
                GuidedDecodingParams params;
                params.type = GuideType::kJsonObject;
                head.requests.front().guidedDecoding = params;
            },
            &EngineMetrics::stallsGuided},
        // A speculative decoder takes one set of sampling parameters per batch.
        {"sampling unlike a speculative batch's", [](LLMGenerationRequest& head) { head.temperature = 0.3F; },
            &EngineMetrics::stallsIncompatible, /*uniformSampling=*/true},
    };
    for (Case const& testCase : cases)
    {
        SCOPED_TRACE(testCase.name);
        FakeScript script;
        script.tokensPerRequest = 20;
        script.uniformSampling = testCase.uniformSampling;
        Gate gate;
        holdFirstDecode(script, gate); // the head lands while the first request is resident, for certain
        EngineConfig config;
        config.maxBatchSize = 2;
        RequestEngine engine(fakeSessions(script), config);

        RequestHandle first = engine.submit(makeRequest("a"));
        ASSERT_TRUE(waitUntil([&] { return engine.resident() > 0; }));
        LLMGenerationRequest head = makeRequest("b");
        testCase.shape(head);
        RequestHandle second = engine.submit(std::move(head));
        RequestHandle third = engine.submit(makeRequest("c"));
        gate.open();

        EXPECT_NO_THROW(first.get());
        EXPECT_NO_THROW(second.get());
        EXPECT_NO_THROW(third.get());
        EXPECT_EQ(script.admissionsIntoRunning.load(), 0) << "nothing shared the session with the head";
        EXPECT_EQ(script.admissions.load(), 3);
        EXPECT_EQ(script.sessions.load(), 1) << "the head is admitted into the same, emptied session";
        EXPECT_GT(engine.metrics().*testCase.stalls, 0U);
    }
}

TEST(RequestEngineTests, AdmissionRefusalIsTheHeadsAlone)
{
    // A transient refusal (kNoCapacity) keeps the head queued for a later step; a rejection fails
    // it with an explicit error. Neither touches the session or the residents.
    FakeScript script;
    script.tokensPerRequest = 40;
    Gate gate;
    holdFirstDecode(script, gate);
    EngineConfig config;
    config.maxBatchSize = 3;
    RequestEngine engine(fakeSessions(script), config);

    RequestHandle first = engine.submit(makeRequest("a"));
    ASSERT_TRUE(waitUntil([&] { return engine.resident() > 0; }));
    script.refusalsLeft = 3; // transient pressure clears after a few steps
    RequestHandle second = engine.submit(makeRequest("b"));
    gate.open();
    ASSERT_TRUE(waitUntil([&] { return script.admissionsIntoRunning.load() == 1; }))
        << "the head joined once the pressure cleared";
    EXPECT_EQ(engine.metrics().stallsNoCapacity, 3U) << "the head waited out the refused steps";

    script.rejectAdmissions = true;
    RequestHandle rejected = engine.submit(makeRequest("c"));
    EXPECT_THROW(rejected.get(), std::runtime_error) << "a rejection is an explicit error, not a hang";
    EXPECT_NO_THROW(first.get());
    EXPECT_NO_THROW(second.get()) << "a transient refusal must not kill the queued request";
    EXPECT_EQ(script.sessions.load(), 1) << "neither outcome disturbs the session";
    EXPECT_EQ(engine.metrics().failed, 1U);
}

TEST(RequestEngineTests, RetiredAdmissionsReleaseTheirBacklogReservation)
{
    // Every accepted request reserves a pending-count slot at submit(); an admission that skipped
    // the release would wedge the engine after maxPendingRequests cumulative admissions --
    // refusing all new work while sitting idle.
    FakeScript script;
    script.tokensPerRequest = 4;
    script.beforeDecode = [](int32_t) { std::this_thread::sleep_for(1ms); };
    EngineConfig config;
    config.maxBatchSize = 2;
    config.maxPendingRequests = 3;
    RequestEngine engine(fakeSessions(script), config);

    // Twice the backlog limit in cumulative pairs, drained pair by pair.
    for (int32_t round = 0; round < config.maxPendingRequests * 2; ++round)
    {
        RequestHandle first = engine.submit(makeRequest("a"));
        ASSERT_TRUE(waitUntil([&] { return engine.resident() > 0; }));
        RequestHandle second = engine.submit(makeRequest("b"));
        EXPECT_NO_THROW(first.get()) << "round " << round;
        EXPECT_NO_THROW(second.get()) << "round " << round;
    }
}

TEST(RequestEngineTests, MetricsAccountForEveryRequestTheEngineSaw)
{
    FakeScript script;
    script.tokensPerRequest = 40;
    Gate gate;
    holdFirstDecode(script, gate);
    EngineConfig config;
    config.maxBatchSize = 2;
    config.maxPendingRequests = 2; // the third concurrent submission is a countable refusal
    RequestEngine engine(fakeSessions(script), config);

    RequestHandle first = engine.submit(makeRequest("a"));
    ASSERT_TRUE(waitUntil([&] { return engine.resident() > 0; }));
    script.refusalsLeft = 2; // two countable no-capacity stalls before the admission lands
    RequestHandle second = engine.submit(makeRequest("b"));
    EXPECT_THROW(engine.submit(makeRequest("c")), SubmitError); // backlog is full
    gate.open();
    EXPECT_NO_THROW(first.get());
    EXPECT_NO_THROW(second.get());

    EngineMetrics const m = engine.metrics();
    EXPECT_EQ(m.submitted, 2U);
    EXPECT_EQ(m.refused, 1U);
    EXPECT_EQ(m.stallsNoCapacity, 2U);
    EXPECT_EQ(m.admittedMidFlight, 1U);
    EXPECT_EQ(m.completed, 2U);
    EXPECT_EQ(m.cancelled, 0U);
    EXPECT_EQ(m.failed, 0U);
    EXPECT_EQ(m.queueLatencyCount, 2U) << "one latency per admission";
    EXPECT_GE(m.queueLatencyTotalUs, m.queueLatencyMaxUs);
    EXPECT_GT(m.queueLatencyMaxUs, 0U) << "the second waited out two refused steps";
}

// Given a session whose context-cache counters take a while to read once a request has finished
// When a request runs to completion and its caller reads the engine's counters as soon as the outcome arrives
// Then the counters already include the step that finished the request
TEST(RequestEngineTests, ContextCacheCountersIncludeTheStepThatReleasedTheCaller)
{
    FakeScript script;
    script.reportsContextCache = true;
    script.contextCacheReadDelay = 200ms;
    RequestEngine engine(fakeSessions(script), EngineConfig{});

    engine.submit(makeRequest("a")).get();
    auto const metrics = engine.contextCacheMetrics();

    ASSERT_TRUE(metrics.has_value());
    EXPECT_EQ(metrics->committedPublications, 1U)
        << "the caller was released before the counters saw the step that finished its request";
}

TEST(RequestEngineTests, AbandonedRequestGetsAnOutcome)
{
    FakeScript script;
    script.tokensPerRequest = 3;
    script.abandonJoiners = true; // admitted -- and then the session loses it without a result
    Gate gate;
    holdFirstDecode(script, gate);
    EngineConfig config;
    config.maxBatchSize = 2;
    RequestEngine engine(fakeSessions(script), config);

    RequestHandle first = engine.submit(makeRequest("a"));
    ASSERT_TRUE(waitUntil([&] { return engine.resident() > 0; }));
    RequestHandle abandoned = engine.submit(makeRequest("b"));
    gate.open();

    EXPECT_NO_THROW(first.get());
    // The session reports nothing for it, so after that step the engine's table holds a resident
    // the session does not: the two ledgers have diverged, which fails the session, and with it
    // the abandoned request. Not a hang, not silence: an explicit error reaches the caller.
    EXPECT_THROW(abandoned.get(), std::runtime_error);
    EXPECT_EQ(engine.metrics().failed, 1U);
    EXPECT_EQ(engine.submit(makeRequest("next")).get().outputTexts.front(), "next");
    EXPECT_EQ(script.sessions.load(), 2) << "the session that lost a resident was replaced";
}
