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

// The request engine over a scripted session with compact execution: admission at step boundaries, pause,
// cancellation, step metrics and request stats, and the fallback after a refused compact step. Step composition
// itself is pinned by the step planner tests.

#include "runtime/llmRuntimeUtils.h"
#include "runtime/runtimeSession.h"
#include "scheduler/requestEngine.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <functional>
#include <future>
#include <initializer_list>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

using namespace trt_edgellm::rt;
using namespace trt_edgellm::rt::scheduler;
using namespace std::chrono_literals;

namespace
{

//! A request named "g..." carries guided decoding.
LLMGenerationRequest makeRequest(std::string prompt, int32_t priority = 0)
{
    LLMGenerationRequest request;
    request.temperature = 0.7F;
    request.topP = 0.9F;
    request.topK = 40;
    request.maxGenerateLength = 128;
    request.priority = priority;
    request.requests.resize(1);
    if (prompt.front() == 'g')
    {
        request.requests[0].guidedDecoding = GuidedDecodingParams{};
    }
    Message message;
    message.contents.push_back(Message::MessageContent{"text", std::move(prompt)});
    request.requests[0].messages.push_back(std::move(message));
    return request;
}

//! Parks the actor at the top of one step until the test releases it.
struct Hold
{
    std::promise<void> reached;
    std::promise<void> release;
    std::shared_future<void> released{release.get_future().share()};

    void park()
    {
        reached.set_value();
        released.wait();
    }

    bool awaitReached()
    {
        return reached.get_future().wait_for(5s) == std::future_status::ready;
    }
};

//! What the compact session does, set before the first submit, and the steps it was asked to run.
struct CompactScript
{
    //! Prompt length per prompt text, else 8 tokens.
    std::map<std::string, int32_t> promptLengths;
    //! Tokens a resident produces before it finishes (final chunk, then one per decode row), else 4.
    std::map<std::string, int32_t> tokensFor;
    //! The actor parks at the top of the second step until released.
    Hold hold;
    //! Runs on the actor thread once a step's plan is logged, before cancellations apply.
    std::function<void(int32_t step, std::string const& line)> beforeStep;
    //! B = 4, S_in = 1024, P = 128: mixed defaults maxNumTokens 640, maxChunkTokens 512.
    SessionCapabilities capabilities{true, {}, 4, 1024, 128};
    //! "[...] " per step: "name:past+query" per prompt chunk ("!" when it samples), "name:d" per decode row.
    std::string steps;
    //! Steps executed, across every session the factory opened.
    int32_t stepCount{0};
    //! This step (counted across sessions; 0 for none) fails as a plugin refusing a compact forward does, and every
    //! later session reports no compact execution.
    int32_t refuseStep{0};
    //! Sessions the factory opens before it throws; 0 for no limit.
    int32_t openableSessions{0};
    int32_t sessions{0};
};

//! Runs any subset of residents per step and refuses a plan the real session would refuse: a pastLength other
//! than the committed one, a chunk outside the remaining prompt, sampling other than at the prompt's end, a decode
//! row before the prompt completed, and, without compact execution, anything but one whole prompt or a decode row
//! for every resident. A step that decodes every resident left after cancellations replays a CUDA graph. Only a
//! live, uncancelled resident whose prompt has not completed can be released for readmission.
class CompactFakeSession final : public RuntimeSession
{
public:
    explicit CompactFakeSession(CompactScript& script)
        : mScript(script)
        , mCompact(script.capabilities.compact)
    {
    }

    AdmissionResult admit(
        LLMGenerationRequest const& request, RequestId, std::shared_ptr<std::atomic<bool> const> cancelToken) override
    {
        Resident resident;
        resident.prompt = request.requests.at(0).messages.at(0).contents.at(0).content;
        ++mEpoch;
        resident.ref = ResidentRef{static_cast<int32_t>(mEpoch), mEpoch};
        resident.promptLength
            = mScript.promptLengths.count(resident.prompt) ? mScript.promptLengths[resident.prompt] : 8;
        resident.budget = mScript.tokensFor.count(resident.prompt) ? mScript.tokensFor[resident.prompt] : 4;
        resident.cancelToken = std::move(cancelToken);
        mResidents.push_back(resident);
        return AdmissionResult{AdmissionResult::Status::kAdmitted, resident.ref, {}, resident.promptLength, 0};
    }

    StepResult execute(std::vector<ScheduledSequenceDescriptor> const& plan) override
    {
        int32_t const step = ++mScript.stepCount;
        auto const isChunk = [](ScheduledSequenceDescriptor const& s) { return s.work == SequenceWork::kContext; };
        bool const anyChunk = std::any_of(plan.begin(), plan.end(), isChunk);
        refuseUnless(mCompact || (anyChunk ? plan.size() == 1 : plan.size() == mResidents.size()));
        std::string line;
        for (ScheduledSequenceDescriptor const& sequence : plan)
        {
            Resident* const resident = find(sequence.identity.resident);
            refuseUnless(resident != nullptr && sequence.pastLength == resident->committed);
            int32_t const remaining = resident->promptLength - resident->committed;
            bool const chunk = isChunk(sequence);
            refuseUnless(chunk ? !resident->promptDone && sequence.queryLength > 0 && sequence.queryLength <= remaining
                        && sequence.selectLastTokenLogits == (sequence.queryLength == remaining)
                               : resident->promptDone && sequence.queryLength == 1 && sequence.selectLastTokenLogits);
            line += (line.empty() ? "" : " ") + resident->prompt + ":"
                + (chunk ? std::to_string(sequence.pastLength) + "+" + std::to_string(sequence.queryLength)
                            + (sequence.selectLastTokenLogits ? "!" : "")
                         : std::string("d"));
        }
        mScript.steps += "[" + line + "] ";
        if (step == 2)
        {
            mScript.hold.park();
        }
        if (mScript.beforeStep)
        {
            mScript.beforeStep(step, line);
        }
        StepResult result;
        result.ok = true;
        result.stepId = static_cast<StepId>(step);
        for (ScheduledSequenceDescriptor const& sequence : plan)
        {
            result.participants.push_back(sequence.identity);
        }
        // Cancellations land at the top of the step, before the forward, as in the real session.
        for (Resident& resident : mResidents)
        {
            resident.cancelled = resident.cancelToken->load();
        }
        if (step == mScript.refuseStep)
        {
            // A cancelled resident stays, retired in place, and no result reports it.
            refuseUnless(mCompact);
            mScript.capabilities.compact = false;
            mScript.capabilities.reason = "a plugin refused compact execution at runtime";
            result.ok = false;
            result.compact = true;
            result.compactRefused = true;
            result.failure = "a plugin refused compact execution; the request can be retried";
            return result;
        }
        retireIf(result, FinishReason::kCancelled, [](Resident const& resident) { return resident.cancelled; });
        for (ScheduledSequenceDescriptor const& sequence : plan)
        {
            Resident* const resident = find(sequence.identity.resident);
            if (resident == nullptr)
            {
                continue; // cancelled above; it ran nothing
            }
            result.execution.push_back(sequence);
            resident->committed += sequence.queryLength;
            if (sequence.work == SequenceWork::kDecode || sequence.selectLastTokenLogits)
            {
                resident->promptDone = true;
                resident->tokens.push_back(static_cast<int32_t>(resident->tokens.size()) + 1);
            }
        }
        bool const decodesEveryResident = !result.execution.empty() && result.execution.size() == mResidents.size()
            && std::none_of(result.execution.begin(), result.execution.end(), isChunk);
        result.compact = !decodesEveryResident;
        result.graphLaunched = decodesEveryResident;
        retireIf(result, FinishReason::kLength,
            [](Resident const& resident) { return static_cast<int32_t>(resident.tokens.size()) >= resident.budget; });
        return result;
    }

    LLMGenerationResponse materialize(BatchResult const& result, std::vector<std::string> const&) const override
    {
        LLMGenerationResponse response;
        response.outputIds.push_back(result.tokenIds);
        response.finishReasons.push_back(result.terminalReason);
        return response;
    }

    std::vector<ResidentRef> residents() const override
    {
        std::vector<ResidentRef> refs;
        for (Resident const& resident : mResidents)
        {
            refs.push_back(resident.ref);
        }
        return refs;
    }

    void releaseForReadmission(ResidentRef ref) override
    {
        Resident const* const resident = find(ref);
        refuseUnless(resident != nullptr && !resident->promptDone && !resident->cancelled);
    }

    bool idle() const noexcept override
    {
        return mResidents.empty();
    }

    bool exclusiveResident() const noexcept override
    {
        return std::any_of(
            mResidents.begin(), mResidents.end(), [](Resident const& resident) { return resident.prompt[0] == 'g'; });
    }

    bool requiresExclusiveAdmission(LLMGenerationRequest const& request) const noexcept override
    {
        return request.requests.front().guidedDecoding.has_value();
    }

    SessionCapabilities capabilities() const override
    {
        return mScript.capabilities;
    }

private:
    struct Resident
    {
        ResidentRef ref;
        std::string prompt;
        int32_t promptLength{0};
        int32_t budget{0};
        int32_t committed{0};
        bool promptDone{false};
        bool cancelled{false};
        std::vector<int32_t> tokens;
        std::shared_ptr<std::atomic<bool> const> cancelToken;
    };

    static void refuseUnless(bool condition)
    {
        if (!condition)
        {
            throw std::runtime_error("execute: the plan breaks the compact execution contract");
        }
    }

    template <typename Predicate>
    void retireIf(StepResult& result, FinishReason reason, Predicate&& retire)
    {
        for (auto it = mResidents.begin(); it != mResidents.end();)
        {
            if (!retire(*it))
            {
                ++it;
                continue;
            }
            BatchResult finished;
            finished.tokenIds = it->tokens;
            finished.generateLength = static_cast<int32_t>(it->tokens.size());
            finished.terminalReason = reason;
            result.finished.emplace_back(it->ref, std::move(finished));
            it = mResidents.erase(it);
        }
    }

    Resident* find(ResidentRef ref)
    {
        auto const it = std::find_if(
            mResidents.begin(), mResidents.end(), [ref](Resident const& resident) { return resident.ref == ref; });
        return it == mResidents.end() ? nullptr : &*it;
    }

    CompactScript& mScript;
    //! Fixed at open, as a session's capabilities are.
    bool const mCompact;
    uint64_t mEpoch{0};
    std::vector<Resident> mResidents;
};

RequestEngine::SessionFactory compactSessions(CompactScript& script)
{
    return [&script](cudaStream_t) -> std::unique_ptr<RuntimeSession> {
        if (script.openableSessions > 0 && script.sessions >= script.openableSessions)
        {
            throw std::runtime_error("the runtime cannot open another session");
        }
        ++script.sessions;
        return std::make_unique<CompactFakeSession>(script);
    };
}

EngineConfig compactConfig(IfbSchedulingMode mode = IfbSchedulingMode::kMixed, int32_t maxBatchSize = 4,
    SchedulingPolicy policy = SchedulingPolicy::kFcfs)
{
    EngineConfig config;
    config.scheduling.ifbSchedulingMode = mode;
    config.maxBatchSize = maxBatchSize;
    config.scheduling.schedulingPolicy = policy;
    config.scheduling.priorityReservedSlots = policy == SchedulingPolicy::kPriority ? 1 : 0;
    return config;
}

//! The message get() fails with, or "completed".
std::string outcomeOf(RequestHandle& handle)
{
    try
    {
        handle.get();
        return "completed";
    }
    catch (std::exception const& error)
    {
        return error.what();
    }
}

} // namespace

TEST(RequestEngineCompactTests, AdmissionAtAStepBoundaryFollowsTheModeThePolicyAndTheSlots)
{
    // Given a compact session (B = 4, S_in = 1024) and a first request A, whose second step is held while the
    //       row's other requests queue
    // When every request has finished
    // Then the steps ran exactly as the row states: mixed admits every head that fits at the next boundary beside
    //      A's decode row, prefillFirst without it; fcfs admits in arrival order whatever the priorities; a guided
    //      request runs alone; with one reserved slot of three, the priority -1 request is admitted first and
    //      priority 0 requests never take the last free slot
    struct Case
    {
        char const* what;
        EngineConfig config;
        //! The first request (A) and the requests queued during its second step, with their priorities.
        std::vector<std::pair<std::string, int32_t>> requests;
        std::string steps;
    };
    std::vector<Case> const cases{
        {"mixed", compactConfig(), {{"a", 0}, {"b", 0}, {"c", 0}, {"d", 0}},
            "[a:0+8!] [a:d] [b:0+8! c:0+8! d:0+8! a:d] [a:d b:d c:d d:d] [b:d c:d d:d] [b:d c:d d:d] "},
        {"prefillFirst", compactConfig(IfbSchedulingMode::kPrefillFirst), {{"a", 0}, {"b", 0}, {"c", 0}, {"d", 0}},
            "[a:0+8!] [a:d] [b:0+8! c:0+8! d:0+8!] [a:d b:d c:d d:d] [a:d b:d c:d d:d] [b:d c:d d:d] "},
        {"fcfs ignores priorities", compactConfig(IfbSchedulingMode::kMixed, 2), {{"a", 0}, {"b", 5}, {"c", -1}},
            "[a:0+8!] [a:d] [b:0+8! a:d] [a:d b:d] [c:0+8! b:d] [b:d c:d] [c:d] [c:d] "},
        {"guided", compactConfig(), {{"a", 0}, {"g", 0}, {"b", 0}},
            "[a:0+8!] [a:d] [a:d] [a:d] [g:0+1024!] [g:d] [g:d] [g:d] [b:0+8!] [b:d] [b:d] [b:d] "},
        {"reserved slot", compactConfig(IfbSchedulingMode::kMixed, 3, SchedulingPolicy::kPriority),
            {{"a", -1}, {"p0", 0}, {"p1", 0}, {"p2", 0}, {"urgent", -1}},
            "[a:0+8!] [a:d] [urgent:0+8! a:d] [a:d urgent:d] [p0:0+8! urgent:d] [urgent:d p0:d] [p1:0+8! p0:d] "
            "[p0:d p1:d] [p2:0+8! p1:d] [p1:d p2:d] [p2:d] [p2:d] "},
    };
    for (Case const& row : cases)
    {
        SCOPED_TRACE(row.what);
        CompactScript script;
        script.promptLengths["g"] = 1024;
        RequestEngine engine(compactSessions(script), row.config);

        std::vector<RequestHandle> handles;
        for (auto const& [prompt, priority] : row.requests)
        {
            handles.push_back(engine.submit(makeRequest(prompt, priority)));
            if (handles.size() == 1)
            {
                ASSERT_TRUE(script.hold.awaitReached());
            }
        }
        script.hold.release.set_value();
        for (RequestHandle& handle : handles)
        {
            EXPECT_EQ(handle.get().outputIds.at(0).size(), 4U);
        }
        EXPECT_EQ(script.steps, row.steps);
    }
}

TEST(RequestEngineCompactTests, PauseWithholdsLowerPriorityResidentsUpToMaxPauseSteps)
{
    // Given mixed with priority and pause, "low" (priority 5) decoding, then "u" and "high" (priority -1), high's
    //       512-token prompt running in four 128-token chunks beside u's decode row, every step taking at least 1 ms
    // When high's prompt runs under the row's maxPauseSteps, and in one row a second priority -1 prompt "high2" (512
    //      tokens) arrives while high's second chunk runs, and in another low is cancelled before the step that runs
    //      high's third chunk
    // Then low is paused while a priority -1 prompt is unfinished, except that with 2 it decodes beside the third
    //      chunk, and high2 extends the pause until its own final chunk; the engine and low count those steps (at
    //      least 1 ms each), u, high and high2 none; the step that retires the cancelled low is not counted as a pause
    struct Case
    {
        int32_t maxPauseSteps;
        bool secondUrgent;
        //! The step at whose top low is cancelled; 0 for none.
        int32_t cancelLowAtStep;
        //! The steps from high's first chunk through the step after the last urgent prompt's final chunk.
        char const* urgentSteps;
        int32_t lowPausedSteps;
    };
    std::vector<Case> const cases{
        {0, false, 0, "[high:0+128 u:d] [high:128+128 u:d] [high:256+128 u:d] [high:384+128! u:d] [u:d high:d low:d] ",
            5},
        {2, false, 0,
            "[high:0+128 u:d] [high:128+128 u:d] [high:256+128 u:d low:d] [high:384+128! u:d] [u:d high:d low:d] ", 4},
        {0, true, 0,
            "[high:0+128 u:d] [high:128+128 u:d] [high:256+128 high2:0+128 u:d] [high:384+128! high2:128+128 u:d] "
            "[high2:256+128 u:d high:d] [high2:384+128! u:d] [u:d high2:d low:d] ",
            7},
        {0, false, 7, "[high:0+128 u:d] [high:128+128 u:d] [high:256+128 u:d] [high:384+128! u:d] [u:d high:d] ", 3},
    };
    for (Case const& row : cases)
    {
        SCOPED_TRACE("maxPauseSteps=" + std::to_string(row.maxPauseSteps) + (row.secondUrgent ? ", high2" : "")
            + (row.cancelLowAtStep != 0 ? ", low cancelled" : ""));
        CompactScript script;
        script.promptLengths["high"] = 512;
        script.promptLengths["high2"] = 512;
        script.tokensFor["low"] = 6;
        script.tokensFor["u"] = 12;
        script.tokensFor["high"] = 2;
        script.tokensFor["high2"] = 2;
        std::optional<RequestHandle> high;
        std::optional<RequestHandle> high2;
        std::function<void(std::optional<RequestHandle>&, char const*)> submitUrgent;
        std::optional<RequestHandle> low;
        script.beforeStep = [&](int32_t step, std::string const&) {
            std::this_thread::sleep_for(1ms);
            if (step == 4)
            {
                submitUrgent(high, "high");
            }
            if (step == 6 && row.secondUrgent)
            {
                submitUrgent(high2, "high2");
            }
            if (step == row.cancelLowAtStep)
            {
                low->cancel();
            }
        };
        EngineConfig config = compactConfig(IfbSchedulingMode::kMixed, 4, SchedulingPolicy::kPriority);
        config.scheduling.preemption = PreemptionMode::kPause;
        config.scheduling.maxChunkTokens = 128;
        config.scheduling.maxPauseSteps = row.maxPauseSteps;
        RequestEngine engine(compactSessions(script), config);
        submitUrgent = [&](std::optional<RequestHandle>& handle, char const* prompt) {
            handle.emplace(engine.submit(makeRequest(prompt, -1)));
        };

        low.emplace(engine.submit(makeRequest("low", 5)));
        ASSERT_TRUE(script.hold.awaitReached());
        RequestHandle u = engine.submit(makeRequest("u", -1));
        script.hold.release.set_value();
        ASSERT_EQ(outcomeOf(u), "completed");
        ASSERT_TRUE(high.has_value());
        ASSERT_EQ(outcomeOf(*high), "completed");
        ASSERT_EQ(high2.has_value(), row.secondUrgent);
        if (high2.has_value())
        {
            ASSERT_EQ(outcomeOf(*high2), "completed");
        }
        // After u's outcome, which is published after the step that retired a cancelled low.
        std::string const lowOutcome = outcomeOf(*low);
        if (row.cancelLowAtStep != 0)
        {
            ASSERT_NE(lowOutcome.find("was cancelled"), std::string::npos) << lowOutcome;
        }
        else
        {
            ASSERT_EQ(lowOutcome, "completed");
        }

        std::string const expected = std::string("[low:0+8!] [low:d] [u:0+8!] [u:d low:d] ") + row.urgentSteps;
        EXPECT_EQ(script.steps.find(expected), 0U) << script.steps;
        EXPECT_EQ(engine.metrics().pausedResidentSteps, static_cast<uint64_t>(row.lowPausedSteps));
        std::vector<std::pair<RequestHandle*, int32_t>> expectedPauses{
            {&*low, row.lowPausedSteps}, {&u, 0}, {&*high, 0}};
        if (high2.has_value())
        {
            expectedPauses.emplace_back(&*high2, 0);
        }
        for (auto const& [handle, paused] : expectedPauses)
        {
            ASSERT_TRUE(handle->stats().has_value());
            EXPECT_EQ(handle->stats()->pausedSteps, paused);
            EXPECT_GE(handle->stats()->pausedTimeUs, static_cast<uint64_t>(paused) * 1000U);
            EXPECT_EQ(handle->stats()->pausedTimeUs == 0, paused == 0);
        }
    }
}

TEST(RequestEngineCompactTests, StepMetricsCountWhatTheStepsExecuted)
{
    // Given mixed (maxNumTokens 640, maxChunkTokens 512) with A decoding and a 725-token prompt X admitted beside it
    // When X is cancelled before its first chunk runs, or after its first 512-token chunk committed, and a 725-token
    //      prompt L submitted meanwhile waits at least 2 ms and runs, its final chunk's step taking at least 3 ms
    // Then X ends cancelled and no step after the cancellation runs it: the step it was planned in runs only A's
    //      decode row; L runs as a 512-token and a 213-token chunk beside A's decode row; the engine counts the steps
    //      by what they executed (prompt, mixed and partial-chunk steps, five full graph decodes, no decode subset);
    //      X's stats are absent until it is ready, then report its 725 prompt tokens, the chunks it ran and no first
    //      token; and L's report its admission, its queue time, its 725 prompt tokens, two chunks and a first token
    //      taken once its final chunk executed, at least its queue time plus that step's 3 ms after submission
    struct Case
    {
        char const* what;
        //! The step before which X is cancelled and L submitted.
        int32_t cancelStep;
        char const* steps;
        uint64_t promptSteps;
        uint64_t mixedSteps;
        uint64_t partialPromptChunks;
        int32_t chunksOfX;
    };
    std::vector<Case> const cases{
        {"before X's first chunk", 3,
            "[a:0+8!] [a:d] [x:0+512 a:d] [long:0+512 a:d] [long:512+213! a:d] [a:d long:d] [long:d] [long:d] ", 3, 2,
            1, 0},
        {"after X's first chunk", 4,
            "[a:0+8!] [a:d] [x:0+512 a:d] [x:512+213! a:d] [long:0+512 a:d] [long:512+213! a:d] [long:d] [long:d] "
            "[long:d] ",
            4, 3, 2, 1},
    };
    for (Case const& row : cases)
    {
        SCOPED_TRACE(row.what);
        CompactScript script;
        script.promptLengths["x"] = 725;
        script.promptLengths["long"] = 725;
        script.tokensFor["a"] = 6;
        std::optional<RequestHandle> x;
        std::optional<RequestHandle> longPrompt;
        std::function<void()> submitLong;
        script.beforeStep = [&](int32_t step, std::string const& line) {
            if (step == row.cancelStep)
            {
                x->cancel();
                submitLong();
                std::this_thread::sleep_for(2ms);
            }
            if (line.find("long:512+213!") != std::string::npos)
            {
                std::this_thread::sleep_for(3ms);
            }
        };
        RequestEngine engine(compactSessions(script), compactConfig());
        submitLong = [&] { longPrompt.emplace(engine.submit(makeRequest("long"))); };

        RequestHandle a = engine.submit(makeRequest("a"));
        ASSERT_TRUE(script.hold.awaitReached());
        x.emplace(engine.submit(makeRequest("x")));
        EXPECT_FALSE(x->stats().has_value()) << "stats were readable before the request was ready";
        script.hold.release.set_value();
        EXPECT_EQ(a.get().outputIds.at(0).size(), 6U);
        EXPECT_NE(outcomeOf(*x).find("was cancelled"), std::string::npos);
        ASSERT_TRUE(longPrompt.has_value());
        EXPECT_EQ(longPrompt->get().outputIds.at(0), (std::vector<int32_t>{1, 2, 3, 4}));

        ASSERT_EQ(script.steps, row.steps);
        EngineMetrics const m = engine.metrics();
        EXPECT_EQ(m.promptSteps, row.promptSteps);
        EXPECT_EQ(m.mixedSteps, row.mixedSteps);
        EXPECT_EQ(m.decodeSteps, 5U);
        EXPECT_EQ(m.decodeGraphSteps, 5U);
        EXPECT_EQ(m.decodeSubsetSteps, 0U);
        EXPECT_EQ(m.partialPromptChunks, row.partialPromptChunks);
        std::optional<RequestStats> const statsX = x->stats();
        ASSERT_TRUE(statsX.has_value());
        EXPECT_EQ(statsX->promptTokens, 725);
        EXPECT_EQ(statsX->promptChunks, row.chunksOfX);
        EXPECT_EQ(statsX->timeToFirstTokenUs, 0U);
        std::optional<RequestStats> const stats = longPrompt->stats();
        ASSERT_TRUE(stats.has_value());
        EXPECT_TRUE(stats->admitted);
        EXPECT_GE(stats->queueTimeUs, 2000U);
        EXPECT_EQ(stats->promptTokens, 725);
        EXPECT_EQ(stats->promptChunks, 2);
        EXPECT_GE(stats->timeToFirstTokenUs, stats->queueTimeUs + 3000U);
    }
}

TEST(RequestEngineCompactTests, ARefusedCompactStepRequeuesWhatSampledNothingAndStopsOnlyAnExplicitMixed)
{
    // Given a session (B = 4, S_in = 1024) that refuses the row's step and then reports no compact execution, under
    //       auto (priority, pause, one session failure allowed), an explicit prefillFirst or an explicit mixed, with
    //       three slots
    // When A decodes, B (1024 tokens) and C (200 tokens) join it and D queues behind them, and the refused step runs
    //      A's decode row (auto, mixed: beside B's final chunk, after B's first chunk ran and with no room for C) or
    //      B's and C's whole prompts (prefillFirst); C is cancelled after the fallback while it waits, or (auto) at
    //      the top of the refused step itself
    // Then one refusal is counted. Under auto and prefillFirst, A, which streamed tokens, fails as a retryable
    //      compact refusal (FailureKind kCompactStepRefused, the reason saying it can be retried); B and C, which
    //      sampled nothing, return to the queue ahead of D and a reopened session serves B with the reference output
    //      and stats of its one run (one prompt chunk), C as cancelled, and D, in single-cohort steps as prefillFirst
    //      without pause (a WARNING says so); every request's queue time is counted once and the engine still serves.
    //      Under mixed, A, B and C fail with the refusal, not retryable, saying that the engine stops serving, D fails
    //      saying that the runtime session is unavailable, naming the mixed mode and the auto alternative, all four as
    //      FailureKind kEngineStopped, and the engine stops. All report why compact is unavailable, the engine counts
    //      exactly the failed and cancelled requests as such, and no request fails to be queued again
    enum class Outcome : uint8_t
    {
        kCompleted,
        kRetryable,
        kRefusedAndStopped,
        kStopped,
        kCancelled,
    };
    struct Case
    {
        IfbSchedulingMode mode;
        int32_t refuseStep;
        //! The step before which C is cancelled; 0 for none.
        int32_t cancelStep;
        std::vector<Outcome> outcomes;
        std::string steps;
        int32_t sessions;
        uint64_t queueLatencyCount;
    };
    using O = Outcome;
    std::vector<Case> const cases{
        {IfbSchedulingMode::kAuto, 4, 5, {O::kRetryable, O::kCompleted, O::kCancelled, O::kCompleted},
            "[a:0+8!] [a:d] [b:0+512 a:d] [b:512+512! a:d] [b:0+1024!] [d:0+8!] [b:d d:d] [b:d d:d] [b:d d:d] ", 2, 4},
        {IfbSchedulingMode::kAuto, 4, 4, {O::kRetryable, O::kCompleted, O::kCancelled, O::kCompleted},
            "[a:0+8!] [a:d] [b:0+512 a:d] [b:512+512! a:d] [b:0+1024!] [d:0+8!] [b:d d:d] [b:d d:d] [b:d d:d] ", 2, 4},
        {IfbSchedulingMode::kPrefillFirst, 3, 4, {O::kRetryable, O::kCompleted, O::kCancelled, O::kCompleted},
            "[a:0+8!] [a:d] [b:0+1024! c:0+200!] [b:0+1024!] [d:0+8!] [b:d d:d] [b:d d:d] [b:d d:d] ", 2, 4},
        {IfbSchedulingMode::kMixed, 4, 0,
            {O::kRefusedAndStopped, O::kRefusedAndStopped, O::kRefusedAndStopped, O::kStopped},
            "[a:0+8!] [a:d] [b:0+512 a:d] [b:512+512! a:d] ", 1, 3},
    };
    for (Case const& row : cases)
    {
        bool const fallsBack = row.mode != IfbSchedulingMode::kMixed;
        SCOPED_TRACE(std::string(toString(row.mode)) + ", C cancelled before step " + std::to_string(row.cancelStep));
        CompactScript script;
        script.refuseStep = row.refuseStep;
        script.promptLengths["b"] = 1024;
        script.promptLengths["c"] = 200;
        script.tokensFor["a"] = 8;
        std::vector<RequestHandle> handles;
        script.beforeStep = [&](int32_t step, std::string const&) {
            if (step == row.cancelStep)
            {
                handles.at(2).cancel();
            }
        };
        EngineConfig config = compactConfig(row.mode, 3);
        if (row.mode == IfbSchedulingMode::kAuto)
        {
            config.scheduling.schedulingPolicy = SchedulingPolicy::kPriority;
            config.scheduling.preemption = PreemptionMode::kPause;
            config.maxConsecutiveSessionFailures = 1;
        }

        testing::internal::CaptureStderr();
        RequestEngine engine(compactSessions(script), config);
        bool const startedCompact = engine.resolvedPolicy().compactExecution;
        handles.push_back(engine.submit(makeRequest("a")));
        bool const reached = script.hold.awaitReached();
        for (char const* prompt : {"b", "c", "d"})
        {
            handles.push_back(engine.submit(makeRequest(prompt)));
        }
        script.hold.release.set_value();
        std::vector<std::string> outcomes;
        std::vector<std::optional<FailureKind>> kinds;
        std::optional<std::vector<int32_t>> outputB;
        for (size_t index = 0; index < handles.size(); ++index)
        {
            try
            {
                std::vector<int32_t> output = handles[index].get().outputIds.at(0);
                if (index == 1)
                {
                    outputB = std::move(output);
                }
                outcomes.push_back("completed");
                kinds.emplace_back();
            }
            catch (RequestFailedError const& error)
            {
                outcomes.push_back(error.what());
                kinds.emplace_back(error.kind());
            }
            catch (std::exception const& error)
            {
                outcomes.push_back(error.what());
                kinds.emplace_back();
            }
        }
        std::string const warnings = testing::internal::GetCapturedStderr();

        ASSERT_TRUE(startedCompact);
        ASSERT_TRUE(reached);
        EngineStatus const status = engine.status();
        for (size_t index = 0; index < outcomes.size(); ++index)
        {
            std::string const& outcome = outcomes[index];
            SCOPED_TRACE("request " + std::string(1, static_cast<char>('a' + index)) + ": " + outcome);
            bool const refused = outcome.find("refused compact execution") != std::string::npos;
            bool const failedWithSession = outcome.find("the runtime session failed") != std::string::npos;
            bool const retryable = outcome.find("can be retried") != std::string::npos;
            switch (row.outcomes[index])
            {
            case O::kCompleted: EXPECT_EQ(outcome, "completed"); break;
            case O::kCancelled: EXPECT_NE(outcome.find("was cancelled"), std::string::npos); break;
            case O::kRetryable:
                EXPECT_TRUE(refused && failedWithSession && retryable);
                EXPECT_EQ(kinds[index], FailureKind::kCompactStepRefused);
                break;
            case O::kRefusedAndStopped:
                EXPECT_TRUE(refused && failedWithSession && !retryable);
                EXPECT_NE(outcome.find("the engine stops serving"), std::string::npos);
                EXPECT_EQ(kinds[index], FailureKind::kEngineStopped);
                break;
            case O::kStopped:
                EXPECT_EQ(kinds[index], FailureKind::kEngineStopped);
                EXPECT_NE(outcome.find("the runtime session is unavailable"), std::string::npos);
                EXPECT_NE(outcome.find("ifbSchedulingMode=mixed requires compact execution"), std::string::npos);
                EXPECT_NE(outcome.find("ifbSchedulingMode=auto"), std::string::npos);
                EXPECT_NE(outcome.find(status.unavailableReason), std::string::npos) << status.unavailableReason;
                break;
            }
        }
        if (fallsBack)
        {
            ASSERT_TRUE(outputB.has_value());
            EXPECT_EQ(*outputB, (std::vector<int32_t>{1, 2, 3, 4}));
            std::optional<RequestStats> const statsB = handles[1].stats();
            ASSERT_TRUE(statsB.has_value());
            EXPECT_EQ(statsB->promptChunks, 1);
        }
        EXPECT_EQ(script.steps, row.steps);
        EXPECT_EQ(script.sessions, row.sessions);
        EngineMetrics const metrics = engine.metrics();
        EXPECT_EQ(metrics.compactRefusals, 1U);
        EXPECT_EQ(metrics.queueLatencyCount, row.queueLatencyCount);
        auto const count = [&row](std::initializer_list<Outcome> kinds) {
            return static_cast<uint64_t>(std::count_if(row.outcomes.begin(), row.outcomes.end(),
                [kinds](Outcome outcome) { return std::find(kinds.begin(), kinds.end(), outcome) != kinds.end(); }));
        };
        EXPECT_EQ(metrics.cancelled, count({O::kCancelled}));
        EXPECT_EQ(metrics.failed, count({O::kRetryable, O::kRefusedAndStopped, O::kStopped}));
        EXPECT_EQ(warnings.find("cannot be queued again"), std::string::npos) << warnings;
        ResolvedSchedulingPolicy const policy = engine.resolvedPolicy();
        EXPECT_EQ(policy.mode, fallsBack ? IfbSchedulingMode::kPrefillFirst : IfbSchedulingMode::kMixed);
        EXPECT_EQ(policy.compactExecution, !fallsBack);
        EXPECT_EQ(policy.preemption, PreemptionMode::kNone);
        EXPECT_EQ(warnings.find("scheduling changed after a plugin refused a compact step: mode=prefillFirst")
                != std::string::npos,
            fallsBack)
            << warnings;
        EXPECT_EQ(status.serving, fallsBack);
        EXPECT_EQ(status.unavailableReason.empty(), fallsBack) << status.unavailableReason;
        EXPECT_NE(status.compactUnavailableReason.find("refused"), std::string::npos)
            << status.compactUnavailableReason;
    }
}

TEST(RequestEngineCompactTests, ARefusedCompactStepStopsTheEngineWhenNoSessionReopens)
{
    // Given a session under auto (B = 4, S_in = 1024) that refuses the step running A's decode row beside B's first
    //       chunk, and a runtime that opens no further session
    // When A decodes and B (1024 tokens) joins it
    // Then A, which streamed tokens, fails with the refusal and the reason the engine stopped, and B, queued again,
    //      fails as the session is unavailable; both as FailureKind kEngineStopped and neither saying it can be
    //      retried, and the engine stops serving with the reason no session opened
    CompactScript script;
    script.refuseStep = 3;
    script.openableSessions = 1;
    script.promptLengths["b"] = 1024;
    script.tokensFor["a"] = 8;
    RequestEngine engine(compactSessions(script), compactConfig(IfbSchedulingMode::kAuto));
    std::vector<RequestHandle> handles;
    handles.push_back(engine.submit(makeRequest("a")));
    ASSERT_TRUE(script.hold.awaitReached());
    handles.push_back(engine.submit(makeRequest("b")));
    script.hold.release.set_value();
    std::vector<std::string> messages;
    std::vector<std::optional<FailureKind>> kinds;
    for (RequestHandle& handle : handles)
    {
        try
        {
            handle.get();
            messages.emplace_back("completed");
            kinds.emplace_back();
        }
        catch (RequestFailedError const& error)
        {
            messages.emplace_back(error.what());
            kinds.emplace_back(error.kind());
        }
    }

    EXPECT_EQ(script.steps, "[a:0+8!] [a:d] [b:0+512 a:d] ");
    std::string const noSession = "the runtime cannot open another session";
    EXPECT_EQ(kinds.at(0), FailureKind::kEngineStopped) << messages.at(0);
    EXPECT_NE(messages.at(0).find("refused compact execution"), std::string::npos) << messages.at(0);
    EXPECT_NE(messages.at(0).find("the engine stopped serving: " + noSession), std::string::npos) << messages.at(0);
    EXPECT_EQ(kinds.at(1), FailureKind::kEngineStopped) << messages.at(1);
    EXPECT_NE(messages.at(1).find("the runtime session is unavailable"), std::string::npos) << messages.at(1);
    for (std::string const& message : messages)
    {
        EXPECT_EQ(message.find("can be retried"), std::string::npos) << message;
    }
    EngineStatus const status = engine.status();
    EXPECT_FALSE(status.serving);
    EXPECT_NE(status.unavailableReason.find(noSession), std::string::npos) << status.unavailableReason;
}
