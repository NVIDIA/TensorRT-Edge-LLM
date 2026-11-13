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

// MultiStageRuntime over host-only fake stages, built through the runner table load() takes. The only GPU work is
// the lane stream load() creates.

#include "runtime/multiStage/multiStageRuntime.h"
#include "runtime/multiStage/stageRegistry.h"
#include "runtime/state/sharedResources.h"
#include "runtime/streaming.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

using namespace trt_edgellm::rt::multiStage;
using nlohmann::json;
using trt_edgellm::rt::AudioStreamChannel;
using trt_edgellm::rt::SharedResources;
using trt_edgellm::rt::StreamChannel;
using ::testing::HasSubstr;
using ::testing::Not;

namespace
{

constexpr char const* kProducerRunner = "unittest.executor.producer";
constexpr char const* kConsumerRunner = "unittest.executor.consumer";
constexpr char const* kVocoderRunner = "unittest.executor.vocoder";
//! Above any count a test asks for, so a producer's buffer never moves under a reader.
constexpr size_t kMaxTokens = 64;
//! Above the steps any test takes per member. A stage stepped past it throws rather than spinning.
constexpr int32_t kStepCap = 64;
//! How long a stage waits for its wake, and a test for that stage to start waiting.
constexpr std::chrono::seconds kWakeWait{2};
//! How long run() may take before every member it still holds is cancelled.
constexpr std::chrono::seconds kRunDeadline{5};

using Strings = std::vector<std::string>;

//! What one stage saw, by batch slot.
struct StageRecord
{
    int32_t built{0};
    int32_t destroyed{0};
    int32_t maxBatch{0};
    int32_t resets{0};
    std::map<int32_t, int32_t> prepares;
    std::map<int32_t, int32_t> steps;
    std::map<int32_t, Strings> finishes;
    std::map<int32_t, std::vector<int32_t>> received;
    //! The sinks prepare() was handed: "text", "audio", both or neither.
    std::map<int32_t, Strings> sinks;
    //! The batch slots in each stepCohort() call.
    std::vector<std::vector<int32_t>> calls;
};

//! What the fakes saw, for a test to read after run(). Only the wake fields are shared with another thread.
struct Probe
{
    std::map<std::string, StageRecord> stages;
    //! For a stage that cancels its own request, as a caller's cancel landing mid-step would.
    MultiStageRuntime* runtime{nullptr};
    std::atomic<int32_t> wakes{0};
    std::atomic<bool> waiting{false};

    void reset()
    {
        stages.clear();
        runtime = nullptr;
        wakes = 0;
        waiting = false;
    }
};

Probe& probe()
{
    static Probe instance;
    return instance;
}

StageRecord& record(std::string const& stage)
{
    return probe().stages[stage];
}

/*!
 * @brief A stage whose config picks its behavior; every knob is off by default.
 *
 * Without a `tokens` input it writes one, a token per step: requestId * 100 + 1, + 2, ... up to `count`
 * (`count_by_slot` per batch slot), the last marking the port complete. With the input it reads what is new
 * each step and reports kBlocked until the port is complete. Per-request state is kept by batch slot.
 *
 * Knobs: `publish` (false: never binds `tokens`); `hold_after` and `hold_steps` (once that many tokens are
 * written, kBlocked for that many steps); `throw_prepare_slot`, `fail_slot` (kFailed on that member's first
 * step), `throw_finish_slot`, `throw_report_slot` and `refuse_slot` (admit() says "busy"), each a batch slot;
 * `throw_on_call` (the stepCohort() call, from 1, that throws); `throw_int` (the other throw_* knobs throw an
 * int rather than a std::exception); `cancel_slot` (cancels that member in its first step); `cancel_at_last` (cancels
 * in the step that reads the last token); `wait_for_wake` (the first step waits for the abandon wake registered in
 * prepare()); `report` (diagnostics() gives the request id and the member's step count).
 */
class FakeStage final : public IStage
{
public:
    explicit FakeStage(StageInitContext const& init)
        : mId(init.config->id)
        , mMaxBatch(init.maxBatch)
        , mReads(init.config->findInput("tokens") != nullptr)
        , mCount(init.config->config.value("count", 4))
        , mCountBySlot(init.config->config.value("count_by_slot", std::vector<int32_t>{}))
        , mPublish(init.config->config.value("publish", true))
        , mHoldAfter(init.config->config.value("hold_after", -1))
        , mHoldSteps(init.config->config.value("hold_steps", 0))
        , mThrowPrepareSlot(init.config->config.value("throw_prepare_slot", -1))
        , mThrowOnCall(init.config->config.value("throw_on_call", -1))
        , mFailSlot(init.config->config.value("fail_slot", -1))
        , mThrowFinishSlot(init.config->config.value("throw_finish_slot", -1))
        , mThrowReportSlot(init.config->config.value("throw_report_slot", -1))
        , mThrowInt(init.config->config.value("throw_int", false))
        , mRefuseSlot(init.config->config.value("refuse_slot", -1))
        , mCancelSlot(init.config->config.value("cancel_slot", -1))
        , mCancelAtLast(init.config->config.value("cancel_at_last", false))
        , mWaitForWake(init.config->config.value("wait_for_wake", false))
        , mReport(init.config->config.value("report", false))
    {
        auto& seen = record(mId);
        ++seen.built;
        seen.maxBatch = init.maxBatch;
    }

    ~FakeStage() override
    {
        ++record(mId).destroyed;
    }

    void preallocate() override
    {
        mRows.resize(static_cast<size_t>(mMaxBatch));
        for (auto& row : mRows)
        {
            row.values.reserve(kMaxTokens);
        }
    }

    bool admit(StageExecContext const& ctx, std::string& reason) override
    {
        if (ctx.slot->batchSlot == mRefuseSlot)
        {
            reason = "busy";
            return false;
        }
        return true;
    }

    void prepare(StageExecContext& ctx) override
    {
        auto const slot = ctx.slot->batchSlot;
        auto& seen = record(mId);
        ++seen.prepares[slot];
        auto& sinks = seen.sinks[slot];
        sinks.clear();
        if (ctx.textSink != nullptr)
        {
            sinks.emplace_back("text");
        }
        if (ctx.audioSink != nullptr)
        {
            sinks.emplace_back("audio");
        }
        if (slot == mThrowPrepareSlot)
        {
            explode("prepare exploded");
        }
        auto& row = mRows.at(static_cast<size_t>(slot));
        row.values.clear();
        row.cursor = 0;
        row.steps = 0;
        row.held = 0;
        if (mWaitForWake)
        {
            {
                std::lock_guard<std::mutex> const guard(mWakeMutex);
                mWoken = false;
            }
            ctx.slot->abandon.onRequest([this]() { wake(); });
        }
    }

    StageStatus step(StageExecContext& ctx) override
    {
        auto const slot = ctx.slot->batchSlot;
        auto& row = mRows.at(static_cast<size_t>(slot));
        ++record(mId).steps[slot];
        if (++row.steps > kStepCap)
        {
            throw std::runtime_error(mId + " was stepped past the cap");
        }
        if (slot == mCancelSlot && row.steps == 1)
        {
            probe().runtime->cancel(*ctx.slot);
        }
        if (mWaitForWake && row.steps == 1)
        {
            probe().waiting = true;
            std::unique_lock<std::mutex> lock(mWakeMutex);
            bool const woken = mWakeCv.wait_for(lock, kWakeWait, [this]() { return mWoken; });
            return woken ? StageStatus::kMore : StageStatus::kFailed;
        }
        if (slot == mFailSlot && row.steps == 1)
        {
            return StageStatus::kFailed;
        }
        return mReads ? read(ctx, row) : write(ctx, row);
    }

    void stepCohort(std::vector<StageExecContext*> const& members, std::vector<StageStatus>& statuses) override
    {
        std::vector<int32_t> slots;
        for (auto const* member : members)
        {
            slots.push_back(member->slot->batchSlot);
        }
        record(mId).calls.push_back(std::move(slots));
        if (++mCalls == mThrowOnCall)
        {
            explode("stepCohort exploded");
        }
        for (size_t i = 0; i < members.size(); ++i)
        {
            statuses[i] = step(*members[i]);
        }
    }

    void finish(StageExecContext& ctx, FinishReason reason) override
    {
        auto const slot = ctx.slot->batchSlot;
        record(mId).finishes[slot].push_back(toString(reason));
        if (slot == mThrowFinishSlot)
        {
            explode("finish exploded");
        }
    }

    void reset() override
    {
        ++record(mId).resets;
    }

    json diagnostics(RequestSlot const& slot) const override
    {
        if (slot.batchSlot == mThrowReportSlot)
        {
            explode("report exploded");
        }
        if (!mReport)
        {
            return json::object();
        }
        return {{"request", slot.requestId}, {"steps", mRows.at(static_cast<size_t>(slot.batchSlot)).steps}};
    }

private:
    struct Row
    {
        std::vector<int32_t> values;
        int64_t cursor{0};
        int32_t steps{0};
        int32_t held{0};
    };

    [[noreturn]] void explode(char const* message) const
    {
        if (mThrowInt)
        {
            throw 1;
        }
        throw std::runtime_error(message);
    }

    StageStatus write(StageExecContext& ctx, Row& row)
    {
        auto const slot = static_cast<size_t>(ctx.slot->batchSlot);
        auto const count = slot < mCountBySlot.size() ? mCountBySlot[slot] : mCount;
        if (static_cast<int32_t>(row.values.size()) == mHoldAfter && row.held < mHoldSteps)
        {
            ++row.held;
            return StageStatus::kBlocked;
        }
        row.values.push_back(
            static_cast<int32_t>(ctx.slot->requestId) * 100 + static_cast<int32_t>(row.values.size()) + 1);
        bool const complete = static_cast<int32_t>(row.values.size()) == count;
        if (mPublish)
        {
            PortValue value{row.values.data(), {static_cast<int64_t>(row.values.size())}, PortKind::kText,
                static_cast<int32_t>(sizeof(int32_t))};
            value.complete = complete;
            ctx.out->bind("tokens", value);
        }
        return complete ? StageStatus::kDone : StageStatus::kMore;
    }

    StageStatus read(StageExecContext& ctx, Row& row)
    {
        auto const* tokens = ctx.in->find("tokens");
        if (tokens == nullptr)
        {
            return StageStatus::kBlocked;
        }
        auto const* values = static_cast<int32_t const*>(tokens->data);
        auto& received = record(mId).received[ctx.slot->batchSlot];
        for (; row.cursor < tokens->shape.front(); ++row.cursor)
        {
            received.push_back(values[row.cursor]);
        }
        if (!tokens->complete)
        {
            return StageStatus::kBlocked;
        }
        if (mCancelAtLast)
        {
            probe().runtime->cancel(*ctx.slot);
        }
        return StageStatus::kDone;
    }

    void wake()
    {
        {
            std::lock_guard<std::mutex> const guard(mWakeMutex);
            mWoken = true;
        }
        ++probe().wakes;
        mWakeCv.notify_all();
    }

    std::string const mId;
    int32_t const mMaxBatch;
    bool const mReads;
    int32_t const mCount;
    std::vector<int32_t> const mCountBySlot;
    bool const mPublish;
    int32_t const mHoldAfter;
    int32_t const mHoldSteps;
    int32_t const mThrowPrepareSlot;
    int32_t const mThrowOnCall;
    int32_t const mFailSlot;
    int32_t const mThrowFinishSlot;
    int32_t const mThrowReportSlot;
    bool const mThrowInt;
    int32_t const mRefuseSlot;
    int32_t const mCancelSlot;
    bool const mCancelAtLast;
    bool const mWaitForWake;
    bool const mReport;
    std::vector<Row> mRows;
    int32_t mCalls{0};
    std::mutex mWakeMutex;
    std::condition_variable mWakeCv;
    bool mWoken{false};
};

std::unique_ptr<IStage> buildFake(StageInitContext const& init)
{
    return std::make_unique<FakeStage>(init);
}

std::unique_ptr<IStage> buildNothing(StageInitContext const& /*init*/)
{
    return nullptr;
}

RunnerDecl declaration(
    char const* name, std::pair<std::string, PortKind> port, std::vector<std::string> inputs, bool batches)
{
    RunnerDecl decl;
    decl.name = name;
    decl.kind = StageKind::kHost;
    decl.ports = {std::move(port)};
    decl.inputs = std::move(inputs);
    decl.batches = batches;
    return decl;
}

//! The fake runners, with @p consumer as the consumer's factory.
StageRegistry fakeRunners(StageFactory consumer = &buildFake)
{
    return {{declaration(kProducerRunner, {"tokens", PortKind::kText}, {}, true), &buildFake},
        {declaration(kConsumerRunner, {"text", PortKind::kText}, {"tokens"}, true), consumer},
        // Does not batch, so a route through it seats one request whatever its max_batch.
        {declaration(kVocoderRunner, {"audio", PortKind::kAudio}, {"tokens"}, false), &buildFake}};
}

//! A stage that reads the producer's tokens and emits one terminal port.
json reader(char const* id, char const* runner, int32_t width, json config, char const* port, char const* kind)
{
    return {{"id", id}, {"runner", runner}, {"max_batch", width}, {"config", std::move(config)},
        {"inputs", json::array({{{"name", "tokens"}, {"from", "producer:tokens"}}})},
        {"emits", json::array({{{"id", port}, {"kind", kind}, {"terminal", true}}})}};
}

//! Producer, then a consumer holding the text terminal, both @p width wide.
json pipeline(json producer = json::object(), json consumer = json::object(), int32_t width = 1)
{
    json const producerStage{{"id", "producer"}, {"runner", kProducerRunner}, {"max_batch", width},
        {"config", std::move(producer)}, {"emits", json::array({{{"id", "tokens"}, {"kind", "text"}}})}};
    json const consumerStage = reader("consumer", kConsumerRunner, width, std::move(consumer), "text", "text");
    return {{"schema_version", MultiStageConfig::kSchemaVersion}, {"label", "executor"},
        {"stages", json::array({producerStage, consumerStage})}, {"terminals", json::array({"consumer:text"})}};
}

//! pipeline() plus a vocoder, also @p width wide, that reads the tokens and holds the audio terminal.
json withVocoder(
    int32_t width, json producer = json::object(), json consumer = json::object(), json vocoder = json::object())
{
    auto topology = pipeline(std::move(producer), std::move(consumer), width);
    topology["stages"].push_back(reader("vocoder", kVocoderRunner, width, std::move(vocoder), "audio", "audio"));
    topology["terminals"].push_back("vocoder:audio");
    return topology;
}

//! What the producer writes for @p requestId.
std::vector<int32_t> tokensOf(uint64_t requestId, int32_t count)
{
    std::vector<int32_t> tokens;
    for (int32_t index = 1; index <= count; ++index)
    {
        tokens.push_back(static_cast<int32_t>(requestId) * 100 + index);
    }
    return tokens;
}

//! "<terminal>:<reason>" for each terminal @p response reports.
Strings reasonsOf(Response const& response)
{
    Strings reasons;
    for (auto const& entry : response.terminalReasons)
    {
        reasons.push_back(entry.first + ":" + toString(entry.second));
    }
    return reasons;
}

//! Whether any stepCohort() call of @p stage held @p slot.
bool stepped(std::string const& stage, int32_t slot)
{
    auto const& calls = record(stage).calls;
    return std::any_of(calls.begin(), calls.end(),
        [slot](std::vector<int32_t> const& call) { return std::find(call.begin(), call.end(), slot) != call.end(); });
}

class MultiStageRuntimeTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        probe().reset();
    }

    void load(MultiStageRuntime& runtime, json const& topology, StageRegistry const& runners = fakeRunners())
    {
        runtime.load(MultiStageConfig::parse(topology), "", mShared, &runners);
    }

    //! Offers one request with a text sink, and an audio sink too when @p audio. Null, with @p error filled, when
    //! it is refused.
    RequestSlot* submit(MultiStageRuntime& runtime, std::string& error, bool audio = false)
    {
        Sinks sinks;
        sinks.text = StreamChannel::create();
        if (audio)
        {
            mAudio.push_back(std::make_unique<AudioStreamChannel>());
            sinks.audio = mAudio.back().get();
        }
        return runtime.submit(PortBindings{}, sinks, error);
    }

    //! Seats @p members text-only requests, each of which must be admitted.
    std::vector<RequestSlot*> seat(MultiStageRuntime& runtime, size_t members)
    {
        std::vector<RequestSlot*> slots;
        for (size_t member = 0; member < members; ++member)
        {
            std::string error;
            slots.push_back(submit(runtime, error));
            EXPECT_NE(slots.back(), nullptr) << error;
        }
        return slots;
    }

    //! run(), cancelling @p slots if it has not returned by kRunDeadline: an executor looping without calling any
    //! stage, as one without its stall check would, then fails the test rather than hanging it.
    std::vector<Response> run(MultiStageRuntime& runtime, std::vector<RequestSlot*> const& slots)
    {
        std::mutex mutex;
        std::condition_variable returned;
        bool done = false;
        std::thread deadline([&]() {
            {
                std::unique_lock<std::mutex> lock(mutex);
                if (returned.wait_for(lock, kRunDeadline, [&done]() { return done; }))
                {
                    return;
                }
            }
            for (auto* slot : slots)
            {
                if (slot != nullptr)
                {
                    runtime.cancel(*slot);
                }
            }
        });
        std::vector<Response> responses;
        EXPECT_NO_THROW(responses = runtime.run());
        {
            std::lock_guard<std::mutex> const guard(mutex);
            done = true;
        }
        returned.notify_one();
        deadline.join();
        return responses;
    }

    SharedResources mShared;
    std::vector<std::unique_ptr<AudioStreamChannel>> mAudio;
};

} // namespace

// Given a producer that publishes one token and then reports kBlocked, resuming only after 20 more steps,
// When the request runs,
// Then run() stops after the first round in which no stage moved: the request fails with an error naming both
// waiting stages, both stages finish it with kFailed, and the producer was stepped only twice.
TEST_F(MultiStageRuntimeTest, AProducerThatStopsShortIsReportedStuck)
{
    MultiStageRuntime runtime;
    load(runtime, pipeline(json{{"count", 4}, {"hold_after", 1}, {"hold_steps", 20}}));
    auto const slots = seat(runtime, 1);

    auto const responses = run(runtime, slots);

    ASSERT_EQ(responses.size(), 1U);
    EXPECT_FALSE(responses[0].ok);
    EXPECT_THAT(responses[0].error, HasSubstr("no stage made progress"));
    EXPECT_THAT(responses[0].error, HasSubstr("'producer'"));
    EXPECT_THAT(responses[0].error, HasSubstr("'consumer'"));
    EXPECT_EQ(reasonsOf(responses[0]), Strings{"consumer:text:failed"});
    EXPECT_EQ(record("producer").finishes[0], Strings{"failed"});
    EXPECT_EQ(record("consumer").finishes[0], Strings{"failed"});
    EXPECT_EQ(record("producer").steps[0], 2);
    EXPECT_EQ(record("consumer").received[0], tokensOf(slots[0]->requestId, 1));
}

// Given a producer that finishes without ever publishing the tokens its consumer reads,
// When the request runs,
// Then the request fails as stuck with an error naming the consumer but not the finished producer, and the
// consumer, never prepared, still gets finish(kFailed).
TEST_F(MultiStageRuntimeTest, AnInputNoStagePublishesIsReportedAgainstItsConsumer)
{
    MultiStageRuntime runtime;
    load(runtime, pipeline(json{{"count", 2}, {"publish", false}}));
    auto const slots = seat(runtime, 1);

    auto const responses = run(runtime, slots);

    ASSERT_EQ(responses.size(), 1U);
    EXPECT_FALSE(responses[0].ok);
    EXPECT_THAT(responses[0].error, HasSubstr("no stage made progress"));
    EXPECT_THAT(responses[0].error, HasSubstr("'consumer'"));
    EXPECT_THAT(responses[0].error, Not(HasSubstr("'producer'")));
    EXPECT_EQ(record("consumer").prepares[0], 0);
    EXPECT_EQ(record("consumer").finishes[0], Strings{"failed"});
}

// Given a producer that publishes eight tokens one per step, and a consumer that reports kBlocked until the last,
// When the request runs,
// Then it completes with every token received in order: a consumer waiting on a producer that still moves is not
// stuck.
TEST_F(MultiStageRuntimeTest, AConsumerWaitingOnAProgressingProducerIsNotStuck)
{
    MultiStageRuntime runtime;
    load(runtime, pipeline(json{{"count", 8}}));
    auto const slots = seat(runtime, 1);

    auto const responses = run(runtime, slots);

    ASSERT_EQ(responses.size(), 1U);
    EXPECT_TRUE(responses[0].ok) << responses[0].error;
    EXPECT_EQ(reasonsOf(responses[0]), Strings{"consumer:text:completed"});
    EXPECT_EQ(record("consumer").received[0], tokensOf(slots[0]->requestId, 8));
}

// Given a cohort of two whose producer's prepare() throws for the second member,
// When the cohort runs,
// Then only that member fails, with the thrown message after the stage's name; it is neither stepped nor handed to
// the consumer, yet both stages finish it with kFailed, and the first member completes with all its tokens.
TEST_F(MultiStageRuntimeTest, APrepareThrowFailsOnlyThatMember)
{
    MultiStageRuntime runtime;
    load(runtime, pipeline(json{{"count", 3}, {"throw_prepare_slot", 1}}, json::object(), 2));
    auto const slots = seat(runtime, 2);

    auto const responses = run(runtime, slots);

    ASSERT_EQ(responses.size(), 2U);
    EXPECT_FALSE(responses[1].ok);
    EXPECT_EQ(responses[1].error, "stage 'producer': prepare exploded");
    EXPECT_EQ(reasonsOf(responses[1]), Strings{"consumer:text:failed"});
    EXPECT_FALSE(stepped("producer", 1));
    EXPECT_EQ(record("consumer").prepares[1], 0);
    EXPECT_EQ(record("producer").finishes[1], Strings{"failed"});
    EXPECT_EQ(record("consumer").finishes[1], Strings{"failed"});
    EXPECT_TRUE(responses[0].ok) << responses[0].error;
    EXPECT_EQ(record("consumer").received[0], tokensOf(slots[0]->requestId, 3));
}

// Given a pipeline whose consumer, the second stage on the route, throws in prepare(),
// When the request runs,
// Then the error names the consumer, the stage that threw.
TEST_F(MultiStageRuntimeTest, APrepareThrowNamesTheStageThatThrew)
{
    MultiStageRuntime runtime;
    load(runtime, pipeline(json::object(), json{{"throw_prepare_slot", 0}}));
    auto const slots = seat(runtime, 1);

    auto const responses = run(runtime, slots);

    ASSERT_EQ(responses.size(), 1U);
    EXPECT_EQ(responses[0].error, "stage 'consumer': prepare exploded");
}

// Given a cohort of two whose consumer, the second stage on the route, throws an int rather than a std::exception from
// prepare() for the second member,
// When the cohort runs,
// Then only that member fails, with "stage 'consumer': unknown exception in prepare()"; the consumer never steps it,
// both stages finish it with kFailed, and the first member is ok.
TEST_F(MultiStageRuntimeTest, ANonStdExceptionFromPrepareFailsOnlyThatMember)
{
    MultiStageRuntime runtime;
    load(runtime, pipeline(json::object(), json{{"throw_prepare_slot", 1}, {"throw_int", true}}, 2));
    auto const slots = seat(runtime, 2);

    auto const responses = run(runtime, slots);

    ASSERT_EQ(responses.size(), 2U);
    EXPECT_EQ(responses[1].error, "stage 'consumer': unknown exception in prepare()");
    EXPECT_FALSE(stepped("consumer", 1));
    EXPECT_EQ(record("producer").finishes[1], Strings{"failed"});
    EXPECT_EQ(record("consumer").finishes[1], Strings{"failed"});
    EXPECT_TRUE(responses[0].ok) << responses[0].error;
}

// Given a cohort of two whose consumer's stepCohort() throws on its second call, after the producer has stepped each
// member twice,
// When the cohort runs,
// Then run() returns rather than throws, both members fail with the thrown message after the stage's name, and both
// stages still finish both members with kFailed.
TEST_F(MultiStageRuntimeTest, AStepCohortThrowFailsEveryLiveMemberAndStillFinishesThem)
{
    MultiStageRuntime runtime;
    load(runtime, pipeline(json{{"count", 4}}, json{{"throw_on_call", 2}}, 2));
    auto const slots = seat(runtime, 2);

    auto const responses = run(runtime, slots);

    ASSERT_EQ(responses.size(), 2U);
    for (int32_t member = 0; member < 2; ++member)
    {
        auto const& response = responses[static_cast<size_t>(member)];
        EXPECT_FALSE(response.ok) << member;
        EXPECT_EQ(response.error, "stage 'consumer': stepCohort exploded") << member;
        EXPECT_EQ(record("producer").finishes[member], Strings{"failed"}) << member;
        EXPECT_EQ(record("consumer").finishes[member], Strings{"failed"}) << member;
        EXPECT_EQ(record("producer").steps[member], 2) << member;
    }
}

// Given a cohort of two whose consumer's stepCohort() throws an int rather than a std::exception on its second call,
// When the cohort runs,
// Then both members fail with "stage 'consumer': unknown exception in stepCohort()".
TEST_F(MultiStageRuntimeTest, ANonStdExceptionFromStepCohortNamesTheStage)
{
    MultiStageRuntime runtime;
    load(runtime, pipeline(json{{"count", 4}}, json{{"throw_on_call", 2}, {"throw_int", true}}, 2));
    auto const slots = seat(runtime, 2);

    auto const responses = run(runtime, slots);

    ASSERT_EQ(responses.size(), 2U);
    for (auto const& response : responses)
    {
        EXPECT_EQ(response.error, "stage 'consumer': unknown exception in stepCohort()") << response.requestId;
    }
}

// Given a cohort of two whose first member is done after two tokens while the second has four more to go, and a
// consumer whose fourth stepCohort() throws,
// When the cohort runs,
// Then only the second member fails: the first, already finished, keeps its tokens and kCompleted.
TEST_F(MultiStageRuntimeTest, AFinishedMemberKeepsItsOutcomeWhenACohortMateThrows)
{
    MultiStageRuntime runtime;
    load(runtime, pipeline(json{{"count_by_slot", {2, 6}}}, json{{"throw_on_call", 4}}, 2));
    auto const slots = seat(runtime, 2);

    auto const responses = run(runtime, slots);

    ASSERT_EQ(responses.size(), 2U);
    EXPECT_TRUE(responses[0].ok) << responses[0].error;
    EXPECT_EQ(reasonsOf(responses[0]), Strings{"consumer:text:completed"});
    EXPECT_EQ(record("consumer").finishes[0], Strings{"completed"});
    EXPECT_EQ(record("consumer").received[0], tokensOf(slots[0]->requestId, 2));
    EXPECT_FALSE(responses[1].ok);
    EXPECT_THAT(responses[1].error, HasSubstr("stepCohort exploded"));
}

// Given a request cancelled after submit() and before run(),
// When the cohort runs,
// Then no stage prepares or steps it, it is ok with its terminal reading kCancelled, and both stages still finish it
// with kCancelled.
TEST_F(MultiStageRuntimeTest, ACancelBeforeRunStartsNoStage)
{
    MultiStageRuntime runtime;
    load(runtime, pipeline());
    auto const slots = seat(runtime, 1);
    ASSERT_NE(slots[0], nullptr);
    runtime.cancel(*slots[0]);

    auto const responses = run(runtime, slots);

    ASSERT_EQ(responses.size(), 1U);
    EXPECT_EQ(record("producer").prepares[0], 0);
    EXPECT_TRUE(record("producer").calls.empty());
    EXPECT_EQ(record("consumer").prepares[0], 0);
    EXPECT_TRUE(responses[0].ok) << responses[0].error;
    EXPECT_EQ(reasonsOf(responses[0]), Strings{"consumer:text:cancelled"});
    EXPECT_EQ(record("producer").finishes[0], Strings{"cancelled"});
    EXPECT_EQ(record("consumer").finishes[0], Strings{"cancelled"});
}

// Given a producer that registers its abandon wake in prepare() and then waits inside its first step until woken,
// When another thread cancels the request during that wait,
// Then the wake fires once, run() returns before that wait would have timed out, without stepping the producer
// again, and the request reads as cancelled, with both stages finishing it with kCancelled.
TEST_F(MultiStageRuntimeTest, ACancelFromAnotherThreadWakesAStageBlockedInItsStep)
{
    MultiStageRuntime runtime;
    load(runtime, pipeline(json{{"wait_for_wake", true}}));
    auto const slots = seat(runtime, 1);
    ASSERT_NE(slots[0], nullptr);
    std::thread canceller([&]() {
        auto const giveUp = std::chrono::steady_clock::now() + kWakeWait;
        while (!probe().waiting && std::chrono::steady_clock::now() < giveUp)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        runtime.cancel(*slots[0]);
    });

    auto const start = std::chrono::steady_clock::now();
    auto const responses = run(runtime, slots);
    auto const elapsed = std::chrono::steady_clock::now() - start;
    canceller.join();

    ASSERT_EQ(responses.size(), 1U);
    EXPECT_EQ(probe().wakes.load(), 1);
    EXPECT_EQ(record("producer").steps[0], 1);
    EXPECT_LT(elapsed, kWakeWait) << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count() << " ms";
    EXPECT_TRUE(responses[0].ok) << responses[0].error;
    EXPECT_EQ(reasonsOf(responses[0]), Strings{"consumer:text:cancelled"});
    EXPECT_EQ(record("producer").finishes[0], Strings{"cancelled"});
    EXPECT_EQ(record("consumer").finishes[0], Strings{"cancelled"});
}

// Given a consumer whose request is cancelled during the step that reads the last token, a step it ends with kDone,
// When the request runs,
// Then the request reads as cancelled, not completed, and both stages finish it with kCancelled: a stage that saw the
// cancel may have stopped short of what it would have produced.
TEST_F(MultiStageRuntimeTest, ACancelSeenDuringTheLastStepReadsAsCancelled)
{
    MultiStageRuntime runtime;
    probe().runtime = &runtime;
    load(runtime, pipeline(json::object(), json{{"cancel_at_last", true}}));
    auto const slots = seat(runtime, 1);

    auto const responses = run(runtime, slots);

    ASSERT_EQ(responses.size(), 1U);
    EXPECT_EQ(record("consumer").received[0], tokensOf(slots[0]->requestId, 4));
    EXPECT_TRUE(responses[0].ok) << responses[0].error;
    EXPECT_EQ(reasonsOf(responses[0]), Strings{"consumer:text:cancelled"});
    EXPECT_EQ(record("producer").finishes[0], Strings{"cancelled"});
    EXPECT_EQ(record("consumer").finishes[0], Strings{"cancelled"});
}

// Given a cohort of two whose second member is cancelled during the producer's first step,
// When the cohort runs,
// Then only that member stops, after that one step, reading as cancelled, while the first completes with all its
// tokens.
TEST_F(MultiStageRuntimeTest, CancellingOneMemberLeavesItsCohortMateRunning)
{
    MultiStageRuntime runtime;
    probe().runtime = &runtime;
    load(runtime, pipeline(json{{"cancel_slot", 1}}, json::object(), 2));
    auto const slots = seat(runtime, 2);

    auto const responses = run(runtime, slots);

    ASSERT_EQ(responses.size(), 2U);
    EXPECT_EQ(record("producer").steps[1], 1);
    EXPECT_TRUE(responses[1].ok) << responses[1].error;
    EXPECT_EQ(reasonsOf(responses[1]), Strings{"consumer:text:cancelled"});
    EXPECT_TRUE(responses[0].ok) << responses[0].error;
    EXPECT_EQ(reasonsOf(responses[0]), Strings{"consumer:text:completed"});
    EXPECT_EQ(record("consumer").received[0], tokensOf(slots[0]->requestId, 4));
}

// Given a cohort of two that is run and taken, and then a second cohort of two,
// When each runs,
// Then responses come back in admission order, request ids keep counting (1, 2, then 3, 4) while batch slots restart
// at 0, 1, and each member receives the tokens written for its own request.
TEST_F(MultiStageRuntimeTest, ResponsesFollowAdmissionOrderAndTakeFreesTheSeats)
{
    MultiStageRuntime runtime;
    load(runtime, pipeline(json{{"count", 2}}, json::object(), 2));
    for (uint64_t const firstId : {uint64_t{1}, uint64_t{3}})
    {
        record("consumer").received.clear();
        auto const slots = seat(runtime, 2);
        ASSERT_TRUE(slots[0] != nullptr && slots[1] != nullptr);

        auto const responses = run(runtime, slots);

        ASSERT_EQ(responses.size(), 2U);
        for (int32_t member = 0; member < 2; ++member)
        {
            auto const& slot = *slots[static_cast<size_t>(member)];
            EXPECT_EQ(responses[static_cast<size_t>(member)].requestId, slot.requestId);
            EXPECT_EQ(slot.requestId, firstId + static_cast<uint64_t>(member));
            EXPECT_EQ(slot.batchSlot, member);
            EXPECT_EQ(record("consumer").received[member], tokensOf(slot.requestId, 2));
        }
        runtime.take();
    }
}

// Given two text-only requests seated on a topology that also has a vocoder, and never run,
// When take() is called,
// Then every stage, the vocoder off their route included, is reset once and none is finished, and two requests can
// be seated again.
TEST_F(MultiStageRuntimeTest, TakeWithoutRunResetsEveryStageAndSkipsFinish)
{
    MultiStageRuntime runtime;
    load(runtime, withVocoder(2));
    seat(runtime, 2);

    runtime.take();

    for (auto const* stage : {"producer", "consumer", "vocoder"})
    {
        EXPECT_EQ(record(stage).resets, 1) << stage;
        EXPECT_TRUE(record(stage).finishes.empty()) << stage;
    }
    std::string error;
    EXPECT_NE(submit(runtime, error), nullptr) << error;
    EXPECT_NE(submit(runtime, error), nullptr) << error;
}

// Given a topology whose consumer runner's factory builds no stage,
// When it is loaded,
// Then load() throws naming the stage and the runner that built nothing, the producer it had already built is
// destroyed, and the same runtime then loads with a table that builds both and serves a request.
TEST_F(MultiStageRuntimeTest, ARunnerThatBuildsNoStageFailsLoadByName)
{
    MultiStageRuntime runtime;
    std::string message;
    try
    {
        load(runtime, pipeline(), fakeRunners(&buildNothing));
    }
    catch (std::exception const& error)
    {
        message = error.what();
    }

    EXPECT_THAT(message, HasSubstr("stage 'consumer': "));
    EXPECT_THAT(message, HasSubstr("runner '" + std::string{kConsumerRunner} + "' built no stage"));
    EXPECT_EQ(record("producer").built, 1);
    EXPECT_EQ(record("producer").destroyed, 1);
    ASSERT_NO_THROW(load(runtime, pipeline()));
    auto const slots = seat(runtime, 1);
    auto const responses = run(runtime, slots);
    ASSERT_EQ(responses.size(), 1U);
    EXPECT_TRUE(responses[0].ok) << responses[0].error;
}

// Given a cohort of two whose consumer returns kFailed on the second member's first step,
// When the cohort runs,
// Then only that member fails, with "stage 'consumer': step failed" and kFailed from both stages, while the first
// completes with all its tokens.
TEST_F(MultiStageRuntimeTest, AStepThatFailsEndsOnlyThatMember)
{
    MultiStageRuntime runtime;
    load(runtime, pipeline(json{{"count", 3}}, json{{"fail_slot", 1}}, 2));
    auto const slots = seat(runtime, 2);

    auto const responses = run(runtime, slots);

    ASSERT_EQ(responses.size(), 2U);
    EXPECT_FALSE(responses[1].ok);
    EXPECT_EQ(responses[1].error, "stage 'consumer': step failed");
    EXPECT_EQ(reasonsOf(responses[1]), Strings{"consumer:text:failed"});
    EXPECT_EQ(record("producer").finishes[1], Strings{"failed"});
    EXPECT_EQ(record("consumer").finishes[1], Strings{"failed"});
    EXPECT_TRUE(responses[0].ok) << responses[0].error;
    EXPECT_EQ(record("consumer").received[0], tokensOf(slots[0]->requestId, 3));
}

// Given a cohort of two whose consumer, the stage holding the text terminal, throws in finish() for the first member,
// When the cohort runs,
// Then that member fails with an error naming the stage and its terminal reads kFailed, while the consumer still
// finishes the second member, which completes.
TEST_F(MultiStageRuntimeTest, AFinishThrowFailsOnlyThatMemberAndItsTerminalSaysSo)
{
    MultiStageRuntime runtime;
    load(runtime, pipeline(json{{"count", 3}}, json{{"throw_finish_slot", 0}}, 2));
    auto const slots = seat(runtime, 2);

    auto const responses = run(runtime, slots);

    ASSERT_EQ(responses.size(), 2U);
    EXPECT_FALSE(responses[0].ok);
    EXPECT_EQ(responses[0].error, "stage 'consumer': finish exploded");
    EXPECT_EQ(reasonsOf(responses[0]), Strings{"consumer:text:failed"});
    EXPECT_EQ(record("consumer").finishes[1], Strings{"completed"});
    EXPECT_TRUE(responses[1].ok) << responses[1].error;
    EXPECT_EQ(reasonsOf(responses[1]), Strings{"consumer:text:completed"});
}

// Given a cohort of two whose consumer's finish() throws an int rather than a std::exception for the first member,
// When the cohort runs,
// Then that member fails with "stage 'consumer': unknown exception in finish()", and the consumer still finishes the
// second member, which completes.
TEST_F(MultiStageRuntimeTest, ANonStdExceptionFromFinishFailsOnlyThatMember)
{
    MultiStageRuntime runtime;
    load(runtime, pipeline(json{{"count", 3}}, json{{"throw_finish_slot", 0}, {"throw_int", true}}, 2));
    auto const slots = seat(runtime, 2);

    auto const responses = run(runtime, slots);

    ASSERT_EQ(responses.size(), 2U);
    EXPECT_EQ(responses[0].error, "stage 'consumer': unknown exception in finish()");
    EXPECT_EQ(record("consumer").finishes[1], Strings{"completed"});
    EXPECT_TRUE(responses[1].ok) << responses[1].error;
}

// Given a topology whose producer and consumer batch two requests and whose vocoder, at max_batch 2, does not batch,
// When requests are seated with and without the audio sink,
// Then the vocoder is built one wide, two text-only requests share a cohort, and a second request that wants audio
// is refused as over a cohort limit of one: only the stages on a request's route cap its cohort.
TEST_F(MultiStageRuntimeTest, ACohortIsCappedOnlyByTheStagesOnItsRoute)
{
    MultiStageRuntime runtime;
    load(runtime, withVocoder(2));
    EXPECT_EQ(record("vocoder").maxBatch, 1);

    std::string error;
    EXPECT_NE(submit(runtime, error), nullptr) << error;
    EXPECT_NE(submit(runtime, error), nullptr) << error;
    runtime.take();
    EXPECT_NE(submit(runtime, error, /*audio=*/true), nullptr) << error;
    EXPECT_EQ(submit(runtime, error, /*audio=*/true), nullptr);
    EXPECT_THAT(error, HasSubstr("cohort is full at 1"));
}

// Given a cohort holding a request that wants text and audio,
// When a request that wants only text is submitted,
// Then it is refused for resolving a different route, although its own route would seat a second request.
TEST_F(MultiStageRuntimeTest, ARequestOnAnotherRouteCannotJoinTheCohort)
{
    MultiStageRuntime runtime;
    load(runtime, withVocoder(2));
    std::string error;
    ASSERT_NE(submit(runtime, error, /*audio=*/true), nullptr) << error;

    EXPECT_EQ(submit(runtime, error), nullptr);
    EXPECT_THAT(error, HasSubstr("resolves a different route"));
}

// Given a consumer that refuses the request a cohort would seat second,
// When two requests are submitted,
// Then the second is refused with "stage 'consumer': refused the request: busy" and takes no seat: the cohort runs
// the first alone.
TEST_F(MultiStageRuntimeTest, AnAdmissionRefusalNamesTheStageAndSeatsNothing)
{
    MultiStageRuntime runtime;
    load(runtime, pipeline(json::object(), json{{"refuse_slot", 1}}, 2));
    std::string error;
    auto* const first = submit(runtime, error);
    ASSERT_NE(first, nullptr) << error;

    EXPECT_EQ(submit(runtime, error), nullptr);
    EXPECT_EQ(error, "stage 'consumer': refused the request: busy");
    EXPECT_EQ(run(runtime, {first}).size(), 1U);
}

// Given a topology whose consumer holds the text terminal and whose vocoder holds the audio terminal,
// When a request with only a text sink runs,
// Then its response gives an outcome for the text terminal alone.
TEST_F(MultiStageRuntimeTest, ATerminalOffTheRouteGetsNoOutcome)
{
    MultiStageRuntime runtime;
    load(runtime, withVocoder(1));
    auto const slots = seat(runtime, 1);

    auto const responses = run(runtime, slots);

    ASSERT_EQ(responses.size(), 1U);
    EXPECT_EQ(reasonsOf(responses[0]), Strings{"consumer:text:completed"});
}

// Given a producer whose port is not a terminal, feeding a consumer that holds the text terminal and a vocoder that
// holds the audio terminal,
// When a request with both sinks runs,
// Then the consumer is handed the text sink alone, the vocoder the audio sink alone, and the producer neither.
TEST_F(MultiStageRuntimeTest, AStageIsHandedOnlyTheSinksOfTheTerminalsItEmits)
{
    MultiStageRuntime runtime;
    load(runtime, withVocoder(1));
    std::string error;
    auto* const slot = submit(runtime, error, /*audio=*/true);
    ASSERT_NE(slot, nullptr) << error;

    auto const responses = run(runtime, {slot});

    ASSERT_EQ(responses.size(), 1U);
    EXPECT_EQ(record("producer").sinks[0], Strings{});
    EXPECT_EQ(record("consumer").sinks[0], Strings{"text"});
    EXPECT_EQ(record("vocoder").sinks[0], Strings{"audio"});
}

// Given a cohort of two whose producer reports each member's request id and step count, whose consumer reports
// nothing, and whose second member fails at the consumer's first step,
// When the cohort runs,
// Then each member, the failed one included, carries the producer's report on its own request, and neither carries an
// entry for the consumer.
TEST_F(MultiStageRuntimeTest, EachMemberGetsItsOwnDiagnosticsAndEmptyOnesAreLeftOut)
{
    MultiStageRuntime runtime;
    load(runtime, pipeline(json{{"count", 2}, {"report", true}}, json{{"fail_slot", 1}}, 2));
    auto const slots = seat(runtime, 2);

    auto const responses = run(runtime, slots);

    ASSERT_EQ(responses.size(), 2U);
    EXPECT_EQ(responses[0].stageDiagnostics, (json{{"producer", {{"request", slots[0]->requestId}, {"steps", 2}}}}));
    EXPECT_FALSE(responses[1].ok);
    EXPECT_EQ(responses[1].stageDiagnostics, (json{{"producer", {{"request", slots[1]->requestId}, {"steps", 1}}}}));
}

// Given a cohort of two whose producer's diagnostics() throws for the second member, both stages otherwise reporting,
// When the cohort runs,
// Then run() returns, and only that member fails, with an error naming the stage and its terminal reading kFailed;
// it keeps the consumer's report, and the first member keeps both.
TEST_F(MultiStageRuntimeTest, ADiagnosticsThrowFailsOnlyThatMemberAndKeepsItsOtherReports)
{
    MultiStageRuntime runtime;
    load(runtime, pipeline(json{{"count", 2}, {"report", true}, {"throw_report_slot", 1}}, json{{"report", true}}, 2));
    auto const slots = seat(runtime, 2);

    auto const responses = run(runtime, slots);

    ASSERT_EQ(responses.size(), 2U);
    EXPECT_FALSE(responses[1].ok);
    EXPECT_EQ(responses[1].error, "stage 'producer': report exploded");
    EXPECT_EQ(reasonsOf(responses[1]), Strings{"consumer:text:failed"});
    EXPECT_EQ(responses[1].stageDiagnostics, (json{{"consumer", {{"request", slots[1]->requestId}, {"steps", 2}}}}));
    EXPECT_TRUE(responses[0].ok) << responses[0].error;
    EXPECT_EQ(responses[0].stageDiagnostics,
        (json{{"producer", {{"request", slots[0]->requestId}, {"steps", 2}}},
            {"consumer", {{"request", slots[0]->requestId}, {"steps", 2}}}}));
}

// Given a request routed through a producer, a consumer and a vocoder that all report, whose consumer's diagnostics()
// throws,
// When it runs,
// Then the error names the consumer, and the request keeps the producer's and the vocoder's reports.
TEST_F(MultiStageRuntimeTest, ADiagnosticsThrowKeepsTheReportsOfTheStagesAroundIt)
{
    MultiStageRuntime runtime;
    load(runtime,
        withVocoder(1, json{{"count", 2}, {"report", true}}, json{{"report", true}, {"throw_report_slot", 0}},
            json{{"report", true}}));
    std::string error;
    auto* const slot = submit(runtime, error, /*audio=*/true);
    ASSERT_NE(slot, nullptr) << error;

    auto const responses = run(runtime, {slot});

    ASSERT_EQ(responses.size(), 1U);
    EXPECT_EQ(responses[0].error, "stage 'consumer': report exploded");
    EXPECT_EQ(responses[0].stageDiagnostics,
        (json{{"producer", {{"request", slot->requestId}, {"steps", 2}}},
            {"vocoder", {{"request", slot->requestId}, {"steps", 2}}}}));
}

// Given a pipeline whose consumer's diagnostics() throws an int rather than a std::exception,
// When the request runs,
// Then run() returns, and the request fails with "stage 'consumer': unknown exception in diagnostics()".
TEST_F(MultiStageRuntimeTest, ANonStdExceptionFromDiagnosticsFailsTheRequestInsteadOfEscaping)
{
    MultiStageRuntime runtime;
    load(runtime, pipeline(json::object(), json{{"throw_report_slot", 0}, {"throw_int", true}}));
    auto const slots = seat(runtime, 1);

    auto const responses = run(runtime, slots);

    ASSERT_EQ(responses.size(), 1U);
    EXPECT_EQ(responses[0].error, "stage 'consumer': unknown exception in diagnostics()");
}
