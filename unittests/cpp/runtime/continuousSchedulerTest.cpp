/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "runtime/continuousScheduler.h"
#include "runtime/state/prefillChunk.h"
#include <chrono>
#include <gtest/gtest.h>

using namespace trt_edgellm::rt;
using namespace std::chrono_literals;
namespace
{
class FakeBackend : public SchedulerBackend
{
public:
    SequenceSlots slots{2, 6144, 8192};
    std::shared_future<void> gate;
    bool fail{};
    void start() override
    {
        if (gate.valid())
        {
            gate.wait();
        }
    }
    SequenceHandle acquire(uint64_t id, std::vector<int32_t> prompt, SequenceOptions options) override
    {
        return slots.acquire(id, std::move(prompt), options);
    }
    SequenceState const& state(SequenceHandle h) const override
    {
        return slots.get(h);
    }
    void prefill(SequenceHandle h) override
    {
        if (fail)
        {
            throw std::runtime_error("injected forward failure");
        }
        auto const& s = state(h);
        slots.commitPrompt(h, nextPrefillChunkSize(static_cast<int32_t>(s.prompt().size()) - s.promptCursor()));
        if (state(h).phase() == SequencePhase::kAwaitingSample)
        {
            slots.acceptToken(h, 7);
        }
    }
    void decode(std::array<SequenceHandle, 2> const& handles, int32_t count) override
    {
        if (count == 2 && handles[0].slot >= handles[1].slot)
        {
            throw std::runtime_error("unordered views");
        }
        for (int i = 0; i < count; ++i)
        {
            slots.commitDecode(handles[i]);
            slots.acceptToken(handles[i], 7);
        }
    }
    void release(SequenceHandle h) override
    {
        slots.finish(h);
        slots.release(h);
    }
};
SchedulerResult result(SchedulerTicket const& ticket)
{
    if (ticket.result().wait_for(3s) != std::future_status::ready)
    {
        throw std::runtime_error("Stranded ticket");
    }
    return ticket.result().get();
}
} // namespace
TEST(ContinuousScheduler, StaggeredReuseAndDecodeFirst)
{
    std::vector<SchedulerEvent> events;
    ContinuousScheduler* owner = nullptr;
    SchedulerTicket b, c;
    bool submitted = false;
    ContinuousScheduler scheduler(std::make_unique<FakeBackend>(), 8, 65536, [&](auto const& e) {
        events.push_back(e);
        if (e.kind == SchedulerEvent::Kind::kDecode && !submitted)
        {
            submitted = true;
            b = owner->submit(std::vector<int32_t>(1025, 1), 12);
            c = owner->submit(std::vector<int32_t>(129, 2), 12);
        }
    });
    owner = &scheduler;
    auto a = scheduler.submit({1}, 6);
    EXPECT_EQ(result(a).tokens.size(), 6U);
    // A completion synchronizes the callback's ticket publication.
    EXPECT_EQ(result(b).tokens.size(), 12U);
    EXPECT_EQ(result(c).tokens.size(), 12U);
    scheduler.close();
    SequenceHandle ah{}, ch{};
    bool bContinued = false, cAdmitted = false, sawPair = false;
    for (auto const& e : events)
    {
        if (e.kind == SchedulerEvent::Kind::kAdmit && e.request == a.id())
        {
            ah = e.handle;
        }
        if (e.kind == SchedulerEvent::Kind::kAdmit && e.request == c.id())
        {
            ch = e.handle;
            cAdmitted = true;
        }
        if (cAdmitted && e.request == b.id() && e.kind == SchedulerEvent::Kind::kPrefill)
        {
            bContinued = true;
        }
        if (e.kind == SchedulerEvent::Kind::kDecode && e.partner)
        {
            sawPair = true;
        }
    }
    EXPECT_EQ(ah.slot, ch.slot);
    EXPECT_GT(ch.generation, ah.generation);
    EXPECT_TRUE(bContinued);
    EXPECT_TRUE(sawPair);
    EXPECT_TRUE(scheduler.healthy());
}
TEST(ContinuousScheduler, QueueBoundsCancellationAndShutdown)
{
    std::promise<void> gate;
    auto backend = std::make_unique<FakeBackend>();
    backend->gate = gate.get_future().share();
    ContinuousScheduler scheduler(std::move(backend), 2, 8);
    auto a = scheduler.submit({1}, 3);
    auto b = scheduler.submit({2}, 3);
    EXPECT_THROW(scheduler.submit({3}, 3), std::runtime_error);
    a.cancel();
    gate.set_value();
    EXPECT_EQ(result(a).status, SchedulerStatus::kCancelled);
    EXPECT_EQ(result(b).status, SchedulerStatus::kCompleted);
    a.cancel();
    auto c = scheduler.submit({4}, 3);
    EXPECT_EQ(result(c).status, SchedulerStatus::kCompleted);
    scheduler.close();
    scheduler.close();
    EXPECT_THROW(scheduler.submit({1}, 1), std::runtime_error);
}
TEST(ContinuousScheduler, ByteBoundAndValidation)
{
    std::promise<void> gate;
    auto backend = std::make_unique<FakeBackend>();
    backend->gate = gate.get_future().share();
    ContinuousScheduler scheduler(std::move(backend), 10, 4);
    auto a = scheduler.submit({1}, 1);
    EXPECT_THROW(scheduler.submit({1}, 1), std::runtime_error);
    EXPECT_THROW(scheduler.submit({}, 1), std::invalid_argument);
    EXPECT_THROW(scheduler.submit({-1}, 1), std::invalid_argument);
    EXPECT_THROW(scheduler.submit({1}, 8192), std::invalid_argument);
    gate.set_value();
    EXPECT_EQ(result(a).tokens.size(), 1U);
}
TEST(ContinuousScheduler, FaultSettlesActiveAndQueued)
{
    std::promise<void> gate;
    auto backend = std::make_unique<FakeBackend>();
    backend->gate = gate.get_future().share();
    backend->fail = true;
    ContinuousScheduler scheduler(std::move(backend));
    auto a = scheduler.submit({1}, 3);
    auto b = scheduler.submit({1}, 3);
    auto c = scheduler.submit({1}, 3);
    gate.set_value();
    for (auto const& ticket : {a, b, c})
    {
        EXPECT_EQ(result(ticket).status, SchedulerStatus::kFailed);
    }
    EXPECT_FALSE(scheduler.healthy());
    EXPECT_THROW(scheduler.submit({1}, 1), std::runtime_error);
}
TEST(ContinuousScheduler, IdleWaitAndWake)
{
    std::atomic<int> idle{0};
    ContinuousScheduler scheduler(std::make_unique<FakeBackend>(), 8, 65536, [&](auto const& e) {
        if (e.kind == SchedulerEvent::Kind::kIdle)
        {
            ++idle;
        }
    });
    std::this_thread::sleep_for(30ms);
    EXPECT_EQ(idle.load(), 1);
    std::this_thread::sleep_for(30ms);
    EXPECT_EQ(idle.load(), 1);
    EXPECT_EQ(result(scheduler.submit({1}, 1)).status, SchedulerStatus::kCompleted);
    scheduler.close();
}
TEST(ContinuousScheduler, TwoPrefillsFairnessAndActiveCancellation)
{
    std::promise<void> gate;
    auto backend = std::make_unique<FakeBackend>();
    backend->gate = gate.get_future().share();
    SchedulerTicket a;
    std::vector<uint64_t> prefills;
    ContinuousScheduler scheduler(std::move(backend), 8, 65536, [&](auto const& e) {
        if (e.kind == SchedulerEvent::Kind::kPrefill)
        {
            prefills.push_back(e.request);
            if (prefills.size() == 3)
            {
                a.cancel();
            }
        }
    });
    a = scheduler.submit(std::vector<int32_t>(1025, 1), 4);
    auto b = scheduler.submit(std::vector<int32_t>(1025, 1), 4);
    gate.set_value();
    EXPECT_EQ(result(a).status, SchedulerStatus::kCancelled);
    EXPECT_EQ(result(b).status, SchedulerStatus::kCompleted);
    scheduler.close();
    ASSERT_GE(prefills.size(), 3U);
    EXPECT_EQ(prefills[0], a.id());
    EXPECT_EQ(prefills[1], b.id());
    EXPECT_EQ(prefills[2], a.id());
}
TEST(ContinuousScheduler, CloseSettlesAllTickets)
{
    std::promise<void> gate;
    auto backend = std::make_unique<FakeBackend>();
    backend->gate = gate.get_future().share();
    ContinuousScheduler scheduler(std::move(backend));
    auto a = scheduler.submit({1}, 3);
    auto b = scheduler.submit({1}, 3);
    auto closing = std::async(std::launch::async, [&] { scheduler.close(); });
    gate.set_value();
    closing.get();
    EXPECT_NE(result(a).status, SchedulerStatus::kFailed);
    EXPECT_NE(result(b).status, SchedulerStatus::kFailed);
}

TEST(ContinuousScheduler, CancelledQueueHeadDoesNotDelayAdmission)
{
    std::promise<void> gate;
    auto backend = std::make_unique<FakeBackend>();
    backend->gate = gate.get_future().share();
    std::vector<SchedulerEvent> events;
    ContinuousScheduler scheduler(std::move(backend), 8, 65536, [&](auto const& e) { events.push_back(e); });
    auto cancelled = scheduler.submit({1}, 3);
    auto a = scheduler.submit({1}, 3);
    auto b = scheduler.submit({1}, 3);
    cancelled.cancel();
    gate.set_value();
    EXPECT_EQ(result(cancelled).status, SchedulerStatus::kCancelled);
    EXPECT_EQ(result(a).status, SchedulerStatus::kCompleted);
    EXPECT_EQ(result(b).status, SchedulerStatus::kCompleted);
    scheduler.close();
    int admissions = 0;
    for (auto const& e : events)
    {
        if (e.kind == SchedulerEvent::Kind::kAdmit)
        {
            ++admissions;
        }
        if (e.kind == SchedulerEvent::Kind::kPrefill)
        {
            EXPECT_EQ(admissions, 2);
            break;
        }
    }
}
TEST(ContinuousScheduler, ConcurrentProducersAndRepeatedReuse)
{
    ContinuousScheduler scheduler(std::make_unique<FakeBackend>(), 128, 65536);
    std::array<std::future<void>, 4> producers;
    for (auto& producer : producers)
    {
        producer = std::async(std::launch::async, [&] {
            for (int i = 0; i < 25; ++i)
            {
                auto ticket = scheduler.submit({1, 2, 3}, 3);
                if (result(ticket).tokens.size() != 3)
                {
                    throw std::runtime_error("Lost result");
                }
                ticket.cancel();
            }
        });
    }
    for (auto& producer : producers)
    {
        producer.get();
    }
    scheduler.close();
    EXPECT_TRUE(scheduler.healthy());
}

TEST(ContinuousScheduler, SlowConsumerDoesNotBlockPartner)
{
    std::promise<void> gate;
    auto backend = std::make_unique<FakeBackend>();
    backend->gate = gate.get_future().share();
    ContinuousScheduler scheduler(std::move(backend));
    SchedulerRequestOptions options;
    options.generation.maxOutputTokens = 10;
    options.streamRecords = 1;
    options.streamBytes = 8;
    auto slow = scheduler.submit({1}, options);
    auto peer = scheduler.submit({1}, 8);
    gate.set_value();
    EXPECT_EQ(result(slow).status, SchedulerStatus::kSlowConsumer);
    EXPECT_EQ(result(peer).tokens.size(), 8U);
    auto read = slow.read(0ms);
    ASSERT_TRUE(read.update);
    EXPECT_TRUE(read.closed);
    EXPECT_TRUE(scheduler.healthy());
}
TEST(ContinuousScheduler, QueuedCancellationWhileBothSlotsOccupied)
{
    ContinuousScheduler* owner = nullptr;
    SchedulerTicket queued;
    std::atomic<bool> cancelledReady{false};
    bool submitted = false;
    ContinuousScheduler scheduler(std::make_unique<FakeBackend>(), 8, 65536, [&](auto const& e) {
        if (e.kind == SchedulerEvent::Kind::kDecode && !submitted)
        {
            submitted = true;
            queued = owner->submit({1}, 3);
            queued.cancel();
        }
        else if (submitted && e.kind == SchedulerEvent::Kind::kDecode)
        {
            cancelledReady = queued.result().wait_for(0ms) == std::future_status::ready;
        }
    });
    owner = &scheduler;
    auto a = scheduler.submit({1}, 20);
    auto b = scheduler.submit({1}, 20);
    result(a);
    result(b);
    scheduler.close();
    EXPECT_EQ(result(queued).status, SchedulerStatus::kCancelled);
    EXPECT_TRUE(cancelledReady);
}
TEST(ContinuousScheduler, QueueAndActiveDeadlines)
{
    std::promise<void> gate;
    auto backend = std::make_unique<FakeBackend>();
    backend->gate = gate.get_future().share();
    SchedulerRequestOptions options;
    options.generation.maxOutputTokens = 5;
    options.queueDeadline = std::chrono::steady_clock::now() - 1ms;
    ContinuousScheduler scheduler(std::move(backend));
    auto expired = scheduler.submit({1}, options);
    gate.set_value();
    EXPECT_EQ(result(expired).status, SchedulerStatus::kDeadline);
    scheduler.close();
    SchedulerRequestOptions active;
    active.generation.maxOutputTokens = 100;
    active.deadline = std::chrono::steady_clock::now() + 20ms;
    bool delayed = false;
    ContinuousScheduler other(std::make_unique<FakeBackend>(), 8, 65536, [&](auto const& e) {
        if (e.kind == SchedulerEvent::Kind::kPrefill && !delayed)
        {
            delayed = true;
            std::this_thread::sleep_for(30ms);
        }
    });
    auto a = other.submit({1}, active);
    EXPECT_EQ(result(a).status, SchedulerStatus::kDeadline);
    EXPECT_EQ(result(other.submit({1}, 3)).status, SchedulerStatus::kCompleted);
}
TEST(ContinuousScheduler, DecodeCancellationAndStartupFailure)
{
    SchedulerTicket a;
    std::promise<void> gate;
    auto backend = std::make_unique<FakeBackend>();
    backend->gate = gate.get_future().share();
    ContinuousScheduler scheduler(std::move(backend), 8, 65536, [&](auto const& e) {
        if (e.kind == SchedulerEvent::Kind::kDecode && e.request == a.id())
        {
            a.cancel();
        }
    });
    a = scheduler.submit({1}, 10);
    auto b = scheduler.submit({1}, 12);
    gate.set_value();
    EXPECT_EQ(result(a).status, SchedulerStatus::kCancelled);
    EXPECT_EQ(result(b).tokens.size(), 12U);
    class BrokenStart : public FakeBackend
    {
        void start() override
        {
            throw std::runtime_error("startup failure");
        }
    };
    ContinuousScheduler broken(std::make_unique<BrokenStart>());
    std::this_thread::sleep_for(10ms);
    EXPECT_FALSE(broken.healthy());
    EXPECT_THROW(broken.submit({1}, 1), std::runtime_error);
}
