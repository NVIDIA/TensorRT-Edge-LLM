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

#include "scheduler/commandQueue.h"
#include "scheduler/requestRecord.h"

#include <gtest/gtest.h>

#include <atomic>
#include <thread>
#include <unordered_set>
#include <vector>

using namespace trt_edgellm::rt;
using namespace trt_edgellm::rt::scheduler;

namespace
{

std::unique_ptr<EngineCommand> makeSubmit(RequestId id)
{
    auto command = std::make_unique<EngineCommand>();
    command->type = CommandType::kSubmit;
    command->requestId = id;
    return command;
}

} // namespace

TEST(CommandQueueTests, CapacityIsNormalizedAtConstruction)
{
    // Rounded up to a power of two for the position masks.
    EXPECT_EQ(CommandQueue(3).capacity(), 4);
    EXPECT_EQ(CommandQueue(16).capacity(), 16);
    EXPECT_THROW(CommandQueue(0), std::runtime_error);
    EXPECT_THROW(CommandQueue(-2), std::runtime_error);

    // A one-cell ring cannot express this sequence protocol: position 1 wraps onto the cell
    // position 0 just published, whose sequence already reads as writable. A producer would
    // overwrite an undelivered command and strand the consumer a lap behind, so pop() would return
    // null forever. Raising the capacity is what keeps that unreachable.
    CommandQueue queue(1);
    ASSERT_EQ(queue.capacity(), CommandQueue::kMinCapacity);
    ASSERT_TRUE(queue.push(makeSubmit(1)));
    ASSERT_TRUE(queue.push(makeSubmit(2)));
    EXPECT_FALSE(queue.push(makeSubmit(3)));
    auto first = queue.pop();
    ASSERT_NE(first, nullptr);
    EXPECT_EQ(first->requestId, 1U);
    auto second = queue.pop();
    ASSERT_NE(second, nullptr);
    EXPECT_EQ(second->requestId, 2U);
    EXPECT_EQ(queue.pop(), nullptr);
}

TEST(CommandQueueTests, RejectedPushDestroysTheCommand)
{
    // push() takes the command by value and there is no retry path, so a full ring must not leak
    // it. Hanging a shared_ptr off the command turns "was it destroyed" into an observable count.
    CommandQueue queue(2);
    auto record = std::make_shared<RequestRecord>();
    ASSERT_EQ(record.use_count(), 1);

    ASSERT_TRUE(queue.push(makeSubmit(1)));
    ASSERT_TRUE(queue.push(makeSubmit(2)));

    auto rejected = makeSubmit(3);
    rejected->record = record;
    ASSERT_EQ(record.use_count(), 2);
    EXPECT_FALSE(queue.push(std::move(rejected)));
    EXPECT_EQ(record.use_count(), 1) << "a rejected command was leaked instead of destroyed";
}

TEST(CommandQueueTests, PushFailsWhenFullAndAcceptsAgainAfterDrain)
{
    CommandQueue queue(2);
    ASSERT_TRUE(queue.push(makeSubmit(1)));
    ASSERT_TRUE(queue.push(makeSubmit(2)));
    // A bounded queue is the backpressure signal: the push reports back rather than growing or
    // blocking, and submit() turns that into a rejection.
    EXPECT_FALSE(queue.push(makeSubmit(3)));

    ASSERT_NE(queue.pop(), nullptr);
    EXPECT_TRUE(queue.push(makeSubmit(3)));
}

TEST(CommandQueueTests, PopsInPushOrderAcrossWraps)
{
    // A full ring drains in push order, and the order survives the positions lapping the ring
    // many times over -- the wrap is where a sequence-protocol bug would scramble or drop.
    CommandQueue queue(2);
    ASSERT_TRUE(queue.push(makeSubmit(1)));
    ASSERT_TRUE(queue.push(makeSubmit(2)));
    for (RequestId id = 1; id <= 200; ++id)
    {
        auto command = queue.pop();
        ASSERT_NE(command, nullptr);
        EXPECT_EQ(command->requestId, id);
        if (id + 2 <= 200)
        {
            ASSERT_TRUE(queue.push(makeSubmit(id + 2)));
        }
    }
    EXPECT_EQ(queue.pop(), nullptr);
}

TEST(CommandQueueTests, DestructorReleasesUndrainedCommands)
{
    // Nothing to assert beyond the absence of a leak; ASAN/LSAN in CI is the real check, and the
    // test documents that abandoning a queue with pending commands is legal.
    auto queue = std::make_unique<CommandQueue>(8);
    for (RequestId id = 1; id <= 5; ++id)
    {
        ASSERT_TRUE(queue->push(makeSubmit(id)));
    }
    EXPECT_TRUE(queue->hasPending());
    queue.reset();
}

TEST(CommandQueueTests, CarriesEveryCommandType)
{
    CommandQueue queue(4);

    auto submit = std::make_unique<EngineCommand>();
    submit->type = CommandType::kSubmit;
    submit->requestId = 7;
    ASSERT_TRUE(queue.push(std::move(submit)));

    auto shutdown = std::make_unique<EngineCommand>();
    shutdown->type = CommandType::kShutdown;
    ASSERT_TRUE(queue.push(std::move(shutdown)));

    auto first = queue.pop();
    ASSERT_NE(first, nullptr);
    EXPECT_EQ(first->type, CommandType::kSubmit);
    EXPECT_EQ(first->requestId, 7U);

    auto second = queue.pop();
    ASSERT_NE(second, nullptr);
    EXPECT_EQ(second->type, CommandType::kShutdown);
}

TEST(CommandQueueTests, ManyProducersLoseNoCommandAndDuplicateNone)
{
    constexpr int32_t kProducers = 8;
    constexpr int32_t kPerProducer = 4000;
    constexpr int32_t kTotal = kProducers * kPerProducer;

    // Deliberately smaller than the total so producers hit the full path and retry.
    CommandQueue queue(64);
    std::atomic<int32_t> produced{0};

    std::vector<std::thread> producers;
    producers.reserve(kProducers);
    for (int32_t producer = 0; producer < kProducers; ++producer)
    {
        producers.emplace_back([&queue, &produced, producer] {
            // Ids are unique across producers so a duplicate or a loss is detectable.
            RequestId const base = static_cast<RequestId>(producer) * kPerProducer + 1;
            for (int32_t offset = 0; offset < kPerProducer;)
            {
                if (queue.push(makeSubmit(base + static_cast<RequestId>(offset))))
                {
                    ++offset;
                    produced.fetch_add(1, std::memory_order_relaxed);
                }
                else
                {
                    std::this_thread::yield();
                }
            }
        });
    }

    std::unordered_set<RequestId> seen;
    seen.reserve(static_cast<size_t>(kTotal));
    // Last id delivered for each producer. A ring that reordered within one producer's stream would
    // show up here; nothing is promised about the interleaving across producers.
    std::vector<RequestId> lastPerProducer(static_cast<size_t>(kProducers), 0);
    while (static_cast<int32_t>(seen.size()) < kTotal)
    {
        if (auto command = queue.pop())
        {
            RequestId const id = command->requestId;
            EXPECT_TRUE(seen.insert(id).second) << "request id " << id << " was delivered twice";

            auto const producer = static_cast<size_t>((id - 1) / kPerProducer);
            ASSERT_LT(producer, lastPerProducer.size());
            EXPECT_GT(id, lastPerProducer[producer])
                << "producer " << producer << " had id " << id << " delivered after " << lastPerProducer[producer];
            lastPerProducer[producer] = id;
        }
        else
        {
            std::this_thread::yield();
        }
    }

    for (auto& producer : producers)
    {
        producer.join();
    }

    EXPECT_EQ(produced.load(std::memory_order_relaxed), kTotal);
    EXPECT_EQ(static_cast<int32_t>(seen.size()), kTotal);
    EXPECT_EQ(queue.pop(), nullptr);
}
