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

#pragma once

#include "scheduler/engineCommand.h"

#include <atomic>
#include <cstdint>
#include <memory>
#include <vector>

namespace trt_edgellm
{
namespace rt
{
namespace scheduler
{

//! Bounded multiple-producer, single-consumer queue carrying commands to the actor.
//!
//! Any thread may push; only the actor thread pops. The ring is preallocated and its cells are
//! reused, so unlike an intrusive linked-list MPSC there are no nodes to free, no ABA on node
//! pointers, and no need for hazard pointers or epochs — the reclamation problem that makes most
//! lock-free queues hard does not arise. The one heap object per command is allocated by the
//! producer and normally destroyed by the actor, which a single consumer makes safe. Two paths
//! destroy it elsewhere: a push onto a full ring destroys it on the producer, and cells own what
//! is still pending when the ring itself is destroyed.
//!
//! Bounded on purpose. A full queue makes push() fail rather than wait or grow, which hands the
//! caller a backpressure signal instead of silently absorbing load it cannot serve.
class CommandQueue
{
public:
    //! Smallest ring the sequence protocol can express. With a single cell, position 1 wraps onto
    //! the cell position 0 just published, whose sequence already reads as writable — a producer
    //! would overwrite an undelivered command and leave the consumer stuck a lap behind.
    static constexpr int32_t kMinCapacity = 2;

    //! @param capacity Ring slots, rounded up to a power of two and to at least kMinCapacity. Must
    //!        be positive.
    explicit CommandQueue(int32_t capacity);

    //! Cells own their commands, so undrained ones are freed with the ring itself. As everywhere
    //! on this class, destruction requires that all use of the queue happens-before it -- which the
    //! engine provides by owning its channel through a shared_ptr, whose final release synchronizes
    //! with every thread that used it.
    ~CommandQueue() = default;

    CommandQueue(CommandQueue const&) = delete;
    CommandQueue& operator=(CommandQueue const&) = delete;
    CommandQueue(CommandQueue&&) = delete;
    CommandQueue& operator=(CommandQueue&&) = delete;

    //! @brief Producer, any thread: hand a command to the actor.
    //! @return false when the ring is full, in which case the command is destroyed. Submission has
    //!         no retry path — a full queue rejects the request outright — so nothing is salvaged.
    bool push(std::unique_ptr<EngineCommand> command);

    //! @brief Consumer, actor thread only: take the next command, or nullptr when none is ready.
    //!        Not the same as "empty": a producer that has claimed the next position but not yet
    //!        published it holds the line, so a later command that already succeeded stays
    //!        invisible until that producer finishes. The gap is a few instructions wide, and the
    //!        actor re-drains every boundary, so a command is delayed by at most one step.
    std::unique_ptr<EngineCommand> pop();

    //! @brief Whether any command is waiting. The actor's idle park asks this before sleeping.
    //!
    //! Advisory: the counters may move between the two loads, and both wrong answers are cheap --
    //! a stale "no" is covered by the producer's wake and the park's timeout, a stale "yes" costs
    //! one empty pop.
    bool hasPending() const noexcept
    {
        return mEnqueuePos.load(std::memory_order_acquire) > mDequeuePos.load(std::memory_order_acquire);
    }

    int32_t capacity() const noexcept
    {
        return mCapacity;
    }

private:
    struct Cell
    {
        //! Sequence number deciding whether this cell is writable, readable, or neither. A producer
        //! that finds it behind the position it claimed knows the ring is full.
        std::atomic<uint64_t> sequence;

        //! Owned. The pointer itself is not atomic and needs no protection: the sequence protocol
        //! grants each cell to exactly one thread at a time, and its release/acquire pair is the
        //! happens-before edge that publishes this field along with everything the command owns.
        std::unique_ptr<EngineCommand> command;
    };

    int32_t const mCapacity;
    uint64_t const mMask;
    std::vector<Cell> mCells;

    //! Claimed by producers with a CAS. Contended; the cell sequence does the publishing.
    std::atomic<uint64_t> mEnqueuePos{0};
    //! Advanced by the actor alone, so it needs no CAS.
    std::atomic<uint64_t> mDequeuePos{0};
};

} // namespace scheduler
} // namespace rt
} // namespace trt_edgellm
