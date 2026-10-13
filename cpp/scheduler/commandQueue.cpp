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

#include "common/checkMacros.h"
#include "common/mathUtils.h"

#include <algorithm>

namespace trt_edgellm
{
namespace rt
{
namespace scheduler
{
CommandQueue::CommandQueue(int32_t capacity)
    : mCapacity(std::max(kMinCapacity, math::ceilToPowerOfTwo(capacity)))
    , mMask(static_cast<uint64_t>(mCapacity) - 1)
    , mCells(static_cast<size_t>(mCapacity))
{
    check::check(capacity > 0, "CommandQueue capacity must be positive.");
    // Seeding cell i with sequence i means "writable at position i": a producer claiming position
    // p finds sequence == p exactly when the cell is free for that lap.
    for (int32_t index = 0; index < mCapacity; ++index)
    {
        mCells[static_cast<size_t>(index)].sequence.store(static_cast<uint64_t>(index), std::memory_order_relaxed);
    }
}

bool CommandQueue::push(std::unique_ptr<EngineCommand> command)
{
    check::check(command != nullptr, "CommandQueue cannot carry a null command.");

    // Loop until a position is claimed; discovering the ring is full ends the whole call instead.
    // Retries are unbounded by design, and bounding them would be a bug: under contention every
    // failed pass means another producer just claimed a position -- the system made progress even
    // though this thread must try again -- and an iteration cap would turn busy moments into
    // spurious "full" rejections.
    uint64_t position = mEnqueuePos.load(std::memory_order_relaxed);
    Cell* cell = nullptr;
    bool claimed = false;
    while (!claimed)
    {
        cell = &mCells[static_cast<size_t>(position & mMask)];
        uint64_t const sequence = cell->sequence.load(std::memory_order_acquire);
        auto const delta = static_cast<int64_t>(sequence) - static_cast<int64_t>(position);
        if (delta < 0)
        {
            // The cell still holds an unconsumed command from the previous lap: the ring is full.
            return false;
        }
        if (delta == 0)
        {
            // Free for this lap. Claiming the position and testing fullness are the same atomic
            // step, so two producers cannot both conclude there is room. On failure the CAS has
            // already refreshed `position` to the current enqueue position for the next pass.
            claimed = mEnqueuePos.compare_exchange_weak(
                position, position + 1, std::memory_order_relaxed, std::memory_order_relaxed);
        }
        else
        {
            // Another producer moved ahead; chase the new position.
            position = mEnqueuePos.load(std::memory_order_relaxed);
        }
    }

    cell->command = std::move(command);
    // Release publishes the command and everything it owns, including the record the caller
    // already holds, before the actor can observe the cell.
    cell->sequence.store(position + 1, std::memory_order_release);
    return true;
}

std::unique_ptr<EngineCommand> CommandQueue::pop()
{
    // Only the actor advances mDequeuePos, so no CAS is needed on this side.
    uint64_t const position = mDequeuePos.load(std::memory_order_relaxed);
    Cell& cell = mCells[static_cast<size_t>(position & mMask)];
    uint64_t const sequence = cell.sequence.load(std::memory_order_acquire);
    if (static_cast<int64_t>(sequence) - static_cast<int64_t>(position + 1) != 0)
    {
        return nullptr;
    }

    std::unique_ptr<EngineCommand> command = std::move(cell.command);
    mDequeuePos.store(position + 1, std::memory_order_relaxed);
    // Hand the cell to the producer that will claim this position one lap later.
    cell.sequence.store(position + static_cast<uint64_t>(mCapacity), std::memory_order_release);
    return command;
}

} // namespace scheduler
} // namespace rt
} // namespace trt_edgellm
