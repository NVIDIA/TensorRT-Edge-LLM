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

#include "runtime/state/residentSlotPool.h"

#include "common/checkMacros.h"

#include <limits>

namespace trt_edgellm::rt
{

ResidentSlotPool::ResidentSlotPool(int32_t capacity)
{
    reset(capacity);
}

void ResidentSlotPool::reset(int32_t capacity)
{
    ELLM_CHECK(capacity >= 0, "Resident slot capacity must not be negative");
    mEpochs.assign(static_cast<size_t>(capacity), 1);
    mOccupied.assign(static_cast<size_t>(capacity), 0);
    mFreeSlots.resize(static_cast<size_t>(capacity));
    for (int32_t slot = 0; slot < capacity; ++slot)
    {
        mFreeSlots[static_cast<size_t>(capacity - slot - 1)] = slot;
    }
}

std::optional<ResidentRef> ResidentSlotPool::acquire()
{
    if (mFreeSlots.empty())
    {
        return std::nullopt;
    }
    CacheSlot const slot = mFreeSlots.back();
    mFreeSlots.pop_back();
    mOccupied[static_cast<size_t>(slot)] = 1;
    return ResidentRef{slot, mEpochs[static_cast<size_t>(slot)]};
}

bool ResidentSlotPool::release(ResidentRef resident)
{
    if (!contains(resident))
    {
        return false;
    }
    size_t const index = static_cast<size_t>(resident.slot);
    ELLM_CHECK(mEpochs[index] != std::numeric_limits<uint64_t>::max(), "Resident slot epoch overflow");
    mOccupied[index] = 0;
    ++mEpochs[index];
    mFreeSlots.push_back(resident.slot);
    return true;
}

bool ResidentSlotPool::contains(ResidentRef resident) const noexcept
{
    return resident.slot >= 0 && static_cast<size_t>(resident.slot) < mEpochs.size()
        && mOccupied[static_cast<size_t>(resident.slot)] != 0
        && mEpochs[static_cast<size_t>(resident.slot)] == resident.epoch;
}

int32_t ResidentSlotPool::capacity() const noexcept
{
    return static_cast<int32_t>(mEpochs.size());
}

int32_t ResidentSlotPool::available() const noexcept
{
    return static_cast<int32_t>(mFreeSlots.size());
}

} // namespace trt_edgellm::rt
