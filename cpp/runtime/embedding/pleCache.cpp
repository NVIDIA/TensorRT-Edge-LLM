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

#include "runtime/embedding/pleCache.h"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>

namespace trt_edgellm
{
namespace rt
{

PleCache::PleCache(RowId rowCount, SlotId slotCount)
    : mRowCount(rowCount)
{
    if (rowCount <= 0 || slotCount <= 0)
    {
        throw std::invalid_argument("PLE row count and slot count must be positive");
    }
    mSlots.resize(slotCount);
    mSlots[0].state = State::kREADY;
}

PleCache::BatchReservation PleCache::reserveBatch(std::vector<std::vector<RowId>> const& requests)
{
    if (mActiveBatch)
    {
        throw std::logic_error("PLE cache already has an active batch");
    }
    if (requests.empty())
    {
        throw std::invalid_argument("PLE batch must contain a request");
    }
    std::vector<RowId> rows;
    for (auto const& request : requests)
    {
        for (RowId row : request)
        {
            if (row < -1 || row >= mRowCount)
            {
                throw std::out_of_range("PLE row is outside the table");
            }
            if (row != -1)
            {
                rows.push_back(row);
            }
        }
    }
    std::sort(rows.begin(), rows.end());
    rows.erase(std::unique(rows.begin(), rows.end()), rows.end());
    if (rows.size() > mSlots.size() - 1)
    {
        throw std::length_error("PLE execution batch requires " + std::to_string(rows.size())
            + " distinct valid rows, but the cache has " + std::to_string(mSlots.size() - 1)
            + " usable slots (slot 0 is reserved). Increase PLE cache slots or reduce the execution chunk/batch; "
              "smaller I/O batches do not reduce the required residency.");
    }

    BatchReservation batch(*this);
    batch.mMisses.reserve(rows.size());
    batch.mProtected.reserve(rows.size());
    batch.mRequests.reserve(requests.size());
    mActiveBatch = true;
    mReservationActive = true;
    // Protect the entire union before CLOCK can evict a hit belonging to a later request.
    for (RowId row : rows)
    {
        auto const it = mRows.find(row);
        if (it != mRows.end())
        {
            Slot& slot = mSlots[it->second];
            ++slot.references;
            slot.referenced = true;
            batch.mProtected.push_back(it->second);
        }
    }
    for (RowId row : rows)
    {
        if (mRows.find(row) == mRows.end())
        {
            SlotId const id = claimSlot();
            mSlots[id] = {row, 1, State::kFILLING, true};
            batch.mMisses.push_back({row, id});
            batch.mProtected.push_back(id);
            mRows.emplace(row, id);
        }
    }
    for (auto const& request : requests)
    {
        std::vector<SlotId> mapping;
        mapping.reserve(request.size());
        for (RowId row : request)
        {
            mapping.push_back(row == -1 ? 0 : mRows.at(row));
        }
        std::vector<SlotId> pinned = mapping;
        std::sort(pinned.begin(), pinned.end());
        pinned.erase(std::unique(pinned.begin(), pinned.end()), pinned.end());
        if (!pinned.empty() && pinned.front() == 0)
        {
            pinned.erase(pinned.begin());
        }
        batch.mRequests.push_back(RequestLease(*this, std::move(mapping), std::move(pinned)));
    }
    return batch;
}

PleCache::SlotId PleCache::claimSlot()
{
    int64_t const usable = static_cast<int64_t>(mSlots.size()) - 1;
    for (int64_t visited = 0; visited < 2 * usable; ++visited)
    {
        SlotId const id = mClock;
        mClock = mClock == usable ? 1 : mClock + 1;
        Slot& slot = mSlots[id];
        if (slot.references != 0)
        {
            continue;
        }
        if (slot.state == State::kFREE)
        {
            return id;
        }
        if (slot.referenced)
        {
            slot.referenced = false;
            continue;
        }
        mRows.erase(slot.row);
        slot = {};
        return id;
    }
    throw std::logic_error("PLE CLOCK found no unpinned slot after admission");
}

void PleCache::unpin(std::vector<SlotId> const& slots) noexcept
{
    for (SlotId id : slots)
    {
        --mSlots[id].references;
    }
}

void PleCache::releaseRequest(std::vector<SlotId> const& slots) noexcept
{
    unpin(slots);
    --mLiveRequests;
    if (mLiveRequests == 0 && !mReservationActive)
    {
        mActiveBatch = false;
    }
}

PleCache::RequestLease::RequestLease(PleCache& owner, std::vector<SlotId> mapping, std::vector<SlotId> pinned)
    : mOwner(&owner)
    , mMapping(std::move(mapping))
    , mPinned(std::move(pinned))
{
    for (SlotId id : mPinned)
    {
        ++mOwner->mSlots[id].references;
    }
    ++mOwner->mLiveRequests;
}

PleCache::RequestLease::RequestLease(RequestLease&& other) noexcept
    : mOwner(std::exchange(other.mOwner, nullptr))
    , mMapping(std::move(other.mMapping))
    , mPinned(std::move(other.mPinned))
{
}

PleCache::RequestLease& PleCache::RequestLease::operator=(RequestLease&& other) noexcept
{
    if (this != &other)
    {
        release();
        mOwner = std::exchange(other.mOwner, nullptr);
        mMapping = std::move(other.mMapping);
        mPinned = std::move(other.mPinned);
    }
    return *this;
}

PleCache::RequestLease::~RequestLease()
{
    release();
}

std::vector<PleCache::SlotId> const& PleCache::RequestLease::mapping() const
{
    if (mOwner == nullptr)
    {
        throw std::logic_error("PLE request lease has been released");
    }
    return mMapping;
}

void PleCache::RequestLease::release() noexcept
{
    if (mOwner != nullptr)
    {
        mOwner->releaseRequest(mPinned);
        mOwner = nullptr;
    }
}

PleCache::BatchReservation::BatchReservation(PleCache& owner)
    : mOwner(&owner)
{
}

PleCache::BatchReservation::BatchReservation(BatchReservation&& other) noexcept
    : mOwner(std::exchange(other.mOwner, nullptr))
    , mMisses(std::move(other.mMisses))
    , mProtected(std::move(other.mProtected))
    , mRequests(std::move(other.mRequests))
{
}

PleCache::BatchReservation& PleCache::BatchReservation::operator=(BatchReservation&& other) noexcept
{
    if (this != &other)
    {
        rollback();
        mOwner = std::exchange(other.mOwner, nullptr);
        mMisses = std::move(other.mMisses);
        mProtected = std::move(other.mProtected);
        mRequests = std::move(other.mRequests);
    }
    return *this;
}

PleCache::BatchReservation::~BatchReservation()
{
    rollback();
}

std::vector<PleCache::Miss> const& PleCache::BatchReservation::misses() const noexcept
{
    return mMisses;
}

void PleCache::BatchReservation::markReady(SlotId id)
{
    if (mOwner == nullptr || id <= 0 || static_cast<size_t>(id) >= mOwner->mSlots.size())
    {
        throw std::logic_error("PLE slot is not reserved by this batch");
    }
    auto& slot = mOwner->mSlots[id];
    if (slot.state != State::kFILLING)
    {
        throw std::logic_error("PLE slot is not filling");
    }
    slot.state = State::kREADY;
}

std::vector<PleCache::RequestLease> PleCache::BatchReservation::commit()
{
    if (mOwner == nullptr)
    {
        throw std::logic_error("PLE reservation is no longer active");
    }
    for (auto const& miss : mMisses)
    {
        if (mOwner->mSlots[miss.slot].state != State::kREADY)
        {
            throw std::logic_error("PLE batch has incomplete fills");
        }
    }
    mOwner->unpin(mProtected);
    mOwner->mReservationActive = false;
    mOwner = nullptr;
    return std::move(mRequests);
}

void PleCache::BatchReservation::rollback() noexcept
{
    if (mOwner != nullptr)
    {
        mRequests.clear();
        mOwner->unpin(mProtected);
        for (auto const& miss : mMisses)
        {
            mOwner->mRows.erase(miss.row);
            mOwner->mSlots[miss.slot] = {};
        }
        mOwner->mReservationActive = false;
        mOwner->mActiveBatch = false;
        mOwner = nullptr;
    }
}

} // namespace rt
} // namespace trt_edgellm
