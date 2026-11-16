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

#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace trt_edgellm
{
namespace rt
{

//! CPU metadata for one immutable PLE table. The owner must outlive every reservation and request lease.
//! Only one batch may be active. Slot zero is permanently reserved for padding; it is not a table row.
class PleCache
{
public:
    using RowId = int64_t;
    using SlotId = int32_t;

    struct Miss
    {
        RowId row;
        SlotId slot;
    };

    class BatchReservation;

    //! Pins one request's distinct rows. Release only after its engine work has completed.
    class RequestLease
    {
    public:
        RequestLease(RequestLease const&) = delete;
        RequestLease& operator=(RequestLease const&) = delete;
        RequestLease(RequestLease&& other) noexcept;
        RequestLease& operator=(RequestLease&& other) noexcept;
        ~RequestLease();

        //! Ordered slot map, including duplicates and padding; valid only while this lease is held.
        std::vector<SlotId> const& mapping() const;
        void release() noexcept;

    private:
        friend class PleCache;
        RequestLease(PleCache& owner, std::vector<SlotId> mapping, std::vector<SlotId> pinned);
        PleCache* mOwner;
        std::vector<SlotId> mMapping;
        std::vector<SlotId> mPinned;
    };

    //! Owns admission pins until commit or rollback. Destruction rolls back an uncommitted fill.
    class BatchReservation
    {
    public:
        BatchReservation(BatchReservation const&) = delete;
        BatchReservation& operator=(BatchReservation const&) = delete;
        BatchReservation(BatchReservation&& other) noexcept;
        BatchReservation& operator=(BatchReservation&& other) noexcept;
        ~BatchReservation();

        std::vector<Miss> const& misses() const noexcept;
        //! Number of distinct non-padding rows protected by this reservation.
        size_t uniqueRows() const noexcept
        {
            return mProtected.size();
        }
        //! Publish a newly reserved row only after its bytes are available to the consumer stream.
        void markReady(SlotId slot);
        //! Requires every miss ready; transfers the independent request pins to the caller.
        std::vector<RequestLease> commit();
        //! Removes new rows, including partially completed fills. Old protected hits survive;
        //! unprotected rows evicted during reservation are not restored.
        void rollback() noexcept;

    private:
        friend class PleCache;
        explicit BatchReservation(PleCache& owner);
        PleCache* mOwner;
        std::vector<Miss> mMisses;
        std::vector<SlotId> mProtected;
        std::vector<RequestLease> mRequests;
    };

    PleCache(RowId rowCount, SlotId slotCount);
    PleCache(PleCache const&) = delete;
    PleCache& operator=(PleCache const&) = delete;
    PleCache(PleCache&&) = delete;
    PleCache& operator=(PleCache&&) = delete;

    //! Validate all rows and union capacity before mutation. -1 maps to padding slot zero.
    //! Throws out_of_range for invalid rows, length_error for capacity, logic_error for an active batch.
    BatchReservation reserveBatch(std::vector<std::vector<RowId>> const& requests);

private:
    enum class State : uint8_t
    {
        kFREE,
        kFILLING,
        kREADY
    };
    struct Slot
    {
        RowId row{-1};
        int32_t references{0};
        State state{State::kFREE};
        bool referenced{false};
    };

    SlotId claimSlot();
    void unpin(std::vector<SlotId> const& slots) noexcept;
    void releaseRequest(std::vector<SlotId> const& slots) noexcept;

    RowId mRowCount;
    std::vector<Slot> mSlots;
    std::unordered_map<RowId, SlotId> mRows;
    SlotId mClock{1};
    int32_t mLiveRequests{0};
    bool mActiveBatch{false};
    bool mReservationActive{false};
};

} // namespace rt
} // namespace trt_edgellm
