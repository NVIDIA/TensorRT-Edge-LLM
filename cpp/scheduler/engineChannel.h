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

#include "scheduler/commandQueue.h"
#include "scheduler/requestIdentity.h"
#include "scheduler/requestRecord.h"

#include <memory>
#include <mutex>
#include <unordered_map>

namespace trt_edgellm
{
namespace rt
{
namespace scheduler
{

//! The command queue plus the gate that decides whether new work may still enter it.
//!
//! Held by shared_ptr by the engine and by every handle it hands out, which is what makes a handle
//! that outlives its engine safe: cancelling through a dead engine reaches this object rather than
//! a destroyed one. Commands posted after the engine is gone are simply never drained, and that
//! costs nothing because the engine terminates every outstanding request before it goes away.
//!
//! Admission is guarded by a mutex rather than an atomic flag, and that is the point of the class.
//! Checking a flag and then pushing are two steps, so a submit that reads the gate as open can be
//! descheduled, have the engine shut down and sweep the queue underneath it, and then publish a
//! request that no actor will ever run -- stranding its caller forever. Closing the gate under the
//! same mutex the push holds makes "accepted" and "will be drained" the same decision. The lock is
//! held for one queue push, is never taken by the actor, and so never serialises execution.
class EngineChannel
{
public:
    explicit EngineChannel(int32_t commandQueueCapacity)
        : mCommands(commandQueueCapacity)
    {
    }

    EngineChannel(EngineChannel const&) = delete;
    EngineChannel& operator=(EngineChannel const&) = delete;

    //! @brief Any thread: offer a request. Fails if admission is closed or the queue is full.
    bool postSubmit(std::unique_ptr<EngineCommand> command)
    {
        std::lock_guard<std::mutex> lock(mAdmissionMutex);
        if (!mAccepting)
        {
            return false;
        }
        return mCommands.push(std::move(command));
    }

    //! @brief Any thread: post a command that stops work rather than creating it.
    //!
    //! Deliberately not gated on admission. A cancel or shutdown arriving after the gate closed is
    //! exactly the case that still needs delivering, and the actor drops anything it can no longer
    //! act on.
    bool post(std::unique_ptr<EngineCommand> command)
    {
        std::lock_guard<std::mutex> lock(mAdmissionMutex);
        return mCommands.push(std::move(command));
    }

    //! @brief Stop accepting submissions. Once this returns, no push of a submit can still be in
    //!        flight, so a sweep of the queue afterwards is final.
    void closeAdmission()
    {
        std::lock_guard<std::mutex> lock(mAdmissionMutex);
        mAccepting = false;
    }

    bool accepting() const
    {
        std::lock_guard<std::mutex> lock(mAdmissionMutex);
        return mAccepting;
    }

    //! @brief Actor thread only.
    std::unique_ptr<EngineCommand> pop()
    {
        return mCommands.pop();
    }

    bool hasPending() const noexcept
    {
        return mCommands.hasPending();
    }

    //! The ring's real slot count (the requested capacity rounded up), so a drain budget sized by
    //! it empties a full queue in one pass.
    int32_t capacity() const noexcept
    {
        return mCommands.capacity();
    }

    //! @brief Any thread: make @p record cancellable by id for as long as it is live. submit()
    //! registers before posting; the actor forgets when the outcome is published.
    void registerRecord(std::shared_ptr<RequestRecord> const& record)
    {
        std::lock_guard<std::mutex> lock(mRecordsMutex);
        mLiveRecords[record->id] = record;
    }

    void forgetRecord(RequestId id)
    {
        std::lock_guard<std::mutex> lock(mRecordsMutex);
        mLiveRecords.erase(id);
    }

    //! @brief Any thread: plant the cancel flag on a live request by id, exactly as
    //! RequestHandle::cancel() does through its own record. Lossless -- no queue is involved, so
    //! overload cannot drop it -- and a no-op for an id that is unknown or already retired.
    //! @return whether a live record was found.
    bool cancelRecord(RequestId id) noexcept
    {
        std::shared_ptr<RequestRecord> record;
        {
            std::lock_guard<std::mutex> lock(mRecordsMutex);
            auto const it = mLiveRecords.find(id);
            if (it == mLiveRecords.end())
            {
                return false;
            }
            record = it->second.lock();
        }
        if (record == nullptr)
        {
            return false;
        }
        record->channel->cancel();
        if (record->runtimeChannel != nullptr && record->runtimeChannel != record->channel)
        {
            record->runtimeChannel->cancel();
        }
        record->wakeWaiters();
        return true;
    }

private:
    CommandQueue mCommands;
    //! Every accepted request that has not yet published an outcome, by id. Weak: the registry
    //! never extends a record's life, and a handle outliving its engine cancels through a dead
    //! entry as a harmless miss.
    std::mutex mRecordsMutex;
    std::unordered_map<RequestId, std::weak_ptr<RequestRecord>> mLiveRecords;
    mutable std::mutex mAdmissionMutex;
    bool mAccepting{true};
};

} // namespace scheduler
} // namespace rt
} // namespace trt_edgellm
