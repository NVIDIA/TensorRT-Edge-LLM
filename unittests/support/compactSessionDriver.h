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

#include "runtime/runtimeSession.h"

#include <gtest/gtest.h>

#include <atomic>
#include <memory>
#include <unordered_map>
#include <vector>

namespace trt_edgellm::rt
{

//! A scheduler's view of a session's residents: what it needs to emit compact plans, advanced from each step's
//! execution record exactly as the engine's ledger is.
class CompactDriver
{
public:
    explicit CompactDriver(RuntimeSession& session)
        : mSession(session)
    {
    }

    ResidentRef admit(LLMGenerationRequest const& request, RequestId requestId,
        std::shared_ptr<std::atomic<bool> const> cancelToken = nullptr)
    {
        AdmissionResult const admitted = mSession.admit(request, requestId, std::move(cancelToken));
        EXPECT_EQ(admitted.status, AdmissionResult::Status::kAdmitted) << admitted.reason;
        mResidents[admitted.ref.slot] = Resident{requestId, admitted.ref, admitted.promptLength, admitted.prefillStart};
        return admitted.ref;
    }

    int32_t promptLength(ResidentRef ref) const
    {
        return mResidents.at(ref.slot).promptLength;
    }

    int32_t committed(ResidentRef ref) const
    {
        return mResidents.at(ref.slot).committed;
    }

    ScheduledSequenceDescriptor chunk(ResidentRef ref, int32_t queryLength) const
    {
        Resident const& resident = mResidents.at(ref.slot);
        return ScheduledSequenceDescriptor{SequenceIdentity{resident.requestId, ref}, SequenceWork::kContext,
            queryLength, resident.committed, resident.committed + queryLength == resident.promptLength};
    }

    //! The rest of @p ref's prompt as one chunk.
    ScheduledSequenceDescriptor rest(ResidentRef ref) const
    {
        return chunk(ref, promptLength(ref) - committed(ref));
    }

    ScheduledSequenceDescriptor decode(ResidentRef ref) const
    {
        Resident const& resident = mResidents.at(ref.slot);
        return ScheduledSequenceDescriptor{
            SequenceIdentity{resident.requestId, ref}, SequenceWork::kDecode, 1, resident.committed, true};
    }

    std::vector<ScheduledSequenceDescriptor> decodeAll() const
    {
        std::vector<ScheduledSequenceDescriptor> plan;
        for (ResidentRef const ref : mSession.residents())
        {
            plan.push_back(decode(ref));
        }
        return plan;
    }

    StepResult run(std::vector<ScheduledSequenceDescriptor> const& plan)
    {
        StepResult result = mSession.execute(plan);
        for (ScheduledSequenceDescriptor const& executed : result.execution)
        {
            auto const it = mResidents.find(executed.identity.resident.slot);
            if (it != mResidents.end() && it->second.ref == executed.identity.resident)
            {
                EXPECT_EQ(executed.pastLength, it->second.committed) << "the runtime ran from another past length";
                it->second.committed += executed.queryLength;
            }
        }
        for (auto const& [ref, snapshot] : result.finished)
        {
            RequestId const requestId = mResidents.at(ref.slot).requestId;
            mOutcomes[requestId] = snapshot.terminalReason;
            if (snapshot.terminalReason != FinishReason::kCancelled && snapshot.terminalReason != FinishReason::kError)
            {
                mResponses[requestId] = mSession.materialize(snapshot, {});
            }
            mResidents.erase(ref.slot);
        }
        return result;
    }

    //! Decode every resident until none is left.
    void drain()
    {
        while (!mSession.idle())
        {
            ASSERT_TRUE(run(decodeAll()).ok);
        }
    }

    std::unordered_map<RequestId, FinishReason> const& outcomes() const
    {
        return mOutcomes;
    }

    std::unordered_map<RequestId, LLMGenerationResponse> const& responses() const
    {
        return mResponses;
    }

    std::vector<int32_t> const& output(RequestId requestId) const
    {
        return mResponses.at(requestId).outputIds.front();
    }

private:
    struct Resident
    {
        RequestId requestId{};
        ResidentRef ref;
        int32_t promptLength{0};
        int32_t committed{0};
    };

    RuntimeSession& mSession;
    std::unordered_map<int32_t, Resident> mResidents;
    std::unordered_map<RequestId, FinishReason> mOutcomes;
    std::unordered_map<RequestId, LLMGenerationResponse> mResponses;
};

} // namespace trt_edgellm::rt
