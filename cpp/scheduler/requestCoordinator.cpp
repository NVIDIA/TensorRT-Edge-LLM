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

#include "scheduler/requestCoordinator.h"

#include "common/checkMacros.h"

#include <algorithm>
#include <limits>
#include <utility>

namespace trt_edgellm
{
namespace rt
{
namespace scheduler
{

void RequestCoordinator::add(ResidentRef ref, std::shared_ptr<RequestRecord> record,
    std::unique_ptr<LLMGenerationRequest> request, AdmissionResult const& admitted)
{
    ELLM_CHECK(record != nullptr && request != nullptr, "RequestCoordinator: a resident needs its record and request.");
    ELLM_CHECK(admitted.status == AdmissionResult::Status::kAdmitted && admitted.ref == ref,
        "RequestCoordinator: only an admitted sequence is seated.");
    ELLM_CHECK(admitted.prefillStart >= 0 && admitted.prefillStart < admitted.promptLength,
        "RequestCoordinator: the reused prefix must leave at least one prompt token to compute.");
    auto const [it, inserted] = mResidents.emplace(
        ref.slot, Entry{ref, std::move(record), std::move(request), admitted.promptLength, admitted.prefillStart});
    ELLM_CHECK(inserted, "RequestCoordinator: a handle was seated twice.");
}

RequestCoordinator::Entry const& RequestCoordinator::live(ResidentRef ref) const
{
    auto const it = mResidents.find(ref.slot);
    ELLM_CHECK(
        it != mResidents.end() && it->second.ref == ref, "RequestCoordinator: a ref is not the table's live owner.");
    return it->second;
}

ScheduledSequenceDescriptor RequestCoordinator::contextStep(ResidentRef ref) const
{
    Entry const& entry = live(ref);
    ELLM_CHECK(!entry.contextCommitted, "RequestCoordinator: this resident's context step already ran.");
    return ScheduledSequenceDescriptor{SequenceIdentity{entry.record->id, ref}, SequenceWork::kContext,
        entry.promptLength - entry.committed, entry.committed, /*selectLastTokenLogits=*/true};
}

RequestCoordinator::Progress RequestCoordinator::progress(ResidentRef ref) const
{
    Entry const& entry = live(ref);
    return Progress{
        SequenceIdentity{entry.record->id, ref}, entry.promptLength, entry.committed, entry.contextCommitted};
}

LLMGenerationRequest const& RequestCoordinator::request(ResidentRef ref) const
{
    return *live(ref).request;
}

RequestRecord* RequestCoordinator::findRecord(ResidentRef ref) noexcept
{
    auto const it = mResidents.find(ref.slot);
    return it == mResidents.end() || it->second.ref != ref ? nullptr : it->second.record.get();
}

ScheduledSequenceDescriptor RequestCoordinator::contextChunk(ResidentRef ref, int32_t queryLength) const
{
    Entry const& entry = live(ref);
    ELLM_CHECK(!entry.contextCommitted, "RequestCoordinator: this resident's prompt already completed.");
    int32_t const remaining = entry.promptLength - entry.committed;
    ELLM_CHECK(queryLength > 0 && queryLength <= remaining,
        "RequestCoordinator: a prompt chunk covers 0 < queryLength <= the remaining prompt.");
    return ScheduledSequenceDescriptor{SequenceIdentity{entry.record->id, ref}, SequenceWork::kContext, queryLength,
        entry.committed, /*selectLastTokenLogits=*/queryLength == remaining};
}

void RequestCoordinator::decodeStep(
    std::vector<ResidentRef> const& residents, std::vector<ScheduledSequenceDescriptor>& plan) const
{
    for (ResidentRef const ref : residents)
    {
        Entry const& entry = live(ref);
        ELLM_CHECK(entry.contextCommitted,
            "RequestCoordinator: a resident cannot decode before its context step committed its prompt.");
        plan.push_back(ScheduledSequenceDescriptor{SequenceIdentity{entry.record->id, ref}, SequenceWork::kDecode,
            /*queryLength=*/1, entry.committed, /*selectLastTokenLogits=*/true});
    }
}

std::vector<RequestCoordinator::TerminalOutcome> RequestCoordinator::commit(
    StepResult const& result, RuntimeSession const& session)
{
    // Two passes: the record is validated first and touches nothing, so a refused record leaves
    // the table exactly as it was and every record still owned -- the caller then fails the
    // session through retireEverything() and no resident loses its outcome half-way through a
    // commit. The second pass cannot throw. A step that names a ref this table does not hold, or
    // that ran from a past length the ledger does not hold, is refused before a single response
    // is materialized.
    for (SequenceIdentity const& participant : result.participants)
    {
        static_cast<void>(live(participant.resident));
    }
    for (ScheduledSequenceDescriptor const& executed : result.execution)
    {
        Entry const& entry = live(executed.identity.resident);
        ELLM_CHECK(executed.pastLength == entry.committed,
            "RequestCoordinator: the runtime executed a step from a past length the ledger does not hold.");
        ELLM_CHECK(executed.work != SequenceWork::kContext || executed.selectLastTokenLogits
                || entry.committed + executed.queryLength < entry.promptLength,
            "RequestCoordinator: a prompt chunk that reaches the end of the prompt must be the final, sampling chunk.");
    }
    std::vector<TerminalOutcome> outcomes;
    std::vector<int32_t> retiring;
    outcomes.reserve(result.finished.size());
    retiring.reserve(result.finished.size());
    for (auto const& [ref, snapshot] : result.finished)
    {
        Entry const& entry = live(ref);
        ELLM_CHECK(std::find(retiring.begin(), retiring.end(), ref.slot) == retiring.end(),
            "RequestCoordinator: a ref finished twice in one step.");
        TerminalOutcome outcome;
        outcome.reason = snapshot.terminalReason;
        if (snapshot.terminalReason != FinishReason::kCancelled && snapshot.terminalReason != FinishReason::kError)
        {
            try
            {
                outcome.response = session.materialize(snapshot, entry.request->requests.front().stopStrings);
            }
            catch (std::exception const& error)
            {
                // The sequence ran to its end and the step's state is sound; only its response could
                // not be assembled. That is this sequence's error, not the session's.
                outcome.reason = FinishReason::kError;
                outcome.errorMessage = std::string("materializing the response failed: ") + error.what();
            }
        }
        outcomes.push_back(std::move(outcome));
        retiring.push_back(ref.slot);
    }

    for (size_t i = 0; i < retiring.size(); ++i)
    {
        auto const it = mResidents.find(retiring[i]);
        outcomes[i].record = std::move(it->second.record);
        mResidents.erase(it);
    }
    // Survivors advance by what the runtime executed for them -- the positions the step added, as
    // the step record reports them -- not by what the plan asked. (A sequence that finished this
    // step was erased above; a participant without an execution record ran nothing and stays.)
    for (ScheduledSequenceDescriptor const& executed : result.execution)
    {
        auto const it = mResidents.find(executed.identity.resident.slot);
        if (it == mResidents.end() || it->second.ref != executed.identity.resident)
        {
            continue;
        }
        it->second.committed += executed.queryLength;
        if (executed.work == SequenceWork::kContext && executed.selectLastTokenLogits)
        {
            it->second.contextCommitted = true;
        }
    }
    return outcomes;
}

std::vector<RequestCoordinator::TerminalOutcome> RequestCoordinator::retireEverything(
    FinishReason reason, std::string errorMessage)
{
    std::vector<TerminalOutcome> outcomes;
    outcomes.reserve(mResidents.size());
    for (auto& [refKey, entry] : mResidents)
    {
        TerminalOutcome outcome;
        outcome.record = std::move(entry.record);
        outcome.reason = reason;
        outcome.errorMessage = errorMessage;
        outcomes.push_back(std::move(outcome));
    }
    mResidents.clear();
    return outcomes;
}

std::vector<RequestCoordinator::Released> RequestCoordinator::releasePromptIncomplete()
{
    std::vector<Released> released;
    for (auto it = mResidents.begin(); it != mResidents.end();)
    {
        if (it->second.contextCommitted)
        {
            ++it;
            continue;
        }
        released.push_back(Released{it->second.ref, std::move(it->second.record), std::move(it->second.request)});
        it = mResidents.erase(it);
    }
    return released;
}

bool RequestCoordinator::describes(std::vector<ResidentRef> const& residents) const noexcept
{
    if (residents.size() != mResidents.size())
    {
        return false;
    }
    return std::all_of(residents.begin(), residents.end(), [this](ResidentRef const ref) {
        auto const it = mResidents.find(ref.slot);
        return it != mResidents.end() && it->second.ref == ref;
    });
}

LLMGenerationRequest const* RequestCoordinator::anyRequest() const noexcept
{
    return mResidents.empty() ? nullptr : mResidents.begin()->second.request.get();
}

int32_t RequestCoordinator::mostUrgentPriority() const noexcept
{
    int32_t priority = std::numeric_limits<int32_t>::max();
    for (auto const& [slot, entry] : mResidents)
    {
        priority = std::min(priority, entry.request->priority);
    }
    return priority;
}

} // namespace scheduler
} // namespace rt
} // namespace trt_edgellm
