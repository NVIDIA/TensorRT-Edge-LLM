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
#include <utility>

namespace trt_edgellm
{
namespace rt
{
namespace scheduler
{

void RequestCoordinator::add(ResidentRef ref, std::shared_ptr<RequestRecord> record, std::vector<std::string> stops)
{
    auto const [it, inserted] = mResidents.emplace(ref.slot, Entry{ref, std::move(record), std::move(stops)});
    ELLM_CHECK(inserted, "RequestCoordinator: a handle was seated twice.");
}

std::vector<RequestCoordinator::TerminalOutcome> RequestCoordinator::commit(
    StepResult const& result, SteppedExecution& stepped)
{
    // Two passes: everything that can throw (a ref that is not the live owner, a materialize
    // failure) runs first and touches nothing, so a throw leaves the table exactly as it was and
    // every record still owned -- the caller then fails the batch through failEverything() and no
    // resident loses its outcome half-way through a commit. The second pass cannot throw.
    std::vector<TerminalOutcome> outcomes;
    std::vector<int32_t> retiring;
    outcomes.reserve(result.finished.size());
    retiring.reserve(result.finished.size());
    for (auto const& [ref, snapshot] : result.finished)
    {
        auto const it = mResidents.find(ref.slot);
        ELLM_CHECK(it != mResidents.end() && it->second.ref == ref,
            "RequestCoordinator: a finished ref is not the table's live owner.");
        ELLM_CHECK(std::find(retiring.begin(), retiring.end(), ref.slot) == retiring.end(),
            "RequestCoordinator: a ref finished twice in one step.");
        TerminalOutcome outcome;
        outcome.reason = snapshot.terminalReason;
        if (snapshot.terminalReason != FinishReason::kCancelled && snapshot.terminalReason != FinishReason::kError)
        {
            outcome.response = stepped.materialize(snapshot, it->second.stopStrings);
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

    return outcomes;
}

std::vector<RequestCoordinator::TerminalOutcome> RequestCoordinator::failEverything()
{
    std::vector<TerminalOutcome> outcomes;
    outcomes.reserve(mResidents.size());
    for (auto& [refKey, entry] : mResidents)
    {
        TerminalOutcome outcome;
        outcome.record = std::move(entry.record);
        outcome.reason = FinishReason::kError;
        outcomes.push_back(std::move(outcome));
    }
    mResidents.clear();
    return outcomes;
}

} // namespace scheduler
} // namespace rt
} // namespace trt_edgellm
