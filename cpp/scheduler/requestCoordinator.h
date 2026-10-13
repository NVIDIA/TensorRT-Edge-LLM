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

#include "runtime/runtimeStepper.h"
#include "scheduler/requestRecord.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace trt_edgellm
{
namespace rt
{
namespace scheduler
{

//! The canonical owner of resident identity under the stepped control plane, and the single
//! point where a StepResult becomes fact.
//!
//! The runtime's dense execution layout (which row a sequence occupies) is transient and
//! runner-private; this table maps each live ResidentRef to its request record, updated only
//! inside commit(). Thread-free by design: the engine's actor thread pumps it, and every
//! transition is deterministic, so the whole scheduling state machine is testable without a
//! thread. One table serves one batch; the engine builds a fresh one per founding.
class RequestCoordinator
{
public:
    //! What commit() decided a record's fate is. The engine publishes these (terminate, counters,
    //! pending-count release) -- publication is actor-side policy, the decision is made here.
    struct TerminalOutcome
    {
        std::shared_ptr<RequestRecord> record;
        FinishReason reason{FinishReason::kNotFinished};
        //! Present for a completed sequence: the materialized response.
        std::optional<LLMGenerationResponse> response;
    };

    //! Seat one resident (the founder at adoption, a joiner after its admission).
    void add(ResidentRef ref, std::shared_ptr<RequestRecord> record, std::vector<std::string> stopStrings);

    //! Fold one StepResult into the table: turn every finished snapshot into a TerminalOutcome
    //! (materializing through @p stepped). Refs are stable handles, so there is nothing to re-key;
    //! deltas need no handling either -- token delivery flows through the runtime's own stream
    //! channels.
    std::vector<TerminalOutcome> commit(StepResult const& result, SteppedExecution& stepped);

    //! Every remaining resident owes its caller an outcome; a dead batch pays that debt here.
    std::vector<TerminalOutcome> failEverything();

    bool empty() const noexcept
    {
        return mResidents.empty();
    }

    //! Visit every resident's record, founder and joiners alike. The cancelling shutdown plants
    //! its flag through this rather than through a second ledger.
    template <typename Visitor>
    void forEachRecord(Visitor&& visit) const
    {
        for (auto const& [slot, entry] : mResidents)
        {
            visit(*entry.record);
        }
    }

    //! Forget every resident without producing outcomes: only for reuse between batches, after
    //! every outcome of the previous one was published.
    void clear() noexcept
    {
        mResidents.clear();
    }

private:
    struct Entry
    {
        ResidentRef ref;
        std::shared_ptr<RequestRecord> record;
        std::vector<std::string> stopStrings;
    };

    //! Keyed by handle slot: a live resident's handle slot is unique by the pool's contract, and
    //! the stored ref's epoch is checked on every lookup so a stale ref fails loudly.
    std::unordered_map<int32_t, Entry> mResidents;
};

} // namespace scheduler
} // namespace rt
} // namespace trt_edgellm
