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

//! The engine's table of residents: the canonical owner of resident identity under the stepped
//! control plane, and the single point where a StepResult becomes fact.
//!
//! The runtime's dense execution layout (which row a sequence occupies) is transient and
//! runner-private; this table maps each live ResidentRef to its request record, its request, and
//! how far its state is committed, and is updated only inside commit(). It lives as long as the
//! session it describes -- residents come and go, the table stays -- and it is what the engine
//! builds steps from: the context step of a newly admitted sequence covers its uncommitted prompt,
//! a decode step advances every resident by one token from its committed length.
//!
//! The ledger mirrors what the runtime executed, never what the plan asked: a step's execution
//! record (StepResult::execution) says how many positions each participant gained, and that is
//! what commit() adds. The two differ under a visual-token pruner, whose context step adds fewer
//! positions than the prompt range it was asked to cover; a ledger advanced from the plan would
//! then build decode steps from a past length the runtime does not have.
//!
//! Thread-free by design: the engine's actor thread pumps it, and every transition is
//! deterministic, so the whole scheduling state machine is testable without a thread.
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
        //! For a failed sequence: why, when the retiring party knows (a failed session names its
        //! failure). Empty for a sequence that failed inside its own step.
        std::string errorMessage;
    };

    //! Seat one resident after its admission. @p admitted supplies the prompt length and the
    //! reused prefix the context step is built from; the request is kept for the compatibility
    //! comparison of later heads and for its stop strings.
    void add(ResidentRef ref, std::shared_ptr<RequestRecord> record, std::unique_ptr<LLMGenerationRequest> request,
        AdmissionResult const& admitted);

    //! The context step of a resident that has not run one yet: its whole uncommitted prompt,
    //! sampling one token.
    ScheduledSequenceDescriptor contextStep(ResidentRef ref) const;

    //! Where one resident stands, for a planner composing compact steps.
    struct Progress
    {
        SequenceIdentity identity;
        //! The prompt as the runtime tokenized it.
        int32_t promptLength{0};
        //! Positions with state behind them (see Entry::committed).
        int32_t committed{0};
        //! The prompt's final chunk ran: the resident decodes from here on.
        bool contextCommitted{false};
    };
    Progress progress(ResidentRef ref) const;

    //! The request a resident was admitted with.
    LLMGenerationRequest const& request(ResidentRef ref) const;

    //! The record of live resident @p ref, or null when the table does not hold it.
    RequestRecord* findRecord(ResidentRef ref) noexcept;

    //! One prompt chunk of a resident whose context has not committed: @p queryLength positions
    //! from its committed length, sampling exactly when the chunk ends the prompt (the final
    //! chunk). Requires 0 < queryLength <= promptLength - committed.
    ScheduledSequenceDescriptor contextChunk(ResidentRef ref, int32_t queryLength) const;

    //! Append to @p plan one decode step over @p residents (the session's live set, in its order): one
    //! query token each, past its committed length.
    void decodeStep(std::vector<ResidentRef> const& residents, std::vector<ScheduledSequenceDescriptor>& plan) const;

    //! Fold one StepResult into the table: advance every executed sequence's committed length by
    //! what the runtime executed for it, mark the context committed when the executed context work
    //! sampled (its final chunk; an earlier chunk only advances committed), and turn every finished
    //! snapshot into a TerminalOutcome (materializing through @p session). Refs are stable
    //! handles, so there is nothing to re-key; deltas need no handling either -- token delivery
    //! flows through the runtime's own stream channels. All-or-nothing: a throw (a ref that is not
    //! the live owner, an execution record that started from a past length the ledger does not
    //! hold, a materialize failure) leaves the table exactly as it was.
    std::vector<TerminalOutcome> commit(StepResult const& result, RuntimeSession const& session);

    //! Every remaining resident owes its caller an outcome; a dead session pays that debt here, or
    //! a cancelling shutdown does. @p reason is what each outcome reports, and @p errorMessage,
    //! for a failure, why -- the caller's message reaches every retired resident.
    std::vector<TerminalOutcome> retireEverything(FinishReason reason, std::string errorMessage = {});

    //! A resident taken off the table without an outcome, to be queued again.
    struct Released
    {
        ResidentRef ref;
        std::shared_ptr<RequestRecord> record;
        std::unique_ptr<LLMGenerationRequest> request;
    };

    //! Take every resident whose prompt's final chunk has not committed -- one that has sampled no token -- off the
    //! table, with its record and request, so the caller can admit it again into a fresh session.
    std::vector<Released> releasePromptIncomplete();

    //! True when @p residents -- the session's live set -- is exactly the set this table holds.
    //! Checked after every step: a sequence the session holds that the table does not, or the
    //! reverse, means the two ledgers have diverged and nothing built from either can be trusted.
    bool describes(std::vector<ResidentRef> const& residents) const noexcept;

    bool empty() const noexcept
    {
        return mResidents.empty();
    }

    int32_t size() const noexcept
    {
        return static_cast<int32_t>(mResidents.size());
    }

    //! Any resident's request, for the compatibility comparison of a head against the batch: every
    //! resident was admitted through the same gate, so they agree on the compared fields and any
    //! one stands for all. Null when nothing is resident.
    LLMGenerationRequest const* anyRequest() const noexcept;

    //! The lowest priority value among the residents' requests (the most urgent one); INT32_MAX
    //! when nothing is resident.
    int32_t mostUrgentPriority() const noexcept;

    //! Visit every resident's record.
    template <typename Visitor>
    void forEachRecord(Visitor&& visit) const
    {
        for (auto const& [slot, entry] : mResidents)
        {
            visit(*entry.record);
        }
    }

private:
    struct Entry
    {
        ResidentRef ref;
        std::shared_ptr<RequestRecord> record;
        std::unique_ptr<LLMGenerationRequest> request;
        //! The prompt as the runtime tokenized it: what the context step feeds.
        int32_t promptLength{0};
        //! Positions the sequence has state behind: the reused prefix at admission, then what each
        //! executed step added. Under a pruner this stays below promptLength after the context step.
        int32_t committed{0};
        //! Whether the prompt's final chunk ran and committed. Decided by the step record -- the
        //! context execution that sampled -- not by comparing committed against promptLength, which
        //! the pruner makes meaningless.
        bool contextCommitted{false};
    };

    Entry const& live(ResidentRef ref) const;

    //! Keyed by handle slot: a live resident's handle slot is unique by the pool's contract, and
    //! the stored ref's epoch is checked on every lookup so a stale ref fails loudly.
    std::unordered_map<int32_t, Entry> mResidents;
};

} // namespace scheduler
} // namespace rt
} // namespace trt_edgellm
