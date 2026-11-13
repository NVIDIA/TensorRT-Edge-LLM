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

#include "runtime/exec/scheduledStep.h"
#include "scheduler/schedulingPolicy.h"

#include <cstdint>
#include <vector>

namespace trt_edgellm
{
namespace rt
{
namespace scheduler
{

//! The ledger view of one resident that plan() reads.
struct ResidentState
{
    //! Lower value = higher priority. Ignored under SchedulingPolicy::kFcfs.
    int32_t priority{0};
    //! Monotonic admission-queue arrival stamp; unique per request.
    uint64_t arrival{0};
    int32_t promptLength{0};
    //! Tokens already in KV and recurrent state (the frontier).
    int32_t committed{0};
    //! The final prompt chunk has committed: the resident decodes. While false, the remaining prompt
    //! promptLength - committed must be positive.
    bool contextCommitted{false};
    //! Steps in a row that withheld this resident under preemption=pause, up to the step being planned.
    int32_t consecutivePausedSteps{0};
};

//! Engine build limits the planner budgets against.
struct StepPlannerLimits
{
    //! B: resident slots, and the bound on the residents passed to plan().
    int32_t maxBatchSize{0};
    //! S_in: per-sequence prefill tokens of one step. T_max = B x S_in.
    int32_t maxInputLength{0};
    //! P: non-final chunks end at a multiple of P.
    int32_t kvPageSize{0};
};

struct PlannedSequence
{
    //! Index of the resident in the array passed to plan().
    int32_t resident{0};
    //! kContext for a prompt chunk, kDecode for one decode token.
    SequenceWork work{SequenceWork::kDecode};
    //! Equals the resident's committed.
    int32_t pastLength{0};
    int32_t queryLength{0};
    //! True for decode rows and for the chunk that completes the prompt; a non-final chunk only advances committed.
    bool selectLastTokenLogits{false};
};

//! A planned step. With residents it always names at least one of them: the first unpaused resident in order
//! gets a decode row or a chunk of at least one token (see StepPlanner). The session decides how a plan runs.
struct StepPlan
{
    //! All prefill chunks in scheduling order, then all decode rows in scheduling order (the compact layout).
    std::vector<PlannedSequence> sequences;
    //! Residents withheld by preemption=pause, ascending index. Residents left out for any other reason
    //! (prefill-first decodes, prefills that got no budget) are simply absent from sequences.
    std::vector<int32_t> paused;
    //! Np, Nd, T_p and T.
    int32_t numPrefill{0};
    int32_t numDecode{0};
    int32_t prefillTokens{0};
    int32_t totalTokens{0};
};

//! @brief Chooses the work of one IFB step from the residents' ledger state.
//!
//! order(x) = (priority, arrival) under SchedulingPolicy::kPriority, else arrival.
//!
//! mixed: decode rows take one token each first; with any decode row the step budget is maxNumTokens
//! and the per-sequence chunk cap maxChunkTokens, otherwise T_max and S_in. prefillFirst: with any
//! prompt-incomplete resident only prefill is scheduled (budget T_max, cap S_in), else every resident
//! decodes; under kPriority that prefill step holds only the prompts of the best (lowest) pending
//! priority value, and the others wait for a later prefill step. Prefills are then filled in order:
//! n = min(remaining, cap, S_in, budget); a non-final chunk is trimmed back to a page boundary of
//! committed + n; a resident left with n <= 0 is skipped so a later, shorter prompt may still fit.
//! The first prompt in order is never skipped: admission bounds a prompt by S_in, and beside decode
//! rows the budget and maxChunkTokens are at least P, so its chunk completes the prompt or reaches
//! the next page boundary.
//!
//! Pause (mixed, priority policy, preemption=pause): h is the prompt-incomplete resident first in order;
//! residents with a larger priority value than h are paused and the plan is made over the rest, so
//! when no active resident decodes h gets the whole-prompt budget. The pause set is recomputed from the
//! ledger on every call and empties once h's final chunk commits. With maxPauseSteps > 0, a resident
//! whose consecutivePausedSteps has reached it is not paused in this plan. Under kFcfs every request
//! counts as priority 0, so nothing is paused.
//!
//! plan() allocates nothing: the buffers are sized for maxBatchSize at construction and the returned
//! plan is overwritten by the next call.
class StepPlanner
{
public:
    //! @p policy comes from resolveSchedulingPolicy() for an engine with @p limits, which validates its budgets.
    //! @throws std::runtime_error when its mode is not mixed or prefillFirst.
    StepPlanner(ResolvedSchedulingPolicy const& policy, StepPlannerLimits const& limits);

    //! @throws std::invalid_argument when @p count exceeds maxBatchSize or a prompt-incomplete resident
    //!         has no remaining prompt.
    StepPlan const& plan(ResidentState const* residents, int32_t count);

    StepPlan const& plan(std::vector<ResidentState> const& residents)
    {
        return plan(residents.data(), static_cast<int32_t>(residents.size()));
    }

private:
    bool precedes(ResidentState const& lhs, int32_t lhsIndex, ResidentState const& rhs, int32_t rhsIndex) const;

    IfbSchedulingMode mMode;
    SchedulingPolicy mSchedulingPolicy;
    PreemptionMode mPreemption;
    int32_t mMaxPauseSteps;
    int64_t mMaxNumTokens;
    int64_t mMaxChunkTokens;
    StepPlannerLimits mLimits;
    int64_t mTokenLimit;

    std::vector<int32_t> mOrder;
    std::vector<uint8_t> mPausedFlags;
    std::vector<PlannedSequence> mDecodes;
    StepPlan mPlan;
};

} // namespace scheduler
} // namespace rt
} // namespace trt_edgellm
