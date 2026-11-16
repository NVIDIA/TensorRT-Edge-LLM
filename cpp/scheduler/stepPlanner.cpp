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

#include "scheduler/stepPlanner.h"

#include "common/checkMacros.h"

#include <algorithm>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>

namespace trt_edgellm
{
namespace rt
{
namespace scheduler
{

StepPlanner::StepPlanner(ResolvedSchedulingPolicy const& policy, StepPlannerLimits const& limits)
    : mMode(policy.mode)
    , mSchedulingPolicy(policy.schedulingPolicy)
    , mPreemption(policy.preemption)
    , mMaxPauseSteps(policy.maxPauseSteps)
    , mMaxNumTokens(policy.budgets.maxNumTokens)
    , mMaxChunkTokens(policy.budgets.maxChunkTokens)
    , mLimits(limits)
    , mTokenLimit(static_cast<int64_t>(limits.maxBatchSize) * limits.maxInputLength)
{
    ELLM_CHECK(limits.maxBatchSize > 0 && limits.maxInputLength > 0 && limits.kvPageSize > 0
            && (mMode == IfbSchedulingMode::kMixed || mMode == IfbSchedulingMode::kPrefillFirst),
        "StepPlanner requires engine limits and a policy resolveSchedulingPolicy resolved to mixed or prefillFirst.");

    auto const capacity = static_cast<size_t>(limits.maxBatchSize);
    mOrder.reserve(capacity);
    mPausedFlags.reserve(capacity);
    mDecodes.reserve(capacity);
    mPlan.sequences.reserve(capacity);
    mPlan.paused.reserve(capacity);
}

bool StepPlanner::precedes(ResidentState const& lhs, int32_t lhsIndex, ResidentState const& rhs, int32_t rhsIndex) const
{
    if (mSchedulingPolicy == SchedulingPolicy::kPriority && lhs.priority != rhs.priority)
    {
        return lhs.priority < rhs.priority;
    }
    if (lhs.arrival != rhs.arrival)
    {
        return lhs.arrival < rhs.arrival;
    }
    return lhsIndex < rhsIndex;
}

StepPlan const& StepPlanner::plan(ResidentState const* residents, int32_t count)
{
    if (count < 0 || count > mLimits.maxBatchSize)
    {
        throw std::invalid_argument("StepPlanner::plan: " + std::to_string(count) + " residents exceed maxBatchSize "
            + std::to_string(mLimits.maxBatchSize) + ".");
    }
    for (int32_t i = 0; i < count; ++i)
    {
        if (!residents[i].contextCommitted && residents[i].promptLength - residents[i].committed <= 0)
        {
            throw std::invalid_argument("StepPlanner::plan: resident " + std::to_string(i)
                + " has an unfinished prompt but committed " + std::to_string(residents[i].committed)
                + " >= promptLength " + std::to_string(residents[i].promptLength) + ".");
        }
    }

    mPlan.sequences.clear();
    mPlan.paused.clear();
    mPlan.numPrefill = 0;
    mPlan.numDecode = 0;
    mPlan.prefillTokens = 0;
    mPlan.totalTokens = 0;
    mDecodes.clear();

    mOrder.resize(static_cast<size_t>(count));
    std::iota(mOrder.begin(), mOrder.end(), 0);
    std::sort(mOrder.begin(), mOrder.end(),
        [&](int32_t lhs, int32_t rhs) { return precedes(residents[lhs], lhs, residents[rhs], rhs); });

    // The first prompt-incomplete resident in order: what pause protects, and the urgency prefillFirst serves.
    auto const head
        = std::find_if(mOrder.begin(), mOrder.end(), [&](int32_t index) { return !residents[index].contextCommitted; });
    mPausedFlags.assign(static_cast<size_t>(count), 0);
    if (head != mOrder.end() && mPreemption == PreemptionMode::kPause
        && mSchedulingPolicy == SchedulingPolicy::kPriority)
    {
        int32_t const headPriority = residents[*head].priority;
        for (int32_t i = 0; i < count; ++i)
        {
            bool const pauseBudgetLeft = mMaxPauseSteps == 0 || residents[i].consecutivePausedSteps < mMaxPauseSteps;
            if (residents[i].priority > headPriority && pauseBudgetLeft)
            {
                mPausedFlags[static_cast<size_t>(i)] = 1;
                mPlan.paused.push_back(i);
            }
        }
    }

    for (int32_t const index : mOrder)
    {
        ResidentState const& resident = residents[index];
        if (mPausedFlags[static_cast<size_t>(index)] == 0 && resident.contextCommitted)
        {
            mDecodes.push_back(PlannedSequence{index, SequenceWork::kDecode, resident.committed, 1, true});
        }
    }

    int64_t budget = mTokenLimit;
    int64_t cap = mLimits.maxInputLength;
    if (mMode == IfbSchedulingMode::kPrefillFirst)
    {
        if (head != mOrder.end())
        {
            mDecodes.clear();
        }
    }
    else if (!mDecodes.empty())
    {
        budget = mMaxNumTokens - static_cast<int64_t>(mDecodes.size());
        cap = std::min<int64_t>(mMaxChunkTokens, mLimits.maxInputLength);
    }

    // prefillFirst under the priority policy runs the most urgent pending prompts alone; the others wait for
    // the next prefill step rather than sharing (and lengthening) this one.
    int32_t const urgency = head != mOrder.end() && mMode == IfbSchedulingMode::kPrefillFirst
            && mSchedulingPolicy == SchedulingPolicy::kPriority
        ? residents[*head].priority
        : std::numeric_limits<int32_t>::max();

    int64_t const page = mLimits.kvPageSize;
    for (int32_t const index : mOrder)
    {
        ResidentState const& resident = residents[index];
        if (mPausedFlags[static_cast<size_t>(index)] != 0 || resident.contextCommitted || resident.priority > urgency)
        {
            continue;
        }
        int64_t const remaining = static_cast<int64_t>(resident.promptLength) - resident.committed;
        int64_t n = std::min({remaining, cap, budget});
        if (n < remaining)
        {
            n = (resident.committed + n) / page * page - resident.committed;
        }
        if (n <= 0)
        {
            continue;
        }
        mPlan.sequences.push_back(PlannedSequence{
            index, SequenceWork::kContext, resident.committed, static_cast<int32_t>(n), n == remaining});
        budget -= n;
        mPlan.prefillTokens += static_cast<int32_t>(n);
    }
    mPlan.numPrefill = static_cast<int32_t>(mPlan.sequences.size());
    mPlan.numDecode = static_cast<int32_t>(mDecodes.size());
    mPlan.totalTokens = mPlan.prefillTokens + mPlan.numDecode;
    mPlan.sequences.insert(mPlan.sequences.end(), mDecodes.begin(), mDecodes.end());
    return mPlan;
}

} // namespace scheduler
} // namespace rt
} // namespace trt_edgellm
