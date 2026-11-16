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

#include <gtest/gtest.h>

#include <string>
#include <vector>

using namespace trt_edgellm::rt;
using namespace trt_edgellm::rt::scheduler;

namespace
{

ResolvedSchedulingPolicy policyOf(IfbSchedulingMode mode, SchedulingPolicy schedulingPolicy = SchedulingPolicy::kFcfs,
    PreemptionMode preemption = PreemptionMode::kNone, int32_t maxNumTokens = 640, int32_t maxChunkTokens = 512)
{
    ResolvedSchedulingPolicy policy;
    policy.mode = mode;
    policy.schedulingPolicy = schedulingPolicy;
    policy.preemption = preemption;
    if (mode == IfbSchedulingMode::kMixed)
    {
        policy.budgets = StepBudgets{maxNumTokens, maxChunkTokens};
    }
    policy.compactExecution = true;
    return policy;
}

ResolvedSchedulingPolicy const kMixed = policyOf(IfbSchedulingMode::kMixed);
ResolvedSchedulingPolicy const kMixedPriority = policyOf(IfbSchedulingMode::kMixed, SchedulingPolicy::kPriority);
ResolvedSchedulingPolicy const kMixedPause
    = policyOf(IfbSchedulingMode::kMixed, SchedulingPolicy::kPriority, PreemptionMode::kPause);
ResolvedSchedulingPolicy const kPrefillFirst = policyOf(IfbSchedulingMode::kPrefillFirst);
ResolvedSchedulingPolicy const kPrefillFirstPriority
    = policyOf(IfbSchedulingMode::kPrefillFirst, SchedulingPolicy::kPriority);

//! B = 4, S_in = 1024 (T_max = 4096), P = 128.
StepPlannerLimits engineLimits(int32_t maxInputLength = 1024)
{
    return StepPlannerLimits{4, maxInputLength, 128};
}

ResidentState decoding(uint64_t arrival, int32_t priority = 0)
{
    return ResidentState{priority, arrival, 9, 9, true};
}

ResidentState prompting(uint64_t arrival, int32_t promptLength, int32_t committed = 0, int32_t priority = 0)
{
    return ResidentState{priority, arrival, promptLength, committed, false};
}

//! One word per sequence: "R<resident>:<past>+<query>" for a chunk, suffixed "!" when it samples,
//! "R<resident>:d" for a decode row; then "(paused R<resident>...)" when residents are paused.
std::string layout(StepPlan const& plan)
{
    std::string text;
    for (auto const& sequence : plan.sequences)
    {
        text += (text.empty() ? "R" : " R") + std::to_string(sequence.resident) + ':';
        if (sequence.work == SequenceWork::kDecode)
        {
            text += sequence.queryLength == 1 && sequence.selectLastTokenLogits ? "d" : "bad-decode";
            continue;
        }
        text += std::to_string(sequence.pastLength) + '+' + std::to_string(sequence.queryLength)
            + (sequence.selectLastTokenLogits ? "!" : "");
    }
    for (size_t i = 0; i < plan.paused.size(); ++i)
    {
        text += (i == 0 ? " (paused R" : " R") + std::to_string(plan.paused[i])
            + (i + 1 == plan.paused.size() ? ")" : "");
    }
    return text;
}

struct Case
{
    char const* what;
    ResolvedSchedulingPolicy policy;
    StepPlannerLimits limits;
    std::vector<ResidentState> residents;
    //! The layout of each step, each committed as a successful step would before the next is planned.
    std::vector<std::string> steps;
};

void expectSteps(Case const& row)
{
    SCOPED_TRACE(row.what);
    StepPlanner planner(row.policy, row.limits);
    std::vector<ResidentState> residents = row.residents;
    for (size_t step = 0; step < row.steps.size(); ++step)
    {
        StepPlan const& plan = planner.plan(residents);
        EXPECT_EQ(layout(plan), row.steps[step]) << "step " << step + 1;
        for (auto const& sequence : plan.sequences)
        {
            ResidentState& resident = residents[static_cast<size_t>(sequence.resident)];
            resident.committed += sequence.queryLength;
            resident.contextCommitted = resident.contextCommitted || sequence.selectLastTokenLogits;
        }
    }
}

} // namespace

TEST(StepPlannerTests, MixedStepsChunkPromptsBesideDecodeRowsWithinTheBudgets)
{
    // Given mixed (maxNumTokens 640, maxChunkTokens 512 unless the row says otherwise) on B = 4, S_in = 1024,
    //       P = 128, and the row's residents in arrival order
    // When steps are planned and committed
    // Then every resident whose prompt completed decodes every step, and prompts take the budget left by the
    //      decode rows in order, each chunk capped by maxChunkTokens and S_in; a non-final chunk ends on a page
    //      boundary, so a prompt that cannot reach one is skipped while a later one that completes still runs; the
    //      final chunk need not be aligned; with no decode row the budget widens to T_max and the cap to S_in
    std::vector<Case> const cases{
        {"a prompt that cannot reach a page is skipped", kMixed, engineLimits(),
            {decoding(0), prompting(1, 1024), prompting(2, 1024), prompting(3, 60)},
            {"R1:0+512 R3:0+60! R0:d", "R1:512+512! R0:d R3:d", "R2:0+512 R0:d R1:d R3:d", "R2:512+512! R0:d R1:d R3:d",
                "R0:d R1:d R2:d R3:d"}},
        {"a reused prefix shortens the first chunk to the page", kMixed, engineLimits(),
            {decoding(0), prompting(1, 1000, 100)}, {"R1:100+412 R0:d", "R1:512+488! R0:d"}},
        {"S_in caps a chunk below maxChunkTokens",
            policyOf(IfbSchedulingMode::kMixed, SchedulingPolicy::kFcfs, PreemptionMode::kNone, 1024, 512),
            engineLimits(256), {decoding(0), prompting(1, 600)}, {"R1:0+256 R0:d"}},
        {"no decode row widens the budget", kMixed, engineLimits(), {prompting(0, 1024), prompting(1, 1024)},
            {"R0:0+1024! R1:0+1024!"}},
    };
    for (Case const& row : cases)
    {
        expectSteps(row);
    }
}

TEST(StepPlannerTests, PrefillFirstStepsRunPromptsWithoutDecodeRows)
{
    // Given prefillFirst on B = 4, S_in = 1024, P = 128 and the row's residents
    // When steps are planned and committed
    // Then a step with a pending prompt holds no decode row and pauses nobody, its prompts share T_max with each
    //      chunk capped at S_in and non-final chunks ending on a page boundary, and the next step without a
    //      pending prompt decodes every resident; under fcfs every pending prompt shares the step whatever its
    //      priority, and under the priority policy only the most urgent prompts do, in arrival order
    std::vector<Case> const cases{
        {"whole prompts while decodes wait", kPrefillFirst, engineLimits(),
            {decoding(0), prompting(1, 1024), prompting(2, 1024)}, {"R1:0+1024! R2:0+1024!", "R0:d R1:d R2:d"}},
        {"T_max is filled and the prompts finish next step", kPrefillFirst, engineLimits(),
            {prompting(0, 1500, 0, 1), prompting(1, 1500), prompting(2, 1500, 0, 1), prompting(3, 1500)},
            {"R0:0+1024 R1:0+1024 R2:0+1024 R3:0+1024", "R0:1024+476! R1:1024+476! R2:1024+476! R3:1024+476!"}},
        {"priority runs the most urgent alone", kPrefillFirstPriority, engineLimits(),
            {decoding(0, 1), prompting(1, 300, 0, 1), prompting(2, 200, 0, -1), prompting(3, 100, 0, -1)},
            {"R2:0+200! R3:0+100!", "R1:0+300!", "R2:d R3:d R0:d R1:d"}},
    };
    for (Case const& row : cases)
    {
        expectSteps(row);
    }
}

TEST(StepPlannerTests, ThePolicyOrdersTheStepAndPauseWithholdsLowerPriorityResidents)
{
    // Given mixed (B = 4, S_in = 1024, maxNumTokens 640, maxChunkTokens 512, P = 128) and the row's residents
    // When steps are planned and committed
    // Then chunks and decode rows each follow arrival order under fcfs and (priority, arrival) under the
    //      priority policy, which still lets every pending prompt share the step; with preemption = pause every
    //      resident ranked below the most urgent pending prompt is paused until that prompt's final chunk
    //      commits, and when that leaves no decode row the prompt runs whole
    std::vector<ResidentState> const outOfOrder{
        prompting(4, 1024, 0, -1), decoding(2, 5), prompting(3, 1024, 0, 5), decoding(1, -1)};
    std::vector<Case> const cases{
        {"fcfs follows arrival", kMixed, engineLimits(), outOfOrder, {"R2:0+512 R3:d R1:d"}},
        {"priority, then arrival", kMixedPriority, engineLimits(), outOfOrder, {"R0:0+512 R3:d R1:d"}},
        {"priority shares the step under mixed", kMixedPriority, engineLimits(),
            {decoding(0, 1), prompting(1, 300, 0, 1), prompting(2, 200, 0, -1), prompting(3, 100, 0, -1)},
            {"R2:0+200! R3:0+100! R1:0+300! R0:d"}},
        {"pause until the final chunk", kMixedPause, engineLimits(), {decoding(0), decoding(1, 1), prompting(2, 1000)},
            {"R2:0+512 R0:d (paused R1)", "R2:512+488! R0:d (paused R1)", "R0:d R2:d R1:d"}},
        {"pausing every decode row", kMixedPause, engineLimits(),
            {decoding(0, 1), prompting(1, 300, 0, 2), prompting(2, 1000)}, {"R2:0+1000! (paused R0 R1)"}},
    };
    for (Case const& row : cases)
    {
        expectSteps(row);
    }
}
