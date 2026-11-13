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

#include "scheduler/schedulingPolicy.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <stdexcept>
#include <string>
#include <vector>

using namespace trt_edgellm::rt;
using namespace trt_edgellm::rt::scheduler;

namespace
{

//! An engine built for B = 4, S_in = 1024 (T_max = 4096) with 128-token KV pages.
SessionCapabilities engineCapabilities(bool compact, int32_t maxBatchSize = 4, int32_t maxInputLength = 1024)
{
    SessionCapabilities capabilities;
    capabilities.compact = compact;
    capabilities.maxBatchSize = maxBatchSize;
    capabilities.maxInputLength = maxInputLength;
    capabilities.kvPageSize = 128;
    if (!compact)
    {
        capabilities.reason
            = "no compact-execution kernels on SM87 for gated_delta_net layers (K head dim 128, V head "
              "dim 128: no packed-varlen GDN kernel for this SM in this build)";
    }
    return capabilities;
}

SessionCapabilities const kCompact = engineCapabilities(true);
SessionCapabilities const kPadded = engineCapabilities(false);
//! T_max = 4 x 32 = 128 < B + P = 132: no mixed step fits.
SessionCapabilities const kTooSmall = engineCapabilities(true, 4, 32);
//! S_in = 256 below the default maxNumTokens - B = 636.
SessionCapabilities const kShortInput = engineCapabilities(true, 4, 256);

constexpr auto kAuto = IfbSchedulingMode::kAuto;
constexpr auto kMixed = IfbSchedulingMode::kMixed;
constexpr auto kPrefillFirst = IfbSchedulingMode::kPrefillFirst;
constexpr auto kSingleCohort = IfbSchedulingMode::kSingleCohort;
constexpr auto kFcfs = SchedulingPolicy::kFcfs;
constexpr auto kPriority = SchedulingPolicy::kPriority;
constexpr auto kNone = PreemptionMode::kNone;
constexpr auto kPause = PreemptionMode::kPause;

//! Resident slots of the engines below.
constexpr int32_t kResidents = 4;

SchedulingOptions configWith(IfbSchedulingMode mode, SchedulingPolicy policy = kFcfs, PreemptionMode preemption = kNone,
    int32_t reservedSlots = 0)
{
    SchedulingOptions options;
    options.ifbSchedulingMode = mode;
    options.schedulingPolicy = policy;
    options.preemption = preemption;
    options.priorityReservedSlots = reservedSlots;
    return options;
}

SchedulingOptions pauseLimit(PreemptionMode preemption, int32_t maxPauseSteps)
{
    SchedulingOptions options = configWith(kMixed, kPriority, preemption, 1);
    options.maxPauseSteps = maxPauseSteps;
    return options;
}

SchedulingOptions budgets(IfbSchedulingMode mode, int32_t maxNumTokens, int32_t maxChunkTokens)
{
    SchedulingOptions options = configWith(mode);
    options.maxNumTokens = maxNumTokens;
    options.maxChunkTokens = maxChunkTokens;
    return options;
}

ResolvedSchedulingPolicy resolve(
    SchedulingOptions const& options, SessionCapabilities const& capabilities, int32_t maxResidents = kResidents)
{
    return resolveSchedulingPolicy(options, maxResidents, capabilities);
}

//! "rejected: <message>" when resolution throws, else the warnings joined by " | ".
std::string outcomeOf(SchedulingOptions const& options, SessionCapabilities const& capabilities, int32_t maxResidents)
{
    try
    {
        std::string warnings;
        for (std::string const& warning : resolve(options, capabilities, maxResidents).warnings)
        {
            warnings += (warnings.empty() ? "" : " | ") + warning;
        }
        return warnings;
    }
    catch (std::invalid_argument const& error)
    {
        return std::string("rejected: ") + error.what();
    }
}

} // namespace

TEST(SchedulingPolicyTests, TheModeFollowsTheRequestAndTheDeploymentsCompactSupport)
{
    // Given each ifbSchedulingMode on a deployment with compact execution, one without, and one too small for a
    //       mixed step
    // When the policy is resolved
    // Then auto picks mixed only where a mixed step can run and otherwise prefillFirst, reporting modeSource auto
    //      with why (the deployment's reason when it has no compact execution); an explicit mode is kept with
    //      modeSource requested and no reason; compact execution is on exactly where the deployment supports it
    //      and the mode is not singleCohort; and only mixed carries step budgets
    struct Case
    {
        IfbSchedulingMode requested;
        SessionCapabilities const& capabilities;
        IfbSchedulingMode mode;
        ModeSource source;
        bool compactExecution;
        //! A prefix of the reason; empty when there is none.
        std::string reason;
    };
    for (Case const& row : {Case{kAuto, kCompact, kMixed, ModeSource::kAuto, true, "compact execution supported"},
             Case{kAuto, kPadded, kPrefillFirst, ModeSource::kAuto, false, kPadded.reason},
             Case{kAuto, kTooSmall, kPrefillFirst, ModeSource::kAuto, true, "T_max = 128 leaves no room"},
             Case{kMixed, kCompact, kMixed, ModeSource::kRequested, true, ""},
             Case{kPrefillFirst, kCompact, kPrefillFirst, ModeSource::kRequested, true, ""},
             Case{kPrefillFirst, kPadded, kPrefillFirst, ModeSource::kRequested, false, ""},
             Case{kSingleCohort, kCompact, kSingleCohort, ModeSource::kRequested, false, ""},
             Case{kSingleCohort, kPadded, kSingleCohort, ModeSource::kRequested, false, ""}})
    {
        ResolvedSchedulingPolicy const resolved = resolve(configWith(row.requested), row.capabilities);
        SCOPED_TRACE(std::string(toString(row.requested)) + " -> " + resolved.reason);
        EXPECT_EQ(resolved.mode, row.mode);
        EXPECT_EQ(resolved.modeSource, row.source);
        EXPECT_EQ(resolved.compactExecution, row.compactExecution);
        EXPECT_EQ(resolved.reason.substr(0, row.reason.size()), row.reason);
        EXPECT_EQ(resolved.reason.empty(), row.reason.empty());
        bool const mixed = row.mode == kMixed;
        EXPECT_EQ(resolved.budgets.maxNumTokens > 0, mixed);
        EXPECT_EQ(resolved.budgets.maxChunkTokens > 0, mixed);
    }
}

TEST(SchedulingPolicyTests, DefaultBudgetsDeriveFromTheEnginesBatchInputLengthAndPageSize)
{
    // Given mixed on an engine with build batch size B, input length S_in and P = 128, and no explicit budget or
    //       only an explicit maxNumTokens
    // When the policy is resolved
    // Then maxNumTokens is min(roundUp_P(512 + B), roundDown_P(T_max)) unless set, and maxChunkTokens is
    //      min(512, roundDown_P(maxNumTokens - B), max(P, roundDown_P(S_in))), with B the engine's batch size, not
    //      EngineConfig::maxBatchSize
    struct Case
    {
        int32_t batch;
        int32_t inputLength;
        int32_t explicitMaxNumTokens;
        int32_t maxNumTokens;
        int32_t maxChunkTokens;
    };
    for (Case const& row : {Case{4, 1024, 0, 640, 512}, Case{4, 128, 0, 512, 128}, Case{200, 1024, 0, 768, 512},
             Case{4, 1024, 132, 132, 128}, Case{4, 1024, 4096, 4096, 512}})
    {
        ResolvedSchedulingPolicy const resolved = resolve(
            budgets(kMixed, row.explicitMaxNumTokens, 0), engineCapabilities(true, row.batch, row.inputLength));
        EXPECT_EQ(resolved.budgets.maxNumTokens, row.maxNumTokens) << "B " << row.batch << ", S_in " << row.inputLength;
        EXPECT_EQ(resolved.budgets.maxChunkTokens, row.maxChunkTokens)
            << "B " << row.batch << ", S_in " << row.inputLength;
    }
}

TEST(SchedulingPolicyTests, EachConfigurationIsRejectedWithItsReasonOrReportsWhatItDoesNotApply)
{
    // Given B = 4, S_in = 1024, P = 128, 4 resident slots unless the row says otherwise, and the row's
    //       configuration and deployment
    // When the policy is resolved
    // Then it is rejected naming the cause when explicit mixed lacks compact execution or room for a mixed step, a
    //      budget is out of range under any mode, pause lacks the priority policy or is set with an explicit
    //      prefillFirst or singleCohort, priorityReservedSlots is outside [0, maxBatchSize - 1] or lacks the priority
    //      policy, or maxPauseSteps is negative or set without pause; otherwise it resolves with exactly one warning
    //      when auto drops pause (preemption then resolves to none and maxPauseSteps to 0), when budgets are set
    //      outside mixed, when an explicit maxChunkTokens exceeds S_in under mixed (it then resolves to S_in), or when
    //      the priority policy reserves no slot of several, and with none otherwise
    struct Case
    {
        SchedulingOptions options;
        SessionCapabilities const& capabilities;
        //! A regex the outcome contains; empty when it resolves without a warning.
        char const* outcome;
        int32_t maxResidents{kResidents};
    };
    std::vector<Case> const cases{
        {configWith(kMixed), kPadded, "rejected: ifbSchedulingMode=mixed.*gated_delta_net layers.*prefillFirst"},
        {configWith(kMixed), kTooSmall, "rejected: ifbSchedulingMode=mixed.*no room for a mixed step.*prefillFirst"},
        {budgets(kMixed, 131, 0), kCompact, "rejected: maxNumTokens 131"},
        {budgets(kMixed, 4097, 0), kCompact, "rejected: maxNumTokens 4097"},
        {budgets(kMixed, 0, 200), kCompact, "rejected: maxChunkTokens 200"},
        {budgets(kMixed, 0, 640), kCompact, "rejected: maxChunkTokens 640"},
        {budgets(kMixed, 0, -128), kCompact, "rejected: maxChunkTokens -128"},
        {budgets(kMixed, 0, 512), kCompact, ""},
        {budgets(kPrefillFirst, 131, 0), kCompact, "rejected: maxNumTokens 131"},
        {configWith(kMixed, kFcfs, kPause), kCompact,
            "rejected: preemption=pause requires schedulingPolicy=priority.*would ever pause"},
        {configWith(kPrefillFirst, kPriority, kPause, 1), kCompact,
            "rejected: preemption=pause requires ifbSchedulingMode=mixed or auto: under prefillFirst"},
        {configWith(kSingleCohort, kPriority, kPause, 1), kCompact,
            "rejected: preemption=pause requires ifbSchedulingMode=mixed or auto: under singleCohort"},
        {configWith(kMixed, kPriority, kPause, 1), kCompact, ""},
        {configWith(kMixed, kPriority, kNone, -1), kCompact, "rejected: priorityReservedSlots -1"},
        {configWith(kMixed, kPriority, kNone, 4), kCompact, "rejected: priorityReservedSlots 4"},
        {configWith(kMixed, kPriority, kNone, 3), kCompact, ""},
        {configWith(kMixed, kFcfs, kNone, 1), kCompact,
            "rejected: priorityReservedSlots requires schedulingPolicy=priority.*stay idle"},
        {pauseLimit(kPause, -1), kCompact, "rejected: maxPauseSteps -1 must be 0"},
        {pauseLimit(kNone, 2), kCompact,
            "rejected: maxPauseSteps 2 must be 0 \\(no limit\\), or positive with preemption=pause"},
        {pauseLimit(kPause, 2), kCompact, ""},
        {configWith(kAuto, kPriority, kPause, 1), kPadded,
            "preemption=pause is not applied: auto resolved to prefillFirst.*gated_delta_net layers"},
        {configWith(kAuto, kPriority, kPause, 1), kCompact, ""},
        {budgets(kPrefillFirst, 0, 256), kCompact, "maxNumTokens and maxChunkTokens are ignored.*prefillFirst"},
        {budgets(kAuto, 0, 256), kPadded, "maxNumTokens and maxChunkTokens are ignored.*prefillFirst"},
        {budgets(kSingleCohort, 0, 256), kCompact, "maxNumTokens and maxChunkTokens are ignored.*singleCohort"},
        {budgets(kMixed, 1024, 0), kCompact, ""},
        {budgets(kMixed, 0, 512), kShortInput,
            "maxChunkTokens 512 exceeds the engine's input length S_in = 256.*using 256"},
        {budgets(kMixed, 0, 256), kShortInput, ""},
        {budgets(kPrefillFirst, 0, 512), kShortInput, "maxNumTokens and maxChunkTokens are ignored.*prefillFirst"},
        {configWith(kMixed, kPriority), kCompact, "priorityReservedSlots=0: while all 4 slots"},
        {configWith(kMixed, kPriority), kCompact, "", 1},
        {configWith(kMixed), kCompact, ""},
    };
    for (size_t row = 0; row < cases.size(); ++row)
    {
        SCOPED_TRACE("row " + std::to_string(row));
        std::string const outcome = outcomeOf(cases[row].options, cases[row].capabilities, cases[row].maxResidents);
        if (*cases[row].outcome == '\0')
        {
            EXPECT_EQ(outcome, "");
        }
        else
        {
            EXPECT_THAT(outcome, ::testing::ContainsRegex(cases[row].outcome));
        }
        EXPECT_EQ(outcome.find(" | "), std::string::npos) << "warned more than once";
    }
    EXPECT_EQ(resolve(configWith(kAuto, kPriority, kPause, 1), kPadded).preemption, kNone);
    EXPECT_EQ(resolve(configWith(kAuto, kPriority, kPause, 1), kCompact).preemption, kPause);
    EXPECT_EQ(resolve(pauseLimit(kPause, 2), kCompact).maxPauseSteps, 2);
    SchedulingOptions autoPause = pauseLimit(kPause, 2);
    autoPause.ifbSchedulingMode = kAuto;
    EXPECT_EQ(resolve(autoPause, kCompact).maxPauseSteps, 2);
    EXPECT_EQ(resolve(autoPause, kPadded).maxPauseSteps, 0);
    EXPECT_EQ(resolve(budgets(kMixed, 0, 512), kShortInput).budgets.maxChunkTokens, 256);
}
