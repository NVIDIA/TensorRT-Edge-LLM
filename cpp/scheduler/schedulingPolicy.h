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

#include <cstdint>
#include <string>
#include <vector>

namespace trt_edgellm
{
namespace rt
{
namespace scheduler
{

//! How a step is composed when prompts and decoding sequences coexist.
enum class IfbSchedulingMode : uint8_t
{
    //! kMixed when the deployment supports compact execution and the step budgets fit the engine,
    //! else kPrefillFirst.
    kAuto,
    //! Decode first, then chunked prefill in the same step. Requires compact execution.
    kMixed,
    //! A step with pending prefill runs prefill only; decoding waits.
    kPrefillFirst,
    //! Single-cohort execution whatever the deployment supports: no compact steps. One
    //! admission per step boundary, its whole prompt prefilled alone, then phase-3 decode steps, so the outputs
    //! reproduce the execution without compact steps bit for bit.
    kSingleCohort,
};

//! Order in which queued requests are offered for admission.
enum class SchedulingPolicy : uint8_t
{
    //! Arrival order.
    kFcfs,
    //! (priority, arrival), lower priority value first.
    kPriority,
};

enum class PreemptionMode : uint8_t
{
    kNone,
    //! Lower-priority residents pause while a higher-priority prompt is prefilled. Mixed mode only; under auto
    //! resolved to prefillFirst it is dropped with a warning (a pending prompt already runs without decode rows).
    kPause,
};

//! How ResolvedSchedulingPolicy::mode was chosen.
enum class ModeSource : uint8_t
{
    //! The configured mode, kept as given.
    kRequested,
    //! Chosen by IfbSchedulingMode::kAuto from the deployment and the step budgets.
    kAuto,
};

//! The in-flight batching scheduling options, named as the bindings and the server name them. B, S_in and P are
//! the engine's build batch size, its input length and the KV page size; T_max = B x S_in. See
//! resolveSchedulingPolicy() for the constraints.
struct SchedulingOptions
{
    IfbSchedulingMode ifbSchedulingMode{IfbSchedulingMode::kAuto};
    //! Token budget of a mixed step with decode rows, in [B + P, T_max]. 0 selects
    //! min(roundUp_P(512 + B), roundDown_P(T_max)).
    int32_t maxNumTokens{0};
    //! Per-sequence prefill budget of a mixed step with decode rows: a multiple of P in [P, maxNumTokens - B].
    //! 0 selects min(512, roundDown_P(maxNumTokens - B)). Either is then capped at max(P, roundDown_P(S_in)), the
    //! most one prompt chunk holds.
    int32_t maxChunkTokens{0};
    SchedulingPolicy schedulingPolicy{SchedulingPolicy::kFcfs};
    PreemptionMode preemption{PreemptionMode::kNone};
    //! Slots held back for requests with priority < 0: any other request is admitted only while more than this
    //! many slots are free.
    int32_t priorityReservedSlots{0};
    //! Under PreemptionMode::kPause, the most steps in a row a resident is paused: after this many it is not paused
    //! in the next step even while a more urgent prompt is unfinished. That guarantees a decode row only to a
    //! decoding resident; one with an unfinished prompt competes for the prompt budget in priority order. 0 = no
    //! limit.
    int32_t maxPauseSteps{0};
};

//! The budgets of a mixed step with decode rows, as the planner applies them.
struct StepBudgets
{
    int32_t maxNumTokens{0};
    int32_t maxChunkTokens{0};
};

//! The policy the engine runs with; see RequestEngine::resolvedPolicy() for when it is resolved.
struct ResolvedSchedulingPolicy
{
    //! kMixed, kPrefillFirst or kSingleCohort; never kAuto.
    IfbSchedulingMode mode{IfbSchedulingMode::kPrefillFirst};
    ModeSource modeSource{ModeSource::kRequested};
    //! Why auto chose the mode; empty for a requested mode.
    std::string reason;
    SchedulingPolicy schedulingPolicy{SchedulingPolicy::kFcfs};
    //! As applied: kNone when auto resolved to prefillFirst.
    PreemptionMode preemption{PreemptionMode::kNone};
    //! Zero outside mixed, whose prefill steps are bounded by the engine (T_max tokens, S_in per sequence) instead.
    StepBudgets budgets;
    int32_t priorityReservedSlots{0};
    //! 0 = no limit, and 0 whenever preemption is kNone.
    int32_t maxPauseSteps{0};
    //! Whether the session executes compact steps.
    bool compactExecution{false};
    //! Options the engine accepted but does not apply as given, one sentence each; logged at startup.
    std::vector<std::string> warnings;
};

//! @brief Refuse what could never apply, or would leave requests waiting forever, whatever the deployment:
//!        priorityReservedSlots outside [0, maxResidents - 1]; reserved slots or pause without the priority policy; a
//!        negative maxPauseSteps, or a positive one without pause; pause with an explicit prefillFirst or
//!        singleCohort. These checks need no engine facts, so a caller can make them before the engine is built.
//! @param maxResidents EngineConfig::maxBatchSize, which bounds priorityReservedSlots.
//! @throws std::invalid_argument naming the option and the alternative.
void validateSchedulingOptions(SchedulingOptions const& options, int32_t maxResidents);

//! @brief Resolve @p options against what the deployment can execute.
//!
//! B, S_in and P come from @p capabilities; @p maxResidents (EngineConfig::maxBatchSize) bounds
//! priorityReservedSlots.
//! @throws std::invalid_argument naming the option and the alternative when an option could never apply or the
//!         deployment cannot serve it: what validateSchedulingOptions() refuses; a budget out of range (in every
//!         mode); mixed without
//!         compact execution or without room for B + P tokens. Under singleCohort, compactExecution is false
//!         whatever @p capabilities reports. ResolvedSchedulingPolicy::warnings reports what is accepted but not
//!         applied as given.
ResolvedSchedulingPolicy resolveSchedulingPolicy(
    SchedulingOptions const& options, int32_t maxResidents, SessionCapabilities const& capabilities);

//! "compact" when @p policy executes compact steps, else "single-cohort".
char const* executionName(ResolvedSchedulingPolicy const& policy) noexcept;

char const* toString(IfbSchedulingMode mode) noexcept;
char const* toString(SchedulingPolicy policy) noexcept;
char const* toString(PreemptionMode mode) noexcept;
char const* toString(ModeSource source) noexcept;

//! Parse a name toString() produces for IfbSchedulingMode, SchedulingPolicy or PreemptionMode.
//! @throws std::invalid_argument naming the option and the accepted names.
template <typename Enum>
Enum parseEnum(std::string const& name);

} // namespace scheduler
} // namespace rt
} // namespace trt_edgellm
