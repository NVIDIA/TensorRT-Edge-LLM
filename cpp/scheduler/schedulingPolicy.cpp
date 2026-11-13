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

#include "common/checkMacros.h"
#include "common/cudaUtils.h"

#include <algorithm>
#include <stdexcept>

namespace trt_edgellm
{
namespace rt
{
namespace scheduler
{

namespace
{

template <typename Enum>
struct NamedValue
{
    Enum value;
    char const* name;
};

//! The one name table of each enum, and the option it configures for parse errors.
template <typename Enum>
struct EnumNames;

template <>
struct EnumNames<IfbSchedulingMode>
{
    static constexpr char const* kOption = "ifbSchedulingMode";
    static constexpr NamedValue<IfbSchedulingMode> kValues[]
        = {{IfbSchedulingMode::kAuto, "auto"}, {IfbSchedulingMode::kMixed, "mixed"},
            {IfbSchedulingMode::kPrefillFirst, "prefillFirst"}, {IfbSchedulingMode::kSingleCohort, "singleCohort"}};
};

template <>
struct EnumNames<SchedulingPolicy>
{
    static constexpr char const* kOption = "schedulingPolicy";
    static constexpr NamedValue<SchedulingPolicy> kValues[]
        = {{SchedulingPolicy::kFcfs, "fcfs"}, {SchedulingPolicy::kPriority, "priority"}};
};

template <>
struct EnumNames<PreemptionMode>
{
    static constexpr char const* kOption = "preemption";
    static constexpr NamedValue<PreemptionMode> kValues[]
        = {{PreemptionMode::kNone, "none"}, {PreemptionMode::kPause, "pause"}};
};

template <>
struct EnumNames<ModeSource>
{
    static constexpr NamedValue<ModeSource> kValues[]
        = {{ModeSource::kRequested, "requested"}, {ModeSource::kAuto, "auto"}};
};

template <typename Enum>
char const* nameOf(Enum value) noexcept
{
    for (NamedValue<Enum> const& entry : EnumNames<Enum>::kValues)
    {
        if (entry.value == value)
        {
            return entry.name;
        }
    }
    return "unknown";
}

//! B, S_in and P of the engine; T_max = B x S_in.
struct EngineLimits
{
    int64_t batch;
    int64_t inputLength;
    int64_t page;

    int64_t tokenLimit() const noexcept
    {
        return batch * inputLength;
    }

    //! Every decode row plus one page of prefill: the least a mixed step with decode rows must hold for a pending
    //! prompt to progress.
    int64_t minStepTokens() const noexcept
    {
        return batch + page;
    }

    std::string describe() const
    {
        return "B = " + std::to_string(batch) + " (engine batch size), S_in = " + std::to_string(inputLength)
            + ", T_max = B x S_in = " + std::to_string(tokenLimit()) + ", P = " + std::to_string(page)
            + " (KV page size)";
    }
};

//! Range-checks the explicit budgets in every mode, so whether a configuration starts does not depend on the mode
//! auto picks, and derives the defaults.
StepBudgets computeStepBudgets(SchedulingOptions const& options, EngineLimits const& limits)
{
    if (options.maxNumTokens != 0
        && (options.maxNumTokens < limits.minStepTokens() || options.maxNumTokens > limits.tokenLimit()))
    {
        throw std::invalid_argument("maxNumTokens " + std::to_string(options.maxNumTokens)
            + " must be in [B + P, T_max] = [" + std::to_string(limits.minStepTokens()) + ", "
            + std::to_string(limits.tokenLimit()) + "] (" + limits.describe() + "), or 0 for the default.");
    }
    int64_t const maxNumTokens = options.maxNumTokens != 0
        ? int64_t{options.maxNumTokens}
        : std::min(static_cast<int64_t>(divUp(512 + limits.batch, limits.page)) * limits.page,
              limits.tokenLimit() / limits.page * limits.page);

    int64_t const chunkLimit = maxNumTokens - limits.batch;
    if (options.maxChunkTokens != 0
        && (options.maxChunkTokens % limits.page != 0 || options.maxChunkTokens < limits.page
            || options.maxChunkTokens > chunkLimit))
    {
        throw std::invalid_argument("maxChunkTokens " + std::to_string(options.maxChunkTokens)
            + " must be a multiple of P in [P, maxNumTokens - B] = [" + std::to_string(limits.page) + ", "
            + std::to_string(chunkLimit) + "] (maxNumTokens = " + std::to_string(maxNumTokens) + ", "
            + limits.describe() + "), or 0 for the default.");
    }
    int64_t const requestedChunk = options.maxChunkTokens != 0
        ? int64_t{options.maxChunkTokens}
        : std::min<int64_t>(512, chunkLimit / limits.page * limits.page);
    // One prompt chunk holds at most S_in tokens; a larger budget would be capped by the planner unseen.
    int64_t const chunkCap = std::max(limits.page, limits.inputLength / limits.page * limits.page);
    return StepBudgets{static_cast<int32_t>(maxNumTokens), static_cast<int32_t>(std::min(chunkCap, requestedChunk))};
}

//! Why a mixed step cannot run on this deployment, complete for display; empty when it can.
std::string mixedBlocker(SessionCapabilities const& capabilities, EngineLimits const& limits, StepBudgets budgets)
{
    if (!capabilities.compact)
    {
        return capabilities.reason.empty() ? std::string("compact execution not available") : capabilities.reason;
    }
    if (budgets.maxNumTokens < limits.minStepTokens())
    {
        return "T_max = " + std::to_string(limits.tokenLimit()) + " leaves no room for a mixed step of B + P = "
            + std::to_string(limits.minStepTokens()) + " tokens (" + limits.describe() + ")";
    }
    return {};
}

//! Sets mode, modeSource and reason. @throws std::invalid_argument when mixed is requested where it cannot run.
void resolveMode(
    IfbSchedulingMode requested, std::string const& blocker, bool compactCapable, ResolvedSchedulingPolicy& resolved)
{
    if (requested == IfbSchedulingMode::kAuto)
    {
        resolved.mode = blocker.empty() ? IfbSchedulingMode::kMixed : IfbSchedulingMode::kPrefillFirst;
        resolved.modeSource = ModeSource::kAuto;
        resolved.reason = blocker.empty() ? std::string("compact execution supported") : blocker;
        return;
    }
    if (requested == IfbSchedulingMode::kMixed && !blocker.empty())
    {
        throw std::invalid_argument(std::string("ifbSchedulingMode=mixed is not supported by this ")
            + (compactCapable ? "engine: " : "deployment: ") + blocker
            + ". Use ifbSchedulingMode=prefillFirst (or auto)"
            + (compactCapable ? ", or rebuild with a larger max input length." : "."));
    }
    resolved.mode = requested;
    resolved.modeSource = ModeSource::kRequested;
}

//! What @p options asked for that @p resolved does not apply as given, one sentence each.
std::vector<std::string> collectWarnings(SchedulingOptions const& options, int32_t maxResidents,
    EngineLimits const& limits, ResolvedSchedulingPolicy const& resolved)
{
    std::vector<std::string> warnings;
    if (options.preemption != resolved.preemption)
    {
        warnings.push_back("preemption=pause is not applied: auto resolved to " + std::string(toString(resolved.mode))
            + " (" + resolved.reason + "), where decoding requests already wait while a prompt is prefilled.");
    }
    if (resolved.mode != IfbSchedulingMode::kMixed && (options.maxNumTokens != 0 || options.maxChunkTokens != 0))
    {
        warnings.push_back(
            std::string(
                "maxNumTokens and maxChunkTokens are ignored: they budget mixed steps, and the mode resolved to ")
            + toString(resolved.mode) + ".");
    }
    if (resolved.mode == IfbSchedulingMode::kMixed && options.maxChunkTokens > resolved.budgets.maxChunkTokens)
    {
        warnings.push_back("maxChunkTokens " + std::to_string(options.maxChunkTokens)
            + " exceeds the engine's input length S_in = " + std::to_string(limits.inputLength)
            + ", the most one prompt chunk holds; using " + std::to_string(resolved.budgets.maxChunkTokens) + ".");
    }
    if (options.schedulingPolicy == SchedulingPolicy::kPriority && options.priorityReservedSlots == 0
        && maxResidents >= 2)
    {
        warnings.push_back("schedulingPolicy=priority with priorityReservedSlots=0: while all "
            + std::to_string(maxResidents)
            + " slots are busy, a request with priority < 0 waits for a running request to finish. To start one "
              "at once, build the engine for maxBatchSize + k and reserve k slots with priorityReservedSlots=k.");
    }
    return warnings;
}

} // namespace

void validateSchedulingOptions(SchedulingOptions const& options, int32_t maxResidents)
{
    if (options.priorityReservedSlots < 0 || options.priorityReservedSlots >= maxResidents)
    {
        throw std::invalid_argument("priorityReservedSlots " + std::to_string(options.priorityReservedSlots)
            + " must be in [0, maxBatchSize - 1] = [0, " + std::to_string(maxResidents - 1)
            + "]: at least one slot must stay open to requests with priority >= 0.");
    }
    bool const pause = options.preemption == PreemptionMode::kPause;
    if ((pause || options.priorityReservedSlots > 0) && options.schedulingPolicy != SchedulingPolicy::kPriority)
    {
        throw std::invalid_argument(std::string(pause ? "preemption=pause" : "priorityReservedSlots")
            + " requires schedulingPolicy=priority: under fcfs requests are admitted in arrival order whatever their "
              "priority, so "
            + (pause ? "no resident would ever pause." : "the reserved slots would only stay idle."));
    }
    if (options.maxPauseSteps < 0 || (options.maxPauseSteps > 0 && !pause))
    {
        throw std::invalid_argument("maxPauseSteps " + std::to_string(options.maxPauseSteps)
            + " must be 0 (no limit), or positive with preemption=pause, the only setting that pauses a resident.");
    }
    if (pause
        && (options.ifbSchedulingMode == IfbSchedulingMode::kPrefillFirst
            || options.ifbSchedulingMode == IfbSchedulingMode::kSingleCohort))
    {
        throw std::invalid_argument(std::string("preemption=pause requires ifbSchedulingMode=mixed or auto: under ")
            + toString(options.ifbSchedulingMode)
            + " a pending prompt already runs while decoding requests wait. Use preemption=none.");
    }
}

ResolvedSchedulingPolicy resolveSchedulingPolicy(
    SchedulingOptions const& options, int32_t maxResidents, SessionCapabilities const& capabilities)
{
    EngineLimits const limits{capabilities.maxBatchSize, capabilities.maxInputLength, capabilities.kvPageSize};
    ELLM_CHECK(limits.batch > 0 && limits.inputLength > 0 && limits.page > 0,
        "the session reported invalid engine limits (" + limits.describe() + ")");
    validateSchedulingOptions(options, maxResidents);
    StepBudgets const budgets = computeStepBudgets(options, limits);

    ResolvedSchedulingPolicy resolved;
    resolveMode(options.ifbSchedulingMode, mixedBlocker(capabilities, limits, budgets), capabilities.compact, resolved);
    bool const mixed = resolved.mode == IfbSchedulingMode::kMixed;
    resolved.schedulingPolicy = options.schedulingPolicy;
    // Only auto reaches a mode other than mixed with pause (validateOptions refuses the explicit ones). Under
    // prefillFirst a pending prompt already runs without decode rows, so urgent prompts are still served first;
    // failing the launch would make the same options work on one deployment and not on another.
    resolved.preemption = mixed ? options.preemption : PreemptionMode::kNone;
    resolved.maxPauseSteps = resolved.preemption == PreemptionMode::kPause ? options.maxPauseSteps : 0;
    resolved.budgets = mixed ? budgets : StepBudgets{};
    resolved.priorityReservedSlots = options.priorityReservedSlots;
    resolved.compactExecution = capabilities.compact && resolved.mode != IfbSchedulingMode::kSingleCohort;
    resolved.warnings = collectWarnings(options, maxResidents, limits, resolved);
    return resolved;
}

char const* executionName(ResolvedSchedulingPolicy const& policy) noexcept
{
    return policy.compactExecution ? "compact" : "single-cohort";
}

char const* toString(IfbSchedulingMode mode) noexcept
{
    return nameOf(mode);
}

char const* toString(SchedulingPolicy policy) noexcept
{
    return nameOf(policy);
}

char const* toString(PreemptionMode mode) noexcept
{
    return nameOf(mode);
}

char const* toString(ModeSource source) noexcept
{
    return nameOf(source);
}

template <typename Enum>
Enum parseEnum(std::string const& name)
{
    std::string accepted;
    for (NamedValue<Enum> const& entry : EnumNames<Enum>::kValues)
    {
        if (name == entry.name)
        {
            return entry.value;
        }
        accepted += (accepted.empty() ? "" : ", ") + std::string(entry.name);
    }
    throw std::invalid_argument(
        "unknown " + std::string(EnumNames<Enum>::kOption) + " '" + name + "'; expected one of " + accepted);
}

template IfbSchedulingMode parseEnum<IfbSchedulingMode>(std::string const& name);
template SchedulingPolicy parseEnum<SchedulingPolicy>(std::string const& name);
template PreemptionMode parseEnum<PreemptionMode>(std::string const& name);

} // namespace scheduler
} // namespace rt
} // namespace trt_edgellm
