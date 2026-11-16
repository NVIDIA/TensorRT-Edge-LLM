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

#include <cstdint>

namespace trt_edgellm
{
namespace rt
{

//! Decoder invocation kind selected by a shape-carrier extent.
enum class ExecutionPhase : int32_t
{
    kContextPrefill = 1,
    kContextChunk = 2,
    kAutoregressiveDecode = 3,
    kSpecDraftProposal = 4,
    kSpecTargetVerify = 5,
    kDiffusionDenoise = 6,
    kDiffusionCommit = 7,
    //! Compact layout: prefill rows first, then one row per decode sequence, without padding (0 <= Nctx <= N;
    //! Nctx = 0 is a pure decode step with T_exec = N). Only plugins that declare compact layout support accept it.
    kMixedPrefillDecode = 8,
};

enum class SpecVerifyLayout : int32_t
{
    kChain,
    kTree,
};

constexpr int64_t specVerifyLayoutExtent(SpecVerifyLayout layout) noexcept
{
    return layout == SpecVerifyLayout::kChain ? 1 : 2;
}

constexpr bool isSpecVerifyLayoutExtentValid(ExecutionPhase phase, int64_t extent) noexcept
{
    return phase == ExecutionPhase::kSpecTargetVerify ? extent == 1 || extent == 2 : extent == 0;
}

constexpr bool isExecutionPhaseExtent(int64_t extent) noexcept
{
    return extent >= static_cast<int64_t>(ExecutionPhase::kContextPrefill)
        && extent <= static_cast<int64_t>(ExecutionPhase::kMixedPrefillDecode);
}

static_assert(static_cast<int32_t>(ExecutionPhase::kContextPrefill) == 1);
static_assert(static_cast<int32_t>(ExecutionPhase::kContextChunk) == 2);
static_assert(static_cast<int32_t>(ExecutionPhase::kAutoregressiveDecode) == 3);
static_assert(static_cast<int32_t>(ExecutionPhase::kSpecDraftProposal) == 4);
static_assert(static_cast<int32_t>(ExecutionPhase::kSpecTargetVerify) == 5);
static_assert(static_cast<int32_t>(ExecutionPhase::kDiffusionDenoise) == 6);
static_assert(static_cast<int32_t>(ExecutionPhase::kDiffusionCommit) == 7);
static_assert(static_cast<int32_t>(ExecutionPhase::kMixedPrefillDecode) == 8);

} // namespace rt
} // namespace trt_edgellm
