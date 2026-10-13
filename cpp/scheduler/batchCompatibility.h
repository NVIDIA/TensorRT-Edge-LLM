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

#include "runtime/llmRuntimeUtils.h"

#include <string>

namespace trt_edgellm
{
namespace rt
{
namespace scheduler
{

//! Whether two requests may occupy the same forward pass.
//!
//! Some request fields configure the engine for the whole step rather than for one sequence: a
//! single temperature is applied to the batch, one set of LoRA weights is bound, one decoder is
//! selected. Two requests that disagree on any of those cannot be batched, and admitting them
//! together would silently give one of them the other's settings.
//!
//! The classification is closed by convention: every field of LLMGenerationRequest is either in
//! the compared list or in the exempt list beside it, and a new field must be placed in one of the
//! two when it is added -- LLMGenerationRequest's own declaration points here for that reason.
//! The failure mode this guards is silent staleness: a new batch-relevant field that nobody
//! classifies is a check that keeps passing while one request quietly generates with another's
//! settings.
//!
//! Exempt fields fall into two groups, and nothing else is exempt:
//!   - per-sequence payload: the prompts themselves, their pre-tokenized form, and the stream each
//!     result is delivered on;
//!   - preprocessing that has already happened by the time a request is admitted, so it cannot
//!     affect a shared step: chat templating and generation-prompt insertion.
//! Audio generation and hidden-state capture are NOT exempt: they are compared (see the list in
//! batchCompatibility.cpp), so enabling one never silently shares a step with a request that has
//! not.
class BatchCompatibility
{
public:
    //! @brief True if @p candidate can join a step that is already running @p resident.
    static bool compatible(LLMGenerationRequest const& resident, LLMGenerationRequest const& candidate) noexcept;

    //! @brief The first field that differs, for diagnostics. Empty when the two are compatible.
    //!
    //! Reported by name because "incompatible request" alone leaves the caller guessing which of
    //! twenty fields it got wrong.
    static std::string firstDifference(
        LLMGenerationRequest const& resident, LLMGenerationRequest const& candidate) noexcept;
};

} // namespace scheduler
} // namespace rt
} // namespace trt_edgellm
