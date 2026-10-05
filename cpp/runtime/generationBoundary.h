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

#include "runtime/state/decodingInferenceContext.h"

#include <cstdint>
#include <functional>
#include <unordered_map>

namespace trt_edgellm
{
namespace rt
{

//! What a running generation batch exposes to whoever schedules work into it.
//!
//! The generation loop is owned by the runtime; admission is owned by a scheduler. This is the
//! seam between them: at every step boundary the loop hands its driver this view, through which a
//! new sequence may join the batch and finished ones may be collected -- without the scheduler
//! seeing the loop's internals, and without the runtime knowing scheduling policy exists.
//!
//! Narrow on purpose. Everything else about the batch (which slot runs where, how prefill is
//! seated, when eviction compacts) stays the loop's private business, so a scheduler cannot come
//! to depend on it.
//!
//! Transitional, and internal to the engine/runtime pair: the callback shape means the runtime
//! invokes scheduling policy from inside its own loop, which is the inverse of the intended
//! control plane. It exists to keep the initial runtime changes reviewable and is not exposed through
//! any public surface (pybind and the server never see it). The replacement is a scheduler-owned
//! step loop over a typed runtime stepper (admit/step/retire returning typed results), at which
//! point this hook is deleted -- do not build new contracts on it.
//! What became of an admission attempt at a step boundary.
enum class AdmitDecision : uint8_t
{
    //! Resident; its result arrives through the ordinary harvest.
    kAdmitted,
    //! Transient resource pressure (context-cache pages held by other leases). Nothing was
    //! modified; the caller should leave the request queued and retry at a later boundary.
    kNoCapacity,
    //! Admission or its prefill failed. The sequence is terminal from birth inside the batch and
    //! its (empty, error) result surfaces through takeCompletedAtOrAbove like any other.
    kFailed,
};

class GenerationBoundary
{
public:
    //! @brief Join one new sequence to the running batch and prefill it.
    //! @throws std::runtime_error for seeds the batch rejects outright; nothing is modified then.
    virtual AdmitDecision admitSequence(SlotSeed seed) = 0;

    //! @brief Admit a caller-level request: tokenization included, so a scheduler never builds a
    //!        SlotSeed by hand or holds a tokenizer.
    //!
    //! Whether the request may share this batch at all -- sampling parameters, adapter, budgets --
    //! is the scheduler's question (BatchCompatibility), answered before calling this.
    //!
    //! @param request A single-sequence request. Its stream channel, stop strings and logit bias
    //!        ride along; its sampling parameters are ignored in favour of the batch's. Scheduler
    //!        policy keeps media requests founder-only until live admission has dedicated coverage.
    //! @throws std::runtime_error for an invalid request or an unsupported media deployment;
    //!         nothing is modified.
    virtual AdmitDecision admitRequest(LLMGenerationRequest const& request, int32_t originalIndex, RequestId requestId)
        = 0;

    //! @brief Turn one finished sequence's result into the caller-facing response, decoding
    //!        included, exactly as the founding request's own assembly would have.
    //!
    //! Lives here because the conversion needs the tokenizer and the stop-string trimming that the
    //! runtime owns; a scheduler holding raw BatchResults would either skip the text or grow its
    //! own copy of both.
    virtual LLMGenerationResponse materializeResult(
        BatchResult const& result, std::vector<std::string> const& stopStrings) const = 0;

    //! @brief Move out results of finished sequences whose original index is >= @p firstIndex.
    //!
    //! Sequences admitted mid-flight carry indices above the founding request's range; harvesting
    //! them at each boundary is what keeps them out of the founding caller's response.
    virtual std::unordered_map<int32_t, BatchResult> takeCompletedAtOrAbove(int32_t firstIndex) = 0;

    //! @brief Sequences currently resident, finished or not. Admission capacity is judged on this.
    virtual int32_t residentCount() const = 0;

protected:
    //! An interface handed out for the duration of one call; nobody owns one.
    ~GenerationBoundary() = default;
};

//! Called by the loop's driver at every step boundary, and once more after the loop ends -- a
//! sequence finishing on the last step still has a result to harvest.
using GenerationBoundaryHook = std::function<void(GenerationBoundary&)>;

} // namespace rt
} // namespace trt_edgellm
