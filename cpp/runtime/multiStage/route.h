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

#include "runtime/multiStage/stageConfig.h"
#include <cstdint>
#include <string>
#include <vector>

namespace trt_edgellm
{
namespace rt
{
namespace multiStage
{

//! What this request actually supplies, built once in submit() from the payload's bound names.
//! No stage may re-derive the modality set, or two stages can disagree on it.
struct RequestCapabilities
{
    //! Fully-qualified request keys, e.g. "request:messages", "request:audio".
    std::vector<std::string> suppliedRequestKeys;

    bool supplies(std::string const& requestKey) const;
};

//! One edge that survived resolution.
struct RouteEdge
{
    int32_t producer{-1}; //!< Stage index. -1 means the value comes from the request.
    int32_t consumer{-1};
    std::string port;      //!< Producer's port id, or the request field.
    std::string inputName; //!< Consumer's input name.
    Transport transport{Transport::kDeviceTensor};
    bool crossLane{false};
};

/*!
 * @brief Which stages run for one request, decided once and then frozen.
 *
 * Optional inputs and optional outputs are the same computation. Two passes,
 * each exactly once -- there is no fixed point to iterate towards:
 *
 *   forward   available: topological order, prune a stage whose source is gone
 *                        according to its OnMissing. Only propagates downstream.
 *   backward  required:  reverse order from the terminals the caller attached a
 *                        sink to. Only propagates upstream.
 *   active  = available AND required, then two checks.
 *
 * The backward pass cannot undo the forward one: a stage the backward pass drops
 * is by construction not upstream of any required stage. Iteration would only be
 * needed if a stage's survival depended on something downstream of it -- a cycle,
 * or a "required only when X is active" input. The topology has neither.
 */
class Route
{
public:
    Route() = default;

    /*!
     * @brief Resolve the active subgraph for one request.
     *
     * @param config    Already validated (see multiStage::validate).
     * @param wanted    Terminal port ids the caller attached a sink to.
     * @param available What the request supplies.
     * @param error     Filled with a human-readable reason when this returns false.
     */
    static bool resolve(MultiStageConfig const& config, std::vector<std::string> const& wanted,
        RequestCapabilities const& available, Route& out, std::string& error);

    //! Topological order of the active stages.
    std::vector<int32_t> const& order() const noexcept
    {
        return mOrder;
    }

    std::vector<RouteEdge> const& edges() const noexcept
    {
        return mEdges;
    }

    bool isActive(int32_t stageIndex) const noexcept;
    //! A Route holds indices, not ids; resolve one with MultiStageConfig::stageIndex() first.
    bool isActive(std::string const& stageId) const = delete;

    //! Two requests may share a batch only when these are equal.
    uint64_t fingerprint() const noexcept
    {
        return mFingerprint;
    }

    //! Inputs of active stages whose source is absent from this route and whose OnMissing is kSkip.
    std::vector<std::pair<int32_t, std::string>> const& skippedInputs() const noexcept
    {
        return mSkipped;
    }

private:
    std::vector<int32_t> mOrder;
    std::vector<RouteEdge> mEdges;
    std::vector<bool> mActive;
    std::vector<std::pair<int32_t, std::string>> mSkipped;
    uint64_t mFingerprint{0};
};

} // namespace multiStage
} // namespace rt
} // namespace trt_edgellm
