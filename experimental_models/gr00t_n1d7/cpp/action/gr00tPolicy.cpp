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

#include "action/gr00tPolicy.h"

#include "common/gr00tBindings.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <nlohmann/json.hpp>
#include <stdexcept>

namespace trt_edgellm
{
namespace gr00t
{

namespace
{

//! numpy's isclose default tolerances, which decide whether a degenerate joint
//! group normalizes to zero or divides by its range.
bool isClose(float a, float b) noexcept
{
    constexpr float kRelTol = 1e-5F;
    constexpr float kAbsTol = 1e-8F;
    return std::fabs(a - b) <= kAbsTol + kRelTol * std::fabs(b);
}

Gr00tNormalization parseNormalization(nlohmann::json const& node, char const* part)
{
    Gr00tNormalization out;
    out.keys = node.at("keys").get<std::vector<std::string>>();
    out.dims = node.at("dims").get<std::vector<int32_t>>();
    out.min = node.at("min").get<std::vector<float>>();
    out.max = node.at("max").get<std::vector<float>>();
    if (out.min.size() != out.max.size() || out.min.empty())
    {
        throw std::runtime_error(std::string("GR00T policy.json ") + part + " bounds are inconsistent");
    }
    return out;
}

} // namespace

Gr00tPolicy::Gr00tPolicy(std::string const& engineDir)
{
    nlohmann::json const contract
        = loadJsonFile(std::filesystem::path(engineDir) / "action" / "policy.json", "policy contract");
    requireIdentity(contract, "action/policy.json");

    mEmbodimentTag = contract.at("embodiment_tag").get<std::string>();
    mClipOutliers = contract.value("clip_outliers", true);
    mState = parseNormalization(contract.at("state"), "state");
    mAction = parseNormalization(contract.at("action"), "action");
    mHorizon = contract.at("action").at("horizon").get<int32_t>();
    mStateHistory = contract.at("state").value("history", 1);
    mVideoKeys = contract.value("video_keys", std::vector<std::string>{});
    if (contract.contains("vision"))
    {
        auto const& vision = contract.at("vision");
        mFormalizeLanguage = vision.value("formalize_language", true);
        mVision.shortestEdge = vision.at("shortest_edge").get<int32_t>();
        mVision.cropFraction = vision.value("crop_fraction", 1.0F);
        mVision.tokensPerView = vision.value("tokens_per_view", 0);
    }
}

std::string Gr00tPolicy::formalizeInstruction(std::string const& instruction) const
{
    if (!mFormalizeLanguage)
    {
        return instruction;
    }
    std::string out;
    out.reserve(instruction.size());
    for (unsigned char const character : instruction)
    {
        // The reference keeps [\w\s] only, on the lower-cased string.
        if (std::isalnum(character) != 0 || character == '_' || std::isspace(character) != 0)
        {
            out.push_back(static_cast<char>(std::tolower(character)));
        }
    }
    return out;
}

std::vector<float> Gr00tPolicy::normalizeState(std::vector<float> const& state, int32_t paddedDim) const
{
    int32_t const width = mState.totalDim();
    if (static_cast<int32_t>(state.size()) != width * mStateHistory)
    {
        throw std::runtime_error("GR00T state must be " + std::to_string(width * mStateHistory) + " values ("
            + std::to_string(mStateHistory) + " x " + std::to_string(width) + ")");
    }
    if (paddedDim < width)
    {
        throw std::runtime_error("GR00T state_dim is narrower than the embodiment's state");
    }

    std::vector<float> out(static_cast<size_t>(mStateHistory) * paddedDim, 0.0F);
    for (int32_t step = 0; step < mStateHistory; ++step)
    {
        for (int32_t i = 0; i < width; ++i)
        {
            float const low = mState.min[i];
            float const high = mState.max[i];
            // A degenerate group normalizes to zero rather than dividing by ~0.
            float value = 0.0F;
            if (!isClose(high, low))
            {
                value = 2.0F * (state[static_cast<size_t>(step) * width + i] - low) / (high - low) - 1.0F;
                if (mClipOutliers)
                {
                    value = std::clamp(value, -1.0F, 1.0F);
                }
            }
            out[static_cast<size_t>(step) * paddedDim + i] = value;
        }
    }
    return out;
}

std::vector<float> Gr00tPolicy::unnormalizeAction(
    std::vector<float> const& chunk, int32_t modelHorizon, int32_t actionDim) const
{
    int32_t const width = mAction.totalDim();
    if (static_cast<int64_t>(chunk.size()) < static_cast<int64_t>(modelHorizon) * actionDim)
    {
        throw std::runtime_error("GR00T action chunk is smaller than [horizon, action_dim]");
    }
    if (mHorizon > modelHorizon || width > actionDim)
    {
        throw std::runtime_error("GR00T policy contract exceeds the action graph's output shape");
    }

    std::vector<float> out(static_cast<size_t>(mHorizon) * width);
    for (int32_t step = 0; step < mHorizon; ++step)
    {
        for (int32_t i = 0; i < width; ++i)
        {
            float const low = mAction.min[i];
            float const value = std::clamp(chunk[static_cast<size_t>(step) * actionDim + i], -1.0F, 1.0F);
            out[static_cast<size_t>(step) * width + i] = (value + 1.0F) * 0.5F * (mAction.max[i] - low) + low;
        }
    }
    return out;
}

} // namespace gr00t
} // namespace trt_edgellm
