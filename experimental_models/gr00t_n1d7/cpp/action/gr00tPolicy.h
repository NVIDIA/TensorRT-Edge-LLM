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

#include <string>
#include <vector>

namespace trt_edgellm
{
namespace gr00t
{

//! \brief Image geometry the eval transform reproduces.
struct Gr00tVision
{
    int32_t shortestEdge{0};
    float cropFraction{1.0F};
    int32_t tokensPerView{0};
};

//! \brief One modality's concatenated normalization bounds.
struct Gr00tNormalization
{
    std::vector<std::string> keys;
    std::vector<int32_t> dims;
    std::vector<float> min;
    std::vector<float> max;

    int32_t totalDim() const noexcept
    {
        return static_cast<int32_t>(min.size());
    }
};

//! \brief Robot-facing conversion around the GR00T action head: a robot state into the
//! model's normalized, zero-padded space and the chunk back out, from the bounds in
//! ``action/policy.json``. Absolute action representations only.
class Gr00tPolicy
{
public:
    //! \param engineDir Directory holding ``action/policy.json``.
    explicit Gr00tPolicy(std::string const& engineDir);

    //! \brief Normalize a robot state and zero-pad it to the model width.
    //! \param state Concatenated joint groups in ``policy.json`` order.
    //! \param paddedDim The action graph's ``state_dim``.
    std::vector<float> normalizeState(std::vector<float> const& state, int32_t paddedDim) const;

    //! \brief Unnormalize a predicted chunk and drop the padding.
    //! \param chunk Row-major [modelHorizon, actionDim] from the action graph.
    //! \return Row-major [horizon(), action().totalDim()] in robot units.
    std::vector<float> unnormalizeAction(
        std::vector<float> const& chunk, int32_t modelHorizon, int32_t actionDim) const;

    //! \brief Lower-case and strip punctuation, as the reference does before
    //! the instruction reaches the tokenizer.
    std::string formalizeInstruction(std::string const& instruction) const;

    Gr00tNormalization const& state() const noexcept
    {
        return mState;
    }

    Gr00tVision const& vision() const noexcept
    {
        return mVision;
    }

    Gr00tNormalization const& action() const noexcept
    {
        return mAction;
    }

    //! \brief Actions the policy emits per request; shorter than the model's horizon.
    int32_t horizon() const noexcept
    {
        return mHorizon;
    }

    std::string const& embodimentTag() const noexcept
    {
        return mEmbodimentTag;
    }

    std::vector<std::string> const& videoKeys() const noexcept
    {
        return mVideoKeys;
    }

private:
    Gr00tNormalization mState;
    Gr00tNormalization mAction;
    std::vector<std::string> mVideoKeys;
    std::string mEmbodimentTag;
    int32_t mHorizon{0};
    int32_t mStateHistory{1};
    bool mClipOutliers{true};
    bool mFormalizeLanguage{true};
    Gr00tVision mVision;
};

} // namespace gr00t
} // namespace trt_edgellm
