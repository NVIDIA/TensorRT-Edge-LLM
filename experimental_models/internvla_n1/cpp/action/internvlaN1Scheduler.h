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
#include <vector>

namespace trt_edgellm
{
namespace internvla_n1
{

//! \brief Flow-matching Euler scheduler for the InternVLA-N1 System-1 denoising loop.
//!
//! Ports the diffusers `FlowMatchEulerDiscreteScheduler` as the reference drives it:
//! `set_timesteps(N, sigmas=linspace(1.0, 1/N, N))` with the default `shift = 1.0`.
//!
//! Under that configuration the schedule collapses to something exact and closed-form --
//! `sigma_i = 1 - i / N`, with a terminal `sigma_N = 0` appended, and `timestep_i = 1000 *
//! sigma_i`. Every step therefore has the same `dt = -1 / N`. That is worth stating because the
//! general scheduler has a resolution-dependent time shift and a sigma-interpolation path, and
//! neither is reachable here; reproducing them would add code that cannot run and risk drifting
//! from the reference.
//!
//! The update is the plain Euler step on the flow parameterization:
//!
//!     x_{i+1} = x_i + (sigma_{i+1} - sigma_i) * v_i
//!
//! where `v_i` is the model output for step `i`. The scheduler is stateless with respect to the
//! sample: it owns only the schedule, so a single instance can drive several trajectories.
class InternVLAN1Scheduler
{
public:
    //! Number of training timesteps the sigmas are scaled by to form model timesteps.
    static constexpr float kNumTrainTimesteps = 1000.0F;

    //! \brief Build the schedule for `numSteps` inference steps.
    //! \throws std::invalid_argument when `numSteps` is not positive.
    explicit InternVLAN1Scheduler(int32_t numSteps);

    int32_t numInferenceSteps() const noexcept
    {
        return mNumSteps;
    }

    //! \brief Model timestep for step `stepIdx`, in [0, kNumTrainTimesteps].
    float timestepAt(int32_t stepIdx) const;

    //! \brief Noise level at step `stepIdx`. `sigmaAt(numInferenceSteps())` is the terminal 0.
    float sigmaAt(int32_t stepIdx) const;

    //! \brief The Euler coefficient for step `stepIdx`: `sigma_{i+1} - sigma_i`.
    //!
    //! Negative by construction -- the loop walks noise down to zero.
    float dtAt(int32_t stepIdx) const;

    //! \brief Full timestep schedule, in the order the loop consumes it.
    std::vector<float> const& timesteps() const noexcept
    {
        return mTimesteps;
    }

    //! \brief Sigma schedule, length `numInferenceSteps() + 1` including the terminal zero.
    std::vector<float> const& sigmas() const noexcept
    {
        return mSigmas;
    }

private:
    int32_t mNumSteps{0};
    std::vector<float> mSigmas;    //!< length mNumSteps + 1
    std::vector<float> mTimesteps; //!< length mNumSteps
};

} // namespace internvla_n1
} // namespace trt_edgellm
