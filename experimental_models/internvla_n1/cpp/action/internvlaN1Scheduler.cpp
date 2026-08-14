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

#include "action/internvlaN1Scheduler.h"

#include <stdexcept>
#include <string>

namespace trt_edgellm
{
namespace internvla_n1
{

InternVLAN1Scheduler::InternVLAN1Scheduler(int32_t numSteps)
    : mNumSteps(numSteps)
{
    if (numSteps <= 0)
    {
        throw std::invalid_argument("InternVLAN1Scheduler: numSteps must be positive, got " + std::to_string(numSteps));
    }

    // sigma_i = 1 - i / N for i in [0, N), then a terminal 0. Computed from the index rather
    // than accumulated, so the terminal value is exactly 0 instead of N roundings away from it.
    mSigmas.reserve(static_cast<size_t>(numSteps) + 1U);
    mTimesteps.reserve(static_cast<size_t>(numSteps));
    for (int32_t i = 0; i < numSteps; ++i)
    {
        float const sigma = 1.0F - static_cast<float>(i) / static_cast<float>(numSteps);
        mSigmas.push_back(sigma);
        mTimesteps.push_back(sigma * kNumTrainTimesteps);
    }
    mSigmas.push_back(0.0F);
}

float InternVLAN1Scheduler::timestepAt(int32_t stepIdx) const
{
    if (stepIdx < 0 || stepIdx >= mNumSteps)
    {
        throw std::out_of_range("InternVLAN1Scheduler::timestepAt: step " + std::to_string(stepIdx) + " outside [0, "
            + std::to_string(mNumSteps) + ")");
    }
    return mTimesteps[static_cast<size_t>(stepIdx)];
}

float InternVLAN1Scheduler::sigmaAt(int32_t stepIdx) const
{
    if (stepIdx < 0 || stepIdx > mNumSteps)
    {
        throw std::out_of_range("InternVLAN1Scheduler::sigmaAt: step " + std::to_string(stepIdx) + " outside [0, "
            + std::to_string(mNumSteps) + "]");
    }
    return mSigmas[static_cast<size_t>(stepIdx)];
}

float InternVLAN1Scheduler::dtAt(int32_t stepIdx) const
{
    return sigmaAt(stepIdx + 1) - sigmaAt(stepIdx);
}

} // namespace internvla_n1
} // namespace trt_edgellm
