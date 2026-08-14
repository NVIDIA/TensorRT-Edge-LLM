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

#include "runtime/internvlaN1DualSystem.h"

#include <utility>

namespace trt_edgellm
{
namespace internvla_n1
{

InternVLAN1DualSystemState::InternVLAN1DualSystemState(Mode mode)
    : mMode(mode)
{
}

void InternVLAN1DualSystemState::publish(Plan plan)
{
    std::lock_guard<std::mutex> const guard(mMutex);
    mPlan = std::move(plan);
    mHasPlan = true;
}

bool InternVLAN1DualSystemState::latest(Plan& out) const
{
    std::lock_guard<std::mutex> const guard(mMutex);
    if (!mHasPlan)
    {
        return false;
    }
    out = mPlan;
    return true;
}

int64_t InternVLAN1DualSystemState::stalenessAt(int64_t currentObservationIndex) const
{
    std::lock_guard<std::mutex> const guard(mMutex);
    if (!mHasPlan)
    {
        return -1;
    }
    return currentObservationIndex - mPlan.observationIndex;
}

bool InternVLAN1DualSystemState::shouldReplan(int64_t observationIndex, int64_t cadence, bool forced) const
{
    if (forced)
    {
        return true;
    }
    if (mMode == Mode::kSync)
    {
        return true;
    }
    if (cadence <= 0)
    {
        return true;
    }
    return observationIndex % cadence == 0;
}

} // namespace internvla_n1
} // namespace trt_edgellm
