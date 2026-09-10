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

InternVLAN1DualSystemDriver::InternVLAN1DualSystemDriver(InternVLAN1DualSystemState& state, Planner planner)
    : mState(state)
    , mPlanner(std::move(planner))
{
    mThread = std::thread(&InternVLAN1DualSystemDriver::run, this);
}

InternVLAN1DualSystemDriver::~InternVLAN1DualSystemDriver() noexcept
{
    stop();
}

void InternVLAN1DualSystemDriver::requestReplan(int64_t observationIndex)
{
    {
        std::lock_guard<std::mutex> const guard(mMutex);
        // Replace rather than queue: a pending request for an older observation is already
        // obsolete once a newer one arrives.
        mPending = observationIndex;
    }
    mWakeCv.notify_one();
}

void InternVLAN1DualSystemDriver::waitIdle()
{
    std::unique_lock<std::mutex> lock(mMutex);
    // mStop is part of the predicate: once the planner thread has exited, nothing will ever
    // notify mIdleCv again, and a waiter without this clause would block forever.
    mIdleCv.wait(lock, [this] { return mStop || (!mBusy && mPending < 0); });
}

void InternVLAN1DualSystemDriver::stop() noexcept
{
    {
        std::lock_guard<std::mutex> const guard(mMutex);
        if (mStop)
        {
            return;
        }
        mStop = true;
    }
    mWakeCv.notify_all();
    mIdleCv.notify_all();
    if (mThread.joinable())
    {
        mThread.join();
    }
}

int64_t InternVLAN1DualSystemDriver::plansCompleted() const noexcept
{
    std::lock_guard<std::mutex> const guard(mMutex);
    return mCompleted;
}

void InternVLAN1DualSystemDriver::run()
{
    while (true)
    {
        int64_t observationIndex = -1;
        {
            std::unique_lock<std::mutex> lock(mMutex);
            mWakeCv.wait(lock, [this] { return mStop || mPending >= 0; });
            if (mStop)
            {
                return;
            }
            observationIndex = mPending;
            mPending = -1;
            mBusy = true;
        }

        // The planner runs unlocked. Holding the mutex across it would make a slow plan block
        // requestReplan, which is exactly the stall this class exists to prevent.
        InternVLAN1DualSystemState::Plan plan = mPlanner(observationIndex);
        plan.observationIndex = observationIndex;
        mState.publish(std::move(plan));

        {
            std::lock_guard<std::mutex> const guard(mMutex);
            mBusy = false;
            ++mCompleted;
        }
        mIdleCv.notify_all();
    }
}

} // namespace internvla_n1
} // namespace trt_edgellm
