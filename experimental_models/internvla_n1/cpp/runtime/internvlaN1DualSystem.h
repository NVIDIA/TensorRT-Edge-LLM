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

#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace trt_edgellm
{
namespace internvla_n1
{

//! \brief The conditioning handoff between the two systems.
//!
//! InternVLA-N1 is an asynchronous dual system: the Qwen2.5-VL planner replans at a low rate
//! while the trajectory head runs at a high one, and the head consumes **the latest plan
//! available** rather than waiting for a fresh one. This holds that shared state.
//!
//! Modelled on the reference agent (`internvla_n1_agent.py`), which runs System 2 on a
//! background thread behind three locks and a `should_infer` flag. The contract kept here is
//! the part that has to be right:
//!
//! * System 1 never blocks on System 2. If no plan has landed yet, `latest()` says so and the
//!   caller decides; it does not stall the control loop.
//! * A plan is published atomically. A half-written conditioning tensor would produce a
//!   plausible trajectory from two different plans, which is worse than a stale one.
//! * Staleness is observable. The head is entitled to run on an old plan, but a caller that
//!   cannot tell how old it is cannot enforce a safety bound.
//!
//! \note This owns synchronization only, not device memory. System 1's own context pool and
//! stream are what let the two overlap at all -- see `InternVLAN1System1Runner`.
class InternVLAN1DualSystemState
{
public:
    //! \brief How the caller drives the two systems.
    enum class Mode : int32_t
    {
        //! System 1 waits for a plan matching the current observation. Deterministic, and what
        //! an offline evaluation wants.
        kSync = 0,
        //! System 1 runs on the newest plan available. What the robot wants.
        kPartialAsync = 1,
    };

    struct Plan
    {
        std::vector<float> conditioning; //!< [2, condLen, latentDim], null row first.
        int64_t condLen{0};
        int64_t latentDim{0};
        //! Observation index this plan was computed from, so the consumer can measure staleness.
        int64_t observationIndex{-1};
    };

    explicit InternVLAN1DualSystemState(Mode mode = Mode::kPartialAsync);

    Mode mode() const noexcept
    {
        return mMode;
    }

    //! \brief Publish a plan. Replaces any previous one; System 2 output is not queued.
    //!
    //! Queueing would be wrong, not merely wasteful: a backlog means the head would steer on
    //! plans that are already superseded.
    void publish(Plan plan);

    //! \brief Copy out the latest plan.
    //! \return false when nothing has been published yet, leaving \p out untouched.
    bool latest(Plan& out) const;

    //! \brief Observations since the newest plan's own observation, or -1 if none published.
    int64_t stalenessAt(int64_t currentObservationIndex) const;

    //! \brief Whether System 2 should be run for this observation.
    //!
    //! True on the cadence, and always when \p forced -- the reference routes look-down frames
    //! through System 2 regardless of mode, because that frame is the one that establishes the
    //! pixel goal.
    bool shouldReplan(int64_t observationIndex, int64_t cadence, bool forced) const;

private:
    Mode mMode;
    mutable std::mutex mMutex;
    Plan mPlan;
    bool mHasPlan{false};
};

//! \brief Runs System 2 on its own thread so System 1 never waits for it.
//!
//! This is the half of the asynchronous contract that is not just synchronization. The
//! reference implementation puts the planner on a Python thread inside the agent; keeping it
//! there would make the runtime depend on the agent, so it lives here instead.
//!
//! The planner is injected rather than owned. What "plan" means -- which frames, which prompt,
//! which engine -- is the caller's, and hard-coding it here would tie this to one deployment.
//! What the runtime owes is the threading, the wake-up, and the guarantee that a slow planner
//! cannot stall the trajectory loop.
//!
//! Replan requests coalesce. If a request arrives while one is in flight, the newer observation
//! index replaces the pending one rather than queueing behind it -- a backlog of plans is a
//! backlog of stale plans.
class InternVLAN1DualSystemDriver
{
public:
    //! Computes a plan for an observation. Runs on the planner thread, never on the caller's.
    using Planner = std::function<InternVLAN1DualSystemState::Plan(int64_t observationIndex)>;

    InternVLAN1DualSystemDriver(InternVLAN1DualSystemState& state, Planner planner);
    ~InternVLAN1DualSystemDriver() noexcept;

    InternVLAN1DualSystemDriver(InternVLAN1DualSystemDriver const&) = delete;
    InternVLAN1DualSystemDriver& operator=(InternVLAN1DualSystemDriver const&) = delete;

    //! \brief Ask for a plan at \p observationIndex. Returns immediately.
    void requestReplan(int64_t observationIndex);

    //! \brief Block until no plan is in flight. For tests and for shutdown, not the hot loop.
    void waitIdle();

    //! \brief Stop the planner thread. Idempotent; also called by the destructor.
    void stop() noexcept;

    //! \brief Plans completed so far.
    int64_t plansCompleted() const noexcept;

private:
    void run();

    InternVLAN1DualSystemState& mState;
    Planner mPlanner;
    std::thread mThread;
    mutable std::mutex mMutex;
    std::condition_variable mWakeCv;
    std::condition_variable mIdleCv;
    int64_t mPending{-1};
    bool mBusy{false};
    bool mStop{false};
    int64_t mCompleted{0};
};

} // namespace internvla_n1
} // namespace trt_edgellm
