/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "common/checkMacros.h"
#include <cstdint>
#include <cuda_runtime_api.h>
#include <functional>
#include <numeric>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace trt_edgellm
{

//! Global profiling control flag
//! When false, no profiling data (metrics or timing) will be recorded
//! This is useful to exclude warmup runs from benchmark statistics
bool getProfilingEnabled();
void setProfilingEnabled(bool enabled);

namespace timer
{

/*!
 * @brief CUDA event pair for timing
 *
 * Lazy-initialized timer pair using CUDA events.
 */
struct TimerPair
{
    cudaEvent_t gpuStart{nullptr}; //!< Start event
    cudaEvent_t gpuEnd{nullptr};   //!< End event
    bool hasStarted{false};        //!< Whether timing has started
    bool isInitialized{false};     //!< Whether events are initialized

    //! @brief Initialize CUDA events
    void initialize()
    {
        if (!isInitialized)
        {
            CUDA_CHECK(cudaEventCreate(&gpuStart));
            CUDA_CHECK(cudaEventCreate(&gpuEnd));
            isInitialized = true;
        }
    }

    //! @brief Destructor - destroys CUDA events
    ~TimerPair()
    {
        if (isInitialized)
        {
            if (gpuStart)
            {
                cudaEventDestroy(gpuStart);
            }
            if (gpuEnd)
            {
                cudaEventDestroy(gpuEnd);
            }
        }
    }

    //! @brief Default constructor
    TimerPair() = default;

    //! @brief Deleted copy constructor
    TimerPair(TimerPair const&) = delete;

    //! @brief Deleted copy assignment
    TimerPair& operator=(TimerPair const&) = delete;

    //! @brief Move constructor
    TimerPair(TimerPair&& other) noexcept
        : gpuStart(other.gpuStart)
        , gpuEnd(other.gpuEnd)
        , hasStarted(other.hasStarted)
        , isInitialized(other.isInitialized)
    {
        other.gpuStart = nullptr;
        other.gpuEnd = nullptr;
        other.hasStarted = false;
        other.isInitialized = false;
    }

    //! @brief Move assignment operator
    TimerPair& operator=(TimerPair&& other) noexcept
    {
        if (this != &other)
        {
            if (isInitialized)
            {
                if (gpuStart)
                {
                    cudaEventDestroy(gpuStart);
                }
                if (gpuEnd)
                {
                    cudaEventDestroy(gpuEnd);
                }
            }

            gpuStart = other.gpuStart;
            gpuEnd = other.gpuEnd;
            hasStarted = other.hasStarted;
            isInitialized = other.isInitialized;

            other.gpuStart = nullptr;
            other.gpuEnd = nullptr;
            other.hasStarted = false;
            other.isInitialized = false;
        }
        return *this;
    }
};

/*!
 * @brief RAII timer session for automatic cleanup
 *
 * Automatically stops timing when session goes out of scope.
 */
class TimerSession
{
public:
    //! @brief Construct active session with callback
    //! @param onEnd Callback to execute on destruction
    TimerSession(std::function<void()> onEnd)
        : mOnEnd(std::move(onEnd))
        , mActive(true)
    {
    }

    //! @brief Construct inactive session
    TimerSession(std::nullptr_t)
        : mOnEnd(nullptr)
        , mActive(false)
    {
    }

    //! @brief Destructor - executes callback if active
    ~TimerSession()
    {
        if (mActive && mOnEnd)
        {
            mOnEnd();
        }
    }

    //! @brief Deleted copy constructor
    TimerSession(TimerSession const&) = delete;

    //! @brief Deleted copy assignment
    TimerSession& operator=(TimerSession const&) = delete;

    //! @brief Move constructor
    TimerSession(TimerSession&& other) noexcept
        : mOnEnd(std::move(other.mOnEnd))
        , mActive(other.mActive)
    {
        other.mActive = false;
    }

    //! @brief Move assignment operator
    TimerSession& operator=(TimerSession&& other) noexcept
    {
        if (this != &other)
        {
            if (mActive && mOnEnd)
            {
                mOnEnd();
            }
            mOnEnd = std::move(other.mOnEnd);
            mActive = other.mActive;
            other.mActive = false;
        }
        return *this;
    }

private:
    std::function<void()> mOnEnd; //!< Cleanup callback
    bool mActive;                 //!< Whether session is active
};

/*!
 * @brief Stage timing data
 *
 * Stores raw timing measurements and calculates derived values on-demand.
 */
struct StageTimingData
{
    std::vector<float> gpuTimesMs; //!< GPU time measurements in milliseconds

    //! @brief Add timing measurement
    //! @param timeMs Time in milliseconds
    void addTiming(float timeMs)
    {
        gpuTimesMs.push_back(timeMs);
    }

    //! @brief Reset all timing data
    void reset()
    {
        gpuTimesMs.clear();
    }

    //! @brief Calculate total GPU time
    //! @return Total time in milliseconds
    float getTotalGpuTimeMs() const
    {
        return std::accumulate(gpuTimesMs.begin(), gpuTimesMs.end(), 0.0f);
    }

    //! @brief Calculate average time per run
    //! @return Average time in milliseconds
    float getAverageTimeMs() const
    {
        return gpuTimesMs.empty() ? 0.0f : getTotalGpuTimeMs() / gpuTimesMs.size();
    }

    //! @brief Get total number of runs
    //! @return Run count
    int64_t getTotalRuns() const
    {
        return static_cast<int64_t>(gpuTimesMs.size());
    }
};

/*!
 * @brief CUDA timer with RAII and deferred calculation
 *
 * Provides stage-based timing using CUDA events with automatic cleanup.
 */
class Timer
{
public:
    //! @brief Default constructor
    Timer() = default;

    //! @brief Destructor
    ~Timer() = default;

    //! @brief Reset all timing data
    void reset();

    /*!
     * @brief Start timing a stage with automatic cleanup
     * @param stageId Stage identifier
     * @param stream CUDA stream (default: 0)
     * @return RAII session that stops timing on destruction
     */
    TimerSession startStage(std::string const& stageId, cudaStream_t stream);

    /*!
     * @brief Get timing data for a stage
     * @param stageId Stage identifier
     * @return Timing data if available, nullopt otherwise
     */
    std::optional<StageTimingData> getTimingData(std::string const& stageId) const;

    /*!
     * @brief Get all timing data
     * @return Map of stage IDs to timing data
     */
    std::unordered_map<std::string, StageTimingData> const& getAllTimingData() const;

private:
    mutable std::unordered_map<std::string, StageTimingData> mTimingData; //!< Timing data per stage

    mutable std::unordered_map<std::string, TimerPair> mTimers;                 //!< Timer pairs per stage
    mutable std::unordered_map<std::string, std::vector<float>> mTimingResults; //!< Pending results
    mutable std::unordered_set<std::string> mPendingTimings;                    //!< Stages with pending timings

    //! @brief Start timer for stage
    void startTimer(std::string const& stageId, cudaStream_t stream);

    //! @brief End timer for stage
    void endTimer(std::string const& stageId, cudaStream_t stream);

    //! @brief Record timing measurement
    void recordTiming(std::string const& stageId) const;

    //! @brief Handle stage completion
    void onStageComplete(std::string const& stageId);
};

/*!
 * @brief Convenience macro for RAII-based stage timing
 *
 * Usage: TIME_STAGE("stage_name", stream);
 */
#define TIME_STAGE(stageId, stream) auto _session = trt_edgellm::gTimer.startStage(stageId, stream)

} // namespace timer

//! Global timer instance
inline timer::Timer gTimer;

} // namespace trt_edgellm