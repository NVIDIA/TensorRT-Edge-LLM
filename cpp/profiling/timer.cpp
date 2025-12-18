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

#include "timer.h"
#include "common/checkMacros.h"
#include "common/logger.h"
#include <cuda_runtime.h>
#include <numeric>

namespace trt_edgellm
{

// Global profiling control flag implementation
static bool gProfilingEnabled = false;

bool getProfilingEnabled()
{
    return gProfilingEnabled;
}

void setProfilingEnabled(bool enabled)
{
    gProfilingEnabled = enabled;
}

namespace timer
{

void Timer::reset()
{
    mTimingData.clear();
    mTimers.clear();
    mTimingResults.clear();
    mPendingTimings.clear();
}

TimerSession Timer::startStage(std::string const& stageId, cudaStream_t stream)
{
    if (!getProfilingEnabled())
    {
        return TimerSession(nullptr);
    }

    startTimer(stageId, stream);
    return TimerSession([this, stageId]() { onStageComplete(stageId); });
}

std::optional<StageTimingData> Timer::getTimingData(std::string const& stageId) const
{
    // Calculate any pending timings for this stage
    if (mPendingTimings.find(stageId) != mPendingTimings.end())
    {
        recordTiming(stageId);
        mPendingTimings.erase(stageId);
    }

    // Aggregate any stored timing results
    auto resultIt = mTimingResults.find(stageId);
    if (resultIt != mTimingResults.end() && !resultIt->second.empty())
    {
        for (float timeMs : resultIt->second)
        {
            mTimingData[stageId].addTiming(timeMs);
        }
        resultIt->second.clear();
    }

    auto it = mTimingData.find(stageId);
    return it != mTimingData.end() ? std::make_optional(it->second) : std::nullopt;
}

std::unordered_map<std::string, StageTimingData> const& Timer::getAllTimingData() const
{
    if (mPendingTimings.size() > 0)
    {
        for (auto const& stageId : mPendingTimings)
        {
            recordTiming(stageId);
        }
        mPendingTimings.clear();
    }
    // Aggregate all stored timing results
    for (auto& [stageId, resultList] : mTimingResults)
    {
        if (!resultList.empty())
        {
            for (float timeMs : resultList)
            {
                mTimingData[stageId].addTiming(timeMs);
            }
            resultList.clear();
        }
    }

    return mTimingData;
}

void Timer::startTimer(std::string const& stageId, cudaStream_t stream)
{
    if (!getProfilingEnabled())
    {
        return;
    }

    // Simple timer management - one timer per stage
    if (mTimers.find(stageId) == mTimers.end())
    {
        mTimers[stageId] = TimerPair();
    }

    auto& timer = mTimers[stageId];

    // If there's a pending timing for this stage, record it first
    if (mPendingTimings.find(stageId) != mPendingTimings.end())
    {
        recordTiming(stageId);
        mPendingTimings.erase(stageId);
    }

    timer.initialize();
    CUDA_CHECK(cudaEventRecord(timer.gpuStart, stream));
    timer.hasStarted = true;
}

void Timer::endTimer(std::string const& stageId, cudaStream_t stream)
{
    if (!getProfilingEnabled())
    {
        return;
    }

    auto it = mTimers.find(stageId);
    if (it == mTimers.end() || !it->second.hasStarted)
    {
        LOG_WARNING("Timer '%s' was not started", stageId.c_str());
        return;
    }

    auto& timer = it->second;
    CUDA_CHECK(cudaEventRecord(timer.gpuEnd, stream));
    timer.hasStarted = false;

    // Mark for deferred calculation
    if (mTimingResults.find(stageId) == mTimingResults.end())
    {
        mTimingResults[stageId] = {};
    }
    mPendingTimings.insert(stageId);
}

void Timer::recordTiming(std::string const& stageId) const
{
    auto it = mTimers.find(stageId);
    if (it == mTimers.end() || !it->second.isInitialized)
    {
        LOG_WARNING("Timer '%s' cannot be found or not initialized", stageId.c_str());
        return;
    }

    auto& timer = it->second;

    // Skip if timer is currently running
    if (timer.hasStarted)
    {
        return;
    }

    // Deferred synchronization - only synchronize the end event
    CUDA_CHECK(cudaEventSynchronize(timer.gpuEnd));

    float gpuElapsedTime = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&gpuElapsedTime, timer.gpuStart, timer.gpuEnd));

    // Store result for aggregation
    mTimingResults[stageId].push_back(gpuElapsedTime);
}

void Timer::onStageComplete(std::string const& stageId)
{
    endTimer(stageId, 0);
}

} // namespace timer
} // namespace trt_edgellm