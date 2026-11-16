/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace trt_edgellm
{

//! Thread-local profiling control flag
//! Controls aggregate counters and GPU stage timing; per-request collection is opt-in.
//! This is useful to exclude warmup runs from benchmark statistics
bool getProfilingEnabled() noexcept;
void setProfilingEnabled(bool enabled) noexcept;

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
    //! @throws std::runtime_error if CUDA event creation fails
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
    ~TimerPair() noexcept
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
    TimerSession(std::function<void()> onEnd) noexcept
        : mOnEnd(std::move(onEnd))
        , mActive(true)
    {
    }

    //! @brief Construct inactive session
    TimerSession(std::nullptr_t) noexcept
        : mOnEnd(nullptr)
        , mActive(false)
    {
    }

    //! @brief Destructor - executes callback if active
    ~TimerSession() noexcept
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
    void reset() noexcept
    {
        gpuTimesMs.clear();
    }

    //! @brief Calculate total GPU time
    //! @return Total time in milliseconds
    float getTotalGpuTimeMs() const noexcept
    {
        return std::accumulate(gpuTimesMs.begin(), gpuTimesMs.end(), 0.0f);
    }

    //! @brief Calculate average time per run
    //! @return Average time in milliseconds
    float getAverageTimeMs() const noexcept
    {
        return gpuTimesMs.empty() ? 0.0f : getTotalGpuTimeMs() / gpuTimesMs.size();
    }

    //! @brief Get total number of runs
    //! @return Run count
    int64_t getTotalRuns() const noexcept
    {
        return static_cast<int64_t>(gpuTimesMs.size());
    }
};

//! One completed invocation; model forwards are children of their semantic stage.
struct TimingSample
{
    std::string stageId;
    std::string parentStage;
    uint64_t requestId{0};
    int32_t round{-1};
    int64_t batchSize{0};
    int64_t queryTokens{0};
    int64_t selectedTokens{0};
    std::vector<int32_t> pastLengths;
    bool forward{false};
    int32_t forwardIndex{-1}; //!< Forward ordinal within this request and round; -1 for stages.
    float gpuTimeMs{0};
};

//! Host-visible acceptance facts collected at the production token readback boundary.
struct AcceptanceSample
{
    uint64_t requestId;
    int32_t round;
    int32_t sequenceIndex;
    int32_t proposedDrafts;
    int32_t acceptedDrafts;
    int32_t outputTokens;
    int32_t bonusTokens;
    int32_t baseMaterializedLength;
    bool provenanceExact;
    bool proposedExact; //!< False when proposedDrafts is a verify-tree capacity, not valid proposal nodes.
};

//! Preallocated event pool. Longer-lived profiling can grow without waiting for earlier stages.
class Timer
{
public:
    void reserve(size_t maxSamples, size_t maxSequences = 1);
    //! Number of timing slots whose CUDA events have been created.
    size_t capacity() const noexcept
    {
        return mPending.size();
    }
    //! When disabled, profiling keeps host-side acceptance records but creates no CUDA events or stage samples.
    void setGpuTimingEnabled(bool enabled) noexcept
    {
        mGpuTimingEnabled = enabled;
    }
    void recordAcceptance(int32_t sequenceIndex, int32_t proposed, int32_t accepted, int32_t output, int32_t bonus,
        int32_t baseLength, bool provenanceExact = false, bool proposedExact = false);
    std::vector<AcceptanceSample> const& getAcceptanceSamples() const noexcept
    {
        return mAcceptanceSamples;
    }
    void reset() noexcept;
    void beginRequest();
    void setRound(int32_t round) noexcept;
    //! Host-known past lengths for the next forward; empty means unavailable without device readback.
    void setPastLengths(std::vector<int32_t> const& lengths);
    TimerSession startStage(std::string const& stageId, cudaStream_t stream);
    TimerSession startForward(
        std::string const& modelRole, int64_t batch, int64_t tokens, int64_t selected, cudaStream_t stream);
    std::optional<StageTimingData> getTimingData(std::string const& stageId) const;
    std::unordered_map<std::string, StageTimingData> const& getAllTimingData() const;
    std::vector<TimingSample> const& getSamples() const;

private:
    void grow(size_t maxSamples, size_t maxSequences);
    struct PendingTiming
    {
        TimerPair events;
        TimingSample sample;
        bool collected{false};
        cudaStream_t stream{};
    };
    TimerSession start(
        std::string const& name, bool forward, int64_t batch, int64_t tokens, int64_t selected, cudaStream_t stream);
    void collect() const;
    mutable std::vector<PendingTiming> mPending;
    size_t mUsed{0};
    size_t mMaxSequences{0};
    bool mGpuTimingEnabled{true};
    std::vector<size_t> mActive;
    uint64_t mRequestId{0};
    int32_t mRound{-1};
    int32_t mNextForwardIndex{0};
    std::vector<int32_t> mPastLengths;
    mutable std::unordered_map<std::string, StageTimingData> mTimingData;
    mutable std::vector<TimingSample> mSamples;
    std::vector<AcceptanceSample> mAcceptanceSamples;
};

//! Keeps legacy profiling call sites attached to the current runtime's event storage.
class TimerProxy
{
public:
    void select(std::shared_ptr<Timer> timer)
    {
        mTimer = std::move(timer);
    }
    void reserve(size_t size, size_t maxSequences = 1)
    {
        mTimer->reserve(size, maxSequences);
    }
    size_t capacity() const noexcept
    {
        return mTimer->capacity();
    }
    void recordAcceptance(int32_t sequenceIndex, int32_t proposed, int32_t accepted, int32_t output, int32_t bonus,
        int32_t baseLength, bool provenanceExact = false, bool proposedExact = false)
    {
        mTimer->recordAcceptance(
            sequenceIndex, proposed, accepted, output, bonus, baseLength, provenanceExact, proposedExact);
    }
    std::vector<AcceptanceSample> const& getAcceptanceSamples() const noexcept
    {
        return mTimer->getAcceptanceSamples();
    }
    void reset() noexcept
    {
        mTimer->reset();
    }
    void beginRequest()
    {
        mTimer->beginRequest();
    }
    void setRound(int32_t round) noexcept
    {
        mTimer->setRound(round);
    }
    void setPastLengths(std::vector<int32_t> const& lengths)
    {
        mTimer->setPastLengths(lengths);
    }
    TimerSession startStage(std::string const& stage, cudaStream_t stream)
    {
        return mTimer->startStage(stage, stream);
    }
    TimerSession startForward(
        std::string const& modelRole, int64_t batch, int64_t tokens, int64_t selected, cudaStream_t stream)
    {
        return mTimer->startForward(modelRole, batch, tokens, selected, stream);
    }
    std::optional<StageTimingData> getTimingData(std::string const& stage) const
    {
        return mTimer->getTimingData(stage);
    }
    std::unordered_map<std::string, StageTimingData> const& getAllTimingData() const
    {
        return mTimer->getAllTimingData();
    }
    std::vector<TimingSample> const& getSamples() const
    {
        return mTimer->getSamples();
    }

private:
    std::shared_ptr<Timer> mTimer{std::make_shared<Timer>()};
};

#define TIME_STAGE(stageId, stream) auto _session = trt_edgellm::gTimer.startStage(stageId, stream)

} // namespace timer

inline thread_local timer::TimerProxy gTimer;

} // namespace trt_edgellm
