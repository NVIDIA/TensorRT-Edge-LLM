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

#include "timer.h"
#include "common/checkMacros.h"
#include "common/logger.h"
#include <algorithm>
#include <cuda_runtime.h>
#include <numeric>

namespace trt_edgellm
{

// Thread-local profiling control flag implementation.
// Multi-device threaded inference enables profiling only on rank 0. Keeping this
// flag thread-local prevents peer ranks from entering the shared timer.
static thread_local bool gProfilingEnabled = false;

bool getProfilingEnabled() noexcept
{
    return gProfilingEnabled;
}

void setProfilingEnabled(bool enabled) noexcept
{
    gProfilingEnabled = enabled;
}

namespace timer
{

void Timer::reserve(size_t maxSamples, size_t maxSequences)
{
    ELLM_CHECK(mActive.empty(), "Cannot resize timing storage inside a timed stage");
    if (mGpuTimingEnabled)
    {
        grow(maxSamples, maxSequences);
    }
}

void Timer::grow(size_t maxSamples, size_t maxSequences)
{
    if (maxSamples <= mPending.size() && maxSequences <= mMaxSequences)
    {
        return;
    }
    mMaxSequences = std::max(mMaxSequences, maxSequences);
    mPending.resize(std::max(maxSamples, mPending.size()));
    for (auto& timing : mPending)
    {
        timing.events.initialize();
        timing.sample.stageId.reserve(96);
        timing.sample.parentStage.reserve(96);
        timing.sample.pastLengths.reserve(mMaxSequences);
    }
    mPastLengths.reserve(mMaxSequences);
    mSamples.reserve(maxSamples);
    mAcceptanceSamples.reserve(mAcceptanceSamples.size() + maxSamples);
    mActive.reserve(maxSamples);
}

void Timer::reset() noexcept
{
    for (size_t i = 0; i < mUsed; ++i)
    {
        if (!mPending[i].events.hasStarted)
        {
            static_cast<void>(cudaEventSynchronize(mPending[i].events.gpuEnd));
        }
    }
    mUsed = 0;
    mActive.clear();
    mTimingData.clear();
    mSamples.clear();
    mAcceptanceSamples.clear();
    mPastLengths.clear();
    mRound = -1;
    mNextForwardIndex = 0;
}

void Timer::recordAcceptance(int32_t sequenceIndex, int32_t proposed, int32_t accepted, int32_t output, int32_t bonus,
    int32_t baseLength, bool provenanceExact, bool proposedExact)
{
    if (getProfilingEnabled())
    {
        mAcceptanceSamples.push_back({mRequestId, mRound, sequenceIndex, proposed, accepted, output, bonus, baseLength,
            provenanceExact, proposedExact});
    }
}

void Timer::beginRequest()
{
    ELLM_CHECK(mActive.empty(), "A profiling request cannot start inside an active stage");
    collect();
    mUsed = 0;
    mAcceptanceSamples.reserve(mAcceptanceSamples.size() + mPending.size());
    ++mRequestId;
    mRound = -1;
    mNextForwardIndex = 0;
    mPastLengths.clear();
}

void Timer::setRound(int32_t round) noexcept
{
    if (mRound != round)
    {
        mNextForwardIndex = 0;
    }
    mRound = round;
    mPastLengths.clear();
}

void Timer::setPastLengths(std::vector<int32_t> const& lengths)
{
    if (getProfilingEnabled())
    {
        mPastLengths = lengths;
    }
}

TimerSession Timer::startStage(std::string const& stageId, cudaStream_t stream)
{
    return start(stageId, false, 0, 0, 0, stream);
}

TimerSession Timer::startForward(
    std::string const& modelRole, int64_t batch, int64_t tokens, int64_t selected, cudaStream_t stream)
{
    return start(modelRole, true, batch, tokens, selected, stream);
}

TimerSession Timer::start(
    std::string const& name, bool forward, int64_t batch, int64_t tokens, int64_t selected, cudaStream_t stream)
{
    if (!getProfilingEnabled() || !mGpuTimingEnabled)
    {
        return TimerSession(nullptr);
    }
    cudaStreamCaptureStatus captureStatus;
    CUDA_CHECK(cudaStreamIsCapturing(stream, &captureStatus));
    if (captureStatus != cudaStreamCaptureStatusNone)
    {
        return TimerSession(nullptr);
    }
    if (mPending.empty())
    {
        constexpr size_t kDefaultCapacity = 16384;
        reserve(kDefaultCapacity);
    }
    if (mUsed == mPending.size())
    {
        // Sessions retain slot indices, so growing the pool preserves unfinished parent stages.
        grow(mPending.size() * 2, mMaxSequences);
    }
    size_t const index = mUsed++;
    auto& timing = mPending[index];
    auto& sample = timing.sample;
    sample.stageId.assign(forward ? "forward/" : "");
    sample.stageId.append(name);
    sample.parentStage = mActive.empty() ? "" : mPending[mActive.back()].sample.stageId;
    sample.requestId = mRequestId;
    sample.round = mRound;
    sample.batchSize = batch;
    sample.queryTokens = tokens;
    sample.selectedTokens = selected;
    sample.forward = forward;
    sample.forwardIndex = forward ? mNextForwardIndex++ : -1;
    sample.gpuTimeMs = 0;
    sample.pastLengths.clear();
    if (forward)
    {
        sample.pastLengths.assign(mPastLengths.begin(), mPastLengths.end());
        mPastLengths.clear();
    }
    timing.collected = false;
    timing.stream = stream;
    CUDA_CHECK(cudaEventRecord(timing.events.gpuStart, stream));
    timing.events.hasStarted = true;
    mActive.push_back(index);
    return TimerSession([this, index]() {
        auto& pending = mPending[index];
        CUDA_CHECK(cudaEventRecord(pending.events.gpuEnd, pending.stream));
        pending.events.hasStarted = false;
        if (!mActive.empty() && mActive.back() == index)
        {
            mActive.pop_back();
        }
    });
}

void Timer::collect() const
{
    for (size_t i = 0; i < mUsed; ++i)
    {
        auto& pending = mPending[i];
        if (pending.collected || pending.events.hasStarted)
        {
            continue;
        }
        CUDA_CHECK(cudaEventSynchronize(pending.events.gpuEnd));
        CUDA_CHECK(cudaEventElapsedTime(&pending.sample.gpuTimeMs, pending.events.gpuStart, pending.events.gpuEnd));
        mTimingData[pending.sample.stageId].addTiming(pending.sample.gpuTimeMs);
        mSamples.push_back(pending.sample);
        pending.collected = true;
    }
}

std::optional<StageTimingData> Timer::getTimingData(std::string const& stageId) const
{
    collect();
    auto const it = mTimingData.find(stageId);
    return it == mTimingData.end() ? std::nullopt : std::make_optional(it->second);
}

std::unordered_map<std::string, StageTimingData> const& Timer::getAllTimingData() const
{
    collect();
    return mTimingData;
}

std::vector<TimingSample> const& Timer::getSamples() const
{
    collect();
    return mSamples;
}

} // namespace timer
} // namespace trt_edgellm
