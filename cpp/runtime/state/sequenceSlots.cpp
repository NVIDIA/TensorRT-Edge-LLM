/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "runtime/state/sequenceSlots.h"

#include <atomic>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace trt_edgellm
{
namespace rt
{
namespace
{
uint64_t nextOwner()
{
    static std::atomic<uint64_t> sNext{1};
    uint64_t candidate = sNext.load(std::memory_order_relaxed);
    do
    {
        if (candidate == std::numeric_limits<uint64_t>::max())
        {
            throw std::overflow_error("Sequence owner IDs exhausted");
        }
    } while (!sNext.compare_exchange_weak(candidate, candidate + 1, std::memory_order_relaxed));
    return candidate;
}
} // namespace

SequenceSlots::SequenceSlots(int32_t slots, int32_t maxInputTokens, int32_t maxSequenceTokens)
    : mOwner(nextOwner())
    , mMaxInputTokens(maxInputTokens)
    , mMaxSequenceTokens(maxSequenceTokens)
{
    if (slots <= 0 || maxInputTokens <= 0 || maxSequenceTokens < maxInputTokens)
    {
        throw std::invalid_argument("Invalid sequence slot limits");
    }
    mSlots.resize(slots);
}

SequenceHandle SequenceSlots::acquire(uint64_t requestId, std::vector<int32_t> prompt, SequenceOptions options)
{
    if (requestId == 0 || prompt.empty() || prompt.size() > static_cast<size_t>(mMaxInputTokens)
        || options.maxOutputTokens <= 0
        || static_cast<int64_t>(prompt.size()) + options.maxOutputTokens > mMaxSequenceTokens
        || !std::isfinite(options.temperature) || options.temperature < 0.0F || !std::isfinite(options.topP)
        || options.topP <= 0.0F || options.topP > 1.0F || options.topK < 0 || options.numLogprobs < 0)
    {
        throw std::invalid_argument("Invalid sequence request or capacity");
    }
    for (auto const& state : mSlots)
    {
        if (state.mPhase != SequencePhase::kFree && state.mRequestId == requestId)
        {
            throw std::invalid_argument("Duplicate live request ID");
        }
    }
    for (size_t slot = 0; slot < mSlots.size(); ++slot)
    {
        auto& state = mSlots[slot];
        if (state.mPhase != SequencePhase::kFree)
        {
            continue;
        }
        if (state.mGeneration == std::numeric_limits<uint64_t>::max())
        {
            continue;
        }
        SequenceState next;
        next.mGeneration = state.mGeneration + 1;
        next.mRequestId = requestId;
        next.mPrompt = std::move(prompt);
        next.mOptions = std::move(options);
        next.mOutput.reserve(next.mOptions.maxOutputTokens);
        next.mPhase = SequencePhase::kPrefill;
        state = std::move(next);
        return {mOwner, state.mGeneration, static_cast<int32_t>(slot)};
    }
    throw std::runtime_error("No free sequence slot");
}

SequenceState const& SequenceSlots::get(SequenceHandle handle) const
{
    if (handle.owner != mOwner || handle.slot < 0 || static_cast<size_t>(handle.slot) >= mSlots.size())
    {
        throw std::invalid_argument("Foreign or invalid sequence handle");
    }
    auto const& state = mSlots[handle.slot];
    if (state.mPhase == SequencePhase::kFree || state.mGeneration != handle.generation)
    {
        throw std::invalid_argument("Stale sequence handle");
    }
    return state;
}

SequenceState& SequenceSlots::checked(SequenceHandle handle)
{
    get(handle);
    return mSlots[handle.slot];
}

void SequenceSlots::commitPrompt(SequenceHandle handle, int32_t count)
{
    auto& state = checked(handle);
    if (state.mPhase != SequencePhase::kPrefill || count <= 0
        || count > static_cast<int32_t>(state.mPrompt.size()) - state.mPromptCursor)
    {
        throw std::logic_error("Invalid prompt commit");
    }
    state.mPromptCursor += count;
    state.mCommittedTokens += count;
    if (state.mPromptCursor == static_cast<int32_t>(state.mPrompt.size()))
    {
        state.mPhase = SequencePhase::kAwaitingSample;
    }
}

void SequenceSlots::commitDecode(SequenceHandle handle)
{
    auto& state = checked(handle);
    if (state.mPhase != SequencePhase::kDecode
        || state.mCommittedTokens + 1 != static_cast<int32_t>(state.mPrompt.size() + state.mOutput.size())
        || state.mCommittedTokens >= mMaxSequenceTokens)
    {
        throw std::logic_error("Decode requires exactly one uncommitted output token");
    }
    ++state.mCommittedTokens;
    state.mPhase = SequencePhase::kAwaitingSample;
}

void SequenceSlots::acceptToken(SequenceHandle handle, int32_t token, uint64_t randomDraws)
{
    auto& state = checked(handle);
    if (state.mPhase != SequencePhase::kAwaitingSample || token < 0
        || randomDraws > std::numeric_limits<uint64_t>::max() - state.mRandomCounter)
    {
        throw std::logic_error("Invalid sampled-token acceptance");
    }
    state.mOutput.push_back(token);
    state.mRandomCounter += randomDraws;
    state.mPhase = static_cast<int32_t>(state.mOutput.size()) == state.mOptions.maxOutputTokens
        ? SequencePhase::kFinished
        : SequencePhase::kDecode;
}

void SequenceSlots::finish(SequenceHandle handle)
{
    checked(handle).mPhase = SequencePhase::kFinished;
}

void SequenceSlots::release(SequenceHandle handle)
{
    auto& state = checked(handle);
    auto const generation = state.mGeneration;
    state = SequenceState{};
    state.mGeneration = generation;
}
} // namespace rt
} // namespace trt_edgellm
