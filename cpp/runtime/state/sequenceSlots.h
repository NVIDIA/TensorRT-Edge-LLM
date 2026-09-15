/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace trt_edgellm
{
namespace rt
{
//! Handles are valid only for one allocation in one slot pool.
struct SequenceHandle
{
    uint64_t owner{};
    uint64_t generation{};
    int32_t slot{-1};
};

//! Request-local controls; sampling and stopping policy are applied by the scheduler, not the step executor.
struct SequenceOptions
{
    int32_t maxOutputTokens{128};
    float temperature{1.0F};
    float topP{1.0F};
    int64_t topK{};
    uint64_t seed{42};
    int32_t numLogprobs{};
    bool enableThinking{};
    bool ignoreEos{};
    int32_t primaryEosTokenId{-1};
    int32_t thinkingStartTokenId{-1};
    int32_t thinkingEndTokenId{-1};
    std::vector<int32_t> eosTokenIds;
    std::vector<std::string> stopStrings;
    std::unordered_map<int32_t, float> logitBias;
};

enum class SequencePhase
{
    kFree,
    kPrefill,
    kAwaitingSample,
    kDecode,
    kFinished,
};

//! Persistent logical state; the physical slot owns its KV, recurrent and convolution rows.
class SequenceState
{
public:
    uint64_t requestId() const noexcept
    {
        return mRequestId;
    }
    SequencePhase phase() const noexcept
    {
        return mPhase;
    }
    int32_t promptCursor() const noexcept
    {
        return mPromptCursor;
    }
    int32_t committedTokens() const noexcept
    {
        return mCommittedTokens;
    }
    uint64_t randomCounter() const noexcept
    {
        return mRandomCounter;
    }
    std::vector<int32_t> const& prompt() const noexcept
    {
        return mPrompt;
    }
    std::vector<int32_t> const& output() const noexcept
    {
        return mOutput;
    }
    SequenceOptions const& options() const noexcept
    {
        return mOptions;
    }

private:
    friend class SequenceSlots;
    uint64_t mRequestId{};
    uint64_t mGeneration{};
    SequencePhase mPhase{SequencePhase::kFree};
    int32_t mPromptCursor{};
    int32_t mCommittedTokens{};
    uint64_t mRandomCounter{};
    std::vector<int32_t> mPrompt;
    std::vector<int32_t> mOutput;
    SequenceOptions mOptions;
};

//! Single-owner bookkeeping with no CUDA dependency; allocation occurs only at admission.
class SequenceSlots
{
public:
    SequenceSlots(int32_t slots, int32_t maxInputTokens, int32_t maxSequenceTokens);
    SequenceSlots(SequenceSlots const&) = delete;
    SequenceSlots& operator=(SequenceSlots const&) = delete;
    SequenceHandle acquire(uint64_t requestId, std::vector<int32_t> prompt, SequenceOptions options);
    SequenceState const& get(SequenceHandle handle) const;
    void commitPrompt(SequenceHandle handle, int32_t count);
    void commitDecode(SequenceHandle handle);
    void acceptToken(SequenceHandle handle, int32_t token, uint64_t randomDraws = 0);
    void finish(SequenceHandle handle);
    void release(SequenceHandle handle);

private:
    SequenceState& checked(SequenceHandle handle);
    uint64_t mOwner;
    int32_t mMaxInputTokens;
    int32_t mMaxSequenceTokens;
    std::vector<SequenceState> mSlots;
};
} // namespace rt
} // namespace trt_edgellm
