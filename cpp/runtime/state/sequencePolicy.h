/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include "runtime/state/sequenceSlots.h"
#include <array>
#include <string_view>

namespace trt_edgellm
{
namespace rt
{
enum class SequenceFinish
{
    kNone,
    kLength,
    kEos,
    kStop
};
struct TokenLogprob
{
    int32_t token{-1};
    double logprob{};
};
struct SequenceSample
{
    int32_t token{-1};
    double logprob{};
    std::array<TokenLogprob, 50> top{};
    int32_t topCount{};
    uint64_t randomCounter{};
    bool thinking{};
};
//! Allocation-free per-token sampler; scratch is shared by the single worker, never by requests' RNG state.
class SequenceSampler
{
public:
    explicit SequenceSampler(int32_t vocabulary);
    SequenceSample sample(float const* logits, SequenceOptions const& options, uint64_t counter);

private:
    struct Entry
    {
        int32_t token{};
        double logit{};
        double weight{};
    };
    std::vector<Entry> mEntries;
};
//! Validate all supported options before a request is enqueued. Returns bounded dynamic metadata bytes.
size_t validateSequenceOptions(SequenceOptions const& options, int32_t vocabulary);

//! Request-owned stopping/text state. Raw BPE pieces are sanitized incrementally before stop matching.
class SequencePolicy
{
public:
    void reset(SequenceOptions options, size_t maxPieceBytes);
    void accept(SequenceSample const& sample, std::string_view piece);
    void finalize();
    SequenceFinish finish() const
    {
        return mFinish;
    }
    std::string_view text() const
    {
        return {mText.data(), mSafeBytes};
    }
    SequenceSample const& last() const
    {
        return mLast;
    }
    std::vector<SequenceSample> const& logprobs() const
    {
        return mLogprobs;
    }

private:
    void appendUtf8(std::string_view piece);
    SequenceOptions mOptions;
    SequenceSample mLast;
    std::vector<SequenceSample> mLogprobs;
    std::string mText;
    std::array<char, 4> mUtf8{};
    int32_t mUtf8Size{};
    int32_t mUtf8Expected{};
    size_t mSafeBytes{};
    size_t mMaxStopBytes{};
    int32_t mGenerated{};
    bool mThinkingDone{};
    SequenceFinish mFinish{SequenceFinish::kNone};
};
} // namespace rt
} // namespace trt_edgellm
