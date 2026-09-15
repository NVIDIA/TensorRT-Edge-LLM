/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "runtime/state/sequencePolicy.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace trt_edgellm
{
namespace rt
{
size_t validateSequenceOptions(SequenceOptions const& o, int32_t vocabulary)
{
    if (o.maxOutputTokens <= 0 || o.maxOutputTokens > 8192 || !std::isfinite(o.temperature) || o.temperature < 0
        || o.temperature > 2 || !std::isfinite(o.topP) || o.topP <= 0 || o.topP > 1 || o.topK < 0 || o.topK > vocabulary
        || o.numLogprobs < 0 || o.numLogprobs > 50 || o.logitBias.size() > 1024 || o.eosTokenIds.size() > 256
        || o.stopStrings.size() > 64)
    {
        throw std::invalid_argument("Invalid per-sequence options");
    }
    auto valid = [&](int32_t id) { return id >= 0 && id < vocabulary; };
    for (auto id : o.eosTokenIds)
    {
        if (!valid(id))
        {
            throw std::invalid_argument("Invalid EOS ID");
        }
    }
    for (auto id : {o.primaryEosTokenId, o.thinkingStartTokenId, o.thinkingEndTokenId})
    {
        if (id != -1 && !valid(id))
        {
            throw std::invalid_argument("Invalid policy token ID");
        }
    }
    size_t bytes = o.eosTokenIds.size() * sizeof(int32_t) + o.logitBias.size() * 64;
    for (auto const& pair : o.logitBias)
    {
        if (!valid(pair.first) || !std::isfinite(pair.second) || pair.second < -100 || pair.second > 100)
        {
            throw std::invalid_argument("Invalid logit bias");
        }
    }
    size_t stops = 0;
    for (auto const& stop : o.stopStrings)
    {
        if (stop.empty() || stop.size() > 4096)
        {
            throw std::invalid_argument("Invalid stop string");
        }
        stops += stop.size() + sizeof(std::string);
    }
    if (stops > 16384)
    {
        throw std::invalid_argument("Stop strings exceed byte limit");
    }
    return bytes + stops;
}
SequenceSampler::SequenceSampler(int32_t vocabulary)
    : mEntries(vocabulary)
{
    if (vocabulary <= 0)
    {
        throw std::invalid_argument("Empty vocabulary");
    }
}
SequenceSample SequenceSampler::sample(float const* logits, SequenceOptions const& o, uint64_t counter)
{
    for (size_t i = 0; i < mEntries.size(); ++i)
    {
        if (!std::isfinite(logits[i]))
        {
            throw std::runtime_error("Nonfinite logits");
        }
        auto const bias = o.logitBias.find(static_cast<int32_t>(i));
        mEntries[i] = {static_cast<int32_t>(i), logits[i] + (bias == o.logitBias.end() ? 0.0 : bias->second), 0};
    }
    auto better
        = [](Entry const& a, Entry const& b) { return a.logit > b.logit || (a.logit == b.logit && a.token < b.token); };
    bool const greedy = o.temperature <= 1e-3F || o.topK == 1
        || (o.topK <= 1 && o.topP >= 1.0F - 1e-6F && std::fabs(o.temperature - 1.0F) <= 1e-3F);
    // Logprob requests retain the original summation order for exact policy compatibility.
    size_t const needed
        = o.numLogprobs > 0 || (!greedy && o.topK == 0) ? mEntries.size() : (greedy ? 1 : static_cast<size_t>(o.topK));
    std::partial_sort(mEntries.begin(), mEntries.begin() + needed, mEntries.end(), better);
    double const maximum = mEntries.front().logit;
    double normalizer = 0;
    for (auto const& entry : mEntries)
    {
        normalizer += std::exp(entry.logit - maximum);
    }
    double const logNormalizer = maximum + std::log(normalizer);
    SequenceSample out;
    out.randomCounter = counter;
    out.topCount = std::min(o.numLogprobs, static_cast<int32_t>(mEntries.size()));
    for (int32_t i = 0; i < out.topCount; ++i)
    {
        out.top[i] = {mEntries[i].token, mEntries[i].logit - logNormalizer};
    }
    size_t selected = 0;
    if (!greedy)
    {
        if (counter == std::numeric_limits<uint64_t>::max())
        {
            throw std::overflow_error("RNG counter exhausted");
        }
        size_t const k = o.topK > 0 ? static_cast<size_t>(o.topK) : mEntries.size();
        double sum = 0;
        for (size_t i = 0; i < k; ++i)
        {
            mEntries[i].weight = std::exp((mEntries[i].logit - maximum) / o.temperature);
            sum += mEntries[i].weight;
        }
        size_t n = 0;
        double retained = 0;
        do
        {
            retained += mEntries[n++].weight;
        } while (n < k && retained < o.topP * sum);
        uint64_t bits = o.seed + 0x9e3779b97f4a7c15ULL * (counter + 1);
        bits = (bits ^ (bits >> 30)) * 0xbf58476d1ce4e5b9ULL;
        bits = (bits ^ (bits >> 27)) * 0x94d049bb133111ebULL;
        bits ^= bits >> 31;
        double const draw = static_cast<double>(bits >> 11) * 0x1.0p-53 * retained;
        double cumulative = mEntries[0].weight;
        while (selected + 1 < n && draw >= cumulative)
        {
            cumulative += mEntries[++selected].weight;
        }
        out.randomCounter = counter + 1;
    }
    out.token = mEntries[selected].token;
    out.logprob = mEntries[selected].logit - logNormalizer;
    return out;
}
void SequencePolicy::reset(SequenceOptions options, size_t maxPieceBytes)
{
    mOptions = std::move(options);
    mLast = {};
    mLogprobs.clear();
    if (mOptions.numLogprobs)
    {
        mLogprobs.reserve(mOptions.maxOutputTokens);
    }
    mText.clear();
    mText.reserve(static_cast<size_t>(mOptions.maxOutputTokens) * std::max(size_t{1}, maxPieceBytes) * 3 + 3);
    mUtf8Size = 0;
    mUtf8Expected = 0;
    mSafeBytes = 0;
    mMaxStopBytes = 0;
    mGenerated = 0;
    mThinkingDone = !mOptions.enableThinking;
    mFinish = SequenceFinish::kNone;
    for (auto const& stop : mOptions.stopStrings)
    {
        mMaxStopBytes = std::max(mMaxStopBytes, stop.size());
    }
}
void SequencePolicy::appendUtf8(std::string_view piece)
{
    for (unsigned char byte : piece)
    {
        if (mUtf8Size)
        {
            unsigned char const lead = static_cast<unsigned char>(mUtf8[0]);
            bool const continuation = byte >= 0x80 && byte <= 0xbf
                && !(mUtf8Size == 1
                    && ((lead == 0xe0 && byte < 0xa0) || (lead == 0xed && byte >= 0xa0) || (lead == 0xf0 && byte < 0x90)
                        || (lead == 0xf4 && byte >= 0x90)));
            if (continuation)
            {
                mUtf8[mUtf8Size++] = static_cast<char>(byte);
                if (mUtf8Size == mUtf8Expected)
                {
                    mText.append(mUtf8.data(), mUtf8Size);
                    mUtf8Size = 0;
                }
                continue;
            }
            mText.append("\xef\xbf\xbd");
            mUtf8Size = 0;
        }
        if (byte < 0x80)
        {
            mText.push_back(static_cast<char>(byte));
        }
        else if (byte >= 0xc2 && byte <= 0xf4)
        {
            mUtf8[0] = static_cast<char>(byte);
            mUtf8Size = 1;
            mUtf8Expected = byte < 0xe0 ? 2 : (byte < 0xf0 ? 3 : 4);
        }
        else
        {
            mText.append("\xef\xbf\xbd");
        }
    }
}
void SequencePolicy::accept(SequenceSample const& sample, std::string_view piece)
{
    if (mFinish != SequenceFinish::kNone)
    {
        throw std::logic_error("Sampling finished policy");
    }
    ++mGenerated;
    if (!mThinkingDone
        && (sample.token == mOptions.thinkingEndTokenId
            || (mGenerated == 1 && sample.token != mOptions.thinkingStartTokenId)))
    {
        mThinkingDone = true;
    }
    mLast = sample;
    mLast.thinking = !mThinkingDone;
    if (mOptions.numLogprobs)
    {
        mLogprobs.push_back(mLast);
    }
    bool const eos = !mOptions.ignoreEos
        && (sample.token == mOptions.primaryEosTokenId
            || (std::find(mOptions.eosTokenIds.begin(), mOptions.eosTokenIds.end(), sample.token)
                    != mOptions.eosTokenIds.end()
                && (!mOptions.enableThinking || mThinkingDone)));
    if (!eos)
    {
        appendUtf8(piece);
    }
    if (eos)
    {
        mFinish = SequenceFinish::kEos;
    }
    else if (mGenerated >= mOptions.maxOutputTokens)
    {
        mFinish = SequenceFinish::kLength;
    }
    if (mFinish != SequenceFinish::kNone && mUtf8Size)
    {
        mText.append("\xef\xbf\xbd");
        mUtf8Size = 0;
    }
    size_t match = std::string::npos;
    for (auto const& stop : mOptions.stopStrings)
    {
        match = std::min(match, mText.find(stop, mSafeBytes));
    }
    if (match != std::string::npos)
    {
        mText.resize(match);
        mFinish = SequenceFinish::kStop;
        mUtf8Size = 0;
    }
    if (mFinish != SequenceFinish::kNone)
    {
        finalize();
        return;
    }
    mSafeBytes = mText.size() > mMaxStopBytes ? mText.size() - mMaxStopBytes : 0;
    if (!mMaxStopBytes)
    {
        mSafeBytes = mText.size();
    }
    // Never split a UTF-8 code point when holding the stop look-behind window.
    while (mSafeBytes < mText.size() && (static_cast<unsigned char>(mText[mSafeBytes]) & 0xc0) == 0x80)
    {
        --mSafeBytes;
    }
}
void SequencePolicy::finalize()
{
    if (mUtf8Size)
    {
        mText.append("\xef\xbf\xbd");
        mUtf8Size = 0;
    }
    mSafeBytes = mText.size();
}
} // namespace rt
} // namespace trt_edgellm
