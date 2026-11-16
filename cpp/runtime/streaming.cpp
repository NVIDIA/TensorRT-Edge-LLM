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

#include "runtime/streaming.h"

#include "common/logger.h"
#include "common/utf8.h"
#include "runtime/llmRuntimeUtils.h"
#include "runtime/state/decodingInferenceContext.h"
#include "tokenizer/tokenizer.h"

#include <algorithm>
#include <stdexcept>

namespace trt_edgellm
{
namespace rt
{

//=============================================================================
// StreamChannel — consumer API + friend-reachable producer API
//=============================================================================

std::shared_ptr<StreamChannel> StreamChannel::create(bool collectMetrics)
{
    // std::make_shared cannot access the private constructor; use the new/shared_ptr form.
    return std::shared_ptr<StreamChannel>(new StreamChannel(collectMetrics));
}

void StreamChannel::push(StreamChunk chunk)
{
    {
        std::lock_guard<std::mutex> lk(mMutex);
        mPending.push_back(std::move(chunk));
    }
    mCv.notify_one();
}

void StreamChannel::finish(FinishReason r)
{
    {
        std::lock_guard<std::mutex> lk(mMutex);
        if (mFinished.load(std::memory_order_relaxed))
        {
            return; // Idempotent — first caller wins.
        }
        mReason.store(r, std::memory_order_release);
        mFinished.store(true, std::memory_order_release);
    }
    mCv.notify_all();
}

void StreamChannel::setOriginalBatchIdx(int32_t idx) noexcept
{
    mOriginalBatchIdx.store(idx, std::memory_order_release);
}

bool StreamChannel::isFinished() const noexcept
{
    return mFinished.load(std::memory_order_acquire);
}

FinishReason StreamChannel::getReason() const noexcept
{
    if (!mFinished.load(std::memory_order_acquire))
    {
        return FinishReason::kNotFinished;
    }
    return mReason.load(std::memory_order_relaxed);
}

int32_t StreamChannel::getOriginalBatchIdx() const noexcept
{
    return mOriginalBatchIdx.load(std::memory_order_acquire);
}

bool StreamChannel::isCancelled() const noexcept
{
    return mCancelled.load(std::memory_order_acquire);
}

void StreamChannel::cancel() noexcept
{
    mCancelled.store(true, std::memory_order_release);
    {
        std::lock_guard<std::mutex> lk(mMutex);
    }
    mCv.notify_all();
}

void StreamChannel::setStreamInterval(int32_t n) noexcept
{
    mStreamInterval.store(std::max(1, n), std::memory_order_release);
}

int32_t StreamChannel::getStreamInterval() const noexcept
{
    return mStreamInterval.load(std::memory_order_acquire);
}

void StreamChannel::setSkipSpecialTokens(bool skip) noexcept
{
    mSkipSpecialTokens.store(skip, std::memory_order_release);
}

bool StreamChannel::getSkipSpecialTokens() const noexcept
{
    return mSkipSpecialTokens.load(std::memory_order_acquire);
}

std::optional<StreamChunk> StreamChannel::tryPop()
{
    std::lock_guard<std::mutex> lk(mMutex);
    if (mPending.empty())
    {
        return std::nullopt;
    }
    StreamChunk out = std::move(mPending.front());
    mPending.pop_front();
    return out;
}

std::optional<StreamChunk> StreamChannel::waitPop(std::chrono::milliseconds timeout)
{
    std::unique_lock<std::mutex> lk(mMutex);
    if (!mCv.wait_for(lk, timeout, [&] {
            return !mPending.empty() || mFinished.load(std::memory_order_acquire)
                || mCancelled.load(std::memory_order_acquire);
        }))
    {
        return std::nullopt;
    }
    if (mPending.empty())
    {
        return std::nullopt;
    }
    StreamChunk out = std::move(mPending.front());
    mPending.pop_front();
    return out;
}

//=============================================================================
// Streaming helpers — free functions (friends of StreamChannel)
//=============================================================================

void attachStreamChannel(std::shared_ptr<StreamChannel> const& channel, int32_t originalIdx)
{
    bool const wasAttached = channel->mAttachedToRequest.exchange(true, std::memory_order_acq_rel);
    if (wasAttached)
    {
        // validateStreamingSubmission should have caught this; belt-and-suspenders.
        throw std::runtime_error("StreamChannel already attached to a live request");
    }
    channel->setOriginalBatchIdx(originalIdx);
}

void detachStreamChannel(std::shared_ptr<StreamChannel> const& channel) noexcept
{
    channel->setOriginalBatchIdx(-1);
    channel->mAttachedToRequest.store(false, std::memory_order_release);
}

bool validateStreamingSubmission(LLMGenerationRequest const& request)
{
    if (request.streamChannels.empty())
    {
        return true;
    }
    if (request.streamChannels.size() != request.requests.size())
    {
        LOG_ERROR("streamChannels size (%zu) must equal requests size (%zu)", request.streamChannels.size(),
            request.requests.size());
        return false;
    }
    for (auto const& ch : request.streamChannels)
    {
        if (!ch)
        {
            continue; // Null entry opts out of streaming for this slot.
        }
        if (ch->isFinished())
        {
            LOG_ERROR("StreamChannel already finished — create a new channel per request");
            return false;
        }
        if (ch->mAttachedToRequest.load(std::memory_order_acquire))
        {
            LOG_ERROR("StreamChannel already attached to a live request — concurrent reuse forbidden");
            return false;
        }
    }
    return true;
}

char const* finishReasonName(FinishReason r) noexcept
{
    switch (r)
    {
    case FinishReason::kNotFinished: return "not-finished";
    case FinishReason::kEndId: return "end-of-sequence";
    case FinishReason::kLength: return "max-length";
    case FinishReason::kCancelled: return "cancelled";
    case FinishReason::kError: return "error";
    case FinishReason::kStopWords: return "stop-words";
    }
    return "?";
}

StopMatchOutcome applyStopStringMatch(
    std::string& buffer, std::vector<std::string> const& stops, size_t maxStopLen, bool isFinal)
{
    StopMatchOutcome out;

    // maxStopLen == 0 ⇒ no stops or all-empty entries — pass-through.
    if (maxStopLen == 0)
    {
        out.emitted = std::move(buffer);
        buffer.clear();
        return out;
    }

    // Earliest-position-wins across all non-empty stops.
    size_t earliest = std::string::npos;
    for (auto const& stop : stops)
    {
        if (stop.empty())
        {
            continue;
        }
        size_t const pos = buffer.find(stop);
        if (pos < earliest)
        {
            earliest = pos;
        }
    }

    if (earliest != std::string::npos)
    {
        // Matched — emit everything before the match, drop the rest.
        out.emitted = buffer.substr(0, earliest);
        buffer.clear();
        out.stopMatched = true;
        return out;
    }

    if (isFinal)
    {
        // Final flush: nothing held back.
        out.emitted = std::move(buffer);
        buffer.clear();
        return out;
    }

    // No match yet: hold back the last (maxStopLen - 1) bytes so a stop string
    // straddling the emit/buffer boundary is detected on the next iteration.
    size_t const holdBack = maxStopLen - 1;
    if (buffer.size() <= holdBack)
    {
        // Whole buffer must be retained — nothing safe to emit.
        return out; // out.emitted stays empty
    }
    size_t const safeLen = buffer.size() - holdBack;
    out.emitted = buffer.substr(0, safeLen);
    buffer.erase(0, safeLen);
    return out;
}

void applyCancellationToFinishState(DecodingInferenceContext& context, int32_t row)
{
    auto& s = context.slotStreams[row];
    if (context.finishedStates[row])
    {
        return; // First-writer-wins: don't overwrite a natural-finish reason.
    }
    bool const channelCancelled = s.channel != nullptr && s.channel->mCancelled.load(std::memory_order_acquire);
    bool const tokenCancelled = s.cancelToken != nullptr && s.cancelToken->load(std::memory_order_acquire);
    if (channelCancelled || tokenCancelled)
    {
        context.finishedStates[row] = 1;
        s.terminalReason = FinishReason::kCancelled;
        LOG_DEBUG("Batch %d finished, reason: cancel", row);
    }
}

void applyCancellationToFinishStates(DecodingInferenceContext& context)
{
    for (int32_t i = 0; i < context.activeBatchSize; ++i)
    {
        applyCancellationToFinishState(context, i);
    }
}

void decodeSlot(DecodingInferenceContext& context, int32_t row, tokenizer::Tokenizer const& tok)
{
    auto& s = context.slotStreams[row];
    auto const& slotStops = context.stopStringsPerSlot[row];
    bool const hasChannel = (s.channel != nullptr);
    bool const slotStopsEnabled = !slotStops.empty();

    s.stopMatchedThisIter = false;
    s.pendingEmitText.clear();

    // Fast path: row needs neither chunk emission nor stop detection.
    if (!hasChannel && !slotStopsEnabled)
    {
        return;
    }

    bool const isFinal = context.finishedStates[row] != 0;

    // Stream-interval gating; non-streaming slots run detection every iter.
    if (hasChannel)
    {
        int32_t const streamInterval = s.channel->getStreamInterval();
        size_t const newlyAvailable = context.tokenIds[row].size() - s.sentTokenCount;
        bool const intervalMet = static_cast<int32_t>(newlyAvailable) >= streamInterval;
        if (!isFinal && !intervalMet)
        {
            return;
        }
    }

    bool const skipSpecial = hasChannel ? s.channel->getSkipSpecialTokens() : s.skipSpecialTokens;
    std::string delta = tokenizer::emitDelta(s, tok, context.tokenIds[row], skipSpecial);
    if (isFinal && !s.pendingBytes.empty())
    {
        delta.append(tokenizer::emitDeltaFlush(s));
    }

    // Stop match overrides kEndId/kLength but not kCancelled/kError. The
    // override itself is applied by applyStopMatch based on stopMatchedThisIter.
    bool const stopActive
        = slotStopsEnabled && s.terminalReason != FinishReason::kCancelled && s.terminalReason != FinishReason::kError;
    if (stopActive)
    {
        s.stopMatchBuffer.append(delta);
        auto outcome = applyStopStringMatch(s.stopMatchBuffer, slotStops, s.maxStopLen, isFinal);
        s.pendingEmitText = std::move(outcome.emitted);
        s.stopMatchedThisIter = outcome.stopMatched;
    }
    else
    {
        s.pendingEmitText = std::move(delta);
    }
}

void decodePerSlot(DecodingInferenceContext& context, tokenizer::Tokenizer const& tok)
{
    for (int32_t i = 0; i < context.activeBatchSize; ++i)
    {
        decodeSlot(context, i, tok);
    }
}

namespace
{
//! -1 until the row's prefill has sampled a token: before that its effective prefill length is not yet set, and
//! the difference would count the whole prompt as reused.
int32_t reusedPromptTokenCountOf(DecodingInferenceContext const& context, size_t row) noexcept
{
    if (row >= context.currentGenerateLengths.size() || context.currentGenerateLengths[row] == 0
        || row >= context.rawBatchedInputIds.size() || row >= context.effectivePrefillLengths.size())
    {
        return -1;
    }
    int32_t const pruned = row < context.prunedPrefillTokens.size() ? context.prunedPrefillTokens[row] : 0;
    return reusedPromptTokens(
        static_cast<int32_t>(context.rawBatchedInputIds[row].size()), context.effectivePrefillLengths[row], pruned);
}
} // namespace

void emitChunk(DecodingInferenceContext& context, int32_t row, tokenizer::Tokenizer const& tokenizer)
{
    auto& s = context.slotStreams[row];
    if (!s.channel)
    {
        return; // non-streaming: output assembled at handleRequest finalization
    }

    bool const isFinal = context.finishedStates[row] != 0;
    // EOS/length can be set after decodePerSlot held back a possible stop prefix.
    if (isFinal && (s.terminalReason == FinishReason::kEndId || s.terminalReason == FinishReason::kLength))
    {
        s.pendingEmitText.append(s.stopMatchBuffer);
        s.stopMatchBuffer.clear();
    }
    if (s.pendingEmitText.empty() && !isFinal)
    {
        return;
    }

    StreamChunk chunk;
    if (s.channel->getCollectMetrics())
    {
        chunk.metrics = s.profiler.snapshot(context.effectivePrefillLengths[row]);
    }
    chunk.promptTokenCount = static_cast<int32_t>(context.rawBatchedInputIds[row].size());
    chunk.reusedPromptTokenCount = reusedPromptTokenCountOf(context, static_cast<size_t>(row));
    // Delta tokens span [lastEmittedTokenCount, sentTokenCount) — accumulates
    // across prior iterations where pendingEmitText was held back.
    size_t const snapLastEmitted = s.lastEmittedTokenCount;
    chunk.tokenIds.assign(context.tokenIds[row].begin() + static_cast<std::ptrdiff_t>(snapLastEmitted),
        context.tokenIds[row].begin() + static_cast<std::ptrdiff_t>(s.sentTokenCount));
    chunk.text = std::move(s.pendingEmitText);
    chunk.finished = isFinal;
    if (isFinal)
    {
        chunk.reason = s.terminalReason;
    }
    if (static_cast<size_t>(row) < context.stepLogprobs.size() && context.numLogprobsPerSlot[row] > 0)
    {
        rt::LogprobsSlot const& logprobs = context.stepLogprobs[row];
        int32_t const topK = context.numLogprobsPerSlot[row];
        size_t const newTokens = s.sentTokenCount - snapLastEmitted;
        for (size_t j = 0; j < newTokens; ++j)
        {
            size_t const stepIdx = s.logprobsEmittedCount + j;
            if (stepIdx >= static_cast<size_t>(logprobs.numSteps))
            {
                break;
            }
            chunk.logprobs.push_back(logprobEntriesForStep(logprobs, static_cast<int32_t>(stepIdx), topK, tokenizer));
        }
        s.logprobsEmittedCount += chunk.logprobs.size();
    }

    s.channel->push(std::move(chunk));
    s.lastEmittedTokenCount = s.sentTokenCount;
    if (isFinal)
    {
        s.channel->finish(s.terminalReason);
    }
}

std::vector<LogprobEntry> logprobEntriesForStep(
    LogprobsSlot const& slot, int32_t step, int32_t topK, tokenizer::Tokenizer const& tokenizer)
{
    auto const* base = slot.data.data() + static_cast<std::ptrdiff_t>(step) * topK;
    std::vector<LogprobEntry> entries;
    entries.reserve(static_cast<size_t>(topK));
    for (int32_t k = 0; k < topK; ++k)
    {
        if (base[k].first < 0)
        {
            break;
        }
        entries.push_back({base[k].first, base[k].second, tokenizer.idToPiece(base[k].first)});
    }
    return entries;
}

void emitChunks(DecodingInferenceContext& context, tokenizer::Tokenizer const& tokenizer)
{
    for (int32_t i = 0; i < context.activeBatchSize; ++i)
    {
        emitChunk(context, i, tokenizer);
    }
}

void emitTokenCallbackForSlot(DecodingInferenceContext& context, int32_t row)
{
    if (!context.onTokenGenerated.has_value())
    {
        return;
    }
    auto const& slotTokens = context.tokenIds[row];
    if (slotTokens.empty() || static_cast<size_t>(row) >= context.callbackEmittedTokenCounts.size())
    {
        return;
    }

    auto const& callback = context.onTokenGenerated.value();
    int32_t emittedCount = context.callbackEmittedTokenCounts[row];
    emittedCount = std::clamp(emittedCount, 0, static_cast<int32_t>(slotTokens.size()));
    bool const slotFinished = context.finishedStates[row] != 0;
    for (int32_t tokenIdx = emittedCount; tokenIdx < static_cast<int32_t>(slotTokens.size()); ++tokenIdx)
    {
        bool const isLastNewToken = tokenIdx + 1 == static_cast<int32_t>(slotTokens.size());
        callback(TokenCallbackInfo{slotTokens[tokenIdx], row, context.generationRound, slotFinished && isLastNewToken});
    }
    context.callbackEmittedTokenCounts[row] = static_cast<int32_t>(slotTokens.size());
}

void emitTokenCallbacks(DecodingInferenceContext& context)
{
    if (!context.onTokenGenerated.has_value())
    {
        return;
    }
    for (int32_t i = 0; i < context.activeBatchSize; ++i)
    {
        emitTokenCallbackForSlot(context, i);
    }
}

//=============================================================================
// StreamChannelFinalizer — RAII terminal-chunk guarantee
//=============================================================================

StreamChannelFinalizer::StreamChannelFinalizer(DecodingInferenceContext& ctx, tokenizer::Tokenizer const& tok) noexcept
    : mCtx(ctx)
    , mTok(tok)
{
}

StreamChannelFinalizer::~StreamChannelFinalizer() noexcept
{
    // Destructor runs during unwind on every exit path from handleRequest.
    // Allocations in chunk assembly may throw on OOM — swallow them because the
    // essential invariant (consumer must unblock) is carried by finish() alone.
    for (size_t i = 0; i < mCtx.slotStreams.size(); ++i)
    {
        auto& s = mCtx.slotStreams[i];
        if (!s.channel)
        {
            continue;
        }
        if (s.channel->isFinished())
        {
            continue; // Already terminated via the normal emit path.
        }

        try
        {
            StreamChunk chunk;
            if (i < mCtx.effectivePrefillLengths.size())
            {
                chunk.metrics = s.profiler.snapshot(mCtx.effectivePrefillLengths[i]);
            }
            if (i < mCtx.rawBatchedInputIds.size())
            {
                chunk.promptTokenCount = static_cast<int32_t>(mCtx.rawBatchedInputIds[i].size());
            }
            chunk.reusedPromptTokenCount = reusedPromptTokenCountOf(mCtx, i);

            // Any tokens generated but not yet pushed (held by streamInterval
            // or appended just before the aborting exit path).
            if (i < mCtx.tokenIds.size() && s.lastEmittedTokenCount < mCtx.tokenIds[i].size())
            {
                chunk.tokenIds.assign(mCtx.tokenIds[i].begin() + static_cast<std::ptrdiff_t>(s.lastEmittedTokenCount),
                    mCtx.tokenIds[i].end());
            }
            if (i < mCtx.stepLogprobs.size() && mCtx.numLogprobsPerSlot[i] > 0)
            {
                rt::LogprobsSlot const& slot = mCtx.stepLogprobs[i];
                int32_t const topK = mCtx.numLogprobsPerSlot[i];
                for (size_t stepIdx = s.logprobsEmittedCount; stepIdx < static_cast<size_t>(slot.numSteps); ++stepIdx)
                {
                    auto const* base = slot.data.data() + static_cast<std::ptrdiff_t>(stepIdx) * topK;
                    std::vector<LogprobEntry> entries;
                    entries.reserve(topK);
                    for (int32_t k = 0; k < topK; ++k)
                    {
                        entries.push_back({base[k].first, base[k].second, mTok.idToPiece(base[k].first)});
                    }
                    chunk.logprobs.push_back(std::move(entries));
                }
            }

            // Decode any un-decoded tokens via idToPiece, then sanitize+flush.
            bool const skipSpecial = s.channel->getSkipSpecialTokens();
            std::string raw;
            if (i < mCtx.tokenIds.size())
            {
                for (size_t k = s.sentTokenCount; k < mCtx.tokenIds[i].size(); ++k)
                {
                    raw.append(mTok.idToPiece(mCtx.tokenIds[i][k], skipSpecial));
                }
            }
            chunk.text = utf8::sanitizeUtf8Streaming(raw, s.pendingBytes);
            chunk.text.append(utf8::sanitizeUtf8Flush(s.pendingBytes));

            chunk.finished = true;
            chunk.reason = FinishReason::kError;
            s.channel->push(std::move(chunk));
        }
        catch (...)
        {
            // Best-effort push; finish() below still unblocks the consumer.
        }

        s.channel->finish(FinishReason::kError);
    }
}

// Explicit instantiation lives next to the template definition (in
// llmRuntimeUtils.cpp). It's forward-declared there with SlotStreamState as
// a complete type via this header's include.

} // namespace rt
} // namespace trt_edgellm
