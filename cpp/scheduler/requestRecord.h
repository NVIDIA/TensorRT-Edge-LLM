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

#include "runtime/llmRuntimeUtils.h"
#include "runtime/streaming.h"
#include "scheduler/requestIdentity.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>

namespace trt_edgellm
{
namespace rt
{
namespace scheduler
{

//! Scheduling statistics of one request, written by the actor and readable once the outcome is published.
struct RequestStats
{
    //! The request was admitted into the session; the fields below stay zero otherwise.
    bool admitted{false};
    //! From submit() to admission.
    uint64_t queueTimeUs{0};
    //! From submit() to the end of the step that sampled the first token; zero when none was sampled.
    uint64_t timeToFirstTokenUs{0};
    //! Prompt positions as tokenized, and the prefix of them served by the context cache.
    int32_t promptTokens{0};
    int32_t reusedPromptTokens{0};
    //! Steps that ran a chunk of this request's prompt (1 when the prompt ran in one step).
    int32_t promptChunks{0};
    //! Steps that withheld this request under preemption=pause, and the host time of those steps.
    int32_t pausedSteps{0};
    uint64_t pausedTimeUs{0};
};

//! Everything about one in-flight request that both the caller and the actor need to see.
//!
//! Held by shared_ptr on both sides on purpose: the caller may drop its RequestHandle while the
//! actor is still mid-request, and the actor must still have somewhere to put the outcome. It also
//! means a cancel arriving after retirement finds nothing rather than dangling.
struct RequestRecord
{
    //! When submit() accepted the request; the queue-latency metric measures from here to the
    //! moment the actor admits it.
    std::chrono::steady_clock::time_point submittedAt{};

    RequestId id{kInvalidRequestId};

    //! The channel the runtime itself writes tokens into, handed straight to the caller.
    //!
    //! Deliberately the runtime's own type rather than a scheduler-owned one. The runtime already
    //! delivers per slot: it pushes through slotStreams[i].channel and moves the entry when the
    //! batch is compacted. A separate channel would have to be bridged onto this one for no gain.
    //! One canonical channel per request: the caller's own when it brought one, the engine's
    //! otherwise, so the channel the runtime finishes is the one stream() hands out.
    std::shared_ptr<StreamChannel> channel;

    //! The request's cancellation, as one bit the request owns: set by cancel() on the caller's
    //! thread, read by the actor before admission and at every drain, and polled by the runtime at
    //! the top of every step the sequence takes part in (it travels into the slot at admission).
    //! Nothing rides the command queue, so overload cannot lose a cancel; and it is not a property
    //! of the token channel, whose two other jobs point the other way.
    std::shared_ptr<std::atomic<bool>> cancelToken{std::make_shared<std::atomic<bool>>(false)};

    LLMGenerationResponse response;
    TerminalStatus status{TerminalStatus::kCompleted};

    //! Set only when status is kExecutionError, so get() can rethrow something actionable.
    std::string errorMessage;
    FailureKind failureKind{FailureKind::kOther};

    //! Publishes the four fields above, and wakes a caller waiting on them.
    //!
    //! This, not the channel's terminal flag, is what says a request is over. The two answer
    //! different questions and are set by different parties: the runtime finishes the channel when
    //! the token stream ends, while the actor records the outcome after execution returns -- and
    //! for a request that never reached the runtime at all, the channel is never touched. Treating
    //! a terminal channel as "the outcome is readable" would sometimes read a response that had not
    //! been written, and would leave a discarded request waiting forever.
    std::mutex outcomeMutex;
    std::condition_variable outcomeReady;
    std::atomic<bool> outcomePublished{false};

    //! Actor-owned until the outcome is published: every write precedes publishOutcome(), so a
    //! reader that observed outcomePublished sees the final values.
    RequestStats stats;

    //! Actor-owned: planned compact steps in a row that paused this request, saturating; zero after any step that
    //! did not (see SchedulingOptions::maxPauseSteps).
    int32_t consecutivePausedSteps{0};

    //! Actor-owned: the request was queued again after a refused compact step (see RequestEngine::failSession), so
    //! its queue latency is already counted.
    bool requeued{false};

    //! @brief Actor: publish the outcome and release anyone waiting for it.
    void publishOutcome(TerminalStatus terminalStatus, std::string message = {}, FailureKind kind = FailureKind::kOther)
    {
        {
            std::lock_guard<std::mutex> lock(outcomeMutex);
            status = terminalStatus;
            errorMessage = std::move(message);
            failureKind = kind;
            outcomePublished.store(true, std::memory_order_release);
        }
        outcomeReady.notify_all();
    }

    bool cancelRequested() const noexcept
    {
        return cancelToken->load(std::memory_order_acquire);
    }

    //! @brief Any thread: ask for this request to stop, and release a caller parked in get().
    //!
    //! Plants the token, then cancels the channel too: that wakes a streaming consumer parked in
    //! waitPop() and lets a consumer that reads only the channel see the cancellation. The empty
    //! critical section orders the wake after a waiter's predicate check, so a get() cannot miss
    //! it. Idempotent.
    void cancel() noexcept
    {
        cancelToken->store(true, std::memory_order_release);
        if (channel != nullptr)
        {
            channel->cancel();
        }
        {
            std::lock_guard<std::mutex> lock(outcomeMutex);
        }
        outcomeReady.notify_all();
    }
};

} // namespace scheduler
} // namespace rt
} // namespace trt_edgellm
