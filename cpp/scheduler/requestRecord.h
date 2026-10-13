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
#include <memory>
#include <mutex>
#include <string>

namespace trt_edgellm
{
namespace rt
{
namespace scheduler
{

//! Everything about one in-flight request that both the caller and the actor need to see.
//!
//! Held by shared_ptr on both sides on purpose: the caller may drop its RequestHandle while the
//! actor is still mid-request, and the actor must still have somewhere to put the outcome. It also
//! means a cancel arriving after retirement finds nothing rather than dangling.
struct RequestRecord
{
    //! When submit() accepted the request; the queue-latency metric measures from here to the
    //! moment the actor starts serving it (founding or mid-flight admission).
    std::chrono::steady_clock::time_point submittedAt{};

    RequestId id{kInvalidRequestId};

    //! The channel the runtime itself writes tokens into, handed straight to the caller.
    //!
    //! Deliberately the runtime's own type rather than a scheduler-owned one. The runtime already
    //! delivers per slot: it pushes through slotStreams[i].channel, reads cancellation from the
    //! same place at the top of every step, and moves the entry when the batch is compacted. A
    //! separate channel would have to be bridged onto this one for no gain.
    std::shared_ptr<StreamChannel> channel;
    //! The channel the runtime actually polls for the cancel flag: the caller's own first channel
    //! when the request brought one, this record's otherwise. Cancellation must set the flag here
    //! or a caller-supplied channel would leave a running request uncancellable.
    std::shared_ptr<StreamChannel> runtimeChannel;

    LLMGenerationResponse response;
    TerminalStatus status{TerminalStatus::kCompleted};

    //! Set only when status is kExecutionError, so get() can rethrow something actionable.
    std::string errorMessage;

    //! Publishes the three fields above, and wakes a caller waiting on them.
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

    //! @brief Actor: publish the outcome and release anyone waiting for it.
    void publishOutcome(TerminalStatus terminalStatus, std::string message = {})
    {
        {
            std::lock_guard<std::mutex> lock(outcomeMutex);
            status = terminalStatus;
            errorMessage = std::move(message);
            outcomePublished.store(true, std::memory_order_release);
        }
        outcomeReady.notify_all();
    }

    //! Wake a get() parked on the outcome without publishing one: it re-evaluates the channel's
    //! cancel flag, which it treats as terminal for an unfinished request. The empty critical
    //! section orders the wake after the waiter's predicate check.
    void wakeWaiters()
    {
        {
            std::lock_guard<std::mutex> lock(outcomeMutex);
        }
        outcomeReady.notify_all();
    }
};

} // namespace scheduler
} // namespace rt
} // namespace trt_edgellm
