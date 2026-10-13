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
#include "runtime/runtimeStepper.h"
#include "scheduler/engineChannel.h"
#include "scheduler/requestCoordinator.h"
#include "scheduler/requestIdentity.h"
#include "scheduler/requestRecord.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <unordered_map>

namespace trt_edgellm
{
namespace rt
{
class LLMInferenceRuntime;

namespace scheduler
{

//! Thrown by submit() when the request is refused outright rather than queued.
//!
//! Refusal is not an error condition of the engine: a bounded queue is the backpressure signal, and
//! the caller is expected to translate this into whatever its transport calls "busy".
class SubmitError : public std::runtime_error
{
public:
    explicit SubmitError(std::string const& message)
        : std::runtime_error(message)
    {
    }
};

struct EngineConfig
{
    //! Commands the queue holds before submit() starts refusing. Rounded up to a power of two and to
    //! CommandQueue::kMinCapacity.
    int32_t commandQueueCapacity{64};

    //! Requests accepted but not yet terminal, across the queue and the actor's own backlog.
    //! Bounding the command queue alone does not bound this: each boundary moves a queue's worth of
    //! commands into actor-owned state, so without this the backlog grows without limit and the
    //! backpressure the bounded queue is supposed to provide never materialises.
    int32_t maxPendingRequests{256};

    //! Requests allowed to be resident at once.
    //!
    //! Above 1, arriving requests join the running batch at step boundaries (in-flight batching),
    //! which needs the batch-aware executor; an engine built on the plain per-request executor has
    //! nowhere to admit into and rejects values above 1 at construction. This is also the KV
    //! capacity bound: page tables partition their pool per slot, so a request beyond this count
    //! has no pages to run on and waits in the queue -- which is the V0 self-exit story, no
    //! eviction of others required.
    int32_t maxBatchSize{1};

    //! How long an idle actor sleeps before waking to look around again. It does not affect a busy
    //! engine, which never reaches the park at all.
    //!
    //! Every operation that gives the actor something to do also wakes it, so this timeout is a
    //! backstop rather than the mechanism: it bounds the damage if a future caller changes actor
    //! state without waking it, turning a permanent hang into a delay of this length.
    std::chrono::milliseconds idleWait{50};
};

//! A point-in-time copy of the engine's counters. Each field is read with relaxed ordering, so
//! the snapshot is per-field accurate but not a single atomic cut -- fine for observability,
//! not for control flow.
struct EngineMetrics
{
    //! Requests submit() accepted / refused (queue full, backlog full, or shutting down).
    uint64_t submitted{};
    uint64_t refused{};
    //! Boundaries where the queue head could not join the running batch, by reason. High stall
    //! counts quantify the cost of FCFS head blocking under a mixed workload.
    uint64_t stallsIncompatible{};
    uint64_t stallsGuided{};
    uint64_t stallsFounderOnly{};
    uint64_t stallsNoCapacity{};
    //! Requests that joined a running batch mid-flight (excludes founders).
    uint64_t admittedMidFlight{};
    //! Running totals of terminal outcomes across all requests, one per TerminalStatus value:
    //! countOutcome() bumps exactly one of them each time a request retires, so they always sum
    //! to the outcomes callers saw. A single request's own state is the TerminalStatus on its
    //! record, not anything here.
    uint64_t completed{};
    uint64_t cancelled{};
    uint64_t failed{};
    //! Time from submit() to the actor starting to serve the request (founding or admission).
    uint64_t queueLatencyTotalUs{};
    uint64_t queueLatencyMaxUs{};
    uint64_t queueLatencyCount{};
};

class RequestEngine;

//! A caller's claim on one in-flight request. Move-only: a channel has exactly one consumer,
//! and copying the handle would quietly create a second one.
class RequestHandle
{
public:
    RequestHandle() = default;
    RequestHandle(std::shared_ptr<EngineChannel> channel, std::shared_ptr<RequestRecord> record) noexcept;

    RequestHandle(RequestHandle&&) noexcept;
    RequestHandle& operator=(RequestHandle&&) noexcept;
    RequestHandle(RequestHandle const&) = delete;
    RequestHandle& operator=(RequestHandle const&) = delete;
    ~RequestHandle() = default;

    RequestId id() const noexcept;

    //! @brief The token channel for this request, as the runtime writes it. Valid while the handle is.
    //!
    //! Tokens appear here as they are produced. To cancel, use RequestHandle::cancel() or
    //! RequestEngine::cancel(): cancelling the channel directly also stops generation at the next
    //! step boundary, but a get() waiting on this request is only released once the actor reports
    //! the outcome.
    StreamChannel& stream();

    //! @brief The same channel with shared ownership, for callers that outlive the handle.
    //!
    //! The pybind layer holds StreamChannel through a shared_ptr holder, so handing it the
    //! reference above would manufacture a second owner of the same object. Anything else
    //! should prefer stream().
    std::shared_ptr<StreamChannel> streamShared() const noexcept;

    //! @brief Block until the request reaches a terminal state, then hand back its response.
    //! @throws std::runtime_error if the request ended in kExecutionError
    //! @throws std::runtime_error if the request was cancelled
    LLMGenerationResponse get();

    //! @brief True once the request has an outcome, so get() would return without blocking.
    //!
    //! Distinct from the channel being finished, which reports only that the token stream ended.
    bool ready() const noexcept;

    //! @brief Ask the engine to stop this request. Non-blocking and idempotent; a request that has
    //!        already retired simply does not match anything the actor still holds.
    //!
    //! Releases this caller immediately, but does not interrupt a request the actor has already
    //! started: that request runs to completion and is then reported as cancelled.
    void cancel() noexcept;

    bool valid() const noexcept
    {
        return mRecord != nullptr;
    }

private:
    //! The early exit of get(): the caller cancelled, and the stream never finished.
    bool cancelledBeforeFinish() const noexcept;

    //! Shared rather than a pointer back to the engine: a handle may outlive it, and cancelling
    //! through a destroyed engine would be a use-after-free.
    std::shared_ptr<EngineChannel> mChannel;
    std::shared_ptr<RequestRecord> mRecord;
};

//! Owns the runtime and the one thread allowed to touch it.
//!
//! submit() and cancel() are callable from any thread: each posts a command to the queue and
//! returns without waiting for the actor. shutdown() is the exception — it joins, so it blocks and
//! must not be called from the actor itself. The actor starts in the constructor and is joined by
//! shutdown(), which the destructor implies, so there is no start() to forget.
//!
//! What the actor serialises is feeding the GPU, which has to be serial anyway. Callers never block
//! on each other, and no caller ever holds a pointer to the runtime, so a data race on it cannot be
//! written rather than being prevented by a lock the caller has to remember to take.
class RequestEngine
{
public:
    //! How the actor opens one founding request under the stepped control plane.
    //!
    //! In production this is bound to LLMInferenceRuntime::beginStepped. It is a seam rather than
    //! a direct call so the actor's admission, cancellation and shutdown paths can be tested
    //! without a GPU: LLMInferenceRuntime is a concrete class and needs a real CUDA stream to
    //! assemble, which would put every one of those tests behind a device. A null return means
    //! the runtime refused the request; a throw carries the reason.
    using SteppedFactory = std::function<std::unique_ptr<SteppedExecution>(LLMGenerationRequest&, cudaStream_t)>;

    //! @param runtime Taken over outright. Nothing else may hold or call it afterwards.
    //! @param stream The CUDA stream the actor drives the runtime on.
    //! @throws std::runtime_error if @p runtime spans more than one rank: the stepped control plane
    //!         runs on a single rank in this release.
    RequestEngine(std::unique_ptr<LLMInferenceRuntime> runtime, cudaStream_t stream, EngineConfig config = {});

    //! @brief Test seam: an engine with no runtime, whose founding batches are whatever
    //!        @p beginStepped hands back.
    RequestEngine(SteppedFactory beginStepped, EngineConfig config);

    ~RequestEngine();

    RequestEngine(RequestEngine const&) = delete;
    RequestEngine& operator=(RequestEngine const&) = delete;
    RequestEngine(RequestEngine&&) = delete;
    RequestEngine& operator=(RequestEngine&&) = delete;

    //! @brief Any thread: hand a request to the actor. Does not wait for it to start.
    //! @throws SubmitError if the engine is shutting down or the command queue is full
    RequestHandle submit(LLMGenerationRequest request);

    //! @brief Any thread: cancel a request by id. Same effect and same guarantees as
    //! RequestHandle::cancel(): the flag is planted immediately on the caller's thread and nothing
    //! rides the command queue, so it cannot be lost under overload. Idempotent; unknown or retired
    //! ids are ignored (ids are never reused).
    void cancel(RequestId id);

    //! @brief Submit one request and block until it finishes, in the shape of the old blocking call.
    //!
    //! Exists so a caller written against LLMInferenceRuntime::handleRequest can move onto the
    //! engine without being restructured first: it reports failure by returning false rather than
    //! throwing, and leaves @p response untouched on failure.
    //!
    //! This is a convenience over submit() plus RequestHandle::get(), not a second execution path.
    //! It gives up everything the engine exists to provide -- the caller occupies a thread for the
    //! whole generation and cannot see tokens as they arrive -- so new code should submit instead.
    //!
    //! @return false if the request was refused, cancelled, or failed during execution.
    bool handleRequest(LLMGenerationRequest const& request, LLMGenerationResponse& response) noexcept;

    //! @brief Any thread except the actor's: stop the actor and join it. Idempotent.
    //!
    //! Unlike the other operations this one blocks, because it joins. kDrain lets every queued
    //! request run; kCancel abandons what is still queued. Neither can interrupt a request already
    //! executing -- the actor is inside it and cannot reach a boundary -- so both wait at least
    //! that long. Both leave every live stream terminal, so no caller stays parked in waitPop().
    //!
    //! The first caller's mode wins; a later kCancel cannot escalate a drain already under way.
    void shutdown(ShutdownMode mode);

    //! @brief Requests the actor has taken off the queue but not yet admitted. Observability only.
    //!
    //! Not a count of everything submitted: a request stays invisible here until the actor reaches a
    //! boundary and drains it, which it cannot do while it is inside a request. Do not use this to wait
    //! for a submission to become visible.
    int32_t queued() const noexcept;

    //! @brief Requests currently holding runtime resources. Observability only.
    int32_t resident() const noexcept;

    //! @brief False once shutdown() has been called.
    bool running() const noexcept;

    //! @brief Counters for observability; see EngineMetrics for the consistency contract.
    EngineMetrics metrics() const noexcept;

private:
    //! Shared tail of both constructors: validate config, then start the actor.
    void start();

    //! Signal the actor that there is something to look at, safely against it being midway
    //! between its predicate check and its wait.
    void wakeActor() noexcept;

    void actorLoop();

    //! The stages the actor cycles through, kept as separate members rather than inlined into
    //! actorLoop() because the boundaries between them are where the engine's invariants hold: no
    //! request is scheduled until the queue has been folded in, nothing becomes resident outside
    //! admission, and the runtime is reached from exactly one of them.

    //! Fold everything the queue holds into actor-owned state. Runs before admission each round, so
    //! a cancel posted since the previous round is seen before its request could be promoted.
    void drainCommandQueue();
    //! Plant the cancel flag on every resident and joiner's runtime channel (idempotent). Called
    //! from drainCommandQueue() whenever a cancelling shutdown is in effect.
    void cancelResidentWork();

    //! Retire every queued request whose cancel flag is set: terminal kCancelled, backlog
    //! reservation released. Runs at every drain and again right before each admission.
    void retireCancelledQueued();

    //! Promote the oldest queued request, leaving it resident. Declines when something is already
    //! resident, when nothing is queued, or when a cancelling shutdown means the request would
    //! never get to finish anyway.
    bool admitNextRequest();

    //! The control plane: the actor owns the loop and drives the runtime's typed stepper tick by
    //! tick -- no callback, no hook. The only member that reaches the runtime, and the only one
    //! that can occupy the actor for the length of a generation. mResidents is the canonical owner
    //! of resident identity for the batch, and its commit() is where StepResults become outcomes.
    void runResidentSteppedRequest();

    //! The head-of-line gate: batch compatibility, guided decoding, and the founder-only options.
    //! Returns false and bumps the matching stall counter when the head must wait for its own
    //! batch.
    bool headMayJoin(LLMGenerationRequest const& head);

    //! Publish coordinator outcomes: terminate each record with the mapped status and drop it
    //! from the live set.
    void publishOutcomes(std::vector<RequestCoordinator::TerminalOutcome> outcomes);

    //! Record the outcome and release anyone waiting on it. The finish reason is the runtime's to
    //! write, and it puts it in the channel itself.
    void terminate(RequestRecord& record, TerminalStatus status, std::string errorMessage = {});

    struct QueuedRequest
    {
        std::shared_ptr<RequestRecord> record;
        std::unique_ptr<LLMGenerationRequest> request;
    };

    std::unique_ptr<LLMInferenceRuntime> mRuntime;
    SteppedFactory mBeginStepped;
    cudaStream_t mStream{};
    EngineConfig const mConfig;

    //! Shared with every outstanding handle so that cancelling through an expired engine is safe.
    std::shared_ptr<EngineChannel> mChannel;

    //! Handed out by submit(), never reused. Starts at 1 so kInvalidRequestId stays distinguishable.
    std::atomic<RequestId> mNextRequestId{1};

    std::atomic<bool> mStopRequested{false};

    //! Atomic and written by shutdown() directly rather than carried only by the command: when the
    //! queue is full the command cannot be posted, and a kCancel that silently degraded to kDrain
    //! would make the destructor wait out an entire backlog.
    std::atomic<ShutdownMode> mShutdownMode{ShutdownMode::kDrain};

    //! The live counters behind EngineMetrics: field for field the same struct, wrapped in
    //! std::atomic because the actor and every submit() thread update them concurrently.
    //! std::atomic is neither copyable nor movable, so this cannot also be the value metrics()
    //! hands out; metrics() copies each field into a plain EngineMetrics instead, which is the one
    //! place the two are joined. Every update is relaxed.
    struct Counters
    {
        std::atomic<uint64_t> submitted{0};
        std::atomic<uint64_t> refused{0};
        std::atomic<uint64_t> stallsIncompatible{0};
        std::atomic<uint64_t> stallsGuided{0};
        std::atomic<uint64_t> stallsFounderOnly{0};
        std::atomic<uint64_t> stallsNoCapacity{0};
        std::atomic<uint64_t> admittedMidFlight{0};
        std::atomic<uint64_t> completed{0};
        std::atomic<uint64_t> cancelled{0};
        std::atomic<uint64_t> failed{0};
        std::atomic<uint64_t> queueLatencyTotalUs{0};
        std::atomic<uint64_t> queueLatencyMaxUs{0};
        std::atomic<uint64_t> queueLatencyCount{0};
    };
    mutable Counters mCounters;

    void countOutcome(TerminalStatus status) noexcept;
    void recordQueueLatency(RequestRecord const& record) noexcept;

    //! Actor-owned from here down. No lock guards these because no other thread reaches them.
    std::deque<QueuedRequest> mQueued;
    //! Every resident of the running batch -- the founder and its joiners -- keyed by ResidentRef.
    //! The one ledger: commit() retires through it, a cancelling shutdown reaches every runtime
    //! channel through it, and it is empty between batches.
    RequestCoordinator mResidents;
    //! First original index available to admissions in the current batch; founding sequences own
    //! the indices below it.
    int32_t mAdmittedIndexBase{0};
    int32_t mNextAdmittedIndex{0};
    std::shared_ptr<RequestRecord> mResidentRecord;
    std::unique_ptr<LLMGenerationRequest> mResidentRequest;
    std::unordered_map<RequestId, std::shared_ptr<RequestRecord>> mLive;

    //! Published for queued()/resident(). Written by the actor, read by anyone.
    std::atomic<int32_t> mQueuedCount{0};
    std::atomic<int32_t> mResidentCount{0};

    //! Accepted and not yet terminal. Raised by submit() and lowered wherever a request reaches a
    //! terminal state, so it spans both the queue and the actor's backlog.
    std::atomic<int32_t> mPendingCount{0};

    //! Lets an idle actor sleep until a command arrives instead of spinning.
    std::mutex mWakeMutex;
    std::condition_variable mWakeCv;

    std::thread mActor;
    std::once_flag mShutdownOnce;
};

} // namespace scheduler
} // namespace rt
} // namespace trt_edgellm
