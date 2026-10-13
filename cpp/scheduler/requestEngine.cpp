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

#include "scheduler/requestEngine.h"

#include "common/checkMacros.h"
#include "common/logger.h"
#include "runtime/llmInferenceRuntime.h"
#include "scheduler/batchCompatibility.h"

#include <algorithm>
#include <chrono>
#include <utility>

namespace trt_edgellm
{
namespace rt
{
namespace scheduler
{

RequestHandle::RequestHandle(std::shared_ptr<EngineChannel> channel, std::shared_ptr<RequestRecord> record) noexcept
    : mChannel(std::move(channel))
    , mRecord(std::move(record))
{
}

RequestHandle::RequestHandle(RequestHandle&& other) noexcept = default;

RequestHandle& RequestHandle::operator=(RequestHandle&& other) noexcept = default;

RequestId RequestHandle::id() const noexcept
{
    return mRecord ? mRecord->id : kInvalidRequestId;
}

StreamChannel& RequestHandle::stream()
{
    check::check(mRecord != nullptr, "RequestHandle::stream() on a moved-from handle.");
    return *mRecord->channel;
}

std::shared_ptr<StreamChannel> RequestHandle::streamShared() const noexcept
{
    return mRecord ? mRecord->channel : nullptr;
}

LLMGenerationResponse RequestHandle::get()
{
    check::check(mRecord != nullptr, "RequestHandle::get() on a moved-from handle.");

    // The outcome flag, not the channel, is what says the request is over: a request dropped
    // before it ever reached the runtime has a channel nobody will ever finish. A caller that
    // cancelled is released as soon as its flag lands rather than when the actor, which may be
    // inside another request for a while yet, publishes the outcome. Every writer of either
    // condition notifies under outcomeMutex (publishOutcome, wakeWaiters), so a single wait
    // cannot miss a wake and needs no timeout.
    {
        std::unique_lock<std::mutex> lock(mRecord->outcomeMutex);
        mRecord->outcomeReady.wait(lock,
            [this] { return mRecord->outcomePublished.load(std::memory_order_acquire) || cancelledBeforeFinish(); });
    }
    if (!mRecord->outcomePublished.load(std::memory_order_acquire))
    {
        throw std::runtime_error("Request " + std::to_string(mRecord->id) + " was cancelled.");
    }

    switch (mRecord->status)
    {
    case TerminalStatus::kCompleted: return std::move(mRecord->response);
    case TerminalStatus::kCancelled:
        throw std::runtime_error("Request " + std::to_string(mRecord->id) + " was cancelled.");
    case TerminalStatus::kExecutionError:
        throw std::runtime_error("Request " + std::to_string(mRecord->id) + " failed: " + mRecord->errorMessage);
    }
    throw std::runtime_error("Request " + std::to_string(mRecord->id) + " ended in an unknown state.");
}

bool RequestHandle::ready() const noexcept
{
    return mRecord != nullptr && mRecord->outcomePublished.load(std::memory_order_acquire);
}

bool RequestHandle::cancelledBeforeFinish() const noexcept
{
    return mRecord->channel->isCancelled() && !mRecord->channel->isFinished();
}

void RequestHandle::cancel() noexcept
{
    if (mRecord == nullptr)
    {
        return;
    }

    // The stream flag is written here, on the caller's thread, so cancellation takes effect
    // immediately even when the actor is deep inside a request and will not reach a tick boundary
    // for seconds. The runtime reads this flag at the top of every step, so it stops a request that
    // is already executing; the actor reads it before every admission and sweeps the queue at every
    // drain, so it drops one that is still queued. Both channels get the flag: the runtime polls
    // the caller's channel, get() watches the record's. Nothing rides the command queue, so
    // overload cannot lose a cancel.
    mRecord->channel->cancel();
    if (mRecord->runtimeChannel != nullptr && mRecord->runtimeChannel != mRecord->channel)
    {
        mRecord->runtimeChannel->cancel();
    }
    mRecord->wakeWaiters();
}

RequestEngine::RequestEngine(std::unique_ptr<LLMInferenceRuntime> runtime, cudaStream_t stream, EngineConfig config)
    : mRuntime(std::move(runtime))
    , mStream(stream)
    , mConfig(config)
    , mChannel(std::make_shared<EngineChannel>(config.commandQueueCapacity))
{
    check::check(mRuntime != nullptr, "RequestEngine requires a runtime.");

    // Rejected at construction rather than discovered later. The stepped control plane runs on a
    // single rank in this release: with tensor parallelism every rank must execute the same typed
    // step, and nothing yet carries the step decisions to the other ranks, so admitting anyway
    // would end in a hung collective.
    check::check(mRuntime->supportsSteppedExecution(),
        "RequestEngine requires a single-rank runtime in this release; this one spans "
            + std::to_string(mRuntime->worldSize()) + " ranks. Tensor-parallel deployments stay on the blocking path.");

    // Same fail-fast rationale: prefillSlotInPlace refuses this deployment, so with a batch size
    // above one every admission attempt would kill the joining request at its first step. A
    // configuration that can never admit is a construction error, not a per-request one.
    if (config.maxBatchSize > 1)
    {
        check::check(mRuntime->supportsSeatedAdmission(),
            "maxBatchSize > 1 admits into a running batch, which this deployment cannot reseat mid-request "
            "(a draft engine, a diffusion backbone, or by-value Mamba state); use maxBatchSize = 1.");
    }
    // The engine's built batch dimension is the physical bound on admissions: above it, appendSlot
    // would seat a row the engine cannot run, and under a context cache every joiner is refused
    // forever. A batch that can never fill is a construction error too.
    check::check(config.maxBatchSize <= mRuntime->maxBatchSize(),
        "maxBatchSize " + std::to_string(config.maxBatchSize) + " exceeds the engine's built batch size "
            + std::to_string(mRuntime->maxBatchSize()) + ".");

    mBeginStepped = [this](LLMGenerationRequest& request, cudaStream_t stream) {
        return mRuntime->beginStepped(request, mResidentRecord->id, stream);
    };
    start();
}

RequestEngine::RequestEngine(SteppedFactory beginStepped, EngineConfig config)
    : mBeginStepped(std::move(beginStepped))
    , mConfig(config)
    , mChannel(std::make_shared<EngineChannel>(config.commandQueueCapacity))
{
    check::check(mBeginStepped != nullptr, "RequestEngine requires a stepped factory.");
    start();
}

void RequestEngine::start()
{
    check::check(mConfig.maxBatchSize >= 1, "RequestEngine requires a positive maxBatchSize.");
    check::check(mConfig.maxPendingRequests > 0, "RequestEngine pending-request limit must be positive.");
    // A non-positive idle wait would turn the actor's park into a busy loop.
    check::check(mConfig.idleWait.count() > 0, "RequestEngine idle wait must be positive.");

    // Last thing before the engine is usable: the actor must not observe a half-built engine.
    mActor = std::thread(&RequestEngine::actorLoop, this);
}

RequestEngine::~RequestEngine()
{
    // Cancel rather than drain, so the destructor abandons the backlog instead of running it. It
    // still waits for a request already executing, because the actor cannot be interrupted inside
    // one; what this avoids is waiting for everything queued behind it. Say so when it happens:
    // the callers only ever see "was cancelled" from get(), which does not tell them why.
    int32_t const inFlight = mPendingCount.load(std::memory_order_acquire);
    if (inFlight > 0)
    {
        LOG_WARNING(
            "RequestEngine destroyed with %d request(s) still in flight; they are cancelled and their callers "
            "see \"was cancelled\" from get(). Call shutdown(ShutdownMode::kDrain) first to let them finish.",
            inFlight);
    }
    shutdown(ShutdownMode::kCancel);
}

RequestHandle RequestEngine::submit(LLMGenerationRequest request)
{
    // Rejected at submit: this version of in-flight batching does not support per-request
    // LoRA. This is deliberately not a SubmitError -- that type means transient
    // backpressure and callers may retry, while this request would be refused forever. The
    // loraWeightsName compatibility field stays in BatchCompatibility's compared list so the
    // invariant is already wired should per-request LoRA be supported later.
    if (!request.loraWeightsName.empty())
    {
        throw std::invalid_argument(
            "V0 in-flight batching does not support per-request LoRA; submit rejects it outright.");
    }
    if (request.requests.size() != 1)
    {
        throw std::invalid_argument("RequestEngine::submit() takes exactly one sequence per request.");
    }
    // The same permanent-refusal semantics for what the stepped plane does not serve yet. Each of
    // these carries response state that only the runtime's drain-time assembly fills, so admitting
    // it would return a silently incomplete result; the message names where the request belongs
    // until its stepped implementation lands (see the IFB support matrix in the docs).
    if (request.generateAudio)
    {
        throw std::invalid_argument(
            "V0 in-flight batching does not serve speech output; run generateAudio requests through the "
            "blocking LLMInferenceRuntime path.");
    }
    if (request.saveSystemPromptKVCache)
    {
        throw std::invalid_argument(
            "V0 in-flight batching does not serve the legacy system-prompt capture; warm the prompt up through "
            "the blocking path before handing the runtime to the engine, or rely on the context cache.");
    }
    if (request.requests.front().pastTrajectory.has_value())
    {
        throw std::invalid_argument(
            "V0 in-flight batching does not serve trajectory execution; run pastTrajectory requests through "
            "the blocking LLMInferenceRuntime path.");
    }

    // Reserve first: a request that would exceed the backlog limit must not consume a queue slot.
    int32_t pending = mPendingCount.load(std::memory_order_relaxed);
    do
    {
        if (pending >= mConfig.maxPendingRequests)
        {
            mCounters.refused.fetch_add(1, std::memory_order_relaxed);
            throw SubmitError("RequestEngine has too many requests in flight; retry later.");
        }
    } while (!mPendingCount.compare_exchange_weak(pending, pending + 1, std::memory_order_relaxed));

    // The reservation must survive only a successful post; the allocations between here and
    // postSubmit() can throw (bad_alloc), and a leaked reservation is permanent capacity loss.
    struct ReservationGuard
    {
        std::atomic<int32_t>& count;
        bool armed{true};
        ~ReservationGuard()
        {
            if (armed)
            {
                count.fetch_sub(1, std::memory_order_relaxed);
            }
        }
    } reservation{mPendingCount};

    auto record = std::make_shared<RequestRecord>();
    record->submittedAt = std::chrono::steady_clock::now();
    record->id = mNextRequestId.fetch_add(1, std::memory_order_relaxed);
    record->channel = StreamChannel::create();

    // One canonical channel per request: the one the runtime writes and finishes is the one
    // stream() hands out and terminate() cancels. A private second channel would leave
    // stream().consume() waiting on a channel nobody finishes, and a cancel-while-queued on a
    // channel the caller never sees. The runtime requires one channel per sequence, so a request
    // with none of its own gets the engine's.
    if (request.streamChannels.empty())
    {
        request.streamChannels.assign(request.requests.size(), record->channel);
    }
    else if (request.streamChannels.front() == nullptr)
    {
        // A null entry opts that sequence out of streaming, but the runtime also reads the cancel
        // flag from the first one; give it the engine's channel so cancellation still reaches the
        // request. The other entries keep their opt-out.
        request.streamChannels.front() = record->channel;
    }
    else
    {
        record->channel = request.streamChannels.front();
    }
    record->runtimeChannel = request.streamChannels.front();

    auto command = std::make_unique<EngineCommand>();
    command->type = CommandType::kSubmit;
    command->requestId = record->id;
    command->request = std::make_unique<LLMGenerationRequest>(std::move(request));
    command->record = record;

    // Registered before the post so cancel(id) can reach the request from the moment submit()
    // returns its handle; forgotten again if the post fails.
    mChannel->registerRecord(record);

    // postSubmit() decides "accepted" and "will be drained" as one step. Splitting them would let
    // this push land after the actor's final sweep, stranding the caller on a stream nobody will
    // ever finish.
    if (!mChannel->postSubmit(std::move(command)))
    {
        mChannel->forgetRecord(record->id);
        mCounters.refused.fetch_add(1, std::memory_order_relaxed);
        throw SubmitError(mChannel->accepting() ? "RequestEngine command queue is full; retry later."
                                                : "RequestEngine is shutting down and no longer accepts requests.");
    }
    reservation.armed = false;

    mCounters.submitted.fetch_add(1, std::memory_order_relaxed);
    wakeActor();
    return RequestHandle(mChannel, std::move(record));
}

void RequestEngine::cancel(RequestId id)
{
    if (id == kInvalidRequestId)
    {
        return;
    }
    // The same flag RequestHandle::cancel() plants, found through the live-record registry, so the
    // id path is exactly as lossless as the handle path: nothing rides the bounded command queue,
    // which drops commands under overload -- the moment cancels matter most. An unknown or retired
    // id is a miss, not an error; ids are never reused, so a late cancel cannot hit a successor.
    if (mChannel->cancelRecord(id))
    {
        wakeActor(); // a queued request is dropped at the next drain rather than the next idle tick
    }
}

bool RequestEngine::handleRequest(LLMGenerationRequest const& request, LLMGenerationResponse& response) noexcept
{
    try
    {
        // Copied rather than moved: the old signature takes a const reference and callers reuse the
        // request object, while submit() needs one it can own.
        RequestHandle handle = submit(request);
        response = handle.get();
        return true;
    }
    catch (std::exception const& error)
    {
        LOG_ERROR("handleRequest failed: %s", error.what());
        return false;
    }
    catch (...)
    {
        LOG_ERROR("handleRequest failed with a non-standard exception.");
        return false;
    }
}

void RequestEngine::shutdown(ShutdownMode mode)
{
    std::call_once(mShutdownOnce, [this, mode] {
        // Close the gate first. Once this returns no submit can still be in flight, so the actor's
        // final sweep of the queue is genuinely final.
        mChannel->closeAdmission();

        // The mode travels through the atomic, not the command: a full queue drops the command,
        // and a mode lost with it would turn a cancel into a drain.
        mShutdownMode.store(mode, std::memory_order_release);
        mStopRequested.store(true, std::memory_order_release);
        LOG_INFO("RequestEngine shutting down (%s).",
            mode == ShutdownMode::kDrain ? "drain: queued requests run to completion"
                                         : "cancel: queued and resident requests are cancelled");

        auto command = std::make_unique<EngineCommand>();
        command->type = CommandType::kShutdown;
        if (!mChannel->post(std::move(command)))
        {
            LOG_WARNING("RequestEngine command queue full at shutdown; relying on the stop flag.");
        }

        wakeActor();
        if (mActor.joinable())
        {
            mActor.join();
        }
    });
}

int32_t RequestEngine::queued() const noexcept
{
    return mQueuedCount.load(std::memory_order_acquire);
}

int32_t RequestEngine::resident() const noexcept
{
    return mResidentCount.load(std::memory_order_acquire);
}

bool RequestEngine::running() const noexcept
{
    return !mStopRequested.load(std::memory_order_acquire);
}

void RequestEngine::wakeActor() noexcept
{
    // Taken and released around the notify so the actor cannot be caught between its predicate
    // check and its wait: it holds this mutex across both, so it has either not checked yet or is
    // already waiting and will see the signal.
    {
        std::lock_guard<std::mutex> lock(mWakeMutex);
    }
    mWakeCv.notify_all();
}

EngineMetrics RequestEngine::metrics() const noexcept
{
    EngineMetrics snapshot;
    snapshot.submitted = mCounters.submitted.load(std::memory_order_relaxed);
    snapshot.refused = mCounters.refused.load(std::memory_order_relaxed);
    snapshot.stallsIncompatible = mCounters.stallsIncompatible.load(std::memory_order_relaxed);
    snapshot.stallsGuided = mCounters.stallsGuided.load(std::memory_order_relaxed);
    snapshot.stallsFounderOnly = mCounters.stallsFounderOnly.load(std::memory_order_relaxed);
    snapshot.stallsNoCapacity = mCounters.stallsNoCapacity.load(std::memory_order_relaxed);
    snapshot.admittedMidFlight = mCounters.admittedMidFlight.load(std::memory_order_relaxed);
    snapshot.completed = mCounters.completed.load(std::memory_order_relaxed);
    snapshot.cancelled = mCounters.cancelled.load(std::memory_order_relaxed);
    snapshot.failed = mCounters.failed.load(std::memory_order_relaxed);
    snapshot.queueLatencyTotalUs = mCounters.queueLatencyTotalUs.load(std::memory_order_relaxed);
    snapshot.queueLatencyMaxUs = mCounters.queueLatencyMaxUs.load(std::memory_order_relaxed);
    snapshot.queueLatencyCount = mCounters.queueLatencyCount.load(std::memory_order_relaxed);
    return snapshot;
}

void RequestEngine::countOutcome(TerminalStatus status) noexcept
{
    switch (status)
    {
    case TerminalStatus::kCompleted: mCounters.completed.fetch_add(1, std::memory_order_relaxed); break;
    case TerminalStatus::kCancelled: mCounters.cancelled.fetch_add(1, std::memory_order_relaxed); break;
    case TerminalStatus::kExecutionError: mCounters.failed.fetch_add(1, std::memory_order_relaxed); break;
    }
}

void RequestEngine::recordQueueLatency(RequestRecord const& record) noexcept
{
    auto const waited = std::chrono::steady_clock::now() - record.submittedAt;
    auto const us = static_cast<uint64_t>(
        std::max<int64_t>(0, std::chrono::duration_cast<std::chrono::microseconds>(waited).count()));
    mCounters.queueLatencyTotalUs.fetch_add(us, std::memory_order_relaxed);
    mCounters.queueLatencyCount.fetch_add(1, std::memory_order_relaxed);
    // Only the actor records latencies, so a plain read-then-store max cannot race itself.
    if (us > mCounters.queueLatencyMaxUs.load(std::memory_order_relaxed))
    {
        mCounters.queueLatencyMaxUs.store(us, std::memory_order_relaxed);
    }
}

void RequestEngine::terminate(RequestRecord& record, TerminalStatus status, std::string errorMessage)
{
    // The single exit for every accepted request, whatever path retired it. Publishing anywhere
    // else would leak the pending-count reservation submit() took -- enough retired requests and
    // the engine refuses all new work while sitting idle.
    mPendingCount.fetch_sub(1, std::memory_order_relaxed);
    countOutcome(status);
    mChannel->forgetRecord(record.id);

    // A request that did not complete leaves its channel terminal too. A completed request's
    // channel was finished by the runtime that produced it; every other retirement -- rejected,
    // cancelled while queued, or a founder that failed before decoding -- has no runtime to do
    // that, and a streaming consumer polls the channel, not the outcome flag. Without this it
    // waits out its transport timeout on a channel nobody will ever touch (chaos-found: one
    // too-long streaming request froze its HTTP worker for the client's full five-minute
    // timeout). cancel() is idempotent, so the paths that already cancelled are unharmed.
    // The outcome goes first: get() reports "cancelled" for a cancelled, unfinished channel, so a
    // waiter that observed the cancel before the outcome would lose the error message behind it.
    // Only the outcome is published here. The channel is the runtime's to finish, and cancelling it
    // as a way of waking a waiter would report a completed request as cancelled -- a caller polls
    // the outcome flag instead, so nothing is left waiting.
    record.publishOutcome(status, std::move(errorMessage));
    if (status != TerminalStatus::kCompleted && record.channel != nullptr)
    {
        record.channel->cancel();
    }
}

void RequestEngine::drainCommandQueue()
{
    // Bounded rather than "until empty": producers can refill a slot as fast as the actor empties
    // one, and an unbounded drain would never reach admission or the stop check. One queue's worth
    // per boundary is enough to keep up without letting producers hold the actor hostage.
    for (int32_t budget = mChannel->capacity(); budget > 0; --budget)
    {
        auto command = mChannel->pop();
        if (command == nullptr)
        {
            break;
        }

        switch (command->type)
        {
        case CommandType::kSubmit:
        {
            mLive.emplace(command->record->id, command->record);
            mQueued.push_back(QueuedRequest{std::move(command->record), std::move(command->request)});
            mQueuedCount.store(static_cast<int32_t>(mQueued.size()), std::memory_order_release);
            break;
        }
        case CommandType::kShutdown:
        {
            // The mode and the stop flag were published by shutdown() before this command was
            // posted; the command only serves to wake the actor promptly. Cancelling the running
            // work is driven by the flag below, not by this command, which a full queue drops.
            mStopRequested.store(true, std::memory_order_release);
            break;
        }
        }
    }

    // Cancellation is a flag on the record, not a command (see RequestHandle::cancel), so the queue
    // is swept for it here: a request cancelled while queued retires now, releasing its backlog
    // reservation, instead of paying a founding or seated prefill first.
    retireCancelledQueued();

    // A cancelling shutdown also plants the flag on the running work, or the destructor would wait
    // out the whole current generation. Keyed on the flag rather than the kShutdown command: a
    // full queue drops the command, while this drain runs at every tick and boundary regardless.
    if (mStopRequested.load(std::memory_order_acquire)
        && mShutdownMode.load(std::memory_order_acquire) == ShutdownMode::kCancel)
    {
        cancelResidentWork();
    }
}

void RequestEngine::retireCancelledQueued()
{
    bool retired = false;
    for (auto it = mQueued.begin(); it != mQueued.end();)
    {
        if (it->record->channel->isCancelled())
        {
            terminate(*it->record, TerminalStatus::kCancelled);
            mLive.erase(it->record->id);
            it = mQueued.erase(it);
            retired = true;
            continue;
        }
        ++it;
    }
    if (retired)
    {
        mQueuedCount.store(static_cast<int32_t>(mQueued.size()), std::memory_order_release);
    }
}

void RequestEngine::cancelResidentWork()
{
    // The founder is in mResidents from the moment its batch opens, so this covers every resident.
    // Before that (between admitNextRequest and the founding) mResidentRecord is the only handle.
    if (mResidentRecord != nullptr && mResidentRecord->runtimeChannel != nullptr)
    {
        mResidentRecord->runtimeChannel->cancel();
    }
    mResidents.forEachRecord([](RequestRecord const& record) {
        if (record.runtimeChannel != nullptr)
        {
            record.runtimeChannel->cancel();
        }
    });
}

bool RequestEngine::admitNextRequest()
{
    if (mResidentRecord != nullptr)
    {
        return false;
    }
    // A head cancelled since the drain must not be founded: it would pay a full prefill and be
    // evicted at the first poll after it.
    retireCancelledQueued();
    if (mQueued.empty())
    {
        return false;
    }
    if (mStopRequested.load(std::memory_order_acquire)
        && mShutdownMode.load(std::memory_order_acquire) == ShutdownMode::kCancel)
    {
        return false;
    }

    QueuedRequest next = std::move(mQueued.front());
    mQueued.pop_front();
    mQueuedCount.store(static_cast<int32_t>(mQueued.size()), std::memory_order_release);

    mResidentRecord = std::move(next.record);
    mResidentRequest = std::move(next.request);
    mResidentCount.store(1, std::memory_order_release);
    recordQueueLatency(*mResidentRecord);
    return true;
}

bool RequestEngine::headMayJoin(LLMGenerationRequest const& head)
{
    if (!BatchCompatibility::compatible(*mResidentRequest, head))
    {
        if (mCounters.stallsIncompatible.fetch_add(1, std::memory_order_relaxed) == 0)
        {
            // Once per engine lifetime is enough for a breadcrumb; the counter carries the rate.
            LOG_DEBUG("Head request stalled on batch incompatibility: %s",
                BatchCompatibility::firstDifference(*mResidentRequest, head).c_str());
        }
        return false;
    }
    // Guided decoding is founder-only: the GuidedDecoder's matchers are numbered by slot, and
    // neither the admission seating swaps nor this actor track them, so a guided head cannot join
    // and a guided resident batch cannot be joined. Waiting rather than rejecting keeps FCFS
    // intact -- the head founds its own batch once the resident one drains.
    auto const anyGuided = [](LLMGenerationRequest const& request) {
        for (auto const& item : request.requests)
        {
            if (item.guidedDecoding.has_value())
            {
                return true;
            }
        }
        return false;
    };
    if (anyGuided(*mResidentRequest) || anyGuided(head))
    {
        mCounters.stallsGuided.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    // Founder-only options, for the same wait-not-reject reason: trajectory execution founds its
    // own batch, the legacy system-prompt capture runs only in the founding prefill path, and
    // speech output is assembled only by the drain-time path. submit() refuses all three today, so
    // this gate is defence in depth: it is what keeps two such requests apart the day one of them
    // is admitted again, since they compare compatible.
    auto const founderOnly = [](LLMGenerationRequest const& request) {
        if (request.saveSystemPromptKVCache || request.generateAudio)
        {
            return true;
        }
        for (auto const& item : request.requests)
        {
            if (item.pastTrajectory.has_value())
            {
                return true;
            }
        }
        return false;
    };
    if (founderOnly(*mResidentRequest) || founderOnly(head))
    {
        mCounters.stallsFounderOnly.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    return true;
}

void RequestEngine::publishOutcomes(std::vector<RequestCoordinator::TerminalOutcome> outcomes)
{
    for (auto& outcome : outcomes)
    {
        RequestRecord& record = *outcome.record;
        // Cancel wins: a cancel that landed after the runtime's top-of-step poll but before the
        // sequence finished naturally in that same step must not read back as completed.
        if (outcome.reason == FinishReason::kCancelled || (record.channel != nullptr && record.channel->isCancelled()))
        {
            terminate(record, TerminalStatus::kCancelled);
        }
        else if (outcome.reason == FinishReason::kError || !outcome.response.has_value())
        {
            terminate(record, TerminalStatus::kExecutionError, "the sequence failed during generation");
        }
        else
        {
            record.response = std::move(*outcome.response);
            terminate(record, TerminalStatus::kCompleted);
        }
        mLive.erase(record.id);
    }
}

void RequestEngine::runResidentSteppedRequest()
{
    if (mResidentRecord == nullptr)
    {
        return;
    }
    mAdmittedIndexBase = static_cast<int32_t>(mResidentRequest->requests.size());
    mNextAdmittedIndex = mAdmittedIndexBase;

    std::unique_ptr<SteppedExecution> stepped;
    std::string failure;
    try
    {
        stepped = mBeginStepped(*mResidentRequest, mStream);
    }
    catch (std::exception const& error)
    {
        failure = error.what();
    }
    catch (...)
    {
        failure = "unknown exception from beginStepped";
    }
    if (stepped == nullptr)
    {
        terminate(*mResidentRecord, TerminalStatus::kExecutionError,
            failure.empty() ? "the runtime refused the request" : failure);
        mLive.erase(mResidentRecord->id);
        mResidentRecord.reset();
        mResidentRequest.reset();
        mResidentCount.store(0, std::memory_order_release);
        return;
    }

    SteppedExecution& execution = *stepped;
    mResidents.clear();
    mResidents.add(execution.residents().front(), mResidentRecord, mResidentRequest->requests.front().stopStrings);

    // The founding prefill executed inside beginStepped; its post-pass is the first tick.
    std::optional<ResidentRef> awaitingPrefill = execution.residents().front();
    bool ok = true;
    try
    {
        while (!execution.residents().empty())
        {
            drainCommandQueue();
            StepResult result;
            if (awaitingPrefill.has_value())
            {
                result = execution.prefill({*awaitingPrefill});
                awaitingPrefill.reset();
            }
            else
            {
                // At most one admission per tick, FCFS; its seated prefill is the next tick. A head
                // cancelled since the drain is retired first rather than seated: its prefill would
                // stall every resident for a tick and be evicted at the next poll.
                retireCancelledQueued();
                if (!mQueued.empty() && static_cast<int32_t>(execution.residents().size()) < mConfig.maxBatchSize
                    && !mStopRequested.load(std::memory_order_acquire) && headMayJoin(*mQueued.front().request))
                {
                    QueuedRequest& head = mQueued.front();
                    int32_t const index = mNextAdmittedIndex;
                    bool admitted = false;
                    try
                    {
                        AdmissionResult const decision = execution.admit(*head.request, index, head.record->id);
                        if (decision.status == AdmissionResult::Status::kNoCapacity)
                        {
                            // Transient pressure: the head keeps its place and is offered again next tick.
                            mCounters.stallsNoCapacity.fetch_add(1, std::memory_order_relaxed);
                        }
                        else if (decision.status == AdmissionResult::Status::kRejected)
                        {
                            terminate(*head.record, TerminalStatus::kExecutionError, decision.reason);
                            mLive.erase(head.record->id);
                            mQueued.pop_front();
                            mQueuedCount.store(static_cast<int32_t>(mQueued.size()), std::memory_order_release);
                        }
                        else
                        {
                            ++mNextAdmittedIndex;
                            mCounters.admittedMidFlight.fetch_add(1, std::memory_order_relaxed);
                            recordQueueLatency(*head.record);
                            mResidents.add(decision.ref, head.record, head.request->requests.front().stopStrings);
                            mResidentCount.store(
                                static_cast<int32_t>(execution.residents().size()), std::memory_order_release);
                            awaitingPrefill = decision.ref;
                            admitted = true;
                            mQueued.pop_front();
                            mQueuedCount.store(static_cast<int32_t>(mQueued.size()), std::memory_order_release);
                        }
                    }
                    catch (std::exception const& error)
                    {
                        // Rejected outright at intent building; the batch was not modified.
                        terminate(*head.record, TerminalStatus::kExecutionError, error.what());
                        mLive.erase(head.record->id);
                        mQueued.pop_front();
                        mQueuedCount.store(static_cast<int32_t>(mQueued.size()), std::memory_order_release);
                    }
                    catch (...)
                    {
                        terminate(*head.record, TerminalStatus::kExecutionError, "unknown exception at admission");
                        mLive.erase(head.record->id);
                        mQueued.pop_front();
                        mQueuedCount.store(static_cast<int32_t>(mQueued.size()), std::memory_order_release);
                    }
                    if (admitted)
                    {
                        continue; // the admission's seated prefill is this batch's next tick
                    }
                }
                result = execution.decode({execution.residents()});
            }
            publishOutcomes(mResidents.commit(result, *stepped));
            mResidentCount.store(static_cast<int32_t>(execution.residents().size()), std::memory_order_release);
            if (!result.ok)
            {
                ok = false;
                break;
            }
        }
    }
    catch (std::exception const& error)
    {
        // A step threw -- a CUDA failure, or a seating the runtime could not undo -- so the batch
        // is gone. Every resident still owes its caller an outcome, and the actor must outlive the
        // batch: an unwind out of actorLoop would take the process down with std::terminate.
        LOG_ERROR("The stepped batch failed: %s", error.what());
        ok = false;
    }
    catch (...)
    {
        LOG_ERROR("The stepped batch failed with an unknown exception.");
        ok = false;
    }
    if (!ok || !mResidents.empty())
    {
        publishOutcomes(mResidents.failEverything());
    }

    LLMGenerationResponse drainResponse;
    try
    {
        if (!stepped->finish(drainResponse))
        {
            LOG_WARNING("Stepped teardown reported failure after all outcomes were published.");
        }
    }
    catch (std::exception const& error)
    {
        LOG_ERROR("Stepped teardown threw after all outcomes were published: %s", error.what());
    }
    catch (...)
    {
        LOG_ERROR("Stepped teardown threw an unknown exception after all outcomes were published.");
    }

    mResidentRecord.reset();
    mResidentRequest.reset();
    mResidentCount.store(0, std::memory_order_release);
}

void RequestEngine::actorLoop()
{
    for (;;)
    {
        drainCommandQueue();

        bool const stopping = mStopRequested.load(std::memory_order_acquire);
        if (stopping && mShutdownMode.load(std::memory_order_acquire) == ShutdownMode::kCancel)
        {
            break;
        }

        if (admitNextRequest())
        {
            runResidentSteppedRequest();
            continue;
        }

        if (stopping)
        {
            // kDrain: nothing resident and nothing left to admit. One more drain before leaving: a
            // submit accepted just before shutdown() closed the gate may still sit in the command
            // queue, and the acquire on the stop flag orders this drain after that close, so it
            // sees every accepted submit. Whatever it finds runs; only then is the queue final.
            drainCommandQueue();
            if (!mQueued.empty())
            {
                continue;
            }
            break;
        }

        // Nothing to do. Park rather than spin; submit, cancel and shutdown all wake the actor
        // explicitly, so the timeout only matters if some future path forgets to.
        std::unique_lock<std::mutex> lock(mWakeMutex);
        mWakeCv.wait_for(lock, mConfig.idleWait,
            [this] { return mChannel->hasPending() || mStopRequested.load(std::memory_order_acquire); });
    }

    // Nobody else will ever run these requests, so leave no caller parked on a stream that will
    // never go terminal.
    int32_t abandoned = 0;
    if (mResidentRecord != nullptr)
    {
        terminate(*mResidentRecord, TerminalStatus::kCancelled);
        mResidentRecord.reset();
        mResidentRequest.reset();
        mResidentCount.store(0, std::memory_order_release);
        ++abandoned;
    }
    for (auto& queued : mQueued)
    {
        terminate(*queued.record, TerminalStatus::kCancelled);
        ++abandoned;
    }
    mQueued.clear();
    mQueuedCount.store(0, std::memory_order_release);

    // A submit that raced the shutdown flag may still be sitting in the queue with a caller parked
    // on its stream.
    while (auto command = mChannel->pop())
    {
        if (command->type == CommandType::kSubmit && command->record != nullptr)
        {
            terminate(*command->record, TerminalStatus::kCancelled);
            ++abandoned;
        }
    }
    mLive.clear();
    if (abandoned > 0)
    {
        // Visible in the server log, not just as "was cancelled" at each caller's get().
        LOG_WARNING("RequestEngine shutdown cancelled %d request(s) that never ran.", abandoned);
    }
}

} // namespace scheduler
} // namespace rt
} // namespace trt_edgellm
