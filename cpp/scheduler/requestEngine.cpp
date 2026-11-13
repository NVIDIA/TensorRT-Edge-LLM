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
#include "profiling/nvtx_wrapper.h"
#include "runtime/llmInferenceRuntime.h"
#include "scheduler/batchCompatibility.h"
#include "scheduler/engineCommand.h"

#include <algorithm>
#include <exception>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace trt_edgellm
{
namespace rt
{
namespace scheduler
{

namespace
{

bool anyGuided(LLMGenerationRequest const& request) noexcept
{
    return std::any_of(request.requests.begin(), request.requests.end(),
        [](LLMGenerationRequest::Request const& item) { return item.guidedDecoding.has_value(); });
}

//! The channel's flag counts as well: a consumer may cancel the channel it holds directly.
bool cancelledByCaller(RequestRecord const& record) noexcept
{
    return record.cancelRequested() || (record.channel != nullptr && record.channel->isCancelled());
}

//! At startup the policy line is INFO; after a refused compact step it is a WARNING naming @p fallbackReason.
void logResolvedPolicy(
    ResolvedSchedulingPolicy const& policy, SessionCapabilities const& capabilities, std::string const& fallbackReason)
{
    std::string const modeChoice = policy.modeSource == ModeSource::kAuto
        ? std::string(toString(policy.modeSource)) + ": " + policy.reason
        : std::string(toString(policy.modeSource));
    std::string const budgets = policy.mode == IfbSchedulingMode::kMixed
        ? "maxNumTokens=" + std::to_string(policy.budgets.maxNumTokens)
            + ", maxChunkTokens=" + std::to_string(policy.budgets.maxChunkTokens)
        : std::string("maxNumTokens=n/a, maxChunkTokens=n/a");
    std::string compact = capabilities.compact ? std::string("available") : std::string("unavailable");
    // auto's reason already is the capability's; the line states it once.
    if (!capabilities.compact && !capabilities.reason.empty() && policy.reason != capabilities.reason)
    {
        compact += " (" + capabilities.reason + ")";
    }
    std::string const trigger = fallbackReason.empty() ? std::string() : " changed after " + fallbackReason;
    std::string const line = format::fmtstr(
        "RequestEngine scheduling%s: mode=%s (%s), execution=%s, schedulingPolicy=%s, preemption=%s, %s, "
        "priorityReservedSlots=%d, maxPauseSteps=%d; engine B=%d, S_in=%d, P=%d; compact execution %s.",
        trigger.c_str(), toString(policy.mode), modeChoice.c_str(), executionName(policy),
        toString(policy.schedulingPolicy), toString(policy.preemption), budgets.c_str(), policy.priorityReservedSlots,
        policy.maxPauseSteps, capabilities.maxBatchSize, capabilities.maxInputLength, capabilities.kvPageSize,
        compact.c_str());
    if (fallbackReason.empty())
    {
        LOG_INFO("%s", line.c_str());
    }
    else
    {
        LOG_WARNING("%s", line.c_str());
    }
    for (std::string const& warning : policy.warnings)
    {
        LOG_WARNING("RequestEngine configuration: %s", warning.c_str());
    }
}

//! Why compact execution is unavailable after a refused compact step, when no session is open to report it.
char const* const kCOMPACT_REFUSED
    = "a plugin refused compact execution at runtime (see the plugin's ERROR log, which identifies the "
      "layer by kind and shape)";

uint64_t elapsedUs(std::chrono::steady_clock::time_point from, std::chrono::steady_clock::time_point to) noexcept
{
    return static_cast<uint64_t>(
        std::max<int64_t>(0, std::chrono::duration_cast<std::chrono::microseconds>(to - from).count()));
}

void raiseMax(std::atomic<uint64_t>& maximum, uint64_t value) noexcept
{
    // Only the actor raises these maxima, so a plain read-then-store cannot race itself.
    if (value > maximum.load(std::memory_order_relaxed))
    {
        maximum.store(value, std::memory_order_relaxed);
    }
}

} // namespace

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
    // cancelled is released as soon as its token lands rather than when the actor, which may be
    // inside a step for a while yet, publishes the outcome. Every writer of either condition
    // notifies under outcomeMutex (publishOutcome, cancel), so a single wait cannot miss a wake
    // and needs no timeout.
    {
        std::unique_lock<std::mutex> lock(mRecord->outcomeMutex);
        mRecord->outcomeReady.wait(lock,
            [this] { return mRecord->outcomePublished.load(std::memory_order_acquire) || mRecord->cancelRequested(); });
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
        throw RequestFailedError(
            "Request " + std::to_string(mRecord->id) + " failed: " + mRecord->errorMessage, mRecord->failureKind);
    }
    throw std::runtime_error("Request " + std::to_string(mRecord->id) + " ended in an unknown state.");
}

bool RequestHandle::ready() const noexcept
{
    return mRecord != nullptr && mRecord->outcomePublished.load(std::memory_order_acquire);
}

std::optional<RequestStats> RequestHandle::stats() const
{
    if (!ready())
    {
        return std::nullopt;
    }
    return mRecord->stats;
}

void RequestHandle::cancel() noexcept
{
    if (mRecord == nullptr)
    {
        return;
    }
    // The token is written here, on the caller's thread, so cancellation takes effect immediately
    // even when the actor is deep inside a step and will not reach a boundary for seconds. The
    // runtime reads the token at the top of every step, so it stops a request that is already
    // executing; the actor reads it before every admission and sweeps the queue at every drain,
    // so it drops one that is still queued. Nothing rides the command queue, so overload cannot
    // lose a cancel.
    mRecord->cancel();
}

RequestEngine::RequestEngine(std::unique_ptr<LLMInferenceRuntime> runtime, cudaStream_t stream, EngineConfig config)
    : mRuntime(std::move(runtime))
    , mStream(stream)
    , mConfig(config)
    , mChannel(std::make_shared<EngineChannel>(config.commandQueueCapacity))
{
    check::check(mRuntime != nullptr, "RequestEngine requires a runtime.");

    // Rejected at construction rather than discovered later. The session runs on a single rank in
    // this release: with tensor parallelism every rank must execute the same step, and nothing yet
    // carries the step decisions to the other ranks, so admitting anyway would end in a hung
    // collective.
    check::check(mRuntime->supportsSteppedExecution(),
        "RequestEngine requires a single-rank runtime in this release; this one spans "
            + std::to_string(mRuntime->worldSize()) + " ranks. Tensor-parallel deployments stay on the blocking path.");

    // Every sequence enters the session by being seated one at a time -- the first no differently
    // from the tenth -- so a deployment that cannot seat sequences (a diffusion backbone, bounded
    // SWA storage, a context cache without live admission) cannot host a session at any batch
    // size. A configuration that can never admit is a construction error, not a per-request one.
    check::check(mRuntime->supportsSeatedAdmission(),
        "RequestEngine admits every sequence into a running session, which this deployment cannot seat (a "
        "diffusion backbone, bounded SWA KV storage, or a context cache without live admission); serve it on the "
        "blocking path.");
    // The engine's built batch dimension is the physical bound on admissions: above it, the runtime
    // would seat a row the engine cannot run, and under a context cache every admission is refused
    // forever. A batch that can never fill is a construction error too.
    check::check(config.maxBatchSize <= mRuntime->maxBatchSize(),
        "maxBatchSize " + std::to_string(config.maxBatchSize) + " exceeds the engine's built batch size "
            + std::to_string(mRuntime->maxBatchSize()) + ".");

    // A singleCohort engine never runs compact steps.
    SessionOptions const options{config.scheduling.ifbSchedulingMode != IfbSchedulingMode::kSingleCohort};
    mOpenSession
        = [this, options](cudaStream_t sessionStream) { return mRuntime->openSession(sessionStream, options); };
    start();
}

RequestEngine::RequestEngine(SessionFactory openSession, EngineConfig config)
    : mOpenSession(std::move(openSession))
    , mConfig(config)
    , mChannel(std::make_shared<EngineChannel>(config.commandQueueCapacity))
{
    check::check(mOpenSession != nullptr, "RequestEngine requires a session factory.");
    start();
}

void RequestEngine::start()
{
    check::check(mConfig.maxBatchSize >= 1, "RequestEngine requires a positive maxBatchSize.");
    check::check(mConfig.maxPendingRequests > 0, "RequestEngine pending-request limit must be positive.");
    check::check(
        mConfig.maxConsecutiveSessionFailures > 0, "RequestEngine consecutive session-failure limit must be positive.");
    // A non-positive idle wait would turn the actor's park into a busy loop.
    check::check(mConfig.idleWait.count() > 0, "RequestEngine idle wait must be positive.");

    // The session is opened on the actor's thread, which is the thread that drives it from then
    // on (CUDA device selection is per thread). The constructor waits for the verdict so that a
    // runtime which refuses to open is a construction error, not the first request's failure.
    mActor = std::thread(&RequestEngine::actorLoop, this);
    {
        std::unique_lock<std::mutex> lock(mSessionMutex);
        mSessionCv.wait(lock, [this] { return mSessionState != SessionState::kOpening; });
    }
    if (mSessionState == SessionState::kUnavailable)
    {
        // The actor returned without running; nothing was accepted, so there is nothing to sweep.
        mActor.join();
        if (!mPolicyError.empty())
        {
            throw std::invalid_argument("RequestEngine scheduling configuration rejected: " + mPolicyError);
        }
        throw std::runtime_error("RequestEngine could not open the runtime session: " + mSessionLostReason);
    }
}

RequestEngine::~RequestEngine()
{
    // Cancel rather than drain, so the destructor abandons the backlog instead of running it. It
    // still waits for a step already executing, because the actor cannot be interrupted inside
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
            "In-flight batching does not support per-request LoRA; submit rejects it outright.");
    }
    if (request.requests.size() != 1)
    {
        throw std::invalid_argument("RequestEngine::submit() takes exactly one sequence per request.");
    }
    // The same permanent-refusal semantics for what the session does not serve yet. Each of
    // these carries response state that only the runtime's blocking-path assembly fills, so
    // admitting it would return a silently incomplete result; the message names where the request
    // belongs until its session implementation lands (see the IFB support matrix in the docs).
    if (request.generateAudio)
    {
        throw std::invalid_argument(
            "In-flight batching does not serve speech output; run generateAudio requests through the blocking "
            "LLMInferenceRuntime path.");
    }
    if (request.requests.front().pastTrajectory.has_value())
    {
        throw std::invalid_argument(
            "In-flight batching does not serve trajectory execution; run pastTrajectory requests through the "
            "blocking LLMInferenceRuntime path.");
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
    // stream() hands out and terminate() finishes. A private second channel would leave
    // stream().consume() waiting on a channel nobody finishes. The runtime requires one channel
    // per sequence, so a request with none of its own gets the engine's.
    if (request.streamChannels.empty())
    {
        request.streamChannels.assign(request.requests.size(), record->channel);
    }
    else if (request.streamChannels.front() == nullptr)
    {
        // A null entry opts that sequence out of streaming; the runtime still needs a channel to
        // finish, so give it the engine's. The other entries keep their opt-out.
        request.streamChannels.front() = record->channel;
    }
    else
    {
        record->channel = request.streamChannels.front();
    }

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
    // The same token RequestHandle::cancel() sets, found through the live-record registry, so the
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

ResolvedSchedulingPolicy RequestEngine::resolvedPolicy() const
{
    std::lock_guard<std::mutex> lock(mStatusMutex);
    return mPolicy;
}

EngineStatus RequestEngine::status() const
{
    std::lock_guard<std::mutex> lock(mStatusMutex);
    return mStatus;
}

EngineMetrics RequestEngine::metrics() const noexcept
{
    EngineMetrics snapshot;
    snapshot.submitted = mCounters.submitted.load(std::memory_order_relaxed);
    snapshot.refused = mCounters.refused.load(std::memory_order_relaxed);
    snapshot.stallsIncompatible = mCounters.stallsIncompatible.load(std::memory_order_relaxed);
    snapshot.stallsGuided = mCounters.stallsGuided.load(std::memory_order_relaxed);
    snapshot.stallsExclusive = mCounters.stallsExclusive.load(std::memory_order_relaxed);
    snapshot.stallsNoCapacity = mCounters.stallsNoCapacity.load(std::memory_order_relaxed);
    snapshot.admittedMidFlight = mCounters.admittedMidFlight.load(std::memory_order_relaxed);
    snapshot.completed = mCounters.completed.load(std::memory_order_relaxed);
    snapshot.cancelled = mCounters.cancelled.load(std::memory_order_relaxed);
    snapshot.failed = mCounters.failed.load(std::memory_order_relaxed);
    snapshot.queueLatencyTotalUs = mCounters.queueLatencyTotalUs.load(std::memory_order_relaxed);
    snapshot.queueLatencyMaxUs = mCounters.queueLatencyMaxUs.load(std::memory_order_relaxed);
    snapshot.queueLatencyCount = mCounters.queueLatencyCount.load(std::memory_order_relaxed);
    snapshot.decodeSteps = mCounters.decodeSteps.load(std::memory_order_relaxed);
    snapshot.decodeGraphSteps = mCounters.decodeGraphSteps.load(std::memory_order_relaxed);
    snapshot.promptSteps = mCounters.promptSteps.load(std::memory_order_relaxed);
    snapshot.mixedSteps = mCounters.mixedSteps.load(std::memory_order_relaxed);
    snapshot.decodeSubsetSteps = mCounters.decodeSubsetSteps.load(std::memory_order_relaxed);
    snapshot.stepSequences = mCounters.stepSequences.load(std::memory_order_relaxed);
    snapshot.stepPromptChunks = mCounters.stepPromptChunks.load(std::memory_order_relaxed);
    snapshot.partialPromptChunks = mCounters.partialPromptChunks.load(std::memory_order_relaxed);
    snapshot.stepDecodeRows = mCounters.stepDecodeRows.load(std::memory_order_relaxed);
    snapshot.stepTokens = mCounters.stepTokens.load(std::memory_order_relaxed);
    snapshot.stepSampledRows = mCounters.stepSampledRows.load(std::memory_order_relaxed);
    snapshot.stepTimeTotalUs = mCounters.stepTimeTotalUs.load(std::memory_order_relaxed);
    snapshot.stepTimeMaxUs = mCounters.stepTimeMaxUs.load(std::memory_order_relaxed);
    snapshot.pausedResidentSteps = mCounters.pausedResidentSteps.load(std::memory_order_relaxed);
    snapshot.compactRefusals = mCounters.compactRefusals.load(std::memory_order_relaxed);
    return snapshot;
}

std::optional<ContextCacheMetrics> RequestEngine::contextCacheMetrics() const
{
    std::lock_guard<std::mutex> lock(mCacheMetricsMutex);
    return mCacheMetrics;
}

void RequestEngine::publishContextCacheMetrics()
{
    if (mSession == nullptr)
    {
        return;
    }
    std::optional<ContextCacheMetrics> snapshot = mSession->contextCacheMetrics();
    std::lock_guard<std::mutex> lock(mCacheMetricsMutex);
    mCacheMetrics = std::move(snapshot);
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

uint64_t RequestEngine::recordQueueLatency(RequestRecord const& record) noexcept
{
    uint64_t const us = elapsedUs(record.submittedAt, std::chrono::steady_clock::now());
    mCounters.queueLatencyTotalUs.fetch_add(us, std::memory_order_relaxed);
    mCounters.queueLatencyCount.fetch_add(1, std::memory_order_relaxed);
    raiseMax(mCounters.queueLatencyMaxUs, us);
    return us;
}

void RequestEngine::terminate(RequestRecord& record, TerminalStatus status, std::string errorMessage, FailureKind kind)
{
    // The single exit for every accepted request, whatever path retired it. Publishing anywhere
    // else would leak the pending-count reservation submit() took -- enough retired requests and
    // the engine refuses all new work while sitting idle.
    mPendingCount.fetch_sub(1, std::memory_order_relaxed);
    countOutcome(status);
    mChannel->forgetRecord(record.id);

    // The outcome goes first: a waiter released by the cancel token reads the outcome only once
    // it is published, so publishing after the channel would let it read a stale status.
    record.publishOutcome(status, std::move(errorMessage), kind);

    // A request that did not complete leaves its channel terminal too, with the reason. A
    // completed request's channel was finished by the runtime that produced it; every other
    // retirement -- rejected, cancelled while queued, failed before its first step -- has no
    // runtime to do that, and a streaming consumer polls the channel, not the outcome flag. Without
    // this it waits out its transport timeout on a channel nobody will ever touch (chaos-found:
    // one too-long streaming request froze its HTTP worker for the client's full five-minute
    // timeout). finish() is idempotent, so a channel the runtime already finished is unharmed.
    if (status != TerminalStatus::kCompleted && record.channel != nullptr)
    {
        record.channel->finish(status == TerminalStatus::kCancelled ? FinishReason::kCancelled : FinishReason::kError);
    }
}

void RequestEngine::publishOutcomes(std::vector<RequestCoordinator::TerminalOutcome> outcomes, FailureKind kind)
{
    for (auto& outcome : outcomes)
    {
        RequestRecord& record = *outcome.record;
        // Cancel wins: a cancel that landed after the runtime's top-of-step poll but before the
        // sequence finished naturally in that same step must not read back as completed.
        if (outcome.reason == FinishReason::kCancelled || cancelledByCaller(record))
        {
            terminate(record, TerminalStatus::kCancelled);
        }
        else if (outcome.reason == FinishReason::kError || !outcome.response.has_value())
        {
            terminate(record, TerminalStatus::kExecutionError,
                outcome.errorMessage.empty() ? std::string("the sequence failed during generation")
                                             : std::move(outcome.errorMessage),
                kind);
        }
        else
        {
            record.response = std::move(*outcome.response);
            terminate(record, TerminalStatus::kCompleted);
        }
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
            enqueue(QueuedRequest{std::move(command->record), std::move(command->request)});
            mQueuedCount.store(static_cast<int32_t>(mQueued.size()), std::memory_order_release);
            break;
        }
        case CommandType::kShutdown:
        {
            // The mode and the stop flag were published by shutdown() before this command was
            // posted; the command only serves to wake the actor promptly.
            mStopRequested.store(true, std::memory_order_release);
            break;
        }
        }
    }

    // Cancellation is a token on the record, not a command (see RequestHandle::cancel), so the
    // queue is swept for it here: a request cancelled while queued retires now, releasing its
    // backlog reservation, instead of paying a context step first.
    retireCancelledQueued();
}

void RequestEngine::retireCancelledQueued()
{
    bool retired = false;
    for (auto it = mQueued.begin(); it != mQueued.end();)
    {
        if (cancelledByCaller(*it->record))
        {
            terminate(*it->record, TerminalStatus::kCancelled);
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

void RequestEngine::enqueue(QueuedRequest queued)
{
    bool const byPriority = mPolicy.schedulingPolicy == SchedulingPolicy::kPriority;
    if (!byPriority && queued.request->priority != 0 && !mWarnedIgnoredPriority)
    {
        mWarnedIgnoredPriority = true;
        LOG_WARNING(
            "Request %llu carries priority %d, which schedulingPolicy=fcfs ignores; requests are admitted in "
            "arrival order. Use schedulingPolicy=priority to honour it. Not repeated for later requests.",
            static_cast<unsigned long long>(queued.record->id), queued.request->priority);
    }
    // Ids are handed out in submit order, so they are the arrival stamp: a drained request lands at the back of its
    // priority, and a requeued one returns to the place it was admitted from.
    auto const key = [byPriority](QueuedRequest const& entry) {
        return std::make_pair(byPriority ? entry.request->priority : 0, entry.record->id);
    };
    auto const position = std::upper_bound(mQueued.begin(), mQueued.end(), queued,
        [&key](QueuedRequest const& lhs, QueuedRequest const& rhs) { return key(lhs) < key(rhs); });
    mQueued.insert(position, std::move(queued));
}

bool RequestEngine::slotAvailableFor(LLMGenerationRequest const& head) const noexcept
{
    if (head.priority < 0)
    {
        return true;
    }
    int32_t const freeSlots = mConfig.maxBatchSize - mResidents.size();
    return freeSlots > mPolicy.priorityReservedSlots;
}

bool RequestEngine::openSession()
{
    try
    {
        mSession = mOpenSession(mStream);
        if (mSession == nullptr)
        {
            mSessionLostReason = "the runtime refused to open a session";
        }
    }
    catch (std::exception const& error)
    {
        mSession.reset();
        mSessionLostReason = error.what();
    }
    catch (...)
    {
        mSession.reset();
        mSessionLostReason = "unknown exception while opening the runtime session";
    }
    return mSession != nullptr;
}

bool RequestEngine::mayAdmit(QueuedRequest& queued)
{
    LLMGenerationRequest const& head = *queued.request;
    // A resident that must run alone keeps the session to itself until it leaves.
    if (mSession->exclusiveResident())
    {
        auto& counter = mResidents.anyRequest() != nullptr && anyGuided(*mResidents.anyRequest())
            ? mCounters.stallsGuided
            : mCounters.stallsExclusive;
        counter.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    // And a head that must run alone waits for the session to empty. Waiting rather than
    // rejecting keeps FCFS intact: the head is admitted into the empty session once the residents
    // have left, and nothing joins it while it runs.
    if (!mResidents.empty() && mSession->requiresExclusiveAdmission(head))
    {
        (anyGuided(head) ? mCounters.stallsGuided : mCounters.stallsExclusive).fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    // Batch-wide parameters (adapter, decoder, thinking mode, cache policies) must agree with the
    // residents'; any one resident stands for all, since each passed this same gate.
    if (LLMGenerationRequest const* resident = mResidents.anyRequest();
        resident != nullptr && !BatchCompatibility::compatible(*resident, head))
    {
        if (mCounters.stallsIncompatible.fetch_add(1, std::memory_order_relaxed) == 0)
        {
            // Once per engine lifetime is enough for a breadcrumb; the counter carries the rate.
            LOG_DEBUG("Head request stalled on batch incompatibility: %s",
                BatchCompatibility::firstDifference(*resident, head).c_str());
        }
        if (mPolicy.schedulingPolicy == SchedulingPolicy::kPriority && !queued.reportedCompatibilityStall
            && head.priority < mResidents.mostUrgentPriority())
        {
            queued.reportedCompatibilityStall = true;
            std::string const field = BatchCompatibility::firstDifference(*resident, head);
            LOG_WARNING(
                "Request %llu (priority %d) outranks every running request but waits until they all finish: its %s "
                "differs from theirs, and requests share a step only when they agree on it. Send urgent requests "
                "with the running requests' %s to start them at once.",
                static_cast<unsigned long long>(queued.record->id), head.priority, field.c_str(), field.c_str());
        }
        return false;
    }
    // A speculative decoder takes one set of sampling parameters per batch. A request on one waits for residents
    // that sample otherwise, and a request on the default decoder waits for speculative residents: requests that
    // sample alike select the same decoder.
    if (LLMGenerationRequest const* resident = mResidents.anyRequest(); resident != nullptr
        && (mSession->requiresUniformSampling(*resident) || mSession->requiresUniformSampling(head))
        && !BatchCompatibility::samplingAgrees(*resident, head))
    {
        if (mCounters.stallsIncompatible.fetch_add(1, std::memory_order_relaxed) == 0)
        {
            LOG_DEBUG("Head request stalled on batch incompatibility: sampling parameters under speculative decoding");
        }
        return false;
    }
    return true;
}

bool RequestEngine::tryAdmitHead()
{
    // Single-cohort execution admits one at a time: the pending one runs its context step before
    // the next is seated (the admission staging buffers are single-occupancy).
    if (!mPlanner.has_value() && mPendingContext.has_value())
    {
        return false;
    }
    // A head cancelled since the drain must not be admitted: it would pay a context step and be
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
    auto const popHead = [this] {
        mQueued.pop_front();
        mQueuedCount.store(static_cast<int32_t>(mQueued.size()), std::memory_order_release);
    };
    QueuedRequest& head = mQueued.front();
    if (mSession == nullptr)
    {
        // A failed step could not be followed by a fresh session (see failSession): nothing will
        // ever run this request, so it fails now, with the reason, instead of waiting forever.
        terminate(*head.record, TerminalStatus::kExecutionError,
            "the runtime session is unavailable: " + mSessionLostReason, FailureKind::kEngineStopped);
        popHead();
        return true;
    }
    if (mResidents.size() >= mConfig.maxBatchSize || !slotAvailableFor(*head.request) || !mayAdmit(head))
    {
        return false;
    }

    bool const midFlight = !mResidents.empty();
    AdmissionResult admitted;
    try
    {
        admitted = mSession->admit(*head.request, head.record->id, head.record->cancelToken);
    }
    catch (SessionLostError const& error)
    {
        // The reservation failed inside the runtime and the session's state is unknown. The head
        // fails with the reason -- offering it to a fresh session would only repeat the failure if
        // it was the cause -- and so does every resident, through failSession, which replaces the
        // session. Caught before std::exception, which it derives from.
        terminate(*head.record, TerminalStatus::kExecutionError, error.what());
        popHead();
        failSession(error.what());
        return true;
    }
    catch (std::exception const& error)
    {
        // Refused outright (validation, tokenization, a prompt the engine cannot hold); the
        // session was not modified.
        terminate(*head.record, TerminalStatus::kExecutionError, error.what());
        popHead();
        return true;
    }
    catch (...)
    {
        terminate(*head.record, TerminalStatus::kExecutionError, "unknown exception at admission");
        popHead();
        return true;
    }
    switch (admitted.status)
    {
    case AdmissionResult::Status::kNoCapacity:
        // Transient pressure: the head keeps its place and is offered again at the next boundary.
        mCounters.stallsNoCapacity.fetch_add(1, std::memory_order_relaxed);
        return false;
    case AdmissionResult::Status::kRejected:
        terminate(*head.record, TerminalStatus::kExecutionError, admitted.reason);
        popHead();
        return true;
    case AdmissionResult::Status::kAdmitted: break;
    }

    if (midFlight)
    {
        mCounters.admittedMidFlight.fetch_add(1, std::memory_order_relaxed);
    }
    RequestStats& stats = head.record->stats;
    stats.admitted = true;
    stats.queueTimeUs = head.record->requeued ? elapsedUs(head.record->submittedAt, std::chrono::steady_clock::now())
                                              : recordQueueLatency(*head.record);
    stats.promptTokens = admitted.promptLength;
    stats.reusedPromptTokens = admitted.prefillStart;
    std::shared_ptr<RequestRecord> const record = head.record;
    std::unique_ptr<LLMGenerationRequest> request = std::move(head.request);
    popHead();
    try
    {
        mResidents.add(admitted.ref, record, std::move(request), admitted);
    }
    catch (std::exception const& error)
    {
        // The session holds a sequence the table refused to seat: the two ledgers disagree, and
        // nothing built from either can be trusted. Not a std::terminate out of the actor -- the
        // admitted request and every resident get an outcome, and the session is replaced.
        terminate(*record, TerminalStatus::kExecutionError, error.what());
        failSession(error.what());
        return true;
    }
    mResidentCount.store(mResidents.size(), std::memory_order_release);
    if (!mPlanner.has_value())
    {
        mPendingContext = admitted.ref;
    }
    return true;
}

bool RequestEngine::runStep()
{
    if (mSession == nullptr || (!mPendingContext.has_value() && mResidents.empty()))
    {
        return false;
    }

    std::string failure;
    bool ok = true;
    bool compactRefused = false;
    auto const stepStart = std::chrono::steady_clock::now();
    mStepPaused.clear();
    try
    {
        // Building the plan is inside the guard too: the ledger refusing to build a step (a ref it
        // does not hold, a decode before the context step committed) is the same divergence a
        // disagreeing commit is, and is answered the same way -- by failing the session, not by an
        // unwind out of the actor.
        std::vector<ScheduledSequenceDescriptor>& plan = mStepPlan;
        plan.clear();
        {
            NVTX_SCOPED_RANGE(nvtxPlan, "IFB_PLAN", nvtx_colors::PURPLE);
            if (mPlanner.has_value())
            {
                planCompactStep(plan);
            }
            else if (mPendingContext.has_value())
            {
                plan.push_back(mResidents.contextStep(*mPendingContext));
            }
            else
            {
                mSession->copyResidents(mStepResidents);
                mResidents.decodeStep(mStepResidents, plan);
            }
        }
        StepResult result;
        {
            NVTX_SCOPED_RANGE(nvtxExecute, "IFB_EXECUTE", nvtx_colors::PURPLE);
            result = mSession->execute(plan);
        }
        // Taken before commit(), so a refusal still re-plans the engine if the ledger then disagrees too.
        compactRefused = !result.ok && result.compactRefused;
        if (result.ok)
        {
            recordStepStats(result, stepStart, std::chrono::steady_clock::now());
        }
        mPendingContext.reset();
        std::vector<RequestCoordinator::TerminalOutcome> outcomes;
        {
            NVTX_SCOPED_RANGE(nvtxCommit, "IFB_COMMIT", nvtx_colors::PURPLE);
            outcomes = mResidents.commit(result, *mSession);
        }
        ok = result.ok;
        if (ok)
        {
            mSessionFailuresInARow = 0;
            // Counted before the outcomes are published, so a caller released by this step sees it.
            countStep(result, elapsedUs(stepStart, std::chrono::steady_clock::now()));
        }
        else
        {
            failure = result.failure.empty() ? std::string("the runtime could not run the step") : result.failure;
        }
        // Before the outcomes, so a caller released by one already sees the step that finished it.
        publishContextCacheMetrics();
        publishOutcomes(std::move(outcomes));
        // The two ledgers must agree after every step. A sequence the session dropped without
        // reporting it would otherwise sit in the table forever, its caller parked; one the table
        // lost would decode without an owner. Either is a divergence that fails the session now,
        // with every resident told, rather than at whatever step happens to trip over it.
        if (ok)
        {
            mSession->copyResidents(mSessionResidents);
            ELLM_CHECK(mResidents.describes(mSessionResidents),
                "the runtime session and the engine's resident table disagree on which sequences are resident.");
        }
    }
    catch (std::exception const& error)
    {
        // A step threw -- a CUDA failure, or a plan the session refused -- so the session's state is
        // undefined. Every resident still owes its caller an outcome, and the actor must outlive the
        // failure: an unwind out of actorLoop would take the process down with std::terminate.
        ok = false;
        failure = error.what();
    }
    catch (...)
    {
        ok = false;
        failure = "unknown exception inside the step";
    }
    mResidentCount.store(mResidents.size(), std::memory_order_release);
    if (!ok)
    {
        failSession(failure, compactRefused);
    }
    return true;
}

void RequestEngine::planCompactStep(std::vector<ScheduledSequenceDescriptor>& plan)
{
    mSession->copyResidents(mStepResidents);
    mStepStates.clear();
    for (ResidentRef const ref : mStepResidents)
    {
        RequestCoordinator::Progress const progress = mResidents.progress(ref);
        RequestRecord const* const record = mResidents.findRecord(ref);
        // Request ids are handed out in submit order, so they are the arrival stamp.
        mStepStates.push_back(
            ResidentState{mResidents.request(ref).priority, progress.identity.requestId, progress.promptLength,
                progress.committed, progress.contextCommitted, record != nullptr ? record->consecutivePausedSteps : 0});
    }
    StepPlan const& step = mPlanner->plan(mStepStates);
    ELLM_CHECK(!step.sequences.empty(), "the step planner scheduled nothing over a non-empty session.");
    notePausedResidents(step);
    for (PlannedSequence const& sequence : step.sequences)
    {
        ResidentRef const ref = mStepResidents[static_cast<size_t>(sequence.resident)];
        if (sequence.work == SequenceWork::kContext)
        {
            plan.push_back(mResidents.contextChunk(ref, sequence.queryLength));
        }
        else
        {
            RequestCoordinator::Progress const progress = mResidents.progress(ref);
            plan.push_back(ScheduledSequenceDescriptor{progress.identity, SequenceWork::kDecode, /*queryLength=*/1,
                progress.committed, /*selectLastTokenLogits=*/true});
        }
    }
    // LOG_DEBUG logs at kVERBOSE; the guard skips formatting the line at any other level.
    if (logger::EdgeLLMLogger::instance().getLevel() == logger::Severity::kVERBOSE)
    {
        LOG_DEBUG("IFB step: residents=%zu Np=%d Nd=%d T_p=%d T=%d paused=%zu", mStepResidents.size(), step.numPrefill,
            step.numDecode, step.prefillTokens, step.totalTokens, step.paused.size());
    }
}

void RequestEngine::notePausedResidents(StepPlan const& step)
{
    // step.paused is ascending, so one pass over the residents pairs each with its pause.
    auto nextPaused = step.paused.begin();
    for (size_t index = 0; index < mStepResidents.size(); ++index)
    {
        bool const paused = nextPaused != step.paused.end() && static_cast<size_t>(*nextPaused) == index;
        if (paused)
        {
            ++nextPaused;
            mStepPaused.push_back(mStepResidents[index]);
        }
        if (RequestRecord* const record = mResidents.findRecord(mStepResidents[index]))
        {
            int32_t& pausedSteps = record->consecutivePausedSteps;
            if (!paused)
            {
                pausedSteps = 0;
            }
            else if (pausedSteps < std::numeric_limits<int32_t>::max())
            {
                ++pausedSteps;
            }
        }
    }
}

void RequestEngine::recordStepStats(StepResult const& result, std::chrono::steady_clock::time_point stepStart,
    std::chrono::steady_clock::time_point executed) noexcept
{
    // A paused resident the step retired (cancelled at its top) was not withheld by it.
    mStepPaused.erase(std::remove_if(mStepPaused.begin(), mStepPaused.end(),
                          [&result](ResidentRef ref) {
                              return std::any_of(result.finished.begin(), result.finished.end(),
                                  [ref](auto const& finished) { return finished.first == ref; });
                          }),
        mStepPaused.end());
    for (ScheduledSequenceDescriptor const& sequence : result.execution)
    {
        if (sequence.work != SequenceWork::kContext)
        {
            continue;
        }
        RequestRecord* const record = mResidents.findRecord(sequence.identity.resident);
        if (record == nullptr)
        {
            continue; // commit() refuses the step and fails the session
        }
        ++record->stats.promptChunks;
        if (sequence.selectLastTokenLogits)
        {
            record->stats.timeToFirstTokenUs = elapsedUs(record->submittedAt, executed);
        }
    }
    uint64_t const stepUs = elapsedUs(stepStart, executed);
    for (ResidentRef const ref : mStepPaused)
    {
        if (RequestRecord* const record = mResidents.findRecord(ref))
        {
            ++record->stats.pausedSteps;
            record->stats.pausedTimeUs += stepUs;
        }
    }
}

void RequestEngine::countStep(StepResult const& result, uint64_t stepTimeUs) noexcept
{
    mCounters.pausedResidentSteps.fetch_add(mStepPaused.size(), std::memory_order_relaxed);
    if (result.execution.empty())
    {
        return;
    }
    uint64_t chunks = 0;
    uint64_t partial = 0;
    uint64_t decodes = 0;
    uint64_t tokens = 0;
    uint64_t sampled = 0;
    for (ScheduledSequenceDescriptor const& executed : result.execution)
    {
        if (executed.work == SequenceWork::kContext)
        {
            ++chunks;
            partial += executed.selectLastTokenLogits ? 0 : 1;
        }
        else
        {
            ++decodes;
        }
        tokens += static_cast<uint64_t>(executed.queryLength);
        sampled += executed.selectLastTokenLogits ? 1 : 0;
    }
    if (chunks > 0)
    {
        mCounters.promptSteps.fetch_add(1, std::memory_order_relaxed);
        if (decodes > 0)
        {
            mCounters.mixedSteps.fetch_add(1, std::memory_order_relaxed);
        }
    }
    else if (!result.compact)
    {
        mCounters.decodeSteps.fetch_add(1, std::memory_order_relaxed);
        if (result.graphLaunched)
        {
            mCounters.decodeGraphSteps.fetch_add(1, std::memory_order_relaxed);
        }
    }
    else
    {
        mCounters.decodeSubsetSteps.fetch_add(1, std::memory_order_relaxed);
    }
    mCounters.stepSequences.fetch_add(chunks + decodes, std::memory_order_relaxed);
    mCounters.stepPromptChunks.fetch_add(chunks, std::memory_order_relaxed);
    mCounters.partialPromptChunks.fetch_add(partial, std::memory_order_relaxed);
    mCounters.stepDecodeRows.fetch_add(decodes, std::memory_order_relaxed);
    mCounters.stepTokens.fetch_add(tokens, std::memory_order_relaxed);
    mCounters.stepSampledRows.fetch_add(sampled, std::memory_order_relaxed);
    mCounters.stepTimeTotalUs.fetch_add(stepTimeUs, std::memory_order_relaxed);
    raiseMax(mCounters.stepTimeMaxUs, stepTimeUs);
}

void RequestEngine::requeuePromptIncomplete(std::string const& reason)
{
    // The next session recomputes a requeued request from its first uncached prompt token; nothing of this
    // session's KV or recurrent state is reused, and its caller has seen no token.
    int32_t requeued = 0;
    for (RequestCoordinator::Released& released : mResidents.releasePromptIncomplete())
    {
        RequestRecord& record = *released.record;
        // The session applied a cancellation at the top of the refused step by retiring the row, which leaves
        // nothing to release.
        if (cancelledByCaller(record))
        {
            terminate(record, TerminalStatus::kCancelled);
            continue;
        }
        try
        {
            mSession->releaseForReadmission(released.ref);
        }
        catch (std::exception const& error)
        {
            LOG_ERROR(
                "Request %llu cannot be queued again: %s", static_cast<unsigned long long>(record.id), error.what());
            terminate(record, TerminalStatus::kExecutionError, "the runtime session failed: " + reason,
                FailureKind::kCompactStepRefused);
            continue;
        }
        // One cancelled from here on is retired from the queue at the next drain, like any queued request.
        record.stats = RequestStats{};
        record.consecutivePausedSteps = 0;
        record.requeued = true;
        enqueue(QueuedRequest{std::move(released.record), std::move(released.request)});
        ++requeued;
    }
    mQueuedCount.store(static_cast<int32_t>(mQueued.size()), std::memory_order_release);
    if (requeued > 0)
    {
        LOG_WARNING(
            "%d request(s) that had sampled no token are queued again at their place and will rerun from the "
            "start of their prompts.",
            requeued);
    }
}

void RequestEngine::failSession(std::string const& reason, bool compactRefused)
{
    if (compactRefused)
    {
        failRefusedCompactStep(reason);
        return;
    }
    LOG_ERROR("The runtime session failed: %s", reason.c_str());
    publishOutcomes(closeSession(reason));
    if (++mSessionFailuresInARow >= mConfig.maxConsecutiveSessionFailures)
    {
        // A fresh session would open -- opening hardly touches the device -- and fail the same
        // way at its first step. Past the bound the engine stops reopening and answers every later
        // request with the reason, as it does when no session can be opened at all.
        mSessionLostReason = reason + " (" + std::to_string(mSessionFailuresInARow)
            + " session failures in a row without a completed step; the session is not reopened)";
        LOG_ERROR(
            "The runtime session failed %d times in a row without completing a step; it is not reopened and every "
            "request will fail until the engine is recreated.",
            mSessionFailuresInARow);
        publishUnavailable();
        return;
    }
    reopenSession();
}

void RequestEngine::failRefusedCompactStep(std::string const& reason)
{
    mCounters.compactRefusals.fetch_add(1, std::memory_order_relaxed);
    if (mConfig.scheduling.ifbSchedulingMode == IfbSchedulingMode::kMixed)
    {
        // An explicit mixed asks for compact steps, which this runtime no longer executes; serving on with
        // single-cohort steps would ignore the configuration silently. The runtime's "can be retried" would mislead.
        std::string const residentReason = std::string(kCOMPACT_REFUSED)
            + "; the engine stops serving because ifbSchedulingMode=mixed requires compact execution";
        LOG_ERROR("The runtime session failed: %s", residentReason.c_str());
        publishOutcomes(closeSession(residentReason), FailureKind::kEngineStopped);
        mSessionLostReason = std::string("ifbSchedulingMode=mixed requires compact execution, and ") + kCOMPACT_REFUSED
            + "; the session is not reopened. Configure ifbSchedulingMode=auto to fall back to single-cohort steps "
              "instead";
        LOG_ERROR("RequestEngine stopped: %s. Every request will fail until the engine is recreated.",
            mSessionLostReason.c_str());
        publishUnavailable(kCOMPACT_REFUSED);
        return;
    }
    LOG_ERROR("The runtime session failed: %s", reason.c_str());
    requeuePromptIncomplete(reason);
    std::vector<RequestCoordinator::TerminalOutcome> failed = closeSession(reason);
    // Not counted toward maxConsecutiveSessionFailures: the reopened session has no compact execution, so the
    // refusal cannot repeat.
    bool serving = reopenSession(kCOMPACT_REFUSED);
    if (serving)
    {
        try
        {
            // The planner must follow the session: a single-cohort session refuses a compact plan, so a stale
            // planner would fail every later session at its first step.
            adoptPolicy(mSession->capabilities(), "a plugin refused a compact step");
        }
        catch (std::exception const& error)
        {
            serving = false;
            mSession.reset();
            mSessionLostReason
                = std::string("the scheduling policy could not be resolved again after a refused compact step: ")
                + error.what();
            LOG_ERROR("%s; every request will fail until the engine is recreated.", mSessionLostReason.c_str());
            publishUnavailable(kCOMPACT_REFUSED);
        }
    }
    if (!serving)
    {
        // The residents' failure is retryable only while the engine serves on.
        for (RequestCoordinator::TerminalOutcome& outcome : failed)
        {
            outcome.errorMessage = std::string("the runtime session failed: ") + kCOMPACT_REFUSED
                + "; the engine stopped serving: " + mSessionLostReason;
        }
    }
    publishOutcomes(std::move(failed), serving ? FailureKind::kCompactStepRefused : FailureKind::kEngineStopped);
}

std::vector<RequestCoordinator::TerminalOutcome> RequestEngine::closeSession(std::string const& reason)
{
    std::vector<RequestCoordinator::TerminalOutcome> outcomes
        = mResidents.retireEverything(FinishReason::kError, "the runtime session failed: " + reason);
    mPendingContext.reset();
    mResidentCount.store(0, std::memory_order_release);
    // Close before reopening: the runtime's one-request latch is held by the failed session, and
    // its destructor releases what the residents held.
    mSession.reset();
    return outcomes;
}

bool RequestEngine::reopenSession(std::string const& compactReason)
{
    if (openSession())
    {
        return true;
    }
    // Not a shutdown: submit() keeps accepting so callers get a fast, explicit failure that
    // names the reason, rather than a busy signal or a caller parked forever.
    LOG_ERROR("The runtime session could not be reopened (%s); every request will fail until the engine is recreated.",
        mSessionLostReason.c_str());
    publishUnavailable(compactReason);
    return false;
}

void RequestEngine::adoptPolicy(SessionCapabilities const& capabilities, std::string const& fallbackReason)
{
    ResolvedSchedulingPolicy policy = resolveSchedulingPolicy(mConfig.scheduling, mConfig.maxBatchSize, capabilities);
    mPlanner.reset();
    if (policy.compactExecution)
    {
        mPlanner.emplace(
            policy, StepPlannerLimits{capabilities.maxBatchSize, capabilities.maxInputLength, capabilities.kvPageSize});
    }
    {
        std::lock_guard<std::mutex> lock(mStatusMutex);
        mPolicy = std::move(policy);
        mStatus.compactUnavailableReason = capabilities.compact ? std::string() : capabilities.reason;
    }
    logResolvedPolicy(mPolicy, capabilities, fallbackReason);
}

void RequestEngine::publishUnavailable(std::string const& compactReason)
{
    std::lock_guard<std::mutex> lock(mStatusMutex);
    mStatus.serving = false;
    mStatus.unavailableReason = mSessionLostReason;
    if (!compactReason.empty())
    {
        mStatus.compactUnavailableReason = compactReason;
    }
}

void RequestEngine::actorLoop()
{
    bool opened = openSession();
    if (opened)
    {
        try
        {
            SessionCapabilities const capabilities = mSession->capabilities();
            adoptPolicy(capabilities);
            size_t const batch = static_cast<size_t>(std::max(capabilities.maxBatchSize, mConfig.maxBatchSize));
            mStepResidents.reserve(batch);
            mSessionResidents.reserve(batch);
            mStepStates.reserve(batch);
            mStepPaused.reserve(batch);
            mStepPlan.reserve(batch);
        }
        catch (std::invalid_argument const& error)
        {
            mPolicyError = error.what();
            mSession.reset();
            opened = false;
        }
        catch (std::exception const& error)
        {
            mSessionLostReason = std::string("setting up the opened session failed: ") + error.what();
            mSession.reset();
            opened = false;
        }
    }
    // Before the constructor returns, so a caller can confirm the cache is empty before its first submit.
    publishContextCacheMetrics();
    {
        std::lock_guard<std::mutex> lock(mSessionMutex);
        mSessionState = opened ? SessionState::kOpen : SessionState::kUnavailable;
    }
    mSessionCv.notify_all();
    if (!opened)
    {
        return; // the constructor throws; nothing was accepted
    }

    for (;;)
    {
        drainCommandQueue();

        bool const stopping = mStopRequested.load(std::memory_order_acquire);
        if (stopping && mShutdownMode.load(std::memory_order_acquire) == ShutdownMode::kCancel)
        {
            break;
        }

        // One boundary: fold in the queue (above), admit at most one head, run one step. An
        // admission is not followed by its context step in the same boundary; the next one runs it,
        // so the queue is drained again in between and a cancel that landed meanwhile is seen.
        // Compact execution admits every head that can join and then steps over everyone; the
        // session applies a cancel that landed meanwhile at the top of that step.
        bool progressed = false;
        if (mPlanner.has_value())
        {
            while (tryAdmitHead())
            {
                progressed = true;
            }
            progressed = runStep() || progressed;
        }
        else
        {
            progressed = tryAdmitHead() || runStep();
        }
        if (progressed)
        {
            continue;
        }

        if (stopping)
        {
            // kDrain: nothing resident and nothing left to admit. One more drain before leaving: a
            // submit accepted just before shutdown() closed the gate may still sit in the command
            // queue, and the acquire on the stop flag orders this drain after that close, so it
            // sees every accepted submit. Whatever it finds runs; only then is the queue final.
            drainCommandQueue();
            if (!mQueued.empty() || !mResidents.empty())
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
    // never go terminal. Residents first: their outcome is cancellation, and closing the session
    // below releases what they held.
    int32_t abandoned = 0;
    std::vector<RequestCoordinator::TerminalOutcome> residents = mResidents.retireEverything(FinishReason::kCancelled);
    abandoned += static_cast<int32_t>(residents.size());
    publishOutcomes(std::move(residents));
    mPendingContext.reset();
    mResidentCount.store(0, std::memory_order_release);
    mSession.reset();

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
    if (abandoned > 0)
    {
        // Visible in the server log, not just as "was cancelled" at each caller's get().
        LOG_WARNING("RequestEngine shutdown cancelled %d request(s).", abandoned);
    }
}

} // namespace scheduler
} // namespace rt
} // namespace trt_edgellm
