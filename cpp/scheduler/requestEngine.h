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
#include "runtime/runtimeSession.h"
#include "scheduler/engineChannel.h"
#include "scheduler/requestCoordinator.h"
#include "scheduler/requestIdentity.h"
#include "scheduler/requestRecord.h"
#include "scheduler/schedulingPolicy.h"
#include "scheduler/stepPlanner.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>

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

//! Thrown by RequestHandle::get() for a request that ended in TerminalStatus::kExecutionError.
class RequestFailedError : public std::runtime_error
{
public:
    RequestFailedError(std::string const& message, FailureKind kind)
        : std::runtime_error(message)
        , mKind(kind)
    {
    }

    FailureKind kind() const noexcept
    {
        return mKind;
    }

private:
    FailureKind mKind;
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
    //! Arriving requests join the session at step boundaries (in-flight batching), up to this many;
    //! beyond it a request waits in the queue for a resident to leave. This is also the KV capacity
    //! bound: page tables partition their pool per slot, so a request beyond this count has no
    //! pages to run on -- it waits, no eviction of others required. May not exceed the batch size
    //! the engine was built with.
    int32_t maxBatchSize{1};

    //! Session failures in a row -- each retiring every resident, closing the session and opening
    //! a fresh one -- with no step completed between them, after which the engine stops reopening
    //! and every later request fails at admission with the last reason. Opening a session hardly
    //! touches the device, so a reopen succeeds even when the failure's cause is sticky (a device
    //! error the runtime cannot clear); without this bound every later request would pay one
    //! failed step and a reopen, forever. A refused compact step (StepResult::compactRefused) is not
    //! counted: its cause cannot repeat, because every later session reports no compact execution.
    int32_t maxConsecutiveSessionFailures{2};

    //! How long an idle actor sleeps before waking to look around again. It does not affect a busy
    //! engine, which never reaches the park at all.
    //!
    //! Every operation that gives the actor something to do also wakes it, so this timeout is a
    //! backstop rather than the mechanism: it bounds the damage if a future caller changes actor
    //! state without waking it, turning a permanent hang into a delay of this length.
    std::chrono::milliseconds idleWait{50};

    //! In-flight batching scheduling, resolved against the deployment when the session opens; see
    //! resolveSchedulingPolicy() for the defaults and constraints and RequestEngine::resolvedPolicy() for the outcome.
    SchedulingOptions scheduling;
};

//! A point-in-time copy of the engine's counters. Each field is read with relaxed ordering, so
//! the snapshot is per-field accurate but not a single atomic cut -- fine for observability,
//! not for control flow.
struct EngineMetrics
{
    //! Requests submit() accepted / refused (queue full, backlog full, or shutting down).
    uint64_t submitted{};
    uint64_t refused{};
    //! Step boundaries at which the queue head could not be admitted, by reason. High stall counts
    //! quantify the cost of FCFS head blocking under a mixed workload.
    //! - incompatible: its batch-wide parameters differ from the residents' (BatchCompatibility).
    //! - guided: it is grammar-constrained, or a constrained resident holds the session.
    //! - exclusive: it needs the session to itself for another reason (media under a visual-token
    //!   pruner), or such a resident holds the session.
    //! - noCapacity: the runtime refused the admission as transient pressure (no free pages).
    uint64_t stallsIncompatible{};
    uint64_t stallsGuided{};
    uint64_t stallsExclusive{};
    uint64_t stallsNoCapacity{};
    //! Requests admitted while other residents were decoding (excludes those that found the
    //! session empty).
    uint64_t admittedMidFlight{};
    //! Running totals of terminal outcomes across all requests, one per TerminalStatus value:
    //! countOutcome() bumps exactly one of them each time a request retires, so they always sum
    //! to the outcomes callers saw. A single request's own state is the TerminalStatus on its
    //! record, not anything here.
    uint64_t completed{};
    uint64_t cancelled{};
    uint64_t failed{};
    //! Time from submit() to the actor admitting the request, once per request: a request queued again after a
    //! refused compact step is not counted at its second admission.
    uint64_t queueLatencyTotalUs{};
    uint64_t queueLatencyMaxUs{};
    uint64_t queueLatencyCount{};
    //! Counted over steps that completed and ran at least one sequence. Steps by kind:
    //! - decodeSteps: every resident left after the step's cancellations decoded one token (phase 3), of
    //!   which decodeGraphSteps ran the base engine from a captured CUDA graph.
    //! - promptSteps: at least one prompt chunk ran (a single-cohort context step, or a compact step with
    //!   Np >= 1), of which mixedSteps also ran decode rows.
    //! - decodeSubsetSteps: only decode rows, with some resident left out (compact, profile 1).
    uint64_t decodeSteps{};
    uint64_t decodeGraphSteps{};
    uint64_t promptSteps{};
    uint64_t mixedSteps{};
    uint64_t decodeSubsetSteps{};
    //! Running sums over the same steps; divide by the step count for averages.
    //! - stepSequences: N, sequences the step ran.
    //! - stepPromptChunks: Np, prompt chunks; partialPromptChunks of them did not end their prompt.
    //! - stepDecodeRows: Nd, decode rows.
    //! - stepTokens: T, query tokens.
    //! - stepSampledRows: L, sequences whose last token was sampled.
    uint64_t stepSequences{};
    uint64_t stepPromptChunks{};
    uint64_t partialPromptChunks{};
    uint64_t stepDecodeRows{};
    uint64_t stepTokens{};
    uint64_t stepSampledRows{};
    //! Actor host time of those steps, from planning through the commit of their results.
    uint64_t stepTimeTotalUs{};
    uint64_t stepTimeMaxUs{};
    //! Residents withheld by preemption=pause, summed over completed steps.
    uint64_t pausedResidentSteps{};
    //! Compact steps a plugin refused (StepResult::compactRefused); see RequestEngine::resolvedPolicy() for what
    //! became of the residents and RequestEngine::status() for what the engine did next.
    uint64_t compactRefusals{};
};

//! Whether the engine still serves requests, for health reporting. Read with RequestEngine::status().
struct EngineStatus
{
    //! False once the engine has no session and opens none again: the failure limit was reached, a session
    //! could not be reopened, or a plugin refused a compact step under an explicit mixed mode. Every later request
    //! then fails at admission with unavailableReason until the engine is recreated.
    bool serving{true};
    //! Why the engine stopped serving; empty while it serves.
    std::string unavailableReason;
    //! Why the current session cannot execute compact steps, as its capabilities state it, or the refusal that
    //! stopped the engine; empty when compact execution is available.
    std::string compactUnavailableReason;
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
    //! @throws RequestFailedError if the request ended in kExecutionError
    //! @throws std::runtime_error if the request was cancelled
    LLMGenerationResponse get();

    //! @brief True once the request has an outcome, so get() would return without blocking.
    //!
    //! Distinct from the channel being finished, which reports only that the token stream ended.
    bool ready() const noexcept;

    //! @brief Ask the engine to stop this request. Non-blocking and idempotent; a request that has
    //!        already retired simply does not match anything the actor still holds.
    //!
    //! Releases this caller immediately, but does not interrupt a step the runtime is already
    //! inside: the request stops at the next step boundary and is then reported as cancelled.
    void cancel() noexcept;

    bool valid() const noexcept
    {
        return mRecord != nullptr;
    }

    //! @brief The request's scheduling statistics, once ready(); empty before that.
    std::optional<RequestStats> stats() const;

private:
    //! Shared rather than a pointer back to the engine: a handle may outlive it, and cancelling
    //! through a destroyed engine would be a use-after-free.
    std::shared_ptr<EngineChannel> mChannel;
    std::shared_ptr<RequestRecord> mRecord;
};

//! Owns the runtime and the one thread allowed to touch it.
//!
//! submit() and cancel() are callable from any thread: each posts a command to the queue and
//! returns without waiting for the actor. shutdown() is the exception -- it joins, so it blocks and
//! must not be called from the actor itself. The actor starts in the constructor and is joined by
//! shutdown(), which the destructor implies, so there is no start() to forget.
//!
//! The actor opens one RuntimeSession for its life and drives it one step at a time: at every
//! step boundary it folds in the command queue, admits at most one queued request (queue head in
//! the configured order, FCFS by default, gated by the residents' compatibility, the session's
//! exclusivity rules and the reserved slots), then runs the pending admission's context step or one
//! decode step over every resident. Under compact execution
//! (ResolvedSchedulingPolicy::compactExecution) it instead admits every queue head that can join,
//! in the same order and under the same gates, and runs the step StepPlanner composes from the
//! residents' ledger: prompt chunks of several residents beside decode rows.
//! Requests leave through the step that evicts them; an empty session idles until the next submit.
//! Feeding the GPU has to be serial anyway, so that is what the actor serialises: callers never
//! block on each other, and no caller ever holds a pointer to the runtime, so a data race on it
//! cannot be written rather than being prevented by a lock the caller has to remember to take.
class RequestEngine
{
public:
    //! How the actor opens its runtime session. In production this is bound to
    //! LLMInferenceRuntime::openSession. It is a seam rather than a direct call so the actor's
    //! admission, scheduling, cancellation and shutdown paths can be tested without a GPU:
    //! LLMInferenceRuntime is a concrete class and needs a real CUDA stream to assemble, which would
    //! put every one of those tests behind a device. A null return means the runtime refused; a
    //! throw carries the reason. Called on the actor's thread, at start and after a failed step.
    using SessionFactory = std::function<std::unique_ptr<RuntimeSession>(cudaStream_t)>;

    //! @param runtime Taken over outright. Nothing else may hold or call it afterwards.
    //! @param stream The CUDA stream the actor drives the runtime on.
    //! @throws std::runtime_error if @p runtime spans more than one rank (the session runs on a
    //!         single rank in this release), cannot seat sequences one at a time (a diffusion
    //!         backbone, bounded SWA storage, a context cache without live admission), or refuses to
    //!         open the session.
    //! @throws std::invalid_argument if the scheduling policy cannot be served by this deployment
    //!         (see resolveSchedulingPolicy).
    RequestEngine(std::unique_ptr<LLMInferenceRuntime> runtime, cudaStream_t stream, EngineConfig config = {});

    //! @brief Test seam: an engine with no runtime, over whatever session @p openSession hands back.
    //! @throws std::runtime_error if the factory refuses the first session.
    //! @throws std::invalid_argument if the scheduling policy cannot be served by the session.
    RequestEngine(SessionFactory openSession, EngineConfig config);

    ~RequestEngine();

    RequestEngine(RequestEngine const&) = delete;
    RequestEngine& operator=(RequestEngine const&) = delete;
    RequestEngine(RequestEngine&&) = delete;
    RequestEngine& operator=(RequestEngine&&) = delete;

    //! @brief Any thread: hand a request to the actor. Does not wait for it to start.
    //! @throws SubmitError if the engine is shutting down or the command queue is full
    //! @throws std::invalid_argument for a request the engine never serves (see the IFB support
    //!         matrix): per-request LoRA, speech output, the legacy system-prompt capture,
    //!         trajectory execution, more than one sequence per request.
    RequestHandle submit(LLMGenerationRequest request);

    //! @brief Any thread: cancel a request by id. Same effect and same guarantees as
    //! RequestHandle::cancel(): the request's cancel token is set immediately on the caller's
    //! thread and nothing rides the command queue, so it cannot be lost under overload.
    //! Idempotent; unknown or retired ids are ignored (ids are never reused).
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
    //! request run; kCancel abandons what is queued and what is resident. Neither can interrupt a
    //! step the runtime is inside -- the actor is in it and cannot reach a boundary -- so both wait
    //! at least that long. Both leave every live stream terminal, so no caller stays parked in
    //! waitPop().
    //!
    //! The first caller's mode wins; a later kCancel cannot escalate a drain already under way.
    void shutdown(ShutdownMode mode);

    //! @brief Requests the actor has taken off the queue but not yet admitted. Observability only.
    //!
    //! Not a count of everything submitted: a request stays invisible here until the actor reaches a
    //! boundary and drains it, which it cannot do while it is inside a step. Do not use this to wait
    //! for a submission to become visible.
    int32_t queued() const noexcept;

    //! @brief Requests currently resident in the session. Observability only.
    int32_t resident() const noexcept;

    //! @brief False once shutdown() has been called.
    bool running() const noexcept;

    //! @brief Counters for observability; see EngineMetrics for the consistency contract.
    EngineMetrics metrics() const noexcept;

    //! @brief The runtime's context-cache counters as the actor last copied them: once when the
    //! session opens, then after every step, before the step's outcomes are published, so a caller
    //! released by a request's outcome already sees the step that finished it. Observability only;
    //! nullopt when the runtime has no context cache.
    //!
    //! The coordinator behind them is the actor's to touch, so the copy is all another thread can see.
    std::optional<ContextCacheMetrics> contextCacheMetrics() const;

    //! @brief Any thread: the scheduling policy in effect.
    //!
    //! Resolved at construction. When a plugin refuses a compact step under any mode but an explicit mixed, a
    //! fresh session opens without compact execution, and the policy is resolved again against it: auto becomes
    //! prefillFirst over single-cohort steps, dropping pause with the startup's warning. Residents that have
    //! sampled a token fail with a retryable reason; the others go back to the queue at their (priority,
    //! arrival) place, keeping their handles and streams, and run again from the start of their prompts. An
    //! explicit mixed instead fails every resident, stops the engine (see status()) and keeps the policy it had.
    ResolvedSchedulingPolicy resolvedPolicy() const;

    //! @brief Any thread: whether the engine still serves requests, and why compact execution is unavailable.
    EngineStatus status() const;

private:
    //! Shared tail of both constructors: validate config, start the actor, and wait for it to
    //! report the session open -- or throw with the actor joined when it could not be.
    void start();

    //! Signal the actor that there is something to look at, safely against it being midway
    //! between its predicate check and its wait.
    void wakeActor() noexcept;

    void actorLoop();

    //! Actor: open the session through the factory. False (and no session) when it refuses or throws.
    bool openSession();

    //! The stages the actor cycles through, kept as separate members rather than inlined into
    //! actorLoop() because the boundaries between them are where the engine's invariants hold: no
    //! request is scheduled until the queue has been folded in, nothing becomes resident outside
    //! admission, and the runtime is reached from exactly two of them (admit and step).

    //! Fold everything the queue holds into actor-owned state. Runs before admission each round, so
    //! a cancel posted since the previous round is seen before its request could be promoted.
    void drainCommandQueue();

    //! Retire every queued request whose cancel token is set: terminal kCancelled, backlog
    //! reservation released. Runs at every drain and again right before each admission.
    void retireCancelledQueued();

    //! The priorityReservedSlots gate: false when @p head may not take one of the free slots.
    bool slotAvailableFor(LLMGenerationRequest const& head) const noexcept;

    //! Offer the queue head to the session. Single-cohort execution: at most one admission per step
    //! boundary, none while an admission's context step is pending. Compact execution: called until
    //! it returns false. Returns true when the head left the queue (admitted, rejected, or failed), false
    //! when it stays (no head, the gate said wait, or the session refused as transient pressure).
    bool tryAdmitHead();

    struct QueuedRequest;

    //! The head-of-line gate: batch compatibility with the residents, the session's exclusivity
    //! rules (a constrained request runs alone), and agreeing sampling parameters on a decoder that
    //! requires them (RuntimeSession::requiresUniformSampling). Returns false and bumps the matching stall counter
    //! when the head must wait. Under SchedulingPolicy::kPriority, a head that outranks every
    //! resident and waits on batch compatibility is reported once, with the field that differs.
    bool mayAdmit(QueuedRequest& head);

    //! Run one step: under compact execution the planned step; otherwise the pending admission's
    //! context step if there is one, else one decode step over every resident. Returns false when
    //! nothing is resident. A step that fails or throws fails the session (see failSession).
    bool runStep();

    //! Compact execution: writes into @p plan the step StepPlanner composes from the ledger, as
    //! descriptors in its order (prompt chunks first, then decode rows).
    void planCompactStep(std::vector<ScheduledSequenceDescriptor>& plan);

    //! Record which residents @p step paused: in mStepPaused for the step's statistics, and in each record's
    //! consecutivePausedSteps, which the next plan reads against maxPauseSteps.
    void notePausedResidents(StepPlan const& step);

    //! Per-request statistics of a completed step, recorded before commit() retires anyone: prompt
    //! chunks, the first token's time, and the time the paused residents waited. Drops from mStepPaused the
    //! residents the step retired, so countStep() counts the same paused residents.
    void recordStepStats(StepResult const& result, std::chrono::steady_clock::time_point stepStart,
        std::chrono::steady_clock::time_point executed) noexcept;

    //! Count what a completed step ran (EngineMetrics), classified by its execution record.
    void countStep(StepResult const& result, uint64_t stepTimeUs) noexcept;

    //! A step the runtime could not run leaves the session's state undefined: every resident owes
    //! its caller an outcome (kExecutionError carrying @p reason), the session is closed, and a
    //! fresh one is opened. If none can be, the engine keeps answering but fails every request it
    //! is given, with the reason, rather than parking callers forever (see
    //! EngineConfig::maxConsecutiveSessionFailures). @p compactRefused marks a step a plugin refused
    //! (failRefusedCompactStep).
    void failSession(std::string const& reason, bool compactRefused = false);

    //! A plugin refused a compact step: an explicit mixed stops the engine; otherwise the residents that sampled no
    //! token are queued again and a fresh session runs under the policy resolved again (see resolvedPolicy()). The
    //! other residents fail as FailureKind::kCompactStepRefused, or as kEngineStopped when no session reopens or the
    //! policy cannot be resolved again, which stops the engine.
    void failRefusedCompactStep(std::string const& reason);

    //! Retire every resident with @p reason and close the session. Returns the residents' outcomes for
    //! publishOutcomes().
    [[nodiscard]] std::vector<RequestCoordinator::TerminalOutcome> closeSession(std::string const& reason);

    //! Open a fresh session after a failed one. False when none opens; the engine is then published unavailable,
    //! with @p compactReason as why compact execution is unavailable when it is non-empty.
    bool reopenSession(std::string const& compactReason = {});

    //! After a refused compact step: queue every resident whose prompt has not completed again at its place, its
    //! stream released by the session unfinished. One the session cannot release fails with @p reason.
    void requeuePromptIncomplete(std::string const& reason);

    //! Resolve the policy against @p capabilities and set up the planner to match; publishes both for
    //! resolvedPolicy() and status(). @p fallbackReason non-empty logs the policy as a WARNING with it.
    //! @throws std::invalid_argument if the configuration cannot be served (see resolveSchedulingPolicy).
    void adoptPolicy(SessionCapabilities const& capabilities, std::string const& fallbackReason = {});

    //! The session is gone for good: publish mSessionLostReason (and @p compactReason, when non-empty) to status().
    void publishUnavailable(std::string const& compactReason = {});

    //! Publish coordinator outcomes: terminate each record with the mapped status, a failure as one of @p kind.
    void publishOutcomes(
        std::vector<RequestCoordinator::TerminalOutcome> outcomes, FailureKind kind = FailureKind::kOther);

    //! Record the outcome and release anyone waiting on it. A request that did not complete has
    //! its channel finished with the matching reason too, so a streaming consumer is not left
    //! parked on a channel the runtime never touched.
    void terminate(RequestRecord& record, TerminalStatus status, std::string errorMessage = {},
        FailureKind kind = FailureKind::kOther);

    struct QueuedRequest
    {
        std::shared_ptr<RequestRecord> record;
        std::unique_ptr<LLMGenerationRequest> request;
        //! Its wait on batch compatibility ahead of lower-priority residents has been reported.
        bool reportedCompatibilityStall{false};
    };

    //! Insert a drained or requeued request in policy order: arrival under kFcfs, (priority, arrival) under
    //! kPriority.
    void enqueue(QueuedRequest queued);

    std::unique_ptr<LLMInferenceRuntime> mRuntime;
    SessionFactory mOpenSession;
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
        std::atomic<uint64_t> stallsExclusive{0};
        std::atomic<uint64_t> stallsNoCapacity{0};
        std::atomic<uint64_t> admittedMidFlight{0};
        std::atomic<uint64_t> completed{0};
        std::atomic<uint64_t> cancelled{0};
        std::atomic<uint64_t> failed{0};
        std::atomic<uint64_t> queueLatencyTotalUs{0};
        std::atomic<uint64_t> queueLatencyMaxUs{0};
        std::atomic<uint64_t> queueLatencyCount{0};
        std::atomic<uint64_t> decodeSteps{0};
        std::atomic<uint64_t> decodeGraphSteps{0};
        std::atomic<uint64_t> promptSteps{0};
        std::atomic<uint64_t> mixedSteps{0};
        std::atomic<uint64_t> decodeSubsetSteps{0};
        std::atomic<uint64_t> stepSequences{0};
        std::atomic<uint64_t> stepPromptChunks{0};
        std::atomic<uint64_t> partialPromptChunks{0};
        std::atomic<uint64_t> stepDecodeRows{0};
        std::atomic<uint64_t> stepTokens{0};
        std::atomic<uint64_t> stepSampledRows{0};
        std::atomic<uint64_t> stepTimeTotalUs{0};
        std::atomic<uint64_t> stepTimeMaxUs{0};
        std::atomic<uint64_t> pausedResidentSteps{0};
        std::atomic<uint64_t> compactRefusals{0};
    };
    mutable Counters mCounters;

    void countOutcome(TerminalStatus status) noexcept;
    //! Returns the latency it recorded, in microseconds.
    uint64_t recordQueueLatency(RequestRecord const& record) noexcept;

    //! Actor: copy the session's context-cache counters for contextCacheMetrics().
    void publishContextCacheMetrics();
    mutable std::mutex mCacheMetricsMutex;
    std::optional<ContextCacheMetrics> mCacheMetrics;

    //! The constructor's handshake with the actor: it waits here until the actor has opened the
    //! session, or reported that it could not.
    enum class SessionState : uint8_t
    {
        kOpening,
        kOpen,
        kUnavailable,
    };
    std::mutex mSessionMutex;
    std::condition_variable mSessionCv;
    SessionState mSessionState{SessionState::kOpening};
    //! The actor is the only writer of mPolicy and mStatus and reads mPolicy without the lock; it writes both
    //! under mStatusMutex, which every other reader takes.
    mutable std::mutex mStatusMutex;
    ResolvedSchedulingPolicy mPolicy;
    EngineStatus mStatus;
    //! Why the scheduling policy was rejected; reported by the constructor.
    std::string mPolicyError;

    //! Actor-owned from here down. No lock guards these because no other thread reaches them.
    //! The session the actor drives. Null only after a failed step could not be followed by a
    //! fresh session; every request then fails at admission with the reason.
    std::unique_ptr<RuntimeSession> mSession;
    std::string mSessionLostReason;
    //! Session failures since the last completed step (see EngineConfig::maxConsecutiveSessionFailures).
    int32_t mSessionFailuresInARow{0};
    std::deque<QueuedRequest> mQueued;
    //! A non-zero priority under kFcfs has been warned about.
    bool mWarnedIgnoredPriority{false};
    //! Every resident of the session, keyed by ResidentRef: the one ledger. commit() retires
    //! through it, steps are built from it, and a cancelling shutdown reaches every resident
    //! through it. Outlives every request, like the session it describes.
    RequestCoordinator mResidents;
    //! The admission whose context step is the next step to run (single-cohort execution).
    std::optional<ResidentRef> mPendingContext;
    //! Set when the session executes compact steps; composes every step from the ledger.
    std::optional<StepPlanner> mPlanner;
    //! Planner input of the current step, index-aligned: the session's residents and their ledger state.
    std::vector<ResidentRef> mStepResidents;
    std::vector<ResidentState> mStepStates;
    //! Residents the current plan paused, counted once the step completes.
    std::vector<ResidentRef> mStepPaused;
    //! The current step's descriptors. Like the vectors above and mSessionResidents, reserved for the engine's
    //! batch size at open so planning a compact step does not allocate.
    std::vector<ScheduledSequenceDescriptor> mStepPlan;
    //! The session's residents after a step, checked against the ledger.
    std::vector<ResidentRef> mSessionResidents;

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
