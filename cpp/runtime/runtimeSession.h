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

#include "common/pagedKvTypes.h"
#include "runtime/exec/scheduledStep.h"
#include "runtime/llmRankRuntime.h"

#include <atomic>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace trt_edgellm
{
namespace rt
{

//! What became of one admission.
struct AdmissionResult
{
    enum class Status : uint8_t
    {
        kAdmitted,
        //! Transient pressure; nothing was modified, retry at a later step.
        kNoCapacity,
        //! The session rejects the request outright; nothing was modified.
        kRejected,
    };

    Status status{Status::kRejected};
    ResidentRef ref;    //!< valid iff kAdmitted
    std::string reason; //!< diagnostic for kRejected
    //! The prompt as the runtime tokenized it (chat template applied, media expanded), and the
    //! prefix of it that already has state behind it (a context-cache hit; zero when cold). The
    //! sequence's context step covers exactly [prefillStart, promptLength).
    int32_t promptLength{0};
    int32_t prefillStart{0};
};

//! Tokens one resident produced in one step.
struct TokenDelta
{
    std::vector<int32_t> tokenIds;
};

//! Thrown by RuntimeSession::admit when a failure inside the reservation -- the context-cache
//! lease, device work -- leaves the session's state unknown. The ordinary refusals
//! (std::invalid_argument, std::runtime_error) modify nothing and the session goes on; after this
//! one the session is not to be driven further, and every resident's outcome is the caller's to
//! publish. The same contract a step's ok == false carries, surfaced from admission.
class SessionLostError : public std::runtime_error
{
    using std::runtime_error::runtime_error;
};

//! What one executed step did. The single commit unit: everything in it happened together, and
//! nothing outside it happened.
struct StepResult
{
    bool ok{false};
    //! The work of a single-cohort step. A compact step names each sequence's work in execution[i].work and
    //! reports kContext here when any sequence ran a prompt chunk, kDecode otherwise.
    SequenceWork work{SequenceWork::kContext};
    //! Runtime-assigned identity of the step whose effects this result commits. Zero when no step
    //! ran: a seated prefill that fails before it reaches the engine files its sequence under
    //! `finished` and executes nothing.
    StepId stepId{0};
    //! The sequences the plan named, in plan order: what the caller asked this step to run.
    std::vector<SequenceIdentity> participants;
    //! What the runtime executed for them, in the same order, and empty when no step ran. A
    //! descriptor's pastLength is the positions the sequence had state for before the step and its
    //! queryLength the positions the step added -- which a visual-token pruner makes fewer than the
    //! prompt range the plan asked a context step to cover. The caller's ledger advances from this,
    //! never from its own plan, so it mirrors what ran rather than what was requested.
    std::vector<ScheduledSequenceDescriptor> execution;
    //! Tokens produced this step, per participating resident (absent for a resident that
    //! produced none).
    std::vector<std::pair<ResidentRef, TokenDelta>> deltas;
    //! Residents that reached a terminal state and were evicted this step, with the finished
    //! sequence snapshot; their physical resident slots are released (and their refs invalidated) as part of this
    //! commit unit. A seat that failed its context step surfaces here on the step whose eviction
    //! files it, not before. Dense-row compaction is runtime-private and never reported: refs are
    //! stable resident identities, so the caller's table needs no re-keying.
    std::vector<std::pair<ResidentRef, BatchResult>> finished;
    //! True when this step published context-cache prefix blocks (cache visibility commits with
    //! the step that produced it, never ahead of it).
    bool publishedPrefix{false};
    //! True when the base engine ran this step from a captured CUDA graph rather than an eager enqueue.
    bool graphLaunched{false};
    //! True when the step ran as one compact (phase-8) forward; false for a single-cohort context step
    //! and for the phase-3 decode step of every resident.
    bool compact{false};
    //! True when ok is false because a plugin refused the compact forward with the device healthy. Compact
    //! execution is then unavailable for the runtime's life: every session opened afterwards reports
    //! capabilities().compact false. The residents still fail -- the layers before the refusing one advanced their
    //! recurrent state in place -- but their requests can be retried.
    bool compactRefused{false};
    //! Why ok is false, complete for display; empty when the runtime reports no specific cause.
    std::string failure;
};

//! What the deployment behind a session can execute, and the engine limits a scheduler derives
//! its step budgets from. Fixed for the life of the runtime, except that a refused compact step
//! (StepResult::compactRefused) makes compact execution unavailable from then on.
struct SessionCapabilities
{
    //! Whether a step may combine prefill chunks of several sequences with decode rows (phase 8).
    bool compact{false};
    //! Why compact execution is unavailable, complete for display. Empty when compact execution is available.
    std::string reason;
    //! The batch size and input length the engine was built with, and the KV page size.
    int32_t maxBatchSize{0};
    int32_t maxInputLength{0};
    int32_t kvPageSize{kTOKENS_PER_PAGE};
};

//! What a scheduler drives: one runtime held open for the life of an engine, that sequences enter
//! and leave one at a time while the ones already resident keep decoding.
//!
//! The session owns what outlives any single request: the resident slot pool, the step counter,
//! the ragged execution scratch, the context-cache lease container and the stream the work runs
//! on. A request owns nothing here beyond its slot: it is admitted with admit(), executed by the
//! steps the caller schedules with execute(), and leaves through a StepResult's finished set when
//! its slot is evicted. An empty session idles; it is not torn down and reopened between
//! requests, so the first sequence into an empty session and the tenth into a full one take the
//! same path.
//!
//! What the session does not decide: which sequences take part in a step, and when. The caller
//! emits the plan -- one ScheduledSequenceDescriptor per participating resident -- and the session
//! checks it against the residents it holds, lowers it to the runtime's execution batch, runs it
//! and reports what happened. Under single-cohort execution a plan is either the context step of
//! one just-admitted sequence or one decode step over every resident, in resident order; under
//! compact execution a plan may combine prompt chunks of several sequences with decode rows (see
//! execute()).
//!
//! Abstract so the scheduler's admission, cancellation and shutdown paths can be tested against a
//! scripted session without a device; RankRuntimeSession is the one over an LLMRankRuntime.
//!
//! Threading: one thread drives a session, the same thread that owns the runtime. Every method
//! must be called from it.
class RuntimeSession
{
public:
    virtual ~RuntimeSession() = default;

    //! @brief Admit one request as a new resident.
    //!
    //! Validation, chat templating / tokenization or the encoder preprocess, the generate-budget
    //! check, the context-cache lease (when reuse is on) and the seat commit. The sequence's
    //! context step is not run here: the caller schedules it as the next execute() -- it must be
    //! the next one, since the admission staging buffers stay valid only until the next admission's
    //! preprocess. A second admit() before that step is a contract violation. Under compact
    //! execution (capabilities().compact) any admitted sequence may have an unfinished prompt,
    //! advanced by the prompt chunks the caller schedules in any later steps.
    //!
    //! A request that must have the session to itself (see requiresExclusiveAdmission) is
    //! admitted only into an empty session and refuses company while resident; a caller admitting
    //! into or beside one violates the contract. Whether two requests may share the session at all
    //! -- adapter, cache policies, and sampling parameters on a decoder that requires uniform sampling
    //! (requiresUniformSampling) -- is the caller's decision (BatchCompatibility), taken before this call; the session
    //! verifies the parameters agree and fails loudly when they do not.
    //!
    //! @param cancelToken The request's cancellation bit, polled by the runtime before the
    //!        sequence's context step and at every step it takes part in; null when the caller
    //!        cancels through the stream channel only.
    //! @throws std::invalid_argument for a request this session never serves (per-request LoRA,
    //!         speech output, the legacy system-prompt capture, trajectory execution, more than one
    //!         sequence). Nothing is modified.
    //! @throws std::runtime_error when validation, templating, tokenization or the encoder
    //!         preprocess fails with the session's stream healthy, or when the prompt exceeds the KV
    //!         capacity or its uncached part exceeds the engine's input length (the
    //!         EDGELLM_INPUT_TOO_LONG marker rides in the message). Nothing is modified.
    //! @throws SessionLostError when the encoder preprocess fails leaving the session's stream in
    //!         error, when the reservation itself fails inside the runtime (a device failure in the
    //!         context-cache lease), or when staging the reserved media fails on the device (a failed
    //!         CUDA call or a stream in error). The session's state is then unknown: the caller fails
    //!         it, residents included, and does not drive it further.
    virtual AdmissionResult admit(LLMGenerationRequest const& request, RequestId requestId,
        std::shared_ptr<std::atomic<bool> const> cancelToken = nullptr) = 0;

    //! @brief Run one step over the sequences named by @p plan.
    //!
    //! The plan is checked against the residents this session holds -- identities, work,
    //! committed lengths and query lengths must describe exactly what the runtime would execute --
    //! and then executed. A plan of every resident as kDecode with one query token each, in
    //! resident order, is a decode step. Otherwise, under single-cohort execution the plan is the
    //! one pending admission as kContext covering its whole not-yet-committed prompt; under compact
    //! execution it names any subset of residents in any order, each at most once, with pastLength
    //! equal to its committed length: a prompt-incomplete resident as kContext with
    //! 0 < queryLength <= its remaining prompt and selectLastTokenLogits exactly when the chunk
    //! ends the prompt, any other resident as kDecode with queryLength 1. Residents the plan does
    //! not name are neither read nor written. A plan naming a sequence the session does not hold,
    //! or describing lengths that disagree with the ledger, is a contract violation and leaves the
    //! session untouched.
    //!
    //! An exclusive resident (see exclusiveResident()) keeps the single-cohort steps even under compact
    //! execution -- a guided sequence, media under a visual-token pruner, media whose token layout only its
    //! encoder determines -- so its prompt step must cover its whole uncommitted prompt in one plan of
    //! its own; admission bounds that prompt by S_in.
    //!
    //! A result with ok == false means the runtime could not run the step; every resident's
    //! outcome is then the caller's to publish (it will not arrive through a later result), and
    //! the session is not to be driven further. compactRefused and failure tell a refused compact
    //! forward apart from the other failures.
    //! @throws std::runtime_error on a device failure inside the step, with the same meaning.
    virtual StepResult execute(std::vector<ScheduledSequenceDescriptor> const& plan) = 0;

    //! Convert a finished snapshot to a response: the tokenizer's decode and the final stop-string
    //! trim, exactly as the blocking path assembles its own.
    virtual LLMGenerationResponse materialize(
        BatchResult const& result, std::vector<std::string> const& stopStrings) const = 0;

    //! The refs of every live resident, in resident (execution) order.
    virtual std::vector<ResidentRef> residents() const = 0;

    //! residents() written into @p out, reusing its capacity: the scheduler reads the set every step.
    virtual void copyResidents(std::vector<ResidentRef>& out) const
    {
        out = residents();
    }

    //! @brief After a step failed with StepResult::compactRefused, before the session closes: release @p ref's
    //!        stream channel unfinished, so its request can be admitted into another session.
    //!
    //! Only for a resident whose prompt has not completed, which has pushed nothing to its channel. The resident
    //! stays until the session closes, which returns its context-cache lease like any other's. A session that
    //! attaches no stream channel has nothing to release.
    //! @throws std::runtime_error when @p ref is not such a resident.
    virtual void releaseForReadmission(ResidentRef ref)
    {
        static_cast<void>(ref);
    }

    //! True when no sequence is resident. The session is not finished; it is waiting.
    virtual bool idle() const noexcept = 0;

    //! True while a resident requires the session to itself (see admit()).
    virtual bool exclusiveResident() const noexcept = 0;

    //! Whether @p request would require the session to itself: a grammar-constrained request, one
    //! with a token callback, media under an active visual-token pruner, or, in a compact session,
    //! media whose token layout only its encoder determines. Pure; the caller asks before deciding
    //! to admit.
    virtual bool requiresExclusiveAdmission(LLMGenerationRequest const& request) const noexcept = 0;

    //! Whether @p request would run on a decoder that takes one set of sampling parameters per batch -- the
    //! deployment's speculative decoder -- so that it shares steps only with requests that sample alike
    //! (BatchCompatibility::samplingAgrees). Pure; the caller asks before deciding to admit. A session whose
    //! decoders all sample per sequence reports false.
    virtual bool requiresUniformSampling(LLMGenerationRequest const& request) const noexcept
    {
        static_cast<void>(request);
        return false;
    }

    //! What this deployment can execute. Constant for the session's life and across reopens until a
    //! step reports compactRefused. A session that does not override it reports single-cohort execution
    //! only.
    virtual SessionCapabilities capabilities() const
    {
        return SessionCapabilities{};
    }

    //! The runtime's context-cache counters, cumulative over the runtime's life rather than this
    //! session's; nullopt when the runtime has no context cache.
    virtual std::optional<ContextCacheMetrics> contextCacheMetrics() const noexcept
    {
        return std::nullopt;
    }
};

//! The RuntimeSession over one LLMRankRuntime. Opened by LLMRankRuntime::openSession(); holds the
//! runtime's one-request latch for its life, so the blocking handleRequest() path is refused until
//! the session is destroyed.
class RankRuntimeSession final : public RuntimeSession
{
public:
    ~RankRuntimeSession() override;

    RankRuntimeSession(RankRuntimeSession const&) = delete;
    RankRuntimeSession& operator=(RankRuntimeSession const&) = delete;

    AdmissionResult admit(LLMGenerationRequest const& request, RequestId requestId,
        std::shared_ptr<std::atomic<bool> const> cancelToken = nullptr) override;
    StepResult execute(std::vector<ScheduledSequenceDescriptor> const& plan) override;
    LLMGenerationResponse materialize(
        BatchResult const& result, std::vector<std::string> const& stopStrings) const override;
    std::vector<ResidentRef> residents() const override;
    void copyResidents(std::vector<ResidentRef>& out) const override;
    void releaseForReadmission(ResidentRef ref) override;
    bool idle() const noexcept override;
    bool exclusiveResident() const noexcept override;
    bool requiresExclusiveAdmission(LLMGenerationRequest const& request) const noexcept override;
    bool requiresUniformSampling(LLMGenerationRequest const& request) const noexcept override;
    SessionCapabilities capabilities() const override;
    std::optional<ContextCacheMetrics> contextCacheMetrics() const noexcept override;

private:
    friend class LLMRankRuntime;

    //! The settings the runtime applies to every resident alike, fixed when a sequence enters an
    //! empty session and checked against every later admission; the caller's compatibility gate
    //! keeps them equal. Sampling, seeds, budgets and logprobs widths are per sequence.
    struct BatchWideParameters
    {
        bool enableThinking{false};
        SpecProposalSampling proposalSampling{SpecProposalSampling::kAuto};
        //! The decoder the batch runs: the deployment's speculative one unless the sampling selects the
        //! default decoder, exactly as the blocking path selects it for a request.
        DecodingStrategy* strategy{nullptr};
    };

    //! A sequence seated by admit() whose prompt has not completed. seat.row is its execution row
    //! only until the next eviction; compact execution locates the row by ref.
    struct PendingAdmission
    {
        LLMRankRuntime::GenerationSession::PendingSeat seat;
        ResidentRef ref;
        bool leased{false};
        bool exclusive{false};
        //! At least one prompt chunk ran (compact execution).
        bool started{false};
    };

    RankRuntimeSession(LLMRankRuntime& runtime, cudaStream_t stream, bool allowCompactExecution);

    //! Runtime-side setup: strategy, context, ragged scratch, identity page tables when there is no
    //! cache, the deployment's compact capability (resolved by the first session), the context-cache
    //! request (opened empty), stream finalizer, generation session.
    //! False on the refusals a blocking request would also have met at this point.
    bool open();

    //! The batch-wide settings @p request asks for, its decoder selected exactly as beginGeneration selects it.
    BatchWideParameters batchWideParametersFor(LLMGenerationRequest const& request) const;
    //! First admission into an empty session: install its batch-wide settings on the context, and its
    //! decoder on the generation session.
    void latchBatchWide(BatchWideParameters const& parameters);
    //! Every later admission: the batch-wide settings must match what is resident.
    void checkBatchWide(BatchWideParameters const& parameters) const;
    //! Whether @p parameters are the residents' batch-wide settings.
    bool sameBatchWide(BatchWideParameters const& parameters) const noexcept;
    //! @p request's generate budget, bounded by the KV capacity behind its own @p promptLength, with the
    //! headroom @p strategy (the batch's decoder) reserves in the base and draft caches; installed on
    //! the seed before the seat.
    //! @throws std::runtime_error when the request's maxGenerateLength is not positive, or when that
    //!         capacity leaves no token to generate.
    int32_t generateBudgetFor(
        LLMGenerationRequest const& request, int32_t promptLength, DecodingStrategy const& strategy);
    //! Open the session's context-cache batch for @p strategy's fills: a speculative decoder's sequences lease a
    //! paired base and draft working set and publish speculative endpoints, any other decoder's a base one. Opened
    //! with no sequence; reopened by latchBatchWide when a fill's decoder needs the other kind.
    bool openCacheRequest(DecodingStrategy const& strategy);

    //! Run the pending admission's context step (prefillSeated and the lookahead-token pass).
    StepResult executeContext(ScheduledSequenceDescriptor const& sequence);
    //! One decode step over every resident.
    StepResult executeDecode(std::vector<ScheduledSequenceDescriptor> const& plan);
    //! One compact phase-8 step over the residents @p plan names, in any order: prompt chunks of
    //! prompt-incomplete residents and decode rows of the others. When the cancellations at the step's
    //! start leave only decode rows of every live resident, the phase-3 decode step runs instead.
    StepResult executeCompact(std::vector<ScheduledSequenceDescriptor> const& plan);
    //! The phase-3 decode step for @p plan's decode rows once the cancelled residents are evicted;
    //! participants are all of @p plan.
    StepResult executeDecodeAfterCancellation(std::vector<ScheduledSequenceDescriptor> const& plan);
    //! The phase-3 decode step for @p plan's decode rows of live residents, which must cover every resident, run in
    //! resident order; participants are all of @p plan.
    StepResult executeDecodeInResidentOrder(std::vector<ScheduledSequenceDescriptor> const& plan);
    //! The media stage of admit(): stage the reserved slot's media, or give the reservation back and rethrow. A
    //! failure on the device is rethrown as SessionLostError.
    void stageMediaOrRelease(LLMRankRuntime::GenerationSession::AdmissionIntent& intent);
    //! The prompt-incomplete resident @p ref names, or end().
    std::vector<PendingAdmission>::iterator findPending(ResidentRef ref);
    //! The execution row of live resident @p ref, or -1.
    int32_t rowOf(ResidentRef ref) const noexcept;
    //! Fill in what a step reports beyond ok: participants from the validated @p plan, the
    //! execution record from the runtime's own step (only when @p executed -- a seated prefill that
    //! failed before reaching the engine leaves the previous step's record behind), deltas as the
    //! tokens a surviving resident holds beyond its reported watermark, and finished as whatever
    //! the step's eviction filed.
    void finalizeResult(StepResult& result, std::vector<ScheduledSequenceDescriptor> const& plan, bool executed);
    //! Per-sequence run metrics, recorded as each sequence retires (the blocking path records
    //! them per request at its end).
    void recordSequenceMetrics(BatchResult const& result) noexcept;

    LLMRankRuntime& mRuntime;
    cudaStream_t mStream{};
    //! Declared first so it is released last: the latch outlives everything below.
    LLMRankRuntime::InProgressGuard mGuard;
    DecodingInferenceContext mContext;
    DecodingStrategy* mStrategy{nullptr};
    std::optional<ManagedKVCacheRequest> mManagedRequest;
    //! The decoder kind mManagedRequest was opened for (openCacheRequest).
    bool mCacheRequestSpeculative{false};
    DecodingTokenStateContract mCacheRequestContract{DecodingTokenStateContract::kCommittedPlusLookahead};
    DecodingKvHeadroom mKvHeadroom{};
    //! Finishes every still-open channel when the session closes; declared after the context it
    //! reads and before the generation session, so it runs after the session and before the
    //! context is destroyed.
    std::optional<StreamChannelFinalizer> mStreamFinalizer;
    std::unique_ptr<LLMRankRuntime::GenerationSession> mSession;

    BatchWideParameters mBatchWide;
    //! SessionOptions::allowCompactExecution: without it, no compact steps.
    bool mAllowCompact{false};
    //! The session executes compact steps (the deployment's compact capability holds and its resources exist).
    bool mCompact{false};
    //! Compact-step scratch, reserved by open() for a compact session and cleared per step.
    std::vector<int32_t> mCompactPlanRows;
    std::vector<LLMRankRuntime::GenerationSession::CompactRow> mCompactOrder;
    std::vector<ResidentRef> mCompletingRefs;
    std::vector<ScheduledSequenceDescriptor> mDecodeOrder;
    //! Prompt-incomplete residents in admission order: at most one under single-cohort execution,
    //! where it is the sequence whose context step runs next.
    std::vector<PendingAdmission> mPending;
    //! Per resident slot: the ref the caller holds for it, and how many of its tokens have been
    //! reported through deltas. A finished sequence is looked up here to name its ref one last
    //! time, after the slot pool has already released it.
    struct Tracked
    {
        ResidentRef ref;
        size_t reported{0};
    };
    std::unordered_map<CacheSlot, Tracked> mTracked;
    bool mExclusiveResident{false};
    //! The exclusive resident needs whole-prompt single-cohort steps (SlotSeed::requiresExclusiveSession).
    bool mSingleCohortResident{false};
    //! A generate budget reduced to the KV capacity has been logged as a WARNING; later ones go to DEBUG.
    bool mWarnedGenerateBudget{false};
};

} // namespace rt
} // namespace trt_edgellm
