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

#include "runtime/exec/scheduledStep.h"
#include "runtime/llmRankRuntime.h"

#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace trt_edgellm
{
namespace rt
{

//! Everything validation, tokenization, and the encoder preprocess produce. Building one touches
//! no resident state; discarding one leaves the batch byte-identical.
struct AdmissionIntent
{
    //! Engine-level identity, carried through for the caller's bookkeeping; the runtime keys
    //! residents by the seed's originalIndex.
    uint64_t requestId{0};
    SlotSeed seed;
};

struct AdmissionResult
{
    enum class Status : uint8_t
    {
        kAdmitted,
        //! Transient pressure; nothing was modified, retry at a later tick.
        kNoCapacity,
        //! The batch rejects the intent outright; nothing was modified.
        kRejected,
    };

    Status status{Status::kRejected};
    ResidentRef ref;    //!< valid iff kAdmitted
    std::string reason; //!< diagnostic for kRejected
};

//! The immutable input to one executed step: which residents participate, materialized by the
//! caller. The prefill-first scheduler runs a single cohort, so a decode view is "every live resident" and a prefill
//! view is the one newly admitted ref; the types exist so the executed set is an explicit input rather than
//! stepper-internal knowledge, and so B=1/B=N views can evolve without an API change.
struct ImmutablePrefillBatch
{
    ResidentRef target;
};

struct ImmutableDecodeBatch
{
    std::vector<ResidentRef> residents;
};

//! Tokens one resident produced in one step.
struct TokenDelta
{
    std::vector<int32_t> tokenIds;
};

//! What one executed step did. The single commit unit: everything in it happened together, and
//! nothing outside it happened.
struct StepResult
{
    bool ok{false};
    SequenceWork work{SequenceWork::kContext};
    //! Runtime-assigned identity of the ScheduledStep whose effects this result commits.
    StepId stepId{0};
    //! Exact ordered logical/physical cohort executed by this step.
    std::vector<SequenceIdentity> participants;
    //! Pointer-free logical execution plan used for cross-rank shape/state consensus.
    std::vector<ScheduledSequenceDescriptor> execution;
    //! Tokens produced this step, per participating resident (absent for a resident that
    //! produced none).
    std::vector<std::pair<ResidentRef, TokenDelta>> deltas;
    //! Residents that reached a terminal state and were evicted this step, with the finished
    //! sequence snapshot; their physical resident slots are released (and their refs invalidated) as part of this
    //! commit unit. A seat that failed its seated prefill surfaces here on the step whose eviction
    //! files it, not before. Dense-row compaction is runtime-private and never reported: refs are
    //! stable resident identities, so the caller's table needs no re-keying.
    std::vector<std::pair<ResidentRef, BatchResult>> finished;
    //! True when this step published context-cache prefix blocks (cache visibility commits with
    //! the step that produced it, never ahead of it).
    bool publishedPrefix{false};
};

//! The runtime's execution facade for the stepped control plane: typed operations over immutable
//! inputs, typed results, no callbacks, and no mutable state lent across the boundary.
//!
//! Wraps a live GenerationSession. In this stage the facade is exercised by unit tests while the
//! engine still drives the loop through handleRequest; the loop moves behind this interface in
//! the next stage, at which point GenerationBoundaryHook is deleted.
//!
//! Threading: one thread drives a stepper, the same thread that owns the session.
class RuntimeStepper
{
public:
    //! Adopts the session's canonical resident identities.
    explicit RuntimeStepper(LLMRankRuntime::GenerationSession& session);

    RuntimeStepper(RuntimeStepper const&) = delete;
    RuntimeStepper& operator=(RuntimeStepper const&) = delete;

    //! Logical admission only: reservation (budget + context-cache lease) and the seat commit at
    //! appendSlot. The seated prefill is not run here -- it is the next prefill tick, and must be
    //! the next operation on this stepper (the admission staging buffers stay valid exactly until
    //! the next admission's preprocess).
    AdmissionResult admit(AdmissionIntent intent);

    //! The seated batch-1 prefill for the pending admission (batch.target must be its ref), or --
    //! when nothing is pending -- the founding prefill of the wrapped session.
    StepResult prefill(ImmutablePrefillBatch const& batch);

    //! One decode step over the live batch. batch.residents must name every live resident: this stage
    //! runs a single cohort, and the view exists so the executed set is an explicit input.
    StepResult decode(ImmutableDecodeBatch const& batch);

    //! The refs of every live resident, in slot order. The caller's view for building batches.
    std::vector<ResidentRef> residents() const;

private:
    //! Derive an operation's deltas and finished set from the state it left behind: deltas are
    //! the tokens a resident holds beyond its reported watermark.
    void finalizeResult(StepResult& result);

    LLMRankRuntime::GenerationSession& mSession;
    //! originalIndex -> canonical physical resident identity for every live sequence.
    std::unordered_map<int32_t, ResidentRef> mRefs;
    //! originalIndex -> how many of the resident's tokens have been reported through deltas. The
    //! founding baseline excludes tokens the founding prefill already produced inside
    //! beginGeneration, so the founding prefill tick reports them as its delta.
    std::unordered_map<int32_t, size_t> mReported;
    //! The admission seated by admit() and awaiting its prefill tick.
    struct PendingAdmission
    {
        LLMRankRuntime::GenerationSession::PendingSeat seat;
        ResidentRef ref;
        int32_t originalIndex{};
        bool leased{false};
    };
    std::optional<PendingAdmission> mPending;
};

//! What the scheduler drives, one typed operation per tick, however many ranks execute it. The
//! single-rank implementation wraps one stepper directly; the thread-parallel implementation
//! broadcasts each operation as a command every rank executes collectively. Either way the
//! scheduler sees the same vocabulary and the runtime never calls back into policy.
class SteppedExecution
{
public:
    virtual ~SteppedExecution() = default;

    //! The refs of every live resident, in slot order (identical on every rank by determinism).
    virtual std::vector<ResidentRef> residents() const = 0;

    //! Logical admission of @p request at @p originalIndex: intent building (validation,
    //! tokenization, encoder preprocess) plus the seat commit. The request itself is the
    //! cross-rank payload -- every rank rebuilds the intent deterministically, exactly the
    //! property the boundary relay relied on. The seated prefill is the next prefill tick.
    virtual AdmissionResult admit(LLMGenerationRequest const& request, int32_t originalIndex, RequestId requestId) = 0;

    virtual StepResult prefill(ImmutablePrefillBatch const& batch) = 0;
    virtual StepResult decode(ImmutableDecodeBatch const& batch) = 0;

    //! Convert a finished snapshot to a response (rank zero's tokenizer and stop trim).
    virtual LLMGenerationResponse materialize(
        BatchResult const& result, std::vector<std::string> const& stopStrings) const = 0;

    //! Everything handleRequest did after the loop. Call once, when no residents remain.
    virtual bool finish(LLMGenerationResponse& response) = 0;

    //! Abandon an incomplete session and release all runtime-owned request state.
    virtual void abort() noexcept = 0;
};

//! One request held open under the stepped control plane: the prepared request, the runtime-side
//! generation state, the session, and the stepper over it -- everything handleRequest kept on its
//! stack, owned across ticks instead. Created by RuntimeCoordinator::beginStepped(); the
//! single-rank SteppedExecution.
class SteppedRequest final : public SteppedExecution
{
public:
    //! Runs everything up to and including the founding prefill (LLMRankRuntime::beginGeneration)
    //! and wires the session and stepper over the result. Returns null on the same refusals that
    //! made handleRequest return false; throws where it threw. @p prepared is the coordinator's
    //! prepared request state and is owned here because the session references it across ticks.
    static std::unique_ptr<SteppedRequest> begin(LLMRankRuntime& runtime, LLMGenerationRequest prepared,
        RequestId requestId, cudaStream_t stream, LLMRankRuntime::TokenBroadcastFn tokenBroadcast = nullptr,
        int32_t parallelRank = -1);

    RuntimeStepper& stepper() noexcept
    {
        return *mStepper;
    }

    std::vector<ResidentRef> residents() const override;
    AdmissionResult admit(LLMGenerationRequest const& request, int32_t originalIndex, RequestId requestId) override;
    StepResult prefill(ImmutablePrefillBatch const& batch) override;
    StepResult decode(ImmutableDecodeBatch const& batch) override;
    LLMGenerationResponse materialize(
        BatchResult const& result, std::vector<std::string> const& stopStrings) const override;
    bool finish(LLMGenerationResponse& response) override;
    void abort() noexcept override;

    //! Stage-one intent building for a joiner (validation, tokenization, encoder preprocess),
    //! against this request's live session. Throws for requests the batch rejects outright.
    AdmissionIntent buildIntent(LLMGenerationRequest const& request, int32_t originalIndex, RequestId requestId);

    ~SteppedRequest() override;

private:
    SteppedRequest(LLMRankRuntime& runtime, LLMGenerationRequest prepared, cudaStream_t stream);

    LLMRankRuntime& mRuntime;
    LLMGenerationRequest mRequest;
    cudaStream_t mStream;
    LLMGenerationResponse mScratchResponse;
    std::unique_ptr<LLMRankRuntime::SteppedGeneration> mGeneration;
    std::unique_ptr<LLMRankRuntime::GenerationSession> mSession;
    std::unique_ptr<RuntimeStepper> mStepper;
};

} // namespace rt
} // namespace trt_edgellm
