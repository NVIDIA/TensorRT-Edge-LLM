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

#include "runtime/runtimeStepper.h"

#include "common/checkMacros.h"
#include "common/logger.h"

#include <exception>
#include <type_traits>

namespace trt_edgellm
{
namespace rt
{

RuntimeStepper::RuntimeStepper(LLMRankRuntime::GenerationSession& session)
    : mSession(session)
{
    DecodingInferenceContext const& context = mSession.mContext;
    for (int32_t slot = 0; slot < context.activeBatchSize; ++slot)
    {
        int32_t const originalIndex = context.batchIndexMapping[static_cast<size_t>(slot)];
        mRefs.emplace(originalIndex, context.residentRefs[static_cast<size_t>(slot)]);
        // Tokens the founding prefill produced inside beginGeneration are not yet reported; the
        // founding prefill tick claims them as its delta.
        size_t const generated = static_cast<size_t>(context.currentGenerateLengths[static_cast<size_t>(slot)]);
        mReported.emplace(originalIndex, context.tokenIds[static_cast<size_t>(slot)].size() - generated);
    }
}

AdmissionResult RuntimeStepper::admit(AdmissionIntent intent)
{
    ELLM_CHECK(!mPending.has_value(), "admit: the previous admission has not run its prefill tick yet.");
    ELLM_CHECK(intent.requestId != 0, "Stepped admission requires a nonzero request ID.");

    // The same invariant admitSequence holds at its entry, for the same reason: it must fire
    // before the lease, or a violation would strand leased pages.
    ELLM_CHECK(!mSession.mContext.hasGuidedDecoding, "A guided batch cannot take an admission.");

    int32_t const originalIndex = intent.seed.originalIndex;
    auto const [refEntry, insertedRef] = mRefs.emplace(originalIndex, ResidentRef{});
    ELLM_CHECK(insertedRef, "admit: original index is already resident.");
    decltype(mReported)::iterator reportedEntry;
    try
    {
        auto const inserted = mReported.emplace(originalIndex, 0);
        ELLM_CHECK(inserted.second, "admit: original index already has a reporting watermark.");
        reportedEntry = inserted.first;
    }
    catch (...)
    {
        mRefs.erase(refEntry);
        throw;
    }

    LLMRankRuntime::GenerationSession::AdmissionIntent inner{std::move(intent.seed), intent.requestId, std::nullopt};
    AdmitDecision reservation;
    try
    {
        reservation = mSession.reserveAdmission(inner);
    }
    catch (...)
    {
        mRefs.erase(refEntry);
        mReported.erase(reportedEntry);
        throw;
    }
    switch (reservation)
    {
    case AdmitDecision::kAdmitted: break;
    case AdmitDecision::kNoCapacity:
        mRefs.erase(refEntry);
        mReported.erase(reportedEntry);
        return {AdmissionResult::Status::kNoCapacity, {}, {}};
    case AdmitDecision::kFailed:
        mRefs.erase(refEntry);
        mReported.erase(reportedEntry);
        return {AdmissionResult::Status::kRejected, {}, "reservation failed"};
    }

    bool const leased = mSession.mManagedRequest != nullptr;
    static_assert(std::is_nothrow_move_constructible_v<PendingAdmission>);
    try
    {
        PendingAdmission pending;
        pending.seat = mSession.seatSlot(std::move(inner));
        pending.ref = mSession.mContext.residentRefs[static_cast<size_t>(pending.seat.slot)];
        pending.originalIndex = originalIndex;
        pending.leased = leased;
        refEntry->second = pending.ref;
        // The seat's working history (prompt from prefillStart on) is not generated output.
        reportedEntry->second = mSession.mContext.tokenIds[static_cast<size_t>(pending.seat.slot)].size();
        mPending.emplace(std::move(pending));
    }
    catch (std::exception const& error)
    {
        // seatSlot already retracted the lease on the throwing path; nothing is resident.
        mRefs.erase(refEntry);
        mReported.erase(reportedEntry);
        return {AdmissionResult::Status::kRejected, {}, error.what()};
    }
    catch (...)
    {
        mRefs.erase(refEntry);
        mReported.erase(reportedEntry);
        throw;
    }
    return {AdmissionResult::Status::kAdmitted, mPending->ref, {}};
}

StepResult RuntimeStepper::prefill(ImmutablePrefillBatch const& batch)
{
    StepResult result;
    if (mPending.has_value())
    {
        ELLM_CHECK(batch.target == mPending->ref, "prefill: the view does not name the pending admission.");
        PendingAdmission pending = std::move(*mPending);
        mPending.reset();
        int32_t const slot = pending.seat.slot;
        AdmitDecision const decision = mSession.prefillSeated(std::move(pending.seat));
        bool const terminal = slot >= 0 && slot < mSession.mContext.activeBatchSize
            && mSession.mContext.finishedStates[static_cast<size_t>(slot)] != 0;
        // A failed prefill and a lookahead token that is already terminal must be filed in this
        // tick. Leaving either resident until decode would execute a row that has no valid next
        // state, or advance once past EOS/length.
        result.ok = !terminal || mSession.performBatchEvictAndSnapshot();
        result.publishedPrefix = pending.leased && decision == AdmitDecision::kAdmitted;
    }
    else
    {
        ELLM_CHECK(!mRefs.empty(), "prefill: nothing is pending and the session has no residents to prime.");
        result.ok = mSession.primeFromPrefill();
    }
    finalizeResult(result);
    return result;
}

StepResult RuntimeStepper::decode(ImmutableDecodeBatch const& batch)
{
    ELLM_CHECK(!mPending.has_value(), "decode: the pending admission must run its prefill tick first.");
    ELLM_CHECK(batch.residents == residents(),
        "decode: the view must name every live resident in execution order; the prefill-first scheduler runs one "
        "cohort.");
    StepResult result;
    result.ok = mSession.advance();
    finalizeResult(result);
    return result;
}

std::vector<ResidentRef> RuntimeStepper::residents() const
{
    DecodingInferenceContext const& context = mSession.mContext;
    std::vector<ResidentRef> refs;
    refs.reserve(static_cast<size_t>(context.activeBatchSize));
    for (int32_t slot = 0; slot < context.activeBatchSize; ++slot)
    {
        auto const it = mRefs.find(context.batchIndexMapping[static_cast<size_t>(slot)]);
        ELLM_CHECK(it != mRefs.end(), "residents: a live slot has no ref; the table is stale.");
        refs.push_back(it->second);
    }
    return refs;
}

void RuntimeStepper::finalizeResult(StepResult& result)
{
    DecodingInferenceContext& context = mSession.mContext;
    result.stepId = context.scheduledStep.id;
    result.participants.reserve(context.scheduledStep.sequences.size());
    result.execution.reserve(context.scheduledStep.sequences.size());
    for (ScheduledSequence const& sequence : context.scheduledStep.sequences)
    {
        result.participants.push_back(sequence.identity);
        result.execution.push_back(ScheduledSequenceDescriptor{sequence.identity, sequence.work, sequence.queryLength,
            sequence.pastLength, sequence.selectLastTokenLogits});
    }
    if (!context.scheduledStep.sequences.empty())
    {
        result.work = context.scheduledStep.sequences.front().work;
    }

    // Deltas: tokens a surviving resident holds beyond its reported watermark. An evicted
    // resident's final tokens travel inside its BatchResult snapshot instead.
    for (int32_t slot = 0; slot < context.activeBatchSize; ++slot)
    {
        int32_t const originalIndex = context.batchIndexMapping[static_cast<size_t>(slot)];
        auto const reported = mReported.find(originalIndex);
        auto const ref = mRefs.find(originalIndex);
        ELLM_CHECK(reported != mReported.end() && ref != mRefs.end(), "finalizeResult: a live slot has no ref.");
        auto const& tokens = context.tokenIds[static_cast<size_t>(slot)];
        if (tokens.size() > reported->second)
        {
            TokenDelta delta;
            delta.tokenIds.assign(tokens.begin() + static_cast<ptrdiff_t>(reported->second), tokens.end());
            result.deltas.emplace_back(ref->second, std::move(delta));
            reported->second = tokens.size();
        }
    }

    // Finished: everything the operation's eviction filed. Releasing the resident slot makes
    // the finished ref part of this commit unit: from the caller's next operation on, the ref is
    // provably stale. Dense-row compaction needs no reporting because refs name physical rows.
    for (auto& [originalIndex, batchResult] : mSession.takeCompletedAtOrAbove(0))
    {
        auto const it = mRefs.find(originalIndex);
        if (it == mRefs.end())
        {
            LOG_WARNING("Stepper: a result for original index %d has no ref; dropping it.", originalIndex);
            continue;
        }
        result.finished.emplace_back(it->second, std::move(batchResult));
        mRefs.erase(it);
        mReported.erase(originalIndex);
    }
}

std::unique_ptr<SteppedRequest> SteppedRequest::begin(LLMRankRuntime& runtime, LLMGenerationRequest prepared,
    RequestId requestId, cudaStream_t stream, LLMRankRuntime::TokenBroadcastFn tokenBroadcast, int32_t parallelRank)
{
    std::unique_ptr<SteppedRequest> stepped(new SteppedRequest(runtime, std::move(prepared), stream));
    stepped->mGeneration = runtime.beginGeneration(stepped->mRequest, stepped->mScratchResponse, stream,
        /*outputThinkerEmbeddings=*/false, std::move(tokenBroadcast), parallelRank, requestId);
    if (stepped->mGeneration == nullptr)
    {
        return nullptr;
    }
    // boundarySchedulingActive: the stepped plane always schedules, so the cancellation consensus
    // runs regardless of the founding request's channels -- same flag the hook path sets.
    stepped->mSession = std::make_unique<LLMRankRuntime::GenerationSession>(runtime, stepped->mGeneration->context,
        *stepped->mGeneration->strategy, stepped->mGeneration->managedRequest(), stepped->mRequest,
        stepped->mGeneration->kvHeadroom, stream, /*boundarySchedulingActive=*/true);
    stepped->mStepper = std::make_unique<RuntimeStepper>(*stepped->mSession);
    return stepped;
}

SteppedRequest::SteppedRequest(LLMRankRuntime& runtime, LLMGenerationRequest prepared, cudaStream_t stream)
    : mRuntime(runtime)
    , mRequest(std::move(prepared))
    , mStream(stream)
{
}

SteppedRequest::~SteppedRequest() = default;

AdmissionIntent SteppedRequest::buildIntent(
    LLMGenerationRequest const& request, int32_t originalIndex, RequestId requestId)
{
    AdmissionIntent intent;
    intent.requestId = requestId;
    intent.seed = mSession->buildAdmissionIntent(request, originalIndex, requestId).seed;
    return intent;
}

std::vector<ResidentRef> SteppedRequest::residents() const
{
    return mStepper->residents();
}

AdmissionResult SteppedRequest::admit(LLMGenerationRequest const& request, int32_t originalIndex, RequestId requestId)
{
    return mStepper->admit(buildIntent(request, originalIndex, requestId));
}

StepResult SteppedRequest::prefill(ImmutablePrefillBatch const& batch)
{
    return mStepper->prefill(batch);
}

StepResult SteppedRequest::decode(ImmutableDecodeBatch const& batch)
{
    return mStepper->decode(batch);
}

LLMGenerationResponse SteppedRequest::materialize(
    BatchResult const& result, std::vector<std::string> const& stopStrings) const
{
    return mSession->materializeResult(result, stopStrings);
}

bool SteppedRequest::finish(LLMGenerationResponse& response)
{
    // The session must be gone before finishGeneration tears the request down: it references the
    // context and the strategy the teardown closes out.
    mStepper.reset();
    mSession.reset();
    return mRuntime.finishGeneration(*mGeneration, mRequest, response, mStream);
}

void SteppedRequest::abort() noexcept
{
    mStepper.reset();
    mSession.reset();
    mGeneration.reset();
}

} // namespace rt
} // namespace trt_edgellm
