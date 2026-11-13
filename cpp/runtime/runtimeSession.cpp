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

#include "runtime/runtimeSession.h"

#include "common/checkMacros.h"
#include "common/logger.h"
#include "profiling/timer.h"
#include "runtime/contextCacheRequest.h"
#include "runtime/debug/layerDebugger.h"
#include "runtime/deviceFailure.h"
#include "runtime/exec/engineExecutor.h"
#include "runtime/managedKVCacheRequest.h"
#include "runtime/streaming.h"
#include "sampler/sampling.h"

#include <algorithm>
#include <exception>
#include <stdexcept>
#include <utility>

namespace trt_edgellm
{
namespace rt
{

RankRuntimeSession::RankRuntimeSession(LLMRankRuntime& runtime, cudaStream_t stream, bool allowCompactExecution)
    : mRuntime(runtime)
    , mStream(stream)
    , mGuard(runtime.mHandleRequestInProgress)
    , mAllowCompact(allowCompactExecution)
{
}

RankRuntimeSession::~RankRuntimeSession()
{
    // Few-layer-validation debug: the dump covers the session's whole life, so it is written here.
    if (mContext.layerDebugger)
    {
        try
        {
            mContext.layerDebugger->flush(mStream);
        }
        catch (std::exception const& error)
        {
            LOG_ERROR("Layer debugger flush failed at session close: %s", error.what());
        }
    }
    if (mContext.activeBatchSize != 0)
    {
        // The caller already owes every resident an outcome (a session is closed with residents
        // only after a failed step or a cancelling shutdown). The members below release what the
        // residents held: the finalizer closes their channels, and dropping the context-cache
        // request returns their leases through the coordinator's abandon path.
        LOG_WARNING("Runtime session closed with %d resident sequence(s).", mContext.activeBatchSize);
    }
    // Member destruction order does the rest: the generation session first (it un-installs the
    // stop predicate the context holds), then the stream finalizer, then the cache request, then
    // the context, and last the latch.
}

bool RankRuntimeSession::open()
{
    // Nothing in a session is rank-parallel: the actor's thread owns the stream, and streaming and
    // callbacks stay on it.
    mRuntime.mTokenBroadcast = nullptr;
    mRuntime.mVisionGather = nullptr;
    mRuntime.mParallelRank = -1;
    if (mRuntime.mDecodingRuntimeContext)
    {
        mRuntime.mDecodingRuntimeContext->tokenBroadcast = nullptr;
        mRuntime.mDecodingRuntimeContext->parallelRank = -1;
    }

    // The strategy is a fill's, chosen by latchBatchWide for the sequence that enters the empty batch;
    // until then the session holds the deployment's default decoder.
    mStrategy = &mRuntime.mDecoderRegistry->defaultDecoder();
    mKvHeadroom = mStrategy->kvHeadroom().plain;
    ELLM_CHECK(mKvHeadroom.baseExtraTokens > 0 && mKvHeadroom.draftExtraTokens == 0,
        "The session's decoder returned invalid KV headroom");

    LLMEngineConfig const& base = mRuntime.mDeployment.base;
    // Zero sequences: every resident arrives through admit() carrying its own sampling parameters,
    // seed, generate budget and logprobs width.
    mContext.initialize(/*batchSize=*/0, /*maxGenLength=*/0, std::nullopt, rt::OptionalInputTensors{},
        /*loraName=*/std::string{}, mStream, base.recurrentPoolRows);
    mContext.logprobsStepOvershoot = mRuntime.mDeployment.maxAcceptedTokensPerRound() - 1;
    // Every cache manager's batch starts empty with the context's; each seat appends its row to the
    // ones its strategy drives (GenerationSession::reserveAdmission) and each eviction compacts
    // them, so they stay the context's size.
    for (auto& cacheManager : mRuntime.mSharedResources->cacheManagers)
    {
        cacheManager->setActiveBatchSize(0);
    }
    // Nothing a previous session left in the per-slot recurrent and convolution state reaches
    // this one: a session that closed with residents (a failed one) evicted none of them, and the
    // admission clears only the seat it takes. A fresh session starts from a cleared pool.
    for (auto& cacheManager : mRuntime.mSharedResources->cacheManagers)
    {
        for (int32_t slot = 0; slot < cacheManager->getConfig().maxBatchSize; ++slot)
        {
            cacheManager->clearResidentSlot(slot, mStream);
        }
    }
    // Runtime-owned per-request state is not the session's: a grammar compiled for a resident of
    // a session that failed would otherwise be found by this session's first eviction and applied
    // to whichever sequence then holds its row.
    mRuntime.mGuidedDecoder.reset();
    mContext.initializeRaggedScratch(RaggedEngineContract{TokenLayoutBackend::kEntryPaddedCompatibility,
        base.maxNumSequences, base.maxQueryLength, base.maxPhysicalTokens, base.recurrentPoolRows,
        /* mixedStepSupported = */ false});
    mContext.layerDebugger = LayerDebugger::fromEnv();

    // Per-request LoRA is refused at admission, so the session runs the base weights for its life.
    if (!mRuntime.selectLoraWeights(mContext.loraWeightsName))
    {
        return false;
    }

    if (mRuntime.mContextCache == nullptr)
    {
        // Without a cache the page tables are identity for the session's life: slot b owns its
        // static page range, and a slot's state is cleared when its sequence retires. Set once
        // here; nothing on the session path mutates them afterwards.
        for (auto& pageTable : mRuntime.mSharedResources->kvPageTables)
        {
            if (!pageTable->isIdentity())
            {
                pageTable->setIdentity();
                pageTable->upload(mStream);
            }
        }
    }

    // Resolved once per runtime; see LLMRankRuntime::mCompactCapability for what replaces it.
    if (mAllowCompact && !mRuntime.mCompactCapability.has_value())
    {
        mRuntime.mCompactCapability = mRuntime.resolveCompactCapability(mContext);
    }
    if (mAllowCompact && mRuntime.mCompactCapability->compact)
    {
        mContext.initializeCompactScratch(RaggedEngineContract{TokenLayoutBackend::kNativeCompactRagged,
            base.maxNumSequences, base.maxQueryLength, base.maxPhysicalTokens, base.recurrentPoolRows,
            /* mixedStepSupported = */ true});
    }
    mCompact = mAllowCompact && mRuntime.mCompactCapability->compact && mRuntime.prepareCompactExecution(mStream);
    if (mCompact)
    {
        size_t const maxSequences = static_cast<size_t>(base.maxNumSequences);
        mCompactPlanRows.reserve(maxSequences);
        mCompactOrder.reserve(maxSequences);
        mCompletingRefs.reserve(maxSequences);
        mDecodeOrder.reserve(maxSequences);
    }

    if (mRuntime.mContextCache != nullptr)
    {
        if (mRuntime.mVisualPruner != nullptr)
        {
            LOG_ERROR("Visual-token pruning is not supported with managed context reuse.");
            return false;
        }
        // The session is one batch for its life, so it holds one cache batch for its life, opened
        // with no sequence: each resident leases its pages at admission, under its own cache
        // policies, and returns them at eviction.
        // The previous session's request was released when it closed (ContextCacheCoordinator::
        // abandon, from the request's destructor); one still open here would hold that session's
        // leases beside this one's for the life of the engine. The coordinator does not refuse a
        // second request, so the invariant is named here.
        ELLM_CHECK(mRuntime.mContextCache->activeRequests() == 0,
            "A runtime session opens on a context cache with no request in flight; the previous session's request "
            "was never released.");
        if (!openCacheRequest(*mStrategy))
        {
            return false;
        }
    }

    mStreamFinalizer.emplace(mContext, *mRuntime.mTokenizer);
    // The cancellation consensus runs on every step: a resident's channel may be cancelled at any
    // time, and no single request's channels could decide this for the whole session.
    mSession = std::make_unique<LLMRankRuntime::GenerationSession>(mRuntime, mContext, *mStrategy,
        mManagedRequest.has_value() ? &*mManagedRequest : nullptr, mKvHeadroom, mKvHeadroom, mStream,
        /*synchronizeCancellation=*/true, /*actionSlots=*/std::vector<bool>{});
    return true;
}

bool RankRuntimeSession::openCacheRequest(DecodingStrategy const& strategy)
{
    ContextCacheBatchAdmission admission;
    admission.speculativeRequest = strategy.isSpeculative();
    admission.tokenStateContract = strategy.tokenStateContract();
    std::optional<ManagedKVCacheRequest> opened
        = ManagedKVCacheRequest::begin(*mRuntime.mContextCache, admission, strategy.kvHeadroom().plain, mStream);
    if (!opened.has_value())
    {
        return false;
    }
    mManagedRequest.emplace(std::move(*opened));
    mCacheRequestSpeculative = admission.speculativeRequest;
    mCacheRequestContract = admission.tokenStateContract;
    // Moves the request into execution. With nothing admitted it binds no page-table row.
    return mManagedRequest->preparePrefill();
}

RankRuntimeSession::BatchWideParameters RankRuntimeSession::batchWideParametersFor(
    LLMGenerationRequest const& request) const
{
    // The decoder is chosen for the request exactly as the blocking path chooses it; it serves the
    // whole batch, so it is batch-wide like the thinking mode and the proposal path.
    return BatchWideParameters{
        request.enableThinking, request.proposalSampling, &mRuntime.mDecoderRegistry->select(request)};
}

int32_t RankRuntimeSession::generateBudgetFor(
    LLMGenerationRequest const& request, int32_t promptLength, DecodingStrategy const& strategy)
{
    if (request.maxGenerateLength <= 0)
    {
        throw std::runtime_error(
            "maxGenerateLength must be positive; got " + std::to_string(request.maxGenerateLength) + ".");
    }
    // The budget is what the KV pool holds behind this prompt alone; a prompt that leaves no room is
    // a permanent refusal, like one that is too long.
    LLMEngineConfig const& base = mRuntime.mDeployment.base;
    DecodingKvHeadroom const headroom = strategy.kvHeadroom().plain;
    int32_t const requested = static_cast<int32_t>(request.maxGenerateLength);
    int32_t budget = clampMaxGenerateLengthForKVCapacity(
        {promptLength}, requested, base.maxKVCacheCapacity, headroom.baseExtraTokens);
    // A speculative decoder's draft cache bounds it too.
    if (headroom.draftExtraTokens > 0)
    {
        budget = clampMaxGenerateLengthForKVCapacity(
            {promptLength}, budget, mRuntime.mDeployment.draft->maxKVCacheCapacity, headroom.draftExtraTokens);
    }
    if (budget <= 0)
    {
        throw std::runtime_error("Insufficient KV cache capacity for generation for this request.");
    }
    if (budget != requested)
    {
        // Every admission is checked, and a server's clients commonly ask for more than the KV holds.
        if (!mWarnedGenerateBudget)
        {
            mWarnedGenerateBudget = true;
            LOG_WARNING(
                "Reduce max generation length to %d (the KV capacity behind a %d-token prompt); later reductions "
                "in this session are logged at DEBUG.",
                budget, promptLength);
        }
        else
        {
            LOG_DEBUG("Reduce max generation length to %d", budget);
        }
    }
    return budget;
}

void RankRuntimeSession::latchBatchWide(BatchWideParameters const& parameters)
{
    ELLM_CHECK(parameters.strategy != nullptr, "latchBatchWide: the batch names no decoder.");
    if (parameters.strategy != mStrategy)
    {
        DecodingKvHeadroom const headroom = parameters.strategy->kvHeadroom().plain;
        ELLM_CHECK(headroom.baseExtraTokens > 0 && headroom.draftExtraTokens >= 0
                && (headroom.draftExtraTokens == 0 || mRuntime.mDeployment.draft.has_value()),
            "The batch's decoder returned invalid KV headroom");
        mSession->rebindStrategy(*parameters.strategy);
        mStrategy = parameters.strategy;
        mKvHeadroom = headroom;
    }
    // The cache batch holds no sequence between fills; one opened for the other kind of decoder is replaced, in
    // place, so the generation session's pointer to it stays valid.
    if (mManagedRequest.has_value()
        && (mCacheRequestSpeculative != mStrategy->isSpeculative()
            || mCacheRequestContract != mStrategy->tokenStateContract()))
    {
        ELLM_CHECK(mContext.activeBatchSize == 0, "latchBatchWide: the cache batch is replaced only between fills.");
        if (!mManagedRequest->finish())
        {
            throw SessionLostError("the context-cache batch could not be closed to change its decoder kind");
        }
        mManagedRequest.reset();
        if (!openCacheRequest(*mStrategy))
        {
            throw SessionLostError("the context-cache batch could not be reopened for the batch's decoder");
        }
    }
    // A fill of the session is the blocking path's unit of generation: its decode rounds, which
    // the token callback reports as generationStep and a result as actualIterations, count from
    // zero like a request's do, not from where the previous fill left off.
    mContext.generationRound = 0;
    mContext.enableThinking = parameters.enableThinking;
    mContext.proposalSampling = parameters.proposalSampling;
    mBatchWide = parameters;
}

void RankRuntimeSession::checkBatchWide(BatchWideParameters const& parameters) const
{
    ELLM_CHECK(sameBatchWide(parameters),
        "admit: the request's batch-wide settings differ from the residents'; the caller's compatibility gate must "
        "keep them equal.");
}

bool RankRuntimeSession::sameBatchWide(BatchWideParameters const& parameters) const noexcept
{
    return parameters.enableThinking == mBatchWide.enableThinking
        && parameters.proposalSampling == mBatchWide.proposalSampling && parameters.strategy == mBatchWide.strategy;
}

AdmissionResult RankRuntimeSession::admit(
    LLMGenerationRequest const& request, RequestId requestId, std::shared_ptr<std::atomic<bool> const> cancelToken)
{
    ELLM_CHECK(requestId != 0, "admit: a nonzero request ID is required.");
    ELLM_CHECK(mCompact || mPending.empty(), "admit: the previous admission has not run its context step yet.");
    ELLM_CHECK(
        !mExclusiveResident, "admit: a resident holds the session exclusively; the caller must wait for it to leave.");

    // Profiling keeps the blocking path's unit of work: a profiling request begins when a sequence
    // enters the empty session and lasts until the session drains; a sequence admitted beside
    // residents is timed under the running request, as a joiner's prefill always was. Every step
    // reselects this runtime's timer, so a capture on another runtime sharing this thread does
    // not take the samples.
    if (idle())
    {
        mRuntime.selectProfilingTimer();
        gTimer.beginRequest();
    }

    // What the session never serves, however the deployment is configured. Permanent refusals,
    // so they are invalid arguments rather than a transient status.
    if (request.requests.size() != 1)
    {
        throw std::invalid_argument("A runtime session admits exactly one sequence per request.");
    }
    if (!request.loraWeightsName.empty())
    {
        throw std::invalid_argument("A runtime session does not serve per-request LoRA.");
    }
    if (request.generateAudio)
    {
        throw std::invalid_argument("A runtime session does not serve speech output; use the blocking path.");
    }
    if (request.requests.front().pastTrajectory.has_value())
    {
        throw std::invalid_argument("A runtime session does not serve trajectory execution; use the blocking path.");
    }
    if (!mRuntime.validateRequestConfig(request) || !validateStreamingSubmission(request))
    {
        throw std::runtime_error("The request failed validation; see the runtime log for the reason.");
    }

    bool const guided = hasGuidedDecoding(request);
    ELLM_CHECK(!guided || idle(), "admit: a grammar-constrained request is admitted only into an empty session.");

    BatchWideParameters const batchWide = batchWideParametersFor(request);
    if (!idle())
    {
        checkBatchWide(batchWide);
    }

    // Stage one: validation, templating / tokenization, the encoder preprocess. Touches no
    // resident state, so a throw here leaves the session exactly as it was -- unless it leaves the
    // session's stream in error: the encoder work runs on that stream, and the next step would fail
    // on it whichever request was blamed, so that is the loss of the session.
    // A compact session runs a media request's encoders only once the reservation holds; a
    // grammar-constrained sequence runs single-cohort steps, which take the encoder outputs from the intent.
    LLMRankRuntime::GenerationSession::AdmissionIntent intent = [&]() {
        try
        {
            return mSession->buildAdmissionIntent(request, /*originalIndex=*/0, requestId,
                /*deferMediaEncoders=*/mCompact && !guided, *batchWide.strategy);
        }
        catch (std::exception const& error)
        {
            if (!streamInError(mStream))
            {
                throw;
            }
            throw SessionLostError(
                std::string("preprocessing the admission left the session's stream in error: ") + error.what());
        }
    }();
    intent.seed.cancelToken = std::move(cancelToken);
    // The blocking path's rule for a speculative prompt checkpoint on a Hybrid+spec deployment, applied to the
    // sequence's resolved policy: the one the context step decides its hybrid endpoint reuse from.
    bool const sequenceCacheEnabled
        = mManagedRequest.has_value() && intent.lookupPolicy != ContextCacheLookupPolicy::kBypass;
    if (shouldUseHybridSpecEndpointReuse(batchWide.strategy->kind(), mRuntime.mHybridSpecContextReuseDeployment,
            sequenceCacheEnabled, sequenceCacheEnabled)
        && (request.recurrentCaptureInterval != 0
            || request.contextCacheCommitPolicy != ContextCacheCommitPolicy::kPrefillStateOnly))
    {
        throw std::invalid_argument(
            "Hybrid speculative context reuse requires endpoint-only capture and the PREFILL_STATE_ONLY commit "
            "policy.");
    }
    int32_t const promptLength = static_cast<int32_t>(intent.seed.promptTokenIds.size());
    LLMEngineConfig const& base = mRuntime.mDeployment.base;
    // A whole prompt is bounded by KV capacity; the engine's input limit bounds only what the
    // context step computes, known once the lease has fixed prefillStart (below). The blocking
    // path's two checks, in its order. The marker is what the server maps to HTTP 413.
    if (promptLength > base.maxKVCacheCapacity)
    {
        LOG_ERROR(
            "Input length (%d) exceeds engine KV cache capacity (%d). Rebuild the engine with a larger "
            "--maxKVCacheCapacity.",
            promptLength, base.maxKVCacheCapacity);
        throw std::runtime_error("EDGELLM_INPUT_TOO_LONG: input length " + std::to_string(promptLength)
            + " exceeds engine max_kv_cache_capacity " + std::to_string(base.maxKVCacheCapacity)
            + " (rebuild engine with a larger --maxKVCacheCapacity)");
    }
    // A token callback is one per context, so a callback-bearing request runs as the only
    // resident: installed once seated, uninstalled when the session empties (finalizeResult).
    bool const callback = request.onTokenGenerated.has_value();
    bool const exclusive = guided || callback || intent.seed.requiresExclusiveSession;
    bool const singleCohort = intent.seed.requiresExclusiveSession;
    ELLM_CHECK(!exclusive || idle(), "admit: this request is admitted only into an empty session.");

    intent.seed.maxGenerateLength = generateBudgetFor(request, promptLength, *batchWide.strategy);
    // A speculative decoder that verifies greedily runs greedy whatever the request asked, and a
    // lossless one keeps the request's sampling -- beginGeneration's rule.
    if (batchWide.strategy->isSpeculative() && !batchWide.strategy->capabilities().supportsLosslessSampling)
    {
        intent.seed.sampling = RowSamplingParams{};
    }
    if (idle())
    {
        latchBatchWide(batchWide);
    }

    // A grammar is compiled before anything is reserved: the constrained sequence is alone in the
    // session, so it will own slot zero, and a compile failure then rejects with nothing to unwind.
    if (guided)
    {
        std::vector<int32_t> const reasoningStartIds = reasoningStartMarkers(*mRuntime.mTokenizer);
        std::vector<int32_t> const reasoningEndIds = reasoningEndMarkers(*mRuntime.mTokenizer);
        intent.seed.guidedReasoningEnded = reasoningClosedInPrompt(
            intent.seed.promptTokenIds, reasoningStartIds, reasoningEndIds, request.enableThinking);
        auto const& guide = *request.requests.front().guidedDecoding;
        std::string failReason;
        mRuntime.mGuidedDecoder.reset();
        if (!mRuntime.mGuidedDecoder.prepareSlot(/*slot=*/0, guide, failReason))
        {
            throw std::runtime_error(
                std::string("Failed to compile the ") + guideTypeName(guide.type) + " guide: " + failReason);
        }
    }

    // Stage two: the reservation -- resident slot, and the page lease when reuse is on. A refusal
    // leaves the session untouched. A throw is a device failure inside the lease path, after which
    // the cache coordinator's state is unknown: it is reported as the loss of the session, not as
    // a refusal of this request, so the caller does not keep scheduling onto it (as an executor
    // whose forward threw is not driven further in TensorRT-LLM or vLLM).
    AdmitDecision reservation;
    try
    {
        reservation = mSession->reserveAdmission(intent);
    }
    catch (std::exception const& error)
    {
        mRuntime.mGuidedDecoder.reset();
        throw SessionLostError(
            std::string("the admission reservation failed and left the session's state unknown: ") + error.what());
    }
    catch (...)
    {
        mRuntime.mGuidedDecoder.reset();
        throw SessionLostError(
            "the admission reservation failed with an unknown error and left the session's state unknown");
    }
    switch (reservation)
    {
    case AdmitDecision::kAdmitted:
    case AdmitDecision::kOverlongPrefill: break;
    case AdmitDecision::kNoCapacity:
        mRuntime.mGuidedDecoder.reset();
        return AdmissionResult{AdmissionResult::Status::kNoCapacity, {}, {}, promptLength, 0};
    case AdmitDecision::kFailed:
        mRuntime.mGuidedDecoder.reset();
        return AdmissionResult{AdmissionResult::Status::kRejected, {}, "reservation failed", promptLength, 0};
    }

    // Only the uncached part of the prompt reaches the engine, so the input limit is checked
    // against it, once the cache lookup has fixed where the context step starts (the blocking path
    // checks the same length). A context cache refuses it before leasing (kOverlongPrefill); without
    // one, the reservation is given back. Either way the request is refused with nothing resident.
    int32_t const contextLength = promptLength - intent.seed.prefillStart;
    if (reservation == AdmitDecision::kOverlongPrefill || contextLength > base.maxSupportedInputLength)
    {
        if (reservation == AdmitDecision::kAdmitted)
        {
            mSession->releaseReservation(intent);
        }
        mRuntime.mGuidedDecoder.reset();
        LOG_ERROR(
            "Prefill length after context reuse (%d) exceeds engine max input length (%d). Rebuild the engine with "
            "a larger --maxInputLen.",
            contextLength, base.maxSupportedInputLength);
        throw std::runtime_error("EDGELLM_INPUT_TOO_LONG: prefill length " + std::to_string(contextLength)
            + " exceeds engine max_input_len " + std::to_string(base.maxSupportedInputLength)
            + " (rebuild engine with a larger --maxInputLen)");
    }

    stageMediaOrRelease(intent);

    // The sequence's results file under its resident slot: unique among live sequences, and every
    // finished result is taken out at the step that files it, so the slot is free to be the key
    // again when a later sequence reuses it.
    intent.seed.originalIndex = intent.resident->slot;
    bool const leased = mManagedRequest.has_value();

    // Stage three: the seat commit. From appendSlot on, the sequence is resident; before it, a
    // failure retracts the lease and releases the slot, and nothing is resident -- so whatever
    // seatSlot throws is a refusal of this request, never a loss of the session.
    PendingAdmission pending;
    try
    {
        pending.seat = mSession->seatSlot(std::move(intent));
    }
    catch (std::exception const& error)
    {
        mRuntime.mGuidedDecoder.reset();
        return AdmissionResult{AdmissionResult::Status::kRejected, {}, error.what(), promptLength, 0};
    }
    catch (...)
    {
        mRuntime.mGuidedDecoder.reset();
        return AdmissionResult{AdmissionResult::Status::kRejected, {},
            "seating the sequence failed with an unknown error", promptLength, 0};
    }
    int32_t const row = pending.seat.row;
    pending.ref = mContext.residentRefs[static_cast<size_t>(row)];
    pending.leased = leased;
    pending.exclusive = exclusive;
    if (guided)
    {
        ELLM_CHECK(row == 0, "admit: a grammar-constrained sequence must own row zero.");
        mContext.hasGuidedDecoding = true;
    }
    mExclusiveResident = exclusive;
    mSingleCohortResident = singleCohort;
    if (callback)
    {
        mContext.onTokenGenerated = request.onTokenGenerated;
    }

    // The seat's working history (prompt from prefillStart on) is not generated output.
    mTracked[pending.ref.slot] = Tracked{pending.ref, mContext.tokenIds[static_cast<size_t>(row)].size()};
    int32_t const prefillStart = pending.seat.prefillStart;
    ResidentRef const ref = pending.ref;
    mPending.push_back(std::move(pending));
    return AdmissionResult{AdmissionResult::Status::kAdmitted, ref, {}, promptLength, prefillStart};
}

void RankRuntimeSession::stageMediaOrRelease(LLMRankRuntime::GenerationSession::AdmissionIntent& intent)
{
    std::exception_ptr failure;
    std::optional<std::string> deviceFailure;
    try
    {
        mSession->stageReservedMedia(intent);
        return;
    }
    catch (std::exception const& error)
    {
        failure = std::current_exception();
        if (failedOnDevice(error, mStream))
        {
            deviceFailure = error.what();
        }
    }
    catch (...)
    {
        failure = std::current_exception();
    }
    if (!deviceFailure.has_value())
    {
        mSession->releaseReservation(intent);
        mRuntime.mGuidedDecoder.reset();
        std::rethrow_exception(failure);
    }
    try
    {
        mSession->releaseReservation(intent);
    }
    catch (...)
    {
        // The device failure is the one to report.
    }
    mRuntime.mGuidedDecoder.reset();
    throw SessionLostError(
        "staging the admission's media failed on the device and left the session's state unknown: " + *deviceFailure);
}

std::vector<RankRuntimeSession::PendingAdmission>::iterator RankRuntimeSession::findPending(ResidentRef ref)
{
    return std::find_if(
        mPending.begin(), mPending.end(), [ref](PendingAdmission const& pending) { return pending.ref == ref; });
}

int32_t RankRuntimeSession::rowOf(ResidentRef ref) const noexcept
{
    for (int32_t row = 0; row < mContext.activeBatchSize; ++row)
    {
        if (mContext.residentRefs[static_cast<size_t>(row)] == ref)
        {
            return row;
        }
    }
    return -1;
}

StepResult RankRuntimeSession::execute(std::vector<ScheduledSequenceDescriptor> const& plan)
{
    ELLM_CHECK(!plan.empty(), "execute: a step names at least one sequence.");
    // Guided and single-cohort residents run alone and keep the single-cohort steps on a compact deployment.
    if (mCompact && !mContext.hasGuidedDecoding && !mSingleCohortResident)
    {
        bool const decodeAll = mPending.empty() && static_cast<int32_t>(plan.size()) == mContext.activeBatchSize
            && std::all_of(plan.begin(), plan.end(), [this](ScheduledSequenceDescriptor const& sequence) {
                   return sequence.work == SequenceWork::kDecode && rowOf(sequence.identity.resident) >= 0;
               });
        if (!decodeAll)
        {
            return executeCompact(plan);
        }
        // Every resident decodes: the phase-3 step and its captured graphs (a duplicate leaves some
        // resident out and fails the check there).
        return executeDecodeInResidentOrder(plan);
    }
    SequenceWork const work = plan.front().work;
    ELLM_CHECK(std::all_of(plan.begin(), plan.end(),
                   [work](ScheduledSequenceDescriptor const& sequence) { return sequence.work == work; }),
        "execute: a step runs one kind of work; mixed context/decode steps are not supported yet.");
    if (work == SequenceWork::kContext)
    {
        ELLM_CHECK(plan.size() == 1, "execute: a context step covers the one pending admission.");
        return executeContext(plan.front());
    }
    return executeDecode(plan);
}

StepResult RankRuntimeSession::executeContext(ScheduledSequenceDescriptor const& sequence)
{
    ELLM_CHECK(mPending.size() == 1, "execute: a context step needs a pending admission.");
    // Checked before the admission is consumed: a plan the session refuses must leave the pending
    // admission, and everything else, exactly as it was.
    {
        PendingAdmission const& pending = mPending.front();
        int32_t const pendingRow = pending.seat.row;
        ELLM_CHECK(
            pendingRow >= 0 && pendingRow < mContext.activeBatchSize, "execute: the pending seat is not a live row.");
        ELLM_CHECK(sequence.identity.resident == pending.ref
                && sequence.identity.requestId == mContext.requestIds[static_cast<size_t>(pendingRow)],
            "execute: the context step does not name the pending admission.");
        ELLM_CHECK(sequence.pastLength == mContext.committedLengths[static_cast<size_t>(pendingRow)]
                && sequence.queryLength
                    == static_cast<int32_t>(mContext.tokenIds[static_cast<size_t>(pendingRow)].size())
                && sequence.selectLastTokenLogits,
            "execute: the context step's token range disagrees with the sequence's ledger (it covers the whole "
            "uncommitted prompt and samples one token).");
    }
    PendingAdmission pending = std::move(mPending.front());
    mPending.clear();
    int32_t const row = pending.seat.row;

    StepResult result;
    result.work = SequenceWork::kContext;
    // A cancel that landed between the admission and this step is honoured before the prefill is
    // paid for -- a request aborted before its first iteration never reaches the model in
    // TensorRT-LLM or vLLM either. The seat is retired unprefilled and filed by this step's
    // eviction; no step ran, so the record is empty.
    applyCancellationToFinishState(mContext, row);
    if (mContext.finishedStates[static_cast<size_t>(row)] != 0)
    {
        mSession->retireSeatUnprefilled(row);
        result.ok = mSession->performBatchEvictAndSnapshot();
        finalizeResult(result, {sequence}, /*executed=*/false);
        return result;
    }
    AdmitDecision decision = AdmitDecision::kFailed;
    try
    {
        decision = mSession->prefillSeated(std::move(pending.seat));
    }
    catch (std::exception const& error)
    {
        // prefillSeated closes a failed admission before it rethrows (failAdmission: the row is
        // terminal with kError and its lease is aborted). When it did, the failure is this
        // sequence's alone: the eviction below files it and, by synchronizing the stream, is also
        // the check that the device survived -- a sticky device error surfaces there and fails
        // the step, and the session with it, as before. A throw that left the row live is not a
        // closed admission and keeps propagating.
        if (mContext.finishedStates[static_cast<size_t>(row)] == 0
            || mContext.slotStreams[static_cast<size_t>(row)].terminalReason != FinishReason::kError)
        {
            throw;
        }
        LOG_ERROR("Request %s: the seated prefill failed, the sequence is retired and the session goes on: %s",
            std::to_string(sequence.identity.requestId).c_str(), error.what());
    }
    bool const terminal = mContext.finishedStates[static_cast<size_t>(row)] != 0;
    // A failed prefill and a lookahead token that is already terminal must be filed in this step.
    // Leaving either resident until decode would execute a row that has no valid next state, or
    // advance once past EOS/length. (The lookahead token itself, and the finish of its channel,
    // already went out through prefillSeated's per-token pipeline.)
    result.ok = !terminal || mSession->performBatchEvictAndSnapshot();
    result.publishedPrefix = pending.leased && decision == AdmitDecision::kAdmitted;
    finalizeResult(result, {sequence}, /*executed=*/decision == AdmitDecision::kAdmitted);
    return result;
}

StepResult RankRuntimeSession::executeDecode(std::vector<ScheduledSequenceDescriptor> const& plan)
{
    ELLM_CHECK(mPending.empty(), "execute: the pending admission must run its context step first.");
    ELLM_CHECK(static_cast<int32_t>(plan.size()) == mContext.activeBatchSize,
        "execute: a decode step names every resident, in resident order.");
    for (size_t row = 0; row < plan.size(); ++row)
    {
        ScheduledSequenceDescriptor const& sequence = plan[row];
        ELLM_CHECK(sequence.identity.requestId == mContext.requestIds[row]
                && sequence.identity.resident == mContext.residentRefs[row],
            "execute: a decode step's sequences must be the residents, in resident order.");
        // The plan asks every resident for one decode round from its committed length. A speculative
        // round may commit more than the one position asked for; the step record reports what it
        // committed, and the caller's ledger advances by that.
        ELLM_CHECK(sequence.queryLength == 1 && sequence.pastLength == mContext.committedLengths[row]
                && sequence.selectLastTokenLogits,
            "execute: a decode step advances each resident by one decode round.");
    }

    StepResult result;
    result.work = SequenceWork::kDecode;
    uint64_t const graphLaunchesBefore = mRuntime.mBaseExecutor->graphLaunches();
    result.ok = mSession->advance();
    result.graphLaunched = mRuntime.mBaseExecutor->graphLaunches() != graphLaunchesBefore;
    finalizeResult(result, plan, /*executed=*/result.ok);
    return result;
}

StepResult RankRuntimeSession::executeCompact(std::vector<ScheduledSequenceDescriptor> const& plan)
{
    using CompactRow = LLMRankRuntime::GenerationSession::CompactRow;
    LLMEngineConfig const& base = mRuntime.mDeployment.base;
    ELLM_CHECK(static_cast<int32_t>(plan.size()) <= base.maxNumSequences,
        "execute: a compact step names more sequences than the engine runs at once.");
    // Checked in full before anything moves: a plan the session refuses leaves it exactly as it was.
    std::vector<int32_t>& rows = mCompactPlanRows;
    rows.clear();
    int64_t tokens = 0;
    bool anyContext = false;
    for (ScheduledSequenceDescriptor const& sequence : plan)
    {
        int32_t const row = rowOf(sequence.identity.resident);
        ELLM_CHECK(row >= 0 && sequence.identity.requestId == mContext.requestIds[static_cast<size_t>(row)],
            "execute: a compact step names a sequence the session does not hold.");
        ELLM_CHECK(
            std::find(rows.begin(), rows.end(), row) == rows.end(), "execute: a compact step names a sequence twice.");
        int32_t const committed = mContext.committedLengths[static_cast<size_t>(row)];
        ELLM_CHECK(sequence.pastLength == committed, "execute: a sequence's pastLength must be its committed length.");
        bool const promptIncomplete = findPending(sequence.identity.resident) != mPending.end();
        if (sequence.work == SequenceWork::kContext)
        {
            int32_t const remaining
                = static_cast<int32_t>(mContext.rawBatchedInputIds[static_cast<size_t>(row)].size()) - committed;
            ELLM_CHECK(promptIncomplete, "execute: a prompt chunk names a resident whose prompt is complete.");
            ELLM_CHECK(sequence.queryLength > 0 && sequence.queryLength <= remaining
                    && sequence.queryLength <= base.maxQueryLength,
                "execute: a prompt chunk covers 0 < queryLength <= the remaining prompt.");
            ELLM_CHECK(sequence.selectLastTokenLogits == (sequence.queryLength == remaining),
                "execute: a prompt chunk samples exactly when it ends the prompt.");
            anyContext = true;
        }
        else
        {
            ELLM_CHECK(sequence.work == SequenceWork::kDecode && !promptIncomplete && sequence.queryLength == 1
                    && sequence.selectLastTokenLogits,
                "execute: a decode row advances a resident whose prompt is complete by its one lookahead token.");
        }
        tokens += sequence.queryLength;
        rows.push_back(row);
    }
    ELLM_CHECK(tokens <= base.maxPhysicalTokens, "execute: a compact step exceeds the engine's token capacity.");

    StepResult result;
    result.work = anyContext ? SequenceWork::kContext : SequenceWork::kDecode;
    // Cancellation lands at the step boundary for every resident, named by the plan or not. A
    // prompt-incomplete one is retired without completing its prompt; any other leaves through this
    // step's eviction with its channel finished, as a decode step would leave it.
    applyCancellationToFinishStates(mContext);
    if (!mRuntime.synchronizeCancellationStates(mContext))
    {
        LOG_ERROR("Failed to synchronize cancellation state across parallel ranks.");
        finalizeResult(result, plan, /*executed=*/false);
        return result;
    }
    for (int32_t row = 0; row < mContext.activeBatchSize; ++row)
    {
        if (mContext.finishedStates[static_cast<size_t>(row)] == 0)
        {
            continue;
        }
        auto const pending = findPending(mContext.residentRefs[static_cast<size_t>(row)]);
        if (pending != mPending.end())
        {
            mSession->retireSeatUnprefilled(row);
            mPending.erase(pending);
        }
        else
        {
            emitChunk(mContext, row, *mRuntime.mTokenizer);
        }
    }

    std::vector<CompactRow>& ordered = mCompactOrder;
    ordered.clear();
    std::vector<ResidentRef>& completing = mCompletingRefs;
    completing.clear();
    for (SequenceWork const work : {SequenceWork::kContext, SequenceWork::kDecode})
    {
        for (size_t index = 0; index < plan.size(); ++index)
        {
            ScheduledSequenceDescriptor const& sequence = plan[index];
            int32_t const row = rows[index];
            if (sequence.work != work || mContext.finishedStates[static_cast<size_t>(row)] != 0)
            {
                continue;
            }
            bool firstChunk = false;
            if (work == SequenceWork::kContext)
            {
                auto const pending = findPending(sequence.identity.resident);
                firstChunk = !pending->started;
                pending->started = true;
                if (sequence.selectLastTokenLogits)
                {
                    completing.push_back(sequence.identity.resident);
                }
            }
            ordered.push_back(CompactRow{row, work, sequence.queryLength, firstChunk});
        }
    }
    if (ordered.empty())
    {
        result.ok = mSession->performBatchEvictAndSnapshot();
        finalizeResult(result, plan, /*executed=*/false);
        return result;
    }
    int32_t const liveResidents = static_cast<int32_t>(
        std::count(mContext.finishedStates.begin(), mContext.finishedStates.begin() + mContext.activeBatchSize, 0));
    bool const decodesEveryLiveResident = mPending.empty() && static_cast<int32_t>(ordered.size()) == liveResidents
        && std::all_of(
            ordered.begin(), ordered.end(), [](CompactRow const& row) { return row.work == SequenceWork::kDecode; });
    if (decodesEveryLiveResident)
    {
        return executeDecodeAfterCancellation(plan);
    }

    result.compact = true;
    result.ok = mSession->executeCompactStep(ordered, result.publishedPrefix, result.compactRefused);
    if (result.compactRefused)
    {
        result.failure
            = "a plugin refused compact execution, which is now disabled for this runtime; the "
              "request can be retried";
    }
    if (result.ok)
    {
        for (ResidentRef const ref : completing)
        {
            mPending.erase(findPending(ref));
        }
    }
    finalizeResult(result, plan, /*executed=*/result.ok);
    return result;
}

StepResult RankRuntimeSession::executeDecodeAfterCancellation(std::vector<ScheduledSequenceDescriptor> const& plan)
{
    // The cancelled residents are filed first, so the phase-3 step and its captured graphs cover
    // exactly the residents left.
    if (!mSession->performBatchEvictAndSnapshot())
    {
        StepResult failed;
        failed.work = SequenceWork::kDecode;
        finalizeResult(failed, plan, /*executed=*/false);
        return failed;
    }
    return executeDecodeInResidentOrder(plan);
}

StepResult RankRuntimeSession::executeDecodeInResidentOrder(std::vector<ScheduledSequenceDescriptor> const& plan)
{
    std::vector<ScheduledSequenceDescriptor>& decodes = mDecodeOrder;
    decodes.clear();
    for (ScheduledSequenceDescriptor const& sequence : plan)
    {
        if (sequence.work == SequenceWork::kDecode && rowOf(sequence.identity.resident) >= 0)
        {
            decodes.push_back(sequence);
        }
    }
    std::sort(decodes.begin(), decodes.end(),
        [this](ScheduledSequenceDescriptor const& left, ScheduledSequenceDescriptor const& right) {
            return rowOf(left.identity.resident) < rowOf(right.identity.resident);
        });
    StepResult result = executeDecode(decodes);
    result.participants.clear();
    for (ScheduledSequenceDescriptor const& sequence : plan)
    {
        result.participants.push_back(sequence.identity);
    }
    return result;
}

void RankRuntimeSession::finalizeResult(
    StepResult& result, std::vector<ScheduledSequenceDescriptor> const& plan, bool executed)
{
    result.participants.reserve(plan.size());
    for (ScheduledSequenceDescriptor const& sequence : plan)
    {
        result.participants.push_back(sequence.identity);
    }
    if (executed)
    {
        result.stepId = mContext.scheduledStep.id;
        result.execution.reserve(mContext.scheduledStep.sequences.size());
        for (ScheduledSequence const& sequence : mContext.scheduledStep.sequences)
        {
            result.execution.push_back(ScheduledSequenceDescriptor{sequence.identity, sequence.work,
                sequence.queryLength, sequence.pastLength, sequence.selectLastTokenLogits});
        }
    }

    // Deltas: tokens a surviving resident holds beyond its reported watermark. An evicted
    // resident's final tokens travel inside its BatchResult snapshot instead.
    for (int32_t row = 0; row < mContext.activeBatchSize; ++row)
    {
        ResidentRef const ref = mContext.residentRefs[static_cast<size_t>(row)];
        auto const tracked = mTracked.find(ref.slot);
        ELLM_CHECK(tracked != mTracked.end() && tracked->second.ref == ref,
            "finalizeResult: a live slot is not tracked; the session's ledger is stale.");
        auto const& tokens = mContext.tokenIds[static_cast<size_t>(row)];
        if (tokens.size() > tracked->second.reported)
        {
            TokenDelta delta;
            delta.tokenIds.assign(tokens.begin() + static_cast<ptrdiff_t>(tracked->second.reported), tokens.end());
            result.deltas.emplace_back(ref, std::move(delta));
            tracked->second.reported = tokens.size();
        }
    }

    // Finished: everything the step's eviction filed, keyed by the resident slot the sequence
    // owned. Releasing the slot made the finished ref part of this commit unit: from the caller's
    // next step on, the ref is provably stale.
    for (auto& [slot, snapshot] : mSession->takeCompletedAtOrAbove(0))
    {
        auto const tracked = mTracked.find(slot);
        ELLM_CHECK(tracked != mTracked.end(), "finalizeResult: a finished sequence was never tracked.");
        recordSequenceMetrics(snapshot);
        result.finished.emplace_back(tracked->second.ref, std::move(snapshot));
        mTracked.erase(tracked);
    }
    if (idle())
    {
        // An exclusive resident leaves an empty session behind by construction, and with it goes
        // the callback it installed.
        mExclusiveResident = false;
        mSingleCohortResident = false;
        mContext.onTokenGenerated.reset();
    }
}

void RankRuntimeSession::recordSequenceMetrics(BatchResult const& result) noexcept
{
    int32_t const rawPromptLength = static_cast<int32_t>(result.rawBatchedInputIds.size());
    int32_t const computedLength = result.effectivePrefillLength;
    mRuntime.mPrefillMetrics.recordRun(reusedPromptTokens(rawPromptLength, computedLength, result.prunedPrefillTokens),
        computedLength, result.prunedPrefillTokens);
    mRuntime.mGenerationMetrics.recordRun(result.generateLength);
}

LLMGenerationResponse RankRuntimeSession::materialize(
    BatchResult const& result, std::vector<std::string> const& stopStrings) const
{
    return mSession->materializeResult(result, stopStrings);
}

std::vector<ResidentRef> RankRuntimeSession::residents() const
{
    return std::vector<ResidentRef>(
        mContext.residentRefs.begin(), mContext.residentRefs.begin() + mContext.activeBatchSize);
}

void RankRuntimeSession::copyResidents(std::vector<ResidentRef>& out) const
{
    out.assign(mContext.residentRefs.begin(), mContext.residentRefs.begin() + mContext.activeBatchSize);
}

void RankRuntimeSession::releaseForReadmission(ResidentRef ref)
{
    int32_t const row = rowOf(ref);
    ELLM_CHECK(row >= 0 && findPending(ref) != mPending.end(),
        "releaseForReadmission: only a resident whose prompt has not completed can be admitted again.");
    // Unattached, the channel is skipped by the stream finalizer when the session closes.
    std::shared_ptr<StreamChannel>& channel = mContext.slotStreams[static_cast<size_t>(row)].channel;
    if (channel != nullptr)
    {
        detachStreamChannel(channel);
        channel.reset();
    }
}

bool RankRuntimeSession::idle() const noexcept
{
    return mContext.activeBatchSize == 0;
}

bool RankRuntimeSession::exclusiveResident() const noexcept
{
    return mExclusiveResident;
}

bool RankRuntimeSession::requiresExclusiveAdmission(LLMGenerationRequest const& request) const noexcept
{
    if (hasGuidedDecoding(request) || request.onTokenGenerated.has_value())
    {
        return true;
    }
    if (mCompact && mRuntime.mediaLayoutNeedsEncoder(request))
    {
        return true;
    }
    if (mRuntime.mVisualPruner == nullptr)
    {
        return false;
    }
    return std::any_of(request.requests.begin(), request.requests.end(), [](LLMGenerationRequest::Request const& item) {
        return !item.imageBuffers.empty() || !item.audioBuffers.empty();
    });
}

bool RankRuntimeSession::requiresUniformSampling(LLMGenerationRequest const& request) const noexcept
{
    // The default decoder samples each row with its own parameters; every speculative decoder reads one set per
    // batch (DecodingInferenceContext::uniformSamplingParams).
    return mRuntime.mDecoderRegistry->select(request).isSpeculative();
}

SessionCapabilities RankRuntimeSession::capabilities() const
{
    LLMEngineConfig const& base = mRuntime.mDeployment.base;
    SessionCapabilities capabilities;
    if (!mAllowCompact)
    {
        capabilities.reason = "compact execution disabled for this session";
    }
    else
    {
        // open() resolves the capability before a session that allows compact execution can open.
        ELLM_CHECK(mRuntime.mCompactCapability.has_value(), "the compact capability of an open session is unresolved");
        capabilities.compact = mRuntime.mCompactCapability->compact;
        capabilities.reason = mRuntime.mCompactCapability->reason;
    }
    capabilities.maxBatchSize = mRuntime.maxBatchSize();
    capabilities.maxInputLength = base.maxSupportedInputLength;
    return capabilities;
}

std::optional<ContextCacheMetrics> RankRuntimeSession::contextCacheMetrics() const noexcept
{
    return mRuntime.getContextCacheMetrics();
}

} // namespace rt
} // namespace trt_edgellm
