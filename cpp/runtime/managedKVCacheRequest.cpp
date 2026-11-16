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

#include "runtime/managedKVCacheRequest.h"

#include "common/checkMacros.h"
#include "common/logger.h"
#include "runtime/llmRuntimeUtils.h"
#include "runtime/state/decodingInferenceContext.h"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <utility>

namespace trt_edgellm
{
namespace rt
{
namespace
{

bool swaOperationSucceeded(SwaKVCacheStatus status, char const* operation)
{
    if (status == SwaKVCacheStatus::kOk)
    {
        return true;
    }
    LOG_ERROR("SWA KV-cache %s failed (%s).", operation,
        status == SwaKVCacheStatus::kPoisoned ? "manager poisoned" : "request failure");
    return false;
}

} // namespace

std::optional<ManagedKVCacheRequest> ManagedKVCacheRequest::begin(ContextCacheCoordinator* contextCache,
    BoundedSwaKVPageManager* swaPageManager, LLMGenerationRequest const& request,
    DecodingInferenceContext const& context, bool speculativeRequest, DecodingKvHeadroom const& headroom,
    DecodingTokenStateContract tokenStateContract, ContextCacheCommitPolicy commitPolicy,
    std::vector<int32_t> const& mediaTokenIds)
{
    ELLM_CHECK(contextCache != nullptr || swaPageManager != nullptr,
        "Managed KV cache request requires at least one lifecycle client");
    std::optional<ContextCacheRequest> contextRequest;
    if (contextCache != nullptr)
    {
        std::optional<ContextCacheRequest> admitted = ContextCacheRequest::begin(*contextCache, request, context,
            speculativeRequest, headroom, mediaTokenIds, tokenStateContract, commitPolicy);
        if (!admitted.has_value())
        {
            return std::nullopt;
        }
        contextRequest.emplace(std::move(*admitted));
    }

    std::optional<BoundedSwaKVPageManager::RequestHandle> swaRequest;
    if (swaPageManager != nullptr)
    {
        ELLM_CHECK(!speculativeRequest && tokenStateContract == DecodingTokenStateContract::kCommittedPlusLookahead
                && headroom.baseExtraTokens == 1 && headroom.draftExtraTokens == 0,
            "Bounded SWA KV cache supports only vanilla one-token decode");
        ELLM_CHECK(context.activeBatchSize > 0
                && context.rawBatchedInputIds.size() == static_cast<size_t>(context.activeBatchSize),
            "Bounded SWA admission requires one raw input per active sequence");
        BoundedSwaKVPageManager::BeginRequestResult admitted = contextRequest.has_value()
            ? swaPageManager->beginRequest(contextRequest->takeSwaStates(), context.stream)
            : swaPageManager->beginRequest(context.residentRefs, context.stream);
        if (!swaOperationSucceeded(admitted.status, "admission") || !admitted.request.has_value())
        {
            return std::nullopt;
        }
        swaRequest.emplace(std::move(*admitted.request));
    }
    return ManagedKVCacheRequest(std::move(contextRequest), swaPageManager, std::move(swaRequest), speculativeRequest);
}

std::optional<ManagedKVCacheRequest> ManagedKVCacheRequest::begin(ContextCacheCoordinator& contextCache,
    ContextCacheBatchAdmission const& admission, DecodingKvHeadroom const& headroom, cudaStream_t stream)
{
    std::optional<ContextCacheRequest> opened = ContextCacheRequest::begin(contextCache, admission, headroom, stream);
    if (!opened.has_value())
    {
        return std::nullopt;
    }
    // Context reuse only: bounded SWA has no live-slot admission, so a batch opened this way never carries it.
    return ManagedKVCacheRequest(
        std::move(opened), /*swaPageManager=*/nullptr, std::nullopt, admission.speculativeRequest);
}

ManagedKVCacheRequest::ManagedKVCacheRequest(std::optional<ContextCacheRequest>&& contextRequest,
    BoundedSwaKVPageManager* swaPageManager, std::optional<BoundedSwaKVPageManager::RequestHandle>&& swaRequest,
    bool speculativeRequest) noexcept
    : mBoundedSwaPageManager(swaPageManager)
    , mSpeculativeRequest(speculativeRequest)
    , mContextRequest(std::move(contextRequest))
    , mSwaRequest(std::move(swaRequest))
{
}

bool ManagedKVCacheRequest::hasContextReuse() const noexcept
{
    return mContextRequest.has_value();
}

bool ManagedKVCacheRequest::hasBoundedSwa() const noexcept
{
    return mSwaRequest.has_value();
}

ContextCacheRequest& ManagedKVCacheRequest::contextRequest()
{
    ELLM_CHECK(hasContextReuse(), "Managed KV cache request has no context-reuse backend");
    return *mContextRequest;
}

ContextCacheRequest const& ManagedKVCacheRequest::contextRequest() const
{
    ELLM_CHECK(hasContextReuse(), "Managed KV cache request has no context-reuse backend");
    return *mContextRequest;
}

BoundedSwaKVPageManager::RequestHandle& ManagedKVCacheRequest::swaRequest()
{
    ELLM_CHECK(
        hasBoundedSwa() && mBoundedSwaPageManager != nullptr, "Managed KV cache request has no bounded-SWA client");
    return *mSwaRequest;
}

BoundedSwaKVPageManager::ExternalPageLifecycle ManagedKVCacheRequest::swaPageLifecycle()
{
    if (!hasContextReuse())
    {
        return {};
    }
    return BoundedSwaKVPageManager::ExternalPageLifecycle{
        [this](int32_t slot, PageId page) { return contextRequest().isExternallyOwnedSwaPage(slot, page); },
        [this](int32_t slot, PageId page) { return contextRequest().replaceSwaPage(slot, page); }};
}

SwaWindowStateView ManagedKVCacheRequest::swaWindowStates(int32_t batchSize) const
{
    ELLM_CHECK(hasBoundedSwa() && batchSize >= 0, "Managed KV cache SWA snapshot request is invalid");
    SwaWindowStateView const states = mBoundedSwaPageManager->states(*mSwaRequest);
    ELLM_CHECK(states.size() == static_cast<size_t>(batchSize),
        "Managed KV cache SWA state view does not match the active batch");
    return states;
}

std::vector<int32_t> const* ManagedKVCacheRequest::prefillStarts() const noexcept
{
    return hasContextReuse() ? &contextRequest().prefillStarts() : nullptr;
}

int32_t ManagedKVCacheRequest::reuseTokenLength(int32_t slot) const
{
    return contextRequest().reuseTokenLength(slot);
}

bool ManagedKVCacheRequest::publishHybridSpecEndpoint(
    int32_t slot, int32_t residentStateLength, Tensor const& baseHiddenStates, int32_t boundaryHiddenRow)
{
    return contextRequest().publishHybridSpecEndpoint(slot, residentStateLength, baseHiddenStates, boundaryHiddenRow);
}

bool ManagedKVCacheRequest::restoreHybridSpecBoundaryHidden(
    int32_t slot, Tensor& baseHiddenStates, int32_t destinationRow)
{
    return contextRequest().restoreHybridSpecBoundaryHidden(slot, baseHiddenStates, destinationRow);
}

bool ManagedKVCacheRequest::preparePrefill()
{
    return !hasContextReuse() || contextRequest().preparePrefill();
}

bool ManagedKVCacheRequest::prepareSwaPrefill(std::vector<int32_t> const& effectiveInputLengths)
{
    if (!hasBoundedSwa())
    {
        return true;
    }
    std::vector<int32_t> logicalInputLengths = effectiveInputLengths;
    if (hasContextReuse())
    {
        ELLM_CHECK(logicalInputLengths.size() == contextRequest().prefillStarts().size(),
            "Managed bounded-SWA prefill lengths do not match context admission");
        for (size_t slot = 0; slot < logicalInputLengths.size(); ++slot)
        {
            ELLM_CHECK(logicalInputLengths[slot]
                    <= std::numeric_limits<int32_t>::max() - contextRequest().prefillStarts()[slot],
                "Managed bounded-SWA logical prefill length exceeds int32");
            logicalInputLengths[slot] += contextRequest().prefillStarts()[slot];
        }
    }
    return swaOperationSucceeded(
        mBoundedSwaPageManager->preparePrefill(swaRequest(), logicalInputLengths, swaPageLifecycle()),
        "SWA prefill preparation");
}

bool ManagedKVCacheRequest::enqueuePrefillCaptures()
{
    return !hasContextReuse() || contextRequest().enqueuePrefillCaptures();
}

bool ManagedKVCacheRequest::completePrefill(
    DecodingInferenceContext const& context, std::vector<int32_t> const& commonStateLengths)
{
    if (hasBoundedSwa()
        && !swaOperationSucceeded(
            mBoundedSwaPageManager->completePrefill(swaRequest(), swaPageLifecycle()), "prefill completion"))
    {
        return false;
    }
    if (!hasContextReuse())
    {
        return true;
    }
    SwaWindowStateView const states = hasBoundedSwa() ? swaWindowStates(context.activeBatchSize) : SwaWindowStateView{};
    return contextRequest().completePrefill(context, commonStateLengths, hasBoundedSwa() ? &states : nullptr);
}

bool ManagedKVCacheRequest::prepareDecodeStep(DecodingInferenceContext const& context,
    DecodingKvHeadroom const& headroom, DecodingKvHeadroom const& jumpForwardHeadroom)
{
    std::vector<int32_t> const* baseExtraPerSlot = nullptr;
    std::vector<int32_t> const* draftExtraPerSlot = nullptr;
    if (context.hasJumpForward)
    {
        auto const activeSlots = static_cast<size_t>(context.activeBatchSize);
        mBaseExtraTokensScratch.resize(activeSlots);
        if (mSpeculativeRequest)
        {
            ELLM_CHECK(context.jumpForwardEnabled.size() >= activeSlots,
                "Decode preparation needs the jump-forward flag of every active slot");
            // Only a decoder with its own draft working set (draft headroom above zero) takes a per-slot draft
            // growth; Gemma4 MTP's assistant owns no KV cache, so its grafting round widens the base alone.
            bool const ownsDraftWorkingSet = headroom.draftExtraTokens > 0;
            if (ownsDraftWorkingSet)
            {
                mDraftExtraTokensScratch.resize(activeSlots);
            }
            for (size_t slot = 0; slot < activeSlots; ++slot)
            {
                DecodingKvHeadroom const& slotHeadroom
                    = context.jumpForwardEnabled[slot] != 0 ? jumpForwardHeadroom : headroom;
                mBaseExtraTokensScratch[slot] = slotHeadroom.baseExtraTokens;
                if (ownsDraftWorkingSet)
                {
                    mDraftExtraTokensScratch[slot] = slotHeadroom.draftExtraTokens;
                }
            }
            if (ownsDraftWorkingSet)
            {
                draftExtraPerSlot = &mDraftExtraTokensScratch;
            }
        }
        else
        {
            ELLM_CHECK(context.pendingTokens.size() >= activeSlots,
                "Decode preparation needs the pending token count of every active slot");
            for (size_t slot = 0; slot < activeSlots; ++slot)
            {
                mBaseExtraTokensScratch[slot] = std::max(1, context.pendingTokens[slot]);
            }
        }
        baseExtraPerSlot = &mBaseExtraTokensScratch;
    }
    if (hasContextReuse())
    {
        if (!contextRequest().prepareDecodeStep(context, headroom, baseExtraPerSlot, draftExtraPerSlot))
        {
            return false;
        }
    }
    if (!hasBoundedSwa())
    {
        return true;
    }
    ELLM_CHECK(headroom.baseExtraTokens == 1 && headroom.draftExtraTokens == 0,
        "Bounded SWA decode headroom changed after admission");
    return swaOperationSucceeded(
        mBoundedSwaPageManager->prepareDecodeStep(swaRequest(), swaPageLifecycle(), baseExtraPerSlot),
        "decode preparation");
}

bool ManagedKVCacheRequest::prepareDecodeStep(
    DecodingInferenceContext const& context, DecodingKvHeadroom const& headroom, std::vector<int32_t> const& decodeRows)
{
    // Bounded SWA advances every resident window together; it has no live-slot admission either.
    ELLM_CHECK(!hasBoundedSwa(), "Managed bounded-SWA requests have no decode-subset step");
    return !hasContextReuse() || contextRequest().prepareDecodeStep(context, headroom, decodeRows);
}

bool ManagedKVCacheRequest::completeDecodeStep(
    DecodingInferenceContext const& context, std::vector<int32_t> const& commonStateLengths)
{
    if (hasBoundedSwa()
        && !swaOperationSucceeded(
            mBoundedSwaPageManager->completeDecodeStep(swaRequest(), swaPageLifecycle()), "decode completion"))
    {
        return false;
    }
    if (!hasContextReuse())
    {
        return true;
    }
    SwaWindowStateView const states = hasBoundedSwa() ? swaWindowStates(context.activeBatchSize) : SwaWindowStateView{};
    return contextRequest().completeDecodeStep(context, commonStateLengths, hasBoundedSwa() ? &states : nullptr);
}

bool ManagedKVCacheRequest::beginBatchCompaction(
    std::vector<int32_t> const& oldToNew, int32_t newBatchSize, Tensor& deviceBatchMapping)
{
    if (hasContextReuse() && !contextRequest().beginBatchCompaction(oldToNew, newBatchSize, deviceBatchMapping))
    {
        return false;
    }
    return !hasBoundedSwa()
        || swaOperationSucceeded(
            mBoundedSwaPageManager->beginBatchCompaction(swaRequest(), oldToNew, newBatchSize, deviceBatchMapping),
            "batch-compaction preparation");
}

bool ManagedKVCacheRequest::completeBatchCompaction(std::vector<int32_t> const& keepMapping)
{
    if (hasBoundedSwa()
        && !swaOperationSucceeded(mBoundedSwaPageManager->compactBatch(swaRequest()), "batch compaction"))
    {
        return false;
    }
    return !hasContextReuse() || contextRequest().completeBatchCompaction(keepMapping);
}

ContextCacheRequest::AdmitSequenceStatus ManagedKVCacheRequest::admitSequence(std::vector<int32_t> const& tokenIds,
    std::string const& loraWeightsName, DecodingKvHeadroom const& headroom, int32_t& prefillStart, ResidentRef resident,
    cudaStream_t stream, ContextCacheLookupPolicy lookupPolicy, ContextCacheCommitPolicy commitPolicy,
    std::vector<int32_t> const& mediaTokenIds, std::vector<Hash128> const& imageContentDigests,
    std::vector<audioUtils::AudioData> const& audioBuffers)
{
    if (!hasContextReuse())
    {
        return ContextCacheRequest::AdmitSequenceStatus::kFailed;
    }
    return contextRequest().admitSequence(tokenIds, loraWeightsName, headroom, prefillStart, resident, stream,
        lookupPolicy, commitPolicy, mediaTokenIds, imageContentDigests, audioBuffers);
}

bool ManagedKVCacheRequest::prepareSequenceAdmission(int32_t sequenceIndex)
{
    return hasContextReuse() && contextRequest().prepareSequenceAdmission(sequenceIndex);
}

bool ManagedKVCacheRequest::enqueueSequenceAdmissionCapture(int32_t sequenceIndex)
{
    return hasContextReuse() && contextRequest().enqueueSequenceAdmissionCapture(sequenceIndex);
}

bool ManagedKVCacheRequest::completeSequenceAdmissionDeviceWork(int32_t sequenceIndex)
{
    return hasContextReuse() && contextRequest().completeSequenceAdmissionDeviceWork(sequenceIndex);
}

bool ManagedKVCacheRequest::advanceSequenceAdmission(int32_t sequenceIndex, int32_t committedLength)
{
    return hasContextReuse() && contextRequest().advanceSequenceAdmission(sequenceIndex, committedLength);
}

bool ManagedKVCacheRequest::abortSequenceAdmission(int32_t sequenceIndex)
{
    return hasContextReuse() && contextRequest().abortSequenceAdmission(sequenceIndex);
}

bool ManagedKVCacheRequest::finalizeSequenceAdmission(int32_t sequenceIndex, int32_t const& lookaheadToken,
    int32_t fullInputLength, std::optional<int32_t> commonStateLength)
{
    if (!hasContextReuse())
    {
        return false;
    }
    return contextRequest().finalizeSequenceAdmission(
        sequenceIndex, lookaheadToken, fullInputLength, commonStateLength);
}

bool ManagedKVCacheRequest::retractSequenceAdmission() noexcept
{
    if (hasContextReuse())
    {
        return contextRequest().retractSequenceAdmission();
    }
    return true;
}

bool ManagedKVCacheRequest::finish()
{
    if (hasBoundedSwa() && !swaOperationSucceeded(mBoundedSwaPageManager->finish(swaRequest()), "request finish"))
    {
        return false;
    }
    return !hasContextReuse() || contextRequest().finish();
}

} // namespace rt
} // namespace trt_edgellm
