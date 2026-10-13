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
    ELLM_CHECK((contextCache != nullptr) != (swaPageManager != nullptr),
        "Managed KV cache request requires exactly one lifecycle backend");
    if (contextCache != nullptr)
    {
        std::optional<ContextCacheRequest> admitted = ContextCacheRequest::begin(*contextCache, request, context,
            speculativeRequest, headroom, mediaTokenIds, tokenStateContract, commitPolicy);
        if (!admitted.has_value())
        {
            return std::nullopt;
        }
        return ManagedKVCacheRequest(std::move(*admitted));
    }

    ELLM_CHECK(!speculativeRequest && tokenStateContract == DecodingTokenStateContract::kCommittedPlusLookahead
            && headroom.baseExtraTokens == 1 && headroom.draftExtraTokens == 0,
        "Bounded SWA KV cache supports only vanilla one-token decode");
    ELLM_CHECK(context.activeBatchSize > 0
            && context.rawBatchedInputIds.size() == static_cast<size_t>(context.activeBatchSize),
        "Bounded SWA admission requires one raw input per active sequence");
    BoundedSwaKVPageManager::BeginRequestResult admitted
        = swaPageManager->beginRequest(context.residentRefs, context.stream);
    if (!swaOperationSucceeded(admitted.status, "admission") || !admitted.request.has_value())
    {
        return std::nullopt;
    }
    return ManagedKVCacheRequest(*swaPageManager, std::move(*admitted.request));
}

ManagedKVCacheRequest::ManagedKVCacheRequest(ContextCacheRequest&& request) noexcept
    : mBackend(std::move(request))
{
}

ManagedKVCacheRequest::ManagedKVCacheRequest(
    BoundedSwaKVPageManager& manager, BoundedSwaKVPageManager::RequestHandle&& request) noexcept
    : mBackend(std::move(request))
    , mBoundedSwaPageManager(&manager)
{
}

bool ManagedKVCacheRequest::hasContextReuse() const noexcept
{
    return std::holds_alternative<ContextCacheRequest>(mBackend);
}

ContextCacheRequest& ManagedKVCacheRequest::contextRequest()
{
    ELLM_CHECK(hasContextReuse(), "Managed KV cache request has no context-reuse backend");
    return std::get<ContextCacheRequest>(mBackend);
}

ContextCacheRequest const& ManagedKVCacheRequest::contextRequest() const
{
    ELLM_CHECK(hasContextReuse(), "Managed KV cache request has no context-reuse backend");
    return std::get<ContextCacheRequest>(mBackend);
}

BoundedSwaKVPageManager::RequestHandle& ManagedKVCacheRequest::swaRequest()
{
    ELLM_CHECK(
        !hasContextReuse() && mBoundedSwaPageManager != nullptr, "Managed KV cache request has no bounded-SWA backend");
    return std::get<BoundedSwaKVPageManager::RequestHandle>(mBackend);
}

std::vector<int32_t> const* ManagedKVCacheRequest::prefillStarts() const noexcept
{
    return hasContextReuse() ? &contextRequest().prefillStarts() : nullptr;
}

int32_t ManagedKVCacheRequest::reuseTokenLength(int32_t slot) const
{
    return contextRequest().reuseTokenLength(slot);
}

bool ManagedKVCacheRequest::publishHybridMtpEndpoint(
    int32_t slot, int32_t residentStateLength, Tensor const& baseHiddenStates, int32_t boundaryHiddenRow)
{
    return contextRequest().publishHybridMtpEndpoint(slot, residentStateLength, baseHiddenStates, boundaryHiddenRow);
}

bool ManagedKVCacheRequest::restoreHybridMtpBoundaryHidden(
    int32_t slot, Tensor& baseHiddenStates, int32_t destinationRow)
{
    return contextRequest().restoreHybridMtpBoundaryHidden(slot, baseHiddenStates, destinationRow);
}

bool ManagedKVCacheRequest::preparePrefill()
{
    return !hasContextReuse() || contextRequest().preparePrefill();
}

bool ManagedKVCacheRequest::prepareSwaPrefill(std::vector<int32_t> const& effectiveInputLengths)
{
    return hasContextReuse()
        || swaOperationSucceeded(
            mBoundedSwaPageManager->preparePrefill(swaRequest(), effectiveInputLengths), "SWA prefill preparation");
}

bool ManagedKVCacheRequest::enqueuePrefillCaptures()
{
    return !hasContextReuse() || contextRequest().enqueuePrefillCaptures();
}

bool ManagedKVCacheRequest::completePrefill(
    DecodingInferenceContext const& context, std::vector<int32_t> const& commonStateLengths)
{
    return hasContextReuse()
        ? contextRequest().completePrefill(context, commonStateLengths)
        : swaOperationSucceeded(mBoundedSwaPageManager->completePrefill(swaRequest()), "prefill completion");
}

bool ManagedKVCacheRequest::prepareDecodeStep(
    DecodingInferenceContext const& context, DecodingKvHeadroom const& headroom)
{
    if (hasContextReuse())
    {
        return contextRequest().prepareDecodeStep(context, headroom);
    }
    ELLM_CHECK(headroom.baseExtraTokens == 1 && headroom.draftExtraTokens == 0,
        "Bounded SWA decode headroom changed after admission");
    return swaOperationSucceeded(mBoundedSwaPageManager->prepareDecodeStep(swaRequest()), "decode preparation");
}

bool ManagedKVCacheRequest::completeDecodeStep(
    DecodingInferenceContext const& context, std::vector<int32_t> const& commonStateLengths)
{
    return hasContextReuse()
        ? contextRequest().completeDecodeStep(context, commonStateLengths)
        : swaOperationSucceeded(mBoundedSwaPageManager->completeDecodeStep(swaRequest()), "decode completion");
}

bool ManagedKVCacheRequest::beginBatchCompaction(
    std::vector<int32_t> const& oldToNew, int32_t newBatchSize, Tensor& deviceBatchMapping)
{
    return hasContextReuse() ? contextRequest().beginBatchCompaction(oldToNew, newBatchSize, deviceBatchMapping)
                             : swaOperationSucceeded(mBoundedSwaPageManager->beginBatchCompaction(
                                                         swaRequest(), oldToNew, newBatchSize, deviceBatchMapping),
                                   "batch-compaction preparation");
}

bool ManagedKVCacheRequest::completeBatchCompaction(std::vector<int32_t> const& keepMapping)
{
    return hasContextReuse()
        ? contextRequest().completeBatchCompaction(keepMapping)
        : swaOperationSucceeded(mBoundedSwaPageManager->compactBatch(swaRequest()), "batch compaction");
}

ContextCacheRequest::AdmitSequenceStatus ManagedKVCacheRequest::admitSequence(std::vector<int32_t> const& tokenIds,
    std::string const& loraWeightsName, DecodingKvHeadroom const& headroom, int32_t& prefillStart, ResidentRef resident,
    cudaStream_t stream, std::vector<int32_t> const& mediaTokenIds,
    std::vector<imageUtils::ImageData> const& imageBuffers, std::vector<audioUtils::AudioData> const& audioBuffers)
{
    if (!hasContextReuse())
    {
        return ContextCacheRequest::AdmitSequenceStatus::kFailed;
    }
    return contextRequest().admitSequence(
        tokenIds, loraWeightsName, headroom, prefillStart, resident, stream, mediaTokenIds, imageBuffers, audioBuffers);
}

bool ManagedKVCacheRequest::finalizeSequenceAdmission(
    int32_t slot, int32_t const& lookaheadToken, int32_t fullInputLength)
{
    if (!hasContextReuse())
    {
        return false;
    }
    return contextRequest().finalizeSequenceAdmission(slot, lookaheadToken, fullInputLength);
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
    return hasContextReuse() ? contextRequest().finish()
                             : swaOperationSucceeded(mBoundedSwaPageManager->finish(swaRequest()), "request finish");
}

} // namespace rt
} // namespace trt_edgellm
