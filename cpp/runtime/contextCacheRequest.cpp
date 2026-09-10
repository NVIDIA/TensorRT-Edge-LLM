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

#include "runtime/contextCacheRequest.h"

#include "common/checkMacros.h"
#include "common/logger.h"
#include "runtime/audioUtils.h"
#include "runtime/imageUtils.h"
#include "runtime/llmRuntimeUtils.h"
#include "runtime/state/contextCache/blockHash.h"
#include "runtime/state/decodingInferenceContext.h"
#include "runtime/streaming.h"

#include <algorithm>
#include <exception>
#include <limits>
#include <string>
#include <string_view>
#include <utility>

namespace trt_edgellm
{
namespace rt
{
namespace
{

ContextCacheLookupPolicy contextCacheLookupPolicy(LLMGenerationRequest const& request, bool outputThinkerEmbeddings)
{
    bool const requiresBypass = request.contextCacheLookupPolicy == ContextCacheLookupPolicy::kBypass
        || request.generateAudio || outputThinkerEmbeddings;
    return requiresBypass ? ContextCacheLookupPolicy::kBypass : ContextCacheLookupPolicy::kUseCache;
}

bool isMediaToken(int32_t tokenId, std::vector<int32_t> const& mediaTokenIds)
{
    return std::find(mediaTokenIds.begin(), mediaTokenIds.end(), tokenId) != mediaTokenIds.end();
}

std::vector<Hash128> buildPerPositionMediaHash(std::vector<int32_t> const& tokenIds,
    std::vector<int32_t> const& mediaTokenIds, std::vector<imageUtils::ImageData> const& imageBuffers,
    std::vector<audioUtils::AudioData> const& audioBuffers)
{
    if (mediaTokenIds.empty())
    {
        return {};
    }

    std::vector<Hash128> imageHashes;
    imageHashes.reserve(imageBuffers.size());
    for (auto const& image : imageBuffers)
    {
        size_t const totalBytes = static_cast<size_t>(image.bytesPerFrame()) * static_cast<size_t>(image.frames);
        std::string_view const bytes(reinterpret_cast<char const*>(image.data()), totalBytes);
        imageHashes.push_back(hashOpaqueIdentity(bytes));
    }

    std::vector<Hash128> audioHashes;
    audioHashes.reserve(audioBuffers.size());
    for (auto const& audio : audioBuffers)
    {
        if (audio.pcm && !audio.pcm->samples.empty())
        {
            size_t const totalBytes = audio.pcm->samples.size() * sizeof(float);
            std::string_view const bytes(reinterpret_cast<char const*>(audio.pcm->samples.data()), totalBytes);
            audioHashes.push_back(hashOpaqueIdentity(bytes));
        }
        else
        {
            audioHashes.push_back(hashOpaqueIdentity(audio.melSpectrogramPath));
        }
    }

    int32_t const imageTokenId = (!mediaTokenIds.empty()) ? mediaTokenIds[0] : -1;
    int32_t const audioTokenId = (mediaTokenIds.size() > 1) ? mediaTokenIds[1] : -1;

    std::vector<Hash128> perPositionHash(tokenIds.size(), Hash128{});
    size_t imageIdx = 0;
    size_t audioIdx = 0;
    bool previousWasMedia = false;
    int32_t previousMediaTokenId = -1;

    for (size_t i = 0; i < tokenIds.size(); ++i)
    {
        int32_t const token = tokenIds[i];
        bool const currentIsMedia = isMediaToken(token, mediaTokenIds);

        if (currentIsMedia)
        {
            // Transition between different media types (e.g. image run → audio run) without an
            // intervening non-media token: close out the previous modality's run.
            if (previousWasMedia && token != previousMediaTokenId)
            {
                if (previousMediaTokenId == imageTokenId)
                {
                    ++imageIdx;
                }
                else if (previousMediaTokenId == audioTokenId)
                {
                    ++audioIdx;
                }
            }

            if (token == imageTokenId)
            {
                ELLM_CHECK(imageIdx < imageHashes.size(),
                    "Image token placeholder at position " + std::to_string(i) + " exceeds provided image count ("
                        + std::to_string(imageHashes.size()) + ")");
                perPositionHash[i] = imageHashes[imageIdx];
            }
            else if (token == audioTokenId)
            {
                ELLM_CHECK(audioIdx < audioHashes.size(),
                    "Audio token placeholder at position " + std::to_string(i) + " exceeds provided audio count ("
                        + std::to_string(audioHashes.size()) + ")");
                perPositionHash[i] = audioHashes[audioIdx];
            }
            previousWasMedia = true;
            previousMediaTokenId = token;
        }
        else
        {
            if (previousWasMedia)
            {
                if (previousMediaTokenId == imageTokenId)
                {
                    ++imageIdx;
                }
                else if (previousMediaTokenId == audioTokenId)
                {
                    ++audioIdx;
                }
            }
            previousWasMedia = false;
            previousMediaTokenId = -1;
        }
    }

    bool const hasAnyMedia
        = std::any_of(perPositionHash.begin(), perPositionHash.end(), [](Hash128 const& h) { return h != Hash128{}; });
    if (!hasAnyMedia)
    {
        return {};
    }
    return perPositionHash;
}

ContextCacheSequenceAdmission makeContextCacheSequenceAdmission(std::vector<int32_t> const& tokenIds,
    std::string const& loraWeightsName, std::vector<int32_t> const& mediaTokenIds,
    std::vector<imageUtils::ImageData> const& imageBuffers, std::vector<audioUtils::AudioData> const& audioBuffers)
{
    ContextCacheSequenceAdmission admission;
    admission.tokenIds = tokenIds;
    if (!loraWeightsName.empty())
    {
        // The runtime-local adapter registry is immutable after construction, so its generation remains zero.
        AdapterKey const adapter{hashOpaqueIdentity(loraWeightsName), 0};
        admission.keyExtras.adapter = adapter;
    }
    admission.perPositionMediaHash = buildPerPositionMediaHash(tokenIds, mediaTokenIds, imageBuffers, audioBuffers);
    return admission;
}

bool contextCacheOperationSucceeded(ContextCacheCoordinatorStatus status, char const* operation)
{
    if (status == ContextCacheCoordinatorStatus::kOk)
    {
        return true;
    }
    LOG_ERROR("Context-cache %s failed (%s).", operation,
        status == ContextCacheCoordinatorStatus::kPoisoned ? "coordinator poisoned" : "request failure");
    return false;
}

} // namespace

std::optional<ContextCacheRequest> ContextCacheRequest::begin(ContextCacheCoordinator& coordinator,
    LLMGenerationRequest const& request, DecodingInferenceContext const& context, bool speculativeRequest,
    DecodingKvHeadroom const& headroom, std::vector<int32_t> const& mediaTokenIds)
{
    static std::vector<imageUtils::ImageData> const kEmptyImageBuffers;
    static std::vector<audioUtils::AudioData> const kEmptyAudioBuffers;

    ContextCacheBatchAdmission admission;
    admission.speculativeRequest = speculativeRequest;
    admission.lookupPolicy = contextCacheLookupPolicy(request, context.outputThinkerEmbeddings);
    admission.commitPolicy = request.contextCacheCommitPolicy;
    admission.replayTailLength = request.contextCacheReplayTailLength;
    admission.sequences.reserve(context.rawBatchedInputIds.size());
    for (size_t seqIdx = 0; seqIdx < context.rawBatchedInputIds.size(); ++seqIdx)
    {
        std::vector<imageUtils::ImageData> const& images
            = (seqIdx < request.requests.size()) ? request.requests[seqIdx].imageBuffers : kEmptyImageBuffers;
        std::vector<audioUtils::AudioData> const& audio
            = (seqIdx < request.requests.size()) ? request.requests[seqIdx].audioBuffers : kEmptyAudioBuffers;
        admission.sequences.push_back(makeContextCacheSequenceAdmission(
            context.rawBatchedInputIds[seqIdx], context.loraWeightsName, mediaTokenIds, images, audio));
    }

    ContextCacheCoordinator::BeginRequestResult admitted
        = coordinator.beginRequest(admission, headroom, context.stream);
    if (!contextCacheOperationSucceeded(admitted.status, "admission") || !admitted.admission.has_value())
    {
        return std::nullopt;
    }
    return ContextCacheRequest{coordinator, std::move(*admitted.admission)};
}

ContextCacheRequest::ContextCacheRequest(
    ContextCacheCoordinator& coordinator, ContextCacheCoordinator::AdmissionResult&& admission) noexcept
    : mCoordinator(coordinator)
    , mRequest(std::move(admission.request))
    , mPrefillStarts(std::move(admission.prefillStarts))
{
}

std::vector<int32_t> const& ContextCacheRequest::prefillStarts() const noexcept
{
    return mPrefillStarts;
}

int32_t ContextCacheRequest::reuseTokenLength(int32_t slot) const noexcept
{
    return mPrefillStarts[static_cast<size_t>(slot)];
}

bool ContextCacheRequest::publishHybridMtpEndpoint(
    int32_t slot, int32_t residentStateLength, Tensor const& baseHiddenStates, int32_t boundaryHiddenRow)
{
    return contextCacheOperationSucceeded(
        mCoordinator.publishHybridMtpEndpoint(mRequest, slot, residentStateLength, baseHiddenStates, boundaryHiddenRow),
        "Hybrid+MTP endpoint publication");
}

bool ContextCacheRequest::restoreHybridMtpBoundaryHidden(int32_t slot, Tensor& baseHiddenStates, int32_t destinationRow)
{
    return contextCacheOperationSucceeded(
        mCoordinator.restoreHybridMtpBoundaryHidden(mRequest, slot, baseHiddenStates, destinationRow),
        "Hybrid+MTP boundary-hidden restore");
}
bool ContextCacheRequest::preparePrefill()
{
    return contextCacheOperationSucceeded(mCoordinator.preparePrefill(mRequest), "prefill preparation");
}

bool ContextCacheRequest::enqueuePrefillCaptures()
{
    return contextCacheOperationSucceeded(mCoordinator.enqueuePrefillCaptures(mRequest), "prefill snapshot capture");
}

bool ContextCacheRequest::completePrefill(
    DecodingInferenceContext const& context, std::vector<int32_t> const& commonStateLengths)
{
    std::vector<ContextCacheSequenceAdvance> progress;
    progress.reserve(static_cast<size_t>(context.activeBatchSize));
    for (int32_t slot = 0; slot < context.activeBatchSize; ++slot)
    {
        ELLM_CHECK(context.currentGenerateLengths[slot] == 1 && !context.tokenIds[slot].empty(),
            "Managed context-cache prefill did not produce one sampled lookahead token");
        progress.push_back(ContextCacheSequenceAdvance{
            &context.tokenIds[slot].back(), 1, static_cast<int32_t>(context.rawBatchedInputIds[slot].size())});
    }
    std::vector<int32_t> const* const commonStateLengthsPtr
        = commonStateLengths.empty() ? nullptr : &commonStateLengths;
    return contextCacheOperationSucceeded(
        mCoordinator.finalizePrefillPublication(mRequest, progress, commonStateLengthsPtr), "prefill publication");
}

bool ContextCacheRequest::prepareDecodeStep(DecodingInferenceContext const& context, DecodingKvHeadroom const& headroom)
{
    ELLM_CHECK(!mTokenCountsBeforeDecode.has_value(),
        "Managed context-cache decode preparation cannot overlap a pending decode step.");
    if (!contextCacheOperationSucceeded(mCoordinator.prepareDecodeStep(mRequest, headroom), "decode preparation"))
    {
        return false;
    }

    std::vector<size_t> tokenCounts;
    tokenCounts.reserve(static_cast<size_t>(context.activeBatchSize));
    for (int32_t slot = 0; slot < context.activeBatchSize; ++slot)
    {
        tokenCounts.push_back(context.tokenIds[slot].size());
    }
    mTokenCountsBeforeDecode.emplace(std::move(tokenCounts));
    return true;
}

bool ContextCacheRequest::completeDecodeStep(
    DecodingInferenceContext const& context, std::vector<int32_t> const& commonStateLengths)
{
    ELLM_CHECK(
        mTokenCountsBeforeDecode.has_value(), "Managed context-cache decode completion has no prepared decode step.");
    std::vector<size_t> tokenCountsBeforeDecode = std::move(*mTokenCountsBeforeDecode);
    mTokenCountsBeforeDecode.reset();
    ELLM_CHECK(static_cast<int32_t>(tokenCountsBeforeDecode.size()) == context.activeBatchSize,
        "Managed context-cache active batch changed during a decode step.");

    std::vector<ContextCacheSequenceAdvance> progress;
    std::vector<int32_t> publishableCompletedSlots;
    progress.reserve(static_cast<size_t>(context.activeBatchSize));
    publishableCompletedSlots.reserve(static_cast<size_t>(context.activeBatchSize));
    for (int32_t slot = 0; slot < context.activeBatchSize; ++slot)
    {
        ELLM_CHECK(!context.tokenIds[slot].empty() && context.currentGenerateLengths[slot] > 0,
            "Managed context-cache decode did not produce a sampled lookahead token");
        size_t const previousTokenCount = tokenCountsBeforeDecode[static_cast<size_t>(slot)];
        ELLM_CHECK(context.tokenIds[slot].size() > previousTokenCount
                && context.tokenIds[slot].size() - previousTokenCount
                    <= static_cast<size_t>(std::numeric_limits<int32_t>::max()),
            "Managed context-cache decode produced an invalid accepted-token delta");
        int32_t const acceptedTokenCount = static_cast<int32_t>(context.tokenIds[slot].size() - previousTokenCount);
        int64_t const committedStateLength = static_cast<int64_t>(context.rawBatchedInputIds[slot].size())
            + static_cast<int64_t>(context.currentGenerateLengths[slot]) - 1;
        ELLM_CHECK(committedStateLength <= static_cast<int64_t>(std::numeric_limits<int32_t>::max()),
            "Managed context-cache committed state length exceeds int32");
        progress.push_back(ContextCacheSequenceAdvance{context.tokenIds[slot].data() + previousTokenCount,
            acceptedTokenCount, static_cast<int32_t>(committedStateLength)});

        FinishReason const terminalReason = context.slotStreams[slot].terminalReason;
        if (context.finishedStates[slot] && terminalReason != FinishReason::kCancelled
            && terminalReason != FinishReason::kError)
        {
            publishableCompletedSlots.push_back(slot);
        }
    }

    std::vector<int32_t> const* const commonStateLengthsPtr
        = commonStateLengths.empty() ? nullptr : &commonStateLengths;
    return contextCacheOperationSucceeded(
        mCoordinator.completeDecodeStep(mRequest, progress, publishableCompletedSlots, commonStateLengthsPtr),
        "decode completion");
}

bool ContextCacheRequest::beginBatchCompaction(
    std::vector<int32_t> const& oldToNew, int32_t newBatchSize, Tensor& deviceBatchMapping)
{
    return contextCacheOperationSucceeded(
        mCoordinator.beginBatchCompaction(mRequest, oldToNew, newBatchSize, deviceBatchMapping),
        "batch-compaction preparation");
}

bool ContextCacheRequest::completeBatchCompaction()
{
    return contextCacheOperationSucceeded(mCoordinator.compactBatch(mRequest), "batch compaction");
}

bool ContextCacheRequest::finish()
{
    ELLM_CHECK(
        !mTokenCountsBeforeDecode.has_value(), "Managed context-cache request cannot finish during a decode step.");
    return contextCacheOperationSucceeded(mCoordinator.finish(mRequest), "request finish");
}

} // namespace rt
} // namespace trt_edgellm
