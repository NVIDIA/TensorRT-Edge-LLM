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

ContextCacheLookupPolicy contextCacheLookupPolicy(
    LLMGenerationRequest const& request, bool outputThinkerEmbeddings, std::vector<int32_t> const& mediaTokenIds)
{
    // A media position keys on the pixels, read on the host; keying one whose pixels are unreadable on the
    // media token alone would reuse whatever the last image cached, that token being every image's placeholder.
    bool const mediaUnreadable = !mediaTokenIds.empty()
        && std::any_of(request.requests.begin(), request.requests.end(), [](auto const& sequence) {
               return std::any_of(sequence.imageBuffers.begin(), sequence.imageBuffers.end(),
                   [](imageUtils::ImageData const& image) { return image.data() == nullptr; });
           });
    bool const requiresBypass = request.contextCacheLookupPolicy == ContextCacheLookupPolicy::kBypass
        || request.generateAudio || outputThinkerEmbeddings || mediaUnreadable;
    return requiresBypass ? ContextCacheLookupPolicy::kBypass : ContextCacheLookupPolicy::kUseCache;
}

bool isMediaToken(int32_t tokenId, std::vector<int32_t> const& mediaTokenIds)
{
    return std::find(mediaTokenIds.begin(), mediaTokenIds.end(), tokenId) != mediaTokenIds.end();
}

std::vector<Hash128> buildPerPositionMediaHash(std::vector<int32_t> const& tokenIds,
    std::vector<int32_t> const& mediaTokenIds, std::vector<imageUtils::ImageData> const& imageBuffers,
    std::vector<audioUtils::AudioData> const& audioBuffers, cudaStream_t stream)
{
    if (mediaTokenIds.empty())
    {
        return {};
    }

    std::vector<Hash128> imageHashes;
    imageHashes.reserve(imageBuffers.size());
    for (auto const& image : imageBuffers)
    {
        std::string_view const bytes(
            reinterpret_cast<char const*>(image.data()), static_cast<size_t>(image.addressedBytes()));
        imageHashes.push_back(hashOpaqueIdentity(bytes, stream, false));
    }

    std::vector<Hash128> audioHashes;
    audioHashes.reserve(audioBuffers.size());
    for (auto const& audio : audioBuffers)
    {
        if (audio.pcm && audio.pcm->numSamples() > 0)
        {
            Tensor const& samples = *audio.pcm->samples;
            size_t const totalBytes = static_cast<size_t>(audio.pcm->numSamples()) * sizeof(float);
            std::string_view const bytes(reinterpret_cast<char const*>(samples.dataPointer<float>()), totalBytes);
            audioHashes.push_back(hashOpaqueIdentity(bytes, stream, false));
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
    std::vector<imageUtils::ImageData> const& imageBuffers, std::vector<audioUtils::AudioData> const& audioBuffers,
    cudaStream_t stream)
{
    ContextCacheSequenceAdmission admission;
    admission.tokenIds = tokenIds;
    if (!loraWeightsName.empty())
    {
        // The runtime-local adapter registry is immutable after construction, so its generation remains zero.
        AdapterKey const adapter{hashOpaqueIdentity(loraWeightsName), 0};
        admission.keyExtras.adapter = adapter;
    }
    admission.perPositionMediaHash
        = buildPerPositionMediaHash(tokenIds, mediaTokenIds, imageBuffers, audioBuffers, stream);
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
    DecodingKvHeadroom const& headroom, std::vector<int32_t> const& mediaTokenIds,
    DecodingTokenStateContract tokenStateContract, ContextCacheCommitPolicy commitPolicy)
{
    static std::vector<imageUtils::ImageData> const kEmptyImageBuffers;
    static std::vector<audioUtils::AudioData> const kEmptyAudioBuffers;
    static std::vector<int32_t> const kEmptyMediaTokenIds;

    ContextCacheBatchAdmission admission;
    admission.speculativeRequest = speculativeRequest;
    admission.lookupPolicy = contextCacheLookupPolicy(request, context.outputThinkerEmbeddings, mediaTokenIds);
    admission.tokenStateContract = tokenStateContract;
    admission.commitPolicy = commitPolicy;
    admission.replayTailLength = request.contextCacheReplayTailLength;
    ELLM_CHECK(context.residentRefs.size() == context.rawBatchedInputIds.size(),
        "Context-cache admission requires one resident identity per input sequence");
    admission.sequences.reserve(context.rawBatchedInputIds.size());
    for (size_t seqIdx = 0; seqIdx < context.rawBatchedInputIds.size(); ++seqIdx)
    {
        std::optional<ContextCacheLookupPolicy> const sequenceLookupPolicy
            = (seqIdx < request.requests.size()) ? request.requests[seqIdx].contextCacheLookupPolicy : std::nullopt;
        ContextCacheLookupPolicy const effectiveLookupPolicy
            = admission.lookupPolicy == ContextCacheLookupPolicy::kBypass
            ? ContextCacheLookupPolicy::kBypass
            : sequenceLookupPolicy.value_or(admission.lookupPolicy);
        std::vector<imageUtils::ImageData> const& images
            = (seqIdx < request.requests.size()) ? request.requests[seqIdx].imageBuffers : kEmptyImageBuffers;
        std::vector<audioUtils::AudioData> const& audio
            = (seqIdx < request.requests.size()) ? request.requests[seqIdx].audioBuffers : kEmptyAudioBuffers;
        ContextCacheSequenceAdmission sequence
            = makeContextCacheSequenceAdmission(context.rawBatchedInputIds[seqIdx], context.loraWeightsName,
                effectiveLookupPolicy == ContextCacheLookupPolicy::kUseCache ? mediaTokenIds : kEmptyMediaTokenIds,
                images, audio, context.stream);
        sequence.resident = context.residentRefs[seqIdx];
        sequence.lookupPolicy = sequenceLookupPolicy;
        admission.sequences.push_back(std::move(sequence));
    }

    ContextCacheCoordinator::BeginRequestResult admitted
        = coordinator.beginRequest(admission, headroom, context.stream);
    if (!contextCacheOperationSucceeded(admitted.status, "admission") || !admitted.admission.has_value())
    {
        return std::nullopt;
    }
    return ContextCacheRequest{coordinator, std::move(*admitted.admission), tokenStateContract};
}

ContextCacheRequest::ContextCacheRequest(ContextCacheCoordinator& coordinator,
    ContextCacheCoordinator::AdmissionResult&& admission, DecodingTokenStateContract tokenStateContract) noexcept
    : mCoordinator(coordinator)
    , mRequest(std::move(admission.request))
    , mTokenStateContract(tokenStateContract)
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

ContextCacheRequest::AdmitSequenceStatus ContextCacheRequest::admitSequence(std::vector<int32_t> const& tokenIds,
    std::string const& loraWeightsName, DecodingKvHeadroom const& headroom, int32_t& prefillStart, ResidentRef resident,
    cudaStream_t stream, std::vector<int32_t> const& mediaTokenIds,
    std::vector<imageUtils::ImageData> const& imageBuffers, std::vector<audioUtils::AudioData> const& audioBuffers)
{
    mPrefillStarts.reserve(mPrefillStarts.size() + 1);
    ContextCacheSequenceAdmission admission = makeContextCacheSequenceAdmission(
        tokenIds, loraWeightsName, mediaTokenIds, imageBuffers, audioBuffers, stream);
    admission.resident = resident;
    ContextCacheCoordinator::AdmitSequenceResult result = mCoordinator.admitSequence(mRequest, admission, headroom);
    if (result.status != ContextCacheCoordinatorStatus::kOk)
    {
        if (result.insufficientCapacity)
        {
            return AdmitSequenceStatus::kNoCapacity;
        }
        contextCacheOperationSucceeded(result.status, "sequence admission");
        return AdmitSequenceStatus::kFailed;
    }
    mPrefillStarts.push_back(result.prefillStart);
    prefillStart = result.prefillStart;
    return AdmitSequenceStatus::kAdmitted;
}

bool ContextCacheRequest::retractSequenceAdmission() noexcept
{
    if (mPrefillStarts.empty())
    {
        return false;
    }
    bool const status = mCoordinator.retractSequenceAdmission(mRequest);
    mPrefillStarts.pop_back();
    return status;
}

bool ContextCacheRequest::finalizeSequenceAdmission(
    int32_t slot, int32_t const& lookaheadToken, int32_t fullInputLength)
{
    return contextCacheOperationSucceeded(mCoordinator.finalizeSequenceAdmission(mRequest, slot,
                                              ContextCacheSequenceAdvance{&lookaheadToken, 1, fullInputLength}),
        "sequence-admission finalization");
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
        if (mTokenStateContract == DecodingTokenStateContract::kFullyCommitted)
        {
            size_t const suffixLength = context.tokenIds[slot].size();
            size_t const fullInputLength = context.rawBatchedInputIds[slot].size();
            ELLM_CHECK(context.currentGenerateLengths[slot] == 0
                    && suffixLength == static_cast<size_t>(context.effectivePrefillLengths[slot]) && suffixLength > 0
                    && suffixLength <= fullInputLength,
                "Fully committed prefill must not append a sampled lookahead token");
            progress.push_back(ContextCacheSequenceAdvance{nullptr, 0, static_cast<int32_t>(fullInputLength)});
        }
        else
        {
            ELLM_CHECK(context.currentGenerateLengths[slot] == 1 && !context.tokenIds[slot].empty(),
                "Managed context-cache prefill did not produce one sampled lookahead token");
            progress.push_back(ContextCacheSequenceAdvance{
                &context.tokenIds[slot].back(), 1, static_cast<int32_t>(context.rawBatchedInputIds[slot].size())});
        }
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
        size_t const previousTokenCount = tokenCountsBeforeDecode[static_cast<size_t>(slot)];
        // A slot cancelled (or failed) at the top of this step is skipped by the decoder and
        // appends nothing; that is a legal zero advance, not a broken step. Only a slot that was
        // still live through the step must have produced its lookahead token.
        if (context.finishedStates[slot] && context.tokenIds[slot].size() == previousTokenCount)
        {
            FinishReason const reason = context.slotStreams[slot].terminalReason;
            ELLM_CHECK(reason == FinishReason::kCancelled || reason == FinishReason::kError,
                "Managed context-cache decode: only a cancelled or failed slot may advance by zero tokens");
            // The hold sentinel, not reconstructed arithmetic: a slot that failed before its
            // admission was finalized (terminal from birth, kError) has no generate length the
            // committed value could be rebuilt from, and the ledger holds the truth either way.
            progress.push_back(
                ContextCacheSequenceAdvance{nullptr, 0, ContextCacheSequenceAdvance::kHoldCommittedStateLength});
            continue;
        }
        ELLM_CHECK(!context.tokenIds[slot].empty() && context.currentGenerateLengths[slot] > 0,
            "Managed context-cache decode did not produce a sampled lookahead token");
        ELLM_CHECK(context.tokenIds[slot].size() > previousTokenCount
                && context.tokenIds[slot].size() - previousTokenCount
                    <= static_cast<size_t>(std::numeric_limits<int32_t>::max()),
            "Managed context-cache decode produced an invalid accepted-token delta");
        int32_t const acceptedTokenCount = static_cast<int32_t>(context.tokenIds[slot].size() - previousTokenCount);
        int64_t const committedStateLength = static_cast<int64_t>(context.rawBatchedInputIds[slot].size())
            + static_cast<int64_t>(context.currentGenerateLengths[slot])
            - (mTokenStateContract == DecodingTokenStateContract::kFullyCommitted ? 0 : 1);
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

bool ContextCacheRequest::completeBatchCompaction(std::vector<int32_t> const& keepMapping)
{
    if (!contextCacheOperationSucceeded(mCoordinator.compactBatch(mRequest), "batch compaction"))
    {
        return false;
    }
    // The coordinator compacted its own per-sequence state; this runtime-side mirror of the
    // reused-prefix lengths must move with it, or reuseTokenLength(slot) reads an evicted
    // sequence's prefix after the first eviction.
    rt::compactVector(keepMapping, mPrefillStarts);
    return true;
}

bool ContextCacheRequest::finish()
{
    ELLM_CHECK(
        !mTokenCountsBeforeDecode.has_value(), "Managed context-cache request cannot finish during a decode step.");
    return contextCacheOperationSucceeded(mCoordinator.finish(mRequest), "request finish");
}

} // namespace rt
} // namespace trt_edgellm
