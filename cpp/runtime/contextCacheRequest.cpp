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

#include "chatTemplate/chatTemplate.h"
#include "common/checkMacros.h"
#include "common/logger.h"
#include "runtime/audioUtils.h"
#include "runtime/imageUtils.h"
#include "runtime/llmRuntimeUtils.h"
#include "runtime/state/contextCache/blockHash.h"
#include "runtime/state/decodingInferenceContext.h"
#include "runtime/streaming.h"
#include "tokenizer/tokenizer.h"

#include <algorithm>
#include <exception>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace trt_edgellm
{
namespace rt
{
namespace
{

//! Whether `imageDigests` carries one identity per image of `request`, sequence by sequence; a request
//! with no images qualifies trivially.
bool imagesAreIdentified(LLMGenerationRequest const& request, std::vector<std::vector<Hash128>> const& imageDigests)
{
    bool const hasImages = std::any_of(request.requests.begin(), request.requests.end(),
        [](auto const& sequence) { return !sequence.imageBuffers.empty(); });
    if (!hasImages)
    {
        return true;
    }
    if (imageDigests.size() != request.requests.size())
    {
        return false;
    }
    for (size_t seqIdx = 0; seqIdx < request.requests.size(); ++seqIdx)
    {
        if (imageDigests[seqIdx].size() != request.requests[seqIdx].imageBuffers.size())
        {
            return false;
        }
    }
    return true;
}

} // namespace

ContextCacheLookupPolicy ContextCacheRequest::lookupPolicyFor(LLMGenerationRequest const& request,
    bool outputThinkerEmbeddings, std::vector<int32_t> const& mediaTokenIds,
    std::vector<std::vector<Hash128>> const& imageDigests)
{
    // A media position keys on the identity multimodal preprocessing computed for its image; one with no
    // identity would key on the media token alone, which is every image's placeholder.
    bool const mediaUnreadable = !mediaTokenIds.empty() && !imagesAreIdentified(request, imageDigests);
    bool const requiresBypass = request.contextCacheLookupPolicy == ContextCacheLookupPolicy::kBypass
        || request.generateAudio || outputThinkerEmbeddings || mediaUnreadable;
    return requiresBypass ? ContextCacheLookupPolicy::kBypass : ContextCacheLookupPolicy::kUseCache;
}

ContextCacheLookupPolicy ContextCacheRequest::sequenceLookupPolicy(
    ContextCacheLookupPolicy requestPolicy, std::optional<ContextCacheLookupPolicy> sequencePolicy) noexcept
{
    return requestPolicy == ContextCacheLookupPolicy::kBypass ? ContextCacheLookupPolicy::kBypass
                                                              : sequencePolicy.value_or(requestPolicy);
}

std::vector<MediaPlaceholderRun> mediaPlaceholderRuns(
    std::vector<int32_t> const& tokenIds, std::vector<int32_t> const& mediaTokenIds)
{
    std::vector<MediaPlaceholderRun> runs;
    if (mediaTokenIds.empty())
    {
        return runs;
    }
    auto const isMediaToken = [&mediaTokenIds](int32_t tokenId) {
        return std::find(mediaTokenIds.begin(), mediaTokenIds.end(), tokenId) != mediaTokenIds.end();
    };
    for (size_t i = 0; i < tokenIds.size(); ++i)
    {
        int32_t const token = tokenIds[i];
        if (!isMediaToken(token))
        {
            continue;
        }
        // A run is a maximal span of one placeholder token: a transition between media types
        // (an image run directly followed by an audio run) closes the previous run.
        if (!runs.empty() && runs.back().end == i && runs.back().tokenId == token)
        {
            runs.back().end = i + 1;
        }
        else
        {
            runs.push_back(MediaPlaceholderRun{token, i, i + 1});
        }
    }
    return runs;
}

MediaPlaceholderCounts countMediaPlaceholders(
    std::vector<int32_t> const& tokenIds, std::vector<int32_t> const& mediaTokenIds)
{
    MediaPlaceholderCounts counts;
    for (MediaPlaceholderRun const& run : mediaPlaceholderRuns(tokenIds, mediaTokenIds))
    {
        (run.tokenId == mediaTokenIds.front() ? counts.images : counts.audios) += 1;
    }
    return counts;
}

namespace
{

std::vector<Hash128> buildPerPositionMediaHash(std::vector<int32_t> const& tokenIds,
    std::vector<int32_t> const& mediaTokenIds, std::vector<Hash128> const& imageContentDigests,
    std::vector<audioUtils::AudioData> const& audioBuffers, cudaStream_t stream)
{
    if (mediaTokenIds.empty())
    {
        return {};
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

    // The deployment's media placeholders, in mediaTokenIds() order: the image token first, the
    // audio token second. Every run of one placeholder is one media item, keyed by its content.
    int32_t const imageTokenId = mediaTokenIds[0];
    std::vector<Hash128> perPositionHash(tokenIds.size(), Hash128{});
    size_t imageIdx = 0;
    size_t audioIdx = 0;
    for (MediaPlaceholderRun const& run : mediaPlaceholderRuns(tokenIds, mediaTokenIds))
    {
        Hash128 hash;
        if (run.tokenId == imageTokenId)
        {
            ELLM_CHECK(imageIdx < imageContentDigests.size(),
                "Image token placeholder at position " + std::to_string(run.begin) + " exceeds provided image count ("
                    + std::to_string(imageContentDigests.size()) + ")");
            hash = imageContentDigests[imageIdx++];
        }
        else
        {
            ELLM_CHECK(audioIdx < audioHashes.size(),
                "Audio token placeholder at position " + std::to_string(run.begin) + " exceeds provided audio count ("
                    + std::to_string(audioHashes.size()) + ")");
            hash = audioHashes[audioIdx++];
        }
        std::fill(perPositionHash.begin() + static_cast<ptrdiff_t>(run.begin),
            perPositionHash.begin() + static_cast<ptrdiff_t>(run.end), hash);
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
    std::vector<Hash128> const& imageContentDigests, std::vector<audioUtils::AudioData> const& audioBuffers,
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
        = buildPerPositionMediaHash(tokenIds, mediaTokenIds, imageContentDigests, audioBuffers, stream);
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
    ContextCacheBatchAdmission const& admission, DecodingKvHeadroom const& headroom, cudaStream_t stream)
{
    ContextCacheCoordinator::BeginRequestResult opened = coordinator.beginRequest(admission, headroom, stream);
    if (opened.overlongPrefillLength.has_value())
    {
        int32_t const prefillLength = *opened.overlongPrefillLength;
        LOG_ERROR(
            "Prefill length after context reuse (%d) exceeds engine max input length (%d). "
            "Rebuild the engine with a larger --max-input-len.",
            prefillLength, coordinator.maxPrefillLength());
        throw std::runtime_error("EDGELLM_INPUT_TOO_LONG: prefill length " + std::to_string(prefillLength)
            + " exceeds engine max_input_len " + std::to_string(coordinator.maxPrefillLength())
            + " (rebuild engine with a larger --max-input-len)");
    }
    if (!contextCacheOperationSucceeded(opened.status, "admission") || !opened.admission.has_value())
    {
        return std::nullopt;
    }
    return ContextCacheRequest{coordinator, std::move(*opened.admission), admission.tokenStateContract};
}

std::optional<ContextCacheRequest> ContextCacheRequest::begin(ContextCacheCoordinator& coordinator,
    LLMGenerationRequest const& request, DecodingInferenceContext const& context, bool speculativeRequest,
    DecodingKvHeadroom const& headroom, std::vector<int32_t> const& mediaTokenIds,
    DecodingTokenStateContract tokenStateContract, ContextCacheCommitPolicy commitPolicy)
{
    static std::vector<Hash128> const kEmptyImageContentDigests;
    static std::vector<audioUtils::AudioData> const kEmptyAudioBuffers;
    static std::vector<int32_t> const kEmptyMediaTokenIds;

    ContextCacheBatchAdmission admission;
    admission.speculativeRequest = speculativeRequest;
    admission.tokenStateContract = tokenStateContract;
    // The request's lookup policy with everything that forces bypass applied. A bypass here binds
    // every sequence; a per-sequence override can add a bypass, never undo one.
    ContextCacheLookupPolicy const requestLookupPolicy
        = lookupPolicyFor(request, context.outputThinkerEmbeddings, mediaTokenIds, context.imageContentDigests);
    ELLM_CHECK(context.residentRefs.size() == context.rawBatchedInputIds.size(),
        "Context-cache admission requires one resident identity per input sequence");
    admission.sequences.reserve(context.rawBatchedInputIds.size());
    for (size_t seqIdx = 0; seqIdx < context.rawBatchedInputIds.size(); ++seqIdx)
    {
        ContextCacheLookupPolicy const effectiveLookupPolicy = sequenceLookupPolicy(requestLookupPolicy,
            (seqIdx < request.requests.size()) ? request.requests[seqIdx].contextCacheLookupPolicy : std::nullopt);
        std::vector<Hash128> const& imageContentDigests = (seqIdx < context.imageContentDigests.size())
            ? context.imageContentDigests[seqIdx]
            : kEmptyImageContentDigests;
        std::vector<audioUtils::AudioData> const& audio
            = (seqIdx < request.requests.size()) ? request.requests[seqIdx].audioBuffers : kEmptyAudioBuffers;
        ContextCacheSequenceAdmission sequence
            = makeContextCacheSequenceAdmission(context.rawBatchedInputIds[seqIdx], context.loraWeightsName,
                effectiveLookupPolicy == ContextCacheLookupPolicy::kUseCache ? mediaTokenIds : kEmptyMediaTokenIds,
                imageContentDigests, audio, context.stream);
        sequence.resident = context.residentRefs[seqIdx];
        sequence.lookupPolicy = effectiveLookupPolicy;
        sequence.commitPolicy = commitPolicy;
        sequence.replayTailLength = request.contextCacheReplayTailLength;
        admission.sequences.push_back(std::move(sequence));
    }
    return begin(coordinator, admission, headroom, context.stream);
}

ContextCacheRequest::ContextCacheRequest(ContextCacheCoordinator& coordinator,
    ContextCacheCoordinator::AdmissionResult&& admission, DecodingTokenStateContract tokenStateContract) noexcept
    : mCoordinator(coordinator)
    , mRequest(std::move(admission.request))
    , mTokenStateContract(tokenStateContract)
    , mPrefillStarts(std::move(admission.prefillStarts))
    , mSwaStates(std::move(admission.swaStates))
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
    cudaStream_t stream, ContextCacheLookupPolicy lookupPolicy, ContextCacheCommitPolicy commitPolicy,
    std::vector<int32_t> const& mediaTokenIds, std::vector<Hash128> const& imageContentDigests,
    std::vector<audioUtils::AudioData> const& audioBuffers)
{
    mPrefillStarts.reserve(mPrefillStarts.size() + 1);
    // A bypassed lookup hashes no media: the pages are private, so keying them costs only host time.
    bool const hashesMedia = lookupPolicy != ContextCacheLookupPolicy::kBypass;
    static std::vector<int32_t> const kNoMediaTokenIds;
    ContextCacheSequenceAdmission admission = makeContextCacheSequenceAdmission(tokenIds, loraWeightsName,
        hashesMedia ? mediaTokenIds : kNoMediaTokenIds, imageContentDigests, audioBuffers, stream);
    admission.resident = resident;
    admission.lookupPolicy = lookupPolicy;
    admission.commitPolicy = commitPolicy;
    ContextCacheCoordinator::AdmitSequenceResult result = mCoordinator.admitSequence(mRequest, admission, headroom);
    if (result.status != ContextCacheCoordinatorStatus::kOk)
    {
        if (result.insufficientCapacity)
        {
            return AdmitSequenceStatus::kNoCapacity;
        }
        if (result.overlongPrefill)
        {
            prefillStart = result.prefillStart;
            return AdmitSequenceStatus::kOverlongPrefill;
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

bool ContextCacheRequest::prepareSequenceAdmission(int32_t sequenceIndex)
{
    return contextCacheOperationSucceeded(
        mCoordinator.prepareSequenceAdmission(mRequest, sequenceIndex), "sequence-admission preparation");
}

bool ContextCacheRequest::enqueueSequenceAdmissionCapture(int32_t sequenceIndex)
{
    return contextCacheOperationSucceeded(
        mCoordinator.enqueueSequenceAdmissionCapture(mRequest, sequenceIndex), "sequence-admission capture");
}

bool ContextCacheRequest::completeSequenceAdmissionDeviceWork(int32_t sequenceIndex)
{
    return contextCacheOperationSucceeded(mCoordinator.completeSequenceAdmissionDeviceWork(mRequest, sequenceIndex),
        "sequence-admission device-work completion");
}

bool ContextCacheRequest::advanceSequenceAdmission(int32_t sequenceIndex, int32_t committedLength)
{
    return contextCacheOperationSucceeded(
        mCoordinator.advanceSequenceAdmission(mRequest, sequenceIndex, committedLength), "sequence-admission advance");
}

bool ContextCacheRequest::abortSequenceAdmission(int32_t sequenceIndex)
{
    return contextCacheOperationSucceeded(
        mCoordinator.abortSequenceAdmission(mRequest, sequenceIndex), "sequence-admission abort");
}

bool ContextCacheRequest::finalizeSequenceAdmission(int32_t sequenceIndex, int32_t const& lookaheadToken,
    int32_t fullInputLength, std::optional<int32_t> commonStateLength)
{
    return contextCacheOperationSucceeded(
        mCoordinator.finalizeSequenceAdmission(mRequest, sequenceIndex,
            ContextCacheSequenceAdvance{&lookaheadToken, 1, fullInputLength}, commonStateLength),
        "sequence-admission finalization");
}

std::vector<SwaKVCacheState> ContextCacheRequest::takeSwaStates()
{
    return std::exchange(mSwaStates, {});
}

bool ContextCacheRequest::isExternallyOwnedSwaPage(int32_t slot, PageId page) const
{
    return mCoordinator.isExternallyOwnedSwaPage(mRequest, slot, page);
}

PageId ContextCacheRequest::replaceSwaPage(int32_t slot, PageId page)
{
    return mCoordinator.replaceSwaPage(mRequest, slot, page);
}

bool ContextCacheRequest::publishHybridSpecEndpoint(
    int32_t slot, int32_t residentStateLength, Tensor const& baseHiddenStates, int32_t boundaryHiddenRow)
{
    return contextCacheOperationSucceeded(mCoordinator.publishHybridSpecEndpoint(
                                              mRequest, slot, residentStateLength, baseHiddenStates, boundaryHiddenRow),
        "Hybrid+spec endpoint publication");
}

bool ContextCacheRequest::restoreHybridSpecBoundaryHidden(
    int32_t slot, Tensor& baseHiddenStates, int32_t destinationRow)
{
    return contextCacheOperationSucceeded(
        mCoordinator.restoreHybridSpecBoundaryHidden(mRequest, slot, baseHiddenStates, destinationRow),
        "Hybrid+spec boundary-hidden restore");
}
bool ContextCacheRequest::preparePrefill()
{
    return contextCacheOperationSucceeded(mCoordinator.preparePrefill(mRequest), "prefill preparation");
}

bool ContextCacheRequest::enqueuePrefillCaptures()
{
    return contextCacheOperationSucceeded(mCoordinator.enqueuePrefillCaptures(mRequest), "prefill snapshot capture");
}

bool ContextCacheRequest::completePrefill(DecodingInferenceContext const& context,
    std::vector<int32_t> const& commonStateLengths, SwaWindowStateView const* swaWindowStates)
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
        mCoordinator.finalizePrefillPublication(mRequest, progress, commonStateLengthsPtr, swaWindowStates),
        "prefill publication");
}

bool ContextCacheRequest::prepareDecodeStep(DecodingInferenceContext const& context, DecodingKvHeadroom const& headroom,
    std::vector<int32_t> const* extraTokensPerSlot, std::vector<int32_t> const* draftExtraTokensPerSlot)
{
    ELLM_CHECK(!mDecodeStepPrepared, "Managed context-cache decode preparation cannot overlap a pending decode step.");
    if (!contextCacheOperationSucceeded(
            mCoordinator.prepareDecodeStep(mRequest, headroom, extraTokensPerSlot, draftExtraTokensPerSlot),
            "decode preparation"))
    {
        return false;
    }
    mDecodeRowsInStep.clear();
    snapshotTokenCountsBeforeDecode(context);
    return true;
}

bool ContextCacheRequest::prepareDecodeStep(
    DecodingInferenceContext const& context, DecodingKvHeadroom const& headroom, std::vector<int32_t> const& decodeRows)
{
    ELLM_CHECK(!mDecodeStepPrepared, "Managed context-cache decode preparation cannot overlap a pending decode step.");
    for (int32_t const row : decodeRows)
    {
        ELLM_CHECK(row >= 0 && row < context.activeBatchSize, "Managed context-cache decode row is outside the batch.");
    }
    if (!contextCacheOperationSucceeded(
            mCoordinator.prepareDecodeStep(mRequest, headroom, decodeRows), "decode-subset preparation"))
    {
        return false;
    }
    mDecodeRowsInStep.assign(static_cast<size_t>(context.activeBatchSize), 0U);
    for (int32_t const row : decodeRows)
    {
        mDecodeRowsInStep[static_cast<size_t>(row)] = 1U;
    }
    snapshotTokenCountsBeforeDecode(context);
    return true;
}

void ContextCacheRequest::snapshotTokenCountsBeforeDecode(DecodingInferenceContext const& context)
{
    mTokenCountsBeforeDecode.clear();
    for (int32_t slot = 0; slot < context.activeBatchSize; ++slot)
    {
        mTokenCountsBeforeDecode.push_back(context.tokenIds[slot].size());
    }
    mDecodeStepPrepared = true;
}

bool ContextCacheRequest::completeDecodeStep(DecodingInferenceContext const& context,
    std::vector<int32_t> const& commonStateLengths, SwaWindowStateView const* swaWindowStates)
{
    ELLM_CHECK(mDecodeStepPrepared, "Managed context-cache decode completion has no prepared decode step.");
    mDecodeStepPrepared = false;
    std::vector<size_t> const& tokenCountsBeforeDecode = mTokenCountsBeforeDecode;
    std::vector<uint8_t> const& decodeRowsInStep = mDecodeRowsInStep;
    ELLM_CHECK(static_cast<int32_t>(tokenCountsBeforeDecode.size()) == context.activeBatchSize,
        "Managed context-cache active batch changed during a decode step.");

    std::vector<ContextCacheSequenceAdvance> progress;
    std::vector<int32_t> publishableCompletedSlots;
    progress.reserve(static_cast<size_t>(context.activeBatchSize));
    publishableCompletedSlots.reserve(static_cast<size_t>(context.activeBatchSize));
    for (int32_t slot = 0; slot < context.activeBatchSize; ++slot)
    {
        if (!decodeRowsInStep.empty() && decodeRowsInStep[static_cast<size_t>(slot)] == 0U)
        {
            progress.push_back(
                ContextCacheSequenceAdvance{nullptr, 0, ContextCacheSequenceAdvance::kHoldCommittedStateLength});
            continue;
        }
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
        // The lookahead contract leaves the slot's pending tokens without KV: one, or more after a jump-forward.
        int32_t const pendingTokens = mTokenStateContract == DecodingTokenStateContract::kFullyCommitted
            ? 0
            : std::max(1, context.pendingTokens[slot]);
        int64_t const committedStateLength = static_cast<int64_t>(context.rawBatchedInputIds[slot].size())
            + static_cast<int64_t>(context.currentGenerateLengths[slot]) - pendingTokens;
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
    return contextCacheOperationSucceeded(mCoordinator.completeDecodeStep(mRequest, progress, publishableCompletedSlots,
                                              commonStateLengthsPtr, swaWindowStates),
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
    ELLM_CHECK(!mDecodeStepPrepared, "Managed context-cache request cannot finish during a decode step.");
    return contextCacheOperationSucceeded(mCoordinator.finish(mRequest), "request finish");
}

int32_t deriveContextCacheReplayTailLength(chat_template::ChatTemplate const& chatTemplate,
    tokenizer::Tokenizer const& tokenizer, LLMGenerationRequest const& request,
    std::vector<int32_t> const& promptTokenIds)
{
    ELLM_CHECK(request.requests.size() == 1,
        "Context-cache replay-tail derivation requires a batch containing exactly one request.");
    ELLM_CHECK(request.applyChatTemplate && request.addGenerationPrompt,
        "Context-cache replay-tail derivation requires the model chat template and generation prompt.");
    ELLM_CHECK(!promptTokenIds.empty(), "Cannot derive a context-cache replay tail from an empty prompt.");

    LLMGenerationRequest::Request committedRequest = request.requests.front();
    // A generation prompt may include suffix tokens that change once an
    // assistant turn is committed. Render that committed state once to find
    // the stable prefix that Hybrid+MTP context reuse may safely retain.
    Message assistantProbe;
    assistantProbe.role = "assistant";
    assistantProbe.contents.push_back(Message::MessageContent{"text", ""});
    assistantProbe.reasoningContent = "context reuse probe";
    assistantProbe.hasReasoningContent = true;
    committedRequest.messages.push_back(std::move(assistantProbe));

    auto options = chat_template::ChatTemplate::optionsFrom(request);
    options.addGenerationPrompt = false;
    LLMGenerationRequest::FormattedRequest committedFormatted;
    ELLM_CHECK(chatTemplate.apply(committedRequest, committedFormatted, options),
        "Failed to render the committed prompt while deriving the context-cache replay tail.");
    auto const committedTokenIds = tokenizer.encode(committedFormatted.formattedCompleteRequest, false);
    ELLM_CHECK(!committedTokenIds.empty(),
        "Failed to tokenize the committed prompt while deriving the context-cache replay tail.");

    size_t commonPrefixLength = 0;
    size_t const comparedLength = std::min(promptTokenIds.size(), committedTokenIds.size());
    while (commonPrefixLength < comparedLength
        && promptTokenIds[commonPrefixLength] == committedTokenIds[commonPrefixLength])
    {
        ++commonPrefixLength;
    }
    ELLM_CHECK(commonPrefixLength > 0, "Chat template has no stable token prefix for context reuse.");

    // MTP needs one stable successor token after the captured predecessor.
    size_t const replayTailLength = promptTokenIds.size() - commonPrefixLength + 1;
    ELLM_CHECK(replayTailLength <= static_cast<size_t>(std::numeric_limits<int32_t>::max()),
        "Context-cache replay tail exceeds the int32 range.");
    return static_cast<int32_t>(replayTailLength);
}

} // namespace rt
} // namespace trt_edgellm
