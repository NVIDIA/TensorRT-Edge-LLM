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

#include "runtime/state/contextCache/contextCacheCoordinator.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace trt_edgellm
{
namespace chat_template
{
class ChatTemplate;
}
namespace tokenizer
{
class Tokenizer;
}
namespace rt
{

struct DecodingInferenceContext;
struct LLMGenerationRequest;
class Tensor;

//! One maximal span of a media placeholder token in a prompt: one media item, in prompt order. A
//! transition between placeholder tokens (an image run directly followed by an audio run) closes
//! the run, as the context cache's per-position media hash keys it.
struct MediaPlaceholderRun
{
    int32_t tokenId{};
    size_t begin{};
    size_t end{};
};

//! The media runs of @p tokenIds, for the deployment's placeholder tokens (LLMRankRuntime::mediaTokenIds()).
std::vector<MediaPlaceholderRun> mediaPlaceholderRuns(
    std::vector<int32_t> const& tokenIds, std::vector<int32_t> const& mediaTokenIds);

//! How many media items a prompt's placeholders stand for, per modality: the image token is
//! mediaTokenIds().front(), every other placeholder is audio.
struct MediaPlaceholderCounts
{
    size_t images{};
    size_t audios{};
};
MediaPlaceholderCounts countMediaPlaceholders(
    std::vector<int32_t> const& tokenIds, std::vector<int32_t> const& mediaTokenIds);

//! Derive the unstable generation-prompt suffix for Hybrid+spec context reuse: the tokens of @p promptTokenIds
//! after the prefix a committed assistant turn renders the same way (plus the one successor token MTP needs).
//!
//! This deliberately renders a committed assistant probe in addition to the already prepared generation prompt.
//! Comparing those token streams identifies the stable prefix that can remain cached. Used only when the caller
//! requests the -1 automatic-derivation sentinel.
int32_t deriveContextCacheReplayTailLength(chat_template::ChatTemplate const& chatTemplate,
    tokenizer::Tokenizer const& tokenizer, LLMGenerationRequest const& request,
    std::vector<int32_t> const& promptTokenIds);

//! Owns the cache-side working set of one batch -- the sequences a blocking request brought with it, or
//! the ones a long-lived session admits one at a time -- and binds it to the context-cache coordinator
//! lifecycle.
class ContextCacheRequest final
{
public:
    //! Open a batch. The admission names what the deployment fixes for every sequence the batch will
    //! hold, and the sequences it starts with, if any; a session starts with none and takes each one
    //! through admitSequence(). The result is left admitted; preparePrefill() enters execution.
    //! A disengaged result means admission failed. Throws the EDGELLM_INPUT_TOO_LONG error when a
    //! sequence's uncached suffix exceeds the engine input limit; nothing is leased or evicted for that sequence.
    static std::optional<ContextCacheRequest> begin(ContextCacheCoordinator& coordinator,
        ContextCacheBatchAdmission const& admission, DecodingKvHeadroom const& headroom, cudaStream_t stream);

    //! Open a batch holding one tokenized request's sequences, under the request's cache policies.
    //! A disengaged result means admission failed.
    //! @param mediaTokenIds Placeholder token IDs for media modalities (e.g. image, audio).
    //!        Positions matching any of these IDs key on the media's content identity: an image's from
    //!        @ref DecodingInferenceContext::imageContentDigests, an audio clip's from its PCM bytes, or
    //!        its mel-spectrogram path when it carries none. A request with an image that has no identity
    //!        bypasses the cache.
    static std::optional<ContextCacheRequest> begin(ContextCacheCoordinator& coordinator,
        LLMGenerationRequest const& request, DecodingInferenceContext const& context, bool speculativeRequest,
        DecodingKvHeadroom const& headroom, std::vector<int32_t> const& mediaTokenIds = {},
        DecodingTokenStateContract tokenStateContract = DecodingTokenStateContract::kCommittedPlusLookahead,
        ContextCacheCommitPolicy commitPolicy = ContextCacheCommitPolicy::kIncludingGeneratedTokens);

    //! The lookup policy a request runs under: its own, unless something forces bypass -- speech
    //! output, thinker hidden-state capture, or an image without a content identity (which would key
    //! on the media token alone, every image's placeholder). @p mediaTokenIds is the deployment's
    //! media placeholder list; empty means the deployment hashes no media. @p imageDigests holds one
    //! identity per image of each sequence, as multimodal preprocessing computed them.
    static ContextCacheLookupPolicy lookupPolicyFor(LLMGenerationRequest const& request, bool outputThinkerEmbeddings,
        std::vector<int32_t> const& mediaTokenIds, std::vector<std::vector<Hash128>> const& imageDigests);

    //! The policy one sequence of a request runs under: the request's (lookupPolicyFor), which the
    //! sequence's own @p sequencePolicy may narrow to bypass but never widen -- a request-wide
    //! bypass binds every sequence.
    static ContextCacheLookupPolicy sequenceLookupPolicy(
        ContextCacheLookupPolicy requestPolicy, std::optional<ContextCacheLookupPolicy> sequencePolicy) noexcept;

    ContextCacheRequest(ContextCacheRequest&&) noexcept = default;
    ContextCacheRequest& operator=(ContextCacheRequest&&) = delete;
    ContextCacheRequest(ContextCacheRequest const&) = delete;
    ContextCacheRequest& operator=(ContextCacheRequest const&) = delete;
    ~ContextCacheRequest() = default;

    //! Per-sequence logical offsets at which runtime prefill begins.
    std::vector<int32_t> const& prefillStarts() const noexcept;

    //! Number of reused prefix tokens for a slot (== the logical prefill start offset).
    int32_t reuseTokenLength(int32_t slot) const noexcept;
    std::vector<SwaKVCacheState> takeSwaStates();
    bool isExternallyOwnedSwaPage(int32_t slot, PageId page) const;
    PageId replaceSwaPage(int32_t slot, PageId page);

    enum class AdmitSequenceStatus : uint8_t
    {
        kAdmitted,
        //! Transient pool pressure: pages free as resident sequences retire, so the caller may
        //! retry at a later step boundary rather than failing the request.
        kNoCapacity,
        //! The uncached suffix exceeds the engine input limit. Nothing was leased or evicted.
        kOverlongPrefill,
        kFailed,
    };

    //! Join one more sequence to this live request: lookup, lease, and row binding.
    //! On kAdmitted, @p prefillStart receives the reused prefix length the seated prefill skips;
    //! on kOverlongPrefill, the reused prefix length the lookup found.
    //! @p imageContentDigests holds one identity per image of the sequence, in prompt order; its media
    //! positions key on them the same way begin() keys the first sequences'.
    //! @p lookupPolicy and @p commitPolicy are the sequence's own; the caller has already applied
    //! whatever forces bypass (see lookupPolicyFor).
    AdmitSequenceStatus admitSequence(std::vector<int32_t> const& tokenIds, std::string const& loraWeightsName,
        DecodingKvHeadroom const& headroom, int32_t& prefillStart, ResidentRef resident, cudaStream_t stream,
        ContextCacheLookupPolicy lookupPolicy, ContextCacheCommitPolicy commitPolicy,
        std::vector<int32_t> const& mediaTokenIds = {}, std::vector<Hash128> const& imageContentDigests = {},
        std::vector<audioUtils::AudioData> const& audioBuffers = {});

    //! @name Sequence-admission transaction (see ContextCacheCoordinator). Several sequences may be in
    //! admission at once; each prefill chunk runs prepare -> (final chunk: capture) -> stream sync ->
    //! complete -> advance (non-final) or finalize (final).
    //! @{

    //! Open the sequence's next prefill chunk; the first one restores any hybrid checkpoint into its resident slot.
    bool prepareSequenceAdmission(int32_t sequenceIndex);

    //! Enqueue the sequence's prompt-end hybrid checkpoint capture after its final chunk, before the stream sync.
    bool enqueueSequenceAdmissionCapture(int32_t sequenceIndex);

    //! Record that the runtime's stream synchronization completed the sequence's open chunk.
    bool completeSequenceAdmissionDeviceWork(int32_t sequenceIndex);

    //! Record a committed non-final chunk: the slot holds the first @p committedLength prompt tokens.
    //! Nothing is published.
    bool advanceSequenceAdmission(int32_t sequenceIndex, int32_t committedLength);

    //! Discard a failed or cancelled sequence-admission transaction before evicting its slot; nothing
    //! of the sequence is published.
    bool abortSequenceAdmission(int32_t sequenceIndex);

    //! Record the final chunk's lookahead token so the slot's ledger matches its cached state, then
    //! publish the ready prefix. Call once, right after the final chunk succeeds. @p commonStateLength is the
    //! prefix a speculative sequence's draft has seated (see ContextCacheCoordinator::finalizeSequenceAdmission).
    bool finalizeSequenceAdmission(int32_t sequenceIndex, int32_t const& lookaheadToken, int32_t fullInputLength,
        std::optional<int32_t> commonStateLength = std::nullopt);
    //! @}

    //! Undo the most recent admitSequence before its slot ever joined the runtime batch: the
    //! recovery path for a seating that threw between lease and slot append.
    bool retractSequenceAdmission() noexcept;

    //! Publish one Hybrid+spec checkpoint at the stable predecessor boundary. Forwards to the coordinator's dedicated
    //! MTP publication entrypoint; the runtime drives this after the folded draft prefill materialized boundary state.
    bool publishHybridSpecEndpoint(
        int32_t slot, int32_t residentStateLength, Tensor const& baseHiddenStates, int32_t boundaryHiddenRow);
    //! Restore the reused checkpoint's saved boundary base-hidden row into baseHiddenStates for the fold micro-forward.
    bool restoreHybridSpecBoundaryHidden(int32_t slot, Tensor& baseHiddenStates, int32_t destinationRow);

    bool preparePrefill();
    bool enqueuePrefillCaptures();
    bool completePrefill(DecodingInferenceContext const& context, std::vector<int32_t> const& commonStateLengths,
        SwaWindowStateView const* swaWindowStates = nullptr);

    //! @p extraTokensPerSlot / @p draftExtraTokensPerSlot: each row's own base and draft growth when jump-forward
    //! changes it (see the coordinator).
    bool prepareDecodeStep(DecodingInferenceContext const& context, DecodingKvHeadroom const& headroom,
        std::vector<int32_t> const* extraTokensPerSlot = nullptr,
        std::vector<int32_t> const* draftExtraTokensPerSlot = nullptr);
    //! Prepare a step in which only @p decodeRows (execution rows with a complete prompt) decode. The
    //! matching completeDecodeStep gives every other row the hold advance and publishes none of them,
    //! whatever the row appended this step (a final prefill chunk's first token) or not (a paused row,
    //! a non-final chunk).
    bool prepareDecodeStep(DecodingInferenceContext const& context, DecodingKvHeadroom const& headroom,
        std::vector<int32_t> const& decodeRows);
    bool completeDecodeStep(DecodingInferenceContext const& context, std::vector<int32_t> const& commonStateLengths,
        SwaWindowStateView const* swaWindowStates = nullptr);

    bool beginBatchCompaction(std::vector<int32_t> const& oldToNew, int32_t newBatchSize, Tensor& deviceBatchMapping);
    //! Compact the per-slot reuse bookkeeping to survivors. @p keepMapping is the runtime's
    //! oldSlot -> newSlot batch mapping (-1 for an evicted slot), the same one performBatchEvict
    //! uses; without it the reuse-length vector desyncs from the batch after the first eviction.
    bool completeBatchCompaction(std::vector<int32_t> const& keepMapping);

    bool finish();

private:
    ContextCacheRequest(ContextCacheCoordinator& coordinator, ContextCacheCoordinator::AdmissionResult&& admission,
        DecodingTokenStateContract tokenStateContract) noexcept;
    void snapshotTokenCountsBeforeDecode(DecodingInferenceContext const& context);

    ContextCacheCoordinator& mCoordinator;
    ContextCacheCoordinator::RequestHandle mRequest;
    DecodingTokenStateContract mTokenStateContract{DecodingTokenStateContract::kCommittedPlusLookahead};
    std::vector<int32_t> mPrefillStarts;
    std::vector<SwaKVCacheState> mSwaStates;
    //! Between prepareDecodeStep() and completeDecodeStep(): the two buffers below describe the prepared step. They
    //! keep their capacity from step to step.
    bool mDecodeStepPrepared{false};
    std::vector<std::size_t> mTokenCountsBeforeDecode;
    //! Rows that decode in the prepared step; empty means every row.
    std::vector<uint8_t> mDecodeRowsInStep;
};

} // namespace rt
} // namespace trt_edgellm
