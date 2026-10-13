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
namespace rt
{

struct DecodingInferenceContext;
struct LLMGenerationRequest;
class Tensor;

//! Owns one admitted runtime request and binds it to the context-cache coordinator lifecycle.
class ContextCacheRequest final
{
public:
    //! Admit one tokenized request and bind its cache resources.
    //! A disengaged result means admission failed.
    //! @param mediaTokenIds Placeholder token IDs for media modalities (e.g. image, audio).
    //!        Positions matching any of these IDs are content-hashed for cache differentiation.
    //! @throws std::runtime_error if a media position has to be hashed and its pixels are not readable
    //!         on the host.
    static std::optional<ContextCacheRequest> begin(ContextCacheCoordinator& coordinator,
        LLMGenerationRequest const& request, DecodingInferenceContext const& context, bool speculativeRequest,
        DecodingKvHeadroom const& headroom, std::vector<int32_t> const& mediaTokenIds = {},
        DecodingTokenStateContract tokenStateContract = DecodingTokenStateContract::kCommittedPlusLookahead,
        ContextCacheCommitPolicy commitPolicy = ContextCacheCommitPolicy::kIncludingGeneratedTokens);

    ContextCacheRequest(ContextCacheRequest&&) noexcept = default;
    ContextCacheRequest& operator=(ContextCacheRequest&&) = delete;
    ContextCacheRequest(ContextCacheRequest const&) = delete;
    ContextCacheRequest& operator=(ContextCacheRequest const&) = delete;
    ~ContextCacheRequest() = default;

    //! Per-sequence logical offsets at which runtime prefill begins.
    std::vector<int32_t> const& prefillStarts() const noexcept;

    //! Number of reused prefix tokens for a slot (== the logical prefill start offset).
    int32_t reuseTokenLength(int32_t slot) const noexcept;

    enum class AdmitSequenceStatus : uint8_t
    {
        kAdmitted,
        //! Transient pool pressure: pages free as resident sequences retire, so the caller may
        //! retry at a later step boundary rather than failing the request.
        kNoCapacity,
        kFailed,
    };

    //! Join one more text-only sequence to this live request: lookup, lease, and row binding.
    //! On kAdmitted, @p prefillStart receives the reused prefix length the seated prefill skips.
    AdmitSequenceStatus admitSequence(std::vector<int32_t> const& tokenIds, std::string const& loraWeightsName,
        DecodingKvHeadroom const& headroom, int32_t& prefillStart, ResidentRef resident, cudaStream_t stream,
        std::vector<int32_t> const& mediaTokenIds = {}, std::vector<imageUtils::ImageData> const& imageBuffers = {},
        std::vector<audioUtils::AudioData> const& audioBuffers = {});

    //! Record the seated prefill's lookahead token so the slot's ledger matches its pages, and
    //! publish the ready prefix blocks. Call once, right after the seated prefill succeeds.
    bool finalizeSequenceAdmission(int32_t slot, int32_t const& lookaheadToken, int32_t fullInputLength);

    //! Undo the most recent admitSequence before its slot ever joined the runtime batch: the
    //! recovery path for a seating that threw between lease and slot append.
    bool retractSequenceAdmission() noexcept;

    //! Publish one Hybrid+MTP checkpoint at the stable predecessor boundary. Forwards to the coordinator's dedicated
    //! MTP publication entrypoint; the runtime drives this after the folded draft prefill materialized boundary state.
    bool publishHybridMtpEndpoint(
        int32_t slot, int32_t residentStateLength, Tensor const& baseHiddenStates, int32_t boundaryHiddenRow);
    //! Restore the reused checkpoint's saved boundary base-hidden row into baseHiddenStates for the fold micro-forward.
    bool restoreHybridMtpBoundaryHidden(int32_t slot, Tensor& baseHiddenStates, int32_t destinationRow);

    bool preparePrefill();
    bool enqueuePrefillCaptures();
    bool completePrefill(DecodingInferenceContext const& context, std::vector<int32_t> const& commonStateLengths);

    bool prepareDecodeStep(DecodingInferenceContext const& context, DecodingKvHeadroom const& headroom);
    bool completeDecodeStep(DecodingInferenceContext const& context, std::vector<int32_t> const& commonStateLengths);

    bool beginBatchCompaction(std::vector<int32_t> const& oldToNew, int32_t newBatchSize, Tensor& deviceBatchMapping);
    //! Compact the per-slot reuse bookkeeping to survivors. @p keepMapping is the runtime's
    //! oldSlot -> newSlot batch mapping (-1 for an evicted slot), the same one performBatchEvict
    //! uses; without it the reuse-length vector desyncs from the batch after the first eviction.
    bool completeBatchCompaction(std::vector<int32_t> const& keepMapping);

    bool finish();

private:
    ContextCacheRequest(ContextCacheCoordinator& coordinator, ContextCacheCoordinator::AdmissionResult&& admission,
        DecodingTokenStateContract tokenStateContract) noexcept;

    ContextCacheCoordinator& mCoordinator;
    ContextCacheCoordinator::RequestHandle mRequest;
    DecodingTokenStateContract mTokenStateContract{DecodingTokenStateContract::kCommittedPlusLookahead};
    std::vector<int32_t> mPrefillStarts;
    std::optional<std::vector<std::size_t>> mTokenCountsBeforeDecode;
};

} // namespace rt
} // namespace trt_edgellm
