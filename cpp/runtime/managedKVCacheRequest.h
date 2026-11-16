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

#include "runtime/contextCacheRequest.h"
#include "runtime/state/boundedSwaKVPageManager.h"

#include <cstdint>
#include <optional>
#include <vector>

namespace trt_edgellm
{
namespace rt
{

//! One runtime request facade over orthogonal execution paging and context-reuse clients.
//!
//! Context reuse owns cache-record references while bounded SWA owns sparse-table transitions. Either client may be
//! active alone, or both may be composed for vanilla bounded-SWA context reuse.
class ManagedKVCacheRequest final
{
public:
    static std::optional<ManagedKVCacheRequest> begin(ContextCacheCoordinator* contextCache,
        BoundedSwaKVPageManager* swaPageManager, LLMGenerationRequest const& request,
        DecodingInferenceContext const& context, bool speculativeRequest, DecodingKvHeadroom const& headroom,
        DecodingTokenStateContract tokenStateContract, ContextCacheCommitPolicy commitPolicy,
        std::vector<int32_t> const& mediaTokenIds = {});

    //! Open a context-reuse batch from a batch admission (see ContextCacheRequest::begin): the form a
    //! session uses, starting with no sequence and admitting each one live. Bounded SWA has no
    //! live-slot admission, so a batch opened this way carries none; a session on such a deployment
    //! is refused before this is reached.
    static std::optional<ManagedKVCacheRequest> begin(ContextCacheCoordinator& contextCache,
        ContextCacheBatchAdmission const& admission, DecodingKvHeadroom const& headroom, cudaStream_t stream);

    ManagedKVCacheRequest(ManagedKVCacheRequest&&) noexcept = default;
    ManagedKVCacheRequest& operator=(ManagedKVCacheRequest&&) = delete;
    ManagedKVCacheRequest(ManagedKVCacheRequest const&) = delete;
    ManagedKVCacheRequest& operator=(ManagedKVCacheRequest const&) = delete;
    ~ManagedKVCacheRequest() = default;

    bool hasContextReuse() const noexcept;
    bool hasBoundedSwa() const noexcept;
    std::vector<int32_t> const* prefillStarts() const noexcept;
    int32_t reuseTokenLength(int32_t slot) const;

    bool publishHybridSpecEndpoint(
        int32_t slot, int32_t residentStateLength, Tensor const& baseHiddenStates, int32_t boundaryHiddenRow);
    bool restoreHybridSpecBoundaryHidden(int32_t slot, Tensor& baseHiddenStates, int32_t destinationRow);

    bool preparePrefill();
    bool prepareSwaPrefill(std::vector<int32_t> const& effectiveInputLengths);
    bool enqueuePrefillCaptures();
    bool completePrefill(DecodingInferenceContext const& context, std::vector<int32_t> const& commonStateLengths);
    //! Grow every slot for the next decode step. With jump-forward, a vanilla slot grows its pages and its SWA window
    //! by its own pending count instead of headroom.baseExtraTokens, and a speculative slot that asked for
    //! jump-forward grows by @p jumpForwardHeadroom instead of @p headroom.
    bool prepareDecodeStep(DecodingInferenceContext const& context, DecodingKvHeadroom const& headroom,
        DecodingKvHeadroom const& jumpForwardHeadroom);
    //! Decode-subset form (see ContextCacheRequest::prepareDecodeStep); context-reuse backend only.
    bool prepareDecodeStep(DecodingInferenceContext const& context, DecodingKvHeadroom const& headroom,
        std::vector<int32_t> const& decodeRows);
    bool completeDecodeStep(DecodingInferenceContext const& context, std::vector<int32_t> const& commonStateLengths);
    bool beginBatchCompaction(std::vector<int32_t> const& oldToNew, int32_t newBatchSize, Tensor& deviceBatchMapping);
    bool completeBatchCompaction(std::vector<int32_t> const& keepMapping);
    bool finish();

    //! @name In-flight admission, context-reuse backend only.
    //! The bounded-SWA backend has no live-slot admission (its per-slot window state cannot join a
    //! running batch), and supportsSeatedAdmission() refuses such deployments before an engine that
    //! would call these is ever constructed; the kFailed/false returns here are the backstop.
    //! @{
    ContextCacheRequest::AdmitSequenceStatus admitSequence(std::vector<int32_t> const& tokenIds,
        std::string const& loraWeightsName, DecodingKvHeadroom const& headroom, int32_t& prefillStart,
        ResidentRef resident, cudaStream_t stream, ContextCacheLookupPolicy lookupPolicy,
        ContextCacheCommitPolicy commitPolicy, std::vector<int32_t> const& mediaTokenIds = {},
        std::vector<Hash128> const& imageContentDigests = {},
        std::vector<audioUtils::AudioData> const& audioBuffers = {});
    bool prepareSequenceAdmission(int32_t sequenceIndex);
    bool enqueueSequenceAdmissionCapture(int32_t sequenceIndex);
    bool completeSequenceAdmissionDeviceWork(int32_t sequenceIndex);
    bool advanceSequenceAdmission(int32_t sequenceIndex, int32_t committedLength);
    bool abortSequenceAdmission(int32_t sequenceIndex);
    bool finalizeSequenceAdmission(int32_t sequenceIndex, int32_t const& lookaheadToken, int32_t fullInputLength,
        std::optional<int32_t> commonStateLength = std::nullopt);
    bool retractSequenceAdmission() noexcept;
    //! @}

private:
    ManagedKVCacheRequest(std::optional<ContextCacheRequest>&& contextRequest, BoundedSwaKVPageManager* swaPageManager,
        std::optional<BoundedSwaKVPageManager::RequestHandle>&& swaRequest, bool speculativeRequest) noexcept;

    ContextCacheRequest& contextRequest();
    ContextCacheRequest const& contextRequest() const;
    BoundedSwaKVPageManager::RequestHandle& swaRequest();
    BoundedSwaKVPageManager::ExternalPageLifecycle swaPageLifecycle();
    SwaWindowStateView swaWindowStates(int32_t batchSize) const;

    BoundedSwaKVPageManager* mBoundedSwaPageManager{};
    bool mSpeculativeRequest{false};
    //! Per-slot base and draft growth of the step being prepared; reused across steps.
    std::vector<int32_t> mBaseExtraTokensScratch;
    std::vector<int32_t> mDraftExtraTokensScratch;
    std::optional<ContextCacheRequest> mContextRequest;
    //! Declared last so SWA work is drained before context-cache leases release their page references.
    std::optional<BoundedSwaKVPageManager::RequestHandle> mSwaRequest;
};

} // namespace rt
} // namespace trt_edgellm
