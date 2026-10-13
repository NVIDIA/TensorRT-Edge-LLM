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
#include <variant>
#include <vector>

namespace trt_edgellm
{
namespace rt
{

//! One runtime request facade over orthogonal execution paging and context-reuse clients.
//!
//! The current support matrix selects exactly one backend: bounded SWA when reuse is off, or context reuse with full
//! KV storage. Keeping the runtime lifecycle behind this facade allows a later adapter to compose both without moving
//! SWA rotation into the context-reuse manager.
class ManagedKVCacheRequest final
{
public:
    static std::optional<ManagedKVCacheRequest> begin(ContextCacheCoordinator* contextCache,
        BoundedSwaKVPageManager* swaPageManager, LLMGenerationRequest const& request,
        DecodingInferenceContext const& context, bool speculativeRequest, DecodingKvHeadroom const& headroom,
        DecodingTokenStateContract tokenStateContract, ContextCacheCommitPolicy commitPolicy,
        std::vector<int32_t> const& mediaTokenIds = {});

    ManagedKVCacheRequest(ManagedKVCacheRequest&&) noexcept = default;
    ManagedKVCacheRequest& operator=(ManagedKVCacheRequest&&) = delete;
    ManagedKVCacheRequest(ManagedKVCacheRequest const&) = delete;
    ManagedKVCacheRequest& operator=(ManagedKVCacheRequest const&) = delete;
    ~ManagedKVCacheRequest() = default;

    bool hasContextReuse() const noexcept;
    std::vector<int32_t> const* prefillStarts() const noexcept;
    int32_t reuseTokenLength(int32_t slot) const;

    bool publishHybridMtpEndpoint(
        int32_t slot, int32_t residentStateLength, Tensor const& baseHiddenStates, int32_t boundaryHiddenRow);
    bool restoreHybridMtpBoundaryHidden(int32_t slot, Tensor& baseHiddenStates, int32_t destinationRow);

    bool preparePrefill();
    bool prepareSwaPrefill(std::vector<int32_t> const& effectiveInputLengths);
    bool enqueuePrefillCaptures();
    bool completePrefill(DecodingInferenceContext const& context, std::vector<int32_t> const& commonStateLengths);
    bool prepareDecodeStep(DecodingInferenceContext const& context, DecodingKvHeadroom const& headroom);
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
        ResidentRef resident, cudaStream_t stream, std::vector<int32_t> const& mediaTokenIds = {},
        std::vector<imageUtils::ImageData> const& imageBuffers = {},
        std::vector<audioUtils::AudioData> const& audioBuffers = {});
    bool finalizeSequenceAdmission(int32_t slot, int32_t const& lookaheadToken, int32_t fullInputLength);
    bool retractSequenceAdmission() noexcept;
    //! @}

private:
    using Backend = std::variant<ContextCacheRequest, BoundedSwaKVPageManager::RequestHandle>;

    explicit ManagedKVCacheRequest(ContextCacheRequest&& request) noexcept;
    ManagedKVCacheRequest(BoundedSwaKVPageManager& manager, BoundedSwaKVPageManager::RequestHandle&& request) noexcept;

    ContextCacheRequest& contextRequest();
    ContextCacheRequest const& contextRequest() const;
    BoundedSwaKVPageManager::RequestHandle& swaRequest();

    Backend mBackend;
    BoundedSwaKVPageManager* mBoundedSwaPageManager{};
};

} // namespace rt
} // namespace trt_edgellm
