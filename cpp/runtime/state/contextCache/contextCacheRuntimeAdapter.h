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

#include "runtime/state/contextCache/contextCacheManager.h"

#include <cstdint>
#include <optional>
#include <vector>

namespace trt_edgellm
{
namespace rt
{

class KVPageTable;

//! Result of runtime-side planning and transactional acquisition.
struct RuntimeCacheAcquireResult
{
    std::optional<CacheRequestLease> lease;
    AcquireStatus status{AcquireStatus::kAcquired};
    ReusePlanKind planKind{ReusePlanKind::kStandard};
    bool forcedCold{false};
};

//! Narrow bridge between the CUDA-free context-cache policy core and runtime-owned page tables.
//!
//! Resource IDs are physical page IDs: binding a lease writes those IDs directly into a KVPageTable. The adapter owns
//! no second allocator. A cache-hit acquisition that cannot obtain its private pages is retried once as an explicit
//! bypass plan; a cold capacity failure is returned to the runtime as admission backpressure.
class ContextCacheRuntimeAdapter
{
public:
    ContextCacheRuntimeAdapter(int32_t pageSize, ResourceDemand capacities, int32_t maxRecords);

    RuntimeCacheAcquireResult acquireVanilla(CacheDomainId domain, std::vector<BlockHash> const& fullBlockHashes,
        int32_t inputTokenCount, LookupPolicy lookupPolicy = LookupPolicy::kUseCache);

    //! Hash exact stored candidate lengths and acquire the longest coherent hybrid checkpoint.
    RuntimeCacheAcquireResult acquireHybrid(CacheDomainId domain, std::vector<int32_t> const& inputTokens,
        std::vector<BlockKeyExtras> const& touchedBlockExtras, RecurrentStateSchemaId schema, bool hasAttention,
        LookupPolicy lookupPolicy = LookupPolicy::kUseCache);

    //! Acquire from exact candidate digests prepared by the runtime. This form is required when partial-block
    //! identity depends on media spans that may begin after an earlier checkpoint in the same page.
    RuntimeCacheAcquireResult acquireHybrid(CacheDomainId domain,
        std::vector<HybridCheckpointCandidate> const& candidates, std::vector<BlockHash> const& fullBlockHashes,
        int32_t inputTokenCount, RecurrentStateSchemaId schema, bool hasAttention,
        LookupPolicy lookupPolicy = LookupPolicy::kUseCache);

    //! Hash exact stored endpoint lengths and acquire one coherent Hybrid+MTP checkpoint.
    RuntimeCacheAcquireResult acquireHybridMtp(CacheDomainId domain, std::vector<int32_t> const& inputTokens,
        std::vector<BlockKeyExtras> const& touchedBlockExtras, RecurrentStateSchemaId schema,
        DraftEngineSignature draftSignature, LookupPolicy lookupPolicy = LookupPolicy::kUseCache);

    //! Acquire from exact Hybrid+MTP candidates prepared by the runtime.
    RuntimeCacheAcquireResult acquireHybridMtp(CacheDomainId domain,
        std::vector<HybridMtpCheckpointCandidate> const& candidates, std::vector<BlockHash> const& fullBlockHashes,
        int32_t inputTokenCount, RecurrentStateSchemaId schema, DraftEngineSignature draftSignature,
        LookupPolicy lookupPolicy = LookupPolicy::kUseCache);

    //! Acquire coherent base/draft state for greedy non-hybrid EAGLE. Initial integration always uses full-page
    //! replay; one-token replay remains disabled until both engines support validated mid-page prefill.
    RuntimeCacheAcquireResult acquireSpec(SpecDecodeMode mode, CacheDomainId domain,
        DraftEngineSignature draftSignature, std::vector<BlockHash> const& fullBlockHashes, int32_t inputTokenCount,
        LookupPolicy lookupPolicy = LookupPolicy::kUseCache);

    //! Replace one page-table row from the lease's ordered physical base-page path. Upload is a separate batched step.
    void bindBaseRow(KVPageTable& table, int32_t slot, CacheRequestLease const& lease) const;

    //! Replace one draft page-table row from the lease's ordered physical speculative draft-page path.
    void bindDraftRow(KVPageTable& table, int32_t slot, CacheRequestLease const& lease) const;

    //! Grow a vanilla lease to at least requiredPages and refresh its host page-table row.
    bool growBaseToPageCount(CacheRequestLease& lease, int32_t requiredPages, KVPageTable& table, int32_t slot);

    //! Atomically grow both EAGLE page paths and refresh their host page-table rows.
    bool growSpecToPageCounts(CacheRequestLease& lease, int32_t requiredBasePages, int32_t requiredDraftPages,
        KVPageTable& baseTable, KVPageTable& draftTable, int32_t slot);

    //! Atomically grow both Hybrid+MTP page paths and refresh their host page-table rows.
    bool growHybridMtpToPageCounts(CacheRequestLease& lease, int32_t requiredBasePages, int32_t requiredDraftPages,
        KVPageTable& baseTable, KVPageTable& draftTable, int32_t slot);

    //! Return exact hybrid checkpoint lengths that may match this input.
    std::vector<int32_t> hybridCandidateLengths(
        CacheDomainId domain, RecurrentStateSchemaId schema, int32_t inputTokenCount) const;

    std::vector<int32_t> hybridMtpCandidateLengths(CacheDomainId domain, RecurrentStateSchemaId schema,
        DraftEngineSignature draftSignature, int32_t inputTokenCount) const;

    //! Commit a ready vanilla or EAGLE boundary and return its canonical physical lineage.
    PublishResult publish(CacheRequestLease& lease, PublishRequest const& request);

    //! Reserve snapshot slots before enqueueing recurrent/partial-KV capture.
    std::optional<HybridSnapshotReservation> reserveHybridSnapshots(
        CacheRequestLease& lease, bool needsPartialKvSnapshot);

    //! Release the active pins acquired for a restored hybrid checkpoint after restore is stream-terminal.
    void releaseRestoredHybridSnapshots(CacheRequestLease& lease);

    //! Commit an already-captured coherent hybrid checkpoint.
    PublishResult publishHybrid(CacheRequestLease& lease, HybridPublishRequest const& request);

    //! Commit an already-captured coherent Hybrid+MTP endpoint.
    PublishResult publishHybridMtp(CacheRequestLease& lease, HybridMtpPublishRequest const& request);

    //! Release producer ownership after hybrid capture has either committed or been abandoned.
    void retireHybridSnapshotReservation(CacheRequestLease& lease, HybridSnapshotReservation const& reservation);

    //! Move an active lease onto the canonical base-page prefix at a safe execution boundary.
    void rebindBasePrefix(CacheRequestLease& lease, std::vector<PageId> const& canonicalPages);

    //! Read-only policy state for diagnostics and invariant tests. Runtime mutation stays behind this adapter.
    ContextCacheManager const& manager() const noexcept;

private:
    int32_t mPageSize{};
    ContextCacheManager mManager;
};

} // namespace rt
} // namespace trt_edgellm
