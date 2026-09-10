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

#include "runtime/state/contextCache/blockIndex.h"
#include "runtime/state/contextCache/cacheRecord.h"
#include "runtime/state/contextCache/evictionPlanner.h"
#include "runtime/state/contextCache/resourcePools.h"
#include "runtime/state/contextCache/reusePlan.h"
#include "runtime/state/contextCache/specStatePlan.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

namespace trt_edgellm
{
namespace rt
{

class ContextCacheManager;

//! Move-only ownership of one request's active context-cache references and page bindings.
//!
//! ContextCacheManager provides no internal synchronization. Callers must serialize all manager and lease access
//! under one single-writer execution contract. A lease must be released or destroyed before its manager is destroyed.
//! Destruction releases every active reference; publication separately adds cache ownership for reusable state.
//! Moving a lease transfers its active-reference cleanup responsibility.
class CacheRequestLease
{
public:
    CacheRequestLease() noexcept = default;
    ~CacheRequestLease() noexcept;
    CacheRequestLease(CacheRequestLease&& other) noexcept;
    CacheRequestLease& operator=(CacheRequestLease&& other) noexcept;
    CacheRequestLease(CacheRequestLease const&) = delete;
    CacheRequestLease& operator=(CacheRequestLease const&) = delete;

    //! Ordered host binding list of base-model logical K-page IDs; the adapter expands it to a kernel [K, V] row.
    std::vector<PageId> const& basePages() const noexcept;
    //! Ordered host binding list of paged spec-state logical K-page IDs; Phase 1 binds EAGLE draft KV pages.
    std::vector<PageId> const& draftPages() const noexcept;
    std::optional<int32_t> recurrentSnapshotSlot() const noexcept;
    std::optional<int32_t> partialKvSnapshotSlot() const noexcept;
    bool valid() const noexcept;
    void release() noexcept;

private:
    friend class ContextCacheManager;

    //! Non-owning back-pointer that makes the lease valid and routes active-reference cleanup.
    ContextCacheManager* mManager{};

    //! Acquisition provenance retained while the request executes so publication can validate the produced prefix.
    ReusePlanMode mMode{ReusePlanMode::kVanilla};
    std::vector<BlockHash> mMatchedBlockHashes;

    //! Every remaining active reference owned by this lease and released by release() or destruction.
    std::vector<ResourceId> mActiveResources;

    //! Ordered physical bindings: the shared prefix followed by request-private pages.
    std::vector<PageId> mBasePages;
    std::vector<PageId> mSpecPages;
    std::optional<SpecReplayDependency> mSpecReplayDependency;
    bool mHybridHasAttention{false};
    std::optional<int32_t> mRecurrentSnapshotBinding;
    std::optional<int32_t> mPartialKvSnapshotBinding;
};

//! Outcome of planning and acquiring one request.
enum class AcquireStatus : uint8_t
{
    kAcquired,
    kInsufficientCapacity,
};

//! A plan is built and consumed under the manager's single-writer contract, so it cannot become stale between
//! planning and acquisition.
struct AcquireResult
{
    std::optional<CacheRequestLease> lease;
    AcquireStatus status{AcquireStatus::kAcquired};
    ReusePlan plan;
};

enum class PublishStatus : uint8_t
{
    //! A new record or draft-state upgrade completed; configured record limits may evict a new record immediately.
    kPublished,
    //! The exact record already existed and was promoted to MRU.
    kExistingRecord,
};

struct PublishRequest
{
    std::vector<BlockHash> fullBlockHashes;
    //! Greatest logical prefix ready for publication. For speculative leases, both base and draft state must be
    //! materialized through this boundary.
    int32_t residentStateLength{};
};

//! Detailed publication outcome describing the record path selected at commit time.
struct PublishResult
{
    PublishStatus status{PublishStatus::kPublished};
    std::optional<RecordId> record;
    //! Canonical base pages owned by the published record. This is never an active-row update recipe.
    std::vector<PageId> canonicalBasePages;
    //! Published boundary represented by canonicalBasePages; later producer pages remain unpublished.
    int32_t publishedBaseFullBlockCount{};
};

//! Request-private snapshot slots reserved before the runtime enqueues a hybrid capture.
struct HybridSnapshotReservation
{
    int32_t recurrentSnapshotSlot{};
    std::optional<int32_t> partialKvSnapshotSlot;
};

struct HybridPublishRequest
{
    std::vector<BlockHash> fullBlockHashes;
    HybridCheckpointKey checkpoint;
    HybridSnapshotReservation snapshots;
};

//! Cumulative outcomes of record eviction. Reclaimed counts include only resources returned to a free list at the
//! eviction point; resources still pinned by an active request are not counted.
struct ContextCacheManagerMetrics
{
    uint64_t evictedRecords{};
    uint64_t reclaimedBaseKvPages{};
    uint64_t reclaimedDraftKvPages{};
    uint64_t reclaimedRecurrentSnapshots{};
    uint64_t reclaimedPartialKvSnapshots{};
};

//! Host orchestrator for the complete context-cache lifecycle under an externally serialized single-writer contract.
//!
//! The manager owns neither a worker thread nor a task queue and is not thread-safe. All manager and lease access that
//! could overlap a mutation must be externally serialized, and no lease may outlive its manager. publish() is the
//! final host commit for a producer whose state is already ready; it neither inspects nor waits on CUDA events. A
//! publication must contain at least one resident full block.
//!
//! The lifecycle is acquire -> model execution -> publish or release. Each acquire method plans and pins the selected
//! hit as one serialized manager operation before eviction and request-private allocation. Publication atomically
//! adds cache ownership, canonical base-block mappings, and one complete record. ResourcePools owns capacity/refcounts,
//! BaseBlockIndex owns canonical base lookup, SpecStateIndex owns coherent spec-state lookup, CacheRecordStore owns
//! endpoints/LRU, and EvictionPlanner selects record victims without mutation.
class ContextCacheManager
{
public:
    ContextCacheManager(int32_t pageSize, ResourceDemand capacities, int32_t maxRecords,
        std::optional<SpecReuseContract> specReuseContract = std::nullopt);

    AcquireResult acquireVanilla(std::vector<BlockHash> const& inputFullBlockHashes, int32_t inputTokenCount,
        ContextCacheLookupPolicy lookupPolicy = ContextCacheLookupPolicy::kUseCache);
    AcquireResult acquireHybrid(std::vector<HybridCheckpointCandidate> const& candidates,
        std::vector<BlockHash> const& inputFullBlockHashes, int32_t inputTokenCount, bool hasAttention,
        ContextCacheLookupPolicy lookupPolicy = ContextCacheLookupPolicy::kUseCache);
    //! Acquire a combined hybrid base + MTP draft lease at an exact recurrent/partial-KV checkpoint. A hit rebinds both
    //! the cached base pages and the equally long coherent draft page path. MTP always retains a private partial page.
    AcquireResult acquireHybridMtp(std::vector<HybridCheckpointCandidate> const& candidates,
        std::vector<BlockHash> const& inputFullBlockHashes, int32_t inputTokenCount,
        ContextCacheLookupPolicy lookupPolicy = ContextCacheLookupPolicy::kUseCache);
    std::vector<int32_t> hybridCandidateLengths(int32_t inputTokenCount) const;
    //! The caller must first select a supported speculative context-reuse handler for this deployment.
    AcquireResult acquireSpec(std::vector<BlockHash> const& inputFullBlockHashes, int32_t inputTokenCount,
        ContextCacheLookupPolicy lookupPolicy = ContextCacheLookupPolicy::kUseCache);
    //! Atomically append private base pages; false leaves the lease and cache metadata unchanged.
    bool growBasePages(CacheRequestLease& lease, int32_t count);
    //! Atomically append base and paged spec-state pages; false leaves the lease and cache metadata unchanged.
    bool growSpecPages(CacheRequestLease& lease, int32_t baseCount, int32_t draftCount);
    //! Reserve unpublished hybrid snapshot storage. Failure is retention pressure and leaves the lease unchanged.
    std::optional<HybridSnapshotReservation> reserveHybridSnapshots(
        CacheRequestLease& lease, bool needsPartialKvSnapshot);
    //! Release hit-snapshot active pins after the stream-ordered restore has completed.
    void releaseRestoredHybridSnapshots(CacheRequestLease& lease);
    //! Release producer ownership after capture is terminal and publication has committed.
    void retireHybridSnapshotReservation(CacheRequestLease& lease, HybridSnapshotReservation const& reservation);
    //! Commit ready full blocks after validating that the acquired prefix still describes this producer.
    PublishResult publish(CacheRequestLease& lease, PublishRequest const& request);
    //! Commit one already-captured exact hybrid checkpoint. The caller must make snapshot writes terminal first.
    PublishResult publishHybrid(CacheRequestLease& lease, HybridPublishRequest const& request);
    //! Commit one already-captured exact hybrid+MTP checkpoint, retaining both the base path and the equally long
    //! coherent draft path through the (exactLength - 1) boundary. The caller must make snapshot writes terminal first.
    PublishResult publishHybridMtp(CacheRequestLease& lease, HybridPublishRequest const& request);

    //! Acquire resources for an externally-constructed reuse plan. The caller may trim or adjust the plan (e.g. media
    //! boundary trimming) before committing it to lease allocation.
    AcquireResult acquire(ReusePlan plan);

    ResourcePools const& pools() const noexcept;
    BaseBlockIndex const& baseIndex() const noexcept;
    SpecStateIndex const& specIndex() const noexcept;
    CacheRecordStore const& records() const noexcept;
    ContextCacheManagerMetrics const& metrics() const noexcept;

private:
    friend class CacheRequestLease;

    struct PreparedPublication;

    void releaseLease(CacheRequestLease& lease) noexcept;
    bool growPages(CacheRequestLease& lease, ResourceDemand const& demand);
    PublishResult commitPreparedPublication(PreparedPublication publication);
    void evictRecord(RecordId id);
    void applyEviction(EvictionPlan const& plan);
    void releaseLeaseResource(CacheRequestLease& lease, ResourceId resource);

    int32_t mPageSize{};
    ResourcePools mPools;
    BaseBlockIndex mBaseIndex;
    SpecStateIndex mSpecIndex;
    std::optional<SpecReuseContract> mSpecReuseContract;
    CacheRecordStore mRecords;
    ContextCacheManagerMetrics mMetrics;
};

} // namespace rt
} // namespace trt_edgellm
