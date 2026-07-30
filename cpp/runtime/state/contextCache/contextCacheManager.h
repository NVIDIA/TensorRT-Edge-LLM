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

#include <cstdint>
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
    //! Ordered host binding list of EAGLE draft logical K-page IDs; empty for a vanilla lease.
    std::vector<PageId> const& draftPages() const noexcept;
    //! Immutable source for the private base page at reuseTokenLength() / pageSize during one-token replay.
    std::vector<PageId> const& baseCowSources() const noexcept;
    //! Immutable source for the corresponding private draft page during one-token replay.
    std::vector<PageId> const& draftCowSources() const noexcept;
    std::optional<int32_t> recurrentSnapshotSlot() const noexcept;
    std::optional<int32_t> partialKvSnapshotSlot() const noexcept;
    int32_t reuseTokenLength() const noexcept;
    bool valid() const noexcept;
    void release() noexcept;

private:
    friend class ContextCacheManager;

    ContextCacheManager* mManager{};
    ReusePlanMode mMode{ReusePlanMode::kVanilla};
    CacheDomainId mDomain{};
    std::optional<DraftEngineSignature> mDraftSignature;
    int32_t mReuseTokenLength{};
    std::vector<BlockHash> mMatchedBlockHashes;
    std::vector<ResourceId> mActiveResources;
    std::vector<PageId> mBasePages;
    std::vector<PageId> mDraftPages;
    std::vector<PageId> mBaseCowSources;
    std::vector<PageId> mDraftCowSources;
    std::optional<SpecReplayDependency> mSpecReplayDependency;
    std::optional<HybridCheckpointKey> mHybridCheckpoint;
    std::optional<HybridMtpCheckpointKey> mHybridMtpCheckpoint;
    std::optional<RecordId> mHybridRecord;
    std::optional<RecurrentStateSchemaId> mRecurrentStateSchema;
    bool mHybridHasAttention{false};
    std::optional<int32_t> mRecurrentSnapshotBinding;
    std::optional<int32_t> mPartialKvSnapshotBinding;
};

//! Outcome of acquiring one already-built reuse plan.
enum class AcquireStatus : uint8_t
{
    kAcquired,
    kStalePlan,
    kInsufficientCapacity,
};

//! Results returned by ContextCacheManager::acquire use kAcquired with a lease and failures without one.
struct AcquireResult
{
    std::optional<CacheRequestLease> lease;
    AcquireStatus status{AcquireStatus::kAcquired};
};

enum class PublishStatus : uint8_t
{
    //! A new record or draft-state upgrade completed; configured record limits may evict a new record immediately.
    kPublished,
    //! The exact record already existed and was promoted to MRU.
    kExistingRecord,
    //! The commit policy suppressed this publication point without mutation.
    kSkippedByPolicy,
};

struct PublishRequest
{
    std::vector<BlockHash> fullBlockHashes;
    //! Base-model state ready for publication.
    int32_t baseResidentStateLength{};
    PublicationPoint point{};
    CommitPolicy policy{};
    //! Accepted linear draft state ready for publication. Required for EAGLE and absent for vanilla.
    //! Only complete draft pages are paired with the base record; this length may lag baseResidentStateLength.
    std::optional<int32_t> draftResidentStateLength;
};

//! Detailed publication outcome used by the runtime adapter to enforce physical lineage.
struct PublishResult
{
    PublishStatus status{PublishStatus::kPublished};
    std::optional<RecordId> record;
    std::vector<PageId> canonicalBasePages;
    int32_t publishedBaseFullBlockCount{};
    //! False when a private duplicate page was canonicalized and already-computed descendants were excluded.
    bool lineageComplete{true};
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
    PublicationPoint point{};
    CommitPolicy policy{};
    HybridSnapshotReservation snapshots;
};

struct HybridMtpPublishRequest
{
    std::vector<BlockHash> fullBlockHashes;
    HybridMtpCheckpointKey checkpoint;
    PublicationPoint point{};
    CommitPolicy policy{};
    HybridSnapshotReservation snapshots;
};

//! Host orchestrator for the complete context-cache lifecycle under an externally serialized single-writer contract.
//!
//! The manager owns neither a worker thread nor a task queue and is not thread-safe. All manager and lease access that
//! could overlap a mutation must be externally serialized, and no lease may outlive its manager. publish() is the
//! final host commit for a producer whose state is already ready; it neither inspects nor waits on CUDA events. A
//! non-skipped publication must contain at least one resident full block.
//!
//! The lifecycle is plan -> acquire -> model execution -> publish or release. Planning is read-only; acquire
//! revalidates and pins hits before eviction, then allocates request-private resources. Publication atomically adds
//! cache ownership, canonical base-block mappings, and one complete record. ResourcePools owns capacity/refcounts,
//! BaseBlockIndex owns canonical base lookup, DraftPathIndex owns coherent EAGLE-path lookup, CacheRecordStore owns
//! endpoints/LRU, and EvictionPlanner selects record victims without mutation.
class ContextCacheManager
{
public:
    ContextCacheManager(int32_t pageSize, ResourceDemand capacities, int32_t maxRecords);

    //! Construct a side-effect-free vanilla plan from the current base block index.
    ReusePlan planVanilla(CacheDomainId domain, std::vector<BlockHash> const& inputFullBlockHashes,
        int32_t inputTokenCount, LookupPolicy lookupPolicy = LookupPolicy::kUseCache) const;
    ReusePlan planHybrid(CacheDomainId domain, RecurrentStateSchemaId schema,
        std::vector<HybridCheckpointCandidate> const& candidates, std::vector<BlockHash> const& inputFullBlockHashes,
        int32_t inputTokenCount, bool hasAttention, LookupPolicy lookupPolicy = LookupPolicy::kUseCache) const;
    ReusePlan planHybridMtp(CacheDomainId domain, RecurrentStateSchemaId schema, DraftEngineSignature draftSignature,
        std::vector<HybridMtpCheckpointCandidate> const& candidates, std::vector<BlockHash> const& inputFullBlockHashes,
        int32_t inputTokenCount, LookupPolicy lookupPolicy = LookupPolicy::kUseCache) const;
    std::vector<int32_t> hybridCandidateLengths(
        CacheDomainId domain, RecurrentStateSchemaId schema, int32_t inputTokenCount) const;
    std::vector<int32_t> hybridMtpCandidateLengths(CacheDomainId domain, RecurrentStateSchemaId schema,
        DraftEngineSignature draftSignature, int32_t inputTokenCount) const;
    //! Construct a side-effect-free speculative plan. The caller must first select greedy, non-hybrid EAGLE on a
    //! supported full-attention or full-allocation SWA deployment; the manager cannot infer sampling policy or model
    //! topology. supportsOneTokenReplay must be true only when both base and draft backends support mid-page prefill.
    ReusePlan planSpec(SpecDecodeMode mode, CacheDomainId domain, DraftEngineSignature draftSignature,
        std::vector<BlockHash> const& inputFullBlockHashes, int32_t inputTokenCount, bool supportsOneTokenReplay,
        LookupPolicy lookupPolicy = LookupPolicy::kUseCache) const;
    //! Revalidate, pin, evict, and allocate atomically. Stale or capacity failure leaves no lasting mutation.
    AcquireResult acquire(ReusePlan const& plan);
    //! Atomically append private base pages; false leaves the lease and cache metadata unchanged.
    bool growBasePages(CacheRequestLease& lease, int32_t count);
    //! Atomically append EAGLE base and draft pages; false leaves the lease and cache metadata unchanged.
    bool growSpecPages(CacheRequestLease& lease, int32_t baseCount, int32_t draftCount);
    //! Atomically append Hybrid+MTP base and draft pages; false leaves the lease and metadata unchanged.
    bool growHybridMtpPages(CacheRequestLease& lease, int32_t baseCount, int32_t draftCount);
    //! Reserve unpublished hybrid snapshot storage. Failure is retention pressure and leaves the lease unchanged.
    std::optional<HybridSnapshotReservation> reserveHybridSnapshots(
        CacheRequestLease& lease, bool needsPartialKvSnapshot);
    //! Release hit-snapshot active pins after the stream-ordered restore has completed.
    void releaseRestoredHybridSnapshots(CacheRequestLease& lease);
    //! Release producer ownership after capture is terminal and publication has committed or been skipped.
    void retireHybridSnapshotReservation(CacheRequestLease& lease, HybridSnapshotReservation const& reservation);
    //! Commit ready full blocks after validating that the acquired prefix still describes this producer. EAGLE
    //! publication retains the complete base boundary and only the separately committed draft boundary.
    PublishStatus publish(CacheRequestLease& lease, PublishRequest const& request);
    //! Same commit as publish(), plus the canonical physical path and lineage boundary selected by the commit.
    PublishResult publishDetailed(CacheRequestLease& lease, PublishRequest const& request);
    //! Commit one already-captured exact hybrid checkpoint. The caller must make snapshot writes terminal first.
    PublishResult publishHybridDetailed(CacheRequestLease& lease, HybridPublishRequest const& request);
    //! Commit one already-captured coherent Hybrid+MTP endpoint. The caller must make all snapshot writes terminal.
    PublishResult publishHybridMtpDetailed(CacheRequestLease& lease, HybridMtpPublishRequest const& request);
    //! Transfer active ownership for a produced prefix to its canonical physical pages at a GPU-idle, externally
    //! serialized boundary. The caller must immediately update the runtime page table before any allocation,
    //! interleaving manager operation, or GPU work can observe the new ownership.
    void rebindBasePrefix(CacheRequestLease& lease, std::vector<PageId> const& canonicalPages);

    ResourcePools const& pools() const noexcept;
    BaseBlockIndex const& baseIndex() const noexcept;
    DraftPathIndex const& draftIndex() const noexcept;
    CacheRecordStore const& records() const noexcept;

private:
    friend class CacheRequestLease;

    void releaseLease(CacheRequestLease& lease) noexcept;
    bool growPages(CacheRequestLease& lease, ResourceDemand const& demand);
    void evictRecord(RecordId id);
    void applyEviction(EvictionPlan const& plan);
    void enforceRecordLimit();
    void releaseLeaseResource(CacheRequestLease& lease, ResourceId resource);

    int32_t mPageSize{};
    ResourcePools mPools;
    BaseBlockIndex mBaseIndex;
    DraftPathIndex mDraftIndex;
    CacheRecordStore mRecords;
};

} // namespace rt
} // namespace trt_edgellm
