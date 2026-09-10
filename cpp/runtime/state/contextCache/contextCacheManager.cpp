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

#include "runtime/state/contextCache/contextCacheManager.h"

#include "common/checkMacros.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iterator>
#include <utility>

namespace trt_edgellm
{
namespace rt
{
namespace
{

int32_t validateManagerConfiguration(int32_t pageSize, ResourceDemand const& capacities, int32_t maxRecords)
{
    ELLM_CHECK(pageSize > 0, "Context cache page size must be positive");
    ELLM_CHECK(capacities.isNonNegative(), "Context cache resource capacities must be non-negative");
    ELLM_CHECK(maxRecords >= 0, "Context cache maximum record count must be non-negative");
    return pageSize;
}

void releaseActiveRefsNoThrow(ResourcePools& pools, std::vector<ResourceId> const& resources) noexcept
{
    // These IDs came from a validated lease. A release failure means corrupted manager-owned metadata and is not
    // recoverable from a destructor, so the noexcept boundary intentionally terminates instead of masking it.
    for (ResourceId const& resource : resources)
    {
        pools.releaseActiveRef(resource);
    }
}

size_t resourceDemandCount(ResourceDemand const& demand) noexcept
{
    size_t count = static_cast<size_t>(demand.baseKvPages);
    count += static_cast<size_t>(demand.draftKvPages);
    count += static_cast<size_t>(demand.recurrentSnapshotSlots);
    count += static_cast<size_t>(demand.partialKvSnapshotSlots);
    return count;
}

size_t totalResourceCount(ReusePlan const& plan) noexcept
{
    return plan.basePageBindings.size() + plan.specPageBindings.size()
        + static_cast<size_t>(plan.recurrentSnapshotBinding.has_value())
        + static_cast<size_t>(plan.partialKvSnapshotBinding.has_value()) + resourceDemandCount(plan.demand);
}

struct MissingBaseMapping
{
    BlockHash hash;
    PageId page{};
};

struct BaseProjection
{
    std::vector<PageId> canonicalPages;
    std::vector<MissingBaseMapping> missingMappings;
};

BaseProjection prepareBaseProjection(
    BaseBlockIndex const& index, std::vector<BlockHash> const& logicalHashes, std::vector<PageId> const& producerPages)
{
    ELLM_CHECK(
        logicalHashes.size() <= producerPages.size(), "Context cache producer path is shorter than its logical path");

    BaseProjection projection;
    projection.canonicalPages.reserve(logicalHashes.size());
    projection.missingMappings.reserve(logicalHashes.size());
    for (size_t block = 0; block < logicalHashes.size(); ++block)
    {
        BlockHash const hash = logicalHashes[block];
        std::optional<PageId> const canonical = index.lookup(hash);
        if (canonical.has_value())
        {
            projection.canonicalPages.push_back(*canonical);
            continue;
        }

        auto const prepared = std::find_if(projection.missingMappings.begin(), projection.missingMappings.end(),
            [&](MissingBaseMapping const& missing) { return missing.hash == hash; });
        if (prepared != projection.missingMappings.end())
        {
            projection.canonicalPages.push_back(prepared->page);
            continue;
        }

        PageId const producerPage = producerPages[block];
        projection.missingMappings.push_back(MissingBaseMapping{hash, producerPage});
        projection.canonicalPages.push_back(producerPage);
    }
    return projection;
}

std::vector<ResourceId> specStateResources(SpecPagedStateRecord const& state)
{
    std::vector<ResourceId> resources;
    resources.reserve(state.pagePath.size());
    for (PageId const page : state.pagePath)
    {
        resources.push_back(ResourceId{ResourceType::kDraftKvPage, page});
    }
    return resources;
}

std::optional<RecordId> selectRecordLimitVictim(CacheRecordStore const& records)
{
    size_t const recordLimit = static_cast<size_t>(records.maxRecords());
    ELLM_CHECK(records.size() <= recordLimit, "Context cache record count already exceeds its configured limit");
    if (recordLimit == 0 || records.size() < recordLimit)
    {
        return std::nullopt;
    }

    std::vector<RecordId> const lru = records.lruToMru();
    ELLM_CHECK(!lru.empty(), "Context cache record limit enforcement found an empty LRU");
    return lru.front();
}

} // namespace

struct ContextCacheManager::PreparedPublication
{
    CacheRecord record;
    std::vector<PageId> canonicalBasePages;
    std::vector<MissingBaseMapping> missingBaseMappings;
    std::vector<ResourceId> cacheResources;
    std::optional<RecordId> recordLimitVictim;
    int32_t publishedBaseFullBlockCount{};
};

namespace
{

class PublicationRollbackGuard
{
public:
    PublicationRollbackGuard(ResourcePools& pools, BaseBlockIndex& baseIndex, SpecStateIndex& specIndex,
        CacheRecordStore& records, std::vector<ResourceId> const& cacheResources,
        std::vector<MissingBaseMapping> const& missingMappings) noexcept
        : mPools(pools)
        , mBaseIndex(baseIndex)
        , mSpecIndex(specIndex)
        , mRecords(records)
        , mCacheResources(cacheResources)
        , mMissingMappings(missingMappings)
    {
    }

    PublicationRollbackGuard(PublicationRollbackGuard const&) = delete;
    PublicationRollbackGuard& operator=(PublicationRollbackGuard const&) = delete;

    ~PublicationRollbackGuard() noexcept
    {
        if (mCommitted)
        {
            return;
        }
        if (mSpecIndexed)
        {
            mSpecIndex.paged().erase(mRecords.get(mInsertedRecord));
        }
        if (mInsertedRecord != 0 && mRecords.contains(mInsertedRecord))
        {
            (void) mRecords.erase(mInsertedRecord);
        }
        while (mInsertedMappingCount > 0)
        {
            --mInsertedMappingCount;
            mBaseIndex.erasePage(mMissingMappings[mInsertedMappingCount].page);
        }
        while (mAddedCacheRefCount > 0)
        {
            --mAddedCacheRefCount;
            mPools.releaseCacheRef(mCacheResources[mAddedCacheRefCount]);
        }
    }

    void cacheRefAdded() noexcept
    {
        ++mAddedCacheRefCount;
    }

    void baseMappingInserted() noexcept
    {
        ++mInsertedMappingCount;
    }

    void recordInserted(RecordId id) noexcept
    {
        mInsertedRecord = id;
    }

    void specIndexed() noexcept
    {
        mSpecIndexed = true;
    }

    void commit() noexcept
    {
        mCommitted = true;
    }

private:
    ResourcePools& mPools;
    BaseBlockIndex& mBaseIndex;
    SpecStateIndex& mSpecIndex;
    CacheRecordStore& mRecords;
    std::vector<ResourceId> const& mCacheResources;
    std::vector<MissingBaseMapping> const& mMissingMappings;
    size_t mAddedCacheRefCount{};
    size_t mInsertedMappingCount{};
    RecordId mInsertedRecord{};
    bool mSpecIndexed{};
    bool mCommitted{};
};

class SpecUpgradeRollbackGuard
{
public:
    SpecUpgradeRollbackGuard(ResourcePools& pools, SpecStateIndex& specIndex, CacheRecord const& upgradedRecord,
        std::vector<ResourceId> const& resources) noexcept
        : mPools(pools)
        , mSpecIndex(specIndex)
        , mUpgradedRecord(upgradedRecord)
        , mResources(resources)
    {
    }

    SpecUpgradeRollbackGuard(SpecUpgradeRollbackGuard const&) = delete;
    SpecUpgradeRollbackGuard& operator=(SpecUpgradeRollbackGuard const&) = delete;

    ~SpecUpgradeRollbackGuard() noexcept
    {
        if (mCommitted)
        {
            return;
        }
        if (mSpecIndexed)
        {
            mSpecIndex.paged().erase(mUpgradedRecord);
        }
        while (mAddedCount > 0)
        {
            --mAddedCount;
            mPools.releaseCacheRef(mResources[mAddedCount]);
        }
    }

    void specIndexed() noexcept
    {
        mSpecIndexed = true;
    }

    void addAll()
    {
        for (ResourceId const& resource : mResources)
        {
            mPools.addCacheRef(resource);
            ++mAddedCount;
        }
    }

    void commit() noexcept
    {
        mCommitted = true;
    }

private:
    ResourcePools& mPools;
    SpecStateIndex& mSpecIndex;
    CacheRecord const& mUpgradedRecord;
    std::vector<ResourceId> const& mResources;
    size_t mAddedCount{};
    bool mSpecIndexed{};
    bool mCommitted{};
};

} // namespace

CacheRequestLease::~CacheRequestLease() noexcept
{
    release();
}

CacheRequestLease::CacheRequestLease(CacheRequestLease&& other) noexcept
{
    *this = std::move(other);
}

CacheRequestLease& CacheRequestLease::operator=(CacheRequestLease&& other) noexcept
{
    if (this != &other)
    {
        release();
        mManager = other.mManager;
        mMode = other.mMode;
        mMatchedBlockHashes = std::move(other.mMatchedBlockHashes);
        mActiveResources = std::move(other.mActiveResources);
        mBasePages = std::move(other.mBasePages);
        mSpecPages = std::move(other.mSpecPages);
        mSpecReplayDependency = other.mSpecReplayDependency;
        mHybridHasAttention = other.mHybridHasAttention;
        mRecurrentSnapshotBinding = other.mRecurrentSnapshotBinding;
        mPartialKvSnapshotBinding = other.mPartialKvSnapshotBinding;

        other.mManager = nullptr;
        other.mMode = ReusePlanMode::kVanilla;
        other.mMatchedBlockHashes.clear();
        other.mActiveResources.clear();
        other.mBasePages.clear();
        other.mSpecPages.clear();
        other.mSpecReplayDependency.reset();
        other.mHybridHasAttention = false;
        other.mRecurrentSnapshotBinding.reset();
        other.mPartialKvSnapshotBinding.reset();
    }
    return *this;
}

std::vector<PageId> const& CacheRequestLease::basePages() const noexcept
{
    return mBasePages;
}

std::vector<PageId> const& CacheRequestLease::draftPages() const noexcept
{
    return mSpecPages;
}

std::optional<int32_t> CacheRequestLease::recurrentSnapshotSlot() const noexcept
{
    return mRecurrentSnapshotBinding;
}

std::optional<int32_t> CacheRequestLease::partialKvSnapshotSlot() const noexcept
{
    return mPartialKvSnapshotBinding;
}

bool CacheRequestLease::valid() const noexcept
{
    return mManager != nullptr;
}

void CacheRequestLease::release() noexcept
{
    if (mManager != nullptr)
    {
        mManager->releaseLease(*this);
        return;
    }
    mMode = ReusePlanMode::kVanilla;
    mMatchedBlockHashes.clear();
    mActiveResources.clear();
    mBasePages.clear();
    mSpecPages.clear();
    mSpecReplayDependency.reset();
    mHybridHasAttention = false;
    mRecurrentSnapshotBinding.reset();
    mPartialKvSnapshotBinding.reset();
}

ContextCacheManager::ContextCacheManager(
    int32_t pageSize, ResourceDemand capacities, int32_t maxRecords, std::optional<SpecReuseContract> specReuseContract)
    : mPageSize(validateManagerConfiguration(pageSize, capacities, maxRecords))
    , mPools(capacities)
    , mSpecReuseContract(specReuseContract)
    , mRecords(maxRecords)
{
}

AcquireResult ContextCacheManager::acquireVanilla(
    std::vector<BlockHash> const& inputFullBlockHashes, int32_t inputTokenCount, ContextCacheLookupPolicy lookupPolicy)
{
    return acquire(makeVanillaReusePlan(inputFullBlockHashes, inputTokenCount, mPageSize, mBaseIndex, lookupPolicy));
}

AcquireResult ContextCacheManager::acquireHybrid(std::vector<HybridCheckpointCandidate> const& candidates,
    std::vector<BlockHash> const& inputFullBlockHashes, int32_t inputTokenCount, bool hasAttention,
    ContextCacheLookupPolicy lookupPolicy)
{
    return acquire(makeHybridReusePlan(
        candidates, inputFullBlockHashes, inputTokenCount, mPageSize, hasAttention, mRecords, lookupPolicy));
}

AcquireResult ContextCacheManager::acquireHybridMtp(std::vector<HybridCheckpointCandidate> const& candidates,
    std::vector<BlockHash> const& inputFullBlockHashes, int32_t inputTokenCount, ContextCacheLookupPolicy lookupPolicy)
{
    return acquire(
        makeHybridMtpReusePlan(candidates, inputFullBlockHashes, inputTokenCount, mPageSize, mRecords, lookupPolicy));
}

std::vector<int32_t> ContextCacheManager::hybridCandidateLengths(int32_t inputTokenCount) const
{
    return mRecords.hybridCandidateLengths(inputTokenCount);
}

AcquireResult ContextCacheManager::acquireSpec(
    std::vector<BlockHash> const& inputFullBlockHashes, int32_t inputTokenCount, ContextCacheLookupPolicy lookupPolicy)
{
    ELLM_CHECK(mSpecReuseContract.has_value(), "Speculative context cache acquire requires a registered contract");
    return acquire(makeSpecReusePlan(SpecReusePlanInput{inputFullBlockHashes, inputTokenCount, mPageSize, lookupPolicy,
                                         mBaseIndex, mSpecIndex, mRecords},
        *mSpecReuseContract));
}

AcquireResult ContextCacheManager::acquire(ReusePlan plan)
{
    CacheRequestLease lease;
    lease.mManager = this;
    lease.mMode = plan.mode;
    lease.mMatchedBlockHashes = plan.matchedBlockHashes;
    lease.mSpecReplayDependency = plan.specReplayDependency;
    lease.mHybridHasAttention = plan.hybridHasAttention;
    lease.mRecurrentSnapshotBinding = plan.recurrentSnapshotBinding;
    lease.mPartialKvSnapshotBinding = plan.partialKvSnapshotBinding;
    lease.mActiveResources.reserve(totalResourceCount(plan));
    size_t const reusedBasePageCount = plan.basePageBindings.size();
    size_t const reusedSpecPageCount = plan.specPageBindings.size();
    lease.mBasePages.reserve(reusedBasePageCount + static_cast<size_t>(plan.demand.baseKvPages));
    lease.mSpecPages.reserve(reusedSpecPageCount + static_cast<size_t>(plan.demand.draftKvPages));

    for (size_t index = 0; index < plan.basePageBindings.size(); ++index)
    {
        PageId const page = plan.basePageBindings[index];
        ResourceId const resource{ResourceType::kBaseKvPage, page};
        mPools.addActiveRef(resource);
        lease.mActiveResources.push_back(resource);
        if (index < reusedBasePageCount)
        {
            lease.mBasePages.push_back(page);
        }
    }
    for (size_t index = 0; index < plan.specPageBindings.size(); ++index)
    {
        PageId const page = plan.specPageBindings[index];
        ResourceId const resource{ResourceType::kDraftKvPage, page};
        mPools.addActiveRef(resource);
        lease.mActiveResources.push_back(resource);
        if (index < reusedSpecPageCount)
        {
            lease.mSpecPages.push_back(page);
        }
    }

    if (plan.recurrentSnapshotBinding.has_value())
    {
        ResourceId const resource{ResourceType::kRecurrentSnapshot, *plan.recurrentSnapshotBinding};
        mPools.addActiveRef(resource);
        lease.mActiveResources.push_back(resource);
    }
    if (plan.partialKvSnapshotBinding.has_value())
    {
        ResourceId const resource{ResourceType::kPartialKvSnapshot, *plan.partialKvSnapshotBinding};
        mPools.addActiveRef(resource);
        lease.mActiveResources.push_back(resource);
    }

    std::optional<RecordId> const protectedRecord = plan.specRecord.has_value() ? plan.specRecord : plan.hybridRecord;
    EvictionPlan const eviction = EvictionPlanner::plan(plan.demand, mPools, mRecords, protectedRecord);
    if (!eviction.feasible)
    {
        return AcquireResult{std::nullopt, AcquireStatus::kInsufficientCapacity, std::move(plan)};
    }

    std::vector<ResourceId> allocated(resourceDemandCount(plan.demand));
    // Every host allocation has completed. A feasible plan makes the remaining eviction/allocation sequence a
    // deterministic metadata commit; allocateInto() performs no dynamic allocation.
    applyEviction(eviction);
    ELLM_CHECK(
        mPools.allocateInto(plan.demand, allocated), "Context cache allocation failed after a feasible eviction plan");
    for (ResourceId const& resource : allocated)
    {
        lease.mActiveResources.push_back(resource);
        if (resource.type == ResourceType::kBaseKvPage)
        {
            lease.mBasePages.push_back(resource.index);
        }
        else if (resource.type == ResourceType::kDraftKvPage)
        {
            lease.mSpecPages.push_back(resource.index);
        }
    }
    if (plan.specRecord.has_value() && mRecords.contains(*plan.specRecord))
    {
        mRecords.touch(*plan.specRecord);
    }
    if (plan.hybridRecord.has_value() && mRecords.contains(*plan.hybridRecord))
    {
        mRecords.touch(*plan.hybridRecord);
    }

    return AcquireResult{std::optional<CacheRequestLease>{std::move(lease)}, AcquireStatus::kAcquired, std::move(plan)};
}

bool ContextCacheManager::growBasePages(CacheRequestLease& lease, int32_t count)
{
    ELLM_CHECK(lease.valid() && lease.mManager == this, "Context cache lease does not belong to this manager");
    ELLM_CHECK(lease.mMode == ReusePlanMode::kVanilla || lease.mMode == ReusePlanMode::kHybrid,
        "Base-only context cache growth cannot be used by a speculative lease");
    ELLM_CHECK(count >= 0, "Context cache growth count must be non-negative");
    return growPages(lease, ResourceDemand{count, 0, 0, 0});
}

bool ContextCacheManager::growSpecPages(CacheRequestLease& lease, int32_t baseCount, int32_t draftCount)
{
    ELLM_CHECK(lease.valid() && lease.mManager == this, "Context cache lease does not belong to this manager");
    ELLM_CHECK(lease.mMode == ReusePlanMode::kSpec || lease.mMode == ReusePlanMode::kHybridMtp,
        "Paired base/spec context cache growth requires a spec or hybrid+MTP lease");
    ELLM_CHECK(baseCount >= 0 && draftCount >= 0, "Context cache growth counts must be non-negative");
    return growPages(lease, ResourceDemand{baseCount, draftCount, 0, 0});
}

std::optional<HybridSnapshotReservation> ContextCacheManager::reserveHybridSnapshots(
    CacheRequestLease& lease, bool needsPartialKvSnapshot)
{
    ELLM_CHECK(lease.valid() && lease.mManager == this, "Context cache lease does not belong to this manager");
    ELLM_CHECK(lease.mMode == ReusePlanMode::kHybrid || lease.mMode == ReusePlanMode::kHybridMtp,
        "Hybrid snapshot reservation requires an exact-checkpoint lease");
    ELLM_CHECK(!needsPartialKvSnapshot || lease.mHybridHasAttention,
        "Pure-recurrent context cache cannot reserve a partial KV snapshot");

    size_t const previousResourceCount = lease.mActiveResources.size();
    ResourceDemand const demand{0, 0, 1, needsPartialKvSnapshot ? 1 : 0};
    if (!growPages(lease, demand))
    {
        return std::nullopt;
    }

    HybridSnapshotReservation reservation;
    bool foundRecurrentSnapshot = false;
    for (size_t index = previousResourceCount; index < lease.mActiveResources.size(); ++index)
    {
        ResourceId const resource = lease.mActiveResources[index];
        if (resource.type == ResourceType::kRecurrentSnapshot)
        {
            reservation.recurrentSnapshotSlot = resource.index;
            foundRecurrentSnapshot = true;
        }
        else if (resource.type == ResourceType::kPartialKvSnapshot)
        {
            reservation.partialKvSnapshotSlot = resource.index;
        }
    }
    ELLM_CHECK(foundRecurrentSnapshot && reservation.recurrentSnapshotSlot >= 0
            && reservation.partialKvSnapshotSlot.has_value() == needsPartialKvSnapshot,
        "Hybrid snapshot reservation returned an incomplete resource set");
    return reservation;
}

void ContextCacheManager::releaseRestoredHybridSnapshots(CacheRequestLease& lease)
{
    ELLM_CHECK(lease.valid() && lease.mManager == this
            && (lease.mMode == ReusePlanMode::kHybrid || lease.mMode == ReusePlanMode::kHybridMtp),
        "Restored hybrid snapshots require an exact-checkpoint lease");
    if (lease.mRecurrentSnapshotBinding.has_value())
    {
        releaseLeaseResource(lease, ResourceId{ResourceType::kRecurrentSnapshot, *lease.mRecurrentSnapshotBinding});
        lease.mRecurrentSnapshotBinding.reset();
    }
    if (lease.mPartialKvSnapshotBinding.has_value())
    {
        releaseLeaseResource(lease, ResourceId{ResourceType::kPartialKvSnapshot, *lease.mPartialKvSnapshotBinding});
        lease.mPartialKvSnapshotBinding.reset();
    }
}

void ContextCacheManager::retireHybridSnapshotReservation(
    CacheRequestLease& lease, HybridSnapshotReservation const& reservation)
{
    ELLM_CHECK(lease.valid() && lease.mManager == this
            && (lease.mMode == ReusePlanMode::kHybrid || lease.mMode == ReusePlanMode::kHybridMtp),
        "Hybrid snapshot retirement requires an exact-checkpoint lease");
    releaseLeaseResource(lease, ResourceId{ResourceType::kRecurrentSnapshot, reservation.recurrentSnapshotSlot});
    if (reservation.partialKvSnapshotSlot.has_value())
    {
        releaseLeaseResource(lease, ResourceId{ResourceType::kPartialKvSnapshot, *reservation.partialKvSnapshotSlot});
    }
}

bool ContextCacheManager::growPages(CacheRequestLease& lease, ResourceDemand const& demand)
{
    if (resourceDemandCount(demand) == 0)
    {
        return true;
    }
    EvictionPlan const eviction = EvictionPlanner::plan(demand, mPools, mRecords);
    if (!eviction.feasible)
    {
        return false;
    }

    size_t const growth = resourceDemandCount(demand);
    lease.mActiveResources.reserve(lease.mActiveResources.size() + growth);
    lease.mBasePages.reserve(lease.mBasePages.size() + static_cast<size_t>(demand.baseKvPages));
    lease.mSpecPages.reserve(lease.mSpecPages.size() + static_cast<size_t>(demand.draftKvPages));
    std::vector<ResourceId> allocated(growth);

    applyEviction(eviction);
    ELLM_CHECK(mPools.allocateInto(demand, allocated),
        "Context cache growth allocation failed after a feasible eviction plan");
    for (ResourceId const& resource : allocated)
    {
        lease.mActiveResources.push_back(resource);
        if (resource.type == ResourceType::kBaseKvPage)
        {
            lease.mBasePages.push_back(resource.index);
        }
        else if (resource.type == ResourceType::kDraftKvPage)
        {
            lease.mSpecPages.push_back(resource.index);
        }
    }
    return true;
}

PublishResult ContextCacheManager::commitPreparedPublication(PreparedPublication publication)
{
    PublicationRollbackGuard rollback(
        mPools, mBaseIndex, mSpecIndex, mRecords, publication.cacheResources, publication.missingBaseMappings);
    for (ResourceId const& resource : publication.cacheResources)
    {
        mPools.addCacheRef(resource);
        rollback.cacheRefAdded();
    }
    for (MissingBaseMapping const& missing : publication.missingBaseMappings)
    {
        BaseInsertResult const inserted = mBaseIndex.insert(missing.hash, missing.page);
        ELLM_CHECK(inserted.inserted && inserted.canonicalPage == missing.page,
            "Context cache base index changed during publication");
        rollback.baseMappingInserted();
    }

    RecordInsertResult const inserted = mRecords.insert(std::move(publication.record));
    ELLM_CHECK(inserted.inserted, "Context cache exact record appeared during publication");
    rollback.recordInserted(inserted.id);
    CacheRecord const& record = mRecords.get(inserted.id);
    if (record.specState.has_value())
    {
        mSpecIndex.paged().insert(record);
        rollback.specIndexed();
    }
    rollback.commit();

    if (mRecords.maxRecords() == 0)
    {
        evictRecord(inserted.id);
    }
    else if (publication.recordLimitVictim.has_value())
    {
        evictRecord(*publication.recordLimitVictim);
    }
    ELLM_CHECK(mRecords.size() <= static_cast<size_t>(mRecords.maxRecords()),
        "Context cache publication did not enforce its record limit");

    std::optional<RecordId> retainedRecord;
    if (mRecords.contains(inserted.id))
    {
        retainedRecord = inserted.id;
    }
    return PublishResult{PublishStatus::kPublished, retainedRecord, std::move(publication.canonicalBasePages),
        publication.publishedBaseFullBlockCount};
}

PublishResult ContextCacheManager::publish(CacheRequestLease& lease, PublishRequest const& request)
{
    ELLM_CHECK(lease.valid() && lease.mManager == this, "Context cache lease does not belong to this manager");
    ELLM_CHECK(lease.mMode == ReusePlanMode::kVanilla || lease.mMode == ReusePlanMode::kSpec,
        "Standard context cache publication requires a vanilla or speculative lease");
    ELLM_CHECK(request.residentStateLength >= 0, "Context cache resident state length must be non-negative");

    bool const specPublication = lease.mMode == ReusePlanMode::kSpec;
    ELLM_CHECK(!specPublication || mSpecReuseContract.has_value(),
        "Context cache spec publication requires its registered contract");
    size_t const residentFullBlocks = static_cast<size_t>(request.residentStateLength / mPageSize);
    size_t const requestedPublishCount = std::min(request.fullBlockHashes.size(), residentFullBlocks);
    ELLM_CHECK(requestedPublishCount > 0, "Context cache publication requires at least one resident full block");
    ELLM_CHECK(
        requestedPublishCount <= lease.mBasePages.size(), "Context cache publication exceeds the lease base page path");
    if (specPublication && mSpecReuseContract->ownsPagedSpecState)
    {
        ELLM_CHECK(requestedPublishCount <= lease.mSpecPages.size(),
            "Context cache publication exceeds the lease spec-state page path");
    }
    size_t const matchedPublishCount = std::min(requestedPublishCount, lease.mMatchedBlockHashes.size());
    for (size_t index = 0; index < matchedPublishCount; ++index)
    {
        ELLM_CHECK(request.fullBlockHashes[index] == lease.mMatchedBlockHashes[index],
            "Context cache publication does not match the acquired block prefix");
    }
    if (lease.mSpecReplayDependency.has_value())
    {
        SpecReplayDependency const& dependency = *lease.mSpecReplayDependency;
        ELLM_CHECK(static_cast<size_t>(dependency.pathBlockCount) <= requestedPublishCount
                && request.fullBlockHashes[static_cast<size_t>(dependency.pathBlockCount - 1)]
                    == dependency.terminalHash,
            "Speculative context cache publication does not match its full-page replay dependency");
    }

    size_t const publishCount = requestedPublishCount;

    std::vector<BlockHash> logicalHashes(
        request.fullBlockHashes.begin(), request.fullBlockHashes.begin() + static_cast<std::ptrdiff_t>(publishCount));
    std::optional<SpecPagedStateRecord> specState;
    if (specPublication)
    {
        size_t const specPublishCount = mSpecReuseContract->ownsPagedSpecState ? publishCount : 0;
        std::vector<PageId> specPages(
            lease.mSpecPages.begin(), lease.mSpecPages.begin() + static_cast<std::ptrdiff_t>(specPublishCount));
        SpecLeaseStateView const leaseState{specPages};
        specState = makeSpecPublishedState(
            SpecPublishStateInput{leaseState, logicalHashes, request.residentStateLength}, *mSpecReuseContract);
    }
    CacheRecordKey const key{logicalHashes.back(), static_cast<int32_t>(publishCount)};
    std::optional<RecordId> const existing = mRecords.find(key);
    if (existing.has_value())
    {
        CacheRecord const& existingRecord = mRecords.get(*existing);
        ELLM_CHECK(existingRecord.basePagePath.size() >= publishCount,
            "Existing context cache record is shorter than its exact key");
        std::vector<PageId> canonicalPages(existingRecord.basePagePath.begin(),
            existingRecord.basePagePath.begin() + static_cast<std::ptrdiff_t>(publishCount));
        if (!specState.has_value())
        {
            mRecords.touch(*existing);
            return PublishResult{PublishStatus::kExistingRecord, existing, std::move(canonicalPages),
                static_cast<int32_t>(publishCount)};
        }

        if (existingRecord.specState.has_value())
        {
            // Duplicate producers for the same logical prefix can materialize different physical pages. The first
            // published record remains canonical; later producer pages stay request-private and are released with the
            // lease.
            mRecords.touch(*existing);
            return PublishResult{PublishStatus::kExistingRecord, existing, std::move(canonicalPages),
                static_cast<int32_t>(publishCount)};
        }

        CacheRecord upgradedRecord = existingRecord;
        upgradedRecord.specState = specState;
        std::vector<ResourceId> specResources = specStateResources(*specState);

        SpecUpgradeRollbackGuard rollback(mPools, mSpecIndex, upgradedRecord, specResources);
        rollback.addAll();
        mSpecIndex.paged().insert(upgradedRecord);
        rollback.specIndexed();

        mRecords.setSpecState(*existing, std::move(*specState));
        rollback.commit();
        return PublishResult{
            PublishStatus::kPublished, existing, std::move(canonicalPages), static_cast<int32_t>(publishCount)};
    }

    BaseProjection projection = prepareBaseProjection(mBaseIndex, logicalHashes, lease.mBasePages);

    CacheRecord record;
    record.key = key;
    record.logicalBlockHashes = std::move(logicalHashes);
    record.basePagePath = projection.canonicalPages;
    if (specState.has_value())
    {
        record.specState = std::move(specState);
    }

    PreparedPublication publication;
    publication.canonicalBasePages = projection.canonicalPages;
    publication.missingBaseMappings = std::move(projection.missingMappings);
    publication.record = std::move(record);
    publication.cacheResources = publication.record.resources();
    publication.recordLimitVictim = selectRecordLimitVictim(mRecords);
    publication.publishedBaseFullBlockCount = static_cast<int32_t>(publishCount);
    return commitPreparedPublication(std::move(publication));
}

PublishResult ContextCacheManager::publishHybrid(CacheRequestLease& lease, HybridPublishRequest const& request)
{
    ELLM_CHECK(lease.valid() && lease.mManager == this, "Context cache lease does not belong to this manager");
    ELLM_CHECK(
        lease.mMode == ReusePlanMode::kHybrid, "Hybrid context cache publication requires an exact-checkpoint lease");

    ELLM_CHECK(
        request.checkpoint.exactLength > 0, "Hybrid context cache publication identity does not match its lease");
    size_t const fullBlockCount = static_cast<size_t>(request.checkpoint.exactLength / mPageSize);
    ELLM_CHECK(request.fullBlockHashes.size() == fullBlockCount,
        "Hybrid context cache publication requires every complete logical block before its exact boundary");
    bool const needsPartialSnapshot = lease.mHybridHasAttention && request.checkpoint.exactLength % mPageSize != 0;
    ELLM_CHECK(request.snapshots.recurrentSnapshotSlot >= 0
            && request.snapshots.partialKvSnapshotSlot.has_value() == needsPartialSnapshot,
        "Hybrid context cache publication has an incomplete snapshot set");
    ELLM_CHECK(!lease.mHybridHasAttention || fullBlockCount <= lease.mBasePages.size(),
        "Hybrid context cache publication exceeds its base page path");

    auto ownsActiveResource = [&](ResourceId const& resource) {
        return std::find(lease.mActiveResources.begin(), lease.mActiveResources.end(), resource)
            != lease.mActiveResources.end();
    };
    ELLM_CHECK(
        ownsActiveResource(ResourceId{ResourceType::kRecurrentSnapshot, request.snapshots.recurrentSnapshotSlot}),
        "Hybrid context cache lease does not own the recurrent snapshot reservation");
    if (needsPartialSnapshot)
    {
        ELLM_CHECK(
            ownsActiveResource(ResourceId{ResourceType::kPartialKvSnapshot, *request.snapshots.partialKvSnapshotSlot}),
            "Hybrid context cache lease does not own the partial KV snapshot reservation");
    }

    size_t const matchedBlockCount = std::min(fullBlockCount, lease.mMatchedBlockHashes.size());
    ELLM_CHECK(std::equal(request.fullBlockHashes.begin(),
                   request.fullBlockHashes.begin() + static_cast<std::ptrdiff_t>(matchedBlockCount),
                   lease.mMatchedBlockHashes.begin()),
        "Hybrid context cache publication does not match the acquired logical prefix");

    std::optional<RecordId> const existing = mRecords.findHybrid(request.checkpoint);
    if (existing.has_value())
    {
        CacheRecord const& record = mRecords.get(*existing);
        mRecords.touch(*existing);
        return PublishResult{
            PublishStatus::kExistingRecord, existing, record.basePagePath, static_cast<int32_t>(fullBlockCount)};
    }

    BaseProjection projection;
    if (lease.mHybridHasAttention)
    {
        projection = prepareBaseProjection(mBaseIndex, request.fullBlockHashes, lease.mBasePages);
    }

    CacheRecord record;
    record.key = CacheRecordKey{request.checkpoint.exactPrefixDigest, static_cast<int32_t>(fullBlockCount)};
    record.logicalBlockHashes = request.fullBlockHashes;
    record.basePagePath = projection.canonicalPages;
    record.recurrentSnapshotSlot = request.snapshots.recurrentSnapshotSlot;
    record.partialKvSnapshotSlot = request.snapshots.partialKvSnapshotSlot;
    record.exactCheckpointLength = request.checkpoint.exactLength;

    PreparedPublication publication;
    publication.canonicalBasePages = projection.canonicalPages;
    publication.missingBaseMappings = std::move(projection.missingMappings);
    publication.record = std::move(record);
    publication.cacheResources = publication.record.resources();
    publication.recordLimitVictim = selectRecordLimitVictim(mRecords);
    publication.publishedBaseFullBlockCount = static_cast<int32_t>(fullBlockCount);
    return commitPreparedPublication(std::move(publication));
}

PublishResult ContextCacheManager::publishHybridMtp(CacheRequestLease& lease, HybridPublishRequest const& request)
{
    ELLM_CHECK(lease.valid() && lease.mManager == this, "Context cache lease does not belong to this manager");
    ELLM_CHECK(
        lease.mMode == ReusePlanMode::kHybridMtp, "Hybrid+MTP context cache publication requires a combined lease");

    ELLM_CHECK(
        request.checkpoint.exactLength > 0, "Hybrid+MTP context cache publication identity does not match its lease");
    // This is the local form of SpecReuseContract::futureDependencyTokens == 1. Keep the boundary private so a
    // consumer can rewrite its draft KV without mutating shared state.
    size_t const fullBlockCount = static_cast<size_t>((request.checkpoint.exactLength - 1) / mPageSize);
    ELLM_CHECK(request.fullBlockHashes.size() == fullBlockCount,
        "Hybrid+MTP context cache publication requires every complete logical block before its exact boundary");
    // MTP always publishes a partial page, so both snapshots must always be present.
    bool const needsPartialSnapshot = true;
    ELLM_CHECK(request.snapshots.recurrentSnapshotSlot >= 0
            && request.snapshots.partialKvSnapshotSlot.has_value() == needsPartialSnapshot,
        "Hybrid+MTP context cache publication has an incomplete snapshot set");
    ELLM_CHECK(fullBlockCount <= lease.mBasePages.size() && fullBlockCount <= lease.mSpecPages.size(),
        "Hybrid+MTP context cache publication exceeds its paired base/draft page paths");

    auto ownsActiveResource = [&](ResourceId const& resource) {
        return std::find(lease.mActiveResources.begin(), lease.mActiveResources.end(), resource)
            != lease.mActiveResources.end();
    };
    ELLM_CHECK(
        ownsActiveResource(ResourceId{ResourceType::kRecurrentSnapshot, request.snapshots.recurrentSnapshotSlot}),
        "Hybrid+MTP context cache lease does not own the recurrent snapshot reservation");
    ELLM_CHECK(
        ownsActiveResource(ResourceId{ResourceType::kPartialKvSnapshot, *request.snapshots.partialKvSnapshotSlot}),
        "Hybrid+MTP context cache lease does not own the bundled partial KV snapshot reservation");

    size_t const matchedBlockCount = std::min(fullBlockCount, lease.mMatchedBlockHashes.size());
    ELLM_CHECK(std::equal(request.fullBlockHashes.begin(),
                   request.fullBlockHashes.begin() + static_cast<std::ptrdiff_t>(matchedBlockCount),
                   lease.mMatchedBlockHashes.begin()),
        "Hybrid+MTP context cache publication does not match the acquired logical prefix");

    std::optional<RecordId> const existing = mRecords.findHybrid(request.checkpoint);
    if (existing.has_value())
    {
        CacheRecord const& record = mRecords.get(*existing);
        mRecords.touch(*existing);
        return PublishResult{
            PublishStatus::kExistingRecord, existing, record.basePagePath, static_cast<int32_t>(fullBlockCount)};
    }

    BaseProjection projection = prepareBaseProjection(mBaseIndex, request.fullBlockHashes, lease.mBasePages);

    std::vector<PageId> draftPages(
        lease.mSpecPages.begin(), lease.mSpecPages.begin() + static_cast<std::ptrdiff_t>(fullBlockCount));

    CacheRecord record;
    record.key = CacheRecordKey{request.checkpoint.exactPrefixDigest, static_cast<int32_t>(fullBlockCount)};
    record.logicalBlockHashes = request.fullBlockHashes;
    record.basePagePath = projection.canonicalPages;
    record.specState = SpecPagedStateRecord{std::move(draftPages)};
    record.recurrentSnapshotSlot = request.snapshots.recurrentSnapshotSlot;
    record.partialKvSnapshotSlot = request.snapshots.partialKvSnapshotSlot;
    record.exactCheckpointLength = request.checkpoint.exactLength;

    PreparedPublication publication;
    publication.canonicalBasePages = projection.canonicalPages;
    publication.missingBaseMappings = std::move(projection.missingMappings);
    publication.record = std::move(record);
    publication.cacheResources = publication.record.resources();
    publication.recordLimitVictim = selectRecordLimitVictim(mRecords);
    publication.publishedBaseFullBlockCount = static_cast<int32_t>(fullBlockCount);
    return commitPreparedPublication(std::move(publication));
}
ResourcePools const& ContextCacheManager::pools() const noexcept
{
    return mPools;
}

BaseBlockIndex const& ContextCacheManager::baseIndex() const noexcept
{
    return mBaseIndex;
}

SpecStateIndex const& ContextCacheManager::specIndex() const noexcept
{
    return mSpecIndex;
}

CacheRecordStore const& ContextCacheManager::records() const noexcept
{
    return mRecords;
}

ContextCacheManagerMetrics const& ContextCacheManager::metrics() const noexcept
{
    return mMetrics;
}

void ContextCacheManager::releaseLease(CacheRequestLease& lease) noexcept
{
    lease.mManager = nullptr;
    releaseActiveRefsNoThrow(mPools, lease.mActiveResources);
    lease.mMode = ReusePlanMode::kVanilla;
    lease.mMatchedBlockHashes.clear();
    lease.mActiveResources.clear();
    lease.mBasePages.clear();
    lease.mSpecPages.clear();
    lease.mSpecReplayDependency.reset();
    lease.mHybridHasAttention = false;
    lease.mRecurrentSnapshotBinding.reset();
    lease.mPartialKvSnapshotBinding.reset();
}

void ContextCacheManager::evictRecord(RecordId id)
{
    ResourceDemand const freeBefore{mPools.freeCount(ResourceType::kBaseKvPage),
        mPools.freeCount(ResourceType::kDraftKvPage), mPools.freeCount(ResourceType::kRecurrentSnapshot),
        mPools.freeCount(ResourceType::kPartialKvSnapshot)};
    CacheRecord const record = mRecords.erase(id);
    if (record.specState.has_value())
    {
        mSpecIndex.paged().erase(record);
    }
    for (PageId const page : record.basePagePath)
    {
        ResourceId const resource{ResourceType::kBaseKvPage, page};
        mPools.releaseCacheRef(resource);
        if (mPools.cacheRefCount(resource) == 0)
        {
            mBaseIndex.erasePage(page);
        }
    }
    if (record.specState.has_value())
    {
        for (PageId const page : record.specState->pagePath)
        {
            mPools.releaseCacheRef(ResourceId{ResourceType::kDraftKvPage, page});
        }
    }
    if (record.recurrentSnapshotSlot.has_value())
    {
        mPools.releaseCacheRef(ResourceId{ResourceType::kRecurrentSnapshot, *record.recurrentSnapshotSlot});
    }
    if (record.partialKvSnapshotSlot.has_value())
    {
        mPools.releaseCacheRef(ResourceId{ResourceType::kPartialKvSnapshot, *record.partialKvSnapshotSlot});
    }

    ResourceDemand const freeAfter{mPools.freeCount(ResourceType::kBaseKvPage),
        mPools.freeCount(ResourceType::kDraftKvPage), mPools.freeCount(ResourceType::kRecurrentSnapshot),
        mPools.freeCount(ResourceType::kPartialKvSnapshot)};
    ++mMetrics.evictedRecords;
    mMetrics.reclaimedBaseKvPages += static_cast<uint64_t>(freeAfter.baseKvPages - freeBefore.baseKvPages);
    mMetrics.reclaimedDraftKvPages += static_cast<uint64_t>(freeAfter.draftKvPages - freeBefore.draftKvPages);
    mMetrics.reclaimedRecurrentSnapshots
        += static_cast<uint64_t>(freeAfter.recurrentSnapshotSlots - freeBefore.recurrentSnapshotSlots);
    mMetrics.reclaimedPartialKvSnapshots
        += static_cast<uint64_t>(freeAfter.partialKvSnapshotSlots - freeBefore.partialKvSnapshotSlots);
}

void ContextCacheManager::applyEviction(EvictionPlan const& plan)
{
    for (RecordId const id : plan.victims)
    {
        evictRecord(id);
    }
}

void ContextCacheManager::releaseLeaseResource(CacheRequestLease& lease, ResourceId resource)
{
    auto const activeResource = std::find(lease.mActiveResources.begin(), lease.mActiveResources.end(), resource);
    ELLM_CHECK(
        activeResource != lease.mActiveResources.end(), "Context cache lease does not own the resource being retired");
    lease.mActiveResources.erase(activeResource);
    mPools.releaseActiveRef(resource);
}

} // namespace rt
} // namespace trt_edgellm
