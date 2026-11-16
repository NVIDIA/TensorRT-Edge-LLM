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
#include "common/logger.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iterator>
#include <limits>
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

std::optional<SwaCacheConfig> validateSwaConfiguration(
    int32_t pageSize, ResourceDemand const& capacities, std::optional<SwaCacheConfig> const& swaConfig)
{
    if (!swaConfig.has_value())
    {
        ELLM_CHECK(capacities.swaKvPages == 0, "SWA page capacity requires an SWA cache configuration");
        return std::nullopt;
    }
    ELLM_CHECK(swaConfig->windowSizeTokens > 0, "Context cache SWA window size must be positive");
    ELLM_CHECK(swaConfig->replacementPageCount > 0, "Context cache SWA replacement-page reservation must be positive");
    int64_t const privatePages
        = computeSwaPrivatePagesPerSlot(swaConfig->windowSizeTokens, pageSize, swaConfig->replacementPageCount);
    ELLM_CHECK(privatePages <= std::numeric_limits<int32_t>::max(),
        "Context cache SWA private-page reservation exceeds int32 range");
    ELLM_CHECK(
        capacities.swaKvPages >= privatePages, "Context cache SWA pool cannot admit one complete bounded reservation");
    return swaConfig;
}

struct SwaRetainedRange
{
    int32_t begin{};
    int32_t end{};
};

SwaRetainedRange activeSwaRange(int32_t endpoint, int32_t pageSize, int32_t windowSize) noexcept
{
    int64_t const retainedEnd = (static_cast<int64_t>(endpoint) + pageSize - 1) / pageSize;
    int64_t const windowStart = std::max<int64_t>(0, static_cast<int64_t>(endpoint) - windowSize);
    return SwaRetainedRange{static_cast<int32_t>(windowStart / pageSize), static_cast<int32_t>(retainedEnd)};
}

SwaRetainedRange retainedSwaRange(
    int32_t endpoint, int32_t pageSize, int32_t windowSize, SwaRetentionMode mode) noexcept
{
    SwaRetainedRange const active = activeSwaRange(endpoint, pageSize, windowSize);
    if (mode == SwaRetentionMode::kFullInputRewind)
    {
        return active;
    }
    int32_t const publishableEndpoint = endpoint - endpoint % pageSize;
    SwaRetainedRange const publishable = activeSwaRange(publishableEndpoint, pageSize, windowSize);
    return SwaRetainedRange{std::max(0, publishable.begin - kSWA_REPLAY_HALO_PAGES), active.end};
}

SwaWindowState makeEmptySwaWindowState(int32_t pageSize, SwaCacheConfig const& config)
{
    SwaWindowState state;
    state.pageSizeTokens = pageSize;
    state.windowSizeTokens = config.windowSizeTokens;
    return state;
}

bool hasBasicSwaWindowShape(SwaWindowState const& state, int32_t pageSize, SwaCacheConfig const& config) noexcept
{
    if (state.pageSizeTokens != pageSize || state.windowSizeTokens != config.windowSizeTokens
        || state.exactResidentTokenCount < 0 || state.retainedLogicalPageBegin < 0
        || state.retainedLogicalPageEnd < state.retainedLogicalPageBegin
        || (state.retentionMode != SwaRetentionMode::kLive && state.retentionMode != SwaRetentionMode::kFullInputRewind)
        || (state.retentionMode == SwaRetentionMode::kFullInputRewind && state.exactResidentTokenCount % pageSize != 0))
    {
        return false;
    }
    SwaRetainedRange const expected
        = retainedSwaRange(state.exactResidentTokenCount, pageSize, config.windowSizeTokens, state.retentionMode);
    if (state.retainedLogicalPageBegin != expected.begin || state.retainedLogicalPageEnd != expected.end
        || state.pageBindings.size() != static_cast<size_t>(expected.end - expected.begin))
    {
        return false;
    }
    for (size_t index = 0; index < state.pageBindings.size(); ++index)
    {
        SwaPageBinding const& binding = state.pageBindings[index];
        if (binding.logicalPage != state.retainedLogicalPageBegin + static_cast<int32_t>(index) || binding.page < 0)
        {
            return false;
        }
        for (size_t previous = 0; previous < index; ++previous)
        {
            if (state.pageBindings[previous].page == binding.page)
            {
                return false;
            }
        }
    }
    return true;
}

std::optional<SwaPageBinding> findSwaBinding(SwaWindowState const& state, int32_t logicalPage) noexcept
{
    if (logicalPage < state.retainedLogicalPageBegin || logicalPage >= state.retainedLogicalPageEnd)
    {
        return std::nullopt;
    }
    size_t const index = static_cast<size_t>(logicalPage - state.retainedLogicalPageBegin);
    if (index >= state.pageBindings.size() || state.pageBindings[index].logicalPage != logicalPage)
    {
        return std::nullopt;
    }
    return state.pageBindings[index];
}

std::optional<SwaWindowState> projectSwaWindowState(SwaWindowState const& source, int32_t endpoint, int32_t pageSize,
    SwaCacheConfig const& config, SwaRetentionMode mode) noexcept
{
    if (endpoint < 0 || endpoint > source.exactResidentTokenCount || !hasBasicSwaWindowShape(source, pageSize, config))
    {
        return std::nullopt;
    }

    SwaRetainedRange const retained = retainedSwaRange(endpoint, pageSize, config.windowSizeTokens, mode);
    SwaWindowState projected = makeEmptySwaWindowState(pageSize, config);
    projected.exactResidentTokenCount = endpoint;
    projected.retainedLogicalPageBegin = retained.begin;
    projected.retainedLogicalPageEnd = retained.end;
    projected.retentionMode = mode;
    for (int32_t logicalPage = retained.begin; logicalPage < retained.end; ++logicalPage)
    {
        std::optional<SwaPageBinding> const binding = findSwaBinding(source, logicalPage);
        if (!binding.has_value())
        {
            return std::nullopt;
        }
        projected.pageBindings.push_back(*binding);
    }
    return projected;
}

ReusePlan makeSwaVanillaPlan(std::vector<BlockHash> const& inputFullBlockHashes, int32_t inputTokenCount,
    int32_t pageSize, SwaCacheConfig const& config, int32_t privatePageLimit, CacheRecordStore const& records,
    ContextCacheLookupPolicy lookupPolicy)
{
    ELLM_CHECK(lookupPolicy == ContextCacheLookupPolicy::kUseCache || lookupPolicy == ContextCacheLookupPolicy::kBypass,
        "Context cache SWA plan has an invalid lookup policy");
    ELLM_CHECK(inputTokenCount >= 0, "Context cache input token count must be non-negative");
    int64_t const fullInputBlockCount = static_cast<int64_t>(inputTokenCount) / pageSize;
    ELLM_CHECK(inputFullBlockHashes.size() <= static_cast<size_t>(fullInputBlockCount),
        "Context cache input has more full-block hashes than full token blocks");
    int64_t const totalInputPages = (static_cast<int64_t>(inputTokenCount) + pageSize - 1) / pageSize;
    auto makeCold = [&]() {
        ReusePlan plan;
        plan.kind = inputTokenCount == 0 ? ReusePlanKind::kStandard : ReusePlanKind::kNoReusablePrefix;
        plan.demand.baseKvPages = static_cast<int32_t>(totalInputPages);
        plan.demand.swaKvPages = privatePageLimit;
        plan.swaWindowState = makeEmptySwaWindowState(pageSize, config);
        return plan;
    };
    if (inputTokenCount == 0 || lookupPolicy == ContextCacheLookupPolicy::kBypass)
    {
        return makeCold();
    }

    auto makePlan = [&](RecordId recordId, int32_t matchedBlockCount) -> std::optional<ReusePlan> {
        CacheRecord const& record = records.get(recordId);
        bool const coherentLogicalPath = record.logicalBlockHashes.size() >= static_cast<size_t>(matchedBlockCount)
            && record.basePagePath.size() == record.logicalBlockHashes.size()
            && std::equal(record.logicalBlockHashes.begin(), record.logicalBlockHashes.begin() + matchedBlockCount,
                inputFullBlockHashes.begin());
        if (!coherentLogicalPath || !record.swaWindowState.has_value()
            || !hasBasicSwaWindowShape(*record.swaWindowState, pageSize, config))
        {
            return std::nullopt;
        }

        int32_t const matchedTokenLength = matchedBlockCount * pageSize;
        if (record.swaWindowState->exactResidentTokenCount
            != static_cast<int64_t>(record.logicalBlockHashes.size()) * pageSize)
        {
            return std::nullopt;
        }
        bool const fullInputMatch
            = inputTokenCount % pageSize == 0 && static_cast<int64_t>(matchedBlockCount) == totalInputPages;
        int32_t const reuseBlockCount = matchedBlockCount - static_cast<int32_t>(fullInputMatch);
        int32_t const reuseTokenLength = reuseBlockCount * pageSize;
        SwaRetentionMode const retentionMode
            = fullInputMatch ? SwaRetentionMode::kFullInputRewind : SwaRetentionMode::kLive;
        std::optional<SwaWindowState> projected
            = projectSwaWindowState(*record.swaWindowState, reuseTokenLength, pageSize, config, retentionMode);
        if (!projected.has_value())
        {
            return std::nullopt;
        }

        ReusePlan plan;
        plan.matchedTokenLength = matchedTokenLength;
        plan.reuseTokenLength = reuseTokenLength;
        plan.matchedBlockHashes.assign(
            inputFullBlockHashes.begin(), inputFullBlockHashes.begin() + static_cast<std::ptrdiff_t>(reuseBlockCount));
        plan.basePageBindings.assign(
            record.basePagePath.begin(), record.basePagePath.begin() + static_cast<std::ptrdiff_t>(reuseBlockCount));
        plan.swaWindowState = std::move(projected);
        if (!plan.swaWindowState->pageBindings.empty())
        {
            plan.swaRecord = recordId;
        }
        ELLM_CHECK(plan.swaWindowState->pageBindings.size() <= static_cast<size_t>(privatePageLimit),
            "Context cache SWA record exceeds the private admission bound");
        plan.demand.baseKvPages = static_cast<int32_t>(totalInputPages - reuseBlockCount);
        plan.demand.swaKvPages = privatePageLimit - static_cast<int32_t>(plan.swaWindowState->pageBindings.size())
            - swaReservationPageCredit(retentionMode);
        plan.kind = fullInputMatch ? ReusePlanKind::kFullInputRewind : ReusePlanKind::kStandard;
        return plan;
    };

    ReusePlan best = makeCold();
    int32_t const maxMatchedBlocks = static_cast<int32_t>(inputFullBlockHashes.size());
    for (int32_t matchedBlockCount = maxMatchedBlocks; matchedBlockCount > 0; --matchedBlockCount)
    {
        CacheRecordKey const key{inputFullBlockHashes[static_cast<size_t>(matchedBlockCount - 1)], matchedBlockCount};
        std::optional<RecordId> const recordId = records.find(key);
        if (recordId.has_value())
        {
            auto plan = makePlan(*recordId, matchedBlockCount);
            if (plan.has_value())
            {
                best = std::move(*plan);
                break;
            }
        }
    }
    if (best.matchedTokenLength == maxMatchedBlocks * pageSize)
    {
        return best;
    }

    // A longer record can cover an earlier branch point while its entire window and replay halo remain resident.
    // Pin the owning record's projected pages, never combine SWA state from independent records.
    std::vector<RecordId> const candidates = records.lruToMru();
    for (auto candidate = candidates.rbegin(); candidate != candidates.rend(); ++candidate)
    {
        CacheRecord const& record = records.get(*candidate);
        size_t const limit = std::min(record.logicalBlockHashes.size(), inputFullBlockHashes.size());
        auto const mismatch = std::mismatch(
            record.logicalBlockHashes.begin(), record.logicalBlockHashes.begin() + limit, inputFullBlockHashes.begin());
        int32_t const matchedBlockCount = static_cast<int32_t>(mismatch.first - record.logicalBlockHashes.begin());
        if (matchedBlockCount * pageSize <= best.matchedTokenLength)
        {
            continue;
        }
        auto plan = makePlan(*candidate, matchedBlockCount);
        if (plan.has_value())
        {
            best = std::move(*plan);
            if (matchedBlockCount == maxMatchedBlocks)
            {
                break;
            }
        }
    }
    return best;
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
    count += static_cast<size_t>(demand.swaKvPages);
    return count;
}

size_t totalResourceCount(ReusePlan const& plan) noexcept
{
    return plan.basePageBindings.size() + plan.specPageBindings.size()
        + static_cast<size_t>(plan.recurrentSnapshotBinding.has_value())
        + static_cast<size_t>(plan.partialKvSnapshotBinding.has_value())
        + (plan.swaWindowState.has_value() ? plan.swaWindowState->pageBindings.size() : 0U)
        + resourceDemandCount(plan.demand);
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
        mPublishable = other.mPublishable;
        mMatchedBlockHashes = std::move(other.mMatchedBlockHashes);
        mActiveResources = std::move(other.mActiveResources);
        mBasePages = std::move(other.mBasePages);
        mSpecPages = std::move(other.mSpecPages);
        mSpecReplayDependency = other.mSpecReplayDependency;
        mHybridHasAttention = other.mHybridHasAttention;
        mRecurrentSnapshotBinding = other.mRecurrentSnapshotBinding;
        mPartialKvSnapshotBinding = other.mPartialKvSnapshotBinding;
        mSwaState = std::move(other.mSwaState);

        other.mManager = nullptr;
        other.mMode = ReusePlanMode::kVanilla;
        other.mPublishable = true;
        other.mMatchedBlockHashes.clear();
        other.mActiveResources.clear();
        other.mBasePages.clear();
        other.mSpecPages.clear();
        other.mSpecReplayDependency.reset();
        other.mHybridHasAttention = false;
        other.mRecurrentSnapshotBinding.reset();
        other.mPartialKvSnapshotBinding.reset();
        other.mSwaState.reset();
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
    mPublishable = true;
    mMatchedBlockHashes.clear();
    mActiveResources.clear();
    mBasePages.clear();
    mSpecPages.clear();
    mSpecReplayDependency.reset();
    mHybridHasAttention = false;
    mRecurrentSnapshotBinding.reset();
    mPartialKvSnapshotBinding.reset();
    mSwaState.reset();
}

ContextCacheManager::ContextCacheManager(int32_t pageSize, ResourceDemand capacities, int32_t maxRecords,
    std::optional<SpecReuseContract> specReuseContract, std::optional<SwaCacheConfig> swaConfig)
    : mPageSize(validateManagerConfiguration(pageSize, capacities, maxRecords))
    , mSwaConfig(validateSwaConfiguration(pageSize, capacities, swaConfig))
    , mSwaPrivatePageLimit(mSwaConfig.has_value()
              ? static_cast<int32_t>(computeSwaPrivatePagesPerSlot(
                    mSwaConfig->windowSizeTokens, mPageSize, mSwaConfig->replacementPageCount))
              : 0)
    , mPools(capacities)
    , mSpecReuseContract(specReuseContract)
    , mRecords(maxRecords)
{
    ELLM_CHECK(!mSwaConfig.has_value() || !mSpecReuseContract.has_value(),
        "Bounded SWA context reuse does not support speculative decoding");
}

ReusePlan ContextCacheManager::planVanilla(std::vector<BlockHash> const& inputFullBlockHashes, int32_t inputTokenCount,
    ContextCacheLookupPolicy lookupPolicy) const
{
    if (mSwaConfig.has_value())
    {
        return makeSwaVanillaPlan(inputFullBlockHashes, inputTokenCount, mPageSize, *mSwaConfig, mSwaPrivatePageLimit,
            mRecords, lookupPolicy);
    }
    return makeVanillaReusePlan(inputFullBlockHashes, inputTokenCount, mPageSize, mBaseIndex, lookupPolicy);
}

AcquireResult ContextCacheManager::acquireVanilla(
    std::vector<BlockHash> const& inputFullBlockHashes, int32_t inputTokenCount, ContextCacheLookupPolicy lookupPolicy)
{
    return acquire(planVanilla(inputFullBlockHashes, inputTokenCount, lookupPolicy));
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
    if (mSwaConfig.has_value())
    {
        size_t const sharedSwaPageCount
            = plan.swaWindowState.has_value() ? plan.swaWindowState->pageBindings.size() : 0U;
        ELLM_CHECK(plan.mode == ReusePlanMode::kVanilla && plan.swaWindowState.has_value()
                && plan.swaWindowState->exactResidentTokenCount == plan.reuseTokenLength
                && hasBasicSwaWindowShape(*plan.swaWindowState, mPageSize, *mSwaConfig)
                && plan.swaRecord.has_value() == !plan.swaWindowState->pageBindings.empty()
                && sharedSwaPageCount <= static_cast<size_t>(
                       mSwaPrivatePageLimit - swaReservationPageCredit(plan.swaWindowState->retentionMode))
                && plan.demand.swaKvPages
                    == mSwaPrivatePageLimit - static_cast<int32_t>(sharedSwaPageCount)
                        - swaReservationPageCredit(plan.swaWindowState->retentionMode),
            "SWA context cache plan is missing its bounded admission state");
    }
    else
    {
        ELLM_CHECK(!plan.swaWindowState.has_value() && !plan.swaRecord.has_value() && plan.demand.swaKvPages == 0,
            "SWA state cannot be acquired by a manager with SWA disabled");
    }

    CacheRequestLease lease;
    lease.mManager = this;
    lease.mMode = plan.mode;
    lease.mMatchedBlockHashes = plan.matchedBlockHashes;
    lease.mSpecReplayDependency = plan.specReplayDependency;
    lease.mHybridHasAttention = plan.hybridHasAttention;
    lease.mRecurrentSnapshotBinding = plan.recurrentSnapshotBinding;
    lease.mPartialKvSnapshotBinding = plan.partialKvSnapshotBinding;
    if (plan.swaWindowState.has_value())
    {
        SwaKVCacheState state;
        static_cast<SwaWindowState&>(state) = *plan.swaWindowState;
        state.reservedPages.reserve(static_cast<size_t>(plan.demand.swaKvPages));
        lease.mSwaState = std::move(state);
    }
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
    if (plan.swaWindowState.has_value())
    {
        for (SwaPageBinding const& binding : plan.swaWindowState->pageBindings)
        {
            ResourceId const resource{ResourceType::kSwaKvPage, binding.page};
            mPools.addActiveRef(resource);
            lease.mActiveResources.push_back(resource);
        }
    }

    std::optional<RecordId> protectedRecord = plan.specRecord.has_value() ? plan.specRecord : plan.hybridRecord;
    if (!protectedRecord.has_value())
    {
        protectedRecord = plan.swaRecord;
    }
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
        else if (resource.type == ResourceType::kSwaKvPage)
        {
            ELLM_CHECK(lease.mSwaState.has_value(), "SWA allocation has no request state");
            lease.mSwaState->reservedPages.push_back(resource.index);
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
    if (plan.swaRecord.has_value() && mRecords.contains(*plan.swaRecord))
    {
        mRecords.touch(*plan.swaRecord);
    }

    return AcquireResult{std::optional<CacheRequestLease>{std::move(lease)}, AcquireStatus::kAcquired, std::move(plan)};
}

bool ContextCacheManager::growBasePages(CacheRequestLease& lease, int32_t count)
{
    ELLM_CHECK(lease.mPublishable, "Read-only prefix pins cannot allocate pages");
    ELLM_CHECK(lease.valid() && lease.mManager == this, "Context cache lease does not belong to this manager");
    ELLM_CHECK(lease.mMode == ReusePlanMode::kVanilla || lease.mMode == ReusePlanMode::kHybrid,
        "Base-only context cache growth cannot be used by a speculative lease");
    ELLM_CHECK(count >= 0, "Context cache growth count must be non-negative");
    return growPages(lease, ResourceDemand{count, 0, 0, 0});
}

bool ContextCacheManager::growSpecPages(CacheRequestLease& lease, int32_t baseCount, int32_t draftCount)
{
    ELLM_CHECK(lease.mPublishable, "Read-only prefix pins cannot allocate pages");
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

CacheRequestLease ContextCacheManager::retainBasePages(std::vector<PageId> const& pages)
{
    CacheRequestLease lease;
    lease.mManager = this;
    lease.mPublishable = false;
    lease.mActiveResources.reserve(pages.size());
    for (PageId const page : pages)
    {
        ResourceId const resource{ResourceType::kBaseKvPage, page};
        mPools.addActiveRef(resource);
        lease.mActiveResources.push_back(resource);
    }
    return lease;
}

std::optional<SwaKVCacheState> ContextCacheManager::takeSwaState(CacheRequestLease& lease)
{
    ELLM_CHECK(lease.valid() && lease.mManager == this, "Context cache lease does not belong to this manager");
    if (!mSwaConfig.has_value())
    {
        ELLM_CHECK(!lease.mSwaState.has_value(), "Non-SWA context cache lease contains SWA state");
        return std::nullopt;
    }
    ELLM_CHECK(lease.mSwaState.has_value(), "Context cache SWA admission state was already transferred");
    return std::exchange(lease.mSwaState, std::nullopt);
}

bool ContextCacheManager::isExternallyOwnedSwaPage(CacheRequestLease const& lease, PageId page) const
{
    ELLM_CHECK(lease.valid() && lease.mManager == this, "Context cache lease does not belong to this manager");
    ResourceId const resource{ResourceType::kSwaKvPage, page};
    ELLM_CHECK(std::find(lease.mActiveResources.begin(), lease.mActiveResources.end(), resource)
            != lease.mActiveResources.end(),
        "Context cache lease does not own the queried SWA page");
    return mPools.cacheRefCount(resource) > 0 || mPools.activeRefCount(resource) > 1;
}

PageId ContextCacheManager::replaceSwaPage(CacheRequestLease& lease, PageId page)
{
    ELLM_CHECK(mSwaConfig.has_value(), "SWA page replacement requires an SWA-enabled context cache");
    ELLM_CHECK(lease.valid() && lease.mManager == this, "Context cache lease does not belong to this manager");
    releaseLeaseResource(lease, ResourceId{ResourceType::kSwaKvPage, page});
    size_t const previousResourceCount = lease.mActiveResources.size();
    ELLM_CHECK(growPages(lease, ResourceDemand{0, 0, 0, 0, 1}),
        "Context cache could not restore the SWA replacement reservation");
    ELLM_CHECK(lease.mActiveResources.size() == previousResourceCount + 1
            && lease.mActiveResources.back().type == ResourceType::kSwaKvPage,
        "Context cache SWA replacement returned an invalid resource");
    PageId const replacement = lease.mActiveResources.back().index;
    ELLM_CHECK(replacement >= 0 && replacement < mPools.capacity(ResourceType::kSwaKvPage),
        "Context cache SWA replacement page is outside the physical pool");
    return replacement;
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
    ELLM_CHECK(lease.mPublishable, "Read-only prefix pins cannot publish context");
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
    std::optional<SwaWindowState> publishedSwaState;
    if (mSwaConfig.has_value())
    {
        ELLM_CHECK(!specPublication && request.swaWindowState.has_value()
                && request.swaWindowState->exactResidentTokenCount == request.residentStateLength,
            "Context cache SWA publication requires a settled vanilla endpoint");
        int32_t const publishedEndpoint = static_cast<int32_t>(publishCount) * mPageSize;
        publishedSwaState = projectSwaWindowState(
            *request.swaWindowState, publishedEndpoint, mPageSize, *mSwaConfig, SwaRetentionMode::kLive);
        ELLM_CHECK(publishedSwaState.has_value(),
            "Context cache SWA publication is missing its coherent window or replay page");
        for (SwaPageBinding const& binding : publishedSwaState->pageBindings)
        {
            ResourceId const resource{ResourceType::kSwaKvPage, binding.page};
            ELLM_CHECK(std::find(lease.mActiveResources.begin(), lease.mActiveResources.end(), resource)
                    != lease.mActiveResources.end(),
                "Context cache SWA publication references a page outside its lease");
        }
    }
    else
    {
        ELLM_CHECK(!request.swaWindowState.has_value(), "SWA state cannot be published by a manager with SWA disabled");
    }

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
        if (mSwaConfig.has_value())
        {
            ELLM_CHECK(existingRecord.swaWindowState.has_value()
                    && hasBasicSwaWindowShape(*existingRecord.swaWindowState, mPageSize, *mSwaConfig)
                    && existingRecord.swaWindowState->exactResidentTokenCount
                        == static_cast<int32_t>(publishCount) * mPageSize,
                "Existing context cache record has no coherent SWA endpoint");
        }
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
    record.swaWindowState = std::move(publishedSwaState);

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
        lease.mMode == ReusePlanMode::kHybridMtp, "Hybrid+spec context cache publication requires a combined lease");

    ELLM_CHECK(
        request.checkpoint.exactLength > 0, "Hybrid+spec context cache publication identity does not match its lease");
    // The coordinator computes the publishable full-block count using the mode's futureDependencyTokens; trust
    // the hash vector size as the authoritative block count rather than recomputing locally.
    size_t const fullBlockCount = request.fullBlockHashes.size();
    bool const hasPartialSnapshot = request.snapshots.partialKvSnapshotSlot.has_value();
    ELLM_CHECK(request.snapshots.recurrentSnapshotSlot >= 0,
        "Hybrid+spec context cache publication has no recurrent snapshot");
    ELLM_CHECK(fullBlockCount <= lease.mBasePages.size() && fullBlockCount <= lease.mSpecPages.size(),
        "Hybrid+spec context cache publication exceeds its paired base/draft page paths");

    auto ownsActiveResource = [&](ResourceId const& resource) {
        return std::find(lease.mActiveResources.begin(), lease.mActiveResources.end(), resource)
            != lease.mActiveResources.end();
    };
    ELLM_CHECK(
        ownsActiveResource(ResourceId{ResourceType::kRecurrentSnapshot, request.snapshots.recurrentSnapshotSlot}),
        "Hybrid+spec context cache lease does not own the recurrent snapshot reservation");
    if (hasPartialSnapshot)
    {
        ELLM_CHECK(
            ownsActiveResource(ResourceId{ResourceType::kPartialKvSnapshot, *request.snapshots.partialKvSnapshotSlot}),
            "Hybrid+spec context cache lease does not own the bundled partial KV snapshot reservation");
    }

    size_t const matchedBlockCount = std::min(fullBlockCount, lease.mMatchedBlockHashes.size());
    ELLM_CHECK(std::equal(request.fullBlockHashes.begin(),
                   request.fullBlockHashes.begin() + static_cast<std::ptrdiff_t>(matchedBlockCount),
                   lease.mMatchedBlockHashes.begin()),
        "Hybrid+spec context cache publication does not match the acquired logical prefix");

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
    lease.mSwaState.reset();
}

void ContextCacheManager::evictRecord(RecordId id)
{
    ResourceDemand const freeBefore{mPools.freeCount(ResourceType::kBaseKvPage),
        mPools.freeCount(ResourceType::kDraftKvPage), mPools.freeCount(ResourceType::kRecurrentSnapshot),
        mPools.freeCount(ResourceType::kPartialKvSnapshot), mPools.freeCount(ResourceType::kSwaKvPage)};
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
    if (record.swaWindowState.has_value())
    {
        for (SwaPageBinding const& binding : record.swaWindowState->pageBindings)
        {
            mPools.releaseCacheRef(ResourceId{ResourceType::kSwaKvPage, binding.page});
        }
    }

    ResourceDemand const freeAfter{mPools.freeCount(ResourceType::kBaseKvPage),
        mPools.freeCount(ResourceType::kDraftKvPage), mPools.freeCount(ResourceType::kRecurrentSnapshot),
        mPools.freeCount(ResourceType::kPartialKvSnapshot), mPools.freeCount(ResourceType::kSwaKvPage)};
    ++mMetrics.evictedRecords;
    mMetrics.reclaimedBaseKvPages += static_cast<uint64_t>(freeAfter.baseKvPages - freeBefore.baseKvPages);
    mMetrics.reclaimedDraftKvPages += static_cast<uint64_t>(freeAfter.draftKvPages - freeBefore.draftKvPages);
    mMetrics.reclaimedRecurrentSnapshots
        += static_cast<uint64_t>(freeAfter.recurrentSnapshotSlots - freeBefore.recurrentSnapshotSlots);
    mMetrics.reclaimedPartialKvSnapshots
        += static_cast<uint64_t>(freeAfter.partialKvSnapshotSlots - freeBefore.partialKvSnapshotSlots);
    mMetrics.reclaimedSwaKvPages += static_cast<uint64_t>(freeAfter.swaKvPages - freeBefore.swaKvPages);
    // Checkpoint records only: one attention-only acquisition can evict many records.
    if (record.exactCheckpointLength.has_value())
    {
        LOG_INFO("Context cache eviction #%llu: least recently used checkpoint at %d tokens",
            static_cast<unsigned long long>(mMetrics.evictedRecords), *record.exactCheckpointLength);
    }
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
