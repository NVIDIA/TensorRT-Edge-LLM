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

void releaseActiveRefsNoThrow(ResourcePools& pools, std::vector<ResourceId> const& resources) noexcept
{
    // These IDs came from a validated lease. A release failure means corrupted manager-owned metadata and is not
    // recoverable from a destructor, so the noexcept boundary intentionally terminates instead of masking it.
    for (ResourceId const& resource : resources)
    {
        pools.releaseActiveRef(resource);
    }
}

void releaseCacheRefsNoThrow(
    ResourcePools& pools, ResourceType type, std::vector<PageId> const& pages, size_t count) noexcept
{
    // Publication rollback cannot safely propagate a second exception while preserving the original failure.
    while (count > 0)
    {
        --count;
        pools.releaseCacheRef(ResourceId{type, pages[count]});
    }
}

void validatePlan(ReusePlan const& plan, int32_t pageSize)
{
    ELLM_CHECK(plan.lookupPolicy == LookupPolicy::kUseCache || plan.lookupPolicy == LookupPolicy::kBypass,
        "Context cache plan has an invalid lookup policy");
    ELLM_CHECK(plan.matchedBlockHashes.size() == plan.basePageBindings.size(),
        "Context cache plan hash and base binding counts must match");
    ELLM_CHECK(plan.demand.isNonNegative(), "Context cache plan demand must be non-negative");
    ELLM_CHECK(plan.inputTokenCount >= 0 && plan.reuseTokenLength >= 0,
        "Context cache plan token lengths must be non-negative");
    for (PageId const page : plan.basePageBindings)
    {
        ELLM_CHECK(page >= 0, "Context cache plan base page bindings must be non-negative");
    }
    for (PageId const page : plan.draftPageBindings)
    {
        ELLM_CHECK(page >= 0, "Context cache plan draft page bindings must be non-negative");
    }

    ELLM_CHECK(plan.basePageBindings.size() <= static_cast<size_t>(std::numeric_limits<int32_t>::max()),
        "Context cache plan contains too many base page bindings");
    if (plan.lookupPolicy == LookupPolicy::kBypass)
    {
        ELLM_CHECK(plan.reuseTokenLength == 0 && plan.matchedBlockHashes.empty() && plan.basePageBindings.empty()
                && plan.baseCowSources.empty() && !plan.draftRecord.has_value() && plan.draftPageBindings.empty()
                && plan.draftCowSources.empty() && !plan.specReplayDependency.has_value()
                && !plan.hybridCheckpoint.has_value() && !plan.hybridRecord.has_value()
                && !plan.hybridMtpCheckpoint.has_value() && !plan.recurrentSnapshotBinding.has_value()
                && !plan.partialKvSnapshotBinding.has_value(),
            "Bypass context cache plan contains reusable state");
    }
    int64_t const totalBasePages = (static_cast<int64_t>(plan.inputTokenCount) + pageSize - 1) / pageSize;
    bool validMode{};
    switch (plan.mode)
    {
    case ReusePlanMode::kVanilla:
    {
        validMode = true;
        ELLM_CHECK(
            plan.baseCowSources.empty(), "Vanilla context cache acquisition does not support base copy-on-write");
        ELLM_CHECK(!plan.draftSignature.has_value() && !plan.draftRecord.has_value() && plan.draftPageBindings.empty()
                && plan.draftCowSources.empty() && !plan.specReplayDependency.has_value()
                && plan.specReplayMode == SpecReplayMode::kNone && !plan.recurrentStateSchema.has_value()
                && !plan.hybridCheckpoint.has_value() && !plan.hybridRecord.has_value()
                && !plan.hybridMtpCheckpoint.has_value() && !plan.recurrentSnapshotBinding.has_value()
                && !plan.partialKvSnapshotBinding.has_value(),
            "Vanilla context cache plan contains speculative state");
        int64_t const bindingCount = static_cast<int64_t>(plan.basePageBindings.size());
        int64_t const expectedReuseLength = bindingCount * static_cast<int64_t>(pageSize);
        ELLM_CHECK(expectedReuseLength == static_cast<int64_t>(plan.reuseTokenLength)
                && expectedReuseLength <= static_cast<int64_t>(plan.inputTokenCount),
            "Context cache plan reuse length is inconsistent with its base bindings");
        int64_t const expectedBaseDemand = totalBasePages - bindingCount;
        ELLM_CHECK(expectedBaseDemand >= 0 && expectedBaseDemand == static_cast<int64_t>(plan.demand.baseKvPages),
            "Context cache plan base demand is inconsistent with its token lengths");
        ELLM_CHECK(plan.inputTokenCount == 0 || plan.inputTokenCount % pageSize != 0 || bindingCount != totalBasePages,
            "Context cache plan must rewind an exact block-aligned input match");
        break;
    }
    case ReusePlanMode::kHybrid:
    {
        validMode = true;
        ELLM_CHECK(
            plan.recurrentStateSchema.has_value(), "Hybrid context cache plan is missing its recurrent-state schema");
        ELLM_CHECK(!plan.draftSignature.has_value() && !plan.draftRecord.has_value() && plan.draftPageBindings.empty()
                && plan.baseCowSources.empty() && plan.draftCowSources.empty() && !plan.specReplayDependency.has_value()
                && plan.specReplayMode == SpecReplayMode::kNone && !plan.hybridMtpCheckpoint.has_value(),
            "Hybrid context cache plan contains speculative state");
        ELLM_CHECK(plan.demand.draftKvPages == 0 && plan.demand.recurrentSnapshotSlots == 0
                && plan.demand.partialKvSnapshotSlots == 0,
            "Hybrid acquisition demand may contain only private base pages");

        bool const hasCheckpoint = plan.hybridCheckpoint.has_value();
        ELLM_CHECK(hasCheckpoint == plan.hybridRecord.has_value()
                && hasCheckpoint == plan.recurrentSnapshotBinding.has_value(),
            "Hybrid context cache hit metadata must be present together");
        if (hasCheckpoint)
        {
            ELLM_CHECK(plan.hybridCheckpoint->domain == plan.domain
                    && plan.hybridCheckpoint->schema == *plan.recurrentStateSchema
                    && plan.hybridCheckpoint->exactLength == plan.reuseTokenLength && plan.reuseTokenLength > 0
                    && plan.reuseTokenLength < plan.inputTokenCount,
                "Hybrid context cache checkpoint identity is inconsistent with the plan");
            bool const partial = plan.reuseTokenLength % pageSize != 0;
            ELLM_CHECK(plan.partialKvSnapshotBinding.has_value() == (plan.hybridHasAttention && partial),
                "Hybrid context cache partial snapshot does not match its exact boundary");
        }
        else
        {
            ELLM_CHECK(plan.reuseTokenLength == 0 && !plan.partialKvSnapshotBinding.has_value(),
                "Hybrid context cache miss contains reusable checkpoint state");
        }

        int64_t const sharedPageCount = static_cast<int64_t>(plan.basePageBindings.size());
        int64_t const expectedSharedPages
            = plan.hybridHasAttention ? static_cast<int64_t>(plan.reuseTokenLength / pageSize) : 0;
        ELLM_CHECK(sharedPageCount == expectedSharedPages,
            "Hybrid context cache base path does not match its exact checkpoint length");
        int64_t const expectedDemand = plan.hybridHasAttention ? totalBasePages - sharedPageCount : 0;
        ELLM_CHECK(expectedDemand >= 0 && expectedDemand == static_cast<int64_t>(plan.demand.baseKvPages),
            "Hybrid context cache private page demand is inconsistent with its checkpoint");
        break;
    }
    case ReusePlanMode::kHybridMtp:
    {
        validMode = true;
        ELLM_CHECK(plan.hybridHasAttention && plan.recurrentStateSchema.has_value() && plan.draftSignature.has_value(),
            "Hybrid+MTP context cache plan is missing its model identity");
        ELLM_CHECK(!plan.draftRecord.has_value() && plan.baseCowSources.empty() && plan.draftCowSources.empty()
                && !plan.specReplayDependency.has_value() && plan.specReplayMode == SpecReplayMode::kNone
                && !plan.hybridCheckpoint.has_value(),
            "Hybrid+MTP context cache plan contains an incompatible replay path");
        ELLM_CHECK(plan.matchedBlockHashes.size() == plan.draftPageBindings.size(),
            "Hybrid+MTP context cache plan hash and draft binding counts must match");
        ELLM_CHECK(plan.demand.recurrentSnapshotSlots == 0 && plan.demand.partialKvSnapshotSlots == 0,
            "Hybrid+MTP acquisition demand may contain only private base and draft pages");

        bool const hasCheckpoint = plan.hybridMtpCheckpoint.has_value();
        ELLM_CHECK(hasCheckpoint == plan.hybridRecord.has_value()
                && hasCheckpoint == plan.recurrentSnapshotBinding.has_value(),
            "Hybrid+MTP context cache hit metadata must be present together");
        if (hasCheckpoint)
        {
            ELLM_CHECK(plan.hybridMtpCheckpoint->domain == plan.domain
                    && plan.hybridMtpCheckpoint->schema == *plan.recurrentStateSchema
                    && plan.hybridMtpCheckpoint->draftSignature == *plan.draftSignature
                    && plan.hybridMtpCheckpoint->exactLength == plan.reuseTokenLength && plan.reuseTokenLength > 0
                    && plan.reuseTokenLength < plan.inputTokenCount,
                "Hybrid+MTP context cache checkpoint identity is inconsistent with the plan");
            // The boundary token (reuseTokenLength - 1) is always retained in a private partial page, so every
            // Hybrid+MTP hit carries a partial snapshot regardless of page alignment.
            bool const partial = true;
            ELLM_CHECK(plan.partialKvSnapshotBinding.has_value() == partial,
                "Hybrid+MTP context cache partial snapshot does not match its exact boundary");
        }
        else
        {
            ELLM_CHECK(plan.reuseTokenLength == 0 && !plan.partialKvSnapshotBinding.has_value(),
                "Hybrid+MTP context cache miss contains reusable checkpoint state");
        }

        int64_t const sharedPageCount = static_cast<int64_t>(plan.basePageBindings.size());
        // One fewer full block than reuseTokenLength/pageSize is shared: the boundary token lives in the private
        // partial page. Equivalent to the natural split for non-aligned lengths, and one page fewer for aligned
        // lengths.
        int64_t const expectedSharedPages = static_cast<int64_t>((plan.reuseTokenLength - 1) / pageSize);
        ELLM_CHECK(sharedPageCount == expectedSharedPages,
            "Hybrid+MTP context cache paths do not match the exact checkpoint length");
        int64_t const expectedDemand = totalBasePages - sharedPageCount;
        ELLM_CHECK(expectedDemand >= 0 && expectedDemand == static_cast<int64_t>(plan.demand.baseKvPages)
                && expectedDemand == static_cast<int64_t>(plan.demand.draftKvPages),
            "Hybrid+MTP private page demand is inconsistent with its checkpoint");
        break;
    }
    case ReusePlanMode::kSpecEagle:
    {
        validMode = true;
        ELLM_CHECK(plan.draftSignature.has_value(), "EAGLE context cache plan is missing its draft signature");
        ELLM_CHECK(plan.matchedBlockHashes.size() == plan.draftPageBindings.size(),
            "EAGLE context cache plan hash and draft binding counts must match");
        ELLM_CHECK(plan.baseCowSources.size() == plan.draftCowSources.size() && plan.baseCowSources.size() <= 1,
            "EAGLE context cache plan requires paired single-page copy-on-write sources");
        ELLM_CHECK(plan.draftRecord.has_value() == !plan.draftPageBindings.empty(),
            "EAGLE context cache plan draft record and bindings must be present together");

        bool validReplayMode{};
        switch (plan.specReplayMode)
        {
        case SpecReplayMode::kNone:
            validReplayMode = true;
            ELLM_CHECK(
                plan.basePageBindings.empty() && plan.baseCowSources.empty() && !plan.specReplayDependency.has_value(),
                "EAGLE paired reuse requires boundary replay");
            break;
        case SpecReplayMode::kOneToken:
            validReplayMode = true;
            ELLM_CHECK(!plan.basePageBindings.empty() && plan.baseCowSources.size() == 1
                    && plan.baseCowSources.back() == plan.basePageBindings.back()
                    && plan.draftCowSources.back() == plan.draftPageBindings.back()
                    && !plan.specReplayDependency.has_value(),
                "EAGLE one-token replay requires the final paired pages as copy-on-write sources");
            break;
        case SpecReplayMode::kFullPage:
            validReplayMode = true;
            ELLM_CHECK(plan.baseCowSources.empty(), "EAGLE full-page replay cannot use copy-on-write sources");
            ELLM_CHECK(plan.specReplayDependency.has_value() == !plan.basePageBindings.empty(),
                "EAGLE full-page reuse requires its original coherent draft boundary");
            if (plan.specReplayDependency.has_value())
            {
                ELLM_CHECK(
                    plan.specReplayDependency->pathBlockCount == static_cast<int64_t>(plan.basePageBindings.size()) + 1,
                    "EAGLE full-page replay dependency is inconsistent with its retained pages");
            }
            break;
        }
        ELLM_CHECK(validReplayMode, "Context cache plan has an invalid EAGLE replay mode");
        ELLM_CHECK(plan.demand.recurrentSnapshotSlots == 0 && plan.demand.partialKvSnapshotSlots == 0,
            "EAGLE context cache plan cannot contain hybrid snapshots");
        ELLM_CHECK(!plan.recurrentStateSchema.has_value() && !plan.hybridCheckpoint.has_value()
                && !plan.hybridMtpCheckpoint.has_value() && !plan.hybridRecord.has_value()
                && !plan.recurrentSnapshotBinding.has_value() && !plan.partialKvSnapshotBinding.has_value(),
            "EAGLE context cache plan contains hybrid checkpoint state");

        int64_t const bindingCount = static_cast<int64_t>(plan.basePageBindings.size());
        int64_t const cowCount = static_cast<int64_t>(plan.baseCowSources.size());
        int64_t const sharedPageCount = bindingCount - cowCount;
        int64_t const expectedReuseLength = plan.specReplayMode == SpecReplayMode::kOneToken
            ? bindingCount * static_cast<int64_t>(pageSize) - 1
            : sharedPageCount * static_cast<int64_t>(pageSize);
        ELLM_CHECK(expectedReuseLength == static_cast<int64_t>(plan.reuseTokenLength)
                && expectedReuseLength <= static_cast<int64_t>(plan.inputTokenCount),
            "EAGLE context cache replay length is inconsistent with its bindings");
        int64_t const expectedDemand = totalBasePages - sharedPageCount;
        ELLM_CHECK(expectedDemand >= 0 && expectedDemand == static_cast<int64_t>(plan.demand.baseKvPages)
                && expectedDemand == static_cast<int64_t>(plan.demand.draftKvPages),
            "EAGLE context cache plan demand is inconsistent with its replay boundary");
        ELLM_CHECK(
            plan.inputTokenCount == 0 || plan.inputTokenCount % pageSize != 0 || sharedPageCount != totalBasePages,
            "EAGLE context cache plan must rewind an exact block-aligned input match");
        break;
    }
    }
    ELLM_CHECK(validMode, "Context cache plan has an invalid reuse mode");
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
    return plan.basePageBindings.size() + plan.draftPageBindings.size()
        + static_cast<size_t>(plan.recurrentSnapshotBinding.has_value())
        + static_cast<size_t>(plan.partialKvSnapshotBinding.has_value()) + resourceDemandCount(plan.demand);
}

struct MissingBaseMapping
{
    BaseBlockKey key;
    PageId page{};
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
        mDomain = other.mDomain;
        mDraftSignature = other.mDraftSignature;
        mReuseTokenLength = other.mReuseTokenLength;
        mMatchedBlockHashes = std::move(other.mMatchedBlockHashes);
        mActiveResources = std::move(other.mActiveResources);
        mBasePages = std::move(other.mBasePages);
        mDraftPages = std::move(other.mDraftPages);
        mBaseCowSources = std::move(other.mBaseCowSources);
        mDraftCowSources = std::move(other.mDraftCowSources);
        mSpecReplayDependency = other.mSpecReplayDependency;
        mHybridCheckpoint = other.mHybridCheckpoint;
        mHybridMtpCheckpoint = other.mHybridMtpCheckpoint;
        mHybridRecord = other.mHybridRecord;
        mRecurrentStateSchema = other.mRecurrentStateSchema;
        mHybridHasAttention = other.mHybridHasAttention;
        mRecurrentSnapshotBinding = other.mRecurrentSnapshotBinding;
        mPartialKvSnapshotBinding = other.mPartialKvSnapshotBinding;

        other.mManager = nullptr;
        other.mMode = ReusePlanMode::kVanilla;
        other.mDomain = {};
        other.mDraftSignature.reset();
        other.mReuseTokenLength = 0;
        other.mMatchedBlockHashes.clear();
        other.mActiveResources.clear();
        other.mBasePages.clear();
        other.mDraftPages.clear();
        other.mBaseCowSources.clear();
        other.mDraftCowSources.clear();
        other.mSpecReplayDependency.reset();
        other.mHybridCheckpoint.reset();
        other.mHybridMtpCheckpoint.reset();
        other.mHybridRecord.reset();
        other.mRecurrentStateSchema.reset();
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
    return mDraftPages;
}

std::vector<PageId> const& CacheRequestLease::baseCowSources() const noexcept
{
    return mBaseCowSources;
}

std::vector<PageId> const& CacheRequestLease::draftCowSources() const noexcept
{
    return mDraftCowSources;
}

std::optional<int32_t> CacheRequestLease::recurrentSnapshotSlot() const noexcept
{
    return mRecurrentSnapshotBinding;
}

std::optional<int32_t> CacheRequestLease::partialKvSnapshotSlot() const noexcept
{
    return mPartialKvSnapshotBinding;
}

int32_t CacheRequestLease::reuseTokenLength() const noexcept
{
    return mReuseTokenLength;
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
    mDomain = {};
    mDraftSignature.reset();
    mReuseTokenLength = 0;
    mMatchedBlockHashes.clear();
    mActiveResources.clear();
    mBasePages.clear();
    mDraftPages.clear();
    mBaseCowSources.clear();
    mDraftCowSources.clear();
    mSpecReplayDependency.reset();
    mHybridCheckpoint.reset();
    mHybridMtpCheckpoint.reset();
    mHybridRecord.reset();
    mRecurrentStateSchema.reset();
    mHybridHasAttention = false;
    mRecurrentSnapshotBinding.reset();
    mPartialKvSnapshotBinding.reset();
}

ContextCacheManager::ContextCacheManager(int32_t pageSize, ResourceDemand capacities, int32_t maxRecords)
    : mPageSize(validateManagerConfiguration(pageSize, capacities, maxRecords))
    , mPools(capacities)
    , mRecords(maxRecords)
{
}

ReusePlan ContextCacheManager::planVanilla(CacheDomainId domain, std::vector<BlockHash> const& inputFullBlockHashes,
    int32_t inputTokenCount, LookupPolicy lookupPolicy) const
{
    return makeVanillaReusePlan(domain, inputFullBlockHashes, inputTokenCount, mPageSize, mBaseIndex, lookupPolicy);
}

ReusePlan ContextCacheManager::planHybrid(CacheDomainId domain, RecurrentStateSchemaId schema,
    std::vector<HybridCheckpointCandidate> const& candidates, std::vector<BlockHash> const& inputFullBlockHashes,
    int32_t inputTokenCount, bool hasAttention, LookupPolicy lookupPolicy) const
{
    return makeHybridReusePlan(domain, schema, candidates, inputFullBlockHashes, inputTokenCount, mPageSize,
        hasAttention, mRecords, lookupPolicy);
}

ReusePlan ContextCacheManager::planHybridMtp(CacheDomainId domain, RecurrentStateSchemaId schema,
    DraftEngineSignature draftSignature, std::vector<HybridMtpCheckpointCandidate> const& candidates,
    std::vector<BlockHash> const& inputFullBlockHashes, int32_t inputTokenCount, LookupPolicy lookupPolicy) const
{
    return makeHybridMtpReusePlan(domain, schema, draftSignature, candidates, inputFullBlockHashes, inputTokenCount,
        mPageSize, mRecords, lookupPolicy);
}

std::vector<int32_t> ContextCacheManager::hybridCandidateLengths(
    CacheDomainId domain, RecurrentStateSchemaId schema, int32_t inputTokenCount) const
{
    return mRecords.hybridCandidateLengths(domain, schema, inputTokenCount);
}

std::vector<int32_t> ContextCacheManager::hybridMtpCandidateLengths(CacheDomainId domain, RecurrentStateSchemaId schema,
    DraftEngineSignature draftSignature, int32_t inputTokenCount) const
{
    return mRecords.hybridMtpCandidateLengths(domain, schema, draftSignature, inputTokenCount);
}

ReusePlan ContextCacheManager::planSpec(SpecDecodeMode mode, CacheDomainId domain, DraftEngineSignature draftSignature,
    std::vector<BlockHash> const& inputFullBlockHashes, int32_t inputTokenCount, bool supportsOneTokenReplay,
    LookupPolicy lookupPolicy) const
{
    return makeSpecReusePlan(mode, domain, draftSignature, inputFullBlockHashes, inputTokenCount, mPageSize,
        supportsOneTokenReplay, mBaseIndex, mDraftIndex, mRecords, lookupPolicy);
}

AcquireResult ContextCacheManager::acquire(ReusePlan const& plan)
{
    validatePlan(plan, mPageSize);

    for (size_t index = 0; index < plan.matchedBlockHashes.size(); ++index)
    {
        std::optional<PageId> const current
            = mBaseIndex.lookup(BaseBlockKey{plan.domain, plan.matchedBlockHashes[index]});
        if (!current.has_value() || *current != plan.basePageBindings[index])
        {
            return AcquireResult{std::nullopt, AcquireStatus::kStalePlan};
        }
    }
    if (plan.draftRecord.has_value())
    {
        size_t const pairedBlockCount = plan.draftPageBindings.size();
        DraftPathMatch const expected{*plan.draftRecord, static_cast<int32_t>(pairedBlockCount)};
        DraftPathKey const key{*plan.draftSignature, plan.domain, plan.matchedBlockHashes.back()};
        if (!mDraftIndex.contains(key, expected) || !mRecords.contains(*plan.draftRecord))
        {
            return AcquireResult{std::nullopt, AcquireStatus::kStalePlan};
        }
        size_t requiredPathBlockCount = pairedBlockCount;
        if (plan.specReplayDependency.has_value())
        {
            SpecReplayDependency const& dependency = *plan.specReplayDependency;
            DraftPathMatch const dependencyMatch{*plan.draftRecord, dependency.pathBlockCount};
            DraftPathKey const dependencyKey{*plan.draftSignature, plan.domain, dependency.terminalHash};
            if (!mDraftIndex.contains(dependencyKey, dependencyMatch))
            {
                return AcquireResult{std::nullopt, AcquireStatus::kStalePlan};
            }
            requiredPathBlockCount = static_cast<size_t>(dependency.pathBlockCount);
        }
        CacheRecord const& record = mRecords.get(*plan.draftRecord);
        if (record.draftSignature != plan.draftSignature || record.draftPagePath.size() < requiredPathBlockCount
            || record.logicalBlockHashes.size() < requiredPathBlockCount
            || !std::equal(plan.draftPageBindings.begin(), plan.draftPageBindings.end(), record.draftPagePath.begin()))
        {
            return AcquireResult{std::nullopt, AcquireStatus::kStalePlan};
        }
        if (plan.specReplayDependency.has_value()
            && record.logicalBlockHashes[requiredPathBlockCount - 1] != plan.specReplayDependency->terminalHash)
        {
            return AcquireResult{std::nullopt, AcquireStatus::kStalePlan};
        }
    }
    if (plan.hybridRecord.has_value())
    {
        bool indexedCheckpoint = false;
        if (plan.hybridCheckpoint.has_value())
        {
            indexedCheckpoint = mRecords.findHybrid(*plan.hybridCheckpoint) == plan.hybridRecord;
        }
        else if (plan.hybridMtpCheckpoint.has_value())
        {
            indexedCheckpoint = mRecords.findHybridMtp(*plan.hybridMtpCheckpoint) == plan.hybridRecord;
        }
        if (!mRecords.contains(*plan.hybridRecord) || !indexedCheckpoint)
        {
            return AcquireResult{std::nullopt, AcquireStatus::kStalePlan};
        }
        CacheRecord const& record = mRecords.get(*plan.hybridRecord);
        if (record.recurrentSnapshotSlot != plan.recurrentSnapshotBinding
            || record.partialKvSnapshotSlot != plan.partialKvSnapshotBinding
            || record.basePagePath != plan.basePageBindings
            || (plan.hybridMtpCheckpoint.has_value()
                && (record.draftSignature != plan.draftSignature || record.draftPagePath != plan.draftPageBindings)))
        {
            return AcquireResult{std::nullopt, AcquireStatus::kStalePlan};
        }
    }

    CacheRequestLease lease;
    lease.mManager = this;
    lease.mMode = plan.mode;
    lease.mDomain = plan.domain;
    lease.mDraftSignature = plan.draftSignature;
    lease.mReuseTokenLength = plan.reuseTokenLength;
    lease.mMatchedBlockHashes = plan.matchedBlockHashes;
    lease.mBaseCowSources = plan.baseCowSources;
    lease.mDraftCowSources = plan.draftCowSources;
    lease.mSpecReplayDependency = plan.specReplayDependency;
    lease.mHybridCheckpoint = plan.hybridCheckpoint;
    lease.mHybridMtpCheckpoint = plan.hybridMtpCheckpoint;
    lease.mHybridRecord = plan.hybridRecord;
    lease.mRecurrentStateSchema = plan.recurrentStateSchema;
    lease.mHybridHasAttention = plan.hybridHasAttention;
    lease.mRecurrentSnapshotBinding = plan.recurrentSnapshotBinding;
    lease.mPartialKvSnapshotBinding = plan.partialKvSnapshotBinding;
    lease.mActiveResources.reserve(totalResourceCount(plan));
    size_t const sharedBasePageCount = plan.basePageBindings.size() - plan.baseCowSources.size();
    size_t const sharedDraftPageCount = plan.draftPageBindings.size() - plan.draftCowSources.size();
    lease.mBasePages.reserve(sharedBasePageCount + static_cast<size_t>(plan.demand.baseKvPages));
    lease.mDraftPages.reserve(sharedDraftPageCount + static_cast<size_t>(plan.demand.draftKvPages));

    for (size_t index = 0; index < plan.basePageBindings.size(); ++index)
    {
        PageId const page = plan.basePageBindings[index];
        ResourceId const resource{ResourceType::kBaseKvPage, page};
        mPools.addActiveRef(resource);
        lease.mActiveResources.push_back(resource);
        if (index < sharedBasePageCount)
        {
            lease.mBasePages.push_back(page);
        }
    }
    for (size_t index = 0; index < plan.draftPageBindings.size(); ++index)
    {
        PageId const page = plan.draftPageBindings[index];
        ResourceId const resource{ResourceType::kDraftKvPage, page};
        mPools.addActiveRef(resource);
        lease.mActiveResources.push_back(resource);
        if (index < sharedDraftPageCount)
        {
            lease.mDraftPages.push_back(page);
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

    std::optional<RecordId> const protectedRecord = plan.draftRecord.has_value() ? plan.draftRecord : plan.hybridRecord;
    EvictionPlan const eviction = EvictionPlanner::plan(plan.demand, mPools, mRecords, protectedRecord);
    if (!eviction.feasible)
    {
        return AcquireResult{std::nullopt, AcquireStatus::kInsufficientCapacity};
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
            lease.mDraftPages.push_back(resource.index);
        }
    }
    if (plan.draftRecord.has_value() && mRecords.contains(*plan.draftRecord))
    {
        mRecords.touch(*plan.draftRecord);
    }
    if (plan.hybridRecord.has_value() && mRecords.contains(*plan.hybridRecord))
    {
        mRecords.touch(*plan.hybridRecord);
    }

    return AcquireResult{std::optional<CacheRequestLease>{std::move(lease)}, AcquireStatus::kAcquired};
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
    ELLM_CHECK(lease.mMode == ReusePlanMode::kSpecEagle, "Speculative context cache growth requires an EAGLE lease");
    ELLM_CHECK(baseCount >= 0 && draftCount >= 0, "Context cache growth counts must be non-negative");
    return growPages(lease, ResourceDemand{baseCount, draftCount, 0, 0});
}

bool ContextCacheManager::growHybridMtpPages(CacheRequestLease& lease, int32_t baseCount, int32_t draftCount)
{
    ELLM_CHECK(lease.valid() && lease.mManager == this, "Context cache lease does not belong to this manager");
    ELLM_CHECK(lease.mMode == ReusePlanMode::kHybridMtp, "Hybrid+MTP context cache growth requires a combined lease");
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
    lease.mDraftPages.reserve(lease.mDraftPages.size() + static_cast<size_t>(demand.draftKvPages));
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
            lease.mDraftPages.push_back(resource.index);
        }
    }
    return true;
}

PublishStatus ContextCacheManager::publish(CacheRequestLease& lease, PublishRequest const& request)
{
    return publishDetailed(lease, request).status;
}

PublishResult ContextCacheManager::publishDetailed(CacheRequestLease& lease, PublishRequest const& request)
{
    ELLM_CHECK(lease.valid() && lease.mManager == this, "Context cache lease does not belong to this manager");
    ELLM_CHECK(lease.mMode != ReusePlanMode::kHybridMtp,
        "Hybrid+MTP context cache state must be published through one combined checkpoint transaction");
    if (request.point == PublicationPoint::kDecodeEnd && request.policy == CommitPolicy::kPrefillStateOnly)
    {
        return PublishResult{PublishStatus::kSkippedByPolicy, std::nullopt, {}, 0, true};
    }
    ELLM_CHECK(request.baseResidentStateLength >= 0, "Context cache base resident state length must be non-negative");

    bool const specPublication = lease.mMode == ReusePlanMode::kSpecEagle;
    ELLM_CHECK(specPublication == request.draftResidentStateLength.has_value(),
        "EAGLE context cache publication requires a separate draft resident state length");
    if (specPublication)
    {
        ELLM_CHECK(*request.draftResidentStateLength >= 0
                && *request.draftResidentStateLength <= request.baseResidentStateLength,
            "EAGLE context cache draft resident state must be a non-negative base-state prefix");
    }

    size_t const residentFullBlocks = static_cast<size_t>(request.baseResidentStateLength / mPageSize);
    size_t const requestedPublishCount = std::min(request.fullBlockHashes.size(), residentFullBlocks);
    ELLM_CHECK(requestedPublishCount > 0, "Context cache publication requires at least one resident full block");
    ELLM_CHECK(
        requestedPublishCount <= lease.mBasePages.size(), "Context cache publication exceeds the lease base page path");
    size_t requestedDraftPublishCount{};
    if (specPublication)
    {
        size_t const draftResidentFullBlocks = static_cast<size_t>(*request.draftResidentStateLength / mPageSize);
        requestedDraftPublishCount = std::min(request.fullBlockHashes.size(), draftResidentFullBlocks);
        ELLM_CHECK(lease.mDraftSignature.has_value() && requestedDraftPublishCount <= requestedPublishCount
                && requestedDraftPublishCount <= lease.mDraftPages.size(),
            "EAGLE context cache publication exceeds its accepted draft page path");
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
            "EAGLE context cache publication does not match its full-page replay dependency");
    }

    // A block whose logical hash already has a different canonical physical page may itself attach to that
    // canonical page, but descendants already computed from the private duplicate cannot extend the canonical
    // lineage. Stop at the first such block and let the runtime rebind before it computes any later descendants.
    size_t publishCount = requestedPublishCount;
    std::optional<size_t> firstPrivateDuplicate;
    size_t const cowBaseBegin = lease.mMatchedBlockHashes.size() - lease.mBaseCowSources.size();
    for (size_t index = 0; index < requestedPublishCount; ++index)
    {
        std::optional<PageId> const canonical
            = mBaseIndex.lookup(BaseBlockKey{lease.mDomain, request.fullBlockHashes[index]});
        if (canonical.has_value() && *canonical != lease.mBasePages[index])
        {
            bool const authorizedCow = index >= cowBaseBegin && index - cowBaseBegin < lease.mBaseCowSources.size()
                && lease.mBaseCowSources[index - cowBaseBegin] == *canonical;
            if (authorizedCow)
            {
                continue;
            }
            firstPrivateDuplicate = index;
            publishCount = index + 1;
            break;
        }
    }

    size_t pairedDraftPublishCount = std::min(requestedDraftPublishCount, publishCount);
    if (firstPrivateDuplicate.has_value())
    {
        // Draft state may be retained only through the physical base prefix it actually consumed. The duplicate
        // base block itself is canonicalized after execution, so its paired draft page is not eligible.
        pairedDraftPublishCount = std::min(pairedDraftPublishCount, *firstPrivateDuplicate);
    }

    std::vector<BlockHash> logicalHashes(
        request.fullBlockHashes.begin(), request.fullBlockHashes.begin() + static_cast<std::ptrdiff_t>(publishCount));
    std::vector<PageId> draftPages;
    if (pairedDraftPublishCount > 0)
    {
        draftPages.assign(lease.mDraftPages.begin(),
            lease.mDraftPages.begin() + static_cast<std::ptrdiff_t>(pairedDraftPublishCount));
    }
    CacheRecordKey const key{lease.mDomain, logicalHashes.back(), static_cast<int32_t>(publishCount)};
    std::optional<RecordId> const existing = mRecords.find(key);
    if (firstPrivateDuplicate.has_value() && !existing.has_value())
    {
        std::vector<PageId> canonicalPages;
        canonicalPages.reserve(publishCount);
        for (size_t index = 0; index < publishCount; ++index)
        {
            std::optional<PageId> const canonical
                = mBaseIndex.lookup(BaseBlockKey{lease.mDomain, logicalHashes[index]});
            ELLM_CHECK(
                canonical.has_value(), "Context cache canonical lineage is missing a page before a private duplicate");
            canonicalPages.push_back(*canonical);
        }
        return PublishResult{PublishStatus::kExistingRecord, std::nullopt, std::move(canonicalPages),
            static_cast<int32_t>(publishCount), false};
    }
    if (existing.has_value())
    {
        CacheRecord const& existingRecord = mRecords.get(*existing);
        bool const existingDraftIsAtLeastAsComplete = existingRecord.draftSignature == lease.mDraftSignature
            && existingRecord.pairedDraftFullBlockCount >= static_cast<int32_t>(pairedDraftPublishCount);
        if (!specPublication || pairedDraftPublishCount == 0 || existingDraftIsAtLeastAsComplete)
        {
            mRecords.touch(*existing);
            return PublishResult{PublishStatus::kExistingRecord, existing,
                std::vector<PageId>(existingRecord.basePagePath.begin(),
                    existingRecord.basePagePath.begin() + static_cast<std::ptrdiff_t>(publishCount)),
                static_cast<int32_t>(publishCount), !firstPrivateDuplicate.has_value()};
        }

        CacheRecord oldRecord = existingRecord;
        CacheRecord upgradedRecord = existingRecord;
        upgradedRecord.draftSignature = lease.mDraftSignature;
        upgradedRecord.draftPagePath = draftPages;
        upgradedRecord.pairedDraftFullBlockCount = static_cast<int32_t>(pairedDraftPublishCount);
        bool const sameSignatureExtension = existingRecord.draftSignature == lease.mDraftSignature;
        size_t draftCacheRefsAdded{};
        try
        {
            for (PageId const page : draftPages)
            {
                mPools.addCacheRef(ResourceId{ResourceType::kDraftKvPage, page});
                ++draftCacheRefsAdded;
            }
            if (sameSignatureExtension)
            {
                mDraftIndex.insertFrom(upgradedRecord, existingRecord.pairedDraftFullBlockCount + 1);
            }
            else
            {
                mDraftIndex.insert(upgradedRecord);
            }
        }
        catch (...)
        {
            releaseCacheRefsNoThrow(mPools, ResourceType::kDraftKvPage, draftPages, draftCacheRefsAdded);
            throw;
        }

        if (!sameSignatureExtension)
        {
            mDraftIndex.erase(oldRecord);
        }
        mRecords.setDraftState(
            *existing, *lease.mDraftSignature, std::move(draftPages), static_cast<int32_t>(pairedDraftPublishCount));
        for (PageId const page : oldRecord.draftPagePath)
        {
            mPools.releaseCacheRef(ResourceId{ResourceType::kDraftKvPage, page});
        }
        return PublishResult{PublishStatus::kPublished, existing,
            std::vector<PageId>(existingRecord.basePagePath.begin(),
                existingRecord.basePagePath.begin() + static_cast<std::ptrdiff_t>(publishCount)),
            static_cast<int32_t>(publishCount), !firstPrivateDuplicate.has_value()};
    }

    size_t const recordLimit = static_cast<size_t>(mRecords.maxRecords());
    ELLM_CHECK(mRecords.size() <= recordLimit, "Context cache record count already exceeds its configured limit");
    std::optional<RecordId> recordLimitVictim;
    if (recordLimit > 0 && mRecords.size() == recordLimit)
    {
        std::vector<RecordId> const lru = mRecords.lruToMru();
        ELLM_CHECK(!lru.empty(), "Context cache record limit enforcement found an empty LRU");
        recordLimitVictim = lru.front();
    }

    std::vector<PageId> canonicalPages;
    canonicalPages.reserve(publishCount);
    std::vector<MissingBaseMapping> missingMappings;
    missingMappings.reserve(publishCount);
    for (size_t index = 0; index < publishCount; ++index)
    {
        BaseBlockKey const blockKey{lease.mDomain, logicalHashes[index]};
        std::optional<PageId> const indexedPage = mBaseIndex.lookup(blockKey);
        if (indexedPage.has_value())
        {
            canonicalPages.push_back(*indexedPage);
            continue;
        }

        auto const proposed = std::find_if(missingMappings.begin(), missingMappings.end(),
            [&](MissingBaseMapping const& missing) { return missing.key == blockKey; });
        if (proposed != missingMappings.end())
        {
            canonicalPages.push_back(proposed->page);
            continue;
        }

        PageId const proposedPage = lease.mBasePages[index];
        missingMappings.push_back(MissingBaseMapping{blockKey, proposedPage});
        canonicalPages.push_back(proposedPage);
    }

    CacheRecord record;
    record.key = key;
    record.logicalBlockHashes = std::move(logicalHashes);
    record.basePagePath = canonicalPages;
    record.baseFullBlockCount = static_cast<int32_t>(publishCount);
    if (pairedDraftPublishCount > 0)
    {
        record.draftSignature = lease.mDraftSignature;
        record.draftPagePath = draftPages;
        record.pairedDraftFullBlockCount = static_cast<int32_t>(pairedDraftPublishCount);
    }

    std::vector<PageId> insertedPages;
    insertedPages.reserve(missingMappings.size());
    size_t baseCacheRefsAdded{};
    size_t draftCacheRefsAdded{};
    RecordId insertedRecord{};
    // Pool ownership, canonical block mappings, and the endpoint record become visible as one transaction. The catch
    // reverses every earlier step if a later container allocation or invariant check fails.
    try
    {
        for (PageId const page : canonicalPages)
        {
            mPools.addCacheRef(ResourceId{ResourceType::kBaseKvPage, page});
            ++baseCacheRefsAdded;
        }
        for (PageId const page : draftPages)
        {
            mPools.addCacheRef(ResourceId{ResourceType::kDraftKvPage, page});
            ++draftCacheRefsAdded;
        }
        for (MissingBaseMapping const& missing : missingMappings)
        {
            BaseInsertResult const inserted = mBaseIndex.insert(missing.key, missing.page);
            ELLM_CHECK(inserted.inserted && inserted.canonicalPage == missing.page,
                "Context cache base index changed during publication");
            insertedPages.push_back(missing.page);
        }

        RecordInsertResult const inserted = mRecords.insert(std::move(record));
        ELLM_CHECK(inserted.inserted, "Context cache exact record appeared during publication");
        insertedRecord = inserted.id;
        if (pairedDraftPublishCount > 0)
        {
            mDraftIndex.insert(mRecords.get(insertedRecord));
        }
    }
    catch (...)
    {
        if (insertedRecord != 0 && mRecords.contains(insertedRecord))
        {
            (void) mRecords.erase(insertedRecord);
        }
        for (PageId const page : insertedPages)
        {
            mBaseIndex.erasePage(page);
        }
        releaseCacheRefsNoThrow(mPools, ResourceType::kDraftKvPage, draftPages, draftCacheRefsAdded);
        releaseCacheRefsNoThrow(mPools, ResourceType::kBaseKvPage, canonicalPages, baseCacheRefsAdded);
        throw;
    }

    if (recordLimit == 0)
    {
        evictRecord(insertedRecord);
    }
    else if (recordLimitVictim.has_value())
    {
        evictRecord(*recordLimitVictim);
    }
    enforceRecordLimit();
    std::optional<RecordId> retainedRecord;
    if (mRecords.contains(insertedRecord))
    {
        retainedRecord = insertedRecord;
    }
    return PublishResult{PublishStatus::kPublished, retainedRecord, std::move(canonicalPages),
        static_cast<int32_t>(publishCount), !firstPrivateDuplicate.has_value()};
}

PublishResult ContextCacheManager::publishHybridDetailed(CacheRequestLease& lease, HybridPublishRequest const& request)
{
    ELLM_CHECK(lease.valid() && lease.mManager == this, "Context cache lease does not belong to this manager");
    ELLM_CHECK(lease.mMode == ReusePlanMode::kHybrid && lease.mRecurrentStateSchema.has_value(),
        "Hybrid context cache publication requires an exact-checkpoint lease");
    if (request.point == PublicationPoint::kDecodeEnd && request.policy == CommitPolicy::kPrefillStateOnly)
    {
        return PublishResult{PublishStatus::kSkippedByPolicy, std::nullopt, {}, 0, true};
    }

    ELLM_CHECK(request.checkpoint.domain == lease.mDomain && request.checkpoint.schema == *lease.mRecurrentStateSchema
            && request.checkpoint.exactLength > 0,
        "Hybrid context cache publication identity does not match its lease");
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

    std::vector<PageId> canonicalPages;
    canonicalPages.reserve(fullBlockCount);
    bool lineageComplete = true;
    if (lease.mHybridHasAttention)
    {
        for (size_t index = 0; index < fullBlockCount; ++index)
        {
            std::optional<PageId> const canonical
                = mBaseIndex.lookup(BaseBlockKey{lease.mDomain, request.fullBlockHashes[index]});
            auto const repeated = std::find(request.fullBlockHashes.begin(),
                request.fullBlockHashes.begin() + static_cast<std::ptrdiff_t>(index), request.fullBlockHashes[index]);
            if (canonical.has_value())
            {
                canonicalPages.push_back(*canonical);
            }
            else if (repeated != request.fullBlockHashes.begin() + static_cast<std::ptrdiff_t>(index))
            {
                size_t const repeatedIndex
                    = static_cast<size_t>(std::distance(request.fullBlockHashes.begin(), repeated));
                canonicalPages.push_back(canonicalPages[repeatedIndex]);
            }
            else
            {
                canonicalPages.push_back(lease.mBasePages[index]);
            }
            if (canonicalPages.back() != lease.mBasePages[index])
            {
                lineageComplete = false;
                break;
            }
        }
    }
    if (!lineageComplete)
    {
        int32_t const canonicalBlockCount = static_cast<int32_t>(canonicalPages.size());
        return PublishResult{
            PublishStatus::kExistingRecord, std::nullopt, std::move(canonicalPages), canonicalBlockCount, false};
    }

    std::optional<RecordId> const existing = mRecords.findHybrid(request.checkpoint);
    if (existing.has_value())
    {
        CacheRecord const& record = mRecords.get(*existing);
        mRecords.touch(*existing);
        return PublishResult{
            PublishStatus::kExistingRecord, existing, record.basePagePath, static_cast<int32_t>(fullBlockCount), true};
    }

    size_t const recordLimit = static_cast<size_t>(mRecords.maxRecords());
    ELLM_CHECK(mRecords.size() <= recordLimit, "Context cache record count already exceeds its configured limit");
    std::optional<RecordId> recordLimitVictim;
    if (recordLimit > 0 && mRecords.size() == recordLimit)
    {
        std::vector<RecordId> const lru = mRecords.lruToMru();
        ELLM_CHECK(!lru.empty(), "Context cache record limit enforcement found an empty LRU");
        recordLimitVictim = lru.front();
    }

    std::vector<MissingBaseMapping> missingMappings;
    if (lease.mHybridHasAttention)
    {
        missingMappings.reserve(fullBlockCount);
        for (size_t index = 0; index < fullBlockCount; ++index)
        {
            BaseBlockKey const key{lease.mDomain, request.fullBlockHashes[index]};
            auto const proposed = std::find_if(missingMappings.begin(), missingMappings.end(),
                [&](MissingBaseMapping const& missing) { return missing.key == key; });
            if (!mBaseIndex.lookup(key).has_value() && proposed == missingMappings.end())
            {
                missingMappings.push_back(MissingBaseMapping{key, canonicalPages[index]});
            }
        }
    }

    CacheRecord record;
    record.key = CacheRecordKey{lease.mDomain, request.checkpoint.exactPrefixDigest,
        static_cast<int32_t>(fullBlockCount), CacheRecordKind::kHybrid};
    record.logicalBlockHashes = request.fullBlockHashes;
    record.basePagePath = canonicalPages;
    record.baseFullBlockCount = lease.mHybridHasAttention ? static_cast<int32_t>(fullBlockCount) : 0;
    record.recurrentSnapshotSlot = request.snapshots.recurrentSnapshotSlot;
    record.partialKvSnapshotSlot = request.snapshots.partialKvSnapshotSlot;
    record.exactCheckpointLength = request.checkpoint.exactLength;
    record.exactCheckpointDigest = request.checkpoint.exactPrefixDigest;
    record.recurrentStateSchema = request.checkpoint.schema;

    std::vector<PageId> insertedPages;
    insertedPages.reserve(missingMappings.size());
    size_t baseCacheRefsAdded{};
    bool recurrentCacheRefAdded = false;
    bool partialCacheRefAdded = false;
    RecordId insertedRecord{};
    try
    {
        for (PageId const page : canonicalPages)
        {
            mPools.addCacheRef(ResourceId{ResourceType::kBaseKvPage, page});
            ++baseCacheRefsAdded;
        }
        mPools.addCacheRef(ResourceId{ResourceType::kRecurrentSnapshot, request.snapshots.recurrentSnapshotSlot});
        recurrentCacheRefAdded = true;
        if (request.snapshots.partialKvSnapshotSlot.has_value())
        {
            mPools.addCacheRef(ResourceId{ResourceType::kPartialKvSnapshot, *request.snapshots.partialKvSnapshotSlot});
            partialCacheRefAdded = true;
        }
        for (MissingBaseMapping const& missing : missingMappings)
        {
            BaseInsertResult const inserted = mBaseIndex.insert(missing.key, missing.page);
            ELLM_CHECK(inserted.inserted && inserted.canonicalPage == missing.page,
                "Context cache base index changed during hybrid publication");
            insertedPages.push_back(missing.page);
        }
        RecordInsertResult const inserted = mRecords.insert(std::move(record));
        ELLM_CHECK(inserted.inserted, "Context cache hybrid checkpoint appeared during publication");
        insertedRecord = inserted.id;
    }
    catch (...)
    {
        if (insertedRecord != 0 && mRecords.contains(insertedRecord))
        {
            (void) mRecords.erase(insertedRecord);
        }
        for (PageId const page : insertedPages)
        {
            mBaseIndex.erasePage(page);
        }
        if (partialCacheRefAdded)
        {
            mPools.releaseCacheRef(
                ResourceId{ResourceType::kPartialKvSnapshot, *request.snapshots.partialKvSnapshotSlot});
        }
        if (recurrentCacheRefAdded)
        {
            mPools.releaseCacheRef(
                ResourceId{ResourceType::kRecurrentSnapshot, request.snapshots.recurrentSnapshotSlot});
        }
        releaseCacheRefsNoThrow(mPools, ResourceType::kBaseKvPage, canonicalPages, baseCacheRefsAdded);
        throw;
    }

    if (recordLimit == 0)
    {
        evictRecord(insertedRecord);
    }
    else if (recordLimitVictim.has_value())
    {
        evictRecord(*recordLimitVictim);
    }
    enforceRecordLimit();
    std::optional<RecordId> retainedRecord;
    if (mRecords.contains(insertedRecord))
    {
        retainedRecord = insertedRecord;
    }
    return PublishResult{PublishStatus::kPublished, retainedRecord, std::move(canonicalPages),
        static_cast<int32_t>(fullBlockCount), true};
}

PublishResult ContextCacheManager::publishHybridMtpDetailed(
    CacheRequestLease& lease, HybridMtpPublishRequest const& request)
{
    ELLM_CHECK(lease.valid() && lease.mManager == this, "Context cache lease does not belong to this manager");
    ELLM_CHECK(lease.mMode == ReusePlanMode::kHybridMtp && lease.mRecurrentStateSchema.has_value()
            && lease.mDraftSignature.has_value(),
        "Hybrid+MTP context cache publication requires a combined lease");
    if (request.point == PublicationPoint::kDecodeEnd && request.policy == CommitPolicy::kPrefillStateOnly)
    {
        return PublishResult{PublishStatus::kSkippedByPolicy, std::nullopt, {}, 0, true};
    }
    ELLM_CHECK(request.point == PublicationPoint::kPrefillEnd,
        "Hybrid+MTP context cache publication supports only prefill endpoints");

    ELLM_CHECK(request.checkpoint.domain == lease.mDomain && request.checkpoint.schema == *lease.mRecurrentStateSchema
            && request.checkpoint.draftSignature == *lease.mDraftSignature && request.checkpoint.exactLength > 0,
        "Hybrid+MTP context cache publication identity does not match its lease");
    // The boundary token (exactLength - 1) is always retained in a private partial page so a consumer can rewrite its
    // draft KV without mutating a shared reused page; reserve one fewer full block than exactLength/pageSize would.
    size_t const fullBlockCount = static_cast<size_t>((request.checkpoint.exactLength - 1) / mPageSize);
    ELLM_CHECK(request.fullBlockHashes.size() == fullBlockCount,
        "Hybrid+MTP publication requires every complete logical block before its exact boundary");
    bool const needsPartialSnapshot = true;
    ELLM_CHECK(request.snapshots.recurrentSnapshotSlot >= 0
            && request.snapshots.partialKvSnapshotSlot.has_value() == needsPartialSnapshot,
        "Hybrid+MTP publication has an incomplete snapshot set");
    ELLM_CHECK(fullBlockCount <= lease.mBasePages.size() && fullBlockCount <= lease.mDraftPages.size(),
        "Hybrid+MTP publication exceeds its paired page paths");

    auto ownsActiveResource = [&](ResourceId const& resource) {
        return std::find(lease.mActiveResources.begin(), lease.mActiveResources.end(), resource)
            != lease.mActiveResources.end();
    };
    ELLM_CHECK(
        ownsActiveResource(ResourceId{ResourceType::kRecurrentSnapshot, request.snapshots.recurrentSnapshotSlot}),
        "Hybrid+MTP lease does not own the recurrent snapshot reservation");
    if (needsPartialSnapshot)
    {
        ELLM_CHECK(
            ownsActiveResource(ResourceId{ResourceType::kPartialKvSnapshot, *request.snapshots.partialKvSnapshotSlot}),
            "Hybrid+MTP lease does not own the bundled partial snapshot reservation");
    }

    size_t const matchedBlockCount = std::min(fullBlockCount, lease.mMatchedBlockHashes.size());
    ELLM_CHECK(std::equal(request.fullBlockHashes.begin(),
                   request.fullBlockHashes.begin() + static_cast<std::ptrdiff_t>(matchedBlockCount),
                   lease.mMatchedBlockHashes.begin()),
        "Hybrid+MTP publication does not match the acquired logical prefix");

    std::vector<PageId> canonicalPages;
    canonicalPages.reserve(fullBlockCount);
    bool lineageComplete = true;
    for (size_t index = 0; index < fullBlockCount; ++index)
    {
        std::optional<PageId> const canonical
            = mBaseIndex.lookup(BaseBlockKey{lease.mDomain, request.fullBlockHashes[index]});
        auto const repeated = std::find(request.fullBlockHashes.begin(),
            request.fullBlockHashes.begin() + static_cast<std::ptrdiff_t>(index), request.fullBlockHashes[index]);
        if (canonical.has_value())
        {
            canonicalPages.push_back(*canonical);
        }
        else if (repeated != request.fullBlockHashes.begin() + static_cast<std::ptrdiff_t>(index))
        {
            size_t const repeatedIndex = static_cast<size_t>(std::distance(request.fullBlockHashes.begin(), repeated));
            canonicalPages.push_back(canonicalPages[repeatedIndex]);
        }
        else
        {
            canonicalPages.push_back(lease.mBasePages[index]);
        }
        if (canonicalPages.back() != lease.mBasePages[index])
        {
            lineageComplete = false;
            break;
        }
    }
    if (!lineageComplete)
    {
        int32_t const canonicalBlockCount = static_cast<int32_t>(canonicalPages.size());
        return PublishResult{
            PublishStatus::kExistingRecord, std::nullopt, std::move(canonicalPages), canonicalBlockCount, false};
    }

    std::optional<RecordId> const existing = mRecords.findHybridMtp(request.checkpoint);
    if (existing.has_value())
    {
        CacheRecord const& record = mRecords.get(*existing);
        mRecords.touch(*existing);
        return PublishResult{
            PublishStatus::kExistingRecord, existing, record.basePagePath, static_cast<int32_t>(fullBlockCount), true};
    }

    size_t const recordLimit = static_cast<size_t>(mRecords.maxRecords());
    ELLM_CHECK(mRecords.size() <= recordLimit, "Context cache record count already exceeds its configured limit");
    std::optional<RecordId> recordLimitVictim;
    if (recordLimit > 0 && mRecords.size() == recordLimit)
    {
        std::vector<RecordId> const lru = mRecords.lruToMru();
        ELLM_CHECK(!lru.empty(), "Context cache record limit enforcement found an empty LRU");
        recordLimitVictim = lru.front();
    }

    std::vector<MissingBaseMapping> missingMappings;
    missingMappings.reserve(fullBlockCount);
    for (size_t index = 0; index < fullBlockCount; ++index)
    {
        BaseBlockKey const key{lease.mDomain, request.fullBlockHashes[index]};
        auto const proposed = std::find_if(missingMappings.begin(), missingMappings.end(),
            [&](MissingBaseMapping const& missing) { return missing.key == key; });
        if (!mBaseIndex.lookup(key).has_value() && proposed == missingMappings.end())
        {
            missingMappings.push_back(MissingBaseMapping{key, canonicalPages[index]});
        }
    }

    std::vector<PageId> const draftPages(
        lease.mDraftPages.begin(), lease.mDraftPages.begin() + static_cast<std::ptrdiff_t>(fullBlockCount));
    CacheRecord record;
    record.key = CacheRecordKey{lease.mDomain, request.checkpoint.exactPrefixDigest,
        static_cast<int32_t>(fullBlockCount), CacheRecordKind::kHybridMtp};
    record.logicalBlockHashes = request.fullBlockHashes;
    record.basePagePath = canonicalPages;
    record.draftSignature = request.checkpoint.draftSignature;
    record.draftPagePath = draftPages;
    record.baseFullBlockCount = static_cast<int32_t>(fullBlockCount);
    record.pairedDraftFullBlockCount = static_cast<int32_t>(fullBlockCount);
    record.recurrentSnapshotSlot = request.snapshots.recurrentSnapshotSlot;
    record.partialKvSnapshotSlot = request.snapshots.partialKvSnapshotSlot;
    record.exactCheckpointLength = request.checkpoint.exactLength;
    record.exactCheckpointDigest = request.checkpoint.exactPrefixDigest;
    record.recurrentStateSchema = request.checkpoint.schema;

    std::vector<PageId> insertedPages;
    insertedPages.reserve(missingMappings.size());
    size_t baseCacheRefsAdded{};
    size_t draftCacheRefsAdded{};
    bool recurrentCacheRefAdded = false;
    bool partialCacheRefAdded = false;
    RecordId insertedRecord{};
    try
    {
        for (PageId const page : canonicalPages)
        {
            mPools.addCacheRef(ResourceId{ResourceType::kBaseKvPage, page});
            ++baseCacheRefsAdded;
        }
        for (PageId const page : draftPages)
        {
            mPools.addCacheRef(ResourceId{ResourceType::kDraftKvPage, page});
            ++draftCacheRefsAdded;
        }
        mPools.addCacheRef(ResourceId{ResourceType::kRecurrentSnapshot, request.snapshots.recurrentSnapshotSlot});
        recurrentCacheRefAdded = true;
        if (request.snapshots.partialKvSnapshotSlot.has_value())
        {
            mPools.addCacheRef(ResourceId{ResourceType::kPartialKvSnapshot, *request.snapshots.partialKvSnapshotSlot});
            partialCacheRefAdded = true;
        }
        for (MissingBaseMapping const& missing : missingMappings)
        {
            BaseInsertResult const inserted = mBaseIndex.insert(missing.key, missing.page);
            ELLM_CHECK(inserted.inserted && inserted.canonicalPage == missing.page,
                "Context cache base index changed during Hybrid+MTP publication");
            insertedPages.push_back(missing.page);
        }
        RecordInsertResult const inserted = mRecords.insert(std::move(record));
        ELLM_CHECK(inserted.inserted, "Context cache Hybrid+MTP checkpoint appeared during publication");
        insertedRecord = inserted.id;
    }
    catch (...)
    {
        if (insertedRecord != 0 && mRecords.contains(insertedRecord))
        {
            (void) mRecords.erase(insertedRecord);
        }
        for (PageId const page : insertedPages)
        {
            mBaseIndex.erasePage(page);
        }
        if (partialCacheRefAdded)
        {
            mPools.releaseCacheRef(
                ResourceId{ResourceType::kPartialKvSnapshot, *request.snapshots.partialKvSnapshotSlot});
        }
        if (recurrentCacheRefAdded)
        {
            mPools.releaseCacheRef(
                ResourceId{ResourceType::kRecurrentSnapshot, request.snapshots.recurrentSnapshotSlot});
        }
        releaseCacheRefsNoThrow(mPools, ResourceType::kDraftKvPage, draftPages, draftCacheRefsAdded);
        releaseCacheRefsNoThrow(mPools, ResourceType::kBaseKvPage, canonicalPages, baseCacheRefsAdded);
        throw;
    }

    if (recordLimit == 0)
    {
        evictRecord(insertedRecord);
    }
    else if (recordLimitVictim.has_value())
    {
        evictRecord(*recordLimitVictim);
    }
    enforceRecordLimit();
    std::optional<RecordId> retainedRecord;
    if (mRecords.contains(insertedRecord))
    {
        retainedRecord = insertedRecord;
    }
    return PublishResult{PublishStatus::kPublished, retainedRecord, std::move(canonicalPages),
        static_cast<int32_t>(fullBlockCount), true};
}

void ContextCacheManager::rebindBasePrefix(CacheRequestLease& lease, std::vector<PageId> const& canonicalPages)
{
    ELLM_CHECK(lease.valid() && lease.mManager == this, "Context cache lease does not belong to this manager");
    ELLM_CHECK(canonicalPages.size() <= lease.mBasePages.size(),
        "Canonical context cache prefix exceeds the lease base page path");

    for (size_t index = 0; index < canonicalPages.size(); ++index)
    {
        PageId const current = lease.mBasePages[index];
        PageId const canonical = canonicalPages[index];
        if (current == canonical)
        {
            continue;
        }

        auto const activeResource = std::find(lease.mActiveResources.begin(), lease.mActiveResources.end(),
            ResourceId{ResourceType::kBaseKvPage, current});
        ELLM_CHECK(activeResource != lease.mActiveResources.end(),
            "Context cache lease is missing the active reference for a rebound base page");

        ResourceId const canonicalResource{ResourceType::kBaseKvPage, canonical};
        mPools.addActiveRef(canonicalResource);
        *activeResource = canonicalResource;
        lease.mBasePages[index] = canonical;
        mPools.releaseActiveRef(ResourceId{ResourceType::kBaseKvPage, current});
    }
}

ResourcePools const& ContextCacheManager::pools() const noexcept
{
    return mPools;
}

BaseBlockIndex const& ContextCacheManager::baseIndex() const noexcept
{
    return mBaseIndex;
}

DraftPathIndex const& ContextCacheManager::draftIndex() const noexcept
{
    return mDraftIndex;
}

CacheRecordStore const& ContextCacheManager::records() const noexcept
{
    return mRecords;
}

void ContextCacheManager::releaseLease(CacheRequestLease& lease) noexcept
{
    lease.mManager = nullptr;
    releaseActiveRefsNoThrow(mPools, lease.mActiveResources);
    lease.mMode = ReusePlanMode::kVanilla;
    lease.mDomain = {};
    lease.mDraftSignature.reset();
    lease.mReuseTokenLength = 0;
    lease.mMatchedBlockHashes.clear();
    lease.mActiveResources.clear();
    lease.mBasePages.clear();
    lease.mDraftPages.clear();
    lease.mBaseCowSources.clear();
    lease.mDraftCowSources.clear();
    lease.mSpecReplayDependency.reset();
    lease.mHybridCheckpoint.reset();
    lease.mHybridMtpCheckpoint.reset();
    lease.mHybridRecord.reset();
    lease.mRecurrentStateSchema.reset();
    lease.mHybridHasAttention = false;
    lease.mRecurrentSnapshotBinding.reset();
    lease.mPartialKvSnapshotBinding.reset();
}

void ContextCacheManager::evictRecord(RecordId id)
{
    CacheRecord const record = mRecords.erase(id);
    if (record.key.kind != CacheRecordKind::kHybridMtp)
    {
        mDraftIndex.erase(record);
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
    for (PageId const page : record.draftPagePath)
    {
        mPools.releaseCacheRef(ResourceId{ResourceType::kDraftKvPage, page});
    }
    if (record.recurrentSnapshotSlot.has_value())
    {
        mPools.releaseCacheRef(ResourceId{ResourceType::kRecurrentSnapshot, *record.recurrentSnapshotSlot});
    }
    if (record.partialKvSnapshotSlot.has_value())
    {
        mPools.releaseCacheRef(ResourceId{ResourceType::kPartialKvSnapshot, *record.partialKvSnapshotSlot});
    }
}

void ContextCacheManager::applyEviction(EvictionPlan const& plan)
{
    for (RecordId const id : plan.victims)
    {
        evictRecord(id);
    }
}

void ContextCacheManager::enforceRecordLimit()
{
    while (mRecords.size() > static_cast<size_t>(mRecords.maxRecords()))
    {
        std::vector<RecordId> const lru = mRecords.lruToMru();
        ELLM_CHECK(!lru.empty(), "Context cache record limit enforcement found an empty LRU");
        evictRecord(lru.front());
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
