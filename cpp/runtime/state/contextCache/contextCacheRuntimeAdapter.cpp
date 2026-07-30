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

#include "runtime/state/contextCache/contextCacheRuntimeAdapter.h"

#include "common/checkMacros.h"
#include "runtime/state/kvPageTable.h"

#include <algorithm>
#include <limits>
#include <utility>

namespace trt_edgellm
{
namespace rt
{

ContextCacheRuntimeAdapter::ContextCacheRuntimeAdapter(int32_t pageSize, ResourceDemand capacities, int32_t maxRecords)
    : mPageSize(pageSize)
    , mManager(pageSize, capacities, maxRecords)
{
}

RuntimeCacheAcquireResult ContextCacheRuntimeAdapter::acquireSpec(SpecDecodeMode mode, CacheDomainId domain,
    DraftEngineSignature draftSignature, std::vector<BlockHash> const& fullBlockHashes, int32_t inputTokenCount,
    LookupPolicy lookupPolicy)
{
    LookupPolicy policy = lookupPolicy;
    bool forcedCold = false;
    constexpr int32_t kMAX_STALE_RETRIES = 2;
    for (int32_t attempt = 0; attempt <= kMAX_STALE_RETRIES; ++attempt)
    {
        ReusePlan plan = mManager.planSpec(mode, domain, draftSignature, fullBlockHashes, inputTokenCount,
            /*supportsOneTokenReplay=*/false, policy);
        bool const cacheDerivedPlan = plan.reuseTokenLength > 0 || plan.kind == ReusePlanKind::kFullInputRewind;
        ReusePlanKind const planKind = plan.kind;
        AcquireResult acquired = mManager.acquire(plan);
        if (acquired.status == AcquireStatus::kStalePlan)
        {
            continue;
        }
        if (acquired.status == AcquireStatus::kInsufficientCapacity && cacheDerivedPlan
            && policy == LookupPolicy::kUseCache)
        {
            policy = LookupPolicy::kBypass;
            forcedCold = true;
            continue;
        }
        return RuntimeCacheAcquireResult{std::move(acquired.lease), acquired.status, planKind, forcedCold};
    }
    return RuntimeCacheAcquireResult{
        std::nullopt, AcquireStatus::kStalePlan, ReusePlanKind::kNoReusablePrefix, forcedCold};
}

RuntimeCacheAcquireResult ContextCacheRuntimeAdapter::acquireHybrid(CacheDomainId domain,
    std::vector<int32_t> const& inputTokens, std::vector<BlockKeyExtras> const& touchedBlockExtras,
    RecurrentStateSchemaId schema, bool hasAttention, LookupPolicy lookupPolicy)
{
    ELLM_CHECK(inputTokens.size() <= static_cast<size_t>(std::numeric_limits<int32_t>::max()),
        "Context cache hybrid input contains too many tokens");
    int32_t const inputTokenCount = static_cast<int32_t>(inputTokens.size());
    size_t const fullBlockCount = inputTokens.size() / static_cast<size_t>(mPageSize);
    size_t const touchedBlockCount
        = (inputTokens.size() + static_cast<size_t>(mPageSize) - 1U) / static_cast<size_t>(mPageSize);
    ELLM_CHECK(touchedBlockExtras.empty() || touchedBlockExtras.size() == touchedBlockCount,
        "Context cache hybrid extras must describe every touched input block");
    std::vector<BlockKeyExtras> const fullBlockExtras = touchedBlockExtras.empty()
        ? std::vector<BlockKeyExtras>{}
        : std::vector<BlockKeyExtras>(
              touchedBlockExtras.begin(), touchedBlockExtras.begin() + static_cast<std::ptrdiff_t>(fullBlockCount));
    std::vector<BlockHash> const fullBlockHashes
        = hashFullBlocks(inputTokens.data(), inputTokens.size(), mPageSize, fullBlockExtras);

    std::vector<int32_t> const candidateLengths = mManager.hybridCandidateLengths(domain, schema, inputTokenCount);
    std::vector<HybridCheckpointCandidate> candidates;
    candidates.reserve(candidateLengths.size());
    for (int32_t const length : candidateLengths)
    {
        size_t const candidateTouchedBlockCount
            = (static_cast<size_t>(length) + static_cast<size_t>(mPageSize) - 1U) / static_cast<size_t>(mPageSize);
        std::vector<BlockKeyExtras> const exactExtras = touchedBlockExtras.empty()
            ? std::vector<BlockKeyExtras>{}
            : std::vector<BlockKeyExtras>(touchedBlockExtras.begin(),
                  touchedBlockExtras.begin() + static_cast<std::ptrdiff_t>(candidateTouchedBlockCount));
        candidates.push_back(HybridCheckpointCandidate{
            length, hashExactPrefix(inputTokens.data(), static_cast<size_t>(length), mPageSize, exactExtras)});
    }

    return acquireHybrid(domain, candidates, fullBlockHashes, inputTokenCount, schema, hasAttention, lookupPolicy);
}

RuntimeCacheAcquireResult ContextCacheRuntimeAdapter::acquireHybrid(CacheDomainId domain,
    std::vector<HybridCheckpointCandidate> const& candidates, std::vector<BlockHash> const& fullBlockHashes,
    int32_t inputTokenCount, RecurrentStateSchemaId schema, bool hasAttention, LookupPolicy lookupPolicy)
{
    ELLM_CHECK(inputTokenCount >= 0, "Context cache hybrid input length must be non-negative");
    LookupPolicy policy = lookupPolicy;
    bool forcedCold = false;
    constexpr int32_t kMAX_STALE_RETRIES = 2;
    for (int32_t attempt = 0; attempt <= kMAX_STALE_RETRIES; ++attempt)
    {
        ReusePlan plan
            = mManager.planHybrid(domain, schema, candidates, fullBlockHashes, inputTokenCount, hasAttention, policy);
        bool const cacheDerivedPlan = plan.reuseTokenLength > 0;
        ReusePlanKind const planKind = plan.kind;
        AcquireResult acquired = mManager.acquire(plan);
        if (acquired.status == AcquireStatus::kStalePlan)
        {
            continue;
        }
        if (acquired.status == AcquireStatus::kInsufficientCapacity && cacheDerivedPlan
            && policy == LookupPolicy::kUseCache)
        {
            policy = LookupPolicy::kBypass;
            forcedCold = true;
            continue;
        }
        return RuntimeCacheAcquireResult{std::move(acquired.lease), acquired.status, planKind, forcedCold};
    }
    return RuntimeCacheAcquireResult{
        std::nullopt, AcquireStatus::kStalePlan, ReusePlanKind::kNoReusablePrefix, forcedCold};
}

RuntimeCacheAcquireResult ContextCacheRuntimeAdapter::acquireHybridMtp(CacheDomainId domain,
    std::vector<int32_t> const& inputTokens, std::vector<BlockKeyExtras> const& touchedBlockExtras,
    RecurrentStateSchemaId schema, DraftEngineSignature draftSignature, LookupPolicy lookupPolicy)
{
    ELLM_CHECK(inputTokens.size() <= static_cast<size_t>(std::numeric_limits<int32_t>::max()),
        "Hybrid+MTP context cache input contains too many tokens");
    int32_t const inputTokenCount = static_cast<int32_t>(inputTokens.size());
    size_t const fullBlockCount = inputTokens.size() / static_cast<size_t>(mPageSize);
    size_t const touchedBlockCount
        = (inputTokens.size() + static_cast<size_t>(mPageSize) - 1U) / static_cast<size_t>(mPageSize);
    ELLM_CHECK(touchedBlockExtras.empty() || touchedBlockExtras.size() == touchedBlockCount,
        "Hybrid+MTP context cache extras must describe every touched input block");
    std::vector<BlockKeyExtras> const fullBlockExtras = touchedBlockExtras.empty()
        ? std::vector<BlockKeyExtras>{}
        : std::vector<BlockKeyExtras>(
              touchedBlockExtras.begin(), touchedBlockExtras.begin() + static_cast<std::ptrdiff_t>(fullBlockCount));
    std::vector<BlockHash> const fullBlockHashes
        = hashFullBlocks(inputTokens.data(), inputTokens.size(), mPageSize, fullBlockExtras);

    std::vector<int32_t> const candidateLengths
        = mManager.hybridMtpCandidateLengths(domain, schema, draftSignature, inputTokenCount);
    std::vector<HybridMtpCheckpointCandidate> candidates;
    candidates.reserve(candidateLengths.size());
    for (int32_t const length : candidateLengths)
    {
        ELLM_CHECK(
            length >= 0 && length < inputTokenCount, "Hybrid+MTP context cache candidate is not an interior endpoint");
        size_t const candidateTouchedBlockCount
            = (static_cast<size_t>(length) + static_cast<size_t>(mPageSize) - 1U) / static_cast<size_t>(mPageSize);
        std::vector<BlockKeyExtras> const exactExtras = touchedBlockExtras.empty()
            ? std::vector<BlockKeyExtras>{}
            : std::vector<BlockKeyExtras>(touchedBlockExtras.begin(),
                  touchedBlockExtras.begin() + static_cast<std::ptrdiff_t>(candidateTouchedBlockCount));
        candidates.push_back(HybridMtpCheckpointCandidate{
            length, hashExactPrefix(inputTokens.data(), static_cast<size_t>(length), mPageSize, exactExtras)});
    }

    return acquireHybridMtp(domain, candidates, fullBlockHashes, inputTokenCount, schema, draftSignature, lookupPolicy);
}

RuntimeCacheAcquireResult ContextCacheRuntimeAdapter::acquireHybridMtp(CacheDomainId domain,
    std::vector<HybridMtpCheckpointCandidate> const& candidates, std::vector<BlockHash> const& fullBlockHashes,
    int32_t inputTokenCount, RecurrentStateSchemaId schema, DraftEngineSignature draftSignature,
    LookupPolicy lookupPolicy)
{
    ELLM_CHECK(inputTokenCount >= 0, "Hybrid+MTP context cache input length must be non-negative");
    LookupPolicy policy = lookupPolicy;
    bool forcedCold = false;
    constexpr int32_t kMAX_STALE_RETRIES = 2;
    for (int32_t attempt = 0; attempt <= kMAX_STALE_RETRIES; ++attempt)
    {
        ReusePlan plan = mManager.planHybridMtp(
            domain, schema, draftSignature, candidates, fullBlockHashes, inputTokenCount, policy);
        bool const cacheDerivedPlan = plan.reuseTokenLength > 0;
        ReusePlanKind const planKind = plan.kind;
        AcquireResult acquired = mManager.acquire(plan);
        if (acquired.status == AcquireStatus::kStalePlan)
        {
            continue;
        }
        if (acquired.status == AcquireStatus::kInsufficientCapacity && cacheDerivedPlan
            && policy == LookupPolicy::kUseCache)
        {
            policy = LookupPolicy::kBypass;
            forcedCold = true;
            continue;
        }
        return RuntimeCacheAcquireResult{std::move(acquired.lease), acquired.status, planKind, forcedCold};
    }
    return RuntimeCacheAcquireResult{
        std::nullopt, AcquireStatus::kStalePlan, ReusePlanKind::kNoReusablePrefix, forcedCold};
}

RuntimeCacheAcquireResult ContextCacheRuntimeAdapter::acquireVanilla(CacheDomainId domain,
    std::vector<BlockHash> const& fullBlockHashes, int32_t inputTokenCount, LookupPolicy lookupPolicy)
{
    LookupPolicy policy = lookupPolicy;
    bool forcedCold = false;
    constexpr int32_t kMAX_STALE_RETRIES = 2;
    for (int32_t attempt = 0; attempt <= kMAX_STALE_RETRIES; ++attempt)
    {
        ReusePlan plan = mManager.planVanilla(domain, fullBlockHashes, inputTokenCount, policy);
        bool const cacheDerivedPlan = plan.reuseTokenLength > 0 || plan.kind == ReusePlanKind::kFullInputRewind;
        ReusePlanKind const planKind = plan.kind;
        AcquireResult acquired = mManager.acquire(plan);
        if (acquired.status == AcquireStatus::kStalePlan)
        {
            continue;
        }
        if (acquired.status == AcquireStatus::kInsufficientCapacity && cacheDerivedPlan
            && policy == LookupPolicy::kUseCache)
        {
            policy = LookupPolicy::kBypass;
            forcedCold = true;
            continue;
        }
        return RuntimeCacheAcquireResult{std::move(acquired.lease), acquired.status, planKind, forcedCold};
    }
    return RuntimeCacheAcquireResult{
        std::nullopt, AcquireStatus::kStalePlan, ReusePlanKind::kNoReusablePrefix, forcedCold};
}

void ContextCacheRuntimeAdapter::bindBaseRow(KVPageTable& table, int32_t slot, CacheRequestLease const& lease) const
{
    ELLM_CHECK(lease.valid(), "Cannot bind an invalid context cache lease");
    ELLM_CHECK(lease.basePages().size() <= static_cast<size_t>(table.maxPagesPerSeq()),
        "Context cache lease exceeds the page-table row capacity");
    table.setRow(slot, lease.basePages().data(), static_cast<int32_t>(lease.basePages().size()));
}

void ContextCacheRuntimeAdapter::bindDraftRow(KVPageTable& table, int32_t slot, CacheRequestLease const& lease) const
{
    ELLM_CHECK(lease.valid(), "Cannot bind an invalid context cache lease");
    ELLM_CHECK(lease.draftPages().size() <= static_cast<size_t>(table.maxPagesPerSeq()),
        "Context cache EAGLE lease exceeds the draft page-table row capacity");
    table.setRow(slot, lease.draftPages().data(), static_cast<int32_t>(lease.draftPages().size()));
}

bool ContextCacheRuntimeAdapter::growBaseToPageCount(
    CacheRequestLease& lease, int32_t requiredPages, KVPageTable& table, int32_t slot)
{
    ELLM_CHECK(requiredPages >= 0, "Required context cache page count must be non-negative");
    int32_t const currentPages = static_cast<int32_t>(lease.basePages().size());
    if (requiredPages <= currentPages)
    {
        return true;
    }
    if (!mManager.growBasePages(lease, requiredPages - currentPages))
    {
        return false;
    }
    bindBaseRow(table, slot, lease);
    return true;
}

bool ContextCacheRuntimeAdapter::growSpecToPageCounts(CacheRequestLease& lease, int32_t requiredBasePages,
    int32_t requiredDraftPages, KVPageTable& baseTable, KVPageTable& draftTable, int32_t slot)
{
    ELLM_CHECK(requiredBasePages >= 0 && requiredDraftPages >= 0,
        "Required EAGLE context cache page counts must be non-negative");
    int32_t const currentBasePages = static_cast<int32_t>(lease.basePages().size());
    int32_t const currentDraftPages = static_cast<int32_t>(lease.draftPages().size());
    int32_t const additionalBasePages = std::max(requiredBasePages - currentBasePages, 0);
    int32_t const additionalDraftPages = std::max(requiredDraftPages - currentDraftPages, 0);
    if (additionalBasePages == 0 && additionalDraftPages == 0)
    {
        return true;
    }
    if (!mManager.growSpecPages(lease, additionalBasePages, additionalDraftPages))
    {
        return false;
    }
    bindBaseRow(baseTable, slot, lease);
    bindDraftRow(draftTable, slot, lease);
    return true;
}

bool ContextCacheRuntimeAdapter::growHybridMtpToPageCounts(CacheRequestLease& lease, int32_t requiredBasePages,
    int32_t requiredDraftPages, KVPageTable& baseTable, KVPageTable& draftTable, int32_t slot)
{
    ELLM_CHECK(requiredBasePages >= 0 && requiredDraftPages >= 0,
        "Required Hybrid+MTP context cache page counts must be non-negative");
    int32_t const currentBasePages = static_cast<int32_t>(lease.basePages().size());
    int32_t const currentDraftPages = static_cast<int32_t>(lease.draftPages().size());
    int32_t const additionalBasePages = std::max(requiredBasePages - currentBasePages, 0);
    int32_t const additionalDraftPages = std::max(requiredDraftPages - currentDraftPages, 0);
    if (additionalBasePages == 0 && additionalDraftPages == 0)
    {
        return true;
    }
    if (!mManager.growHybridMtpPages(lease, additionalBasePages, additionalDraftPages))
    {
        return false;
    }
    bindBaseRow(baseTable, slot, lease);
    bindDraftRow(draftTable, slot, lease);
    return true;
}

std::vector<int32_t> ContextCacheRuntimeAdapter::hybridCandidateLengths(
    CacheDomainId domain, RecurrentStateSchemaId schema, int32_t inputTokenCount) const
{
    return mManager.hybridCandidateLengths(domain, schema, inputTokenCount);
}

std::vector<int32_t> ContextCacheRuntimeAdapter::hybridMtpCandidateLengths(CacheDomainId domain,
    RecurrentStateSchemaId schema, DraftEngineSignature draftSignature, int32_t inputTokenCount) const
{
    return mManager.hybridMtpCandidateLengths(domain, schema, draftSignature, inputTokenCount);
}

PublishResult ContextCacheRuntimeAdapter::publish(CacheRequestLease& lease, PublishRequest const& request)
{
    return mManager.publishDetailed(lease, request);
}

std::optional<HybridSnapshotReservation> ContextCacheRuntimeAdapter::reserveHybridSnapshots(
    CacheRequestLease& lease, bool needsPartialKvSnapshot)
{
    return mManager.reserveHybridSnapshots(lease, needsPartialKvSnapshot);
}

void ContextCacheRuntimeAdapter::releaseRestoredHybridSnapshots(CacheRequestLease& lease)
{
    mManager.releaseRestoredHybridSnapshots(lease);
}

PublishResult ContextCacheRuntimeAdapter::publishHybrid(CacheRequestLease& lease, HybridPublishRequest const& request)
{
    return mManager.publishHybridDetailed(lease, request);
}

PublishResult ContextCacheRuntimeAdapter::publishHybridMtp(
    CacheRequestLease& lease, HybridMtpPublishRequest const& request)
{
    return mManager.publishHybridMtpDetailed(lease, request);
}

void ContextCacheRuntimeAdapter::retireHybridSnapshotReservation(
    CacheRequestLease& lease, HybridSnapshotReservation const& reservation)
{
    mManager.retireHybridSnapshotReservation(lease, reservation);
}

void ContextCacheRuntimeAdapter::rebindBasePrefix(CacheRequestLease& lease, std::vector<PageId> const& canonicalPages)
{
    mManager.rebindBasePrefix(lease, canonicalPages);
}

ContextCacheManager const& ContextCacheRuntimeAdapter::manager() const noexcept
{
    return mManager;
}

} // namespace rt
} // namespace trt_edgellm
