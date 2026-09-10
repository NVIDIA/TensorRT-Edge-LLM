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
#include "runtime/config/llmEngineConfig.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

using namespace trt_edgellm::rt;

static_assert(!std::is_copy_constructible_v<CacheRequestLease>);
static_assert(!std::is_copy_assignable_v<CacheRequestLease>);
static_assert(std::is_nothrow_move_constructible_v<CacheRequestLease>);
static_assert(std::is_nothrow_move_assignable_v<CacheRequestLease>);
static_assert(std::is_nothrow_destructible_v<CacheRequestLease>);

namespace
{

constexpr int32_t kPAGE_SIZE = 4;
constexpr BlockHash kHASH_A{0x1111111111111111ULL, 0xAAAAAAAAAAAAAAAAULL};
constexpr BlockHash kHASH_B{0x2222222222222222ULL, 0xBBBBBBBBBBBBBBBBULL};
constexpr BlockHash kHASH_C{0x3333333333333333ULL, 0xCCCCCCCCCCCCCCCCULL};
constexpr BlockHash kHASH_D{0x4444444444444444ULL, 0xDDDDDDDDDDDDDDDDULL};
constexpr BlockHash kHASH_E{0x5555555555555555ULL, 0xEEEEEEEEEEEEEEEEULL};

struct PublishedRecord
{
    PublishStatus status{};
    std::vector<PageId> producerPages;
    RecordId id{};
};

struct PublishedSpecRecord
{
    PublishStatus status{};
    std::vector<PageId> producerBasePages;
    std::vector<PageId> producerDraftPages;
    RecordId id{};
};

CacheRequestLease takeLease(AcquireResult& result)
{
    if (!result.lease.has_value())
    {
        throw std::runtime_error("Test context cache acquisition failed");
    }
    return std::move(*result.lease);
}

CacheRequestLease acquirePrivatePages(ContextCacheManager& manager, int32_t count)
{
    AcquireResult result = manager.acquireVanilla({}, count * kPAGE_SIZE);
    return takeLease(result);
}

SpecReuseContract eagleSpecReuseContract()
{
    return SpecReuseContract{/*ownsPagedSpecState=*/true, /*futureDependencyTokens=*/1};
}

std::vector<PageId> const& eaglePagePath(CacheRecord const& record)
{
    ELLM_CHECK(record.specState.has_value(), "Test expected an EAGLE spec-state record");
    return record.specState->pagePath;
}

bool hasSpecState(CacheRecord const& record) noexcept
{
    return record.specState.has_value();
}

PublishedRecord publishPrivateRecord(ContextCacheManager& manager, std::vector<BlockHash> hashes)
{
    CacheRequestLease lease = acquirePrivatePages(manager, static_cast<int32_t>(hashes.size()));
    std::vector<PageId> const producerPages = lease.basePages();
    PublishStatus const status
        = manager.publish(lease, PublishRequest{hashes, static_cast<int32_t>(hashes.size()) * kPAGE_SIZE}).status;
    std::vector<RecordId> const lru = manager.records().lruToMru();
    if (lru.empty())
    {
        throw std::runtime_error("Test context cache publication did not retain a record");
    }
    lease.release();
    return PublishedRecord{status, producerPages, lru.back()};
}

PublishedSpecRecord publishPrivateSpecRecord(ContextCacheManager& manager, std::vector<BlockHash> hashes)
{
    AcquireResult result = manager.acquireSpec(hashes, static_cast<int32_t>(hashes.size()) * kPAGE_SIZE);
    CacheRequestLease lease = takeLease(result);
    std::vector<PageId> const producerBasePages = lease.basePages();
    std::vector<PageId> const producerDraftPages = lease.draftPages();
    PublishStatus const status
        = manager.publish(lease, PublishRequest{hashes, static_cast<int32_t>(hashes.size()) * kPAGE_SIZE}).status;
    std::vector<RecordId> const lru = manager.records().lruToMru();
    if (lru.empty())
    {
        throw std::runtime_error("Test speculative context cache publication did not retain a record");
    }
    lease.release();
    return PublishedSpecRecord{status, producerBasePages, producerDraftPages, lru.back()};
}

void expectPoolState(ContextCacheManager const& manager, ResourceType type, std::vector<int32_t> const& activeRefs,
    std::vector<int32_t> const& cacheRefs, int32_t freeCount)
{
    ResourcePools const& pools = manager.pools();
    ASSERT_EQ(activeRefs.size(), cacheRefs.size());
    ASSERT_EQ(pools.capacity(type), static_cast<int32_t>(activeRefs.size()));
    EXPECT_EQ(pools.freeCount(type), freeCount);
    for (int32_t index = 0; index < pools.capacity(type); ++index)
    {
        ResourceId const resource{type, index};
        EXPECT_EQ(pools.activeRefCount(resource), activeRefs[static_cast<size_t>(index)]);
        EXPECT_EQ(pools.cacheRefCount(resource), cacheRefs[static_cast<size_t>(index)]);
    }
}

void expectOnlyBaseState(ContextCacheManager const& manager, std::vector<int32_t> const& activeRefs,
    std::vector<int32_t> const& cacheRefs, int32_t freeCount)
{
    expectPoolState(manager, ResourceType::kBaseKvPage, activeRefs, cacheRefs, freeCount);
    expectPoolState(manager, ResourceType::kDraftKvPage, {}, {}, 0);
    expectPoolState(manager, ResourceType::kRecurrentSnapshot, {}, {}, 0);
    expectPoolState(manager, ResourceType::kPartialKvSnapshot, {}, {}, 0);
}

} // namespace

TEST(ContextCacheManagerTests, AcquirePinsHitsBeforeEvictionAndAllocatesPrivatePages)
{
    EXPECT_THROW((void) ContextCacheManager(0, ResourceDemand{}, 0), std::runtime_error);
    EXPECT_THROW((void) ContextCacheManager(-1, ResourceDemand{}, 0), std::runtime_error);
    EXPECT_THROW((void) ContextCacheManager(kPAGE_SIZE, ResourceDemand{-1, 0, 0, 0}, 0), std::runtime_error);
    EXPECT_THROW((void) ContextCacheManager(kPAGE_SIZE, ResourceDemand{}, -1), std::runtime_error);

    ContextCacheManager manager(kPAGE_SIZE, ResourceDemand{2, 0, 0, 0}, 2);
    PublishedRecord const first = publishPrivateRecord(manager, {kHASH_A});
    PublishedRecord const second = publishPrivateRecord(manager, {kHASH_B});
    ASSERT_EQ(first.status, PublishStatus::kPublished);
    ASSERT_EQ(second.status, PublishStatus::kPublished);
    ASSERT_EQ(first.producerPages, std::vector<PageId>{0});
    ASSERT_EQ(second.producerPages, std::vector<PageId>{1});
    ASSERT_EQ(manager.records().lruToMru(), std::vector<RecordId>({first.id, second.id}));
    expectOnlyBaseState(manager, {0, 0}, {1, 1}, 0);

    AcquireResult result = manager.acquireVanilla({kHASH_A}, kPAGE_SIZE + 1);
    ReusePlan const& plan = result.plan;
    ASSERT_EQ(plan.basePageBindings, std::vector<PageId>{0});
    ASSERT_EQ(plan.demand.baseKvPages, 1);

    ASSERT_TRUE(result.lease.has_value());
    EXPECT_EQ(result.status, AcquireStatus::kAcquired);
    CacheRequestLease lease = takeLease(result);
    EXPECT_TRUE(lease.valid());
    EXPECT_EQ(plan.reuseTokenLength, kPAGE_SIZE);
    EXPECT_EQ(lease.basePages(), std::vector<PageId>({0, 1}));
    EXPECT_TRUE(manager.records().lruToMru().empty());
    EXPECT_FALSE(manager.baseIndex().lookup(kHASH_A).has_value());
    EXPECT_FALSE(manager.baseIndex().lookup(kHASH_B).has_value());
    expectOnlyBaseState(manager, {1, 1}, {0, 0}, 0);

    lease.release();

    EXPECT_FALSE(lease.valid());
    EXPECT_TRUE(lease.basePages().empty());
    expectOnlyBaseState(manager, {0, 0}, {0, 0}, 2);
}

TEST(ContextCacheManagerTests, BypassAcquireBuildsAForcedColdPlan)
{
    ContextCacheManager manager(kPAGE_SIZE, ResourceDemand{1, 0, 0, 0}, 1);
    PublishedRecord const cached = publishPrivateRecord(manager, {kHASH_A});
    ASSERT_EQ(cached.status, PublishStatus::kPublished);
    ASSERT_EQ(manager.pools().freeCount(ResourceType::kBaseKvPage), 0);

    AcquireResult result = manager.acquireVanilla({kHASH_A}, kPAGE_SIZE, ContextCacheLookupPolicy::kBypass);
    ReusePlan const& cold = result.plan;
    EXPECT_EQ(cold.kind, ReusePlanKind::kNoReusablePrefix);
    EXPECT_EQ(cold.reuseTokenLength, 0);
    EXPECT_TRUE(cold.matchedBlockHashes.empty());
    EXPECT_TRUE(cold.basePageBindings.empty());
    EXPECT_EQ(cold.demand.baseKvPages, 1);

    ASSERT_EQ(result.status, AcquireStatus::kAcquired);
    CacheRequestLease lease = takeLease(result);
    EXPECT_EQ(lease.basePages(), std::vector<PageId>{cached.producerPages.front()});
    EXPECT_TRUE(manager.records().lruToMru().empty());
    EXPECT_EQ(manager.baseIndex().size(), 0U);
}

TEST(ContextCacheManagerTests, StandardPublicationRejectsHybridLease)
{
    ContextCacheManager manager(kPAGE_SIZE, ResourceDemand{1, 0, 0, 0}, 1);
    AcquireResult result = manager.acquireHybrid({}, {kHASH_A}, /*inputTokenCount=*/kPAGE_SIZE, /*hasAttention=*/true);
    CacheRequestLease lease = takeLease(result);

    EXPECT_THROW((void) manager.publish(lease, PublishRequest{{kHASH_A}, kPAGE_SIZE}), std::runtime_error);
    EXPECT_TRUE(manager.records().lruToMru().empty());
    lease.release();
}

TEST(ContextCacheManagerTests, HybridCheckpointPublishesAndAcquiresAllStateAtomically)
{
    constexpr BlockHash kEXACT_DIGEST{0x9191919191919191ULL, 0xA1A1A1A1A1A1A1A1ULL};
    ContextCacheManager manager(kPAGE_SIZE, ResourceDemand{4, 0, 2, 2}, 2);

    AcquireResult coldResult = manager.acquireHybrid({}, {kHASH_A}, /*inputTokenCount=*/6, /*hasAttention=*/true);
    ReusePlan const& cold = coldResult.plan;
    EXPECT_EQ(cold.mode, ReusePlanMode::kHybrid);
    EXPECT_EQ(cold.demand.baseKvPages, 2);
    CacheRequestLease coldLease = takeLease(coldResult);
    ASSERT_EQ(coldLease.basePages(), std::vector<PageId>({0, 1}));

    std::optional<HybridSnapshotReservation> const snapshots = manager.reserveHybridSnapshots(coldLease, true);
    ASSERT_TRUE(snapshots.has_value());
    EXPECT_EQ(snapshots->recurrentSnapshotSlot, 0);
    EXPECT_EQ(snapshots->partialKvSnapshotSlot, std::optional<int32_t>{0});

    HybridCheckpointKey const checkpoint{kEXACT_DIGEST, 6};
    PublishResult const published
        = manager.publishHybrid(coldLease, HybridPublishRequest{{kHASH_A}, checkpoint, *snapshots});
    ASSERT_EQ(published.status, PublishStatus::kPublished);
    ASSERT_TRUE(published.record.has_value());
    EXPECT_EQ(published.canonicalBasePages, std::vector<PageId>{0});
    manager.retireHybridSnapshotReservation(coldLease, *snapshots);
    coldLease.release();

    CacheRecord const& record = manager.records().get(*published.record);
    EXPECT_EQ(record.hybridKey(), std::optional<HybridCheckpointKey>{checkpoint});
    EXPECT_EQ(record.basePagePath, std::vector<PageId>{0});
    EXPECT_EQ(record.recurrentSnapshotSlot, std::optional<int32_t>{0});
    EXPECT_EQ(record.partialKvSnapshotSlot, std::optional<int32_t>{0});

    AcquireResult hitResult = manager.acquireHybrid(
        {{6, kEXACT_DIGEST}}, {kHASH_A, kHASH_B}, /*inputTokenCount=*/10, /*hasAttention=*/true);
    ReusePlan const& hit = hitResult.plan;
    ASSERT_EQ(hit.reuseTokenLength, 6);
    ASSERT_EQ(hitResult.status, AcquireStatus::kAcquired);
    CacheRequestLease hitLease = takeLease(hitResult);
    EXPECT_EQ(hitLease.basePages(), std::vector<PageId>({0, 2, 3}));
    EXPECT_EQ(hitLease.recurrentSnapshotSlot(), std::optional<int32_t>{0});
    EXPECT_EQ(hitLease.partialKvSnapshotSlot(), std::optional<int32_t>{0});
    manager.releaseRestoredHybridSnapshots(hitLease);
    EXPECT_FALSE(hitLease.recurrentSnapshotSlot().has_value());
    EXPECT_FALSE(hitLease.partialKvSnapshotSlot().has_value());
    hitLease.release();
}

TEST(ContextCacheManagerTests, HybridMtpCheckpointPublishesAndAcquiresPairedBaseAndDraftState)
{
    constexpr BlockHash kEXACT_DIGEST{0x9494949494949494ULL, 0xA4A4A4A4A4A4A4A4ULL};
    ContextCacheManager manager(kPAGE_SIZE, ResourceDemand{8, 8, 2, 2}, 2);

    // Cold hybrid+MTP acquire: a speculative deployment reserves the full input in both the base and draft pools.
    AcquireResult coldResult = manager.acquireHybridMtp({}, {kHASH_A}, /*inputTokenCount=*/6);
    ReusePlan const& cold = coldResult.plan;
    EXPECT_EQ(cold.mode, ReusePlanMode::kHybridMtp);
    EXPECT_EQ(cold.demand.baseKvPages, 2);
    EXPECT_EQ(cold.demand.draftKvPages, 2);
    CacheRequestLease coldLease = takeLease(coldResult);
    ASSERT_EQ(coldLease.basePages(), std::vector<PageId>({0, 1}));
    ASSERT_EQ(coldLease.draftPages(), std::vector<PageId>({0, 1}));

    // MTP always bundles a partial-KV snapshot alongside the recurrent snapshot.
    std::optional<HybridSnapshotReservation> const snapshots = manager.reserveHybridSnapshots(coldLease, true);
    ASSERT_TRUE(snapshots.has_value());
    EXPECT_EQ(snapshots->recurrentSnapshotSlot, 0);
    EXPECT_EQ(snapshots->partialKvSnapshotSlot, std::optional<int32_t>{0});

    // exactLength 6 with pageSize 4 keeps the boundary token (index 5) private, so only (6 - 1) / 4 == 1 full block is
    // published for both base and draft state.
    HybridCheckpointKey const checkpoint{kEXACT_DIGEST, 6};
    PublishResult const published
        = manager.publishHybridMtp(coldLease, HybridPublishRequest{{kHASH_A}, checkpoint, *snapshots});
    ASSERT_EQ(published.status, PublishStatus::kPublished);
    ASSERT_TRUE(published.record.has_value());
    EXPECT_EQ(published.publishedBaseFullBlockCount, 1);
    EXPECT_EQ(published.canonicalBasePages, std::vector<PageId>{0});
    manager.retireHybridSnapshotReservation(coldLease, *snapshots);
    coldLease.release();

    // The published record carries paired base/draft paths of size (exactLength - 1) / pageSize plus both snapshots.
    CacheRecord const& record = manager.records().get(*published.record);
    EXPECT_EQ(record.hybridKey(), std::optional<HybridCheckpointKey>{checkpoint});
    ASSERT_TRUE(record.specState.has_value());
    ASSERT_EQ(record.specState->pagePath.size(), static_cast<size_t>((6 - 1) / kPAGE_SIZE));
    EXPECT_FALSE(record.specState->pagePath.empty());
    EXPECT_EQ(record.specState->pagePath, std::vector<PageId>{0});
    EXPECT_EQ(record.basePagePath, std::vector<PageId>{0});
    EXPECT_EQ(record.recurrentSnapshotSlot, std::optional<int32_t>{0});
    EXPECT_EQ(record.partialKvSnapshotSlot, std::optional<int32_t>{0});
    EXPECT_EQ(record.exactCheckpointLength, std::optional<int32_t>{6});

    // Hit: the exact checkpoint rebinds the cached base AND draft page paths plus both snapshot slots.
    AcquireResult hitResult
        = manager.acquireHybridMtp({{6, kEXACT_DIGEST}}, {kHASH_A, kHASH_B}, /*inputTokenCount=*/10);
    ReusePlan const& hit = hitResult.plan;
    EXPECT_EQ(hit.mode, ReusePlanMode::kHybridMtp);
    ASSERT_EQ(hit.reuseTokenLength, 6);
    ASSERT_EQ(hitResult.status, AcquireStatus::kAcquired);
    EXPECT_EQ(hit.basePageBindings, std::vector<PageId>{0});
    EXPECT_EQ(hit.specPageBindings, std::vector<PageId>{0});
    CacheRequestLease hitLease = takeLease(hitResult);
    ASSERT_FALSE(hitLease.basePages().empty());
    ASSERT_FALSE(hitLease.draftPages().empty());
    EXPECT_EQ(hitLease.basePages().front(), 0);
    EXPECT_EQ(hitLease.draftPages().front(), 0);
    EXPECT_EQ(hitLease.basePages().size(), hitLease.draftPages().size());
    EXPECT_EQ(hitLease.recurrentSnapshotSlot(), std::optional<int32_t>{0});
    EXPECT_EQ(hitLease.partialKvSnapshotSlot(), std::optional<int32_t>{0});
    manager.releaseRestoredHybridSnapshots(hitLease);
    EXPECT_FALSE(hitLease.recurrentSnapshotSlot().has_value());
    EXPECT_FALSE(hitLease.partialKvSnapshotSlot().has_value());
    hitLease.release();
}

TEST(ContextCacheManagerTests, PureRecurrentCheckpointNeedsNoBaseOrPartialPage)
{
    constexpr BlockHash kEXACT_DIGEST{0x9292929292929292ULL, 0xA2A2A2A2A2A2A2A2ULL};
    ContextCacheManager manager(kPAGE_SIZE, ResourceDemand{0, 0, 2, 0}, 2);

    AcquireResult coldResult = manager.acquireHybrid({}, {}, /*inputTokenCount=*/3, /*hasAttention=*/false);
    CacheRequestLease coldLease = takeLease(coldResult);
    EXPECT_TRUE(coldLease.basePages().empty());
    std::optional<HybridSnapshotReservation> const snapshots = manager.reserveHybridSnapshots(coldLease, false);
    ASSERT_TRUE(snapshots.has_value());
    ASSERT_FALSE(snapshots->partialKvSnapshotSlot.has_value());

    HybridCheckpointKey const checkpoint{kEXACT_DIGEST, 3};
    PublishResult const published = manager.publishHybrid(coldLease, HybridPublishRequest{{}, checkpoint, *snapshots});
    ASSERT_EQ(published.status, PublishStatus::kPublished);
    manager.retireHybridSnapshotReservation(coldLease, *snapshots);
    coldLease.release();

    AcquireResult hitResult
        = manager.acquireHybrid({{3, kEXACT_DIGEST}}, {kHASH_A}, /*inputTokenCount=*/5, /*hasAttention=*/false);
    ReusePlan const& hit = hitResult.plan;
    EXPECT_EQ(hit.reuseTokenLength, 3);
    EXPECT_TRUE(hit.basePageBindings.empty());
    EXPECT_EQ(hit.demand.baseKvPages, 0);
    CacheRequestLease hitLease = takeLease(hitResult);
    EXPECT_TRUE(hitLease.basePages().empty());
    EXPECT_TRUE(hitLease.recurrentSnapshotSlot().has_value());
    EXPECT_FALSE(hitLease.partialKvSnapshotSlot().has_value());
}

TEST(ContextCacheManagerTests, HybridPublicationProjectsCanonicalOverlapAndRetainsDescendant)
{
    constexpr BlockHash kEXACT_DIGEST{0x9393939393939393ULL, 0xA3A3A3A3A3A3A3A3ULL};
    ContextCacheManager manager(kPAGE_SIZE, ResourceDemand{5, 0, 2, 0}, 4);
    PublishedRecord const cached = publishPrivateRecord(manager, {kHASH_A});
    ASSERT_EQ(cached.producerPages, std::vector<PageId>{0});

    AcquireResult coldResult = manager.acquireHybrid(
        {}, {kHASH_A, kHASH_B}, 2 * kPAGE_SIZE, /*hasAttention=*/true, ContextCacheLookupPolicy::kBypass);
    CacheRequestLease coldLease = takeLease(coldResult);
    ASSERT_EQ(coldLease.basePages(), std::vector<PageId>({1, 2}));
    std::optional<HybridSnapshotReservation> const snapshots = manager.reserveHybridSnapshots(coldLease, false);
    ASSERT_TRUE(snapshots.has_value());

    HybridCheckpointKey const checkpoint{kEXACT_DIGEST, 2 * kPAGE_SIZE};
    PublishResult const published
        = manager.publishHybrid(coldLease, HybridPublishRequest{{kHASH_A, kHASH_B}, checkpoint, *snapshots});

    ASSERT_EQ(published.status, PublishStatus::kPublished);
    ASSERT_TRUE(published.record.has_value());
    EXPECT_EQ(published.publishedBaseFullBlockCount, 2);
    EXPECT_EQ(published.canonicalBasePages, std::vector<PageId>({0, 2}));
    EXPECT_EQ(coldLease.basePages(), std::vector<PageId>({1, 2}));
    EXPECT_EQ(manager.records().findHybrid(checkpoint), published.record);
    EXPECT_EQ(manager.records().get(*published.record).basePagePath, std::vector<PageId>({0, 2}));
    EXPECT_EQ(manager.baseIndex().lookup(kHASH_B), std::optional<PageId>{2});
    expectPoolState(manager, ResourceType::kBaseKvPage, {0, 1, 1, 0, 0}, {2, 0, 1, 0, 0}, 2);
    expectPoolState(manager, ResourceType::kRecurrentSnapshot, {1, 0}, {1, 0}, 1);

    manager.retireHybridSnapshotReservation(coldLease, *snapshots);
    coldLease.release();

    expectPoolState(manager, ResourceType::kBaseKvPage, {0, 0, 0, 0, 0}, {2, 0, 1, 0, 0}, 3);
    expectPoolState(manager, ResourceType::kRecurrentSnapshot, {0, 0}, {1, 0}, 1);
}

TEST(ContextCacheManagerTests, PublicationProjectsCanonicalOverlapWithoutRebindingLease)
{
    ContextCacheManager manager(kPAGE_SIZE, ResourceDemand{4, 0, 0, 0}, 4);
    PublishedRecord const cached = publishPrivateRecord(manager, {kHASH_A});
    ASSERT_EQ(cached.producerPages, std::vector<PageId>{0});

    AcquireResult result
        = manager.acquireVanilla({kHASH_A, kHASH_B}, 2 * kPAGE_SIZE, ContextCacheLookupPolicy::kBypass);
    CacheRequestLease lease = takeLease(result);
    ASSERT_EQ(lease.basePages(), std::vector<PageId>({1, 2}));

    PublishResult const published = manager.publish(lease, PublishRequest{{kHASH_A, kHASH_B}, 2 * kPAGE_SIZE});

    ASSERT_EQ(published.status, PublishStatus::kPublished);
    ASSERT_TRUE(published.record.has_value());
    EXPECT_EQ(published.publishedBaseFullBlockCount, 2);
    EXPECT_EQ(published.canonicalBasePages, std::vector<PageId>({0, 2}));
    EXPECT_EQ(lease.basePages(), std::vector<PageId>({1, 2}));
    EXPECT_EQ(manager.records().get(*published.record).basePagePath, std::vector<PageId>({0, 2}));
    EXPECT_EQ(manager.baseIndex().lookup(kHASH_B), std::optional<PageId>{2});
    expectOnlyBaseState(manager, {0, 1, 1, 0}, {2, 0, 1, 0}, 1);

    lease.release();

    expectOnlyBaseState(manager, {0, 0, 0, 0}, {2, 0, 1, 0}, 2);
}

TEST(ContextCacheManagerTests, LeaseDestructorReleasesEveryUnpublishedResource)
{
    ContextCacheManager manager(kPAGE_SIZE, ResourceDemand{2, 1, 1, 1}, 1, eagleSpecReuseContract());

    {
        AcquireResult result = manager.acquireSpec({}, kPAGE_SIZE);
        ASSERT_TRUE(result.lease.has_value());
        CacheRequestLease source = takeLease(result);
        ASSERT_FALSE(result.lease->valid());

        CacheRequestLease moved(std::move(source));
        EXPECT_FALSE(source.valid());
        EXPECT_TRUE(moved.valid());
        ASSERT_EQ(moved.basePages(), std::vector<PageId>{0});

        CacheRequestLease destination = acquirePrivatePages(manager, 1);
        ASSERT_EQ(destination.basePages(), std::vector<PageId>{1});
        destination = std::move(moved);

        EXPECT_FALSE(moved.valid());
        EXPECT_TRUE(destination.valid());
        EXPECT_EQ(destination.basePages(), std::vector<PageId>{0});
        expectPoolState(manager, ResourceType::kBaseKvPage, {1, 0}, {0, 0}, 1);
        expectPoolState(manager, ResourceType::kDraftKvPage, {1}, {0}, 0);
        expectPoolState(manager, ResourceType::kRecurrentSnapshot, {0}, {0}, 1);
        expectPoolState(manager, ResourceType::kPartialKvSnapshot, {0}, {0}, 1);
    }

    expectPoolState(manager, ResourceType::kBaseKvPage, {0, 0}, {0, 0}, 2);
    expectPoolState(manager, ResourceType::kDraftKvPage, {0}, {0}, 1);

    {
        AcquireResult result = manager.acquireHybrid({}, {}, kPAGE_SIZE, /*hasAttention=*/true);
        CacheRequestLease lease = takeLease(result);
        std::optional<HybridSnapshotReservation> const snapshots = manager.reserveHybridSnapshots(lease, true);
        ASSERT_TRUE(snapshots.has_value());
        expectPoolState(manager, ResourceType::kRecurrentSnapshot, {1}, {0}, 0);
        expectPoolState(manager, ResourceType::kPartialKvSnapshot, {1}, {0}, 0);
    }

    expectPoolState(manager, ResourceType::kRecurrentSnapshot, {0}, {0}, 1);
    expectPoolState(manager, ResourceType::kPartialKvSnapshot, {0}, {0}, 1);

    CacheRequestLease explicitRelease = acquirePrivatePages(manager, 1);
    ASSERT_TRUE(explicitRelease.valid());
    explicitRelease.release();
    explicitRelease.release();
    EXPECT_FALSE(explicitRelease.valid());

    expectPoolState(manager, ResourceType::kBaseKvPage, {0, 0}, {0, 0}, 2);
    expectPoolState(manager, ResourceType::kDraftKvPage, {0}, {0}, 1);
    expectPoolState(manager, ResourceType::kRecurrentSnapshot, {0}, {0}, 1);
    expectPoolState(manager, ResourceType::kPartialKvSnapshot, {0}, {0}, 1);
}

TEST(ContextCacheManagerTests, InfeasibleAcquireRollsBackPinsWithoutEviction)
{
    ContextCacheManager manager(kPAGE_SIZE, ResourceDemand{1, 0, 0, 0}, 1);
    PublishedRecord const cached = publishPrivateRecord(manager, {kHASH_A});
    ASSERT_EQ(cached.status, PublishStatus::kPublished);
    ASSERT_EQ(cached.producerPages, std::vector<PageId>{0});
    AcquireResult const infeasible = manager.acquireVanilla({kHASH_A}, kPAGE_SIZE + 1);

    EXPECT_FALSE(infeasible.lease.has_value());
    EXPECT_EQ(infeasible.status, AcquireStatus::kInsufficientCapacity);
    EXPECT_EQ(manager.records().lruToMru(), std::vector<RecordId>{cached.id});
    EXPECT_EQ(manager.baseIndex().lookup(kHASH_A), std::optional<PageId>{0});
    expectOnlyBaseState(manager, {0}, {1}, 0);
}

TEST(ContextCacheManagerTests, EvictionMetricsCountOnlyResourcesReturnedToFreeLists)
{
    ContextCacheManager manager(kPAGE_SIZE, ResourceDemand{2, 0, 0, 0}, 1);
    publishPrivateRecord(manager, {kHASH_A});
    publishPrivateRecord(manager, {kHASH_B});

    ContextCacheManagerMetrics const& retainedMetrics = manager.metrics();
    EXPECT_EQ(retainedMetrics.evictedRecords, 1U);
    EXPECT_EQ(retainedMetrics.reclaimedBaseKvPages, 1U);
    EXPECT_EQ(retainedMetrics.reclaimedDraftKvPages, 0U);
    EXPECT_EQ(retainedMetrics.reclaimedRecurrentSnapshots, 0U);
    EXPECT_EQ(retainedMetrics.reclaimedPartialKvSnapshots, 0U);

    ContextCacheManager transientManager(kPAGE_SIZE, ResourceDemand{1, 0, 0, 0}, 0);
    CacheRequestLease transient = acquirePrivatePages(transientManager, 1);
    EXPECT_EQ(
        transientManager.publish(transient, PublishRequest{{kHASH_A}, kPAGE_SIZE}).status, PublishStatus::kPublished);
    EXPECT_EQ(transientManager.metrics().evictedRecords, 1U);
    EXPECT_EQ(transientManager.metrics().reclaimedBaseKvPages, 0U);
    transient.release();
    EXPECT_EQ(transientManager.metrics().reclaimedBaseKvPages, 0U);
}

TEST(ContextCacheManagerTests, PublishAddsCacheRefsBeforeDroppingActivity)
{
    ContextCacheManager manager(kPAGE_SIZE, ResourceDemand{1, 0, 0, 0}, 1);
    CacheRequestLease lease = acquirePrivatePages(manager, 1);
    ASSERT_EQ(lease.basePages(), std::vector<PageId>{0});
    expectOnlyBaseState(manager, {1}, {0}, 0);

    PublishRequest const negativeLength{{kHASH_A}, -1};
    EXPECT_THROW((void) manager.publish(lease, negativeLength), std::runtime_error);
    PublishRequest const noFullBlock{{kHASH_A}, kPAGE_SIZE - 1};
    EXPECT_THROW((void) manager.publish(lease, noFullBlock), std::runtime_error);
    EXPECT_TRUE(manager.records().lruToMru().empty());
    expectOnlyBaseState(manager, {1}, {0}, 0);

    PublishStatus const status = manager.publish(lease, PublishRequest{{kHASH_A}, kPAGE_SIZE}).status;

    EXPECT_EQ(status, PublishStatus::kPublished);
    ASSERT_EQ(manager.records().size(), 1U);
    RecordId const id = manager.records().lruToMru().front();
    CacheRecordKey const key{kHASH_A, 1};
    EXPECT_EQ(manager.records().find(key), std::optional<RecordId>{id});
    EXPECT_EQ(manager.records().get(id).logicalBlockHashes, std::vector<BlockHash>{kHASH_A});
    EXPECT_EQ(manager.records().get(id).basePagePath, std::vector<PageId>{0});
    EXPECT_EQ(manager.baseIndex().lookup(kHASH_A), std::optional<PageId>{0});
    EXPECT_TRUE(lease.valid());
    expectOnlyBaseState(manager, {1}, {1}, 0);

    lease.release();

    EXPECT_FALSE(lease.valid());
    expectOnlyBaseState(manager, {0}, {1}, 0);
}

TEST(ContextCacheManagerTests, LaterPublicationUsesCanonicalPageChosenByEarlierProducer)
{
    ContextCacheManager manager(kPAGE_SIZE, ResourceDemand{3, 0, 0, 0}, 3);
    CacheRequestLease firstProducer = acquirePrivatePages(manager, 1);
    CacheRequestLease secondProducer = acquirePrivatePages(manager, 2);
    ASSERT_EQ(firstProducer.basePages(), std::vector<PageId>{0});
    ASSERT_EQ(secondProducer.basePages(), std::vector<PageId>({1, 2}));

    PublishStatus const firstStatus = manager.publish(firstProducer, PublishRequest{{kHASH_A}, kPAGE_SIZE}).status;
    PublishResult const second = manager.publish(secondProducer, PublishRequest{{kHASH_A, kHASH_B}, 2 * kPAGE_SIZE});

    EXPECT_EQ(firstStatus, PublishStatus::kPublished);
    ASSERT_EQ(second.status, PublishStatus::kPublished);
    ASSERT_TRUE(second.record.has_value());
    EXPECT_EQ(second.canonicalBasePages, std::vector<PageId>({0, 2}));
    EXPECT_EQ(secondProducer.basePages(), std::vector<PageId>({1, 2}));
    EXPECT_EQ(manager.records().get(*second.record).basePagePath, std::vector<PageId>({0, 2}));
    EXPECT_EQ(manager.baseIndex().lookup(kHASH_A), std::optional<PageId>{0});
    EXPECT_EQ(manager.baseIndex().lookup(kHASH_B), std::optional<PageId>{2});
    EXPECT_EQ(manager.baseIndex().size(), 2U);
    expectOnlyBaseState(manager, {1, 1, 1}, {2, 0, 1}, 0);

    firstProducer.release();
    secondProducer.release();

    expectOnlyBaseState(manager, {0, 0, 0}, {2, 0, 1}, 1);
}

TEST(ContextCacheManagerTests, PublicationRollbackRestoresRefsAfterBaseMappingConflict)
{
    ContextCacheManager rollbackManager(kPAGE_SIZE, ResourceDemand{1, 0, 0, 0}, 2);
    CacheRequestLease conflictingLease = acquirePrivatePages(rollbackManager, 1);
    ASSERT_EQ(conflictingLease.basePages(), std::vector<PageId>{0});
    ASSERT_EQ(rollbackManager.publish(conflictingLease, PublishRequest{{kHASH_C}, kPAGE_SIZE}).status,
        PublishStatus::kPublished);
    RecordId const canonicalRecord = rollbackManager.records().lruToMru().front();
    PublishRequest const conflictingPublication{{kHASH_D}, kPAGE_SIZE};

    EXPECT_THROW((void) rollbackManager.publish(conflictingLease, conflictingPublication), std::runtime_error);

    EXPECT_EQ(rollbackManager.records().lruToMru(), std::vector<RecordId>{canonicalRecord});
    EXPECT_EQ(rollbackManager.baseIndex().lookup(kHASH_C), std::optional<PageId>{0});
    EXPECT_FALSE(rollbackManager.baseIndex().lookup(kHASH_D).has_value());
    expectOnlyBaseState(rollbackManager, {1}, {1}, 0);

    conflictingLease.release();

    expectOnlyBaseState(rollbackManager, {0}, {1}, 0);
}

TEST(ContextCacheManagerTests, EvictingOneBranchPreservesSharedAncestorPages)
{
    ContextCacheManager manager(kPAGE_SIZE, ResourceDemand{4, 0, 0, 0}, 2);
    PublishedRecord const firstBranch = publishPrivateRecord(manager, {kHASH_A, kHASH_B, kHASH_C});
    ASSERT_EQ(firstBranch.status, PublishStatus::kPublished);
    ASSERT_EQ(firstBranch.producerPages, std::vector<PageId>({0, 1, 2}));

    AcquireResult secondResult = manager.acquireVanilla({kHASH_A, kHASH_B, kHASH_D}, 3 * kPAGE_SIZE);
    ReusePlan const& secondPlan = secondResult.plan;
    ASSERT_EQ(secondPlan.basePageBindings, std::vector<PageId>({0, 1}));
    ASSERT_TRUE(secondResult.lease.has_value());
    CacheRequestLease secondLease = takeLease(secondResult);
    ASSERT_EQ(secondLease.basePages(), std::vector<PageId>({0, 1, 3}));
    PublishStatus const secondStatus
        = manager.publish(secondLease, PublishRequest{{kHASH_A, kHASH_B, kHASH_D}, 3 * kPAGE_SIZE}).status;
    ASSERT_EQ(secondStatus, PublishStatus::kPublished);
    RecordId const secondBranch = manager.records().lruToMru().back();
    secondLease.release();

    ASSERT_EQ(manager.records().lruToMru(), std::vector<RecordId>({firstBranch.id, secondBranch}));
    expectOnlyBaseState(manager, {0, 0, 0, 0}, {2, 2, 1, 1}, 0);

    CacheRequestLease replacementLease = acquirePrivatePages(manager, 1);
    ASSERT_EQ(replacementLease.basePages(), std::vector<PageId>{2});
    EXPECT_EQ(manager.records().lruToMru(), std::vector<RecordId>{secondBranch});
    EXPECT_EQ(manager.baseIndex().lookup(kHASH_A), std::optional<PageId>{0});
    EXPECT_EQ(manager.baseIndex().lookup(kHASH_B), std::optional<PageId>{1});
    EXPECT_FALSE(manager.baseIndex().lookup(kHASH_C).has_value());
    EXPECT_EQ(manager.baseIndex().lookup(kHASH_D), std::optional<PageId>{3});
    expectOnlyBaseState(manager, {0, 0, 1, 0}, {1, 1, 0, 1}, 0);

    PublishStatus const replacementStatus
        = manager.publish(replacementLease, PublishRequest{{kHASH_E}, kPAGE_SIZE}).status;
    ASSERT_EQ(replacementStatus, PublishStatus::kPublished);
    RecordId const replacement = manager.records().lruToMru().back();
    replacementLease.release();

    EXPECT_EQ(manager.records().lruToMru(), std::vector<RecordId>({secondBranch, replacement}));
    EXPECT_EQ(manager.records().get(secondBranch).basePagePath, std::vector<PageId>({0, 1, 3}));
    EXPECT_EQ(manager.baseIndex().lookup(kHASH_E), std::optional<PageId>{2});
    expectOnlyBaseState(manager, {0, 0, 0, 0}, {1, 1, 1, 1}, 0);
}

TEST(ContextCacheManagerTests, BlockHitDoesNotTouchOwningRecordRecency)
{
    ContextCacheManager manager(kPAGE_SIZE, ResourceDemand{3, 0, 0, 0}, 3);
    PublishedRecord const first = publishPrivateRecord(manager, {kHASH_C});
    PublishedRecord const second = publishPrivateRecord(manager, {kHASH_A});
    ASSERT_EQ(first.status, PublishStatus::kPublished);
    ASSERT_EQ(second.status, PublishStatus::kPublished);
    std::vector<RecordId> const originalLru{first.id, second.id};
    ASSERT_EQ(manager.records().lruToMru(), originalLru);

    AcquireResult result = manager.acquireVanilla({kHASH_A}, kPAGE_SIZE + 1);
    ASSERT_TRUE(result.lease.has_value());
    CacheRequestLease lease = takeLease(result);

    EXPECT_EQ(lease.basePages(), std::vector<PageId>({1, 2}));
    EXPECT_EQ(manager.records().lruToMru(), originalLru);
    expectOnlyBaseState(manager, {0, 1, 1}, {1, 1, 0}, 0);

    PublishRequest const mismatchedPrefix{{kHASH_C}, kPAGE_SIZE};
    EXPECT_THROW((void) manager.publish(lease, mismatchedPrefix), std::runtime_error);

    EXPECT_TRUE(lease.valid());
    EXPECT_EQ(lease.basePages(), std::vector<PageId>({1, 2}));
    EXPECT_EQ(manager.records().lruToMru(), originalLru);
    EXPECT_EQ(manager.records().get(first.id).basePagePath, std::vector<PageId>{0});
    EXPECT_EQ(manager.records().get(second.id).basePagePath, std::vector<PageId>{1});
    EXPECT_EQ(manager.baseIndex().lookup(kHASH_C), std::optional<PageId>{0});
    EXPECT_EQ(manager.baseIndex().lookup(kHASH_A), std::optional<PageId>{1});
    EXPECT_EQ(manager.baseIndex().size(), 2U);
    expectOnlyBaseState(manager, {0, 1, 1}, {1, 1, 0}, 0);

    lease.release();

    EXPECT_EQ(manager.records().lruToMru(), originalLru);
    EXPECT_EQ(manager.baseIndex().lookup(kHASH_A), std::optional<PageId>{1});
    expectOnlyBaseState(manager, {0, 0, 0}, {1, 1, 0}, 1);
}

TEST(ContextCacheManagerTests, ExactDuplicatePublicationMovesExistingRecordToMru)
{
    ContextCacheManager manager(kPAGE_SIZE, ResourceDemand{3, 0, 0, 0}, 3);
    PublishedRecord const first = publishPrivateRecord(manager, {kHASH_A});
    PublishedRecord const second = publishPrivateRecord(manager, {kHASH_B});
    ASSERT_EQ(first.status, PublishStatus::kPublished);
    ASSERT_EQ(second.status, PublishStatus::kPublished);
    ASSERT_EQ(manager.records().lruToMru(), std::vector<RecordId>({first.id, second.id}));

    AcquireResult duplicateResult = manager.acquireVanilla({kHASH_A}, kPAGE_SIZE);
    ReusePlan const& duplicatePlan = duplicateResult.plan;
    ASSERT_EQ(duplicatePlan.kind, ReusePlanKind::kFullInputRewind);
    ASSERT_TRUE(duplicateResult.lease.has_value());
    EXPECT_EQ(duplicateResult.status, AcquireStatus::kAcquired);
    CacheRequestLease duplicateLease = takeLease(duplicateResult);
    ASSERT_EQ(duplicateLease.basePages(), std::vector<PageId>{2});

    PublishStatus const status = manager.publish(duplicateLease, PublishRequest{{kHASH_A}, kPAGE_SIZE}).status;

    EXPECT_EQ(status, PublishStatus::kExistingRecord);
    EXPECT_EQ(manager.records().lruToMru(), std::vector<RecordId>({second.id, first.id}));
    EXPECT_EQ(manager.records().size(), 2U);
    EXPECT_EQ(manager.records().get(first.id).basePagePath, std::vector<PageId>{0});
    EXPECT_EQ(manager.baseIndex().lookup(kHASH_A), std::optional<PageId>{0});
    EXPECT_EQ(manager.baseIndex().size(), 2U);
    expectOnlyBaseState(manager, {0, 0, 1}, {1, 1, 0}, 0);

    duplicateLease.release();

    expectOnlyBaseState(manager, {0, 0, 0}, {1, 1, 0}, 1);
}

TEST(ContextCacheManagerTests, ExistingRecordPublicationDoesNotRebindActiveLease)
{
    ContextCacheManager manager(kPAGE_SIZE, ResourceDemand{2, 0, 0, 0}, 2);
    PublishedRecord const cached = publishPrivateRecord(manager, {kHASH_A});
    ASSERT_EQ(cached.producerPages, std::vector<PageId>{0});

    AcquireResult acquired = manager.acquireVanilla({kHASH_A}, kPAGE_SIZE);
    CacheRequestLease lease = takeLease(acquired);
    ASSERT_EQ(lease.basePages(), std::vector<PageId>{1});
    expectOnlyBaseState(manager, {0, 1}, {1, 0}, 0);

    PublishResult const published = manager.publish(lease, PublishRequest{{kHASH_A}, kPAGE_SIZE});
    ASSERT_EQ(published.status, PublishStatus::kExistingRecord);
    ASSERT_EQ(published.canonicalBasePages, std::vector<PageId>{0});

    EXPECT_EQ(lease.basePages(), std::vector<PageId>{1});
    expectOnlyBaseState(manager, {0, 1}, {1, 0}, 0);

    lease.release();

    expectOnlyBaseState(manager, {0, 0}, {1, 0}, 1);
}

TEST(ContextCacheManagerTests, DynamicGrowthAtomicallyAddsRequestedPages)
{
    ContextCacheManager manager(kPAGE_SIZE, ResourceDemand{3, 0, 0, 0}, 1);
    ContextCacheManager foreignManager(kPAGE_SIZE, ResourceDemand{1, 0, 0, 0}, 1);
    CacheRequestLease lease = acquirePrivatePages(manager, 1);
    ASSERT_EQ(lease.basePages(), std::vector<PageId>{0});

    EXPECT_THROW((void) foreignManager.growBasePages(lease, 0), std::runtime_error);
    EXPECT_THROW((void) manager.growBasePages(lease, -1), std::runtime_error);
    EXPECT_TRUE(manager.growBasePages(lease, 0));
    EXPECT_EQ(lease.basePages(), std::vector<PageId>{0});
    expectOnlyBaseState(manager, {1, 0, 0}, {0, 0, 0}, 2);

    bool hugeGrowthSucceeded{};
    EXPECT_NO_THROW(hugeGrowthSucceeded = manager.growBasePages(lease, std::numeric_limits<int32_t>::max()));
    EXPECT_FALSE(hugeGrowthSucceeded);
    EXPECT_EQ(lease.basePages(), std::vector<PageId>{0});
    expectOnlyBaseState(manager, {1, 0, 0}, {0, 0, 0}, 2);

    EXPECT_TRUE(manager.growBasePages(lease, 2));
    EXPECT_EQ(lease.basePages(), std::vector<PageId>({0, 1, 2}));
    expectOnlyBaseState(manager, {1, 1, 1}, {0, 0, 0}, 0);

    std::vector<PageId> const beforeFailedGrowth = lease.basePages();
    EXPECT_FALSE(manager.growBasePages(lease, 1));
    EXPECT_EQ(lease.basePages(), beforeFailedGrowth);
    expectOnlyBaseState(manager, {1, 1, 1}, {0, 0, 0}, 0);

    lease.release();

    expectOnlyBaseState(manager, {0, 0, 0}, {0, 0, 0}, 3);
    expectOnlyBaseState(foreignManager, {0}, {0}, 1);
}

TEST(ContextCacheManagerTests, RecordLimitEvictsTheLruRecord)
{
    ContextCacheManager manager(kPAGE_SIZE, ResourceDemand{3, 0, 0, 0}, 2);
    PublishedRecord const first = publishPrivateRecord(manager, {kHASH_A});
    PublishedRecord const second = publishPrivateRecord(manager, {kHASH_B});
    ASSERT_EQ(first.status, PublishStatus::kPublished);
    ASSERT_EQ(second.status, PublishStatus::kPublished);
    ASSERT_EQ(manager.records().lruToMru(), std::vector<RecordId>({first.id, second.id}));

    CacheRequestLease thirdLease = acquirePrivatePages(manager, 1);
    ASSERT_EQ(thirdLease.basePages(), std::vector<PageId>{2});
    PublishStatus const status = manager.publish(thirdLease, PublishRequest{{kHASH_C}, kPAGE_SIZE}).status;

    EXPECT_EQ(status, PublishStatus::kPublished);
    ASSERT_EQ(manager.records().size(), 2U);
    std::vector<RecordId> const lru = manager.records().lruToMru();
    ASSERT_EQ(lru.size(), 2U);
    EXPECT_EQ(lru.front(), second.id);
    EXPECT_NE(lru.back(), first.id);
    EXPECT_NE(lru.back(), second.id);
    EXPECT_FALSE(manager.records().find(CacheRecordKey{kHASH_A, 1}).has_value());
    EXPECT_EQ(manager.records().find(CacheRecordKey{kHASH_B, 1}), std::optional<RecordId>{second.id});
    EXPECT_FALSE(manager.baseIndex().lookup(kHASH_A).has_value());
    EXPECT_EQ(manager.baseIndex().lookup(kHASH_B), std::optional<PageId>{1});
    EXPECT_EQ(manager.baseIndex().lookup(kHASH_C), std::optional<PageId>{2});
    expectOnlyBaseState(manager, {0, 0, 1}, {0, 1, 1}, 1);

    thirdLease.release();

    expectOnlyBaseState(manager, {0, 0, 0}, {0, 1, 1}, 1);

    ContextCacheManager zeroLimit(kPAGE_SIZE, ResourceDemand{1, 0, 0, 0}, 0);
    CacheRequestLease transient = acquirePrivatePages(zeroLimit, 1);
    PublishStatus const zeroStatus = zeroLimit.publish(transient, PublishRequest{{kHASH_D}, kPAGE_SIZE}).status;
    EXPECT_EQ(zeroStatus, PublishStatus::kPublished);
    EXPECT_TRUE(zeroLimit.records().lruToMru().empty());
    EXPECT_EQ(zeroLimit.baseIndex().size(), 0U);
    expectOnlyBaseState(zeroLimit, {1}, {0}, 0);

    transient.release();

    expectOnlyBaseState(zeroLimit, {0}, {0}, 1);
}

TEST(ContextCacheManagerTests, SpecAcquirePublishesPairedStateAndReplaysOneFullPage)
{
    ContextCacheManager manager(kPAGE_SIZE, ResourceDemand{4, 4, 0, 0}, 2, eagleSpecReuseContract());
    PublishedSpecRecord const published = publishPrivateSpecRecord(manager, {kHASH_A, kHASH_B});
    ASSERT_EQ(published.status, PublishStatus::kPublished);
    ASSERT_EQ(published.producerBasePages, std::vector<PageId>({0, 1}));
    ASSERT_EQ(published.producerDraftPages, std::vector<PageId>({0, 1}));

    CacheRecord const& record = manager.records().get(published.id);
    EXPECT_EQ(record.basePagePath, std::vector<PageId>({0, 1}));
    EXPECT_EQ(eaglePagePath(record), std::vector<PageId>({0, 1}));
    EXPECT_EQ(manager.baseIndex().lookup(kHASH_A), std::optional<PageId>{0});
    EXPECT_EQ(manager.baseIndex().lookup(kHASH_B), std::optional<PageId>{1});
    EXPECT_EQ(manager.specIndex().paged().lookupLongest({kHASH_A, kHASH_B}, 2),
        (std::optional<SpecPagedStateMatch>{SpecPagedStateMatch{published.id, 2}}));
    expectPoolState(manager, ResourceType::kBaseKvPage, {0, 0, 0, 0}, {1, 1, 0, 0}, 2);
    expectPoolState(manager, ResourceType::kDraftKvPage, {0, 0, 0, 0}, {1, 1, 0, 0}, 2);

    AcquireResult result = manager.acquireSpec({kHASH_A, kHASH_B}, 2 * kPAGE_SIZE + 1);
    ReusePlan const& spec = result.plan;
    EXPECT_EQ(spec.basePageBindings, std::vector<PageId>{0});
    EXPECT_EQ(spec.specPageBindings, std::vector<PageId>{0});
    EXPECT_EQ(spec.specReplayMode, SpecReplayMode::kFullPage);
    ASSERT_EQ(result.status, AcquireStatus::kAcquired);
    CacheRequestLease lease = takeLease(result);

    EXPECT_EQ(lease.basePages(), std::vector<PageId>({0, 2, 3}));
    EXPECT_EQ(lease.draftPages(), std::vector<PageId>({0, 2, 3}));
    EXPECT_EQ(spec.reuseTokenLength, kPAGE_SIZE);
    expectPoolState(manager, ResourceType::kBaseKvPage, {1, 0, 1, 1}, {1, 1, 0, 0}, 0);
    expectPoolState(manager, ResourceType::kDraftKvPage, {1, 0, 1, 1}, {1, 1, 0, 0}, 0);

    lease.release();
    expectPoolState(manager, ResourceType::kBaseKvPage, {0, 0, 0, 0}, {1, 1, 0, 0}, 2);
    expectPoolState(manager, ResourceType::kDraftKvPage, {0, 0, 0, 0}, {1, 1, 0, 0}, 2);
}

TEST(ContextCacheManagerTests, BaseOnlyHitAllocatesColdPairedState)
{
    ContextCacheManager manager(kPAGE_SIZE, ResourceDemand{5, 3, 0, 0}, 3, eagleSpecReuseContract());
    PublishedRecord const baseOnly = publishPrivateRecord(manager, {kHASH_A, kHASH_B});
    ASSERT_EQ(baseOnly.status, PublishStatus::kPublished);

    AcquireResult result = manager.acquireSpec({kHASH_A, kHASH_B}, 2 * kPAGE_SIZE + 1);
    ASSERT_EQ(result.status, AcquireStatus::kAcquired);
    EXPECT_EQ(result.plan.reuseTokenLength, 0);
    CacheRequestLease lease = takeLease(result);
    EXPECT_EQ(lease.basePages().size(), 3U);
    EXPECT_EQ(lease.draftPages().size(), 3U);
}

TEST(ContextCacheManagerTests, VanillaPublicationPreservesPairedRecordForLaterSpecReuse)
{
    ContextCacheManager manager(kPAGE_SIZE, ResourceDemand{4, 4, 0, 0}, 2, eagleSpecReuseContract());
    PublishedSpecRecord const published = publishPrivateSpecRecord(manager, {kHASH_A, kHASH_B});
    ASSERT_EQ(published.status, PublishStatus::kPublished);

    AcquireResult vanillaResult = manager.acquireVanilla({kHASH_A, kHASH_B}, 2 * kPAGE_SIZE + 1);
    ASSERT_EQ(vanillaResult.status, AcquireStatus::kAcquired);
    CacheRequestLease vanillaLease = takeLease(vanillaResult);
    ASSERT_TRUE(vanillaLease.draftPages().empty());
    EXPECT_EQ(manager.publish(vanillaLease, PublishRequest{{kHASH_A, kHASH_B}, 2 * kPAGE_SIZE}).status,
        PublishStatus::kExistingRecord);
    vanillaLease.release();

    ASSERT_EQ(manager.records().size(), 1U);
    CacheRecord const& retained = manager.records().get(published.id);
    EXPECT_EQ(eaglePagePath(retained), published.producerDraftPages);
    EXPECT_EQ(manager.specIndex().paged().lookupLongest({kHASH_A, kHASH_B}, 2),
        (std::optional<SpecPagedStateMatch>{SpecPagedStateMatch{published.id, 2}}));

    AcquireResult specResult = manager.acquireSpec({kHASH_A, kHASH_B}, 2 * kPAGE_SIZE + 1);
    ReusePlan const& specPlan = specResult.plan;
    EXPECT_EQ(specPlan.specRecord, std::optional<RecordId>{published.id});
    EXPECT_EQ(specPlan.specPageBindings, std::vector<PageId>{published.producerDraftPages.front()});
    takeLease(specResult).release();
}

TEST(ContextCacheManagerTests, SpecDraftUpgradeUsesCanonicalProjectedBasePath)
{
    ContextCacheManager manager(kPAGE_SIZE, ResourceDemand{6, 4, 0, 0}, 2, eagleSpecReuseContract());
    AcquireResult baseResult = manager.acquireVanilla({kHASH_A, kHASH_B}, 2 * kPAGE_SIZE);
    CacheRequestLease baseLease = takeLease(baseResult);
    ASSERT_EQ(baseLease.basePages(), std::vector<PageId>({0, 1}));
    ASSERT_EQ(manager.publish(baseLease, PublishRequest{{kHASH_A, kHASH_B}, 2 * kPAGE_SIZE}).status,
        PublishStatus::kPublished);
    baseLease.release();

    RecordId const recordId = manager.records().lruToMru().front();
    ASSERT_FALSE(hasSpecState(manager.records().get(recordId)));

    AcquireResult specResult = manager.acquireSpec({kHASH_A, kHASH_B}, 2 * kPAGE_SIZE);
    ASSERT_EQ(specResult.plan.kind, ReusePlanKind::kNoReusablePrefix);
    CacheRequestLease specLease = takeLease(specResult);
    ASSERT_EQ(specLease.basePages(), std::vector<PageId>({2, 3}));
    ASSERT_EQ(specLease.draftPages(), std::vector<PageId>({0, 1}));

    PublishResult const upgrade = manager.publish(specLease, PublishRequest{{kHASH_A, kHASH_B}, 2 * kPAGE_SIZE});
    EXPECT_EQ(upgrade.status, PublishStatus::kPublished);
    EXPECT_EQ(upgrade.record, std::optional<RecordId>{recordId});
    EXPECT_EQ(upgrade.canonicalBasePages, std::vector<PageId>({0, 1}));
    EXPECT_EQ(specLease.basePages(), std::vector<PageId>({2, 3}));
    specLease.release();

    ASSERT_EQ(manager.records().size(), 1U);
    EXPECT_EQ(manager.records().lruToMru(), std::vector<RecordId>{recordId});
    EXPECT_EQ(manager.records().get(recordId).basePagePath, std::vector<PageId>({0, 1}));
    EXPECT_EQ(eaglePagePath(manager.records().get(recordId)), std::vector<PageId>({0, 1}));
    EXPECT_EQ(manager.specIndex().paged().lookupLongest({kHASH_A, kHASH_B}, 2),
        (std::optional<SpecPagedStateMatch>{SpecPagedStateMatch{recordId, 2}}));
    expectPoolState(manager, ResourceType::kBaseKvPage, {0, 0, 0, 0, 0, 0}, {1, 1, 0, 0, 0, 0}, 4);
    expectPoolState(manager, ResourceType::kDraftKvPage, {0, 0, 0, 0}, {1, 1, 0, 0}, 2);
}

TEST(ContextCacheManagerTests, EvictingUpgradedSpecRecordReleasesDraftIndexAndReferences)
{
    ContextCacheManager manager(kPAGE_SIZE, ResourceDemand{4, 4, 0, 0}, 1, eagleSpecReuseContract());
    PublishedRecord const baseOnly = publishPrivateRecord(manager, {kHASH_A, kHASH_B});
    ASSERT_FALSE(hasSpecState(manager.records().get(baseOnly.id)));

    AcquireResult upgradeResult = manager.acquireSpec({kHASH_A, kHASH_B}, 2 * kPAGE_SIZE);
    CacheRequestLease upgradeLease = takeLease(upgradeResult);
    std::vector<PageId> const upgradedDraftPages = upgradeLease.draftPages();
    ASSERT_EQ(manager.publish(upgradeLease, PublishRequest{{kHASH_A, kHASH_B}, 2 * kPAGE_SIZE}).status,
        PublishStatus::kPublished);
    upgradeLease.release();

    ASSERT_EQ(eaglePagePath(manager.records().get(baseOnly.id)), upgradedDraftPages);
    ASSERT_EQ(manager.specIndex().paged().lookupLongest({kHASH_A, kHASH_B}, 2),
        (std::optional<SpecPagedStateMatch>{SpecPagedStateMatch{baseOnly.id, 2}}));
    for (PageId const page : upgradedDraftPages)
    {
        ASSERT_EQ(manager.pools().cacheRefCount(ResourceId{ResourceType::kDraftKvPage, page}), 1);
    }

    PublishedSpecRecord const replacement = publishPrivateSpecRecord(manager, {kHASH_C});
    ASSERT_EQ(replacement.status, PublishStatus::kPublished);
    EXPECT_FALSE(manager.records().contains(baseOnly.id));
    EXPECT_FALSE(manager.specIndex().paged().lookupLongest({kHASH_A, kHASH_B}, 2).has_value());
    EXPECT_EQ(manager.specIndex().paged().lookupLongest({kHASH_C}, 1),
        (std::optional<SpecPagedStateMatch>{SpecPagedStateMatch{replacement.id, 1}}));
    for (PageId const page : upgradedDraftPages)
    {
        EXPECT_EQ(manager.pools().activeRefCount(ResourceId{ResourceType::kDraftKvPage, page}), 0);
        EXPECT_EQ(manager.pools().cacheRefCount(ResourceId{ResourceType::kDraftKvPage, page}), 0);
    }
    EXPECT_EQ(manager.pools().freeCount(ResourceType::kDraftKvPage), 3);
}

TEST(ContextCacheManagerTests, SpecPublicationUsesCommonMaterializedBoundary)
{
    ContextCacheManager manager(kPAGE_SIZE, ResourceDemand{4, 4, 0, 0}, 2, eagleSpecReuseContract());
    AcquireResult result = manager.acquireSpec({kHASH_A, kHASH_B}, 2 * kPAGE_SIZE);
    CacheRequestLease lease = takeLease(result);

    PublishResult const published = manager.publish(lease, PublishRequest{{kHASH_A, kHASH_B}, kPAGE_SIZE});
    EXPECT_EQ(published.status, PublishStatus::kPublished);
    EXPECT_EQ(published.publishedBaseFullBlockCount, 1);
    lease.release();

    ASSERT_EQ(manager.records().size(), 1U);
    RecordId const recordId = manager.records().lruToMru().front();
    CacheRecord const& record = manager.records().get(recordId);
    EXPECT_EQ(record.basePagePath, std::vector<PageId>{0});
    EXPECT_EQ(eaglePagePath(record), std::vector<PageId>{0});
    EXPECT_EQ(manager.specIndex().paged().lookupLongest({kHASH_A, kHASH_B}, 2),
        (std::optional<SpecPagedStateMatch>{SpecPagedStateMatch{recordId, 1}}));
    expectPoolState(manager, ResourceType::kBaseKvPage, {0, 0, 0, 0}, {1, 0, 0, 0}, 3);
    expectPoolState(manager, ResourceType::kDraftKvPage, {0, 0, 0, 0}, {1, 0, 0, 0}, 3);
}

TEST(ContextCacheManagerTests, SpecFullPageReplayPublishesProjectedBaseWithoutRebinding)
{
    ContextCacheManager manager(kPAGE_SIZE, ResourceDemand{6, 6, 0, 0}, 2, eagleSpecReuseContract());
    PublishedSpecRecord const prefix = publishPrivateSpecRecord(manager, {kHASH_A, kHASH_B});

    AcquireResult result = manager.acquireSpec({kHASH_A, kHASH_B, kHASH_C}, 3 * kPAGE_SIZE);
    ReusePlan const& plan = result.plan;
    ASSERT_EQ(plan.specReplayMode, SpecReplayMode::kFullPage);
    ASSERT_EQ(plan.reuseTokenLength, kPAGE_SIZE);
    CacheRequestLease lease = takeLease(result);
    ASSERT_EQ(lease.basePages(), std::vector<PageId>({0, 2, 3}));
    ASSERT_EQ(lease.draftPages(), std::vector<PageId>({0, 2, 3}));

    PublishResult const published = manager.publish(lease, PublishRequest{{kHASH_A, kHASH_B, kHASH_C}, 3 * kPAGE_SIZE});
    ASSERT_EQ(published.status, PublishStatus::kPublished);
    ASSERT_TRUE(published.record.has_value());
    EXPECT_EQ(published.publishedBaseFullBlockCount, 3);
    EXPECT_EQ(published.canonicalBasePages, std::vector<PageId>({0, 1, 3}));
    EXPECT_EQ(lease.basePages(), std::vector<PageId>({0, 2, 3}));
    EXPECT_EQ(lease.draftPages(), std::vector<PageId>({0, 2, 3}));
    lease.release();

    EXPECT_EQ(manager.records().lruToMru(), std::vector<RecordId>({prefix.id, *published.record}));
    EXPECT_EQ(manager.records().get(*published.record).basePagePath, std::vector<PageId>({0, 1, 3}));
    EXPECT_EQ(eaglePagePath(manager.records().get(*published.record)), std::vector<PageId>({0, 2, 3}));
    EXPECT_EQ(manager.baseIndex().lookup(kHASH_C), std::optional<PageId>{3});
    EXPECT_EQ(manager.specIndex().paged().lookupLongest({kHASH_A, kHASH_B, kHASH_C}, 3),
        (std::optional<SpecPagedStateMatch>{SpecPagedStateMatch{*published.record, 3}}));
    EXPECT_EQ(eaglePagePath(manager.records().get(prefix.id)), std::vector<PageId>({0, 1}));
    expectPoolState(manager, ResourceType::kBaseKvPage, {0, 0, 0, 0, 0, 0}, {2, 2, 0, 1, 0, 0}, 3);
    expectPoolState(manager, ResourceType::kDraftKvPage, {0, 0, 0, 0, 0, 0}, {2, 1, 1, 1, 0, 0}, 2);
}

TEST(ContextCacheManagerTests, SpecFullPageReplayRejectsPublicationBeforeDependencyBoundary)
{
    ContextCacheManager manager(kPAGE_SIZE, ResourceDemand{6, 6, 0, 0}, 2, eagleSpecReuseContract());
    PublishedSpecRecord const prefix = publishPrivateSpecRecord(manager, {kHASH_A, kHASH_B});

    AcquireResult result = manager.acquireSpec({kHASH_A, kHASH_B, kHASH_C}, 3 * kPAGE_SIZE);
    ASSERT_EQ(result.plan.specReplayMode, SpecReplayMode::kFullPage);
    ASSERT_TRUE(result.plan.specReplayDependency.has_value());
    ASSERT_EQ(result.plan.specReplayDependency->pathBlockCount, 2);
    CacheRequestLease lease = takeLease(result);

    try
    {
        static_cast<void>(manager.publish(lease, PublishRequest{{kHASH_A}, kPAGE_SIZE}));
        FAIL() << "Publication must cover the complete replay dependency boundary";
    }
    catch (std::runtime_error const& error)
    {
        EXPECT_NE(std::string{error.what()}.find("does not match its full-page replay dependency"), std::string::npos)
            << error.what();
    }
    EXPECT_EQ(manager.records().lruToMru(), std::vector<RecordId>{prefix.id});
}

TEST(ContextCacheManagerTests, SpecFullPageReplayRejectsChangedBoundaryHash)
{
    ContextCacheManager manager(kPAGE_SIZE, ResourceDemand{6, 6, 0, 0}, 2, eagleSpecReuseContract());
    PublishedSpecRecord const prefix = publishPrivateSpecRecord(manager, {kHASH_A, kHASH_B});

    AcquireResult result = manager.acquireSpec({kHASH_A, kHASH_B, kHASH_C}, 3 * kPAGE_SIZE);
    ASSERT_EQ(result.plan.specReplayMode, SpecReplayMode::kFullPage);
    ASSERT_TRUE(result.plan.specReplayDependency.has_value());
    CacheRequestLease lease = takeLease(result);

    EXPECT_THROW(static_cast<void>(manager.publish(lease, PublishRequest{{kHASH_A, kHASH_D, kHASH_E}, 3 * kPAGE_SIZE})),
        std::runtime_error);
    EXPECT_EQ(manager.records().lruToMru(), std::vector<RecordId>{prefix.id});
    EXPECT_FALSE(manager.baseIndex().lookup(kHASH_D).has_value());
    EXPECT_FALSE(manager.baseIndex().lookup(kHASH_E).has_value());
}

TEST(ContextCacheManagerTests, SpecGrowthIsAtomicAcrossBaseAndDraftPools)
{
    ContextCacheManager manager(kPAGE_SIZE, ResourceDemand{3, 2, 0, 0}, 1, eagleSpecReuseContract());
    AcquireResult result = manager.acquireSpec({}, 1);
    CacheRequestLease lease = takeLease(result);
    ASSERT_EQ(lease.basePages(), std::vector<PageId>{0});
    ASSERT_EQ(lease.draftPages(), std::vector<PageId>{0});

    EXPECT_THROW((void) manager.growSpecPages(lease, -1, 0), std::runtime_error);
    EXPECT_TRUE(manager.growSpecPages(lease, 1, 1));
    EXPECT_EQ(lease.basePages(), std::vector<PageId>({0, 1}));
    EXPECT_EQ(lease.draftPages(), std::vector<PageId>({0, 1}));
    EXPECT_FALSE(manager.growSpecPages(lease, 1, 1));
    EXPECT_EQ(lease.basePages(), std::vector<PageId>({0, 1}));
    EXPECT_EQ(lease.draftPages(), std::vector<PageId>({0, 1}));
    expectPoolState(manager, ResourceType::kBaseKvPage, {1, 1, 0}, {0, 0, 0}, 1);
    expectPoolState(manager, ResourceType::kDraftKvPage, {1, 1}, {0, 0}, 0);

    ContextCacheManager vanillaManager(kPAGE_SIZE, ResourceDemand{1, 1, 0, 0}, 1);
    CacheRequestLease vanillaLease = acquirePrivatePages(vanillaManager, 1);
    EXPECT_THROW((void) vanillaManager.growSpecPages(vanillaLease, 0, 0), std::runtime_error);
    EXPECT_THROW((void) manager.growBasePages(lease, 0), std::runtime_error);
}

TEST(ContextCacheManagerTests, SpecHitIsTreatedAsMruDuringPressureEviction)
{
    ContextCacheManager manager(kPAGE_SIZE, ResourceDemand{4, 4, 0, 0}, 2, eagleSpecReuseContract());
    PublishedSpecRecord const first = publishPrivateSpecRecord(manager, {kHASH_A, kHASH_B});
    PublishedSpecRecord const second = publishPrivateSpecRecord(manager, {kHASH_C});
    ASSERT_EQ(manager.records().lruToMru(), std::vector<RecordId>({first.id, second.id}));

    AcquireResult result = manager.acquireSpec({kHASH_A, kHASH_B}, 2 * kPAGE_SIZE + 1);
    ReusePlan const& hit = result.plan;
    ASSERT_EQ(hit.specRecord, std::optional<RecordId>{first.id});
    ASSERT_EQ(hit.demand.baseKvPages, 2);
    ASSERT_EQ(hit.demand.draftKvPages, 2);

    ASSERT_EQ(result.status, AcquireStatus::kAcquired);
    CacheRequestLease lease = takeLease(result);

    EXPECT_EQ(manager.records().lruToMru(), std::vector<RecordId>{first.id});
    EXPECT_TRUE(manager.records().contains(first.id));
    EXPECT_FALSE(manager.records().contains(second.id));
    EXPECT_EQ(manager.specIndex().paged().lookupLongest({kHASH_A, kHASH_B}, 2),
        (std::optional<SpecPagedStateMatch>{SpecPagedStateMatch{first.id, 2}}));
    EXPECT_FALSE(manager.specIndex().paged().lookupLongest({kHASH_C}, 1).has_value());

    lease.release();
    expectPoolState(manager, ResourceType::kBaseKvPage, {0, 0, 0, 0}, {1, 1, 0, 0}, 2);
    expectPoolState(manager, ResourceType::kDraftKvPage, {0, 0, 0, 0}, {1, 1, 0, 0}, 2);
}

TEST(ContextCacheManagerTests, InfeasibleSpecAcquirePreservesLruAndTypedReferences)
{
    ContextCacheManager manager(kPAGE_SIZE, ResourceDemand{3, 3, 0, 0}, 2, eagleSpecReuseContract());
    PublishedSpecRecord const first = publishPrivateSpecRecord(manager, {kHASH_A, kHASH_B});
    PublishedSpecRecord const second = publishPrivateSpecRecord(manager, {kHASH_C});
    std::vector<RecordId> const lruBefore{first.id, second.id};
    ASSERT_EQ(manager.records().lruToMru(), lruBefore);

    AcquireResult result = manager.acquireSpec({kHASH_A, kHASH_B, kHASH_C}, 3 * kPAGE_SIZE + 1);
    ReusePlan const& hit = result.plan;
    ASSERT_EQ(hit.specRecord, std::optional<RecordId>{first.id});
    ASSERT_EQ(hit.demand.baseKvPages, 3);
    ASSERT_EQ(hit.demand.draftKvPages, 3);

    EXPECT_EQ(result.status, AcquireStatus::kInsufficientCapacity);
    EXPECT_FALSE(result.lease.has_value());
    EXPECT_EQ(manager.records().lruToMru(), lruBefore);
    EXPECT_EQ(manager.specIndex().paged().lookupLongest({kHASH_A, kHASH_B}, 2),
        (std::optional<SpecPagedStateMatch>{SpecPagedStateMatch{first.id, 2}}));
    EXPECT_EQ(manager.specIndex().paged().lookupLongest({kHASH_C}, 1),
        (std::optional<SpecPagedStateMatch>{SpecPagedStateMatch{second.id, 1}}));
    expectPoolState(manager, ResourceType::kBaseKvPage, {0, 0, 0}, {1, 1, 1}, 0);
    expectPoolState(manager, ResourceType::kDraftKvPage, {0, 0, 0}, {1, 1, 1}, 0);
}

TEST(ContextCacheManagerTests, SpecEvictionRemovesBothIndices)
{
    ContextCacheManager manager(kPAGE_SIZE, ResourceDemand{3, 3, 0, 0}, 1, eagleSpecReuseContract());
    PublishedSpecRecord const first = publishPrivateSpecRecord(manager, {kHASH_A, kHASH_B});

    PublishedSpecRecord const second = publishPrivateSpecRecord(manager, {kHASH_C});
    ASSERT_EQ(second.status, PublishStatus::kPublished);
    ASSERT_NE(second.id, first.id);

    EXPECT_FALSE(manager.baseIndex().lookup(kHASH_A).has_value());
    EXPECT_FALSE(manager.baseIndex().lookup(kHASH_B).has_value());
    EXPECT_FALSE(manager.specIndex().paged().lookupLongest({kHASH_A, kHASH_B}, 2).has_value());
    EXPECT_EQ(manager.baseIndex().lookup(kHASH_C), std::optional<PageId>{2});
    EXPECT_EQ(manager.specIndex().paged().lookupLongest({kHASH_C}, 1),
        (std::optional<SpecPagedStateMatch>{SpecPagedStateMatch{second.id, 1}}));
    expectPoolState(manager, ResourceType::kBaseKvPage, {0, 0, 0}, {0, 0, 1}, 2);
    expectPoolState(manager, ResourceType::kDraftKvPage, {0, 0, 0}, {0, 0, 1}, 2);
}
