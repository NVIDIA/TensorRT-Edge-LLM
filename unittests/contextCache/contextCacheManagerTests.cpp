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

#include "runtime/config/llmEngineConfig.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
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
constexpr CacheDomainId kDOMAIN{0x1010101010101010ULL, 0x2020202020202020ULL};
constexpr DraftEngineSignature kDRAFT_SIGNATURE{0x3030303030303030ULL, 0x4040404040404040ULL};
constexpr DraftEngineSignature kOTHER_DRAFT_SIGNATURE{0x5050505050505050ULL, 0x6060606060606060ULL};
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
    ReusePlan const plan = manager.planVanilla(kDOMAIN, {}, count * kPAGE_SIZE);
    AcquireResult result = manager.acquire(plan);
    return takeLease(result);
}

PublishedRecord publishPrivateRecord(ContextCacheManager& manager, std::vector<BlockHash> hashes)
{
    CacheRequestLease lease = acquirePrivatePages(manager, static_cast<int32_t>(hashes.size()));
    std::vector<PageId> const producerPages = lease.basePages();
    PublishStatus const status = manager.publish(lease,
        PublishRequest{hashes, static_cast<int32_t>(hashes.size()) * kPAGE_SIZE, PublicationPoint::kPrefillEnd,
            CommitPolicy::kIncludingGeneratedTokens});
    std::vector<RecordId> const lru = manager.records().lruToMru();
    if (lru.empty())
    {
        throw std::runtime_error("Test context cache publication did not retain a record");
    }
    lease.release();
    return PublishedRecord{status, producerPages, lru.back()};
}

PublishedSpecRecord publishPrivateSpecRecord(
    ContextCacheManager& manager, DraftEngineSignature signature, std::vector<BlockHash> hashes)
{
    ReusePlan const plan = manager.planSpec(
        SpecDecodeMode::kEAGLE, kDOMAIN, signature, hashes, static_cast<int32_t>(hashes.size()) * kPAGE_SIZE, true);
    AcquireResult result = manager.acquire(plan);
    CacheRequestLease lease = takeLease(result);
    std::vector<PageId> const producerBasePages = lease.basePages();
    std::vector<PageId> const producerDraftPages = lease.draftPages();
    PublishStatus const status = manager.publish(lease,
        PublishRequest{hashes, static_cast<int32_t>(hashes.size()) * kPAGE_SIZE, PublicationPoint::kPrefillEnd,
            CommitPolicy::kIncludingGeneratedTokens, static_cast<int32_t>(hashes.size()) * kPAGE_SIZE});
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

    ReusePlan const plan = manager.planVanilla(kDOMAIN, {kHASH_A}, kPAGE_SIZE + 1);
    ASSERT_EQ(plan.basePageBindings, std::vector<PageId>{0});
    ASSERT_EQ(plan.demand.baseKvPages, 1);
    AcquireResult result = manager.acquire(plan);

    ASSERT_TRUE(result.lease.has_value());
    EXPECT_EQ(result.status, AcquireStatus::kAcquired);
    CacheRequestLease lease = takeLease(result);
    EXPECT_TRUE(lease.valid());
    EXPECT_EQ(lease.reuseTokenLength(), kPAGE_SIZE);
    EXPECT_EQ(lease.basePages(), std::vector<PageId>({0, 1}));
    EXPECT_TRUE(manager.records().lruToMru().empty());
    EXPECT_FALSE(manager.baseIndex().lookup(BaseBlockKey{kDOMAIN, kHASH_A}).has_value());
    EXPECT_FALSE(manager.baseIndex().lookup(BaseBlockKey{kDOMAIN, kHASH_B}).has_value());
    expectOnlyBaseState(manager, {1, 1}, {0, 0}, 0);

    lease.release();

    EXPECT_FALSE(lease.valid());
    EXPECT_TRUE(lease.basePages().empty());
    EXPECT_EQ(lease.reuseTokenLength(), 0);
    expectOnlyBaseState(manager, {0, 0}, {0, 0}, 2);
}

TEST(ContextCacheManagerTests, BypassPlanningBuildsAndAcquiresAForcedColdPlan)
{
    ContextCacheManager manager(kPAGE_SIZE, ResourceDemand{1, 0, 0, 0}, 1);
    PublishedRecord const cached = publishPrivateRecord(manager, {kHASH_A});
    ASSERT_EQ(cached.status, PublishStatus::kPublished);
    ASSERT_EQ(manager.pools().freeCount(ResourceType::kBaseKvPage), 0);

    ReusePlan const hit = manager.planVanilla(kDOMAIN, {kHASH_A}, kPAGE_SIZE, LookupPolicy::kUseCache);
    EXPECT_EQ(hit.kind, ReusePlanKind::kFullInputRewind);

    ReusePlan const cold = manager.planVanilla(kDOMAIN, {kHASH_A}, kPAGE_SIZE, LookupPolicy::kBypass);
    EXPECT_EQ(cold.kind, ReusePlanKind::kNoReusablePrefix);
    EXPECT_EQ(cold.reuseTokenLength, 0);
    EXPECT_TRUE(cold.matchedBlockHashes.empty());
    EXPECT_TRUE(cold.basePageBindings.empty());
    EXPECT_EQ(cold.demand.baseKvPages, 1);

    AcquireResult result = manager.acquire(cold);
    ASSERT_EQ(result.status, AcquireStatus::kAcquired);
    CacheRequestLease lease = takeLease(result);
    EXPECT_EQ(lease.basePages(), std::vector<PageId>{cached.producerPages.front()});
    EXPECT_TRUE(manager.records().lruToMru().empty());
    EXPECT_EQ(manager.baseIndex().size(), 0U);
}

TEST(ContextCacheManagerTests, HybridCheckpointPublishesAndAcquiresAllStateAtomically)
{
    constexpr RecurrentStateSchemaId kSCHEMA{0x7171717171717171ULL, 0x8181818181818181ULL};
    constexpr BlockHash kEXACT_DIGEST{0x9191919191919191ULL, 0xA1A1A1A1A1A1A1A1ULL};
    ContextCacheManager manager(kPAGE_SIZE, ResourceDemand{4, 0, 2, 2}, 2);

    ReusePlan const cold = manager.planHybrid(kDOMAIN, kSCHEMA, {}, {kHASH_A}, /*inputTokenCount=*/6,
        /*hasAttention=*/true);
    EXPECT_EQ(cold.mode, ReusePlanMode::kHybrid);
    EXPECT_EQ(cold.demand.baseKvPages, 2);
    AcquireResult coldResult = manager.acquire(cold);
    CacheRequestLease coldLease = takeLease(coldResult);
    ASSERT_EQ(coldLease.basePages(), std::vector<PageId>({0, 1}));

    std::optional<HybridSnapshotReservation> const snapshots = manager.reserveHybridSnapshots(coldLease, true);
    ASSERT_TRUE(snapshots.has_value());
    EXPECT_EQ(snapshots->recurrentSnapshotSlot, 0);
    EXPECT_EQ(snapshots->partialKvSnapshotSlot, std::optional<int32_t>{0});

    HybridCheckpointKey const checkpoint{kDOMAIN, kEXACT_DIGEST, 6, kSCHEMA};
    PublishResult const published = manager.publishHybridDetailed(coldLease,
        HybridPublishRequest{
            {kHASH_A}, checkpoint, PublicationPoint::kPrefillEnd, CommitPolicy::kIncludingGeneratedTokens, *snapshots});
    ASSERT_EQ(published.status, PublishStatus::kPublished);
    ASSERT_TRUE(published.record.has_value());
    EXPECT_TRUE(published.lineageComplete);
    EXPECT_EQ(published.canonicalBasePages, std::vector<PageId>{0});
    manager.retireHybridSnapshotReservation(coldLease, *snapshots);
    coldLease.release();

    CacheRecord const& record = manager.records().get(*published.record);
    EXPECT_EQ(record.hybridKey(), std::optional<HybridCheckpointKey>{checkpoint});
    EXPECT_EQ(record.basePagePath, std::vector<PageId>{0});
    EXPECT_EQ(record.recurrentSnapshotSlot, std::optional<int32_t>{0});
    EXPECT_EQ(record.partialKvSnapshotSlot, std::optional<int32_t>{0});

    ReusePlan const hit = manager.planHybrid(kDOMAIN, kSCHEMA, {{6, kEXACT_DIGEST}}, {kHASH_A, kHASH_B},
        /*inputTokenCount=*/10, /*hasAttention=*/true);
    ASSERT_EQ(hit.reuseTokenLength, 6);
    AcquireResult hitResult = manager.acquire(hit);
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

TEST(ContextCacheManagerTests, HybridMtpEndpointPublishesAcquiresAndGrowsAllStateAtomically)
{
    constexpr RecurrentStateSchemaId kSCHEMA{0x7171717171717172ULL, 0x8181818181818182ULL};
    constexpr BlockHash kEXACT_DIGEST{0x9191919191919192ULL, 0xA1A1A1A1A1A1A1A2ULL};
    ContextCacheManager manager(kPAGE_SIZE, ResourceDemand{5, 5, 2, 2}, 1);

    ReusePlan const cold = manager.planHybridMtp(kDOMAIN, kSCHEMA, kDRAFT_SIGNATURE, {}, {kHASH_A},
        /*inputTokenCount=*/6);
    EXPECT_EQ(cold.mode, ReusePlanMode::kHybridMtp);
    EXPECT_EQ(cold.demand.baseKvPages, 2);
    EXPECT_EQ(cold.demand.draftKvPages, 2);
    AcquireResult coldResult = manager.acquire(cold);
    CacheRequestLease coldLease = takeLease(coldResult);
    ASSERT_EQ(coldLease.basePages(), std::vector<PageId>({0, 1}));
    ASSERT_EQ(coldLease.draftPages(), std::vector<PageId>({0, 1}));

    std::optional<HybridSnapshotReservation> const snapshots = manager.reserveHybridSnapshots(coldLease, true);
    ASSERT_TRUE(snapshots.has_value());
    HybridMtpCheckpointKey const checkpoint{kDOMAIN, kEXACT_DIGEST, 6, kSCHEMA, kDRAFT_SIGNATURE};
    PublishResult const published = manager.publishHybridMtpDetailed(coldLease,
        HybridMtpPublishRequest{
            {kHASH_A}, checkpoint, PublicationPoint::kPrefillEnd, CommitPolicy::kPrefillStateOnly, *snapshots});
    ASSERT_EQ(published.status, PublishStatus::kPublished);
    ASSERT_TRUE(published.record.has_value());
    EXPECT_TRUE(published.lineageComplete);
    EXPECT_EQ(published.canonicalBasePages, std::vector<PageId>{0});
    manager.retireHybridSnapshotReservation(coldLease, *snapshots);
    coldLease.release();

    CacheRecord const& record = manager.records().get(*published.record);
    EXPECT_EQ(record.hybridMtpKey(), std::optional<HybridMtpCheckpointKey>{checkpoint});
    EXPECT_EQ(record.basePagePath, std::vector<PageId>{0});
    EXPECT_EQ(record.draftPagePath, std::vector<PageId>{0});
    EXPECT_EQ(record.recurrentSnapshotSlot, std::optional<int32_t>{0});
    EXPECT_EQ(record.partialKvSnapshotSlot, std::optional<int32_t>{0});
    EXPECT_FALSE(manager.draftIndex().lookupLongest(kDRAFT_SIGNATURE, kDOMAIN, {kHASH_A}, 1).has_value());

    ReusePlan const hit = manager.planHybridMtp(
        kDOMAIN, kSCHEMA, kDRAFT_SIGNATURE, {{6, kEXACT_DIGEST}}, {kHASH_A, kHASH_B}, /*inputTokenCount=*/10);
    ASSERT_EQ(hit.reuseTokenLength, 6);
    AcquireResult hitResult = manager.acquire(hit);
    ASSERT_EQ(hitResult.status, AcquireStatus::kAcquired);
    CacheRequestLease hitLease = takeLease(hitResult);
    ASSERT_EQ(hitLease.basePages().size(), 3U);
    ASSERT_EQ(hitLease.draftPages().size(), 3U);
    EXPECT_EQ(hitLease.basePages().front(), 0);
    EXPECT_EQ(hitLease.draftPages().front(), 0);
    EXPECT_EQ(hitLease.recurrentSnapshotSlot(), std::optional<int32_t>{0});
    EXPECT_EQ(hitLease.partialKvSnapshotSlot(), std::optional<int32_t>{0});

    EXPECT_TRUE(manager.growHybridMtpPages(hitLease, 1, 1));
    ASSERT_EQ(hitLease.basePages().size(), 4U);
    ASSERT_EQ(hitLease.draftPages().size(), 4U);
    EXPECT_EQ(hitLease.basePages().front(), 0);
    EXPECT_EQ(hitLease.draftPages().front(), 0);
    std::vector<PageId> const baseBeforeFailedGrowth = hitLease.basePages();
    std::vector<PageId> const draftBeforeFailedGrowth = hitLease.draftPages();
    EXPECT_FALSE(manager.growHybridMtpPages(hitLease, 1, 2));
    EXPECT_EQ(hitLease.basePages(), baseBeforeFailedGrowth);
    EXPECT_EQ(hitLease.draftPages(), draftBeforeFailedGrowth);

    manager.releaseRestoredHybridSnapshots(hitLease);
    EXPECT_FALSE(hitLease.recurrentSnapshotSlot().has_value());
    EXPECT_FALSE(hitLease.partialKvSnapshotSlot().has_value());
    hitLease.release();

    ReusePlan const replacementCold = manager.planHybridMtp(kDOMAIN, kSCHEMA, kDRAFT_SIGNATURE, {}, {kHASH_B},
        /*inputTokenCount=*/6, LookupPolicy::kBypass);
    AcquireResult replacementResult = manager.acquire(replacementCold);
    CacheRequestLease replacementLease = takeLease(replacementResult);
    std::optional<HybridSnapshotReservation> const replacementSnapshots
        = manager.reserveHybridSnapshots(replacementLease, true);
    ASSERT_TRUE(replacementSnapshots.has_value());
    HybridMtpCheckpointKey const replacementCheckpoint{kDOMAIN, kHASH_D, 6, kSCHEMA, kDRAFT_SIGNATURE};
    PublishResult const replacementPublished = manager.publishHybridMtpDetailed(replacementLease,
        HybridMtpPublishRequest{{kHASH_B}, replacementCheckpoint, PublicationPoint::kPrefillEnd,
            CommitPolicy::kPrefillStateOnly, *replacementSnapshots});
    ASSERT_EQ(replacementPublished.status, PublishStatus::kPublished);
    manager.retireHybridSnapshotReservation(replacementLease, *replacementSnapshots);
    replacementLease.release();

    EXPECT_FALSE(manager.records().findHybridMtp(checkpoint).has_value());
    EXPECT_TRUE(manager.records().findHybridMtp(replacementCheckpoint).has_value());
    EXPECT_EQ(manager.pools().cacheRefCount(ResourceId{ResourceType::kBaseKvPage, 0}), 0);
    EXPECT_EQ(manager.pools().cacheRefCount(ResourceId{ResourceType::kDraftKvPage, 0}), 0);
    EXPECT_EQ(manager.pools().cacheRefCount(ResourceId{ResourceType::kRecurrentSnapshot, 0}), 0);
    EXPECT_EQ(manager.pools().cacheRefCount(ResourceId{ResourceType::kPartialKvSnapshot, 0}), 0);
}

TEST(ContextCacheManagerTests, HybridMtpPageAlignedBoundaryStaysInPrivatePartialPage)
{
    // A page-aligned checkpoint (exactLength == 2 * kPAGE_SIZE) must retain its successor-dependent boundary token in a
    // private partial page so the consumer can rewrite the boundary draft slot without mutating a shared reused page.
    // The publication therefore reserves one fewer full block than exactLength/pageSize and captures a full-size
    // (kPAGE_SIZE-token) partial page.
    constexpr RecurrentStateSchemaId kSCHEMA{0x7171717171717173ULL, 0x8181818181818183ULL};
    constexpr BlockHash kEXACT_DIGEST{0x9191919191919193ULL, 0xA1A1A1A1A1A1A1A3ULL};
    constexpr int32_t kALIGNED_LENGTH = 2 * kPAGE_SIZE;
    ContextCacheManager manager(kPAGE_SIZE, ResourceDemand{5, 5, 2, 2}, 1);

    ReusePlan const cold = manager.planHybridMtp(
        kDOMAIN, kSCHEMA, kDRAFT_SIGNATURE, {}, {kHASH_A, kHASH_B}, /*inputTokenCount=*/kALIGNED_LENGTH);
    EXPECT_EQ(cold.mode, ReusePlanMode::kHybridMtp);
    AcquireResult coldResult = manager.acquire(cold);
    CacheRequestLease coldLease = takeLease(coldResult);
    ASSERT_GE(coldLease.basePages().size(), 2U);
    ASSERT_GE(coldLease.draftPages().size(), 2U);

    std::optional<HybridSnapshotReservation> const snapshots = manager.reserveHybridSnapshots(coldLease, true);
    ASSERT_TRUE(snapshots.has_value());
    // fullBlockCount == (8 - 1) / 4 == 1, so only the first block is published as a full logical block.
    HybridMtpCheckpointKey const checkpoint{kDOMAIN, kEXACT_DIGEST, kALIGNED_LENGTH, kSCHEMA, kDRAFT_SIGNATURE};
    PublishResult const published = manager.publishHybridMtpDetailed(coldLease,
        HybridMtpPublishRequest{
            {kHASH_A}, checkpoint, PublicationPoint::kPrefillEnd, CommitPolicy::kPrefillStateOnly, *snapshots});
    ASSERT_EQ(published.status, PublishStatus::kPublished);
    ASSERT_TRUE(published.record.has_value());
    EXPECT_EQ(published.canonicalBasePages, std::vector<PageId>{0});
    manager.retireHybridSnapshotReservation(coldLease, *snapshots);
    coldLease.release();

    CacheRecord const& record = manager.records().get(*published.record);
    EXPECT_EQ(record.basePagePath, std::vector<PageId>{0});
    EXPECT_EQ(record.draftPagePath, std::vector<PageId>{0});
    EXPECT_EQ(record.partialKvSnapshotSlot, std::optional<int32_t>{0});

    ReusePlan const hit = manager.planHybridMtp(kDOMAIN, kSCHEMA, kDRAFT_SIGNATURE, {{kALIGNED_LENGTH, kEXACT_DIGEST}},
        {kHASH_A, kHASH_B, kHASH_C}, /*inputTokenCount=*/3 * kPAGE_SIZE);
    ASSERT_EQ(hit.mode, ReusePlanMode::kHybridMtp);
    EXPECT_EQ(hit.reuseTokenLength, kALIGNED_LENGTH);
    AcquireResult hitResult = manager.acquire(hit);
    ASSERT_EQ(hitResult.status, AcquireStatus::kAcquired);
    CacheRequestLease hitLease = takeLease(hitResult);
    EXPECT_EQ(hitLease.basePages().front(), 0);
    EXPECT_EQ(hitLease.partialKvSnapshotSlot(), std::optional<int32_t>{0});
    manager.releaseRestoredHybridSnapshots(hitLease);
    hitLease.release();

    EXPECT_EQ(manager.pools().cacheRefCount(ResourceId{ResourceType::kPartialKvSnapshot, 0}), 1);
}

TEST(ContextCacheManagerTests, OneHybridMtpSnapshotSlotEvictsPriorConversationBeforePublication)
{
    constexpr RecurrentStateSchemaId kSCHEMA{0x7171717171717172ULL, 0x8181818181818182ULL};
    constexpr BlockHash kFIRST_EXACT_DIGEST{0x9191919191919192ULL, 0xA1A1A1A1A1A1A1A2ULL};
    constexpr BlockHash kSECOND_EXACT_DIGEST{0x9292929292929293ULL, 0xA2A2A2A2A2A2A2A3ULL};
    ContextCacheManager manager(kPAGE_SIZE, ResourceDemand{4, 4, 1, 1}, 2);

    ReusePlan const firstCold = manager.planHybridMtp(kDOMAIN, kSCHEMA, kDRAFT_SIGNATURE, {}, {kHASH_A},
        /*inputTokenCount=*/6, LookupPolicy::kBypass);
    AcquireResult firstResult = manager.acquire(firstCold);
    CacheRequestLease firstLease = takeLease(firstResult);
    std::optional<HybridSnapshotReservation> const firstSnapshots = manager.reserveHybridSnapshots(firstLease, true);
    ASSERT_TRUE(firstSnapshots.has_value());
    HybridMtpCheckpointKey const firstCheckpoint{kDOMAIN, kFIRST_EXACT_DIGEST, 6, kSCHEMA, kDRAFT_SIGNATURE};
    PublishResult const firstPublished = manager.publishHybridMtpDetailed(firstLease,
        HybridMtpPublishRequest{{kHASH_A}, firstCheckpoint, PublicationPoint::kPrefillEnd,
            CommitPolicy::kPrefillStateOnly, *firstSnapshots});
    ASSERT_TRUE(firstPublished.record.has_value());
    ASSERT_TRUE(firstPublished.lineageComplete);
    manager.retireHybridSnapshotReservation(firstLease, *firstSnapshots);
    firstLease.release();
    ASSERT_EQ(manager.baseIndex().lookup(BaseBlockKey{kDOMAIN, kHASH_A}), std::optional<PageId>{0});

    ReusePlan const secondCold = manager.planHybridMtp(kDOMAIN, kSCHEMA, kDRAFT_SIGNATURE, {}, {kHASH_A},
        /*inputTokenCount=*/6, LookupPolicy::kBypass);
    AcquireResult secondResult = manager.acquire(secondCold);
    CacheRequestLease secondLease = takeLease(secondResult);
    ASSERT_NE(secondLease.basePages().front(), 0);
    std::optional<HybridSnapshotReservation> const secondSnapshots = manager.reserveHybridSnapshots(secondLease, true);
    ASSERT_TRUE(secondSnapshots.has_value());
    EXPECT_FALSE(manager.records().findHybridMtp(firstCheckpoint).has_value());
    EXPECT_FALSE(manager.baseIndex().lookup(BaseBlockKey{kDOMAIN, kHASH_A}).has_value());

    HybridMtpCheckpointKey const secondCheckpoint{kDOMAIN, kSECOND_EXACT_DIGEST, 6, kSCHEMA, kDRAFT_SIGNATURE};
    PublishResult const secondPublished = manager.publishHybridMtpDetailed(secondLease,
        HybridMtpPublishRequest{{kHASH_A}, secondCheckpoint, PublicationPoint::kPrefillEnd,
            CommitPolicy::kPrefillStateOnly, *secondSnapshots});
    EXPECT_TRUE(secondPublished.lineageComplete);
    EXPECT_TRUE(manager.records().findHybridMtp(secondCheckpoint).has_value());
    EXPECT_EQ(manager.baseIndex().lookup(BaseBlockKey{kDOMAIN, kHASH_A}),
        std::optional<PageId>{secondLease.basePages().front()});
    manager.retireHybridSnapshotReservation(secondLease, *secondSnapshots);
    secondLease.release();
}

TEST(ContextCacheManagerTests, PureRecurrentCheckpointNeedsNoBaseOrPartialPage)
{
    constexpr RecurrentStateSchemaId kSCHEMA{0x7272727272727272ULL, 0x8282828282828282ULL};
    constexpr BlockHash kEXACT_DIGEST{0x9292929292929292ULL, 0xA2A2A2A2A2A2A2A2ULL};
    ContextCacheManager manager(kPAGE_SIZE, ResourceDemand{0, 0, 2, 0}, 2);

    ReusePlan const cold = manager.planHybrid(kDOMAIN, kSCHEMA, {}, {}, /*inputTokenCount=*/3, /*hasAttention=*/false);
    AcquireResult coldResult = manager.acquire(cold);
    CacheRequestLease coldLease = takeLease(coldResult);
    EXPECT_TRUE(coldLease.basePages().empty());
    std::optional<HybridSnapshotReservation> const snapshots = manager.reserveHybridSnapshots(coldLease, false);
    ASSERT_TRUE(snapshots.has_value());
    ASSERT_FALSE(snapshots->partialKvSnapshotSlot.has_value());

    HybridCheckpointKey const checkpoint{kDOMAIN, kEXACT_DIGEST, 3, kSCHEMA};
    PublishResult const published = manager.publishHybridDetailed(coldLease,
        HybridPublishRequest{
            {}, checkpoint, PublicationPoint::kPrefillEnd, CommitPolicy::kIncludingGeneratedTokens, *snapshots});
    ASSERT_EQ(published.status, PublishStatus::kPublished);
    manager.retireHybridSnapshotReservation(coldLease, *snapshots);
    coldLease.release();

    ReusePlan const hit = manager.planHybrid(kDOMAIN, kSCHEMA, {{3, kEXACT_DIGEST}}, {kHASH_A},
        /*inputTokenCount=*/5, /*hasAttention=*/false);
    EXPECT_EQ(hit.reuseTokenLength, 3);
    EXPECT_TRUE(hit.basePageBindings.empty());
    EXPECT_EQ(hit.demand.baseKvPages, 0);
    AcquireResult hitResult = manager.acquire(hit);
    CacheRequestLease hitLease = takeLease(hitResult);
    EXPECT_TRUE(hitLease.basePages().empty());
    EXPECT_TRUE(hitLease.recurrentSnapshotSlot().has_value());
    EXPECT_FALSE(hitLease.partialKvSnapshotSlot().has_value());
}

TEST(ContextCacheManagerTests, HybridPublicationStopsBeforeDescendantsOfAPrivateDuplicate)
{
    constexpr RecurrentStateSchemaId kSCHEMA{0x7373737373737373ULL, 0x8383838383838383ULL};
    constexpr BlockHash kEXACT_DIGEST{0x9393939393939393ULL, 0xA3A3A3A3A3A3A3A3ULL};
    ContextCacheManager manager(kPAGE_SIZE, ResourceDemand{5, 0, 2, 0}, 4);
    PublishedRecord const cached = publishPrivateRecord(manager, {kHASH_A});
    ASSERT_EQ(cached.producerPages, std::vector<PageId>{0});

    ReusePlan const cold = manager.planHybrid(kDOMAIN, kSCHEMA, {}, {kHASH_A, kHASH_B}, 2 * kPAGE_SIZE,
        /*hasAttention=*/true, LookupPolicy::kBypass);
    AcquireResult coldResult = manager.acquire(cold);
    CacheRequestLease coldLease = takeLease(coldResult);
    ASSERT_EQ(coldLease.basePages(), std::vector<PageId>({1, 2}));
    std::optional<HybridSnapshotReservation> const snapshots = manager.reserveHybridSnapshots(coldLease, false);
    ASSERT_TRUE(snapshots.has_value());

    HybridCheckpointKey const checkpoint{kDOMAIN, kEXACT_DIGEST, 2 * kPAGE_SIZE, kSCHEMA};
    PublishResult const published = manager.publishHybridDetailed(coldLease,
        HybridPublishRequest{{kHASH_A, kHASH_B}, checkpoint, PublicationPoint::kPrefillEnd,
            CommitPolicy::kIncludingGeneratedTokens, *snapshots});

    EXPECT_EQ(published.status, PublishStatus::kExistingRecord);
    EXPECT_FALSE(published.lineageComplete);
    EXPECT_EQ(published.publishedBaseFullBlockCount, 1);
    EXPECT_EQ(published.canonicalBasePages, std::vector<PageId>{0});
    EXPECT_FALSE(manager.records().findHybrid(checkpoint).has_value());
    EXPECT_EQ(coldLease.basePages(), std::vector<PageId>({1, 2}));
}

TEST(ContextCacheManagerTests, HybridMtpPublicationStopsBeforeDescendantsOfAPrivateDuplicate)
{
    constexpr RecurrentStateSchemaId kSCHEMA{0x7474747474747474ULL, 0x8484848484848484ULL};
    constexpr BlockHash kEXACT_DIGEST{0x9494949494949494ULL, 0xA4A4A4A4A4A4A4A4ULL};
    ContextCacheManager manager(kPAGE_SIZE, ResourceDemand{5, 2, 2, 1}, 4);
    PublishedRecord const cached = publishPrivateRecord(manager, {kHASH_A});
    ASSERT_EQ(cached.producerPages, std::vector<PageId>{0});

    ReusePlan const cold = manager.planHybridMtp(kDOMAIN, kSCHEMA, kDRAFT_SIGNATURE, {}, {kHASH_A},
        /*inputTokenCount=*/6, LookupPolicy::kBypass);
    AcquireResult coldResult = manager.acquire(cold);
    CacheRequestLease coldLease = takeLease(coldResult);
    std::vector<PageId> const privateBasePages = coldLease.basePages();
    std::vector<PageId> const privateDraftPages = coldLease.draftPages();
    ASSERT_EQ(privateBasePages.size(), 2);
    ASSERT_EQ(privateDraftPages.size(), 2);
    std::optional<HybridSnapshotReservation> const snapshots = manager.reserveHybridSnapshots(coldLease, true);
    ASSERT_TRUE(snapshots.has_value());

    HybridMtpCheckpointKey const checkpoint{kDOMAIN, kEXACT_DIGEST, 6, kSCHEMA, kDRAFT_SIGNATURE};
    PublishResult const published = manager.publishHybridMtpDetailed(coldLease,
        HybridMtpPublishRequest{
            {kHASH_A}, checkpoint, PublicationPoint::kPrefillEnd, CommitPolicy::kPrefillStateOnly, *snapshots});

    EXPECT_EQ(published.status, PublishStatus::kExistingRecord);
    EXPECT_FALSE(published.lineageComplete);
    EXPECT_EQ(published.publishedBaseFullBlockCount, 1);
    EXPECT_EQ(published.canonicalBasePages, std::vector<PageId>{0});
    EXPECT_FALSE(manager.records().findHybridMtp(checkpoint).has_value());
    EXPECT_EQ(coldLease.basePages(), privateBasePages);
    EXPECT_EQ(coldLease.draftPages(), privateDraftPages);
    manager.retireHybridSnapshotReservation(coldLease, *snapshots);
}

TEST(ContextCacheManagerTests, PublicationStopsBeforeDescendantsOfAPrivateDuplicate)
{
    ContextCacheManager manager(kPAGE_SIZE, ResourceDemand{4, 0, 0, 0}, 4);
    PublishedRecord const cached = publishPrivateRecord(manager, {kHASH_A});
    ASSERT_EQ(cached.producerPages, std::vector<PageId>{0});

    ReusePlan const cold = manager.planVanilla(kDOMAIN, {kHASH_A, kHASH_B}, 2 * kPAGE_SIZE, LookupPolicy::kBypass);
    AcquireResult result = manager.acquire(cold);
    CacheRequestLease lease = takeLease(result);
    ASSERT_EQ(lease.basePages(), std::vector<PageId>({1, 2}));

    PublishResult const published = manager.publishDetailed(lease,
        PublishRequest{{kHASH_A, kHASH_B}, 2 * kPAGE_SIZE, PublicationPoint::kPrefillEnd,
            CommitPolicy::kIncludingGeneratedTokens});

    EXPECT_EQ(published.status, PublishStatus::kExistingRecord);
    EXPECT_FALSE(published.lineageComplete);
    EXPECT_EQ(published.publishedBaseFullBlockCount, 1);
    EXPECT_EQ(published.canonicalBasePages, std::vector<PageId>{0});
    EXPECT_FALSE(manager.records().find(CacheRecordKey{kDOMAIN, kHASH_B, 2}).has_value());
    EXPECT_FALSE(manager.baseIndex().lookup(BaseBlockKey{kDOMAIN, kHASH_B}).has_value());
}

TEST(ContextCacheManagerTests, LeaseDestructorReleasesEveryUnpublishedResource)
{
    ContextCacheManager manager(kPAGE_SIZE, ResourceDemand{2, 1, 1, 1}, 1);

    {
        ReusePlan plan = manager.planVanilla(kDOMAIN, {}, kPAGE_SIZE);
        plan.demand.draftKvPages = 1;
        plan.demand.recurrentSnapshotSlots = 1;
        plan.demand.partialKvSnapshotSlots = 1;
        AcquireResult result = manager.acquire(plan);
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
        expectPoolState(manager, ResourceType::kRecurrentSnapshot, {1}, {0}, 0);
        expectPoolState(manager, ResourceType::kPartialKvSnapshot, {1}, {0}, 0);
    }

    expectPoolState(manager, ResourceType::kBaseKvPage, {0, 0}, {0, 0}, 2);
    expectPoolState(manager, ResourceType::kDraftKvPage, {0}, {0}, 1);
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
    ReusePlan const plan = manager.planVanilla(kDOMAIN, {kHASH_A}, kPAGE_SIZE + 1);

    AcquireResult const infeasible = manager.acquire(plan);

    EXPECT_FALSE(infeasible.lease.has_value());
    EXPECT_EQ(infeasible.status, AcquireStatus::kInsufficientCapacity);
    EXPECT_EQ(manager.records().lruToMru(), std::vector<RecordId>{cached.id});
    EXPECT_EQ(manager.baseIndex().lookup(BaseBlockKey{kDOMAIN, kHASH_A}), std::optional<PageId>{0});
    expectOnlyBaseState(manager, {0}, {1}, 0);

    ReusePlan malformed = plan;
    malformed.matchedBlockHashes.push_back(kHASH_B);
    EXPECT_THROW((void) manager.acquire(malformed), std::runtime_error);

    malformed = plan;
    malformed.baseCowSources.push_back(0);
    EXPECT_THROW((void) manager.acquire(malformed), std::runtime_error);

    malformed = plan;
    malformed.demand.draftKvPages = -1;
    EXPECT_THROW((void) manager.acquire(malformed), std::runtime_error);

    malformed = plan;
    malformed.demand.baseKvPages = 0;
    EXPECT_THROW((void) manager.acquire(malformed), std::runtime_error);

    malformed = plan;
    malformed.inputTokenCount = kPAGE_SIZE;
    malformed.demand.baseKvPages = 0;
    malformed.kind = ReusePlanKind::kStandard;
    EXPECT_THROW((void) manager.acquire(malformed), std::runtime_error);

    EXPECT_EQ(manager.records().lruToMru(), std::vector<RecordId>{cached.id});
    EXPECT_EQ(manager.baseIndex().lookup(BaseBlockKey{kDOMAIN, kHASH_A}), std::optional<PageId>{0});
    expectOnlyBaseState(manager, {0}, {1}, 0);

    ContextCacheManager staleManager(kPAGE_SIZE, ResourceDemand{2, 0, 0, 0}, 1);
    PublishedRecord const staleRecord = publishPrivateRecord(staleManager, {kHASH_A});
    ASSERT_EQ(staleRecord.status, PublishStatus::kPublished);
    ReusePlan const stalePlan = staleManager.planVanilla(kDOMAIN, {kHASH_A}, kPAGE_SIZE + 1);
    ASSERT_EQ(stalePlan.basePageBindings, std::vector<PageId>{0});
    PublishedRecord const replacement = publishPrivateRecord(staleManager, {kHASH_B});
    ASSERT_EQ(replacement.status, PublishStatus::kPublished);
    ASSERT_EQ(replacement.producerPages, std::vector<PageId>{1});
    ASSERT_FALSE(staleManager.baseIndex().lookup(BaseBlockKey{kDOMAIN, kHASH_A}).has_value());

    AcquireResult const stale = staleManager.acquire(stalePlan);

    EXPECT_FALSE(stale.lease.has_value());
    EXPECT_EQ(stale.status, AcquireStatus::kStalePlan);
    EXPECT_EQ(staleManager.records().lruToMru(), std::vector<RecordId>{replacement.id});
    EXPECT_EQ(staleManager.baseIndex().lookup(BaseBlockKey{kDOMAIN, kHASH_B}), std::optional<PageId>{1});
    expectOnlyBaseState(staleManager, {0, 0}, {0, 1}, 1);
}

TEST(ContextCacheManagerTests, PublishAddsCacheRefsBeforeDroppingActivity)
{
    ContextCacheManager manager(kPAGE_SIZE, ResourceDemand{1, 0, 0, 0}, 1);
    CacheRequestLease lease = acquirePrivatePages(manager, 1);
    ASSERT_EQ(lease.basePages(), std::vector<PageId>{0});
    expectOnlyBaseState(manager, {1}, {0}, 0);

    PublishRequest const negativeLength{
        {kHASH_A}, -1, PublicationPoint::kPrefillEnd, CommitPolicy::kIncludingGeneratedTokens};
    EXPECT_THROW((void) manager.publish(lease, negativeLength), std::runtime_error);
    PublishRequest const noFullBlock{
        {kHASH_A}, kPAGE_SIZE - 1, PublicationPoint::kPrefillEnd, CommitPolicy::kIncludingGeneratedTokens};
    EXPECT_THROW((void) manager.publish(lease, noFullBlock), std::runtime_error);
    EXPECT_TRUE(manager.records().lruToMru().empty());
    expectOnlyBaseState(manager, {1}, {0}, 0);

    PublishStatus const status = manager.publish(lease,
        PublishRequest{{kHASH_A}, kPAGE_SIZE, PublicationPoint::kPrefillEnd, CommitPolicy::kIncludingGeneratedTokens});

    EXPECT_EQ(status, PublishStatus::kPublished);
    ASSERT_EQ(manager.records().size(), 1U);
    RecordId const id = manager.records().lruToMru().front();
    CacheRecordKey const key{kDOMAIN, kHASH_A, 1};
    EXPECT_EQ(manager.records().find(key), std::optional<RecordId>{id});
    EXPECT_EQ(manager.records().get(id).logicalBlockHashes, std::vector<BlockHash>{kHASH_A});
    EXPECT_EQ(manager.records().get(id).basePagePath, std::vector<PageId>{0});
    EXPECT_EQ(manager.baseIndex().lookup(BaseBlockKey{kDOMAIN, kHASH_A}), std::optional<PageId>{0});
    EXPECT_TRUE(lease.valid());
    expectOnlyBaseState(manager, {1}, {1}, 0);

    lease.release();

    EXPECT_FALSE(lease.valid());
    expectOnlyBaseState(manager, {0}, {1}, 0);
}

TEST(ContextCacheManagerTests, DuplicatePublicationDoesNotAttachPrivateDescendants)
{
    ContextCacheManager manager(kPAGE_SIZE, ResourceDemand{3, 0, 0, 0}, 3);
    PublishedRecord const first = publishPrivateRecord(manager, {kHASH_A});
    ASSERT_EQ(first.status, PublishStatus::kPublished);
    ASSERT_EQ(first.producerPages, std::vector<PageId>{0});

    CacheRequestLease duplicateProducer = acquirePrivatePages(manager, 2);
    ASSERT_EQ(duplicateProducer.basePages(), std::vector<PageId>({1, 2}));
    PublishStatus const status = manager.publish(duplicateProducer,
        PublishRequest{{kHASH_A, kHASH_B}, 2 * kPAGE_SIZE, PublicationPoint::kPrefillEnd,
            CommitPolicy::kIncludingGeneratedTokens});

    EXPECT_EQ(status, PublishStatus::kExistingRecord);
    ASSERT_EQ(manager.records().size(), 1U);
    EXPECT_FALSE(manager.records().find(CacheRecordKey{kDOMAIN, kHASH_B, 2}).has_value());
    EXPECT_EQ(manager.baseIndex().lookup(BaseBlockKey{kDOMAIN, kHASH_A}), std::optional<PageId>{0});
    EXPECT_FALSE(manager.baseIndex().lookup(BaseBlockKey{kDOMAIN, kHASH_B}).has_value());
    EXPECT_EQ(manager.baseIndex().size(), 1U);
    expectOnlyBaseState(manager, {0, 1, 1}, {1, 0, 0}, 0);

    duplicateProducer.release();

    EXPECT_FALSE(duplicateProducer.valid());
    EXPECT_EQ(manager.records().lruToMru(), std::vector<RecordId>{first.id});
    expectOnlyBaseState(manager, {0, 0, 0}, {1, 0, 0}, 2);

    ContextCacheManager rollbackManager(kPAGE_SIZE, ResourceDemand{1, 0, 0, 0}, 2);
    CacheRequestLease conflictingLease = acquirePrivatePages(rollbackManager, 1);
    ASSERT_EQ(conflictingLease.basePages(), std::vector<PageId>{0});
    ASSERT_EQ(rollbackManager.publish(conflictingLease,
                  PublishRequest{
                      {kHASH_C}, kPAGE_SIZE, PublicationPoint::kPrefillEnd, CommitPolicy::kIncludingGeneratedTokens}),
        PublishStatus::kPublished);
    RecordId const canonicalRecord = rollbackManager.records().lruToMru().front();
    PublishRequest const conflictingPublication{
        {kHASH_D}, kPAGE_SIZE, PublicationPoint::kPrefillEnd, CommitPolicy::kIncludingGeneratedTokens};

    EXPECT_THROW((void) rollbackManager.publish(conflictingLease, conflictingPublication), std::runtime_error);

    EXPECT_EQ(rollbackManager.records().lruToMru(), std::vector<RecordId>{canonicalRecord});
    EXPECT_EQ(rollbackManager.baseIndex().lookup(BaseBlockKey{kDOMAIN, kHASH_C}), std::optional<PageId>{0});
    EXPECT_FALSE(rollbackManager.baseIndex().lookup(BaseBlockKey{kDOMAIN, kHASH_D}).has_value());
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

    ReusePlan const secondPlan = manager.planVanilla(kDOMAIN, {kHASH_A, kHASH_B, kHASH_D}, 3 * kPAGE_SIZE);
    ASSERT_EQ(secondPlan.basePageBindings, std::vector<PageId>({0, 1}));
    AcquireResult secondResult = manager.acquire(secondPlan);
    ASSERT_TRUE(secondResult.lease.has_value());
    CacheRequestLease secondLease = takeLease(secondResult);
    ASSERT_EQ(secondLease.basePages(), std::vector<PageId>({0, 1, 3}));
    PublishStatus const secondStatus = manager.publish(secondLease,
        PublishRequest{{kHASH_A, kHASH_B, kHASH_D}, 3 * kPAGE_SIZE, PublicationPoint::kPrefillEnd,
            CommitPolicy::kIncludingGeneratedTokens});
    ASSERT_EQ(secondStatus, PublishStatus::kPublished);
    RecordId const secondBranch = manager.records().lruToMru().back();
    secondLease.release();

    ASSERT_EQ(manager.records().lruToMru(), std::vector<RecordId>({firstBranch.id, secondBranch}));
    expectOnlyBaseState(manager, {0, 0, 0, 0}, {2, 2, 1, 1}, 0);

    CacheRequestLease replacementLease = acquirePrivatePages(manager, 1);
    ASSERT_EQ(replacementLease.basePages(), std::vector<PageId>{2});
    EXPECT_EQ(manager.records().lruToMru(), std::vector<RecordId>{secondBranch});
    EXPECT_EQ(manager.baseIndex().lookup(BaseBlockKey{kDOMAIN, kHASH_A}), std::optional<PageId>{0});
    EXPECT_EQ(manager.baseIndex().lookup(BaseBlockKey{kDOMAIN, kHASH_B}), std::optional<PageId>{1});
    EXPECT_FALSE(manager.baseIndex().lookup(BaseBlockKey{kDOMAIN, kHASH_C}).has_value());
    EXPECT_EQ(manager.baseIndex().lookup(BaseBlockKey{kDOMAIN, kHASH_D}), std::optional<PageId>{3});
    expectOnlyBaseState(manager, {0, 0, 1, 0}, {1, 1, 0, 1}, 0);

    PublishStatus const replacementStatus = manager.publish(replacementLease,
        PublishRequest{{kHASH_E}, kPAGE_SIZE, PublicationPoint::kPrefillEnd, CommitPolicy::kIncludingGeneratedTokens});
    ASSERT_EQ(replacementStatus, PublishStatus::kPublished);
    RecordId const replacement = manager.records().lruToMru().back();
    replacementLease.release();

    EXPECT_EQ(manager.records().lruToMru(), std::vector<RecordId>({secondBranch, replacement}));
    EXPECT_EQ(manager.records().get(secondBranch).basePagePath, std::vector<PageId>({0, 1, 3}));
    EXPECT_EQ(manager.baseIndex().lookup(BaseBlockKey{kDOMAIN, kHASH_E}), std::optional<PageId>{2});
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

    ReusePlan const plan = manager.planVanilla(kDOMAIN, {kHASH_A}, kPAGE_SIZE + 1);
    EXPECT_EQ(manager.records().lruToMru(), originalLru);
    AcquireResult result = manager.acquire(plan);
    ASSERT_TRUE(result.lease.has_value());
    CacheRequestLease lease = takeLease(result);

    EXPECT_EQ(lease.basePages(), std::vector<PageId>({1, 2}));
    EXPECT_EQ(manager.records().lruToMru(), originalLru);
    expectOnlyBaseState(manager, {0, 1, 1}, {1, 1, 0}, 0);

    PublishRequest const mismatchedPrefix{
        {kHASH_C}, kPAGE_SIZE, PublicationPoint::kPrefillEnd, CommitPolicy::kIncludingGeneratedTokens};
    EXPECT_THROW((void) manager.publish(lease, mismatchedPrefix), std::runtime_error);

    EXPECT_TRUE(lease.valid());
    EXPECT_EQ(lease.basePages(), std::vector<PageId>({1, 2}));
    EXPECT_EQ(manager.records().lruToMru(), originalLru);
    EXPECT_EQ(manager.records().get(first.id).basePagePath, std::vector<PageId>{0});
    EXPECT_EQ(manager.records().get(second.id).basePagePath, std::vector<PageId>{1});
    EXPECT_EQ(manager.baseIndex().lookup(BaseBlockKey{kDOMAIN, kHASH_C}), std::optional<PageId>{0});
    EXPECT_EQ(manager.baseIndex().lookup(BaseBlockKey{kDOMAIN, kHASH_A}), std::optional<PageId>{1});
    EXPECT_EQ(manager.baseIndex().size(), 2U);
    expectOnlyBaseState(manager, {0, 1, 1}, {1, 1, 0}, 0);

    lease.release();

    EXPECT_EQ(manager.records().lruToMru(), originalLru);
    EXPECT_EQ(manager.baseIndex().lookup(BaseBlockKey{kDOMAIN, kHASH_A}), std::optional<PageId>{1});
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

    ReusePlan const duplicatePlan = manager.planVanilla(kDOMAIN, {kHASH_A}, kPAGE_SIZE);
    ASSERT_EQ(duplicatePlan.kind, ReusePlanKind::kFullInputRewind);
    AcquireResult duplicateResult = manager.acquire(duplicatePlan);
    ASSERT_TRUE(duplicateResult.lease.has_value());
    EXPECT_EQ(duplicateResult.status, AcquireStatus::kAcquired);
    CacheRequestLease duplicateLease = takeLease(duplicateResult);
    ASSERT_EQ(duplicateLease.basePages(), std::vector<PageId>{2});

    PublishStatus const status = manager.publish(duplicateLease,
        PublishRequest{{kHASH_A}, kPAGE_SIZE, PublicationPoint::kPrefillEnd, CommitPolicy::kIncludingGeneratedTokens});

    EXPECT_EQ(status, PublishStatus::kExistingRecord);
    EXPECT_EQ(manager.records().lruToMru(), std::vector<RecordId>({second.id, first.id}));
    EXPECT_EQ(manager.records().size(), 2U);
    EXPECT_EQ(manager.records().get(first.id).basePagePath, std::vector<PageId>{0});
    EXPECT_EQ(manager.baseIndex().lookup(BaseBlockKey{kDOMAIN, kHASH_A}), std::optional<PageId>{0});
    EXPECT_EQ(manager.baseIndex().size(), 2U);
    expectOnlyBaseState(manager, {0, 0, 1}, {1, 1, 0}, 0);

    duplicateLease.release();

    expectOnlyBaseState(manager, {0, 0, 0}, {1, 1, 0}, 1);
}

TEST(ContextCacheManagerTests, RebindBasePrefixTransfersTheLeaseActiveReference)
{
    ContextCacheManager manager(kPAGE_SIZE, ResourceDemand{2, 0, 0, 0}, 2);
    PublishedRecord const cached = publishPrivateRecord(manager, {kHASH_A});
    ASSERT_EQ(cached.producerPages, std::vector<PageId>{0});

    ReusePlan const duplicate = manager.planVanilla(kDOMAIN, {kHASH_A}, kPAGE_SIZE);
    AcquireResult acquired = manager.acquire(duplicate);
    CacheRequestLease lease = takeLease(acquired);
    ASSERT_EQ(lease.basePages(), std::vector<PageId>{1});
    expectOnlyBaseState(manager, {0, 1}, {1, 0}, 0);

    PublishResult const published = manager.publishDetailed(lease,
        PublishRequest{{kHASH_A}, kPAGE_SIZE, PublicationPoint::kPrefillEnd, CommitPolicy::kIncludingGeneratedTokens});
    ASSERT_FALSE(published.lineageComplete);
    ASSERT_EQ(published.canonicalBasePages, std::vector<PageId>{0});

    manager.rebindBasePrefix(lease, published.canonicalBasePages);

    EXPECT_EQ(lease.basePages(), std::vector<PageId>{0});
    expectOnlyBaseState(manager, {1, 0}, {1, 0}, 1);
}

TEST(ContextCacheManagerTests, PrefillStateOnlySkipsDecodePublication)
{
    ContextCacheManager manager(kPAGE_SIZE, ResourceDemand{1, 0, 0, 0}, 1);
    CacheRequestLease lease = acquirePrivatePages(manager, 1);
    ASSERT_EQ(lease.basePages(), std::vector<PageId>{0});
    PublishRequest const skipped{{kHASH_A}, -1, PublicationPoint::kDecodeEnd, CommitPolicy::kPrefillStateOnly};

    CacheRequestLease invalidLease;
    EXPECT_THROW((void) manager.publish(invalidLease, skipped), std::runtime_error);
    PublishStatus const status = manager.publish(lease, skipped);

    EXPECT_EQ(status, PublishStatus::kSkippedByPolicy);
    EXPECT_TRUE(manager.records().lruToMru().empty());
    EXPECT_EQ(manager.baseIndex().size(), 0U);
    EXPECT_TRUE(lease.valid());
    expectOnlyBaseState(manager, {1}, {0}, 0);

    lease.release();

    expectOnlyBaseState(manager, {0}, {0}, 1);
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
    PublishStatus const status = manager.publish(thirdLease,
        PublishRequest{{kHASH_C}, kPAGE_SIZE, PublicationPoint::kPrefillEnd, CommitPolicy::kIncludingGeneratedTokens});

    EXPECT_EQ(status, PublishStatus::kPublished);
    ASSERT_EQ(manager.records().size(), 2U);
    std::vector<RecordId> const lru = manager.records().lruToMru();
    ASSERT_EQ(lru.size(), 2U);
    EXPECT_EQ(lru.front(), second.id);
    EXPECT_NE(lru.back(), first.id);
    EXPECT_NE(lru.back(), second.id);
    EXPECT_FALSE(manager.records().find(CacheRecordKey{kDOMAIN, kHASH_A, 1}).has_value());
    EXPECT_EQ(manager.records().find(CacheRecordKey{kDOMAIN, kHASH_B, 1}), std::optional<RecordId>{second.id});
    EXPECT_FALSE(manager.baseIndex().lookup(BaseBlockKey{kDOMAIN, kHASH_A}).has_value());
    EXPECT_EQ(manager.baseIndex().lookup(BaseBlockKey{kDOMAIN, kHASH_B}), std::optional<PageId>{1});
    EXPECT_EQ(manager.baseIndex().lookup(BaseBlockKey{kDOMAIN, kHASH_C}), std::optional<PageId>{2});
    expectOnlyBaseState(manager, {0, 0, 1}, {0, 1, 1}, 1);

    thirdLease.release();

    expectOnlyBaseState(manager, {0, 0, 0}, {0, 1, 1}, 1);

    ContextCacheManager zeroLimit(kPAGE_SIZE, ResourceDemand{1, 0, 0, 0}, 0);
    CacheRequestLease transient = acquirePrivatePages(zeroLimit, 1);
    PublishStatus const zeroStatus = zeroLimit.publish(transient,
        PublishRequest{{kHASH_D}, kPAGE_SIZE, PublicationPoint::kPrefillEnd, CommitPolicy::kIncludingGeneratedTokens});
    EXPECT_EQ(zeroStatus, PublishStatus::kPublished);
    EXPECT_TRUE(zeroLimit.records().lruToMru().empty());
    EXPECT_EQ(zeroLimit.baseIndex().size(), 0U);
    expectOnlyBaseState(zeroLimit, {1}, {0}, 0);

    transient.release();

    expectOnlyBaseState(zeroLimit, {0}, {0}, 1);
}

TEST(ContextCacheManagerTests, SpecAcquirePublishesPairedStateAndReturnsCowBindings)
{
    ContextCacheManager manager(kPAGE_SIZE, ResourceDemand{4, 4, 0, 0}, 2);
    PublishedSpecRecord const published = publishPrivateSpecRecord(manager, kDRAFT_SIGNATURE, {kHASH_A, kHASH_B});
    ASSERT_EQ(published.status, PublishStatus::kPublished);
    ASSERT_EQ(published.producerBasePages, std::vector<PageId>({0, 1}));
    ASSERT_EQ(published.producerDraftPages, std::vector<PageId>({0, 1}));

    CacheRecord const& record = manager.records().get(published.id);
    EXPECT_EQ(record.basePagePath, std::vector<PageId>({0, 1}));
    EXPECT_EQ(record.draftSignature, std::optional<DraftEngineSignature>{kDRAFT_SIGNATURE});
    EXPECT_EQ(record.draftPagePath, std::vector<PageId>({0, 1}));
    EXPECT_EQ(record.pairedDraftFullBlockCount, 2);
    EXPECT_EQ(manager.baseIndex().lookup(BaseBlockKey{kDOMAIN, kHASH_A}), std::optional<PageId>{0});
    EXPECT_EQ(manager.baseIndex().lookup(BaseBlockKey{kDOMAIN, kHASH_B}), std::optional<PageId>{1});
    EXPECT_EQ(manager.draftIndex().lookupLongest(kDRAFT_SIGNATURE, kDOMAIN, {kHASH_A, kHASH_B}, 2),
        (std::optional<DraftPathMatch>{DraftPathMatch{published.id, 2}}));
    expectPoolState(manager, ResourceType::kBaseKvPage, {0, 0, 0, 0}, {1, 1, 0, 0}, 2);
    expectPoolState(manager, ResourceType::kDraftKvPage, {0, 0, 0, 0}, {1, 1, 0, 0}, 2);

    ReusePlan const vanilla = manager.planVanilla(kDOMAIN, {kHASH_A, kHASH_B}, 2 * kPAGE_SIZE + 1);
    EXPECT_EQ(vanilla.basePageBindings, std::vector<PageId>({0, 1}));
    EXPECT_EQ(vanilla.reuseTokenLength, 2 * kPAGE_SIZE);

    ReusePlan const spec = manager.planSpec(
        SpecDecodeMode::kEAGLE, kDOMAIN, kDRAFT_SIGNATURE, {kHASH_A, kHASH_B}, 2 * kPAGE_SIZE + 1, true);
    EXPECT_EQ(spec.baseCowSources, std::vector<PageId>{1});
    EXPECT_EQ(spec.draftCowSources, std::vector<PageId>{1});
    AcquireResult result = manager.acquire(spec);
    ASSERT_EQ(result.status, AcquireStatus::kAcquired);
    CacheRequestLease lease = takeLease(result);

    EXPECT_EQ(lease.basePages(), std::vector<PageId>({0, 2, 3}));
    EXPECT_EQ(lease.draftPages(), std::vector<PageId>({0, 2, 3}));
    EXPECT_EQ(lease.baseCowSources(), std::vector<PageId>{1});
    EXPECT_EQ(lease.draftCowSources(), std::vector<PageId>{1});
    EXPECT_EQ(lease.reuseTokenLength(), 2 * kPAGE_SIZE - 1);
    expectPoolState(manager, ResourceType::kBaseKvPage, {1, 1, 1, 1}, {1, 1, 0, 0}, 0);
    expectPoolState(manager, ResourceType::kDraftKvPage, {1, 1, 1, 1}, {1, 1, 0, 0}, 0);

    lease.release();
    expectPoolState(manager, ResourceType::kBaseKvPage, {0, 0, 0, 0}, {1, 1, 0, 0}, 2);
    expectPoolState(manager, ResourceType::kDraftKvPage, {0, 0, 0, 0}, {1, 1, 0, 0}, 2);
}

TEST(ContextCacheManagerTests, SpecDraftUpgradeRequiresTheCanonicalPhysicalBasePath)
{
    ContextCacheManager manager(kPAGE_SIZE, ResourceDemand{6, 4, 0, 0}, 2);
    ReusePlan const coldSpec
        = manager.planSpec(SpecDecodeMode::kEAGLE, kDOMAIN, kDRAFT_SIGNATURE, {kHASH_A, kHASH_B}, 2 * kPAGE_SIZE, true);
    ASSERT_EQ(coldSpec.kind, ReusePlanKind::kNoReusablePrefix);
    AcquireResult firstResult = manager.acquire(coldSpec);
    CacheRequestLease firstLease = takeLease(firstResult);
    ASSERT_EQ(firstLease.basePages(), std::vector<PageId>({0, 1}));
    ASSERT_EQ(firstLease.draftPages(), std::vector<PageId>({0, 1}));

    PublishStatus const baseOnly = manager.publish(firstLease,
        PublishRequest{{kHASH_A, kHASH_B}, 2 * kPAGE_SIZE, PublicationPoint::kPrefillEnd,
            CommitPolicy::kIncludingGeneratedTokens, 0});
    ASSERT_EQ(baseOnly, PublishStatus::kPublished);
    RecordId const recordId = manager.records().lruToMru().front();
    ASSERT_FALSE(manager.records().get(recordId).draftSignature.has_value());

    PublishStatus const upgrade = manager.publish(firstLease,
        PublishRequest{{kHASH_A, kHASH_B}, 2 * kPAGE_SIZE, PublicationPoint::kPrefillEnd,
            CommitPolicy::kIncludingGeneratedTokens, 2 * kPAGE_SIZE});
    EXPECT_EQ(upgrade, PublishStatus::kPublished);
    firstLease.release();

    ASSERT_EQ(manager.records().size(), 1U);
    EXPECT_EQ(manager.records().lruToMru(), std::vector<RecordId>{recordId});
    EXPECT_EQ(manager.records().get(recordId).basePagePath, std::vector<PageId>({0, 1}));
    EXPECT_EQ(manager.records().get(recordId).draftSignature, std::optional<DraftEngineSignature>{kDRAFT_SIGNATURE});
    EXPECT_EQ(manager.records().get(recordId).draftPagePath, std::vector<PageId>({0, 1}));
    expectPoolState(manager, ResourceType::kBaseKvPage, {0, 0, 0, 0, 0, 0}, {1, 1, 0, 0, 0, 0}, 4);
    expectPoolState(manager, ResourceType::kDraftKvPage, {0, 0, 0, 0}, {1, 1, 0, 0}, 2);

    ReusePlan const replacementPlan = manager.planSpec(
        SpecDecodeMode::kEAGLE, kDOMAIN, kOTHER_DRAFT_SIGNATURE, {kHASH_A, kHASH_B}, 2 * kPAGE_SIZE, true);
    ASSERT_EQ(replacementPlan.kind, ReusePlanKind::kNoReusablePrefix);
    AcquireResult replacementResult = manager.acquire(replacementPlan);
    CacheRequestLease replacementLease = takeLease(replacementResult);
    ASSERT_EQ(replacementLease.draftPages(), std::vector<PageId>({2, 3}));

    PublishResult const replacement = manager.publishDetailed(replacementLease,
        PublishRequest{{kHASH_A, kHASH_B}, 2 * kPAGE_SIZE, PublicationPoint::kPrefillEnd,
            CommitPolicy::kIncludingGeneratedTokens, 2 * kPAGE_SIZE});
    EXPECT_EQ(replacement.status, PublishStatus::kExistingRecord);
    EXPECT_FALSE(replacement.lineageComplete);
    EXPECT_EQ(replacement.publishedBaseFullBlockCount, 1);
    replacementLease.release();

    CacheRecord const& retained = manager.records().get(recordId);
    EXPECT_EQ(retained.draftSignature, std::optional<DraftEngineSignature>{kDRAFT_SIGNATURE});
    EXPECT_EQ(retained.draftPagePath, std::vector<PageId>({0, 1}));
    EXPECT_EQ(manager.draftIndex().lookupLongest(kDRAFT_SIGNATURE, kDOMAIN, {kHASH_A, kHASH_B}, 2),
        (std::optional<DraftPathMatch>{DraftPathMatch{recordId, 2}}));
    EXPECT_FALSE(
        manager.draftIndex().lookupLongest(kOTHER_DRAFT_SIGNATURE, kDOMAIN, {kHASH_A, kHASH_B}, 2).has_value());
    expectPoolState(manager, ResourceType::kDraftKvPage, {0, 0, 0, 0}, {1, 1, 0, 0}, 2);
}

TEST(ContextCacheManagerTests, SpecPublicationTracksCommittedDraftBoundarySeparately)
{
    ContextCacheManager manager(kPAGE_SIZE, ResourceDemand{6, 6, 0, 0}, 2);
    ReusePlan const cold = manager.planSpec(
        SpecDecodeMode::kEAGLE, kDOMAIN, kDRAFT_SIGNATURE, {kHASH_A, kHASH_B, kHASH_C}, 3 * kPAGE_SIZE, true);
    AcquireResult coldResult = manager.acquire(cold);
    CacheRequestLease coldLease = takeLease(coldResult);

    PublishStatus const partialDraft = manager.publish(coldLease,
        PublishRequest{{kHASH_A, kHASH_B, kHASH_C}, 3 * kPAGE_SIZE, PublicationPoint::kDecodeEnd,
            CommitPolicy::kIncludingGeneratedTokens, 2 * kPAGE_SIZE});
    EXPECT_EQ(partialDraft, PublishStatus::kPublished);
    coldLease.release();

    ASSERT_EQ(manager.records().size(), 1U);
    RecordId const recordId = manager.records().lruToMru().front();
    CacheRecord const& partial = manager.records().get(recordId);
    EXPECT_EQ(partial.baseFullBlockCount, 3);
    EXPECT_EQ(partial.basePagePath, std::vector<PageId>({0, 1, 2}));
    EXPECT_EQ(partial.pairedDraftFullBlockCount, 2);
    EXPECT_EQ(partial.draftPagePath, std::vector<PageId>({0, 1}));
    EXPECT_EQ(manager.draftIndex().lookupLongest(kDRAFT_SIGNATURE, kDOMAIN, {kHASH_A, kHASH_B, kHASH_C}, 3),
        (std::optional<DraftPathMatch>{DraftPathMatch{recordId, 2}}));

    ReusePlan const vanilla = manager.planVanilla(kDOMAIN, {kHASH_A, kHASH_B, kHASH_C}, 3 * kPAGE_SIZE + 1);
    EXPECT_EQ(vanilla.basePageBindings, std::vector<PageId>({0, 1, 2}));
    ReusePlan const spec = manager.planSpec(
        SpecDecodeMode::kEAGLE, kDOMAIN, kDRAFT_SIGNATURE, {kHASH_A, kHASH_B, kHASH_C}, 3 * kPAGE_SIZE + 1, true);
    EXPECT_EQ(spec.draftRecord, std::optional<RecordId>{recordId});
    EXPECT_EQ(spec.draftPageBindings, std::vector<PageId>({0, 1}));
    EXPECT_EQ(spec.reuseTokenLength, 2 * kPAGE_SIZE - 1);

    ReusePlan const extend = manager.planSpec(
        SpecDecodeMode::kEAGLE, kDOMAIN, kDRAFT_SIGNATURE, {kHASH_A, kHASH_B, kHASH_C}, 3 * kPAGE_SIZE, true);
    AcquireResult extendResult = manager.acquire(extend);
    CacheRequestLease extendLease = takeLease(extendResult);
    PublishResult const fullDraft = manager.publishDetailed(extendLease,
        PublishRequest{{kHASH_A, kHASH_B, kHASH_C}, 3 * kPAGE_SIZE, PublicationPoint::kDecodeEnd,
            CommitPolicy::kIncludingGeneratedTokens, 3 * kPAGE_SIZE});
    EXPECT_EQ(fullDraft.status, PublishStatus::kExistingRecord);
    EXPECT_FALSE(fullDraft.lineageComplete);
    EXPECT_EQ(fullDraft.publishedBaseFullBlockCount, 3);
    extendLease.release();

    CacheRecord const& complete = manager.records().get(recordId);
    EXPECT_EQ(complete.basePagePath, std::vector<PageId>({0, 1, 2}));
    EXPECT_EQ(complete.pairedDraftFullBlockCount, 2);
    EXPECT_EQ(complete.draftPagePath.size(), 2U);
    EXPECT_EQ(manager.draftIndex().lookupLongest(kDRAFT_SIGNATURE, kDOMAIN, {kHASH_A, kHASH_B, kHASH_C}, 3),
        (std::optional<DraftPathMatch>{DraftPathMatch{recordId, 2}}));
}

TEST(ContextCacheManagerTests, SpecPublicationWithNoFullDraftPageRetainsBaseOnly)
{
    ContextCacheManager manager(kPAGE_SIZE, ResourceDemand{2, 2, 0, 0}, 1);
    ReusePlan const plan
        = manager.planSpec(SpecDecodeMode::kEAGLE, kDOMAIN, kDRAFT_SIGNATURE, {kHASH_A}, kPAGE_SIZE, true);
    AcquireResult result = manager.acquire(plan);
    CacheRequestLease lease = takeLease(result);

    PublishStatus const status = manager.publish(lease,
        PublishRequest{{kHASH_A}, kPAGE_SIZE, PublicationPoint::kDecodeEnd, CommitPolicy::kIncludingGeneratedTokens,
            kPAGE_SIZE - 1});
    EXPECT_EQ(status, PublishStatus::kPublished);
    lease.release();

    ASSERT_EQ(manager.records().size(), 1U);
    CacheRecord const& record = manager.records().get(manager.records().lruToMru().front());
    EXPECT_EQ(record.basePagePath, std::vector<PageId>{0});
    EXPECT_FALSE(record.draftSignature.has_value());
    EXPECT_TRUE(record.draftPagePath.empty());
    EXPECT_EQ(record.pairedDraftFullBlockCount, 0);
    EXPECT_FALSE(manager.draftIndex().lookupLongest(kDRAFT_SIGNATURE, kDOMAIN, {kHASH_A}, 1).has_value());

    ReusePlan const vanilla = manager.planVanilla(kDOMAIN, {kHASH_A}, kPAGE_SIZE + 1);
    EXPECT_EQ(vanilla.basePageBindings, std::vector<PageId>{0});
    ReusePlan const spec
        = manager.planSpec(SpecDecodeMode::kEAGLE, kDOMAIN, kDRAFT_SIGNATURE, {kHASH_A}, kPAGE_SIZE + 1, true);
    EXPECT_EQ(spec.kind, ReusePlanKind::kNoReusablePrefix);
    EXPECT_TRUE(spec.basePageBindings.empty());
    expectPoolState(manager, ResourceType::kBaseKvPage, {0, 0}, {1, 0}, 1);
    expectPoolState(manager, ResourceType::kDraftKvPage, {0, 0}, {0, 0}, 2);
}

TEST(ContextCacheManagerTests, SpecFullPageReplayRequiresRebindBeforePublishingDescendants)
{
    ContextCacheManager manager(kPAGE_SIZE, ResourceDemand{6, 6, 0, 0}, 2);
    PublishedSpecRecord const prefix = publishPrivateSpecRecord(manager, kDRAFT_SIGNATURE, {kHASH_A, kHASH_B});

    ReusePlan const plan = manager.planSpec(
        SpecDecodeMode::kEAGLE, kDOMAIN, kDRAFT_SIGNATURE, {kHASH_A, kHASH_B, kHASH_C}, 3 * kPAGE_SIZE, false);
    ASSERT_EQ(plan.specReplayMode, SpecReplayMode::kFullPage);
    ASSERT_EQ(plan.reuseTokenLength, kPAGE_SIZE);
    AcquireResult result = manager.acquire(plan);
    CacheRequestLease lease = takeLease(result);
    ASSERT_EQ(lease.basePages(), std::vector<PageId>({0, 2, 3}));
    ASSERT_EQ(lease.draftPages(), std::vector<PageId>({0, 2, 3}));

    PublishResult const published = manager.publishDetailed(lease,
        PublishRequest{{kHASH_A, kHASH_B, kHASH_C}, 3 * kPAGE_SIZE, PublicationPoint::kPrefillEnd,
            CommitPolicy::kIncludingGeneratedTokens, 3 * kPAGE_SIZE});
    EXPECT_EQ(published.status, PublishStatus::kExistingRecord);
    EXPECT_FALSE(published.lineageComplete);
    EXPECT_EQ(published.publishedBaseFullBlockCount, 2);
    EXPECT_EQ(published.canonicalBasePages, std::vector<PageId>({0, 1}));
    lease.release();

    EXPECT_EQ(manager.records().lruToMru(), std::vector<RecordId>{prefix.id});
    EXPECT_FALSE(manager.baseIndex().lookup(BaseBlockKey{kDOMAIN, kHASH_C}).has_value());
    EXPECT_EQ(manager.draftIndex().lookupLongest(kDRAFT_SIGNATURE, kDOMAIN, {kHASH_A, kHASH_B, kHASH_C}, 3),
        (std::optional<DraftPathMatch>{DraftPathMatch{prefix.id, 2}}));
    EXPECT_EQ(manager.records().get(prefix.id).draftPagePath, std::vector<PageId>({0, 1}));
}

TEST(ContextCacheManagerTests, SpecFullPageReplayRejectsChangedShiftedTokenDependency)
{
    ContextCacheManager manager(kPAGE_SIZE, ResourceDemand{6, 6, 0, 0}, 2);
    PublishedSpecRecord const prefix = publishPrivateSpecRecord(manager, kDRAFT_SIGNATURE, {kHASH_A, kHASH_B});

    ReusePlan const plan = manager.planSpec(
        SpecDecodeMode::kEAGLE, kDOMAIN, kDRAFT_SIGNATURE, {kHASH_A, kHASH_B, kHASH_C}, 3 * kPAGE_SIZE, false);
    ASSERT_EQ(plan.specReplayMode, SpecReplayMode::kFullPage);
    AcquireResult result = manager.acquire(plan);
    CacheRequestLease lease = takeLease(result);

    EXPECT_THROW((void) manager.publish(lease,
                     PublishRequest{{kHASH_A, kHASH_D, kHASH_E}, 3 * kPAGE_SIZE, PublicationPoint::kPrefillEnd,
                         CommitPolicy::kIncludingGeneratedTokens, 3 * kPAGE_SIZE}),
        std::runtime_error);
    EXPECT_EQ(manager.records().lruToMru(), std::vector<RecordId>{prefix.id});
    EXPECT_FALSE(manager.baseIndex().lookup(BaseBlockKey{kDOMAIN, kHASH_D}).has_value());
    EXPECT_FALSE(manager.baseIndex().lookup(BaseBlockKey{kDOMAIN, kHASH_E}).has_value());
}

TEST(ContextCacheManagerTests, SpecGrowthIsAtomicAcrossBaseAndDraftPools)
{
    ContextCacheManager manager(kPAGE_SIZE, ResourceDemand{3, 2, 0, 0}, 1);
    ReusePlan const plan = manager.planSpec(SpecDecodeMode::kEAGLE, kDOMAIN, kDRAFT_SIGNATURE, {}, 1, true);
    AcquireResult result = manager.acquire(plan);
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
    ContextCacheManager manager(kPAGE_SIZE, ResourceDemand{3, 3, 0, 0}, 2);
    PublishedSpecRecord const first = publishPrivateSpecRecord(manager, kDRAFT_SIGNATURE, {kHASH_A});
    PublishedSpecRecord const second = publishPrivateSpecRecord(manager, kDRAFT_SIGNATURE, {kHASH_B});
    ASSERT_EQ(manager.records().lruToMru(), std::vector<RecordId>({first.id, second.id}));

    ReusePlan const hit
        = manager.planSpec(SpecDecodeMode::kEAGLE, kDOMAIN, kDRAFT_SIGNATURE, {kHASH_A}, kPAGE_SIZE + 1, true);
    ASSERT_EQ(hit.draftRecord, std::optional<RecordId>{first.id});
    ASSERT_EQ(hit.demand.baseKvPages, 2);
    ASSERT_EQ(hit.demand.draftKvPages, 2);

    AcquireResult result = manager.acquire(hit);
    ASSERT_EQ(result.status, AcquireStatus::kAcquired);
    CacheRequestLease lease = takeLease(result);

    EXPECT_EQ(manager.records().lruToMru(), std::vector<RecordId>{first.id});
    EXPECT_TRUE(manager.records().contains(first.id));
    EXPECT_FALSE(manager.records().contains(second.id));
    EXPECT_EQ(manager.draftIndex().lookupLongest(kDRAFT_SIGNATURE, kDOMAIN, {kHASH_A}, 1),
        (std::optional<DraftPathMatch>{DraftPathMatch{first.id, 1}}));
    EXPECT_FALSE(manager.draftIndex().lookupLongest(kDRAFT_SIGNATURE, kDOMAIN, {kHASH_B}, 1).has_value());

    lease.release();
    expectPoolState(manager, ResourceType::kBaseKvPage, {0, 0, 0}, {1, 0, 0}, 2);
    expectPoolState(manager, ResourceType::kDraftKvPage, {0, 0, 0}, {1, 0, 0}, 2);
}

TEST(ContextCacheManagerTests, InfeasibleSpecAcquirePreservesLruAndTypedReferences)
{
    ContextCacheManager manager(kPAGE_SIZE, ResourceDemand{2, 2, 0, 0}, 2);
    PublishedSpecRecord const first = publishPrivateSpecRecord(manager, kDRAFT_SIGNATURE, {kHASH_A});
    PublishedSpecRecord const second = publishPrivateSpecRecord(manager, kDRAFT_SIGNATURE, {kHASH_B});
    std::vector<RecordId> const lruBefore{first.id, second.id};
    ASSERT_EQ(manager.records().lruToMru(), lruBefore);

    ReusePlan const hit
        = manager.planSpec(SpecDecodeMode::kEAGLE, kDOMAIN, kDRAFT_SIGNATURE, {kHASH_A}, kPAGE_SIZE + 1, true);
    ASSERT_EQ(hit.draftRecord, std::optional<RecordId>{first.id});
    ASSERT_EQ(hit.demand.baseKvPages, 2);
    ASSERT_EQ(hit.demand.draftKvPages, 2);

    AcquireResult result = manager.acquire(hit);

    EXPECT_EQ(result.status, AcquireStatus::kInsufficientCapacity);
    EXPECT_FALSE(result.lease.has_value());
    EXPECT_EQ(manager.records().lruToMru(), lruBefore);
    EXPECT_EQ(manager.draftIndex().lookupLongest(kDRAFT_SIGNATURE, kDOMAIN, {kHASH_A}, 1),
        (std::optional<DraftPathMatch>{DraftPathMatch{first.id, 1}}));
    EXPECT_EQ(manager.draftIndex().lookupLongest(kDRAFT_SIGNATURE, kDOMAIN, {kHASH_B}, 1),
        (std::optional<DraftPathMatch>{DraftPathMatch{second.id, 1}}));
    expectPoolState(manager, ResourceType::kBaseKvPage, {0, 0}, {1, 1}, 0);
    expectPoolState(manager, ResourceType::kDraftKvPage, {0, 0}, {1, 1}, 0);
}

TEST(ContextCacheManagerTests, SpecEvictionRemovesBothIndicesAndMakesOldPlanStale)
{
    ContextCacheManager manager(kPAGE_SIZE, ResourceDemand{3, 3, 0, 0}, 1);
    PublishedSpecRecord const first = publishPrivateSpecRecord(manager, kDRAFT_SIGNATURE, {kHASH_A});
    ReusePlan const stalePlan
        = manager.planSpec(SpecDecodeMode::kEAGLE, kDOMAIN, kDRAFT_SIGNATURE, {kHASH_A}, kPAGE_SIZE + 1, true);
    ASSERT_EQ(stalePlan.draftRecord, std::optional<RecordId>{first.id});

    PublishedSpecRecord const second = publishPrivateSpecRecord(manager, kDRAFT_SIGNATURE, {kHASH_B});
    ASSERT_EQ(second.status, PublishStatus::kPublished);
    ASSERT_NE(second.id, first.id);

    EXPECT_FALSE(manager.baseIndex().lookup(BaseBlockKey{kDOMAIN, kHASH_A}).has_value());
    EXPECT_FALSE(manager.draftIndex().lookupLongest(kDRAFT_SIGNATURE, kDOMAIN, {kHASH_A}, 1).has_value());
    EXPECT_EQ(manager.baseIndex().lookup(BaseBlockKey{kDOMAIN, kHASH_B}), std::optional<PageId>{1});
    EXPECT_EQ(manager.draftIndex().lookupLongest(kDRAFT_SIGNATURE, kDOMAIN, {kHASH_B}, 1),
        (std::optional<DraftPathMatch>{DraftPathMatch{second.id, 1}}));
    expectPoolState(manager, ResourceType::kBaseKvPage, {0, 0, 0}, {0, 1, 0}, 2);
    expectPoolState(manager, ResourceType::kDraftKvPage, {0, 0, 0}, {0, 1, 0}, 2);

    AcquireResult const stale = manager.acquire(stalePlan);
    EXPECT_EQ(stale.status, AcquireStatus::kStalePlan);
    EXPECT_FALSE(stale.lease.has_value());
    expectPoolState(manager, ResourceType::kBaseKvPage, {0, 0, 0}, {0, 1, 0}, 2);
    expectPoolState(manager, ResourceType::kDraftKvPage, {0, 0, 0}, {0, 1, 0}, 2);
}

TEST(ContextCacheManagerTests, SpecPlanValidationRejectsMissingOrBypassedReplay)
{
    ContextCacheManager manager(kPAGE_SIZE, ResourceDemand{4, 4, 0, 0}, 2);
    PublishedSpecRecord const published = publishPrivateSpecRecord(manager, kDRAFT_SIGNATURE, {kHASH_A, kHASH_B});
    ASSERT_EQ(published.status, PublishStatus::kPublished);

    ReusePlan bypassedReplay = manager.planSpec(
        SpecDecodeMode::kEAGLE, kDOMAIN, kDRAFT_SIGNATURE, {kHASH_A, kHASH_B}, 2 * kPAGE_SIZE + 1, true);
    ASSERT_EQ(bypassedReplay.specReplayMode, SpecReplayMode::kOneToken);
    bypassedReplay.baseCowSources.clear();
    bypassedReplay.draftCowSources.clear();
    bypassedReplay.specReplayMode = SpecReplayMode::kNone;
    bypassedReplay.reuseTokenLength = 2 * kPAGE_SIZE;
    bypassedReplay.demand = ResourceDemand{1, 1, 0, 0};
    ASSERT_THROW((void) manager.acquire(bypassedReplay), std::runtime_error);

    ReusePlan fullPage = manager.planSpec(
        SpecDecodeMode::kEAGLE, kDOMAIN, kDRAFT_SIGNATURE, {kHASH_A, kHASH_B, kHASH_C}, 3 * kPAGE_SIZE, false);
    ASSERT_TRUE(fullPage.specReplayDependency.has_value());
    ReusePlan missingDependency = fullPage;
    missingDependency.specReplayDependency.reset();
    EXPECT_THROW((void) manager.acquire(missingDependency), std::runtime_error);

    ReusePlan changedDependency = fullPage;
    changedDependency.specReplayDependency->terminalHash = kHASH_D;
    AcquireResult const staleDependency = manager.acquire(changedDependency);
    EXPECT_EQ(staleDependency.status, AcquireStatus::kStalePlan);
    EXPECT_FALSE(staleDependency.lease.has_value());

    ReusePlan missingBindings = manager.planSpec(SpecDecodeMode::kEAGLE, kDOMAIN, kOTHER_DRAFT_SIGNATURE, {}, 1, true);
    ASSERT_TRUE(missingBindings.basePageBindings.empty());
    ASSERT_TRUE(missingBindings.draftPageBindings.empty());
    missingBindings.baseCowSources = {0};
    missingBindings.draftCowSources = {0};
    missingBindings.specReplayMode = SpecReplayMode::kOneToken;
    EXPECT_THROW((void) manager.acquire(missingBindings), std::runtime_error);

    ReusePlan unknownMode = manager.planSpec(SpecDecodeMode::kEAGLE, kDOMAIN, kOTHER_DRAFT_SIGNATURE, {}, 1, true);
    unknownMode.mode = static_cast<ReusePlanMode>(std::numeric_limits<uint8_t>::max());
    EXPECT_THROW((void) manager.acquire(unknownMode), std::runtime_error);

    ReusePlan unknownReplay = manager.planSpec(SpecDecodeMode::kEAGLE, kDOMAIN, kOTHER_DRAFT_SIGNATURE, {}, 1, true);
    unknownReplay.specReplayMode = static_cast<SpecReplayMode>(std::numeric_limits<uint8_t>::max());
    EXPECT_THROW((void) manager.acquire(unknownReplay), std::runtime_error);
}

TEST(ContextCacheManagerTests, PublicationRequiresModeAppropriateDraftBoundary)
{
    ContextCacheManager vanillaManager(kPAGE_SIZE, ResourceDemand{1, 0, 0, 0}, 1);
    CacheRequestLease vanillaLease = acquirePrivatePages(vanillaManager, 1);
    EXPECT_THROW((void) vanillaManager.publish(vanillaLease,
                     PublishRequest{{kHASH_A}, kPAGE_SIZE, PublicationPoint::kPrefillEnd,
                         CommitPolicy::kIncludingGeneratedTokens, kPAGE_SIZE}),
        std::runtime_error);

    ContextCacheManager specManager(kPAGE_SIZE, ResourceDemand{1, 1, 0, 0}, 1);
    ReusePlan const specPlan
        = specManager.planSpec(SpecDecodeMode::kEAGLE, kDOMAIN, kDRAFT_SIGNATURE, {kHASH_A}, kPAGE_SIZE, true);
    AcquireResult specResult = specManager.acquire(specPlan);
    CacheRequestLease specLease = takeLease(specResult);
    EXPECT_THROW((void) specManager.publish(specLease,
                     PublishRequest{{kHASH_A}, kPAGE_SIZE, PublicationPoint::kPrefillEnd,
                         CommitPolicy::kIncludingGeneratedTokens}),
        std::runtime_error);
    EXPECT_THROW((void) specManager.publish(specLease,
                     PublishRequest{{kHASH_A}, kPAGE_SIZE, PublicationPoint::kPrefillEnd,
                         CommitPolicy::kIncludingGeneratedTokens, kPAGE_SIZE + 1}),
        std::runtime_error);
}
