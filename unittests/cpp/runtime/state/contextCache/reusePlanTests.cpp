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

#include "runtime/state/contextCache/reusePlan.h"

#include "runtime/state/contextCache/blockIndex.h"
#include "runtime/state/contextCache/cacheRecord.h"
#include "runtime/state/contextCache/resourcePools.h"
#include "runtime/state/contextCache/specStatePlan.h"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

using namespace trt_edgellm::rt;

namespace
{

constexpr int32_t kPAGE_SIZE = 128;
constexpr BlockHash kHASH_A{0x1111111111111111ULL, 0xAAAAAAAAAAAAAAAAULL};
constexpr BlockHash kHASH_B{0x2222222222222222ULL, 0xBBBBBBBBBBBBBBBBULL};
constexpr BlockHash kHASH_C{0x3333333333333333ULL, 0xCCCCCCCCCCCCCCCCULL};
constexpr std::array<ResourceType, 4> kRESOURCE_TYPES{ResourceType::kBaseKvPage, ResourceType::kDraftKvPage,
    ResourceType::kRecurrentSnapshot, ResourceType::kPartialKvSnapshot};

struct PoolState
{
    std::array<int32_t, 4> freeCounts{};
    std::vector<std::tuple<ResourceId, int32_t, int32_t>> referenceCounts;
    std::vector<ResourceId> freeAllocationOrder;
};

bool operator==(PoolState const& lhs, PoolState const& rhs)
{
    return lhs.freeCounts == rhs.freeCounts && lhs.referenceCounts == rhs.referenceCounts
        && lhs.freeAllocationOrder == rhs.freeAllocationOrder;
}

PoolState capturePoolState(ResourcePools const& pools)
{
    PoolState state;
    ResourceDemand freeDemand;
    for (size_t typeIndex = 0; typeIndex < kRESOURCE_TYPES.size(); ++typeIndex)
    {
        ResourceType const type = kRESOURCE_TYPES[typeIndex];
        state.freeCounts[typeIndex] = pools.freeCount(type);
        freeDemand.get(type) = state.freeCounts[typeIndex];
        for (int32_t index = 0; index < pools.capacity(type); ++index)
        {
            ResourceId const resource{type, index};
            state.referenceCounts.emplace_back(resource, pools.activeRefCount(resource), pools.cacheRefCount(resource));
        }
    }

    ResourcePools poolsCopy = pools;
    auto freeResources = poolsCopy.allocate(freeDemand);
    if (!freeResources.has_value())
    {
        throw std::runtime_error("Test free resource allocation failed");
    }
    state.freeAllocationOrder = std::move(*freeResources);
    return state;
}

CacheRecord makeRecord(BlockHash hash, std::vector<PageId> basePages, std::vector<PageId> draftPages = {})
{
    CacheRecord record;
    record.key = CacheRecordKey{hash, 1};
    record.logicalBlockHashes = {hash};
    record.basePagePath = std::move(basePages);
    if (!draftPages.empty())
    {
        record.specState = SpecPagedStateRecord{std::move(draftPages)};
    }
    return record;
}

RecordId insertSpecRecord(CacheRecordStore& records, SpecStateIndex& specIndex, std::vector<BlockHash> logicalHashes,
    std::vector<PageId> basePages, std::vector<PageId> draftPages)
{
    CacheRecord record;
    record.key = CacheRecordKey{logicalHashes.back(), static_cast<int32_t>(logicalHashes.size())};
    record.logicalBlockHashes = std::move(logicalHashes);
    record.basePagePath = std::move(basePages);
    record.specState = SpecPagedStateRecord{std::move(draftPages)};
    RecordInsertResult const inserted = records.insert(std::move(record));
    if (!inserted.inserted)
    {
        throw std::runtime_error("Test spec record insertion was not unique");
    }
    specIndex.paged().insert(records.get(inserted.id));
    return inserted.id;
}

void expectDemand(ResourceDemand const& actual, ResourceDemand const& expected)
{
    EXPECT_EQ(actual.baseKvPages, expected.baseKvPages);
    EXPECT_EQ(actual.draftKvPages, expected.draftKvPages);
    EXPECT_EQ(actual.recurrentSnapshotSlots, expected.recurrentSnapshotSlots);
    EXPECT_EQ(actual.partialKvSnapshotSlots, expected.partialKvSnapshotSlots);
}

void expectRecordEqual(CacheRecord const& actual, CacheRecord const& expected)
{
    EXPECT_EQ(actual.id, expected.id);
    EXPECT_EQ(actual.key, expected.key);
    EXPECT_EQ(actual.logicalBlockHashes, expected.logicalBlockHashes);
    EXPECT_EQ(actual.basePagePath, expected.basePagePath);
    EXPECT_EQ(actual.specState, expected.specState);
    EXPECT_EQ(actual.recurrentSnapshotSlot, expected.recurrentSnapshotSlot);
    EXPECT_EQ(actual.partialKvSnapshotSlot, expected.partialKvSnapshotSlot);
    EXPECT_EQ(actual.exactCheckpointLength, expected.exactCheckpointLength);
}

ReusePlan makeEagleReusePlan(std::vector<BlockHash> const& inputFullBlockHashes, int32_t inputTokenCount,
    BaseBlockIndex const& baseIndex, SpecStateIndex const& specIndex, CacheRecordStore const& records,
    ContextCacheLookupPolicy lookupPolicy = ContextCacheLookupPolicy::kUseCache)
{
    SpecReuseContract const contract{/*ownsPagedSpecState=*/true, /*futureDependencyTokens=*/1};
    return makeSpecReusePlan(SpecReusePlanInput{inputFullBlockHashes, inputTokenCount, kPAGE_SIZE, lookupPolicy,
                                 baseIndex, specIndex, records},
        contract);
}

} // namespace

TEST(ContextCacheReusePlanTests, MissAllocatesEveryInputPage)
{
    BaseBlockIndex index;

    EXPECT_THROW((void) makeVanillaReusePlan({}, 1, 0, index), std::runtime_error);
    EXPECT_THROW((void) makeVanillaReusePlan({}, 1, -1, index), std::runtime_error);
    EXPECT_THROW((void) makeVanillaReusePlan({}, -1, kPAGE_SIZE, index), std::runtime_error);
    EXPECT_THROW((void) makeVanillaReusePlan({kHASH_A}, 0, kPAGE_SIZE, index), std::runtime_error);
    EXPECT_THROW((void) makeVanillaReusePlan({kHASH_A}, kPAGE_SIZE - 1, kPAGE_SIZE, index), std::runtime_error);

    ReusePlan const empty = makeVanillaReusePlan({}, 0, kPAGE_SIZE, index);
    EXPECT_EQ(empty.reuseTokenLength, 0);
    EXPECT_TRUE(empty.matchedBlockHashes.empty());
    EXPECT_TRUE(empty.basePageBindings.empty());
    expectDemand(empty.demand, ResourceDemand{});
    EXPECT_EQ(empty.kind, ReusePlanKind::kStandard);

    ReusePlan const miss = makeVanillaReusePlan({kHASH_A, kHASH_B}, 257, kPAGE_SIZE, index);
    EXPECT_EQ(miss.reuseTokenLength, 0);
    EXPECT_TRUE(miss.matchedBlockHashes.empty());
    EXPECT_TRUE(miss.basePageBindings.empty());
    expectDemand(miss.demand, ResourceDemand{3, 0, 0, 0});
    EXPECT_EQ(miss.kind, ReusePlanKind::kNoReusablePrefix);

    int32_t const maxInputTokenCount = std::numeric_limits<int32_t>::max();
    ReusePlan const maximumInput = makeVanillaReusePlan({}, maxInputTokenCount, kPAGE_SIZE, index);
    EXPECT_EQ(maximumInput.demand.baseKvPages, 16777216);
    EXPECT_EQ(maximumInput.kind, ReusePlanKind::kNoReusablePrefix);
}

TEST(ContextCacheReusePlanTests, FullBlocksReuseAndPartialTailGetsPrivatePage)
{
    BaseBlockIndex index;
    EXPECT_TRUE(index.insert(kHASH_A, 5).inserted);
    EXPECT_TRUE(index.insert(kHASH_B, 7).inserted);

    ReusePlan const partialTail = makeVanillaReusePlan({kHASH_A, kHASH_B}, 2 * kPAGE_SIZE + 1, kPAGE_SIZE, index);

    EXPECT_EQ(partialTail.matchedBlockHashes, std::vector<BlockHash>({kHASH_A, kHASH_B}));
    EXPECT_EQ(partialTail.basePageBindings, std::vector<PageId>({5, 7}));
    EXPECT_EQ(partialTail.matchedTokenLength, 2 * kPAGE_SIZE);
    EXPECT_EQ(partialTail.reuseTokenLength, 2 * kPAGE_SIZE);
    expectDemand(partialTail.demand, ResourceDemand{1, 0, 0, 0});
    EXPECT_EQ(partialTail.kind, ReusePlanKind::kStandard);

    ReusePlan const fewerHashes = makeVanillaReusePlan({kHASH_A, kHASH_B}, 3 * kPAGE_SIZE, kPAGE_SIZE, index);
    EXPECT_EQ(fewerHashes.matchedBlockHashes, partialTail.matchedBlockHashes);
    EXPECT_EQ(fewerHashes.basePageBindings, partialTail.basePageBindings);
    EXPECT_EQ(fewerHashes.reuseTokenLength, partialTail.reuseTokenLength);
    expectDemand(fewerHashes.demand, ResourceDemand{1, 0, 0, 0});
    EXPECT_EQ(fewerHashes.kind, ReusePlanKind::kStandard);
}

TEST(ContextCacheReusePlanTests, MatchStopsBeforeAnIndexHole)
{
    BaseBlockIndex index;
    EXPECT_TRUE(index.insert(kHASH_A, 11).inserted);
    EXPECT_TRUE(index.insert(kHASH_C, 13).inserted);

    ReusePlan const plan = makeVanillaReusePlan({kHASH_A, kHASH_B, kHASH_C}, 3 * kPAGE_SIZE, kPAGE_SIZE, index);

    EXPECT_EQ(plan.matchedBlockHashes, std::vector<BlockHash>{kHASH_A});
    EXPECT_EQ(plan.basePageBindings, std::vector<PageId>{11});
    EXPECT_EQ(plan.matchedTokenLength, kPAGE_SIZE);
    EXPECT_EQ(plan.reuseTokenLength, kPAGE_SIZE);
    expectDemand(plan.demand, ResourceDemand{2, 0, 0, 0});
    EXPECT_EQ(plan.kind, ReusePlanKind::kStandard);
    EXPECT_FALSE(index.lookup(kHASH_B).has_value());
    EXPECT_EQ(index.lookup(kHASH_C), std::optional<PageId>{13});
}

TEST(ContextCacheReusePlanTests, ExactBlockAlignedInputRewindsOneFullPage)
{
    BaseBlockIndex index;
    EXPECT_TRUE(index.insert(kHASH_A, 17).inserted);
    EXPECT_TRUE(index.insert(kHASH_B, 19).inserted);

    ReusePlan const plan = makeVanillaReusePlan({kHASH_A, kHASH_B}, 2 * kPAGE_SIZE, kPAGE_SIZE, index);

    EXPECT_EQ(plan.matchedBlockHashes, std::vector<BlockHash>{kHASH_A});
    EXPECT_EQ(plan.basePageBindings, std::vector<PageId>{17});
    EXPECT_EQ(plan.matchedBlockHashes.size(), plan.basePageBindings.size());
    EXPECT_EQ(plan.reuseTokenLength, kPAGE_SIZE);
    expectDemand(plan.demand, ResourceDemand{1, 0, 0, 0});
    EXPECT_EQ(plan.kind, ReusePlanKind::kFullInputRewind);
}

TEST(ContextCacheReusePlanTests, HybridPlanningRequiresOneCompleteExactCheckpoint)
{
    constexpr int32_t kHYBRID_PAGE_SIZE = 4;
    constexpr BlockHash kEXACT_DIGEST{0xBBBBBBBBBBBBBBB0ULL, 0xCCCCCCCCCCCCCCC0ULL};

    CacheRecord record;
    record.key = CacheRecordKey{kEXACT_DIGEST, 1};
    record.logicalBlockHashes = {kHASH_A};
    record.basePagePath = {7};
    record.recurrentSnapshotSlot = 3;
    record.partialKvSnapshotSlot = 4;
    record.exactCheckpointLength = 6;

    CacheRecordStore records(4);
    RecordInsertResult const inserted = records.insert(record);
    ASSERT_TRUE(inserted.inserted);

    ReusePlan const hit = makeHybridReusePlan({{6, kEXACT_DIGEST}}, {kHASH_A, kHASH_B},
        /*inputTokenCount=*/10, kHYBRID_PAGE_SIZE, /*hasAttention=*/true, records);
    EXPECT_EQ(hit.mode, ReusePlanMode::kHybrid);
    EXPECT_EQ(hit.matchedTokenLength, 6);
    EXPECT_EQ(hit.reuseTokenLength, 6);
    EXPECT_EQ(hit.basePageBindings, std::vector<PageId>{7});
    EXPECT_EQ(hit.matchedBlockHashes, std::vector<BlockHash>{kHASH_A});
    EXPECT_EQ(hit.hybridRecord, std::optional<RecordId>{inserted.id});
    EXPECT_EQ(hit.recurrentSnapshotBinding, std::optional<int32_t>{3});
    EXPECT_EQ(hit.partialKvSnapshotBinding, std::optional<int32_t>{4});
    expectDemand(hit.demand, ResourceDemand{2, 0, 0, 0});

    ReusePlan const exactInput = makeHybridReusePlan({{6, kEXACT_DIGEST}}, {kHASH_A},
        /*inputTokenCount=*/6, kHYBRID_PAGE_SIZE, /*hasAttention=*/true, records);
    EXPECT_EQ(exactInput.kind, ReusePlanKind::kNoReusablePrefix);
    EXPECT_EQ(exactInput.reuseTokenLength, 0);
    expectDemand(exactInput.demand, ResourceDemand{2, 0, 0, 0});

    CacheRecord incomplete = record;
    incomplete.key = CacheRecordKey{kHASH_C, 2};
    incomplete.logicalBlockHashes = {kHASH_A, kHASH_B};
    incomplete.basePagePath = {7, 8};
    incomplete.partialKvSnapshotSlot.reset();
    incomplete.exactCheckpointLength = 9;
    RecordInsertResult const incompleteInserted = records.insert(incomplete);
    ASSERT_TRUE(incompleteInserted.inserted);

    ReusePlan const skipsIncomplete = makeHybridReusePlan({{9, kHASH_C}, {6, kEXACT_DIGEST}}, {kHASH_A, kHASH_B},
        /*inputTokenCount=*/10, kHYBRID_PAGE_SIZE,
        /*hasAttention=*/true, records);
    EXPECT_EQ(skipsIncomplete.reuseTokenLength, 6);
    EXPECT_EQ(skipsIncomplete.hybridRecord, std::optional<RecordId>{inserted.id});
}

TEST(ContextCacheReusePlanTests, HybridMtpPlanningBindsBaseAndDraftAtExactCheckpoint)
{
    constexpr int32_t kHYBRID_PAGE_SIZE = 4;
    constexpr BlockHash kEXACT_DIGEST{0xBBBBBBBBBBBBBBB0ULL, 0xCCCCCCCCCCCCCCC0ULL};

    // fullBlockCount == (exactLength - 1) / pageSize == (6 - 1) / 4 == 1; totalInputPages == ceil(10 / 4) == 3.
    CacheRecord record;
    record.key = CacheRecordKey{kEXACT_DIGEST, 1};
    record.logicalBlockHashes = {kHASH_A};
    record.basePagePath = {7};
    record.specState = SpecPagedStateRecord{{8}};
    record.recurrentSnapshotSlot = 3;
    record.partialKvSnapshotSlot = 4;
    record.exactCheckpointLength = 6;

    CacheRecordStore records(4);
    RecordInsertResult const inserted = records.insert(record);
    ASSERT_TRUE(inserted.inserted);

    ReusePlan const hit = makeHybridMtpReusePlan({{6, kEXACT_DIGEST}}, {kHASH_A, kHASH_B},
        /*inputTokenCount=*/10, kHYBRID_PAGE_SIZE, records);
    EXPECT_EQ(hit.mode, ReusePlanMode::kHybridMtp);
    EXPECT_TRUE(hit.hybridHasAttention);
    EXPECT_EQ(hit.matchedTokenLength, 6);
    EXPECT_EQ(hit.reuseTokenLength, 6);
    EXPECT_EQ(hit.matchedBlockHashes, std::vector<BlockHash>{kHASH_A});
    EXPECT_EQ(hit.basePageBindings, std::vector<PageId>{7});
    EXPECT_EQ(hit.specPageBindings, std::vector<PageId>{8});
    EXPECT_EQ(hit.hybridRecord, std::optional<RecordId>{inserted.id});
    EXPECT_EQ(hit.recurrentSnapshotBinding, std::optional<int32_t>{3});
    EXPECT_EQ(hit.partialKvSnapshotBinding, std::optional<int32_t>{4});
    // privatePageCount == totalInputPages - fullBlockCount == 3 - 1 == 2, shared by base and draft demand.
    expectDemand(hit.demand, ResourceDemand{2, 2, 0, 0});
}

TEST(ContextCacheReusePlanTests, HybridMtpPlanningSkipsRecordWithoutDraftPath)
{
    constexpr int32_t kHYBRID_PAGE_SIZE = 4;
    constexpr BlockHash kEXACT_DIGEST{0xBBBBBBBBBBBBBBB0ULL, 0xCCCCCCCCCCCCCCC0ULL};

    // Same checkpoint as the positive case but with an EMPTY draft path: the candidate must be rejected.
    CacheRecord record;
    record.key = CacheRecordKey{kEXACT_DIGEST, 1};
    record.logicalBlockHashes = {kHASH_A};
    record.basePagePath = {7};
    record.recurrentSnapshotSlot = 3;
    record.partialKvSnapshotSlot = 4;
    record.exactCheckpointLength = 6;

    CacheRecordStore records(4);
    ASSERT_TRUE(records.insert(record).inserted);

    ReusePlan const cold = makeHybridMtpReusePlan({{6, kEXACT_DIGEST}}, {kHASH_A, kHASH_B},
        /*inputTokenCount=*/10, kHYBRID_PAGE_SIZE, records);
    EXPECT_EQ(cold.mode, ReusePlanMode::kHybridMtp);
    EXPECT_EQ(cold.kind, ReusePlanKind::kNoReusablePrefix);
    EXPECT_EQ(cold.reuseTokenLength, 0);
    EXPECT_FALSE(cold.hybridRecord.has_value());
    EXPECT_TRUE(cold.basePageBindings.empty());
    EXPECT_TRUE(cold.specPageBindings.empty());
    // totalInputPages == 3; MTP cold reserves the full input for BOTH base and draft pools (speculative deployment).
    expectDemand(cold.demand, ResourceDemand{3, 3, 0, 0});
}

TEST(ContextCacheReusePlanTests, PlanningDoesNotChangePoolRefsOrRecordRecency)
{
    ResourcePools pools(ResourceDemand{4, 2, 1, 1});
    auto const resources = pools.allocate(ResourceDemand{2, 1, 1, 1});
    ASSERT_TRUE(resources.has_value());
    ASSERT_EQ(resources->size(), 5U);
    ResourceId const basePageA{ResourceType::kBaseKvPage, 0};
    ResourceId const basePageB{ResourceType::kBaseKvPage, 1};
    ResourceId const draftPage{ResourceType::kDraftKvPage, 0};
    ResourceId const recurrentSnapshot{ResourceType::kRecurrentSnapshot, 0};
    ResourceId const partialSnapshot{ResourceType::kPartialKvSnapshot, 0};
    EXPECT_EQ(
        *resources, std::vector<ResourceId>({basePageA, basePageB, draftPage, recurrentSnapshot, partialSnapshot}));

    CacheRecord firstRecord = makeRecord(kHASH_A, {basePageA.index}, {draftPage.index});
    firstRecord.recurrentSnapshotSlot = recurrentSnapshot.index;
    firstRecord.partialKvSnapshotSlot = partialSnapshot.index;
    firstRecord.exactCheckpointLength = 97;
    CacheRecord secondRecord = makeRecord(kHASH_B, {basePageB.index});

    CacheRecordStore records(2);
    RecordInsertResult const first = records.insert(firstRecord);
    RecordInsertResult const second = records.insert(secondRecord);
    ASSERT_TRUE(first.inserted);
    ASSERT_TRUE(second.inserted);
    for (ResourceId const& resource : firstRecord.resources())
    {
        pools.addCacheRef(resource);
    }
    for (ResourceId const& resource : secondRecord.resources())
    {
        pools.addCacheRef(resource);
    }
    for (ResourceId const& resource : *resources)
    {
        pools.releaseActiveRef(resource);
    }
    records.touch(first.id);

    BaseBlockIndex index;
    EXPECT_TRUE(index.insert(kHASH_A, basePageA.index).inserted);
    EXPECT_TRUE(index.insert(kHASH_B, basePageB.index).inserted);

    PoolState const poolStateBefore = capturePoolState(pools);
    size_t const indexSizeBefore = index.size();
    BaseLookupResult const indexPrefixBefore = index.lookupPrefix({kHASH_A, kHASH_B});
    std::optional<PageId> const indexLookupBefore = index.lookup(kHASH_B);
    std::vector<RecordId> const lruBefore = records.lruToMru();
    CacheRecord const firstBefore = records.get(first.id);
    CacheRecord const secondBefore = records.get(second.id);
    std::optional<RecordId> const firstLookupBefore = records.find(firstRecord.key);
    std::optional<RecordId> const secondLookupBefore = records.find(secondRecord.key);

    ReusePlan const plan = makeVanillaReusePlan({kHASH_A, kHASH_B}, 2 * kPAGE_SIZE + 1, kPAGE_SIZE, index);

    EXPECT_EQ(plan.basePageBindings, std::vector<PageId>({basePageA.index, basePageB.index}));
    expectDemand(plan.demand, ResourceDemand{1, 0, 0, 0});
    EXPECT_TRUE(capturePoolState(pools) == poolStateBefore);
    EXPECT_EQ(index.size(), indexSizeBefore);
    BaseLookupResult const indexPrefixAfter = index.lookupPrefix({kHASH_A, kHASH_B});
    EXPECT_EQ(indexPrefixAfter.pageIds, indexPrefixBefore.pageIds);
    EXPECT_EQ(indexPrefixAfter.matchedHashes, indexPrefixBefore.matchedHashes);
    EXPECT_EQ(index.lookup(kHASH_B), indexLookupBefore);
    EXPECT_EQ(records.lruToMru(), lruBefore);
    EXPECT_EQ(records.size(), 2U);
    EXPECT_EQ(records.find(firstRecord.key), firstLookupBefore);
    EXPECT_EQ(records.find(secondRecord.key), secondLookupBefore);
    expectRecordEqual(records.get(first.id), firstBefore);
    expectRecordEqual(records.get(second.id), secondBefore);
}

TEST(ContextCacheReusePlanTests, SpecPlanningIgnoresBaseOnlyHits)
{
    BaseBlockIndex baseIndex;
    EXPECT_TRUE(baseIndex.insert(kHASH_A, 5).inserted);
    EXPECT_TRUE(baseIndex.insert(kHASH_B, 7).inserted);
    SpecStateIndex specIndex;
    CacheRecordStore records(1);

    ReusePlan const plan = makeEagleReusePlan({kHASH_A, kHASH_B}, 2 * kPAGE_SIZE + 1, baseIndex, specIndex, records);

    EXPECT_EQ(plan.mode, ReusePlanMode::kSpec);
    EXPECT_FALSE(plan.specRecord.has_value());
    EXPECT_TRUE(plan.matchedBlockHashes.empty());
    EXPECT_TRUE(plan.basePageBindings.empty());
    EXPECT_TRUE(plan.specPageBindings.empty());
    EXPECT_FALSE(plan.specReplayDependency.has_value());
    EXPECT_EQ(plan.reuseTokenLength, 0);
    EXPECT_EQ(plan.specReplayMode, SpecReplayMode::kNone);
    EXPECT_EQ(plan.kind, ReusePlanKind::kNoReusablePrefix);
    expectDemand(plan.demand, ResourceDemand{3, 3, 0, 0});
}

TEST(ContextCacheReusePlanTests, SpecPlanningUsesOneCoherentPathAndOneFullPageReplay)
{
    BaseBlockIndex baseIndex;
    EXPECT_TRUE(baseIndex.insert(kHASH_A, 17).inserted);
    EXPECT_TRUE(baseIndex.insert(kHASH_B, 19).inserted);
    SpecStateIndex specIndex;
    CacheRecordStore records(1);
    RecordId const record = insertSpecRecord(records, specIndex, {kHASH_A, kHASH_B}, {17, 19}, {23, 29});

    ReusePlan const plan = makeEagleReusePlan({kHASH_A, kHASH_B}, 2 * kPAGE_SIZE + 1, baseIndex, specIndex, records);

    EXPECT_EQ(plan.specRecord, std::optional<RecordId>{record});
    EXPECT_EQ(plan.matchedBlockHashes, std::vector<BlockHash>{kHASH_A});
    EXPECT_EQ(plan.basePageBindings, std::vector<PageId>{17});
    EXPECT_EQ(plan.specPageBindings, std::vector<PageId>{23});
    ASSERT_TRUE(plan.specReplayDependency.has_value());
    EXPECT_EQ(plan.specReplayDependency->pathBlockCount, 2);
    EXPECT_EQ(plan.specReplayDependency->terminalHash, kHASH_B);
    EXPECT_EQ(plan.matchedTokenLength, 2 * kPAGE_SIZE);
    EXPECT_EQ(plan.reuseTokenLength, kPAGE_SIZE);
    EXPECT_EQ(plan.specReplayMode, SpecReplayMode::kFullPage);
    EXPECT_EQ(plan.kind, ReusePlanKind::kStandard);
    expectDemand(plan.demand, ResourceDemand{2, 2, 0, 0});
}

TEST(ContextCacheReusePlanTests, SpecPlanningExactInputRewindsOneFullPage)
{
    BaseBlockIndex baseIndex;
    EXPECT_TRUE(baseIndex.insert(kHASH_A, 31).inserted);
    EXPECT_TRUE(baseIndex.insert(kHASH_B, 37).inserted);
    SpecStateIndex specIndex;
    CacheRecordStore records(1);
    RecordId const record = insertSpecRecord(records, specIndex, {kHASH_A, kHASH_B}, {31, 37}, {41, 43});

    ReusePlan const plan = makeEagleReusePlan({kHASH_A, kHASH_B}, 2 * kPAGE_SIZE, baseIndex, specIndex, records);

    EXPECT_EQ(plan.specRecord, std::optional<RecordId>{record});
    EXPECT_EQ(plan.matchedBlockHashes, std::vector<BlockHash>{kHASH_A});
    EXPECT_EQ(plan.basePageBindings, std::vector<PageId>{31});
    EXPECT_EQ(plan.specPageBindings, std::vector<PageId>{41});
    ASSERT_TRUE(plan.specReplayDependency.has_value());
    EXPECT_EQ(plan.specReplayDependency->pathBlockCount, 2);
    EXPECT_EQ(plan.specReplayDependency->terminalHash, kHASH_B);
    EXPECT_EQ(plan.matchedTokenLength, 2 * kPAGE_SIZE);
    EXPECT_EQ(plan.reuseTokenLength, kPAGE_SIZE);
    EXPECT_EQ(plan.specReplayMode, SpecReplayMode::kFullPage);
    EXPECT_EQ(plan.kind, ReusePlanKind::kFullInputRewind);
    expectDemand(plan.demand, ResourceDemand{1, 1, 0, 0});

    ReusePlan const singleBlock = makeEagleReusePlan({kHASH_A}, kPAGE_SIZE, baseIndex, specIndex, records);
    EXPECT_TRUE(singleBlock.basePageBindings.empty());
    EXPECT_TRUE(singleBlock.specPageBindings.empty());
    EXPECT_FALSE(singleBlock.specRecord.has_value());
    EXPECT_FALSE(singleBlock.specReplayDependency.has_value());
}
