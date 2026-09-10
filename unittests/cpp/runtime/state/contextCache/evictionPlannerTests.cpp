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

#include "runtime/state/contextCache/evictionPlanner.h"

#include "runtime/state/contextCache/cacheRecord.h"
#include "runtime/state/contextCache/resourcePools.h"

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

using namespace trt_edgellm::rt;

namespace
{

constexpr std::array<ResourceType, 4> kRESOURCE_TYPES{ResourceType::kBaseKvPage, ResourceType::kDraftKvPage,
    ResourceType::kRecurrentSnapshot, ResourceType::kPartialKvSnapshot};
constexpr BlockHash kHASH_A{0x1111111111111111ULL, 0xAAAAAAAAAAAAAAAAULL};
constexpr BlockHash kHASH_B{0x2222222222222222ULL, 0xBBBBBBBBBBBBBBBBULL};
constexpr BlockHash kHASH_C{0x3333333333333333ULL, 0xCCCCCCCCCCCCCCCCULL};

struct RecordState
{
    RecordId id{};
    CacheRecordKey key{};
    std::vector<BlockHash> logicalBlockHashes;
    std::vector<PageId> basePagePath;
    std::optional<SpecPagedStateRecord> specState;
    std::optional<int32_t> recurrentSnapshotSlot;
    std::optional<int32_t> partialKvSnapshotSlot;
    std::optional<int32_t> exactCheckpointLength;
    std::vector<ResourceId> resources;
    std::optional<RecordId> exactKeyLookup;
};

bool operator==(RecordState const& lhs, RecordState const& rhs)
{
    return lhs.id == rhs.id && lhs.key == rhs.key && lhs.logicalBlockHashes == rhs.logicalBlockHashes
        && lhs.basePagePath == rhs.basePagePath && lhs.specState == rhs.specState
        && lhs.recurrentSnapshotSlot == rhs.recurrentSnapshotSlot
        && lhs.partialKvSnapshotSlot == rhs.partialKvSnapshotSlot
        && lhs.exactCheckpointLength == rhs.exactCheckpointLength && lhs.resources == rhs.resources
        && lhs.exactKeyLookup == rhs.exactKeyLookup;
}

struct PlannerState
{
    std::array<int32_t, 4> freeCounts{};
    std::vector<std::tuple<ResourceId, int32_t, int32_t>> referenceCounts;
    std::vector<RecordId> lruToMru;
    size_t recordCount{};
    std::vector<RecordState> recordsInLruOrder;
    std::vector<ResourceId> freeAllocationOrder;
};

bool operator==(PlannerState const& lhs, PlannerState const& rhs)
{
    return lhs.freeCounts == rhs.freeCounts && lhs.referenceCounts == rhs.referenceCounts
        && lhs.lruToMru == rhs.lruToMru && lhs.recordCount == rhs.recordCount
        && lhs.recordsInLruOrder == rhs.recordsInLruOrder && lhs.freeAllocationOrder == rhs.freeAllocationOrder;
}

CacheRecord makeRecord(BlockHash logicalHash, std::vector<ResourceId> const& resources)
{
    CacheRecord record;
    std::vector<PageId> specPagePath;
    for (ResourceId const& resource : resources)
    {
        switch (resource.type)
        {
        case ResourceType::kBaseKvPage: record.basePagePath.push_back(resource.index); break;
        case ResourceType::kDraftKvPage: specPagePath.push_back(resource.index); break;
        case ResourceType::kRecurrentSnapshot: record.recurrentSnapshotSlot = resource.index; break;
        case ResourceType::kPartialKvSnapshot: record.partialKvSnapshotSlot = resource.index; break;
        }
    }
    size_t const logicalBlockCount = record.basePagePath.size();
    record.key = CacheRecordKey{logicalHash, static_cast<int32_t>(logicalBlockCount)};
    record.logicalBlockHashes.assign(logicalBlockCount, logicalHash);
    if (logicalBlockCount == 0 && record.recurrentSnapshotSlot.has_value())
    {
        record.exactCheckpointLength = 1;
    }
    if (!specPagePath.empty())
    {
        record.specState = SpecPagedStateRecord{std::move(specPagePath)};
    }
    return record;
}

std::vector<ResourceId> allocateResources(ResourcePools& pools, ResourceDemand const& demand)
{
    auto resources = pools.allocate(demand);
    if (!resources.has_value())
    {
        throw std::runtime_error("Test resource allocation failed");
    }
    return std::move(*resources);
}

RecordId insertCachedRecord(CacheRecordStore& records, ResourcePools& pools, CacheRecord record)
{
    std::vector<ResourceId> const resources = record.resources();
    RecordInsertResult const inserted = records.insert(std::move(record));
    if (!inserted.inserted)
    {
        throw std::runtime_error("Test cache record insertion was not unique");
    }
    for (ResourceId const& resource : resources)
    {
        pools.addCacheRef(resource);
    }
    return inserted.id;
}

void releaseActiveRefs(ResourcePools& pools, std::vector<ResourceId> const& resources)
{
    for (ResourceId const& resource : resources)
    {
        pools.releaseActiveRef(resource);
    }
}

PlannerState captureState(ResourcePools const& pools, CacheRecordStore const& records)
{
    PlannerState state;
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

    state.lruToMru = records.lruToMru();
    state.recordCount = records.size();
    for (RecordId const id : state.lruToMru)
    {
        CacheRecord const& record = records.get(id);
        state.recordsInLruOrder.push_back(RecordState{record.id, record.key, record.logicalBlockHashes,
            record.basePagePath, record.specState, record.recurrentSnapshotSlot, record.partialKvSnapshotSlot,
            record.exactCheckpointLength, record.resources(), records.find(record.key)});
    }
    return state;
}

void expectStateUnchanged(PlannerState const& before, ResourcePools const& pools, CacheRecordStore const& records)
{
    EXPECT_TRUE(captureState(pools, records) == before);
}

void expectDemand(ResourceDemand const& actual, ResourceDemand const& expected)
{
    EXPECT_EQ(actual.baseKvPages, expected.baseKvPages);
    EXPECT_EQ(actual.draftKvPages, expected.draftKvPages);
    EXPECT_EQ(actual.recurrentSnapshotSlots, expected.recurrentSnapshotSlots);
    EXPECT_EQ(actual.partialKvSnapshotSlots, expected.partialKvSnapshotSlots);
}

} // namespace

TEST(ContextCacheEvictionPlannerTests, FreeCapacityReturnsEmptyFeasiblePlan)
{
    ResourcePools pools(ResourceDemand{1, 1, 1, 1});
    CacheRecordStore records(1);
    RecordInsertResult const inserted = records.insert(makeRecord(kHASH_A, {{ResourceType::kBaseKvPage, 99}}));
    ASSERT_TRUE(inserted.inserted);

    PlannerState const before = captureState(pools, records);
    EvictionPlan const plan = EvictionPlanner::plan(ResourceDemand{1, 1, 1, 1}, pools, records);

    EXPECT_TRUE(plan.feasible);
    EXPECT_TRUE(plan.victims.empty());
    expectDemand(plan.reclaimed, ResourceDemand{});
    expectStateUnchanged(before, pools, records);

    PlannerState const beforeInvalidDemand = captureState(pools, records);
    EXPECT_THROW((void) EvictionPlanner::plan(ResourceDemand{-1, 0, 0, 0}, pools, records), std::runtime_error);
    expectStateUnchanged(beforeInvalidDemand, pools, records);
}

TEST(ContextCacheEvictionPlannerTests, SelectsRecordsFromLruToMru)
{
    ResourcePools pools(ResourceDemand{3, 0, 0, 0});
    std::vector<ResourceId> const resources = allocateResources(pools, ResourceDemand{3, 0, 0, 0});
    CacheRecordStore records(3);
    RecordId const first = insertCachedRecord(records, pools, makeRecord(kHASH_A, {resources[0]}));
    RecordId const second = insertCachedRecord(records, pools, makeRecord(kHASH_B, {resources[1]}));
    RecordId const third = insertCachedRecord(records, pools, makeRecord(kHASH_C, {resources[2]}));
    releaseActiveRefs(pools, resources);
    records.touch(first);
    ASSERT_EQ(records.lruToMru(), std::vector<RecordId>({second, third, first}));

    PlannerState const before = captureState(pools, records);
    EvictionPlan const plan = EvictionPlanner::plan(ResourceDemand{2, 0, 0, 0}, pools, records);

    EXPECT_TRUE(plan.feasible);
    EXPECT_EQ(plan.victims, std::vector<RecordId>({second, third}));
    expectDemand(plan.reclaimed, ResourceDemand{2, 0, 0, 0});
    expectStateUnchanged(before, pools, records);

    ResourcePools freePlusReclaimedPools(ResourceDemand{3, 0, 0, 0});
    std::vector<ResourceId> const cachedResources
        = allocateResources(freePlusReclaimedPools, ResourceDemand{2, 0, 0, 0});
    CacheRecordStore freePlusReclaimedRecords(2);
    RecordId const freePlusReclaimedFirst = insertCachedRecord(
        freePlusReclaimedRecords, freePlusReclaimedPools, makeRecord(kHASH_A, {cachedResources[0]}));
    RecordId const freePlusReclaimedSecond = insertCachedRecord(
        freePlusReclaimedRecords, freePlusReclaimedPools, makeRecord(kHASH_B, {cachedResources[1]}));
    releaseActiveRefs(freePlusReclaimedPools, cachedResources);
    ASSERT_EQ(freePlusReclaimedPools.freeCount(ResourceType::kBaseKvPage), 1);
    ASSERT_EQ(
        freePlusReclaimedRecords.lruToMru(), std::vector<RecordId>({freePlusReclaimedFirst, freePlusReclaimedSecond}));

    PlannerState const beforeFreePlusReclaimed = captureState(freePlusReclaimedPools, freePlusReclaimedRecords);
    EvictionPlan const freePlusReclaimedPlan
        = EvictionPlanner::plan(ResourceDemand{2, 0, 0, 0}, freePlusReclaimedPools, freePlusReclaimedRecords);

    EXPECT_TRUE(freePlusReclaimedPlan.feasible);
    EXPECT_EQ(freePlusReclaimedPlan.victims, std::vector<RecordId>{freePlusReclaimedFirst});
    expectDemand(freePlusReclaimedPlan.reclaimed, ResourceDemand{1, 0, 0, 0});
    expectStateUnchanged(beforeFreePlusReclaimed, freePlusReclaimedPools, freePlusReclaimedRecords);
}

TEST(ContextCacheEvictionPlannerTests, SharedAncestorRequiresMultipleVictims)
{
    ResourcePools pools(ResourceDemand{4, 0, 0, 0});
    std::vector<ResourceId> const resources = allocateResources(pools, ResourceDemand{4, 0, 0, 0});
    ResourceId const pageA = resources[0];
    ResourceId const pageB = resources[1];
    ResourceId const pageC = resources[2];
    ResourceId const pageD = resources[3];
    CacheRecordStore records(2);
    RecordId const first = insertCachedRecord(records, pools, makeRecord(kHASH_A, {pageA, pageB, pageC}));
    RecordId const second = insertCachedRecord(records, pools, makeRecord(kHASH_B, {pageA, pageB, pageD}));
    releaseActiveRefs(pools, resources);
    ASSERT_EQ(pools.cacheRefCount(pageA), 2);
    ASSERT_EQ(pools.cacheRefCount(pageB), 2);

    PlannerState const before = captureState(pools, records);
    EvictionPlan const plan = EvictionPlanner::plan(ResourceDemand{2, 0, 0, 0}, pools, records);

    EXPECT_TRUE(plan.feasible);
    EXPECT_EQ(plan.victims, std::vector<RecordId>({first, second}));
    expectDemand(plan.reclaimed, ResourceDemand{4, 0, 0, 0});
    expectStateUnchanged(before, pools, records);

    ResourcePools sharedOnlyPools(ResourceDemand{2, 0, 0, 0});
    std::vector<ResourceId> const sharedOnlyResources = allocateResources(sharedOnlyPools, ResourceDemand{2, 0, 0, 0});
    CacheRecordStore sharedOnlyRecords(2);
    RecordId const sharedOnlyFirst
        = insertCachedRecord(sharedOnlyRecords, sharedOnlyPools, makeRecord(kHASH_A, sharedOnlyResources));
    RecordId const sharedOnlySecond
        = insertCachedRecord(sharedOnlyRecords, sharedOnlyPools, makeRecord(kHASH_B, sharedOnlyResources));
    releaseActiveRefs(sharedOnlyPools, sharedOnlyResources);
    ASSERT_EQ(sharedOnlyPools.cacheRefCount(sharedOnlyResources[0]), 2);
    ASSERT_EQ(sharedOnlyPools.cacheRefCount(sharedOnlyResources[1]), 2);

    PlannerState const beforeSharedOnly = captureState(sharedOnlyPools, sharedOnlyRecords);
    EvictionPlan const sharedOnlyPlan
        = EvictionPlanner::plan(ResourceDemand{2, 0, 0, 0}, sharedOnlyPools, sharedOnlyRecords);

    EXPECT_TRUE(sharedOnlyPlan.feasible);
    EXPECT_EQ(sharedOnlyPlan.victims, std::vector<RecordId>({sharedOnlyFirst, sharedOnlySecond}));
    expectDemand(sharedOnlyPlan.reclaimed, ResourceDemand{2, 0, 0, 0});
    expectStateUnchanged(beforeSharedOnly, sharedOnlyPools, sharedOnlyRecords);
}

TEST(ContextCacheEvictionPlannerTests, ActiveReferencePreventsReclaim)
{
    ResourcePools pools(ResourceDemand{3, 0, 0, 0});
    std::vector<ResourceId> const resources = allocateResources(pools, ResourceDemand{3, 0, 0, 0});
    ResourceId const sharedPage = resources[0];
    CacheRecordStore records(2);
    RecordId const first = insertCachedRecord(records, pools, makeRecord(kHASH_A, {sharedPage, resources[1]}));
    RecordId const second = insertCachedRecord(records, pools, makeRecord(kHASH_B, {sharedPage, resources[2]}));
    pools.addActiveRef(sharedPage);
    releaseActiveRefs(pools, resources);
    ASSERT_EQ(pools.activeRefCount(sharedPage), 1);
    ASSERT_EQ(pools.cacheRefCount(sharedPage), 2);

    PlannerState const beforeFeasiblePlan = captureState(pools, records);
    EvictionPlan const feasiblePlan = EvictionPlanner::plan(ResourceDemand{2, 0, 0, 0}, pools, records);

    EXPECT_TRUE(feasiblePlan.feasible);
    EXPECT_EQ(feasiblePlan.victims, std::vector<RecordId>({first, second}));
    expectDemand(feasiblePlan.reclaimed, ResourceDemand{2, 0, 0, 0});
    expectStateUnchanged(beforeFeasiblePlan, pools, records);

    PlannerState const beforeInfeasiblePlan = captureState(pools, records);
    EvictionPlan const infeasiblePlan = EvictionPlanner::plan(ResourceDemand{3, 0, 0, 0}, pools, records);

    EXPECT_FALSE(infeasiblePlan.feasible);
    EXPECT_TRUE(infeasiblePlan.victims.empty());
    expectDemand(infeasiblePlan.reclaimed, ResourceDemand{});
    expectStateUnchanged(beforeInfeasiblePlan, pools, records);
}

TEST(ContextCacheEvictionPlannerTests, InfeasiblePlanReturnsNoVictimsAndMutatesNothing)
{
    ResourcePools pools(ResourceDemand{2, 0, 0, 0});
    std::vector<ResourceId> const resources = allocateResources(pools, ResourceDemand{2, 0, 0, 0});
    CacheRecordStore records(1);
    insertCachedRecord(records, pools, makeRecord(kHASH_A, {resources[0]}));
    pools.addActiveRef(resources[1]);
    releaseActiveRefs(pools, resources);

    PlannerState const before = captureState(pools, records);
    EvictionPlan const plan = EvictionPlanner::plan(ResourceDemand{2, 0, 0, 0}, pools, records);

    EXPECT_FALSE(plan.feasible);
    EXPECT_TRUE(plan.victims.empty());
    expectDemand(plan.reclaimed, ResourceDemand{});
    expectStateUnchanged(before, pools, records);

    ResourcePools inconsistentPools(ResourceDemand{1, 0, 0, 0});
    std::vector<ResourceId> const inconsistentResources
        = allocateResources(inconsistentPools, ResourceDemand{1, 0, 0, 0});
    CacheRecordStore inconsistentRecords(1);
    RecordInsertResult const inconsistentRecord
        = inconsistentRecords.insert(makeRecord(kHASH_A, {inconsistentResources.front()}));
    ASSERT_TRUE(inconsistentRecord.inserted);
    PlannerState const beforeInvariantViolation = captureState(inconsistentPools, inconsistentRecords);

    EXPECT_THROW((void) EvictionPlanner::plan(ResourceDemand{1, 0, 0, 0}, inconsistentPools, inconsistentRecords),
        std::runtime_error);
    expectStateUnchanged(beforeInvariantViolation, inconsistentPools, inconsistentRecords);
}

TEST(ContextCacheEvictionPlannerTests, DemandAcrossTypesMustBeSatisfiedTogether)
{
    ResourcePools pools(ResourceDemand{1, 1, 1, 1});
    std::vector<ResourceId> const resources = allocateResources(pools, ResourceDemand{1, 1, 1, 1});
    ResourceId const basePage = resources[0];
    ResourceId const draftPage = resources[1];
    ResourceId const recurrentSnapshot = resources[2];
    ResourceId const partialSnapshot = resources[3];
    CacheRecordStore records(2);
    RecordId const first = insertCachedRecord(records, pools, makeRecord(kHASH_A, {basePage, draftPage}));
    RecordId const second
        = insertCachedRecord(records, pools, makeRecord(kHASH_B, {recurrentSnapshot, partialSnapshot}));
    releaseActiveRefs(pools, resources);

    PlannerState const before = captureState(pools, records);
    EvictionPlan const plan = EvictionPlanner::plan(ResourceDemand{1, 1, 1, 1}, pools, records);

    EXPECT_TRUE(plan.feasible);
    EXPECT_EQ(plan.victims, std::vector<RecordId>({first, second}));
    expectDemand(plan.reclaimed, ResourceDemand{1, 1, 1, 1});
    expectStateUnchanged(before, pools, records);
}
