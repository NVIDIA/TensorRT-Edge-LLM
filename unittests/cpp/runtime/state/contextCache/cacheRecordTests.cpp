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

#include "runtime/state/contextCache/cacheRecord.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <functional>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace trt_edgellm::rt;

static_assert(!std::is_copy_constructible_v<CacheRecordStore>, "CacheRecordStore must not be copy constructible");
static_assert(!std::is_copy_assignable_v<CacheRecordStore>, "CacheRecordStore must not be copy assignable");
static_assert(!std::is_move_constructible_v<CacheRecordStore>, "CacheRecordStore must not be move constructible");
static_assert(!std::is_move_assignable_v<CacheRecordStore>, "CacheRecordStore must not be move assignable");

namespace
{

constexpr BlockHash kHASH_A{0x1111111111111111ULL, 0xAAAAAAAAAAAAAAAAULL};
constexpr BlockHash kHASH_B{0x2222222222222222ULL, 0xBBBBBBBBBBBBBBBBULL};
constexpr BlockHash kHASH_C{0x3333333333333333ULL, 0xCCCCCCCCCCCCCCCCULL};
constexpr BlockHash kHASH_D{0x4444444444444444ULL, 0xDDDDDDDDDDDDDDDDULL};

CacheRecord makeRecord(std::vector<BlockHash> logicalBlockHashes, std::vector<PageId> basePagePath)
{
    CacheRecord record;
    record.key = CacheRecordKey{logicalBlockHashes.back(), static_cast<int32_t>(logicalBlockHashes.size())};
    record.logicalBlockHashes = std::move(logicalBlockHashes);
    record.basePagePath = std::move(basePagePath);
    return record;
}

SpecPagedStateRecord makePagedSpecState(std::vector<PageId> pagePath)
{
    return SpecPagedStateRecord{std::move(pagePath)};
}

std::vector<PageId> specPagePath(CacheRecord const& record)
{
    if (!record.specState.has_value())
    {
        return {};
    }
    return record.specState->pagePath;
}

} // namespace

TEST(ContextCacheRecordStoreTests, RecordOwnsItsCompleteBasePath)
{
    EXPECT_THROW((void) CacheRecordStore(-1), std::runtime_error);

    CacheRecordStore store(0);
    EXPECT_EQ(store.maxRecords(), 0);

    CacheRecord record = makeRecord({kHASH_A, kHASH_B, kHASH_C}, {10, 11, 12});
    constexpr RecordId kCALLER_RECORD_ID = 91;
    record.id = kCALLER_RECORD_ID;
    record.specState = makePagedSpecState({20, 21, 22});
    record.recurrentSnapshotSlot = 30;
    record.partialKvSnapshotSlot = 40;
    record.exactCheckpointLength = 96;
    CacheRecordKey const key = record.key;

    RecordInsertResult const inserted = store.insert(record);

    EXPECT_TRUE(inserted.inserted);
    EXPECT_NE(inserted.id, RecordId{});
    EXPECT_NE(inserted.id, kCALLER_RECORD_ID);
    EXPECT_EQ(store.size(), 1U);
    EXPECT_EQ(store.lruToMru(), std::vector<RecordId>{inserted.id});
    EXPECT_EQ(store.find(key), std::optional<RecordId>{inserted.id});

    record.logicalBlockHashes.clear();
    record.basePagePath.assign({90});
    record.specState = makePagedSpecState({91});
    record.recurrentSnapshotSlot = 92;
    record.partialKvSnapshotSlot = 93;

    CacheRecord const& stored = store.get(inserted.id);
    EXPECT_EQ(stored.id, inserted.id);
    EXPECT_EQ(stored.logicalBlockHashes, std::vector<BlockHash>({kHASH_A, kHASH_B, kHASH_C}));
    EXPECT_EQ(stored.basePagePath, std::vector<PageId>({10, 11, 12}));
    ASSERT_TRUE(stored.specState.has_value());
    EXPECT_EQ(specPagePath(stored), std::vector<PageId>({20, 21, 22}));
    EXPECT_EQ(stored.recurrentSnapshotSlot, std::optional<int32_t>{30});
    EXPECT_EQ(stored.partialKvSnapshotSlot, std::optional<int32_t>{40});
    EXPECT_EQ(stored.exactCheckpointLength, std::optional<int32_t>{96});
    std::vector<ResourceId> const expectedResources{{ResourceType::kBaseKvPage, 10}, {ResourceType::kBaseKvPage, 11},
        {ResourceType::kBaseKvPage, 12}, {ResourceType::kDraftKvPage, 20}, {ResourceType::kDraftKvPage, 21},
        {ResourceType::kDraftKvPage, 22}, {ResourceType::kRecurrentSnapshot, 30},
        {ResourceType::kPartialKvSnapshot, 40}};
    EXPECT_EQ(stored.resources(), expectedResources);

    auto expectInvalid = [&](CacheRecord invalid) {
        EXPECT_THROW((void) store.insert(std::move(invalid)), std::runtime_error);
        EXPECT_EQ(store.size(), 1U);
        EXPECT_EQ(store.lruToMru(), std::vector<RecordId>{inserted.id});
        EXPECT_EQ(store.find(key), std::optional<RecordId>{inserted.id});
    };

    CacheRecord valid = stored;
    CacheRecord invalid = valid;
    invalid.key.fullBlockCount = 0;
    expectInvalid(std::move(invalid));

    invalid = valid;
    invalid.logicalBlockHashes.pop_back();
    expectInvalid(std::move(invalid));

    invalid = makeRecord({kHASH_A, kHASH_B, kHASH_C}, {10, 11, 12});
    invalid.key.terminalHash = kHASH_D;
    expectInvalid(std::move(invalid));

    invalid = valid;
    invalid.basePagePath = {10, 11, 12, 13};
    expectInvalid(std::move(invalid));

    invalid = valid;
    invalid.specState->pagePath.pop_back();
    expectInvalid(std::move(invalid));

    invalid = valid;
    invalid.specState = makePagedSpecState({20, 21, 22, 23});
    expectInvalid(std::move(invalid));

    invalid = valid;
    invalid.basePagePath[1] = -1;
    expectInvalid(std::move(invalid));

    invalid = valid;
    invalid.specState->pagePath[1] = -1;
    expectInvalid(std::move(invalid));

    invalid = valid;
    invalid.recurrentSnapshotSlot = -1;
    expectInvalid(std::move(invalid));

    invalid = valid;
    invalid.partialKvSnapshotSlot = -1;
    expectInvalid(std::move(invalid));

    invalid = valid;
    invalid.exactCheckpointLength = -1;
    expectInvalid(std::move(invalid));
}

TEST(ContextCacheRecordStoreTests, ExactDuplicateReturnsExistingRecord)
{
    CacheRecordStore store(2);
    CacheRecord first = makeRecord({kHASH_A, kHASH_B}, {10, 11});
    first.recurrentSnapshotSlot = 12;
    RecordInsertResult const firstInsert = store.insert(first);
    ASSERT_TRUE(firstInsert.inserted);

    CacheRecord other = makeRecord({kHASH_D}, {20});
    RecordInsertResult const otherInsert = store.insert(other);
    ASSERT_TRUE(otherInsert.inserted);
    EXPECT_EQ(store.lruToMru(), std::vector<RecordId>({firstInsert.id, otherInsert.id}));

    CacheRecord duplicate = makeRecord({kHASH_A, kHASH_B}, {90, 91});
    duplicate.id = 999;
    duplicate.specState = makePagedSpecState({92, 93});
    duplicate.partialKvSnapshotSlot = 94;
    duplicate.recurrentSnapshotSlot = 95;
    duplicate.exactCheckpointLength = 96;

    RecordInsertResult const duplicateInsert = store.insert(std::move(duplicate));

    EXPECT_FALSE(duplicateInsert.inserted);
    EXPECT_EQ(duplicateInsert.id, firstInsert.id);
    EXPECT_EQ(store.size(), 2U);
    EXPECT_EQ(store.lruToMru(), std::vector<RecordId>({otherInsert.id, firstInsert.id}));
    CacheRecord const& stored = store.get(firstInsert.id);
    EXPECT_EQ(stored.basePagePath, std::vector<PageId>({10, 11}));
    EXPECT_FALSE(stored.specState.has_value());
    EXPECT_EQ(stored.recurrentSnapshotSlot, std::optional<int32_t>{12});
    EXPECT_FALSE(stored.partialKvSnapshotSlot.has_value());
    EXPECT_FALSE(stored.exactCheckpointLength.has_value());
    EXPECT_EQ(store.find(first.key), std::optional<RecordId>{firstInsert.id});
    EXPECT_FALSE(store.find(CacheRecordKey{kHASH_C, 1}).has_value());

    CacheRecordKey const changedTerminal{kHASH_C, first.key.fullBlockCount};
    CacheRecordKey const changedCount{first.key.terminalHash, first.key.fullBlockCount + 1};
    CacheRecordKey const equalKey = first.key;
    EXPECT_TRUE(first.key == equalKey);
    EXPECT_FALSE(first.key == changedTerminal);
    EXPECT_FALSE(first.key == changedCount);
    EXPECT_EQ(std::hash<CacheRecordKey>{}(first.key), std::hash<CacheRecordKey>{}(equalKey));

    std::unordered_map<CacheRecordKey, int32_t> keyedValues;
    EXPECT_TRUE(keyedValues.emplace(first.key, 1).second);
    EXPECT_TRUE(keyedValues.emplace(changedTerminal, 2).second);
    EXPECT_TRUE(keyedValues.emplace(changedCount, 3).second);
    EXPECT_EQ(keyedValues.size(), 3U);
    ASSERT_NE(keyedValues.find(first.key), keyedValues.end());
    EXPECT_EQ(keyedValues.find(first.key)->second, 1);
    ASSERT_NE(keyedValues.find(changedTerminal), keyedValues.end());
    EXPECT_EQ(keyedValues.find(changedTerminal)->second, 2);
    ASSERT_NE(keyedValues.find(changedCount), keyedValues.end());
    EXPECT_EQ(keyedValues.find(changedCount)->second, 3);
}

TEST(ContextCacheRecordStoreTests, HybridIndexUsesExactDigestAndLength)
{
    constexpr BlockHash kEXACT_DIGEST{0xD0D0D0D0D0D0D0D0ULL, 0xE0E0E0E0E0E0E0E0ULL};

    CacheRecord partial = makeRecord({kHASH_A}, {10});
    partial.key.terminalHash = kEXACT_DIGEST;
    partial.recurrentSnapshotSlot = 20;
    partial.partialKvSnapshotSlot = 21;
    partial.exactCheckpointLength = 6;

    CacheRecordStore store(4);
    RecordInsertResult const inserted = store.insert(partial);
    ASSERT_TRUE(inserted.inserted);
    HybridCheckpointKey const key{kEXACT_DIGEST, 6};
    EXPECT_EQ(store.findHybrid(key), std::optional<RecordId>{inserted.id});
    EXPECT_FALSE(store.findHybrid(HybridCheckpointKey{kEXACT_DIGEST, 7}).has_value());
    EXPECT_EQ(store.hybridCandidateLengths(7), std::vector<int32_t>{6});
    EXPECT_TRUE(store.hybridCandidateLengths(6).empty());

    CacheRecord pureRecurrent;
    pureRecurrent.key = CacheRecordKey{kHASH_B, 0};
    pureRecurrent.recurrentSnapshotSlot = 22;
    pureRecurrent.exactCheckpointLength = 3;
    RecordInsertResult const pureInserted = store.insert(pureRecurrent);
    ASSERT_TRUE(pureInserted.inserted);
    EXPECT_EQ(store.hybridCandidateLengths(8), std::vector<int32_t>({6, 3}));

    CacheRecord const erased = store.erase(inserted.id);
    EXPECT_EQ(erased.hybridKey(), std::optional<HybridCheckpointKey>{key});
    EXPECT_FALSE(store.findHybrid(key).has_value());
}

TEST(ContextCacheRecordStoreTests, InsertAndExplicitTouchMoveOnlyThatRecordToMru)
{
    CacheRecordStore store(2);
    RecordInsertResult const first = store.insert(makeRecord({kHASH_A}, {10}));
    RecordInsertResult const second = store.insert(makeRecord({kHASH_B}, {11}));
    RecordInsertResult const third = store.insert(makeRecord({kHASH_C}, {12}));
    ASSERT_TRUE(first.inserted);
    ASSERT_TRUE(second.inserted);
    ASSERT_TRUE(third.inserted);

    EXPECT_EQ(store.maxRecords(), 2);
    EXPECT_EQ(store.size(), 3U);
    EXPECT_EQ(store.lruToMru(), std::vector<RecordId>({first.id, second.id, third.id}));

    EXPECT_EQ(store.get(first.id).basePagePath, std::vector<PageId>{10});
    EXPECT_EQ(store.lruToMru(), std::vector<RecordId>({first.id, second.id, third.id}));

    store.touch(first.id);
    EXPECT_EQ(store.lruToMru(), std::vector<RecordId>({second.id, third.id, first.id}));

    store.touch(third.id);
    EXPECT_EQ(store.lruToMru(), std::vector<RecordId>({second.id, first.id, third.id}));

    store.touch(third.id);
    EXPECT_EQ(store.lruToMru(), std::vector<RecordId>({second.id, first.id, third.id}));

    EXPECT_THROW((void) store.get(RecordId{}), std::runtime_error);
    EXPECT_THROW(store.touch(RecordId{}), std::runtime_error);
    EXPECT_EQ(store.lruToMru(), std::vector<RecordId>({second.id, first.id, third.id}));
}

TEST(ContextCacheRecordStoreTests, EraseRemovesExactKeyAndLruEntry)
{
    CacheRecordStore store(3);
    RecordInsertResult const first = store.insert(makeRecord({kHASH_A}, {10}));
    CacheRecord middleRecord = makeRecord({kHASH_A, kHASH_B}, {10, 11});
    middleRecord.specState = makePagedSpecState({20, 21});
    middleRecord.recurrentSnapshotSlot = 30;
    middleRecord.partialKvSnapshotSlot = 40;
    middleRecord.exactCheckpointLength = 64;
    CacheRecordKey const middleKey = middleRecord.key;
    RecordInsertResult const middle = store.insert(middleRecord);
    RecordInsertResult const last = store.insert(makeRecord({kHASH_C}, {12}));
    ASSERT_TRUE(first.inserted);
    ASSERT_TRUE(middle.inserted);
    ASSERT_TRUE(last.inserted);

    CacheRecord erased = store.erase(middle.id);

    EXPECT_EQ(erased.id, middle.id);
    EXPECT_TRUE(erased.key == middleKey);
    EXPECT_EQ(erased.logicalBlockHashes, std::vector<BlockHash>({kHASH_A, kHASH_B}));
    EXPECT_EQ(erased.basePagePath, std::vector<PageId>({10, 11}));
    EXPECT_EQ(specPagePath(erased), std::vector<PageId>({20, 21}));
    EXPECT_EQ(erased.recurrentSnapshotSlot, std::optional<int32_t>{30});
    EXPECT_EQ(erased.partialKvSnapshotSlot, std::optional<int32_t>{40});
    EXPECT_EQ(erased.exactCheckpointLength, std::optional<int32_t>{64});
    EXPECT_FALSE(store.find(middleKey).has_value());
    EXPECT_EQ(store.size(), 2U);
    EXPECT_EQ(store.lruToMru(), std::vector<RecordId>({first.id, last.id}));
    EXPECT_THROW((void) store.get(middle.id), std::runtime_error);
    EXPECT_THROW((void) store.erase(middle.id), std::runtime_error);

    RecordInsertResult const reinserted = store.insert(erased);
    EXPECT_TRUE(reinserted.inserted);
    EXPECT_NE(reinserted.id, middle.id);
    EXPECT_EQ(store.find(middleKey), std::optional<RecordId>{reinserted.id});
    EXPECT_EQ(store.get(reinserted.id).basePagePath, std::vector<PageId>({10, 11}));
    EXPECT_EQ(store.lruToMru(), std::vector<RecordId>({first.id, last.id, reinserted.id}));
}

TEST(ContextCacheRecordStoreTests, BranchRecordsRetainSharedAncestorsIndependently)
{
    CacheRecordStore store(2);
    CacheRecord firstBranchRecord = makeRecord({kHASH_A, kHASH_B, kHASH_C}, {10, 11, 12});
    CacheRecord secondBranchRecord = makeRecord({kHASH_A, kHASH_B, kHASH_D}, {10, 11, 13});
    CacheRecordKey const firstBranchKey = firstBranchRecord.key;
    CacheRecordKey const secondBranchKey = secondBranchRecord.key;

    RecordInsertResult const firstBranch = store.insert(firstBranchRecord);
    RecordInsertResult const secondBranch = store.insert(secondBranchRecord);
    ASSERT_TRUE(firstBranch.inserted);
    ASSERT_TRUE(secondBranch.inserted);

    EXPECT_EQ(store.get(firstBranch.id).logicalBlockHashes, std::vector<BlockHash>({kHASH_A, kHASH_B, kHASH_C}));
    EXPECT_EQ(store.get(firstBranch.id).basePagePath, std::vector<PageId>({10, 11, 12}));
    EXPECT_EQ(store.get(secondBranch.id).logicalBlockHashes, std::vector<BlockHash>({kHASH_A, kHASH_B, kHASH_D}));
    EXPECT_EQ(store.get(secondBranch.id).basePagePath, std::vector<PageId>({10, 11, 13}));

    CacheRecord const erased = store.erase(firstBranch.id);

    EXPECT_EQ(erased.basePagePath, std::vector<PageId>({10, 11, 12}));
    EXPECT_FALSE(store.find(firstBranchKey).has_value());
    EXPECT_EQ(store.find(secondBranchKey), std::optional<RecordId>{secondBranch.id});
    EXPECT_EQ(store.get(secondBranch.id).logicalBlockHashes, std::vector<BlockHash>({kHASH_A, kHASH_B, kHASH_D}));
    EXPECT_EQ(store.get(secondBranch.id).basePagePath, std::vector<PageId>({10, 11, 13}));
    std::vector<ResourceId> const survivingResources{
        {ResourceType::kBaseKvPage, 10}, {ResourceType::kBaseKvPage, 11}, {ResourceType::kBaseKvPage, 13}};
    EXPECT_EQ(store.get(secondBranch.id).resources(), survivingResources);
    EXPECT_EQ(store.lruToMru(), std::vector<RecordId>{secondBranch.id});
    EXPECT_EQ(store.size(), 1U);
}

TEST(ContextCacheRecordStoreTests, AddingSpecStatePreservesBaseIdentityAndPromotesRecord)
{
    CacheRecordStore store(2);
    RecordInsertResult const first = store.insert(makeRecord({kHASH_A, kHASH_B}, {10, 11}));
    RecordInsertResult const second = store.insert(makeRecord({kHASH_C}, {12}));
    ASSERT_TRUE(first.inserted);
    ASSERT_TRUE(second.inserted);
    ASSERT_EQ(store.lruToMru(), std::vector<RecordId>({first.id, second.id}));

    EXPECT_THROW(store.setSpecState(first.id, makePagedSpecState({20})), std::runtime_error);
    EXPECT_THROW(store.setSpecState(first.id, makePagedSpecState({20, -1})), std::runtime_error);
    EXPECT_FALSE(store.get(first.id).specState.has_value());

    store.setSpecState(first.id, makePagedSpecState({20, 21}));

    CacheRecord const& upgraded = store.get(first.id);
    EXPECT_EQ(upgraded.key, (CacheRecordKey{kHASH_B, 2}));
    EXPECT_EQ(upgraded.logicalBlockHashes, std::vector<BlockHash>({kHASH_A, kHASH_B}));
    EXPECT_EQ(upgraded.basePagePath, std::vector<PageId>({10, 11}));
    EXPECT_EQ(specPagePath(upgraded), std::vector<PageId>({20, 21}));
    EXPECT_EQ(store.lruToMru(), std::vector<RecordId>({second.id, first.id}));

    store.setSpecState(first.id, makePagedSpecState({20, 21}));
    EXPECT_THROW(store.setSpecState(first.id, makePagedSpecState({30, 31})), std::runtime_error);
    EXPECT_EQ(specPagePath(store.get(first.id)), std::vector<PageId>({20, 21}));
    EXPECT_EQ(store.get(first.id).basePagePath, std::vector<PageId>({10, 11}));
}
