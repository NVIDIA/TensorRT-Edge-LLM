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

constexpr CacheDomainId kDOMAIN{0x1010101010101010ULL, 0x2020202020202020ULL};
constexpr CacheDomainId kOTHER_DOMAIN{0x3030303030303030ULL, 0x4040404040404040ULL};
constexpr DraftEngineSignature kDRAFT_SIGNATURE{0x5050505050505050ULL, 0x6060606060606060ULL};
constexpr DraftEngineSignature kOTHER_DRAFT_SIGNATURE{0x7070707070707070ULL, 0x8080808080808080ULL};
constexpr BlockHash kHASH_A{0x1111111111111111ULL, 0xAAAAAAAAAAAAAAAAULL};
constexpr BlockHash kHASH_B{0x2222222222222222ULL, 0xBBBBBBBBBBBBBBBBULL};
constexpr BlockHash kHASH_C{0x3333333333333333ULL, 0xCCCCCCCCCCCCCCCCULL};
constexpr BlockHash kHASH_D{0x4444444444444444ULL, 0xDDDDDDDDDDDDDDDDULL};

CacheRecord makeRecord(
    CacheDomainId domain, std::vector<BlockHash> logicalBlockHashes, std::vector<PageId> basePagePath)
{
    CacheRecord record;
    record.key = CacheRecordKey{domain, logicalBlockHashes.back(), static_cast<int32_t>(logicalBlockHashes.size())};
    record.logicalBlockHashes = std::move(logicalBlockHashes);
    record.basePagePath = std::move(basePagePath);
    record.baseFullBlockCount = static_cast<int32_t>(record.basePagePath.size());
    return record;
}

} // namespace

TEST(ContextCacheRecordStoreTests, RecordOwnsItsCompleteBasePath)
{
    EXPECT_THROW((void) CacheRecordStore(-1), std::runtime_error);

    CacheRecordStore store(0);
    EXPECT_EQ(store.maxRecords(), 0);

    CacheRecord record = makeRecord(kDOMAIN, {kHASH_A, kHASH_B, kHASH_C}, {10, 11, 12});
    constexpr RecordId kCALLER_RECORD_ID = 91;
    record.id = kCALLER_RECORD_ID;
    record.draftSignature = kDRAFT_SIGNATURE;
    record.draftPagePath = {20, 21};
    record.recurrentSnapshotSlot = 30;
    record.partialKvSnapshotSlot = 40;
    record.baseFullBlockCount = 2;
    record.pairedDraftFullBlockCount = 1;
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
    record.draftPagePath.assign({91});
    record.recurrentSnapshotSlot = 92;
    record.partialKvSnapshotSlot = 93;

    CacheRecord const& stored = store.get(inserted.id);
    EXPECT_EQ(stored.id, inserted.id);
    EXPECT_EQ(stored.logicalBlockHashes, std::vector<BlockHash>({kHASH_A, kHASH_B, kHASH_C}));
    EXPECT_EQ(stored.basePagePath, std::vector<PageId>({10, 11, 12}));
    EXPECT_EQ(stored.draftSignature, std::optional<DraftEngineSignature>{kDRAFT_SIGNATURE});
    EXPECT_EQ(stored.draftPagePath, std::vector<PageId>({20, 21}));
    EXPECT_EQ(stored.recurrentSnapshotSlot, std::optional<int32_t>{30});
    EXPECT_EQ(stored.partialKvSnapshotSlot, std::optional<int32_t>{40});
    EXPECT_EQ(stored.exactCheckpointLength, std::optional<int32_t>{96});
    std::vector<ResourceId> const expectedResources{{ResourceType::kBaseKvPage, 10}, {ResourceType::kBaseKvPage, 11},
        {ResourceType::kBaseKvPage, 12}, {ResourceType::kDraftKvPage, 20}, {ResourceType::kDraftKvPage, 21},
        {ResourceType::kRecurrentSnapshot, 30}, {ResourceType::kPartialKvSnapshot, 40}};
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

    invalid = valid;
    invalid.key.terminalHash = kHASH_D;
    expectInvalid(std::move(invalid));

    invalid = valid;
    invalid.baseFullBlockCount = -1;
    expectInvalid(std::move(invalid));

    invalid = valid;
    invalid.pairedDraftFullBlockCount = -1;
    expectInvalid(std::move(invalid));

    invalid = valid;
    invalid.basePagePath = {10};
    invalid.baseFullBlockCount = 2;
    expectInvalid(std::move(invalid));

    invalid = valid;
    invalid.basePagePath = {10, 11, 12, 13};
    invalid.baseFullBlockCount = 4;
    expectInvalid(std::move(invalid));

    invalid = valid;
    invalid.draftPagePath = {20};
    invalid.pairedDraftFullBlockCount = 2;
    expectInvalid(std::move(invalid));

    invalid = valid;
    invalid.draftPagePath = {20, 21, 22, 23};
    invalid.pairedDraftFullBlockCount = 4;
    expectInvalid(std::move(invalid));

    invalid = valid;
    invalid.basePagePath[1] = -1;
    expectInvalid(std::move(invalid));

    invalid = valid;
    invalid.draftPagePath[1] = -1;
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
    CacheRecord first = makeRecord(kDOMAIN, {kHASH_A, kHASH_B}, {10, 11});
    first.recurrentSnapshotSlot = 12;
    RecordInsertResult const firstInsert = store.insert(first);
    ASSERT_TRUE(firstInsert.inserted);

    CacheRecord other = makeRecord(kOTHER_DOMAIN, {kHASH_D}, {20});
    RecordInsertResult const otherInsert = store.insert(other);
    ASSERT_TRUE(otherInsert.inserted);
    EXPECT_EQ(store.lruToMru(), std::vector<RecordId>({firstInsert.id, otherInsert.id}));

    CacheRecord duplicate = makeRecord(kDOMAIN, {kHASH_A, kHASH_B}, {90, 91});
    duplicate.id = 999;
    duplicate.draftSignature = kDRAFT_SIGNATURE;
    duplicate.draftPagePath = {92};
    duplicate.pairedDraftFullBlockCount = 1;
    duplicate.partialKvSnapshotSlot = 93;
    duplicate.exactCheckpointLength = 94;

    RecordInsertResult const duplicateInsert = store.insert(std::move(duplicate));

    EXPECT_FALSE(duplicateInsert.inserted);
    EXPECT_EQ(duplicateInsert.id, firstInsert.id);
    EXPECT_EQ(store.size(), 2U);
    EXPECT_EQ(store.lruToMru(), std::vector<RecordId>({otherInsert.id, firstInsert.id}));
    CacheRecord const& stored = store.get(firstInsert.id);
    EXPECT_EQ(stored.basePagePath, std::vector<PageId>({10, 11}));
    EXPECT_FALSE(stored.draftSignature.has_value());
    EXPECT_TRUE(stored.draftPagePath.empty());
    EXPECT_EQ(stored.recurrentSnapshotSlot, std::optional<int32_t>{12});
    EXPECT_FALSE(stored.partialKvSnapshotSlot.has_value());
    EXPECT_FALSE(stored.exactCheckpointLength.has_value());
    EXPECT_EQ(store.find(first.key), std::optional<RecordId>{firstInsert.id});
    EXPECT_FALSE(store.find(CacheRecordKey{kOTHER_DOMAIN, kHASH_C, 1}).has_value());

    CacheRecordKey const changedDomain{kOTHER_DOMAIN, first.key.terminalHash, first.key.fullBlockCount};
    CacheRecordKey const changedTerminal{first.key.domain, kHASH_C, first.key.fullBlockCount};
    CacheRecordKey const changedCount{first.key.domain, first.key.terminalHash, first.key.fullBlockCount + 1};
    CacheRecordKey const equalKey = first.key;
    EXPECT_TRUE(first.key == equalKey);
    EXPECT_FALSE(first.key == changedDomain);
    EXPECT_FALSE(first.key == changedTerminal);
    EXPECT_FALSE(first.key == changedCount);
    EXPECT_EQ(std::hash<CacheRecordKey>{}(first.key), std::hash<CacheRecordKey>{}(equalKey));

    std::unordered_map<CacheRecordKey, int32_t> keyedValues;
    EXPECT_TRUE(keyedValues.emplace(first.key, 1).second);
    EXPECT_TRUE(keyedValues.emplace(changedDomain, 2).second);
    EXPECT_TRUE(keyedValues.emplace(changedTerminal, 3).second);
    EXPECT_TRUE(keyedValues.emplace(changedCount, 4).second);
    EXPECT_EQ(keyedValues.size(), 4U);
    ASSERT_NE(keyedValues.find(first.key), keyedValues.end());
    EXPECT_EQ(keyedValues.find(first.key)->second, 1);
    ASSERT_NE(keyedValues.find(changedDomain), keyedValues.end());
    EXPECT_EQ(keyedValues.find(changedDomain)->second, 2);
    ASSERT_NE(keyedValues.find(changedTerminal), keyedValues.end());
    EXPECT_EQ(keyedValues.find(changedTerminal)->second, 3);
    ASSERT_NE(keyedValues.find(changedCount), keyedValues.end());
    EXPECT_EQ(keyedValues.find(changedCount)->second, 4);
}

TEST(ContextCacheRecordStoreTests, HybridIndexUsesExactDigestLengthAndSchema)
{
    constexpr RecurrentStateSchemaId kSCHEMA{0x9090909090909090ULL, 0xA0A0A0A0A0A0A0A0ULL};
    constexpr RecurrentStateSchemaId kOTHER_SCHEMA{0xB0B0B0B0B0B0B0B0ULL, 0xC0C0C0C0C0C0C0C0ULL};
    constexpr BlockHash kEXACT_DIGEST{0xD0D0D0D0D0D0D0D0ULL, 0xE0E0E0E0E0E0E0E0ULL};

    CacheRecord partial = makeRecord(kDOMAIN, {kHASH_A}, {10});
    partial.key.terminalHash = kEXACT_DIGEST;
    partial.recurrentSnapshotSlot = 20;
    partial.partialKvSnapshotSlot = 21;
    partial.exactCheckpointLength = 6;
    partial.exactCheckpointDigest = kEXACT_DIGEST;
    partial.recurrentStateSchema = kSCHEMA;

    CacheRecordStore store(4);
    RecordInsertResult const inserted = store.insert(partial);
    ASSERT_TRUE(inserted.inserted);
    HybridCheckpointKey const key{kDOMAIN, kEXACT_DIGEST, 6, kSCHEMA};
    EXPECT_EQ(store.findHybrid(key), std::optional<RecordId>{inserted.id});
    EXPECT_FALSE(store.findHybrid(HybridCheckpointKey{kDOMAIN, kEXACT_DIGEST, 6, kOTHER_SCHEMA}).has_value());
    EXPECT_EQ(store.hybridCandidateLengths(kDOMAIN, kSCHEMA, 7), std::vector<int32_t>{6});
    EXPECT_TRUE(store.hybridCandidateLengths(kDOMAIN, kSCHEMA, 6).empty());

    CacheRecord pureRecurrent;
    pureRecurrent.key = CacheRecordKey{kDOMAIN, kHASH_B, 0};
    pureRecurrent.recurrentSnapshotSlot = 22;
    pureRecurrent.exactCheckpointLength = 3;
    pureRecurrent.exactCheckpointDigest = kHASH_B;
    pureRecurrent.recurrentStateSchema = kSCHEMA;
    RecordInsertResult const pureInserted = store.insert(pureRecurrent);
    ASSERT_TRUE(pureInserted.inserted);
    EXPECT_EQ(store.hybridCandidateLengths(kDOMAIN, kSCHEMA, 8), std::vector<int32_t>({6, 3}));

    CacheRecord const erased = store.erase(inserted.id);
    EXPECT_EQ(erased.hybridKey(), std::optional<HybridCheckpointKey>{key});
    EXPECT_FALSE(store.findHybrid(key).has_value());
}

TEST(ContextCacheRecordStoreTests, HybridMtpIdentityMergesSuccessorVariantsAndKeepsHybridEndpointSeparate)
{
    constexpr RecurrentStateSchemaId kSCHEMA{0x9191919191919191ULL, 0xA1A1A1A1A1A1A1A1ULL};
    constexpr BlockHash kEXACT_DIGEST{0xD1D1D1D1D1D1D1D1ULL, 0xE1E1E1E1E1E1E1E1ULL};

    CacheRecord hybrid = makeRecord(kDOMAIN, {kHASH_A}, {10});
    hybrid.key = CacheRecordKey{kDOMAIN, kEXACT_DIGEST, 1, CacheRecordKind::kHybrid};
    hybrid.recurrentSnapshotSlot = 20;
    hybrid.partialKvSnapshotSlot = 30;
    hybrid.exactCheckpointLength = 6;
    hybrid.exactCheckpointDigest = kEXACT_DIGEST;
    hybrid.recurrentStateSchema = kSCHEMA;

    // Two Hybrid+MTP checkpoints over the same prefix that used to differ only in the token that followed the
    // checkpoint. The successor is no longer part of identity (it is recomputed at restore from the saved boundary
    // hidden), so the two now share one Hybrid+MTP key and the second insert reuses the first record.
    CacheRecord firstMtp = hybrid;
    firstMtp.key.kind = CacheRecordKind::kHybridMtp;
    firstMtp.basePagePath = {11};
    firstMtp.draftSignature = kDRAFT_SIGNATURE;
    firstMtp.draftPagePath = {40};
    firstMtp.pairedDraftFullBlockCount = 1;
    firstMtp.recurrentSnapshotSlot = 21;
    firstMtp.partialKvSnapshotSlot = 31;

    CacheRecord secondMtp = firstMtp;
    secondMtp.basePagePath = {12};
    secondMtp.draftPagePath = {41};
    secondMtp.recurrentSnapshotSlot = 22;
    secondMtp.partialKvSnapshotSlot = 32;

    CacheRecordStore store(4);
    RecordInsertResult const hybridInserted = store.insert(hybrid);
    RecordInsertResult const firstInserted = store.insert(firstMtp);
    RecordInsertResult const secondInserted = store.insert(secondMtp);
    ASSERT_TRUE(hybridInserted.inserted);
    ASSERT_TRUE(firstInserted.inserted);
    // Same (domain, digest, length, schema, draft-signature) identity as firstMtp -> reuses it, no new record.
    EXPECT_FALSE(secondInserted.inserted);
    EXPECT_EQ(secondInserted.id, firstInserted.id);
    EXPECT_EQ(store.size(), 2U);

    HybridCheckpointKey const hybridKey{kDOMAIN, kEXACT_DIGEST, 6, kSCHEMA};
    HybridMtpCheckpointKey const mtpKey{kDOMAIN, kEXACT_DIGEST, 6, kSCHEMA, kDRAFT_SIGNATURE};
    EXPECT_EQ(store.findHybrid(hybridKey), std::optional<RecordId>{hybridInserted.id});
    EXPECT_EQ(store.findHybridMtp(mtpKey), std::optional<RecordId>{firstInserted.id});
    EXPECT_EQ(store.hybridMtpCandidateLengths(kDOMAIN, kSCHEMA, kDRAFT_SIGNATURE, 7), std::vector<int32_t>{6});
    EXPECT_TRUE(store.hybridMtpCandidateLengths(kDOMAIN, kSCHEMA, kDRAFT_SIGNATURE, 6).empty());
    EXPECT_EQ(std::hash<HybridMtpCheckpointKey>{}(mtpKey), std::hash<HybridMtpCheckpointKey>{}(mtpKey));

    // The pure-hybrid endpoint over the same digest keeps its own identity and is unaffected by erasing the MTP record.
    CacheRecord const erased = store.erase(firstInserted.id);
    EXPECT_EQ(erased.hybridMtpKey(), std::optional<HybridMtpCheckpointKey>{mtpKey});
    EXPECT_FALSE(store.findHybridMtp(mtpKey).has_value());
    EXPECT_EQ(store.findHybrid(hybridKey), std::optional<RecordId>{hybridInserted.id});
}

TEST(ContextCacheRecordStoreTests, InsertAndExplicitTouchMoveOnlyThatRecordToMru)
{
    CacheRecordStore store(2);
    RecordInsertResult const first = store.insert(makeRecord(kDOMAIN, {kHASH_A}, {10}));
    RecordInsertResult const second = store.insert(makeRecord(kDOMAIN, {kHASH_B}, {11}));
    RecordInsertResult const third = store.insert(makeRecord(kDOMAIN, {kHASH_C}, {12}));
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
    RecordInsertResult const first = store.insert(makeRecord(kDOMAIN, {kHASH_A}, {10}));
    CacheRecord middleRecord = makeRecord(kDOMAIN, {kHASH_A, kHASH_B}, {10, 11});
    middleRecord.draftSignature = kDRAFT_SIGNATURE;
    middleRecord.draftPagePath = {20, 21};
    middleRecord.pairedDraftFullBlockCount = 2;
    middleRecord.recurrentSnapshotSlot = 30;
    middleRecord.partialKvSnapshotSlot = 40;
    middleRecord.exactCheckpointLength = 64;
    CacheRecordKey const middleKey = middleRecord.key;
    RecordInsertResult const middle = store.insert(middleRecord);
    RecordInsertResult const last = store.insert(makeRecord(kDOMAIN, {kHASH_C}, {12}));
    ASSERT_TRUE(first.inserted);
    ASSERT_TRUE(middle.inserted);
    ASSERT_TRUE(last.inserted);

    CacheRecord erased = store.erase(middle.id);

    EXPECT_EQ(erased.id, middle.id);
    EXPECT_TRUE(erased.key == middleKey);
    EXPECT_EQ(erased.logicalBlockHashes, std::vector<BlockHash>({kHASH_A, kHASH_B}));
    EXPECT_EQ(erased.basePagePath, std::vector<PageId>({10, 11}));
    EXPECT_EQ(erased.draftSignature, std::optional<DraftEngineSignature>{kDRAFT_SIGNATURE});
    EXPECT_EQ(erased.draftPagePath, std::vector<PageId>({20, 21}));
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
    CacheRecord firstBranchRecord = makeRecord(kDOMAIN, {kHASH_A, kHASH_B, kHASH_C}, {10, 11, 12});
    CacheRecord secondBranchRecord = makeRecord(kDOMAIN, {kHASH_A, kHASH_B, kHASH_D}, {10, 11, 13});
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

TEST(ContextCacheRecordStoreTests, ReplacingDraftStatePreservesBaseIdentityAndPromotesRecord)
{
    CacheRecordStore store(2);
    RecordInsertResult const first = store.insert(makeRecord(kDOMAIN, {kHASH_A, kHASH_B}, {10, 11}));
    RecordInsertResult const second = store.insert(makeRecord(kDOMAIN, {kHASH_C}, {12}));
    ASSERT_TRUE(first.inserted);
    ASSERT_TRUE(second.inserted);
    ASSERT_EQ(store.lruToMru(), std::vector<RecordId>({first.id, second.id}));

    store.setDraftState(first.id, kDRAFT_SIGNATURE, {20, 21}, 2);

    CacheRecord const& upgraded = store.get(first.id);
    EXPECT_EQ(upgraded.key, (CacheRecordKey{kDOMAIN, kHASH_B, 2}));
    EXPECT_EQ(upgraded.logicalBlockHashes, std::vector<BlockHash>({kHASH_A, kHASH_B}));
    EXPECT_EQ(upgraded.basePagePath, std::vector<PageId>({10, 11}));
    EXPECT_EQ(upgraded.draftSignature, std::optional<DraftEngineSignature>{kDRAFT_SIGNATURE});
    EXPECT_EQ(upgraded.draftPagePath, std::vector<PageId>({20, 21}));
    EXPECT_EQ(upgraded.pairedDraftFullBlockCount, 2);
    EXPECT_EQ(store.lruToMru(), std::vector<RecordId>({second.id, first.id}));

    EXPECT_THROW(store.setDraftState(first.id, kOTHER_DRAFT_SIGNATURE, {30, -1}, 2), std::runtime_error);
    EXPECT_EQ(store.get(first.id).draftSignature, std::optional<DraftEngineSignature>{kDRAFT_SIGNATURE});
    EXPECT_EQ(store.get(first.id).draftPagePath, std::vector<PageId>({20, 21}));

    store.setDraftState(first.id, kOTHER_DRAFT_SIGNATURE, {30, 31}, 2);

    EXPECT_EQ(store.get(first.id).draftSignature, std::optional<DraftEngineSignature>{kOTHER_DRAFT_SIGNATURE});
    EXPECT_EQ(store.get(first.id).draftPagePath, std::vector<PageId>({30, 31}));
    EXPECT_EQ(store.get(first.id).basePagePath, std::vector<PageId>({10, 11}));
}
