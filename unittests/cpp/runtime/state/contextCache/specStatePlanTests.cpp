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

#include "runtime/state/contextCache/specStatePlan.h"

#include "runtime/state/contextCache/blockIndex.h"
#include "runtime/state/contextCache/cacheRecord.h"
#include "runtime/state/contextCache/reusePlan.h"
#include "runtime/state/contextCache/specReuseContract.h"
#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <utility>
#include <vector>

using namespace trt_edgellm::rt;

namespace
{

struct ContractExpectation
{
    SpecReuseContract contract;
    int32_t replayPageCount;
};

constexpr int32_t kPAGE_SIZE = 128;
constexpr BlockHash kHASH_A{0x1111111111111111ULL, 0xAAAAAAAAAAAAAAAAULL};
constexpr BlockHash kHASH_B{0x2222222222222222ULL, 0xBBBBBBBBBBBBBBBBULL};
constexpr SpecReuseContract kEAGLE_CONTRACT{/*ownsPagedSpecState=*/true, /*futureDependencyTokens=*/1};
constexpr SpecReuseContract kGEMMA4_MTP_CONTRACT{/*ownsPagedSpecState=*/false, /*futureDependencyTokens=*/0};
constexpr SpecReuseContract kBLOCK_DRAFT_CONTRACT{/*ownsPagedSpecState=*/true, /*futureDependencyTokens=*/0};

RecordId insertSpecRecord(CacheRecordStore& records, SpecStateIndex& specIndex)
{
    CacheRecord record;
    record.key = CacheRecordKey{kHASH_B, 2};
    record.logicalBlockHashes = {kHASH_A, kHASH_B};
    record.basePagePath = {5, 7};
    record.specState = SpecPagedStateRecord{{11, 13}};
    RecordInsertResult const inserted = records.insert(std::move(record));
    specIndex.paged().insert(records.get(inserted.id));
    return inserted.id;
}

void expectCorruptIndexedRecordRejected(
    std::function<void(CacheRecord&)> const& corrupt, std::string const& expectedMessage)
{
    BaseBlockIndex baseIndex;
    EXPECT_TRUE(baseIndex.insert(kHASH_A, 5).inserted);
    EXPECT_TRUE(baseIndex.insert(kHASH_B, 7).inserted);
    SpecStateIndex specIndex;
    CacheRecordStore records(4);
    RecordId const recordId = insertSpecRecord(records, specIndex);
    corrupt(const_cast<CacheRecord&>(records.get(recordId)));

    try
    {
        static_cast<void>(makeSpecReusePlan(SpecReusePlanInput{{kHASH_A, kHASH_B}, 2 * kPAGE_SIZE + 1, kPAGE_SIZE,
                                                ContextCacheLookupPolicy::kUseCache, baseIndex, specIndex, records},
            kEAGLE_CONTRACT));
        FAIL() << "Expected corrupt indexed spec-state metadata to be rejected";
    }
    catch (std::runtime_error const& error)
    {
        EXPECT_NE(std::string{error.what()}.find(expectedMessage), std::string::npos) << error.what();
    }
}

} // namespace

TEST(ContextCacheSpecStatePlanTests, ContractRowsDriveReuseConsequences)
{
    constexpr std::array<ContractExpectation, 3> kEXPECTATIONS{{
        {kEAGLE_CONTRACT, 1},
        {kGEMMA4_MTP_CONTRACT, 0},
        {kBLOCK_DRAFT_CONTRACT, 0},
    }};

    for (ContractExpectation const& expected : kEXPECTATIONS)
    {
        EXPECT_EQ(replayPages(expected.contract, kPAGE_SIZE), expected.replayPageCount);
    }
}

TEST(ContextCacheSpecStatePlanTests, UnknownSpecDecodeModeIsRejected)
{
    DeploymentConfig deployment;
    deployment.base.specDecodeType = static_cast<SpecDecodeMode>(std::numeric_limits<int32_t>::max());
    deployment.draft = LLMEngineConfig{};
    deployment.specConfig = SpecDecodeConfig{};

    EXPECT_THROW(resolveSpecReuseContract(deployment), std::runtime_error);
}

TEST(ContextCacheSpecStatePlanTests, Gemma4MTPConsumesBaseHitWithoutSpecResources)
{
    BaseBlockIndex baseIndex;
    EXPECT_TRUE(baseIndex.insert(kHASH_A, 5).inserted);
    EXPECT_TRUE(baseIndex.insert(kHASH_B, 7).inserted);
    SpecStateIndex specIndex;
    CacheRecordStore records(4);
    SpecReuseContract const& contract = kGEMMA4_MTP_CONTRACT;

    ReusePlan const plan = makeSpecReusePlan(SpecReusePlanInput{{kHASH_A, kHASH_B}, 2 * kPAGE_SIZE + 1, kPAGE_SIZE,
                                                 ContextCacheLookupPolicy::kUseCache, baseIndex, specIndex, records},
        contract);

    EXPECT_EQ(plan.matchedTokenLength, 2 * kPAGE_SIZE);
    EXPECT_EQ(plan.reuseTokenLength, 2 * kPAGE_SIZE);
    EXPECT_EQ(plan.basePageBindings, std::vector<PageId>({5, 7}));
    EXPECT_TRUE(plan.specPageBindings.empty());
    EXPECT_EQ(plan.demand.baseKvPages, 1);
    EXPECT_EQ(plan.demand.draftKvPages, 0);

    std::vector<PageId> const noPages;
    SpecLeaseStateView const leaseState{noPages};
    EXPECT_EQ(makeSpecPublishedState(SpecPublishStateInput{leaseState, {kHASH_A, kHASH_B}, 2 * kPAGE_SIZE}, contract),
        std::nullopt);
}

TEST(ContextCacheSpecStatePlanTests, EagleRetainsOnePageReplayBehavior)
{
    BaseBlockIndex baseIndex;
    EXPECT_TRUE(baseIndex.insert(kHASH_A, 5).inserted);
    EXPECT_TRUE(baseIndex.insert(kHASH_B, 7).inserted);
    SpecStateIndex specIndex;
    CacheRecordStore records(4);
    RecordId const recordId = insertSpecRecord(records, specIndex);
    SpecReuseContract const& contract = kEAGLE_CONTRACT;

    ReusePlan const plan = makeSpecReusePlan(SpecReusePlanInput{{kHASH_A, kHASH_B}, 2 * kPAGE_SIZE + 1, kPAGE_SIZE,
                                                 ContextCacheLookupPolicy::kUseCache, baseIndex, specIndex, records},
        contract);

    EXPECT_EQ(plan.matchedTokenLength, 2 * kPAGE_SIZE);
    EXPECT_EQ(plan.reuseTokenLength, kPAGE_SIZE);
    EXPECT_EQ(plan.basePageBindings, std::vector<PageId>({5}));
    EXPECT_EQ(plan.specPageBindings, std::vector<PageId>({11}));
    EXPECT_EQ(plan.specRecord, std::optional<RecordId>{recordId});
    EXPECT_EQ(plan.demand.baseKvPages, 2);
    EXPECT_EQ(plan.demand.draftKvPages, 2);
    EXPECT_EQ(plan.specReplayMode, SpecReplayMode::kFullPage);
}

TEST(ContextCacheSpecStatePlanTests, EagleDoesNotReportReplayWhenRewindLeavesNoReusablePrefix)
{
    BaseBlockIndex baseIndex;
    EXPECT_TRUE(baseIndex.insert(kHASH_A, 5).inserted);
    SpecStateIndex specIndex;
    CacheRecordStore records(4);
    CacheRecord record;
    record.key = CacheRecordKey{kHASH_A, 1};
    record.logicalBlockHashes = {kHASH_A};
    record.basePagePath = {5};
    record.specState = SpecPagedStateRecord{{11}};
    RecordInsertResult const inserted = records.insert(std::move(record));
    specIndex.paged().insert(records.get(inserted.id));

    ReusePlan const plan = makeSpecReusePlan(SpecReusePlanInput{{kHASH_A}, kPAGE_SIZE + 1, kPAGE_SIZE,
                                                 ContextCacheLookupPolicy::kUseCache, baseIndex, specIndex, records},
        kEAGLE_CONTRACT);

    EXPECT_EQ(plan.matchedTokenLength, kPAGE_SIZE);
    EXPECT_EQ(plan.reuseTokenLength, 0);
    EXPECT_TRUE(plan.basePageBindings.empty());
    EXPECT_EQ(plan.specReplayMode, SpecReplayMode::kNone);
}

TEST(ContextCacheSpecStatePlanTests, BaseOnlyHitWithoutSpecStateIsCold)
{
    BaseBlockIndex baseIndex;
    EXPECT_TRUE(baseIndex.insert(kHASH_A, 5).inserted);
    EXPECT_TRUE(baseIndex.insert(kHASH_B, 7).inserted);
    SpecStateIndex specIndex;
    CacheRecordStore records(4);
    SpecReuseContract const& contract = kEAGLE_CONTRACT;

    ReusePlan const plan = makeSpecReusePlan(SpecReusePlanInput{{kHASH_A, kHASH_B}, 2 * kPAGE_SIZE + 1, kPAGE_SIZE,
                                                 ContextCacheLookupPolicy::kUseCache, baseIndex, specIndex, records},
        contract);
    EXPECT_EQ(plan.reuseTokenLength, 0);
    EXPECT_TRUE(plan.basePageBindings.empty());
    EXPECT_TRUE(plan.specPageBindings.empty());
    EXPECT_EQ(plan.demand.baseKvPages, 3);
    EXPECT_EQ(plan.demand.draftKvPages, 3);
}

TEST(ContextCacheSpecStatePlanTests, BypassDoesNotBindMatchingSpecState)
{
    BaseBlockIndex baseIndex;
    EXPECT_TRUE(baseIndex.insert(kHASH_A, 5).inserted);
    EXPECT_TRUE(baseIndex.insert(kHASH_B, 7).inserted);
    SpecStateIndex specIndex;
    CacheRecordStore records(4);
    insertSpecRecord(records, specIndex);

    ReusePlan const plan = makeSpecReusePlan(SpecReusePlanInput{{kHASH_A, kHASH_B}, 2 * kPAGE_SIZE + 1, kPAGE_SIZE,
                                                 ContextCacheLookupPolicy::kBypass, baseIndex, specIndex, records},
        kEAGLE_CONTRACT);

    EXPECT_EQ(plan.reuseTokenLength, 0);
    EXPECT_TRUE(plan.basePageBindings.empty());
    EXPECT_TRUE(plan.specPageBindings.empty());
    EXPECT_FALSE(plan.specRecord.has_value());
    EXPECT_EQ(plan.specReplayMode, SpecReplayMode::kNone);
}

TEST(ContextCacheSpecStatePlanTests, PublishedSpecStateRejectsPageAndHashCountMismatch)
{
    std::vector<PageId> const pages{11};
    SpecLeaseStateView const leaseState{pages};
    EXPECT_THROW(
        makeSpecPublishedState(SpecPublishStateInput{leaseState, {kHASH_A, kHASH_B}, 2 * kPAGE_SIZE}, kEAGLE_CONTRACT),
        std::runtime_error);

    std::vector<PageId> const noPages;
    SpecLeaseStateView const emptyLeaseState{noPages};
    EXPECT_THROW(makeSpecPublishedState(SpecPublishStateInput{emptyLeaseState, {kHASH_A}, kPAGE_SIZE}, kEAGLE_CONTRACT),
        std::runtime_error);
    EXPECT_THROW(
        makeSpecPublishedState(SpecPublishStateInput{leaseState, {}, kPAGE_SIZE}, kEAGLE_CONTRACT), std::runtime_error);
}

TEST(ContextCacheSpecStatePlanTests, EveryContractRowDrivesPerPoolDemandWithoutMutatingMetadata)
{
    constexpr std::array<SpecReuseContract, 3> kCONTRACTS{kEAGLE_CONTRACT, kGEMMA4_MTP_CONTRACT, kBLOCK_DRAFT_CONTRACT};
    for (SpecReuseContract const& contract : kCONTRACTS)
    {
        BaseBlockIndex baseIndex;
        EXPECT_TRUE(baseIndex.insert(kHASH_A, 5).inserted);
        EXPECT_TRUE(baseIndex.insert(kHASH_B, 7).inserted);
        SpecStateIndex specIndex;
        CacheRecordStore records(4);
        if (contract.ownsPagedSpecState)
        {
            insertSpecRecord(records, specIndex);
        }
        size_t const baseIndexSize = baseIndex.size();
        std::optional<SpecPagedStateMatch> const specMatchBefore
            = specIndex.paged().lookupLongest({kHASH_A, kHASH_B}, 2);
        size_t const recordCount = records.size();
        std::vector<RecordId> const recency = records.lruToMru();

        ReusePlan const plan
            = makeSpecReusePlan(SpecReusePlanInput{{kHASH_A, kHASH_B}, 2 * kPAGE_SIZE + 1, kPAGE_SIZE,
                                    ContextCacheLookupPolicy::kUseCache, baseIndex, specIndex, records},
                contract);

        int32_t const replay = replayPages(contract, kPAGE_SIZE);
        EXPECT_EQ(plan.reuseTokenLength, (2 - replay) * kPAGE_SIZE);
        EXPECT_EQ(plan.demand.baseKvPages, 1 + replay);
        EXPECT_EQ(plan.demand.draftKvPages, contract.ownsPagedSpecState ? 1 + replay : 0);
        EXPECT_EQ(plan.specPageBindings.size(), contract.ownsPagedSpecState ? static_cast<size_t>(2 - replay) : 0U);
        EXPECT_EQ(baseIndex.size(), baseIndexSize);
        EXPECT_EQ(specIndex.paged().lookupLongest({kHASH_A, kHASH_B}, 2), specMatchBefore);
        EXPECT_EQ(records.size(), recordCount);
        EXPECT_EQ(records.lruToMru(), recency);
    }
}

TEST(ContextCacheSpecStatePlanTests, MissingIndexedRecordIsRejectedWithoutPartialBinding)
{
    BaseBlockIndex baseIndex;
    EXPECT_TRUE(baseIndex.insert(kHASH_A, 5).inserted);
    EXPECT_TRUE(baseIndex.insert(kHASH_B, 7).inserted);
    SpecStateIndex specIndex;
    CacheRecordStore records(4);
    RecordId const record = insertSpecRecord(records, specIndex);
    static_cast<void>(records.erase(record));
    SpecReuseContract const& contract = kEAGLE_CONTRACT;

    EXPECT_THROW(makeSpecReusePlan(SpecReusePlanInput{{kHASH_A, kHASH_B}, 2 * kPAGE_SIZE + 1, kPAGE_SIZE,
                                       ContextCacheLookupPolicy::kUseCache, baseIndex, specIndex, records},
                     contract),
        std::runtime_error);
    EXPECT_EQ(baseIndex.size(), 2U);
    EXPECT_TRUE(specIndex.paged().lookupLongest({kHASH_A, kHASH_B}, 2).has_value());
    EXPECT_EQ(records.size(), 0U);
}

TEST(ContextCacheSpecStatePlanTests, RejectsIndexedRecordWithoutSpecState)
{
    expectCorruptIndexedRecordRejected(
        [](CacheRecord& record) { record.specState.reset(); }, "does not describe a paged spec-state record");
}

TEST(ContextCacheSpecStatePlanTests, RejectsIndexedRecordWithShortSpecPagePath)
{
    expectCorruptIndexedRecordRejected([](CacheRecord& record) { record.specState->pagePath.resize(1); },
        "does not describe a coherent record prefix");
}

TEST(ContextCacheSpecStatePlanTests, RejectsIndexedRecordWithShortLogicalHashPath)
{
    expectCorruptIndexedRecordRejected(
        [](CacheRecord& record) { record.logicalBlockHashes.resize(1); }, "does not describe a coherent record prefix");
}

TEST(ContextCacheSpecStatePlanTests, RejectsIndexedRecordWithMismatchedTerminalHash)
{
    expectCorruptIndexedRecordRejected([](CacheRecord& record) { record.logicalBlockHashes.back() = kHASH_A; },
        "does not describe a coherent record prefix");
}
