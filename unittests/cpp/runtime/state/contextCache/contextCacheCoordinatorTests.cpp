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

#include "runtime/state/contextCache/contextCacheCoordinator.h"

#include "action/actionKvBatch.h"
#include "common/checkMacros.h"
#include "common/pagedKvTypes.h"
#include "common/tensor.h"
#include "runtime/hybridCacheManager.h"
#include "runtime/state/kvPageTable.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <memory>
#include <numeric>
#include <utility>
#include <vector>

using namespace trt_edgellm;
using namespace trt_edgellm::rt;

namespace
{

constexpr int32_t kMAX_BATCH{3};
constexpr int32_t kMAX_SEQUENCE_LENGTH{512};

std::vector<int32_t> makeTokens(int32_t count)
{
    std::vector<int32_t> tokens(static_cast<size_t>(count));
    std::iota(tokens.begin(), tokens.end(), 1);
    return tokens;
}

LLMEngineConfig makeEngineConfig()
{
    LLMEngineConfig config;
    config.modelType = "coordinator-test";
    config.hiddenSize = 64;
    config.numDecoderLayers = 1;
    config.numAttentionLayers = 1;
    config.numKVHeads = 1;
    config.headDim = 8;
    config.rotaryDim = 8;
    config.maxSupportedBatchSize = kMAX_BATCH;
    config.maxSupportedInputLength = kMAX_SEQUENCE_LENGTH;
    config.maxKVCacheCapacity = kMAX_SEQUENCE_LENGTH;
    int64_t const minimumActivePages = computeMinimumKvPoolPages(kMAX_BATCH, kMAX_SEQUENCE_LENGTH);
    ELLM_CHECK(minimumActivePages <= kMAX_KV_POOL_PAGES, "Test KV pool page count must fit int32.");
    config.kvPoolPages = static_cast<int32_t>(minimumActivePages);
    config.kvCacheDtype = DataType::kHALF;
    config.layerTypes = {HybridCacheManager::LayerType::kAttention};
    config.kvLayerConfigs = {KVLayerConfig{config.numKVHeads, config.headDim}};
    return config;
}

HybridCacheManager::Config makeCacheConfig(LLMEngineConfig const& engine)
{
    KVCacheManager::Config kvConfig{
        /*.numAttentionLayers=*/engine.numAttentionLayers,
        /*.maxBatchSize=*/engine.maxSupportedBatchSize,
        /*.maxSequenceLength=*/engine.maxKVCacheCapacity,
        /*.layerConfigs=*/engine.kvLayerConfigs,
        /*.kvCacheType=*/engine.kvCacheDtype,
        /*.numPages=*/engine.kvPoolPages,
    };
    MambaCacheManager::Config mambaConfig{
        /*.numRecurrentLayers=*/0,
        /*.maxBatchSize=*/engine.maxSupportedBatchSize,
    };
    return HybridCacheManager::Config{
        /*.layerTypes=*/engine.layerTypes,
        /*.kvConfig=*/std::move(kvConfig),
        /*.mambaConfig=*/std::move(mambaConfig),
        /*.maxBatchSize=*/engine.maxSupportedBatchSize,
    };
}

class ContextCacheCoordinatorTests : public ::testing::Test
{
protected:
    void SetUp() override
    {
        ASSERT_EQ(cudaStreamCreate(&mStream), cudaSuccess);
        mEngine = makeEngineConfig();
        mDeployment = DeploymentConfig{mEngine, std::nullopt, std::nullopt};
        mCache = std::make_unique<HybridCacheManager>(makeCacheConfig(mEngine), mStream);
        KVCacheManager const& kv = mCache->getKVCacheManager();
        mPageTable = std::make_unique<KVPageTable>(kMAX_BATCH, pagesPerSlot(kv.maxCapPadded()), kv.numPages());
        mPageTable->setIdentity();
        mPageTable->upload(mStream);
        ASSERT_EQ(cudaStreamSynchronize(mStream), cudaSuccess);
        createCoordinator();
    }

    void TearDown() override
    {
        if (mCoordinator != nullptr)
        {
            EXPECT_EQ(mCoordinator->shutdown(), ContextCacheCoordinatorStatus::kOk);
            mCoordinator.reset();
        }
        mPageTable.reset();
        mCache.reset();
        EXPECT_EQ(cudaStreamDestroy(mStream), cudaSuccess);
    }

    void createCoordinator(ContextCacheCoordinator::StreamSynchronizer synchronizer = {})
    {
        ContextCachePhysicalResources resources{*mCache, *mPageTable, nullptr, nullptr};
        mCoordinator
            = std::make_unique<ContextCacheCoordinator>(ContextCacheConfig{/*.enabled=*/true, /*.maxRecords=*/16},
                mDeployment, validateContextCacheDeployment(mDeployment), resources, mStream, std::move(synchronizer));
    }

    ContextCacheCoordinator::BeginRequestResult begin(std::vector<std::vector<int32_t>> const& batch,
        ContextCacheLookupPolicy lookupPolicy = ContextCacheLookupPolicy::kUseCache,
        ContextCacheCommitPolicy commitPolicy = ContextCacheCommitPolicy::kIncludingGeneratedTokens)
    {
        ContextCacheBatchAdmission admission;
        for (size_t slot = 0; slot < batch.size(); ++slot)
        {
            ContextCacheSequenceAdmission sequence{batch[slot], {}, {}, ResidentRef{static_cast<int32_t>(slot), 1}};
            sequence.lookupPolicy = lookupPolicy;
            sequence.commitPolicy = commitPolicy;
            admission.sequences.push_back(std::move(sequence));
        }
        return mCoordinator->beginRequest(admission, DecodingKvHeadroom{1, 0}, mStream);
    }

    void finalizePrefillWithLengths(
        ContextCacheCoordinator::AdmissionResult& admission, std::vector<int32_t> const& inputLengths)
    {
        ASSERT_EQ(inputLengths.size(), admission.prefillStarts.size());
        ASSERT_EQ(mCoordinator->preparePrefill(admission.request), ContextCacheCoordinatorStatus::kOk);
        ASSERT_EQ(cudaStreamSynchronize(mStream), cudaSuccess);
        std::vector<int32_t> lookahead(admission.prefillStarts.size(), 9001);
        std::vector<ContextCacheSequenceAdvance> progress;
        progress.reserve(admission.prefillStarts.size());
        for (size_t slot = 0; slot < admission.prefillStarts.size(); ++slot)
        {
            progress.push_back(ContextCacheSequenceAdvance{&lookahead[slot], 1, inputLengths[slot]});
        }
        ASSERT_EQ(
            mCoordinator->finalizePrefillPublication(admission.request, progress), ContextCacheCoordinatorStatus::kOk);
    }

    void finish(ContextCacheCoordinator::AdmissionResult& admission)
    {
        ASSERT_EQ(mCoordinator->finish(admission.request), ContextCacheCoordinatorStatus::kOk);
    }

    cudaStream_t mStream{};
    LLMEngineConfig mEngine;
    DeploymentConfig mDeployment;
    std::unique_ptr<HybridCacheManager> mCache;
    std::unique_ptr<KVPageTable> mPageTable;
    std::unique_ptr<ContextCacheCoordinator> mCoordinator;
};

TEST_F(ContextCacheCoordinatorTests, PublishesColdPrefixAndReusesLongestFullBlock)
{
    auto first = begin({makeTokens(129)});
    ASSERT_EQ(first.status, ContextCacheCoordinatorStatus::kOk);
    ASSERT_TRUE(first.admission.has_value());
    EXPECT_EQ(first.admission->prefillStarts[0], 0);
    finalizePrefillWithLengths(*first.admission, {129});
    finish(*first.admission);

    auto second = begin({makeTokens(130)});
    ASSERT_EQ(second.status, ContextCacheCoordinatorStatus::kOk);
    ASSERT_TRUE(second.admission.has_value());
    EXPECT_EQ(second.admission->prefillStarts[0], kTOKENS_PER_PAGE);
    finalizePrefillWithLengths(*second.admission, {130});
    finish(*second.admission);
}

TEST_F(ContextCacheCoordinatorTests, AdmissionUsesSchedulerOwnedResidentSlot)
{
    for (int32_t slot = 0; slot < kMAX_BATCH; ++slot)
    {
        mPageTable->setRow(slot, nullptr, 0);
    }
    mPageTable->upload(mStream);
    ASSERT_EQ(cudaStreamSynchronize(mStream), cudaSuccess);

    ContextCacheSequenceAdmission sequence{makeTokens(129), {}};
    sequence.resident = ResidentRef{2, 7};
    ContextCacheBatchAdmission batch;
    batch.sequences.push_back(std::move(sequence));
    auto request = mCoordinator->beginRequest(batch, DecodingKvHeadroom{1, 0}, mStream);
    ASSERT_EQ(request.status, ContextCacheCoordinatorStatus::kOk);
    ASSERT_TRUE(request.admission.has_value());
    ASSERT_EQ(mCoordinator->preparePrefill(request.admission->request), ContextCacheCoordinatorStatus::kOk);
    ASSERT_EQ(cudaStreamSynchronize(mStream), cudaSuccess);

    EXPECT_TRUE(std::all_of(mPageTable->hostRow(0), mPageTable->hostRow(0) + mPageTable->maxPagesPerSeq(),
        [](int32_t page) { return page == -1; }));
    EXPECT_GE(mPageTable->hostRow(2)[0], 0);
    finish(*request.admission);
}

TEST_F(ContextCacheCoordinatorTests, ExactFullInputHitReportsMatchButRewindsExecution)
{
    auto producer = begin({makeTokens(129)});
    ASSERT_TRUE(producer.admission.has_value());
    finalizePrefillWithLengths(*producer.admission, {129});
    finish(*producer.admission);

    ContextCacheMetrics const beforeReplay = mCoordinator->metrics();
    auto replay = begin({makeTokens(kTOKENS_PER_PAGE)});
    ASSERT_TRUE(replay.admission.has_value());
    EXPECT_EQ(replay.admission->prefillStarts[0], 0);
    ContextCacheMetrics const afterReplay = mCoordinator->metrics();
    EXPECT_EQ(afterReplay.matchedTokens - beforeReplay.matchedTokens, kTOKENS_PER_PAGE);
    EXPECT_EQ(afterReplay.fullInputRewindPlans - beforeReplay.fullInputRewindPlans, 1U);
    finish(*replay.admission);
}

TEST_F(ContextCacheCoordinatorTests, BypassUsesManagedPagesWithoutPublishing)
{
    auto bypass = begin({makeTokens(129)}, ContextCacheLookupPolicy::kBypass);
    ASSERT_TRUE(bypass.admission.has_value());
    finalizePrefillWithLengths(*bypass.admission, {129});
    finish(*bypass.admission);
    EXPECT_EQ(mCoordinator->manager().records().size(), 0U);

    auto lookup = begin({makeTokens(130)});
    ASSERT_TRUE(lookup.admission.has_value());
    EXPECT_EQ(lookup.admission->prefillStarts[0], 0);
    finish(*lookup.admission);
}

TEST_F(ContextCacheCoordinatorTests, BatchAppliesLookupPolicyPerSequence)
{
    auto producer = begin({makeTokens(129)});
    ASSERT_TRUE(producer.admission.has_value());
    finalizePrefillWithLengths(*producer.admission, {129});
    finish(*producer.admission);

    ContextCacheBatchAdmission batch;
    ContextCacheSequenceAdmission bypass{makeTokens(130), {}, {}, ResidentRef{0, 2}};
    bypass.lookupPolicy = ContextCacheLookupPolicy::kBypass;
    batch.sequences.push_back(std::move(bypass));
    batch.sequences.push_back(ContextCacheSequenceAdmission{makeTokens(130), {}, {}, ResidentRef{1, 1}});

    ContextCacheMetrics const before = mCoordinator->metrics();
    auto request = mCoordinator->beginRequest(batch, DecodingKvHeadroom{1, 0}, mStream);
    ASSERT_TRUE(request.admission.has_value());
    EXPECT_EQ(request.admission->prefillStarts, (std::vector<int32_t>{0, kTOKENS_PER_PAGE}));
    ContextCacheMetrics const after = mCoordinator->metrics();
    EXPECT_EQ(after.lookupBypassSequences - before.lookupBypassSequences, 1U);
    EXPECT_EQ(after.hitSequences - before.hitSequences, 1U);
    finish(*request.admission);
}

TEST_F(ContextCacheCoordinatorTests, MetricsClassifyPlansPublicationsAndCurrentOccupancy)
{
    auto bypass = begin({makeTokens(129)}, ContextCacheLookupPolicy::kBypass);
    ASSERT_TRUE(bypass.admission.has_value());
    finish(*bypass.admission);

    auto producer = begin({makeTokens(129)});
    ASSERT_TRUE(producer.admission.has_value());
    finalizePrefillWithLengths(*producer.admission, {129});
    finish(*producer.admission);

    auto existing = begin({makeTokens(130)});
    ASSERT_TRUE(existing.admission.has_value());
    finalizePrefillWithLengths(*existing.admission, {130});
    finish(*existing.admission);

    auto rewind = begin({makeTokens(kTOKENS_PER_PAGE)});
    ASSERT_TRUE(rewind.admission.has_value());
    finish(*rewind.admission);

    ContextCacheMetrics const metrics = mCoordinator->metrics();
    EXPECT_EQ(metrics.admittedSequences, 4U);
    EXPECT_EQ(metrics.hitSequences, 2U);
    EXPECT_EQ(metrics.lookupBypassSequences, 1U);
    EXPECT_EQ(metrics.forcedColdSequences, 0U);
    EXPECT_EQ(metrics.standardPlans, 1U);
    EXPECT_EQ(metrics.noReusablePrefixPlans, 2U);
    EXPECT_EQ(metrics.fullInputRewindPlans, 1U);
    EXPECT_EQ(metrics.publicationAttempts, 2U);
    EXPECT_EQ(metrics.committedPublications, 1U);
    EXPECT_EQ(metrics.existingPublications, 1U);
    EXPECT_EQ(metrics.publishedEndpoints, 2U);
    EXPECT_EQ(metrics.currentRecords, 1U);
    EXPECT_EQ(metrics.baseKvPages.capacity, mCoordinator->manager().pools().capacity(ResourceType::kBaseKvPage));
    EXPECT_EQ(metrics.baseKvPages.free, metrics.baseKvPages.capacity - 1);
    EXPECT_EQ(metrics.draftKvPages.capacity, 0);
    EXPECT_EQ(metrics.recurrentSnapshots.capacity, 0);
    EXPECT_EQ(metrics.partialKvSnapshots.capacity, 0);
}

TEST_F(ContextCacheCoordinatorTests, PrefillOnlyPolicyDoesNotAttemptDecodePublication)
{
    constexpr int32_t kInputLength{2 * kTOKENS_PER_PAGE - 1};
    auto request = begin(
        {makeTokens(kInputLength)}, ContextCacheLookupPolicy::kUseCache, ContextCacheCommitPolicy::kPrefillStateOnly);
    ASSERT_TRUE(request.admission.has_value());
    finalizePrefillWithLengths(*request.admission, {kInputLength});
    size_t const recordsAfterPrefill = mCoordinator->manager().records().size();

    ASSERT_EQ(mCoordinator->prepareDecodeStep(request.admission->request, DecodingKvHeadroom{1, 0}),
        ContextCacheCoordinatorStatus::kOk);
    ASSERT_EQ(cudaStreamSynchronize(mStream), cudaSuccess);
    int32_t const nextLookahead = 9002;
    std::vector<ContextCacheSequenceAdvance> progress{ContextCacheSequenceAdvance{&nextLookahead, 1, kInputLength + 1}};
    ASSERT_EQ(mCoordinator->completeDecodeStep(request.admission->request, progress, {0}),
        ContextCacheCoordinatorStatus::kOk);
    EXPECT_EQ(mCoordinator->manager().records().size(), recordsAfterPrefill);
    EXPECT_EQ(mCoordinator->metrics().publicationAttempts, 1U);
    EXPECT_EQ(mCoordinator->metrics().committedPublications, 1U);
    finish(*request.admission);
}

TEST_F(ContextCacheCoordinatorTests, SequenceIdentityAppliesToGeneratedPageBoundary)
{
    constexpr Hash128 kIDENTITY_A{0x1112131415161718ULL, 0x2122232425262728ULL};
    constexpr Hash128 kIDENTITY_B{0x3132333435363738ULL, 0x4142434445464748ULL};
    constexpr int32_t kINPUT_LENGTH{kTOKENS_PER_PAGE - 1};

    ContextCacheSequenceAdmission producerSequence;
    producerSequence.tokenIds = makeTokens(kINPUT_LENGTH);
    producerSequence.keyExtras.isolationDigest = kIDENTITY_A;
    producerSequence.resident = ResidentRef{0, 1};
    ContextCacheBatchAdmission producerBatch;
    producerBatch.sequences.push_back(std::move(producerSequence));
    ContextCacheCoordinator::BeginRequestResult producer
        = mCoordinator->beginRequest(producerBatch, DecodingKvHeadroom{1, 0}, mStream);
    ASSERT_EQ(producer.status, ContextCacheCoordinatorStatus::kOk);
    ASSERT_TRUE(producer.admission.has_value());
    finalizePrefillWithLengths(*producer.admission, {kINPUT_LENGTH});

    ASSERT_EQ(mCoordinator->prepareDecodeStep(producer.admission->request, DecodingKvHeadroom{1, 0}),
        ContextCacheCoordinatorStatus::kOk);
    ASSERT_EQ(cudaStreamSynchronize(mStream), cudaSuccess);
    int32_t const nextLookahead = 9002;
    std::vector<ContextCacheSequenceAdvance> progress{ContextCacheSequenceAdvance{&nextLookahead, 1, kTOKENS_PER_PAGE}};
    ASSERT_EQ(mCoordinator->completeDecodeStep(producer.admission->request, progress, {0}),
        ContextCacheCoordinatorStatus::kOk);
    finish(*producer.admission);

    std::vector<int32_t> publishedTokens = makeTokens(kINPUT_LENGTH);
    publishedTokens.push_back(9001);
    publishedTokens.push_back(9003);
    ContextCacheSequenceAdmission matchingSequence;
    matchingSequence.tokenIds = publishedTokens;
    matchingSequence.keyExtras.isolationDigest = kIDENTITY_A;
    matchingSequence.resident = ResidentRef{0, 2};
    ContextCacheBatchAdmission matchingBatch;
    matchingBatch.sequences.push_back(std::move(matchingSequence));
    ContextCacheCoordinator::BeginRequestResult matching
        = mCoordinator->beginRequest(matchingBatch, DecodingKvHeadroom{1, 0}, mStream);
    ASSERT_EQ(matching.status, ContextCacheCoordinatorStatus::kOk);
    ASSERT_TRUE(matching.admission.has_value());
    EXPECT_EQ(matching.admission->prefillStarts[0], kTOKENS_PER_PAGE);
    finish(*matching.admission);

    ContextCacheSequenceAdmission isolatedSequence;
    isolatedSequence.tokenIds = std::move(publishedTokens);
    isolatedSequence.keyExtras.isolationDigest = kIDENTITY_B;
    isolatedSequence.resident = ResidentRef{0, 3};
    ContextCacheBatchAdmission isolatedBatch;
    isolatedBatch.sequences.push_back(std::move(isolatedSequence));
    ContextCacheCoordinator::BeginRequestResult isolated
        = mCoordinator->beginRequest(isolatedBatch, DecodingKvHeadroom{1, 0}, mStream);
    ASSERT_EQ(isolated.status, ContextCacheCoordinatorStatus::kOk);
    ASSERT_TRUE(isolated.admission.has_value());
    EXPECT_EQ(isolated.admission->prefillStarts[0], 0);
    finish(*isolated.admission);
}

TEST_F(ContextCacheCoordinatorTests, RejectsMalformedBatchProgressBeforeMutation)
{
    auto request = begin({makeTokens(129), makeTokens(130)});
    ASSERT_TRUE(request.admission.has_value());
    ASSERT_EQ(mCoordinator->preparePrefill(request.admission->request), ContextCacheCoordinatorStatus::kOk);
    ASSERT_EQ(cudaStreamSynchronize(mStream), cudaSuccess);

    std::vector<int32_t> lookahead{9001, 9002};
    std::vector<ContextCacheSequenceAdvance> malformed{
        ContextCacheSequenceAdvance{&lookahead[0], 1, 129},
        ContextCacheSequenceAdvance{&lookahead[1], 1, 129},
    };
    EXPECT_THROW(mCoordinator->finalizePrefillPublication(request.admission->request, malformed), std::runtime_error);

    malformed[1].committedStateLength = 130;
    EXPECT_EQ(mCoordinator->finalizePrefillPublication(request.admission->request, malformed),
        ContextCacheCoordinatorStatus::kOk);
    finish(*request.admission);
}

TEST_F(ContextCacheCoordinatorTests, ResidentCompactionPreservesPhysicalRowsAcrossMiddleEviction)
{
    auto request = begin({makeTokens(129), makeTokens(130), makeTokens(131)});
    ASSERT_TRUE(request.admission.has_value());
    finalizePrefillWithLengths(*request.admission, {129, 130, 131});

    std::vector<int32_t> const survivorRow0(
        mPageTable->hostRow(0), mPageTable->hostRow(0) + mPageTable->maxPagesPerSeq());
    std::vector<int32_t> const survivorRow2(
        mPageTable->hostRow(2), mPageTable->hostRow(2) + mPageTable->maxPagesPerSeq());
    Tensor deviceMapping({kMAX_BATCH}, rt::DeviceType::kGPU, DataType::kINT32, "residentCompactionMapping");

    ASSERT_EQ(mCoordinator->beginBatchCompaction(request.admission->request, {0, -1, 1}, 2, deviceMapping),
        ContextCacheCoordinatorStatus::kOk);
    ASSERT_EQ(mCoordinator->compactBatch(request.admission->request), ContextCacheCoordinatorStatus::kOk);
    EXPECT_TRUE(std::equal(survivorRow0.begin(), survivorRow0.end(), mPageTable->hostRow(0)));
    EXPECT_TRUE(std::equal(survivorRow2.begin(), survivorRow2.end(), mPageTable->hostRow(2)));
    EXPECT_TRUE(std::all_of(mPageTable->hostRow(1), mPageTable->hostRow(1) + mPageTable->maxPagesPerSeq(),
        [](int32_t page) { return page == -1; }));

    ASSERT_EQ(mCoordinator->prepareDecodeStep(request.admission->request, DecodingKvHeadroom{128, 0}),
        ContextCacheCoordinatorStatus::kOk);
    ASSERT_EQ(cudaStreamSynchronize(mStream), cudaSuccess);
    EXPECT_GE(mPageTable->hostRow(0)[2], 0);
    EXPECT_GE(mPageTable->hostRow(2)[2], 0);
    EXPECT_TRUE(std::all_of(mPageTable->hostRow(1), mPageTable->hostRow(1) + mPageTable->maxPagesPerSeq(),
        [](int32_t page) { return page == -1; }));
    finish(*request.admission);
}

// Given two decoding sequences, one within a page of the per-sequence capacity and one whose jump-forward left a
//      pending run of 33 tokens
// When the decode step is prepared with each slot's own growth
// Then it succeeds and only the jumped slot gains a page, whereas growing every slot by the largest run fails the
//      request because the near-capacity slot would need a fifth page
TEST_F(ContextCacheCoordinatorTests, DecodeGrowthIsPerSlotWhenAJumpLeavesAPendingRun)
{
    constexpr int32_t kNearCapacity{kMAX_SEQUENCE_LENGTH - 1};
    constexpr int32_t kShort{100};
    constexpr int32_t kPendingRun{33};
    ASSERT_EQ(mPageTable->maxPagesPerSeq(), 4);
    {
        auto request = begin({makeTokens(kNearCapacity), makeTokens(kShort)});
        ASSERT_TRUE(request.admission.has_value());
        finalizePrefillWithLengths(*request.admission, {kNearCapacity, kShort});
        ASSERT_EQ(mPageTable->hostRow(1)[1], -1) << "the short sequence owns one page after its prefill";

        std::vector<int32_t> const extraTokensPerSlot{1, kPendingRun};
        ASSERT_EQ(
            mCoordinator->prepareDecodeStep(request.admission->request, DecodingKvHeadroom{1, 0}, &extraTokensPerSlot),
            ContextCacheCoordinatorStatus::kOk);
        ASSERT_EQ(cudaStreamSynchronize(mStream), cudaSuccess);
        EXPECT_GE(mPageTable->hostRow(1)[1], 0) << "the jumped slot gained the page its run needs";
        EXPECT_EQ(mPageTable->hostRow(1)[2], -1);
        finish(*request.admission);
    }
    {
        auto request = begin({makeTokens(kNearCapacity), makeTokens(kShort)});
        ASSERT_TRUE(request.admission.has_value());
        finalizePrefillWithLengths(*request.admission, {kNearCapacity, kShort});
        EXPECT_EQ(mCoordinator->prepareDecodeStep(request.admission->request, DecodingKvHeadroom{kPendingRun, 0}),
            ContextCacheCoordinatorStatus::kRequestFailed)
            << "a batch-wide growth would push the near-capacity slot past its page-table row";
        finish(*request.admission);
    }
}

TEST_F(ContextCacheCoordinatorTests, CompactionRejectsSurvivorReordering)
{
    auto request = begin({makeTokens(129), makeTokens(130)});
    ASSERT_TRUE(request.admission.has_value());
    finalizePrefillWithLengths(*request.admission, {129, 130});
    Tensor deviceMapping({kMAX_BATCH}, rt::DeviceType::kGPU, DataType::kINT32, "coordinatorTestMapping");

    EXPECT_THROW(
        mCoordinator->beginBatchCompaction(request.admission->request, {1, 0}, 2, deviceMapping), std::runtime_error);
    finish(*request.admission);
}

TEST_F(ContextCacheCoordinatorTests, AdmitsASequenceMidRequestAndReusesThePublishedPrefix)
{
    // The first request publishes two full blocks (256 tokens + lookahead), then a sequence sharing that
    // prefix joins the live request: its lookup must land on the published pages, and its own
    // finalization must publish, so a later request can reuse what the admitted sequence computed.
    constexpr int32_t kPrefix = 2 * kTOKENS_PER_PAGE;
    auto first = begin({makeTokens(kPrefix)});
    ASSERT_TRUE(first.admission.has_value());
    finalizePrefillWithLengths(*first.admission, {kPrefix});
    std::vector<PageId> const untouchedRow(
        mPageTable->hostRow(1), mPageTable->hostRow(1) + mPageTable->maxPagesPerSeq());
    std::vector<PageId> const rowBeforeAdmission(
        mPageTable->hostRow(2), mPageTable->hostRow(2) + mPageTable->maxPagesPerSeq());

    std::vector<int32_t> joiningTokens = makeTokens(kPrefix);
    joiningTokens.push_back(7001); // diverges after the shared prefix
    ContextCacheCoordinator::AdmitSequenceResult admitted = mCoordinator->admitSequence(first.admission->request,
        ContextCacheSequenceAdmission{joiningTokens, {}, {}, ResidentRef{2, 1}}, DecodingKvHeadroom{1, 0});
    ASSERT_EQ(admitted.status, ContextCacheCoordinatorStatus::kOk);
    EXPECT_EQ(admitted.prefillStart, kPrefix) << "the shared prefix must be reused, not recomputed";
    EXPECT_FALSE(std::equal(rowBeforeAdmission.begin(), rowBeforeAdmission.end(), mPageTable->hostRow(2)))
        << "the lease must bind the physical resident row";
    EXPECT_EQ(mPageTable->hostRow(2)[0], mPageTable->hostRow(0)[0])
        << "the admitted row must share the published first prefix";
    EXPECT_TRUE(std::equal(untouchedRow.begin(), untouchedRow.end(), mPageTable->hostRow(1)))
        << "logical execution order must not choose a page-table row";

    // The seated prefill sampled one lookahead token; finalization aligns the ledger and publishes.
    int32_t const lookahead = 7002;
    ASSERT_EQ(mCoordinator->finalizeSequenceAdmission(first.admission->request, 1,
                  ContextCacheSequenceAdvance{&lookahead, 1, static_cast<int32_t>(joiningTokens.size())}),
        ContextCacheCoordinatorStatus::kOk);

    // Both slots advance one decode step together -- the admitted slot is an ordinary batch member.
    ASSERT_EQ(mCoordinator->prepareDecodeStep(first.admission->request, DecodingKvHeadroom{1, 0}),
        ContextCacheCoordinatorStatus::kOk);
    ASSERT_EQ(cudaStreamSynchronize(mStream), cudaSuccess);
    std::vector<int32_t> const next{7003, 7004};
    std::vector<ContextCacheSequenceAdvance> progress{ContextCacheSequenceAdvance{&next[0], 1, kPrefix + 1},
        ContextCacheSequenceAdvance{&next[1], 1, static_cast<int32_t>(joiningTokens.size()) + 1}};
    ASSERT_EQ(
        mCoordinator->completeDecodeStep(first.admission->request, progress, {}), ContextCacheCoordinatorStatus::kOk);
    EXPECT_TRUE(std::equal(untouchedRow.begin(), untouchedRow.end(), mPageTable->hostRow(1)));

    // Both leases (and the published records) reference the shared prefix pages. The first
    // finishes and is compacted out; that must not strand or free pages the survivor still reads.
    std::vector<int32_t> const sharedPages(mPageTable->hostRow(0), mPageTable->hostRow(0) + 2);
    for (PageId const page : sharedPages)
    {
        EXPECT_GE(mCoordinator->manager().pools().activeRefCount({ResourceType::kBaseKvPage, page}), 2);
    }
    // The runtime keeps the cache manager's active view in step with the batch; mirror that here
    // so the slot-state compaction sees a two-wide batch.
    mCache->setActiveBatchSize(2);
    Tensor deviceMapping({2}, rt::DeviceType::kGPU, DataType::kINT32, "coordinatorTestMapping");
    ASSERT_EQ(mCoordinator->beginBatchCompaction(first.admission->request, {-1, 0}, 1, deviceMapping),
        ContextCacheCoordinatorStatus::kOk);
    ASSERT_EQ(mCoordinator->compactBatch(first.admission->request), ContextCacheCoordinatorStatus::kOk);
    for (PageId const page : sharedPages)
    {
        EXPECT_GE(mCoordinator->manager().pools().activeRefCount({ResourceType::kBaseKvPage, page}), 1)
            << "compacting the first out must not free pages the admitted sequence still reads";
    }
    finish(*first.admission);

    // The admitted sequence's own full blocks are findable afterwards.
    auto reader = begin({joiningTokens});
    ASSERT_TRUE(reader.admission.has_value());
    EXPECT_EQ(reader.admission->prefillStarts[0], kPrefix)
        << "the admitted sequence's published prefix must be visible to later lookups";
    finalizePrefillWithLengths(*reader.admission, {static_cast<int32_t>(joiningTokens.size())});
    finish(*reader.admission);
}

TEST_F(ContextCacheCoordinatorTests, FirstSequenceAdmitsLikeAnyLater)
{
    // A request admitted with no sequences is the shape a long-lived session opens with: nothing
    // is leased at admission, and every sequence -- the first included -- joins through
    // admitSequence while the request executes. The first admission must lease, bind its row,
    // publish, and retire like a sequence that joined a populated batch.
    constexpr int32_t kPrefix = 2 * kTOKENS_PER_PAGE;
    auto producer = begin({makeTokens(kPrefix)});
    ASSERT_TRUE(producer.admission.has_value());
    finalizePrefillWithLengths(*producer.admission, {kPrefix});
    finish(*producer.admission);

    EXPECT_EQ(mCoordinator->activeRequests(), 0) << "a finished request is no longer counted";
    auto session = begin({});
    ASSERT_EQ(session.status, ContextCacheCoordinatorStatus::kOk);
    ASSERT_TRUE(session.admission.has_value());
    EXPECT_TRUE(session.admission->prefillStarts.empty());
    EXPECT_EQ(mCoordinator->activeRequests(), 1);
    // preparePrefill moves the request into execution; with nothing admitted it binds no row.
    ASSERT_EQ(mCoordinator->preparePrefill(session.admission->request), ContextCacheCoordinatorStatus::kOk);
    ASSERT_EQ(cudaStreamSynchronize(mStream), cudaSuccess);
    EXPECT_EQ(mCache->getActiveBatchSize(), 0);

    std::vector<int32_t> joiningTokens = makeTokens(kPrefix);
    joiningTokens.push_back(7201);
    ContextCacheCoordinator::AdmitSequenceResult admitted = mCoordinator->admitSequence(session.admission->request,
        ContextCacheSequenceAdmission{joiningTokens, {}, {}, ResidentRef{1, 1}}, DecodingKvHeadroom{1, 0});
    ASSERT_EQ(admitted.status, ContextCacheCoordinatorStatus::kOk);
    EXPECT_EQ(admitted.prefillStart, kPrefix) << "the first sequence of an empty request still looks up its prefix";
    EXPECT_EQ(mPageTable->hostRow(1)[0], mPageTable->hostRow(0)[0])
        << "the admitted row must be bound to the published producer prefix";

    int32_t const lookahead = 7202;
    ASSERT_EQ(mCoordinator->finalizeSequenceAdmission(session.admission->request, 0,
                  ContextCacheSequenceAdvance{&lookahead, 1, static_cast<int32_t>(joiningTokens.size())}),
        ContextCacheCoordinatorStatus::kOk);

    ASSERT_EQ(mCoordinator->prepareDecodeStep(session.admission->request, DecodingKvHeadroom{1, 0}),
        ContextCacheCoordinatorStatus::kOk);
    ASSERT_EQ(cudaStreamSynchronize(mStream), cudaSuccess);
    int32_t const next = 7203;
    std::vector<ContextCacheSequenceAdvance> progress{
        ContextCacheSequenceAdvance{&next, 1, static_cast<int32_t>(joiningTokens.size()) + 1}};
    ASSERT_EQ(
        mCoordinator->completeDecodeStep(session.admission->request, progress, {}), ContextCacheCoordinatorStatus::kOk);

    // The sequence retires through the ordinary compaction, and the request keeps executing with
    // nothing resident -- it is not finished, only empty. (The runtime's steps materialize the
    // cache manager's execution lengths for the live batch before any compaction; this test runs
    // no step, so it sets the batch size the way a step would have left it.)
    mCache->setActiveBatchSize(1);
    Tensor deviceMapping({kMAX_BATCH}, rt::DeviceType::kGPU, DataType::kINT32, "coordinatorTestMapping");
    ASSERT_EQ(mCoordinator->beginBatchCompaction(session.admission->request, {-1}, 0, deviceMapping),
        ContextCacheCoordinatorStatus::kOk);
    ASSERT_EQ(mCoordinator->compactBatch(session.admission->request), ContextCacheCoordinatorStatus::kOk);

    // Admitting again into the emptied request is the same operation, on the same row, as the
    // first time.
    ContextCacheCoordinator::AdmitSequenceResult readmitted = mCoordinator->admitSequence(session.admission->request,
        ContextCacheSequenceAdmission{joiningTokens, {}, {}, ResidentRef{1, 2}}, DecodingKvHeadroom{1, 0});
    ASSERT_EQ(readmitted.status, ContextCacheCoordinatorStatus::kOk);
    EXPECT_EQ(readmitted.prefillStart, kPrefix);
    ASSERT_EQ(mCoordinator->finalizeSequenceAdmission(session.admission->request, 0,
                  ContextCacheSequenceAdvance{&lookahead, 1, static_cast<int32_t>(joiningTokens.size())}),
        ContextCacheCoordinatorStatus::kOk);

    // Policies travel with the sequence -- an empty request has none to hand down: a bypass
    // admission beside the resident leases private pages that neither look up nor publish.
    ContextCacheSequenceAdmission bypass{joiningTokens, {}, {}, ResidentRef{2, 1}};
    bypass.lookupPolicy = ContextCacheLookupPolicy::kBypass;
    ContextCacheMetrics const before = mCoordinator->metrics();
    ContextCacheCoordinator::AdmitSequenceResult bypassed
        = mCoordinator->admitSequence(session.admission->request, bypass, DecodingKvHeadroom{1, 0});
    ASSERT_EQ(bypassed.status, ContextCacheCoordinatorStatus::kOk);
    EXPECT_EQ(bypassed.prefillStart, 0) << "a bypass sequence must not reuse the published prefix";
    EXPECT_EQ(mCoordinator->metrics().lookupBypassSequences - before.lookupBypassSequences, 1U);
    ASSERT_EQ(mCoordinator->finalizeSequenceAdmission(session.admission->request, 1,
                  ContextCacheSequenceAdvance{&lookahead, 1, static_cast<int32_t>(joiningTokens.size())}),
        ContextCacheCoordinatorStatus::kOk);
    finish(*session.admission);
}

TEST_F(ContextCacheCoordinatorTests, RetractingAnAdmissionUnwindsTheLeaseAndTheRowStaysUsable)
{
    // The recovery path for a seating that threw between lease and slot append: the retraction
    // must release the lease and clear the row so a later admission can take the same seat. A
    // retraction from a request that admitted nothing, on the other hand, is the caller's
    // bookkeeping disagreeing with the coordinator's, and the row state can no longer be trusted.
    constexpr int32_t kPrefix = 2 * kTOKENS_PER_PAGE;
    auto first = begin({makeTokens(kPrefix)});
    ASSERT_TRUE(first.admission.has_value());
    finalizePrefillWithLengths(*first.admission, {kPrefix});

    std::vector<int32_t> joiningTokens = makeTokens(kPrefix);
    joiningTokens.push_back(7101);
    ContextCacheCoordinator::AdmitSequenceResult admitted = mCoordinator->admitSequence(first.admission->request,
        ContextCacheSequenceAdmission{joiningTokens, {}, {}, ResidentRef{1, 1}}, DecodingKvHeadroom{1, 0});
    ASSERT_EQ(admitted.status, ContextCacheCoordinatorStatus::kOk);

    mCoordinator->retractSequenceAdmission(first.admission->request);

    // The first is alone again and fully functional: it can decode, and the seat is reusable.
    ASSERT_EQ(mCoordinator->prepareDecodeStep(first.admission->request, DecodingKvHeadroom{1, 0}),
        ContextCacheCoordinatorStatus::kOk);
    ASSERT_EQ(cudaStreamSynchronize(mStream), cudaSuccess);
    int32_t const sampled = 7102;
    std::vector<ContextCacheSequenceAdvance> progress{ContextCacheSequenceAdvance{&sampled, 1, kPrefix + 1}};
    ASSERT_EQ(
        mCoordinator->completeDecodeStep(first.admission->request, progress, {}), ContextCacheCoordinatorStatus::kOk);

    ContextCacheCoordinator::AdmitSequenceResult readmitted = mCoordinator->admitSequence(first.admission->request,
        ContextCacheSequenceAdmission{joiningTokens, {}, {}, ResidentRef{1, 2}}, DecodingKvHeadroom{1, 0});
    EXPECT_EQ(readmitted.status, ContextCacheCoordinatorStatus::kOk) << "the retracted seat must be admittable again";
    int32_t const lookahead = 7103;
    ASSERT_EQ(mCoordinator->finalizeSequenceAdmission(first.admission->request, 1,
                  ContextCacheSequenceAdvance{&lookahead, 1, static_cast<int32_t>(joiningTokens.size())}),
        ContextCacheCoordinatorStatus::kOk);
    finish(*first.admission);

    // Retracting from an empty request poisons the coordinator.
    auto empty = begin({});
    ASSERT_TRUE(empty.admission.has_value());
    ASSERT_EQ(mCoordinator->preparePrefill(empty.admission->request), ContextCacheCoordinatorStatus::kOk);
    EXPECT_FALSE(mCoordinator->retractSequenceAdmission(empty.admission->request));
    EXPECT_EQ(begin({makeTokens(4)}).status, ContextCacheCoordinatorStatus::kPoisoned);
}

// Given an executing request whose sequence 0 was admitted with the request and whose sequence 1 joined through
//       admitSequence and was finalized
// When abortSequenceAdmission is called for either of them
// Then each call throws: neither sequence is in admission
TEST_F(ContextCacheCoordinatorTests, AbortingASequenceNotInAdmissionThrows)
{
    constexpr int32_t kPrefix = 2 * kTOKENS_PER_PAGE;
    auto first = begin({makeTokens(kPrefix)});
    ASSERT_TRUE(first.admission.has_value());
    finalizePrefillWithLengths(*first.admission, {kPrefix});

    std::vector<int32_t> joiningTokens = makeTokens(kPrefix);
    joiningTokens.push_back(7301);
    ContextCacheCoordinator::AdmitSequenceResult admitted = mCoordinator->admitSequence(first.admission->request,
        ContextCacheSequenceAdmission{joiningTokens, {}, {}, ResidentRef{1, 1}}, DecodingKvHeadroom{1, 0});
    ASSERT_EQ(admitted.status, ContextCacheCoordinatorStatus::kOk);
    int32_t const lookahead = 7302;
    ASSERT_EQ(mCoordinator->finalizeSequenceAdmission(first.admission->request, 1,
                  ContextCacheSequenceAdvance{&lookahead, 1, static_cast<int32_t>(joiningTokens.size())}),
        ContextCacheCoordinatorStatus::kOk);

    EXPECT_THROW(mCoordinator->abortSequenceAdmission(first.admission->request, 0), std::runtime_error);
    EXPECT_THROW(mCoordinator->abortSequenceAdmission(first.admission->request, 1), std::runtime_error);
    finish(*first.admission);
}

TEST_F(ContextCacheCoordinatorTests, AnUnfinalizedAdmissionHoldsAtZeroAdvanceInsteadOfFailingTheBatch)
{
    // A seated prefill that failed leaves its slot terminal from birth: never finalized, no
    // generate length. The zero advance's hold sentinel must be legal for it -- the earlier
    // rebuilt-arithmetic form threw here and killed the whole first request.
    constexpr int32_t kPrefix = 2 * kTOKENS_PER_PAGE;
    auto first = begin({makeTokens(kPrefix)});
    ASSERT_TRUE(first.admission.has_value());
    finalizePrefillWithLengths(*first.admission, {kPrefix});

    std::vector<int32_t> joiningTokens = makeTokens(kPrefix);
    joiningTokens.push_back(7201);
    ContextCacheCoordinator::AdmitSequenceResult admitted = mCoordinator->admitSequence(first.admission->request,
        ContextCacheSequenceAdmission{joiningTokens, {}, {}, ResidentRef{1, 1}}, DecodingKvHeadroom{1, 0});
    ASSERT_EQ(admitted.status, ContextCacheCoordinatorStatus::kOk);
    // No finalizeSequenceAdmission: the seated prefill failed and the slot is a kError ghost.

    ASSERT_EQ(mCoordinator->prepareDecodeStep(first.admission->request, DecodingKvHeadroom{1, 0}),
        ContextCacheCoordinatorStatus::kOk);
    ASSERT_EQ(cudaStreamSynchronize(mStream), cudaSuccess);
    int32_t const sampled = 7202;
    std::vector<ContextCacheSequenceAdvance> progress{ContextCacheSequenceAdvance{&sampled, 1, kPrefix + 1},
        ContextCacheSequenceAdvance{nullptr, 0, ContextCacheSequenceAdvance::kHoldCommittedStateLength}};
    EXPECT_EQ(
        mCoordinator->completeDecodeStep(first.admission->request, progress, {}), ContextCacheCoordinatorStatus::kOk)
        << "a ghost slot's zero advance must not fail the first batch";
    finish(*first.admission);
}

TEST_F(ContextCacheCoordinatorTests, ACancelledSlotAdvancesByZeroWithoutFailingTheStep)
{
    // A slot cancelled at the top of a step appends nothing that step; the decode completion must
    // treat that as a legal zero advance holding the committed length, not a broken invariant that
    // fails the whole batch (the chaos-found failure mode: one client disconnect killing every
    // request sharing the batch).
    auto request = begin({makeTokens(100), makeTokens(101)});
    ASSERT_TRUE(request.admission.has_value());
    finalizePrefillWithLengths(*request.admission, {100, 101});

    ASSERT_EQ(mCoordinator->prepareDecodeStep(request.admission->request, DecodingKvHeadroom{1, 0}),
        ContextCacheCoordinatorStatus::kOk);
    ASSERT_EQ(cudaStreamSynchronize(mStream), cudaSuccess);
    int32_t const sampled = 9002;
    std::vector<ContextCacheSequenceAdvance> progress{
        ContextCacheSequenceAdvance{&sampled, 1, 101}, // slot 0 decoded normally
        // slot 1 was cancelled: a zero advance carries the hold sentinel, not a rebuilt length --
        // a slot that failed before its admission was finalized has none to rebuild.
        ContextCacheSequenceAdvance{nullptr, 0, ContextCacheSequenceAdvance::kHoldCommittedStateLength},
    };
    EXPECT_EQ(
        mCoordinator->completeDecodeStep(request.admission->request, progress, {}), ContextCacheCoordinatorStatus::kOk);

    // The sentinel is mandatory: a zero advance carrying a concrete length -- even the correct
    // one -- is a producer that thinks it knows better than the ledger.
    ASSERT_EQ(mCoordinator->prepareDecodeStep(request.admission->request, DecodingKvHeadroom{1, 0}),
        ContextCacheCoordinatorStatus::kOk);
    ASSERT_EQ(cudaStreamSynchronize(mStream), cudaSuccess);
    std::vector<ContextCacheSequenceAdvance> crooked{
        ContextCacheSequenceAdvance{&sampled, 1, 102},
        ContextCacheSequenceAdvance{nullptr, 0, 101},
    };
    EXPECT_THROW(mCoordinator->completeDecodeStep(request.admission->request, crooked, {}), std::runtime_error);
    finish(*request.admission);
}

TEST_F(ContextCacheCoordinatorTests, AdmissionRefusalOnPoolPressureLeavesTheRequestIntact)
{
    // The cache manager refuses pools smaller than maxBatch * pagesPerSlot, so within one request
    // a free batch slot always has quota -- pool pressure on admission comes from OVERLAPPING
    // leases, the ownership shape in-flight batching created. A bystander request holds a full
    // sequence's pages; the running batch then has a slot free but not enough pages behind it.
    // The refusal must say "transient capacity", and the first must keep decoding as if the
    // admission attempt never happened.
    auto bystander = begin({makeTokens(kMAX_SEQUENCE_LENGTH - 4)});
    ASSERT_TRUE(bystander.admission.has_value());
    auto pressured = begin({makeTokens(kMAX_SEQUENCE_LENGTH - 1), makeTokens(kMAX_SEQUENCE_LENGTH - 2)});
    ASSERT_TRUE(pressured.admission.has_value());
    finalizePrefillWithLengths(*pressured.admission, {kMAX_SEQUENCE_LENGTH - 1, kMAX_SEQUENCE_LENGTH - 2});

    std::vector<int32_t> disjoint(static_cast<size_t>(kMAX_SEQUENCE_LENGTH - 1));
    std::iota(disjoint.begin(), disjoint.end(), 100000);
    ContextCacheCoordinator::AdmitSequenceResult refused = mCoordinator->admitSequence(pressured.admission->request,
        ContextCacheSequenceAdmission{disjoint, {}, {}, ResidentRef{2, 1}}, DecodingKvHeadroom{1, 0});
    EXPECT_EQ(refused.status, ContextCacheCoordinatorStatus::kRequestFailed);
    EXPECT_TRUE(refused.insufficientCapacity) << "pool pressure must read as retry-later, not failure";

    // The first decodes on undisturbed.
    ASSERT_EQ(mCoordinator->prepareDecodeStep(pressured.admission->request, DecodingKvHeadroom{1, 0}),
        ContextCacheCoordinatorStatus::kOk);
    ASSERT_EQ(cudaStreamSynchronize(mStream), cudaSuccess);
    std::vector<int32_t> const next{9002, 9003};
    std::vector<ContextCacheSequenceAdvance> progress{ContextCacheSequenceAdvance{&next[0], 1, kMAX_SEQUENCE_LENGTH},
        ContextCacheSequenceAdvance{&next[1], 1, kMAX_SEQUENCE_LENGTH - 1}};
    ASSERT_EQ(mCoordinator->completeDecodeStep(pressured.admission->request, progress, {}),
        ContextCacheCoordinatorStatus::kOk);
    finish(*pressured.admission);
    finish(*bystander.admission);
}

TEST_F(ContextCacheCoordinatorTests, FailedDrainQuarantinesOwnershipUntilShutdownSucceeds)
{
    ASSERT_EQ(mCoordinator->shutdown(), ContextCacheCoordinatorStatus::kOk);
    mCoordinator.reset();
    int32_t synchronizeCalls = 0;
    createCoordinator([&](cudaStream_t stream) {
        ++synchronizeCalls;
        if (synchronizeCalls == 1)
        {
            return cudaErrorUnknown;
        }
        return cudaStreamSynchronize(stream);
    });

    auto request = begin({makeTokens(129)});
    ASSERT_TRUE(request.admission.has_value());
    ASSERT_EQ(mCoordinator->preparePrefill(request.admission->request), ContextCacheCoordinatorStatus::kOk);
    request.admission.reset();
    EXPECT_EQ(synchronizeCalls, 1);

    auto poisoned = begin({makeTokens(129)});
    EXPECT_EQ(poisoned.status, ContextCacheCoordinatorStatus::kPoisoned);
    EXPECT_EQ(mCoordinator->shutdown(), ContextCacheCoordinatorStatus::kOk);
    EXPECT_EQ(synchronizeCalls, 2);
    EXPECT_EQ(mCoordinator->manager().pools().freeCount(ResourceType::kBaseKvPage), mEngine.kvPoolPages);
}

TEST_F(ContextCacheCoordinatorTests, ConcurrentRequestsMayOverlapInLifetime)
{
    auto first = begin({makeTokens(129)});
    ASSERT_EQ(first.status, ContextCacheCoordinatorStatus::kOk);
    ASSERT_TRUE(first.admission.has_value());

    // The single-occupancy gate used to reject this while the first handle was still alive.
    auto second = begin({makeTokens(130)});
    EXPECT_EQ(second.status, ContextCacheCoordinatorStatus::kOk);
    ASSERT_TRUE(second.admission.has_value());

    first.admission.reset();
    second.admission.reset();
    EXPECT_EQ(mCoordinator->shutdown(), ContextCacheCoordinatorStatus::kOk);
    EXPECT_EQ(mCoordinator->manager().pools().freeCount(ResourceType::kBaseKvPage), mEngine.kvPoolPages);
}

TEST_F(ContextCacheCoordinatorTests, FailedDrainQuarantinesEveryResidentRequest)
{
    ASSERT_EQ(mCoordinator->shutdown(), ContextCacheCoordinatorStatus::kOk);
    mCoordinator.reset();
    int32_t synchronizeCalls = 0;
    createCoordinator([&](cudaStream_t stream) {
        // Both resident requests fail to drain; only shutdown is allowed to succeed.
        if (++synchronizeCalls <= 2)
        {
            return cudaErrorUnknown;
        }
        return cudaStreamSynchronize(stream);
    });

    auto first = begin({makeTokens(129)});
    ASSERT_TRUE(first.admission.has_value());
    ASSERT_EQ(mCoordinator->preparePrefill(first.admission->request), ContextCacheCoordinatorStatus::kOk);
    auto second = begin({makeTokens(130)});
    ASSERT_TRUE(second.admission.has_value());
    ASSERT_EQ(mCoordinator->preparePrefill(second.admission->request), ContextCacheCoordinatorStatus::kOk);

    // A single quarantine slot used to std::terminate() on the second failure.
    first.admission.reset();
    second.admission.reset();
    EXPECT_EQ(synchronizeCalls, 2);

    EXPECT_EQ(mCoordinator->shutdown(), ContextCacheCoordinatorStatus::kOk);
    EXPECT_EQ(synchronizeCalls, 4);
    EXPECT_EQ(mCoordinator->manager().pools().freeCount(ResourceType::kBaseKvPage), mEngine.kvPoolPages);
}

TEST_F(ContextCacheCoordinatorTests, OversizedAdmissionDoesNotLeaveRequestTokenHeld)
{
    EXPECT_THROW(begin({makeTokens(kMAX_SEQUENCE_LENGTH + 1)}), std::runtime_error);
    auto valid = begin({makeTokens(1)});
    EXPECT_EQ(valid.status, ContextCacheCoordinatorStatus::kOk);
    ASSERT_TRUE(valid.admission.has_value());
    finish(*valid.admission);
}

} // namespace

TEST_F(ContextCacheCoordinatorTests, ActionCaptureSurvivesRequestFinishAndPreservesWarmPrefix)
{
    // Given a published prefix, when its LLM request finishes before action materialization,
    // then action pins retain the physical pages and a later LLM request still reuses the original prefix.
    auto first = begin({makeTokens(300)});
    ASSERT_EQ(first.status, ContextCacheCoordinatorStatus::kOk);
    ASSERT_TRUE(first.admission.has_value());
    finalizePrefillWithLengths(*first.admission, {300});
    std::vector<int32_t> const prefix(mPageTable->hostRow(0), mPageTable->hostRow(0) + 3);
    auto& pool = mCache->getCombinedKVCache(0);
    ASSERT_EQ(
        cudaMemsetAsync(pool.rawPointer(), 0x3c, pool.getShape().volume() * sizeof(uint16_t), mStream), cudaSuccess);
    auto& manager = mCoordinator->manager();
    ActionKvBatchCollector collector(kMAX_BATCH, 5, mPageTable->numPages(), &manager);
    collector.beginRequest({true}, {-20});
    Tensor lengths({1}, rt::DeviceType::kCPU, DataType::kINT32);
    lengths.dataPointer<int32_t>()[0] = 300;
    ASSERT_EQ(cudaMemcpyAsync(mCache->getKVCacheLengths().rawPointer(), lengths.rawPointer(), sizeof(int32_t),
                  cudaMemcpyHostToDevice, mStream),
        cudaSuccess);
    collector.captureFinished(*mPageTable, mCache->getKVCacheLengths(), {1}, {0}, {{0, 1}}, mStream);
    finish(*first.admission);
    for (PageId const page : prefix)
    {
        EXPECT_EQ(manager.pools().activeRefCount({ResourceType::kBaseKvPage, page}), 1);
    }
    auto const batch = collector.materialize(mStream);
    EXPECT_EQ(std::vector<int32_t>(batch.pageTable.hostRow(0), batch.pageTable.hostRow(0) + 2),
        (std::vector<int32_t>{prefix[0], prefix[1]}));
    PageId const terminalPage = batch.pageTable.hostRow(0)[2];
    EXPECT_EQ(terminalPage, prefix[2]);
    EXPECT_EQ(manager.pools().cacheRefCount({ResourceType::kBaseKvPage, terminalPage}), 0);
    collector.releaseMaterialized(mStream);
    EXPECT_EQ(manager.pools().freeCount(ResourceType::kBaseKvPage), mPageTable->numPages() - 2);
    auto second = begin({makeTokens(300)});
    ASSERT_EQ(second.status, ContextCacheCoordinatorStatus::kOk);
    ASSERT_TRUE(second.admission.has_value());
    EXPECT_EQ(second.admission->prefillStarts[0], 256);
    finalizePrefillWithLengths(*second.admission, {300});
    finish(*second.admission);
}
