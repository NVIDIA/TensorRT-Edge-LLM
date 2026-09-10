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

using namespace nvinfer1;
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
        admission.lookupPolicy = lookupPolicy;
        admission.commitPolicy = commitPolicy;
        for (auto const& tokens : batch)
        {
            admission.sequences.push_back(ContextCacheSequenceAdmission{tokens, {}});
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

TEST_F(ContextCacheCoordinatorTests, CompactionMovesRowsWithoutCopyingOrRenumberingPages)
{
    auto request = begin({makeTokens(129), makeTokens(130), makeTokens(131)});
    ASSERT_TRUE(request.admission.has_value());
    finalizePrefillWithLengths(*request.admission, {129, 130, 131});

    std::vector<int32_t> oldRow0(mPageTable->hostRow(0), mPageTable->hostRow(0) + 2);
    std::vector<int32_t> oldRow1(mPageTable->hostRow(1), mPageTable->hostRow(1) + 2);
    std::vector<int32_t> oldRow2(mPageTable->hostRow(2), mPageTable->hostRow(2) + 2);
    Tensor deviceMapping({kMAX_BATCH}, rt::DeviceType::kGPU, DataType::kINT32, "coordinatorTestMapping");

    ASSERT_EQ(mCoordinator->beginBatchCompaction(request.admission->request, {0, -1, 1}, 2, deviceMapping),
        ContextCacheCoordinatorStatus::kOk);
    ASSERT_EQ(mCoordinator->compactBatch(request.admission->request), ContextCacheCoordinatorStatus::kOk);
    EXPECT_TRUE(std::equal(oldRow0.begin(), oldRow0.end(), mPageTable->hostRow(0)));
    EXPECT_TRUE(std::equal(oldRow2.begin(), oldRow2.end(), mPageTable->hostRow(1)));
    for (PageId const page : oldRow1)
    {
        EXPECT_EQ(mCoordinator->manager().pools().activeRefCount({ResourceType::kBaseKvPage, page}), 0);
    }
    finish(*request.admission);
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

TEST_F(ContextCacheCoordinatorTests, OversizedAdmissionDoesNotLeaveRequestTokenHeld)
{
    EXPECT_THROW(begin({makeTokens(kMAX_SEQUENCE_LENGTH + 1)}), std::runtime_error);
    auto valid = begin({makeTokens(1)});
    EXPECT_EQ(valid.status, ContextCacheCoordinatorStatus::kOk);
    ASSERT_TRUE(valid.admission.has_value());
    finish(*valid.admission);
}

} // namespace
