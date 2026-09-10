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

constexpr int32_t kMAX_BATCH{2};
constexpr int32_t kMAX_SEQUENCE_LENGTH{512};

std::vector<int32_t> makeTokens(int32_t count)
{
    std::vector<int32_t> tokens(static_cast<size_t>(count));
    std::iota(tokens.begin(), tokens.end(), 1);
    return tokens;
}

std::vector<PageId> const& eaglePagePath(CacheRecord const& record)
{
    ELLM_CHECK(record.specState.has_value(), "Test expected an EAGLE spec-state record");
    return record.specState->pagePath;
}

LLMEngineConfig makeAttentionConfig(char const* modelType, int32_t layers)
{
    LLMEngineConfig config;
    config.modelType = modelType;
    config.hiddenSize = 64;
    config.numDecoderLayers = layers;
    config.numAttentionLayers = layers;
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
    config.layerTypes.assign(static_cast<size_t>(layers), HybridCacheManager::LayerType::kAttention);
    config.kvLayerConfigs.assign(static_cast<size_t>(layers), KVLayerConfig{config.numKVHeads, config.headDim});
    return config;
}

DeploymentConfig makeEagleDeployment()
{
    DeploymentConfig deployment;
    deployment.base = makeAttentionConfig("spec-coordinator-base", 3);
    deployment.base.specDecodeType = SpecDecodeMode::kEAGLE;
    deployment.base.isSpecDecodeBase = true;
    deployment.base.specTargetLayerIds = {0, 2};
    deployment.draft = makeAttentionConfig("spec-coordinator-draft", 2);
    deployment.draft->specDecodeType = SpecDecodeMode::kEAGLE;
    deployment.draft->baseModelHiddenSize = deployment.base.hiddenSize * 2;
    SpecDecodeConfig spec;
    spec.draftingTopK = 2;
    spec.draftingStep = 2;
    spec.verifySize = 4;
    deployment.specConfig = spec;
    return deployment;
}

HybridCacheManager::Config makeCacheConfig(LLMEngineConfig const& engine)
{
    KVCacheManager::Config kvConfig{engine.numAttentionLayers, engine.maxSupportedBatchSize, engine.maxKVCacheCapacity,
        engine.kvLayerConfigs, engine.kvCacheDtype, engine.kvPoolPages};
    MambaCacheManager::Config mambaConfig{/*.numRecurrentLayers=*/0, engine.maxSupportedBatchSize};
    return HybridCacheManager::Config{
        engine.layerTypes, std::move(kvConfig), std::move(mambaConfig), engine.maxSupportedBatchSize};
}

class ContextCacheSpecCoordinatorTests : public ::testing::Test
{
protected:
    void SetUp() override
    {
        ASSERT_EQ(cudaStreamCreate(&mStream), cudaSuccess);
        mDeployment = makeEagleDeployment();
        mBaseCache = std::make_unique<HybridCacheManager>(makeCacheConfig(mDeployment.base), mStream);
        mDraftCache = std::make_unique<HybridCacheManager>(makeCacheConfig(*mDeployment.draft), mStream);
        mBasePageTable = makePageTable(*mBaseCache);
        mDraftPageTable = makePageTable(*mDraftCache);
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
        mDraftPageTable.reset();
        mBasePageTable.reset();
        mDraftCache.reset();
        mBaseCache.reset();
        EXPECT_EQ(cudaStreamDestroy(mStream), cudaSuccess);
    }

    std::unique_ptr<KVPageTable> makePageTable(HybridCacheManager& cache)
    {
        KVCacheManager const& kv = cache.getKVCacheManager();
        auto table = std::make_unique<KVPageTable>(kMAX_BATCH, pagesPerSlot(kv.maxCapPadded()), kv.numPages());
        table->setIdentity();
        table->upload(mStream);
        return table;
    }

    void createCoordinator(ContextCacheCoordinator::StreamSynchronizer synchronizer = {})
    {
        ContextCachePhysicalResources resources{*mBaseCache, *mBasePageTable, mDraftCache.get(), mDraftPageTable.get()};
        ContextCacheConfig config{/*.enabled=*/true, /*.maxRecords=*/16};
        mCoordinator = std::make_unique<ContextCacheCoordinator>(config, mDeployment,
            validateContextCacheDeployment(mDeployment), resources, mStream, std::move(synchronizer));
    }

    DecodingKvHeadroom specHeadroom() const
    {
        auto const& spec = *mDeployment.specConfig;
        if (mDeployment.base.specDecodeType == SpecDecodeMode::kGemma4MTP)
        {
            return {spec.draftingStep + 1, 0};
        }
        return {spec.verifySize, spec.draftingStep * spec.draftingTopK};
    }

    ContextCacheCoordinator::AdmissionResult begin(std::vector<int32_t> tokens, bool speculativeRequest = true)
    {
        return beginBatch({std::move(tokens)}, speculativeRequest);
    }

    ContextCacheCoordinator::AdmissionResult beginBatch(
        std::vector<std::vector<int32_t>> batch, bool speculativeRequest = true)
    {
        ContextCacheBatchAdmission admission;
        admission.speculativeRequest = speculativeRequest;
        DecodingKvHeadroom const headroom = speculativeRequest ? specHeadroom() : DecodingKvHeadroom{1, 0};
        for (auto& tokens : batch)
        {
            admission.sequences.push_back(ContextCacheSequenceAdmission{std::move(tokens), {}});
        }
        ContextCacheCoordinator::BeginRequestResult result = mCoordinator->beginRequest(admission, headroom, mStream);
        EXPECT_EQ(result.status, ContextCacheCoordinatorStatus::kOk);
        EXPECT_TRUE(result.admission.has_value());
        return std::move(*result.admission);
    }

    void freezePrefill(ContextCacheCoordinator::AdmissionResult& request, int32_t inputLength, int32_t lookahead = 9001)
    {
        ASSERT_EQ(mCoordinator->preparePrefill(request.request), ContextCacheCoordinatorStatus::kOk);
        // The runtime path has already consumed the base-prefill synchronization before it enqueues EAGLE
        // draft initialization. The coordinator deliberately keeps that later draft work pending.
        ASSERT_EQ(cudaStreamSynchronize(mStream), cudaSuccess);
        std::vector<ContextCacheSequenceAdvance> progress{ContextCacheSequenceAdvance{&lookahead, 1, inputLength}};
        std::vector<int32_t> commonLengths{inputLength};
        ASSERT_EQ(mCoordinator->finalizePrefillPublication(request.request, progress, &commonLengths),
            ContextCacheCoordinatorStatus::kOk);
    }

    void finalizeVanillaPrefill(
        ContextCacheCoordinator::AdmissionResult& request, int32_t inputLength, int32_t lookahead = 9001)
    {
        ASSERT_EQ(mCoordinator->preparePrefill(request.request), ContextCacheCoordinatorStatus::kOk);
        ASSERT_EQ(cudaStreamSynchronize(mStream), cudaSuccess);
        std::vector<ContextCacheSequenceAdvance> progress{ContextCacheSequenceAdvance{&lookahead, 1, inputLength}};
        ASSERT_EQ(
            mCoordinator->finalizePrefillPublication(request.request, progress), ContextCacheCoordinatorStatus::kOk);
    }

    void completeDecode(ContextCacheCoordinator::AdmissionResult& request, std::vector<int32_t> const& acceptedTokens,
        int32_t committedLength, int32_t commonLength, bool publishTerminal)
    {
        ASSERT_EQ(mCoordinator->prepareDecodeStep(request.request, specHeadroom()), ContextCacheCoordinatorStatus::kOk);
        // EAGLE verification owns the existing round synchronization that makes all preceding row uploads and draft
        // work ready before this callback.
        ASSERT_EQ(cudaStreamSynchronize(mStream), cudaSuccess);
        std::vector<ContextCacheSequenceAdvance> progress{ContextCacheSequenceAdvance{
            acceptedTokens.data(), static_cast<int32_t>(acceptedTokens.size()), committedLength}};
        std::vector<int32_t> commonLengths{commonLength};
        std::vector<int32_t> const completedSlots = publishTerminal ? std::vector<int32_t>{0} : std::vector<int32_t>{};
        ASSERT_EQ(mCoordinator->completeDecodeStep(request.request, progress, completedSlots, &commonLengths),
            ContextCacheCoordinatorStatus::kOk);
    }

    cudaStream_t mStream{};
    DeploymentConfig mDeployment;
    std::unique_ptr<HybridCacheManager> mBaseCache;
    std::unique_ptr<HybridCacheManager> mDraftCache;
    std::unique_ptr<KVPageTable> mBasePageTable;
    std::unique_ptr<KVPageTable> mDraftPageTable;
    std::unique_ptr<ContextCacheCoordinator> mCoordinator;
};

TEST_F(ContextCacheSpecCoordinatorTests, FirstVerificationPublishesPairAndNextRequestUsesFullPageReplay)
{
    constexpr int32_t kInputLength{2 * kTOKENS_PER_PAGE};
    auto producer = begin(makeTokens(kInputLength));
    freezePrefill(producer, kInputLength);
    EXPECT_EQ(mCoordinator->manager().records().size(), 0U);

    completeDecode(producer, {41, 42}, kInputLength + 2, kInputLength, false);
    ASSERT_EQ(mCoordinator->manager().records().size(), 1U);
    CacheRecord const& record
        = mCoordinator->manager().records().get(mCoordinator->manager().records().lruToMru().front());
    EXPECT_EQ(record.basePagePath.size(), 2U);
    EXPECT_EQ(eaglePagePath(record).size(), 2U);
    EXPECT_EQ(mCoordinator->finish(producer.request), ContextCacheCoordinatorStatus::kOk);

    std::vector<int32_t> continuation = makeTokens(kInputLength);
    continuation.push_back(77);
    ContextCacheMetrics const metricsBeforeConsumer = mCoordinator->metrics();
    auto consumer = begin(std::move(continuation));
    EXPECT_EQ(consumer.prefillStarts, std::vector<int32_t>{kTOKENS_PER_PAGE});
    ContextCacheMetrics const metricsAfterConsumer = mCoordinator->metrics();
    EXPECT_EQ(
        metricsAfterConsumer.matchedTokens, metricsBeforeConsumer.matchedTokens + static_cast<uint64_t>(kInputLength));
    EXPECT_EQ(metricsAfterConsumer.specFullPageReplays, metricsBeforeConsumer.specFullPageReplays + 1U);
    EXPECT_EQ(mCoordinator->finish(consumer.request), ContextCacheCoordinatorStatus::kOk);
}

TEST_F(ContextCacheSpecCoordinatorTests, SpeculativeMediaAdmissionBypassesLookupAndPublication)
{
    constexpr int32_t kInputLength{2 * kTOKENS_PER_PAGE + 1};
    std::vector<int32_t> const tokens = makeTokens(kInputLength);
    std::vector<Hash128> mediaHashes(static_cast<size_t>(kInputLength));
    Hash128 const mediaHash{0x1234U, 0x5678U};
    mediaHashes[kTOKENS_PER_PAGE - 1] = mediaHash;
    mediaHashes[kTOKENS_PER_PAGE] = mediaHash;
    mediaHashes[kTOKENS_PER_PAGE + 1] = mediaHash;

    ContextCacheBatchAdmission admission;
    admission.speculativeRequest = true;
    admission.lookupPolicy = ContextCacheLookupPolicy::kUseCache;
    admission.sequences.push_back(ContextCacheSequenceAdmission{tokens, {}, std::move(mediaHashes)});

    ContextCacheMetrics const before = mCoordinator->metrics();
    ContextCacheCoordinator::BeginRequestResult result = mCoordinator->beginRequest(admission, specHeadroom(), mStream);
    ASSERT_EQ(result.status, ContextCacheCoordinatorStatus::kOk);
    ASSERT_TRUE(result.admission.has_value());
    EXPECT_EQ(result.admission->prefillStarts, std::vector<int32_t>{0});
    EXPECT_EQ(mCoordinator->metrics().lookupBypassSequences, before.lookupBypassSequences + 1U);

    freezePrefill(*result.admission, kInputLength);
    completeDecode(*result.admission, {41, 42}, kInputLength + 2, kInputLength, false);
    EXPECT_EQ(mCoordinator->manager().records().size(), 0U);
    EXPECT_EQ(mCoordinator->finish(result.admission->request), ContextCacheCoordinatorStatus::kOk);
}

TEST_F(ContextCacheSpecCoordinatorTests, MediaInOneSpeculativeBatchSlotBypassesEverySlot)
{
    constexpr int32_t kInputLength{kTOKENS_PER_PAGE + 1};
    std::vector<Hash128> mediaHashes(static_cast<size_t>(kInputLength));
    mediaHashes[kTOKENS_PER_PAGE - 1] = Hash128{0x1234U, 0x5678U};
    mediaHashes[kTOKENS_PER_PAGE] = Hash128{0x1234U, 0x5678U};

    ContextCacheBatchAdmission admission;
    admission.speculativeRequest = true;
    admission.lookupPolicy = ContextCacheLookupPolicy::kUseCache;
    admission.sequences.push_back(ContextCacheSequenceAdmission{makeTokens(kInputLength), {}});
    admission.sequences.push_back(ContextCacheSequenceAdmission{makeTokens(kInputLength), {}, std::move(mediaHashes)});

    ContextCacheMetrics const before = mCoordinator->metrics();
    ContextCacheCoordinator::BeginRequestResult result = mCoordinator->beginRequest(admission, specHeadroom(), mStream);
    ASSERT_EQ(result.status, ContextCacheCoordinatorStatus::kOk);
    ASSERT_TRUE(result.admission.has_value());
    EXPECT_EQ(result.admission->prefillStarts, (std::vector<int32_t>{0, 0}));
    EXPECT_EQ(mCoordinator->metrics().lookupBypassSequences, before.lookupBypassSequences + 2U);
    EXPECT_EQ(mCoordinator->finish(result.admission->request), ContextCacheCoordinatorStatus::kOk);
}

TEST_F(ContextCacheSpecCoordinatorTests, DecodeEndPublishesOnlyTheCommonBaseDraftBoundary)
{
    constexpr int32_t kInputLength{2 * kTOKENS_PER_PAGE - 1};
    auto producer = begin(makeTokens(kInputLength));
    freezePrefill(producer, kInputLength);
    completeDecode(producer, {41, 42}, kInputLength + 2, kInputLength, false);
    completeDecode(producer, {43, 44}, kInputLength + 4, kInputLength + 2, true);

    CacheRecord const& record
        = mCoordinator->manager().records().get(mCoordinator->manager().records().lruToMru().back());
    EXPECT_EQ(record.basePagePath.size(), 2U);
    EXPECT_EQ(eaglePagePath(record).size(), 2U);
    EXPECT_FALSE(record.exactCheckpointLength.has_value());
    EXPECT_EQ(mCoordinator->finish(producer.request), ContextCacheCoordinatorStatus::kOk);
}

TEST_F(ContextCacheSpecCoordinatorTests, PrefillOnlyCompletionSynchronizesDraftBeforePublishingAndCompaction)
{
    ASSERT_EQ(mCoordinator->shutdown(), ContextCacheCoordinatorStatus::kOk);
    mCoordinator.reset();
    int32_t synchronizations{};
    createCoordinator([&](cudaStream_t stream) {
        ++synchronizations;
        return cudaStreamSynchronize(stream);
    });

    constexpr int32_t kInputLength{kTOKENS_PER_PAGE};
    auto request = begin(makeTokens(kInputLength));
    freezePrefill(request, kInputLength);
    Tensor deviceMapping({kMAX_BATCH}, rt::DeviceType::kGPU, DataType::kINT32, "specCoordinatorMapping");
    ASSERT_EQ(mCoordinator->beginBatchCompaction(request.request, {-1}, 0, deviceMapping),
        ContextCacheCoordinatorStatus::kOk);
    ASSERT_EQ(mCoordinator->compactBatch(request.request), ContextCacheCoordinatorStatus::kOk);
    EXPECT_EQ(synchronizations, 2);
    EXPECT_EQ(mCoordinator->manager().records().size(), 1U);
    EXPECT_EQ(mCoordinator->finish(request.request), ContextCacheCoordinatorStatus::kOk);
}

TEST_F(ContextCacheSpecCoordinatorTests, AbandonedPendingEagleInitializationDoesNotPublishPair)
{
    constexpr int32_t kInputLength{2 * kTOKENS_PER_PAGE};
    {
        auto request = begin(makeTokens(kInputLength));
        freezePrefill(request, kInputLength);
        EXPECT_EQ(mCoordinator->manager().records().size(), 0U);
    }

    EXPECT_EQ(mCoordinator->manager().records().size(), 0U);
    ContextCacheMetrics const metricsBeforeNext = mCoordinator->metrics();
    auto next = begin(makeTokens(kInputLength));
    EXPECT_EQ(next.prefillStarts, std::vector<int32_t>{0});
    ContextCacheMetrics const metricsAfterNext = mCoordinator->metrics();
    EXPECT_EQ(metricsAfterNext.matchedTokens, metricsBeforeNext.matchedTokens);
    EXPECT_EQ(metricsAfterNext.specFullPageReplays, metricsBeforeNext.specFullPageReplays);
    EXPECT_EQ(mCoordinator->finish(next.request), ContextCacheCoordinatorStatus::kOk);
}

TEST_F(ContextCacheSpecCoordinatorTests, VanillaRequestReusesBaseSideOfPairedRecordWithoutTouchingDraftRows)
{
    constexpr int32_t kInputLength{2 * kTOKENS_PER_PAGE};
    auto producer = begin(makeTokens(kInputLength));
    freezePrefill(producer, kInputLength);
    completeDecode(producer, {41, 42}, kInputLength + 2, kInputLength, false);
    EXPECT_EQ(mCoordinator->finish(producer.request), ContextCacheCoordinatorStatus::kOk);

    std::vector<int32_t> continuation = makeTokens(kInputLength);
    continuation.push_back(77);
    ContextCacheMetrics const metricsBeforeVanillaConsumer = mCoordinator->metrics();
    auto consumer = begin(std::move(continuation), false);
    EXPECT_EQ(consumer.prefillStarts, std::vector<int32_t>{kInputLength});
    ContextCacheMetrics const metricsAfterVanillaConsumer = mCoordinator->metrics();
    EXPECT_EQ(metricsAfterVanillaConsumer.matchedTokens,
        metricsBeforeVanillaConsumer.matchedTokens + static_cast<uint64_t>(kInputLength));
    EXPECT_EQ(metricsAfterVanillaConsumer.specFullPageReplays, metricsBeforeVanillaConsumer.specFullPageReplays);

    std::vector<int32_t> const draftRowBefore(
        mDraftPageTable->hostRow(0), mDraftPageTable->hostRow(0) + mDraftPageTable->maxPagesPerSeq());
    finalizeVanillaPrefill(consumer, kInputLength + 1);
    EXPECT_TRUE(std::equal(draftRowBefore.begin(), draftRowBefore.end(), mDraftPageTable->hostRow(0)));
    Tensor deviceMapping({kMAX_BATCH}, rt::DeviceType::kGPU, DataType::kINT32, "vanillaOnEagleMapping");
    ASSERT_EQ(mCoordinator->beginBatchCompaction(consumer.request, {-1}, 0, deviceMapping),
        ContextCacheCoordinatorStatus::kOk);
    ASSERT_EQ(mCoordinator->compactBatch(consumer.request), ContextCacheCoordinatorStatus::kOk);
    EXPECT_TRUE(std::equal(draftRowBefore.begin(), draftRowBefore.end(), mDraftPageTable->hostRow(0)));
    EXPECT_EQ(mCoordinator->finish(consumer.request), ContextCacheCoordinatorStatus::kOk);

    std::vector<int32_t> eagleContinuation = makeTokens(kInputLength);
    eagleContinuation.push_back(77);
    eagleContinuation.push_back(78);
    ContextCacheMetrics const metricsBeforeEagleConsumer = mCoordinator->metrics();
    auto eagleConsumer = begin(std::move(eagleContinuation));
    EXPECT_EQ(eagleConsumer.prefillStarts, std::vector<int32_t>{kTOKENS_PER_PAGE});
    ContextCacheMetrics const metricsAfterEagleConsumer = mCoordinator->metrics();
    EXPECT_EQ(metricsAfterEagleConsumer.matchedTokens,
        metricsBeforeEagleConsumer.matchedTokens + static_cast<uint64_t>(kInputLength));
    EXPECT_EQ(metricsAfterEagleConsumer.specFullPageReplays, metricsBeforeEagleConsumer.specFullPageReplays + 1U);
    EXPECT_EQ(mCoordinator->finish(eagleConsumer.request), ContextCacheCoordinatorStatus::kOk);
}

TEST_F(ContextCacheSpecCoordinatorTests, FirstRoundCompactionRemovesTerminalSlotAndKeepsEagleSurvivorExecutable)
{
    constexpr int32_t kInputLength{2 * kTOKENS_PER_PAGE};
    auto request = beginBatch({makeTokens(kInputLength), makeTokens(kInputLength)});
    ASSERT_EQ(mCoordinator->preparePrefill(request.request), ContextCacheCoordinatorStatus::kOk);
    ASSERT_EQ(cudaStreamSynchronize(mStream), cudaSuccess);
    std::vector<int32_t> lookahead{9001, 9002};
    std::vector<ContextCacheSequenceAdvance> progress{
        ContextCacheSequenceAdvance{&lookahead[0], 1, kInputLength},
        ContextCacheSequenceAdvance{&lookahead[1], 1, kInputLength},
    };
    std::vector<int32_t> commonLengths{kInputLength, kInputLength};
    ASSERT_EQ(mCoordinator->finalizePrefillPublication(request.request, progress, &commonLengths),
        ContextCacheCoordinatorStatus::kOk);

    std::vector<int32_t> const survivingDraftRow(
        mDraftPageTable->hostRow(1), mDraftPageTable->hostRow(1) + mDraftPageTable->maxPagesPerSeq());
    Tensor deviceMapping({kMAX_BATCH}, rt::DeviceType::kGPU, DataType::kINT32, "firstRoundEagleMapping");
    ASSERT_EQ(mCoordinator->beginBatchCompaction(request.request, {-1, 0}, 1, deviceMapping),
        ContextCacheCoordinatorStatus::kOk);
    ASSERT_EQ(mCoordinator->compactBatch(request.request), ContextCacheCoordinatorStatus::kOk);
    EXPECT_TRUE(std::equal(survivingDraftRow.begin(), survivingDraftRow.end(), mDraftPageTable->hostRow(0)));

    completeDecode(request, {41, 42}, kInputLength + 2, kInputLength, false);
    EXPECT_EQ(mCoordinator->finish(request.request), ContextCacheCoordinatorStatus::kOk);
}

TEST_F(ContextCacheSpecCoordinatorTests, BatchRowsResolveDifferentPairedReuseLengthsIndependently)
{
    constexpr int32_t kShortLength{2 * kTOKENS_PER_PAGE};
    constexpr int32_t kLongLength{3 * kTOKENS_PER_PAGE};

    auto shortProducer = begin(makeTokens(kShortLength));
    freezePrefill(shortProducer, kShortLength);
    completeDecode(shortProducer, {41, 42}, kShortLength + 2, kShortLength, false);
    ASSERT_EQ(mCoordinator->finish(shortProducer.request), ContextCacheCoordinatorStatus::kOk);

    auto longProducer = begin(makeTokens(kLongLength));
    freezePrefill(longProducer, kLongLength);
    completeDecode(longProducer, {43, 44}, kLongLength + 2, kLongLength, false);
    ASSERT_EQ(mCoordinator->finish(longProducer.request), ContextCacheCoordinatorStatus::kOk);

    std::vector<int32_t> shortContinuation = makeTokens(kShortLength);
    shortContinuation.push_back(71);
    std::vector<int32_t> longContinuation = makeTokens(kLongLength);
    longContinuation.push_back(72);
    auto consumer = beginBatch({std::move(shortContinuation), std::move(longContinuation)});

    EXPECT_EQ(consumer.prefillStarts, (std::vector<int32_t>{kTOKENS_PER_PAGE, 2 * kTOKENS_PER_PAGE}));
    EXPECT_NE(mBasePageTable->hostRow(0)[0], mBasePageTable->hostRow(1)[0]);
    EXPECT_NE(mBasePageTable->hostRow(0)[1], mBasePageTable->hostRow(1)[1]);
    EXPECT_NE(mDraftPageTable->hostRow(0)[0], mDraftPageTable->hostRow(1)[0]);
    EXPECT_NE(mDraftPageTable->hostRow(0)[1], mDraftPageTable->hostRow(1)[1]);
    EXPECT_EQ(mCoordinator->finish(consumer.request), ContextCacheCoordinatorStatus::kOk);
}

TEST_F(ContextCacheSpecCoordinatorTests, EagleRequestDoesNotConsumeBaseOnlyRecord)
{
    constexpr int32_t kInputLength{2 * kTOKENS_PER_PAGE};
    auto producer = begin(makeTokens(kInputLength), false);
    finalizeVanillaPrefill(producer, kInputLength);
    EXPECT_EQ(mCoordinator->finish(producer.request), ContextCacheCoordinatorStatus::kOk);

    ASSERT_EQ(mCoordinator->manager().records().size(), 1U);
    CacheRecord const& record
        = mCoordinator->manager().records().get(mCoordinator->manager().records().lruToMru().front());
    EXPECT_EQ(record.basePagePath.size(), 2U);
    EXPECT_FALSE(record.specState.has_value());

    std::vector<int32_t> continuation = makeTokens(kInputLength);
    continuation.push_back(77);
    ContextCacheMetrics const metricsBeforeConsumer = mCoordinator->metrics();
    auto consumer = begin(std::move(continuation));
    EXPECT_EQ(consumer.prefillStarts, std::vector<int32_t>{0});
    ContextCacheMetrics const metricsAfterConsumer = mCoordinator->metrics();
    EXPECT_EQ(metricsAfterConsumer.matchedTokens, metricsBeforeConsumer.matchedTokens);
    EXPECT_EQ(metricsAfterConsumer.specFullPageReplays, metricsBeforeConsumer.specFullPageReplays);
    EXPECT_EQ(mCoordinator->finish(consumer.request), ContextCacheCoordinatorStatus::kOk);
}

TEST_F(ContextCacheSpecCoordinatorTests, InitialHeadroomAllowsUnequalBaseAndDraftPagePaths)
{
    ContextCacheMetrics const before = mCoordinator->metrics();
    ContextCacheBatchAdmission admission;
    admission.speculativeRequest = true;
    DecodingKvHeadroom const headroom{/*baseExtraTokens=*/2, /*draftExtraTokens=*/130};
    admission.sequences.push_back(ContextCacheSequenceAdmission{makeTokens(kTOKENS_PER_PAGE - 1), {}});

    ContextCacheCoordinator::BeginRequestResult result = mCoordinator->beginRequest(admission, headroom, mStream);
    ASSERT_EQ(result.status, ContextCacheCoordinatorStatus::kOk);
    ASSERT_TRUE(result.admission.has_value());
    ContextCacheMetrics const admitted = mCoordinator->metrics();
    EXPECT_EQ(before.baseKvPages.free - admitted.baseKvPages.free, 2);
    EXPECT_EQ(before.draftKvPages.free - admitted.draftKvPages.free, 3);
    EXPECT_EQ(mCoordinator->preparePrefill(result.admission->request), ContextCacheCoordinatorStatus::kOk);
    EXPECT_EQ(mCoordinator->finish(result.admission->request), ContextCacheCoordinatorStatus::kOk);
}

class ContextCacheGemma4MtpCoordinatorTests : public ::testing::Test
{
protected:
    void SetUp() override
    {
        ASSERT_EQ(cudaStreamCreate(&mStream), cudaSuccess);
        mDeployment.base = makeAttentionConfig("gemma4-mtp-coordinator-base", 3);
        mDeployment.base.specDecodeType = SpecDecodeMode::kGemma4MTP;
        mDeployment.base.isSpecDecodeBase = true;
        mDeployment.draft = makeAttentionConfig("gemma4-mtp-coordinator-assistant", 1);
        mDeployment.draft->specDecodeType = SpecDecodeMode::kGemma4MTP;
        mDeployment.draft->hasOwnKVCache = false;
        mDeployment.draft->sharesTargetKV = true;
        SpecDecodeConfig spec;
        spec.draftingTopK = 1;
        spec.draftingStep = 4;
        spec.verifySize = 5;
        mDeployment.specConfig = spec;

        mBaseCache = std::make_unique<HybridCacheManager>(makeCacheConfig(mDeployment.base), mStream);
        KVCacheManager const& kv = mBaseCache->getKVCacheManager();
        mBasePageTable = std::make_unique<KVPageTable>(kMAX_BATCH, pagesPerSlot(kv.maxCapPadded()), kv.numPages());
        mBasePageTable->setIdentity();
        mBasePageTable->upload(mStream);
        ASSERT_EQ(cudaStreamSynchronize(mStream), cudaSuccess);
        ContextCachePhysicalResources resources{*mBaseCache, *mBasePageTable, nullptr, nullptr};
        mCoordinator
            = std::make_unique<ContextCacheCoordinator>(ContextCacheConfig{/*.enabled=*/true, /*.maxRecords=*/16},
                mDeployment, validateContextCacheDeployment(mDeployment), resources, mStream);
    }

    void TearDown() override
    {
        if (mCoordinator != nullptr)
        {
            EXPECT_EQ(mCoordinator->shutdown(), ContextCacheCoordinatorStatus::kOk);
        }
        mCoordinator.reset();
        mBasePageTable.reset();
        mBaseCache.reset();
        EXPECT_EQ(cudaStreamDestroy(mStream), cudaSuccess);
    }

    ContextCacheCoordinator::AdmissionResult begin(std::vector<int32_t> tokens)
    {
        ContextCacheBatchAdmission admission;
        admission.speculativeRequest = true;
        admission.sequences.push_back(ContextCacheSequenceAdmission{std::move(tokens), {}});
        ContextCacheCoordinator::BeginRequestResult result
            = mCoordinator->beginRequest(admission, DecodingKvHeadroom{5, 0}, mStream);
        EXPECT_EQ(result.status, ContextCacheCoordinatorStatus::kOk);
        EXPECT_TRUE(result.admission.has_value());
        return std::move(*result.admission);
    }

    void publishPrefill(ContextCacheCoordinator::AdmissionResult& admission, int32_t inputLength)
    {
        ASSERT_EQ(mCoordinator->preparePrefill(admission.request), ContextCacheCoordinatorStatus::kOk);
        ASSERT_EQ(cudaStreamSynchronize(mStream), cudaSuccess);
        int32_t lookahead = 9001;
        std::vector<ContextCacheSequenceAdvance> advances{ContextCacheSequenceAdvance{&lookahead, 1, inputLength}};
        std::vector<int32_t> commonLengths{inputLength};
        ASSERT_EQ(mCoordinator->finalizePrefillPublication(admission.request, advances, &commonLengths),
            ContextCacheCoordinatorStatus::kOk);
    }

    cudaStream_t mStream{};
    DeploymentConfig mDeployment;
    std::unique_ptr<HybridCacheManager> mBaseCache;
    std::unique_ptr<KVPageTable> mBasePageTable;
    std::unique_ptr<ContextCacheCoordinator> mCoordinator;
};

TEST_F(ContextCacheGemma4MtpCoordinatorTests, PrefillUsesCommittedBaseBoundaryWithoutAssistantStateLengths)
{
    constexpr int32_t kInputLength{kTOKENS_PER_PAGE};
    auto request = begin(makeTokens(kInputLength));
    ASSERT_EQ(mCoordinator->preparePrefill(request.request), ContextCacheCoordinatorStatus::kOk);
    ASSERT_EQ(cudaStreamSynchronize(mStream), cudaSuccess);

    int32_t lookahead = 9001;
    std::vector<ContextCacheSequenceAdvance> advances{ContextCacheSequenceAdvance{&lookahead, 1, kInputLength}};
    EXPECT_EQ(mCoordinator->finalizePrefillPublication(request.request, advances), ContextCacheCoordinatorStatus::kOk);
}

TEST_F(ContextCacheGemma4MtpCoordinatorTests, DecodeUsesCommittedBaseBoundaryWithoutAssistantStateLengths)
{
    constexpr int32_t kInputLength{kTOKENS_PER_PAGE};
    auto request = begin(makeTokens(kInputLength));
    ASSERT_EQ(mCoordinator->preparePrefill(request.request), ContextCacheCoordinatorStatus::kOk);
    ASSERT_EQ(cudaStreamSynchronize(mStream), cudaSuccess);
    int32_t lookahead = 9001;
    std::vector<ContextCacheSequenceAdvance> prefillAdvances{ContextCacheSequenceAdvance{&lookahead, 1, kInputLength}};
    ASSERT_EQ(
        mCoordinator->finalizePrefillPublication(request.request, prefillAdvances), ContextCacheCoordinatorStatus::kOk);

    ASSERT_EQ(
        mCoordinator->prepareDecodeStep(request.request, DecodingKvHeadroom{5, 0}), ContextCacheCoordinatorStatus::kOk);
    ASSERT_EQ(cudaStreamSynchronize(mStream), cudaSuccess);
    std::vector<int32_t> accepted{41, 42};
    std::vector<ContextCacheSequenceAdvance> decodeAdvances{ContextCacheSequenceAdvance{
        accepted.data(), static_cast<int32_t>(accepted.size()), kInputLength + static_cast<int32_t>(accepted.size())}};
    EXPECT_EQ(
        mCoordinator->completeDecodeStep(request.request, decodeAdvances, {}), ContextCacheCoordinatorStatus::kOk);
}

TEST_F(ContextCacheGemma4MtpCoordinatorTests, AdmissionReservesFullVerifyWindowBeforeFirstDecode)
{
    constexpr int32_t kInputLength{kTOKENS_PER_PAGE - 1};
    auto request = begin(makeTokens(kInputLength));
    ASSERT_EQ(mCoordinator->preparePrefill(request.request), ContextCacheCoordinatorStatus::kOk);
    ASSERT_EQ(cudaStreamSynchronize(mStream), cudaSuccess);
    int32_t lookahead = 9001;
    std::vector<ContextCacheSequenceAdvance> prefillAdvances{ContextCacheSequenceAdvance{&lookahead, 1, kInputLength}};
    ASSERT_EQ(
        mCoordinator->finalizePrefillPublication(request.request, prefillAdvances), ContextCacheCoordinatorStatus::kOk);

    ContextCacheMetrics const before = mCoordinator->metrics();
    ASSERT_EQ(before.baseKvPages.capacity - before.baseKvPages.free, 2);
    ASSERT_EQ(
        mCoordinator->prepareDecodeStep(request.request, DecodingKvHeadroom{5, 0}), ContextCacheCoordinatorStatus::kOk);
    ASSERT_EQ(cudaStreamSynchronize(mStream), cudaSuccess);
    ContextCacheMetrics const after = mCoordinator->metrics();
    EXPECT_EQ(after.baseKvPages.capacity - after.baseKvPages.free, 2);
}

TEST_F(ContextCacheGemma4MtpCoordinatorTests, BaseOnlyRecordSupportsPartialAndExactFullInputHits)
{
    constexpr int32_t kInputLength{2 * kTOKENS_PER_PAGE};
    auto producer = begin(makeTokens(kInputLength));
    publishPrefill(producer, kInputLength);
    ASSERT_EQ(mCoordinator->finish(producer.request), ContextCacheCoordinatorStatus::kOk);

    ASSERT_EQ(mCoordinator->manager().records().size(), 1U);
    CacheRecord const& record
        = mCoordinator->manager().records().get(mCoordinator->manager().records().lruToMru().front());
    EXPECT_FALSE(record.specState.has_value());
    EXPECT_EQ(mCoordinator->metrics().draftKvPages.capacity, 0);

    auto exactHit = begin(makeTokens(kInputLength));
    EXPECT_EQ(exactHit.prefillStarts, std::vector<int32_t>{kTOKENS_PER_PAGE});
    EXPECT_EQ(mCoordinator->finish(exactHit.request), ContextCacheCoordinatorStatus::kOk);

    std::vector<int32_t> continuation = makeTokens(kInputLength);
    continuation.push_back(77);
    auto partialHit = begin(std::move(continuation));
    EXPECT_EQ(partialHit.prefillStarts, std::vector<int32_t>{kInputLength});
    EXPECT_EQ(mCoordinator->finish(partialHit.request), ContextCacheCoordinatorStatus::kOk);
    EXPECT_GE(mCoordinator->metrics().reusedTokens, static_cast<uint64_t>(3 * kTOKENS_PER_PAGE));
}

} // namespace
