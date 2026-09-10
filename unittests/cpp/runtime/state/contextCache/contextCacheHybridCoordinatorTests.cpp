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
#include "common/cudaMacros.h"
#include "common/pagedKvTypes.h"
#include "runtime/hybridCacheManager.h"
#include "runtime/state/contextCache/hybridSnapshotStorage.h"
#include "runtime/state/kvPageTable.h"

#include <cuda_fp16.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <numeric>
#include <optional>
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

LLMEngineConfig makeHybridEngineConfig()
{
    LLMEngineConfig config;
    config.modelType = "hybrid-coordinator-test";
    config.hiddenSize = 64;
    config.numDecoderLayers = 2;
    config.numAttentionLayers = 1;
    config.numLinearAttnLayers = 1;
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
    config.layerTypes = {HybridCacheManager::LayerType::kAttention, HybridCacheManager::LayerType::kMamba};
    config.kvLayerConfigs = {KVLayerConfig{config.numKVHeads, config.headDim}};
    config.recurrentStateNumHeads = 2;
    config.recurrentStateHeadDim = 3;
    config.recurrentStateSize = 4;
    config.convDim = 5;
    config.convKernel = 6;
    config.recurrentStateDtype = DataType::kHALF;
    config.convStateDtype = DataType::kHALF;
    return config;
}

LLMEngineConfig makePureRecurrentEngineConfig()
{
    LLMEngineConfig config = makeHybridEngineConfig();
    config.modelType = "pure-recurrent-coordinator-test";
    config.numDecoderLayers = 1;
    config.numAttentionLayers = 0;
    config.numKVHeads = 0;
    config.headDim = 0;
    config.rotaryDim = 0;
    config.layerTypes = {HybridCacheManager::LayerType::kMamba};
    config.kvLayerConfigs.clear();
    return config;
}

HybridCacheManager::Config makeCacheConfig(LLMEngineConfig const& engine)
{
    KVCacheManager::Config kvConfig{engine.numAttentionLayers, engine.maxSupportedBatchSize, engine.maxKVCacheCapacity,
        engine.kvLayerConfigs, engine.kvCacheDtype, engine.kvPoolPages};
    MambaCacheManager::Config mambaConfig{engine.numLinearAttnLayers, engine.maxSupportedBatchSize,
        engine.recurrentStateNumHeads, engine.recurrentStateHeadDim, engine.recurrentStateSize, engine.convDim,
        engine.convKernel, 0, engine.recurrentStateDtype, engine.convStateDtype};
    return HybridCacheManager::Config{
        engine.layerTypes, std::move(kvConfig), std::move(mambaConfig), engine.maxSupportedBatchSize};
}

class ContextCacheHybridCoordinatorTests : public ::testing::Test
{
protected:
    void SetUp() override
    {
        ASSERT_EQ(cudaStreamCreate(&mStream), cudaSuccess);
        mEngine = makeHybridEngineConfig();
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

    void createCoordinator(ContextCacheCoordinator::StreamSynchronizer synchronizer = {}, int32_t snapshotSlots = 4)
    {
        ASSERT_GT(snapshotSlots, 0);
        ContextCachePhysicalResources resources{*mCache, *mPageTable, nullptr, nullptr};
        ContextCacheConfig config{/*.enabled=*/true, /*.maxRecords=*/16,
            /*.recurrentSnapshotPoolBytes=*/
            static_cast<int64_t>(static_cast<size_t>(snapshotSlots)
                * HybridSnapshotStorage::recurrentBytesPerSlot(mCache->getMambaCacheManager().getConfig())),
            /*.partialKvSnapshotPoolBytes=*/
            static_cast<int64_t>(static_cast<size_t>(snapshotSlots)
                * HybridSnapshotStorage::partialKvBytesPerSlot(mCache->getKVCacheManager().getConfig()))};
        mCoordinator = std::make_unique<ContextCacheCoordinator>(config, mDeployment,
            validateContextCacheDeployment(mDeployment), resources, mStream, std::move(synchronizer));
    }

    void recreateAsPureRecurrent()
    {
        ASSERT_EQ(mCoordinator->shutdown(), ContextCacheCoordinatorStatus::kOk);
        mCoordinator.reset();
        mPageTable.reset();
        mCache.reset();
        mEngine = makePureRecurrentEngineConfig();
        mDeployment = DeploymentConfig{mEngine, std::nullopt, std::nullopt};
        mCache = std::make_unique<HybridCacheManager>(makeCacheConfig(mEngine), mStream);
        KVCacheManager const& kv = mCache->getKVCacheManager();
        mPageTable = std::make_unique<KVPageTable>(kMAX_BATCH, pagesPerSlot(kv.maxCapPadded()), kv.numPages());
        mPageTable->setIdentity();
        mPageTable->upload(mStream);
        ASSERT_EQ(cudaStreamSynchronize(mStream), cudaSuccess);
        createCoordinator();
    }

    ContextCacheCoordinator::AdmissionResult begin(std::vector<int32_t> tokens, BlockKeyExtras keyExtras = {})
    {
        ContextCacheBatchAdmission admission;
        admission.sequences.push_back(ContextCacheSequenceAdmission{std::move(tokens), std::move(keyExtras)});
        ContextCacheCoordinator::BeginRequestResult result
            = mCoordinator->beginRequest(admission, DecodingKvHeadroom{1, 0}, mStream);
        EXPECT_EQ(result.status, ContextCacheCoordinatorStatus::kOk);
        EXPECT_TRUE(result.admission.has_value());
        return std::move(*result.admission);
    }

    void seedRecurrent(uint8_t recurrentPattern, uint8_t convPattern)
    {
        MambaCacheManager& mamba = mCache->getMambaCacheManager();
        MambaCacheManager::Config const& config = mamba.getConfig();
        size_t const recurrentBytes = static_cast<size_t>(config.recurrentStateNumHeads * config.recurrentStateHeadDim
                                          * config.recurrentStateSize)
            * sizeof(half);
        size_t const convBytes = static_cast<size_t>(config.convDim * config.convKernel) * sizeof(half);
        CUDA_CHECK(cudaMemsetAsync(mamba.getRecurrentState(0).rawPointer(), recurrentPattern, recurrentBytes, mStream));
        CUDA_CHECK(cudaMemsetAsync(mamba.getConvState(0).rawPointer(), convPattern, convBytes, mStream));
    }

    void expectRecurrent(uint8_t recurrentPattern, uint8_t convPattern)
    {
        MambaCacheManager& mamba = mCache->getMambaCacheManager();
        MambaCacheManager::Config const& config = mamba.getConfig();
        size_t const recurrentBytes = static_cast<size_t>(config.recurrentStateNumHeads * config.recurrentStateHeadDim
                                          * config.recurrentStateSize)
            * sizeof(half);
        size_t const convBytes = static_cast<size_t>(config.convDim * config.convKernel) * sizeof(half);
        std::vector<uint8_t> recurrent(recurrentBytes);
        std::vector<uint8_t> conv(convBytes);
        ASSERT_EQ(cudaMemcpy(recurrent.data(), mamba.getRecurrentState(0).rawPointer(), recurrentBytes,
                      cudaMemcpyDeviceToHost),
            cudaSuccess);
        ASSERT_EQ(cudaMemcpy(conv.data(), mamba.getConvState(0).rawPointer(), convBytes, cudaMemcpyDeviceToHost),
            cudaSuccess);
        EXPECT_TRUE(std::all_of(recurrent.begin(), recurrent.end(),
            [recurrentPattern](uint8_t value) { return value == recurrentPattern; }));
        EXPECT_TRUE(
            std::all_of(conv.begin(), conv.end(), [convPattern](uint8_t value) { return value == convPattern; }));
    }

    void seedKvPage(PageId page, uint8_t keyPattern, uint8_t valuePattern)
    {
        KVCacheManager& kv = mCache->getKVCacheManager();
        KVLayerConfig const& layer = kv.getLayerConfig(0);
        size_t const pageBytes
            = static_cast<size_t>(kTOKENS_PER_PAGE * layer.numKVHeads * layer.headDim) * sizeof(half);
        CUDA_CHECK(cudaMemsetAsync(static_cast<uint8_t*>(kv.kPoolPtr(0)) + static_cast<size_t>(page) * pageBytes,
            keyPattern, pageBytes, mStream));
        CUDA_CHECK(cudaMemsetAsync(static_cast<uint8_t*>(kv.vPoolPtr(0)) + static_cast<size_t>(page) * pageBytes,
            valuePattern, pageBytes, mStream));
    }

    void expectKvPrefix(PageId page, int32_t validTokens, uint8_t keyPattern, uint8_t valuePattern)
    {
        KVCacheManager& kv = mCache->getKVCacheManager();
        KVLayerConfig const& layer = kv.getLayerConfig(0);
        size_t const tokenBytes = static_cast<size_t>(layer.numKVHeads * layer.headDim) * sizeof(half);
        size_t const copiedBytes = static_cast<size_t>(validTokens) * tokenBytes;
        size_t const pageBytes = static_cast<size_t>(kTOKENS_PER_PAGE) * tokenBytes;
        std::vector<uint8_t> keys(copiedBytes);
        std::vector<uint8_t> values(copiedBytes);
        ASSERT_EQ(
            cudaMemcpy(keys.data(), static_cast<uint8_t const*>(kv.kPoolPtr(0)) + static_cast<size_t>(page) * pageBytes,
                copiedBytes, cudaMemcpyDeviceToHost),
            cudaSuccess);
        ASSERT_EQ(cudaMemcpy(values.data(),
                      static_cast<uint8_t const*>(kv.vPoolPtr(0)) + static_cast<size_t>(page) * pageBytes, copiedBytes,
                      cudaMemcpyDeviceToHost),
            cudaSuccess);
        EXPECT_TRUE(std::all_of(keys.begin(), keys.end(), [keyPattern](uint8_t value) { return value == keyPattern; }));
        EXPECT_TRUE(
            std::all_of(values.begin(), values.end(), [valuePattern](uint8_t value) { return value == valuePattern; }));
    }

    void publishPrefill(ContextCacheCoordinator::AdmissionResult& request, int32_t inputLength, int32_t lookahead)
    {
        ASSERT_EQ(mCoordinator->enqueuePrefillCaptures(request.request), ContextCacheCoordinatorStatus::kOk);
        ASSERT_EQ(cudaStreamSynchronize(mStream), cudaSuccess);
        std::vector<ContextCacheSequenceAdvance> progress{ContextCacheSequenceAdvance{&lookahead, 1, inputLength}};
        ASSERT_EQ(
            mCoordinator->finalizePrefillPublication(request.request, progress), ContextCacheCoordinatorStatus::kOk);
    }

    cudaStream_t mStream{};
    LLMEngineConfig mEngine;
    DeploymentConfig mDeployment;
    std::unique_ptr<HybridCacheManager> mCache;
    std::unique_ptr<KVPageTable> mPageTable;
    std::unique_ptr<ContextCacheCoordinator> mCoordinator;
};

TEST_F(ContextCacheHybridCoordinatorTests, PrefillCheckpointRestoresRecurrentAndPartialKvAtomically)
{
    constexpr int32_t kInputLength{kTOKENS_PER_PAGE + 1};
    auto producer = begin(makeTokens(kInputLength));
    ASSERT_EQ(mCoordinator->preparePrefill(producer.request), ContextCacheCoordinatorStatus::kOk);
    PageId const producerPartialPage = mPageTable->hostRow(0)[1];
    seedRecurrent(0x31, 0x61);
    seedKvPage(producerPartialPage, 0x21, 0x71);
    publishPrefill(producer, kInputLength, 9001);
    ASSERT_EQ(mCoordinator->manager().records().size(), 1U);

    seedRecurrent(0, 0);
    seedKvPage(producerPartialPage, 0, 0);
    ASSERT_EQ(cudaStreamSynchronize(mStream), cudaSuccess);
    ASSERT_EQ(mCoordinator->finish(producer.request), ContextCacheCoordinatorStatus::kOk);

    ContextCacheMetrics const metricsBeforeConsumer = mCoordinator->metrics();
    auto consumer = begin(makeTokens(kInputLength + 1));
    ASSERT_EQ(consumer.prefillStarts, std::vector<int32_t>{kInputLength});
    ContextCacheMetrics const metricsAfterConsumer = mCoordinator->metrics();
    EXPECT_EQ(
        metricsAfterConsumer.matchedTokens, metricsBeforeConsumer.matchedTokens + static_cast<uint64_t>(kInputLength));
    ASSERT_EQ(mCoordinator->preparePrefill(consumer.request), ContextCacheCoordinatorStatus::kOk);
    ASSERT_EQ(cudaStreamSynchronize(mStream), cudaSuccess);
    expectRecurrent(0x31, 0x61);
    expectKvPrefix(mPageTable->hostRow(0)[1], 1, 0x21, 0x71);
    EXPECT_EQ(mCoordinator->metrics().hybridRestores, 1U);
    EXPECT_EQ(mCoordinator->finish(consumer.request), ContextCacheCoordinatorStatus::kOk);
}

TEST_F(ContextCacheHybridCoordinatorTests, CandidateHashMatchesPublishedAlignedAndPartialPrefixesWithExtras)
{
    std::vector<int32_t> const checkpointLengths{kTOKENS_PER_PAGE, kTOKENS_PER_PAGE + 1};
    for (size_t index = 0; index < checkpointLengths.size(); ++index)
    {
        int32_t const checkpointLength = checkpointLengths[index];
        BlockKeyExtras identity;
        identity.isolationDigest = Hash128{0x5152535455565758ULL + index, 0x6162636465666768ULL + index};

        auto producer = begin(makeTokens(checkpointLength), identity);
        ASSERT_EQ(mCoordinator->preparePrefill(producer.request), ContextCacheCoordinatorStatus::kOk);
        publishPrefill(producer, checkpointLength, 9001);
        ASSERT_EQ(mCoordinator->finish(producer.request), ContextCacheCoordinatorStatus::kOk);

        ContextCacheMetrics const metricsBeforeConsumer = mCoordinator->metrics();
        auto consumer = begin(makeTokens(checkpointLength + 1), identity);
        EXPECT_EQ(consumer.prefillStarts, std::vector<int32_t>{checkpointLength});
        ContextCacheMetrics const metricsAfterConsumer = mCoordinator->metrics();
        EXPECT_EQ(metricsAfterConsumer.matchedTokens,
            metricsBeforeConsumer.matchedTokens + static_cast<uint64_t>(checkpointLength));
        EXPECT_EQ(mCoordinator->finish(consumer.request), ContextCacheCoordinatorStatus::kOk);
    }
}

TEST_F(ContextCacheHybridCoordinatorTests, DecodeEndCheckpointUsesOneCaptureSyncAndBecomesReusable)
{
    ASSERT_EQ(mCoordinator->shutdown(), ContextCacheCoordinatorStatus::kOk);
    mCoordinator.reset();
    int32_t captureSyncs{};
    createCoordinator([&](cudaStream_t stream) {
        ++captureSyncs;
        return cudaStreamSynchronize(stream);
    });

    constexpr int32_t kInputLength{kTOKENS_PER_PAGE + 1};
    auto producer = begin(makeTokens(kInputLength));
    ASSERT_EQ(mCoordinator->preparePrefill(producer.request), ContextCacheCoordinatorStatus::kOk);
    seedRecurrent(0x31, 0x61);
    seedKvPage(mPageTable->hostRow(0)[1], 0x21, 0x71);
    publishPrefill(producer, kInputLength, 9001);

    ASSERT_EQ(mCoordinator->prepareDecodeStep(producer.request, DecodingKvHeadroom{1, 0}),
        ContextCacheCoordinatorStatus::kOk);
    seedRecurrent(0x32, 0x62);
    seedKvPage(mPageTable->hostRow(0)[1], 0x22, 0x72);
    ASSERT_EQ(cudaStreamSynchronize(mStream), cudaSuccess);
    int32_t const nextLookahead = 9002;
    std::vector<ContextCacheSequenceAdvance> progress{ContextCacheSequenceAdvance{&nextLookahead, 1, kInputLength + 1}};
    ASSERT_EQ(mCoordinator->completeDecodeStep(producer.request, progress, {0}), ContextCacheCoordinatorStatus::kOk);
    EXPECT_EQ(captureSyncs, 1);
    EXPECT_EQ(mCoordinator->metrics().hybridCaptureSynchronizations, 1U);
    ASSERT_EQ(mCoordinator->finish(producer.request), ContextCacheCoordinatorStatus::kOk);

    std::vector<int32_t> continuation = makeTokens(kInputLength);
    continuation.push_back(9001);
    continuation.push_back(42);
    auto consumer = begin(std::move(continuation));
    EXPECT_EQ(consumer.prefillStarts, std::vector<int32_t>{kInputLength + 1});
    ASSERT_EQ(mCoordinator->preparePrefill(consumer.request), ContextCacheCoordinatorStatus::kOk);
    ASSERT_EQ(cudaStreamSynchronize(mStream), cudaSuccess);
    expectRecurrent(0x32, 0x62);
    expectKvPrefix(mPageTable->hostRow(0)[1], 2, 0x22, 0x72);
    EXPECT_EQ(mCoordinator->finish(consumer.request), ContextCacheCoordinatorStatus::kOk);
}

TEST_F(ContextCacheHybridCoordinatorTests, SnapshotPressureSkipsCaptureAndPreservesReusableRecord)
{
    ASSERT_EQ(mCoordinator->shutdown(), ContextCacheCoordinatorStatus::kOk);
    mCoordinator.reset();
    createCoordinator({}, 1);

    constexpr int32_t kCheckpointLength{kTOKENS_PER_PAGE + 1};
    auto producer = begin(makeTokens(kCheckpointLength));
    ASSERT_EQ(mCoordinator->preparePrefill(producer.request), ContextCacheCoordinatorStatus::kOk);
    publishPrefill(producer, kCheckpointLength, 9001);
    ASSERT_EQ(mCoordinator->finish(producer.request), ContextCacheCoordinatorStatus::kOk);
    ASSERT_EQ(mCoordinator->manager().records().size(), 1U);

    auto consumer = begin(makeTokens(kCheckpointLength + 1));
    ASSERT_EQ(consumer.prefillStarts, std::vector<int32_t>{kCheckpointLength});
    ASSERT_EQ(mCoordinator->preparePrefill(consumer.request), ContextCacheCoordinatorStatus::kOk);
    publishPrefill(consumer, kCheckpointLength + 1, 9002);

    ContextCacheMetrics const metrics = mCoordinator->metrics();
    EXPECT_EQ(metrics.hybridSnapshotPressureSkips, 1U);
    EXPECT_EQ(metrics.publicationAttempts, 1U);
    EXPECT_EQ(metrics.currentRecords, 1U);
    EXPECT_EQ(metrics.recurrentSnapshots.free, 0);
    EXPECT_EQ(metrics.recurrentSnapshots.capacity, 1);
    EXPECT_EQ(metrics.partialKvSnapshots.free, 0);
    EXPECT_EQ(metrics.partialKvSnapshots.capacity, 1);
    EXPECT_EQ(mCoordinator->finish(consumer.request), ContextCacheCoordinatorStatus::kOk);
}

TEST_F(ContextCacheHybridCoordinatorTests, PureRecurrentCheckpointNeedsNoKvPage)
{
    recreateAsPureRecurrent();
    auto producer = begin(makeTokens(3));
    ASSERT_EQ(mCoordinator->preparePrefill(producer.request), ContextCacheCoordinatorStatus::kOk);
    seedRecurrent(0x41, 0x71);
    publishPrefill(producer, 3, 9001);
    seedRecurrent(0, 0);
    ASSERT_EQ(cudaStreamSynchronize(mStream), cudaSuccess);
    ASSERT_EQ(mCoordinator->finish(producer.request), ContextCacheCoordinatorStatus::kOk);

    auto consumer = begin(makeTokens(4));
    EXPECT_EQ(consumer.prefillStarts, std::vector<int32_t>{3});
    ASSERT_EQ(mCoordinator->preparePrefill(consumer.request), ContextCacheCoordinatorStatus::kOk);
    ASSERT_EQ(cudaStreamSynchronize(mStream), cudaSuccess);
    expectRecurrent(0x41, 0x71);
    EXPECT_EQ(mCoordinator->finish(consumer.request), ContextCacheCoordinatorStatus::kOk);
}

} // namespace
