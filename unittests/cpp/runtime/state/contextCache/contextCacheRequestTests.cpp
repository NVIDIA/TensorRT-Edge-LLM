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

#include "runtime/contextCacheRequest.h"

#include "common/checkMacros.h"
#include "common/pagedKvTypes.h"
#include "common/tensor.h"
#include "runtime/config/deploymentConfig.h"
#include "runtime/decoding/decodingStrategy.h"
#include "runtime/hybridCacheManager.h"
#include "runtime/llmRuntimeUtils.h"
#include "runtime/state/contextCache/contextCacheDeployment.h"
#include "runtime/state/contextCache/hybridSnapshotStorage.h"
#include "runtime/state/decodingInferenceContext.h"
#include "runtime/state/kvPageTable.h"
#include "runtime/streaming.h"

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <utility>
#include <vector>

using namespace nvinfer1;
using namespace trt_edgellm;
using namespace trt_edgellm::rt;

namespace
{

constexpr int32_t kMAX_BATCH{3};
constexpr int32_t kMAX_SEQUENCE_LENGTH{512};
constexpr int32_t kMAX_GENERATE_LENGTH{64};

std::vector<int32_t> makeTokens(int32_t count)
{
    std::vector<int32_t> tokens(static_cast<size_t>(count));
    std::iota(tokens.begin(), tokens.end(), 1);
    return tokens;
}

LLMEngineConfig makeAttentionConfig(char const* modelType, int32_t layers = 1)
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

LLMEngineConfig makeHybridConfig()
{
    LLMEngineConfig config = makeAttentionConfig("context-cache-request-hybrid");
    config.numDecoderLayers = 2;
    config.numLinearAttnLayers = 1;
    config.layerTypes.push_back(HybridCacheManager::LayerType::kMamba);
    config.recurrentStateNumHeads = 2;
    config.recurrentStateHeadDim = 3;
    config.recurrentStateSize = 4;
    config.convDim = 5;
    config.convKernel = 6;
    config.recurrentStateDtype = DataType::kHALF;
    config.convStateDtype = DataType::kHALF;
    return config;
}

DeploymentConfig makeEagleDeployment()
{
    DeploymentConfig deployment;
    deployment.base = makeAttentionConfig("context-cache-request-eagle-base", 3);
    deployment.base.specDecodeType = SpecDecodeMode::kEAGLE;
    deployment.base.isSpecDecodeBase = true;
    deployment.base.specTargetLayerIds = {0, 2};
    deployment.draft = makeAttentionConfig("context-cache-request-eagle-draft", 2);
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
    MambaCacheManager::Config mambaConfig{engine.numLinearAttnLayers, engine.maxSupportedBatchSize,
        engine.recurrentStateNumHeads, engine.recurrentStateHeadDim, engine.recurrentStateSize, engine.convDim,
        engine.convKernel, 0, engine.recurrentStateDtype, engine.convStateDtype};
    return HybridCacheManager::Config{
        engine.layerTypes, std::move(kvConfig), std::move(mambaConfig), engine.maxSupportedBatchSize};
}

std::unique_ptr<KVPageTable> makePageTable(HybridCacheManager& cache, cudaStream_t stream)
{
    KVCacheManager const& kv = cache.getKVCacheManager();
    auto table = std::make_unique<KVPageTable>(kMAX_BATCH, pagesPerSlot(kv.maxCapPadded()), kv.numPages());
    table->setIdentity();
    table->upload(stream);
    return table;
}

DecodingInferenceContext makeContext(
    std::vector<std::vector<int32_t>> inputs, cudaStream_t stream, std::string const& loraName = "")
{
    DecodingInferenceContext context;
    context.initialize(static_cast<int32_t>(inputs.size()), kMAX_GENERATE_LENGTH, std::nullopt,
        rt::OptionalInputTensors{}, loraName, stream);
    context.rawBatchedInputIds = std::move(inputs);
    context.outputThinkerEmbeddings = false;
    return context;
}

void stagePrefillResult(ContextCacheRequest const& request, DecodingInferenceContext& context)
{
    std::vector<int32_t> const& prefillStarts = request.prefillStarts();
    ASSERT_EQ(prefillStarts.size(), context.rawBatchedInputIds.size());
    for (size_t slot = 0; slot < context.rawBatchedInputIds.size(); ++slot)
    {
        int32_t const start = prefillStarts[slot];
        std::vector<int32_t> const& raw = context.rawBatchedInputIds[slot];
        ASSERT_GE(start, 0);
        ASSERT_LE(static_cast<size_t>(start), raw.size());
        context.tokenIds[slot].assign(raw.begin() + start, raw.end());
        context.tokenIds[slot].push_back(9001 + static_cast<int32_t>(slot));
        context.effectivePrefillLengths[slot] = static_cast<int32_t>(raw.size()) - start;
        context.currentGenerateLengths[slot] = 1;
    }
}

void completeRuntimePrefill(ContextCacheRequest& request, DecodingInferenceContext& context,
    std::vector<int32_t> const& commonStateLengths = {})
{
    ASSERT_TRUE(request.preparePrefill());
    stagePrefillResult(request, context);
    ASSERT_TRUE(request.enqueuePrefillCaptures());
    ASSERT_EQ(cudaStreamSynchronize(context.stream), cudaSuccess);
    ASSERT_TRUE(request.completePrefill(context, commonStateLengths));
}

class ContextCacheRequestTests : public ::testing::Test
{
protected:
    void SetUp() override
    {
        ASSERT_EQ(cudaStreamCreate(&mStream), cudaSuccess);
        mDeployment
            = DeploymentConfig{makeAttentionConfig("context-cache-request-vanilla"), std::nullopt, std::nullopt};
        mCache = std::make_unique<HybridCacheManager>(makeCacheConfig(mDeployment.base), mStream);
        mPageTable = makePageTable(*mCache, mStream);
        ASSERT_EQ(cudaStreamSynchronize(mStream), cudaSuccess);
        ContextCachePhysicalResources resources{*mCache, *mPageTable, nullptr, nullptr};
        mCoordinator
            = std::make_unique<ContextCacheCoordinator>(ContextCacheConfig{/*.enabled=*/true, /*.maxRecords=*/16},
                mDeployment, validateContextCacheDeployment(mDeployment), resources, mStream);
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

    std::optional<ContextCacheRequest> begin(DecodingInferenceContext const& context, bool speculativeRequest = false)
    {
        LLMGenerationRequest request{};
        return ContextCacheRequest::begin(
            *mCoordinator, request, context, speculativeRequest, DecodingKvHeadroom{1, 0});
    }

    cudaStream_t mStream{};
    DeploymentConfig mDeployment;
    std::unique_ptr<HybridCacheManager> mCache;
    std::unique_ptr<KVPageTable> mPageTable;
    std::unique_ptr<ContextCacheCoordinator> mCoordinator;
};

TEST_F(ContextCacheRequestTests, ColdPublishThenWarmLookupPreservesLoraIdentity)
{
    DecodingInferenceContext producerContext = makeContext({makeTokens(129)}, mStream, "adapter-a");
    auto producer = begin(producerContext);
    ASSERT_TRUE(producer.has_value());
    EXPECT_EQ(producer->prefillStarts(), std::vector<int32_t>{0});
    completeRuntimePrefill(*producer, producerContext);
    ASSERT_TRUE(producer->finish());

    DecodingInferenceContext consumerContext = makeContext({makeTokens(130)}, mStream, "adapter-a");
    auto consumer = begin(consumerContext);
    ASSERT_TRUE(consumer.has_value());
    EXPECT_EQ(consumer->prefillStarts(), std::vector<int32_t>{kTOKENS_PER_PAGE});
    ASSERT_TRUE(consumer->finish());

    DecodingInferenceContext isolatedContext = makeContext({makeTokens(130)}, mStream, "adapter-b");
    auto isolated = begin(isolatedContext);
    ASSERT_TRUE(isolated.has_value());
    EXPECT_EQ(isolated->prefillStarts(), std::vector<int32_t>{0});
    ASSERT_TRUE(isolated->finish());
}

TEST_F(ContextCacheRequestTests, VanillaDecodePublishesOneTokenPageBoundaryAndRejectsPairingMisuse)
{
    DecodingInferenceContext context = makeContext({makeTokens(255)}, mStream);
    auto request = begin(context);
    ASSERT_TRUE(request.has_value());
    completeRuntimePrefill(*request, context);

    ASSERT_TRUE(request->prepareDecodeStep(context, DecodingKvHeadroom{1, 0}));
    EXPECT_THROW(request->prepareDecodeStep(context, DecodingKvHeadroom{1, 0}), std::runtime_error);
    ASSERT_EQ(cudaStreamSynchronize(mStream), cudaSuccess);
    context.tokenIds[0].push_back(9002);
    ++context.currentGenerateLengths[0];
    context.finishedStates[0] = 1;
    context.slotStreams[0].terminalReason = FinishReason::kLength;
    ASSERT_TRUE(request->completeDecodeStep(context, {}));
    EXPECT_THROW(request->completeDecodeStep(context, {}), std::runtime_error);

    ASSERT_FALSE(mCoordinator->manager().records().lruToMru().empty());
    CacheRecord const& record
        = mCoordinator->manager().records().get(mCoordinator->manager().records().lruToMru().back());
    EXPECT_EQ(record.basePagePath.size(), 2U);
    ASSERT_TRUE(request->finish());
}

TEST_F(ContextCacheRequestTests, CancelledAndErrorSlotsAreNotPublished)
{
    std::vector<int32_t> secondPrompt = makeTokens(255);
    secondPrompt.back() = 10000;
    DecodingInferenceContext context = makeContext({makeTokens(255), std::move(secondPrompt)}, mStream);
    auto request = begin(context);
    ASSERT_TRUE(request.has_value());
    completeRuntimePrefill(*request, context);

    ContextCacheMetrics const beforeDecode = mCoordinator->metrics();
    ASSERT_TRUE(request->prepareDecodeStep(context, DecodingKvHeadroom{1, 0}));
    ASSERT_EQ(cudaStreamSynchronize(mStream), cudaSuccess);
    for (int32_t slot = 0; slot < context.activeBatchSize; ++slot)
    {
        context.tokenIds[slot].push_back(9100 + slot);
        ++context.currentGenerateLengths[slot];
        context.finishedStates[slot] = 1;
    }
    context.slotStreams[0].terminalReason = FinishReason::kCancelled;
    context.slotStreams[1].terminalReason = FinishReason::kError;
    ASSERT_TRUE(request->completeDecodeStep(context, {}));

    ContextCacheMetrics const afterDecode = mCoordinator->metrics();
    EXPECT_EQ(afterDecode.publicationAttempts, beforeDecode.publicationAttempts);
    ASSERT_TRUE(request->finish());
}

TEST_F(ContextCacheRequestTests, BatchCompactionUsesThePreparedMapping)
{
    DecodingInferenceContext context = makeContext({makeTokens(8), makeTokens(9)}, mStream);
    auto request = begin(context);
    ASSERT_TRUE(request.has_value());
    completeRuntimePrefill(*request, context);

    std::vector<PageId> const survivingRow(
        mPageTable->hostRow(1), mPageTable->hostRow(1) + mPageTable->maxPagesPerSeq());
    Tensor deviceMapping({kMAX_BATCH}, rt::DeviceType::kGPU, DataType::kINT32, "contextCacheRequestMapping");
    ASSERT_TRUE(request->beginBatchCompaction({-1, 0}, 1, deviceMapping));
    std::vector<int32_t> uploadedMapping(2);
    ASSERT_EQ(cudaMemcpy(uploadedMapping.data(), deviceMapping.rawPointer(), uploadedMapping.size() * sizeof(int32_t),
                  cudaMemcpyDeviceToHost),
        cudaSuccess);
    EXPECT_EQ(uploadedMapping, (std::vector<int32_t>{-1, 0}));
    ASSERT_TRUE(request->completeBatchCompaction());
    EXPECT_TRUE(std::equal(survivingRow.begin(), survivingRow.end(), mPageTable->hostRow(0)));
    ASSERT_TRUE(request->finish());
}

TEST_F(ContextCacheRequestTests, EarlyDestructionAbandonsPendingRequest)
{
    DecodingInferenceContext abandonedContext = makeContext({makeTokens(8)}, mStream);
    {
        auto abandoned = begin(abandonedContext);
        ASSERT_TRUE(abandoned.has_value());
        ASSERT_TRUE(abandoned->preparePrefill());
    }

    DecodingInferenceContext nextContext = makeContext({makeTokens(8)}, mStream);
    auto next = begin(nextContext);
    ASSERT_TRUE(next.has_value());
    ASSERT_TRUE(next->finish());
}

TEST_F(ContextCacheRequestTests, AdmissionUsesValidatedCoordinatorContract)
{
    DecodingInferenceContext context = makeContext({makeTokens(8)}, mStream);
    EXPECT_THROW(static_cast<void>(begin(context, true)), std::runtime_error);
    auto vanilla = begin(context);
    ASSERT_TRUE(vanilla.has_value());
    ASSERT_TRUE(vanilla->finish());
}

class ContextCacheRequestHybridTests : public ::testing::Test
{
protected:
    void SetUp() override
    {
        ASSERT_EQ(cudaStreamCreate(&mStream), cudaSuccess);
        mDeployment = DeploymentConfig{makeHybridConfig(), std::nullopt, std::nullopt};
        mCache = std::make_unique<HybridCacheManager>(makeCacheConfig(mDeployment.base), mStream);
        mPageTable = makePageTable(*mCache, mStream);
        ASSERT_EQ(cudaStreamSynchronize(mStream), cudaSuccess);
        constexpr int32_t kSNAPSHOT_SLOTS{4};
        ContextCacheConfig config{/*.enabled=*/true, /*.maxRecords=*/16,
            /*.recurrentSnapshotPoolBytes=*/
            static_cast<int64_t>(static_cast<size_t>(kSNAPSHOT_SLOTS)
                * HybridSnapshotStorage::recurrentBytesPerSlot(mCache->getMambaCacheManager().getConfig())),
            /*.partialKvSnapshotPoolBytes=*/
            static_cast<int64_t>(static_cast<size_t>(kSNAPSHOT_SLOTS)
                * HybridSnapshotStorage::partialKvBytesPerSlot(mCache->getKVCacheManager().getConfig()))};
        ContextCachePhysicalResources resources{*mCache, *mPageTable, nullptr, nullptr};
        mCoordinator = std::make_unique<ContextCacheCoordinator>(
            config, mDeployment, validateContextCacheDeployment(mDeployment), resources, mStream);
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

    std::optional<ContextCacheRequest> begin(DecodingInferenceContext const& context)
    {
        LLMGenerationRequest request{};
        return ContextCacheRequest::begin(*mCoordinator, request, context, false, DecodingKvHeadroom{1, 0});
    }

    cudaStream_t mStream{};
    DeploymentConfig mDeployment;
    std::unique_ptr<HybridCacheManager> mCache;
    std::unique_ptr<KVPageTable> mPageTable;
    std::unique_ptr<ContextCacheCoordinator> mCoordinator;
};

TEST_F(ContextCacheRequestHybridTests, PrefillCapturePublishesAnExactReusableCheckpoint)
{
    constexpr int32_t kINPUT_LENGTH{kTOKENS_PER_PAGE + 1};
    DecodingInferenceContext producerContext = makeContext({makeTokens(kINPUT_LENGTH)}, mStream);
    auto producer = begin(producerContext);
    ASSERT_TRUE(producer.has_value());
    completeRuntimePrefill(*producer, producerContext);
    ASSERT_TRUE(producer->finish());

    DecodingInferenceContext consumerContext = makeContext({makeTokens(kINPUT_LENGTH + 1)}, mStream);
    auto consumer = begin(consumerContext);
    ASSERT_TRUE(consumer.has_value());
    EXPECT_EQ(consumer->prefillStarts(), std::vector<int32_t>{kINPUT_LENGTH});
    ContextCacheMetrics const beforeRestore = mCoordinator->metrics();
    ASSERT_TRUE(consumer->preparePrefill());
    ASSERT_EQ(cudaStreamSynchronize(mStream), cudaSuccess);
    EXPECT_EQ(mCoordinator->metrics().hybridRestores, beforeRestore.hybridRestores + 1U);
    ASSERT_TRUE(consumer->finish());
}

class ContextCacheRequestEagleTests : public ::testing::Test
{
protected:
    void SetUp() override
    {
        ASSERT_EQ(cudaStreamCreate(&mStream), cudaSuccess);
        mDeployment = makeEagleDeployment();
        mBaseCache = std::make_unique<HybridCacheManager>(makeCacheConfig(mDeployment.base), mStream);
        mDraftCache = std::make_unique<HybridCacheManager>(makeCacheConfig(*mDeployment.draft), mStream);
        mBasePageTable = makePageTable(*mBaseCache, mStream);
        mDraftPageTable = makePageTable(*mDraftCache, mStream);
        ASSERT_EQ(cudaStreamSynchronize(mStream), cudaSuccess);
        ContextCachePhysicalResources resources{*mBaseCache, *mBasePageTable, mDraftCache.get(), mDraftPageTable.get()};
        mCoordinator
            = std::make_unique<ContextCacheCoordinator>(ContextCacheConfig{/*.enabled=*/true, /*.maxRecords=*/16},
                mDeployment, validateContextCacheDeployment(mDeployment), resources, mStream);
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

    std::optional<ContextCacheRequest> begin(DecodingInferenceContext const& context, bool speculativeRequest = true)
    {
        LLMGenerationRequest request{};
        DecodingKvHeadroom const headroom = speculativeRequest ? DecodingKvHeadroom{4, 4} : DecodingKvHeadroom{1, 0};
        return ContextCacheRequest::begin(*mCoordinator, request, context, speculativeRequest, headroom);
    }

    cudaStream_t mStream{};
    DeploymentConfig mDeployment;
    std::unique_ptr<HybridCacheManager> mBaseCache;
    std::unique_ptr<HybridCacheManager> mDraftCache;
    std::unique_ptr<KVPageTable> mBasePageTable;
    std::unique_ptr<KVPageTable> mDraftPageTable;
    std::unique_ptr<ContextCacheCoordinator> mCoordinator;
};

TEST_F(ContextCacheRequestEagleTests, VanillaRequestDoesNotAcquireDraftState)
{
    ContextCacheMetrics const before = mCoordinator->metrics();
    DecodingInferenceContext context = makeContext({makeTokens(2 * kTOKENS_PER_PAGE)}, mStream);

    auto request = begin(context, false);

    ASSERT_TRUE(request.has_value());
    ContextCacheMetrics const admitted = mCoordinator->metrics();
    EXPECT_LT(admitted.baseKvPages.free, before.baseKvPages.free);
    EXPECT_EQ(admitted.draftKvPages.free, before.draftKvPages.free);
    ASSERT_TRUE(request->finish());
}

TEST_F(ContextCacheRequestEagleTests, ForwardsCommonMaterializedStateAndUsesPairedReplay)
{
    constexpr int32_t kINPUT_LENGTH{2 * kTOKENS_PER_PAGE};
    DecodingInferenceContext producerContext = makeContext({makeTokens(kINPUT_LENGTH)}, mStream);
    auto producer = begin(producerContext);
    ASSERT_TRUE(producer.has_value());
    completeRuntimePrefill(*producer, producerContext, {kINPUT_LENGTH});
    EXPECT_EQ(mCoordinator->manager().records().size(), 0U);

    ASSERT_TRUE(producer->prepareDecodeStep(producerContext, DecodingKvHeadroom{4, 4}));
    ASSERT_EQ(cudaStreamSynchronize(mStream), cudaSuccess);
    producerContext.tokenIds[0].push_back(41);
    producerContext.tokenIds[0].push_back(42);
    producerContext.currentGenerateLengths[0] += 2;
    ASSERT_TRUE(producer->completeDecodeStep(producerContext, {kINPUT_LENGTH}));
    EXPECT_EQ(mCoordinator->manager().records().size(), 1U);
    EXPECT_EQ(mCoordinator->metrics().specPairPublications, 1U);
    ASSERT_TRUE(producer->finish());

    std::vector<int32_t> continuation = makeTokens(kINPUT_LENGTH);
    continuation.push_back(77);
    DecodingInferenceContext consumerContext = makeContext({std::move(continuation)}, mStream);
    auto consumer = begin(consumerContext);
    ASSERT_TRUE(consumer.has_value());
    EXPECT_EQ(consumer->prefillStarts(), std::vector<int32_t>{kTOKENS_PER_PAGE});
    EXPECT_EQ(mCoordinator->metrics().specFullPageReplays, 1U);
    ASSERT_TRUE(consumer->finish());
}

} // namespace
