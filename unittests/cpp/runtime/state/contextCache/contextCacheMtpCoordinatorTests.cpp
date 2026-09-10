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

#include "common/cudaMacros.h"
#include "common/pagedKvTypes.h"
#include "common/tensor.h"
#include "runtime/hybridCacheManager.h"
#include "runtime/state/contextCache/hybridSnapshotStorage.h"
#include "runtime/state/kvPageTable.h"

#include <cuda_fp16.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
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
constexpr int32_t kHIDDEN_SIZE{64};

std::vector<int32_t> makeTokens(int32_t count)
{
    std::vector<int32_t> tokens(static_cast<size_t>(count));
    std::iota(tokens.begin(), tokens.end(), 1);
    return tokens;
}

LLMEngineConfig makeMtpBaseConfig()
{
    LLMEngineConfig config;
    config.modelType = "mtp-coordinator-base";
    config.hiddenSize = kHIDDEN_SIZE;
    config.numDecoderLayers = 2;
    config.numAttentionLayers = 1;
    config.numLinearAttnLayers = 1;
    config.numKVHeads = 1;
    config.headDim = 8;
    config.rotaryDim = 8;
    config.maxSupportedBatchSize = kMAX_BATCH;
    config.maxSupportedInputLength = kMAX_SEQUENCE_LENGTH;
    config.maxKVCacheCapacity = kMAX_SEQUENCE_LENGTH;
    config.kvPoolPages = computeMinimumKvPoolPages(kMAX_BATCH, kMAX_SEQUENCE_LENGTH);
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
    config.specDecodeType = SpecDecodeMode::kMTP;
    config.isSpecDecodeBase = true;
    return config;
}

LLMEngineConfig makeMtpDraftConfig()
{
    LLMEngineConfig config;
    config.modelType = "mtp-coordinator-draft";
    config.hiddenSize = kHIDDEN_SIZE;
    config.numDecoderLayers = 1;
    config.numAttentionLayers = 1;
    config.numLinearAttnLayers = 0;
    config.numKVHeads = 1;
    config.headDim = 8;
    config.rotaryDim = 8;
    config.maxSupportedBatchSize = kMAX_BATCH;
    config.maxSupportedInputLength = kMAX_SEQUENCE_LENGTH;
    config.maxKVCacheCapacity = kMAX_SEQUENCE_LENGTH;
    config.kvPoolPages = computeMinimumKvPoolPages(kMAX_BATCH, kMAX_SEQUENCE_LENGTH);
    config.kvCacheDtype = DataType::kHALF;
    config.layerTypes = {HybridCacheManager::LayerType::kAttention};
    config.kvLayerConfigs = {KVLayerConfig{config.numKVHeads, config.headDim}};
    config.specDecodeType = SpecDecodeMode::kMTP;
    config.isSpecDecodeBase = false;
    // hasOwnKVCache defaults true and sharesTargetKV defaults false, matching the independent-draft MTP contract.
    return config;
}

DeploymentConfig makeMtpDeployment()
{
    DeploymentConfig deployment;
    deployment.base = makeMtpBaseConfig();
    deployment.draft = makeMtpDraftConfig();
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
    MambaCacheManager::Config mambaConfig = engine.numLinearAttnLayers > 0
        ? MambaCacheManager::Config{engine.numLinearAttnLayers, engine.maxSupportedBatchSize,
              engine.recurrentStateNumHeads, engine.recurrentStateHeadDim, engine.recurrentStateSize, engine.convDim,
              engine.convKernel, 0, engine.recurrentStateDtype, engine.convStateDtype}
        : MambaCacheManager::Config{/*.numRecurrentLayers=*/0, engine.maxSupportedBatchSize};
    return HybridCacheManager::Config{
        engine.layerTypes, std::move(kvConfig), std::move(mambaConfig), engine.maxSupportedBatchSize};
}

class ContextCacheMtpCoordinatorTests : public ::testing::Test
{
protected:
    void SetUp() override
    {
        ASSERT_EQ(cudaStreamCreate(&mStream), cudaSuccess);
        mDeployment = makeMtpDeployment();
        mBaseCache = std::make_unique<HybridCacheManager>(makeCacheConfig(mDeployment.base), mStream);
        mDraftCache = std::make_unique<HybridCacheManager>(makeCacheConfig(*mDeployment.draft), mStream);
        mBasePageTable = makePageTable(*mBaseCache);
        mDraftPageTable = makePageTable(*mDraftCache);
        mHidden = std::make_unique<Tensor>(Coords{kMAX_BATCH, kMAX_SEQUENCE_LENGTH, kHIDDEN_SIZE},
            trt_edgellm::rt::DeviceType::kGPU, DataType::kHALF, "ContextCacheMtpCoordinatorTests::hidden");
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
        mHidden.reset();
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

    //! Per-slot cost of one Hybrid+MTP checkpoint: the recurrent axis also carries the boundary hidden row, and the
    //! partial-KV axis also carries the paired draft page.
    size_t recurrentBytesPerSlot() const
    {
        return HybridSnapshotStorage::recurrentBytesPerSlot(mBaseCache->getMambaCacheManager().getConfig())
            + HybridSnapshotStorage::boundaryHiddenBytesPerSlot(kHIDDEN_SIZE, DataType::kHALF);
    }

    size_t partialKvBytesPerSlot() const
    {
        return HybridSnapshotStorage::partialKvBytesPerSlot(mBaseCache->getKVCacheManager().getConfig())
            + HybridSnapshotStorage::partialKvBytesPerSlot(mDraftCache->getKVCacheManager().getConfig());
    }

    void createCoordinator(ContextCacheCoordinator::StreamSynchronizer synchronizer = {}, int32_t snapshotSlots = 4)
    {
        ASSERT_GT(snapshotSlots, 0);
        createCoordinatorWithBudgets(static_cast<int64_t>(static_cast<size_t>(snapshotSlots) * recurrentBytesPerSlot()),
            static_cast<int64_t>(static_cast<size_t>(snapshotSlots) * partialKvBytesPerSlot()),
            std::move(synchronizer));
    }

    void createCoordinatorWithBudgets(int64_t recurrentPoolBytes, int64_t partialKvPoolBytes,
        ContextCacheCoordinator::StreamSynchronizer synchronizer = {})
    {
        ContextCachePhysicalResources resources{*mBaseCache, *mBasePageTable, mDraftCache.get(), mDraftPageTable.get()};
        ContextCacheConfig config{/*.enabled=*/true, /*.maxRecords=*/16, recurrentPoolBytes, partialKvPoolBytes};
        mCoordinator = std::make_unique<ContextCacheCoordinator>(config, mDeployment,
            validateContextCacheDeployment(mDeployment), resources, mStream, std::move(synchronizer));
    }

    ContextCacheCoordinator::AdmissionResult begin(std::vector<int32_t> tokens,
        ContextCacheLookupPolicy lookupPolicy = ContextCacheLookupPolicy::kUseCache, bool speculativeRequest = true)
    {
        ContextCacheBatchAdmission admission;
        admission.speculativeRequest = speculativeRequest;
        DecodingKvHeadroom const headroom = speculativeRequest ? DecodingKvHeadroom{4, 2} : DecodingKvHeadroom{1, 0};
        admission.lookupPolicy = lookupPolicy;
        admission.sequences.push_back(ContextCacheSequenceAdmission{std::move(tokens), {}});
        ContextCacheCoordinator::BeginRequestResult result = mCoordinator->beginRequest(admission, headroom, mStream);
        EXPECT_EQ(result.status, ContextCacheCoordinatorStatus::kOk);
        EXPECT_TRUE(result.admission.has_value());
        return std::move(*result.admission);
    }

    void seedRecurrent(uint8_t recurrentPattern, uint8_t convPattern)
    {
        MambaCacheManager& mamba = mBaseCache->getMambaCacheManager();
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
        MambaCacheManager& mamba = mBaseCache->getMambaCacheManager();
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

    void seedKvPage(HybridCacheManager& cache, PageId page, uint8_t keyPattern, uint8_t valuePattern)
    {
        KVCacheManager& kv = cache.getKVCacheManager();
        KVLayerConfig const& layer = kv.getLayerConfig(0);
        size_t const pageBytes
            = static_cast<size_t>(kTOKENS_PER_PAGE * layer.numKVHeads * layer.headDim) * sizeof(half);
        CUDA_CHECK(cudaMemsetAsync(static_cast<uint8_t*>(kv.kPoolPtr(0)) + static_cast<size_t>(page) * pageBytes,
            keyPattern, pageBytes, mStream));
        CUDA_CHECK(cudaMemsetAsync(static_cast<uint8_t*>(kv.vPoolPtr(0)) + static_cast<size_t>(page) * pageBytes,
            valuePattern, pageBytes, mStream));
    }

    void expectKvPrefix(
        HybridCacheManager& cache, PageId page, int32_t validTokens, uint8_t keyPattern, uint8_t valuePattern)
    {
        KVCacheManager& kv = cache.getKVCacheManager();
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

    size_t hiddenRowBytes() const
    {
        return static_cast<size_t>(kHIDDEN_SIZE) * sizeof(half);
    }

    void seedHiddenRow(int32_t batchSlot, int32_t position, uint8_t pattern)
    {
        size_t const rowBytes = hiddenRowBytes();
        size_t const rowOffset = (static_cast<size_t>(batchSlot) * static_cast<size_t>(kMAX_SEQUENCE_LENGTH)
                                     + static_cast<size_t>(position))
            * rowBytes;
        CUDA_CHECK(
            cudaMemsetAsync(static_cast<uint8_t*>(mHidden->rawPointer()) + rowOffset, pattern, rowBytes, mStream));
    }

    void expectHiddenRow(int32_t batchSlot, int32_t position, uint8_t pattern)
    {
        size_t const rowBytes = hiddenRowBytes();
        size_t const rowOffset = (static_cast<size_t>(batchSlot) * static_cast<size_t>(kMAX_SEQUENCE_LENGTH)
                                     + static_cast<size_t>(position))
            * rowBytes;
        std::vector<uint8_t> row(rowBytes);
        ASSERT_EQ(cudaMemcpy(row.data(), static_cast<uint8_t const*>(mHidden->rawPointer()) + rowOffset, rowBytes,
                      cudaMemcpyDeviceToHost),
            cudaSuccess);
        EXPECT_TRUE(std::all_of(row.begin(), row.end(), [pattern](uint8_t value) { return value == pattern; }));
    }

    cudaStream_t mStream{};
    DeploymentConfig mDeployment;
    std::unique_ptr<HybridCacheManager> mBaseCache;
    std::unique_ptr<HybridCacheManager> mDraftCache;
    std::unique_ptr<KVPageTable> mBasePageTable;
    std::unique_ptr<KVPageTable> mDraftPageTable;
    std::unique_ptr<Tensor> mHidden;
    std::unique_ptr<ContextCacheCoordinator> mCoordinator;
};

// The kHybridMtp deployment must construct with a draft cache and boundary-hidden snapshot storage.
TEST_F(ContextCacheMtpCoordinatorTests, ConstructsHybridMtpDeployment)
{
    ContextCacheDeploymentProfile const profile = validateContextCacheDeployment(mDeployment);
    EXPECT_EQ(profile.baseStateKind, ContextCacheModelStateKind::kHybrid);
    EXPECT_TRUE(profile.isSpeculative());
    ASSERT_NE(mCoordinator, nullptr);
    EXPECT_EQ(mCoordinator->manager().records().size(), 0U);
}

// A cold MTP producer publishes an exact checkpoint via publishHybridMtpEndpoint; a longer consumer then rebinds the
// paired base+draft path and restores the recurrent state, base partial page, and boundary hidden row on the hit.
TEST_F(ContextCacheMtpCoordinatorTests, PublishEndpointBecomesReusableAndRestoresPairedState)
{
    constexpr int32_t kInputLength{kTOKENS_PER_PAGE + 1};
    constexpr int32_t kResidentLength{kInputLength}; // predecessor boundary chosen by the (future) runtime stage
    constexpr int32_t kBoundaryRow{kInputLength - 1};
    // fullBlockCount = (residentStateLength - 1) / P = 1, so the boundary token lands in the private partial page 1.
    constexpr int32_t kPartialTokens{kResidentLength - kTOKENS_PER_PAGE};

    auto producer = begin(makeTokens(kInputLength));
    ASSERT_EQ(mCoordinator->preparePrefill(producer.request), ContextCacheCoordinatorStatus::kOk);
    PageId const producerBaseBoundaryPage = mBasePageTable->hostRow(0)[1];
    seedRecurrent(0x31, 0x61);
    seedKvPage(*mBaseCache, producerBaseBoundaryPage, 0x21, 0x71);
    seedHiddenRow(0, kBoundaryRow, 0x51);
    ASSERT_EQ(cudaStreamSynchronize(mStream), cudaSuccess);

    ASSERT_EQ(mCoordinator->publishHybridMtpEndpoint(producer.request, 0, kResidentLength, *mHidden, kBoundaryRow),
        ContextCacheCoordinatorStatus::kOk);
    ASSERT_EQ(mCoordinator->manager().records().size(), 1U);

    // Scrub every live surface so a successful restore can only come from the published checkpoint.
    seedRecurrent(0, 0);
    seedKvPage(*mBaseCache, producerBaseBoundaryPage, 0, 0);
    seedHiddenRow(0, kBoundaryRow, 0);
    ASSERT_EQ(cudaStreamSynchronize(mStream), cudaSuccess);
    ASSERT_EQ(mCoordinator->finish(producer.request), ContextCacheCoordinatorStatus::kOk);

    ContextCacheMetrics const before = mCoordinator->metrics();
    auto consumer = begin(makeTokens(kInputLength + 1));
    ASSERT_EQ(consumer.prefillStarts, std::vector<int32_t>{kResidentLength});
    ContextCacheMetrics const after = mCoordinator->metrics();
    EXPECT_EQ(after.matchedTokens, before.matchedTokens + static_cast<uint64_t>(kResidentLength));

    ASSERT_EQ(mCoordinator->preparePrefill(consumer.request), ContextCacheCoordinatorStatus::kOk);
    // The runtime fold restores the boundary hidden row within its own prefill stream (no coordinator sync here).
    ASSERT_EQ(mCoordinator->restoreHybridMtpBoundaryHidden(consumer.request, 0, *mHidden, kBoundaryRow),
        ContextCacheCoordinatorStatus::kOk);
    ASSERT_EQ(cudaStreamSynchronize(mStream), cudaSuccess);

    expectRecurrent(0x31, 0x61);
    expectKvPrefix(*mBaseCache, mBasePageTable->hostRow(0)[1], kPartialTokens, 0x21, 0x71);
    expectHiddenRow(0, kBoundaryRow, 0x51);
    EXPECT_EQ(mCoordinator->metrics().hybridRestores, 1U);
    EXPECT_EQ(mCoordinator->finish(consumer.request), ContextCacheCoordinatorStatus::kOk);
}

TEST_F(ContextCacheMtpCoordinatorTests, VanillaRequestOnMtpDeploymentRestoresOnlyBaseHybridState)
{
    constexpr int32_t kInputLength{kTOKENS_PER_PAGE + 1};
    constexpr int32_t kResidentLength{kInputLength};
    constexpr int32_t kBoundaryRow{kInputLength - 1};

    auto producer = begin(makeTokens(kInputLength));
    ASSERT_EQ(mCoordinator->preparePrefill(producer.request), ContextCacheCoordinatorStatus::kOk);
    seedRecurrent(0x31, 0x61);
    seedHiddenRow(0, kBoundaryRow, 0x51);
    ASSERT_EQ(cudaStreamSynchronize(mStream), cudaSuccess);
    ASSERT_EQ(mCoordinator->publishHybridMtpEndpoint(producer.request, 0, kResidentLength, *mHidden, kBoundaryRow),
        ContextCacheCoordinatorStatus::kOk);
    ASSERT_EQ(mCoordinator->finish(producer.request), ContextCacheCoordinatorStatus::kOk);

    auto vanillaConsumer
        = begin(makeTokens(kInputLength + 1), ContextCacheLookupPolicy::kUseCache, /*speculativeRequest=*/false);
    ASSERT_EQ(vanillaConsumer.prefillStarts, std::vector<int32_t>{kResidentLength});
    EXPECT_EQ(mCoordinator->preparePrefill(vanillaConsumer.request), ContextCacheCoordinatorStatus::kOk);
    EXPECT_EQ(cudaStreamSynchronize(mStream), cudaSuccess);
    EXPECT_EQ(mCoordinator->finish(vanillaConsumer.request), ContextCacheCoordinatorStatus::kOk);
}

// The publish guard skips empty prefixes and bypass requests without creating a record.
TEST_F(ContextCacheMtpCoordinatorTests, PublishEndpointGuardsSkipEmptyAndBypass)
{
    constexpr int32_t kInputLength{kTOKENS_PER_PAGE + 1};

    auto producer = begin(makeTokens(kInputLength));
    ASSERT_EQ(mCoordinator->preparePrefill(producer.request), ContextCacheCoordinatorStatus::kOk);
    ASSERT_EQ(cudaStreamSynchronize(mStream), cudaSuccess);
    // Non-positive resident length is a no-op.
    ASSERT_EQ(mCoordinator->publishHybridMtpEndpoint(producer.request, 0, 0, *mHidden, 0),
        ContextCacheCoordinatorStatus::kOk);
    EXPECT_EQ(mCoordinator->manager().records().size(), 0U);
    ASSERT_EQ(mCoordinator->finish(producer.request), ContextCacheCoordinatorStatus::kOk);

    auto bypass = begin(makeTokens(kInputLength), ContextCacheLookupPolicy::kBypass);
    ASSERT_EQ(mCoordinator->preparePrefill(bypass.request), ContextCacheCoordinatorStatus::kOk);
    ASSERT_EQ(cudaStreamSynchronize(mStream), cudaSuccess);
    ASSERT_EQ(mCoordinator->publishHybridMtpEndpoint(bypass.request, 0, kInputLength, *mHidden, kInputLength - 1),
        ContextCacheCoordinatorStatus::kOk);
    EXPECT_EQ(mCoordinator->manager().records().size(), 0U);
    EXPECT_EQ(mCoordinator->finish(bypass.request), ContextCacheCoordinatorStatus::kOk);
}

// The draft engine reaches its KV only through the draft page table, so every page the coordinator leases -- the record
// pages a hit rebinds and the private boundary page the paired restore writes -- must be named by that table before any
// draft-side work runs. The table is identity-mapped until uploaded, and a consumer's pages are not identity: this is
// the invariant a content restore alone does not establish.
TEST_F(ContextCacheMtpCoordinatorTests, PrefillPublishesTheLeasedDraftPagePathToTheDraftPageTable)
{
    constexpr int32_t kInputLength{kTOKENS_PER_PAGE + 1};
    constexpr int32_t kResidentLength{kInputLength};
    constexpr int32_t kBoundaryRow{kInputLength - 1};
    constexpr int32_t kPartialTokens{kResidentLength - kTOKENS_PER_PAGE};

    auto producer = begin(makeTokens(kInputLength));
    ASSERT_EQ(mCoordinator->preparePrefill(producer.request), ContextCacheCoordinatorStatus::kOk);
    PageId const producerDraftRecordPage = mDraftPageTable->hostRow(0)[0];
    PageId const producerDraftBoundaryPage = mDraftPageTable->hostRow(0)[1];
    seedKvPage(*mDraftCache, producerDraftRecordPage, 0x13, 0x43);
    seedKvPage(*mDraftCache, producerDraftBoundaryPage, 0x24, 0x54);
    seedHiddenRow(0, kBoundaryRow, 0x51);
    ASSERT_EQ(cudaStreamSynchronize(mStream), cudaSuccess);

    ASSERT_EQ(mCoordinator->publishHybridMtpEndpoint(producer.request, 0, kResidentLength, *mHidden, kBoundaryRow),
        ContextCacheCoordinatorStatus::kOk);
    // The boundary page is private and goes back to the pool at finish; only the snapshot can bring its content back.
    seedKvPage(*mDraftCache, producerDraftBoundaryPage, 0, 0);
    ASSERT_EQ(cudaStreamSynchronize(mStream), cudaSuccess);
    ASSERT_EQ(mCoordinator->finish(producer.request), ContextCacheCoordinatorStatus::kOk);

    auto consumer = begin(makeTokens(kInputLength + 1));
    ASSERT_EQ(consumer.prefillStarts, std::vector<int32_t>{kResidentLength});
    ASSERT_EQ(mCoordinator->preparePrefill(consumer.request), ContextCacheCoordinatorStatus::kOk);
    ASSERT_EQ(cudaStreamSynchronize(mStream), cudaSuccess);

    // The private page is drawn from further along the pool's free ring than the released one, so the row diverges from
    // the identity mapping the table would still hold had the prefill skipped the upload.
    PageId const consumerDraftBoundaryPage = mDraftPageTable->hostRow(0)[1];
    EXPECT_NE(consumerDraftBoundaryPage, 1);
    EXPECT_EQ(mDraftPageTable->hostRow(0)[0], producerDraftRecordPage);
    expectKvPrefix(*mDraftCache, mDraftPageTable->hostRow(0)[0], kTOKENS_PER_PAGE, 0x13, 0x43);
    expectKvPrefix(*mDraftCache, consumerDraftBoundaryPage, kPartialTokens, 0x24, 0x54);
    EXPECT_EQ(mCoordinator->finish(consumer.request), ContextCacheCoordinatorStatus::kOk);
}

// Snapshot pool sizing must charge a slot for every slab the storage allocates against it, or the pools overrun the
// caller's declared device-memory ceiling by the paired draft partial-KV page and the boundary hidden row.
TEST_F(ContextCacheMtpCoordinatorTests, SnapshotPoolSizingChargesThePairedDraftAndBoundaryHiddenSlabs)
{
    constexpr int32_t kSlots{4};
    size_t const baseOnlyRecurrent
        = HybridSnapshotStorage::recurrentBytesPerSlot(mBaseCache->getMambaCacheManager().getConfig());
    size_t const baseOnlyPartialKv
        = HybridSnapshotStorage::partialKvBytesPerSlot(mBaseCache->getKVCacheManager().getConfig());
    ASSERT_GT(recurrentBytesPerSlot(), baseOnlyRecurrent);
    ASSERT_GT(partialKvBytesPerSlot(), baseOnlyPartialKv);

    createCoordinatorWithBudgets(static_cast<int64_t>(static_cast<size_t>(kSlots) * baseOnlyRecurrent),
        static_cast<int64_t>(static_cast<size_t>(kSlots) * baseOnlyPartialKv));
    ContextCacheMetrics const underpriced = mCoordinator->metrics();
    EXPECT_LT(underpriced.recurrentSnapshots.capacity, kSlots);
    EXPECT_LT(underpriced.partialKvSnapshots.capacity, kSlots);

    createCoordinator({}, kSlots);
    ContextCacheMetrics const priced = mCoordinator->metrics();
    EXPECT_EQ(priced.recurrentSnapshots.capacity, kSlots);
    EXPECT_EQ(priced.partialKvSnapshots.capacity, kSlots);
}

} // namespace
