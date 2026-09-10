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

#include "runtime/state/contextCache/hybridSnapshotStorage.h"

#include "common/cudaMacros.h"
#include "common/pagedKvTypes.h"

#include <cuda_fp16.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

using namespace nvinfer1;
using namespace trt_edgellm::rt;

namespace
{

HybridCacheManager::Config makeHybridConfig()
{
    HybridCacheManager::Config config;
    config.layerTypes = {HybridCacheManager::LayerType::kAttention, HybridCacheManager::LayerType::kMamba,
        HybridCacheManager::LayerType::kAttention, HybridCacheManager::LayerType::kMamba};
    config.kvConfig = KVCacheManager::Config{/*.numAttentionLayers=*/2, /*.maxBatchSize=*/2,
        /*.maxSequenceLength=*/256, /*.layerConfigs=*/{KVLayerConfig{2, 4}, KVLayerConfig{1, 8}},
        /*.kvCacheType=*/DataType::kHALF, /*.numPages=*/8};
    config.mambaConfig = MambaCacheManager::Config{/*.numRecurrentLayers=*/2, /*.maxBatchSize=*/2,
        /*.recurrentStateNumHeads=*/2, /*.recurrentStateHeadDim=*/3, /*.recurrentStateSize=*/4,
        /*.convDim=*/5, /*.convKernel=*/6, /*.maxIntermediateSeqLen=*/0,
        /*.recurrentStateType=*/DataType::kHALF, /*.convStateType=*/DataType::kHALF};
    config.maxBatchSize = 2;
    return config;
}

} // namespace

TEST(HybridSnapshotStorageTests, ReportsExactBytesPerSlot)
{
    HybridCacheManager::Config const config = makeHybridConfig();
    EXPECT_EQ(HybridSnapshotStorage::recurrentBytesPerSlot(config.mambaConfig), 216U);
    EXPECT_EQ(HybridSnapshotStorage::partialKvBytesPerSlot(config.kvConfig), 8192U);
}

TEST(HybridSnapshotStorageTests, RejectsLiveRecurrentShapeDrift)
{
    cudaStream_t stream{};
    HybridCacheManager::Config const config = makeHybridConfig();
    HybridCacheManager cacheManager(config, stream);
    Tensor& recurrent = cacheManager.getMambaCacheManager().getRecurrentState(0);
    ASSERT_TRUE(recurrent.reshape({config.mambaConfig.maxBatchSize, config.mambaConfig.recurrentStateNumHeads,
        config.mambaConfig.recurrentStateHeadDim, config.mambaConfig.recurrentStateSize - 1}));

    EXPECT_THROW((void) HybridSnapshotStorage(cacheManager, 2, 2), std::runtime_error);
}

TEST(HybridSnapshotStorageTests, RejectsLiveKvPageGeometryDrift)
{
    cudaStream_t stream{};
    HybridCacheManager::Config const config = makeHybridConfig();
    HybridCacheManager cacheManager(config, stream);
    KVCacheManager& kv = cacheManager.getKVCacheManager();
    KVLayerConfig const& layerConfig = kv.getLayerConfig(0);
    Tensor& pool = kv.getCombinedKVCache(0);
    ASSERT_TRUE(pool.reshape({2, kv.numPages(), kTOKENS_PER_PAGE / 2, layerConfig.numKVHeads, layerConfig.headDim}));

    EXPECT_THROW((void) HybridSnapshotStorage(cacheManager, 2, 2), std::runtime_error);
}

TEST(HybridSnapshotStorageTests, CaptureAndRestoreAreByteExactWithExtraRetainedPages)
{
    cudaStream_t stream{};
    HybridCacheManager::Config const config = makeHybridConfig();
    HybridCacheManager cacheManager(config, stream);
    HybridSnapshotStorage storage(cacheManager, 2, 2);

    int32_t constexpr kBatchSlot{1};
    int32_t constexpr kSnapshotSlot{1};
    MambaCacheManager& mamba = cacheManager.getMambaCacheManager();
    size_t const recurrentBytes = static_cast<size_t>(2 * 3 * 4) * sizeof(half);
    size_t const convBytes = static_cast<size_t>(5 * 6) * sizeof(half);
    for (int32_t layer = 0; layer < mamba.numLayers(); ++layer)
    {
        uint8_t const recurrentPattern = static_cast<uint8_t>(0x31 + layer);
        uint8_t const convPattern = static_cast<uint8_t>(0x61 + layer);
        auto* recurrent = static_cast<uint8_t*>(mamba.getRecurrentState(layer).rawPointer());
        auto* conv = static_cast<uint8_t*>(mamba.getConvState(layer).rawPointer());
        CUDA_CHECK(cudaMemsetAsync(recurrent + kBatchSlot * recurrentBytes, recurrentPattern, recurrentBytes, stream));
        CUDA_CHECK(cudaMemsetAsync(conv + kBatchSlot * convBytes, convPattern, convBytes, stream));
    }

    storage.captureRecurrent(kSnapshotSlot, kBatchSlot, stream);
    storage.zeroRecurrent(kBatchSlot, stream);
    storage.restoreRecurrent(kSnapshotSlot, kBatchSlot, stream);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

    for (int32_t layer = 0; layer < mamba.numLayers(); ++layer)
    {
        std::vector<uint8_t> recurrentHost(recurrentBytes);
        std::vector<uint8_t> convHost(convBytes);
        auto const* recurrent = static_cast<uint8_t const*>(mamba.getRecurrentState(layer).rawPointer());
        auto const* conv = static_cast<uint8_t const*>(mamba.getConvState(layer).rawPointer());
        ASSERT_EQ(cudaMemcpy(recurrentHost.data(), recurrent + kBatchSlot * recurrentBytes, recurrentBytes,
                      cudaMemcpyDeviceToHost),
            cudaSuccess);
        ASSERT_EQ(
            cudaMemcpy(convHost.data(), conv + kBatchSlot * convBytes, convBytes, cudaMemcpyDeviceToHost), cudaSuccess);
        EXPECT_TRUE(std::all_of(recurrentHost.begin(), recurrentHost.end(),
            [layer](uint8_t value) { return value == static_cast<uint8_t>(0x31 + layer); }));
        EXPECT_TRUE(std::all_of(convHost.begin(), convHost.end(),
            [layer](uint8_t value) { return value == static_cast<uint8_t>(0x61 + layer); }));
    }

    int32_t constexpr kSourcePage{7};
    int32_t constexpr kDestinationPage{0};
    int32_t constexpr kValidTokens{17};
    KVCacheManager& kv = cacheManager.getKVCacheManager();
    for (int32_t layer = 0; layer < kv.numLayers(); ++layer)
    {
        KVLayerConfig const& layerConfig = kv.getLayerConfig(layer);
        size_t const tokenBytes = static_cast<size_t>(layerConfig.numKVHeads * layerConfig.headDim) * sizeof(half);
        size_t const pageBytes = tokenBytes * static_cast<size_t>(kTOKENS_PER_PAGE);
        auto* keys = static_cast<uint8_t*>(kv.kPoolPtr(layer));
        auto* values = static_cast<uint8_t*>(kv.vPoolPtr(layer));
        CUDA_CHECK(
            cudaMemsetAsync(keys + kSourcePage * pageBytes, static_cast<uint8_t>(0x21 + layer), pageBytes, stream));
        CUDA_CHECK(
            cudaMemsetAsync(values + kSourcePage * pageBytes, static_cast<uint8_t>(0x71 + layer), pageBytes, stream));
        CUDA_CHECK(cudaMemsetAsync(keys + kDestinationPage * pageBytes, 0, pageBytes, stream));
        CUDA_CHECK(cudaMemsetAsync(values + kDestinationPage * pageBytes, 0, pageBytes, stream));
    }

    storage.capturePartialKv(kSnapshotSlot, kSourcePage, kValidTokens, stream);
    for (int32_t layer = 0; layer < kv.numLayers(); ++layer)
    {
        KVLayerConfig const& layerConfig = kv.getLayerConfig(layer);
        size_t const pageBytes = static_cast<size_t>(layerConfig.numKVHeads * layerConfig.headDim) * sizeof(half)
            * static_cast<size_t>(kTOKENS_PER_PAGE);
        CUDA_CHECK(
            cudaMemsetAsync(static_cast<uint8_t*>(kv.kPoolPtr(layer)) + kSourcePage * pageBytes, 0, pageBytes, stream));
        CUDA_CHECK(
            cudaMemsetAsync(static_cast<uint8_t*>(kv.vPoolPtr(layer)) + kSourcePage * pageBytes, 0, pageBytes, stream));
    }
    storage.restorePartialKv(kSnapshotSlot, kDestinationPage, kValidTokens, stream);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

    for (int32_t layer = 0; layer < kv.numLayers(); ++layer)
    {
        KVLayerConfig const& layerConfig = kv.getLayerConfig(layer);
        size_t const tokenBytes = static_cast<size_t>(layerConfig.numKVHeads * layerConfig.headDim) * sizeof(half);
        size_t const pageBytes = tokenBytes * static_cast<size_t>(kTOKENS_PER_PAGE);
        size_t const copiedBytes = tokenBytes * static_cast<size_t>(kValidTokens);
        std::vector<uint8_t> keyHost(pageBytes);
        std::vector<uint8_t> valueHost(pageBytes);
        ASSERT_EQ(
            cudaMemcpy(keyHost.data(), static_cast<uint8_t const*>(kv.kPoolPtr(layer)) + kDestinationPage * pageBytes,
                pageBytes, cudaMemcpyDeviceToHost),
            cudaSuccess);
        ASSERT_EQ(
            cudaMemcpy(valueHost.data(), static_cast<uint8_t const*>(kv.vPoolPtr(layer)) + kDestinationPage * pageBytes,
                pageBytes, cudaMemcpyDeviceToHost),
            cudaSuccess);
        for (size_t byte = 0; byte < pageBytes; ++byte)
        {
            EXPECT_EQ(keyHost[byte], byte < copiedBytes ? static_cast<uint8_t>(0x21 + layer) : 0U);
            EXPECT_EQ(valueHost[byte], byte < copiedBytes ? static_cast<uint8_t>(0x71 + layer) : 0U);
        }
    }
}

TEST(HybridSnapshotStorageTests, BoundaryHiddenCaptureAndRestoreIsByteExact)
{
    cudaStream_t stream{};
    HybridCacheManager::Config const config = makeHybridConfig();
    HybridCacheManager cacheManager(config, stream);
    int32_t constexpr kBoundaryHiddenDim{8};
    int32_t constexpr kBatch{2};
    int32_t constexpr kSeq{3};
    HybridSnapshotStorage storage(
        cacheManager, 2, 2, /*draftCacheManager=*/nullptr, kBoundaryHiddenDim, DataType::kHALF);
    ASSERT_EQ(storage.boundaryHiddenDim(), kBoundaryHiddenDim);

    size_t const rowBytes = static_cast<size_t>(kBoundaryHiddenDim) * sizeof(half);
    Tensor source(
        {kBatch, kSeq, kBoundaryHiddenDim}, trt_edgellm::rt::DeviceType::kGPU, DataType::kHALF, "boundaryHiddenSource");
    Tensor destination(
        {kBatch, kSeq, kBoundaryHiddenDim}, trt_edgellm::rt::DeviceType::kGPU, DataType::kHALF, "boundaryHiddenDest");

    int32_t constexpr kSourceBatch{1};
    int32_t constexpr kSourcePosition{2};
    int32_t constexpr kSnapshotSlot{1};
    int32_t constexpr kDestBatch{0};
    int32_t constexpr kDestPosition{1};
    size_t const sourceRow = (static_cast<size_t>(kSourceBatch) * kSeq + kSourcePosition) * rowBytes;
    size_t const destRow = (static_cast<size_t>(kDestBatch) * kSeq + kDestPosition) * rowBytes;

    CUDA_CHECK(cudaMemsetAsync(source.rawPointer(), 0, static_cast<size_t>(kBatch * kSeq) * rowBytes, stream));
    CUDA_CHECK(cudaMemsetAsync(static_cast<uint8_t*>(source.rawPointer()) + sourceRow, 0x5A, rowBytes, stream));
    CUDA_CHECK(cudaMemsetAsync(destination.rawPointer(), 0, static_cast<size_t>(kBatch * kSeq) * rowBytes, stream));

    storage.captureBoundaryHidden(kSnapshotSlot, source, kSourceBatch, kSourcePosition, stream);
    CUDA_CHECK(cudaMemsetAsync(source.rawPointer(), 0, static_cast<size_t>(kBatch * kSeq) * rowBytes, stream));
    storage.restoreBoundaryHidden(kSnapshotSlot, destination, kDestBatch, kDestPosition, stream);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

    std::vector<uint8_t> destHost(static_cast<size_t>(kBatch * kSeq) * rowBytes);
    ASSERT_EQ(
        cudaMemcpy(destHost.data(), destination.rawPointer(), destHost.size(), cudaMemcpyDeviceToHost), cudaSuccess);
    for (size_t byte = 0; byte < destHost.size(); ++byte)
    {
        bool const inRestoredRow = byte >= destRow && byte < destRow + rowBytes;
        EXPECT_EQ(destHost[byte], inRestoredRow ? 0x5AU : 0U);
    }
}

TEST(HybridSnapshotStorageTests, DisabledBoundaryAndDraftMethodsThrow)
{
    cudaStream_t stream{};
    HybridCacheManager::Config const config = makeHybridConfig();
    HybridCacheManager cacheManager(config, stream);
    HybridSnapshotStorage storage(cacheManager, 2, 2);
    EXPECT_EQ(storage.boundaryHiddenDim(), 0);

    Tensor hidden({2, 3, 8}, trt_edgellm::rt::DeviceType::kGPU, DataType::kHALF, "hidden");
    EXPECT_THROW(storage.captureBoundaryHidden(0, hidden, 0, 0, stream), std::runtime_error);
    EXPECT_THROW(storage.restoreBoundaryHidden(0, hidden, 0, 0, stream), std::runtime_error);
    EXPECT_THROW(storage.capturePartialKv(0, /*base=*/0, /*draft=*/0, 4, stream), std::runtime_error);
    EXPECT_THROW(storage.restorePartialKv(0, /*base=*/0, /*draft=*/0, 4, stream), std::runtime_error);
}

TEST(HybridSnapshotStorageTests, PairedBaseAndDraftPartialKvRoundTrip)
{
    cudaStream_t stream{};
    HybridCacheManager::Config const config = makeHybridConfig();
    HybridCacheManager baseCache(config, stream);
    HybridCacheManager draftCache(config, stream);
    HybridSnapshotStorage storage(baseCache, 2, 2, &draftCache);

    int32_t constexpr kSnapshotSlot{1};
    int32_t constexpr kSourcePage{7};
    int32_t constexpr kDestinationPage{0};
    int32_t constexpr kValidTokens{17};

    auto fillPools = [&](HybridCacheManager& cache, uint8_t keyBase, uint8_t valueBase) {
        KVCacheManager& kv = cache.getKVCacheManager();
        for (int32_t layer = 0; layer < kv.numLayers(); ++layer)
        {
            KVLayerConfig const& layerConfig = kv.getLayerConfig(layer);
            size_t const pageBytes = static_cast<size_t>(layerConfig.numKVHeads * layerConfig.headDim) * sizeof(half)
                * static_cast<size_t>(kTOKENS_PER_PAGE);
            auto* keys = static_cast<uint8_t*>(kv.kPoolPtr(layer));
            auto* values = static_cast<uint8_t*>(kv.vPoolPtr(layer));
            CUDA_CHECK(cudaMemsetAsync(
                keys + kSourcePage * pageBytes, static_cast<uint8_t>(keyBase + layer), pageBytes, stream));
            CUDA_CHECK(cudaMemsetAsync(
                values + kSourcePage * pageBytes, static_cast<uint8_t>(valueBase + layer), pageBytes, stream));
            CUDA_CHECK(cudaMemsetAsync(keys + kDestinationPage * pageBytes, 0, pageBytes, stream));
            CUDA_CHECK(cudaMemsetAsync(values + kDestinationPage * pageBytes, 0, pageBytes, stream));
        }
    };
    fillPools(baseCache, 0x21, 0x71);
    fillPools(draftCache, 0x31, 0x81);

    storage.capturePartialKv(kSnapshotSlot, kSourcePage, kSourcePage, kValidTokens, stream);
    auto zeroSource = [&](HybridCacheManager& cache) {
        KVCacheManager& kv = cache.getKVCacheManager();
        for (int32_t layer = 0; layer < kv.numLayers(); ++layer)
        {
            KVLayerConfig const& layerConfig = kv.getLayerConfig(layer);
            size_t const pageBytes = static_cast<size_t>(layerConfig.numKVHeads * layerConfig.headDim) * sizeof(half)
                * static_cast<size_t>(kTOKENS_PER_PAGE);
            CUDA_CHECK(cudaMemsetAsync(
                static_cast<uint8_t*>(kv.kPoolPtr(layer)) + kSourcePage * pageBytes, 0, pageBytes, stream));
            CUDA_CHECK(cudaMemsetAsync(
                static_cast<uint8_t*>(kv.vPoolPtr(layer)) + kSourcePage * pageBytes, 0, pageBytes, stream));
        }
    };
    zeroSource(baseCache);
    zeroSource(draftCache);
    storage.restorePartialKv(kSnapshotSlot, kDestinationPage, kDestinationPage, kValidTokens, stream);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

    auto verify = [&](HybridCacheManager& cache, uint8_t keyBase, uint8_t valueBase) {
        KVCacheManager& kv = cache.getKVCacheManager();
        for (int32_t layer = 0; layer < kv.numLayers(); ++layer)
        {
            KVLayerConfig const& layerConfig = kv.getLayerConfig(layer);
            size_t const tokenBytes = static_cast<size_t>(layerConfig.numKVHeads * layerConfig.headDim) * sizeof(half);
            size_t const pageBytes = tokenBytes * static_cast<size_t>(kTOKENS_PER_PAGE);
            size_t const copiedBytes = tokenBytes * static_cast<size_t>(kValidTokens);
            std::vector<uint8_t> keyHost(pageBytes);
            std::vector<uint8_t> valueHost(pageBytes);
            ASSERT_EQ(cudaMemcpy(keyHost.data(),
                          static_cast<uint8_t const*>(kv.kPoolPtr(layer)) + kDestinationPage * pageBytes, pageBytes,
                          cudaMemcpyDeviceToHost),
                cudaSuccess);
            ASSERT_EQ(cudaMemcpy(valueHost.data(),
                          static_cast<uint8_t const*>(kv.vPoolPtr(layer)) + kDestinationPage * pageBytes, pageBytes,
                          cudaMemcpyDeviceToHost),
                cudaSuccess);
            for (size_t byte = 0; byte < pageBytes; ++byte)
            {
                EXPECT_EQ(keyHost[byte], byte < copiedBytes ? static_cast<uint8_t>(keyBase + layer) : 0U);
                EXPECT_EQ(valueHost[byte], byte < copiedBytes ? static_cast<uint8_t>(valueBase + layer) : 0U);
            }
        }
    };
    verify(baseCache, 0x21, 0x71);
    verify(draftCache, 0x31, 0x81);
}
