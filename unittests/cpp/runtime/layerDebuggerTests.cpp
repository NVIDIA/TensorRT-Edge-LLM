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

#include "common/cudaUtils.h"
#include "common/safetensorsUtils.h"
#include "runtime/debug/layerDebugger.h"
#include "runtime/state/kvPageTable.h"
#include "runtime/state/residentSlotPool.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <cuda_fp16.h>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

using namespace trt_edgellm;

namespace
{

//! Teacher forcing is addressed by *original* request row, not by active slot, because the
//! runtime compacts its per-slot vectors when a sequence finishes. These tests drive the
//! host-side entry points directly: they need neither a GPU nor an engine.
class LayerDebuggerForcedRows : public ::testing::Test
{
protected:
    void SetUp() override
    {
        mDir = std::filesystem::temp_directory_path() / "edgellm_layer_debugger_test";
        std::filesystem::create_directories(mDir);
        std::filesystem::path const tokensPath = mDir / "forced_tokens.txt";
        // Row r generates tokens 10*r + step, so a row mix-up is unmistakable.
        std::ofstream(tokensPath) << "0 1 2\n10 11 12\n20 21 22\n";

        setenv("EDGELLM_DUMP_LOGITS_KVCACHE_LAYERS", "1", 1);
        setenv("EDGELLM_DUMP_LOGITS_KVCACHE_DIR", mDir.c_str(), 1);
        setenv("EDGELLM_FORCE_TOKENS_FILE", tokensPath.c_str(), 1);
    }

    void TearDown() override
    {
        unsetenv("EDGELLM_DUMP_LOGITS_KVCACHE_LAYERS");
        unsetenv("EDGELLM_DUMP_LOGITS_KVCACHE_DIR");
        unsetenv("EDGELLM_FORCE_TOKENS_FILE");
        std::filesystem::remove_all(mDir);
    }

    std::filesystem::path mDir;
};

// One request's whole lifecycle, because a debugger claims its block of force-token rows on
// first use from a process-wide counter: a second instance would start at a different base.
//
// Slot 0 finishes mid-request, the runtime compacts the batch, and both the vanilla and the
// speculative entry points must keep addressing each survivor by the row it came from.
TEST_F(LayerDebuggerForcedRows, SurvivesBatchCompaction)
{
    auto debugger = rt::LayerDebugger::fromEnv();
    ASSERT_NE(debugger, nullptr);

    // Prefill: three active slots, still one-to-one with the request rows.
    std::vector<int32_t> const genLengths3{0, 0, 0};
    std::vector<int32_t> const identity{0, 1, 2};
    std::vector<int32_t> tokens{-1, -1, -1};
    debugger->applyForcedTokens(genLengths3, identity, tokens.data(), 3);
    EXPECT_EQ(tokens[0], 0);
    EXPECT_EQ(tokens[1], 10);
    EXPECT_EQ(tokens[2], 20);

    // Slot 0 finishes. Rows 1 and 2 move down to slots 0 and 1, and batchIndexMapping is what
    // records where they came from.
    std::vector<int32_t> const genLengths2{1, 1};
    std::vector<int32_t> const compacted{1, 2};
    std::vector<int32_t> survivors{-1, -1};
    debugger->applyForcedTokens(genLengths2, compacted, survivors.data(), 2);
    EXPECT_EQ(survivors[0], 11) << "survivor was fed the evicted sequence's golden tokens";
    EXPECT_EQ(survivors[1], 21);

    // The speculative path addresses rows the same way, and trims the acceptance at the first
    // token that disagrees rather than overwriting one in place. Slot 1 still carries row 2,
    // whose remaining golden tokens are 21 then 22; the second proposal disagrees.
    constexpr int32_t kMaxAcceptDepth = 3;
    std::vector<int32_t> const genLengthsSpec{1, 1};
    std::vector<int32_t> acceptedTokenIds{10, 11, 0, 21, 99, 0};
    std::vector<int32_t> acceptLengths{1, 3};
    std::vector<int32_t> ownTokens;

    bool const trimmed = debugger->applyForcedAcceptance(
        genLengthsSpec, compacted, acceptLengths.data(), acceptedTokenIds.data(), ownTokens, 2, kMaxAcceptDepth);

    EXPECT_TRUE(trimmed);
    EXPECT_EQ(acceptLengths[1], 2) << "acceptance must stop at the divergence so the replaced "
                                      "token's cache entry is left uncommitted";
    EXPECT_EQ(acceptedTokenIds[3], 21) << "the matching prefix is untouched";
    EXPECT_EQ(acceptedTokenIds[4], 22) << "the divergent token is replaced by the golden's";
    ASSERT_EQ(ownTokens.size(), 2u);
    EXPECT_EQ(ownTokens[1], 99) << "the engine's own choice is kept as the divergence signal";
}

TEST_F(LayerDebuggerForcedRows, DumpsNonContiguousResidentRowsInExecutionOrder)
{
    setenv("EDGELLM_DUMP_LOGITS_KVCACHE_LAYERS", "2", 1);
    cudaStream_t stream{};
    ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);

    constexpr int32_t maxSlots = 3;
    rt::HybridCacheManager::Config config{};
    config.layerTypes = {rt::HybridCacheManager::LayerType::kMamba, rt::HybridCacheManager::LayerType::kAttention};
    config.kvConfig
        = rt::KVCacheManager::Config{1, maxSlots, 32, {rt::KVLayerConfig{1, 64}}, nvinfer1::DataType::kHALF};
    config.mambaConfig.numRecurrentLayers = 1;
    config.mambaConfig.maxBatchSize = maxSlots;
    config.mambaConfig.recurrentStateNumHeads = 1;
    config.mambaConfig.recurrentStateHeadDim = 1;
    config.mambaConfig.recurrentStateSize = 2;
    config.mambaConfig.convDim = 2;
    config.mambaConfig.convKernel = 2;
    config.maxBatchSize = maxSlots;
    rt::HybridCacheManager cacheManager(config, stream);
    rt::KVPageTable pageTable(maxSlots, /*maxPagesPerSeq=*/1, /*numPages=*/maxSlots);
    pageTable.setIdentity();

    auto fillDevice = [&](rt::Tensor& destination, std::vector<half> const& values) {
        rt::Tensor pinned({static_cast<int64_t>(values.size())}, rt::DeviceType::kCPU, nvinfer1::DataType::kHALF,
            "layerDebuggerTest.pinned");
        std::memcpy(pinned.rawPointer(), values.data(), values.size() * sizeof(half));
        CUDA_CHECK(cudaMemcpyAsync(destination.rawPointer(), pinned.rawPointer(), values.size() * sizeof(half),
            cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    };

    fillDevice(cacheManager.getRecurrentState(0),
        {__float2half(1), __float2half(1), __float2half(2), __float2half(2), __float2half(3), __float2half(3)});
    fillDevice(cacheManager.getConvState(0),
        {__float2half(11), __float2half(11), __float2half(11), __float2half(11), __float2half(22), __float2half(22),
            __float2half(22), __float2half(22), __float2half(33), __float2half(33), __float2half(33),
            __float2half(33)});

    auto [kView, vView] = cacheManager.getSeparateKVCache(1);
    size_t const rowElements = static_cast<size_t>(kView.getShape()[1] * kView.getShape()[2] * kView.getShape()[3]);
    std::vector<half> kValues(static_cast<size_t>(maxSlots) * rowElements);
    std::vector<half> vValues(static_cast<size_t>(maxSlots) * rowElements);
    for (int32_t slot = 0; slot < maxSlots; ++slot)
    {
        std::fill_n(kValues.begin() + slot * rowElements, rowElements, __float2half(slot + 1));
        std::fill_n(vValues.begin() + slot * rowElements, rowElements, __float2half(slot + 11));
    }
    fillDevice(kView, kValues);
    fillDevice(vView, vValues);

    rt::Tensor logits({2, 4}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT, "logits");
    CUDA_CHECK(cudaMemsetAsync(logits.rawPointer(), 0, logits.getMemoryCapacity(), stream));
    auto debugger = rt::LayerDebugger::fromEnv();
    ASSERT_NE(debugger, nullptr);
    std::vector<rt::ResidentRef> const residents{{0, 1}, {2, 1}};
    debugger->dumpRound(
        cacheManager, pageTable, /*swaPageTable=*/nullptr, logits, {1, 1}, {0, 2}, residents, nullptr, 2, stream);
    debugger->flush(stream);

    std::filesystem::path dump;
    for (auto const& entry : std::filesystem::directory_iterator(mDir))
    {
        if (entry.path().extension() == ".safetensors")
        {
            dump = entry.path();
        }
    }
    ASSERT_FALSE(dump.empty());
    std::vector<rt::Tensor> tensors;
    ASSERT_TRUE(rt::safetensors::loadSafetensors(dump, tensors, stream));
    auto findTensor = [&](std::string const& name) -> rt::Tensor* {
        auto const found = std::find_if(tensors.begin(), tensors.end(),
            [&](rt::Tensor const& tensor) { return tensor.getName().find(name) != std::string::npos; });
        return found == tensors.end() ? nullptr : &*found;
    };
    auto readHalf = [&](rt::Tensor const& tensor) {
        std::vector<half> host(static_cast<size_t>(tensor.getShape().volume()));
        rt::Tensor pinned({static_cast<int64_t>(host.size())}, rt::DeviceType::kCPU, nvinfer1::DataType::kHALF,
            "layerDebuggerTest.readback");
        CUDA_CHECK(cudaMemcpyAsync(
            pinned.rawPointer(), tensor.rawPointer(), host.size() * sizeof(half), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        std::memcpy(host.data(), pinned.rawPointer(), host.size() * sizeof(half));
        return host;
    };

    rt::Tensor* recurrent = findTensor("layer_0.recurrent_state");
    ASSERT_NE(recurrent, nullptr);
    auto const recurrentRows = readHalf(*recurrent);
    EXPECT_EQ(__half2float(recurrentRows.front()), 1.0F);
    EXPECT_EQ(__half2float(recurrentRows[2]), 3.0F);

    rt::Tensor* kv = findTensor("layer_1.kv");
    ASSERT_NE(kv, nullptr);
    auto const kvRows = readHalf(*kv);
    size_t const dumpedHalfElements = rowElements;
    size_t const dumpedBatchElements = 2 * dumpedHalfElements;
    EXPECT_EQ(__half2float(kvRows.front()), 1.0F);
    EXPECT_EQ(__half2float(kvRows[dumpedHalfElements]), 11.0F);
    EXPECT_EQ(__half2float(kvRows[dumpedBatchElements]), 3.0F);
    EXPECT_EQ(__half2float(kvRows[dumpedBatchElements + dumpedHalfElements]), 13.0F);

    ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

TEST_F(LayerDebuggerForcedRows, DumpsFullAndReducedKvLayersThroughTheirOwnNonIdentityPageTables)
{
    setenv("EDGELLM_DUMP_LOGITS_KVCACHE_LAYERS", "2", 1);
    cudaStream_t stream{};
    ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);

    constexpr int32_t maxSlots = 3;
    constexpr int32_t maxSequenceLength = 2 * rt::kTOKENS_PER_PAGE;
    constexpr int32_t fullPages = maxSlots * 2;
    int32_t const swaPages = static_cast<int32_t>(rt::computeMinimumSwaPoolPages(maxSlots, rt::kTOKENS_PER_PAGE));

    rt::HybridCacheManager::Config config{};
    config.layerTypes = {rt::HybridCacheManager::LayerType::kAttention, rt::HybridCacheManager::LayerType::kAttention};
    config.kvConfig = rt::KVCacheManager::Config{/*.numAttentionLayers=*/2,
        /*.maxBatchSize=*/maxSlots,
        /*.maxSequenceLength=*/maxSequenceLength,
        /*.layerConfigs=*/{rt::KVLayerConfig{1, 64}, rt::KVLayerConfig{1, 64, rt::kTOKENS_PER_PAGE}},
        /*.kvCacheType=*/nvinfer1::DataType::kHALF,
        /*.numPages=*/fullPages,
        /*.numSwaPages=*/swaPages,
        /*.useBoundedSwaKVCache=*/true};
    config.mambaConfig.numRecurrentLayers = 0;
    config.mambaConfig.maxBatchSize = maxSlots;
    config.maxBatchSize = maxSlots;
    rt::HybridCacheManager cacheManager(config, stream);

    rt::KVPageTable fullTable(maxSlots, /*maxPagesPerSeq=*/2, fullPages);
    std::vector<int32_t> const fullSlot0{5};
    std::vector<int32_t> const fullSlot2{4};
    fullTable.setRow(0, fullSlot0.data(), static_cast<int32_t>(fullSlot0.size()));
    fullTable.setRow(2, fullSlot2.data(), static_cast<int32_t>(fullSlot2.size()));

    rt::KVPageTable swaTable(maxSlots, /*maxPagesPerSeq=*/2, swaPages, rt::KVPageTable::Mode::kSparseWindow);
    swaTable.setEntry(/*slot=*/0, /*logicalPage=*/0, /*kPageId=*/2);
    swaTable.setEntry(/*slot=*/2, /*logicalPage=*/1, /*kPageId=*/7);

    auto fillPoolByPage = [&](int32_t layer, float base) {
        rt::Tensor& pool = cacheManager.getCombinedKVCache(layer);
        auto const& shape = pool.getShape();
        int64_t const pageElements = shape[2] * shape[3] * shape[4];
        int32_t const pages = static_cast<int32_t>(shape[1]);
        std::vector<half> values(static_cast<size_t>(shape.volume()));
        for (int32_t kv = 0; kv < 2; ++kv)
        {
            for (int32_t page = 0; page < pages; ++page)
            {
                std::fill_n(values.begin() + (static_cast<int64_t>(kv) * pages + page) * pageElements, pageElements,
                    __float2half(base + kv * 100 + page));
            }
        }
        rt::Tensor pinned({static_cast<int64_t>(values.size())}, rt::DeviceType::kCPU, nvinfer1::DataType::kHALF,
            "layerDebuggerTest.poolStaging");
        std::memcpy(pinned.rawPointer(), values.data(), values.size() * sizeof(half));
        CUDA_CHECK(cudaMemcpyAsync(
            pool.rawPointer(), pinned.rawPointer(), values.size() * sizeof(half), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    };
    fillPoolByPage(/*layer=*/0, /*base=*/10.0F);
    fillPoolByPage(/*layer=*/1, /*base=*/20.0F);

    rt::Tensor logits({2, 4}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT, "logits");
    CUDA_CHECK(cudaMemsetAsync(logits.rawPointer(), 0, logits.getMemoryCapacity(), stream));
    auto debugger = rt::LayerDebugger::fromEnv();
    ASSERT_NE(debugger, nullptr);
    std::vector<rt::ResidentRef> const residents{{2, 1}, {0, 1}};
    debugger->dumpRound(cacheManager, fullTable, &swaTable, logits, {1, 1}, {2, 0}, residents, nullptr, 2, stream);
    debugger->flush(stream);

    std::filesystem::path dump;
    for (auto const& entry : std::filesystem::directory_iterator(mDir))
    {
        if (entry.path().extension() == ".safetensors")
        {
            dump = entry.path();
        }
    }
    ASSERT_FALSE(dump.empty());
    std::vector<rt::Tensor> tensors;
    ASSERT_TRUE(rt::safetensors::loadSafetensors(dump, tensors, stream));
    auto findTensor = [&](std::string const& name) -> rt::Tensor* {
        auto const found = std::find_if(tensors.begin(), tensors.end(),
            [&](rt::Tensor const& tensor) { return tensor.getName().find(name) != std::string::npos; });
        return found == tensors.end() ? nullptr : &*found;
    };
    auto readHalf = [&](rt::Tensor const& tensor) {
        std::vector<half> host(static_cast<size_t>(tensor.getShape().volume()));
        rt::Tensor pinned({static_cast<int64_t>(host.size())}, rt::DeviceType::kCPU, nvinfer1::DataType::kHALF,
            "layerDebuggerTest.readback");
        CUDA_CHECK(cudaMemcpyAsync(
            pinned.rawPointer(), tensor.rawPointer(), host.size() * sizeof(half), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        std::memcpy(host.data(), pinned.rawPointer(), host.size() * sizeof(half));
        return host;
    };

    rt::Tensor* full = findTensor("layer_0.kv");
    ASSERT_NE(full, nullptr);
    EXPECT_EQ(full->getShape(), rt::Coords({2, 2, 1, maxSequenceLength, 64}));
    auto const fullValues = readHalf(*full);
    size_t const fullHalf = static_cast<size_t>(maxSequenceLength) * 64;
    size_t const fullBatch = 2 * fullHalf;
    EXPECT_EQ(__half2float(fullValues[0]), 14.0F);
    EXPECT_EQ(__half2float(fullValues[fullHalf]), 114.0F);
    EXPECT_EQ(__half2float(fullValues[fullBatch]), 15.0F);

    rt::Tensor* reduced = findTensor("layer_1.kv");
    ASSERT_NE(reduced, nullptr);
    EXPECT_EQ(reduced->getShape(), rt::Coords({2, 2, 1, maxSequenceLength, 64}));
    auto const reducedValues = readHalf(*reduced);
    size_t const reducedHalf = static_cast<size_t>(maxSequenceLength) * 64;
    size_t const reducedBatch = 2 * reducedHalf;
    size_t const logicalPage1 = static_cast<size_t>(rt::kTOKENS_PER_PAGE) * 64;
    EXPECT_EQ(__half2float(reducedValues[0]), 0.0F);
    EXPECT_EQ(__half2float(reducedValues[logicalPage1]), 27.0F);
    EXPECT_EQ(__half2float(reducedValues[reducedHalf + logicalPage1]), 127.0F);
    EXPECT_EQ(__half2float(reducedValues[reducedBatch]), 22.0F);
    EXPECT_EQ(__half2float(reducedValues[reducedBatch + reducedHalf]), 122.0F);

    ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

} // namespace
