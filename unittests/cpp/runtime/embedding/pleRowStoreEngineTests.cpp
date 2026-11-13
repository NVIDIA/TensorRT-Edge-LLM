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

#include "builder/builderUtils.h"
#include "common/checkMacros.h"
#include "common/trtInterop.h"
#include "common/trtUtils.h"
#include "runtime/embedding/pleRowStore.h"
#include "runtime/exec/engineExecutor.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <unordered_map>

namespace trt_edgellm
{
namespace rt
{
namespace
{
constexpr char kPOOL[]{"ple_cache_pool"};
constexpr char kMAP[]{"ple_row_map"};
constexpr char kOUTPUT[]{"ple_embeddings"};
constexpr int64_t kHEADS{16};
constexpr int64_t kWIDTH{160};
constexpr int64_t kMAX_BATCH{2};
constexpr int64_t kMAX_SEQ{8};

struct CancelledAfterEnqueue : std::exception
{
};

class PleRowStoreEngineTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        auto const* fixture = std::getenv("EDGELLM_PLE_COMPONENT_ONNX");
        if (fixture == nullptr)
        {
            GTEST_SKIP() << "Set EDGELLM_PLE_COMPONENT_ONNX to a table-bound component fixture.";
        }
        CUDA_CHECK(cudaStreamCreateWithFlags(&mStream, cudaStreamNonBlocking));
        mPlugin = loadEdgellmPluginLib();
        ASSERT_NE(mPlugin, nullptr);
        std::filesystem::path const onnxPath(fixture);
        auto descriptor = onnxPath;
        descriptor.replace_extension(".json");
        std::ifstream metadataFile(descriptor);
        auto const metadata = nlohmann::json::parse(metadataFile);
        mScale = metadata.at("scale").get<double>();
        int32_t const capacity = metadata.at("capacity").get<int32_t>();
        ASSERT_EQ(capacity, 17);
        mTable.emplace(PleTable::open(
            metadata.at("artifact_descriptor").get<std::string>(), metadata.at("table_id").get<std::string>(), mScale));

        builder::applyCompileWorkarounds();
        auto [trtBuilder, network] = builder::createBuilderAndNetwork();
        ASSERT_NE(trtBuilder, nullptr);
        ASSERT_NE(network, nullptr);
        auto parser = builder::parseOnnxModel(network.get(), onnxPath.string());
        ASSERT_NE(parser, nullptr);
        ASSERT_EQ(network->getNbInputs(), 2);
        ASSERT_EQ(network->getNbOutputs(), 1);
        for (int32_t i = 0; i < network->getNbInputs(); ++i)
        {
            network->getInput(i)->setAllowedFormats(1U << static_cast<uint32_t>(nvinfer1::TensorFormat::kLINEAR));
        }
        mOutputType = fromTrt(network->getOutput(0)->getType());
        ASSERT_TRUE(mOutputType == trt_edgellm::DataType::kHALF || mOutputType == trt_edgellm::DataType::kBF16);
        auto config = builder::createBuilderConfig(trtBuilder.get());
        ASSERT_NE(config, nullptr);
        config->setProfileStream(mStream);
        auto* profile = trtBuilder->createOptimizationProfile();
        ASSERT_NE(profile, nullptr);
        auto const poolShape = builder::createDims({capacity, kWIDTH});
        ASSERT_TRUE(builder::setOptimizationProfile(profile, kPOOL, poolShape, poolShape, poolShape));
        ASSERT_TRUE(builder::setOptimizationProfile(profile, kMAP, builder::createDims({1, 1, kHEADS}),
            builder::createDims({1, 3, kHEADS}), builder::createDims({kMAX_BATCH, kMAX_SEQ, kHEADS})));
        ASSERT_GE(config->addOptimizationProfile(profile), 0);
        auto enginePath = onnxPath;
        enginePath.replace_extension(".engine");
        ASSERT_TRUE(
            builder::buildAndSerializeEngine(trtBuilder.get(), network.get(), config.get(), enginePath.string()));

        TensorRegistry registry;
        registry.addTensor({kPOOL, TensorIO::kInput, trt_edgellm::DataType::kFP8, {fixed(capacity), fixed(kWIDTH)}});
        registry.addTensor({kMAP, TensorIO::kInput, trt_edgellm::DataType::kINT32,
            {sym(&InferenceDims::batch), sym(&InferenceDims::seqLen), fixed(kHEADS)}});
        registry.addTensor({kOUTPUT, TensorIO::kOutput, mOutputType,
            {sym(&InferenceDims::batch), sym(&InferenceDims::seqLen), fixed(kHEADS * kWIDTH)}});
        mExecutor = EngineExecutor::createWithRegistry(enginePath, std::move(registry));
        ASSERT_NE(mExecutor, nullptr);
        mContextMemory = Tensor({std::max<int64_t>(1, mExecutor->getRequiredContextMemorySize())}, DeviceType::kGPU,
            trt_edgellm::DataType::kINT8);
        ASSERT_TRUE(mExecutor->setContextMemory(mContextMemory));
        PleIoConfig io;
        io.fillBytes = 3 * mTable->sourceRowBytes();
        io.readBytes = 8192;
        mStore = std::make_unique<PleRowStore>(*mTable, capacity, mStream, io);
        mMap = Tensor({kMAX_BATCH, kMAX_SEQ, kHEADS}, DeviceType::kGPU, trt_edgellm::DataType::kINT32);
        mHostMap = Tensor({kMAX_BATCH, kMAX_SEQ, kHEADS}, DeviceType::kCPU, trt_edgellm::DataType::kINT32);
        mOutput = Tensor({kMAX_BATCH, kMAX_SEQ, kHEADS * kWIDTH}, DeviceType::kGPU, mOutputType);
        mHostOutput = Tensor({kMAX_BATCH, kMAX_SEQ, kHEADS * kWIDTH}, DeviceType::kCPU, mOutputType);
        mBindings.set(kPOOL, mStore->pool());
        mBindings.set(kMAP, mMap);
        mBindings.set(kOUTPUT, mOutput);
    }

    void TearDown() override
    {
        if (mStream != nullptr)
        {
            EXPECT_EQ(cudaStreamSynchronize(mStream), cudaSuccess);
            mExecutor.reset();
            mStore.reset();
            EXPECT_EQ(cudaStreamDestroy(mStream), cudaSuccess);
            mStream = nullptr;
        }
    }

    std::vector<int64_t> rows(int64_t first, int64_t distinct, int64_t tokens = 1)
    {
        std::vector<int64_t> result(static_cast<size_t>(tokens * kHEADS));
        for (size_t i = 0; i < result.size(); ++i)
        {
            result[i] = first + static_cast<int64_t>(i) % distinct;
        }
        return result;
    }

    void run(std::vector<std::vector<int64_t>> const& requests, bool capture = false, bool cancelAfterEnqueue = false)
    {
        auto leases = mStore->prepare(requests);
        int64_t const batch = static_cast<int64_t>(requests.size());
        int64_t seq{1};
        for (auto const& request : requests)
        {
            ASSERT_EQ(request.size() % kHEADS, 0);
            seq = std::max<int64_t>(seq, static_cast<int64_t>(request.size()) / kHEADS);
        }
        ASSERT_LE(batch, kMAX_BATCH);
        ASSERT_LE(seq, kMAX_SEQ);
        auto* map = mHostMap.dataPointer<int32_t>();
        std::fill_n(map, batch * seq * kHEADS, 0);
        for (size_t i = 0; i < leases.size(); ++i)
        {
            std::copy(leases[i].mapping().begin(), leases[i].mapping().end(), map + i * seq * kHEADS);
        }
        try
        {
            CUDA_CHECK(cudaMemcpyAsync(
                mMap.rawPointer(), map, batch * seq * kHEADS * sizeof(int32_t), cudaMemcpyHostToDevice, mStream));
            InferenceDims const dims{batch, seq, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, batch + 1, 0};
            if (!mExecutor->prepare(0, dims, mBindings, mStream) || (capture && !mExecutor->captureGraph(mStream))
                || !mExecutor->execute(mStream))
            {
                throw std::runtime_error("PLE component execution failed");
            }
            if (cancelAfterEnqueue)
            {
                throw CancelledAfterEnqueue{};
            }
            CUDA_CHECK(cudaMemcpyAsync(mHostOutput.rawPointer(), mOutput.rawPointer(),
                batch * seq * kHEADS * kWIDTH * sizeof(half), cudaMemcpyDeviceToHost, mStream));
            CUDA_CHECK(cudaStreamSynchronize(mStream));
        }
        catch (...)
        {
            // The queued engine may still consume the pool/map even when its request has been cancelled.
            if (cudaStreamSynchronize(mStream) != cudaSuccess)
            {
                std::terminate();
            }
            throw;
        }
        std::unordered_map<int64_t, std::array<uint8_t, kWIDTH>> reference;
        for (auto const& request : requests)
        {
            for (int64_t row : request)
            {
                if (row == -1 || reference.count(row))
                {
                    continue;
                }
                auto const location = mTable->locate(row);
                std::ifstream file(mTable->shards()[location.shardIndex].path, std::ios::binary);
                file.seekg(location.byteOffset);
                file.read(reinterpret_cast<char*>(reference[row].data()), kWIDTH);
                ASSERT_EQ(file.gcount(), kWIDTH);
            }
        }
        for (int64_t b = 0; b < batch; ++b)
        {
            for (int64_t index = 0; index < seq * kHEADS; ++index)
            {
                int64_t const row = index < static_cast<int64_t>(requests[b].size()) ? requests[b][index] : -1;
                for (int64_t column = 0; column < kWIDTH; ++column)
                {
                    __nv_fp8_e4m3 value;
                    uint8_t const raw = row == -1 ? 0 : reference.at(row)[column];
                    std::memcpy(&value, &raw, sizeof(raw));
                    float const scaled = static_cast<float>(value) * static_cast<float>(mScale);
                    auto const outputIndex = (b * seq * kHEADS + index) * kWIDTH + column;
                    float const expected = mOutputType == trt_edgellm::DataType::kHALF
                        ? __half2float(__float2half_rn(scaled))
                        : __bfloat162float(__float2bfloat16_rn(scaled));
                    float const actual = mOutputType == trt_edgellm::DataType::kHALF
                        ? __half2float(static_cast<half const*>(mHostOutput.rawPointer())[outputIndex])
                        : __bfloat162float(static_cast<__nv_bfloat16 const*>(mHostOutput.rawPointer())[outputIndex]);
                    ASSERT_EQ(actual, expected) << "batch=" << b << " row=" << row << " column=" << column;
                }
            }
        }
    }

    cudaStream_t mStream{};
    std::unique_ptr<void, DlDeleter> mPlugin;
    std::optional<PleTable> mTable;
    std::unique_ptr<PleRowStore> mStore;
    std::unique_ptr<EngineExecutor> mExecutor;
    double mScale{};
    trt_edgellm::DataType mOutputType{};
    Tensor mContextMemory;
    Tensor mMap;
    Tensor mHostMap;
    Tensor mOutput;
    Tensor mHostOutput;
    TensorMap mBindings;
};

TEST_F(PleRowStoreEngineTest, RequestsShareRowsEvictAndReplayAtStableAddresses)
{
    auto* const poolAddress = mStore->pool().rawPointer();
    auto const first = rows(0, 12);
    auto const second = rows(4, 12);
    ASSERT_NO_FATAL_FAILURE(run({first, second}, true));
    EXPECT_EQ(mStore->stats().misses, 16);
    EXPECT_GT(mStore->stats().fillWaves, 1);
    ASSERT_NO_FATAL_FAILURE(run({second, first}));
    EXPECT_EQ(mStore->stats().readCalls, 0);
    ASSERT_NO_FATAL_FAILURE(run({rows(64, 16, 3)}, true));
    EXPECT_EQ(mStore->stats().misses, 16);
    ASSERT_NO_FATAL_FAILURE(run({rows(64, 8, 2), rows(72, 8)}));
    EXPECT_EQ(mStore->stats().readCalls, 0);
    ASSERT_NO_FATAL_FAILURE(run({rows(112, 16), rows(112, 8)}));
    ASSERT_NO_FATAL_FAILURE(run({first, second}));
    EXPECT_EQ(mStore->pool().rawPointer(), poolAddress);
}

TEST_F(PleRowStoreEngineTest, PaddingCapacityRejectionAndCancellationLeaveCacheUsable)
{
    std::vector<int64_t> const zeros(kHEADS, -1);
    ASSERT_NO_FATAL_FAILURE(run({zeros, zeros}, true));
    EXPECT_EQ(mStore->stats().readCalls, 0);
    {
        auto cancelledBeforeExecution = mStore->prepare({rows(0, 16)});
    }
    ASSERT_NO_FATAL_FAILURE(run({rows(0, 16)}));
    EXPECT_EQ(mStore->stats().readCalls, 0);
    EXPECT_THROW(mStore->prepare({rows(0, 16), rows(16, 1)}), std::length_error);
    ASSERT_NO_FATAL_FAILURE(run({rows(0, 16)}));
    EXPECT_EQ(mStore->stats().readCalls, 0);
    EXPECT_THROW(run({rows(64, 16)}, false, true), CancelledAfterEnqueue);
    ASSERT_NO_FATAL_FAILURE(run({rows(64, 16)}));
    EXPECT_EQ(mStore->stats().readCalls, 0);
    ASSERT_NO_FATAL_FAILURE(run({rows(112, 16), zeros}));
}

} // namespace
} // namespace rt
} // namespace trt_edgellm
