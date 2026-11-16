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

#include "common/checkMacros.h"
#include "kernels/embeddingKernels/pleGather.h"
#include "runtime/embedding/pleIoBudget.h"
#include "runtime/preprocess/gemma4EmbeddingPreprocessor.h"
#include "scratchDir.h"

#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <memory>
#include <vector>

namespace trt_edgellm
{
namespace rt
{
namespace
{
class Gemma4PlePreprocessorTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        mDirectory = makeScratchDir("gemma-ple-preprocessor");
        CUDA_CHECK(cudaStreamCreateWithFlags(&mStream, cudaStreamNonBlocking));
        mConfig.gemmaPleEnabled = true;
        mConfig.gemmaPleLayerCount = 3;
        mConfig.gemmaPleHiddenSize = 4;
        mConfig.padTokenId = 7;
        mConfig.imageTokenId = 30;
        mConfig.audioTokenId = 40;
        mConfig.plePool = PlePoolConfig{8, 12, PleTable::DType::kFP16, 1.0};
        mConfig.plePool.source = {"weight", PleTable::DType::kFP16, 8 * 12 * 2, 1.0, {{0, "0030"}}, "table.bin", 0};
        mConfig.plePool.checkpointDir = mDirectory.string();
        writeSource();
    }
    void TearDown() override
    {
        EXPECT_EQ(cudaStreamSynchronize(mStream), cudaSuccess);
        EXPECT_EQ(cudaStreamDestroy(mStream), cudaSuccess);
        std::filesystem::remove_all(mDirectory);
    }
    static uint16_t payload(int32_t row, int32_t column)
    {
        return static_cast<uint16_t>(0x3000 + row * 16 + column);
    }
    void writeSource()
    {
        std::ofstream output(mDirectory / "table.bin", std::ios::binary);
        for (int32_t row = 0; row < 8; ++row)
        {
            for (int32_t column = 0; column < 12; ++column)
            {
                uint16_t const value = payload(row, column);
                output.write(reinterpret_cast<char const*>(&value), sizeof(value));
            }
        }
        ASSERT_TRUE(output.good());
    }
    void embed(Gemma4EmbeddingPreprocessor& preprocessor, std::vector<int32_t> const& ids, int64_t batch = 2,
        std::vector<int32_t> const& queryLengths = {}, bool provideHostInputs = true, bool flat = false)
    {
        Tensor host({batch, static_cast<int64_t>(ids.size()) / batch}, DeviceType::kCPU, trt_edgellm::DataType::kINT32);
        Tensor device(host.getShape(), DeviceType::kGPU, trt_edgellm::DataType::kINT32);
        std::copy(ids.begin(), ids.end(), host.dataPointer<int32_t>());
        CUDA_CHECK(cudaMemcpyAsync(
            device.rawPointer(), host.rawPointer(), ids.size() * sizeof(int32_t), cudaMemcpyHostToDevice, mStream));
        Tensor hostLengths;
        Tensor deviceLengths;
        if (!queryLengths.empty())
        {
            hostLengths
                = Tensor({static_cast<int64_t>(queryLengths.size())}, DeviceType::kCPU, trt_edgellm::DataType::kINT32);
            deviceLengths = Tensor(hostLengths.getShape(), DeviceType::kGPU, trt_edgellm::DataType::kINT32);
            std::copy(queryLengths.begin(), queryLengths.end(), hostLengths.dataPointer<int32_t>());
            CUDA_CHECK(cudaMemcpyAsync(deviceLengths.rawPointer(), hostLengths.rawPointer(),
                queryLengths.size() * sizeof(int32_t), cudaMemcpyHostToDevice, mStream));
        }
        try
        {
            check::check(host.reshape({static_cast<int64_t>(ids.size())}), "Host token snapshot reshape failed");
            if (flat)
            {
                check::check(device.reshape({static_cast<int64_t>(ids.size())}), "Flat token reshape failed");
            }
            preprocessor.embed(device, mStream, provideHostInputs ? queryLengths : std::vector<int32_t>{},
                queryLengths.empty() ? nullptr : &deviceLengths, provideHostInputs ? &host : nullptr);
        }
        catch (...)
        {
            CUDA_CHECK(cudaStreamSynchronize(mStream));
            throw;
        }
        CUDA_CHECK(cudaStreamSynchronize(mStream));
    }
    void verify(TensorMap const& tensors, std::vector<int32_t> const& ids)
    {
        auto const* pool = tensors.get("ple_cache_pool");
        auto const* map = tensors.get("ple_row_map");
        ASSERT_NE(pool, nullptr);
        ASSERT_NE(map, nullptr);
        Tensor output({static_cast<int64_t>(ids.size()), 12}, DeviceType::kGPU, trt_edgellm::DataType::kHALF);
        Tensor host(output.getShape(), DeviceType::kCPU, trt_edgellm::DataType::kHALF);
        CUDA_CHECK(kernel::launchPleGather(
            pool->rawPointer(), map->dataPointer<int32_t>(), output.rawPointer(), ids.size(), 24, mStream));
        CUDA_CHECK(
            cudaMemcpyAsync(host.rawPointer(), output.rawPointer(), ids.size() * 24, cudaMemcpyDeviceToHost, mStream));
        CUDA_CHECK(cudaStreamSynchronize(mStream));
        auto const* bits = static_cast<uint16_t const*>(host.rawPointer());
        for (size_t token = 0; token < ids.size(); ++token)
        {
            auto const id = ids[token];
            bool const media = (mConfig.imageTokenId >= 0 && id == mConfig.imageTokenId)
                || (mConfig.audioTokenId >= 0 && id == mConfig.audioTokenId);
            int32_t const lookupId = media ? mConfig.padTokenId : id;
            bool const padding = lookupId < 0 || lookupId >= 8;
            for (int32_t column = 0; column < 12; ++column)
            {
                EXPECT_EQ(bits[token * 12 + column], padding ? 0 : payload(lookupId, column));
            }
        }
    }
    std::filesystem::path mDirectory;
    cudaStream_t mStream{};
    LLMEngineConfig mConfig;
};

TEST_F(Gemma4PlePreprocessorTest, SharedIoBudgetTracksCachedPreprocessorsAndPreservesOutput)
{
    auto budget = std::make_shared<PleIoBudget>();
    PleRuntimeConfig const policy{PleResidency::kCACHED, 5};
    std::vector<int32_t> const ids{0, 7, 30, 40, 9, 1, -1, 0, 8, 2};
    {
        TensorMap firstTensors;
        Gemma4EmbeddingPreprocessor first(mConfig, 2, 5, firstTensors, mStream, policy, {}, budget);
        size_t const firstBytes = budget->usedBytes();
        ASSERT_GT(firstBytes, 0);
        {
            TensorMap secondTensors;
            Gemma4EmbeddingPreprocessor second(mConfig, 2, 5, secondTensors, mStream, policy, {}, budget);
            EXPECT_GT(budget->usedBytes(), firstBytes);
            EXPECT_LE(budget->usedBytes(), PleIoBudget::kLIMIT_BYTES);
            embed(first, ids);
            embed(second, ids);
            ASSERT_NO_FATAL_FAILURE(verify(firstTensors, ids));
            ASSERT_NO_FATAL_FAILURE(verify(secondTensors, ids));
        }
        EXPECT_EQ(budget->usedBytes(), firstBytes);
        embed(first, {5, 6, 5, -1});
        ASSERT_NO_FATAL_FAILURE(verify(firstTensors, {5, 6, 5, -1}));
    }
    EXPECT_EQ(budget->usedBytes(), 0);
}

TEST_F(Gemma4PlePreprocessorTest, ResidentAndCachedSelectIdenticalRowsAndMapMediaToPadRow)
{
    // Given media placeholders and a configured pad row, when either residency mode prepares PLE rows,
    // then both modes select the pad row for media and preserve ordinary token rows.
    for (auto const residency : {PleResidency::kRESIDENT, PleResidency::kCACHED})
    {
        TensorMap tensors;
        PleRuntimeConfig const policy{residency, 5};
        Gemma4EmbeddingPreprocessor preprocessor(mConfig, 2, 5, tensors, mStream, policy);
        ASSERT_EQ(tensors.size(), 2);
        auto* const map = tensors.get("ple_row_map");
        void* const mapAddress = map->rawPointer();
        auto* const pool = tensors.get("ple_cache_pool");
        EXPECT_EQ(pool->getShape()[0], residency == PleResidency::kRESIDENT ? 9 : 5);
        for (auto const& ids :
            std::vector<std::vector<int32_t>>{{0, 7, 30, 40, 9, 1, -1, 0, 8, 2}, {5, 6, 5, -1}, {0, 7, 1, 2}})
        {
            embed(preprocessor, ids);
            ASSERT_NO_FATAL_FAILURE(verify(tensors, ids));
            EXPECT_EQ(map->rawPointer(), mapAddress);
            EXPECT_EQ(map->getShape(), (Coords{static_cast<int64_t>(ids.size()), 1}));
        }
    }
}

TEST_F(Gemma4PlePreprocessorTest, ResidentNeedsNoBackingReadsAndReshapePreservesPreparedRows)
{
    for (auto const residency : {PleResidency::kRESIDENT, PleResidency::kCACHED})
    {
        mConfig.plePool.source = {"weight", PleTable::DType::kFP16, 8 * 12 * 2, 1.0, {{0, "0030"}}, "table.bin", 0};
        mConfig.plePool.checkpointDir = mDirectory.string();
        writeSource();
        TensorMap tensors;
        PleRuntimeConfig const policy{residency, 5};
        Gemma4EmbeddingPreprocessor preprocessor(mConfig, 2, 5, tensors, mStream, policy);
        preprocessor.reshapeOutputsTokenMajor(10);
        ASSERT_NO_FATAL_FAILURE(verify(tensors, std::vector<int32_t>(10, -1)));
        embed(preprocessor, {0, 7, 0, 7});
        std::filesystem::resize_file(mDirectory / "table.bin", 0);
        preprocessor.reshapeOutputsTokenMajor(4);
        ASSERT_NO_FATAL_FAILURE(verify(tensors, {0, 7, 0, 7}));
        if (residency == PleResidency::kRESIDENT)
        {
            embed(preprocessor, {1, 6, 2, 5});
            ASSERT_NO_FATAL_FAILURE(verify(tensors, {1, 6, 2, 5}));
        }
        else
        {
            EXPECT_THROW(embed(preprocessor, {1, 6, 2, 5}), std::runtime_error);
        }
    }
}

TEST_F(Gemma4PlePreprocessorTest, UnevenRowsMaskPaddingWithinExactCacheCapacity)
{
    for (auto const residency : {PleResidency::kRESIDENT, PleResidency::kCACHED})
    {
        TensorMap tensors;
        PleRuntimeConfig const policy{residency, 3};
        Gemma4EmbeddingPreprocessor preprocessor(mConfig, 2, 3, tensors, mStream, policy);
        // Padding token zero must not admit a third row into a two-row cache.
        embed(preprocessor, {1, 1, 1, 2, 0, 0}, 2, {3, 1});
        preprocessor.reshapeOutputsTokenMajor(6);
        ASSERT_NO_FATAL_FAILURE(verify(tensors, {1, 1, 1, 2, -1, -1}));
        embed(preprocessor, {0, 0, 0, 2, 0, 0}, 2, {1, 1});
        ASSERT_NO_FATAL_FAILURE(verify(tensors, {0, -1, -1, 2, -1, -1}));
    }
}

TEST_F(Gemma4PlePreprocessorTest, ResidentMapUsesDeviceQueryLengthsForVerificationPadding)
{
    TensorMap tensors;
    Gemma4EmbeddingPreprocessor preprocessor(mConfig, 2, 3, tensors, mStream);
    embed(preprocessor, {0, 1, 2, 5, 0, 0}, 2, {3, 1}, false);
    ASSERT_NO_FATAL_FAILURE(verify(tensors, {0, 1, 2, 5, -1, -1}));
    embed(preprocessor, {0, 1, 2, 5, 0, 0}, 2, {1, 3}, false);
    ASSERT_NO_FATAL_FAILURE(verify(tensors, {0, -1, -1, 5, 0, 0}));
}

TEST_F(Gemma4PlePreprocessorTest, FlatVerificationPreservesRequestPadding)
{
    // Given flat candidates for two requests, when mapping PLE rows, then only each request's valid rows survive.
    TensorMap tensors;
    Gemma4EmbeddingPreprocessor preprocessor(mConfig, 2, 3, tensors, mStream);
    embed(preprocessor, {0, 1, 2, 5, 0, 0}, 2, {3, 1}, false, true);
    Tensor host({6}, DeviceType::kCPU, trt_edgellm::DataType::kINT32);
    CUDA_CHECK(cudaMemcpyAsync(host.rawPointer(), tensors.get("ple_row_map")->rawPointer(), 6 * sizeof(int32_t),
        cudaMemcpyDeviceToHost, mStream));
    CUDA_CHECK(cudaStreamSynchronize(mStream));
    std::vector<int32_t> const actual(host.dataPointer<int32_t>(), host.dataPointer<int32_t>() + 6);
    EXPECT_EQ(actual, (std::vector<int32_t>{1, 2, 3, 6, 0, 0}));
}

TEST_F(Gemma4PlePreprocessorTest, CachedRequiresHostSnapshotBeforeChangingPreparedRows)
{
    TensorMap tensors;
    PleRuntimeConfig const policy{PleResidency::kCACHED, 3};
    Gemma4EmbeddingPreprocessor preprocessor(mConfig, 2, 3, tensors, mStream, policy);
    embed(preprocessor, {1, 2}, 2, {1, 1});
    Tensor device({2, 1}, DeviceType::kGPU, trt_edgellm::DataType::kINT32);
    Tensor shortHost({1}, DeviceType::kCPU, trt_edgellm::DataType::kINT32);
    EXPECT_THROW(preprocessor.embed(device, mStream), std::exception);
    EXPECT_THROW(preprocessor.embed(device, mStream, {}, nullptr, &shortHost), std::exception);
    ASSERT_NO_FATAL_FAILURE(verify(tensors, {1, 2}));
    embed(preprocessor, {5, 6}, 2, {1, 1});
    ASSERT_NO_FATAL_FAILURE(verify(tensors, {5, 6}));
}

TEST_F(Gemma4PlePreprocessorTest, AdmissionAndCompactionRebuildMapInExecutionOrder)
{
    for (auto const residency : {PleResidency::kRESIDENT, PleResidency::kCACHED})
    {
        TensorMap tensors;
        PleRuntimeConfig const policy{residency, 3};
        Gemma4EmbeddingPreprocessor preprocessor(mConfig, 2, 3, tensors, mStream, policy);
        auto const mapAddress = tensors.get("ple_row_map")->rawPointer();
        embed(preprocessor, {1, 2}, 2, {1, 1});
        ASSERT_NO_FATAL_FAILURE(verify(tensors, {1, 2}));
        embed(preprocessor, {5, 6, 5}, 1, {3});
        ASSERT_NO_FATAL_FAILURE(verify(tensors, {5, 6, 5}));
        embed(preprocessor, {6, 1}, 2, {1, 1});
        ASSERT_NO_FATAL_FAILURE(verify(tensors, {6, 1}));
        embed(preprocessor, {0}, 1, {1});
        ASSERT_NO_FATAL_FAILURE(verify(tensors, {0}));
        EXPECT_EQ(tensors.get("ple_row_map")->rawPointer(), mapAddress);
    }
}

TEST_F(Gemma4PlePreprocessorTest, CapturedGatherReadsUpdatedMapAfterReshape)
{
    for (auto const residency : {PleResidency::kRESIDENT, PleResidency::kCACHED})
    {
        TensorMap tensors;
        PleRuntimeConfig const policy{residency, 3};
        Gemma4EmbeddingPreprocessor preprocessor(mConfig, 2, 3, tensors, mStream, policy);
        preprocessor.reshapeOutputsTokenMajor(2);
        Tensor output({2, 12}, DeviceType::kGPU, trt_edgellm::DataType::kHALF);
        Tensor host(output.getShape(), DeviceType::kCPU, trt_edgellm::DataType::kHALF);
        CUDA_CHECK(cudaStreamSynchronize(mStream));
        cudaGraph_t graph{};
        cudaGraphExec_t executable{};
        CUDA_CHECK(cudaStreamBeginCapture(mStream, cudaStreamCaptureModeThreadLocal));
        CUDA_CHECK(kernel::launchPleGather(tensors.get("ple_cache_pool")->rawPointer(),
            tensors.get("ple_row_map")->dataPointer<int32_t>(), output.rawPointer(), 2, 24, mStream));
        CUDA_CHECK(cudaStreamEndCapture(mStream, &graph));
        CUDA_CHECK(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
        for (auto const& ids : std::vector<std::vector<int32_t>>{{0, 2}, {6, 1}})
        {
            embed(preprocessor, ids, 2, {1, 1});
            preprocessor.reshapeOutputsTokenMajor(2);
            CUDA_CHECK(cudaGraphLaunch(executable, mStream));
            CUDA_CHECK(cudaMemcpyAsync(host.rawPointer(), output.rawPointer(), 48, cudaMemcpyDeviceToHost, mStream));
            CUDA_CHECK(cudaStreamSynchronize(mStream));
            for (int32_t token = 0; token < 2; ++token)
            {
                for (int32_t column = 0; column < 12; ++column)
                {
                    EXPECT_EQ(static_cast<uint16_t const*>(host.rawPointer())[token * 12 + column],
                        payload(ids[token], column));
                }
            }
        }
        CUDA_CHECK(cudaGraphExecDestroy(executable));
        CUDA_CHECK(cudaGraphDestroy(graph));
    }
}

TEST_F(Gemma4PlePreprocessorTest, ExplicitFileOverridesUnavailableCheckpoint)
{
    mConfig.plePool.checkpointDir = (mDirectory / "missing-checkpoint").string();
    for (auto const residency : {PleResidency::kRESIDENT, PleResidency::kCACHED})
    {
        TensorMap tensors;
        PleRuntimeConfig const policy{residency, 9, (mDirectory / "table.bin").string()};
        Gemma4EmbeddingPreprocessor preprocessor(mConfig, 2, 3, tensors, mStream, policy);
        embed(preprocessor, {0, 1, 2, 5, 6, 7});
        verify(tensors, {0, 1, 2, 5, 6, 7});
    }
}

TEST_F(Gemma4PlePreprocessorTest, ExplicitFileDoesNotFallBackToValidCheckpoint)
{
    TensorMap tensors;
    PleRuntimeConfig const policy{PleResidency::kRESIDENT, 0, (mDirectory / "missing.bin").string()};
    EXPECT_THROW(Gemma4EmbeddingPreprocessor(mConfig, 2, 3, tensors, mStream, policy), std::exception);
}

TEST_F(Gemma4PlePreprocessorTest, RequestedCheckpointOverridesRecordedDirectory)
{
    mConfig.plePool.checkpointDir = (mDirectory / "missing-checkpoint").string();
    TensorMap tensors;
    Gemma4EmbeddingPreprocessor preprocessor(mConfig, 2, 3, tensors, mStream, {}, mDirectory);
    embed(preprocessor, {0, 1, 2, 5, 6, 7});
    verify(tensors, {0, 1, 2, 5, 6, 7});
}

TEST_F(Gemma4PlePreprocessorTest, RejectsMissingSourceEvenWithExplicitFile)
{
    mConfig.plePool.source = {};
    TensorMap tensors;
    EXPECT_THROW(Gemma4EmbeddingPreprocessor(mConfig, 2, 3, tensors, mStream), std::exception);
    PleRuntimeConfig const policy{PleResidency::kRESIDENT, 0, (mDirectory / "table.bin").string()};
    EXPECT_THROW(Gemma4EmbeddingPreprocessor(mConfig, 2, 3, tensors, mStream, policy), std::exception);
}

TEST_F(Gemma4PlePreprocessorTest, RejectsInvalidSourceAndCapacity)
{
    TensorMap tensors;
    auto bad = mConfig;
    bad.plePool.rowWidth = 11;
    EXPECT_THROW(Gemma4EmbeddingPreprocessor(bad, 2, 5, tensors, mStream), std::exception);
    PleRuntimeConfig const invalid{PleResidency::kCACHED, 0};
    EXPECT_THROW(Gemma4EmbeddingPreprocessor(mConfig, 2, 5, tensors, mStream, invalid), std::exception);
    Gemma4EmbeddingPreprocessor preprocessor(mConfig, 2, 5, tensors, mStream);
    EXPECT_THROW(preprocessor.reshapeOutputsTokenMajor(0), std::exception);
    EXPECT_THROW(preprocessor.reshapeOutputsTokenMajor(11), std::exception);
    EXPECT_THROW(embed(preprocessor, {0, 1, 2}, 3), std::exception);
    EXPECT_THROW(embed(preprocessor, {0, 1, 2, 5, 6, 7}, 1), std::exception);
    EXPECT_THROW(embed(preprocessor, {0, 1}, 2, {1}), std::exception);
    EXPECT_THROW(embed(preprocessor, {0, 1}, 2, {1, 2}), std::exception);
    EXPECT_THROW(embed(preprocessor, {0, 1}, 2, {-1, 1}), std::exception);
}

} // namespace
} // namespace rt
} // namespace trt_edgellm
