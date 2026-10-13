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

#include "allReducePluginTestUtils.h"

#include "common/cudaUtils.h"
#include "runtime/multiDevice/backends/nccl/tensorParallelNcclResources.h"
#include "runtime/multiDevice/ncclCollectiveBackend.h"

#include <cstdint>
#include <dlfcn.h>
#include <exception>
#include <gtest/gtest.h>
#include <string>
#include <thread>
#include <vector>

using namespace trt_edgellm;
using namespace trt_edgellm::rt;
using namespace trt_edgellm::test;

namespace
{

constexpr int32_t kTpSize{2};
constexpr int64_t kNumElements{4096};
constexpr float kTolerance{1e-2F};

using RegisterNcclPathFn = bool (*)(int, void*, void*);
using UnregisterNcclPathFn = bool (*)(int, void*);

RegisterNcclPathFn registerNcclPathSymbol() noexcept
{
    if (loadPluginLibrary() == nullptr)
    {
        return nullptr;
    }
    return reinterpret_cast<RegisterNcclPathFn>(dlsym(RTLD_DEFAULT, "edgellmRegisterNcclCommForAllReducePlugin"));
}

UnregisterNcclPathFn unregisterNcclPathSymbol() noexcept
{
    if (loadPluginLibrary() == nullptr)
    {
        return nullptr;
    }
    return reinterpret_cast<UnregisterNcclPathFn>(dlsym(RTLD_DEFAULT, "edgellmUnregisterNcclCommForAllReducePlugin"));
}

nvinfer1::DynamicPluginTensorDesc makeDynamicDesc(nvinfer1::DataType dataType, nvinfer1::TensorFormat format) noexcept
{
    nvinfer1::DynamicPluginTensorDesc desc{};
    desc.desc = makeLinearDesc(dataType, kNumElements);
    desc.desc.format = format;
    desc.min = desc.desc.dims;
    desc.opt = desc.desc.dims;
    desc.max = desc.desc.dims;
    return desc;
}

struct RankOutcome
{
    int32_t enqueueStatus{-1};
    std::vector<float> output{};
    std::string failure{};
};

//! Run all-reduce through the plugin. Every rank contributes
//! rank + 1, so a correct 2-rank AllReduce leaves 3 in every element. For example, rank 0 will have filled tensors
//! with 1.0, rank 1 will have filled tensors with 2.0, the results should give 3.0 in every element.
void runRank(int32_t rank, nvinfer1::DataType dataType, RankOutcome& outcome) noexcept
{
    if (cudaSetDevice(rank) != cudaSuccess)
    {
        outcome.failure = "cudaSetDevice failed for rank " + std::to_string(rank);
        return;
    }

    size_t const bytes = static_cast<size_t>(kNumElements) * dataTypeSize(dataType);
    DeviceBuffer input(bytes);
    DeviceBuffer output(bytes);
    if (input.get() == nullptr || output.get() == nullptr)
    {
        outcome.failure = "cudaMalloc failed for rank " + std::to_string(rank);
        return;
    }

    std::vector<uint8_t> const hostInput = makeHostBuffer(dataType, kNumElements, static_cast<float>(rank + 1));
    if (cudaMemcpy(input.get(), hostInput.data(), bytes, cudaMemcpyHostToDevice) != cudaSuccess)
    {
        outcome.failure = "Host to device copy failed for rank " + std::to_string(rank);
        return;
    }

    AllReducePluginOwner plugin(kTpSize);
    if (!plugin.valid())
    {
        outcome.failure = "AllReducePlugin creation failed for rank " + std::to_string(rank);
        return;
    }

    cudaStream_t stream{};
    if (cudaStreamCreate(&stream) != cudaSuccess)
    {
        outcome.failure = "cudaStreamCreate failed for rank " + std::to_string(rank);
        return;
    }

    nvinfer1::PluginTensorDesc const desc = makeLinearDesc(dataType, kNumElements);
    outcome.enqueueStatus = enqueueAllReduce(plugin.runtime(), desc, input.get(), output.get(), stream);
    cudaError_t const syncError = cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);

    if (outcome.enqueueStatus != 0)
    {
        outcome.failure
            = "enqueue returned " + std::to_string(outcome.enqueueStatus) + " for rank " + std::to_string(rank);
        return;
    }
    if (syncError != cudaSuccess)
    {
        outcome.failure
            = std::string("Stream sync failed for rank ") + std::to_string(rank) + ": " + cudaGetErrorString(syncError);
        return;
    }

    std::vector<uint8_t> hostOutput(bytes, 0);
    if (cudaMemcpy(hostOutput.data(), output.get(), bytes, cudaMemcpyDeviceToHost) != cudaSuccess)
    {
        outcome.failure = "Device to host copy failed for rank " + std::to_string(rank);
        return;
    }
    outcome.output = readHostBuffer(dataType, hostOutput, kNumElements);
}

} // namespace

TEST(AllReducePluginTest, IdentityPathCopiesInputWhenTpSizeIsOne)
{
    if (detectCudaDeviceCount() < 1)
    {
        GTEST_SKIP() << "No CUDA device is available.";
    }

    auto const dataType = nvinfer1::DataType::kFLOAT;
    size_t const bytes = static_cast<size_t>(kNumElements) * dataTypeSize(dataType);

    AllReducePluginOwner plugin(1);
    ASSERT_TRUE(plugin.valid()) << "Cannot create AllReducePlugin from " << pluginLibraryPath();

    DeviceBuffer input(bytes);
    DeviceBuffer output(bytes);
    ASSERT_NE(input.get(), nullptr);
    ASSERT_NE(output.get(), nullptr);

    std::vector<uint8_t> const hostInput = makeHostBuffer(dataType, kNumElements, 7.0F);
    ASSERT_EQ(cudaMemcpy(input.get(), hostInput.data(), bytes, cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemset(output.get(), 0, bytes), cudaSuccess);

    cudaStream_t stream{};
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

    nvinfer1::PluginTensorDesc const desc = makeLinearDesc(dataType, kNumElements);
    EXPECT_EQ(enqueueAllReduce(plugin.runtime(), desc, input.get(), output.get(), stream), 0);
    EXPECT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    cudaStreamDestroy(stream);

    std::vector<uint8_t> hostOutput(bytes, 0);
    ASSERT_EQ(cudaMemcpy(hostOutput.data(), output.get(), bytes, cudaMemcpyDeviceToHost), cudaSuccess);

    std::vector<float> const values = readHostBuffer(dataType, hostOutput, kNumElements);
    for (size_t index = 0; index < values.size(); ++index)
    {
        ASSERT_NEAR(values[index], 7.0F, kTolerance) << "Mismatch at element " << index;
    }
}

TEST(AllReducePluginTest, FailsWhenNoPathIsRegistered)
{
    if (detectCudaDeviceCount() < 1)
    {
        GTEST_SKIP() << "No CUDA device is available.";
    }

    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);

    auto const dataType = nvinfer1::DataType::kFLOAT;
    size_t const bytes = static_cast<size_t>(kNumElements) * dataTypeSize(dataType);

    AllReducePluginOwner plugin(kTpSize);
    ASSERT_TRUE(plugin.valid()) << "Cannot create AllReducePlugin from " << pluginLibraryPath();

    DeviceBuffer input(bytes);
    DeviceBuffer output(bytes);
    ASSERT_NE(input.get(), nullptr);
    ASSERT_NE(output.get(), nullptr);

    cudaStream_t stream{};
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

    nvinfer1::PluginTensorDesc const desc = makeLinearDesc(dataType, kNumElements);
    EXPECT_EQ(enqueueAllReduce(plugin.runtime(), desc, input.get(), output.get(), stream), -1);
    cudaStreamDestroy(stream);
}

TEST(AllReducePluginTest, RejectsUnsupportedFormatCombinations)
{
    AllReducePluginOwner plugin(kTpSize);
    ASSERT_TRUE(plugin.valid()) << "Cannot create AllReducePlugin from " << pluginLibraryPath();

    nvinfer1::IPluginV3OneBuild* const build = plugin.build();
    ASSERT_NE(build, nullptr);

    for (auto const dataType : {nvinfer1::DataType::kHALF, nvinfer1::DataType::kFLOAT, nvinfer1::DataType::kBF16})
    {
        SCOPED_TRACE("dataType=" + std::to_string(static_cast<int32_t>(dataType)));
        nvinfer1::DynamicPluginTensorDesc inOut[2]{
            makeDynamicDesc(dataType, nvinfer1::TensorFormat::kLINEAR),
            makeDynamicDesc(dataType, nvinfer1::TensorFormat::kLINEAR),
        };
        EXPECT_TRUE(build->supportsFormatCombination(0, inOut, 1, 1));
        EXPECT_TRUE(build->supportsFormatCombination(1, inOut, 1, 1));
    }

    for (auto const dataType : {nvinfer1::DataType::kINT8, nvinfer1::DataType::kINT32})
    {
        SCOPED_TRACE("dataType=" + std::to_string(static_cast<int32_t>(dataType)));
        nvinfer1::DynamicPluginTensorDesc inOut[2]{
            makeDynamicDesc(dataType, nvinfer1::TensorFormat::kLINEAR),
            makeDynamicDesc(dataType, nvinfer1::TensorFormat::kLINEAR),
        };
        EXPECT_FALSE(build->supportsFormatCombination(0, inOut, 1, 1));
    }

    nvinfer1::DynamicPluginTensorDesc nonLinear[2]{
        makeDynamicDesc(nvinfer1::DataType::kHALF, nvinfer1::TensorFormat::kCHW32),
        makeDynamicDesc(nvinfer1::DataType::kHALF, nvinfer1::TensorFormat::kLINEAR),
    };
    EXPECT_FALSE(build->supportsFormatCombination(0, nonLinear, 1, 1));

    nvinfer1::DynamicPluginTensorDesc mixedTypes[2]{
        makeDynamicDesc(nvinfer1::DataType::kHALF, nvinfer1::TensorFormat::kLINEAR),
        makeDynamicDesc(nvinfer1::DataType::kFLOAT, nvinfer1::TensorFormat::kLINEAR),
    };
    EXPECT_FALSE(build->supportsFormatCombination(1, mixedTypes, 1, 1));

    nvinfer1::DynamicPluginTensorDesc valid[2]{
        makeDynamicDesc(nvinfer1::DataType::kHALF, nvinfer1::TensorFormat::kLINEAR),
        makeDynamicDesc(nvinfer1::DataType::kHALF, nvinfer1::TensorFormat::kLINEAR),
    };
    EXPECT_FALSE(build->supportsFormatCombination(2, valid, 1, 1));
}

TEST(AllReducePluginTest, NcclPathRegistrationValidatesArguments)
{
    RegisterNcclPathFn const registerPath = registerNcclPathSymbol();
    UnregisterNcclPathFn const unregisterPath = unregisterNcclPathSymbol();
    ASSERT_NE(registerPath, nullptr) << "Cannot resolve the NCCL registration symbol in " << pluginLibraryPath();
    ASSERT_NE(unregisterPath, nullptr) << "Cannot resolve the NCCL unregistration symbol in " << pluginLibraryPath();

    // The registry only stores and compares these handles, so opaque non-null
    // values are enough to exercise validation and ownership.
    auto* const communicator = reinterpret_cast<void*>(0x1000);
    auto* const otherCommunicator = reinterpret_cast<void*>(0x2000);
    auto* const allReduceFunction = reinterpret_cast<void*>(0x3000);

    EXPECT_FALSE(registerPath(-1, communicator, allReduceFunction));
    EXPECT_FALSE(registerPath(0, nullptr, allReduceFunction));
    EXPECT_FALSE(registerPath(0, communicator, nullptr));

    ASSERT_TRUE(registerPath(0, communicator, allReduceFunction));
    EXPECT_FALSE(unregisterPath(0, otherCommunicator));
    EXPECT_FALSE(unregisterPath(1, communicator));
    EXPECT_FALSE(unregisterPath(0, nullptr));
    EXPECT_TRUE(unregisterPath(0, communicator));
    EXPECT_FALSE(unregisterPath(0, communicator));
}

TEST(AllReducePluginTest, NcclPathSumsAcrossTwoRanks)
{
    if (detectCudaDeviceCount() < kTpSize)
    {
        GTEST_SKIP() << "The NCCL AllReduce path needs " << kTpSize << " CUDA devices, found "
                     << detectCudaDeviceCount();
    }

    ASSERT_NE(loadPluginLibrary(), nullptr) << "Failed to load " << pluginLibraryPath() << ": " << dlerror();

    try
    {
        NcclCollectiveBackend::load();
    }
    catch (std::exception const& e)
    {
        GTEST_SKIP() << "NCCL runtime is unavailable: " << e.what();
    }

    auto resources = createTensorParallelNcclResources(kTpSize, {0, 1}, {0, 1}, {}, true);
    ASSERT_NE(resources, nullptr);
    ASSERT_TRUE(resources->registered());

    for (auto const dataType : {nvinfer1::DataType::kHALF, nvinfer1::DataType::kFLOAT, nvinfer1::DataType::kBF16})
    {
        SCOPED_TRACE("dataType=" + std::to_string(static_cast<int32_t>(dataType)));

        std::vector<RankOutcome> outcomes(kTpSize);
        std::vector<std::thread> workers;
        workers.reserve(kTpSize);
        for (int32_t rank = 0; rank < kTpSize; ++rank)
        {
            workers.emplace_back([rank, dataType, &outcomes] { runRank(rank, dataType, outcomes[rank]); });
        }
        for (auto& worker : workers)
        {
            worker.join();
        }

        for (int32_t rank = 0; rank < kTpSize; ++rank)
        {
            SCOPED_TRACE("rank=" + std::to_string(rank));
            ASSERT_TRUE(outcomes[rank].failure.empty()) << outcomes[rank].failure;
            ASSERT_EQ(outcomes[rank].output.size(), static_cast<size_t>(kNumElements));
            for (size_t index = 0; index < outcomes[rank].output.size(); ++index)
            {
                ASSERT_NEAR(outcomes[rank].output[index], 3.0F, kTolerance) << "Mismatch at element " << index;
            }
        }
    }
}
