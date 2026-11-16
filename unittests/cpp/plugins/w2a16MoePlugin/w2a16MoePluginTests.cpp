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

//! W2A16MoePlugin: a TensorRT build/serialize/deserialize/run round trip.

#if defined(CUTE_DSL_W2A16_MOE_ENABLED)

#include "common/cudaUtils.h"
#include "common/logger.h"
#include "common/trtInterop.h"
#include "kernels/moe/w2a16_cutedsl/cuteDslW2A16MoeRunner.h"
#include "testPluginLoader.h"

#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <cuda_bf16.h>
#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

namespace trt_edgellm::plugins
{
namespace
{

using namespace nvinfer1;
// Plugin I/O descriptors use TensorRT types, not trt_edgellm::DataType.
using nvinfer1::DataType;

constexpr char const* kName{"W2A16MoePlugin"};
constexpr char const* kVersion{"1"};
constexpr uint32_t kFixtureMagic = 0x57324131;

IPluginCreatorV3One* getCreator()
{
    if (test::loadPluginLibrary() == nullptr)
    {
        return nullptr;
    }
    auto* creator = getPluginRegistry()->getCreator(kName, kVersion, "");
    return creator == nullptr ? nullptr : static_cast<IPluginCreatorV3One*>(creator);
}

struct Fields
{
    int32_t experts{240};
    int32_t inter{1536};
    std::vector<PluginField> list;
    PluginFieldCollection collection{};

    Fields()
    {
        list.emplace_back("num_experts", &experts, PluginFieldType::kINT32, 1);
        list.emplace_back("moe_inter_size", &inter, PluginFieldType::kINT32, 1);
        collection = {static_cast<int32_t>(list.size()), list.data()};
    }
};

class W2A16MoePluginTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        if (getCreator() == nullptr
            || !CuteDslW2A16MoeRunner::canImplement(1, 240, 1536, getSMVersion(), getDeviceMultiProcessorCount()))
        {
            GTEST_SKIP() << "W2A16MoePlugin creator or W2A16 kernel unavailable";
        }
    }
};

} // namespace

namespace
{
template <typename T>
std::vector<T> readVector(std::ifstream& file, size_t count)
{
    std::vector<T> values(count);
    file.read(reinterpret_cast<char*>(values.data()), count * sizeof(T));
    if (!file)
    {
        throw std::runtime_error("Truncated W2A16 fixture");
    }
    return values;
}

std::ifstream openFixture(std::filesystem::path const& path, uint32_t experts, uint32_t inter, uint32_t tokens)
{
    std::ifstream file(path, std::ios::binary);
    std::array<uint32_t, 4> header{};
    file.read(reinterpret_cast<char*>(header.data()), sizeof(header));
    if (!file || header != std::array<uint32_t, 4>{kFixtureMagic, experts, inter, tokens})
    {
        throw std::runtime_error("Missing or incompatible W2A16 fixture: " + path.string());
    }
    return file;
}

struct DeviceBuffer
{
    void* ptr{};
    explicit DeviceBuffer(size_t bytes)
    {
        CUDA_CHECK(cudaMalloc(&ptr, bytes));
    }
    template <typename T>
    explicit DeviceBuffer(std::vector<T> const& host)
        : DeviceBuffer(host.size() * sizeof(T))
    {
        CUDA_CHECK(cudaMemcpy(ptr, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice));
    }
    ~DeviceBuffer()
    {
        cudaFree(ptr);
    }
    DeviceBuffer(DeviceBuffer const&) = delete;
    DeviceBuffer& operator=(DeviceBuffer const&) = delete;
};
} // namespace

// Given an engine built from a W2A16MoePlugin node (Bolt layer 3, E=240, I=1536, 1-16 tokens) and serialized,
// When it is deserialized and run for 1, 8 and 12 tokens with checkpoint fixtures through the workspace TensorRT
// reserved,
// Then each output matches the independent checkpoint oracle.
TEST_F(W2A16MoePluginTest, EngineRoundTripMatchesOracle)
{
    char const* root = std::getenv("EDGE_LLM_W2A16_PER_LAYER_TEST_DATA");
    if (root == nullptr)
    {
        GTEST_SKIP() << "Set EDGE_LLM_W2A16_PER_LAYER_TEST_DATA to per-layer checkpoint fixtures";
    }
    int32_t constexpr kE = 240, kI = 1536, kMaxTokens = 16;
    auto const dir = std::filesystem::path(root) / "layer3";
    int64_t const qWords = int64_t{kE} * kI * 1024 / 16;
    int64_t const sWords = int64_t{kE} * kI * 1024 / 512;

    auto builder = std::unique_ptr<IBuilder>(createInferBuilder(TrtLogger::instance()));
    auto network = std::unique_ptr<INetworkDefinition>(
        builder->createNetworkV2(1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kSTRONGLY_TYPED)));
    auto config = std::unique_ptr<IBuilderConfig>(builder->createBuilderConfig());
    ASSERT_TRUE(builder && network && config);
    std::array<ITensor*, 9> inputs{network->addInput("x", DataType::kBF16, Dims2{-1, 1024}),
        network->addInput("q1", DataType::kINT32, Dims{1, {qWords}}),
        network->addInput("q2", DataType::kINT32, Dims{1, {qWords}}),
        network->addInput("s1", DataType::kINT32, Dims{1, {sWords}}),
        network->addInput("s2", DataType::kINT32, Dims{1, {sWords}}),
        network->addInput("unit", DataType::kFLOAT, Dims{1, {kE}}),
        network->addInput("lut", DataType::kINT8, Dims{1, {2048}}),
        network->addInput("ids", DataType::kINT32, Dims2{-1, 9}),
        network->addInput("weights", DataType::kFLOAT, Dims2{-1, 9})};
    Fields onnx;
    std::unique_ptr<IPluginV3> plugin(getCreator()->createPlugin("moe", &onnx.collection, TensorRTPhase::kBUILD));
    ASSERT_NE(plugin, nullptr);
    auto* layer = network->addPluginV3(inputs.data(), 9, nullptr, 0, *plugin);
    ASSERT_NE(layer, nullptr);
    layer->getOutput(0)->setName("y");
    network->markOutput(*layer->getOutput(0));
    auto* profile = builder->createOptimizationProfile();
    for (char const* name : {"x", "ids", "weights"})
    {
        int64_t const width = std::strcmp(name, "x") == 0 ? 1024 : 9;
        profile->setDimensions(name, OptProfileSelector::kMIN, Dims2{1, width});
        profile->setDimensions(name, OptProfileSelector::kOPT, Dims2{8, width});
        profile->setDimensions(name, OptProfileSelector::kMAX, Dims2{kMaxTokens, width});
    }
    config->addOptimizationProfile(profile);
    auto plan = std::unique_ptr<IHostMemory>(builder->buildSerializedNetwork(*network, *config));
    ASSERT_NE(plan, nullptr);

    auto runtime = std::unique_ptr<IRuntime>(createInferRuntime(TrtLogger::instance()));
    auto engine = std::unique_ptr<ICudaEngine>(runtime->deserializeCudaEngine(plan->data(), plan->size()));
    ASSERT_NE(engine, nullptr);
    auto context = std::unique_ptr<IExecutionContext>(engine->createExecutionContext());
    ASSERT_NE(context, nullptr);

    auto weightsFile = openFixture(dir / "weights.bin", kE, kI, 0);
    DeviceBuffer q1(readVector<int32_t>(weightsFile, qWords));
    DeviceBuffer s1(readVector<int32_t>(weightsFile, sWords));
    DeviceBuffer q2(readVector<int32_t>(weightsFile, qWords));
    DeviceBuffer s2(readVector<int32_t>(weightsFile, sWords));
    DeviceBuffer lut(readVector<int8_t>(weightsFile, 2048));
    DeviceBuffer unit(readVector<float>(weightsFile, kE));
    cudaStream_t stream{};
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    for (int32_t const tokens : {1, 8, 12})
    {
        auto file = openFixture(dir / ("m" + std::to_string(tokens) + "_hot.bin"), kE, kI, tokens);
        DeviceBuffer x(readVector<__nv_bfloat16>(file, int64_t{tokens} * 1024));
        DeviceBuffer ids(readVector<int32_t>(file, int64_t{tokens} * 9));
        DeviceBuffer routing(readVector<float>(file, int64_t{tokens} * 9));
        auto const expected = readVector<__nv_bfloat16>(file, int64_t{tokens} * 1024);
        DeviceBuffer y(int64_t{tokens} * 1024 * sizeof(__nv_bfloat16));
        CUDA_CHECK(cudaMemset(y.ptr, 0xff, int64_t{tokens} * 1024 * sizeof(__nv_bfloat16)));
        ASSERT_TRUE(context->setInputShape("x", Dims2{tokens, 1024}));
        ASSERT_TRUE(context->setInputShape("ids", Dims2{tokens, 9}));
        ASSERT_TRUE(context->setInputShape("weights", Dims2{tokens, 9}));
        for (auto const& [name, buffer] : std::vector<std::pair<char const*, void*>>{{"x", x.ptr}, {"q1", q1.ptr},
                 {"q2", q2.ptr}, {"s1", s1.ptr}, {"s2", s2.ptr}, {"unit", unit.ptr}, {"lut", lut.ptr}, {"ids", ids.ptr},
                 {"weights", routing.ptr}, {"y", y.ptr}})
        {
            ASSERT_TRUE(context->setTensorAddress(name, buffer));
        }
        ASSERT_TRUE(context->enqueueV3(stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        std::vector<__nv_bfloat16> actual(int64_t{tokens} * 1024);
        CUDA_CHECK(cudaMemcpy(actual.data(), y.ptr, actual.size() * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
        double diff = 0, norm = 0;
        for (size_t i = 0; i < actual.size(); ++i)
        {
            double const a = __bfloat162float(actual[i]);
            double const b = __bfloat162float(expected[i]);
            diff += (a - b) * (a - b);
            norm += b * b;
        }
        EXPECT_LE(std::sqrt(diff / norm), 0.01) << "tokens=" << tokens;
    }
    CUDA_CHECK(cudaStreamDestroy(stream));
}

} // namespace trt_edgellm::plugins

#endif // CUTE_DSL_W2A16_MOE_ENABLED
