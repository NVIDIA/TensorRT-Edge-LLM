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

#if defined(CUTE_DSL_W2A16_GEMM_ENABLED)
#include "common/cudaUtils.h"
#include "common/tensor.h"
#include "common/trtInterop.h"
#include "scratchDir.h"
#include "testPluginLoader.h"
#include <NvInfer.h>
#include <array>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <gtest/gtest.h>
#include <memory>
#include <vector>

namespace trt_edgellm::plugins
{
namespace
{
using namespace nvinfer1;
using rt::DeviceType;
using rt::Tensor;
using trt_edgellm::DataType;
constexpr int32_t kK = 4096;
constexpr int32_t kMAX_M = 8192;

struct EngineCase
{
    int32_t n, projection, rank;
};

struct Graph
{
    cudaGraph_t graph{};
    cudaGraphExec_t exec{};
    ~Graph()
    {
        if (exec)
        {
            cudaGraphExecDestroy(exec);
        }
        if (graph)
        {
            cudaGraphDestroy(graph);
        }
    }
};

std::ifstream openFixture(std::filesystem::path const& path, int32_t n, int32_t m)
{
    std::ifstream file(path, std::ios::binary);
    std::array<uint32_t, 4> header{};
    file.read(reinterpret_cast<char*>(header.data()), sizeof(header));
    ELLM_CHECK(
        file && header == (std::array<uint32_t, 4>{0x57324431, static_cast<uint32_t>(n), kK, static_cast<uint32_t>(m)}),
        "Missing or incompatible W2A16 fixture: " + path.string());
    return file;
}

void readTensor(std::ifstream& file, Tensor& tensor)
{
    file.read(static_cast<char*>(tensor.rawPointer()), tensor.getMemoryCapacity());
    ELLM_CHECK(file.good(), "Truncated W2A16 fixture");
}

class W2A16GemmEngineTest : public ::testing::TestWithParam<EngineCase>
{
protected:
    void SetUp() override
    {
        int device{};
        cudaDeviceProp props{};
        CUDA_CHECK(cudaGetDevice(&device));
        CUDA_CHECK(cudaGetDeviceProperties(&props, device));
        if (props.major != 12 || (props.minor != 0 && props.minor != 1)
            || (props.minor == 1 && props.multiProcessorCount != 48))
        {
            GTEST_SKIP() << "Requires SM120 or GB10";
        }
        auto const* root = std::getenv("EDGELLM_W2A16_GEMM_FIXTURES");
        if (!root)
        {
            GTEST_SKIP() << "Set EDGELLM_W2A16_GEMM_FIXTURES to provisioned fixtures";
        }
        mFixtures = std::filesystem::path(root) / "fixtures-boundary-full"
            / ("projection" + std::to_string(GetParam().projection));
        ELLM_CHECK(test::loadPluginLibrary(), "Cannot load Edge-LLM plugin library");
        mDir = makeScratchDir("w2a16GemmEngineTests");
        CUDA_CHECK(cudaStreamCreateWithFlags(&mStream, cudaStreamNonBlocking));
    }

    void TearDown() override
    {
        if (mStream)
        {
            CUDA_CHECK(cudaStreamSynchronize(mStream));
            CUDA_CHECK(cudaStreamDestroy(mStream));
        }
        if (!mDir.empty())
        {
            std::filesystem::remove_all(mDir);
        }
    }

    Dims inputShape(int32_t m) const
    {
        int32_t const batch = m > 1 && m % 2 == 0 ? 2 : 1;
        return GetParam().rank == 2 ? Dims{2, {m, kK}} : Dims{3, {batch, m / batch, kK}};
    }

    void saveEngine()
    {
        auto const c = GetParam();
        int32_t const packing = c.n % 128 == 0 ? 128 : 8;
        auto weights = openFixture(mFixtures / "weights.bin", c.n, 0);
        Tensor values({int64_t{c.n} * kK / 4}, DeviceType::kCPU, DataType::kINT8);
        Tensor bases({kK / 256, c.n}, DeviceType::kCPU, DataType::kHALF);
        Tensor lookup({256, 8}, DeviceType::kCPU, DataType::kBF16);
        for (auto* tensor : {&values, &bases, &lookup})
        {
            readTensor(weights, *tensor);
        }
        auto builder = std::unique_ptr<IBuilder>(createInferBuilder(TrtLogger::instance()));
        ELLM_CHECK(builder != nullptr, "Cannot create builder");
        auto network = std::unique_ptr<INetworkDefinition>(
            builder->createNetworkV2(1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kSTRONGLY_TYPED)));
        auto config = std::unique_ptr<IBuilderConfig>(builder->createBuilderConfig());
        ELLM_CHECK(network && config, "Cannot create network/config");
        auto* profile = builder->createOptimizationProfile();
        ELLM_CHECK(profile != nullptr, "Cannot create profile");
        auto const dynamicShape = c.rank == 2 ? Dims{2, {-1, kK}} : Dims{3, {-1, -1, kK}};
        auto* input = network->addInput("input", nvinfer1::DataType::kBF16, dynamicShape);
        ELLM_CHECK(input != nullptr, "Cannot create activation input");
        ELLM_CHECK(profile->setDimensions("input", OptProfileSelector::kMIN, inputShape(1))
                && profile->setDimensions("input", OptProfileSelector::kOPT, inputShape(16))
                && profile->setDimensions("input", OptProfileSelector::kMAX, inputShape(kMAX_M)),
            "Cannot set dynamic profile");
        ELLM_CHECK(config->addOptimizationProfile(profile) >= 0, "Cannot add dynamic profile");
        config->setBuilderOptimizationLevel(0);
        auto* creator = static_cast<IPluginCreatorV3One*>(getPluginRegistry()->getCreator("W2A16GemmPlugin", "1", ""));
        ELLM_CHECK(creator != nullptr, "Missing W2A16 creator");
        std::array<PluginField, 2> fields{PluginField{"gemm_n", &c.n, PluginFieldType::kINT32, 1},
            PluginField{"gemm_k", &kK, PluginFieldType::kINT32, 1}};
        PluginFieldCollection fc{static_cast<int32_t>(fields.size()), fields.data()};
        auto plugin = std::unique_ptr<IPluginV3>(creator->createPlugin("dense", &fc, TensorRTPhase::kBUILD));
        ELLM_CHECK(plugin != nullptr, "Cannot create W2A16 plugin");
        std::array<Dims, 3> const dims{
            Dims{4, {c.n / packing, kK / 256, packing, 64}}, Dims{2, {kK / 256, c.n}}, Dims{2, {256, 8}}};
        std::array<Tensor*, 3> tensors{&values, &bases, &lookup};
        std::array<ITensor*, 4> inputs{input};
        for (size_t i = 0; i < tensors.size(); ++i)
        {
            auto const& tensor = *tensors[i];
            Weights const weight{toTrt(tensor.getDataType()), tensor.rawPointer(), tensor.getShape().volume()};
            auto* layer = network->addConstant(dims[i], weight);
            ELLM_CHECK(layer != nullptr, "Cannot add packed weight constant");
            inputs[i + 1] = layer->getOutput(0);
        }
        auto* layer = network->addPluginV3(inputs.data(), inputs.size(), nullptr, 0, *plugin);
        ELLM_CHECK(layer != nullptr, "Cannot add W2A16 layer");
        layer->getOutput(0)->setName("output");
        network->markOutput(*layer->getOutput(0));
        auto plan = std::unique_ptr<IHostMemory>(builder->buildSerializedNetwork(*network, *config));
        ELLM_CHECK(plan != nullptr, "Cannot build W2A16 engine");
        std::ofstream file(mDir / "dense.engine", std::ios::binary);
        file.write(static_cast<char const*>(plan->data()), plan->size());
        ELLM_CHECK(file.good(), "Cannot save W2A16 engine");
    }

    cudaStream_t mStream{};
    std::filesystem::path mFixtures, mDir;
};

TEST_P(W2A16GemmEngineTest, ReloadDynamicShapesAndGraphReplay)
{
    // Given an SM120 or GB10 W2A16 single-layer engine, a dynamic profile and independent reference fixtures,
    // when the saved engine is reloaded and M changes, including split-K shapes, then inference and
    // CUDA Graph replay produce finite outputs with relative L2 <= 0.01 and cosine >= 0.999.
    saveEngine();
    std::ifstream file(mDir / "dense.engine", std::ios::binary | std::ios::ate);
    ELLM_CHECK(file && file.tellg() > 0, "Cannot read saved engine");
    std::vector<char> plan(static_cast<size_t>(file.tellg()));
    file.seekg(0);
    file.read(plan.data(), plan.size());
    ELLM_CHECK(file.good(), "Truncated saved engine");
    auto runtime = std::unique_ptr<IRuntime>(createInferRuntime(TrtLogger::instance()));
    ELLM_CHECK(runtime != nullptr, "Cannot create runtime");
    auto engine = std::unique_ptr<ICudaEngine>(runtime->deserializeCudaEngine(plan.data(), plan.size()));
    ELLM_CHECK(engine != nullptr, "Cannot reload W2A16 engine");
    auto context = std::unique_ptr<IExecutionContext>(engine->createExecutionContext());
    ELLM_CHECK(context != nullptr, "Cannot create W2A16 execution context");
    int32_t const n = GetParam().n;
    for (int32_t m : {kMAX_M, 9, 1, 17, 128, 9})
    {
        SCOPED_TRACE("M=" + std::to_string(m));
        Tensor input({m, kK}, DeviceType::kGPU, DataType::kBF16);
        Tensor output({m, n}, DeviceType::kGPU, DataType::kBF16);
        Tensor hostInput({m, kK}, DeviceType::kCPU, DataType::kBF16);
        Tensor expected({m, n}, DeviceType::kCPU, DataType::kBF16);
        Tensor actual({m, n}, DeviceType::kCPU, DataType::kBF16);
        ELLM_CHECK(context->setInputShape("input", inputShape(m)), "Cannot set runtime shape");
        ELLM_CHECK(context->setTensorAddress("input", input.rawPointer())
                && context->setTensorAddress("output", output.rawPointer()),
            "Cannot bind activation/output");
        auto inputs = openFixture(mFixtures / ("m" + std::to_string(m) + ".bin"), n, m);
        Graph graph;
        for (int32_t variant = 0; variant < 2; ++variant)
        {
            SCOPED_TRACE("input=" + std::to_string(variant));
            readTensor(inputs, hostInput);
            readTensor(inputs, expected);
            CUDA_CHECK(cudaMemcpyAsync(input.rawPointer(), hostInput.rawPointer(), input.getMemoryCapacity(),
                cudaMemcpyHostToDevice, mStream));
            for (bool replay : {false, true})
            {
                SCOPED_TRACE(replay ? "graph" : "enqueue");
                CUDA_CHECK(cudaMemsetAsync(output.rawPointer(), 0xff, output.getMemoryCapacity(), mStream));
                if (replay)
                {
                    CUDA_CHECK(cudaGraphLaunch(graph.exec, mStream));
                }
                else
                {
                    ELLM_CHECK(context->enqueueV3(mStream), "W2A16 enqueue failed");
                }
                CUDA_CHECK(cudaMemcpyAsync(actual.rawPointer(), output.rawPointer(), output.getMemoryCapacity(),
                    cudaMemcpyDeviceToHost, mStream));
                CUDA_CHECK(cudaStreamSynchronize(mStream));
                double diff = 0, aa = 0, bb = 0, ab = 0;
                bool finite = true;
                for (int64_t i = 0; i < int64_t{m} * n; ++i)
                {
                    double const a = __bfloat162float(actual.dataPointer<__nv_bfloat16>()[i]);
                    double const b = __bfloat162float(expected.dataPointer<__nv_bfloat16>()[i]);
                    finite = finite && std::isfinite(a);
                    diff += (a - b) * (a - b);
                    aa += a * a;
                    bb += b * b;
                    ab += a * b;
                }
                EXPECT_TRUE(finite);
                EXPECT_LE(std::sqrt(diff / bb), 0.01);
                EXPECT_GE(ab / std::sqrt(aa * bb), 0.999);
                if (!graph.exec)
                {
                    CUDA_CHECK(cudaStreamBeginCapture(mStream, cudaStreamCaptureModeThreadLocal));
                    bool const captured = context->enqueueV3(mStream);
                    CUDA_CHECK(cudaStreamEndCapture(mStream, &graph.graph));
                    ELLM_CHECK(captured, "W2A16 capture failed");
                    CUDA_CHECK(instantiateCudaGraph(&graph.exec, graph.graph));
                }
            }
        }
    }
}
INSTANTIATE_TEST_SUITE_P(PackingAndRanks, W2A16GemmEngineTest,
    ::testing::Values(EngineCase{128, 2, 2}, EngineCase{128, 2, 3}, EngineCase{136, 5, 2}, EngineCase{136, 5, 3}));
} // namespace
} // namespace trt_edgellm::plugins
#endif
