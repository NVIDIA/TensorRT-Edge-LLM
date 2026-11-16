/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#if defined(CUTE_DSL_SSD_SM12X_ENABLED)

#include "common/cudaUtils.h"
#include "common/tensor.h"
#include "common/trtInterop.h"
#include "testPluginLoader.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <gtest/gtest.h>
#include <memory>
#include <numeric>
#include <random>
#include <tuple>
#include <vector>

using namespace trt_edgellm;

namespace
{
struct PackedPluginCudaResources
{
    cudaStream_t stream{};
    cudaGraph_t graph{};
    cudaGraphExec_t executable{};

    ~PackedPluginCudaResources()
    {
        if (executable != nullptr)
        {
            cudaGraphExecDestroy(executable);
        }
        if (graph != nullptr)
        {
            cudaGraphDestroy(graph);
        }
        if (stream != nullptr)
        {
            cudaStreamDestroy(stream);
        }
    }
};

using PackedPluginLengths = std::array<std::array<int32_t, 7>, 2>;

struct PackedPlugin
{
    std::unique_ptr<nvinfer1::IPluginV3> plugin;
    nvinfer1::IPluginV3OneBuild* build{};
    nvinfer1::IPluginV3OneRuntime* runtime{};
    std::array<nvinfer1::DynamicPluginTensorDesc, 16> dynamic{};
    std::array<nvinfer1::PluginTensorDesc, 16> input{};
    std::array<nvinfer1::PluginTensorDesc, 2> output{};
    int32_t inputs{};

    PackedPlugin(int32_t tokens, int32_t batch, int32_t prefill, int32_t heads, int32_t dim, int32_t stateDim,
        int32_t groups, int32_t slots, int32_t softplus = 1, int32_t verify = 0)
    {
        using namespace nvinfer1;
        check::check(test::loadPluginLibrary() != nullptr, "Plugin library must load");
        auto* creator = static_cast<IPluginCreatorV3One*>(getPluginRegistry()->getCreator("update_ssm_state", "1", ""));
        check::check(creator != nullptr, "Mamba plugin creator must be registered");
        std::array<PluginField, 2> fields{{{"dt_softplus", &softplus, PluginFieldType::kINT32, 1},
            {"use_spec_verify_state", &verify, PluginFieldType::kINT32, 1}}};
        PluginFieldCollection collection{static_cast<int32_t>(fields.size()), fields.data()};
        plugin.reset(creator->createPlugin("packed_ssd_test", &collection, TensorRTPhase::kRUNTIME));
        check::check(plugin != nullptr, "Mamba plugin must be created");
        build = static_cast<IPluginV3OneBuild*>(plugin->getCapabilityInterface(PluginCapabilityType::kBUILD));
        runtime = static_cast<IPluginV3OneRuntime*>(plugin->getCapabilityInterface(PluginCapabilityType::kRUNTIME));
        auto desc = [&](int32_t index, std::initializer_list<int64_t> shape,
                        nvinfer1::DataType type = nvinfer1::DataType::kHALF) {
            auto& value = input[index];
            value.dims.nbDims = static_cast<int32_t>(shape.size());
            std::copy(shape.begin(), shape.end(), value.dims.d);
            value.type = type;
            value.format = TensorFormat::kLINEAR;
            dynamic[index].desc = value;
            dynamic[index].min = value.dims;
            dynamic[index].max = value.dims;
        };
        desc(0, {tokens, heads, dim});
        desc(1, {heads}, nvinfer1::DataType::kFLOAT);
        desc(2, {tokens, groups, stateDim});
        desc(3, {tokens, groups, stateDim});
        desc(4, {heads});
        desc(5, {tokens, heads});
        desc(6, {heads});
        desc(7, {slots, heads, dim, stateDim});
        desc(8, {batch}, nvinfer1::DataType::kINT32);
        desc(9, {batch + 1}, nvinfer1::DataType::kINT32);
        desc(10, {batch}, nvinfer1::DataType::kINT32);
        desc(11, {8}, nvinfer1::DataType::kINT32);
        desc(12, {prefill}, nvinfer1::DataType::kINT32);
        desc(13, {tokens}, nvinfer1::DataType::kINT32);
        desc(14, {tokens}, nvinfer1::DataType::kINT32);
        desc(15, {0}, nvinfer1::DataType::kINT32);
        inputs = verify ? 16 : 13;
        output = {input[0], input[7]};
        check::check(build->configurePlugin(dynamic.data(), inputs, nullptr, verify ? 6 : 2) == 0,
            "Mamba plugin configuration must succeed");
    }

    size_t workspaceSize() const
    {
        return build->getWorkspaceSize(dynamic.data(), inputs, nullptr, inputs == 16 ? 6 : 2);
    }
};

class PackedSsdPluginTest : public testing::TestWithParam<std::tuple<std::array<int32_t, 2>, int32_t>>
{
};

TEST_P(PackedSsdPluginTest, RaggedStatePoolAndGraphReplay)
{
    // Given SM12x FP16 unequal packed prefill lengths, empty sequences, reordered nonzero state slots, and decode;
    // when MambaPlugin enqueues and replays a CUDA Graph with refreshed metadata;
    // then output/final state match an independent FP32 recurrence and unused/empty-sequence states stay unchanged.
    auto const& [shape, prefill] = GetParam();
    int32_t const dim = shape[0];
    int32_t const dstate = shape[1];
    if (getSMVersion() != 120 && getSMVersion() != 121)
    {
        GTEST_SKIP() << "Packed SSD plugin requires SM120/121";
    }
    constexpr int32_t heads{4}, groups{2}, slots{9}, batch{7};
    PackedPluginLengths lengths{};
    if (prefill == 0)
    {
        lengths = {{{1, 1, 1, 1, 1, 1, 1}, {1, 1, 1, 1, 1, 1, 1}}};
    }
    else if (prefill == batch)
    {
        lengths = {{{0, 129, 65, 511, 1, 0, 128}, {128, 0, 511, 1, 65, 129, 0}}};
    }
    else
    {
        lengths = {{{0, 129, 65, 511, 1, 1, 1}, {511, 1, 0, 129, 65, 1, 1}}};
    }
    int32_t const tokens = std::accumulate(lengths[0].begin(), lengths[0].end(), int32_t{0});
    PackedPlugin plugin(tokens, batch, prefill, heads, dim, dstate, groups, slots);
    size_t const stateSize = static_cast<size_t>(slots) * heads * dim * dstate;
    std::mt19937 random(1280);
    std::normal_distribution<float> normal(0.F, 0.25F);
    auto randomHalf = [&](size_t count) {
        std::vector<half> values(count);
        for (auto& value : values)
        {
            value = __float2half(normal(random));
        }
        return values;
    };
    auto x = randomHalf(static_cast<size_t>(tokens) * heads * dim);
    auto dt = randomHalf(tokens * heads);
    auto b = randomHalf(tokens * groups * dstate);
    auto c = randomHalf(b.size());
    auto d = randomHalf(heads);
    auto initial = randomHalf(stateSize);
    std::vector<half> bias(heads, __float2half(-2.F));
    std::vector<float> a{-0.2F, -0.4F, -0.7F, -1.F};
    std::vector<int32_t> cu(batch + 1, 0);
    std::partial_sum(lengths[0].begin(), lengths[0].end(), cu.begin() + 1);
    std::vector<int32_t> indices{5, 1, 7, 0, 6, 4, 2};
    PackedPluginCudaResources resources;
    CUDA_CHECK(cudaStreamCreateWithFlags(&resources.stream, cudaStreamNonBlocking));
    auto const stream = resources.stream;
    std::vector<rt::Tensor> allocations;
    allocations.reserve(12);
    std::vector<rt::Tensor> hostAllocations;
    auto host = [&](void const* source, size_t bytes) {
        hostAllocations.emplace_back(
            rt::Coords{static_cast<int64_t>(bytes)}, rt::DeviceType::kCPU, trt_edgellm::DataType::kINT8);
        void* result = hostAllocations.back().rawPointer();
        if (source != nullptr)
        {
            std::memcpy(result, source, bytes);
        }
        return result;
    };
    auto device = [&](void const* source, size_t bytes) {
        allocations.emplace_back(
            rt::Coords{static_cast<int64_t>(bytes)}, rt::DeviceType::kGPU, trt_edgellm::DataType::kINT8);
        void* result = allocations.back().rawPointer();
        if (source != nullptr)
        {
            CUDA_CHECK(cudaMemcpyAsync(result, host(source, bytes), bytes, cudaMemcpyHostToDevice, stream));
        }
        return result;
    };
    std::array<void const*, 16> inputs{};
    inputs[0] = device(x.data(), x.size() * sizeof(half));
    inputs[1] = device(a.data(), a.size() * sizeof(float));
    inputs[2] = device(b.data(), b.size() * sizeof(half));
    inputs[3] = device(c.data(), c.size() * sizeof(half));
    inputs[4] = device(d.data(), d.size() * sizeof(half));
    inputs[5] = device(dt.data(), dt.size() * sizeof(half));
    inputs[6] = device(bias.data(), bias.size() * sizeof(half));
    void* state = device(initial.data(), stateSize * sizeof(half));
    inputs[7] = state;
    void* queryLengths = device(lengths[0].data(), batch * sizeof(int32_t));
    inputs[8] = queryLengths;
    void* offsets = device(cu.data(), cu.size() * sizeof(int32_t));
    inputs[9] = offsets;
    void* stateIndices = device(indices.data(), indices.size() * sizeof(int32_t));
    inputs[10] = stateIndices;
    std::array<void*, 2> outputs{device(nullptr, x.size() * sizeof(half)), state};
    void* workspace = device(nullptr, plugin.workspaceSize());
    auto enqueue = [&] {
        return plugin.runtime->enqueue(
            plugin.input.data(), plugin.output.data(), inputs.data(), outputs.data(), workspace, stream);
    };
    check::check(enqueue() == 0, "Plugin warmup must succeed before graph capture");
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    int32_t const captureStatus = enqueue();
    CUDA_CHECK(cudaStreamEndCapture(stream, &resources.graph));
    ASSERT_EQ(captureStatus, 0);
    CUDA_CHECK(instantiateCudaGraph(&resources.executable, resources.graph));

    for (int32_t replay = 0; replay < 2; ++replay)
    {
        SCOPED_TRACE(testing::Message() << "replay " << replay << ", total tokens " << tokens);
        std::partial_sum(lengths[replay].begin(), lengths[replay].end(), cu.begin() + 1);
        if (replay == 1)
        {
            std::rotate(indices.begin(), indices.begin() + 2, indices.end());
        }
        CUDA_CHECK(cudaMemcpyAsync(queryLengths, host(lengths[replay].data(), batch * sizeof(int32_t)),
            batch * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(offsets, host(cu.data(), cu.size() * sizeof(int32_t)), cu.size() * sizeof(int32_t),
            cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(stateIndices, host(indices.data(), indices.size() * sizeof(int32_t)),
            indices.size() * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(state, host(initial.data(), stateSize * sizeof(half)), stateSize * sizeof(half),
            cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemsetAsync(outputs[0], 0xFF, x.size() * sizeof(half), stream));
        CUDA_CHECK(cudaGraphLaunch(resources.executable, stream));
        auto* output = static_cast<half*>(host(nullptr, x.size() * sizeof(half)));
        auto* actual = static_cast<half*>(host(nullptr, stateSize * sizeof(half)));
        CUDA_CHECK(cudaMemcpyAsync(output, outputs[0], x.size() * sizeof(half), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(actual, state, stateSize * sizeof(half), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        std::vector<float> expectedState(stateSize), expectedOutput(x.size());
        std::transform(
            initial.begin(), initial.end(), expectedState.begin(), [](half value) { return __half2float(value); });
        std::array<bool, slots> touched{};
        for (int32_t seq = 0; seq < batch; ++seq)
        {
            int32_t const slot = indices[seq];
            touched[slot] = cu[seq] < cu[seq + 1];
            for (int32_t token = cu[seq]; token < cu[seq + 1]; ++token)
            {
                for (int32_t h = 0; h < heads; ++h)
                {
                    float const delta
                        = std::log1p(std::exp(__half2float(dt[token * heads + h]) + __half2float(bias[h])));
                    float const keep = std::exp(a[h] * delta);
                    for (int32_t channel = 0; channel < dim; ++channel)
                    {
                        size_t const xi = (static_cast<size_t>(token) * heads + h) * dim + channel;
                        float y = __half2float(d[h]) * __half2float(x[xi]);
                        for (int32_t n = 0; n < dstate; ++n)
                        {
                            size_t const si = ((static_cast<size_t>(slot) * heads + h) * dim + channel) * dstate + n;
                            size_t const bi = (static_cast<size_t>(token) * groups + h / (heads / groups)) * dstate + n;
                            expectedState[si]
                                = keep * expectedState[si] + delta * __half2float(x[xi]) * __half2float(b[bi]);
                            y += expectedState[si] * __half2float(c[bi]);
                        }
                        expectedOutput[xi] = y;
                    }
                }
            }
        }
        for (size_t i = 0; i < x.size(); ++i)
        {
            ASSERT_NEAR(__half2float(output[i]), expectedOutput[i], 0.004F + 0.005F * std::abs(expectedOutput[i]))
                << "output " << i;
        }
        for (size_t i = 0; i < stateSize; ++i)
        {
            if (touched[i / (heads * dim * dstate)])
            {
                ASSERT_NEAR(__half2float(actual[i]), expectedState[i], 0.002F + 0.005F * std::abs(expectedState[i]))
                    << "state " << i;
            }
            else
            {
                ASSERT_EQ(std::memcmp(&actual[i], &initial[i], sizeof(half)), 0) << "inactive state " << i;
            }
        }
    }
}

INSTANTIATE_TEST_SUITE_P(Sm12x, PackedSsdPluginTest,
    testing::Combine(
        testing::Values(std::array<int32_t, 2>{64, 64}, std::array<int32_t, 2>{64, 128}, std::array<int32_t, 2>{80, 64},
            std::array<int32_t, 2>{80, 128}, std::array<int32_t, 2>{128, 64}, std::array<int32_t, 2>{128, 128}),
        testing::Values(0, 5, 7)));

class UnsupportedPackedSsdPluginTest : public testing::TestWithParam<std::array<int32_t, 4>>
{
};

TEST_P(UnsupportedPackedSsdPluginTest, RejectsBeforeChangingState)
{
    // Given an SM12x packed layer with unsupported dimensions or speculative verification;
    // when MambaPlugin receives a packed prefill enqueue;
    // then it rejects the invocation and preserves the resident state.
    if (getSMVersion() != 120 && getSMVersion() != 121)
    {
        GTEST_SKIP() << "SM12x rejection contract";
    }
    auto const& config = GetParam();
    constexpr int32_t tokens{129}, heads{4}, groups{2}, slots{3};
    PackedPlugin plugin(tokens, 1, 1, heads, config[0], config[1], groups, slots, config[2], config[3]);
    PackedPluginCudaResources resources;
    CUDA_CHECK(cudaStreamCreateWithFlags(&resources.stream, cudaStreamNonBlocking));
    auto const stream = resources.stream;
    std::vector<rt::Tensor> allocations;
    std::array<void const*, 16> inputs{};
    for (int32_t i = 0; i < plugin.inputs; ++i)
    {
        if (rt::toCoords(plugin.input[i].dims).volume() == 0)
        {
            continue;
        }
        allocations.emplace_back(
            rt::toCoords(plugin.input[i].dims), rt::DeviceType::kGPU, fromTrt(plugin.input[i].type));
        inputs[i] = allocations.back().rawPointer();
        CUDA_CHECK(cudaMemsetAsync(allocations.back().rawPointer(), 0, allocations.back().getMemoryCapacity(), stream));
    }
    rt::Tensor metadata(rt::Coords{4}, rt::DeviceType::kCPU, trt_edgellm::DataType::kINT32);
    auto* metadataPtr = static_cast<int32_t*>(metadata.rawPointer());
    metadataPtr[0] = tokens;
    metadataPtr[1] = 0;
    metadataPtr[2] = tokens;
    metadataPtr[3] = 1;
    CUDA_CHECK(
        cudaMemcpyAsync(const_cast<void*>(inputs[8]), metadataPtr, sizeof(int32_t), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(
        const_cast<void*>(inputs[9]), metadataPtr + 1, 2 * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(
        const_cast<void*>(inputs[10]), metadataPtr + 3, sizeof(int32_t), cudaMemcpyHostToDevice, stream));
    size_t const stateBytes = static_cast<size_t>(slots) * heads * config[0] * config[1] * sizeof(half);
    void* state = const_cast<void*>(inputs[7]);
    CUDA_CHECK(cudaMemsetAsync(state, 0x3C, stateBytes, stream));
    rt::Tensor output(rt::Coords{tokens, heads, config[0]}, rt::DeviceType::kGPU, trt_edgellm::DataType::kHALF);
    rt::Tensor workspace(
        rt::Coords{static_cast<int64_t>(plugin.workspaceSize())}, rt::DeviceType::kGPU, trt_edgellm::DataType::kINT8);
    std::array<void*, 6> outputs{output.rawPointer(), state};
    EXPECT_NE(plugin.runtime->enqueue(plugin.input.data(), plugin.output.data(), inputs.data(), outputs.data(),
                  workspace.rawPointer(), stream),
        0);
    rt::Tensor actual(rt::Coords{static_cast<int64_t>(stateBytes)}, rt::DeviceType::kCPU, trt_edgellm::DataType::kINT8);
    auto* actualPtr = static_cast<uint8_t*>(actual.rawPointer());
    CUDA_CHECK(cudaMemcpyAsync(actualPtr, state, stateBytes, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    EXPECT_TRUE(std::all_of(actualPtr, actualPtr + stateBytes, [](uint8_t value) { return value == 0x3C; }));
}

INSTANTIATE_TEST_SUITE_P(Sm12x, UnsupportedPackedSsdPluginTest,
    testing::Values(std::array<int32_t, 4>{32, 128, 1, 0}, std::array<int32_t, 4>{64, 32, 1, 0},
        std::array<int32_t, 4>{64, 128, 1, 1}));
} // namespace
#endif // CUTE_DSL_SSD_SM12X_ENABLED
