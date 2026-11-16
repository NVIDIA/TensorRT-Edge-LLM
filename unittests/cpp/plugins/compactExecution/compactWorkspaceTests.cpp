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

// The compact workspace a recurrent plugin reserves at build time is enough for a real compact step: each test
// enqueues one with exactly the reserved bytes followed by guard bytes.

#include "common/cudaUtils.h"
#include "kernels/compactPhase/compactPhaseSupport.h"
#include "testPluginLoader.h"

#include <NvInfer.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <memory>
#include <random>
#include <vector>

namespace trt_edgellm::plugins
{
namespace
{
using namespace nvinfer1;
using DataType = nvinfer1::DataType;

constexpr int32_t kPoolRows{4};
constexpr size_t kGuardBytes{64 * 1024};
constexpr uint8_t kGuardByte{0xA5};
//! The execution-phase marker extent of a compact step (rt::ExecutionPhase::kMixedPrefillDecode).
constexpr int64_t kCompactPhase{8};

template <typename T>
struct DeviceBuffer
{
    explicit DeviceBuffer(size_t count)
        : count(count)
    {
        EXPECT_EQ(cudaMalloc(&data, count * sizeof(T)), cudaSuccess);
    }
    DeviceBuffer(DeviceBuffer const&) = delete;
    DeviceBuffer& operator=(DeviceBuffer const&) = delete;
    ~DeviceBuffer()
    {
        cudaFree(data);
    }
    void upload(std::vector<T> const& host)
    {
        ASSERT_EQ(host.size(), count);
        ASSERT_EQ(cudaMemcpy(data, host.data(), count * sizeof(T), cudaMemcpyHostToDevice), cudaSuccess);
    }
    std::vector<T> download() const
    {
        std::vector<T> host(count);
        EXPECT_EQ(cudaMemcpy(host.data(), data, count * sizeof(T), cudaMemcpyDeviceToHost), cudaSuccess);
        return host;
    }
    T* data{};
    size_t count{};
};

template <typename T>
std::vector<T> randomValues(size_t count, std::mt19937& gen, float mean, float scale)
{
    std::normal_distribution<float> dist(mean, scale);
    std::vector<T> values(count);
    for (T& value : values)
    {
        value = static_cast<T>(dist(gen));
    }
    return values;
}

PluginTensorDesc descOf(DataType type, std::initializer_list<int64_t> dims)
{
    PluginTensorDesc desc{};
    desc.format = TensorFormat::kLINEAR;
    desc.type = type;
    desc.dims.nbDims = static_cast<int32_t>(dims.size());
    int32_t axis = 0;
    for (int64_t const dim : dims)
    {
        desc.dims.d[axis++] = dim;
    }
    return desc;
}

std::vector<DynamicPluginTensorDesc> asProfileMax(std::vector<PluginTensorDesc> const& maxDescs)
{
    std::vector<DynamicPluginTensorDesc> dynamic(maxDescs.size());
    for (size_t i = 0; i < maxDescs.size(); ++i)
    {
        dynamic[i].desc = maxDescs[i];
        dynamic[i].min = maxDescs[i].dims;
        dynamic[i].max = maxDescs[i].dims;
    }
    return dynamic;
}

//! One plugin layer through TensorRT's lifecycle: the build plugin is configured for the profile's max shapes and
//! queried for its workspace; the runtime plugin is created from the fields the build plugin serializes, and the copy
//! its attachToContext() returns enqueues.
class PluginLayer
{
public:
    PluginLayer(char const* pluginName, std::vector<PluginField> const& fields,
        std::vector<PluginTensorDesc> const& maxInputs, std::vector<PluginTensorDesc> const& maxOutputs)
    {
        auto* creator = static_cast<IPluginCreatorV3One*>(getPluginRegistry()->getCreator(pluginName, "1", ""));
        EXPECT_NE(creator, nullptr) << pluginName;
        if (creator == nullptr)
        {
            return;
        }
        PluginFieldCollection const fc{static_cast<int32_t>(fields.size()), fields.data()};
        std::unique_ptr<IPluginV3> const buildPlugin(creator->createPlugin(pluginName, &fc, TensorRTPhase::kBUILD));
        EXPECT_NE(buildPlugin, nullptr) << pluginName;
        if (buildPlugin == nullptr)
        {
            return;
        }
        auto* build
            = static_cast<IPluginV3OneBuild*>(buildPlugin->getCapabilityInterface(PluginCapabilityType::kBUILD));
        std::vector<DynamicPluginTensorDesc> const in = asProfileMax(maxInputs);
        std::vector<DynamicPluginTensorDesc> const out = asProfileMax(maxOutputs);
        int32_t const nbIn = static_cast<int32_t>(in.size());
        int32_t const nbOut = static_cast<int32_t>(out.size());
        EXPECT_EQ(build->configurePlugin(in.data(), nbIn, out.data(), nbOut), 0);
        mWorkspaceBytes = build->getWorkspaceSize(in.data(), nbIn, out.data(), nbOut);

        auto* buildRuntime
            = static_cast<IPluginV3OneRuntime*>(buildPlugin->getCapabilityInterface(PluginCapabilityType::kRUNTIME));
        std::unique_ptr<IPluginV3> const deserialized(
            creator->createPlugin("", buildRuntime->getFieldsToSerialize(), TensorRTPhase::kRUNTIME));
        EXPECT_NE(deserialized, nullptr) << pluginName;
        if (deserialized != nullptr)
        {
            auto* runtime = static_cast<IPluginV3OneRuntime*>(
                deserialized->getCapabilityInterface(PluginCapabilityType::kRUNTIME));
            mAttached.reset(runtime->attachToContext(nullptr));
        }
    }

    size_t workspaceBytes() const
    {
        return mWorkspaceBytes;
    }

    struct StepResult
    {
        int32_t status{-1};
        bool guardIntact{false};
    };

    //! Enqueue one step with exactly workspaceBytes() of workspace followed by guard bytes.
    StepResult enqueue(std::vector<PluginTensorDesc> const& inputs, std::vector<PluginTensorDesc> const& outputs,
        std::vector<void const*> const& inputData, std::vector<void*> const& outputData)
    {
        StepResult result;
        EXPECT_NE(mAttached, nullptr);
        if (mAttached == nullptr)
        {
            return result;
        }
        auto* runtime
            = static_cast<IPluginV3OneRuntime*>(mAttached->getCapabilityInterface(PluginCapabilityType::kRUNTIME));
        EXPECT_EQ(runtime->onShapeChange(inputs.data(), static_cast<int32_t>(inputs.size()), outputs.data(),
                      static_cast<int32_t>(outputs.size())),
            0);
        DeviceBuffer<uint8_t> workspace(mWorkspaceBytes + kGuardBytes);
        EXPECT_EQ(cudaMemset(workspace.data, kGuardByte, workspace.count), cudaSuccess);
        cudaStream_t stream{};
        EXPECT_EQ(cudaStreamCreate(&stream), cudaSuccess);
        result.status = runtime->enqueue(
            inputs.data(), outputs.data(), inputData.data(), outputData.data(), workspace.data, stream);
        EXPECT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
        EXPECT_EQ(cudaStreamDestroy(stream), cudaSuccess);
        std::vector<uint8_t> const after = workspace.download();
        result.guardIntact = true;
        for (size_t i = mWorkspaceBytes; i < after.size(); ++i)
        {
            result.guardIntact = result.guardIntact && after[i] == kGuardByte;
        }
        return result;
    }

private:
    size_t mWorkspaceBytes{0};
    std::unique_ptr<IPluginV3> mAttached;
};

//! Whether row @p row of a [kPoolRows, ...] state pool is the same in @p before and @p after.
template <typename T>
bool poolRowEqual(std::vector<T> const& before, std::vector<T> const& after, int32_t row)
{
    size_t const rowElems = before.size() / kPoolRows;
    return std::memcmp(before.data() + row * rowElems, after.data() + row * rowElems, rowElems * sizeof(T)) == 0;
}

class CompactWorkspaceTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        ASSERT_NE(test::loadPluginLibrary(), nullptr);
        mSm = getSMVersion();
    }

    //! The step on pool rows 3 and 1 left rows 0 and 2 and the guard bytes as they were and updated rows 3 and 1.
    template <typename T>
    static void expectStepWithinItsWorkspace(
        PluginLayer::StepResult const& result, std::vector<T> const& before, std::vector<T> const& after)
    {
        EXPECT_EQ(result.status, 0);
        EXPECT_TRUE(result.guardIntact) << "the step wrote past the reserved workspace";
        EXPECT_FALSE(poolRowEqual(before, after, 3)) << "the state row of the sequence on pool row 3 was not updated";
        EXPECT_FALSE(poolRowEqual(before, after, 1)) << "the state row of the sequence on pool row 1 was not updated";
        EXPECT_TRUE(poolRowEqual(before, after, 0) && poolRowEqual(before, after, 2))
            << "a row outside the step changed";
    }

    int32_t mSm{0};
};

} // namespace

// Given a gated_delta_net layer (K = V = 128, two Q/K and two value heads) on a GPU where it accepts compact execution,
//       and exactly the workspace its build reserves for profiles of up to 2 sequences and 64 token rows, followed by
//       guard bytes
// When the context-attached copy of the plugin deserialized from the build's fields enqueues a compact step of one
//      5-row prefill sequence on pool row 3 and one decode sequence on pool row 1 over a 4-row resident state pool
// Then it returns 0, updates pool rows 3 and 1, and leaves pool rows 0 and 2 and the guard bytes unchanged
TEST_F(CompactWorkspaceTest, GatedDeltaNetStepFitsTheWorkspaceItsBuildReserved)
{
    constexpr int32_t kHeads{2};
    constexpr int32_t kHeadDim{128};
    if (!gatedDeltaNetCompactSupport(kHeadDim, kHeadDim, mSm, false, false).supported)
    {
        GTEST_SKIP() << "gated_delta_net has no compact execution on SM" << mSm;
    }
    int32_t const zero{0};
    std::vector<PluginField> const fields{PluginField("k_dim", &kHeadDim, PluginFieldType::kINT32, 1),
        PluginField("v_dim", &kHeadDim, PluginFieldType::kINT32, 1),
        PluginField("use_spec_verify_state", &zero, PluginFieldType::kINT32, 1),
        PluginField("use_diffusion_state", &zero, PluginFieldType::kINT32, 1)};
    // Inputs: q, k, v, a, b, A_log, dt_bias, h0 pool, query lengths, offsets, state indices, phase marker, Nctx
    // carrier. Outputs: o and the aliased pool.
    auto const descs = [](int64_t tokens, int64_t sequences, int64_t contexts, int64_t marker) {
        return std::vector<PluginTensorDesc>{descOf(DataType::kHALF, {tokens, kHeads, kHeadDim}),
            descOf(DataType::kHALF, {tokens, kHeads, kHeadDim}), descOf(DataType::kHALF, {tokens, kHeads, kHeadDim}),
            descOf(DataType::kHALF, {tokens, kHeads}), descOf(DataType::kHALF, {tokens, kHeads}),
            descOf(DataType::kFLOAT, {kHeads}), descOf(DataType::kHALF, {kHeads}),
            descOf(DataType::kFLOAT, {kPoolRows, kHeads, kHeadDim, kHeadDim}), descOf(DataType::kINT32, {sequences}),
            descOf(DataType::kINT32, {sequences + 1}), descOf(DataType::kINT32, {sequences}),
            descOf(DataType::kINT32, {marker}), descOf(DataType::kINT32, {contexts})};
    };
    auto const outDescs = [](int64_t tokens) {
        return std::vector<PluginTensorDesc>{descOf(DataType::kHALF, {tokens, kHeads, kHeadDim}),
            descOf(DataType::kFLOAT, {kPoolRows, kHeads, kHeadDim, kHeadDim})};
    };
    PluginLayer layer("gated_delta_net", fields, descs(64, 2, 2, kCompactPhase), outDescs(64));
    ASSERT_GT(layer.workspaceBytes(), 0U);

    constexpr int32_t kTokens{6};
    std::mt19937 gen(1729);
    size_t const qkElems = static_cast<size_t>(kTokens) * kHeads * kHeadDim;
    size_t const gateElems = static_cast<size_t>(kTokens) * kHeads;
    size_t const poolElems = static_cast<size_t>(kPoolRows) * kHeads * kHeadDim * kHeadDim;
    DeviceBuffer<__half> q(qkElems), k(qkElems), v(qkElems), a(gateElems), b(gateElems), dtBias(kHeads), o(qkElems);
    DeviceBuffer<float> logDecay(kHeads), pool(poolElems);
    DeviceBuffer<int32_t> queryLengths(2), offsets(3), stateIndices(2), marker(kCompactPhase), carrier(1);
    q.upload(randomValues<__half>(qkElems, gen, 0.0F, 1.0F));
    k.upload(randomValues<__half>(qkElems, gen, 0.0F, 1.0F));
    v.upload(randomValues<__half>(qkElems, gen, 0.0F, 1.0F));
    a.upload(randomValues<__half>(gateElems, gen, 0.0F, 1.0F));
    b.upload(randomValues<__half>(gateElems, gen, 0.0F, 1.0F));
    dtBias.upload(randomValues<__half>(kHeads, gen, 0.0F, 0.1F));
    logDecay.upload({-0.5F, 0.25F});
    std::vector<float> const before = randomValues<float>(poolElems, gen, 0.0F, 0.1F);
    pool.upload(before);
    queryLengths.upload({5, 1});
    offsets.upload({0, 5, kTokens});
    stateIndices.upload({3, 1});

    PluginLayer::StepResult const result = layer.enqueue(descs(kTokens, 2, 1, kCompactPhase), outDescs(kTokens),
        {q.data, k.data, v.data, a.data, b.data, logDecay.data, dtBias.data, pool.data, queryLengths.data, offsets.data,
            stateIndices.data, marker.data, carrier.data},
        {o.data, pool.data});
    expectStepWithinItsWorkspace(result, before, pool.download());
}

// Given a depthwise causal_conv1d layer (64 channels, width 4) on a GPU where it accepts compact execution, and exactly
//       the workspace its build reserves for profiles of up to 2 sequences and 64 token rows, followed by guard bytes
// When the context-attached copy of the plugin deserialized from the build's fields enqueues a compact step of two
//      prefill sequences, 5 rows on pool row 3 and 7 rows on pool row 1, over a 4-row resident state pool (as many
//      prompt sequences as the profile allows)
// Then it returns 0, updates pool rows 3 and 1, and leaves pool rows 0 and 2 and the guard bytes unchanged
TEST_F(CompactWorkspaceTest, CausalConv1dStepFitsTheWorkspaceItsBuildReserved)
{
    constexpr int32_t kChannels{64};
    constexpr int32_t kWidth{4};
    if (!causalConv1dCompactSupport(mSm, kChannels, kWidth, /*dilation=*/1, /*groups=*/0, false).supported)
    {
        GTEST_SKIP() << "causal_conv1d has no compact execution on SM" << mSm;
    }
    int32_t const padding{kWidth - 1};
    int32_t const zero{0};
    std::vector<PluginField> const fields{PluginField("padding", &padding, PluginFieldType::kINT32, 1),
        PluginField("use_spec_verify_state", &zero, PluginFieldType::kINT32, 1)};
    // Inputs: x, weight, bias, conv state pool, query lengths, offsets, state indices, phase marker, Nctx carrier.
    // Outputs: y and the aliased pool.
    auto const descs = [](int64_t tokens, int64_t sequences, int64_t contexts) {
        return std::vector<PluginTensorDesc>{descOf(DataType::kHALF, {tokens, kChannels}),
            descOf(DataType::kHALF, {kChannels, 1, kWidth}), descOf(DataType::kHALF, {kChannels}),
            descOf(DataType::kHALF, {kPoolRows, kChannels, kWidth - 1}), descOf(DataType::kINT32, {sequences}),
            descOf(DataType::kINT32, {sequences + 1}), descOf(DataType::kINT32, {sequences}),
            descOf(DataType::kINT32, {kCompactPhase}), descOf(DataType::kINT32, {contexts})};
    };
    auto const outDescs = [](int64_t tokens) {
        return std::vector<PluginTensorDesc>{
            descOf(DataType::kHALF, {tokens, kChannels}), descOf(DataType::kHALF, {kPoolRows, kChannels, kWidth - 1})};
    };
    PluginLayer layer("causal_conv1d", fields, descs(64, 2, 2), outDescs(64));
    ASSERT_GT(layer.workspaceBytes(), 0U);

    constexpr int32_t kTokens{12};
    std::mt19937 gen(4242);
    size_t const xElems = static_cast<size_t>(kTokens) * kChannels;
    size_t const poolElems = static_cast<size_t>(kPoolRows) * kChannels * (kWidth - 1);
    DeviceBuffer<__half> x(xElems), weight(kChannels * kWidth), bias(kChannels), y(xElems), pool(poolElems);
    DeviceBuffer<int32_t> queryLengths(2), offsets(3), stateIndices(2), marker(kCompactPhase), carrier(2);
    x.upload(randomValues<__half>(xElems, gen, 0.0F, 1.0F));
    weight.upload(randomValues<__half>(kChannels * kWidth, gen, 0.0F, 0.5F));
    bias.upload(randomValues<__half>(kChannels, gen, 0.0F, 0.1F));
    std::vector<__half> const before = randomValues<__half>(poolElems, gen, 0.0F, 1.0F);
    pool.upload(before);
    queryLengths.upload({5, 7});
    offsets.upload({0, 5, kTokens});
    stateIndices.upload({3, 1});

    PluginLayer::StepResult const result = layer.enqueue(descs(kTokens, 2, 2), outDescs(kTokens),
        {x.data, weight.data, bias.data, pool.data, queryLengths.data, offsets.data, stateIndices.data, marker.data,
            carrier.data},
        {y.data, pool.data});
    expectStepWithinItsWorkspace(result, before, pool.download());
}

// Given a Mamba2 SSD update_ssm_state layer (8 heads, head dim 64, state size 128, 2 groups) on a GPU where it accepts
//       compact execution, and exactly the workspace its build reserves for profiles of up to 2 sequences and 256
//       token rows, followed by guard bytes
// When the context-attached copy of the plugin deserialized from the build's fields enqueues a compact step of two
//      prefill sequences, 130 rows on pool row 3 and 126 rows on pool row 1, over a 4-row resident state pool (as
//      many prompt sequences and rows as the profile allows)
// Then it returns 0, updates pool rows 3 and 1, and leaves pool rows 0 and 2 and the guard bytes unchanged
TEST_F(CompactWorkspaceTest, MambaSsdStepFitsTheWorkspaceItsBuildReserved)
{
    constexpr int32_t kHeads{8};
    constexpr int32_t kHeadDim{64};
    constexpr int32_t kStateDim{128};
    constexpr int32_t kGroups{2};
    if (!mamba2SsdCompactSupport(kHeadDim, kStateDim, mSm, false).supported)
    {
        GTEST_SKIP() << "update_ssm_state has no compact execution on SM" << mSm;
    }
    int32_t const heads{kHeads};
    int32_t const headDim{kHeadDim};
    int32_t const stateDim{kStateDim};
    int32_t const groups{kGroups};
    int32_t const one{1};
    int32_t const zero{0};
    std::vector<PluginField> const fields{PluginField("dim", &headDim, PluginFieldType::kINT32, 1),
        PluginField("dstate", &stateDim, PluginFieldType::kINT32, 1),
        PluginField("nheads", &heads, PluginFieldType::kINT32, 1),
        PluginField("ngroups", &groups, PluginFieldType::kINT32, 1),
        PluginField("dt_softplus", &one, PluginFieldType::kINT32, 1),
        PluginField("use_spec_verify_state", &zero, PluginFieldType::kINT32, 1)};
    // Inputs: x, A, B, C, D, dt, dt_bias, state pool, query lengths, offsets, state indices, phase marker, Nctx
    // carrier. Outputs: y and the aliased pool.
    auto const descs = [](int64_t tokens, int64_t sequences, int64_t contexts) {
        return std::vector<PluginTensorDesc>{descOf(DataType::kHALF, {tokens, kHeads, kHeadDim}),
            descOf(DataType::kFLOAT, {kHeads}), descOf(DataType::kHALF, {tokens, kGroups, kStateDim}),
            descOf(DataType::kHALF, {tokens, kGroups, kStateDim}), descOf(DataType::kHALF, {kHeads}),
            descOf(DataType::kHALF, {tokens, kHeads}), descOf(DataType::kHALF, {kHeads}),
            descOf(DataType::kHALF, {kPoolRows, kHeads, kHeadDim, kStateDim}), descOf(DataType::kINT32, {sequences}),
            descOf(DataType::kINT32, {sequences + 1}), descOf(DataType::kINT32, {sequences}),
            descOf(DataType::kINT32, {kCompactPhase}), descOf(DataType::kINT32, {contexts})};
    };
    auto const outDescs = [](int64_t tokens) {
        return std::vector<PluginTensorDesc>{descOf(DataType::kHALF, {tokens, kHeads, kHeadDim}),
            descOf(DataType::kHALF, {kPoolRows, kHeads, kHeadDim, kStateDim})};
    };
    constexpr int32_t kTokens{256};
    PluginLayer layer("update_ssm_state", fields, descs(kTokens, 2, 2), outDescs(kTokens));
    ASSERT_GT(layer.workspaceBytes(), 0U);

    std::mt19937 gen(2718);
    size_t const xElems = static_cast<size_t>(kTokens) * kHeads * kHeadDim;
    size_t const bcElems = static_cast<size_t>(kTokens) * kGroups * kStateDim;
    size_t const dtElems = static_cast<size_t>(kTokens) * kHeads;
    size_t const poolElems = static_cast<size_t>(kPoolRows) * kHeads * kHeadDim * kStateDim;
    DeviceBuffer<__half> x(xElems), bIn(bcElems), cIn(bcElems), d(kHeads), dt(dtElems), dtBias(kHeads), y(xElems),
        pool(poolElems);
    DeviceBuffer<float> aIn(kHeads);
    DeviceBuffer<int32_t> queryLengths(2), offsets(3), stateIndices(2), marker(kCompactPhase), carrier(2);
    x.upload(randomValues<__half>(xElems, gen, 0.0F, 1.0F));
    aIn.upload(randomValues<float>(kHeads, gen, -1.0F, 0.1F));
    bIn.upload(randomValues<__half>(bcElems, gen, 0.0F, 1.0F));
    cIn.upload(randomValues<__half>(bcElems, gen, 0.0F, 1.0F));
    d.upload(randomValues<__half>(kHeads, gen, 0.0F, 1.0F));
    dt.upload(randomValues<__half>(dtElems, gen, 0.0F, 1.0F));
    dtBias.upload(randomValues<__half>(kHeads, gen, 0.0F, 0.1F));
    std::vector<__half> const before = randomValues<__half>(poolElems, gen, 0.0F, 0.5F);
    pool.upload(before);
    queryLengths.upload({130, 126});
    offsets.upload({0, 130, kTokens});
    stateIndices.upload({3, 1});

    PluginLayer::StepResult const result = layer.enqueue(descs(kTokens, 2, 2), outDescs(kTokens),
        {x.data, aIn.data, bIn.data, cIn.data, d.data, dt.data, dtBias.data, pool.data, queryLengths.data, offsets.data,
            stateIndices.data, marker.data, carrier.data},
        {y.data, pool.data});
    expectStepWithinItsWorkspace(result, before, pool.download());
}

} // namespace trt_edgellm::plugins
