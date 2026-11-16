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

#include "testPluginLoader.h"

#include "common/tensor.h"
#include "kernels/speculative/lowRankCandidateHeadKernels.h"

#include <NvInfer.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace trt_edgellm::plugins
{
namespace
{
using namespace nvinfer1;
using nvinfer1::DataType;

constexpr int32_t kHidden{256};
constexpr int32_t kRank{16};
constexpr int32_t kVocab{4096};
constexpr int32_t kCandidates{32};
constexpr int32_t kMaxRows{80};
constexpr int32_t kDecodeRows{16};
constexpr int32_t kOnnxOutputs{2};
constexpr int32_t kNumInputs{4};
constexpr int32_t kNumOutputs{2};

Dims matrix(int64_t rows, int64_t cols)
{
    Dims dims{};
    dims.nbDims = 2;
    dims.d[0] = rows;
    dims.d[1] = cols;
    return dims;
}

//! A [1..maxRows, cols] tensor of the profile (opt below max), or a fixed [rows, cols] weight when maxRows is 0.
DynamicPluginTensorDesc describe(DataType type, int64_t rows, int64_t cols, int64_t maxRows = 0)
{
    DynamicPluginTensorDesc desc{};
    desc.desc.type = type;
    desc.desc.format = TensorFormat::kLINEAR;
    desc.desc.dims = matrix(maxRows == 0 ? rows : -1, cols);
    desc.min = matrix(maxRows == 0 ? rows : 1, cols);
    desc.opt = matrix(maxRows == 0 ? rows : maxRows / 2, cols);
    desc.max = matrix(maxRows == 0 ? rows : maxRows, cols);
    return desc;
}

std::unique_ptr<IPluginV3> createHead(
    IPluginCreatorV3One& creator, std::vector<PluginField> const& fields, TensorRTPhase phase)
{
    PluginFieldCollection const fc{static_cast<int32_t>(fields.size()), fields.data()};
    return std::unique_ptr<IPluginV3>(creator.createPlugin("low_rank_candidate_head", &fc, phase));
}

std::map<std::string, int64_t> readFields(PluginFieldCollection const& fc)
{
    std::map<std::string, int64_t> values;
    for (int32_t i = 0; i < fc.nbFields; ++i)
    {
        PluginField const& field = fc.fields[i];
        values[field.name] = field.type == PluginFieldType::kINT64 ? *static_cast<int64_t const*>(field.data)
                                                                   : *static_cast<int32_t const*>(field.data);
    }
    return values;
}

IPluginCreatorV3One* headCreator()
{
    return static_cast<IPluginCreatorV3One*>(getPluginRegistry()->getCreator("LowRankCandidateHeadPlugin", "1", ""));
}

//! A build-phase plugin from only the ONNX attributes: num_candidates kCandidates and onnxscript's _outputs.
std::unique_ptr<IPluginV3> createBuildHead(IPluginCreatorV3One& creator)
{
    int32_t const candidates{kCandidates};
    int32_t const onnxOutputs{kOnnxOutputs};
    return createHead(creator,
        {PluginField("num_candidates", &candidates, PluginFieldType::kINT32, 1),
            PluginField("_outputs", &onnxOutputs, PluginFieldType::kINT32, 1)},
        TensorRTPhase::kBUILD);
}

std::array<DynamicPluginTensorDesc, kNumInputs> headInputs(int64_t maxRows)
{
    return {describe(DataType::kHALF, 0, kHidden, maxRows), describe(DataType::kHALF, kRank, kHidden),
        describe(DataType::kHALF, kVocab, kRank), describe(DataType::kHALF, kVocab, kHidden)};
}

std::array<DynamicPluginTensorDesc, kNumOutputs> headOutputs(int64_t maxRows)
{
    return {describe(DataType::kINT32, 0, kCandidates, maxRows), describe(DataType::kFLOAT, 0, kCandidates, maxRows)};
}

//! Device buffers of one head run over kMaxRows random hidden rows and random weights (fixed seed).
class HeadBuffers
{
public:
    HeadBuffers()
    {
        std::mt19937 generator(7);
        std::normal_distribution<float> normal(0.0F, 1.0F);
        for (int32_t input = 0; input < kNumInputs; ++input)
        {
            std::vector<__half> host(static_cast<size_t>(kShapes[input][0] * kShapes[input][1]));
            for (auto& value : host)
            {
                value = __float2half(normal(generator) * kScales[input]);
            }
            mInputs[input] = rt::Tensor(
                {kShapes[input][0], kShapes[input][1]}, rt::DeviceType::kGPU, trt_edgellm::DataType::kHALF);
            EXPECT_EQ(cudaMemcpy(mInputs[input].rawPointer(), host.data(), host.size() * sizeof(__half),
                          cudaMemcpyHostToDevice),
                cudaSuccess);
        }
        EXPECT_EQ(cudaStreamCreate(&mStream), cudaSuccess);
    }

    ~HeadBuffers()
    {
        static_cast<void>(cudaStreamDestroy(mStream));
    }

    HeadBuffers(HeadBuffers const&) = delete;
    HeadBuffers& operator=(HeadBuffers const&) = delete;

    //! Runs `runtime` over all kMaxRows rows with `workspaceBytes` of workspace, followed by a guard region that must
    //! stay untouched; returns the ids and log-probs as bytes, or nothing when enqueue fails.
    std::vector<uint8_t> run(IPluginV3OneRuntime& runtime, size_t workspaceBytes)
    {
        constexpr size_t kGuardBytes{1 << 20};
        constexpr uint8_t kGuardByte{0x5A};
        rt::Tensor workspaceTensor(
            {static_cast<int64_t>(workspaceBytes + kGuardBytes)}, rt::DeviceType::kGPU, trt_edgellm::DataType::kINT8);
        auto* const workspace = static_cast<uint8_t*>(workspaceTensor.rawPointer());
        EXPECT_EQ(cudaMemset(workspace + workspaceBytes, kGuardByte, kGuardBytes), cudaSuccess);
        std::array<PluginTensorDesc, kNumInputs> inputs{};
        std::array<void const*, kNumInputs> inputPointers{};
        for (int32_t input = 0; input < kNumInputs; ++input)
        {
            inputs[input].dims = matrix(input == 0 ? kMaxRows : kShapes[input][0], kShapes[input][1]);
            inputs[input].type = DataType::kHALF;
            inputs[input].format = TensorFormat::kLINEAR;
            inputPointers[input] = mInputs[input].rawPointer();
        }
        std::array<PluginTensorDesc, kNumOutputs> outputs{};
        outputs[0].dims = matrix(kMaxRows, kCandidates);
        outputs[0].type = DataType::kINT32;
        outputs[1].dims = matrix(kMaxRows, kCandidates);
        outputs[1].type = DataType::kFLOAT;
        std::array<void*, kNumOutputs> outputPointers{mIds.rawPointer(), mLogProbs.rawPointer()};
        size_t const idBytes = static_cast<size_t>(kMaxRows) * kCandidates * sizeof(int32_t);
        size_t const logProbBytes = static_cast<size_t>(kMaxRows) * kCandidates * sizeof(float);
        // Poisoned, so that an enqueue that writes nothing cannot return what an earlier run left in the outputs.
        EXPECT_EQ(cudaMemset(mIds.rawPointer(), 0xFF, idBytes), cudaSuccess);
        EXPECT_EQ(cudaMemset(mLogProbs.rawPointer(), 0xFF, logProbBytes), cudaSuccess);
        EXPECT_EQ(runtime.onShapeChange(inputs.data(), kNumInputs, outputs.data(), kNumOutputs), 0);
        int32_t const status = runtime.enqueue(
            inputs.data(), outputs.data(), inputPointers.data(), outputPointers.data(), workspace, mStream);
        EXPECT_EQ(cudaStreamSynchronize(mStream), cudaSuccess);
        std::vector<uint8_t> result(idBytes + logProbBytes);
        EXPECT_EQ(cudaMemcpy(result.data(), mIds.rawPointer(), idBytes, cudaMemcpyDeviceToHost), cudaSuccess);
        EXPECT_EQ(cudaMemcpy(result.data() + idBytes, mLogProbs.rawPointer(), logProbBytes, cudaMemcpyDeviceToHost),
            cudaSuccess);
        std::vector<uint8_t> guard(kGuardBytes);
        EXPECT_EQ(
            cudaMemcpy(guard.data(), workspace + workspaceBytes, kGuardBytes, cudaMemcpyDeviceToHost), cudaSuccess);
        EXPECT_TRUE(std::all_of(guard.begin(), guard.end(), [](uint8_t byte) { return byte == kGuardByte; }))
            << "the head wrote past the " << workspaceBytes << " workspace bytes it was given";
        return status == 0 ? result : std::vector<uint8_t>{};
    }

    //! The ids and log-probs the head kernel itself writes for all kMaxRows rows, as bytes.
    std::vector<uint8_t> runKernel(cudaDeviceProp const& prop)
    {
        kernel::LowRankCandidateHeadRunner const runner(
            kernel::LowRankCandidateHeadShape{kHidden, kRank, kVocab, kCandidates}, kMaxRows, prop);
        rt::Tensor workspace(
            {static_cast<int64_t>(runner.workspaceBytes())}, rt::DeviceType::kGPU, trt_edgellm::DataType::kINT8);
        runner.run(mInputs[0], mInputs[1], mInputs[2], mInputs[3], mIds, mLogProbs, workspace.rawPointer(), mStream);
        EXPECT_EQ(cudaStreamSynchronize(mStream), cudaSuccess);
        size_t const idBytes = static_cast<size_t>(kMaxRows) * kCandidates * sizeof(int32_t);
        std::vector<uint8_t> result(2 * idBytes);
        EXPECT_EQ(cudaMemcpy(result.data(), mIds.rawPointer(), idBytes, cudaMemcpyDeviceToHost), cudaSuccess);
        EXPECT_EQ(
            cudaMemcpy(result.data() + idBytes, mLogProbs.rawPointer(), idBytes, cudaMemcpyDeviceToHost), cudaSuccess);
        return result;
    }

private:
    static constexpr std::array<std::array<int64_t, 2>, kNumInputs> kShapes{
        {{kMaxRows, kHidden}, {kRank, kHidden}, {kVocab, kRank}, {kVocab, kHidden}}};
    static constexpr std::array<float, kNumInputs> kScales{1.0F, 0.0625F, 0.25F, 0.0625F};

    std::array<rt::Tensor, kNumInputs> mInputs;
    rt::Tensor mIds{{kMaxRows, kCandidates}, rt::DeviceType::kGPU, trt_edgellm::DataType::kINT32};
    rt::Tensor mLogProbs{{kMaxRows, kCandidates}, rt::DeviceType::kGPU, trt_edgellm::DataType::kFLOAT};
    cudaStream_t mStream{};
};

// Given a LowRankCandidateHeadPlugin created for the build with only the ONNX attributes (num_candidates 32 and
//       onnxscript's _outputs) and configured for hidden [1..16, 256] (a decode profile), down_proj [16, 256],
//       up_proj [4096, 16] and lm_head [4096, 256], whose getWorkspaceSize reserves D bytes on this device, then
//       configured for hidden [1..80, 256] (a prefill profile), whose getWorkspaceSize reserves W bytes, and the
//       M < W bytes the smallest configuration of the head needs on this device
// When engines deserialize its serialized fields with workspace_bytes W, W - 1, M and M - 1, attach each plugin to an
//      execution context on this device, and run every attached plugin on the same 80 random hidden rows
// Then D equals W (the smaller profile, queried before the larger one is configured, still reserves what the runner
//      of the larger one is laid out for); the build serializes hidden_size 256, rank 16, vocab_size 4096, max_rows
//      80 and W; every plugin attaches; the plugins with W, W - 1 and M write bitwise the candidate ids and log-probs
//      of the head kernel run directly on the same inputs, and leave the memory past their reserved workspace
//      untouched; and the one with M - 1 accepts the shapes in onShapeChange, but its enqueue fails: an engine that
//      reserved less than this device prefers runs a smaller configuration instead of failing, and fails only when no
//      configuration fits
TEST(LowRankCandidateHeadPluginTest, EnqueueFitsTheReservedWorkspaceOrRefuses)
{
    ASSERT_NE(test::loadPluginLibrary(), nullptr) << test::pluginLoadError();
    IPluginCreatorV3One* const creator = headCreator();
    ASSERT_NE(creator, nullptr);

    std::unique_ptr<IPluginV3> buildPlugin = createBuildHead(*creator);
    ASSERT_NE(buildPlugin, nullptr);
    auto* build = static_cast<IPluginV3OneBuild*>(buildPlugin->getCapabilityInterface(PluginCapabilityType::kBUILD));
    auto const decodeInputs = headInputs(kDecodeRows);
    auto const decodeOutputs = headOutputs(kDecodeRows);
    ASSERT_EQ(build->configurePlugin(decodeInputs.data(), kNumInputs, decodeOutputs.data(), kNumOutputs), 0);
    size_t const decodeWorkspaceBytes
        = build->getWorkspaceSize(decodeInputs.data(), kNumInputs, decodeOutputs.data(), kNumOutputs);
    auto const inputs = headInputs(kMaxRows);
    auto const outputs = headOutputs(kMaxRows);
    ASSERT_EQ(build->configurePlugin(inputs.data(), kNumInputs, outputs.data(), kNumOutputs), 0);
    size_t const workspaceBytes = build->getWorkspaceSize(inputs.data(), kNumInputs, outputs.data(), kNumOutputs);
    ASSERT_GT(workspaceBytes, 0U);
    EXPECT_EQ(decodeWorkspaceBytes, workspaceBytes);

    auto* buildRuntime
        = static_cast<IPluginV3OneRuntime*>(buildPlugin->getCapabilityInterface(PluginCapabilityType::kRUNTIME));
    PluginFieldCollection const* serialized = buildRuntime->getFieldsToSerialize();
    ASSERT_NE(serialized, nullptr);
    std::map<std::string, int64_t> const expected{{"num_candidates", kCandidates}, {"hidden_size", kHidden},
        {"rank", kRank}, {"vocab_size", kVocab}, {"max_rows", kMaxRows},
        {"workspace_bytes", static_cast<int64_t>(workspaceBytes)}};
    EXPECT_EQ(serialized->nbFields, static_cast<int32_t>(expected.size()));
    EXPECT_EQ(readFields(*serialized), expected);

    int32_t device{};
    ASSERT_EQ(cudaGetDevice(&device), cudaSuccess);
    cudaDeviceProp prop{};
    ASSERT_EQ(cudaGetDeviceProperties(&prop, device), cudaSuccess);
    kernel::LowRankCandidateHeadShape const shape{kHidden, kRank, kVocab, kCandidates};
    size_t const minimumBytes = kernel::LowRankCandidateHeadRunner::minimumWorkspaceBytes(shape, kMaxRows, prop);
    ASSERT_LT(minimumBytes, workspaceBytes) << "the smallest configuration must be smaller than the preferred one";

    HeadBuffers buffers;
    std::vector<uint8_t> const reference = buffers.runKernel(prop);
    for (int64_t const reserved : {static_cast<int64_t>(workspaceBytes), static_cast<int64_t>(workspaceBytes) - 1,
             static_cast<int64_t>(minimumBytes), static_cast<int64_t>(minimumBytes) - 1})
    {
        SCOPED_TRACE("workspace_bytes " + std::to_string(reserved) + " of " + std::to_string(workspaceBytes)
            + ", smallest " + std::to_string(minimumBytes));
        std::vector<PluginField> fields;
        for (int32_t i = 0; i < serialized->nbFields; ++i)
        {
            PluginField const& field = serialized->fields[i];
            bool const isWorkspace = std::strcmp(field.name, "workspace_bytes") == 0;
            fields.emplace_back(field.name, isWorkspace ? &reserved : field.data, field.type, field.length);
        }
        std::unique_ptr<IPluginV3> deserialized = createHead(*creator, fields, TensorRTPhase::kRUNTIME);
        ASSERT_NE(deserialized, nullptr);
        auto* runtime
            = static_cast<IPluginV3OneRuntime*>(deserialized->getCapabilityInterface(PluginCapabilityType::kRUNTIME));
        std::unique_ptr<IPluginV3> attached(runtime->attachToContext(nullptr));
        ASSERT_NE(attached, nullptr);
        auto* attachedRuntime
            = static_cast<IPluginV3OneRuntime*>(attached->getCapabilityInterface(PluginCapabilityType::kRUNTIME));
        std::vector<uint8_t> const candidates = buffers.run(*attachedRuntime, static_cast<size_t>(reserved));
        if (reserved >= static_cast<int64_t>(minimumBytes))
        {
            EXPECT_TRUE(candidates == reference) << "the plugin's candidates differ from the head kernel's";
        }
        else
        {
            EXPECT_TRUE(candidates.empty()) << "enqueue succeeded with a workspace no configuration fits";
        }
    }
}

// Given a LowRankCandidateHeadPlugin created for the build with only the ONNX attributes (num_candidates 32 and
//       onnxscript's _outputs), not yet configured
// When it is attached to an execution context first, and the attached plugin is then configured for hidden [1..80,
//      256], down_proj [16, 256], up_proj [4096, 16] and lm_head [4096, 256] and run with its getWorkspaceSize bytes
//      on 80 random hidden rows (the order TensorRT uses to time a plugin layer that the compiler backend does not
//      absorb, as on SM80-SM90)
// Then attaching returns a plugin, and the run writes bitwise the candidate ids and log-probs of the head kernel run
//      directly on the same inputs and leaves the memory past the workspace untouched
TEST(LowRankCandidateHeadPluginTest, AttachBeforeConfigureRunsTheConfiguredHead)
{
    ASSERT_NE(test::loadPluginLibrary(), nullptr) << test::pluginLoadError();
    IPluginCreatorV3One* const creator = headCreator();
    ASSERT_NE(creator, nullptr);

    std::unique_ptr<IPluginV3> buildPlugin = createBuildHead(*creator);
    ASSERT_NE(buildPlugin, nullptr);
    auto* buildRuntime
        = static_cast<IPluginV3OneRuntime*>(buildPlugin->getCapabilityInterface(PluginCapabilityType::kRUNTIME));
    std::unique_ptr<IPluginV3> attached(buildRuntime->attachToContext(nullptr));
    ASSERT_NE(attached, nullptr);
    auto* build = static_cast<IPluginV3OneBuild*>(attached->getCapabilityInterface(PluginCapabilityType::kBUILD));
    auto const inputs = headInputs(kMaxRows);
    auto const outputs = headOutputs(kMaxRows);
    ASSERT_EQ(build->configurePlugin(inputs.data(), kNumInputs, outputs.data(), kNumOutputs), 0);
    size_t const workspaceBytes = build->getWorkspaceSize(inputs.data(), kNumInputs, outputs.data(), kNumOutputs);
    ASSERT_GT(workspaceBytes, 0U);

    int32_t device{};
    ASSERT_EQ(cudaGetDevice(&device), cudaSuccess);
    cudaDeviceProp prop{};
    ASSERT_EQ(cudaGetDeviceProperties(&prop, device), cudaSuccess);
    HeadBuffers buffers;
    std::vector<uint8_t> const reference = buffers.runKernel(prop);
    auto* attachedRuntime
        = static_cast<IPluginV3OneRuntime*>(attached->getCapabilityInterface(PluginCapabilityType::kRUNTIME));
    EXPECT_TRUE(buffers.run(*attachedRuntime, workspaceBytes) == reference)
        << "the plugin's candidates differ from the head kernel's";
}

} // namespace
} // namespace trt_edgellm::plugins
