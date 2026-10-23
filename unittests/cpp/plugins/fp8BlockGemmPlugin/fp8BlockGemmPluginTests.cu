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
#include "kernels/fp8BlockwiseGemm/cuteDslFp8BlockwiseGemmRunner.h"
#include "plugins/fp8BlockGemmPlugin/fp8BlockGemmPlugin.h"

#include <NvInfer.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace trt_edgellm::plugins
{
namespace
{

constexpr int32_t kBlockSize{128};
constexpr size_t kWorkspaceAlignment{256};
constexpr float kFp8Max{448.0F};
constexpr double kNrmseThreshold{0.002};
constexpr double kAbsoluteTolerance{0.02};
constexpr double kRelativeTolerance{0.005};

struct ShapeCase
{
    char const* name;
    int32_t m;
    int32_t n;
    int32_t k;
};

template <typename T>
class DeviceBuffer
{
public:
    explicit DeviceBuffer(size_t count)
    {
        CUDA_CHECK(cudaMalloc(&mPointer, count * sizeof(T)));
    }

    ~DeviceBuffer()
    {
        if (mPointer != nullptr)
        {
            (void) cudaFree(mPointer);
        }
    }

    DeviceBuffer(DeviceBuffer const&) = delete;
    DeviceBuffer& operator=(DeviceBuffer const&) = delete;

    T* get() const noexcept
    {
        return mPointer;
    }

private:
    T* mPointer{};
};

class NonBlockingStream
{
public:
    NonBlockingStream()
    {
        CUDA_CHECK(cudaStreamCreateWithFlags(&mStream, cudaStreamNonBlocking));
    }

    ~NonBlockingStream()
    {
        if (mStream != nullptr)
        {
            (void) cudaStreamDestroy(mStream);
        }
    }

    NonBlockingStream(NonBlockingStream const&) = delete;
    NonBlockingStream& operator=(NonBlockingStream const&) = delete;

    cudaStream_t get() const noexcept
    {
        return mStream;
    }

private:
    cudaStream_t mStream{};
};

struct PluginDescriptors
{
    std::array<nvinfer1::DynamicPluginTensorDesc, 3> dynamicInputs;
    std::array<nvinfer1::DynamicPluginTensorDesc, 1> dynamicOutputs;
    std::array<nvinfer1::PluginTensorDesc, 3> runtimeInputs;
    std::array<nvinfer1::PluginTensorDesc, 1> runtimeOutputs;
};

nvinfer1::Dims makeDims(std::initializer_list<int64_t> extents)
{
    nvinfer1::Dims dims{};
    dims.nbDims = static_cast<int32_t>(extents.size());
    std::copy(extents.begin(), extents.end(), dims.d);
    return dims;
}

nvinfer1::PluginTensorDesc makeTensorDesc(nvinfer1::DataType type, nvinfer1::Dims const& dims)
{
    nvinfer1::PluginTensorDesc desc{};
    desc.dims = dims;
    desc.type = type;
    desc.format = nvinfer1::TensorFormat::kLINEAR;
    desc.scale = 1.0F;
    return desc;
}

nvinfer1::DynamicPluginTensorDesc makeDynamicDesc(nvinfer1::DataType type, nvinfer1::Dims const& dimensions,
    nvinfer1::Dims const& minDimensions, nvinfer1::Dims const& optDimensions, nvinfer1::Dims const& maxDimensions)
{
    nvinfer1::DynamicPluginTensorDesc desc{};
    desc.desc = makeTensorDesc(type, dimensions);
    desc.min = minDimensions;
    desc.opt = optDimensions;
    desc.max = maxDimensions;
    return desc;
}

PluginDescriptors makeDescriptors(int32_t m, int32_t n, int32_t k, int32_t maxM = 0)
{
    if (maxM == 0)
    {
        maxM = m;
    }
    int32_t const nBlocks = n / kBlockSize;
    int32_t const kBlocks = k / kBlockSize;

    PluginDescriptors descriptors{};
    descriptors.dynamicInputs[0] = makeDynamicDesc(
        nvinfer1::DataType::kHALF, makeDims({-1, k}), makeDims({1, k}), makeDims({m, k}), makeDims({maxM, k}));
    descriptors.dynamicInputs[1] = makeDynamicDesc(
        nvinfer1::DataType::kINT8, makeDims({n, k}), makeDims({n, k}), makeDims({n, k}), makeDims({n, k}));
    descriptors.dynamicInputs[2] = makeDynamicDesc(nvinfer1::DataType::kFLOAT, makeDims({nBlocks, kBlocks}),
        makeDims({nBlocks, kBlocks}), makeDims({nBlocks, kBlocks}), makeDims({nBlocks, kBlocks}));
    descriptors.dynamicOutputs[0] = makeDynamicDesc(
        nvinfer1::DataType::kHALF, makeDims({-1, n}), makeDims({1, n}), makeDims({m, n}), makeDims({maxM, n}));

    descriptors.runtimeInputs[0] = makeTensorDesc(nvinfer1::DataType::kHALF, makeDims({m, k}));
    descriptors.runtimeInputs[1] = makeTensorDesc(nvinfer1::DataType::kINT8, makeDims({n, k}));
    descriptors.runtimeInputs[2] = makeTensorDesc(nvinfer1::DataType::kFLOAT, makeDims({nBlocks, kBlocks}));
    descriptors.runtimeOutputs[0] = makeTensorDesc(nvinfer1::DataType::kHALF, makeDims({m, n}));
    return descriptors;
}

size_t alignWorkspace(size_t bytes)
{
    return (bytes + kWorkspaceAlignment - 1) & ~(kWorkspaceAlignment - 1);
}

size_t expectedWorkspaceSize(int32_t numTokens, int32_t gemmK)
{
    size_t const activationBytes = static_cast<size_t>(numTokens) * gemmK;
    size_t const scaleBytes = static_cast<size_t>(numTokens) * (gemmK / kBlockSize) * sizeof(float);
    return alignWorkspace(activationBytes) + alignWorkspace(scaleBytes);
}

bool isSm110()
{
    int32_t deviceCount{};
    if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount == 0)
    {
        return false;
    }
    int32_t device{};
    int32_t major{};
    int32_t minor{};
    return cudaGetDevice(&device) == cudaSuccess
        && cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device) == cudaSuccess
        && cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device) == cudaSuccess
        && major * 10 + minor == 110;
}

float fp8ToFloat(uint8_t raw)
{
    static_assert(sizeof(__nv_fp8_e4m3) == sizeof(raw));
    __nv_fp8_e4m3 value{};
    std::memcpy(&value, &raw, sizeof(raw));
    return static_cast<float>(value);
}

int32_t serializedInt(nvinfer1::PluginFieldCollection const& fields, char const* name)
{
    for (int32_t index = 0; index < fields.nbFields; ++index)
    {
        nvinfer1::PluginField const& field = fields.fields[index];
        if (field.name != nullptr && std::strcmp(field.name, name) == 0)
        {
            EXPECT_EQ(field.type, nvinfer1::PluginFieldType::kINT32);
            EXPECT_EQ(field.length, 1);
            EXPECT_NE(field.data, nullptr);
            return field.data == nullptr ? 0 : *static_cast<int32_t const*>(field.data);
        }
    }
    ADD_FAILURE() << "Missing serialized field " << name;
    return 0;
}

void initializeInputs(ShapeCase const& shape, std::vector<half>& activation, std::vector<uint8_t>& weight,
    std::vector<float>& weightScales)
{
    constexpr std::array<uint8_t, 8> kFp8Codes{0x28U, 0x30U, 0x34U, 0x38U, 0xA8U, 0xB0U, 0xB4U, 0xB8U};
    constexpr std::array<float, 4> kScaleValues{0.015625F, 0.03125F, 0.0625F, 0.125F};

    for (int32_t token = 0; token < shape.m; ++token)
    {
        for (int32_t k = 0; k < shape.k; ++k)
        {
            int32_t const centered = (token * 17 + k * 13) % 29 - 14;
            int32_t const groupFactor = 1 + (token + k / kBlockSize) % 4;
            activation[static_cast<size_t>(token) * shape.k + k]
                = __float2half_rn(static_cast<float>(centered * groupFactor) / 64.0F);
        }
    }

    for (int32_t n = 0; n < shape.n; ++n)
    {
        for (int32_t k = 0; k < shape.k; ++k)
        {
            size_t const codeIndex = static_cast<size_t>((n * 3 + k * 5 + n / kBlockSize + k / kBlockSize) & 7);
            weight[static_cast<size_t>(n) * shape.k + k] = kFp8Codes[codeIndex];
        }
    }

    int32_t const nBlocks = shape.n / kBlockSize;
    int32_t const kBlocks = shape.k / kBlockSize;
    for (int32_t nBlock = 0; nBlock < nBlocks; ++nBlock)
    {
        for (int32_t kBlock = 0; kBlock < kBlocks; ++kBlock)
        {
            size_t const scaleIndex = static_cast<size_t>(nBlock) * kBlocks + kBlock;
            weightScales[scaleIndex] = kScaleValues[static_cast<size_t>((nBlock * 3 + kBlock) & 3)];
        }
    }
}

void validateActivationScales(
    ShapeCase const& shape, std::vector<half> const& activation, std::vector<float> const& activationScales)
{
    int32_t const kBlocks = shape.k / kBlockSize;
    for (int32_t token = 0; token < shape.m; ++token)
    {
        for (int32_t kBlock = 0; kBlock < kBlocks; ++kBlock)
        {
            float maximum{};
            for (int32_t offset = 0; offset < kBlockSize; ++offset)
            {
                int32_t const k = kBlock * kBlockSize + offset;
                maximum
                    = std::max(maximum, std::abs(__half2float(activation[static_cast<size_t>(token) * shape.k + k])));
            }
            float const expected = maximum > 0.0F ? maximum / kFp8Max : 1.0F;
            float const actual = activationScales[static_cast<size_t>(token) * kBlocks + kBlock];
            EXPECT_NEAR(actual, expected, std::max(1.0e-7F, expected * 1.0e-6F))
                << "token=" << token << " kBlock=" << kBlock;
        }
    }
}

void validateOutput(ShapeCase const& shape, std::vector<uint8_t> const& quantizedActivation,
    std::vector<float> const& activationScales, std::vector<uint8_t> const& weight,
    std::vector<float> const& weightScales, std::vector<half> const& output)
{
    int32_t const kBlocks = shape.k / kBlockSize;
    double squaredError{};
    double squaredReference{};
    size_t violations{};
    double maximumAbsoluteError{};

    for (int32_t token = 0; token < shape.m; ++token)
    {
        for (int32_t n = 0; n < shape.n; ++n)
        {
            double reference{};
            for (int32_t k = 0; k < shape.k; ++k)
            {
                float const activation = fp8ToFloat(quantizedActivation[static_cast<size_t>(token) * shape.k + k]);
                float const activationScale = activationScales[static_cast<size_t>(token) * kBlocks + k / kBlockSize];
                float const weightValue = fp8ToFloat(weight[static_cast<size_t>(n) * shape.k + k]);
                float const weightScale = weightScales[static_cast<size_t>(n / kBlockSize) * kBlocks + k / kBlockSize];
                reference = std::fma(static_cast<double>(activation * activationScale),
                    static_cast<double>(weightValue * weightScale), reference);
            }

            double const actual = __half2float(output[static_cast<size_t>(token) * shape.n + n]);
            double const difference = std::abs(actual - reference);
            maximumAbsoluteError = std::max(maximumAbsoluteError, difference);
            squaredError += difference * difference;
            squaredReference += reference * reference;
            if (difference > kAbsoluteTolerance + kRelativeTolerance * std::abs(reference))
            {
                ++violations;
            }
        }
    }

    double const nrmse = std::sqrt(squaredError / std::max(squaredReference, std::numeric_limits<double>::min()));
    EXPECT_LE(nrmse, kNrmseThreshold) << shape.name << " max_abs=" << maximumAbsoluteError;
    EXPECT_EQ(violations, 0U) << shape.name << " max_abs=" << maximumAbsoluteError;
}

void runNumericalCase(ShapeCase const& shape, bool captureAndReplay)
{
    std::vector<half> hostActivation(static_cast<size_t>(shape.m) * shape.k);
    std::vector<uint8_t> hostWeight(static_cast<size_t>(shape.n) * shape.k);
    std::vector<float> hostWeightScales(static_cast<size_t>(shape.n / kBlockSize) * (shape.k / kBlockSize));
    std::vector<half> hostOutput(static_cast<size_t>(shape.m) * shape.n);
    initializeInputs(shape, hostActivation, hostWeight, hostWeightScales);

    Fp8BlockGemmPlugin plugin(shape.name, shape.n, shape.k);
    PluginDescriptors descriptors = makeDescriptors(shape.m, shape.n, shape.k);
    ASSERT_EQ(plugin.configurePlugin(descriptors.dynamicInputs.data(), descriptors.dynamicInputs.size(),
                  descriptors.dynamicOutputs.data(), descriptors.dynamicOutputs.size()),
        0);
    ASSERT_EQ(plugin.onShapeChange(descriptors.runtimeInputs.data(), descriptors.runtimeInputs.size(),
                  descriptors.runtimeOutputs.data(), descriptors.runtimeOutputs.size()),
        0);

    size_t const workspaceBytes = plugin.getWorkspaceSize(descriptors.dynamicInputs.data(),
        descriptors.dynamicInputs.size(), descriptors.dynamicOutputs.data(), descriptors.dynamicOutputs.size());
    ASSERT_EQ(workspaceBytes, expectedWorkspaceSize(shape.m, shape.k));

    DeviceBuffer<half> activation(hostActivation.size());
    DeviceBuffer<uint8_t> weight(hostWeight.size());
    DeviceBuffer<float> weightScales(hostWeightScales.size());
    DeviceBuffer<half> output(hostOutput.size());
    DeviceBuffer<uint8_t> workspace(workspaceBytes);
    NonBlockingStream stream;

    CUDA_CHECK(cudaMemcpyAsync(activation.get(), hostActivation.data(), hostActivation.size() * sizeof(half),
        cudaMemcpyHostToDevice, stream.get()));
    CUDA_CHECK(
        cudaMemcpyAsync(weight.get(), hostWeight.data(), hostWeight.size(), cudaMemcpyHostToDevice, stream.get()));
    CUDA_CHECK(cudaMemcpyAsync(weightScales.get(), hostWeightScales.data(), hostWeightScales.size() * sizeof(float),
        cudaMemcpyHostToDevice, stream.get()));

    std::array<void const*, 3> inputs{activation.get(), weight.get(), weightScales.get()};
    std::array<void*, 1> outputs{output.get()};
    ASSERT_EQ(plugin.enqueue(descriptors.runtimeInputs.data(), descriptors.runtimeOutputs.data(), inputs.data(),
                  outputs.data(), workspace.get(), stream.get()),
        0);
    CUDA_CHECK(cudaStreamSynchronize(stream.get()));

    std::vector<half> warmupOutput;
    if (captureAndReplay)
    {
        warmupOutput.resize(hostOutput.size());
        CUDA_CHECK(cudaMemcpyAsync(warmupOutput.data(), output.get(), warmupOutput.size() * sizeof(half),
            cudaMemcpyDeviceToHost, stream.get()));
        CUDA_CHECK(cudaStreamSynchronize(stream.get()));

        cudaGraph_t graph{};
        ASSERT_EQ(cudaStreamBeginCapture(stream.get(), cudaStreamCaptureModeThreadLocal), cudaSuccess);
        int32_t const enqueueResult = plugin.enqueue(descriptors.runtimeInputs.data(),
            descriptors.runtimeOutputs.data(), inputs.data(), outputs.data(), workspace.get(), stream.get());
        cudaError_t const endCaptureResult = cudaStreamEndCapture(stream.get(), &graph);
        ASSERT_EQ(enqueueResult, 0);
        ASSERT_EQ(endCaptureResult, cudaSuccess);
        ASSERT_NE(graph, nullptr);

        cudaGraphExec_t graphExec{};
        ASSERT_EQ(cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0), cudaSuccess);
        ASSERT_EQ(cudaMemsetAsync(output.get(), 0, hostOutput.size() * sizeof(half), stream.get()), cudaSuccess);
        ASSERT_EQ(cudaGraphLaunch(graphExec, stream.get()), cudaSuccess);
        ASSERT_EQ(cudaGraphLaunch(graphExec, stream.get()), cudaSuccess);
        CUDA_CHECK(cudaStreamSynchronize(stream.get()));
        EXPECT_EQ(cudaGraphExecDestroy(graphExec), cudaSuccess);
        EXPECT_EQ(cudaGraphDestroy(graph), cudaSuccess);
    }

    size_t const activationBytes = static_cast<size_t>(shape.m) * shape.k;
    size_t const scaleOffset = alignWorkspace(activationBytes);
    size_t const activationScaleCount = static_cast<size_t>(shape.m) * (shape.k / kBlockSize);
    std::vector<uint8_t> quantizedActivation(activationBytes);
    std::vector<float> activationScales(activationScaleCount);
    CUDA_CHECK(cudaMemcpyAsync(
        quantizedActivation.data(), workspace.get(), activationBytes, cudaMemcpyDeviceToHost, stream.get()));
    CUDA_CHECK(cudaMemcpyAsync(activationScales.data(), workspace.get() + scaleOffset,
        activationScaleCount * sizeof(float), cudaMemcpyDeviceToHost, stream.get()));
    CUDA_CHECK(cudaMemcpyAsync(
        hostOutput.data(), output.get(), hostOutput.size() * sizeof(half), cudaMemcpyDeviceToHost, stream.get()));
    CUDA_CHECK(cudaStreamSynchronize(stream.get()));

    if (captureAndReplay)
    {
        EXPECT_EQ(std::memcmp(warmupOutput.data(), hostOutput.data(), hostOutput.size() * sizeof(half)), 0);
    }
    validateActivationScales(shape, hostActivation, activationScales);
    validateOutput(shape, quantizedActivation, activationScales, hostWeight, hostWeightScales, hostOutput);
}

TEST(Fp8BlockGemmPluginTest, SerializedFieldsRecreatePlugin)
{
    constexpr int32_t kGemmN{4096};
    constexpr int32_t kGemmK{2048};
    Fp8BlockGemmPlugin plugin("serialize", kGemmN, kGemmK);

    nvinfer1::PluginFieldCollection const* fields = plugin.getFieldsToSerialize();
    ASSERT_NE(fields, nullptr);
    ASSERT_EQ(fields->nbFields, 2);
    EXPECT_EQ(serializedInt(*fields, "gemm_n"), kGemmN);
    EXPECT_EQ(serializedInt(*fields, "gemm_k"), kGemmK);

    Fp8BlockGemmPluginCreator creator;
    std::unique_ptr<nvinfer1::IPluginV3> restored(
        creator.createPlugin("deserialize", fields, nvinfer1::TensorRTPhase::kRUNTIME));
    ASSERT_NE(restored, nullptr);
    auto* const restoredPlugin = static_cast<Fp8BlockGemmPlugin*>(restored.get());
    nvinfer1::PluginFieldCollection const* restoredFields = restoredPlugin->getFieldsToSerialize();
    ASSERT_NE(restoredFields, nullptr);
    EXPECT_EQ(serializedInt(*restoredFields, "gemm_n"), kGemmN);
    EXPECT_EQ(serializedInt(*restoredFields, "gemm_k"), kGemmK);
}

TEST(Fp8BlockGemmPluginTest, RejectsInvalidAttributesAndDescriptors)
{
    EXPECT_THROW(Fp8BlockGemmPlugin("invalid_n", 129, 128), std::invalid_argument);
    EXPECT_THROW(Fp8BlockGemmPlugin("invalid_k", 128, 127), std::invalid_argument);
    EXPECT_THROW(Fp8BlockGemmPlugin("zero_n", 0, 128), std::invalid_argument);

    Fp8BlockGemmPluginCreator creator;
    int32_t const invalidN{129};
    int32_t const validK{128};
    std::array<nvinfer1::PluginField, 2> invalidFields{
        nvinfer1::PluginField{"gemm_n", &invalidN, nvinfer1::PluginFieldType::kINT32, 1},
        nvinfer1::PluginField{"gemm_k", &validK, nvinfer1::PluginFieldType::kINT32, 1}};
    nvinfer1::PluginFieldCollection const invalidCollection{
        static_cast<int32_t>(invalidFields.size()), invalidFields.data()};
    EXPECT_EQ(creator.createPlugin("invalid", &invalidCollection, nvinfer1::TensorRTPhase::kBUILD), nullptr);

    Fp8BlockGemmPlugin plugin("descriptors", 256, 256);
    PluginDescriptors descriptors = makeDescriptors(7, 256, 256);
    descriptors.dynamicInputs[0].desc.type = nvinfer1::DataType::kFLOAT;
    EXPECT_EQ(plugin.configurePlugin(descriptors.dynamicInputs.data(), descriptors.dynamicInputs.size(),
                  descriptors.dynamicOutputs.data(), descriptors.dynamicOutputs.size()),
        -1);

    descriptors = makeDescriptors(7, 256, 256);
    descriptors.dynamicInputs[2].desc.dims = makeDims({1, 2});
    EXPECT_EQ(plugin.configurePlugin(descriptors.dynamicInputs.data(), descriptors.dynamicInputs.size(),
                  descriptors.dynamicOutputs.data(), descriptors.dynamicOutputs.size()),
        -1);

    descriptors = makeDescriptors(7, 256, 256);
    descriptors.runtimeOutputs[0].dims = makeDims({7, 128});
    EXPECT_EQ(plugin.onShapeChange(descriptors.runtimeInputs.data(), descriptors.runtimeInputs.size(),
                  descriptors.runtimeOutputs.data(), descriptors.runtimeOutputs.size()),
        -1);
    EXPECT_EQ(plugin.enqueue(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr), -1);
}

TEST(Fp8BlockGemmPluginTest, ReportsAlignedWorkspaceForMaximumProfile)
{
    constexpr int32_t kM{129};
    constexpr int32_t kN{4096};
    constexpr int32_t kK{2048};
    Fp8BlockGemmPlugin plugin("workspace", kN, kK);
    PluginDescriptors descriptors = makeDescriptors(1, kN, kK, kM);

    EXPECT_EQ(plugin.getWorkspaceSize(descriptors.dynamicInputs.data(), descriptors.dynamicInputs.size(),
                  descriptors.dynamicOutputs.data(), descriptors.dynamicOutputs.size()),
        expectedWorkspaceSize(kM, kK));
    EXPECT_EQ(plugin.getWorkspaceSize(nullptr, descriptors.dynamicInputs.size(), descriptors.dynamicOutputs.data(),
                  descriptors.dynamicOutputs.size()),
        0U);

    descriptors.dynamicInputs[0].max = makeDims({0, kK});
    EXPECT_EQ(plugin.getWorkspaceSize(descriptors.dynamicInputs.data(), descriptors.dynamicInputs.size(),
                  descriptors.dynamicOutputs.data(), descriptors.dynamicOutputs.size()),
        0U);
}

TEST(Fp8BlockGemmPluginTest, GpuNumericsCoverTokenBoundariesAndTailTiles)
{
    if (!isSm110() || !kernels::CuteDslFp8BlockwiseGemmRunner::isSupported(110, 1, kBlockSize, kBlockSize))
    {
        GTEST_SKIP() << "FP8 blockwise CuTe DSL execution requires an enabled SM110 artifact";
    }

    constexpr std::array<int32_t, 10> kTokenCounts{1, 2, 7, 31, 127, 128, 129, 255, 256, 4096};
    for (int32_t const tokenCount : kTokenCounts)
    {
        std::string const name = "m" + std::to_string(tokenCount) + "_n128_k128";
        ShapeCase const shape{name.c_str(), tokenCount, kBlockSize, kBlockSize};
        SCOPED_TRACE(name);
        runNumericalCase(shape, false);
    }
}

TEST(Fp8BlockGemmPluginTest, GpuNumericsCoverModelProjectionShapesAndNonuniformScales)
{
    if (!isSm110() || !kernels::CuteDslFp8BlockwiseGemmRunner::isSupported(110, 1, kBlockSize, kBlockSize))
    {
        GTEST_SKIP() << "FP8 blockwise CuTe DSL execution requires an enabled SM110 artifact";
    }

    constexpr std::array<ShapeCase, 5> kShapes{{
        {"nonuniform_scales", 7, 256, 256},
        {"qkv_projection", 1, 4096, 2048},
        {"attention_output", 1, 2048, 2048},
        {"mlp_gate_up", 1, 6144, 2048},
        {"mlp_down", 1, 2048, 6144},
    }};
    for (ShapeCase const& shape : kShapes)
    {
        SCOPED_TRACE(shape.name);
        runNumericalCase(shape, false);
    }
}

TEST(Fp8BlockGemmPluginTest, RejectsMisalignedWorkspace)
{
    if (!isSm110() || !kernels::CuteDslFp8BlockwiseGemmRunner::isSupported(110, 1, kBlockSize, kBlockSize))
    {
        GTEST_SKIP() << "FP8 blockwise CuTe DSL execution requires an enabled SM110 artifact";
    }

    ShapeCase const shape{"misaligned_workspace", 1, kBlockSize, kBlockSize};
    Fp8BlockGemmPlugin plugin(shape.name, shape.n, shape.k);
    PluginDescriptors descriptors = makeDescriptors(shape.m, shape.n, shape.k);
    DeviceBuffer<half> activation(static_cast<size_t>(shape.m) * shape.k);
    DeviceBuffer<uint8_t> weight(static_cast<size_t>(shape.n) * shape.k);
    DeviceBuffer<float> weightScales(static_cast<size_t>(shape.n / kBlockSize) * (shape.k / kBlockSize));
    DeviceBuffer<half> output(static_cast<size_t>(shape.m) * shape.n);
    size_t const workspaceBytes = expectedWorkspaceSize(shape.m, shape.k);
    DeviceBuffer<uint8_t> workspace(workspaceBytes + 1);
    NonBlockingStream stream;
    std::array<void const*, 3> inputs{activation.get(), weight.get(), weightScales.get()};
    std::array<void*, 1> outputs{output.get()};

    EXPECT_EQ(plugin.enqueue(descriptors.runtimeInputs.data(), descriptors.runtimeOutputs.data(), inputs.data(),
                  outputs.data(), workspace.get() + 1, stream.get()),
        -1);
}

TEST(Fp8BlockGemmPluginTest, CapturesAndReplaysCudaGraph)
{
    if (!isSm110() || !kernels::CuteDslFp8BlockwiseGemmRunner::isSupported(110, 1, kBlockSize, kBlockSize))
    {
        GTEST_SKIP() << "FP8 blockwise CuTe DSL execution requires an enabled SM110 artifact";
    }

    runNumericalCase({"graph_m129_n256_k256", 129, 256, 256}, true);
}

} // namespace
} // namespace trt_edgellm::plugins
