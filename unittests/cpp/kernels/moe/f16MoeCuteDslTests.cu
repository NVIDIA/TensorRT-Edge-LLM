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

#if defined(CUTE_DSL_F16_MOE_ENABLED)

#include "common/cudaUtils.h"
#include "common/tensor.h"
#include "kernels/moe/f16MoeSupportKernels.h"
#include "kernels/moe/f16_cutedsl/cuteDslF16MoeRunner.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <random>
#include <string>
#include <utility>
#include <vector>

using namespace trt_edgellm;

namespace
{

constexpr int32_t kNUM_EXPERTS{128};
constexpr int32_t kNUM_EXPERTS_LARGE{256};
constexpr int32_t kTOP_K{8};
constexpr int32_t kHIDDEN_SIZE{128};
constexpr int32_t kINTER_SIZE{128};
constexpr int32_t kACT_SWIGLU{2};
constexpr int32_t kACT_RELU2{4};
constexpr int32_t kSWIGLU_CHUNK_ROWS{64};
constexpr int32_t kSWIGLU_STORAGE_ROWS{2 * kSWIGLU_CHUNK_ROWS};
constexpr float kATOL{0.05F};
constexpr float kRTOL{0.02F};
constexpr float kROUTER_DENOMINATOR{36.0F};
constexpr float kINPUT_MIN{-0.5F};
constexpr float kINPUT_MAX{0.5F};
constexpr float kFC1_WEIGHT_MIN{-0.2F};
constexpr float kFC1_WEIGHT_MAX{0.2F};
constexpr float kFC2_WEIGHT_MIN{-0.15F};
constexpr float kFC2_WEIGHT_MAX{0.15F};

struct MoeCase
{
    std::string name;
    int32_t numTokens;
    int32_t activationType;
    uint32_t seed;
    int32_t interSize{kINTER_SIZE};
    bool identityFc2{};
    int32_t numExperts{kNUM_EXPERTS};
};

struct CaseData
{
    MoeCase config;
    int32_t fc1Rows;
    std::vector<__half> hiddenStates;
    std::vector<int32_t> topkIds;
    std::vector<float> topkWeights;
    std::vector<__half> fc1Weights;
    std::vector<__half> fc2Weights;
};

struct RunResult
{
    int32_t status;
    std::vector<float> output;
};

size_t flatWeightIndex(int32_t expert, int32_t row, int32_t column, int32_t rows, int32_t columns)
{
    return (static_cast<size_t>(expert) * static_cast<size_t>(rows) + static_cast<size_t>(row))
        * static_cast<size_t>(columns)
        + static_cast<size_t>(column);
}

float roundToFp16(float value)
{
    return __half2float(__float2half_rn(value));
}

float silu(float value)
{
    return value / (1.0F + std::exp(-value));
}

CaseData makeCase(MoeCase config)
{
    int32_t const fc1Rows = config.activationType == kACT_SWIGLU ? 2 * config.interSize : config.interSize;
    CaseData data{std::move(config), fc1Rows};
    int32_t const numExperts = data.config.numExperts;
    size_t const numRoutes = static_cast<size_t>(data.config.numTokens) * kTOP_K;
    data.hiddenStates.resize(static_cast<size_t>(data.config.numTokens) * kHIDDEN_SIZE);
    data.topkIds.resize(numRoutes);
    data.topkWeights.resize(numRoutes);
    data.fc1Weights.resize(static_cast<size_t>(numExperts) * fc1Rows * kHIDDEN_SIZE);
    data.fc2Weights.resize(static_cast<size_t>(numExperts) * kHIDDEN_SIZE * data.config.interSize);

    std::mt19937 generator{data.config.seed};
    std::uniform_real_distribution<float> inputDistribution{kINPUT_MIN, kINPUT_MAX};
    std::uniform_real_distribution<float> fc1Distribution{kFC1_WEIGHT_MIN, kFC1_WEIGHT_MAX};
    std::uniform_real_distribution<float> fc2Distribution{kFC2_WEIGHT_MIN, kFC2_WEIGHT_MAX};

    std::generate(data.hiddenStates.begin(), data.hiddenStates.end(),
        [&] { return __float2half_rn(inputDistribution(generator)); });

    for (int32_t token = 0; token < data.config.numTokens; ++token)
    {
        for (int32_t slot = 0; slot < kTOP_K; ++slot)
        {
            size_t const route = static_cast<size_t>(token) * kTOP_K + slot;
            data.topkIds[route] = (slot * 17 + token * 13) % numExperts;
            data.topkWeights[route] = static_cast<float>(slot + 1) / kROUTER_DENOMINATOR;
        }
    }

    if (data.config.activationType == kACT_SWIGLU)
    {
        for (int32_t expert = 0; expert < numExperts; ++expert)
        {
            for (int32_t intermediate = 0; intermediate < data.config.interSize; ++intermediate)
            {
                int32_t const chunk = intermediate / kSWIGLU_CHUNK_ROWS;
                int32_t const rowInChunk = intermediate % kSWIGLU_CHUNK_ROWS;
                int32_t const upRow = chunk * kSWIGLU_STORAGE_ROWS + rowInChunk;
                int32_t const gateRow = upRow + kSWIGLU_CHUNK_ROWS;
                for (int32_t hidden = 0; hidden < kHIDDEN_SIZE; ++hidden)
                {
                    data.fc1Weights[flatWeightIndex(expert, upRow, hidden, fc1Rows, kHIDDEN_SIZE)]
                        = __float2half_rn(fc1Distribution(generator));
                    data.fc1Weights[flatWeightIndex(expert, gateRow, hidden, fc1Rows, kHIDDEN_SIZE)]
                        = __float2half_rn(fc1Distribution(generator));
                }
            }
        }
    }
    else
    {
        std::generate(data.fc1Weights.begin(), data.fc1Weights.end(),
            [&] { return __float2half_rn(fc1Distribution(generator)); });
    }

    if (data.config.identityFc2)
    {
        std::fill(data.fc2Weights.begin(), data.fc2Weights.end(), __float2half_rn(0.0F));
        for (int32_t expert = 0; expert < numExperts; ++expert)
        {
            for (int32_t index = 0; index < kHIDDEN_SIZE; ++index)
            {
                data.fc2Weights[flatWeightIndex(expert, index, index, kHIDDEN_SIZE, data.config.interSize)]
                    = __float2half_rn(1.0F);
            }
        }
    }
    else
    {
        std::generate(data.fc2Weights.begin(), data.fc2Weights.end(),
            [&] { return __float2half_rn(fc2Distribution(generator)); });
    }
    return data;
}

float fc1Dot(CaseData const& data, int32_t token, int32_t expert, int32_t row)
{
    float accumulator{0.0F};
    for (int32_t hidden = 0; hidden < kHIDDEN_SIZE; ++hidden)
    {
        size_t const inputIndex = static_cast<size_t>(token) * kHIDDEN_SIZE + hidden;
        size_t const weightIndex = flatWeightIndex(expert, row, hidden, data.fc1Rows, kHIDDEN_SIZE);
        accumulator += __half2float(data.hiddenStates[inputIndex]) * __half2float(data.fc1Weights[weightIndex]);
    }
    return accumulator;
}

std::vector<float> computeReference(CaseData const& data)
{
    std::vector<float> output(static_cast<size_t>(data.config.numTokens) * kHIDDEN_SIZE, 0.0F);
    std::vector<float> activated(static_cast<size_t>(data.config.interSize));

    for (int32_t token = 0; token < data.config.numTokens; ++token)
    {
        for (int32_t slot = 0; slot < kTOP_K; ++slot)
        {
            size_t const route = static_cast<size_t>(token) * kTOP_K + slot;
            int32_t const expert = data.topkIds[route];
            if (data.config.activationType == kACT_SWIGLU)
            {
                for (int32_t intermediate = 0; intermediate < data.config.interSize; ++intermediate)
                {
                    int32_t const chunk = intermediate / kSWIGLU_CHUNK_ROWS;
                    int32_t const rowInChunk = intermediate % kSWIGLU_CHUNK_ROWS;
                    int32_t const upRow = chunk * kSWIGLU_STORAGE_ROWS + rowInChunk;
                    int32_t const gateRow = upRow + kSWIGLU_CHUNK_ROWS;
                    float const up = roundToFp16(fc1Dot(data, token, expert, upRow));
                    float const gate = roundToFp16(fc1Dot(data, token, expert, gateRow));
                    activated[intermediate] = roundToFp16(up * silu(gate));
                }
            }
            else
            {
                for (int32_t intermediate = 0; intermediate < data.config.interSize; ++intermediate)
                {
                    float const projection = std::max(roundToFp16(fc1Dot(data, token, expert, intermediate)), 0.0F);
                    activated[intermediate] = roundToFp16(projection * projection);
                }
            }

            for (int32_t hidden = 0; hidden < kHIDDEN_SIZE; ++hidden)
            {
                float accumulator{0.0F};
                for (int32_t intermediate = 0; intermediate < data.config.interSize; ++intermediate)
                {
                    size_t const weightIndex
                        = flatWeightIndex(expert, hidden, intermediate, kHIDDEN_SIZE, data.config.interSize);
                    accumulator += activated[intermediate] * __half2float(data.fc2Weights[weightIndex]);
                }
                output[static_cast<size_t>(token) * kHIDDEN_SIZE + hidden]
                    += data.topkWeights[route] * roundToFp16(accumulator);
            }
        }
        for (int32_t hidden = 0; hidden < kHIDDEN_SIZE; ++hidden)
        {
            size_t const outputIndex = static_cast<size_t>(token) * kHIDDEN_SIZE + hidden;
            output[outputIndex] = roundToFp16(output[outputIndex]);
        }
    }
    return output;
}

RunResult runCase(CaseData const& data)
{
    using rt::DeviceType;
    int32_t const numTokens = data.config.numTokens;
    int32_t const numExperts = data.config.numExperts;
    rt::Tensor hiddenStates({numTokens, kHIDDEN_SIZE}, DeviceType::kGPU, nvinfer1::DataType::kHALF);
    rt::Tensor topkIds({numTokens, kTOP_K}, DeviceType::kGPU, nvinfer1::DataType::kINT32);
    rt::Tensor topkWeights({numTokens, kTOP_K}, DeviceType::kGPU, nvinfer1::DataType::kFLOAT);
    rt::Tensor fc1Weights({numExperts, data.fc1Rows, kHIDDEN_SIZE}, DeviceType::kGPU, nvinfer1::DataType::kHALF);
    rt::Tensor fc2Weights(
        {numExperts, kHIDDEN_SIZE, data.config.interSize}, DeviceType::kGPU, nvinfer1::DataType::kHALF);
    rt::Tensor output({numTokens, kHIDDEN_SIZE}, DeviceType::kGPU, nvinfer1::DataType::kHALF);

    CUDA_CHECK(cudaMemcpy(hiddenStates.rawPointer(), data.hiddenStates.data(),
        data.hiddenStates.size() * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        topkIds.rawPointer(), data.topkIds.data(), data.topkIds.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(topkWeights.rawPointer(), data.topkWeights.data(), data.topkWeights.size() * sizeof(float),
        cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(fc1Weights.rawPointer(), data.fc1Weights.data(), data.fc1Weights.size() * sizeof(__half),
        cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(fc2Weights.rawPointer(), data.fc2Weights.data(), data.fc2Weights.size() * sizeof(__half),
        cudaMemcpyHostToDevice));

    size_t const workspaceBytes = CuteDslF16MoeRunner::getWorkspaceSize(
        numTokens * kTOP_K, numExperts, kHIDDEN_SIZE, data.config.interSize, data.config.activationType);
    EXPECT_GT(workspaceBytes, 0U);
    rt::Tensor workspace({static_cast<int64_t>(workspaceBytes)}, DeviceType::kGPU, nvinfer1::DataType::kINT8);

    CuteDslF16MoeParams params{};
    params.numTokens = numTokens;
    params.numExperts = numExperts;
    params.topK = kTOP_K;
    params.hiddenSize = kHIDDEN_SIZE;
    params.moeInterSize = data.config.interSize;
    params.activationType = data.config.activationType;
    params.persistentBlockCount = CuteDslF16MoeRunner::getPersistentBlockCount();
    params.hiddenStates = hiddenStates.rawPointer();
    params.topkIds = topkIds.dataPointer<int32_t>();
    params.topkWeights = topkWeights.dataPointer<float>();
    params.fc1Weights = fc1Weights.rawPointer();
    params.fc2Weights = fc2Weights.rawPointer();
    params.output = output.rawPointer();

    cudaStream_t stream{};
    int32_t const status = CuteDslF16MoeRunner::run(params, workspace.rawPointer(), stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<__half> outputFp16(static_cast<size_t>(numTokens) * kHIDDEN_SIZE);
    CUDA_CHECK(
        cudaMemcpy(outputFp16.data(), output.rawPointer(), outputFp16.size() * sizeof(__half), cudaMemcpyDeviceToHost));
    std::vector<float> outputFp32(outputFp16.size());
    std::transform(
        outputFp16.begin(), outputFp16.end(), outputFp32.begin(), [](__half value) { return __half2float(value); });
    return {status, std::move(outputFp32)};
}

std::vector<MoeCase> testCases()
{
    return {
        {"decode_swiglu", 1, kACT_SWIGLU, 0xF160001U},
        {"prefill_swiglu", 8, kACT_SWIGLU, 0xF160008U},
        {"decode_relu2", 1, kACT_RELU2, 0xF164001U},
        {"prefill_relu2", 8, kACT_RELU2, 0xF164008U},
        {"prefill_swiglu_identity_fc2", 8, kACT_SWIGLU, 0xF162008U, kINTER_SIZE, true},
        {"prefill_relu2_identity_fc2", 8, kACT_RELU2, 0xF164208U, kINTER_SIZE, true},
        {"prefill_swiglu_i64", 8, kACT_SWIGLU, 0xF166408U, 64},
        {"decode_swiglu_e256", 1, kACT_SWIGLU, 0xF168001U, kINTER_SIZE, false, kNUM_EXPERTS_LARGE},
        {"prefill_swiglu_e256", 8, kACT_SWIGLU, 0xF168008U, kINTER_SIZE, false, kNUM_EXPERTS_LARGE},
        {"prefill_relu2_e256", 8, kACT_RELU2, 0xF168108U, kINTER_SIZE, false, kNUM_EXPERTS_LARGE},
    };
}

void validateRoutingCase(int32_t numTokens, int32_t topK, int32_t numExperts)
{
    using rt::DeviceType;
    constexpr int32_t kTEST_HIDDEN_SIZE{8};
    constexpr int32_t kTEST_INTER_SIZE{8};
    constexpr int32_t kTEST_FC1_N{2 * kTEST_INTER_SIZE};
    constexpr int32_t kSHAPE_VALUES_PER_EXPERT{4};
    constexpr int32_t kSTRIDE_VALUES_PER_EXPERT{6};
    constexpr int32_t kADDRESS_VALUES_PER_EXPERT{3};
    int32_t const routedRows = numTokens * topK;
    SCOPED_TRACE(
        ::testing::Message() << "numTokens=" << numTokens << ", topK=" << topK << ", numExperts=" << numExperts);

    std::vector<int32_t> topkIdsHost(routedRows);
    std::vector<int32_t> expectedCounts(numExperts, 0);
    for (int32_t token = 0; token < numTokens; ++token)
    {
        for (int32_t slot = 0; slot < topK; ++slot)
        {
            int32_t const expandedRow = token * topK + slot;
            int32_t const expert = (token * 13 + slot * 17) % numExperts;
            topkIdsHost[expandedRow] = expert;
            ++expectedCounts[expert];
        }
    }
    std::vector<int32_t> expectedOffsets(numExperts + 1, 0);
    for (int32_t expert = 0; expert < numExperts; ++expert)
    {
        expectedOffsets[expert + 1] = expectedOffsets[expert] + expectedCounts[expert];
    }

    rt::Tensor topkIds({numTokens, topK}, DeviceType::kGPU, nvinfer1::DataType::kINT32);
    rt::Tensor expertCounts({numExperts}, DeviceType::kGPU, nvinfer1::DataType::kINT32);
    rt::Tensor expertOffsets({numExperts + 1}, DeviceType::kGPU, nvinfer1::DataType::kINT32);
    rt::Tensor expertWriteOffsets({numExperts}, DeviceType::kGPU, nvinfer1::DataType::kINT32);
    rt::Tensor sortedToExpanded({routedRows}, DeviceType::kGPU, nvinfer1::DataType::kINT32);
    rt::Tensor expandedToSorted({routedRows}, DeviceType::kGPU, nvinfer1::DataType::kINT32);
    rt::Tensor fc1ProblemShapes({numExperts, kSHAPE_VALUES_PER_EXPERT}, DeviceType::kGPU, nvinfer1::DataType::kINT32);
    rt::Tensor fc1Strides({numExperts, kSTRIDE_VALUES_PER_EXPERT}, DeviceType::kGPU, nvinfer1::DataType::kINT32);
    rt::Tensor fc1Addresses({numExperts, kADDRESS_VALUES_PER_EXPERT}, DeviceType::kGPU, nvinfer1::DataType::kINT64);
    rt::Tensor fc2ProblemShapes({numExperts, kSHAPE_VALUES_PER_EXPERT}, DeviceType::kGPU, nvinfer1::DataType::kINT32);
    rt::Tensor fc2Strides({numExperts, kSTRIDE_VALUES_PER_EXPERT}, DeviceType::kGPU, nvinfer1::DataType::kINT32);
    rt::Tensor fc2Addresses({numExperts, kADDRESS_VALUES_PER_EXPERT}, DeviceType::kGPU, nvinfer1::DataType::kINT64);
    rt::Tensor gatheredInput({routedRows, kTEST_HIDDEN_SIZE}, DeviceType::kGPU, nvinfer1::DataType::kHALF);
    rt::Tensor fc1Weights({numExperts, kTEST_FC1_N, kTEST_HIDDEN_SIZE}, DeviceType::kGPU, nvinfer1::DataType::kHALF);
    rt::Tensor rawFc1({routedRows, kTEST_FC1_N}, DeviceType::kGPU, nvinfer1::DataType::kHALF);
    rt::Tensor activatedFc1({routedRows, kTEST_INTER_SIZE}, DeviceType::kGPU, nvinfer1::DataType::kHALF);
    rt::Tensor fc2Weights(
        {numExperts, kTEST_HIDDEN_SIZE, kTEST_INTER_SIZE}, DeviceType::kGPU, nvinfer1::DataType::kHALF);
    rt::Tensor routedFc2({routedRows, kTEST_HIDDEN_SIZE}, DeviceType::kGPU, nvinfer1::DataType::kHALF);

    CUDA_CHECK(cudaMemcpy(
        topkIds.rawPointer(), topkIdsHost.data(), topkIdsHost.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(sortedToExpanded.rawPointer(), -1, static_cast<size_t>(routedRows) * sizeof(int32_t)));
    CUDA_CHECK(cudaMemset(expandedToSorted.rawPointer(), -1, static_cast<size_t>(routedRows) * sizeof(int32_t)));

    kernel::F16MoeRoutingBuffers const routingBuffers{expertCounts.dataPointer<int32_t>(),
        expertOffsets.dataPointer<int32_t>(), expertWriteOffsets.dataPointer<int32_t>(),
        sortedToExpanded.dataPointer<int32_t>(), expandedToSorted.dataPointer<int32_t>()};
    kernel::F16MoeGemmMetadata const fc1Metadata{fc1ProblemShapes.dataPointer<int32_t>(),
        fc1Strides.dataPointer<int32_t>(), fc1Addresses.dataPointer<int64_t>()};
    kernel::F16MoeGemmMetadata const fc2Metadata{fc2ProblemShapes.dataPointer<int32_t>(),
        fc2Strides.dataPointer<int32_t>(), fc2Addresses.dataPointer<int64_t>()};
    kernel::F16MoeGemmSetup const fc1Setup{fc1Metadata, gatheredInput.rawPointer(), fc1Weights.rawPointer(),
        rawFc1.rawPointer(), kTEST_FC1_N, kTEST_HIDDEN_SIZE};
    kernel::F16MoeGemmSetup const fc2Setup{fc2Metadata, activatedFc1.rawPointer(), fc2Weights.rawPointer(),
        routedFc2.rawPointer(), kTEST_HIDDEN_SIZE, kTEST_INTER_SIZE};
    cudaStream_t stream{};
    CUDA_CHECK(kernel::buildF16MoeRoutingAndGemmMetadata(
        routingBuffers, fc1Setup, fc2Setup, topkIds.dataPointer<int32_t>(), numTokens, topK, numExperts, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<int32_t> countsHost(numExperts);
    std::vector<int32_t> offsetsHost(numExperts + 1);
    std::vector<int32_t> writeOffsetsHost(numExperts);
    std::vector<int32_t> sortedToExpandedHost(routedRows);
    std::vector<int32_t> expandedToSortedHost(routedRows);
    std::vector<int32_t> fc1ShapesHost(static_cast<size_t>(numExperts) * kSHAPE_VALUES_PER_EXPERT);
    std::vector<int32_t> fc2ShapesHost(static_cast<size_t>(numExperts) * kSHAPE_VALUES_PER_EXPERT);
    CUDA_CHECK(cudaMemcpy(
        countsHost.data(), expertCounts.rawPointer(), countsHost.size() * sizeof(int32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(
        offsetsHost.data(), expertOffsets.rawPointer(), offsetsHost.size() * sizeof(int32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(writeOffsetsHost.data(), expertWriteOffsets.rawPointer(),
        writeOffsetsHost.size() * sizeof(int32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(sortedToExpandedHost.data(), sortedToExpanded.rawPointer(),
        sortedToExpandedHost.size() * sizeof(int32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(expandedToSortedHost.data(), expandedToSorted.rawPointer(),
        expandedToSortedHost.size() * sizeof(int32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(fc1ShapesHost.data(), fc1ProblemShapes.rawPointer(), fc1ShapesHost.size() * sizeof(int32_t),
        cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(fc2ShapesHost.data(), fc2ProblemShapes.rawPointer(), fc2ShapesHost.size() * sizeof(int32_t),
        cudaMemcpyDeviceToHost));

    EXPECT_EQ(countsHost, expectedCounts);
    EXPECT_EQ(offsetsHost, expectedOffsets);
    for (int32_t expert = 0; expert < numExperts; ++expert)
    {
        EXPECT_EQ(writeOffsetsHost[expert], expectedOffsets[expert + 1]);
        size_t const shapeBase = static_cast<size_t>(expert) * kSHAPE_VALUES_PER_EXPERT;
        EXPECT_EQ(fc1ShapesHost[shapeBase], expectedCounts[expert]);
        EXPECT_EQ(fc2ShapesHost[shapeBase], expectedCounts[expert]);
    }
    for (int32_t expandedRow = 0; expandedRow < routedRows; ++expandedRow)
    {
        int32_t const expert = topkIdsHost[expandedRow];
        int32_t const sortedRow = expandedToSortedHost[expandedRow];
        ASSERT_GE(sortedRow, expectedOffsets[expert]);
        ASSERT_LT(sortedRow, expectedOffsets[expert + 1]);
        EXPECT_EQ(sortedToExpandedHost[sortedRow], expandedRow);
    }
}

} // namespace

// Validate the device-resident routing and dual-GEMM descriptor ABI independently of AOT execution.
// Irregular expert counts, including empty experts, catch FC1/FC2 shape, stride, and address
// cross-wiring directly instead of only surfacing as an end-to-end numerical mismatch.
TEST(F16MoeCuteDslTest, routingAndDualGemmMetadata)
{
    using rt::DeviceType;
    constexpr int32_t kTEST_NUM_EXPERTS{6};
    constexpr int32_t kTEST_NUM_TOKENS{3};
    constexpr int32_t kTEST_TOP_K{4};
    constexpr int32_t kTEST_HIDDEN_SIZE{128};
    constexpr int32_t kTEST_INTER_SIZE{64};
    constexpr int32_t kSHAPE_VALUES_PER_EXPERT{4};
    constexpr int32_t kSTRIDE_VALUES_PER_EXPERT{6};
    constexpr int32_t kADDRESS_VALUES_PER_EXPERT{3};
    constexpr int32_t kFC1_N{2 * kTEST_INTER_SIZE};
    constexpr int32_t kROUTED_ROWS{kTEST_NUM_TOKENS * kTEST_TOP_K};
    constexpr int64_t kHALF_BYTES{static_cast<int64_t>(sizeof(__half))};
    constexpr int32_t kMETADATA_EXPERTS[]{0, 3, 5};

    std::vector<int32_t> const topkIdsHost{2, 0, 2, 5, 2, 5, 0, 2, 5, 2, 1, 0};
    std::vector<int32_t> const expectedCounts{3, 1, 5, 0, 0, 3};
    std::vector<int32_t> const expectedOffsets{0, 3, 4, 9, 9, 9, 12};
    ASSERT_EQ(topkIdsHost.size(), static_cast<size_t>(kROUTED_ROWS));

    rt::Tensor topkIds({kTEST_NUM_TOKENS, kTEST_TOP_K}, DeviceType::kGPU, nvinfer1::DataType::kINT32);
    rt::Tensor expertCounts({kTEST_NUM_EXPERTS}, DeviceType::kGPU, nvinfer1::DataType::kINT32);
    rt::Tensor expertOffsets({kTEST_NUM_EXPERTS + 1}, DeviceType::kGPU, nvinfer1::DataType::kINT32);
    rt::Tensor expertWriteOffsets({kTEST_NUM_EXPERTS}, DeviceType::kGPU, nvinfer1::DataType::kINT32);
    rt::Tensor sortedToExpanded({kROUTED_ROWS}, DeviceType::kGPU, nvinfer1::DataType::kINT32);
    rt::Tensor expandedToSorted({kROUTED_ROWS}, DeviceType::kGPU, nvinfer1::DataType::kINT32);

    rt::Tensor fc1ProblemShapes(
        {kTEST_NUM_EXPERTS, kSHAPE_VALUES_PER_EXPERT}, DeviceType::kGPU, nvinfer1::DataType::kINT32);
    rt::Tensor fc1Strides({kTEST_NUM_EXPERTS, kSTRIDE_VALUES_PER_EXPERT}, DeviceType::kGPU, nvinfer1::DataType::kINT32);
    rt::Tensor fc1Addresses(
        {kTEST_NUM_EXPERTS, kADDRESS_VALUES_PER_EXPERT}, DeviceType::kGPU, nvinfer1::DataType::kINT64);
    rt::Tensor fc2ProblemShapes(
        {kTEST_NUM_EXPERTS, kSHAPE_VALUES_PER_EXPERT}, DeviceType::kGPU, nvinfer1::DataType::kINT32);
    rt::Tensor fc2Strides({kTEST_NUM_EXPERTS, kSTRIDE_VALUES_PER_EXPERT}, DeviceType::kGPU, nvinfer1::DataType::kINT32);
    rt::Tensor fc2Addresses(
        {kTEST_NUM_EXPERTS, kADDRESS_VALUES_PER_EXPERT}, DeviceType::kGPU, nvinfer1::DataType::kINT64);

    rt::Tensor gatheredInput({kROUTED_ROWS, kTEST_HIDDEN_SIZE}, DeviceType::kGPU, nvinfer1::DataType::kHALF);
    rt::Tensor fc1Weights({kTEST_NUM_EXPERTS, kFC1_N, kTEST_HIDDEN_SIZE}, DeviceType::kGPU, nvinfer1::DataType::kHALF);
    rt::Tensor rawFc1({kROUTED_ROWS, kFC1_N}, DeviceType::kGPU, nvinfer1::DataType::kHALF);
    rt::Tensor activatedFc1({kROUTED_ROWS, kTEST_INTER_SIZE}, DeviceType::kGPU, nvinfer1::DataType::kHALF);
    rt::Tensor fc2Weights(
        {kTEST_NUM_EXPERTS, kTEST_HIDDEN_SIZE, kTEST_INTER_SIZE}, DeviceType::kGPU, nvinfer1::DataType::kHALF);
    rt::Tensor routedFc2({kROUTED_ROWS, kTEST_HIDDEN_SIZE}, DeviceType::kGPU, nvinfer1::DataType::kHALF);

    CUDA_CHECK(cudaMemcpy(
        topkIds.rawPointer(), topkIdsHost.data(), topkIdsHost.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(sortedToExpanded.rawPointer(), -1, static_cast<size_t>(kROUTED_ROWS) * sizeof(int32_t)));
    CUDA_CHECK(cudaMemset(expandedToSorted.rawPointer(), -1, static_cast<size_t>(kROUTED_ROWS) * sizeof(int32_t)));

    kernel::F16MoeRoutingBuffers const routingBuffers{expertCounts.dataPointer<int32_t>(),
        expertOffsets.dataPointer<int32_t>(), expertWriteOffsets.dataPointer<int32_t>(),
        sortedToExpanded.dataPointer<int32_t>(), expandedToSorted.dataPointer<int32_t>()};
    kernel::F16MoeGemmMetadata const fc1Metadata{fc1ProblemShapes.dataPointer<int32_t>(),
        fc1Strides.dataPointer<int32_t>(), fc1Addresses.dataPointer<int64_t>()};
    kernel::F16MoeGemmMetadata const fc2Metadata{fc2ProblemShapes.dataPointer<int32_t>(),
        fc2Strides.dataPointer<int32_t>(), fc2Addresses.dataPointer<int64_t>()};
    kernel::F16MoeGemmSetup const fc1Setup{fc1Metadata, gatheredInput.rawPointer(), fc1Weights.rawPointer(),
        rawFc1.rawPointer(), kFC1_N, kTEST_HIDDEN_SIZE};
    kernel::F16MoeGemmSetup const fc2Setup{fc2Metadata, activatedFc1.rawPointer(), fc2Weights.rawPointer(),
        routedFc2.rawPointer(), kTEST_HIDDEN_SIZE, kTEST_INTER_SIZE};
    cudaStream_t stream{};
    CUDA_CHECK(kernel::buildF16MoeRoutingAndGemmMetadata(routingBuffers, fc1Setup, fc2Setup,
        topkIds.dataPointer<int32_t>(), kTEST_NUM_TOKENS, kTEST_TOP_K, kTEST_NUM_EXPERTS, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<int32_t> countsHost(kTEST_NUM_EXPERTS);
    std::vector<int32_t> offsetsHost(kTEST_NUM_EXPERTS + 1);
    std::vector<int32_t> sortedToExpandedHost(kROUTED_ROWS);
    std::vector<int32_t> expandedToSortedHost(kROUTED_ROWS);
    CUDA_CHECK(cudaMemcpy(
        countsHost.data(), expertCounts.rawPointer(), countsHost.size() * sizeof(int32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(
        offsetsHost.data(), expertOffsets.rawPointer(), offsetsHost.size() * sizeof(int32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(sortedToExpandedHost.data(), sortedToExpanded.rawPointer(),
        sortedToExpandedHost.size() * sizeof(int32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(expandedToSortedHost.data(), expandedToSorted.rawPointer(),
        expandedToSortedHost.size() * sizeof(int32_t), cudaMemcpyDeviceToHost));

    EXPECT_EQ(countsHost, expectedCounts);
    EXPECT_EQ(offsetsHost, expectedOffsets);

    std::vector<bool> expandedRowsSeen(kROUTED_ROWS, false);
    for (int32_t expert = 0; expert < kTEST_NUM_EXPERTS; ++expert)
    {
        for (int32_t sortedRow = expectedOffsets[expert]; sortedRow < expectedOffsets[expert + 1]; ++sortedRow)
        {
            int32_t const expandedRow = sortedToExpandedHost[sortedRow];
            ASSERT_GE(expandedRow, 0);
            ASSERT_LT(expandedRow, kROUTED_ROWS);
            EXPECT_FALSE(expandedRowsSeen[expandedRow]);
            expandedRowsSeen[expandedRow] = true;
            EXPECT_EQ(topkIdsHost[expandedRow], expert);
            EXPECT_EQ(expandedToSortedHost[expandedRow], sortedRow);
        }
    }
    for (int32_t expandedRow = 0; expandedRow < kROUTED_ROWS; ++expandedRow)
    {
        int32_t const sortedRow = expandedToSortedHost[expandedRow];
        ASSERT_GE(sortedRow, 0);
        ASSERT_LT(sortedRow, kROUTED_ROWS);
        EXPECT_TRUE(expandedRowsSeen[expandedRow]);
        EXPECT_EQ(sortedToExpandedHost[sortedRow], expandedRow);
    }

    std::vector<int32_t> shapesHost(static_cast<size_t>(kTEST_NUM_EXPERTS) * kSHAPE_VALUES_PER_EXPERT);
    std::vector<int32_t> stridesHost(static_cast<size_t>(kTEST_NUM_EXPERTS) * kSTRIDE_VALUES_PER_EXPERT);
    std::vector<int64_t> addressesHost(static_cast<size_t>(kTEST_NUM_EXPERTS) * kADDRESS_VALUES_PER_EXPERT);
    auto validateMetadata
        = [&](char const* stage, void const* problemShapesDevice, void const* stridesDevice,
              void const* addressesDevice, void const* input, void const* weights, void* output, int32_t n, int32_t k) {
              SCOPED_TRACE(stage);
              CUDA_CHECK(cudaMemcpy(
                  shapesHost.data(), problemShapesDevice, shapesHost.size() * sizeof(int32_t), cudaMemcpyDeviceToHost));
              CUDA_CHECK(cudaMemcpy(
                  stridesHost.data(), stridesDevice, stridesHost.size() * sizeof(int32_t), cudaMemcpyDeviceToHost));
              CUDA_CHECK(cudaMemcpy(addressesHost.data(), addressesDevice, addressesHost.size() * sizeof(int64_t),
                  cudaMemcpyDeviceToHost));

              int64_t const inputAddress = static_cast<int64_t>(reinterpret_cast<std::uintptr_t>(input));
              int64_t const weightsAddress = static_cast<int64_t>(reinterpret_cast<std::uintptr_t>(weights));
              int64_t const outputAddress = static_cast<int64_t>(reinterpret_cast<std::uintptr_t>(output));
              // Check the first nonempty, an empty, and the final nonempty expert.
              for (int32_t const expert : kMETADATA_EXPERTS)
              {
                  SCOPED_TRACE(::testing::Message() << "expert=" << expert);
                  size_t const shapeBase = static_cast<size_t>(expert) * kSHAPE_VALUES_PER_EXPERT;
                  EXPECT_EQ(shapesHost[shapeBase], expectedCounts[expert]);
                  EXPECT_EQ(shapesHost[shapeBase + 1], n);
                  EXPECT_EQ(shapesHost[shapeBase + 2], k);
                  EXPECT_EQ(shapesHost[shapeBase + 3], 1);

                  size_t const strideBase = static_cast<size_t>(expert) * kSTRIDE_VALUES_PER_EXPERT;
                  EXPECT_EQ(stridesHost[strideBase], k);
                  EXPECT_EQ(stridesHost[strideBase + 1], 1);
                  EXPECT_EQ(stridesHost[strideBase + 2], k);
                  EXPECT_EQ(stridesHost[strideBase + 3], 1);
                  EXPECT_EQ(stridesHost[strideBase + 4], n);
                  EXPECT_EQ(stridesHost[strideBase + 5], 1);

                  size_t const addressBase = static_cast<size_t>(expert) * kADDRESS_VALUES_PER_EXPERT;
                  int64_t const rowOffset = expectedOffsets[expert];
                  EXPECT_EQ(addressesHost[addressBase], inputAddress + rowOffset * k * kHALF_BYTES);
                  EXPECT_EQ(addressesHost[addressBase + 1],
                      weightsAddress + static_cast<int64_t>(expert) * n * k * kHALF_BYTES);
                  EXPECT_EQ(addressesHost[addressBase + 2], outputAddress + rowOffset * n * kHALF_BYTES);
              }
          };

    validateMetadata("fc1", fc1ProblemShapes.rawPointer(), fc1Strides.rawPointer(), fc1Addresses.rawPointer(),
        gatheredInput.rawPointer(), fc1Weights.rawPointer(), rawFc1.rawPointer(), kFC1_N, kTEST_HIDDEN_SIZE);
    validateMetadata("fc2", fc2ProblemShapes.rawPointer(), fc2Strides.rawPointer(), fc2Addresses.rawPointer(),
        activatedFc1.rawPointer(), fc2Weights.rawPointer(), routedFc2.rawPointer(), kTEST_HIDDEN_SIZE,
        kTEST_INTER_SIZE);
}

TEST(F16MoeCuteDslTest, fusedRoutingAndFallbackBoundary)
{
    constexpr int32_t kTEST_TOKENS{17};
    constexpr int32_t kMAX_FUSED_TOKENS{256};
    for (int32_t const numExperts : {kNUM_EXPERTS, kNUM_EXPERTS_LARGE})
    {
        for (int32_t topK = 1; topK <= kTOP_K; ++topK)
        {
            validateRoutingCase(kTEST_TOKENS, topK, numExperts);
        }
        validateRoutingCase(kMAX_FUSED_TOKENS, kTOP_K, numExperts);
        validateRoutingCase(kMAX_FUSED_TOKENS + 1, kTOP_K, numExperts);
    }
}

TEST(F16MoeCuteDslTest, accuracy)
{
    int32_t const smVersion = getSMVersion();
    bool const supportedHardware = smVersion == 80 || smVersion == 86 || smVersion == 87 || smVersion == 89
        || smVersion == 100 || smVersion == 101 || smVersion == 103 || smVersion == 110 || smVersion == 120
        || smVersion == 121;
    if (!supportedHardware)
    {
        GTEST_SKIP() << "FP16 MoE CuTeDSL runner test has no artifact family for SM" << smVersion;
    }
    ASSERT_GT(CuteDslF16MoeRunner::getPersistentBlockCount(), 0);
    EXPECT_TRUE(CuteDslF16MoeRunner::canImplement(kHIDDEN_SIZE, kINTER_SIZE, kNUM_EXPERTS, 1, smVersion, kACT_SWIGLU));
    EXPECT_TRUE(
        CuteDslF16MoeRunner::canImplement(kHIDDEN_SIZE, kINTER_SIZE, kNUM_EXPERTS, kTOP_K, smVersion, kACT_RELU2));
    EXPECT_TRUE(CuteDslF16MoeRunner::canImplement(
        kHIDDEN_SIZE, kINTER_SIZE, kNUM_EXPERTS_LARGE, kTOP_K, smVersion, kACT_SWIGLU));
    EXPECT_FALSE(CuteDslF16MoeRunner::canImplement(kHIDDEN_SIZE, kINTER_SIZE, 64, kTOP_K, smVersion, kACT_SWIGLU));
    EXPECT_FALSE(CuteDslF16MoeRunner::canImplement(kHIDDEN_SIZE, kINTER_SIZE, 192, kTOP_K, smVersion, kACT_SWIGLU));
    EXPECT_FALSE(CuteDslF16MoeRunner::canImplement(kHIDDEN_SIZE, kINTER_SIZE, kNUM_EXPERTS, kTOP_K, 90, kACT_SWIGLU));
    int32_t const mismatchedArtifactSm = smVersion == 110 ? 100 : 110;
    EXPECT_FALSE(CuteDslF16MoeRunner::canImplement(
        kHIDDEN_SIZE, kINTER_SIZE, kNUM_EXPERTS, kTOP_K, mismatchedArtifactSm, kACT_SWIGLU));
    EXPECT_FALSE(CuteDslF16MoeRunner::canImplement(kHIDDEN_SIZE, kINTER_SIZE, kNUM_EXPERTS, 0, smVersion, kACT_SWIGLU));
    EXPECT_FALSE(
        CuteDslF16MoeRunner::canImplement(kHIDDEN_SIZE, kINTER_SIZE, kNUM_EXPERTS, kTOP_K + 1, smVersion, kACT_SWIGLU));

    for (MoeCase const& config : testCases())
    {
        SCOPED_TRACE(::testing::Message() << "case=" << config.name);
        ASSERT_TRUE(CuteDslF16MoeRunner::canImplement(
            kHIDDEN_SIZE, config.interSize, config.numExperts, kTOP_K, smVersion, config.activationType));
        CaseData const data = makeCase(config);
        std::vector<float> const reference = computeReference(data);
        RunResult const result = runCase(data);
        ASSERT_EQ(result.status, 0);
        ASSERT_EQ(result.output.size(), reference.size());

        bool anyNonzero{false};
        for (size_t index = 0; index < result.output.size(); ++index)
        {
            ASSERT_TRUE(std::isfinite(result.output[index])) << "index=" << index;
            anyNonzero = anyNonzero || result.output[index] != 0.0F;
            float const absoluteError = std::fabs(result.output[index] - reference[index]);
            float const tolerance = kATOL + kRTOL * std::fabs(reference[index]);
            EXPECT_LE(absoluteError, tolerance)
                << "index=" << index << ", actual=" << result.output[index] << ", reference=" << reference[index];
        }
        EXPECT_TRUE(anyNonzero);
    }
}

#endif // defined(CUTE_DSL_F16_MOE_ENABLED)
