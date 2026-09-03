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
#include "runtime/decoding/logitBias.h"
#include "runtime/llmRuntimeUtils.h"
#include "runtime/state/decodingInferenceContext.h"
#include "testUtils.h"

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace trt_edgellm;

namespace
{

using BiasMap = std::unordered_map<int32_t, float>;

rt::LLMGenerationRequest makeRequest(std::vector<BiasMap> slotBiases)
{
    rt::LLMGenerationRequest request{};
    request.requests.resize(slotBiases.size());
    for (size_t i = 0; i < slotBiases.size(); ++i)
    {
        request.requests[i].logitBias = std::move(slotBiases[i]);
    }
    return request;
}

rt::DecodingInferenceContext makeContext(int32_t batchSize, cudaStream_t stream)
{
    rt::DecodingInferenceContext context;
    context.initialize(batchSize, 4, std::nullopt, rt::OptionalInputTensors{}, "", stream);
    return context;
}

void expectDeviceFloats(rt::Tensor const& tensor, std::vector<float> const& expected)
{
    auto const actual = copyDeviceToHost<float>(tensor);
    ASSERT_EQ(actual.size(), expected.size());
    for (size_t i = 0; i < expected.size(); ++i)
    {
        EXPECT_FLOAT_EQ(actual[i], expected[i]) << "flat index " << i;
    }
}

class LogitBiasTest : public ::testing::Test
{
protected:
    static constexpr int32_t kMAX_BATCH_SIZE = 4;

    void SetUp() override
    {
        CUDA_CHECK(cudaStreamCreate(&mStream));
        rt::allocateLogitBias(mLogitBias, kMAX_BATCH_SIZE);
    }

    void TearDown() override
    {
        CUDA_CHECK(cudaStreamDestroy(mStream));
    }

    cudaStream_t mStream{};
    rt::LogitBias mLogitBias;
};

TEST(LogitBiasPolicyTest, DetectsBiasInAnyRequestSlot)
{
    auto request = makeRequest({{}, {}, {{17, 1.0F}}});
    EXPECT_TRUE(rt::hasLogitBias(request));

    request.requests[2].logitBias.clear();
    EXPECT_FALSE(rt::hasLogitBias(request));
    EXPECT_FALSE(rt::hasLogitBias(rt::LLMGenerationRequest{}));
}

TEST_F(LogitBiasTest, EmptySlotsClearPriorUploadAndAreANoop)
{
    auto context = makeContext(2, mStream);

    auto populatedRequest = makeRequest({{{0, 9.0F}}, {{1, -4.0F}}});
    rt::prepareLogitBias(mLogitBias, populatedRequest, context);
    rt::Tensor primingLogits({2, 3}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT, "primingLogits");
    copyHostToDevice<float>(primingLogits, std::vector<float>(6, 0.0F));
    rt::applyLogitBias(mLogitBias, primingLogits, context, mStream);
    CUDA_CHECK(cudaStreamSynchronize(mStream));
    EXPECT_FALSE(context.logitBiasGpuDirty);

    auto request = makeRequest({{}, {}});
    rt::prepareLogitBias(mLogitBias, request, context);

    EXPECT_FALSE(context.hasLogitBias);
    EXPECT_FALSE(context.logitBiasGpuDirty);
    ASSERT_EQ(context.logitBiasPerSlot.size(), 2U);
    EXPECT_TRUE(context.logitBiasPerSlot[0].empty());
    EXPECT_TRUE(context.logitBiasPerSlot[1].empty());

    std::vector<float> const input{1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F};
    rt::Tensor logits({2, 3}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT, "emptyBiasLogits");
    copyHostToDevice(logits, input);

    rt::applyLogitBias(mLogitBias, logits, context, mStream);
    CUDA_CHECK(cudaStreamSynchronize(mStream));
    expectDeviceFloats(logits, input);
}

TEST_F(LogitBiasTest, AppliesFullVocabBiasesAcrossMixedSlotsAndReusesUpload)
{
    auto request = makeRequest({{{1, 2.5F}, {3, -1.0F}}, {}, {{0, 4.0F}}});
    auto context = makeContext(3, mStream);
    rt::prepareLogitBias(mLogitBias, request, context);

    EXPECT_TRUE(context.hasLogitBias);
    EXPECT_TRUE(context.logitBiasGpuDirty);
    ASSERT_EQ(context.logitBiasPerSlot.size(), 3U);
    EXPECT_EQ(context.logitBiasPerSlot[0], request.requests[0].logitBias);
    EXPECT_TRUE(context.logitBiasPerSlot[1].empty());
    EXPECT_EQ(context.logitBiasPerSlot[2], request.requests[2].logitBias);

    std::vector<float> const zeros(12, 0.0F);
    std::vector<float> const expected{0.0F, 2.5F, 0.0F, -1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 4.0F, 0.0F, 0.0F, 0.0F};
    rt::Tensor prefillLogits({3, 4}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT, "mixedPrefillLogits");
    copyHostToDevice(prefillLogits, zeros);

    rt::applyLogitBias(mLogitBias, prefillLogits, context, mStream);
    CUDA_CHECK(cudaStreamSynchronize(mStream));
    EXPECT_FALSE(context.logitBiasGpuDirty);
    EXPECT_EQ(mLogitBias.hostOffsets, (std::vector<int32_t>{0, 2, 2, 3}));
    expectDeviceFloats(prefillLogits, expected);

    rt::Tensor decodeLogits({3, 4}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT, "mixedDecodeLogits");
    copyHostToDevice(decodeLogits, zeros);
    rt::applyLogitBias(mLogitBias, decodeLogits, context, mStream);
    CUDA_CHECK(cudaStreamSynchronize(mStream));
    EXPECT_FALSE(context.logitBiasGpuDirty);
    expectDeviceFloats(decodeLogits, expected);
}

TEST_F(LogitBiasTest, AppliesBiasToEverySpecVerificationRowPerSlot)
{
    // Two slots, one with a bias on row 0, the other with biases on rows 1 and 2.
    auto request = makeRequest({{{1, 2.0F}}, {{0, -1.0F}, {3, 4.0F}}});
    auto context = makeContext(2, mStream);
    rt::prepareLogitBias(mLogitBias, request, context);

    constexpr int32_t kROWS_PER_SLOT = 3;
    rt::Tensor logits({2, kROWS_PER_SLOT, 4}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT, "specRowsLogits");
    copyHostToDevice<float>(logits, std::vector<float>(24, 0.0F));

    rt::applyLogitBiasRepeatedRows(mLogitBias, logits, context, kROWS_PER_SLOT, mStream);
    CUDA_CHECK(cudaStreamSynchronize(mStream));

    // clang-format off
    std::vector<float> const expected{
        // slot 0, verify row 0
        0.0F, 2.0F, 0.0F, 0.0F,
        // slot 0, verify row 1
        0.0F, 2.0F, 0.0F, 0.0F,
        // slot 0, verify row 2
        0.0F, 2.0F, 0.0F, 0.0F,
        // slot 1, verify row 0
        -1.0F, 0.0F, 0.0F, 4.0F,
        // slot 1, verify row 1
        -1.0F, 0.0F, 0.0F, 4.0F,
        // slot 1, verify row 2
        -1.0F, 0.0F, 0.0F, 4.0F,
    };
    // clang-format on
    EXPECT_EQ(logits.getShape().getNumDims(), 3);
    EXPECT_EQ(logits.getShape()[0], 2);
    EXPECT_EQ(logits.getShape()[1], kROWS_PER_SLOT);
    EXPECT_EQ(logits.getShape()[2], 4);
    expectDeviceFloats(logits, expected);
}

TEST_F(LogitBiasTest, MapsFullVocabIdsIntoReducedVocabAndSkipsMissingTokens)
{
    rt::Tensor reducedToFull({3}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32, "reducedToFullVocabMap");
    copyHostToDevice<int32_t>(reducedToFull, {4, 1, 7});
    rt::setLogitBiasVocabMap(mLogitBias, reducedToFull, 8, 3, mStream);
    CUDA_CHECK(cudaStreamSynchronize(mStream));

    auto request = makeRequest({{{1, 2.0F}, {4, -1.5F}, {6, 100.0F}, {7, 3.0F}}});
    auto context = makeContext(1, mStream);
    rt::prepareLogitBias(mLogitBias, request, context);

    ASSERT_EQ(context.logitBiasPerSlot.size(), 1U);
    EXPECT_EQ(context.logitBiasPerSlot[0].size(), 3U);
    EXPECT_FLOAT_EQ(context.logitBiasPerSlot[0].at(0), -1.5F);
    EXPECT_FLOAT_EQ(context.logitBiasPerSlot[0].at(1), 2.0F);
    EXPECT_FLOAT_EQ(context.logitBiasPerSlot[0].at(2), 3.0F);

    rt::Tensor logits({1, 3}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT, "reducedVocabLogits");
    copyHostToDevice<float>(logits, {0.0F, 0.0F, 0.0F});
    rt::applyLogitBias(mLogitBias, logits, context, mStream);
    CUDA_CHECK(cudaStreamSynchronize(mStream));
    expectDeviceFloats(logits, {-1.5F, 2.0F, 3.0F});
}

TEST_F(LogitBiasTest, CompactedSlotsAreReuploadedWithNewOffsets)
{
    auto request = makeRequest({{{0, 1.0F}}, {{1, 2.0F}}, {{2, 3.0F}}});
    auto context = makeContext(3, mStream);
    rt::prepareLogitBias(mLogitBias, request, context);

    rt::Tensor initialLogits({3, 4}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT, "initialLogits");
    copyHostToDevice<float>(initialLogits, std::vector<float>(12, 0.0F));
    rt::applyLogitBias(mLogitBias, initialLogits, context, mStream);
    CUDA_CHECK(cudaStreamSynchronize(mStream));
    EXPECT_FALSE(context.logitBiasGpuDirty);

    std::vector<int32_t> const batchMapping{0, -1, 1};
    rt::compactVector(batchMapping, context.logitBiasPerSlot);
    context.activeBatchSize = 2;
    context.hasLogitBias = std::any_of(context.logitBiasPerSlot.begin(), context.logitBiasPerSlot.end(),
        [](auto const& slotBias) { return !slotBias.empty(); });
    context.logitBiasGpuDirty = context.hasLogitBias;

    rt::Tensor compactedLogits({2, 4}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT, "compactedLogits");
    copyHostToDevice<float>(compactedLogits, std::vector<float>(8, 0.0F));
    rt::applyLogitBias(mLogitBias, compactedLogits, context, mStream);
    CUDA_CHECK(cudaStreamSynchronize(mStream));

    EXPECT_FALSE(context.logitBiasGpuDirty);
    EXPECT_EQ(mLogitBias.hostOffsets, (std::vector<int32_t>{0, 1, 2}));
    expectDeviceFloats(compactedLogits, {1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 3.0F, 0.0F});
}

} // namespace
