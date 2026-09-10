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

#include "common/cudaUtils.h"
#include "kernels/moe/moeSigmoidGroupTopkKernels.h"
#include "references.h"
#include "testUtils.h"
#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>
#include <numeric>
#include <set>
#include <vector>

using namespace trt_edgellm;
using namespace trt_edgellm::kernel;
using namespace nvinfer1;

// ============================================================================
// Test Configuration and Runner
// ============================================================================

struct SigmoidGroupTopkTestConfig
{
    int32_t numTokens;
    int32_t numExperts;
    int32_t topK;
    int32_t nGroup;
    int32_t topkGroup;
    bool normTopkProb = false;
    float routedScalingFactor = 1.0f;
    std::vector<float> const* correctionBias = nullptr;
    float rtol = 1e-4f;
    float atol = 1e-5f;
    std::string description;
};

struct SigmoidGroupTopkTestResult
{
    std::vector<float> weights;
    std::vector<int32_t> indices;
};

class SigmoidGroupTopkTestRunner
{
public:
    static SigmoidGroupTopkTestResult run(
        std::vector<float> const& input, SigmoidGroupTopkTestConfig const& config, cudaStream_t stream = nullptr)
    {
        int32_t const numTokens = config.numTokens;
        int32_t const numExperts = config.numExperts;
        int32_t const topK = config.topK;

        auto gatingOutputDevice = rt::Tensor({numTokens, numExperts}, rt::DeviceType::kGPU, DataType::kFLOAT);
        auto topkWeightsDevice = rt::Tensor({numTokens, topK}, rt::DeviceType::kGPU, DataType::kFLOAT);
        auto topkIndicesDevice = rt::Tensor({numTokens, topK}, rt::DeviceType::kGPU, DataType::kINT32);

        CUDA_CHECK(cudaMemcpy(
            gatingOutputDevice.rawPointer(), input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice));

        std::unique_ptr<rt::Tensor> correctionBiasDevice;
        rt::OptionalInputTensor correctionBiasOpt = std::nullopt;
        if (config.correctionBias != nullptr)
        {
            correctionBiasDevice
                = std::make_unique<rt::Tensor>(rt::Coords{numExperts}, rt::DeviceType::kGPU, DataType::kFLOAT);
            CUDA_CHECK(cudaMemcpy(correctionBiasDevice->rawPointer(), config.correctionBias->data(),
                numExperts * sizeof(float), cudaMemcpyHostToDevice));
            correctionBiasOpt = *correctionBiasDevice;
        }

        moeSigmoidGroupTopk(gatingOutputDevice, topkWeightsDevice, topkIndicesDevice, topK, config.nGroup,
            config.topkGroup, config.normTopkProb, config.routedScalingFactor, stream, correctionBiasOpt);
        CUDA_CHECK(cudaDeviceSynchronize());

        SigmoidGroupTopkTestResult result;
        result.weights = copyDeviceToHost<float>(topkWeightsDevice);
        result.indices = copyDeviceToHost<int32_t>(topkIndicesDevice);

        return result;
    }

    static void runAndVerify(
        std::vector<float> const& input, SigmoidGroupTopkTestConfig const& config, cudaStream_t stream = nullptr)
    {
        std::vector<float> expectedWeights;
        std::vector<int32_t> expectedIndices;
        referenceSigmoidGroupTopk(input, config.correctionBias, expectedWeights, expectedIndices, config.numTokens,
            config.numExperts, config.topK, config.nGroup, config.topkGroup, config.normTopkProb,
            config.routedScalingFactor);

        auto result = run(input, config, stream);

        verify(result, expectedWeights, expectedIndices, config);
    }

    static void verifyWeightsSum(SigmoidGroupTopkTestResult const& result, SigmoidGroupTopkTestConfig const& config)
    {
        for (int32_t t = 0; t < config.numTokens; t++)
        {
            float sum = 0.0f;
            for (int32_t k = 0; k < config.topK; k++)
            {
                sum += result.weights[t * config.topK + k];
            }
            float expected = config.routedScalingFactor;
            EXPECT_TRUE(isclose(sum, expected, config.rtol, config.atol))
                << config.description << " Token " << t << ": weights don't sum to scalingFactor. Sum=" << sum
                << ", expected=" << expected;
        }
    }

private:
    static void verify(SigmoidGroupTopkTestResult const& result, std::vector<float> const& expectedWeights,
        std::vector<int32_t> const& expectedIndices, SigmoidGroupTopkTestConfig const& config)
    {
        for (int32_t t = 0; t < config.numTokens; t++)
        {
            std::vector<std::pair<int32_t, float>> resultPairs(config.topK);
            std::vector<std::pair<int32_t, float>> expectedPairs(config.topK);
            for (int32_t k = 0; k < config.topK; k++)
            {
                int32_t idx = t * config.topK + k;
                resultPairs[k] = {result.indices[idx], result.weights[idx]};
                expectedPairs[k] = {expectedIndices[idx], expectedWeights[idx]};
            }

            // Sort by expert index for order-independent comparison
            std::sort(resultPairs.begin(), resultPairs.end());
            std::sort(expectedPairs.begin(), expectedPairs.end());

            for (int32_t k = 0; k < config.topK; k++)
            {
                EXPECT_EQ(resultPairs[k].first, expectedPairs[k].first)
                    << config.description << " Token " << t << ", TopK " << k << ": index mismatch";
                EXPECT_TRUE(isclose(resultPairs[k].second, expectedPairs[k].second, config.rtol, config.atol))
                    << config.description << " Token " << t << ", TopK " << k << ": weight mismatch. Expected "
                    << expectedPairs[k].second << ", got " << resultPairs[k].second;
            }
        }
    }
};

// ============================================================================
// Test Fixture
// ============================================================================

class SigmoidGroupTopkTest : public ::testing::Test
{
protected:
    cudaStream_t stream = nullptr;

    void SetUp() override
    {
        CUDA_CHECK(cudaStreamCreate(&stream));
    }

    void TearDown() override
    {
        if (stream)
        {
            CUDA_CHECK(cudaStreamDestroy(stream));
            stream = nullptr;
        }
    }
};

// ============================================================================
// Tests
// ============================================================================

TEST_F(SigmoidGroupTopkTest, BasicNemotronH_256E_8G)
{
    // NemotronH config: 256 experts, 8 groups, topk_group=4, top_k=8
    SigmoidGroupTopkTestConfig config{.numTokens = 4,
        .numExperts = 256,
        .topK = 8,
        .nGroup = 8,
        .topkGroup = 4,
        .routedScalingFactor = 1.0f,
        .description = "NemotronH 256E/8G: "};

    std::vector<float> input(config.numTokens * config.numExperts);
    uniformFloatInitialization(input, -2.0f, 2.0f);

    SigmoidGroupTopkTestRunner::runAndVerify(input, config, stream);
}

TEST_F(SigmoidGroupTopkTest, SmallConfig_16E_4G)
{
    SigmoidGroupTopkTestConfig config{.numTokens = 8,
        .numExperts = 16,
        .topK = 4,
        .nGroup = 4,
        .topkGroup = 2,
        .routedScalingFactor = 1.0f,
        .description = "Small 16E/4G: "};

    std::vector<float> input(config.numTokens * config.numExperts);
    uniformFloatInitialization(input, -2.0f, 2.0f);

    SigmoidGroupTopkTestRunner::runAndVerify(input, config, stream);
}

TEST_F(SigmoidGroupTopkTest, CorrectionBias)
{
    int32_t const numExperts = 16;
    int32_t const nGroup = 4;

    // Create input where all logits are zero (sigmoid = 0.5)
    SigmoidGroupTopkTestConfig config{.numTokens = 4,
        .numExperts = numExperts,
        .topK = 4,
        .nGroup = nGroup,
        .topkGroup = 2,
        .routedScalingFactor = 1.0f,
        .description = "CorrectionBias: "};

    std::vector<float> input(config.numTokens * numExperts, 0.0f);

    // Bias strongly favors experts in group 3 (experts 12-15)
    std::vector<float> correctionBias(numExperts, 0.0f);
    correctionBias[12] = 2.0f;
    correctionBias[13] = 1.5f;
    correctionBias[14] = 1.0f;
    correctionBias[15] = 0.5f;
    // Also give some bias to group 1 (experts 4-7)
    correctionBias[4] = 0.8f;
    correctionBias[5] = 0.6f;
    config.correctionBias = &correctionBias;

    auto result = SigmoidGroupTopkTestRunner::run(input, config, stream);

    // With uniform logits (all 0 → sigmoid 0.5), the bias determines group selection.
    // Group 3 has highest group score, so experts from group 3 should be present.
    for (int32_t t = 0; t < config.numTokens; t++)
    {
        std::set<int32_t> selectedExperts;
        for (int32_t k = 0; k < config.topK; k++)
        {
            selectedExperts.insert(result.indices[t * config.topK + k]);
        }
        // Expert 12 should always be selected (highest biased score)
        EXPECT_TRUE(selectedExperts.count(12) > 0)
            << "Token " << t << ": expert 12 should be selected due to highest bias";
    }

    // Also verify against CPU reference
    SigmoidGroupTopkTestRunner::runAndVerify(input, config, stream);
}

TEST_F(SigmoidGroupTopkTest, Renormalization)
{
    SigmoidGroupTopkTestConfig config{.numTokens = 8,
        .numExperts = 32,
        .topK = 4,
        .nGroup = 4,
        .topkGroup = 2,
        .normTopkProb = true,
        .routedScalingFactor = 1.0f,
        .description = "Renormalization: "};

    std::vector<float> input(config.numTokens * config.numExperts);
    uniformFloatInitialization(input, -2.0f, 2.0f);

    auto result = SigmoidGroupTopkTestRunner::run(input, config, stream);

    // After renormalization with scalingFactor=1.0, weights should sum to 1.0
    SigmoidGroupTopkTestRunner::verifyWeightsSum(result, config);
    SigmoidGroupTopkTestRunner::runAndVerify(input, config, stream);
}

TEST_F(SigmoidGroupTopkTest, ScalingFactor)
{
    float const scalingFactor = 2.5f;

    SigmoidGroupTopkTestConfig config{.numTokens = 4,
        .numExperts = 16,
        .topK = 4,
        .nGroup = 4,
        .topkGroup = 2,
        .normTopkProb = false,
        .routedScalingFactor = scalingFactor,
        .description = "ScalingFactor: "};

    std::vector<float> input(config.numTokens * config.numExperts);
    uniformFloatInitialization(input, -1.0f, 1.0f);

    // Run without scaling
    SigmoidGroupTopkTestConfig configNoScale = config;
    configNoScale.routedScalingFactor = 1.0f;
    auto resultNoScale = SigmoidGroupTopkTestRunner::run(input, configNoScale, stream);

    // Run with scaling
    auto resultScaled = SigmoidGroupTopkTestRunner::run(input, config, stream);

    // Verify scaled weights = unscaled weights * scalingFactor
    for (int32_t t = 0; t < config.numTokens; t++)
    {
        for (int32_t k = 0; k < config.topK; k++)
        {
            int32_t const idx = t * config.topK + k;
            float expected = resultNoScale.weights[idx] * scalingFactor;
            EXPECT_TRUE(isclose(resultScaled.weights[idx], expected, 1e-4f, 1e-5f))
                << "Token " << t << ", TopK " << k << ": scaled weight mismatch";
            EXPECT_EQ(resultScaled.indices[idx], resultNoScale.indices[idx])
                << "Token " << t << ", TopK " << k << ": scaling should not change expert selection";
        }
    }

    // Verify against reference
    SigmoidGroupTopkTestRunner::runAndVerify(input, config, stream);
}

TEST_F(SigmoidGroupTopkTest, LargeScale)
{
    // 1024 tokens, 256 experts — NemotronH production scale
    SigmoidGroupTopkTestConfig config{.numTokens = 1024,
        .numExperts = 256,
        .topK = 8,
        .nGroup = 8,
        .topkGroup = 4,
        .normTopkProb = true,
        .routedScalingFactor = 1.0f,
        .description = "LargeScale 1024T/256E: "};

    std::vector<float> input(config.numTokens * config.numExperts);
    uniformFloatInitialization(input, -3.0f, 3.0f);

    SigmoidGroupTopkTestRunner::runAndVerify(input, config, stream);
}

TEST_F(SigmoidGroupTopkTest, SingleToken)
{
    SigmoidGroupTopkTestConfig config{.numTokens = 1,
        .numExperts = 16,
        .topK = 4,
        .nGroup = 4,
        .topkGroup = 2,
        .routedScalingFactor = 1.0f,
        .description = "SingleToken: "};

    std::vector<float> input(config.numExperts);
    uniformFloatInitialization(input, -2.0f, 2.0f);

    SigmoidGroupTopkTestRunner::runAndVerify(input, config, stream);
}

TEST_F(SigmoidGroupTopkTest, AllGroupsSelected)
{
    // When topkGroup == nGroup, all groups are selected — behaves like flat top-K
    SigmoidGroupTopkTestConfig config{.numTokens = 4,
        .numExperts = 16,
        .topK = 4,
        .nGroup = 4,
        .topkGroup = 4,
        .routedScalingFactor = 1.0f,
        .description = "AllGroupsSelected: "};

    std::vector<float> input(config.numTokens * config.numExperts);
    uniformFloatInitialization(input, -2.0f, 2.0f);

    SigmoidGroupTopkTestRunner::runAndVerify(input, config, stream);
}

TEST_F(SigmoidGroupTopkTest, RenormWithScaling)
{
    // Combine renormalization + scaling factor
    float const scalingFactor = 3.0f;
    SigmoidGroupTopkTestConfig config{.numTokens = 8,
        .numExperts = 32,
        .topK = 4,
        .nGroup = 4,
        .topkGroup = 2,
        .normTopkProb = true,
        .routedScalingFactor = scalingFactor,
        .description = "RenormWithScaling: "};

    std::vector<float> input(config.numTokens * config.numExperts);
    uniformFloatInitialization(input, -2.0f, 2.0f);

    auto result = SigmoidGroupTopkTestRunner::run(input, config, stream);

    // After renormalization, weights sum to 1.0, then scaled by scalingFactor
    SigmoidGroupTopkTestRunner::verifyWeightsSum(result, config);
    SigmoidGroupTopkTestRunner::runAndVerify(input, config, stream);
}
