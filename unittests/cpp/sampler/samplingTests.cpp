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

#include "common/checkMacros.h"
#include "common/tensor.h"
#include "references.h"
#include "sampler/diffusionGemma/diffusionGemmaSampling.h"
#include "sampler/sampling.h"
#include "testUtils.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <functional>
#include <gtest/gtest.h>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <set>
#include <sstream>
#include <vector>

using namespace trt_edgellm;

// Test configuration
int32_t const ACCURACY_BATCH_SIZE = 4;
int32_t const ACCURACY_VOCAB_SIZE = 20;
uint64_t const TEST_SEED = 42;

// Test fixture for sampling tests
class SamplingTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Initialize CUDA
        CUDA_CHECK(cudaSetDevice(0));
    }

    void TearDown() override
    {
        // Cleanup is handled by individual tests
    }

    static DiffusionCanvasInitParams makeDiffusionInitParams(int32_t vocabSize, uint64_t randomOffset = 0ULL)
    {
        DiffusionCanvasInitParams params{};
        params.vocabSize = vocabSize;
        params.random.offset = randomOffset;
        return params;
    }

    static DiffusionCanvasUpdateParams makeDiffusionUpdateParams(float entropyThreshold, float entropyBound,
        int32_t stabilityWindow, bool forceAccept, int32_t vocabSize, uint64_t randomSeed = kDefaultDiffusionRandomSeed,
        uint64_t randomOffset = 0ULL)
    {
        DiffusionCanvasUpdateParams params{};
        params.entropyThreshold = entropyThreshold;
        params.entropyBound = entropyBound;
        params.stabilityWindow = stabilityWindow;
        params.forceAccept = forceAccept;
        params.vocabSize = vocabSize;
        params.random.seed = randomSeed;
        params.random.offset = randomOffset;
        return params;
    }

    // Generate deterministic test logits (FP32 only)
    void generateTestLogits(
        rt::Tensor& logitsTensor, std::vector<std::vector<float>>& hostLogits, int batchSize, int vocabSize)
    {
        hostLogits.resize(batchSize);
        std::vector<float> flatHostLogits(batchSize * vocabSize);

        // Generate deterministic but varied logits for testing using testUtils
        for (int b = 0; b < batchSize; ++b)
        {
            hostLogits[b].resize(vocabSize);
            uniformFloatInitialization(hostLogits[b], -2.0f, 2.0f);

            // Add some structure to make testing more interesting
            if (vocabSize <= 20)
            {
                hostLogits[b][0] += 3.0f; // Make token 0 very likely
                hostLogits[b][1] += 2.0f; // Make token 1 second most likely
                for (int v = 0; v < 5; ++v)
                {
                    hostLogits[b][v] += 1.0f; // Make first 5 tokens more likely
                }
            }
            else
            {
                hostLogits[b][0] += 5.0f; // Make token 0 very likely
                for (int v = 0; v < 10; ++v)
                {
                    hostLogits[b][v] += 1.0f; // Make first 10 tokens more likely
                }
            }

            // Ensure minimum differences between logits to prevent numerical instability
            std::sort(hostLogits[b].begin(), hostLogits[b].end(), std::greater<float>());
            for (int v = 1; v < vocabSize; ++v)
            {
                if (std::abs(hostLogits[b][v] - hostLogits[b][v - 1]) < 0.01f)
                {
                    hostLogits[b][v] = hostLogits[b][v - 1] - 0.01f;
                }
            }

            // Shuffle the logits to ensure we're not testing with sorted data
            std::random_device rd;
            std::mt19937 gen(rd());
            std::shuffle(hostLogits[b].begin(), hostLogits[b].end(), gen);

            // Copy to flat array (FP32 only)
            for (int v = 0; v < vocabSize; ++v)
            {
                flatHostLogits[b * vocabSize + v] = hostLogits[b][v];
            }
        }

        // Copy host data to device memory
        copyHostToDevice<float>(logitsTensor, flatHostLogits);
    }

    // Validate sampling results (FP32 only)
    bool validateSamplingResults(std::vector<int32_t> const& gpuResults,
        std::vector<std::vector<float>> const& hostLogits, SamplingParams const& params)
    {
        bool allValid = true;
        for (int b = 0; b < static_cast<int>(gpuResults.size()); ++b)
        {
            std::set<int32_t> allowedTokens;

            if (params.useTopK && params.useTopP)
            {
                // Combined top-k and top-p
                allowedTokens
                    = getCombinedAllowedTokensRef(hostLogits[b], params.topK, params.topP, params.temperature);
            }
            else if (params.useTopK)
            {
                // Top-k only
                allowedTokens = getTopKAllowedTokensRef(hostLogits[b], params.topK);
            }
            else if (params.useTopP)
            {
                // Top-p only
                allowedTokens = getTopPAllowedTokensRef(hostLogits[b], params.topP, params.temperature);
            }

            // Check if token is in allowed set
            if (allowedTokens.count(gpuResults[b]) == 0)
            {
                // Output detailed debug info without throwing
                std::cout << "=== SAMPLING VALIDATION FAILED ===" << std::endl;
                std::cout << "Batch " << b << ": Token " << gpuResults[b] << " not in allowed set" << std::endl;
                std::cout << "Allowed tokens: ";
                for (auto token : allowedTokens)
                {
                    std::cout << token << " ";
                }
                std::cout << std::endl;
                std::cout << "Logits for batch " << b << ": ";
                for (size_t i = 0; i < hostLogits[b].size(); ++i)
                {
                    std::cout << hostLogits[b][i];
                    if (i < hostLogits[b].size() - 1)
                        std::cout << ", ";
                }
                std::cout << std::endl;
                std::cout << "=================================" << std::endl;

                allValid = false;
            }
        }
        return allValid;
    }

    // Validate selectAllTopK results - checks indices and raw values (FP32 only)
    bool validateSelectAllTopKResults(std::vector<int32_t> const& gpuIndices, std::vector<float> const& gpuValues,
        std::vector<std::vector<float>> const& hostInput, int topK, int batchSize, bool checkValues)
    {
        bool allValid = true;
        for (int b = 0; b < batchSize; ++b)
        {
            // Get expected top-K elements (just raw values, no transformation)
            auto expectedResults = returnAllTopKReference(hostInput[b], topK);

            // Check that we got the right number of elements
            int expectedSize = std::min(topK, static_cast<int>(hostInput[b].size()));
            if (static_cast<int>(expectedResults.size()) != expectedSize)
            {
                std::cout << "Wrong number of elements - expected " << expectedSize << ", got "
                          << expectedResults.size() << std::endl;
                allValid = false;
                continue;
            }

            // Check that GPU indices match expected indices and values match
            for (int k = 0; k < expectedSize; ++k)
            {
                if (b * topK + k >= static_cast<int>(gpuIndices.size()))
                {
                    std::cout << "Index out of bounds for gpuIndices at batch " << b << " position " << k << std::endl;
                    allValid = false;
                    continue;
                }

                int32_t gpuIdx = gpuIndices[b * topK + k];

                // Check if the index is within valid range
                if (gpuIdx < 0 || gpuIdx >= static_cast<int>(hostInput[b].size()))
                {
                    std::cout << "Invalid index " << gpuIdx << " at batch " << b << " position " << k << std::endl;
                    allValid = false;
                    continue;
                }

                // Check if this index is in the expected top-K results
                bool found = false;
                for (auto const& expected : expectedResults)
                {
                    if (expected.second == gpuIdx)
                    {
                        found = true;
                        break;
                    }
                }

                if (!found)
                {
                    std::cout << "Index " << gpuIdx << " not found in expected top-K results at batch " << b
                              << " position " << k << std::endl;
                    allValid = false;
                }

                // Check values if requested
                if (checkValues && !gpuValues.empty())
                {
                    float gpuValue = gpuValues[b * topK + k];
                    float expectedValue = hostInput[b][gpuIdx];
                    float relativeError = std::abs(gpuValue - expectedValue) / (std::abs(expectedValue) + 1e-6f);

                    if (relativeError > 1e-5f)
                    {
                        std::cout << "Value mismatch at batch " << b << " position " << k << ": GPU=" << gpuValue
                                  << ", expected=" << expectedValue << std::endl;
                        allValid = false;
                    }
                }
            }
        }
        return allValid;
    }
};

// Test temperature = 0.0f parameter override behavior
TEST_F(SamplingTest, TemperatureZeroParameterOverride)
{
    // Test that when temperature = 0.0f, the SamplingParams constructor
    // correctly overrides topK to 1 and topP to 1.0f regardless of input

    // Test case 1: Correct config (should not trigger warning)
    {
        SamplingParams params1(4, 20, 0.0f, 1, 1.0f);
        EXPECT_EQ(params1.temperature, 0.0f);
        EXPECT_EQ(params1.topK, 1);
        EXPECT_EQ(params1.topP, 1.0f);
        EXPECT_TRUE(params1.useTopK);
        EXPECT_FALSE(params1.useTopP);
    }

    // Test case 2: Incorrect config (should trigger warning and override)
    {
        SamplingParams params2(4, 20, 0.0f, 20, 0.9f);
        EXPECT_EQ(params2.temperature, 0.0f);
        EXPECT_EQ(params2.topK, 1);    // Should be overridden
        EXPECT_EQ(params2.topP, 1.0f); // Should be overridden
        EXPECT_TRUE(params2.useTopK);
        EXPECT_FALSE(params2.useTopP);
    }

    // Test case 3: Another incorrect config
    {
        SamplingParams params3(4, 20, 0.0f, 5, 0.8f);
        EXPECT_EQ(params3.temperature, 0.0f);
        EXPECT_EQ(params3.topK, 1);    // Should be overridden
        EXPECT_EQ(params3.topP, 1.0f); // Should be overridden
        EXPECT_TRUE(params3.useTopK);
        EXPECT_FALSE(params3.useTopP);
    }
}

TEST(SamplingUtilsTest, ShouldUseNonGreedySampling)
{
    EXPECT_FALSE(trt_edgellm::shouldUseNonGreedySampling(1.0f, 0, 1.0f));
    EXPECT_FALSE(trt_edgellm::shouldUseNonGreedySampling(0.7f, 1, 0.95f)); // topK=1 forces greedy
    EXPECT_FALSE(trt_edgellm::shouldUseNonGreedySampling(1.2f, 1, 0.5f));  // topK=1 forces greedy
    EXPECT_TRUE(trt_edgellm::shouldUseNonGreedySampling(0.7f, 0, 1.0f));
    EXPECT_TRUE(trt_edgellm::shouldUseNonGreedySampling(1.2f, 0, 1.0f));
    EXPECT_TRUE(trt_edgellm::shouldUseNonGreedySampling(1.0f, 2, 1.0f));
    EXPECT_TRUE(trt_edgellm::shouldUseNonGreedySampling(1.0f, 0, 0.95f));
}

TEST(SamplingUtilsTest, NearZeroTemperatureForcesGreedySampling)
{
    constexpr float kNEAR_ZERO_TEMPERATURE = 5e-4F;
    constexpr float kNORMALIZATION_THRESHOLD = 1e-3F;

    EXPECT_FALSE(trt_edgellm::shouldUseNonGreedySampling(0.0F, 0, 1.0F));
    EXPECT_FALSE(trt_edgellm::shouldUseNonGreedySampling(kNEAR_ZERO_TEMPERATURE, 2, 1.0F));
    EXPECT_FALSE(trt_edgellm::shouldUseNonGreedySampling(kNEAR_ZERO_TEMPERATURE, 0, 0.95F));
    EXPECT_FALSE(trt_edgellm::shouldUseNonGreedySampling(kNEAR_ZERO_TEMPERATURE, 2, 0.95F));

    // Invalid negative temperatures are not normalized here; SamplingParams rejects them.
    EXPECT_TRUE(trt_edgellm::shouldUseNonGreedySampling(-1.0F, 2, 1.0F));

    // SamplingParams only normalizes temperatures strictly below the threshold.
    EXPECT_TRUE(trt_edgellm::shouldUseNonGreedySampling(kNORMALIZATION_THRESHOLD, 2, 1.0F));
    EXPECT_TRUE(trt_edgellm::shouldUseNonGreedySampling(kNORMALIZATION_THRESHOLD, 0, 0.95F));
}

TEST_F(SamplingTest, ApplyLogitBiasAffectsGreedySamplingPerBatch)
{
    constexpr int32_t kBATCH_SIZE = 2;
    constexpr int32_t kVOCAB_SIZE = 6;
    rt::Tensor logitsTensor({kBATCH_SIZE, kVOCAB_SIZE}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT);
    copyHostToDevice<float>(logitsTensor, {5.0f, 4.0f, 3.0f, 2.0f, 1.0f, 0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});

    rt::Tensor biasTokenIds({2}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);
    rt::Tensor biasValues({2}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT);
    rt::Tensor biasOffsets({kBATCH_SIZE + 1}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);
    copyHostToDevice<int32_t>(biasTokenIds, {3, 0});
    copyHostToDevice<float>(biasValues, {4.5f, 10.0f});
    copyHostToDevice<int32_t>(biasOffsets, {0, 1, 2});

    applyLogitBias(logitsTensor, biasTokenIds, biasValues, biasOffsets, 0);
    CUDA_CHECK(cudaDeviceSynchronize());

    rt::Tensor selectedIndicesTensor({kBATCH_SIZE, 1}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);
    size_t workspaceSize = getSelectAllTopKWorkspaceSize(kBATCH_SIZE, kVOCAB_SIZE, 1);
    rt::Tensor workspaceTensor({static_cast<int64_t>(workspaceSize)}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT8);

    selectAllTopK(logitsTensor, std::nullopt, selectedIndicesTensor, 1, workspaceTensor, 0);
    CUDA_CHECK(cudaDeviceSynchronize());

    auto const gpuResults = copyDeviceToHost<int32_t>(selectedIndicesTensor);
    ASSERT_EQ(gpuResults.size(), 2U);
    EXPECT_EQ(gpuResults[0], 3);
    EXPECT_EQ(gpuResults[1], 0);
}

TEST_F(SamplingTest, ApplyLogitBiasCanBanGreedyTopToken)
{
    constexpr int32_t kBATCH_SIZE = 1;
    constexpr int32_t kVOCAB_SIZE = 4;
    rt::Tensor logitsTensor({kBATCH_SIZE, kVOCAB_SIZE}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT);
    copyHostToDevice<float>(logitsTensor, {10.0f, 9.0f, 8.0f, 7.0f});

    rt::Tensor biasTokenIds({1}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);
    rt::Tensor biasValues({1}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT);
    rt::Tensor biasOffsets({kBATCH_SIZE + 1}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);
    copyHostToDevice<int32_t>(biasTokenIds, {0});
    copyHostToDevice<float>(biasValues, {-100.0f});
    copyHostToDevice<int32_t>(biasOffsets, {0, 1});

    applyLogitBias(logitsTensor, biasTokenIds, biasValues, biasOffsets, 0);
    CUDA_CHECK(cudaDeviceSynchronize());

    rt::Tensor selectedIndicesTensor({kBATCH_SIZE, 1}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);
    size_t workspaceSize = getSelectAllTopKWorkspaceSize(kBATCH_SIZE, kVOCAB_SIZE, 1);
    rt::Tensor workspaceTensor({static_cast<int64_t>(workspaceSize)}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT8);

    selectAllTopK(logitsTensor, std::nullopt, selectedIndicesTensor, 1, workspaceTensor, 0);
    CUDA_CHECK(cudaDeviceSynchronize());

    auto const gpuResults = copyDeviceToHost<int32_t>(selectedIndicesTensor);
    ASSERT_EQ(gpuResults.size(), 1U);
    EXPECT_EQ(gpuResults[0], 1);
}

TEST_F(SamplingTest, ApplyLogitBiasRepeatedRowsUsesOwningBatchSlot)
{
    constexpr int32_t kBATCH_SIZE = 2;
    constexpr int32_t kROWS_PER_SLOT = 2;
    constexpr int32_t kVOCAB_SIZE = 5;
    rt::Tensor logitsTensor(
        {kBATCH_SIZE * kROWS_PER_SLOT, kVOCAB_SIZE}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT);
    copyHostToDevice<float>(logitsTensor, std::vector<float>(kBATCH_SIZE * kROWS_PER_SLOT * kVOCAB_SIZE, 0.0F));

    // CSR bias layout: slot i owns entries [biasOffsets[i], biasOffsets[i + 1]).
    // Each entry adds biasValues[j] to token biasTokenIds[j] for that slot.
    rt::Tensor biasTokenIds({3}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);
    rt::Tensor biasValues({3}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT);
    rt::Tensor biasOffsets({kBATCH_SIZE + 1}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);
    copyHostToDevice<int32_t>(biasTokenIds, {1, 0, 4});
    copyHostToDevice<float>(biasValues, {3.0F, -2.0F, 5.0F});
    copyHostToDevice<int32_t>(biasOffsets, {0, 1, 3});

    applyLogitBiasRepeatedRows(logitsTensor, biasTokenIds, biasValues, biasOffsets, kROWS_PER_SLOT, 0);
    CUDA_CHECK(cudaDeviceSynchronize());

    // clang-format off
    std::vector<float> const expected{
        // flattened row 0, slot 0
        0.0F, 3.0F, 0.0F, 0.0F, 0.0F,
        // flattened row 1, slot 0
        0.0F, 3.0F, 0.0F, 0.0F, 0.0F,
        // flattened row 2, slot 1
        -2.0F, 0.0F, 0.0F, 0.0F, 5.0F,
        // flattened row 3, slot 1
        -2.0F, 0.0F, 0.0F, 0.0F, 5.0F,
    };
    // clang-format on
    auto const actual = copyDeviceToHost<float>(logitsTensor);
    ASSERT_EQ(actual.size(), expected.size());
    for (size_t i = 0; i < expected.size(); ++i)
    {
        EXPECT_FLOAT_EQ(actual[i], expected[i]) << "flat index " << i;
    }
}

TEST_F(SamplingTest, SelectArgmaxAndComputeEntropy)
{
    constexpr int32_t rows = 2;
    constexpr int32_t vocabSize = 4;
    std::vector<float> const hostLogits{
        8.0F,
        0.0F,
        -2.0F,
        -4.0F,
        -1.0F,
        2.0F,
        5.0F,
        4.0F,
    };

    rt::Tensor logits({rows, vocabSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT);
    rt::Tensor indices({rows, 1}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);
    rt::Tensor entropy({rows}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT);
    copyHostToDevice<float>(logits, hostLogits);

    selectArgmaxAndComputeEntropy(logits, indices, entropy, /*temperature=*/1.0F, /*stream=*/0);
    CUDA_CHECK(cudaDeviceSynchronize());

    auto const hostIndices = copyDeviceToHost<int32_t>(indices);
    auto const hostEntropy = copyDeviceToHost<float>(entropy);
    EXPECT_EQ(hostIndices[0], 0);
    EXPECT_EQ(hostIndices[1], 2);

    auto entropyRef = [](std::vector<float> const& logitsRow) {
        float const maxLogit = *std::max_element(logitsRow.begin(), logitsRow.end());
        float sumExp = 0.0F;
        float weightedShift = 0.0F;
        for (float const logit : logitsRow)
        {
            float const shifted = logit - maxLogit;
            float const expShifted = std::exp(shifted);
            sumExp += expShifted;
            weightedShift += expShifted * shifted;
        }
        return std::log(sumExp) - weightedShift / sumExp;
    };

    EXPECT_NEAR(hostEntropy[0], entropyRef({8.0F, 0.0F, -2.0F, -4.0F}), 1e-4F);
    EXPECT_NEAR(hostEntropy[1], entropyRef({-1.0F, 2.0F, 5.0F, 4.0F}), 1e-4F);
}

TEST_F(SamplingTest, DiffusionFusedSamplerMatchesSeparateKernels)
{
    constexpr int32_t rows = 3;
    constexpr int32_t vocabSize = 7;
    std::vector<float> const hostLogits{
        2.0F,
        -1.0F,
        0.5F,
        4.0F,
        -3.0F,
        1.0F,
        0.0F,
        -2.0F,
        3.0F,
        1.5F,
        0.0F,
        5.0F,
        -1.5F,
        2.5F,
        1.0F,
        2.0F,
        6.0F,
        -2.5F,
        0.5F,
        4.5F,
        -0.5F,
    };

    rt::Tensor logits({rows, vocabSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT);
    rt::Tensor separateSampled({rows}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);
    rt::Tensor separateIndices({rows, 1}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);
    rt::Tensor separateEntropy({rows}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT);
    rt::Tensor fusedSampled({rows}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);
    rt::Tensor fusedIndices({rows, 1}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);
    rt::Tensor fusedEntropy({rows}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT);
    copyHostToDevice<float>(logits, hostLogits);

    DiffusionRandomParams random{};
    random.seed = 123ULL;
    random.offset = 17ULL;
    float const temperature = 0.8F;
    selectArgmaxAndComputeEntropy(logits, separateIndices, separateEntropy, temperature, /*stream=*/0);
    sampleDiffusionTokensFromLogits(logits, separateSampled, temperature, /*stream=*/0, random);
    sampleDiffusionTokensAndComputeEntropy(
        logits, fusedSampled, fusedIndices, fusedEntropy, temperature, /*stream=*/0, random);
    CUDA_CHECK(cudaDeviceSynchronize());

    EXPECT_EQ(copyDeviceToHost<int32_t>(fusedSampled), copyDeviceToHost<int32_t>(separateSampled));
    EXPECT_EQ(copyDeviceToHost<int32_t>(fusedIndices), copyDeviceToHost<int32_t>(separateIndices));
    auto const fusedEntropyHost = copyDeviceToHost<float>(fusedEntropy);
    auto const separateEntropyHost = copyDeviceToHost<float>(separateEntropy);
    ASSERT_EQ(fusedEntropyHost.size(), separateEntropyHost.size());
    for (size_t i = 0; i < fusedEntropyHost.size(); ++i)
    {
        EXPECT_NEAR(fusedEntropyHost[i], separateEntropyHost[i], 1e-5F);
    }
}

TEST_F(SamplingTest, DiffusionArgmaxOnlySamplerMatchesFusedArgmax)
{
    constexpr int32_t rows = 3;
    constexpr int32_t vocabSize = 7;
    std::vector<float> const hostLogits{
        2.0F,
        -1.0F,
        0.5F,
        4.0F,
        -3.0F,
        1.0F,
        0.0F,
        -2.0F,
        3.0F,
        1.5F,
        0.0F,
        5.0F,
        -1.5F,
        2.5F,
        1.0F,
        2.0F,
        6.0F,
        -2.5F,
        0.5F,
        4.5F,
        -0.5F,
    };

    rt::Tensor logits({rows, vocabSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT);
    rt::Tensor sampled({rows}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);
    rt::Tensor fusedIndices({rows, 1}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);
    rt::Tensor argmaxOnlyIndices({rows, 1}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);
    rt::Tensor entropy({rows}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT);
    copyHostToDevice<float>(logits, hostLogits);

    DiffusionRandomParams random{};
    random.seed = 123ULL;
    random.offset = 17ULL;
    float const temperature = 0.4F;
    sampleDiffusionTokensAndComputeEntropy(logits, sampled, fusedIndices, entropy, temperature, /*stream=*/0, random);
    selectDiffusionArgmaxFromLogits(logits, argmaxOnlyIndices, temperature, /*stream=*/0);
    CUDA_CHECK(cudaDeviceSynchronize());

    EXPECT_EQ(copyDeviceToHost<int32_t>(argmaxOnlyIndices), copyDeviceToHost<int32_t>(fusedIndices));
}

TEST_F(SamplingTest, DiffusionCanvasSamplerUpdatesOnDevice)
{
    constexpr int32_t batchSize = 2;
    constexpr int32_t canvasLen = 3;
    constexpr int32_t vocabSize = 8;
    constexpr int32_t rows = batchSize * canvasLen;
    std::vector<int32_t> const hostArgmax{1, 0, 2, 3, 2, 0};
    std::vector<int32_t> const hostSampled{6, 5, 4, 3, 2, 1};
    std::vector<float> const hostEntropy{0.01F, 0.20F, 0.20F, 0.01F, 0.20F, 0.20F};

    rt::Tensor argmax({rows, 1}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);
    rt::Tensor sampled({batchSize, canvasLen}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);
    rt::Tensor entropy({rows}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT);
    rt::Tensor canvas({batchSize, canvasLen}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);
    rt::Tensor argmaxCanvas({batchSize, canvasLen}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);
    rt::Tensor previous({batchSize, canvasLen}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);
    rt::Tensor stable({batchSize, canvasLen}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);
    rt::Tensor accepted({batchSize, canvasLen}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT8);
    rt::Tensor prefix({batchSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);
    rt::Tensor commitCanvas({batchSize, 2}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);
    copyHostToDevice<int32_t>(argmax, hostArgmax);
    copyHostToDevice<int32_t>(sampled, hostSampled);
    copyHostToDevice<float>(entropy, hostEntropy);

    initializeDiffusionCanvas(
        canvas, previous, stable, accepted, prefix, makeDiffusionInitParams(vocabSize), /*stream=*/0);
    diffusionSampleAndUpdateCanvas(sampled, argmax, entropy, canvas, argmaxCanvas, previous, stable, accepted, prefix,
        makeDiffusionUpdateParams(
            /*entropyThreshold=*/1e-3F, /*entropyBound=*/0.1F, /*stabilityWindow=*/2, /*forceAccept=*/false, vocabSize),
        /*stream=*/0);
    compactDiffusionCanvas(argmaxCanvas, commitCanvas, batchSize, canvasLen, /*blockLen=*/2, /*stream=*/0);
    CUDA_CHECK(cudaDeviceSynchronize());

    auto const hostCanvas = copyDeviceToHost<int32_t>(canvas);
    auto const hostArgmaxCanvas = copyDeviceToHost<int32_t>(argmaxCanvas);
    auto const hostPrevious = copyDeviceToHost<int32_t>(previous);
    auto const hostStable = copyDeviceToHost<int32_t>(stable);
    auto const hostAccepted = copyDeviceToHost<int8_t>(accepted);
    auto const hostPrefix = copyDeviceToHost<int32_t>(prefix);
    auto const hostCommitCanvas = copyDeviceToHost<int32_t>(commitCanvas);

    EXPECT_EQ(hostArgmaxCanvas, hostArgmax);
    EXPECT_EQ(hostPrevious, hostArgmax);
    EXPECT_EQ(hostStable, std::vector<int32_t>(rows, 1));
    EXPECT_EQ(hostPrefix, (std::vector<int32_t>{0, 0}));
    EXPECT_EQ(static_cast<int32_t>(hostAccepted[0]), 1);
    EXPECT_EQ(static_cast<int32_t>(hostAccepted[1]), 1);
    EXPECT_EQ(static_cast<int32_t>(hostAccepted[2]), 0);
    EXPECT_EQ(static_cast<int32_t>(hostAccepted[3]), 1);
    EXPECT_EQ(static_cast<int32_t>(hostAccepted[4]), 1);
    EXPECT_EQ(static_cast<int32_t>(hostAccepted[5]), 0);

    EXPECT_EQ(hostCanvas[0], hostSampled[0]);
    EXPECT_EQ(hostCanvas[1], hostSampled[1]);
    EXPECT_GE(hostCanvas[2], 0);
    EXPECT_LT(hostCanvas[2], vocabSize);
    EXPECT_EQ(hostCanvas[3], hostSampled[3]);
    EXPECT_EQ(hostCanvas[4], hostSampled[4]);
    EXPECT_GE(hostCanvas[5], 0);
    EXPECT_LT(hostCanvas[5], vocabSize);
    EXPECT_EQ(hostCommitCanvas, (std::vector<int32_t>{1, 0, 3, 2}));
}

TEST_F(SamplingTest, DiffusionCanvasSamplerEntropyBudgetSortsAndIgnoresNonFinite)
{
    constexpr int32_t batchSize = 1;
    constexpr int32_t canvasLen = 5;
    constexpr int32_t vocabSize = 16;
    constexpr int32_t rows = batchSize * canvasLen;
    std::vector<int32_t> const hostArgmax{10, 11, 12, 13, 14};
    std::vector<int32_t> const hostSampled{0, 1, 2, 3, 4};
    std::vector<float> const hostEntropy{
        0.4F,
        0.1F,
        std::numeric_limits<float>::infinity(),
        0.2F,
        0.1F,
    };

    rt::Tensor argmax({rows, 1}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);
    rt::Tensor sampled({batchSize, canvasLen}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);
    rt::Tensor entropy({rows}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT);
    rt::Tensor canvas({batchSize, canvasLen}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);
    rt::Tensor argmaxCanvas({batchSize, canvasLen}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);
    rt::Tensor previous({batchSize, canvasLen}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);
    rt::Tensor stable({batchSize, canvasLen}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);
    rt::Tensor accepted({batchSize, canvasLen}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT8);
    rt::Tensor prefix({batchSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);
    copyHostToDevice<int32_t>(argmax, hostArgmax);
    copyHostToDevice<int32_t>(sampled, hostSampled);
    copyHostToDevice<float>(entropy, hostEntropy);

    initializeDiffusionCanvas(
        canvas, previous, stable, accepted, prefix, makeDiffusionInitParams(vocabSize), /*stream=*/0);
    diffusionSampleAndUpdateCanvas(sampled, argmax, entropy, canvas, argmaxCanvas, previous, stable, accepted, prefix,
        makeDiffusionUpdateParams(
            /*entropyThreshold=*/1e-3F, /*entropyBound=*/0.25F, /*stabilityWindow=*/2, /*forceAccept=*/false,
            vocabSize),
        /*stream=*/0);
    CUDA_CHECK(cudaDeviceSynchronize());

    auto const hostCanvas = copyDeviceToHost<int32_t>(canvas);
    auto const hostAccepted = copyDeviceToHost<int8_t>(accepted);
    auto const hostPrefix = copyDeviceToHost<int32_t>(prefix);

    EXPECT_EQ(hostPrefix, (std::vector<int32_t>{0}));
    EXPECT_EQ(static_cast<int32_t>(hostAccepted[0]), 0);
    EXPECT_EQ(static_cast<int32_t>(hostAccepted[1]), 1);
    EXPECT_EQ(static_cast<int32_t>(hostAccepted[2]), 0);
    EXPECT_EQ(static_cast<int32_t>(hostAccepted[3]), 1);
    EXPECT_EQ(static_cast<int32_t>(hostAccepted[4]), 1);
    EXPECT_EQ(hostCanvas[1], hostSampled[1]);
    EXPECT_EQ(hostCanvas[3], hostSampled[3]);
    EXPECT_EQ(hostCanvas[4], hostSampled[4]);
    EXPECT_GE(hostCanvas[0], 0);
    EXPECT_LT(hostCanvas[0], vocabSize);
    EXPECT_GE(hostCanvas[2], 0);
    EXPECT_LT(hostCanvas[2], vocabSize);
}

TEST_F(SamplingTest, DiffusionCanvasSamplerHonorsValidLengths)
{
    constexpr int32_t batchSize = 2;
    constexpr int32_t canvasLen = 4;
    constexpr int32_t vocabSize = 8;
    constexpr int32_t rows = batchSize * canvasLen;
    std::vector<int32_t> const hostArgmax{1, 2, 3, 4, 5, 6, 7, 0};
    std::vector<int32_t> const hostSampled{1, 2, 3, 4, 5, 6, 7, 0};
    std::vector<float> const hostEntropy{0.001F, 0.001F, 0.001F, 0.001F, 0.001F, 0.001F, 1000.0F, 1000.0F};
    std::vector<int32_t> const hostValidLengths{4, 2};

    rt::Tensor argmax({rows, 1}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);
    rt::Tensor sampled({batchSize, canvasLen}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);
    rt::Tensor entropy({rows}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT);
    rt::Tensor validLengths({batchSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);
    rt::Tensor canvas({batchSize, canvasLen}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);
    rt::Tensor argmaxCanvas({batchSize, canvasLen}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);
    rt::Tensor previous({batchSize, canvasLen}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);
    rt::Tensor stable({batchSize, canvasLen}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);
    rt::Tensor accepted({batchSize, canvasLen}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT8);
    rt::Tensor prefix({batchSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);
    copyHostToDevice<int32_t>(argmax, hostArgmax);
    copyHostToDevice<int32_t>(sampled, hostSampled);
    copyHostToDevice<float>(entropy, hostEntropy);
    copyHostToDevice<int32_t>(validLengths, hostValidLengths);

    initializeDiffusionCanvas(
        canvas, previous, stable, accepted, prefix, makeDiffusionInitParams(vocabSize), /*stream=*/0);
    diffusionSampleAndUpdateCanvas(sampled, argmax, entropy, canvas, argmaxCanvas, previous, stable, accepted, prefix,
        makeDiffusionUpdateParams(
            /*entropyThreshold=*/0.01F, /*entropyBound=*/-1.0F, /*stabilityWindow=*/1, /*forceAccept=*/true, vocabSize),
        /*stream=*/0, rt::OptionalInputTensor{std::cref(validLengths)});
    CUDA_CHECK(cudaDeviceSynchronize());

    auto const hostPrefix = copyDeviceToHost<int32_t>(prefix);
    auto const hostCanvas = copyDeviceToHost<int32_t>(canvas);
    auto const hostArgmaxCanvas = copyDeviceToHost<int32_t>(argmaxCanvas);
    auto const hostAccepted = copyDeviceToHost<int8_t>(accepted);

    EXPECT_EQ(hostPrefix, (std::vector<int32_t>{4, 2}));
    EXPECT_EQ(hostCanvas[0], 1);
    EXPECT_EQ(hostCanvas[1], 2);
    EXPECT_EQ(hostCanvas[2], 3);
    EXPECT_EQ(hostCanvas[3], 4);
    EXPECT_EQ(hostCanvas[4], 5);
    EXPECT_EQ(hostCanvas[5], 6);
    EXPECT_EQ(hostArgmaxCanvas[4], 5);
    EXPECT_EQ(hostArgmaxCanvas[5], 6);
    EXPECT_EQ(static_cast<int32_t>(hostAccepted[6]), 0);
    EXPECT_EQ(static_cast<int32_t>(hostAccepted[7]), 0);
}

TEST_F(SamplingTest, CompactDiffusionCanvasSupportsVariableCommitLengths)
{
    constexpr int32_t batchSize = 3;
    constexpr int32_t canvasLen = 5;
    constexpr int32_t maxBlockLen = 4;
    constexpr int32_t padTokenId = 99;
    std::vector<int32_t> const hostCanvas{
        10,
        11,
        12,
        13,
        14,
        20,
        21,
        22,
        23,
        24,
        30,
        31,
        32,
        33,
        34,
    };
    std::vector<int32_t> const hostCommitLengths{1, 3, 4};

    rt::Tensor canvas({batchSize, canvasLen}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);
    rt::Tensor commitLengths({batchSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);
    rt::Tensor commitCanvas({batchSize, maxBlockLen}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);
    copyHostToDevice<int32_t>(canvas, hostCanvas);
    copyHostToDevice<int32_t>(commitLengths, hostCommitLengths);

    compactDiffusionCanvas(canvas, commitLengths, commitCanvas, batchSize, canvasLen, maxBlockLen, padTokenId,
        /*stream=*/0);
    CUDA_CHECK(cudaDeviceSynchronize());

    auto const hostCommitCanvas = copyDeviceToHost<int32_t>(commitCanvas);
    EXPECT_EQ(hostCommitCanvas,
        (std::vector<int32_t>{
            10,
            99,
            99,
            99,
            20,
            21,
            22,
            99,
            30,
            31,
            32,
            33,
        }));
}

TEST_F(SamplingTest, DiffusionCanvasConvergesAfterStableWindow)
{
    constexpr int32_t batchSize = 1;
    constexpr int32_t canvasLen = 2;
    constexpr int32_t vocabSize = 4;
    constexpr int32_t rows = batchSize * canvasLen;
    std::vector<int32_t> const hostArgmax{0, 1};
    std::vector<float> const hostEntropy{0.001F, 0.001F};

    rt::Tensor argmax({rows, 1}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);
    rt::Tensor sampled({batchSize, canvasLen}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);
    rt::Tensor entropy({rows}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT);
    rt::Tensor canvas({batchSize, canvasLen}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);
    rt::Tensor argmaxCanvas({batchSize, canvasLen}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);
    rt::Tensor previous({batchSize, canvasLen}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);
    rt::Tensor stable({batchSize, canvasLen}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);
    rt::Tensor accepted({batchSize, canvasLen}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT8);
    rt::Tensor prefix({batchSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);
    copyHostToDevice<int32_t>(argmax, hostArgmax);
    copyHostToDevice<int32_t>(sampled, hostArgmax);
    copyHostToDevice<float>(entropy, hostEntropy);

    initializeDiffusionCanvas(
        canvas, previous, stable, accepted, prefix, makeDiffusionInitParams(vocabSize), /*stream=*/0);
    diffusionSampleAndUpdateCanvas(sampled, argmax, entropy, canvas, argmaxCanvas, previous, stable, accepted, prefix,
        makeDiffusionUpdateParams(
            /*entropyThreshold=*/0.01F, /*entropyBound=*/-1.0F, /*stabilityWindow=*/2, /*forceAccept=*/false,
            vocabSize),
        /*stream=*/0);
    CUDA_CHECK(cudaDeviceSynchronize());

    auto hostPrefix = copyDeviceToHost<int32_t>(prefix);
    auto hostStable = copyDeviceToHost<int32_t>(stable);
    EXPECT_EQ(hostPrefix, (std::vector<int32_t>{0}));
    EXPECT_EQ(hostStable, std::vector<int32_t>(rows, 1));

    diffusionSampleAndUpdateCanvas(sampled, argmax, entropy, canvas, argmaxCanvas, previous, stable, accepted, prefix,
        makeDiffusionUpdateParams(
            /*entropyThreshold=*/0.01F, /*entropyBound=*/-1.0F, /*stabilityWindow=*/2, /*forceAccept=*/false,
            vocabSize),
        /*stream=*/0);
    CUDA_CHECK(cudaDeviceSynchronize());

    hostPrefix = copyDeviceToHost<int32_t>(prefix);
    hostStable = copyDeviceToHost<int32_t>(stable);
    EXPECT_EQ(hostPrefix, (std::vector<int32_t>{0}));
    EXPECT_EQ(hostStable, std::vector<int32_t>(rows, 2));

    diffusionSampleAndUpdateCanvas(sampled, argmax, entropy, canvas, argmaxCanvas, previous, stable, accepted, prefix,
        makeDiffusionUpdateParams(
            /*entropyThreshold=*/0.01F, /*entropyBound=*/-1.0F, /*stabilityWindow=*/2, /*forceAccept=*/false,
            vocabSize),
        /*stream=*/0);
    CUDA_CHECK(cudaDeviceSynchronize());

    hostPrefix = copyDeviceToHost<int32_t>(prefix);
    hostStable = copyDeviceToHost<int32_t>(stable);
    auto const hostCanvas = copyDeviceToHost<int32_t>(canvas);
    auto const hostArgmaxCanvas = copyDeviceToHost<int32_t>(argmaxCanvas);
    EXPECT_EQ(hostPrefix, (std::vector<int32_t>{2}));
    EXPECT_EQ(hostStable, std::vector<int32_t>(rows, 3));
    EXPECT_EQ(hostArgmaxCanvas, hostArgmax);
    EXPECT_EQ(hostCanvas, hostArgmax);
}

TEST_F(SamplingTest, DiffusionCanvasStabilityWindowOneRequiresPreviousObservation)
{
    constexpr int32_t batchSize = 1;
    constexpr int32_t canvasLen = 2;
    constexpr int32_t vocabSize = 4;
    constexpr int32_t rows = batchSize * canvasLen;
    std::vector<int32_t> const hostArgmax{0, 1};
    std::vector<float> const hostEntropy{0.001F, 0.001F};

    rt::Tensor argmax({rows, 1}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);
    rt::Tensor sampled({batchSize, canvasLen}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);
    rt::Tensor entropy({rows}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT);
    rt::Tensor canvas({batchSize, canvasLen}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);
    rt::Tensor argmaxCanvas({batchSize, canvasLen}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);
    rt::Tensor previous({batchSize, canvasLen}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);
    rt::Tensor stable({batchSize, canvasLen}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);
    rt::Tensor accepted({batchSize, canvasLen}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT8);
    rt::Tensor prefix({batchSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);
    copyHostToDevice<int32_t>(argmax, hostArgmax);
    copyHostToDevice<int32_t>(sampled, hostArgmax);
    copyHostToDevice<float>(entropy, hostEntropy);

    initializeDiffusionCanvas(
        canvas, previous, stable, accepted, prefix, makeDiffusionInitParams(vocabSize), /*stream=*/0);
    diffusionSampleAndUpdateCanvas(sampled, argmax, entropy, canvas, argmaxCanvas, previous, stable, accepted, prefix,
        makeDiffusionUpdateParams(
            /*entropyThreshold=*/0.01F, /*entropyBound=*/-1.0F, /*stabilityWindow=*/1, /*forceAccept=*/false,
            vocabSize),
        /*stream=*/0);
    CUDA_CHECK(cudaDeviceSynchronize());

    auto hostPrefix = copyDeviceToHost<int32_t>(prefix);
    auto hostStable = copyDeviceToHost<int32_t>(stable);
    EXPECT_EQ(hostPrefix, (std::vector<int32_t>{0}));
    EXPECT_EQ(hostStable, std::vector<int32_t>(rows, 1));

    diffusionSampleAndUpdateCanvas(sampled, argmax, entropy, canvas, argmaxCanvas, previous, stable, accepted, prefix,
        makeDiffusionUpdateParams(
            /*entropyThreshold=*/0.01F, /*entropyBound=*/-1.0F, /*stabilityWindow=*/1, /*forceAccept=*/false,
            vocabSize),
        /*stream=*/0);
    CUDA_CHECK(cudaDeviceSynchronize());

    hostPrefix = copyDeviceToHost<int32_t>(prefix);
    hostStable = copyDeviceToHost<int32_t>(stable);
    auto const hostCanvas = copyDeviceToHost<int32_t>(canvas);
    auto const hostArgmaxCanvas = copyDeviceToHost<int32_t>(argmaxCanvas);
    EXPECT_EQ(hostPrefix, (std::vector<int32_t>{2}));
    EXPECT_EQ(hostStable, std::vector<int32_t>(rows, 2));
    EXPECT_EQ(hostArgmaxCanvas, hostArgmax);
    EXPECT_EQ(hostCanvas, hostArgmax);

    std::vector<int32_t> const changedArgmax{2, 3};
    copyHostToDevice<int32_t>(argmax, changedArgmax);
    copyHostToDevice<int32_t>(sampled, changedArgmax);
    diffusionSampleAndUpdateCanvas(sampled, argmax, entropy, canvas, argmaxCanvas, previous, stable, accepted, prefix,
        makeDiffusionUpdateParams(
            /*entropyThreshold=*/0.01F, /*entropyBound=*/-1.0F, /*stabilityWindow=*/1, /*forceAccept=*/false,
            vocabSize),
        /*stream=*/0);
    CUDA_CHECK(cudaDeviceSynchronize());

    EXPECT_EQ(copyDeviceToHost<int32_t>(prefix), (std::vector<int32_t>{2}));
    EXPECT_EQ(copyDeviceToHost<int32_t>(stable), std::vector<int32_t>(rows, 2));
    EXPECT_EQ(copyDeviceToHost<int32_t>(argmaxCanvas), hostArgmax);
    EXPECT_EQ(copyDeviceToHost<int32_t>(canvas), hostArgmax);
}

// Unified sampling tests (accuracy only)
class SamplingTestSuites : public SamplingTest
{
protected:
    struct TestResult
    {
        std::string methodName;
        int batchSize;
        int vocabSize;
        int topK;
        float topP;
        float temperature;
        bool accuracyPassed;
        std::string errorMessage;
    };

    TestResult runSamplingAccuracyTest(
        std::string const& methodName, int batchSize, int vocabSize, int topK, float topP, float temperature)
    {
        TestResult result;
        result.methodName = methodName;
        result.batchSize = batchSize;
        result.vocabSize = vocabSize;
        result.topK = topK;
        result.topP = topP;
        result.temperature = temperature;
        result.accuracyPassed = true;
        result.errorMessage = "";

        // Create tensors for the test
        rt::Tensor logitsTensor({batchSize, vocabSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT);
        rt::Tensor selectedIndicesTensor({batchSize, 1}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);

        std::vector<std::vector<float>> hostLogits;
        generateTestLogits(logitsTensor, hostLogits, batchSize, vocabSize);

        // Run accuracy test
        SamplingParams params(batchSize, vocabSize, temperature, topK, topP);
        size_t workspaceSize = getTopKtopPSamplingWorkspaceSize(batchSize, vocabSize, params);
        rt::Tensor workspaceTensor(
            {static_cast<int64_t>(workspaceSize)}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT8);

        topKtopPSamplingFromLogits(logitsTensor, selectedIndicesTensor, params, workspaceTensor, 0, TEST_SEED, 0);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Copy results back to host
        auto const gpuResults = copyDeviceToHost<int32_t>(selectedIndicesTensor);

        // Run validation and get result
        bool validationPassed = validateSamplingResults(gpuResults, hostLogits, params);

        // Set result based on validation
        result.accuracyPassed = validationPassed;
        if (!validationPassed)
        {
            result.errorMessage = "Sampling validation failed - check output for details";
        }

        // Single Google Test assertion for comprehensive validation
        EXPECT_TRUE(validationPassed) << "Sampling validation failed for " << methodName
                                      << " with batchSize=" << batchSize << ", vocabSize=" << vocabSize
                                      << ", topK=" << topK << ", topP=" << topP << ", temperature=" << temperature;

        return result;
    }
};

// SelectAllTopK tests - simplified to only test raw value return functionality
class ReturnAllTopKTests : public SamplingTest
{
protected:
    struct TestResult
    {
        std::string methodName;
        int batchSize;
        int vocabSize;
        int topK;
        bool accuracyPassed;
        std::string errorMessage;
    };

    TestResult runReturnAllTopKAccuracyTest(int batchSize, int vocabSize, int topK, bool testValues)
    {
        TestResult result;
        result.methodName = "SelectAllTopK";
        result.batchSize = batchSize;
        result.vocabSize = vocabSize;
        result.topK = topK;
        result.accuracyPassed = true;
        result.errorMessage = "";

        // Create tensors for the test
        rt::Tensor inputTensor({batchSize, vocabSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT);
        rt::Tensor topKIndicesTensor({batchSize, topK}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);
        rt::OptionalOutputTensor topKValuesOptional = std::nullopt;
        rt::Tensor topKValuesTensor;

        // Always test with values when requested
        if (testValues)
        {
            topKValuesTensor = rt::Tensor({batchSize, topK}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT);
            topKValuesOptional = std::ref(topKValuesTensor);
        }

        std::vector<std::vector<float>> hostLogits;

        // Generate test data (logits/raw values)
        generateTestLogits(inputTensor, hostLogits, batchSize, vocabSize);

        // Run test
        size_t workspaceSize = getSelectAllTopKWorkspaceSize(batchSize, vocabSize, topK);
        rt::Tensor workspaceTensor(
            {static_cast<int64_t>(workspaceSize)}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT8);

        selectAllTopK(inputTensor, topKValuesOptional, topKIndicesTensor, topK, workspaceTensor, 0);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Copy results back to host
        auto const gpuIndices = copyDeviceToHost<int32_t>(topKIndicesTensor);

        std::vector<float> gpuValues;
        if (testValues)
        {
            gpuValues = copyDeviceToHost<float>(topKValuesTensor);
        }

        bool validationPassed
            = validateSelectAllTopKResults(gpuIndices, gpuValues, hostLogits, topK, batchSize, testValues);

        // Set result based on validation
        result.accuracyPassed = validationPassed;
        if (!validationPassed)
        {
            result.errorMessage = "SelectAllTopK validation failed - check output for details";
        }

        // Single Google Test assertion for comprehensive validation
        EXPECT_TRUE(validationPassed) << "SelectAllTopK validation failed for batchSize=" << batchSize
                                      << ", vocabSize=" << vocabSize << ", topK=" << topK;

        return result;
    }
};

// Sampling tests
TEST_F(SamplingTestSuites, SamplingAccuracy)
{
    std::vector<SamplingTestSuites::TestResult> accuracyResults;

    // Test configurations
    struct SamplingConfig
    {
        std::string methodName;
        int topK;
        float topP;
        float temperature;
    };

    std::vector<SamplingConfig> configs = {
        {"TopK", 20, 1.0f, 1.0f},
        {"TopK", 50, 1.0f, 1.0f},
        {"TopK", 100, 1.0f, 1.0f},
        {"TopP", 0, 0.9f, 1.0f},
        {"TopP", 0, 0.95f, 1.0f},
        {"TopP", 0, 0.99f, 1.0f},
        {"TopKTopP", 20, 0.9f, 1.0f},
        {"TopKTopP", 50, 0.95f, 1.0f},
        {"TopKTopP", 100, 0.99f, 1.0f},
        {"TopK", 20, 1.0f, 0.5f},
        {"TopK", 20, 1.0f, 1.5f},
        {"TopP", 0, 0.9f, 0.5f},
        {"TopP", 0, 0.9f, 1.5f},
        // Temperature = 0.0f tests - should always pick topK = 1 regardless of config
        {"TempZero", 1, 1.0f, 0.0f},  // Correct config for temperature = 0.0f
        {"TempZero", 20, 0.9f, 0.0f}, // Incorrect config - should be overridden to topK = 1, topP = 1.0f
    };

    // Run accuracy tests with small vocab size
    for (int batchSize : {1, 4})
    {
        for (auto const& config : configs)
        {
            auto result = runSamplingAccuracyTest(
                config.methodName, batchSize, ACCURACY_VOCAB_SIZE, config.topK, config.topP, config.temperature);
            accuracyResults.push_back(result);
        }
    }

    // Print accuracy results table
    std::cout << "\nSampling Accuracy Results (FP32 only):" << std::endl;
    std::cout << "Method   | Batch | AccVocabSize | TopK | TopP  | Temp  | Accuracy" << std::endl;
    std::cout << "---------|-------|--------------|------|-------|-------|----------" << std::endl;

    bool allAccuracyTestsPassed = true;
    std::vector<std::string> accuracyErrorMessages;

    for (auto const& result : accuracyResults)
    {
        std::string topKStr = (result.topK == 0) ? "N/A" : std::to_string(result.topK);

        std::string topPStr;
        if (result.topP == 1.0f)
        {
            topPStr = "N/A";
        }
        else
        {
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(2) << result.topP;
            topPStr = oss.str();
        }

        std::ostringstream tempOss;
        tempOss << std::fixed << std::setprecision(2) << result.temperature;
        std::string tempStr = tempOss.str();

        std::string accuracyStr = result.accuracyPassed ? "PASS" : "FAIL";

        if (!result.accuracyPassed)
        {
            allAccuracyTestsPassed = false;
            accuracyErrorMessages.push_back(result.errorMessage);
        }

        std::cout << std::setw(8) << result.methodName << " | " << std::setw(5) << result.batchSize << " | "
                  << std::setw(12) << result.vocabSize << " | " << std::setw(4) << topKStr << " | " << std::setw(5)
                  << topPStr << " | " << std::setw(5) << tempStr << " | " << std::setw(8) << accuracyStr << std::endl;
    }

    // Print summary
    if (allAccuracyTestsPassed)
    {
        std::cout << "\nAll sampling accuracy tests PASSED!" << std::endl;
    }
    else
    {
        std::cout << "\nSome sampling accuracy tests FAILED!" << std::endl;
        std::cout << "Error details:" << std::endl;
        for (auto const& error : accuracyErrorMessages)
        {
            std::cout << "  - " << error << std::endl;
        }
    }
}

// SelectAllTopK tests - simplified to only test returning indices and raw values
TEST_F(ReturnAllTopKTests, SelectAllTopKAccuracy)
{
    std::vector<ReturnAllTopKTests::TestResult> accuracyResults;

    // Simplified test configurations - just test different topK values and batch sizes
    // Boolean parameters are no longer tested as they are ignored
    std::vector<int> topKValues = {5, 10, 20};

    // Run accuracy tests with small vocab size
    for (int batchSize : {1, 4})
    {
        for (int topK : topKValues)
        {
            // Test with values
            auto result = runReturnAllTopKAccuracyTest(batchSize, ACCURACY_VOCAB_SIZE, topK, true);
            accuracyResults.push_back(result);
        }
    }

    // Print accuracy results table
    std::cout << "\nSelectAllTopK Accuracy Results (FP32 only - Raw Values):" << std::endl;
    std::cout << "Batch | TopK | AccVocabSize | Accuracy" << std::endl;
    std::cout << "------|------|--------------|----------" << std::endl;

    bool allAccuracyTestsPassed = true;
    std::vector<std::string> accuracyErrorMessages;

    for (auto const& result : accuracyResults)
    {
        std::string accuracyStr = result.accuracyPassed ? "PASS" : "FAIL";

        if (!result.accuracyPassed)
        {
            allAccuracyTestsPassed = false;
            accuracyErrorMessages.push_back(result.errorMessage);
        }

        std::cout << std::setw(5) << result.batchSize << " | " << std::setw(4) << result.topK << " | " << std::setw(12)
                  << result.vocabSize << " | " << std::setw(8) << accuracyStr << std::endl;
    }

    // Print summary
    if (allAccuracyTestsPassed)
    {
        std::cout << "\nAll SelectAllTopK accuracy tests PASSED!" << std::endl;
    }
    else
    {
        std::cout << "\nSome SelectAllTopK accuracy tests FAILED!" << std::endl;
        std::cout << "Error details:" << std::endl;
        for (auto const& error : accuracyErrorMessages)
        {
            std::cout << "  - " << error << std::endl;
        }
    }
}
