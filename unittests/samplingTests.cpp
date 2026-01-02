/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "sampler/sampling.h"
#include "testUtils.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <iomanip>
#include <iostream>
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
        CUDA_CHECK(cudaMemcpy(logitsTensor.rawPointer(), flatHostLogits.data(), batchSize * vocabSize * sizeof(float),
            cudaMemcpyHostToDevice));
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
        std::vector<int32_t> gpuResults(batchSize);
        CUDA_CHECK(cudaMemcpy(gpuResults.data(), selectedIndicesTensor.rawPointer(), batchSize * sizeof(int32_t),
            cudaMemcpyDeviceToHost));

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
        std::vector<int32_t> gpuIndices(batchSize * topK);
        CUDA_CHECK(cudaMemcpy(gpuIndices.data(), topKIndicesTensor.rawPointer(), batchSize * topK * sizeof(int32_t),
            cudaMemcpyDeviceToHost));

        std::vector<float> gpuValues;
        if (testValues)
        {
            gpuValues.resize(batchSize * topK);
            CUDA_CHECK(cudaMemcpy(gpuValues.data(), topKValuesTensor.rawPointer(), batchSize * topK * sizeof(float),
                cudaMemcpyDeviceToHost));
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
        {"TopK", 20, 1.0f, 1.0f}, {"TopK", 50, 1.0f, 1.0f}, {"TopK", 100, 1.0f, 1.0f}, {"TopP", 0, 0.9f, 1.0f},
        {"TopP", 0, 0.95f, 1.0f}, {"TopP", 0, 0.99f, 1.0f}, {"TopKTopP", 20, 0.9f, 1.0f}, {"TopKTopP", 50, 0.95f, 1.0f},
        {"TopKTopP", 100, 0.99f, 1.0f}, {"TopK", 20, 1.0f, 0.5f}, {"TopK", 20, 1.0f, 1.5f}, {"TopP", 0, 0.9f, 0.5f},
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
