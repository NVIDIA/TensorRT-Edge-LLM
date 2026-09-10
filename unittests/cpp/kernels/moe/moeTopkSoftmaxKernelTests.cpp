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
#include "kernels/moe/moeTopkSoftmaxKernels.h"
#include "references.h"
#include "testUtils.h"
#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>
#include <iomanip>
#include <numeric>
#include <set>
#include <tuple>
#include <vector>

using namespace trt_edgellm;
using namespace trt_edgellm::kernel;
using namespace nvinfer1;

// ============================================================================
// Test Configuration and Runner
// ============================================================================

struct MoeTestConfig
{
    int32_t numTokens;
    int32_t numExperts;
    int32_t topk;
    bool renormalize = false;
    float moeSoftcapping = 0.0f;
    std::vector<float> const* correctionBias = nullptr;
    DataType inputDtype = DataType::kFLOAT;
    float rtol = 1e-4f;
    float atol = 1e-5f;
    std::string description;
};

struct MoeTestResult
{
    std::vector<float> weights;
    std::vector<int32_t> indices;
};

// Centralized test runner for MoE TopK Softmax kernel
class MoeTopkSoftmaxTestRunner
{
public:
    // Run kernel and return results (FP32 input)
    static MoeTestResult run(
        std::vector<float> const& input, MoeTestConfig const& config, cudaStream_t stream = nullptr)
    {
        int32_t const numTokens = config.numTokens;
        int32_t const numExperts = config.numExperts;
        int32_t const topk = config.topk;

        // Create device tensors
        auto gatingOutputDevice = rt::Tensor({numTokens, numExperts}, rt::DeviceType::kGPU, config.inputDtype);
        auto topkWeightsDevice = rt::Tensor({numTokens, topk}, rt::DeviceType::kGPU, DataType::kFLOAT);
        auto topkIndicesDevice = rt::Tensor({numTokens, topk}, rt::DeviceType::kGPU, DataType::kINT32);

        // Copy input to device (handle dtype conversion)
        copyInputToDevice(input, gatingOutputDevice, config.inputDtype);

        // Setup correction bias if provided
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

        // Allocate workspace
        size_t workspaceSize = getMoeTopkSoftmaxWorkspaceSize(numTokens, numExperts);
        void* workspace = nullptr;
        if (workspaceSize > 0)
        {
            CUDA_CHECK(cudaMalloc(&workspace, workspaceSize));
        }

        // Execute kernel
        moeTopkSoftmax(gatingOutputDevice, topkWeightsDevice, topkIndicesDevice, topk, workspace, workspaceSize, stream,
            config.renormalize, config.moeSoftcapping, correctionBiasOpt);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Copy results back
        MoeTestResult result;
        result.weights = copyDeviceToHost<float>(topkWeightsDevice);
        result.indices = copyDeviceToHost<int32_t>(topkIndicesDevice);

        // Cleanup
        if (workspace)
        {
            CUDA_CHECK(cudaFree(workspace));
        }

        return result;
    }

    // Run kernel and verify against reference
    static void runAndVerify(
        std::vector<float> const& input, MoeTestConfig const& config, cudaStream_t stream = nullptr)
    {
        // Compute reference
        std::vector<float> expectedWeights;
        std::vector<int32_t> expectedIndices;
        referenceMoeTopkSoftmax(input, config.correctionBias, expectedWeights, expectedIndices, config.numTokens,
            config.numExperts, config.topk, config.renormalize, config.moeSoftcapping);

        // Run kernel
        auto result = run(input, config, stream);

        // Verify results
        verify(result, expectedWeights, expectedIndices, config);
    }

    // Verify weights sum to 1 (for renormalized outputs)
    static void verifyWeightsSum(MoeTestResult const& result, MoeTestConfig const& config)
    {
        for (int32_t t = 0; t < config.numTokens; t++)
        {
            float sum = 0.0f;
            for (int32_t k = 0; k < config.topk; k++)
            {
                sum += result.weights[t * config.topk + k];
            }
            EXPECT_TRUE(isclose(sum, 1.0f, config.rtol, config.atol))
                << config.description << " Token " << t << ": weights don't sum to 1. Sum=" << sum;
        }
    }

private:
    static void copyInputToDevice(std::vector<float> const& input, rt::Tensor& deviceTensor, DataType dtype)
    {
        size_t const numElements = input.size();
        if (dtype == DataType::kFLOAT)
        {
            CUDA_CHECK(cudaMemcpy(
                deviceTensor.rawPointer(), input.data(), numElements * sizeof(float), cudaMemcpyHostToDevice));
        }
        else if (dtype == DataType::kHALF)
        {
            std::vector<half> inputHalf(numElements);
            for (size_t i = 0; i < numElements; i++)
            {
                inputHalf[i] = __float2half(input[i]);
            }
            CUDA_CHECK(cudaMemcpy(
                deviceTensor.rawPointer(), inputHalf.data(), numElements * sizeof(half), cudaMemcpyHostToDevice));
        }
        else if (dtype == DataType::kBF16)
        {
            std::vector<__nv_bfloat16> inputBF16(numElements);
            for (size_t i = 0; i < numElements; i++)
            {
                inputBF16[i] = __float2bfloat16(input[i]);
            }
            CUDA_CHECK(cudaMemcpy(deviceTensor.rawPointer(), inputBF16.data(), numElements * sizeof(__nv_bfloat16),
                cudaMemcpyHostToDevice));
        }
    }

    static void verify(MoeTestResult const& result, std::vector<float> const& expectedWeights,
        std::vector<int32_t> const& expectedIndices, MoeTestConfig const& config)
    {
        for (int32_t t = 0; t < config.numTokens; t++)
        {
            // Collect (expert_index, weight) pairs for this token
            std::vector<std::pair<int32_t, float>> resultPairs(config.topk);
            std::vector<std::pair<int32_t, float>> expectedPairs(config.topk);
            for (int32_t k = 0; k < config.topk; k++)
            {
                int32_t idx = t * config.topk + k;
                resultPairs[k] = {result.indices[idx], result.weights[idx]};
                expectedPairs[k] = {expectedIndices[idx], expectedWeights[idx]};
            }

            // Sort by expert index for order-independent comparison.
            // Near-equal experts can legitimately appear in different positional order
            // between the GPU kernel (parallel warp reduction) and the CPU reference
            // (sequential), because different floating-point summation orders yield
            // slightly different softmax values that can flip the relative ranking of
            // borderline candidates.
            std::sort(resultPairs.begin(), resultPairs.end());
            std::sort(expectedPairs.begin(), expectedPairs.end());

            for (int32_t k = 0; k < config.topk; k++)
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
// Test Fixture with Stream Management
// ============================================================================

class MoeTopkSoftmaxTest : public ::testing::Test
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
// Helper to generate test inputs
// ============================================================================

std::vector<float> generateDeterministicInput(int32_t numTokens, int32_t numExperts)
{
    std::vector<float> input(numTokens * numExperts);
    for (int32_t t = 0; t < numTokens; t++)
    {
        for (int32_t e = 0; e < numExperts; e++)
        {
            // Create a pattern where lower-indexed experts have higher values
            input[t * numExperts + e] = static_cast<float>(numExperts - e) + (t % 3) * 0.1f;
        }
    }
    return input;
}

std::vector<float> generateRandomInput(int32_t numTokens, int32_t numExperts, float low = -2.0f, float high = 2.0f)
{
    std::vector<float> input(numTokens * numExperts);
    uniformFloatInitialization(input, low, high);
    return input;
}

// ============================================================================
// Tests
// ============================================================================

TEST_F(MoeTopkSoftmaxTest, BasicFunctionality_8Experts)
{
    MoeTestConfig config{.numTokens = 0, .numExperts = 8, .topk = 2, .description = "Basic 8 experts: "};

    for (int32_t numTokens : {1, 4, 16, 64})
    {
        config.numTokens = numTokens;
        auto input = generateDeterministicInput(numTokens, config.numExperts);
        MoeTopkSoftmaxTestRunner::runAndVerify(input, config, stream);
    }
}

TEST_F(MoeTopkSoftmaxTest, PowerOf2ExpertCounts)
{
    MoeTestConfig config{.numTokens = 8, .numExperts = 0, .topk = 2, .renormalize = true};

    for (int32_t numExperts : {2, 4, 8, 16, 32, 64, 128, 256})
    {
        config.numExperts = numExperts;
        config.description = "Power-of-2 experts=" + std::to_string(numExperts) + ": ";
        auto input = generateRandomInput(config.numTokens, numExperts);

        auto result = MoeTopkSoftmaxTestRunner::run(input, config, stream);
        MoeTopkSoftmaxTestRunner::verifyWeightsSum(result, config);
        MoeTopkSoftmaxTestRunner::runAndVerify(input, config, stream);
    }
}

TEST_F(MoeTopkSoftmaxTest, NonPowerOf2ExpertCounts)
{
    MoeTestConfig config{.numTokens = 8, .numExperts = 0, .topk = 2, .description = ""};

    for (int32_t numExperts : {3, 5, 6, 7, 9, 10, 12, 15, 17, 24, 48})
    {
        config.numExperts = numExperts;
        config.description = "Non-power-of-2 experts=" + std::to_string(numExperts) + ": ";

        // Workspace should be required for non-power-of-2
        size_t workspaceSize = getMoeTopkSoftmaxWorkspaceSize(config.numTokens, numExperts);
        EXPECT_GT(workspaceSize, 0u) << "Workspace should be required for numExperts=" << numExperts;

        auto input = generateRandomInput(config.numTokens, numExperts);
        MoeTopkSoftmaxTestRunner::runAndVerify(input, config, stream);
    }
}

TEST_F(MoeTopkSoftmaxTest, RenormalizationTest)
{
    MoeTestConfig config{.numTokens = 16, .numExperts = 8, .topk = 0};

    for (int32_t topk : {1, 2, 3, 4})
    {
        config.topk = topk;
        config.renormalize = true;
        config.description = "Renormalize topk=" + std::to_string(topk) + ": ";

        auto input = generateRandomInput(config.numTokens, config.numExperts, -3.0f, 3.0f);
        auto result = MoeTopkSoftmaxTestRunner::run(input, config, stream);
        MoeTopkSoftmaxTestRunner::verifyWeightsSum(result, config);
    }
}

TEST_F(MoeTopkSoftmaxTest, SoftcappingTest)
{
    // Create input with large values that will be affected by softcapping
    int32_t const numTokens = 8;
    int32_t const numExperts = 8;
    std::vector<float> input(numTokens * numExperts);
    for (int32_t t = 0; t < numTokens; t++)
    {
        for (int32_t e = 0; e < numExperts; e++)
        {
            input[t * numExperts + e] = static_cast<float>((numExperts - e) * 10 - 40);
        }
    }

    MoeTestConfig config{.numTokens = numTokens, .numExperts = numExperts, .topk = 2};

    for (float moeSoftcapping : {0.0f, 10.0f, 30.0f, 50.0f})
    {
        config.moeSoftcapping = moeSoftcapping;
        config.description = "Softcapping=" + std::to_string(moeSoftcapping) + ": ";
        MoeTopkSoftmaxTestRunner::runAndVerify(input, config, stream);
    }
}

TEST_F(MoeTopkSoftmaxTest, CorrectionBiasTest)
{
    int32_t const numTokens = 8;
    int32_t const numExperts = 8;
    int32_t const topk = 2;

    // Create input where all experts have equal logits
    std::vector<float> input(numTokens * numExperts, 0.0f);

    // Create bias that favors expert 5 and 6
    std::vector<float> correctionBias(numExperts, 0.0f);
    correctionBias[5] = 2.0f; // Strong preference for expert 5
    correctionBias[6] = 1.0f; // Moderate preference for expert 6

    MoeTestConfig config{.numTokens = numTokens,
        .numExperts = numExperts,
        .topk = topk,
        .correctionBias = &correctionBias,
        .description = "Correction bias: "};

    auto result = MoeTopkSoftmaxTestRunner::run(input, config, stream);

    // Verify that expert 5 and 6 are selected due to bias
    for (int32_t t = 0; t < numTokens; t++)
    {
        EXPECT_EQ(result.indices[t * topk + 0], 5) << "Token " << t << ": expected expert 5 as top-1 due to bias";
        EXPECT_EQ(result.indices[t * topk + 1], 6) << "Token " << t << ": expected expert 6 as top-2 due to bias";
    }

    MoeTopkSoftmaxTestRunner::runAndVerify(input, config, stream);
}

TEST_F(MoeTopkSoftmaxTest, FP16InputTest)
{
    auto [rtol, atol] = getTolerance<half>();
    MoeTestConfig config{.numTokens = 16,
        .numExperts = 8,
        .topk = 2,
        .renormalize = true,
        .inputDtype = DataType::kHALF,
        .rtol = rtol,
        .atol = atol,
        .description = "FP16: "};

    // Use deterministic input with large gaps to avoid precision issues
    auto input = generateDeterministicInput(config.numTokens, config.numExperts);
    auto result = MoeTopkSoftmaxTestRunner::run(input, config, stream);

    // Verify top-2 indices are 0 and 1 (highest values in deterministic input)
    for (int32_t t = 0; t < config.numTokens; t++)
    {
        std::set<int32_t> expectedSet = {0, 1};
        std::set<int32_t> actualSet = {result.indices[t * config.topk], result.indices[t * config.topk + 1]};
        EXPECT_EQ(actualSet, expectedSet) << "FP16: Token " << t << ": unexpected expert selection";
    }

    MoeTopkSoftmaxTestRunner::verifyWeightsSum(result, config);
}

TEST_F(MoeTopkSoftmaxTest, BF16InputTest)
{
    auto [rtol, atol] = getTolerance<__nv_bfloat16>();
    MoeTestConfig config{.numTokens = 16,
        .numExperts = 8,
        .topk = 2,
        .renormalize = true,
        .inputDtype = DataType::kBF16,
        .rtol = rtol,
        .atol = atol,
        .description = "BF16: "};

    // Use deterministic input with large gaps to avoid precision issues
    auto input = generateDeterministicInput(config.numTokens, config.numExperts);
    auto result = MoeTopkSoftmaxTestRunner::run(input, config, stream);

    // Verify top-2 indices are 0 and 1 (highest values in deterministic input)
    for (int32_t t = 0; t < config.numTokens; t++)
    {
        std::set<int32_t> expectedSet = {0, 1};
        std::set<int32_t> actualSet = {result.indices[t * config.topk], result.indices[t * config.topk + 1]};
        EXPECT_EQ(actualSet, expectedSet) << "BF16: Token " << t << ": unexpected expert selection";
    }

    MoeTopkSoftmaxTestRunner::verifyWeightsSum(result, config);
}

TEST_F(MoeTopkSoftmaxTest, LargeScaleTest)
{
    MoeTestConfig config{
        .numTokens = 1024, .numExperts = 64, .topk = 4, .renormalize = true, .description = "Large scale: "};

    auto input = generateRandomInput(config.numTokens, config.numExperts, -3.0f, 3.0f);
    auto result = MoeTopkSoftmaxTestRunner::run(input, config, stream);

    MoeTopkSoftmaxTestRunner::verifyWeightsSum(result, config);
    MoeTopkSoftmaxTestRunner::runAndVerify(input, config, stream);
}

TEST_F(MoeTopkSoftmaxTest, EdgeCases)
{
    // Test case 1: Single token, topk=1
    {
        std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
        MoeTestConfig config{.numTokens = 1, .numExperts = 8, .topk = 1, .description = "Single token topk=1: "};

        auto result = MoeTopkSoftmaxTestRunner::run(input, config, stream);
        EXPECT_EQ(result.indices[0], 7) << "Should select expert 7 (highest logit)";
        MoeTopkSoftmaxTestRunner::runAndVerify(input, config, stream);
    }

    // Test case 2: topk equals numExperts
    {
        MoeTestConfig config{
            .numTokens = 4, .numExperts = 4, .topk = 4, .renormalize = true, .description = "topk=numExperts: "};

        auto input = generateRandomInput(config.numTokens, config.numExperts, -1.0f, 1.0f);
        auto result = MoeTopkSoftmaxTestRunner::run(input, config, stream);
        MoeTopkSoftmaxTestRunner::verifyWeightsSum(result, config);
    }
}

TEST_F(MoeTopkSoftmaxTest, CombinedFeaturesTest)
{
    int32_t const numExperts = 8;

    // Create correction bias
    std::vector<float> correctionBias(numExperts);
    for (int32_t e = 0; e < numExperts; e++)
    {
        correctionBias[e] = static_cast<float>(e) * 0.5f - 2.0f;
    }

    MoeTestConfig config{.numTokens = 8,
        .numExperts = numExperts,
        .topk = 3,
        .renormalize = true,
        .moeSoftcapping = 20.0f,
        .correctionBias = &correctionBias,
        .description = "Combined features: "};

    // Create input with large variance
    std::vector<float> input(config.numTokens * numExperts);
    for (int32_t t = 0; t < config.numTokens; t++)
    {
        for (int32_t e = 0; e < numExperts; e++)
        {
            input[t * numExperts + e] = static_cast<float>((t + e) * 5 - 20);
        }
    }

    auto result = MoeTopkSoftmaxTestRunner::run(input, config, stream);
    MoeTopkSoftmaxTestRunner::verifyWeightsSum(result, config);
    MoeTopkSoftmaxTestRunner::runAndVerify(input, config, stream);
}

// ============================================================================
// Performance Tests
// ============================================================================

TEST_F(MoeTopkSoftmaxTest, DISABLED_PerformanceBenchmark)
{

    std::cout << "\n========== MoE TopK Softmax Performance Benchmark ==========" << std::endl;
    std::cout << "Reference Model: Qwen3-MoE (128 experts, topk=8)" << std::endl;
    std::cout << "Metrics: Latency (us), Throughput (K tokens/s)" << std::endl;
    std::cout << "============================================================\n" << std::endl;

    struct PerfConfig
    {
        std::string description;
        int32_t numTokens;
        int32_t numExperts;
        int32_t topk;
    };

    std::vector<PerfConfig> configs = {
        {"Qwen3-MoE decode (bs=1)", 1, 128, 8},
        {"Qwen3-MoE decode (bs=4)", 4, 128, 8},
        {"Qwen3-MoE prefill", 128, 128, 8},
        {"8 experts, topk=2", 4, 8, 2},
        {"64 experts, topk=6", 4, 64, 6},
    };

    constexpr int numWarmup = 50;
    constexpr int numIterations = 200;

    std::cout << std::left << std::setw(32) << "Configuration" << std::setw(15) << "Latency (us)" << std::setw(18)
              << "Throughput (K/s)" << std::setw(10) << "Path" << std::endl;
    std::cout << std::string(75, '-') << std::endl;

    for (auto const& cfg : configs)
    {
        auto input = generateRandomInput(cfg.numTokens, cfg.numExperts, -3.0f, 3.0f);

        auto gatingOutputDevice = rt::Tensor({cfg.numTokens, cfg.numExperts}, rt::DeviceType::kGPU, DataType::kFLOAT);
        auto topkWeightsDevice = rt::Tensor({cfg.numTokens, cfg.topk}, rt::DeviceType::kGPU, DataType::kFLOAT);
        auto topkIndicesDevice = rt::Tensor({cfg.numTokens, cfg.topk}, rt::DeviceType::kGPU, DataType::kINT32);

        copyHostToDevice<float>(gatingOutputDevice, input);

        size_t workspaceSize = getMoeTopkSoftmaxWorkspaceSize(cfg.numTokens, cfg.numExperts);
        void* workspace = nullptr;
        if (workspaceSize > 0)
        {
            CUDA_CHECK(cudaMalloc(&workspace, workspaceSize));
        }

        bool isPow2 = (cfg.numExperts != 0) && ((cfg.numExperts & (cfg.numExperts - 1)) == 0);
        std::string pathStr = (isPow2 && cfg.numExperts <= 256) ? "Fused" : "Fallback";

        // Warmup
        for (int i = 0; i < numWarmup; ++i)
        {
            moeTopkSoftmax(gatingOutputDevice, topkWeightsDevice, topkIndicesDevice, cfg.topk, workspace, workspaceSize,
                stream, true);
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));

        // Benchmark
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        CUDA_CHECK(cudaEventRecord(start, stream));
        for (int iter = 0; iter < numIterations; ++iter)
        {
            moeTopkSoftmax(gatingOutputDevice, topkWeightsDevice, topkIndicesDevice, cfg.topk, workspace, workspaceSize,
                stream, true);
        }
        CUDA_CHECK(cudaEventRecord(stop, stream));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float totalMs = 0;
        CUDA_CHECK(cudaEventElapsedTime(&totalMs, start, stop));
        float avgLatencyUs = (totalMs / numIterations) * 1000.0f;
        float throughputKTokPerSec = (static_cast<float>(cfg.numTokens) / (avgLatencyUs / 1e6f)) / 1e3f;

        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));

        std::cout << std::left << std::setw(32) << cfg.description << std::setw(15) << std::fixed
                  << std::setprecision(2) << avgLatencyUs << std::setw(18) << std::setprecision(1)
                  << throughputKTokPerSec << std::setw(10) << pathStr << std::endl;

        if (workspace)
        {
            CUDA_CHECK(cudaFree(workspace));
        }
    }

    std::cout << "\n============================================================" << std::endl;
}
