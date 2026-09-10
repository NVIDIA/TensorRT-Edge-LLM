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

#include "kernels/moe/moePerExpertScaleKernels.h"
#include "testUtils.h"
#include <gtest/gtest.h>
#include <vector>

using namespace trt_edgellm::kernel;

class MoePerExpertScaleTest : public ::testing::Test
{
protected:
    void runAndVerify(std::vector<float>& weights, std::vector<int32_t> const& ids, std::vector<float> const& scales,
        int32_t numTokens, int32_t topK)
    {
        int32_t const count = numTokens * topK;

        // Compute expected output on CPU.
        std::vector<float> expected(weights);
        for (int32_t i = 0; i < count; ++i)
        {
            expected[i] *= scales[ids[i]];
        }

        // Allocate device memory.
        float* dWeights = nullptr;
        int32_t* dIds = nullptr;
        float* dScales = nullptr;
        CUDA_CHECK(cudaMalloc(&dWeights, count * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dIds, count * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&dScales, scales.size() * sizeof(float)));

        CUDA_CHECK(cudaMemcpy(dWeights, weights.data(), count * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dIds, ids.data(), count * sizeof(int32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dScales, scales.data(), scales.size() * sizeof(float), cudaMemcpyHostToDevice));

        // Run kernel.
        applyPerExpertScale(dWeights, dIds, dScales, numTokens, topK, nullptr);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Copy back and verify.
        std::vector<float> result(count);
        CUDA_CHECK(cudaMemcpy(result.data(), dWeights, count * sizeof(float), cudaMemcpyDeviceToHost));

        for (int32_t i = 0; i < count; ++i)
        {
            EXPECT_FLOAT_EQ(result[i], expected[i]) << "Mismatch at index " << i;
        }

        CUDA_CHECK(cudaFree(dWeights));
        CUDA_CHECK(cudaFree(dIds));
        CUDA_CHECK(cudaFree(dScales));
    }
};

TEST_F(MoePerExpertScaleTest, BasicSmall)
{
    // 2 tokens, top-2, 4 experts
    std::vector<float> weights = {0.6f, 0.4f, 0.7f, 0.3f};
    std::vector<int32_t> ids = {0, 2, 1, 3};
    std::vector<float> scales = {1.0f, 2.0f, 0.5f, 3.0f};

    runAndVerify(weights, ids, scales, /*numTokens=*/2, /*topK=*/2);
}

TEST_F(MoePerExpertScaleTest, IdentityScale)
{
    // All scales = 1.0 should leave weights unchanged.
    int32_t const numTokens = 4;
    int32_t const topK = 8;
    int32_t const numExperts = 128;
    int32_t const count = numTokens * topK;

    std::vector<float> weights(count);
    uniformFloatInitialization(weights, 0.0f, 1.0f);
    std::vector<float> original(weights);

    std::vector<int32_t> ids(count);
    uniformIntInitialization(ids, 0, numExperts - 1);

    std::vector<float> scales(numExperts, 1.0f);

    runAndVerify(weights, ids, scales, numTokens, topK);
}

TEST_F(MoePerExpertScaleTest, ZeroScale)
{
    // Scale = 0 should zero out the weight.
    std::vector<float> weights = {0.5f, 0.5f};
    std::vector<int32_t> ids = {0, 1};
    std::vector<float> scales = {0.0f, 0.0f};

    runAndVerify(weights, ids, scales, /*numTokens=*/1, /*topK=*/2);
}

TEST_F(MoePerExpertScaleTest, LargeBatch)
{
    // Stress test: 1024 tokens, top-8, 128 experts (matches Gemma4 26B-A4B config).
    int32_t const numTokens = 1024;
    int32_t const topK = 8;
    int32_t const numExperts = 128;
    int32_t const count = numTokens * topK;

    std::vector<float> weights(count);
    uniformFloatInitialization(weights, 0.0f, 1.0f);

    std::vector<int32_t> ids(count);
    uniformIntInitialization(ids, 0, numExperts - 1);

    std::vector<float> scales(numExperts);
    uniformFloatInitialization(scales, 0.5f, 2.0f);

    runAndVerify(weights, ids, scales, numTokens, topK);
}

TEST_F(MoePerExpertScaleTest, SingleToken)
{
    // Edge case: 1 token, top-1.
    std::vector<float> weights = {1.0f};
    std::vector<int32_t> ids = {5};
    std::vector<float> scales(8, 1.0f);
    scales[5] = 0.75f;

    runAndVerify(weights, ids, scales, /*numTokens=*/1, /*topK=*/1);
}
