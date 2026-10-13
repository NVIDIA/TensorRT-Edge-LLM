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
#include "kernels/speculative/dflash2CandidateSelector.h"
#include "testUtils.h"

#include <cuda_fp16.h>
#include <gtest/gtest.h>

#include <cmath>
#include <vector>

using namespace trt_edgellm;
using namespace trt_edgellm::kernel;
using namespace nvinfer1;

namespace
{
std::vector<half> halfVector(std::vector<float> const& values)
{
    std::vector<half> result(values.size());
    for (size_t i = 0; i < values.size(); ++i)
    {
        result[i] = __float2half(values[i]);
    }
    return result;
}
} // namespace

TEST(DFlash2CandidateSelector, GreedyWalkUsesSelectedPredecessorAndOneHotQ)
{
    constexpr int32_t batch = 1;
    constexpr int32_t steps = 3;
    constexpr int32_t topK = 4;
    constexpr int32_t rank = 32;
    constexpr int32_t vocab = 16;

    rt::Tensor ids({batch, steps, topK}, rt::DeviceType::kGPU, DataType::kINT32);
    rt::Tensor unary({batch, steps, topK}, rt::DeviceType::kGPU, DataType::kFLOAT);
    rt::Tensor projected({batch, steps, rank}, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor anchor({batch}, rt::DeviceType::kGPU, DataType::kINT32);
    rt::Tensor predecessor({vocab, rank}, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor successor({vocab, rank}, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor uniforms({batch, steps}, rt::DeviceType::kGPU, DataType::kFLOAT);
    rt::Tensor temperatures({batch}, rt::DeviceType::kGPU, DataType::kFLOAT);
    rt::Tensor greedy({batch}, rt::DeviceType::kGPU, DataType::kINT32);
    rt::Tensor tokens({batch, steps}, rt::DeviceType::kGPU, DataType::kINT32);
    rt::Tensor q({batch, steps, topK}, rt::DeviceType::kGPU, DataType::kFLOAT);

    copyHostToDevice<int32_t>(ids, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    copyHostToDevice<float>(unary, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0});
    std::vector<float> z(batch * steps * rank, 1.0F);
    std::vector<float> pred(vocab * rank, 0.0F);
    std::vector<float> succ(vocab * rank, 0.0F);
    // anchor 0 prefers token 3; selected token 3 then prefers 6; token 6 prefers 12.
    for (int32_t r = 0; r < rank; ++r)
    {
        pred[0 * rank + r] = 1.0F;
        succ[3 * rank + r] = 0.25F;
        pred[3 * rank + r] = 1.0F;
        succ[6 * rank + r] = 0.5F;
        pred[6 * rank + r] = 1.0F;
        succ[12 * rank + r] = 0.75F;
    }
    copyHostToDevice<half>(projected, halfVector(z));
    copyHostToDevice<int32_t>(anchor, {0});
    copyHostToDevice<half>(predecessor, halfVector(pred));
    copyHostToDevice<half>(successor, halfVector(succ));
    copyHostToDevice<float>(uniforms, {0.1F, 0.2F, 0.3F});
    copyHostToDevice<float>(temperatures, {1.0F});
    copyHostToDevice<int32_t>(greedy, {1});

    launchDFlash2CandidateSelector(
        ids, unary, projected, anchor, predecessor, successor, uniforms, temperatures, greedy, tokens, q, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    EXPECT_EQ(copyDeviceToHost<int32_t>(tokens), (std::vector<int32_t>{3, 6, 12}));
    auto const probs = copyDeviceToHost<float>(q);
    std::vector<int32_t> const selectedIndex{2, 1, 3};
    for (int32_t step = 0; step < steps; ++step)
    {
        float sum = 0.0F;
        for (int32_t k = 0; k < topK; ++k)
        {
            sum += probs[step * topK + k];
            EXPECT_FLOAT_EQ(probs[step * topK + k], k == selectedIndex[step] ? 1.0F : 0.0F);
        }
        EXPECT_FLOAT_EQ(sum, 1.0F);
    }
}

TEST(DFlash2CandidateSelector, ProbabilisticWalkUsesInverseCdfAndReturnsNormalizedQ)
{
    constexpr int32_t batch{1};
    constexpr int32_t steps{2};
    constexpr int32_t topK{4};
    constexpr int32_t rank{32};
    constexpr int32_t vocab{16};
    rt::Tensor ids({batch, steps, topK}, rt::DeviceType::kGPU, DataType::kINT32);
    rt::Tensor unary({batch, steps, topK}, rt::DeviceType::kGPU, DataType::kFLOAT);
    rt::Tensor projected({batch, steps, rank}, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor anchor({batch}, rt::DeviceType::kGPU, DataType::kINT32);
    rt::Tensor predecessor({vocab, rank}, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor successor({vocab, rank}, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor uniforms({batch, steps}, rt::DeviceType::kGPU, DataType::kFLOAT);
    rt::Tensor temperatures({batch}, rt::DeviceType::kGPU, DataType::kFLOAT);
    rt::Tensor greedy({batch}, rt::DeviceType::kGPU, DataType::kINT32);
    rt::Tensor tokens({batch, steps}, rt::DeviceType::kGPU, DataType::kINT32);
    rt::Tensor q({batch, steps, topK}, rt::DeviceType::kGPU, DataType::kFLOAT);

    copyHostToDevice<int32_t>(ids, {1, 2, 3, 4, 5, 6, 7, 8});
    std::vector<float> const logits{std::log(0.1F), std::log(0.2F), std::log(0.3F), std::log(0.4F), std::log(0.1F),
        std::log(0.2F), std::log(0.3F), std::log(0.4F)};
    copyHostToDevice<float>(unary, logits);
    copyHostToDevice<half>(projected, halfVector(std::vector<float>(batch * steps * rank, 0.0F)));
    copyHostToDevice<int32_t>(anchor, {0});
    copyHostToDevice<half>(predecessor, halfVector(std::vector<float>(vocab * rank, 0.0F)));
    copyHostToDevice<half>(successor, halfVector(std::vector<float>(vocab * rank, 0.0F)));
    copyHostToDevice<float>(uniforms, {0.05F, 0.95F});
    copyHostToDevice<float>(temperatures, {1.0F});
    copyHostToDevice<int32_t>(greedy, {0});

    launchDFlash2CandidateSelector(
        ids, unary, projected, anchor, predecessor, successor, uniforms, temperatures, greedy, tokens, q, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    EXPECT_EQ(copyDeviceToHost<int32_t>(tokens), (std::vector<int32_t>{1, 8}));
    auto const probabilities = copyDeviceToHost<float>(q);
    for (int32_t step = 0; step < steps; ++step)
    {
        for (int32_t k = 0; k < topK; ++k)
        {
            EXPECT_NEAR(probabilities[step * topK + k], 0.1F * static_cast<float>(k + 1), 1.0e-5F);
        }
    }
}

TEST(DFlash2CandidateSelector, SupportsProductionExtentsAndMultipleBatches)
{
    constexpr int32_t batch{2};
    constexpr int32_t steps{15};
    constexpr int32_t topK{16};
    constexpr int32_t rank{256};
    constexpr int32_t vocab{1024};
    rt::Tensor ids({batch, steps, topK}, rt::DeviceType::kGPU, DataType::kINT32);
    rt::Tensor unary({batch, steps, topK}, rt::DeviceType::kGPU, DataType::kFLOAT);
    rt::Tensor projected({batch, steps, rank}, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor anchor({batch}, rt::DeviceType::kGPU, DataType::kINT32);
    rt::Tensor predecessor({vocab, rank}, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor successor({vocab, rank}, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor uniforms({batch, steps}, rt::DeviceType::kGPU, DataType::kFLOAT);
    rt::Tensor temperatures({batch}, rt::DeviceType::kGPU, DataType::kFLOAT);
    rt::Tensor greedy({batch}, rt::DeviceType::kGPU, DataType::kINT32);
    rt::Tensor tokens({batch, steps}, rt::DeviceType::kGPU, DataType::kINT32);
    rt::Tensor q({batch, steps, topK}, rt::DeviceType::kGPU, DataType::kFLOAT);

    std::vector<int32_t> hostIds(batch * steps * topK);
    std::vector<float> hostUnary(batch * steps * topK, 0.0F);
    std::vector<float> hostUniforms(batch * steps, 0.999F);
    for (int32_t b = 0; b < batch; ++b)
    {
        for (int32_t step = 0; step < steps; ++step)
        {
            int32_t const row = b * steps + step;
            for (int32_t k = 0; k < topK; ++k)
            {
                hostIds[row * topK + k] = row * topK + k;
                if (b == 0 && k == step % topK)
                {
                    hostUnary[row * topK + k] = 1.0F;
                }
            }
        }
    }
    copyHostToDevice<int32_t>(ids, hostIds);
    copyHostToDevice<float>(unary, hostUnary);
    copyHostToDevice<half>(projected, halfVector(std::vector<float>(batch * steps * rank, 0.0F)));
    copyHostToDevice<int32_t>(anchor, {0, 1});
    copyHostToDevice<half>(predecessor, halfVector(std::vector<float>(vocab * rank, 0.0F)));
    copyHostToDevice<half>(successor, halfVector(std::vector<float>(vocab * rank, 0.0F)));
    copyHostToDevice<float>(uniforms, hostUniforms);
    copyHostToDevice<float>(temperatures, {1.0F, 1.0F});
    copyHostToDevice<int32_t>(greedy, {1, 0});

    launchDFlash2CandidateSelector(
        ids, unary, projected, anchor, predecessor, successor, uniforms, temperatures, greedy, tokens, q, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    auto const hostTokens = copyDeviceToHost<int32_t>(tokens);
    auto const probabilities = copyDeviceToHost<float>(q);
    for (int32_t step = 0; step < steps; ++step)
    {
        int32_t const greedyIndex = step % topK;
        EXPECT_EQ(hostTokens[step], hostIds[step * topK + greedyIndex]);
        EXPECT_EQ(hostTokens[steps + step], hostIds[(steps + step) * topK + topK - 1]);
        for (int32_t k = 0; k < topK; ++k)
        {
            EXPECT_FLOAT_EQ(probabilities[(step * topK) + k], k == greedyIndex ? 1.0F : 0.0F);
            EXPECT_NEAR(probabilities[((steps + step) * topK) + k], 1.0F / topK, 1.0e-6F);
        }
    }
}

TEST(DFlash2CandidateSelector, RejectsMismatchedOutputShape)
{
    constexpr int32_t batch{1};
    constexpr int32_t steps{2};
    constexpr int32_t topK{4};
    constexpr int32_t rank{32};
    constexpr int32_t vocab{16};
    rt::Tensor ids({batch, steps, topK}, rt::DeviceType::kGPU, DataType::kINT32);
    rt::Tensor unary({batch, steps, topK}, rt::DeviceType::kGPU, DataType::kFLOAT);
    rt::Tensor projected({batch, steps, rank}, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor anchor({batch}, rt::DeviceType::kGPU, DataType::kINT32);
    rt::Tensor predecessor({vocab, rank}, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor successor({vocab, rank}, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor uniforms({batch, steps}, rt::DeviceType::kGPU, DataType::kFLOAT);
    rt::Tensor temperatures({batch}, rt::DeviceType::kGPU, DataType::kFLOAT);
    rt::Tensor greedy({batch}, rt::DeviceType::kGPU, DataType::kINT32);
    rt::Tensor tokens({batch, steps}, rt::DeviceType::kGPU, DataType::kINT32);
    rt::Tensor q({batch, steps, topK - 1}, rt::DeviceType::kGPU, DataType::kFLOAT);

    EXPECT_THROW(launchDFlash2CandidateSelector(ids, unary, projected, anchor, predecessor, successor, uniforms,
                     temperatures, greedy, tokens, q, nullptr),
        std::runtime_error);
}
