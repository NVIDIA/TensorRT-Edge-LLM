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
#include "kernels/speculative/speculativeSampling.h"
#include "testUtils.h"

#include <NvInfer.h>
#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <vector>

using namespace trt_edgellm;
using namespace trt_edgellm::kernel;
using namespace nvinfer1;

TEST(SpeculativeSamplingKernel, RejectsSupportLargerThanKernelCapacity)
{
    rt::Tensor values({1, 129}, rt::DeviceType::kGPU, DataType::kFLOAT);
    rt::Tensor probabilities({1, 129}, rt::DeviceType::kGPU, DataType::kFLOAT);
    EXPECT_THROW(speculativeNormalizeTopKTopP(values, probabilities, 1.0F, 0.95F, nullptr), std::runtime_error);
}

TEST(SpeculativeSamplingKernel, SparseTopKTopPIncludesCrossingTokenAndRenormalizes)
{
    rt::Tensor values({1, 4}, rt::DeviceType::kGPU, DataType::kFLOAT);
    rt::Tensor probabilities({1, 4}, rt::DeviceType::kGPU, DataType::kFLOAT);
    copyHostToDevice<float>(values, {std::log(0.50F), std::log(0.30F), std::log(0.15F), std::log(0.05F)});

    speculativeNormalizeTopKTopP(values, probabilities, /*temperature=*/1.0F, /*topP=*/0.75F, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    auto const actual = copyDeviceToHost<float>(probabilities);
    ASSERT_EQ(actual.size(), 4U);
    EXPECT_NEAR(actual[0], 0.625F, 1.0e-5F);
    EXPECT_NEAR(actual[1], 0.375F, 1.0e-5F);
    EXPECT_FLOAT_EQ(actual[2], 0.0F);
    EXPECT_FLOAT_EQ(actual[3], 0.0F);
}

TEST(SpeculativeSamplingKernel, SparseVerifierAcceptsPrefixThenSamplesExactResidual)
{
    constexpr int32_t batch{1};
    constexpr int32_t proposalLen{2};
    constexpr int32_t verifyLen{3};
    constexpr int32_t targetK{3};
    constexpr int32_t proposalK{2};

    rt::Tensor targetProbs({batch, verifyLen, targetK}, rt::DeviceType::kGPU, DataType::kFLOAT);
    rt::Tensor targetIds({batch, verifyLen, targetK}, rt::DeviceType::kGPU, DataType::kINT32);
    rt::Tensor proposalProbs({batch, proposalLen, proposalK}, rt::DeviceType::kGPU, DataType::kFLOAT);
    rt::Tensor proposalIds({batch, proposalLen, proposalK}, rt::DeviceType::kGPU, DataType::kINT32);
    rt::Tensor proposalTokens({batch, proposalLen}, rt::DeviceType::kGPU, DataType::kINT32);
    rt::Tensor proposalLengths({batch}, rt::DeviceType::kGPU, DataType::kINT32);
    rt::Tensor uniforms({batch, 2 * proposalLen + 1}, rt::DeviceType::kGPU, DataType::kFLOAT);
    rt::Tensor accepted({batch, verifyLen}, rt::DeviceType::kGPU, DataType::kINT32);
    rt::Tensor lengths({batch}, rt::DeviceType::kGPU, DataType::kINT32);
    rt::Tensor indices({batch, verifyLen}, rt::DeviceType::kGPU, DataType::kINT32);

    copyHostToDevice<float>(targetProbs, {0.6F, 0.3F, 0.1F, 0.1F, 0.7F, 0.2F, 0.2F, 0.3F, 0.5F});
    copyHostToDevice<int32_t>(targetIds, {10, 11, 12, 20, 21, 22, 30, 31, 32});
    copyHostToDevice<float>(proposalProbs, {0.5F, 0.5F, 0.5F, 0.5F});
    copyHostToDevice<int32_t>(proposalIds, {10, 13, 20, 22});
    copyHostToDevice<int32_t>(proposalTokens, {10, 20});
    copyHostToDevice<int32_t>(proposalLengths, {proposalLen});
    // accept[0], accept[1], residual[0], residual[1], bonus
    copyHostToDevice<float>(uniforms, {0.9F, 0.5F, 0.1F, 0.5F, 0.1F});

    speculativeSparseAccept(targetProbs, targetIds, proposalProbs, proposalIds, proposalTokens, proposalLengths,
        uniforms, accepted, lengths, &indices, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    EXPECT_EQ(copyDeviceToHost<int32_t>(lengths), (std::vector<int32_t>{2}));
    auto const acceptedHost = copyDeviceToHost<int32_t>(accepted);
    EXPECT_EQ(acceptedHost[0], 10);
    EXPECT_EQ(acceptedHost[1], 21);
    EXPECT_EQ(copyDeviceToHost<int32_t>(indices), (std::vector<int32_t>{0, 1, 2}));
}

TEST(SpeculativeSamplingKernel, DenseTargetVerifierSamplesResidualOutsideProposalSupport)
{
    constexpr int32_t batch{2};
    constexpr int32_t proposalLen{1};
    constexpr int32_t verifyLen{2};
    constexpr int32_t vocabSize{5};
    constexpr int32_t proposalK{2};
    rt::Tensor targetProbs({batch, verifyLen, vocabSize}, rt::DeviceType::kGPU, DataType::kFLOAT);
    rt::Tensor proposalProbs({batch, proposalLen, proposalK}, rt::DeviceType::kGPU, DataType::kFLOAT);
    rt::Tensor proposalIds({batch, proposalLen, proposalK}, rt::DeviceType::kGPU, DataType::kINT32);
    rt::Tensor proposalTokens({batch, proposalLen}, rt::DeviceType::kGPU, DataType::kINT32);
    rt::Tensor proposalLengths({batch}, rt::DeviceType::kGPU, DataType::kINT32);
    rt::Tensor uniforms({batch, 2 * proposalLen + 1}, rt::DeviceType::kGPU, DataType::kFLOAT);
    rt::Tensor accepted({batch, verifyLen}, rt::DeviceType::kGPU, DataType::kINT32);
    rt::Tensor lengths({batch}, rt::DeviceType::kGPU, DataType::kINT32);
    rt::Tensor indices({batch, verifyLen}, rt::DeviceType::kGPU, DataType::kINT32);

    copyHostToDevice<float>(targetProbs,
        {0.05F, 0.10F, 0.15F, 0.60F, 0.10F, 0.10F, 0.20F, 0.30F, 0.15F, 0.25F, 0.70F, 0.10F, 0.10F, 0.05F, 0.05F, 0.20F,
            0.10F, 0.30F, 0.25F, 0.15F});
    copyHostToDevice<float>(proposalProbs, {0.8F, 0.2F, 0.4F, 0.6F});
    copyHostToDevice<int32_t>(proposalIds, {0, 1, 0, 1});
    copyHostToDevice<int32_t>(proposalTokens, {0, 1});
    copyHostToDevice<int32_t>(proposalLengths, {1, 1});
    copyHostToDevice<float>(uniforms, {0.9F, 0.8F, 0.0F, 0.9F, 0.7F, 0.0F});

    speculativeDenseTargetAccept(targetProbs, proposalProbs, proposalIds, proposalTokens, proposalLengths, uniforms,
        accepted, lengths, &indices, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    EXPECT_EQ(copyDeviceToHost<int32_t>(lengths), (std::vector<int32_t>{1, 1}));
    auto const output = copyDeviceToHost<int32_t>(accepted);
    EXPECT_EQ(output[0], 3);
    EXPECT_EQ(output[verifyLen], 2);
    EXPECT_EQ(copyDeviceToHost<int32_t>(indices), (std::vector<int32_t>{0, 1, 0, 1}));
}

TEST(SpeculativeSamplingKernel, ZeroLengthProposalSamplesTargetBonusOnly)
{
    constexpr int32_t batch{1};
    constexpr int32_t proposalLen{2};
    constexpr int32_t verifyLen{3};
    constexpr int32_t support{2};
    rt::Tensor targetProbs({batch, verifyLen, support}, rt::DeviceType::kGPU, DataType::kFLOAT);
    rt::Tensor targetIds({batch, verifyLen, support}, rt::DeviceType::kGPU, DataType::kINT32);
    rt::Tensor proposalProbs({batch, proposalLen, support}, rt::DeviceType::kGPU, DataType::kFLOAT);
    rt::Tensor proposalIds({batch, proposalLen, support}, rt::DeviceType::kGPU, DataType::kINT32);
    rt::Tensor proposalTokens({batch, proposalLen}, rt::DeviceType::kGPU, DataType::kINT32);
    rt::Tensor proposalLengths({batch}, rt::DeviceType::kGPU, DataType::kINT32);
    rt::Tensor uniforms({batch, 2 * proposalLen + 1}, rt::DeviceType::kGPU, DataType::kFLOAT);
    rt::Tensor accepted({batch, verifyLen}, rt::DeviceType::kGPU, DataType::kINT32);
    rt::Tensor lengths({batch}, rt::DeviceType::kGPU, DataType::kINT32);
    rt::Tensor indices({batch, verifyLen}, rt::DeviceType::kGPU, DataType::kINT32);

    copyHostToDevice<float>(targetProbs, {0.25F, 0.75F, 0.9F, 0.1F, 0.6F, 0.4F});
    copyHostToDevice<int32_t>(targetIds, {10, 11, 20, 21, 30, 31});
    copyHostToDevice<float>(proposalProbs, {0.5F, 0.5F, 0.5F, 0.5F});
    copyHostToDevice<int32_t>(proposalIds, {40, 41, 42, 43});
    copyHostToDevice<int32_t>(proposalTokens, {40, 42});
    copyHostToDevice<int32_t>(proposalLengths, {0});
    // The final lane is the bonus draw. 0.5 selects token 11 from target row 0.
    copyHostToDevice<float>(uniforms, {0.0F, 0.0F, 0.0F, 0.0F, 0.5F});

    speculativeSparseAccept(targetProbs, targetIds, proposalProbs, proposalIds, proposalTokens, proposalLengths,
        uniforms, accepted, lengths, &indices, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    EXPECT_EQ(copyDeviceToHost<int32_t>(lengths), (std::vector<int32_t>{1}));
    EXPECT_EQ(copyDeviceToHost<int32_t>(accepted)[0], 11);
    EXPECT_EQ(copyDeviceToHost<int32_t>(indices)[0], 0);
}

TEST(SpeculativeSamplingKernel, SparseVerifierClampsAcceptLengthInsideExistingKernel)
{
    constexpr int32_t batch{2};
    constexpr int32_t proposalLen{2};
    constexpr int32_t verifyLen{3};
    constexpr int32_t support{1};
    rt::Tensor targetProbs({batch, verifyLen, support}, rt::DeviceType::kGPU, DataType::kFLOAT);
    rt::Tensor targetIds({batch, verifyLen, support}, rt::DeviceType::kGPU, DataType::kINT32);
    rt::Tensor proposalProbs({batch, proposalLen, support}, rt::DeviceType::kGPU, DataType::kFLOAT);
    rt::Tensor proposalIds({batch, proposalLen, support}, rt::DeviceType::kGPU, DataType::kINT32);
    rt::Tensor proposalTokens({batch, proposalLen}, rt::DeviceType::kGPU, DataType::kINT32);
    rt::Tensor proposalLengths({batch}, rt::DeviceType::kGPU, DataType::kINT32);
    rt::Tensor uniforms({batch, 2 * proposalLen + 1}, rt::DeviceType::kGPU, DataType::kFLOAT);
    rt::Tensor accepted({batch, verifyLen}, rt::DeviceType::kGPU, DataType::kINT32);
    rt::Tensor lengths({batch}, rt::DeviceType::kGPU, DataType::kINT32);
    rt::Tensor maxAcceptLengths({batch}, rt::DeviceType::kGPU, DataType::kINT32);

    copyHostToDevice<float>(targetProbs, {1.0F, 1.0F, 1.0F, 1.0F, 1.0F, 1.0F});
    copyHostToDevice<int32_t>(targetIds, {10, 20, 30, 10, 20, 30});
    copyHostToDevice<float>(proposalProbs, {1.0F, 1.0F, 1.0F, 1.0F});
    copyHostToDevice<int32_t>(proposalIds, {10, 20, 10, 20});
    copyHostToDevice<int32_t>(proposalTokens, {10, 20, 10, 20});
    copyHostToDevice<int32_t>(proposalLengths, {proposalLen, proposalLen});
    copyHostToDevice<float>(uniforms, std::vector<float>(batch * (2 * proposalLen + 1), 0.0F));
    copyHostToDevice<int32_t>(maxAcceptLengths, {1, 2});

    speculativeSparseAccept(targetProbs, targetIds, proposalProbs, proposalIds, proposalTokens, proposalLengths,
        uniforms, accepted, lengths, nullptr, nullptr, &maxAcceptLengths);
    CUDA_CHECK(cudaDeviceSynchronize());

    EXPECT_EQ(copyDeviceToHost<int32_t>(lengths), (std::vector<int32_t>{1, 2}));
}

TEST(SpeculativeSamplingKernel, SparseVerifierIsStatisticallyLossless)
{
    constexpr int32_t batch{50000};
    constexpr int32_t proposalLen{1};
    constexpr int32_t verifyLen{2};
    constexpr int32_t support{2};
    std::vector<float> targetProbs(batch * verifyLen * support);
    std::vector<int32_t> targetIds(batch * verifyLen * support);
    std::vector<float> proposalProbs(batch * proposalLen * support);
    std::vector<int32_t> proposalIds(batch * proposalLen * support);
    std::vector<int32_t> proposalTokens(batch);
    std::vector<int32_t> proposalLengths(batch, proposalLen);
    std::vector<float> uniforms(batch * 3);
    std::mt19937 generator{12345};
    std::uniform_real_distribution<float> uniform(0.0F, 1.0F);
    for (int32_t b = 0; b < batch; ++b)
    {
        for (int32_t row = 0; row < verifyLen; ++row)
        {
            int32_t const offset = (b * verifyLen + row) * support;
            targetProbs[offset] = 0.7F;
            targetProbs[offset + 1] = 0.3F;
            targetIds[offset] = 10;
            targetIds[offset + 1] = 11;
        }
        proposalProbs[b * support] = 0.4F;
        proposalProbs[b * support + 1] = 0.6F;
        proposalIds[b * support] = 10;
        proposalIds[b * support + 1] = 11;
        proposalTokens[b] = uniform(generator) < 0.4F ? 10 : 11;
        uniforms[b * 3] = uniform(generator);
        uniforms[b * 3 + 1] = uniform(generator);
        uniforms[b * 3 + 2] = uniform(generator);
    }

    rt::Tensor targetProbsGpu({batch, verifyLen, support}, rt::DeviceType::kGPU, DataType::kFLOAT);
    rt::Tensor targetIdsGpu({batch, verifyLen, support}, rt::DeviceType::kGPU, DataType::kINT32);
    rt::Tensor proposalProbsGpu({batch, proposalLen, support}, rt::DeviceType::kGPU, DataType::kFLOAT);
    rt::Tensor proposalIdsGpu({batch, proposalLen, support}, rt::DeviceType::kGPU, DataType::kINT32);
    rt::Tensor proposalTokensGpu({batch, proposalLen}, rt::DeviceType::kGPU, DataType::kINT32);
    rt::Tensor proposalLengthsGpu({batch}, rt::DeviceType::kGPU, DataType::kINT32);
    rt::Tensor uniformsGpu({batch, 3}, rt::DeviceType::kGPU, DataType::kFLOAT);
    rt::Tensor accepted({batch, verifyLen}, rt::DeviceType::kGPU, DataType::kINT32);
    rt::Tensor lengths({batch}, rt::DeviceType::kGPU, DataType::kINT32);
    copyHostToDevice<float>(targetProbsGpu, targetProbs);
    copyHostToDevice<int32_t>(targetIdsGpu, targetIds);
    copyHostToDevice<float>(proposalProbsGpu, proposalProbs);
    copyHostToDevice<int32_t>(proposalIdsGpu, proposalIds);
    copyHostToDevice<int32_t>(proposalTokensGpu, proposalTokens);
    copyHostToDevice<int32_t>(proposalLengthsGpu, proposalLengths);
    copyHostToDevice<float>(uniformsGpu, uniforms);

    speculativeSparseAccept(targetProbsGpu, targetIdsGpu, proposalProbsGpu, proposalIdsGpu, proposalTokensGpu,
        proposalLengthsGpu, uniformsGpu, accepted, lengths, nullptr, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    auto const output = copyDeviceToHost<int32_t>(accepted);
    int32_t token10Count = 0;
    for (int32_t b = 0; b < batch; ++b)
    {
        token10Count += output[b * verifyLen] == 10;
    }
    float const observed = static_cast<float>(token10Count) / batch;
    EXPECT_NEAR(observed, 0.7F, 0.01F);
}

TEST(SpeculativeSamplingKernel, NonFiniteProposalDistributionRejectsAndSamplesTarget)
{
    constexpr int32_t batch{1};
    constexpr int32_t proposalLen{1};
    constexpr int32_t verifyLen{2};
    constexpr int32_t targetK{2};
    constexpr int32_t proposalK{2};
    rt::Tensor targetProbs({batch, verifyLen, targetK}, rt::DeviceType::kGPU, DataType::kFLOAT);
    rt::Tensor targetIds({batch, verifyLen, targetK}, rt::DeviceType::kGPU, DataType::kINT32);
    rt::Tensor proposalProbs({batch, proposalLen, proposalK}, rt::DeviceType::kGPU, DataType::kFLOAT);
    rt::Tensor proposalIds({batch, proposalLen, proposalK}, rt::DeviceType::kGPU, DataType::kINT32);
    rt::Tensor proposalTokens({batch, proposalLen}, rt::DeviceType::kGPU, DataType::kINT32);
    rt::Tensor proposalLengths({batch}, rt::DeviceType::kGPU, DataType::kINT32);
    rt::Tensor uniforms({batch, 2 * proposalLen + 1}, rt::DeviceType::kGPU, DataType::kFLOAT);
    rt::Tensor accepted({batch, verifyLen}, rt::DeviceType::kGPU, DataType::kINT32);
    rt::Tensor lengths({batch}, rt::DeviceType::kGPU, DataType::kINT32);

    copyHostToDevice<float>(targetProbs, {0.75F, 0.25F, 0.6F, 0.4F});
    copyHostToDevice<int32_t>(targetIds, {10, 11, 20, 21});
    copyHostToDevice<float>(proposalProbs, {NAN, NAN});
    copyHostToDevice<int32_t>(proposalIds, {10, 12});
    copyHostToDevice<int32_t>(proposalTokens, {10});
    copyHostToDevice<int32_t>(proposalLengths, {proposalLen});
    copyHostToDevice<float>(uniforms, {0.1F, 0.9F, 0.5F});

    speculativeSparseAccept(targetProbs, targetIds, proposalProbs, proposalIds, proposalTokens, proposalLengths,
        uniforms, accepted, lengths, nullptr, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    EXPECT_EQ(copyDeviceToHost<int32_t>(lengths), (std::vector<int32_t>{1}));
    EXPECT_EQ(copyDeviceToHost<int32_t>(accepted)[0], 11);
}
