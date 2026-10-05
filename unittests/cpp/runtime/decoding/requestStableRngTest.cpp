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

#include "runtime/decoding/requestStableRng.h"
#include "common/cudaUtils.h"
#include "kernels/speculative/requestStableRngKernels.h"
#include "testUtils.h"

#include <NvInfer.h>
#include <gtest/gtest.h>

#include <array>
#include <cstdint>

using namespace trt_edgellm;
using namespace trt_edgellm::rt;

TEST(RequestStableRngTest, SameRequestPositionAndPurposeIsStable)
{
    constexpr uint64_t seed = 123456789ULL;
    for (uint64_t position = 0; position < 128; ++position)
    {
        for (uint64_t lane = 0; lane < 16; ++lane)
        {
            float const first = requestStableUniform(seed, position, SpecRandomPurpose::kProposal, lane);
            float const replay = requestStableUniform(seed, position, SpecRandomPurpose::kProposal, lane);
            EXPECT_FLOAT_EQ(first, replay);
            EXPECT_GT(first, 0.0F);
            EXPECT_LT(first, 1.0F);
        }
    }
}

TEST(RequestStableRngTest, PurposesUseDisjointStreams)
{
    constexpr uint64_t seed = 42ULL;
    constexpr uint64_t position = 17ULL;
    std::array<float, 4> values{
        requestStableUniform(seed, position, SpecRandomPurpose::kProposal, 0),
        requestStableUniform(seed, position, SpecRandomPurpose::kAccept, 0),
        requestStableUniform(seed, position, SpecRandomPurpose::kResidual, 0),
        requestStableUniform(seed, position, SpecRandomPurpose::kBonus, 0),
    };
    for (size_t i = 0; i < values.size(); ++i)
    {
        for (size_t j = i + 1; j < values.size(); ++j)
        {
            EXPECT_NE(values[i], values[j]);
        }
    }
}

TEST(RequestStableRngTest, AbsolutePositionDoesNotDependOnBatchSlotOrRound)
{
    constexpr uint64_t seed = 987654321ULL;
    constexpr uint64_t absolutePosition = 23ULL;
    float const batchOne = requestStableUniform(seed, absolutePosition, SpecRandomPurpose::kAccept, 3);
    float const afterReorder = requestStableUniform(seed, absolutePosition, SpecRandomPurpose::kAccept, 3);
    EXPECT_FLOAT_EQ(batchOne, afterReorder);
}

TEST(RequestStableRngTest, AbsolutePositionUsesFullPromptAcrossContextCacheReuse)
{
    constexpr uint64_t kFullPromptTokens{1536};
    constexpr uint64_t kGeneratedTokens{37};
    constexpr uint64_t kCachedReplaySuffixTokens{129};

    auto const absolutePosition = requestStableNextAbsolutePosition(kFullPromptTokens, kGeneratedTokens);
    EXPECT_EQ(absolutePosition, 1573);
    EXPECT_NE(absolutePosition, kCachedReplaySuffixTokens + kGeneratedTokens);
}

TEST(RequestStableRngTest, SamplingPolicyPreservesLegacyVanillaByDefault)
{
    EXPECT_FALSE(shouldUseRequestStableSampling(/*supportsLosslessSampling=*/false, /*hasExplicitSeed=*/false));
    EXPECT_TRUE(shouldUseRequestStableSampling(/*supportsLosslessSampling=*/false, /*hasExplicitSeed=*/true));
    EXPECT_TRUE(shouldUseRequestStableSampling(/*supportsLosslessSampling=*/true, /*hasExplicitSeed=*/false));
}

TEST(RequestStableRngTest, CudaFillMatchesLogicalRequestHashAfterSlotReorder)
{
    constexpr int32_t batch{2};
    constexpr int32_t proposalLen{3};
    rt::Tensor seeds({batch}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT64);
    rt::Tensor positions({batch}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT64);
    rt::Tensor proposals({batch, proposalLen}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT);
    rt::Tensor accepts({batch, 2 * proposalLen + 1}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT);
    copyHostToDevice<int64_t>(seeds, {29, 17});
    copyHostToDevice<int64_t>(positions, {103, 41});

    kernel::launchRequestStableSpecUniforms(reinterpret_cast<uint64_t const*>(seeds.rawPointer()),
        reinterpret_cast<uint64_t const*>(positions.rawPointer()), proposals.dataPointer<float>(),
        accepts.dataPointer<float>(), batch, proposalLen, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    auto const proposalHost = copyDeviceToHost<float>(proposals);
    auto const acceptHost = copyDeviceToHost<float>(accepts);
    for (int32_t b = 0; b < batch; ++b)
    {
        uint64_t const seed = b == 0 ? 29 : 17;
        uint64_t const position = b == 0 ? 103 : 41;
        for (int32_t step = 0; step < proposalLen; ++step)
        {
            EXPECT_FLOAT_EQ(proposalHost[b * proposalLen + step],
                requestStableUniform(seed, position + step, SpecRandomPurpose::kProposal, 0));
            EXPECT_FLOAT_EQ(acceptHost[b * (2 * proposalLen + 1) + step],
                requestStableUniform(seed, position + step, SpecRandomPurpose::kAccept, 0));
        }
    }
}
