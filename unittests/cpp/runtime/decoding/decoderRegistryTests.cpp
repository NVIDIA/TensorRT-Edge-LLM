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

#include "runtime/decoding/decoderRegistry.h"

#include <gtest/gtest.h>

#include <cstdint>

using namespace trt_edgellm;

namespace
{

rt::LLMGenerationRequest makeSamplingRequest(float temperature, int64_t topK, float topP)
{
    rt::LLMGenerationRequest request{};
    request.temperature = temperature;
    request.topK = topK;
    request.topP = topP;
    return request;
}

TEST(DecoderRegistryPolicyTest, EagleUsesDefaultDecoderOnlyForEffectiveNonGreedySampling)
{
    auto request = makeSamplingRequest(1.0F, 1, 1.0F);
    EXPECT_FALSE(rt::shouldSelectDefaultDecoder(rt::DecodingStrategyKind::kEAGLE, request));

    request = makeSamplingRequest(1.0F, 50, 1.0F);
    EXPECT_TRUE(rt::shouldSelectDefaultDecoder(rt::DecodingStrategyKind::kEAGLE, request));

    // SamplingParams normalizes this common configuration to greedy top-1.
    request = makeSamplingRequest(0.0F, 50, 1.0F);
    EXPECT_FALSE(rt::shouldSelectDefaultDecoder(rt::DecodingStrategyKind::kEAGLE, request));
}

TEST(DecoderRegistryPolicyTest, ExplicitDisableAndOtherSpeculativeModesRemainUnchanged)
{
    auto request = makeSamplingRequest(1.0F, 50, 1.0F);
    EXPECT_FALSE(rt::shouldSelectDefaultDecoder(rt::DecodingStrategyKind::kMTP, request));
    EXPECT_FALSE(rt::shouldSelectDefaultDecoder(rt::DecodingStrategyKind::kDFlash, request));
    EXPECT_FALSE(rt::shouldSelectDefaultDecoder(rt::DecodingStrategyKind::kGemma4MTP, request));
    EXPECT_FALSE(rt::shouldSelectDefaultDecoder(rt::DecodingStrategyKind::kDSpark, request));

    request.disableSpecDecode = true;
    EXPECT_TRUE(rt::shouldSelectDefaultDecoder(rt::DecodingStrategyKind::kEAGLE, request));
    EXPECT_TRUE(rt::shouldSelectDefaultDecoder(rt::DecodingStrategyKind::kMTP, request));
}

} // namespace
