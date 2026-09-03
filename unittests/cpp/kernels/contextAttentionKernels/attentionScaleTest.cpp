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

#include "kernels/contextAttentionKernels/attentionScaleUtils.h"

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <optional>
#include <stdexcept>
#include <vector>

using namespace trt_edgellm;

TEST(AttentionScaleTest, ResolvesLegacyDefault)
{
    int32_t constexpr kHEAD_SIZE = 128;
    float const expected = 1.0F / std::sqrt(static_cast<float>(kHEAD_SIZE));

    EXPECT_FLOAT_EQ(resolveAttentionScale(std::nullopt, kHEAD_SIZE), expected);
}

TEST(AttentionScaleTest, PreservesConfiguredValue)
{
    float constexpr kCUSTOM_SCALE = 0.37F;

    EXPECT_FLOAT_EQ(resolveAttentionScale(kCUSTOM_SCALE, /*headSize=*/0), kCUSTOM_SCALE);
}

TEST(AttentionScaleTest, RejectsNonPositiveAndNonFiniteValues)
{
    std::vector<float> const invalidValues{0.0F, -1.0F, std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN()};

    for (float const value : invalidValues)
    {
        EXPECT_THROW((void) resolveAttentionScale(value, /*headSize=*/0), std::runtime_error);
    }
}

TEST(AttentionScaleTest, RejectsInvalidHeadSizeForLegacyDefault)
{
    EXPECT_THROW((void) resolveAttentionScale(std::nullopt, 0), std::runtime_error);
    EXPECT_THROW((void) resolveAttentionScale(std::nullopt, -1), std::runtime_error);
}
