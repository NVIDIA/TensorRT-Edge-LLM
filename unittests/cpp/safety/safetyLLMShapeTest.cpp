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

#include "builder/safety/safetyLLMShapeUtils.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <initializer_list>
#include <limits>
#include <stdexcept>

namespace trt_edgellm::safety::builder
{
namespace
{
nvinfer1::Dims dims(std::initializer_list<int64_t> values)
{
    nvinfer1::Dims result{};
    result.nbDims = static_cast<int32_t>(values.size());
    std::copy(values.begin(), values.end(), result.d);
    return result;
}

void expectShape(nvinfer1::Dims const& actual, std::initializer_list<int64_t> expected)
{
    ASSERT_EQ(actual.nbDims, expected.size());
    EXPECT_TRUE(std::equal(expected.begin(), expected.end(), actual.d));
}

nvinfer1::Dims resolve(std::string_view name, nvinfer1::Dims const& exported,
    SafetyLLMStaticShapeProfile const& profile = {3, 17, 513, 256})
{
    return getSafetyLLMInputShape(name, exported, profile, 1024, 4, 128, 64);
}
} // namespace

TEST(SafetyLLMShapeTest, ContiguousCachePreservesExactCapacity)
{
    // Given a non-page-aligned capacity, when resolving KV, then retain [B,2,Hkv,capacity,D] without padding.
    expectShape(resolve("past_key_values_0", dims({-1, 2, 4, -1, 128})), {3, 2, 4, 513, 128});
    expectShape(resolve("past_key_values_7", dims({-1, 2, 4, -1, 128}), {1, 1, 513, 0}), {1, 2, 4, 513, 128});
}

TEST(SafetyLLMShapeTest, ResolvesBatchedBindings)
{
    // Given legacy decoder inputs, when resolving shapes, then use explicit batch and sequence axes.
    expectShape(resolve("inputs_embeds", dims({-1, -1, 1024})), {3, 17, 1024});
    expectShape(resolve("rope_rotary_cos_sin", dims({-1, -1, 64})), {3, 513, 64});
    expectShape(resolve("context_lengths", dims({-1})), {3});
    expectShape(resolve("kvcache_start_index", dims({-1})), {3});
    expectShape(resolve("last_token_ids", dims({-1, 1})), {3, 1});
    expectShape(resolve("input_ids", dims({-1, -1})), {3, 17});
    expectShape(resolve("image_embeds", dims({-1, 1024})), {256, 1024});
    expectShape(resolve("packed_attention_mask", dims({-1, 1, 1})), {3, 1, 1});
    expectShape(resolve("attention_position_ids", dims({-1, 1})), {3, 1});
}

TEST(SafetyLLMShapeTest, RejectsTokenMajorAndPagedInputs)
{
    // Given Standard layouts, when validating any static or dynamic input, then reject rather than reinterpret axes.
    EXPECT_THROW(resolve("inputs_embeds", dims({-1, 1024})), std::invalid_argument);
    EXPECT_THROW(resolve("inputs_embeds", dims({51, 1024})), std::invalid_argument);
    EXPECT_THROW(resolve("inputs_embeds", dims({3, 17, 1024, 1})), std::invalid_argument);
    EXPECT_THROW(resolve("past_key_values_0", dims({2, -1, 128, 4, 128})), std::invalid_argument);
    for (auto const* name : {"kv_page_table", "positions", "query_start_offsets", "unknown_input"})
    {
        SCOPED_TRACE(name);
        EXPECT_THROW(resolve(name, dims({3, 2, 5})), std::invalid_argument);
    }
}

TEST(SafetyLLMShapeTest, ChecksFixedDimensions)
{
    // Given a fully static export, when applying MSS, then matching dimensions pass and mismatches fail.
    expectShape(resolve("past_key_values_0", dims({3, 2, 4, 513, 128})), {3, 2, 4, 513, 128});
    for (int32_t axis = 0; axis < 5; ++axis)
    {
        auto shape = dims({3, 2, 4, 513, 128});
        ++shape.d[axis];
        EXPECT_THROW(resolve("past_key_values_0", shape), std::invalid_argument);
    }
    EXPECT_THROW(resolve("inputs_embeds", dims({-2, -1, 1024})), std::invalid_argument);
}

TEST(SafetyLLMShapeTest, RejectsInvalidProfileDimensions)
{
    // Given invalid or overflowing profile sizes, when resolving an input, then fail before producing TRT dimensions.
    int64_t const tooLarge = static_cast<int64_t>(std::numeric_limits<int32_t>::max()) + 1;
    for (int64_t const value : {int64_t{0}, int64_t{-1}, tooLarge})
    {
        EXPECT_THROW(resolve("inputs_embeds", dims({-1, -1, 1024}), {value, 17, 513, 0}), std::invalid_argument);
        EXPECT_THROW(resolve("inputs_embeds", dims({-1, -1, 1024}), {3, value, 513, 0}), std::invalid_argument);
        EXPECT_THROW(resolve("inputs_embeds", dims({-1, -1, 1024}), {3, 17, value, 0}), std::invalid_argument);
        EXPECT_THROW(resolve("image_embeds", dims({-1, 1024}), {3, 17, 513, value}), std::invalid_argument);
    }
    EXPECT_THROW(resolve("inputs_embeds", dims({-1, -1, 1024}), {3, 514, 513, 0}), std::invalid_argument);
}

TEST(SafetyLLMShapeTest, RejectsInvalidModelDimensionsAndVolumeOverflow)
{
    // Given invalid model dimensions or a KV volume beyond int64, when resolving, then fail without overflow.
    int64_t const largest = std::numeric_limits<int32_t>::max();
    SafetyLLMStaticShapeProfile const profile{3, 17, 513, 0};
    for (int32_t axis = 0; axis < 4; ++axis)
    {
        for (int64_t const value : {int64_t{0}, int64_t{-1}, largest + 1})
        {
            int64_t model[] = {1024, 4, 128, 64};
            model[axis] = value;
            EXPECT_THROW(getSafetyLLMInputShape(
                             "inputs_embeds", dims({-1, -1, -1}), profile, model[0], model[1], model[2], model[3]),
                std::invalid_argument);
        }
    }
    EXPECT_THROW(
        getSafetyLLMInputShape("inputs_embeds", dims({-1, -1, -1}), profile, 1024, 4, 128, 129), std::invalid_argument);
    EXPECT_THROW(
        resolve("past_key_values_0", dims({-1, 2, 4, -1, 128}), {largest, 1, largest, 0}), std::invalid_argument);
}
} // namespace trt_edgellm::safety::builder
