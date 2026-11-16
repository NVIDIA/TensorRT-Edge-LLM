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

#include "builder/safety/internVLConfig.h"
#include "builder/safety/safetyVisualShapeUtils.h"

#include <gtest/gtest.h>

#include <limits>
#include <stdexcept>

namespace trt_edgellm::safety::builder
{
using Json = nlohmann::json;

namespace
{
Json modelConfig(double ratio = 0.5)
{
    return {{"vision_config", {{"image_size", {448, 448}}, {"patch_size", {14, 14}}}}, {"downsample_ratio", ratio}};
}
} // namespace

TEST(SafetyVisualShapeTest, StandardAndCustomDownsampling)
{
    // Given 256/64-token models, when resolving MSS 512, then select two/eight tiles respectively.
    EXPECT_EQ(getInternVLTokensPerBlock(modelConfig()), 256);
    EXPECT_EQ(getInternVLTokensPerBlock(modelConfig(0.25)), 64);
    EXPECT_EQ(getInternVLTokensPerBlock(modelConfig(1.0)), 1024);
    EXPECT_EQ(getSafetyVisualTileCount(512, getInternVLTokensPerBlock(modelConfig())), 2);
    EXPECT_EQ(getSafetyVisualTileCount(512, getInternVLTokensPerBlock(modelConfig(0.25))), 8);
    EXPECT_EQ(getSafetyVisualTileCount(64, 64), 1);
}

TEST(SafetyVisualShapeTest, UsesBothSpatialDimensions)
{
    // Given a rectangular tile and patches, when deriving tokens, then use both axes without squaring height.
    auto config = modelConfig();
    config["vision_config"]["image_size"] = {224, 448};
    config["vision_config"]["patch_size"] = {14, 28};
    EXPECT_EQ(getInternVLTokensPerBlock(config), 64);
    config["vision_config"]["patch_size"] = {14, 14};
    EXPECT_EQ(getInternVLTokensPerBlock(config), 128);
}

TEST(SafetyVisualShapeTest, SupportsScalarModelDimensions)
{
    // Given scalar model dimensions, when deriving tokens, then match normalized [height,width] exports.
    auto config = modelConfig(0.25);
    config["vision_config"]["image_size"] = 448;
    config["vision_config"]["patch_size"] = 14;
    EXPECT_EQ(getInternVLTokensPerBlock(config), 64);
}

TEST(SafetyVisualShapeTest, MatchesExporterPixelShuffleTruncation)
{
    // Given a non-integral reciprocal, when deriving tokens, then truncate the scale like the exported graph.
    EXPECT_EQ(getInternVLTokensPerBlock(modelConfig(0.375)), 256);
    EXPECT_EQ(getInternVLTokensPerBlock(modelConfig(0.6)), 1024);
    EXPECT_EQ(getInternVLTokensPerBlock(modelConfig(0.5 - 1e-9)), 256);
    EXPECT_EQ(getInternVLTokensPerBlock(modelConfig(0.5 + 1e-9)), 1024);
}

TEST(SafetyVisualShapeTest, MatchesExporterForNonPowerOfTwoScale)
{
    // Given a grid divisible by three, when pixel-shuffle uses scale three, then derive the exact output grid.
    auto config = modelConfig(1.0 / 3.0);
    config["vision_config"]["image_size"] = {420, 420};
    EXPECT_EQ(getInternVLTokensPerBlock(config), 100);
    config["vision_config"]["image_size"] = {448, 448};
    EXPECT_THROW(getInternVLTokensPerBlock(config), std::invalid_argument);
}

TEST(SafetyVisualShapeTest, BoundsTokenProduct)
{
    // Given the largest supported dimension, when multiplying both axes, then preserve int64 without overflow.
    auto config = modelConfig(1.0);
    int64_t const largest = std::numeric_limits<int32_t>::max();
    config["vision_config"]["image_size"] = {largest, largest};
    config["vision_config"]["patch_size"] = {1, 1};
    EXPECT_EQ(getInternVLTokensPerBlock(config), largest * largest);
    config["vision_config"]["image_size"] = {largest + 1, largest};
    EXPECT_THROW(getInternVLTokensPerBlock(config), std::invalid_argument);
}

TEST(SafetyVisualShapeTest, RequiresExportedDownsampleRatio)
{
    // Given a legacy config without the ratio, when resolving tile tokens, then fail instead of assuming 256.
    auto config = modelConfig();
    config.erase("downsample_ratio");
    EXPECT_THROW(getInternVLTokensPerBlock(config), std::invalid_argument);
}

TEST(SafetyVisualShapeTest, RejectsInvalidDownsampleRatios)
{
    // Given invalid or fractional downsampled grids, when resolving tokens, then reject without rounding.
    for (double const ratio :
        {0.0, -0.5, 1.5, 0.3, 0.01, std::numeric_limits<double>::min(), std::numeric_limits<double>::denorm_min(),
            std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN()})
    {
        SCOPED_TRACE(ratio);
        EXPECT_THROW(getInternVLTokensPerBlock(modelConfig(ratio)), std::invalid_argument);
    }
    auto config = modelConfig();
    config["downsample_ratio"] = "0.5";
    EXPECT_THROW(getInternVLTokensPerBlock(config), std::invalid_argument);
}

TEST(SafetyVisualShapeTest, RejectsInvalidSpatialDimensions)
{
    // Given malformed dimensions, when deriving tile tokens, then fail before division, truncation, or overflow.
    for (auto const* key : {"image_size", "patch_size"})
    {
        for (Json const& invalid : {Json(448.5), Json{448}, Json{448, 448, 448}, Json{0, 14}, Json{14, -1},
                 Json{448.5, 448}, Json{"448", 448}, Json{std::numeric_limits<uint64_t>::max(), 14}})
        {
            auto config = modelConfig();
            config["vision_config"][key] = invalid;
            SCOPED_TRACE(std::string(key) + invalid.dump());
            EXPECT_THROW(getInternVLTokensPerBlock(config), std::invalid_argument);
        }
    }
    auto config = modelConfig();
    config["vision_config"]["image_size"] = {449, 448};
    EXPECT_THROW(getInternVLTokensPerBlock(config), std::invalid_argument);
}

TEST(SafetyVisualShapeTest, RejectsInvalidMssWithoutRounding)
{
    // Given zero, negative, or partial-tile MSS values, when converting to tiles, then reject rather than round.
    for (int64_t const tokens : {0, -64, 63, 65})
    {
        EXPECT_THROW(getSafetyVisualTileCount(tokens, 64), std::invalid_argument);
    }
    EXPECT_THROW(getSafetyVisualTileCount(512, 0), std::invalid_argument);
    EXPECT_THROW(getSafetyVisualTileCount(512, -64), std::invalid_argument);
}
} // namespace trt_edgellm::safety::builder
