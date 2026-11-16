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

// Contract tests: host-side, no GPU and no engine. The bounds below are the
// percentiles the builder resolves out of GR00T-N1.7-LIBERO/libero_10.

#include "action/gr00tPolicy.h"
#include "multimodal/gr00tVisualRunner.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <string>
#include <vector>

using namespace trt_edgellm;

namespace
{

//! The action graph's output shape, which is embodiment-independent and wider
//! than any single robot.
constexpr int32_t kModelHorizon = 40;
constexpr int32_t kActionDim = 132;
constexpr int32_t kStateDim = 132;

constexpr char const* kLiberoContract = R"JSON({
  "export_id": "1111111111111111aaaaaaaaaaaaaaaa",
  "frontend": "direct",
  "embodiment_tag": "libero_sim",
  "embodiment_id": 2,
  "use_percentiles": true,
  "clip_outliers": true,
  "state": {
    "keys": ["x", "y", "z", "roll", "pitch", "yaw", "gripper"],
    "dims": [1, 1, 1, 1, 1, 1, 2],
    "total_dim": 8,
    "min": [-0.38999009132385254, -0.28383004665374756, 0.4479505717754364, 1.8810230493545532,
            -2.8866775035858154, -1.159900426864624, 0.0020664597395807505, -0.0400138795375824],
    "max": [0.15302616357803345, 0.32915401458740234, 1.2546923160552979, 3.3035426139831543,
            2.7496531009674072, 0.6893712878227234, 0.04004855826497078, -0.0017598443664610386],
    "history": 1
  },
  "action": {
    "keys": ["x", "y", "z", "roll", "pitch", "yaw", "gripper"],
    "dims": [1, 1, 1, 1, 1, 1, 1],
    "total_dim": 7,
    "min": [-0.6348214149475098, -0.7741071581840515, -0.7633928656578064, -0.09749999642372131,
            -0.14820000529289246, -0.2742857038974762, 0.0],
    "max": [0.7714285850524902, 0.8464285731315613, 0.9375, 0.13928571343421936,
            0.15964286029338837, 0.3246428668498993, 1.0],
    "horizon": 16
  },
  "video_keys": ["image", "wrist_image"],
  "vision": {
    "shortest_edge": 256, "crop_fraction": 0.95, "tokens_per_view": 64, "formalize_language": true
  }
})JSON";

class Gr00tPolicyTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        mDir = std::filesystem::temp_directory_path() / "edgellm_gr00t_policy_test";
        std::filesystem::remove_all(mDir);
        std::filesystem::create_directories(mDir / "action");
    }

    void TearDown() override
    {
        std::filesystem::remove_all(mDir);
    }

    gr00t::Gr00tPolicy makePolicy(std::string const& contract = kLiberoContract)
    {
        std::ofstream(mDir / "action" / "policy.json") << contract;
        return gr00t::Gr00tPolicy(mDir.string());
    }

    std::filesystem::path mDir;
};

void expectResizeMatches(std::vector<unsigned char> source, int64_t height, int64_t width, int64_t maxSize,
    int64_t expectedHeight, int64_t expectedWidth, std::vector<unsigned char> const& golden)
{
    gr00t::smallestMaxSize(source, height, width, 3, maxSize);

    EXPECT_EQ(height, expectedHeight);
    EXPECT_EQ(width, expectedWidth);
    ASSERT_EQ(source.size(), golden.size());
    for (size_t i = 0; i < golden.size(); ++i)
    {
        // cv2 rounds the accumulator in fixed point, so one LSB is the floor.
        EXPECT_NEAR(static_cast<int>(source[i]), static_cast<int>(golden[i]), 1) << "element " << i;
    }
}

} // namespace

//! The reference's min-max map puts q01 at -1 and q99 at +1, clips beyond them, and
//! the state is zero-padded to the graph's width past the embodiment's own dimensions.
TEST_F(Gr00tPolicyTest, StateNormalizesToTheUnitRangeAndZeroPads)
{
    gr00t::Gr00tPolicy const policy = makePolicy();
    // q01 on dim 0, q99 on dim 1, the midpoint on dim 2, a quarter of the range on
    // dims 3-6 and a value past q99 on dim 7, so one vector pins endpoints, interior and clip.
    std::vector<float> const robot{
        -0.389990091F, 0.329154015F, 0.851321444F, 2.23665294F, -1.47759485F, -0.697582498F, 0.0115619844F, 0.5F};

    std::vector<float> const normalized = policy.normalizeState(robot, kStateDim);

    ASSERT_EQ(normalized.size(), static_cast<size_t>(kStateDim));
    std::vector<float> const expected{-1.0F, 1.0F, 0.0F, -0.5F, -0.5F, -0.5F, -0.5F, 1.0F};
    for (size_t i = 0; i < expected.size(); ++i)
    {
        EXPECT_NEAR(normalized[i], expected[i], 1e-5F) << "dim " << i;
    }
    for (size_t i = expected.size(); i < normalized.size(); ++i)
    {
        EXPECT_EQ(normalized[i], 0.0F) << "padding at " << i;
    }
}

//! A joint group whose percentiles coincide has no range to map onto; the
//! reference emits zero there instead of dividing by it.
TEST_F(Gr00tPolicyTest, DegenerateStateDimensionNormalizesToZero)
{
    std::string contract(kLiberoContract);
    auto const at = contract.find("0.0020664597395807505");
    ASSERT_NE(at, std::string::npos);
    contract.replace(at, std::string("0.0020664597395807505").size(), "0.04004855826497078");

    gr00t::Gr00tPolicy const policy = makePolicy(contract);
    std::vector<float> robot(8, 0.0F);
    robot[6] = 0.04004855826497078F;

    std::vector<float> const normalized = policy.normalizeState(robot, kStateDim);

    EXPECT_EQ(normalized[6], 0.0F);
}

//! The chunk comes back in robot units on the policy's own horizon and width,
//! after the reference's clip to [-1, 1].
TEST_F(Gr00tPolicyTest, ActionUnnormalizesToRobotUnitsAndDropsPadding)
{
    gr00t::Gr00tPolicy const policy = makePolicy();
    std::vector<float> chunk(static_cast<size_t>(kModelHorizon) * kActionDim, 7.0F);
    // +-1 land on q99 and q01; the last value is out of range, so it also pins
    // the clip the reference applies before the map.
    std::vector<float> const row{1.0F, 0.5F, 0.0F, -0.5F, -1.0F, 0.25F, 3.0F};
    std::copy(row.begin(), row.end(), chunk.begin());

    std::vector<float> const robot = policy.unnormalizeAction(chunk, kModelHorizon, kActionDim);

    // The policy emits its own horizon, not the graph's.
    ASSERT_EQ(robot.size(), static_cast<size_t>(policy.horizon()) * policy.action().totalDim());
    std::vector<float> const expected{
        0.771428585F, 0.44129464F, 0.0870535672F, -0.038303569F, -0.148200005F, 0.100044653F, 1.0F};
    for (size_t d = 0; d < expected.size(); ++d)
    {
        EXPECT_NEAR(robot[d], expected[d], 1e-6F) << "dim " << d;
    }
}

//! Every contract field the runtime reads comes out of policy.json as the builder wrote it.
TEST_F(Gr00tPolicyTest, PolicyContractIsParsed)
{
    gr00t::Gr00tPolicy const policy = makePolicy();

    EXPECT_EQ(policy.embodimentTag(), "libero_sim");
    EXPECT_EQ(policy.horizon(), 16);
    EXPECT_EQ(policy.action().totalDim(), 7);
    EXPECT_EQ(policy.vision().shortestEdge, 256);
    EXPECT_NEAR(policy.vision().cropFraction, 0.95F, 1e-6F);
    EXPECT_EQ(policy.vision().tokensPerView, 64);
    ASSERT_EQ(policy.videoKeys().size(), 2U);
    EXPECT_EQ(policy.videoKeys()[0], "image");
    EXPECT_EQ(policy.videoKeys()[1], "wrist_image");
}

//! Downscale: the shorter side 6 goes to 4, so 8 goes to 5 and both axes
//! average partial source pixels.
TEST_F(Gr00tPolicyTest, DownscaleMatchesTheReferenceAreaResampler)
{
    std::vector<unsigned char> const source{139, 74, 229, 241, 169, 65, 6, 160, 149, 106, 38, 175, 188, 205, 175, 229,
        98, 249, 10, 148, 95, 86, 147, 198, 66, 39, 106, 213, 171, 45, 167, 57, 69, 81, 55, 14, 153, 178, 215, 76, 52,
        39, 250, 72, 215, 50, 161, 223, 112, 172, 161, 233, 38, 17, 89, 1, 221, 6, 242, 127, 143, 6, 60, 210, 94, 25,
        166, 33, 249, 189, 12, 204, 33, 57, 124, 30, 199, 149, 202, 119, 72, 220, 3, 209, 122, 136, 147, 77, 133, 82,
        115, 87, 162, 230, 70, 71, 240, 62, 47, 184, 31, 34, 63, 65, 146, 197, 142, 253, 81, 133, 240, 113, 30, 247,
        103, 122, 31, 19, 42, 129, 141, 136, 33, 149, 161, 0, 178, 141, 107, 35, 107, 130, 72, 27, 217, 254, 99, 13,
        195, 206, 59, 228, 235, 202};
    std::vector<unsigned char> const golden{159, 103, 139, 107, 126, 106, 137, 120, 155, 156, 107, 169, 80, 141, 180,
        145, 111, 99, 132, 55, 124, 89, 122, 100, 170, 69, 92, 162, 56, 221, 94, 90, 103, 144, 105, 116, 160, 92, 201,
        97, 103, 137, 141, 75, 101, 169, 114, 28, 139, 117, 99, 82, 90, 198, 180, 100, 97, 161, 173, 130};

    expectResizeMatches(source, 6, 8, 4, 4, 5, golden);
}

//! Upscale: the crop step leaves the shorter side below the target, so the
//! second SmallestMaxSize always runs in this direction.
TEST_F(Gr00tPolicyTest, UpscaleMatchesTheReferenceAreaResampler)
{
    std::vector<unsigned char> const source{183, 47, 69, 179, 82, 35, 71, 159, 110, 121, 77, 87, 3, 126, 44, 253, 233,
        225, 90, 119, 137, 120, 30, 55, 33, 165, 87, 216, 229, 167, 3, 41, 109, 203, 129, 219, 63, 101, 207, 156, 27,
        220, 93, 29, 143, 200, 63, 11};
    std::vector<unsigned char> const golden{183, 47, 69, 180, 73, 44, 125, 121, 73, 84, 139, 104, 121, 77, 87, 48, 106,
        50, 188, 173, 146, 160, 162, 154, 94, 107, 113, 120, 42, 63, 18, 146, 66, 180, 210, 163, 141, 156, 160, 75, 80,
        127, 162, 80, 137, 41, 149, 117, 161, 171, 164, 113, 108, 149, 70, 57, 130, 202, 113, 167, 63, 101, 207, 133,
        46, 217, 125, 28, 182, 120, 38, 110, 200, 63, 11};

    expectResizeMatches(source, 4, 4, 5, 5, 5, golden);
}
