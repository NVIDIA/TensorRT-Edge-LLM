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

// Contract tests: host-side, no GPU and no engine.

#include "runtime/pi05Policy.h"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <sstream>
#include <string>
#include <vector>

using namespace trt_edgellm;

namespace
{

//! One export run's stamp; the engines staged beside a contract carry the same one.
constexpr char const* kExportId = "1111111111111111aaaaaaaaaaaaaaaa";

//! The pi05_libero checkpoint's own norm_stats.json.
constexpr char const* kLiberoNormStats = R"JSON({
  "norm_stats": {
    "state": {
      "q01": [-0.3524468903720379, -0.26824864755272865, 0.04083745917417109, 1.5317653684616088,
              -2.7152330031871794, -1.076538143157959, 0.001715825623134151, -0.04003722561979666],
      "q99": [0.13891278689503672, 0.3251991607129573, 1.2568962905768304, 3.26276856803894,
              2.4437233173847197, 0.5638469840288161, 0.04030780866963323, -0.0017131616945378486]
    },
    "actions": {
      "q01": [-0.747375, -0.796125, -0.9375, -0.11580300460159779, -0.16942972007393836,
              -0.194502209174633, -1.0],
      "q99": [0.937125, 0.8594999999999999, 0.937125, 0.1402260055720806, 0.18103543001413347,
              0.3115457148551941, 0.9996]
    }
  }
})JSON";

//! Even statistics over the fourteen ALOHA joints, shared with the DROID contract at its
//! first eight. The goldens below were generated against exactly these.
constexpr char const* kEvenNormStats = R"JSON({
  "norm_stats": {
    "state": {
      "q01": [-1.2, -1.1231, -1.0462, -0.9692, -0.8923, -0.8154, -0.7385, -0.6615, -0.5846,
              -0.5077, -0.4308, -0.3538, -0.2769, -0.2],
      "q99": [0.3, 0.3846, 0.4692, 0.5538, 0.6385, 0.7231, 0.8077, 0.8923, 0.9769, 1.0615,
              1.1462, 1.2308, 1.3154, 1.4]
    },
    "actions": {
      "q01": [-1.2, -1.1231, -1.0462, -0.9692, -0.8923, -0.8154, -0.7385, -0.6615, -0.5846,
              -0.5077, -0.4308, -0.3538, -0.2769, -0.2],
      "q99": [0.3, 0.3846, 0.4692, 0.5538, 0.6385, 0.7231, 0.8077, 0.8923, 0.9769, 1.0615,
              1.1462, 1.2308, 1.3154, 1.4]
    }
  }
})JSON";

//! One openpi configuration as the exporter transcribes it, with what a test needs to
//! drive it: the statistics it normalizes against and a state in the robot's own units.
struct ContractCase
{
    char const* adapter;
    char const* policyConfig;
    char const* normStats;
    int32_t stateDim;
    int32_t robotActionDim;
    int32_t horizon;
    bool discreteStateInput;
    char const* cameras; //!< the manifest's "cameras" object
    std::vector<std::string> required;
    std::vector<std::string> optional;
    std::vector<std::string> ignored;
    std::vector<float> state;
    char const* task;
    //! The prompt openpi builds from \c state, empty where the task text is the prompt.
    char const* prompt;
};

std::vector<ContractCase> const& contractCases()
{
    static std::vector<ContractCase> const cases{
        {"libero", "pi05_libero", kLiberoNormStats, 8, 7, 10, false,
            R"JSON({"slots": [{"name": "observation/image", "required": true},
                              {"name": "observation/wrist_image", "required": true}], "ignored": []})JSON",
            {"observation/image", "observation/wrist_image"}, {}, {},
            {0.0F, 0.1F, 0.2F, 1.6F, 0.0F, -0.5F, 0.01F, -0.02F}, "pick_up the black bowl", "pick up the black bowl"},
        {"droid", "pi05_droid", kEvenNormStats, 8, 8, 15, true,
            R"JSON({"slots": [{"name": "observation/exterior_image_1_left", "required": true},
                              {"name": "observation/wrist_image_left", "required": true}], "ignored": []})JSON",
            {"observation/exterior_image_1_left", "observation/wrist_image_left"}, {}, {},
            {0.11F, -0.42F, 0.83F, -0.05F, 0.6F, -0.9F, 0.27F, 0.5F}, "put the mug on the plate",
            "Task: put the mug on the plate, State: 223 119 255 154 249 -1 166 191;\nAction: "},
        {"aloha", "pi05_aloha", kEvenNormStats, 14, 14, 50, true,
            R"JSON({"slots": [{"name": "cam_high", "required": true},
                              {"name": "cam_left_wrist", "required": false},
                              {"name": "cam_right_wrist", "required": false}],
                    "ignored": ["cam_low"]})JSON",
            {"cam_high"}, {"cam_left_wrist", "cam_right_wrist"}, {"cam_low"},
            {0.998F, -0.0701F, -0.7932F, -0.4675F, 0.5225F, -0.7262F, 0.35F, -0.5043F, -0.8981F, 0.135F, -0.9007F,
                -0.1727F, -0.4471F, 0.72F},
            "fold_the towel", "Task: fold the towel, State: 255 202 255 84 236 14 63 25 243 60 -1 29 -1 60;\nAction: "},
    };
    return cases;
}

//! Stage one engine directory's worth of contract files and hand back its path.
class Pi05PolicyTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        mDir = std::filesystem::temp_directory_path() / "edgellm_pi05_policy_test";
        std::filesystem::remove_all(mDir);
        std::filesystem::create_directories(mDir / "assets");
    }

    void TearDown() override
    {
        std::filesystem::remove_all(mDir);
    }

    pi05::Pi05Policy makePolicy(ContractCase const& c)
    {
        std::ofstream(mDir / "assets" / "norm_stats.json") << c.normStats;
        std::ofstream(mDir / "policy.json") << R"JSON({
  "contract_version": 1,
  "model_family": "pi05",
  "policy_config": ")JSON" << c.policyConfig << R"JSON(",
  "adapter": ")JSON" << c.adapter << R"JSON(",
  "discrete_state_input": )JSON" << (c.discreteStateInput ? "true" : "false")
                                            << R"JSON(,
  "export_id": ")JSON" << kExportId << R"JSON(",
  "state": {"dim": )JSON" << c.stateDim << R"JSON(, "num_bins": 256},
  "action": {"dim": )JSON" << c.robotActionDim
                                            << R"JSON(, "max_dim": 32, "horizon": )JSON" << c.horizon << R"JSON(},
  "cameras": )JSON" << c.cameras << R"JSON(,
  "image_resolution": [224, 224],
  "tokenizer": {"max_length": 200, "vocab_size": 257152, "add_bos": true, "add_eos": false}
})JSON";
        return pi05::Pi05Policy(mDir.string());
    }

    //! Stage a bundle whose policy.json is \p manifest verbatim, for the rejection cases.
    void writeManifest(char const* normStats, std::string const& manifest)
    {
        std::ofstream(mDir / "assets" / "norm_stats.json") << normStats;
        std::ofstream(mDir / "policy.json") << manifest;
    }

    //! Views naming \p names, each carrying a path so the source check passes.
    static std::vector<pi05::Pi05CameraView> viewsNamed(std::vector<std::string> const& names)
    {
        std::vector<pi05::Pi05CameraView> views;
        for (std::string const& name : names)
        {
            views.push_back(pi05::Pi05CameraView{name, "frame.png"});
        }
        return views;
    }

    std::filesystem::path mDir;
};

//! openpi resizes uint8 and only then maps to [-1, 1], so the goldens are the reference's
//! own bytes.
void expectMatchesGolden(std::vector<float> const& planar, std::vector<uint8_t> const& golden)
{
    ASSERT_EQ(planar.size(), golden.size());
    for (size_t i = 0; i < golden.size(); ++i)
    {
        EXPECT_NEAR(planar[i], static_cast<float>(golden[i]) / 255.0F * 2.0F - 1.0F, 1e-6F) << "element " << i;
    }
}

} // namespace

TEST_F(Pi05PolicyTest, ContractsFollowTheOpenpiConfigurations)
{
    for (ContractCase const& c : contractCases())
    {
        SCOPED_TRACE(c.policyConfig);
        pi05::Pi05Policy const policy = makePolicy(c);
        pi05::Pi05Contract const& contract = policy.contract();
        EXPECT_EQ(contract.stateDim, c.stateDim);
        EXPECT_EQ(contract.robotActionDim, c.robotActionDim);
        EXPECT_EQ(contract.actionHorizon, c.horizon);
        EXPECT_EQ(contract.discreteStateInput, c.discreteStateInput);
        EXPECT_EQ(contract.maxTokenLen, 200);

        // The prompt: task text alone, or openpi's discretized-state template.
        std::vector<float> const adapted = policy.adaptInputState(c.state);
        EXPECT_EQ(policy.buildPrompt(c.task, adapted), c.prompt);

        // Required slots alone are enough, and the optional ones are simply left out.
        std::vector<pi05::Pi05CameraView> const required = viewsNamed(c.required);
        EXPECT_EQ(policy.resolveActiveViews(required).size(), c.required.size());
        std::vector<std::string> every = c.required;
        every.insert(every.end(), c.optional.begin(), c.optional.end());
        EXPECT_EQ(policy.resolveActiveViews(viewsNamed(every)).size(), every.size());

        // A name the configuration ignores is dropped rather than rejected; one it has
        // never heard of is rejected, since a swapped view produces a plausible chunk.
        std::vector<std::string> withIgnored = c.required;
        withIgnored.insert(withIgnored.end(), c.ignored.begin(), c.ignored.end());
        EXPECT_EQ(policy.resolveActiveViews(viewsNamed(withIgnored)).size(), c.required.size());
        std::vector<std::string> unknown = c.required;
        unknown.emplace_back("cam_nonexistent");
        EXPECT_THROW(policy.resolveActiveViews(viewsNamed(unknown)), std::invalid_argument);
        EXPECT_THROW(
            policy.resolveActiveViews(viewsNamed({c.required.front(), c.required.front()})), std::invalid_argument);
        // Dropping a required slot is an error, not a shorter prefix.
        EXPECT_THROW(policy.resolveActiveViews(
                         viewsNamed(c.optional.empty() ? std::vector<std::string>{c.required.back()} : c.optional)),
            std::invalid_argument);
    }
}

//! Each row is a manifest a hand-edit or a stale export could produce, and the substring
//! the refusal must name. Without these, deleting a guard in loadContract breaks nothing.
TEST_F(Pi05PolicyTest, MalformedContractsAreRefused)
{
    struct RejectionCase
    {
        char const* what;
        char const* policyConfig;
        char const* adapter;
        bool discreteStateInput;
        int32_t numBins;
        char const* expected;
    };

    constexpr RejectionCase kCases[]{
        {"unknown configuration", "pi05_bimanual", "aloha", true, 256, "does not implement"},
        {"adapter from another embodiment", "pi05_droid", "libero", true, 256, "openpi defines it as adapter"},
        {"prompt form flipped", "pi05_droid", "droid", false, 256, "discrete_state_input"},
        {"state bins openpi never uses", "pi05_droid", "droid", true, 128, "256"},
    };

    for (RejectionCase const& c : kCases)
    {
        SCOPED_TRACE(c.what);
        std::ostringstream manifest;
        manifest << R"JSON({
  "contract_version": 1,
  "model_family": "pi05",
  "policy_config": ")JSON"
                 << c.policyConfig << R"JSON(",
  "adapter": ")JSON"
                 << c.adapter << R"JSON(",
  "discrete_state_input": )JSON"
                 << (c.discreteStateInput ? "true" : "false") << R"JSON(,
  "export_id": ")JSON"
                 << kExportId << R"JSON(",
  "state": {"dim": 8, "num_bins": )JSON"
                 << c.numBins << R"JSON(},
  "action": {"dim": 8, "max_dim": 32, "horizon": 15},
  "cameras": {"slots": [{"name": "observation/exterior_image_1_left", "required": true}], "ignored": []},
  "image_resolution": [224, 224],
  "tokenizer": {"max_length": 200, "vocab_size": 257152, "add_bos": true, "add_eos": false}
})JSON";
        writeManifest(kEvenNormStats, manifest.str());
        try
        {
            pi05::Pi05Policy const policy(mDir.string());
            ADD_FAILURE() << "accepted a manifest with " << c.what;
        }
        catch (std::exception const& error)
        {
            EXPECT_NE(std::string(error.what()).find(c.expected), std::string::npos)
                << "refused for the wrong reason: " << error.what();
        }
    }
}

TEST_F(Pi05PolicyTest, QuantileActionsUnnormalizeToRobotUnits)
{
    pi05::Pi05Policy const policy = makePolicy(contractCases().front());
    // One timestep of the 32-wide chunk the engine emits; only the first 7 are the robot's.
    std::vector<float> normalized(32, 7.0F);
    std::vector<float> const row{1.0F, 0.5F, 0.0F, -0.5F, -1.0F, 0.25F, -0.25F};
    std::copy(row.begin(), row.end(), normalized.begin());

    std::vector<float> const robot = policy.unnormalizeActions(normalized, 1, 32);

    ASSERT_EQ(robot.size(), 7U);
    // openpi widens the span by kQuantileEpsilon, so +1 overshoots q99 by half of it and
    // -1 lands on q01; dims 0 and 4 pin the two ends of the map.
    std::vector<float> const expected{
        0.937125921F, 0.44559449F, -0.000187039375F, -0.0517954975F, -0.169429719F, 0.121778399F, -0.250149667F};
    for (size_t d = 0; d < expected.size(); ++d)
    {
        EXPECT_NEAR(robot[d], expected[d], 1e-6F) << "dim " << d;
    }
}

//! Expected values come from running openpi's own transforms over the statistics and chunk
//! below, never from recomputing the same formula here.
TEST_F(Pi05PolicyTest, AlohaActionsMatchTheOpenpiTransforms)
{
    ContractCase const& aloha = contractCases().back();
    pi05::Pi05Policy const policy = makePolicy(aloha);

    constexpr int32_t kHorizon = 3;
    std::vector<float> const normalized{0.1746F, 0.1793F, 0.4264F, -0.2104F, -0.1399F, 0.2955F, 0.1193F, 0.6078F,
        -0.1864F, 0.9012F, 0.0838F, -0.2757F, -0.7444F, -0.4592F, -0.6763F, 0.9433F, -0.3514F, -0.8137F, 0.0808F,
        0.4075F, -0.0518F, 0.602F, -0.303F, 0.2948F, -0.9682F, -0.3018F, -0.5866F, -0.561F, 0.9893F, 0.7552F, 0.827F,
        0.5655F, -0.7243F, 0.5868F, -0.0004F, 0.8709F, 0.7334F, -0.3427F, 0.3211F, 0.379F, -0.2349F, 0.6338F, -0.3802F,
        -0.1459F, -0.3889F, -0.9986F, 0.1892F, 0.9434F, 0.4434F, 0.6327F, -0.7686F, -0.5513F, 0.0989F, -0.085F,
        -0.8136F, -0.2679F, 0.7925F, -0.8913F, -0.5613F, -0.4805F, -0.3419F, 0.6683F, 0.8158F, 0.82F, -0.3897F, 0.9698F,
        0.0703F, -0.45F, 0.5929F, -0.4811F, 0.8667F, -0.122F, -0.6316F, -0.0043F, -0.7559F, 0.6074F, 0.9549F, 0.687F,
        0.9331F, 0.8844F, -0.5681F, -0.9065F, -0.0376F, 0.0151F, -0.9227F, -0.8244F, -0.2397F, 0.6773F, -0.689F,
        -0.9656F, 0.3148F, -0.1093F, -0.8327F, -0.9634F, 0.318F, 0.8449F};

    // openpi's _decode_state: the joint flips, then both grippers out of the Aloha
    // runtime's linear space.
    std::vector<float> const adapted = policy.adaptInputState(aloha.state);
    std::vector<float> const expectedState{0.998F, 0.0701F, 0.7932F, -0.4675F, 0.5225F, -0.7262F, -0.355548725F,
        -0.5043F, 0.8981F, -0.135F, -0.9007F, -0.1727F, -0.4471F, 0.176387981F};
    ASSERT_EQ(adapted.size(), expectedState.size());
    for (size_t d = 0; d < expectedState.size(); ++d)
    {
        EXPECT_NEAR(adapted[d], expectedState[d], 1e-6F) << "state dim " << d;
    }

    std::vector<float> const robot = policy.postprocessActions(normalized, kHorizon, 32, adapted);
    std::vector<float> const expected{0.678950591F, 0.163984109F, -0.827783998F, -0.835419205F, 0.288520971F,
        -0.545035969F, 0.613422047F, 0.0833006292F, -0.94871861F, -0.848982468F, -0.476923156F, 0.0473632498F,
        -0.520503952F, 0.663513835F, 0.00477511768F, -0.143209958F, -0.50439742F, -0.0120087334F, 0.956945217F,
        -1.03597165F, 0.687280721F, -0.0944542042F, -0.910852209F, -0.63918032F, -0.842787389F, 0.15020386F,
        -0.237472439F, 0.45922454F, 0.255725309F, -0.431934711F, -0.557966844F, -1.01787472F, 0.84940644F, -1.14243591F,
        0.886969985F, -0.483681362F, -0.601128467F, -0.138526718F, -1.13902705F, 0.747043824F, 0.832394633F,
        1.0976191F};
    ASSERT_EQ(robot.size(), expected.size());
    for (size_t i = 0; i < expected.size(); ++i)
    {
        EXPECT_NEAR(robot[i], expected[i], 1e-5F) << "element " << i;
    }

    // The absolute step is what ties the chunk to the request: a different state has to
    // move the joints and leave both grippers where they were.
    std::vector<float> moved = adapted;
    moved[0] += 0.25F;
    moved[6] += 0.25F;
    std::vector<float> const shifted = policy.postprocessActions(normalized, kHorizon, 32, moved);
    EXPECT_NEAR(shifted[0], robot[0] + 0.25F, 1e-5F);
    EXPECT_NEAR(shifted[6], robot[6], 1e-5F);
}

//! 8x8 -> 7x7 is LIBERO's 256 -> 224 ratio. Only a real downscale widens the filter
//! support, which is where a fixed two-tap bilinear diverges from the reference.
TEST_F(Pi05PolicyTest, DownscaleAtTheLiberoRatioMatchesTheReference)
{
    std::vector<uint8_t> const rgb{78, 204, 64, 34, 16, 250, 233, 32, 103, 122, 13, 204, 138, 172, 208, 127, 124, 100,
        12, 151, 222, 203, 251, 153, 211, 235, 80, 182, 176, 41, 88, 7, 141, 241, 73, 124, 223, 123, 222, 37, 84, 55,
        200, 102, 211, 60, 159, 237, 53, 138, 57, 140, 55, 21, 7, 18, 118, 67, 241, 138, 148, 221, 56, 33, 89, 208, 34,
        193, 76, 167, 197, 242, 177, 243, 188, 250, 92, 195, 51, 159, 144, 36, 75, 222, 86, 85, 118, 94, 59, 57, 79, 37,
        219, 116, 234, 130, 213, 138, 152, 113, 99, 19, 176, 169, 19, 47, 154, 254, 119, 163, 122, 70, 255, 89, 16, 219,
        40, 224, 81, 35, 126, 236, 5, 89, 126, 246, 188, 201, 215, 100, 100, 63, 81, 192, 156, 171, 194, 167, 128, 117,
        191, 124, 43, 131, 2, 183, 247, 240, 153, 163, 21, 209, 158, 222, 211, 214, 98, 50, 144, 140, 55, 227, 117, 251,
        0, 40, 29, 251, 223, 53, 151, 34, 104, 187, 90, 52, 42, 26, 241, 78, 231, 69, 193, 141, 27, 163, 232, 210, 0,
        212, 206, 123};
    std::vector<float> planar(3U * 7U * 7U);
    pi05::resizeWithPad(rgb.data(), 8, 8, 7, 7, planar.data());

    // openpi_client.image_tools.resize_with_pad on the same pixels.
    std::vector<uint8_t> const golden{94, 100, 183, 147, 127, 63, 156, 168, 141, 115, 198, 100, 121, 100, 115, 139, 36,
        90, 83, 53, 131, 193, 163, 86, 72, 90, 57, 150, 150, 121, 143, 88, 118, 98, 150, 64, 162, 114, 141, 127, 128,
        73, 166, 72, 71, 134, 167, 193, 188, 182, 38, 26, 94, 136, 136, 220, 197, 106, 53, 135, 110, 116, 162, 161, 71,
        113, 172, 112, 117, 198, 175, 115, 145, 120, 144, 136, 153, 184, 136, 162, 151, 203, 173, 74, 189, 163, 197,
        135, 92, 181, 63, 81, 103, 139, 151, 45, 165, 183, 91, 186, 141, 200, 138, 185, 176, 67, 63, 132, 152, 128, 153,
        228, 105, 100, 145, 125, 123, 137, 187, 160, 100, 142, 179, 92, 114, 91, 78, 152, 138, 140, 139, 133, 100, 181,
        183, 173, 160, 186, 97, 190, 170, 150, 77, 143, 178, 48, 120};
    expectMatchesGolden(planar, golden);
}

//! 5x3 into a 4x4 frame: the ratio leaves a 4x2 content box with one black row above and
//! below, so one golden pins the downsample, the centring and the [0, 1] -> [-1, 1] shift.
TEST_F(Pi05PolicyTest, PreprocessedViewMatchesTheReferenceResizeWithPad)
{
    std::vector<uint8_t> const rgb{0, 32, 64, 16, 48, 80, 32, 64, 96, 48, 80, 112, 64, 96, 128, 128, 160, 192, 144, 176,
        208, 160, 192, 224, 176, 208, 240, 192, 224, 255, 255, 224, 192, 240, 208, 176, 224, 192, 160, 208, 176, 144,
        192, 160, 128};
    std::vector<float> planar(3U * 4U * 4U);
    pi05::resizeWithPad(rgb.data(), 3, 5, 4, 4, planar.data());

    std::vector<uint8_t> const golden{0, 0, 0, 0, 52, 71, 89, 108, 206, 202, 198, 193, 0, 0, 0, 0, 0, 0, 0, 0, 84, 103,
        121, 140, 199, 194, 190, 185, 0, 0, 0, 0, 0, 0, 0, 0, 116, 135, 153, 172, 191, 186, 182, 177, 0, 0, 0, 0};
    expectMatchesGolden(planar, golden);
}
