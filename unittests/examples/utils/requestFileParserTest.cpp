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

#include "requestFileParser.h"
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <string>

using namespace trt_edgellm;
using namespace trt_edgellm::exampleUtils;

namespace
{

class RequestFileParserTest : public ::testing::Test
{
protected:
    void TearDown() override
    {
        std::filesystem::remove(mPath);
    }

    //! Writes a one-request file whose `guided_decoding` is @p guidedDecoding, plus @p extraRequests.
    std::filesystem::path writeRequests(std::string const& guidedDecoding, std::string const& extraRequests = "")
    {
        std::string const testName = ::testing::UnitTest::GetInstance()->current_test_info()->name();
        mPath = std::filesystem::temp_directory_path() / ("requestFileParserTest_" + testName + ".json");
        std::ofstream file(mPath);
        file << R"({"requests": [{"messages": [{"role": "user", "content": "hi"}], "guided_decoding": )"
             << guidedDecoding << "}" << extraRequests << "]}";
        return mPath;
    }

    std::filesystem::path mPath;
};

TEST_F(RequestFileParserTest, JumpForwardRidesOnTheSelectedMode)
{
    // Given a request whose guided_decoding sets a mode and jump_forward, and one that sets the mode alone,
    // when parsed, then the first carries that mode with jump-forward on and the second has it off.
    auto const [loraMap, batches] = parseRequestFile(writeRequests(R"({"regex": "[a-z]+", "jump_forward": true})"));
    ASSERT_EQ(batches.size(), 1U);
    auto const& guided = batches.front().requests.front().guidedDecoding;
    ASSERT_TRUE(guided.has_value());
    EXPECT_EQ(guided->type, rt::GuideType::kRegex);
    EXPECT_TRUE(guided->jumpForward);

    auto const [unusedLoraMap, withoutFlag] = parseRequestFile(writeRequests(R"({"regex": "[a-z]+"})"));
    ASSERT_EQ(withoutFlag.size(), 1U);
    EXPECT_FALSE(withoutFlag.front().requests.front().guidedDecoding->jumpForward);
}

TEST_F(RequestFileParserTest, OverrideTurnsJumpForwardOnForGuidedRequestsOnly)
{
    // Given a guided and an unguided request, when parsed with the jump-forward override, then only
    // the guided request gets the flag and the unguided one stays unconstrained.
    auto const path = writeRequests(
        R"({"choice": ["yes", "no"]})", R"(, {"messages": [{"role": "user", "content": "free text"}]})");
    auto const [loraMap, batches] = parseRequestFile(path, /*batchSizeOverride=*/-1,
        /*maxGenerateLengthOverride=*/-1, /*numLogprobsOverride=*/-1, /*jumpForwardOverride=*/true);
    ASSERT_EQ(batches.size(), 2U);
    EXPECT_TRUE(batches[0].requests.front().guidedDecoding->jumpForward);
    EXPECT_FALSE(batches[1].requests.front().guidedDecoding.has_value());
}

} // namespace
