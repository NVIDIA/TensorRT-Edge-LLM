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

// Bundle-identity checks only: they run before any ONNX is parsed, so the staged
// config.json files below need no model beside them.

#include "builder/gr00tBuilder.h"

#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <string>

using namespace trt_edgellm;

namespace
{

constexpr char const* kExportA = "1111111111111111aaaaaaaaaaaaaaaa";
constexpr char const* kExportB = "2222222222222222bbbbbbbbbbbbbbbb";

class Gr00tBuilderTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        mRoot = std::filesystem::temp_directory_path() / "edgellm_gr00t_builder_test";
        std::filesystem::remove_all(mRoot);
        mOnnx = mRoot / "onnx";
        mEngines = mRoot / "engines";
        std::filesystem::create_directories(mOnnx);
        std::filesystem::create_directories(mEngines);
    }

    void TearDown() override
    {
        std::filesystem::remove_all(mRoot);
    }

    //! One component's identity-bearing files, holding only what the check reads.
    static void writeComponent(std::filesystem::path const& root, std::string const& component,
        std::string const& exportId, std::string const& frontend = "onnx")
    {
        std::filesystem::create_directories(root / component);
        std::string const body = R"({"export_id": ")" + exportId + R"(", "frontend": ")" + frontend
            + R"(", "component": ")" + component + R"("})";
        std::ofstream(root / component / "config.json") << body;
        if (component == "action")
        {
            std::ofstream(root / "action" / "policy.json") << body;
        }
    }

    static void writeBundle(std::filesystem::path const& root, std::string const& exportId, std::string const& frontend)
    {
        for (std::string const& component : gr00t::policyComponents())
        {
            writeComponent(root, component, exportId, frontend);
        }
    }

    std::filesystem::path mRoot;
    std::filesystem::path mOnnx;
    std::filesystem::path mEngines;
};

} // namespace

//! Rebuilding one component beside the same export passes; a component from another export,
//! from the direct route on either side, or without an identity is refused before anything is built.
TEST_F(Gr00tBuilderTest, ComponentsFromAnotherBuildAreRefused)
{
    writeBundle(mOnnx, kExportA, "onnx");
    writeBundle(mEngines, kExportA, "onnx");
    EXPECT_TRUE(gr00t::verifyGr00tExportIds(mOnnx, mEngines, {"action"}));
    EXPECT_TRUE(gr00t::verifyGr00tExportIds(mOnnx, mEngines, gr00t::policyComponents()));

    writeComponent(mOnnx, "action", kExportB);
    EXPECT_FALSE(gr00t::verifyGr00tExportIds(mOnnx, mEngines, gr00t::policyComponents()));
    EXPECT_FALSE(gr00t::verifyGr00tExportIds(mOnnx, mEngines, {"action"}));

    writeBundle(mOnnx, kExportA, "onnx");
    writeComponent(mEngines, "visual", kExportA, "direct");
    EXPECT_FALSE(gr00t::verifyGr00tExportIds(mOnnx, mEngines, {"action"}));

    std::filesystem::remove_all(mEngines);
    std::filesystem::create_directories(mEngines);
    writeComponent(mOnnx, "action", kExportA, "direct");
    EXPECT_FALSE(gr00t::verifyGr00tExportIds(mOnnx, mEngines, {"action"}));
    writeBundle(mOnnx, kExportA, "onnx");

    writeBundle(mEngines, kExportA, "onnx");
    std::ofstream(mEngines / "visual" / "config.json") << R"({"component": "visual"})";
    EXPECT_FALSE(gr00t::verifyGr00tExportIds(mOnnx, mEngines, {"action"}));
    EXPECT_TRUE(gr00t::verifyGr00tExportIds(mOnnx, mEngines, gr00t::policyComponents()));
}
