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

// Export-identity checks only: they run before any ONNX is parsed, so the staged
// config.json files below need no model beside them.

#include "builder/pi05Builder.h"

#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <string>
#include <vector>

using namespace trt_edgellm;

namespace
{

constexpr char const* kExportA = "1111111111111111aaaaaaaaaaaaaaaa";
constexpr char const* kExportB = "2222222222222222bbbbbbbbbbbbbbbb";

std::vector<std::string> const kAllComponents{"visual", "prefix", "action"};

class Pi05BuilderTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        mRoot = std::filesystem::temp_directory_path() / "edgellm_pi05_builder_test";
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

    //! One component's config.json, holding only what the identity check reads.
    static void writeComponent(
        std::filesystem::path const& root, std::string const& component, std::string const& exportId)
    {
        std::filesystem::create_directories(root / component);
        std::ofstream(root / component / "config.json")
            << R"({"component": ")" << component << R"(", "export_id": ")" << exportId << R"("})";
    }

    //! An action config carrying the hoist flag where the exporter writes it: top level.
    static void writeActionConfig(std::filesystem::path const& root, std::string const& exportId, bool hoisted)
    {
        std::filesystem::create_directories(root / "action");
        std::ofstream(root / "action" / "config.json")
            << R"({"component": "action", "export_id": ")" << exportId << R"(", "hoisted_adarms_cond": )"
            << (hoisted ? "true" : "false") << "}";
    }

    static void writeContract(std::filesystem::path const& root, std::string const& exportId)
    {
        std::ofstream(root / "policy.json") << R"({"export_id": ")" << exportId << R"("})";
    }

    //! A whole export or a whole engine root, every component stamped alike.
    static void writeExport(std::filesystem::path const& root, std::string const& exportId)
    {
        writeContract(root, exportId);
        for (std::string const& component : kAllComponents)
        {
            writeComponent(root, component, exportId);
        }
    }

    std::filesystem::path mRoot;
    std::filesystem::path mOnnx;
    std::filesystem::path mEngines;
};

} // namespace

TEST_F(Pi05BuilderTest, OneExportIntoAMatchingEngineRootIsAccepted)
{
    writeExport(mOnnx, kExportA);
    writeExport(mEngines, kExportA);
    // Staged sidecars carry no config.json and are not components.
    std::filesystem::create_directories(mEngines / "assets");
    std::ofstream(mEngines / "assets" / "norm_stats.json") << "{}";

    EXPECT_TRUE(pi05::verifyPi05ExportIds(mOnnx, mEngines, {"action"}));
}

TEST_F(Pi05BuilderTest, AStaleSidecarSurvivingAFullRebuildIsRejected)
{
    // assets/ has no config.json, so nothing marks it as export A's and rebuilding
    // every component from a B that ships none of its own leaves it in place; the
    // runtime would then unnormalize B's actions with A's mean/std.
    writeExport(mEngines, kExportA);
    std::filesystem::create_directories(mEngines / "assets");
    std::ofstream(mEngines / "assets" / "norm_stats.json") << R"({"norm_stats": "export A"})";
    writeExport(mOnnx, kExportB);
    ASSERT_FALSE(std::filesystem::exists(mOnnx / "assets"));

    EXPECT_FALSE(pi05::verifyPi05ExportIds(mOnnx, mEngines, kAllComponents));
}

TEST_F(Pi05BuilderTest, HoistedExportRequiresCond)
{
    writeExport(mOnnx, kExportA);
    writeActionConfig(mOnnx, kExportA, /*hoisted=*/true);
    writeComponent(mOnnx, "cond", kExportA);

    EXPECT_EQ(pi05::policyComponents(mOnnx), (std::vector<std::string>{"visual", "prefix", "action", "cond"}));
}

TEST_F(Pi05BuilderTest, UnhoistedExportHasNoCond)
{
    writeExport(mOnnx, kExportA);
    writeActionConfig(mOnnx, kExportA, /*hoisted=*/false);

    EXPECT_EQ(pi05::policyComponents(mOnnx), kAllComponents);
}

TEST_F(Pi05BuilderTest, HoistedExportMissingCondFailsTheBuild)
{
    // The directory is gone but the contract still claims it: building the other three
    // and reporting success would leave a bundle the runtime cannot load.
    writeExport(mOnnx, kExportA);
    writeActionConfig(mOnnx, kExportA, /*hoisted=*/true);
    ASSERT_FALSE(std::filesystem::exists(mOnnx / "cond"));

    std::vector<std::string> const components = pi05::policyComponents(mOnnx);
    EXPECT_NE(std::find(components.begin(), components.end(), "cond"), components.end());
    EXPECT_FALSE(pi05::buildPi05Policy(mOnnx, mEngines, components, /*maxBatchSize=*/1));
}
