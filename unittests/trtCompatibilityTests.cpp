/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "common/logger.h"
#include <NvInfer.h>
#include <gtest/gtest.h>
#include <memory>

using namespace trt_edgellm;
using namespace trt_edgellm::logger;

class TRTCompatibilityTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        LOG_INFO("Testing TensorRT compatibility with version %d.%d.%d", NV_TENSORRT_MAJOR, NV_TENSORRT_MINOR,
            NV_TENSORRT_PATCH);
    }
};

// Test that TensorRT version macros are accessible and valid
TEST_F(TRTCompatibilityTest, VersionMacrosExist)
{
    // We support TensorRT 10.x and 11.x
    EXPECT_GE(NV_TENSORRT_MAJOR, 10) << "TensorRT version must be at least 10.x";
    EXPECT_LE(NV_TENSORRT_MAJOR, 11) << "TensorRT version must be at most 11.x";
    EXPECT_GE(NV_TENSORRT_MINOR, 0);
    EXPECT_GE(NV_TENSORRT_PATCH, 0);
}

// Test network creation with version-specific flags
TEST_F(TRTCompatibilityTest, NetworkCreationWithVersionFlags)
{
    auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(gLogger));
    ASSERT_NE(builder, nullptr) << "Failed to create TensorRT builder";

    // This mirrors the logic in cpp/builder/builder.cpp
#if NV_TENSORRT_MAJOR >= 11
    auto const networkFlags = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kSTRONGLY_TYPED);
    LOG_INFO("Using kSTRONGLY_TYPED flag (TRT 11.x)");
#else
    auto const networkFlags = 0U; // Default flags for TensorRT 10.x
    LOG_INFO("Using default network flags (TRT 10.x)");
#endif

    auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(networkFlags));
    ASSERT_NE(network, nullptr) << "Failed to create network definition";
}

// Test builder config with version-specific flags
TEST_F(TRTCompatibilityTest, BuilderConfigFlags)
{
    auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(gLogger));
    ASSERT_NE(builder, nullptr) << "Failed to create TensorRT builder";

    auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    ASSERT_NE(config, nullptr) << "Failed to create builder config";

    // This mirrors the logic in cpp/builder/builder.cpp
#if NV_TENSORRT_MAJOR >= 11
    config->setFlag(nvinfer1::BuilderFlag::kMONITOR_MEMORY);
    LOG_INFO("Set kMONITOR_MEMORY flag (TRT 11.x)");
#else
    LOG_INFO("Skipping kMONITOR_MEMORY flag (TRT 10.x)");
#endif

    // Config should be valid regardless of version
    SUCCEED();
}

// Test that we can create a runtime (used by engine runners)
TEST_F(TRTCompatibilityTest, RuntimeCreation)
{
    auto runtime = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(gLogger));
    ASSERT_NE(runtime, nullptr) << "Failed to create TensorRT runtime";
}

// Test version string formatting (useful for diagnostics)
TEST_F(TRTCompatibilityTest, VersionStringFormat)
{
    char versionStr[32];
    snprintf(versionStr, sizeof(versionStr), "%d.%d.%d", NV_TENSORRT_MAJOR, NV_TENSORRT_MINOR, NV_TENSORRT_PATCH);

    std::string version(versionStr);
    EXPECT_FALSE(version.empty());
    EXPECT_NE(version.find('.'), std::string::npos) << "Version string should contain dots";

    LOG_INFO("TensorRT version string: %s", versionStr);
}
