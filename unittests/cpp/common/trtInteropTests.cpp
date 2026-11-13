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

#include "common/logger.h"
#include "common/tensor.h"
#include "common/trtInterop.h"
#include "common/trtUtils.h"

#include <NvInferRuntime.h>
#include <cstdlib>
#include <gtest/gtest.h>
#include <string>

#if !defined(_WIN32)
#include "testPluginLoader.h"
#endif

using namespace trt_edgellm;

namespace
{

class TrtLoggerTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        mSavedLevel = gLogger.getLevel();
    }

    void TearDown() override
    {
        gLogger.setLevel(mSavedLevel);
    }

private:
    logger::Severity mSavedLevel{};
};

#if !defined(_WIN32)
//! Points loadEdgellmPluginLib() at this build's plugin library, which is not next to the test executable.
class PluginLogLevelTest : public TrtLoggerTest
{
protected:
    void SetUp() override
    {
        TrtLoggerTest::SetUp();
        char const* const path = std::getenv("EDGELLM_PLUGIN_PATH");
        mHadPluginPath = path != nullptr;
        mSavedPluginPath = mHadPluginPath ? path : "";
        setenv("EDGELLM_PLUGIN_PATH", test::pluginLibraryPath().c_str(), /*overwrite=*/1);
    }

    void TearDown() override
    {
        if (mHadPluginPath)
        {
            setenv("EDGELLM_PLUGIN_PATH", mSavedPluginPath.c_str(), /*overwrite=*/1);
        }
        else
        {
            unsetenv("EDGELLM_PLUGIN_PATH");
        }
        TrtLoggerTest::TearDown();
    }

private:
    bool mHadPluginPath{false};
    std::string mSavedPluginPath;
};
#endif

} // namespace

// Given TensorRT dims of rank 3, and a tensor of rank 3
// When they cross the TensorRT boundary through toCoords and toTrtDims
// Then the rank and every extent survive in both directions
TEST(TrtInteropTest, DimsConversionKeepsRankAndExtents)
{
    nvinfer1::Dims dims{};
    dims.nbDims = 3;
    dims.d[0] = 7;
    dims.d[1] = 8;
    dims.d[2] = 9;
    rt::Coords const coords = rt::toCoords(dims);
    ASSERT_EQ(coords.volume(), 504);
    ASSERT_EQ(coords.getNumDims(), 3);
    ASSERT_EQ(coords[0], 7);
    ASSERT_EQ(coords[1], 8);
    ASSERT_EQ(coords[2], 9);

    nvinfer1::Dims const roundTrip = rt::toTrtDims(coords);
    ASSERT_EQ(roundTrip.nbDims, 3);
    ASSERT_EQ(roundTrip.d[0], 7);
    ASSERT_EQ(roundTrip.d[1], 8);
    ASSERT_EQ(roundTrip.d[2], 9);

    rt::Tensor const tensor({2, 3, 4}, rt::DeviceType::kCPU, DataType::kFLOAT);
    nvinfer1::Dims const tensorDims = rt::toTrtDims(tensor);
    ASSERT_EQ(tensorDims.nbDims, 3);
    ASSERT_EQ(tensorDims.d[0], 2);
    ASSERT_EQ(tensorDims.d[1], 3);
    ASSERT_EQ(tensorDims.d[2], 4);
}

// Given gLogger at VERBOSE with source locations shown
// When TensorRT logs one message at each severity through TrtLogger::instance()
// Then every message reaches gLogger tagged [TensorRT]: INFO and VERBOSE on stdout, WARNING and ERROR on stderr
TEST_F(TrtLoggerTest, ForwardsTensorRTMessagesToGLogger)
{
    gLogger.setLevel(logger::Severity::kVERBOSE);
    gLogger.setShowLocation(true);
    nvinfer1::ILogger* tensorrtLogger = &TrtLogger::instance();

    testing::internal::CaptureStdout();
    testing::internal::CaptureStderr();
    tensorrtLogger->log(nvinfer1::ILogger::Severity::kINFO, "TensorRT info message");
    tensorrtLogger->log(nvinfer1::ILogger::Severity::kWARNING, "TensorRT warning message");
    tensorrtLogger->log(nvinfer1::ILogger::Severity::kERROR, "TensorRT error message");
    tensorrtLogger->log(nvinfer1::ILogger::Severity::kVERBOSE, "TensorRT debug message");
    std::string const coutOutput = testing::internal::GetCapturedStdout();
    std::string const cerrOutput = testing::internal::GetCapturedStderr();

    EXPECT_TRUE(coutOutput.find("[TensorRT]") != std::string::npos);
    EXPECT_TRUE(coutOutput.find("TensorRT info message") != std::string::npos);
    EXPECT_TRUE(coutOutput.find("TensorRT debug message") != std::string::npos);

    EXPECT_TRUE(cerrOutput.find("[TensorRT]") != std::string::npos);
    EXPECT_TRUE(cerrOutput.find("TensorRT warning message") != std::string::npos);
    EXPECT_TRUE(cerrOutput.find("TensorRT error message") != std::string::npos);
}

#if !defined(_WIN32)
// Given the plugin library, which keeps its own logger
// When the host loads it with gLogger at VERBOSE, and again with gLogger at ERROR
// Then the plugin library prints its debug messages after the first load and not after the second
TEST_F(PluginLogLevelTest, PluginLibraryFollowsHostLogLevel)
{
    // TensorRT keeps the plugin's creators registered, so the library must stay mapped after this test.
    ASSERT_NE(test::loadPluginLibrary(), nullptr);
    std::string const pluginDebugMessage = "Edge-LLM plugin library initialized";

    gLogger.setLevel(logger::Severity::kVERBOSE);
    testing::internal::CaptureStdout();
    auto const verbosePlugin = loadEdgellmPluginLib();
    std::string const verboseOutput = testing::internal::GetCapturedStdout();
    ASSERT_NE(verbosePlugin, nullptr);
    EXPECT_NE(verboseOutput.find(pluginDebugMessage), std::string::npos);

    gLogger.setLevel(logger::Severity::kERROR);
    testing::internal::CaptureStdout();
    auto const quietPlugin = loadEdgellmPluginLib();
    std::string const quietOutput = testing::internal::GetCapturedStdout();
    ASSERT_NE(quietPlugin, nullptr);
    EXPECT_EQ(quietOutput.find(pluginDebugMessage), std::string::npos);
}
#endif
