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
#include "testUtils.h"
#include <NvInferRuntime.h>
#include <gtest/gtest.h>
#include <regex>
#include <sstream>

using namespace trt_edgellm;
using namespace trt_edgellm::logger;

class LoggerTest : public ::testing::Test
{
protected:
    void SetUp() override
    {

        gLogger.setLevel(nvinfer1::ILogger::Severity::kVERBOSE);
        gLogger.setShowTimestamp(true);
        gLogger.setShowLocation(true);
        gLogger.setShowFunction(true);

        // Capture output streams
        originalCoutBuffer = std::cout.rdbuf();
        originalCerrBuffer = std::cerr.rdbuf();

        // Redirect to our stringstreams
        std::cout.rdbuf(coutCapture.rdbuf());
        std::cerr.rdbuf(cerrCapture.rdbuf());
    }

    void TearDown() override
    {
        // Restore original streams
        std::cout.rdbuf(originalCoutBuffer);
        std::cerr.rdbuf(originalCerrBuffer);

        // Clear capture buffers
        coutCapture.str("");
        coutCapture.clear();
        cerrCapture.str("");
        cerrCapture.clear();
    }

    std::string getCoutOutput()
    {
        std::cout.flush();
        return coutCapture.str();
    }

    std::string getCerrOutput()
    {
        std::cerr.flush();
        return cerrCapture.str();
    }

    void clearOutput()
    {
        coutCapture.str("");
        coutCapture.clear();
        cerrCapture.str("");
        cerrCapture.clear();
    }

private:
    std::streambuf* originalCoutBuffer;
    std::streambuf* originalCerrBuffer;
    std::ostringstream coutCapture;
    std::ostringstream cerrCapture;
};

TEST_F(LoggerTest, BasicLoggingMacros)
{
    // Test all basic logging macros
    LOG_DEBUG("Debug message: %d", 42);
    LOG_INFO("Info message: %s", "test");
    LOG_WARNING("Warning message: %f", 3.14);
    LOG_ERROR("Error message: %c", 'X');

    std::string coutOutput = getCoutOutput();
    std::string cerrOutput = getCerrOutput();

    // Debug and Info should go to cout
    EXPECT_TRUE(coutOutput.find("[DEBUG]") != std::string::npos);
    EXPECT_TRUE(coutOutput.find("Debug message: 42") != std::string::npos);
    EXPECT_TRUE(coutOutput.find("[INFO]") != std::string::npos);
    EXPECT_TRUE(coutOutput.find("Info message: test") != std::string::npos);

    // Warning and Error should go to cerr
    EXPECT_TRUE(cerrOutput.find("[WARNING]") != std::string::npos);
    EXPECT_TRUE(cerrOutput.find("Warning message: 3.14") != std::string::npos);
    EXPECT_TRUE(cerrOutput.find("[ERROR]") != std::string::npos);
    EXPECT_TRUE(cerrOutput.find("Error message: X") != std::string::npos);
}

TEST_F(LoggerTest, ConditionalLoggingMacros)
{
    bool trueCondition = true;
    bool falseCondition = false;

    LOG_DEBUG_IF(trueCondition, "Debug should appear: %d", 1);
    LOG_DEBUG_IF(falseCondition, "Debug should not appear: %d", 2);

    LOG_INFO_IF(trueCondition, "Info should appear: %d", 3);
    LOG_INFO_IF(falseCondition, "Info should not appear: %d", 4);

    LOG_WARNING_IF(trueCondition, "Warning should appear: %d", 5);
    LOG_WARNING_IF(falseCondition, "Warning should not appear: %d", 6);

    LOG_ERROR_IF(trueCondition, "Error should appear: %d", 7);
    LOG_ERROR_IF(falseCondition, "Error should not appear: %d", 8);

    std::string coutOutput = getCoutOutput();
    std::string cerrOutput = getCerrOutput();

    // Should appear
    EXPECT_TRUE(coutOutput.find("Debug should appear: 1") != std::string::npos);
    EXPECT_TRUE(coutOutput.find("Info should appear: 3") != std::string::npos);
    EXPECT_TRUE(cerrOutput.find("Warning should appear: 5") != std::string::npos);
    EXPECT_TRUE(cerrOutput.find("Error should appear: 7") != std::string::npos);

    // Should not appear
    EXPECT_TRUE(coutOutput.find("should not appear") == std::string::npos);
    EXPECT_TRUE(cerrOutput.find("should not appear") == std::string::npos);
}

TEST_F(LoggerTest, LogLevelFiltering)
{
    // Test VERBOSE level (should show all)
    gLogger.setLevel(nvinfer1::ILogger::Severity::kVERBOSE);
    clearOutput();

    LOG_DEBUG("Debug at VERBOSE level");
    LOG_INFO("Info at VERBOSE level");
    LOG_WARNING("Warn at VERBOSE level");
    LOG_ERROR("Error at VERBOSE level");

    std::string coutOutput = getCoutOutput();
    std::string cerrOutput = getCerrOutput();

    EXPECT_TRUE(coutOutput.find("Debug at VERBOSE level") != std::string::npos);
    EXPECT_TRUE(coutOutput.find("Info at VERBOSE level") != std::string::npos);
    EXPECT_TRUE(cerrOutput.find("Warn at VERBOSE level") != std::string::npos);
    EXPECT_TRUE(cerrOutput.find("Error at VERBOSE level") != std::string::npos);

    // Test INFO level (should hide DEBUG)
    gLogger.setLevel(nvinfer1::ILogger::Severity::kINFO);
    clearOutput();

    LOG_DEBUG("Debug at INFO level");
    LOG_INFO("Info at INFO level");
    LOG_WARNING("Warn at INFO level");
    LOG_ERROR("Error at INFO level");

    coutOutput = getCoutOutput();
    cerrOutput = getCerrOutput();

    EXPECT_TRUE(coutOutput.find("Debug at INFO level") == std::string::npos);
    EXPECT_TRUE(coutOutput.find("Info at INFO level") != std::string::npos);
    EXPECT_TRUE(cerrOutput.find("Warn at INFO level") != std::string::npos);
    EXPECT_TRUE(cerrOutput.find("Error at INFO level") != std::string::npos);

    // Test WARNING level (should hide DEBUG and INFO)
    gLogger.setLevel(nvinfer1::ILogger::Severity::kWARNING);
    clearOutput();

    LOG_DEBUG("Debug at WARNING level");
    LOG_INFO("Info at WARNING level");
    LOG_WARNING("Warn at WARNING level");
    LOG_ERROR("Error at WARNING level");

    coutOutput = getCoutOutput();
    cerrOutput = getCerrOutput();

    EXPECT_TRUE(coutOutput.find("Debug at WARNING level") == std::string::npos);
    EXPECT_TRUE(coutOutput.find("Info at WARNING level") == std::string::npos);
    EXPECT_TRUE(cerrOutput.find("Warn at WARNING level") != std::string::npos);
    EXPECT_TRUE(cerrOutput.find("Error at WARNING level") != std::string::npos);

    // Test ERROR level (should only show ERROR)
    gLogger.setLevel(nvinfer1::ILogger::Severity::kERROR);
    clearOutput();

    LOG_DEBUG("Debug at ERROR level");
    LOG_INFO("Info at ERROR level");
    LOG_WARNING("Warn at ERROR level");
    LOG_ERROR("Error at ERROR level");

    coutOutput = getCoutOutput();
    cerrOutput = getCerrOutput();

    EXPECT_TRUE(coutOutput.find("Debug at ERROR level") == std::string::npos);
    EXPECT_TRUE(coutOutput.find("Info at ERROR level") == std::string::npos);
    EXPECT_TRUE(cerrOutput.find("Warn at ERROR level") == std::string::npos);
    EXPECT_TRUE(cerrOutput.find("Error at ERROR level") != std::string::npos);
}

TEST_F(LoggerTest, TensorRTSeverityLevels)
{
    // Test direct nvinfer1::ILogger::Severity usage
    gLogger.setLevel(nvinfer1::ILogger::Severity::kVERBOSE);
    EXPECT_EQ(gLogger.getLevel(), nvinfer1::ILogger::Severity::kVERBOSE);

    gLogger.setLevel(nvinfer1::ILogger::Severity::kINFO);
    EXPECT_EQ(gLogger.getLevel(), nvinfer1::ILogger::Severity::kINFO);

    gLogger.setLevel(nvinfer1::ILogger::Severity::kWARNING);
    EXPECT_EQ(gLogger.getLevel(), nvinfer1::ILogger::Severity::kWARNING);

    gLogger.setLevel(nvinfer1::ILogger::Severity::kERROR);
    EXPECT_EQ(gLogger.getLevel(), nvinfer1::ILogger::Severity::kERROR);
}

TEST_F(LoggerTest, TensorRTLoggerIntegration)
{
    // Test TensorRT ILogger interface
    nvinfer1::ILogger* tensorrtLogger = &gLogger;

    clearOutput();
    tensorrtLogger->log(nvinfer1::ILogger::Severity::kINFO, "TensorRT info message");
    tensorrtLogger->log(nvinfer1::ILogger::Severity::kWARNING, "TensorRT warning message");
    tensorrtLogger->log(nvinfer1::ILogger::Severity::kERROR, "TensorRT error message");
    tensorrtLogger->log(nvinfer1::ILogger::Severity::kVERBOSE, "TensorRT debug message");

    std::string coutOutput = getCoutOutput();
    std::string cerrOutput = getCerrOutput();

    // Check that TensorRT messages are properly tagged
    EXPECT_TRUE(coutOutput.find("[TensorRT]") != std::string::npos);
    EXPECT_TRUE(coutOutput.find("TensorRT info message") != std::string::npos);
    EXPECT_TRUE(coutOutput.find("TensorRT debug message") != std::string::npos);

    EXPECT_TRUE(cerrOutput.find("[TensorRT]") != std::string::npos);
    EXPECT_TRUE(cerrOutput.find("TensorRT warning message") != std::string::npos);
    EXPECT_TRUE(cerrOutput.find("TensorRT error message") != std::string::npos);
}

TEST_F(LoggerTest, LocationTracking)
{
    clearOutput();
    LOG_INFO("Test location tracking");

    std::string output = getCoutOutput();

    // Should contain file name
    EXPECT_TRUE(output.find("loggerTests.cpp") != std::string::npos);

    // Should contain function name
    EXPECT_TRUE(output.find("TestBody") != std::string::npos);

    // Should contain line number (approximate check)
    EXPECT_TRUE(output.find(":") != std::string::npos);
}

TEST_F(LoggerTest, ConfigurableFormatting)
{
    // Test with all formatting enabled
    gLogger.setShowTimestamp(true);
    gLogger.setShowLocation(true);
    gLogger.setShowFunction(true);

    clearOutput();
    LOG_INFO("Full formatting test");
    std::string fullOutput = getCoutOutput();

    // Should contain timestamp pattern [HH:MM:SS.mmm]
    EXPECT_TRUE(std::regex_search(fullOutput, std::regex(R"(\[\d{2}:\d{2}:\d{2}\.\d{3}\])")));

    // Should contain location info
    EXPECT_TRUE(fullOutput.find("loggerTests.cpp") != std::string::npos);
    EXPECT_TRUE(fullOutput.find("TestBody") != std::string::npos);

    // Test with timestamp disabled
    gLogger.setShowTimestamp(false);
    clearOutput();
    LOG_INFO("No timestamp test");
    std::string noTimestampOutput = getCoutOutput();

    EXPECT_FALSE(std::regex_search(noTimestampOutput, std::regex(R"(\[\d{2}:\d{2}:\d{2}\.\d{3}\])")));
    EXPECT_TRUE(noTimestampOutput.find("loggerTests.cpp") != std::string::npos);

    // Test with location disabled
    gLogger.setShowTimestamp(true);
    gLogger.setShowLocation(false);
    clearOutput();
    LOG_INFO("No location test");
    std::string noLocationOutput = getCoutOutput();

    EXPECT_TRUE(std::regex_search(noLocationOutput, std::regex(R"(\[\d{2}:\d{2}:\d{2}\.\d{3}\])")));
    EXPECT_TRUE(noLocationOutput.find("loggerTests.cpp") == std::string::npos);

    // Test with function names disabled
    gLogger.setShowLocation(true);
    gLogger.setShowFunction(false);
    clearOutput();
    LOG_INFO("No function test");
    std::string noFunctionOutput = getCoutOutput();

    EXPECT_TRUE(noFunctionOutput.find("loggerTests.cpp") != std::string::npos);
    // Function name should not appear in location part
    EXPECT_TRUE(noFunctionOutput.find(":TestBody") == std::string::npos);
}

TEST_F(LoggerTest, FunctionTracing)
{
    clearOutput();

    {
        LOG_TRACE_FUNCTION();
        LOG_INFO("Inside traced function");
    }

    std::string output = getCoutOutput();

    // Should contain entry message
    EXPECT_TRUE(output.find("-> Entering") != std::string::npos);
    EXPECT_TRUE(output.find("TestBody") != std::string::npos);

    // Should contain the regular log message
    EXPECT_TRUE(output.find("Inside traced function") != std::string::npos);

    // Should contain exit message
    EXPECT_TRUE(output.find("<- Exiting") != std::string::npos);
}

TEST_F(LoggerTest, DirectConfiguration)
{
    // Test direct logger configuration
    gLogger.setLevel(nvinfer1::ILogger::Severity::kWARNING);
    EXPECT_EQ(gLogger.getLevel(), nvinfer1::ILogger::Severity::kWARNING);

    // Test that INFO messages don't appear at WARNING level
    LOG_INFO("This should not appear");

    std::string output = getCoutOutput();
    // At WARNING level, INFO messages should not appear
    EXPECT_TRUE(output.empty());
}

TEST_F(LoggerTest, FormattingEdgeCases)
{
    gLogger.setShowTimestamp(false);
    gLogger.setShowLocation(false);
    clearOutput();

    // Test empty message
    LOG_INFO("");

    // Test message with special characters
    LOG_INFO("Special chars: !@#$%^&*()");

    // Test message with newlines
    LOG_INFO("Line 1\nLine 2");

    // Test very long message
    std::string longMessage(1000, 'A');
    LOG_INFO("Long message: %s", longMessage.c_str());

    // Test multiple format specifiers
    LOG_INFO("Multiple: %d %s %.2f %c", 42, "test", 3.14, 'X');

    std::string output = getCoutOutput();

    // Should contain all test cases
    EXPECT_TRUE(output.find("Special chars: !@#$%^&*()") != std::string::npos);
    EXPECT_TRUE(output.find("Line 1\nLine 2") != std::string::npos);
    EXPECT_TRUE(output.find(longMessage) != std::string::npos);
    EXPECT_TRUE(output.find("Multiple: 42 test 3.14 X") != std::string::npos);
}

TEST_F(LoggerTest, SourceLocationFunctionality)
{
    // Test SourceLocation structure directly
    SourceLocation loc(__FILE__, __FUNCTION__, __LINE__);

    EXPECT_STREQ(loc.file, __FILE__);
    EXPECT_STREQ(loc.function, __FUNCTION__);
    EXPECT_GT(loc.lineNumber, 0);

    // Test filename extraction using filesystem::path directly
    std::filesystem::path p(loc.file);
    std::string filename = p.filename().string();
    EXPECT_TRUE(filename.find("loggerTests.cpp") != std::string::npos);
}

// Helper function to test scoped function tracer
void tracedFunction()
{
    LOG_TRACE_FUNCTION();
    LOG_INFO("Inside traced helper function");
}

TEST_F(LoggerTest, ScopedFunctionTracerHelper)
{
    clearOutput();
    tracedFunction();

    std::string output = getCoutOutput();

    // Should contain entry for the helper function
    EXPECT_TRUE(output.find("-> Entering tracedFunction") != std::string::npos);
    EXPECT_TRUE(output.find("Inside traced helper function") != std::string::npos);
    EXPECT_TRUE(output.find("<- Exiting tracedFunction") != std::string::npos);
}

TEST_F(LoggerTest, TensorRTSeverityValues)
{
    // Test that TensorRT severity values are correctly ordered (lower is more severe)
    EXPECT_LT(
        static_cast<int>(nvinfer1::ILogger::Severity::kERROR), static_cast<int>(nvinfer1::ILogger::Severity::kWARNING));
    EXPECT_LT(
        static_cast<int>(nvinfer1::ILogger::Severity::kWARNING), static_cast<int>(nvinfer1::ILogger::Severity::kINFO));
    EXPECT_LT(
        static_cast<int>(nvinfer1::ILogger::Severity::kINFO), static_cast<int>(nvinfer1::ILogger::Severity::kVERBOSE));

    // Verify the actual numeric values match TensorRT specification
    EXPECT_EQ(static_cast<int>(nvinfer1::ILogger::Severity::kERROR), 1);
    EXPECT_EQ(static_cast<int>(nvinfer1::ILogger::Severity::kWARNING), 2);
    EXPECT_EQ(static_cast<int>(nvinfer1::ILogger::Severity::kINFO), 3);
    EXPECT_EQ(static_cast<int>(nvinfer1::ILogger::Severity::kVERBOSE), 4);
}

TEST_F(LoggerTest, DirectLoggerMethods)
{
    // Test direct method calls on logger
    gLogger.setShowTimestamp(false);
    gLogger.setShowLocation(false);
    clearOutput();

    SourceLocation loc("test_file.cpp", "test_function", 123);

    gLogger.debug("Direct debug call", loc);
    gLogger.info("Direct info call", loc);
    gLogger.warning("Direct warning call", loc);
    gLogger.error("Direct error call", loc);

    std::string coutOutput = getCoutOutput();
    std::string cerrOutput = getCerrOutput();

    EXPECT_TRUE(coutOutput.find("Direct debug call") != std::string::npos);
    EXPECT_TRUE(coutOutput.find("Direct info call") != std::string::npos);
    EXPECT_TRUE(cerrOutput.find("Direct warning call") != std::string::npos);
    EXPECT_TRUE(cerrOutput.find("Direct error call") != std::string::npos);
}
