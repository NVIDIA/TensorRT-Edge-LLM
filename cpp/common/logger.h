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

#pragma once

#include "stringUtils.h"
#include <NvInferRuntime.h>
#include <chrono>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

namespace trt_edgellm
{

namespace logger
{

/*!
 * @brief Source code location information for automatic tracking
 *
 * Captures file, function, and line number information for logging.
 */
struct SourceLocation
{
    char const* file;     //!< Source file path
    char const* function; //!< Function name
    int32_t lineNumber;   //!< Line number

    /*!
     * @brief Constructor with manual location capture
     * @param f Source file path
     * @param func Function name
     * @param l Line number
     */
    SourceLocation(char const* f, char const* func, int32_t l)
        : file(f)
        , function(func)
        , lineNumber(l)
    {
    }
};

/**
 * @brief Enhanced Logger with automatic location tracking and nvinfer1 compatibility
 *
 * Features:
 * - Automatic source location tracking (file:line:function)
 * - Configurable formatting (timestamps, location info)
 * - nvinfer1::ILogger interface for TensorRT integration
 * - Multiple log levels with performance optimizations
 */
class EdgeLLMLogger : public nvinfer1::ILogger
{
public:
    EdgeLLMLogger() = default;
    ~EdgeLLMLogger() = default;

    /*!
     * @brief nvinfer1::ILogger interface implementation for TensorRT integration
     * @param severity Log severity level
     * @param msg Log message
     */
    void log(nvinfer1::ILogger::Severity severity, char const* msg) noexcept override
    {
        // Create source location for external library messages
        SourceLocation extLoc("TensorRT", "TensorRT_Internal", 0);
        logWithLocation(severity, msg, extLoc);
    }

    /*!
     * @brief Core logging function with automatic location tracking and formatting
     * @param level Log severity level
     * @param msg Log message
     * @param loc Source location information
     */
    void logWithLocation(nvinfer1::ILogger::Severity level, std::string const& msg, SourceLocation const& loc)
    {
        if (!shouldLog(level))
        {
            return;
        }

        // Format and output the message
        std::string formattedMsg = formatLogEntry(level, msg, loc);
        std::ostream& stream = (level <= nvinfer1::ILogger::Severity::kWARNING) ? std::cerr : std::cout;
        stream << formattedMsg << std::endl;
    }

    /*!
     * @brief Log debug message with location tracking
     * @param msg Log message
     * @param loc Source location information
     */
    void debug(std::string const& msg, SourceLocation const& loc)
    {
        logWithLocation(nvinfer1::ILogger::Severity::kVERBOSE, msg, loc);
    }

    /*!
     * @brief Log info message with location tracking
     * @param msg Log message
     * @param loc Source location information
     */
    void info(std::string const& msg, SourceLocation const& loc)
    {
        logWithLocation(nvinfer1::ILogger::Severity::kINFO, msg, loc);
    }

    /*!
     * @brief Log warning message with location tracking
     * @param msg Log message
     * @param loc Source location information
     */
    void warning(std::string const& msg, SourceLocation const& loc)
    {
        logWithLocation(nvinfer1::ILogger::Severity::kWARNING, msg, loc);
    }

    /*!
     * @brief Log error message with location tracking
     * @param msg Log message
     * @param loc Source location information
     */
    void error(std::string const& msg, SourceLocation const& loc)
    {
        logWithLocation(nvinfer1::ILogger::Severity::kERROR, msg, loc);
    }

    /*!
     * @brief Set minimum logging level
     * @param level Minimum severity level to log
     */
    void setLevel(nvinfer1::ILogger::Severity level)
    {
        mMinLevel = level;
    }

    /*!
     * @brief Get current logging level
     * @return Current minimum severity level
     */
    nvinfer1::ILogger::Severity getLevel() const
    {
        return mMinLevel;
    }

    /*!
     * @brief Configure whether to show timestamps in log output
     * @param show true to show timestamps, false to hide
     */
    void setShowTimestamp(bool show)
    {
        mShowTimestamp = show;
    }

    /*!
     * @brief Configure whether to show location info in log output
     * @param show true to show location, false to hide
     */
    void setShowLocation(bool show)
    {
        mShowLocation = show;
    }

    /*!
     * @brief Configure whether to show function names in log output
     * @param show true to show function names, false to hide
     */
    void setShowFunction(bool show)
    {
        mShowFunction = show;
    }

private:
    nvinfer1::ILogger::Severity mMinLevel = nvinfer1::ILogger::Severity::kINFO;
    bool mShowTimestamp = true;
    bool mShowLocation = true;
    bool mShowFunction = true;

    bool shouldLog(nvinfer1::ILogger::Severity level) const
    {
        return level <= mMinLevel; // Note: lower values are more severe in TensorRT
    }

    std::string formatLogEntry(
        nvinfer1::ILogger::Severity level, std::string const& msg, SourceLocation const& loc) const
    {
        std::ostringstream oss;

        // Timestamp
        if (mShowTimestamp)
        {
            auto now = std::chrono::system_clock::now();
            auto time_t = std::chrono::system_clock::to_time_t(now);
            auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;

            oss << "[" << std::put_time(std::localtime(&time_t), "%H:%M:%S") << "." << std::setfill('0') << std::setw(3)
                << ms.count() << "] ";
        }

        // Log level
        oss << "[" << getLevelString(level) << "]";

        // Location information
        if (mShowLocation && loc.file)
        {
            // Check if this is TensorRT message
            if (std::string(loc.file) == "TensorRT")
            {
                oss << " [TensorRT]";
            }
            else
            {
                std::filesystem::path p(loc.file);
                oss << " [" << p.filename().string() << ":" << loc.lineNumber;
                if (mShowFunction && loc.function)
                {
                    oss << ":" << loc.function;
                }
                oss << "]";
            }
        }

        // Message
        oss << " " << msg;

        return oss.str();
    }

    char const* getLevelString(nvinfer1::ILogger::Severity level) const
    {
        switch (level)
        {
        case nvinfer1::ILogger::Severity::kVERBOSE: return "DEBUG";
        case nvinfer1::ILogger::Severity::kINFO: return "INFO";
        case nvinfer1::ILogger::Severity::kWARNING: return "WARNING";
        case nvinfer1::ILogger::Severity::kERROR: return "ERROR";
        default: return "UNKNOWN";
        }
    }
};

/*!
 * @brief RAII-based function tracer for automatic entry/exit logging
 *
 * Creates automatic log messages when entering and exiting a scope.
 * Useful for tracing function execution flow.
 */
class ScopedFunctionTracer
{
public:
    /*!
     * @brief Constructor that logs function entry
     * @param logger Logger instance to use
     * @param funcName Name of the function being traced
     * @param loc Source location information
     */
    ScopedFunctionTracer(EdgeLLMLogger& logger, char const* funcName, SourceLocation const& loc)
        : mLogger(logger)
        , mFuncName(funcName)
        , mLoc(loc)
    {
        mLogger.debug("-> Entering " + mFuncName, mLoc);
    }

    /*!
     * @brief Destructor that logs function exit
     */
    ~ScopedFunctionTracer()
    {
        mLogger.debug("<- Exiting " + mFuncName, mLoc);
    }

private:
    EdgeLLMLogger& mLogger;
    std::string mFuncName;
    SourceLocation mLoc;
};

} // namespace logger

inline logger::EdgeLLMLogger gLogger{};

// Primary logging macros with automatic location tracking
// Usage: LOG_DEBUG("Value: %d", value); LOG_INFO("Message: %s", msg);

#define LOG_DEBUG(...)                                                                                                 \
    gLogger.debug(format::fmtstr(__VA_ARGS__), trt_edgellm::logger::SourceLocation(__FILE__, __FUNCTION__, __LINE__))

#define LOG_INFO(...)                                                                                                  \
    gLogger.info(format::fmtstr(__VA_ARGS__), trt_edgellm::logger::SourceLocation(__FILE__, __FUNCTION__, __LINE__))

#define LOG_WARNING(...)                                                                                               \
    gLogger.warning(format::fmtstr(__VA_ARGS__), trt_edgellm::logger::SourceLocation(__FILE__, __FUNCTION__, __LINE__))

#define LOG_ERROR(...)                                                                                                 \
    gLogger.error(format::fmtstr(__VA_ARGS__), trt_edgellm::logger::SourceLocation(__FILE__, __FUNCTION__, __LINE__))

// Conditional logging macros for performance-critical code
#define LOG_DEBUG_IF(condition, ...)                                                                                   \
    do                                                                                                                 \
    {                                                                                                                  \
        if (condition)                                                                                                 \
        {                                                                                                              \
            LOG_DEBUG(__VA_ARGS__);                                                                                    \
        }                                                                                                              \
    } while (0)

#define LOG_INFO_IF(condition, ...)                                                                                    \
    do                                                                                                                 \
    {                                                                                                                  \
        if (condition)                                                                                                 \
        {                                                                                                              \
            LOG_INFO(__VA_ARGS__);                                                                                     \
        }                                                                                                              \
    } while (0)

#define LOG_WARNING_IF(condition, ...)                                                                                 \
    do                                                                                                                 \
    {                                                                                                                  \
        if (condition)                                                                                                 \
        {                                                                                                              \
            LOG_WARNING(__VA_ARGS__);                                                                                  \
        }                                                                                                              \
    } while (0)

#define LOG_ERROR_IF(condition, ...)                                                                                   \
    do                                                                                                                 \
    {                                                                                                                  \
        if (condition)                                                                                                 \
        {                                                                                                              \
            LOG_ERROR(__VA_ARGS__);                                                                                    \
        }                                                                                                              \
    } while (0)

// Function tracing macro for automatic entry/exit logging
#define LOG_TRACE_FUNCTION()                                                                                           \
    trt_edgellm::logger::ScopedFunctionTracer gTracer(                                                                 \
        gLogger, __FUNCTION__, trt_edgellm::logger::SourceLocation(__FILE__, __FUNCTION__, __LINE__))

} // namespace trt_edgellm