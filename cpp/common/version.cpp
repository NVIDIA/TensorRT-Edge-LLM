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

#include "version.h"
#include "bindingNames.h"
#include "logger.h"
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace trt_edgellm
{
namespace version
{

namespace
{
//! Parse version string into major, minor, patch components
//! @return true if parsing was successful, false otherwise
bool parseVersion(std::string const& version, int& major, int& minor, int& patch)
{
    if (version.empty())
    {
        return false;
    }

    std::stringstream ss(version);
    char dot1, dot2;

    // Try to parse: major.minor.patch
    if (!(ss >> major >> dot1 >> minor >> dot2 >> patch))
    {
        return false;
    }

    // Verify dots are correct
    if (dot1 != '.' || dot2 != '.')
    {
        return false;
    }

    // Verify no extra characters after patch number
    std::string remaining;
    if (ss >> remaining)
    {
        return false;
    }

    // Verify all parts are non-negative
    if (major < 0 || minor < 0 || patch < 0)
    {
        return false;
    }

    return true;
}
} // anonymous namespace

bool checkVersion(std::string const& modelVersion) noexcept
{
    try
    {
        if (modelVersion.empty())
        {
            LOG_WARNING("Model does not have %s. Current runtime version: %s", binding_names::kEdgellmVersion,
                kRUNTIME_VERSION);
            return false;
        }

        int major, minor, patch;
        if (!parseVersion(modelVersion, major, minor, patch))
        {
            LOG_ERROR("Invalid model version format: %s. Expected major.minor.patch", modelVersion.c_str());
            return false;
        }

        // Reject versions < 0.9.0
        if (major == 0 && minor < 9)
        {
            LOG_ERROR(
                "ONNX model version %s is no longer supported. Minimum supported version is 0.9.0 "
                "Please re-export your model with the latest tensorrt-edgellm.",
                modelVersion.c_str());
            return false;
        }

        if (modelVersion != std::string(kRUNTIME_VERSION))
        {
            LOG_WARNING("Model version %s does not match runtime version %s. Consider re-exporting or re-building.",
                modelVersion.c_str(), kRUNTIME_VERSION);
            return false;
        }

        return true;
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("Exception thrown while checking model version: '%s'", e.what());
        return false;
    }
    catch (...)
    {
        return false;
    }
}

} // namespace version
} // namespace trt_edgellm
