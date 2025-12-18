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

#include "version.h"
#include "bindingNames.h"
#include "logger.h"
#include <fstream>
#include <sstream>

namespace trt_edgellm
{
namespace version
{

namespace
{
bool isValidVersionFormat(std::string const& version)
{
    if (version.empty())
    {
        return false;
    }

    std::stringstream ss(version);
    int major, minor, patch, build;
    char dot1, dot2, dot3;

    // Try to parse: major.minor.patch.build
    if (!(ss >> major >> dot1 >> minor >> dot2 >> patch >> dot3 >> build))
    {
        return false;
    }

    // Verify dots are correct
    if (dot1 != '.' || dot2 != '.' || dot3 != '.')
    {
        return false;
    }

    // Verify no extra characters after build number
    std::string remaining;
    if (ss >> remaining)
    {
        return false;
    }

    // Verify all parts are non-negative
    if (major < 0 || minor < 0 || patch < 0 || build < 0)
    {
        return false;
    }

    return true;
}
} // anonymous namespace

bool checkVersion(std::string const& modelVersion)
{
    if (modelVersion.empty())
    {
        LOG_WARNING(
            "Model does not have %s. Current runtime version: %s", binding_names::kEdgellmVersion, kRUNTIME_VERSION);
        return false;
    }

    if (!isValidVersionFormat(modelVersion))
    {
        LOG_ERROR("Invalid model version format: %s. Expected major.minor.patch.build", modelVersion.c_str());
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

} // namespace version
} // namespace trt_edgellm
