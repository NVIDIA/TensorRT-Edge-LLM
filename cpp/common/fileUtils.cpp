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

#include "fileUtils.h"
#include "logger.h"

#include <cstdlib>
#include <filesystem>

namespace trt_edgellm
{
namespace file_io
{

bool copyFile(std::string const& srcPath, std::string const& dstPath)
{
    std::filesystem::path const src{srcPath};
    if (!std::filesystem::exists(src))
    {
        LOG_INFO("Failed to open file for reading: %s", srcPath.c_str());
        return false;
    }
    std::filesystem::path const dst{dstPath};
    if (std::filesystem::exists(dst) && std::filesystem::equivalent(src, dst))
    {
        LOG_INFO("Source and target file path are same, skip copying.");
    }
    else
    {
        try
        {
            auto const options = std::filesystem::copy_options::overwrite_existing;
            std::filesystem::copy(src, dst, options);
            LOG_INFO("Successfully copied %s to %s", srcPath.c_str(), dstPath.c_str());
        }
        catch (std::filesystem::filesystem_error& e)
        {
            LOG_ERROR("Error copying %s to %s - %s", srcPath.c_str(), dstPath.c_str(), e.what());
            return false;
        }
    }
    return true;
}

} // namespace file_io
} // namespace trt_edgellm
