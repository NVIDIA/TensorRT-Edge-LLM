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

#include "logger.h"
#include "stringUtils.h"
#include <NvInfer.h>
#include <dlfcn.h>
#include <memory>
#include <stdexcept>

namespace trt_edgellm
{

/*!
 * @brief Custom deleter for dynamic library handles
 *
 * Handles proper cleanup of dynamically loaded libraries using dlclose.
 */
struct DlDeleter
{
    //! @brief Delete operator for library handles
    //! @param handle Library handle to close
    void operator()(void* handle) const noexcept
    {
        if (handle)
        {
            dlclose(handle);
        }
    }
};

/*!
 * @brief Load TensorRT Edge-LLM plugin library
 *
 * Loads the plugin library from the path specified by EDGELLM_PLUGIN_PATH
 * environment variable. If not set, defaults to build/libNvInfer_edgellm_plugin.so.
 *
 * @return Unique pointer to library handle, or nullptr on failure
 */
inline std::unique_ptr<void, DlDeleter> loadEdgellmPluginLib(void)
{
    char const* pluginPath = std::getenv("EDGELLM_PLUGIN_PATH");

    if (pluginPath != nullptr)
    {
        LOG_INFO("EDGELLM_PLUGIN_PATH: %s", pluginPath);
    }
    else
    {
        LOG_INFO("EDGELLM_PLUGIN_PATH variable is not set. Default to build/libNvInfer_edgellm_plugin.so");
        pluginPath = "build/libNvInfer_edgellm_plugin.so";
    }

    auto handle = std::unique_ptr<void, DlDeleter>(dlopen(pluginPath, RTLD_LAZY));
    if (!handle)
    {
        LOG_ERROR("Cannot open plugin library: %s", dlerror());
        return std::unique_ptr<void, DlDeleter>(nullptr);
    }
    return handle;
}

} // namespace trt_edgellm
