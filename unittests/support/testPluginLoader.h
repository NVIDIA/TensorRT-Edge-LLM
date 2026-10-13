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

#pragma once

#include <array>
#include <cstddef>
#include <dlfcn.h>
#include <limits.h>
#include <string>
#include <unistd.h>

namespace trt_edgellm
{
namespace test
{

//! The plugin links into the build tree's root while the test executables sit in
//! subdirectories, so a path derived from /proc/self/exe looks in the wrong place.
//! EDGELLM_PLUGIN_PATH is the linker's own answer, passed down by
//! unittests/CMakeLists.txt. The executable-relative form stays as the fallback
//! for a binary run outside that build.
inline std::string const& pluginLibraryPath()
{
    static std::string const path = [] {
#ifdef EDGELLM_PLUGIN_PATH
        return std::string{EDGELLM_PLUGIN_PATH};
#else
        std::array<char, PATH_MAX> executablePath{};
        ssize_t const size = readlink("/proc/self/exe", executablePath.data(), executablePath.size() - 1);
        if (size <= 0)
        {
            return std::string{"./libNvInfer_edgellm_plugin.so"};
        }

        std::string const executable(executablePath.data(), static_cast<size_t>(size));
        size_t const separator = executable.find_last_of('/');
        if (separator == std::string::npos)
        {
            return std::string{"./libNvInfer_edgellm_plugin.so"};
        }
        return executable.substr(0, separator + 1) + "libNvInfer_edgellm_plugin.so";
#endif
    }();
    return path;
}

//! TensorRT retains registered plugin creators for the process lifetime, and the
//! plugin owns the AllReduce path registry, so the library must stay mapped after
//! the first test that opens it. RTLD_GLOBAL lets dlsym(RTLD_DEFAULT, ...) in the
//! runtime resolve the plugin's C-ABI registration symbols.
inline void* loadPluginLibrary() noexcept
{
    static void* const handle = dlopen(pluginLibraryPath().c_str(), RTLD_NOW | RTLD_GLOBAL);
    return handle;
}

} // namespace test
} // namespace trt_edgellm
