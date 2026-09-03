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

#include "runtime/multiDevice/backends/nccl/tensorParallelNcclResources.h"
#include "common/cudaUtils.h"
#include "runtime/multiDevice/ncclCollectiveBackend.h"

#include <array>
#include <dlfcn.h>
#include <exception>
#include <gtest/gtest.h>
#include <limits.h>
#include <string>
#include <unistd.h>

using namespace trt_edgellm;
using namespace trt_edgellm::rt;

namespace
{

//! The plugin is built into the build tree's root while the test executables sit
//! in a subdirectory, so deriving the path from /proc/self/exe would look in the
//! wrong place. EDGELLM_PLUGIN_PATH is the linker's own answer, passed down by
//! unittests/CMakeLists.txt; the executable-relative form stays as the fallback
//! for a binary run outside that build.
std::string const& pluginLibraryPath()
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

void* loadPluginLibrary() noexcept
{
    // TensorRT retains registered plugin creators for the process lifetime, so
    // the library must remain loaded after this smoke test completes.
    static void* const handle = dlopen(pluginLibraryPath().c_str(), RTLD_NOW | RTLD_GLOBAL);
    return handle;
}

} // namespace

TEST(TensorParallelNcclResourcesTest, RegistersOwnedCommunicatorsWithPlugin)
{
    int32_t constexpr kTpSize = 2;
    int32_t const deviceCount = detectCudaDeviceCount();
    if (deviceCount < kTpSize)
    {
        GTEST_SKIP() << "NCCL plugin-resource smoke test requires " << kTpSize << " CUDA devices, found "
                     << deviceCount;
    }

    void* const pluginHandle = loadPluginLibrary();
    ASSERT_NE(pluginHandle, nullptr) << "Failed to load " << pluginLibraryPath() << ": " << dlerror();

    try
    {
        NcclCollectiveBackend::load();
    }
    catch (std::exception const& e)
    {
        GTEST_SKIP() << "NCCL runtime is unavailable: " << e.what();
    }

    auto resources = createTensorParallelNcclResources(kTpSize, {0, 1}, {0, 1}, {}, true);
    ASSERT_NE(resources, nullptr);
    EXPECT_EQ(resources->type(), AllReducePathType::kNccl);
    EXPECT_TRUE(resources->registered());

    RuntimeCollectiveResources* const collectives = resources->runtimeCollectives();
    ASSERT_NE(collectives, nullptr);
    EXPECT_NE(collectives->communicatorForRank(0), nullptr);
    EXPECT_NE(collectives->communicatorForRank(1), nullptr);
}
