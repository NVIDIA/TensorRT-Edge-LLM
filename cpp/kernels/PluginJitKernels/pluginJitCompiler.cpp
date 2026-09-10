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

#include "pluginJitCompiler.h"

#include "common/logger.h"

#include <chrono>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include <nvrtc.h>

namespace trt_edgellm
{
namespace
{

void checkNvrtc(nvrtcResult const result, char const* const call)
{
    if (result != NVRTC_SUCCESS)
    {
        throw std::runtime_error(std::string("NVRTC call failed in ") + call + ": " + nvrtcGetErrorString(result));
    }
}

struct NvrtcProgramDeleter
{
    void operator()(std::remove_pointer_t<nvrtcProgram>* program) const noexcept
    {
        if (program != nullptr)
        {
            nvrtcProgram handle{program};
            (void) nvrtcDestroyProgram(&handle);
        }
    }
};

using NvrtcProgramPtr = std::unique_ptr<std::remove_pointer_t<nvrtcProgram>, NvrtcProgramDeleter>;

std::string getProgramLog(nvrtcProgram const program)
{
    size_t logSize{0};
    checkNvrtc(nvrtcGetProgramLogSize(program, &logSize), "nvrtcGetProgramLogSize");
    if (logSize == 0)
    {
        return {};
    }
    std::string log(logSize, '\0');
    checkNvrtc(nvrtcGetProgramLog(program, log.data()), "nvrtcGetProgramLog");
    return log;
}

} // namespace

int32_t getPluginJitNvrtcMajorVersion()
{
    int major{};
    int minor{};
    checkNvrtc(nvrtcVersion(&major, &minor), "nvrtcVersion");
    return static_cast<int32_t>(major);
}

std::vector<uint8_t> compilePluginJitKernel(
    PluginJitProgram const program, std::vector<std::string> const& optionStrings, std::string const& description)
{
    PluginJitEmbeddedSources const& embedded = getPluginJitEmbeddedSources(program);

    std::vector<char const*> headerContents;
    std::vector<char const*> headerNames;
    headerContents.reserve(embedded.headers.size());
    headerNames.reserve(embedded.headers.size());
    for (auto const& [name, content] : embedded.headers)
    {
        headerNames.push_back(name);
        headerContents.push_back(content);
    }

    std::vector<char const*> options;
    options.reserve(optionStrings.size());
    for (std::string const& option : optionStrings)
    {
        options.push_back(option.c_str());
    }

    nvrtcProgram rawProgram{};
    checkNvrtc(nvrtcCreateProgram(&rawProgram, embedded.mainSource, embedded.mainSourceName,
                   static_cast<int32_t>(headerContents.size()), headerContents.data(), headerNames.data()),
        "nvrtcCreateProgram");
    NvrtcProgramPtr const nvrtcProgram(rawProgram);

    auto const compileStart = std::chrono::steady_clock::now();
    nvrtcResult const compileResult
        = nvrtcCompileProgram(nvrtcProgram.get(), static_cast<int32_t>(options.size()), options.data());
    auto const compileEnd = std::chrono::steady_clock::now();
    auto const compileMs = std::chrono::duration_cast<std::chrono::milliseconds>(compileEnd - compileStart).count();

    std::string const compileLog = getProgramLog(nvrtcProgram.get());
    if (compileResult != NVRTC_SUCCESS)
    {
        throw std::runtime_error("Failed to NVRTC compile " + description + "\n" + compileLog);
    }
    if (!compileLog.empty())
    {
        LOG_DEBUG("NVRTC compile log for %s:\n%s", description.c_str(), compileLog.c_str());
    }

    // Plugin JIT callers compile for the real device architecture and require
    // a native module image. A virtual-architecture PTX image is deliberately
    // not part of this production contract: unavailable, failed, or empty
    // CUBIN retrieval must fail engine build instead of deferring compilation
    // or module selection to runtime.
    size_t cubinSize{0};
    checkNvrtc(nvrtcGetCUBINSize(nvrtcProgram.get(), &cubinSize), "nvrtcGetCUBINSize");
    if (cubinSize == 0)
    {
        throw std::runtime_error("NVRTC returned an empty cubin for " + description);
    }
    std::vector<uint8_t> cubin(cubinSize);
    checkNvrtc(nvrtcGetCUBIN(nvrtcProgram.get(), reinterpret_cast<char*>(cubin.data())), "nvrtcGetCUBIN");
    LOG_INFO("Compiled %s with NVRTC (%zu bytes, %lld ms)", description.c_str(), cubin.size(),
        static_cast<long long>(compileMs));
    return cubin;
}

} // namespace trt_edgellm
