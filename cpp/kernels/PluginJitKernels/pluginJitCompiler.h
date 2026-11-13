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

#include "pluginJitCompileCache.h"
#include "pluginJitEmbeddedSources.h"
#include "pluginJitKey.h"

#include <cstdint>
#include <string>
#include <vector>

namespace trt_edgellm
{

int32_t getPluginJitNvrtcMajorVersion();

//! NVRTC-compile `program` with `options` for the real device architecture.
//! \throws std::runtime_error when compilation fails or the result is empty or not a native cubin
std::vector<uint8_t> compilePluginJitKernel(
    PluginJitProgram program, std::vector<std::string> const& options, std::string const& description);

//! Compile once per distinct key, process-wide and shared across threads (see
//! PluginJitCompileCache), and return the key, the native cubin and the digest binding the two.
//! The caller validates `key` first; `options` is the NVRTC option list derived from it.
template <typename Key>
PluginJitKernel<Key> getOrCompilePluginJitKernel(PluginJitProgram const program, Key const& key,
    std::vector<std::string> const& options, std::string const& description)
{
    static PluginJitCompileCache<Key, PluginJitKernel<Key>, PluginJitKeyHasher<Key>> sCache;
    return sCache.getOrCompile(key, [&] {
        PluginJitKernel<Key> kernel;
        kernel.key = key;
        kernel.cubin = compilePluginJitKernel(program, options, description);
        kernel.digest = computePluginJitDigest(key, kernel.cubin.data(), kernel.cubin.size());
        return kernel;
    });
}

} // namespace trt_edgellm
