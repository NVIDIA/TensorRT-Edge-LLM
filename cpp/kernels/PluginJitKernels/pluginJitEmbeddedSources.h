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

#include <cstdint>
#include <utility>
#include <vector>

namespace trt_edgellm
{

enum class PluginJitProgram : uint32_t
{
    kXQA = 0,
    kNVFP4_A16_BLACKWELL_GEMV = 1,
};

struct PluginJitEmbeddedSources
{
    char const* mainSourceName;
    char const* mainSource;
    std::vector<std::pair<char const*, char const*>> headers;
};

PluginJitEmbeddedSources const& getPluginJitEmbeddedSources(PluginJitProgram program);

} // namespace trt_edgellm
