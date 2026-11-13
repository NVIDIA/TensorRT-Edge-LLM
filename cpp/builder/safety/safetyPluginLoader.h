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

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <initializer_list>
#include <memory>
#include <string>

namespace trt_edgellm::safety::builder
{

class SafetyPluginRegistration;

struct SafetyPluginDescriptor
{
    char const* pluginNamespace;
    char const* pluginName;
    char const* pluginVersion;
};

//! Load the Safety plugin library and register the required creators.
//! The returned object keeps the library loaded and deregisters creators before
//! unloading it.
std::unique_ptr<SafetyPluginRegistration> loadSafetyPlugins(
    std::string const& configuredPath, std::initializer_list<SafetyPluginDescriptor> requiredPlugins);

class SafetyPluginRegistration
{
public:
    virtual ~SafetyPluginRegistration() noexcept = default;

    //! Bind attention creators before parsing ONNX; remote tuning performs no local CUDA setup.
    virtual bool initializeAttentionModules(bool remoteAutoTuning) noexcept = 0;

    //! Complete profiling while TRT objects and their borrowed buffers are still alive.
    virtual bool completeGpuWork() noexcept = 0;

    //! Destroy all TRT objects first. Returns false if GPU completion or teardown fails.
    virtual bool finishBuild() noexcept = 0;
};

} // namespace trt_edgellm::safety::builder
