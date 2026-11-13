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

#include "runtime/multiDevice/multiDevicePluginResourceFactory.h"

#include "common/stringUtils.h"
#include "runtime/multiDevice/multiDevicePluginResources.h"
#include "runtime/multiDevice/tensorParallelPluginResources.h"

#include <array>
#include <stdexcept>
#include <utility>

namespace trt_edgellm
{
namespace rt
{

namespace
{

using PluginResourceCreator
    = std::unique_ptr<MultiDevicePluginResources> (*)(MultiDevicePluginResourceConfig const& config);

std::unique_ptr<MultiDevicePluginResources> createTensorParallelPluginResources(
    MultiDevicePluginResourceConfig const& config)
{
    TensorParallelPluginResourcesConfig const resourcesConfig{
        config.groupConfig.size, config.localRanks, config.localDevices, config.backendHandles,
        config.ownsBackendHandles,
    };
    return std::make_unique<TensorParallelPluginResources>(resourcesConfig);
}

using PluginResourceEntry = std::pair<ParallelType, PluginResourceCreator>;

std::array<PluginResourceEntry, 1> const& pluginResourceRegistry()
{
    static std::array<PluginResourceEntry, 1> const kRegistry{
        PluginResourceEntry{ParallelType::kTensor, createTensorParallelPluginResources},
    };
    return kRegistry;
}

} // namespace

std::unique_ptr<MultiDevicePluginResources> createMultiDevicePluginResources(
    MultiDevicePluginResourceConfig const& config)
{
    for (PluginResourceEntry const& entry : pluginResourceRegistry())
    {
        if (entry.first == config.groupConfig.type)
        {
            return entry.second(config);
        }
    }
    throw std::runtime_error(format::fmtstr(
        "No plugin resource adapter is registered for '%s' parallelism.", parallelTypeName(config.groupConfig.type)));
}

} // namespace rt
} // namespace trt_edgellm
