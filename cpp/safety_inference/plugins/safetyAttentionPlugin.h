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

#include "safetyAttentionContract.h"
#include "safetyPluginExport.h"

#include <NvInferSafeRecorder.h>

namespace trt_edgellm
{
namespace safety
{

class SafetyAttentionModules;

//! Initialization-only creator routing; serialize with all registry/builder/graph-creation operations.
//! Plugins capture a borrowed owner; clones retain it. The owner must outlive every plugin and graph.
class EDGELLM_SAFETY_PLUGIN_EXPORT SafetyAttentionModuleScope
{
public:
    explicit SafetyAttentionModuleScope(SafetyAttentionModules& modules) noexcept;
    ~SafetyAttentionModuleScope() noexcept;
    SafetyAttentionModuleScope(SafetyAttentionModuleScope const&) = delete;
    SafetyAttentionModuleScope& operator=(SafetyAttentionModuleScope const&) = delete;

private:
    SafetyAttentionModules* mPrevious;
};

//! Obtain a process-lifetime creator owned by the linked Safety plugin DSO.
//! The DSO must remain loaded until process exit, like the Safety registry itself.
EDGELLM_SAFETY_PLUGIN_EXPORT nvinfer2::safe::ISafePluginCreatorV3One& safetyAttentionCreator(
    AttentionRole role) noexcept;

} // namespace safety
} // namespace trt_edgellm

//! Host builder/remote tuning entry point. LLM uses "trt"; ViT uses "trt_edgellm".
extern "C" EDGELLM_SAFETY_PLUGIN_EXPORT nvinfer1::IPluginCreatorInterface* getSafetyPluginCreator(
    char const* pluginNamespace, char const* pluginName) noexcept;

//! Host-only bridge: the session and creator routing belong to the same plugin DSO.
//! A local caller keeps its CUDA context current and loads before initializing cudart.
extern "C" EDGELLM_SAFETY_PLUGIN_EXPORT int32_t createSafetyAttentionBuildSession(
    void** session, bool loadModules) noexcept;

//! Call only after GPU completion and destruction of all borrowing plugins.
//! On a terminal CUDA failure, discard the owner without unloading modules.
extern "C" EDGELLM_SAFETY_PLUGIN_EXPORT int32_t destroySafetyAttentionBuildSession(
    void* session, bool releaseModules) noexcept;
