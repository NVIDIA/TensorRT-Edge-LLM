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

#include "safetyAttentionPlugin.h"
#include <NvInferVersion.h>

namespace trt_edgellm
{
namespace safety
{

//! Return the plugin-DSO-owned process recorder used for all registry operations.
//! The recorder, creators, and their DSO remain alive until process exit.
EDGELLM_SAFETY_PLUGIN_EXPORT nvinfer2::safe::ISafeRecorder& processRegistrationRecorder() noexcept;

//! Register admitted creators against processRegistrationRecorder().
//! Initialization only and externally serialized with other registry operations. Any error is terminal;
//! do not retry registration or substitute the Standard registry.
EDGELLM_SAFETY_PLUGIN_EXPORT nvinfer1::ErrorCode registerSafetyPlugins(
    nvinfer2::safe::ISafePluginRegistry& registry) noexcept;

} // namespace safety
} // namespace trt_edgellm
