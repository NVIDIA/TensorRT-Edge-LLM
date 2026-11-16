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

#include "../plugins/safetyPluginRegistry.h"

namespace trt_edgellm
{
namespace safety
{

//! Register admitted plugins and bind the executor-owned modules while creating a graph.
//! modules must be loaded before any other CUDA APIs and outlive the graph and its clones.
//! The executor owns fixed IO/workspace, warmup, sealing, async error checks, and graph teardown.
//! Registry operations and graph creation must be externally serialized.
//! companionSoPath is the absolute path to the trusted companion .so built with the engine.
nvinfer1::ErrorCode createSafetyGraph(SafetyAttentionModules& modules, nvinfer2::safe::ITRTGraph*& graph,
    void const* engine, int64_t engineBytes, nvinfer2::safe::ISafeRecorder& graphRecorder,
    char const* companionSoPath) noexcept;

} // namespace safety
} // namespace trt_edgellm
