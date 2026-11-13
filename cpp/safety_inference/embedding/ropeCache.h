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

#include "safety_inference/runtime/resources/runtimeResource.h"

namespace trt_edgellm::safety
{

struct RopeCacheSpec
{
    int32_t contextLength{0};
    int32_t rotaryDimension{0};
    float theta{0.0F};
    float scale{1.0F};
};

SafetyStatus initializeRopeCache(RopeCacheSpec const& spec, RuntimeMemoryView upload, RuntimeMemoryView destination,
    cudaStream_t stream, cudaEvent_t completion) noexcept;

} // namespace trt_edgellm::safety
