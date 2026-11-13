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

#include "runtimeResource.h"

namespace trt_edgellm::safety
{

//! SafetyRuntime owns the object; payload storage is borrowed from PreparedResources.
//! Destroy the object before releasing declared resources. Destruction performs no CUDA/DLI calls.
class SafetyModelLocalData
{
public:
    virtual ~SafetyModelLocalData() = default;
};

//! Configured from immutable artifacts and stage metadata during Initialization.
class SafetyModelLocalDataFactory
{
public:
    virtual ~SafetyModelLocalDataFactory() = default;
    virtual SafetyStatus declareResources(ResourceDeclaration& output) const noexcept = 0;
    //! Construct metadata over prepared storage; transfer object ownership to SafetyRuntime.
    virtual SafetyStatus create(
        PreparedResources const& resources, std::unique_ptr<SafetyModelLocalData>& output) const noexcept = 0;
};

struct SafetyModelLocalDataPlan
{
    std::size_t resourceSlot{0};
    std::string name{};
    SafetyModelLocalDataFactory const* factory{nullptr};
};

} // namespace trt_edgellm::safety
