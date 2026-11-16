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

#include "../runtime/model_inference/modelInferenceTypes.h"
#include "safetyTensorView.h"

#include <cuda_runtime_api.h>

#include <string>
#include <vector>

namespace trt_edgellm
{
namespace safety
{

struct SafetyProfileMetadata;
struct SafetyProfileResources;

using ResourceSlotId = size_t;

//! A model-local buffer supplied by the declared-resource owner during Initialization.
struct SafetyModelLocalResourceView
{
    std::string name;
    void* data{nullptr};
    size_t sizeInBytes{0};
};

struct SafetyModelLocalResourceRequirement
{
    std::string name;
    size_t minimumBytes{0};
};

//! Logical assignment only; no resource ownership or device handles.
struct LMStagePlan
{
    std::string id;
    ResourceSlotId slot{0};
    size_t executorIndex{0};
    std::vector<int64_t> profiles;
    std::vector<SafetyModelLocalResourceRequirement> modelLocalResources;
};

struct LMStageProfileView
{
    SafetyProfileMetadata const* metadata{nullptr};
    SafetyProfileResources const* resources{nullptr};
    std::vector<SafetyTensorView> tensors{};
};

//! Immutable after readiness. Stage/slot pairs sharing an executor also share its fixed IO storage.
//! Model-local buffers, executor metadata, IO buffers, and streams are non-owning views.
struct LMStageContext
{
    LMStagePlan plan;
    cudaStream_t mainStream{nullptr};
    std::vector<LMStageProfileView> profiles;
    std::vector<SafetyModelLocalResourceView> modelLocalResources;
    bool complete{false};

    //! Pure cached lookup; returned metadata borrows the context's fixed IO storage.
    SafetyStatus getTensorView(int64_t profile, std::string_view name, SafetyTensorView const*& output) const noexcept;

    //! Name-only lookup requires a singleton profile; it never chooses among multiple admitted profiles.
    SafetyStatus getTensorView(std::string_view name, SafetyTensorView const*& output) const noexcept;
};

namespace detail
{
//! Initialization-only SDK adapter; cached lookups do not allocate or inspect SDK state.
SafetyStatus initializeTensorViews(LMStageProfileView& profile) noexcept;
} // namespace detail

} // namespace safety
} // namespace trt_edgellm
