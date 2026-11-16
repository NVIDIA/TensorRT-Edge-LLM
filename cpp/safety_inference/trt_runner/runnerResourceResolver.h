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

#include "../runtime/resources/runtimeResource.h"
#include "lmStageContext.h"

namespace trt_edgellm::safety
{

//! Initialization-only conversion of a linear, non-vectorized fixed-profile binding.
SafetyStatus describeRuntimeTensor(SafetyTensorView const& tensor, int64_t profile, RuntimeTensorDesc& output) noexcept;

//! Runtime-owned object published after its factory has resolved prepared storage.
struct ModelLocalDataBinding
{
    ResourceSlotId slot{0};
    std::string name{};
    SafetyModelLocalData* object{nullptr};
};

//! Converts prepared Runner bindings into stable borrowed resource views during Initialization.
//! Runner IO/streams and the model-local registry outlive all consumers. The registry may grow while
//! factories are initialized; published objects stay at stable addresses. Destruction releases only metadata.
class RunnerResourceResolver final : public BorrowedResourceResolver
{
public:
    explicit RunnerResourceResolver(std::vector<ModelLocalDataBinding> const& modelLocalData) noexcept;
    RunnerResourceResolver(RunnerResourceResolver const&) = delete;
    RunnerResourceResolver& operator=(RunnerResourceResolver const&) = delete;

    //! Snapshot singleton-profile contexts before resolving any declarations; callable once on success.
    SafetyStatus prepare(std::vector<LMStageContext> const& contexts) noexcept;
    SafetyStatus resolve(
        RuntimeBorrowedResourceRequest const& request, RuntimeBorrowedResource& output) const noexcept override;

private:
    struct Stage
    {
        std::string id{};
        ResourceSlotId slot{0};
        cudaStream_t stream{nullptr};
        std::vector<RuntimeTensorDesc> descriptors{};
        std::vector<RuntimeTensorView> tensors{};
    };
    std::vector<ModelLocalDataBinding> const& mModelLocalData;
    std::vector<Stage> mStages{};
};

} // namespace trt_edgellm::safety
