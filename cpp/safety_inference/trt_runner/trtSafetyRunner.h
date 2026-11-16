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

#include "../runtime/resources/trtResourceManager.h"
#include "../runtime/safetyErrorState.h"

#include "lmStageContext.h"

#include <memory>
#include <vector>

namespace trt_edgellm
{
namespace safety
{

struct SafetyRuntimeInitSpec
{
    std::vector<SafetyGraphConfig> executors;
    std::vector<LMStagePlan> stages;
    std::vector<size_t> executorMainStreamIds{};
    std::vector<SafetyIOAliasGroup> ioAliases{};
};

//! Initialization callback supplies known-valid warmup inputs on the context's explicit main stream.
//! Data enqueued on other streams must establish dependencies with that stream before returning.
using LMStageWarmupCallback = SafetyStatus (*)(LMStageContext const&, int64_t profile, void* userData);

//! Owns graphs and maps stage/slot pairs to fixed resources in one current CUDA context.
//! SafetyRuntime owns the resource manager and governs lifecycle transitions.
//! The shared runtime error state and recorder outlive its graphs and terminal containment.
//! One active request, externally serialized; destruction performs no CUDA/DLI calls.
class TRTSafetyRunner
{
public:
    TRTSafetyRunner(
        LifecycleState& lifecycleState, SafetyErrorState& errorState, TRTResourceManager& resources) noexcept;
    ~TRTSafetyRunner() noexcept = default;
    TRTSafetyRunner(TRTSafetyRunner const&) = delete;
    TRTSafetyRunner& operator=(TRTSafetyRunner const&) = delete;

    //! Initialization only. Validate assignments, query graphs, allocate/bind IO, and publish context metadata.
    //! The caller creates the context before this call; modules load before any CUDA runtime API.
    SafetyStatus initialize(SafetyRuntimeInitSpec const& spec) noexcept;

    //! Complete a context once from caller-owned model-local resources, all stable through Deinitialization.
    SafetyStatus completeContext(
        char const* stage, ResourceSlotId slot, std::vector<SafetyModelLocalResourceView> const& views) noexcept;

    //! Initialization only. Warm every declared stage/profile, then seal all executors for Operational use.
    SafetyStatus prepare(LMStageWarmupCallback warmup, void* userData = nullptr) noexcept;

    //! Operational catalog dispatch requires one immutable profile for the stage/slot.
    //! Preparation updates preallocated payloads on the selected context stream before execution.
    SafetyStatus runLMStage(LMStageId stage, ResourceSlotId slot, LMStageOutputs& outputs,
        LMStagePrepareCallback prepare = nullptr, void* userData = nullptr) noexcept;

    //! Healthy Deinit Preparation: destroy graphs/modules and stop dispatch, retaining IO/streams.
    SafetyStatus deinitializeGraphs() noexcept;

    //! Release IO/streams after graphs and caller-owned model-local/declared resources are destroyed.
    SafetyStatus deinitializeResources() noexcept;

    std::vector<LMStageContext> const& contexts() const noexcept;
    bool fatal() const noexcept;

private:
    bool observeErrors() noexcept;
    SafetyStatus fail(nvinfer2::safe::ErrorCode code, char const* message) noexcept;
    LMStageContext* findContext(char const* stage, ResourceSlotId slot) noexcept;
    LMStageContext* findContext(LMStageId stage, ResourceSlotId slot) noexcept;

    SafetyErrorState& mErrorState;
    TRTResourceManager& mResources;
    std::vector<std::unique_ptr<TRTSafetyGraphExecutor>> mExecutors;
    std::vector<TRTSafetyGraphExecutor*> mExecutorViews;
    std::vector<LMStageContext> mContexts;
    LifecycleState const& mLifecycleState;
};

} // namespace safety
} // namespace trt_edgellm
