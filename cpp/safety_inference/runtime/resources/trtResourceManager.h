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

#include "../lifecycleState.h"
#include "../safetyErrorState.h"

#include "../../trt_runner/trtSafetyGraphExecutor.h"

#include <string>
#include <vector>

namespace trt_edgellm
{
namespace safety
{

//! Stable IO storage for one fixed profile; tensor order matches SafetyProfileMetadata.
struct SafetyProfileResources
{
    int64_t profileIndex{0};
    std::vector<nvinfer2::safe::TypedArray> bindings;
};

//! Explicit exact in-place IO storage for one alias-safe executor; partial overlap is not admitted.
struct SafetyIOAliasGroup
{
    size_t executorIndex{0};
    std::vector<std::string> tensors;
};

//! Owns IO allocations and streams for serialized execution in one caller-owned CUDA context.
//! SafetyRuntime keeps this owner alive until every borrowing graph is destroyed.
//! All operations are externally serialized. Destruction performs no CUDA/DLI calls.
class TRTResourceManager
{
public:
    TRTResourceManager(LifecycleState& lifecycleState, SafetyErrorState& errorState) noexcept;
    ~TRTResourceManager() noexcept = default;
    TRTResourceManager(TRTResourceManager const&) = delete;
    TRTResourceManager& operator=(TRTResourceManager const&) = delete;

    //! Initialization only: allocate and bind every admitted profile once.
    //! A CUDA/DLI failure retains partial resources for application containment without rollback.
    SafetyStatus prepareResources(std::vector<TRTSafetyGraphExecutor*> const& executors,
        std::vector<size_t> const& mainStreamIds = {}, std::vector<SafetyIOAliasGroup> const& aliases = {}) noexcept;

    //! Healthy Deinit Preparation only, after every borrowing graph has been destroyed.
    SafetyStatus deinitialize(std::vector<TRTSafetyGraphExecutor*> const& executors) noexcept;

    std::vector<SafetyProfileResources> const& profiles(size_t executorIndex) const noexcept;
    cudaStream_t mainStream(size_t executorIndex) const noexcept;
    bool fatal() const noexcept;

private:
    struct Allocation
    {
        void* data{nullptr};
        nvinfer2::safe::MemoryPlacement placement{nvinfer2::safe::MemoryPlacement::kNONE};
        uint64_t bytes{0};
    };
    struct ExecutorResources
    {
        TRTSafetyGraphExecutor* executor{nullptr};
        std::vector<Allocation> allocations;
        std::vector<size_t> tensorAllocations;
        std::vector<SafetyProfileResources> profiles;
        cudaStream_t main{nullptr};
        size_t mainStreamId{0};
        std::vector<cudaStream_t> auxiliary;
    };

    SafetyStatus validateExecutors(std::vector<TRTSafetyGraphExecutor*> const& executors) noexcept;
    SafetyStatus validateAliases(
        std::vector<TRTSafetyGraphExecutor*> const& executors, std::vector<SafetyIOAliasGroup> const& aliases) noexcept;
    SafetyStatus planStorage(std::vector<TRTSafetyGraphExecutor*> const& executors,
        std::vector<size_t> const& mainStreamIds, std::vector<SafetyIOAliasGroup> const& aliases) noexcept;
    SafetyStatus createStreams(ExecutorResources& owned) noexcept;
    SafetyStatus allocateIO(ExecutorResources& owned) noexcept;
    SafetyStatus bindIO(ExecutorResources& owned) noexcept;
    bool isHealthy() const noexcept;
    bool dependenciesHealthy() const noexcept;
    bool observeErrors() noexcept;
    bool observeDependencies() noexcept;
    bool updateStatus(cudaError_t status, nvinfer2::safe::ErrorCode code, char const* message) noexcept;
    SafetyStatus fail(nvinfer2::safe::ErrorCode code, char const* message) noexcept;
    SafetyErrorState& mErrorState;
    std::vector<ExecutorResources> mExecutors;
    std::vector<cudaStream_t> mMainStreams;
    LifecycleState const& mLifecycleState;
};

} // namespace safety
} // namespace trt_edgellm
