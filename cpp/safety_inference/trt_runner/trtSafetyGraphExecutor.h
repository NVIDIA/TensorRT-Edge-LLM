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

#include "../runtime/safetyErrorState.h"

#include "../kernels/safetyAttentionModules.h"
#include "../runtime/safetyStatus.h"

#include <NvInferSafeRuntime.h>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace trt_edgellm
{
namespace safety
{

//! Borrowed safe engine and matched companion library, with the admitted fixed IO profiles.
struct SafetyGraphConfig
{
    void const* engine{nullptr};
    int64_t engineBytes{0};
    char const* companionSoPath{nullptr};
    std::vector<int64_t> declaredProfiles;
    //! Capture admits GPU IO payloads only; zero-byte metadata does not require storage.
    bool enableCudaGraphCapture{false};
};

//! Descriptor names and physical layouts remain valid until executor destruction.
struct SafetyTensorMetadata
{
    std::string name;
    nvinfer2::safe::TensorDescriptor descriptor;
};

//! Tensor order is the graph's IO index order for this admitted profile.
struct SafetyProfileMetadata
{
    int64_t profileIndex{0};
    std::vector<SafetyTensorMetadata> tensors;
};

//! Single-context graph owner; callers serialize every operation and keep its CUDA context current.
//! The shared error state, borrowed bindings, and streams outlive healthy deinitialize().
//! Destruction performs no CUDA or DLI calls. A fatal failure requires application containment, without cleanup retry.
class TRTSafetyGraphExecutor
{
public:
    explicit TRTSafetyGraphExecutor(SafetyErrorState& errorState) noexcept;
    ~TRTSafetyGraphExecutor() noexcept = default;
    TRTSafetyGraphExecutor(TRTSafetyGraphExecutor const&) = delete;
    TRTSafetyGraphExecutor& operator=(TRTSafetyGraphExecutor const&) = delete;
    TRTSafetyGraphExecutor(TRTSafetyGraphExecutor&&) = delete;
    TRTSafetyGraphExecutor& operator=(TRTSafetyGraphExecutor&&) = delete;

    //! Load modules for every application executor before initialize() or other CUDA runtime APIs.
    SafetyStatus loadModules() noexcept;

    //! Initialization only: create the graph and query metadata for admitted profiles.
    SafetyStatus initialize(SafetyGraphConfig const& config) noexcept;

    //! Initialization only: bind each profile once, using one fixed stream configuration for the executor.
    //! Tensor buffers and streams are borrowed, and callers supply valid warmup input contents.
    SafetyStatus bindProfile(int64_t profile, std::vector<nvinfer2::safe::TypedArray> const& bindings,
        cudaStream_t main, std::vector<cudaStream_t> const& aux) noexcept;

    //! Execute and synchronize one bound profile during initialization to prepare fixed plugin bindings.
    //! Capture also performs one validation replay; caller-supplied inputs must remain valid for that replay.
    SafetyStatus warmup(int64_t profile) noexcept;

    //! Enter Operational after every admitted profile is bound and warmed.
    SafetyStatus seal() noexcept;

    //! Operational only: execute the selected fixed profile and check host and device errors after completion.
    SafetyStatus execute(int64_t profile) noexcept;

    //! Healthy Deinit Preparation only: destroy the graph before releasing its modules.
    //! Immutable profile metadata remains valid until executor destruction.
    SafetyStatus deinitialize() noexcept;

    //! Includes asynchronously reported recorder failures, even before the next operation observes them.
    bool fatal() const noexcept;
    bool hasGraph() const noexcept;
    std::vector<SafetyProfileMetadata> const& profiles() const noexcept;
    int32_t requiredAuxStreams() const noexcept;

private:
    enum class State
    {
        kEMPTY,
        kMODULES_LOADED,
        kINITIALIZED,
        kSEALED,
        kCLOSED
    };

    struct ProfileState
    {
        bool bound{false};
        bool warmed{false};
        cudaGraph_t capturedGraph{nullptr};
        cudaGraphExec_t executableGraph{nullptr};
    };

    bool observeErrors() noexcept;
    SafetyStatus fail(nvinfer2::safe::ErrorCode code, char const* message) noexcept;
    bool updateStatus(nvinfer2::safe::ErrorCode status, char const* message) noexcept;
    bool updateStatus(cudaError_t status, char const* message) noexcept;
    bool updateStatus(CUresult status, char const* message) noexcept;
    SafetyStatus capture(size_t profile) noexcept;
    SafetyStatus run(size_t profile, bool captured) noexcept;
    size_t findProfile(int64_t profile) const noexcept;

    SafetyErrorState& mErrorState;
    SafetyAttentionModules mModules;
    nvinfer2::safe::ITRTGraph* mGraph{nullptr};
    nvinfer2::safe::RuntimeErrorInformation* mErrorBuffer{nullptr};
    std::vector<SafetyProfileMetadata> mProfiles;
    std::vector<ProfileState> mProfileStates;
    cudaStream_t mMainStream{nullptr};
    std::vector<cudaStream_t> mAuxStreams;
    int32_t mRequiredAuxStreams{0};
    bool mEnableCudaGraphCapture{false};
    State mState{State::kEMPTY};
};

} // namespace safety
} // namespace trt_edgellm
