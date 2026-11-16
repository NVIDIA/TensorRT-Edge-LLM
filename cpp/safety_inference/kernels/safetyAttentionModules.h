/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "safetyAttentionKernels.h"

#include <array>
#include <atomic>
#include <cuda.h>

namespace trt_edgellm
{
namespace safety
{

//! Executor-owned modules for one CUDA context. Keep that context current for every call.
//! Outlives all borrowing plugins/graphs; setup and teardown must be externally serialized.
//! Destruction performs no CUDA calls: release explicitly during healthy Deinit Preparation.
class __attribute__((visibility("default"))) SafetyAttentionModules
{
public:
    SafetyAttentionModules() noexcept = default;
    SafetyAttentionModules(SafetyAttentionModules const&) = delete;
    SafetyAttentionModules& operator=(SafetyAttentionModules const&) = delete;

    //! Safe CUDA cuModuleLoadData requires loading immediately after context creation, before other CUDA APIs.
    CUresult load() noexcept;

    //! Initialization only: resolve/configure functions after all application modules and cudart initialize.
    //! Shared plugins may call repeatedly before sealing; no repeated CUDA work is performed.
    CUresult prepare() noexcept;

    //! Seal after graph warmup, before Operational; new or changed tensor bindings are then rejected.
    CUresult seal() noexcept;

    //! Executor calls after GPU completion and destruction of every borrowing graph/plugin.
    //! Failure is terminal for this owner; no rollback, retry or automatic cleanup is attempted.
    CUresult release() noexcept;
    bool ready() const noexcept;

    //! Admit fixed bindings before any CUDA work. Only Initialization may encode new tensor maps.
    CUresult prepareBindings(fmha::PreparedBindings& bindings, AttentionLaunch const& shape, bool visual,
        half const* queries, half const* keys, half const* values, half* output) noexcept;

    //! Launch a prepared variant; all pointers are borrowed from fixed plugin bindings.
    //! Device metadata must be validated on stream before this call; valid gates all FMHA device reads.
    CUresult launch(fmha::PreparedBindings& bindings, AttentionLaunch const& shape, bool visual, half const* queries,
        half const* keys, half const* values, int32_t const* cuQueryLengths, int32_t const* cuKvLengths,
        int32_t const* sequenceEnds, half* output, int32_t const* valid, cudaStream_t stream) noexcept;

private:
    enum class State
    {
        kEMPTY,
        kLOADING,
        kLOADED,
        kREADY,
        kOPERATIONAL,
        kFAILED,
        kCLOSED
    };
    static_assert(std::atomic<State>::is_always_lock_free);
    CUresult fail(CUresult status) noexcept;
    std::atomic<State> mState{State::kEMPTY};
    // Two engine roles, each with D64 and D128; checked against the admitted artifact inventory.
    std::array<CUmodule, 4> mModules{};
    std::array<CUfunction, 4> mFunctions{};
    int32_t mSmCount{};
};

} // namespace safety
} // namespace trt_edgellm
