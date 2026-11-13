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

#include "safety_inference/kernels/safetyAttentionModules.h"
#include "safety_inference/trt_runner/safetyGraphInitialization.h"

#if defined(__QNXNTO__) && !defined(EDGELLM_TEST_QNX_STANDARD)
#include <cuda_runtime_api_safe_ex.h>
#endif

#include <atomic>
#include <cstdio>
#include <type_traits>

namespace trt_edgellm
{
namespace safety
{
namespace engine_test
{
constexpr int32_t kTOKENS{5};
constexpr int32_t kHEADS{2};
constexpr int32_t kHEAD_SIZE{64};
constexpr int32_t kELEMENTS{kTOKENS * kHEADS * kHEAD_SIZE};
constexpr float kSCALE{0.125F};

//! Test recorder retains the latest error and a saturated count; storage must outlive registry/graph users.
class Recorder final : public nvinfer2::safe::ISafeRecorder
{
public:
    int32_t getNbErrors() const noexcept override
    {
        return mHasError.load() ? 1 : 0;
    }
    nvinfer1::ErrorCode getErrorCode(int32_t) const noexcept override
    {
        return mCode.load();
    }
    ErrorDesc getErrorDesc(int32_t) const noexcept override
    {
        return "Safety engine smoke recorded an SDK error";
    }
    bool hasOverflowed() const noexcept override
    {
        return false;
    }
    void clear() noexcept override
    {
        mHasError = false;
    }
    bool reportError(nvinfer1::ErrorCode code, ErrorDesc description) noexcept override
    {
        mCode = code;
        mHasError = true;
        std::fprintf(
            stderr, "Safety recorder: %d %s\n", static_cast<int>(code), description == nullptr ? "" : description);
        return true;
    }
    RefCount incRefCount() noexcept override
    {
        return ++mReferences;
    }
    RefCount decRefCount() noexcept override
    {
        return --mReferences;
    }

private:
    std::atomic<bool> mHasError{false};
    std::atomic<nvinfer1::ErrorCode> mCode{nvinfer1::ErrorCode::kUNSPECIFIED_ERROR};
    std::atomic<int32_t> mReferences{0};
};

inline bool check(nvinfer1::ErrorCode status, char const* operation)
{
    if (status == nvinfer1::ErrorCode::kSUCCESS)
    {
        return true;
    }
    std::fprintf(stderr, "%s failed: Safety status %d\n", operation, static_cast<int>(status));
    return false;
}
inline bool check(CUresult status, char const* operation)
{
    if (status == CUDA_SUCCESS)
    {
        return true;
    }
    std::fprintf(stderr, "%s failed: CUDA driver status %d\n", operation, static_cast<int>(status));
    return false;
}
inline bool check(cudaError_t status, char const* operation)
{
    if (status == cudaSuccess)
    {
        return true;
    }
    std::fprintf(stderr, "%s failed: CUDA runtime status %d\n", operation, static_cast<int>(status));
    return false;
}

// CUDA Safety and native headers expose different public context-creation signatures.
template <typename CreateContext>
inline CUresult createContext(CreateContext create, CUcontext& context, CUdevice device)
{
    if constexpr (std::is_invocable_v<CreateContext, CUcontext*, unsigned int, CUdevice>)
    {
        return create(&context, 0U, device);
    }
    else
    {
        return create(&context, nullptr, 0U, device);
    }
}

//! Complete Safe CUDA's process-wide readiness handshake before the external VM transition.
inline bool prepareForOperational()
{
#if defined(__QNXNTO__) && !defined(EDGELLM_TEST_QNX_STANDARD)
    return check(cudaSafeExNotifyNvDvmsTransitionToOperational(), "cudaSafeExNotifyNvDvmsTransitionToOperational");
#else
    return true;
#endif
}

inline bool initializeModules(CUcontext& context, SafetyAttentionModules& modules)
{
    CUdevice device{};
    size_t stackLimit{};
    // Safe operational-capable APIs cannot lazily initialize cudart's state for a driver-created context.
    return check(cuInit(0), "cuInit") && check(cuDeviceGet(&device, 0), "cuDeviceGet")
        && check(createContext(&cuCtxCreate, context, device), "cuCtxCreate")
        && check(modules.load(), "SafetyAttentionModules::load")
        && check(cudaDeviceGetLimit(&stackLimit, cudaLimitStackSize), "cudaDeviceGetLimit(runtime initialization)");
}
} // namespace engine_test
} // namespace safety
} // namespace trt_edgellm
