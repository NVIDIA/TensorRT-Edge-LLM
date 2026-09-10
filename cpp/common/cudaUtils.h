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

#include "common/checkMacros.h"
#include "common/logger.h"
#include <cstdint>
#include <cuda_runtime.h>

namespace trt_edgellm
{

/*!
 * @brief Divide and round up utility function
 *
 * Computes ceiling division: (a + n - 1) / n
 *
 * @tparam T1 Type of dividend
 * @tparam T2 Type of divisor
 * @param a Dividend
 * @param n Divisor
 * @return Ceiling of a/n
 */
template <typename T1, typename T2>
__host__ __device__ inline size_t divUp(const T1& a, const T2& n) noexcept
{
    size_t tmp_a = static_cast<size_t>(a);
    size_t tmp_n = static_cast<size_t>(n);
    return (tmp_a + tmp_n - 1) / tmp_n;
}

/*!
 * @brief Get CUDA compute capability version, cached once
 *
 * Returns the compute capability as an integer (e.g., 89 for SM 8.9).
 *
 * @return Compute capability version (major * 10 + minor)
 * @throws std::runtime_error If CUDA device query fails
 */
inline int getSMVersion()
{
    // Edge-LLM runs on one target GPU per process. Query its hardware constant once, then reuse it.
    static int const smVersion = []() {
        int device{-1};
        CUDA_CHECK(cudaGetDevice(&device));
        int smMajor = 0;
        int smMinor = 0;
        CUDA_CHECK(cudaDeviceGetAttribute(&smMajor, cudaDevAttrComputeCapabilityMajor, device));
        CUDA_CHECK(cudaDeviceGetAttribute(&smMinor, cudaDevAttrComputeCapabilityMinor, device));
        return smMajor * 10 + smMinor;
    }();
    return smVersion;
}

/*!
 * @brief Multiprocessor (SM) count of the current device, cached once.
 *
 * Used as the runtime max_active_clusters / sm_count argument of persistent
 * CuTe DSL kernels so AOT artifacts size their persistent grid for the GPU
 * they actually launch on, instead of a value baked at export time on the
 * build machine. For cluster shape (1,1) the max active cluster count of the
 * full-shared-memory persistent kernels equals the SM count.
 *
 * @return Multiprocessor count (>= 1)
 * @throws std::runtime_error If the current device or its SM count cannot be queried
 */
inline int32_t getDeviceMultiProcessorCount()
{
    // Edge-LLM runs on one target GPU per process. Query its hardware constant
    // once, then reuse it on every kernel launch.
    static int32_t const smCount = []() {
        int device{-1};
        CUDA_CHECK(cudaGetDevice(&device));
        ELLM_CHECK(device >= 0, "Invalid CUDA device");

        int count = 0;
        CUDA_CHECK(cudaDeviceGetAttribute(&count, cudaDevAttrMultiProcessorCount, device));
        ELLM_CHECK(count > 0, "Invalid CUDA SM count");
        return count;
    }();
    return smCount;
}

/*!
 * @brief Instantiate a CUDA graph with handling CUDA version.
 *
 * This function wraps cudaGraphInstantiate and abstracts away the API difference
 * between CUDA versions before and after 12.0. For CUDA < 12.0, it uses the legacy
 * signature with extra arguments; for CUDA >= 12.0, it uses the simplified signature.
 *
 * @param exec Pointer to the cudaGraphExec_t to be created.
 * @param graph The cudaGraph_t to instantiate.
 * @return cudaError_t indicating success or failure of the instantiation.
 */
inline cudaError_t instantiateCudaGraph(cudaGraphExec_t* exec, cudaGraph_t graph) noexcept
{
#if CUDA_VERSION < 12000
    return cudaGraphInstantiate(exec, graph, nullptr, nullptr, 0);
#else
    return cudaGraphInstantiate(exec, graph, 0);
#endif
}

/*!
 * @brief Detect available CUDA devices without nvidia-smi.
 *
 * Uses the CUDA runtime API only, which keeps the helper usable on platforms
 * where nvidia-smi is unavailable.
 *
 * @return Number of available CUDA devices, or 0 if CUDA runtime discovery fails.
 */
inline int32_t detectCudaDeviceCount() noexcept
{
    int deviceCount = 0;
    cudaError_t const err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess)
    {
        LOG_WARNING("Failed to query CUDA device count: %s", cudaGetErrorString(err));
        return 0;
    }
    return static_cast<int32_t>(deviceCount);
}

/*!
 * @brief Return true when @p deviceId names an available CUDA runtime device.
 */
inline bool isValidCudaDeviceId(int32_t deviceId) noexcept
{
    if (deviceId < 0)
    {
        return false;
    }

    int32_t const deviceCount = detectCudaDeviceCount();
    if (deviceId >= deviceCount)
    {
        LOG_WARNING("Invalid CUDA device id %d; detected %d CUDA devices.", deviceId, deviceCount);
        return false;
    }
    return true;
}

/*!
 * @brief Temporarily switch the active CUDA device for the current thread.
 *
 * Use this instead of a raw cudaSetDevice when scoped initialization must run on
 * a target device without leaking that thread-local device change back to the
 * caller. Construction throws through CUDA_CHECK when the current or requested
 * device cannot be selected; destruction is noexcept and intentionally ignores
 * restore failures.
 */
class CudaDeviceGuard
{
public:
    explicit CudaDeviceGuard(int32_t device)
        : mTargetDevice(static_cast<int>(device))
    {
        CUDA_CHECK(cudaGetDevice(&mPreviousDevice));
        if (mPreviousDevice != mTargetDevice)
        {
            CUDA_CHECK(cudaSetDevice(mTargetDevice));
        }
    }

    ~CudaDeviceGuard() noexcept
    {
        if (mPreviousDevice >= 0 && mPreviousDevice != mTargetDevice)
        {
            cudaError_t const error = cudaSetDevice(mPreviousDevice);
            if (error != cudaSuccess)
            {
                LOG_ERROR("Failed to restore CUDA device %d after scoped switch to device %d: %s", mPreviousDevice,
                    mTargetDevice, cudaGetErrorString(error));
            }
        }
    }

    CudaDeviceGuard(CudaDeviceGuard const&) = delete;
    CudaDeviceGuard& operator=(CudaDeviceGuard const&) = delete;

private:
    int mPreviousDevice{-1};
    int mTargetDevice{-1};
};

} // namespace trt_edgellm
