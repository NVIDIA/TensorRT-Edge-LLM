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
 * @brief Get CUDA compute capability version
 *
 * Returns the compute capability as an integer (e.g., 89 for SM 8.9).
 *
 * @return Compute capability version (major * 10 + minor)
 * @throws std::runtime_error If CUDA device query fails
 */
inline int getSMVersion()
{
    int device{-1};
    CUDA_CHECK(cudaGetDevice(&device));
    int sm_major = 0;
    int sm_minor = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&sm_major, cudaDevAttrComputeCapabilityMajor, device));
    CUDA_CHECK(cudaDeviceGetAttribute(&sm_minor, cudaDevAttrComputeCapabilityMinor, device));
    return sm_major * 10 + sm_minor;
}

/*!
 * @brief Multiprocessor count of the current CUDA device.
 *
 * Persistent CuTe DSL kernels use this runtime value instead of baking the
 * build machine's SM count into an AOT artifact.
 */
inline int32_t getDeviceMultiProcessorCount()
{
    static int32_t const smCount = []() {
        int device{-1};
        CUDA_CHECK(cudaGetDevice(&device));
        ELLM_CHECK(device >= 0, "Invalid CUDA device");

        int count{0};
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

} // namespace trt_edgellm
