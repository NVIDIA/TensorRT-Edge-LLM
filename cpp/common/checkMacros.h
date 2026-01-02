/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

#include "stringUtils.h"

namespace trt_edgellm
{

namespace check
{

/*!
 * @brief Check condition and throw exception if false
 *
 * @param condition Condition to check
 * @param errorMsg Error message to include in exception
 * @throw std::runtime_error if condition is false
 */
inline void check(bool condition, std::string errorMsg)
{
    if (!condition)
    {
        throw std::runtime_error(errorMsg);
    }
}

/*!
 * @brief Internal helper to check CUDA runtime errors
 *
 * @param result CUDA error code
 * @param func Function name string
 * @param file Source file name
 * @param line Source line number
 * @throw std::runtime_error if CUDA error occurred
 */
inline void _checkCuda(cudaError_t result, char const* const func, [[maybe_unused]] char const* const file,
    [[maybe_unused]] int const line)
{
    if (result)
    {
        throw std::runtime_error(format::fmtstr("CUDA runtime error in %s: %s", func, cudaGetErrorString(result)));
    }
}

/*!
 * @brief Internal helper to check CUDA driver API errors
 *
 * @param result CUDA driver error code
 * @param func Function name string
 * @param file Source file name
 * @param line Source line number
 * @throw std::runtime_error if CUDA driver error occurred
 */
inline void _checkCudaDriver(
    CUresult result, char const* const func, [[maybe_unused]] char const* const file, [[maybe_unused]] int const line)
{
    if (result)
    {
        char const* errorName = nullptr;
        if (cuGetErrorName(result, &errorName) != CUDA_SUCCESS)
        {
            errorName = "CUDA driver API error happened, but we failed to get error name.";
        }
        throw std::runtime_error(format::fmtstr("CUDA driver API error in %s: %s", func, errorName));
    }
}

} // namespace check

/*!
 * @brief Check CUDA runtime API calls
 *
 * Wraps CUDA runtime API calls and throws exception on error.
 * Usage: CUDA_CHECK(cudaMalloc(&ptr, size));
 */
#define CUDA_CHECK(stat)                                                                                               \
    do                                                                                                                 \
    {                                                                                                                  \
        trt_edgellm::check::_checkCuda((stat), #stat, __FILE__, __LINE__);                                             \
    } while (0)

/*!
 * @brief Check CUDA driver API calls
 *
 * Wraps CUDA driver API calls and throws exception on error.
 * Usage: CUDA_DRIVER_CHECK(cuMemAlloc(&dptr, size));
 */
#define CUDA_DRIVER_CHECK(stat)                                                                                        \
    do                                                                                                                 \
    {                                                                                                                  \
        trt_edgellm::check::_checkCudaDriver((stat), #stat, __FILE__, __LINE__);                                       \
    } while (0)

} // namespace trt_edgellm
