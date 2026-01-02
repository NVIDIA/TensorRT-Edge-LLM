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

#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace trt_edgellm
{
namespace kernel
{

/*!
 * @brief Data vectorization helper for efficient global memory access
 *
 * Provides efficient vectorized memory load/store operations for CUDA kernels.
 * Specialized for float and half types.
 *
 * @tparam T Element type
 */
template <typename T>
struct DVec
{
    static constexpr uint32_t vec_size = 0; //!< Number of elements in vector

    //! @brief Access element at index
    //! @param idx Element index
    //! @return Reference to element
    inline T& operator[](uint32_t idx);

    //! @brief Access element at index (const)
    //! @param idx Element index
    //! @return Const reference to element
    inline T const& operator[](uint32_t idx) const;

    //! @brief Load vector from global memory
    //! @param ptr Source pointer
    inline void load(T const* ptr);

    //! @brief Store vector to global memory
    //! @param ptr Destination pointer
    inline void store(T* ptr) const;
};

//! \cond INTERNAL
/*!
 * @brief Vectorization specialization for float[8]
 *
 * Optimized for loading cos/sin cache and other float data.
 * Uses two float4 loads for 8 floats (32 bytes total).
 */
template <>
struct DVec<float>
{
    float4 data[2];                         //!< Storage for 8 floats using two float4
    static constexpr uint32_t vec_size = 8; //!< Vector size

    //! @brief Access element at index
    //! @param idx Element index (0-7)
    //! @return Reference to float element
    __device__ __forceinline__ float& operator[](uint32_t idx)
    {
        return ((float*) (data))[idx];
    }

    //! @brief Access element at index (const)
    //! @param idx Element index (0-7)
    //! @return Const reference to float element
    __device__ __forceinline__ float const& operator[](uint32_t idx) const
    {
        return ((float const*) (data))[idx];
    }

    //! @brief Load 8 floats from global memory
    //! @param ptr Source pointer (must be aligned)
    __device__ __forceinline__ void load(float const* ptr)
    {
        data[0] = *(reinterpret_cast<float4 const*>(ptr));
        data[1] = *(reinterpret_cast<float4 const*>(ptr + 4));
    }

    //! @brief Store 8 floats to global memory
    //! @param ptr Destination pointer (must be aligned)
    __device__ __forceinline__ void store(float* ptr) const
    {
        *(reinterpret_cast<float4*>(ptr)) = data[0];
        *(reinterpret_cast<float4*>(ptr + 4)) = data[1];
    }
};
//! \endcond

//! \cond INTERNAL
/*!
 * @brief Vectorization specialization for half[8]
 *
 * Most commonly used vectorization for half precision data.
 * Uses uint4 (16 bytes) for optimal memory coalescing.
 */
template <>
struct DVec<half>
{
    uint4 data;                             //!< Storage for 8 halves as uint4 (16 bytes)
    static constexpr uint32_t vec_size = 8; //!< Vector size

    //! @brief Access element at index
    //! @param idx Element index (0-7)
    //! @return Reference to half element
    __device__ __forceinline__ half& operator[](uint32_t idx)
    {
        return reinterpret_cast<half*>(&data)[idx];
    }

    //! @brief Access element at index (const)
    //! @param idx Element index (0-7)
    //! @return Const reference to half element
    __device__ __forceinline__ half const& operator[](uint32_t idx) const
    {
        return reinterpret_cast<half const*>(&data)[idx];
    }

    //! @brief Load 8 halves from global memory
    //! @param ptr Source pointer (must be 16-byte aligned)
    __device__ __forceinline__ void load(half const* ptr)
    {
        data = *(reinterpret_cast<uint4 const*>(ptr));
    }

    //! @brief Store 8 halves to global memory
    //! @param ptr Destination pointer (must be 16-byte aligned)
    __device__ __forceinline__ void store(half* ptr) const
    {
        *(reinterpret_cast<uint4*>(ptr)) = data;
    }
};
//! \endcond

} // namespace kernel
} // namespace trt_edgellm
