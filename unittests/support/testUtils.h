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

#include <cmath>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#if SUPPORTS_FP8
#include <cuda_fp8.h>
#endif
#include <cuda_runtime.h>
#include <functional>
#include <gtest/gtest.h>
#include <iomanip>
#include <iostream>
#include <ostream>
#include <random>
#include <sstream>
#include <utility>
#include <vector>

#include "common/checkMacros.h"
#include "common/tensor.h"

template <typename T>
bool isclose(T a, T b, float rtol, float atol)
{
    float af = static_cast<float>(a);
    float bf = static_cast<float>(b);
    return fabs(af - bf) <= (atol + rtol * fabs(bf));
}

template <typename T>
void uniformFloatInitialization(std::vector<T>& vec, float a = -5.f, float b = 5.f)
{
    std::random_device rd{};
    std::mt19937 gen{rd()};
    std::uniform_real_distribution d{a, b};
    for (size_t i = 0; i < vec.size(); ++i)
    {
        vec[i] = T(d(gen));
    }
}

template <typename T>
void uniformIntInitialization(std::vector<T>& vec, int low, int high)
{
    std::random_device rd{};
    std::mt19937 gen{rd()};
    std::uniform_int_distribution d{low, high};
    for (size_t i = 0; i < vec.size(); ++i)
    {
        vec[i] = T(d(gen));
    }
}

template <typename T>
typename std::enable_if<std::is_arithmetic<T>::value, std::ostream&>::type operator<<(
    std::ostream& os, std::vector<T> const& vec)
{
    os << "[";

    for (size_t i = 0; i < vec.size(); ++i)
    {
        // Print bools as 'true'/'false' instead of 1/0
        if constexpr (std::is_same_v<T, bool>)
        {
            os << std::boolalpha << vec[i];
        }
        else
        {
            os << vec[i];
        }
        if (i < vec.size() - 1)
        {
            os << ", ";
        }
    }

    os << "]";
    return os;
}

template <typename T>
static std::pair<float, float> getTolerance()
{
    if constexpr (std::is_same_v<T, float>)
    {
        return {1e-4f, 1e-6f}; // rtol, atol for FP32
    }
    else if constexpr (std::is_same_v<T, half>)
    {
        return {1e-2f, 1e-2f}; // rtol, atol for FP16
    }
    else if constexpr (std::is_same_v<T, __nv_bfloat16>)
    {
        return {0.15f, 0.02f}; // rtol, atol for BF16
    }
    else
    {
        return {1e-4f, 1e-6f}; // Default
    }
}

// A template class to hold any callable function
template <typename F>
class Defer
{
public:
    // Constructor captures the function to be deferred
    explicit Defer(F func)
        : func_(std::move(func))
    {
    }

    // Destructor executes the function when the object goes out of scope
    ~Defer()
    {
        func_();
    }

    // Disable copying to prevent the deferred function
    // from being called multiple times.
    Defer(Defer const&) = delete;
    Defer& operator=(Defer const&) = delete;

private:
    F func_; // The stored function (e.g., a lambda)
};

// Workaround for CUDA12/13 Thor re-numbering. The kernels themselves have version compatibility.
inline void applyThorSMRenumberWAR(int32_t& smVersion)
{
    if (smVersion == 110)
    {
        smVersion = 101;
    }
}

template <typename T>
void copyHostToDevice(trt_edgellm::rt::Tensor& tensor, std::vector<T> const& hostData)
{
    auto const volume = static_cast<size_t>(tensor.getShape().volume());
    ASSERT_EQ(volume, hostData.size());
    CUDA_CHECK(cudaMemcpy(tensor.rawPointer(), hostData.data(), volume * sizeof(T), cudaMemcpyHostToDevice));
}

template <typename T>
std::vector<T> copyDeviceToHost(trt_edgellm::rt::Tensor const& tensor)
{
    auto const volume = static_cast<size_t>(tensor.getShape().volume());
    std::vector<T> hostData(volume);
    CUDA_CHECK(cudaMemcpy(hostData.data(), tensor.rawPointer(), volume * sizeof(T), cudaMemcpyDeviceToHost));
    return hostData;
}

inline std::string formatTensorIndex(trt_edgellm::rt::Coords const& shape, int64_t flatIndex)
{
    std::vector<int64_t> coords(static_cast<size_t>(shape.getNumDims()));
    int64_t residual = flatIndex;

    for (int32_t dim = shape.getNumDims() - 1; dim >= 0; --dim)
    {
        int64_t const dimSize = shape[dim];
        coords[static_cast<size_t>(dim)] = residual % dimSize;
        residual /= dimSize;
    }

    std::ostringstream oss;
    oss << "[";
    for (size_t dim = 0; dim < coords.size(); ++dim)
    {
        if (dim > 0)
        {
            oss << ", ";
        }
        oss << coords[dim];
    }
    oss << "]";
    return oss.str();
}

#if SUPPORTS_FP8
// Per-tensor quantize FP16 → FP8 E4M3: fp8 ≈ fp16 / scale (saturating cast).
inline std::vector<__nv_fp8_e4m3> quantizeHalfToFp8(std::vector<half> const& src, float scale)
{
    std::vector<__nv_fp8_e4m3> dst(src.size());
    for (size_t i = 0; i < src.size(); ++i)
    {
        dst[i] = static_cast<__nv_fp8_e4m3>(__half2float(src[i]) / scale);
    }
    return dst;
}

// Per-tensor dequantize FP8 → FP16: fp16 = fp8 * scale.
inline std::vector<half> dequantizeFp8ToHalf(std::vector<__nv_fp8_e4m3> const& src, float scale)
{
    std::vector<half> dst(src.size());
    for (size_t i = 0; i < src.size(); ++i)
    {
        dst[i] = __float2half(static_cast<float>(src[i]) * scale);
    }
    return dst;
}
#endif // SUPPORTS_FP8
