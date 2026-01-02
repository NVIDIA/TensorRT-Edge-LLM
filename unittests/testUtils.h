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

#include <cmath>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <functional>
#include <iomanip>
#include <iostream>
#include <ostream>
#include <random>
#include <utility>
#include <vector>

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

class KvCacheIndexer
{
public:
    KvCacheIndexer(
        int32_t const batchSize, int32_t const kvHeadNum, int32_t const kvCacheCapacity, int32_t const headSize)
    {
        mBatchSize = batchSize;
        mKvHeadNum = kvHeadNum;
        mKvCacheCapacity = kvCacheCapacity;
        mHeadSize = headSize;
    }

    int32_t indexK(int32_t const b, int32_t const hk, int32_t const cacheIdx, int32_t const d)
    {
        // Linear KVCache has layout of [B, 2, Hkv, S_capacity, D].
        return b * 2 * mKvHeadNum * mKvCacheCapacity * mHeadSize + hk * mKvCacheCapacity * mHeadSize
            + cacheIdx * mHeadSize + d;
    }

    int32_t indexV(int32_t const b, int32_t const hv, int32_t const cacheIdx, int32_t const d)
    {
        // Linear KVCache has layout of [B, 2, Hkv, S_capacity, D].
        // V cache need to offset the whole kCache buffer for the sequence.
        return b * 2 * mKvHeadNum * mKvCacheCapacity * mHeadSize + (mKvHeadNum + hv) * mKvCacheCapacity * mHeadSize
            + cacheIdx * mHeadSize + d;
    }

private:
    int32_t mBatchSize;
    int32_t mKvHeadNum;
    int32_t mKvCacheCapacity;
    int32_t mHeadSize;
};

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