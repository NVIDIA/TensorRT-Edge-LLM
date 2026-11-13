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

#include <cstddef>
#include <cstdint>

namespace trt_edgellm
{
namespace kernel
{

struct ModelOptInt4Accessor
{
    uint8_t const* weight;
    int32_t K;

    __device__ uint8_t operator()(int32_t n, int32_t k) const
    {
        uint8_t const packed = weight[static_cast<size_t>(n / 2) * K + k];
        uint8_t const nibble = (n & 1) ? static_cast<uint8_t>(packed >> 4) : static_cast<uint8_t>(packed & 0xF);
        int32_t const signedValue = nibble < 8 ? static_cast<int32_t>(nibble) : static_cast<int32_t>(nibble) - 16;
        return static_cast<uint8_t>(signedValue + 8);
    }
};

struct GptqInt4Accessor
{
    int32_t const* qweight;
    int32_t const* qzeros;
    int32_t const* activationPermutation;
    int32_t N;
    int32_t K;
    int32_t groupSize;
    int32_t zeroPointOffset;
    bool outputMajor;

    __device__ uint8_t operator()(int32_t n, int32_t k) const
    {
        int32_t const sourceK = activationPermutation == nullptr ? k : activationPermutation[k];
        int32_t const word = outputMajor ? qweight[static_cast<size_t>(n) * (K / 8) + sourceK / 8]
                                         : qweight[static_cast<size_t>(sourceK / 8) * N + n];
        int32_t value = (word >> (4 * (sourceK % 8))) & 0xF;
        if (qzeros != nullptr)
        {
            int32_t const group = k / groupSize;
            int32_t const zeroWord = outputMajor
                ? qzeros[static_cast<size_t>(n / 8) * (K / groupSize) + group]
                : qzeros[static_cast<size_t>(group) * (N / 8) + static_cast<size_t>(n / 8)];
            int32_t const zero = (zeroWord >> (4 * (n % 8))) & 0xF;
            value = max(0, min(15, value - zero - zeroPointOffset + 8));
        }
        return static_cast<uint8_t>(value);
    }
};

struct AwqInt4Accessor
{
    int32_t const* qweight;
    int32_t N;

    __device__ uint8_t operator()(int32_t n, int32_t k) const
    {
        constexpr int32_t reverseOrder[8] = {0, 4, 1, 5, 2, 6, 3, 7};
        int32_t const packedIndex = reverseOrder[n % 8];
        int32_t const word = qweight[static_cast<size_t>(k) * (N / 8) + static_cast<size_t>(n / 8)];
        return static_cast<uint8_t>((word >> (4 * packedIndex)) & 0xF);
    }
};

} // namespace kernel
} // namespace trt_edgellm
