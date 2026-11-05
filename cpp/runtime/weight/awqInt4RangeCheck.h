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
#include <cstring>

namespace trt_edgellm
{
namespace rt
{

//! qweight and qzeros contain K and K/groupSize rows of N/8 packed words.
inline bool awqInt4ValuesFit(
    uint8_t const* qweight, uint8_t const* qzeros, int32_t K, int32_t packedN, int32_t groupSize)
{
    for (int32_t k = 0; k < K; ++k)
    {
        size_t const weightRow = static_cast<size_t>(k) * packedN;
        size_t const zeroRow = static_cast<size_t>(k / groupSize) * packedN;
        for (int32_t column = 0; column < packedN; ++column)
        {
            uint32_t weightWord;
            uint32_t zeroWord;
            std::memcpy(&weightWord, qweight + (weightRow + column) * sizeof(weightWord), sizeof(weightWord));
            std::memcpy(&zeroWord, qzeros + (zeroRow + column) * sizeof(zeroWord), sizeof(zeroWord));
            for (int32_t nibble = 0; nibble < 8; ++nibble)
            {
                int32_t const value = static_cast<int32_t>((weightWord >> (4 * nibble)) & 0xF);
                int32_t const zero = static_cast<int32_t>((zeroWord >> (4 * nibble)) & 0xF);
                if (value - zero < -8 || value - zero > 7)
                {
                    return false;
                }
            }
        }
    }
    return true;
}

} // namespace rt
} // namespace trt_edgellm
