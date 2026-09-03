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

#include <cstdint>
#include <limits>

namespace trt_edgellm::nvfp4_a16_blackwell
{

inline constexpr int32_t kTargetSm{110};
inline constexpr int32_t kNTile{128};
inline constexpr int32_t kKTile{64};

constexpr bool isSupportedProblemShape(int64_t n, int64_t k) noexcept
{
    constexpr int64_t kMAX_DIMENSION{std::numeric_limits<int32_t>::max()};
    return n > 0 && n <= kMAX_DIMENSION && k > 0 && k <= kMAX_DIMENSION && n % kNTile == 0 && k % kKTile == 0;
}

constexpr bool isTmaRepresentableProblem(int64_t m, int64_t n, int64_t k) noexcept
{
    constexpr int64_t kMAX_DIMENSION{std::numeric_limits<int32_t>::max()};
    constexpr uint64_t kTMA_BYTE_STRIDE_LIMIT{uint64_t{1} << 40};
    if (m <= 0 || m > kMAX_DIMENSION || !isSupportedProblemShape(n, k))
    {
        return false;
    }

    uint64_t const mUnsigned{static_cast<uint64_t>(m)};
    uint64_t const nUnsigned{static_cast<uint64_t>(n)};
    uint64_t const kUnsigned{static_cast<uint64_t>(k)};
    uint64_t const activationElements{mUnsigned * kUnsigned};
    uint64_t const outputElements{mUnsigned * nUnsigned};
    uint64_t const weightElements{nUnsigned * kUnsigned};
    return activationElements * 2 < kTMA_BYTE_STRIDE_LIMIT && outputElements * 2 < kTMA_BYTE_STRIDE_LIMIT
        && weightElements / 2 < kTMA_BYTE_STRIDE_LIMIT && weightElements / 16 < kTMA_BYTE_STRIDE_LIMIT;
}

} // namespace trt_edgellm::nvfp4_a16_blackwell
