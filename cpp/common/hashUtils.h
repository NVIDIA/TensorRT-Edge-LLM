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

#include <functional>

namespace trt_edgellm
{
namespace hash_utils
{
/*!
 * @brief Combine hash values using boost-style hash combination
 *
 * Combines a hash seed with the hash of a new value using the boost hash_combine
 * algorithm. Useful for creating composite hash values.
 *
 * @tparam T Type of value to hash
 * @param seed Hash seed to combine with (modified in-place)
 * @param value Value to hash and combine
 */
template <typename T>
inline void hashCombine(size_t& seed, T const& value)
{
    constexpr size_t kDELTA = 0x9e3779b9;
    seed ^= std::hash<T>()(value) + kDELTA + (seed << 6) + (seed >> 2);
}
} // namespace hash_utils
} // namespace trt_edgellm
