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

#include "pluginJitDigest.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace trt_edgellm
{

//! Contract of a plugin JIT key type `Key`, shared by the compile cache, the serialized blob, the
//! digest and the module registry:
//!
//!     static constexpr size_t kNumWords;
//!     std::array<uint32_t, kNumWords> keyWords() const noexcept;                      // every field, fixed order
//!     static Key fromKeyWords(std::array<uint32_t, kNumWords> const& words) noexcept;   // inverse of keyWords
//!     bool operator==(Key const& other) const noexcept;
//!
//! Hashing, serialization and the digest are all derived from the words, so a key struct declares
//! its fields once and cannot drift from its serialized form. Adding a JIT plugin takes the key
//! struct, an NVRTC option builder (see getOrCompilePluginJitKernel) and a bindFunctions callback
//! (see PluginJitModuleRegistry).

//! One FNV-1a step: xor-then-multiply, so no field can be shifted out of range however many follow.
inline size_t mixPluginJitHash(size_t const hash, size_t const value) noexcept
{
    return (hash ^ value) * static_cast<size_t>(kPLUGIN_JIT_FNV_PRIME);
}

template <size_t N>
size_t hashPluginJitWords(
    std::array<uint32_t, N> const& words, size_t hash = static_cast<size_t>(kPLUGIN_JIT_FNV_OFFSET_BASIS)) noexcept
{
    for (uint32_t const word : words)
    {
        hash = mixPluginJitHash(hash, word);
    }
    return hash;
}

//! std::hash-style functor over the key words.
template <typename Key>
struct PluginJitKeyHasher
{
    size_t operator()(Key const& key) const noexcept
    {
        return hashPluginJitWords(key.keyWords());
    }
};

//! A compiled plugin JIT module: the key it was compiled for, the native cubin and the digest
//! binding the two.
template <typename Key>
struct PluginJitKernel
{
    Key key{};
    std::vector<uint8_t> cubin;
    PluginJitDigest digest{};
};

} // namespace trt_edgellm
