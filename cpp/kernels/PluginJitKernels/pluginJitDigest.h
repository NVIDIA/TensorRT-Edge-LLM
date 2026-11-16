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

//! 64-bit FNV-1a offset basis and prime. Seed and step of every plugin JIT hash: the compile-cache
//! and module-registry hashers and the low lane of PluginJitDigest.
inline constexpr uint64_t kPLUGIN_JIT_FNV_OFFSET_BASIS{0xCBF29CE484222325ULL};
inline constexpr uint64_t kPLUGIN_JIT_FNV_PRIME{0x100000001B3ULL};

//! 128-bit digest that binds a plugin JIT cubin to the key it was compiled for. It travels in every
//! serialized JIT blob and is re-verified before a module reaches cuModuleLoadData.
struct PluginJitDigest
{
    uint64_t lo{};
    uint64_t hi{};

    bool operator==(PluginJitDigest const& other) const noexcept
    {
        return lo == other.lo && hi == other.hi;
    }

    bool operator!=(PluginJitDigest const& other) const noexcept
    {
        return !(*this == other);
    }
};

//! Two independent 64-bit FNV-1a lanes (the FNV prime and a golden-ratio prime) fed byte by byte.
//! The lane constants are part of the serialized blob contract; changing them invalidates every
//! shipped engine.
class PluginJitDigestBuilder
{
public:
    void appendU32(uint32_t const value) noexcept
    {
        for (int32_t shift = 0; shift < 32; shift += 8)
        {
            appendByte(static_cast<uint8_t>(value >> shift));
        }
    }

    void appendBytes(uint8_t const* const data, size_t const size) noexcept
    {
        for (size_t index = 0; index < size; ++index)
        {
            appendByte(data[index]);
        }
    }

    PluginJitDigest get() const noexcept
    {
        return {mLo, mHi};
    }

private:
    void appendByte(uint8_t const value) noexcept
    {
        constexpr uint64_t kALT_PRIME{0x9E3779B185EBCA87ULL};
        mLo = (mLo ^ value) * kPLUGIN_JIT_FNV_PRIME;
        mHi = (mHi ^ value) * kALT_PRIME;
    }

    uint64_t mLo{kPLUGIN_JIT_FNV_OFFSET_BASIS};
    uint64_t mHi{0x6C62272E07BB0142ULL};
};

//! Digest of the key words followed by the cubin bytes (see pluginJitKey.h for the key contract).
template <typename Key>
PluginJitDigest computePluginJitDigest(Key const& key, void const* const cubinData, size_t const cubinSize) noexcept
{
    PluginJitDigestBuilder builder;
    for (uint32_t const word : key.keyWords())
    {
        builder.appendU32(word);
    }
    builder.appendBytes(static_cast<uint8_t const*>(cubinData), cubinSize);
    return builder.get();
}

} // namespace trt_edgellm
