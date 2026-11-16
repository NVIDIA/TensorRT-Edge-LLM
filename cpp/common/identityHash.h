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
#include <cuda_runtime.h>
#include <functional>
#include <string_view>

namespace trt_edgellm
{
namespace rt
{

//! Two-word digest used as a stable identity for token blocks, media content and adapters.
struct Hash128
{
    uint64_t hi{};
    uint64_t lo{};
};

inline bool operator==(Hash128 const& lhs, Hash128 const& rhs) noexcept
{
    return lhs.hi == rhs.hi && lhs.lo == rhs.lo;
}

inline bool operator!=(Hash128 const& lhs, Hash128 const& rhs) noexcept
{
    return !(lhs == rhs);
}

//! Standard 128-bit FNV-1a offset basis and prime (2^88 + 2^8 + 0x3B), as (hi, lo) uint64 halves.
inline constexpr uint64_t kFNV128_OFFSET_HI = 0x6C62272E07BB0142ULL;
inline constexpr uint64_t kFNV128_OFFSET_LO = 0x62B821756295C58DULL;
inline constexpr uint64_t kFNV128_PRIME_HI = 0x0000000001000000ULL;
inline constexpr uint64_t kFNV128_PRIME_LO = 0x000000000000013BULL;

//! Running state of a host-side 128-bit FNV-1a computation; finishHash() turns it into a Hash128.
struct FnvValue
{
    uint64_t hi;
    uint64_t lo;
};

inline constexpr FnvValue kFNV_OFFSET{kFNV128_OFFSET_HI, kFNV128_OFFSET_LO};

//! Fold one byte into the state. The two-word unsigned arithmetic supplies the modulo 2^128 operation.
inline void fnvByte(FnvValue& state, uint8_t value) noexcept
{
    state.lo ^= static_cast<uint64_t>(value);

    // FNV_prime is 2^88 + 0x13B. Split the low word into 32-bit limbs to compute the carry portably.
    constexpr uint64_t kFNV_LOW_FACTOR = 0x13B;
    uint64_t const lowLow = static_cast<uint32_t>(state.lo) * kFNV_LOW_FACTOR;
    uint64_t const lowHigh = (state.lo >> 32U) * kFNV_LOW_FACTOR;
    uint64_t const productLow = lowLow + (lowHigh << 32U);
    uint64_t const carry = (lowHigh >> 32U) + static_cast<uint64_t>(productLow < lowLow);
    state.hi = state.hi * kFNV_LOW_FACTOR + carry + (state.lo << 24U);
    state.lo = productLow;
}

//! Fold a 64-bit integer least-significant byte first, so the digest does not depend on host byte order.
inline void fnvU64(FnvValue& state, uint64_t value) noexcept
{
    constexpr size_t kBYTE_COUNT = sizeof(value);
    constexpr size_t kBITS_PER_BYTE = 8U;
    for (size_t index = 0; index < kBYTE_COUNT; ++index)
    {
        fnvByte(state, static_cast<uint8_t>(value >> (kBITS_PER_BYTE * index)));
    }
}

inline Hash128 finishHash(FnvValue state) noexcept
{
    return Hash128{state.hi, state.lo};
}

//! Deterministic 128-bit FNV-1a digest of exact opaque bytes, used as the identity of media content and adapters.
//!
//! When @p stream is non-null and @p cpuOnly is false, payloads above an internal threshold are hashed on the GPU
//! using a parallel chunked kernel. The result is bit-identical to the CPU path. Callers on hot paths with large
//! payloads (raw audio, for instance) should pass a valid stream and set cpuOnly=false.
Hash128 hashOpaqueIdentity(std::string_view bytes, cudaStream_t stream = nullptr, bool cpuOnly = true);

} // namespace rt
} // namespace trt_edgellm

namespace std
{

template <>
struct hash<trt_edgellm::rt::Hash128>
{
    size_t operator()(trt_edgellm::rt::Hash128 const& value) const noexcept
    {
        constexpr uint64_t kFOLD_MULTIPLIER = 0x9E3779B97F4A7C15ULL;
        return static_cast<size_t>(value.hi ^ (value.lo * kFOLD_MULTIPLIER));
    }
};

} // namespace std
