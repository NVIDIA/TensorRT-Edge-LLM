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

#include <cmath>
#include <cstdint>

#if defined(__CUDACC__)
#define EDGELLM_COUNTER_RNG_HD __host__ __device__
#else
#define EDGELLM_COUNTER_RNG_HD
#endif

namespace trt_edgellm
{
namespace counter_rng
{

//! Each uniform takes 24 random bits, the width of a float mantissa.
constexpr uint64_t kUNIFORM_BIT_MASK = 0xFFFFFFULL;
constexpr float kUNIFORM_SCALE = 1.0F / 16777216.0F;
//! Bit positions of the two non-overlapping uniforms in one 64-bit draw.
constexpr uint32_t kFIRST_UNIFORM_SHIFT = 40U;
constexpr uint32_t kSECOND_UNIFORM_SHIFT = 16U;
constexpr float kTWO_PI = 6.28318530717958647692F;

//! SplitMix64 finalizer.
EDGELLM_COUNTER_RNG_HD constexpr uint64_t splitMix64(uint64_t value) noexcept
{
    value += 0x9E3779B97F4A7C15ULL;
    value = (value ^ (value >> 30U)) * 0xBF58476D1CE4E5B9ULL;
    value = (value ^ (value >> 27U)) * 0x94D049BB133111EBULL;
    return value ^ (value >> 31U);
}

//! Random bits keyed on (seed, offset, lane). Every key is evaluated independently, so no state is carried.
EDGELLM_COUNTER_RNG_HD constexpr uint64_t counterBits(uint64_t seed, uint64_t offset, uint64_t lane) noexcept
{
    return splitMix64(splitMix64(splitMix64(seed) ^ offset) ^ lane);
}

//! Uniform from the top 24 bits. Never 0; the largest value rounds to 1.
EDGELLM_COUNTER_RNG_HD constexpr float uniformFromBits(uint64_t bits) noexcept
{
    uint32_t const mantissa = static_cast<uint32_t>((bits >> kFIRST_UNIFORM_SHIFT) & kUNIFORM_BIT_MASK);
    return (static_cast<float>(mantissa) + 0.5F) * kUNIFORM_SCALE;
}

EDGELLM_COUNTER_RNG_HD constexpr float counterUniform(uint64_t seed, uint64_t offset, uint64_t lane) noexcept
{
    return uniformFromBits(counterBits(seed, offset, lane));
}

//! Standard-normal sample by Box-Muller: u1 is uniformFromBits(bits), which is never 0, so the log stays
//! finite; u2 takes the second uniform's bits.
EDGELLM_COUNTER_RNG_HD inline float normalFromBits(uint64_t bits) noexcept
{
    float const u1 = uniformFromBits(bits);
    float const u2 = static_cast<float>((bits >> kSECOND_UNIFORM_SHIFT) & kUNIFORM_BIT_MASK) * kUNIFORM_SCALE;
    return sqrtf(-2.0F * logf(u1)) * cosf(kTWO_PI * u2);
}

} // namespace counter_rng
} // namespace trt_edgellm

#undef EDGELLM_COUNTER_RNG_HD
