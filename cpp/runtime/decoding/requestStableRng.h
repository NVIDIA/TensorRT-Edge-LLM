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

#include "common/counterRng.h"

#include <cstdint>

#if defined(__CUDACC__)
#define EDGELLM_RNG_HD __host__ __device__
#else
#define EDGELLM_RNG_HD
#endif

namespace trt_edgellm
{
namespace rt
{

enum class SpecRandomPurpose : uint64_t
{
    kTarget = 0xD6E8FEB86659FD93ULL,
    kProposal = 0xA0761D6478BD642FULL,
    kAccept = 0xE7037ED1A0B428DBULL,
    kResidual = 0x8EBC6AF09C88C6E3ULL,
    kBonus = 0x589965CC75374CC3ULL,
};

EDGELLM_RNG_HD constexpr uint64_t requestStableNextAbsolutePosition(
    uint64_t fullPromptTokenCount, uint64_t generatedTokenCount) noexcept
{
    return fullPromptTokenCount + generatedTokenCount;
}

EDGELLM_RNG_HD constexpr uint64_t requestStableRandomBits(
    uint64_t requestSeed, uint64_t absolutePosition, SpecRandomPurpose purpose, uint64_t lane) noexcept
{
    uint64_t value = counter_rng::splitMix64(requestSeed);
    value ^= counter_rng::splitMix64(absolutePosition + 0xD1B54A32D192ED03ULL);
    value ^= counter_rng::splitMix64(static_cast<uint64_t>(purpose));
    value ^= counter_rng::splitMix64(lane + 0xDB4F0B9175AE2165ULL);
    return counter_rng::splitMix64(value);
}

EDGELLM_RNG_HD constexpr float requestStableUniform(
    uint64_t requestSeed, uint64_t absolutePosition, SpecRandomPurpose purpose, uint64_t lane) noexcept
{
    return counter_rng::uniformFromBits(requestStableRandomBits(requestSeed, absolutePosition, purpose, lane));
}

} // namespace rt
} // namespace trt_edgellm

#undef EDGELLM_RNG_HD
