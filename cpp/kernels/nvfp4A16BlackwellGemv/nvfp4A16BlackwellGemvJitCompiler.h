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

#include "kernels/nvfp4A16BlackwellSupport.h"

#include <cstddef>
#include <cstdint>
#include <tuple>
#include <vector>

namespace trt_edgellm
{

inline constexpr uint32_t kNVFP4_A16_BLACKWELL_GEMV_LAYOUT_ABI{1U};
inline constexpr uint32_t kNVFP4_A16_BLACKWELL_GEMV_SOURCE_ABI{4U};

enum class Nvfp4A16BlackwellGemvDataType : uint32_t
{
    kHALF = 0,
    kBF16 = 1,
};

struct Nvfp4A16BlackwellGemvJitKey
{
    int32_t sm{nvfp4_a16_blackwell::kTargetSm};
    uint32_t layout{kNVFP4_A16_BLACKWELL_GEMV_LAYOUT_ABI};
    int32_t n{};
    int32_t k{};
    Nvfp4A16BlackwellGemvDataType dataType{Nvfp4A16BlackwellGemvDataType::kHALF};
    uint32_t sourceAbi{kNVFP4_A16_BLACKWELL_GEMV_SOURCE_ABI};

    auto asTuple() const noexcept
    {
        return std::tie(sm, layout, n, k, dataType, sourceAbi);
    }

    bool operator==(Nvfp4A16BlackwellGemvJitKey const& other) const noexcept
    {
        return asTuple() == other.asTuple();
    }
};

struct Nvfp4A16BlackwellGemvJitDigest
{
    uint64_t lo{};
    uint64_t hi{};

    bool operator==(Nvfp4A16BlackwellGemvJitDigest const& other) const noexcept
    {
        return lo == other.lo && hi == other.hi;
    }
};

struct Nvfp4A16BlackwellGemvJitKernel
{
    Nvfp4A16BlackwellGemvJitKey key;
    Nvfp4A16BlackwellGemvJitDigest digest;
    std::vector<uint8_t> cubin;
};

bool canCompileNvfp4A16BlackwellGemvJitKernel(Nvfp4A16BlackwellGemvJitKey const& key) noexcept;

Nvfp4A16BlackwellGemvJitKernel compileNvfp4A16BlackwellGemvJitKernel(Nvfp4A16BlackwellGemvJitKey const& key);

Nvfp4A16BlackwellGemvJitDigest computeNvfp4A16BlackwellGemvJitDigest(
    Nvfp4A16BlackwellGemvJitKey const& key, void const* cubinData, size_t cubinSize);

std::vector<uint8_t> serializeNvfp4A16BlackwellGemvJitKernel(Nvfp4A16BlackwellGemvJitKernel const& kernel);

Nvfp4A16BlackwellGemvJitKernel deserializeNvfp4A16BlackwellGemvJitKernel(void const* data, size_t size);

} // namespace trt_edgellm
