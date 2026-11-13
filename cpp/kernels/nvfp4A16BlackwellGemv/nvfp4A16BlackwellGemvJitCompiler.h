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

#include "kernels/PluginJitKernels/pluginJitKey.h"
#include "kernels/nvfp4A16BlackwellSupport.h"

#include <array>
#include <cstddef>
#include <cstdint>
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

//! Plugin JIT key (see pluginJitKey.h for the contract shared with the blob, digest and registry).
struct Nvfp4A16BlackwellGemvJitKey
{
    int32_t sm{nvfp4_a16_blackwell::kTargetSm};
    uint32_t layout{kNVFP4_A16_BLACKWELL_GEMV_LAYOUT_ABI};
    int32_t n{};
    int32_t k{};
    Nvfp4A16BlackwellGemvDataType dataType{Nvfp4A16BlackwellGemvDataType::kHALF};
    uint32_t sourceAbi{kNVFP4_A16_BLACKWELL_GEMV_SOURCE_ABI};

    static constexpr size_t kNumWords{6};

    //! The fields in blob, digest and hash order.
    std::array<uint32_t, kNumWords> keyWords() const noexcept
    {
        return {static_cast<uint32_t>(sm), layout, static_cast<uint32_t>(n), static_cast<uint32_t>(k),
            static_cast<uint32_t>(dataType), sourceAbi};
    }

    static Nvfp4A16BlackwellGemvJitKey fromKeyWords(std::array<uint32_t, kNumWords> const& words) noexcept
    {
        Nvfp4A16BlackwellGemvJitKey key;
        key.sm = static_cast<int32_t>(words[0]);
        key.layout = words[1];
        key.n = static_cast<int32_t>(words[2]);
        key.k = static_cast<int32_t>(words[3]);
        key.dataType = static_cast<Nvfp4A16BlackwellGemvDataType>(words[4]);
        key.sourceAbi = words[5];
        return key;
    }

    bool operator==(Nvfp4A16BlackwellGemvJitKey const& other) const noexcept
    {
        return keyWords() == other.keyWords();
    }
};

using Nvfp4A16BlackwellGemvJitKernel = PluginJitKernel<Nvfp4A16BlackwellGemvJitKey>;

bool canCompileNvfp4A16BlackwellGemvJitKernel(Nvfp4A16BlackwellGemvJitKey const& key) noexcept;

Nvfp4A16BlackwellGemvJitKernel compileNvfp4A16BlackwellGemvJitKernel(Nvfp4A16BlackwellGemvJitKey const& key);

std::vector<uint8_t> serializeNvfp4A16BlackwellGemvJitKernel(Nvfp4A16BlackwellGemvJitKernel const& kernel);

Nvfp4A16BlackwellGemvJitKernel deserializeNvfp4A16BlackwellGemvJitKernel(void const* data, size_t size);

} // namespace trt_edgellm
