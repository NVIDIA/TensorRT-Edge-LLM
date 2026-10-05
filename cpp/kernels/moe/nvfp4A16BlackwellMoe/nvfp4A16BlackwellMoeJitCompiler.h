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

//! NVRTC specialisation of the Thor W4A16 MoE CUDA-core kernels
//! (kernelSrcs/nvfp4A16BlackwellMoe/nvfp4A16BlackwellMoeKernels.cu): routing,
//! tile layout, gather, decode FC1/FC2 and their split-K reduces.  Every shape
//! parameter of the layer is baked into the cubin; the plugin compiles once per
//! layer at engine build and serializes the bundle, so the runtime never needs
//! NVRTC.  Mirrors nvfp4A16BlackwellGemvJitCompiler.

inline constexpr uint32_t kNVFP4_A16_BLACKWELL_MOE_LAYOUT_ABI{1U};
inline constexpr uint32_t kNVFP4_A16_BLACKWELL_MOE_SOURCE_ABI{1U};

enum class Nvfp4A16BlackwellMoeDataType : uint32_t
{
    kHALF = 0,
    kBF16 = 1,
};

struct Nvfp4A16BlackwellMoeJitKey
{
    int32_t sm{nvfp4_a16_blackwell::kTargetSm};
    uint32_t layout{kNVFP4_A16_BLACKWELL_MOE_LAYOUT_ABI};
    int32_t numExperts{};
    int32_t topK{};
    int32_t hiddenSize{};
    int32_t interSize{};
    int32_t interSizePadded{};
    int32_t fc1SplitK{1};
    int32_t fc2SplitK{1};
    int32_t fc2PrefetchSlots{0};
    Nvfp4A16BlackwellMoeDataType dataType{Nvfp4A16BlackwellMoeDataType::kHALF};
    uint32_t sourceAbi{kNVFP4_A16_BLACKWELL_MOE_SOURCE_ABI};

    auto asTuple() const noexcept
    {
        return std::tie(sm, layout, numExperts, topK, hiddenSize, interSize, interSizePadded, fc1SplitK, fc2SplitK,
            fc2PrefetchSlots, dataType, sourceAbi);
    }

    bool operator==(Nvfp4A16BlackwellMoeJitKey const& other) const noexcept
    {
        return asTuple() == other.asTuple();
    }
};

struct Nvfp4A16BlackwellMoeJitDigest
{
    uint64_t lo{};
    uint64_t hi{};

    bool operator==(Nvfp4A16BlackwellMoeJitDigest const& other) const noexcept
    {
        return lo == other.lo && hi == other.hi;
    }
};

struct Nvfp4A16BlackwellMoeJitKernel
{
    Nvfp4A16BlackwellMoeJitKey key;
    Nvfp4A16BlackwellMoeJitDigest digest;
    std::vector<uint8_t> cubin;
};

//! Static shared memory the baked FC1 / FC2 kernels declare (must stay within
//! the 48 KB static limit; the key is rejected otherwise).
int32_t getNvfp4A16BlackwellMoeFc1SharedBytes(Nvfp4A16BlackwellMoeJitKey const& key) noexcept;
int32_t getNvfp4A16BlackwellMoeFc2SharedBytes(Nvfp4A16BlackwellMoeJitKey const& key) noexcept;

//! Why a key cannot be compiled, or nullptr when it can.
char const* describeNvfp4A16BlackwellMoeJitKeyProblem(Nvfp4A16BlackwellMoeJitKey const& key) noexcept;

bool canCompileNvfp4A16BlackwellMoeJitKernel(Nvfp4A16BlackwellMoeJitKey const& key) noexcept;

Nvfp4A16BlackwellMoeJitKernel compileNvfp4A16BlackwellMoeJitKernel(Nvfp4A16BlackwellMoeJitKey const& key);

Nvfp4A16BlackwellMoeJitDigest computeNvfp4A16BlackwellMoeJitDigest(
    Nvfp4A16BlackwellMoeJitKey const& key, void const* cubinData, size_t cubinSize);

std::vector<uint8_t> serializeNvfp4A16BlackwellMoeJitKernel(Nvfp4A16BlackwellMoeJitKernel const& kernel);

Nvfp4A16BlackwellMoeJitKernel deserializeNvfp4A16BlackwellMoeJitKernel(void const* data, size_t size);

} // namespace trt_edgellm
