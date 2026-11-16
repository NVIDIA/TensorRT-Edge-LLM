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

//! NVRTC specialisation of the Thor W4A16 MoE CUDA-core kernels
//! (kernelSrcs/nvfp4A16BlackwellMoe/nvfp4A16BlackwellMoeKernels.cu): routing,
//! tile layout, gather, decode FC1/FC2 and their split-K reduces.  Every shape
//! parameter of the layer is baked into the cubin; the plugin compiles once per
//! layer at engine build and serializes the bundle, so the runtime never needs
//! NVRTC.  Mirrors nvfp4A16BlackwellGemvJitCompiler; the key follows the plugin JIT
//! key contract in pluginJitKey.h.

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

    static constexpr size_t kNumWords{12};

    //! The fields in blob, digest and hash order.
    std::array<uint32_t, kNumWords> keyWords() const noexcept
    {
        return {static_cast<uint32_t>(sm), layout, static_cast<uint32_t>(numExperts), static_cast<uint32_t>(topK),
            static_cast<uint32_t>(hiddenSize), static_cast<uint32_t>(interSize), static_cast<uint32_t>(interSizePadded),
            static_cast<uint32_t>(fc1SplitK), static_cast<uint32_t>(fc2SplitK), static_cast<uint32_t>(fc2PrefetchSlots),
            static_cast<uint32_t>(dataType), sourceAbi};
    }

    static Nvfp4A16BlackwellMoeJitKey fromKeyWords(std::array<uint32_t, kNumWords> const& words) noexcept
    {
        Nvfp4A16BlackwellMoeJitKey key;
        key.sm = static_cast<int32_t>(words[0]);
        key.layout = words[1];
        key.numExperts = static_cast<int32_t>(words[2]);
        key.topK = static_cast<int32_t>(words[3]);
        key.hiddenSize = static_cast<int32_t>(words[4]);
        key.interSize = static_cast<int32_t>(words[5]);
        key.interSizePadded = static_cast<int32_t>(words[6]);
        key.fc1SplitK = static_cast<int32_t>(words[7]);
        key.fc2SplitK = static_cast<int32_t>(words[8]);
        key.fc2PrefetchSlots = static_cast<int32_t>(words[9]);
        key.dataType = static_cast<Nvfp4A16BlackwellMoeDataType>(words[10]);
        key.sourceAbi = words[11];
        return key;
    }

    bool operator==(Nvfp4A16BlackwellMoeJitKey const& other) const noexcept
    {
        return keyWords() == other.keyWords();
    }
};

using Nvfp4A16BlackwellMoeJitKernel = PluginJitKernel<Nvfp4A16BlackwellMoeJitKey>;

//! Static shared memory the baked FC1 / FC2 kernels declare (must stay within
//! the 48 KB static limit; the key is rejected otherwise).
int32_t getNvfp4A16BlackwellMoeFc1SharedBytes(Nvfp4A16BlackwellMoeJitKey const& key) noexcept;
int32_t getNvfp4A16BlackwellMoeFc2SharedBytes(Nvfp4A16BlackwellMoeJitKey const& key) noexcept;

//! Why a key cannot be compiled, or nullptr when it can.
char const* describeNvfp4A16BlackwellMoeJitKeyProblem(Nvfp4A16BlackwellMoeJitKey const& key) noexcept;

bool canCompileNvfp4A16BlackwellMoeJitKernel(Nvfp4A16BlackwellMoeJitKey const& key) noexcept;

Nvfp4A16BlackwellMoeJitKernel compileNvfp4A16BlackwellMoeJitKernel(Nvfp4A16BlackwellMoeJitKey const& key);

std::vector<uint8_t> serializeNvfp4A16BlackwellMoeJitKernel(Nvfp4A16BlackwellMoeJitKernel const& kernel);

Nvfp4A16BlackwellMoeJitKernel deserializeNvfp4A16BlackwellMoeJitKernel(void const* data, size_t size);

} // namespace trt_edgellm
