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
#include "nvfp4A16BlackwellMoeJitCompiler.h"

#include "common/stringUtils.h"
#include "kernels/PluginJitKernels/pluginJitBlob.h"
#include "kernels/PluginJitKernels/pluginJitCompiler.h"

#include <stdexcept>
#include <string>
#include <vector>

namespace trt_edgellm
{
namespace
{

constexpr PluginJitBlobFormat kJIT_BLOB_FORMAT{"NVFP4-A16 Blackwell MoE JIT", 0x4D363134U /* "416M" */};
constexpr int32_t kMAX_STATIC_SMEM_BYTES{48 * 1024};
constexpr int32_t kMAX_SPLIT_K{8};
constexpr int32_t kMAX_EXPERTS{512};
constexpr int32_t kMAX_TOP_K{32};
constexpr int32_t kN_TILE{128};
constexpr int32_t kK_TILE{64};
constexpr int32_t kSMEM_BYTES_PER_STAGED_TILE{(kK_TILE + 4) * 4};
constexpr int32_t kCODE_BYTES_PER_TILE{kN_TILE * kK_TILE / 2};
constexpr int32_t kSCALE_BYTES_PER_TILE{kN_TILE * kK_TILE / 16};

int32_t tilesPerSplit(int32_t const kBlocks, int32_t const splitK) noexcept
{
    return splitK > 0 ? (kBlocks + splitK - 1) / splitK : 0;
}

void validateKey(Nvfp4A16BlackwellMoeJitKey const& key)
{
    if (char const* problem = describeNvfp4A16BlackwellMoeJitKeyProblem(key); problem != nullptr)
    {
        throw std::invalid_argument(std::string("NVFP4-A16 Blackwell MoE JIT: ") + problem);
    }
}

std::string keyToString(Nvfp4A16BlackwellMoeJitKey const& key)
{
    return format::fmtstr(
        "SM%d, layout=%u, E=%d, topK=%d, H=%d, I=%d, I_pad=%d, fc1SplitK=%d, fc2SplitK=%d, "
        "fc2PrefetchSlots=%d, dtype=%u, source ABI=%u",
        key.sm, key.layout, key.numExperts, key.topK, key.hiddenSize, key.interSize, key.interSizePadded, key.fc1SplitK,
        key.fc2SplitK, key.fc2PrefetchSlots, static_cast<uint32_t>(key.dataType), key.sourceAbi);
}

std::vector<std::string> buildNvrtcOptions(Nvfp4A16BlackwellMoeJitKey const& key)
{
    return {"--std=c++17", "--use_fast_math", "--device-as-default-execution-space", "--gpu-architecture=sm_110a",
        "-DNDEBUG", "-DMOE_NUM_EXPERTS=" + std::to_string(key.numExperts), "-DMOE_TOP_K=" + std::to_string(key.topK),
        "-DMOE_HIDDEN=" + std::to_string(key.hiddenSize), "-DMOE_INTER=" + std::to_string(key.interSize),
        "-DMOE_INTER_PAD=" + std::to_string(key.interSizePadded), "-DMOE_FC1_SPLIT_K=" + std::to_string(key.fc1SplitK),
        "-DMOE_FC2_SPLIT_K=" + std::to_string(key.fc2SplitK),
        "-DMOE_FC2_PREFETCH_SLOTS=" + std::to_string(key.fc2PrefetchSlots),
        "-DMOE_DATA_TYPE=" + std::to_string(static_cast<uint32_t>(key.dataType)),
        "-DMOE_LAYOUT_ABI=" + std::to_string(key.layout), "-DMOE_SOURCE_ABI=" + std::to_string(key.sourceAbi)};
}

} // namespace

int32_t getNvfp4A16BlackwellMoeFc1SharedBytes(Nvfp4A16BlackwellMoeJitKey const& key) noexcept
{
    return tilesPerSplit(key.hiddenSize / kK_TILE, key.fc1SplitK) * kSMEM_BYTES_PER_STAGED_TILE;
}

int32_t getNvfp4A16BlackwellMoeFc2SharedBytes(Nvfp4A16BlackwellMoeJitKey const& key) noexcept
{
    int32_t const tiles = tilesPerSplit(key.interSize / kK_TILE, key.fc2SplitK);
    return key.topK * tiles * kSMEM_BYTES_PER_STAGED_TILE
        + key.fc2PrefetchSlots * tiles * (kCODE_BYTES_PER_TILE + kSCALE_BYTES_PER_TILE) + 32;
}

char const* describeNvfp4A16BlackwellMoeJitKeyProblem(Nvfp4A16BlackwellMoeJitKey const& key) noexcept
{
    if (key.sm != nvfp4_a16_blackwell::kTargetSm)
    {
        return "requires SM110";
    }
    if (key.layout != kNVFP4_A16_BLACKWELL_MOE_LAYOUT_ABI)
    {
        return "unsupported layout ABI";
    }
    if (key.sourceAbi != kNVFP4_A16_BLACKWELL_MOE_SOURCE_ABI)
    {
        return "unsupported source ABI";
    }
    if (key.dataType != Nvfp4A16BlackwellMoeDataType::kHALF && key.dataType != Nvfp4A16BlackwellMoeDataType::kBF16)
    {
        return "unsupported data type";
    }
    if (key.numExperts <= 0 || key.numExperts > kMAX_EXPERTS)
    {
        return "numExperts must be in [1, 512]";
    }
    if (key.topK <= 0 || key.topK > kMAX_TOP_K || key.topK > key.numExperts)
    {
        return "topK must be in [1, 32] and <= numExperts";
    }
    if (key.hiddenSize <= 0 || key.hiddenSize % kN_TILE != 0)
    {
        return "hiddenSize must be a positive multiple of 128";
    }
    if (key.interSize <= 0 || key.interSize % kK_TILE != 0)
    {
        return "interSize must be a positive multiple of 64";
    }
    if (key.interSizePadded < key.interSize || key.interSizePadded % kN_TILE != 0
        || key.interSizePadded - key.interSize >= kN_TILE)
    {
        return "interSizePadded must be interSize rounded up to a multiple of 128";
    }
    if (key.fc1SplitK < 1 || key.fc1SplitK > kMAX_SPLIT_K || key.fc1SplitK > key.hiddenSize / kK_TILE)
    {
        return "fc1SplitK out of range";
    }
    if (key.fc2SplitK < 1 || key.fc2SplitK > kMAX_SPLIT_K || key.fc2SplitK > key.interSize / kK_TILE)
    {
        return "fc2SplitK out of range";
    }
    if (key.fc2PrefetchSlots < 0 || key.fc2PrefetchSlots > key.topK)
    {
        return "fc2PrefetchSlots must be in [0, topK]";
    }
    if (getNvfp4A16BlackwellMoeFc1SharedBytes(key) > kMAX_STATIC_SMEM_BYTES
        || getNvfp4A16BlackwellMoeFc2SharedBytes(key) > kMAX_STATIC_SMEM_BYTES)
    {
        return "FC1/FC2 shared-memory staging exceeds the 48 KB static limit (raise split-K or drop prefetch slots)";
    }
    return nullptr;
}

bool canCompileNvfp4A16BlackwellMoeJitKernel(Nvfp4A16BlackwellMoeJitKey const& key) noexcept
{
    return describeNvfp4A16BlackwellMoeJitKeyProblem(key) == nullptr;
}

Nvfp4A16BlackwellMoeJitKernel compileNvfp4A16BlackwellMoeJitKernel(Nvfp4A16BlackwellMoeJitKey const& key)
{
    validateKey(key);
    return getOrCompilePluginJitKernel(PluginJitProgram::kNVFP4_A16_BLACKWELL_MOE, key, buildNvrtcOptions(key),
        "NVFP4-A16 Blackwell MoE kernels for " + keyToString(key));
}

std::vector<uint8_t> serializeNvfp4A16BlackwellMoeJitKernel(Nvfp4A16BlackwellMoeJitKernel const& kernel)
{
    validateKey(kernel.key);
    return serializePluginJitKernel(kJIT_BLOB_FORMAT, kernel);
}

Nvfp4A16BlackwellMoeJitKernel deserializeNvfp4A16BlackwellMoeJitKernel(void const* const data, size_t const size)
{
    Nvfp4A16BlackwellMoeJitKernel kernel
        = deserializePluginJitKernel<Nvfp4A16BlackwellMoeJitKey>(kJIT_BLOB_FORMAT, data, size);
    validateKey(kernel.key);
    return kernel;
}

} // namespace trt_edgellm
