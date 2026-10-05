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
#include "kernels/PluginJitKernels/pluginJitCompileCache.h"
#include "kernels/PluginJitKernels/pluginJitCompiler.h"

#include <array>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace trt_edgellm
{
namespace
{

constexpr uint32_t kJIT_BLOB_MAGIC{0x454F4D4EU}; // "NMOE"
constexpr uint32_t kJIT_BLOB_VERSION{1U};
constexpr int32_t kMAX_STATIC_SMEM_BYTES{48 * 1024};
constexpr int32_t kMAX_SPLIT_K{8};
constexpr int32_t kMAX_EXPERTS{512};
constexpr int32_t kMAX_TOP_K{32};
constexpr int32_t kN_TILE{128};
constexpr int32_t kK_TILE{64};
constexpr int32_t kSMEM_BYTES_PER_STAGED_TILE{(kK_TILE + 4) * 4};
constexpr int32_t kCODE_BYTES_PER_TILE{kN_TILE * kK_TILE / 2};
constexpr int32_t kSCALE_BYTES_PER_TILE{kN_TILE * kK_TILE / 16};

struct Nvfp4A16BlackwellMoeJitKeyHasher
{
    size_t operator()(Nvfp4A16BlackwellMoeJitKey const& key) const noexcept
    {
        auto mix = [](size_t hash, size_t value) noexcept {
            constexpr size_t kPRIME{0x100000001B3ULL};
            return (hash ^ value) * kPRIME;
        };
        size_t hash{0xCBF29CE484222325ULL};
        hash = mix(hash, static_cast<size_t>(key.sm));
        hash = mix(hash, static_cast<size_t>(key.layout));
        hash = mix(hash, static_cast<size_t>(key.numExperts));
        hash = mix(hash, static_cast<size_t>(key.topK));
        hash = mix(hash, static_cast<size_t>(key.hiddenSize));
        hash = mix(hash, static_cast<size_t>(key.interSize));
        hash = mix(hash, static_cast<size_t>(key.interSizePadded));
        hash = mix(hash, static_cast<size_t>(key.fc1SplitK));
        hash = mix(hash, static_cast<size_t>(key.fc2SplitK));
        hash = mix(hash, static_cast<size_t>(key.fc2PrefetchSlots));
        hash = mix(hash, static_cast<size_t>(static_cast<uint32_t>(key.dataType)));
        return mix(hash, static_cast<size_t>(key.sourceAbi));
    }
};

bool isNativeCubin(void const* const data, size_t const size) noexcept
{
    constexpr std::array<uint8_t, 4> kELF_MAGIC{0x7FU, 0x45U, 0x4CU, 0x46U};
    if (data == nullptr || size < kELF_MAGIC.size())
    {
        return false;
    }
    auto const* const bytes = static_cast<uint8_t const*>(data);
    return bytes[0] == kELF_MAGIC[0] && bytes[1] == kELF_MAGIC[1] && bytes[2] == kELF_MAGIC[2]
        && bytes[3] == kELF_MAGIC[3];
}

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

void appendU32(std::vector<uint8_t>& output, uint32_t const value)
{
    for (int32_t shift = 0; shift < 32; shift += 8)
    {
        output.push_back(static_cast<uint8_t>((value >> shift) & 0xFFU));
    }
}

void appendU64(std::vector<uint8_t>& output, uint64_t const value)
{
    for (int32_t shift = 0; shift < 64; shift += 8)
    {
        output.push_back(static_cast<uint8_t>((value >> shift) & 0xFFU));
    }
}

uint32_t readU32(uint8_t const*& cursor, size_t& remaining)
{
    if (remaining < sizeof(uint32_t))
    {
        throw std::runtime_error("Truncated NVFP4-A16 Blackwell MoE JIT blob");
    }
    uint32_t value{0};
    for (int32_t shift = 0; shift < 32; shift += 8)
    {
        value |= static_cast<uint32_t>(*cursor++) << shift;
    }
    remaining -= sizeof(uint32_t);
    return value;
}

uint64_t readU64(uint8_t const*& cursor, size_t& remaining)
{
    if (remaining < sizeof(uint64_t))
    {
        throw std::runtime_error("Truncated NVFP4-A16 Blackwell MoE JIT blob");
    }
    uint64_t value{0};
    for (int32_t shift = 0; shift < 64; shift += 8)
    {
        value |= static_cast<uint64_t>(*cursor++) << shift;
    }
    remaining -= sizeof(uint64_t);
    return value;
}

class DigestBuilder
{
public:
    void appendU32(uint32_t const value) noexcept
    {
        for (int32_t shift = 0; shift < 32; shift += 8)
        {
            appendByte(static_cast<uint8_t>(value >> shift));
        }
    }

    void appendBytes(uint8_t const* data, size_t const size) noexcept
    {
        for (size_t index = 0; index < size; ++index)
        {
            appendByte(data[index]);
        }
    }

    Nvfp4A16BlackwellMoeJitDigest get() const noexcept
    {
        return {mLo, mHi};
    }

private:
    void appendByte(uint8_t const value) noexcept
    {
        constexpr uint64_t kFNV_PRIME{0x100000001B3ULL};
        constexpr uint64_t kALT_PRIME{0x9E3779B185EBCA87ULL};
        mLo = (mLo ^ value) * kFNV_PRIME;
        mHi = (mHi ^ value) * kALT_PRIME;
    }

    uint64_t mLo{0xCBF29CE484222325ULL};
    uint64_t mHi{0x6C62272E07BB0142ULL};
};

//! The key fields in the order they enter the blob and the digest.
std::array<uint32_t, 12> keyWords(Nvfp4A16BlackwellMoeJitKey const& key) noexcept
{
    return {static_cast<uint32_t>(key.sm), key.layout, static_cast<uint32_t>(key.numExperts),
        static_cast<uint32_t>(key.topK), static_cast<uint32_t>(key.hiddenSize), static_cast<uint32_t>(key.interSize),
        static_cast<uint32_t>(key.interSizePadded), static_cast<uint32_t>(key.fc1SplitK),
        static_cast<uint32_t>(key.fc2SplitK), static_cast<uint32_t>(key.fc2PrefetchSlots),
        static_cast<uint32_t>(key.dataType), key.sourceAbi};
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

Nvfp4A16BlackwellMoeJitDigest computeNvfp4A16BlackwellMoeJitDigest(
    Nvfp4A16BlackwellMoeJitKey const& key, void const* const cubinData, size_t const cubinSize)
{
    validateKey(key);
    if (!isNativeCubin(cubinData, cubinSize))
    {
        throw std::invalid_argument("NVFP4-A16 Blackwell MoE JIT payload must be a native ELF cubin");
    }
    DigestBuilder digest;
    for (uint32_t const word : keyWords(key))
    {
        digest.appendU32(word);
    }
    digest.appendBytes(static_cast<uint8_t const*>(cubinData), cubinSize);
    return digest.get();
}

Nvfp4A16BlackwellMoeJitKernel compileNvfp4A16BlackwellMoeJitKernel(Nvfp4A16BlackwellMoeJitKey const& key)
{
    static PluginJitCompileCache<Nvfp4A16BlackwellMoeJitKey, Nvfp4A16BlackwellMoeJitKernel,
        Nvfp4A16BlackwellMoeJitKeyHasher>
        sCache;
    return sCache.getOrCompile(key, [&key] {
        validateKey(key);
        Nvfp4A16BlackwellMoeJitKernel kernel;
        kernel.key = key;
        kernel.cubin = compilePluginJitKernel(PluginJitProgram::kNVFP4_A16_BLACKWELL_MOE, buildNvrtcOptions(key),
            "NVFP4-A16 Blackwell MoE kernels for " + keyToString(key));
        kernel.digest = computeNvfp4A16BlackwellMoeJitDigest(key, kernel.cubin.data(), kernel.cubin.size());
        return kernel;
    });
}

std::vector<uint8_t> serializeNvfp4A16BlackwellMoeJitKernel(Nvfp4A16BlackwellMoeJitKernel const& kernel)
{
    validateKey(kernel.key);
    if (!isNativeCubin(kernel.cubin.data(), kernel.cubin.size())
        || kernel.cubin.size() > std::numeric_limits<uint32_t>::max())
    {
        throw std::invalid_argument("NVFP4-A16 Blackwell MoE JIT payload must be a native ELF cubin with 32-bit size");
    }
    Nvfp4A16BlackwellMoeJitDigest const digest
        = computeNvfp4A16BlackwellMoeJitDigest(kernel.key, kernel.cubin.data(), kernel.cubin.size());
    if (!(digest == kernel.digest))
    {
        throw std::invalid_argument("NVFP4-A16 Blackwell MoE JIT kernel digest does not match its payload");
    }
    std::vector<uint8_t> blob;
    appendU32(blob, kJIT_BLOB_MAGIC);
    appendU32(blob, kJIT_BLOB_VERSION);
    for (uint32_t const word : keyWords(kernel.key))
    {
        appendU32(blob, word);
    }
    appendU32(blob, static_cast<uint32_t>(kernel.cubin.size()));
    appendU64(blob, kernel.digest.lo);
    appendU64(blob, kernel.digest.hi);
    blob.insert(blob.end(), kernel.cubin.begin(), kernel.cubin.end());
    return blob;
}

Nvfp4A16BlackwellMoeJitKernel deserializeNvfp4A16BlackwellMoeJitKernel(void const* const data, size_t const size)
{
    if (data == nullptr)
    {
        throw std::invalid_argument("NVFP4-A16 Blackwell MoE JIT blob must not be null");
    }
    auto const* cursor = static_cast<uint8_t const*>(data);
    size_t remaining = size;
    uint32_t const magic = readU32(cursor, remaining);
    uint32_t const version = readU32(cursor, remaining);
    if (magic != kJIT_BLOB_MAGIC || version != kJIT_BLOB_VERSION)
    {
        throw std::runtime_error("Unsupported NVFP4-A16 Blackwell MoE JIT blob header");
    }
    Nvfp4A16BlackwellMoeJitKernel kernel;
    kernel.key.sm = static_cast<int32_t>(readU32(cursor, remaining));
    kernel.key.layout = readU32(cursor, remaining);
    kernel.key.numExperts = static_cast<int32_t>(readU32(cursor, remaining));
    kernel.key.topK = static_cast<int32_t>(readU32(cursor, remaining));
    kernel.key.hiddenSize = static_cast<int32_t>(readU32(cursor, remaining));
    kernel.key.interSize = static_cast<int32_t>(readU32(cursor, remaining));
    kernel.key.interSizePadded = static_cast<int32_t>(readU32(cursor, remaining));
    kernel.key.fc1SplitK = static_cast<int32_t>(readU32(cursor, remaining));
    kernel.key.fc2SplitK = static_cast<int32_t>(readU32(cursor, remaining));
    kernel.key.fc2PrefetchSlots = static_cast<int32_t>(readU32(cursor, remaining));
    kernel.key.dataType = static_cast<Nvfp4A16BlackwellMoeDataType>(readU32(cursor, remaining));
    kernel.key.sourceAbi = readU32(cursor, remaining);
    uint32_t const cubinSize = readU32(cursor, remaining);
    kernel.digest.lo = readU64(cursor, remaining);
    kernel.digest.hi = readU64(cursor, remaining);
    if (cubinSize == 0 || remaining < cubinSize)
    {
        throw std::runtime_error("Truncated or empty NVFP4-A16 Blackwell MoE JIT cubin payload");
    }
    kernel.cubin.assign(cursor, cursor + cubinSize);
    cursor += cubinSize;
    remaining -= cubinSize;
    if (remaining != 0)
    {
        throw std::runtime_error(
            format::fmtstr("NVFP4-A16 Blackwell MoE JIT blob has %zu trailing byte(s)", remaining));
    }
    if (!isNativeCubin(kernel.cubin.data(), kernel.cubin.size()))
    {
        throw std::runtime_error("NVFP4-A16 Blackwell MoE JIT blob does not contain a native ELF cubin");
    }
    validateKey(kernel.key);
    Nvfp4A16BlackwellMoeJitDigest const digest
        = computeNvfp4A16BlackwellMoeJitDigest(kernel.key, kernel.cubin.data(), kernel.cubin.size());
    if (!(digest == kernel.digest))
    {
        throw std::runtime_error("NVFP4-A16 Blackwell MoE JIT blob digest mismatch");
    }
    return kernel;
}

} // namespace trt_edgellm
