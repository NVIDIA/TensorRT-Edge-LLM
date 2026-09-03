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

#include "nvfp4A16BlackwellGemvJitCompiler.h"

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

constexpr uint32_t kJIT_BLOB_MAGIC{0x4247564EU};
constexpr uint32_t kJIT_BLOB_VERSION{1U};

struct Nvfp4A16BlackwellGemvJitKeyHasher
{
    size_t operator()(Nvfp4A16BlackwellGemvJitKey const& key) const noexcept
    {
        auto mix = [](size_t hash, size_t value) noexcept {
            constexpr size_t kPRIME{0x100000001B3ULL};
            return (hash ^ value) * kPRIME;
        };

        using DataTypeValue = std::underlying_type_t<Nvfp4A16BlackwellGemvDataType>;
        size_t hash{0xCBF29CE484222325ULL};
        hash = mix(hash, static_cast<size_t>(key.sm));
        hash = mix(hash, static_cast<size_t>(key.layout));
        hash = mix(hash, static_cast<size_t>(key.n));
        hash = mix(hash, static_cast<size_t>(key.k));
        hash = mix(hash, static_cast<size_t>(static_cast<DataTypeValue>(key.dataType)));
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

void validateKey(Nvfp4A16BlackwellGemvJitKey const& key)
{
    if (key.sm != nvfp4_a16_blackwell::kTargetSm)
    {
        throw std::invalid_argument("NVFP4-A16 Blackwell GEMV JIT requires SM110");
    }
    if (key.layout != kNVFP4_A16_BLACKWELL_GEMV_LAYOUT_ABI)
    {
        throw std::invalid_argument("Unsupported NVFP4-A16 Blackwell GEMV layout ABI");
    }
    if (key.sourceAbi != kNVFP4_A16_BLACKWELL_GEMV_SOURCE_ABI)
    {
        throw std::invalid_argument("Unsupported NVFP4-A16 Blackwell GEMV source ABI");
    }
    if (!nvfp4_a16_blackwell::isSupportedProblemShape(key.n, key.k))
    {
        throw std::invalid_argument("Unsupported NVFP4-A16 Blackwell GEMV N/K specialization");
    }
    switch (key.dataType)
    {
    case Nvfp4A16BlackwellGemvDataType::kHALF:
    case Nvfp4A16BlackwellGemvDataType::kBF16: break;
    default: throw std::invalid_argument("Unsupported NVFP4-A16 Blackwell GEMV data type");
    }
}

std::string keyToString(Nvfp4A16BlackwellGemvJitKey const& key)
{
    return format::fmtstr("SM%d, layout=%u, N=%d, K=%d, dtype=%u, source ABI=%u", key.sm, key.layout, key.n, key.k,
        static_cast<uint32_t>(key.dataType), key.sourceAbi);
}

std::vector<std::string> buildNvrtcOptions(Nvfp4A16BlackwellGemvJitKey const& key)
{
    return {"--std=c++17", "--use_fast_math", "--device-as-default-execution-space", "--gpu-architecture=sm_110a",
        "-DNDEBUG", "-DGEMV_N=" + std::to_string(key.n), "-DGEMV_K=" + std::to_string(key.k),
        "-DGEMV_DATA_TYPE=" + std::to_string(static_cast<uint32_t>(key.dataType)),
        "-DGEMV_LAYOUT_ABI=" + std::to_string(key.layout), "-DGEMV_SOURCE_ABI=" + std::to_string(key.sourceAbi)};
}

void appendU32(std::vector<uint8_t>& output, uint32_t const value)
{
    constexpr int32_t kBITS_PER_BYTE{8};
    constexpr int32_t kU32_BITS{32};
    for (int32_t shift = 0; shift < kU32_BITS; shift += kBITS_PER_BYTE)
    {
        output.push_back(static_cast<uint8_t>((value >> shift) & 0xFFU));
    }
}

void appendU64(std::vector<uint8_t>& output, uint64_t const value)
{
    constexpr int32_t kBITS_PER_BYTE{8};
    constexpr int32_t kU64_BITS{64};
    for (int32_t shift = 0; shift < kU64_BITS; shift += kBITS_PER_BYTE)
    {
        output.push_back(static_cast<uint8_t>((value >> shift) & 0xFFU));
    }
}

uint32_t readU32(uint8_t const*& cursor, size_t& remaining)
{
    if (remaining < sizeof(uint32_t))
    {
        throw std::runtime_error("Truncated NVFP4-A16 Blackwell GEMV JIT blob");
    }
    constexpr int32_t kBITS_PER_BYTE{8};
    constexpr int32_t kU32_BITS{32};
    uint32_t value{0};
    for (int32_t shift = 0; shift < kU32_BITS; shift += kBITS_PER_BYTE)
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
        throw std::runtime_error("Truncated NVFP4-A16 Blackwell GEMV JIT blob");
    }
    constexpr int32_t kBITS_PER_BYTE{8};
    constexpr int32_t kU64_BITS{64};
    uint64_t value{0};
    for (int32_t shift = 0; shift < kU64_BITS; shift += kBITS_PER_BYTE)
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
        constexpr int32_t kBITS_PER_BYTE{8};
        constexpr int32_t kU32_BITS{32};
        for (int32_t shift = 0; shift < kU32_BITS; shift += kBITS_PER_BYTE)
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

    Nvfp4A16BlackwellGemvJitDigest get() const noexcept
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

} // namespace

bool canCompileNvfp4A16BlackwellGemvJitKernel(Nvfp4A16BlackwellGemvJitKey const& key) noexcept
{
    try
    {
        validateKey(key);
        return true;
    }
    catch (...)
    {
        return false;
    }
}

Nvfp4A16BlackwellGemvJitDigest computeNvfp4A16BlackwellGemvJitDigest(
    Nvfp4A16BlackwellGemvJitKey const& key, void const* const cubinData, size_t const cubinSize)
{
    validateKey(key);
    if (!isNativeCubin(cubinData, cubinSize))
    {
        throw std::invalid_argument("NVFP4-A16 Blackwell GEMV JIT payload must be a native ELF cubin");
    }

    DigestBuilder digest;
    digest.appendU32(static_cast<uint32_t>(key.sm));
    digest.appendU32(key.layout);
    digest.appendU32(static_cast<uint32_t>(key.n));
    digest.appendU32(static_cast<uint32_t>(key.k));
    digest.appendU32(static_cast<uint32_t>(key.dataType));
    digest.appendU32(key.sourceAbi);
    digest.appendBytes(static_cast<uint8_t const*>(cubinData), cubinSize);
    return digest.get();
}

Nvfp4A16BlackwellGemvJitKernel compileNvfp4A16BlackwellGemvJitKernel(Nvfp4A16BlackwellGemvJitKey const& key)
{
    static PluginJitCompileCache<Nvfp4A16BlackwellGemvJitKey, Nvfp4A16BlackwellGemvJitKernel,
        Nvfp4A16BlackwellGemvJitKeyHasher>
        sCache;
    return sCache.getOrCompile(key, [&key] {
        validateKey(key);
        Nvfp4A16BlackwellGemvJitKernel kernel;
        kernel.key = key;
        kernel.cubin = compilePluginJitKernel(PluginJitProgram::kNVFP4_A16_BLACKWELL_GEMV, buildNvrtcOptions(key),
            "NVFP4-A16 Blackwell GEMV kernel for " + keyToString(key));
        kernel.digest = computeNvfp4A16BlackwellGemvJitDigest(key, kernel.cubin.data(), kernel.cubin.size());
        return kernel;
    });
}

std::vector<uint8_t> serializeNvfp4A16BlackwellGemvJitKernel(Nvfp4A16BlackwellGemvJitKernel const& kernel)
{
    validateKey(kernel.key);
    if (!isNativeCubin(kernel.cubin.data(), kernel.cubin.size())
        || kernel.cubin.size() > std::numeric_limits<uint32_t>::max())
    {
        throw std::invalid_argument("NVFP4-A16 Blackwell GEMV JIT payload must be a native ELF cubin with 32-bit size");
    }
    Nvfp4A16BlackwellGemvJitDigest const digest
        = computeNvfp4A16BlackwellGemvJitDigest(kernel.key, kernel.cubin.data(), kernel.cubin.size());
    if (!(digest == kernel.digest))
    {
        throw std::invalid_argument("NVFP4-A16 Blackwell GEMV JIT kernel digest does not match its payload");
    }

    std::vector<uint8_t> blob;
    appendU32(blob, kJIT_BLOB_MAGIC);
    appendU32(blob, kJIT_BLOB_VERSION);
    appendU32(blob, static_cast<uint32_t>(kernel.key.sm));
    appendU32(blob, kernel.key.layout);
    appendU32(blob, static_cast<uint32_t>(kernel.key.n));
    appendU32(blob, static_cast<uint32_t>(kernel.key.k));
    appendU32(blob, static_cast<uint32_t>(kernel.key.dataType));
    appendU32(blob, kernel.key.sourceAbi);
    appendU32(blob, static_cast<uint32_t>(kernel.cubin.size()));
    appendU64(blob, kernel.digest.lo);
    appendU64(blob, kernel.digest.hi);
    blob.insert(blob.end(), kernel.cubin.begin(), kernel.cubin.end());
    return blob;
}

Nvfp4A16BlackwellGemvJitKernel deserializeNvfp4A16BlackwellGemvJitKernel(void const* const data, size_t const size)
{
    if (data == nullptr)
    {
        throw std::invalid_argument("NVFP4-A16 Blackwell GEMV JIT blob must not be null");
    }

    auto const* cursor = static_cast<uint8_t const*>(data);
    size_t remaining = size;
    uint32_t const magic = readU32(cursor, remaining);
    uint32_t const version = readU32(cursor, remaining);
    if (magic != kJIT_BLOB_MAGIC || version != kJIT_BLOB_VERSION)
    {
        throw std::runtime_error("Unsupported NVFP4-A16 Blackwell GEMV JIT blob header");
    }

    Nvfp4A16BlackwellGemvJitKernel kernel;
    kernel.key.sm = static_cast<int32_t>(readU32(cursor, remaining));
    kernel.key.layout = readU32(cursor, remaining);
    kernel.key.n = static_cast<int32_t>(readU32(cursor, remaining));
    kernel.key.k = static_cast<int32_t>(readU32(cursor, remaining));
    kernel.key.dataType = static_cast<Nvfp4A16BlackwellGemvDataType>(readU32(cursor, remaining));
    kernel.key.sourceAbi = readU32(cursor, remaining);
    uint32_t const cubinSize = readU32(cursor, remaining);
    kernel.digest.lo = readU64(cursor, remaining);
    kernel.digest.hi = readU64(cursor, remaining);
    if (cubinSize == 0 || remaining < cubinSize)
    {
        throw std::runtime_error("Truncated or empty NVFP4-A16 Blackwell GEMV JIT cubin payload");
    }
    kernel.cubin.assign(cursor, cursor + cubinSize);
    cursor += cubinSize;
    remaining -= cubinSize;
    if (remaining != 0)
    {
        throw std::runtime_error(
            format::fmtstr("NVFP4-A16 Blackwell GEMV JIT blob has %zu trailing byte(s)", remaining));
    }
    if (!isNativeCubin(kernel.cubin.data(), kernel.cubin.size()))
    {
        throw std::runtime_error("NVFP4-A16 Blackwell GEMV JIT blob does not contain a native ELF cubin");
    }

    validateKey(kernel.key);
    Nvfp4A16BlackwellGemvJitDigest const digest
        = computeNvfp4A16BlackwellGemvJitDigest(kernel.key, kernel.cubin.data(), kernel.cubin.size());
    if (!(digest == kernel.digest))
    {
        throw std::runtime_error("NVFP4-A16 Blackwell GEMV JIT blob digest mismatch");
    }
    return kernel;
}

} // namespace trt_edgellm
