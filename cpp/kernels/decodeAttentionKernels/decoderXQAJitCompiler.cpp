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

#include "decoderXQAJitCompiler.h"

#include "common/stringUtils.h"
#include "kernels/PluginJitKernels/pluginJitCompileCache.h"
#include "kernels/PluginJitKernels/pluginJitCompiler.h"
#include "xqaKernelTypes.h"

#include <algorithm>
#include <array>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace trt_edgellm
{
namespace
{
constexpr int32_t kHEAD_DIM_512{512};

//! Hash for the compile cache below. Must mix every field XQAJitKey::operator==
//! compares, or distinct kernel variants collide into one bucket.
struct XQAJitKeyHasher
{
    size_t operator()(XQAJitKey const& key) const noexcept
    {
        auto mix = [](size_t hash, size_t value) noexcept {
            // 64-bit FNV-1a style step: multiply-then-xor, so no field can be
            // shifted out of range no matter how many are added later.
            constexpr size_t kPRIME{0x100000001B3ULL};
            return (hash ^ value) * kPRIME;
        };

        using DataTypeValue = std::underlying_type_t<nvinfer1::DataType>;
        size_t hash{0xCBF29CE484222325ULL};
        hash = mix(hash, static_cast<size_t>(key.sm));
        hash = mix(hash, static_cast<size_t>(static_cast<DataTypeValue>(key.dataType)));
        hash = mix(hash, static_cast<size_t>(static_cast<DataTypeValue>(key.kvDataType)));
        hash = mix(hash, static_cast<size_t>(key.headSize));
        hash = mix(hash, static_cast<size_t>(key.qHeadsPerKv));
        hash = mix(hash, static_cast<size_t>(key.tokensPerPage));
        hash = mix(hash, static_cast<size_t>(key.slidingWindow));
        hash = mix(hash, static_cast<size_t>(key.specDecode));
        return hash;
    }
};

int32_t getKVCacheEnum(nvinfer1::DataType kvDataType)
{
    if (kvDataType == nvinfer1::DataType::kHALF)
    {
        return xqa::kernels::kKV_CACHE_ELEM_INPUT;
    }
    if (kvDataType == nvinfer1::DataType::kFP8)
    {
        return xqa::kernels::kKV_CACHE_ELEM_FP8_E4M3;
    }
    throw std::runtime_error("XQA NVRTC JIT supports only FP16 and FP8 KV cache.");
}

std::string getGpuArchitectureOption(int32_t sm)
{
    constexpr int32_t kSM100{100};
    constexpr int32_t kSM101{101};
    constexpr int32_t kCUDA13_MAJOR{13};

    if (sm == kSM100)
    {
        return "--gpu-architecture=sm_100a";
    }
    if (sm == kSM101)
    {
        return getPluginJitNvrtcMajorVersion() >= kCUDA13_MAJOR ? "--gpu-architecture=sm_110a"
                                                                : "--gpu-architecture=sm_101a";
    }
    return "--gpu-architecture=sm_" + std::to_string(sm);
}

std::vector<std::string> buildNvrtcOptions(XQAJitKey const& key)
{
    if (key.dataType != nvinfer1::DataType::kHALF)
    {
        throw std::runtime_error("XQA NVRTC JIT supports only FP16 Q/O tensors.");
    }

    std::vector<std::string> options;
    options.emplace_back("--std=c++17");
    options.emplace_back("--use_fast_math");
    options.emplace_back("--device-as-default-execution-space");
    options.emplace_back(getGpuArchitectureOption(key.sm));
    // No -I flags needed: all headers are passed as virtual includes to nvrtcCreateProgram.
    options.emplace_back("-DGENERATE_CUBIN=1");
    options.emplace_back("-DNDEBUG");
    options.emplace_back("-DINFINITY=__int_as_float(0x7f800000)");
    options.emplace_back("-DINPUT_FP16=1");
    options.emplace_back("-DBEAM_WIDTH=1");
    options.emplace_back("-DTOKENS_PER_PAGE=" + std::to_string(key.tokensPerPage));
    options.emplace_back("-DHEAD_ELEMS=" + std::to_string(key.headSize));
    options.emplace_back("-DCACHE_ELEM_ENUM=" + std::to_string(getKVCacheEnum(key.kvDataType)));
    options.emplace_back("-DSLIDING_WINDOW=" + std::to_string(key.slidingWindow ? 1 : 0));
    options.emplace_back("-DSPEC_DEC=" + std::to_string(key.specDecode ? 1 : 0));
    options.emplace_back("-DHEAD_GRP_SIZE=" + std::to_string(key.specDecode ? 0 : key.qHeadsPerKv));
    options.emplace_back("-DM_TILESIZE=" + std::to_string(getXQAJitMTileSize(key)));

    if (key.headSize == kHEAD_DIM_512 && (key.sm == 86 || key.sm == 89))
    {
        options.emplace_back("-DTILED_QKV_STAGING_HEAD_DIM512=1");
    }
    if (key.headSize == kHEAD_DIM_512 && (key.sm == 120 || key.sm == 121))
    {
        options.emplace_back("-DXQA_2CTA_HEAD_DIM512=1");
    }

    return options;
}

std::string keyToString(XQAJitKey const& key)
{
    return format::fmtstr(
        "SM%d, dtype=%d, kv_dtype=%d, head_dim=%d, q_heads_per_kv=%d, tokens_per_page=%d, sliding_window=%d, "
        "spec_decode=%d",
        key.sm, static_cast<int32_t>(key.dataType), static_cast<int32_t>(key.kvDataType), key.headSize, key.qHeadsPerKv,
        key.tokensPerPage, key.slidingWindow ? 1 : 0, key.specDecode ? 1 : 0);
}

//! Format version for the serialized XQA JIT kernel blob. Bump on any layout change.
constexpr uint32_t kXQA_JIT_BLOB_VERSION{1};

void appendU32(std::vector<uint8_t>& out, uint32_t value)
{
    for (int32_t shift = 0; shift < 32; shift += 8)
    {
        out.push_back(static_cast<uint8_t>((value >> shift) & 0xFFU));
    }
}

uint32_t readU32(uint8_t const*& cursor, size_t& remaining)
{
    if (remaining < sizeof(uint32_t))
    {
        throw std::runtime_error("Truncated XQA JIT kernel blob.");
    }
    uint32_t value{0};
    for (int32_t shift = 0; shift < 32; shift += 8)
    {
        value |= static_cast<uint32_t>(*cursor++) << shift;
    }
    remaining -= sizeof(uint32_t);
    return value;
}

//! Serialize the key field by field. A memcpy of the struct would also copy
//! padding bytes, which makes engines non-reproducible.
void appendKey(std::vector<uint8_t>& out, XQAJitKey const& key)
{
    appendU32(out, static_cast<uint32_t>(key.sm));
    appendU32(out, static_cast<uint32_t>(key.dataType));
    appendU32(out, static_cast<uint32_t>(key.kvDataType));
    appendU32(out, static_cast<uint32_t>(key.headSize));
    appendU32(out, static_cast<uint32_t>(key.qHeadsPerKv));
    appendU32(out, static_cast<uint32_t>(key.tokensPerPage));
    appendU32(out, key.slidingWindow ? 1U : 0U);
    appendU32(out, key.specDecode ? 1U : 0U);
}

XQAJitKey readKey(uint8_t const*& cursor, size_t& remaining)
{
    XQAJitKey key{};
    key.sm = static_cast<int32_t>(readU32(cursor, remaining));
    key.dataType = static_cast<nvinfer1::DataType>(readU32(cursor, remaining));
    key.kvDataType = static_cast<nvinfer1::DataType>(readU32(cursor, remaining));
    key.headSize = static_cast<int32_t>(readU32(cursor, remaining));
    key.qHeadsPerKv = static_cast<int32_t>(readU32(cursor, remaining));
    key.tokensPerPage = static_cast<int32_t>(readU32(cursor, remaining));
    key.slidingWindow = readU32(cursor, remaining) != 0U;
    key.specDecode = readU32(cursor, remaining) != 0U;
    return key;
}

} // namespace

std::vector<uint8_t> serializeXQAJitKernels(std::vector<XQAJitKernel> const& kernels)
{
    std::vector<uint8_t> blob;
    appendU32(blob, kXQA_JIT_BLOB_VERSION);
    appendU32(blob, static_cast<uint32_t>(kernels.size()));
    for (XQAJitKernel const& kernel : kernels)
    {
        appendKey(blob, kernel.key);
        appendU32(blob, static_cast<uint32_t>(kernel.cubin.size()));
        blob.insert(blob.end(), kernel.cubin.begin(), kernel.cubin.end());
    }
    return blob;
}

std::vector<XQAJitKernel> deserializeXQAJitKernels(void const* data, size_t size)
{
    auto const* cursor = static_cast<uint8_t const*>(data);
    size_t remaining = size;

    uint32_t const version = readU32(cursor, remaining);
    if (version != kXQA_JIT_BLOB_VERSION)
    {
        throw std::runtime_error(format::fmtstr(
            "Unsupported XQA JIT kernel blob version %u (expected %u).", version, kXQA_JIT_BLOB_VERSION));
    }

    uint32_t const count = readU32(cursor, remaining);
    std::vector<XQAJitKernel> kernels;
    kernels.reserve(count);
    for (uint32_t i = 0; i < count; ++i)
    {
        XQAJitKernel kernel;
        kernel.key = readKey(cursor, remaining);
        uint32_t const cubinSize = readU32(cursor, remaining);
        if (remaining < cubinSize)
        {
            throw std::runtime_error("Truncated XQA JIT kernel blob: cubin payload is short.");
        }
        kernel.cubin.assign(cursor, cursor + cubinSize);
        cursor += cubinSize;
        remaining -= cubinSize;
        kernels.push_back(std::move(kernel));
    }

    if (remaining != 0)
    {
        throw std::runtime_error(format::fmtstr("XQA JIT kernel blob has %zu trailing byte(s).", remaining));
    }
    return kernels;
}

bool canCompileXQAKernel(int32_t numQHeads, int32_t numKVHeads, int32_t headSize, int32_t smVersion,
    nvinfer1::DataType dataType, nvinfer1::DataType kvDataType) noexcept
{
    // SM versions with an XQA decode kernel at all.
    constexpr std::array<int32_t, 9> kALLOWED_SM_VERSIONS{80, 86, 87, 89, 90, 100, 101, 120, 121};
    // Of those, the ones with native FP8 converts. Listed explicitly rather
    // than tested as `smVersion >= 89`: a numeric threshold silently admits
    // every future SM added to kALLOWED_SM_VERSIONS, and whether that SM has
    // usable FP8 is a question someone has to answer deliberately.
    constexpr std::array<int32_t, 6> kFP8_CAPABLE_SM_VERSIONS{89, 90, 100, 101, 120, 121};

    auto const contains = [](auto const& versions, int32_t sm) noexcept {
        return std::find(versions.begin(), versions.end(), sm) != versions.end();
    };

    bool const checkHeadNumbers = numQHeads % numKVHeads == 0;
    bool const checkType = dataType == nvinfer1::DataType::kHALF;
    // NVRTC will happily compile CACHE_ELEM_ENUM=2 for an SM without native
    // FP8 converts, but they fall back to software emulation: the cubin comes
    // out ~3.6x larger and correspondingly slow, with no diagnostic.
    bool const checkKVType = kvDataType == nvinfer1::DataType::kHALF
        || (kvDataType == nvinfer1::DataType::kFP8 && contains(kFP8_CAPABLE_SM_VERSIONS, smVersion));
    bool const checkSMVersion = contains(kALLOWED_SM_VERSIONS, smVersion);

    // Current kernel list supports
    // (1) Head ratio 1-8 for head_dim {32, 64, 128}
    // (2) Head ratio 16 for head_dim 128 only (NemotronH).
    // (3) Head ratio 2, 4, 6, 8 for head_dim 256
    //     (4/6/8 for Qwen3.5-MoE / Qwen3.5-Omni Thinker+Talker;
    //      2 for Qwen3.5-Omni Talker decode attention — 16 Q heads / 8 KV heads).
    // (4) Head ratio 2, 4, 8, 16 for head_dim 512.
    //     (2 for Gemma4 E4B assistant: 4 Q heads / 2 KV heads;
    //      4 for Gemma4 E4B: 8 Q heads / 2 KV heads;
    //      8 for Gemma4 E2B: 8 Q heads / 1 KV head;
    //      16 for Gemma4 Unified 12B global attention / single-KV-head decode).
    //     The pre-JIT list additionally gated d512 on which cubins had been
    //     generated for the SM; JIT compiles on demand, so only the ratio
    //     constraint remains.
    int32_t const headRatio = numQHeads / numKVHeads;
    bool const checkQHeadPerKV
        = ((headSize == 32 || headSize == 64 || headSize == 128) && headRatio >= 1 && headRatio <= 8)
        || (headSize == 128 && headRatio == 16)
        || (headSize == 256 && (headRatio == 2 || headRatio == 4 || headRatio == 6 || headRatio == 8))
        || (headSize == 512 && (headRatio == 2 || headRatio == 4 || headRatio == 8 || headRatio == 16));

    return checkHeadNumbers && checkType && checkKVType && checkSMVersion && checkQHeadPerKV;
}

XQAJitResult compileXQAKernel(XQAJitKey const& key)
{
    static PluginJitCompileCache<XQAJitKey, XQAJitResult, XQAJitKeyHasher> sCache;
    return sCache.getOrCompile(key, [&key] {
        XQAJitResult result;
        result.cubin = compilePluginJitKernel(
            PluginJitProgram::kXQA, buildNvrtcOptions(key), "XQA kernel for " + keyToString(key));
        return result;
    });
}

} // namespace trt_edgellm
