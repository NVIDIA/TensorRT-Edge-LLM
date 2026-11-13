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
#include "kernels/PluginJitKernels/pluginJitBlob.h"
#include "kernels/PluginJitKernels/pluginJitCompiler.h"
#include "xqaKernelTypes.h"

#include <algorithm>
#include <array>
#include <stdexcept>
#include <string>
#include <vector>

namespace trt_edgellm
{
namespace
{
constexpr int32_t kHEAD_DIM_512{512};

constexpr PluginJitBlobFormat kXQA_JIT_BLOB_FORMAT{"XQA JIT", 0x4151584EU /* "NXQA" */};

void validateXQAJitKey(XQAJitKey const& key)
{
    if (key.contiguousQuerySwa && (!key.specDecode || !key.slidingWindow))
    {
        throw std::runtime_error("Contiguous-query XQA SWA requires spec-decode and sliding-window attention.");
    }
}

int32_t getKVCacheEnum(trt_edgellm::DataType kvDataType)
{
    if (kvDataType == trt_edgellm::DataType::kHALF)
    {
        return xqa::kernels::kKV_CACHE_ELEM_INPUT;
    }
    if (kvDataType == trt_edgellm::DataType::kFP8)
    {
        return xqa::kernels::kKV_CACHE_ELEM_FP8_E4M3;
    }
    throw std::runtime_error("XQA NVRTC JIT supports only FP16 and FP8 KV cache.");
}

std::string getGpuArchitectureOption(int32_t sm)
{
    constexpr int32_t kSM100{100};
    constexpr int32_t kSM101{101};
    constexpr int32_t kSM103{103};
    constexpr int32_t kCUDA13_MAJOR{13};

    if (sm == kSM100 || sm == kSM103)
    {
        return "--gpu-architecture=sm_" + std::to_string(sm) + "a";
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
    validateXQAJitKey(key);
    if (key.dataType != trt_edgellm::DataType::kHALF)
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
    options.emplace_back("-DCONTIGUOUS_QUERY_SWA=" + std::to_string(key.contiguousQuerySwa ? 1 : 0));
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
        "spec_decode=%d, contiguous_query_swa=%d",
        key.sm, static_cast<int32_t>(key.dataType), static_cast<int32_t>(key.kvDataType), key.headSize, key.qHeadsPerKv,
        key.tokensPerPage, key.slidingWindow ? 1 : 0, key.specDecode ? 1 : 0, key.contiguousQuerySwa ? 1 : 0);
}

} // namespace

std::vector<uint8_t> serializeXQAJitKernels(std::vector<XQAJitKernel> const& kernels)
{
    for (XQAJitKernel const& kernel : kernels)
    {
        validateXQAJitKey(kernel.key);
    }
    return serializePluginJitKernels(kXQA_JIT_BLOB_FORMAT, kernels);
}

std::vector<XQAJitKernel> deserializeXQAJitKernels(void const* data, size_t size)
{
    std::vector<XQAJitKernel> kernels = deserializePluginJitKernels<XQAJitKey>(kXQA_JIT_BLOB_FORMAT, data, size);
    for (XQAJitKernel const& kernel : kernels)
    {
        validateXQAJitKey(kernel.key);
    }
    return kernels;
}

XQAJitKernel compileXQAKernel(XQAJitKey const& key)
{
    validateXQAJitKey(key);
    return getOrCompilePluginJitKernel(
        PluginJitProgram::kXQA, key, buildNvrtcOptions(key), "XQA kernel for " + keyToString(key));
}

} // namespace trt_edgellm
