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
#include "kernels/PluginJitKernels/pluginJitBlob.h"
#include "kernels/PluginJitKernels/pluginJitCompiler.h"

#include <stdexcept>
#include <string>
#include <vector>

namespace trt_edgellm
{
namespace
{

constexpr PluginJitBlobFormat kJIT_BLOB_FORMAT{"NVFP4-A16 Blackwell GEMV JIT", 0x56363134U /* "416V" */};

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

Nvfp4A16BlackwellGemvJitKernel compileNvfp4A16BlackwellGemvJitKernel(Nvfp4A16BlackwellGemvJitKey const& key)
{
    validateKey(key);
    return getOrCompilePluginJitKernel(PluginJitProgram::kNVFP4_A16_BLACKWELL_GEMV, key, buildNvrtcOptions(key),
        "NVFP4-A16 Blackwell GEMV kernel for " + keyToString(key));
}

std::vector<uint8_t> serializeNvfp4A16BlackwellGemvJitKernel(Nvfp4A16BlackwellGemvJitKernel const& kernel)
{
    validateKey(kernel.key);
    return serializePluginJitKernel(kJIT_BLOB_FORMAT, kernel);
}

Nvfp4A16BlackwellGemvJitKernel deserializeNvfp4A16BlackwellGemvJitKernel(void const* const data, size_t const size)
{
    Nvfp4A16BlackwellGemvJitKernel kernel
        = deserializePluginJitKernel<Nvfp4A16BlackwellGemvJitKey>(kJIT_BLOB_FORMAT, data, size);
    validateKey(kernel.key);
    return kernel;
}

} // namespace trt_edgellm
