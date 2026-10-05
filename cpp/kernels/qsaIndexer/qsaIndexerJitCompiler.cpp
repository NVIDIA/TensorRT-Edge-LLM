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

#include "qsaIndexerJitCompiler.h"

#include "common/pagedKvTypes.h"
#include "common/stringUtils.h"
#include "kernels/PluginJitKernels/pluginJitCompileCache.h"
#include "kernels/PluginJitKernels/pluginJitCompiler.h"
#include "qsaIndexerKernels.h"

#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace trt_edgellm
{
namespace
{

struct QsaIndexerJitKeyHasher
{
    size_t operator()(QsaIndexerJitKey const& key) const noexcept
    {
        auto mix = [](size_t hash, size_t value) noexcept {
            constexpr size_t kPRIME{0x100000001B3ULL};
            return (hash ^ value) * kPRIME;
        };

        using DataTypeValue = std::underlying_type_t<QsaIndexerJitDataType>;
        size_t hash{0xCBF29CE484222325ULL};
        hash = mix(hash, static_cast<size_t>(key.sm));
        return mix(hash, static_cast<size_t>(static_cast<DataTypeValue>(key.dataType)));
    }
};

void validateKey(QsaIndexerJitKey const& key)
{
    if (key.sm <= 0)
    {
        throw std::invalid_argument("QSA indexer JIT requires a positive SM version");
    }
    switch (key.dataType)
    {
    case QsaIndexerJitDataType::kHALF:
    case QsaIndexerJitDataType::kBF16: break;
    default: throw std::invalid_argument("Unsupported QSA indexer JIT data type");
    }
}

std::string keyToString(QsaIndexerJitKey const& key)
{
    return format::fmtstr("SM%d, dtype=%u", key.sm, static_cast<uint32_t>(key.dataType));
}

//! The kernels are arch-generic, so compile for the plain (non-"a") architecture. Thor is
//! numbered SM 10.1 by CUDA 12 and SM 11.0 by CUDA 13; map either spelling to whichever
//! name the linked NVRTC understands (same workaround as the XQA JIT compiler).
std::string getGpuArchitectureOption(int32_t sm)
{
    constexpr int32_t kSM101{101};
    constexpr int32_t kSM110{110};
    constexpr int32_t kCUDA13_MAJOR{13};

    if (sm == kSM101 || sm == kSM110)
    {
        return getPluginJitNvrtcMajorVersion() >= kCUDA13_MAJOR ? "--gpu-architecture=sm_110"
                                                                : "--gpu-architecture=sm_101";
    }
    return "--gpu-architecture=sm_" + std::to_string(sm);
}

std::vector<std::string> buildNvrtcOptions(QsaIndexerJitKey const& key)
{
    // --use_fast_math matches the Release nvcc flags the AOT kernels were validated with.
    // The geometry defines are the device source's only definition of the indexer and pool
    // constants.
    return {"--std=c++17", "--use_fast_math", "--device-as-default-execution-space", getGpuArchitectureOption(key.sm),
        "-DNDEBUG", "-DQSA_INDEXER_DATA_TYPE=" + std::to_string(static_cast<uint32_t>(key.dataType)),
        "-DQSA_INDEXER_NUM_HEADS=" + std::to_string(kernel::kQSA_INDEXER_NUM_HEADS),
        "-DQSA_INDEXER_HEAD_DIM=" + std::to_string(kernel::kQSA_INDEXER_HEAD_DIM),
        "-DQSA_INDEXER_COMPRESS_RATIO=" + std::to_string(kernel::kQSA_COMPRESS_RATIO),
        "-DQSA_INDEXER_INDEX_BUDGET=" + std::to_string(kernel::kQSA_INDEX_BUDGET),
        "-DQSA_INDEXER_INDEX_WIDTH=" + std::to_string(kernel::kQSA_INDEX_WIDTH),
        "-DQSA_INDEXER_ROTARY_DIM=" + std::to_string(kernel::kQSA_INDEXER_ROTARY_DIM),
        "-DQSA_INDEXER_TOPK_THREADS=" + std::to_string(kernel::kQSA_INDEXER_TOPK_THREADS),
        "-DQSA_INDEXER_SCORES_COLUMNS_PER_CTA=" + std::to_string(kernel::kQSA_INDEXER_SCORES_COLUMNS_PER_CTA),
        "-DQSA_INDEXER_SCORES_COLUMNS_PER_WARP=" + std::to_string(kernel::kQSA_INDEXER_SCORES_COLUMNS_PER_WARP),
        "-DQSA_INDEXER_TOKENS_PER_PAGE=" + std::to_string(rt::kTOKENS_PER_PAGE),
        "-DQSA_INDEXER_UNUSED_PAGE_ENTRY=" + std::to_string(rt::kUNUSED_PAGE_ENTRY)};
}

} // namespace

bool canCompileQsaIndexerJitKernel(QsaIndexerJitKey const& key) noexcept
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

QsaIndexerJitKernel compileQsaIndexerJitKernel(QsaIndexerJitKey const& key)
{
    static PluginJitCompileCache<QsaIndexerJitKey, QsaIndexerJitKernel, QsaIndexerJitKeyHasher> sCache;
    return sCache.getOrCompile(key, [&key] {
        validateKey(key);
        QsaIndexerJitKernel kernel;
        kernel.key = key;
        kernel.cubin = compilePluginJitKernel(
            PluginJitProgram::kQSA_INDEXER, buildNvrtcOptions(key), "QSA indexer kernels for " + keyToString(key));
        return kernel;
    });
}

} // namespace trt_edgellm
