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

#include "nvfp4A16BlackwellGemvJitRunner.h"

#include "common/checkMacros.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include <array>
#include <cstdint>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <unordered_map>

namespace trt_edgellm
{

struct Nvfp4A16BlackwellGemvLoadedModule
{
    ~Nvfp4A16BlackwellGemvLoadedModule()
    {
        if (module != nullptr)
        {
            (void) cuModuleUnload(module);
        }
    }

    CUcontext context{};
    CUmodule module{};
    std::array<std::array<CUfunction, 3>, 5> mainFunctions{};
    std::array<std::array<CUfunction, 3>, 5> reduceFunctions{};
    std::array<CUfunction, 2> runtimeSplitMainFunctions{};
};

namespace
{

constexpr int32_t kWARP_SIZE{32};
constexpr int32_t kWARPS_PER_BLOCK{8};
constexpr int32_t kTHREADS_PER_BLOCK{kWARP_SIZE * kWARPS_PER_BLOCK};
constexpr int32_t kREDUCE_THREADS{256};
constexpr std::array<int32_t, 5> kSUPPORTED_M{1, 2, 4, 8, 16};
constexpr std::array<int32_t, 3> kSUPPORTED_SPLIT_K{1, 2, 4};

bool isSupportedM(int32_t const m) noexcept
{
    for (int32_t const supportedM : kSUPPORTED_M)
    {
        if (m == supportedM)
        {
            return true;
        }
    }
    return false;
}

bool isSupportedSplitK(int32_t const splitK) noexcept
{
    return splitK == 1 || splitK == 2 || splitK == 4;
}

bool checkedMultiply(size_t const lhs, size_t const rhs, size_t& product) noexcept
{
    if (lhs != 0 && rhs > std::numeric_limits<size_t>::max() / lhs)
    {
        return false;
    }
    product = lhs * rhs;
    return true;
}

bool getWorkspaceSizeChecked(
    Nvfp4A16BlackwellGemvJitKey const& key, int32_t const m, int32_t const splitK, size_t& workspaceSize) noexcept
{
    workspaceSize = 0;
    if (splitK == 1)
    {
        return true;
    }

    size_t elements{};
    size_t rows{};
    return checkedMultiply(static_cast<size_t>(splitK), static_cast<size_t>(m), rows)
        && checkedMultiply(rows, static_cast<size_t>(key.n), elements)
        && checkedMultiply(elements, sizeof(float), workspaceSize);
}

bool getMainGridXChecked(Nvfp4A16BlackwellGemvJitKey const& key, uint32_t& gridX) noexcept
{
    uint64_t const blocks = static_cast<uint64_t>(key.n) / nvfp4_a16_blackwell::kNTile;
    if (blocks == 0 || blocks > std::numeric_limits<uint32_t>::max())
    {
        return false;
    }
    gridX = static_cast<uint32_t>(blocks);
    return true;
}

bool getReduceGridXChecked(
    Nvfp4A16BlackwellGemvJitKey const& key, int32_t const m, int32_t const splitK, uint32_t& gridX) noexcept
{
    gridX = 0;
    if (splitK == 1)
    {
        return true;
    }

    uint64_t const outputPairs = static_cast<uint64_t>(m) * static_cast<uint64_t>(key.n) / 2U;
    uint64_t const blocks = (outputPairs - 1U) / static_cast<uint64_t>(kREDUCE_THREADS) + 1U;
    if (blocks > std::numeric_limits<uint32_t>::max())
    {
        return false;
    }
    gridX = static_cast<uint32_t>(blocks);
    return true;
}

size_t getMIndex(int32_t const m)
{
    for (size_t index = 0; index < kSUPPORTED_M.size(); ++index)
    {
        if (m == kSUPPORTED_M[index])
        {
            return index;
        }
    }
    throw std::invalid_argument("Unsupported NVFP4-A16 Blackwell GEMV M specialization");
}

size_t getSplitKIndex(int32_t const splitK)
{
    for (size_t index = 0; index < kSUPPORTED_SPLIT_K.size(); ++index)
    {
        if (splitK == kSUPPORTED_SPLIT_K[index])
        {
            return index;
        }
    }
    throw std::invalid_argument("Unsupported NVFP4-A16 Blackwell GEMV split-K specialization");
}

size_t getRuntimeSplitMIndex(int32_t const m)
{
    if (m == 8)
    {
        return 0;
    }
    if (m == 16)
    {
        return 1;
    }
    throw std::invalid_argument("Unsupported NVFP4-A16 Blackwell GEMV runtime split-K M specialization");
}

struct RegistryKey
{
    CUcontext context{};
    Nvfp4A16BlackwellGemvJitKey jitKey{};
    Nvfp4A16BlackwellGemvJitDigest digest{};

    bool operator==(RegistryKey const& other) const noexcept
    {
        return context == other.context && jitKey == other.jitKey && digest == other.digest;
    }
};

struct RegistryKeyHasher
{
    size_t operator()(RegistryKey const& key) const noexcept
    {
        auto mix = [](size_t hash, size_t value) noexcept {
            constexpr size_t kPRIME{0x100000001B3ULL};
            return (hash ^ value) * kPRIME;
        };

        using DataTypeValue = std::underlying_type_t<Nvfp4A16BlackwellGemvDataType>;
        size_t hash{0xCBF29CE484222325ULL};
        hash = mix(hash, reinterpret_cast<uintptr_t>(key.context));
        hash = mix(hash, static_cast<size_t>(key.jitKey.sm));
        hash = mix(hash, static_cast<size_t>(key.jitKey.layout));
        hash = mix(hash, static_cast<size_t>(key.jitKey.n));
        hash = mix(hash, static_cast<size_t>(key.jitKey.k));
        hash = mix(hash, static_cast<size_t>(static_cast<DataTypeValue>(key.jitKey.dataType)));
        hash = mix(hash, static_cast<size_t>(key.jitKey.sourceAbi));
        hash = mix(hash, static_cast<size_t>(key.digest.lo));
        return mix(hash, static_cast<size_t>(key.digest.hi));
    }
};

class KernelRegistry
{
public:
    std::shared_ptr<Nvfp4A16BlackwellGemvLoadedModule> load(Nvfp4A16BlackwellGemvJitKernel const& kernel)
    {
        if (!canCompileNvfp4A16BlackwellGemvJitKernel(kernel.key) || kernel.cubin.empty())
        {
            throw std::invalid_argument("Invalid NVFP4-A16 Blackwell GEMV JIT key or cubin");
        }
        Nvfp4A16BlackwellGemvJitDigest const actualDigest
            = computeNvfp4A16BlackwellGemvJitDigest(kernel.key, kernel.cubin.data(), kernel.cubin.size());
        if (!(actualDigest == kernel.digest))
        {
            throw std::invalid_argument("NVFP4-A16 Blackwell GEMV JIT cubin digest mismatch");
        }

        CUDA_CHECK(cudaFree(nullptr));
        CUcontext context{};
        CUDA_DRIVER_CHECK(cuCtxGetCurrent(&context));
        if (context == nullptr)
        {
            throw std::runtime_error("NVFP4-A16 Blackwell GEMV JIT module load requires a current CUDA context");
        }

        RegistryKey const registryKey{context, kernel.key, kernel.digest};
        std::lock_guard<std::mutex> lock(mMutex);
        auto const existing = mModules.find(registryKey);
        if (existing != mModules.end())
        {
            if (auto loaded = existing->second.lock())
            {
                return loaded;
            }
            mModules.erase(existing);
        }

        auto loaded = std::make_shared<Nvfp4A16BlackwellGemvLoadedModule>();
        loaded->context = context;
        CUDA_DRIVER_CHECK(cuModuleLoadData(&loaded->module, kernel.cubin.data()));
        try
        {
            for (size_t index = 0; index < kSUPPORTED_M.size(); ++index)
            {
                for (size_t splitIndex = 0; splitIndex < kSUPPORTED_SPLIT_K.size(); ++splitIndex)
                {
                    std::string const mSuffix = std::to_string(kSUPPORTED_M[index]);
                    std::string const splitSuffix = std::to_string(kSUPPORTED_SPLIT_K[splitIndex]);
                    std::string const mainName = "nvfp4_a16_blackwell_gemv_m" + mSuffix + "_sk" + splitSuffix;
                    std::string const reduceName = "nvfp4_a16_blackwell_gemv_reduce_m" + mSuffix + "_sk" + splitSuffix;
                    CUDA_DRIVER_CHECK(cuModuleGetFunction(
                        &loaded->mainFunctions[index][splitIndex], loaded->module, mainName.c_str()));
                    CUDA_DRIVER_CHECK(cuModuleGetFunction(
                        &loaded->reduceFunctions[index][splitIndex], loaded->module, reduceName.c_str()));
                }
            }
            CUDA_DRIVER_CHECK(cuModuleGetFunction(
                &loaded->runtimeSplitMainFunctions[0], loaded->module, "nvfp4_a16_blackwell_gemv_m8_runtime"));
            CUDA_DRIVER_CHECK(cuModuleGetFunction(
                &loaded->runtimeSplitMainFunctions[1], loaded->module, "nvfp4_a16_blackwell_gemv_m16_runtime"));
        }
        catch (...)
        {
            loaded.reset();
            throw;
        }
        mModules.emplace(registryKey, loaded);
        return loaded;
    }

private:
    std::mutex mMutex;
    std::unordered_map<RegistryKey, std::weak_ptr<Nvfp4A16BlackwellGemvLoadedModule>, RegistryKeyHasher> mModules;
};

KernelRegistry& getKernelRegistry()
{
    static KernelRegistry registry;
    return registry;
}

} // namespace

bool isNvfp4A16BlackwellGemvJitSupported(
    Nvfp4A16BlackwellGemvJitKey const& key, int32_t const m, int32_t const splitK) noexcept
{
    if (!canCompileNvfp4A16BlackwellGemvJitKernel(key) || !isSupportedM(m) || !isSupportedSplitK(splitK))
    {
        return false;
    }
    size_t workspaceSize{};
    uint32_t mainGridX{};
    uint32_t reduceGridX{};
    return getWorkspaceSizeChecked(key, m, splitK, workspaceSize) && getMainGridXChecked(key, mainGridX)
        && getReduceGridXChecked(key, m, splitK, reduceGridX);
}

size_t getNvfp4A16BlackwellGemvJitWorkspaceSize(
    Nvfp4A16BlackwellGemvJitKey const& key, int32_t const m, int32_t const splitK) noexcept
{
    if (!isNvfp4A16BlackwellGemvJitSupported(key, m, splitK) || splitK == 1)
    {
        return 0;
    }
    size_t workspaceSize{};
    return getWorkspaceSizeChecked(key, m, splitK, workspaceSize) ? workspaceSize : 0;
}

void Nvfp4A16BlackwellGemvJitRunner::load(Nvfp4A16BlackwellGemvJitKernel const& kernel)
{
    mLoadedModule = getKernelRegistry().load(kernel);
    mKey = kernel.key;
    mDigest = kernel.digest;
}

void Nvfp4A16BlackwellGemvJitRunner::launch(void const* const activation, uint8_t const* const qweights,
    uint8_t const* const blockScales, float const* const globalScale, void* const output, void* const workspace,
    size_t const workspaceSize, int32_t const m, int32_t const splitK, cudaStream_t const stream) const
{
    if (!isLoaded())
    {
        throw std::runtime_error(
            "NVFP4-A16 Blackwell GEMV JIT runner must be loaded before enqueue or CUDA Graph capture");
    }
    if (!isNvfp4A16BlackwellGemvJitSupported(mKey, m, splitK))
    {
        throw std::invalid_argument("Unsupported NVFP4-A16 Blackwell GEMV JIT launch configuration");
    }
    if (activation == nullptr || qweights == nullptr || blockScales == nullptr || globalScale == nullptr
        || output == nullptr)
    {
        throw std::invalid_argument("NVFP4-A16 Blackwell GEMV JIT launch received a null required argument");
    }

    size_t const requiredWorkspace = getNvfp4A16BlackwellGemvJitWorkspaceSize(mKey, m, splitK);
    if (requiredWorkspace > 0 && (workspace == nullptr || workspaceSize < requiredWorkspace))
    {
        throw std::invalid_argument("NVFP4-A16 Blackwell GEMV JIT split-K workspace is null or undersized");
    }
    if ((reinterpret_cast<uintptr_t>(activation) & 15U) != 0 || (reinterpret_cast<uintptr_t>(qweights) & 15U) != 0
        || (reinterpret_cast<uintptr_t>(blockScales) & 1U) != 0 || (reinterpret_cast<uintptr_t>(globalScale) & 3U) != 0
        || (reinterpret_cast<uintptr_t>(output) & 3U) != 0
        || (requiredWorkspace > 0 && (reinterpret_cast<uintptr_t>(workspace) & 3U) != 0))
    {
        throw std::invalid_argument("NVFP4-A16 Blackwell GEMV JIT buffers do not satisfy vector-load alignment");
    }

    CUcontext context{};
    CUDA_DRIVER_CHECK(cuCtxGetCurrent(&context));
    if (context != mLoadedModule->context)
    {
        throw std::runtime_error("NVFP4-A16 Blackwell GEMV JIT runner is attached to a different CUDA context");
    }

    size_t const mIndex = getMIndex(m);
    size_t const splitKIndex = getSplitKIndex(splitK);
    uint32_t gridX{};
    if (!getMainGridXChecked(mKey, gridX))
    {
        throw std::overflow_error("NVFP4-A16 Blackwell GEMV JIT main launch grid exceeds the CUDA grid limit");
    }
    uint32_t const gridY = static_cast<uint32_t>(splitK);
    void const* activationArg = activation;
    uint8_t const* qweightsArg = qweights;
    uint8_t const* blockScalesArg = blockScales;
    float const* globalScaleArg = globalScale;
    void* outputArg = output;
    void* workspaceArg = workspace;
    bool const useRuntimeSplitMain = splitK == 1 && m >= 8;
    if (useRuntimeSplitMain)
    {
        int32_t splitKArg = splitK;
        void* mainParams[]{
            &activationArg, &qweightsArg, &blockScalesArg, &globalScaleArg, &outputArg, &workspaceArg, &splitKArg};
        CUDA_DRIVER_CHECK(cuLaunchKernel(mLoadedModule->runtimeSplitMainFunctions[getRuntimeSplitMIndex(m)], gridX,
            gridY, 1U, kTHREADS_PER_BLOCK, 1U, 1U, 0U, stream, mainParams, nullptr));
    }
    else
    {
        void* mainParams[]{&activationArg, &qweightsArg, &blockScalesArg, &globalScaleArg, &outputArg, &workspaceArg};
        CUDA_DRIVER_CHECK(cuLaunchKernel(mLoadedModule->mainFunctions[mIndex][splitKIndex], gridX, gridY, 1U,
            kTHREADS_PER_BLOCK, 1U, 1U, 0U, stream, mainParams, nullptr));
    }

    if (splitK > 1)
    {
        uint32_t reduceBlocks{};
        if (!getReduceGridXChecked(mKey, m, splitK, reduceBlocks))
        {
            throw std::overflow_error("NVFP4-A16 Blackwell GEMV JIT reduction grid exceeds the CUDA grid limit");
        }
        void* reduceParams[]{&workspaceArg, &globalScaleArg, &outputArg};
        CUDA_DRIVER_CHECK(cuLaunchKernel(mLoadedModule->reduceFunctions[mIndex][splitKIndex], reduceBlocks, 1U, 1U,
            kREDUCE_THREADS, 1U, 1U, 0U, stream, reduceParams, nullptr));
    }
}

} // namespace trt_edgellm
