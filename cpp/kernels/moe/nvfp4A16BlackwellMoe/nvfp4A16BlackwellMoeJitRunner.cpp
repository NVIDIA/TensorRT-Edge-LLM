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
#include "nvfp4A16BlackwellMoeJitRunner.h"

#include "common/checkMacros.h"
#include "common/cudaMacros.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include <array>
#include <cstdint>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <type_traits>
#include <unordered_map>

namespace trt_edgellm
{

struct Nvfp4A16BlackwellMoeLoadedModule
{
    ~Nvfp4A16BlackwellMoeLoadedModule()
    {
        if (module != nullptr)
        {
            (void) cuModuleUnload(module);
        }
    }

    CUcontext context{};
    CUmodule module{};
    CUfunction route{};
    CUfunction layout{};
    CUfunction gather{};
    CUfunction fc1{};
    CUfunction fc1Reduce{};
    CUfunction fc2{};
    CUfunction fc2Reduce{};
};

namespace
{

constexpr uint32_t kTHREADS_PER_BLOCK{256U};
constexpr uint32_t kLAYOUT_THREADS{1024U};
constexpr uint32_t kWARPS_PER_BLOCK{8U};
constexpr int32_t kN_TILE{128};

struct RegistryKey
{
    CUcontext context{};
    Nvfp4A16BlackwellMoeJitKey jitKey{};
    Nvfp4A16BlackwellMoeJitDigest digest{};

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
        size_t hash{0xCBF29CE484222325ULL};
        hash = mix(hash, reinterpret_cast<uintptr_t>(key.context));
        hash = mix(hash, static_cast<size_t>(key.jitKey.numExperts));
        hash = mix(hash, static_cast<size_t>(key.jitKey.topK));
        hash = mix(hash, static_cast<size_t>(key.jitKey.hiddenSize));
        hash = mix(hash, static_cast<size_t>(key.jitKey.interSize));
        hash = mix(hash, static_cast<size_t>(key.jitKey.fc1SplitK));
        hash = mix(hash, static_cast<size_t>(key.jitKey.fc2SplitK));
        hash = mix(hash, static_cast<size_t>(key.jitKey.fc2PrefetchSlots));
        hash = mix(hash, static_cast<size_t>(static_cast<uint32_t>(key.jitKey.dataType)));
        hash = mix(hash, static_cast<size_t>(key.digest.lo));
        return mix(hash, static_cast<size_t>(key.digest.hi));
    }
};

class KernelRegistry
{
public:
    std::shared_ptr<Nvfp4A16BlackwellMoeLoadedModule> load(Nvfp4A16BlackwellMoeJitKernel const& kernel)
    {
        if (!canCompileNvfp4A16BlackwellMoeJitKernel(kernel.key) || kernel.cubin.empty())
        {
            throw std::invalid_argument("Invalid NVFP4-A16 Blackwell MoE JIT key or cubin");
        }
        Nvfp4A16BlackwellMoeJitDigest const actualDigest
            = computeNvfp4A16BlackwellMoeJitDigest(kernel.key, kernel.cubin.data(), kernel.cubin.size());
        if (!(actualDigest == kernel.digest))
        {
            throw std::invalid_argument("NVFP4-A16 Blackwell MoE JIT cubin digest mismatch");
        }
        CUDA_CHECK(cudaFree(nullptr));
        CUcontext context{};
        CUDA_DRIVER_CHECK(cuCtxGetCurrent(&context));
        if (context == nullptr)
        {
            throw std::runtime_error("NVFP4-A16 Blackwell MoE JIT module load requires a current CUDA context");
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
        auto loaded = std::make_shared<Nvfp4A16BlackwellMoeLoadedModule>();
        loaded->context = context;
        CUDA_DRIVER_CHECK(cuModuleLoadData(&loaded->module, kernel.cubin.data()));
        try
        {
            CUDA_DRIVER_CHECK(cuModuleGetFunction(&loaded->route, loaded->module, "nvfp4_a16_blackwell_moe_route"));
            CUDA_DRIVER_CHECK(cuModuleGetFunction(&loaded->layout, loaded->module, "nvfp4_a16_blackwell_moe_layout"));
            CUDA_DRIVER_CHECK(cuModuleGetFunction(&loaded->gather, loaded->module, "nvfp4_a16_blackwell_moe_gather"));
            CUDA_DRIVER_CHECK(cuModuleGetFunction(&loaded->fc1, loaded->module, "nvfp4_a16_blackwell_moe_fc1"));
            CUDA_DRIVER_CHECK(
                cuModuleGetFunction(&loaded->fc1Reduce, loaded->module, "nvfp4_a16_blackwell_moe_fc1_reduce"));
            CUDA_DRIVER_CHECK(cuModuleGetFunction(&loaded->fc2, loaded->module, "nvfp4_a16_blackwell_moe_fc2"));
            CUDA_DRIVER_CHECK(
                cuModuleGetFunction(&loaded->fc2Reduce, loaded->module, "nvfp4_a16_blackwell_moe_fc2_reduce"));
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
    std::unordered_map<RegistryKey, std::weak_ptr<Nvfp4A16BlackwellMoeLoadedModule>, RegistryKeyHasher> mModules;
};

KernelRegistry& getKernelRegistry()
{
    static KernelRegistry registry;
    return registry;
}

//! Driver-API launch with the programmatic-stream-serialization attribute when
//! requested (and supported by the toolchain); plain launch otherwise.
void launchKernel(CUfunction const function, uint32_t const gridX, uint32_t const gridY, uint32_t const gridZ,
    uint32_t const block, void** const params, bool const enablePdl, cudaStream_t const stream)
{
#if SUPPORTS_PROGRAMMATIC_DEPENDENT_LAUNCH
    if (enablePdl)
    {
        CUlaunchAttribute attribute{};
        attribute.id = CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION;
        attribute.value.programmaticStreamSerializationAllowed = 1;
        CUlaunchConfig config{};
        config.gridDimX = gridX;
        config.gridDimY = gridY;
        config.gridDimZ = gridZ;
        config.blockDimX = block;
        config.blockDimY = 1U;
        config.blockDimZ = 1U;
        config.sharedMemBytes = 0U;
        config.hStream = stream;
        config.attrs = &attribute;
        config.numAttrs = 1U;
        CUDA_DRIVER_CHECK(cuLaunchKernelEx(&config, function, params, nullptr));
        return;
    }
#else
    (void) enablePdl;
#endif
    CUDA_DRIVER_CHECK(cuLaunchKernel(function, gridX, gridY, gridZ, block, 1U, 1U, 0U, stream, params, nullptr));
}

uint32_t checkedGrid(int64_t const blocks, char const* const what)
{
    if (blocks <= 0 || blocks > static_cast<int64_t>(std::numeric_limits<int32_t>::max()))
    {
        throw std::overflow_error(std::string("NVFP4-A16 Blackwell MoE JIT ") + what + " grid is out of range");
    }
    return static_cast<uint32_t>(blocks);
}

uint32_t pairBlocks(int64_t const elements)
{
    return checkedGrid((elements / 2 + kTHREADS_PER_BLOCK - 1) / kTHREADS_PER_BLOCK, "reduce");
}

} // namespace

void Nvfp4A16BlackwellMoeJitRunner::load(Nvfp4A16BlackwellMoeJitKernel const& kernel)
{
    mLoadedModule = getKernelRegistry().load(kernel);
    mKey = kernel.key;
    mDigest = kernel.digest;
}

namespace
{
void requireLoaded(std::shared_ptr<Nvfp4A16BlackwellMoeLoadedModule> const& module)
{
    if (module == nullptr)
    {
        throw std::runtime_error(
            "NVFP4-A16 Blackwell MoE JIT runner must be loaded before enqueue or CUDA Graph capture");
    }
}
} // namespace

void Nvfp4A16BlackwellMoeJitRunner::launchRoute(float const* const logits, float const* const correctionBias,
    int32_t const numTokens, bool const normTopkProb, float const routedScalingFactor, int32_t* const topkIndices,
    float* const topkWeights, bool const enablePdl, cudaStream_t const stream) const
{
    requireLoaded(mLoadedModule);
    if (logits == nullptr || topkIndices == nullptr || topkWeights == nullptr || numTokens <= 0)
    {
        throw std::invalid_argument("NVFP4-A16 Blackwell MoE JIT routing launch received an invalid argument");
    }
    float const* logitsArg = logits;
    float const* biasArg = correctionBias;
    int32_t tokensArg = numTokens;
    int32_t normArg = normTopkProb ? 1 : 0;
    float scaleArg = routedScalingFactor;
    int32_t* indicesArg = topkIndices;
    float* weightsArg = topkWeights;
    void* params[]{&logitsArg, &biasArg, &tokensArg, &normArg, &scaleArg, &indicesArg, &weightsArg};
    uint32_t const grid
        = checkedGrid((static_cast<int64_t>(numTokens) + kWARPS_PER_BLOCK - 1) / kWARPS_PER_BLOCK, "routing");
    launchKernel(mLoadedModule->route, grid, 1U, 1U, kTHREADS_PER_BLOCK, params, enablePdl, stream);
}

void Nvfp4A16BlackwellMoeJitRunner::launchLayout(int32_t const* const topkIndices, int32_t const numSlots,
    int32_t const tokenTile, int32_t* const permutedIdx, int32_t* const tileGroupIdx, int32_t* const numValidTiles,
    bool const enablePdl, cudaStream_t const stream) const
{
    requireLoaded(mLoadedModule);
    if (topkIndices == nullptr || permutedIdx == nullptr || tileGroupIdx == nullptr || numValidTiles == nullptr
        || numSlots < 0 || tokenTile <= 0)
    {
        throw std::invalid_argument("NVFP4-A16 Blackwell MoE JIT layout launch received an invalid argument");
    }
    int32_t const* indicesArg = topkIndices;
    int32_t slotsArg = numSlots;
    int32_t tileArg = tokenTile;
    int32_t* permutedArg = permutedIdx;
    int32_t* groupArg = tileGroupIdx;
    int32_t* validArg = numValidTiles;
    void* params[]{&indicesArg, &slotsArg, &tileArg, &permutedArg, &groupArg, &validArg};
    launchKernel(mLoadedModule->layout, 1U, 1U, 1U, kLAYOUT_THREADS, params, enablePdl, stream);
}

void Nvfp4A16BlackwellMoeJitRunner::launchGather(void const* const hiddenStates, int32_t const* const permutedIdx,
    int32_t const* const numValidTiles, int32_t const tokenTile, int32_t const numTokens, int64_t const maxRowsPadded,
    void* const permutedActivations, void* const output, bool const enablePdl, cudaStream_t const stream) const
{
    requireLoaded(mLoadedModule);
    if (hiddenStates == nullptr || permutedIdx == nullptr || numValidTiles == nullptr || permutedActivations == nullptr
        || output == nullptr || tokenTile <= 0 || numTokens <= 0 || maxRowsPadded <= 0)
    {
        throw std::invalid_argument("NVFP4-A16 Blackwell MoE JIT gather launch received an invalid argument");
    }
    void const* hiddenArg = hiddenStates;
    int32_t const* permutedIdxArg = permutedIdx;
    int32_t const* validArg = numValidTiles;
    int32_t tileArg = tokenTile;
    int32_t tokensArg = numTokens;
    int64_t rowsArg = maxRowsPadded;
    void* permutedArg = permutedActivations;
    void* outputArg = output;
    void* params[]{&hiddenArg, &permutedIdxArg, &validArg, &tileArg, &tokensArg, &rowsArg, &permutedArg, &outputArg};
    uint32_t const grid = checkedGrid(maxRowsPadded + numTokens, "gather");
    launchKernel(mLoadedModule->gather, grid, 1U, 1U, kTHREADS_PER_BLOCK, params, enablePdl, stream);
}

void Nvfp4A16BlackwellMoeJitRunner::launchFc1(void const* const hiddenStates, int32_t const* const topkIndices,
    void const* const qweights, void const* const blockScales, float const* const globalScales, void* const fc1Output,
    float* const partials, int32_t const numTokens, bool const enablePdl, cudaStream_t const stream) const
{
    requireLoaded(mLoadedModule);
    if (hiddenStates == nullptr || topkIndices == nullptr || qweights == nullptr || blockScales == nullptr
        || globalScales == nullptr || fc1Output == nullptr || numTokens <= 0
        || (mKey.fc1SplitK > 1 && partials == nullptr))
    {
        throw std::invalid_argument("NVFP4-A16 Blackwell MoE JIT FC1 launch received an invalid argument");
    }
    void const* hiddenArg = hiddenStates;
    int32_t const* indicesArg = topkIndices;
    void const* qweightsArg = qweights;
    void const* scalesArg = blockScales;
    float const* globalArg = globalScales;
    void* outputArg = fc1Output;
    float* partialsArg = partials;
    int32_t tokensArg = numTokens;
    void* params[]{&hiddenArg, &indicesArg, &qweightsArg, &scalesArg, &globalArg, &outputArg, &partialsArg, &tokensArg};
    uint32_t const gridX = checkedGrid(mKey.interSizePadded / kN_TILE, "FC1");
    uint32_t const gridY = checkedGrid(static_cast<int64_t>(numTokens) * mKey.topK, "FC1 slots");
    launchKernel(mLoadedModule->fc1, gridX, gridY, static_cast<uint32_t>(mKey.fc1SplitK), kTHREADS_PER_BLOCK, params,
        enablePdl, stream);
    if (mKey.fc1SplitK > 1)
    {
        void* reduceParams[]{&indicesArg, &globalArg, &partialsArg, &outputArg, &tokensArg};
        uint32_t const blocks = pairBlocks(static_cast<int64_t>(numTokens) * mKey.topK * mKey.interSizePadded);
        launchKernel(mLoadedModule->fc1Reduce, blocks, 1U, 1U, kTHREADS_PER_BLOCK, reduceParams, enablePdl, stream);
    }
}

void Nvfp4A16BlackwellMoeJitRunner::launchFc2(void const* const fc1Output, int32_t const* const topkIndices,
    float const* const topkWeights, void const* const qweights, void const* const blockScales,
    float const* const globalScales, void* const output, float* const partials, int32_t const numTokens,
    bool const enablePdl, cudaStream_t const stream) const
{
    requireLoaded(mLoadedModule);
    if (fc1Output == nullptr || topkIndices == nullptr || topkWeights == nullptr || qweights == nullptr
        || blockScales == nullptr || globalScales == nullptr || output == nullptr || numTokens <= 0
        || (mKey.fc2SplitK > 1 && partials == nullptr))
    {
        throw std::invalid_argument("NVFP4-A16 Blackwell MoE JIT FC2 launch received an invalid argument");
    }
    void const* activationArg = fc1Output;
    int32_t const* indicesArg = topkIndices;
    float const* weightsArg = topkWeights;
    void const* qweightsArg = qweights;
    void const* scalesArg = blockScales;
    float const* globalArg = globalScales;
    void* outputArg = output;
    float* partialsArg = partials;
    int32_t tokensArg = numTokens;
    void* params[]{&activationArg, &indicesArg, &weightsArg, &qweightsArg, &scalesArg, &globalArg, &outputArg,
        &partialsArg, &tokensArg};
    uint32_t const gridX = checkedGrid(mKey.hiddenSize / kN_TILE, "FC2");
    uint32_t const gridY = checkedGrid(numTokens, "FC2 tokens");
    launchKernel(mLoadedModule->fc2, gridX, gridY, static_cast<uint32_t>(mKey.fc2SplitK), kTHREADS_PER_BLOCK, params,
        enablePdl, stream);
    if (mKey.fc2SplitK > 1)
    {
        void* reduceParams[]{&partialsArg, &outputArg, &tokensArg};
        uint32_t const blocks = pairBlocks(static_cast<int64_t>(numTokens) * mKey.hiddenSize);
        launchKernel(mLoadedModule->fc2Reduce, blocks, 1U, 1U, kTHREADS_PER_BLOCK, reduceParams, enablePdl, stream);
    }
}

} // namespace trt_edgellm
