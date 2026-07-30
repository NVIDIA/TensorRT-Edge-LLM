/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "runtime/state/loraManager.h"

#include "common/bindingNames.h"
#include "common/checkMacros.h"
#include "common/logger.h"
#include "common/safetensorsUtils.h"
#include "runtime/exec/engineExecutor.h"

#include <NvInferRuntime.h>
#include <limits>
#include <set>
#include <stdexcept>

namespace trt_edgellm
{
namespace rt
{

namespace
{

void advanceAdapterGeneration(std::map<std::string, uint64_t>& generations, std::string const& name)
{
    uint64_t const current = generations[name];
    ELLM_CHECK(
        current < std::numeric_limits<uint64_t>::max(), "LoRAManager: adapter generation exhausted for '" + name + "'");
    generations[name] = current + 1;
}

//! Enumerate LoRA weight tensor names from an engine's I/O bindings.
std::vector<std::string> collectLoraWeightsTensorNames(nvinfer1::ICudaEngine const& engine)
{
    std::vector<std::string> loraTensorNames;
    int32_t const numIO = engine.getNbIOTensors();
    for (int32_t i = 0; i < numIO; ++i)
    {
        std::string const name(engine.getIOTensorName(i));
        if (binding_names::isLoraBinding(name))
        {
            loraTensorNames.push_back(name);
        }
    }
    return loraTensorNames;
}

} // namespace

void LoRAManager::loadWeights(std::string const& name, std::filesystem::path const& path, cudaStream_t stream)
{
    LOG_INFO("LoRAManager: loading adapter '%s' from %s", name.c_str(), path.string().c_str());

    std::vector<rt::Tensor> tensors;
    bool const ok = safetensors::loadSafetensors(path, tensors, stream);
    ELLM_CHECK(ok, "LoRAManager: failed to load safetensors from " + path.string());

    WeightMap weights;
    for (auto& t : tensors)
    {
        std::string bindingName = t.getName();
        weights.emplace(std::move(bindingName), std::move(t));
    }

    LOG_INFO("LoRAManager: adapter '%s' loaded with %zu binding(s)", name.c_str(), weights.size());

    mAdapters[name] = std::move(weights);
    advanceAdapterGeneration(mAdapterGenerations, name);
}

void LoRAManager::addWeights(std::string const& name, std::map<std::string, rt::Tensor> weights)
{
    LOG_INFO("LoRAManager: adding adapter '%s' with %zu binding(s)", name.c_str(), weights.size());
    mAdapters[name] = std::move(weights);
    advanceAdapterGeneration(mAdapterGenerations, name);
}

void LoRAManager::switchWeights(std::string const& name)
{
    auto const it = mAdapters.find(name);
    ELLM_CHECK(it != mAdapters.end(), "LoRAManager: adapter '" + name + "' not found");
    mActiveAdapterName = name;
    LOG_INFO("LoRAManager: switched to adapter '%s'", name.c_str());
}

void LoRAManager::resetWeights()
{
    mActiveAdapterName.clear();
    LOG_INFO("LoRAManager: adapter deactivated");
}

rt::Tensor& LoRAManager::getActiveWeight(std::string const& bindingName)
{
    if (mActiveAdapterName.empty())
    {
        // Lazy-initialize dummy tensor on first use. Zero-fill so any TRT
        // engine that reads it (rank=1 LoRA contraction on inactive adapters)
        // sees deterministic zeros rather than uninitialized GPU memory.
        if (mDummyTensor.isEmpty())
        {
            mDummyTensor = rt::Tensor({1}, DeviceType::kGPU, nvinfer1::DataType::kHALF, "LoRAManager::dummy");
            CUDA_CHECK(cudaMemset(mDummyTensor.rawPointer(), 0, mDummyTensor.getMemoryCapacity()));
        }
        return mDummyTensor;
    }

    auto const adapterIt = mAdapters.find(mActiveAdapterName);
    ELLM_CHECK(adapterIt != mAdapters.end(),
        "LoRAManager: active adapter '" + mActiveAdapterName + "' disappeared unexpectedly");

    auto const weightIt = adapterIt->second.find(bindingName);
    ELLM_CHECK(weightIt != adapterIt->second.end(),
        "LoRAManager: binding '" + bindingName + "' not found in adapter '" + mActiveAdapterName + "'");

    return weightIt->second;
}

std::string const& LoRAManager::getActiveAdapterName() const noexcept
{
    return mActiveAdapterName;
}

uint64_t LoRAManager::getAdapterGeneration(std::string const& name) const
{
    auto const generation = mAdapterGenerations.find(name);
    ELLM_CHECK(generation != mAdapterGenerations.end(), "LoRAManager: adapter '" + name + "' not found");
    return generation->second;
}

uint64_t LoRAManager::getActiveAdapterGeneration() const
{
    return mActiveAdapterName.empty() ? 0U : getAdapterGeneration(mActiveAdapterName);
}

std::vector<std::string> LoRAManager::getBindingNames() const
{
    std::set<std::string> names;
    for (auto const& [adapterName, weightMap] : mAdapters)
    {
        for (auto const& [bindingName, tensor] : weightMap)
        {
            names.insert(bindingName);
        }
    }
    return {names.begin(), names.end()};
}

std::vector<std::string> LoRAManager::getAdapterNames() const
{
    std::vector<std::string> names;
    names.reserve(mAdapters.size());
    for (auto const& [adapterName, weightMap] : mAdapters)
    {
        names.push_back(adapterName);
    }
    return names;
}

bool LoRAManager::hasActiveAdapter() const noexcept
{
    return !mActiveAdapterName.empty();
}

bool LoRAManager::hasWeightFor(std::string const& bindingName) const noexcept
{
    if (mActiveAdapterName.empty())
    {
        return false;
    }
    auto const adapterIt = mAdapters.find(mActiveAdapterName);
    if (adapterIt == mAdapters.end())
    {
        return false;
    }
    return adapterIt->second.find(bindingName) != adapterIt->second.end();
}

void LoRAManager::initializeEngineBindings(EngineExecutor const& runner)
{
    constexpr int32_t kEmptyLoraRank = 1;

    std::vector<std::string> names = collectLoraWeightsTensorNames(runner.getEngine());
    std::map<std::string, rt::Tensor> dummyTensors;

    for (auto const& name : names)
    {
        nvinfer1::Dims const maxShape
            = runner.getProfileShape(name.c_str(), /*profileIndex=*/0, nvinfer1::OptProfileSelector::kMAX);
        nvinfer1::Dims dummyShape = maxShape;
        if (name.find(binding_names::kLoraAPrefix) != std::string::npos)
        {
            // LoRA A: [k, rank] → set rank (last dim) to 1
            dummyShape.d[dummyShape.nbDims - 1] = kEmptyLoraRank;
        }
        else if (name.find(binding_names::kLoraBPrefix) != std::string::npos)
        {
            // LoRA B: [rank, n] → set rank (first dim) to 1
            dummyShape.d[0] = kEmptyLoraRank;
        }
        std::vector<int64_t> shape(dummyShape.nbDims);
        for (int32_t d = 0; d < dummyShape.nbDims; ++d)
        {
            shape[d] = dummyShape.d[d];
        }
        // Zero-fill: when no adapter is active these dummies are bound directly
        // to the engine's lora_A_*/lora_B_* inputs, so the rank-1 contraction
        // must read deterministic zeros, not uninitialized cudaMalloc memory.
        auto [it, inserted] = dummyTensors.emplace(
            name, rt::Tensor(shape, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF, "loraDummy_" + name));
        CUDA_CHECK(cudaMemset(it->second.rawPointer(), 0, it->second.getMemoryCapacity()));
    }

    mEngineBindingNames = std::move(names);
    mEngineDummyTensors = std::move(dummyTensors);
    LOG_INFO("LoRAManager: registered %zu engine LoRA binding name(s)", mEngineBindingNames.size());
}

void LoRAManager::refreshTensorMap(TensorMap& map)
{
    for (auto const& engineBindingName : mEngineBindingNames)
    {
        // Exact name match works for non-fused projections. Fused engines use
        // a different naming convention (e.g. `qkv_proj.*` vs the adapter's
        // separate `q_proj.*` / `k_proj.*` / `v_proj.*`); for those bindings
        // we fall back to the per-binding dummy tensor.
        if (hasWeightFor(engineBindingName))
        {
            map.set(engineBindingName, getActiveWeight(engineBindingName));
            continue;
        }

        auto it = mEngineDummyTensors.find(engineBindingName);
        if (it != mEngineDummyTensors.end())
        {
            map.set(engineBindingName, it->second);
        }
    }
}

} // namespace rt
} // namespace trt_edgellm
