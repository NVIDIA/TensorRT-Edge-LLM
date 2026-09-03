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

#include "runtime/exec/engineExecutor.h"

#include "common/bindingNames.h"
#include "common/checkMacros.h"
#include "common/hashUtils.h"
#include "common/logger.h"
#include "common/trtUtils.h"
#include "runtime/exec/registryBuilder.h"
#include <stdexcept>
#include <string_view>

namespace trt_edgellm
{
namespace rt
{
namespace
{
bool engineHasIOTensor(nvinfer1::ICudaEngine const& engine, char const* tensorName)
{
    if (tensorName == nullptr)
    {
        return false;
    }

    int32_t const numIO = engine.getNbIOTensors();
    for (int32_t i = 0; i < numIO; ++i)
    {
        char const* const name = engine.getIOTensorName(i);
        if (name != nullptr && std::string_view{name} == tensorName)
        {
            return true;
        }
    }
    return false;
}

bool engineHasInputTensor(nvinfer1::ICudaEngine const& engine, char const* tensorName)
{
    if (tensorName == nullptr)
    {
        return false;
    }

    int32_t const numIO = engine.getNbIOTensors();
    for (int32_t i = 0; i < numIO; ++i)
    {
        char const* const name = engine.getIOTensorName(i);
        if (name != nullptr && std::string_view{name} == tensorName)
        {
            return engine.getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT;
        }
    }
    return false;
}
} // namespace

//! TRT-backed EngineExecutor: owns an IRuntime, ICudaEngine, and IExecutionContext plus the CUDA-graph cache.
//! Private to this translation unit -- callers obtain it only through EngineExecutor's factories.
class TrtEngineExecutor final : public EngineExecutor
{
public:
    /*!
     * @brief Construct from a serialized TRT engine file.
     *
     * Reads the engine, creates an IRuntime, deserializes the engine, and creates an IExecutionContext with
     * USER_MANAGED allocation.
     *
     * @throws std::runtime_error On I/O or deserialization failure
     */
    TrtEngineExecutor(std::filesystem::path const& enginePath, TensorRegistry registry);

    //! @brief Destructor -- destroys all captured CUDA graphs.
    ~TrtEngineExecutor() noexcept override;

    bool prepare(int32_t profileIndex, InferenceDims const& dims, TensorMap const& map, cudaStream_t stream) override;
    bool execute(cudaStream_t stream) override;
    bool captureGraph(cudaStream_t stream) override;
    int64_t getRequiredContextMemorySize() const override;
    bool setContextMemory(Tensor& sharedMem) override;
    int32_t getNumIOTensors() const override;
    char const* getIOTensorName(int32_t index) const override;
    bool hasIOTensor(char const* name) const override;
    nvinfer1::DataType getBindingDataType(char const* name) const override;
    nvinfer1::Dims getProfileShape(
        char const* name, int32_t profileIndex, nvinfer1::OptProfileSelector selector) const override;
    void setProfiler(nvinfer1::IProfiler* profiler) noexcept override;
    nvinfer1::ICudaEngine const& getEngine() const noexcept override;

private:
    AuxStreamSet mAuxStreams{};
    std::unique_ptr<nvinfer1::IRuntime> mRuntime;
    std::unique_ptr<nvinfer1::ICudaEngine> mEngine;
    std::unique_ptr<nvinfer1::IExecutionContext> mContext;
    TensorRegistry mRegistry;
    int32_t mCurrentProfileIndex{-1};

    //! A captured CUDA graph together with its binding snapshot for verification.
    struct CapturedGraph
    {
        cudaGraph_t graph{nullptr};
        cudaGraphExec_t exec{nullptr};
        BindingSnapshot snapshot;
    };

    //! Graph cache keyed by a hash of all binding addresses + shapes.
    std::unordered_map<size_t, CapturedGraph> mGraphs;

    //! Hash the current binding addresses and shapes into a single key.
    size_t computeBindingHash() const;

    //! Build a full snapshot of the current binding state.
    BindingSnapshot snapshotBindings() const;
};

TrtEngineExecutor::TrtEngineExecutor(std::filesystem::path const& enginePath, TensorRegistry registry)
    : mRegistry(std::move(registry))
{
    LOG_INFO("loading engine file: %s", enginePath.string().c_str());

    mRuntime = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(gLogger));
    ELLM_CHECK(mRuntime != nullptr, "failed to create TRT IRuntime");

    mEngine = deserializeCudaEngineFromFile(*mRuntime, enginePath);

    // Use USER_MANAGED allocation so context memory can be shared across runners.
    mContext = std::unique_ptr<nvinfer1::IExecutionContext>(
        mEngine->createExecutionContext(nvinfer1::ExecutionContextAllocationStrategy::kUSER_MANAGED));
    ELLM_CHECK(mContext != nullptr, "failed to create execution context");

    setNonBlockingAuxStreams(mContext.get(), mEngine.get(), mAuxStreams);

    auto registerOptionalTreeMetadata = [&](char const* name) {
        if (engineHasInputTensor(*mEngine, name) && !mRegistry.contains(name))
        {
            mRegistry.addTensor({name, TensorIO::kInput, nvinfer1::DataType::kINT32,
                {sym(&InferenceDims::batch), sym(&InferenceDims::attnMaskSeqLen)}});
        }
    };
    registerOptionalTreeMetadata(binding_names::kTreeParentIds);
    registerOptionalTreeMetadata(binding_names::kTreeDepths);

    if (engineHasInputTensor(*mEngine, binding_names::kSkipSoftmaxScale)
        && !mRegistry.contains(binding_names::kSkipSoftmaxScale))
    {
        mRegistry.addTensor({binding_names::kSkipSoftmaxScale, TensorIO::kInput, nvinfer1::DataType::kINT8,
            {sym(&InferenceDims::skipSoftmaxScaleLen)}});
    }

    LOG_INFO("engine loaded successfully (%d I/O tensors)", mEngine->getNbIOTensors());
}

std::unique_ptr<EngineExecutor> EngineExecutor::createForLLM(std::filesystem::path const& enginePath,
    LLMEngineConfig const& cfg, std::optional<int32_t> specDecodeBaseOutputHiddenDim)
{
    auto registry = buildRegistryForLLM(cfg, specDecodeBaseOutputHiddenDim);
    return std::unique_ptr<EngineExecutor>(new TrtEngineExecutor(enginePath, std::move(registry)));
}

std::unique_ptr<EngineExecutor> EngineExecutor::createForDraft(
    std::filesystem::path const& enginePath, DeploymentConfig const& bundle)
{
    switch (bundle.specDecodeMode())
    {
    case SpecDecodeMode::kMTP:
    case SpecDecodeMode::kEAGLE:
    {
        auto registry = buildRegistryForSpecDecodeDraft(bundle);
        return std::unique_ptr<EngineExecutor>(new TrtEngineExecutor(enginePath, std::move(registry)));
    }
    case SpecDecodeMode::kDFlash:
    case SpecDecodeMode::kJetSpec:
    {
        auto registry = buildRegistryForDFlashDraft(bundle);
        return std::unique_ptr<EngineExecutor>(new TrtEngineExecutor(enginePath, std::move(registry)));
    }
    case SpecDecodeMode::kGemma4MTP:
    {
        auto registry = buildRegistryForGemma4MTPDraft(bundle);
        return std::unique_ptr<EngineExecutor>(new TrtEngineExecutor(enginePath, std::move(registry)));
    }
    case SpecDecodeMode::kDSpark:
    {
        auto registry = buildRegistryForDSparkDraft(bundle);
        return std::unique_ptr<EngineExecutor>(new TrtEngineExecutor(enginePath, std::move(registry)));
    }
    case SpecDecodeMode::kNONE:
    default: ELLM_CHECK(false, "createForDraft requires a speculative decoding deployment with a draft engine.");
    }
    return nullptr;
}

TrtEngineExecutor::~TrtEngineExecutor() noexcept
{
    for (auto& [hash, cg] : mGraphs)
    {
        if (cg.exec)
        {
            cudaGraphExecDestroy(cg.exec);
        }
        if (cg.graph)
        {
            cudaGraphDestroy(cg.graph);
        }
    }
    mGraphs.clear();
}

bool TrtEngineExecutor::prepare(
    int32_t profileIndex, InferenceDims const& dims, TensorMap const& map, cudaStream_t stream)
{
    if (auto bad = firstInvalidMember(dims, mRegistry.referencedMembers()); bad != nullptr)
    {
        auto const name = dimName(bad);
        LOG_ERROR(
            "EngineExecutor::prepare: InferenceDims.%.*s = %lld (must be > 0); "
            "caller likely bypassed an LLMEngineConfig recipe method",
            static_cast<int>(name.size()), name.data(), static_cast<long long>(dims.*bad));
        return false;
    }

    if (mCurrentProfileIndex != profileIndex && !mContext->setOptimizationProfileAsync(profileIndex, stream))
    {
        LOG_ERROR("failed to set optimization profile %d", profileIndex);
        return false;
    }
    mCurrentProfileIndex = profileIndex;

    if (!mRegistry.bindAll(mContext.get(), map, dims))
    {
        LOG_ERROR("bindAll failed for profile %d", profileIndex);
        return false;
    }

    // Bind any TensorMap entries not covered by the registry. We rebind
    // address+shape on every call (not just when getTensorAddress is null)
    // because TRT does NOT clear bindings on setOptimizationProfileAsync —
    // shapes and addresses persist across profile switches and across
    // execute() calls. Unregistered tensors that change shape per step
    // (notably kvcache_start_index, which uses [0] for initial prefill and
    // [batch] for chunked prefill / decode) would otherwise stick at
    // whatever shape was set by the previous step.
    //
    // LoRA weights are model-dependent and populated into the TensorMap by
    // LoRAManager::refreshTensorMap() before prepare() is called.
    int32_t const numIO = mEngine->getNbIOTensors();
    for (int32_t i = 0; i < numIO; ++i)
    {
        char const* name = mEngine->getIOTensorName(i);
        if (mRegistry.contains(name))
        {
            // Already bound by bindAll above; leave alone.
            continue;
        }
        Tensor* tensor = map.get(name);
        if (tensor == nullptr)
        {
            LOG_ERROR(
                "engine binding '%s' is neither registry-bound nor present in the TensorMap; "
                "every engine I/O tensor must be handled by the registry, by LoRAManager, or by an explicit "
                "TensorMap entry. Most likely the runtime is missing a binding for a model variant.",
                name);
            return false;
        }
        if (!mContext->setTensorAddress(name, tensor->rawPointer()))
        {
            LOG_ERROR("setTensorAddress failed for binding '%s'", name);
            return false;
        }
        if (mEngine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT)
        {
            if (!mContext->setInputShape(name, tensor->getTRTDims()))
            {
                LOG_ERROR("setInputShape failed for binding '%s'", name);
                return false;
            }
        }
    }

    return true;
}

bool TrtEngineExecutor::execute(cudaStream_t stream)
{
    size_t const hash = computeBindingHash();
    auto it = mGraphs.find(hash);
    if (it != mGraphs.end())
    {
        BindingSnapshot const current = snapshotBindings();
        if (current == it->second.snapshot)
        {
            cudaError_t const err = cudaGraphLaunch(it->second.exec, stream);
            if (err == cudaSuccess)
            {
                return true;
            }
            LOG_WARNING("cudaGraphLaunch failed (%s), falling back to enqueueV3", cudaGetErrorString(err));
        }
    }

    return mContext->enqueueV3(stream);
}

bool TrtEngineExecutor::captureGraph(cudaStream_t stream)
{
    // Warmup: run one enqueue to ensure all internal TRT state is initialized.
    if (!mContext->enqueueV3(stream))
    {
        LOG_ERROR("warmup enqueueV3 failed");
        return false;
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    auto result = captureTRTCudaGraph(mContext.get(), stream);
    if (!result.has_value())
    {
        LOG_WARNING("CUDA graph capture failed");
        return false;
    }

    size_t const hash = computeBindingHash();
    BindingSnapshot const snap = snapshotBindings();

    // If there was a previous graph for this hash, destroy it first.
    auto it = mGraphs.find(hash);
    if (it != mGraphs.end())
    {
        if (it->second.exec)
        {
            cudaGraphExecDestroy(it->second.exec);
        }
        if (it->second.graph)
        {
            cudaGraphDestroy(it->second.graph);
        }
    }

    CapturedGraph cg{};
    cg.graph = result->first;
    cg.exec = result->second;
    cg.snapshot = snap;
    mGraphs[hash] = cg;

    LOG_INFO("captured graph (hash=0x%zx)", hash);
    return true;
}

int64_t TrtEngineExecutor::getRequiredContextMemorySize() const
{
    // Use getDeviceMemorySizeV2() to get the max across ALL profiles.
    // SpecDecode base engines have multiple profiles (prefill + verification) with
    // different memory requirements. Using per-profile size can underallocate.
    return mEngine->getDeviceMemorySizeV2();
}

bool TrtEngineExecutor::setContextMemory(Tensor& sharedMem)
{
    mContext->setDeviceMemoryV2(sharedMem.rawPointer(), sharedMem.getMemoryCapacity());
    return true;
}

int32_t TrtEngineExecutor::getNumIOTensors() const
{
    return mEngine->getNbIOTensors();
}

char const* TrtEngineExecutor::getIOTensorName(int32_t index) const
{
    return mEngine->getIOTensorName(index);
}

bool TrtEngineExecutor::hasIOTensor(char const* name) const
{
    return engineHasIOTensor(*mEngine, name);
}

nvinfer1::DataType TrtEngineExecutor::getBindingDataType(char const* name) const
{
    return mEngine->getTensorDataType(name);
}

nvinfer1::Dims TrtEngineExecutor::getProfileShape(
    char const* name, int32_t profileIndex, nvinfer1::OptProfileSelector selector) const
{
    return mEngine->getProfileShape(name, profileIndex, selector);
}

void TrtEngineExecutor::setProfiler(nvinfer1::IProfiler* profiler) noexcept
{
    mContext->setProfiler(profiler);
}

nvinfer1::ICudaEngine const& TrtEngineExecutor::getEngine() const noexcept
{
    return *mEngine;
}

// ---------------------------------------------------------------------------
// BindingSnapshot
// ---------------------------------------------------------------------------

bool EngineExecutor::BindingSnapshot::operator==(BindingSnapshot const& rhs) const noexcept
{
    if (bindings.size() != rhs.bindings.size())
    {
        return false;
    }
    for (size_t i = 0; i < bindings.size(); ++i)
    {
        if (bindings[i].first != rhs.bindings[i].first)
        {
            return false;
        }
        if (!dimsEqual(bindings[i].second, rhs.bindings[i].second))
        {
            return false;
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

size_t TrtEngineExecutor::computeBindingHash() const
{
    size_t seed = 0;
    int32_t const numIO = mEngine->getNbIOTensors();
    for (int32_t i = 0; i < numIO; ++i)
    {
        char const* name = mEngine->getIOTensorName(i);
        auto const addr = reinterpret_cast<uintptr_t>(mContext->getTensorAddress(name));
        nvinfer1::Dims const shape = mContext->getTensorShape(name);

        hash_utils::hashCombine(seed, addr);
        hash_utils::hashCombine(seed, shape.nbDims);
        for (int32_t d = 0; d < shape.nbDims; ++d)
        {
            hash_utils::hashCombine(seed, shape.d[d]);
        }
    }
    return seed;
}

EngineExecutor::BindingSnapshot TrtEngineExecutor::snapshotBindings() const
{
    BindingSnapshot snap;
    int32_t const numIO = mEngine->getNbIOTensors();
    snap.bindings.reserve(numIO);
    for (int32_t i = 0; i < numIO; ++i)
    {
        char const* name = mEngine->getIOTensorName(i);
        auto const addr = reinterpret_cast<uintptr_t>(mContext->getTensorAddress(name));
        nvinfer1::Dims const shape = mContext->getTensorShape(name);
        snap.bindings.emplace_back(addr, shape);
    }
    return snap;
}

} // namespace rt
} // namespace trt_edgellm
