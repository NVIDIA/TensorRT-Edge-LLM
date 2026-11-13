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
#include "common/trtInterop.h"
#include "common/trtUtils.h"
#include "profiling/nvtx_wrapper.h"
#include "runtime/exec/registryBuilder.h"
#include "runtime/exec/rtxRuntimeCache.h"
#include <cstdint>
#include <set>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace trt_edgellm
{
namespace rt
{
bool validateRaggedInferenceDims(
    InferenceDims const& dims, int32_t profileIndex, bool allowSelectBeyondPhysicalTokens) noexcept
{
    bool const compact = dims.executionPhaseLen == static_cast<int64_t>(ExecutionPhase::kMixedPrefillDecode);
    if (dims.batch <= 0 || dims.seqLen <= 0 || (!compact && dims.seqLen % dims.batch != 0)
        || dims.queryOffsetLen != dims.batch + 1 || dims.selectLen <= 0
        || (!allowSelectBeyondPhysicalTokens && dims.selectLen > dims.seqLen) || dims.attnMaskSeqLen != dims.seqLen
        || !isExecutionPhaseExtent(dims.executionPhaseLen)
        || !isSpecVerifyLayoutExtentValid(static_cast<ExecutionPhase>(dims.executionPhaseLen), dims.specVerifyLayoutLen)
        || dims.contextSequenceCount < 0 || dims.contextSequenceCount > dims.batch)
    {
        return false;
    }
    if (compact)
    {
        // Every sequence owns at least one row and decode sequences own exactly one, so T >= N is T_p >= Np.
        if (dims.seqLen < dims.batch || dims.selectLen > dims.batch
            || profileIndex != compactStepProfileIndex(dims.contextSequenceCount))
        {
            return false;
        }
        return dims.contextSequenceCount > 0 || dims.seqLen == dims.batch;
    }
    auto const phase = static_cast<ExecutionPhase>(dims.executionPhaseLen);
    bool const contextPhase = phase == ExecutionPhase::kContextPrefill || phase == ExecutionPhase::kContextChunk;
    if (dims.contextSequenceCount != (contextPhase ? dims.batch : 0))
    {
        return false;
    }
    constexpr int32_t kDecodeProfile = 1;
    bool const vanillaDecode = phase == ExecutionPhase::kAutoregressiveDecode;
    return profileIndex != kDecodeProfile || !vanillaDecode || dims.seqLen == dims.batch;
}

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
    TrtEngineExecutor(std::filesystem::path const& enginePath, TensorRegistry registry,
        std::shared_ptr<RtxRuntimeCache> runtimeCache, EngineRefitOptions refit = {});

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
    trt_edgellm::DataType getBindingDataType(char const* name) const override;
    nvinfer1::Dims getProfileShape(
        char const* name, int32_t profileIndex, nvinfer1::OptProfileSelector selector) const override;
    nvinfer1::Dims getTensorShape(char const* name) const override;
    nvinfer1::ICudaEngine const& getEngine() const noexcept override;
    uint64_t graphLaunches() const noexcept override
    {
        return mGraphLaunches;
    }
    bool dedicateCompactContext(int32_t profileIndex, cudaStream_t stream) override;
    bool dedicateContextChunkContext(int32_t profileIndex, cudaStream_t stream) override;
    void setContextChunkRouting(bool enabled) override;

private:
    //! Create the dedicated context on @p profileIndex, or report whether the existing one is on it.
    bool ensureDedicatedContext(int32_t profileIndex, cudaStream_t stream, char const* purpose);

    AuxStreamSet mAuxStreams{};
    AuxStreamSet mDedicatedAuxStreams{};
    std::unique_ptr<nvinfer1::IRuntime> mRuntime;
    std::unique_ptr<nvinfer1::ICudaEngine> mEngine;
#if IS_TRT_RTX
    //! The config and shared cache outlive the context that JITs kernels into them.
    std::shared_ptr<RtxRuntimeCache> mRuntimeCache;
    std::unique_ptr<nvinfer1::IRuntimeConfig> mRuntimeConfig;
#endif
    //! Original context; it serves every step except the phase-8 steps routed to mDedicatedContext.
    std::unique_ptr<nvinfer1::IExecutionContext> mContext;
    std::unique_ptr<nvinfer1::IExecutionContext> mDedicatedContext;
    TensorRegistry mRegistry;
    std::string mNvtxName;
    //! Profile currently selected on mContext.
    int32_t mCurrentProfileIndex{-1};
    int32_t mDedicatedProfileIndex{-1};
    //! Set while a ScopedContextChunkRouting is alive: phase-2 context chunks of mDedicatedProfileIndex run on the
    //! dedicated context.
    bool mRouteContextChunks{false};
    //! Context and profile of the last prepare(); execute() and captureGraph() act on them.
    nvinfer1::IExecutionContext* mActiveContext{nullptr};
    int32_t mActiveProfileIndex{-1};
    void* mContextMemory{nullptr};
    size_t mContextMemoryBytes{0};

    //! A captured CUDA graph together with its binding snapshot for verification.
    struct CapturedGraph
    {
        cudaGraph_t graph{nullptr};
        cudaGraphExec_t exec{nullptr};
        BindingSnapshot snapshot;
    };

    //! Graph cache keyed by a hash of the context, profile, and all binding addresses + shapes.
    std::unordered_map<size_t, CapturedGraph> mGraphs;
    //! (context, profile) pairs that own at least one captured graph; execute() looks up only those.
    std::set<std::pair<nvinfer1::IExecutionContext const*, int32_t>> mGraphScopes;
    uint64_t mGraphLaunches{0};

    std::unique_ptr<nvinfer1::IExecutionContext> createContext();

    //! Hash a binding snapshot of the active context into a graph-cache key.
    size_t graphKey(BindingSnapshot const& snapshot) const;

    //! Build a full snapshot of the current binding state.
    BindingSnapshot snapshotBindings() const;
};

TrtEngineExecutor::TrtEngineExecutor(std::filesystem::path const& enginePath, TensorRegistry registry,
    [[maybe_unused]] std::shared_ptr<RtxRuntimeCache> runtimeCache, EngineRefitOptions refit)
    : mRegistry(std::move(registry))
    , mNvtxName(enginePath.filename().string())
{
    LOG_INFO("loading engine file: %s", enginePath.string().c_str());

    mRuntime = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(TrtLogger::instance()));
    ELLM_CHECK(mRuntime != nullptr, "failed to create TRT IRuntime");

    mEngine = deserializeCudaEngineFromFile(*mRuntime, enginePath);
    refitEngineIfNeeded(*mEngine, std::move(refit));

#if IS_TRT_RTX
    mRuntimeCache = std::move(runtimeCache);
    if (mRuntimeCache != nullptr)
    {
        mRuntimeConfig = mRuntimeCache->createConfig(*mEngine);
    }
#endif
    mContext = createContext();
    ELLM_CHECK(mContext != nullptr, "failed to create execution context");
    mActiveContext = mContext.get();

    setNonBlockingAuxStreams(mContext.get(), mEngine.get(), mAuxStreams);

    auto registerOptionalTreeMetadata = [&](char const* name) {
        if (engineHasInputTensor(*mEngine, name) && !mRegistry.contains(name))
        {
            mRegistry.addTensor(
                {name, TensorIO::kInput, trt_edgellm::DataType::kINT32, {sym(&InferenceDims::attnMaskSeqLen)}});
        }
    };
    registerOptionalTreeMetadata(binding_names::kTreeParentIds);
    registerOptionalTreeMetadata(binding_names::kTreeDepths);
    if (engineHasInputTensor(*mEngine, binding_names::kValidTreeCounts)
        && !mRegistry.contains(binding_names::kValidTreeCounts))
    {
        mRegistry.addTensor({binding_names::kValidTreeCounts, TensorIO::kInput, trt_edgellm::DataType::kINT32,
            {sym(&InferenceDims::batch)}});
    }

    if (engineHasInputTensor(*mEngine, binding_names::kSkipSoftmaxPrefillScale)
        && !mRegistry.contains(binding_names::kSkipSoftmaxPrefillScale))
    {
        mRegistry.addTensor({binding_names::kSkipSoftmaxPrefillScale, TensorIO::kInput, trt_edgellm::DataType::kINT32,
            {sym(&InferenceDims::skipSoftmaxPrefillScaleLen)}});
    }
    if (engineHasInputTensor(*mEngine, binding_names::kSkipSoftmaxDecodeScale)
        && !mRegistry.contains(binding_names::kSkipSoftmaxDecodeScale))
    {
        mRegistry.addTensor({binding_names::kSkipSoftmaxDecodeScale, TensorIO::kInput, trt_edgellm::DataType::kINT32,
            {sym(&InferenceDims::skipSoftmaxDecodeScaleLen)}});
    }
    if (engineHasInputTensor(*mEngine, binding_names::kSpecVerifyLayout)
        && !mRegistry.contains(binding_names::kSpecVerifyLayout))
    {
        mRegistry.addTensor({binding_names::kSpecVerifyLayout, TensorIO::kInput, trt_edgellm::DataType::kINT32,
            {sym(&InferenceDims::specVerifyLayoutLen)}});
    }

    LOG_INFO("engine loaded successfully (%d I/O tensors)", mEngine->getNbIOTensors());
}

std::unique_ptr<nvinfer1::IExecutionContext> TrtEngineExecutor::createContext()
{
    std::unique_ptr<nvinfer1::IExecutionContext> context;
#if IS_TRT_RTX
    if (mRuntimeConfig != nullptr)
    {
        context = std::unique_ptr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext(mRuntimeConfig.get()));
        if (context != nullptr)
        {
            return context;
        }
        LOG_WARNING("createExecutionContext(runtimeConfig) failed; falling back to default context");
        mRuntimeConfig.reset();
    }
#endif
    // Use USER_MANAGED allocation so context memory can be shared across runners.
    context = std::unique_ptr<nvinfer1::IExecutionContext>(
        mEngine->createExecutionContext(nvinfer1::ExecutionContextAllocationStrategy::kUSER_MANAGED));
    return context;
}

bool TrtEngineExecutor::ensureDedicatedContext(int32_t profileIndex, cudaStream_t stream, char const* purpose)
{
    if (mDedicatedContext != nullptr)
    {
        return mDedicatedProfileIndex == profileIndex;
    }
    if (profileIndex < 0 || profileIndex >= mEngine->getNbOptimizationProfiles())
    {
        LOG_ERROR("%s: profile %d is out of range", purpose, profileIndex);
        return false;
    }
    std::unique_ptr<nvinfer1::IExecutionContext> context = createContext();
    if (context == nullptr)
    {
        LOG_WARNING("%s: failed to create a context; profile %d keeps the shared context", purpose, profileIndex);
        return false;
    }
    if (mContextMemory != nullptr)
    {
        context->setDeviceMemoryV2(mContextMemory, static_cast<int64_t>(mContextMemoryBytes));
    }
    if (!context->setOptimizationProfileAsync(profileIndex, stream))
    {
        LOG_WARNING("%s: failed to select profile %d; it keeps the shared context", purpose, profileIndex);
        return false;
    }
    AuxStreamSet auxStreams;
    try
    {
        setNonBlockingAuxStreams(context.get(), mEngine.get(), auxStreams);
    }
    catch (std::exception const& error)
    {
        throw std::runtime_error(std::string("the auxiliary streams of the dedicated context (profile ")
            + std::to_string(profileIndex) + ") could not be created: " + error.what());
    }
    mDedicatedAuxStreams = std::move(auxStreams);
    mDedicatedContext = std::move(context);
    mDedicatedProfileIndex = profileIndex;
    return true;
}

bool TrtEngineExecutor::dedicateCompactContext(int32_t profileIndex, cudaStream_t stream)
{
    bool const hadContext = mDedicatedContext != nullptr;
    if (!ensureDedicatedContext(profileIndex, stream, "dedicateCompactContext"))
    {
        return false;
    }
    if (!hadContext)
    {
        LOG_INFO("compact steps of optimization profile %d run on a dedicated execution context", profileIndex);
    }
    return true;
}

bool TrtEngineExecutor::dedicateContextChunkContext(int32_t profileIndex, cudaStream_t stream)
{
    bool const hadContext = mDedicatedContext != nullptr;
    if (!ensureDedicatedContext(profileIndex, stream, "dedicateContextChunkContext"))
    {
        return false;
    }
    if (!hadContext)
    {
        LOG_INFO("context-chunk steps of optimization profile %d run on a dedicated execution context", profileIndex);
    }
    return true;
}

void TrtEngineExecutor::setContextChunkRouting(bool enabled)
{
    mRouteContextChunks = enabled;
}

std::unique_ptr<EngineExecutor> EngineExecutor::createWithRegistry(
    std::filesystem::path const& enginePath, TensorRegistry registry, EngineRefitOptions refit)
{
    return std::unique_ptr<EngineExecutor>(
        new TrtEngineExecutor(enginePath, std::move(registry), nullptr, std::move(refit)));
}

std::unique_ptr<EngineExecutor> EngineExecutor::createForLLM(std::filesystem::path const& enginePath,
    LLMEngineConfig const& cfg, std::optional<int32_t> specDecodeBaseOutputHiddenDim,
    std::shared_ptr<RtxRuntimeCache> runtimeCache, EngineRefitOptions refit)
{
    auto registry = buildRegistryForLLM(cfg, specDecodeBaseOutputHiddenDim);
    return std::unique_ptr<EngineExecutor>(
        new TrtEngineExecutor(enginePath, std::move(registry), std::move(runtimeCache), std::move(refit)));
}

std::unique_ptr<EngineExecutor> EngineExecutor::createForDraft(std::filesystem::path const& enginePath,
    DeploymentConfig const& bundle, std::shared_ptr<RtxRuntimeCache> runtimeCache, EngineRefitOptions refit)
{
    switch (bundle.specDecodeMode())
    {
    case SpecDecodeMode::kMTP:
    case SpecDecodeMode::kEAGLE:
    {
        auto registry = buildRegistryForSpecDecodeDraft(bundle);
        return std::unique_ptr<EngineExecutor>(
            new TrtEngineExecutor(enginePath, std::move(registry), std::move(runtimeCache), std::move(refit)));
    }
    case SpecDecodeMode::kDFlash:
    case SpecDecodeMode::kJetSpec:
    case SpecDecodeMode::kDomino:
    {
        auto registry = buildRegistryForDFlashDraft(bundle);
        return std::unique_ptr<EngineExecutor>(
            new TrtEngineExecutor(enginePath, std::move(registry), std::move(runtimeCache), std::move(refit)));
    }
    case SpecDecodeMode::kGemma4MTP:
    {
        auto registry = buildRegistryForGemma4MTPDraft(bundle);
        return std::unique_ptr<EngineExecutor>(
            new TrtEngineExecutor(enginePath, std::move(registry), std::move(runtimeCache), std::move(refit)));
    }
    case SpecDecodeMode::kDSpark:
    {
        auto registry = buildRegistryForDSparkDraft(bundle);
        return std::unique_ptr<EngineExecutor>(
            new TrtEngineExecutor(enginePath, std::move(registry), std::move(runtimeCache), std::move(refit)));
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
    if (mRegistry.contains(binding_names::kExecutionPhaseMarker) && !isExecutionPhaseExtent(dims.executionPhaseLen))
    {
        LOG_ERROR("EngineExecutor::prepare: invalid execution phase extent %lld",
            static_cast<long long>(dims.executionPhaseLen));
        return false;
    }
    if (mRegistry.contains(binding_names::kSpecVerifyLayout)
        && !isSpecVerifyLayoutExtentValid(
            static_cast<ExecutionPhase>(dims.executionPhaseLen), dims.specVerifyLayoutLen))
    {
        LOG_ERROR("EngineExecutor::prepare: invalid spec verify layout extent %lld for execution phase %lld",
            static_cast<long long>(dims.specVerifyLayoutLen), static_cast<long long>(dims.executionPhaseLen));
        return false;
    }
    bool const hasIndependentDeltaPortal = mRegistry.contains(binding_names::kDFlashTargetHiddenConcat);
    if (mRegistry.contains(binding_names::kPositions)
        && !validateRaggedInferenceDims(dims, profileIndex, hasIndependentDeltaPortal))
    {
        LOG_ERROR("EngineExecutor::prepare: inconsistent ragged step dimensions for profile %d", profileIndex);
        return false;
    }
    if (auto bad = firstInvalidMember(dims, mRegistry.referencedMembers()); bad != nullptr)
    {
        auto const name = dimName(bad);
        LOG_ERROR(
            "EngineExecutor::prepare: InferenceDims.%.*s = %lld (must be > 0); "
            "caller likely bypassed an LLMEngineConfig recipe method",
            static_cast<int>(name.size()), name.data(), static_cast<long long>(dims.*bad));
        return false;
    }

    nvinfer1::IExecutionContext* context = mDedicatedContext.get();
    if (!runsOnCompactContext(profileIndex, dims, mDedicatedProfileIndex)
        && !runsOnContextChunkContext(profileIndex, dims, mDedicatedProfileIndex, mRouteContextChunks))
    {
        context = mContext.get();
        if (mCurrentProfileIndex != profileIndex && !context->setOptimizationProfileAsync(profileIndex, stream))
        {
            LOG_ERROR("failed to set optimization profile %d", profileIndex);
            return false;
        }
        mCurrentProfileIndex = profileIndex;
    }
    mActiveContext = context;
    mActiveProfileIndex = profileIndex;

    if (!mRegistry.bindAll(context, map, dims))
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
        if (!context->setTensorAddress(name, tensor->rawPointer()))
        {
            LOG_ERROR("setTensorAddress failed for binding '%s'", name);
            return false;
        }
        if (mEngine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT)
        {
            if (!context->setInputShape(name, toTrtDims(*tensor)))
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
    NVTX_SCOPED_RANGE(engineExecution, mNvtxName.c_str(), nvtx_colors::BLUE);
    if (mGraphScopes.count({mActiveContext, mActiveProfileIndex}) != 0)
    {
        BindingSnapshot const snapshot = snapshotBindings();
        auto const it = mGraphs.find(graphKey(snapshot));
        if (it != mGraphs.end() && snapshot == it->second.snapshot)
        {
            cudaError_t const err = cudaGraphLaunch(it->second.exec, stream);
            if (err == cudaSuccess)
            {
                ++mGraphLaunches;
                return true;
            }
            LOG_WARNING("cudaGraphLaunch failed (%s), falling back to enqueueV3", cudaGetErrorString(err));
        }
    }
    return mActiveContext->enqueueV3(stream);
}

bool TrtEngineExecutor::captureGraph(cudaStream_t stream)
{
    // Warmup: run one enqueue to ensure all internal TRT state is initialized.
    if (!mActiveContext->enqueueV3(stream))
    {
        LOG_ERROR("warmup enqueueV3 failed");
        return false;
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    auto result = captureTRTCudaGraph(mActiveContext, stream);
    if (!result.has_value())
    {
        LOG_WARNING("CUDA graph capture failed");
        return false;
    }

    BindingSnapshot const snap = snapshotBindings();
    size_t const hash = graphKey(snap);

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
    mGraphScopes.emplace(mActiveContext, mActiveProfileIndex);

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
    mContextMemory = sharedMem.rawPointer();
    mContextMemoryBytes = static_cast<size_t>(sharedMem.getMemoryCapacity());
    mContext->setDeviceMemoryV2(mContextMemory, static_cast<int64_t>(mContextMemoryBytes));
    if (mDedicatedContext != nullptr)
    {
        mDedicatedContext->setDeviceMemoryV2(mContextMemory, static_cast<int64_t>(mContextMemoryBytes));
    }
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

trt_edgellm::DataType TrtEngineExecutor::getBindingDataType(char const* name) const
{
    return fromTrt(mEngine->getTensorDataType(name));
}

nvinfer1::Dims TrtEngineExecutor::getProfileShape(
    char const* name, int32_t profileIndex, nvinfer1::OptProfileSelector selector) const
{
    return mEngine->getProfileShape(name, profileIndex, selector);
}

nvinfer1::Dims TrtEngineExecutor::getTensorShape(char const* name) const
{
    return mEngine->getTensorShape(name);
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

size_t TrtEngineExecutor::graphKey(BindingSnapshot const& snapshot) const
{
    size_t key = computeExecutionGraphKey(reinterpret_cast<uintptr_t>(mEngine.get()), mActiveProfileIndex, snapshot);
    hash_utils::hashCombine(key, reinterpret_cast<uintptr_t>(mActiveContext));
    return key;
}

size_t computeExecutionGraphKey(
    uintptr_t engineIdentity, int32_t profileIndex, EngineExecutor::BindingSnapshot const& snapshot)
{
    size_t seed = 0;
    hash_utils::hashCombine(seed, engineIdentity);
    hash_utils::hashCombine(seed, profileIndex);
    for (auto const& [address, shape] : snapshot.bindings)
    {
        hash_utils::hashCombine(seed, address);
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
        auto const addr = reinterpret_cast<uintptr_t>(mActiveContext->getTensorAddress(name));
        nvinfer1::Dims const shape = mActiveContext->getTensorShape(name);
        snap.bindings.emplace_back(addr, shape);
    }
    return snap;
}

} // namespace rt
} // namespace trt_edgellm
