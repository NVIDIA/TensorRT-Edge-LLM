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

#pragma once

#include "common/executionPhase.h"
#include "common/tensor.h"
#include "common/trtUtils.h"
#include "runtime/config/deploymentConfig.h"
#include "runtime/config/llmEngineConfig.h"
#include "runtime/exec/tensorMap.h"
#include "runtime/exec/tensorRegistry.h"
#include "runtime/weight/refitWeights.h"
#include <NvInferRuntime.h>
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace trt_edgellm
{
namespace rt
{

//! Optimization profile of a compact phase-8 step: profile 0 when it carries context sequences, profile 1 for a
//! decode-only subset (Np == 0, whose context-sequence carrier is [0]).
constexpr int32_t compactStepProfileIndex(int64_t contextSequenceCount) noexcept
{
    return contextSequenceCount > 0 ? 0 : 1;
}

//! Whether a step runs on the compact context dedicated to @p dedicatedProfileIndex (-1: none): only phase-8 steps
//! of that profile do; every other step, on any profile, keeps the executor's original context.
constexpr bool runsOnCompactContext(
    int32_t profileIndex, InferenceDims const& dims, int32_t dedicatedProfileIndex) noexcept
{
    return dedicatedProfileIndex >= 0 && profileIndex == dedicatedProfileIndex
        && dims.executionPhaseLen == static_cast<int64_t>(ExecutionPhase::kMixedPrefillDecode);
}

//! Whether a step runs on the dedicated context because it is a phase-2 context chunk of @p dedicatedProfileIndex
//! issued while routing is on (@p routesContextChunks: a ScopedContextChunkRouting is alive). The vanilla decoder
//! turns it on around its jump-forward catch-up step; other context chunks keep the original context.
constexpr bool runsOnContextChunkContext(
    int32_t profileIndex, InferenceDims const& dims, int32_t dedicatedProfileIndex, bool routesContextChunks) noexcept
{
    return routesContextChunks && dedicatedProfileIndex >= 0 && profileIndex == dedicatedProfileIndex
        && dims.executionPhaseLen == static_cast<int64_t>(ExecutionPhase::kContextChunk);
}

//! Validate cross-binding relations that TensorRT optimization profiles cannot express.
bool validateRaggedInferenceDims(
    InferenceDims const& dims, int32_t profileIndex, bool allowSelectBeyondPhysicalTokens = false) noexcept;
class RtxRuntimeCache;

/*!
 * @brief Engine execution interface with a prepare/execute split.
 *
 * The production implementation owns a TRT runtime, engine, and execution context:
 *
 *  - @c prepare() sets the optimization profile and delegates binding to
 *    TensorRegistry::bindAll().
 *  - @c execute() replays a cached CUDA graph when the binding state matches,
 *    otherwise falls back to enqueueV3.
 *  - @c captureGraph() captures a new CUDA graph for the current bindings.
 *
 * EngineExecutor knows nothing about models, phases, or features. It is abstract so callers can be exercised without a
 * serialized engine on disk; the factories below are the only way to obtain the TRT-backed implementation.
 *
 * Implementations divide into two roles. `prepare`/`execute`/`captureGraph`/`setContextMemory`/
 * `getRequiredContextMemorySize` run per request and a substitute must implement them. The introspection accessors
 * (`getEngine` and the binding queries) run once, during startup validation, and a substitute may reject them.
 */
class EngineExecutor
{
public:
    virtual ~EngineExecutor() = default;

    //! Build an executor for an internal engine with explicitly supplied tensor bindings.
    //! Optional refit ownership is consumed before context creation; its stream must be non-default.
    static std::unique_ptr<EngineExecutor> createWithRegistry(
        std::filesystem::path const& enginePath, TensorRegistry registry, EngineRefitOptions refit = {});

    //! Build an EngineExecutor for a vanilla single-engine LLM or a SpecDecode
    //! base engine. The factory builds the TensorRegistry internally via
    //! `buildRegistryForLLM(cfg)`.
    //! Optional refit ownership is consumed before context creation; its stream must be non-default.
    static std::unique_ptr<EngineExecutor> createForLLM(std::filesystem::path const& enginePath,
        LLMEngineConfig const& cfg, std::optional<int32_t> specDecodeBaseOutputHiddenDim = std::nullopt,
        std::shared_ptr<RtxRuntimeCache> runtimeCache = nullptr, EngineRefitOptions refit = {});

    //! Build an EngineExecutor for a speculative decoding draft engine. The
    //! factory chooses the draft binding registry from `bundle.specDecodeMode()`.
    static std::unique_ptr<EngineExecutor> createForDraft(std::filesystem::path const& enginePath,
        DeploymentConfig const& bundle, std::shared_ptr<RtxRuntimeCache> runtimeCache = nullptr,
        EngineRefitOptions refit = {});

    EngineExecutor(EngineExecutor const&) = delete;
    EngineExecutor& operator=(EngineExecutor const&) = delete;

    /*!
     * @brief Switch optimization profile, resolve shapes, bind all tensors.
     *
     * @param profileIndex TRT optimization profile index
     * @param dims Symbolic dimension values for this step
     * @param map Name-to-tensor mapping
     * @param stream CUDA stream for the async profile switch
     * @return True on success
     */
    virtual bool prepare(int32_t profileIndex, InferenceDims const& dims, TensorMap const& map, cudaStream_t stream)
        = 0;

    /*!
     * @brief Execute inference.
     *
     * Replays a cached CUDA graph if one matches the current bindings,
     * otherwise falls back to enqueueV3.
     *
     * @param stream CUDA stream
     * @return True on success
     */
    virtual bool execute(cudaStream_t stream) = 0;

    /*!
     * @brief Capture a CUDA graph for the current binding state (after prepare()).
     *
     * Performs a warmup enqueue, then captures via cudaStreamBeginCapture.
     * The captured graph is keyed by a binding hash with full snapshot
     * verification.
     *
     * @param stream CUDA stream (must not be the default stream)
     * @return True if capture succeeded
     */
    virtual bool captureGraph(cudaStream_t stream) = 0;

    /*!
     * @brief Query required device memory for the execution context.
     * @return Required memory size in bytes
     */
    virtual int64_t getRequiredContextMemorySize() const = 0;

    /*!
     * @brief Provide shared device memory for the execution context.
     *
     * @param sharedMem Tensor whose memory will back the TRT context
     * @return True on success
     */
    virtual bool setContextMemory(Tensor& sharedMem) = 0;

    //! @brief Return the number of I/O tensors in the engine.
    virtual int32_t getNumIOTensors() const = 0;

    //! @brief Return the name of the i-th I/O tensor.
    virtual char const* getIOTensorName(int32_t index) const = 0;

    //! @brief Return whether the engine exposes a named I/O tensor.
    virtual bool hasIOTensor(char const* name) const = 0;

    //! @brief Return the data type of a named binding.
    virtual trt_edgellm::DataType getBindingDataType(char const* name) const = 0;

    //! @brief Return a profile shape (min/opt/max) for a named binding.
    virtual nvinfer1::Dims getProfileShape(
        char const* name, int32_t profileIndex, nvinfer1::OptProfileSelector selector) const = 0;

    //! @brief Return the engine's build-time shape of a named input or output, with -1 for each dynamic dimension.
    //!
    //! Unlike getProfileShape(), this also answers for outputs, whose static dimensions no profile records. An unknown
    //! name returns `Dims{-1, {}}` (nbDims == -1).
    virtual nvinfer1::Dims getTensorShape(char const* name) const = 0;

    //! @brief Access the underlying TRT engine for generic introspection.
    virtual nvinfer1::ICudaEngine const& getEngine() const noexcept = 0;

    //! @brief execute() calls that replayed a captured CUDA graph, over the executor's life.
    virtual uint64_t graphLaunches() const noexcept
    {
        return 0;
    }

    /*!
     * @brief Run every later phase-8 (compact) step of @p profileIndex on an execution context that never leaves
     *        that profile; every other step keeps the original context (see runsOnCompactContext).
     *
     * TensorRT updates layer resources on the first enqueue after an optimization-profile switch, which costs
     * milliseconds of host time on each switch when compact steps alternate with decode steps. The dedicated
     * context shares this executor's device memory, which is sound because every step runs in the order of one
     * stream.
     *
     * @return True when phase-8 steps of @p profileIndex run on a dedicated context after the call; false when
     *         they keep the original context
     * @throws std::runtime_error when the dedicated context's auxiliary streams cannot be created; the runtime then
     *         records compact execution as unavailable (LLMRankRuntime::prepareCompactExecution)
     */
    virtual bool dedicateCompactContext([[maybe_unused]] int32_t profileIndex, [[maybe_unused]] cudaStream_t stream)
    {
        return false;
    }

    /*!
     * @brief Create the dedicated context on @p profileIndex (the compact one, when both exist) for the phase-2
     *        context-chunk steps issued inside a ScopedContextChunkRouting; see runsOnContextChunkContext.
     *
     * Jump-forward catch-up steps are context chunks that alternate with decode steps, so without this each one
     * costs the original context two profile switches.
     *
     * @return True when such steps run on a dedicated context after the call
     * @throws std::runtime_error when the dedicated context's auxiliary streams cannot be created
     */
    virtual bool dedicateContextChunkContext(
        [[maybe_unused]] int32_t profileIndex, [[maybe_unused]] cudaStream_t stream)
    {
        return false;
    }

    //! Route the phase-2 context-chunk steps prepared while @p enabled to the dedicated context.
    virtual void setContextChunkRouting([[maybe_unused]] bool enabled) {}

    //! Turns context-chunk routing on for its lifetime: one prepare()/execute() pair.
    class ScopedContextChunkRouting
    {
    public:
        explicit ScopedContextChunkRouting(EngineExecutor& executor)
            : mExecutor(executor)
        {
            mExecutor.setContextChunkRouting(true);
        }

        ~ScopedContextChunkRouting()
        {
            mExecutor.setContextChunkRouting(false);
        }

        ScopedContextChunkRouting(ScopedContextChunkRouting const&) = delete;
        ScopedContextChunkRouting& operator=(ScopedContextChunkRouting const&) = delete;

    private:
        EngineExecutor& mExecutor;
    };

    //! @brief Snapshot of binding addresses and shapes — used for graph-cache verification.
    struct BindingSnapshot
    {
        std::vector<std::pair<uintptr_t, nvinfer1::Dims>> bindings;

        bool operator==(BindingSnapshot const& rhs) const noexcept;
    };

protected:
    EngineExecutor() = default;
};

size_t computeExecutionGraphKey(
    uintptr_t engineIdentity, int32_t profileIndex, EngineExecutor::BindingSnapshot const& snapshot);

} // namespace rt
} // namespace trt_edgellm
