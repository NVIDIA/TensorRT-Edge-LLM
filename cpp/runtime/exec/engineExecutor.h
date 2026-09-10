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

#include "common/tensor.h"
#include "common/trtUtils.h"
#include "runtime/config/deploymentConfig.h"
#include "runtime/config/llmEngineConfig.h"
#include "runtime/exec/tensorMap.h"
#include "runtime/exec/tensorRegistry.h"
#include <NvInferRuntime.h>
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <filesystem>
#include <memory>
#include <optional>
#include <unordered_map>
#include <utility>
#include <vector>

namespace trt_edgellm
{
namespace rt
{

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
 * Implementations divide into two roles. `prepare`/`execute`/`captureGraph`/`setContextMemory`/`setProfiler`/
 * `getRequiredContextMemorySize` run per request and a substitute must implement them. The introspection accessors
 * (`getEngine` and the binding queries) run once, during startup validation, and a substitute may reject them.
 */
class EngineExecutor
{
public:
    virtual ~EngineExecutor() = default;

    //! Build an EngineExecutor for a vanilla single-engine LLM or a SpecDecode
    //! base engine. The factory builds the TensorRegistry internally via
    //! `buildRegistryForLLM(cfg)`.
    static std::unique_ptr<EngineExecutor> createForLLM(std::filesystem::path const& enginePath,
        LLMEngineConfig const& cfg, std::optional<int32_t> specDecodeBaseOutputHiddenDim = std::nullopt);

    //! Build an EngineExecutor for a speculative decoding draft engine. The
    //! factory chooses the draft binding registry from `bundle.specDecodeMode()`.
    static std::unique_ptr<EngineExecutor> createForDraft(
        std::filesystem::path const& enginePath, DeploymentConfig const& bundle);

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
    virtual nvinfer1::DataType getBindingDataType(char const* name) const = 0;

    //! @brief Return a profile shape (min/opt/max) for a named binding.
    virtual nvinfer1::Dims getProfileShape(
        char const* name, int32_t profileIndex, nvinfer1::OptProfileSelector selector) const = 0;

    //! @brief Attach a TRT profiler to the execution context.
    //!
    //! The profiler receives per-layer timing callbacks during enqueueV3.
    //! Must be called before execute() for the profiler to receive data.
    //! Passing nullptr detaches any previously set profiler.
    virtual void setProfiler(nvinfer1::IProfiler* profiler) noexcept = 0;

    //! @brief Access the underlying TRT engine for generic introspection.
    virtual nvinfer1::ICudaEngine const& getEngine() const noexcept = 0;

    //! @brief Snapshot of binding addresses and shapes — used for graph-cache verification.
    struct BindingSnapshot
    {
        std::vector<std::pair<uintptr_t, nvinfer1::Dims>> bindings;

        bool operator==(BindingSnapshot const& rhs) const noexcept;
    };

protected:
    EngineExecutor() = default;
};

} // namespace rt
} // namespace trt_edgellm
