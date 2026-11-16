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

#pragma once

#include "common/tensor.h"
#include "runtime/exec/tensorMap.h"
#include "runtime/weight/refitWeights.h"

#include <cuda_runtime.h>
#include <filesystem>
#include <functional>
#include <optional>
#include <string_view>
#include <vector>

namespace trt_edgellm
{
namespace rt
{

class EngineExecutor;

//! Owns immutable, final-layout model weights prepared during initialization.
//!
//! Checkpoint transforms write directly from bounded CUDA-mapped source ranges.
//! Inputs, embeddings and plugin resources use a persistent arena; refit weights
//! use a separate temporary arena transferred to engine initialization.
class ExternalWeightManager
{
public:
    ExternalWeightManager() = default;
    ~ExternalWeightManager() noexcept;

    ExternalWeightManager(ExternalWeightManager const&) = delete;
    ExternalWeightManager& operator=(ExternalWeightManager const&) = delete;

    ExternalWeightManager(ExternalWeightManager&& other) noexcept;
    ExternalWeightManager& operator=(ExternalWeightManager&& other) noexcept;

    //! Load plugin-ready sidecars or load and transform an original checkpoint.
    //!
    //! componentCheckpointDir overrides the checkpoint recorded while building
    //! this engine. targetCheckpointDir supplies target-owned fallback weights
    //! for a separately packaged draft.
    //!
    //! The stream is synchronized before returning. All tensors have final layouts;
    //! temporary refit storage remains owned here until takeRefitWeights().
    //! checkCancelled runs synchronously between completed weight operations and may throw to abort loading.
    void load(std::filesystem::path const& engineDir, std::filesystem::path const& configPath, cudaStream_t stream,
        std::filesystem::path const& componentCheckpointDir = {}, std::filesystem::path const& targetCheckpointDir = {},
        std::optional<int32_t> tpRank = std::nullopt, std::optional<int32_t> tpSize = std::nullopt,
        std::function<void()> const& checkCancelled = {});

    void validateAgainstEngine(EngineExecutor const& executor, std::string_view engineLabel);

    void registerTensorMapEntries(TensorMap& map);

    //! Validate against a raw engine and point its inputs at the loaded weights.
    //!
    //! Encoder runners drive TensorRT directly instead of through EngineExecutor
    //! and TensorMap, so they bind here. Weight addresses are immutable for the
    //! life of the context, so one call at load time is enough.
    void bindToContext(
        nvinfer1::ICudaEngine const& engine, nvinfer1::IExecutionContext& context, std::string_view engineLabel);

    //! Transfer the checkpoint-backed embedding prepared during load(), if any.
    std::optional<Tensor> takeEmbedding();

    //! Transfer temporary refit storage once after load; null for ordinary engines.
    //! The returned owner must survive refit completion. Throws before load or on repeated transfer.
    std::unique_ptr<RefitWeightSet> takeRefitWeights();

    size_t size() const noexcept
    {
        return mWeights.size();
    }

private:
    struct PluginResourceRegistration
    {
        int32_t deviceId;
        int32_t resourceId;
    };

    void releasePluginResources() noexcept;

    // One allocation owns every checkpoint-backed engine input and plugin
    // resource. Individual tensors below are stable, non-owning views into this
    // storage.
    Tensor mWeightStorage{};
    std::unique_ptr<RefitWeightSet> mRefitWeights{};
    std::vector<Tensor> mWeights{};
    std::vector<Tensor> mPluginResourceTensors{};
    std::vector<PluginResourceRegistration> mPluginResourceRegistrations{};
    std::optional<Tensor> mEmbedding{};
    bool mLoaded{false};
    bool mValidated{false};
    bool mRegistered{false};
    bool mRefitTaken{false};
};

} // namespace rt
} // namespace trt_edgellm
