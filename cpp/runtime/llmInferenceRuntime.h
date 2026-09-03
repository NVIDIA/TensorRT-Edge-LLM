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
#include "profiling/metrics.h"
#include "runtime/config/deploymentConfig.h"
#include "runtime/llmRuntimeUtils.h"
#include "runtime/modelArtifacts.h"
#include "runtime/multiDevice/parallelConfig.h"
#include "runtime/preprocess/visualTokenPruner.h"
#include "runtime/state/contextCache/contextCacheConfig.h"
#include "runtime/state/contextCache/contextCacheMetrics.h"

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace trt_edgellm
{
namespace rt
{

class LLMRankRuntime;
class RuntimeCoordinator;

/*!
 * @brief Public LLM inference runtime for single-rank and multi-rank execution.
 *
 * LLMInferenceRuntime is the stable user-facing API. It delegates execution to a
 * RuntimeCoordinator, which resolves the parallel plan and owns one or more
 * internal LLMRankRuntime instances. Single-device execution is represented as a
 * size-1 parallel plan rather than a separate runtime implementation.
 */
class LLMInferenceRuntime
{
public:
    //! Options for constructing the unified runtime with explicit parallel execution settings.
    struct ParallelExecutionConfig
    {
        ParallelConfig parallelConfig{};
        std::optional<SpecDecodeDraftingConfig> draftingConfig{};
        ContextCacheConfig contextCacheConfig{};
        std::string checkpointDir{};
        std::string draftCheckpointDir{};
        std::vector<int32_t> localRanks{};
        std::unordered_map<int32_t, int32_t> localRankDevices{};
        std::vector<cudaStream_t> localStreams{};
        bool ownsLocalStreams{true};
        std::vector<ParallelBackendHandles> backendHandles{};
    };

    LLMInferenceRuntime(std::string const& engineDir, std::string const& multimodalEngineDir,
        std::unordered_map<std::string, std::string> const& loraWeightsMap,
        SpecDecodeDraftingConfig const& draftingConfig, cudaStream_t stream,
        ContextCacheConfig const& contextCacheConfig = {}, std::string const& checkpointDir = "",
        std::string const& draftCheckpointDir = "");

    LLMInferenceRuntime(std::string const& engineDir, std::string const& multimodalEngineDir,
        std::unordered_map<std::string, std::string> const& loraWeightsMap, cudaStream_t stream,
        ContextCacheConfig const& contextCacheConfig = {}, std::string const& checkpointDir = "");

    LLMInferenceRuntime(std::string const& engineDir, std::string const& multimodalEngineDir,
        std::unordered_map<std::string, std::string> const& loraWeightsMap, ParallelExecutionConfig config);

    LLMInferenceRuntime(ModelArtifacts&& artifacts, std::string const& engineDir,
        std::string const& multimodalEngineDir, std::unordered_map<std::string, std::string> const& loraWeightsMap,
        std::optional<SpecDecodeDraftingConfig> const& draftingConfig, cudaStream_t stream,
        ContextCacheConfig const& contextCacheConfig = {});

    ~LLMInferenceRuntime() noexcept;

    bool captureDecodingCUDAGraph(cudaStream_t stream);

    bool handleRequest(LLMGenerationRequest const& request, LLMGenerationResponse& response, cudaStream_t stream,
        bool outputThinkerEmbeddings = false);

    /*! \brief Return the input size for an explicit text token-count request. */
    std::vector<int32_t> countPromptTokens(LLMGenerationRequest const& request) const;

    bool genAndSaveSystemPromptKVCache(
        std::string const& prompt, std::string const& loraWeightsName, cudaStream_t stream);

    void setActionNoiseSeed(int32_t seed);

    /*! \brief Enable visual-token pruning for supported single-device VLM execution. */
    void setVisualPrunerConfig(VisualPrunerConfig const& config);

    metrics::LLMPrefillMetrics const& getPrefillMetrics() const;
    metrics::SpecDecodeGenerationMetrics const& getSpecDecodeGenerationMetrics() const;
    char const* getSpeculativeDecodingStrategyName() const;
    metrics::LLMGenerationMetrics const& getGenerationMetrics() const;
    std::optional<ContextCacheMetrics> getContextCacheMetrics() const;
    metrics::MultimodalMetrics getMultimodalMetrics() const;
    rt::Tensor const& getEmbeddingTable() const;
    rt::Tensor const* getBaseModelHiddenStates(int32_t layerIdx) const;
    int32_t getBaseModelPrefillLength() const;
    std::vector<std::vector<int32_t>> const& getBaseModelInputTokenIds() const;
    bool hasDraftModel() const;

    //! True when this runtime instance owns the requested global rank.
    bool ownsGlobalRank(int32_t globalRank) const noexcept;

private:
    void initializeCoordinator(std::string const& engineDir, std::string const& multimodalEngineDir,
        std::unordered_map<std::string, std::string> const& loraWeightsMap, ParallelExecutionConfig config,
        std::unique_ptr<ModelArtifacts> modelArtifacts = nullptr);

    LLMRankRuntime& rootRuntime();
    LLMRankRuntime const& rootRuntime() const;

    std::unique_ptr<RuntimeCoordinator> mCoordinator;
};

} // namespace rt
} // namespace trt_edgellm
