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
#include "runtime/config/deploymentOptions.h"
#include "runtime/config/runtimeResources.h"
#include "runtime/llmRuntimeUtils.h"
#include "runtime/modelArtifacts.h"
#include "runtime/multiDevice/parallelConfig.h"
#include "runtime/preprocess/visualTokenPruner.h"
#include "runtime/sessionOptions.h"
#include "runtime/state/contextCache/contextCacheConfig.h"
#include "runtime/state/contextCache/contextCacheMetrics.h"

#include <filesystem>
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

class RuntimeSession;

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
    //! Construct a single-device runtime using a caller-owned stream.
    LLMInferenceRuntime(DeploymentOptions options, cudaStream_t stream);

    //! Construct a runtime with explicit rank placement and execution-resource ownership.
    explicit LLMInferenceRuntime(DeploymentOptions options, RuntimeResources resources = {});

    //! Assemble using artifact-owned model/source configuration and caller-owned runtime policies.
    //! Nonempty load options must agree with the artifacts; omitted drafting inherits their resolved mode.
    LLMInferenceRuntime(ModelArtifacts&& artifacts, DeploymentOptions options, cudaStream_t stream);

    LLMInferenceRuntime(std::string const& engineDir, std::string const& multimodalEngineDir,
        std::unordered_map<std::string, std::string> const& loraWeightsMap,
        SpecDecodeDraftingConfig const& draftingConfig, cudaStream_t stream,
        ContextCacheConfig const& contextCacheConfig = {}, std::string const& checkpointDir = "",
        std::string const& draftCheckpointDir = "",
        std::optional<std::filesystem::path> const& runtimeCachePath = std::nullopt,
        PleRuntimeConfig const& pleConfig = {});

    LLMInferenceRuntime(std::string const& engineDir, std::string const& multimodalEngineDir,
        std::unordered_map<std::string, std::string> const& loraWeightsMap, cudaStream_t stream,
        ContextCacheConfig const& contextCacheConfig = {}, std::string const& checkpointDir = "",
        std::optional<std::filesystem::path> const& runtimeCachePath = std::nullopt,
        PleRuntimeConfig const& pleConfig = {});

    LLMInferenceRuntime(ModelArtifacts&& artifacts, std::string const& engineDir,
        std::string const& multimodalEngineDir, std::unordered_map<std::string, std::string> const& loraWeightsMap,
        std::optional<SpecDecodeDraftingConfig> const& draftingConfig, cudaStream_t stream,
        ContextCacheConfig const& contextCacheConfig = {});

    ~LLMInferenceRuntime() noexcept;

    bool captureDecodingCUDAGraph(cudaStream_t stream);

    bool handleRequest(LLMGenerationRequest const& request, LLMGenerationResponse& response, cudaStream_t stream,
        bool outputThinkerEmbeddings = false);

    //! Empty while reusable; a permanent failure reason until the process restarts otherwise.
    std::string unavailableReason() const;

    /*! \brief Return the input size for an explicit text token-count request. */
    std::vector<int32_t> countPromptTokens(LLMGenerationRequest const& request) const;

    void setActionNoiseSeed(int32_t seed);

    /*! \brief Enable visual-token pruning for supported single-device VLM execution. */
    void setVisualPrunerConfig(VisualPrunerConfig const& config);

    metrics::LLMPrefillMetrics const& getPrefillMetrics() const;
    metrics::SpecDecodeGenerationMetrics const& getSpecDecodeGenerationMetrics() const;
    char const* getSpeculativeDecodingStrategyName() const;
    char const* getDefaultDecodingStrategyName() const;
    metrics::LLMGenerationMetrics const& getGenerationMetrics() const;
    std::optional<ContextCacheMetrics> getContextCacheMetrics() const;
    metrics::MultimodalMetrics getMultimodalMetrics() const;
    rt::Tensor const& getEmbeddingTable() const;
    rt::Tensor const* getBaseModelHiddenStates(int32_t layerIdx) const;
    int32_t getBaseModelPrefillLength() const;
    std::vector<std::vector<int32_t>> const& getBaseModelInputTokenIds() const;
    bool hasDraftModel() const;

    //! @brief Whether the deployment can take boundary admissions (see LLMRankRuntime).
    bool supportsSeatedAdmission() const;

    //! @brief The engine's built batch dimension (see LLMRankRuntime::maxBatchSize).
    int32_t maxBatchSize() const;

    //! True when this runtime can host a runtime session (inline single-rank only in this release).
    bool supportsSteppedExecution() const noexcept;

    //! Open the runtime session a scheduler drives for the engine's life; see RuntimeSession and
    //! RuntimeCoordinator::openSession. Null when the runtime refuses: a request is in flight, the
    //! deployment cannot seat sequences into a running batch (supportsSeatedAdmission), or the
    //! session's own setup fails. @p options choose whether the session may run compact steps (see
    //! SessionOptions).
    std::unique_ptr<RuntimeSession> openSession(cudaStream_t stream, SessionOptions options = {});

    //! @brief Ranks this runtime spans. 1 when it runs on a single device.
    //!
    //! Exposed because a caller's own structure can depend on it: a scheduler that varies which
    //! requests take part in a step has to keep that decision identical on every rank, or the
    //! collectives inside a forward pass disagree on their shapes.
    int32_t worldSize() const noexcept;

    //! True when this runtime instance owns the requested global rank.
    bool ownsGlobalRank(int32_t globalRank) const noexcept;

private:
    void initializeCoordinator(DeploymentOptions options, RuntimeResources resources,
        std::unique_ptr<ModelArtifacts> modelArtifacts = nullptr);

    LLMRankRuntime& rootRuntime();
    LLMRankRuntime const& rootRuntime() const;

    std::unique_ptr<RuntimeCoordinator> mCoordinator;
};

} // namespace rt
} // namespace trt_edgellm
