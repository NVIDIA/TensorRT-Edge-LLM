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

#include "llmInferenceRuntime.h"

#include "common/checkMacros.h"
#include "common/logger.h"
#include "runtime/llmRankRuntime.h"
#include "runtime/multiDevice/runtimeCoordinator.h"

#include <exception>
#include <utility>

namespace trt_edgellm
{
namespace rt
{

LLMInferenceRuntime::LLMInferenceRuntime(std::string const& engineDir, std::string const& multimodalEngineDir,
    std::unordered_map<std::string, std::string> const& loraWeightsMap, SpecDecodeDraftingConfig const& draftingConfig,
    cudaStream_t stream, ContextCacheConfig const& contextCacheConfig, std::string const& checkpointDir,
    std::string const& draftCheckpointDir)
{
    ParallelExecutionConfig config{};
    config.localRanks = {0};
    config.localStreams = {stream};
    config.ownsLocalStreams = false;
    config.draftingConfig = draftingConfig;
    config.contextCacheConfig = contextCacheConfig;
    config.checkpointDir = checkpointDir;
    config.draftCheckpointDir = draftCheckpointDir;
    auto artifacts = std::make_unique<ModelArtifacts>(
        ModelArtifacts::loadFromEngineDir(engineDir, draftingConfig, checkpointDir, draftCheckpointDir, stream));
    initializeCoordinator(engineDir, multimodalEngineDir, loraWeightsMap, std::move(config), std::move(artifacts));
}

LLMInferenceRuntime::LLMInferenceRuntime(std::string const& engineDir, std::string const& multimodalEngineDir,
    std::unordered_map<std::string, std::string> const& loraWeightsMap, cudaStream_t stream,
    ContextCacheConfig const& contextCacheConfig, std::string const& checkpointDir)
{
    ParallelExecutionConfig config{};
    config.localRanks = {0};
    config.localStreams = {stream};
    config.ownsLocalStreams = false;
    config.contextCacheConfig = contextCacheConfig;
    config.checkpointDir = checkpointDir;
    auto artifacts = std::make_unique<ModelArtifacts>(
        ModelArtifacts::loadFromEngineDir(engineDir, std::nullopt, checkpointDir, "", stream));
    initializeCoordinator(engineDir, multimodalEngineDir, loraWeightsMap, std::move(config), std::move(artifacts));
}

LLMInferenceRuntime::LLMInferenceRuntime(std::string const& engineDir, std::string const& multimodalEngineDir,
    std::unordered_map<std::string, std::string> const& loraWeightsMap, ParallelExecutionConfig config)
{
    initializeCoordinator(engineDir, multimodalEngineDir, loraWeightsMap, std::move(config));
}

LLMInferenceRuntime::LLMInferenceRuntime(ModelArtifacts&& artifacts, std::string const& engineDir,
    std::string const& multimodalEngineDir, std::unordered_map<std::string, std::string> const& loraWeightsMap,
    std::optional<SpecDecodeDraftingConfig> const& draftingConfig, cudaStream_t stream,
    ContextCacheConfig const& contextCacheConfig)
{
    ParallelExecutionConfig config{};
    config.localRanks = {0};
    config.localStreams = {stream};
    config.ownsLocalStreams = false;
    config.draftingConfig = draftingConfig;
    config.contextCacheConfig = contextCacheConfig;
    initializeCoordinator(engineDir, multimodalEngineDir, loraWeightsMap, std::move(config),
        std::make_unique<ModelArtifacts>(std::move(artifacts)));
}

LLMInferenceRuntime::~LLMInferenceRuntime() noexcept = default;

void LLMInferenceRuntime::initializeCoordinator(std::string const& engineDir, std::string const& multimodalEngineDir,
    std::unordered_map<std::string, std::string> const& loraWeightsMap, ParallelExecutionConfig config,
    std::unique_ptr<ModelArtifacts> modelArtifacts)
{
    RuntimeCoordinator::Config coordinatorConfig{};
    coordinatorConfig.engineDir = engineDir;
    coordinatorConfig.multimodalEngineDir = multimodalEngineDir;
    coordinatorConfig.loraWeightsMap = loraWeightsMap;
    coordinatorConfig.parallelConfig = config.parallelConfig;
    coordinatorConfig.draftingConfig = config.draftingConfig;
    coordinatorConfig.contextCacheConfig = config.contextCacheConfig;
    coordinatorConfig.checkpointDir = std::move(config.checkpointDir);
    coordinatorConfig.draftCheckpointDir = std::move(config.draftCheckpointDir);
    coordinatorConfig.localRanks = std::move(config.localRanks);
    coordinatorConfig.localRankDevices = std::move(config.localRankDevices);
    coordinatorConfig.localStreams = std::move(config.localStreams);
    coordinatorConfig.ownsLocalStreams = config.ownsLocalStreams;
    coordinatorConfig.backendHandles = std::move(config.backendHandles);
    coordinatorConfig.modelArtifacts = std::move(modelArtifacts);
    mCoordinator = std::make_unique<RuntimeCoordinator>(std::move(coordinatorConfig));
}

LLMRankRuntime& LLMInferenceRuntime::rootRuntime()
{
    ELLM_CHECK(mCoordinator != nullptr, "Runtime coordinator is not initialized.");
    return mCoordinator->rootRuntime();
}

LLMRankRuntime const& LLMInferenceRuntime::rootRuntime() const
{
    ELLM_CHECK(mCoordinator != nullptr, "Runtime coordinator is not initialized.");
    return mCoordinator->rootRuntime();
}

bool LLMInferenceRuntime::captureDecodingCUDAGraph(cudaStream_t stream)
{
    try
    {
        ELLM_CHECK(mCoordinator != nullptr, "Runtime coordinator is not initialized.");
        return mCoordinator->captureDecodingCUDAGraph(stream);
    }
    catch (std::exception const& e)
    {
        LOG_WARNING("CUDA graph capture failed with exception: %s", e.what());
        static_cast<void>(cudaGetLastError());
        return false;
    }
    catch (...)
    {
        LOG_WARNING("CUDA graph capture failed with an unknown exception.");
        static_cast<void>(cudaGetLastError());
        return false;
    }
}

bool LLMInferenceRuntime::handleRequest(LLMGenerationRequest const& request, LLMGenerationResponse& response,
    cudaStream_t stream, bool outputThinkerEmbeddings)
{
    ELLM_CHECK(mCoordinator != nullptr, "Runtime coordinator is not initialized.");
    bool const dispatched
        = mCoordinator->dispatchRequest(request, getProfilingEnabled(), outputThinkerEmbeddings, stream);
    bool const succeeded = dispatched && mCoordinator->localRanksSucceeded();
    response = LLMGenerationResponse{};
    if (mCoordinator->ownsGlobalRank(0))
    {
        response = mCoordinator->takeRankResponse(0);
    }
    return succeeded;
}

std::vector<int32_t> LLMInferenceRuntime::countPromptTokens(LLMGenerationRequest const& request) const
{
    return rootRuntime().countPromptTokens(request);
}

bool LLMInferenceRuntime::genAndSaveSystemPromptKVCache(
    std::string const& prompt, std::string const& loraWeightsName, cudaStream_t stream)
{
    ELLM_CHECK(mCoordinator != nullptr, "Runtime coordinator is not initialized.");
    return mCoordinator->genAndSaveSystemPromptKVCache(prompt, loraWeightsName, stream);
}

void LLMInferenceRuntime::setActionNoiseSeed(int32_t seed)
{
    if (mCoordinator != nullptr)
    {
        rootRuntime().setActionNoiseSeed(seed);
    }
}

void LLMInferenceRuntime::setVisualPrunerConfig(VisualPrunerConfig const& config)
{
    ELLM_CHECK(mCoordinator != nullptr, "Runtime coordinator is not initialized.");
    mCoordinator->setVisualPrunerConfig(config);
}

metrics::LLMPrefillMetrics const& LLMInferenceRuntime::getPrefillMetrics() const
{
    return rootRuntime().getPrefillMetrics();
}

metrics::SpecDecodeGenerationMetrics const& LLMInferenceRuntime::getSpecDecodeGenerationMetrics() const
{
    return rootRuntime().getSpecDecodeGenerationMetrics();
}

char const* LLMInferenceRuntime::getSpeculativeDecodingStrategyName() const
{
    return rootRuntime().getSpeculativeDecodingStrategyName();
}

metrics::LLMGenerationMetrics const& LLMInferenceRuntime::getGenerationMetrics() const
{
    return rootRuntime().getGenerationMetrics();
}

std::optional<ContextCacheMetrics> LLMInferenceRuntime::getContextCacheMetrics() const
{
    return rootRuntime().getContextCacheMetrics();
}

metrics::MultimodalMetrics LLMInferenceRuntime::getMultimodalMetrics() const
{
    return rootRuntime().getMultimodalMetrics();
}

rt::Tensor const& LLMInferenceRuntime::getEmbeddingTable() const
{
    return rootRuntime().getEmbeddingTable();
}

rt::Tensor const* LLMInferenceRuntime::getBaseModelHiddenStates(int32_t layerIdx) const
{
    return rootRuntime().getBaseModelHiddenStates(layerIdx);
}

int32_t LLMInferenceRuntime::getBaseModelPrefillLength() const
{
    return rootRuntime().getBaseModelPrefillLength();
}

std::vector<std::vector<int32_t>> const& LLMInferenceRuntime::getBaseModelInputTokenIds() const
{
    return rootRuntime().getBaseModelInputTokenIds();
}

bool LLMInferenceRuntime::hasDraftModel() const
{
    return rootRuntime().hasDraftModel();
}

bool LLMInferenceRuntime::ownsGlobalRank(int32_t globalRank) const noexcept
{
    return mCoordinator ? mCoordinator->ownsGlobalRank(globalRank) : globalRank == 0;
}

} // namespace rt
} // namespace trt_edgellm
