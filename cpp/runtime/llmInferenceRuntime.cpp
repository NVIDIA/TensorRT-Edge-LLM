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
#include "runtime/runtimeSession.h"

#include <exception>
#include <utility>

namespace trt_edgellm
{
namespace rt
{

namespace
{
RuntimeResources singleDeviceResources(cudaStream_t stream)
{
    RuntimeResources resources{};
    resources.localRanks = {0};
    resources.localStreams = {stream};
    resources.ownsLocalStreams = false;
    return resources;
}
} // namespace

LLMInferenceRuntime::LLMInferenceRuntime(DeploymentOptions options, cudaStream_t stream)
{
    auto artifacts = std::make_unique<ModelArtifacts>(ModelArtifacts::loadFromEngineDir(options.engineDir,
        options.draftingConfig, options.checkpointDir, options.draftCheckpointDir, options.contextCacheConfig.enabled,
        stream, options.runtimeCachePath, options.offloadEmbedding));
    initializeCoordinator(std::move(options), singleDeviceResources(stream), std::move(artifacts));
}

LLMInferenceRuntime::LLMInferenceRuntime(DeploymentOptions options, RuntimeResources resources)
{
    initializeCoordinator(std::move(options), std::move(resources));
}

LLMInferenceRuntime::LLMInferenceRuntime(ModelArtifacts&& artifacts, DeploymentOptions options, cudaStream_t stream)
{
    artifacts.validateDeploymentOptions(options);
    options.draftingConfig = artifacts.deployment.draftingConfig();
    initializeCoordinator(
        std::move(options), singleDeviceResources(stream), std::make_unique<ModelArtifacts>(std::move(artifacts)));
}

LLMInferenceRuntime::LLMInferenceRuntime(std::string const& engineDir, std::string const& multimodalEngineDir,
    std::unordered_map<std::string, std::string> const& loraWeightsMap, SpecDecodeDraftingConfig const& draftingConfig,
    cudaStream_t stream, ContextCacheConfig const& contextCacheConfig, std::string const& checkpointDir,
    std::string const& draftCheckpointDir, std::optional<std::filesystem::path> const& runtimeCachePath,
    PleRuntimeConfig const& pleConfig)
    : LLMInferenceRuntime(DeploymentOptions{engineDir, multimodalEngineDir, loraWeightsMap, draftingConfig,
                              contextCacheConfig, pleConfig, checkpointDir, draftCheckpointDir, {}, runtimeCachePath},
          stream)
{
}

LLMInferenceRuntime::LLMInferenceRuntime(std::string const& engineDir, std::string const& multimodalEngineDir,
    std::unordered_map<std::string, std::string> const& loraWeightsMap, cudaStream_t stream,
    ContextCacheConfig const& contextCacheConfig, std::string const& checkpointDir,
    std::optional<std::filesystem::path> const& runtimeCachePath, PleRuntimeConfig const& pleConfig)
    : LLMInferenceRuntime(DeploymentOptions{engineDir, multimodalEngineDir, loraWeightsMap, std::nullopt,
                              contextCacheConfig, pleConfig, checkpointDir, {}, {}, runtimeCachePath},
          stream)
{
}

LLMInferenceRuntime::LLMInferenceRuntime(ModelArtifacts&& artifacts, std::string const& engineDir,
    std::string const& multimodalEngineDir, std::unordered_map<std::string, std::string> const& loraWeightsMap,
    std::optional<SpecDecodeDraftingConfig> const& draftingConfig, cudaStream_t stream,
    ContextCacheConfig const& contextCacheConfig)
    : LLMInferenceRuntime(std::move(artifacts),
          DeploymentOptions{engineDir, multimodalEngineDir, loraWeightsMap, draftingConfig, contextCacheConfig}, stream)
{
}

LLMInferenceRuntime::~LLMInferenceRuntime() noexcept = default;

void LLMInferenceRuntime::initializeCoordinator(
    DeploymentOptions options, RuntimeResources resources, std::unique_ptr<ModelArtifacts> modelArtifacts)
{
    RuntimeCoordinator::Config coordinatorConfig{std::move(options), std::move(resources), std::move(modelArtifacts)};
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

int32_t LLMInferenceRuntime::worldSize() const noexcept
{
    return mCoordinator != nullptr ? mCoordinator->worldSize() : 1;
}

bool LLMInferenceRuntime::handleRequest(LLMGenerationRequest const& request, LLMGenerationResponse& response,
    cudaStream_t stream, bool outputThinkerEmbeddings)
{
    ELLM_CHECK(mCoordinator != nullptr, "Runtime coordinator is not initialized.");
    bool const dispatched
        = mCoordinator->dispatchRequest(request, getProfilingEnabled(), outputThinkerEmbeddings, stream);
    bool const succeeded = dispatched && mCoordinator->localRanksSucceeded();
    if (mCoordinator->ownsGlobalRank(0))
    {
        rootRuntime().selectProfilingTimer();
    }
    response = LLMGenerationResponse{};
    if (mCoordinator->ownsGlobalRank(0))
    {
        response = mCoordinator->takeRankResponse(0);
    }
    return succeeded;
}

bool LLMInferenceRuntime::supportsSteppedExecution() const noexcept
{
    return mCoordinator != nullptr && mCoordinator->supportsSteppedExecution();
}

std::string LLMInferenceRuntime::unavailableReason() const
{
    return mCoordinator->unavailableReason();
}

std::unique_ptr<RuntimeSession> LLMInferenceRuntime::openSession(cudaStream_t stream, SessionOptions options)
{
    ELLM_CHECK(mCoordinator != nullptr, "Runtime coordinator is not initialized.");
    return mCoordinator->openSession(getProfilingEnabled(), stream, options);
}

std::vector<int32_t> LLMInferenceRuntime::countPromptTokens(LLMGenerationRequest const& request) const
{
    ELLM_CHECK(mCoordinator != nullptr, "Runtime coordinator is not initialized.");
    return mCoordinator->countPromptTokens(request);
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

char const* LLMInferenceRuntime::getDefaultDecodingStrategyName() const
{
    return rootRuntime().getDefaultDecodingStrategyName();
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

bool LLMInferenceRuntime::supportsSeatedAdmission() const
{
    return rootRuntime().supportsSeatedAdmission();
}

int32_t LLMInferenceRuntime::maxBatchSize() const
{
    return rootRuntime().maxBatchSize();
}

bool LLMInferenceRuntime::ownsGlobalRank(int32_t globalRank) const noexcept
{
    return mCoordinator ? mCoordinator->ownsGlobalRank(globalRank) : globalRank == 0;
}

} // namespace rt
} // namespace trt_edgellm
