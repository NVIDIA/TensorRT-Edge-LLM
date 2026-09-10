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

#include "multimodal/common/multimodalRunner.h"
#include "common/checkMacros.h"
#include "common/trtUtils.h"
#include "multimodal/cosmos3/cosmos3EdgeViTRunner.h"
#include "multimodal/gemma4/gemma4AudioRunner.h"
#include "multimodal/gemma4/gemma4UnifiedAudioRunner.h"
#include "multimodal/gemma4/gemma4UnifiedVisionRunner.h"
#include "multimodal/gemma4/gemma4ViTRunner.h"
#include "multimodal/internvl/internViTRunner.h"
#include "multimodal/nemotron_omni/nemotronOmniAudioRunner.h"
#include "multimodal/nemotron_omni/nemotronOmniViTRunner.h"
#include "multimodal/phi4mm/phi4mmViTRunner.h"
#include "multimodal/qwen2/qwenViTRunner.h"
#include "multimodal/qwen2_5/qwen25vlViTRunner.h"
#include "multimodal/qwen3/qwen3vlViTRunner.h"
#include "multimodal/qwen3_omni/audioRunner.h"
#include "multimodal/qwen3_omni/qwen3omniViTRunner.h"
#include "profiling/layerProfiler.h"
#include "profiling/metrics.h"
#include "profiling/timer.h"
#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <stdexcept>

namespace trt_edgellm
{
namespace rt
{

MultimodalRunner::MultimodalRunner(std::string const& engineDir, cudaStream_t stream)
{
    mRuntime = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(gLogger));

    // Construct engine path from directory (engineDir already points to visual/ subdirectory)
    std::string enginePath = engineDir + "/visual.engine";

    // Load engine
    mVisualEngine = deserializeCudaEngineFromFile(*mRuntime, enginePath);

    // Create context with user-managed memory (no device memory allocated here).
    // The context object is needed by subclasses for tensor binding during initialization.
    // Device memory must be provided via setContextMemory() before infer().
    mVisualContext = std::unique_ptr<nvinfer1::IExecutionContext>(
        mVisualEngine->createExecutionContext(nvinfer1::ExecutionContextAllocationStrategy::kUSER_MANAGED));
    bool const profileSet = mVisualContext->setOptimizationProfileAsync(0, stream);
    ELLM_CHECK(profileSet, "Failed to set optimization profile for visual engine");

    setNonBlockingAuxStreams(mVisualContext.get(), mVisualEngine.get(), mAuxStreams);

    if (trt_edgellm::layerProfiler::LayerProfiler::getInstance().isEnabled())
    {
        mVisualContext->setProfiler(&trt_edgellm::layerProfiler::LayerProfiler::getInstance());
    }
}

void MultimodalRunner::loadExternalWeights(
    std::string const& engineDir, std::string const& checkpointDir, cudaStream_t stream)
{
    if (mExternalWeightsLoaded)
    {
        return;
    }
    mExternalWeightsLoaded = true;

    std::filesystem::path const configPath = std::filesystem::path(engineDir) / "config.json";
    auto* engine = mAudioEngine ? mAudioEngine.get() : mVisualEngine.get();
    auto* context = mAudioEngine ? mAudioContext.get() : mVisualContext.get();
    if (!engine || !context || !std::filesystem::exists(configPath))
    {
        return;
    }

    mExternalWeights = std::make_unique<ExternalWeightManager>();
    mExternalWeights->load(std::filesystem::path(engineDir), configPath, stream, checkpointDir);
    if (mExternalWeights->size() == 0)
    {
        mExternalWeights.reset();
        return;
    }
    mExternalWeights->bindToContext(*engine, *context, mAudioEngine ? "audio" : "visual");
}

int64_t MultimodalRunner::getRequiredContextMemorySize() const
{
    auto* engine = mAudioEngine ? mAudioEngine.get() : mVisualEngine.get();
    return engine ? engine->getDeviceMemorySizeV2() : 0;
}

bool MultimodalRunner::setContextMemory(rt::Tensor& sharedContextMemory)
{
    // Pick the audio pair for audio-only runners, otherwise the visual pair.
    // If neither is populated there is nothing to configure (e.g. default-constructed runner).
    auto* engine = mAudioEngine ? mAudioEngine.get() : mVisualEngine.get();
    auto* context = mAudioEngine ? mAudioContext.get() : mVisualContext.get();
    if (!engine)
    {
        return true;
    }

    int64_t const requiredSize = getRequiredContextMemorySize();
    if (sharedContextMemory.getMemoryCapacity() < requiredSize)
    {
        LOG_ERROR("Shared context memory (%zu bytes) is smaller than required (%zu bytes)",
            static_cast<size_t>(sharedContextMemory.getMemoryCapacity()), static_cast<size_t>(requiredSize));
        return false;
    }

    context->setDeviceMemoryV2(sharedContextMemory.rawPointer(), sharedContextMemory.getMemoryCapacity());
    return true;
}

namespace
{
//! \brief Construct a QwenViTRunner-family runner, then run its two-phase initialize().
//!
//! External weights are bound between the two phases: initialize() may enqueue
//! the engine, which needs every input address already set.
template <typename RunnerT>
std::unique_ptr<RunnerT> makeInitializedQwenViTRunner(std::string const& engineDir, int32_t llmMaxBatchSize,
    int64_t llmMaxPositionEmbeddings, cudaStream_t stream, std::string const& checkpointDir)
{
    auto runner = std::make_unique<RunnerT>(engineDir, llmMaxBatchSize, llmMaxPositionEmbeddings, stream);
    runner->loadExternalWeights(engineDir, checkpointDir, stream);
    runner->initialize(stream);
    return runner;
}
} // namespace

std::unique_ptr<MultimodalRunner> MultimodalRunner::create(std::string const& multimodalEngineDir,
    int32_t llmMaxBatchSize, int64_t llmMaxPositionEmbeddings, cudaStream_t stream, std::string const& checkpointDir)
{
    std::unique_ptr<MultimodalRunner> multimodalRunner;

    // Read config.json to determine model type
    std::string configPath = multimodalEngineDir + "/config.json";
    std::ifstream configFileStream(configPath);
    ELLM_CHECK(configFileStream.is_open(), "Failed to open config file: " + configPath);

    nlohmann::json jsonConfig;
    try
    {
        jsonConfig = nlohmann::json::parse(configFileStream);
        configFileStream.close();
    }
    catch (nlohmann::json::parse_error const& e)
    {
        throw std::runtime_error("Failed to parse config file: " + std::string(e.what()));
    }

    std::string modelTypeStr = jsonConfig["model_type"].get<std::string>();
    multimodal::ModelType modelType = multimodal::stringToModelType(modelTypeStr);

    // Qwen vision family: base QwenViTRunner == Qwen2-VL; each later model is a subclass that extends it.
    if (modelType == multimodal::ModelType::QWEN2_VL)
    {
        multimodalRunner = makeInitializedQwenViTRunner<QwenViTRunner>(
            multimodalEngineDir, llmMaxBatchSize, llmMaxPositionEmbeddings, stream, checkpointDir);
    }
    else if (modelType == multimodal::ModelType::QWEN2_5_VL)
    {
        multimodalRunner = makeInitializedQwenViTRunner<Qwen25VLViTRunner>(
            multimodalEngineDir, llmMaxBatchSize, llmMaxPositionEmbeddings, stream, checkpointDir);
    }
    else if (modelType == multimodal::ModelType::QWEN3_VL || modelType == multimodal::ModelType::QWEN3_5)
    {
        multimodalRunner = makeInitializedQwenViTRunner<Qwen3VLViTRunner>(
            multimodalEngineDir, llmMaxBatchSize, llmMaxPositionEmbeddings, stream, checkpointDir);
    }
    else if (modelType == multimodal::ModelType::COSMOS3_EDGE)
    {
        multimodalRunner = makeInitializedQwenViTRunner<Cosmos3EdgeViTRunner>(
            multimodalEngineDir, llmMaxBatchSize, llmMaxPositionEmbeddings, stream, checkpointDir);
    }
    else if (modelType == multimodal::ModelType::QWEN3_OMNI_AUDIO_ENCODER
        || modelType == multimodal::ModelType::QWEN3_OMNI_NEXT_AUDIO_ENCODER)
    {
        // Qwen3OmniAudioRunner handles both variants (it branches internally on the
        // config model_type for the Next encoder's 8x-downsample front end).
        multimodalRunner = std::make_unique<Qwen3OmniAudioRunner>(multimodalEngineDir, stream);
    }
    else if (modelType == multimodal::ModelType::QWEN3_OMNI_VISION_ENCODER)
    {
        multimodalRunner = makeInitializedQwenViTRunner<Qwen3OmniViTRunner>(
            multimodalEngineDir, llmMaxBatchSize, llmMaxPositionEmbeddings, stream, checkpointDir);
    }
    else if (modelType == multimodal::ModelType::INTERNVL)
    {
        multimodalRunner = std::make_unique<InternViTRunner>(multimodalEngineDir, stream);
    }
    else if (modelType == multimodal::ModelType::PHI4MM)
    {
        multimodalRunner = std::make_unique<Phi4MMViTRunner>(multimodalEngineDir, stream);
    }
    else if (modelType == multimodal::ModelType::GEMMA4_VISION)
    {
        multimodalRunner = std::make_unique<Gemma4ViTRunner>(multimodalEngineDir, stream);
    }
    else if (modelType == multimodal::ModelType::GEMMA4_UNIFIED_VISION)
    {
        multimodalRunner = std::make_unique<Gemma4UnifiedVisionRunner>(multimodalEngineDir, stream);
    }
    else if (modelType == multimodal::ModelType::GEMMA4_UNIFIED_AUDIO)
    {
        multimodalRunner = std::make_unique<Gemma4UnifiedAudioRunner>(multimodalEngineDir, stream);
    }
    else if (modelType == multimodal::ModelType::NEMOTRON_OMNI_VISION_ENCODER)
    {
        multimodalRunner = std::make_unique<NemotronOmniViTRunner>(multimodalEngineDir, stream);
    }
    else if (modelType == multimodal::ModelType::NEMOTRON_OMNI_AUDIO_ENCODER)
    {
        multimodalRunner = std::make_unique<NemotronOmniAudioRunner>(multimodalEngineDir, stream);
    }
    else if (modelType == multimodal::ModelType::GEMMA4_AUDIO_ENCODER)
    {
        multimodalRunner = std::make_unique<Gemma4AudioRunner>(multimodalEngineDir, stream);
    }
    else
    {
        throw std::runtime_error("Unsupported model type: " + modelTypeStr);
    }

    // The Qwen family already bound inside its factory helper, before
    // initialize(); this is a no-op there and the real call for everyone else.
    multimodalRunner->loadExternalWeights(multimodalEngineDir, checkpointDir, stream);

    return multimodalRunner;
}

rt::Tensor& MultimodalRunner::getOutputEmbedding()
{
    return mOutputEmbedding;
}

rt::OptionalInputTensors MultimodalRunner::getDeepstackFeatures()
{
    return {};
}

bool MultimodalRunner::preprocessSystemPrompt([[maybe_unused]] std::string const& systemPrompt,
    [[maybe_unused]] tokenizer::Tokenizer const* tokenizer, [[maybe_unused]] rt::OptionalOutputTensor mropeCosSinOut,
    [[maybe_unused]] cudaStream_t stream)
{
    // Default implementation is to do nothing for system prompt preprocessing and ND-RoPE parameter generation.
    return true;
}

} // namespace rt
} // namespace trt_edgellm
