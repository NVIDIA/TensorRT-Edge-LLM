/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "multimodalRunner.h"
#include "common/mmapReader.h"
#include "multimodal/internViTRunner.h"
#include "multimodal/phi4mmViTRunner.h"
#include "multimodal/qwenViTRunner.h"
#include "profiling/metrics.h"
#include "profiling/timer.h"
#include <algorithm>
#include <cstdlib>
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

    // Construct engine path from directory
    std::string enginePath = engineDir + "/visual.engine";

    // Load engine
    auto mmapReader = std::make_unique<file_io::MmapReader>(enginePath);
    mVisualEngine = std::unique_ptr<nvinfer1::ICudaEngine>(
        mRuntime->deserializeCudaEngine(mmapReader->getData(), mmapReader->getSize()));

    // Create context and set optimization profile
    mContext = std::unique_ptr<nvinfer1::IExecutionContext>(mVisualEngine->createExecutionContext());
    if (!mContext->setOptimizationProfileAsync(0, stream))
    {
        LOG_ERROR("Failed to set optimization profile to the engine");
        throw std::runtime_error("Failed to set optimization profile to the engine");
    }
}

std::unique_ptr<MultimodalRunner> MultimodalRunner::create(std::string const& multimodalEngineDir,
    int32_t llmMaxBatchSize, int64_t llmMaxPositionEmbeddings, cudaStream_t stream)
{
    std::unique_ptr<MultimodalRunner> multimodalRunner;

    // Read config.json to determine model type
    std::string configPath = multimodalEngineDir + "/config.json";
    std::ifstream configFileStream(configPath);
    if (!configFileStream.is_open())
    {
        throw std::runtime_error("Failed to open config file: " + configPath);
    }

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

    if (modelType == multimodal::ModelType::QWEN2_VL || modelType == multimodal::ModelType::QWEN2_5_VL
        || modelType == multimodal::ModelType::QWEN3_VL)
    {
        multimodalRunner
            = std::make_unique<QwenViTRunner>(multimodalEngineDir, llmMaxBatchSize, llmMaxPositionEmbeddings, stream);
    }
    else if (modelType == multimodal::ModelType::INTERNVL)
    {
        multimodalRunner = std::make_unique<InternViTRunner>(multimodalEngineDir, stream);
    }
    else if (modelType == multimodal::ModelType::PHI4MM)
    {
        multimodalRunner = std::make_unique<Phi4MMViTRunner>(multimodalEngineDir, stream);
    }
    else
    {
        throw std::runtime_error("Unsupported model type: " + modelTypeStr);
    }

    return multimodalRunner;
}

rt::Tensor& MultimodalRunner::getOutputEmbedding()
{
    return mOutputEmbedding;
}

rt::OptionalInputTensors MultimodalRunner::getExtraVisualFeatures()
{
    return {};
}

bool MultimodalRunner::preprocessSystemPrompt(std::string const& systemPrompt, tokenizer::Tokenizer* tokenizer,
    rt::Tensor& ropeRotaryCosSinDevice, cudaStream_t stream)
{
    // Default implementation is to do nothing for system prompt preprocessing and ND-RoPE parameter generation.
    return true;
}

} // namespace rt
} // namespace trt_edgellm
