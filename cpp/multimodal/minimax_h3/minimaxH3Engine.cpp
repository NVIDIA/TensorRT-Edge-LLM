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

#include "multimodal/minimax_h3/minimaxH3Engine.h"

#include "common/checkMacros.h"
#include "common/logger.h"
#include "common/trtInterop.h"

#include <filesystem>
#include <fstream>

namespace trt_edgellm
{
namespace rt
{
namespace minimax_h3
{

MiniMaxH3Engine::MiniMaxH3Engine(std::string const& artifactDir, std::string const& component, cudaStream_t stream)
    : mComponent(component)
{
    std::filesystem::path const directory(artifactDir);
    std::filesystem::path const configPath = directory / "config.json";
    std::ifstream configFile(configPath);
    ELLM_CHECK(configFile.is_open(), "Missing MiniMax-H3 config: " + configPath.string());
    configFile >> mConfig;

    mRuntime.reset(nvinfer1::createInferRuntime(TrtLogger::instance()));
    ELLM_CHECK(mRuntime != nullptr, "Failed to create a TensorRT runtime");
    std::filesystem::path const enginePath = directory / (mComponent + ".engine");
    mEngine = deserializeCudaEngineFromFile(*mRuntime, enginePath);
    ELLM_CHECK(mEngine != nullptr, "Failed to load MiniMax-H3 engine: " + enginePath.string());
    mWeights.load(directory, configPath, stream);
    rt::refitEngineIfNeeded(*mEngine, rt::EngineRefitOptions{mWeights.takeRefitWeights(), stream});
    mContext.reset(mEngine->createExecutionContext(nvinfer1::ExecutionContextAllocationStrategy::kSTATIC));
    ELLM_CHECK(mContext != nullptr, "Failed to create a context for MiniMax-H3 " + mComponent);
    setNonBlockingAuxStreams(mContext.get(), mEngine.get(), mAuxStreams);
    if (mWeights.size() > 0)
    {
        mWeights.bindToContext(*mEngine, *mContext, mComponent);
    }
    LOG_INFO("Loaded MiniMax-H3 %s engine (%zu checkpoint weights, context memory %.1f MiB)", mComponent.c_str(),
        mWeights.size(), static_cast<double>(mEngine->getDeviceMemorySizeV2()) / (1024.0 * 1024.0));
}

void MiniMaxH3Engine::bind(char const* name, rt::Tensor& tensor)
{
    bind(name, tensor.rawPointer(), tensor.getShape());
}

void MiniMaxH3Engine::bind(char const* name, void* data, rt::Coords const& inputShape)
{
    if (mEngine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT)
    {
        ELLM_CHECK(mContext->setInputShape(name, toTrtDims(inputShape)),
            "Invalid shape for MiniMax-H3 " + mComponent + " input " + name);
    }
    ELLM_CHECK(mContext->setTensorAddress(name, data), "Failed to bind MiniMax-H3 " + mComponent + " tensor " + name);
}

void MiniMaxH3Engine::enqueue(cudaStream_t stream)
{
    ELLM_CHECK(mContext->enqueueV3(stream), "MiniMax-H3 " + mComponent + " enqueue failed");
}

} // namespace minimax_h3
} // namespace rt
} // namespace trt_edgellm
