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

/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/denoiseHeadRunner.h"

#include "common/checkMacros.h"
#include "common/cudaUtils.h"
#include "common/logger.h"
#include "common/mmapReader.h"

#include <cuda_fp16.h>
#include <filesystem>
#include <fstream>
#include <stdexcept>

namespace trt_edgellm
{
namespace rt
{
namespace
{

rt::Coords coordsFromJson(nlohmann::json const& shape)
{
    std::vector<int64_t> dims;
    dims.reserve(shape.size());
    for (auto const& dim : shape)
    {
        dims.push_back(dim.get<int64_t>());
    }
    return rt::Coords(dims);
}

nvinfer1::DataType dataTypeFromTorchString(std::string const& dtype)
{
    if (dtype == "torch.float16" || dtype == "float16" || dtype == "fp16")
    {
        return nvinfer1::DataType::kHALF;
    }
    if (dtype == "torch.float32" || dtype == "float32" || dtype == "fp32")
    {
        return nvinfer1::DataType::kFLOAT;
    }
    throw std::runtime_error("DenoiseHeadRunner: unsupported tensor dtype: " + dtype);
}

std::string resolveIOTensorName(nvinfer1::ICudaEngine const* engine, std::string const& preferredName,
    nvinfer1::TensorIOMode mode, int32_t modeIndex = 0)
{
    for (int32_t i = 0; i < engine->getNbIOTensors(); ++i)
    {
        auto const* name = engine->getIOTensorName(i);
        if (name != nullptr && preferredName == name && engine->getTensorIOMode(name) == mode)
        {
            return preferredName;
        }
    }

    int32_t seen = 0;
    for (int32_t i = 0; i < engine->getNbIOTensors(); ++i)
    {
        auto const* name = engine->getIOTensorName(i);
        if (name != nullptr && engine->getTensorIOMode(name) == mode)
        {
            if (seen == modeIndex)
            {
                LOG_WARNING(
                    "DenoiseHeadRunner: tensor name '%s' is not present in the TensorRT engine; using binding '%s'",
                    preferredName.c_str(), name);
                return name;
            }
            ++seen;
        }
    }

    throw std::runtime_error("DenoiseHeadRunner: failed to resolve TensorRT I/O tensor name: " + preferredName);
}

std::string resolveEnginePath(std::string const& engineDir, nlohmann::json const& configJson)
{
    std::string const engineFile = configJson.value("engine_file", std::string{"denoise_head.engine"});
    std::filesystem::path const preferred = std::filesystem::path(engineDir) / engineFile;
    if (std::filesystem::exists(preferred))
    {
        return preferred.string();
    }

    for (char const* candidate : {"denoise_head.engine", "head.engine"})
    {
        std::filesystem::path const path = std::filesystem::path(engineDir) / candidate;
        if (std::filesystem::exists(path))
        {
            return path.string();
        }
    }

    return preferred.string();
}

int64_t tensorBytes(rt::Tensor const& tensor)
{
    return static_cast<int64_t>(tensor.getShape().volume()) * rt::utils::getTypeSize(tensor.getDataType());
}

bool copyTensorToDevice(rt::Tensor& dst, rt::Tensor const& src, cudaStream_t stream)
{
    auto const dstBytes = tensorBytes(dst);
    auto const srcBytes = tensorBytes(src);
    if (dstBytes == srcBytes)
    {
        CUDA_CHECK(cudaMemcpyAsync(
            dst.rawPointer(), src.rawPointer(), static_cast<size_t>(dstBytes), cudaMemcpyDeviceToDevice, stream));
        return true;
    }

    auto const elements = dst.getShape().volume();
    if (src.getShape().volume() != elements)
    {
        LOG_ERROR("DenoiseHeadRunner: copy shape mismatch, dst=%ld src=%ld", elements, src.getShape().volume());
        return false;
    }

    auto const srcType = src.getDataType();
    auto const dstType = dst.getDataType();
    if (srcType == nvinfer1::DataType::kHALF && dstType == nvinfer1::DataType::kFLOAT)
    {
        std::vector<half> hostSrc(static_cast<std::size_t>(elements));
        CUDA_CHECK(cudaMemcpyAsync(
            hostSrc.data(), src.rawPointer(), static_cast<size_t>(srcBytes), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        std::vector<float> hostDst(static_cast<std::size_t>(elements));
        for (int64_t i = 0; i < elements; ++i)
        {
            hostDst[static_cast<std::size_t>(i)] = __half2float(hostSrc[static_cast<std::size_t>(i)]);
        }
        CUDA_CHECK(cudaMemcpyAsync(
            dst.rawPointer(), hostDst.data(), static_cast<size_t>(dstBytes), cudaMemcpyHostToDevice, stream));
        return true;
    }

    if (srcType == nvinfer1::DataType::kFLOAT && dstType == nvinfer1::DataType::kHALF)
    {
        std::vector<float> hostSrc(static_cast<std::size_t>(elements));
        CUDA_CHECK(cudaMemcpyAsync(
            hostSrc.data(), src.rawPointer(), static_cast<size_t>(srcBytes), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        std::vector<half> hostDst(static_cast<std::size_t>(elements));
        for (int64_t i = 0; i < elements; ++i)
        {
            hostDst[static_cast<std::size_t>(i)] = __float2half(hostSrc[static_cast<std::size_t>(i)]);
        }
        CUDA_CHECK(cudaMemcpyAsync(
            dst.rawPointer(), hostDst.data(), static_cast<size_t>(dstBytes), cudaMemcpyHostToDevice, stream));
        return true;
    }

    LOG_ERROR("DenoiseHeadRunner: unsupported dtype conversion src=%d dst=%d", static_cast<int>(srcType),
        static_cast<int>(dstType));
    return false;
}

} // namespace

DenoiseHeadRunner::DenoiseHeadRunner(std::string const& engineDir, cudaStream_t stream)
    : mEngineDir(engineDir)
{
    if (!loadConfig())
    {
        throw std::runtime_error("DenoiseHeadRunner: failed to load config from " + engineDir);
    }
    if (!loadEngine(stream))
    {
        throw std::runtime_error("DenoiseHeadRunner: failed to load TensorRT engine from " + engineDir);
    }
    if (!validateAndFillConfig())
    {
        throw std::runtime_error("DenoiseHeadRunner: failed to validate config for " + engineDir);
    }
    if (!allocateBuffers())
    {
        throw std::runtime_error("DenoiseHeadRunner: failed to allocate buffers for " + engineDir);
    }

    LOG_INFO("DenoiseHeadRunner loaded from %s (%s -> %s, last_hidden=%s, pred_latents=%s)", engineDir.c_str(),
        mInputName.c_str(), mOutputName.c_str(), mInputShape.formatString().c_str(),
        mOutputShape.formatString().c_str());
}

bool DenoiseHeadRunner::loadConfig()
{
    auto const configPath = mEngineDir + "/config.json";
    std::ifstream configFile(configPath);
    if (!configFile)
    {
        LOG_ERROR("DenoiseHeadRunner: failed to open config file: %s", configPath.c_str());
        return false;
    }

    try
    {
        configFile >> mConfigJson;
    }
    catch (nlohmann::json::parse_error const& e)
    {
        LOG_ERROR("DenoiseHeadRunner: failed to parse config file: %s", e.what());
        return false;
    }
    return true;
}

bool DenoiseHeadRunner::loadEngine(cudaStream_t stream)
{
    auto const enginePath = resolveEnginePath(mEngineDir, mConfigJson);
    if (!std::filesystem::exists(enginePath))
    {
        LOG_ERROR("DenoiseHeadRunner: engine not found at %s", enginePath.c_str());
        return false;
    }

    file_io::MmapReader engineFileReader(enginePath);
    mRuntime.reset(nvinfer1::createInferRuntime(gLogger));
    if (!mRuntime)
    {
        LOG_ERROR("DenoiseHeadRunner: failed to create TensorRT runtime for %s", enginePath.c_str());
        return false;
    }

    mEngine.reset(mRuntime->deserializeCudaEngine(engineFileReader.getData(), engineFileReader.getSize()));
    if (!mEngine)
    {
        LOG_ERROR("DenoiseHeadRunner: failed to deserialize TensorRT engine: %s", enginePath.c_str());
        return false;
    }

    mContext.reset(mEngine->createExecutionContext(nvinfer1::ExecutionContextAllocationStrategy::kUSER_MANAGED));
    if (!mContext)
    {
        LOG_ERROR("DenoiseHeadRunner: failed to create TensorRT execution context for %s", enginePath.c_str());
        return false;
    }

    if (!mContext->setOptimizationProfileAsync(0, stream))
    {
        LOG_ERROR("DenoiseHeadRunner: failed to set optimization profile for %s", enginePath.c_str());
        return false;
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return true;
}

bool DenoiseHeadRunner::validateAndFillConfig()
{
    auto const modelType = mConfigJson.value("model_type", std::string{"cosmos_denoise_head"});
    auto const component = mConfigJson.value("component", std::string{"denoise_head"});
    bool const validVisionHead = modelType == "cosmos_denoise_head" || component == "denoise_head";
    bool const validSoundHead = modelType == "cosmos_denoise_head_sound" || component == "denoise_head_sound";
    if (!validVisionHead && !validSoundHead)
    {
        LOG_ERROR("DenoiseHeadRunner: unexpected model_type=%s component=%s", modelType.c_str(), component.c_str());
        return false;
    }

    if (mConfigJson.contains("input_names") && !mConfigJson.at("input_names").empty())
    {
        mInputName = mConfigJson.at("input_names").at(0).get<std::string>();
    }
    if (mConfigJson.contains("output_names") && !mConfigJson.at("output_names").empty())
    {
        mOutputName = mConfigJson.at("output_names").at(0).get<std::string>();
    }

    if (!mConfigJson.contains("inputs") || !mConfigJson.at("inputs").contains(mInputName))
    {
        LOG_ERROR("DenoiseHeadRunner: config is missing input metadata for %s", mInputName.c_str());
        return false;
    }
    if (!mConfigJson.contains("outputs") || mConfigJson.at("outputs").empty())
    {
        LOG_ERROR("DenoiseHeadRunner: config is missing output metadata");
        return false;
    }

    auto const& inputMeta = mConfigJson.at("inputs").at(mInputName);
    auto const& outputMeta = mConfigJson.at("outputs").at(0);

    mInputShape = coordsFromJson(inputMeta.at("shape"));
    mOutputShape = coordsFromJson(outputMeta.at("shape"));
    mInputType = dataTypeFromTorchString(inputMeta.at("dtype").get<std::string>());
    mOutputType = dataTypeFromTorchString(outputMeta.at("dtype").get<std::string>());

    if (mInputShape.getNumDims() != 2)
    {
        LOG_ERROR("DenoiseHeadRunner: expected rank-2 last_hidden_state input, got %d dims", mInputShape.getNumDims());
        return false;
    }
    if (mOutputShape.getNumDims() != 3 && mOutputShape.getNumDims() != 5)
    {
        LOG_ERROR(
            "DenoiseHeadRunner: expected rank-3 pred_sound_latents or rank-5 pred_vision_latents output, got %d dims",
            mOutputShape.getNumDims());
        return false;
    }

    mInputName = resolveIOTensorName(mEngine.get(), mInputName, nvinfer1::TensorIOMode::kINPUT, 0);
    mOutputName = resolveIOTensorName(mEngine.get(), mOutputName, nvinfer1::TensorIOMode::kOUTPUT, 0);
    return true;
}

int64_t DenoiseHeadRunner::getRequiredContextMemorySize() const
{
    return mEngine ? mEngine->getDeviceMemorySizeV2() : 0;
}

bool DenoiseHeadRunner::allocateBuffers()
{
    mLastHiddenTensor = rt::Tensor(mInputShape, rt::DeviceType::kGPU, mInputType, mInputName);
    mPredLatentsTensor = rt::Tensor(mOutputShape, rt::DeviceType::kGPU, mOutputType, mOutputName);
    return bindTensors();
}

bool DenoiseHeadRunner::bindTensors() noexcept
{
    if (!mContext->setInputShape(mInputName.c_str(), mLastHiddenTensor.getShape().getTRTDims()))
    {
        LOG_ERROR("DenoiseHeadRunner: failed to set input shape for %s", mInputName.c_str());
        return false;
    }
    if (!mContext->setTensorAddress(mInputName.c_str(), mLastHiddenTensor.rawPointer()))
    {
        LOG_ERROR("DenoiseHeadRunner: failed to bind input tensor %s", mInputName.c_str());
        return false;
    }
    if (!mContext->setTensorAddress(mOutputName.c_str(), mPredLatentsTensor.rawPointer()))
    {
        LOG_ERROR("DenoiseHeadRunner: failed to bind output tensor %s", mOutputName.c_str());
        return false;
    }
    return true;
}

bool DenoiseHeadRunner::setContextMemory(rt::Tensor& sharedContextMemory)
{
    if (!mEngine)
    {
        return true;
    }

    int64_t const requiredSize = getRequiredContextMemorySize();
    if (sharedContextMemory.getMemoryCapacity() < requiredSize)
    {
        LOG_ERROR("DenoiseHeadRunner: shared context memory (%zu bytes) is smaller than required (%zu bytes)",
            static_cast<size_t>(sharedContextMemory.getMemoryCapacity()), static_cast<size_t>(requiredSize));
        return false;
    }

    mContext->setDeviceMemoryV2(sharedContextMemory.rawPointer(), sharedContextMemory.getMemoryCapacity());
    return true;
}

bool DenoiseHeadRunner::copyLastHiddenFrom(rt::Tensor const& src, cudaStream_t stream)
{
    if (src.getShape() != mLastHiddenTensor.getShape())
    {
        LOG_ERROR("DenoiseHeadRunner: last_hidden shape %s does not match engine input %s",
            src.getShape().formatString().c_str(), mLastHiddenTensor.getShape().formatString().c_str());
        return false;
    }
    return copyTensorToDevice(mLastHiddenTensor, src, stream);
}

bool DenoiseHeadRunner::runHead(cudaStream_t stream) noexcept
{
    if (!bindTensors())
    {
        return false;
    }
    if (!mContext->enqueueV3(stream))
    {
        LOG_ERROR("DenoiseHeadRunner: enqueueV3 failed.");
        return false;
    }
    return true;
}

} // namespace rt
} // namespace trt_edgellm
