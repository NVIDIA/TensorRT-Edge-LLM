/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/embedRunner.h"

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
    throw std::runtime_error("EmbedRunner: unsupported tensor dtype: " + dtype);
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
                    "EmbedRunner: tensor name '%s' is not present in the TensorRT engine; using binding '%s'",
                    preferredName.c_str(), name);
                return name;
            }
            ++seen;
        }
    }

    throw std::runtime_error("EmbedRunner: failed to resolve TensorRT I/O tensor name: " + preferredName);
}

std::string resolveEnginePath(std::string const& engineDir, nlohmann::json const& configJson)
{
    std::string const engineFile = configJson.value("engine_file", std::string{"embed.engine"});
    std::filesystem::path const preferred = std::filesystem::path(engineDir) / engineFile;
    if (std::filesystem::exists(preferred))
    {
        return preferred.string();
    }

    for (char const* candidate : {"embed.engine", "gen_embed.engine"})
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
        CUDA_CHECK(cudaMemcpyAsync(dst.rawPointer(), src.rawPointer(), static_cast<size_t>(dstBytes),
            cudaMemcpyDeviceToDevice, stream));
        return true;
    }

    auto const elements = dst.getShape().volume();
    if (src.getShape().volume() != elements)
    {
        LOG_ERROR("EmbedRunner: copy shape mismatch, dst=%ld src=%ld", elements, src.getShape().volume());
        return false;
    }

    auto const srcType = src.getDataType();
    auto const dstType = dst.getDataType();
    if (srcType == nvinfer1::DataType::kHALF && dstType == nvinfer1::DataType::kFLOAT)
    {
        std::vector<half> hostSrc(static_cast<std::size_t>(elements));
        CUDA_CHECK(cudaMemcpyAsync(hostSrc.data(), src.rawPointer(), static_cast<size_t>(srcBytes),
            cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        std::vector<float> hostDst(static_cast<std::size_t>(elements));
        for (int64_t i = 0; i < elements; ++i)
        {
            hostDst[static_cast<std::size_t>(i)] = __half2float(hostSrc[static_cast<std::size_t>(i)]);
        }
        CUDA_CHECK(cudaMemcpyAsync(dst.rawPointer(), hostDst.data(), static_cast<size_t>(dstBytes),
            cudaMemcpyHostToDevice, stream));
        return true;
    }

    if (srcType == nvinfer1::DataType::kFLOAT && dstType == nvinfer1::DataType::kHALF)
    {
        std::vector<float> hostSrc(static_cast<std::size_t>(elements));
        CUDA_CHECK(cudaMemcpyAsync(hostSrc.data(), src.rawPointer(), static_cast<size_t>(srcBytes),
            cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        std::vector<half> hostDst(static_cast<std::size_t>(elements));
        for (int64_t i = 0; i < elements; ++i)
        {
            hostDst[static_cast<std::size_t>(i)] = __float2half(hostSrc[static_cast<std::size_t>(i)]);
        }
        CUDA_CHECK(cudaMemcpyAsync(dst.rawPointer(), hostDst.data(), static_cast<size_t>(dstBytes),
            cudaMemcpyHostToDevice, stream));
        return true;
    }

    LOG_ERROR("EmbedRunner: unsupported dtype conversion src=%d dst=%d", static_cast<int>(srcType),
        static_cast<int>(dstType));
    return false;
}

} // namespace

EmbedRunner::EmbedRunner(std::string const& engineDir, cudaStream_t stream)
    : mEngineDir(engineDir)
{
    if (!loadConfig())
    {
        throw std::runtime_error("EmbedRunner: failed to load config from " + engineDir);
    }
    if (!loadEngine(stream))
    {
        throw std::runtime_error("EmbedRunner: failed to load TensorRT engine from " + engineDir);
    }
    if (!validateAndFillConfig())
    {
        throw std::runtime_error("EmbedRunner: failed to validate config for " + engineDir);
    }
    if (!allocateBuffers())
    {
        throw std::runtime_error("EmbedRunner: failed to allocate buffers for " + engineDir);
    }

    LOG_INFO("EmbedRunner loaded from %s (%s + %s -> %s, latents=%s, gen_seq=%s)", engineDir.c_str(),
        mLatentsInputName.c_str(), mTimestepInputName.c_str(), mOutputName.c_str(),
        mLatentsShape.formatString().c_str(), mOutputShape.formatString().c_str());
}

bool EmbedRunner::loadConfig()
{
    auto const configPath = mEngineDir + "/config.json";
    std::ifstream configFile(configPath);
    if (!configFile)
    {
        LOG_ERROR("EmbedRunner: failed to open config file: %s", configPath.c_str());
        return false;
    }

    try
    {
        configFile >> mConfigJson;
    }
    catch (nlohmann::json::parse_error const& e)
    {
        LOG_ERROR("EmbedRunner: failed to parse config file: %s", e.what());
        return false;
    }
    return true;
}

bool EmbedRunner::loadEngine(cudaStream_t stream)
{
    auto const enginePath = resolveEnginePath(mEngineDir, mConfigJson);
    if (!std::filesystem::exists(enginePath))
    {
        LOG_ERROR("EmbedRunner: engine not found at %s", enginePath.c_str());
        return false;
    }

    file_io::MmapReader engineFileReader(enginePath);
    mRuntime.reset(nvinfer1::createInferRuntime(gLogger));
    if (!mRuntime)
    {
        LOG_ERROR("EmbedRunner: failed to create TensorRT runtime for %s", enginePath.c_str());
        return false;
    }

    mEngine.reset(mRuntime->deserializeCudaEngine(engineFileReader.getData(), engineFileReader.getSize()));
    if (!mEngine)
    {
        LOG_ERROR("EmbedRunner: failed to deserialize TensorRT engine: %s", enginePath.c_str());
        return false;
    }

    mContext.reset(mEngine->createExecutionContext(nvinfer1::ExecutionContextAllocationStrategy::kUSER_MANAGED));
    if (!mContext)
    {
        LOG_ERROR("EmbedRunner: failed to create TensorRT execution context for %s", enginePath.c_str());
        return false;
    }

    if (!mContext->setOptimizationProfileAsync(0, stream))
    {
        LOG_ERROR("EmbedRunner: failed to set optimization profile for %s", enginePath.c_str());
        return false;
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return true;
}

bool EmbedRunner::validateAndFillConfig()
{
    auto const modelType = mConfigJson.value("model_type", std::string{"cosmos_embed"});
    auto const component = mConfigJson.value("component", std::string{"embed"});
    if (modelType != "cosmos_embed" && component != "embed")
    {
        LOG_ERROR("EmbedRunner: unexpected model_type=%s component=%s", modelType.c_str(), component.c_str());
        return false;
    }

    if (mConfigJson.contains("input_names") && mConfigJson.at("input_names").size() >= 2U)
    {
        mLatentsInputName = mConfigJson.at("input_names").at(0).get<std::string>();
        mTimestepInputName = mConfigJson.at("input_names").at(1).get<std::string>();
    }
    if (mConfigJson.contains("output_names") && !mConfigJson.at("output_names").empty())
    {
        mOutputName = mConfigJson.at("output_names").at(0).get<std::string>();
    }

    if (!mConfigJson.contains("inputs"))
    {
        LOG_ERROR("EmbedRunner: config is missing inputs metadata");
        return false;
    }
    if (!mConfigJson.contains("outputs") || mConfigJson.at("outputs").empty())
    {
        LOG_ERROR("EmbedRunner: config is missing output metadata");
        return false;
    }

    auto const& inputs = mConfigJson.at("inputs");
    if (!inputs.contains(mLatentsInputName) || !inputs.contains(mTimestepInputName))
    {
        LOG_ERROR("EmbedRunner: config is missing input metadata for %s and/or %s", mLatentsInputName.c_str(),
            mTimestepInputName.c_str());
        return false;
    }

    auto const& latentsMeta = inputs.at(mLatentsInputName);
    auto const& timestepMeta = inputs.at(mTimestepInputName);
    auto const& outputMeta = mConfigJson.at("outputs").at(0);

    mLatentsShape = coordsFromJson(latentsMeta.at("shape"));
    if (timestepMeta.at("shape").empty())
    {
        mTimestepIsScalar = true;
        mTimestepShape = rt::Coords({1});
    }
    else
    {
        mTimestepShape = coordsFromJson(timestepMeta.at("shape"));
    }
    mOutputShape = coordsFromJson(outputMeta.at("shape"));
    mLatentsType = dataTypeFromTorchString(latentsMeta.at("dtype").get<std::string>());
    mTimestepType = dataTypeFromTorchString(timestepMeta.at("dtype").get<std::string>());
    mOutputType = dataTypeFromTorchString(outputMeta.at("dtype").get<std::string>());

    if (mLatentsShape.getNumDims() != 5)
    {
        LOG_ERROR("EmbedRunner: expected rank-5 vision latents, got %d dims", mLatentsShape.getNumDims());
        return false;
    }
    if (mOutputShape.getNumDims() != 2)
    {
        LOG_ERROR("EmbedRunner: expected rank-2 gen_seq output, got %d dims", mOutputShape.getNumDims());
        return false;
    }

    mLatentsInputName = resolveIOTensorName(mEngine.get(), mLatentsInputName, nvinfer1::TensorIOMode::kINPUT, 0);
    mTimestepInputName = resolveIOTensorName(mEngine.get(), mTimestepInputName, nvinfer1::TensorIOMode::kINPUT, 1);
    mOutputName = resolveIOTensorName(mEngine.get(), mOutputName, nvinfer1::TensorIOMode::kOUTPUT, 0);
    return true;
}

int64_t EmbedRunner::getRequiredContextMemorySize() const
{
    return mEngine ? mEngine->getDeviceMemorySizeV2() : 0;
}

bool EmbedRunner::allocateBuffers()
{
    mLatentsTensor = rt::Tensor(mLatentsShape, rt::DeviceType::kGPU, mLatentsType, mLatentsInputName);
    mTimestepTensor = rt::Tensor(mTimestepShape, rt::DeviceType::kGPU, mTimestepType, mTimestepInputName);
    mGenSeqTensor = rt::Tensor(mOutputShape, rt::DeviceType::kGPU, mOutputType, mOutputName);
    return bindTensors();
}

bool EmbedRunner::bindTensors() noexcept
{
    if (!mContext->setInputShape(mLatentsInputName.c_str(), mLatentsTensor.getShape().getTRTDims()))
    {
        LOG_ERROR("EmbedRunner: failed to set input shape for %s", mLatentsInputName.c_str());
        return false;
    }
    nvinfer1::Dims timestepDims = mTimestepIsScalar ? nvinfer1::Dims{} : mTimestepTensor.getShape().getTRTDims();
    if (!mContext->setInputShape(mTimestepInputName.c_str(), timestepDims))
    {
        LOG_ERROR("EmbedRunner: failed to set input shape for %s", mTimestepInputName.c_str());
        return false;
    }
    if (!mContext->setTensorAddress(mLatentsInputName.c_str(), mLatentsTensor.rawPointer()))
    {
        LOG_ERROR("EmbedRunner: failed to bind input tensor %s", mLatentsInputName.c_str());
        return false;
    }
    if (!mContext->setTensorAddress(mTimestepInputName.c_str(), mTimestepTensor.rawPointer()))
    {
        LOG_ERROR("EmbedRunner: failed to bind input tensor %s", mTimestepInputName.c_str());
        return false;
    }
    if (!mContext->setTensorAddress(mOutputName.c_str(), mGenSeqTensor.rawPointer()))
    {
        LOG_ERROR("EmbedRunner: failed to bind output tensor %s", mOutputName.c_str());
        return false;
    }
    return true;
}

bool EmbedRunner::setContextMemory(rt::Tensor& sharedContextMemory)
{
    if (!mEngine)
    {
        return true;
    }

    int64_t const requiredSize = getRequiredContextMemorySize();
    if (sharedContextMemory.getMemoryCapacity() < requiredSize)
    {
        LOG_ERROR("EmbedRunner: shared context memory (%zu bytes) is smaller than required (%zu bytes)",
            static_cast<size_t>(sharedContextMemory.getMemoryCapacity()), static_cast<size_t>(requiredSize));
        return false;
    }

    mContext->setDeviceMemoryV2(sharedContextMemory.rawPointer(), sharedContextMemory.getMemoryCapacity());
    return true;
}

bool EmbedRunner::copyLatentsFrom(rt::Tensor const& src, cudaStream_t stream)
{
    if (src.getShape() != mLatentsTensor.getShape())
    {
        LOG_ERROR("EmbedRunner: latent shape %s does not match engine input %s", src.getShape().formatString().c_str(),
            mLatentsTensor.getShape().formatString().c_str());
        return false;
    }
    return copyTensorToDevice(mLatentsTensor, src, stream);
}

bool EmbedRunner::setTimestep(float timestep, cudaStream_t stream)
{
    if (mTimestepType == nvinfer1::DataType::kFLOAT)
    {
        if (mTimestepShape.volume() == 1)
        {
            CUDA_CHECK(cudaMemcpyAsync(mTimestepTensor.rawPointer(), &timestep, sizeof(float), cudaMemcpyHostToDevice,
                stream));
            return true;
        }
        std::vector<float> values(static_cast<std::size_t>(mTimestepShape.volume()), timestep);
        CUDA_CHECK(cudaMemcpyAsync(mTimestepTensor.rawPointer(), values.data(), values.size() * sizeof(float),
            cudaMemcpyHostToDevice, stream));
        return true;
    }

    if (mTimestepType == nvinfer1::DataType::kHALF)
    {
        half const value = __float2half(timestep);
        if (mTimestepShape.volume() == 1)
        {
            CUDA_CHECK(
                cudaMemcpyAsync(mTimestepTensor.rawPointer(), &value, sizeof(half), cudaMemcpyHostToDevice, stream));
            return true;
        }
        std::vector<half> values(static_cast<std::size_t>(mTimestepShape.volume()), value);
        CUDA_CHECK(cudaMemcpyAsync(mTimestepTensor.rawPointer(), values.data(), values.size() * sizeof(half),
            cudaMemcpyHostToDevice, stream));
        return true;
    }

    LOG_ERROR("EmbedRunner: unsupported timestep dtype.");
    return false;
}

bool EmbedRunner::embed(cudaStream_t stream) noexcept
{
    if (!bindTensors())
    {
        return false;
    }
    if (!mContext->enqueueV3(stream))
    {
        LOG_ERROR("EmbedRunner: enqueueV3 failed.");
        return false;
    }
    return true;
}

} // namespace rt
} // namespace trt_edgellm
