/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/motBackboneRunner.h"

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
    throw std::runtime_error("MotBackboneRunner: unsupported tensor dtype: " + dtype);
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
                    "MotBackboneRunner: tensor name '%s' is not present in the TensorRT engine; using binding '%s'",
                    preferredName.c_str(), name);
                return name;
            }
            ++seen;
        }
    }

    throw std::runtime_error("MotBackboneRunner: failed to resolve TensorRT I/O tensor name: " + preferredName);
}

std::string resolveEnginePath(std::string const& engineDir, nlohmann::json const& configJson)
{
    std::string const engineFile = configJson.value("engine_file", std::string{"mot_backbone.engine"});
    std::filesystem::path const preferred = std::filesystem::path(engineDir) / engineFile;
    if (std::filesystem::exists(preferred))
    {
        return preferred.string();
    }

    for (char const* candidate : {"mot_backbone.engine", "backbone.engine"})
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
        LOG_ERROR("MotBackboneRunner: copy shape mismatch, dst=%ld src=%ld", elements, src.getShape().volume());
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

    LOG_ERROR("MotBackboneRunner: unsupported dtype conversion src=%d dst=%d", static_cast<int>(srcType),
        static_cast<int>(dstType));
    return false;
}

} // namespace

MotBackboneRunner::MotBackboneRunner(std::string const& engineDir, cudaStream_t stream)
    : mEngineDir(engineDir)
{
    if (!loadConfig())
    {
        throw std::runtime_error("MotBackboneRunner: failed to load config from " + engineDir);
    }
    if (!loadEngine(stream))
    {
        throw std::runtime_error("MotBackboneRunner: failed to load TensorRT engine from " + engineDir);
    }
    if (!validateAndFillConfig())
    {
        throw std::runtime_error("MotBackboneRunner: failed to validate config for " + engineDir);
    }
    if (!allocateBuffers())
    {
        throw std::runtime_error("MotBackboneRunner: failed to allocate buffers for " + engineDir);
    }

    LOG_INFO("MotBackboneRunner loaded from %s (und_seq=%s, gen_seq=%s, last_hidden=%s)", engineDir.c_str(),
        mUndSeqShape.formatString().c_str(), mGenSeqShape.formatString().c_str(), mOutputShape.formatString().c_str());
}

bool MotBackboneRunner::loadConfig()
{
    auto const configPath = mEngineDir + "/config.json";
    std::ifstream configFile(configPath);
    if (!configFile)
    {
        LOG_ERROR("MotBackboneRunner: failed to open config file: %s", configPath.c_str());
        return false;
    }

    try
    {
        configFile >> mConfigJson;
    }
    catch (nlohmann::json::parse_error const& e)
    {
        LOG_ERROR("MotBackboneRunner: failed to parse config file: %s", e.what());
        return false;
    }
    return true;
}

bool MotBackboneRunner::loadEngine(cudaStream_t stream)
{
    auto const enginePath = resolveEnginePath(mEngineDir, mConfigJson);
    if (!std::filesystem::exists(enginePath))
    {
        LOG_ERROR("MotBackboneRunner: engine not found at %s", enginePath.c_str());
        return false;
    }

    file_io::MmapReader engineFileReader(enginePath);
    mRuntime.reset(nvinfer1::createInferRuntime(gLogger));
    if (!mRuntime)
    {
        LOG_ERROR("MotBackboneRunner: failed to create TensorRT runtime for %s", enginePath.c_str());
        return false;
    }

    mEngine.reset(mRuntime->deserializeCudaEngine(engineFileReader.getData(), engineFileReader.getSize()));
    if (!mEngine)
    {
        LOG_ERROR("MotBackboneRunner: failed to deserialize TensorRT engine: %s", enginePath.c_str());
        return false;
    }

    mContext.reset(mEngine->createExecutionContext(nvinfer1::ExecutionContextAllocationStrategy::kUSER_MANAGED));
    if (!mContext)
    {
        LOG_ERROR("MotBackboneRunner: failed to create TensorRT execution context for %s", enginePath.c_str());
        return false;
    }

    if (!mContext->setOptimizationProfileAsync(0, stream))
    {
        LOG_ERROR("MotBackboneRunner: failed to set optimization profile for %s", enginePath.c_str());
        return false;
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return true;
}

bool MotBackboneRunner::validateAndFillConfig()
{
    auto const modelType = mConfigJson.value("model_type", std::string{"cosmos_mot_backbone"});
    auto const component = mConfigJson.value("component", std::string{"mot_backbone"});
    if (modelType != "cosmos_mot_backbone" && component != "mot_backbone")
    {
        LOG_ERROR("MotBackboneRunner: unexpected model_type=%s component=%s", modelType.c_str(), component.c_str());
        return false;
    }

    if (mConfigJson.contains("input_names") && mConfigJson.at("input_names").size() >= 6U)
    {
        auto const& names = mConfigJson.at("input_names");
        mUndSeqInputName = names.at(0).get<std::string>();
        mGenSeqInputName = names.at(1).get<std::string>();
        mCosUndInputName = names.at(2).get<std::string>();
        mSinUndInputName = names.at(3).get<std::string>();
        mCosGenInputName = names.at(4).get<std::string>();
        mSinGenInputName = names.at(5).get<std::string>();
    }
    if (mConfigJson.contains("output_names") && !mConfigJson.at("output_names").empty())
    {
        mOutputName = mConfigJson.at("output_names").at(0).get<std::string>();
    }

    if (!mConfigJson.contains("inputs"))
    {
        LOG_ERROR("MotBackboneRunner: config is missing inputs metadata");
        return false;
    }
    if (!mConfigJson.contains("outputs") || mConfigJson.at("outputs").empty())
    {
        LOG_ERROR("MotBackboneRunner: config is missing output metadata");
        return false;
    }

    auto const& inputs = mConfigJson.at("inputs");
    for (auto const* name :
        {&mUndSeqInputName, &mGenSeqInputName, &mCosUndInputName, &mSinUndInputName, &mCosGenInputName, &mSinGenInputName})
    {
        if (!inputs.contains(*name))
        {
            LOG_ERROR("MotBackboneRunner: config is missing input metadata for %s", name->c_str());
            return false;
        }
    }

    auto const& undMeta = inputs.at(mUndSeqInputName);
    auto const& genMeta = inputs.at(mGenSeqInputName);
    auto const& cosUndMeta = inputs.at(mCosUndInputName);
    auto const& cosGenMeta = inputs.at(mCosGenInputName);
    auto const& outputMeta = mConfigJson.at("outputs").at(0);

    mUndSeqShape = coordsFromJson(undMeta.at("shape"));
    mGenSeqShape = coordsFromJson(genMeta.at("shape"));
    mRotaryUndShape = coordsFromJson(cosUndMeta.at("shape"));
    mRotaryGenShape = coordsFromJson(cosGenMeta.at("shape"));
    mOutputShape = coordsFromJson(outputMeta.at("shape"));

    mSeqType = dataTypeFromTorchString(undMeta.at("dtype").get<std::string>());
    mRotaryType = dataTypeFromTorchString(cosUndMeta.at("dtype").get<std::string>());
    mOutputType = dataTypeFromTorchString(outputMeta.at("dtype").get<std::string>());

    if (mUndSeqShape.getNumDims() != 2 || mGenSeqShape.getNumDims() != 2)
    {
        LOG_ERROR("MotBackboneRunner: expected rank-2 und_seq and gen_seq tensors");
        return false;
    }
    if (mRotaryUndShape.getNumDims() != 2 || mRotaryGenShape.getNumDims() != 2)
    {
        LOG_ERROR("MotBackboneRunner: expected rank-2 rotary tensors");
        return false;
    }
    if (mOutputShape.getNumDims() != 2)
    {
        LOG_ERROR("MotBackboneRunner: expected rank-2 last_hidden_state output");
        return false;
    }
    if (mUndSeqShape[0] + mGenSeqShape[0] != mOutputShape[0])
    {
        LOG_ERROR("MotBackboneRunner: output seq dim %ld != und_len %ld + gen_len %ld", mOutputShape[0], mUndSeqShape[0],
            mGenSeqShape[0]);
        return false;
    }

    mUndSeqInputName = resolveIOTensorName(mEngine.get(), mUndSeqInputName, nvinfer1::TensorIOMode::kINPUT, 0);
    mGenSeqInputName = resolveIOTensorName(mEngine.get(), mGenSeqInputName, nvinfer1::TensorIOMode::kINPUT, 1);
    mCosUndInputName = resolveIOTensorName(mEngine.get(), mCosUndInputName, nvinfer1::TensorIOMode::kINPUT, 2);
    mSinUndInputName = resolveIOTensorName(mEngine.get(), mSinUndInputName, nvinfer1::TensorIOMode::kINPUT, 3);
    mCosGenInputName = resolveIOTensorName(mEngine.get(), mCosGenInputName, nvinfer1::TensorIOMode::kINPUT, 4);
    mSinGenInputName = resolveIOTensorName(mEngine.get(), mSinGenInputName, nvinfer1::TensorIOMode::kINPUT, 5);
    mOutputName = resolveIOTensorName(mEngine.get(), mOutputName, nvinfer1::TensorIOMode::kOUTPUT, 0);
    return true;
}

int64_t MotBackboneRunner::getRequiredContextMemorySize() const
{
    return mEngine ? mEngine->getDeviceMemorySizeV2() : 0;
}

bool MotBackboneRunner::allocateBuffers()
{
    mUndSeqTensor = rt::Tensor(mUndSeqShape, rt::DeviceType::kGPU, mSeqType, mUndSeqInputName);
    mGenSeqTensor = rt::Tensor(mGenSeqShape, rt::DeviceType::kGPU, mSeqType, mGenSeqInputName);
    mCosUndTensor = rt::Tensor(mRotaryUndShape, rt::DeviceType::kGPU, mRotaryType, mCosUndInputName);
    mSinUndTensor = rt::Tensor(mRotaryUndShape, rt::DeviceType::kGPU, mRotaryType, mSinUndInputName);
    mCosGenTensor = rt::Tensor(mRotaryGenShape, rt::DeviceType::kGPU, mRotaryType, mCosGenInputName);
    mSinGenTensor = rt::Tensor(mRotaryGenShape, rt::DeviceType::kGPU, mRotaryType, mSinGenInputName);
    mLastHiddenTensor = rt::Tensor(mOutputShape, rt::DeviceType::kGPU, mOutputType, mOutputName);
    return bindTensors();
}

bool MotBackboneRunner::bindTensors() noexcept
{
    struct InputBinding
    {
        std::string const& name;
        rt::Tensor& tensor;
    };

    InputBinding const bindings[] = {{mUndSeqInputName, mUndSeqTensor}, {mGenSeqInputName, mGenSeqTensor},
        {mCosUndInputName, mCosUndTensor}, {mSinUndInputName, mSinUndTensor}, {mCosGenInputName, mCosGenTensor},
        {mSinGenInputName, mSinGenTensor}};

    for (auto const& binding : bindings)
    {
        if (!mContext->setInputShape(binding.name.c_str(), binding.tensor.getShape().getTRTDims()))
        {
            LOG_ERROR("MotBackboneRunner: failed to set input shape for %s", binding.name.c_str());
            return false;
        }
        if (!mContext->setTensorAddress(binding.name.c_str(), binding.tensor.rawPointer()))
        {
            LOG_ERROR("MotBackboneRunner: failed to bind input tensor %s", binding.name.c_str());
            return false;
        }
    }

    if (!mContext->setTensorAddress(mOutputName.c_str(), mLastHiddenTensor.rawPointer()))
    {
        LOG_ERROR("MotBackboneRunner: failed to bind output tensor %s", mOutputName.c_str());
        return false;
    }
    return true;
}

bool MotBackboneRunner::setContextMemory(rt::Tensor& sharedContextMemory)
{
    if (!mEngine)
    {
        return true;
    }

    int64_t const requiredSize = getRequiredContextMemorySize();
    if (sharedContextMemory.getMemoryCapacity() < requiredSize)
    {
        LOG_ERROR("MotBackboneRunner: shared context memory (%zu bytes) is smaller than required (%zu bytes)",
            static_cast<size_t>(sharedContextMemory.getMemoryCapacity()), static_cast<size_t>(requiredSize));
        return false;
    }

    mContext->setDeviceMemoryV2(sharedContextMemory.rawPointer(), sharedContextMemory.getMemoryCapacity());
    return true;
}

bool MotBackboneRunner::copyTensorInput(
    rt::Tensor& dst, rt::Tensor const& src, char const* tensorLabel, cudaStream_t stream)
{
    if (src.getShape() != dst.getShape())
    {
        LOG_ERROR("MotBackboneRunner: %s shape %s does not match engine input %s", tensorLabel,
            src.getShape().formatString().c_str(), dst.getShape().formatString().c_str());
        return false;
    }
    return copyTensorToDevice(dst, src, stream);
}

bool MotBackboneRunner::copyUndSeqFrom(rt::Tensor const& src, cudaStream_t stream)
{
    return copyTensorInput(mUndSeqTensor, src, "und_seq", stream);
}

bool MotBackboneRunner::copyGenSeqFrom(rt::Tensor const& src, cudaStream_t stream)
{
    return copyTensorInput(mGenSeqTensor, src, "gen_seq", stream);
}

bool MotBackboneRunner::copyRotaryFrom(CosmosTextPhase0 const& phase0, cudaStream_t stream)
{
    rt::Coords const expectedCosUnd({phase0.undLen, mRotaryUndShape[1]});
    rt::Coords const expectedSinUnd({phase0.undLen, mRotaryUndShape[1]});
    rt::Coords const expectedCosGen({phase0.genLen, mRotaryGenShape[1]});
    rt::Coords const expectedSinGen({phase0.genLen, mRotaryGenShape[1]});

    if (expectedCosUnd != mRotaryUndShape || expectedSinUnd != mRotaryUndShape || expectedCosGen != mRotaryGenShape
        || expectedSinGen != mRotaryGenShape)
    {
        LOG_ERROR(
            "MotBackboneRunner: rotary shapes from Phase 0 (und=%s gen=%s) do not match engine bindings (und=%s gen=%s).",
            expectedCosUnd.formatString().c_str(), expectedCosGen.formatString().c_str(),
            mRotaryUndShape.formatString().c_str(), mRotaryGenShape.formatString().c_str());
        return false;
    }

    return copyTensorInput(mCosUndTensor, phase0.cosUnd, "cos_und", stream)
        && copyTensorInput(mSinUndTensor, phase0.sinUnd, "sin_und", stream)
        && copyTensorInput(mCosGenTensor, phase0.cosGen, "cos_gen", stream)
        && copyTensorInput(mSinGenTensor, phase0.sinGen, "sin_gen", stream);
}

bool MotBackboneRunner::runBackbone(cudaStream_t stream) noexcept
{
    if (!bindTensors())
    {
        return false;
    }
    if (!mContext->enqueueV3(stream))
    {
        LOG_ERROR("MotBackboneRunner: enqueueV3 failed.");
        return false;
    }
    return true;
}

} // namespace rt
} // namespace trt_edgellm
