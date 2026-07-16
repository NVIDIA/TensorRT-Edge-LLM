/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "action/actionContextRunner.h"

#include "common/checkMacros.h"
#include "common/logger.h"
#include "common/mmapReader.h"

#include <cuda_fp16.h>
#include <fstream>
#include <stdexcept>
#include <vector>

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
    throw std::runtime_error("ActionContextRunner: unsupported tensor dtype: " + dtype);
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
                    "ActionContextRunner: tensor name '%s' is not present in the TensorRT engine; using binding '%s'",
                    preferredName.c_str(), name);
                return name;
            }
            ++seen;
        }
    }

    throw std::runtime_error("ActionContextRunner: failed to resolve TensorRT I/O tensor name: " + preferredName);
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
        CUDA_CHECK(cudaMemcpyAsync(dst.rawPointer(), src.rawPointer(), dstBytes, cudaMemcpyDeviceToDevice, stream));
        return true;
    }

    auto const elements = dst.getShape().volume();
    if (src.getShape().volume() != elements)
    {
        LOG_ERROR("ActionContextRunner: copy shape mismatch, dst=%ld src=%ld", elements, src.getShape().volume());
        return false;
    }

    auto const srcType = src.getDataType();
    auto const dstType = dst.getDataType();
    if (srcType == nvinfer1::DataType::kHALF && dstType == nvinfer1::DataType::kFLOAT)
    {
        std::vector<half> hostSrc(static_cast<std::size_t>(elements));
        CUDA_CHECK(cudaMemcpyAsync(hostSrc.data(), src.rawPointer(), srcBytes, cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        std::vector<float> hostDst(static_cast<std::size_t>(elements));
        for (int64_t i = 0; i < elements; ++i)
        {
            hostDst[static_cast<std::size_t>(i)] = __half2float(hostSrc[static_cast<std::size_t>(i)]);
        }
        CUDA_CHECK(cudaMemcpyAsync(dst.rawPointer(), hostDst.data(), dstBytes, cudaMemcpyHostToDevice, stream));
        return true;
    }

    if (srcType == nvinfer1::DataType::kFLOAT && dstType == nvinfer1::DataType::kHALF)
    {
        std::vector<float> hostSrc(static_cast<std::size_t>(elements));
        CUDA_CHECK(cudaMemcpyAsync(hostSrc.data(), src.rawPointer(), srcBytes, cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        std::vector<half> hostDst(static_cast<std::size_t>(elements));
        for (int64_t i = 0; i < elements; ++i)
        {
            hostDst[static_cast<std::size_t>(i)] = __float2half(hostSrc[static_cast<std::size_t>(i)]);
        }
        CUDA_CHECK(cudaMemcpyAsync(dst.rawPointer(), hostDst.data(), dstBytes, cudaMemcpyHostToDevice, stream));
        return true;
    }

    LOG_ERROR("ActionContextRunner: unsupported dtype conversion src=%d dst=%d", static_cast<int>(srcType),
        static_cast<int>(dstType));
    return false;
}

} // namespace

ActionContextRunner::ActionContextRunner(std::string const& engineDir, cudaStream_t stream)
    : mEngineDir(engineDir)
{
    if (!loadConfig())
    {
        throw std::runtime_error("ActionContextRunner: failed to load config");
    }
    if (!loadEngine(stream))
    {
        throw std::runtime_error("ActionContextRunner: failed to load TensorRT engine");
    }
    if (!validateAndFillConfig())
    {
        throw std::runtime_error("ActionContextRunner: failed to validate config");
    }
    if (!allocateBuffers())
    {
        throw std::runtime_error("ActionContextRunner: failed to allocate buffers");
    }
}

bool ActionContextRunner::loadConfig()
{
    auto const configPath = mEngineDir + "/config.json";
    std::ifstream configFile(configPath);
    if (!configFile)
    {
        LOG_ERROR("ActionContextRunner: failed to open config file: %s", configPath.c_str());
        return false;
    }

    try
    {
        configFile >> mConfigJson;
    }
    catch (nlohmann::json::parse_error const& e)
    {
        LOG_ERROR("ActionContextRunner: failed to parse config file: %s", e.what());
        return false;
    }
    return true;
}

bool ActionContextRunner::loadEngine(cudaStream_t stream)
{
    auto const engineFile = mConfigJson.value("engine_file", std::string{"action_context.engine"});
    auto const enginePath = mEngineDir + "/" + engineFile;

    file_io::MmapReader engineFileReader(enginePath);
    mRuntime.reset(nvinfer1::createInferRuntime(gLogger));
    if (!mRuntime)
    {
        LOG_ERROR("ActionContextRunner: failed to create TensorRT runtime for %s", enginePath.c_str());
        return false;
    }

    mEngine.reset(mRuntime->deserializeCudaEngine(engineFileReader.getData(), engineFileReader.getSize()));
    if (!mEngine)
    {
        LOG_ERROR("ActionContextRunner: failed to deserialize TensorRT engine: %s", enginePath.c_str());
        return false;
    }

    mContext.reset(mEngine->createExecutionContext());
    if (!mContext)
    {
        LOG_ERROR("ActionContextRunner: failed to create TensorRT execution context for %s", enginePath.c_str());
        return false;
    }

    if (!mContext->setOptimizationProfileAsync(0, stream))
    {
        LOG_ERROR("ActionContextRunner: failed to set optimization profile for %s", enginePath.c_str());
        return false;
    }
    return true;
}

int64_t ActionContextRunner::getRequiredContextMemorySize() const
{
    return mEngine ? mEngine->getDeviceMemorySizeV2() : 0;
}

bool ActionContextRunner::setContextMemory(rt::Tensor& contextMemory)
{
    // action_context uses TensorRT-managed device memory (see loadEngine).
    (void) contextMemory;
    return true;
}

bool ActionContextRunner::validateAndFillConfig()
{
    auto const modelTypeStr = mConfigJson.value("model_type", std::string{"action_context"});
    if (modelTypeStr != "action_context")
    {
        LOG_ERROR("ActionContextRunner: invalid model type: %s", modelTypeStr.c_str());
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
        LOG_ERROR("ActionContextRunner: config is missing input metadata for %s", mInputName.c_str());
        return false;
    }
    if (!mConfigJson.contains("outputs") || mConfigJson.at("outputs").empty())
    {
        LOG_ERROR("ActionContextRunner: config is missing output metadata");
        return false;
    }

    auto const& inputMeta = mConfigJson.at("inputs").at(mInputName);
    auto const& outputMeta = mConfigJson.at("outputs").at(0);

    mInputShape = coordsFromJson(inputMeta.at("shape"));
    mOutputShape = coordsFromJson(outputMeta.at("shape"));
    mInputType = dataTypeFromTorchString(inputMeta.at("dtype").get<std::string>());
    mOutputType = dataTypeFromTorchString(outputMeta.at("dtype").get<std::string>());

    if (mInputShape.getNumDims() != 3 || mOutputShape.getNumDims() != 3)
    {
        LOG_ERROR("ActionContextRunner: expected rank-3 input/output tensors");
        return false;
    }

    mMaxBatchSize = static_cast<int32_t>(mInputShape[0]);
    mMaxSeqLen = static_cast<int32_t>(mInputShape[1]);
    mHiddenSize = static_cast<int32_t>(mInputShape[2]);
    mContextHiddenSize = static_cast<int32_t>(mOutputShape[2]);

    mMaxBatchSize = mConfigJson.value("max_batch_size", mMaxBatchSize);
    mMaxSeqLen = mConfigJson.value("context_seq_len", mMaxSeqLen);
    mContextHiddenSize = mConfigJson.value("context_hidden_size", mContextHiddenSize);

    mInputName = resolveIOTensorName(mEngine.get(), mInputName, nvinfer1::TensorIOMode::kINPUT, 0);
    mOutputName = resolveIOTensorName(mEngine.get(), mOutputName, nvinfer1::TensorIOMode::kOUTPUT, 0);
    return true;
}

bool ActionContextRunner::allocateBuffers()
{
    mInputTensor = rt::Tensor(mInputShape, rt::DeviceType::kGPU, mInputType, mInputName);
    mOutputTensor = rt::Tensor(mOutputShape, rt::DeviceType::kGPU, mOutputType, mOutputName);
    return bindTensors();
}

bool ActionContextRunner::reshapeForContext(int32_t batchSize, int32_t seqLen)
{
    if (batchSize <= 0 || batchSize > mMaxBatchSize)
    {
        LOG_ERROR("ActionContextRunner: batchSize=%d exceeds max batch=%d", batchSize, mMaxBatchSize);
        return false;
    }
    if (seqLen <= 0 || seqLen > mMaxSeqLen)
    {
        LOG_ERROR("ActionContextRunner: seqLen=%d exceeds max seq=%d", seqLen, mMaxSeqLen);
        return false;
    }

    auto inputShape = mInputShape;
    inputShape[0] = batchSize;
    inputShape[1] = seqLen;
    auto outputShape = mOutputShape;
    outputShape[0] = batchSize;
    outputShape[1] = seqLen;

    if (!mInputTensor.reshape(inputShape) || !mOutputTensor.reshape(outputShape))
    {
        LOG_ERROR("ActionContextRunner: failed to reshape tensors for batch=%d seq=%d", batchSize, seqLen);
        return false;
    }
    return bindTensors();
}

bool ActionContextRunner::bindTensors() noexcept
{
    if (!mContext->setInputShape(mInputName.c_str(), mInputTensor.getShape().getTRTDims()))
    {
        LOG_ERROR("ActionContextRunner: failed to set input shape for %s", mInputName.c_str());
        return false;
    }
    if (!mContext->setTensorAddress(mInputName.c_str(), mInputTensor.rawPointer()))
    {
        LOG_ERROR("ActionContextRunner: failed to bind input tensor %s", mInputName.c_str());
        return false;
    }
    if (!mContext->setTensorAddress(mOutputName.c_str(), mOutputTensor.rawPointer()))
    {
        LOG_ERROR("ActionContextRunner: failed to bind output tensor %s", mOutputName.c_str());
        return false;
    }
    return true;
}

bool ActionContextRunner::copyLmHiddenFrom(rt::Tensor const& lmHidden, cudaStream_t stream, int32_t validSeqLen)
{
    if (validSeqLen < 0)
    {
        return copyTensorToDevice(mInputTensor, lmHidden, stream);
    }

    if (lmHidden.getShape().getNumDims() != 3 || mInputTensor.getShape().getNumDims() != 3)
    {
        LOG_ERROR("ActionContextRunner: padded copy expects rank-3 lm_hidden tensors");
        return false;
    }

    int32_t const batch = static_cast<int32_t>(mInputTensor.getShape()[0]);
    int32_t const dstSeqLen = static_cast<int32_t>(mInputTensor.getShape()[1]);
    int32_t const hidden = static_cast<int32_t>(mInputTensor.getShape()[2]);

    CUDA_CHECK(cudaMemsetAsync(mInputTensor.rawPointer(), 0, tensorBytes(mInputTensor), stream));
    if (validSeqLen == 0)
    {
        return true;
    }

    if (validSeqLen > dstSeqLen)
    {
        LOG_ERROR(
            "ActionContextRunner: invalid validSeqLen=%d for padded copy (dst seq=%d)", validSeqLen, dstSeqLen);
        return false;
    }
    if (lmHidden.getShape()[0] != batch || lmHidden.getShape()[2] != hidden
        || lmHidden.getShape()[1] < validSeqLen)
    {
        LOG_ERROR("ActionContextRunner: lmHidden shape %s incompatible with padded copy to %s (validSeqLen=%d)",
            lmHidden.getShape().formatString().c_str(), mInputTensor.getShape().formatString().c_str(), validSeqLen);
        return false;
    }

    size_t const rowBytes = static_cast<size_t>(hidden) * rt::utils::getTypeSize(mInputTensor.getDataType());
    size_t const copyBytes = static_cast<size_t>(validSeqLen) * rowBytes;
    int64_t const srcSeqLen = lmHidden.getShape()[1];
    for (int32_t b = 0; b < batch; ++b)
    {
        auto* dstRow = static_cast<char*>(mInputTensor.rawPointer()) + static_cast<size_t>(b) * dstSeqLen * rowBytes;
        auto const* srcRow
            = static_cast<char const*>(lmHidden.rawPointer()) + static_cast<size_t>(b) * srcSeqLen * rowBytes;
        CUDA_CHECK(cudaMemcpyAsync(dstRow, srcRow, copyBytes, cudaMemcpyDeviceToDevice, stream));
    }
    return true;
}

bool ActionContextRunner::resetExecutionContext(cudaStream_t stream)
{
    mContext.reset();
    mContext.reset(mEngine->createExecutionContext());
    if (!mContext)
    {
        LOG_ERROR("ActionContextRunner: failed to recreate TensorRT execution context");
        return false;
    }
    if (!mContext->setOptimizationProfileAsync(0, stream))
    {
        LOG_ERROR("ActionContextRunner: failed to set optimization profile after reset");
        return false;
    }
    return bindTensors();
}

bool ActionContextRunner::infer(cudaStream_t stream) noexcept
{
    if (!bindTensors())
    {
        return false;
    }
    return mContext->enqueueV3(stream);
}

} // namespace rt
} // namespace trt_edgellm
