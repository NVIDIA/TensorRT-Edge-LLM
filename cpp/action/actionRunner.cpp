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

#include "action/actionRunner.h"

#include "action/actionUtils.h"
#include "common/bindingNames.h"
#include "common/checkMacros.h"
#include "common/logger.h"
#include "common/mmapReader.h"
#include "kernels/posEncoding/initializeCosSinCache.h"

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <random>
#include <stdexcept>
#include <vector>

using namespace nvinfer1;
using Json = nlohmann::json;

namespace trt_edgellm
{
namespace rt
{
namespace
{

rt::Coords coordsFromJson(Json const& shape)
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
    if (dtype == "torch.int32" || dtype == "int32")
    {
        return nvinfer1::DataType::kINT32;
    }
    if (dtype == "torch.int64" || dtype == "int64")
    {
        return nvinfer1::DataType::kINT64;
    }
    if (dtype == "torch.bool" || dtype == "bool")
    {
        return nvinfer1::DataType::kBOOL;
    }
    if (dtype == "torch.uint8" || dtype == "uint8")
    {
        return nvinfer1::DataType::kUINT8;
    }
    throw std::runtime_error("Unsupported action tensor dtype: " + dtype);
}

std::size_t tensorBytes(rt::Tensor const& tensor)
{
    return static_cast<std::size_t>(tensor.getShape().volume()) * rt::utils::getTypeSize(tensor.getDataType());
}

std::string resolveActionEnginePath(std::string const& engineDir, Json const& configJson)
{
    std::string const engineFile = configJson.value("engine_file", std::string{"action.engine"});
    std::filesystem::path const preferred = std::filesystem::path(engineDir) / engineFile;
    if (std::filesystem::exists(preferred))
    {
        return preferred.string();
    }

    for (char const* candidate : {"action.engine", "diffusion.engine"})
    {
        std::filesystem::path const path = std::filesystem::path(engineDir) / candidate;
        if (std::filesystem::exists(path))
        {
            return path.string();
        }
    }

    return preferred.string();
}

std::string resolveVelocityOutputName(Json const& configJson, nvinfer1::ICudaEngine const* engine)
{
    std::string preferred;
    if (configJson.contains("output_names") && !configJson.at("output_names").empty())
    {
        preferred = configJson.at("output_names").at(0).get<std::string>();
    }
    else if (configJson.contains("outputs") && !configJson.at("outputs").empty()
        && configJson.at("outputs").at(0).contains("name"))
    {
        preferred = configJson.at("outputs").at(0).at("name").get<std::string>();
    }
    else
    {
        preferred = "velocity";
    }

    for (int32_t i = 0; i < engine->getNbIOTensors(); ++i)
    {
        char const* name = engine->getIOTensorName(i);
        if (name != nullptr && preferred == name && engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kOUTPUT)
        {
            return preferred;
        }
    }

    for (int32_t i = 0; i < engine->getNbIOTensors(); ++i)
    {
        char const* name = engine->getIOTensorName(i);
        if (name != nullptr && engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kOUTPUT)
        {
            LOG_WARNING("ActionRunner: output name '%s' not found in engine; using '%s'", preferred.c_str(), name);
            return name;
        }
    }

    throw std::runtime_error("ActionRunner: failed to resolve velocity output tensor name");
}

bool bindTensor(nvinfer1::ICudaEngine const* engine, nvinfer1::IExecutionContext* context, std::string const& name,
    rt::Tensor& tensor)
{
    if (engine->getTensorIOMode(name.c_str()) == nvinfer1::TensorIOMode::kINPUT)
    {
        if (!context->setInputShape(name.c_str(), tensor.getShape().getTRTDims()))
        {
            LOG_ERROR("ActionRunner: failed to set input shape for %s", name.c_str());
            return false;
        }
    }

    if (!context->setTensorAddress(name.c_str(), tensor.rawPointer()))
    {
        LOG_ERROR("ActionRunner: failed to bind tensor %s", name.c_str());
        return false;
    }

    return true;
}

bool copyTensorToInput(rt::Tensor& dst, rt::Tensor const& src, cudaStream_t stream)
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
        LOG_ERROR("ActionRunner: copy shape mismatch, dst=%ld src=%ld", elements, src.getShape().volume());
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

    LOG_ERROR("ActionRunner: unsupported dtype conversion src=%d dst=%d", static_cast<int>(srcType),
        static_cast<int>(dstType));
    return false;
}

void fillNormal(rt::Tensor& tensor, std::mt19937& rng)
{
    std::normal_distribution<float> distribution(0.0F, 1.0F);
    auto const elements = tensor.getShape().volume();

    if (tensor.getDataType() == nvinfer1::DataType::kFLOAT)
    {
        auto* data = tensor.dataPointer<float>();
        for (int64_t i = 0; i < elements; ++i)
        {
            data[i] = distribution(rng);
        }
        return;
    }

    if (tensor.getDataType() == nvinfer1::DataType::kHALF)
    {
        auto* data = tensor.dataPointer<half>();
        for (int64_t i = 0; i < elements; ++i)
        {
            data[i] = __float2half(distribution(rng));
        }
        return;
    }

    throw std::runtime_error("ActionRunner: noise initialization only supports fp16/fp32 actions");
}

void updateActionTensor(rt::Tensor& actions, rt::Tensor const& predVelocity, float stepSize)
{
    auto const elements = actions.getShape().volume();
    check::check(actions.getDataType() == predVelocity.getDataType(), "ActionRunner dtype mismatch");
    check::check(actions.getShape().volume() == predVelocity.getShape().volume(), "ActionRunner shape mismatch");

    if (actions.getDataType() == nvinfer1::DataType::kFLOAT)
    {
        auto* actionData = actions.dataPointer<float>();
        auto const* predData = predVelocity.dataPointer<float>();
        for (int64_t i = 0; i < elements; ++i)
        {
            actionData[i] += stepSize * predData[i];
        }
        return;
    }

    if (actions.getDataType() == nvinfer1::DataType::kHALF)
    {
        auto* actionData = actions.dataPointer<half>();
        auto const* predData = predVelocity.dataPointer<half>();
        for (int64_t i = 0; i < elements; ++i)
        {
            auto const updated = __half2float(actionData[i]) + stepSize * __half2float(predData[i]);
            actionData[i] = __float2half(updated);
        }
        return;
    }

    throw std::runtime_error("ActionRunner: action update only supports fp16/fp32 actions");
}

bool fillHostTensorFromFloats(rt::Tensor& tensor, std::vector<float> const& values)
{
    auto const elements = tensor.getShape().volume();
    if (static_cast<std::size_t>(elements) != values.size())
    {
        LOG_ERROR("ActionRunner: expected %ld values but got %zu", elements, values.size());
        return false;
    }

    if (tensor.getDataType() == nvinfer1::DataType::kFLOAT)
    {
        auto* data = tensor.dataPointer<float>();
        for (int64_t i = 0; i < elements; ++i)
        {
            data[i] = values[static_cast<std::size_t>(i)];
        }
        return true;
    }

    if (tensor.getDataType() == nvinfer1::DataType::kHALF)
    {
        auto* data = tensor.dataPointer<half>();
        for (int64_t i = 0; i < elements; ++i)
        {
            data[i] = __float2half(values[static_cast<std::size_t>(i)]);
        }
        return true;
    }

    if (tensor.getDataType() == nvinfer1::DataType::kINT64)
    {
        auto* data = tensor.dataPointer<int64_t>();
        for (int64_t i = 0; i < elements; ++i)
        {
            data[i] = static_cast<int64_t>(values[static_cast<std::size_t>(i)]);
        }
        return true;
    }

    if (tensor.getDataType() == nvinfer1::DataType::kINT32)
    {
        auto* data = tensor.dataPointer<int32_t>();
        for (int64_t i = 0; i < elements; ++i)
        {
            data[i] = static_cast<int32_t>(values[static_cast<std::size_t>(i)]);
        }
        return true;
    }

    LOG_ERROR("ActionRunner: unsupported tensor dtype for host fill: %d", static_cast<int>(tensor.getDataType()));
    return false;
}

} // namespace

ActionRunner::ActionRunner(
    std::string const& engineDir, cudaStream_t stream, LinearKVCache::CacheConfig const& kvCacheConfig)
    : mStream(stream)
{
    LOG_DEBUG("Loading action runner from %s", engineDir.c_str());

    std::string const configPath = engineDir + "/config.json";
    {
        std::ifstream configFile(configPath);
        if (configFile)
        {
            try
            {
                configFile >> mConfigJson;
            }
            catch (Json::parse_error const& e)
            {
                LOG_WARNING("ActionRunner: failed to parse %s: %s", configPath.c_str(), e.what());
                mConfigJson = Json::object();
            }
        }
    }

    std::string const actionEnginePath = resolveActionEnginePath(engineDir, mConfigJson);

    mRuntime = std::unique_ptr<IRuntime>(createInferRuntime(gLogger));
    if (!mRuntime)
    {
        throw std::runtime_error("Failed to create TensorRT runtime");
    }

    auto mmapReader = std::make_unique<file_io::MmapReader>(actionEnginePath);
    if (mmapReader->getData() == nullptr)
    {
        throw std::runtime_error("Failed to read engine file: " + actionEnginePath);
    }

    mEngine
        = std::unique_ptr<ICudaEngine>(mRuntime->deserializeCudaEngine(mmapReader->getData(), mmapReader->getSize()));
    if (!mEngine)
    {
        throw std::runtime_error("Failed to deserialize engine from: " + actionEnginePath);
    }

    mContext = std::unique_ptr<IExecutionContext>(
        mEngine->createExecutionContext(ExecutionContextAllocationStrategy::kUSER_MANAGED));
    if (!mContext)
    {
        throw std::runtime_error("Failed to create execution context");
    }

    if (!mContext->setOptimizationProfileAsync(0, stream))
    {
        throw std::runtime_error("Failed to set optimization profile");
    }

    if (!parseModelConfig(configPath))
    {
        throw std::runtime_error("Failed to parse model config");
    }

    try
    {
        allocateTensors(kvCacheConfig);
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("ActionRunner tensor allocation failed: %s", e.what());
        throw;
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));
}

int64_t ActionRunner::getRequiredContextMemorySize() const
{
    return mEngine ? mEngine->getDeviceMemorySizeV2() : 0;
}

bool ActionRunner::setContextMemory(rt::Tensor& sharedContextMemory)
{
    int64_t const requiredSize = getRequiredContextMemorySize();
    if (sharedContextMemory.getMemoryCapacity() < requiredSize)
    {
        LOG_ERROR("Shared context memory (%lld bytes) is smaller than required (%lld bytes)",
            static_cast<long long>(sharedContextMemory.getMemoryCapacity()), static_cast<long long>(requiredSize));
        return false;
    }
    mExecContextMemory = sharedContextMemory.rawPointer();
    mExecContextMemoryCapacity = sharedContextMemory.getMemoryCapacity();
    mContext->setDeviceMemoryV2(mExecContextMemory, mExecContextMemoryCapacity);
    return true;
}

bool ActionRunner::resetExecutionContext(cudaStream_t stream)
{
    if (!mEngine || mExecContextMemory == nullptr || mExecContextMemoryCapacity <= 0)
    {
        LOG_ERROR("ActionRunner: cannot reset execution context before shared memory is configured");
        return false;
    }

    mContext.reset();
    mContext.reset(mEngine->createExecutionContext(ExecutionContextAllocationStrategy::kUSER_MANAGED));
    if (!mContext)
    {
        LOG_ERROR("ActionRunner: failed to recreate execution context");
        return false;
    }
    if (!mContext->setOptimizationProfileAsync(0, stream))
    {
        LOG_ERROR("ActionRunner: failed to set optimization profile after reset");
        return false;
    }
    mContext->setDeviceMemoryV2(mExecContextMemory, mExecContextMemoryCapacity);
    return true;
}

bool ActionRunner::engineHasTensor(std::string const& name) const noexcept
{
    if (!mEngine)
    {
        return false;
    }
    for (int32_t i = 0; i < mEngine->getNbIOTensors(); ++i)
    {
        char const* tensorName = mEngine->getIOTensorName(i);
        if (tensorName != nullptr && name == tensorName)
        {
            return true;
        }
    }
    return false;
}

bool ActionRunner::parseModelConfig(std::string const& configPath)
{
    if (mConfigJson.empty())
    {
        std::ifstream configFile(configPath);
        if (!configFile.is_open())
        {
            LOG_ERROR("Failed to open config file: %s", configPath.c_str());
            return false;
        }
        try
        {
            configFile >> mConfigJson;
        }
        catch (Json::parse_error const& e)
        {
            LOG_ERROR("Failed to parse config file %s: %s", configPath.c_str(), e.what());
            return false;
        }
    }

    std::string const modelTypeStr = mConfigJson.value("model_type", std::string{});
    if (modelTypeStr == "alpamayo1" || modelTypeStr == "ALPAMAYO1")
    {
        mModelType = action::ActionModelType::ALPAMAYO1;
    }

    bool const hasPrefixKV
        = engineHasTensor(binding_names::kNoiseTrajectory) && engineHasTensor(binding_names::formatKCacheName(0, true));
    bool const hasVelocityInputs = mConfigJson.contains("input_names") && mConfigJson.contains("inputs");

    if (hasPrefixKV || mModelType == action::ActionModelType::ALPAMAYO1)
    {
        mContextHandoff = ActionContextHandoff::PREFIX_KV;
        mRolloutMode = ActionRolloutMode::FLOW_MATCHING;
        mModelType = action::ActionModelType::ALPAMAYO1;
        mUsesMRope = engineHasTensor(binding_names::kRopeCosSin);

        try
        {
            mConfig.ropeTheta = mConfigJson.at("rope_theta").get<float>();
            mConfig.numDecoderLayers = mConfigJson.at("num_hidden_layers").get<int32_t>();
            mConfig.numTrajTokens = mConfigJson.at("num_traj_tokens").get<int32_t>();
            mConfig.trajTokenStart = mConfigJson.at("traj_token_start").get<int32_t>();
        }
        catch (Json::exception const& e)
        {
            LOG_ERROR("Failed to read Alpamayo fields from %s: %s", configPath.c_str(), e.what());
            return false;
        }

        auto const& ropeParams = mConfigJson.contains("rope_scaling") ? mConfigJson["rope_scaling"] : mConfigJson;
        if (ropeParams.contains("mrope_section"))
        {
            auto section = ropeParams["mrope_section"].get<std::vector<int32_t>>();
            if (section.size() >= 3)
            {
                mConfig.mropeSectionH = section[1];
                mConfig.mropeSectionW = section[2];
            }
        }
        if (mUsesMRope && (mConfig.mropeSectionH <= 0 || mConfig.mropeSectionW <= 0))
        {
            LOG_ERROR("ActionRunner: mrope_section required when rope_cos_sin binding is present");
            return false;
        }

        if (mConfigJson.contains("builder_config") && mConfigJson["builder_config"].contains("max_kv_cache_capacity"))
        {
            mConfig.maxKVCacheCapacity = mConfigJson["builder_config"]["max_kv_cache_capacity"].get<int32_t>();
        }
        else
        {
            LOG_ERROR("max_kv_cache_capacity not found in builder_config section of %s", configPath.c_str());
            return false;
        }

        mConfig.numInferenceTimesteps = mConfigJson.value("num_inference_timesteps", kDefaultDenoiseSteps);
        if (mConfig.numInferenceTimesteps <= 0)
        {
            mConfig.numInferenceTimesteps = kDefaultDenoiseSteps;
        }

        Dims const maxNoise = mEngine->getProfileShape(binding_names::kNoiseTrajectory, 0, OptProfileSelector::kMAX);
        mConfig.actionHorizon = static_cast<int32_t>(maxNoise.d[1]);
        mConfig.actionDim = static_cast<int32_t>(maxNoise.d[2]);
        return true;
    }

    if (hasVelocityInputs)
    {
        mContextHandoff = ActionContextHandoff::CONTEXT_TENSOR;
        mRolloutMode = ActionRolloutMode::VELOCITY;

        mConfig.numInferenceTimesteps = mConfigJson.value("num_inference_timesteps", 0);
        if (mConfig.numInferenceTimesteps <= 0)
        {
            mConfig.numInferenceTimesteps = mConfigJson.value("num_inference_steps", 1);
        }
        mNumTimestepBuckets = mConfigJson.value("num_timestep_buckets", 1);
        mRolloutDtSign = mConfigJson.value("rollout_dt_sign", 1);

        mNoiseInputName = mConfigJson.value("noise_input_name", std::string{});
        if (mNoiseInputName.empty())
        {
            if (engineHasTensor("actions"))
            {
                mNoiseInputName = "actions";
            }
            else if (engineHasTensor("x_t"))
            {
                mNoiseInputName = "x_t";
            }
            else if (engineHasTensor(binding_names::kNoiseTrajectory))
            {
                mNoiseInputName = binding_names::kNoiseTrajectory;
            }
            else
            {
                LOG_ERROR("ActionRunner: could not resolve noise input name");
                return false;
            }
        }

        mTimestepSchedule = mConfigJson.value("timestep_schedule", std::string{"discrete_buckets"});
        if (mTimestepSchedule == "groot_buckets")
        {
            mTimestepSchedule = "discrete_buckets";
        }
        else if (mTimestepSchedule == "pi05_flow")
        {
            mTimestepSchedule = "continuous_flow";
        }

        mInputNames.clear();
        for (auto const& nameJson : mConfigJson.at("input_names"))
        {
            auto const name = nameJson.get<std::string>();
            if (!mConfigJson.at("inputs").contains(name))
            {
                LOG_ERROR("ActionRunner: missing input metadata for %s", name.c_str());
                return false;
            }
            mInputNames.push_back(name);
        }

        if (!hasInputTensor(mNoiseInputName) || !hasInputTensor(mTimestepName))
        {
            LOG_ERROR("ActionRunner: velocity config must define '%s' and '%s'", mNoiseInputName.c_str(),
                mTimestepName.c_str());
            return false;
        }

        buildEngineBindingNames();

        mLmToActionSlots.clear();
        mLmWiredInputNames.clear();
        if (mConfigJson.contains("lm_to_action_slots"))
        {
            for (auto const& pairJson : mConfigJson.at("lm_to_action_slots"))
            {
                if (!pairJson.is_array() || pairJson.size() != 2)
                {
                    LOG_ERROR("ActionRunner: lm_to_action_slots entries must be [lm_idx, action_idx]");
                    return false;
                }
                auto const lmSlot = pairJson.at(0).get<int32_t>();
                auto const actionSlot = pairJson.at(1).get<int32_t>();
                mLmToActionSlots.emplace_back(lmSlot, actionSlot);
                mLmWiredInputNames.push_back(mInputNames[static_cast<std::size_t>(actionSlot)]);
            }
        }

        auto const actionShape = coordsFromJson(mConfigJson.at("inputs").at(mNoiseInputName).at("shape"));
        if (actionShape.getNumDims() > 0)
        {
            mMaxActionBatchSize = static_cast<int32_t>(actionShape[0]);
            if (actionShape.getNumDims() > 1)
            {
                mConfig.actionHorizon = static_cast<int32_t>(actionShape[1]);
            }
            if (actionShape.getNumDims() > 2)
            {
                mConfig.actionDim = static_cast<int32_t>(actionShape[2]);
            }
        }

        mPredVelocityName = resolveVelocityOutputName(mConfigJson, mEngine.get());
        return true;
    }

    LOG_ERROR("ActionRunner: unrecognized action engine layout in %s", configPath.c_str());
    return false;
}

void ActionRunner::allocateTensors(LinearKVCache::CacheConfig const& kvCacheConfig)
{
    if (mContextHandoff == ActionContextHandoff::PREFIX_KV)
    {
        allocatePrefixKVTensors(kvCacheConfig);
        return;
    }
    if (mContextHandoff == ActionContextHandoff::CONTEXT_TENSOR)
    {
        allocateVelocityTensors();
        return;
    }
    throw std::runtime_error("ActionRunner: cannot allocate tensors for unknown context handoff");
}

void ActionRunner::allocatePrefixKVTensors(LinearKVCache::CacheConfig const& kvCacheConfig)
{
    Dims const maxNoise = mEngine->getProfileShape(binding_names::kNoiseTrajectory, 0, OptProfileSelector::kMAX);
    int32_t const maxBatch = static_cast<int32_t>(maxNoise.d[0]);
    mMaxActionBatchSize = maxBatch;

    mNumKVHeads = static_cast<int32_t>(kvCacheConfig.numKVHeads);
    mMaxSequenceLength = static_cast<int32_t>(kvCacheConfig.maxSequenceLength);
    mKvHeadDim = static_cast<int32_t>(kvCacheConfig.headDim);

    nvinfer1::Dims noiseShape = mContext->getTensorShape(binding_names::kNoiseTrajectory);
    mConfig.actionHorizon = static_cast<int32_t>(noiseShape.d[1]);
    mConfig.actionDim = static_cast<int32_t>(noiseShape.d[2]);

    rt::Coords const noiseCoords({maxBatch, mConfig.actionHorizon, mConfig.actionDim});

    mNoiseDevice
        = rt::Tensor(noiseCoords, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT, "ActionRunner::mNoiseDevice");
    mNoiseHost = rt::Tensor(noiseCoords, rt::DeviceType::kCPU, nvinfer1::DataType::kFLOAT, "ActionRunner::mNoiseHost");
    mDenoisedDevice
        = rt::Tensor(noiseCoords, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT, "ActionRunner::mDenoisedDevice");
    mDenoisedHost
        = rt::Tensor(noiseCoords, rt::DeviceType::kCPU, nvinfer1::DataType::kFLOAT, "ActionRunner::mDenoisedHost");

    mTimeStepsT0Device
        = rt::Tensor({1}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT, "ActionRunner::mTimeStepsT0Device");
    mTimeStepsT1Device
        = rt::Tensor({1}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT, "ActionRunner::mTimeStepsT1Device");
    mTimeStepsT0Host = rt::Tensor(std::vector<int64_t>{mConfig.numInferenceTimesteps}, rt::DeviceType::kCPU,
        nvinfer1::DataType::kFLOAT, "ActionRunner::mTimeStepsT0Host");
    mTimeStepsT1Host = rt::Tensor(std::vector<int64_t>{mConfig.numInferenceTimesteps}, rt::DeviceType::kCPU,
        nvinfer1::DataType::kFLOAT, "ActionRunner::mTimeStepsT1Host");

    if (mUsesMRope)
    {
        nvinfer1::Dims ropeCosSinDims = mContext->getTensorShape(binding_names::kRopeCosSin);
        mRopeHeadDim = ropeCosSinDims.d[2];
        mRopeCosSinDevice = rt::Tensor({maxBatch, mConfig.actionHorizon, mRopeHeadDim}, rt::DeviceType::kGPU,
            nvinfer1::DataType::kFLOAT, "ActionRunner::mRopeCosSinDevice");
        mRopePositionIdsHost = rt::Tensor({maxBatch, 3, mConfig.actionHorizon}, rt::DeviceType::kCPU,
            nvinfer1::DataType::kINT64, "ActionRunner::mRopePositionIdsHost");
        mRopePositionIdsDevice = rt::Tensor({maxBatch, 3, mConfig.actionHorizon}, rt::DeviceType::kGPU,
            nvinfer1::DataType::kINT64, "ActionRunner::mRopePositionIdsDevice");
    }

    mPositionIdsHost = rt::Tensor({maxBatch, mConfig.actionHorizon}, rt::DeviceType::kCPU, nvinfer1::DataType::kINT32,
        "ActionRunner::mPositionIdsHost");
    mPositionIdsDevice = rt::Tensor({maxBatch, mConfig.actionHorizon}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32,
        "ActionRunner::mPositionIdsDevice");

    mKvcacheActualLengthsHost = rt::Tensor(std::vector<int64_t>{maxBatch}, rt::DeviceType::kCPU,
        nvinfer1::DataType::kINT32, "ActionRunner::mKvcacheActualLengthsHost");
    mKvcacheActualLengthsBroadcastDevice = rt::Tensor(std::vector<int64_t>{maxBatch}, rt::DeviceType::kGPU,
        nvinfer1::DataType::kINT32, "ActionRunner::mKvcacheActualLengthsBroadcastDevice");

    rt::Coords const kvcacheShape{
        maxBatch, kvCacheConfig.numKVHeads, kvCacheConfig.maxSequenceLength, kvCacheConfig.headDim};
    mKCacheLayers.resize(static_cast<size_t>(mConfig.numDecoderLayers));
    mVCacheLayers.resize(static_cast<size_t>(mConfig.numDecoderLayers));
    for (int32_t i = 0; i < mConfig.numDecoderLayers; ++i)
    {
        mKCacheLayers[i] = rt::Tensor(
            kvcacheShape, rt::DeviceType::kGPU, kvCacheConfig.kvCacheTypeTRT, "ActionRunner::mKCacheLayer");
        mVCacheLayers[i] = rt::Tensor(
            kvcacheShape, rt::DeviceType::kGPU, kvCacheConfig.kvCacheTypeTRT, "ActionRunner::mVCacheLayer");
    }
}

void ActionRunner::allocateVelocityTensors()
{
    mInputTensors.clear();
    mInputTensors.reserve(mInputNames.size());
    for (auto const& name : mInputNames)
    {
        mInputTensors.emplace_back(coordsFromJson(mConfigJson.at("inputs").at(name).at("shape")), rt::DeviceType::kGPU,
            dataTypeFromTorchString(mConfigJson.at("inputs").at(name).at("dtype").get<std::string>()), name);
    }

    mPredVelocity = rt::Tensor(coordsFromJson(mConfigJson.at("outputs").at(0).at("shape")), rt::DeviceType::kGPU,
        dataTypeFromTorchString(mConfigJson.at("outputs").at(0).at("dtype").get<std::string>()), mPredVelocityName);

    mNoiseHost = rt::Tensor(
        getActions().getShape(), rt::DeviceType::kCPU, getActions().getDataType(), "ActionRunner::mNoiseHost");
    mPredVelocityHost = rt::Tensor(
        mPredVelocity.getShape(), rt::DeviceType::kCPU, mPredVelocity.getDataType(), "ActionRunner::mPredVelocityHost");

    auto const& timestepMeta = mConfigJson.at("inputs").at(mTimestepName);
    mTimestepHost = rt::Tensor(coordsFromJson(timestepMeta.at("shape")), rt::DeviceType::kCPU,
        dataTypeFromTorchString(timestepMeta.at("dtype").get<std::string>()), "ActionRunner::mTimestepHost");

    if (!bindVelocityTensors())
    {
        throw std::runtime_error("ActionRunner: failed to bind velocity tensors during allocation");
    }
}

bool ActionRunner::reshapeActionTensorsForActiveBatch(int32_t activeBatchSize)
{
    if (mContextHandoff == ActionContextHandoff::PREFIX_KV)
    {
        rt::Coords const noiseShape({activeBatchSize, mConfig.actionHorizon, mConfig.actionDim});
        bool ok = true;
        ok &= mNoiseDevice.reshape(noiseShape);
        ok &= mNoiseHost.reshape(noiseShape);
        ok &= mDenoisedDevice.reshape(noiseShape);
        ok &= mDenoisedHost.reshape(noiseShape);
        if (mUsesMRope)
        {
            ok &= mRopeCosSinDevice.reshape({activeBatchSize, mConfig.actionHorizon, mRopeHeadDim});
            ok &= mRopePositionIdsHost.reshape({activeBatchSize, 3, mConfig.actionHorizon});
            ok &= mRopePositionIdsDevice.reshape({activeBatchSize, 3, mConfig.actionHorizon});
        }
        ok &= mPositionIdsHost.reshape({activeBatchSize, mConfig.actionHorizon});
        ok &= mPositionIdsDevice.reshape({activeBatchSize, mConfig.actionHorizon});
        ok &= mKvcacheActualLengthsHost.reshape({activeBatchSize});
        if (!ok)
        {
            LOG_ERROR("ActionRunner: tensor reshape failed for activeBatchSize=%d", activeBatchSize);
        }
        return ok;
    }

    if (mContextHandoff == ActionContextHandoff::CONTEXT_TENSOR)
    {
        bool ok = true;
        ok &= mNoiseHost.reshape({activeBatchSize, mConfig.actionHorizon, mConfig.actionDim});
        for (std::size_t inputIdx = 0; inputIdx < mInputTensors.size(); ++inputIdx)
        {
            auto const& inputName = mInputNames[inputIdx];
            if (isRolloutManagedInput(inputName))
            {
                continue;
            }
            auto& tensor = mInputTensors[inputIdx];
            auto shape = tensor.getShape();
            if (shape.getNumDims() > 0)
            {
                shape[0] = activeBatchSize;
                ok &= tensor.reshape(shape);
            }
        }
        auto predShape = mPredVelocity.getShape();
        if (predShape.getNumDims() > 0)
        {
            predShape[0] = activeBatchSize;
            ok &= mPredVelocity.reshape(predShape);
            ok &= mPredVelocityHost.reshape(predShape);
        }
        if (!ok)
        {
            LOG_ERROR("ActionRunner: velocity tensor reshape failed for activeBatchSize=%d", activeBatchSize);
        }
        return ok;
    }

    return false;
}

void ActionRunner::initializeNoiseTrajectory(int32_t randomSeed, int32_t activeBatchSize)
{
    if (mContextHandoff == ActionContextHandoff::PREFIX_KV)
    {
        size_t const elemCount = static_cast<size_t>(activeBatchSize) * static_cast<size_t>(mConfig.actionHorizon)
            * static_cast<size_t>(mConfig.actionDim);
        float* data = static_cast<float*>(mNoiseHost.rawPointer());
        std::mt19937 generator(static_cast<std::mt19937::result_type>(randomSeed));
        std::normal_distribution<float> dist(0.0f, 1.0f);
        for (size_t i = 0; i < elemCount; ++i)
        {
            data[i] = dist(generator);
        }
        return;
    }

    if (mContextHandoff == ActionContextHandoff::CONTEXT_TENSOR)
    {
        mRng.seed(static_cast<uint32_t>(randomSeed));
        fillNormal(mNoiseHost, mRng);
    }
}

bool ActionRunner::preprocess(LLMGenerationRequest const& request, std::vector<std::vector<int32_t>>& batchedInputIds,
    tokenizer::Tokenizer const* tokenizer)
{
    int32_t const numUniqueRequests = static_cast<int32_t>(request.requests.size());
    int32_t const actionBatchSize = (request.actionBatchSize > 0) ? request.actionBatchSize : numUniqueRequests;

    if (mContextHandoff == ActionContextHandoff::PREFIX_KV && mConfig.numTrajTokens > 0 && tokenizer != nullptr)
    {
        tokenizer::TokenToRanks const& specialTokens = tokenizer->getSpecialTokensEncoder();
        for (int32_t i = 0; i < numUniqueRequests; ++i)
        {
            std::vector<int32_t>& tokenIds = batchedInputIds[i];
            LLMGenerationRequest::Request const& req = request.requests[i];
            if (!req.pastTrajectory)
            {
                continue;
            }

            auto const itStart = specialTokens.find(kTrajHistoryStartStr);
            auto const itPad = specialTokens.find(kTrajHistoryPadStr);
            auto const itEnd = specialTokens.find(kTrajHistoryEndStr);
            if (itStart == specialTokens.end() || itPad == specialTokens.end() || itEnd == specialTokens.end())
            {
                continue;
            }

            tokenizer::Rank const startId = itStart->second;
            tokenizer::Rank const padId = itPad->second;
            tokenizer::Rank const endId = itEnd->second;

            std::vector<tokenizer::Rank> const actualTokens = action_utils::trajectoryToTokenIds(
                *req.pastTrajectory, mConfig.numTrajTokens, mConfig.trajTokenStart);

            size_t scanIdx = 0;
            while (scanIdx < tokenIds.size())
            {
                if (tokenIds[scanIdx] != startId)
                {
                    ++scanIdx;
                    continue;
                }
                size_t endMarkerIdx = scanIdx + 1;
                while (endMarkerIdx < tokenIds.size() && tokenIds[endMarkerIdx] == padId)
                {
                    ++endMarkerIdx;
                }
                if (endMarkerIdx >= tokenIds.size() || tokenIds[endMarkerIdx] != endId)
                {
                    ++scanIdx;
                    continue;
                }

                size_t const numPads = endMarkerIdx - scanIdx - 1;
                if (numPads != actualTokens.size())
                {
                    LOG_ERROR("Trajectory placeholder token count (%zu) does not match encoding length (%zu).", numPads,
                        actualTokens.size());
                    return false;
                }

                for (size_t k = 0; k < actualTokens.size(); ++k)
                {
                    tokenIds[scanIdx + 1 + k] = actualTokens[k];
                }
                scanIdx += 2 + actualTokens.size();
            }
        }
    }

    if (actionBatchSize > mMaxActionBatchSize)
    {
        LOG_ERROR(
            "Requested action batch size %d exceeds engine max batch size %d", actionBatchSize, mMaxActionBatchSize);
        return false;
    }

    mActiveActionBatchSize = actionBatchSize;
    if (mContextHandoff == ActionContextHandoff::CONTEXT_TENSOR && !reshapeActionTensorsForActiveBatch(actionBatchSize))
    {
        LOG_ERROR("ActionRunner: failed to reshape tensors for action batch size %d", actionBatchSize);
        return false;
    }

    try
    {
        initializeNoiseTrajectory(mNoiseSeed, actionBatchSize);
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("ActionRunner noise initialization failed: %s", e.what());
        return false;
    }

    return true;
}

std::pair<rt::Tensor&, rt::Tensor&> ActionRunner::getSeparateKVCacheForDecoderLayer(
    cudaStream_t stream, LinearKVCache& kvcache, int32_t decoderLayerIdx, int32_t activeBatchSize)
{
    LinearKVCache::CacheConfig const& config = kvcache.getConfig();
    int64_t const blockElems = config.numKVHeads * config.maxSequenceLength * config.headDim;
    size_t const elemSize = rt::utils::getTypeSize(config.kvCacheTypeTRT);
    size_t const blockBytes = static_cast<size_t>(blockElems) * elemSize;
    size_t const combinedBatchStride = 2ULL * blockBytes;

    rt::Tensor combined = kvcache.getCombinedKVCacheForDecoderLayer(decoderLayerIdx);
    char const* src = static_cast<char const*>(combined.rawPointer());
    char* dstK = static_cast<char*>(mKCacheLayers[decoderLayerIdx].rawPointer());
    char* dstV = static_cast<char*>(mVCacheLayers[decoderLayerIdx].rawPointer());

    int32_t const llmBatch = static_cast<int32_t>(config.maxBatchSize);
    for (int32_t b = 0; b < activeBatchSize; ++b)
    {
        int32_t const srcSlot = b % llmBatch;
        CUDA_CHECK(cudaMemcpyAsync(dstK + static_cast<size_t>(b) * blockBytes,
            src + static_cast<size_t>(srcSlot) * combinedBatchStride, blockBytes, cudaMemcpyDeviceToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(dstV + static_cast<size_t>(b) * blockBytes,
            src + static_cast<size_t>(srcSlot) * combinedBatchStride + blockBytes, blockBytes, cudaMemcpyDeviceToDevice,
            stream));
    }
    return {mKCacheLayers[decoderLayerIdx], mVCacheLayers[decoderLayerIdx]};
}

void ActionRunner::setDynamicInputShapes(int32_t activeBatchSize)
{
    if (mContextHandoff != ActionContextHandoff::PREFIX_KV)
    {
        return;
    }

    Dims const kvCacheStartIndexShape = {1, {activeBatchSize}};
    Dims const noiseShape = {3, {activeBatchSize, mConfig.actionHorizon, mConfig.actionDim}};
    Dims const ropeIdxShape = {2, {activeBatchSize, mConfig.actionHorizon}};
    Dims const kvShape = {
        4, {activeBatchSize, static_cast<int64_t>(mNumKVHeads), static_cast<int64_t>(mMaxSequenceLength), mKvHeadDim}};

    bool status = true;
    status &= mContext->setInputShape(binding_names::kKVCacheStartIndex, kvCacheStartIndexShape);
    status &= mContext->setInputShape(binding_names::kNoiseTrajectory, noiseShape);
    if (mUsesMRope)
    {
        Dims const ropeCosShape = {3, {activeBatchSize, mConfig.actionHorizon, mRopeHeadDim}};
        status &= mContext->setInputShape(binding_names::kRopeCosSin, ropeCosShape);
    }
    status &= mContext->setInputShape(binding_names::kAttentionPosId, ropeIdxShape);
    for (int32_t i = 0; i < mConfig.numDecoderLayers; ++i)
    {
        status &= mContext->setInputShape(binding_names::formatKCacheName(i, true).c_str(), kvShape);
        status &= mContext->setInputShape(binding_names::formatVCacheName(i, true).c_str(), kvShape);
    }
    if (!status)
    {
        LOG_ERROR("ActionRunner: setDynamicInputShapes failed for activeBatchSize=%d", activeBatchSize);
        throw std::runtime_error("setDynamicInputShapes failed");
    }
}

int32_t const* ActionRunner::getActualKVLengths(cudaStream_t stream, int32_t activeBatchSize)
{
    CUDA_CHECK(cudaMemcpyAsync(mKvcacheActualLengthsHost.rawPointer(), mKvcacheActualLengthsDevice,
        static_cast<size_t>(activeBatchSize) * sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return static_cast<int32_t const*>(mKvcacheActualLengthsHost.rawPointer());
}

std::vector<std::vector<FutureTrajectoryPoint>> ActionRunner::sampleTrajectory(cudaStream_t stream,
    int32_t activeBatchSize, LinearKVCache& kvcache, std::vector<int64_t> const& vlmOutputsRopeDeltas)
{
    std::vector<std::vector<FutureTrajectoryPoint>> result;
    if (mContextHandoff != ActionContextHandoff::PREFIX_KV)
    {
        LOG_ERROR("ActionRunner::sampleTrajectory requires PREFIX_KV context handoff");
        return result;
    }

    if (mUsesMRope && static_cast<int32_t>(vlmOutputsRopeDeltas.size()) != activeBatchSize)
    {
        LOG_ERROR("vlmOutputsRopeDeltas size %zu != activeBatchSize %d", vlmOutputsRopeDeltas.size(), activeBatchSize);
        return result;
    }

    if (!reshapeActionTensorsForActiveBatch(activeBatchSize))
    {
        LOG_ERROR("reshapeActionTensorsForActiveBatch failed for activeBatchSize=%d", activeBatchSize);
        return result;
    }

    int32_t const llmBatch = static_cast<int32_t>(kvcache.getConfig().maxBatchSize);
    auto* llmLengthsDevice = static_cast<int32_t*>(kvcache.getKVCacheLengths().rawPointer());
    if (llmBatch < activeBatchSize)
    {
        auto* broadcastLengths = static_cast<int32_t*>(mKvcacheActualLengthsBroadcastDevice.rawPointer());
        for (int32_t b = 0; b < activeBatchSize; ++b)
        {
            int32_t const srcSlot = b % llmBatch;
            CUDA_CHECK(cudaMemcpyAsync(
                broadcastLengths + b, llmLengthsDevice + srcSlot, sizeof(int32_t), cudaMemcpyDeviceToDevice, stream));
        }
        mKvcacheActualLengthsDevice = broadcastLengths;
    }
    else
    {
        mKvcacheActualLengthsDevice = llmLengthsDevice;
    }

    setDynamicInputShapes(activeBatchSize);

    bool setEngineIOStatus{true};
    setEngineIOStatus &= mContext->setTensorAddress(binding_names::kKVCacheStartIndex, mKvcacheActualLengthsDevice);
    setEngineIOStatus &= mContext->setTensorAddress(binding_names::kNoiseTrajectory, mNoiseDevice.rawPointer());
    setEngineIOStatus &= mContext->setTensorAddress(binding_names::kTimeStepsT0, mTimeStepsT0Device.rawPointer());
    setEngineIOStatus &= mContext->setTensorAddress(binding_names::kTimeStepsT1, mTimeStepsT1Device.rawPointer());
    setEngineIOStatus &= mContext->setTensorAddress(binding_names::kDenoisedTrajectory, mDenoisedDevice.rawPointer());
    if (mUsesMRope)
    {
        setEngineIOStatus &= mContext->setTensorAddress(binding_names::kRopeCosSin, mRopeCosSinDevice.rawPointer());
    }
    setEngineIOStatus &= mContext->setTensorAddress(binding_names::kAttentionPosId, mPositionIdsDevice.rawPointer());

    for (int32_t i = 0; i < mConfig.numDecoderLayers; ++i)
    {
        auto [kCacheBlock, vCacheBlock] = getSeparateKVCacheForDecoderLayer(stream, kvcache, i, activeBatchSize);
        setEngineIOStatus
            &= mContext->setTensorAddress(binding_names::formatKCacheName(i, true).c_str(), kCacheBlock.rawPointer());
        setEngineIOStatus
            &= mContext->setTensorAddress(binding_names::formatVCacheName(i, true).c_str(), vCacheBlock.rawPointer());
        setEngineIOStatus
            &= mContext->setTensorAddress(binding_names::formatKCacheName(i, false).c_str(), kCacheBlock.rawPointer());
        setEngineIOStatus
            &= mContext->setTensorAddress(binding_names::formatVCacheName(i, false).c_str(), vCacheBlock.rawPointer());
    }

    if (!setEngineIOStatus)
    {
        LOG_ERROR("ActionRunner: failed to bind prefix-KV action engine tensors");
        return result;
    }

    for (int32_t i = 0; i < mConfig.numInferenceTimesteps; ++i)
    {
        static_cast<float*>(mTimeStepsT0Host.rawPointer())[i]
            = static_cast<float>(i) / static_cast<float>(mConfig.numInferenceTimesteps);
        static_cast<float*>(mTimeStepsT1Host.rawPointer())[i]
            = static_cast<float>(i + 1) / static_cast<float>(mConfig.numInferenceTimesteps);
    }

    int32_t const* lengthsHost = getActualKVLengths(stream, activeBatchSize);

    int32_t* positionIdsPtr = static_cast<int32_t*>(mPositionIdsHost.rawPointer());
    for (int32_t b = 0; b < activeBatchSize; ++b)
    {
        for (int32_t w = 0; w < mConfig.actionHorizon; ++w)
        {
            positionIdsPtr[b * mConfig.actionHorizon + w] = w;
        }
    }

    if (mUsesMRope)
    {
        int64_t* mropePosPtr = static_cast<int64_t*>(mRopePositionIdsHost.rawPointer());
        for (int32_t b = 0; b < activeBatchSize; ++b)
        {
            int64_t const basePos = vlmOutputsRopeDeltas[b] + static_cast<int64_t>(lengthsHost[b]);
            for (int32_t dim = 0; dim < 3; ++dim)
            {
                for (int32_t w = 0; w < mConfig.actionHorizon; ++w)
                {
                    mropePosPtr[b * mConfig.actionHorizon * 3 + dim * mConfig.actionHorizon + w]
                        = static_cast<int64_t>(w) + basePos;
                }
            }
        }

        CUDA_CHECK(cudaMemcpyAsync(mRopePositionIdsDevice.rawPointer(), mRopePositionIdsHost.rawPointer(),
            static_cast<size_t>(activeBatchSize) * 3ULL * static_cast<size_t>(mConfig.actionHorizon) * sizeof(int64_t),
            cudaMemcpyHostToDevice, stream));

        kernel::initializeMRopeCosSin(static_cast<float*>(mRopeCosSinDevice.rawPointer()),
            static_cast<int64_t*>(mRopePositionIdsDevice.rawPointer()), mConfig.ropeTheta,
            static_cast<int64_t>(mRopeHeadDim), mConfig.actionHorizon, activeBatchSize, true, mConfig.mropeSectionH,
            mConfig.mropeSectionW, stream);
    }

    CUDA_CHECK(cudaMemcpyAsync(mNoiseDevice.rawPointer(), mNoiseHost.rawPointer(),
        static_cast<size_t>(activeBatchSize) * static_cast<size_t>(mConfig.actionHorizon)
            * static_cast<size_t>(mConfig.actionDim) * sizeof(float),
        cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(mPositionIdsDevice.rawPointer(), mPositionIdsHost.rawPointer(),
        static_cast<size_t>(activeBatchSize) * static_cast<size_t>(mConfig.actionHorizon) * sizeof(int32_t),
        cudaMemcpyHostToDevice, stream));

    for (int32_t i = 0; i < mConfig.numInferenceTimesteps; ++i)
    {
        float* t0Ptr = static_cast<float*>(mTimeStepsT0Host.rawPointer()) + i;
        float* t1Ptr = static_cast<float*>(mTimeStepsT1Host.rawPointer()) + i;

        CUDA_CHECK(
            cudaMemcpyAsync(mTimeStepsT0Device.rawPointer(), t0Ptr, sizeof(float), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(
            cudaMemcpyAsync(mTimeStepsT1Device.rawPointer(), t1Ptr, sizeof(float), cudaMemcpyHostToDevice, stream));

        if (!mContext->enqueueV3(stream))
        {
            LOG_ERROR("ActionRunner: enqueueV3 failed at denoise step %d", i);
            return result;
        }

        CUDA_CHECK(cudaMemcpyAsync(mNoiseDevice.rawPointer(), mDenoisedDevice.rawPointer(),
            static_cast<size_t>(activeBatchSize) * static_cast<size_t>(mConfig.actionHorizon)
                * static_cast<size_t>(mConfig.actionDim) * sizeof(float),
            cudaMemcpyDeviceToDevice, stream));
    }

    CUDA_CHECK(cudaMemcpyAsync(mDenoisedHost.rawPointer(), mDenoisedDevice.rawPointer(),
        static_cast<size_t>(activeBatchSize) * static_cast<size_t>(mConfig.actionHorizon)
            * static_cast<size_t>(mConfig.actionDim) * sizeof(float),
        cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    result.resize(activeBatchSize);
    float* output = static_cast<float*>(mDenoisedHost.rawPointer());
    for (int32_t b = 0; b < activeBatchSize; ++b)
    {
        result[b].reserve(mConfig.actionHorizon);
        float const* row = output + b * (mConfig.actionHorizon * mConfig.actionDim);
        for (int32_t w = 0; w < mConfig.actionHorizon; ++w)
        {
            result[b].emplace_back(row[w * mConfig.actionDim], row[w * mConfig.actionDim + 1]);
        }
    }
    return result;
}

std::size_t ActionRunner::velocityInputIndex(std::string const& name) const
{
    for (std::size_t i = 0; i < mInputNames.size(); ++i)
    {
        if (mInputNames[i] == name)
        {
            return i;
        }
    }
    throw std::runtime_error("ActionRunner: unknown input tensor: " + name);
}

bool ActionRunner::hasInputTensor(std::string const& name) const noexcept
{
    for (auto const& inputName : mInputNames)
    {
        if (inputName == name)
        {
            return true;
        }
    }
    return false;
}

rt::Tensor& ActionRunner::getActions()
{
    return mInputTensors.at(velocityInputIndex(mNoiseInputName));
}

rt::Tensor const& ActionRunner::getActions() const
{
    return mInputTensors.at(velocityInputIndex(mNoiseInputName));
}

bool ActionRunner::copyInputFrom(std::string const& name, rt::Tensor const& src, cudaStream_t stream)
{
    if (!hasInputTensor(name))
    {
        LOG_ERROR("ActionRunner: unknown input tensor %s", name.c_str());
        return false;
    }
    auto& dst = mInputTensors.at(velocityInputIndex(name));
    return copyTensorToInput(dst, src, stream);
}

bool ActionRunner::wireLanguageOutputs(std::vector<rt::Tensor const*> const& languageOutputs, cudaStream_t stream)
{
    if (!mLmToActionSlots.empty())
    {
        for (auto const& [lmSlot, actionSlot] : mLmToActionSlots)
        {
            if (lmSlot < 0 || static_cast<std::size_t>(lmSlot) >= languageOutputs.size()
                || languageOutputs[static_cast<std::size_t>(lmSlot)] == nullptr)
            {
                LOG_ERROR(
                    "ActionRunner: invalid language output slot %d (num outputs=%zu)", lmSlot, languageOutputs.size());
                return false;
            }
            auto const& actionName = mInputNames[static_cast<std::size_t>(actionSlot)];
            if (!copyInputFrom(actionName, *languageOutputs[static_cast<std::size_t>(lmSlot)], stream))
            {
                return false;
            }
        }
        return true;
    }

    if (hasInputTensor(mContextEmbedsName) && languageOutputs.size() == 1 && languageOutputs.front() != nullptr)
    {
        return copyInputFrom(mContextEmbedsName, *languageOutputs.front(), stream);
    }

    return true;
}

bool ActionRunner::isRolloutManagedInput(std::string const& name) const noexcept
{
    if (name == mNoiseInputName || name == mTimestepName)
    {
        return true;
    }
    for (auto const& wiredName : mLmWiredInputNames)
    {
        if (wiredName == name)
        {
            return true;
        }
    }
    return false;
}

bool ActionRunner::preparePi05SuffixInputs(int32_t activeBatchSize, int32_t prefixValidLen, cudaStream_t stream)
{
    if (mContextHandoff != ActionContextHandoff::CONTEXT_TENSOR)
    {
        return true;
    }
    if (!hasInputTensor("position_ids") || !hasInputTensor("attention_mask"))
    {
        return true;
    }

    int32_t const prefixSeqLen = mConfigJson.value("prefix_seq_len", prefixValidLen);
    int32_t const suffixLen = mConfig.actionHorizon;
    if (suffixLen <= 0 || prefixSeqLen <= 0)
    {
        LOG_ERROR("ActionRunner::preparePi05SuffixInputs requires positive prefix_seq_len and action horizon");
        return false;
    }

    int32_t const totalKeys = prefixSeqLen + suffixLen;
    constexpr float kOpenPiAttentionMaskValue = -2.3819763e38F;

    auto& positionIdsTensor = mInputTensors.at(velocityInputIndex("position_ids"));
    auto& attentionMaskTensor = mInputTensors.at(velocityInputIndex("attention_mask"));

    rt::Tensor positionIdsHost(positionIdsTensor.getShape(), rt::DeviceType::kCPU, positionIdsTensor.getDataType(),
        "ActionRunner::preparePi05SuffixInputs::positionIdsHost");
    rt::Tensor attentionMaskHost(attentionMaskTensor.getShape(), rt::DeviceType::kCPU,
        attentionMaskTensor.getDataType(), "ActionRunner::preparePi05SuffixInputs::attentionMaskHost");

    std::vector<int64_t> suffixAttAr(static_cast<std::size_t>(suffixLen), 0);
    if (!suffixAttAr.empty())
    {
        suffixAttAr[0] = 1;
    }
    std::vector<int64_t> suffixAttCumsum(static_cast<std::size_t>(suffixLen), 0);
    int64_t attCumsumRunning{0};
    for (int32_t idx = 0; idx < suffixLen; ++idx)
    {
        attCumsumRunning += suffixAttAr[static_cast<std::size_t>(idx)];
        suffixAttCumsum[static_cast<std::size_t>(idx)] = attCumsumRunning;
    }

    int64_t* positionIdsData = positionIdsHost.dataPointer<int64_t>();
    float* attentionMaskData = attentionMaskHost.dataPointer<float>();
    for (int32_t batchIdx = 0; batchIdx < activeBatchSize; ++batchIdx)
    {
        for (int32_t queryIdx = 0; queryIdx < suffixLen; ++queryIdx)
        {
            positionIdsData[(static_cast<std::size_t>(batchIdx) * static_cast<std::size_t>(suffixLen))
                + static_cast<std::size_t>(queryIdx)] = static_cast<int64_t>(prefixValidLen + queryIdx);

            for (int32_t keyIdx = 0; keyIdx < totalKeys; ++keyIdx)
            {
                bool canAttend{false};
                if (keyIdx < prefixSeqLen)
                {
                    canAttend = keyIdx < prefixValidLen;
                }
                else
                {
                    int32_t const suffixKeyIdx = keyIdx - prefixSeqLen;
                    canAttend = suffixAttCumsum[static_cast<std::size_t>(queryIdx)]
                        >= suffixAttCumsum[static_cast<std::size_t>(suffixKeyIdx)];
                }

                std::size_t const maskOffset = (static_cast<std::size_t>(batchIdx) * static_cast<std::size_t>(suffixLen)
                                                   * static_cast<std::size_t>(totalKeys))
                    + (static_cast<std::size_t>(queryIdx) * static_cast<std::size_t>(totalKeys))
                    + static_cast<std::size_t>(keyIdx);
                attentionMaskData[maskOffset] = canAttend ? 0.0F : kOpenPiAttentionMaskValue;
            }
        }
    }

    CUDA_CHECK(cudaMemcpyAsync(positionIdsTensor.rawPointer(), positionIdsHost.rawPointer(),
        tensorBytes(positionIdsTensor), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(attentionMaskTensor.rawPointer(), attentionMaskHost.rawPointer(),
        tensorBytes(attentionMaskTensor), cudaMemcpyHostToDevice, stream));
    return true;
}

bool ActionRunner::wireStaticInputs(LLMGenerationRequest const& request, cudaStream_t stream)
{
    if (mContextHandoff != ActionContextHandoff::CONTEXT_TENSOR)
    {
        return true;
    }

    int32_t const numUniqueRequests = static_cast<int32_t>(request.requests.size());
    int32_t const actionBatchSize
        = (request.actionBatchSize > 0) ? request.actionBatchSize : std::max(mActiveActionBatchSize, 1);

    auto resolveEmbodimentId = [&](LLMGenerationRequest::Request const& req) -> int64_t {
        if (req.embodimentId.has_value())
        {
            return req.embodimentId.value();
        }
        if (request.embodimentId.has_value())
        {
            return request.embodimentId.value();
        }
        if (mConfigJson.contains("embodiment_id") && mConfigJson.at("embodiment_id").is_array()
            && !mConfigJson.at("embodiment_id").empty())
        {
            return mConfigJson.at("embodiment_id").at(0).get<int64_t>();
        }
        return 31;
    };

    for (auto const& name : mInputNames)
    {
        if (isRolloutManagedInput(name))
        {
            continue;
        }

        auto& tensor = mInputTensors.at(velocityInputIndex(name));
        auto const elements = static_cast<std::size_t>(tensor.getShape().volume());
        std::vector<float> hostValues(elements, 0.0F);

        if (name == "state")
        {
            for (int32_t batchIdx = 0; batchIdx < actionBatchSize; ++batchIdx)
            {
                int32_t const reqIdx = numUniqueRequests > 0 ? batchIdx % numUniqueRequests : 0;
                auto const& req = request.requests[static_cast<std::size_t>(reqIdx)];
                if (!req.robotState.empty())
                {
                    if (req.robotState.size() != elements)
                    {
                        LOG_ERROR("ActionRunner: robot state size mismatch for batch %d (expected %zu, got %zu)",
                            batchIdx, elements, req.robotState.size());
                        return false;
                    }
                    hostValues = req.robotState;
                }
                break;
            }
        }
        else if (name == "embodiment_id")
        {
            int64_t const embodimentId = resolveEmbodimentId(
                request.requests.empty() ? LLMGenerationRequest::Request{} : request.requests.front());
            hostValues.assign(elements, static_cast<float>(embodimentId));
        }

        rt::Tensor hostTensor(tensor.getShape(), rt::DeviceType::kCPU, tensor.getDataType(), name + "_host");
        if (!fillHostTensorFromFloats(hostTensor, hostValues))
        {
            return false;
        }
        CUDA_CHECK(cudaMemcpyAsync(
            tensor.rawPointer(), hostTensor.rawPointer(), tensorBytes(tensor), cudaMemcpyHostToDevice, stream));
    }

    return true;
}

bool ActionRunner::copyActionsToHost(std::vector<std::vector<float>>& actionsPerBatch, cudaStream_t stream) const
{
    if (mContextHandoff != ActionContextHandoff::CONTEXT_TENSOR)
    {
        return false;
    }

    rt::Tensor const& actions = getActions();
    int32_t const batchSize = mActiveActionBatchSize > 0 ? mActiveActionBatchSize : mMaxActionBatchSize;
    int64_t const perBatchElements = (actions.getShape().getNumDims() > 0)
        ? actions.getShape().volume() / std::max(actions.getShape()[0], int64_t{1})
        : actions.getShape().volume();

    rt::Tensor actionsHost(
        actions.getShape(), rt::DeviceType::kCPU, actions.getDataType(), "ActionRunner::actionsHost");
    CUDA_CHECK(cudaMemcpyAsync(
        actionsHost.rawPointer(), actions.rawPointer(), tensorBytes(actions), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    actionsPerBatch.clear();
    actionsPerBatch.resize(static_cast<std::size_t>(batchSize));
    for (int32_t batchIdx = 0; batchIdx < batchSize; ++batchIdx)
    {
        actionsPerBatch[static_cast<std::size_t>(batchIdx)].resize(static_cast<std::size_t>(perBatchElements));
        auto& dst = actionsPerBatch[static_cast<std::size_t>(batchIdx)];
        if (actions.getDataType() == nvinfer1::DataType::kFLOAT)
        {
            auto const* src = actionsHost.dataPointer<float>()
                + static_cast<std::size_t>(batchIdx) * static_cast<std::size_t>(perBatchElements);
            std::copy(src, src + static_cast<std::size_t>(perBatchElements), dst.begin());
        }
        else if (actions.getDataType() == nvinfer1::DataType::kHALF)
        {
            auto const* src = actionsHost.dataPointer<half>()
                + static_cast<std::size_t>(batchIdx) * static_cast<std::size_t>(perBatchElements);
            for (int64_t i = 0; i < perBatchElements; ++i)
            {
                dst[static_cast<std::size_t>(i)] = __half2float(src[static_cast<std::size_t>(i)]);
            }
        }
        else
        {
            LOG_ERROR("ActionRunner: unsupported action dtype for export");
            return false;
        }
    }

    return true;
}

bool ActionRunner::bindVelocityTensors() noexcept
{
    for (std::size_t i = 0; i < mInputNames.size(); ++i)
    {
        if (!bindTensor(mEngine.get(), mContext.get(), engineBindingName(i), mInputTensors[i]))
        {
            return false;
        }
    }
    return bindTensor(mEngine.get(), mContext.get(), mPredVelocityName, mPredVelocity);
}

void ActionRunner::buildEngineBindingNames()
{
    std::vector<std::string> engineInputNames;
    for (int32_t i = 0; i < mEngine->getNbIOTensors(); ++i)
    {
        char const* name = mEngine->getIOTensorName(i);
        if (name != nullptr && mEngine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT)
        {
            engineInputNames.emplace_back(name);
        }
    }

    if (engineInputNames.size() < mInputNames.size())
    {
        throw std::runtime_error("ActionRunner: engine has fewer inputs than config input_names");
    }

    mEngineBindingNames.clear();
    mEngineBindingNames.reserve(mInputNames.size());
    for (std::size_t i = 0; i < mInputNames.size(); ++i)
    {
        auto const& logicalName = mInputNames[i];
        if (engineHasTensor(logicalName))
        {
            mEngineBindingNames.push_back(logicalName);
        }
        else
        {
            mEngineBindingNames.push_back(engineInputNames[i]);
            LOG_WARNING("ActionRunner: logical input '%s' mapped to engine binding '%s'", logicalName.c_str(),
                engineInputNames[i].c_str());
        }
    }
}

std::string const& ActionRunner::engineBindingName(std::size_t inputIndex) const
{
    return mEngineBindingNames.at(inputIndex);
}

bool ActionRunner::setTimestepForStep(int32_t step, cudaStream_t stream)
{
    auto const elements = mTimestepHost.getShape().volume();

    if (mTimestepSchedule == "continuous_flow")
    {
        if (mTimestepHost.getDataType() != nvinfer1::DataType::kFLOAT)
        {
            LOG_ERROR("ActionRunner: continuous_flow schedule requires float32 timestep");
            return false;
        }

        auto const dt = -1.0F / static_cast<float>(mConfig.numInferenceTimesteps);
        auto const timestepValue = 1.0F + static_cast<float>(step) * dt;
        auto* data = mTimestepHost.dataPointer<float>();
        for (int64_t i = 0; i < elements; ++i)
        {
            data[i] = timestepValue;
        }
    }
    else
    {
        check::check(mNumTimestepBuckets > 0, "ActionRunner num_timestep_buckets must be positive");
        auto const bucket = static_cast<int64_t>(std::floor(static_cast<float>(step)
            / static_cast<float>(mConfig.numInferenceTimesteps) * static_cast<float>(mNumTimestepBuckets)));

        if (mTimestepHost.getDataType() == nvinfer1::DataType::kINT64)
        {
            auto* data = mTimestepHost.dataPointer<int64_t>();
            for (int64_t i = 0; i < elements; ++i)
            {
                data[i] = bucket;
            }
        }
        else if (mTimestepHost.getDataType() == nvinfer1::DataType::kINT32)
        {
            auto* data = mTimestepHost.dataPointer<int32_t>();
            for (int64_t i = 0; i < elements; ++i)
            {
                data[i] = static_cast<int32_t>(bucket);
            }
        }
        else
        {
            LOG_ERROR("ActionRunner: discrete_buckets schedule requires int32 or int64 timestep");
            return false;
        }
    }

    auto& timestepTensor = mInputTensors.at(velocityInputIndex(mTimestepName));
    CUDA_CHECK(cudaMemcpyAsync(timestepTensor.rawPointer(), mTimestepHost.rawPointer(), tensorBytes(timestepTensor),
        cudaMemcpyHostToDevice, stream));
    return true;
}

bool ActionRunner::updateActionsOnHost(cudaStream_t stream, float stepSize)
{
    CUDA_CHECK(cudaMemcpyAsync(mPredVelocityHost.rawPointer(), mPredVelocity.rawPointer(), tensorBytes(mPredVelocity),
        cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(
        mNoiseHost.rawPointer(), getActions().rawPointer(), tensorBytes(getActions()), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    updateActionTensor(mNoiseHost, mPredVelocityHost, stepSize);

    CUDA_CHECK(cudaMemcpyAsync(
        getActions().rawPointer(), mNoiseHost.rawPointer(), tensorBytes(getActions()), cudaMemcpyHostToDevice, stream));
    return true;
}

bool ActionRunner::sampleActions(cudaStream_t stream)
{
    if (mContextHandoff != ActionContextHandoff::CONTEXT_TENSOR)
    {
        LOG_ERROR("ActionRunner::sampleActions requires CONTEXT_TENSOR handoff");
        return false;
    }

    check::check(mConfig.numInferenceTimesteps > 0, "ActionRunner num_inference_timesteps must be positive");

    int32_t const activeBatchSize = mActiveActionBatchSize > 0 ? mActiveActionBatchSize : mMaxActionBatchSize;
    if (!reshapeActionTensorsForActiveBatch(activeBatchSize))
    {
        return false;
    }

    CUDA_CHECK(cudaMemcpyAsync(
        getActions().rawPointer(), mNoiseHost.rawPointer(), tensorBytes(getActions()), cudaMemcpyHostToDevice, stream));

    auto const stepSize = static_cast<float>(mRolloutDtSign) / static_cast<float>(mConfig.numInferenceTimesteps);

    for (int32_t step = 0; step < mConfig.numInferenceTimesteps; ++step)
    {
        if (!setTimestepForStep(step, stream) || !bindVelocityTensors() || !mContext->enqueueV3(stream)
            || !updateActionsOnHost(stream, stepSize))
        {
            return false;
        }
    }

    return true;
}

} // namespace rt
} // namespace trt_edgellm
