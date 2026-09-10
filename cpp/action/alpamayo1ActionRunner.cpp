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

#include "alpamayo1ActionRunner.h"
#include "action/actionUtils.h"
#include "common/bindingNames.h"
#include "common/checkMacros.h"
#include "common/logger.h"
#include "common/pagedKvTypes.h"
#include "common/trtUtils.h"
#include "kernels/kvCacheUtilKernels/kvCacheUtilsKernels.h"
#include "kernels/posEncoding/initializeCosSinCache.h"
#include "profiling/metrics.h"
#include "profiling/nvtx_wrapper.h"
#include "profiling/timer.h"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <random>
#include <stdexcept>
#include <vector>

using namespace trt_edgellm;
using namespace nvinfer1;
using Json = nlohmann::json;

namespace trt_edgellm
{
namespace rt
{

bool Alpamayo1ActionRunner::preprocess(LLMGenerationRequest const& request,
    std::vector<std::vector<int32_t>>& batchedInputIds, tokenizer::Tokenizer const* tokenizer)
{
    tokenizer::TokenToRanks const& specialTokens = tokenizer->getSpecialTokensEncoder();
    int32_t const requestBatchSize = static_cast<int32_t>(request.requests.size());
    int32_t const actionBatchSize = static_cast<int32_t>(std::count_if(request.requests.begin(), request.requests.end(),
        [](auto const& req) { return req.pastTrajectory.has_value(); }));
    for (int32_t i = 0; i < requestBatchSize; ++i)
    {
        std::vector<int32_t>& tokenIds = batchedInputIds[i];
        LLMGenerationRequest::Request const& req = request.requests[i];

        if (!req.pastTrajectory)
        {
            LOG_WARNING("Request %d has no past trajectory; skipping.", i);
            continue;
        }

        auto const itStart = specialTokens.find(kTrajHistoryStartStr);
        auto const itPad = specialTokens.find(kTrajHistoryPadStr);
        auto const itEnd = specialTokens.find(kTrajHistoryEndStr);
        if (itStart == specialTokens.end() || itPad == specialTokens.end() || itEnd == specialTokens.end())
        {
            LOG_WARNING(
                "Special tokens <|traj_history_start|>, <|traj_history_pad|>, or <|traj_history_end|> not found in "
                "tokenizer; skipping trajectory tokenization.");
            continue;
        }

        tokenizer::Rank const startId = itStart->second;
        tokenizer::Rank const padId = itPad->second;
        tokenizer::Rank const endId = itEnd->second;

        std::vector<tokenizer::Rank> const actualTokens
            = action_utils::trajectoryToTokenIds(*req.pastTrajectory, mConfig.numTrajTokens, mConfig.trajTokenStart);

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
                LOG_ERROR("Trajectory placeholder token count (%zu) does not match trajectory encoding length (%zu).",
                    numPads, actualTokens.size());
                return false;
            }

            for (size_t k = 0; k < actualTokens.size(); ++k)
            {
                tokenIds[scanIdx + 1 + k] = actualTokens[k];
            }
            // <|traj_history_start|> and <|traj_history_end|>, plus the trajectory tokens
            scanIdx += 2 + actualTokens.size();
        }
    }

    if (actionBatchSize > mMaxActionBatchSize)
    {
        LOG_ERROR("action batch size %d exceeds engine max batch size %d", actionBatchSize, mMaxActionBatchSize);
        return false;
    }

    try
    {
        initializeNoiseTrajectory(mNoiseSeed, actionBatchSize);
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("Initial noise trajectory initialization failed: %s", e.what());
        return false;
    }
    return true;
}

Alpamayo1ActionRunner::Alpamayo1ActionRunner(std::string const& engineDir, std::string const& checkpointDir,
    cudaStream_t stream, KVCacheManager::Config const& kvCacheConfig)
    : mStream(stream)
{
    LOG_DEBUG("Loading action runner from %s", engineDir.c_str());

    std::string actionEnginePath = engineDir + "/action.engine";

    // Load runtime and deserialize engine.
    mRuntime = std::unique_ptr<IRuntime>(createInferRuntime(gLogger));
    ELLM_CHECK(mRuntime, "Failed to create TensorRT runtime");

    mEngine = deserializeCudaEngineFromFile(*mRuntime, actionEnginePath);

    mContext = std::unique_ptr<IExecutionContext>(
        mEngine->createExecutionContext(ExecutionContextAllocationStrategy::kUSER_MANAGED));
    ELLM_CHECK(mContext, "Failed to create execution context");

    bool const profileSet = mContext->setOptimizationProfileAsync(0, stream);
    ELLM_CHECK(profileSet, "Failed to set optimization profile");

    setNonBlockingAuxStreams(mContext.get(), mEngine.get(), mAuxStreams);

    bool const configParsed = parseModelConfig(engineDir + "/config.json");
    ELLM_CHECK(configParsed, "Failed to parse model config");

    mExternalWeights = std::make_unique<ExternalWeightManager>();
    mExternalWeights->load(engineDir, engineDir + "/config.json", stream, checkpointDir);
    if (mExternalWeights->size() > 0)
    {
        mExternalWeights->bindToContext(*mEngine, *mContext, "action");
    }

    try
    {
        allocateTensors(kvCacheConfig);
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("Alpamayo1ActionRunner tensor allocation failed: %s", e.what());
        throw;
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));
}

int64_t Alpamayo1ActionRunner::getRequiredContextMemorySize() const
{
    return mEngine ? mEngine->getDeviceMemorySizeV2() : 0;
}

bool Alpamayo1ActionRunner::setContextMemory(rt::Tensor& sharedContextMemory)
{
    int64_t const requiredSize = getRequiredContextMemorySize();
    if (sharedContextMemory.getMemoryCapacity() < requiredSize)
    {
        LOG_ERROR("Shared context memory (%zu bytes) is smaller than action engine requires (%zu bytes)",
            static_cast<size_t>(sharedContextMemory.getMemoryCapacity()), static_cast<size_t>(requiredSize));
        return false;
    }
    mContext->setDeviceMemoryV2(sharedContextMemory.rawPointer(), sharedContextMemory.getMemoryCapacity());
    return true;
}

void Alpamayo1ActionRunner::setDynamicInputShapes(int32_t activeBatchSize)
{
    Dims const kvCacheStartIndexShape = {1, {activeBatchSize}};
    Dims const noiseShape = {3, {activeBatchSize, mNumWaypoints, 2}};
    Dims const ropeCosShape = {3, {activeBatchSize, mNumWaypoints, mRopeHeadDim}};
    Dims const ropeIdxShape = {2, {activeBatchSize, mNumWaypoints}};
    Dims const kvShape = {4,
        {activeBatchSize, static_cast<int64_t>(mConfig.numKVHeads), static_cast<int64_t>(mMaxSequenceLength),
            mConfig.headDim}};

    bool status = true;
    status &= mContext->setInputShape(binding_names::kKVCacheStartIndex, kvCacheStartIndexShape);
    status &= mContext->setInputShape(binding_names::kNoiseTrajectory, noiseShape);
    status &= mContext->setInputShape(binding_names::kRopeCosSin, ropeCosShape);
    status &= mContext->setInputShape(binding_names::kAttentionPosId, ropeIdxShape);
    for (int32_t i = 0; i < mConfig.numDecoderLayers; ++i)
    {
        status &= mContext->setInputShape(binding_names::formatKCacheName(i, true).c_str(), kvShape);
        status &= mContext->setInputShape(binding_names::formatVCacheName(i, true).c_str(), kvShape);
    }
    if (!status)
    {
        LOG_ERROR("setDynamicInputShapes failed for activeBatchSize=%d", activeBatchSize);
        throw std::runtime_error("setDynamicInputShapes failed");
    }
}

std::vector<std::vector<FutureTrajectoryPoint>> Alpamayo1ActionRunner::sampleTrajectory(
    cudaStream_t stream, HybridCacheManager const& kvcache, ActionKvBatchView const& batch)
{
    int32_t const activeBatchSize = batch.batchSize;
    TIME_STAGE(metrics::StageNames::kACTION_INFERENCE, stream);
    NVTX_SCOPED_RANGE(nvtx_action_sample,
        ("ACTION_SAMPLE_TRAJECTORY[BS=" + std::to_string(activeBatchSize) + "]").c_str(), nvtx_colors::PURPLE);

    std::vector<std::vector<FutureTrajectoryPoint>> result;
    if (activeBatchSize <= 0 || activeBatchSize > mMaxActionBatchSize)
    {
        LOG_ERROR("Invalid action batch size %d (maximum %d)", activeBatchSize, mMaxActionBatchSize);
        return result;
    }

    if (!reshapeActionTensorsForActiveBatch(activeBatchSize))
    {
        LOG_ERROR("reshapeActionTensorsForActiveBatch failed for activeBatchSize=%d", activeBatchSize);
        return result;
    }

    if (batch.kvLengthsHost.getDeviceType() != DeviceType::kCPU
        || batch.kvLengthsDevice.getDeviceType() != DeviceType::kGPU
        || batch.kvLengthsHost.getDataType() != DataType::kINT32
        || batch.kvLengthsDevice.getDataType() != DataType::kINT32
        || batch.kvLengthsHost.getShape().volume() < activeBatchSize
        || batch.kvLengthsDevice.getShape().volume() < activeBatchSize
        || static_cast<int32_t>(batch.mropeDeltas.size()) != activeBatchSize)
    {
        LOG_ERROR("Invalid action KV batch view for batch size %d", activeBatchSize);
        return result;
    }
    int32_t const* const lengthsHost = batch.kvLengthsHost.dataPointer<int32_t>();
    for (int32_t b = 0; b < activeBatchSize; ++b)
    {
        if (lengthsHost[b] <= 0 || lengthsHost[b] > mMaxSequenceLength)
        {
            LOG_ERROR("Invalid action KV cache length for row %d: %d", b, lengthsHost[b]);
            return result;
        }
    }

    setDynamicInputShapes(activeBatchSize);

    bool setEngineIOStatus{true};
    setEngineIOStatus &= mContext->setTensorAddress(
        binding_names::kKVCacheStartIndex, const_cast<void*>(batch.kvLengthsDevice.rawPointer()));
    setEngineIOStatus
        &= mContext->setTensorAddress(binding_names::kNoiseTrajectory, mNoiseTrajectoryDevice.rawPointer());
    setEngineIOStatus &= mContext->setTensorAddress(binding_names::kTimeStepsT0, mTimeStepsT0Device.rawPointer());
    setEngineIOStatus &= mContext->setTensorAddress(binding_names::kTimeStepsT1, mTimeStepsT1Device.rawPointer());
    setEngineIOStatus
        &= mContext->setTensorAddress(binding_names::kDenoisedTrajectory, mDenoisedTrajectoryDevice.rawPointer());
    setEngineIOStatus &= mContext->setTensorAddress(binding_names::kRopeCosSin, mRopeCosSinDevice.rawPointer());
    setEngineIOStatus &= mContext->setTensorAddress(binding_names::kAttentionPosId, mPositionIdsDevice.rawPointer());

    KVCacheManager::Config const& kvConfig = kvcache.getKVCacheManager().getConfig();
    size_t const elemSize = rt::utils::getTypeSize(kvConfig.kvCacheType);
    for (int32_t i = 0; i < mConfig.numDecoderLayers; ++i)
    {
        rt::Tensor& kCacheBlock = mKCacheLayers[static_cast<size_t>(i)];
        rt::Tensor& vCacheBlock = mVCacheLayers[static_cast<size_t>(i)];
        kernel::gatherPagedKVToHeadMajor(kvcache.getCombinedKVCache(i).rawPointer(), kCacheBlock.rawPointer(),
            vCacheBlock.rawPointer(), batch.pageTable.kernelView().dataPointer<int32_t>(),
            batch.kvLengthsDevice.dataPointer<int32_t>(), batch.pageTable.maxPagesPerSeq(), activeBatchSize,
            mMaxSequenceLength, mConfig.numKVHeads, mConfig.headDim, elemSize, stream);
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
        LOG_ERROR("Failed to bind action engine input tensors.");
        throw std::runtime_error("Failed to bind input tensors.");
    }

    // Initialize time steps for denoising iterations
    float* t0Host = static_cast<float*>(mTimeStepsT0Host.rawPointer());
    float* t1Host = static_cast<float*>(mTimeStepsT1Host.rawPointer());
    for (int32_t i = 0; i < kNumDenoiseSteps; ++i)
    {
        t0Host[i] = static_cast<float>(i) / kNumDenoiseSteps;
        t1Host[i] = static_cast<float>(i + 1) / kNumDenoiseSteps;
    }

    // Initialize indices for indexing into the rope table
    int32_t* positionIdsPtr = static_cast<int32_t*>(mPositionIdsHost.rawPointer());
    for (int32_t b = 0; b < activeBatchSize; ++b)
    {
        for (int32_t w = 0; w < mNumWaypoints; ++w)
        {
            positionIdsPtr[b * mNumWaypoints + w] = w;
        }
    }

    // Compute position IDs with rope delta offset
    int64_t* mropePosPtr = static_cast<int64_t*>(mRopePositionIdsHost.rawPointer());
    for (int32_t b = 0; b < activeBatchSize; ++b)
    {
        int64_t const basePos = batch.mropeDeltas[static_cast<size_t>(b)] + static_cast<int64_t>(lengthsHost[b]);
        for (int32_t dim = 0; dim < 3; ++dim)
        {
            for (int32_t w = 0; w < mNumWaypoints; ++w)
            {
                mropePosPtr[b * mNumWaypoints * 3 + dim * mNumWaypoints + w] = static_cast<int64_t>(w) + basePos;
            }
        }
    }

    CUDA_CHECK(cudaMemcpyAsync(mRopePositionIdsDevice.rawPointer(), mRopePositionIdsHost.rawPointer(),
        static_cast<size_t>(activeBatchSize) * 3ULL * static_cast<size_t>(mNumWaypoints) * sizeof(int64_t),
        cudaMemcpyHostToDevice, stream));

    kernel::initializeMRopeCosSin(static_cast<float*>(mRopeCosSinDevice.rawPointer()),
        static_cast<int64_t*>(mRopePositionIdsDevice.rawPointer()), mConfig.ropeTheta,
        static_cast<int64_t>(mRopeHeadDim), mNumWaypoints, activeBatchSize, true, mConfig.mropeSectionH,
        mConfig.mropeSectionW, stream);

    // Copy inputs to device
    CUDA_CHECK(cudaMemcpyAsync(mNoiseTrajectoryDevice.rawPointer(), mNoiseTrajectoryHost.rawPointer(),
        static_cast<size_t>(activeBatchSize) * static_cast<size_t>(mNumWaypoints) * 2ULL * sizeof(float),
        cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(mPositionIdsDevice.rawPointer(), mPositionIdsHost.rawPointer(),
        static_cast<size_t>(activeBatchSize) * static_cast<size_t>(mNumWaypoints) * sizeof(int32_t),
        cudaMemcpyHostToDevice, stream));

    // Run denoising iterations
    for (int32_t i = 0; i < kNumDenoiseSteps; ++i)
    {
        float* t0Ptr = static_cast<float*>(mTimeStepsT0Host.rawPointer()) + i;
        float* t1Ptr = static_cast<float*>(mTimeStepsT1Host.rawPointer()) + i;

        CUDA_CHECK(
            cudaMemcpyAsync(mTimeStepsT0Device.rawPointer(), t0Ptr, sizeof(float), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(
            cudaMemcpyAsync(mTimeStepsT1Device.rawPointer(), t1Ptr, sizeof(float), cudaMemcpyHostToDevice, stream));

        if (!mContext->enqueueV3(stream))
        {
            LOG_ERROR("enqueueV3 failed at denoise step %d", i);
            return result;
        }

        CUDA_CHECK(cudaMemcpyAsync(mNoiseTrajectoryDevice.rawPointer(), mDenoisedTrajectoryDevice.rawPointer(),
            static_cast<size_t>(activeBatchSize) * static_cast<size_t>(mNumWaypoints) * 2ULL * sizeof(float),
            cudaMemcpyDeviceToDevice, stream));
    }

    // Copy result back to host
    CUDA_CHECK(cudaMemcpyAsync(mDenoisedTrajectoryHost.rawPointer(), mDenoisedTrajectoryDevice.rawPointer(),
        static_cast<size_t>(activeBatchSize) * static_cast<size_t>(mNumWaypoints) * 2ULL * sizeof(float),
        cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Extract trajectory points
    result.resize(activeBatchSize);
    float* accelKappa = static_cast<float*>(mDenoisedTrajectoryHost.rawPointer());
    for (int32_t b = 0; b < activeBatchSize; ++b)
    {
        result[b].reserve(mNumWaypoints);
        float const* row = accelKappa + b * (mNumWaypoints * 2);
        for (int32_t w = 0; w < mNumWaypoints; ++w)
        {
            result[b].emplace_back(row[w * 2], row[w * 2 + 1]);
        }
    }
    return result;
}

bool Alpamayo1ActionRunner::parseModelConfig(std::string const& configPath)
{
    Json jsonConfig;
    std::ifstream configFile(configPath);
    if (!configFile.is_open())
    {
        LOG_ERROR("Failed to open config file: %s", configPath.c_str());
        return false;
    }
    try
    {
        jsonConfig = Json::parse(configFile);
        configFile.close();
    }
    catch (Json::parse_error const& e)
    {
        LOG_ERROR("Failed to parse config file %s with error: %s", configPath.c_str(), e.what());
        return false;
    }

    try
    {
        mConfig.ropeTheta = jsonConfig.at("rope_theta").get<float>();
        mConfig.numDecoderLayers = jsonConfig.at("num_hidden_layers").get<int32_t>();
        mConfig.numTrajTokens = jsonConfig.at("num_traj_tokens").get<int32_t>();
        mConfig.trajTokenStart = jsonConfig.at("traj_token_start").get<int32_t>();
        mConfig.numKVHeads = jsonConfig.at("num_key_value_heads").get<int32_t>();
        mConfig.headDim = jsonConfig.at("head_dim").get<int32_t>();
    }
    catch (Json::exception const& e)
    {
        LOG_ERROR("Failed to read required fields from %s: %s", configPath.c_str(), e.what());
        return false;
    }

    auto const& ropeParams = jsonConfig.contains("rope_scaling") ? jsonConfig["rope_scaling"] : jsonConfig;
    if (ropeParams.contains("mrope_section"))
    {
        auto section = ropeParams["mrope_section"].get<std::vector<int32_t>>();
        if (section.size() >= 3)
        {
            mConfig.mropeSectionH = section[1];
            mConfig.mropeSectionW = section[2];
        }
    }
    if (mConfig.mropeSectionH <= 0 || mConfig.mropeSectionW <= 0)
    {
        LOG_ERROR("Failed to parse mrope_section from action config. Got H=%d, W=%d", mConfig.mropeSectionH,
            mConfig.mropeSectionW);
        return false;
    }

    // Read max_kv_cache_capacity from builder_config section
    if (jsonConfig.contains("builder_config") && jsonConfig["builder_config"].contains("max_kv_cache_capacity"))
    {
        mConfig.maxKVCacheCapacity = jsonConfig["builder_config"]["max_kv_cache_capacity"].get<int32_t>();
    }
    else
    {
        LOG_ERROR("max_kv_cache_capacity not found in builder_config section of %s", configPath.c_str());
        return false;
    }

    return true;
}

void Alpamayo1ActionRunner::allocateTensors(KVCacheManager::Config const& kvCacheConfig)
{
    Dims const maxNoise = mEngine->getProfileShape(binding_names::kNoiseTrajectory, 0, OptProfileSelector::kMAX);
    int32_t const maxBatch = std::min(static_cast<int32_t>(maxNoise.d[0]), kvCacheConfig.maxBatchSize);
    mMaxActionBatchSize = maxBatch;

    mMaxSequenceLength = static_cast<int32_t>(kvCacheConfig.maxSequenceLength);

    nvinfer1::Dims noiseShape = mContext->getTensorShape(binding_names::kNoiseTrajectory);
    mNumWaypoints = noiseShape.d[1];
    rt::Coords noiseCoords({maxBatch, mNumWaypoints, 2});

    mNoiseTrajectoryDevice = rt::Tensor(
        noiseCoords, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT, "Alpamayo1ActionRunner::mNoiseTrajectoryDevice");
    mNoiseTrajectoryHost = rt::Tensor(
        noiseCoords, rt::DeviceType::kCPU, nvinfer1::DataType::kFLOAT, "Alpamayo1ActionRunner::mNoiseTrajectoryHost");

    mDenoisedTrajectoryDevice = rt::Tensor(noiseCoords, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT,
        "Alpamayo1ActionRunner::mDenoisedTrajectoryDevice");
    mDenoisedTrajectoryHost = rt::Tensor(noiseCoords, rt::DeviceType::kCPU, nvinfer1::DataType::kFLOAT,
        "Alpamayo1ActionRunner::mDenoisedTrajectoryHost");

    mTimeStepsT0Device = rt::Tensor(
        {1}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT, "Alpamayo1ActionRunner::mTimeStepsT0Device");
    mTimeStepsT1Device = rt::Tensor(
        {1}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT, "Alpamayo1ActionRunner::mTimeStepsT1Device");
    mTimeStepsT0Host = rt::Tensor(std::vector<int64_t>{kNumDenoiseSteps}, rt::DeviceType::kCPU,
        nvinfer1::DataType::kFLOAT, "Alpamayo1ActionRunner::mTimeStepsT0Host");
    mTimeStepsT1Host = rt::Tensor(std::vector<int64_t>{kNumDenoiseSteps}, rt::DeviceType::kCPU,
        nvinfer1::DataType::kFLOAT, "Alpamayo1ActionRunner::mTimeStepsT1Host");

    nvinfer1::Dims ropeCosSinDims = mContext->getTensorShape(binding_names::kRopeCosSin);
    mRopeHeadDim = ropeCosSinDims.d[2];
    mRopeCosSinDevice = rt::Tensor({maxBatch, mNumWaypoints, mRopeHeadDim}, rt::DeviceType::kGPU,
        nvinfer1::DataType::kFLOAT, "Alpamayo1ActionRunner::mRopeCosSinDevice");

    mPositionIdsHost = rt::Tensor({maxBatch, mNumWaypoints}, rt::DeviceType::kCPU, nvinfer1::DataType::kINT32,
        "Alpamayo1ActionRunner::mPositionIdsHost");
    mPositionIdsDevice = rt::Tensor({maxBatch, mNumWaypoints}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32,
        "Alpamayo1ActionRunner::mPositionIdsDevice");

    mRopePositionIdsHost = rt::Tensor({maxBatch, 3, mNumWaypoints}, rt::DeviceType::kCPU, nvinfer1::DataType::kINT64,
        "Alpamayo1ActionRunner::mRopePositionIdsHost");
    mRopePositionIdsDevice = rt::Tensor({maxBatch, 3, mNumWaypoints}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT64,
        "Alpamayo1ActionRunner::mRopePositionIdsDevice");

    rt::Coords const kvcacheShape{
        kvCacheConfig.maxBatchSize, mConfig.numKVHeads, kvCacheConfig.maxSequenceLength, mConfig.headDim};
    mKCacheLayers.resize(static_cast<size_t>(mConfig.numDecoderLayers));
    mVCacheLayers.resize(static_cast<size_t>(mConfig.numDecoderLayers));
    for (int32_t i = 0; i < mConfig.numDecoderLayers; ++i)
    {
        mKCacheLayers[i] = rt::Tensor(
            kvcacheShape, DeviceType::kGPU, kvCacheConfig.kvCacheType, "Alpamayo1ActionRunner::mKCacheLayer");
        mVCacheLayers[i] = rt::Tensor(
            kvcacheShape, DeviceType::kGPU, kvCacheConfig.kvCacheType, "Alpamayo1ActionRunner::mVCacheLayer");
    }
}

bool Alpamayo1ActionRunner::reshapeActionTensorsForActiveBatch(int32_t activeBatchSize)
{
    rt::Coords const noiseShape({activeBatchSize, mNumWaypoints, 2});
    bool ok = true;
    ok &= mNoiseTrajectoryDevice.reshape(noiseShape);
    ok &= mNoiseTrajectoryHost.reshape(noiseShape);
    ok &= mDenoisedTrajectoryDevice.reshape(noiseShape);
    ok &= mDenoisedTrajectoryHost.reshape(noiseShape);
    ok &= mRopeCosSinDevice.reshape({activeBatchSize, mNumWaypoints, mRopeHeadDim});
    ok &= mPositionIdsHost.reshape({activeBatchSize, mNumWaypoints});
    ok &= mPositionIdsDevice.reshape({activeBatchSize, mNumWaypoints});
    ok &= mRopePositionIdsHost.reshape({activeBatchSize, 3, mNumWaypoints});
    ok &= mRopePositionIdsDevice.reshape({activeBatchSize, 3, mNumWaypoints});
    if (!ok)
    {
        LOG_ERROR("Tensor reshape for active batch size failed.");
    }
    return ok;
}

void Alpamayo1ActionRunner::initializeNoiseTrajectory(int32_t randomSeed, int32_t activeBatchSize)
{
    size_t const elemCount = static_cast<size_t>(activeBatchSize) * static_cast<size_t>(mNumWaypoints) * 2ULL;
    float* data = static_cast<float*>(mNoiseTrajectoryHost.rawPointer());
    std::mt19937 generator(static_cast<std::mt19937::result_type>(randomSeed));
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (size_t i = 0; i < elemCount; ++i)
    {
        data[i] = dist(generator);
    }
}

} // namespace rt
} // namespace trt_edgellm
