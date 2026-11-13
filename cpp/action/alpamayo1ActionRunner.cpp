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
#include "common/executionPhase.h"
#include "common/logger.h"
#include "common/pagedKvTypes.h"
#include "common/trtInterop.h"
#include "common/trtUtils.h"
#include "kernels/kvCacheUtilKernels/kvCacheUtilsKernels.h"
#include "kernels/posEncoding/initializeCosSinCache.h"
#include "profiling/metrics.h"
#include "profiling/nvtx_wrapper.h"
#include "profiling/timer.h"
#include "runtime/weight/refitWeights.h"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
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
            LOG_ERROR(
                "Special tokens <|traj_history_start|>, <|traj_history|>, or <|traj_history_end|> not found in "
                "tokenizer; cannot preprocess an Alpamayo action request.");
            return false;
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

    if (isRoot())
    {
        try
        {
            initializeNoiseTrajectory(mNoiseSeed, actionBatchSize);
        }
        catch (std::exception const& e)
        {
            LOG_ERROR("Initial noise trajectory initialization failed: %s", e.what());
            return false;
        }
    }
    return true;
}

Alpamayo1ActionRunner::Alpamayo1ActionRunner(std::string const& engineDir, std::string const& checkpointDir,
    cudaStream_t stream, KVCacheManager::Config const& kvCacheConfig, int32_t tensorParallelSize,
    int32_t tensorParallelRank, std::vector<std::array<float, 2>> kvCacheScales)
    : mStream(stream)
    , mTensorParallelSize(tensorParallelSize)
    , mTensorParallelRank(tensorParallelRank)
    , mUseFp8KV(kvCacheConfig.kvCacheType == DataType::kFP8)
    , mKVCacheScales(std::move(kvCacheScales))
{
    ELLM_CHECK(mTensorParallelSize > 0 && mTensorParallelRank >= 0 && mTensorParallelRank < mTensorParallelSize,
        "Invalid tensor-parallel mapping for Alpamayo action runner");

    bool const configParsed = parseModelConfig(engineDir + "/config.json");
    ELLM_CHECK(configParsed, "Failed to parse model config");
    ELLM_CHECK(kvCacheConfig.layerConfigs.size() == static_cast<size_t>(mConfig.numDecoderLayers),
        "Action runner requires one KV cache layer for every Alpamayo decoder layer");
    ELLM_CHECK(!kvCacheConfig.layerConfigs.empty(), "Action runner requires at least one KV cache layer");
    mLocalKVHeads = kvCacheConfig.layerConfigs.front().numKVHeads;
    for (auto const& layer : kvCacheConfig.layerConfigs)
    {
        ELLM_CHECK(layer.numKVHeads == mLocalKVHeads && layer.headDim == mConfig.headDim,
            "Action runner requires uniform KV head dimensions across decoder layers");
    }
    ELLM_CHECK(mLocalKVHeads * mTensorParallelSize == mConfig.numKVHeads,
        "Action engine KV heads must equal the sum of the tensor-parallel LLM KV-head shards");
    if (mTensorParallelSize > 1)
    {
        ELLM_CHECK(kvCacheConfig.kvCacheType == DataType::kHALF || mUseFp8KV,
            "Dense action gather supports only FP16 or FP8 LLM KV caches");
    }
    if (mUseFp8KV)
    {
        ELLM_CHECK(mKVCacheScales.size() == static_cast<size_t>(mConfig.numDecoderLayers),
            "FP8 Alpamayo action inference requires one K/V dequant scale pair per decoder layer");
        if (mTensorParallelSize == 1)
        {
            ELLM_CHECK(mKVCacheScales == mConfig.kvCacheScales,
                "Action FP8 KV scales do not match the LLM cache. Export both engines from the same checkpoint.");
        }
    }

    if (isRoot())
    {
        LOG_DEBUG("Loading action runner from %s", engineDir.c_str());

        std::string actionEnginePath = engineDir + "/action.engine";

        mRuntime = std::unique_ptr<IRuntime>(createInferRuntime(TrtLogger::instance()));
        ELLM_CHECK(mRuntime, "Failed to create TensorRT runtime");

        mEngine = deserializeCudaEngineFromFile(*mRuntime, actionEnginePath);

        mExternalWeights = std::make_unique<ExternalWeightManager>();
        mExternalWeights->load(engineDir, engineDir + "/config.json", stream, checkpointDir);
        refitEngineIfNeeded(*mEngine, EngineRefitOptions{mExternalWeights->takeRefitWeights(), stream});

        bool hasPagedPrefix = false;
        bool hasDensePrefix = false;
        for (int32_t i = 0; i < mEngine->getNbIOTensors(); ++i)
        {
            std::string const name = mEngine->getIOTensorName(i);
            hasPagedPrefix |= name == binding_names::formatKVCacheName(0, true);
            hasDensePrefix |= name == binding_names::formatKCacheName(0, true);
        }
        ELLM_CHECK(mTensorParallelSize == 1 ? (hasPagedPrefix && !hasDensePrefix) : (hasDensePrefix && !hasPagedPrefix),
            "Action engine KV contract does not match LLM tensor parallelism. "
            "For ONNX export, match --tp-size to the LLM. For direct action builds, use "
            "--reuse-llm-kv-cache=true for LLM TP=1 or --reuse-llm-kv-cache=false for LLM TP>1. "
            "Action --tp-size defaults to 1 and can be omitted.");
        if (mTensorParallelSize == 1)
        {
            int64_t const actualPages = kvCacheConfig.numPages == 0
                ? computeMinimumKvPoolPages(kvCacheConfig.maxBatchSize, kvCacheConfig.maxSequenceLength)
                : kvCacheConfig.numPages;
            for (int32_t layer = 0; layer < mConfig.numDecoderLayers; ++layer)
            {
                std::string const name = binding_names::formatKVCacheName(layer, true);
                validateKVPoolDataType(
                    toTrt(kvCacheConfig.kvCacheType), mEngine->getTensorDataType(name.c_str()), name);
                std::string const outputName = binding_names::formatKVCacheName(layer, false);
                validateKVPoolDataType(
                    nvinfer1::DataType::kHALF, mEngine->getTensorDataType(outputName.c_str()), outputName);
                std::string const actionName = binding_names::formatActionKVCacheName(layer);
                validateKVPoolDataType(
                    nvinfer1::DataType::kHALF, mEngine->getTensorDataType(actionName.c_str()), actionName);
                Dims const minShape = mEngine->getProfileShape(name.c_str(), 0, OptProfileSelector::kMIN);
                Dims const optShape = mEngine->getProfileShape(name.c_str(), 0, OptProfileSelector::kOPT);
                Dims const maxShape = mEngine->getProfileShape(name.c_str(), 0, OptProfileSelector::kMAX);
                ELLM_CHECK(minShape.nbDims == 5 && optShape.nbDims == 5 && maxShape.nbDims == 5,
                    "Action engine requires canonical paged KV inputs.");
                validateKVPoolPageCount(actualPages, minShape.d[1], optShape.d[1], maxShape.d[1]);
            }
        }

        mContext = std::unique_ptr<IExecutionContext>(
            mEngine->createExecutionContext(ExecutionContextAllocationStrategy::kUSER_MANAGED));
        ELLM_CHECK(mContext, "Failed to create execution context");

        bool const profileSet = mContext->setOptimizationProfileAsync(0, stream);
        ELLM_CHECK(profileSet, "Failed to set optimization profile");

        setNonBlockingAuxStreams(mContext.get(), mEngine.get(), mAuxStreams);

        if (mExternalWeights->size() > 0)
        {
            mExternalWeights->bindToContext(*mEngine, *mContext, "action");
        }
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
    if (!mContext)
    {
        return true;
    }
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

void Alpamayo1ActionRunner::setDenseInputShapes(int32_t activeBatchSize)
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
    status &= mContext->setInputShape(binding_names::kActionAttentionPosId, ropeIdxShape);
    for (int32_t i = 0; i < mConfig.numDecoderLayers; ++i)
    {
        status &= mContext->setInputShape(binding_names::formatKCacheName(i, true).c_str(), kvShape);
        status &= mContext->setInputShape(binding_names::formatVCacheName(i, true).c_str(), kvShape);
    }
    if (!status)
    {
        LOG_ERROR("setDenseInputShapes failed for activeBatchSize=%d", activeBatchSize);
        throw std::runtime_error("setDenseInputShapes failed");
    }
}

void Alpamayo1ActionRunner::setPagedInputShapes(int32_t activeBatchSize, HybridCacheManager const& kvcache)
{
    Dims const batchShape = {1, {activeBatchSize}};
    Dims const queryOffsetsShape = {1, {activeBatchSize + 1}};
    Dims const contextMaskSelectorShape = {1, {activeBatchSize}};
    Dims const executionPhaseShape = {1, {static_cast<int64_t>(ExecutionPhase::kDiffusionDenoise)}};
    Dims const contextSequenceCountShape = {1, {0}};
    Dims const noiseShape = {3, {activeBatchSize, mNumWaypoints, 2}};
    Dims const ropeCosShape = {3, {activeBatchSize, mNumWaypoints, mRopeHeadDim}};
    int32_t const prefixMaxPagesPerSeq = rt::computeMaxPagesPerSeq(mMaxSequenceLength);
    Dims const pageTableShape = {3, {activeBatchSize, 2, prefixMaxPagesPerSeq}};

    bool status = true;
    status &= mContext->setInputShape(binding_names::kQueryLengths, batchShape);
    status &= mContext->setInputShape(binding_names::kPastLengths, batchShape);
    status &= mContext->setInputShape(binding_names::kAttentionSequenceLengths, batchShape);
    status &= mContext->setInputShape(binding_names::kQueryStartOffsets, queryOffsetsShape);
    status &= mContext->setInputShape(binding_names::kKVPageTable, pageTableShape);
    status &= mContext->setInputShape(binding_names::kContextMaskSelector, contextMaskSelectorShape);
    status &= mContext->setInputShape(binding_names::kExecutionPhaseMarker, executionPhaseShape);
    status &= mContext->setInputShape(binding_names::kContextSequenceCountCarrier, contextSequenceCountShape);
    status &= mContext->setInputShape(binding_names::kNoiseTrajectory, noiseShape);
    status &= mContext->setInputShape(binding_names::kRopeCosSin, ropeCosShape);
    for (int32_t i = 0; i < mConfig.numDecoderLayers; ++i)
    {
        auto const& pool = kvcache.getCombinedKVCache(i);
        status &= mContext->setInputShape(binding_names::formatKVCacheName(i, true).c_str(), rt::toTrtDims(pool));
    }
    if (!status)
    {
        LOG_ERROR("setPagedInputShapes failed for activeBatchSize=%d", activeBatchSize);
        throw std::runtime_error("setPagedInputShapes failed");
    }
    ELLM_CHECK(mContext->setInputShape(
                   binding_names::kActionPageTable, Dims{3, {activeBatchSize, 2, mActionPageTable->maxPagesPerSeq()}}),
        "Failed to set action page-table shape.");
    for (int32_t layer = 0; layer < mConfig.numDecoderLayers; ++layer)
    {
        auto const& shape = mActionKVPools[layer].getShape();
        ELLM_CHECK(mContext->setInputShape(binding_names::formatActionKVCacheName(layer).c_str(),
                       Dims{5, {shape[0], shape[1], shape[2], shape[3], shape[4]}}),
            "Failed to set FP16 action pool shape.");
    }
}

bool Alpamayo1ActionRunner::gatherDenseKV(
    cudaStream_t stream, HybridCacheManager const& kvcache, ActionKvBatchView const& batch)
{
    int32_t const activeBatchSize = batch.batchSize;
    int32_t const* const lengthsHost = batch.kvLengthsHost.dataPointer<int32_t>();
    KVCacheManager::Config const& kvConfig = kvcache.getKVCacheManager().getConfig();
    size_t const elemSize = rt::utils::getTypeSize(kvConfig.kvCacheType);
    int32_t const liveSequenceLength = *std::max_element(lengthsHost, lengthsHost + activeBatchSize);
    size_t const localPlaneElements
        = static_cast<size_t>(activeBatchSize) * liveSequenceLength * mLocalKVHeads * mConfig.headDim;
    size_t const localPlaneBytes = localPlaneElements * sizeof(uint16_t);
    size_t const sendElements = 2ULL * localPlaneElements;
    ELLM_CHECK(sendElements <= static_cast<size_t>(std::numeric_limits<int32_t>::max()),
        "Action KV TP all-gather element count exceeds int32");
    for (int32_t i = 0; i < mConfig.numDecoderLayers; ++i)
    {
        auto* const localK = static_cast<uint8_t*>(mLocalKVScratch.rawPointer());
        auto* const localV = localK + localPlaneBytes;
        float const kScale = mUseFp8KV ? mKVCacheScales[static_cast<size_t>(i)][0] : 1.0F;
        float const vScale = mUseFp8KV ? mKVCacheScales[static_cast<size_t>(i)][1] : 1.0F;
        kernel::gatherPagedKVToSplit(kvcache.getCombinedKVCache(i).rawPointer(), localK, localV,
            batch.pageTable.kernelView().dataPointer<int32_t>(), batch.kvLengthsDevice.dataPointer<int32_t>(),
            batch.pageTable.maxPagesPerSeq(), activeBatchSize, liveSequenceLength, mLocalKVHeads, mConfig.headDim,
            elemSize, mUseFp8KV, kScale, vScale, stream);

        ELLM_CHECK(mKvAllGather, "Action KV TP all-gather callback was not initialized");
        if (!mKvAllGather(mLocalKVScratch.rawPointer(), mGatheredKVScratch.rawPointer(),
                static_cast<int32_t>(sendElements), stream))
        {
            LOG_ERROR("Action KV TP all-gather failed for decoder layer %d", i);
            return false;
        }

        if (isRoot())
        {
            rt::Tensor& kCacheBlock = mKCacheLayers[static_cast<size_t>(i)];
            rt::Tensor& vCacheBlock = mVCacheLayers[static_cast<size_t>(i)];
            kernel::mergeGatheredSplitKVToHeadMajor(mGatheredKVScratch.rawPointer(), kCacheBlock.rawPointer(),
                vCacheBlock.rawPointer(), mTensorParallelSize, activeBatchSize, liveSequenceLength, mMaxSequenceLength,
                mLocalKVHeads, mConfig.headDim, stream);
        }
    }

    return true;
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

    if (mTensorParallelSize > 1)
    {
        if (!gatherDenseKV(stream, kvcache, batch) || !isRoot())
        {
            return result;
        }
        if (!reshapeActionTensorsForActiveBatch(activeBatchSize))
        {
            LOG_ERROR("reshapeActionTensorsForActiveBatch failed for activeBatchSize=%d", activeBatchSize);
            return result;
        }
        setDenseInputShapes(activeBatchSize);
    }
    else
    {
        setPagedInputShapes(activeBatchSize, kvcache);
    }

    bool setEngineIOStatus{true};
    if (mTensorParallelSize == 1)
    {
        int32_t* const queryLengths = mQueryLengthsHost.dataPointer<int32_t>();
        int32_t* const attentionLengths = mAttentionSequenceLengthsHost.dataPointer<int32_t>();
        int32_t* const queryOffsets = mQueryStartOffsetsHost.dataPointer<int32_t>();
        queryOffsets[0] = 0;
        for (int32_t b = 0; b < activeBatchSize; ++b)
        {
            queryLengths[b] = mNumWaypoints;
            attentionLengths[b] = lengthsHost[b] + mNumWaypoints;
            queryOffsets[b + 1] = queryOffsets[b] + mNumWaypoints;
        }

        CUDA_CHECK(cudaMemcpyAsync(mQueryLengthsDevice.rawPointer(), mQueryLengthsHost.rawPointer(),
            static_cast<size_t>(activeBatchSize) * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(
            cudaMemcpyAsync(mAttentionSequenceLengthsDevice.rawPointer(), mAttentionSequenceLengthsHost.rawPointer(),
                static_cast<size_t>(activeBatchSize) * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(mQueryStartOffsetsDevice.rawPointer(), mQueryStartOffsetsHost.rawPointer(),
            static_cast<size_t>(activeBatchSize + 1) * sizeof(int32_t), cudaMemcpyHostToDevice, stream));

        setEngineIOStatus
            &= mContext->setTensorAddress(binding_names::kActionPageTable, mActionPageTable->kernelView().rawPointer());
        setEngineIOStatus &= mContext->setTensorAddress(binding_names::kQueryLengths, mQueryLengthsDevice.rawPointer());
        setEngineIOStatus &= mContext->setTensorAddress(
            binding_names::kPastLengths, const_cast<void*>(batch.kvLengthsDevice.rawPointer()));
        setEngineIOStatus &= mContext->setTensorAddress(
            binding_names::kAttentionSequenceLengths, mAttentionSequenceLengthsDevice.rawPointer());
        setEngineIOStatus
            &= mContext->setTensorAddress(binding_names::kQueryStartOffsets, mQueryStartOffsetsDevice.rawPointer());
        setEngineIOStatus &= mContext->setTensorAddress(
            binding_names::kKVPageTable, const_cast<void*>(batch.pageTable.kernelView().rawPointer()));
        setEngineIOStatus
            &= mContext->setTensorAddress(binding_names::kContextMaskSelector, mContextMaskSelectorDevice.rawPointer());
        setEngineIOStatus &= mContext->setTensorAddress(
            binding_names::kExecutionPhaseMarker, mExecutionPhaseMarkerDevice.rawPointer());
        setEngineIOStatus &= mContext->setTensorAddress(
            binding_names::kContextSequenceCountCarrier, mContextSequenceCountCarrierDevice.rawPointer());
    }
    else
    {
        setEngineIOStatus &= mContext->setTensorAddress(
            binding_names::kKVCacheStartIndex, const_cast<void*>(batch.kvLengthsDevice.rawPointer()));
        setEngineIOStatus
            &= mContext->setTensorAddress(binding_names::kActionAttentionPosId, mPositionIdsDevice.rawPointer());
    }
    setEngineIOStatus
        &= mContext->setTensorAddress(binding_names::kNoiseTrajectory, mNoiseTrajectoryDevice.rawPointer());
    setEngineIOStatus &= mContext->setTensorAddress(binding_names::kTimeStepsT0, mTimeStepsT0Device.rawPointer());
    setEngineIOStatus &= mContext->setTensorAddress(binding_names::kTimeStepsT1, mTimeStepsT1Device.rawPointer());
    setEngineIOStatus
        &= mContext->setTensorAddress(binding_names::kDenoisedTrajectory, mDenoisedTrajectoryDevice.rawPointer());
    setEngineIOStatus &= mContext->setTensorAddress(binding_names::kRopeCosSin, mRopeCosSinDevice.rawPointer());

    if (mTensorParallelSize == 1)
    {
        for (int32_t i = 0; i < mConfig.numDecoderLayers; ++i)
        {
            rt::Tensor const& kvPool = kvcache.getCombinedKVCache(i);
            setEngineIOStatus &= mContext->setTensorAddress(
                binding_names::formatKVCacheName(i, true).c_str(), const_cast<void*>(kvPool.rawPointer()));
            setEngineIOStatus &= mContext->setTensorAddress(
                binding_names::formatKVCacheName(i, false).c_str(), mActionKVPools[i].rawPointer());
            setEngineIOStatus &= mContext->setTensorAddress(
                binding_names::formatActionKVCacheName(i).c_str(), mActionKVPools[i].rawPointer());
        }
    }
    else
    {
        for (int32_t i = 0; i < mConfig.numDecoderLayers; ++i)
        {
            rt::Tensor& kCacheBlock = mKCacheLayers[static_cast<size_t>(i)];
            rt::Tensor& vCacheBlock = mVCacheLayers[static_cast<size_t>(i)];
            setEngineIOStatus &= mContext->setTensorAddress(
                binding_names::formatKCacheName(i, true).c_str(), kCacheBlock.rawPointer());
            setEngineIOStatus &= mContext->setTensorAddress(
                binding_names::formatVCacheName(i, true).c_str(), vCacheBlock.rawPointer());
            setEngineIOStatus &= mContext->setTensorAddress(
                binding_names::formatKCacheName(i, false).c_str(), kCacheBlock.rawPointer());
            setEngineIOStatus &= mContext->setTensorAddress(
                binding_names::formatVCacheName(i, false).c_str(), vCacheBlock.rawPointer());
        }
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

    if (mTensorParallelSize > 1)
    {
        // Initialize indices for indexing into the rope table
        int32_t* positionIdsPtr = static_cast<int32_t*>(mPositionIdsHost.rawPointer());
        for (int32_t b = 0; b < activeBatchSize; ++b)
        {
            for (int32_t w = 0; w < mNumWaypoints; ++w)
            {
                positionIdsPtr[b * mNumWaypoints + w] = w;
            }
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

    if (mTensorParallelSize > 1)
    {
        CUDA_CHECK(cudaMemcpyAsync(mPositionIdsDevice.rawPointer(), mPositionIdsHost.rawPointer(),
            static_cast<size_t>(activeBatchSize) * static_cast<size_t>(mNumWaypoints) * sizeof(int32_t),
            cudaMemcpyHostToDevice, stream));
    }

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
            CUDA_CHECK(cudaStreamSynchronize(stream));
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
        mConfig.numActionTokens = jsonConfig.value("n_diffusion_tokens", 0);
        mConfig.kvCacheScales = jsonConfig.value("kv_cache_scales", std::vector<std::array<float, 2>>{});
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

    // Read action-engine dimensions from builder_config.
    if (jsonConfig.contains("builder_config") && jsonConfig["builder_config"].contains("max_kv_cache_capacity"))
    {
        mConfig.maxKVCacheCapacity = jsonConfig["builder_config"]["max_kv_cache_capacity"].get<int32_t>();
        mConfig.maxActionBatchSize = jsonConfig["builder_config"].value("max_batch_size", 0);
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
    if (isRoot())
    {
        Dims const maxNoise = mEngine->getProfileShape(binding_names::kNoiseTrajectory, 0, OptProfileSelector::kMAX);
        ELLM_CHECK(maxNoise.nbDims == 3 && maxNoise.d[0] > 0, "Action engine has an invalid batch profile");
        if (mTensorParallelSize > 1)
        {
            ELLM_CHECK(mConfig.maxActionBatchSize == maxNoise.d[0],
                "Action engine batch profile does not match its config.json");
        }
        mMaxActionBatchSize = std::min(static_cast<int32_t>(maxNoise.d[0]), kvCacheConfig.maxBatchSize);
    }
    else
    {
        ELLM_CHECK(mConfig.maxActionBatchSize > 0, "Action config must define max_batch_size for tensor parallelism");
        mMaxActionBatchSize = std::min(mConfig.maxActionBatchSize, kvCacheConfig.maxBatchSize);
    }
    int32_t const maxBatch = mMaxActionBatchSize;
    mMaxSequenceLength = static_cast<int32_t>(kvCacheConfig.maxSequenceLength);

    if (mTensorParallelSize > 1)
    {
        rt::Coords const localScratchShape{2, maxBatch, mMaxSequenceLength, mLocalKVHeads, mConfig.headDim};
        mLocalKVScratch = rt::Tensor(
            localScratchShape, DeviceType::kGPU, DataType::kHALF, "Alpamayo1ActionRunner::mLocalKVScratch");
        mGatheredKVScratch
            = rt::Tensor({mTensorParallelSize, 2, maxBatch, mMaxSequenceLength, mLocalKVHeads, mConfig.headDim},
                DeviceType::kGPU, DataType::kHALF, "Alpamayo1ActionRunner::mGatheredKVScratch");
    }

    if (!isRoot())
    {
        return;
    }

    nvinfer1::Dims noiseShape = mContext->getTensorShape(binding_names::kNoiseTrajectory);
    ELLM_CHECK(noiseShape.nbDims == 3, "Action engine noise trajectory must be rank 3");
    mNumWaypoints = noiseShape.d[1];
    if (mTensorParallelSize == 1)
    {
        ELLM_CHECK(mNumWaypoints == mConfig.numActionTokens, "Action engine token count does not match config.json.");
        int32_t const pagesPerAction = computeMaxPagesPerSeq(mNumWaypoints);
        auto const actionShape
            = mEngine->getProfileShape(binding_names::formatActionKVCacheName(0).c_str(), 0, OptProfileSelector::kMAX);
        int32_t const actionPages = actionShape.d[1];
        ELLM_CHECK(actionShape.nbDims == 5 && actionPages >= static_cast<int64_t>(maxBatch) * pagesPerAction,
            "Action scratch profile does not cover the configured batch.");
        mActionPageTable = std::make_unique<KVPageTable>(maxBatch, pagesPerAction, actionPages);
        std::vector<std::vector<int32_t>> rows(maxBatch);
        std::vector<KVPageTableRowUpdate> updates;
        for (int32_t b = 0; b < maxBatch; ++b)
        {
            for (int32_t page = 0; page < pagesPerAction; ++page)
            {
                rows[b].push_back(b * pagesPerAction + page);
            }
            updates.push_back({b, rows[b].data(), pagesPerAction});
        }
        mActionPageTable->setRows(updates);
        mActionPageTable->upload(mStream);
        for (int32_t layer = 0; layer < mConfig.numDecoderLayers; ++layer)
        {
            std::string const name = binding_names::formatActionKVCacheName(layer);
            Dims const shape = mEngine->getProfileShape(name.c_str(), 0, OptProfileSelector::kMAX);
            Dims const minShape = mEngine->getProfileShape(name.c_str(), 0, OptProfileSelector::kMIN);
            Dims const optShape = mEngine->getProfileShape(name.c_str(), 0, OptProfileSelector::kOPT);
            ELLM_CHECK(shape.nbDims == 5 && minShape.nbDims == 5 && optShape.nbDims == 5 && shape.d[0] == 2
                    && shape.d[2] == kTOKENS_PER_PAGE && shape.d[3] == mConfig.numKVHeads
                    && shape.d[4] == mConfig.headDim,
                "Invalid FP16 action scratch geometry.");
            ELLM_CHECK(minShape.d[1] == actionPages && optShape.d[1] == actionPages && shape.d[1] == actionPages,
                "Action scratch pool profile mismatch for " + name + ": expected pages=" + std::to_string(actionPages)
                    + ", MIN/OPT/MAX=" + std::to_string(minShape.d[1]) + "/" + std::to_string(optShape.d[1]) + "/"
                    + std::to_string(shape.d[1]) + ". Rebuild action with consistent scratch profiles across layers.");
            mActionKVPools.emplace_back(
                rt::Coords{2, actionPages, kTOKENS_PER_PAGE, mConfig.numKVHeads, mConfig.headDim}, DeviceType::kGPU,
                DataType::kHALF, "Alpamayo1ActionRunner::actionKV");
        }
    }
    rt::Coords noiseCoords({maxBatch, mNumWaypoints, 2});

    mNoiseTrajectoryDevice = rt::Tensor(noiseCoords, rt::DeviceType::kGPU, trt_edgellm::DataType::kFLOAT,
        "Alpamayo1ActionRunner::mNoiseTrajectoryDevice");
    mNoiseTrajectoryHost = rt::Tensor(noiseCoords, rt::DeviceType::kCPU, trt_edgellm::DataType::kFLOAT,
        "Alpamayo1ActionRunner::mNoiseTrajectoryHost");

    mDenoisedTrajectoryDevice = rt::Tensor(noiseCoords, rt::DeviceType::kGPU, trt_edgellm::DataType::kFLOAT,
        "Alpamayo1ActionRunner::mDenoisedTrajectoryDevice");
    mDenoisedTrajectoryHost = rt::Tensor(noiseCoords, rt::DeviceType::kCPU, trt_edgellm::DataType::kFLOAT,
        "Alpamayo1ActionRunner::mDenoisedTrajectoryHost");

    mTimeStepsT0Device = rt::Tensor(
        {1}, rt::DeviceType::kGPU, trt_edgellm::DataType::kFLOAT, "Alpamayo1ActionRunner::mTimeStepsT0Device");
    mTimeStepsT1Device = rt::Tensor(
        {1}, rt::DeviceType::kGPU, trt_edgellm::DataType::kFLOAT, "Alpamayo1ActionRunner::mTimeStepsT1Device");
    mTimeStepsT0Host = rt::Tensor(std::vector<int64_t>{kNumDenoiseSteps}, rt::DeviceType::kCPU,
        trt_edgellm::DataType::kFLOAT, "Alpamayo1ActionRunner::mTimeStepsT0Host");
    mTimeStepsT1Host = rt::Tensor(std::vector<int64_t>{kNumDenoiseSteps}, rt::DeviceType::kCPU,
        trt_edgellm::DataType::kFLOAT, "Alpamayo1ActionRunner::mTimeStepsT1Host");

    if (mTensorParallelSize == 1)
    {
        mQueryLengthsHost = rt::Tensor({maxBatch}, rt::DeviceType::kCPU, trt_edgellm::DataType::kINT32,
            "Alpamayo1ActionRunner::mQueryLengthsHost");
        mQueryLengthsDevice = rt::Tensor({maxBatch}, rt::DeviceType::kGPU, trt_edgellm::DataType::kINT32,
            "Alpamayo1ActionRunner::mQueryLengthsDevice");
        mAttentionSequenceLengthsHost = rt::Tensor({maxBatch}, rt::DeviceType::kCPU, trt_edgellm::DataType::kINT32,
            "Alpamayo1ActionRunner::mAttentionSequenceLengthsHost");
        mAttentionSequenceLengthsDevice = rt::Tensor({maxBatch}, rt::DeviceType::kGPU, trt_edgellm::DataType::kINT32,
            "Alpamayo1ActionRunner::mAttentionSequenceLengthsDevice");
        mQueryStartOffsetsHost = rt::Tensor({maxBatch + 1}, rt::DeviceType::kCPU, trt_edgellm::DataType::kINT32,
            "Alpamayo1ActionRunner::mQueryStartOffsetsHost");
        mQueryStartOffsetsDevice = rt::Tensor({maxBatch + 1}, rt::DeviceType::kGPU, trt_edgellm::DataType::kINT32,
            "Alpamayo1ActionRunner::mQueryStartOffsetsDevice");
        mContextMaskSelectorDevice = rt::Tensor({maxBatch}, rt::DeviceType::kGPU, trt_edgellm::DataType::kINT32,
            "Alpamayo1ActionRunner::mContextMaskSelectorDevice");
        CUDA_CHECK(cudaMemsetAsync(
            mContextMaskSelectorDevice.rawPointer(), 0, mContextMaskSelectorDevice.getMemoryCapacity(), mStream));
        mExecutionPhaseMarkerDevice = rt::Tensor({static_cast<int32_t>(ExecutionPhase::kDiffusionDenoise)},
            rt::DeviceType::kGPU, trt_edgellm::DataType::kINT32, "Alpamayo1ActionRunner::mExecutionPhaseMarkerDevice");
        CUDA_CHECK(cudaMemsetAsync(
            mExecutionPhaseMarkerDevice.rawPointer(), 0, mExecutionPhaseMarkerDevice.getMemoryCapacity(), mStream));
        mContextSequenceCountCarrierDevice = rt::Tensor({1}, rt::DeviceType::kGPU, trt_edgellm::DataType::kINT32,
            "Alpamayo1ActionRunner::mContextSequenceCountCarrierDevice");
        check::check(
            mContextSequenceCountCarrierDevice.reshape({0}), "Action context sequence count carrier reshape failed");
    }
    nvinfer1::Dims ropeCosSinDims = mContext->getTensorShape(binding_names::kRopeCosSin);
    ELLM_CHECK(ropeCosSinDims.nbDims == 3, "Action engine RoPE input must be rank 3");
    mRopeHeadDim = ropeCosSinDims.d[2];
    mRopeCosSinDevice = rt::Tensor({maxBatch, mNumWaypoints, mRopeHeadDim}, rt::DeviceType::kGPU,
        trt_edgellm::DataType::kFLOAT, "Alpamayo1ActionRunner::mRopeCosSinDevice");

    if (mTensorParallelSize > 1)
    {
        mPositionIdsHost = rt::Tensor({maxBatch, mNumWaypoints}, rt::DeviceType::kCPU, trt_edgellm::DataType::kINT32,
            "Alpamayo1ActionRunner::mPositionIdsHost");
        mPositionIdsDevice = rt::Tensor({maxBatch, mNumWaypoints}, rt::DeviceType::kGPU, trt_edgellm::DataType::kINT32,
            "Alpamayo1ActionRunner::mPositionIdsDevice");
    }

    mRopePositionIdsHost = rt::Tensor({maxBatch, 3, mNumWaypoints}, rt::DeviceType::kCPU, trt_edgellm::DataType::kINT64,
        "Alpamayo1ActionRunner::mRopePositionIdsHost");
    mRopePositionIdsDevice = rt::Tensor({maxBatch, 3, mNumWaypoints}, rt::DeviceType::kGPU,
        trt_edgellm::DataType::kINT64, "Alpamayo1ActionRunner::mRopePositionIdsDevice");
    if (mTensorParallelSize > 1)
    {
        rt::Coords const kvcacheShape{maxBatch, mConfig.numKVHeads, kvCacheConfig.maxSequenceLength, mConfig.headDim};
        mKCacheLayers.resize(static_cast<size_t>(mConfig.numDecoderLayers));
        mVCacheLayers.resize(static_cast<size_t>(mConfig.numDecoderLayers));
        for (int32_t i = 0; i < mConfig.numDecoderLayers; ++i)
        {
            mKCacheLayers[i]
                = rt::Tensor(kvcacheShape, DeviceType::kGPU, DataType::kHALF, "Alpamayo1ActionRunner::mKCacheLayer");
            mVCacheLayers[i]
                = rt::Tensor(kvcacheShape, DeviceType::kGPU, DataType::kHALF, "Alpamayo1ActionRunner::mVCacheLayer");
        }
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
