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

#pragma once

#include "benchLogger.h"
#include "common/bindingNames.h"
#include "common/checkMacros.h"
#include "common/cudaUtils.h"
#include "common/logger.h"
#include "common/tensor.h"
#include "kernels/speculative/ddtreeKernels.h"
#include "kernels/speculative/dflashRuntimeKernels.h"
#include "profiling/layerProfiler.h"
#include "runtime/config/deploymentConfig.h"
#include "runtime/exec/tensorMap.h"
#include "runtime/hybridCacheManager.h"
#include "runtime/state/pipelineIO.h"
#include "runtime/state/sharedResources.h"

#include <cstdlib>
#include <cstring>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <functional>
#include <limits>
#include <map>
#include <random>
#include <string>
#include <vector>

using namespace trt_edgellm;

// ==================== Fill Helpers ====================

//! Fill a device tensor with random floating-point data, converting to the target dtype.
inline void fillRandomData(rt::Tensor& tensor, float minVal, float maxVal, nvinfer1::DataType dtype, uint64_t seed = 0)
{
    size_t vol = tensor.getShape().volume();
    std::vector<float> hostData(vol);
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dis(minVal, maxVal);

    for (size_t i = 0; i < vol; ++i)
    {
        hostData[i] = dis(gen);
    }

    if (dtype == nvinfer1::DataType::kFLOAT)
    {
        CUDA_CHECK(cudaMemcpy(tensor.rawPointer(), hostData.data(), vol * sizeof(float), cudaMemcpyHostToDevice));
    }
    else if (dtype == nvinfer1::DataType::kHALF)
    {
        std::vector<half> halfData(vol);
        for (size_t i = 0; i < vol; ++i)
            halfData[i] = __float2half(hostData[i]);
        CUDA_CHECK(cudaMemcpy(tensor.rawPointer(), halfData.data(), vol * sizeof(half), cudaMemcpyHostToDevice));
    }
    else if (dtype == nvinfer1::DataType::kBF16)
    {
        std::vector<__nv_bfloat16> bf16Data(vol);
        for (size_t i = 0; i < vol; ++i)
            bf16Data[i] = __float2bfloat16(hostData[i]);
        CUDA_CHECK(
            cudaMemcpy(tensor.rawPointer(), bf16Data.data(), vol * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    }
}

//! Fill a device tensor with a constant int32 value.
inline void fillInt32(rt::Tensor& tensor, int32_t val)
{
    size_t vol = tensor.getShape().volume();
    std::vector<int32_t> hostData(vol, val);
    CUDA_CHECK(cudaMemcpy(tensor.rawPointer(), hostData.data(), vol * sizeof(int32_t), cudaMemcpyHostToDevice));
}

//! Scalar parameters shared by DFlash draft tensor-map allocation and input-fill.
struct DFlashDraftBenchParams
{
    int32_t batchSize;
    int32_t blockSize;
    //! Delta length used by this benchmark run. Serves as `maxDeltaLen` when sizing
    //! tensor allocations in `buildDFlashDraftTensorMap`, and as the runtime `deltaLen`
    //! written into the scratch delta-lengths tensor by `fillDFlashDraftInputs`.
    int32_t deltaLen;
    int32_t pastKVLen; //!< Draft KV cache length before applying `deltaLen` (fill only).
    uint64_t seed;     //!< RNG seed for random tensor fills (fill only).
};

//! Bench-owned tensors that are bound into the DFlash draft TensorMap but are NOT
//! part of production `PipelineIO`. In production, `DFlashDecoder` stores the
//! draft delta lengths as a private runtime member (`mDraftDeltaLens`, see
//! `cpp/runtime/decoding/dflashDecoder.cpp`), so `PipelineIO` intentionally
//! does not carry this field. The bench mirrors that split with this scratch.
struct DFlashDraftBenchScratch
{
    rt::Tensor dflashDeltaLengths;
};

//! Allocate and bind the tensors consumed and produced by a DFlash draft forward.
//!
//! `scratch` is populated by this function and must outlive `tensorMap`. Passing it
//! by reference (rather than returning by value) is deliberate: `TensorMap` stores
//! non-owning pointers into `scratch.dflashDeltaLengths`, so relying on NRVO/copy
//! elision at the return site would leave those pointers dangling whenever the
//! compiler chose not to elide. Caller-owned scope keeps the tensor address stable.
inline void buildDFlashDraftTensorMap(rt::DeploymentConfig const& deployment, DFlashDraftBenchParams const& params,
    rt::HybridCacheManager& draftCacheManager, rt::SharedResources& sharedResources, rt::PipelineIO& io,
    DFlashDraftBenchScratch& scratch, rt::TensorMap& tensorMap)
{
    ELLM_CHECK(deployment.draft.has_value(), "DFlash draft tensor map requires a draft engine config");
    ELLM_CHECK(deployment.specConfig.has_value(), "DFlash draft tensor map requires a spec-decode config");
    ELLM_CHECK(params.batchSize > 0 && params.blockSize > 0 && params.deltaLen > 0,
        "DFlash draft tensor dimensions must all be positive");

    rt::LLMEngineConfig const& draftCfg = *deployment.draft;
    ELLM_CHECK(draftCfg.ropeConfig.type != rt::RopeType::kMRope,
        "llm_bench DFlash draft tensor-map helper does not support mRoPE");

    int32_t const draftHiddenSize = deployment.specConfig->draftHiddenSize;
    int32_t const baseOutputHiddenDim = deployment.specConfig->baseOutputHiddenDim;
    int32_t const draftVocabSize = draftCfg.outputVocabSize;
    int32_t const packedMaskLen = static_cast<int32_t>(divUp(params.blockSize, 32));

    io.inputsEmbeds = rt::Tensor({params.batchSize, params.blockSize, draftHiddenSize}, rt::DeviceType::kGPU,
        nvinfer1::DataType::kHALF, "DFlashBench::inputsEmbeds");
    io.baseHiddenStates = rt::Tensor({params.batchSize, params.deltaLen, baseOutputHiddenDim}, rt::DeviceType::kGPU,
        nvinfer1::DataType::kHALF, "DFlashBench::targetHidden");
    io.outputLogits = rt::Tensor({params.batchSize, params.blockSize, draftVocabSize}, rt::DeviceType::kGPU,
        nvinfer1::DataType::kFLOAT, "DFlashBench::outputLogits");
    io.packedAttentionMask = rt::Tensor({params.batchSize, params.blockSize, packedMaskLen}, rt::DeviceType::kGPU,
        nvinfer1::DataType::kINT32, "DFlashBench::packedAttentionMask");
    io.specDecodePositionIds = rt::Tensor({params.batchSize, params.blockSize}, rt::DeviceType::kGPU,
        nvinfer1::DataType::kINT32, "DFlashBench::positionIds");
    io.contextLengths = rt::Tensor(
        {params.batchSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32, "DFlashBench::contextLengths");

    scratch.dflashDeltaLengths
        = rt::Tensor({params.batchSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32, "DFlashBench::deltaLengths");

    tensorMap.set(binding_names::kInputsEmbeds, io.inputsEmbeds);
    tensorMap.set(binding_names::kDFlashTargetHiddenConcat, io.baseHiddenStates);
    tensorMap.set(binding_names::kLogits, io.outputLogits);
    tensorMap.set(binding_names::kAttentionMask, io.packedAttentionMask);
    tensorMap.set(binding_names::kAttentionPosId, io.specDecodePositionIds);
    tensorMap.set(binding_names::kContextLengths, io.contextLengths);
    tensorMap.set(binding_names::kDFlashDeltaLengths, scratch.dflashDeltaLengths);

    auto& kvManager = draftCacheManager.getKVCacheManager();
    int32_t localAttentionIndex = 0;
    for (auto const layerType : draftCfg.layerTypes)
    {
        if (layerType != rt::HybridCacheManager::LayerType::kAttention)
        {
            continue;
        }
        auto& combinedKV = kvManager.getCombinedKVCache(localAttentionIndex);
        tensorMap.set(binding_names::formatKVCacheName(localAttentionIndex, /*isPast=*/true), combinedKV);
        tensorMap.set(binding_names::formatKVCacheName(localAttentionIndex, /*isPast=*/false), combinedKV);
        ++localAttentionIndex;
    }
    tensorMap.set(binding_names::kKVCacheStartIndex, draftCacheManager.getKVCacheLengths());
    tensorMap.set(binding_names::kRopeCosSin,
        sharedResources.ropePool.getOrCreate(
            draftCfg.ropeConfig, draftCfg.rotaryDim, deployment.base.maxKVCacheCapacity, nullptr));
}

//! Fill DFlash draft inputs and prepare production-equivalent proposal metadata.
//!
//! `params.pastKVLen` is the draft cache length before applying `params.deltaLen`.
//! Consequently, the production preparation kernel writes context lengths equal to
//! `pastKVLen + deltaLen + blockSize`.
inline void fillDFlashDraftInputs(rt::PipelineIO& io, rt::HybridCacheManager& draftCacheManager,
    DFlashDraftBenchScratch& scratch, DFlashDraftBenchParams const& params, cudaStream_t stream)
{
    ELLM_CHECK(params.batchSize > 0 && params.blockSize > 0, "DFlash draft input dimensions must be positive");
    ELLM_CHECK(params.blockSize <= 1024, "DFlash proposal block size must not exceed 1024");
    ELLM_CHECK(
        params.pastKVLen >= 0 && params.deltaLen >= 0, "DFlash draft cache and delta lengths must be non-negative");

    auto const inputsEmbedsShape = io.inputsEmbeds.getShape();
    auto const targetHiddenShape = io.baseHiddenStates.getShape();
    auto const deltaLengthsShape = scratch.dflashDeltaLengths.getShape();
    auto const packedMaskShape = io.packedAttentionMask.getShape();
    auto const positionIdsShape = io.specDecodePositionIds.getShape();
    auto const contextLengthsShape = io.contextLengths.getShape();
    int64_t const packedMaskLen = static_cast<int64_t>(divUp(params.blockSize, 32));

    ELLM_CHECK(inputsEmbedsShape.getNumDims() == 3 && inputsEmbedsShape[0] == params.batchSize
            && inputsEmbedsShape[1] == params.blockSize && inputsEmbedsShape[2] > 0,
        "DFlash inputsEmbeds must be [batchSize, blockSize, draftHiddenSize]");
    ELLM_CHECK(targetHiddenShape.getNumDims() == 3 && targetHiddenShape[0] == params.batchSize
            && targetHiddenShape[1] >= params.deltaLen && targetHiddenShape[2] > 0,
        "DFlash target hidden must be [batchSize, maxDeltaLen, baseOutputHiddenDim] with maxDeltaLen >= deltaLen");
    ELLM_CHECK(deltaLengthsShape == rt::Coords{params.batchSize}, "DFlash delta lengths must be [batchSize]");
    ELLM_CHECK((packedMaskShape == rt::Coords{params.batchSize, params.blockSize, packedMaskLen}),
        "DFlash packed attention mask must be [batchSize, blockSize, divUp(blockSize, 32)]");
    ELLM_CHECK((positionIdsShape == rt::Coords{params.batchSize, params.blockSize}),
        "DFlash attention position IDs must be [batchSize, blockSize]");
    ELLM_CHECK(contextLengthsShape == rt::Coords{params.batchSize}, "DFlash context lengths must be [batchSize]");

    auto const& cacheConfig = draftCacheManager.getKVCacheManager().getConfig();
    ELLM_CHECK(params.batchSize <= cacheConfig.maxBatchSize, "DFlash batch size exceeds the draft KV cache capacity");
    int64_t const preparedContextLength = static_cast<int64_t>(params.pastKVLen) + static_cast<int64_t>(params.deltaLen)
        + static_cast<int64_t>(params.blockSize);
    ELLM_CHECK(preparedContextLength <= std::numeric_limits<int32_t>::max(),
        "DFlash prepared context length exceeds the INT32 range");
    ELLM_CHECK(preparedContextLength <= cacheConfig.maxSequenceLength,
        "DFlash prepared context length exceeds the draft KV cache capacity");

    fillRandomData(io.inputsEmbeds, -1.0F, 1.0F, nvinfer1::DataType::kHALF, params.seed);
    fillRandomData(io.baseHiddenStates, -1.0F, 1.0F, nvinfer1::DataType::kHALF, params.seed + 1);
    fillInt32(scratch.dflashDeltaLengths, params.deltaLen);

    rt::Tensor reuseLengths(
        {params.batchSize}, rt::DeviceType::kCPU, nvinfer1::DataType::kINT32, "DFlashBench::reuseLengths");
    std::vector<int32_t> hostReuseLengths(static_cast<size_t>(params.batchSize), params.pastKVLen);
    std::memcpy(reuseLengths.rawPointer(), hostReuseLengths.data(), hostReuseLengths.size() * sizeof(int32_t));
    draftCacheManager.resetForNewSequences(reuseLengths, stream);

    kernel::launchDFlashPrepareProposalInputs(draftCacheManager.getKVCacheLengths().dataPointer<int32_t>(),
        scratch.dflashDeltaLengths.dataPointer<int32_t>(), params.blockSize,
        io.packedAttentionMask.dataPointer<int32_t>(), io.specDecodePositionIds.dataPointer<int32_t>(),
        io.contextLengths.dataPointer<int32_t>(), false, params.batchSize, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

//! Owning input, output, and workspace tensors for one DDTree build benchmark.
struct DDTreeBuildScratch
{
    rt::Tensor draftLogits;
    rt::Tensor lastAcceptedTokens;
    rt::Tensor baseKVCacheLengths;

    rt::Tensor treeTokenIds;
    rt::Tensor treeDepths;
    rt::Tensor treeParentIds;
    rt::Tensor treeNodeScores;
    rt::Tensor validCounts;
    rt::Tensor verifyTokenIds;
    rt::Tensor specDecodePositionIds;
    rt::Tensor packedAttentionMask;
    rt::Tensor verifyTreeMask;
    rt::Tensor contextLengths;
    rt::Tensor selectTokenIndices;
    rt::Tensor workspace;
};

//! Allocate all DDTree build inputs, outputs, and the exact production workspace size.
inline DDTreeBuildScratch allocateDDTreeBuildScratch(
    int32_t batchSize, int32_t blockSize, int32_t draftVocabSize, int32_t verifySize, int32_t candidateTopK)
{
    ELLM_CHECK(batchSize > 0 && blockSize > 1 && draftVocabSize > 0 && verifySize > 0 && candidateTopK > 0,
        "DDTree build requires positive dimensions and candidateTopK, with blockSize > 1");
    ELLM_CHECK(verifySize <= kernel::kDDTreeMaxVerifySize,
        "DDTree verifySize exceeds the production kernel limit of " + std::to_string(kernel::kDDTreeMaxVerifySize));
    ELLM_CHECK(candidateTopK <= kernel::kDDTreeMaxCandidateTopK,
        "DDTree candidateTopK exceeds the production kernel limit of "
            + std::to_string(kernel::kDDTreeMaxCandidateTopK));
    ELLM_CHECK(candidateTopK <= draftVocabSize, "DDTree candidateTopK must not exceed draftVocabSize");

    DDTreeBuildScratch scratch;
    scratch.draftLogits = rt::Tensor({batchSize, blockSize, draftVocabSize}, rt::DeviceType::kGPU,
        nvinfer1::DataType::kFLOAT, "DFlashBench::ddtreeDraftLogits");
    scratch.lastAcceptedTokens
        = rt::Tensor({batchSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32, "DFlashBench::lastAcceptedTokens");
    scratch.baseKVCacheLengths
        = rt::Tensor({batchSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32, "DFlashBench::baseKVCacheLengths");

    rt::Coords const nodeShape{batchSize, verifySize};
    scratch.treeTokenIds
        = rt::Tensor(nodeShape, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32, "DFlashBench::treeTokenIds");
    scratch.treeDepths
        = rt::Tensor(nodeShape, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32, "DFlashBench::treeDepths");
    scratch.treeParentIds
        = rt::Tensor(nodeShape, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32, "DFlashBench::treeParentIds");
    scratch.treeNodeScores
        = rt::Tensor(nodeShape, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT, "DFlashBench::treeNodeScores");
    scratch.validCounts
        = rt::Tensor({batchSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32, "DFlashBench::validCounts");
    scratch.verifyTokenIds
        = rt::Tensor(nodeShape, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32, "DFlashBench::verifyTokenIds");
    scratch.specDecodePositionIds
        = rt::Tensor(nodeShape, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32, "DFlashBench::specDecodePositionIds");
    scratch.packedAttentionMask = rt::Tensor({batchSize, verifySize, static_cast<int64_t>(divUp(verifySize, 32))},
        rt::DeviceType::kGPU, nvinfer1::DataType::kINT32, "DFlashBench::packedAttentionMask");
    scratch.verifyTreeMask = rt::Tensor({batchSize, verifySize, verifySize}, rt::DeviceType::kGPU,
        nvinfer1::DataType::kINT8, "DFlashBench::verifyTreeMask");
    scratch.contextLengths
        = rt::Tensor({batchSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32, "DFlashBench::contextLengths");
    scratch.selectTokenIndices
        = rt::Tensor(nodeShape, rt::DeviceType::kGPU, nvinfer1::DataType::kINT64, "DFlashBench::selectTokenIndices");

    size_t const workspaceSize
        = kernel::getDDTreeBuildWorkspaceSize(batchSize, blockSize, verifySize, draftVocabSize, candidateTopK);
    ELLM_CHECK(workspaceSize > 0, "DDTree build workspace size must be positive");
    scratch.workspace = rt::Tensor({static_cast<int64_t>(workspaceSize)}, rt::DeviceType::kGPU,
        nvinfer1::DataType::kUINT8, "DFlashBench::ddtreeWorkspace");
    return scratch;
}

//! Fill a device tensor with a constant int8 value.
inline void fillInt8(rt::Tensor& tensor, int8_t val)
{
    size_t vol = tensor.getShape().volume();
    std::vector<int8_t> hostData(vol, val);
    CUDA_CHECK(cudaMemcpy(tensor.rawPointer(), hostData.data(), vol * sizeof(int8_t), cudaMemcpyHostToDevice));
}

// ==================== Run Loop Templates ====================

//! Run warmup iterations with layer profiling disabled.
template <typename ResetFn, typename StepFn>
int runWarmupLoop(std::string const& modeName, int32_t warmupCount, ResetFn const& resetState, StepFn const& step,
    cudaStream_t stream)
{
    LOG_INFO("=== %s Warmup (%d iterations) ===", modeName.c_str(), warmupCount);
    layerProfiler::disableLayerProfilers();

    for (int i = 0; i < warmupCount; ++i)
    {
        resetState();
        if (!step())
        {
            LOG_ERROR("%s warmup iteration %d failed", modeName.c_str(), i);
            return EXIT_FAILURE;
        }
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return EXIT_SUCCESS;
}

//! Run layer profiling loop: collects per-layer timing and kernel breakdown per iteration.
template <typename ResetFn, typename StepFn>
int runLayerProfilingLoop(std::string const& modeName, int32_t iterations, bool accumulateForCsv,
    ResetFn const& resetState, StepFn const& step, std::map<std::string, LayerMetadata> const& layerMetadata,
    std::vector<KernelTimes>& timesPerIter, OrderedLayerTimings& layerTimings, cudaStream_t stream)
{
    LOG_INFO("=== %s Layer Profiling (non-CUDA-graph, %d iterations) ===", modeName.c_str(), iterations);

    for (int iter = 0; iter < iterations; ++iter)
    {
        resetState();
        layerProfiler::LayerProfiler::getInstance().reset();
        layerProfiler::enableLayerProfilers();

        if (!step())
        {
            LOG_ERROR("%s iteration %d failed", modeName.c_str(), iter);
            return EXIT_FAILURE;
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));

        layerProfiler::disableLayerProfilers();

        auto metrics = layerProfiler::LayerProfiler::getInstance().getMetrics();
        KernelTimes times = extractKernelTimes(metrics, layerMetadata);
        timesPerIter.push_back(times);

        if (accumulateForCsv)
        {
            accumulateLayerTimings(metrics, layerTimings);
        }
    }

    return EXIT_SUCCESS;
}

//! Run E2E timing with repeated single-step iterations. Optionally captures a CUDA graph first.
template <typename ResetFn, typename StepFn>
float runRepeatedE2ETiming(
    std::string const& modeName, int32_t iterations, ResetFn const& resetState, StepFn const& step, cudaStream_t stream,
    bool useCudaGraph = false, std::function<bool()> const& captureGraph = []() { return false; })
{
    if (useCudaGraph)
    {
        LOG_INFO("=== Capturing CUDA Graph for %s ===", modeName.c_str());
        resetState();
        if (captureGraph())
        {
            LOG_INFO("CUDA graph captured successfully.");
        }
        else
        {
            LOG_WARNING("Failed to capture CUDA graph, falling back to non-graph execution.");
        }
    }

    LOG_INFO("=== %s E2E Timing (%d iterations) ===", modeName.c_str(), iterations);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    std::vector<float> e2eTimes;
    for (int iter = 0; iter < iterations; ++iter)
    {
        resetState();

        CUDA_CHECK(cudaEventRecord(start, stream));
        if (!step())
        {
            LOG_ERROR("%s E2E iteration %d failed", modeName.c_str(), iter);
            CUDA_CHECK(cudaEventDestroy(start));
            CUDA_CHECK(cudaEventDestroy(stop));
            return -1.0f;
        }
        CUDA_CHECK(cudaEventRecord(stop, stream));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        e2eTimes.push_back(ms);
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    std::vector<double> e2eTimesDouble(e2eTimes.begin(), e2eTimes.end());
    auto [mean, std] = computeStats(e2eTimesDouble);
    LOG_INFO("%s E2E Time: %.4f +/- %.4f ms", modeName.c_str(), mean, std);
    return static_cast<float>(mean);
}

//! Run sequential E2E timing: runs decodeSteps in a single timed block. Used for osl>1 decode.
template <typename ResetFn, typename StepFn, typename PostStepFn, typename CaptureGraphFn>
float runSequentialE2ETiming(std::string const& modeName, int32_t decodeSteps, ResetFn const& resetState,
    StepFn const& step, PostStepFn const& postStep, bool useCudaGraph, CaptureGraphFn const& captureGraph,
    cudaStream_t stream)
{
    if (useCudaGraph)
    {
        LOG_INFO("=== Capturing CUDA Graph for %s ===", modeName.c_str());
        resetState();
        if (captureGraph())
        {
            LOG_INFO("CUDA graph captured successfully.");
        }
        else
        {
            LOG_WARNING("Failed to capture CUDA graph, falling back to non-graph execution.");
        }
    }

    LOG_INFO("=== %s E2E Timing (steps=%d) ===", modeName.c_str(), decodeSteps);

    resetState();

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start, stream));
    for (int32_t t = 0; t < decodeSteps; ++t)
    {
        if (!step())
        {
            LOG_ERROR("%s E2E step %d/%d failed", modeName.c_str(), t, decodeSteps);
            CUDA_CHECK(cudaEventDestroy(start));
            CUDA_CHECK(cudaEventDestroy(stop));
            return -1.0f;
        }
        postStep(t);
    }
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    float totalTimeMs = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&totalTimeMs, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    LOG_INFO("E2E Time: %.3f ms (steps=%d)", totalTimeMs, decodeSteps);
    LOG_INFO("Per-step avg: %.3f ms", totalTimeMs / decodeSteps);
    LOG_INFO("Throughput: %.2f tokens/sec", 1000.0f * decodeSteps / totalTimeMs);

    return totalTimeMs;
}
