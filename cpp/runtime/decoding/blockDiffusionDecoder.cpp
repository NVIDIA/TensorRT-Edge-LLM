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

#include "runtime/decoding/blockDiffusionDecoder.h"

#include "common/bindingNames.h"
#include "common/checkMacros.h"
#include "common/cudaUtils.h"
#include "common/logger.h"
#include "kernels/embeddingKernels/embeddingKernels.h"
#include "profiling/metrics.h"
#include "profiling/timer.h"
#include "runtime/decoding/decoderUtils.h"
#include "runtime/state/pipelineIO.h"
#include "sampler/diffusionGemma/diffusionGemmaSampling.h"
#include "sampler/sampling.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cuda_fp16.h>
#include <filesystem>
#include <functional>
#include <optional>
#include <vector>

namespace trt_edgellm
{
namespace rt
{
namespace
{
constexpr int32_t kDecodeProfile{1};
constexpr int32_t kDefaultRuntimeDenoisingSteps{16};
} // namespace

BlockDiffusionDecoder::BlockDiffusionDecoder(
    DecodingRuntimeContext& runtime, std::filesystem::path const&, cudaStream_t stream)
    : mRuntime(runtime)
    , mCanvasLen(std::max(1, runtime.deployment.base.diffusionCanvasLength))
    , mMaxConditioningSeqLen(std::max(mCanvasLen, runtime.deployment.base.maxSupportedInputLength))
    , mMaxDenoisingSteps(std::max(1, runtime.deployment.base.diffusionMaxDenoisingSteps))
{
    if (runtime.deployment.base.diffusionSelfConditioningSize > 0)
    {
        ELLM_CHECK(runtime.deployment.base.diffusionUnifiedConditioning,
            "DiffusionGemma requires unified self-conditioning in dllm.engine. Re-export the model with the latest "
            "DiffusionGemma exporter and rebuild the engine.");
        LOG_INFO("DiffusionGemma self-conditioning is unified into dllm.engine.");
    }

    int32_t const maxBatch = runtime.maxRuntimeBatchSize;
    mCanvasIds = Tensor({maxBatch, mMaxConditioningSeqLen}, DeviceType::kGPU, nvinfer1::DataType::kINT32,
        "BlockDiffusionDecoder::canvasIds");
    mArgmaxCanvasIds = Tensor(
        {maxBatch, mCanvasLen}, DeviceType::kGPU, nvinfer1::DataType::kINT32, "BlockDiffusionDecoder::argmaxCanvasIds");
    mSampledCanvasIds = Tensor({maxBatch, mCanvasLen}, DeviceType::kGPU, nvinfer1::DataType::kINT32,
        "BlockDiffusionDecoder::sampledCanvasIds");
    mCommitCanvasIds = Tensor(
        {maxBatch, mCanvasLen}, DeviceType::kGPU, nvinfer1::DataType::kINT32, "BlockDiffusionDecoder::commitCanvasIds");
    mCommitLengths
        = Tensor({maxBatch}, DeviceType::kGPU, nvinfer1::DataType::kINT32, "BlockDiffusionDecoder::commitLengths");
    mPreviousArgmaxIds = Tensor({maxBatch, mCanvasLen}, DeviceType::kGPU, nvinfer1::DataType::kINT32,
        "BlockDiffusionDecoder::previousArgmaxIds");
    mStableCounts = Tensor(
        {maxBatch, mCanvasLen}, DeviceType::kGPU, nvinfer1::DataType::kINT32, "BlockDiffusionDecoder::stableCounts");
    mAcceptedMask = Tensor(
        {maxBatch, mCanvasLen}, DeviceType::kGPU, nvinfer1::DataType::kINT8, "BlockDiffusionDecoder::acceptedMask");
    mPrefixLengths
        = Tensor({maxBatch}, DeviceType::kGPU, nvinfer1::DataType::kINT32, "BlockDiffusionDecoder::prefixLengths");
    mHostPrefixLengths
        = Tensor({maxBatch}, DeviceType::kCPU, nvinfer1::DataType::kINT32, "BlockDiffusionDecoder::hostPrefixLengths");
    mSelfConditioningEmbedsA = Tensor({maxBatch, mMaxConditioningSeqLen, runtime.deployment.base.hiddenSize},
        DeviceType::kGPU, nvinfer1::DataType::kHALF, "BlockDiffusionDecoder::selfConditioningEmbedsA");
    mSelfConditioningEmbedsB = Tensor({maxBatch, mMaxConditioningSeqLen, runtime.deployment.base.hiddenSize},
        DeviceType::kGPU, nvinfer1::DataType::kHALF, "BlockDiffusionDecoder::selfConditioningEmbedsB");
    mSelfConditioningTemperature = Tensor(
        {1}, DeviceType::kGPU, nvinfer1::DataType::kFLOAT, "BlockDiffusionDecoder::selfConditioningTemperature");
    mHostSelfConditioningTemperature = Tensor(
        {1}, DeviceType::kCPU, nvinfer1::DataType::kFLOAT, "BlockDiffusionDecoder::hostSelfConditioningTemperature");

    mHostSelfConditioningTemperature.dataPointer<float>()[0] = denoiseTemperature(0);
    mCommittedLengthsScratch.resize(maxBatch);
    mRemainingLengthsScratch.resize(maxBatch);
    mValidCanvasLengthsScratch.resize(maxBatch);
    mCommitLengthsScratch.resize(maxBatch);
    int32_t const maxPhysicalTokens = maxBatch * mMaxConditioningSeqLen;
    mCanvasMetadataScratch.positions.reserve(maxPhysicalTokens);
    mCanvasMetadataScratch.queryStartOffsets.reserve(maxBatch + 1);
    mCanvasMetadataScratch.queryLengths.reserve(maxBatch);
    mCanvasMetadataScratch.pastLengths.reserve(maxBatch);
    mCanvasMetadataScratch.attentionSequenceLengths.reserve(maxBatch);
    mCanvasMetadataScratch.stateIndices.reserve(maxBatch);
    mCanvasMetadataScratch.logitsIndices.reserve(maxPhysicalTokens);
    CUDA_CHECK(cudaMemcpyAsync(mSelfConditioningTemperature.rawPointer(), mHostSelfConditioningTemperature.rawPointer(),
        sizeof(float), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemsetAsync(mCanvasIds.rawPointer(), 0, mCanvasIds.getMemoryCapacity(), stream));
    CUDA_CHECK(cudaMemsetAsync(
        mSelfConditioningEmbedsA.rawPointer(), 0, mSelfConditioningEmbedsA.getMemoryCapacity(), stream));
    CUDA_CHECK(cudaMemsetAsync(
        mSelfConditioningEmbedsB.rawPointer(), 0, mSelfConditioningEmbedsB.getMemoryCapacity(), stream));
    bindUnifiedBackboneTensors();
}

bool BlockDiffusionDecoder::initializeCanvas(int32_t batchSize, int32_t canvasLen, cudaStream_t stream)
{
    check::check(mRuntime.sampling.hostPackedTokenIds.reshape({batchSize, canvasLen}), "Tensor reshape failed");
    check::check(mCanvasIds.reshape({batchSize, canvasLen}), "Tensor reshape failed");
    check::check(mArgmaxCanvasIds.reshape({batchSize, canvasLen}), "Tensor reshape failed");
    check::check(mSampledCanvasIds.reshape({batchSize, canvasLen}), "Tensor reshape failed");
    check::check(mPreviousArgmaxIds.reshape({batchSize, canvasLen}), "Tensor reshape failed");
    check::check(mStableCounts.reshape({batchSize, canvasLen}), "Tensor reshape failed");
    check::check(mAcceptedMask.reshape({batchSize, canvasLen}), "Tensor reshape failed");
    check::check(mPrefixLengths.reshape({batchSize}), "Tensor reshape failed");
    DiffusionCanvasInitParams initParams{};
    initParams.vocabSize = mRuntime.deployment.base.vocabSize;
    initParams.random.offset = mRandomOffset;
    initializeDiffusionCanvas(
        mCanvasIds, mPreviousArgmaxIds, mStableCounts, mAcceptedMask, mPrefixLengths, initParams, stream);
    mRandomOffset += static_cast<uint64_t>(batchSize) * static_cast<uint64_t>(canvasLen);
    return true;
}

bool BlockDiffusionDecoder::prepareCanvasMetadata(int32_t batchSize, int32_t canvasLen, InferenceDims const& dims,
    cudaStream_t stream, std::vector<int32_t> const* queryLengths, std::vector<int32_t> const* pastLengths,
    std::vector<ResidentRef> const* residentRefs)
{
    ExecutionPhase const phase = executionPhase(dims);
    check::check(phase == ExecutionPhase::kDiffusionDenoise || phase == ExecutionPhase::kDiffusionCommit,
        "Diffusion metadata requires a diffusion execution phase");
    bool const denoisePhase = phase == ExecutionPhase::kDiffusionDenoise;
    PipelineIO& io = mRuntime.base.pipelineIO;
    RaggedExecutionBatch& batch = mCanvasMetadataScratch;
    int32_t const physicalTokens = batchSize * canvasLen;
    batch.shape = {batchSize, physicalTokens, physicalTokens, canvasLen, 0, 0, physicalTokens};
    batch.positions.resize(physicalTokens);
    batch.queryStartOffsets.resize(batchSize + 1);
    batch.queryLengths.resize(batchSize);
    batch.pastLengths.resize(batchSize);
    batch.attentionSequenceLengths.resize(batchSize);
    batch.stateIndices.resize(batchSize);
    batch.logitsIndices.resize(physicalTokens);
    check::check(io.hostContextLengths.reshape({batchSize}), "Tensor reshape failed");
    int32_t* hostCtx = io.hostContextLengths.dataPointer<int32_t>();
    for (int32_t b = 0; b < batchSize; ++b)
    {
        int32_t const queryLen = queryLengths != nullptr ? (*queryLengths)[b] : canvasLen;
        int32_t const pastLen = pastLengths != nullptr ? (*pastLengths)[b] : 0;
        check::check(queryLen > 0 && queryLen <= canvasLen, "DiffusionGemma query length must be in (0, canvasLen].");
        hostCtx[b] = queryLen;
        batch.queryStartOffsets[b] = b * canvasLen;
        batch.queryLengths[b] = queryLen;
        batch.pastLengths[b] = pastLen;
        batch.attentionSequenceLengths[b] = pastLen + queryLen;
        batch.stateIndices[b] = residentRefs == nullptr ? b : residentRefs->at(static_cast<size_t>(b)).slot;
        for (int32_t i = 0; i < canvasLen; ++i)
        {
            int32_t const row = b * canvasLen + i;
            bool const valid = i < queryLen;
            batch.positions[row] = valid ? pastLen + i : -1;
            batch.logitsIndices[row] = row;
        }
    }
    batch.queryStartOffsets[batchSize] = physicalTokens;
    io.uploadRaggedMetadata(batch, stream);
    prepareRaggedKVPageTable(io, *mRuntime.base.sharedResources.kvPageTables[0], batchSize, stream);
    prepareRaggedRope(io, mRuntime.base.sharedResources, mRuntime.deployment.base, physicalTokens, batchSize, stream);
    check::check(io.contextLengths.reshape({batchSize}), "Tensor reshape failed");
    CUDA_CHECK(cudaMemcpyAsync(
        io.contextLengths.rawPointer(), hostCtx, batchSize * sizeof(int32_t), cudaMemcpyHostToDevice, stream));

    int32_t const selectLen = canvasLen;
    check::check(io.selectTokenIndices.reshape({batchSize * selectLen}), "Tensor reshape failed");
    check::check(io.hostSelectTokenIndices.reshape({batchSize * selectLen}), "Tensor reshape failed");
    int64_t* hostSelect = io.hostSelectTokenIndices.dataPointer<int64_t>();
    for (int32_t b = 0; b < batchSize; ++b)
    {
        for (int32_t i = 0; i < selectLen; ++i)
        {
            hostSelect[b * selectLen + i] = static_cast<int64_t>(b) * selectLen + i;
        }
    }
    CUDA_CHECK(cudaMemcpyAsync(io.selectTokenIndices.rawPointer(), hostSelect, batchSize * selectLen * sizeof(int64_t),
        cudaMemcpyHostToDevice, stream));

    check::check(io.phaseIsEncoder.reshape({1}), "Tensor reshape failed");
    check::check(io.hostPhaseIsEncoder.reshape({1}), "Tensor reshape failed");
    int32_t* hostPhase = io.hostPhaseIsEncoder.dataPointer<int32_t>();
    int32_t const phaseValue = denoisePhase ? 0 : 1;
    hostPhase[0] = phaseValue;
    CUDA_CHECK(
        cudaMemcpyAsync(io.phaseIsEncoder.rawPointer(), hostPhase, sizeof(int32_t), cudaMemcpyHostToDevice, stream));

    int32_t const contextMaskSelectorLen = denoisePhase ? batchSize : 0;
    check::check(io.contextMaskSelector.reshape({contextMaskSelectorLen}), "Tensor reshape failed");
    if (contextMaskSelectorLen > 0)
    {
        CUDA_CHECK(cudaMemsetAsync(io.contextMaskSelector.rawPointer(), 0,
            static_cast<size_t>(contextMaskSelectorLen) * sizeof(int32_t), stream));
    }
    io.recordStepHostUploads(stream);
    return true;
}

bool BlockDiffusionDecoder::updateSelfConditioningTemperature(float temperature, cudaStream_t stream)
{
    mRuntime.base.pipelineIO.waitForStepHostStaging();
    check::check(mSelfConditioningTemperature.reshape({1}), "Tensor reshape failed");
    check::check(mHostSelfConditioningTemperature.reshape({1}), "Tensor reshape failed");
    mHostSelfConditioningTemperature.dataPointer<float>()[0] = temperature;
    CUDA_CHECK(cudaMemcpyAsync(mSelfConditioningTemperature.rawPointer(), mHostSelfConditioningTemperature.rawPointer(),
        sizeof(float), cudaMemcpyHostToDevice, stream));
    mRuntime.base.pipelineIO.recordStepHostUploads(stream);
    return true;
}

void BlockDiffusionDecoder::bindUnifiedBackboneTensors()
{
    PipelineIO& io = mRuntime.base.pipelineIO;
    bindDiffusionUnifiedBackboneTensors(mRuntime.base.tensorMap, io, io.outputLogits, mCanvasIds,
        mSelfConditioningEmbedsA, mSelfConditioningEmbedsB, mSelfConditioningTemperature);
    mCurrentDenoiseLogits = &io.outputLogits;
}

void BlockDiffusionDecoder::bindDefaultSelfConditioningTensors()
{
    bindDiffusionUnifiedBackboneSelfConditioningTensors(
        mRuntime.base.tensorMap, mSelfConditioningEmbedsA, mSelfConditioningEmbedsB);
    mCurrentDenoiseLogits = &mRuntime.base.pipelineIO.outputLogits;
}

Tensor& BlockDiffusionDecoder::currentDenoiseLogits() noexcept
{
    if (mCurrentDenoiseLogits != nullptr)
    {
        return *mCurrentDenoiseLogits;
    }
    return mRuntime.base.pipelineIO.outputLogits;
}

bool BlockDiffusionDecoder::prepareUnifiedConditioning(
    int32_t batchSize, int32_t canvasLen, int32_t step, float temperature, cudaStream_t stream)
{
    PipelineIO& io = mRuntime.base.pipelineIO;
    Tensor& prevSelfConditioningEmbeds = (step % 2 == 0) ? mSelfConditioningEmbedsA : mSelfConditioningEmbedsB;
    Tensor& nextSelfConditioningEmbeds = (step % 2 == 0) ? mSelfConditioningEmbedsB : mSelfConditioningEmbedsA;

    check::check(mCanvasIds.reshape({batchSize * canvasLen}), "Tensor reshape failed");
    check::check(
        io.inputsEmbeds.reshape({batchSize * canvasLen, mRuntime.deployment.base.hiddenSize}), "Tensor reshape failed");
    check::check(prevSelfConditioningEmbeds.reshape({batchSize * canvasLen, mRuntime.deployment.base.hiddenSize}),
        "Tensor reshape failed");
    check::check(nextSelfConditioningEmbeds.reshape({batchSize * canvasLen, mRuntime.deployment.base.hiddenSize}),
        "Tensor reshape failed");
    check::check(io.outputLogits.reshape({batchSize * canvasLen, mRuntime.deployment.base.outputVocabSize}),
        "Tensor reshape failed");
    if (step == 0)
    {
        size_t const feedbackBytes = static_cast<size_t>(batchSize) * static_cast<size_t>(canvasLen)
            * static_cast<size_t>(mRuntime.deployment.base.hiddenSize) * sizeof(__half);
        CUDA_CHECK(cudaMemsetAsync(prevSelfConditioningEmbeds.rawPointer(), 0, feedbackBytes, stream));
    }
    if (!updateSelfConditioningTemperature(temperature, stream))
    {
        return false;
    }

    bindDiffusionUnifiedBackboneSelfConditioningTensors(
        mRuntime.base.tensorMap, prevSelfConditioningEmbeds, nextSelfConditioningEmbeds);
    mCurrentDenoiseLogits = &io.outputLogits;
    return true;
}

bool BlockDiffusionDecoder::runDenoiseStep(DenoiseStepParams const& params)
{
    check::check(params.validCanvasLengths != nullptr, "DiffusionGemma valid canvas lengths must be provided.");
    if (!prepareUnifiedConditioning(
            params.batchSize, params.canvasLen, params.step, params.selfConditioningTemperature, params.stream))
    {
        return false;
    }
    if (mRuntime.preprocess.gemma4Ple)
    {
        mRuntime.preprocess.gemma4Ple->embed(mCanvasIds, params.stream);
    }
    Tensor& denoiseLogits = currentDenoiseLogits();
    check::check(denoiseLogits.reshape({params.batchSize * params.canvasLen, mRuntime.deployment.base.outputVocabSize}),
        "Tensor reshape failed");
    auto const denoiseDims = mRuntime.deployment.base.denoiseDims(params.batchSize, params.canvasLen);
    if (!prepareCanvasMetadata(params.batchSize, params.canvasLen, denoiseDims, params.stream,
            params.validCanvasLengths, params.committedLengths, params.residentRefs))
    {
        return false;
    }

    {
        TIME_STAGE(metrics::StageNames::kBLOCK_DIFFUSION_DENOISE, params.stream);
        bool status
            = mRuntime.base.executor.prepare(kDecodeProfile, denoiseDims, mRuntime.base.tensorMap, params.stream);
        if (status)
        {
            status = mRuntime.base.executor.execute(params.stream);
        }
        if (!status)
        {
            LOG_ERROR("Failed to execute DiffusionGemma denoise backbone step.");
            return false;
        }
    }
    return true;
}

int32_t BlockDiffusionDecoder::effectiveMaxDenoisingSteps(DecodingInferenceContext const& context) const noexcept
{
    if (context.diffusionMaxDenoisingSteps <= 0)
    {
        return std::min(mMaxDenoisingSteps, kDefaultRuntimeDenoisingSteps);
    }
    return std::clamp(context.diffusionMaxDenoisingSteps, 1, mMaxDenoisingSteps);
}

float BlockDiffusionDecoder::denoiseTemperature(int32_t step, int32_t maxDenoisingSteps) const noexcept
{
    auto const& cfg = mRuntime.deployment.base;
    if (maxDenoisingSteps <= 0)
    {
        return cfg.diffusionTMin;
    }
    int32_t const clampedStep = std::clamp(step, 0, maxDenoisingSteps - 1);
    float const remaining = static_cast<float>(maxDenoisingSteps - clampedStep);
    float const fraction = remaining / static_cast<float>(maxDenoisingSteps);
    return cfg.diffusionTMin + (cfg.diffusionTMax - cfg.diffusionTMin) * fraction;
}

float BlockDiffusionDecoder::denoiseTemperature(int32_t step) const noexcept
{
    return denoiseTemperature(step, mMaxDenoisingSteps);
}

bool BlockDiffusionDecoder::copyCanvasStateToHost(int32_t batchSize, int32_t canvasLen, cudaStream_t stream)
{
    check::check(mRuntime.sampling.hostPackedTokenIds.reshape({batchSize, canvasLen}), "Tensor reshape failed");
    check::check(mHostPrefixLengths.reshape({batchSize}), "Tensor reshape failed");
    CUDA_CHECK(cudaMemcpyAsync(mRuntime.sampling.hostPackedTokenIds.rawPointer(), mArgmaxCanvasIds.rawPointer(),
        static_cast<size_t>(batchSize) * canvasLen * sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(mHostPrefixLengths.rawPointer(), mPrefixLengths.rawPointer(),
        static_cast<size_t>(batchSize) * sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return true;
}

bool BlockDiffusionDecoder::sampleCanvasEntropyBound(
    int32_t batchSize, int32_t canvasLen, int32_t step, int32_t maxDenoisingSteps, cudaStream_t stream)
{
    PipelineIO& io = mRuntime.base.pipelineIO;
    Tensor& denoiseLogits = currentDenoiseLogits();
    int32_t const rows = batchSize * canvasLen;
    check::check(denoiseLogits.reshape({rows, mRuntime.deployment.base.outputVocabSize}), "Tensor reshape failed");
    check::check(mRuntime.sampling.indices.reshape({rows, 1}), "Tensor reshape failed");
    check::check(mRuntime.sampling.scores.reshape({rows}), "Tensor reshape failed");
    float const temperature = denoiseTemperature(step, maxDenoisingSteps);
    check::check(mCanvasIds.reshape({batchSize, canvasLen}), "Tensor reshape failed");
    check::check(mSampledCanvasIds.reshape({batchSize, canvasLen}), "Tensor reshape failed");
    bool const forceAccept = (step + 1) >= maxDenoisingSteps;
    if (forceAccept)
    {
        selectDiffusionArgmaxFromLogits(denoiseLogits, mRuntime.sampling.indices, temperature, stream);
    }
    else
    {
        DiffusionRandomParams sampleRandomParams{};
        sampleRandomParams.offset = mRandomOffset;
        sampleDiffusionTokensAndComputeEntropy(denoiseLogits, mSampledCanvasIds, mRuntime.sampling.indices,
            mRuntime.sampling.scores, temperature, stream, sampleRandomParams);
    }
    uint64_t const renoiseRandomOffset = mRandomOffset + static_cast<uint64_t>(rows);
    if (mRuntime.deployment.base.reducedVocabSize > 0)
    {
        mapReducedVocabToFullVocab(mRuntime.sampling.indices, mRuntime.sampling.baseVocabMappingTable, stream);
        if (!forceAccept)
        {
            mapReducedVocabToFullVocab(mSampledCanvasIds, mRuntime.sampling.baseVocabMappingTable, stream);
        }
    }

    auto const& cfg = mRuntime.deployment.base;
    DiffusionCanvasUpdateParams updateParams{};
    updateParams.entropyThreshold = cfg.diffusionEntropyThreshold;
    updateParams.entropyBound = cfg.diffusionEntropyBound;
    updateParams.stabilityWindow = cfg.diffusionStabilityWindow;
    updateParams.forceAccept = forceAccept;
    updateParams.vocabSize = cfg.vocabSize;
    updateParams.random.offset = renoiseRandomOffset;
    diffusionSampleAndUpdateCanvas(mSampledCanvasIds, mRuntime.sampling.indices, mRuntime.sampling.scores, mCanvasIds,
        mArgmaxCanvasIds, mPreviousArgmaxIds, mStableCounts, mAcceptedMask, mPrefixLengths, updateParams, stream,
        OptionalInputTensor{std::cref(io.contextLengths)});
    mRandomOffset += static_cast<uint64_t>(rows) * 2U;

    return true;
}

void BlockDiffusionDecoder::fillCommittedLengths(
    DecodingInferenceContext const& context, std::vector<int32_t>& committedLengths) const
{
    int32_t const batchSize = context.activeBatchSize;
    committedLengths.resize(std::max(0, batchSize));
    check::check(static_cast<int32_t>(context.effectivePrefillLengths.size()) >= batchSize,
        "DiffusionGemma context effectivePrefillLengths is smaller than active batch size.");
    check::check(static_cast<int32_t>(context.currentGenerateLengths.size()) >= batchSize,
        "DiffusionGemma context currentGenerateLengths is smaller than active batch size.");

    for (int32_t b = 0; b < batchSize; ++b)
    {
        committedLengths[b] = context.effectivePrefillLengths[b] + context.currentGenerateLengths[b];
    }
}

bool BlockDiffusionDecoder::compactCommitCanvas(int32_t batchSize, int32_t canvasLen, int32_t maxBlockLen,
    std::vector<int32_t> const& commitLengths, cudaStream_t stream)
{
    check::check(batchSize > 0, "DiffusionGemma commit batch size must be positive.");
    check::check(static_cast<int32_t>(commitLengths.size()) == batchSize,
        "DiffusionGemma commit length vector size must match batch size.");
    check::check(maxBlockLen > 0 && maxBlockLen <= canvasLen, "DiffusionGemma commit block length is out of range.");
    for (int32_t length : commitLengths)
    {
        check::check(
            length > 0 && length <= maxBlockLen, "DiffusionGemma per-batch commit length must be in (0, maxBlockLen].");
    }

    check::check(mCommitCanvasIds.reshape({batchSize, maxBlockLen}), "Tensor reshape failed");
    check::check(mCommitLengths.reshape({batchSize}), "Tensor reshape failed");
    CUDA_CHECK(cudaMemcpyAsync(mCommitLengths.rawPointer(), commitLengths.data(),
        static_cast<size_t>(batchSize) * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
    int32_t padTokenId = mRuntime.tokenizer.getPadId();
    if (padTokenId < 0)
    {
        padTokenId = std::max<int32_t>(mRuntime.tokenizer.getEosId(), 0);
    }
    compactDiffusionCanvas(
        mArgmaxCanvasIds, mCommitLengths, mCommitCanvasIds, batchSize, canvasLen, maxBlockLen, padTokenId, stream);
    return true;
}

bool BlockDiffusionDecoder::commitBlock(
    DecodingInferenceContext& context, std::vector<int32_t> const& commitLengths, int32_t canvasLen)
{
    int32_t const batchSize = context.activeBatchSize;
    int32_t const commitSeqLen = canvasLen;
    PipelineIO& io = mRuntime.base.pipelineIO;
    if (!compactCommitCanvas(batchSize, canvasLen, commitSeqLen, commitLengths, context.stream))
    {
        return false;
    }
    check::check(mCanvasIds.reshape({batchSize * commitSeqLen}), "Tensor reshape failed");

    check::check(io.inputsEmbeds.reshape({batchSize * commitSeqLen, mRuntime.deployment.base.hiddenSize}),
        "Tensor reshape failed");
    kernel::embeddingLookup(mCommitCanvasIds, mRuntime.preprocess.embedding.table,
        mRuntime.preprocess.embedding.scalesAsOptional(), io.inputsEmbeds, context.stream);
    if (mRuntime.preprocess.gemma4Ple)
    {
        mRuntime.preprocess.gemma4Ple->embed(mCommitCanvasIds, context.stream);
    }
    bindDefaultSelfConditioningTensors();
    check::check(mSelfConditioningEmbedsA.reshape({batchSize * commitSeqLen, mRuntime.deployment.base.hiddenSize}),
        "Tensor reshape failed");
    check::check(mSelfConditioningEmbedsB.reshape({batchSize * commitSeqLen, mRuntime.deployment.base.hiddenSize}),
        "Tensor reshape failed");
    if (!updateSelfConditioningTemperature(denoiseTemperature(0), context.stream))
    {
        return false;
    }

    check::check(io.outputLogits.reshape({batchSize * commitSeqLen, mRuntime.deployment.base.outputVocabSize}),
        "Tensor reshape failed");
    auto const commitDims = mRuntime.deployment.base.diffusionCommitDims(batchSize, commitSeqLen);
    if (!prepareCanvasMetadata(batchSize, commitSeqLen, commitDims, context.stream, &commitLengths,
            &mCommittedLengthsScratch, &context.residentRefs))
    {
        return false;
    }

    {
        TIME_STAGE(metrics::StageNames::kBLOCK_DIFFUSION_COMMIT, context.stream);
        bool status
            = mRuntime.base.executor.prepare(kDecodeProfile, commitDims, mRuntime.base.tensorMap, context.stream);
        if (status)
        {
            status = mRuntime.base.executor.execute(context.stream);
        }
        if (!status)
        {
            LOG_ERROR("Failed to execute DiffusionGemma commit backbone step.");
            return false;
        }
    }
    mRuntime.base.cacheManager.commitSequenceLength(io.contextLengths, context.stream);

    int32_t const* hostCanvas = mRuntime.sampling.hostPackedTokenIds.dataPointer<int32_t>();
    for (int32_t b = 0; b < batchSize; ++b)
    {
        for (int32_t i = 0; i < commitLengths[b]; ++i)
        {
            context.tokenIds[b].push_back(hostCanvas[b * canvasLen + i]);
            context.currentGenerateLengths[b] += 1;
        }
    }
    return true;
}

bool BlockDiffusionDecoder::decodeStep(DecodingInferenceContext& context)
{
    TIME_STAGE(metrics::StageNames::kLLM_GENERATION, context.stream);

    int32_t const batchSize = context.activeBatchSize;
    fillCommittedLengths(context, mCommittedLengthsScratch);
    std::vector<int32_t> const& committedLengths = mCommittedLengthsScratch;
    int32_t const maxCommittedLen
        = committedLengths.empty() ? 0 : *std::max_element(committedLengths.begin(), committedLengths.end());

    std::vector<int32_t>& remainingLengths = mRemainingLengthsScratch;
    remainingLengths.assign(batchSize, 0);
    int32_t maxRemaining = 0;
    for (int32_t b = 0; b < batchSize; ++b)
    {
        remainingLengths[b] = context.maxGenerateLength - context.currentGenerateLengths[b];
        maxRemaining = std::max(maxRemaining, remainingLengths[b]);
    }
    if (maxRemaining <= 0)
    {
        for (int32_t b = 0; b < batchSize; ++b)
        {
            context.finishedStates[b] = 1;
            context.slotStreams[b].terminalReason = FinishReason::kLength;
        }
        return true;
    }
    int32_t const availableKVLen = mRuntime.deployment.base.maxKVCacheCapacity - maxCommittedLen;
    if (availableKVLen <= 0)
    {
        for (int32_t b = 0; b < batchSize; ++b)
        {
            context.finishedStates[b] = 1;
            context.slotStreams[b].terminalReason = FinishReason::kLength;
        }
        return true;
    }

    // DiffusionGemma denoises a full block, then output policy trims by EOS or
    // maxGenerateLength. Shortening the canvas to the remaining token budget
    // changes the bidirectional denoise distribution and can bias position 0
    // toward EOS on short-answer workloads.
    int32_t const canvasLen = std::max(1, std::min(mCanvasLen, availableKVLen));
    std::vector<int32_t>& validCanvasLengths = mValidCanvasLengthsScratch;
    validCanvasLengths.assign(batchSize, canvasLen);
    if (!initializeCanvas(batchSize, canvasLen, context.stream))
    {
        return false;
    }

    int32_t const maxDenoisingSteps = effectiveMaxDenoisingSteps(context);
    int32_t denoiseSteps = 0;
    for (int32_t step = 0; step < maxDenoisingSteps; ++step)
    {
        float const temperature = denoiseTemperature(step, maxDenoisingSteps);
        DenoiseStepParams const denoiseParams{batchSize, canvasLen, step, &validCanvasLengths, &committedLengths,
            &context.residentRefs, temperature, context.stream};
        if (!runDenoiseStep(denoiseParams))
        {
            return false;
        }
        if (!sampleCanvasEntropyBound(batchSize, canvasLen, step, maxDenoisingSteps, context.stream))
        {
            return false;
        }
        denoiseSteps = step + 1;
        bool const forceAccept = (step + 1) >= maxDenoisingSteps;
        if (forceAccept)
        {
            break;
        }
    }

    if (!copyCanvasStateToHost(batchSize, canvasLen, context.stream))
    {
        return false;
    }

    int32_t const* hostCanvas = mRuntime.sampling.hostPackedTokenIds.dataPointer<int32_t>();
    int32_t const* hostPrefix = mHostPrefixLengths.dataPointer<int32_t>();
    int32_t minAcceptedPrefix = canvasLen;
    std::vector<int32_t>& commitLengths = mCommitLengthsScratch;
    commitLengths.assign(batchSize, 1);
    int32_t maxBlockLen = 1;
    for (int32_t b = 0; b < batchSize; ++b)
    {
        int32_t acceptedPrefix = hostPrefix[b];
        if (acceptedPrefix <= 0)
        {
            acceptedPrefix = 1;
        }
        minAcceptedPrefix = std::min(minAcceptedPrefix, acceptedPrefix);
        int32_t const validLen = validCanvasLengths[b];
        int32_t const remaining = std::max(1, std::min(validLen, remainingLengths[b]));
        int32_t length = std::min(acceptedPrefix, remaining);
        int32_t const savedGenerateLength = context.currentGenerateLengths[b];
        for (int32_t i = 0; i < length; ++i)
        {
            int32_t const tokenId = hostCanvas[b * canvasLen + i];
            context.currentGenerateLengths[b] = savedGenerateLength + i + 1;
            bool const shouldStop = context.shouldStopAfterAcceptedToken
                ? context.shouldStopAfterAcceptedToken(b, tokenId)
                : mRuntime.tokenizer.isEosId(tokenId);
            context.currentGenerateLengths[b] = savedGenerateLength;
            if (shouldStop)
            {
                length = i + 1;
                break;
            }
        }
        commitLengths[b] = length;
        maxBlockLen = std::max(maxBlockLen, length);
    }
    LOG_DEBUG("DiffusionGemma finalized commit maxBlockLen=%d minAcceptedPrefix=%d canvasLen=%d denoiseSteps=%d",
        maxBlockLen, minAcceptedPrefix, canvasLen, denoiseSteps);
    if (context.numLogprobs > 0)
    {
        Tensor& denoiseLogits = currentDenoiseLogits();
        int32_t const rows = batchSize * canvasLen;
        check::check(denoiseLogits.reshape({rows, mRuntime.deployment.base.outputVocabSize}), "Tensor reshape failed");
        decoder_utils::enqueueLogprobsD2H(denoiseLogits, rows, mRuntime, context.numLogprobs, context.stream);
        CUDA_CHECK(cudaStreamSynchronize(context.stream));
        decoder_utils::collectSpecLogprobsFromHost(
            mRuntime, context, batchSize, canvasLen, commitLengths.data(), context.numLogprobs);
    }
    return commitBlock(context, commitLengths, canvasLen);
}

bool BlockDiffusionDecoder::captureCudaGraphs(cudaStream_t stream)
{
    uint64_t const savedRandomOffset = mRandomOffset;
    int32_t const captureCanvasLen = std::min(mCanvasLen, mRuntime.deployment.base.maxKVCacheCapacity);
    if (captureCanvasLen <= 0)
    {
        LOG_WARNING("Skipping DiffusionGemma CUDA graph capture because max KV cache capacity is not positive.");
        return false;
    }
    static constexpr int32_t kSimulateCacheLength{128};
    int32_t const simulateCacheLength
        = std::min(kSimulateCacheLength, mRuntime.deployment.base.maxKVCacheCapacity - captureCanvasLen);

    struct ScopeGuard
    {
        std::function<void()> cleanup;
        ~ScopeGuard() noexcept
        {
            if (cleanup)
            {
                cleanup();
            }
        }
    } stateGuard{[&]() noexcept {
        mRandomOffset = savedRandomOffset;
        bindDefaultSelfConditioningTensors();
        std::vector<int32_t> zeroCacheLens(mRuntime.maxRuntimeBatchSize, 0);
        Tensor zeroCacheLensTensor(
            zeroCacheLens.data(), {mRuntime.maxRuntimeBatchSize}, DeviceType::kCPU, nvinfer1::DataType::kINT32);
        mRuntime.base.cacheManager.resetForNewSequences(zeroCacheLensTensor, stream);
    }};

    // Simulate a mid-sequence denoise state, matching the pattern used by other
    // graph-capture paths. KV cache bindings stay at physical capacity so graph
    // keys are independent of prompt/decode length; actual lengths are carried
    // by query_lengths and past_lengths.
    for (int32_t batchSize = 1; batchSize <= mRuntime.maxRuntimeBatchSize; ++batchSize)
    {
        std::vector<int32_t> simCacheLens(batchSize, simulateCacheLength);
        Tensor simCacheLensTensor(simCacheLens.data(), {batchSize}, DeviceType::kCPU, nvinfer1::DataType::kINT32);
        mRuntime.base.cacheManager.resetForNewSequences(simCacheLensTensor, stream);

        std::vector<int32_t> validCanvasLengths(batchSize, captureCanvasLen);
        if (!initializeCanvas(batchSize, captureCanvasLen, stream))
        {
            LOG_ERROR("Failed to initialize DiffusionGemma canvas for CUDA graph capture. batchSize=%d, canvasLen=%d.",
                batchSize, captureCanvasLen);
            return false;
        }

        int32_t const captureDenoiseSteps = std::min(mMaxDenoisingSteps, 2);
        for (int32_t step = 0; step < captureDenoiseSteps; ++step)
        {
            float const selfConditioningTemperature = denoiseTemperature(step);
            if (!prepareUnifiedConditioning(batchSize, captureCanvasLen, step, selfConditioningTemperature, stream))
            {
                LOG_ERROR(
                    "Failed to prepare DiffusionGemma conditioning for CUDA graph capture. batchSize=%d, canvasLen=%d, "
                    "step=%d.",
                    batchSize, captureCanvasLen, step);
                return false;
            }
            if (mRuntime.preprocess.gemma4Ple)
            {
                mRuntime.preprocess.gemma4Ple->embed(mCanvasIds, stream);
            }
            auto const denoiseDims = mRuntime.deployment.base.denoiseDims(batchSize, captureCanvasLen);
            if (!prepareCanvasMetadata(
                    batchSize, captureCanvasLen, denoiseDims, stream, &validCanvasLengths, &simCacheLens))
            {
                LOG_ERROR(
                    "Failed to prepare DiffusionGemma canvas metadata for CUDA graph capture. batchSize=%d, "
                    "canvasLen=%d, step=%d.",
                    batchSize, captureCanvasLen, step);
                return false;
            }

            if (!mRuntime.base.captureGraph(denoiseDims, stream))
            {
                LOG_ERROR("Failed to capture DiffusionGemma denoise CUDA graph. batchSize=%d, canvasLen=%d, step=%d.",
                    batchSize, captureCanvasLen, step);
                return false;
            }
        }

        check::check(mCommitCanvasIds.reshape({batchSize, captureCanvasLen}), "Tensor reshape failed");
        CUDA_CHECK(cudaMemsetAsync(mCommitCanvasIds.rawPointer(), 0,
            static_cast<size_t>(batchSize) * static_cast<size_t>(captureCanvasLen) * sizeof(int32_t), stream));
        check::check(mRuntime.base.pipelineIO.inputsEmbeds.reshape(
                         {batchSize * captureCanvasLen, mRuntime.deployment.base.hiddenSize}),
            "Tensor reshape failed");
        kernel::embeddingLookup(mCommitCanvasIds, mRuntime.preprocess.embedding.table,
            mRuntime.preprocess.embedding.scalesAsOptional(), mRuntime.base.pipelineIO.inputsEmbeds, stream);
        if (mRuntime.preprocess.gemma4Ple)
        {
            mRuntime.preprocess.gemma4Ple->embed(mCommitCanvasIds, stream);
        }
        bindDefaultSelfConditioningTensors();
        check::check(
            mSelfConditioningEmbedsA.reshape({batchSize * captureCanvasLen, mRuntime.deployment.base.hiddenSize}),
            "Tensor reshape failed");
        check::check(
            mSelfConditioningEmbedsB.reshape({batchSize * captureCanvasLen, mRuntime.deployment.base.hiddenSize}),
            "Tensor reshape failed");
        if (!updateSelfConditioningTemperature(denoiseTemperature(0), stream))
        {
            return false;
        }
        check::check(mRuntime.base.pipelineIO.outputLogits.reshape(
                         {batchSize * captureCanvasLen, mRuntime.deployment.base.outputVocabSize}),
            "Tensor reshape failed");
        auto const commitDims = mRuntime.deployment.base.diffusionCommitDims(batchSize, captureCanvasLen);
        if (!prepareCanvasMetadata(batchSize, captureCanvasLen, commitDims, stream, &validCanvasLengths, &simCacheLens))
        {
            LOG_ERROR(
                "Failed to prepare DiffusionGemma commit metadata for CUDA graph capture. batchSize=%d, canvasLen=%d.",
                batchSize, captureCanvasLen);
            return false;
        }
        if (!mRuntime.base.captureGraph(commitDims, stream))
        {
            LOG_ERROR("Failed to capture DiffusionGemma commit CUDA graph. batchSize=%d, canvasLen=%d.", batchSize,
                captureCanvasLen);
            return false;
        }
    }
    return true;
}

int64_t BlockDiffusionDecoder::getRequiredContextMemorySize() const noexcept
{
    return 0;
}

void BlockDiffusionDecoder::setContextMemory(Tensor&) {}

} // namespace rt
} // namespace trt_edgellm
