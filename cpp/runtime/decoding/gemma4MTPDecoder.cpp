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

#include "runtime/decoding/gemma4MTPDecoder.h"

#include "common/checkMacros.h"
#include "common/cudaUtils.h"
#include "common/logger.h"
#include "common/mathUtils.h"
#include "kernels/embeddingKernels/embeddingKernels.h"
#include "kernels/speculative/batchEvictKernels.h"
#include "kernels/speculative/dflashRuntimeKernels.h"
#include "kernels/speculative/eagleAcceptKernels.h"
#include "kernels/speculative/gemma4MTPRuntimeKernels.h"
#include "profiling/metrics.h"
#include "profiling/nvtx_wrapper.h"
#include "profiling/timer.h"
#include "runtime/decoding/decoderUtils.h"
#include "runtime/state/pipelineIO.h"
#include "sampler/sampling.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace trt_edgellm
{
namespace rt
{
namespace
{
constexpr int32_t kDecodeProfile{1};
} // namespace

Gemma4MTPDecoder::Gemma4MTPDecoder(DecodingRuntimeContext& runtime, std::filesystem::path const& engineDir,
    SpecDecodeDraftingConfig const& draftingConfig, cudaStream_t stream)
    : mRuntime(runtime)
{
    check::check(mRuntime.deployment.draft.has_value(), "Gemma4 MTP requires a draft model config.");
    check::check(mRuntime.deployment.specConfig.has_value(), "Gemma4 MTP requires a drafting config.");
    check::check(runtime.deployment.base.specDecodeType == SpecDecodeMode::kGemma4MTP,
        "Gemma4 MTP decoding requires a base engine exported with spec_decode_type=gemma4_mtp and engine_role=base.");
    check::check(runtime.deployment.base.reducedVocabSize == 0,
        "Gemma4 MTP currently requires a full-vocabulary base engine for greedy verification.");
    check::check(runtime.deployment.draft->sharesTargetKV && !runtime.deployment.draft->hasOwnKVCache,
        "Gemma4 MTP assistant must share target KV and must not own draft KV cache.");

    mDraftExecutor = decoder_utils::loadDraftEngine(engineDir, mRuntime.deployment);
    buildTensorMapForGemma4MTPDraft(
        mDraftTensorMap, mRuntime.base.pipelineIO, mRuntime.base.sharedResources, mRuntime.deployment);

    mDraftExternalWeightManager.load(engineDir, engineDir / "draft_config.json", stream);
    mDraftExternalWeightManager.validateAgainstEngine(*mDraftExecutor, "gemma4_mtp_draft");
    mDraftExternalWeightManager.registerTensorMapEntries(mDraftTensorMap);

    int32_t const maxRuntimeBatchSize = mRuntime.maxRuntimeBatchSize;
    int32_t const maxDraftStep = draftingConfig.draftingStep;
    int32_t const maxAcceptDepth = maxDraftStep + 1;

    mSeedTokenIds
        = Tensor({maxRuntimeBatchSize}, DeviceType::kGPU, nvinfer1::DataType::kINT32, "Gemma4MTP::seedTokenIds");
    mHostSeedTokenIds
        = Tensor({maxRuntimeBatchSize}, DeviceType::kCPU, nvinfer1::DataType::kINT32, "Gemma4MTP::hostSeedTokenIds");
    mSeedHiddenSourceTokenIndices = Tensor(
        {maxRuntimeBatchSize}, DeviceType::kGPU, nvinfer1::DataType::kINT32, "Gemma4MTP::seedHiddenSourceTokenIndices");
    mHostSeedHiddenSourceTokenIndices = Tensor({maxRuntimeBatchSize}, DeviceType::kCPU, nvinfer1::DataType::kINT32,
        "Gemma4MTP::hostSeedHiddenSourceTokenIndices");
    mDraftTokenIds = Tensor(
        {maxRuntimeBatchSize, maxDraftStep}, DeviceType::kGPU, nvinfer1::DataType::kINT32, "Gemma4MTP::draftTokenIds");
    mVerifyTokenIds = Tensor({maxRuntimeBatchSize, maxAcceptDepth}, DeviceType::kGPU, nvinfer1::DataType::kINT32,
        "Gemma4MTP::verifyTokenIds");
    mAcceptedTokenIds = Tensor({maxRuntimeBatchSize, maxAcceptDepth}, DeviceType::kGPU, nvinfer1::DataType::kINT32,
        "Gemma4MTP::acceptedTokenIds");
    mAcceptedTokenIndices = Tensor({maxRuntimeBatchSize, maxAcceptDepth}, DeviceType::kGPU, nvinfer1::DataType::kINT32,
        "Gemma4MTP::acceptedTokenIndices");
    mAcceptLength
        = Tensor({maxRuntimeBatchSize}, DeviceType::kGPU, nvinfer1::DataType::kINT32, "Gemma4MTP::acceptLength");
    mHostAcceptLengths
        = Tensor({maxRuntimeBatchSize}, DeviceType::kCPU, nvinfer1::DataType::kINT32, "Gemma4MTP::hostAcceptLengths");
    mHostAcceptedTokenIds = Tensor({maxRuntimeBatchSize, maxAcceptDepth}, DeviceType::kCPU, nvinfer1::DataType::kINT32,
        "Gemma4MTP::hostAcceptedTokenIds");
    mArgmaxScratch = Tensor(
        {maxRuntimeBatchSize * maxAcceptDepth}, DeviceType::kGPU, nvinfer1::DataType::kINT32, "Gemma4MTP::argmax");
}

bool Gemma4MTPDecoder::decodeStep(DecodingInferenceContext& context)
{
    NVTX_SCOPED_RANGE(nvtx_gemma4_mtp_decode, "Gemma4MTPDecoder::decodeStep", nvtx_colors::GREEN);
    return prepareSeed(context) && runAssistantDraftChain(context) && runBaseVerification(context)
        && acceptAndCommit(context) && updateNextSeed(context);
}

bool Gemma4MTPDecoder::captureCudaGraphs(cudaStream_t stream)
{
    bool draftCaptureStatus{true};
    bool baseVerificationCaptureStatus{true};

    static constexpr int32_t kSimulateCacheLength{128};

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
        std::vector<int32_t> zeroCacheLens(mRuntime.maxRuntimeBatchSize, 0);
        Tensor zeroCacheLensTensor(
            zeroCacheLens.data(), {mRuntime.maxRuntimeBatchSize}, DeviceType::kCPU, nvinfer1::DataType::kINT32);
        mRuntime.base.cacheManager.resetForNewSequences(zeroCacheLensTensor, stream);
        if (!mRuntime.base.executor.prepare(
                kDecodeProfile, mRuntime.deployment.base.resetDims(), mRuntime.base.tensorMap, stream))
        {
            LOG_ERROR("failed to reset base executor context during graph-capture teardown");
        }
    }};

    int32_t const draftingStep = mRuntime.deployment.specConfig->draftingStep;
    int32_t const verifySize = draftingStep + 1;
    int32_t const baseHiddenSize = mRuntime.deployment.specConfig->baseOutputHiddenDim;
    int32_t const draftVocabSize = mRuntime.deployment.draft->outputVocabSize;

    for (int32_t batchSize = 1; batchSize <= mRuntime.maxRuntimeBatchSize; ++batchSize)
    {
        std::vector<int32_t> simCacheLens(batchSize, kSimulateCacheLength);
        Tensor simCacheLensTensor(simCacheLens.data(), {batchSize}, DeviceType::kCPU, nvinfer1::DataType::kINT32);
        mRuntime.base.cacheManager.resetForNewSequences(simCacheLensTensor, stream);

        {
            check::check(
                mRuntime.base.pipelineIO.inputsEmbeds.reshape({batchSize, 1, mRuntime.deployment.base.hiddenSize}),
                "Tensor reshape failed");
            check::check(mRuntime.base.pipelineIO.draftHiddenStatesIn.reshape({batchSize, 1, baseHiddenSize}),
                "Tensor reshape failed");
            check::check(mRuntime.base.pipelineIO.draftHiddenStatesOut.reshape({batchSize, 1, baseHiddenSize}),
                "Tensor reshape failed");
            check::check(
                mRuntime.base.pipelineIO.outputLogits.reshape({batchSize, draftVocabSize}), "Tensor reshape failed");

            auto const draftDims = mRuntime.deployment.draft->decodeDims(batchSize);
            if (mDraftExecutor->prepare(kDecodeProfile, draftDims, mDraftTensorMap, stream))
            {
                draftCaptureStatus &= mDraftExecutor->captureGraph(stream);
            }
            else
            {
                LOG_WARNING("Gemma4 MTP: failed to prepare assistant draft graph capture (batch=%d)", batchSize);
                draftCaptureStatus = false;
            }
        }

        {
            int32_t const selectTokenSize = batchSize * verifySize;
            check::check(mRuntime.preprocess.idsInput.reshape({batchSize, verifySize}), "Tensor reshape failed");
            check::check(mRuntime.base.pipelineIO.inputsEmbeds.reshape(
                             {batchSize, verifySize, mRuntime.deployment.base.hiddenSize}),
                "Tensor reshape failed");
            check::check(mRuntime.base.pipelineIO.outputLogits.reshape(
                             {selectTokenSize, mRuntime.deployment.base.outputVocabSize}),
                "Tensor reshape failed");
            check::check(mRuntime.base.pipelineIO.baseHiddenStates.reshape({selectTokenSize, baseHiddenSize}),
                "Tensor reshape failed");
            check::check(mRuntime.base.pipelineIO.packedAttentionMask.reshape(
                             {batchSize, verifySize, static_cast<int64_t>(divUp(verifySize, 32))}),
                "Tensor reshape failed");
            check::check(
                mRuntime.base.pipelineIO.selectTokenIndices.reshape({batchSize, verifySize}), "Tensor reshape failed");
            check::check(mRuntime.base.pipelineIO.contextLengths.reshape({batchSize}), "Tensor reshape failed");
            check::check(mRuntime.base.pipelineIO.specDecodePositionIds.reshape({batchSize, verifySize}),
                "Tensor reshape failed");
            if (mRuntime.deployment.base.pleEnabled)
            {
                check::check(mRuntime.preprocess.gemma4Ple,
                    "Gemma4 MTP base config has PLE enabled but the Gemma4 PLE preprocessor is missing.");
                mRuntime.preprocess.gemma4Ple->reshapeOutputs(batchSize, verifySize);
            }

            Tensor const& baseKVCacheLengths = mRuntime.base.cacheManager.getKVCacheLengths();
            kernel::launchDFlashPrepareBaseVerifyInputs(baseKVCacheLengths.dataPointer<int32_t>(), verifySize,
                mRuntime.base.pipelineIO.packedAttentionMask.dataPointer<int32_t>(),
                mRuntime.base.pipelineIO.specDecodePositionIds.dataPointer<int32_t>(),
                mRuntime.base.pipelineIO.selectTokenIndices.dataPointer<int64_t>(),
                mRuntime.base.pipelineIO.contextLengths.dataPointer<int32_t>(), batchSize, stream);

            auto const verifyDims = mRuntime.deployment.base.specVerifyDims(batchSize, verifySize);
            baseVerificationCaptureStatus &= mRuntime.base.captureGraph(verifyDims, stream);
        }
    }

    LOG_INFO("Gemma4MTPDecoder: CUDA graph capture complete (draft=%s, baseVerify=%s)",
        draftCaptureStatus ? "ok" : "FAILED", baseVerificationCaptureStatus ? "ok" : "FAILED");

    return draftCaptureStatus && baseVerificationCaptureStatus;
}

int64_t Gemma4MTPDecoder::getRequiredContextMemorySize() const noexcept
{
    return mDraftExecutor ? mDraftExecutor->getRequiredContextMemorySize() : 0;
}

void Gemma4MTPDecoder::setContextMemory(Tensor& memory)
{
    if (mDraftExecutor)
    {
        mDraftExecutor->setContextMemory(memory);
    }
}

bool Gemma4MTPDecoder::hasSystemPromptKVCache(SystemPromptCacheKey const& key) const
{
    return mSystemPromptCacheKeys.count(key) > 0;
}

void Gemma4MTPDecoder::restoreSystemPromptKVCache(SystemPromptCacheKey const&, int32_t, cudaStream_t)
{
    // Gemma4 assistant owns no KV cache. The base cache restore already restores
    // the target KV that assistant shared-KV attention will read.
}

bool Gemma4MTPDecoder::runSystemPromptPrefill(DecodingInferenceContext&)
{
    // No assistant KV prefill exists for Gemma4 MTP.
    return true;
}

void Gemma4MTPDecoder::saveSystemPromptKVCache(
    SystemPromptCacheKey const& key, std::string const&, std::vector<tokenizer::Rank> const&, int32_t, cudaStream_t)
{
    mSystemPromptCacheKeys[key] = true;
}

void Gemma4MTPDecoder::resetForNewSequences(Tensor&, cudaStream_t)
{
    // No draft-owned KV cache to reset.
}

void Gemma4MTPDecoder::onBatchEvict(std::vector<int32_t> const& batchMapping, int32_t oldActiveBatch,
    int32_t newActiveBatch, Tensor& deviceBatchMapping, cudaStream_t stream)
{
    if (newActiveBatch <= 0)
    {
        return;
    }

    auto compactIfBatchShaped = [&](Tensor& tensor) {
        if (!tensor.isEmpty() && tensor.getShape().getNumDims() > 0 && tensor.getShape()[0] == oldActiveBatch)
        {
            Coords const oldShape = tensor.getShape();
            kernel::compactTensorBatch(tensor, deviceBatchMapping, tensor, oldActiveBatch, newActiveBatch, stream);
            std::vector<int64_t> newShape;
            newShape.reserve(oldShape.getNumDims());
            newShape.push_back(newActiveBatch);
            for (int32_t dim = 1; dim < oldShape.getNumDims(); ++dim)
            {
                newShape.push_back(oldShape[dim]);
            }
            check::check(tensor.reshape(newShape), "Tensor reshape failed");
        }
    };

    compactIfBatchShaped(mSeedTokenIds);
    compactIfBatchShaped(mDraftTokenIds);
    compactIfBatchShaped(mVerifyTokenIds);
    compactIfBatchShaped(mAcceptedTokenIds);
    compactIfBatchShaped(mAcceptedTokenIndices);
    compactIfBatchShaped(mAcceptLength);
    compactIfBatchShaped(mRuntime.base.pipelineIO.draftHiddenStatesIn);
    compactIfBatchShaped(mRuntime.base.pipelineIO.draftHiddenStatesOut);
    compactIfBatchShaped(mRuntime.base.pipelineIO.baseHiddenStates);

    if (!mHostAcceptLengths.isEmpty() && mHostAcceptLengths.getShape().getNumDims() > 0
        && mHostAcceptLengths.getShape()[0] == oldActiveBatch)
    {
        std::vector<int32_t> compacted(static_cast<size_t>(newActiveBatch));
        int32_t const* oldLengths = mHostAcceptLengths.dataPointer<int32_t>();
        for (int32_t oldIdx = 0; oldIdx < oldActiveBatch; ++oldIdx)
        {
            int32_t const newIdx = oldIdx < static_cast<int32_t>(batchMapping.size()) ? batchMapping[oldIdx] : -1;
            if (newIdx >= 0 && newIdx < newActiveBatch)
            {
                compacted[static_cast<size_t>(newIdx)] = oldLengths[oldIdx];
            }
        }
        std::copy(compacted.begin(), compacted.end(), mHostAcceptLengths.dataPointer<int32_t>());
        check::check(mHostAcceptLengths.reshape({newActiveBatch}), "Tensor reshape failed");
    }
    if (!mHostSeedTokenIds.isEmpty() && mHostSeedTokenIds.getShape().getNumDims() > 0
        && mHostSeedTokenIds.getShape()[0] == oldActiveBatch)
    {
        check::check(mHostSeedTokenIds.reshape({newActiveBatch}), "Tensor reshape failed");
    }
}

bool Gemma4MTPDecoder::prepareSeed(DecodingInferenceContext& context)
{
    TIME_STAGE(metrics::StageNames::kSPEC_DECODE_DRAFT_PROPOSAL, context.stream);
    int32_t const activeBatchSize = context.activeBatchSize;
    int32_t const baseHiddenSize = mRuntime.deployment.specConfig->baseOutputHiddenDim;
    int32_t const draftingStep = mRuntime.deployment.specConfig->draftingStep;
    int32_t const maxAcceptDepth = draftingStep + 1;

    check::check(mSeedTokenIds.reshape({activeBatchSize}), "Tensor reshape failed");
    check::check(mDraftTokenIds.reshape({activeBatchSize, draftingStep}), "Tensor reshape failed");
    check::check(mVerifyTokenIds.reshape({activeBatchSize, maxAcceptDepth}), "Tensor reshape failed");
    check::check(mAcceptedTokenIds.reshape({activeBatchSize, maxAcceptDepth}), "Tensor reshape failed");
    check::check(mAcceptedTokenIndices.reshape({activeBatchSize, maxAcceptDepth}), "Tensor reshape failed");
    check::check(mAcceptLength.reshape({activeBatchSize}), "Tensor reshape failed");
    check::check(mArgmaxScratch.reshape({activeBatchSize * maxAcceptDepth}), "Tensor reshape failed");
    check::check(mRuntime.preprocess.idsInput.reshape({activeBatchSize, 1}), "Tensor reshape failed");
    check::check(
        mRuntime.base.pipelineIO.inputsEmbeds.reshape({activeBatchSize, 1, mRuntime.deployment.base.hiddenSize}),
        "Tensor reshape failed");
    check::check(mRuntime.base.pipelineIO.draftHiddenStatesIn.reshape({activeBatchSize, 1, baseHiddenSize}),
        "Tensor reshape failed");
    check::check(mRuntime.base.pipelineIO.draftHiddenStatesOut.reshape({activeBatchSize, 1, baseHiddenSize}),
        "Tensor reshape failed");
    check::check(
        mRuntime.base.pipelineIO.outputLogits.reshape({activeBatchSize, mRuntime.deployment.draft->outputVocabSize}),
        "Tensor reshape failed");

    check::check(mHostSeedTokenIds.reshape({activeBatchSize}), "Tensor reshape failed");
    int32_t* hostSeedTokens = mHostSeedTokenIds.dataPointer<int32_t>();
    for (int32_t batchIdx = 0; batchIdx < activeBatchSize; ++batchIdx)
    {
        check::check(!context.tokenIds[batchIdx].empty(), "Gemma4 MTP requires a seed token per batch.");
        hostSeedTokens[batchIdx] = context.tokenIds[batchIdx].back();
    }
    CUDA_CHECK(cudaMemcpyAsync(mSeedTokenIds.rawPointer(), hostSeedTokens, activeBatchSize * sizeof(int32_t),
        cudaMemcpyHostToDevice, context.stream));
    CUDA_CHECK(cudaMemcpyAsync(mRuntime.preprocess.idsInput.rawPointer(), hostSeedTokens,
        activeBatchSize * sizeof(int32_t), cudaMemcpyHostToDevice, context.stream));

    kernel::embeddingLookup(mRuntime.preprocess.idsInput, mRuntime.preprocess.embedding.table,
        mRuntime.preprocess.embedding.scalesAsOptional(), mRuntime.base.pipelineIO.inputsEmbeds, context.stream);

    Tensor& sourceHiddenStates = mRuntime.base.pipelineIO.baseHiddenStates;
    Coords const sourceShape = sourceHiddenStates.getShape();
    check::check(sourceShape.getNumDims() == 3,
        "Gemma4 MTP expects base hidden states shaped [B, sourceSeqLen, hiddenDim] before seeding assistant.");
    check::check(sourceShape[0] == activeBatchSize && sourceShape[2] == baseHiddenSize,
        "Gemma4 MTP base hidden state shape does not match active batch or hidden size.");

    int64_t const sourceSeqLen = sourceShape[1];
    check::check(mHostSeedHiddenSourceTokenIndices.reshape({activeBatchSize}), "Tensor reshape failed");
    int32_t* hostSourceTokenIndices = mHostSeedHiddenSourceTokenIndices.dataPointer<int32_t>();
    int32_t const* previousAcceptLengths = (!mHostAcceptLengths.isEmpty() && context.generationRound > 0)
        ? mHostAcceptLengths.dataPointer<int32_t>()
        : nullptr;
    for (int32_t batchIdx = 0; batchIdx < activeBatchSize; ++batchIdx)
    {
        int64_t sourceTokenIdx = 0;
        if (context.generationRound == 0)
        {
            check::check(context.effectivePrefillLengths[batchIdx] > 0,
                "Gemma4 MTP requires positive effective prefill length.");
            sourceTokenIdx = context.effectivePrefillLengths[batchIdx] - 1;
        }
        else
        {
            check::check(previousAcceptLengths != nullptr && previousAcceptLengths[batchIdx] > 0,
                "Gemma4 MTP requires previous accept lengths after round 0.");
            sourceTokenIdx = previousAcceptLengths[batchIdx] - 1;
        }
        check::check(sourceTokenIdx >= 0 && sourceTokenIdx < sourceSeqLen,
            "Gemma4 MTP seed hidden source token index is out of range.");
        hostSourceTokenIndices[batchIdx] = static_cast<int32_t>(sourceTokenIdx);
    }

    check::check(mSeedHiddenSourceTokenIndices.reshape({activeBatchSize}), "Tensor reshape failed");
    CUDA_CHECK(cudaMemcpyAsync(mSeedHiddenSourceTokenIndices.rawPointer(), hostSourceTokenIndices,
        static_cast<size_t>(activeBatchSize) * sizeof(int32_t), cudaMemcpyHostToDevice, context.stream));
    kernel::launchGemma4MTPGatherSeedHidden(sourceHiddenStates.rawPointer(),
        mRuntime.base.pipelineIO.draftHiddenStatesIn.rawPointer(), mSeedHiddenSourceTokenIndices.dataPointer<int32_t>(),
        activeBatchSize, sourceSeqLen, baseHiddenSize, utils::getTypeSize(sourceHiddenStates.getDataType()),
        context.stream);

    // Root-token semantics: context.tokenIds already contains the prefill-sampled
    // root token. The root seeds assistant drafting and may be fed to target
    // verification to materialize target KV/hidden, but it must never be appended
    // through mAcceptedTokenIds a second time.
    return true;
}

bool Gemma4MTPDecoder::runAssistantDraftChain(DecodingInferenceContext& context)
{
    TIME_STAGE(metrics::StageNames::kSPEC_DECODE_DRAFT_PROPOSAL, context.stream);
    NVTX_SCOPED_RANGE(nvtx_gemma4_mtp_draft, "Gemma4MTPDecoder::runAssistantDraftChain", nvtx_colors::DARK_ORANGE);

    if (!mDraftExecutor)
    {
        LOG_ERROR("Gemma4 MTP draft engine not loaded.");
        return false;
    }

    int32_t const activeBatchSize = context.activeBatchSize;
    int32_t const draftingStep = mRuntime.deployment.specConfig->draftingStep;
    int32_t const baseHiddenSize = mRuntime.deployment.specConfig->baseOutputHiddenDim;
    int32_t const draftVocabSize = mRuntime.deployment.draft->outputVocabSize;

    auto const draftDims = mRuntime.deployment.draft->decodeDims(activeBatchSize);

    for (int32_t step = 0; step < draftingStep; ++step)
    {
        check::check(
            mRuntime.base.pipelineIO.outputLogits.reshape({activeBatchSize, draftVocabSize}), "Tensor reshape failed");
        check::check(mRuntime.base.pipelineIO.draftHiddenStatesIn.reshape({activeBatchSize, 1, baseHiddenSize}),
            "Tensor reshape failed");
        check::check(mRuntime.base.pipelineIO.draftHiddenStatesOut.reshape({activeBatchSize, 1, baseHiddenSize}),
            "Tensor reshape failed");

        bool draftSuccess = mDraftExecutor->prepare(kDecodeProfile, draftDims, mDraftTensorMap, context.stream);
        if (draftSuccess)
        {
            draftSuccess = mDraftExecutor->execute(context.stream);
        }
        if (!draftSuccess)
        {
            LOG_ERROR("Gemma4 MTP assistant draft step %d failed.", step);
            return false;
        }

        check::check(mRuntime.sampling.indices.reshape({activeBatchSize, 1}), "Tensor reshape failed");
        selectAllTopK(mRuntime.base.pipelineIO.outputLogits, std::nullopt, mRuntime.sampling.indices, 1,
            mRuntime.sampling.workspace, context.stream);

        kernel::launchGemma4MTPStoreDraftToken(mRuntime.sampling.indices.dataPointer<int32_t>(),
            mDraftTokenIds.dataPointer<int32_t>(), activeBatchSize, draftingStep, step, context.stream);

        if (step + 1 < draftingStep)
        {
            kernel::embeddingLookup(mRuntime.sampling.indices, mRuntime.preprocess.embedding.table,
                mRuntime.preprocess.embedding.scalesAsOptional(), mRuntime.base.pipelineIO.inputsEmbeds,
                context.stream);
            CUDA_CHECK(cudaMemcpyAsync(mRuntime.base.pipelineIO.draftHiddenStatesIn.rawPointer(),
                mRuntime.base.pipelineIO.draftHiddenStatesOut.rawPointer(),
                activeBatchSize * baseHiddenSize
                    * utils::getTypeSize(mRuntime.base.pipelineIO.draftHiddenStatesIn.getDataType()),
                cudaMemcpyDeviceToDevice, context.stream));
        }
    }

    return true;
}

bool Gemma4MTPDecoder::runBaseVerification(DecodingInferenceContext& context)
{
    TIME_STAGE(metrics::StageNames::kSPEC_DECODE_BASE_VERIFICATION, context.stream);
    NVTX_SCOPED_RANGE(nvtx_gemma4_mtp_verify, "Gemma4MTPDecoder::runBaseVerification", nvtx_colors::MAGENTA);

    int32_t const activeBatchSize = context.activeBatchSize;
    int32_t const draftingStep = mRuntime.deployment.specConfig->draftingStep;
    int32_t const verifySize = draftingStep + 1;
    int32_t const baseHiddenSize = mRuntime.deployment.specConfig->baseOutputHiddenDim;

    check::check(mVerifyTokenIds.reshape({activeBatchSize, verifySize}), "Tensor reshape failed");
    kernel::launchGemma4MTPBuildVerifyTokens(mSeedTokenIds.dataPointer<int32_t>(),
        mDraftTokenIds.dataPointer<int32_t>(), mVerifyTokenIds.dataPointer<int32_t>(), activeBatchSize, draftingStep,
        context.stream);

    check::check(mRuntime.preprocess.idsInput.reshape({activeBatchSize, verifySize}), "Tensor reshape failed");
    CUDA_CHECK(cudaMemcpyAsync(mRuntime.preprocess.idsInput.rawPointer(), mVerifyTokenIds.rawPointer(),
        activeBatchSize * verifySize * sizeof(int32_t), cudaMemcpyDeviceToDevice, context.stream));

    check::check(mRuntime.base.pipelineIO.inputsEmbeds.reshape(
                     {activeBatchSize, verifySize, mRuntime.deployment.base.hiddenSize}),
        "Tensor reshape failed");
    kernel::embeddingLookup(mRuntime.preprocess.idsInput, mRuntime.preprocess.embedding.table,
        mRuntime.preprocess.embedding.scalesAsOptional(), mRuntime.base.pipelineIO.inputsEmbeds, context.stream);
    if (mRuntime.deployment.base.pleEnabled)
    {
        check::check(mRuntime.preprocess.gemma4Ple,
            "Gemma4 MTP base config has PLE enabled but the Gemma4 PLE preprocessor is missing.");
        mRuntime.preprocess.gemma4Ple->embed(mRuntime.preprocess.idsInput, context.stream);
    }

    int32_t const selectTokenSize = activeBatchSize * verifySize;
    check::check(
        mRuntime.base.pipelineIO.outputLogits.reshape({selectTokenSize, mRuntime.deployment.base.outputVocabSize}),
        "Tensor reshape failed");
    check::check(
        mRuntime.base.pipelineIO.baseHiddenStates.reshape({selectTokenSize, baseHiddenSize}), "Tensor reshape failed");

    check::check(mRuntime.base.pipelineIO.packedAttentionMask.reshape(
                     {activeBatchSize, verifySize, static_cast<int64_t>(divUp(verifySize, 32))}),
        "Tensor reshape failed");
    check::check(
        mRuntime.base.pipelineIO.selectTokenIndices.reshape({activeBatchSize, verifySize}), "Tensor reshape failed");
    check::check(mRuntime.base.pipelineIO.contextLengths.reshape({activeBatchSize}), "Tensor reshape failed");
    check::check(
        mRuntime.base.pipelineIO.specDecodePositionIds.reshape({activeBatchSize, verifySize}), "Tensor reshape failed");

    Tensor const& baseKVCacheLengths = mRuntime.base.cacheManager.getKVCacheLengths();
    kernel::launchDFlashPrepareBaseVerifyInputs(baseKVCacheLengths.dataPointer<int32_t>(), verifySize,
        mRuntime.base.pipelineIO.packedAttentionMask.dataPointer<int32_t>(),
        mRuntime.base.pipelineIO.specDecodePositionIds.dataPointer<int32_t>(),
        mRuntime.base.pipelineIO.selectTokenIndices.dataPointer<int64_t>(),
        mRuntime.base.pipelineIO.contextLengths.dataPointer<int32_t>(), activeBatchSize, context.stream);

    auto const verifyDims = mRuntime.deployment.base.specVerifyDims(activeBatchSize, verifySize);
    bool verifySuccess
        = mRuntime.base.executor.prepare(kDecodeProfile, verifyDims, mRuntime.base.tensorMap, context.stream);
    if (verifySuccess)
    {
        verifySuccess = mRuntime.base.execute(context.stream);
    }
    if (!verifySuccess)
    {
        LOG_ERROR("Gemma4 MTP base verification execution failed.");
        return false;
    }

    check::check(mRuntime.base.pipelineIO.outputLogits.reshape(
                     {activeBatchSize, verifySize, mRuntime.deployment.base.outputVocabSize}),
        "Tensor reshape failed");
    check::check(mRuntime.base.pipelineIO.baseHiddenStates.reshape({activeBatchSize, verifySize, baseHiddenSize}),
        "Tensor reshape failed");

    return true;
}

bool Gemma4MTPDecoder::acceptAndCommit(DecodingInferenceContext& context)
{
    int32_t const activeBatchSize = context.activeBatchSize;
    int32_t const verifySize = mRuntime.deployment.specConfig->draftingStep + 1;

    check::check(mAcceptedTokenIds.reshape({activeBatchSize, verifySize}), "Tensor reshape failed");
    check::check(mAcceptLength.reshape({activeBatchSize}), "Tensor reshape failed");
    check::check(mArgmaxScratch.reshape({activeBatchSize * verifySize}), "Tensor reshape failed");

    kernel::sequentialAccept(mRuntime.base.pipelineIO.outputLogits, mVerifyTokenIds, mAcceptedTokenIds, mAcceptLength,
        mArgmaxScratch, activeBatchSize, verifySize, mRuntime.deployment.base.outputVocabSize, context.stream);

    // Enqueue logprobs device work + D2H so the copies ride the accept-length sync below.
    // Gemma4 MTP verification is a sequential chain: verify row j is accepted position j, so
    // extraction runs directly on the [B * verifySize, vocab] logits without a gather.
    if (context.numLogprobs > 0)
    {
        check::check(mRuntime.base.pipelineIO.outputLogits.reshape(
                         {activeBatchSize * verifySize, mRuntime.deployment.base.outputVocabSize}),
            "Tensor reshape failed");
        decoder_utils::enqueueLogprobsD2H(mRuntime.base.pipelineIO.outputLogits, activeBatchSize * verifySize, mRuntime,
            context.numLogprobs, context.stream);
    }

    check::check(mHostAcceptLengths.reshape({activeBatchSize}), "Tensor reshape failed");
    int32_t* hostAcceptLengths = mHostAcceptLengths.dataPointer<int32_t>();
    CUDA_CHECK(cudaMemcpyAsync(hostAcceptLengths, mAcceptLength.rawPointer(),
        static_cast<size_t>(activeBatchSize) * sizeof(int32_t), cudaMemcpyDeviceToHost, context.stream));
    CUDA_CHECK(cudaStreamSynchronize(context.stream));

    bool clampedAcceptLength = false;
    for (int32_t batchIdx = 0; batchIdx < activeBatchSize; ++batchIdx)
    {
        int32_t const remaining = std::max(0, context.maxGenerateLength - context.currentGenerateLengths[batchIdx]);
        int32_t const clampedLength = std::min(hostAcceptLengths[batchIdx], remaining);
        if (clampedLength != hostAcceptLengths[batchIdx])
        {
            hostAcceptLengths[batchIdx] = clampedLength;
            clampedAcceptLength = true;
        }
    }
    if (clampedAcceptLength)
    {
        CUDA_CHECK(cudaMemcpyAsync(mAcceptLength.rawPointer(), hostAcceptLengths,
            static_cast<size_t>(activeBatchSize) * sizeof(int32_t), cudaMemcpyHostToDevice, context.stream));
    }

    mRuntime.base.cacheManager.commitSequenceLength(mAcceptLength, context.stream);

    decoder_utils::appendAcceptedTokens(context, mHostAcceptLengths, mHostAcceptedTokenIds, mAcceptLength,
        mAcceptedTokenIds, verifySize, mRuntime.tokenizer, context.stream);

    if (context.numLogprobs > 0)
    {
        decoder_utils::collectSpecLogprobsFromHost(mRuntime, context, activeBatchSize, verifySize,
            mHostAcceptLengths.dataPointer<int32_t>(), context.numLogprobs);
    }

    return true;
}

bool Gemma4MTPDecoder::updateNextSeed(DecodingInferenceContext&)
{
    // Next round seed is read from context.tokenIds.back(); its companion
    // target hidden is gathered from baseHiddenStates using mHostAcceptLengths.
    return true;
}

} // namespace rt
} // namespace trt_edgellm
