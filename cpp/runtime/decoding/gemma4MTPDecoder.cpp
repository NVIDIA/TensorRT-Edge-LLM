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
#include "kernels/common/argmaxKernel.h"
#include "runtime/decoding/specDecodeUtils.h"

#include "common/checkMacros.h"
#include "common/cudaUtils.h"
#include "common/logger.h"
#include "common/mathUtils.h"
#include "kernels/embeddingKernels/embeddingKernels.h"
#include "kernels/speculative/batchEvictKernels.h"
#include "kernels/speculative/ddtreeKernels.h"
#include "kernels/speculative/dflashRuntimeKernels.h"
#include "kernels/speculative/gemma4MTPRuntimeKernels.h"
#include "kernels/speculative/speculativeKernels.h"
#include "profiling/metrics.h"
#include "profiling/nvtx_wrapper.h"
#include "profiling/timer.h"
#include "runtime/decoding/decoderUtils.h"
#include "runtime/decoding/guidedDecoder.h"
#include "runtime/decoding/logitBias.h"
#include "runtime/state/pipelineIO.h"
#include "sampler/sampling.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

namespace trt_edgellm
{
namespace rt
{
namespace
{
constexpr int32_t kDecodeProfile{1};

void fillRectangularMetadata(RaggedExecutionBatch& batch, int32_t batchSize, int32_t queryWidth,
    int32_t const* pastLengths, int32_t const* stateIndices, bool allLogits)
{
    batch.shape = {};
    batch.positions.clear();
    batch.queryStartOffsets.clear();
    batch.queryLengths.clear();
    batch.pastLengths.clear();
    batch.attentionSequenceLengths.clear();
    batch.stateIndices.clear();
    batch.logitsIndices.clear();
    int32_t const physicalTokens = batchSize * queryWidth;
    batch.shape.numSequences = batchSize;
    batch.shape.validTokens = physicalTokens;
    batch.shape.physicalTokens = physicalTokens;
    batch.shape.queryWidth = queryWidth;
    batch.positions.resize(physicalTokens);
    batch.queryStartOffsets.resize(batchSize + 1);
    batch.queryLengths.assign(batchSize, queryWidth);
    batch.pastLengths.assign(pastLengths, pastLengths + batchSize);
    batch.attentionSequenceLengths.resize(batchSize);
    batch.stateIndices.assign(stateIndices, stateIndices + batchSize);
    if (allLogits)
    {
        batch.logitsIndices.resize(physicalTokens);
    }
    for (int32_t batchIdx = 0; batchIdx < batchSize; ++batchIdx)
    {
        int32_t const start = batchIdx * queryWidth;
        batch.queryStartOffsets[batchIdx] = start;
        batch.attentionSequenceLengths[batchIdx] = pastLengths[batchIdx] + queryWidth;
        for (int32_t tokenIdx = 0; tokenIdx < queryWidth; ++tokenIdx)
        {
            int32_t const row = start + tokenIdx;
            batch.positions[row] = pastLengths[batchIdx] + tokenIdx;
            if (allLogits)
            {
                batch.logitsIndices[row] = row;
            }
        }
    }
    batch.queryStartOffsets[batchSize] = physicalTokens;
    batch.shape.numLogits = allLogits ? physicalTokens : 0;
}
} // namespace

Gemma4MTPDecoder::Gemma4MTPDecoder(DecodingRuntimeContext& runtime, std::unique_ptr<EngineExecutor> draftExecutor,
    ExternalWeightManager draftWeights, cudaStream_t stream)
    : mRuntime(runtime)
    , mDraftExecutor(std::move(draftExecutor))
{
    check::check(mRuntime.deployment.draft.has_value(), "Gemma4 MTP requires a draft model config.");
    check::check(mRuntime.deployment.specConfig.has_value(), "Gemma4 MTP requires a drafting config.");
    check::check(runtime.deployment.base.specDecodeType == SpecDecodeMode::kGemma4MTP,
        "Gemma4 MTP decoding requires a base engine exported with spec_decode_type=gemma4_mtp and engine_role=base.");
    check::check(runtime.deployment.base.reducedVocabSize == 0,
        "Gemma4 MTP currently requires a full-vocabulary base engine for greedy verification.");
    check::check(runtime.deployment.draft->sharesTargetKV && !runtime.deployment.draft->hasOwnKVCache,
        "Gemma4 MTP assistant must share target KV and must not own draft KV cache.");

    ELLM_CHECK(mDraftExecutor != nullptr, "Gemma4 MTP decoding requires a validated draft engine.");
    buildTensorMapForGemma4MTPDraft(
        mDraftTensorMap, mRuntime.base.pipelineIO, mRuntime.base.sharedResources, mRuntime.deployment);

    mDraftExternalWeightManager = std::move(draftWeights);
    mDraftExternalWeightManager.registerTensorMapEntries(mDraftTensorMap);

    int32_t const maxRuntimeBatchSize = mRuntime.maxRuntimeBatchSize;
    auto const& specConfig = *mRuntime.deployment.specConfig;
    int32_t const maxDraftStep = specConfig.draftSteps;
    int32_t const maxAcceptDepth = maxDraftStep + 1;
    // A grafting round verifies up to jumpVerifyTokenCount rows and accepts up to jumpMaxPathLength tokens.
    int32_t const verifyCapacity = std::max(maxAcceptDepth, specConfig.jumpVerifyTokenCount);
    int32_t const acceptCapacity = std::max(maxAcceptDepth, specConfig.jumpMaxPathLength);
    mAcceptCapacity = acceptCapacity;
    mUseTree = specConfig.draftTopK > 1;
    int32_t const maxPhysicalTokens = maxRuntimeBatchSize * maxAcceptDepth;
    mRaggedMetadataScratch.positions.reserve(maxPhysicalTokens);
    mRaggedMetadataScratch.queryStartOffsets.reserve(maxRuntimeBatchSize + 1);
    mRaggedMetadataScratch.queryLengths.reserve(maxRuntimeBatchSize);
    mRaggedMetadataScratch.pastLengths.reserve(maxRuntimeBatchSize);
    mRaggedMetadataScratch.attentionSequenceLengths.reserve(maxRuntimeBatchSize);
    mRaggedMetadataScratch.stateIndices.reserve(maxRuntimeBatchSize);
    mRaggedMetadataScratch.logitsIndices.reserve(maxPhysicalTokens);

    mSeedTokenIds
        = Tensor({maxRuntimeBatchSize}, DeviceType::kGPU, trt_edgellm::DataType::kINT32, "Gemma4MTP::seedTokenIds");
    mHostSeedTokenIds
        = Tensor({maxRuntimeBatchSize}, DeviceType::kCPU, trt_edgellm::DataType::kINT32, "Gemma4MTP::hostSeedTokenIds");
    mSeedHiddenSourceTokenIndices = Tensor({maxRuntimeBatchSize}, DeviceType::kGPU, trt_edgellm::DataType::kINT32,
        "Gemma4MTP::seedHiddenSourceTokenIndices");
    mHostSeedHiddenSourceTokenIndices = Tensor({maxRuntimeBatchSize}, DeviceType::kCPU, trt_edgellm::DataType::kINT32,
        "Gemma4MTP::hostSeedHiddenSourceTokenIndices");
    mDraftTokenIds = Tensor({maxRuntimeBatchSize, maxDraftStep}, DeviceType::kGPU, trt_edgellm::DataType::kINT32,
        "Gemma4MTP::draftTokenIds");
    mVerifyTokenIds = Tensor({maxRuntimeBatchSize, verifyCapacity}, DeviceType::kGPU, trt_edgellm::DataType::kINT32,
        "Gemma4MTP::verifyTokenIds");
    mAcceptedTokenIds = Tensor({maxRuntimeBatchSize, acceptCapacity}, DeviceType::kGPU, trt_edgellm::DataType::kINT32,
        "Gemma4MTP::acceptedTokenIds");
    mAcceptedTokenIndices = Tensor({maxRuntimeBatchSize, acceptCapacity}, DeviceType::kGPU,
        trt_edgellm::DataType::kINT32, "Gemma4MTP::acceptedTokenIndices");
    mAcceptLength
        = Tensor({maxRuntimeBatchSize}, DeviceType::kGPU, trt_edgellm::DataType::kINT32, "Gemma4MTP::acceptLength");
    mHostAcceptLengths = Tensor(
        {maxRuntimeBatchSize}, DeviceType::kCPU, trt_edgellm::DataType::kINT32, "Gemma4MTP::hostAcceptLengths");
    mHostSeatStaging = Tensor(
        {maxRuntimeBatchSize, 2}, DeviceType::kCPU, trt_edgellm::DataType::kINT32, "Gemma4MTP::hostSeatStaging");
    mHostAcceptedTokenIds = Tensor({maxRuntimeBatchSize, acceptCapacity}, DeviceType::kCPU,
        trt_edgellm::DataType::kINT32, "Gemma4MTP::hostAcceptedTokenIds");
    mArgmaxScratch = Tensor(
        {maxRuntimeBatchSize * verifyCapacity}, DeviceType::kGPU, trt_edgellm::DataType::kINT32, "Gemma4MTP::argmax");
    mAcceptedHiddenStates
        = Tensor({maxRuntimeBatchSize * acceptCapacity, mRuntime.deployment.specConfig->baseOutputHiddenDim},
            DeviceType::kGPU, trt_edgellm::DataType::kHALF, "Gemma4MTP::acceptedHidden");

    if (mUseTree)
    {
        int32_t const verifyTokenCount = mRuntime.deployment.specConfig->verifyTokenCount;
        int32_t const proposalDepthSize = maxDraftStep + 1;
        int32_t const draftVocabSize = mRuntime.deployment.draft->outputVocabSize;
        mStackedDraftLogits = Tensor({maxRuntimeBatchSize, proposalDepthSize, draftVocabSize}, DeviceType::kGPU,
            trt_edgellm::DataType::kFLOAT, "Gemma4MTP::stackedDraftLogits");
        mTreeTokenIds = Tensor({maxRuntimeBatchSize, verifyTokenCount}, DeviceType::kGPU, trt_edgellm::DataType::kINT32,
            "Gemma4MTP::treeTokenIds");
        mTreeNodeScores = Tensor({maxRuntimeBatchSize, verifyTokenCount}, DeviceType::kGPU,
            trt_edgellm::DataType::kFLOAT, "Gemma4MTP::treeNodeScores");
        mValidCounts
            = Tensor({maxRuntimeBatchSize}, DeviceType::kGPU, trt_edgellm::DataType::kINT32, "Gemma4MTP::validCounts");
        size_t const workspaceSize = kernel::getDDTreeBuildWorkspaceSize(maxRuntimeBatchSize, proposalDepthSize,
            verifyTokenCount, draftVocabSize, mRuntime.deployment.specConfig->draftTopK);
        ELLM_CHECK(workspaceSize > 0, "Gemma4MTPDecoder: invalid tree-build workspace size.");
        mTreeBuildWorkspace = Tensor({static_cast<int64_t>(workspaceSize)}, DeviceType::kGPU,
            trt_edgellm::DataType::kUINT8, "Gemma4MTP::treeBuildWorkspace");
        CUDA_CHECK(
            cudaMemsetAsync(mStackedDraftLogits.rawPointer(), 0, mStackedDraftLogits.getMemoryCapacity(), stream));
        CUDA_CHECK(cudaMemsetAsync(mSeedTokenIds.rawPointer(), 0, mSeedTokenIds.getMemoryCapacity(), stream));
    }

    // A tree takes forced chains under any node; a chain takes one, which replaces the draft after its node.
    mJump.initialize(mRuntime.tokenizer, mRuntime.guidedDecoder, maxRuntimeBatchSize,
        mUseTree ? JumpGraftGeometry::kTree : JumpGraftGeometry::kChain, specConfig, specConfig.verifyTokenCount);
}

DecodingKvHeadroomPolicy Gemma4MTPDecoder::kvHeadroom() const
{
    auto const& config = *mRuntime.deployment.specConfig;
    ELLM_CHECK(
        config.verifyTokenCount > 0 && config.draftSteps > 0 && config.jumpVerifyTokenCount >= config.verifyTokenCount,
        "Gemma4 MTP KV headroom is outside the supported range");
    // The assistant owns no KV cache; a grafting round verifies jumpVerifyTokenCount rows.
    return {/*.plain=*/{config.verifyTokenCount, 0}, /*.jumpForward=*/{config.jumpVerifyTokenCount, 0}};
}

bool Gemma4MTPDecoder::decodeStep(DecodingInferenceContext& context)
{
    if (mUseTree && ::trt_edgellm::shouldUseNonGreedySampling(context.uniformSamplingParams()))
    {
        LOG_ERROR(
            "Gemma4MTPDecoder: tree drafting supports greedy decoding only; build a standalone target engine "
            "for non-greedy sampling.");
        return false;
    }
    NVTX_SCOPED_RANGE(nvtx_gemma4_mtp_decode, "Gemma4MTPDecoder::decodeStep", nvtx_colors::GREEN);
    return runDraftProposal(context) && runBaseVerification(context) && acceptAndCommit(context)
        && updateNextSeed(context);
}

bool Gemma4MTPDecoder::runDraftProposal(DecodingInferenceContext& context)
{
    TIME_STAGE(metrics::StageNames::kSPEC_DECODE_DRAFT_PROPOSAL, context.stream);
    return prepareSeed(context) && runAssistantDraftChain(context);
}

bool Gemma4MTPDecoder::captureCudaGraphs(cudaStream_t stream)
{
    bool draftCaptureStatus{true};
    bool baseVerificationCaptureStatus{true};

    DecodingInferenceContext captureContext;
    captureContext.stream = stream;
    captureContext.residentRefs.resize(mRuntime.maxRuntimeBatchSize);
    for (int32_t slot = 0; slot < mRuntime.maxRuntimeBatchSize; ++slot)
    {
        captureContext.residentRefs[slot].slot = slot;
    }

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
            zeroCacheLens.data(), {mRuntime.maxRuntimeBatchSize}, DeviceType::kCPU, trt_edgellm::DataType::kINT32);
        mRuntime.base.cacheManager.resetForNewSequences(zeroCacheLensTensor, stream);
        if (!mRuntime.base.executor.prepare(
                kDecodeProfile, mRuntime.deployment.base.resetDims(), mRuntime.base.tensorMap, stream))
        {
            LOG_ERROR("failed to reset base executor context during graph-capture teardown");
        }
    }};

    auto const& specConfig = *mRuntime.deployment.specConfig;
    int32_t const baseHiddenSize = specConfig.baseOutputHiddenDim;
    int32_t const draftVocabSize = mRuntime.deployment.draft->outputVocabSize;
    // A deployment that can jump also verifies grafting rounds at jumpVerifyTokenCount rows.
    std::vector<int32_t> verifyWidths{specConfig.verifyTokenCount};
    mJump.addGraphTiers(verifyWidths, nullptr);

    for (int32_t batchSize = 1; batchSize <= mRuntime.maxRuntimeBatchSize; ++batchSize)
    {
        std::vector<int32_t> simCacheLens(batchSize, kSimulateCacheLength);
        std::vector<int32_t> simStateIndices(batchSize);
        std::iota(simStateIndices.begin(), simStateIndices.end(), 0);
        Tensor simCacheLensTensor(simCacheLens.data(), {batchSize}, DeviceType::kCPU, trt_edgellm::DataType::kINT32);
        mRuntime.base.cacheManager.resetForNewSequences(simCacheLensTensor, stream);

        fillRectangularMetadata(mRaggedMetadataScratch, batchSize, 1, simCacheLens.data(), simStateIndices.data(),
            /*allLogits=*/false);
        RaggedExecutionBatch& draftMetadata = mRaggedMetadataScratch;
        for (int32_t batchIdx = 0; batchIdx < batchSize; ++batchIdx)
        {
            draftMetadata.positions[batchIdx] = simCacheLens[batchIdx] - 1;
            draftMetadata.attentionSequenceLengths[batchIdx] = simCacheLens[batchIdx];
        }
        mRuntime.base.pipelineIO.uploadRaggedMetadata(draftMetadata, stream);
        prepareRaggedKVPageTable(
            mRuntime.base.pipelineIO, *mRuntime.base.sharedResources.kvPageTables[0], batchSize, stream);
        prepareRaggedRope(mRuntime.base.pipelineIO, mRuntime.base.sharedResources, *mRuntime.deployment.draft,
            batchSize, batchSize, stream);

        {
            check::check(
                mRuntime.base.pipelineIO.inputsEmbeds.reshape({batchSize, mRuntime.deployment.base.hiddenSize}),
                "Tensor reshape failed");
            check::check(mRuntime.base.pipelineIO.draftHiddenStatesIn.reshape({batchSize, baseHiddenSize}),
                "Tensor reshape failed");
            check::check(mRuntime.base.pipelineIO.draftHiddenStatesOut.reshape({batchSize, baseHiddenSize}),
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

        for (int32_t const verifyTokenCount : verifyWidths)
        {
            bool const wide = verifyTokenCount > specConfig.verifyTokenCount;
            auto& io = mRuntime.base.pipelineIO;
            if (mUseTree && !wide)
            {
                buildTreeVerifyInputs(batchSize, stream);
            }
            else if (mUseTree)
            {
                Tensor& ids = mRuntime.preprocess.idsInput;
                check::check(ids.reshape({batchSize, verifyTokenCount}), "Tensor reshape failed");
                CUDA_CHECK(cudaMemsetAsync(
                    ids.rawPointer(), 0, static_cast<size_t>(batchSize) * verifyTokenCount * sizeof(int32_t), stream));
                Tensor& mask = mJump.grafter().mask();
                check::check(mask.reshape({batchSize, verifyTokenCount, verifyTokenCount}), "Tensor reshape failed");
                CUDA_CHECK(cudaMemsetAsync(mask.rawPointer(), 0,
                    static_cast<size_t>(batchSize) * verifyTokenCount * verifyTokenCount, stream));
                check::check(io.specTreeParentIds.reshape({batchSize, verifyTokenCount}), "Tensor reshape failed");
                check::check(io.specTreeDepths.reshape({batchSize, verifyTokenCount}), "Tensor reshape failed");
                mJump.prepareGraftedTreeDescriptors(
                    mRuntime.base.pipelineIO, mRuntime.base.cacheManager.getKVCacheLengths(), batchSize, stream);
            }
            else
            {
                check::check(mVerifyTokenIds.reshape({batchSize, verifyTokenCount}), "Tensor reshape failed");
                CUDA_CHECK(cudaMemsetAsync(
                    mVerifyTokenIds.rawPointer(), 0, batchSize * verifyTokenCount * sizeof(int32_t), stream));
            }
            captureContext.activeBatchSize = batchSize;
            // A grafted tree lives in idsInput; its node counts are bound per round.
            Tensor const& verifyIds
                = !mUseTree ? mVerifyTokenIds : (wide ? mRuntime.preprocess.idsInput : mTreeTokenIds);
            SpecVerifyInputs const verify{verifyIds, mUseTree && !wide ? &mValidCounts : nullptr,
                mUseTree ? &io.specTreeParentIds : nullptr, mUseTree ? &io.specTreeDepths : nullptr, batchSize,
                verifyTokenCount, specConfig.maxPathLength, specConfig.verifyLayout, mUseTree};
            ELLM_CHECK(!mRuntime.deployment.base.gemmaPleEnabled || mRuntime.preprocess.gemma4Ple,
                "Gemma4 MTP base requires its configured PLE preprocessor");
            spec_decode::prepareTargetVerifyInputs(verify, mRuntime, captureContext, 0);
            auto const verifyDims = mRuntime.deployment.base.specVerifyDims(batchSize, verifyTokenCount, verify.layout);
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

void Gemma4MTPDecoder::resetForNewSequences(Tensor&, cudaStream_t)
{
    // No draft-owned KV cache to reset; the next round seeds from the prefill.
    mDraftPrimed = false;
}

void Gemma4MTPDecoder::beginBatch() noexcept
{
    mDraftPrimed = false;
}

void Gemma4MTPDecoder::appendRow(int32_t executionRow) noexcept
{
    // Within capacity by construction: the session appends no row past the runtime batch.
    check::check(mAcceptLength.reshape({executionRow + 1}) && mHostAcceptLengths.reshape({executionRow + 1}),
        "Tensor reshape failed");
}

bool Gemma4MTPDecoder::seatRow(DecodingInferenceContext& context, int32_t executionRow)
{
    // The assistant has no KV of its own (it reads the target's), so seating is the seed alone: the first sampled
    // token with the target hidden state at the last prompt position, as a resident whose verification accepted
    // one token.
    check::check(mRuntime.base.pipelineIO.baseHiddenStates.reshape(
                     {context.effectivePrefillLengths[0], mRuntime.deployment.specConfig->baseOutputHiddenDim}),
        "Tensor reshape failed");
    decoder_utils::seatAcceptedSpan(context, executionRow, mAcceptedTokenIds, mAcceptLength, mHostSeatStaging,
        mAcceptedHiddenStates, mAcceptCapacity, mRuntime.base.pipelineIO.baseHiddenStates, context.stream);
    mHostAcceptLengths.dataPointer<int32_t>()[executionRow] = 1;
    mDraftPrimed = true;
    return true;
}

void Gemma4MTPDecoder::onBatchEvict(std::vector<int32_t> const& batchMapping, int32_t oldActiveBatch,
    int32_t newActiveBatch, Tensor& deviceBatchMapping, cudaStream_t stream)
{
    if (newActiveBatch <= 0)
    {
        mDraftPrimed = false;
        return;
    }

    // What a round leaves for the next one is the seed of each row: the accepted span's hidden states and the
    // accepted length that locates the seed among them (the seed token is the context's). Everything else here is
    // per-round scratch, which prepareSeed rewrites for the batch that is left.
    if (mDraftPrimed)
    {
        check::check(mAcceptedHiddenStates.reshape({static_cast<int64_t>(oldActiveBatch) * mAcceptCapacity,
                         mRuntime.deployment.specConfig->baseOutputHiddenDim}),
            "Tensor reshape failed");
        kernel::compactExecutionTensorBatch(
            mAcceptedHiddenStates, deviceBatchMapping, oldActiveBatch, newActiveBatch, stream);
    }
    else
    {
        // Before the first round (the blocking path's round zero), the pipeline buffer holds the target's prefill
        // hidden states the first seed is gathered from.
        kernel::compactExecutionTensorBatch(
            mRuntime.base.pipelineIO.baseHiddenStates, deviceBatchMapping, oldActiveBatch, newActiveBatch, stream);
    }

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
}

bool Gemma4MTPDecoder::prepareSeed(DecodingInferenceContext& context)
{
    int32_t const activeBatchSize = context.activeBatchSize;
    int32_t const baseHiddenSize = mRuntime.deployment.specConfig->baseOutputHiddenDim;
    int32_t const draftSteps = mRuntime.deployment.specConfig->draftSteps;
    int32_t const maxAcceptDepth = draftSteps + 1;

    check::check(mSeedTokenIds.reshape({activeBatchSize}), "Tensor reshape failed");
    check::check(mDraftTokenIds.reshape({activeBatchSize, draftSteps}), "Tensor reshape failed");
    check::check(mVerifyTokenIds.reshape({activeBatchSize, maxAcceptDepth}), "Tensor reshape failed");
    check::check(mAcceptedTokenIds.reshape({activeBatchSize, maxAcceptDepth}), "Tensor reshape failed");
    check::check(mAcceptedTokenIndices.reshape({activeBatchSize, maxAcceptDepth}), "Tensor reshape failed");
    check::check(mAcceptLength.reshape({activeBatchSize}), "Tensor reshape failed");
    check::check(mArgmaxScratch.reshape({activeBatchSize * maxAcceptDepth}), "Tensor reshape failed");
    check::check(mRuntime.preprocess.idsInput.reshape({activeBatchSize, 1}), "Tensor reshape failed");
    check::check(mRuntime.base.pipelineIO.inputsEmbeds.reshape({activeBatchSize, mRuntime.deployment.base.hiddenSize}),
        "Tensor reshape failed");
    check::check(mRuntime.base.pipelineIO.draftHiddenStatesIn.reshape({activeBatchSize, baseHiddenSize}),
        "Tensor reshape failed");
    check::check(mRuntime.base.pipelineIO.draftHiddenStatesOut.reshape({activeBatchSize, baseHiddenSize}),
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
        mRuntime.preprocess.embedding.scalesAsOptional(), mRuntime.preprocess.embedding.descriptor,
        mRuntime.base.pipelineIO.inputsEmbeds, context.stream);

    // The first round of a request seeds from the target's prefill hidden states; every later one (and a seated
    // row's first) from the accepted span the decoder carried from the last verification.
    if (mDraftPrimed)
    {
        check::check(
            mAcceptedHiddenStates.reshape({static_cast<int64_t>(activeBatchSize) * mAcceptCapacity, baseHiddenSize}),
            "Tensor reshape failed");
    }
    Tensor& sourceHiddenStates = mDraftPrimed ? mAcceptedHiddenStates : mRuntime.base.pipelineIO.baseHiddenStates;
    Coords const sourceShape = sourceHiddenStates.getShape();
    check::check(
        sourceShape.getNumDims() == 2 && sourceShape[0] % activeBatchSize == 0 && sourceShape[1] == baseHiddenSize,
        "Gemma4 MTP expects token-major base hidden states [T, hiddenDim] before seeding assistant.");

    int64_t const sourceSeqLen = sourceShape[0] / activeBatchSize;
    check::check(mHostSeedHiddenSourceTokenIndices.reshape({activeBatchSize}), "Tensor reshape failed");
    int32_t* hostSourceTokenIndices = mHostSeedHiddenSourceTokenIndices.dataPointer<int32_t>();
    int32_t const* previousAcceptLengths
        = (!mHostAcceptLengths.isEmpty() && mDraftPrimed) ? mHostAcceptLengths.dataPointer<int32_t>() : nullptr;
    for (int32_t batchIdx = 0; batchIdx < activeBatchSize; ++batchIdx)
    {
        int64_t sourceTokenIdx = 0;
        if (!mDraftPrimed)
        {
            check::check(context.effectivePrefillLengths[batchIdx] > 0,
                "Gemma4 MTP requires positive effective prefill length.");
            sourceTokenIdx = context.effectivePrefillLengths[batchIdx] - 1;
        }
        else
        {
            check::check(previousAcceptLengths != nullptr && previousAcceptLengths[batchIdx] > 0,
                "Gemma4 MTP requires the previous round's accept lengths.");
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
    mDraftPrimed = true;

    // Root-token semantics: context.tokenIds already contains the prefill-sampled
    // root token. The root seeds assistant drafting and may be fed to target
    // verification to materialize target KV/hidden, but it must never be appended
    // through mAcceptedTokenIds a second time.
    return true;
}

bool Gemma4MTPDecoder::runAssistantDraftChain(DecodingInferenceContext& context)
{
    NVTX_SCOPED_RANGE(nvtx_gemma4_mtp_draft, "Gemma4MTPDecoder::runAssistantDraftChain", nvtx_colors::DARK_ORANGE);

    if (!mDraftExecutor)
    {
        LOG_ERROR("Gemma4 MTP draft engine not loaded.");
        return false;
    }

    int32_t const activeBatchSize = context.activeBatchSize;
    int32_t const draftSteps = mRuntime.deployment.specConfig->draftSteps;
    int32_t const baseHiddenSize = mRuntime.deployment.specConfig->baseOutputHiddenDim;
    int32_t const draftVocabSize = mRuntime.deployment.draft->outputVocabSize;

    auto const draftDims = mRuntime.deployment.draft->decodeDims(activeBatchSize);
    RaggedExecutionBatch& metadata = mRaggedMetadataScratch;
    metadata.shape = {};
    metadata.positions.clear();
    metadata.queryStartOffsets.clear();
    metadata.queryLengths.clear();
    metadata.pastLengths.clear();
    metadata.attentionSequenceLengths.clear();
    metadata.stateIndices.clear();
    metadata.logitsIndices.clear();
    metadata.shape.numSequences = activeBatchSize;
    metadata.shape.validTokens = activeBatchSize;
    metadata.shape.physicalTokens = activeBatchSize;
    metadata.shape.queryWidth = 1;
    metadata.positions.resize(activeBatchSize);
    metadata.queryStartOffsets.resize(activeBatchSize + 1);
    metadata.queryLengths.assign(activeBatchSize, 1);
    metadata.pastLengths.resize(activeBatchSize);
    metadata.attentionSequenceLengths.resize(activeBatchSize);
    metadata.stateIndices.resize(activeBatchSize);
    for (int32_t batchIdx = 0; batchIdx < activeBatchSize; ++batchIdx)
    {
        int32_t const targetLength = context.committedLengths[batchIdx];
        check::check(targetLength > 0, "Gemma4 MTP assistant requires a non-empty shared target KV sequence.");
        metadata.positions[batchIdx] = targetLength - 1;
        metadata.queryStartOffsets[batchIdx] = batchIdx;
        metadata.pastLengths[batchIdx] = targetLength;
        metadata.attentionSequenceLengths[batchIdx] = targetLength;
        metadata.stateIndices[batchIdx] = context.residentRefs[batchIdx].slot;
    }
    metadata.queryStartOffsets[activeBatchSize] = activeBatchSize;
    mRuntime.base.pipelineIO.uploadRaggedMetadata(metadata, context.stream);
    prepareRaggedKVPageTable(
        mRuntime.base.pipelineIO, *mRuntime.base.sharedResources.kvPageTables[0], activeBatchSize, context.stream);
    prepareRaggedRope(mRuntime.base.pipelineIO, mRuntime.base.sharedResources, *mRuntime.deployment.draft,
        activeBatchSize, activeBatchSize, context.stream);

    for (int32_t step = 0; step < draftSteps; ++step)
    {
        check::check(
            mRuntime.base.pipelineIO.outputLogits.reshape({activeBatchSize, draftVocabSize}), "Tensor reshape failed");
        check::check(mRuntime.base.pipelineIO.draftHiddenStatesIn.reshape({activeBatchSize, baseHiddenSize}),
            "Tensor reshape failed");
        check::check(mRuntime.base.pipelineIO.draftHiddenStatesOut.reshape({activeBatchSize, baseHiddenSize}),
            "Tensor reshape failed");

        bool draftSuccess;
        {
            auto forward
                = gTimer.startForward("draft", draftDims.batch, draftDims.seqLen, draftDims.selectLen, context.stream);
            draftSuccess = mDraftExecutor->prepare(kDecodeProfile, draftDims, mDraftTensorMap, context.stream);
            if (draftSuccess)
            {
                draftSuccess = mDraftExecutor->execute(context.stream);
            }
        }
        if (!draftSuccess)
        {
            LOG_ERROR("Gemma4 MTP assistant draft step %d failed.", step);
            return false;
        }

        if (mUseTree)
        {
            int32_t const proposalDepthSize = draftSteps + 1;
            size_t const rowBytes = static_cast<size_t>(draftVocabSize) * sizeof(float);
            CUDA_CHECK(cudaMemcpy2DAsync(
                static_cast<char*>(mStackedDraftLogits.rawPointer()) + static_cast<size_t>(step + 1) * rowBytes,
                static_cast<size_t>(proposalDepthSize) * rowBytes, mRuntime.base.pipelineIO.outputLogits.rawPointer(),
                rowBytes, rowBytes, activeBatchSize, cudaMemcpyDeviceToDevice, context.stream));
        }

        check::check(mRuntime.sampling.indices.reshape({activeBatchSize, 1}), "Tensor reshape failed");
        selectAllTopK(mRuntime.base.pipelineIO.outputLogits, std::nullopt, mRuntime.sampling.indices, 1,
            mRuntime.sampling.workspace, context.stream);

        kernel::launchGemma4MTPStoreDraftToken(mRuntime.sampling.indices.dataPointer<int32_t>(),
            mDraftTokenIds.dataPointer<int32_t>(), activeBatchSize, draftSteps, step, context.stream);

        if (step + 1 < draftSteps)
        {
            kernel::embeddingLookup(mRuntime.sampling.indices, mRuntime.preprocess.embedding.table,
                mRuntime.preprocess.embedding.scalesAsOptional(), mRuntime.preprocess.embedding.descriptor,
                mRuntime.base.pipelineIO.inputsEmbeds, context.stream);
            CUDA_CHECK(cudaMemcpyAsync(mRuntime.base.pipelineIO.draftHiddenStatesIn.rawPointer(),
                mRuntime.base.pipelineIO.draftHiddenStatesOut.rawPointer(),
                activeBatchSize * baseHiddenSize
                    * utils::getTypeSize(mRuntime.base.pipelineIO.draftHiddenStatesIn.getDataType()),
                cudaMemcpyDeviceToDevice, context.stream));
        }
    }

    return true;
}

bool Gemma4MTPDecoder::buildTreeVerifyInputs(int32_t activeBatchSize, cudaStream_t stream)
{
    NVTX_SCOPED_RANGE(nvtx_gemma4_mtp_tree_build, "Gemma4MTPDecoder::buildTreeVerifyInputs", nvtx_colors::LIGHT_ORANGE);

    auto const& config = *mRuntime.deployment.specConfig;
    int32_t const verifyTokenCount = config.verifyTokenCount;
    int32_t const proposalDepthSize = config.draftSteps + 1;
    int32_t const draftVocabSize = mRuntime.deployment.draft->outputVocabSize;

    check::check(
        mStackedDraftLogits.reshape({activeBatchSize, proposalDepthSize, draftVocabSize}), "Tensor reshape failed");
    check::check(mSeedTokenIds.reshape({activeBatchSize}), "Tensor reshape failed");
    check::check(mTreeTokenIds.reshape({activeBatchSize, verifyTokenCount}), "Tensor reshape failed");
    check::check(mTreeNodeScores.reshape({activeBatchSize, verifyTokenCount}), "Tensor reshape failed");
    check::check(mValidCounts.reshape({activeBatchSize}), "Tensor reshape failed");
    check::check(mRuntime.base.pipelineIO.specTreeParentIds.reshape({activeBatchSize, verifyTokenCount}),
        "Tensor reshape failed");
    check::check(
        mRuntime.base.pipelineIO.specTreeDepths.reshape({activeBatchSize, verifyTokenCount}), "Tensor reshape failed");
    check::check(mRuntime.preprocess.idsInput.reshape({activeBatchSize, verifyTokenCount}), "Tensor reshape failed");
    check::check(mRuntime.base.pipelineIO.specDecodePositionIds.reshape({activeBatchSize, verifyTokenCount}),
        "Tensor reshape failed");
    check::check(mRuntime.base.pipelineIO.packedAttentionMask.reshape(
                     {activeBatchSize, verifyTokenCount, static_cast<int64_t>(divUp(verifyTokenCount, 32))}),
        "Tensor reshape failed");
    check::check(mRuntime.base.pipelineIO.contextLengths.reshape({activeBatchSize}), "Tensor reshape failed");
    check::check(mRuntime.base.pipelineIO.selectTokenIndices.reshape({activeBatchSize, verifyTokenCount}),
        "Tensor reshape failed");

    Tensor const& baseKVCacheLengths = mRuntime.base.cacheManager.getKVCacheLengths();
    kernel::DDTreeBuildParams const buildParams{{mStackedDraftLogits, mSeedTokenIds, baseKVCacheLengths, nullptr},
        {mTreeTokenIds, mRuntime.base.pipelineIO.specTreeDepths, mRuntime.base.pipelineIO.specTreeParentIds,
            mTreeNodeScores, mValidCounts, mRuntime.preprocess.idsInput, mRuntime.base.pipelineIO.specDecodePositionIds,
            mRuntime.base.pipelineIO.packedAttentionMask, mRuntime.base.pipelineIO.contextLengths,
            mRuntime.base.pipelineIO.selectTokenIndices},
        config.draftTopK, mTreeBuildWorkspace.rawPointer(),
        static_cast<size_t>(mTreeBuildWorkspace.getMemoryCapacity()), stream};
    kernel::ddtreeBuild(buildParams);
    return true;
}

bool Gemma4MTPDecoder::runBaseVerification(DecodingInferenceContext& context)
{
    TIME_STAGE(metrics::StageNames::kSPEC_DECODE_BASE_VERIFICATION, context.stream);
    NVTX_SCOPED_RANGE(nvtx_gemma4_mtp_verify, "Gemma4MTPDecoder::runBaseVerification", nvtx_colors::MAGENTA);

    int32_t const activeBatchSize = context.activeBatchSize;
    int32_t const draftSteps = mRuntime.deployment.specConfig->draftSteps;
    int32_t const baseHiddenSize = mRuntime.deployment.specConfig->baseOutputHiddenDim;

    if (mUseTree)
    {
        buildTreeVerifyInputs(activeBatchSize, context.stream);
    }
    else
    {
        check::check(mVerifyTokenIds.reshape({activeBatchSize, mRuntime.deployment.specConfig->verifyTokenCount}),
            "Tensor reshape failed");
        kernel::launchGemma4MTPBuildVerifyTokens(mSeedTokenIds.dataPointer<int32_t>(),
            mDraftTokenIds.dataPointer<int32_t>(), mVerifyTokenIds.dataPointer<int32_t>(), activeBatchSize, draftSteps,
            context.stream);
    }
    finishVerifyInput(context);

    SpecVerifyInputs const verify = roundVerifyInputs(activeBatchSize);
    int32_t const verifyTokenCount = verify.tokensPerRequest;
    ELLM_CHECK(!mRuntime.deployment.base.gemmaPleEnabled || mRuntime.preprocess.gemma4Ple,
        "Gemma4 MTP base requires its configured PLE preprocessor");
    spec_decode::prepareTargetVerifyInputs(verify, mRuntime, context, 0);
    auto const verifyDims = mRuntime.deployment.base.specVerifyDims(activeBatchSize, verifyTokenCount, verify.layout);
    bool verifySuccess;
    {
        auto forward
            = gTimer.startForward("base", verifyDims.batch, verifyDims.seqLen, verifyDims.selectLen, context.stream);
        verifySuccess
            = mRuntime.base.executor.prepare(kDecodeProfile, verifyDims, mRuntime.base.tensorMap, context.stream);
        if (verifySuccess)
        {
            verifySuccess = mRuntime.base.executor.execute(context.stream);
        }
    }
    if (!verifySuccess)
    {
        LOG_ERROR("Gemma4 MTP base verification execution failed.");
        return false;
    }

    check::check(mRuntime.base.pipelineIO.outputLogits.reshape(
                     {activeBatchSize, verifyTokenCount, mRuntime.deployment.base.outputVocabSize}),
        "Tensor reshape failed");
    // GCOVR_EXCL_START
    if (context.hasLogitBias)
    {
        applyLogitBiasRepeatedRows(
            mRuntime.logitBias, mRuntime.base.pipelineIO.outputLogits, context, verifyTokenCount, context.stream);
    }
    // GCOVR_EXCL_STOP
    check::check(
        mRuntime.base.pipelineIO.baseHiddenStates.reshape({activeBatchSize * verifyTokenCount, baseHiddenSize}),
        "Tensor reshape failed");

    return true;
}

void Gemma4MTPDecoder::finishVerifyInput(DecodingInferenceContext& context)
{
    auto& io = mRuntime.base.pipelineIO;
    int32_t const activeBatchSize = context.activeBatchSize;
    mJump.reset();
    SpecVerifyInputs const drafted = roundVerifyInputs(activeBatchSize);

    // A slot that may jump has its draft walked on the host before verification; a round that grafts a chain verifies
    // at the grafting width and pads the other slots.
    using Walk = JumpGraftSession::Walk;
    Walk const walk = mJump.walk(context, drafted);
    if (mJump.wide())
    {
        mJump.grafter().upload(mUseTree ? mRuntime.preprocess.idsInput : mVerifyTokenIds,
            mUseTree ? &io.specTreeParentIds : nullptr, mUseTree ? &io.specTreeDepths : nullptr, context.stream);
        if (mUseTree)
        {
            mJump.prepareGraftedTreeDescriptors(
                io, mRuntime.base.cacheManager.getKVCacheLengths(), activeBatchSize, context.stream);
        }
    }

    // After a walk that grafted nothing the guided decoder already holds the draft; a grafting round captures the
    // grafted input with its node counts, which end each slot's mask walk.
    if (context.hasGuidedDecoding && walk != Walk::kNoChain)
    {
        SpecVerifyInputs captured = roundVerifyInputs(activeBatchSize);
        captured.forcedFlags = mJump.forcedFlags();
        mRuntime.guidedDecoder.captureDraft(captured, context.stream);
    }
}

SpecVerifyInputs Gemma4MTPDecoder::roundVerifyInputs(int32_t activeBatchSize) const
{
    auto const& config = *mRuntime.deployment.specConfig;
    auto& io = mRuntime.base.pipelineIO;
    // A grafted tree lives only in idsInput; mTreeTokenIds holds the tree as drafted.
    Tensor const& tokenIds
        = !mUseTree ? mVerifyTokenIds : (mJump.wide() ? mRuntime.preprocess.idsInput : mTreeTokenIds);
    Tensor const* const validCounts = mJump.validCounts(mUseTree ? &mValidCounts : nullptr);
    int32_t const width = mJump.verifyWidth(config.verifyTokenCount);
    return SpecVerifyInputs{tokenIds, validCounts, mUseTree ? &io.specTreeParentIds : nullptr,
        mUseTree ? &io.specTreeDepths : nullptr, activeBatchSize, width, config.maxPathLength, config.verifyLayout,
        mUseTree};
}

bool Gemma4MTPDecoder::acceptAndCommit(DecodingInferenceContext& context)
{
    auto const& config = *mRuntime.deployment.specConfig;
    int32_t const activeBatchSize = context.activeBatchSize;
    // A grafting round verifies the grafted input at the jump width and accepts up to its path capacity. The layout
    // stays the deployment's: a grafted chain is still a chain, and its accepted path a prefix of it.
    SpecVerifyInputs const verify = roundVerifyInputs(activeBatchSize);
    int32_t const verifyTokenCount = verify.tokensPerRequest;
    int32_t const maxAcceptDepth = mJump.acceptDepth(std::min(config.draftSteps + 1, config.verifyTokenCount));
    int32_t const baseHiddenSize = config.baseOutputHiddenDim;
    int32_t const vocabSize = mRuntime.deployment.base.outputVocabSize;

    check::check(mAcceptedTokenIds.reshape({activeBatchSize, maxAcceptDepth}), "Tensor reshape failed");
    check::check(mAcceptedTokenIndices.reshape({activeBatchSize, maxAcceptDepth}), "Tensor reshape failed");
    check::check(mAcceptLength.reshape({activeBatchSize}), "Tensor reshape failed");

    if (context.hasGuidedDecoding)
    {
        applyGuidedDecodingMaskForDraftTree(mRuntime.guidedDecoder, context, mRuntime.base.pipelineIO.outputLogits,
            activeBatchSize, verifyTokenCount, context.stream);
    }

    Tensor targetTokenIds;
    if (mUseTree)
    {
        check::check(mRuntime.base.pipelineIO.outputLogits.reshape({activeBatchSize * verifyTokenCount, vocabSize}),
            "Tensor reshape failed");
        targetTokenIds = kernel::selectSpecTargetTokens(mRuntime.base.pipelineIO.outputLogits, std::nullopt,
            mRuntime.sampling.workspace, activeBatchSize, verifyTokenCount, context.stream);
    }
    else
    {
        check::check(mArgmaxScratch.reshape({activeBatchSize * verifyTokenCount}), "Tensor reshape failed");
        kernel::invokeRowwiseArgmax<float>(mRuntime.base.pipelineIO.outputLogits.dataPointer<float>(),
            activeBatchSize * verifyTokenCount, vocabSize, mArgmaxScratch.dataPointer<int32_t>(), context.stream);
        CUDA_CHECK(cudaGetLastError());
        targetTokenIds = Tensor(mArgmaxScratch.rawPointer(), {activeBatchSize * verifyTokenCount}, DeviceType::kGPU,
            trt_edgellm::DataType::kINT32);
    }
    // A grafting round's forced tokens are read off the accepted rows, which a chain commit does not need otherwise.
    Tensor* const acceptedRows = mUseTree || mJump.wide() ? &mAcceptedTokenIndices : nullptr;
    spec_decode::accept(
        GreedyAcceptInputs{verify, targetTokenIds, verify.executionValidTokenCounts, mJump.forcedFlags()},
        mAcceptedTokenIds, mAcceptLength, acceptedRows, context.stream,
        decoder_utils::forcedAcceptanceLength(mRuntime, context));

    decoder_utils::clampAcceptLengthsToRemainingGeneration(context, mAcceptLength, context.stream);

    spec_decode::commitAcceptedTargetPath(mRuntime, context,
        SpecAcceptResult{mAcceptedTokenIds, mAcceptLength, mUseTree ? &mAcceptedTokenIndices : nullptr, maxAcceptDepth},
        verifyTokenCount);
    check::check(mRuntime.base.pipelineIO.baseHiddenStates.reshape({activeBatchSize * maxAcceptDepth, baseHiddenSize}),
        "Tensor reshape failed");
    // The round's accepted span is maxAcceptDepth rows per slot (wider in a grafting round); the carried copy keeps a
    // fixed stride so the next seed and a compaction read it without knowing the round.
    decoder_utils::copyHiddenRowsStrided(mAcceptedHiddenStates, mAcceptCapacity,
        mRuntime.base.pipelineIO.baseHiddenStates, maxAcceptDepth, activeBatchSize, baseHiddenSize, context.stream);

    if (context.maxNumLogprobs() > 0)
    {
        if (mUseTree)
        {
            int32_t const gatheredRows = activeBatchSize * maxAcceptDepth;
            check::check(mRuntime.logprobs.gatheredLogits.reshape({gatheredRows, vocabSize}), "Tensor reshape failed");
            gatherSpecVerifyAcceptedLogitRows(mRuntime.base.pipelineIO.outputLogits, mAcceptedTokenIndices,
                mRuntime.logprobs.gatheredLogits, activeBatchSize, verifyTokenCount, maxAcceptDepth, vocabSize,
                context.stream);
            decoder_utils::enqueueLogprobsD2H(
                mRuntime.logprobs.gatheredLogits, gatheredRows, mRuntime, context.maxNumLogprobs(), context.stream);
        }
        else
        {
            check::check(mRuntime.base.pipelineIO.outputLogits.reshape({activeBatchSize * verifyTokenCount, vocabSize}),
                "Tensor reshape failed");
            decoder_utils::enqueueLogprobsD2H(mRuntime.base.pipelineIO.outputLogits, activeBatchSize * verifyTokenCount,
                mRuntime, context.maxNumLogprobs(), context.stream);
        }
    }

    std::optional<decoder_utils::ForcedPathInfo> forcedPath = mJump.forcedPath(mAcceptedTokenIndices);
    // A grafting round proposes the plain round's drafts; its forced tokens are neither drafts nor bonus.
    decoder_utils::appendAcceptedTokens(context, mHostAcceptLengths, mHostAcceptedTokenIds, mAcceptLength,
        mAcceptedTokenIds, maxAcceptDepth, mRuntime.tokenizer, context.stream, config.verifyTokenCount - 1, nullptr,
        decoder_utils::forcedAcceptanceLength(mRuntime, context), !mUseTree, forcedPath ? &*forcedPath : nullptr);

    if (context.hasGuidedDecoding)
    {
        advanceGuidedDecodingForCommitted(mRuntime.guidedDecoder, context, mHostAcceptedTokenIds.dataPointer<int32_t>(),
            mHostAcceptLengths.dataPointer<int32_t>(), maxAcceptDepth, activeBatchSize);
    }

    if (context.maxNumLogprobs() > 0)
    {
        decoder_utils::collectSpecLogprobsFromHost(mRuntime, context, activeBatchSize, maxAcceptDepth,
            mHostAcceptLengths.dataPointer<int32_t>(), context.maxNumLogprobs());
    }

    mJump.account(context, mHostAcceptLengths.dataPointer<int32_t>(), mHostAcceptedTokenIds.dataPointer<int32_t>(),
        maxAcceptDepth);

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
