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

#include "runtime/decoding/eagleDecoder.h"
#include "common/bindingNames.h"
#include "common/checkMacros.h"
#include "common/cudaUtils.h"
#include "common/logger.h"
#include "common/mathUtils.h"
#include "common/safetensorsUtils.h"
#include "kernels/embeddingKernels/embeddingKernels.h"
#include "kernels/posEncoding/applyRopeWriteKV.h"
#include "kernels/speculative/batchEvictKernels.h"
#include "kernels/speculative/speculativeKernels.h"
#include "profiling/metrics.h"
#include "profiling/nvtx_wrapper.h"
#include "profiling/timer.h"
#include "runtime/config/llmEngineConfig.h"
#include "runtime/decoding/decoderUtils.h"
#include "runtime/decoding/guidedDecoder.h"
#include "runtime/decoding/logitBias.h"
#include "runtime/decoding/specDecodeUtils.h"
#include "runtime/preprocess/embeddingPreprocessor.h"
#include "sampler/sampling.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <functional>
#include <limits>
#include <optional>
#include <string>
#include <vector>

namespace trt_edgellm
{
namespace rt
{
namespace
{
constexpr int32_t kPrefillProfile{0};
constexpr int32_t kDecodeProfile{1};
} // namespace

EagleDecoder::EagleDecoder(DecodingRuntimeContext& runtime, std::filesystem::path const& engineDir,
    std::unique_ptr<EngineExecutor> draftExecutor, ExternalWeightManager draftWeights, cudaStream_t stream)
    : mRuntime(runtime)
    , mDraftCacheManager(*runtime.base.sharedResources.cacheManagers[1])
    , mDraftExecutor(std::move(draftExecutor))
{
    check::check(mRuntime.deployment.draft.has_value(), "SpecDecode drafting strategy requires a draft model config.");
    check::check(
        mRuntime.deployment.specConfig.has_value(), "SpecDecode drafting strategy requires a drafting config.");

    ELLM_CHECK(mDraftExecutor != nullptr, "EAGLE decoding requires a validated draft engine.");

    int32_t const maxRuntimeBatchSize = mRuntime.maxRuntimeBatchSize;
    int32_t const effectiveMaxDraftProposalSize = mRuntime.deployment.effectiveMaxDraftProposalSize();
    int32_t const effectiveDraftTopK = mRuntime.deployment.specConfig->draftTopK;
    // A jump-forward grafting round accepts up to jumpMaxPathLength tokens, the plain round's path length when the
    // deployment cannot jump.
    int32_t const effectiveMaxAcceptDepth = mRuntime.deployment.specConfig->jumpMaxPathLength;
    int32_t const draftFullTableLength = 1 + effectiveDraftTopK
        + (mRuntime.deployment.specConfig->draftSteps - 1) * effectiveDraftTopK * effectiveDraftTopK;

    mDraftProposalSize = Tensor(
        {maxRuntimeBatchSize}, DeviceType::kGPU, trt_edgellm::DataType::kINT32, "SpecDecode::draftProposalSize");
    mDraftAttentionMask = Tensor({maxRuntimeBatchSize, effectiveMaxDraftProposalSize, effectiveMaxDraftProposalSize},
        DeviceType::kGPU, trt_edgellm::DataType::kINT8, "SpecDecode::draftAttentionMask");

    buildTensorMapForSpecDecodeDraft(
        mDraftTensorMap, mRuntime.base.pipelineIO, mRuntime.base.sharedResources, *mRuntime.deployment.draft);

    // Publish externalized draft-engine weights into the draft tensor map, mirroring the base engine. A no-op
    // when the draft model has no externalized weights.
    mDraftExternalWeightManager = std::move(draftWeights);
    mDraftExternalWeightManager.registerTensorMapEntries(mDraftTensorMap);

    mDraftTokenIdsFullTable = Tensor({maxRuntimeBatchSize, draftFullTableLength}, DeviceType::kGPU,
        trt_edgellm::DataType::kINT32, "SpecDecode::idsFull");
    mDraftTokenScoreFullTable = Tensor({maxRuntimeBatchSize, draftFullTableLength}, DeviceType::kGPU,
        trt_edgellm::DataType::kFLOAT, "SpecDecode::scoreFull");
    mDraftTokenPredecessorFullTable = Tensor({maxRuntimeBatchSize, draftFullTableLength}, DeviceType::kGPU,
        trt_edgellm::DataType::kINT32, "SpecDecode::predFull");
    mDraftVocabMappingTable = Tensor({mRuntime.deployment.draft->outputVocabSize}, DeviceType::kGPU,
        trt_edgellm::DataType::kINT32, "SpecDecode::draftVocabMap");
    mDraftRootTokenId
        = Tensor({maxRuntimeBatchSize}, DeviceType::kGPU, trt_edgellm::DataType::kINT32, "SpecDecode::rootToken");
    mDraftTokenIdsTable = Tensor({maxRuntimeBatchSize, effectiveDraftTopK * effectiveDraftTopK}, DeviceType::kGPU,
        trt_edgellm::DataType::kINT32, "SpecDecode::idsTable");
    mDraftTokenScoresTable = Tensor({maxRuntimeBatchSize, effectiveDraftTopK * effectiveDraftTopK}, DeviceType::kGPU,
        trt_edgellm::DataType::kFLOAT, "SpecDecode::scoresTable");
    mDraftTokenIntermediateScores = Tensor({maxRuntimeBatchSize, effectiveDraftTopK}, DeviceType::kGPU,
        trt_edgellm::DataType::kFLOAT, "SpecDecode::intermediateScores");
    mDraftTokenIntermediateParents = Tensor({maxRuntimeBatchSize, effectiveDraftTopK}, DeviceType::kGPU,
        trt_edgellm::DataType::kINT32, "SpecDecode::intermediateParents");
    mAcceptedTokenIds = Tensor({maxRuntimeBatchSize, effectiveMaxAcceptDepth}, DeviceType::kGPU,
        trt_edgellm::DataType::kINT32, "SpecDecode::acceptedIds");
    mAcceptedTokenIndices = Tensor({maxRuntimeBatchSize, effectiveMaxAcceptDepth}, DeviceType::kGPU,
        trt_edgellm::DataType::kINT32, "SpecDecode::acceptedIndices");
    mAcceptLength
        = Tensor({maxRuntimeBatchSize}, DeviceType::kGPU, trt_edgellm::DataType::kINT32, "SpecDecode::acceptLength");
    mCommonStateTracker.reserve(maxRuntimeBatchSize);
    mHostAcceptLengths = Tensor(
        {maxRuntimeBatchSize}, DeviceType::kCPU, trt_edgellm::DataType::kINT32, "SpecDecode::hostAcceptLengths");
    mHostSeatStaging = Tensor(
        {maxRuntimeBatchSize, 2}, DeviceType::kCPU, trt_edgellm::DataType::kINT32, "SpecDecode::hostSeatStaging");
    mHostAcceptedTokenIds = Tensor({maxRuntimeBatchSize, effectiveMaxAcceptDepth}, DeviceType::kCPU,
        trt_edgellm::DataType::kINT32, "SpecDecode::hostAcceptedIds");
    mAcceptedHiddenStates
        = Tensor({maxRuntimeBatchSize * effectiveMaxAcceptDepth, mRuntime.deployment.specConfig->baseOutputHiddenDim},
            DeviceType::kGPU, trt_edgellm::DataType::kHALF, "SpecDecode::acceptedHidden");

    if (hasCandidateHead())
    {
        // The draft head emits one candidate row per selected row: B for prefill and accept, B x draftTopK for a
        // proposal level. The shared `logits` map entry stays but is not bound: the candidate registry has no logits.
        int32_t const draftCandidateCount = mRuntime.deployment.draft->draftCandidateCount;
        mDraftSupportIds = Tensor({maxRuntimeBatchSize * effectiveDraftTopK, draftCandidateCount}, DeviceType::kGPU,
            trt_edgellm::DataType::kINT32, "SpecDecode::draftSupportIds");
        mDraftSupportLogProbs = Tensor({maxRuntimeBatchSize * effectiveDraftTopK, draftCandidateCount},
            DeviceType::kGPU, trt_edgellm::DataType::kFLOAT, "SpecDecode::draftSupportLogProbs");
        mDraftTensorMap.set(binding_names::kSpecProposalSupportIds, mDraftSupportIds);
        mDraftTensorMap.set(binding_names::kSpecProposalSupportLogProbs, mDraftSupportLogProbs);
    }

    // A chain draft grafts as a tree too: grafted nodes sit past the draft chain.
    mJump.initialize(mRuntime.tokenizer, mRuntime.guidedDecoder, maxRuntimeBatchSize, JumpGraftGeometry::kTree,
        *mRuntime.deployment.specConfig, mRuntime.deployment.specConfig->verifyTokenCount);

    // EAGLE kernels consume draft-token -> target-token offsets. Older
    // reduced-vocab exports already store offsets in d2t.safetensors, while
    // newer/full-vocab EAGLE3 exports store direct target token ids. Normalize
    // both forms once at load time. Candidate-head ids are full-vocabulary ids, so their table stays zero.
    CUDA_CHECK(
        cudaMemsetAsync(mDraftVocabMappingTable.rawPointer(), 0, mDraftVocabMappingTable.getMemoryCapacity(), stream));
    if (!hasCandidateHead())
    {
        auto const d2tPath = engineDir / "d2t.safetensors";
        if (std::filesystem::exists(d2tPath))
        {
            std::vector<Tensor> d2tTensors;
            if (!safetensors::loadSafetensors(d2tPath, d2tTensors, stream))
            {
                LOG_ERROR("Failed to load d2t.safetensors from model directory: %s", engineDir.c_str());
                throw std::runtime_error("Failed to load d2t.safetensors from model directory: " + engineDir.string());
            }
            check::check(d2tTensors.size() == 1, "d2t.safetensors should contain exactly one tensor");
            check::check(d2tTensors[0].getShape().getNumDims() == 1, "d2t tensor should be 1D");
            check::check(d2tTensors[0].getShape()[0] == mRuntime.deployment.draft->outputVocabSize,
                "d2t tensor length should match draft vocab size");
            mDraftVocabMappingTable = std::move(d2tTensors[0]);

            int32_t const draftVocabSize = static_cast<int32_t>(mDraftVocabMappingTable.getShape()[0]);
            int32_t const baseVocabSize = mRuntime.deployment.base.vocabSize;
            std::vector<int32_t> hostD2T(static_cast<size_t>(draftVocabSize));
            CUDA_CHECK(cudaMemcpyAsync(hostD2T.data(), mDraftVocabMappingTable.dataPointer<int32_t>(),
                static_cast<size_t>(draftVocabSize) * sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            bool directMapValid{true};
            bool offsetMapValid{true};
            std::vector<int32_t> offsetTargets(static_cast<size_t>(draftVocabSize));
            for (int32_t draftTokenId = 0; draftTokenId < draftVocabSize; ++draftTokenId)
            {
                int32_t const rawValue = hostD2T[static_cast<size_t>(draftTokenId)];
                directMapValid = directMapValid && rawValue >= 0 && rawValue < baseVocabSize;
                int64_t const offsetTarget = static_cast<int64_t>(draftTokenId) + rawValue;
                bool const offsetValid = offsetTarget >= 0 && offsetTarget < baseVocabSize;
                offsetMapValid = offsetMapValid && offsetValid;
                if (offsetValid)
                {
                    offsetTargets[static_cast<size_t>(draftTokenId)] = static_cast<int32_t>(offsetTarget);
                }
            }
            check::check(
                directMapValid || offsetMapValid, "EAGLE d2t table is neither a valid direct map nor offset map.");

            auto uniqueCount = [](std::vector<int32_t> values) {
                std::sort(values.begin(), values.end());
                return static_cast<int64_t>(std::unique(values.begin(), values.end()) - values.begin());
            };
            bool useOffsetMap{false};
            if (offsetMapValid && !directMapValid)
            {
                useOffsetMap = true;
            }
            else if (offsetMapValid && directMapValid)
            {
                int64_t const directUniqueCount = uniqueCount(hostD2T);
                int64_t const offsetUniqueCount = uniqueCount(offsetTargets);
                useOffsetMap = offsetUniqueCount > directUniqueCount;
            }

            if (!useOffsetMap)
            {
                for (int32_t draftTokenId = 0; draftTokenId < draftVocabSize; ++draftTokenId)
                {
                    int32_t const targetTokenId = hostD2T[static_cast<size_t>(draftTokenId)];
                    hostD2T[static_cast<size_t>(draftTokenId)] = targetTokenId - draftTokenId;
                }
            }
            LOG_INFO("Loaded EAGLE d2t.safetensors as %s table", useOffsetMap ? "offset" : "direct");
            CUDA_CHECK(cudaMemcpyAsync(mDraftVocabMappingTable.dataPointer<int32_t>(), hostD2T.data(),
                static_cast<size_t>(draftVocabSize) * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
        }
    }
}

DecodingKvHeadroomPolicy EagleDecoder::kvHeadroom() const
{
    auto const& config = *mRuntime.deployment.specConfig;
    int64_t const draftExtra = static_cast<int64_t>(config.draftSteps) * config.draftTopK;
    ELLM_CHECK(config.verifyTokenCount > 0 && draftExtra > 0
            && draftExtra <= static_cast<int64_t>(std::numeric_limits<int32_t>::max()),
        "EAGLE KV headroom is outside the supported range");
    // A grafting round verifies jumpVerifyTokenCount nodes, and the draft accept step after it binds jumpMaxPathLength
    // rows behind a draft length one accepted path short of the base's.
    int64_t const jumpDraftExtra = std::max<int64_t>(draftExtra, config.jumpMaxPathLength - 1);
    ELLM_CHECK(config.jumpVerifyTokenCount >= config.verifyTokenCount && config.verifyTokenCount > 0
            && jumpDraftExtra > 0 && jumpDraftExtra <= static_cast<int64_t>(std::numeric_limits<int32_t>::max()),
        "EAGLE jump-forward KV headroom is outside the supported range");
    return {/*.plain=*/{config.verifyTokenCount, static_cast<int32_t>(draftExtra)},
        /*.jumpForward=*/{config.jumpVerifyTokenCount, static_cast<int32_t>(jumpDraftExtra)}};
}

int64_t EagleDecoder::getRequiredContextMemorySize() const noexcept
{
    return mDraftExecutor ? mDraftExecutor->getRequiredContextMemorySize() : 0;
}

void EagleDecoder::setContextMemory(Tensor& memory)
{
    if (mDraftExecutor)
    {
        mDraftExecutor->setContextMemory(memory);
    }
}

bool EagleDecoder::hasCandidateHead() const noexcept
{
    return mRuntime.deployment.draft->draftCandidateCount > 0;
}

void EagleDecoder::reshapeDraftHeadOutputs(int64_t rows)
{
    if (hasCandidateHead())
    {
        int64_t const draftCandidateCount = mRuntime.deployment.draft->draftCandidateCount;
        check::check(mDraftSupportIds.reshape({rows, draftCandidateCount})
                && mDraftSupportLogProbs.reshape({rows, draftCandidateCount}),
            "Tensor reshape failed");
        return;
    }
    check::check(mRuntime.base.pipelineIO.outputLogits.reshape({rows, mRuntime.deployment.draft->outputVocabSize}),
        "Tensor reshape failed");
}

void EagleDecoder::selectDraftChildren(int32_t rows, int32_t draftTopK, cudaStream_t stream)
{
    check::check(mRuntime.sampling.indices.reshape({rows, draftTopK}), "Tensor reshape failed");
    check::check(mRuntime.sampling.scores.reshape({rows, draftTopK}), "Tensor reshape failed");
    reshapeDraftHeadOutputs(rows);
    if (hasCandidateHead())
    {
        kernel::sliceCandidateTopK(mDraftSupportIds, mDraftSupportLogProbs, draftTopK, mRuntime.sampling.indices,
            mRuntime.sampling.scores, stream);
        return;
    }
    selectAllTopK(mRuntime.base.pipelineIO.outputLogits, std::ref(mRuntime.sampling.scores), mRuntime.sampling.indices,
        draftTopK, mRuntime.sampling.workspace, stream);
}

bool EagleDecoder::decodeStep(DecodingInferenceContext& context)
{
    // initializeForGeneration may already have run the draft prefill ahead of a context-cache publication; its
    // outputs then feed this round's proposal directly.
    bool const draftPrefillRanAhead = mCommonStateTracker.draftPrefillOutputsPending();
    if (!draftPrefillRanAhead && !mDraftPrimed)
    {
        if (!runDraftModelPrefill(context))
        {
            LOG_ERROR("Failed to execute prefill step for draft model.");
            return false;
        }
    }
    else if (!draftPrefillRanAhead && !runDraftModelAcceptToken(context))
    {
        LOG_ERROR("Failed to execute accept token step for draft model.");
        return false;
    }
    mCommonStateTracker.materializePending(context.activeBatchSize);

    if (!constructDraftProposal(context))
    {
        LOG_ERROR("Failed to construct draft proposal.");
        return false;
    }
    mCommonStateTracker.consumeDraftPrefillOutputs();

    if (!runBaseModelVerification(context))
    {
        LOG_ERROR("Failed to verify draft proposal with base model.");
        return false;
    }
    return true;
}

bool EagleDecoder::initializeForGeneration(DecodingInferenceContext& context)
{
    if (context.speculativeDraftPrefillComplete)
    {
        return true;
    }
    mCommonStateTracker.initialize(context);

    if (!runDraftModelPrefill(context))
    {
        LOG_ERROR("Failed to initialize the EAGLE draft model for generation.");
        return false;
    }
    mCommonStateTracker.markDraftPrefillOutputsPending();
    context.speculativeDraftPrefillComplete = true;
    return true;
}

void EagleDecoder::beginBatch() noexcept
{
    mDraftPrimed = false;
}

void EagleDecoder::appendRow(int32_t executionRow) noexcept
{
    // Within capacity by construction: the session appends no row past the runtime batch.
    check::check(mAcceptedTokenIds.reshape({executionRow + 1, mAcceptedTokenIds.getShape()[1]})
            && mAcceptLength.reshape({executionRow + 1}),
        "Tensor reshape failed");
    mCommonStateTracker.appendRow(executionRow);
}

bool EagleDecoder::seatRow(DecodingInferenceContext& context, int32_t executionRow)
{
    int32_t const maxAcceptDepth = static_cast<int32_t>(mAcceptedTokenIds.getShape()[1]);
    // The draft prefill covers the prompt but its last position, which the next accept pass takes with the first
    // sampled token, like a resident's accepted span.
    if (context.effectivePrefillLengths[0] > 1 && !runDraftModelPrefill(context, /*heldBackTokens=*/1))
    {
        LOG_ERROR("Failed to prefill the EAGLE draft model for a seated row.");
        return false;
    }
    int32_t const prefillTokens = context.effectivePrefillLengths[0];
    check::check(mRuntime.base.pipelineIO.baseHiddenStates.reshape(
                     {prefillTokens, mRuntime.deployment.specConfig->baseOutputHiddenDim}),
        "Tensor reshape failed");
    decoder_utils::seatAcceptedSpan(context, executionRow, mAcceptedTokenIds, mAcceptLength, mHostSeatStaging,
        mAcceptedHiddenStates, maxAcceptDepth, mRuntime.base.pipelineIO.baseHiddenStates, context.stream);
    // Both models now hold the prompt but its last position, which the next decode step's draft pass takes.
    mCommonStateTracker.seatRow(
        executionRow, static_cast<int32_t>(context.rawBatchedInputIds[0].size()) - 1, /*pendingLength=*/1);
    mDraftPrimed = true;
    return true;
}

bool EagleDecoder::prefillSeatedChunk(DecodingInferenceContext& context)
{
    return runDraftModelPrefill(context);
}

std::vector<int32_t> const& EagleDecoder::commonMaterializedStateLengths() const noexcept
{
    return mCommonStateTracker.commonMaterializedStateLengths();
}

bool EagleDecoder::runDraftModelPrefill(DecodingInferenceContext& context, int32_t heldBackTokens)
{
    mRuntime.base.pipelineIO.waitForStepHostStaging();
    assert(mDraftExecutor != nullptr);
    assert(mRuntime.deployment.specConfig.has_value());
    assert(mRuntime.deployment.draft.has_value());
    TIME_STAGE(metrics::StageNames::kSPEC_DECODE_DRAFT_PREFILL, context.stream);
    NVTX_SCOPED_RANGE(nvtx_draft_prefill,
        ("SPEC_DECODE_DRAFT_PREFILL[R" + std::to_string(context.generationRound) + ","
            + std::to_string(context.activeBatchSize) + "]")
            .c_str(),
        nvtx_colors::DARK_ORANGE);

    int32_t const activeBatchSize = context.activeBatchSize;
    int32_t const draftHiddenSize = mRuntime.deployment.specConfig->draftHiddenSize;
    // Over the active batch: a seated row's prefill runs as a batch of one while the per-slot vectors keep every
    // resident's entry.
    int32_t const targetPrefillLength = *std::max_element(
        context.effectivePrefillLengths.begin(), context.effectivePrefillLengths.begin() + activeBatchSize);
    // Held-back positions trail each row; only a seated row (a batch of one) holds any back, so the rows the pass
    // reads stay a prefix of the target's prefill outputs.
    check::check(heldBackTokens == 0 || activeBatchSize == 1, "Only a seated row's draft prefill holds positions back");
    int32_t const inputIdsLength = targetPrefillLength - heldBackTokens;

    check::check(mRuntime.base.pipelineIO.baseHiddenStates.getShape().getNumDims() == 2
            && mRuntime.base.pipelineIO.baseHiddenStates.getShape()[0] == activeBatchSize * targetPrefillLength,
        "BaseHiddenStates token dimension shall match the physical prefill token count.");

    check::check(mRuntime.preprocess.idsInput.reshape({activeBatchSize, inputIdsLength}), "Tensor reshape failed");
    check::check(
        mRuntime.base.pipelineIO.draftHiddenStatesIn.reshape({activeBatchSize, inputIdsLength, draftHiddenSize}),
        "Tensor reshape failed");
    reshapeDraftHeadOutputs(activeBatchSize);
    check::check(mRuntime.base.pipelineIO.draftHiddenStatesOut.reshape({activeBatchSize, draftHiddenSize}),
        "Tensor reshape failed");
    check::check(mRuntime.base.pipelineIO.hostContextLengths.reshape({activeBatchSize}), "Tensor reshape failed");

    decoder_utils::zeroActiveRegion(mRuntime.base.pipelineIO.draftHiddenStatesIn, context.stream);

    check::check(
        mRuntime.sampling.hostPackedTokenIds.reshape({activeBatchSize, inputIdsLength}), "Tensor reshape failed");
    int32_t* hostPackedTokenIdsData = mRuntime.sampling.hostPackedTokenIds.dataPointer<int32_t>();
    std::fill_n(hostPackedTokenIdsData, activeBatchSize * inputIdsLength, 0);
    for (int32_t i = 0; i < activeBatchSize; ++i)
    {
        std::copy(context.tokenIds[i].begin() + 1, context.tokenIds[i].end() - heldBackTokens,
            hostPackedTokenIdsData + i * inputIdsLength);
    }
    CUDA_CHECK(cudaMemcpyAsync(mRuntime.preprocess.idsInput.rawPointer(), hostPackedTokenIdsData,
        activeBatchSize * inputIdsLength * sizeof(int32_t), cudaMemcpyHostToDevice, context.stream));
    mRuntime.base.pipelineIO.recordStepHostUploads(context.stream);

    check::check(mRuntime.base.pipelineIO.inputsEmbeds.reshape({activeBatchSize, inputIdsLength, draftHiddenSize}),
        "Tensor reshape failed");
    // Unified embedding lookup (text, or vision by inserting image embeddings at the image
    // placeholder positions). Reuses the shared EmbeddingPreprocessor, which generates the
    // multimodal indices on-device and shares this decoder's embedding table.
    mRuntime.preprocess.embeddingPreprocessor.embed(
        mRuntime.preprocess.idsInput, context.visualEmbeddings, std::nullopt, mRuntime.base.pipelineIO, context.stream);
    int32_t* ctxLenData = mRuntime.base.pipelineIO.hostContextLengths.dataPointer<int32_t>();
    for (int32_t i = 0; i < activeBatchSize; ++i)
    {
        ctxLenData[i] = context.effectivePrefillLengths[i] - heldBackTokens;
    }
    CUDA_CHECK(cudaMemcpyAsync(mRuntime.base.pipelineIO.contextLengths.rawPointer(), ctxLenData,
        activeBatchSize * sizeof(int32_t), cudaMemcpyHostToDevice, context.stream));
    mRuntime.base.pipelineIO.recordStepHostUploads(context.stream);

    check::check(mRuntime.base.pipelineIO.selectTokenIndices.reshape({activeBatchSize, 1}), "Tensor reshape failed");
    check::check(mRuntime.base.pipelineIO.hostSelectTokenIndices.reshape({activeBatchSize, 1}),
        "hostSelectTokenIndices reshape failed");
    int64_t* selectData = mRuntime.base.pipelineIO.hostSelectTokenIndices.dataPointer<int64_t>();
    for (int32_t i = 0; i < activeBatchSize; ++i)
    {
        selectData[i] = context.effectivePrefillLengths[i] - heldBackTokens - 1;
    }
    CUDA_CHECK(cudaMemcpyAsync(mRuntime.base.pipelineIO.selectTokenIndices.rawPointer(),
        mRuntime.base.pipelineIO.hostSelectTokenIndices.rawPointer(), activeBatchSize * sizeof(int64_t),
        cudaMemcpyHostToDevice, context.stream));
    mRuntime.base.pipelineIO.recordStepHostUploads(context.stream);

    ExecutionPhase const phase = decoder_utils::contextPrefillPhase(context.prefillStartLengths, activeBatchSize);
    auto const prefillDims = mRuntime.deployment.draft->prefillDims(activeBatchSize, inputIdsLength, phase);
    decoder_utils::prepareSpecPrefillRaggedBindings(
        mRuntime, *mRuntime.deployment.draft, 1, context, inputIdsLength, prefillDims, context.stream, heldBackTokens);
    int32_t const physicalTokens = activeBatchSize * inputIdsLength;
    check::check(
        mRuntime.base.pipelineIO.inputsEmbeds.reshape({physicalTokens, draftHiddenSize}), "Tensor reshape failed");
    check::check(mRuntime.base.pipelineIO.baseHiddenStates.reshape(
                     {physicalTokens, mRuntime.deployment.specConfig->baseOutputHiddenDim}),
        "Tensor reshape failed");
    check::check(mRuntime.base.pipelineIO.draftHiddenStatesIn.reshape({physicalTokens, draftHiddenSize}),
        "Tensor reshape failed");
    reshapeDraftHeadOutputs(activeBatchSize);
    check::check(mRuntime.base.pipelineIO.draftHiddenStatesOut.reshape({activeBatchSize, draftHiddenSize}),
        "Tensor reshape failed");

    bool prefillSuccess;
    {
        auto forward = gTimer.startForward(
            "draft", prefillDims.batch, prefillDims.seqLen, prefillDims.selectLen, context.stream);
        prefillSuccess = mDraftExecutor->prepare(kPrefillProfile, prefillDims, mDraftTensorMap, context.stream);
        if (prefillSuccess)
        {
            prefillSuccess = mDraftExecutor->execute(context.stream);
        }
    }
    if (prefillSuccess)
    {
        mDraftCacheManager.commitSequenceLength(mRuntime.base.pipelineIO.contextLengths, context.stream);
        mDraftPrimed = true;
    }
    return prefillSuccess;
}

bool EagleDecoder::constructDraftProposal(DecodingInferenceContext& context)
{
    assert(mDraftExecutor != nullptr);
    assert(mRuntime.deployment.specConfig.has_value());
    assert(mRuntime.deployment.draft.has_value());
    TIME_STAGE(metrics::StageNames::kSPEC_DECODE_DRAFT_PROPOSAL, context.stream);
    NVTX_SCOPED_RANGE(nvtx_construct_proposal,
        ("SPEC_DECODE_DRAFT_PROPOSAL[R" + std::to_string(context.generationRound) + ","
            + std::to_string(context.activeBatchSize) + "]")
            .c_str(),
        nvtx_colors::LIGHT_ORANGE);

    int32_t const activeBatchSize = context.activeBatchSize;
    int32_t const draftHiddenSize = mRuntime.deployment.specConfig->draftHiddenSize;
    int32_t const baseOutputHiddenDim = mRuntime.deployment.specConfig->baseOutputHiddenDim;
    int32_t const draftTopK = mRuntime.deployment.specConfig->draftTopK;
    int32_t const draftFullTableLength = static_cast<int32_t>(mDraftTokenIdsFullTable.getShape()[1]);

    check::check(mDraftTokenIdsFullTable.reshape({activeBatchSize, draftFullTableLength}), "Tensor reshape failed");
    check::check(mDraftTokenScoreFullTable.reshape({activeBatchSize, draftFullTableLength}), "Tensor reshape failed");
    check::check(
        mDraftTokenPredecessorFullTable.reshape({activeBatchSize, draftFullTableLength}), "Tensor reshape failed");
    check::check(mDraftRootTokenId.reshape({activeBatchSize}), "Tensor reshape failed");
    check::check(mDraftTokenIdsTable.reshape({activeBatchSize, draftTopK * draftTopK}), "Tensor reshape failed");
    check::check(mDraftTokenScoresTable.reshape({activeBatchSize, draftTopK * draftTopK}), "Tensor reshape failed");
    check::check(mDraftTokenIntermediateScores.reshape({activeBatchSize, draftTopK}), "Tensor reshape failed");
    check::check(mDraftTokenIntermediateParents.reshape({activeBatchSize, draftTopK}), "Tensor reshape failed");

    std::vector<int32_t> rootTokenIds(activeBatchSize);
    for (int32_t i = 0; i < activeBatchSize; ++i)
    {
        rootTokenIds[i] = context.tokenIds[i].back();
    }
    CUDA_CHECK(cudaMemcpyAsync(mDraftRootTokenId.rawPointer(), rootTokenIds.data(), activeBatchSize * sizeof(int32_t),
        cudaMemcpyHostToDevice, context.stream));

    selectDraftChildren(activeBatchSize, draftTopK, context.stream);

    kernel::initializeDraftTreeTables(mRuntime.sampling.indices, mRuntime.sampling.scores, mDraftRootTokenId,
        mDraftVocabMappingTable, mDraftTokenIdsFullTable, mDraftTokenScoreFullTable, mDraftTokenPredecessorFullTable,
        draftTopK, context.stream);

    int32_t const paddedDraftProposalSize = mRuntime.deployment.specConfig->draftSteps * draftTopK;
    check::check(
        mRuntime.preprocess.idsInput.reshape({activeBatchSize, paddedDraftProposalSize}), "Tensor reshape failed");
    check::check(mRuntime.base.pipelineIO.baseHiddenStates.reshape(
                     {activeBatchSize, paddedDraftProposalSize, baseOutputHiddenDim}),
        "Tensor reshape failed");
    check::check(mRuntime.base.pipelineIO.draftHiddenStatesIn.reshape(
                     {activeBatchSize, paddedDraftProposalSize, draftHiddenSize}),
        "Tensor reshape failed");

    // Must stay below the reshapes: only the bound region is cleared.
    decoder_utils::zeroActiveRegion(mRuntime.base.pipelineIO.baseHiddenStates, context.stream);
    decoder_utils::zeroActiveRegion(mRuntime.base.pipelineIO.draftHiddenStatesIn, context.stream);

    check::check(mDraftProposalSize.reshape({activeBatchSize}), "Tensor reshape failed");
    check::check(mDraftAttentionMask.reshape({activeBatchSize, paddedDraftProposalSize, paddedDraftProposalSize}),
        "Tensor reshape failed");
    check::check(mRuntime.base.pipelineIO.packedAttentionMask.reshape({activeBatchSize, paddedDraftProposalSize,
                     static_cast<int64_t>(divUp(paddedDraftProposalSize, 32))}),
        "Tensor reshape failed");

    kernel::assembleInitialDraftTreeInput(mDraftTokenIdsFullTable, mRuntime.base.pipelineIO.draftHiddenStatesOut,
        mRuntime.preprocess.idsInput, mRuntime.base.pipelineIO.draftHiddenStatesIn, mDraftProposalSize,
        mDraftAttentionMask, draftTopK, context.stream);

    reshapeDraftHeadOutputs(activeBatchSize * draftTopK);
    check::check(mRuntime.base.pipelineIO.draftHiddenStatesOut.reshape({activeBatchSize, draftTopK, draftHiddenSize}),
        "Tensor reshape failed");

    for (int32_t round = 0; round < mRuntime.deployment.specConfig->draftSteps - 1; round++)
    {
        if (round == 0)
        {
            kernel::assembleInitialIntermediateData(mRuntime.sampling.scores, mDraftTokenIntermediateParents,
                mDraftTokenIntermediateScores, draftTopK, context.stream);
        }
        else
        {
            check::check(mRuntime.sampling.indices.reshape({activeBatchSize, draftTopK}), "Tensor reshape failed");
            check::check(mRuntime.sampling.scores.reshape({activeBatchSize, draftTopK}), "Tensor reshape failed");
            selectAllTopK(mDraftTokenScoresTable, std::ref(mRuntime.sampling.scores), mRuntime.sampling.indices,
                draftTopK, mRuntime.sampling.workspace, context.stream);
            check::check(mRuntime.base.pipelineIO.draftHiddenStatesIn.reshape(
                             {activeBatchSize, paddedDraftProposalSize, draftHiddenSize}),
                "Tensor reshape failed");
            kernel::assembleDraftTreeInput(mDraftTokenIdsTable, mRuntime.base.pipelineIO.draftHiddenStatesOut,
                mRuntime.sampling.indices, mRuntime.preprocess.idsInput, mRuntime.base.pipelineIO.draftHiddenStatesIn,
                mDraftProposalSize, mDraftAttentionMask, draftTopK, round, context.stream);
            kernel::assembleIntermediateData(mRuntime.sampling.scores, mRuntime.sampling.indices,
                mDraftTokenIntermediateScores, mDraftTokenIntermediateParents, draftTopK, round, context.stream);
        }

        reshapeDraftHeadOutputs(activeBatchSize * draftTopK);
        check::check(
            mRuntime.base.pipelineIO.draftHiddenStatesOut.reshape({activeBatchSize, draftTopK, draftHiddenSize}),
            "Tensor reshape failed");
        check::check(
            mRuntime.base.pipelineIO.inputsEmbeds.reshape({activeBatchSize, paddedDraftProposalSize, draftHiddenSize}),
            "Tensor reshape failed");
        kernel::embeddingLookup(mRuntime.preprocess.idsInput, mRuntime.preprocess.embedding.table,
            mRuntime.preprocess.embedding.scalesAsOptional(), mRuntime.preprocess.embedding.descriptor,
            mRuntime.base.pipelineIO.inputsEmbeds, context.stream);

        {
            Tensor const& draftKVCacheLengths = mDraftCacheManager.getKVCacheLengths();
            check::check(mRuntime.base.pipelineIO.selectTokenIndices.reshape({activeBatchSize, draftTopK}),
                "Tensor reshape failed");
            check::check(mRuntime.base.pipelineIO.contextLengths.reshape({activeBatchSize}), "Tensor reshape failed");
            check::check(
                mRuntime.base.pipelineIO.specDecodePositionIds.reshape({activeBatchSize, paddedDraftProposalSize}),
                "Tensor reshape failed");
            kernel::prepareSpecDraftProposalInputs(mDraftAttentionMask, mDraftProposalSize, draftKVCacheLengths,
                mRuntime.base.pipelineIO.packedAttentionMask, mRuntime.base.pipelineIO.specDecodePositionIds,
                mRuntime.base.pipelineIO.selectTokenIndices, mRuntime.base.pipelineIO.contextLengths, context.stream);
            decoder_utils::prepareSpecRaggedBindings(mRuntime, *mRuntime.deployment.draft, 1,
                mRuntime.base.pipelineIO.specDecodePositionIds, draftKVCacheLengths, &mDraftProposalSize,
                mRuntime.base.pipelineIO.selectTokenIndices, activeBatchSize * draftTopK, &context.residentRefs,
                activeBatchSize, paddedDraftProposalSize,
                mRuntime.deployment.draft->proposalDims(activeBatchSize, paddedDraftProposalSize, draftTopK),
                context.stream);
        }

        int32_t const proposalTokens = activeBatchSize * paddedDraftProposalSize;
        check::check(
            mRuntime.base.pipelineIO.inputsEmbeds.reshape({proposalTokens, draftHiddenSize}), "Tensor reshape failed");
        check::check(mRuntime.base.pipelineIO.baseHiddenStates.reshape({proposalTokens, baseOutputHiddenDim}),
            "Tensor reshape failed");
        check::check(mRuntime.base.pipelineIO.draftHiddenStatesIn.reshape({proposalTokens, draftHiddenSize}),
            "Tensor reshape failed");
        reshapeDraftHeadOutputs(activeBatchSize * draftTopK);
        check::check(
            mRuntime.base.pipelineIO.draftHiddenStatesOut.reshape({activeBatchSize * draftTopK, draftHiddenSize}),
            "Tensor reshape failed");
        check::check(mRuntime.base.pipelineIO.specDecodePositionIds.reshape({proposalTokens}), "Tensor reshape failed");
        check::check(mRuntime.base.pipelineIO.packedAttentionMask.reshape(
                         {proposalTokens, static_cast<int64_t>(divUp(paddedDraftProposalSize, 32))}),
            "Tensor reshape failed");

        auto const proposalDims
            = mRuntime.deployment.draft->proposalDims(activeBatchSize, paddedDraftProposalSize, draftTopK);
        {
            auto forward = gTimer.startForward(
                "draft", proposalDims.batch, proposalDims.seqLen, proposalDims.selectLen, context.stream);
            check::check(mDraftExecutor->prepare(kDecodeProfile, proposalDims, mDraftTensorMap, context.stream),
                "Failed to prepare draft model for draft proposal step.");
            check::check(
                mDraftExecutor->execute(context.stream), "Failed to execute draft model for draft proposal step.");
        }

        check::check(
            mRuntime.base.pipelineIO.draftHiddenStatesOut.reshape({activeBatchSize * draftTopK, draftHiddenSize}),
            "Tensor reshape failed");
        selectDraftChildren(activeBatchSize * draftTopK, draftTopK, context.stream);

        check::check(
            mRuntime.sampling.indices.reshape({activeBatchSize, draftTopK * draftTopK}), "Tensor reshape failed");
        check::check(
            mRuntime.sampling.scores.reshape({activeBatchSize, draftTopK * draftTopK}), "Tensor reshape failed");
        kernel::computeCuScoresAndTranslateToken(mRuntime.sampling.indices, mRuntime.sampling.scores,
            mDraftTokenIntermediateScores, mDraftVocabMappingTable, mDraftTokenIdsTable, mDraftTokenScoresTable,
            draftTopK, context.stream);
        kernel::updateDraftTreeFullTables(mDraftTokenIdsTable, mDraftTokenScoresTable, mDraftTokenIntermediateParents,
            mDraftTokenIdsFullTable, mDraftTokenScoreFullTable, mDraftTokenPredecessorFullTable, draftTopK, round,
            context.stream);
    }

    int32_t const verifyTokenCount = mRuntime.deployment.specConfig->verifyTokenCount;
    check::check(mRuntime.sampling.indices.reshape({activeBatchSize, verifyTokenCount}), "Tensor reshape failed");
    if (hasCandidateHead() || mRuntime.deployment.base.numLinearAttnLayers > 0)
    {
        // The root must stay at slot 0 even when a child ties its score 0, which candidate log-probs (normalized over
        // Kc entries only) make likely.
        kernel::selectVerifyTreeNodes(mDraftTokenScoreFullTable, verifyTokenCount, mRuntime.sampling.indices,
            std::nullopt, draftTopK, context.stream);
    }
    else
    {
        selectAllTopK(mDraftTokenScoreFullTable, std::nullopt, mRuntime.sampling.indices, verifyTokenCount,
            mRuntime.sampling.workspace, context.stream);
    }

    check::check(mRuntime.preprocess.idsInput.reshape({activeBatchSize, verifyTokenCount}), "Tensor reshape failed");
    check::check(
        mDraftAttentionMask.reshape({activeBatchSize, verifyTokenCount, verifyTokenCount}), "Tensor reshape failed");
    check::check(mRuntime.base.pipelineIO.specTreeParentIds.reshape({activeBatchSize, verifyTokenCount}),
        "Tensor reshape failed");
    check::check(
        mRuntime.base.pipelineIO.specTreeDepths.reshape({activeBatchSize, verifyTokenCount}), "Tensor reshape failed");
    kernel::buildSpecVerificationTree(mDraftTokenIdsFullTable, mDraftTokenPredecessorFullTable,
        mRuntime.sampling.indices, mRuntime.preprocess.idsInput, mDraftAttentionMask,
        std::ref(mRuntime.base.pipelineIO.specTreeParentIds), std::ref(mRuntime.base.pipelineIO.specTreeDepths),
        context.stream);

    // A slot that may jump has its tree walked on the host before verification; a round that grafts a chain widens
    // the verify input to the grafting width and pads the other slots.
    using Walk = JumpGraftSession::Walk;
    SpecVerifyInputs const drafted{mRuntime.preprocess.idsInput, nullptr, &mRuntime.base.pipelineIO.specTreeParentIds,
        nullptr, activeBatchSize, verifyTokenCount, mRuntime.deployment.specConfig->maxPathLength,
        SpecVerifyLayout::kTree};
    Walk const walk = mJump.walk(context, drafted);
    if (mJump.wide())
    {
        mJump.grafter().upload(mRuntime.preprocess.idsInput, &mRuntime.base.pipelineIO.specTreeParentIds,
            &mRuntime.base.pipelineIO.specTreeDepths, context.stream);
    }
    int32_t const verifyWidth = mJump.verifyWidth(verifyTokenCount);
    Tensor const* const validCounts = mJump.validCounts(nullptr);
    Tensor const* const forcedFlags = mJump.forcedFlags();
    check::check(mRuntime.base.pipelineIO.packedAttentionMask.reshape(
                     {activeBatchSize, verifyWidth, static_cast<int64_t>(divUp(verifyWidth, 32))}),
        "Tensor reshape failed");

    // After a walk that grafted nothing the host already holds this round's tree; a wide round captures the grafted
    // tree, padded past each slot's nodes.
    if (context.hasGuidedDecoding && walk != Walk::kNoChain)
    {
        ELLM_CHECK(!mRuntime.guidedDecoder.draftCopyPending(), "The draft tree walk left its copy unwaited");
        mRuntime.guidedDecoder.captureDraft(
            {mRuntime.preprocess.idsInput, validCounts, &mRuntime.base.pipelineIO.specTreeParentIds, nullptr,
                activeBatchSize, verifyWidth, mRuntime.deployment.specConfig->maxPathLength, SpecVerifyLayout::kTree,
                false, forcedFlags},
            context.stream);
    }

    return true;
}

bool EagleDecoder::runBaseModelVerification(DecodingInferenceContext& context)
{
    assert(mDraftExecutor != nullptr);
    assert(mRuntime.deployment.specConfig.has_value());
    TIME_STAGE(metrics::StageNames::kSPEC_DECODE_BASE_VERIFICATION, context.stream);
    NVTX_SCOPED_RANGE(nvtx_verify,
        ("SPEC_DECODE_VERIFY[R" + std::to_string(context.generationRound) + ","
            + std::to_string(context.activeBatchSize) + "]")
            .c_str(),
        nvtx_colors::MAGENTA);

    auto const& specConfig = *mRuntime.deployment.specConfig;
    int32_t const activeBatchSize = context.activeBatchSize;
    // A wide round verifies the grafted tree and accepts up to the grafted path capacity, as a tree even on a chain
    // deployment: grafted nodes branch off the draft chain, so the base's recurrent layers take the tree recurrence
    // and the accepted path is indexed to move their KV rows into place.
    int32_t const verifyTokenCount = mJump.verifyWidth(specConfig.verifyTokenCount);
    int32_t const maxAcceptDepth = mJump.acceptDepth(specConfig.draftSteps + 1);
    SpecVerifyLayout const roundLayout = mJump.wide() ? SpecVerifyLayout::kTree : specConfig.verifyLayout;
    SpecVerifyLayout const acceptLayout = roundLayout;
    Tensor const* const validCounts = mJump.validCounts(nullptr);
    Tensor const* const forcedFlags = mJump.forcedFlags();
    Tensor const& treeMask = mJump.wide() ? mJump.grafter().mask() : mDraftAttentionMask;

    check::check(mRuntime.preprocess.idsInput.getShape()[0] == activeBatchSize
            && mRuntime.preprocess.idsInput.getShape()[1] == verifyTokenCount,
        "IdsInput shall have shape [batch_size, verify_size]");
    check::check(treeMask.getShape()[0] == activeBatchSize && treeMask.getShape()[1] == verifyTokenCount
            && treeMask.getShape()[2] == verifyTokenCount,
        "Draft attention mask shall have shape [batch_size, verify_size, verify_size]");

    {
        Tensor const& baseKVCacheLengths = mRuntime.base.cacheManager.getKVCacheLengths();
        check::check(mRuntime.base.pipelineIO.selectTokenIndices.reshape({activeBatchSize, verifyTokenCount}),
            "Tensor reshape failed");
        check::check(mRuntime.base.pipelineIO.contextLengths.reshape({activeBatchSize}), "Tensor reshape failed");
        check::check(mRuntime.base.pipelineIO.specDecodePositionIds.reshape({activeBatchSize, verifyTokenCount}),
            "Tensor reshape failed");
        kernel::prepareSpecFullTreeInputs(treeMask, baseKVCacheLengths, mRuntime.base.pipelineIO.packedAttentionMask,
            mRuntime.base.pipelineIO.specDecodePositionIds, mRuntime.base.pipelineIO.selectTokenIndices,
            mRuntime.base.pipelineIO.contextLengths, context.stream);
    }
    SpecVerifyInputs const verifyInputs{mRuntime.preprocess.idsInput, validCounts,
        &mRuntime.base.pipelineIO.specTreeParentIds, &mRuntime.base.pipelineIO.specTreeDepths, activeBatchSize,
        verifyTokenCount, specConfig.maxPathLength, roundLayout, true};
    spec_decode::prepareTargetVerifyInputs(verifyInputs, mRuntime, context, 0);
    auto const verifyDims
        = mRuntime.deployment.base.specVerifyDims(activeBatchSize, verifyTokenCount, verifyInputs.layout);
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
        LOG_ERROR("Failed to execute base verification step for base model.");
        return false;
    }

    // GCOVR_EXCL_START
    if (context.hasLogitBias)
    {
        applyLogitBiasRepeatedRows(
            mRuntime.logitBias, mRuntime.base.pipelineIO.outputLogits, context, verifyTokenCount, context.stream);
    }
    // GCOVR_EXCL_STOP

    if (context.hasGuidedDecoding)
    {
        applyGuidedDecodingMaskForDraftTree(mRuntime.guidedDecoder, context, mRuntime.base.pipelineIO.outputLogits,
            activeBatchSize, verifyTokenCount, context.stream);
    }

    check::check(mAcceptedTokenIds.reshape({activeBatchSize, maxAcceptDepth}), "Tensor reshape failed");
    check::check(mAcceptedTokenIndices.reshape({activeBatchSize, maxAcceptDepth}), "Tensor reshape failed");
    check::check(mAcceptLength.reshape({activeBatchSize}), "Tensor reshape failed");

    OptionalInputTensor vocabMappingTable = (mRuntime.deployment.base.reducedVocabSize > 0)
        ? std::optional{std::ref(mRuntime.sampling.baseVocabMappingTable)}
        : std::nullopt;
    Tensor const targetTokenIds = kernel::selectSpecTargetTokens(mRuntime.base.pipelineIO.outputLogits,
        vocabMappingTable, mRuntime.sampling.workspace, activeBatchSize, verifyTokenCount, context.stream);
    SpecVerifyInputs const verify{mRuntime.preprocess.idsInput, validCounts,
        &mRuntime.base.pipelineIO.specTreeParentIds, &mRuntime.base.pipelineIO.specTreeDepths, activeBatchSize,
        verifyTokenCount, maxAcceptDepth, acceptLayout, true};
    spec_decode::accept(GreedyAcceptInputs{verify, targetTokenIds, validCounts, forcedFlags}, mAcceptedTokenIds,
        mAcceptLength, &mAcceptedTokenIndices, context.stream,
        decoder_utils::forcedAcceptanceLength(mRuntime, context));

    decoder_utils::clampAcceptLengthsToRemainingGeneration(context, mAcceptLength, context.stream);

    spec_decode::commitAcceptedTargetPath(mRuntime, context,
        SpecAcceptResult{mAcceptedTokenIds, mAcceptLength, &mAcceptedTokenIndices, maxAcceptDepth}, verifyTokenCount,
        acceptLayout);
    decoder_utils::copyLeadingElements(mAcceptedHiddenStates, mRuntime.base.pipelineIO.baseHiddenStates,
        mRuntime.base.pipelineIO.baseHiddenStates.getShape().volume(), context.stream);

    // Enqueue logprobs device work + D2H before appendAcceptedTokens so everything rides
    // that call's single round synchronization.
    if (context.maxNumLogprobs() > 0)
    {
        int32_t const vocabSize = mRuntime.deployment.base.outputVocabSize;
        int32_t const gatheredRows = activeBatchSize * maxAcceptDepth;
        check::check(mRuntime.logprobs.gatheredLogits.reshape({gatheredRows, vocabSize}), "Tensor reshape failed");
        gatherSpecVerifyAcceptedLogitRows(mRuntime.base.pipelineIO.outputLogits, mAcceptedTokenIndices,
            mRuntime.logprobs.gatheredLogits, activeBatchSize, verifyTokenCount, maxAcceptDepth, vocabSize,
            context.stream);
        decoder_utils::enqueueLogprobsD2H(
            mRuntime.logprobs.gatheredLogits, gatheredRows, mRuntime, context.maxNumLogprobs(), context.stream);
    }

    std::optional<decoder_utils::ForcedPathInfo> forcedPath = mJump.forcedPath(mAcceptedTokenIndices);
    decoder_utils::appendAcceptedTokens(context, mHostAcceptLengths, mHostAcceptedTokenIds, mAcceptLength,
        mAcceptedTokenIds, maxAcceptDepth, mRuntime.tokenizer, context.stream, specConfig.verifyTokenCount - 1, nullptr,
        decoder_utils::forcedAcceptanceLength(mRuntime, context), true, forcedPath ? &*forcedPath : nullptr);

    if (context.hasGuidedDecoding)
    {
        advanceGuidedDecodingForCommitted(mRuntime.guidedDecoder, context, mHostAcceptedTokenIds.dataPointer<int32_t>(),
            mHostAcceptLengths.dataPointer<int32_t>(), maxAcceptDepth, activeBatchSize);
    }
    int32_t const* const hostAcceptLengths = mHostAcceptLengths.dataPointer<int32_t>();
    mCommonStateTracker.recordAccepted(hostAcceptLengths, activeBatchSize);

    if (context.maxNumLogprobs() > 0)
    {
        decoder_utils::collectSpecLogprobsFromHost(mRuntime, context, activeBatchSize, maxAcceptDepth,
            mHostAcceptLengths.dataPointer<int32_t>(), context.maxNumLogprobs());
    }

    mJump.account(context, mHostAcceptLengths.dataPointer<int32_t>(), mHostAcceptedTokenIds.dataPointer<int32_t>(),
        maxAcceptDepth);

    return true;
}

bool EagleDecoder::runDraftModelAcceptToken(DecodingInferenceContext& context)
{
    assert(mDraftExecutor != nullptr);
    assert(mRuntime.deployment.specConfig.has_value());
    assert(mRuntime.deployment.draft.has_value());
    NVTX_SCOPED_RANGE(nvtx_draft_accept,
        ("SPEC_DECODE_DRAFT_ACCEPT[R" + std::to_string(context.generationRound) + ","
            + std::to_string(context.activeBatchSize) + "]")
            .c_str(),
        nvtx_colors::YELLOW);
    TIME_STAGE(metrics::StageNames::kSPEC_DECODE_DRAFT_ACCEPT, context.stream);

    int32_t const activeBatchSize = context.activeBatchSize;
    int32_t const draftHiddenSize = mRuntime.deployment.specConfig->draftHiddenSize;
    int64_t const inputIdsLength = mAcceptedTokenIds.getShape()[1];

    check::check(mRuntime.preprocess.idsInput.reshape({activeBatchSize, inputIdsLength}), "Tensor reshape failed");
    check::check(
        mRuntime.base.pipelineIO.draftHiddenStatesIn.reshape({activeBatchSize, inputIdsLength, draftHiddenSize}),
        "Tensor reshape failed");
    reshapeDraftHeadOutputs(activeBatchSize);
    check::check(mRuntime.base.pipelineIO.draftHiddenStatesOut.reshape({activeBatchSize, draftHiddenSize}),
        "Tensor reshape failed");

    decoder_utils::zeroActiveRegion(mRuntime.base.pipelineIO.draftHiddenStatesIn, context.stream);

    for (int32_t i = 0; i < activeBatchSize; ++i)
    {
        CUDA_CHECK(
            cudaMemcpyAsync(static_cast<int32_t*>(mRuntime.preprocess.idsInput.rawPointer()) + i * inputIdsLength,
                static_cast<int32_t*>(mAcceptedTokenIds.rawPointer()) + i * mAcceptedTokenIds.getShape()[1],
                inputIdsLength * sizeof(int32_t), cudaMemcpyDeviceToDevice, context.stream));
    }

    check::check(mRuntime.base.pipelineIO.inputsEmbeds.reshape({activeBatchSize, inputIdsLength, draftHiddenSize}),
        "Tensor reshape failed");
    kernel::embeddingLookup(mRuntime.preprocess.idsInput, mRuntime.preprocess.embedding.table,
        mRuntime.preprocess.embedding.scalesAsOptional(), mRuntime.preprocess.embedding.descriptor,
        mRuntime.base.pipelineIO.inputsEmbeds, context.stream);
    {
        int32_t const acceptedTokenNum = static_cast<int32_t>(inputIdsLength);
        Tensor const& draftKVCacheLengths = mDraftCacheManager.getKVCacheLengths();
        check::check(
            mRuntime.base.pipelineIO.selectTokenIndices.reshape({activeBatchSize, 1}), "Tensor reshape failed");
        check::check(mRuntime.base.pipelineIO.contextLengths.reshape({activeBatchSize}), "Tensor reshape failed");
        check::check(mRuntime.base.pipelineIO.specDecodePositionIds.reshape({activeBatchSize, acceptedTokenNum}),
            "Tensor reshape failed");
        check::check(mRuntime.base.pipelineIO.packedAttentionMask.reshape(
                         {activeBatchSize, acceptedTokenNum, static_cast<int64_t>(divUp(acceptedTokenNum, 32))}),
            "Tensor reshape failed");
        kernel::prepareSpecAcceptedDraftInputs(draftKVCacheLengths, mAcceptLength,
            mRuntime.base.pipelineIO.packedAttentionMask, mRuntime.base.pipelineIO.specDecodePositionIds,
            mRuntime.base.pipelineIO.selectTokenIndices, mRuntime.base.pipelineIO.contextLengths, context.stream);
        decoder_utils::prepareSpecRaggedBindings(mRuntime, *mRuntime.deployment.draft, 1,
            mRuntime.base.pipelineIO.specDecodePositionIds, draftKVCacheLengths, &mAcceptLength,
            mRuntime.base.pipelineIO.selectTokenIndices, activeBatchSize, &context.residentRefs, activeBatchSize,
            static_cast<int32_t>(inputIdsLength),
            mRuntime.deployment.draft->acceptDims(activeBatchSize, inputIdsLength), context.stream);
    }

    int32_t const physicalTokens = activeBatchSize * static_cast<int32_t>(inputIdsLength);
    check::check(
        mRuntime.base.pipelineIO.inputsEmbeds.reshape({physicalTokens, draftHiddenSize}), "Tensor reshape failed");
    check::check(mRuntime.base.pipelineIO.baseHiddenStates.reshape(
                     {physicalTokens, mRuntime.deployment.specConfig->baseOutputHiddenDim}),
        "Tensor reshape failed");
    decoder_utils::copyLeadingElements(mRuntime.base.pipelineIO.baseHiddenStates, mAcceptedHiddenStates,
        mRuntime.base.pipelineIO.baseHiddenStates.getShape().volume(), context.stream);
    check::check(mRuntime.base.pipelineIO.draftHiddenStatesIn.reshape({physicalTokens, draftHiddenSize}),
        "Tensor reshape failed");
    check::check(mRuntime.base.pipelineIO.specDecodePositionIds.reshape({physicalTokens}), "Tensor reshape failed");
    check::check(mRuntime.base.pipelineIO.packedAttentionMask.reshape(
                     {physicalTokens, static_cast<int64_t>(divUp(inputIdsLength, 32))}),
        "Tensor reshape failed");

    auto const acceptDims = mRuntime.deployment.draft->acceptDims(activeBatchSize, inputIdsLength);
    {
        auto forward
            = gTimer.startForward("draft", acceptDims.batch, acceptDims.seqLen, acceptDims.selectLen, context.stream);
        check::check(mDraftExecutor->prepare(kDecodeProfile, acceptDims, mDraftTensorMap, context.stream),
            "Failed to prepare draft model for accept token step.");
        check::check(mDraftExecutor->execute(context.stream), "Failed to execute draft model for accept token step.");
    }
    mDraftCacheManager.commitSequenceLength(mAcceptLength, context.stream);

    return true;
}

bool EagleDecoder::captureCudaGraphs(cudaStream_t stream)
{
    bool draftProposalCaptureStatus{true};
    bool draftAcceptCaptureStatus{true};
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
        mDraftCacheManager.resetForNewSequences(zeroCacheLensTensor, stream);
        if (!mRuntime.base.executor.prepare(
                kDecodeProfile, mRuntime.deployment.base.resetDims(), mRuntime.base.tensorMap, stream))
        {
            LOG_ERROR("failed to reset base executor context during graph-capture teardown");
        }
    }};

    int32_t const draftTopK = mRuntime.deployment.specConfig->draftTopK;
    int32_t const paddedDraftProposalSize = mRuntime.deployment.specConfig->draftSteps * draftTopK;
    int32_t const draftSteps = mRuntime.deployment.specConfig->draftSteps;
    int32_t const draftHiddenSize = mRuntime.deployment.specConfig->draftHiddenSize;
    int32_t const baseOutputHiddenDim = mRuntime.deployment.specConfig->baseOutputHiddenDim;

    // A deployment that can jump also runs wide rounds: the draft re-reads up to jumpMaxPathLength accepted tokens
    // and the base verifies jumpVerifyTokenCount nodes.
    std::vector<int32_t> acceptLengths;
    for (int32_t acceptLength = 1; acceptLength <= draftSteps + 1; ++acceptLength)
    {
        acceptLengths.push_back(acceptLength);
    }
    std::vector<int32_t> verifyWidths{mRuntime.deployment.specConfig->verifyTokenCount};
    mJump.addGraphTiers(verifyWidths, &acceptLengths);

    for (int32_t batchSize = 1; batchSize <= mRuntime.maxRuntimeBatchSize; ++batchSize)
    {
        std::vector<int32_t> simCacheLens(batchSize, kSimulateCacheLength);
        Tensor simCacheLensTensor(simCacheLens.data(), {batchSize}, DeviceType::kCPU, trt_edgellm::DataType::kINT32);
        mRuntime.base.cacheManager.resetForNewSequences(simCacheLensTensor, stream);
        mDraftCacheManager.resetForNewSequences(simCacheLensTensor, stream);
        std::vector<int32_t> simCtxLens(batchSize, kSimulateCacheLength + paddedDraftProposalSize);
        CUDA_CHECK(cudaMemcpyAsync(mRuntime.base.pipelineIO.contextLengths.rawPointer(), simCtxLens.data(),
            batchSize * sizeof(int32_t), cudaMemcpyHostToDevice, stream));

        {
            check::check(mRuntime.base.pipelineIO.baseHiddenStates.reshape(
                             {batchSize, paddedDraftProposalSize, baseOutputHiddenDim}),
                "Tensor reshape failed");
            check::check(mRuntime.base.pipelineIO.draftHiddenStatesIn.reshape(
                             {batchSize, paddedDraftProposalSize, draftHiddenSize}),
                "Tensor reshape failed");
            check::check(mDraftProposalSize.reshape({batchSize}), "Tensor reshape failed");
            check::check(mDraftAttentionMask.reshape({batchSize, paddedDraftProposalSize, paddedDraftProposalSize}),
                "Tensor reshape failed");
            check::check(mRuntime.base.pipelineIO.packedAttentionMask.reshape({batchSize, paddedDraftProposalSize,
                             static_cast<int64_t>(divUp(paddedDraftProposalSize, 32))}),
                "Tensor reshape failed");
            reshapeDraftHeadOutputs(batchSize * draftTopK);
            check::check(mRuntime.base.pipelineIO.draftHiddenStatesOut.reshape({batchSize, draftTopK, draftHiddenSize}),
                "Tensor reshape failed");
            check::check(
                mRuntime.base.pipelineIO.inputsEmbeds.reshape({batchSize, paddedDraftProposalSize, draftHiddenSize}),
                "Tensor reshape failed");

            Tensor const& draftKVCacheLengths = mDraftCacheManager.getKVCacheLengths();
            check::check(
                mRuntime.base.pipelineIO.selectTokenIndices.reshape({batchSize, draftTopK}), "Tensor reshape failed");
            check::check(mRuntime.base.pipelineIO.specDecodePositionIds.reshape({batchSize, paddedDraftProposalSize}),
                "Tensor reshape failed");
            kernel::prepareSpecDraftProposalInputs(mDraftAttentionMask, mDraftProposalSize, draftKVCacheLengths,
                mRuntime.base.pipelineIO.packedAttentionMask, mRuntime.base.pipelineIO.specDecodePositionIds,
                mRuntime.base.pipelineIO.selectTokenIndices, mRuntime.base.pipelineIO.contextLengths, stream);
            decoder_utils::prepareSpecRaggedBindings(mRuntime, *mRuntime.deployment.draft, 1,
                mRuntime.base.pipelineIO.specDecodePositionIds, draftKVCacheLengths, &mDraftProposalSize,
                mRuntime.base.pipelineIO.selectTokenIndices, batchSize * draftTopK, nullptr, batchSize,
                paddedDraftProposalSize,
                mRuntime.deployment.draft->proposalDims(batchSize, paddedDraftProposalSize, draftTopK), stream);
            int32_t const proposalTokens = batchSize * paddedDraftProposalSize;
            check::check(mRuntime.base.pipelineIO.inputsEmbeds.reshape({proposalTokens, draftHiddenSize}),
                "Tensor reshape failed");
            check::check(mRuntime.base.pipelineIO.baseHiddenStates.reshape({proposalTokens, baseOutputHiddenDim}),
                "Tensor reshape failed");
            check::check(mRuntime.base.pipelineIO.draftHiddenStatesIn.reshape({proposalTokens, draftHiddenSize}),
                "Tensor reshape failed");
            reshapeDraftHeadOutputs(batchSize * draftTopK);
            check::check(
                mRuntime.base.pipelineIO.draftHiddenStatesOut.reshape({batchSize * draftTopK, draftHiddenSize}),
                "Tensor reshape failed");
            check::check(
                mRuntime.base.pipelineIO.specDecodePositionIds.reshape({proposalTokens}), "Tensor reshape failed");
            check::check(mRuntime.base.pipelineIO.packedAttentionMask.reshape(
                             {proposalTokens, static_cast<int64_t>(divUp(paddedDraftProposalSize, 32))}),
                "Tensor reshape failed");

            auto const proposalDims
                = mRuntime.deployment.draft->proposalDims(batchSize, paddedDraftProposalSize, draftTopK);
            if (mDraftExecutor->prepare(kDecodeProfile, proposalDims, mDraftTensorMap, stream))
            {
                draftProposalCaptureStatus &= mDraftExecutor->captureGraph(stream);
            }
            else
            {
                draftProposalCaptureStatus = false;
            }
        }

        {
            reshapeDraftHeadOutputs(batchSize);
            check::check(mRuntime.base.pipelineIO.draftHiddenStatesOut.reshape({batchSize, draftHiddenSize}),
                "Tensor reshape failed");
            for (int32_t const acceptLength : acceptLengths)
            {
                check::check(
                    mRuntime.base.pipelineIO.baseHiddenStates.reshape({batchSize, acceptLength, baseOutputHiddenDim}),
                    "Tensor reshape failed");
                check::check(
                    mRuntime.base.pipelineIO.draftHiddenStatesIn.reshape({batchSize, acceptLength, draftHiddenSize}),
                    "Tensor reshape failed");
                check::check(mAcceptLength.reshape({batchSize}), "Tensor reshape failed");
                std::vector<int32_t> acceptLengthsVec(batchSize, acceptLength);
                CUDA_CHECK(cudaMemcpyAsync(mAcceptLength.rawPointer(), acceptLengthsVec.data(),
                    batchSize * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
                check::check(mRuntime.base.pipelineIO.inputsEmbeds.reshape({batchSize, acceptLength, draftHiddenSize}),
                    "Tensor reshape failed");
                check::check(mRuntime.base.pipelineIO.packedAttentionMask.reshape(
                                 {batchSize, acceptLength, static_cast<int64_t>(divUp(acceptLength, 32))}),
                    "Tensor reshape failed");

                Tensor const& draftKVCacheLengths = mDraftCacheManager.getKVCacheLengths();
                check::check(
                    mRuntime.base.pipelineIO.selectTokenIndices.reshape({batchSize, 1}), "Tensor reshape failed");
                check::check(mRuntime.base.pipelineIO.contextLengths.reshape({batchSize}), "Tensor reshape failed");
                check::check(mRuntime.base.pipelineIO.specDecodePositionIds.reshape({batchSize, acceptLength}),
                    "Tensor reshape failed");
                kernel::prepareSpecAcceptedDraftInputs(draftKVCacheLengths, mAcceptLength,
                    mRuntime.base.pipelineIO.packedAttentionMask, mRuntime.base.pipelineIO.specDecodePositionIds,
                    mRuntime.base.pipelineIO.selectTokenIndices, mRuntime.base.pipelineIO.contextLengths, stream);
                decoder_utils::prepareSpecRaggedBindings(mRuntime, *mRuntime.deployment.draft, 1,
                    mRuntime.base.pipelineIO.specDecodePositionIds, draftKVCacheLengths, &mAcceptLength,
                    mRuntime.base.pipelineIO.selectTokenIndices, batchSize, nullptr, batchSize, acceptLength,
                    mRuntime.deployment.draft->acceptDims(batchSize, acceptLength), stream);
                int32_t const acceptTokens = batchSize * acceptLength;
                check::check(mRuntime.base.pipelineIO.inputsEmbeds.reshape({acceptTokens, draftHiddenSize}),
                    "Tensor reshape failed");
                check::check(mRuntime.base.pipelineIO.baseHiddenStates.reshape({acceptTokens, baseOutputHiddenDim}),
                    "Tensor reshape failed");
                check::check(mRuntime.base.pipelineIO.draftHiddenStatesIn.reshape({acceptTokens, draftHiddenSize}),
                    "Tensor reshape failed");
                check::check(
                    mRuntime.base.pipelineIO.specDecodePositionIds.reshape({acceptTokens}), "Tensor reshape failed");
                check::check(mRuntime.base.pipelineIO.packedAttentionMask.reshape(
                                 {acceptTokens, static_cast<int64_t>(divUp(acceptLength, 32))}),
                    "Tensor reshape failed");

                auto const acceptDims = mRuntime.deployment.draft->acceptDims(batchSize, acceptLength);
                if (mDraftExecutor->prepare(kDecodeProfile, acceptDims, mDraftTensorMap, stream))
                {
                    draftAcceptCaptureStatus &= mDraftExecutor->captureGraph(stream);
                }
                else
                {
                    draftAcceptCaptureStatus = false;
                }
            }
        }

        for (int32_t const verifyTokenCount : verifyWidths)
        {
            Tensor& treeMask = verifyTokenCount > verifyWidths.front() ? mJump.grafter().mask() : mDraftAttentionMask;
            check::check(treeMask.reshape({batchSize, verifyTokenCount, verifyTokenCount}), "Tensor reshape failed");
            check::check(mRuntime.base.pipelineIO.packedAttentionMask.reshape(
                             {batchSize, verifyTokenCount, static_cast<int64_t>(divUp(verifyTokenCount, 32))}),
                "Tensor reshape failed");
            check::check(mRuntime.preprocess.idsInput.reshape({batchSize, verifyTokenCount}), "Tensor reshape failed");
            CUDA_CHECK(cudaMemsetAsync(
                mRuntime.preprocess.idsInput.rawPointer(), 0, batchSize * verifyTokenCount * sizeof(int32_t), stream));
            check::check(mRuntime.base.pipelineIO.specTreeParentIds.reshape({batchSize, verifyTokenCount}),
                "Tensor reshape failed");
            check::check(mRuntime.base.pipelineIO.specTreeDepths.reshape({batchSize, verifyTokenCount}),
                "Tensor reshape failed");
            Tensor const& baseKVCacheLengths = mRuntime.base.cacheManager.getKVCacheLengths();
            check::check(mRuntime.base.pipelineIO.selectTokenIndices.reshape({batchSize, verifyTokenCount}),
                "Tensor reshape failed");
            check::check(mRuntime.base.pipelineIO.contextLengths.reshape({batchSize}), "Tensor reshape failed");
            check::check(mRuntime.base.pipelineIO.specDecodePositionIds.reshape({batchSize, verifyTokenCount}),
                "Tensor reshape failed");
            kernel::prepareSpecFullTreeInputs(treeMask, baseKVCacheLengths,
                mRuntime.base.pipelineIO.packedAttentionMask, mRuntime.base.pipelineIO.specDecodePositionIds,
                mRuntime.base.pipelineIO.selectTokenIndices, mRuntime.base.pipelineIO.contextLengths, stream);
            captureContext.activeBatchSize = batchSize;
            // The grafting tier verifies a tree even on a chain deployment, like the round it is replayed for.
            SpecVerifyLayout const tierLayout = verifyTokenCount > verifyWidths.front()
                ? SpecVerifyLayout::kTree
                : mRuntime.deployment.specConfig->verifyLayout;
            SpecVerifyInputs const verifyInputs{mRuntime.preprocess.idsInput, nullptr,
                &mRuntime.base.pipelineIO.specTreeParentIds, &mRuntime.base.pipelineIO.specTreeDepths, batchSize,
                verifyTokenCount, mRuntime.deployment.specConfig->maxPathLength, tierLayout, true};
            spec_decode::prepareTargetVerifyInputs(verifyInputs, mRuntime, captureContext, 0);
            auto const verifyDims
                = mRuntime.deployment.base.specVerifyDims(batchSize, verifyTokenCount, verifyInputs.layout);
            baseVerificationCaptureStatus &= mRuntime.base.captureGraph(verifyDims, stream);
        }
    }

    return draftProposalCaptureStatus && draftAcceptCaptureStatus && baseVerificationCaptureStatus;
}

void EagleDecoder::resetForNewSequences(Tensor& reuseLengths, cudaStream_t stream)
{
    mDraftCacheManager.resetForNewSequences(reuseLengths, stream);
    mCommonStateTracker.reset();
    mDraftPrimed = false;
}

void EagleDecoder::onBatchEvict(std::vector<int32_t> const& batchMapping, int32_t oldActiveBatch,
    int32_t newActiveBatch, Tensor& deviceBatchMapping, cudaStream_t stream)
{
    mCommonStateTracker.compact(batchMapping, oldActiveBatch, newActiveBatch);

    if (mCommonStateTracker.draftPrefillOutputsPending())
    {
        auto compactDraftPrefillOutput = [&](Tensor& tensor, char const* name) {
            auto const shape = tensor.getShape();
            ELLM_CHECK(shape.getNumDims() == 2 && shape[0] == oldActiveBatch,
                std::string("EAGLE ") + name + " does not match the prefill batch");
            if (newActiveBatch > 0)
            {
                kernel::compactTensorBatch(tensor, deviceBatchMapping, tensor, oldActiveBatch, newActiveBatch, stream);
                check::check(
                    tensor.reshape({newActiveBatch, shape[1]}), std::string("Failed to compact EAGLE ") + name);
            }
        };
        if (hasCandidateHead())
        {
            compactDraftPrefillOutput(mDraftSupportIds, "draft-prefill support ids");
            compactDraftPrefillOutput(mDraftSupportLogProbs, "draft-prefill support log-probs");
        }
        else
        {
            compactDraftPrefillOutput(mRuntime.base.pipelineIO.outputLogits, "draft-prefill logits");
        }
        compactDraftPrefillOutput(mRuntime.base.pipelineIO.draftHiddenStatesOut, "draft-prefill hidden state");
    }

    auto const baseHiddenShape = mRuntime.base.pipelineIO.baseHiddenStates.getShape();
    if (mDraftPrimed)
    {
        // Between rounds the carried span is the decoder's own copy; the pipeline buffer is scratch.
        if (newActiveBatch > 0)
        {
            check::check(mAcceptedHiddenStates.reshape(
                             {oldActiveBatch * mAcceptedTokenIds.getShape()[1], mAcceptedHiddenStates.getShape()[1]}),
                "Tensor reshape failed");
            kernel::compactExecutionTensorBatch(
                mAcceptedHiddenStates, deviceBatchMapping, oldActiveBatch, newActiveBatch, stream);
        }
    }
    // Before the first draft prefill (the blocking path's round zero), the pipeline buffer holds the target's
    // prefill hidden states that prefill still has to read.
    else if (baseHiddenShape.getNumDims() == 2 && oldActiveBatch > 0 && baseHiddenShape[0] % oldActiveBatch == 0
        && newActiveBatch > 0)
    {
        int32_t const rowsPerSequence = static_cast<int32_t>(baseHiddenShape[0] / oldActiveBatch);
        int32_t const hiddenSize = static_cast<int32_t>(baseHiddenShape[1]);
        check::check(mRuntime.base.pipelineIO.baseHiddenStates.reshape({oldActiveBatch, rowsPerSequence, hiddenSize}),
            "Tensor reshape failed");
        kernel::compactTensorBatch(mRuntime.base.pipelineIO.baseHiddenStates, deviceBatchMapping,
            mRuntime.base.pipelineIO.baseHiddenStates, oldActiveBatch, newActiveBatch, stream);
        check::check(mRuntime.base.pipelineIO.baseHiddenStates.reshape({newActiveBatch * rowsPerSequence, hiddenSize}),
            "Tensor reshape failed");
    }
    else if (baseHiddenShape.getNumDims() == 3 && baseHiddenShape[0] == oldActiveBatch && newActiveBatch > 0)
    {
        kernel::compactTensorBatch(mRuntime.base.pipelineIO.baseHiddenStates, deviceBatchMapping,
            mRuntime.base.pipelineIO.baseHiddenStates, oldActiveBatch, newActiveBatch, stream);
        auto const dim1 = static_cast<int32_t>(mRuntime.base.pipelineIO.baseHiddenStates.getShape()[1]);
        auto const dim2 = static_cast<int32_t>(mRuntime.base.pipelineIO.baseHiddenStates.getShape()[2]);
        check::check(
            mRuntime.base.pipelineIO.baseHiddenStates.reshape({newActiveBatch, dim1, dim2}), "Tensor reshape failed");
    }

    if (mAcceptedTokenIds.getShape()[0] == oldActiveBatch && newActiveBatch > 0)
    {
        kernel::compactTensorBatch(
            mAcceptedTokenIds, deviceBatchMapping, mAcceptedTokenIds, oldActiveBatch, newActiveBatch, stream);
        auto const maxAcceptDepth = static_cast<int32_t>(mAcceptedTokenIds.getShape()[1]);
        check::check(mAcceptedTokenIds.reshape({newActiveBatch, maxAcceptDepth}), "Tensor reshape failed");
    }

    if (mAcceptLength.getShape()[0] == oldActiveBatch && newActiveBatch > 0)
    {
        kernel::compactTensorBatch(
            mAcceptLength, deviceBatchMapping, mAcceptLength, oldActiveBatch, newActiveBatch, stream);
        check::check(mAcceptLength.reshape({newActiveBatch}), "Tensor reshape failed");
    }
    if (newActiveBatch == 0)
    {
        mDraftPrimed = false;
    }
}

} // namespace rt
} // namespace trt_edgellm
