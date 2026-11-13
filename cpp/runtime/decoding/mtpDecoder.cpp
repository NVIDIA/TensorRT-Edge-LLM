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

#include "runtime/decoding/mtpDecoder.h"
#include "common/bindingNames.h"
#include "common/checkMacros.h"
#include "common/cudaUtils.h"
#include "common/logger.h"
#include "common/mathUtils.h"
#include "common/safetensorsUtils.h"
#include "kernels/dart/dartGatherKernels.h"
#include "kernels/embeddingKernels/embeddingKernels.h"
#include "kernels/posEncoding/applyRopeWriteKV.h"
#include "kernels/speculative/batchEvictKernels.h"
#include "kernels/speculative/ddtreeKernels.h"
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
#include <filesystem>
#include <functional>
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

MTPDecoder::MTPDecoder(DecodingRuntimeContext& runtime, std::filesystem::path const& engineDir,
    std::unique_ptr<EngineExecutor> draftExecutor, ExternalWeightManager draftWeights, cudaStream_t stream)
    : mRuntime(runtime)
    , mDraftCacheManager(*runtime.base.sharedResources.cacheManagers[1])
    , mDraftExecutor(std::move(draftExecutor))
{
    check::check(mRuntime.deployment.draft.has_value(), "SpecDecode drafting strategy requires a draft model config.");
    check::check(
        mRuntime.deployment.specConfig.has_value(), "SpecDecode drafting strategy requires a drafting config.");
    check::check(runtime.deployment.base.specDecodeType == SpecDecodeMode::kMTP,
        "MTP decoding requires a base engine exported with spec_decode_type=mtp and engine_role=base.");

    ELLM_CHECK(mDraftExecutor != nullptr, "MTP decoding requires a validated draft engine.");

    int32_t const maxRuntimeBatchSize = mRuntime.maxRuntimeBatchSize;
    int32_t const effectiveMaxDraftProposalSize = mRuntime.deployment.effectiveMaxDraftProposalSize();
    // Tree drafting (draftTopK > 1) does NOT expand branches with the draft engine:
    // the engine walks one greedy chain (top-1 per round) and keeps each round's full
    // logits row; branching happens afterwards in ddtreeBuild, which picks the top
    // draftTopK candidates per depth row. The EAGLE draft-table machinery below only
    // drives that chain walk, so it is sized with topK = 1 in tree mode; draftTopK
    // itself is consumed as the tree-build candidate fanout (see buildTreeVerifyInputs).
    mUseTree = mRuntime.deployment.specConfig->draftTopK > 1;
    int32_t const effectiveDraftTopK = mUseTree ? 1 : mRuntime.deployment.specConfig->draftTopK;
    // A jump-forward grafting round accepts up to jumpMaxPathLength tokens, and the draft accept step after it re-reads
    // them all.
    int32_t const effectiveMaxAcceptDepth
        = std::max(mRuntime.deployment.specConfig->draftSteps + 1, mRuntime.deployment.specConfig->jumpMaxPathLength);
    int32_t const draftFullTableLength = 1 + effectiveDraftTopK
        + (mRuntime.deployment.specConfig->draftSteps - 1) * effectiveDraftTopK * effectiveDraftTopK;

    mDraftProposalSize = Tensor(
        {maxRuntimeBatchSize}, DeviceType::kGPU, trt_edgellm::DataType::kINT32, "SpecDecode::draftProposalSize");
    mDraftAttentionMask = Tensor({maxRuntimeBatchSize, effectiveMaxDraftProposalSize, effectiveMaxDraftProposalSize},
        DeviceType::kGPU, trt_edgellm::DataType::kINT8, "SpecDecode::draftAttentionMask");

    buildTensorMapForSpecDecodeDraft(
        mDraftTensorMap, mRuntime.base.pipelineIO, mRuntime.base.sharedResources, *mRuntime.deployment.draft);

    // Publish externalized draft-engine weights into the draft tensor map, mirroring the base engine. A no-op
    // when the draft was exported without --externalize-weights.
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

    if (mUseTree)
    {
        int32_t const verifyTokenCount = mRuntime.deployment.specConfig->verifyTokenCount;
        int32_t const draftVocabSize = mRuntime.deployment.draft->outputVocabSize;
        // Tree proposal rows: row 0 is the root placeholder (ignored by the builder),
        // rows 1..draftSteps hold the per-depth candidate logits.
        int32_t const proposalDepthSize = mRuntime.deployment.specConfig->draftSteps + 1;
        mStackedDraftLogits = Tensor({maxRuntimeBatchSize, proposalDepthSize, draftVocabSize}, DeviceType::kGPU,
            trt_edgellm::DataType::kFLOAT, "SpecDecode::stackedDraftLogits");
        mTreeTokenIds = Tensor({maxRuntimeBatchSize, verifyTokenCount}, DeviceType::kGPU, trt_edgellm::DataType::kINT32,
            "SpecDecode::treeTokenIds");
        mTreeNodeScores = Tensor({maxRuntimeBatchSize, verifyTokenCount}, DeviceType::kGPU,
            trt_edgellm::DataType::kFLOAT, "SpecDecode::treeNodeScores");
        mValidCounts
            = Tensor({maxRuntimeBatchSize}, DeviceType::kGPU, trt_edgellm::DataType::kINT32, "SpecDecode::validCounts");
        size_t const buildWorkspaceSize = kernel::getDDTreeBuildWorkspaceSize(maxRuntimeBatchSize, proposalDepthSize,
            verifyTokenCount, draftVocabSize, mRuntime.deployment.specConfig->draftTopK);
        ELLM_CHECK(buildWorkspaceSize > 0, "MTPDecoder: invalid tree-build workspace size.");
        mTreeBuildWorkspace = Tensor({static_cast<int64_t>(buildWorkspaceSize)}, DeviceType::kGPU,
            trt_edgellm::DataType::kUINT8, "SpecDecode::treeBuildWorkspace");
        CUDA_CHECK(
            cudaMemsetAsync(mStackedDraftLogits.rawPointer(), 0, mStackedDraftLogits.getMemoryCapacity(), stream));
        // Zero the root ids as well so the capture-time tree build (which runs on the
        // zero-initialized logits) is fully deterministic.
        CUDA_CHECK(cudaMemsetAsync(mDraftRootTokenId.rawPointer(), 0, mDraftRootTokenId.getMemoryCapacity(), stream));
    }

    // A tree takes forced chains under any node; a chain takes one, which replaces the draft after its node.
    mJump.initialize(mRuntime.tokenizer, mRuntime.guidedDecoder, maxRuntimeBatchSize,
        mUseTree ? JumpGraftGeometry::kTree : JumpGraftGeometry::kChain, *mRuntime.deployment.specConfig,
        mRuntime.deployment.specConfig->verifyTokenCount);

    if (mRuntime.deployment.draft->reducedVocabSize > 0)
    {
        // The sidecar is DIRECT (full = T[i]), while the EAGLE utility kernels consume
        // OFFSETS (full = i + T[i]); normalize it once at load.
        ELLM_CHECK(!mUseTree,
            "MTP tree drafting (draftTopK > 1) does not support a reduced draft vocabulary; use "
            "--specDraftTopK 1 or re-export the draft without --draft-reduced-vocab-dir.");
        auto const draftVocabMapPath = engineDir / binding_names::kDraftVocabMapFileName;
        ELLM_CHECK(std::filesystem::exists(draftVocabMapPath),
            "Draft engine declares reduced_vocab_size > 0 but " + std::string(binding_names::kDraftVocabMapFileName)
                + " is missing from engine directory");
        std::vector<Tensor> vocabMapTensors;
        ELLM_CHECK(safetensors::loadSafetensors(draftVocabMapPath, vocabMapTensors, stream),
            "Failed to load " + std::string(binding_names::kDraftVocabMapFileName) + " from engine directory");
        check::check(vocabMapTensors.size() == 1,
            std::string(binding_names::kDraftVocabMapFileName) + " should contain exactly one tensor");
        // dataPointer<int32_t>() below is an unchecked reinterpret_cast: a
        // wrong-dtype sidecar (e.g. int64) of the right length would pass the
        // shape/range checks and silently corrupt the table, so refuse it here.
        check::check(vocabMapTensors[0].getDataType() == trt_edgellm::DataType::kINT32,
            "draft vocab_map tensor should be INT32");
        check::check(vocabMapTensors[0].getShape().getNumDims() == 1, "draft vocab_map tensor should be 1D");
        int32_t const reducedVocabSize = static_cast<int32_t>(vocabMapTensors[0].getShape()[0]);
        check::check(reducedVocabSize == mRuntime.deployment.draft->outputVocabSize,
            "draft vocab_map tensor length should match the draft model reduced vocab size");

        int32_t const baseVocabSize = mRuntime.deployment.base.vocabSize;
        std::vector<int32_t> hostMap(static_cast<size_t>(reducedVocabSize));
        CUDA_CHECK(cudaMemcpyAsync(hostMap.data(), vocabMapTensors[0].dataPointer<int32_t>(),
            static_cast<size_t>(reducedVocabSize) * sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        decoder_utils::directVocabMapToOffsets(hostMap, baseVocabSize);
        CUDA_CHECK(cudaMemcpyAsync(mDraftVocabMappingTable.dataPointer<int32_t>(), hostMap.data(),
            static_cast<size_t>(reducedVocabSize) * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
        LOG_INFO(
            "MTPDecoder: reduced draft vocabulary active (%d of %d base tokens); "
            "loaded %s as an offset table",
            reducedVocabSize, baseVocabSize, binding_names::kDraftVocabMapFileName);
    }
    else
    {
        // MTP full-vocab: identity vocab mapping (zero-fill offsets)
        CUDA_CHECK(cudaMemsetAsync(
            mDraftVocabMappingTable.rawPointer(), 0, mDraftVocabMappingTable.getMemoryCapacity(), stream));
    }
}

DecodingKvHeadroomPolicy MTPDecoder::kvHeadroom() const
{
    auto const& config = *mRuntime.deployment.specConfig;
    ELLM_CHECK(config.verifyTokenCount > 0 && config.draftSteps > 0, "MTP KV headroom must be positive");
    ELLM_CHECK(config.jumpVerifyTokenCount >= config.verifyTokenCount,
        "MTP jump-forward KV headroom is outside the supported range");
    // A grafting round verifies jumpVerifyTokenCount rows, and the draft accept step after it binds jumpMaxPathLength
    // rows behind a draft length one accepted path short of the base's.
    int32_t const jumpDraftExtra = std::max(config.draftSteps, config.jumpMaxPathLength - 1);
    return {/*.plain=*/{config.verifyTokenCount, config.draftSteps},
        /*.jumpForward=*/{config.jumpVerifyTokenCount, jumpDraftExtra}};
}

int64_t MTPDecoder::getRequiredContextMemorySize() const noexcept
{
    return mDraftExecutor ? mDraftExecutor->getRequiredContextMemorySize() : 0;
}

void MTPDecoder::setContextMemory(Tensor& memory)
{
    if (mDraftExecutor)
    {
        mDraftExecutor->setContextMemory(memory);
    }
}

bool MTPDecoder::initializeForGeneration(DecodingInferenceContext& context)
{
    if (context.speculativeDraftPrefillComplete)
    {
        return true;
    }
    // The common-state tracker must be initialized on every path so that finalizePrefillPublication receives valid
    // materialized state lengths (the coordinator's HybridSpecPolicy asserts non-null). The draft prefill only runs
    // when a hybrid checkpoint was actually restored — bypass and cold-cache requests skip it.
    mCommonStateTracker.initialize(context);
    if (!context.hybridSpecEndpointReuse)
    {
        return true;
    }
    if (!runDraftModelPrefill(context))
    {
        LOG_ERROR("Failed to run the Hybrid+spec pre-publication draft prefill.");
        return false;
    }
    mCommonStateTracker.markDraftPrefillOutputsPending();
    context.speculativeDraftPrefillComplete = true;
    return true;
}

void MTPDecoder::beginBatch() noexcept
{
    mDraftPrimed = false;
}

void MTPDecoder::appendRow(int32_t executionRow) noexcept
{
    // Within capacity by construction: the session appends no row past the runtime batch.
    check::check(mAcceptedTokenIds.reshape({executionRow + 1, mAcceptedTokenIds.getShape()[1]})
            && mAcceptLength.reshape({executionRow + 1}),
        "Tensor reshape failed");
    mCommonStateTracker.appendRow(executionRow);
}

bool MTPDecoder::seatRow(DecodingInferenceContext& context, int32_t executionRow)
{
    int32_t const maxAcceptDepth = static_cast<int32_t>(mAcceptedTokenIds.getShape()[1]);
    // The draft prefill covers the prompt but its last position, which the next accept pass takes with the first
    // sampled token, like a resident's accepted span.
    if (context.effectivePrefillLengths[0] > 1 && !runDraftModelPrefill(context, /*heldBackTokens=*/1))
    {
        LOG_ERROR("Failed to prefill the MTP draft model for a seated row.");
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

bool MTPDecoder::prefillSeatedChunk(DecodingInferenceContext& context)
{
    return runDraftModelPrefill(context);
}

std::vector<int32_t> const& MTPDecoder::commonMaterializedStateLengths() const noexcept
{
    return mCommonStateTracker.commonMaterializedStateLengths();
}

bool MTPDecoder::decodeStep(DecodingInferenceContext& context)
{
    if (mUseTree && ::trt_edgellm::shouldUseNonGreedySampling(context.uniformSamplingParams()))
    {
        LOG_ERROR(
            "MTPDecoder: tree drafting supports greedy decoding only; route the request through vanilla or use "
            "draftTopK=1.");
        return false;
    }
    // Draft KV for a round's accepted tokens is written lazily, by the *next* round's accept-token pass, so the draft
    // cache trails the base cache by the last accepted span (see ContextCacheCommitPolicy::kPrefillStateOnly).
    // The Hybrid+spec endpoint-reuse path already ran the draft prefill in initializeForGeneration; its outputs feed
    // this round's proposal directly.
    bool const draftPrefillRanAhead = mCommonStateTracker.draftPrefillOutputsPending();
    if (!draftPrefillRanAhead && !mDraftPrimed)
    {
        if (!runDraftModelPrefill(context))
        {
            LOG_ERROR("Failed to execute prefill step for draft model.");
            return false;
        }
        context.speculativeDraftPrefillComplete = true;
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
    mCommonStateTracker.recordAccepted(mHostAcceptLengths.dataPointer<int32_t>(), context.activeBatchSize);
    return true;
}

bool MTPDecoder::runDraftModelPrefill(DecodingInferenceContext& context, int32_t heldBackTokens)
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
    int32_t const baseOutputHiddenDim = mRuntime.deployment.specConfig->baseOutputHiddenDim;
    int32_t const draftVocabSize = mRuntime.deployment.draft->outputVocabSize;
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
        mRuntime.base.pipelineIO.draftHiddenStatesIn.reshape({activeBatchSize, inputIdsLength, baseOutputHiddenDim}),
        "Tensor reshape failed");
    check::check(
        mRuntime.base.pipelineIO.outputLogits.reshape({activeBatchSize, draftVocabSize}), "Tensor reshape failed");
    check::check(mRuntime.base.pipelineIO.draftHiddenStatesOut.reshape({activeBatchSize, baseOutputHiddenDim}),
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
    check::check(mRuntime.base.pipelineIO.draftHiddenStatesIn.reshape({physicalTokens, baseOutputHiddenDim}),
        "Tensor reshape failed");
    check::check(
        mRuntime.base.pipelineIO.outputLogits.reshape({activeBatchSize, draftVocabSize}), "Tensor reshape failed");
    check::check(mRuntime.base.pipelineIO.draftHiddenStatesOut.reshape({activeBatchSize, baseOutputHiddenDim}),
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

bool MTPDecoder::constructDraftProposal(DecodingInferenceContext& context)
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
    int32_t const draftVocabSize = mRuntime.deployment.draft->outputVocabSize;
    // Tree mode drafts the chain with topK=1 mechanics; draftTopK only sets the
    // tree-build candidate fanout consumed in buildTreeVerifyInputs.
    int32_t const draftTopK = mUseTree ? 1 : mRuntime.deployment.specConfig->draftTopK;
    int32_t const draftFullTableLength = static_cast<int32_t>(mDraftTokenIdsFullTable.getShape()[1]);

    // Tree mode: stash the current frontier's full logits row per batch as the
    // depth-`depthRow` candidates for the tree builder. outputLogits holds one
    // contiguous [draftVocabSize] FLOAT row per batch entry at every call site.
    auto const copyLogitsRowToStack = [&](int32_t depthRow) {
        int32_t const proposalDepthSize = mRuntime.deployment.specConfig->draftSteps + 1;
        size_t const rowBytes = static_cast<size_t>(draftVocabSize) * sizeof(float);
        CUDA_CHECK(cudaMemcpy2DAsync(
            static_cast<char*>(mStackedDraftLogits.rawPointer()) + static_cast<size_t>(depthRow) * rowBytes,
            static_cast<size_t>(proposalDepthSize) * rowBytes, mRuntime.base.pipelineIO.outputLogits.rawPointer(),
            rowBytes, rowBytes, activeBatchSize, cudaMemcpyDeviceToDevice, context.stream));
    };

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

    if (mUseTree)
    {
        // Depth-1 candidates come "for free" from the logits that predicted the root's
        // successor (base prefill logits on round 0, draft accept-step logits afterwards).
        copyLogitsRowToStack(/*depthRow=*/1);
    }

    check::check(mRuntime.sampling.indices.reshape({activeBatchSize, draftTopK}), "Tensor reshape failed");
    check::check(mRuntime.sampling.scores.reshape({activeBatchSize, draftTopK}), "Tensor reshape failed");
    selectAllTopK(mRuntime.base.pipelineIO.outputLogits, std::ref(mRuntime.sampling.scores), mRuntime.sampling.indices,
        draftTopK, mRuntime.sampling.workspace, context.stream);

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
                     {activeBatchSize, paddedDraftProposalSize, baseOutputHiddenDim}),
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

    check::check(mRuntime.base.pipelineIO.outputLogits.reshape({activeBatchSize, draftTopK, draftVocabSize}),
        "Tensor reshape failed");
    check::check(
        mRuntime.base.pipelineIO.draftHiddenStatesOut.reshape({activeBatchSize, draftTopK, baseOutputHiddenDim}),
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
                             {activeBatchSize, paddedDraftProposalSize, baseOutputHiddenDim}),
                "Tensor reshape failed");
            kernel::assembleDraftTreeInput(mDraftTokenIdsTable, mRuntime.base.pipelineIO.draftHiddenStatesOut,
                mRuntime.sampling.indices, mRuntime.preprocess.idsInput, mRuntime.base.pipelineIO.draftHiddenStatesIn,
                mDraftProposalSize, mDraftAttentionMask, draftTopK, round, context.stream);
            kernel::assembleIntermediateData(mRuntime.sampling.scores, mRuntime.sampling.indices,
                mDraftTokenIntermediateScores, mDraftTokenIntermediateParents, draftTopK, round, context.stream);
        }

        check::check(mRuntime.base.pipelineIO.outputLogits.reshape({activeBatchSize, draftTopK, draftVocabSize}),
            "Tensor reshape failed");
        check::check(
            mRuntime.base.pipelineIO.draftHiddenStatesOut.reshape({activeBatchSize, draftTopK, baseOutputHiddenDim}),
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
        check::check(mRuntime.base.pipelineIO.draftHiddenStatesIn.reshape({proposalTokens, baseOutputHiddenDim}),
            "Tensor reshape failed");
        check::check(mRuntime.base.pipelineIO.outputLogits.reshape({activeBatchSize * draftTopK, draftVocabSize}),
            "Tensor reshape failed");
        check::check(
            mRuntime.base.pipelineIO.draftHiddenStatesOut.reshape({activeBatchSize * draftTopK, baseOutputHiddenDim}),
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

        check::check(mRuntime.base.pipelineIO.outputLogits.reshape({activeBatchSize * draftTopK, draftVocabSize}),
            "Tensor reshape failed");
        if (mUseTree)
        {
            // This round's chain-frontier logits are the depth-(round+2) candidates.
            copyLogitsRowToStack(/*depthRow=*/round + 2);
        }
        check::check(
            mRuntime.base.pipelineIO.draftHiddenStatesOut.reshape({activeBatchSize * draftTopK, baseOutputHiddenDim}),
            "Tensor reshape failed");
        check::check(
            mRuntime.sampling.indices.reshape({activeBatchSize * draftTopK, draftTopK}), "Tensor reshape failed");
        check::check(
            mRuntime.sampling.scores.reshape({activeBatchSize * draftTopK, draftTopK}), "Tensor reshape failed");
        selectAllTopK(mRuntime.base.pipelineIO.outputLogits, std::ref(mRuntime.sampling.scores),
            mRuntime.sampling.indices, draftTopK, mRuntime.sampling.workspace, context.stream);

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

    if (mUseTree)
    {
        // Tree mode: the tree builder consumes the stacked per-depth logits and
        // emits all base-verify inputs directly; the EAGLE full-table selection below
        // is chain-only.
        if (!buildTreeVerifyInputs(activeBatchSize, context.stream))
        {
            return false;
        }
        // Outside buildTreeVerifyInputs on purpose: graph capture calls that function with an unpopulated tree, and
        // copying it would leave the guided decoder's host buffer holding garbage for the first real step.
        finishVerifyInput(context);
        return true;
    }

    check::check(mRuntime.sampling.indices.reshape({activeBatchSize, mRuntime.deployment.specConfig->verifyTokenCount}),
        "Tensor reshape failed");
    selectAllTopK(mDraftTokenScoreFullTable, std::nullopt, mRuntime.sampling.indices,
        mRuntime.deployment.specConfig->verifyTokenCount, mRuntime.sampling.workspace, context.stream);

    check::check(
        mRuntime.preprocess.idsInput.reshape({activeBatchSize, mRuntime.deployment.specConfig->verifyTokenCount}),
        "Tensor reshape failed");
    int64_t const verifyTokenCount = mRuntime.deployment.specConfig->verifyTokenCount;
    int64_t const draftTableStride = mDraftTokenIdsFullTable.getShape()[1];
    // Selected indices are local to each request's draft table.
    for (int32_t batchIdx = 0; batchIdx < activeBatchSize; ++batchIdx)
    {
        kernel::gatherRows(mRuntime.preprocess.idsInput.dataPointer<int32_t>() + batchIdx * verifyTokenCount,
            mDraftTokenIdsFullTable.dataPointer<int32_t>() + batchIdx * draftTableStride,
            mRuntime.sampling.indices.dataPointer<int32_t>() + batchIdx * verifyTokenCount, verifyTokenCount,
            sizeof(int32_t), context.stream);
    }

    finishVerifyInput(context);
    return true;
}

void MTPDecoder::finishVerifyInput(DecodingInferenceContext& context)
{
    auto const& specConfig = *mRuntime.deployment.specConfig;
    auto& io = mRuntime.base.pipelineIO;
    int32_t const activeBatchSize = context.activeBatchSize;
    SpecVerifyLayout const layout = mUseTree ? SpecVerifyLayout::kTree : SpecVerifyLayout::kChain;
    Tensor* const parents = mUseTree ? &io.specTreeParentIds : nullptr;
    SpecVerifyInputs const drafted{mUseTree ? mTreeTokenIds : mRuntime.preprocess.idsInput,
        mUseTree ? &mValidCounts : nullptr, parents, nullptr, activeBatchSize, specConfig.verifyTokenCount,
        specConfig.maxPathLength, layout};

    // A slot that may jump has its draft walked on the host before verification; a round that grafts a chain verifies
    // at the grafting width and pads the other slots.
    using Walk = JumpGraftSession::Walk;
    Walk const walk = mJump.walk(context, drafted);
    if (mJump.wide())
    {
        mJump.grafter().upload(
            mRuntime.preprocess.idsInput, parents, mUseTree ? &io.specTreeDepths : nullptr, context.stream);
        if (mUseTree)
        {
            mJump.prepareGraftedTreeDescriptors(
                io, mRuntime.base.cacheManager.getKVCacheLengths(), activeBatchSize, context.stream);
        }
    }

    // Copied while the stream still holds only drafting work, so the mask fill waits on just this copy instead of on
    // the verify forward enqueued after it. After a walk that grafted nothing the guided decoder already holds the
    // draft; a grafting round captures the grafted input with its node counts, which end each slot's mask walk.
    if (context.hasGuidedDecoding && walk != Walk::kNoChain)
    {
        mRuntime.guidedDecoder.captureDraft(mJump.wide()
                ? SpecVerifyInputs{mRuntime.preprocess.idsInput, mJump.grafter().validCounts(), parents, nullptr,
                      activeBatchSize, specConfig.jumpVerifyTokenCount, specConfig.maxPathLength, layout, false,
                      mJump.grafter().forcedFlags()}
                : drafted,
            context.stream);
    }
}

bool MTPDecoder::buildTreeVerifyInputs(int32_t activeBatchSize, cudaStream_t stream)
{
    NVTX_SCOPED_RANGE(nvtx_mtp_tree_build, "MTPDecoder::buildTreeVerifyInputs", nvtx_colors::LIGHT_ORANGE);

    int32_t const verifyTokenCount = mRuntime.deployment.specConfig->verifyTokenCount;
    int32_t const proposalDepthSize = mRuntime.deployment.specConfig->draftSteps + 1;
    int32_t const draftVocabSize = mRuntime.deployment.draft->outputVocabSize;

    check::check(
        mStackedDraftLogits.reshape({activeBatchSize, proposalDepthSize, draftVocabSize}), "Tensor reshape failed");
    // Also reshape here (not only in constructDraftProposal): graph capture calls this
    // directly, and ddtreeBuild requires rootTokenIds to be [batch].
    check::check(mDraftRootTokenId.reshape({activeBatchSize}), "Tensor reshape failed");
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

    // Reduced-vocabulary MTP is chain-only. ddtreeBuild expects a DIRECT map
    // (full = T[i]), while mDraftVocabMappingTable holds OFFSETS (full = i + T[i]).
    Tensor const& baseKVCacheLengths = mRuntime.base.cacheManager.getKVCacheLengths();
    kernel::DDTreeBuildParams const buildParams{{mStackedDraftLogits, mDraftRootTokenId, baseKVCacheLengths, nullptr},
        {mTreeTokenIds, mRuntime.base.pipelineIO.specTreeDepths, mRuntime.base.pipelineIO.specTreeParentIds,
            mTreeNodeScores, mValidCounts, mRuntime.preprocess.idsInput, mRuntime.base.pipelineIO.specDecodePositionIds,
            mRuntime.base.pipelineIO.packedAttentionMask, mRuntime.base.pipelineIO.contextLengths,
            mRuntime.base.pipelineIO.selectTokenIndices},
        mRuntime.deployment.specConfig->draftTopK, mTreeBuildWorkspace.rawPointer(),
        static_cast<size_t>(mTreeBuildWorkspace.getMemoryCapacity()), stream};
    kernel::ddtreeBuild(buildParams);

    return true;
}

bool MTPDecoder::runBaseModelVerification(DecodingInferenceContext& context)
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
    // A grafting round verifies the grafted input at the jump width and accepts up to its path capacity. The layout
    // stays the deployment's: a grafted chain is still a chain, and its accepted path a prefix of it.
    int32_t const verifyTokenCount = mJump.verifyWidth(specConfig.verifyTokenCount);
    Tensor const* const validCounts = mJump.validCounts(mUseTree ? &mValidCounts : nullptr);
    Tensor const* const forcedFlags = mJump.forcedFlags();

    check::check(mRuntime.preprocess.idsInput.getShape()[0] == activeBatchSize
            && mRuntime.preprocess.idsInput.getShape()[1] == verifyTokenCount,
        "IdsInput shall have shape [batch_size, verify_size]");

    SpecVerifyInputs const verifyInputs{mRuntime.preprocess.idsInput, validCounts,
        mUseTree ? &mRuntime.base.pipelineIO.specTreeParentIds : nullptr,
        mUseTree ? &mRuntime.base.pipelineIO.specTreeDepths : nullptr, activeBatchSize, verifyTokenCount,
        specConfig.maxPathLength, specConfig.verifyLayout, mUseTree};
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
        // After the logit bias, as in vanilla decode: the grammar is a hard constraint and has to
        // land last. Target selection takes the base's top-1 per row, so masking here is what
        // makes every accepted token grammar-legal.
        applyGuidedDecodingMaskForDraftTree(mRuntime.guidedDecoder, context, mRuntime.base.pipelineIO.outputLogits,
            activeBatchSize, verifyTokenCount, context.stream);
    }

    // A tree with fewer verify nodes than the full chain depth caps the acceptable
    // path length at verifyTokenCount.
    int32_t const chainAcceptDepth = specConfig.draftSteps + 1;
    int32_t const plainAcceptDepth
        = mUseTree ? std::min(chainAcceptDepth, specConfig.verifyTokenCount) : chainAcceptDepth;
    int32_t const maxAcceptDepth = mJump.acceptDepth(plainAcceptDepth);
    check::check(mAcceptedTokenIds.reshape({activeBatchSize, maxAcceptDepth}), "Tensor reshape failed");
    check::check(mAcceptedTokenIndices.reshape({activeBatchSize, maxAcceptDepth}), "Tensor reshape failed");
    check::check(mAcceptLength.reshape({activeBatchSize}), "Tensor reshape failed");

    OptionalInputTensor vocabMappingTable = (mRuntime.deployment.base.reducedVocabSize > 0)
        ? std::optional{std::ref(mRuntime.sampling.baseVocabMappingTable)}
        : std::nullopt;
    // The grafted input lives only in idsInput; mTreeTokenIds holds the tree as drafted.
    Tensor const& acceptTokenIds = mUseTree && !mJump.wide() ? mTreeTokenIds : mRuntime.preprocess.idsInput;
    Tensor const targetTokenIds = kernel::selectSpecTargetTokens(mRuntime.base.pipelineIO.outputLogits,
        vocabMappingTable, mRuntime.sampling.workspace, activeBatchSize, verifyTokenCount, context.stream);
    SpecVerifyInputs const acceptVerify{acceptTokenIds, verifyInputs.executionValidTokenCounts, verifyInputs.parentIds,
        verifyInputs.depths, activeBatchSize, verifyTokenCount, verifyInputs.maxPathLength, verifyInputs.layout,
        verifyInputs.metadataReady};
    spec_decode::accept(GreedyAcceptInputs{acceptVerify, targetTokenIds, validCounts, forcedFlags}, mAcceptedTokenIds,
        mAcceptLength, &mAcceptedTokenIndices, context.stream,
        decoder_utils::forcedAcceptanceLength(mRuntime, context));

    decoder_utils::clampAcceptLengthsToRemainingGeneration(context, mAcceptLength, context.stream);

    // Few-layer-validation teacher forcing (no-op unless EDGELLM_FORCE_TOKENS_FILE is set). Must
    // run before the KV-cache commit below: trimming the acceptance is what keeps a replaced
    // token's stale cache entry from being committed.
    std::vector<int32_t> ownTokens;
    decoder_utils::applyForcedAcceptance(context, mHostAcceptLengths, mHostAcceptedTokenIds, mAcceptLength,
        mAcceptedTokenIds, ownTokens, maxAcceptDepth, context.stream);

    spec_decode::commitAcceptedTargetPath(mRuntime, context,
        SpecAcceptResult{mAcceptedTokenIds, mAcceptLength, &mAcceptedTokenIndices, maxAcceptDepth}, verifyTokenCount);
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
    // A grafting round proposes the plain round's drafts; its forced tokens are neither drafts nor bonus.
    decoder_utils::appendAcceptedTokens(context, mHostAcceptLengths, mHostAcceptedTokenIds, mAcceptLength,
        mAcceptedTokenIds, maxAcceptDepth, mRuntime.tokenizer, context.stream, specConfig.verifyTokenCount - 1, nullptr,
        decoder_utils::forcedAcceptanceLength(mRuntime, context), !mUseTree, forcedPath ? &*forcedPath : nullptr);

    if (context.hasGuidedDecoding)
    {
        // Uses the accept lengths appendAcceptedTokens just rewrote, so the grammar advances over
        // exactly the tokens that reached the output and stops where EOS or max length did.
        advanceGuidedDecodingForCommitted(mRuntime.guidedDecoder, context, mHostAcceptedTokenIds.dataPointer<int32_t>(),
            mHostAcceptLengths.dataPointer<int32_t>(), maxAcceptDepth, activeBatchSize);
    }

    // Few-layer-validation dump (no-op unless EDGELLM_DUMP_LOGITS_KVCACHE_* are set).
    decoder_utils::dumpSpecRound(context, mRuntime.base.cacheManager, *mRuntime.base.sharedResources.kvPageTables[0],
        mRuntime.base.sharedResources.getSwaKVPageTable(0), mRuntime.base.pipelineIO.outputLogits,
        mAcceptedTokenIndices, mHostAcceptLengths, ownTokens, verifyTokenCount, maxAcceptDepth, context.stream);

    if (context.maxNumLogprobs() > 0)
    {
        decoder_utils::collectSpecLogprobsFromHost(mRuntime, context, activeBatchSize, maxAcceptDepth,
            mHostAcceptLengths.dataPointer<int32_t>(), context.maxNumLogprobs());
    }

    mJump.account(context, mHostAcceptLengths.dataPointer<int32_t>(), mHostAcceptedTokenIds.dataPointer<int32_t>(),
        maxAcceptDepth);

    return true;
}

bool MTPDecoder::runDraftModelAcceptToken(DecodingInferenceContext& context)
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
    int32_t const baseOutputHiddenDim = mRuntime.deployment.specConfig->baseOutputHiddenDim;
    int32_t const draftVocabSize = mRuntime.deployment.draft->outputVocabSize;
    int64_t const inputIdsLength = mAcceptedTokenIds.getShape()[1];

    check::check(mRuntime.preprocess.idsInput.reshape({activeBatchSize, inputIdsLength}), "Tensor reshape failed");
    check::check(
        mRuntime.base.pipelineIO.draftHiddenStatesIn.reshape({activeBatchSize, inputIdsLength, baseOutputHiddenDim}),
        "Tensor reshape failed");
    check::check(
        mRuntime.base.pipelineIO.outputLogits.reshape({activeBatchSize, draftVocabSize}), "Tensor reshape failed");
    check::check(mRuntime.base.pipelineIO.draftHiddenStatesOut.reshape({activeBatchSize, baseOutputHiddenDim}),
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
    check::check(mRuntime.base.pipelineIO.draftHiddenStatesIn.reshape({physicalTokens, baseOutputHiddenDim}),
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

bool MTPDecoder::captureCudaGraphs(cudaStream_t stream)
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

    // Tree mode drafts a chain (topK=1 shapes); draftTopK is only the tree candidate fanout.
    int32_t const draftTopK = mUseTree ? 1 : mRuntime.deployment.specConfig->draftTopK;
    int32_t const paddedDraftProposalSize = mRuntime.deployment.specConfig->draftSteps * draftTopK;
    int32_t const draftSteps = mRuntime.deployment.specConfig->draftSteps;
    int32_t const draftHiddenSize = mRuntime.deployment.specConfig->draftHiddenSize;
    int32_t const baseOutputHiddenDim = mRuntime.deployment.specConfig->baseOutputHiddenDim;
    int32_t const draftVocabSize = mRuntime.deployment.draft->outputVocabSize;

    // A deployment that can jump also runs grafting rounds: the base verifies jumpVerifyTokenCount rows and the draft
    // re-reads up to jumpMaxPathLength accepted tokens. Tree mode caps a plain round's accepted path at
    // verifyTokenCount (see runBaseModelVerification).
    auto const& specConfig = *mRuntime.deployment.specConfig;
    int32_t const plainAcceptDepth = mUseTree ? std::min(draftSteps + 1, specConfig.verifyTokenCount) : draftSteps + 1;
    std::vector<int32_t> acceptLengths;
    for (int32_t acceptLength = 1; acceptLength <= plainAcceptDepth; ++acceptLength)
    {
        acceptLengths.push_back(acceptLength);
    }
    std::vector<int32_t> verifyWidths{specConfig.verifyTokenCount};
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
                             {batchSize, paddedDraftProposalSize, baseOutputHiddenDim}),
                "Tensor reshape failed");
            check::check(mDraftProposalSize.reshape({batchSize}), "Tensor reshape failed");
            check::check(mDraftAttentionMask.reshape({batchSize, paddedDraftProposalSize, paddedDraftProposalSize}),
                "Tensor reshape failed");
            check::check(mRuntime.base.pipelineIO.packedAttentionMask.reshape({batchSize, paddedDraftProposalSize,
                             static_cast<int64_t>(divUp(paddedDraftProposalSize, 32))}),
                "Tensor reshape failed");
            check::check(mRuntime.base.pipelineIO.outputLogits.reshape({batchSize, draftTopK, draftVocabSize}),
                "Tensor reshape failed");
            check::check(
                mRuntime.base.pipelineIO.draftHiddenStatesOut.reshape({batchSize, draftTopK, baseOutputHiddenDim}),
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
            check::check(mRuntime.base.pipelineIO.draftHiddenStatesIn.reshape({proposalTokens, baseOutputHiddenDim}),
                "Tensor reshape failed");
            check::check(mRuntime.base.pipelineIO.outputLogits.reshape({batchSize * draftTopK, draftVocabSize}),
                "Tensor reshape failed");
            check::check(
                mRuntime.base.pipelineIO.draftHiddenStatesOut.reshape({batchSize * draftTopK, baseOutputHiddenDim}),
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
            check::check(
                mRuntime.base.pipelineIO.outputLogits.reshape({batchSize, draftVocabSize}), "Tensor reshape failed");
            check::check(mRuntime.base.pipelineIO.draftHiddenStatesOut.reshape({batchSize, baseOutputHiddenDim}),
                "Tensor reshape failed");
            for (int32_t const acceptLength : acceptLengths)
            {
                check::check(
                    mRuntime.base.pipelineIO.baseHiddenStates.reshape({batchSize, acceptLength, baseOutputHiddenDim}),
                    "Tensor reshape failed");
                check::check(mRuntime.base.pipelineIO.draftHiddenStatesIn.reshape(
                                 {batchSize, acceptLength, baseOutputHiddenDim}),
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
                check::check(mRuntime.base.pipelineIO.draftHiddenStatesIn.reshape({acceptTokens, baseOutputHiddenDim}),
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
            bool const wide = verifyTokenCount > specConfig.verifyTokenCount;
            check::check(mRuntime.preprocess.idsInput.reshape({batchSize, verifyTokenCount}), "Tensor reshape failed");
            if (wide)
            {
                CUDA_CHECK(cudaMemsetAsync(mRuntime.preprocess.idsInput.rawPointer(), 0,
                    static_cast<size_t>(batchSize) * verifyTokenCount * sizeof(int32_t), stream));
            }
            if (mUseTree && !wide)
            {
                buildTreeVerifyInputs(batchSize, stream);
            }
            else if (mUseTree)
            {
                Tensor& mask = mJump.grafter().mask();
                check::check(mask.reshape({batchSize, verifyTokenCount, verifyTokenCount}), "Tensor reshape failed");
                CUDA_CHECK(cudaMemsetAsync(mask.rawPointer(), 0,
                    static_cast<size_t>(batchSize) * verifyTokenCount * verifyTokenCount, stream));
                check::check(mRuntime.base.pipelineIO.specTreeParentIds.reshape({batchSize, verifyTokenCount}),
                    "Tensor reshape failed");
                check::check(mRuntime.base.pipelineIO.specTreeDepths.reshape({batchSize, verifyTokenCount}),
                    "Tensor reshape failed");
                mJump.prepareGraftedTreeDescriptors(
                    mRuntime.base.pipelineIO, mRuntime.base.cacheManager.getKVCacheLengths(), batchSize, stream);
            }
            captureContext.activeBatchSize = batchSize;
            SpecVerifyInputs const verifyInputs{mRuntime.preprocess.idsInput,
                mUseTree && !wide ? &mValidCounts : nullptr,
                mUseTree ? &mRuntime.base.pipelineIO.specTreeParentIds : nullptr,
                mUseTree ? &mRuntime.base.pipelineIO.specTreeDepths : nullptr, batchSize, verifyTokenCount,
                specConfig.maxPathLength, specConfig.verifyLayout, mUseTree};
            spec_decode::prepareTargetVerifyInputs(verifyInputs, mRuntime, captureContext, 0);
            auto const verifyDims
                = mRuntime.deployment.base.specVerifyDims(batchSize, verifyTokenCount, verifyInputs.layout);
            baseVerificationCaptureStatus &= mRuntime.base.captureGraph(verifyDims, stream);
        }
    }

    return draftProposalCaptureStatus && draftAcceptCaptureStatus && baseVerificationCaptureStatus;
}

void MTPDecoder::resetForNewSequences(Tensor& reuseLengths, cudaStream_t stream)
{
    mCommonStateTracker.reset();
    mDraftCacheManager.resetForNewSequences(reuseLengths, stream);
    mDraftPrimed = false;
}

void MTPDecoder::onBatchEvict(std::vector<int32_t> const& batchMapping, int32_t oldActiveBatch, int32_t newActiveBatch,
    Tensor& deviceBatchMapping, cudaStream_t stream)
{
    mCommonStateTracker.compact(batchMapping, oldActiveBatch, newActiveBatch);
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
