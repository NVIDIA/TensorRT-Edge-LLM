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
#include "kernels/embeddingKernels/embeddingKernels.h"
#include "kernels/gdnKernels/gdnTreeChunkKernels.h"
#include "kernels/posEncoding/applyRopeWriteKV.h"
#include "kernels/speculative/batchEvictKernels.h"
#include "kernels/speculative/ddtreeKernels.h"
#include "kernels/speculative/eagleAcceptKernels.h"
#include "kernels/speculative/eagleUtilKernels.h"
#include "profiling/metrics.h"
#include "profiling/nvtx_wrapper.h"
#include "profiling/timer.h"
#include "runtime/config/llmEngineConfig.h"
#include "runtime/decoding/decoderUtils.h"
#include "sampler/sampling.h"

#include <algorithm>
#include <cassert>
#include <cmath>
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
    SpecDecodeDraftingConfig const& draftingConfig, cudaStream_t stream)
    : mRuntime(runtime)
    , mDraftCacheManager(*runtime.base.sharedResources.cacheManagers[1])
{
    check::check(mRuntime.deployment.draft.has_value(), "SpecDecode drafting strategy requires a draft model config.");
    check::check(
        mRuntime.deployment.specConfig.has_value(), "SpecDecode drafting strategy requires a drafting config.");
    check::check(runtime.deployment.base.specDecodeType == SpecDecodeMode::kMTP,
        "MTP decoding requires a base engine exported with spec_decode_type=mtp and engine_role=base.");

    mDraftExecutor = decoder_utils::loadDraftEngine(engineDir, mRuntime.deployment);
    auto const draftLogitsType = mDraftExecutor->getBindingDataType(binding_names::kLogits);
    mConvertDraftLogits = draftLogitsType == nvinfer1::DataType::kHALF;
    ELLM_CHECK(draftLogitsType == nvinfer1::DataType::kFLOAT || mConvertDraftLogits,
        "MTP draft logits must be FLOAT or HALF.");

    int32_t const maxRuntimeBatchSize = mRuntime.maxRuntimeBatchSize;
    int32_t const effectiveMaxDraftProposalSize = mRuntime.deployment.effectiveMaxDraftProposalSize();
    // MTP always walks one greedy draft chain (top-1 per round) and keeps each round's full logits row.
    // ddtreeBuild then picks draftingTopK candidates per depth row; draftingTopK=1 is a degenerate chain tree.
    int32_t const effectiveDraftTopK = 1;
    int32_t const effectiveMaxAcceptDepth = draftingConfig.draftingStep + 1;
    int32_t const draftFullTableLength
        = 1 + effectiveDraftTopK + (draftingConfig.draftingStep - 1) * effectiveDraftTopK * effectiveDraftTopK;

    mDraftProposalSize
        = Tensor({maxRuntimeBatchSize}, DeviceType::kGPU, nvinfer1::DataType::kINT32, "SpecDecode::draftProposalSize");
    mDraftAttentionMask = Tensor({maxRuntimeBatchSize, effectiveMaxDraftProposalSize, effectiveMaxDraftProposalSize},
        DeviceType::kGPU, nvinfer1::DataType::kINT8, "SpecDecode::draftAttentionMask");

    buildTensorMapForSpecDecodeDraft(
        mDraftTensorMap, mRuntime.base.pipelineIO, mRuntime.base.sharedResources, *mRuntime.deployment.draft);
    if (mConvertDraftLogits)
    {
        mDraftOutputLogits = Tensor({maxRuntimeBatchSize, mRuntime.deployment.draft->outputVocabSize}, DeviceType::kGPU,
            nvinfer1::DataType::kHALF, "SpecDecode::draftOutputLogits");
        mDraftTensorMap.set(binding_names::kLogits, mDraftOutputLogits);
    }

    // Publish externalized draft-engine weights into the draft tensor map,
    // mirroring the base engine. Loaded from draft_config.json; a no-op when
    // the draft was exported without --externalize-weights.
    mDraftExternalWeightManager.load(engineDir, engineDir / "draft_config.json", stream);
    mDraftExternalWeightManager.validateAgainstEngine(*mDraftExecutor, "draft");
    mDraftExternalWeightManager.registerTensorMapEntries(mDraftTensorMap);

    mDraftTokenIdsFullTable = Tensor({maxRuntimeBatchSize, draftFullTableLength}, DeviceType::kGPU,
        nvinfer1::DataType::kINT32, "SpecDecode::idsFull");
    mDraftTokenScoreFullTable = Tensor({maxRuntimeBatchSize, draftFullTableLength}, DeviceType::kGPU,
        nvinfer1::DataType::kFLOAT, "SpecDecode::scoreFull");
    mDraftTokenPredecessorFullTable = Tensor({maxRuntimeBatchSize, draftFullTableLength}, DeviceType::kGPU,
        nvinfer1::DataType::kINT32, "SpecDecode::predFull");
    mDraftVocabMappingTable = Tensor({mRuntime.deployment.draft->outputVocabSize}, DeviceType::kGPU,
        nvinfer1::DataType::kINT32, "SpecDecode::draftVocabMap");
    mDraftRootTokenId
        = Tensor({maxRuntimeBatchSize}, DeviceType::kGPU, nvinfer1::DataType::kINT32, "SpecDecode::rootToken");
    mDraftTokenIdsTable = Tensor({maxRuntimeBatchSize, effectiveDraftTopK * effectiveDraftTopK}, DeviceType::kGPU,
        nvinfer1::DataType::kINT32, "SpecDecode::idsTable");
    mDraftTokenScoresTable = Tensor({maxRuntimeBatchSize, effectiveDraftTopK * effectiveDraftTopK}, DeviceType::kGPU,
        nvinfer1::DataType::kFLOAT, "SpecDecode::scoresTable");
    mDraftTokenIntermediateScores = Tensor({maxRuntimeBatchSize, effectiveDraftTopK}, DeviceType::kGPU,
        nvinfer1::DataType::kFLOAT, "SpecDecode::intermediateScores");
    mDraftTokenIntermediateParents = Tensor({maxRuntimeBatchSize, effectiveDraftTopK}, DeviceType::kGPU,
        nvinfer1::DataType::kINT32, "SpecDecode::intermediateParents");
    mAcceptedTokenIds = Tensor({maxRuntimeBatchSize, effectiveMaxAcceptDepth}, DeviceType::kGPU,
        nvinfer1::DataType::kINT32, "SpecDecode::acceptedIds");
    mAcceptedTokenIndices = Tensor({maxRuntimeBatchSize, effectiveMaxAcceptDepth}, DeviceType::kGPU,
        nvinfer1::DataType::kINT32, "SpecDecode::acceptedIndices");
    mAcceptLength
        = Tensor({maxRuntimeBatchSize}, DeviceType::kGPU, nvinfer1::DataType::kINT32, "SpecDecode::acceptLength");
    mHostAcceptLengths
        = Tensor({maxRuntimeBatchSize}, DeviceType::kCPU, nvinfer1::DataType::kINT32, "SpecDecode::hostAcceptLengths");
    mHostAcceptedTokenIds = Tensor({maxRuntimeBatchSize, effectiveMaxAcceptDepth}, DeviceType::kCPU,
        nvinfer1::DataType::kINT32, "SpecDecode::hostAcceptedIds");

    int32_t const verifySize = draftingConfig.verifySize;
    int32_t const draftVocabSize = mRuntime.deployment.draft->outputVocabSize;
    // Tree proposal rows: row 0 is the root placeholder (ignored by the builder),
    // rows 1..draftingStep hold the per-depth candidate logits.
    int32_t const proposalDepthSize = draftingConfig.draftingStep + 1;
    mStackedDraftLogits = Tensor({maxRuntimeBatchSize, proposalDepthSize, draftVocabSize}, DeviceType::kGPU,
        nvinfer1::DataType::kFLOAT, "SpecDecode::stackedDraftLogits");
    mTreeTokenIds = Tensor(
        {maxRuntimeBatchSize, verifySize}, DeviceType::kGPU, nvinfer1::DataType::kINT32, "SpecDecode::treeTokenIds");
    mTreeNodeScores = Tensor(
        {maxRuntimeBatchSize, verifySize}, DeviceType::kGPU, nvinfer1::DataType::kFLOAT, "SpecDecode::treeNodeScores");
    mValidCounts
        = Tensor({maxRuntimeBatchSize}, DeviceType::kGPU, nvinfer1::DataType::kINT32, "SpecDecode::validCounts");
    mVerifyTreeMask = Tensor({maxRuntimeBatchSize, verifySize, verifySize}, DeviceType::kGPU, nvinfer1::DataType::kINT8,
        "SpecDecode::verifyTreeMask");
    size_t const buildWorkspaceSize = kernel::getDDTreeBuildWorkspaceSize(
        maxRuntimeBatchSize, proposalDepthSize, verifySize, draftVocabSize, draftingConfig.draftingTopK);
    ELLM_CHECK(buildWorkspaceSize > 0, "MTPDecoder: invalid tree-build workspace size.");
    mTreeBuildWorkspace = Tensor({static_cast<int64_t>(buildWorkspaceSize)}, DeviceType::kGPU,
        nvinfer1::DataType::kUINT8, "SpecDecode::treeBuildWorkspace");
    CUDA_CHECK(cudaMemsetAsync(mStackedDraftLogits.rawPointer(), 0, mStackedDraftLogits.getMemoryCapacity(), stream));
    // Zero the root ids as well so the capture-time tree build (which runs on the
    // zero-initialized logits) is fully deterministic.
    CUDA_CHECK(cudaMemsetAsync(mDraftRootTokenId.rawPointer(), 0, mDraftRootTokenId.getMemoryCapacity(), stream));

    // MTP: identity vocab mapping (zero-fill)
    CUDA_CHECK(
        cudaMemsetAsync(mDraftVocabMappingTable.rawPointer(), 0, mDraftVocabMappingTable.getMemoryCapacity(), stream));
}

int64_t MTPDecoder::getRequiredContextMemorySize() const noexcept
{
    return mDraftExecutor ? mDraftExecutor->getRequiredContextMemorySize() : 0;
}

void MTPDecoder::reshapeDraftLogits(rt::Coords const& shape)
{
    check::check(mRuntime.base.pipelineIO.outputLogits.reshape(shape), "Tensor reshape failed");
    if (mConvertDraftLogits)
    {
        check::check(mDraftOutputLogits.reshape(shape), "MTP draft-logits reshape failed");
    }
}

void MTPDecoder::convertDraftLogits(cudaStream_t stream)
{
    if (mConvertDraftLogits)
    {
        kernel::convertLogitsToFloat(mDraftOutputLogits, mRuntime.base.pipelineIO.outputLogits, stream);
    }
}

void MTPDecoder::setContextMemory(Tensor& memory)
{
    if (mDraftExecutor)
    {
        mDraftExecutor->setContextMemory(memory);
    }
}

bool MTPDecoder::decodeStep(DecodingInferenceContext& context)
{
    if (context.generationRound == 0)
    {
        if (!prepareFirstDecodeStep(context))
        {
            LOG_ERROR("Failed to execute prefill step for draft model.");
            return false;
        }
    }
    else if (!runDraftModelAcceptToken(context))
    {
        LOG_ERROR("Failed to execute accept token step for draft model.");
        return false;
    }

    if (!constructDraftProposal(context))
    {
        LOG_ERROR("Failed to construct draft proposal.");
        return false;
    }

    if (!runBaseModelVerification(context))
    {
        LOG_ERROR("Failed to verify draft proposal with base model.");
        return false;
    }
    return true;
}

bool MTPDecoder::prepareFirstDecodeStep(DecodingInferenceContext& context)
{
    ELLM_CHECK(context.generationRound == 0, "MTP draft prefill preparation is only valid before round zero");
    if (context.speculativeDraftPrefillComplete)
    {
        return true;
    }
    if (!runDraftModelPrefill(context))
    {
        return false;
    }
    context.speculativeDraftPrefillComplete = true;
    return true;
}

bool MTPDecoder::runDraftModelPrefill(DecodingInferenceContext& context)
{
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
    int32_t const draftVocabSize = mRuntime.deployment.draft->outputVocabSize;
    int32_t const inputIdsLength
        = *std::max_element(context.effectivePrefillLengths.begin(), context.effectivePrefillLengths.end());

    check::check(mRuntime.base.pipelineIO.baseHiddenStates.getShape()[0] == activeBatchSize
            && mRuntime.base.pipelineIO.baseHiddenStates.getShape()[1] == inputIdsLength,
        "BaseHiddenStates shape [batch, seq_len, hidden_dim] shall match active prefill shape.");

    check::check(mRuntime.preprocess.idsInput.reshape({activeBatchSize, inputIdsLength}), "Tensor reshape failed");
    check::check(
        mRuntime.base.pipelineIO.draftHiddenStatesIn.reshape({activeBatchSize, inputIdsLength, draftHiddenSize}),
        "Tensor reshape failed");
    reshapeDraftLogits({activeBatchSize, draftVocabSize});
    check::check(mRuntime.base.pipelineIO.draftHiddenStatesOut.reshape({activeBatchSize, draftHiddenSize}),
        "Tensor reshape failed");
    check::check(mRuntime.base.pipelineIO.hostContextLengths.reshape({activeBatchSize}), "Tensor reshape failed");

    // Zero only the live view. getMemoryCapacity() is the max-input-length allocation and
    // can be hundreds of MiB; memsetting it every draft-prefill dominates mem-op time.
    CUDA_CHECK(cudaMemsetAsync(mRuntime.base.pipelineIO.draftHiddenStatesIn.rawPointer(), 0,
        static_cast<size_t>(mRuntime.base.pipelineIO.draftHiddenStatesIn.getActiveBytes()), context.stream));

    check::check(
        mRuntime.sampling.hostPackedTokenIds.reshape({activeBatchSize, inputIdsLength}), "Tensor reshape failed");
    int32_t* hostPackedTokenIdsData = mRuntime.sampling.hostPackedTokenIds.dataPointer<int32_t>();
    for (int32_t i = 0; i < activeBatchSize; ++i)
    {
        std::copy(
            context.tokenIds[i].begin() + 1, context.tokenIds[i].end(), hostPackedTokenIdsData + i * inputIdsLength);
    }
    CUDA_CHECK(cudaMemcpyAsync(mRuntime.preprocess.idsInput.rawPointer(), hostPackedTokenIdsData,
        activeBatchSize * inputIdsLength * sizeof(int32_t), cudaMemcpyHostToDevice, context.stream));

    check::check(mRuntime.base.pipelineIO.inputsEmbeds.reshape({activeBatchSize, inputIdsLength, draftHiddenSize}),
        "Tensor reshape failed");
    if (context.visualEmbeddings.has_value())
    {
        Tensor const& imageEmbedsTensor = context.visualEmbeddings.value().get();
        kernel::embeddingLookupWithImageInsertion(mRuntime.preprocess.idsInput, mRuntime.preprocess.embedding.table,
            mRuntime.preprocess.embedding.scalesAsOptional(), imageEmbedsTensor, mRuntime.base.pipelineIO.inputsEmbeds,
            context.stream);
    }
    else
    {
        kernel::embeddingLookup(mRuntime.preprocess.idsInput, mRuntime.preprocess.embedding.table,
            mRuntime.preprocess.embedding.scalesAsOptional(), mRuntime.base.pipelineIO.inputsEmbeds, context.stream);
    }
    if (mRuntime.preprocess.gemma4Ple)
    {
        mRuntime.preprocess.gemma4Ple->embed(mRuntime.preprocess.idsInput, context.stream);
    }

    int32_t* ctxLenData = mRuntime.base.pipelineIO.hostContextLengths.dataPointer<int32_t>();
    for (int32_t i = 0; i < activeBatchSize; ++i)
    {
        ctxLenData[i] = context.effectivePrefillLengths[i];
    }
    CUDA_CHECK(cudaMemcpyAsync(mRuntime.base.pipelineIO.contextLengths.rawPointer(), ctxLenData,
        activeBatchSize * sizeof(int32_t), cudaMemcpyHostToDevice, context.stream));

    check::check(mRuntime.base.pipelineIO.selectTokenIndices.reshape({activeBatchSize, 1}), "Tensor reshape failed");
    check::check(mRuntime.base.pipelineIO.hostSelectTokenIndices.reshape({activeBatchSize, 1}),
        "hostSelectTokenIndices reshape failed");
    int64_t* selectData = mRuntime.base.pipelineIO.hostSelectTokenIndices.dataPointer<int64_t>();
    for (int32_t i = 0; i < activeBatchSize; ++i)
    {
        selectData[i] = context.effectivePrefillLengths[i] - 1;
    }
    CUDA_CHECK(cudaMemcpyAsync(mRuntime.base.pipelineIO.selectTokenIndices.rawPointer(),
        mRuntime.base.pipelineIO.hostSelectTokenIndices.rawPointer(), activeBatchSize * sizeof(int64_t),
        cudaMemcpyHostToDevice, context.stream));

    bool const draftKVAllEmpty = mDraftCacheManager.getKVCacheAllEmpty();
    auto const prefillDims = mRuntime.deployment.draft->prefillDims(activeBatchSize, inputIdsLength, draftKVAllEmpty);
    bool prefillSuccess = mDraftExecutor->prepare(kPrefillProfile, prefillDims, mDraftTensorMap, context.stream);
    if (prefillSuccess)
    {
        prefillSuccess = mDraftExecutor->execute(context.stream);
    }
    if (prefillSuccess)
    {
        convertDraftLogits(context.stream);
        mDraftCacheManager.commitSequenceLength(mRuntime.base.pipelineIO.contextLengths, context.stream);
        for (int32_t i = 0; i < activeBatchSize; ++i)
        {
            if (context.sequenceCacheStates[i].has_value()
                && context.sequenceCacheStates[i]->draftResidentStateLength.has_value())
            {
                *context.sequenceCacheStates[i]->draftResidentStateLength += context.effectivePrefillLengths[i];
            }
        }
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
    // The draft engine always walks one chain; draftingTopK only sets the tree-build candidate fanout.
    int32_t const draftTopK = 1;
    int32_t const draftFullTableLength = static_cast<int32_t>(mDraftTokenIdsFullTable.getShape()[1]);

    // Stash the current frontier's full logits row per batch as the depth-`depthRow` candidates for the tree builder.
    // outputLogits holds one contiguous [draftVocabSize] FLOAT row per batch entry at every call site.
    auto const copyLogitsRowToStack = [&](int32_t depthRow) {
        int32_t const proposalDepthSize = mRuntime.deployment.specConfig->draftingStep + 1;
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

    // Depth-1 candidates come "for free" from the logits that predicted the root's
    // successor (base prefill logits on round 0, draft accept-step logits afterwards).
    copyLogitsRowToStack(/*depthRow=*/1);

    check::check(mRuntime.sampling.indices.reshape({activeBatchSize, draftTopK}), "Tensor reshape failed");
    check::check(mRuntime.sampling.scores.reshape({activeBatchSize, draftTopK}), "Tensor reshape failed");
    selectAllTopK(mRuntime.base.pipelineIO.outputLogits, std::ref(mRuntime.sampling.scores), mRuntime.sampling.indices,
        draftTopK, mRuntime.sampling.workspace, context.stream);

    kernel::initializeDraftTreeTables(mRuntime.sampling.indices, mRuntime.sampling.scores, mDraftRootTokenId,
        mDraftVocabMappingTable, mDraftTokenIdsFullTable, mDraftTokenScoreFullTable, mDraftTokenPredecessorFullTable,
        draftTopK, context.stream);

    int32_t const paddedDraftProposalSize = mRuntime.deployment.specConfig->draftingStep * draftTopK;
    check::check(
        mRuntime.preprocess.idsInput.reshape({activeBatchSize, paddedDraftProposalSize}), "Tensor reshape failed");
    check::check(mRuntime.base.pipelineIO.baseHiddenStates.reshape(
                     {activeBatchSize, paddedDraftProposalSize, baseOutputHiddenDim}),
        "Tensor reshape failed");
    check::check(mRuntime.base.pipelineIO.draftHiddenStatesIn.reshape(
                     {activeBatchSize, paddedDraftProposalSize, draftHiddenSize}),
        "Tensor reshape failed");
    // Reshape first, then zero only the proposal-sized live view. Memsetting getMemoryCapacity()
    // (max input length × hidden) every draft round was ~70ms+ of GPU memset in nsys captures.
    CUDA_CHECK(cudaMemsetAsync(mRuntime.base.pipelineIO.baseHiddenStates.rawPointer(), 0,
        static_cast<size_t>(mRuntime.base.pipelineIO.baseHiddenStates.getActiveBytes()), context.stream));
    CUDA_CHECK(cudaMemsetAsync(mRuntime.base.pipelineIO.draftHiddenStatesIn.rawPointer(), 0,
        static_cast<size_t>(mRuntime.base.pipelineIO.draftHiddenStatesIn.getActiveBytes()), context.stream));
    check::check(mDraftProposalSize.reshape({activeBatchSize}), "Tensor reshape failed");
    check::check(mDraftAttentionMask.reshape({activeBatchSize, paddedDraftProposalSize, paddedDraftProposalSize}),
        "Tensor reshape failed");
    check::check(mRuntime.base.pipelineIO.packedAttentionMask.reshape({activeBatchSize, paddedDraftProposalSize,
                     static_cast<int64_t>(divUp(paddedDraftProposalSize, 32))}),
        "Tensor reshape failed");

    kernel::assembleInitialDraftTreeInput(mDraftTokenIdsFullTable, mRuntime.base.pipelineIO.draftHiddenStatesOut,
        mRuntime.preprocess.idsInput, mRuntime.base.pipelineIO.draftHiddenStatesIn, mDraftProposalSize,
        mDraftAttentionMask, draftTopK, context.stream);

    reshapeDraftLogits({activeBatchSize, draftTopK, draftVocabSize});
    check::check(mRuntime.base.pipelineIO.draftHiddenStatesOut.reshape({activeBatchSize, draftTopK, draftHiddenSize}),
        "Tensor reshape failed");

    for (int32_t round = 0; round < mRuntime.deployment.specConfig->draftingStep - 1; round++)
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
            kernel::assembleDraftTreeInput(mDraftTokenIdsTable, mRuntime.base.pipelineIO.draftHiddenStatesOut,
                mRuntime.sampling.indices, mRuntime.preprocess.idsInput, mRuntime.base.pipelineIO.draftHiddenStatesIn,
                mDraftProposalSize, mDraftAttentionMask, draftTopK, round, context.stream);
            kernel::assembleIntermediateData(mRuntime.sampling.scores, mRuntime.sampling.indices,
                mDraftTokenIntermediateScores, mDraftTokenIntermediateParents, draftTopK, round, context.stream);
        }

        check::check(mRuntime.base.pipelineIO.outputLogits.reshape({activeBatchSize, draftTopK, draftVocabSize}),
            "Tensor reshape failed");
        check::check(
            mRuntime.base.pipelineIO.draftHiddenStatesOut.reshape({activeBatchSize, draftTopK, draftHiddenSize}),
            "Tensor reshape failed");
        check::check(
            mRuntime.base.pipelineIO.inputsEmbeds.reshape({activeBatchSize, paddedDraftProposalSize, draftHiddenSize}),
            "Tensor reshape failed");
        kernel::embeddingLookup(mRuntime.preprocess.idsInput, mRuntime.preprocess.embedding.table,
            mRuntime.preprocess.embedding.scalesAsOptional(), mRuntime.base.pipelineIO.inputsEmbeds, context.stream);

        {
            Tensor const& draftKVCacheLengths = mDraftCacheManager.getKVCacheLengths();
            check::check(mRuntime.base.pipelineIO.selectTokenIndices.reshape({activeBatchSize, draftTopK}),
                "Tensor reshape failed");
            check::check(mRuntime.base.pipelineIO.contextLengths.reshape({activeBatchSize}), "Tensor reshape failed");
            check::check(
                mRuntime.base.pipelineIO.specDecodePositionIds.reshape({activeBatchSize, paddedDraftProposalSize}),
                "Tensor reshape failed");
            kernel::prepareEagleDraftProposalInputs(mDraftAttentionMask, mDraftProposalSize, draftKVCacheLengths,
                mRuntime.base.pipelineIO.packedAttentionMask, mRuntime.base.pipelineIO.specDecodePositionIds,
                mRuntime.base.pipelineIO.selectTokenIndices, mRuntime.base.pipelineIO.contextLengths, context.stream);
        }

        auto const proposalDims
            = mRuntime.deployment.draft->proposalDims(activeBatchSize, paddedDraftProposalSize, draftTopK);
        check::check(mDraftExecutor->prepare(kDecodeProfile, proposalDims, mDraftTensorMap, context.stream),
            "Failed to prepare draft model for draft proposal step.");
        check::check(mDraftExecutor->execute(context.stream), "Failed to execute draft model for draft proposal step.");
        convertDraftLogits(context.stream);

        check::check(mRuntime.base.pipelineIO.outputLogits.reshape({activeBatchSize * draftTopK, draftVocabSize}),
            "Tensor reshape failed");
        // This round's chain-frontier logits are the depth-(round+2) candidates.
        copyLogitsRowToStack(/*depthRow=*/round + 2);
        check::check(
            mRuntime.base.pipelineIO.draftHiddenStatesOut.reshape({activeBatchSize * draftTopK, draftHiddenSize}),
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

    return buildTreeVerifyInputs(activeBatchSize, context.stream);
}

bool MTPDecoder::buildTreeVerifyInputs(int32_t activeBatchSize, cudaStream_t stream)
{
    NVTX_SCOPED_RANGE(nvtx_mtp_tree_build, "MTPDecoder::buildTreeVerifyInputs", nvtx_colors::LIGHT_ORANGE);

    int32_t const verifySize = mRuntime.deployment.specConfig->verifySize;
    int32_t const proposalDepthSize = mRuntime.deployment.specConfig->draftingStep + 1;
    int32_t const draftVocabSize = mRuntime.deployment.draft->outputVocabSize;

    check::check(
        mStackedDraftLogits.reshape({activeBatchSize, proposalDepthSize, draftVocabSize}), "Tensor reshape failed");
    check::check(mTreeTokenIds.reshape({activeBatchSize, verifySize}), "Tensor reshape failed");
    check::check(mTreeNodeScores.reshape({activeBatchSize, verifySize}), "Tensor reshape failed");
    check::check(mValidCounts.reshape({activeBatchSize}), "Tensor reshape failed");
    check::check(
        mRuntime.base.pipelineIO.specTreeParentIds.reshape({activeBatchSize, verifySize}), "Tensor reshape failed");
    check::check(
        mRuntime.base.pipelineIO.specTreeDepths.reshape({activeBatchSize, verifySize}), "Tensor reshape failed");
    check::check(mRuntime.preprocess.idsInput.reshape({activeBatchSize, verifySize}), "Tensor reshape failed");
    check::check(
        mRuntime.base.pipelineIO.specDecodePositionIds.reshape({activeBatchSize, verifySize}), "Tensor reshape failed");
    check::check(mRuntime.base.pipelineIO.packedAttentionMask.reshape(
                     {activeBatchSize, verifySize, static_cast<int64_t>(divUp(verifySize, 32))}),
        "Tensor reshape failed");
    check::check(mVerifyTreeMask.reshape({activeBatchSize, verifySize, verifySize}), "Tensor reshape failed");
    check::check(mRuntime.base.pipelineIO.contextLengths.reshape({activeBatchSize}), "Tensor reshape failed");
    check::check(
        mRuntime.base.pipelineIO.selectTokenIndices.reshape({activeBatchSize, verifySize}), "Tensor reshape failed");

    // MTP has no draft vocab reduction, so no reduced-to-full mapping is passed.
    Tensor const& baseKVCacheLengths = mRuntime.base.cacheManager.getKVCacheLengths();
    kernel::DDTreeBuildParams const buildParams{{mStackedDraftLogits, mDraftRootTokenId, baseKVCacheLengths, nullptr},
        {mTreeTokenIds, mRuntime.base.pipelineIO.specTreeDepths, mRuntime.base.pipelineIO.specTreeParentIds,
            mTreeNodeScores, mValidCounts, mRuntime.preprocess.idsInput, mRuntime.base.pipelineIO.specDecodePositionIds,
            mRuntime.base.pipelineIO.packedAttentionMask, mVerifyTreeMask, mRuntime.base.pipelineIO.contextLengths,
            mRuntime.base.pipelineIO.selectTokenIndices},
        mRuntime.deployment.specConfig->draftingTopK, mTreeBuildWorkspace.rawPointer(),
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

    int32_t const activeBatchSize = context.activeBatchSize;
    int32_t const baseOutputHiddenDim = mRuntime.deployment.specConfig->baseOutputHiddenDim;

    Tensor const& verifyTreeMask = mVerifyTreeMask;
    check::check(mRuntime.preprocess.idsInput.getShape()[0] == activeBatchSize
            && mRuntime.preprocess.idsInput.getShape()[1] == mRuntime.deployment.specConfig->verifySize,
        "IdsInput shall have shape [batch_size, verify_size]");
    check::check(verifyTreeMask.getShape()[0] == activeBatchSize
            && verifyTreeMask.getShape()[1] == mRuntime.deployment.specConfig->verifySize
            && verifyTreeMask.getShape()[2] == mRuntime.deployment.specConfig->verifySize,
        "Verify tree mask shall have shape [batch_size, verify_size, verify_size]");

    check::check(mRuntime.base.pipelineIO.inputsEmbeds.reshape({activeBatchSize,
                     mRuntime.deployment.specConfig->verifySize, mRuntime.deployment.base.hiddenSize}),
        "Tensor reshape failed");
    kernel::embeddingLookup(mRuntime.preprocess.idsInput, mRuntime.preprocess.embedding.table,
        mRuntime.preprocess.embedding.scalesAsOptional(), mRuntime.base.pipelineIO.inputsEmbeds, context.stream);
    if (mRuntime.preprocess.gemma4Ple)
    {
        mRuntime.preprocess.gemma4Ple->embed(mRuntime.preprocess.idsInput, context.stream);
    }

    int32_t const selectTokenSize = activeBatchSize * mRuntime.deployment.specConfig->verifySize;
    check::check(
        mRuntime.base.pipelineIO.outputLogits.reshape({selectTokenSize, mRuntime.deployment.base.outputVocabSize}),
        "Tensor reshape failed");
    check::check(mRuntime.base.pipelineIO.baseHiddenStates.reshape({selectTokenSize, baseOutputHiddenDim}),
        "Tensor reshape failed");

    // ddtreeBuild already emitted the packed mask, position ids, select indices, and context lengths.

    if (mRuntime.preprocess.deepstack)
    {
        mRuntime.preprocess.deepstack->useZeroTarget(mRuntime.base.tensorMap);
    }

    // MTP: inline prepareBaseVerificationState
    mRuntime.base.cacheManager.getMambaCacheManager().reshapeIntermediateStates(
        context.activeBatchSize, mRuntime.deployment.specConfig->verifySize);
    if (!mRuntime.base.pipelineIO.specVerifyPhaseMarker.isEmpty())
    {
        check::check(mRuntime.base.pipelineIO.specVerifyPhaseMarker.reshape({1}), "Tensor reshape failed");
    }

    auto const verifyDims
        = mRuntime.deployment.base.specVerifyDims(activeBatchSize, mRuntime.deployment.specConfig->verifySize);
    bool verifySuccess
        = mRuntime.base.executor.prepare(kDecodeProfile, verifyDims, mRuntime.base.tensorMap, context.stream);
    if (verifySuccess)
    {
        verifySuccess = mRuntime.base.execute(context.stream);
    }
    if (!verifySuccess)
    {
        LOG_ERROR("Failed to execute base verification step for base model.");
        return false;
    }

    // A tree with fewer verify nodes than the full chain depth caps the acceptable
    // path length at verifySize.
    int32_t const chainAcceptDepth = mRuntime.deployment.specConfig->draftingStep + 1;
    int32_t const maxAcceptDepth = std::min(chainAcceptDepth, mRuntime.deployment.specConfig->verifySize);
    check::check(mAcceptedTokenIds.reshape({activeBatchSize, maxAcceptDepth}), "Tensor reshape failed");
    check::check(mAcceptedTokenIndices.reshape({activeBatchSize, maxAcceptDepth}), "Tensor reshape failed");
    check::check(mAcceptLength.reshape({activeBatchSize}), "Tensor reshape failed");

    OptionalInputTensor vocabMappingTable = (mRuntime.deployment.base.reducedVocabSize > 0)
        ? std::optional{std::cref(mRuntime.sampling.baseVocabMappingTable)}
        : std::nullopt;
    Tensor const& acceptTokenIds = mTreeTokenIds;
    kernel::eagleAccept(mRuntime.base.pipelineIO.outputLogits, acceptTokenIds, verifyTreeMask, mAcceptedTokenIds,
        mAcceptedTokenIndices, mAcceptLength, vocabMappingTable, mRuntime.sampling.workspace.rawPointer(),
        mRuntime.sampling.workspace.getMemoryCapacity(), context.stream);

    auto& cacheMgrBase = mRuntime.base.cacheManager;
    Tensor const& kvCacheLengths = cacheMgrBase.getKVCacheLengths();
    auto& kvMgrBase = cacheMgrBase.getKVCacheManager();
    auto const kvHeadDimGroups = cacheMgrBase.getKVHeadDimGroups();
    auto const kvCacheType = kvMgrBase.getConfig().kvCacheType;

    check::check(mRuntime.base.pipelineIO.baseHiddenStates.reshape(
                     {activeBatchSize, mRuntime.deployment.specConfig->verifySize, baseOutputHiddenDim}),
        "Tensor reshape failed");

    auto const& basePageTable = *mRuntime.base.sharedResources.kvPageTables[0];
    int32_t const* basePageTablePtr = basePageTable.kernelView().dataPointer<int32_t>();
    int32_t const baseMaxPagesPerSeq = basePageTable.maxPagesPerSeq();
    for (auto const& group : kvHeadDimGroups)
    {
        kernel::eagleBaseCommitKVCache(mAcceptedTokenIndices, mAcceptLength, kvCacheLengths, group.deviceLayerInfos,
            group.numLayers, group.headDim, group.maxKVHeads, activeBatchSize, maxAcceptDepth, kvCacheType,
            context.stream, basePageTablePtr, baseMaxPagesPerSeq);
    }
    kernel::eagleBaseAssembleHiddenState(
        mAcceptedTokenIndices, mAcceptLength, mRuntime.base.pipelineIO.baseHiddenStates, context.stream);
    mRuntime.base.cacheManager.commitSequenceLength(mAcceptLength, context.stream);

    // Hybrid MTP replays the accepted recurrent path and scatters the final
    // accepted conv checkpoint.
    auto& mambaMgr = mRuntime.base.cacheManager.getMambaCacheManager();
    if (mambaMgr.hasIntermediateRecurrentStates() || mambaMgr.hasIntermediateConvStates())
    {
        check::check(kernel::gdnTreeChunkVerifyEnabled(mRuntime.deployment.specConfig->verifySize),
            "Hybrid MTP requires compact GDN speculative verification.");
        mambaMgr.replayCommitAcceptedTreeStates(mAcceptedTokenIndices, mAcceptLength, context.stream);
    }

    check::check(
        mRuntime.base.pipelineIO.baseHiddenStates.reshape({activeBatchSize, maxAcceptDepth, baseOutputHiddenDim}),
        "Tensor reshape failed");

    // Enqueue logprobs device work + D2H before appendAcceptedTokens so everything rides
    // that call's single round synchronization.
    if (context.numLogprobs > 0)
    {
        int32_t const verifyTreeSize = mRuntime.deployment.specConfig->verifySize;
        int32_t const vocabSize = mRuntime.deployment.base.outputVocabSize;
        int32_t const gatheredRows = activeBatchSize * maxAcceptDepth;
        check::check(mRuntime.logprobs.gatheredLogits.reshape({gatheredRows, vocabSize}), "Tensor reshape failed");
        gatherSpecVerifyAcceptedLogitRows(mRuntime.base.pipelineIO.outputLogits, mAcceptedTokenIndices,
            mRuntime.logprobs.gatheredLogits, activeBatchSize, verifyTreeSize, maxAcceptDepth, vocabSize,
            context.stream);
        decoder_utils::enqueueLogprobsD2H(
            mRuntime.logprobs.gatheredLogits, gatheredRows, mRuntime, context.numLogprobs, context.stream);
    }

    decoder_utils::appendAcceptedTokens(context, mHostAcceptLengths, mHostAcceptedTokenIds, mAcceptLength,
        mAcceptedTokenIds, maxAcceptDepth, mRuntime.tokenizer, context.stream);

    if (context.numLogprobs > 0)
    {
        decoder_utils::collectSpecLogprobsFromHost(mRuntime, context, activeBatchSize, maxAcceptDepth,
            mHostAcceptLengths.dataPointer<int32_t>(), context.numLogprobs);
    }

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
    int32_t const draftVocabSize = mRuntime.deployment.draft->outputVocabSize;
    int64_t const inputIdsLength = mRuntime.base.pipelineIO.baseHiddenStates.getShape()[1];

    check::check(mRuntime.preprocess.idsInput.reshape({activeBatchSize, inputIdsLength}), "Tensor reshape failed");
    check::check(
        mRuntime.base.pipelineIO.draftHiddenStatesIn.reshape({activeBatchSize, inputIdsLength, draftHiddenSize}),
        "Tensor reshape failed");
    reshapeDraftLogits({activeBatchSize, draftVocabSize});
    check::check(mRuntime.base.pipelineIO.draftHiddenStatesOut.reshape({activeBatchSize, draftHiddenSize}),
        "Tensor reshape failed");

    CUDA_CHECK(cudaMemsetAsync(mRuntime.base.pipelineIO.draftHiddenStatesIn.rawPointer(), 0,
        static_cast<size_t>(mRuntime.base.pipelineIO.draftHiddenStatesIn.getActiveBytes()), context.stream));

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
        mRuntime.preprocess.embedding.scalesAsOptional(), mRuntime.base.pipelineIO.inputsEmbeds, context.stream);
    if (mRuntime.preprocess.gemma4Ple)
    {
        mRuntime.preprocess.gemma4Ple->embed(mRuntime.preprocess.idsInput, context.stream);
    }

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
        kernel::prepareEagleAcceptDecodeTokenInputs(draftKVCacheLengths, mAcceptLength,
            mRuntime.base.pipelineIO.packedAttentionMask, mRuntime.base.pipelineIO.specDecodePositionIds,
            mRuntime.base.pipelineIO.selectTokenIndices, mRuntime.base.pipelineIO.contextLengths, context.stream);
    }

    auto const acceptDims = mRuntime.deployment.draft->acceptDims(activeBatchSize, inputIdsLength);
    check::check(mDraftExecutor->prepare(kDecodeProfile, acceptDims, mDraftTensorMap, context.stream),
        "Failed to prepare draft model for accept token step.");
    check::check(mDraftExecutor->execute(context.stream), "Failed to execute draft model for accept token step.");
    convertDraftLogits(context.stream);
    mDraftCacheManager.commitSequenceLength(mAcceptLength, context.stream);
    int32_t const* acceptedLengths = mHostAcceptLengths.dataPointer<int32_t>();
    for (int32_t i = 0; i < activeBatchSize; ++i)
    {
        if (context.sequenceCacheStates[i].has_value()
            && context.sequenceCacheStates[i]->draftResidentStateLength.has_value())
        {
            *context.sequenceCacheStates[i]->draftResidentStateLength += acceptedLengths[i];
        }
    }

    return true;
}

bool MTPDecoder::captureCudaGraphs(cudaStream_t stream)
{
    bool draftProposalCaptureStatus{true};
    bool draftAcceptCaptureStatus{true};
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
        mDraftCacheManager.resetForNewSequences(zeroCacheLensTensor, stream);
        if (!mRuntime.base.executor.prepare(
                kDecodeProfile, mRuntime.deployment.base.resetDims(), mRuntime.base.tensorMap, stream))
        {
            LOG_ERROR("failed to reset base executor context during graph-capture teardown");
        }
    }};

    // MTP drafts a chain (topK=1 shapes); draftingTopK is only the tree candidate fanout.
    int32_t const draftTopK = 1;
    int32_t const paddedDraftProposalSize = mRuntime.deployment.specConfig->draftingStep * draftTopK;
    int32_t const draftingStep = mRuntime.deployment.specConfig->draftingStep;
    int32_t const draftHiddenSize = mRuntime.deployment.specConfig->draftHiddenSize;
    int32_t const baseOutputHiddenDim = mRuntime.deployment.specConfig->baseOutputHiddenDim;
    int32_t const draftVocabSize = mRuntime.deployment.draft->outputVocabSize;

    for (int32_t batchSize = 1; batchSize <= mRuntime.maxRuntimeBatchSize; ++batchSize)
    {
        std::vector<int32_t> simCacheLens(batchSize, kSimulateCacheLength);
        Tensor simCacheLensTensor(simCacheLens.data(), {batchSize}, DeviceType::kCPU, nvinfer1::DataType::kINT32);
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
            reshapeDraftLogits({batchSize, draftTopK, draftVocabSize});
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
            kernel::prepareEagleDraftProposalInputs(mDraftAttentionMask, mDraftProposalSize, draftKVCacheLengths,
                mRuntime.base.pipelineIO.packedAttentionMask, mRuntime.base.pipelineIO.specDecodePositionIds,
                mRuntime.base.pipelineIO.selectTokenIndices, mRuntime.base.pipelineIO.contextLengths, stream);

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
            reshapeDraftLogits({batchSize, draftVocabSize});
            check::check(mRuntime.base.pipelineIO.draftHiddenStatesOut.reshape({batchSize, draftHiddenSize}),
                "Tensor reshape failed");
            int32_t const maxAcceptDepth = std::min(draftingStep + 1, mRuntime.deployment.specConfig->verifySize);
            for (int32_t acceptLength = 1; acceptLength <= maxAcceptDepth; acceptLength++)
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
                kernel::prepareEagleAcceptDecodeTokenInputs(draftKVCacheLengths, mAcceptLength,
                    mRuntime.base.pipelineIO.packedAttentionMask, mRuntime.base.pipelineIO.specDecodePositionIds,
                    mRuntime.base.pipelineIO.selectTokenIndices, mRuntime.base.pipelineIO.contextLengths, stream);

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

        {
            int32_t const verifySize = mRuntime.deployment.specConfig->verifySize;
            int32_t const selectTokenSize = batchSize * verifySize;
            int32_t const packedMaskLen = static_cast<int32_t>(divUp(verifySize, 32));
            check::check(mRuntime.base.pipelineIO.outputLogits.reshape(
                             {selectTokenSize, mRuntime.deployment.base.outputVocabSize}),
                "Tensor reshape failed");
            check::check(mRuntime.base.pipelineIO.baseHiddenStates.reshape({selectTokenSize, baseOutputHiddenDim}),
                "Tensor reshape failed");
            check::check(mRuntime.base.pipelineIO.packedAttentionMask.reshape({batchSize, verifySize, packedMaskLen}),
                "Tensor reshape failed");
            check::check(mRuntime.base.pipelineIO.inputsEmbeds.reshape(
                             {batchSize, verifySize, mRuntime.deployment.base.hiddenSize}),
                "Tensor reshape failed");
            check::check(mRuntime.preprocess.idsInput.reshape({batchSize, verifySize}), "Tensor reshape failed");
            check::check(
                mRuntime.base.pipelineIO.selectTokenIndices.reshape({batchSize, verifySize}), "Tensor reshape failed");
            check::check(mRuntime.base.pipelineIO.contextLengths.reshape({batchSize}), "Tensor reshape failed");
            check::check(mRuntime.base.pipelineIO.specDecodePositionIds.reshape({batchSize, verifySize}),
                "Tensor reshape failed");
            if (mRuntime.preprocess.gemma4Ple)
            {
                mRuntime.preprocess.gemma4Ple->reshapeOutputs(batchSize, verifySize);
            }

            // Run the real tree builder to populate all verify metadata. At capture time the stacked logits buffer
            // holds its zero initialization, which is a valid uniform distribution and yields a deterministic tree;
            // replays read whatever metadata later real builds write into these same buffers.
            buildTreeVerifyInputs(batchSize, stream);
            if (mRuntime.preprocess.deepstack)
            {
                mRuntime.preprocess.deepstack->useZeroTarget(mRuntime.base.tensorMap);
            }

            // MTP: inline prepareBaseVerificationCapture
            mRuntime.base.cacheManager.getMambaCacheManager().reshapeIntermediateStates(batchSize, verifySize);
            if (!mRuntime.base.pipelineIO.specVerifyPhaseMarker.isEmpty())
            {
                check::check(mRuntime.base.pipelineIO.specVerifyPhaseMarker.reshape({1}), "Tensor reshape failed");
            }

            auto const verifyDims = mRuntime.deployment.base.specVerifyDims(batchSize, verifySize);
            baseVerificationCaptureStatus &= mRuntime.base.captureGraph(verifyDims, stream);
        }
    }

    return draftProposalCaptureStatus && draftAcceptCaptureStatus && baseVerificationCaptureStatus;
}

bool MTPDecoder::hasSystemPromptKVCache(SystemPromptCacheKey const& key) const
{
    return mSystemPromptKVCacheDraft.find(key) != mSystemPromptKVCacheDraft.end();
}

void MTPDecoder::restoreSystemPromptKVCache(SystemPromptCacheKey const& key, int32_t batchIdx, cudaStream_t stream)
{
    check::check(mSystemPromptKVCacheDraft.count(key) > 0, "System prompt cache missing for draft model");
    auto& cacheMgrDraft = mDraftCacheManager;
    cacheMgrDraft.restoreKVCache(mSystemPromptKVCacheDraft[key].kvCacheLayers, batchIdx, stream);
}

bool MTPDecoder::runSystemPromptPrefill(DecodingInferenceContext& context)
{
    if (!runDraftModelPrefill(context))
    {
        return false;
    }
    return true;
}

void MTPDecoder::saveSystemPromptKVCache(SystemPromptCacheKey const& key, std::string const& prompt,
    std::vector<tokenizer::Rank> const& tokenizedPrompt, int32_t promptIdsLength, cudaStream_t stream)
{
    auto& cacheMgrDraft = mDraftCacheManager;
    constexpr int32_t CACHE_BATCH_IDX{0};

    SystemPromptKVCache savedKVCacheDraft;
    savedKVCacheDraft.systemPrompt = prompt;
    savedKVCacheDraft.tokenizedPrompt = tokenizedPrompt;
    savedKVCacheDraft.kvCacheLayers = cacheMgrDraft.captureKVCache(CACHE_BATCH_IDX, promptIdsLength, stream);
    mSystemPromptKVCacheDraft.insert({key, std::move(savedKVCacheDraft)});
}

void MTPDecoder::resetForNewSequences(Tensor& reuseLengths, cudaStream_t stream)
{
    mDraftCacheManager.resetForNewSequences(reuseLengths, stream);
}

void MTPDecoder::onBatchEvict(std::vector<int32_t> const& batchMapping, int32_t oldActiveBatch, int32_t newActiveBatch,
    Tensor& deviceBatchMapping, cudaStream_t stream)
{
    KVPageTable& draftPageTable = *mRuntime.base.sharedResources.kvPageTables[1];
    if (draftPageTable.isIdentity())
    {
        mDraftCacheManager.compactBatch(deviceBatchMapping, oldActiveBatch, newActiveBatch, stream);
    }
    else
    {
        mDraftCacheManager.compactBatchSlotState(deviceBatchMapping, oldActiveBatch, newActiveBatch, stream);
        draftPageTable.compactRows(batchMapping, newActiveBatch);
        draftPageTable.upload(stream);
    }
    mDraftCacheManager.setActiveBatchSize(newActiveBatch);

    if (mRuntime.base.pipelineIO.baseHiddenStates.getShape().getNumDims() == 3
        && mRuntime.base.pipelineIO.baseHiddenStates.getShape()[0] == oldActiveBatch && newActiveBatch > 0)
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
}

} // namespace rt
} // namespace trt_edgellm
