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
#include "common/checkMacros.h"
#include "common/cudaUtils.h"
#include "common/logger.h"
#include "common/mathUtils.h"
#include "common/safetensorsUtils.h"
#include "kernels/embeddingKernels/embeddingKernels.h"
#include "kernels/posEncoding/applyRopeWriteKV.h"
#include "kernels/speculative/batchEvictKernels.h"
#include "kernels/speculative/eagleAcceptKernels.h"
#include "kernels/speculative/eagleUtilKernels.h"
#include "profiling/metrics.h"
#include "profiling/nvtx_wrapper.h"
#include "profiling/timer.h"
#include "runtime/config/llmEngineConfig.h"
#include "runtime/decoding/decoderUtils.h"
#include "runtime/decoding/logitBias.h"
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
    SpecDecodeDraftingConfig const& draftingConfig, std::unique_ptr<EngineExecutor> draftExecutor,
    ExternalWeightManager draftWeights, cudaStream_t stream)
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
    int32_t const effectiveDraftTopK = draftingConfig.draftingTopK;
    int32_t const effectiveMaxAcceptDepth = draftingConfig.draftingStep + 1;
    int32_t const draftFullTableLength
        = 1 + effectiveDraftTopK + (draftingConfig.draftingStep - 1) * effectiveDraftTopK * effectiveDraftTopK;

    mDraftProposalSize
        = Tensor({maxRuntimeBatchSize}, DeviceType::kGPU, nvinfer1::DataType::kINT32, "SpecDecode::draftProposalSize");
    mDraftAttentionMask = Tensor({maxRuntimeBatchSize, effectiveMaxDraftProposalSize, effectiveMaxDraftProposalSize},
        DeviceType::kGPU, nvinfer1::DataType::kINT8, "SpecDecode::draftAttentionMask");

    buildTensorMapForSpecDecodeDraft(
        mDraftTensorMap, mRuntime.base.pipelineIO, mRuntime.base.sharedResources, *mRuntime.deployment.draft);

    // Publish externalized draft-engine weights into the draft tensor map, mirroring the base engine. A no-op
    // when the draft model has no externalized weights.
    mDraftExternalWeightManager = std::move(draftWeights);
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

    // EAGLE kernels consume draft-token -> target-token offsets. Older
    // reduced-vocab exports already store offsets in d2t.safetensors, while
    // newer/full-vocab EAGLE3 exports store direct target token ids. Normalize
    // both forms once at load time.
    CUDA_CHECK(
        cudaMemsetAsync(mDraftVocabMappingTable.rawPointer(), 0, mDraftVocabMappingTable.getMemoryCapacity(), stream));
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

DecodingKvHeadroom EagleDecoder::requiredKvHeadroom() const
{
    auto const& config = *mRuntime.deployment.specConfig;
    int64_t const draftExtra = static_cast<int64_t>(config.draftingStep) * config.draftingTopK;
    ELLM_CHECK(config.verifySize > 0 && draftExtra > 0
            && draftExtra <= static_cast<int64_t>(std::numeric_limits<int32_t>::max()),
        "EAGLE KV headroom is outside the supported range");
    return {config.verifySize, static_cast<int32_t>(draftExtra)};
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

bool EagleDecoder::decodeStep(DecodingInferenceContext& context)
{
    if (context.generationRound == 0 && !mCommonStateTracker.draftPrefillOutputsPending())
    {
        if (!runDraftModelPrefill(context))
        {
            LOG_ERROR("Failed to execute prefill step for draft model.");
            return false;
        }
    }
    else if (context.generationRound > 0 && !runDraftModelAcceptToken(context))
    {
        LOG_ERROR("Failed to execute accept token step for draft model.");
        return false;
    }
    mCommonStateTracker.materializePending(context.generationRound, context.activeBatchSize);

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
    mCommonStateTracker.initialize(context);

    if (!runDraftModelPrefill(context))
    {
        LOG_ERROR("Failed to initialize the EAGLE draft model for generation.");
        return false;
    }
    mCommonStateTracker.markDraftPrefillOutputsPending();
    return true;
}

std::vector<int32_t> const& EagleDecoder::commonMaterializedStateLengths() const noexcept
{
    return mCommonStateTracker.commonMaterializedStateLengths();
}

bool EagleDecoder::runDraftModelPrefill(DecodingInferenceContext& context)
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
    check::check(
        mRuntime.base.pipelineIO.outputLogits.reshape({activeBatchSize, draftVocabSize}), "Tensor reshape failed");
    check::check(mRuntime.base.pipelineIO.draftHiddenStatesOut.reshape({activeBatchSize, draftHiddenSize}),
        "Tensor reshape failed");
    check::check(mRuntime.base.pipelineIO.hostContextLengths.reshape({activeBatchSize}), "Tensor reshape failed");

    decoder_utils::zeroActiveRegion(mRuntime.base.pipelineIO.draftHiddenStatesIn, context.stream);

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
    // Unified embedding lookup (text, or vision by inserting image embeddings at the image
    // placeholder positions). Reuses the shared EmbeddingPreprocessor, which generates the
    // multimodal indices on-device and shares this decoder's embedding table.
    mRuntime.preprocess.embeddingPreprocessor.embed(
        mRuntime.preprocess.idsInput, context.visualEmbeddings, std::nullopt, mRuntime.base.pipelineIO, context.stream);
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
        mDraftCacheManager.commitSequenceLength(mRuntime.base.pipelineIO.contextLengths, context.stream);
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
    int32_t const draftVocabSize = mRuntime.deployment.draft->outputVocabSize;
    int32_t const draftTopK = mRuntime.deployment.specConfig->draftingTopK;
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

        check::check(mRuntime.base.pipelineIO.outputLogits.reshape({activeBatchSize * draftTopK, draftVocabSize}),
            "Tensor reshape failed");
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

    check::check(mRuntime.sampling.indices.reshape({activeBatchSize, mRuntime.deployment.specConfig->verifySize}),
        "Tensor reshape failed");
    selectAllTopK(mDraftTokenScoreFullTable, std::nullopt, mRuntime.sampling.indices,
        mRuntime.deployment.specConfig->verifySize, mRuntime.sampling.workspace, context.stream);

    check::check(mRuntime.preprocess.idsInput.reshape({activeBatchSize, mRuntime.deployment.specConfig->verifySize}),
        "Tensor reshape failed");
    check::check(mDraftAttentionMask.reshape({activeBatchSize, mRuntime.deployment.specConfig->verifySize,
                     mRuntime.deployment.specConfig->verifySize}),
        "Tensor reshape failed");
    check::check(mRuntime.base.pipelineIO.packedAttentionMask.reshape(
                     {activeBatchSize, mRuntime.deployment.specConfig->verifySize,
                         static_cast<int64_t>(divUp(mRuntime.deployment.specConfig->verifySize, 32))}),
        "Tensor reshape failed");
    kernel::constructVerificationDraftTree(mDraftTokenIdsFullTable, mDraftTokenPredecessorFullTable,
        mRuntime.sampling.indices, mRuntime.preprocess.idsInput, mDraftAttentionMask, context.stream);

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

    int32_t const activeBatchSize = context.activeBatchSize;
    int32_t const baseOutputHiddenDim = mRuntime.deployment.specConfig->baseOutputHiddenDim;

    check::check(mRuntime.preprocess.idsInput.getShape()[0] == activeBatchSize
            && mRuntime.preprocess.idsInput.getShape()[1] == mRuntime.deployment.specConfig->verifySize,
        "IdsInput shall have shape [batch_size, verify_size]");
    check::check(mDraftAttentionMask.getShape()[0] == activeBatchSize
            && mDraftAttentionMask.getShape()[1] == mRuntime.deployment.specConfig->verifySize
            && mDraftAttentionMask.getShape()[2] == mRuntime.deployment.specConfig->verifySize,
        "Draft attention mask shall have shape [batch_size, verify_size, verify_size]");

    check::check(mRuntime.base.pipelineIO.inputsEmbeds.reshape({activeBatchSize,
                     mRuntime.deployment.specConfig->verifySize, mRuntime.deployment.base.hiddenSize}),
        "Tensor reshape failed");
    kernel::embeddingLookup(mRuntime.preprocess.idsInput, mRuntime.preprocess.embedding.table,
        mRuntime.preprocess.embedding.scalesAsOptional(), mRuntime.base.pipelineIO.inputsEmbeds, context.stream);
    int32_t const selectTokenSize = activeBatchSize * mRuntime.deployment.specConfig->verifySize;
    check::check(
        mRuntime.base.pipelineIO.outputLogits.reshape({selectTokenSize, mRuntime.deployment.base.outputVocabSize}),
        "Tensor reshape failed");
    check::check(mRuntime.base.pipelineIO.baseHiddenStates.reshape({selectTokenSize, baseOutputHiddenDim}),
        "Tensor reshape failed");

    {
        int32_t const verifySize = mRuntime.deployment.specConfig->verifySize;
        Tensor const& baseKVCacheLengths = mRuntime.base.cacheManager.getKVCacheLengths();
        check::check(mRuntime.base.pipelineIO.selectTokenIndices.reshape({activeBatchSize, verifySize}),
            "Tensor reshape failed");
        check::check(mRuntime.base.pipelineIO.contextLengths.reshape({activeBatchSize}), "Tensor reshape failed");
        check::check(mRuntime.base.pipelineIO.specDecodePositionIds.reshape({activeBatchSize, verifySize}),
            "Tensor reshape failed");
        kernel::prepareEagleBaseTreeDecodingInputs(mDraftAttentionMask, baseKVCacheLengths,
            mRuntime.base.pipelineIO.packedAttentionMask, mRuntime.base.pipelineIO.specDecodePositionIds,
            mRuntime.base.pipelineIO.selectTokenIndices, mRuntime.base.pipelineIO.contextLengths, context.stream);
    }

    if (mRuntime.preprocess.deepstack)
    {
        mRuntime.preprocess.deepstack->useZeroTarget(mRuntime.base.tensorMap);
    }

    // Eagle: no prepareBaseVerificationState needed

    auto const verifyDims
        = mRuntime.deployment.base.specVerifyDims(activeBatchSize, mRuntime.deployment.specConfig->verifySize);
    bool verifySuccess
        = mRuntime.base.executor.prepare(kDecodeProfile, verifyDims, mRuntime.base.tensorMap, context.stream);
    if (verifySuccess)
    {
        verifySuccess = mRuntime.base.executor.execute(context.stream);
    }
    if (!verifySuccess)
    {
        LOG_ERROR("Failed to execute base verification step for base model.");
        return false;
    }

    // GCOVR_EXCL_START
    if (context.hasLogitBias)
    {
        applyLogitBiasRepeatedRows(mRuntime.logitBias, mRuntime.base.pipelineIO.outputLogits, context,
            mRuntime.deployment.specConfig->verifySize, context.stream);
    }
    // GCOVR_EXCL_STOP

    int32_t const maxAcceptDepth = mRuntime.deployment.specConfig->draftingStep + 1;
    check::check(mAcceptedTokenIds.reshape({activeBatchSize, maxAcceptDepth}), "Tensor reshape failed");
    check::check(mAcceptedTokenIndices.reshape({activeBatchSize, maxAcceptDepth}), "Tensor reshape failed");
    check::check(mAcceptLength.reshape({activeBatchSize}), "Tensor reshape failed");

    OptionalInputTensor vocabMappingTable = (mRuntime.deployment.base.reducedVocabSize > 0)
        ? std::optional{std::ref(mRuntime.sampling.baseVocabMappingTable)}
        : std::nullopt;
    kernel::eagleAccept(mRuntime.base.pipelineIO.outputLogits, mRuntime.preprocess.idsInput, mDraftAttentionMask,
        mAcceptedTokenIds, mAcceptedTokenIndices, mAcceptLength, vocabMappingTable,
        mRuntime.sampling.workspace.rawPointer(), mRuntime.sampling.workspace.getMemoryCapacity(), context.stream);

    auto& cacheMgrBase = mRuntime.base.cacheManager;
    Tensor const& kvCacheLengths = cacheMgrBase.getKVCacheLengths();
    auto& kvMgrBase = cacheMgrBase.getKVCacheManager();
    auto const kvHeadDimGroups = cacheMgrBase.getKVHeadDimGroups();
    auto const kvCacheType = kvMgrBase.getConfig().kvCacheType;

    check::check(mRuntime.base.pipelineIO.baseHiddenStates.reshape(
                     {activeBatchSize, mRuntime.deployment.specConfig->verifySize, baseOutputHiddenDim}),
        "Tensor reshape failed");

    // Accepted-KV writes must follow the same page table as the base AttentionPlugin binding.
    auto const& basePageTable = *mRuntime.base.sharedResources.kvPageTables[0];
    int32_t const* basePageTablePtr = basePageTable.kernelView().dataPointer<int32_t>();
    int32_t const baseNumPages = kvMgrBase.numPages();
    int32_t const baseMaxPagesPerSeq = basePageTable.maxPagesPerSeq();

    decoder_utils::clampAcceptLengthsToRemainingGeneration(context, mHostAcceptLengths, mAcceptLength, context.stream);

    for (auto const& group : kvHeadDimGroups)
    {
        kernel::eagleBaseCommitKVCache(mAcceptedTokenIndices, mAcceptLength, kvCacheLengths, group.deviceLayerInfos,
            group.numLayers, group.headDim, group.maxKVHeads, activeBatchSize, maxAcceptDepth, kvCacheType,
            context.stream, basePageTablePtr, baseNumPages, baseMaxPagesPerSeq);
    }

    kernel::eagleBaseAssembleHiddenState(
        mAcceptedTokenIndices, mAcceptLength, mRuntime.base.pipelineIO.baseHiddenStates, context.stream);
    mRuntime.base.cacheManager.commitSequenceLength(mAcceptLength, context.stream);
    // Eagle: no commitAcceptedBaseState needed

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
    int32_t const* const hostAcceptLengths = mHostAcceptLengths.dataPointer<int32_t>();
    mCommonStateTracker.recordAccepted(hostAcceptLengths, activeBatchSize);

    if (context.numLogprobs > 0)
    {
        decoder_utils::collectSpecLogprobsFromHost(mRuntime, context, activeBatchSize, maxAcceptDepth,
            mHostAcceptLengths.dataPointer<int32_t>(), context.numLogprobs);
    }

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
    int32_t const draftVocabSize = mRuntime.deployment.draft->outputVocabSize;
    int64_t const inputIdsLength = mRuntime.base.pipelineIO.baseHiddenStates.getShape()[1];

    check::check(mRuntime.preprocess.idsInput.reshape({activeBatchSize, inputIdsLength}), "Tensor reshape failed");
    check::check(
        mRuntime.base.pipelineIO.draftHiddenStatesIn.reshape({activeBatchSize, inputIdsLength, draftHiddenSize}),
        "Tensor reshape failed");
    check::check(
        mRuntime.base.pipelineIO.outputLogits.reshape({activeBatchSize, draftVocabSize}), "Tensor reshape failed");
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
        mRuntime.preprocess.embedding.scalesAsOptional(), mRuntime.base.pipelineIO.inputsEmbeds, context.stream);
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
    mDraftCacheManager.commitSequenceLength(mAcceptLength, context.stream);

    return true;
}

bool EagleDecoder::captureCudaGraphs(cudaStream_t stream)
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

    int32_t const draftTopK = mRuntime.deployment.specConfig->draftingTopK;
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
            check::check(mRuntime.base.pipelineIO.outputLogits.reshape({batchSize, draftTopK, draftVocabSize}),
                "Tensor reshape failed");
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
            check::check(
                mRuntime.base.pipelineIO.outputLogits.reshape({batchSize, draftVocabSize}), "Tensor reshape failed");
            check::check(mRuntime.base.pipelineIO.draftHiddenStatesOut.reshape({batchSize, draftHiddenSize}),
                "Tensor reshape failed");
            for (int32_t acceptLength = 1; acceptLength <= draftingStep + 1; acceptLength++)
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
            check::check(mRuntime.base.pipelineIO.outputLogits.reshape(
                             {selectTokenSize, mRuntime.deployment.base.outputVocabSize}),
                "Tensor reshape failed");
            check::check(mRuntime.base.pipelineIO.baseHiddenStates.reshape({selectTokenSize, baseOutputHiddenDim}),
                "Tensor reshape failed");
            check::check(mDraftAttentionMask.reshape({batchSize, verifySize, verifySize}), "Tensor reshape failed");
            check::check(mRuntime.base.pipelineIO.packedAttentionMask.reshape(
                             {batchSize, verifySize, static_cast<int64_t>(divUp(verifySize, 32))}),
                "Tensor reshape failed");
            check::check(mRuntime.base.pipelineIO.inputsEmbeds.reshape(
                             {batchSize, verifySize, mRuntime.deployment.base.hiddenSize}),
                "Tensor reshape failed");
            check::check(mRuntime.preprocess.idsInput.reshape({batchSize, verifySize}), "Tensor reshape failed");
            if (mRuntime.preprocess.gemma4Ple)
            {
                mRuntime.preprocess.gemma4Ple->reshapeOutputs(batchSize, verifySize);
            }

            Tensor const& baseKVCacheLengths = mRuntime.base.cacheManager.getKVCacheLengths();
            check::check(
                mRuntime.base.pipelineIO.selectTokenIndices.reshape({batchSize, verifySize}), "Tensor reshape failed");
            check::check(mRuntime.base.pipelineIO.contextLengths.reshape({batchSize}), "Tensor reshape failed");
            check::check(mRuntime.base.pipelineIO.specDecodePositionIds.reshape({batchSize, verifySize}),
                "Tensor reshape failed");
            kernel::prepareEagleBaseTreeDecodingInputs(mDraftAttentionMask, baseKVCacheLengths,
                mRuntime.base.pipelineIO.packedAttentionMask, mRuntime.base.pipelineIO.specDecodePositionIds,
                mRuntime.base.pipelineIO.selectTokenIndices, mRuntime.base.pipelineIO.contextLengths, stream);
            if (mRuntime.preprocess.deepstack)
            {
                mRuntime.preprocess.deepstack->useZeroTarget(mRuntime.base.tensorMap);
            }

            // Eagle: no prepareBaseVerificationCapture needed

            auto const verifyDims = mRuntime.deployment.base.specVerifyDims(batchSize, verifySize);
            baseVerificationCaptureStatus &= mRuntime.base.captureGraph(verifyDims, stream);
        }
    }

    return draftProposalCaptureStatus && draftAcceptCaptureStatus && baseVerificationCaptureStatus;
}

bool EagleDecoder::hasSystemPromptKVCache(SystemPromptCacheKey const& key) const
{
    return mSystemPromptKVCacheDraft.find(key) != mSystemPromptKVCacheDraft.end();
}

void EagleDecoder::restoreSystemPromptKVCache(SystemPromptCacheKey const& key, int32_t batchIdx, cudaStream_t stream)
{
    check::check(mSystemPromptKVCacheDraft.count(key) > 0, "System prompt cache missing for draft model");
    auto& cacheMgrDraft = mDraftCacheManager;
    cacheMgrDraft.restoreKVCache(mSystemPromptKVCacheDraft[key].kvCacheLayers, batchIdx, stream);
}

bool EagleDecoder::runSystemPromptPrefill(DecodingInferenceContext& context)
{
    if (!runDraftModelPrefill(context))
    {
        return false;
    }
    return true;
}

void EagleDecoder::saveSystemPromptKVCache(SystemPromptCacheKey const& key, std::string const& prompt,
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

void EagleDecoder::resetForNewSequences(Tensor& reuseLengths, cudaStream_t stream)
{
    mDraftCacheManager.resetForNewSequences(reuseLengths, stream);
    mCommonStateTracker.reset();
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
        compactDraftPrefillOutput(mRuntime.base.pipelineIO.outputLogits, "draft-prefill logits");
        compactDraftPrefillOutput(mRuntime.base.pipelineIO.draftHiddenStatesOut, "draft-prefill hidden state");
    }

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
