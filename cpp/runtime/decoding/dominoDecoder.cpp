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

#include "runtime/decoding/dominoDecoder.h"

#include "common/bindingNames.h"
#include "common/checkMacros.h"
#include "common/cudaUtils.h"
#include "common/logger.h"
#include "common/safetensorsUtils.h"
#include "kernels/speculative/dominoRolloutKernels.h"
#include "profiling/nvtx_wrapper.h"
#include "runtime/config/llmEngineConfig.h"
#include "runtime/decoding/dflashDecodeUtils.h"

#include <algorithm>
#include <filesystem>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace trt_edgellm
{
namespace rt
{
namespace
{

Tensor takeRequiredHeadsTensor(std::vector<Tensor>& tensors, std::string const& name)
{
    auto const it
        = std::find_if(tensors.begin(), tensors.end(), [&](Tensor const& tensor) { return tensor.getName() == name; });
    check::check(it != tensors.end(), "Domino heads sidecar is missing tensor '" + name + "'");
    Tensor output = std::move(*it);
    tensors.erase(it);
    return output;
}

void validateHeadsTensor(Tensor const& tensor, std::string const& name, int32_t rows, int32_t cols)
{
    check::check(tensor.getDataType() == trt_edgellm::DataType::kHALF, name + " must be FP16");
    check::check(tensor.getShape().getNumDims() == 2, name + " must be rank-2");
    check::check(tensor.getShape()[0] == rows && tensor.getShape()[1] == cols,
        name + " must have shape [" + std::to_string(rows) + ", " + std::to_string(cols) + "]");
}

} // namespace

DominoDecoder::DominoDecoder(DecodingRuntimeContext& runtime, std::filesystem::path const& engineDir,
    dflash_utils::CachedBlockDraftRuntimeConfig blockDraftConfig, std::unique_ptr<EngineExecutor> draftExecutor,
    ExternalWeightManager draftWeights, cudaStream_t stream)
    : DFlashDecoder(
          runtime, engineDir, std::move(blockDraftConfig), std::move(draftExecutor), std::move(draftWeights), stream)
{
    ELLM_CHECK(mBlockDraft.correctionHead == dflash_utils::BlockDraftCorrectionHead::kDomino,
        "DominoDecoder requires a Domino cached-block-draft runtime config.");
    loadCorrectionHead(engineDir, stream);
}

void DominoDecoder::reshapeDraftOutputs(int32_t batchSize)
{
    int32_t const BS = mBlockDraft.blockSize;
    check::check(mPoolIds.reshape({batchSize, mBlockDraft.dominoPoolSize}), "Tensor reshape failed");
    check::check(mPoolLogits.reshape({batchSize, BS - 1, mBlockDraft.dominoPoolSize}), "Tensor reshape failed");
    check::check(mPoolCodebook.reshape({batchSize, mBlockDraft.dominoPoolSize, mBlockDraft.dominoProjDim}),
        "Tensor reshape failed");
    check::check(mZProj.reshape({batchSize, BS - 2, mBlockDraft.dominoProjDim}), "Tensor reshape failed");
    check::check(mStackedScores.reshape({batchSize, BS, mBlockDraft.dominoPoolSize}), "Tensor reshape failed");
}

bool DominoDecoder::produceProposal(DecodingInferenceContext& context)
{
    return runRollout(context);
}

void DominoDecoder::finalizeTreeTokens(DecodingInferenceContext& context)
{
    kernel::dominoRemapPoolSlots(mTreeTokenIds, mVerifyTokenIds, mValidCounts, mPoolIds, context.activeBatchSize,
        mBlockDraft.verifyTokenCount, mBlockDraft.dominoPoolSize, context.stream);
}

void DominoDecoder::loadCorrectionHead(std::filesystem::path const& engineDir, cudaStream_t stream)
{
    auto const& deployment = mRuntime.deployment;
    LLMEngineConfig const& draftCfg = *deployment.draft;
    int32_t const maxBatch = deployment.maxRuntimeBatchSize();
    int32_t const poolSize = mBlockDraft.dominoPoolSize;
    int32_t const projDim = mBlockDraft.dominoProjDim;
    int32_t const gruHiddenDim = mBlockDraft.dominoGruHiddenDim;
    int32_t const BS = mBlockDraft.blockSize;
    // Positions 1..BS-1 are proposals; position 0 of the pool logits is the uncorrected first proposal
    // and the GRU corrects the remaining BS-2 positions.
    int32_t const feedbackLen = BS - 2;

    ELLM_CHECK(feedbackLen >= 1, "Domino requires block_size >= 3.");
    ELLM_CHECK(poolSize > 0 && projDim > 0 && gruHiddenDim > 0,
        "Domino requires a positive pool/projection/GRU contract in the draft engine config.");
    ELLM_CHECK(projDim % 2 == 0 && gruHiddenDim % 2 == 0,
        "Domino rollout kernels require even projection and GRU dimensions.");

    mPoolIds
        = Tensor({maxBatch, poolSize}, DeviceType::kGPU, trt_edgellm::DataType::kINT32, "Domino::poolCandidateIds");
    mPoolLogits = Tensor(
        {maxBatch, BS - 1, poolSize}, DeviceType::kGPU, trt_edgellm::DataType::kFLOAT, "Domino::poolBaseLogits");
    mStackedScores
        = Tensor({maxBatch, BS, poolSize}, DeviceType::kGPU, trt_edgellm::DataType::kFLOAT, "Domino::stackedScores");
    mPoolCodebook
        = Tensor({maxBatch, poolSize, projDim}, DeviceType::kGPU, trt_edgellm::DataType::kHALF, "Domino::poolCodebook");
    mZProj = Tensor({maxBatch, feedbackLen, projDim}, DeviceType::kGPU, trt_edgellm::DataType::kHALF, "Domino::zProj");
    for (auto& state : mGruState)
    {
        state = Tensor({maxBatch, gruHiddenDim}, DeviceType::kGPU, trt_edgellm::DataType::kHALF, "Domino::gruState");
    }
    mCorrection = Tensor({maxBatch, projDim}, DeviceType::kGPU, trt_edgellm::DataType::kFLOAT, "Domino::correction");
    mWinnerSlots = Tensor({maxBatch}, DeviceType::kGPU, trt_edgellm::DataType::kINT64, "Domino::winnerSlots");

    mDraftTensorMap.set(binding_names::kSpecPoolCandidateIds, mPoolIds);
    mDraftTensorMap.set(binding_names::kSpecPoolBaseLogits, mPoolLogits);
    mDraftTensorMap.set(binding_names::kSpecPoolCodebook, mPoolCodebook);
    mDraftTensorMap.set(binding_names::kSpecProposalZProj, mZProj);

    std::string const headsFile = draftCfg.dominoHeadsFile.empty() ? std::string(binding_names::kDominoHeadsFileName)
                                                                   : draftCfg.dominoHeadsFile;
    auto const headsPath = engineDir / headsFile;
    ELLM_CHECK(std::filesystem::exists(headsPath), "Domino heads sidecar missing: " + headsPath.string());
    std::vector<Tensor> headsTensors;
    ELLM_CHECK(safetensors::loadSafetensors(headsPath, headsTensors, stream),
        "Failed to load Domino heads sidecar from " + headsPath.string());
    mGruInputTable = takeRequiredHeadsTensor(headsTensors, "gru_input_table");
    mGruWeightHh = takeRequiredHeadsTensor(headsTensors, "gru_weight_hh");
    mProjStateWeight = takeRequiredHeadsTensor(headsTensors, "proj_state_weight");
    check::check(headsTensors.empty(), "Domino heads sidecar contains unexpected tensors");
    check::check(mGruInputTable.getDataType() == trt_edgellm::DataType::kHALF
            && mGruInputTable.getShape().getNumDims() == 2 && mGruInputTable.getShape()[1] == 3 * gruHiddenDim
            && mGruInputTable.getShape()[0] >= mBlockDraft.draftVocabSize,
        "gru_input_table must be FP16 [vocab >= " + std::to_string(mBlockDraft.draftVocabSize) + ", "
            + std::to_string(3 * gruHiddenDim) + "]");
    validateHeadsTensor(mGruWeightHh, "gru_weight_hh", 3 * gruHiddenDim, gruHiddenDim);
    validateHeadsTensor(mProjStateWeight, "proj_state_weight", projDim, gruHiddenDim);

    LOG_INFO("Domino correction head initialized: poolSize=%d, projDim=%d, gruHiddenDim=%d, feedbackLen=%d, heads=%s",
        poolSize, projDim, gruHiddenDim, feedbackLen, headsPath.string().c_str());
}

bool DominoDecoder::runRollout(DecodingInferenceContext& context)
{
    NVTX_SCOPED_RANGE(nvtx_domino_rollout, "DominoDecoder::runRollout", nvtx_colors::LIGHT_ORANGE);

    int32_t const activeBatchSize = context.activeBatchSize;
    int32_t const BS = mBlockDraft.blockSize;
    int32_t const feedbackLen = BS - 2;
    int32_t const poolSize = mBlockDraft.dominoPoolSize;
    int32_t const gruHiddenDim = mBlockDraft.dominoGruHiddenDim;

    // The chain lives in draftTokenIds positions [1, BS-1]; position 0 stays unused. Every position's
    // pool scores also go to mStackedScores (row d = proposal position d), which the DDTree builder
    // turns into the verify tree; with candidateTopK=1 that reproduces this chain.
    check::check(mDraftTokenIds.reshape({activeBatchSize, BS}), "Tensor reshape failed");
    kernel::dominoSelectFirstProposal(mPoolLogits, BS - 1, mPoolIds, mDraftTokenIds, &mStackedScores, BS, mWinnerSlots,
        /*draftTokenStride=*/BS, /*tokenOffset=*/1, activeBatchSize, poolSize, context.stream);

    // Seed the GRU from [last_accepted, first_draft]; the state never crosses rounds.
    CUDA_CHECK(cudaMemsetAsync(mGruState[0].rawPointer(), 0,
        static_cast<size_t>(activeBatchSize) * gruHiddenDim * sizeof(half), context.stream));
    kernel::dominoGruUpdate(mGruInputTable, mLastAcceptedTokens, /*tokenStride=*/1, /*tokenOffset=*/0, mGruWeightHh,
        mGruState[0], mGruState[1], activeBatchSize, gruHiddenDim, context.stream);
    kernel::dominoGruUpdate(mGruInputTable, mDraftTokenIds, /*tokenStride=*/BS, /*tokenOffset=*/1, mGruWeightHh,
        mGruState[1], mGruState[0], activeBatchSize, gruHiddenDim, context.stream);

    int32_t current = 0;
    for (int32_t step = 0; step < feedbackLen; ++step)
    {
        kernel::DominoStepLayout const layout{/*zRow=*/step, /*zRows=*/feedbackLen, /*logitsRow=*/step + 1,
            /*poolRows=*/BS - 1, /*scoreRow=*/step + 1, /*scoreRows=*/BS};
        kernel::dominoCorrectStep(mZProj, mPoolLogits, mPoolCodebook, mPoolIds, mProjStateWeight, mGruState[current],
            mDraftTokenIds, &mStackedScores, kernel::DominoCorrectScratch{mCorrection, mWinnerSlots}, layout,
            /*draftTokenStride=*/BS, /*tokenOffset=*/2 + step, activeBatchSize, poolSize, mBlockDraft.dominoProjDim,
            gruHiddenDim, context.stream);
        if (step + 1 < feedbackLen)
        {
            kernel::dominoGruUpdate(mGruInputTable, mDraftTokenIds, /*tokenStride=*/BS, /*tokenOffset=*/2 + step,
                mGruWeightHh, mGruState[current], mGruState[1 - current], activeBatchSize, gruHiddenDim,
                context.stream);
            current = 1 - current;
        }
    }

    return checkCudaLastError("Domino rollout");
}

} // namespace rt
} // namespace trt_edgellm
