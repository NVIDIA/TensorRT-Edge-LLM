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

#include "runtime/decoding/dflashDecoder.h"
#include "runtime/decoding/specDecodeUtils.h"

#include "common/bindingNames.h"
#include "common/checkMacros.h"
#include "common/cudaUtils.h"
#include "common/logger.h"
#include "common/mathUtils.h"
#include "common/safetensorsUtils.h"
#include "kernels/embeddingKernels/embeddingKernels.h"
#include "kernels/posEncoding/applyRopeWriteKV.h"
#include "kernels/speculative/dflash2CandidateSelector.h"
#include "kernels/speculative/dflashRuntimeKernels.h"
#include "kernels/speculative/dsparkKernels.h"
#include "kernels/speculative/requestStableRngKernels.h"
#include "kernels/speculative/speculativeKernels.h"
#include "kernels/speculative/speculativeSampling.h"
#include "profiling/metrics.h"
#include "profiling/nvtx_wrapper.h"
#include "profiling/timer.h"
#include "runtime/config/llmEngineConfig.h"
#include "runtime/decoding/decoderUtils.h"
#include "runtime/decoding/dflashDecodeUtils.h"
#include "runtime/decoding/guidedDecoder.h"
#include "runtime/decoding/logitBias.h"
#include "runtime/decoding/requestStableRng.h"
#include "sampler/sampling.h"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <optional>
#include <utility>
#include <vector>

namespace trt_edgellm
{
namespace rt
{
namespace
{
Tensor takeRequiredSelectorTensor(std::vector<Tensor>& tensors, std::string const& name)
{
    auto const it
        = std::find_if(tensors.begin(), tensors.end(), [&](Tensor const& tensor) { return tensor.getName() == name; });
    check::check(it != tensors.end(), "DFlash2 selector sidecar is missing tensor '" + name + "'");
    Tensor output = std::move(*it);
    tensors.erase(it);
    return output;
}

void validateSelectorCodebook(Tensor const& tensor, std::string const& name, int32_t vocabSize, int32_t rank)
{
    check::check(tensor.getDataType() == trt_edgellm::DataType::kHALF, name + " must be FP16");
    check::check(tensor.getShape().getNumDims() == 2, name + " must be rank-2");
    check::check(tensor.getShape()[0] == vocabSize && tensor.getShape()[1] == rank,
        name + " must have shape [" + std::to_string(vocabSize) + ", " + std::to_string(rank) + "]");
}

//! The LiLiCorr draft gathers raw rows of this table in-graph, so it must be a dense, unscaled FP16 [V, H] table.
void validateLiLiCorrEmbeddingTable(EmbeddingData const& embedding, int64_t vocabSize, int64_t hiddenSize)
{
    auto const& table = embedding.table;
    auto const& descriptor = embedding.descriptor;
    std::string const expected = "[" + std::to_string(vocabSize) + ", " + std::to_string(hiddenSize) + "]";
    ELLM_CHECK(descriptor.format == kernel::EmbeddingTableFormat::kFP16
            && table.getDataType() == trt_edgellm::DataType::kHALF
            && embedding.tableScalingFactor.getShape().volume() == 0,
        "LiLiCorr draft binds the runtime embedding table as its FP16 "
            + std::string(binding_names::kLiLiCorrEmbeddingTable) + " input; export the embedding unquantized (FP16).");
    ELLM_CHECK(descriptor.modelOutputScale == 1.0F,
        std::string("LiLiCorr draft gathers raw embedding rows and cannot apply an embedding output scale."));
    ELLM_CHECK(
        table.getShape().getNumDims() == 2 && table.getShape()[0] == vocabSize && table.getShape()[1] == hiddenSize,
        "LiLiCorr draft embedding table must have shape " + expected);
    ELLM_CHECK(table.getDeviceType() == DeviceType::kGPU,
        std::string("LiLiCorr drafts do not support --offloadEmbedding: the draft binds the embedding table as an "
                    "engine input, so it must reside on the GPU."));
}
} // namespace

void DFlashDecoder::initializeDFlash2(
    std::filesystem::path const& engineDir, ExternalWeightManager draftWeights, cudaStream_t stream)
{
    auto const& deployment = mRuntime.deployment;
    auto const& baseCfg = deployment.base;
    ELLM_CHECK(deployment.specConfig.has_value(), "DFlashDecoder: specConfig is required.");
    ELLM_CHECK(deployment.draft.has_value(), "DFlashDecoder: draft config is required.");
    ELLM_CHECK(isCachedBlockDraftMode(baseCfg.specDecodeType),
        "DFlashDecoder requires a cached-block speculative base engine.");
    ELLM_CHECK(baseCfg.specDecodeType == mBlockDraft.userMode,
        "DFlashDecoder normalized user mode does not match the base engine config.");
    ELLM_CHECK(baseCfg.specDecodeType == SpecDecodeMode::kDFlash && baseCfg.dflashVersion == DFlashVersion::kV2,
        "DFlashDecoder requires spec_decode_type=dflash with dflash_config.version=2.");
    bool const tree = useTreeVerification();
    ELLM_CHECK(!tree || !isLiLiCorr(), "DFlash2 LiLiCorr drafts do not use DDTree; draftTopK must be 1.");
    ELLM_CHECK(mBlockDraft.blockSize >= 2 && mBlockDraft.blockSize <= 16
            && mBlockDraft.proposalLen == mBlockDraft.blockSize - 1
            && (tree ? mBlockDraft.verifyTokenCount > mBlockDraft.candidateTopK
                     : mBlockDraft.verifyTokenCount == mBlockDraft.blockSize),
        "DFlashDecoder requires block_size in [2, 16] with verify_size=block_size (chain) or verify_size > "
        "draftTopK (DDTree).");

    int32_t const maxBatch = deployment.maxRuntimeBatchSize();

    ELLM_CHECK(mDraftExecutor != nullptr, std::string(userModeName()) + " decoding requires a validated draft engine.");

    mDraftInputsEmbeds = Tensor({maxBatch, mBlockDraft.blockSize, mBlockDraft.draftHiddenSize}, DeviceType::kGPU,
        trt_edgellm::DataType::kHALF, "DFlash2Draft::inputsEmbeds");
    mDraftTargetHidden = Tensor({maxBatch, mBlockDraft.blockSize, mBlockDraft.baseOutputHiddenDim}, DeviceType::kGPU,
        trt_edgellm::DataType::kHALF, "DFlash2Draft::targetHiddenScratch");
    int32_t const proposalLen = mBlockDraft.proposalLen;
    int32_t const supportK = proposalSupportK();
    int32_t const selectorRank = deployment.draft->specSelectorRank;
    mDraftTokenIds
        = Tensor({maxBatch, proposalLen}, DeviceType::kGPU, trt_edgellm::DataType::kINT32, "DFlash2Draft::proposalIds");
    mProposalSupportIds = Tensor({maxBatch, proposalLen, supportK}, DeviceType::kGPU, trt_edgellm::DataType::kINT32,
        "DFlash2Draft::proposalSupportIds");
    mProposalSupportProbs = Tensor({maxBatch, proposalLen, supportK}, DeviceType::kGPU, trt_edgellm::DataType::kFLOAT,
        "DFlash2Draft::proposalSupportProbs");
    if (isLiLiCorr())
    {
        mLiLiCorrTransitionScores = Tensor({maxBatch, proposalLen, supportK, supportK}, DeviceType::kGPU,
            trt_edgellm::DataType::kFLOAT, "DFlash2Draft::lilicorrTransitionScores");
    }
    else
    {
        mProposalUnaryValues = Tensor({maxBatch, proposalLen, supportK}, DeviceType::kGPU,
            trt_edgellm::DataType::kFLOAT, "DFlash2Draft::proposalUnaryValues");
        mProposalProjectedHidden = Tensor({maxBatch, proposalLen, selectorRank}, DeviceType::kGPU,
            trt_edgellm::DataType::kHALF, "DFlash2Draft::proposalProjectedHidden");
    }
    mProposalUniforms = Tensor(
        {maxBatch, proposalLen}, DeviceType::kGPU, trt_edgellm::DataType::kFLOAT, "DFlash2Draft::proposalUniforms");
    mSamplingTemperatures
        = Tensor({maxBatch}, DeviceType::kGPU, trt_edgellm::DataType::kFLOAT, "DFlash2Draft::temperatures");
    mProposalGreedyMask
        = Tensor({maxBatch}, DeviceType::kGPU, trt_edgellm::DataType::kINT32, "DFlash2Draft::greedyMask");
    mLastAcceptedTokens
        = Tensor({maxBatch}, DeviceType::kGPU, trt_edgellm::DataType::kINT32, "DFlash2Draft::lastAcceptedTokens");

    int32_t const packedMaskLen = divUp(mBlockDraft.blockSize, 32);
    mDraftPackedAttentionMask = Tensor({maxBatch, mBlockDraft.blockSize, packedMaskLen}, DeviceType::kGPU,
        trt_edgellm::DataType::kINT32, "DFlash2Draft::packedMask");
    mDraftAttentionPosId = Tensor({maxBatch, mBlockDraft.blockSize}, DeviceType::kGPU, trt_edgellm::DataType::kINT32,
        "DFlash2Draft::positionIds");
    mDraftContextLengths
        = Tensor({maxBatch}, DeviceType::kGPU, trt_edgellm::DataType::kINT32, "DFlash2Draft::contextLengths");
    mDraftDeltaLenCommit
        = Tensor({maxBatch}, DeviceType::kGPU, trt_edgellm::DataType::kINT32, "DFlash2Draft::deltaLenCommit");
    mDraftDeltaLens = Tensor({maxBatch}, DeviceType::kGPU, trt_edgellm::DataType::kINT32, "DFlash2Draft::deltaLens");

    mDraftTensorMap.set(binding_names::kInputsEmbeds, mDraftInputsEmbeds);
    mDraftTensorMap.set(binding_names::kDFlashTargetHiddenConcat, mDraftTargetHidden);
    mDraftTensorMap.set(binding_names::kSpecProposalSupportIds, mProposalSupportIds);
    if (isLiLiCorr())
    {
        mDraftTensorMap.set(binding_names::kSpecLiLiCorrTransitionScores, mLiLiCorrTransitionScores);
        validateLiLiCorrEmbeddingTable(mRuntime.preprocess.embedding, baseCfg.vocabSize, mBlockDraft.draftHiddenSize);
        mDraftTensorMap.set(binding_names::kLiLiCorrEmbeddingTable, mRuntime.preprocess.embedding.table);
    }
    else
    {
        mDraftTensorMap.set(binding_names::kSpecProposalUnaryValues, mProposalUnaryValues);
        mDraftTensorMap.set(binding_names::kSpecProposalProjectedHidden, mProposalProjectedHidden);
    }
    mDraftTensorMap.set(binding_names::kAttentionMask, mDraftPackedAttentionMask);
    mDraftTensorMap.set(binding_names::kAttentionPosId, mDraftAttentionPosId);
    LLMEngineConfig const& draftCfg = *deployment.draft;
    initializeDraftRaggedBindings(draftCfg);

    // KV cache bindings: bind to draft cache manager's combined KV cache (index 1). Unified on
    // the paged-pool view — the engine's past/present_key_values_i binding for DFlash's own draft
    // cache is the same [2, numPages, kTOKENS_PER_PAGE, numKVHeads, headDim] contract as the main
    // model and EAGLE/MTP drafts.
    auto& kvMgr = mDraftCacheManager.getKVCacheManager();
    int32_t localAttnIdx = 0;
    for (int32_t absIdx = 0; absIdx < static_cast<int32_t>(draftCfg.layerTypes.size()); ++absIdx)
    {
        if (draftCfg.layerTypes[absIdx] != HybridCacheManager::LayerType::kAttention)
        {
            continue;
        }
        auto& combinedKV = kvMgr.getCombinedKVCache(localAttnIdx);
        mDraftTensorMap.set(binding_names::formatKVCacheName(localAttnIdx, /*isPast=*/true), combinedKV);
        mDraftTensorMap.set(binding_names::formatKVCacheName(localAttnIdx, /*isPast=*/false), combinedKV);
        ++localAttnIdx;
    }
    mDraftExternalWeightManager = std::move(draftWeights);
    mDraftExternalWeightManager.registerTensorMapEntries(mDraftTensorMap);

    if (!isLiLiCorr())
    {
        std::string const selectorFile = draftCfg.dflash2SelectorFile.empty()
            ? std::string(binding_names::kDFlash2SelectorFileName)
            : draftCfg.dflash2SelectorFile;
        auto const selectorPath = engineDir / selectorFile;
        ELLM_CHECK(std::filesystem::exists(selectorPath), "DFlash2 selector sidecar missing: " + selectorPath.string());
        std::vector<Tensor> selectorTensors;
        ELLM_CHECK(safetensors::loadSafetensors(selectorPath, selectorTensors, stream),
            "Failed to load DFlash2 selector sidecar from " + selectorPath.string());
        mSelectorPredecessorCodebook = takeRequiredSelectorTensor(selectorTensors, "predecessor_codebook");
        mSelectorSuccessorCodebook = takeRequiredSelectorTensor(selectorTensors, "successor_codebook");
        check::check(selectorTensors.empty(), "DFlash2 selector sidecar contains unexpected tensors");
        validateSelectorCodebook(
            mSelectorPredecessorCodebook, "predecessor_codebook", draftCfg.outputVocabSize, selectorRank);
        validateSelectorCodebook(
            mSelectorSuccessorCodebook, "successor_codebook", draftCfg.outputVocabSize, selectorRank);
    }

    mHostDraftInputIds = Tensor({maxBatch, mBlockDraft.blockSize}, DeviceType::kCPU, trt_edgellm::DataType::kINT32,
        "DFlash2Draft::hostInputIds");
    mHostLastAcceptedTokens
        = Tensor({maxBatch}, DeviceType::kCPU, trt_edgellm::DataType::kINT32, "DFlash2Draft::hostLastAcceptedTokens");
    mHostDeltaLens = Tensor({maxBatch}, DeviceType::kCPU, trt_edgellm::DataType::kINT32, "DFlash2Draft::hostDeltaLens");

    // A jump-forward grafting round verifies jumpVerifyTokenCount rows and accepts up to jumpMaxPathLength tokens.
    auto const& specConfig = *deployment.specConfig;
    mVerifyTokenIds = Tensor({maxBatch, specConfig.jumpVerifyTokenCount}, DeviceType::kGPU,
        trt_edgellm::DataType::kINT32, "DFlash2::verifyTokenIds");
    mTreeTokenIds = Tensor({maxBatch, specConfig.jumpVerifyTokenCount}, DeviceType::kGPU, trt_edgellm::DataType::kINT32,
        "DFlash2::treeTokenIds");
    mTreeNodeScores = Tensor({maxBatch, mBlockDraft.verifyTokenCount}, DeviceType::kGPU, trt_edgellm::DataType::kFLOAT,
        "DFlash2::treeNodeScores");
    mValidCounts = Tensor({maxBatch}, DeviceType::kGPU, trt_edgellm::DataType::kINT32, "DFlash2::validCounts");
    Coords const tableShape{maxBatch, proposalLen, supportK, mBlockDraft.candidateTopK};
    mTreeCandidateTokenIds
        = Tensor(tableShape, DeviceType::kGPU, trt_edgellm::DataType::kINT32, "DFlash2::treeCandidateTokenIds");
    mTreeCandidateLogProbs
        = Tensor(tableShape, DeviceType::kGPU, trt_edgellm::DataType::kFLOAT, "DFlash2::treeCandidateLogProbs");
    mTreeCandidateChildStates
        = Tensor(tableShape, DeviceType::kGPU, trt_edgellm::DataType::kINT32, "DFlash2::treeCandidateChildStates");
    // One accepted root-to-leaf path holds at most blockSize nodes.
    int32_t const maxAcceptBufferSize
        = std::max(std::min(mBlockDraft.blockSize, mBlockDraft.verifyTokenCount), specConfig.jumpMaxPathLength);
    mAcceptedTokenIds = Tensor(
        {maxBatch, maxAcceptBufferSize}, DeviceType::kGPU, trt_edgellm::DataType::kINT32, "DFlash2::acceptedTokenIds");
    mAcceptedTokenIndices = Tensor({maxBatch, maxAcceptBufferSize}, DeviceType::kGPU, trt_edgellm::DataType::kINT32,
        "DFlash2::acceptedTokenIndices");
    mAcceptLength = Tensor({maxBatch}, DeviceType::kGPU, trt_edgellm::DataType::kINT32, "DFlash2::acceptLength");
    mCommonStateTracker.reserve(maxBatch);
    mHostAcceptLengths
        = Tensor({maxBatch}, DeviceType::kCPU, trt_edgellm::DataType::kINT32, "DFlash2::hostAcceptLengths");
    mHostSeatStaging
        = Tensor({maxBatch, 2}, DeviceType::kCPU, trt_edgellm::DataType::kINT32, "DFlash2::hostSeatStaging");
    mHostAcceptedTokenIds = Tensor(
        {maxBatch, maxAcceptBufferSize}, DeviceType::kCPU, trt_edgellm::DataType::kINT32, "DFlash2::hostAcceptedIds");
    mAcceptedHiddenStates = Tensor({maxBatch * maxAcceptBufferSize, mBlockDraft.baseOutputHiddenDim}, DeviceType::kGPU,
        trt_edgellm::DataType::kHALF, "DFlash2::acceptedHidden");
    mRemainingGenerationLengths
        = Tensor({maxBatch}, DeviceType::kGPU, trt_edgellm::DataType::kINT32, "DFlash2::remainingGenerationLengths");
    mHostRemainingGenerationLengths = Tensor(
        {maxBatch}, DeviceType::kCPU, trt_edgellm::DataType::kINT32, "DFlash2::hostRemainingGenerationLengths");

    // Lossless chain acceptance only: a DDTree verifies greedily, and the dense target probabilities alone would
    // take B * N * vocab * 4 bytes.
    if (!tree)
    {
        constexpr int32_t kMaxTargetTopK{128};
        mTargetTopKValues = Tensor({maxBatch * mBlockDraft.verifyTokenCount, kMaxTargetTopK}, DeviceType::kGPU,
            trt_edgellm::DataType::kFLOAT, "DFlash2::targetTopKValues");
        mTargetTopKIds = Tensor({maxBatch * mBlockDraft.verifyTokenCount, kMaxTargetTopK}, DeviceType::kGPU,
            trt_edgellm::DataType::kINT32, "DFlash2::targetTopKIds");
        mTargetTopKProbs = Tensor({maxBatch * mBlockDraft.verifyTokenCount, kMaxTargetTopK}, DeviceType::kGPU,
            trt_edgellm::DataType::kFLOAT, "DFlash2::targetTopKProbs");
        mTargetProbabilities
            = Tensor({maxBatch * mBlockDraft.verifyTokenCount, mRuntime.deployment.base.outputVocabSize},
                DeviceType::kGPU, trt_edgellm::DataType::kFLOAT, "DFlash2::targetProbabilities");
    }
    mProposalLengths = Tensor({maxBatch}, DeviceType::kGPU, trt_edgellm::DataType::kINT32, "DFlash2::proposalLengths");
    mAcceptUniforms = Tensor(
        {maxBatch, 2 * proposalLen + 1}, DeviceType::kGPU, trt_edgellm::DataType::kFLOAT, "DFlash2::acceptUniforms");
    mRequestSeeds = Tensor({maxBatch}, DeviceType::kGPU, trt_edgellm::DataType::kINT64, "DFlash2::requestSeeds");
    mNextAbsolutePositions
        = Tensor({maxBatch}, DeviceType::kGPU, trt_edgellm::DataType::kINT64, "DFlash2::nextAbsolutePositions");
    mHostRequestSeeds
        = Tensor({maxBatch}, DeviceType::kCPU, trt_edgellm::DataType::kINT64, "DFlash2::hostRequestSeeds");
    mHostNextPositions
        = Tensor({maxBatch}, DeviceType::kCPU, trt_edgellm::DataType::kINT64, "DFlash2::hostNextPositions");
    mHostTemperatures
        = Tensor({maxBatch}, DeviceType::kCPU, trt_edgellm::DataType::kFLOAT, "DFlash2::hostTemperatures");
    mHostGreedyMask = Tensor({maxBatch}, DeviceType::kCPU, trt_edgellm::DataType::kINT32, "DFlash2::hostGreedyMask");

    LOG_INFO(
        "DFlashDecoder initialized: user_mode=%s, proposal_attention=%s, tree_policy=%s, blockSize=%d, "
        "proposalLen=%d, verifyTokenCount=%d, runtimeDraftTopK=%d, %s=%d, maskTokenId=%d, maxBatch=%d, "
        "draftHiddenSize=%d, baseOutputHiddenDim=%d, draftVocabSize=%d",
        userModeName(), dflash_utils::proposalAttentionPolicyName(mBlockDraft.proposalAttention),
        dflash_utils::blockDraftTreePolicyName(mBlockDraft.treePolicy), mBlockDraft.blockSize, mBlockDraft.proposalLen,
        mBlockDraft.verifyTokenCount, mBlockDraft.candidateTopK,
        isLiLiCorr() ? "lilicorrCandidateTopK" : "selectorTopK", supportK, mBlockDraft.maskTokenId, maxBatch,
        mBlockDraft.draftHiddenSize, mBlockDraft.baseOutputHiddenDim, mBlockDraft.draftVocabSize);
}

void DFlashDecoder::reshapeDFlash2DraftOutputs(int32_t batchSize)
{
    int32_t const proposalLen = mBlockDraft.proposalLen;
    int32_t const supportK = proposalSupportK();
    check::check(mProposalSupportIds.reshape({batchSize, proposalLen, supportK}), "Tensor reshape failed");
    if (isLiLiCorr())
    {
        check::check(
            mLiLiCorrTransitionScores.reshape({batchSize, proposalLen, supportK, supportK}), "Tensor reshape failed");
    }
    else
    {
        check::check(mProposalUnaryValues.reshape({batchSize, proposalLen, supportK}), "Tensor reshape failed");
        check::check(
            mProposalProjectedHidden.reshape({batchSize, proposalLen, mRuntime.deployment.draft->specSelectorRank}),
            "Tensor reshape failed");
    }
}

bool DFlashDecoder::prepareV2Proposal(DecodingInferenceContext& context)
{
    NVTX_SCOPED_RANGE(
        nvtx_dflash2_prepare_verify, "DFlashDecoder::prepareBlockDraftVerifyInputs", nvtx_colors::LIGHT_ORANGE);

    int32_t const activeBatchSize = context.activeBatchSize;
    int32_t const verifyTokenCount = mBlockDraft.verifyTokenCount;
    check::check(mVerifyTokenIds.reshape({activeBatchSize, verifyTokenCount}), "Tensor reshape failed");
    check::check(mDraftTokenIds.reshape({activeBatchSize, mBlockDraft.proposalLen}), "Tensor reshape failed");
    check::check(mProposalLengths.reshape({activeBatchSize}), "Tensor reshape failed");
    kernel::dsparkFillProposalLengths(mProposalLengths, activeBatchSize, mBlockDraft.proposalLen, context.stream);
    check::check(mAcceptLength.reshape({activeBatchSize}), "Tensor reshape failed");
    // Prior-round counts are dead here; acceptance overwrites this scratch before commit.
    kernel::dsparkBuildVerifyTokens(mLastAcceptedTokens, mDraftTokenIds, mProposalLengths, mVerifyTokenIds,
        mAcceptLength, activeBatchSize, mBlockDraft.proposalLen, mBlockDraft.proposalLen, context.stream);

    finishVerifyInput(context);
    if (!checkCudaLastError("prepare DFlash verify inputs"))
    {
        return false;
    }
    return true;
}

void DFlashDecoder::selectDFlash2TreeCandidates(int32_t activeBatchSize, cudaStream_t stream)
{
    int32_t const proposalLen = mBlockDraft.proposalLen;
    int32_t const selectorTopK = mRuntime.deployment.draft->specSelectorTopK;
    int32_t const selectorRank = mRuntime.deployment.draft->specSelectorRank;
    Coords const tableShape{activeBatchSize, proposalLen, selectorTopK, mBlockDraft.candidateTopK};
    check::check(mProposalSupportIds.reshape({activeBatchSize, proposalLen, selectorTopK})
            && mProposalUnaryValues.reshape({activeBatchSize, proposalLen, selectorTopK})
            && mProposalProjectedHidden.reshape({activeBatchSize, proposalLen, selectorRank})
            && mLastAcceptedTokens.reshape({activeBatchSize}) && mTreeCandidateTokenIds.reshape(tableShape)
            && mTreeCandidateLogProbs.reshape(tableShape) && mTreeCandidateChildStates.reshape(tableShape),
        "Tensor reshape failed");
    kernel::dflash2SelectorTreeCandidates(mProposalSupportIds, mProposalUnaryValues, mProposalProjectedHidden,
        mLastAcceptedTokens, mSelectorPredecessorCodebook, mSelectorSuccessorCodebook, mTreeCandidateTokenIds,
        mTreeCandidateLogProbs, mTreeCandidateChildStates, stream);
}

bool DFlashDecoder::runV2Acceptance(
    DecodingInferenceContext& context, int32_t verifyTokenCount, int32_t maxAcceptLength)
{
    int32_t const activeBatchSize = context.activeBatchSize;
    RowSamplingParams const shared = context.uniformSamplingParams();
    bool const targetNonGreedy = ::trt_edgellm::shouldUseNonGreedySampling(shared);
    int32_t const targetTopK = targetNonGreedy ? static_cast<int32_t>(shared.topK) : 1;
    int32_t const targetRows = activeBatchSize * verifyTokenCount;
    bool const useDenseTarget = targetNonGreedy && targetTopK == 0;
    if (useDenseTarget)
    {
        int32_t const vocabSize = mRuntime.deployment.base.outputVocabSize;
        check::check(mRuntime.base.pipelineIO.outputLogits.reshape({targetRows, vocabSize}), "Tensor reshape failed");
        check::check(mTargetProbabilities.reshape({targetRows, vocabSize}), "Tensor reshape failed");
        topPProbabilitiesFromLogits(mRuntime.base.pipelineIO.outputLogits, mTargetProbabilities, shared.temperature,
            shared.topP, mRuntime.sampling.workspace, context.stream);
        check::check(
            mTargetProbabilities.reshape({activeBatchSize, verifyTokenCount, vocabSize}), "Tensor reshape failed");
    }
    else
    {
        ELLM_CHECK(targetTopK >= 1 && targetTopK <= dflash_utils::kDFlash2MaxSamplingSupport,
            "DFlash2 optimized sparse verification requires bounded sampling support.");
        check::check(mTargetTopKValues.reshape({targetRows, targetTopK}), "Tensor reshape failed");
        check::check(mTargetTopKIds.reshape({targetRows, targetTopK}), "Tensor reshape failed");
        check::check(mTargetTopKProbs.reshape({targetRows, targetTopK}), "Tensor reshape failed");
        selectAllTopK(mRuntime.base.pipelineIO.outputLogits, std::ref(mTargetTopKValues), mTargetTopKIds, targetTopK,
            mRuntime.sampling.workspace, context.stream);
        kernel::speculativeNormalizeTopKTopP(mTargetTopKValues, mTargetTopKProbs,
            targetNonGreedy ? shared.temperature : 1.0F, targetNonGreedy ? shared.topP : 1.0F, context.stream);
        check::check(
            mTargetTopKProbs.reshape({activeBatchSize, verifyTokenCount, targetTopK}), "Tensor reshape failed");
        check::check(mTargetTopKIds.reshape({activeBatchSize, verifyTokenCount, targetTopK}), "Tensor reshape failed");
    }
    check::check(mDraftTokenIds.reshape({activeBatchSize, mBlockDraft.proposalLen}), "Tensor reshape failed");
    check::check(mProposalSupportIds.reshape({activeBatchSize, mBlockDraft.proposalLen, proposalSupportK()}),
        "Tensor reshape failed");
    check::check(mProposalSupportProbs.reshape({activeBatchSize, mBlockDraft.proposalLen, proposalSupportK()}),
        "Tensor reshape failed");
    check::check(mRemainingGenerationLengths.reshape({activeBatchSize}), "Tensor reshape failed");
    check::check(mHostRemainingGenerationLengths.reshape({activeBatchSize}), "Tensor reshape failed");
    int32_t* hostRemainingGenerationLengths = mHostRemainingGenerationLengths.dataPointer<int32_t>();
    for (int32_t batchIdx = 0; batchIdx < activeBatchSize; ++batchIdx)
    {
        hostRemainingGenerationLengths[batchIdx]
            = std::max(0, context.generateBudget(batchIdx) - context.currentGenerateLengths[batchIdx]);
    }
    CUDA_CHECK(cudaMemcpyAsync(mRemainingGenerationLengths.rawPointer(), hostRemainingGenerationLengths,
        activeBatchSize * sizeof(int32_t), cudaMemcpyHostToDevice, context.stream));

    if (useDenseTarget)
    {
        spec_decode::accept(
            DenseTargetSparseDraftAcceptInputs{mTargetProbabilities, mProposalSupportProbs, mProposalSupportIds,
                mDraftTokenIds, mProposalLengths, mAcceptUniforms, &mRemainingGenerationLengths},
            mAcceptedTokenIds, mAcceptLength, &mAcceptedTokenIndices, context.stream,
            decoder_utils::forcedAcceptanceLength(mRuntime, context));
    }
    else
    {
        spec_decode::accept(
            SparseProbabilityAcceptInputs{mTargetTopKProbs, mTargetTopKIds, mProposalSupportProbs, mProposalSupportIds,
                mDraftTokenIds, mProposalLengths, mAcceptUniforms, &mRemainingGenerationLengths},
            mAcceptedTokenIds, mAcceptLength, &mAcceptedTokenIndices, context.stream,
            decoder_utils::forcedAcceptanceLength(mRuntime, context));
    }
    if (!checkCudaLastError("DFlash2 accept"))
    {
        return false;
    }
    spec_decode::commitAcceptedTargetPath(mRuntime, context,
        SpecAcceptResult{mAcceptedTokenIds, mAcceptLength, &mAcceptedTokenIndices, maxAcceptLength}, verifyTokenCount);

    return true;
}

} // namespace rt
} // namespace trt_edgellm
