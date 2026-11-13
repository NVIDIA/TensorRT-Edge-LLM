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

#include "runtime/decoding/dsparkDecoder.h"
#include "common/bindingNames.h"
#include "common/checkMacros.h"
#include "common/cudaUtils.h"
#include "common/logger.h"
#include "common/mathUtils.h"
#include "common/safetensorsUtils.h"
#include "kernels/contextAttentionKernels/utilKernels.h"
#include "kernels/embeddingKernels/embeddingKernels.h"
#include "kernels/gdnKernels/gdnTreeChunkKernels.h"
#include "kernels/nvfp4A16BlackwellGemm/nvfp4A16BlackwellGemmRunner.h"
#include "kernels/speculative/ddtreeKernels.h"
#include "kernels/speculative/dflashRuntimeKernels.h"
#include "kernels/speculative/dsparkKernels.h"
#include "kernels/speculative/speculativeKernels.h"
#include "profiling/metrics.h"
#include "profiling/nvtx_wrapper.h"
#include "profiling/timer.h"
#include "runtime/config/llmEngineConfig.h"
#include "runtime/decoding/decoderUtils.h"
#include "runtime/decoding/guidedDecoder.h"
#include "runtime/decoding/logitBias.h"
#include "runtime/decoding/specDecodeUtils.h"
#include "sampler/sampling.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cuda_fp16.h>
#include <filesystem>
#include <fstream>
#include <functional>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace trt_edgellm
{
namespace rt
{
namespace
{
constexpr int32_t kPrefillProfile{0};
constexpr int32_t kDecodeProfile{1};
constexpr int32_t kDSparkMaxSparseTopK{128};
constexpr uint64_t kDSparkSamplingSeed{0x44535041524B2026ULL};

bool dsparkCanUseTopKFastPath(int64_t topK, float topP) noexcept
{
    return topK > 1 && topK <= kDSparkMaxSparseTopK && topP >= 1.0F - 1e-6F;
}

uint64_t dsparkRandomRoundOffset(int32_t generationRound, int32_t maxBatchSize, int32_t proposalLen)
{
    uint64_t const uniformsPerRound = static_cast<uint64_t>(maxBatchSize) * static_cast<uint64_t>(3 * proposalLen + 1);
    return static_cast<uint64_t>(generationRound) * uniformsPerRound;
}

//! A delta covering prompt positions (the first forward of a request, or a seated row's) runs on the prefill
//! profile; an accepted span runs on the decode one.
int32_t dsparkDraftProfile(bool prefillDelta)
{
    return prefillDelta ? kPrefillProfile : kDecodeProfile;
}

Tensor takeRequiredTensor(std::vector<Tensor>& tensors, std::string const& name)
{
    auto it
        = std::find_if(tensors.begin(), tensors.end(), [&](Tensor const& tensor) { return tensor.getName() == name; });
    check::check(it != tensors.end(), "DSpark heads sidecar is missing tensor '" + name + "'");
    Tensor out = std::move(*it);
    tensors.erase(it);
    return out;
}

void validateMatrix(Tensor const& tensor, std::string const& name, int64_t rows, int64_t cols)
{
    check::check(tensor.getDataType() == trt_edgellm::DataType::kHALF, name + " must be FP16");
    check::check(tensor.getShape().getNumDims() == 2, name + " must be rank-2");
    check::check(tensor.getShape()[0] == rows,
        name + " row count mismatch: expected " + std::to_string(rows) + ", got "
            + std::to_string(tensor.getShape()[0]));
    check::check(tensor.getShape()[1] == cols,
        name + " column count mismatch: expected " + std::to_string(cols) + ", got "
            + std::to_string(tensor.getShape()[1]));
}

} // namespace

DSparkDecoder::DSparkDecoder(DecodingRuntimeContext& runtime, std::filesystem::path const& engineDir,
    std::unique_ptr<EngineExecutor> draftExecutor, ExternalWeightManager draftWeights, cudaStream_t stream)
    : mRuntime(runtime)
    , mDraftCacheManager(*runtime.base.sharedResources.cacheManagers[1])
    , mDraftExecutor(std::move(draftExecutor))
{
    auto const& deployment = runtime.deployment;
    auto const& baseCfg = deployment.base;
    ELLM_CHECK(deployment.specConfig.has_value(), "DSparkDecoder: specConfig is required.");
    ELLM_CHECK(deployment.draft.has_value(), "DSparkDecoder: draft config is required.");
    ELLM_CHECK(baseCfg.specDecodeType == SpecDecodeMode::kDSpark,
        "DSparkDecoder requires a base engine exported with spec_decode_type=dspark and engine_role=base.");
    ELLM_CHECK(baseCfg.reducedVocabSize == 0, "DSpark Phase 1 does not support reduced-vocabulary base engines.");

    auto const& draftCfg = *deployment.draft;
    std::string markovType = draftCfg.dsparkMarkovHeadType.empty() ? "vanilla" : draftCfg.dsparkMarkovHeadType;
    ELLM_CHECK(markovType == "vanilla",
        "DSparkDecoder Phase 1 supports markov_head_type=vanilla only; got '" + markovType + "'.");
    ELLM_CHECK(draftCfg.dsparkMarkovRank > 0, "DSparkDecoder requires dspark_config.markov_rank > 0.");

    mCausalProposalMask = draftCfg.specDraftCausalHead;
    mUseTree = mRuntime.deployment.specConfig->draftTopK > 1;
    mVerifyLen = deployment.specConfig->verifyTokenCount;
    mProposalLen = mUseTree ? draftCfg.specDraftBlockSize : mVerifyLen - 1;
    mCurrentProposalLen = mProposalLen;
    mDraftSlotOffset = draftCfg.dsparkSampleFromAnchor ? 0 : 1;
    mDraftBlockLen = mProposalLen + mDraftSlotOffset;
    mCurrentVerifyLen = mVerifyLen;
    ELLM_CHECK(mProposalLen > 0, "DSparkDecoder requires verifyTokenCount >= 2.");
    ELLM_CHECK(
        mProposalLen <= draftCfg.specDraftBlockSize, "DSparkDecoder proposal length exceeds dspark_config.block_size.");
    mMaskTokenId = draftCfg.specDraftMaskTokenId > 0 ? draftCfg.specDraftMaskTokenId : baseCfg.specDraftMaskTokenId;
    mDraftHiddenSize = deployment.specConfig->draftHiddenSize;
    mBaseOutputHiddenDim = deployment.specConfig->baseOutputHiddenDim;
    mDraftVocabSize = draftCfg.outputVocabSize;
    mTargetVocabSize = baseCfg.outputVocabSize;
    ELLM_CHECK(mMaskTokenId >= 0 && mMaskTokenId < mTargetVocabSize,
        "DSparkDecoder: mask token id (" + std::to_string(mMaskTokenId) + ") out of target vocab range [0, "
            + std::to_string(mTargetVocabSize) + ").");
    if (draftCfg.reducedVocabSize == 0)
    {
        ELLM_CHECK(mTargetVocabSize == mDraftVocabSize,
            "DSparkDecoder requires matching base and draft vocab sizes when no draft vocab map is configured.");
    }
    mMarkovRank = draftCfg.dsparkMarkovRank;
    mHasConfidenceHead = draftCfg.dsparkEnableConfidenceHead;
    mConfidenceHeadWithMarkov = draftCfg.dsparkConfidenceHeadWithMarkov;
    mSchedulerMode = mRuntime.deployment.specConfig->dsparkSchedulerMode;
    mConfidenceThreshold = mRuntime.deployment.specConfig->dsparkConfidenceThreshold;
    mMinScheduledProposalLen = std::max(1, mRuntime.deployment.specConfig->dsparkMinProposalLen);
    mMaxScheduledProposalLen = mRuntime.deployment.specConfig->dsparkMaxProposalLen <= 0
        ? mProposalLen
        : std::min(mProposalLen, mRuntime.deployment.specConfig->dsparkMaxProposalLen);
    ELLM_CHECK(mMinScheduledProposalLen <= mMaxScheduledProposalLen,
        "DSpark scheduler min proposal length exceeds max proposal length.");
    ELLM_CHECK(mSchedulerMode == DSparkSchedulerMode::kOff || mHasConfidenceHead,
        "DSpark scheduler requires confidence head sidecar tensors.");
    mUseTreeScheduler = mUseTree && mSchedulerMode != DSparkSchedulerMode::kOff;

    int32_t const maxBatch = deployment.maxRuntimeBatchSize();
    int32_t const maxSeqForDraft = baseCfg.maxKVCacheCapacity;

    ELLM_CHECK(mDraftExecutor != nullptr, "DSpark decoding requires a validated draft engine.");

    mDraftInputsEmbeds = Tensor({maxBatch, mDraftBlockLen, mDraftHiddenSize}, DeviceType::kGPU,
        trt_edgellm::DataType::kHALF, "DSpark::draftInputsEmbeds");
    mDraftTargetHidden = Tensor({maxBatch, maxSeqForDraft, mBaseOutputHiddenDim}, DeviceType::kGPU,
        trt_edgellm::DataType::kHALF, "DSpark::draftTargetHidden");
    mDraftOutputLogits = Tensor({maxBatch, mDraftBlockLen, mDraftVocabSize}, DeviceType::kGPU,
        trt_edgellm::DataType::kFLOAT, "DSpark::draftOutputLogits");
    mDraftHiddenStates = Tensor({maxBatch, mDraftBlockLen, mDraftHiddenSize}, DeviceType::kGPU,
        trt_edgellm::DataType::kHALF, "DSpark::draftHiddenStates");

    int32_t const packedMaskLen = divUp(mDraftBlockLen, 32);
    mDraftPackedAttentionMask = Tensor({maxBatch, mDraftBlockLen, packedMaskLen}, DeviceType::kGPU,
        trt_edgellm::DataType::kINT32, "DSpark::draftPackedMask");
    mDraftAttentionPosId = Tensor(
        {maxBatch, mDraftBlockLen}, DeviceType::kGPU, trt_edgellm::DataType::kINT32, "DSpark::draftAttentionPosId");
    mDraftContextLengths
        = Tensor({maxBatch}, DeviceType::kGPU, trt_edgellm::DataType::kINT32, "DSpark::draftContextLengths");
    mDraftDeltaLenCommit
        = Tensor({maxBatch}, DeviceType::kGPU, trt_edgellm::DataType::kINT32, "DSpark::draftDeltaLenCommit");
    mDraftDeltaLens = Tensor({maxBatch}, DeviceType::kGPU, trt_edgellm::DataType::kINT32, "DSpark::draftDeltaLens");
    int64_t const maxDeltaTokens = static_cast<int64_t>(maxBatch) * maxSeqForDraft;
    mDraftDeltaRopeCosSin = Tensor({maxDeltaTokens, draftCfg.rotaryDim}, DeviceType::kGPU,
        trt_edgellm::DataType::kFLOAT, "DSpark::draftDeltaRopeCosSin");
    mDraftDeltaPositions
        = Tensor({maxDeltaTokens}, DeviceType::kGPU, trt_edgellm::DataType::kINT32, "DSpark::draftDeltaPositions");
    mDraftDeltaTokenToSequence = Tensor(
        {maxDeltaTokens}, DeviceType::kGPU, trt_edgellm::DataType::kINT32, "DSpark::draftDeltaTokenToSequence");

    mDraftTensorMap.set(binding_names::kInputsEmbeds, mDraftInputsEmbeds);
    mDraftTensorMap.set(binding_names::kDFlashTargetHiddenConcat, mDraftTargetHidden);
    mDraftTensorMap.set(binding_names::kLogits, mDraftOutputLogits);
    mDraftTensorMap.set(binding_names::kDSparkHiddenStates, mDraftHiddenStates);
    mDraftTensorMap.set(binding_names::kAttentionMask, mDraftPackedAttentionMask);
    mDraftTensorMap.set(binding_names::kAttentionPosId, mDraftAttentionPosId);
    mDraftTensorMap.set(binding_names::kDFlashDeltaRopeCosSin, mDraftDeltaRopeCosSin);
    mDraftTensorMap.set(binding_names::kDFlashDeltaPositions, mDraftDeltaPositions);
    mDraftTensorMap.set(binding_names::kDFlashDeltaTokenToSequence, mDraftDeltaTokenToSequence);
    auto& io = mRuntime.base.pipelineIO;
    mDraftTensorMap.set(binding_names::kPositions, io.positions);
    mDraftTensorMap.set(binding_names::kQueryStartOffsets, io.queryStartOffsets);
    mDraftTensorMap.set(binding_names::kQueryLengths, io.queryLengths);
    mDraftTensorMap.set(binding_names::kPastLengths, io.pastLengths);
    mDraftTensorMap.set(binding_names::kAttentionSequenceLengths, io.attentionSequenceLengths);
    mDraftTensorMap.set(binding_names::kStateIndices, io.stateIndices);
    mDraftTensorMap.set(binding_names::kExecutionPhaseMarker, io.executionPhaseMarker);
    mDraftTensorMap.set(binding_names::kContextSequenceCountCarrier, io.contextSequenceCountCarrier);

    // KV cache bindings: bind to draft cache manager's combined KV cache (index 1). DSpark
    // uses the same cached-draft path as DFlash, so the engine expects the paged-pool view
    // [2, numPages, kTOKENS_PER_PAGE, numKVHeads, headDim], not the legacy slot-shaped alias.
    {
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
    }

    mDraftTensorMap.set(binding_names::kKVPageTable, io.raggedKVPageTable);
    mDraftTensorMap.set(binding_names::kRopeCosSin, io.raggedRopeCosSin);

    mDraftExternalWeightManager = std::move(draftWeights);
    mDraftExternalWeightManager.registerTensorMapEntries(mDraftTensorMap);

    // A jump-forward grafting round verifies jumpVerifyTokenCount rows and accepts up to jumpMaxPathLength tokens.
    auto const& specConfig = *deployment.specConfig;
    int32_t const acceptCapacity = std::max(mVerifyLen, specConfig.jumpMaxPathLength);
    mAcceptCapacity = acceptCapacity;
    mDraftTokenIds
        = Tensor({maxBatch, mProposalLen}, DeviceType::kGPU, trt_edgellm::DataType::kINT32, "DSpark::draftTokenIds");
    mVerifyTokenIds
        = Tensor({maxBatch, mVerifyLen}, DeviceType::kGPU, trt_edgellm::DataType::kINT32, "DSpark::verifyTokenIds");
    mAcceptedTokenIds
        = Tensor({maxBatch, acceptCapacity}, DeviceType::kGPU, trt_edgellm::DataType::kINT32, "DSpark::acceptedIds");
    mAcceptLength = Tensor({maxBatch}, DeviceType::kGPU, trt_edgellm::DataType::kINT32, "DSpark::acceptLength");
    mCommonStateTracker.reserve(maxBatch);
    mAcceptedHiddenStates = Tensor({static_cast<int64_t>(maxBatch) * acceptCapacity, mBaseOutputHiddenDim},
        DeviceType::kGPU, trt_edgellm::DataType::kHALF, "DSpark::acceptedHidden");
    mHostAcceptLengths
        = Tensor({maxBatch}, DeviceType::kCPU, trt_edgellm::DataType::kINT32, "DSpark::hostAcceptLengths");
    mHostSeatStaging
        = Tensor({maxBatch, 2}, DeviceType::kCPU, trt_edgellm::DataType::kINT32, "DSpark::hostSeatStaging");
    mHostAcceptedTokenIds = Tensor(
        {maxBatch, acceptCapacity}, DeviceType::kCPU, trt_edgellm::DataType::kINT32, "DSpark::hostAcceptedIds");
    mHostDraftInputIds = Tensor(
        {maxBatch, mDraftBlockLen}, DeviceType::kCPU, trt_edgellm::DataType::kINT32, "DSpark::hostDraftInputIds");
    mHostLastAcceptedTokens
        = Tensor({maxBatch}, DeviceType::kCPU, trt_edgellm::DataType::kINT32, "DSpark::hostLastAcceptedTokens");
    mHostDeltaLens = Tensor({maxBatch}, DeviceType::kCPU, trt_edgellm::DataType::kINT32, "DSpark::hostDeltaLens");
    mHostProposalLengths
        = Tensor({maxBatch}, DeviceType::kCPU, trt_edgellm::DataType::kINT32, "DSpark::hostProposalLengths");
    mDraftProbabilities = Tensor({maxBatch, mProposalLen, mDraftVocabSize}, DeviceType::kGPU,
        trt_edgellm::DataType::kFLOAT, "DSpark::draftProbabilities");
    mDraftStepLogits
        = Tensor({maxBatch, mDraftVocabSize}, DeviceType::kGPU, trt_edgellm::DataType::kFLOAT, "DSpark::stepLogits");
    mMarkovGreedySlots = Tensor(
        {maxBatch, mProposalLen}, DeviceType::kGPU, trt_edgellm::DataType::kINT64, "DSpark::markovGreedySlots");
    mDraftStepProbabilities = Tensor(
        {maxBatch, mDraftVocabSize}, DeviceType::kGPU, trt_edgellm::DataType::kFLOAT, "DSpark::stepProbabilities");
    mDraftStepTopKValues = Tensor(
        {maxBatch, kDSparkMaxSparseTopK}, DeviceType::kGPU, trt_edgellm::DataType::kFLOAT, "DSpark::stepTopKValues");
    mDraftStepTopKIndices = Tensor(
        {maxBatch, kDSparkMaxSparseTopK}, DeviceType::kGPU, trt_edgellm::DataType::kINT32, "DSpark::stepTopKIndices");
    mDraftTopKProbabilities = Tensor({maxBatch, mProposalLen, kDSparkMaxSparseTopK}, DeviceType::kGPU,
        trt_edgellm::DataType::kFLOAT, "DSpark::draftTopKProbabilities");
    mDraftTopKIndices = Tensor({maxBatch, mProposalLen, kDSparkMaxSparseTopK}, DeviceType::kGPU,
        trt_edgellm::DataType::kINT32, "DSpark::draftTopKIndices");
    mTargetProbabilities = Tensor({maxBatch, mVerifyLen, baseCfg.outputVocabSize}, DeviceType::kGPU,
        trt_edgellm::DataType::kFLOAT, "DSpark::targetProbabilities");
    mTargetTopKValues = Tensor({maxBatch * mVerifyLen, kDSparkMaxSparseTopK}, DeviceType::kGPU,
        trt_edgellm::DataType::kFLOAT, "DSpark::targetTopKValues");
    mTargetTopKIndices = Tensor({maxBatch * mVerifyLen, kDSparkMaxSparseTopK}, DeviceType::kGPU,
        trt_edgellm::DataType::kINT32, "DSpark::targetTopKIndices");
    mTargetTopKProbabilities = Tensor({maxBatch * mVerifyLen, kDSparkMaxSparseTopK}, DeviceType::kGPU,
        trt_edgellm::DataType::kFLOAT, "DSpark::targetTopKProbabilities");
    mDraftUniforms
        = Tensor({maxBatch, mProposalLen}, DeviceType::kGPU, trt_edgellm::DataType::kFLOAT, "DSpark::draftUniforms");
    mAcceptUniforms = Tensor(
        {maxBatch, 2 * mProposalLen + 1}, DeviceType::kGPU, trt_edgellm::DataType::kFLOAT, "DSpark::acceptUniforms");
    mConfidenceScores
        = Tensor({maxBatch, mProposalLen}, DeviceType::kGPU, trt_edgellm::DataType::kFLOAT, "DSpark::confidenceScores");
    mProposalLengths = Tensor({maxBatch}, DeviceType::kGPU, trt_edgellm::DataType::kINT32, "DSpark::proposalLengths");
    mAcceptValidTokenCounts
        = Tensor({maxBatch}, DeviceType::kGPU, trt_edgellm::DataType::kINT32, "DSpark::acceptValidTokenCounts");
    mArgmaxScratch
        = Tensor({maxBatch * mVerifyLen}, DeviceType::kGPU, trt_edgellm::DataType::kINT32, "DSpark::argmaxScratch");
    mLastAcceptedTokens
        = Tensor({maxBatch}, DeviceType::kGPU, trt_edgellm::DataType::kINT32, "DSpark::lastAcceptedTokens");

    if (mUseTree)
    {
        int32_t const proposalDepthSize = mProposalLen + 1;
        mStackedMarkovLogits = Tensor({maxBatch, proposalDepthSize, mDraftVocabSize}, DeviceType::kGPU,
            trt_edgellm::DataType::kFLOAT, "DSpark::stackedMarkovLogits");
        mTreeTokenIds = Tensor({maxBatch, specConfig.jumpVerifyTokenCount}, DeviceType::kGPU,
            trt_edgellm::DataType::kINT32, "DSpark::treeTokenIds");
        mTreeNodeScores
            = Tensor({maxBatch, mVerifyLen}, DeviceType::kGPU, trt_edgellm::DataType::kFLOAT, "DSpark::treeNodeScores");
        int32_t const candidateTopK = mRuntime.deployment.specConfig->draftTopK;
        mValidCounts = Tensor({maxBatch}, DeviceType::kGPU, trt_edgellm::DataType::kINT32, "DSpark::validCounts");
        mAcceptedTokenIndices = Tensor({maxBatch, acceptCapacity}, DeviceType::kGPU, trt_edgellm::DataType::kINT32,
            "DSpark::acceptedTokenIndices");
        size_t const buildWorkspaceSize = kernel::getDDTreeBuildWorkspaceSize(
            maxBatch, proposalDepthSize, mVerifyLen, mDraftVocabSize, candidateTopK);
        ELLM_CHECK(buildWorkspaceSize > 0, "DSparkDecoder: invalid tree-build workspace size.");
        mTreeBuildWorkspace = Tensor({static_cast<int64_t>(buildWorkspaceSize)}, DeviceType::kGPU,
            trt_edgellm::DataType::kUINT8, "DSpark::treeBuildWorkspace");
        CUDA_CHECK(
            cudaMemsetAsync(mStackedMarkovLogits.rawPointer(), 0, mStackedMarkovLogits.getMemoryCapacity(), stream));
        CUDA_CHECK(cudaMemsetAsync(mDraftOutputLogits.rawPointer(), 0, mDraftOutputLogits.getMemoryCapacity(), stream));
        CUDA_CHECK(
            cudaMemsetAsync(mLastAcceptedTokens.rawPointer(), 0, mLastAcceptedTokens.getMemoryCapacity(), stream));
    }

    if (specConfig.maxJumpTokens > 0)
    {
        // The next draft forward takes a grafting round's accepted path as its target-hidden delta.
        ELLM_CHECK(specConfig.jumpMaxPathLength <= mProposalLen + 1,
            "DSpark jump-forward accepted path exceeds the draft's target-hidden delta profile.");
        // A tree takes forced chains under any node; a chain takes one, which replaces the draft after its node.
        mJump.initialize(mRuntime.tokenizer, mRuntime.guidedDecoder, maxBatch,
            mUseTree ? JumpGraftGeometry::kTree : JumpGraftGeometry::kChain, specConfig, mVerifyLen);
        if (!mUseTree)
        {
            mAcceptedTokenIndices = Tensor({maxBatch, acceptCapacity}, DeviceType::kGPU, trt_edgellm::DataType::kINT32,
                "DSpark::acceptedTokenIndices");
        }
    }

    loadDraftVocabMap(engineDir, stream);
    loadHeadSidecars(engineDir, stream);

    LOG_INFO(
        "DSparkDecoder initialized: proposalLen=%d verifyLen=%d maskTokenId=%d maxBatch=%d "
        "draftHiddenSize=%d baseOutputHiddenDim=%d draftVocabSize=%d markovRank=%d confidence=%s",
        mProposalLen, mVerifyLen, mMaskTokenId, maxBatch, mDraftHiddenSize, mBaseOutputHiddenDim, mDraftVocabSize,
        mMarkovRank, mHasConfidenceHead ? "true" : "false");
}

DecodingKvHeadroomPolicy DSparkDecoder::kvHeadroom() const
{
    // The draft query block and the next target-hidden delta share the speculative headroom. A grafting round verifies
    // jumpVerifyTokenCount rows, and its accepted path is the next delta; a chain's equals the plain round's.
    auto const& specConfig = *mRuntime.deployment.specConfig;
    int32_t const maxAcceptedLength = std::min(mProposalLen + 1, mVerifyLen);
    return {/*.plain=*/{mVerifyLen, std::max(mDraftBlockLen, maxAcceptedLength)},
        /*.jumpForward=*/{specConfig.jumpVerifyTokenCount, std::max(mDraftBlockLen, specConfig.jumpMaxPathLength)}};
}

void DSparkDecoder::loadDraftVocabMap(std::filesystem::path const& engineDir, cudaStream_t stream)
{
    auto const& draftCfg = *mRuntime.deployment.draft;
    if (draftCfg.reducedVocabSize == 0)
    {
        return;
    }

    auto const mapPath = engineDir / binding_names::kDraftVocabMapFileName;
    ELLM_CHECK(std::filesystem::exists(mapPath),
        "DSpark draft declares reduced_vocab_size > 0 but " + std::string(binding_names::kDraftVocabMapFileName)
            + " is missing from the engine directory.");
    std::vector<Tensor> tensors;
    ELLM_CHECK(safetensors::loadSafetensors(mapPath, tensors, stream),
        "Failed to load DSpark draft vocabulary map from " + mapPath.string());
    check::check(tensors.size() == 1, "DSpark draft vocabulary map must contain exactly one tensor");
    check::check(
        tensors[0].getDataType() == trt_edgellm::DataType::kINT32, "DSpark draft vocabulary map must be INT32");
    check::check(tensors[0].getShape().getNumDims() == 1 && tensors[0].getShape()[0] == mDraftVocabSize,
        "DSpark draft vocabulary map length must match the draft output vocabulary");
    mDraftVocabMappingTable = std::move(tensors[0]);
    mHasDraftVocabMap = true;
}

void DSparkDecoder::loadHeadSidecars(std::filesystem::path const& engineDir, cudaStream_t stream)
{
    auto const& draftCfg = *mRuntime.deployment.draft;
    std::string const headsFile = draftCfg.dsparkHeadsFile.empty() ? std::string(binding_names::kDSparkHeadsFileName)
                                                                   : draftCfg.dsparkHeadsFile;
    auto const headsPath = engineDir / headsFile;
    ELLM_CHECK(std::filesystem::exists(headsPath), "DSpark heads sidecar missing: " + headsPath.string());

    std::vector<Tensor> tensors;
    ELLM_CHECK(safetensors::loadSafetensors(headsPath, tensors, stream),
        "Failed to load DSpark heads sidecar from " + headsPath.string());

    mMarkovW1 = takeRequiredTensor(tensors, "markov_w1");
    validateMatrix(mMarkovW1, "markov_w1", mTargetVocabSize, mMarkovRank);
    loadW2(engineDir, headsFile, tensors, stream);

    if (mHasConfidenceHead)
    {
        mConfidenceWeight = takeRequiredTensor(tensors, "confidence_weight");
        mConfidenceBias = takeRequiredTensor(tensors, "confidence_bias");
        check::check(mConfidenceWeight.getDataType() == trt_edgellm::DataType::kHALF, "confidence_weight must be FP16");
        check::check(mConfidenceBias.getDataType() == trt_edgellm::DataType::kHALF, "confidence_bias must be FP16");
        check::check(mConfidenceWeight.getShape().getNumDims() == 1, "confidence_weight must be rank-1");
        int64_t const expectedConfidenceDim = static_cast<int64_t>(mDraftHiddenSize)
            + (mConfidenceHeadWithMarkov ? static_cast<int64_t>(mMarkovRank) : 0);
        check::check(mConfidenceWeight.getShape()[0] == expectedConfidenceDim,
            "confidence_weight size mismatch: expected " + std::to_string(expectedConfidenceDim) + ", got "
                + std::to_string(mConfidenceWeight.getShape()[0]));
        check::check(mConfidenceBias.getShape().getNumDims() == 1 && mConfidenceBias.getShape()[0] == 1,
            "confidence_bias must be shape [1]");
    }
}

void DSparkDecoder::loadW2(std::filesystem::path const& engineDir, std::string const& headsFile,
    std::vector<Tensor>& tensors, cudaStream_t stream)
{
    bool const hasDenseW2 = std::any_of(
        tensors.begin(), tensors.end(), [](Tensor const& tensor) { return tensor.getName() == "markov_w2"; });
    if (hasDenseW2)
    {
        mMarkovW2 = takeRequiredTensor(tensors, "markov_w2");
        validateMatrix(mMarkovW2, "markov_w2", mDraftVocabSize, mMarkovRank);
    }
    auto const requested = mRuntime.deployment.specConfig->dsparkW2Backend;
    auto const& draftCfg = *mRuntime.deployment.draft;
    auto const infoPath = engineDir
        / (draftCfg.dsparkHeadsInfoFile.empty() ? std::string(binding_names::kDSparkHeadsInfoFileName)
                                                : draftCfg.dsparkHeadsInfoFile);
    std::string format;
    if (std::filesystem::exists(infoPath))
    {
        std::ifstream input(infoPath);
        ELLM_CHECK(input.good(), "Cannot read DSpark heads metadata: " + infoPath.string());
        format = nlohmann::json::parse(input).value("w2_format", std::string{});
        ELLM_CHECK(format.empty() || format == "fp16" || format == "nvfp4", "Invalid DSpark w2_format");
    }
    bool const packedExists = std::filesystem::exists(engineDir / binding_names::kDSparkW2Nvfp4FileName);
    ELLM_CHECK(format != "fp16" || hasDenseW2, "DSpark metadata declares FP16 but markov_w2 is missing");
    ELLM_CHECK(format != "nvfp4" || packedExists, "DSpark metadata declares NVFP4 but packed W2 is missing");
    bool const hasPackedW2 = format == "nvfp4"
        || (format.empty() && packedExists && (!hasDenseW2 || requested == DSparkW2Backend::kW4A16));
    ELLM_CHECK(hasDenseW2 || hasPackedW2, "DSpark W2 weights are missing");
    if (hasPackedW2)
    {
        ELLM_CHECK(
            headsFile == binding_names::kDSparkHeadsFileName, "NVFP4 W2 requires the default DSpark heads sidecar");
        ELLM_CHECK(mMarkovRank > 0 && mMarkovRank % 16 == 0, "NVFP4 W2 requires rank divisible by 16");
        std::vector<Tensor> packed;
        ELLM_CHECK(safetensors::loadSafetensors(engineDir / binding_names::kDSparkW2Nvfp4FileName, packed, stream),
            "Failed to load packed NVFP4 W2 sidecar");
        Tensor packedWeight = takeRequiredTensor(packed, "packed_weight");
        Tensor blockScales = takeRequiredTensor(packed, "block_scales");
        mMarkovW2GlobalScale = takeRequiredTensor(packed, "global_scale");
        auto validateBytes = [](Tensor const& tensor, int64_t rows, int64_t cols) {
            ELLM_CHECK(tensor.getDataType() == trt_edgellm::DataType::kUINT8 && tensor.getShape().getNumDims() == 2
                    && tensor.getShape()[0] == rows && tensor.getShape()[1] == cols,
                "Invalid NVFP4 W2 packed tensor");
        };
        validateBytes(packedWeight, mDraftVocabSize, mMarkovRank / 2);
        validateBytes(blockScales, mDraftVocabSize, mMarkovRank / 16);
        ELLM_CHECK(mMarkovW2GlobalScale.getDataType() == trt_edgellm::DataType::kFLOAT
                && mMarkovW2GlobalScale.getShape().getNumDims() == 1 && mMarkovW2GlobalScale.getShape()[0] == 1,
            "Invalid NVFP4 W2 global scale");
        Tensor hostGlobalScale({1}, DeviceType::kCPU, trt_edgellm::DataType::kFLOAT);
        CUDA_CHECK(cudaMemcpyAsync(hostGlobalScale.rawPointer(), mMarkovW2GlobalScale.rawPointer(), sizeof(float),
            cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        float const globalScale = *static_cast<float const*>(hostGlobalScale.rawPointer());
        ELLM_CHECK(
            std::isfinite(globalScale) && globalScale > 0.0F, "NVFP4 W2 global scale must be finite and positive");
        if (!hasDenseW2)
        {
            mMarkovW2 = Tensor({mDraftVocabSize, mMarkovRank}, DeviceType::kGPU, trt_edgellm::DataType::kHALF,
                "DSpark::markovW2Fallback");
            kernel::dsparkDequantizeW2(packedWeight, blockScales, mMarkovW2GlobalScale, mMarkovW2, stream);
        }
        using namespace trt_edgellm::kernels;
        int32_t const maxBatch = mRuntime.deployment.maxRuntimeBatchSize();
        float const kernelGlobalScale = __half2float(__float2half_rn(globalScale));
        bool supported = !mHasDraftVocabMap && getSMVersion() == 110 && mMarkovRank == 512 && mDraftVocabSize % 128 == 0
            && std::isfinite(kernelGlobalScale) && kernelGlobalScale > 0.0F;
        for (int32_t batch = 1; supported && batch <= maxBatch; ++batch)
        {
            supported = Nvfp4A16BlackwellGemmRunner::isSupported(
                110, Nvfp4A16BlackwellDtype::kFp16, batch, mDraftVocabSize, mMarkovRank);
        }
        mW2Backend = resolveDSparkW2Backend(requested, true, supported);
        if (mW2Backend == DSparkW2Backend::kW4A16)
        {
            prepareW4A16(packedWeight, blockScales, stream);
        }
        else
        {
            LOG_INFO("DSparkDecoder: native NVFP4 W2 uses dense fallback on this configuration.");
        }
        // Packed source tensors must outlive fallback materialization and layout conversion.
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    else
    {
        mW2Backend = resolveDSparkW2Backend(requested, false, false);
    }

    if (mW2Backend == DSparkW2Backend::kFp8)
    {
        int32_t const lanesPerToken = mMarkovRank / 16;
        ELLM_CHECK(mMarkovRank % 16 == 0 && kernel::dsparkFusedGreedySupported(mMarkovRank)
                && (lanesPerToken & (lanesPerToken - 1)) == 0,
            "FP8 W2 requires markov_rank == 16 * 2^k and markov_rank <= 512");
        mMarkovW2Fp8 = Tensor(
            {mDraftVocabSize, mMarkovRank}, DeviceType::kGPU, trt_edgellm::DataType::kUINT8, "DSpark::markovW2Fp8");
        mMarkovW2RowScales
            = Tensor({mDraftVocabSize}, DeviceType::kGPU, trt_edgellm::DataType::kHALF, "DSpark::markovW2RowScales");
        kernel::dsparkQuantizeMarkovW2Fp8(
            mMarkovW2, mMarkovW2Fp8, mMarkovW2RowScales, mDraftVocabSize, mMarkovRank, stream);
        LOG_INFO("DSparkDecoder: markov_w2 converted to FP8 E4M3 (per-row scales).");
    }
    LOG_INFO("DSpark W2: storage=%s, requested=%s, selected=%s", hasPackedW2 ? "nvfp4" : "fp16",
        dsparkW2BackendName(requested), dsparkW2BackendName(mW2Backend));
    if (requested == DSparkW2Backend::kAuto && hasPackedW2 && mW2Backend == DSparkW2Backend::kFp16)
    {
        LOG_INFO(
            "DSpark W2: W4A16 requires SM110, rank 512, aligned vocabulary, supported scales and AOT variants; "
            "using FP16 fallback (SM%d, rank=%d, vocab=%d, reduced_vocab=%d).",
            getSMVersion(), mMarkovRank, mDraftVocabSize, mHasDraftVocabMap);
    }
}

void DSparkDecoder::prepareW4A16(Tensor const& packedWeight, Tensor const& blockScales, cudaStream_t stream)
{
    using namespace trt_edgellm::kernels;
    int32_t const maxBatch = mRuntime.deployment.maxRuntimeBatchSize();
    mMarkovW2Opaque = Tensor({mDraftVocabSize / 128, mMarkovRank / 64, 128, 32}, DeviceType::kGPU,
        trt_edgellm::DataType::kUINT8, "DSpark::markovW2Opaque");
    mMarkovW2OpaqueScales = Tensor({mDraftVocabSize / 128, mMarkovRank / 64, 128, 4}, DeviceType::kGPU,
        trt_edgellm::DataType::kUINT8, "DSpark::markovW2OpaqueScales");
    mMarkovTcgen5Activation = Tensor(
        {maxBatch, mMarkovRank}, DeviceType::kGPU, trt_edgellm::DataType::kHALF, "DSpark::markovTcgen5Activation");
    mMarkovTcgen5Correction = Tensor(
        {maxBatch, mDraftVocabSize}, DeviceType::kGPU, trt_edgellm::DataType::kHALF, "DSpark::markovTcgen5Correction");
    kernel::dsparkRepackW2Tcgen5(
        packedWeight, blockScales, mMarkovW2Opaque, mMarkovW2OpaqueScales, mDraftVocabSize, mMarkovRank, stream);
    for (int32_t batch = 1; batch <= maxBatch; ++batch)
    {
        ELLM_CHECK(Nvfp4A16BlackwellGemmRunner::isSupported(
                       110, Nvfp4A16BlackwellDtype::kFp16, batch, mDraftVocabSize, mMarkovRank),
            "Missing TCGen5 W2 AOT variant");
        CUDA_CHECK(Nvfp4A16BlackwellGemmRunner::prepare(
            Nvfp4A16BlackwellDtype::kFp16, batch, mDraftVocabSize, mMarkovRank, stream));
    }
    Nvfp4A16BlackwellGemmParams const params{mMarkovTcgen5Activation.rawPointer(), mMarkovW2Opaque.rawPointer(),
        mMarkovW2OpaqueScales.rawPointer(), static_cast<float const*>(mMarkovW2GlobalScale.rawPointer()),
        mMarkovTcgen5Correction.rawPointer(), maxBatch, mDraftVocabSize, mMarkovRank, Nvfp4A16BlackwellDtype::kFp16};
    ELLM_CHECK(Nvfp4A16BlackwellGemmRunner::getWorkspaceSize(params) == 0, "Unexpected TCGen5 W2 workspace");
    LOG_INFO("DSparkDecoder: NVFP4 W2 TCGen5 W4A16 backend enabled.");
}

bool DSparkDecoder::decodeStep(DecodingInferenceContext& context)
{
    NVTX_SCOPED_RANGE(nvtx_dspark_decode, "DSparkDecoder::decodeStep", nvtx_colors::GREEN);
    cudaGetLastError();

    bool const usePendingPrefillProposal = mCommonStateTracker.shouldUsePendingPrefillProposal(context.generationRound);
    if (!usePendingPrefillProposal && !runDraftForward(context, /*prefillDelta=*/!mDraftPrimed))
    {
        LOG_ERROR("DSparkDecoder: draft forward failed.");
        return false;
    }
    mCommonStateTracker.materializePending(context.activeBatchSize);
    mCommonStateTracker.consumeDraftPrefillOutputs();

    // The first round's draft is built before prefill-completed slots are evicted, and the grafter's per-slot state
    // does not follow that compaction, so the walk runs here, on the batch the round verifies.
    mJump.reset();
    if (mJump.active(context) || mCaptureDeferred)
    {
        finishVerifyInput(context);
    }
    mCaptureDeferred = false;

    if (!runBaseVerification(context))
    {
        LOG_ERROR("DSparkDecoder: base verification failed.");
        return false;
    }
    mCommonStateTracker.recordAccepted(mHostAcceptLengths.dataPointer<int32_t>(), context.activeBatchSize);

    return true;
}

bool DSparkDecoder::initializeForGeneration(DecodingInferenceContext& context)
{
    if (context.speculativeDraftPrefillComplete)
    {
        return true;
    }
    mCommonStateTracker.initialize(context);

    if (!runDraftForward(context, /*prefillDelta=*/true))
    {
        LOG_ERROR("DSparkDecoder: failed to initialize draft state for generation.");
        return false;
    }
    mCommonStateTracker.markDraftPrefillOutputsPending();
    context.speculativeDraftPrefillComplete = true;
    return true;
}

void DSparkDecoder::beginBatch() noexcept
{
    mDraftPrimed = false;
}

void DSparkDecoder::appendRow(int32_t executionRow) noexcept
{
    // Within capacity by construction: the session appends no row past the runtime batch.
    check::check(mAcceptLength.reshape({executionRow + 1}) && mHostAcceptLengths.reshape({executionRow + 1}),
        "Tensor reshape failed");
    mCommonStateTracker.appendRow(executionRow);
}

bool DSparkDecoder::seatRow(DecodingInferenceContext& context, int32_t executionRow)
{
    int32_t const prefillTokens = context.effectivePrefillLengths[0];
    // The draft takes the target's prompt hidden states as a delta but the last one, which the next decode step
    // carries beside every resident's accepted span; the proposal this forward makes is not used.
    // Under Hybrid+spec endpoint reuse the checkpoint needs draft state at every prompt position, so the prefill
    // writes the last one too and then rolls the draft length back by one; the next decode step rewrites it.
    bool const writesLastPosition = context.hybridSpecEndpointReuse;
    int32_t const heldBackTokens = writesLastPosition ? 0 : 1;
    if (prefillTokens > heldBackTokens && !runDraftForward(context, /*prefillDelta=*/true, heldBackTokens))
    {
        LOG_ERROR("DSparkDecoder: failed to prefill the draft for a seated row.");
        return false;
    }
    if (writesLastPosition)
    {
        mDraftCacheManager.commitSequenceLength(-1, context.stream);
    }
    check::check(mRuntime.base.pipelineIO.baseHiddenStates.reshape({prefillTokens, mBaseOutputHiddenDim}),
        "Tensor reshape failed");
    // The pending delta is the last prompt position, as for a resident whose verification accepted one token.
    decoder_utils::seatAcceptedSpan(context, executionRow, mAcceptedTokenIds, mAcceptLength, mHostSeatStaging,
        mAcceptedHiddenStates, mAcceptCapacity, mRuntime.base.pipelineIO.baseHiddenStates, context.stream);
    mHostAcceptLengths.dataPointer<int32_t>()[executionRow] = 1;
    // Both models now hold the prompt but its last position, which the next decode step's draft pass takes.
    mCommonStateTracker.seatRow(
        executionRow, static_cast<int32_t>(context.rawBatchedInputIds[0].size()) - 1, /*pendingLength=*/1);
    mDraftPrimed = true;
    return true;
}

bool DSparkDecoder::prefillSeatedChunk(DecodingInferenceContext& context)
{
    return runDraftForward(context, /*prefillDelta=*/true);
}

std::vector<int32_t> const& DSparkDecoder::commonMaterializedStateLengths() const noexcept
{
    return mCommonStateTracker.commonMaterializedStateLengths();
}

// Keep these orchestration helpers in the decoder layer because logit-bias application
// consumes DecodingInferenceContext and runtime-owned sparse bias buffers; the kernel
// layer remains independent of request-local bias state.

// GCOVR_EXCL_START
void DSparkDecoder::dsparkBiasMarkovGreedy(
    DecodingInferenceContext& context, int32_t activeBatchSize, int32_t proposalLen)
{
    int32_t const logitsStride = mDraftBlockLen;
    int32_t constexpr topK = 1;
    int32_t const proposalDepthSize = proposalLen + 1;
    check::check(mDraftStepLogits.reshape({activeBatchSize, mDraftVocabSize}), "Tensor reshape failed");
    check::check(mDraftStepTopKValues.reshape({activeBatchSize, topK}), "Tensor reshape failed");
    check::check(mDraftStepTopKIndices.reshape({activeBatchSize, topK}), "Tensor reshape failed");
    if (mUseTree)
    {
        check::check(mStackedMarkovLogits.reshape({activeBatchSize, proposalDepthSize, mDraftVocabSize}),
            "Tensor reshape failed");
    }
    size_t const rowBytes = static_cast<size_t>(mDraftVocabSize) * sizeof(float);
    for (int32_t step = 0; step < proposalLen; ++step)
    {
        kernel::dsparkBuildMarkovLogits(mDraftOutputLogits, mMarkovW1, mMarkovW2, mLastAcceptedTokens, mDraftTokenIds,
            mDraftStepLogits, activeBatchSize, step, proposalLen, mDraftVocabSize, mMarkovRank, context.stream,
            logitsStride, mDraftSlotOffset);
        applyLogitBias(mRuntime.logitBias, mDraftStepLogits, context, context.stream);
        if (mUseTree)
        {
            CUDA_CHECK(cudaMemcpy2DAsync(
                static_cast<char*>(mStackedMarkovLogits.rawPointer()) + static_cast<size_t>(step + 1) * rowBytes,
                static_cast<size_t>(proposalDepthSize) * rowBytes, mDraftStepLogits.rawPointer(), rowBytes, rowBytes,
                activeBatchSize, cudaMemcpyDeviceToDevice, context.stream));
        }
        selectAllTopK(mDraftStepLogits, std::ref(mDraftStepTopKValues), mDraftStepTopKIndices, topK,
            mRuntime.sampling.workspace, context.stream);
        kernel::dsparkStoreDraftStepTop1(
            mDraftStepTopKIndices, mDraftTokenIds, activeBatchSize, step, proposalLen, context.stream);
    }
}

// DSpark stochastic verification consumes the proposal distribution q(y). With bias b,
// q_b(y) = softmax(logits(y) + b_y), so adding b after sampling would use the wrong
// acceptance distribution. Materialize the biased q_b rows before sampling and storage.
void DSparkDecoder::dsparkBiasMarkovSample(
    DecodingInferenceContext& context, int32_t activeBatchSize, int32_t proposalLen)
{
    RowSamplingParams const sampling = context.uniformSamplingParams();
    int32_t const logitsStride = mDraftBlockLen;
    check::check(mDraftStepLogits.reshape({activeBatchSize, mDraftVocabSize}), "Tensor reshape failed");
    check::check(mDraftStepProbabilities.reshape({activeBatchSize, mDraftVocabSize}), "Tensor reshape failed");
    for (int32_t step = 0; step < proposalLen; ++step)
    {
        kernel::dsparkBuildMarkovLogits(mDraftOutputLogits, mMarkovW1, mMarkovW2, mLastAcceptedTokens, mDraftTokenIds,
            mDraftStepLogits, activeBatchSize, step, proposalLen, mDraftVocabSize, mMarkovRank, context.stream,
            logitsStride, mDraftSlotOffset);
        applyLogitBias(mRuntime.logitBias, mDraftStepLogits, context, context.stream);
        kernel::dsparkLogitsToProbabilities(mDraftStepLogits, mDraftStepProbabilities, activeBatchSize, mDraftVocabSize,
            sampling.temperature, static_cast<int32_t>(sampling.topK), sampling.topP, context.stream);
        kernel::dsparkSampleProbabilityRows(mDraftStepProbabilities, mDraftUniforms, mDraftTokenIds, activeBatchSize,
            step, proposalLen, mDraftVocabSize, context.stream);
        kernel::dsparkStoreDraftStepProbabilities(mDraftStepProbabilities, mDraftProbabilities, activeBatchSize, step,
            proposalLen, mDraftVocabSize, context.stream);
    }
}
// GCOVR_EXCL_STOP

bool DSparkDecoder::runDraftForward(DecodingInferenceContext& context, bool prefillDelta, int32_t heldBackTokens)
{
    TIME_STAGE(metrics::StageNames::kSPEC_DECODE_DRAFT_PROPOSAL, context.stream);
    NVTX_SCOPED_RANGE(nvtx_dspark_draft, "DSparkDecoder::runDraftForward", nvtx_colors::DARK_ORANGE);

    if (mUseTree && ::trt_edgellm::shouldUseNonGreedySampling(context.uniformSamplingParams()))
    {
        LOG_ERROR("DSparkDecoder: tree drafting supports greedy decoding only; disable sampling or use draftTopK=1.");
        return false;
    }
    if (!mDraftExecutor)
    {
        LOG_ERROR("DSparkDecoder: draft engine not loaded.");
        return false;
    }

    int32_t const activeBatchSize = context.activeBatchSize;
    int32_t const proposalLen = mProposalLen;
    int32_t const blockLen = mDraftBlockLen;

    check::check(mRuntime.preprocess.idsInput.reshape({activeBatchSize, blockLen}), "Tensor reshape failed");
    check::check(mHostDraftInputIds.reshape({activeBatchSize, blockLen}), "Tensor reshape failed");
    check::check(mHostLastAcceptedTokens.reshape({activeBatchSize}), "Tensor reshape failed");
    check::check(mLastAcceptedTokens.reshape({activeBatchSize}), "Tensor reshape failed");
    mHostStagingFence.wait();
    int32_t* hostDraftInputIds = mHostDraftInputIds.dataPointer<int32_t>();
    int32_t* hostLastAccepted = mHostLastAcceptedTokens.dataPointer<int32_t>();
    for (int32_t b = 0; b < activeBatchSize; ++b)
    {
        hostLastAccepted[b] = context.tokenIds[b].back();
        hostDraftInputIds[b * blockLen] = hostLastAccepted[b];
        for (int32_t j = 1; j < blockLen; ++j)
        {
            hostDraftInputIds[b * blockLen + j] = mMaskTokenId;
        }
    }
    CUDA_CHECK(cudaMemcpyAsync(mRuntime.preprocess.idsInput.rawPointer(), mHostDraftInputIds.rawPointer(),
        activeBatchSize * blockLen * sizeof(int32_t), cudaMemcpyHostToDevice, context.stream));
    CUDA_CHECK(cudaMemcpyAsync(mLastAcceptedTokens.rawPointer(), mHostLastAcceptedTokens.rawPointer(),
        activeBatchSize * sizeof(int32_t), cudaMemcpyHostToDevice, context.stream));

    check::check(mDraftInputsEmbeds.reshape({activeBatchSize, blockLen, mDraftHiddenSize}), "Tensor reshape failed");
    kernel::embeddingLookup(mRuntime.preprocess.idsInput, mRuntime.preprocess.embedding.table,
        mRuntime.preprocess.embedding.scalesAsOptional(), mRuntime.preprocess.embedding.descriptor, mDraftInputsEmbeds,
        context.stream);

    // The first forward of a request (and a seated row's) takes the target's prefill hidden states as its delta;
    // later rounds take the accepted span the decoder carried from the last verification.
    check::check(heldBackTokens == 0 || (prefillDelta && activeBatchSize == 1),
        "Only a seated row's draft prefill holds positions back");
    int64_t maxDeltaLen;
    int64_t sourceSeqLen;
    check::check(mHostDeltaLens.reshape({activeBatchSize}), "Tensor reshape failed");
    int32_t* hostDeltaLens = mHostDeltaLens.dataPointer<int32_t>();

    if (prefillDelta)
    {
        // Over the active batch: a seated row's prefill runs as a batch of one while the per-slot vectors keep every
        // resident's entry.
        sourceSeqLen = *std::max_element(
            context.effectivePrefillLengths.begin(), context.effectivePrefillLengths.begin() + activeBatchSize);
        maxDeltaLen = 0;
        for (int32_t b = 0; b < activeBatchSize; ++b)
        {
            int32_t const prefillLen = context.effectivePrefillLengths[b] - heldBackTokens;
            hostDeltaLens[b] = prefillLen;
            maxDeltaLen = std::max(maxDeltaLen, static_cast<int64_t>(prefillLen));
        }
    }
    else
    {
        sourceSeqLen = mAcceptCapacity;
        int32_t const* hostAccLens = mHostAcceptLengths.dataPointer<int32_t>();
        maxDeltaLen = 0;
        for (int32_t b = 0; b < activeBatchSize; ++b)
        {
            hostDeltaLens[b] = hostAccLens[b];
            maxDeltaLen = std::max(maxDeltaLen, static_cast<int64_t>(hostAccLens[b]));
        }
    }

    check::check(mDraftDeltaLens.reshape({activeBatchSize}), "Tensor reshape failed");
    CUDA_CHECK(cudaMemcpyAsync(mDraftDeltaLens.rawPointer(), mHostDeltaLens.rawPointer(),
        activeBatchSize * sizeof(int32_t), cudaMemcpyHostToDevice, context.stream));

    check::check(
        mDraftTargetHidden.reshape({activeBatchSize, maxDeltaLen, mBaseOutputHiddenDim}), "Tensor reshape failed");
    {
        size_t const elementBytes = utils::getTypeSize(mDraftTargetHidden.getDataType());
        size_t const rowBytes = static_cast<size_t>(mBaseOutputHiddenDim) * elementBytes;
        size_t const dstPitch = static_cast<size_t>(maxDeltaLen) * rowBytes;
        size_t const srcPitch = static_cast<size_t>(sourceSeqLen) * rowBytes;
        size_t const widthBytes = static_cast<size_t>(maxDeltaLen) * rowBytes;
        Tensor const& hiddenSource = prefillDelta ? mRuntime.base.pipelineIO.baseHiddenStates : mAcceptedHiddenStates;
        CUDA_CHECK(cudaMemcpy2DAsync(mDraftTargetHidden.rawPointer(), dstPitch, hiddenSource.rawPointer(), srcPitch,
            widthBytes, activeBatchSize, cudaMemcpyDeviceToDevice, context.stream));
    }

    int32_t const pmLen = divUp(blockLen, 32);
    check::check(mDraftPackedAttentionMask.reshape({activeBatchSize, blockLen, pmLen}), "Tensor reshape failed");
    check::check(mDraftAttentionPosId.reshape({activeBatchSize, blockLen}), "Tensor reshape failed");
    check::check(mDraftContextLengths.reshape({activeBatchSize}), "Tensor reshape failed");
    Tensor const& draftCacheLengths = mDraftCacheManager.getKVCacheLengths();
    kernel::launchDFlashPrepareProposalInputs(draftCacheLengths.dataPointer<int32_t>(),
        mDraftDeltaLens.dataPointer<int32_t>(), blockLen, mDraftPackedAttentionMask.dataPointer<int32_t>(),
        mDraftAttentionPosId.dataPointer<int32_t>(), mDraftContextLengths.dataPointer<int32_t>(),
        mRuntime.base.pipelineIO.positions.dataPointer<int32_t>(),
        mRuntime.base.pipelineIO.queryStartOffsets.dataPointer<int32_t>(),
        mRuntime.base.pipelineIO.queryLengths.dataPointer<int32_t>(),
        mRuntime.base.pipelineIO.pastLengths.dataPointer<int32_t>(),
        mRuntime.base.pipelineIO.attentionSequenceLengths.dataPointer<int32_t>(),
        mRuntime.base.pipelineIO.stateIndices.dataPointer<int32_t>(), mCausalProposalMask, activeBatchSize,
        context.stream);
    prepareDraftRaggedBindings(
        activeBatchSize, blockLen, static_cast<int32_t>(maxDeltaLen), context.stream, &context.residentRefs);

    check::check(mDraftOutputLogits.reshape({activeBatchSize, blockLen, mDraftVocabSize}), "Tensor reshape failed");
    check::check(mDraftHiddenStates.reshape({activeBatchSize, blockLen, mDraftHiddenSize}), "Tensor reshape failed");

    int32_t const draftKVCapacity = mRuntime.deployment.draft->maxKVCacheCapacity;
    InferenceDims const draftDims{
        /*.batch=*/activeBatchSize,
        /*.seqLen=*/activeBatchSize * blockLen,
        /*.kvLen=*/draftKVCapacity,
        /*.selectLen=*/static_cast<int64_t>(activeBatchSize) * maxDeltaLen,
        /*.attnMaskSeqLen=*/activeBatchSize * blockLen,
        /*.ropeBatch=*/1,
        /*.packedMaskLen=*/static_cast<int64_t>(pmLen),
        /*.contextMaskSelectorLen=*/0,
        /*.startIndexLen=*/activeBatchSize,
        /*.executionPhaseLen=*/static_cast<int64_t>(ExecutionPhase::kSpecDraftProposal),
        /*.specVerifyLayoutLen=*/0,
        /*.skipSoftmaxPrefillScaleLen=*/0,
        /*.skipSoftmaxDecodeScaleLen=*/0,
        /*.swaKVCacheModeLen=*/0,
        /*.queryOffsetLen=*/activeBatchSize + 1,
        /*.contextSequenceCount=*/0,
    };

    cudaGetLastError();
    bool draftSuccess;
    {
        auto forward
            = gTimer.startForward("draft", draftDims.batch, draftDims.seqLen, draftDims.selectLen, context.stream);
        draftSuccess
            = mDraftExecutor->prepare(dsparkDraftProfile(prefillDelta), draftDims, mDraftTensorMap, context.stream);
        if (draftSuccess)
        {
            draftSuccess = mDraftExecutor->execute(context.stream);
        }
    }
    if (!draftSuccess)
    {
        LOG_ERROR("DSparkDecoder: draft engine execution failed.");
        return false;
    }

    if (prefillDelta)
    {
        check::check(mDraftDeltaLenCommit.reshape({activeBatchSize}), "Tensor reshape failed");
        CUDA_CHECK(cudaMemcpyAsync(mDraftDeltaLenCommit.rawPointer(), mHostDeltaLens.rawPointer(),
            activeBatchSize * sizeof(int32_t), cudaMemcpyHostToDevice, context.stream));
        mDraftCacheManager.commitSequenceLength(mDraftDeltaLenCommit, context.stream);
    }
    else
    {
        check::check(mAcceptLength.reshape({activeBatchSize}), "Tensor reshape failed");
        mDraftCacheManager.commitSequenceLength(mAcceptLength, context.stream);
    }
    mHostStagingFence.record(context.stream);
    mDraftPrimed = true;

    check::check(mDraftTokenIds.reshape({activeBatchSize, proposalLen}), "Tensor reshape failed");
    RowSamplingParams const sampling = context.uniformSamplingParams();
    bool const stochasticSampling = ::trt_edgellm::shouldUseNonGreedySampling(sampling);
    mDraftDistribution = DraftDistribution::kNone;
    if (stochasticSampling && mHasDraftVocabMap)
    {
        LOG_ERROR(
            "DSparkDecoder: reduced-vocabulary DSpark supports greedy decoding only; stochastic verification "
            "requires a full-vocabulary draft probability distribution.");
        return false;
    }
    if (stochasticSampling)
    {
        if (mUseTree)
        {
            LOG_ERROR(
                "DSparkDecoder: DDTree drafting supports greedy decoding only; disable sampling or use "
                "draftTopK=1.");
            return false;
        }
        check::check(
            mDraftProbabilities.reshape({activeBatchSize, proposalLen, mDraftVocabSize}), "Tensor reshape failed");
        check::check(mDraftUniforms.reshape({activeBatchSize, proposalLen}), "Tensor reshape failed");
        uint64_t const randomOffset
            = dsparkRandomRoundOffset(context.generationRound, mRuntime.maxRuntimeBatchSize, proposalLen);
        kernel::dsparkFillUniforms(
            mDraftUniforms, activeBatchSize * proposalLen, kDSparkSamplingSeed, randomOffset, context.stream);
        check::check(mDraftStepLogits.reshape({activeBatchSize, mDraftVocabSize}), "Tensor reshape failed");
        if (dsparkCanUseTopKFastPath(sampling.topK, sampling.topP))
        {
            mDraftDistribution = DraftDistribution::kTopK;
            int32_t const samplingTopK = static_cast<int32_t>(sampling.topK);
            check::check(mDraftStepTopKValues.reshape({activeBatchSize, samplingTopK}), "Tensor reshape failed");
            check::check(mDraftStepTopKIndices.reshape({activeBatchSize, samplingTopK}), "Tensor reshape failed");
            check::check(
                mDraftTopKProbabilities.reshape({activeBatchSize, proposalLen, samplingTopK}), "Tensor reshape failed");
            check::check(
                mDraftTopKIndices.reshape({activeBatchSize, proposalLen, samplingTopK}), "Tensor reshape failed");
            for (int32_t step = 0; step < proposalLen; ++step)
            {
                kernel::dsparkBuildMarkovLogits(mDraftOutputLogits, mMarkovW1, mMarkovW2, mLastAcceptedTokens,
                    mDraftTokenIds, mDraftStepLogits, activeBatchSize, step, proposalLen, mDraftVocabSize, mMarkovRank,
                    context.stream, blockLen, mDraftSlotOffset);
                // GCOVR_EXCL_START
                if (context.hasLogitBias)
                {
                    applyLogitBias(mRuntime.logitBias, mDraftStepLogits, context, context.stream);
                }
                // GCOVR_EXCL_STOP
                selectAllTopK(mDraftStepLogits, std::ref(mDraftStepTopKValues), mDraftStepTopKIndices, samplingTopK,
                    mRuntime.sampling.workspace, context.stream);
                kernel::dsparkSampleTopKRowsAndStore(mDraftStepTopKValues, mDraftStepTopKIndices, mDraftUniforms,
                    mDraftTokenIds, mDraftTopKProbabilities, mDraftTopKIndices, activeBatchSize, step, proposalLen,
                    samplingTopK, sampling.temperature, context.stream);
            }
        }
        else
        {
            mDraftDistribution = DraftDistribution::kDense;
            // GCOVR_EXCL_START
            if (context.hasLogitBias)
            {
                dsparkBiasMarkovSample(context, activeBatchSize, proposalLen);
            }
            // GCOVR_EXCL_STOP
            else
            {
                kernel::dsparkVanillaMarkovSample(mDraftOutputLogits, mMarkovW1, mMarkovW2, mLastAcceptedTokens,
                    mDraftUniforms, mDraftTokenIds, mDraftProbabilities, mDraftStepLogits, mDraftStepProbabilities,
                    activeBatchSize, proposalLen, mDraftVocabSize, mMarkovRank, sampling.temperature,
                    static_cast<int32_t>(sampling.topK), sampling.topP, context.stream, blockLen, mDraftSlotOffset);
            }
        }
    }
    else
    {
        // GCOVR_EXCL_START
        if (context.hasLogitBias)
        {
            dsparkBiasMarkovGreedy(context, activeBatchSize, proposalLen);
        }
        // GCOVR_EXCL_STOP
        else
        {
            if (!mHasDraftVocabMap && kernel::dsparkFusedGreedySupported(mMarkovRank))
            {
                check::check(mMarkovGreedySlots.reshape({activeBatchSize, proposalLen}), "Tensor reshape failed");
                CUDA_CHECK(cudaMemsetAsync(mMarkovGreedySlots.rawPointer(), 0,
                    static_cast<size_t>(activeBatchSize) * proposalLen * sizeof(uint64_t), context.stream));
                for (int32_t step = 0; step < proposalLen; ++step)
                {
                    switch (mW2Backend)
                    {
                    case DSparkW2Backend::kW4A16:
                        kernel::dsparkMarkovGreedyStepTcgen5(mDraftOutputLogits, mMarkovW1, mMarkovW2Opaque,
                            mMarkovW2OpaqueScales, mMarkovW2GlobalScale, mLastAcceptedTokens, mMarkovGreedySlots,
                            mMarkovTcgen5Activation, mMarkovTcgen5Correction,
                            mUseTree ? &mStackedMarkovLogits : nullptr, step + 1, activeBatchSize, step, proposalLen,
                            mDraftVocabSize, mMarkovRank, context.stream, blockLen, mDraftSlotOffset);
                        break;
                    case DSparkW2Backend::kFp8:
                        kernel::dsparkMarkovGreedyFusedStepFp8(mDraftOutputLogits, mMarkovW1, mMarkovW2Fp8,
                            mMarkovW2RowScales, mLastAcceptedTokens, mMarkovGreedySlots,
                            mUseTree ? &mStackedMarkovLogits : nullptr, step + 1, activeBatchSize, step, proposalLen,
                            mDraftVocabSize, mMarkovRank, context.stream, blockLen, mDraftSlotOffset);
                        break;
                    case DSparkW2Backend::kFp16:
                        kernel::dsparkMarkovGreedyFusedStep(mDraftOutputLogits, mMarkovW1, mMarkovW2,
                            mLastAcceptedTokens, mMarkovGreedySlots, mUseTree ? &mStackedMarkovLogits : nullptr,
                            step + 1, activeBatchSize, step, proposalLen, mDraftVocabSize, mMarkovRank, context.stream,
                            blockLen, mDraftSlotOffset);
                        break;
                    default: throw std::logic_error("Unresolved DSpark W2 backend");
                    }
                }
                kernel::dsparkFinalizeGreedyDraftTokens(
                    mMarkovGreedySlots, mDraftTokenIds, activeBatchSize * proposalLen, context.stream);
            }
            else
            {
                check::check(mDraftStepLogits.reshape({activeBatchSize, mDraftVocabSize}), "Tensor reshape failed");
                check::check(mDraftStepTopKValues.reshape({activeBatchSize, 1}), "Tensor reshape failed");
                check::check(mDraftStepTopKIndices.reshape({activeBatchSize, 1}), "Tensor reshape failed");
                int32_t const proposalDepthSize = proposalLen + 1;
                if (mUseTree)
                {
                    check::check(mStackedMarkovLogits.reshape({activeBatchSize, proposalDepthSize, mDraftVocabSize}),
                        "Tensor reshape failed");
                }
                size_t const rowBytes = static_cast<size_t>(mDraftVocabSize) * sizeof(float);
                for (int32_t step = 0; step < proposalLen; ++step)
                {
                    kernel::dsparkBuildMarkovLogits(mDraftOutputLogits, mMarkovW1, mMarkovW2, mLastAcceptedTokens,
                        mDraftTokenIds, mDraftStepLogits, activeBatchSize, step, proposalLen, mDraftVocabSize,
                        mMarkovRank, context.stream, blockLen, mDraftSlotOffset);
                    if (mUseTree)
                    {
                        CUDA_CHECK(cudaMemcpy2DAsync(static_cast<char*>(mStackedMarkovLogits.rawPointer())
                                + static_cast<size_t>(step + 1) * rowBytes,
                            static_cast<size_t>(proposalDepthSize) * rowBytes, mDraftStepLogits.rawPointer(), rowBytes,
                            rowBytes, activeBatchSize, cudaMemcpyDeviceToDevice, context.stream));
                    }
                    selectAllTopK(mDraftStepLogits, std::ref(mDraftStepTopKValues), mDraftStepTopKIndices, /*topK=*/1,
                        mRuntime.sampling.workspace, context.stream);
                    if (mHasDraftVocabMap)
                    {
                        mapReducedVocabToFullVocab(mDraftStepTopKIndices, mDraftVocabMappingTable, context.stream);
                    }
                    CUDA_CHECK(cudaMemcpy2DAsync(
                        static_cast<char*>(mDraftTokenIds.rawPointer()) + static_cast<size_t>(step) * sizeof(int32_t),
                        static_cast<size_t>(proposalLen) * sizeof(int32_t), mDraftStepTopKIndices.rawPointer(),
                        sizeof(int32_t), sizeof(int32_t), activeBatchSize, cudaMemcpyDeviceToDevice, context.stream));
                }
            }
        }
    }

    check::check(mProposalLengths.reshape({activeBatchSize}), "Tensor reshape failed");
    mCurrentProposalLen = proposalLen;
    if (!mUseTree && mSchedulerMode != DSparkSchedulerMode::kOff && mHasConfidenceHead)
    {
        check::check(mConfidenceScores.reshape({activeBatchSize, proposalLen}), "Tensor reshape failed");
        if (mSchedulerMode == DSparkSchedulerMode::kSPS)
        {
            kernel::dsparkComputeConfidenceAndSPSProposalLengths(mDraftHiddenStates, mMarkovW1, mConfidenceWeight,
                mConfidenceBias, mLastAcceptedTokens, mDraftTokenIds, mConfidenceScores, mProposalLengths,
                activeBatchSize, proposalLen, mDraftHiddenSize, mMarkovRank, mConfidenceHeadWithMarkov,
                mConfidenceThreshold, mMinScheduledProposalLen, mMaxScheduledProposalLen, context.stream, blockLen,
                mDraftSlotOffset);
        }
        else
        {
            kernel::dsparkComputeConfidenceAndProposalLengths(mDraftHiddenStates, mMarkovW1, mConfidenceWeight,
                mConfidenceBias, mLastAcceptedTokens, mDraftTokenIds, mConfidenceScores, mProposalLengths,
                activeBatchSize, proposalLen, mDraftHiddenSize, mMarkovRank, mConfidenceHeadWithMarkov,
                mConfidenceThreshold, mMinScheduledProposalLen, mMaxScheduledProposalLen, context.stream, blockLen,
                mDraftSlotOffset);
        }

        check::check(mHostProposalLengths.reshape({activeBatchSize}), "Tensor reshape failed");
        CUDA_CHECK(cudaMemcpyAsync(mHostProposalLengths.rawPointer(), mProposalLengths.rawPointer(),
            activeBatchSize * sizeof(int32_t), cudaMemcpyDeviceToHost, context.stream));
        CUDA_CHECK(cudaStreamSynchronize(context.stream));
        int32_t maxScheduledProposalLen = 1;
        int32_t const* hostProposalLengths = mHostProposalLengths.dataPointer<int32_t>();
        for (int32_t batchIdx = 0; batchIdx < activeBatchSize; ++batchIdx)
        {
            maxScheduledProposalLen = std::max(maxScheduledProposalLen, hostProposalLengths[batchIdx]);
        }
        mCurrentProposalLen = std::max(1, std::min(proposalLen, maxScheduledProposalLen));
        mScheduledProposalLengths.resize(activeBatchSize);
        for (int32_t batchIdx = 0; batchIdx < activeBatchSize; ++batchIdx)
        {
            mScheduledProposalLengths[batchIdx] = std::min(hostProposalLengths[batchIdx], mCurrentProposalLen);
        }
    }
    else
    {
        kernel::dsparkFillProposalLengths(mProposalLengths, activeBatchSize, proposalLen, context.stream);
        mScheduledProposalLengths.clear();
    }
    if (mUseTree)
    {
        mCurrentVerifyLen = mVerifyLen;
        if (mUseTreeScheduler)
        {
            check::check(mConfidenceScores.reshape({activeBatchSize, proposalLen}), "Tensor reshape failed");
            kernel::dsparkComputeConfidenceScores(mDraftHiddenStates, mMarkovW1, mConfidenceWeight, mConfidenceBias,
                mLastAcceptedTokens, mDraftTokenIds, mConfidenceScores, activeBatchSize, proposalLen, mDraftHiddenSize,
                mMarkovRank, mConfidenceHeadWithMarkov, context.stream, blockLen, mDraftSlotOffset);
        }
        if (!buildTreeVerifyInputs(activeBatchSize, context.stream, mUseTreeScheduler))
        {
            return false;
        }
        // A jump-forward round captures its tree once grafted (finishVerifyInput).
        mCaptureDeferred = context.hasGuidedDecoding && mJump.active(context);
        if (context.hasGuidedDecoding && !mJump.active(context))
        {
            // Graph capture builds placeholder tree tensors, so capture only the populated runtime tree.
            mRuntime.guidedDecoder.captureDraft(
                {mTreeTokenIds, &mValidCounts, &mRuntime.base.pipelineIO.specTreeParentIds, nullptr, activeBatchSize,
                    mCurrentVerifyLen, mRuntime.deployment.specConfig->maxPathLength, SpecVerifyLayout::kTree},
                context.stream);
        }
        return true;
    }
    // A grafted chain may run past a slot's scheduled length, so a jump-forward request verifies the whole proposal;
    // each slot keeps its scheduled length as its node count.
    if (mJump.active(context))
    {
        mCurrentProposalLen = proposalLen;
    }
    mCurrentVerifyLen = mCurrentProposalLen + 1;

    check::check(mVerifyTokenIds.reshape({activeBatchSize, mCurrentVerifyLen}), "Tensor reshape failed");
    check::check(mAcceptValidTokenCounts.reshape({activeBatchSize}), "Tensor reshape failed");
    kernel::dsparkBuildVerifyTokens(mLastAcceptedTokens, mDraftTokenIds, mProposalLengths, mVerifyTokenIds,
        mAcceptValidTokenCounts, activeBatchSize, mProposalLen, mCurrentProposalLen, context.stream);

    // A jump-forward round captures its chain once grafted (finishVerifyInput).
    mCaptureDeferred = context.hasGuidedDecoding && mJump.active(context);
    if (context.hasGuidedDecoding && !mJump.active(context))
    {
        // The chain scheduler can shorten the proposal, so the row count is this step's verify
        // length rather than the deployment's verify size.
        mRuntime.guidedDecoder.captureDraft(
            {mVerifyTokenIds, nullptr, nullptr, nullptr, activeBatchSize, mCurrentVerifyLen,
                mRuntime.deployment.specConfig->maxPathLength, SpecVerifyLayout::kChain},
            context.stream);
    }

    return true;
}

void DSparkDecoder::finishVerifyInput(DecodingInferenceContext& context)
{
    auto const& specConfig = *mRuntime.deployment.specConfig;
    auto& io = mRuntime.base.pipelineIO;
    int32_t const activeBatchSize = context.activeBatchSize;
    Tensor& ids = mUseTree ? mTreeTokenIds : mVerifyTokenIds;
    Tensor* const parents = mUseTree ? &io.specTreeParentIds : nullptr;
    // A scheduled chain's node counts are its slots' scheduled lengths; an unscheduled chain's are all V.
    bool const scheduledChain = !mUseTree && !mScheduledProposalLengths.empty();
    Tensor const* const draftCounts = mUseTree ? &mValidCounts : (scheduledChain ? &mAcceptValidTokenCounts : nullptr);
    SpecVerifyInputs const drafted{ids, draftCounts, parents, nullptr, activeBatchSize, mCurrentVerifyLen,
        specConfig.maxPathLength, specConfig.verifyLayout};

    // A slot that may jump has its draft walked on the host before verification. A tree that takes chains verifies at
    // the grafting width and pads the other slots; a chain takes its chain in place. Either way a grafting round's node
    // counts are the grafted lengths, so draft rows a graft cut are padding, not nodes.
    using Walk = JumpGraftSession::Walk;
    Walk const walk = mJump.walk(context, drafted);
    if (mJump.wide())
    {
        mJump.grafter().upload(ids, parents, mUseTree ? &io.specTreeDepths : nullptr, context.stream);
        if (mUseTree)
        {
            mJump.prepareGraftedTreeDescriptors(
                io, mRuntime.base.cacheManager.getKVCacheLengths(), activeBatchSize, context.stream);
            mCurrentVerifyLen = specConfig.jumpVerifyTokenCount;
        }
    }

    // After a walk that grafted nothing the guided decoder already holds the draft; a grafting round captures the
    // grafted input with its node counts, which end each slot's mask walk, and its forced flags.
    if (context.hasGuidedDecoding && walk != Walk::kNoChain)
    {
        mRuntime.guidedDecoder.captureDraft(mJump.wide()
                ? SpecVerifyInputs{ids, mJump.grafter().validCounts(), parents, nullptr, activeBatchSize,
                      mCurrentVerifyLen, specConfig.maxPathLength, specConfig.verifyLayout, false,
                      mJump.grafter().forcedFlags()}
                : drafted,
            context.stream);
    }
}

bool DSparkDecoder::buildTreeVerifyInputs(int32_t activeBatchSize, cudaStream_t stream, bool useConfidence)
{
    NVTX_SCOPED_RANGE(nvtx_dspark_tree_build, "DSparkDecoder::buildTreeVerifyInputs", nvtx_colors::LIGHT_ORANGE);
    int32_t const verifyTokenCount = mVerifyLen;
    check::check(mLastAcceptedTokens.reshape({activeBatchSize}), "Tensor reshape failed");
    check::check(mTreeTokenIds.reshape({activeBatchSize, verifyTokenCount}), "Tensor reshape failed");
    check::check(
        mRuntime.base.pipelineIO.specTreeDepths.reshape({activeBatchSize, verifyTokenCount}), "Tensor reshape failed");
    check::check(mRuntime.base.pipelineIO.specTreeParentIds.reshape({activeBatchSize, verifyTokenCount}),
        "Tensor reshape failed");
    check::check(mTreeNodeScores.reshape({activeBatchSize, verifyTokenCount}), "Tensor reshape failed");
    check::check(mValidCounts.reshape({activeBatchSize}), "Tensor reshape failed");
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
    kernel::DDTreeBuildOutputs const outputs{mTreeTokenIds, mRuntime.base.pipelineIO.specTreeDepths,
        mRuntime.base.pipelineIO.specTreeParentIds, mTreeNodeScores, mValidCounts, mRuntime.preprocess.idsInput,
        mRuntime.base.pipelineIO.specDecodePositionIds, mRuntime.base.pipelineIO.packedAttentionMask,
        mRuntime.base.pipelineIO.contextLengths, mRuntime.base.pipelineIO.selectTokenIndices};
    int32_t const proposalDepthSize = mProposalLen + 1;
    check::check(
        mStackedMarkovLogits.reshape({activeBatchSize, proposalDepthSize, mDraftVocabSize}), "Tensor reshape failed");
    kernel::DDTreeBuildParams const buildParams{
        {mStackedMarkovLogits, mLastAcceptedTokens, baseKVCacheLengths,
            mHasDraftVocabMap ? &mDraftVocabMappingTable : nullptr, useConfidence ? &mConfidenceScores : nullptr,
            useConfidence ? mConfidenceThreshold : 0.0F},
        outputs, mRuntime.deployment.specConfig->draftTopK, mTreeBuildWorkspace.rawPointer(),
        static_cast<size_t>(mTreeBuildWorkspace.getMemoryCapacity()), stream};
    kernel::ddtreeBuild(buildParams);

    return true;
}

void DSparkDecoder::prepareDraftRaggedBindings(int32_t activeBatchSize, int32_t executionWidth, int32_t deltaWidth,
    cudaStream_t stream, std::vector<ResidentRef> const* residentRefs)
{
    auto& io = mRuntime.base.pipelineIO;
    auto const& cfg = *mRuntime.deployment.draft;
    int32_t const proposalTokens = activeBatchSize * executionWidth;
    int32_t const deltaTokens = activeBatchSize * deltaWidth;

    check::check(mDraftInputsEmbeds.reshape({proposalTokens, mDraftHiddenSize}), "Tensor reshape failed");
    check::check(mDraftOutputLogits.reshape({proposalTokens, mDraftVocabSize}), "Tensor reshape failed");
    check::check(mDraftHiddenStates.reshape({proposalTokens, mDraftHiddenSize}), "Tensor reshape failed");
    check::check(mDraftPackedAttentionMask.reshape({proposalTokens, static_cast<int64_t>(divUp(executionWidth, 32))}),
        "Tensor reshape failed");
    check::check(mDraftAttentionPosId.reshape({proposalTokens}), "Tensor reshape failed");
    check::check(io.positions.reshape({proposalTokens}), "Tensor reshape failed");
    check::check(io.queryStartOffsets.reshape({activeBatchSize + 1}), "Tensor reshape failed");
    check::check(io.queryLengths.reshape({activeBatchSize}), "Tensor reshape failed");
    check::check(io.pastLengths.reshape({activeBatchSize}), "Tensor reshape failed");
    check::check(io.attentionSequenceLengths.reshape({activeBatchSize}), "Tensor reshape failed");
    check::check(io.contextSequenceCountCarrier.reshape({0}), "Tensor reshape failed");
    io.uploadStateIndices(residentRefs, activeBatchSize, stream);
    prepareRaggedKVPageTable(io, *mRuntime.base.sharedResources.kvPageTables[1], activeBatchSize, stream);
    prepareRaggedRope(io, mRuntime.base.sharedResources, cfg, proposalTokens, activeBatchSize, stream);

    check::check(mDraftDeltaPositions.reshape({deltaTokens}), "Tensor reshape failed");
    check::check(mDraftDeltaTokenToSequence.reshape({deltaTokens}), "Tensor reshape failed");
    check::check(mDraftDeltaRopeCosSin.reshape({deltaTokens, cfg.rotaryDim}), "Tensor reshape failed");
    kernel::launchDFlashPrepareDeltaMetadata(mDraftCacheManager.getKVCacheLengths().dataPointer<int32_t>(),
        mDraftDeltaLens.dataPointer<int32_t>(), deltaWidth, mDraftDeltaPositions.dataPointer<int32_t>(),
        mDraftDeltaTokenToSequence.dataPointer<int32_t>(), activeBatchSize, stream);
    Tensor const& ropeSource = cfg.ropeConfig.type == RopeType::kMRope
        ? io.mropeCosSin
        : mRuntime.base.sharedResources.ropePool.getOrCreate(
              cfg.ropeConfig, cfg.rotaryDim, cfg.maxKVCacheCapacity, stream);
    int32_t const sourceRows = cfg.ropeConfig.type == RopeType::kMRope ? cfg.recurrentPoolRows : 1;
    kernel::launchDFlashGatherDeltaRope(ropeSource.dataPointer<float>(), mDraftDeltaRopeCosSin.dataPointer<float>(),
        mDraftDeltaPositions.dataPointer<int32_t>(), mDraftDeltaTokenToSequence.dataPointer<int32_t>(),
        sourceRows == 1 ? nullptr : io.stateIndices.dataPointer<int32_t>(), deltaTokens, activeBatchSize, sourceRows,
        cfg.maxKVCacheCapacity, cfg.rotaryDim, stream);
    check::check(mDraftTargetHidden.reshape({deltaTokens, mBaseOutputHiddenDim}), "Tensor reshape failed");
}

bool DSparkDecoder::runBaseVerification(DecodingInferenceContext& context)
{
    TIME_STAGE(metrics::StageNames::kSPEC_DECODE_BASE_VERIFICATION, context.stream);
    NVTX_SCOPED_RANGE(nvtx_dspark_verify, "DSparkDecoder::runBaseVerification", nvtx_colors::MAGENTA);

    int32_t const activeBatchSize = context.activeBatchSize;
    int32_t const verifyLen = mCurrentVerifyLen;
    // A grafting round verifies the grafted input: its node counts end each slot's rows, and acceptance takes its
    // forced nodes whatever the base predicts. The layout stays the deployment's.
    Tensor const* verifyCounts = mUseTree ? &mValidCounts : nullptr;
    Tensor const* forcedFlags = nullptr;
    if (mJump.wide())
    {
        verifyCounts = mJump.grafter().validCounts();
        forcedFlags = mJump.grafter().forcedFlags();
    }

    SpecVerifyInputs const verify{mUseTree ? mTreeTokenIds : mVerifyTokenIds, verifyCounts,
        mUseTree ? &mRuntime.base.pipelineIO.specTreeParentIds : nullptr,
        mUseTree ? &mRuntime.base.pipelineIO.specTreeDepths : nullptr, activeBatchSize, verifyLen,
        std::min(verifyLen, mRuntime.deployment.specConfig->maxPathLength),
        mRuntime.deployment.specConfig->verifyLayout, mUseTree};
    spec_decode::prepareTargetVerifyInputs(verify, mRuntime, context, 0);

    cudaGetLastError();
    auto const verifyDims = mRuntime.deployment.base.specVerifyDims(
        activeBatchSize, verifyLen, mRuntime.deployment.specConfig->verifyLayout);
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
        LOG_ERROR("DSparkDecoder: base verification execution failed.");
        return false;
    }

    check::check(mAcceptLength.reshape({activeBatchSize}), "Tensor reshape failed");
    int32_t const baseVocabSize = mRuntime.deployment.base.outputVocabSize;
    check::check(mRuntime.base.pipelineIO.outputLogits.reshape({activeBatchSize, verifyLen, baseVocabSize}),
        "Tensor reshape failed");
    // GCOVR_EXCL_START
    if (context.hasLogitBias)
    {
        applyLogitBiasRepeatedRows(
            mRuntime.logitBias, mRuntime.base.pipelineIO.outputLogits, context, verifyLen, context.stream);
    }
    // GCOVR_EXCL_STOP

    if (context.hasGuidedDecoding)
    {
        // One hook covers all three accept paths below: each of them reads outputLogits.
        applyGuidedDecodingMaskForDraftTree(mRuntime.guidedDecoder, context, mRuntime.base.pipelineIO.outputLogits,
            activeBatchSize, verifyLen, context.stream);
    }

    if (mUseTree)
    {
        NVTX_SCOPED_RANGE(nvtx_dspark_tree_accept, "DSparkDecoder::acceptAndCommitTree", nvtx_colors::LIGHT_BLUE);
        int32_t const maxAcceptLength = mJump.acceptDepth(std::min(mProposalLen + 1, verifyLen));
        check::check(mAcceptedTokenIds.reshape({activeBatchSize, maxAcceptLength}), "Tensor reshape failed");
        check::check(mAcceptedTokenIndices.reshape({activeBatchSize, maxAcceptLength}), "Tensor reshape failed");
        check::check(mRuntime.base.pipelineIO.outputLogits.reshape({activeBatchSize * verifyLen, baseVocabSize}),
            "Tensor reshape failed");
        Tensor const targetTokenIds = kernel::selectSpecTargetTokens(mRuntime.base.pipelineIO.outputLogits,
            std::nullopt, mRuntime.sampling.workspace, activeBatchSize, verifyLen, context.stream);
        SpecVerifyInputs const verify{mTreeTokenIds, verifyCounts, &mRuntime.base.pipelineIO.specTreeParentIds,
            &mRuntime.base.pipelineIO.specTreeDepths, activeBatchSize, verifyLen, maxAcceptLength,
            SpecVerifyLayout::kTree, true};
        spec_decode::accept(GreedyAcceptInputs{verify, targetTokenIds, verifyCounts, forcedFlags}, mAcceptedTokenIds,
            mAcceptLength, &mAcceptedTokenIndices, context.stream,
            decoder_utils::forcedAcceptanceLength(mRuntime, context));

        decoder_utils::clampAcceptLengthsToRemainingGeneration(context, mAcceptLength, context.stream);
        commitAcceptedTreePath(context, verifyLen, maxAcceptLength);
        // The hidden compaction leaves rows at stride maxAcceptLength, not verifyLen.
        decoder_utils::copyHiddenRowsStrided(mAcceptedHiddenStates, mAcceptCapacity,
            mRuntime.base.pipelineIO.baseHiddenStates, maxAcceptLength, activeBatchSize, mBaseOutputHiddenDim,
            context.stream);

        if (context.maxNumLogprobs() > 0)
        {
            int32_t const gatheredRows = activeBatchSize * maxAcceptLength;
            check::check(
                mRuntime.logprobs.gatheredLogits.reshape({gatheredRows, baseVocabSize}), "Tensor reshape failed");
            gatherSpecVerifyAcceptedLogitRows(mRuntime.base.pipelineIO.outputLogits, mAcceptedTokenIndices,
                mRuntime.logprobs.gatheredLogits, activeBatchSize, verifyLen, maxAcceptLength, baseVocabSize,
                context.stream);
            decoder_utils::enqueueLogprobsD2H(
                mRuntime.logprobs.gatheredLogits, gatheredRows, mRuntime, context.maxNumLogprobs(), context.stream);
        }

        std::optional<decoder_utils::ForcedPathInfo> forcedPath = mJump.forcedPath(mAcceptedTokenIndices);
        // Tree convention: every non-root node in the verify tree is a proposal, matching DFlash DDTree. A grafting
        // round proposes the plain round's drafts; its forced tokens are neither drafts nor bonus.
        decoder_utils::appendAcceptedTokens(context, mHostAcceptLengths, mHostAcceptedTokenIds, mAcceptLength,
            mAcceptedTokenIds, maxAcceptLength, mRuntime.tokenizer, context.stream, mVerifyLen - 1, nullptr,
            decoder_utils::forcedAcceptanceLength(mRuntime, context), false, forcedPath ? &*forcedPath : nullptr);

        if (context.hasGuidedDecoding)
        {
            advanceGuidedDecodingForCommitted(mRuntime.guidedDecoder, context,
                mHostAcceptedTokenIds.dataPointer<int32_t>(), mHostAcceptLengths.dataPointer<int32_t>(),
                maxAcceptLength, activeBatchSize);
        }
        if (context.maxNumLogprobs() > 0)
        {
            decoder_utils::collectSpecLogprobsFromHost(mRuntime, context, activeBatchSize, maxAcceptLength,
                mHostAcceptLengths.dataPointer<int32_t>(), context.maxNumLogprobs());
        }
        mJump.account(context, mHostAcceptLengths.dataPointer<int32_t>(), mHostAcceptedTokenIds.dataPointer<int32_t>(),
            maxAcceptLength);
        return true;
    }
    check::check(mAcceptedTokenIds.reshape({activeBatchSize, verifyLen}), "Tensor reshape failed");

    RowSamplingParams const sampling = context.uniformSamplingParams();
    bool const stochasticSampling = ::trt_edgellm::shouldUseNonGreedySampling(sampling);
    if (stochasticSampling)
    {
        bool const sparseTopK = dsparkCanUseTopKFastPath(sampling.topK, sampling.topP);
        int32_t const samplingTopK = sparseTopK ? static_cast<int32_t>(sampling.topK) : 0;
        if (sparseTopK)
        {
            int32_t const targetRows = activeBatchSize * verifyLen;
            check::check(
                mRuntime.base.pipelineIO.outputLogits.reshape({targetRows, baseVocabSize}), "Tensor reshape failed");
            check::check(mTargetTopKValues.reshape({targetRows, samplingTopK}), "Tensor reshape failed");
            check::check(mTargetTopKIndices.reshape({targetRows, samplingTopK}), "Tensor reshape failed");
            check::check(mTargetTopKProbabilities.reshape({targetRows, samplingTopK}), "Tensor reshape failed");
            selectAllTopK(mRuntime.base.pipelineIO.outputLogits, std::ref(mTargetTopKValues), mTargetTopKIndices,
                samplingTopK, mRuntime.sampling.workspace, context.stream);
            kernel::dsparkNormalizeTopKRows(mTargetTopKValues, mTargetTopKProbabilities, targetRows, samplingTopK,
                sampling.temperature, context.stream);
            check::check(
                mTargetTopKProbabilities.reshape({activeBatchSize, verifyLen, samplingTopK}), "Tensor reshape failed");
            check::check(
                mTargetTopKIndices.reshape({activeBatchSize, verifyLen, samplingTopK}), "Tensor reshape failed");
        }
        else
        {
            check::check(
                mTargetProbabilities.reshape({activeBatchSize, verifyLen, baseVocabSize}), "Tensor reshape failed");
            kernel::dsparkLogitsToProbabilities(mRuntime.base.pipelineIO.outputLogits, mTargetProbabilities,
                activeBatchSize * verifyLen, baseVocabSize, sampling.temperature, static_cast<int32_t>(sampling.topK),
                sampling.topP, context.stream);
        }

        int32_t const acceptUniformStride = 2 * mProposalLen + 1;
        check::check(mAcceptUniforms.reshape({activeBatchSize, acceptUniformStride}), "Tensor reshape failed");
        uint64_t const randomOffset
            = dsparkRandomRoundOffset(context.generationRound, mRuntime.maxRuntimeBatchSize, mProposalLen)
            + static_cast<uint64_t>(mRuntime.maxRuntimeBatchSize) * static_cast<uint64_t>(mProposalLen);
        kernel::dsparkFillUniforms(
            mAcceptUniforms, activeBatchSize * acceptUniformStride, kDSparkSamplingSeed, randomOffset, context.stream);
        if (sparseTopK)
        {
            spec_decode::accept(
                SparseProbabilityAcceptInputs{mTargetTopKProbabilities, mTargetTopKIndices, mDraftTopKProbabilities,
                    mDraftTopKIndices, mDraftTokenIds, mProposalLengths, mAcceptUniforms, nullptr},
                mAcceptedTokenIds, mAcceptLength, nullptr, context.stream,
                decoder_utils::forcedAcceptanceLength(mRuntime, context));
        }
        else
        {
            spec_decode::accept(DenseProbabilityAcceptInputs{mTargetProbabilities, mDraftProbabilities, mDraftTokenIds,
                                    mProposalLengths, mAcceptUniforms, activeBatchSize, mProposalLen,
                                    mCurrentProposalLen, baseVocabSize},
                mAcceptedTokenIds, mAcceptLength, context.stream,
                decoder_utils::forcedAcceptanceLength(mRuntime, context));
        }
    }
    else
    {
        check::check(mArgmaxScratch.reshape({activeBatchSize, verifyLen}), "Tensor reshape failed");
        kernel::dsparkSelectTargetTokens(mRuntime.base.pipelineIO.outputLogits, mArgmaxScratch, activeBatchSize,
            verifyLen, baseVocabSize, context.stream);
        SpecVerifyInputs const verify{mVerifyTokenIds, nullptr, nullptr, nullptr, activeBatchSize, verifyLen, verifyLen,
            SpecVerifyLayout::kChain, false};
        // A grafting round's accepted rows tell its forced positions apart.
        Tensor* const acceptedRows = mJump.wide() ? &mAcceptedTokenIndices : nullptr;
        if (acceptedRows != nullptr)
        {
            check::check(acceptedRows->reshape({activeBatchSize, verifyLen}), "Tensor reshape failed");
        }
        spec_decode::accept(GreedyAcceptInputs{verify, mArgmaxScratch,
                                mJump.wide() ? verifyCounts : &mAcceptValidTokenCounts, forcedFlags},
            mAcceptedTokenIds, mAcceptLength, acceptedRows, context.stream,
            decoder_utils::forcedAcceptanceLength(mRuntime, context));
    }

    decoder_utils::clampAcceptLengthsToRemainingGeneration(context, mAcceptLength, context.stream);
    spec_decode::commitAcceptedTargetPath(
        mRuntime, context, SpecAcceptResult{mAcceptedTokenIds, mAcceptLength, nullptr, verifyLen}, verifyLen);
    decoder_utils::copyHiddenRowsStrided(mAcceptedHiddenStates, mAcceptCapacity,
        mRuntime.base.pipelineIO.baseHiddenStates, verifyLen, activeBatchSize, mBaseOutputHiddenDim, context.stream);

    if (context.maxNumLogprobs() > 0)
    {
        check::check(mRuntime.base.pipelineIO.outputLogits.reshape({activeBatchSize * verifyLen, baseVocabSize}),
            "Tensor reshape failed");
        decoder_utils::enqueueLogprobsD2H(mRuntime.base.pipelineIO.outputLogits, activeBatchSize * verifyLen, mRuntime,
            context.maxNumLogprobs(), context.stream);
    }

    std::optional<decoder_utils::ForcedPathInfo> forcedPath = mJump.forcedPath(mAcceptedTokenIndices);
    decoder_utils::appendAcceptedTokens(context, mHostAcceptLengths, mHostAcceptedTokenIds, mAcceptLength,
        mAcceptedTokenIds, verifyLen, mRuntime.tokenizer, context.stream, mCurrentProposalLen,
        mScheduledProposalLengths.empty() ? nullptr : mScheduledProposalLengths.data(),
        decoder_utils::forcedAcceptanceLength(mRuntime, context), true, forcedPath ? &*forcedPath : nullptr);

    if (context.hasGuidedDecoding)
    {
        advanceGuidedDecodingForCommitted(mRuntime.guidedDecoder, context, mHostAcceptedTokenIds.dataPointer<int32_t>(),
            mHostAcceptLengths.dataPointer<int32_t>(), verifyLen, activeBatchSize);
    }
    if (context.maxNumLogprobs() > 0)
    {
        decoder_utils::collectSpecLogprobsFromHost(mRuntime, context, activeBatchSize, verifyLen,
            mHostAcceptLengths.dataPointer<int32_t>(), context.maxNumLogprobs());
    }
    mJump.account(
        context, mHostAcceptLengths.dataPointer<int32_t>(), mHostAcceptedTokenIds.dataPointer<int32_t>(), verifyLen);

    return true;
}

void DSparkDecoder::commitAcceptedTreePath(
    DecodingInferenceContext& context, int32_t verifyTokenCount, int32_t maxAcceptLength)
{
    NVTX_SCOPED_RANGE(nvtx_dspark_tree_commit, "DSparkDecoder::commitAcceptedTreePath", nvtx_colors::LIGHT_GREEN);
    spec_decode::commitAcceptedTargetPath(mRuntime, context,
        SpecAcceptResult{mAcceptedTokenIds, mAcceptLength, &mAcceptedTokenIndices, maxAcceptLength}, verifyTokenCount);
}

bool DSparkDecoder::captureCudaGraphs(cudaStream_t stream)
{
    bool draftProposalCaptureStatus{true};
    bool baseVerificationCaptureStatus{true};

    DecodingInferenceContext captureContext;
    captureContext.stream = stream;
    captureContext.residentRefs.resize(mRuntime.maxRuntimeBatchSize);
    for (int32_t slot = 0; slot < mRuntime.maxRuntimeBatchSize; ++slot)
    {
        captureContext.residentRefs[slot].slot = slot;
    }

    static constexpr int32_t kSimulateCacheLength{128};
    int32_t const proposalLen = mProposalLen;
    int32_t const verifyLen = mVerifyLen;
    int32_t const draftKVCapacity = mRuntime.deployment.draft->maxKVCacheCapacity;

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

    for (int32_t batchSize = 1; batchSize <= mRuntime.maxRuntimeBatchSize; ++batchSize)
    {
        std::vector<int32_t> simCacheLens(batchSize, kSimulateCacheLength);
        Tensor simCacheLensTensor(simCacheLens.data(), {batchSize}, DeviceType::kCPU, trt_edgellm::DataType::kINT32);
        mRuntime.base.cacheManager.resetForNewSequences(simCacheLensTensor, stream);
        mDraftCacheManager.resetForNewSequences(simCacheLensTensor, stream);

        int32_t const blockLen = mDraftBlockLen;
        int32_t const draftPmLen = divUp(blockLen, 32);
        // Delta = last round's accept length; capped by the draft profile (block + 1),
        // which tree-mode verifyLen (node budget) can exceed. A grafting round may accept up to jumpMaxPathLength.
        int32_t const maxSimDeltaLen
            = std::max(std::min(proposalLen + 1, verifyLen), mRuntime.deployment.specConfig->jumpMaxPathLength);
        for (int32_t simDeltaLen = 1; simDeltaLen <= maxSimDeltaLen; ++simDeltaLen)
        {
            check::check(mDraftInputsEmbeds.reshape({batchSize, blockLen, mDraftHiddenSize}), "Tensor reshape failed");
            check::check(
                mDraftTargetHidden.reshape({batchSize, static_cast<int64_t>(simDeltaLen), mBaseOutputHiddenDim}),
                "Tensor reshape failed");
            check::check(mDraftOutputLogits.reshape({batchSize, blockLen, mDraftVocabSize}), "Tensor reshape failed");
            check::check(mDraftHiddenStates.reshape({batchSize, blockLen, mDraftHiddenSize}), "Tensor reshape failed");
            check::check(mDraftPackedAttentionMask.reshape({batchSize, blockLen, draftPmLen}), "Tensor reshape failed");
            check::check(mDraftAttentionPosId.reshape({batchSize, blockLen}), "Tensor reshape failed");
            check::check(mDraftContextLengths.reshape({batchSize}), "Tensor reshape failed");

            std::vector<int32_t> simDeltaLens(batchSize, simDeltaLen);
            check::check(mDraftDeltaLens.reshape({batchSize}), "Tensor reshape failed");
            CUDA_CHECK(cudaMemcpyAsync(mDraftDeltaLens.rawPointer(), simDeltaLens.data(), batchSize * sizeof(int32_t),
                cudaMemcpyHostToDevice, stream));

            Tensor const& draftCacheLengths = mDraftCacheManager.getKVCacheLengths();
            kernel::launchDFlashPrepareProposalInputs(draftCacheLengths.dataPointer<int32_t>(),
                mDraftDeltaLens.dataPointer<int32_t>(), blockLen, mDraftPackedAttentionMask.dataPointer<int32_t>(),
                mDraftAttentionPosId.dataPointer<int32_t>(), mDraftContextLengths.dataPointer<int32_t>(),
                mRuntime.base.pipelineIO.positions.dataPointer<int32_t>(),
                mRuntime.base.pipelineIO.queryStartOffsets.dataPointer<int32_t>(),
                mRuntime.base.pipelineIO.queryLengths.dataPointer<int32_t>(),
                mRuntime.base.pipelineIO.pastLengths.dataPointer<int32_t>(),
                mRuntime.base.pipelineIO.attentionSequenceLengths.dataPointer<int32_t>(),
                mRuntime.base.pipelineIO.stateIndices.dataPointer<int32_t>(), mCausalProposalMask, batchSize, stream);
            prepareDraftRaggedBindings(batchSize, blockLen, simDeltaLen, stream);

            InferenceDims const draftDims{
                /*.batch=*/batchSize,
                /*.seqLen=*/batchSize * blockLen,
                /*.kvLen=*/draftKVCapacity,
                /*.selectLen=*/static_cast<int64_t>(batchSize) * simDeltaLen,
                /*.attnMaskSeqLen=*/batchSize * blockLen,
                /*.ropeBatch=*/1,
                /*.packedMaskLen=*/static_cast<int64_t>(draftPmLen),
                /*.contextMaskSelectorLen=*/0,
                /*.startIndexLen=*/batchSize,
                /*.executionPhaseLen=*/static_cast<int64_t>(ExecutionPhase::kSpecDraftProposal),
                /*.specVerifyLayoutLen=*/0,
                /*.skipSoftmaxPrefillScaleLen=*/0,
                /*.skipSoftmaxDecodeScaleLen=*/0,
                /*.swaKVCacheModeLen=*/0,
                /*.queryOffsetLen=*/batchSize + 1,
                /*.contextSequenceCount=*/0,
            };

            if (mDraftExecutor->prepare(kDecodeProfile, draftDims, mDraftTensorMap, stream))
            {
                draftProposalCaptureStatus &= mDraftExecutor->captureGraph(stream);
            }
            else
            {
                LOG_WARNING(
                    "DSpark: failed to prepare draft for graph capture (batch=%d, delta=%d)", batchSize, simDeltaLen);
                draftProposalCaptureStatus = false;
            }
        }

        // Tree always verifies the full node budget; chain replays every window.
        int32_t const minSimVerifyLen = mUseTree ? verifyLen : 2;
        for (int32_t simVerifyLen = minSimVerifyLen; simVerifyLen <= verifyLen; ++simVerifyLen)
        {
            if (mUseTree)
            {
                CUDA_CHECK(cudaMemsetAsync(
                    mStackedMarkovLogits.rawPointer(), 0, mStackedMarkovLogits.getMemoryCapacity(), stream));
                buildTreeVerifyInputs(batchSize, stream, /*useConfidence=*/false);
            }
            else
            {
                check::check(mVerifyTokenIds.reshape({batchSize, simVerifyLen}), "Tensor reshape failed");
                CUDA_CHECK(cudaMemsetAsync(
                    mVerifyTokenIds.rawPointer(), 0, batchSize * simVerifyLen * sizeof(int32_t), stream));
            }
            captureContext.activeBatchSize = batchSize;
            SpecVerifyInputs const verify{mUseTree ? mTreeTokenIds : mVerifyTokenIds,
                mUseTree ? &mValidCounts : nullptr, mUseTree ? &mRuntime.base.pipelineIO.specTreeParentIds : nullptr,
                mUseTree ? &mRuntime.base.pipelineIO.specTreeDepths : nullptr, batchSize, simVerifyLen,
                std::min(simVerifyLen, mRuntime.deployment.specConfig->maxPathLength),
                mRuntime.deployment.specConfig->verifyLayout, mUseTree};
            spec_decode::prepareTargetVerifyInputs(verify, mRuntime, captureContext, 0);
            auto const verifyDims = mRuntime.deployment.base.specVerifyDims(batchSize, simVerifyLen, verify.layout);
            baseVerificationCaptureStatus &= mRuntime.base.captureGraph(verifyDims, stream);
        }

        // A tree that can jump also verifies grafting rounds of jumpVerifyTokenCount rows.
        int32_t const jumpVerifyLen = mRuntime.deployment.specConfig->jumpVerifyTokenCount;
        if (mUseTree && jumpVerifyLen > verifyLen)
        {
            auto& io = mRuntime.base.pipelineIO;
            check::check(mTreeTokenIds.reshape({batchSize, jumpVerifyLen}), "Tensor reshape failed");
            CUDA_CHECK(cudaMemsetAsync(mTreeTokenIds.rawPointer(), 0,
                static_cast<size_t>(batchSize) * jumpVerifyLen * sizeof(int32_t), stream));
            Tensor& mask = mJump.grafter().mask();
            check::check(mask.reshape({batchSize, jumpVerifyLen, jumpVerifyLen}), "Tensor reshape failed");
            CUDA_CHECK(cudaMemsetAsync(
                mask.rawPointer(), 0, static_cast<size_t>(batchSize) * jumpVerifyLen * jumpVerifyLen, stream));
            check::check(io.specTreeParentIds.reshape({batchSize, jumpVerifyLen}), "Tensor reshape failed");
            check::check(io.specTreeDepths.reshape({batchSize, jumpVerifyLen}), "Tensor reshape failed");
            mJump.prepareGraftedTreeDescriptors(
                mRuntime.base.pipelineIO, mRuntime.base.cacheManager.getKVCacheLengths(), batchSize, stream);
            captureContext.activeBatchSize = batchSize;
            SpecVerifyInputs const verify{mTreeTokenIds, nullptr, &io.specTreeParentIds, &io.specTreeDepths, batchSize,
                jumpVerifyLen, std::min(jumpVerifyLen, mRuntime.deployment.specConfig->maxPathLength),
                mRuntime.deployment.specConfig->verifyLayout, true};
            spec_decode::prepareTargetVerifyInputs(verify, mRuntime, captureContext, 0);
            auto const verifyDims = mRuntime.deployment.base.specVerifyDims(batchSize, jumpVerifyLen, verify.layout);
            baseVerificationCaptureStatus &= mRuntime.base.captureGraph(verifyDims, stream);
        }
    }

    LOG_INFO("DSparkDecoder: CUDA graph capture complete (draft=%s, baseVerify=%s)",
        draftProposalCaptureStatus ? "ok" : "FAILED", baseVerificationCaptureStatus ? "ok" : "FAILED");

    return draftProposalCaptureStatus && baseVerificationCaptureStatus;
}

int64_t DSparkDecoder::getRequiredContextMemorySize() const noexcept
{
    return mDraftExecutor ? mDraftExecutor->getRequiredContextMemorySize() : 0;
}

void DSparkDecoder::setContextMemory(Tensor& memory)
{
    if (mDraftExecutor)
    {
        mDraftExecutor->setContextMemory(memory);
    }
}

void DSparkDecoder::resetForNewSequences(Tensor& reuseLengths, cudaStream_t stream)
{
    mDraftCacheManager.resetForNewSequences(reuseLengths, stream);
    mCommonStateTracker.reset();
    mDraftPrimed = false;
}

void DSparkDecoder::onBatchEvict(std::vector<int32_t> const& batchMapping, int32_t oldActiveBatch,
    int32_t newActiveBatch, Tensor& deviceBatchMapping, cudaStream_t stream)
{
    ELLM_CHECK(batchMapping.size() == static_cast<size_t>(oldActiveBatch),
        "DSpark batch mapping does not match the old active batch");
    mCommonStateTracker.compact(batchMapping, oldActiveBatch, newActiveBatch);
    if (newActiveBatch == 0)
    {
        mDraftPrimed = false;
        return;
    }

    auto compactHostTensor = [&](Tensor& tensor) {
        if (tensor.isEmpty() || tensor.getShape().getNumDims() == 0 || tensor.getShape()[0] != oldActiveBatch)
        {
            return;
        }
        std::vector<int32_t> compacted(static_cast<size_t>(newActiveBatch));
        int32_t const* source = tensor.dataPointer<int32_t>();
        for (int32_t oldSlot = 0; oldSlot < oldActiveBatch; ++oldSlot)
        {
            int32_t const newSlot = batchMapping[static_cast<size_t>(oldSlot)];
            if (newSlot >= 0)
            {
                compacted[static_cast<size_t>(newSlot)] = source[oldSlot];
            }
        }
        std::copy(compacted.begin(), compacted.end(), tensor.dataPointer<int32_t>());
        check::check(tensor.reshape({newActiveBatch}), "Tensor reshape failed");
    };
    compactHostTensor(mHostAcceptLengths);

    auto compactDeviceTensor = [&](Tensor& tensor) {
        if (tensor.isEmpty() || newActiveBatch == 0)
        {
            return;
        }
        kernel::compactExecutionTensorBatch(tensor, deviceBatchMapping, oldActiveBatch, newActiveBatch, stream);
    };
    if (mDraftPrimed)
    {
        // Between rounds the carried span is the decoder's own copy; the pipeline buffer is scratch.
        check::check(mAcceptedHiddenStates.reshape(
                         {static_cast<int64_t>(oldActiveBatch) * mAcceptCapacity, mBaseOutputHiddenDim}),
            "Tensor reshape failed");
        compactDeviceTensor(mAcceptedHiddenStates);
    }
    else
    {
        // Before the first draft forward (the blocking path's round zero), the pipeline buffer holds the target's
        // prefill hidden states that forward still has to read.
        compactDeviceTensor(mRuntime.base.pipelineIO.baseHiddenStates);
    }
    if (mCommonStateTracker.draftPrefillOutputsPending())
    {
        // Only what the pending proposal's draft forward wrote holds this batch's rows; other buffers keep an earlier
        // batch's shape.
        auto compactPendingTensor = [&](Tensor& tensor) {
            ELLM_CHECK(tensor.getShape().getNumDims() > 0 && tensor.getShape()[0] == oldActiveBatch,
                "DSpark pending proposal tensor " + tensor.getName() + " does not hold the batch being compacted");
            kernel::compactExecutionTensorBatch(tensor, deviceBatchMapping, oldActiveBatch, newActiveBatch, stream);
        };
        compactPendingTensor(mDraftOutputLogits);
        compactPendingTensor(mDraftHiddenStates);
        compactPendingTensor(mDraftTokenIds);
        compactPendingTensor(mProposalLengths);
        compactPendingTensor(mLastAcceptedTokens);
        compactPendingTensor(mRuntime.preprocess.idsInput);
        if (mSchedulerMode != DSparkSchedulerMode::kOff)
        {
            compactPendingTensor(mConfidenceScores);
        }
        switch (mDraftDistribution)
        {
        case DraftDistribution::kNone: break;
        case DraftDistribution::kDense: compactPendingTensor(mDraftProbabilities); break;
        case DraftDistribution::kTopK:
            compactPendingTensor(mDraftTopKProbabilities);
            compactPendingTensor(mDraftTopKIndices);
            break;
        }
        if (mUseTree)
        {
            // The tree build also writes the target's verification inputs; a chain's are written at verification.
            compactPendingTensor(mStackedMarkovLogits);
            compactPendingTensor(mTreeTokenIds);
            compactPendingTensor(mRuntime.base.pipelineIO.specTreeDepths);
            compactPendingTensor(mRuntime.base.pipelineIO.specTreeParentIds);
            compactPendingTensor(mTreeNodeScores);
            compactPendingTensor(mValidCounts);
            compactPendingTensor(mRuntime.base.pipelineIO.specDecodePositionIds);
            compactPendingTensor(mRuntime.base.pipelineIO.packedAttentionMask);
            compactPendingTensor(mRuntime.base.pipelineIO.contextLengths);
            kernel::compactExecutionRowIndices(mRuntime.base.pipelineIO.selectTokenIndices, deviceBatchMapping,
                oldActiveBatch, newActiveBatch, mCurrentVerifyLen, stream);
        }
        else
        {
            compactPendingTensor(mVerifyTokenIds);
            compactPendingTensor(mAcceptValidTokenCounts);
            if (!mScheduledProposalLengths.empty())
            {
                compactVector(batchMapping, mScheduledProposalLengths);
            }
        }
    }
    else
    {
        // Acceptance lengths are first written by base verification, after the pending-prefill proposal is consumed.
        compactDeviceTensor(mAcceptLength);
    }
}

} // namespace rt
} // namespace trt_edgellm
