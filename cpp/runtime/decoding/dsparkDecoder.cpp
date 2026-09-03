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
#include "kernels/embeddingKernels/embeddingKernels.h"
#include "kernels/gdnKernels/gdnTreeChunkKernels.h"
#include "kernels/speculative/ddtreeKernels.h"
#include "kernels/speculative/dflashRuntimeKernels.h"
#include "kernels/speculative/dsparkKernels.h"
#include "kernels/speculative/eagleAcceptKernels.h"
#include "kernels/speculative/eagleUtilKernels.h"
#include "profiling/metrics.h"
#include "profiling/nvtx_wrapper.h"
#include "profiling/timer.h"
#include "runtime/config/llmEngineConfig.h"
#include "runtime/decoding/decoderUtils.h"
#include "runtime/decoding/logitBias.h"
#include "sampler/sampling.h"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <functional>
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

int32_t dsparkDraftProfileForRound(int32_t generationRound)
{
    return generationRound == 0 ? kPrefillProfile : kDecodeProfile;
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
    check::check(tensor.getDataType() == nvinfer1::DataType::kHALF, name + " must be FP16");
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
    SpecDecodeDraftingConfig const& draftingConfig, std::unique_ptr<EngineExecutor> draftExecutor,
    ExternalWeightManager draftWeights, cudaStream_t stream)
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

    mUseTree = draftingConfig.draftingTopK > 1;
    mVerifyLen = deployment.specConfig->verifySize;
    // Tree mode drafts the full block; verifySize is the DDTree node budget, not blockSize + 1.
    mProposalLen = mUseTree ? draftCfg.specDraftBlockSize : mVerifyLen - 1;
    mCurrentProposalLen = mProposalLen;
    mCurrentVerifyLen = mVerifyLen;
    ELLM_CHECK(mProposalLen > 0, "DSparkDecoder requires verifySize >= 2.");
    ELLM_CHECK(
        mProposalLen <= draftCfg.specDraftBlockSize, "DSparkDecoder proposal length exceeds dspark_config.block_size.");
    mMaskTokenId = draftCfg.specDraftMaskTokenId > 0 ? draftCfg.specDraftMaskTokenId : baseCfg.specDraftMaskTokenId;
    mDraftHiddenSize = deployment.specConfig->draftHiddenSize;
    mBaseOutputHiddenDim = deployment.specConfig->baseOutputHiddenDim;
    mDraftVocabSize = draftCfg.outputVocabSize;
    ELLM_CHECK(mMaskTokenId >= 0 && mMaskTokenId < mDraftVocabSize,
        "DSparkDecoder: mask token id (" + std::to_string(mMaskTokenId) + ") out of draft vocab range [0, "
            + std::to_string(mDraftVocabSize) + ").");
    ELLM_CHECK(baseCfg.outputVocabSize == mDraftVocabSize,
        "DSparkDecoder requires base and draft output vocab sizes to match for exact stochastic verification.");
    mMarkovRank = draftCfg.dsparkMarkovRank;
    mHasConfidenceHead = draftCfg.dsparkEnableConfidenceHead;
    mConfidenceHeadWithMarkov = draftCfg.dsparkConfidenceHeadWithMarkov;
    mSchedulerMode = draftingConfig.dsparkSchedulerMode;
    mConfidenceThreshold = draftingConfig.dsparkConfidenceThreshold;
    mMinScheduledProposalLen = std::max(1, draftingConfig.dsparkMinProposalLen);
    mMaxScheduledProposalLen = draftingConfig.dsparkMaxProposalLen <= 0
        ? mProposalLen
        : std::min(mProposalLen, draftingConfig.dsparkMaxProposalLen);
    ELLM_CHECK(mMinScheduledProposalLen <= mMaxScheduledProposalLen,
        "DSpark scheduler min proposal length exceeds max proposal length.");
    ELLM_CHECK(mSchedulerMode == DSparkSchedulerMode::kOff || mHasConfidenceHead,
        "DSpark scheduler requires confidence head sidecar tensors.");

    int32_t const maxBatch = deployment.maxRuntimeBatchSize();
    int32_t const maxSeqForDraft = baseCfg.maxKVCacheCapacity;

    ELLM_CHECK(mDraftExecutor != nullptr, "DSpark decoding requires a validated draft engine.");

    mDraftInputsEmbeds = Tensor({maxBatch, mProposalLen, mDraftHiddenSize}, DeviceType::kGPU, nvinfer1::DataType::kHALF,
        "DSpark::draftInputsEmbeds");
    mDraftTargetHidden = Tensor({maxBatch, maxSeqForDraft, mBaseOutputHiddenDim}, DeviceType::kGPU,
        nvinfer1::DataType::kHALF, "DSpark::draftTargetHidden");
    mDraftOutputLogits = Tensor({maxBatch, mProposalLen, mDraftVocabSize}, DeviceType::kGPU, nvinfer1::DataType::kFLOAT,
        "DSpark::draftOutputLogits");
    mDraftHiddenStates = Tensor({maxBatch, mProposalLen, mDraftHiddenSize}, DeviceType::kGPU, nvinfer1::DataType::kHALF,
        "DSpark::draftHiddenStates");

    int32_t const packedMaskLen = divUp(mProposalLen, 32);
    mDraftPackedAttentionMask = Tensor({maxBatch, mProposalLen, packedMaskLen}, DeviceType::kGPU,
        nvinfer1::DataType::kINT32, "DSpark::draftPackedMask");
    mDraftAttentionPosId
        = Tensor({maxBatch, mProposalLen}, DeviceType::kGPU, nvinfer1::DataType::kINT32, "DSpark::draftAttentionPosId");
    mDraftContextLengths
        = Tensor({maxBatch}, DeviceType::kGPU, nvinfer1::DataType::kINT32, "DSpark::draftContextLengths");
    mDraftDeltaLenCommit
        = Tensor({maxBatch}, DeviceType::kGPU, nvinfer1::DataType::kINT32, "DSpark::draftDeltaLenCommit");
    mDraftDeltaLens = Tensor({maxBatch}, DeviceType::kGPU, nvinfer1::DataType::kINT32, "DSpark::draftDeltaLens");

    mDraftTensorMap.set(binding_names::kInputsEmbeds, mDraftInputsEmbeds);
    mDraftTensorMap.set(binding_names::kDFlashTargetHiddenConcat, mDraftTargetHidden);
    mDraftTensorMap.set(binding_names::kLogits, mDraftOutputLogits);
    mDraftTensorMap.set(binding_names::kDSparkHiddenStates, mDraftHiddenStates);
    mDraftTensorMap.set(binding_names::kAttentionMask, mDraftPackedAttentionMask);
    mDraftTensorMap.set(binding_names::kAttentionPosId, mDraftAttentionPosId);
    mDraftTensorMap.set(binding_names::kContextLengths, mDraftContextLengths);
    mDraftTensorMap.set(binding_names::kDFlashDeltaLengths, mDraftDeltaLens);

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

    mDraftTensorMap.set(binding_names::kKVCacheStartIndex, mDraftCacheManager.getKVCacheLengths());

    // The draft target update and proposal attention share the managed draft page table.
    mDraftTensorMap.set(binding_names::kKVPageTable, mRuntime.base.sharedResources.kvPageTables[1]->kernelView());

    if (draftCfg.ropeConfig.type == RopeType::kMRope)
    {
        mDraftTensorMap.set(binding_names::kRopeCosSin, mRuntime.base.pipelineIO.mropeCosSin);
    }
    else
    {
        mDraftTensorMap.set(binding_names::kRopeCosSin,
            mRuntime.base.sharedResources.ropePool.getOrCreate(
                draftCfg.ropeConfig, draftCfg.rotaryDim, baseCfg.maxKVCacheCapacity, nullptr));
    }

    mDraftExternalWeightManager = std::move(draftWeights);
    mDraftExternalWeightManager.registerTensorMapEntries(mDraftTensorMap);

    mDraftTokenIds
        = Tensor({maxBatch, mProposalLen}, DeviceType::kGPU, nvinfer1::DataType::kINT32, "DSpark::draftTokenIds");
    mVerifyTokenIds
        = Tensor({maxBatch, mVerifyLen}, DeviceType::kGPU, nvinfer1::DataType::kINT32, "DSpark::verifyTokenIds");
    mAcceptedTokenIds
        = Tensor({maxBatch, mVerifyLen}, DeviceType::kGPU, nvinfer1::DataType::kINT32, "DSpark::acceptedIds");
    mAcceptLength = Tensor({maxBatch}, DeviceType::kGPU, nvinfer1::DataType::kINT32, "DSpark::acceptLength");
    mHostAcceptLengths = Tensor({maxBatch}, DeviceType::kCPU, nvinfer1::DataType::kINT32, "DSpark::hostAcceptLengths");
    mHostAcceptedTokenIds
        = Tensor({maxBatch, mVerifyLen}, DeviceType::kCPU, nvinfer1::DataType::kINT32, "DSpark::hostAcceptedIds");
    mHostDraftInputIds
        = Tensor({maxBatch, mProposalLen}, DeviceType::kCPU, nvinfer1::DataType::kINT32, "DSpark::hostDraftInputIds");
    mHostLastAcceptedTokens
        = Tensor({maxBatch}, DeviceType::kCPU, nvinfer1::DataType::kINT32, "DSpark::hostLastAcceptedTokens");
    mHostDeltaLens = Tensor({maxBatch}, DeviceType::kCPU, nvinfer1::DataType::kINT32, "DSpark::hostDeltaLens");
    mHostProposalLengths
        = Tensor({maxBatch}, DeviceType::kCPU, nvinfer1::DataType::kINT32, "DSpark::hostProposalLengths");
    mDraftProbabilities = Tensor({maxBatch, mProposalLen, mDraftVocabSize}, DeviceType::kGPU,
        nvinfer1::DataType::kFLOAT, "DSpark::draftProbabilities");
    mDraftStepLogits
        = Tensor({maxBatch, mDraftVocabSize}, DeviceType::kGPU, nvinfer1::DataType::kFLOAT, "DSpark::stepLogits");
    mDraftStepProbabilities = Tensor(
        {maxBatch, mDraftVocabSize}, DeviceType::kGPU, nvinfer1::DataType::kFLOAT, "DSpark::stepProbabilities");
    mDraftStepTopKValues = Tensor(
        {maxBatch, kDSparkMaxSparseTopK}, DeviceType::kGPU, nvinfer1::DataType::kFLOAT, "DSpark::stepTopKValues");
    mDraftStepTopKIndices = Tensor(
        {maxBatch, kDSparkMaxSparseTopK}, DeviceType::kGPU, nvinfer1::DataType::kINT32, "DSpark::stepTopKIndices");
    mDraftTopKProbabilities = Tensor({maxBatch, mProposalLen, kDSparkMaxSparseTopK}, DeviceType::kGPU,
        nvinfer1::DataType::kFLOAT, "DSpark::draftTopKProbabilities");
    mDraftTopKIndices = Tensor({maxBatch, mProposalLen, kDSparkMaxSparseTopK}, DeviceType::kGPU,
        nvinfer1::DataType::kINT32, "DSpark::draftTopKIndices");
    mTargetProbabilities = Tensor({maxBatch, mVerifyLen, baseCfg.outputVocabSize}, DeviceType::kGPU,
        nvinfer1::DataType::kFLOAT, "DSpark::targetProbabilities");
    mTargetTopKValues = Tensor({maxBatch * mVerifyLen, kDSparkMaxSparseTopK}, DeviceType::kGPU,
        nvinfer1::DataType::kFLOAT, "DSpark::targetTopKValues");
    mTargetTopKIndices = Tensor({maxBatch * mVerifyLen, kDSparkMaxSparseTopK}, DeviceType::kGPU,
        nvinfer1::DataType::kINT32, "DSpark::targetTopKIndices");
    mTargetTopKProbabilities = Tensor({maxBatch * mVerifyLen, kDSparkMaxSparseTopK}, DeviceType::kGPU,
        nvinfer1::DataType::kFLOAT, "DSpark::targetTopKProbabilities");
    mDraftUniforms
        = Tensor({maxBatch, mProposalLen}, DeviceType::kGPU, nvinfer1::DataType::kFLOAT, "DSpark::draftUniforms");
    mAcceptUniforms = Tensor(
        {maxBatch, 2 * mProposalLen + 1}, DeviceType::kGPU, nvinfer1::DataType::kFLOAT, "DSpark::acceptUniforms");
    mConfidenceScores
        = Tensor({maxBatch, mProposalLen}, DeviceType::kGPU, nvinfer1::DataType::kFLOAT, "DSpark::confidenceScores");
    mProposalLengths = Tensor({maxBatch}, DeviceType::kGPU, nvinfer1::DataType::kINT32, "DSpark::proposalLengths");
    mArgmaxScratch
        = Tensor({maxBatch * mVerifyLen}, DeviceType::kGPU, nvinfer1::DataType::kINT32, "DSpark::argmaxScratch");
    mLastAcceptedTokens
        = Tensor({maxBatch}, DeviceType::kGPU, nvinfer1::DataType::kINT32, "DSpark::lastAcceptedTokens");

    if (mUseTree)
    {
        // Tree proposal rows: row 0 is the root placeholder (ignored by the builder),
        // rows 1..blockSize hold the per-depth Markov-corrected candidate logits.
        int32_t const proposalDepthSize = mProposalLen + 1;
        mStackedMarkovLogits = Tensor({maxBatch, proposalDepthSize, mDraftVocabSize}, DeviceType::kGPU,
            nvinfer1::DataType::kFLOAT, "DSpark::stackedMarkovLogits");
        mTreeTokenIds
            = Tensor({maxBatch, mVerifyLen}, DeviceType::kGPU, nvinfer1::DataType::kINT32, "DSpark::treeTokenIds");
        mTreeNodeDepths
            = Tensor({maxBatch, mVerifyLen}, DeviceType::kGPU, nvinfer1::DataType::kINT32, "DSpark::treeNodeDepths");
        mTreeParentIds
            = Tensor({maxBatch, mVerifyLen}, DeviceType::kGPU, nvinfer1::DataType::kINT32, "DSpark::treeParentIds");
        mTreeNodeScores
            = Tensor({maxBatch, mVerifyLen}, DeviceType::kGPU, nvinfer1::DataType::kFLOAT, "DSpark::treeNodeScores");
        mValidCounts = Tensor({maxBatch}, DeviceType::kGPU, nvinfer1::DataType::kINT32, "DSpark::validCounts");
        mVerifyTreeMask = Tensor(
            {maxBatch, mVerifyLen, mVerifyLen}, DeviceType::kGPU, nvinfer1::DataType::kINT8, "DSpark::verifyTreeMask");
        mAcceptedTokenIndices = Tensor(
            {maxBatch, mVerifyLen}, DeviceType::kGPU, nvinfer1::DataType::kINT32, "DSpark::acceptedTokenIndices");
        size_t const buildWorkspaceSize = kernel::getDDTreeBuildWorkspaceSize(
            maxBatch, proposalDepthSize, mVerifyLen, mDraftVocabSize, draftingConfig.draftingTopK);
        ELLM_CHECK(buildWorkspaceSize > 0, "DSparkDecoder: invalid tree-build workspace size.");
        mTreeBuildWorkspace = Tensor({static_cast<int64_t>(buildWorkspaceSize)}, DeviceType::kGPU,
            nvinfer1::DataType::kUINT8, "DSpark::treeBuildWorkspace");
        // Zeroed so the capture-time tree build (before any real drafting) is deterministic.
        CUDA_CHECK(
            cudaMemsetAsync(mStackedMarkovLogits.rawPointer(), 0, mStackedMarkovLogits.getMemoryCapacity(), stream));
        CUDA_CHECK(
            cudaMemsetAsync(mLastAcceptedTokens.rawPointer(), 0, mLastAcceptedTokens.getMemoryCapacity(), stream));

        // Tree scheduling = log(conf) growth bias + threshold as a survival floor (0 = bias only).
        mUseTreeScheduler = mSchedulerMode != DSparkSchedulerMode::kOff && mHasConfidenceHead;
    }

    loadHeadSidecars(engineDir, stream);

    LOG_INFO(
        "DSparkDecoder initialized: proposalLen=%d verifyLen=%d maskTokenId=%d maxBatch=%d "
        "draftHiddenSize=%d baseOutputHiddenDim=%d draftVocabSize=%d markovRank=%d confidence=%s",
        mProposalLen, mVerifyLen, mMaskTokenId, maxBatch, mDraftHiddenSize, mBaseOutputHiddenDim, mDraftVocabSize,
        mMarkovRank, mHasConfidenceHead ? "true" : "false");
}

DecodingKvHeadroom DSparkDecoder::requiredKvHeadroom() const
{
    return {mVerifyLen, mProposalLen};
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
    mMarkovW2 = takeRequiredTensor(tensors, "markov_w2");
    validateMatrix(mMarkovW1, "markov_w1", mDraftVocabSize, mMarkovRank);
    validateMatrix(mMarkovW2, "markov_w2", mDraftVocabSize, mMarkovRank);

    if (mHasConfidenceHead)
    {
        mConfidenceWeight = takeRequiredTensor(tensors, "confidence_weight");
        mConfidenceBias = takeRequiredTensor(tensors, "confidence_bias");
        check::check(mConfidenceWeight.getDataType() == nvinfer1::DataType::kHALF, "confidence_weight must be FP16");
        check::check(mConfidenceBias.getDataType() == nvinfer1::DataType::kHALF, "confidence_bias must be FP16");
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

bool DSparkDecoder::decodeStep(DecodingInferenceContext& context)
{
    NVTX_SCOPED_RANGE(nvtx_dspark_decode, "DSparkDecoder::decodeStep", nvtx_colors::GREEN);
    cudaGetLastError();

    bool const usePendingPrefillProposal = mCommonStateTracker.shouldUsePendingPrefillProposal(context.generationRound);
    if (!usePendingPrefillProposal && !runDraftForward(context))
    {
        LOG_ERROR("DSparkDecoder: draft forward failed.");
        return false;
    }
    mCommonStateTracker.materializePending(context.generationRound, context.activeBatchSize);
    mCommonStateTracker.consumeDraftPrefillOutputs();

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
    mCommonStateTracker.initialize(context);

    if (!runDraftForward(context))
    {
        LOG_ERROR("DSparkDecoder: failed to initialize draft state for generation.");
        return false;
    }
    mCommonStateTracker.markDraftPrefillOutputsPending();
    return true;
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
            mDraftStepLogits, activeBatchSize, step, proposalLen, mDraftVocabSize, mMarkovRank, context.stream);
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
    check::check(mDraftStepLogits.reshape({activeBatchSize, mDraftVocabSize}), "Tensor reshape failed");
    check::check(mDraftStepProbabilities.reshape({activeBatchSize, mDraftVocabSize}), "Tensor reshape failed");
    for (int32_t step = 0; step < proposalLen; ++step)
    {
        kernel::dsparkBuildMarkovLogits(mDraftOutputLogits, mMarkovW1, mMarkovW2, mLastAcceptedTokens, mDraftTokenIds,
            mDraftStepLogits, activeBatchSize, step, proposalLen, mDraftVocabSize, mMarkovRank, context.stream);
        applyLogitBias(mRuntime.logitBias, mDraftStepLogits, context, context.stream);
        kernel::dsparkLogitsToProbabilities(mDraftStepLogits, mDraftStepProbabilities, activeBatchSize, mDraftVocabSize,
            context.temperature, static_cast<int32_t>(context.topK), context.topP, context.stream);
        kernel::dsparkSampleProbabilityRows(mDraftStepProbabilities, mDraftUniforms, mDraftTokenIds, activeBatchSize,
            step, proposalLen, mDraftVocabSize, context.stream);
        kernel::dsparkStoreDraftStepProbabilities(mDraftStepProbabilities, mDraftProbabilities, activeBatchSize, step,
            proposalLen, mDraftVocabSize, context.stream);
    }
}
// GCOVR_EXCL_STOP

bool DSparkDecoder::runDraftForward(DecodingInferenceContext& context)
{
    TIME_STAGE(metrics::StageNames::kSPEC_DECODE_DRAFT_PROPOSAL, context.stream);
    NVTX_SCOPED_RANGE(nvtx_dspark_draft, "DSparkDecoder::runDraftForward", nvtx_colors::DARK_ORANGE);

    if (!mDraftExecutor)
    {
        LOG_ERROR("DSparkDecoder: draft engine not loaded.");
        return false;
    }

    int32_t const activeBatchSize = context.activeBatchSize;
    int32_t const proposalLen = mProposalLen;

    check::check(mRuntime.preprocess.idsInput.reshape({activeBatchSize, proposalLen}), "Tensor reshape failed");
    check::check(mHostDraftInputIds.reshape({activeBatchSize, proposalLen}), "Tensor reshape failed");
    check::check(mHostLastAcceptedTokens.reshape({activeBatchSize}), "Tensor reshape failed");
    int32_t* hostDraftInputIds = mHostDraftInputIds.dataPointer<int32_t>();
    int32_t* hostLastAccepted = mHostLastAcceptedTokens.dataPointer<int32_t>();
    for (int32_t b = 0; b < activeBatchSize; ++b)
    {
        hostLastAccepted[b] = context.tokenIds[b].back();
        hostDraftInputIds[b * proposalLen] = hostLastAccepted[b];
        for (int32_t j = 1; j < proposalLen; ++j)
        {
            hostDraftInputIds[b * proposalLen + j] = mMaskTokenId;
        }
    }
    CUDA_CHECK(cudaMemcpyAsync(mRuntime.preprocess.idsInput.rawPointer(), mHostDraftInputIds.rawPointer(),
        activeBatchSize * proposalLen * sizeof(int32_t), cudaMemcpyHostToDevice, context.stream));
    CUDA_CHECK(cudaMemcpyAsync(mLastAcceptedTokens.rawPointer(), mHostLastAcceptedTokens.rawPointer(),
        activeBatchSize * sizeof(int32_t), cudaMemcpyHostToDevice, context.stream));

    check::check(mDraftInputsEmbeds.reshape({activeBatchSize, proposalLen, mDraftHiddenSize}), "Tensor reshape failed");
    kernel::embeddingLookup(mRuntime.preprocess.idsInput, mRuntime.preprocess.embedding.table,
        mRuntime.preprocess.embedding.scalesAsOptional(), mDraftInputsEmbeds, context.stream);

    int64_t maxDeltaLen;
    int64_t sourceSeqLen;
    check::check(mHostDeltaLens.reshape({activeBatchSize}), "Tensor reshape failed");
    int32_t* hostDeltaLens = mHostDeltaLens.dataPointer<int32_t>();

    if (context.generationRound == 0)
    {
        sourceSeqLen = mRuntime.base.pipelineIO.baseHiddenStates.getShape()[1];
        maxDeltaLen = 0;
        for (int32_t b = 0; b < activeBatchSize; ++b)
        {
            int32_t const prefillLen = context.effectivePrefillLengths[b];
            hostDeltaLens[b] = prefillLen;
            maxDeltaLen = std::max(maxDeltaLen, static_cast<int64_t>(prefillLen));
        }
    }
    else
    {
        sourceSeqLen = mLastBaseVerifyHiddenStride > 0 ? mLastBaseVerifyHiddenStride : mVerifyLen;
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
        CUDA_CHECK(cudaMemcpy2DAsync(mDraftTargetHidden.rawPointer(), dstPitch,
            mRuntime.base.pipelineIO.baseHiddenStates.rawPointer(), srcPitch, widthBytes, activeBatchSize,
            cudaMemcpyDeviceToDevice, context.stream));
    }

    int32_t const pmLen = divUp(proposalLen, 32);
    check::check(mDraftPackedAttentionMask.reshape({activeBatchSize, proposalLen, pmLen}), "Tensor reshape failed");
    check::check(mDraftAttentionPosId.reshape({activeBatchSize, proposalLen}), "Tensor reshape failed");
    check::check(mDraftContextLengths.reshape({activeBatchSize}), "Tensor reshape failed");
    Tensor const& draftCacheLengths = mDraftCacheManager.getKVCacheLengths();
    kernel::launchDFlashPrepareProposalInputs(draftCacheLengths.dataPointer<int32_t>(),
        mDraftDeltaLens.dataPointer<int32_t>(), proposalLen, mDraftPackedAttentionMask.dataPointer<int32_t>(),
        mDraftAttentionPosId.dataPointer<int32_t>(), mDraftContextLengths.dataPointer<int32_t>(), false,
        activeBatchSize, context.stream);

    check::check(mDraftOutputLogits.reshape({activeBatchSize, proposalLen, mDraftVocabSize}), "Tensor reshape failed");
    check::check(mDraftHiddenStates.reshape({activeBatchSize, proposalLen, mDraftHiddenSize}), "Tensor reshape failed");

    int32_t const draftKVCapacity = mRuntime.deployment.draft->maxKVCacheCapacity;
    InferenceDims const draftDims{
        /*.batch=*/activeBatchSize,
        /*.seqLen=*/proposalLen,
        /*.kvLen=*/draftKVCapacity,
        /*.selectLen=*/static_cast<int64_t>(maxDeltaLen),
        /*.attnMaskSeqLen=*/proposalLen,
        /*.ropeBatch=*/1,
        /*.packedMaskLen=*/static_cast<int64_t>(pmLen),
        /*.contextMaskSelectorLen=*/0,
        /*.startIndexLen=*/activeBatchSize,
        /*.specVerifyPhaseLen=*/0,
        /*.skipSoftmaxScaleLen=*/0,
    };

    cudaGetLastError();
    bool draftSuccess = mDraftExecutor->prepare(
        dsparkDraftProfileForRound(context.generationRound), draftDims, mDraftTensorMap, context.stream);
    if (draftSuccess)
    {
        draftSuccess = mDraftExecutor->execute(context.stream);
    }
    if (!draftSuccess)
    {
        LOG_ERROR("DSparkDecoder: draft engine execution failed.");
        return false;
    }

    if (context.generationRound == 0)
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

    check::check(mDraftTokenIds.reshape({activeBatchSize, proposalLen}), "Tensor reshape failed");
    bool const stochasticSampling
        = ::trt_edgellm::shouldUseNonGreedySampling(context.temperature, context.topK, context.topP);
    if (stochasticSampling)
    {
        if (mUseTree)
        {
            LOG_ERROR(
                "DSparkDecoder: DDTree drafting supports greedy decoding only; disable sampling or use "
                "draftingTopK=1.");
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
        if (dsparkCanUseTopKFastPath(context.topK, context.topP))
        {
            int32_t const samplingTopK = static_cast<int32_t>(context.topK);
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
                    context.stream);
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
                    samplingTopK, context.temperature, context.stream);
            }
        }
        else
        {
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
                    activeBatchSize, proposalLen, mDraftVocabSize, mMarkovRank, context.temperature,
                    static_cast<int32_t>(context.topK), context.topP, context.stream);
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
            // Greedy, chain and tree alike: per-step Markov-corrected row + top-1. Shared
            // code makes tree candidateTopK=1 reproduce the chain structurally, and this
            // path outruns the old fused batch-tile argmax kernel at every batch size.
            int32_t const proposalDepthSize = proposalLen + 1;
            check::check(mDraftStepLogits.reshape({activeBatchSize, mDraftVocabSize}), "Tensor reshape failed");
            check::check(mDraftStepTopKValues.reshape({activeBatchSize, 1}), "Tensor reshape failed");
            check::check(mDraftStepTopKIndices.reshape({activeBatchSize, 1}), "Tensor reshape failed");
            if (mUseTree)
            {
                check::check(mStackedMarkovLogits.reshape({activeBatchSize, proposalDepthSize, mDraftVocabSize}),
                    "Tensor reshape failed");
            }
            size_t const rowBytes = static_cast<size_t>(mDraftVocabSize) * sizeof(float);
            for (int32_t step = 0; step < proposalLen; ++step)
            {
                kernel::dsparkBuildMarkovLogits(mDraftOutputLogits, mMarkovW1, mMarkovW2, mLastAcceptedTokens,
                    mDraftTokenIds, mDraftStepLogits, activeBatchSize, step, proposalLen, mDraftVocabSize, mMarkovRank,
                    context.stream);
                if (mUseTree)
                {
                    // Step logits become the depth-(step+1) candidate row (row 0 is the root placeholder).
                    CUDA_CHECK(cudaMemcpy2DAsync(static_cast<char*>(mStackedMarkovLogits.rawPointer())
                            + static_cast<size_t>(step + 1) * rowBytes,
                        static_cast<size_t>(proposalDepthSize) * rowBytes, mDraftStepLogits.rawPointer(), rowBytes,
                        rowBytes, activeBatchSize, cudaMemcpyDeviceToDevice, context.stream));
                }
                selectAllTopK(mDraftStepLogits, std::ref(mDraftStepTopKValues), mDraftStepTopKIndices, /*topK=*/1,
                    mRuntime.sampling.workspace, context.stream);
                CUDA_CHECK(cudaMemcpy2DAsync(
                    static_cast<char*>(mDraftTokenIds.rawPointer()) + static_cast<size_t>(step) * sizeof(int32_t),
                    static_cast<size_t>(proposalLen) * sizeof(int32_t), mDraftStepTopKIndices.rawPointer(),
                    sizeof(int32_t), sizeof(int32_t), activeBatchSize, cudaMemcpyDeviceToDevice, context.stream));
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
                mConfidenceThreshold, mMinScheduledProposalLen, mMaxScheduledProposalLen, context.stream);
        }
        else
        {
            kernel::dsparkComputeConfidenceAndProposalLengths(mDraftHiddenStates, mMarkovW1, mConfidenceWeight,
                mConfidenceBias, mLastAcceptedTokens, mDraftTokenIds, mConfidenceScores, mProposalLengths,
                activeBatchSize, proposalLen, mDraftHiddenSize, mMarkovRank, mConfidenceHeadWithMarkov,
                mConfidenceThreshold, mMinScheduledProposalLen, mMaxScheduledProposalLen, context.stream);
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
    }
    else
    {
        kernel::dsparkFillProposalLengths(mProposalLengths, activeBatchSize, proposalLen, context.stream);
    }
    if (mUseTree)
    {
        mCurrentVerifyLen = mVerifyLen;
        if (mUseTreeScheduler)
        {
            check::check(mConfidenceScores.reshape({activeBatchSize, proposalLen}), "Tensor reshape failed");
            kernel::dsparkComputeConfidenceScores(mDraftHiddenStates, mMarkovW1, mConfidenceWeight, mConfidenceBias,
                mLastAcceptedTokens, mDraftTokenIds, mConfidenceScores, activeBatchSize, proposalLen, mDraftHiddenSize,
                mMarkovRank, mConfidenceHeadWithMarkov, context.stream);
        }
        return buildTreeVerifyInputs(activeBatchSize, context.stream, mUseTreeScheduler);
    }
    mCurrentVerifyLen = mCurrentProposalLen + 1;

    check::check(mVerifyTokenIds.reshape({activeBatchSize, mCurrentVerifyLen}), "Tensor reshape failed");
    kernel::dsparkBuildVerifyTokens(mLastAcceptedTokens, mDraftTokenIds, mVerifyTokenIds, activeBatchSize, mProposalLen,
        mCurrentProposalLen, context.stream);

    return true;
}

bool DSparkDecoder::buildTreeVerifyInputs(int32_t activeBatchSize, cudaStream_t stream, bool useConfidence)
{
    int32_t const verifySize = mVerifyLen;
    int32_t const proposalDepthSize = mProposalLen + 1;

    check::check(
        mStackedMarkovLogits.reshape({activeBatchSize, proposalDepthSize, mDraftVocabSize}), "Tensor reshape failed");
    // Also reshape here (not only in runDraftForward): graph capture calls this
    // directly, and ddtreeBuild requires rootTokenIds to be [batch].
    check::check(mLastAcceptedTokens.reshape({activeBatchSize}), "Tensor reshape failed");
    check::check(mTreeTokenIds.reshape({activeBatchSize, verifySize}), "Tensor reshape failed");
    check::check(mTreeNodeDepths.reshape({activeBatchSize, verifySize}), "Tensor reshape failed");
    check::check(mTreeParentIds.reshape({activeBatchSize, verifySize}), "Tensor reshape failed");
    check::check(mTreeNodeScores.reshape({activeBatchSize, verifySize}), "Tensor reshape failed");
    check::check(mValidCounts.reshape({activeBatchSize}), "Tensor reshape failed");
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

    // Depths/parents stay decoder-local: the pure-attention base has no tree-metadata
    // engine inputs. No draft vocab reduction, so no reduced-to-full mapping.
    Tensor const& baseKVCacheLengths = mRuntime.base.cacheManager.getKVCacheLengths();
    kernel::DDTreeBuildParams const buildParams{
        {mStackedMarkovLogits, mLastAcceptedTokens, baseKVCacheLengths, nullptr,
            useConfidence ? &mConfidenceScores : nullptr, useConfidence ? mConfidenceThreshold : 0.0F},
        {mTreeTokenIds, mTreeNodeDepths, mTreeParentIds, mTreeNodeScores, mValidCounts, mRuntime.preprocess.idsInput,
            mRuntime.base.pipelineIO.specDecodePositionIds, mRuntime.base.pipelineIO.packedAttentionMask,
            mVerifyTreeMask, mRuntime.base.pipelineIO.contextLengths, mRuntime.base.pipelineIO.selectTokenIndices},
        mRuntime.deployment.specConfig->draftingTopK, mTreeBuildWorkspace.rawPointer(),
        static_cast<size_t>(mTreeBuildWorkspace.getMemoryCapacity()), stream};
    kernel::ddtreeBuild(buildParams);

    return true;
}

bool DSparkDecoder::runBaseVerification(DecodingInferenceContext& context)
{
    TIME_STAGE(metrics::StageNames::kSPEC_DECODE_BASE_VERIFICATION, context.stream);
    NVTX_SCOPED_RANGE(nvtx_dspark_verify, "DSparkDecoder::runBaseVerification", nvtx_colors::MAGENTA);

    int32_t const activeBatchSize = context.activeBatchSize;
    int32_t const verifyLen = mCurrentVerifyLen;

    check::check(mRuntime.preprocess.idsInput.reshape({activeBatchSize, verifyLen}), "Tensor reshape failed");
    if (!mUseTree)
    {
        // Tree mode: ddtreeBuild already wrote the verify token ids into idsInput.
        CUDA_CHECK(cudaMemcpyAsync(mRuntime.preprocess.idsInput.rawPointer(), mVerifyTokenIds.rawPointer(),
            activeBatchSize * verifyLen * sizeof(int32_t), cudaMemcpyDeviceToDevice, context.stream));
    }

    check::check(mRuntime.base.pipelineIO.inputsEmbeds.reshape(
                     {activeBatchSize, verifyLen, mRuntime.deployment.base.hiddenSize}),
        "Tensor reshape failed");
    kernel::embeddingLookup(mRuntime.preprocess.idsInput, mRuntime.preprocess.embedding.table,
        mRuntime.preprocess.embedding.scalesAsOptional(), mRuntime.base.pipelineIO.inputsEmbeds, context.stream);

    int32_t const selectTokenSize = activeBatchSize * verifyLen;
    check::check(
        mRuntime.base.pipelineIO.outputLogits.reshape({selectTokenSize, mRuntime.deployment.base.outputVocabSize}),
        "Tensor reshape failed");
    check::check(mRuntime.base.pipelineIO.baseHiddenStates.reshape({selectTokenSize, mBaseOutputHiddenDim}),
        "Tensor reshape failed");

    int32_t const pmLen = divUp(verifyLen, 32);
    check::check(mRuntime.base.pipelineIO.packedAttentionMask.reshape({activeBatchSize, verifyLen, pmLen}),
        "Tensor reshape failed");
    check::check(
        mRuntime.base.pipelineIO.selectTokenIndices.reshape({activeBatchSize, verifyLen}), "Tensor reshape failed");
    check::check(mRuntime.base.pipelineIO.contextLengths.reshape({activeBatchSize}), "Tensor reshape failed");
    check::check(
        mRuntime.base.pipelineIO.specDecodePositionIds.reshape({activeBatchSize, verifyLen}), "Tensor reshape failed");

    if (!mUseTree)
    {
        // Tree mode: buildTreeVerifyInputs already emitted the base verify metadata.
        Tensor const& baseKVCacheLengths = mRuntime.base.cacheManager.getKVCacheLengths();
        kernel::launchDFlashPrepareBaseVerifyInputs(baseKVCacheLengths.dataPointer<int32_t>(), verifyLen,
            mRuntime.base.pipelineIO.packedAttentionMask.dataPointer<int32_t>(),
            mRuntime.base.pipelineIO.specDecodePositionIds.dataPointer<int32_t>(),
            mRuntime.base.pipelineIO.selectTokenIndices.dataPointer<int64_t>(),
            mRuntime.base.pipelineIO.contextLengths.dataPointer<int32_t>(), activeBatchSize, context.stream);
    }

    if (mRuntime.preprocess.deepstack)
    {
        mRuntime.preprocess.deepstack->useZeroTarget(mRuntime.base.tensorMap);
    }

    mRuntime.base.cacheManager.getMambaCacheManager().reshapeIntermediateStates(activeBatchSize, verifyLen);

    cudaGetLastError();
    auto const verifyDims = mRuntime.deployment.base.specVerifyDims(activeBatchSize, verifyLen);
    bool verifySuccess
        = mRuntime.base.executor.prepare(kDecodeProfile, verifyDims, mRuntime.base.tensorMap, context.stream);
    if (verifySuccess)
    {
        verifySuccess = mRuntime.base.executor.execute(context.stream);
    }
    if (!verifySuccess)
    {
        LOG_ERROR("DSparkDecoder: base verification execution failed.");
        return false;
    }

    check::check(mAcceptedTokenIds.reshape({activeBatchSize, verifyLen}), "Tensor reshape failed");
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

    if (mUseTree)
    {
        int32_t const maxAcceptLength = std::min(mProposalLen + 1, verifyLen);
        check::check(mAcceptedTokenIds.reshape({activeBatchSize, maxAcceptLength}), "Tensor reshape failed");
        check::check(mAcceptedTokenIndices.reshape({activeBatchSize, maxAcceptLength}), "Tensor reshape failed");
        check::check(mRuntime.base.pipelineIO.outputLogits.reshape({activeBatchSize * verifyLen, baseVocabSize}),
            "Tensor reshape failed");
        kernel::eagleAccept(mRuntime.base.pipelineIO.outputLogits, mTreeTokenIds, mVerifyTreeMask, mAcceptedTokenIds,
            mAcceptedTokenIndices, mAcceptLength, std::nullopt, mRuntime.sampling.workspace.rawPointer(),
            mRuntime.sampling.workspace.getMemoryCapacity(), context.stream);

        decoder_utils::clampAcceptLengthsToRemainingGeneration(
            context, mHostAcceptLengths, mAcceptLength, context.stream);
        commitAcceptedTreePath(context, verifyLen, maxAcceptLength);
        // The hidden compaction leaves rows at stride maxAcceptLength, not verifyLen.
        mLastBaseVerifyHiddenStride = maxAcceptLength;

        decoder_utils::appendAcceptedTokens(context, mHostAcceptLengths, mHostAcceptedTokenIds, mAcceptLength,
            mAcceptedTokenIds, maxAcceptLength, mRuntime.tokenizer, context.stream);
        return true;
    }

    bool const stochasticSampling
        = ::trt_edgellm::shouldUseNonGreedySampling(context.temperature, context.topK, context.topP);
    if (stochasticSampling)
    {
        bool const sparseTopK = dsparkCanUseTopKFastPath(context.topK, context.topP);
        int32_t const samplingTopK = sparseTopK ? static_cast<int32_t>(context.topK) : 0;
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
                context.temperature, context.stream);
        }
        else
        {
            check::check(
                mTargetProbabilities.reshape({activeBatchSize, verifyLen, baseVocabSize}), "Tensor reshape failed");
            kernel::dsparkLogitsToProbabilities(mRuntime.base.pipelineIO.outputLogits, mTargetProbabilities,
                activeBatchSize * verifyLen, baseVocabSize, context.temperature, static_cast<int32_t>(context.topK),
                context.topP, context.stream);
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
            kernel::dsparkSparseTopKAccept(mTargetTopKProbabilities, mTargetTopKIndices, mDraftTopKProbabilities,
                mDraftTopKIndices, mDraftTokenIds, mProposalLengths, mAcceptUniforms, mAcceptedTokenIds, mAcceptLength,
                activeBatchSize, mProposalLen, mCurrentProposalLen, samplingTopK, samplingTopK, context.stream);
        }
        else
        {
            kernel::dsparkProbabilisticAccept(mTargetProbabilities, mDraftProbabilities, mDraftTokenIds,
                mProposalLengths, mAcceptUniforms, mAcceptedTokenIds, mAcceptLength, activeBatchSize, mProposalLen,
                mCurrentProposalLen, baseVocabSize, context.stream);
        }
    }
    else
    {
        kernel::dsparkGreedyAccept(mRuntime.base.pipelineIO.outputLogits, mDraftTokenIds, mProposalLengths,
            mAcceptedTokenIds, mAcceptLength, mArgmaxScratch, activeBatchSize, mProposalLen, mCurrentProposalLen,
            baseVocabSize, context.stream);
    }

    decoder_utils::clampAcceptLengthsToRemainingGeneration(context, mHostAcceptLengths, mAcceptLength, context.stream);
    mRuntime.base.cacheManager.commitSequenceLength(mAcceptLength, context.stream);

    check::check(mRuntime.base.pipelineIO.baseHiddenStates.reshape({activeBatchSize, verifyLen, mBaseOutputHiddenDim}),
        "Tensor reshape failed");
    mLastBaseVerifyHiddenStride = verifyLen;

    mRuntime.base.cacheManager.getMambaCacheManager().scatterAcceptedLinearStates(mAcceptLength, context.stream);

    decoder_utils::appendAcceptedTokens(context, mHostAcceptLengths, mHostAcceptedTokenIds, mAcceptLength,
        mAcceptedTokenIds, verifyLen, mRuntime.tokenizer, context.stream);

    return true;
}

void DSparkDecoder::commitAcceptedTreePath(
    DecodingInferenceContext& context, int32_t verifySize, int32_t maxAcceptLength)
{
    int32_t const activeBatchSize = context.activeBatchSize;
    auto& cacheMgrBase = mRuntime.base.cacheManager;
    Tensor const& kvCacheLengths = cacheMgrBase.getKVCacheLengths();
    auto& kvMgrBase = cacheMgrBase.getKVCacheManager();
    auto const kvHeadDimGroups = cacheMgrBase.getKVHeadDimGroups();
    auto const kvCacheType = kvMgrBase.getConfig().kvCacheType;
    auto const& basePageTable = *mRuntime.base.sharedResources.kvPageTables[0];
    int32_t const* basePageTablePtr = basePageTable.kernelView().dataPointer<int32_t>();
    int32_t const baseNumPages = kvMgrBase.numPages();
    int32_t const baseMaxPagesPerSeq = basePageTable.maxPagesPerSeq();
    auto& mambaMgr = cacheMgrBase.getMambaCacheManager();
    bool const hasHybridStates = mambaMgr.hasIntermediateRecurrentStates() || mambaMgr.hasIntermediateConvStates();

    check::check(mRuntime.base.pipelineIO.baseHiddenStates.reshape({activeBatchSize, verifySize, mBaseOutputHiddenDim}),
        "Tensor reshape failed");
    // Branching-tree accept can skip nodes, so commit compacts accepted KV rows using accepted verify indices.
    for (auto const& group : kvHeadDimGroups)
    {
        kernel::eagleBaseCommitKVCache(mAcceptedTokenIndices, mAcceptLength, kvCacheLengths, group.deviceLayerInfos,
            group.numLayers, group.headDim, group.maxKVHeads, activeBatchSize, maxAcceptLength, kvCacheType,
            context.stream, basePageTablePtr, baseNumPages, baseMaxPagesPerSeq);
    }
    kernel::eagleBaseAssembleHiddenState(
        mAcceptedTokenIndices, mAcceptLength, mRuntime.base.pipelineIO.baseHiddenStates, context.stream);
    cacheMgrBase.commitSequenceLength(mAcceptLength, context.stream);
    if (hasHybridStates)
    {
        // Must use the same predicate as the plugin's verify dispatch (see
        // dflashDecoder.cpp::commitAcceptedTreePath for the full rationale).
        if (kernel::gdnTreeChunkVerifyEnabled(verifySize))
        {
            mambaMgr.replayCommitAcceptedTreeStates(mAcceptedTokenIndices, mAcceptLength, context.stream);
        }
        else
        {
            mambaMgr.scatterAcceptedTreeStates(mAcceptedTokenIndices, mAcceptLength, context.stream);
        }
    }

    check::check(
        mRuntime.base.pipelineIO.baseHiddenStates.reshape({activeBatchSize, maxAcceptLength, mBaseOutputHiddenDim}),
        "Tensor reshape failed");
}

bool DSparkDecoder::captureCudaGraphs(cudaStream_t stream)
{
    bool draftProposalCaptureStatus{true};
    bool baseVerificationCaptureStatus{true};

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
            zeroCacheLens.data(), {mRuntime.maxRuntimeBatchSize}, DeviceType::kCPU, nvinfer1::DataType::kINT32);
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
        Tensor simCacheLensTensor(simCacheLens.data(), {batchSize}, DeviceType::kCPU, nvinfer1::DataType::kINT32);
        mRuntime.base.cacheManager.resetForNewSequences(simCacheLensTensor, stream);
        mDraftCacheManager.resetForNewSequences(simCacheLensTensor, stream);

        int32_t const draftPmLen = divUp(proposalLen, 32);
        // Delta = last round's accept length; capped by the draft profile (block + 1),
        // which tree-mode verifyLen (node budget) can exceed.
        int32_t const maxSimDeltaLen = std::min(proposalLen + 1, verifyLen);
        for (int32_t simDeltaLen = 1; simDeltaLen <= maxSimDeltaLen; ++simDeltaLen)
        {
            check::check(
                mDraftInputsEmbeds.reshape({batchSize, proposalLen, mDraftHiddenSize}), "Tensor reshape failed");
            check::check(
                mDraftTargetHidden.reshape({batchSize, static_cast<int64_t>(simDeltaLen), mBaseOutputHiddenDim}),
                "Tensor reshape failed");
            check::check(
                mDraftOutputLogits.reshape({batchSize, proposalLen, mDraftVocabSize}), "Tensor reshape failed");
            check::check(
                mDraftHiddenStates.reshape({batchSize, proposalLen, mDraftHiddenSize}), "Tensor reshape failed");
            check::check(
                mDraftPackedAttentionMask.reshape({batchSize, proposalLen, draftPmLen}), "Tensor reshape failed");
            check::check(mDraftAttentionPosId.reshape({batchSize, proposalLen}), "Tensor reshape failed");
            check::check(mDraftContextLengths.reshape({batchSize}), "Tensor reshape failed");

            std::vector<int32_t> simDeltaLens(batchSize, simDeltaLen);
            check::check(mDraftDeltaLens.reshape({batchSize}), "Tensor reshape failed");
            CUDA_CHECK(cudaMemcpyAsync(mDraftDeltaLens.rawPointer(), simDeltaLens.data(), batchSize * sizeof(int32_t),
                cudaMemcpyHostToDevice, stream));

            Tensor const& draftCacheLengths = mDraftCacheManager.getKVCacheLengths();
            kernel::launchDFlashPrepareProposalInputs(draftCacheLengths.dataPointer<int32_t>(),
                mDraftDeltaLens.dataPointer<int32_t>(), proposalLen, mDraftPackedAttentionMask.dataPointer<int32_t>(),
                mDraftAttentionPosId.dataPointer<int32_t>(), mDraftContextLengths.dataPointer<int32_t>(), false,
                batchSize, stream);

            InferenceDims const draftDims{
                /*.batch=*/batchSize,
                /*.seqLen=*/proposalLen,
                /*.kvLen=*/draftKVCapacity,
                /*.selectLen=*/static_cast<int64_t>(simDeltaLen),
                /*.attnMaskSeqLen=*/proposalLen,
                /*.ropeBatch=*/1,
                /*.packedMaskLen=*/static_cast<int64_t>(draftPmLen),
                /*.contextMaskSelectorLen=*/0,
                /*.startIndexLen=*/batchSize,
                /*.specVerifyPhaseLen=*/0,
                /*.skipSoftmaxScaleLen=*/0,
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
            int32_t const selectTokenSize = batchSize * simVerifyLen;
            check::check(mRuntime.base.pipelineIO.outputLogits.reshape(
                             {selectTokenSize, mRuntime.deployment.base.outputVocabSize}),
                "Tensor reshape failed");
            check::check(mRuntime.base.pipelineIO.baseHiddenStates.reshape({selectTokenSize, mBaseOutputHiddenDim}),
                "Tensor reshape failed");
            check::check(mRuntime.base.pipelineIO.inputsEmbeds.reshape(
                             {batchSize, simVerifyLen, mRuntime.deployment.base.hiddenSize}),
                "Tensor reshape failed");
            check::check(mRuntime.base.pipelineIO.packedAttentionMask.reshape(
                             {batchSize, simVerifyLen, static_cast<int64_t>(divUp(simVerifyLen, 32))}),
                "Tensor reshape failed");
            check::check(mRuntime.base.pipelineIO.selectTokenIndices.reshape({batchSize, simVerifyLen}),
                "Tensor reshape failed");
            check::check(mRuntime.base.pipelineIO.contextLengths.reshape({batchSize}), "Tensor reshape failed");
            check::check(mRuntime.base.pipelineIO.specDecodePositionIds.reshape({batchSize, simVerifyLen}),
                "Tensor reshape failed");

            if (mUseTree)
            {
                // Capture-time build runs on the zero-initialized stacked logits;
                // confidence is skipped so the build stays deterministic.
                buildTreeVerifyInputs(batchSize, stream, /*useConfidence=*/false);
            }
            else
            {
                Tensor const& baseKVCacheLengths = mRuntime.base.cacheManager.getKVCacheLengths();
                kernel::launchDFlashPrepareBaseVerifyInputs(baseKVCacheLengths.dataPointer<int32_t>(), simVerifyLen,
                    mRuntime.base.pipelineIO.packedAttentionMask.dataPointer<int32_t>(),
                    mRuntime.base.pipelineIO.specDecodePositionIds.dataPointer<int32_t>(),
                    mRuntime.base.pipelineIO.selectTokenIndices.dataPointer<int64_t>(),
                    mRuntime.base.pipelineIO.contextLengths.dataPointer<int32_t>(), batchSize, stream);
            }

            if (mRuntime.preprocess.deepstack)
            {
                mRuntime.preprocess.deepstack->useZeroTarget(mRuntime.base.tensorMap);
            }

            mRuntime.base.cacheManager.getMambaCacheManager().reshapeIntermediateStates(batchSize, simVerifyLen);

            auto const verifyDims = mRuntime.deployment.base.specVerifyDims(batchSize, simVerifyLen);
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

bool DSparkDecoder::hasSystemPromptKVCache(SystemPromptCacheKey const& key) const
{
    return mSystemPromptKVCacheDraft.find(key) != mSystemPromptKVCacheDraft.end();
}

void DSparkDecoder::restoreSystemPromptKVCache(SystemPromptCacheKey const& key, int32_t batchIdx, cudaStream_t stream)
{
    check::check(mSystemPromptKVCacheDraft.count(key) > 0, "DSpark system prompt cache missing for draft model");
    mDraftCacheManager.restoreKVCache(mSystemPromptKVCacheDraft[key].kvCacheLayers, batchIdx, stream);
}

bool DSparkDecoder::runSystemPromptPrefill(DecodingInferenceContext& context)
{
    int32_t const activeBatchSize = context.activeBatchSize;
    int64_t const prefillLen = mRuntime.base.pipelineIO.baseHiddenStates.getShape()[1];
    int32_t const proposalLen = mProposalLen;

    check::check(mRuntime.preprocess.idsInput.reshape({activeBatchSize, proposalLen}), "Tensor reshape failed");
    check::check(mHostDraftInputIds.reshape({activeBatchSize, proposalLen}), "Tensor reshape failed");
    int32_t* hostDraftInputIds = mHostDraftInputIds.dataPointer<int32_t>();
    for (int32_t b = 0; b < activeBatchSize; ++b)
    {
        hostDraftInputIds[b * proposalLen] = context.tokenIds[b].back();
        for (int32_t j = 1; j < proposalLen; ++j)
        {
            hostDraftInputIds[b * proposalLen + j] = mMaskTokenId;
        }
    }
    CUDA_CHECK(cudaMemcpyAsync(mRuntime.preprocess.idsInput.rawPointer(), mHostDraftInputIds.rawPointer(),
        activeBatchSize * proposalLen * sizeof(int32_t), cudaMemcpyHostToDevice, context.stream));

    check::check(mDraftInputsEmbeds.reshape({activeBatchSize, proposalLen, mDraftHiddenSize}), "Tensor reshape failed");
    kernel::embeddingLookup(mRuntime.preprocess.idsInput, mRuntime.preprocess.embedding.table,
        mRuntime.preprocess.embedding.scalesAsOptional(), mDraftInputsEmbeds, context.stream);

    check::check(
        mDraftTargetHidden.reshape({activeBatchSize, prefillLen, mBaseOutputHiddenDim}), "Tensor reshape failed");
    size_t const targetHiddenBytes = static_cast<size_t>(activeBatchSize) * prefillLen * mBaseOutputHiddenDim
        * utils::getTypeSize(mDraftTargetHidden.getDataType());
    CUDA_CHECK(cudaMemcpyAsync(mDraftTargetHidden.rawPointer(), mRuntime.base.pipelineIO.baseHiddenStates.rawPointer(),
        targetHiddenBytes, cudaMemcpyDeviceToDevice, context.stream));

    check::check(mHostDeltaLens.reshape({activeBatchSize}), "Tensor reshape failed");
    int32_t* hostDeltaLens = mHostDeltaLens.dataPointer<int32_t>();
    std::fill_n(hostDeltaLens, activeBatchSize, static_cast<int32_t>(prefillLen));
    check::check(mDraftDeltaLens.reshape({activeBatchSize}), "Tensor reshape failed");
    CUDA_CHECK(cudaMemcpyAsync(mDraftDeltaLens.rawPointer(), mHostDeltaLens.rawPointer(),
        activeBatchSize * sizeof(int32_t), cudaMemcpyHostToDevice, context.stream));

    int32_t const pmLen = divUp(proposalLen, 32);
    check::check(mDraftPackedAttentionMask.reshape({activeBatchSize, proposalLen, pmLen}), "Tensor reshape failed");
    check::check(mDraftAttentionPosId.reshape({activeBatchSize, proposalLen}), "Tensor reshape failed");
    check::check(mDraftContextLengths.reshape({activeBatchSize}), "Tensor reshape failed");

    Tensor const& draftCacheLengths = mDraftCacheManager.getKVCacheLengths();
    kernel::launchDFlashPrepareProposalInputs(draftCacheLengths.dataPointer<int32_t>(),
        mDraftDeltaLens.dataPointer<int32_t>(), proposalLen, mDraftPackedAttentionMask.dataPointer<int32_t>(),
        mDraftAttentionPosId.dataPointer<int32_t>(), mDraftContextLengths.dataPointer<int32_t>(), false,
        activeBatchSize, context.stream);

    check::check(mDraftOutputLogits.reshape({activeBatchSize, proposalLen, mDraftVocabSize}), "Tensor reshape failed");
    check::check(mDraftHiddenStates.reshape({activeBatchSize, proposalLen, mDraftHiddenSize}), "Tensor reshape failed");
    int32_t const draftKVCapacity = mRuntime.deployment.draft->maxKVCacheCapacity;
    InferenceDims const draftDims{activeBatchSize, proposalLen, draftKVCapacity, prefillLen, proposalLen, 1,
        static_cast<int64_t>(pmLen), 0, activeBatchSize, 0, 0};

    cudaGetLastError();
    bool ok = mDraftExecutor->prepare(kPrefillProfile, draftDims, mDraftTensorMap, context.stream);
    if (ok)
    {
        ok = mDraftExecutor->execute(context.stream);
    }
    if (!ok)
    {
        LOG_ERROR("DSpark: system prompt draft prefill failed.");
        return false;
    }

    check::check(mDraftDeltaLenCommit.reshape({activeBatchSize}), "Tensor reshape failed");
    CUDA_CHECK(cudaMemcpyAsync(mDraftDeltaLenCommit.rawPointer(), mDraftDeltaLens.rawPointer(),
        activeBatchSize * sizeof(int32_t), cudaMemcpyDeviceToDevice, context.stream));
    mDraftCacheManager.commitSequenceLength(mDraftDeltaLenCommit, context.stream);

    return true;
}

void DSparkDecoder::saveSystemPromptKVCache(SystemPromptCacheKey const& key, std::string const& prompt,
    std::vector<tokenizer::Rank> const& tokenizedPrompt, int32_t promptIdsLength, cudaStream_t stream)
{
    constexpr int32_t CACHE_BATCH_IDX{0};
    SystemPromptKVCache savedCache;
    savedCache.systemPrompt = prompt;
    savedCache.tokenizedPrompt = tokenizedPrompt;
    savedCache.kvCacheLayers = mDraftCacheManager.captureKVCache(CACHE_BATCH_IDX, promptIdsLength, stream);
    mSystemPromptKVCacheDraft.insert({key, std::move(savedCache)});
}

void DSparkDecoder::resetForNewSequences(Tensor& reuseLengths, cudaStream_t stream)
{
    mDraftCacheManager.resetForNewSequences(reuseLengths, stream);
    mCommonStateTracker.reset();
    mLastBaseVerifyHiddenStride = 0;
}

void DSparkDecoder::onBatchEvict(std::vector<int32_t> const& batchMapping, int32_t oldActiveBatch,
    int32_t newActiveBatch, Tensor& deviceBatchMapping, cudaStream_t stream)
{
    ELLM_CHECK(batchMapping.size() == static_cast<size_t>(oldActiveBatch),
        "DSpark batch mapping does not match the old active batch");
    mCommonStateTracker.compact(batchMapping, oldActiveBatch, newActiveBatch);
    if (newActiveBatch == 0)
    {
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
        if (tensor.isEmpty() || tensor.getShape().getNumDims() == 0 || tensor.getShape()[0] != oldActiveBatch
            || newActiveBatch == 0)
        {
            return;
        }
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
    };
    compactDeviceTensor(mRuntime.base.pipelineIO.baseHiddenStates);
    if (mCommonStateTracker.draftPrefillOutputsPending())
    {
        compactDeviceTensor(mDraftOutputLogits);
        compactDeviceTensor(mDraftHiddenStates);
        compactDeviceTensor(mDraftTokenIds);
        compactDeviceTensor(mVerifyTokenIds);
        compactDeviceTensor(mDraftProbabilities);
        compactDeviceTensor(mDraftTopKProbabilities);
        compactDeviceTensor(mDraftTopKIndices);
        compactDeviceTensor(mConfidenceScores);
        compactDeviceTensor(mProposalLengths);
        compactDeviceTensor(mLastAcceptedTokens);
        compactDeviceTensor(mStackedMarkovLogits);
        compactDeviceTensor(mTreeTokenIds);
        compactDeviceTensor(mTreeNodeDepths);
        compactDeviceTensor(mTreeParentIds);
        compactDeviceTensor(mTreeNodeScores);
        compactDeviceTensor(mValidCounts);
        compactDeviceTensor(mVerifyTreeMask);
        compactDeviceTensor(mRuntime.preprocess.idsInput);
        compactDeviceTensor(mRuntime.base.pipelineIO.specDecodePositionIds);
        compactDeviceTensor(mRuntime.base.pipelineIO.packedAttentionMask);
        compactDeviceTensor(mRuntime.base.pipelineIO.contextLengths);
        compactDeviceTensor(mRuntime.base.pipelineIO.selectTokenIndices);
    }
    else
    {
        // Acceptance lengths are first written by base verification, after the pending-prefill proposal is consumed.
        compactDeviceTensor(mAcceptLength);
    }
}

} // namespace rt
} // namespace trt_edgellm
