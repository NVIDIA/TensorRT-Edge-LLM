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

#include "common/bindingNames.h"
#include "common/checkMacros.h"
#include "common/cudaUtils.h"
#include "common/logger.h"
#include "common/mathUtils.h"
#include "common/safetensorsUtils.h"
#include "kernels/contextAttentionKernels/utilKernels.h"
#include "kernels/embeddingKernels/embeddingKernels.h"
#include "kernels/posEncoding/applyRopeWriteKV.h"
#include "kernels/speculative/ddtreeKernels.h"
#include "kernels/speculative/dflash2CandidateSelector.h"
#include "kernels/speculative/dflashRuntimeKernels.h"
#include "kernels/speculative/lilicorrChainWalk.h"
#include "kernels/speculative/requestStableRngKernels.h"
#include "kernels/speculative/speculativeKernels.h"
#include "profiling/metrics.h"
#include "profiling/nvtx_wrapper.h"
#include "profiling/timer.h"
#include "runtime/config/llmEngineConfig.h"
#include "runtime/decoding/decoderUtils.h"
#include "runtime/decoding/dflashDecodeUtils.h"
#include "runtime/decoding/guidedDecoder.h"
#include "runtime/decoding/logitBias.h"
#include "runtime/decoding/requestStableRng.h"
#include "runtime/decoding/specDecodeUtils.h"
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
constexpr int32_t kPrefillProfile{0};
constexpr int32_t kDecodeProfile{1};

//! A delta covering prompt positions (the first forward of a request, or a seated row's) runs on the prefill
//! profile; an accepted span runs on the decode one.
int32_t dflashDraftProfile(bool prefillDelta)
{
    return prefillDelta ? kPrefillProfile : kDecodeProfile;
}

void setPackedAncestorBit(
    std::vector<int32_t>& packedMask, int32_t batchIdx, int32_t nodeIdx, int32_t ancestorIdx, int32_t verifyTokenCount)
{
    constexpr int32_t kMaskBitsPerWord{32};
    int32_t const packedMaskLen = static_cast<int32_t>(divUp(verifyTokenCount, kMaskBitsPerWord));
    int32_t const wordIdx = ancestorIdx / kMaskBitsPerWord;
    int32_t const bitIdx = ancestorIdx % kMaskBitsPerWord;
    int64_t const offset = static_cast<int64_t>(batchIdx) * verifyTokenCount * packedMaskLen
        + static_cast<int64_t>(nodeIdx) * packedMaskLen + wordIdx;
    packedMask[offset] |= static_cast<int32_t>(1U << bitIdx);
}
} // namespace

DFlashDecoder::DFlashDecoder(DecodingRuntimeContext& runtime, std::filesystem::path const& engineDir,
    dflash_utils::CachedBlockDraftRuntimeConfig blockDraftConfig, std::unique_ptr<EngineExecutor> draftExecutor,
    ExternalWeightManager draftWeights, cudaStream_t stream)
    : mRuntime(runtime)
    , mDraftCacheManager(*runtime.base.sharedResources.cacheManagers[1])
    , mVersion(runtime.deployment.base.dflashVersion)
    , mBlockDraft(std::move(blockDraftConfig))
    , mDraftExecutor(std::move(draftExecutor))
{
    auto const& deployment = runtime.deployment;
    auto const& baseCfg = deployment.base;
    ELLM_CHECK(deployment.specConfig.has_value(), "DFlashDecoder: specConfig is required.");
    ELLM_CHECK(deployment.draft.has_value(), "DFlashDecoder: draft config is required.");
    ELLM_CHECK(isCachedBlockDraftMode(baseCfg.specDecodeType),
        "DFlashDecoder requires a base engine exported with spec_decode_type=dflash or jetspec.");
    ELLM_CHECK(baseCfg.specDecodeType == mBlockDraft.userMode,
        "DFlashDecoder normalized user mode does not match the base engine config.");
    LLMEngineConfig const& draftCfg = *deployment.draft;

    if (mVersion == DFlashVersion::kV2)
    {
        initializeDFlash2(engineDir, std::move(draftWeights), stream);
        initializeJumpGrafter();
        return;
    }

    int32_t const maxBatch = deployment.maxRuntimeBatchSize();

    ELLM_CHECK(mDraftExecutor != nullptr, std::string(userModeName()) + " decoding requires a validated draft engine.");

    mDraftInputsEmbeds = Tensor({maxBatch, mBlockDraft.blockSize, mBlockDraft.draftHiddenSize}, DeviceType::kGPU,
        trt_edgellm::DataType::kHALF, "DFlashDraft::inputsEmbeds");
    mDraftTargetHidden = Tensor({maxBatch, mBlockDraft.blockSize, mBlockDraft.baseOutputHiddenDim}, DeviceType::kGPU,
        trt_edgellm::DataType::kHALF, "DFlashDraft::targetHiddenScratch");
    // Variants whose draft engine emits a token proposal instead of logits (Domino) override every hook that
    // reads this buffer, so it is only allocated when the engine exposes the binding.
    bool const draftEmitsLogits = mDraftExecutor->hasIOTensor(binding_names::kLogits);
    if (draftEmitsLogits)
    {
        mDraftOutputLogits = Tensor({maxBatch, mBlockDraft.blockSize, mBlockDraft.draftVocabSize}, DeviceType::kGPU,
            trt_edgellm::DataType::kFLOAT, "DFlashDraft::outputLogits");
    }

    int32_t const packedMaskLen = divUp(mBlockDraft.blockSize, 32);
    mDraftPackedAttentionMask = Tensor({maxBatch, mBlockDraft.blockSize, packedMaskLen}, DeviceType::kGPU,
        trt_edgellm::DataType::kINT32, "DFlashDraft::packedMask");
    mDraftAttentionPosId = Tensor(
        {maxBatch, mBlockDraft.blockSize}, DeviceType::kGPU, trt_edgellm::DataType::kINT32, "DFlashDraft::positionIds");
    mDraftContextLengths
        = Tensor({maxBatch}, DeviceType::kGPU, trt_edgellm::DataType::kINT32, "DFlashDraft::contextLengths");
    mDraftDeltaLenCommit
        = Tensor({maxBatch}, DeviceType::kGPU, trt_edgellm::DataType::kINT32, "DFlashDraft::deltaLenCommit");
    mDraftDeltaLens = Tensor({maxBatch}, DeviceType::kGPU, trt_edgellm::DataType::kINT32, "DFlashDraft::deltaLens");
    mDraftTensorMap.set(binding_names::kInputsEmbeds, mDraftInputsEmbeds);
    mDraftTensorMap.set(binding_names::kDFlashTargetHiddenConcat, mDraftTargetHidden);
    if (draftEmitsLogits)
    {
        mDraftTensorMap.set(binding_names::kLogits, mDraftOutputLogits);
    }
    mDraftTensorMap.set(binding_names::kAttentionMask, mDraftPackedAttentionMask);
    mDraftTensorMap.set(binding_names::kAttentionPosId, mDraftAttentionPosId);
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

    mDraftTokenIds = Tensor(
        {maxBatch, mBlockDraft.blockSize}, DeviceType::kGPU, trt_edgellm::DataType::kINT32, "DFlashDraft::tokenIds");
    mHostDraftInputIds = Tensor({maxBatch, mBlockDraft.blockSize}, DeviceType::kCPU, trt_edgellm::DataType::kINT32,
        "DFlashDraft::hostInputIds");
    mHostLastAcceptedTokens
        = Tensor({maxBatch}, DeviceType::kCPU, trt_edgellm::DataType::kINT32, "DFlashDraft::hostLastAcceptedTokens");
    mHostDeltaLens = Tensor({maxBatch}, DeviceType::kCPU, trt_edgellm::DataType::kINT32, "DFlashDraft::hostDeltaLens");
    mLastAcceptedTokens
        = Tensor({maxBatch}, DeviceType::kGPU, trt_edgellm::DataType::kINT32, "DFlashDraft::lastAcceptedTokens");

    // A jump-forward grafting round verifies jumpVerifyTokenCount rows and accepts up to jumpMaxPathLength tokens.
    auto const& specConfig = *deployment.specConfig;
    mTreeTokenIds = Tensor({maxBatch, specConfig.jumpVerifyTokenCount}, DeviceType::kGPU, trt_edgellm::DataType::kINT32,
        "DFlash::treeTokenIds");
    mTreeNodeScores = Tensor({maxBatch, mBlockDraft.verifyTokenCount}, DeviceType::kGPU, trt_edgellm::DataType::kFLOAT,
        "DFlash::treeNodeScores");
    mValidCounts = Tensor({maxBatch}, DeviceType::kGPU, trt_edgellm::DataType::kINT32, "DFlash::validCounts");
    mVerifyTokenIds = Tensor({maxBatch, specConfig.jumpVerifyTokenCount}, DeviceType::kGPU,
        trt_edgellm::DataType::kINT32, "DFlash::verifyTokenIds");
    int32_t const maxAcceptBufferSize
        = std::max(useTreeVerification() ? std::min(mBlockDraft.blockSize, mBlockDraft.verifyTokenCount)
                                         : mBlockDraft.verifyTokenCount,
            specConfig.jumpMaxPathLength);
    mAcceptedTokenIds = Tensor(
        {maxBatch, maxAcceptBufferSize}, DeviceType::kGPU, trt_edgellm::DataType::kINT32, "DFlash::acceptedTokenIds");
    mAcceptedTokenIndices = Tensor({maxBatch, maxAcceptBufferSize}, DeviceType::kGPU, trt_edgellm::DataType::kINT32,
        "DFlash::acceptedTokenIndices");
    mAcceptLength = Tensor({maxBatch}, DeviceType::kGPU, trt_edgellm::DataType::kINT32, "DFlash::acceptLength");
    mCommonStateTracker.reserve(maxBatch);
    mHostAcceptLengths
        = Tensor({maxBatch}, DeviceType::kCPU, trt_edgellm::DataType::kINT32, "DFlash::hostAcceptLengths");
    mHostSeatStaging
        = Tensor({maxBatch, 2}, DeviceType::kCPU, trt_edgellm::DataType::kINT32, "DFlash::hostSeatStaging");
    mHostAcceptedTokenIds = Tensor(
        {maxBatch, maxAcceptBufferSize}, DeviceType::kCPU, trt_edgellm::DataType::kINT32, "DFlash::hostAcceptedIds");
    mAcceptedHiddenStates = Tensor({maxBatch * maxAcceptBufferSize, mBlockDraft.baseOutputHiddenDim}, DeviceType::kGPU,
        trt_edgellm::DataType::kHALF, "DFlash::acceptedHidden");

    size_t const buildWorkspaceSize = useTreeVerification()
        ? kernel::getDDTreeBuildWorkspaceSize(maxBatch, mBlockDraft.blockSize, mBlockDraft.verifyTokenCount,
              deployment.draft->outputVocabSize, mBlockDraft.candidateTopK)
        : 1U;
    ELLM_CHECK(buildWorkspaceSize > 0, "DFlashDecoder: DDTree build workspace size must be > 0.");
    mBuildWorkspace = Tensor({static_cast<int64_t>(buildWorkspaceSize)}, DeviceType::kGPU,
        trt_edgellm::DataType::kUINT8, "DFlash::buildWorkspace");

    initializeJumpGrafter();

    LOG_INFO(
        "DFlashDecoder initialized: user_mode=%s, proposal_attention=%s, tree_policy=%s, blockSize=%d, "
        "proposalLen=%d, verifyTokenCount=%d, candidateTopK=%d, maskTokenId=%d, maxBatch=%d, draftHiddenSize=%d, "
        "baseOutputHiddenDim=%d, draftVocabSize=%d",
        userModeName(), dflash_utils::proposalAttentionPolicyName(mBlockDraft.proposalAttention),
        dflash_utils::blockDraftTreePolicyName(mBlockDraft.treePolicy), mBlockDraft.blockSize, mBlockDraft.proposalLen,
        mBlockDraft.verifyTokenCount, mBlockDraft.candidateTopK, mBlockDraft.maskTokenId, maxBatch,
        mBlockDraft.draftHiddenSize, mBlockDraft.baseOutputHiddenDim, mBlockDraft.draftVocabSize);

    // Load draft vocab map when the draft engine config declares vocab reduction.
    // Gating on the config (not file existence) makes draft vocab reduction an
    // explicit feature toggle: a missing file is a hard error, and a stray file
    // in a non-reduced engine directory is ignored.
    if (deployment.draft->reducedVocabSize > 0)
    {
        auto const draftVocabMapPath = engineDir / binding_names::kDraftVocabMapFileName;
        ELLM_CHECK(std::filesystem::exists(draftVocabMapPath),
            "Draft engine declares reduced_vocab_size > 0 but " + std::string(binding_names::kDraftVocabMapFileName)
                + " is missing from engine directory");
        std::vector<Tensor> vocabMapTensors;
        ELLM_CHECK(safetensors::loadSafetensors(draftVocabMapPath, vocabMapTensors, stream),
            "Failed to load " + std::string(binding_names::kDraftVocabMapFileName) + " from engine directory");
        check::check(vocabMapTensors.size() == 1,
            std::string(binding_names::kDraftVocabMapFileName) + " should contain exactly one tensor");
        check::check(vocabMapTensors[0].getShape().getNumDims() == 1, "draft vocab_map tensor should be 1D");
        check::check(vocabMapTensors[0].getShape()[0] == mBlockDraft.draftVocabSize,
            "draft vocab_map tensor length should match draft model reduced vocab size");
        mDraftVocabMappingTable = std::move(vocabMapTensors[0]);
        mHasDraftVocabMap = true;
        LOG_INFO("DFlashDecoder: draft vocab map loaded (%d reduced -> full vocab tokens)",
            static_cast<int32_t>(mDraftVocabMappingTable.getShape()[0]));
    }
}

void DFlashDecoder::initializeJumpGrafter()
{
    auto const& specConfig = *mRuntime.deployment.specConfig;
    if (specConfig.maxJumpTokens <= 0)
    {
        return;
    }
    // The next draft forward takes a grafting round's accepted path as its delta, which one block bounds.
    ELLM_CHECK(specConfig.jumpMaxPathLength <= mBlockDraft.blockSize,
        std::string(userModeName()) + " jump-forward accepted path exceeds the draft block size.");
    // A DDTree takes forced chains under any node; a linear block takes one, which replaces the draft after its node.
    mJump.initialize(mRuntime.tokenizer, mRuntime.guidedDecoder, mRuntime.deployment.maxRuntimeBatchSize(),
        useTreeVerification() ? JumpGraftGeometry::kTree : JumpGraftGeometry::kChain, specConfig,
        specConfig.verifyTokenCount);
}

void DFlashDecoder::reshapeDraftOutputs(int32_t batchSize)
{
    check::check(mDraftOutputLogits.reshape({batchSize, mBlockDraft.blockSize, mBlockDraft.draftVocabSize}),
        "Tensor reshape failed");
}

void DFlashDecoder::selectLinearProposalTokens(DecodingInferenceContext& context)
{
    int32_t const activeBatchSize = context.activeBatchSize;
    check::check(mDraftOutputLogits.reshape({activeBatchSize * mBlockDraft.blockSize, mBlockDraft.draftVocabSize}),
        "Tensor reshape failed");
    check::check(mDraftTokenIds.reshape({activeBatchSize * mBlockDraft.blockSize, 1}), "Tensor reshape failed");
    selectAllTopK(mDraftOutputLogits, std::nullopt, mDraftTokenIds, 1, mRuntime.sampling.workspace, context.stream);
    check::check(mDraftTokenIds.reshape({activeBatchSize, mBlockDraft.blockSize}), "Tensor reshape failed");
    if (mHasDraftVocabMap)
    {
        check::check(mDraftTokenIds.reshape({activeBatchSize * mBlockDraft.blockSize}), "Tensor reshape failed");
        mapReducedVocabToFullVocab(mDraftTokenIds, mDraftVocabMappingTable, context.stream);
        check::check(mDraftTokenIds.reshape({activeBatchSize, mBlockDraft.blockSize}), "Tensor reshape failed");
    }
}

bool DFlashDecoder::decodeStep(DecodingInferenceContext& context)
{
    if (greedyOnlyAcceptance() && ::trt_edgellm::shouldUseNonGreedySampling(context.uniformSamplingParams()))
    {
        LOG_ERROR("DFlashDecoder: %s verification supports greedy decoding only; route the request through vanilla.",
            useTreeVerification() ? "DDTree" : "V1");
        return false;
    }
    if (isV2() && useTreeVerification() && context.proposalSampling == SpecProposalSampling::kProbabilistic)
    {
        LOG_ERROR(
            "DFlashDecoder: DFlash2 DDTree builds greedy trees; spec_proposal_sampling=probabilistic requires "
            "--spec-draft-top-k 1.");
        return false;
    }
    NVTX_SCOPED_RANGE(nvtx_dflash_decode, "DFlashDecoder::decodeStep", nvtx_colors::GREEN);
    cudaGetLastError();

    {
        TIME_STAGE(metrics::StageNames::kSPEC_DECODE_DRAFT_PROPOSAL, context.stream);
        bool const usePendingPrefillProposal
            = mCommonStateTracker.shouldUsePendingPrefillProposal(context.generationRound);
        if (!usePendingPrefillProposal && !runDraftForward(context, /*prefillDelta=*/!mDraftPrimed))
        {
            LOG_ERROR("DFlashDecoder: draft forward failed.");
            return false;
        }
        mCommonStateTracker.materializePending(context.activeBatchSize);
        if (!prepareBlockDraftVerifyInputs(context))
        {
            LOG_ERROR("DFlashDecoder: verify input preparation failed.");
            return false;
        }
    }
    mCommonStateTracker.consumeDraftPrefillOutputs();

    if (!runBaseVerification(context))
    {
        LOG_ERROR("DFlashDecoder: base verification failed.");
        return false;
    }
    mCommonStateTracker.recordAccepted(mHostAcceptLengths.dataPointer<int32_t>(), context.activeBatchSize);

    return true;
}

bool DFlashDecoder::initializeForGeneration(DecodingInferenceContext& context)
{
    if (context.speculativeDraftPrefillComplete)
    {
        return true;
    }
    mCommonStateTracker.initialize(context);

    if (!runDraftForward(context, /*prefillDelta=*/true))
    {
        LOG_ERROR("DFlashDecoder: failed to initialize draft state for generation.");
        return false;
    }
    mCommonStateTracker.markDraftPrefillOutputsPending();
    context.speculativeDraftPrefillComplete = true;
    return true;
}

void DFlashDecoder::beginBatch() noexcept
{
    mDraftPrimed = false;
}

void DFlashDecoder::appendRow(int32_t executionRow) noexcept
{
    // Within capacity by construction: the session appends no row past the runtime batch.
    check::check(mAcceptLength.reshape({executionRow + 1}) && mHostAcceptLengths.reshape({executionRow + 1}),
        "Tensor reshape failed");
    mCommonStateTracker.appendRow(executionRow);
}

bool DFlashDecoder::seatRow(DecodingInferenceContext& context, int32_t executionRow)
{
    int32_t const prefillTokens = context.effectivePrefillLengths[0];
    // The draft takes the target's prompt hidden states as a delta but the last one, which the next decode step
    // carries beside every resident's accepted span; the proposal this forward makes from the block is not used.
    // Under Hybrid+spec endpoint reuse the checkpoint needs draft state at every prompt position, so the prefill
    // writes the last one too and then rolls the draft length back by one; the next decode step rewrites it.
    bool const writesLastPosition = context.hybridSpecEndpointReuse;
    int32_t const heldBackTokens = writesLastPosition ? 0 : 1;
    if (prefillTokens > heldBackTokens && !runDraftForward(context, /*prefillDelta=*/true, heldBackTokens))
    {
        LOG_ERROR("DFlashDecoder: failed to prefill the draft for a seated row.");
        return false;
    }
    if (writesLastPosition)
    {
        mDraftCacheManager.commitSequenceLength(-1, context.stream);
    }
    check::check(mRuntime.base.pipelineIO.baseHiddenStates.reshape({prefillTokens, mBlockDraft.baseOutputHiddenDim}),
        "Tensor reshape failed");
    // The pending delta is the last prompt position, as for a resident whose verification accepted one token.
    decoder_utils::seatAcceptedSpan(context, executionRow, mAcceptedTokenIds, mAcceptLength, mHostSeatStaging,
        mAcceptedHiddenStates, static_cast<int32_t>(mAcceptedTokenIds.getShape()[1]),
        mRuntime.base.pipelineIO.baseHiddenStates, context.stream);
    mHostAcceptLengths.dataPointer<int32_t>()[executionRow] = 1;
    // Both models now hold the prompt but its last position, which the next decode step's draft pass takes.
    mCommonStateTracker.seatRow(
        executionRow, static_cast<int32_t>(context.rawBatchedInputIds[0].size()) - 1, /*pendingLength=*/1);
    mDraftPrimed = true;
    return true;
}

bool DFlashDecoder::prefillSeatedChunk(DecodingInferenceContext& context)
{
    return runDraftForward(context, /*prefillDelta=*/true);
}

std::vector<int32_t> const& DFlashDecoder::commonMaterializedStateLengths() const noexcept
{
    return mCommonStateTracker.commonMaterializedStateLengths();
}

DecodingKvHeadroomPolicy DFlashDecoder::kvHeadroom() const
{
    int32_t const jumpVerifyTokenCount = mRuntime.deployment.specConfig->jumpVerifyTokenCount;
    ELLM_CHECK(jumpVerifyTokenCount >= mBlockDraft.verifyTokenCount,
        std::string(userModeName()) + " jump-forward KV headroom is outside the supported range");
    // A grafting round verifies jumpVerifyTokenCount rows; the draft still proposes one block.
    return {/*.plain=*/{mBlockDraft.verifyTokenCount, mBlockDraft.blockSize},
        /*.jumpForward=*/{jumpVerifyTokenCount, mBlockDraft.blockSize}};
}

bool DFlashDecoder::runDraftForward(DecodingInferenceContext& context, bool prefillDelta, int32_t heldBackTokens)
{
    NVTX_SCOPED_RANGE(nvtx_dflash_draft, "DFlashDecoder::runDraftForward", nvtx_colors::DARK_ORANGE);

    if (!mDraftExecutor)
    {
        LOG_ERROR("DFlashDecoder: draft engine not loaded.");
        return false;
    }

    int32_t const activeBatchSize = context.activeBatchSize;
    int32_t const BS = mBlockDraft.blockSize;

    // Step 1: Prepare draft input token IDs: [last_accepted_token, mask_id, ..., mask_id].
    check::check(mRuntime.preprocess.idsInput.reshape({activeBatchSize, BS}), "Tensor reshape failed");
    check::check(mHostDraftInputIds.reshape({activeBatchSize, BS}), "Tensor reshape failed");
    check::check(mHostLastAcceptedTokens.reshape({activeBatchSize}), "Tensor reshape failed");
    check::check(mLastAcceptedTokens.reshape({activeBatchSize}), "Tensor reshape failed");
    mHostStagingFence.wait();
    int32_t* hostDraftInputIds = mHostDraftInputIds.dataPointer<int32_t>();
    int32_t* hostLastAccepted = mHostLastAcceptedTokens.dataPointer<int32_t>();
    for (int32_t b = 0; b < activeBatchSize; ++b)
    {
        hostLastAccepted[b] = context.tokenIds[b].back();
        hostDraftInputIds[b * BS] = hostLastAccepted[b];
        for (int32_t j = 1; j < BS; ++j)
        {
            hostDraftInputIds[b * BS + j] = mBlockDraft.maskTokenId;
        }
    }
    CUDA_CHECK(cudaMemcpyAsync(mRuntime.preprocess.idsInput.rawPointer(), mHostDraftInputIds.rawPointer(),
        activeBatchSize * BS * sizeof(int32_t), cudaMemcpyHostToDevice, context.stream));
    CUDA_CHECK(cudaMemcpyAsync(mLastAcceptedTokens.rawPointer(), mHostLastAcceptedTokens.rawPointer(),
        activeBatchSize * sizeof(int32_t), cudaMemcpyHostToDevice, context.stream));

    if (isV2())
    {
        RowSamplingParams const shared = context.uniformSamplingParams();
        bool const targetNonGreedy = ::trt_edgellm::shouldUseNonGreedySampling(shared);
        bool proposalGreedy = !targetNonGreedy;
        if (context.proposalSampling == SpecProposalSampling::kGreedy)
        {
            proposalGreedy = true;
        }
        else if (context.proposalSampling == SpecProposalSampling::kProbabilistic)
        {
            proposalGreedy = false;
        }
        check::check(mHostRequestSeeds.reshape({activeBatchSize}), "Tensor reshape failed");
        check::check(mHostNextPositions.reshape({activeBatchSize}), "Tensor reshape failed");
        check::check(mHostTemperatures.reshape({activeBatchSize}), "Tensor reshape failed");
        check::check(mHostGreedyMask.reshape({activeBatchSize}), "Tensor reshape failed");
        for (int32_t b = 0; b < activeBatchSize; ++b)
        {
            mHostRequestSeeds.dataPointer<int64_t>()[b] = static_cast<int64_t>(context.samplingSeeds[b]);
            mHostNextPositions.dataPointer<int64_t>()[b] = static_cast<int64_t>(requestStableNextAbsolutePosition(
                context.rawBatchedInputIds[b].size(), context.currentGenerateLengths[b]));
            mHostTemperatures.dataPointer<float>()[b] = shared.temperature;
            mHostGreedyMask.dataPointer<int32_t>()[b] = proposalGreedy ? 1 : 0;
        }
        check::check(mRequestSeeds.reshape({activeBatchSize}), "Tensor reshape failed");
        check::check(mNextAbsolutePositions.reshape({activeBatchSize}), "Tensor reshape failed");
        check::check(mSamplingTemperatures.reshape({activeBatchSize}), "Tensor reshape failed");
        check::check(mProposalGreedyMask.reshape({activeBatchSize}), "Tensor reshape failed");
        check::check(mProposalUniforms.reshape({activeBatchSize, mBlockDraft.proposalLen}), "Tensor reshape failed");
        check::check(
            mAcceptUniforms.reshape({activeBatchSize, 2 * mBlockDraft.proposalLen + 1}), "Tensor reshape failed");
        CUDA_CHECK(cudaMemcpyAsync(mRequestSeeds.rawPointer(), mHostRequestSeeds.rawPointer(),
            activeBatchSize * sizeof(int64_t), cudaMemcpyHostToDevice, context.stream));
        CUDA_CHECK(cudaMemcpyAsync(mNextAbsolutePositions.rawPointer(), mHostNextPositions.rawPointer(),
            activeBatchSize * sizeof(int64_t), cudaMemcpyHostToDevice, context.stream));
        CUDA_CHECK(cudaMemcpyAsync(mSamplingTemperatures.rawPointer(), mHostTemperatures.rawPointer(),
            activeBatchSize * sizeof(float), cudaMemcpyHostToDevice, context.stream));
        CUDA_CHECK(cudaMemcpyAsync(mProposalGreedyMask.rawPointer(), mHostGreedyMask.rawPointer(),
            activeBatchSize * sizeof(int32_t), cudaMemcpyHostToDevice, context.stream));
        kernel::launchRequestStableSpecUniforms(reinterpret_cast<uint64_t const*>(mRequestSeeds.rawPointer()),
            reinterpret_cast<uint64_t const*>(mNextAbsolutePositions.rawPointer()),
            mProposalUniforms.dataPointer<float>(), mAcceptUniforms.dataPointer<float>(), activeBatchSize,
            mBlockDraft.proposalLen, context.stream);
    }

    // Step 2: Embed the draft inputs.
    check::check(
        mDraftInputsEmbeds.reshape({activeBatchSize, BS, mBlockDraft.draftHiddenSize}), "Tensor reshape failed");
    kernel::embeddingLookup(mRuntime.preprocess.idsInput, mRuntime.preprocess.embedding.table,
        mRuntime.preprocess.embedding.scalesAsOptional(), mRuntime.preprocess.embedding.descriptor, mDraftInputsEmbeds,
        context.stream);

    // Step 3: Prepare target-hidden delta from base hidden states.
    // The first forward of a request (and a seated row's) uses prefill hidden states; later rounds use the accepted
    // span's hidden states the decoder carried from the last verification.
    // The draft KV update plugin consumes per-batch delta_lengths to skip padded rows.
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
        sourceSeqLen = mAcceptedTokenIds.getShape()[1];
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

    bindTargetHiddenDelta(prefillDelta ? mRuntime.base.pipelineIO.baseHiddenStates : mAcceptedHiddenStates,
        activeBatchSize, maxDeltaLen, sourceSeqLen, prefillDelta, context.stream);

    // Step 4: Prepare proposal attention inputs.
    int32_t const pmLen = divUp(BS, 32);
    check::check(mDraftPackedAttentionMask.reshape({activeBatchSize, BS, pmLen}), "Tensor reshape failed");
    check::check(mDraftAttentionPosId.reshape({activeBatchSize, BS}), "Tensor reshape failed");
    check::check(mDraftContextLengths.reshape({activeBatchSize}), "Tensor reshape failed");
    Tensor const& draftCacheLengths = mDraftCacheManager.getKVCacheLengths();
    kernel::launchDFlashPrepareProposalInputs(draftCacheLengths.dataPointer<int32_t>(),
        mDraftDeltaLens.dataPointer<int32_t>(), BS, mDraftPackedAttentionMask.dataPointer<int32_t>(),
        mDraftAttentionPosId.dataPointer<int32_t>(), mDraftContextLengths.dataPointer<int32_t>(),
        mRuntime.base.pipelineIO.positions.dataPointer<int32_t>(),
        mRuntime.base.pipelineIO.queryStartOffsets.dataPointer<int32_t>(),
        mRuntime.base.pipelineIO.queryLengths.dataPointer<int32_t>(),
        mRuntime.base.pipelineIO.pastLengths.dataPointer<int32_t>(),
        mRuntime.base.pipelineIO.attentionSequenceLengths.dataPointer<int32_t>(),
        mRuntime.base.pipelineIO.stateIndices.dataPointer<int32_t>(), causalProposalMask(), activeBatchSize,
        context.stream);
    prepareDraftRaggedBindings(
        activeBatchSize, BS, static_cast<int32_t>(maxDeltaLen), context.stream, &context.residentRefs);

    // Step 5: Execute the DFlash draft engine.
    if (isV2())
    {
        check::check(mDraftTokenIds.reshape({activeBatchSize, mBlockDraft.proposalLen}), "Tensor reshape failed");
        check::check(mProposalSupportProbs.reshape({activeBatchSize, mBlockDraft.proposalLen, proposalSupportK()}),
            "Tensor reshape failed");
        reshapeDFlash2DraftOutputs(activeBatchSize);
    }
    else
    {
        reshapeDraftOutputs(activeBatchSize);
    }
    int32_t const draftKVCapacity = mRuntime.deployment.draft->maxKVCacheCapacity;
    InferenceDims const draftDims{
        /*.batch=*/activeBatchSize,
        /*.seqLen=*/activeBatchSize * BS,
        /*.kvLen=*/draftKVCapacity,
        /*.selectLen=*/static_cast<int64_t>(activeBatchSize) * maxDeltaLen,
        /*.attnMaskSeqLen=*/activeBatchSize * BS,
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

    bool draftSuccess;
    {
        auto forward
            = gTimer.startForward("draft", draftDims.batch, draftDims.seqLen, draftDims.selectLen, context.stream);
        draftSuccess
            = mDraftExecutor->prepare(dflashDraftProfile(prefillDelta), draftDims, mDraftTensorMap, context.stream);
        if (draftSuccess)
        {
            draftSuccess = mDraftExecutor->execute(context.stream);
        }
    }
    if (!draftSuccess)
    {
        LOG_ERROR("DFlashDecoder: draft engine execution failed.");
        return false;
    }

    if (isLiLiCorr())
    {
        kernel::launchLiLiCorrChainWalk(mProposalSupportIds, mLiLiCorrTransitionScores, mProposalUniforms,
            mSamplingTemperatures, mProposalGreedyMask, mDraftTokenIds, mProposalSupportProbs, context.stream);
    }
    else if (isV2())
    {
        // A DDTree scores every parent state in buildTreeVerifyInputs instead of walking one chain here.
        if (!useTreeVerification())
        {
            kernel::launchDFlash2CandidateSelector(mProposalSupportIds, mProposalUnaryValues, mProposalProjectedHidden,
                mLastAcceptedTokens, mSelectorPredecessorCodebook, mSelectorSuccessorCodebook, mProposalUniforms,
                mSamplingTemperatures, mProposalGreedyMask, mDraftTokenIds, mProposalSupportProbs, context.stream);
        }
    }
    else if (!produceProposal(context))
    {
        return false;
    }

    // Step 6: Commit per-batch delta lengths to the draft cache manager.
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

    return true;
}

bool DFlashDecoder::prepareBlockDraftVerifyInputs(DecodingInferenceContext& context)
{
    if (isV2() && !useTreeVerification())
    {
        return prepareV2Proposal(context);
    }
    NVTX_SCOPED_RANGE(
        nvtx_dflash_prepare_verify, "DFlashDecoder::prepareBlockDraftVerifyInputs", nvtx_colors::LIGHT_ORANGE);

    int32_t const activeBatchSize = context.activeBatchSize;
    int32_t const verifyTokenCount = mBlockDraft.verifyTokenCount;
    check::check(mVerifyTokenIds.reshape({activeBatchSize, verifyTokenCount}), "Tensor reshape failed");

    if (!useTreeVerification())
    {
        selectLinearProposalTokens(context);
        kernel::launchDFlashBuildLinearVerifyInputs(mLastAcceptedTokens.dataPointer<int32_t>(),
            mDraftTokenIds.dataPointer<int32_t>(), mVerifyTokenIds.dataPointer<int32_t>(), activeBatchSize,
            mBlockDraft.proposalLen, mBlockDraft.blockSize, verifyTokenCount, context.stream);
    }
    else if (!buildTreeVerifyInputs(context))
    {
        return false;
    }

    finishVerifyInput(context);
    if (!checkCudaLastError("prepare DFlash verify inputs"))
    {
        return false;
    }
    return true;
}

void DFlashDecoder::finishVerifyInput(DecodingInferenceContext& context)
{
    auto const& specConfig = *mRuntime.deployment.specConfig;
    auto& io = mRuntime.base.pipelineIO;
    int32_t const activeBatchSize = context.activeBatchSize;
    bool const tree = useTreeVerification();
    Tensor& ids = tree ? mTreeTokenIds : mVerifyTokenIds;
    Tensor* const parents = tree ? &io.specTreeParentIds : nullptr;
    SpecVerifyInputs const drafted{ids, tree ? &mValidCounts : nullptr, parents, nullptr, activeBatchSize,
        mBlockDraft.verifyTokenCount, specConfig.maxPathLength, specConfig.verifyLayout};

    // A slot that may jump has its draft walked on the host before verification; a round that grafts a chain verifies
    // at the grafting width and pads the other slots.
    using Walk = JumpGraftSession::Walk;
    Walk const walk = mJump.walk(context, drafted);
    if (mJump.wide())
    {
        mJump.grafter().upload(ids, parents, tree ? &io.specTreeDepths : nullptr, context.stream);
        if (tree)
        {
            mJump.prepareGraftedTreeDescriptors(
                io, mRuntime.base.cacheManager.getKVCacheLengths(), activeBatchSize, context.stream);
        }
    }

    // After a walk that grafted nothing the guided decoder already holds the draft; a grafting round captures the
    // grafted input with its node counts, which end each slot's mask walk.
    if (context.hasGuidedDecoding && walk != Walk::kNoChain)
    {
        mRuntime.guidedDecoder.captureDraft(mJump.wide()
                ? SpecVerifyInputs{ids, mJump.grafter().validCounts(), parents, nullptr, activeBatchSize,
                      specConfig.jumpVerifyTokenCount, specConfig.maxPathLength, specConfig.verifyLayout, false,
                      mJump.grafter().forcedFlags()}
                : drafted,
            context.stream);
    }
}

bool DFlashDecoder::buildTreeVerifyInputs(DecodingInferenceContext& context)
{
    NVTX_SCOPED_RANGE(nvtx_dflash_ddtree_build, "DFlashDecoder::buildTreeVerifyInputs", nvtx_colors::LIGHT_ORANGE);

    int32_t const activeBatchSize = context.activeBatchSize;
    int32_t const verifyTokenCount = mBlockDraft.verifyTokenCount;
    check::check(mTreeTokenIds.reshape({activeBatchSize, verifyTokenCount}), "Tensor reshape failed");
    check::check(mTreeNodeScores.reshape({activeBatchSize, verifyTokenCount}), "Tensor reshape failed");
    check::check(mValidCounts.reshape({activeBatchSize}), "Tensor reshape failed");
    check::check(mRuntime.base.pipelineIO.specTreeParentIds.reshape({activeBatchSize, verifyTokenCount}),
        "Tensor reshape failed");
    check::check(
        mRuntime.base.pipelineIO.specTreeDepths.reshape({activeBatchSize, verifyTokenCount}), "Tensor reshape failed");
    check::check(mVerifyTokenIds.reshape({activeBatchSize, verifyTokenCount}), "Tensor reshape failed");
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
        mRuntime.base.pipelineIO.specTreeParentIds, mTreeNodeScores, mValidCounts, mVerifyTokenIds,
        mRuntime.base.pipelineIO.specDecodePositionIds, mRuntime.base.pipelineIO.packedAttentionMask,
        mRuntime.base.pipelineIO.contextLengths, mRuntime.base.pipelineIO.selectTokenIndices};
    if (isV2())
    {
        selectDFlash2TreeCandidates(activeBatchSize, context.stream);
        kernel::ddtreeAssemble({{mTreeCandidateTokenIds, mTreeCandidateLogProbs, &mTreeCandidateChildStates},
            mLastAcceptedTokens, baseKVCacheLengths, nullptr, 0.0F, outputs, context.stream});
        return true;
    }
    TreeCandidateLogits const candidates = treeCandidateLogits();
    kernel::DDTreeBuildParams const buildParams{
        {candidates.logits, mLastAcceptedTokens, baseKVCacheLengths, candidates.vocabMap}, outputs,
        mBlockDraft.candidateTopK, mBuildWorkspace.rawPointer(),
        static_cast<size_t>(mBuildWorkspace.getMemoryCapacity()), context.stream, candidates.firstCandidateLogitsRow};
    kernel::ddtreeBuild(buildParams);
    finalizeTreeTokens(context);

    return true;
}

bool DFlashDecoder::captureDraftCudaGraphs(cudaStream_t stream)
{
    bool draftProposalCaptureStatus{true};

    static constexpr int32_t kSimulateCacheLength{128};
    int32_t const BS = mBlockDraft.blockSize;
    int32_t const draftKVCapacity = mRuntime.deployment.draft->maxKVCacheCapacity;

    for (int32_t batchSize = 1; batchSize <= mRuntime.maxRuntimeBatchSize; ++batchSize)
    {
        std::vector<int32_t> simCacheLens(batchSize, kSimulateCacheLength);
        Tensor simCacheLensTensor(simCacheLens.data(), {batchSize}, DeviceType::kCPU, trt_edgellm::DataType::kINT32);
        mDraftCacheManager.resetForNewSequences(simCacheLensTensor, stream);

        int32_t const pmLen = divUp(BS, 32);
        for (int32_t simDeltaLen = 1; simDeltaLen <= BS; ++simDeltaLen)
        {
            check::check(
                mDraftInputsEmbeds.reshape({batchSize, BS, mBlockDraft.draftHiddenSize}), "Tensor reshape failed");
            check::check(mDraftTargetHidden.reshape(
                             {batchSize, static_cast<int64_t>(simDeltaLen), mBlockDraft.baseOutputHiddenDim}),
                "Tensor reshape failed");
            mDraftTensorMap.set(binding_names::kDFlashTargetHiddenConcat, mDraftTargetHidden);
            if (isV2())
            {
                reshapeDFlash2DraftOutputs(batchSize);
            }
            else
            {
                reshapeDraftOutputs(batchSize);
            }
            check::check(mDraftPackedAttentionMask.reshape({batchSize, BS, pmLen}), "Tensor reshape failed");
            check::check(mDraftAttentionPosId.reshape({batchSize, BS}), "Tensor reshape failed");
            check::check(mDraftContextLengths.reshape({batchSize}), "Tensor reshape failed");

            std::vector<int32_t> simDeltaLens(batchSize, simDeltaLen);
            check::check(mDraftDeltaLens.reshape({batchSize}), "Tensor reshape failed");
            CUDA_CHECK(cudaMemcpyAsync(mDraftDeltaLens.rawPointer(), simDeltaLens.data(), batchSize * sizeof(int32_t),
                cudaMemcpyHostToDevice, stream));

            Tensor const& draftCacheLengths = mDraftCacheManager.getKVCacheLengths();
            kernel::launchDFlashPrepareProposalInputs(draftCacheLengths.dataPointer<int32_t>(),
                mDraftDeltaLens.dataPointer<int32_t>(), BS, mDraftPackedAttentionMask.dataPointer<int32_t>(),
                mDraftAttentionPosId.dataPointer<int32_t>(), mDraftContextLengths.dataPointer<int32_t>(),
                mRuntime.base.pipelineIO.positions.dataPointer<int32_t>(),
                mRuntime.base.pipelineIO.queryStartOffsets.dataPointer<int32_t>(),
                mRuntime.base.pipelineIO.queryLengths.dataPointer<int32_t>(),
                mRuntime.base.pipelineIO.pastLengths.dataPointer<int32_t>(),
                mRuntime.base.pipelineIO.attentionSequenceLengths.dataPointer<int32_t>(),
                mRuntime.base.pipelineIO.stateIndices.dataPointer<int32_t>(), causalProposalMask(), batchSize, stream);
            prepareDraftRaggedBindings(batchSize, BS, simDeltaLen, stream);

            InferenceDims const draftDims{
                /*.batch=*/batchSize,
                /*.seqLen=*/batchSize * BS,
                /*.kvLen=*/draftKVCapacity,
                /*.selectLen=*/static_cast<int64_t>(batchSize) * simDeltaLen,
                /*.attnMaskSeqLen=*/batchSize * BS,
                /*.ropeBatch=*/1,
                /*.packedMaskLen=*/static_cast<int64_t>(pmLen),
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
                LOG_WARNING("DFlashDecoder: failed to prepare draft graph capture (batch=%d, delta=%d)", batchSize,
                    simDeltaLen);
                draftProposalCaptureStatus = false;
            }
        }
    }

    return draftProposalCaptureStatus;
}

bool DFlashDecoder::runBaseVerification(DecodingInferenceContext& context)
{
    TIME_STAGE(metrics::StageNames::kSPEC_DECODE_BASE_VERIFICATION, context.stream);
    NVTX_SCOPED_RANGE(nvtx_dflash_verify, "DFlashDecoder::runBaseVerification", nvtx_colors::MAGENTA);

    int32_t const activeBatchSize = context.activeBatchSize;
    int32_t const BS = mBlockDraft.blockSize;
    // A grafting round verifies the grafted input at the jump width and accepts up to its path capacity. The layout
    // stays the deployment's: a grafted linear block is still a chain, and its accepted path a prefix of it.
    int32_t const verifyTokenCount = mJump.verifyWidth(mBlockDraft.verifyTokenCount);
    int32_t const plainAcceptLength
        = useTreeVerification() ? std::min(BS, mBlockDraft.verifyTokenCount) : mBlockDraft.verifyTokenCount;
    int32_t const maxAcceptLength = mJump.acceptDepth(plainAcceptLength);

    cudaGetLastError();
    bool const verifySuccess = executeBaseVerification(context, verifyTokenCount);
    if (!verifySuccess)
    {
        return false;
    }

    check::check(mAcceptedTokenIds.reshape({activeBatchSize, maxAcceptLength}), "Tensor reshape failed");
    check::check(mAcceptedTokenIndices.reshape({activeBatchSize, maxAcceptLength}), "Tensor reshape failed");
    check::check(mAcceptLength.reshape({activeBatchSize}), "Tensor reshape failed");
    check::check(mRuntime.base.pipelineIO.outputLogits.reshape(
                     {activeBatchSize * verifyTokenCount, mRuntime.deployment.base.outputVocabSize}),
        "Tensor reshape failed");
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

    // DFlash2's rejection sampling takes no forced tokens. A grafting round is greedy, since sampling requests do not
    // jump, and goes through the greedy walk, which accepts what the rejection sampling would at its top-1 target.
    if (!greedyOnlyAcceptance() && !mJump.wide())
    {
        if (!runV2Acceptance(context, verifyTokenCount, maxAcceptLength))
        {
            return false;
        }
    }
    else
    {
        Tensor const& acceptTokenIds = useTreeVerification() ? mTreeTokenIds : mVerifyTokenIds;
        Tensor const targetTokenIds = kernel::selectSpecTargetTokens(mRuntime.base.pipelineIO.outputLogits,
            std::nullopt, mRuntime.sampling.workspace, activeBatchSize, verifyTokenCount, context.stream);
        Tensor const* validCounts = mJump.validCounts(useTreeVerification() ? &mValidCounts : nullptr);
        Tensor const* const forcedFlags = mJump.forcedFlags();
        SpecVerifyInputs const verify{acceptTokenIds, validCounts, &mRuntime.base.pipelineIO.specTreeParentIds,
            &mRuntime.base.pipelineIO.specTreeDepths, activeBatchSize, verifyTokenCount, maxAcceptLength,
            mRuntime.deployment.specConfig->verifyLayout, useTreeVerification()};
        spec_decode::accept(GreedyAcceptInputs{verify, targetTokenIds, validCounts, forcedFlags}, mAcceptedTokenIds,
            mAcceptLength, &mAcceptedTokenIndices, context.stream,
            decoder_utils::forcedAcceptanceLength(mRuntime, context));
        if (!checkCudaLastError("accept"))
        {
            return false;
        }

        decoder_utils::clampAcceptLengthsToRemainingGeneration(context, mAcceptLength, context.stream);

        spec_decode::commitAcceptedTargetPath(mRuntime, context,
            SpecAcceptResult{mAcceptedTokenIds, mAcceptLength, &mAcceptedTokenIndices, maxAcceptLength},
            verifyTokenCount);
    }

    // The accepted span's hidden states feed the next draft forward's delta; a seated prefill may reuse the
    // pipeline buffer before then.
    decoder_utils::copyLeadingElements(mAcceptedHiddenStates, mRuntime.base.pipelineIO.baseHiddenStates,
        mRuntime.base.pipelineIO.baseHiddenStates.getShape().volume(), context.stream);

    // Enqueue logprobs device work + D2H before appendAcceptedTokens so everything rides
    // that call's single round synchronization. Verify rows are tree nodes (DDTree) or linear
    // block positions; either way the accepted rows can be non-contiguous in outputLogits, so
    // gather them via the accepted verify indices first (same as EAGLE/MTP).
    if (context.maxNumLogprobs() > 0)
    {
        int32_t const vocabSize = mRuntime.deployment.base.outputVocabSize;
        int32_t const gatheredRows = activeBatchSize * maxAcceptLength;
        check::check(mRuntime.logprobs.gatheredLogits.reshape({gatheredRows, vocabSize}), "Tensor reshape failed");
        gatherSpecVerifyAcceptedLogitRows(mRuntime.base.pipelineIO.outputLogits, mAcceptedTokenIndices,
            mRuntime.logprobs.gatheredLogits, activeBatchSize, verifyTokenCount, maxAcceptLength, vocabSize,
            context.stream);
        decoder_utils::enqueueLogprobsD2H(
            mRuntime.logprobs.gatheredLogits, gatheredRows, mRuntime, context.maxNumLogprobs(), context.stream);
    }

    std::optional<decoder_utils::ForcedPathInfo> forcedPath = mJump.forcedPath(mAcceptedTokenIndices);
    // Step 8: Append accepted tokens to context (includes the round's D2H sync). A grafting round proposes the plain
    // round's drafts; its forced tokens are neither drafts nor bonus.
    decoder_utils::appendAcceptedTokens(context, mHostAcceptLengths, mHostAcceptedTokenIds, mAcceptLength,
        mAcceptedTokenIds, maxAcceptLength, mRuntime.tokenizer, context.stream, mBlockDraft.verifyTokenCount - 1,
        nullptr, decoder_utils::forcedAcceptanceLength(mRuntime, context), !useTreeVerification(),
        forcedPath ? &*forcedPath : nullptr);

    if (context.hasGuidedDecoding)
    {
        advanceGuidedDecodingForCommitted(mRuntime.guidedDecoder, context, mHostAcceptedTokenIds.dataPointer<int32_t>(),
            mHostAcceptLengths.dataPointer<int32_t>(), maxAcceptLength, activeBatchSize);
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

bool DFlashDecoder::executeBaseVerification(DecodingInferenceContext& context, int32_t verifyTokenCount)
{
    int32_t const activeBatchSize = context.activeBatchSize;
    bool const selectedTree = useTreeVerification();
    Tensor const* const validCounts = mJump.validCounts(selectedTree ? &mValidCounts : nullptr);
    SpecVerifyInputs const verify{selectedTree ? mTreeTokenIds : mVerifyTokenIds, validCounts,
        selectedTree ? &mRuntime.base.pipelineIO.specTreeParentIds : nullptr,
        selectedTree ? &mRuntime.base.pipelineIO.specTreeDepths : nullptr, activeBatchSize, verifyTokenCount,
        mRuntime.deployment.specConfig->maxPathLength, mRuntime.deployment.specConfig->verifyLayout, selectedTree};
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
    if (!checkCudaLastError("base verification execute"))
    {
        return false;
    }
    if (!verifySuccess)
    {
        LOG_ERROR("DFlashDecoder: base verification execution failed.");
        return false;
    }
    return true;
}

void DFlashDecoder::bindTargetHiddenDelta(Tensor& source, int32_t activeBatchSize, int64_t maxDeltaLen,
    int64_t sourceSeqLen, bool allowLargeDelta, cudaStream_t stream)
{
    auto const compactTargetHidden = [&](Tensor& targetHidden) {
        check::check(targetHidden.reshape({activeBatchSize, maxDeltaLen, mBlockDraft.baseOutputHiddenDim}),
            "Tensor reshape failed");
        size_t const elementBytes = utils::getTypeSize(targetHidden.getDataType());
        size_t const rowBytes = static_cast<size_t>(mBlockDraft.baseOutputHiddenDim) * elementBytes;
        size_t const dstPitch = static_cast<size_t>(maxDeltaLen) * rowBytes;
        size_t const srcPitch = static_cast<size_t>(sourceSeqLen) * rowBytes;
        size_t const widthBytes = static_cast<size_t>(maxDeltaLen) * rowBytes;
        CUDA_CHECK(cudaMemcpy2DAsync(targetHidden.rawPointer(), dstPitch, source.rawPointer(), srcPitch, widthBytes,
            activeBatchSize, cudaMemcpyDeviceToDevice, stream));
        mDraftTensorMap.set(binding_names::kDFlashTargetHiddenConcat, targetHidden);
    };

    if (sourceSeqLen == maxDeltaLen)
    {
        check::check(
            source.reshape({activeBatchSize, maxDeltaLen, mBlockDraft.baseOutputHiddenDim}), "Tensor reshape failed");
        // DFlash V2 draft graphs are captured with mDraftTargetHidden bound, so a V2 fully accepted block is copied
        // below: binding the source would change the graph's binding snapshot and force an eager draft launch.
        if (!isV2() || maxDeltaLen > mBlockDraft.blockSize)
        {
            mDraftTensorMap.set(binding_names::kDFlashTargetHiddenConcat, source);
            return;
        }
    }

    check::check(allowLargeDelta || maxDeltaLen <= mBlockDraft.blockSize,
        "DFlash decode target-hidden delta exceeds block-size scratch.");
    if (maxDeltaLen <= mBlockDraft.blockSize)
    {
        compactTargetHidden(mDraftTargetHidden);
        return;
    }

    int32_t const reserveBatchSize = mRuntime.maxRuntimeBatchSize;
    int64_t const requiredBytes = static_cast<int64_t>(reserveBatchSize) * maxDeltaLen * mBlockDraft.baseOutputHiddenDim
        * static_cast<int64_t>(utils::getTypeSize(trt_edgellm::DataType::kHALF));
    if (mDraftPrefillTargetHidden.getMemoryCapacity() < requiredBytes)
    {
        mDraftPrefillTargetHidden = Tensor{};
        mDraftPrefillTargetHidden = Tensor({reserveBatchSize, maxDeltaLen, mBlockDraft.baseOutputHiddenDim},
            DeviceType::kGPU, trt_edgellm::DataType::kHALF, "DFlashDraft::prefillTargetHiddenScratch");
    }
    compactTargetHidden(mDraftPrefillTargetHidden);
}

void DFlashDecoder::initializeDraftRaggedBindings(LLMEngineConfig const& draftCfg)
{
    int32_t const maxBatch = mRuntime.deployment.maxRuntimeBatchSize();
    int64_t const maxDeltaTokens = static_cast<int64_t>(maxBatch) * draftCfg.maxKVCacheCapacity;
    mDraftDeltaRopeCosSin = Tensor({maxDeltaTokens, draftCfg.rotaryDim}, DeviceType::kGPU,
        trt_edgellm::DataType::kFLOAT, "DFlashDraft::deltaRopeCosSin");
    mDraftDeltaPositions
        = Tensor({maxDeltaTokens}, DeviceType::kGPU, trt_edgellm::DataType::kINT32, "DFlashDraft::deltaPositions");
    mDraftDeltaTokenToSequence = Tensor(
        {maxDeltaTokens}, DeviceType::kGPU, trt_edgellm::DataType::kINT32, "DFlashDraft::deltaTokenToSequence");

    auto& io = mRuntime.base.pipelineIO;
    mDraftTensorMap.set(binding_names::kPositions, io.positions);
    mDraftTensorMap.set(binding_names::kQueryStartOffsets, io.queryStartOffsets);
    mDraftTensorMap.set(binding_names::kQueryLengths, io.queryLengths);
    mDraftTensorMap.set(binding_names::kPastLengths, io.pastLengths);
    mDraftTensorMap.set(binding_names::kAttentionSequenceLengths, io.attentionSequenceLengths);
    mDraftTensorMap.set(binding_names::kStateIndices, io.stateIndices);
    mDraftTensorMap.set(binding_names::kExecutionPhaseMarker, io.executionPhaseMarker);
    mDraftTensorMap.set(binding_names::kContextSequenceCountCarrier, io.contextSequenceCountCarrier);
    mDraftTensorMap.set(binding_names::kKVPageTable, io.raggedKVPageTable);
    mDraftTensorMap.set(binding_names::kRopeCosSin, io.raggedRopeCosSin);
    mDraftTensorMap.set(binding_names::kDFlashDeltaRopeCosSin, mDraftDeltaRopeCosSin);
    mDraftTensorMap.set(binding_names::kDFlashDeltaPositions, mDraftDeltaPositions);
    mDraftTensorMap.set(binding_names::kDFlashDeltaTokenToSequence, mDraftDeltaTokenToSequence);
}

void DFlashDecoder::prepareDraftRaggedBindings(int32_t activeBatchSize, int32_t proposalLen, int32_t deltaWidth,
    cudaStream_t stream, std::vector<ResidentRef> const* residentRefs)
{
    auto& io = mRuntime.base.pipelineIO;
    auto const& cfg = *mRuntime.deployment.draft;
    int32_t const proposalTokens = activeBatchSize * proposalLen;
    int32_t const deltaTokens = activeBatchSize * deltaWidth;

    check::check(
        mDraftInputsEmbeds.reshape({proposalTokens, mBlockDraft.draftHiddenSize}), "DFlash draft input reshape failed");
    if (!isV2() && mDraftTensorMap.contains(binding_names::kLogits))
    {
        check::check(mDraftOutputLogits.reshape({proposalTokens, mBlockDraft.draftVocabSize}),
            "DFlash draft logits reshape failed");
    }
    check::check(mDraftPackedAttentionMask.reshape({proposalTokens, static_cast<int64_t>(divUp(proposalLen, 32))}),
        "DFlash draft attention mask reshape failed");
    check::check(mDraftAttentionPosId.reshape({proposalTokens}), "DFlash draft attention position reshape failed");
    check::check(io.positions.reshape({proposalTokens}), "DFlash draft positions reshape failed");
    check::check(io.queryStartOffsets.reshape({activeBatchSize + 1}), "DFlash draft query offsets reshape failed");
    check::check(io.queryLengths.reshape({activeBatchSize}), "DFlash draft query lengths reshape failed");
    check::check(io.pastLengths.reshape({activeBatchSize}), "DFlash draft past lengths reshape failed");
    check::check(io.attentionSequenceLengths.reshape({activeBatchSize}),
        "DFlash draft attention sequence lengths reshape failed");
    check::check(io.contextSequenceCountCarrier.reshape({0}), "DFlash draft context carrier reshape failed");
    io.uploadStateIndices(residentRefs, activeBatchSize, stream);
    prepareRaggedKVPageTable(io, *mRuntime.base.sharedResources.kvPageTables[1], activeBatchSize, stream);
    prepareRaggedRope(io, mRuntime.base.sharedResources, cfg, proposalTokens, activeBatchSize, stream);

    check::check(mDraftDeltaPositions.reshape({deltaTokens}), "DFlash delta positions reshape failed");
    check::check(mDraftDeltaTokenToSequence.reshape({deltaTokens}), "DFlash delta token-to-sequence reshape failed");
    check::check(mDraftDeltaRopeCosSin.reshape({deltaTokens, cfg.rotaryDim}), "DFlash delta RoPE reshape failed");
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

    Tensor* targetHidden = mDraftTensorMap.get(binding_names::kDFlashTargetHiddenConcat);
    check::check(targetHidden != nullptr, "DFlash target-hidden delta binding is missing");
    check::check(targetHidden->reshape({deltaTokens, mBlockDraft.baseOutputHiddenDim}),
        "DFlash target-hidden delta reshape failed");
}

bool DFlashDecoder::capturePreparedBaseVerification(int32_t batchSize, int32_t verifyTokenCount, cudaStream_t stream)
{
    DecodingInferenceContext captureContext;
    captureContext.stream = stream;
    captureContext.activeBatchSize = batchSize;
    captureContext.residentRefs.resize(batchSize);
    for (int32_t slot = 0; slot < batchSize; ++slot)
    {
        captureContext.residentRefs[slot].slot = slot;
    }
    bool const selectedTree = useTreeVerification();
    if (!selectedTree)
    {
        check::check(mVerifyTokenIds.reshape({batchSize, verifyTokenCount}), "Tensor reshape failed");
        CUDA_CHECK(
            cudaMemsetAsync(mVerifyTokenIds.rawPointer(), 0, batchSize * verifyTokenCount * sizeof(int32_t), stream));
    }
    SpecVerifyInputs const verify{selectedTree ? mTreeTokenIds : mVerifyTokenIds,
        selectedTree ? &mValidCounts : nullptr, selectedTree ? &mRuntime.base.pipelineIO.specTreeParentIds : nullptr,
        selectedTree ? &mRuntime.base.pipelineIO.specTreeDepths : nullptr, batchSize, verifyTokenCount,
        mRuntime.deployment.specConfig->maxPathLength, mRuntime.deployment.specConfig->verifyLayout, selectedTree};
    spec_decode::prepareTargetVerifyInputs(verify, mRuntime, captureContext, 0);
    auto const verifyDims = mRuntime.deployment.base.specVerifyDims(batchSize, verifyTokenCount, verify.layout);
    bool const captured = mRuntime.base.captureGraph(verifyDims, stream);
    if (!captured)
    {
        LOG_WARNING("DFlashDecoder: failed to capture base verify graph (batch=%d, verifyTokenCount=%d)", batchSize,
            verifyTokenCount);
    }
    return captured;
}

bool DFlashDecoder::checkCudaLastError(char const* stage) const
{
    cudaError_t const err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        LOG_ERROR("DFlashDecoder: CUDA error after %s: %s", stage, cudaGetErrorString(err));
        return false;
    }
    return true;
}

bool DFlashDecoder::captureCudaGraphs(cudaStream_t stream)
{
    bool draftProposalCaptureStatus = captureDraftCudaGraphs(stream);
    bool baseVerificationCaptureStatus{true};

    static constexpr int32_t kSimulateCacheLength{128};
    // A DDTree deployment that can jump also verifies grafting rounds of jumpVerifyTokenCount rows.
    std::vector<int32_t> verifyWidths{mBlockDraft.verifyTokenCount};
    mJump.addGraphTiers(verifyWidths, nullptr);

    // ScopeGuard: reset cache state after capture
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
        // Simulate a cache state with some tokens already committed
        std::vector<int32_t> simCacheLens(batchSize, kSimulateCacheLength);
        Tensor simCacheLensTensor(simCacheLens.data(), {batchSize}, DeviceType::kCPU, trt_edgellm::DataType::kINT32);
        mRuntime.base.cacheManager.resetForNewSequences(simCacheLensTensor, stream);
        mDraftCacheManager.resetForNewSequences(simCacheLensTensor, stream);

        if (!useTreeVerification())
        {
            baseVerificationCaptureStatus &= capturePreparedBaseVerification(batchSize, verifyWidths.front(), stream);
            continue;
        }

        // --- Base verification CUDA graph capture ---
        for (int32_t const verifyTokenCount : verifyWidths)
        {
            int32_t const packedMaskLen = static_cast<int32_t>(divUp(verifyTokenCount, 32));
            int32_t const selectTokenSize = batchSize * verifyTokenCount;
            std::vector<int32_t> idsInput(static_cast<size_t>(selectTokenSize), 0);
            std::vector<int32_t> treeParentIds(static_cast<size_t>(selectTokenSize), -1);
            std::vector<int32_t> treeDepths(static_cast<size_t>(selectTokenSize), 0);
            std::vector<int32_t> positionIds(static_cast<size_t>(selectTokenSize), kSimulateCacheLength);
            std::vector<int64_t> selectTokenIndices(static_cast<size_t>(selectTokenSize), 0);
            std::vector<int32_t> contextLengths(
                static_cast<size_t>(batchSize), kSimulateCacheLength + verifyTokenCount);
            std::vector<int32_t> validCounts(static_cast<size_t>(batchSize), verifyTokenCount);
            std::vector<int32_t> packedAncestorMask(
                static_cast<size_t>(batchSize) * verifyTokenCount * packedMaskLen, 0);

            check::check(mRuntime.base.pipelineIO.specTreeParentIds.reshape({batchSize, verifyTokenCount}),
                "Tensor reshape failed");
            check::check(mRuntime.base.pipelineIO.specTreeDepths.reshape({batchSize, verifyTokenCount}),
                "Tensor reshape failed");
            check::check(mTreeTokenIds.reshape({batchSize, verifyTokenCount}), "Tensor reshape failed");
            check::check(mVerifyTokenIds.reshape({batchSize, verifyTokenCount}), "Tensor reshape failed");
            check::check(mValidCounts.reshape({batchSize}), "Tensor reshape failed");

            for (int32_t batchIdx = 0; batchIdx < batchSize; ++batchIdx)
            {
                int32_t const batchOffset = batchIdx * verifyTokenCount;
                for (int32_t nodeIdx = 0; nodeIdx < verifyTokenCount; ++nodeIdx)
                {
                    int32_t const flatIdx = batchOffset + nodeIdx;
                    selectTokenIndices[flatIdx] = flatIdx;
                    if (nodeIdx > 0)
                    {
                        treeParentIds[flatIdx] = 0;
                        treeDepths[flatIdx] = 1;
                        positionIds[flatIdx] = kSimulateCacheLength + treeDepths[flatIdx];
                        setPackedAncestorBit(packedAncestorMask, batchIdx, nodeIdx, 0, verifyTokenCount);
                    }
                    setPackedAncestorBit(packedAncestorMask, batchIdx, nodeIdx, nodeIdx, verifyTokenCount);
                }
            }

            CUDA_CHECK(cudaMemcpyAsync(mVerifyTokenIds.rawPointer(), idsInput.data(), idsInput.size() * sizeof(int32_t),
                cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaMemcpyAsync(mTreeTokenIds.rawPointer(), idsInput.data(), idsInput.size() * sizeof(int32_t),
                cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaMemcpyAsync(mRuntime.base.pipelineIO.specTreeParentIds.rawPointer(), treeParentIds.data(),
                treeParentIds.size() * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaMemcpyAsync(mRuntime.base.pipelineIO.specTreeDepths.rawPointer(), treeDepths.data(),
                treeDepths.size() * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaMemcpyAsync(mRuntime.base.pipelineIO.specDecodePositionIds.rawPointer(), positionIds.data(),
                positionIds.size() * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(
                cudaMemcpyAsync(mRuntime.base.pipelineIO.selectTokenIndices.rawPointer(), selectTokenIndices.data(),
                    selectTokenIndices.size() * sizeof(int64_t), cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaMemcpyAsync(mRuntime.base.pipelineIO.contextLengths.rawPointer(), contextLengths.data(),
                contextLengths.size() * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaMemcpyAsync(mValidCounts.rawPointer(), validCounts.data(),
                validCounts.size() * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(
                cudaMemcpyAsync(mRuntime.base.pipelineIO.packedAttentionMask.rawPointer(), packedAncestorMask.data(),
                    packedAncestorMask.size() * sizeof(int32_t), cudaMemcpyHostToDevice, stream));

            baseVerificationCaptureStatus &= capturePreparedBaseVerification(batchSize, verifyTokenCount, stream);
        }
    }

    LOG_INFO("DFlashDecoder: CUDA graph capture complete (draft=%s, baseVerify=%s)",
        draftProposalCaptureStatus ? "ok" : "FAILED", baseVerificationCaptureStatus ? "ok" : "FAILED");

    return draftProposalCaptureStatus && baseVerificationCaptureStatus;
}

int64_t DFlashDecoder::getRequiredContextMemorySize() const noexcept
{
    return mDraftExecutor ? mDraftExecutor->getRequiredContextMemorySize() : 0;
}

void DFlashDecoder::setContextMemory(Tensor& memory)
{
    if (mDraftExecutor)
    {
        mDraftExecutor->setContextMemory(memory);
    }
}

void DFlashDecoder::resetForNewSequences(Tensor& reuseLengths, cudaStream_t stream)
{
    mDraftCacheManager.resetForNewSequences(reuseLengths, stream);
    CUDA_CHECK(cudaMemsetAsync(mRuntime.base.pipelineIO.packedAttentionMask.rawPointer(), 0,
        mRuntime.base.pipelineIO.packedAttentionMask.getMemoryCapacity(), stream));
    CUDA_CHECK(cudaMemsetAsync(mRuntime.base.pipelineIO.specDecodePositionIds.rawPointer(), 0,
        mRuntime.base.pipelineIO.specDecodePositionIds.getMemoryCapacity(), stream));
    mCommonStateTracker.reset();
    mDraftPrimed = false;
}

void DFlashDecoder::onBatchEvict(std::vector<int32_t> const& batchMapping, int32_t oldActiveBatch,
    int32_t newActiveBatch, Tensor& deviceBatchMapping, cudaStream_t stream)
{
    ELLM_CHECK(batchMapping.size() == static_cast<size_t>(oldActiveBatch),
        "DFlash batch mapping does not match the old active batch");
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
                         {oldActiveBatch * mAcceptedTokenIds.getShape()[1], mBlockDraft.baseOutputHiddenDim}),
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
        if (isV2() && useTreeVerification())
        {
            // The tree is built from the raw draft outputs when the pending proposal is consumed.
            compactDeviceTensor(mProposalSupportIds);
            compactDeviceTensor(mProposalUnaryValues);
            compactDeviceTensor(mProposalProjectedHidden);
        }
        else if (isV2())
        {
            compactDeviceTensor(mDraftTokenIds);
            compactDeviceTensor(mProposalSupportIds);
            compactDeviceTensor(mProposalSupportProbs);
            compactDeviceTensor(mAcceptUniforms);
        }
        else
        {
            compactDeviceTensor(pendingProposalTensor());
        }
        compactDeviceTensor(mLastAcceptedTokens);
    }
    else
    {
        // Acceptance lengths are first written by base verification, after the pending-prefill proposal is consumed.
        compactDeviceTensor(mAcceptLength);
    }
}

} // namespace rt
} // namespace trt_edgellm
