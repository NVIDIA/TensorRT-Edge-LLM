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
#include "kernels/embeddingKernels/embeddingKernels.h"
#include "kernels/gdnKernels/gdnTreeChunkKernels.h"
#include "kernels/posEncoding/applyRopeWriteKV.h"
#include "kernels/speculative/ddtreeKernels.h"
#include "kernels/speculative/dflashRuntimeKernels.h"
#include "kernels/speculative/eagleAcceptKernels.h"
#include "kernels/speculative/eagleUtilKernels.h"
#include "profiling/metrics.h"
#include "profiling/nvtx_wrapper.h"
#include "profiling/timer.h"
#include "runtime/config/llmEngineConfig.h"
#include "runtime/decoding/decoderUtils.h"
#include "runtime/decoding/dflashDecodeUtils.h"
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

int32_t dflashDraftProfileForRound(int32_t generationRound)
{
    return generationRound == 0 ? kPrefillProfile : kDecodeProfile;
}

void setPackedAncestorBit(
    std::vector<int32_t>& packedMask, int32_t batchIdx, int32_t nodeIdx, int32_t ancestorIdx, int32_t verifySize)
{
    constexpr int32_t kMaskBitsPerWord{32};
    int32_t const packedMaskLen = static_cast<int32_t>(divUp(verifySize, kMaskBitsPerWord));
    int32_t const wordIdx = ancestorIdx / kMaskBitsPerWord;
    int32_t const bitIdx = ancestorIdx % kMaskBitsPerWord;
    int64_t const offset = static_cast<int64_t>(batchIdx) * verifySize * packedMaskLen
        + static_cast<int64_t>(nodeIdx) * packedMaskLen + wordIdx;
    packedMask[offset] |= static_cast<int32_t>(1U << bitIdx);
}
} // namespace

DFlashDecoder::DFlashDecoder(DecodingRuntimeContext& runtime, std::filesystem::path const& engineDir,
    SpecDecodeDraftingConfig const& /* draftingConfig */, cudaStream_t stream)
    : mRuntime(runtime)
    , mDraftCacheManager(*runtime.base.sharedResources.cacheManagers[1])
{
    auto const& deployment = runtime.deployment;
    auto const& baseCfg = deployment.base;
    ELLM_CHECK(deployment.specConfig.has_value(), "DFlashDecoder: specConfig is required.");
    ELLM_CHECK(deployment.draft.has_value(), "DFlashDecoder: draft config is required.");
    ELLM_CHECK(baseCfg.specDecodeType == SpecDecodeMode::kDFlash,
        "DFlashDecoder requires a base engine exported with spec_decode_type=dflash and engine_role=base.");
    ELLM_CHECK(baseCfg.reducedVocabSize == 0, "DFlash does not support reduced-vocabulary base engines.");

    mBlockSize = dflash_utils::runtimeBlockSize(deployment);
    mUseDDTree = dflash_utils::shouldUseDDTree(deployment);
    mVerifySize = mUseDDTree ? deployment.specConfig->verifySize : mBlockSize;
    mCandidateTopK = deployment.specConfig->draftingTopK;
    ELLM_CHECK(mCandidateTopK >= 1, "DFlashDecoder requires draftingTopK >= 1.");
    if (mUseDDTree)
    {
        ELLM_CHECK(mCandidateTopK <= kernel::kDDTreeMaxCandidateTopK,
            "DFlashDecoder DDTree draftingTopK exceeds the current candidateTopK limit.");
    }
    // Draft's mask_token_id is authoritative: the draft was trained expecting that
    // specific mask id. Base config may carry an uninitialized exporter default
    // (e.g. 248070 > vocab_size on Qwen3-8B) which, if fed to the draft engine,
    // produces out-of-vocab embeddings and degenerate proposals. Prefer the draft's
    // value and only fall back to base when the draft doesn't specify one.
    mMaskTokenId
        = deployment.draft->dflashMaskTokenId > 0 ? deployment.draft->dflashMaskTokenId : baseCfg.dflashMaskTokenId;
    mDraftHiddenSize = deployment.specConfig->draftHiddenSize;
    mBaseOutputHiddenDim = deployment.specConfig->baseOutputHiddenDim;
    mDraftVocabSize = deployment.draft->outputVocabSize;
    ELLM_CHECK(mMaskTokenId >= 0 && mMaskTokenId < mDraftVocabSize,
        "DFlashDecoder: mask_token_id (" + std::to_string(mMaskTokenId) + ") out of draft vocab range [0,"
            + std::to_string(mDraftVocabSize) + ").");

    int32_t const maxBatch = deployment.maxRuntimeBatchSize();

    auto const draftEnginePath = engineDir / "spec_draft.engine";
    LOG_INFO("DFlashDecoder: loading draft engine from %s", draftEnginePath.string().c_str());
    mDraftExecutor = EngineExecutor::createForDraft(draftEnginePath, deployment);
    validateAgainstEngine(*deployment.draft, *mDraftExecutor, "dflash_draft");

    mDraftInputsEmbeds = Tensor({maxBatch, mBlockSize, mDraftHiddenSize}, DeviceType::kGPU, nvinfer1::DataType::kHALF,
        "DFlashDraft::inputsEmbeds");
    mDraftTargetHidden = Tensor({maxBatch, mBlockSize, mBaseOutputHiddenDim}, DeviceType::kGPU,
        nvinfer1::DataType::kHALF, "DFlashDraft::targetHiddenScratch");
    mDraftOutputLogits = Tensor({maxBatch, mBlockSize, mDraftVocabSize}, DeviceType::kGPU, nvinfer1::DataType::kFLOAT,
        "DFlashDraft::outputLogits");

    int32_t const packedMaskLen = divUp(mBlockSize, 32);
    mDraftPackedAttentionMask = Tensor(
        {maxBatch, mBlockSize, packedMaskLen}, DeviceType::kGPU, nvinfer1::DataType::kINT32, "DFlashDraft::packedMask");
    mDraftAttentionPosId
        = Tensor({maxBatch, mBlockSize}, DeviceType::kGPU, nvinfer1::DataType::kINT32, "DFlashDraft::positionIds");
    mDraftContextLengths
        = Tensor({maxBatch}, DeviceType::kGPU, nvinfer1::DataType::kINT32, "DFlashDraft::contextLengths");
    mDraftDeltaLenCommit
        = Tensor({maxBatch}, DeviceType::kGPU, nvinfer1::DataType::kINT32, "DFlashDraft::deltaLenCommit");
    mDraftDeltaLens = Tensor({maxBatch}, DeviceType::kGPU, nvinfer1::DataType::kINT32, "DFlashDraft::deltaLens");

    mDraftTensorMap.set(binding_names::kInputsEmbeds, mDraftInputsEmbeds);
    mDraftTensorMap.set(binding_names::kDFlashTargetHiddenConcat, mRuntime.base.pipelineIO.baseHiddenStates);
    mDraftTensorMap.set(binding_names::kLogits, mDraftOutputLogits);
    mDraftTensorMap.set(binding_names::kAttentionMask, mDraftPackedAttentionMask);
    mDraftTensorMap.set(binding_names::kAttentionPosId, mDraftAttentionPosId);
    mDraftTensorMap.set(binding_names::kContextLengths, mDraftContextLengths);
    mDraftTensorMap.set(binding_names::kDFlashDeltaLengths, mDraftDeltaLens);

    // KV cache bindings: bind to draft cache manager's combined KV cache (index 1). Unified on
    // the paged-pool view — the engine's past/present_key_values_i binding for DFlash's own draft
    // cache is the same [2, numPages, kTOKENS_PER_PAGE, numKVHeads, headDim] contract as the main
    // model and EAGLE/MTP drafts (see KVCacheManager::getCombinedKVCachePoolView()).
    auto& kvMgr = mDraftCacheManager.getKVCacheManager();
    LLMEngineConfig const& draftCfg = *deployment.draft;
    int32_t localAttnIdx = 0;
    for (int32_t absIdx = 0; absIdx < static_cast<int32_t>(draftCfg.layerTypes.size()); ++absIdx)
    {
        if (draftCfg.layerTypes[absIdx] != HybridCacheManager::LayerType::kAttention)
        {
            continue;
        }
        auto& combinedKV = kvMgr.getCombinedKVCachePoolView(localAttnIdx);
        mDraftTensorMap.set(binding_names::formatKVCacheName(localAttnIdx, /*isPast=*/true), combinedKV);
        mDraftTensorMap.set(binding_names::formatKVCacheName(localAttnIdx, /*isPast=*/false), combinedKV);
        ++localAttnIdx;
    }
    mDraftTensorMap.set(binding_names::kKVCacheStartIndex, mDraftCacheManager.getKVCacheLengths());

    // kv_page_table: static identity mapping for the draft's proposal self-attention
    // (shared resource index 1) — see SharedResources::kvPageTables.
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

    mDraftTokenIds
        = Tensor({maxBatch, mBlockSize}, DeviceType::kGPU, nvinfer1::DataType::kINT32, "DFlashDraft::tokenIds");
    mHostDraftInputIds
        = Tensor({maxBatch, mBlockSize}, DeviceType::kCPU, nvinfer1::DataType::kINT32, "DFlashDraft::hostInputIds");
    mHostLastAcceptedTokens
        = Tensor({maxBatch}, DeviceType::kCPU, nvinfer1::DataType::kINT32, "DFlashDraft::hostLastAcceptedTokens");
    mHostDeltaLens = Tensor({maxBatch}, DeviceType::kCPU, nvinfer1::DataType::kINT32, "DFlashDraft::hostDeltaLens");
    mLastAcceptedTokens
        = Tensor({maxBatch}, DeviceType::kGPU, nvinfer1::DataType::kINT32, "DFlashDraft::lastAcceptedTokens");

    mTreeTokenIds
        = Tensor({maxBatch, mVerifySize}, DeviceType::kGPU, nvinfer1::DataType::kINT32, "DFlash::treeTokenIds");
    mTreeNodeScores
        = Tensor({maxBatch, mVerifySize}, DeviceType::kGPU, nvinfer1::DataType::kFLOAT, "DFlash::treeNodeScores");
    mValidCounts = Tensor({maxBatch}, DeviceType::kGPU, nvinfer1::DataType::kINT32, "DFlash::validCounts");
    mVerifyTokenIds
        = Tensor({maxBatch, mVerifySize}, DeviceType::kGPU, nvinfer1::DataType::kINT32, "DFlash::verifyTokenIds");
    mVerifyTreeMask = Tensor(
        {maxBatch, mVerifySize, mVerifySize}, DeviceType::kGPU, nvinfer1::DataType::kINT8, "DFlash::verifyTreeMask");
    mAcceptedTokenIds
        = Tensor({maxBatch, mBlockSize}, DeviceType::kGPU, nvinfer1::DataType::kINT32, "DFlash::acceptedTokenIds");
    mAcceptedTokenIndices
        = Tensor({maxBatch, mBlockSize}, DeviceType::kGPU, nvinfer1::DataType::kINT32, "DFlash::acceptedTokenIndices");
    mAcceptLength = Tensor({maxBatch}, DeviceType::kGPU, nvinfer1::DataType::kINT32, "DFlash::acceptLength");
    mHostAcceptLengths = Tensor({maxBatch}, DeviceType::kCPU, nvinfer1::DataType::kINT32, "DFlash::hostAcceptLengths");
    mHostAcceptedTokenIds
        = Tensor({maxBatch, mBlockSize}, DeviceType::kCPU, nvinfer1::DataType::kINT32, "DFlash::hostAcceptedIds");

    size_t const buildWorkspaceSize = mUseDDTree ? kernel::getDDTreeBuildWorkspaceSize(maxBatch, mBlockSize,
                                                       mVerifySize, deployment.draft->outputVocabSize, mCandidateTopK)
                                                 : 1U;
    ELLM_CHECK(buildWorkspaceSize > 0, "DFlashDecoder: DDTree build workspace size must be > 0.");
    mBuildWorkspace = Tensor({static_cast<int64_t>(buildWorkspaceSize)}, DeviceType::kGPU, nvinfer1::DataType::kUINT8,
        "DFlash::buildWorkspace");

    LOG_INFO(
        "DFlashDecoder initialized: mode=%s, blockSize=%d, verifySize=%d, candidateTopK=%d, maskTokenId=%d, "
        "maxBatch=%d, draftHiddenSize=%d, baseOutputHiddenDim=%d, draftVocabSize=%d",
        mUseDDTree ? "branching-tree" : "linear-tree", mBlockSize, mVerifySize, mCandidateTopK, mMaskTokenId, maxBatch,
        mDraftHiddenSize, mBaseOutputHiddenDim, mDraftVocabSize);

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
        check::check(vocabMapTensors[0].getShape()[0] == mDraftVocabSize,
            "draft vocab_map tensor length should match draft model reduced vocab size");
        mDraftVocabMappingTable = std::move(vocabMapTensors[0]);
        mHasDraftVocabMap = true;
        LOG_INFO("DFlashDecoder: draft vocab map loaded (%d reduced -> full vocab tokens)",
            static_cast<int32_t>(mDraftVocabMappingTable.getShape()[0]));
    }
}

bool DFlashDecoder::decodeStep(DecodingInferenceContext& context)
{
    NVTX_SCOPED_RANGE(nvtx_dflash_decode, "DFlashDecoder::decodeStep", nvtx_colors::GREEN);
    cudaGetLastError();

    if (!runDraftForward(context))
    {
        LOG_ERROR("DFlashDecoder: draft forward failed.");
        return false;
    }
    if (!prepareDFlashVerifyInputs(context))
    {
        LOG_ERROR("DFlashDecoder: verify input preparation failed.");
        return false;
    }

    if (!runBaseVerification(context))
    {
        LOG_ERROR("DFlashDecoder: base verification failed.");
        return false;
    }

    return true;
}

bool DFlashDecoder::runDraftForward(DecodingInferenceContext& context)
{
    TIME_STAGE(metrics::StageNames::kSPEC_DECODE_DRAFT_PROPOSAL, context.stream);
    NVTX_SCOPED_RANGE(nvtx_dflash_draft, "DFlashDecoder::runDraftForward", nvtx_colors::DARK_ORANGE);

    if (!mDraftExecutor)
    {
        LOG_ERROR("DFlashDecoder: draft engine not loaded.");
        return false;
    }

    int32_t const activeBatchSize = context.activeBatchSize;
    int32_t const BS = mBlockSize;

    // Step 1: Prepare draft input token IDs: [last_accepted_token, mask_id, ..., mask_id].
    check::check(mRuntime.preprocess.idsInput.reshape({activeBatchSize, BS}), "Tensor reshape failed");
    check::check(mHostDraftInputIds.reshape({activeBatchSize, BS}), "Tensor reshape failed");
    check::check(mHostLastAcceptedTokens.reshape({activeBatchSize}), "Tensor reshape failed");
    check::check(mLastAcceptedTokens.reshape({activeBatchSize}), "Tensor reshape failed");
    int32_t* hostDraftInputIds = mHostDraftInputIds.dataPointer<int32_t>();
    int32_t* hostLastAccepted = mHostLastAcceptedTokens.dataPointer<int32_t>();
    for (int32_t b = 0; b < activeBatchSize; ++b)
    {
        hostLastAccepted[b] = context.tokenIds[b].back();
        hostDraftInputIds[b * BS] = hostLastAccepted[b];
        for (int32_t j = 1; j < BS; ++j)
        {
            hostDraftInputIds[b * BS + j] = mMaskTokenId;
        }
    }
    CUDA_CHECK(cudaMemcpyAsync(mRuntime.preprocess.idsInput.rawPointer(), mHostDraftInputIds.rawPointer(),
        activeBatchSize * BS * sizeof(int32_t), cudaMemcpyHostToDevice, context.stream));
    CUDA_CHECK(cudaMemcpyAsync(mLastAcceptedTokens.rawPointer(), mHostLastAcceptedTokens.rawPointer(),
        activeBatchSize * sizeof(int32_t), cudaMemcpyHostToDevice, context.stream));

    // Step 2: Embed the draft inputs.
    check::check(mDraftInputsEmbeds.reshape({activeBatchSize, BS, mDraftHiddenSize}), "Tensor reshape failed");
    kernel::embeddingLookup(mRuntime.preprocess.idsInput, mRuntime.preprocess.embedding.table,
        mRuntime.preprocess.embedding.scalesAsOptional(), mDraftInputsEmbeds, context.stream);

    // Step 3: Prepare target-hidden delta from base hidden states.
    // Round 0 uses prefill hidden states; later rounds use accepted base hidden states.
    // The draft KV update plugin consumes per-batch delta_lengths to skip padded rows.
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
        sourceSeqLen = mRuntime.base.pipelineIO.baseHiddenStates.getShape()[1];
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

    auto const compactTargetHidden = [&](Tensor& targetHidden) {
        check::check(
            targetHidden.reshape({activeBatchSize, maxDeltaLen, mBaseOutputHiddenDim}), "Tensor reshape failed");
        size_t const elementBytes = utils::getTypeSize(targetHidden.getDataType());
        size_t const rowBytes = static_cast<size_t>(mBaseOutputHiddenDim) * elementBytes;
        size_t const dstPitch = static_cast<size_t>(maxDeltaLen) * rowBytes;
        size_t const srcPitch = static_cast<size_t>(sourceSeqLen) * rowBytes;
        size_t const widthBytes = static_cast<size_t>(maxDeltaLen) * rowBytes;
        CUDA_CHECK(cudaMemcpy2DAsync(targetHidden.rawPointer(), dstPitch,
            mRuntime.base.pipelineIO.baseHiddenStates.rawPointer(), srcPitch, widthBytes, activeBatchSize,
            cudaMemcpyDeviceToDevice, context.stream));
        mDraftTensorMap.set(binding_names::kDFlashTargetHiddenConcat, targetHidden);
    };

    // TensorRT reads dflash_target_hidden_concat as a compact [B, selectLen, H]
    // tensor. Bind baseHiddenStates directly only when its batch stride already
    // matches selectLen; otherwise compact into a scratch buffer first.
    if (context.generationRound == 0 && sourceSeqLen == maxDeltaLen)
    {
        // This intentionally narrows baseHiddenStates to the compact draft binding shape. The base runner reshapes
        // and rebinds it before the next base-engine enqueue.
        check::check(
            mRuntime.base.pipelineIO.baseHiddenStates.reshape({activeBatchSize, maxDeltaLen, mBaseOutputHiddenDim}),
            "Tensor reshape failed");
        mDraftTensorMap.set(binding_names::kDFlashTargetHiddenConcat, mRuntime.base.pipelineIO.baseHiddenStates);
    }
    else
    {
        check::check(context.generationRound == 0 || maxDeltaLen <= mBlockSize,
            "DFlash decode target-hidden delta exceeds block-size scratch.");
        if (maxDeltaLen <= mBlockSize)
        {
            compactTargetHidden(mDraftTargetHidden);
        }
        else
        {
            // This is only for an uncommon round-0 layout where baseHiddenStates
            // is wider than the active max prefill length. Keep it lazy so the
            // normal DFlash path does not pay a max-input/max-KV allocation.
            int32_t const reserveBatchSize = mRuntime.maxRuntimeBatchSize;
            int64_t const requiredBytes = static_cast<int64_t>(reserveBatchSize) * maxDeltaLen * mBaseOutputHiddenDim
                * static_cast<int64_t>(utils::getTypeSize(nvinfer1::DataType::kHALF));
            if (mDraftPrefillTargetHidden.getMemoryCapacity() < requiredBytes)
            {
                mDraftPrefillTargetHidden = Tensor{};
                mDraftPrefillTargetHidden = Tensor({reserveBatchSize, maxDeltaLen, mBaseOutputHiddenDim},
                    DeviceType::kGPU, nvinfer1::DataType::kHALF, "DFlashDraft::prefillTargetHiddenScratch");
            }
            compactTargetHidden(mDraftPrefillTargetHidden);
        }
    }

    // Step 4: Prepare proposal attention inputs.
    int32_t const pmLen = divUp(BS, 32);
    check::check(mDraftPackedAttentionMask.reshape({activeBatchSize, BS, pmLen}), "Tensor reshape failed");
    check::check(mDraftAttentionPosId.reshape({activeBatchSize, BS}), "Tensor reshape failed");
    check::check(mDraftContextLengths.reshape({activeBatchSize}), "Tensor reshape failed");
    Tensor const& draftCacheLengths = mDraftCacheManager.getKVCacheLengths();
    kernel::launchDFlashPrepareProposalInputs(draftCacheLengths.dataPointer<int32_t>(),
        mDraftDeltaLens.dataPointer<int32_t>(), BS, mDraftPackedAttentionMask.dataPointer<int32_t>(),
        mDraftAttentionPosId.dataPointer<int32_t>(), mDraftContextLengths.dataPointer<int32_t>(), activeBatchSize,
        context.stream);

    // Step 5: Execute the DFlash draft engine.
    check::check(mDraftOutputLogits.reshape({activeBatchSize, BS, mDraftVocabSize}), "Tensor reshape failed");
    int32_t const draftKVCapacity = mRuntime.deployment.draft->maxKVCacheCapacity;
    InferenceDims const draftDims{
        /*.batch=*/activeBatchSize,
        /*.seqLen=*/BS,
        /*.kvLen=*/draftKVCapacity,
        /*.selectLen=*/static_cast<int64_t>(maxDeltaLen),
        /*.attnMaskSeqLen=*/BS,
        /*.ropeBatch=*/1,
        /*.packedMaskLen=*/static_cast<int64_t>(pmLen),
        /*.startIndexLen=*/activeBatchSize,
        /*.specVerifyPhaseLen=*/0,
    };

    bool draftSuccess = mDraftExecutor->prepare(
        dflashDraftProfileForRound(context.generationRound), draftDims, mDraftTensorMap, context.stream);
    if (draftSuccess)
    {
        draftSuccess = mDraftExecutor->execute(context.stream);
    }
    if (!draftSuccess)
    {
        LOG_ERROR("DFlashDecoder: draft engine execution failed.");
        return false;
    }

    // Step 6: Commit per-batch delta lengths to the draft cache manager.
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

    // The DFlashTargetKVCacheUpdate plugin has no page-table input -- it always
    // writes at slot b's own contiguous row (DFlash is identity-only: it opts out of reuse).
    // Assert that invariant against the draft cache manager's REAL page table (the same object
    // bound to the draft AttentionPlugin layers, see kKVPageTable above) rather than just
    // documenting it: HybridCacheManager::compactBatch preserves row == slot for the draft cache
    // today, so this should always pass, but if that ever regresses (or non-identity draft reuse
    // is ever introduced) DFlash would otherwise silently write through the wrong physical slot.
    // `hostRow()` reads the host mirror directly -- no device sync needed.
    try
    {
        auto const& draftPageTable = *mRuntime.base.sharedResources.kvPageTables[1];
        int32_t const draftMaxPagesPerSeq = draftPageTable.maxPagesPerSeq();
        for (int32_t b = 0; b < activeBatchSize; ++b)
        {
            kernel::checkDFlashPageTableIdentity(draftPageTable.hostRow(b), b, draftMaxPagesPerSeq);
        }
    }
    catch (std::exception const& e)
    {
        LOG_ERROR(
            "DFlashDecoder: draft page table is not identity-mapped (%s); DFlashTargetKVCacheUpdate is "
            "identity-only and would silently write through the wrong physical slot. Failing the request.",
            e.what());
        return false;
    }

    return true;
}

bool DFlashDecoder::prepareDFlashVerifyInputs(DecodingInferenceContext& context)
{
    NVTX_SCOPED_RANGE(
        nvtx_dflash_prepare_verify, "DFlashDecoder::prepareDFlashVerifyInputs", nvtx_colors::LIGHT_ORANGE);

    int32_t const activeBatchSize = context.activeBatchSize;
    int32_t const verifySize = mUseDDTree ? mVerifySize : mBlockSize;
    check::check(mVerifyTokenIds.reshape({activeBatchSize, verifySize}), "Tensor reshape failed");
    check::check(mVerifyTreeMask.reshape({activeBatchSize, verifySize, verifySize}), "Tensor reshape failed");

    if (!mUseDDTree)
    {
        // Linear DFlash uses candidateTopK == 1: select one draft token per block
        // position, then build a causal verify tree in the shared verify buffers.
        check::check(
            mDraftOutputLogits.reshape({activeBatchSize * mBlockSize, mDraftVocabSize}), "Tensor reshape failed");
        check::check(mDraftTokenIds.reshape({activeBatchSize * mBlockSize, 1}), "Tensor reshape failed");
        selectAllTopK(mDraftOutputLogits, std::nullopt, mDraftTokenIds, 1, mRuntime.sampling.workspace, context.stream);
        check::check(mDraftTokenIds.reshape({activeBatchSize, mBlockSize}), "Tensor reshape failed");

        // Remap reduced-vocab draft IDs to full-vocab IDs before base verify.
        if (mHasDraftVocabMap)
        {
            check::check(mDraftTokenIds.reshape({activeBatchSize * mBlockSize}), "Tensor reshape failed");
            mapReducedVocabToFullVocab(mDraftTokenIds, mDraftVocabMappingTable, context.stream);
            check::check(mDraftTokenIds.reshape({activeBatchSize, mBlockSize}), "Tensor reshape failed");
        }

        kernel::launchDFlashBuildLinearVerifyInputs(mLastAcceptedTokens.dataPointer<int32_t>(),
            mDraftTokenIds.dataPointer<int32_t>(), mVerifyTokenIds.dataPointer<int32_t>(),
            mVerifyTreeMask.dataPointer<int8_t>(), activeBatchSize, mBlockSize, context.stream);
        prepareLinearBaseVerificationMetadata(activeBatchSize, verifySize, context.stream);
    }
    else if (!buildTreeVerifyInputs(context))
    {
        return false;
    }

    copyVerifyTokenIdsToBaseInput(activeBatchSize, verifySize, context.stream);
    if (!checkCudaLastError("prepare DFlash verify inputs"))
    {
        return false;
    }
    return true;
}

bool DFlashDecoder::buildTreeVerifyInputs(DecodingInferenceContext& context)
{
    TIME_STAGE(metrics::StageNames::kSPEC_DECODE_DRAFT_PROPOSAL, context.stream);
    NVTX_SCOPED_RANGE(nvtx_dflash_ddtree_build, "DFlashDecoder::buildTreeVerifyInputs", nvtx_colors::LIGHT_ORANGE);

    int32_t const activeBatchSize = context.activeBatchSize;
    int32_t const verifySize = mVerifySize;
    check::check(mTreeTokenIds.reshape({activeBatchSize, verifySize}), "Tensor reshape failed");
    check::check(mTreeNodeScores.reshape({activeBatchSize, verifySize}), "Tensor reshape failed");
    check::check(mValidCounts.reshape({activeBatchSize}), "Tensor reshape failed");
    check::check(
        mRuntime.base.pipelineIO.specTreeParentIds.reshape({activeBatchSize, verifySize}), "Tensor reshape failed");
    check::check(
        mRuntime.base.pipelineIO.specTreeDepths.reshape({activeBatchSize, verifySize}), "Tensor reshape failed");
    check::check(mVerifyTokenIds.reshape({activeBatchSize, verifySize}), "Tensor reshape failed");
    check::check(
        mRuntime.base.pipelineIO.specDecodePositionIds.reshape({activeBatchSize, verifySize}), "Tensor reshape failed");
    check::check(mRuntime.base.pipelineIO.packedAttentionMask.reshape(
                     {activeBatchSize, verifySize, static_cast<int64_t>(divUp(verifySize, 32))}),
        "Tensor reshape failed");
    check::check(mVerifyTreeMask.reshape({activeBatchSize, verifySize, verifySize}), "Tensor reshape failed");
    check::check(mRuntime.base.pipelineIO.contextLengths.reshape({activeBatchSize}), "Tensor reshape failed");
    check::check(
        mRuntime.base.pipelineIO.selectTokenIndices.reshape({activeBatchSize, verifySize}), "Tensor reshape failed");

    Tensor const& baseKVCacheLengths = mRuntime.base.cacheManager.getKVCacheLengths();
    kernel::DDTreeBuildParams const buildParams{{mDraftOutputLogits, mLastAcceptedTokens, baseKVCacheLengths,
                                                    mHasDraftVocabMap ? &mDraftVocabMappingTable : nullptr},
        {mTreeTokenIds, mRuntime.base.pipelineIO.specTreeDepths, mRuntime.base.pipelineIO.specTreeParentIds,
            mTreeNodeScores, mValidCounts, mVerifyTokenIds, mRuntime.base.pipelineIO.specDecodePositionIds,
            mRuntime.base.pipelineIO.packedAttentionMask, mVerifyTreeMask, mRuntime.base.pipelineIO.contextLengths,
            mRuntime.base.pipelineIO.selectTokenIndices},
        mCandidateTopK, mBuildWorkspace.rawPointer(), static_cast<size_t>(mBuildWorkspace.getMemoryCapacity()),
        context.stream};
    kernel::ddtreeBuild(buildParams);

    return true;
}

bool DFlashDecoder::captureDraftCudaGraphs(cudaStream_t stream)
{
    bool draftProposalCaptureStatus{true};

    static constexpr int32_t kSimulateCacheLength{128};
    int32_t const BS = mBlockSize;
    int32_t const draftKVCapacity = mRuntime.deployment.draft->maxKVCacheCapacity;

    for (int32_t batchSize = 1; batchSize <= mRuntime.maxRuntimeBatchSize; ++batchSize)
    {
        std::vector<int32_t> simCacheLens(batchSize, kSimulateCacheLength);
        Tensor simCacheLensTensor(simCacheLens.data(), {batchSize}, DeviceType::kCPU, nvinfer1::DataType::kINT32);
        mDraftCacheManager.resetForNewSequences(simCacheLensTensor, stream);

        int32_t const pmLen = divUp(BS, 32);
        for (int32_t simDeltaLen = 1; simDeltaLen <= BS; ++simDeltaLen)
        {
            check::check(mDraftInputsEmbeds.reshape({batchSize, BS, mDraftHiddenSize}), "Tensor reshape failed");
            check::check(
                mDraftTargetHidden.reshape({batchSize, static_cast<int64_t>(simDeltaLen), mBaseOutputHiddenDim}),
                "Tensor reshape failed");
            mDraftTensorMap.set(binding_names::kDFlashTargetHiddenConcat, mDraftTargetHidden);
            check::check(mDraftOutputLogits.reshape({batchSize, BS, mDraftVocabSize}), "Tensor reshape failed");
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
                mDraftAttentionPosId.dataPointer<int32_t>(), mDraftContextLengths.dataPointer<int32_t>(), batchSize,
                stream);

            InferenceDims const draftDims{
                /*.batch=*/batchSize,
                /*.seqLen=*/BS,
                /*.kvLen=*/draftKVCapacity,
                /*.selectLen=*/static_cast<int64_t>(simDeltaLen),
                /*.attnMaskSeqLen=*/BS,
                /*.ropeBatch=*/1,
                /*.packedMaskLen=*/static_cast<int64_t>(pmLen),
                /*.startIndexLen=*/batchSize,
                /*.specVerifyPhaseLen=*/0,
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
    int32_t const BS = mBlockSize;
    int32_t const verifySize = mUseDDTree ? mVerifySize : BS;
    int32_t const maxAcceptLength = mUseDDTree ? std::min(BS, verifySize) : BS;

    cudaGetLastError();
    bool const verifySuccess = executeBaseVerification(context, verifySize);
    if (!verifySuccess)
    {
        return false;
    }

    check::check(mAcceptedTokenIds.reshape({activeBatchSize, maxAcceptLength}), "Tensor reshape failed");
    check::check(mAcceptedTokenIndices.reshape({activeBatchSize, maxAcceptLength}), "Tensor reshape failed");
    check::check(mAcceptLength.reshape({activeBatchSize}), "Tensor reshape failed");
    check::check(mRuntime.base.pipelineIO.outputLogits.reshape(
                     {activeBatchSize * verifySize, mRuntime.deployment.base.outputVocabSize}),
        "Tensor reshape failed");

    Tensor const& acceptTokenIds = mUseDDTree ? mTreeTokenIds : mVerifyTokenIds;
    // DFlash reuses the EAGLE accept utility for both linear-tree and branching-tree verification:
    // Step 1: compute base top-1 tokens for every verify node from base logits.
    // Step 2: accept the selected token and record the verify-node index.
    // Step 3: follow the next matching child through the verify tree mask.
    kernel::eagleAccept(mRuntime.base.pipelineIO.outputLogits, acceptTokenIds, mVerifyTreeMask, mAcceptedTokenIds,
        mAcceptedTokenIndices, mAcceptLength, std::nullopt, mRuntime.sampling.workspace.rawPointer(),
        mRuntime.sampling.workspace.getMemoryCapacity(), context.stream);

    if (mUseDDTree)
    {
        commitAcceptedTreePath(context, verifySize, maxAcceptLength);
    }
    else
    {
        mRuntime.base.cacheManager.commitSequenceLength(mAcceptLength, context.stream);

        check::check(mRuntime.base.pipelineIO.baseHiddenStates.reshape({activeBatchSize, BS, mBaseOutputHiddenDim}),
            "Tensor reshape failed");

        auto& mambaMgr = mRuntime.base.cacheManager.getMambaCacheManager();
        if (mambaMgr.hasIntermediateRecurrentStates() || mambaMgr.hasIntermediateConvStates())
        {
            check::check(kernel::gdnTreeChunkVerifyEnabled(verifySize),
                "Hybrid DFlash requires compact GDN speculative verification.");
            mambaMgr.replayCommitAcceptedTreeStates(mAcceptedTokenIndices, mAcceptLength, context.stream);
        }
    }

    // Enqueue logprobs device work + D2H before appendAcceptedTokens so everything rides
    // that call's single round synchronization. Verify rows are tree nodes (DDTree) or linear
    // block positions; either way the accepted rows can be non-contiguous in outputLogits, so
    // gather them via the accepted verify indices first (same as EAGLE/MTP).
    if (context.numLogprobs > 0)
    {
        int32_t const vocabSize = mRuntime.deployment.base.outputVocabSize;
        int32_t const gatheredRows = activeBatchSize * maxAcceptLength;
        check::check(mRuntime.logprobs.gatheredLogits.reshape({gatheredRows, vocabSize}), "Tensor reshape failed");
        gatherSpecVerifyAcceptedLogitRows(mRuntime.base.pipelineIO.outputLogits, mAcceptedTokenIndices,
            mRuntime.logprobs.gatheredLogits, activeBatchSize, verifySize, maxAcceptLength, vocabSize, context.stream);
        decoder_utils::enqueueLogprobsD2H(
            mRuntime.logprobs.gatheredLogits, gatheredRows, mRuntime, context.numLogprobs, context.stream);
    }

    // Step 8: Append accepted tokens to context (includes the round's D2H sync)
    decoder_utils::appendAcceptedTokens(context, mHostAcceptLengths, mHostAcceptedTokenIds, mAcceptLength,
        mAcceptedTokenIds, maxAcceptLength, mRuntime.tokenizer, context.stream);

    if (context.numLogprobs > 0)
    {
        decoder_utils::collectSpecLogprobsFromHost(mRuntime, context, activeBatchSize, maxAcceptLength,
            mHostAcceptLengths.dataPointer<int32_t>(), context.numLogprobs);
    }

    return true;
}

bool DFlashDecoder::executeBaseVerification(DecodingInferenceContext& context, int32_t verifySize)
{
    int32_t const activeBatchSize = context.activeBatchSize;
    reshapeBaseVerificationInputsOutputs(activeBatchSize, verifySize);
    runBaseVerificationEmbeddingLookup(activeBatchSize, verifySize, context.stream, /*reshapeGemmaPleOutputs=*/false);
    prepareCommonBaseVerificationInputs(activeBatchSize, verifySize);

    auto const verifyDims = mRuntime.deployment.base.specVerifyDims(activeBatchSize, verifySize);
    bool verifySuccess
        = mRuntime.base.executor.prepare(kDecodeProfile, verifyDims, mRuntime.base.tensorMap, context.stream);
    if (verifySuccess)
    {
        verifySuccess = mRuntime.base.execute(context.stream);
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

void DFlashDecoder::copyVerifyTokenIdsToBaseInput(int32_t batchSize, int32_t verifySize, cudaStream_t stream)
{
    check::check(mRuntime.preprocess.idsInput.reshape({batchSize, verifySize}), "Tensor reshape failed");
    CUDA_CHECK(cudaMemcpyAsync(mRuntime.preprocess.idsInput.rawPointer(), mVerifyTokenIds.rawPointer(),
        static_cast<size_t>(batchSize) * verifySize * sizeof(int32_t), cudaMemcpyDeviceToDevice, stream));
}

void DFlashDecoder::reshapeBaseVerificationForCapture(int32_t batchSize, int32_t verifySize, bool includeTreeMetadata)
{
    check::check(mRuntime.preprocess.idsInput.reshape({batchSize, verifySize}), "Tensor reshape failed");
    reshapeBaseVerificationInputsOutputs(batchSize, verifySize);
    if (includeTreeMetadata)
    {
        check::check(
            mRuntime.base.pipelineIO.specTreeParentIds.reshape({batchSize, verifySize}), "Tensor reshape failed");
        check::check(mRuntime.base.pipelineIO.specTreeDepths.reshape({batchSize, verifySize}), "Tensor reshape failed");
    }
}

void DFlashDecoder::prepareLinearBaseVerificationMetadata(int32_t batchSize, int32_t verifySize, cudaStream_t stream)
{
    check::check(mRuntime.base.pipelineIO.packedAttentionMask.reshape(
                     {batchSize, verifySize, static_cast<int64_t>(divUp(verifySize, 32))}),
        "Tensor reshape failed");
    check::check(mRuntime.base.pipelineIO.selectTokenIndices.reshape({batchSize, verifySize}), "Tensor reshape failed");
    check::check(mRuntime.base.pipelineIO.contextLengths.reshape({batchSize}), "Tensor reshape failed");
    check::check(
        mRuntime.base.pipelineIO.specDecodePositionIds.reshape({batchSize, verifySize}), "Tensor reshape failed");

    Tensor const& baseKVCacheLengths = mRuntime.base.cacheManager.getKVCacheLengths();
    kernel::launchDFlashPrepareBaseVerifyInputs(baseKVCacheLengths.dataPointer<int32_t>(), verifySize,
        mRuntime.base.pipelineIO.packedAttentionMask.dataPointer<int32_t>(),
        mRuntime.base.pipelineIO.specDecodePositionIds.dataPointer<int32_t>(),
        mRuntime.base.pipelineIO.selectTokenIndices.dataPointer<int64_t>(),
        mRuntime.base.pipelineIO.contextLengths.dataPointer<int32_t>(), batchSize, stream);
}

void DFlashDecoder::runBaseVerificationEmbeddingLookup(
    int32_t batchSize, int32_t verifySize, cudaStream_t stream, bool reshapeGemmaPleOutputs)
{
    check::check(
        mRuntime.base.pipelineIO.inputsEmbeds.reshape({batchSize, verifySize, mRuntime.deployment.base.hiddenSize}),
        "Tensor reshape failed");
    kernel::embeddingLookup(mRuntime.preprocess.idsInput, mRuntime.preprocess.embedding.table,
        mRuntime.preprocess.embedding.scalesAsOptional(), mRuntime.base.pipelineIO.inputsEmbeds, stream);
    if (reshapeGemmaPleOutputs && mRuntime.preprocess.gemma4Ple)
    {
        mRuntime.preprocess.gemma4Ple->reshapeOutputs(batchSize, verifySize);
    }
}

bool DFlashDecoder::capturePreparedBaseVerification(int32_t batchSize, int32_t verifySize, cudaStream_t stream)
{
    prepareCommonBaseVerificationInputs(batchSize, verifySize);
    auto const verifyDims = mRuntime.deployment.base.specVerifyDims(batchSize, verifySize);
    bool const captured = mRuntime.base.captureGraph(verifyDims, stream);
    if (!captured)
    {
        LOG_WARNING(
            "DFlashDecoder: failed to capture base verify graph (batch=%d, verifySize=%d)", batchSize, verifySize);
    }
    return captured;
}

void DFlashDecoder::reshapeBaseVerificationInputsOutputs(int32_t batchSize, int32_t verifySize)
{
    int32_t const selectTokenSize = batchSize * verifySize;
    check::check(
        mRuntime.base.pipelineIO.inputsEmbeds.reshape({batchSize, verifySize, mRuntime.deployment.base.hiddenSize}),
        "Tensor reshape failed");
    check::check(
        mRuntime.base.pipelineIO.outputLogits.reshape({selectTokenSize, mRuntime.deployment.base.outputVocabSize}),
        "Tensor reshape failed");
    check::check(mRuntime.base.pipelineIO.baseHiddenStates.reshape({selectTokenSize, mBaseOutputHiddenDim}),
        "Tensor reshape failed");
    check::check(mRuntime.base.pipelineIO.packedAttentionMask.reshape(
                     {batchSize, verifySize, static_cast<int64_t>(divUp(verifySize, 32))}),
        "Tensor reshape failed");
    check::check(mRuntime.base.pipelineIO.selectTokenIndices.reshape({batchSize, verifySize}), "Tensor reshape failed");
    check::check(mRuntime.base.pipelineIO.contextLengths.reshape({batchSize}), "Tensor reshape failed");
    check::check(
        mRuntime.base.pipelineIO.specDecodePositionIds.reshape({batchSize, verifySize}), "Tensor reshape failed");
}

void DFlashDecoder::prepareCommonBaseVerificationInputs(int32_t batchSize, int32_t verifySize)
{
    if (mRuntime.preprocess.deepstack)
    {
        mRuntime.preprocess.deepstack->useZeroTarget(mRuntime.base.tensorMap);
    }

    mRuntime.base.cacheManager.getMambaCacheManager().reshapeIntermediateStates(batchSize, verifySize);
    if (!mRuntime.base.pipelineIO.specVerifyPhaseMarker.isEmpty())
    {
        check::check(mRuntime.base.pipelineIO.specVerifyPhaseMarker.reshape({1}), "Tensor reshape failed");
    }
}

void DFlashDecoder::commitAcceptedTreePath(
    DecodingInferenceContext& context, int32_t verifySize, int32_t maxAcceptLength)
{
    int32_t const activeBatchSize = context.activeBatchSize;
    auto& cacheMgrBase = mRuntime.base.cacheManager;
    Tensor const& kvCacheLengths = cacheMgrBase.getKVCacheLengths();
    auto& kvMgrBase = cacheMgrBase.getKVCacheManager();
    auto const kvHeadDimGroups = cacheMgrBase.getKVHeadDimGroups();
    auto const kvCacheType = kvMgrBase.getConfig().kvCacheType;
    auto& mambaMgr = cacheMgrBase.getMambaCacheManager();
    bool const hasHybridStates = mambaMgr.hasIntermediateRecurrentStates() || mambaMgr.hasIntermediateConvStates();

    check::check(mRuntime.base.pipelineIO.baseHiddenStates.reshape({activeBatchSize, verifySize, mBaseOutputHiddenDim}),
        "Tensor reshape failed");
    // Branching-tree accept can skip nodes, so commit compacts accepted KV rows using accepted verify indices.
    for (auto const& group : kvHeadDimGroups)
    {
        kernel::eagleBaseCommitKVCache(mAcceptedTokenIndices, mAcceptLength, kvCacheLengths, group.deviceLayerInfos,
            group.numLayers, group.headDim, group.maxKVHeads, activeBatchSize, maxAcceptLength, kvCacheType,
            context.stream);
    }
    kernel::eagleBaseAssembleHiddenState(
        mAcceptedTokenIndices, mAcceptLength, mRuntime.base.pipelineIO.baseHiddenStates, context.stream);
    cacheMgrBase.commitSequenceLength(mAcceptLength, context.stream);
    if (hasHybridStates)
    {
        check::check(kernel::gdnTreeChunkVerifyEnabled(verifySize),
            "Hybrid DFlash requires compact GDN speculative verification.");
        mambaMgr.replayCommitAcceptedTreeStates(mAcceptedTokenIndices, mAcceptLength, context.stream);
    }

    check::check(
        mRuntime.base.pipelineIO.baseHiddenStates.reshape({activeBatchSize, maxAcceptLength, mBaseOutputHiddenDim}),
        "Tensor reshape failed");
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
    int32_t const BS = mBlockSize;
    int32_t const verifySize = mUseDDTree ? mVerifySize : BS;
    int32_t const packedMaskLen = static_cast<int32_t>(divUp(verifySize, 32));

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
        // Simulate a cache state with some tokens already committed
        std::vector<int32_t> simCacheLens(batchSize, kSimulateCacheLength);
        Tensor simCacheLensTensor(simCacheLens.data(), {batchSize}, DeviceType::kCPU, nvinfer1::DataType::kINT32);
        mRuntime.base.cacheManager.resetForNewSequences(simCacheLensTensor, stream);
        mDraftCacheManager.resetForNewSequences(simCacheLensTensor, stream);

        // --- Base verification CUDA graph capture ---
        {
            if (mUseDDTree)
            {
                int32_t const selectTokenSize = batchSize * verifySize;
                std::vector<int32_t> idsInput(static_cast<size_t>(selectTokenSize), 0);
                std::vector<int32_t> treeParentIds(static_cast<size_t>(selectTokenSize), -1);
                std::vector<int32_t> treeDepths(static_cast<size_t>(selectTokenSize), 0);
                std::vector<int32_t> positionIds(static_cast<size_t>(selectTokenSize), kSimulateCacheLength);
                std::vector<int64_t> selectTokenIndices(static_cast<size_t>(selectTokenSize), 0);
                std::vector<int32_t> contextLengths(static_cast<size_t>(batchSize), kSimulateCacheLength + verifySize);
                std::vector<int32_t> validCounts(static_cast<size_t>(batchSize), verifySize);
                std::vector<int32_t> packedAncestorMask(static_cast<size_t>(batchSize) * verifySize * packedMaskLen, 0);

                int32_t const childDepth = mBlockSize > 1 ? 1 : 0;
                reshapeBaseVerificationForCapture(batchSize, verifySize, /*includeTreeMetadata=*/true);
                check::check(mTreeTokenIds.reshape({batchSize, verifySize}), "Tensor reshape failed");
                check::check(mVerifyTokenIds.reshape({batchSize, verifySize}), "Tensor reshape failed");
                check::check(mValidCounts.reshape({batchSize}), "Tensor reshape failed");

                for (int32_t batchIdx = 0; batchIdx < batchSize; ++batchIdx)
                {
                    int32_t const batchOffset = batchIdx * verifySize;
                    for (int32_t nodeIdx = 0; nodeIdx < verifySize; ++nodeIdx)
                    {
                        int32_t const flatIdx = batchOffset + nodeIdx;
                        selectTokenIndices[flatIdx] = nodeIdx;
                        if (nodeIdx > 0 && childDepth > 0)
                        {
                            treeParentIds[flatIdx] = 0;
                            treeDepths[flatIdx] = childDepth;
                            positionIds[flatIdx] = kSimulateCacheLength + childDepth;
                            setPackedAncestorBit(packedAncestorMask, batchIdx, nodeIdx, 0, verifySize);
                        }
                        setPackedAncestorBit(packedAncestorMask, batchIdx, nodeIdx, nodeIdx, verifySize);
                    }
                }

                CUDA_CHECK(cudaMemcpyAsync(mVerifyTokenIds.rawPointer(), idsInput.data(),
                    idsInput.size() * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
                CUDA_CHECK(cudaMemcpyAsync(mTreeTokenIds.rawPointer(), idsInput.data(),
                    idsInput.size() * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
                copyVerifyTokenIdsToBaseInput(batchSize, verifySize, stream);
                CUDA_CHECK(cudaMemcpyAsync(mRuntime.base.pipelineIO.specTreeParentIds.rawPointer(),
                    treeParentIds.data(), treeParentIds.size() * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
                CUDA_CHECK(cudaMemcpyAsync(mRuntime.base.pipelineIO.specTreeDepths.rawPointer(), treeDepths.data(),
                    treeDepths.size() * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
                CUDA_CHECK(cudaMemcpyAsync(mRuntime.base.pipelineIO.specDecodePositionIds.rawPointer(),
                    positionIds.data(), positionIds.size() * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
                CUDA_CHECK(
                    cudaMemcpyAsync(mRuntime.base.pipelineIO.selectTokenIndices.rawPointer(), selectTokenIndices.data(),
                        selectTokenIndices.size() * sizeof(int64_t), cudaMemcpyHostToDevice, stream));
                CUDA_CHECK(cudaMemcpyAsync(mRuntime.base.pipelineIO.contextLengths.rawPointer(), contextLengths.data(),
                    contextLengths.size() * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
                CUDA_CHECK(cudaMemcpyAsync(mValidCounts.rawPointer(), validCounts.data(),
                    validCounts.size() * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
                CUDA_CHECK(cudaMemcpyAsync(mRuntime.base.pipelineIO.packedAttentionMask.rawPointer(),
                    packedAncestorMask.data(), packedAncestorMask.size() * sizeof(int32_t), cudaMemcpyHostToDevice,
                    stream));

                runBaseVerificationEmbeddingLookup(batchSize, verifySize, stream, /*reshapeGemmaPleOutputs=*/true);
                baseVerificationCaptureStatus &= capturePreparedBaseVerification(batchSize, verifySize, stream);
            }
            else
            {
                reshapeBaseVerificationForCapture(batchSize, verifySize, /*includeTreeMetadata=*/false);
                prepareLinearBaseVerificationMetadata(batchSize, verifySize, stream);
                baseVerificationCaptureStatus &= capturePreparedBaseVerification(batchSize, verifySize, stream);
            }
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

bool DFlashDecoder::hasSystemPromptKVCache(SystemPromptCacheKey const& key) const
{
    return mSystemPromptKVCacheDraft.find(key) != mSystemPromptKVCacheDraft.end();
}

void DFlashDecoder::restoreSystemPromptKVCache(SystemPromptCacheKey const& key, int32_t batchIdx, cudaStream_t stream)
{
    check::check(mSystemPromptKVCacheDraft.count(key) > 0, "DFlash system prompt cache missing for draft model");
    mDraftCacheManager.restoreKVCache(mSystemPromptKVCacheDraft[key].kvCacheLayers, batchIdx, stream);
}

bool DFlashDecoder::runSystemPromptPrefill(DecodingInferenceContext& context)
{
    int32_t const activeBatchSize = context.activeBatchSize;
    int64_t const prefillLen = mRuntime.base.pipelineIO.baseHiddenStates.getShape()[1];
    int32_t const BS = mBlockSize;

    check::check(mRuntime.preprocess.idsInput.reshape({activeBatchSize, BS}), "Tensor reshape failed");
    check::check(mHostDraftInputIds.reshape({activeBatchSize, BS}), "Tensor reshape failed");
    int32_t* hostDraftInputIds = mHostDraftInputIds.dataPointer<int32_t>();
    for (int32_t b = 0; b < activeBatchSize; ++b)
    {
        hostDraftInputIds[b * BS] = context.tokenIds[b].back();
        for (int32_t j = 1; j < BS; ++j)
        {
            hostDraftInputIds[b * BS + j] = mMaskTokenId;
        }
    }
    CUDA_CHECK(cudaMemcpyAsync(mRuntime.preprocess.idsInput.rawPointer(), mHostDraftInputIds.rawPointer(),
        activeBatchSize * BS * sizeof(int32_t), cudaMemcpyHostToDevice, context.stream));

    check::check(mDraftInputsEmbeds.reshape({activeBatchSize, BS, mDraftHiddenSize}), "Tensor reshape failed");
    kernel::embeddingLookup(mRuntime.preprocess.idsInput, mRuntime.preprocess.embedding.table,
        mRuntime.preprocess.embedding.scalesAsOptional(), mDraftInputsEmbeds, context.stream);

    // This intentionally narrows baseHiddenStates to the compact draft binding shape. The base runner reshapes and
    // rebinds it before the next base-engine enqueue.
    check::check(mRuntime.base.pipelineIO.baseHiddenStates.reshape({activeBatchSize, prefillLen, mBaseOutputHiddenDim}),
        "Tensor reshape failed");
    mDraftTensorMap.set(binding_names::kDFlashTargetHiddenConcat, mRuntime.base.pipelineIO.baseHiddenStates);

    check::check(mHostDeltaLens.reshape({activeBatchSize}), "Tensor reshape failed");
    int32_t* hostDeltaLens = mHostDeltaLens.dataPointer<int32_t>();
    std::fill_n(hostDeltaLens, activeBatchSize, static_cast<int32_t>(prefillLen));
    check::check(mDraftDeltaLens.reshape({activeBatchSize}), "Tensor reshape failed");
    CUDA_CHECK(cudaMemcpyAsync(mDraftDeltaLens.rawPointer(), mHostDeltaLens.rawPointer(),
        activeBatchSize * sizeof(int32_t), cudaMemcpyHostToDevice, context.stream));

    int32_t const pmLen = divUp(BS, 32);
    check::check(mDraftPackedAttentionMask.reshape({activeBatchSize, BS, pmLen}), "Tensor reshape failed");
    check::check(mDraftAttentionPosId.reshape({activeBatchSize, BS}), "Tensor reshape failed");
    check::check(mDraftContextLengths.reshape({activeBatchSize}), "Tensor reshape failed");

    Tensor const& draftCacheLengths = mDraftCacheManager.getKVCacheLengths();
    kernel::launchDFlashPrepareProposalInputs(draftCacheLengths.dataPointer<int32_t>(),
        mDraftDeltaLens.dataPointer<int32_t>(), BS, mDraftPackedAttentionMask.dataPointer<int32_t>(),
        mDraftAttentionPosId.dataPointer<int32_t>(), mDraftContextLengths.dataPointer<int32_t>(), activeBatchSize,
        context.stream);

    check::check(mDraftOutputLogits.reshape({activeBatchSize, BS, mDraftVocabSize}), "Tensor reshape failed");
    int32_t const draftKVCapacity = mRuntime.deployment.draft->maxKVCacheCapacity;
    InferenceDims const draftDims{
        /*.batch=*/activeBatchSize,
        /*.seqLen=*/BS,
        /*.kvLen=*/draftKVCapacity,
        /*.selectLen=*/prefillLen,
        /*.attnMaskSeqLen=*/BS,
        /*.ropeBatch=*/1,
        /*.packedMaskLen=*/static_cast<int64_t>(pmLen),
        /*.startIndexLen=*/activeBatchSize,
        /*.specVerifyPhaseLen=*/0,
    };

    bool ok = mDraftExecutor->prepare(kPrefillProfile, draftDims, mDraftTensorMap, context.stream);
    if (ok)
    {
        ok = mDraftExecutor->execute(context.stream);
    }
    if (!ok)
    {
        LOG_ERROR("DFlashDecoder: system prompt draft prefill failed.");
        return false;
    }

    check::check(mDraftDeltaLenCommit.reshape({activeBatchSize}), "Tensor reshape failed");
    CUDA_CHECK(cudaMemcpyAsync(mDraftDeltaLenCommit.rawPointer(), mDraftDeltaLens.rawPointer(),
        activeBatchSize * sizeof(int32_t), cudaMemcpyDeviceToDevice, context.stream));
    mDraftCacheManager.commitSequenceLength(mDraftDeltaLenCommit, context.stream);

    // Invariant (not runtime-checked): see the comment after the equivalent commit in
    // runDraftForward — DFlash's target-KV update assumes row == slot for the draft cache, guaranteed
    // by HybridCacheManager::compactBatch; no live page table exists yet to check against.

    return true;
}

void DFlashDecoder::saveSystemPromptKVCache(SystemPromptCacheKey const& key, std::string const& prompt,
    std::vector<tokenizer::Rank> const& tokenizedPrompt, int32_t promptIdsLength, cudaStream_t stream)
{
    constexpr int32_t kCacheBatchIdx{0};
    SystemPromptKVCache savedCache;
    savedCache.systemPrompt = prompt;
    savedCache.tokenizedPrompt = tokenizedPrompt;
    savedCache.kvCacheLayers = mDraftCacheManager.captureKVCache(kCacheBatchIdx, promptIdsLength, stream);
    mSystemPromptKVCacheDraft.insert({key, std::move(savedCache)});
}

void DFlashDecoder::resetForNewSequences(Tensor& reuseLengths, cudaStream_t stream)
{
    mDraftCacheManager.resetForNewSequences(reuseLengths, stream);
}

void DFlashDecoder::onBatchEvict(std::vector<int32_t> const& /* batchMapping */, int32_t oldActiveBatch,
    int32_t newActiveBatch, Tensor& deviceBatchMapping, cudaStream_t stream)
{
    mDraftCacheManager.compactBatch(deviceBatchMapping, oldActiveBatch, newActiveBatch, stream);
    mDraftCacheManager.setActiveBatchSize(newActiveBatch);
}

} // namespace rt
} // namespace trt_edgellm
