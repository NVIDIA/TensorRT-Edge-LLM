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

#include "runtime/state/pipelineIO.h"

#include "common/bindingNames.h"
#include "common/checkMacros.h"
#include "common/cudaUtils.h"
#include "common/logger.h"
#include "common/stringUtils.h"
#include "kernels/posEncoding/initializeCosSinCache.h"

#include <algorithm>
#include <stdexcept>

namespace trt_edgellm
{
namespace rt
{
void allocateBasicIO(
    PipelineIO& io, int32_t maxBatch, int32_t maxSeq, int32_t hiddenSize, int32_t vocabSize, nvinfer1::DataType dtype)
{
    io.inputsEmbeds = Tensor({maxBatch, maxSeq, hiddenSize}, DeviceType::kGPU, dtype, "PipelineIO::inputsEmbeds");
    // outputLogits is always FLOAT for the sampler; vocabSize is kept here to hold the
    // allocation next to the other "basic" tensors and to match the existing signature.
    io.outputLogits
        = Tensor({maxBatch, vocabSize}, DeviceType::kGPU, nvinfer1::DataType::kFLOAT, "PipelineIO::outputLogits");
    io.selectTokenIndices
        = Tensor({maxBatch, 1}, DeviceType::kGPU, nvinfer1::DataType::kINT64, "PipelineIO::selectTokenIndices");
    io.contextLengths = Tensor({maxBatch}, DeviceType::kGPU, nvinfer1::DataType::kINT32, "PipelineIO::contextLengths");
    io.hostContextLengths
        = Tensor({maxBatch}, DeviceType::kCPU, nvinfer1::DataType::kINT32, "PipelineIO::hostContextLengths");
    io.hostSelectTokenIndices
        = Tensor({maxBatch, 1}, DeviceType::kCPU, nvinfer1::DataType::kINT64, "PipelineIO::hostSelectTokenIndices");
}

void allocateDeepstackEmbeds(
    PipelineIO& io, int32_t numFeatures, int32_t maxBatch, int32_t maxSeq, int32_t hiddenSize, nvinfer1::DataType dtype)
{
    io.deepstackEmbeds.clear();
    io.deepstackEmbeds.reserve(numFeatures);
    for (int32_t i = 0; i < numFeatures; ++i)
    {
        io.deepstackEmbeds.emplace_back(
            Coords{maxBatch, maxSeq, hiddenSize}, DeviceType::kGPU, dtype, "PipelineIO::deepstackEmbeds");
    }
}

void allocateSpecDecodeHiddenStates(PipelineIO& io, int32_t maxBatch, int32_t maxSeq, int32_t baseHiddenDim,
    int32_t draftHiddenDim, nvinfer1::DataType dtype, bool allocateDraftHiddenStates)
{
    io.baseHiddenStates
        = Tensor({maxBatch, maxSeq, baseHiddenDim}, DeviceType::kGPU, dtype, "PipelineIO::baseHiddenStates");
    if (!allocateDraftHiddenStates)
    {
        return;
    }
    io.draftHiddenStatesIn
        = Tensor({maxBatch, maxSeq, draftHiddenDim}, DeviceType::kGPU, dtype, "PipelineIO::draftHiddenStatesIn");
    io.draftHiddenStatesOut
        = Tensor({maxBatch, maxSeq, draftHiddenDim}, DeviceType::kGPU, dtype, "PipelineIO::draftHiddenStatesOut");
}

void allocateMRope(PipelineIO& io, int32_t maxBatch, int32_t maxKVCacheCapacity, int32_t rotaryDim)
{
    io.mropeCosSin = Tensor({maxBatch, maxKVCacheCapacity, rotaryDim}, DeviceType::kGPU, nvinfer1::DataType::kFLOAT,
        "PipelineIO::mropeCosSin");
}

void StreamingPrefillBuffers::populateFromPrefill(Tensor const& liveInputEmbeds, Tensor const& liveEngineHiddenStates,
    int32_t batch, int32_t prefillLen, int32_t hiddenSize, int32_t maxBatch, int32_t maxSeq, cudaStream_t stream)
{
    auto const dtype = nvinfer1::DataType::kHALF;
    if (inputEmbeds.isEmpty())
    {
        inputEmbeds = Tensor(
            {maxBatch, maxSeq, hiddenSize}, DeviceType::kGPU, dtype, "PipelineIO::streamingPrefill.inputEmbeds");
        engineHiddenStates = Tensor(
            {maxBatch, maxSeq, hiddenSize}, DeviceType::kGPU, dtype, "PipelineIO::streamingPrefill.engineHiddenStates");
    }
    check::check(inputEmbeds.reshape({batch, prefillLen, hiddenSize}), "Tensor reshape failed");
    check::check(engineHiddenStates.reshape({batch, prefillLen, hiddenSize}), "Tensor reshape failed");

    size_t const bytes = static_cast<size_t>(batch) * prefillLen * hiddenSize * sizeof(__half);
    CUDA_CHECK(cudaMemcpyAsync(
        inputEmbeds.rawPointer(), liveInputEmbeds.rawPointer(), bytes, cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(
        engineHiddenStates.rawPointer(), liveEngineHiddenStates.rawPointer(), bytes, cudaMemcpyDeviceToDevice, stream));
}

void bindRopeTensors(TensorMap& map, PipelineIO& io, SharedResources& res, LLMEngineConfig const& cfg)
{
    if (cfg.useDualRope)
    {
        map.set(binding_names::kRopeCosSinSliding,
            res.ropePool.getOrCreate(cfg.slidingRopeConfig, cfg.slidingRotaryDim, cfg.maxKVCacheCapacity, nullptr));
        map.set(binding_names::kRopeCosSinFull,
            res.ropePool.getOrCreate(cfg.fullRopeConfig, cfg.fullRotaryDim, cfg.maxKVCacheCapacity, nullptr));
        return;
    }

    if (cfg.ropeConfig.type == RopeType::kMRope)
    {
        map.set(binding_names::kRopeCosSin, io.mropeCosSin);
    }
    else
    {
        map.set(binding_names::kRopeCosSin,
            res.ropePool.getOrCreate(cfg.ropeConfig, cfg.rotaryDim, cfg.maxKVCacheCapacity, nullptr));
    }
}

void buildTensorMap(
    TensorMap& map, PipelineIO& io, SharedResources& res, LLMEngineConfig const& cfg, int32_t kvCacheIndex)
{
    // Core I/O
    map.set(binding_names::kInputsEmbeds, io.inputsEmbeds);
    map.set(binding_names::kLogits, io.outputLogits);
    map.set(binding_names::kContextLengths, io.contextLengths);
    map.set(binding_names::kLastTokenIds, io.selectTokenIndices);
    if (cfg.useVisionBidirectionalAttention)
    {
        map.set(binding_names::kVisionBlockIds, io.visionBlockIds);
    }

    bindRopeTensors(map, io, res, cfg);

    // Hybrid cache routing: walk absolute decoder-layer indices, route by
    // `cfg.layerTypes[absIdx]`, and bind per-layer tensors using LOCAL indices.
    auto& cacheMgr = *res.cacheManagers[kvCacheIndex];
    auto& kvMgr = cacheMgr.getKVCacheManager();
    auto& mambaMgr = cacheMgr.getMambaCacheManager();

    int32_t localAttnIdx = 0;
    int32_t localMambaIdx = 0;
    for (int32_t absIdx = 0; absIdx < static_cast<int32_t>(cfg.layerTypes.size()); ++absIdx)
    {
        if (cfg.layerTypes[absIdx] == rt::HybridCacheManager::LayerType::kAttention)
        {
            // Check if this attention layer shares KV cache from a donor layer.
            int32_t const donorIdx
                = (!cfg.kvSharingDonors.empty() && localAttnIdx < static_cast<int32_t>(cfg.kvSharingDonors.size()))
                ? cfg.kvSharingDonors[localAttnIdx]
                : -1;

            // Plugin (combined KV): bind to donor's tensor if shared, else own tensor. Bind the
            // pool-shaped view — the AttentionPlugin engine binding contract is the paged pool
            // [2, numPages, kTOKENS_PER_PAGE, numKVHeads, headDim], not the internal slot-shaped
            // allocation (see KVCacheManager::getCombinedKVCachePoolView()).
            auto& combinedKV = (donorIdx >= 0) ? kvMgr.getCombinedKVCachePoolView(donorIdx)
                                               : kvMgr.getCombinedKVCachePoolView(localAttnIdx);
            map.set(binding_names::formatKVCacheName(localAttnIdx, /*isPast=*/true), combinedKV);
            map.set(binding_names::formatKVCacheName(localAttnIdx, /*isPast=*/false), combinedKV); // alias: in-place
            ++localAttnIdx;
        }
        else if (cfg.layerTypes[absIdx] == rt::HybridCacheManager::LayerType::kMamba)
        {
            auto& rec = mambaMgr.getRecurrentState(localMambaIdx);
            auto& conv = mambaMgr.getConvState(localMambaIdx);
            map.set(binding_names::formatRecurrentStateName(localMambaIdx, /*isPast=*/true), rec);
            map.set(binding_names::formatRecurrentStateName(localMambaIdx, /*isPast=*/false), rec);
            map.set(binding_names::formatConvStateName(localMambaIdx, /*isPast=*/true), conv);
            map.set(binding_names::formatConvStateName(localMambaIdx, /*isPast=*/false), conv);
            // Spec-decode hybrid base: bind the per-layer intermediate state outputs.
            // `hasIntermediateRecurrentStates()` is true iff the MambaCacheManager
            // was built with `maxIntermediateSeqLen > 0` (set by createForSpecDecode
            // for hybrid MTP bases). EAGLE3 base lacks recurrent layers entirely,
            // so this branch wouldn't fire for it regardless.
            if (mambaMgr.hasIntermediateRecurrentStates())
            {
                map.set(binding_names::formatIntermediateRecurrentStateName(localMambaIdx),
                    mambaMgr.getIntermediateRecurrentState(localMambaIdx));
            }
            if (mambaMgr.hasIntermediateConvStates())
            {
                map.set(binding_names::formatIntermediateConvStateName(localMambaIdx),
                    mambaMgr.getIntermediateConvState(localMambaIdx));
            }
            ++localMambaIdx;
        }
        else
        {
            check::check(false, format::fmtstr("buildTensorMap: unknown LayerType at absolute layer index %d", absIdx));
        }
    }

    // kvcache_start_index: single stable binding — the KV cache manager's
    // `kvCacheLengths` tensor. The registry resolves its shape from
    // `InferenceDims::startIndexLen` each call (0 for initial-prefill sentinel,
    // `batch` otherwise), so the same address serves every phase without a
    // per-step rebind.
    map.set(binding_names::kKVCacheStartIndex, cacheMgr.getKVCacheLengths());

    // kv_page_table: one stable TensorMap binding per cache manager. SharedResources initializes identity rows;
    // production context reuse updates the bound tensor in place with leased physical page IDs.
    map.set(binding_names::kKVPageTable, res.kvPageTables[kvCacheIndex]->kernelView());

    // Deepstack: initial bind is the shared zero buffer (sized large enough
    // to cover the worst-case non-prefill shape). DeepstackBinding (owned by
    // the runtime) swaps to `io.deepstackEmbeds[i]` just before base prefill
    // and back on non-prefill phases.
    for (size_t i = 0; i < io.deepstackEmbeds.size(); ++i)
    {
        map.set(binding_names::formatDeepstackEmbedsName(static_cast<int32_t>(i)), res.zeroBuffer);
    }

    // Hidden states output. SpecDecode base engines write their target features
    // into baseHiddenStates. The vanilla LLM path uses
    // outputHiddenStates instead (shape uses cfg.hiddenSize). Either or neither is
    // bound here; the engine introspection in EngineExecutor::prepare() will set
    // the address only if the engine actually exposes the binding.
    if (cfg.isSpecDecodeBase && !io.baseHiddenStates.isEmpty())
    {
        map.set(binding_names::kOutputHiddenStates, io.baseHiddenStates);
    }
    else if (!io.outputHiddenStates.isEmpty())
    {
        map.set(binding_names::kOutputHiddenStates, io.outputHiddenStates);
    }

    // SpecDecode base-engine verification bindings. The base engine's verification
    // profile (also reused during prefill/decode via the dummy [B, 1, 1] shape)
    // reads the packed attention mask and position IDs. For vanilla LLMs these
    // tensors are empty and the bindings are not set.
    if (cfg.isSpecDecodeBase && !io.packedAttentionMask.isEmpty())
    {
        map.set(binding_names::kAttentionMask, io.packedAttentionMask);
        map.set(binding_names::kAttentionPosId, io.specDecodePositionIds);
    }
    if (!io.specVerifyPhaseMarker.isEmpty())
    {
        map.set(binding_names::kSpecVerifyPhaseMarker, io.specVerifyPhaseMarker);
    }
    if (!io.specTreeParentIds.isEmpty())
    {
        map.set(binding_names::kTreeParentIds, io.specTreeParentIds);
    }
    if (!io.specTreeDepths.isEmpty())
    {
        map.set(binding_names::kTreeDepths, io.specTreeDepths);
    }

    // LoRA bindings are NOT set here because adapter tensor names may differ
    // from engine binding names (e.g. fused QKV).  LoRAManager::refreshTensorMap()
    // populates them after buildTensorMap().
}

void buildTensorMapForSpecDecodeDraft(TensorMap& map, PipelineIO& io, SharedResources& res, LLMEngineConfig const& cfg)
{
    // Reuse the shared buildTensorMap for common bindings (core I/O, RoPE,
    // KV cache, kvcache_start_index). Draft engine uses kvCacheIndex=1.
    buildTensorMap(map, io, res, cfg, /*kvCacheIndex=*/1);

    // Draft-specific hidden-state bindings: the base model's hidden states feed
    // the draft engine as input; the draft engine produces its own hidden states
    // on output (the kOutputHiddenStates entry added by buildTensorMap —
    // gated on cfg.isSpecDecodeBase which is false for the draft config —
    // is overridden here regardless).
    map.set(binding_names::kBaseModelHiddenStates, io.baseHiddenStates);
    map.set(binding_names::kDraftModelHiddenStates, io.draftHiddenStatesIn);
    map.set(binding_names::kOutputHiddenStates, io.draftHiddenStatesOut);

    // Attention mask and position IDs for proposal decoding. The TRT engine expects
    // the INT32 packed mask (not the INT8 unpacked one). Position IDs are written
    // by proposal/verify input preparation kernels before each execute.
    map.set(binding_names::kAttentionMask, io.packedAttentionMask);
    map.set(binding_names::kAttentionPosId, io.specDecodePositionIds);
}

void buildTensorMapForGemma4MTPDraft(
    TensorMap& map, PipelineIO& io, SharedResources& res, DeploymentConfig const& bundle)
{
    check::check(bundle.draft.has_value(), "buildTensorMapForGemma4MTPDraft requires bundle.draft");
    check::check(bundle.specConfig.has_value(), "buildTensorMapForGemma4MTPDraft requires bundle.specConfig");
    check::check(bundle.specDecodeMode() == SpecDecodeMode::kGemma4MTP,
        "buildTensorMapForGemma4MTPDraft requires spec_decode_type=gemma4_mtp");
    check::check(!res.cacheManagers.empty(), "buildTensorMapForGemma4MTPDraft requires base cache manager");

    LLMEngineConfig const& draftCfg = *bundle.draft;

    map.set(binding_names::kInputsEmbeds, io.inputsEmbeds);
    map.set(binding_names::kLogits, io.outputLogits);
    map.set(binding_names::kBaseModelHiddenStates, io.draftHiddenStatesIn);
    map.set(binding_names::kOutputHiddenStates, io.draftHiddenStatesOut);

    bindRopeTensors(map, io, res, draftCfg);

    auto& baseCacheManager = *res.cacheManagers[0];
    map.set(binding_names::kContextLengths, baseCacheManager.getKVCacheLengths());
    // kv_page_table: the assistant reads the target model's paged pool, so it binds the same target table as the
    // base engine. Context reuse is currently rejected for Gemma4 MTP deployments.
    map.set(binding_names::kKVPageTable, res.kvPageTables[0]->kernelView());
    for (auto const& entry : draftCfg.gemma4MTPKVSharingMap)
    {
        rt::Tensor& targetKV = baseCacheManager.getCombinedKVCachePoolView(entry.targetAbsoluteLayerIdx);
        map.set(binding_names::formatKVCacheName(entry.assistantLayerIdx, /*isPast=*/true), targetKV);
    }
}

PipelineIO PipelineIO::createForLLM(LLMEngineConfig const& cfg, cudaStream_t stream)
{
    PipelineIO io;

    allocateBasicIO(io, cfg.maxSupportedBatchSize, cfg.maxSupportedInputLength, cfg.hiddenSize, cfg.outputVocabSize,
        nvinfer1::DataType::kHALF);

    if (cfg.useVisionBidirectionalAttention)
    {
        io.visionBlockIds = Tensor({cfg.maxSupportedBatchSize, cfg.maxSupportedInputLength}, DeviceType::kGPU,
            nvinfer1::DataType::kINT32, "PipelineIO::visionBlockIds");
    }

    if (cfg.numDeepstackFeatures > 0)
    {
        allocateDeepstackEmbeds(io, cfg.numDeepstackFeatures, cfg.maxSupportedBatchSize, cfg.maxSupportedInputLength,
            cfg.hiddenSize, nvinfer1::DataType::kHALF);
        LOG_INFO("Allocated %d deepstack embeds tensors with shape [%d, %d, %d]", cfg.numDeepstackFeatures,
            cfg.maxSupportedBatchSize, cfg.maxSupportedInputLength, cfg.hiddenSize);
    }

    // Engine-output hidden states for the vanilla LLM path. Always allocated:
    // streaming consumers (Qwen3-Omni Talker) read it; if the engine emits
    // hidden_states but no consumer is set, the buffer is harmless write-target;
    // if the engine has no hidden_states output the binding is silently skipped.
    io.outputHiddenStates = Tensor({cfg.maxSupportedBatchSize, cfg.maxSupportedInputLength, cfg.hiddenSize},
        DeviceType::kGPU, nvinfer1::DataType::kHALF, "PipelineIO::outputHiddenStates");

    if (cfg.ropeConfig.type == RopeType::kMRope)
    {
        allocateMRope(io, cfg.maxSupportedBatchSize, cfg.maxKVCacheCapacity, cfg.rotaryDim);
        // Initialize MRoPE cache for all batch slots using text-only sequential positions.
        kernel::initializeTextOnlyMRopeCosSin(io.mropeCosSin.dataPointer<float>(), cfg.ropeConfig.rotaryTheta,
            cfg.rotaryDim, cfg.maxKVCacheCapacity, cfg.maxSupportedBatchSize, stream);
    }

    return io;
}

PipelineIO PipelineIO::createForSpecDecode(
    DeploymentConfig const& bundle, int32_t maxRuntimeBatchSize, cudaStream_t stream)
{
    check::check(bundle.draft.has_value(), "PipelineIO::createForSpecDecode requires DeploymentConfig.draft to be set");
    check::check(bundle.specConfig.has_value(),
        "PipelineIO::createForSpecDecode requires DeploymentConfig.specConfig to be set");

    PipelineIO io;

    int32_t const maxDraftProposalSize = bundle.specConfig->maxDraftProposalSize;
    int32_t const draftHiddenSize = bundle.specConfig->draftHiddenSize;
    int32_t const baseOutputHiddenDim = bundle.specConfig->baseOutputHiddenDim;
    int32_t const draftRuntimeHiddenSize
        = bundle.specDecodeMode() == SpecDecodeMode::kGemma4MTP ? baseOutputHiddenDim : draftHiddenSize;
    int32_t const draftVocabSize = bundle.draft->vocabSize;

    // Use max of base and draft dimensions for shared tensors
    int32_t const maxInputLength = std::max(bundle.base.maxSupportedInputLength, bundle.draft->maxSupportedInputLength);
    int32_t const effectiveMaxDraftProposalSize
        = std::max({maxDraftProposalSize, bundle.specConfig->verifySize, bundle.specConfig->dflashBlockSize});
    int32_t const maxLogitsSize = maxRuntimeBatchSize * effectiveMaxDraftProposalSize;
    int32_t const maxVocabSize = std::max(bundle.base.outputVocabSize, draftVocabSize);
    int32_t const maxTensorSeqLen = std::max(maxInputLength, effectiveMaxDraftProposalSize);

    allocateBasicIO(
        io, maxRuntimeBatchSize, maxTensorSeqLen, bundle.base.hiddenSize, maxVocabSize, nvinfer1::DataType::kHALF);

    // Override outputLogits to support proposal-sized outputs: [maxLogitsSize, maxVocabSize].
    // dtype is kFLOAT (matching allocateBasicIO); only the shape changes for SpecDecode.
    io.outputLogits = rt::Tensor(
        {maxLogitsSize, maxVocabSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT, "PipelineIO::outputLogits");

    // Allocate hidden states for SpecDecode. DFlash binds the draft target-hidden
    // input directly to baseHiddenStates, so it does not need the generic
    // EAGLE/MTP draft hidden-state ping-pong buffers.
    allocateSpecDecodeHiddenStates(io, maxRuntimeBatchSize, maxTensorSeqLen, baseOutputHiddenDim,
        draftRuntimeHiddenSize, nvinfer1::DataType::kHALF, bundle.specDecodeMode() != SpecDecodeMode::kDFlash);

    if (bundle.base.numDeepstackFeatures > 0)
    {
        allocateDeepstackEmbeds(io, bundle.base.numDeepstackFeatures, maxRuntimeBatchSize, maxInputLength,
            bundle.base.hiddenSize, nvinfer1::DataType::kHALF);
        LOG_INFO("Allocated %d deepstack embeds tensors with shape [%d, %d, %d]", bundle.base.numDeepstackFeatures,
            maxRuntimeBatchSize, maxInputLength, bundle.base.hiddenSize);
    }

    if (bundle.base.ropeConfig.type == RopeType::kMRope)
    {
        allocateMRope(io, maxRuntimeBatchSize, bundle.base.maxKVCacheCapacity, bundle.base.rotaryDim);
        kernel::initializeTextOnlyMRopeCosSin(io.mropeCosSin.dataPointer<float>(), bundle.base.ropeConfig.rotaryTheta,
            bundle.base.rotaryDim, bundle.base.maxKVCacheCapacity, maxRuntimeBatchSize, stream);
    }

    // SpecDecode-specific engine I/O: packed attention mask, position IDs, and a
    // proposal-sized selectTokenIndices override (the default allocateBasicIO gives
    // [maxBatch, 1], but verification needs up to [maxBatch,
    // effectiveMaxDraftProposalSize]). Zero-initialise the mask buffer so the
    // [B, 1, 1] dummy reshape during prefill/decode sees known-zero bytes.
    int64_t const packedMaskLen = static_cast<int64_t>(divUp(effectiveMaxDraftProposalSize, 32));
    io.packedAttentionMask = Tensor({maxRuntimeBatchSize, effectiveMaxDraftProposalSize, packedMaskLen},
        DeviceType::kGPU, nvinfer1::DataType::kINT32, "PipelineIO::packedAttentionMask");
    CUDA_CHECK(
        cudaMemsetAsync(io.packedAttentionMask.rawPointer(), 0, io.packedAttentionMask.getMemoryCapacity(), stream));

    io.specDecodePositionIds = Tensor({maxRuntimeBatchSize, effectiveMaxDraftProposalSize}, DeviceType::kGPU,
        nvinfer1::DataType::kINT32, "PipelineIO::specDecodePositionIds");
    CUDA_CHECK(cudaMemsetAsync(
        io.specDecodePositionIds.rawPointer(), 0, io.specDecodePositionIds.getMemoryCapacity(), stream));

    io.selectTokenIndices = Tensor({maxRuntimeBatchSize, effectiveMaxDraftProposalSize}, DeviceType::kGPU,
        nvinfer1::DataType::kINT64, "PipelineIO::selectTokenIndices");
    CUDA_CHECK(
        cudaMemsetAsync(io.selectTokenIndices.rawPointer(), 0, io.selectTokenIndices.getMemoryCapacity(), stream));

    io.specVerifyPhaseMarker
        = Tensor({1}, DeviceType::kGPU, nvinfer1::DataType::kINT32, "PipelineIO::specVerifyPhaseMarker");
    CUDA_CHECK(cudaMemsetAsync(
        io.specVerifyPhaseMarker.rawPointer(), 0, io.specVerifyPhaseMarker.getMemoryCapacity(), stream));

    SpecDecodeMode const specMode = bundle.specDecodeMode();
    bool const useSpecTree = specMode == SpecDecodeMode::kMTP
        || (specMode == SpecDecodeMode::kDFlash && bundle.specConfig->draftingTopK > 1);
    if (useSpecTree)
    {
        io.specTreeParentIds = Tensor({maxRuntimeBatchSize, effectiveMaxDraftProposalSize}, DeviceType::kGPU,
            nvinfer1::DataType::kINT32, "PipelineIO::specTreeParentIds");
        CUDA_CHECK(
            cudaMemsetAsync(io.specTreeParentIds.rawPointer(), 0, io.specTreeParentIds.getMemoryCapacity(), stream));

        io.specTreeDepths = Tensor({maxRuntimeBatchSize, effectiveMaxDraftProposalSize}, DeviceType::kGPU,
            nvinfer1::DataType::kINT32, "PipelineIO::specTreeDepths");
        CUDA_CHECK(cudaMemsetAsync(io.specTreeDepths.rawPointer(), 0, io.specTreeDepths.getMemoryCapacity(), stream));
    }

    return io;
}

} // namespace rt
} // namespace trt_edgellm
