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

#include "runtime/exec/registryBuilder.h"
#include "common/bindingNames.h"
#include "common/checkMacros.h"
#include "common/pagedKvTypes.h"

namespace trt_edgellm
{
namespace rt
{

// Note: `sym(...)` and `fixed(...)` helpers are now provided by inferenceDims.h.
// Symbolic-dim registration (formerly `addCommonSymbolicDims` / `addSymbolicDim`)
// is no longer needed — every symbolic reference is a pointer-to-member of
// `InferenceDims`, so the set of dims exists by construction of the type.

//! Page count for paged-pool KV-cache bindings
//! [2, numPages_i, kTOKENS_PER_PAGE, numKVHeads, headDim]. Full-only layers use the serialized
//! full-pool count. SWA-capable layers use either their bounded physical count or the full-pool
//! count according to the runtime-selected storage policy.
constexpr int32_t kTokensPerPage = rt::kTOKENS_PER_PAGE;

static int32_t computeNumPages(LLMEngineConfig const& cfg, KVLayerConfig const& layerConfig)
{
    int64_t const minimumActivePages = rt::computeMinimumKvPoolPages(cfg.maxSupportedBatchSize, cfg.maxKVCacheCapacity);
    ELLM_CHECK(cfg.kvPoolPages >= minimumActivePages && cfg.kvPoolPages <= rt::kMAX_KV_POOL_PAGES,
        "KV pool page count (" + std::to_string(cfg.kvPoolPages) + ") is outside [" + std::to_string(minimumActivePages)
            + ", " + std::to_string(rt::kMAX_KV_POOL_PAGES) + "].");
    return cfg.getKVPoolPagesForLayer(layerConfig);
}

void addRopeTensorSpecs(TensorRegistry& reg, LLMEngineConfig const& cfg)
{
    auto addRopeTensor = [&](char const* name, int32_t rotaryDim) {
        reg.addTensor(
            {name, TensorIO::kInput, nvinfer1::DataType::kFLOAT, {sym(&InferenceDims::seqLen), fixed(rotaryDim)}});
    };

    if (cfg.useDualRope)
    {
        addRopeTensor(binding_names::kRopeCosSinSliding, cfg.slidingRotaryDim);
        addRopeTensor(binding_names::kRopeCosSinFull, cfg.fullRotaryDim);
        return;
    }

    addRopeTensor(binding_names::kRopeCosSin, cfg.rotaryDim);
}

//! Add the dynamic page-table binding. AttentionPlugin cross-checks its row count against the packed QKV batch,
//! so the first dimension must track the active batch rather than the full physical table extent.
void addKVPageTableSpec(TensorRegistry& reg, LLMEngineConfig const& cfg)
{
    int32_t const maxPagesPerSeq = rt::computeMaxPagesPerSeq(cfg.maxKVCacheCapacity);
    reg.addTensor({binding_names::kKVPageTable, TensorIO::kInput, nvinfer1::DataType::kINT32,
        {sym(&InferenceDims::batch), fixed(2), fixed(maxPagesPerSeq)}});
}

void addUnifiedDecoderMetadata(TensorRegistry& reg, bool includeLogitsIndices = true)
{
    reg.addTensor(
        {binding_names::kPositions, TensorIO::kInput, nvinfer1::DataType::kINT32, {sym(&InferenceDims::seqLen)}});
    reg.addTensor({binding_names::kQueryStartOffsets, TensorIO::kInput, nvinfer1::DataType::kINT32,
        {sym(&InferenceDims::queryOffsetLen)}});
    for (char const* name : {binding_names::kQueryLengths, binding_names::kPastLengths,
             binding_names::kAttentionSequenceLengths, binding_names::kStateIndices})
    {
        reg.addTensor({name, TensorIO::kInput, nvinfer1::DataType::kINT32, {sym(&InferenceDims::batch)}});
    }
    if (includeLogitsIndices)
    {
        reg.addTensor({binding_names::kLogitsIndices, TensorIO::kInput, nvinfer1::DataType::kINT64,
            {sym(&InferenceDims::selectLen)}});
    }
    reg.addTensor({binding_names::kExecutionPhaseMarker, TensorIO::kInput, nvinfer1::DataType::kINT32,
        {sym(&InferenceDims::executionPhaseLen)}});
    reg.addTensor({binding_names::kContextSequenceCountCarrier, TensorIO::kInput, nvinfer1::DataType::kINT32,
        {sym(&InferenceDims::contextSequenceCount)}});
}

TensorRegistry buildRegistryForLLM(LLMEngineConfig const& cfg, std::optional<int32_t> specDecodeBaseOutputHiddenDim)
{
    TensorRegistry reg;

    // ---------------------------------------------------------------
    // Core I/O tensors (always present)
    // ---------------------------------------------------------------

    reg.addTensor({binding_names::kInputsEmbeds, TensorIO::kInput, nvinfer1::DataType::kHALF,
        {sym(&InferenceDims::seqLen), fixed(cfg.hiddenSize)}});
    addUnifiedDecoderMetadata(reg, !cfg.isDiffusionBackbone);

    reg.addTensor({binding_names::kLogits, TensorIO::kOutput, nvinfer1::DataType::kFLOAT,
        {sym(&InferenceDims::selectLen), fixed(cfg.outputVocabSize)}});

    if (cfg.isDiffusionBackbone)
    {
        // phase_is_encoder: invocation scalar. Non-zero selects encoder-phase layer scalars.
        reg.addTensor({binding_names::kPhaseIsEncoder, TensorIO::kInput, nvinfer1::DataType::kINT32, {fixed(1)}});

        // select_token_indices: [batch, select_len] INT64. Denoise selects the full canvas.
        reg.addTensor({binding_names::kSelectTokenIndices, TensorIO::kInput, nvinfer1::DataType::kINT64,
            {sym(&InferenceDims::selectLen)}});

        if (cfg.diffusionUnifiedConditioning)
        {
            // Unified DiffusionGemma ONNX embeds the self-conditioning graph in the backbone.
            // These inputs share ONNX's seq_len symbol with inputs_embeds, so
            // TRT requires equal binding dimensions even in encoder prefill
            // where the branch does not read conditioning values.
            reg.addTensor({binding_names::kCanvasIds, TensorIO::kInput, nvinfer1::DataType::kINT32,
                {sym(&InferenceDims::seqLen)}});
            reg.addTensor({binding_names::kPrevSelfConditioningEmbeds, TensorIO::kInput, nvinfer1::DataType::kHALF,
                {sym(&InferenceDims::seqLen), fixed(cfg.hiddenSize)}});
            reg.addTensor({binding_names::kSelfConditioningTemperature, TensorIO::kInput, nvinfer1::DataType::kFLOAT,
                {fixed(1)}});
            reg.addTensor({binding_names::kNextSelfConditioningEmbeds, TensorIO::kOutput, nvinfer1::DataType::kHALF,
                {sym(&InferenceDims::selectLen), fixed(cfg.hiddenSize)}});
        }
    }
    addKVPageTableSpec(reg, cfg);
    if (cfg.supportsBoundedSwaKVCache())
    {
        int32_t const maxPagesPerSeq = rt::computeMaxPagesPerSeq(cfg.maxKVCacheCapacity);
        reg.addTensor({binding_names::kSwaKVPageTable, TensorIO::kInput, nvinfer1::DataType::kINT32,
            {sym(&InferenceDims::batch), fixed(2), fixed(maxPagesPerSeq)}});
        reg.addTensor({binding_names::kSwaKVCacheMode, TensorIO::kInput, nvinfer1::DataType::kINT8,
            {sym(&InferenceDims::swaKVCacheModeLen)}});
    }

    if (cfg.useVisionBidirectionalAttention)
    {
        std::vector<ShapeDim> const shape{sym(&InferenceDims::seqLen)};
        reg.addTensor({binding_names::kVisionBlockIds, TensorIO::kInput, nvinfer1::DataType::kINT32, shape});
    }
    if (cfg.contextMaskSelectorEnabled)
    {
        reg.addTensor({binding_names::kContextMaskSelector, TensorIO::kInput, nvinfer1::DataType::kINT32,
            {sym(&InferenceDims::contextMaskSelectorLen)}});
    }

    // RoPE cache inputs: single binding for single-RoPE models, explicit
    // sliding/full bindings for mixed-attention dual-RoPE models.
    // For non-MRope, rope_batch is always 1 (TRT broadcasts); for MRope, rope_batch = activeBatchSize.
    addRopeTensorSpecs(reg, cfg);

    // ---------------------------------------------------------------
    // Per-layer KV / recurrent / conv state (hybrid routing by layer_types)
    // ---------------------------------------------------------------
    //
    // Walk the absolute decoder-layer indices and emit one spec per layer.
    // For attention layers we use `cfg.kvLayerConfigs[localAttnIdx]` so a model
    // with heterogeneous head configs (Gemma4, Qwen3-Next, etc.) gets the
    // correct per-layer shape. The %d suffix in the binding name is always a
    // LOCAL index (0..numAttn-1 for attention, 0..numMamba-1 for recurrent).
    int32_t localAttnIdx = 0;
    int32_t localMambaIdx = 0;
    for (int32_t absIdx = 0; absIdx < static_cast<int32_t>(cfg.layerTypes.size()); ++absIdx)
    {
        if (cfg.layerTypes[absIdx] == rt::HybridCacheManager::LayerType::kAttention)
        {
            auto const& lc = cfg.kvLayerConfigs[localAttnIdx];
            auto addKVCacheTensor = [&](char const* tmpl, TensorIO io, std::vector<ShapeDim> const& shape) {
                reg.addTensor({std::string(tmpl) + "_" + std::to_string(localAttnIdx), io, cfg.kvCacheDtype, shape});
            };
            // Plugin: paged pool, 5D [2, numPages, kTOKENS_PER_PAGE, numKVHeads, headDim].
            // numPages is fixed per layer for the life of this engine context.
            int32_t const numPages = computeNumPages(cfg, lc);
            std::vector<ShapeDim> const shape{
                fixed(2), fixed(numPages), fixed(kTokensPerPage), fixed(lc.numKVHeads), fixed(lc.headDim)};
            addKVCacheTensor(binding_names::kPastKeyValuesTemplate, TensorIO::kInput, shape);
            addKVCacheTensor(binding_names::kPresentKeyValuesTemplate, TensorIO::kOutput, shape);
            ++localAttnIdx;
        }
        else // kMamba
        {
            auto addMambaTensor
                = [&](char const* tmpl, TensorIO io, nvinfer1::DataType dtype, std::vector<ShapeDim> const& shape) {
                      reg.addTensor({std::string(tmpl) + "_" + std::to_string(localMambaIdx), io, dtype, shape});
                  };
            // recurrent_state_%d: [residentPoolRows, recurrentStateNumHeads, recurrentStateHeadDim, stateSize]
            std::vector<ShapeDim> const recShape{fixed(cfg.recurrentPoolRows), fixed(cfg.recurrentStateNumHeads),
                fixed(cfg.recurrentStateHeadDim), fixed(cfg.recurrentStateSize)};
            addMambaTensor(binding_names::kRecurrentStateTemplate, TensorIO::kInput, cfg.recurrentStateDtype, recShape);
            addMambaTensor(
                binding_names::kPresentRecurrentStateTemplate, TensorIO::kOutput, cfg.recurrentStateDtype, recShape);
            // conv_state_%d: [residentPoolRows, convDim, convKernel]
            std::vector<ShapeDim> const convShape{
                fixed(cfg.recurrentPoolRows), fixed(cfg.convDim), fixed(cfg.convKernel)};
            addMambaTensor(binding_names::kConvStateTemplate, TensorIO::kInput, cfg.convStateDtype, convShape);
            addMambaTensor(binding_names::kPresentConvStateTemplate, TensorIO::kOutput, cfg.convStateDtype, convShape);

            // Hybrid MTP/DFlash/JetSpec/DSpark base: per-layer intermediate state outputs
            // written during prefill/verification so accepted recurrent/conv state snapshots
            // can be committed after speculative verification.
            // recurrentSpecVerifyUsesReplay selects which output set the engine declares
            // (replay stash vs full-state snapshot).
            //
            // Intermediate state is token-major and indexed by the verification token row.
            if (cfg.specDecodeType == SpecDecodeMode::kMTP || isCachedBlockDraftMode(cfg.specDecodeType)
                || cfg.specDecodeType == SpecDecodeMode::kDSpark)
            {
                bool const useReplay = cfg.recurrentSpecVerifyUsesReplay;
                if (useReplay)
                {
                    // replay_da_state_%d: [tokens, recurrentNumHeads]
                    addMambaTensor(binding_names::kReplayDaStateTemplate, TensorIO::kOutput, nvinfer1::DataType::kFLOAT,
                        {sym(&InferenceDims::seqLen), fixed(cfg.recurrentStateNumHeads)});
                    // replay_u_state_%d: [tokens, recurrentNumHeads, recurrentHeadDim]
                    addMambaTensor(binding_names::kReplayUStateTemplate, TensorIO::kOutput, nvinfer1::DataType::kFLOAT,
                        {sym(&InferenceDims::seqLen), fixed(cfg.recurrentStateNumHeads),
                            fixed(cfg.recurrentStateHeadDim)});
                    // replay_b_state_%d: [tokens, recurrentNumGroups, recurrentStateSize]
                    addMambaTensor(binding_names::kReplayBStateTemplate, TensorIO::kOutput, nvinfer1::DataType::kFLOAT,
                        {sym(&InferenceDims::seqLen), fixed(cfg.recurrentStateNumGroups),
                            fixed(cfg.recurrentStateSize)});
                    // replay_dt_state_%d: [tokens, recurrentNumHeads]
                    addMambaTensor(binding_names::kReplayDtStateTemplate, TensorIO::kOutput, nvinfer1::DataType::kFLOAT,
                        {sym(&InferenceDims::seqLen), fixed(cfg.recurrentStateNumHeads)});
                }
                else
                {
                    // intermediate_recurrent_state_%d: [tokens, recurrentNumHeads, recurrentHeadDim, dstate]
                    std::vector<ShapeDim> const interRecShape{sym(&InferenceDims::seqLen),
                        fixed(cfg.recurrentStateNumHeads), fixed(cfg.recurrentStateHeadDim),
                        fixed(cfg.recurrentStateSize)};
                    addMambaTensor(binding_names::kIntermediateRecurrentStateTemplate, TensorIO::kOutput,
                        cfg.recurrentStateDtype, interRecShape);
                }
                if (cfg.convDim > 0 && cfg.convKernel > 0)
                {
                    std::vector<ShapeDim> const interConvShape{
                        sym(&InferenceDims::seqLen), fixed(cfg.convDim), fixed(cfg.convKernel)};
                    addMambaTensor(binding_names::kIntermediateConvStateTemplate, TensorIO::kOutput, cfg.convStateDtype,
                        interConvShape);
                }
            }
            ++localMambaIdx;
        }
    }

    // ---------------------------------------------------------------
    // Deepstack (Qwen3-VL / Qwen3-Omni)
    // ---------------------------------------------------------------
    if (!cfg.isDiffusionBackbone && cfg.numDeepstackFeatures > 0)
    {
        // deepstack_embeds_%d: [physicalTokens, hiddenSize] HALF — one per feature.
        // DeepstackBinding swaps the backing tensor (real per-request buffer
        // vs. shared zero buffer) between prefill and non-prefill phases.
        std::vector<ShapeDim> const shape{sym(&InferenceDims::seqLen), fixed(cfg.hiddenSize)};
        reg.addTensor({std::string(binding_names::kDeepstackEmbedsTemplate) + "_%d", TensorIO::kInput,
            nvinfer1::DataType::kHALF, shape, /*perLayer=*/cfg.numDeepstackFeatures});
    }

    // ---------------------------------------------------------------
    // SpecDecode verification bindings (base engine side)
    // ---------------------------------------------------------------
    if (cfg.isSpecDecodeBase)
    {
        // Token-major hidden-state feedback rows.
        // The concrete output hidden dim is strategy-specific and is consolidated
        // in DeploymentConfig::specDecode.
        int32_t const baseOutputHiddenDim = specDecodeBaseOutputHiddenDim.value_or(cfg.hiddenSize * 3);
        reg.addTensor({binding_names::kOutputHiddenStates, TensorIO::kOutput, nvinfer1::DataType::kHALF,
            {sym(&InferenceDims::seqLen), fixed(baseOutputHiddenDim)}});

        // attention_mask: [physicalTokens, packed_mask_len] INT32. Each row stores
        // the packed mask for one sequence-local proposal span.
        reg.addTensor({binding_names::kAttentionMask, TensorIO::kInput, nvinfer1::DataType::kINT32,
            {sym(&InferenceDims::attnMaskSeqLen), sym(&InferenceDims::packedMaskLen)}});

        // attention_pos_id: [physicalTokens] INT32
        reg.addTensor({binding_names::kAttentionPosId, TensorIO::kInput, nvinfer1::DataType::kINT32,
            {sym(&InferenceDims::attnMaskSeqLen)}});
    }

    // ---------------------------------------------------------------
    // LoRA
    // ---------------------------------------------------------------
    // LoRA bindings are dynamic (enumerated from the engine). The registry
    // builder cannot know the exact tensor names at compile time because they
    // depend on which model layers have LoRA adapters. LoRA tensors are handled
    // separately by the EngineExecutor via engine introspection (getLoraWeightsTensorNames).
    // Therefore we intentionally skip LoRA here.

    return reg;
}

TensorRegistry buildRegistryForSpecDecodeDraft(DeploymentConfig const& bundle)
{
    check::check(bundle.draft.has_value(), "buildRegistryForSpecDecodeDraft: bundle.draft must be set");
    check::check(bundle.specConfig.has_value(), "buildRegistryForSpecDecodeDraft: bundle.specConfig must be set");
    TensorRegistry reg;

    LLMEngineConfig const& cfg = *bundle.draft;
    int32_t const draftHiddenSize = bundle.specConfig->draftHiddenSize;
    int32_t const baseOutputHiddenDim = bundle.specConfig->baseOutputHiddenDim;
    int32_t const draftVocabSize = cfg.outputVocabSize;

    // ---------------------------------------------------------------
    // Core I/O tensors
    // ---------------------------------------------------------------

    // All token-aligned draft portals share the physical token row address space.
    reg.addTensor({binding_names::kInputsEmbeds, TensorIO::kInput, nvinfer1::DataType::kHALF,
        {sym(&InferenceDims::seqLen), fixed(draftHiddenSize)}});
    addUnifiedDecoderMetadata(reg);

    // hidden_states_input: [physicalTokens, baseOutputHiddenDim] HALF
    reg.addTensor({binding_names::kBaseModelHiddenStates, TensorIO::kInput, nvinfer1::DataType::kHALF,
        {sym(&InferenceDims::seqLen), fixed(baseOutputHiddenDim)}});

    // hidden_states_from_draft: [physicalTokens, draftHiddenSize] HALF
    reg.addTensor({binding_names::kDraftModelHiddenStates, TensorIO::kInput, nvinfer1::DataType::kHALF,
        {sym(&InferenceDims::seqLen), fixed(draftHiddenSize)}});

    addKVPageTableSpec(reg, cfg);

    if (cfg.contextMaskSelectorEnabled)
    {
        reg.addTensor({binding_names::kContextMaskSelector, TensorIO::kInput, nvinfer1::DataType::kINT32,
            {sym(&InferenceDims::contextMaskSelectorLen)}});
    }

    // RoPE cache inputs: single binding for single-RoPE models, explicit
    // sliding/full bindings for mixed-attention dual-RoPE models.
    addRopeTensorSpecs(reg, cfg);

    // attention_mask: [physicalTokens, packed_mask_len] INT32. Each row stores
    // the packed mask for one sequence-local proposal span.
    reg.addTensor({binding_names::kAttentionMask, TensorIO::kInput, nvinfer1::DataType::kINT32,
        {sym(&InferenceDims::attnMaskSeqLen), sym(&InferenceDims::packedMaskLen)}});

    // attention_pos_id: [physicalTokens] INT32
    reg.addTensor({binding_names::kAttentionPosId, TensorIO::kInput, nvinfer1::DataType::kINT32,
        {sym(&InferenceDims::attnMaskSeqLen)}});

    // ---------------------------------------------------------------
    // Outputs
    // ---------------------------------------------------------------

    // logits: [selectedRows, draftVocabSize] FLOAT
    reg.addTensor({binding_names::kLogits, TensorIO::kOutput, nvinfer1::DataType::kFLOAT,
        {sym(&InferenceDims::selectLen), fixed(draftVocabSize)}});

    // hidden_states (output): [selectedRows, draftHiddenSize] HALF
    reg.addTensor({binding_names::kOutputHiddenStates, TensorIO::kOutput, nvinfer1::DataType::kHALF,
        {sym(&InferenceDims::selectLen), fixed(draftHiddenSize)}});

    // ---------------------------------------------------------------
    // KV cache (draft engine always uses plugin path)
    // ---------------------------------------------------------------
    //
    // Draft engines today are uniform (single numKVHeads / headDim across all
    // attention layers), and their `layerTypes` are broadcast from
    // `numAttentionLayers` by `parseDraftEngineConfig`. We still walk per-layer
    // so this stays forward-compatible with future heterogeneous draft configs.
    {
        int32_t localAttnIdx = 0;
        for (int32_t absIdx = 0; absIdx < static_cast<int32_t>(cfg.layerTypes.size()); ++absIdx)
        {
            if (cfg.layerTypes[absIdx] != rt::HybridCacheManager::LayerType::kAttention)
            {
                // Current draft engines are not expected to contain Mamba layers. If a
                // future config exercises this branch it's a config error, not a
                // registry-builder concern.
                continue;
            }
            auto const& lc = cfg.kvLayerConfigs[localAttnIdx];
            // Plugin: paged pool, 5D [2, numPages, kTOKENS_PER_PAGE, numKVHeads, headDim] — same
            // contract as the base LLM engine (see buildRegistryForLLM); EAGLE/MTP drafts share the
            // AttentionPlugin binding via KVCacheManager's paged-pool view.
            int32_t const numPages = computeNumPages(cfg, lc);
            std::vector<ShapeDim> const shape{
                fixed(2), fixed(numPages), fixed(kTokensPerPage), fixed(lc.numKVHeads), fixed(lc.headDim)};
            auto addKVCacheTensor = [&](char const* tmpl, TensorIO io) {
                reg.addTensor({std::string(tmpl) + "_" + std::to_string(localAttnIdx), io, cfg.kvCacheDtype, shape});
            };
            addKVCacheTensor(binding_names::kPastKeyValuesTemplate, TensorIO::kInput);
            addKVCacheTensor(binding_names::kPresentKeyValuesTemplate, TensorIO::kOutput);
            ++localAttnIdx;
        }
    }

    return reg;
}

TensorRegistry buildRegistryForDFlashDraft(DeploymentConfig const& bundle)
{
    check::check(bundle.draft.has_value(), "buildRegistryForDFlashDraft: bundle.draft must be set");
    check::check(bundle.specConfig.has_value(), "buildRegistryForDFlashDraft: bundle.specConfig must be set");

    TensorRegistry reg;
    LLMEngineConfig const& cfg = *bundle.draft;
    bool const isDFlash2 = cfg.dflashVersion == DFlashVersion::kV2;
    int32_t const draftHiddenSize = bundle.specConfig->draftHiddenSize;
    int32_t const baseOutputHiddenDim = bundle.specConfig->baseOutputHiddenDim;
    int32_t const draftVocabSize = cfg.outputVocabSize;

    reg.addTensor({binding_names::kInputsEmbeds, TensorIO::kInput, nvinfer1::DataType::kHALF,
        {sym(&InferenceDims::seqLen), fixed(draftHiddenSize)}});
    addUnifiedDecoderMetadata(reg, /*includeLogitsIndices=*/false);

    // Token-major target-hidden delta portal.
    reg.addTensor({binding_names::kDFlashTargetHiddenConcat, TensorIO::kInput, nvinfer1::DataType::kHALF,
        {sym(&InferenceDims::selectLen), fixed(baseOutputHiddenDim)}});

    if (isDFlash2)
    {
        int32_t const proposalLen = bundle.specConfig->dflashBlockSize - 1;
        int32_t const selectorTopK = cfg.specSelectorTopK;
        check::check(proposalLen >= 1 && proposalLen <= 15 && selectorTopK == 16,
            "DFlash2 draft registry requires runtime block_size in [2, 16] and selector_top_k=16");
        reg.addTensor({binding_names::kSpecProposalSupportIds, TensorIO::kOutput, nvinfer1::DataType::kINT32,
            {sym(&InferenceDims::batch), fixed(proposalLen), fixed(selectorTopK)}});
        reg.addTensor({binding_names::kSpecProposalUnaryValues, TensorIO::kOutput, nvinfer1::DataType::kFLOAT,
            {sym(&InferenceDims::batch), fixed(proposalLen), fixed(selectorTopK)}});
        reg.addTensor({binding_names::kSpecProposalProjectedHidden, TensorIO::kOutput, nvinfer1::DataType::kHALF,
            {sym(&InferenceDims::batch), fixed(proposalLen), fixed(cfg.specSelectorRank)}});
    }
    else
    {
        // logits: [physicalTokens, draftVocabSize] FLOAT
        reg.addTensor({binding_names::kLogits, TensorIO::kOutput, nvinfer1::DataType::kFLOAT,
            {sym(&InferenceDims::seqLen), fixed(draftVocabSize)}});
    }

    addKVPageTableSpec(reg, cfg);

    reg.addTensor({binding_names::kDFlashDeltaRopeCosSin, TensorIO::kInput, nvinfer1::DataType::kFLOAT,
        {sym(&InferenceDims::selectLen), fixed(cfg.rotaryDim)}});
    reg.addTensor({binding_names::kDFlashDeltaPositions, TensorIO::kInput, nvinfer1::DataType::kINT32,
        {sym(&InferenceDims::selectLen)}});
    reg.addTensor({binding_names::kDFlashDeltaTokenToSequence, TensorIO::kInput, nvinfer1::DataType::kINT32,
        {sym(&InferenceDims::selectLen)}});

    // rope_rotary_cos_sin: [physicalTokens, rotaryDim] FLOAT
    reg.addTensor({binding_names::kRopeCosSin, TensorIO::kInput, nvinfer1::DataType::kFLOAT,
        {sym(&InferenceDims::seqLen), fixed(cfg.rotaryDim)}});

    // attention mask: [physicalTokens, packedMaskLen] INT32
    reg.addTensor({binding_names::kAttentionMask, TensorIO::kInput, nvinfer1::DataType::kINT32,
        {sym(&InferenceDims::attnMaskSeqLen), sym(&InferenceDims::packedMaskLen)}});

    // attention positions: [physicalTokens] INT32
    reg.addTensor({binding_names::kAttentionPosId, TensorIO::kInput, nvinfer1::DataType::kINT32,
        {sym(&InferenceDims::attnMaskSeqLen)}});

    // Per-layer KV cache (plugin path: combined KV)
    {
        int32_t localAttnIdx = 0;
        for (int32_t absIdx = 0; absIdx < static_cast<int32_t>(cfg.layerTypes.size()); ++absIdx)
        {
            if (cfg.layerTypes[absIdx] != rt::HybridCacheManager::LayerType::kAttention)
            {
                continue;
            }
            auto const& lc = cfg.kvLayerConfigs[localAttnIdx];
            // DFlash's own combined draft cache uses the same paged-pool contract and the exact
            // serialized engine page count.
            int32_t const numPages = cfg.kvPoolPages;
            std::vector<ShapeDim> const shape{
                fixed(2), fixed(numPages), fixed(kTokensPerPage), fixed(lc.numKVHeads), fixed(lc.headDim)};
            auto addKVCacheTensor = [&](char const* tmpl, TensorIO io) {
                reg.addTensor({std::string(tmpl) + "_" + std::to_string(localAttnIdx), io, cfg.kvCacheDtype, shape});
            };
            addKVCacheTensor(binding_names::kPastKeyValuesTemplate, TensorIO::kInput);
            addKVCacheTensor(binding_names::kPresentKeyValuesTemplate, TensorIO::kOutput);
            ++localAttnIdx;
        }
    }

    return reg;
}

TensorRegistry buildRegistryForGemma4MTPDraft(DeploymentConfig const& bundle)
{
    check::check(bundle.draft.has_value(), "buildRegistryForGemma4MTPDraft: bundle.draft must be set");
    check::check(bundle.specConfig.has_value(), "buildRegistryForGemma4MTPDraft: bundle.specConfig must be set");
    check::check(bundle.specDecodeMode() == SpecDecodeMode::kGemma4MTP,
        "buildRegistryForGemma4MTPDraft requires spec_decode_type=gemma4_mtp");

    TensorRegistry reg;
    LLMEngineConfig const& draftCfg = *bundle.draft;
    int32_t const baseOutputHiddenDim = bundle.specConfig->baseOutputHiddenDim;
    int32_t const draftVocabSize = draftCfg.outputVocabSize;

    reg.addTensor({binding_names::kInputsEmbeds, TensorIO::kInput, nvinfer1::DataType::kHALF,
        {sym(&InferenceDims::seqLen), fixed(baseOutputHiddenDim)}});
    addUnifiedDecoderMetadata(reg, /*includeLogitsIndices=*/false);

    // hidden_states_input: [B, 1, Hb] target hidden seed or assistant feedback hidden.
    reg.addTensor({binding_names::kBaseModelHiddenStates, TensorIO::kInput, nvinfer1::DataType::kHALF,
        {sym(&InferenceDims::seqLen), fixed(baseOutputHiddenDim)}});

    addKVPageTableSpec(reg, bundle.base);

    addRopeTensorSpecs(reg, draftCfg);

    // logits: [B, vocab] full-logits correctness path.
    reg.addTensor({binding_names::kLogits, TensorIO::kOutput, nvinfer1::DataType::kFLOAT,
        {sym(&InferenceDims::selectLen), fixed(draftVocabSize)}});

    // hidden_states: [B, 1, Hb] assistant feedback hidden in target backbone space.
    reg.addTensor({binding_names::kOutputHiddenStates, TensorIO::kOutput, nvinfer1::DataType::kHALF,
        {sym(&InferenceDims::seqLen), fixed(baseOutputHiddenDim)}});

    for (auto const& entry : draftCfg.gemma4MTPKVSharingMap)
    {
        check::check(entry.assistantLayerIdx >= 0 && entry.assistantLayerIdx < draftCfg.numAttentionLayers,
            "buildRegistryForGemma4MTPDraft: invalid assistant layer index");
        check::check(entry.targetAttentionLayerIdx >= 0
                && entry.targetAttentionLayerIdx < static_cast<int32_t>(bundle.base.kvLayerConfigs.size()),
            "buildRegistryForGemma4MTPDraft: invalid target attention layer index");

        // The assistant binds the TARGET model's paged pool tensor directly, so the
        // expected shape is the target pool contract [2, numPages, kTOKENS_PER_PAGE,
        // numKVHeads, headDim] with the target's runtime-chosen numPages.
        auto const& targetKV = bundle.base.kvLayerConfigs[entry.targetAttentionLayerIdx];
        int32_t const targetNumPages = computeNumPages(bundle.base, targetKV);
        std::vector<ShapeDim> const shape{fixed(2), fixed(targetNumPages), fixed(kTokensPerPage),
            fixed(targetKV.numKVHeads), fixed(targetKV.headDim)};
        reg.addTensor({binding_names::formatKVCacheName(entry.assistantLayerIdx, /*isPast=*/true), TensorIO::kInput,
            bundle.base.kvCacheDtype, shape});
    }

    return reg;
}

TensorRegistry buildRegistryForDSparkDraft(DeploymentConfig const& bundle)
{
    check::check(bundle.draft.has_value(), "buildRegistryForDSparkDraft: bundle.draft must be set");
    check::check(bundle.specConfig.has_value(), "buildRegistryForDSparkDraft: bundle.specConfig must be set");

    TensorRegistry reg;
    LLMEngineConfig const& cfg = *bundle.draft;
    int32_t const draftHiddenSize = bundle.specConfig->draftHiddenSize;
    int32_t const baseOutputHiddenDim = bundle.specConfig->baseOutputHiddenDim;
    int32_t const draftVocabSize = cfg.outputVocabSize;

    reg.addTensor({binding_names::kInputsEmbeds, TensorIO::kInput, nvinfer1::DataType::kHALF,
        {sym(&InferenceDims::seqLen), fixed(draftHiddenSize)}});
    addUnifiedDecoderMetadata(reg, /*includeLogitsIndices=*/false);

    // dflash_target_hidden_concat: [deltaTokens, baseOutputHiddenDim] HALF
    reg.addTensor({binding_names::kDFlashTargetHiddenConcat, TensorIO::kInput, nvinfer1::DataType::kHALF,
        {sym(&InferenceDims::selectLen), fixed(baseOutputHiddenDim)}});

    // logits: [physicalTokens, draftVocabSize] FLOAT
    reg.addTensor({binding_names::kLogits, TensorIO::kOutput, nvinfer1::DataType::kFLOAT,
        {sym(&InferenceDims::seqLen), fixed(draftVocabSize)}});

    // dspark_hidden_states: [physicalTokens, draftHiddenSize] HALF
    reg.addTensor({binding_names::kDSparkHiddenStates, TensorIO::kOutput, nvinfer1::DataType::kHALF,
        {sym(&InferenceDims::seqLen), fixed(draftHiddenSize)}});

    addKVPageTableSpec(reg, cfg);

    reg.addTensor({binding_names::kDFlashDeltaRopeCosSin, TensorIO::kInput, nvinfer1::DataType::kFLOAT,
        {sym(&InferenceDims::selectLen), fixed(cfg.rotaryDim)}});
    reg.addTensor({binding_names::kDFlashDeltaPositions, TensorIO::kInput, nvinfer1::DataType::kINT32,
        {sym(&InferenceDims::selectLen)}});
    reg.addTensor({binding_names::kDFlashDeltaTokenToSequence, TensorIO::kInput, nvinfer1::DataType::kINT32,
        {sym(&InferenceDims::selectLen)}});

    // rope_rotary_cos_sin: [physicalTokens, rotaryDim] FLOAT
    reg.addTensor({binding_names::kRopeCosSin, TensorIO::kInput, nvinfer1::DataType::kFLOAT,
        {sym(&InferenceDims::seqLen), fixed(cfg.rotaryDim)}});

    // attention mask: [physicalTokens, packedMaskLen] INT32
    reg.addTensor({binding_names::kAttentionMask, TensorIO::kInput, nvinfer1::DataType::kINT32,
        {sym(&InferenceDims::attnMaskSeqLen), sym(&InferenceDims::packedMaskLen)}});

    // attention positions: [physicalTokens] INT32
    reg.addTensor({binding_names::kAttentionPosId, TensorIO::kInput, nvinfer1::DataType::kINT32,
        {sym(&InferenceDims::attnMaskSeqLen)}});

    // Per-layer KV cache (plugin path: combined KV)
    {
        int32_t localAttnIdx = 0;
        for (int32_t absIdx = 0; absIdx < static_cast<int32_t>(cfg.layerTypes.size()); ++absIdx)
        {
            if (cfg.layerTypes[absIdx] != rt::HybridCacheManager::LayerType::kAttention)
            {
                continue;
            }
            auto const& lc = cfg.kvLayerConfigs[localAttnIdx];
            // DSpark uses the exact serialized engine page count.
            int32_t const numPages = cfg.kvPoolPages;
            std::vector<ShapeDim> const shape{
                fixed(2), fixed(numPages), fixed(kTokensPerPage), fixed(lc.numKVHeads), fixed(lc.headDim)};
            auto addKVCacheTensor = [&](char const* tmpl, TensorIO io) {
                reg.addTensor({std::string(tmpl) + "_" + std::to_string(localAttnIdx), io, cfg.kvCacheDtype, shape});
            };
            addKVCacheTensor(binding_names::kPastKeyValuesTemplate, TensorIO::kInput);
            addKVCacheTensor(binding_names::kPresentKeyValuesTemplate, TensorIO::kOutput);
            ++localAttnIdx;
        }
    }

    return reg;
}

} // namespace rt
} // namespace trt_edgellm
