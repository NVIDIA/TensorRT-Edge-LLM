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
#include "kernels/gdnKernels/gdnTreeChunkKernels.h"

namespace trt_edgellm
{
namespace rt
{

// Note: `sym(...)` and `fixed(...)` helpers are now provided by inferenceDims.h.
// Symbolic-dim registration (formerly `addCommonSymbolicDims` / `addSymbolicDim`)
// is no longer needed — every symbolic reference is a pointer-to-member of
// `InferenceDims`, so the set of dims exists by construction of the type.

//! Page count for paged-pool KV-cache bindings
//! [2, numPages, kTOKENS_PER_PAGE, numKVHeads, headDim]. A single fixed value per engine
//! (not resized per inference step), matching the serialized builder profile and
//! KVCacheManager::numPages() (see sharedResources.cpp).
constexpr int32_t kTokensPerPage = rt::kTOKENS_PER_PAGE;

static int32_t computeNumPages(LLMEngineConfig const& cfg)
{
    int32_t const floorPages = rt::computeKvPoolFloorPages(cfg.maxSupportedBatchSize, cfg.maxKVCacheCapacity);
    ELLM_CHECK(cfg.kvPoolPages >= floorPages,
        "KV pool page count (" + std::to_string(cfg.kvPoolPages) + ") is smaller than the active-capacity floor ("
            + std::to_string(floorPages) + ").");
    return cfg.kvPoolPages;
}

static int32_t computeFloorNumPages(int32_t maxBatchSize, int32_t maxKVCacheCapacity)
{
    return rt::computeKvPoolFloorPages(maxBatchSize, maxKVCacheCapacity);
}

void addRopeTensorSpecs(TensorRegistry& reg, LLMEngineConfig const& cfg)
{
    auto addRopeTensor = [&](char const* name, int32_t rotaryDim) {
        reg.addTensor({name, TensorIO::kInput, nvinfer1::DataType::kFLOAT,
            {sym(&InferenceDims::ropeBatch), sym(&InferenceDims::kvLen), fixed(rotaryDim)}});
    };

    if (cfg.useDualRope)
    {
        addRopeTensor(binding_names::kRopeCosSinSliding, cfg.slidingRotaryDim);
        addRopeTensor(binding_names::kRopeCosSinFull, cfg.fullRotaryDim);
        return;
    }

    addRopeTensor(binding_names::kRopeCosSin, cfg.rotaryDim);
}

TensorRegistry buildRegistryForLLM(LLMEngineConfig const& cfg, std::optional<int32_t> specDecodeBaseOutputHiddenDim)
{
    TensorRegistry reg;

    // ---------------------------------------------------------------
    // Core I/O tensors (always present)
    // ---------------------------------------------------------------

    // inputs_embeds: [batch, seq_len, hiddenSize] HALF
    reg.addTensor({binding_names::kInputsEmbeds, TensorIO::kInput, nvinfer1::DataType::kHALF,
        {sym(&InferenceDims::batch), sym(&InferenceDims::seqLen), fixed(cfg.hiddenSize)}});

    // logits: [batch, outputVocabSize] FLOAT for vanilla. SpecDecode engines may
    // expose [batch, selectLen, outputVocabSize] or the equivalent flattened
    // [batch * selectLen, outputVocabSize]. Output bindings are address-only, so
    // the registry uses the common 2D runtime view.
    reg.addTensor({binding_names::kLogits, TensorIO::kOutput, nvinfer1::DataType::kFLOAT,
        {sym(&InferenceDims::batch), fixed(cfg.outputVocabSize)}});

    // context_lengths: [batch] INT32
    reg.addTensor(
        {binding_names::kContextLengths, TensorIO::kInput, nvinfer1::DataType::kINT32, {sym(&InferenceDims::batch)}});

    // last_token_ids: [batch, select_len] INT64 — always [batch, 1] for vanilla, varies for SpecDecode.
    reg.addTensor({binding_names::kLastTokenIds, TensorIO::kInput, nvinfer1::DataType::kINT64,
        {sym(&InferenceDims::batch), sym(&InferenceDims::selectLen)}});

    // kvcache_start_index: [start_index_len] INT32. The engine's context profile
    // uses shape [0] as a sentinel for "initial prefill of an empty KV cache";
    // chunked prefill, decode, and verification use [batch] start offsets.
    // InferenceDims::startIndexLen carries this per-phase: prefillDims sets it to 0
    // when kvCacheAllEmpty, else batch; all other recipes
    // set it to batch. Shape 0 is engine-valid here — TRT reads 0 bytes from
    // the bound address and the engine branches to the initial-prefill path.
    reg.addTensor({binding_names::kKVCacheStartIndex, TensorIO::kInput, nvinfer1::DataType::kINT32,
        {sym(&InferenceDims::startIndexLen)}});

    if (cfg.useVisionBidirectionalAttention)
    {
        reg.addTensor({binding_names::kVisionBlockIds, TensorIO::kInput, nvinfer1::DataType::kINT32,
            {sym(&InferenceDims::batch), sym(&InferenceDims::seqLen)}});
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
            // numPages is fixed for the life of this engine context (see computeNumPages).
            int32_t const numPages = computeNumPages(cfg);
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
            // recurrent_state_%d: [batch, recurrentStateNumHeads, recurrentStateHeadDim, recurrentStateSize]
            std::vector<ShapeDim> const recShape{sym(&InferenceDims::batch), fixed(cfg.recurrentStateNumHeads),
                fixed(cfg.recurrentStateHeadDim), fixed(cfg.recurrentStateSize)};
            addMambaTensor(binding_names::kRecurrentStateTemplate, TensorIO::kInput, cfg.recurrentStateDtype, recShape);
            addMambaTensor(
                binding_names::kPresentRecurrentStateTemplate, TensorIO::kOutput, cfg.recurrentStateDtype, recShape);
            // conv_state_%d: [batch, convDim, convKernel]
            std::vector<ShapeDim> const convShape{
                sym(&InferenceDims::batch), fixed(cfg.convDim), fixed(cfg.convKernel)};
            addMambaTensor(binding_names::kConvStateTemplate, TensorIO::kInput, cfg.convStateDtype, convShape);
            addMambaTensor(binding_names::kPresentConvStateTemplate, TensorIO::kOutput, cfg.convStateDtype, convShape);

            // Hybrid MTP/DFlash base only: per-layer intermediate state outputs
            // written during verification. Recurrent state uses an opaque
            // compact replay buffer; conv state retains per-node checkpoints.
            //
            // intermediate_recurrent_state_%d: [batch, compactElements]
            // intermediate_conv_state_%d:      [batch, seqLen, convDim, convKernel]
            if (cfg.specDecodeType == SpecDecodeMode::kMTP || cfg.specDecodeType == SpecDecodeMode::kDFlash)
            {
                int32_t const h = kernel::gdnTreeChunkNumKeyHeads(
                    cfg.convDim, cfg.recurrentStateNumHeads, cfg.recurrentStateHeadDim, cfg.recurrentStateSize);
                ELLM_CHECK(cfg.recurrentStateDtype == nvinfer1::DataType::kFLOAT && h > 0
                        && cfg.recurrentStateNumHeads % h == 0,
                    "Hybrid speculative base has an invalid compact GDN replay-buffer contract.");
                int64_t const compactElements = kernel::gdnTreeChunkBufferElements(h, cfg.recurrentStateNumHeads);
                std::vector<ShapeDim> const interRecShape{sym(&InferenceDims::batch), fixed(compactElements)};
                addMambaTensor(binding_names::kIntermediateRecurrentStateTemplate, TensorIO::kOutput,
                    nvinfer1::DataType::kFLOAT, interRecShape);
                if (cfg.convDim > 0 && cfg.convKernel > 0)
                {
                    std::vector<ShapeDim> const interConvShape{sym(&InferenceDims::batch), sym(&InferenceDims::seqLen),
                        fixed(cfg.convDim), fixed(cfg.convKernel)};
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
    if (cfg.numDeepstackFeatures > 0)
    {
        // deepstack_embeds_%d: [batch, seq_len, hiddenSize] HALF — one per feature.
        // DeepstackBinding swaps the backing tensor (real per-request buffer
        // vs. shared zero buffer) between prefill and non-prefill phases.
        reg.addTensor({std::string(binding_names::kDeepstackEmbedsTemplate) + "_%d", TensorIO::kInput,
            nvinfer1::DataType::kHALF, {sym(&InferenceDims::batch), sym(&InferenceDims::seqLen), fixed(cfg.hiddenSize)},
            /*perLayer=*/cfg.numDeepstackFeatures});
    }

    // ---------------------------------------------------------------
    // SpecDecode verification bindings (base engine side)
    // ---------------------------------------------------------------
    if (cfg.isSpecDecodeBase)
    {
        // hidden_states: output, [batch, outputHiddenDim] for vanilla decode,
        // or [batch, seq_len, outputHiddenDim] for prefill/verification — use symbolic.
        // The concrete output hidden dim is strategy-specific and is consolidated
        // in DeploymentConfig::specDecode.
        int32_t const baseOutputHiddenDim = specDecodeBaseOutputHiddenDim.value_or(cfg.hiddenSize * 3);
        reg.addTensor({binding_names::kOutputHiddenStates, TensorIO::kOutput, nvinfer1::DataType::kHALF,
            {sym(&InferenceDims::batch), fixed(baseOutputHiddenDim)}});

        // attention_mask: [batch, attn_seq_len, packed_mask_len] INT32 for proposal verification
        // packed_mask_len = divUp(attn_seq_len, 32): each INT32 stores 32 mask bits.
        // attn_seq_len is decoupled from seq_len so prefill/decode/reset can pin
        // it to 1 (engine then uses standard causal attention) while verify,
        // proposal, and accept use the effective proposal size.
        reg.addTensor({binding_names::kAttentionMask, TensorIO::kInput, nvinfer1::DataType::kINT32,
            {sym(&InferenceDims::batch), sym(&InferenceDims::attnMaskSeqLen), sym(&InferenceDims::packedMaskLen)}});

        // attention_pos_id: [batch, attn_seq_len] INT32
        reg.addTensor({binding_names::kAttentionPosId, TensorIO::kInput, nvinfer1::DataType::kINT32,
            {sym(&InferenceDims::batch), sym(&InferenceDims::attnMaskSeqLen)}});

        if ((cfg.specDecodeType == SpecDecodeMode::kMTP || cfg.specDecodeType == SpecDecodeMode::kDFlash)
            && cfg.numLinearAttnLayers > 0)
        {
            reg.addTensor({binding_names::kSpecVerifyPhaseMarker, TensorIO::kInput, nvinfer1::DataType::kINT32,
                {sym(&InferenceDims::specVerifyPhaseLen)}});
        }
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

    // inputs_embeds: [batch, seq_len, draftHiddenSize] HALF
    reg.addTensor({binding_names::kInputsEmbeds, TensorIO::kInput, nvinfer1::DataType::kHALF,
        {sym(&InferenceDims::batch), sym(&InferenceDims::seqLen), fixed(draftHiddenSize)}});

    // hidden_states_input (base model hidden states): [batch, seq_len, baseOutputHiddenDim] HALF
    reg.addTensor({binding_names::kBaseModelHiddenStates, TensorIO::kInput, nvinfer1::DataType::kHALF,
        {sym(&InferenceDims::batch), sym(&InferenceDims::seqLen), fixed(baseOutputHiddenDim)}});

    // hidden_states_from_draft (draft model hidden states): [batch, seq_len, draftHiddenSize] HALF
    reg.addTensor({binding_names::kDraftModelHiddenStates, TensorIO::kInput, nvinfer1::DataType::kHALF,
        {sym(&InferenceDims::batch), sym(&InferenceDims::seqLen), fixed(draftHiddenSize)}});

    // last_token_ids: [batch, select_len] INT64
    reg.addTensor({binding_names::kLastTokenIds, TensorIO::kInput, nvinfer1::DataType::kINT64,
        {sym(&InferenceDims::batch), sym(&InferenceDims::selectLen)}});

    // context_lengths: [batch] INT32
    reg.addTensor(
        {binding_names::kContextLengths, TensorIO::kInput, nvinfer1::DataType::kINT32, {sym(&InferenceDims::batch)}});

    // kvcache_start_index: [start_index_len] INT32. Draft engine always uses
    // the plugin path: shape [0] sentinel during round-0 prefill (empty draft
    // KV cache); [batch] for proposal / accept.
    reg.addTensor({binding_names::kKVCacheStartIndex, TensorIO::kInput, nvinfer1::DataType::kINT32,
        {sym(&InferenceDims::startIndexLen)}});

    // RoPE cache inputs: single binding for single-RoPE models, explicit
    // sliding/full bindings for mixed-attention dual-RoPE models.
    addRopeTensorSpecs(reg, cfg);

    // attention_mask: [batch, attn_seq_len, packed_mask_len] INT32 — tree decoding mask
    // packed_mask_len = divUp(attn_seq_len, 32): each INT32 stores 32 mask bits.
    // attn_seq_len is decoupled from seq_len so prefill/decode/reset can pin
    // it to 1 (engine then uses standard causal attention) while tree
    // verify/proposal/accept use the effective tree size.
    reg.addTensor({binding_names::kAttentionMask, TensorIO::kInput, nvinfer1::DataType::kINT32,
        {sym(&InferenceDims::batch), sym(&InferenceDims::attnMaskSeqLen), sym(&InferenceDims::packedMaskLen)}});

    // attention_pos_id: [batch, attn_seq_len] INT32
    reg.addTensor({binding_names::kAttentionPosId, TensorIO::kInput, nvinfer1::DataType::kINT32,
        {sym(&InferenceDims::batch), sym(&InferenceDims::attnMaskSeqLen)}});

    // ---------------------------------------------------------------
    // Outputs
    // ---------------------------------------------------------------

    // logits: [batch, draftVocabSize] FLOAT
    reg.addTensor({binding_names::kLogits, TensorIO::kOutput, nvinfer1::DataType::kFLOAT,
        {sym(&InferenceDims::batch), fixed(draftVocabSize)}});

    // hidden_states (output): [batch, draftHiddenSize] HALF
    reg.addTensor({binding_names::kOutputHiddenStates, TensorIO::kOutput, nvinfer1::DataType::kHALF,
        {sym(&InferenceDims::batch), fixed(draftHiddenSize)}});

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
            int32_t const numPages = computeNumPages(cfg);
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
    int32_t const draftHiddenSize = bundle.specConfig->draftHiddenSize;
    int32_t const baseOutputHiddenDim = bundle.specConfig->baseOutputHiddenDim;
    int32_t const draftVocabSize = cfg.outputVocabSize;

    // inputs_embeds: [batch, seq_len, draftHiddenSize] HALF
    reg.addTensor({binding_names::kInputsEmbeds, TensorIO::kInput, nvinfer1::DataType::kHALF,
        {sym(&InferenceDims::batch), sym(&InferenceDims::seqLen), fixed(draftHiddenSize)}});

    // dflash_target_hidden_concat: [batch, selectLen, baseOutputHiddenDim] HALF — target hidden delta
    reg.addTensor({binding_names::kDFlashTargetHiddenConcat, TensorIO::kInput, nvinfer1::DataType::kHALF,
        {sym(&InferenceDims::batch), sym(&InferenceDims::selectLen), fixed(baseOutputHiddenDim)}});

    // logits: [batch, seq_len, draftVocabSize] FLOAT
    reg.addTensor({binding_names::kLogits, TensorIO::kOutput, nvinfer1::DataType::kFLOAT,
        {sym(&InferenceDims::batch), sym(&InferenceDims::seqLen), fixed(draftVocabSize)}});

    // context_lengths: [batch] INT32
    reg.addTensor(
        {binding_names::kContextLengths, TensorIO::kInput, nvinfer1::DataType::kINT32, {sym(&InferenceDims::batch)}});

    // kvcache_start_index: [startIndexLen] INT32
    reg.addTensor({binding_names::kKVCacheStartIndex, TensorIO::kInput, nvinfer1::DataType::kINT32,
        {sym(&InferenceDims::startIndexLen)}});

    // dflash_delta_lengths: [batch] INT32 — per-batch delta lengths for multi-batch
    reg.addTensor({binding_names::kDFlashDeltaLengths, TensorIO::kInput, nvinfer1::DataType::kINT32,
        {sym(&InferenceDims::batch)}});

    // rope_rotary_cos_sin: [ropeBatch, kvLen, rotaryDim] FLOAT
    reg.addTensor({binding_names::kRopeCosSin, TensorIO::kInput, nvinfer1::DataType::kFLOAT,
        {sym(&InferenceDims::ropeBatch), sym(&InferenceDims::kvLen), fixed(cfg.rotaryDim)}});

    // attention_mask: [batch, attnMaskSeqLen, packedMaskLen] INT32
    reg.addTensor({binding_names::kAttentionMask, TensorIO::kInput, nvinfer1::DataType::kINT32,
        {sym(&InferenceDims::batch), sym(&InferenceDims::attnMaskSeqLen), sym(&InferenceDims::packedMaskLen)}});

    // attention_pos_id: [batch, attnMaskSeqLen] INT32
    reg.addTensor({binding_names::kAttentionPosId, TensorIO::kInput, nvinfer1::DataType::kINT32,
        {sym(&InferenceDims::batch), sym(&InferenceDims::attnMaskSeqLen)}});

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
            // DFlash's own combined draft cache: unified on the paged-pool contract shared with
            // the main model and EAGLE/MTP drafts — [2, numPages, kTOKENS_PER_PAGE, numKVHeads,
            // headDim] — but NOT extended by the pool-capacity profile range:
            // DFlashTargetKVCacheUpdatePlugin recovers maxBatch/cap at enqueue time by dividing
            // numPages by the builder-configured pages_per_slot attribute, so this binding always
            // stays at the active-capacity floor (see computeFloorNumPages). This cache still has
            // no page table of its own (identity-contiguous, reuse opt-out).
            int32_t const numPages = computeFloorNumPages(cfg.maxSupportedBatchSize, cfg.maxKVCacheCapacity);
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

    // inputs_embeds: [B, 1, Hb] target/base embedding table output.
    reg.addTensor({binding_names::kInputsEmbeds, TensorIO::kInput, nvinfer1::DataType::kHALF,
        {sym(&InferenceDims::batch), sym(&InferenceDims::seqLen), fixed(baseOutputHiddenDim)}});

    // hidden_states_input: [B, 1, Hb] target hidden seed or assistant feedback hidden.
    reg.addTensor({binding_names::kBaseModelHiddenStates, TensorIO::kInput, nvinfer1::DataType::kHALF,
        {sym(&InferenceDims::batch), sym(&InferenceDims::seqLen), fixed(baseOutputHiddenDim)}});

    // context_lengths: [B] target KV lengths.
    reg.addTensor(
        {binding_names::kContextLengths, TensorIO::kInput, nvinfer1::DataType::kINT32, {sym(&InferenceDims::batch)}});

    addRopeTensorSpecs(reg, draftCfg);

    // logits: [B, vocab] full-logits correctness path.
    reg.addTensor({binding_names::kLogits, TensorIO::kOutput, nvinfer1::DataType::kFLOAT,
        {sym(&InferenceDims::batch), fixed(draftVocabSize)}});

    // hidden_states: [B, 1, Hb] assistant feedback hidden in target backbone space.
    reg.addTensor({binding_names::kOutputHiddenStates, TensorIO::kOutput, nvinfer1::DataType::kHALF,
        {sym(&InferenceDims::batch), sym(&InferenceDims::seqLen), fixed(baseOutputHiddenDim)}});

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
        int32_t const targetNumPages = computeNumPages(bundle.base);
        std::vector<ShapeDim> const shape{fixed(2), fixed(targetNumPages), fixed(kTokensPerPage),
            fixed(targetKV.numKVHeads), fixed(targetKV.headDim)};
        reg.addTensor({binding_names::formatKVCacheName(entry.assistantLayerIdx, /*isPast=*/true), TensorIO::kInput,
            bundle.base.kvCacheDtype, shape});
    }

    return reg;
}

} // namespace rt
} // namespace trt_edgellm
