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

#pragma once

#include "runtime/config/inferenceDims.h"
#include "runtime/embedding/pleTable.h"
#include "runtime/hybridCacheManager.h"
#include "runtime/kvCacheManager.h"
#include "runtime/llmRuntimeUtils.h"
#include "runtime/recurrentStateManager.h"

#include <array>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace trt_edgellm
{
namespace rt
{

class EngineExecutor;

//! Speculative decoding strategy mode.
enum class SpecDecodeMode : int32_t
{
    kNONE,
    kEAGLE,
    kMTP,
    kDFlash,
    kJetSpec,
    kGemma4MTP,
    kDSpark,
    kDomino,
};

enum class DFlashVersion : int32_t
{
    kV1 = 1,
    kV2 = 2,
};

//! DFlash V2 candidate projector, selected by `dflash_config.projector_type`.
enum class DFlashProjector : int32_t
{
    kSelector = 0,
    kLiLiCorr = 1,
};

//! Runtime storage policy for engines carrying bounded-SWA capability metadata.
enum class SwaKVCacheMode : int32_t
{
    kFull,
    kBounded,
};

char const* specDecodeModeName(SpecDecodeMode mode) noexcept;
bool isCachedBlockDraftMode(SpecDecodeMode mode) noexcept;

enum class RaggedBackendKind : int32_t
{
    kNone,
    kEntryPaddedCompatibility,
};

//! Gemma4 MTP assistant-layer to target-layer shared-KV mapping.
struct Gemma4MTPKVSharingEntry
{
    int32_t assistantLayerIdx{-1};       //!< Assistant local attention-layer index.
    int32_t targetAttentionLayerIdx{-1}; //!< Target local attention-layer index.
    int32_t targetAbsoluteLayerIdx{-1};  //!< Target absolute decoder-layer index, derived during validation.
};

//! Immutable external table metadata for the PLE pool/map interface.
struct PlePoolConfig
{
    int64_t rowCount{};
    int64_t rowWidth{};
    PleTable::DType dtype{PleTable::DType::kFP16};
    double scale{1.0};
    PleTable::Source source{};
    std::string checkpointDir{};
    int64_t heads{1};                  //!< Rows per token; their widths concatenate into one embedding
    std::string layerMultipliersKey{}; //!< int64 [ngram_size], odd position multipliers
    std::string headVocabSizesKey{};   //!< int64 [heads], pairwise-coprime moduli
    std::string headOffsetsKey{};      //!< int64 [heads], start row of each head's slice
};

//! One decoder layer's stateful components. The table is dense over num_hidden_layers, absolutely
//! indexed (`layers[i].absLayerIdx == i`), so a bare MLP/MoE layer still occupies its slot with no
//! kv and no states. It is the single source of truth for which states a layer carries and how each
//! is shaped.
struct LayerSpec
{
    int32_t absLayerIdx{};
    //! Set when the layer attends; absent for a purely recurrent layer. "Is attention" := has kv.
    std::optional<KVLayerConfig> kv{};
    //! LOCAL attention index donating this layer's KV, or -1 when it owns its own.
    int32_t kvSharingDonor{-1};
    //! Per-slot states this layer carries, at most one per kind. Each `shape` is one slot's row:
    //! the pool prepends its own slot count (see `RecurrentStateManager`).
    std::vector<StateSpec> states{};

    bool isAttention() const noexcept
    {
        return kv.has_value();
    }

    //! Carries mixer state, i.e. a state other than `kNgramConv`.
    bool isRecurrent() const noexcept;

    //! The state of the given lineage, or nullptr when this layer declares none.
    StateSpec const* findState(StateKind kind) const;
};

//! Parsed model and engine capabilities shared across runtime components.
struct LLMEngineConfig
{
    // --- Core model dimensions ---
    int32_t hiddenSize{};         //!< Model hidden dimension
    int32_t outputVocabSize{};    //!< Actual output vocab (reduced if vocab reduction active)
    int32_t numAttentionLayers{}; //!< Number of attention layers needing KV cache
    int32_t numKVHeads{};         //!< Number of key-value heads
    int32_t numAttentionHeads{};  //!< Number of query heads; 0 when the config omits num_attention_heads
    //! Width of the Qwen Sparse Attention indexer state on each KV pool row; > 0 when every attention layer is a
    //! QsaAttentionPlugin layer, 0 when the config omits indexer_head_dim.
    int32_t indexerHeadDim{};
    int32_t headDim{};                 //!< Dimension of each attention head
    int32_t maxSupportedBatchSize{};   //!< Maximum supported batch size
    int32_t tpSize{1};                 //!< Tensor-parallel size the engine was built for
    int32_t maxSupportedInputLength{}; //!< Maximum supported input length
    int32_t maxKVCacheCapacity{};      //!< Maximum KV cache capacity (sequence length)
    RaggedBackendKind raggedBackend{RaggedBackendKind::kNone};
    int32_t maxNumSequences{};
    int32_t maxQueryLength{};
    int32_t maxPhysicalTokens{};
    int32_t recurrentPoolRows{};
    //! builder_config.compact_execution is true: every compact-capable plugin layer reserved the compact workspace
    bool compactExecution{false};
    //! Prefill skip-softmax scale table (`skip_softmax_prefill_scale_table`): rows of (seqlen_bin, S)
    //! sorted by bin, empty = dense prefill.
    //! Uploaded once to the `skip_softmax_prefill_scale` binding; the FMHA kernel floor-looks-up S by
    //! each sequence's KV length. A flat S is the one-row table {0, S}.
    std::vector<std::array<int64_t, 2>> skipSoftmaxPrefillScaleTable;
    //! Decode skip-softmax scale table (`skip_softmax_decode_scale_table`), same row format.
    //! Uploaded once to the `skip_softmax_decode_scale` binding; empty = dense decode.
    std::vector<std::array<int64_t, 2>> skipSoftmaxDecodeScaleTable;
    int32_t kvPoolPages{}; //!< Exact physical K-page count serialized in KV binding shapes
    //! Total physical K-page budget for the independent SWA pool. Zero means
    //! no reduced pool; reduced markers require a positive persisted value with bounded publication headroom.
    int32_t numSwaPages{};
    //! Runtime-only storage selection. The exported per-layer capacity markers remain unchanged.
    SwaKVCacheMode swaKVCacheMode{SwaKVCacheMode::kBounded};
    int32_t rotaryDim{};         //!< Rotary embedding dimension
    int32_t numDecoderLayers{};  //!< Total decoder layers (attention + linear)
    int32_t vocabSize{};         //!< Full vocabulary size
    int32_t reducedVocabSize{0}; //!< 0 = no vocab reduction
    int32_t diffusionCanvasLength{0};
    int32_t diffusionMaxDenoisingSteps{0};
    int32_t diffusionSelfConditioningSize{0};
    float diffusionTMax{0.8F};
    float diffusionTMin{0.4F};
    float diffusionEntropyBound{0.1F};
    float diffusionEntropyThreshold{0.005F};
    float rmsNormEps{1.0e-6F};
    int32_t diffusionStabilityWindow{2};

    // --- Feature flags ---
    bool isSpecDecodeBase{false};             //!< Base engine exposes speculative decoding verification bindings
    bool isDiffusionBackbone{false};          //!< DiffusionGemma phase-aware transformer backbone engine
    bool diffusionUnifiedConditioning{false}; //!< Backbone engine owns DiffusionGemma self-conditioning inputs
    bool contextMaskSelectorEnabled{false};   //!< Engine exposes context_mask_selector binding
    SpecDecodeMode specDecodeType{
        SpecDecodeMode::kNONE}; //!< Speculative decoding strategy mode (parsed from spec_decode_type)
    //! KV cache data type. Parsed from required top-level `kv_cache_dtype` in
    //! `config.json` (written by `llm_export.py`). Accepted values:
    //! "fp16" → kHALF, "fp8" → kFP8, "int8" → kINT8, "bf16" → kBF16.
    //! The runtime validates this against the engine's actual KV binding dtype.
    trt_edgellm::DataType kvCacheDtype{trt_edgellm::DataType::kHALF};
    //! Per-attention-layer K/V scales that convert FP8 KV values back to FP16.
    //! Empty when the export does not provide scales for an external KV consumer.
    std::vector<std::array<float, 2>> kvCacheScales;

    //! Recurrent state data type (hybrid models only). Parsed from required
    //! top-level `recurrent_state_dtype` when `numLinearAttnLayers > 0`;
    //! left at the default otherwise. Runtime validates against the engine's
    //! recurrent-state binding dtype.
    trt_edgellm::DataType recurrentStateDtype{trt_edgellm::DataType::kHALF};
    //! Conv state data type (hybrid models only). Same shape as `recurrentStateDtype`.
    trt_edgellm::DataType convStateDtype{trt_edgellm::DataType::kHALF};

    // --- RoPE ---
    RopeConfig ropeConfig{};             //!< Full RoPE configuration
    bool useDualRope{false};             //!< Use separate RoPE caches for sliding/full attention
    RopeConfig slidingRopeConfig{};      //!< RoPE configuration for sliding attention layers
    RopeConfig fullRopeConfig{};         //!< RoPE configuration for full attention layers
    int32_t slidingRotaryDim{};          //!< Rotary dimension for sliding attention RoPE
    int32_t fullRotaryDim{};             //!< Rotary dimension for full attention RoPE
    bool useContextDependentRope{false}; //!< Use context-dependent RoPE

    // --- Optional feature fields ---
    int32_t numDeepstackFeatures{0}; //!< Deepstack features (Qwen3-VL/Qwen3-Omni)
    int32_t maxSupportedLoraRank{0}; //!< Maximum LoRA rank (0 = no LoRA)
    PlePoolConfig plePool{};         //!< External immutable table for the pool/map ABI.

    // --- Gemma4 per-layer embeddings ---
    bool gemmaPleEnabled{false};   //!< One embedding slice per decoder layer
    int32_t gemmaPleLayerCount{0}; //!< Number of slices (0 = disabled)
    int32_t gemmaPleHiddenSize{0}; //!< Hidden dimension of each slice (0 = disabled)

    // --- Qwen4-Exp n-gram embedding ---
    std::vector<int32_t> ngramLayers{}; //!< Injected decoder layers, zero-indexed; empty = disabled
    int32_t ngramEmbedDim{0};           //!< Width of the gathered n-gram vector handed to the engine
    int32_t ngramConvKernel{0};         //!< Kernel of the dilated depthwise conv; dilation is ngramSize
    int32_t ngramConvDim{0};            //!< Channels of that conv; spans every residual stream
    int32_t ngramSize{0};               //!< Longest n-gram order hashed into a row ID
    int32_t ngramHeadsPerOrder{0};      //!< Hash heads per n-gram order
    //! Data type of the injection's dilated-conv state. Independent of `convStateDtype`: the
    //! injected layer need not be recurrent, so a model may bind this state and no other.
    trt_edgellm::DataType ngramConvStateDtype{trt_edgellm::DataType::kHALF};

    //! Row IDs looked up per token: one per head, over every order but unigram.
    int32_t ngramHeads() const noexcept
    {
        return (ngramSize - 1) * ngramHeadsPerOrder;
    }

    //! Tokens of history the hash reaches back over, and therefore the state carried across calls.
    int32_t ngramContextLength() const noexcept
    {
        return ngramSize - 1;
    }

    //! Steps of convolution state PLE carries per sequence.
    int32_t ngramConvStateLen() const noexcept
    {
        return (ngramConvKernel - 1) * ngramSize;
    }

    bool ngramEnabled() const noexcept
    {
        return !ngramLayers.empty();
    }

    //! Whether this engine binds the shared PLE row pool; add new pool users here, not at call sites.
    bool usesPlePool() const noexcept
    {
        return gemmaPleEnabled || ngramEnabled();
    }

    //! Additional EOS token IDs parsed from config.json `eos_token_id` array.
    //! Models like Gemma4 have multiple EOS tokens (e.g. [1, 106] for <eos> and <turn|>).
    //! Empty if `eos_token_id` is absent or scalar.
    std::vector<int32_t> eosTokenIds{};

    // --- Multimodal token IDs ---
    int32_t padTokenId{-1};   //!< PLE lookup token ID for multimodal positions (-1 = unused)
    int32_t imageTokenId{-1}; //!< Special token ID for image (-1 = unused)
    int32_t audioTokenId{-1}; //!< Special token ID for audio (-1 = unused)
    //! Gemma4 Unified prefill uses block-causal image attention.
    bool useVisionBidirectionalAttention{false};

    // --- Hybrid model (Mamba/GDN) state dimensions ---
    int32_t numLinearAttnLayers{0};     //!< Number of linear attention / recurrent layers
    int32_t recurrentStateNumHeads{0};  //!< Recurrent state heads (hv for GDN, mamba_num_heads for Mamba)
    int32_t recurrentStateHeadDim{0};   //!< Recurrent state head dimension
    int32_t recurrentStateSize{0};      //!< Recurrent state dimension (v for GDN, dstate for Mamba)
    int32_t recurrentStateNumGroups{0}; //!< Mamba B/C group count (spec-verify replay-B extent; 0 for GDN)
    //! Selects Mamba factorized replay storage, parsed from `recurrent_spec_verify_mode`
    //! ("replay" → true, "snapshot"/absent → false).
    bool recurrentSpecVerifyUsesReplay{false};
    int32_t convDim{0};    //!< Conv1d channel dimension
    int32_t convKernel{0}; //!< Conv1d kernel width

    // --- SpecDecode engine limits (per-engine) ---
    //! Max seq_len the base engine accepts for proposal verification. Parsed from
    //! `builder_config.max_spec_verify_tokens` when `isSpecDecodeBase == true`;
    //! 0 otherwise. Consumers prefer the consolidated `DeploymentConfig::specDecode`
    //! when the deployment view is available.
    int32_t maxSpecVerifyTokens{0};

    //! Max seq_len the draft engine accepts for proposal / draft generation.
    //! Parsed from `builder_config.max_spec_draft_tokens` by `parseDraftEngineConfig`;
    //! 0 on base / vanilla engines. Consumers prefer the consolidated
    //! `DeploymentConfig::specDecode` when the deployment view is available.
    int32_t maxSpecDraftTokens{0};

    //! Hidden dim the draft engine expects for its `hidden_states_input` binding
    //! (== the base engine's hidden-state output dim as seen by the draft).
    //! Parsed from top-level `base_model_hidden_size` by `parseDraftEngineConfig`;
    //! left at 0 on base / vanilla engines. Differs from `base.hiddenSize` for
    //! EAGLE-3 (`= base.hiddenSize * 3`, multi-layer concat) and equals
    //! `base.hiddenSize` for MTP. The deployment factory copies this into
    //! `DeploymentConfig::specDecode->baseOutputHiddenDim`.
    int32_t baseModelHiddenSize{0};

    //! Cached draft proposal block size for DFlash/JetSpec/DSpark. Parsed from the
    //! mode-specific config object (`dflash_config`, `jetspec_config`, or `dspark_config`).
    int32_t specDraftBlockSize{0};

    //! Mask token ID used to seed cached draft input blocks for DFlash/JetSpec/DSpark.
    int32_t specDraftMaskTokenId{0};

    //! Versioned DFlash engine/runtime ABI. Missing metadata defaults to V1.
    DFlashVersion dflashVersion{DFlashVersion::kV1};
    DFlashProjector dflashProjector{DFlashProjector::kSelector};
    //! LiLiCorr candidate count per proposal slot (`dflash_config.lilicorr_candidate_topk`); 0 for the selector.
    int32_t specLiLiCorrTopK{0};
    int32_t specSelectorTopK{0};
    int32_t specSelectorRank{0};
    std::string dflash2SelectorFile;
    int32_t specConvKernelSize{0};
    int32_t specConvGroupSize{0};
    bool specSupportsProbabilistic{false};

    //! Domino runtime correction head metadata (cached block draft + GRU rollout).
    int32_t dominoGruHiddenDim{0};
    int32_t dominoProjDim{0};
    int32_t dominoPrefixLen{0};
    int32_t dominoPoolSize{0};
    std::string dominoHeadsFile{};

    //! EAGLE draft candidate head (`eagle3_config.candidate_head`): candidates per row emitted as
    //! `spec_proposal_support_*` instead of `logits` (0 = dense logits head).
    int32_t draftCandidateCount{0};

    //! Whether cached draft proposal self-attention is causal. JetSpec uses causal rows.
    bool specDraftCausalHead{false};

    //! Target decoder-layer IDs whose hidden states are concatenated for cached drafts.
    //! EAGLE base engines own this contract through `eagle_hidden_state_layers`;
    //! DFlash/DSpark keep their mode-specific target-layer metadata on the engine
    //! that exports it.
    std::vector<int32_t> specTargetLayerIds{};

    // --- Gemma4 MTP shared-target-KV metadata ---
    std::string modelType;              //!< Top-level model type string, if exported.
    bool sharesTargetKV{false};         //!< Draft assistant reads target KV cache.
    bool hasOwnKVCache{true};           //!< Draft assistant owns/writes its own KV cache.
    bool constantDraftPositions{false}; //!< Assistant draft position IDs stay constant through a chain.
    bool returnsFeedbackHidden{false};  //!< Assistant emits backbone-space feedback hidden states.
    int32_t assistantHiddenSize{0};     //!< Assistant internal hidden dim, when distinct from hiddenSize.
    std::vector<Gemma4MTPKVSharingEntry> gemma4MTPKVSharingMap{}; //!< Assistant -> target KV sharing map.

    //! DSpark Markov/confidence sidecar metadata.
    bool dsparkEnableConfidenceHead{false};
    bool dsparkConfidenceHeadWithMarkov{false};
    std::string dsparkMarkovHeadType{};
    int32_t dsparkMarkovRank{0};
    std::string dsparkHeadsFile{};
    std::string dsparkHeadsInfoFile{};
    //! Whether the draft engine baked the contiguous-query sliding-window XQA
    //! variant, which reconstructs each query row's position as
    //! firstQueryPosition + queryRow. This describes the draft engine query ABI;
    //! it does not gate the base engine's DDTree verification topology.
    bool dsparkContiguousQuerySwa{false};
    //! When true the anchor slot itself is a proposal and the draft query block is
    //! `block_size` wide; when false slot 0 is the bonus token, the block carries one
    //! extra mask slot and proposals start at slot 1.
    bool dsparkSampleFromAnchor{true};

    // --- Per-layer type routing (hybrid cache) ---

    //! Absolute decoder-layer -> attention|mamba. Populated either from the
    //! canonical `layer_types` field in config.json or by broadcasting scalar
    //! `numAttentionLayers`/`numLinearAttnLayers` for back-compat. Size equals
    //! the number of stateful decoder layers (mlp/moe excluded on the Python
    //! side).
    std::vector<HybridCacheManager::LayerType> layerTypes{};

    //! Per-layer stateful components, dense over num_hidden_layers, absolutely indexed
    //! (`layers[i].absLayerIdx == i`). Restates the `layerTypes` / `kvLayerConfigs` / `kvSharingDonors`
    //! arrays and the global recurrent/n-gram geometry in one table, so a consumer can iterate a layer's
    //! states without knowing which mixer produced them. Empty when derivation cannot recover absolute
    //! indices (e.g. a compact-indexed `layer_types` that drops mlp/moe layers).
    std::vector<LayerSpec> layers{};

    //! Per-attention-layer KV config. Size equals the attention count in
    //! `layerTypes`. Indexed by LOCAL attention-layer index (0..numAttn-1),
    //! NOT absolute decoder-layer index.
    std::vector<KVLayerConfig> kvLayerConfigs{};

    //! Per-attention-layer KV sharing donor index. Size equals the attention count
    //! in `layerTypes`. Value of -1 means the layer owns its own KV cache (normal).
    //! A value >= 0 means this layer shares the KV cache from the donor attention
    //! layer at that LOCAL index (the donor's cache is bound to this layer's KV input).
    //! Used for Gemma4's KV sharing where the last N layers reuse a donor's cache.
    std::vector<int32_t> kvSharingDonors{};

    //! Whether exported layer metadata declares bounded SWA storage capability.
    bool supportsBoundedSwaKVCache() const;

    //! Whether this runtime instance actively uses bounded SWA storage.
    bool usesBoundedSwaKVCache() const;

    //! Select the runtime storage policy without changing exported capability metadata.
    void setSwaKVCacheMode(SwaKVCacheMode mode) noexcept;

    //! Shape-only mode input length: 1 for bounded storage and 0 for full storage.
    int32_t getSwaKVCacheModeInputLength() const;

    //! Bounded-capable profile page count for one layer, independent of the active runtime policy.
    int32_t getBoundedKVPoolPagesForLayer(KVLayerConfig const& layerConfig) const;

    //! Static engine profile selectors: min, smaller-policy opt, and max across bounded and full page counts.
    std::array<int32_t, 3> getKVPoolPageProfileForLayer(KVLayerConfig const& layerConfig) const;

    //! Active physical page count for one layer's KV pool binding.
    int32_t getKVPoolPagesForLayer(KVLayerConfig const& layerConfig) const;

    // ------------------------------------------------------------------
    // InferenceDims recipe methods
    //
    // Every `InferenceDims` the runtime hands to `EngineExecutor::prepare` should be
    // produced by one of these methods. Each returns a fully-specified
    // struct with all fields set — no implicit defaults.
    //
    // The second parameter (where present) is named for the caller-side
    // concept, not a destination field, because each input fans out to
    // 2–3 fields inside the method body. See the recipe bodies for the
    // fan-out.
    // ------------------------------------------------------------------

    //! skipSoftmaxPrefillScaleTable flattened to the INT32 [bin_0, S_0, bin_1, S_1, ...] binding payload.
    std::vector<int32_t> flattenedSkipSoftmaxPrefillScaleTable() const;

    //! skipSoftmaxDecodeScaleTable flattened the same way.
    std::vector<int32_t> flattenedSkipSoftmaxDecodeScaleTable() const;

    //! Element count of the full `skip_softmax_prefill_scale` binding (0 = dense prefill).
    int64_t skipSoftmaxPrefillScaleTableLen() const;

    //! Element count of the `skip_softmax_decode_scale` binding (0 = dense decode).
    int64_t skipSoftmaxDecodeScaleTableLen() const;

    //! skip_softmax_prefill_scale binding length for a prefill of `seqLen` new tokens: 0 (dense cubin, zero
    //! tax) when the whole batch is known to sit below the table's first skippable length, else the
    //! full table. Chunked prefill (kContextChunk) has KV history, so its total length is unknown
    //! here: the table is bound and the kernel's per-sequence guard decides.
    int64_t skipSoftmaxPrefillScaleLen(int64_t seqLen, ExecutionPhase phase) const;

    //! Prefill dims (vanilla LLM, SpecDecode base, and SpecDecode draft).
    //! seqLen is the prompt length being processed this step.
    //! kvCacheAllEmpty signals whether this is the initial prefill of an empty
    //! KV cache — this drives autoregressive engines' `kvcache_start_index`
    //! shape to `[0]` (engine's "initial prefill" sentinel) instead of `[batch]`.
    //! DiffusionGemma keeps `kvcache_start_index` at `[batch]` and uses
    //! `context_mask_selector` as its attention-mask sentinel.
    InferenceDims prefillDims(int64_t batch, int64_t seqLen, ExecutionPhase phase) const;

    //! Vanilla single-token decode dims.
    //! seqLen is always 1 here; packedMaskLen is 1 (no proposal mask in vanilla).
    InferenceDims decodeDims(int64_t batch) const;

    //! Compact phase-8 dims: numSequences rows packed into numTokens tokens, context sequences first.
    //! numContextSequences is the carrier extent Np (0 only for a decode-only subset, where numTokens must equal
    //! numSequences); numLogits is the logits row count L in [1, numSequences], placeholder rows included (the
    //! compact builder emits L = numSequences). Execute on profile compactStepProfileIndex(Np) (engineExecutor.h).
    InferenceDims compactDims(
        int64_t numSequences, int64_t numTokens, int64_t numContextSequences, int64_t numLogits) const;

    //! DiffusionGemma denoise dims. Uses existing chunked-prefill execution:
    //! seqLen/selectLen are the canvas length, kvcache_start_index is [batch],
    //! and context_mask_selector is [batch] to select non-causal PADDING mask.
    InferenceDims denoiseDims(int64_t batch, int64_t canvasLen) const;

    //! DiffusionGemma commit dims. Reruns finalized tokens causally before KV
    //! commit; context_mask_selector is [0], so AttentionPlugin keeps causal mask.
    InferenceDims diffusionCommitDims(int64_t batch, int64_t commitLen) const;

    //! SpecDecode base verification dims.
    //! verifyTokenCount feeds three fields: seqLen, selectLen, and packedMaskLen.
    //! This is the only recipe where selectLen != 1.
    InferenceDims specVerifyDims(int64_t batch, int64_t verifyTokenCount, SpecVerifyLayout verifyLayout) const;

    //! SpecDecode draft proposal dims.
    //! proposalSize feeds seqLen, attnMaskSeqLen, and packedMaskLen.
    //! draftTopK feeds selectLen — the draft proposal selects draftTopK tokens
    //! per sequence, matching the 3D logits output shape
    //! [batch, draftTopK, draftVocabSize].
    InferenceDims proposalDims(int64_t batch, int64_t proposalSize, int64_t draftTopK) const;

    //! SpecDecode draft accept-token dims.
    //! acceptLen is in [1, maxPathLength]; feeds seqLen and packedMaskLen.
    InferenceDims acceptDims(int64_t batch, int64_t acceptLen) const;

    //! Clean-slate dims for CUDA-graph capture reset.
    //! All 1s except kvLen = maxKVCacheCapacity. ropeBatch is fixed at 1
    //! even for MRope models — this matches the pre-migration behavior in
    //! both runtimes (reset is a binding placeholder, not an inference step).
    InferenceDims resetDims() const;
};

//! Parse a `config.json` file (the same format used by the existing runtime)
//! into an `LLMEngineConfig`.
//!
//! @param configPath  Path to `config.json`.
//! @return Parsed configuration.
//! @throws std::runtime_error if file cannot be opened/parsed or required fields are missing.
LLMEngineConfig parseEngineConfig(std::filesystem::path const& configPath, std::optional<int32_t> rank = std::nullopt,
    std::optional<int32_t> expectedWorldSize = std::nullopt);

//! Parse a SpecDecode draft engine's `config.json` into an `LLMEngineConfig`.
//!
//! The draft config carries a reduced field set (no `builder_config.spec_base`,
//! its own `draft_vocab_size`). `max_spec_draft_tokens` is required and is
//! parsed into `cfg.maxSpecDraftTokens`; `cfg.maxSpecVerifyTokens` stays at 0 on
//! the draft side. `isSpecDecodeBase` is left false because this is the
//! draft — not the base — engine. Cross-engine fields (`draftHiddenSize`,
//! `baseOutputHiddenDim`) are not stored on `LLMEngineConfig`; they are
//! derived in `createDeploymentConfig` and live on `DeploymentConfig::specConfig`.
//!
//! @param configPath  Path to the draft engine's `config.json`.
//! @return Parsed configuration.
//! @throws std::runtime_error on parse failure or missing required fields.
LLMEngineConfig parseDraftEngineConfig(std::filesystem::path const& configPath);

//! Format the config as a human-readable string (for logging).
std::string formatEngineConfig(LLMEngineConfig const& config);

bool isCachedBlockDraftBase(LLMEngineConfig const& config) noexcept;
bool isCachedBlockDraftDraft(LLMEngineConfig const& config) noexcept;

//! Cross-check an engine's KV / recurrent / conv binding dtypes against their
//! parsed-config counterparts. The parsed config is the source of truth; this
//! raises a runtime_error with an actionable message on any mismatch.
//!
//! @param config  Parsed `LLMEngineConfig` (from `parseEngineConfig` or similar).
//! @param executor Engine whose binding dtypes should match.
//! @param engineLabel Short label ("base" / "draft") used in error messages.
//! @throws std::runtime_error on any KV / recurrent / conv dtype mismatch.
void validateAgainstEngine(LLMEngineConfig const& config, EngineExecutor const& executor, char const* engineLabel);

} // namespace rt
} // namespace trt_edgellm
