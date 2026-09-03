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
#include "runtime/hybridCacheManager.h"
#include "runtime/kvCacheManager.h"
#include "runtime/llmRuntimeUtils.h"

#include <NvInferRuntime.h>
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
};

char const* specDecodeModeName(SpecDecodeMode mode) noexcept;
bool isCachedBlockDraftMode(SpecDecodeMode mode) noexcept;

//! Gemma4 MTP assistant-layer to target-layer shared-KV mapping.
struct Gemma4MTPKVSharingEntry
{
    int32_t assistantLayerIdx{-1};       //!< Assistant local attention-layer index.
    int32_t targetAttentionLayerIdx{-1}; //!< Target local attention-layer index.
    int32_t targetAbsoluteLayerIdx{-1};  //!< Target absolute decoder-layer index, derived during validation.
};

//! Unified configuration for base, vanilla decode, and SpecDecode draft engines.
//!
//! Replaces `LLMEngineRunnerConfig` and `EagleDraftEngineRunnerConfig` with a
//! single structure that can be parsed once and used across all runtime
//! components (KV cache, RoPE, LoRA, preprocessors, etc.).
struct LLMEngineConfig
{
    // --- Core model dimensions ---
    int32_t hiddenSize{};                //!< Model hidden dimension
    int32_t outputVocabSize{};           //!< Actual output vocab (reduced if vocab reduction active)
    int32_t numAttentionLayers{};        //!< Number of attention layers needing KV cache
    int32_t numKVHeads{};                //!< Number of key-value heads
    int32_t headDim{};                   //!< Dimension of each attention head
    int32_t maxSupportedBatchSize{};     //!< Maximum supported batch size
    int32_t maxSupportedInputLength{};   //!< Maximum supported input length
    int32_t maxKVCacheCapacity{};        //!< Maximum KV cache capacity (sequence length)
    int64_t skipSoftmaxScaleOverride{0}; //!< skip-softmax scale-factor override (0 = disabled)
    int32_t kvPoolPages{};               //!< Exact physical K-page count serialized in KV binding shapes
    int32_t rotaryDim{};                 //!< Rotary embedding dimension
    int32_t numDecoderLayers{};          //!< Total decoder layers (attention + linear)
    int32_t vocabSize{};                 //!< Full vocabulary size
    int32_t reducedVocabSize{0};         //!< 0 = no vocab reduction
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
    nvinfer1::DataType kvCacheDtype{nvinfer1::DataType::kHALF};

    //! Recurrent state data type (hybrid models only). Parsed from required
    //! top-level `recurrent_state_dtype` when `numLinearAttnLayers > 0`;
    //! left at the default otherwise. Runtime validates against the engine's
    //! recurrent-state binding dtype.
    nvinfer1::DataType recurrentStateDtype{nvinfer1::DataType::kHALF};
    //! Conv state data type (hybrid models only). Same shape as `recurrentStateDtype`.
    nvinfer1::DataType convStateDtype{nvinfer1::DataType::kHALF};

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
    bool pleEnabled{false};          //!< Gemma4 PLE runtime preprocessor enabled
    int32_t numPleInputs{0};         //!< Number of PLE tensors bound into the engine (0 = disabled)
    int32_t pleHiddenSize{0};        //!< Hidden dimension of each PLE tensor (0 = disabled)
    int32_t maxSupportedLoraRank{0}; //!< Maximum LoRA rank (0 = no LoRA)

    //! Additional EOS token IDs parsed from config.json `eos_token_id` array.
    //! Models like Gemma4 have multiple EOS tokens (e.g. [1, 106] for <eos> and <turn|>).
    //! Empty if `eos_token_id` is absent or scalar.
    std::vector<int32_t> eosTokenIds{};

    // --- Multimodal token IDs ---
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
    //! MTP spec-verify recurrent-state commit mode, parsed from `recurrent_spec_verify_mode`
    //! ("replay" → true, "snapshot"/absent → false).
    bool recurrentSpecVerifyUsesReplay{false};
    int32_t convDim{0};    //!< Conv1d channel dimension
    int32_t convKernel{0}; //!< Conv1d kernel width

    // --- SpecDecode engine limits (per-engine) ---
    //! Max seq_len the base engine accepts for proposal verification. Parsed from
    //! `builder_config.max_verify_tree_size` when `isSpecDecodeBase == true`;
    //! 0 otherwise. Consumers prefer the consolidated `DeploymentConfig::specDecode`
    //! when the deployment view is available.
    int32_t maxVerifyTreeSize{0};

    //! Max seq_len the draft engine accepts for proposal / draft generation.
    //! Parsed from `builder_config.max_draft_tree_size` by `parseDraftEngineConfig`;
    //! 0 on base / vanilla engines. Consumers prefer the consolidated
    //! `DeploymentConfig::specDecode` when the deployment view is available.
    int32_t maxDraftTreeSize{0};

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

    // --- Per-layer type routing (hybrid cache) ---

    //! Absolute decoder-layer -> attention|mamba. Populated either from the
    //! canonical `layer_types` field in config.json or by broadcasting scalar
    //! `numAttentionLayers`/`numLinearAttnLayers` for back-compat. Size equals
    //! the number of stateful decoder layers (mlp/moe excluded on the Python
    //! side).
    std::vector<HybridCacheManager::LayerType> layerTypes{};

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

    //! Prefill dims (vanilla LLM, SpecDecode base, and SpecDecode draft).
    //! seqLen is the prompt length being processed this step.
    //! kvCacheAllEmpty signals whether this is the initial prefill of an empty
    //! KV cache — this drives autoregressive engines' `kvcache_start_index`
    //! shape to `[0]` (engine's "initial prefill" sentinel) instead of `[batch]`.
    //! DiffusionGemma keeps `kvcache_start_index` at `[batch]` and uses
    //! `context_mask_selector` as its attention-mask sentinel.
    InferenceDims prefillDims(int64_t batch, int64_t seqLen, bool kvCacheAllEmpty) const;

    //! Vanilla single-token decode dims.
    //! seqLen is always 1 here; packedMaskLen is 1 (no proposal mask in vanilla).
    InferenceDims decodeDims(int64_t batch) const;

    //! DiffusionGemma denoise dims. Uses existing chunked-prefill execution:
    //! seqLen/selectLen are the canvas length, kvcache_start_index is [batch],
    //! and context_mask_selector is [batch] to select non-causal PADDING mask.
    InferenceDims denoiseDims(int64_t batch, int64_t canvasLen) const;

    //! DiffusionGemma commit dims. Reruns finalized tokens causally before KV
    //! commit; context_mask_selector is [0], so AttentionPlugin keeps causal mask.
    InferenceDims diffusionCommitDims(int64_t batch, int64_t commitLen) const;

    //! SpecDecode base verification dims.
    //! verifySize feeds three fields: seqLen, selectLen, and packedMaskLen.
    //! This is the only recipe where selectLen != 1.
    InferenceDims specVerifyDims(int64_t batch, int64_t verifySize) const;

    //! SpecDecode draft proposal dims.
    //! proposalSize feeds seqLen, attnMaskSeqLen, and packedMaskLen.
    //! draftTopK feeds selectLen — the draft proposal selects draftTopK tokens
    //! per sequence, matching the 3D logits output shape
    //! [batch, draftTopK, draftVocabSize].
    InferenceDims proposalDims(int64_t batch, int64_t proposalSize, int64_t draftTopK) const;

    //! SpecDecode draft accept-token dims.
    //! acceptLen is in [1, draftingStep+1]; feeds seqLen and packedMaskLen.
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
//! its own `draft_vocab_size`). `max_draft_tree_size` is required and is
//! parsed into `cfg.maxDraftTreeSize`; `cfg.maxVerifyTreeSize` stays at 0 on
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
