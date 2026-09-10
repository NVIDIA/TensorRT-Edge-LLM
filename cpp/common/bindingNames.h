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

#include <string>

namespace trt_edgellm
{
/*!
 * @namespace binding_names
 * @brief Unified tensor binding names for TensorRT engines
 *
 * This namespace provides a centralized location for all tensor binding names
 * used across both the builder and runtime components to ensure consistency
 * and avoid duplication.
 */
namespace binding_names
{

/*! @name Core LLM Input/Output Bindings
 * @{
 */

/*!
 * @brief Input embeddings tensor - contains the embedded input sequence
 *
 * Shape: [batch_size, sequence_length, hidden_size] (FLOAT16)
 */
inline constexpr char const* kInputsEmbeds = "inputs_embeds";

/*!
 * @brief Gemma4 per-layer token-identity embedding input template.
 *
 * Template: "ple_token_embeds_{layer_idx}"
 * Shape: [batch_size, sequence_length, ple_hidden_size] (FLOAT16/BFLOAT16)
 */
inline constexpr char const* kPleTokenEmbedsTemplate = "ple_token_embeds";

/*!
 * @brief Context lengths tensor - specifies the actual length of each sequence in the batch
 *
 * Shape: [batch_size] (INT32)
 */
inline constexpr char const* kContextLengths = "context_lengths";

/*!
 * @brief Last token IDs tensor - indices of the last tokens to extract from hidden states
 *
 * Shape: [batch_size] for Eagle models, [batch_size, 1] for vanilla models (INT64)
 */
inline constexpr char const* kLastTokenIds = "last_token_ids";

/*!
 * @brief DiffusionGemma phase selector. Non-zero selects encoder-phase layer scalars.
 */
inline constexpr char const* kPhaseIsEncoder = "phase_is_encoder";

/*!
 * @brief DiffusionGemma hidden-state gather indices for logits output.
 */
inline constexpr char const* kSelectTokenIndices = "select_token_indices";

/*!
 * @brief DiffusionGemma context-mask shape selector: [0] causal, [B] padding/non-causal.
 */
inline constexpr char const* kContextMaskSelector = "context_mask_selector";

/*!
 * @brief DiffusionGemma fused conditioning engine canvas token IDs.
 */
inline constexpr char const* kCanvasIds = "canvas_ids";

/*!
 * @brief DiffusionGemma fused conditioning engine previous hidden feedback embeddings.
 */
inline constexpr char const* kPrevSelfConditioningEmbeds = "prev_self_conditioning_embeds";

/*!
 * @brief DiffusionGemma fused conditioning engine next hidden feedback embeddings.
 */
inline constexpr char const* kNextSelfConditioningEmbeds = "next_self_conditioning_embeds";

/*!
 * @brief DiffusionGemma fused conditioning engine feedback softmax temperature.
 */
inline constexpr char const* kSelfConditioningTemperature = "self_conditioning_temperature";

/*!
 * @brief Output logits tensor - probability distribution over vocabulary
 *
 * Shape: [batch_size, vocab_size] or [select_tokens, vocab_size] (FLOAT32)
 */
inline constexpr char const* kLogits = "logits";

/*!
 * @brief Output hidden states tensor - intermediate representations for speculative decoding
 *
 * Shape: [batch_size, sequence_length, hidden_dim] (FLOAT16)
 */
inline constexpr char const* kOutputHiddenStates = "hidden_states";

/*!
 * @brief Thinker accept-layer hidden states, the Qwen3-Omni-Next Talker's feed.
 *
 * Shape: [batch_size, sequence_length, thinker_hidden_size] (FLOAT16)
 * Pre-norm output of decoder layer `accept_hidden_layer - 1`, i.e. HF's
 * `hidden_states[k]`. Emitted only by Omni-Next Thinker engines built as a
 * speculative-decoding base, where `hidden_states` is contractually the draft's
 * post-norm feed and so cannot also carry this.
 */
inline constexpr char const* kAcceptHiddenStates = "accept_hidden_states";

/*!
 * @brief Cached speculative draft model input: concatenated target hidden states.
 *
 * Shape: [batch_size, context_length, base_output_hidden_dim] (FLOAT16)
 */
inline constexpr char const* kDFlashTargetHiddenConcat = "dflash_target_hidden_concat";

/*!
 * @brief Cached speculative draft model input: per-batch delta lengths for multi-batch.
 *
 * Shape: [batch_size] (INT32)
 */
inline constexpr char const* kDFlashDeltaLengths = "dflash_delta_lengths";

/*!
 * @brief DSpark draft model output: final proposal hidden states.
 *
 * Shape: [batch_size, proposal_length, hidden_size] (FLOAT16)
 */
inline constexpr char const* kDSparkHiddenStates = "dspark_hidden_states";

/*!
 * @brief DSpark Markov/confidence head sidecar file names.
 */
inline constexpr char const* kDSparkHeadsFileName = "dspark_heads.safetensors";
inline constexpr char const* kDSparkHeadsInfoFileName = "dspark_heads_info.json";

/*!
 * @brief Gemma4 PLE token-identity embedding table sidecar.
 *
 * Contains tensor "weight" with shape [vocab_size_per_layer_input,
 * num_ple_inputs * ple_hidden_size].
 */
inline constexpr char const* kPleEmbeddingFileName = "ple_embedding.safetensors";

/*! @} */

/*! @name Positional Encoding Bindings
 * @{
 */

/*!
 * @brief Rotary positional encoding cos/sin cache tensor
 *
 * Shape: [batch_size, max_seq_len, rotary_dim] (FLOAT32)
 */
inline constexpr char const* kRopeCosSin = "rope_rotary_cos_sin";

/*!
 * @brief Rotary positional encoding cos/sin cache tensor for sliding attention layers
 *
 * Shape: [batch_size, max_seq_len, sliding_rotary_dim] (FLOAT32)
 */
inline constexpr char const* kRopeCosSinSliding = "rope_rotary_cos_sin_sliding";

/*!
 * @brief Rotary positional encoding cos/sin cache tensor for full attention layers
 *
 * Shape: [batch_size, max_seq_len, full_rotary_dim] (FLOAT32)
 */
inline constexpr char const* kRopeCosSinFull = "rope_rotary_cos_sin_full";

/*! @} */

/*! @name KV Cache Bindings
 * @{
 */

/*!
 * @brief KV cache start index tensor - starting position for KV cache reuse
 *
 * Shape: [batch_size] (INT32)
 */
inline constexpr char const* kKVCacheStartIndex = "kvcache_start_index";

/*!
 * @brief KV page-table tensor - per-request page indices into the paged two-pool KV cache
 *
 * K page ids in row 0, derived V page ids (K + numPages) in row 1.
 *
 * Shape: [batch_size, 2, max_pages_per_seq] (INT32)
 */
inline constexpr char const* kKVPageTable = "kv_page_table";

/*!
 * @brief Past key-value cache tensor template - use with layer index formatting
 *
 * Template: "past_key_values_{layer_idx}"
 * Shape: [2, num_pages, kTOKENS_PER_PAGE, num_kv_heads, head_dim] paged pool (FLOAT16/FP8)
 */
inline constexpr char const* kPastKeyValuesTemplate = "past_key_values";

/*!
 * @brief Present key-value cache tensor template - use with layer index formatting
 *
 * Template: "present_key_values_{layer_idx}"
 * Shape: [2, num_pages, kTOKENS_PER_PAGE, num_kv_heads, head_dim] paged pool, aliases past (FLOAT16/FP8)
 */
inline constexpr char const* kPresentKeyValuesTemplate = "present_key_values";

/*!
 * @brief K cache tensor template for TensorRT native KVCacehUpdate operations - use with layer index formatting
 *
 * Template: "k_cache_{layer_idx}"
 * Shape: [batch_size, num_kv_heads, seq_len, head_dim] (FLOAT16)
 */
inline constexpr char const* kKCacheTemplate = "k_cache";

/*!
 * @brief V cache tensor template for TensorRT native KVCacheUpdate operations - use with layer index formatting
 *
 * Template: "v_cache_{layer_idx}"
 * Shape: [batch_size, num_kv_heads, seq_len, head_dim] (FLOAT16)
 */
inline constexpr char const* kVCacheTemplate = "v_cache";

/*!
 * @brief Present K cache tensor template for TensorRT native KVCacheUpdate operations - use with layer index formatting
 *
 * Template: "present_k_cache_{layer_idx}"
 * Shape: [batch_size, num_kv_heads, seq_len, head_dim] (FLOAT16)
 */
inline constexpr char const* kPresentKCacheTemplate = "present_k_cache";

/*!
 * @brief Present V cache tensor template for TensorRT native KVCacheUpdate operations - use with layer index formatting
 *
 * Template: "present_v_cache_{layer_idx}"
 * Shape: [batch_size, num_kv_heads, seq_len, head_dim] (FLOAT16)
 */
inline constexpr char const* kPresentVCacheTemplate = "present_v_cache";

/*! @} */

/*! @name Alpamayo 1 action head input/output bindings
 * @{
 */

/*!
 * @brief Noise trajectory input for one denoising step
 *
 * Shape: [batch_size, num_waypoints, 2] (FLOAT32)
 */
inline constexpr char const* kNoiseTrajectory = "noise_trajectory";

/*!
 * @brief Denoising time step t0 (start)
 *
 * Shape: [1] (FLOAT32)
 */
inline constexpr char const* kTimeStepsT0 = "time_steps_t0";

/*!
 * @brief Denoising time step t1 (end)
 *
 * Shape: [1] (FLOAT32)
 */
inline constexpr char const* kTimeStepsT1 = "time_steps_t1";

/*!
 * @brief Denoised trajectory output from one denoising step
 *
 * Shape: [batch_size, num_waypoints, 2] (FLOAT32)
 */
inline constexpr char const* kDenoisedTrajectory = "denoised_trajectory";

/*! @} */

/*! @name Recurrent State Bindings (Mamba / GDN / linear-attention layers)
 * @{
 */

/*!
 * @brief Past recurrent state tensor template
 *
 * Template: "recurrent_state_{recurrent_layer_idx}"
 * Shape: [batch_size, recurrentNumHeads, recurrentHeadDim, recurrentStateSize]
 */
inline constexpr char const* kRecurrentStateTemplate = "recurrent_state";

/*!
 * @brief Present recurrent state tensor template
 *
 * Template: "present_recurrent_state_{recurrent_layer_idx}"
 * Shape: [batch_size, recurrentNumHeads, recurrentHeadDim, recurrentStateSize]
 */
inline constexpr char const* kPresentRecurrentStateTemplate = "present_recurrent_state";

/*!
 * @brief Past conv state tensor template for recurrent layers
 *
 * Template: "conv_state_{recurrent_layer_idx}"
 * Shape: [batch_size, conv_dim, conv_kernel_size] (FLOAT16)
 */
inline constexpr char const* kConvStateTemplate = "conv_state";

/*!
 * @brief Present conv state tensor template for recurrent layers
 *
 * Template: "present_conv_state_{recurrent_layer_idx}"
 * Shape: [batch_size, conv_dim, conv_kernel_size] (FLOAT16)
 */
inline constexpr char const* kPresentConvStateTemplate = "present_conv_state";

/*!
 * @brief Intermediate conv state output template for MTP speculative decoding
 *
 * Template: "intermediate_conv_state_{recurrent_layer_idx}"
 * Shape: [batch_size, seq_len, conv_dim, conv_kernel_size] (FLOAT16)
 */
inline constexpr char const* kIntermediateConvStateTemplate = "intermediate_conv_state";

/*!
 * @brief Intermediate recurrent state output template for MTP speculative decoding
 *
 * Template: "intermediate_recurrent_state_{recurrent_layer_idx}"
 * Shape: [batch_size, seq_len, recurrentNumHeads, recurrentHeadDim, recurrentStateSize] (FLOAT32)
 */
inline constexpr char const* kIntermediateRecurrentStateTemplate = "intermediate_recurrent_state";

/*!
 * @brief Mamba spec-verify replay-stash output templates (FP32). Instead of a per-token full-state
 * snapshot, the Mamba plugin stashes the minimal per-token replay inputs; the runtime reconstructs
 * the accepted recurrent state from them after verification.
 *   dA: [batch, seq_len, recurrentNumHeads]
 *   u:  [batch, seq_len, recurrentNumHeads, recurrentHeadDim]
 *   B:  [batch, seq_len, recurrentNumGroups, recurrentStateSize]
 */
inline constexpr char const* kReplayDaStateTemplate = "replay_da_state";
inline constexpr char const* kReplayUStateTemplate = "replay_u_state";
inline constexpr char const* kReplayBStateTemplate = "replay_b_state";

/*! @} */

/*! @name Eagle Speculative Decoding Bindings
 * @{
 */

/*!
 * @brief Base model hidden states input for Eagle draft models
 *
 * Shape: [batch_size, sequence_length, base_hidden_dim] (FLOAT16)
 */
inline constexpr char const* kBaseModelHiddenStates = "hidden_states_input";

/*!
 * @brief Draft model hidden states input for Eagle draft models
 *
 * Shape: [batch_size, sequence_length, draft_hidden_dim] (FLOAT16)
 */
inline constexpr char const* kDraftModelHiddenStates = "hidden_states_from_draft";

/*!
 * @brief Attention mask for Eagle models - packed tree attention mask
 *
 * Shape: [batch_size, tree_size, packed_mask_len] (INT32 for base, INT8 for draft)
 */
inline constexpr char const* kAttentionMask = "attention_mask";

/*! Gemma4 Unified prefill block IDs: [batch_size, sequence_length] INT32. */
inline constexpr char const* kVisionBlockIds = "vision_block_ids";

/*!
 * @brief Attention position IDs for Eagle models
 *
 * Shape: [batch_size, tree_size] (INT32)
 */
inline constexpr char const* kAttentionPosId = "attention_pos_id";

/*!
 * @brief Shape-only marker for speculative verification in hybrid MTP/DFlash base engines
 *
 * Shape: [0] for normal prefill/decode/reset, [1] for MTP/DFlash verification (INT32).
 * The INT32 payload is ignored. Hybrid GDN/conv plugins only need to distinguish
 * speculative verify seq_len > 1 from ordinary prefill seq_len > 1.
 */
inline constexpr char const* kSpecVerifyPhaseMarker = "spec_verify_phase_marker";

/*!
 * @brief Runtime skip-softmax override carrier (optional AttentionPlugin input).
 *
 * 1-D INT8 dummy whose SHAPE (dims[0] = integer scale factor S) conveys the
 * override; 0 keeps the engine-carried calibrated default.
 */
inline constexpr char const* kSkipSoftmaxScale = "skip_softmax_scale";

/*!
 * @brief DDTree parent node ids for hybrid DFlash base verification
 *
 * Shape: [batch_size, verify_tree_size] (INT32). Each entry points to the
 * flattened parent node whose hybrid state is used to evaluate the current node.
 */
inline constexpr char const* kTreeParentIds = "tree_parent_ids";

/*!
 * @brief DDTree node depths for hybrid DFlash base verification
 *
 * Shape: [batch_size, verify_tree_size] (INT32). Depth is used for positional
 * metadata and for tree-state kernels that need node order information.
 */
inline constexpr char const* kTreeDepths = "tree_depths";

/*! @} */

/*! @name Visual Encoder Bindings (Qwen-VL, InternVL)
 * @{
 */

/*!
 * @brief Visual input tensor for vision transformers
 *
 * Shape: [sequence_length, input_dim] for Qwen-VL, [num_blocks, channels, height, width] for InternVL
 */
inline constexpr char const* kVisualInput = "input";

/*!
 * @brief Pixel-shuffle gather indices for the Nemotron-Omni visual engine
 *
 * Shape: [num_out_tokens, scale^2] (INT64). Grid-dependent, computed by the
 * runtime so one engine serves square image tiles and non-square video grids.
 */
inline constexpr char const* kVisualShuffleIndices = "shuffle_indices";

/*!
 * @brief Visual output tensor from vision transformers
 *
 * Shape: [num_image_tokens, hidden_size] (FLOAT16)
 */
inline constexpr char const* kVisualOutput = "output";

/*!
 * @brief Rotary positional embeddings for visual inputs
 *
 * Shape: [sequence_length, embed_dim] (FLOAT32)
 */
inline constexpr char const* kRotaryPosEmb = "rotary_pos_emb";

/*!
 * @brief Gemma4 pixel position ids for visual patch embeddings and 2-D RoPE
 *
 * Shape: [sequence_length, 2] (INT64)
 */
inline constexpr char const* kPixelPositionIds = "pixel_position_ids";

/*!
 * @brief Cumulative sequence lengths for ragged ViT attention
 *
 * Shape: [num_images + 1] (INT32)
 */
inline constexpr char const* kCuSeqlens = "cu_seqlens";

/*!
 * @brief KV sequence lengths for TRT-native attention (TRT >= 11).
 *
 * Same data as cu_seqlens but must be a separate tensor — TRT IAttentionV2
 * requires distinct tensors for query_lengths and kv_lengths inputs.
 *
 * Shape: [num_images + 1] (INT32)
 */
inline constexpr char const* kKvLengths = "kv_lengths";

/*!
 * @brief Window KV sequence lengths for Qwen2.5-VL TRT-native attention.
 *
 * Same data as cu_window_seqlens but must be a separate tensor for
 * TRT_Attention query_lengths/kv_lengths inputs in window-attention blocks.
 *
 * Shape: [num_windows + 1] (INT32)
 */
inline constexpr char const* kKvLengthsWindow = "kv_lengths_window";

/*!
 * @brief Shape-only input used to convey runtime max sequence-length for FMHA launch
 *
 * Shape: [max_seqlen] (INT32)
 */
inline constexpr char const* kMaxSeqLenCarrier = "max_seqlen_carrier";

/*!
 * @brief Gemma4 position-aware pooling weights
 *
 * Shape: [num_image_tokens, sequence_length] (FLOAT16)
 */
inline constexpr char const* kPoolingWeights = "pooling_weights";

/*!
 * @brief Cumulative window sequence lengths for Qwen2.5-VL window attention
 *
 * Shape: [num_windows + 1] (INT32)
 */
inline constexpr char const* kCuWindowSeqlens = "cu_window_seqlens";

/*!
 * @brief Window index for Qwen2.5-VL sliding window attention
 *
 * Shape: [num_windows] (INT64)
 */
inline constexpr char const* kWindowIndex = "window_index";

/*!
 * @brief Reverse window index for Qwen2.5-VL sliding window attention
 *
 * Shape: [num_windows] (INT64)
 */
inline constexpr char const* kReverseWindowIndex = "reverse_window_index";

/*!
 * @brief Fast position embeddings index tensor for Qwen3-VL vision model
 *
 * Shape: [4, sequence_length] (INT64)
 */
inline constexpr char const* kFastPosEmbIdx = "fast_pos_embed_idx";

/*!
 * @brief Fast position embeddings weight tensor for Qwen3-VL vision model
 *
 * Shape: [4, sequence_length] (FLOAT16)
 */
inline constexpr char const* kFastPosEmbWeight = "fast_pos_embed_weight";

/*!
 * @brief Deepstack features tensor for Qwen3-VL vision model (visual encoder output)
 *
 * Shape: [num_image_tokens, hidden_size] (FLOAT16)
 */
inline constexpr char const* kDeepstackFeaturesTemplate = "deepstack_features";

/*!
 * @brief Deepstack embeddings tensor template for Qwen3-VL text model (LLM input)
 *
 * Template: "deepstack_embeds_{layer_idx}" where layer_idx is 0, 1, or 2
 * Shape: [batch_size, sequence_length, hidden_size] (FLOAT16)
 */
inline constexpr char const* kDeepstackEmbedsTemplate = "deepstack_embeds";

/*! @} */

/*! @name Vocabulary Mapping Configuration
 * @{
 */

/*!
 * @brief JSON configuration key for reduced vocabulary size
 *
 * Used to check if the model uses vocabulary reduction optimization
 */
inline constexpr char const* kReducedVocabSizeKey = "reduced_vocab_size";

/*!
 * @brief Vocabulary mapping file name
 *
 * SafeTensors file containing mapping between full and reduced vocabulary
 */
inline constexpr char const* kVocabMapFileName = "vocab_map.safetensors";

/*!
 * @brief Draft model vocabulary mapping file name
 *
 * SafeTensors file containing mapping from reduced draft vocabulary to full vocabulary.
 *
 * @note Currently consumed only by DFlashDecoder (gated on the draft engine config's
 *       reduced_vocab_size > 0) and produced only by the DFlash draft export path. If a
 *       future decoder adopts draft vocab reduction, it must (a) gate the load on the same
 *       config field and (b) be added as a consumer of this constant.
 */
inline constexpr char const* kDraftVocabMapFileName = "draft_vocab_map.safetensors";

/*! @} */

/*! @name Audio Encoder Bindings (Qwen3-Omni)
 * @{
 */

/*!
 * @brief Raw framed PCM input used by encoder-free audio embedders
 *
 * Shape: [1, num_frames, frame_size] (FLOAT16)
 */
inline constexpr char const* kAudioInputFeatures = "input_features";

/*!
 * @brief Audio padded features tensor - chunked and padded Mel-spectrogram
 *
 * Shape: [num_chunks, mel_bins, max_chunk_len] (FLOAT16)
 */
inline constexpr char const* kAudioPaddedFeatures = "padded_feature";

/*!
 * @brief Audio padded mask indices - nonzero indices from mask
 *
 * Shape: [num_valid_elements, 2] (INT64)
 * Each row is [chunk_idx, position_idx] indicating valid positions after CNN downsampling
 */
inline constexpr char const* kAudioPaddedMaskIndices = "padded_mask_after_cnn_indices";

/*!
 * @brief Audio attention mask - block-diagonal mask for chunk-wise attention
 *
 * Shape: [num_attention_elems, num_attention_elems] (FLOAT16)
 * Block-diagonal matrix where each block corresponds to one audio chunk
 */
inline constexpr char const* kAudioAttentionMask = "attention_mask";

/*!
 * @brief Audio encoder output - audio embeddings
 *
 * Shape: [num_audio_tokens, hidden_size] (FLOAT16)
 */
inline constexpr char const* kAudioOutput = "last_hidden_state";

/*! @} */

/*! @name CodePredictor Bindings (Qwen3-Omni)
 * @{
 */

/*!
 * @brief LM head weight tensor - dynamically bound weight for CodePredictor
 *
 * Shape: [vocab_size, hidden_size] (FLOAT16)
 * This is used for dynamic lm_head selection in CodePredictor (15 different heads for RVQ layers)
 */
inline constexpr char const* kLmHeadWeight = "lm_head_weight";

/*!
 * @brief Stacked CodePredictor lm_heads + device-selected head index
 *
 * lm_heads: [num_heads, vocab_size, hidden_size] (FLOAT16), lm_head_idx: [1] (INT32).
 * The head is gathered inside the engine, so bindings stay step-invariant and one
 * CUDA graph serves every decode step.
 */
inline constexpr char const* kLmHeads = "lm_heads";
inline constexpr char const* kLmHeadIdx = "lm_head_idx";

/*! @} */

/*! @name Code2Wav Vocoder Bindings (Qwen3-Omni)
 * @{
 */

/*!
 * @brief Code2Wav input codes tensor - RVQ codec codes for vocoder
 *
 * Shape: [batch_size, num_quantizers, sequence_length] (INT32)
 * num_quantizers: 15 for Qwen3-Omni
 */
inline constexpr char const* kCode2WavCodes = "codes";

/*!
 * @brief Code2Wav output waveform tensor - generated audio waveform
 *
 * Shape: [batch_size, 1, waveform_length] (FLOAT32)
 * Values in range [-1.0, 1.0]
 */
inline constexpr char const* kCode2WavWaveform = "waveform";

/*! @} */

/*! @name LoRA (Low-Rank Adaptation) Bindings
 * @{
 */

/*!
 * @brief LoRA A weight matrix prefix - use with layer/component specific suffixes
 *
 * Template: "lora_A_{component}_{layer}"
 * Shape: [gemm_k, lora_rank] (FLOAT16)
 */
inline constexpr char const* kLoraAPrefix = "lora_A";

/*!
 * @brief LoRA B weight matrix prefix - use with layer/component specific suffixes
 *
 * Template: "lora_B_{component}_{layer}"
 * Shape: [lora_rank, gemm_n] (FLOAT16)
 */
inline constexpr char const* kLoraBPrefix = "lora_B";

/*!
 * @brief EDGELLM version
 *
 * Value: "major.minor.patch.build"
 * Example: "0.5.0.0"
 */
inline constexpr char const* kEdgellmVersion = "edgellm_version";

/*! @} */

/*! @name Utility Functions
 * @{
 */

/*!
 * @brief Format KV cache binding name for a specific layer
 *
 * @param layerIdx The decoder layer index
 * @param isPast Whether this is past (true) or present (false) key-values
 * @return Formatted binding name like "past_key_values_0" or "present_key_values_0"
 */
inline std::string formatKVCacheName(int32_t layerIdx, bool isPast = true)
{
    return std::string(isPast ? kPastKeyValuesTemplate : kPresentKeyValuesTemplate) + "_" + std::to_string(layerIdx);
}

/*!
 * @brief Format K cache binding name for a specific layer (TensorRT native operations)
 *
 * @param layerIdx The decoder layer index
 * @param isPast Whether this is past (true) or present (false) K cache
 * @return Formatted binding name like "k_cache_0" or "present_k_cache_0"
 */
inline std::string formatKCacheName(int32_t layerIdx, bool isPast = true)
{
    return std::string(isPast ? kKCacheTemplate : kPresentKCacheTemplate) + "_" + std::to_string(layerIdx);
}

/*!
 * @brief Format V cache binding name for a specific layer (TensorRT native operations)
 *
 * @param layerIdx The decoder layer index
 * @param isPast Whether this is past (true) or present (false) V cache
 * @return Formatted binding name like "v_cache_0" or "present_v_cache_0"
 */
inline std::string formatVCacheName(int32_t layerIdx, bool isPast = true)
{
    return std::string(isPast ? kVCacheTemplate : kPresentVCacheTemplate) + "_" + std::to_string(layerIdx);
}

/*!
 * @brief Format recurrent state binding name for a specific recurrent layer
 *
 * @param recurrentLayerIdx The recurrent layer index (0-based, only counting recurrent layers)
 * @param isPast Whether this is past (true) or present (false) recurrent state
 * @return Formatted binding name like "recurrent_state_0" or "present_recurrent_state_0"
 */
inline std::string formatRecurrentStateName(int32_t recurrentLayerIdx, bool isPast = true)
{
    return std::string(isPast ? kRecurrentStateTemplate : kPresentRecurrentStateTemplate) + "_"
        + std::to_string(recurrentLayerIdx);
}

/*!
 * @brief Check if a binding name is a recurrent state tensor
 *
 * @param bindingName The tensor binding name to check
 * @return True if the binding is a recurrent state tensor
 */
inline bool isRecurrentStateBinding(std::string const& bindingName)
{
    if (bindingName.find(kIntermediateRecurrentStateTemplate) != std::string::npos)
    {
        return false;
    }
    return bindingName.find(kRecurrentStateTemplate) != std::string::npos
        || bindingName.find(kPresentRecurrentStateTemplate) != std::string::npos;
}

/*!
 * @brief Format conv state binding name for a specific recurrent layer
 *
 * @param recurrentLayerIdx The recurrent layer index (0-based, only counting recurrent layers)
 * @param isPast Whether this is past (true) or present (false) conv state
 * @return Formatted binding name like "conv_state_0" or "present_conv_state_0"
 */
inline std::string formatConvStateName(int32_t recurrentLayerIdx, bool isPast = true)
{
    return std::string(isPast ? kConvStateTemplate : kPresentConvStateTemplate) + "_"
        + std::to_string(recurrentLayerIdx);
}

/*!
 * @brief Check if a binding name is a conv state tensor
 *
 * @param bindingName The tensor binding name to check
 * @return True if the binding is a conv state tensor
 */
inline bool isConvStateBinding(std::string const& bindingName)
{
    if (bindingName.find(kIntermediateConvStateTemplate) != std::string::npos)
    {
        return false;
    }
    return bindingName.find(kConvStateTemplate) != std::string::npos
        || bindingName.find(kPresentConvStateTemplate) != std::string::npos;
}

/*!
 * @brief Format intermediate recurrent state binding name for MTP
 *
 * @param recurrentLayerIdx The recurrent layer index (0-based)
 * @return Formatted binding name like "intermediate_recurrent_state_0"
 */
inline std::string formatIntermediateRecurrentStateName(int32_t recurrentLayerIdx)
{
    return std::string(kIntermediateRecurrentStateTemplate) + "_" + std::to_string(recurrentLayerIdx);
}

/*!
 * @brief Format intermediate conv state binding name for MTP
 *
 * @param recurrentLayerIdx The recurrent layer index (0-based)
 * @return Formatted binding name like "intermediate_conv_state_0"
 */
inline std::string formatIntermediateConvStateName(int32_t recurrentLayerIdx)
{
    return std::string(kIntermediateConvStateTemplate) + "_" + std::to_string(recurrentLayerIdx);
}

/*!
 * @brief Format the Mamba spec-verify replay-stash binding names (dA / u / B).
 * @param recurrentLayerIdx The recurrent layer index (0-based)
 */
inline std::string formatReplayDaStateName(int32_t recurrentLayerIdx)
{
    return std::string(kReplayDaStateTemplate) + "_" + std::to_string(recurrentLayerIdx);
}

inline std::string formatReplayUStateName(int32_t recurrentLayerIdx)
{
    return std::string(kReplayUStateTemplate) + "_" + std::to_string(recurrentLayerIdx);
}

inline std::string formatReplayBStateName(int32_t recurrentLayerIdx)
{
    return std::string(kReplayBStateTemplate) + "_" + std::to_string(recurrentLayerIdx);
}

/*!
 * @brief Check if a binding name is a LoRA weight tensor
 *
 * @param bindingName The tensor binding name to check
 * @return True if the binding is a LoRA weight tensor
 */
inline bool isLoraBinding(std::string const& bindingName) noexcept
{
    return bindingName.find(kLoraAPrefix) != std::string::npos || bindingName.find(kLoraBPrefix) != std::string::npos;
}

/*!
 * @brief Check if a binding name is a KV cache tensor
 *
 * @param bindingName The tensor binding name to check
 * @return True if the binding is a KV cache tensor
 */
inline bool isKVCacheBinding(std::string const& bindingName) noexcept
{
    return bindingName.find(kPastKeyValuesTemplate) != std::string::npos
        || bindingName.find(kPresentKeyValuesTemplate) != std::string::npos
        || bindingName.find(kKCacheTemplate) != std::string::npos
        || bindingName.find(kVCacheTemplate) != std::string::npos
        || bindingName.find(kPresentKCacheTemplate) != std::string::npos
        || bindingName.find(kPresentVCacheTemplate) != std::string::npos;
}

/*!
 * @brief Format deepstack features binding name for a specific layer
 *
 * @param layerIdx The layer index
 * @return Formatted binding name like "deepstack_features_0"
 */
inline std::string formatDeepstackFeaturesName(int32_t layerIdx)
{
    return std::string(kDeepstackFeaturesTemplate) + "_" + std::to_string(layerIdx);
}

/*!
 * @brief Format deepstack embeddings binding name for a specific index
 *
 * @param embedIdx The embedding index (0, 1, or 2 for Qwen3VL)
 * @return Formatted binding name like "deepstack_embeds_0"
 */
inline std::string formatDeepstackEmbedsName(int32_t embedIdx)
{
    return std::string(kDeepstackEmbedsTemplate) + "_" + std::to_string(embedIdx);
}

/*!
 * @brief Format Gemma4 PLE token embedding input binding name.
 *
 * @param layerIdx The decoder layer index
 * @return Formatted binding name like "ple_token_embeds_0"
 */
inline std::string formatPleTokenEmbedsName(int32_t layerIdx)
{
    return std::string(kPleTokenEmbedsTemplate) + "_" + std::to_string(layerIdx);
}

/*! @} */

} // namespace binding_names
} // namespace trt_edgellm
