/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * @brief Input token IDs tensor - contains the tokenized input sequence
 *
 * Shape: [batch_size, sequence_length] (INT32)
 */
inline constexpr char const* kInputIds = "input_ids";

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
 * @brief Past key-value cache tensor template - use with layer index formatting
 *
 * Template: "past_key_values.{layer_idx}"
 * Shape: [batch_size, 2, num_kv_heads, seq_len, head_dim] (FLOAT16)
 */
inline constexpr char const* kPastKeyValuesTemplate = "past_key_values";

/*!
 * @brief Present key-value cache tensor template - use with layer index formatting
 *
 * Template: "present_key_values.{layer_idx}"
 * Shape: [batch_size, 2, num_kv_heads, seq_len, head_dim] (FLOAT16)
 */
inline constexpr char const* kPresentKeyValuesTemplate = "present_key_values";

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

/*!
 * @brief Attention position IDs for Eagle models
 *
 * Shape: [batch_size, tree_size] (INT32)
 */
inline constexpr char const* kAttentionPosId = "attention_pos_id";

/*! @} */

/*! @name Vision-Language Model (VLM) Bindings
 * @{
 */

/*!
 * @brief Multimodal image embeddings tensor
 *
 * Shape: [num_image_tokens, hidden_size] (FLOAT16)
 */
inline constexpr char const* kImageEmbeds = "image_embeds";

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
 * @brief Visual output tensor from vision transformers
 *
 * Shape: [num_image_tokens, hidden_size] (FLOAT16)
 */
inline constexpr char const* kVisualOutput = "output";

/*!
 * @brief Rotary positional embeddings for visual inputs (Qwen-VL specific)
 *
 * Shape: [sequence_length, embed_dim] (FLOAT32)
 */
inline constexpr char const* kRotaryPosEmb = "rotary_pos_emb";

/*!
 * @brief Window attention mask for Qwen2.5-VL models
 *
 * Shape: [1, sequence_length, sequence_length] (FLOAT16)
 */
inline constexpr char const* kWindowAttentionMask = "window_attention_mask";

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
 * @brief Deepstack features tensor for Qwen3-VL vision model
 *
 * Shape: [num_image_tokens, hidden_size] (FLOAT16)
 */
inline constexpr char const* kDeepstackFeaturesTemplate = "deepstack_features";

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
 * Example: "0.4.0.0"
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
 * @return Formatted binding name like "past_key_values.0" or "present_key_values.0"
 */
inline std::string formatKVCacheName(int32_t layerIdx, bool isPast = true)
{
    return std::string(isPast ? kPastKeyValuesTemplate : kPresentKeyValuesTemplate) + "." + std::to_string(layerIdx);
}

/*!
 * @brief Check if a binding name is a LoRA weight tensor
 *
 * @param bindingName The tensor binding name to check
 * @return True if the binding is a LoRA weight tensor
 */
inline bool isLoraBinding(std::string const& bindingName)
{
    return bindingName.find(kLoraAPrefix) != std::string::npos || bindingName.find(kLoraBPrefix) != std::string::npos;
}

/*!
 * @brief Check if a binding name is a KV cache tensor
 *
 * @param bindingName The tensor binding name to check
 * @return True if the binding is a KV cache tensor
 */
inline bool isKVCacheBinding(std::string const& bindingName)
{
    return bindingName.find(kPastKeyValuesTemplate) != std::string::npos
        || bindingName.find(kPresentKeyValuesTemplate) != std::string::npos;
}

/*!
 * @brief Format deepstack features binding name for a specific layer
 *
 * @param layerIdx The layer index
 * @return Formatted binding name like "deepstack_features.0"
 */
inline std::string formatDeepstackFeaturesName(int32_t layerIdx)
{
    return std::string(kDeepstackFeaturesTemplate) + "." + std::to_string(layerIdx);
}

/*! @} */

} // namespace binding_names
} // namespace trt_edgellm
