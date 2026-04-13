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

#include "common/tensor.h"
#include "runtime/audioUtils.h"
#include "runtime/imageUtils.h"

#include <cstdint>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace trt_edgellm
{
namespace rt
{

/*!
 * @brief Message with role and contents
 */
struct Message
{
    struct MessageContent
    {
        std::string type;    //!< Content type (text, image)
        std::string content; //!< Text content when content type is text. Image data will be stored in corresponding
                             //!< imageBuffers.
    };
    std::string role;                     //!< Message role (system, user, assistant)
    std::vector<MessageContent> contents; //!< Contents of the message
};

/*! \brief LLM Generation Request structure
 */
struct LLMGenerationRequest
{
    //! \cond INTERNAL
    /*!
     * @brief Formatted request structure containing chat template output
     */
    struct FormattedRequest
    {
        std::string formattedSystemPrompt;    //!< Formatted prefix system prompt that can be used for KVCache saving.
        std::string formattedCompleteRequest; //!< Formatted complete request (including prefix system prompt)
    };

    /*! \brief Request structure containing structured messages
     */
    struct Request
    {
        std::vector<Message> messages; //!< Structured messages (required - use chat template format)
        std::vector<rt::imageUtils::ImageData> imageBuffers; //!< Optional image data for multimodal inputs
        std::vector<rt::audioUtils::AudioData> audioBuffers; //!< Optional audio data for multimodal inputs (Qwen3-Omni)
        std::string preparedVisualInputPath; //!< Optional path to preprocessed visual inputs (pixel_values + image_grid_thw)
        std::string dumpPrefillKVCachePath; //!< Optional output path to dump prefill KV cache as safetensors
        bool prefillOnly{false}; //!< If true, stop after prefill (useful with dumpPrefillKVCachePath)

        mutable FormattedRequest formatted; //!< Formatted request (populated by tokenizer or user-provided)
    };
    //! \endcond
    std::vector<Request> requests; //!< Vector of requests for a batch
    mutable std::vector<FormattedRequest>
        formattedRequests;           //!< Formatted requests (mutable to allow runtime modification)
    float temperature;               //!< Temperature parameter for sampling
    float topP;                      //!< Top-p (nucleus) sampling parameter
    int64_t topK;                    //!< Top-k sampling parameter
    int64_t maxGenerateLength;       //!< Max length of the generated tokens
    std::string loraWeightsName{""}; //!< Name of the LoRA weights. Default to empty string for no LoRA weights

    // Whether to save system prompt KV cache of this request to be used by later requests
    bool saveSystemPromptKVCache{false};
    // Whether to apply chat template formatting. If false, raw messages will be concatenated without special tokens
    bool applyChatTemplate{true};
    // Whether to add generation prompt (e.g., assistant header) at the end. Only effective when
    // applyChatTemplate=true..
    bool addGenerationPrompt{true};
    // Whether to enable thinking mode for models that support it. Default is disabled.
    bool enableThinking{false};
    // Always disable speculative decoding for this request even if Eagle Draft engine is loaded.
    bool disableSpecDecode{false};
};

/*! \brief LLM Generation Response structure
 */
struct LLMGenerationResponse
{
    std::vector<std::vector<int32_t>> outputIds; //!< Generated token IDs for each request in the batch
    std::vector<std::string> outputTexts;        //!< Generated text strings for each request in the batch
};

/*! \brief RoPE (Rotary Position Embedding) type enumeration
 */
enum class RopeType
{
    kDefault,  //!< Default 1-D RoPE that specified by the original paper
    kDynamic,  //!< Dynamic RoPE type used by InternVL-3
    kLongRope, //!< Long RoPE type used by Phi-4
    kMRope,    //!< MRope type used by Qwen2-VL
    kNoRope,   //!< No positional encoding (e.g., Nemotron-Nano)
};

/*! \brief Long-Rope specific parameters */
struct LongRopeParams
{
    int32_t originalMaxPositionEmbeddings{-1}; //!< Original maximum position embeddings from training
    std::vector<float> longFactor;             //!< Long factor array for each rotary dimension
    std::vector<float> shortFactor;            //!< Short factor array for each rotary dimension
};

/*! \brief RoPE configuration structure with optional Long-Rope parameters
 *
 *  Contains common RoPE fields and (optionally) Long-Rope specific parameters when type==kLongRope.
 */
struct RopeConfig
{
    RopeType type{RopeType::kDefault};        //!< Type of RoPE to use
    float rotaryScale{1.0F};                  //!< Scaling factor for rotary embeddings
    float rotaryTheta{100000.0F};             //!< Base frequency for rotary embeddings
    int32_t maxPositionEmbeddings{32768};     //!< Maximum position embeddings supported
    std::optional<LongRopeParams> longRope{}; //!< Long-Rope specific parameters
};

/*! \brief Collect rope configuration from the model config
 *
 *  Parses the common RoPE fields as well as LongRoPE-specific parameters when the
 *  model requests the longrope variant. Default values are used if certain fields
 *  are not specified in the model config.
 *
 *  \param config [JSON] The model config file supplied with the model
 *  \return The parsed rope configuration
 *  \throws nlohmann::json::type_error if JSON value types don't match expected types
 */
RopeConfig collectRopeConfig(nlohmann::json const& config);

/*! \brief Initialize the rope cos/sin cache tensor for persistent type of RoPE (default, longrope)
 *
 *  \param cosSinCache [GPU] The tensor to store the rope cos/sin cache
 *  \param config [RopeConfig] The basic rope configuration
 *  \param stream [CUDA stream] The stream to execute the initialization
 *  \return True if the initialization is successful, false otherwise
 */
bool initializeRopeCosSinCache(rt::Tensor& cosSinCache, RopeConfig const& config, cudaStream_t stream) noexcept;

/*! \brief Initialize an identity cos/sin cache for models without positional encoding (NoPE)
 *
 *  Fills the first half of each position's rotaryDim with 1.0 (cos) and the
 *  second half with 0.0 (sin), making the RoPE kernel a pass-through.
 *
 *  \param cosSinCache [GPU] The tensor to fill, shape [1, maxLength, rotaryDim]
 *  \param stream [CUDA stream] The stream to execute the copy
 *  \return True on success
 */
bool initializeNopeCosSinCache(rt::Tensor& cosSinCache, cudaStream_t stream) noexcept;

/*! \brief Initialize the rope cos/sin cache tensor for long rope type
 *
 *  \param shortCosSinCache [GPU] The tensor to store the short rope cos/sin cache
 *  \param longCosSinCache [GPU] The tensor to store the long rope cos/sin cache
 *  \param config [RopeConfig] The rope configuration
 *  \param stream [CUDA stream] The stream to execute the initialization
 *  \return True if the initialization is successful, false otherwise
 *  \throws std::runtime_error if CUDA operations fail
 */
bool initializeLongRopeCosSinCache(
    rt::Tensor& shortCosSinCache, rt::Tensor& longCosSinCache, RopeConfig const& config, cudaStream_t stream);

/*!
 * @brief Format rope configuration into string
 */
std::string formatRopeConfig(RopeConfig const& config);

/**
 * @brief Compact CPU vector by removing evicted batches
 *
 * This utility function compacts a std::vector by removing elements at evicted batch indices.
 * Used for batch eviction to remove finished sequences from CPU context vectors.
 *
 * @tparam T Element type
 * @param batchMapping      [oldActiveBatch] CPU vector (const input), mapping[i] = newBatchIdx or -1 (evict)
 * @param vec               Vector to compact (output, modified in-place)
 * @throws std::invalid_argument if sizes of input vectors don't match
 */
template <typename T>
void compactVector(std::vector<int32_t> const& batchMapping, std::vector<T>& vec);

/**
 * @brief Build batch mapping from finished states
 *
 * Creates a mapping vector that maps old batch indices to new batch indices.
 * Finished batches are marked with -1 for eviction.
 *
 * @param finishedStates    [oldActiveBatch] CPU vector indicating which batches are finished (0=not finished,
 * 1=finished)
 * @return Vector mapping old batch indices to new indices (-1 for evicted batches)
 */
std::vector<int32_t> buildBatchMapping(std::vector<int8_t> const& finishedStates);

} // namespace rt
} // namespace trt_edgellm
