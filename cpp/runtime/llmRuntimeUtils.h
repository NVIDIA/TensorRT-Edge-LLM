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

#include "action/alpamayo1RunnerUtils.h"
#include "common/tensor.h"
#include "runtime/audioUtils.h"
#include "runtime/imageUtils.h"
#include "runtime/state/contextCache/contextCacheConfig.h"

#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <unordered_map>
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
        std::string type;    //!< Content type (text, image, trajectory)
        std::string content; //!< Text content when content type is text. Image data will be stored in corresponding
                             //!< imageBuffers. For type "trajectory", data is stored in Request::pastTrajectory.
    };
    std::string role;                     //!< Message role (system, user, assistant)
    std::vector<MessageContent> contents; //!< Contents of the message
};

// Streaming types (StreamChannel, StreamChunk, SlotStreamState, FinishReason,
// StreamChannelFinalizer, and streaming free functions) live in
// `runtime/streaming.h`. LLMGenerationRequest::streamChannels holds a vector
// of shared_ptr<StreamChannel>, which only needs a forward declaration here —
// consumers that actually manipulate channels include `runtime/streaming.h`.
class StreamChannel;

// Forward-declared as an opaque enum (underlying type fixed in streaming.h
// via `: uint8_t`) so LLMGenerationResponse::finishReasons can carry per-slot
// reasons; consumers that inspect the enum values include `runtime/streaming.h`.
enum class FinishReason : uint8_t;

/*! \brief Per-token callback info delivered during Thinker decode */
struct TokenCallbackInfo
{
    int32_t tokenId;        //!< Sampled token ID
    int32_t batchIdx;       //!< Batch index (0-based)
    int32_t generationStep; //!< Decode iteration (0-based)
    bool isFinished;        //!< True when this batch element hit EOS
};

/*! \brief Per-token callback type for streaming Thinker output.
 *  Invoked inside the decode loop after cudaStreamSynchronize. */
using TokenCallback = std::function<void(TokenCallbackInfo const&)>;

/*! \brief One top-K log-probability entry for a single generated token.
 *  `piece` is the raw token bytes (Tokenizer::idToPiece), possibly not valid
 *  UTF-8 on its own (byte-level BPE); filled at assembly time, not on the
 *  decoder hot path which only writes tokenId/logprob. */
struct LogprobEntry
{
    int32_t tokenId;   //!< Token ID
    float logprob;     //!< Natural-log probability (<= 0)
    std::string piece; //!< Raw token bytes; empty until filled at assembly time
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
        std::optional<std::vector<PastTrajectoryPoint>>
            pastTrajectory; //!< Optional past trajectory for Alpamayo (e.g. ego x,y,z history)

        //! Stop strings; generation halts on the earliest match and trims it from output.
        std::vector<std::string> stopStrings;

        //! Sparse per-token logit bias map keyed by full tokenizer token ID.
        std::unordered_map<int32_t, float> logitBias;

        mutable FormattedRequest formatted; //!< Formatted request (populated by tokenizer or user-provided)
    };
    //! \endcond
    std::vector<Request> requests; //!< Vector of requests for a batch
    mutable std::vector<FormattedRequest>
        formattedRequests; //!< Formatted requests (mutable to allow runtime modification)
    std::vector<std::vector<int32_t>> preTokenizedInputIds; //!< Optional pre-tokenized text inputs, one per request.
    float temperature;                                      //!< Temperature parameter for sampling
    float topP;                                             //!< Top-p (nucleus) sampling parameter
    int64_t topK;                                           //!< Top-k sampling parameter
    int64_t maxGenerateLength;                              //!< Max length of the generated tokens
    int32_t diffusionMaxDenoisingSteps{0}; //!< Optional DiffusionGemma denoise-step override (0 = runtime default)
    std::string loraWeightsName{""};       //!< Name of the LoRA weights. Default to empty string for no LoRA weights

    // Whether to save system prompt KV cache of this request to be used by later requests
    bool saveSystemPromptKVCache{false};
    // Whether to apply chat template formatting. If false, raw messages will be concatenated without special tokens
    bool applyChatTemplate{true};
    // Whether to add generation prompt (e.g., assistant header) at the end. Only effective when
    // applyChatTemplate=true..
    bool addGenerationPrompt{true};
    // Whether to enable thinking mode for models that support it. Default is disabled.
    bool enableThinking{false};
    // Disable speculative decoding for this request when the loaded engine contract supports vanilla fallback.
    bool disableSpecDecode{false};

    //! Number of top log-probabilities to return per generated token (0 = disabled, max = kMaxLogprobsK).
    //! Logprobs are computed as log(softmax(logits)) and returned in LLMGenerationResponse::logprobs.
    int32_t numLogprobs{0};

    //! Per-slot streaming channels. Size 0 disables streaming globally.
    //! When non-empty the size must equal `requests.size()` and individual entries may be null
    //! to opt out on a per-slot basis. Channels must not already be finished or concurrently
    //! attached to another in-flight request.
    std::vector<std::shared_ptr<StreamChannel>> streamChannels;

    // Audio generation flag (Qwen3-Omni Talker)
    bool generateAudio{false};
    //!< Whether to enable hidden states capture for Talker pipeline

    int32_t acceptHiddenLayer{0};
    //!< Hidden layer index for Talker (from talker_config.accept_hidden_layer)

    //! Optional per-token callback invoked after each decode step.
    //! Called after cudaStreamSynchronize inside the decode loop.
    //! When nullopt (default), zero overhead — no callback is invoked.
    std::optional<TokenCallback> onTokenGenerated;

    //! Per-request context-cache lookup behavior. This is effective only when the runtime cache is enabled.
    ContextCacheLookupPolicy contextCacheLookupPolicy{ContextCacheLookupPolicy::kUseCache};

    //! Ready endpoints to retain when the context cache is enabled.
    ContextCacheCommitPolicy contextCacheCommitPolicy{ContextCacheCommitPolicy::kIncludingGeneratedTokens};

    //! Hybrid+MTP boundary-replay tail length carried into the context cache. Not consumed yet.
    int32_t contextCacheReplayTailLength{0};

    //! Periodic recurrent-state capture interval (0 disables). Hybrid+MTP endpoint reuse requires this to be 0 so the
    //! recurrent snapshot lands only at the stable predecessor boundary (mirrors reference request validation).
    int32_t recurrentCaptureInterval{0};
};

/*! \brief LLM Generation Response structure
 */
struct LLMGenerationResponse
{
    std::vector<std::vector<int32_t>> outputIds; //!< Generated token IDs for each request in the batch
    std::vector<std::string> outputTexts;        //!< Generated text strings for each request in the batch
    //! Top log-probabilities per generated token: logprobs[batch][step] = [LogprobEntry, ...].
    //! Sorted by descending probability. Populated only when LLMGenerationRequest::numLogprobs > 0.
    std::vector<std::vector<std::vector<LogprobEntry>>> logprobs;
    //!< Future trajectory waypoints (e.g. accel, kappa) per batch item; populated when action engine is used
    std::vector<std::vector<FutureTrajectoryPoint>> outputTrajectories;

    std::vector<rt::audioUtils::AudioData> outputAudios; //!< Generated audio data (Qwen3-Omni only)

    //! Why each request halted (EOS, length, stop string, cancel, error); see `runtime/streaming.h`.
    std::vector<FinishReason> finishReasons;

    //! Prompt length per request, counted after chat templating and media expansion.
    std::vector<int32_t> inputTokenCounts;
};

/*! \brief RoPE (Rotary Position Embedding) type enumeration
 */
enum class RopeType
{
    kDefault,      //!< Default 1-D RoPE that specified by the original paper
    kDynamic,      //!< Dynamic RoPE type used by InternVL-3
    kProportional, //!< Proportional RoPE with identity no-position angle slots
    kLongRope,     //!< Long RoPE type used by Phi-4
    kMRope,        //!< MRope type used by Qwen2-VL
    kNoRope,       //!< No positional encoding (e.g., Nemotron-Nano)
    kYarn,         //!< YaRN NTK-by-parts scaling
};

/*! \brief Long-Rope specific parameters */
struct LongRopeParams
{
    int32_t originalMaxPositionEmbeddings{-1}; //!< Original maximum position embeddings from training
    std::vector<float> longFactor;             //!< Long factor array for each rotary dimension
    std::vector<float> shortFactor;            //!< Short factor array for each rotary dimension
};

/*! \brief YaRN specific parameters (NTK-by-parts interpolation) */
struct YarnParams
{
    int32_t originalMaxPositionEmbeddings{-1}; //!< Pre-YaRN training length; the interpolation reference
    float factor{1.0F};                        //!< Context-extension factor (rope_scaling.factor)
    float betaFast{32.0F};                     //!< High-frequency correction boundary (rotations)
    float betaSlow{1.0F};                      //!< Low-frequency correction boundary (rotations)
    float mscale{1.0F};                        //!< Attention magnitude scale applied to cos/sin
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
    float partialRotaryFactor{1.0F};          //!< Fraction of head angles rotated by proportional RoPE
    int32_t maxPositionEmbeddings{32768};     //!< Maximum position embeddings supported
    std::optional<LongRopeParams> longRope{}; //!< Long-Rope specific parameters
    std::optional<YarnParams> yarn{};         //!< YaRN specific parameters
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

//=============================================================================
// Embedding Loading Utilities
//=============================================================================

/*! \brief FP8 embedding block size (fixed at 128)
 *
 *  This matches the quantization granularity: scales shape is [vocabSize, hiddenSize/128]
 */
inline constexpr int64_t kFP8EmbeddingBlockSize = 128;

/*! \brief Embedding data - supports both FP16 and FP8 formats
 *
 *  The embedding table datatype determines the format:
 *  - FP16: table is FP16, tableScalingFactor is empty
 *  - FP8: table is FP8 (E4M3), tableScalingFactor contains FP32 per-group scales
 *
 *  The kernel functions automatically dispatch based on table.getDataType().
 */
struct EmbeddingData
{
    rt::Tensor table;              //!< Embedding table [vocabSize, hiddenSize] (FP16 or FP8)
    rt::Tensor tableScalingFactor; //!< FP32 per-group scales [vocabSize, hiddenSize/128] (only if FP8)

    //! \brief Returns scales as OptionalInputTensor (std::nullopt when FP16, reference when FP8)
    rt::OptionalInputTensor scalesAsOptional() const
    {
        return tableScalingFactor.getShape().volume() > 0 ? rt::OptionalInputTensor{tableScalingFactor} : std::nullopt;
    }
};

/*! \brief Load embedding table from safetensors file (auto-detects FP16 vs FP8 by dtype)
 *
 *  Loads embedding.safetensors and detects format by checking the "embedding" tensor dtype:
 *  - FP8: loads "embedding" (FP8) + "embedding_scale" (FP32)
 *  - FP16: loads "embedding" (FP16)
 *
 *  \param embeddingPath Path to embedding.safetensors file
 *  \param stream CUDA stream for async operations
 *  \return EmbeddingData with loaded tensors and format flag
 *  \throws std::runtime_error if file not found, tensors missing, or invalid dtypes
 */
EmbeddingData loadEmbeddingTable(std::filesystem::path const& embeddingPath, cudaStream_t stream);

/*!
 * @brief Clamp max generation length against KV-cache capacity across the full active batch
 *
 * Uses the smallest remaining KV budget across all active sequences so the shared
 * generation limit cannot overrun any batch item.
 *
 * @param effectivePrefillLengths Effective prefill lengths for each active sequence
 * @param requestedMaxGenerateLength User-requested max generation length
 * @param kvCacheCapacity Total KV-cache capacity available to the runtime
 * @param kvCacheReserveLength Extra KV reserve required by the runtime mode
 * @return Clamped max generation length, never below 0
 */
int32_t clampMaxGenerateLengthForKVCapacity(std::vector<int32_t> const& effectivePrefillLengths,
    int32_t requestedMaxGenerateLength, int32_t kvCacheCapacity, int32_t kvCacheReserveLength);

/*!
 * @brief Generate multimodal indices for embeddingLookup kernel
 *
 * Scans input IDs and generates sequential indices for audio/image embeddings.
 * Audio and image indices are tracked independently, both globally across batches.
 *
 * @param inputIds Input token IDs on CPU [batchSize, seqLen]
 * @param audioTokenId Special token ID for audio, or std::nullopt if no audio
 * @param imageTokenId Special token ID for image, or std::nullopt if no image
 * @return multimodalIndices tensor on CPU [batchSize, seqLen]
 */
rt::Tensor generateMultimodalIndices(
    rt::Tensor const& inputIds, std::optional<int32_t> audioTokenId, std::optional<int32_t> imageTokenId);

/*! \brief Build Gemma4 block IDs from host token IDs.
 *
 * Image placeholders form one block for each contiguous run.  Such positions
 * receive a non-negative run ID; text, audio, and padding receive -1.  This
 * matches Transformers' get_block_sequence_ids_for_mask semantics.
 */
rt::Tensor generateVisionBlockIds(rt::Tensor const& inputIds, int32_t imageTokenId);

} // namespace rt
} // namespace trt_edgellm
