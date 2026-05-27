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
#include "profiling/metrics.h"
#include "runtime/llmEngineRunner.h"
#include "runtime/llmInferenceSpecDecodeRuntime.h"
#include "runtime/llmRuntimeUtils.h"
#include "tokenizer/tokenizer.h"
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace trt_edgellm
{

// Forward declaration
struct SamplingParams;

namespace rt
{

// ========== Constants ==========

namespace talker_constants
{
constexpr int32_t kDefaultNumRvqLayers = 15; //!< Default RVQ layers (Omni=15, TTS=31; auto-detected at runtime)
constexpr int32_t kAssistantPrefixLen = 3;   //!< Assistant prefix tokens ([:3])
constexpr int32_t kAssistantTrailingSuffix
    = 5; //!< Trailing tokens to strip from end of sequence ("<|im_end|>\n<|im_start|>assistant\n")
constexpr int32_t kNonStreamingPrefixRows = 8;     //!< Fixed prefix rows in non-streaming prefill (rows 0-7)
constexpr int32_t kCodePredictorPrefillSeqLen = 2; //!< CodePredictor prefill sequence length
constexpr int32_t kCodecEmbeddingCount = 6;        //!< Number of codec embeddings to add

// Audio output constants (Qwen3-Omni codec: 12.5 Hz frame rate, 24 kHz mono PCM output)
constexpr int32_t kAudioSampleRate = 24000;     //!< Output PCM sample rate (Hz)
constexpr int32_t kAudioSamplesPerFrame = 1920; //!< Samples produced per codec frame (24000 / 12.5)

// Chat-template token IDs (from Qwen3-Omni tokenizer vocabulary)
constexpr int32_t kImStartTokenId = 151644; //!< <|im_start|>
constexpr int32_t kAssistantRoleId = 77091; //!< "assistant" role token
constexpr int32_t kUserRoleId = 872;        //!< "user" role token
constexpr int32_t kSystemRoleId = 8948;     //!< "system" role token
constexpr int32_t kAudioTokenId = 151675;   //!< Audio placeholder token
constexpr int32_t kImageTokenId = 151655;   //!< Image placeholder token
constexpr int32_t kVideoTokenId = 151656;   //!< Video placeholder token
} // namespace talker_constants

/*!
 * @brief Talker runtime for Qwen3-Omni RVQ code generation
 *
 * LLM-based codec encoder that generates RVQ codes from text tokens and hidden states.
 * Manages two LLM engines (Talker + CodePredictor) and MLP projection layers.
 *
 * Pipeline:
 *   1. MLP Projection: thinker embed (layer 0) → talker embeddings via text_projection
 *   2. Talker LLM: generate codec tokens autoregressively
 *   3. CodePredictor: generate multi-layer codebook codes (Omni: 15, TTS: 31)
 *   4. Return RVQ codes (vocoding done separately at example layer)
 *
 * Architecture Philosophy:
 *   - Talker is an LLM decoder, NOT a multimodal input encoder
 *   - Similar to LLMInferenceSpecDecodeRuntime, manages multiple LLM engines
 *   - Standalone runtime, not dependent on MultimodalRunner hierarchy
 *   - Code2Wav vocoding is separated for better modularity
 */
class Qwen3OmniTTSRuntime
{
public:
    /*!
     * @brief Construct and fully initialize the TTS runtime
     * @param talkerEngineDir Directory containing talker engine, MLP weights, embedding table, etc.
     * @param codePredictorEngineDir Directory containing code_predictor engine and codec embeddings
     * @param tokenizerDir Directory containing tokenizer files. If empty, defaults to talkerEngineDir/../
     * @param stream CUDA stream for operations
     * @throws std::runtime_error on any initialization failure
     */
    Qwen3OmniTTSRuntime(std::string const& talkerEngineDir, std::string const& codePredictorEngineDir,
        std::string const& tokenizerDir, cudaStream_t stream);

    //! @brief Destructor
    ~Qwen3OmniTTSRuntime();

    // ========== Core API ==========

    /*!
     * @brief Talker audio generation request structure
     *
     * Contains sampling parameters and input data for audio generation.
     * Sampling parameters are provided per-request (not from config.json).
     */
    struct TalkerGenerationRequest
    {
        int32_t maxAudioLength{4096}; //!< Maximum number of audio codec tokens to generate

        // Talker/CodePredictor sampling parameters (independent from Thinker)
        // 0 = use PyTorch defaults: temperature=0.9, top_k=50, top_p=1.0
        float talkerTemperature{0};     //!< Talker temperature (0 = default 0.9)
        int32_t talkerTopK{0};          //!< Talker top-K (0 = default 50)
        float talkerTopP{0};            //!< Talker top-P (0 = default 1.0)
        float repetitionPenalty{1.05f}; //!< Repetition penalty applied to seen codec tokens (1.0 = disabled)

        // Speaker selection (optional, defaults to config default)
        std::string speakerName{""}; //!< Speaker name (e.g., "f245", "m02") - empty means use default
        //!< CustomVoice language conditioning (e.g., "chinese", "english"). Empty = no language path.
        //!< Lower-cased by the runtime before lookup in TalkerConfig::codecLanguageId.
        std::string language{""};
        int32_t speakerId{-1}; //!< Speaker ID - if >= 0, overrides speakerName

        // Input: conversation messages for this request (runtime tokenizes internally)
        std::vector<Message> messages;
        bool applyChatTemplate{true};   //!< Whether to apply chat template formatting
        bool addGenerationPrompt{true}; //!< Whether to add generation prompt at the end
        bool enableThinking{false};     //!< Whether to enable thinking mode
    };

    /*!
     * @brief Talker audio generation response structure
     *
     * Contains generated RVQ codes and metadata.
     */
    struct TalkerGenerationResponse
    {
        // RVQ codes: [batchSize][numFrames][mNumCodesPerFrame]
        std::vector<std::vector<std::vector<int32_t>>> batchRvqCodes;

        // Metadata
        std::vector<int32_t> numFramesPerSample; //!< Number of audio frames generated per batch sample
        bool success{false};                     //!< Whether generation succeeded
    };

    /*!
     * @brief Get required hidden state layer indices from thinker
     * @return Vector containing {0} for layer 0 (embed) and {14} for accept_hidden_layer
     */
    std::vector<int32_t> getThinkerHiddenLayerIndices() const
    {
        return {0, 14};
    }

    /*!
     * @brief Generate audio with RVQ codes (batched)
     *
     * Implements the complete nested generation loop for a batch of requests:
     * - Talker generation loop (autoregressive, batched engine execution)
     * - CodePredictor generation (mNumRvqLayers per Talker step, per-batch)
     * - Residual connections
     * - Sampling at Runtime Layer (batched)
     *
     * This is the main entry point for audio generation, analogous to
     * LLMInferenceSpecDecodeRuntime::handleRequest() for standard LLM inference.
     *
     * @note Sampling parameters (temperature, topK, topP, repetitionPenalty) are taken
     *       from requests[0] and applied uniformly to all batches. This matches
     *       LLMInferenceSpecDecodeRuntime's design where SamplingParams is shared across the batch.
     *
     * @param requests Batch of requests, each containing per-batch input data
     * @param response Response containing generated RVQ codes [batchSize][frames][codes]
     * @param stream CUDA stream for execution
     * @return True if generation succeeded, false otherwise
     */
    bool handleAudioGeneration(
        std::vector<TalkerGenerationRequest> const& requests, TalkerGenerationResponse& response, cudaStream_t stream);

    //! @brief Convenience wrapper for single-request audio generation
    bool handleAudioGeneration(
        TalkerGenerationRequest const& request, TalkerGenerationResponse& response, cudaStream_t stream)
    {
        return handleAudioGeneration(std::vector<TalkerGenerationRequest>{request}, response, stream);
    }

    /*!
     * @brief Request structure for Omni inference (Thinker output as input)
     *
     * Non-streaming: provide fullText (formatted prompt + generated text), which will be
     * tokenized internally to reconstruct layer-0 embeddings via the Thinker embedding table.
     */
    struct OmniGenerationRequest
    {
        std::string fullText;              //!< Complete formatted text (if textTokenIds empty, tokenized internally)
        std::vector<int32_t> textTokenIds; //!< Full token sequence: inputTokenIds + outputIds (including EOS)

        //!< Non-owning pointer to this batch's prefill layer-0 embeddings (with multimodal features).
        //!< Must point to a [1, prefillLength, thinkerHiddenSize] FP16 (GPU) view for this batch.
        //!< Caller slices from the full [BS, prefillLen, H] tensor. Generated token embeddings
        //!< are reconstructed from the TTS embedding table internally.
        rt::Tensor const* thinkerPrefillEmbeds{nullptr};

        //!< Non-owning pointer to this batch's layer-14 hidden states (prefill only).
        //!< Must point to a [1, prefillLength, thinkerHiddenSize] FP16 (GPU) view for this batch.
        //!< Only user-segment multimodal token positions are read.
        rt::Tensor const* thinkerHiddenStates{nullptr};

        int32_t prefillLength{0}; //!< Number of prefill tokens (layer0/layer14 cover [0, prefillLength))

        int32_t maxAudioLength{4096};
        float talkerTemperature{0};
        int32_t talkerTopK{0};
        float talkerTopP{0};
        float repetitionPenalty{1.05f};
        std::string speakerName{""};
        int32_t speakerId{-1};
    };

    /*!
     * @brief Generate audio from external Thinker hidden states (Omni inference path, batched)
     *
     * Instead of tokenizing text and looking up embeddings internally (TTS path),
     * this API accepts pre-computed Thinker layer-0 hidden states and projects them
     * through the MLP to produce Talker input. Used when integrating with llm_inference.
     *
     * @note Sampling parameters (temperature, topK, topP, repetitionPenalty) are taken
     *       from requests[0] and applied uniformly to all batches. This matches
     *       LLMInferenceSpecDecodeRuntime's design where SamplingParams is shared across the batch.
     *
     * @param requests Batch of requests, each containing per-batch thinker embeddings
     * @param response Response containing generated RVQ codes [batchSize][frames][codes]
     * @param stream CUDA stream for execution
     * @return True if generation succeeded, false otherwise
     */
    bool handleAudioGenerationFromThinker(
        std::vector<OmniGenerationRequest> const& requests, TalkerGenerationResponse& response, cudaStream_t stream);

    //! @brief Convenience wrapper for single-request Omni audio generation
    bool handleAudioGenerationFromThinker(
        OmniGenerationRequest const& request, TalkerGenerationResponse& response, cudaStream_t stream)
    {
        return handleAudioGenerationFromThinker(std::vector<OmniGenerationRequest>{request}, response, stream);
    }

    // ========== Thinker-Talker Streaming Pipeline ==========

    /*!
     * @brief Configuration for Thinker→Talker streaming pipeline
     */
    using AudioChunkCallback = std::function<void(std::vector<std::vector<int32_t>> const& chunkRvqCodes)>;

    struct ThinkerTalkerStreamingConfig
    {
        int32_t talkerPrefillThreshold{4};    //!< Start Talker prefill after this many assistant tokens
        int32_t codecChunkFrames{0};          //!< Vocode every N frames during flush (0 = disabled)
        AudioChunkCallback onAudioChunkReady; //!< Called with chunk RVQ codes [frames][16] when ready
    };

    /*!
     * @brief Streaming generation: Thinker and Talker run interleaved on the same CUDA stream
     *
     * Uses LLMGenerationRequest::onTokenGenerated to receive per-token callbacks from
     * the Thinker's decode loop. When enough assistant tokens accumulate, Talker prefill
     * is triggered. Subsequent Thinker tokens incrementally extend trailing_text_hidden,
     * and Talker decode steps are interleaved.
     *
     * @param thinkerRuntime  Thinker LLM runtime (will call handleRequest internally)
     * @param thinkerRequest  Thinker request (onTokenGenerated will be overwritten)
     * @param streamingConfig  Pipeline tuning parameters
     * @param talkerResponse  Output: generated RVQ codes
     * @param stream  CUDA stream (shared by Thinker and Talker)
     * @return True if the full pipeline succeeded
     */
    bool handleStreamingGeneration(LLMInferenceSpecDecodeRuntime& thinkerRuntime, LLMGenerationRequest& thinkerRequest,
        LLMGenerationResponse& thinkerResponse, ThinkerTalkerStreamingConfig const& streamingConfig,
        OmniGenerationRequest const& omniBaseRequest, TalkerGenerationResponse& talkerResponse, cudaStream_t stream);

    /*!
     * @brief Get performance metrics for Talker pipeline (legacy, for backward compat)
     * @return Reference to metrics object
     */
    metrics::MultimodalMetrics const& getMetrics() const
    {
        return mMultimodalMetrics;
    }

    //! @brief Get Omni-specific Talker metrics (frames, RVQ codes, prefill time, exit reason)
    metrics::OmniTalkerMetrics const& getOmniTalkerMetrics() const
    {
        return mOmniTalkerMetrics;
    }

    //! @brief Get Omni audio latency metrics (TTFA, RTF, E2E)
    metrics::OmniLatencyMetrics const& getOmniLatencyMetrics() const
    {
        return mOmniLatencyMetrics;
    }

    //! @brief Get mutable reference to latency metrics (for E2E timing set at example layer)
    metrics::OmniLatencyMetrics& getMutableOmniLatencyMetrics()
    {
        return mOmniLatencyMetrics;
    }

    //! @brief Get the TTFA end event (first codec token sampled) for external timing
    cudaEvent_t getTtfaEndEvent() const
    {
        return mTtfaEnd;
    }

    /*!
     * @brief Capture CUDA graphs for decoding steps (same pattern as LLMInferenceSpecDecodeRuntime).
     * @param stream CUDA stream for capture
     * @return True if all graphs captured successfully
     */
    bool captureDecodingCUDAGraph(cudaStream_t stream);

    /*!
     * @brief Get speaker ID by name
     * @param speakerName Speaker name (e.g., "f245", "m02")
     * @return Speaker ID, or default speaker ID if not found
     */
    int32_t getSpeakerIdByName(std::string const& speakerName) const;

private:
    // ========== Internal Methods ==========

    void initializeTTSEmbeddings(cudaStream_t stream);

    //! @param perBatchContextLengths Optional per-batch context lengths for padded batched prefill.
    //!        If empty, all batches use the full seqLen dimension of inputEmbeds.
    bool executeTalkerPrefillStep(rt::Tensor const& inputEmbeds, rt::Tensor& outputLogits,
        rt::Tensor& outputHiddenStates, cudaStream_t stream, std::vector<int64_t> const& perBatchContextLengths = {});

    //! Run CodePredictor for one frame of one batch element (batch=1 engine call internally).
    //! Called inside the per-batch loop of the decode frame.
    bool runCodePredictorGenerationForFrame(int32_t codecToken, rt::Tensor const& talkerHiddenState,
        SamplingParams const& samplingParams, std::vector<int32_t>& outputCodes, cudaStream_t stream);

    //! Compute residual connection for one batch element.
    bool computeResidualConnection(std::vector<int32_t> const& codes, rt::Tensor const* trailingTextHidden,
        int32_t generationStep, rt::Tensor& outputResidual, cudaStream_t stream);

    //! Extract last hidden state from Talker hidden states buffer for one batch element.
    bool extractTalkerLastHidden(
        rt::Tensor const& talkerHiddenStates, rt::Tensor& outputLastHidden, cudaStream_t stream);

    // ========== Per-Batch State for Generation Loop ==========

    /*!
     * @brief Per-batch state tracked during Talker generation loop
     *
     * Analogous to Thinker's per-batch finishedStates + outputIds tracking.
     * Used by both non-streaming (runTalkerGenerationLoop) and streaming (callback) paths.
     */
    struct PerBatchTalkerState
    {
        int32_t codecToken{-1};  //!< Current codec token for this batch
        int32_t talkerFrames{0}; //!< Number of audio frames generated so far
        bool finished{false};    //!< True when codec EOS or maxAudioLength reached
        bool talkerError{false}; //!< True on any Talker failure

        std::unordered_set<int32_t> seenTokenSet;   //!< Host-side seen tokens for repetition penalty
        int32_t numSeenTokens{0};                   //!< Count of unique seen tokens
        std::vector<std::vector<int32_t>> rvqCodes; //!< Generated RVQ codes [numFrames][numCodesPerFrame]
    };

    /*!
     * @brief Batched Talker generation loop (non-streaming)
     *
     * Runs batched Talker + CodePredictor decode loop. Talker decode steps use
     * batched engine execution; CodePredictor runs per-batch (each frame resets KV cache).
     * Follows the same pattern as Thinker's decode loop: finished batches idle until all done.
     *
     * @param states  Per-batch state (codecToken, finished, seenTokens, rvqCodes)
     * @param activeBatchSize  Number of active batches
     * @param maxFrames  Maximum audio frames per batch
     * @param talkerSamplingParams  Talker sampling params (batchSize = activeBatchSize)
     * @param predictorSamplingParams  CodePredictor sampling params (batchSize = 1)
     * @param repetitionPenalty  Repetition penalty factor
     * @param trailingTextHiddens  Per-batch trailing text hidden pointers (nullable entries)
     * @param stream  CUDA stream
     * @return True if generation succeeded
     */
    //! @param prefillSeqLens Per-batch prefill sequence lengths for correct hidden-state extraction
    //!        after batched prefill with padding. Empty for single-batch callers.
    bool runTalkerGenerationLoop(std::vector<PerBatchTalkerState>& states, int32_t activeBatchSize, int32_t maxFrames,
        SamplingParams const& talkerSamplingParams, SamplingParams const& predictorSamplingParams,
        float repetitionPenalty, std::vector<rt::Tensor const*> const& trailingTextHiddens, cudaStream_t stream,
        std::vector<int64_t> const& prefillSeqLens = {});

    /*!
     * @brief Run a single Talker decode frame (used by the Thinker-Talker streaming path).
     *
     * Single-frame variant of runTalkerGenerationLoop's inner step. Called from inside the
     * Thinker decode callback to interleave Talker frames with Thinker tokens on the same
     * CUDA stream. Operates on batch=1 internally.
     */
    bool runSingleTalkerDecodeFrame(int32_t& codecToken, SamplingParams const& talkerSamplingParams,
        SamplingParams const& predictorSamplingParams, rt::Tensor const* trailingPtr, int32_t frameIdx,
        std::unordered_set<int32_t>& seenTokenSet, int32_t& numSeenTokens, float repetitionPenalty,
        std::vector<std::vector<int32_t>>& rvqCodes, cudaStream_t stream);

    // ========== Segment Parsing and Prefill Construction ==========

    //! Parsed segment from chat-template tokenized text
    struct SegmentInfo
    {
        int64_t startPos;
        int64_t endPos;
        int32_t roleId;
    };

    /*!
     * @brief Parse segments, project through MLP, and build Talker prefill embeddings
     *
     * Shared by streaming and non-streaming Omni paths. Performs:
     * 1. Segment parsing by <|im_start|> positions
     * 2. text_projection MLP on all tokens
     * 3. hidden_projection on multimodal tokens in user segments (if available)
     * 4. Assemble user segments + restructured assistant preamble → mTalkerInputEmbeds
     * 5. Fill initial trailing text hidden states
     *
     * @param textTokenIds  Full token sequence (input + generated)
     * @param prefillEmbedPtr  Thinker layer-0 prefill embeddings (nullable for fallback)
     * @param prefillHiddenPtr  Thinker layer-14 prefill hidden states (nullable)
     * @param prefillLen  Number of prefill tokens
     * @param thinkerEmbedTable  Thinker embedding table for generated token lookup
     * @param speakerId  Speaker codec token ID
     * @param trailingTextHidden  Output buffer for trailing text hidden states
     * @param[out] trailingCount  Number of trailing tokens written
     * @param[out] outSeqLen  Total prefill sequence length
     * @param stream  CUDA stream
     * @return True on success
     */
    bool buildTalkerPrefillFromSegments(std::vector<int32_t> const& textTokenIds, rt::Tensor const* prefillEmbedPtr,
        rt::Tensor const* prefillHiddenPtr, int32_t prefillLen, rt::Tensor const& thinkerEmbedTable, int32_t speakerId,
        rt::Tensor& trailingTextHidden, int32_t& trailingCount, int64_t& outSeqLen, cudaStream_t stream);

    // ========== Configuration Structure ==========

    /*!
     * @brief Talker configuration parameters
     */
    struct TalkerConfig
    {
        // Model dimensions (read from config, not hardcoded)
        int32_t thinkerHiddenSize{};       //!< Thinker hidden dimension (read from config)
        int32_t talkerHiddenSize{};        //!< Talker hidden dimension (read from config)
        int32_t talkerVocabSize{};         //!< Talker vocabulary size (read from config)
        int32_t codePredictorHiddenSize{}; //!< CodePredictor hidden dimension (read from CodePredictor config)
        int32_t codebookSize{};            //!< Codebook vocabulary size per layer (read from config or hardcoded)
        int32_t numCodeGroups{};           //!< Number of codebook groups (Omni=16, TTS=32), from config.json
        int32_t maxSeqLen{};               //!< Maximum input sequence length from thinker (read from config)

        // TTS special tokens (from thinker vocab, projected through text_projection)
        int32_t ttsPadTokenId{}; //!< TTS pad token (151671)
        int32_t ttsBosTokenId{}; //!< TTS begin-of-sequence (151672)
        int32_t ttsEosTokenId{}; //!< TTS end-of-sequence (151673)

        // Codec special tokens (from talker vocab, used directly)
        int32_t codecNothinkId{};  //!< Codec no-think control token (2155)
        int32_t codecThinkId{};    //!< Codec think control token (CustomVoice + language path)
        int32_t codecThinkBosId{}; //!< Codec think begin-of-sequence (2156)
        int32_t codecThinkEosId{}; //!< Codec think end-of-sequence (2157)
        int32_t codecPadId{};      //!< Codec padding token (2148)
        int32_t codecBosId{};      //!< Codec begin-of-sequence (2149)
        int32_t codecEosId{};      //!< Codec end-of-sequence

        // Speaker configuration (read from config)
        int32_t defaultSpeakerId{}; //!< Default speaker ID (e.g., 2301 for f245)

        //!< CustomVoice language conditioning: map of lower-case language name -> codec token ID.
        //!< Empty when the model is not a CustomVoice language-conditioned variant.
        std::unordered_map<std::string, int32_t> codecLanguageId{};
    };

    // ========== Configuration and Initialization ==========

    /*!
     * @brief Validate and fill configuration from talker config file
     * @param talkerEngineDir Directory containing talker engine files
     * @return True on success, false on failure
     */
    bool validateAndFillConfig(std::string const& talkerEngineDir);

    /*!
     * @brief Initialize Talker and CodePredictor engine runners
     * @param talkerEngineDir Directory containing talker engine files
     * @param codePredictorEngineDir Directory containing code predictor engine files
     * @return True on success, false on failure
     */
    bool initializeEngineRunners(std::string const& talkerEngineDir, std::string const& codePredictorEngineDir);

    /*!
     * @brief Load CodePredictor lm_head weights and small_to_mtp_projection
     * @param codePredictorEngineDir Directory containing code predictor engine files
     * @return True on success, false on failure
     */
    bool loadCodePredictorWeights(std::string const& codePredictorEngineDir);

    /*!
     * @brief Allocate device buffers for Talker pipeline
     * @return True on success, false on failure
     */
    bool allocateBuffer();

    TalkerConfig mTalkerConfig{};                           //!< Talker configuration
    std::unordered_map<std::string, int32_t> mSpeakerIdMap; //!< Speaker name to ID mapping
    int32_t mMaxBatchSize{1}; //!< Maximum batch size from Talker engine config (min of Talker and CodePredictor)

    int32_t mNumRvqLayers{talker_constants::kDefaultNumRvqLayers};
    int32_t mNumCodesPerFrame{talker_constants::kDefaultNumRvqLayers + 1};

    std::unique_ptr<tokenizer::Tokenizer> mTokenizer; //!< Tokenizer for text-to-token-ID conversion

    std::unique_ptr<LLMEngineRunner> mTalkerLLMRunner;     //!< Talker LLM engine runner
    std::unique_ptr<LLMEngineRunner> mCodePredictorRunner; //!< CodePredictor engine runner

    LLMEngineRunnerConfig mTalkerLLMConfig;     //!< Talker LLM configuration
    LLMEngineRunnerConfig mCodePredictorConfig; //!< CodePredictor configuration

    //! Shared GPU execution context memory for Talker and CodePredictor (kUSER_MANAGED).
    rt::Tensor mSharedExecContextMemory;

    // cuBLAS handle removed — GEMM is now via CuTe DSL compiled kernels (CuteDslGemmRunner).

    // Projects text tokens from thinker embedding space (layer 0) to talker input space
    rt::Tensor mTextFC1Weight; //!< FC1 weight [thinkerHidden, thinkerHidden] FP16 column-major
    rt::Tensor mTextFC1Bias;   //!< FC1 bias [thinkerHidden] FP16
    rt::Tensor mTextFC2Weight; //!< FC2 weight [talkerHidden, thinkerHidden] FP16 column-major
    rt::Tensor mTextFC2Bias;   //!< FC2 bias [talkerHidden] FP16

    // Projects multimodal tokens from thinker hidden space (layer 14) to talker input space
    rt::Tensor mHiddenFC1Weight; //!< FC1 weight [thinkerHidden, thinkerHidden] FP16 column-major
    rt::Tensor mHiddenFC1Bias;   //!< FC1 bias [thinkerHidden] FP16
    rt::Tensor mHiddenFC2Weight; //!< FC2 weight [talkerHidden, thinkerHidden] FP16 column-major
    rt::Tensor mHiddenFC2Bias;   //!< FC2 bias [talkerHidden] FP16

    // Optional Talker-to-CodePredictor projection
    rt::Tensor mSmallToMtpWeight; //!< Linear weight [1024, 2048] FP16
    rt::Tensor mSmallToMtpBias;   //!< Linear bias [1024] FP16
    bool mUseSmallToMtpProjection{false};
    bool mIsOmni{false}; //!< True for Omni family checkpoints

    // ========== Embedding Tables ==========
    rt::Tensor mTextEmbeddingTable; //!< Text embedding table [thinkerVocabSize, thinkerHiddenSize] (for standalone TTS)
    rt::Tensor mTalkerEmbeddingTable; //!< Talker LLM embedding table [vocabSize, hiddenSize]
    std::vector<rt::Tensor>
        mCodePredictorEmbeddingTables; //!< CodePredictor embedding tables (mNumRvqLayers) [codebookSize, hiddenSize]

    // CodePredictor LM Heads (bound via setLmHeadWeight before each decode step)
    // ONNX has lm_head_weight as a dynamic input tensor, switched per RVQ layer
    std::vector<rt::Tensor>
        mCodePredictorLmHeadWeights; //!< CodePredictor lm_head weights (mNumRvqLayers) [vocabSize, hiddenSize]

    // TTS special token embeddings (initialized from thinker embedding table)
    // Initialized in constructor from Thinker embedding table
    rt::Tensor mTtsPadEmbed; //!< TTS pad embedding [talkerHiddenSize] FP16
    rt::Tensor mTtsBosEmbed; //!< TTS bos embedding [talkerHiddenSize] FP16
    rt::Tensor mTtsEosEmbed; //!< TTS eos embedding [talkerHiddenSize] FP16

    // ========== Workspace Tensors (allocated at maxBatchSize) ==========
    // Buffers used for per-batch prefill (not batched engine execution, reused per-batch)
    rt::Tensor mThinkerEmbedBuffer; //!< Text embedding output [maxSeqLen, thinkerHiddenSize] FP16
    rt::Tensor mGpuTokenIdsBuffer;  //!< Token IDs upload buffer [1, maxSeqLen] INT32
    rt::Tensor mMLPWorkspace;       //!< MLP intermediate results [maxSeqLen, thinkerHiddenSize] FP16
    rt::Tensor mProjectedBuffer;    //!< Projected tokens [maxSeqLen, talkerHiddenSize] FP16
    rt::Tensor mTalkerInputEmbeds;  //!< Talker input embeddings [maxBS, maxSeqLen, talkerHiddenSize] FP16
    rt::Tensor mSamplingWorkspace;  //!< Workspace for sampling operations

    // Talker LLM workspace (batched)
    rt::Tensor mTalkerLogits;            //!< Talker output logits [maxBS, vocabSize] FP32
    rt::Tensor mTalkerSelectedIndices;   //!< Selected token indices [maxBS, 1] INT32
    rt::Tensor mHostSelectedTokenIds;    //!< Host selected tokens [maxBS] INT32
    rt::Tensor mHostTalkerContextLength; //!< Host context lengths [maxBS] INT32
    rt::Tensor mSeenCodecTokensBuf;      //!< Per-batch seen codec tokens [maxBS, maxKVCacheCapacity] INT32

    // CodePredictor workspace (batch=1 for per-batch CodePredictor calls)
    rt::Tensor mCodePredictorLogits;                     //!< CodePredictor output logits [1, codebookSize] FP32
    std::vector<rt::Tensor> mCodePredictorLogitsPerHead; //!< Per-lm_head logits for CUDA graph capture
    bool mCodePredictorGraphsCaptured{false};
    rt::Tensor mCodePredictorSelectedIndices;   //!< Selected code indices [1, 1] INT32
    rt::Tensor mCodePredictorPrefillInput;      //!< Prefill input [1, 2, cpHidden] FP16
    rt::Tensor mCodePredictorCodecIds;          //!< Codec token IDs [1, 1] INT32
    rt::Tensor mCodePredictorCodecEmbed;        //!< Projected codec embed [1, 1, cpHidden] FP16
    rt::Tensor mRawCodecEmbed;                  //!< Raw codec embed [1, 1, talkerHidden] FP16
    rt::Tensor mSmallToMtpProjectedHidden;      //!< Projected talker hidden [1, cpHidden] FP16
    rt::Tensor mHostSelectedCodeIds;            //!< Host selected codes [1] INT32
    rt::Tensor mHostCodePredictorContextLength; //!< Host CodePredictor context length [1] INT32

    // Residual + decode buffers (batched for Talker, batch=1 for CodePredictor)
    rt::Tensor mResidualEmbedBuffer; //!< Residual embedding [maxBS, 1, talkerHidden] FP16
    rt::Tensor mTalkerDecodingIds;   //!< Talker decoding token IDs [maxBS, 1] INT32
    rt::Tensor mTalkerDecodingEmbed; //!< Talker decoding embedding [maxBS, 1, talkerHidden] FP16

    // KVCache reset helper
    rt::Tensor mHostReuseKVCacheLengths; //!< Host KVCache reuse lengths [maxBS] INT32

    // Generation loop workspace (batched for Talker, batch=1 for CodePredictor)
    rt::Tensor mTalkerHiddenStatesBuffer;        //!< Talker hidden states [maxBS, maxSeqLen, talkerHidden] FP16
    rt::Tensor mCodePredictorHiddenStatesBuffer; //!< CodePredictor hidden states [1, numCodesPerFrame, cpHidden] FP16
    rt::Tensor mTalkerLastHidden;                //!< Extracted last hidden [maxBS, talkerHidden] FP16
    rt::Tensor mCodecHiddensBuffer;              //!< Codec hiddens [1, numCodesPerFrame, talkerHidden] FP16

    cudaStream_t mStream{nullptr};                   //!< CUDA stream for operations
    metrics::MultimodalMetrics mMultimodalMetrics;   //!< Performance metrics for Talker pipeline (legacy)
    metrics::OmniTalkerMetrics mOmniTalkerMetrics;   //!< Omni-specific Talker metrics
    metrics::OmniLatencyMetrics mOmniLatencyMetrics; //!< Audio latency metrics (TTFA, TTFC, RTF)

    cudaEvent_t mTtfaStart{nullptr}; //!< TTFA start event (pipeline entry)
    cudaEvent_t mTtfaEnd{nullptr};   //!< TTFA end event (first codec token sampled)

    /*!
     * @brief Perform MLP projection from thinker embed to talker input space (non-streaming)
     *
     * Builds the complete non-streaming prefill buffer: 8 fixed prefix rows +
     * N text token rows + 2 suffix rows. Total outputSeqLen = seqLen + 2.
     *
     * @param thinkerEmbed Embedded token sequence [seqLen, thinkerHiddenSize]
     * @param speakerId Speaker ID for codec embedding
     * @param output Projected talker input embeddings [seqLen+2, talkerHiddenSize]
     * @param outputSeqLen seqLen + 2
     * @param stream CUDA stream
     * @return True on success, false on failure
     */
    bool projectToTalkerInput(rt::Tensor const& thinkerEmbed, int32_t speakerId, int32_t langId, rt::Tensor& output,
        int64_t& outputSeqLen, cudaStream_t stream);

    //! Embed token IDs, run MLP projection, and reshape buffers ready for Talker prefill.
    //! Populates mTalkerInputEmbeds and mTalkerHiddenStatesBuffer as side effects.
    //! \param[out] outSeqLen  seqLen + 2 (non-streaming prefill length)
    bool prepareTalkerInput(std::vector<int32_t> const& textTokenIds, TalkerGenerationRequest const& request,
        int64_t& outSeqLen, cudaStream_t stream);

    /*!
     * @brief Execute CodePredictor prefill step using CUDA Graph
     *
     * Performs prefill inference for one codebook layer using pre-captured CUDA Graph.
     * The graph already has the correct lm_head bound.
     *
     * @param codecTokenEmbeds Codec token embeddings [1, 2, hiddenSize] — concat([past_hidden, embed(code_0)])
     * @param generationStep Which lm_head/graph to use (0-14)
     * @param outputLogits Output logits [1, seqLen, codebookSize] (engine output)
     * @param outputHiddenStates Output hidden states for residual connection
     * @param stream CUDA stream
     * @return True on success, false on failure
     */
    bool executeCodePredictorPrefillStep(rt::Tensor const& codecTokenEmbeds, int32_t generationStep,
        rt::Tensor& outputLogits, rt::Tensor& outputHiddenStates, cudaStream_t stream);

    /*!
     * @brief Execute CodePredictor decoding step using CUDA Graph
     *
     * Performs single-step decoding for one codebook layer using pre-captured CUDA Graph.
     * The graph already has the correct lm_head bound.
     *
     * @param tokenId Current code token ID
     * @param embeddingTableIndex Which embedding table to use (0-14)
     * @param generationStep Which lm_head/graph to use (0-14)
     * @param outputLogits Output logits [1, 1, codebookSize] (engine output)
     * @param outputHiddenStates Output hidden states for next residual connection
     * @param stream CUDA stream
     * @return True on success, false on failure
     */
    bool executeCodePredictorDecodingStep(int32_t tokenId, int32_t embeddingTableIndex, int32_t generationStep,
        rt::Tensor& outputLogits, rt::Tensor& outputHiddenStates, cudaStream_t stream);

    /*!
     * @brief Load Talker weights from safetensors files
     *
     * Loads text_projection MLP weights, text embedding table, and Talker embedding table.
     *
     * @param weightsDir Directory containing weight files
     * @param stream CUDA stream
     * @return True on success, false on failure
     */
    bool loadTalkerWeights(std::string const& weightsDir, cudaStream_t stream);

    // ========== Incremental Trailing Hidden Helpers (for Thinker-Talker streaming) ==========

    /*!
     * @brief Project a single token through text_projection and write to trailingTextHidden
     *
     * Performs: embed_tokens(tokenId) → text_projection(embed) → trailingTextHidden[trailingIdx]
     * Uses pre-allocated mStreamingTokenId / mStreamingTokenEmbed / mStreamingMlpWork buffers
     * to avoid per-call cudaMalloc overhead.
     */
    void appendTrailingToken(int32_t tokenId, rt::Tensor const& thinkerEmbedTable, rt::Tensor& trailingTextHidden,
        int32_t trailingIdx, cudaStream_t stream);

    /*!
     * @brief Append tts_eos embedding at the end of trailingTextHidden
     */
    void finalizeTrailing(rt::Tensor& trailingTextHidden, int32_t trailingIdx, cudaStream_t stream);

    // Pre-allocated trailing text hidden buffer (shared by streaming and non-streaming Omni paths)
    // Non-streaming multi-batch: [maxBS, maxSeqLen+1, H] — each batch has its own trailing region
    // Streaming (batch=1): uses slot 0 only
    rt::Tensor mStreamingTrailingHidden; //!< [maxBS * (maxSeqLen+1), talkerHiddenSize] FP16 GPU

    // Pre-allocated gather/scatter index buffer for multimodal token projection
    rt::Tensor mGatherIndicesBuffer; //!< [maxSeqLen] INT32 GPU — indices for invokeGather/invokeScatter

    // Pre-allocated single-token workspace for appendTrailingToken (avoids per-call cudaMalloc)
    rt::Tensor mStreamingTokenId;    //!< [1, 1] INT32 GPU — single token ID upload buffer
    rt::Tensor mStreamingTokenEmbed; //!< [1, thinkerHiddenSize] FP16 GPU — embedding lookup result
    rt::Tensor mStreamingProjOut;    //!< [1, talkerHiddenSize] FP16 GPU — text_projection output
    rt::Tensor mStreamingMlpWork;    //!< [1, thinkerHiddenSize] FP16 GPU — MLP intermediate
};

} // namespace rt
} // namespace trt_edgellm
