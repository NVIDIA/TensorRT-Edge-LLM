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

#include "chatTemplate/chatTemplate.h"
#include "common/tensor.h"
#include "kernels/talkerMLPKernels/talkerMLPKernels.h"
#include "profiling/metrics.h"
#include "runtime/config/deploymentConfig.h"
#include "runtime/config/llmEngineConfig.h"
#include "runtime/exec/engineExecutor.h"
#include "runtime/exec/raggedBatchBuilder.h"
#include "runtime/exec/tensorMap.h"
#include "runtime/llmInferenceRuntime.h"
#include "runtime/llmRuntimeUtils.h"
#include "runtime/state/pipelineIO.h"
#include "runtime/state/sharedResources.h"
#include "tokenizer/tokenizer.h"
#include <cuda_fp16.h>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace trt_edgellm
{

// Forward declaration
struct SamplingParams;

namespace rt
{

class CloneEncoderRunner;

// ========== Constants ==========

namespace talker_constants
{
constexpr int32_t kDefaultNumRvqLayers = 15; //!< Default RVQ layers (Omni=15, TTS=31; auto-detected at runtime)
constexpr int32_t kAssistantPrefixLen = 3;   //!< Assistant prefix tokens ([:3])
constexpr int32_t kAssistantTrailingSuffix
    = 5; //!< Trailing tokens to strip from end of sequence ("<|im_end|>\n<|im_start|>assistant\n")
constexpr int32_t kNonStreamingPrefixRows = 8;     //!< Fixed prefix rows in non-streaming prefill (rows 0-7)
constexpr int32_t kPrefixRowsWithLanguage = 9;     //!< Prefix rows with CustomVoice language conditioning
constexpr int32_t kCodePredictorPrefillSeqLen = 2; //!< CodePredictor prefill sequence length
constexpr int32_t kCodecEmbeddingCount = 6;        //!< Number of codec embeddings to add

// CodePredictor sampling defaults. Qwen3-Omni dense/MoE hardcode
// ``code_predictor.generate(top_k=50, top_p=0.8)`` with the HF
// ``GenerationConfig`` temperature default (1.0):
//   https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_omni/modeling_qwen3_omni.py#L3408
//   https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_omni_moe/modeling_qwen3_omni_moe.py#L3184
// Qwen3-Omni Next routes the ``subtalker_*`` generate kwargs instead
// (defaults: temperature=0.9, top_k=50, top_p=1.0).
constexpr float kCPSamplingTemperature = 1.0f;
constexpr int32_t kCPSamplingTopK = 50;
constexpr float kCPSamplingTopP = 0.8f;
constexpr float kCPSamplingTemperatureNext = 0.9f;
constexpr float kCPSamplingTopPNext = 1.0f;
//! Upper bound on positions per CP speculative-decoding verify pass; bounds the SPD workspace.
constexpr int32_t kCpSpecMaxK = 8;
//! mCpRowIdx planes: draft hidden/head selectors, then verify hidden/head/table selectors.
constexpr int32_t kCpRowIdxPlanes = 5;

//! mCpStoreIdx planes: destination row in the codec hidden buffer, then table selector.
constexpr int32_t kCpStoreIdxPlanes = 2;

//! Philox seed for CP speculative decoding; the offset advances monotonically per draw so a
//! rerun with the same inputs reproduces the same accept decisions.
constexpr uint64_t kCpSpecSamplingSeed = 0x51ED270BULL;

// Audio output constants (Qwen3-Omni codec: 12.5 Hz frame rate, 24 kHz mono PCM output)
constexpr int32_t kAudioSampleRate = 24000;     //!< Output PCM sample rate (Hz)
constexpr int32_t kAudioSamplesPerFrame = 1920; //!< Samples produced per codec frame (24000 / 12.5)

// Default chat-template token IDs (Qwen3-Omni tokenizer). Variant-specific engines
// override via talker config.json; TalkerConfig picks up the actual values there.
constexpr int32_t kImStartTokenId = 151644;
constexpr int32_t kAssistantRoleId = 77091;
constexpr int32_t kUserRoleId = 872;
constexpr int32_t kSystemRoleId = 8948;
constexpr int32_t kAudioTokenId = 151675;
constexpr int32_t kImageTokenId = 151655;
constexpr int32_t kVideoTokenId = 151656;

// Qwen3-Next Omni tokenizer (vocab 248320).
constexpr int32_t kImStartTokenIdNext = 248045;
constexpr int32_t kImEndTokenIdNext = 248046;
constexpr int32_t kNlTokenIdNext = 198;
constexpr int32_t kDoubleNlTokenIdNext = 271; //!< "\n\n" — trails the template's empty think block
constexpr int32_t kThinkOpenTokenIdNext = 248068;
constexpr int32_t kThinkCloseTokenIdNext = 248069;
constexpr int32_t kAssistantRoleIdNext = 74455;
constexpr int32_t kUserRoleIdNext = 846;
constexpr int32_t kSystemRoleIdNext = 8678;
constexpr int32_t kAudioTokenIdNext = 248076;
constexpr int32_t kImageTokenIdNext = 248056;
constexpr int32_t kVideoTokenIdNext = 248057;

//! OmniNext Talker text-chunk size (talker_text_in_chunk_n).
constexpr int32_t kTextInChunkN = 4;
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
 *   - Similar to LLMInferenceRuntime, manages multiple LLM engines
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
     * @param checkpointDir HF/ModelOpt checkpoint used by checkpoint-backed Talker weights
     * @param stream CUDA stream for operations
     * @throws std::runtime_error on any initialization failure
     */
    //! @param cloneEncoderDir Optional directory with the voice-clone reference encoder
    //!        engines (speaker_encoder.engine / speech_tokenizer_encoder.engine, Base
    //!        checkpoints). Empty disables voice cloning.
    //! @param cpSpecVerifySize CodePredictor verify window: one committed RVQ depth plus
    //!        `cpSpecVerifySize - 1` drafted depths. 0 disables speculation and keeps the
    //!        autoregressive loop; 1 is rejected because it leaves no draft slot. The runtime
    //!        falls back to the autoregressive loop when the engine or checkpoint cannot
    //!        support the requested width.
    Qwen3OmniTTSRuntime(std::string const& talkerEngineDir, std::string const& codePredictorEngineDir,
        std::string const& tokenizerDir, std::string const& cloneEncoderDir, cudaStream_t stream,
        std::string const& checkpointDir = "", int32_t cpSpecVerifySize = 0);

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

        // CodePredictor (sub-talker) sampling parameters, independent from the Talker.
        // Sub-talker == CodePredictor (CP). Naming follows HF's ``subtalker_*`` generate kwargs.
        // 0 = use HF's hardcoded code_predictor.generate defaults (kCPSampling*: 1.0/50/0.8, or the
        // Next per-arch defaults); these do NOT inherit the talker* request values.
        float subtalkerTemperature{0}; //!< CP (sub-talker) temperature (0 = arch default)
        int32_t subtalkerTopK{0};      //!< CP (sub-talker) top-K (0 = kCPSamplingTopK)
        float subtalkerTopP{0};        //!< CP (sub-talker) top-P (0 = arch default)

        // Speaker selection (optional, defaults to config default)
        std::string speakerName{""}; //!< Speaker name (e.g., "f245", "m02") - empty means use default
        int32_t speakerId{-1};       //!< Speaker ID - if >= 0, overrides speakerName

        // ---- Voice conditioning ----
        // Two prefill paths consume disjoint subsets of these fields, dispatched by
        // isOmniNext(): the legacy Qwen3-TTS path (prepareTalkerInput) uses
        // languageName / instructText / refAudioPath; the Qwen3-Omni Next path
        // (buildQwen3OmniNextTalkerPrefill) uses talkerLanguage / assistantInstruct /
        // promptSpeakerCodes / systemInstruct. CodePredictor sampling (subtalker*
        // above) is shared by both.

        //!< [Qwen3-TTS] CustomVoice language conditioning (optional). Empty or "auto" keeps the
        //!< historical no-language prefill. A known language name (e.g. "chinese", "english";
        //!< matched case-insensitively against the engine config codec_language_id map) switches
        //!< the Talker prefill to the 9-row language-conditioned layout. Unknown names fall back
        //!< to no-language with a warning. Dialect speakers (spk_is_dialect in config) override
        //!< this automatically when language is "auto" or "chinese", matching the PyTorch reference.
        std::string languageName{""};

        //!< [Qwen3-TTS] CustomVoice/VoiceDesign instruction control (optional). Natural-language
        //!< style instruction (e.g. "Speak in a whisper"). Wrapped as a user turn, projected
        //!< through text_projection, and prepended to the Talker prefill. Empty = no instruction.
        std::string instructText{""};

        //!< [Qwen3-TTS] Voice clone (Base checkpoints, optional): reference audio file (wav/mp3/flac).
        //!< The reference encoders run on-device (requires cloneEncoderDir at construction).
        //!< With refText set, ICL mode conditions on (transcript, codec codes) of the
        //!< reference; without it, x-vector-only mode clones timbre alone.
        std::string refAudioPath{""};
        std::string refText{""}; //!< Reference transcript (enables ICL mode)

        //! [Qwen3-Omni Next] Custom voice: reference-voice codec codes, [frames][num_code_groups].
        //! When non-empty, replaces the built-in speaker codec rows and skips the
        //! per-speaker system prompt (HF ``prompt_speaker_codes`` semantics).
        std::vector<std::vector<int32_t>> promptSpeakerCodes;

        //! [Qwen3-Omni Next] Style/emotion instruction name resolved via config
        //! ``talker_assistant_prompt_id_mapping`` (e.g. "cheerful"); empty = none.
        std::string assistantInstruct{""};
        //! [Qwen3-Omni Next] Target language resolved via config ``talker_language_id``
        //! (e.g. "chinese"); empty or "auto" = model decides (codec_nothink path).
        std::string talkerLanguage{""};
        //! [Qwen3-Omni Next] Free-text system instruction inserted after the system role trio
        //! (HF ``talker_system_instruct_ids``); tokenized by the runtime. Empty = none.
        std::string systemInstruct{""};

        // Input: conversation messages for this request (runtime tokenizes internally)
        std::vector<Message> messages;
        bool applyChatTemplate{true};   //!< Whether to apply chat template formatting
        bool addGenerationPrompt{true}; //!< Whether to add generation prompt at the end
        bool enableThinking{false};     //!< Whether to enable thinking mode

        //!< Optional streaming: emit RVQ codes via callback every N frames.
        //!< 0 = non-streaming (default); >0 enables streaming for this request.
        int32_t streamingChunkFrames{0};

        //!< Streaming callback invoked from the Talker generation thread with:
        //!<   chunkRvqCodes: codes generated since the last callback ([frames][numCodesPerFrame])
        //!<   isFinal: true on the last callback for this request (post-EOS / post-maxFrames flush);
        //!<            invoked exactly once with isFinal=true per request that has streaming enabled,
        //!<            even if chunkRvqCodes is empty (signals end-of-stream).
        //!< Each request gets its own callback; per-batch streams are fully independent.
        std::function<void(std::vector<std::vector<int32_t>> const& chunkRvqCodes, bool isFinal)> onChunkReady;
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
     * @brief Hidden-state layer indices the Talker reads from the Thinker
     *        portal. Layer 0 is the input embedding; the second index is the
     *        decoder layer whose pre-norm hidden_states the Talker consumes,
     *        sourced from the Talker config's ``accept_hidden_layer`` field.
     *        Must match the layer the Thinker engine was exported to emit on
     *        its ``hidden_states`` output.
     */
    std::vector<int32_t> getThinkerHiddenLayerIndices() const
    {
        return {0, mTalkerConfig.acceptHiddenLayer};
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
     * LLMInferenceRuntime::handleRequest() for standard LLM inference.
     *
     * @note Sampling parameters (temperature, topK, topP, repetitionPenalty) are taken
     *       from requests[0] and applied uniformly to all batches. This matches
     *       LLMInferenceRuntime's design where SamplingParams is shared across the batch.
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
     *       LLMInferenceRuntime's design where SamplingParams is shared across the batch.
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
     *
     * Callback contract matches the TTS streaming path (see TalkerGenerationRequest::onChunkReady):
     *   chunkRvqCodes: codes since the last callback ([frames][numCodesPerFrame])
     *   isFinal: true on the last callback for this request (post-EOS / post-maxAudioLength flush);
     *            invoked exactly once with isFinal=true when streaming is enabled, even if the chunk
     *            is empty (acts as the end-of-stream signal).
     */
    using AudioChunkCallback
        = std::function<void(std::vector<std::vector<int32_t>> const& chunkRvqCodes, bool isFinal)>;

    struct ThinkerTalkerStreamingConfig
    {
        int32_t talkerPrefillThreshold{4};    //!< Start Talker prefill after this many assistant tokens
        int32_t codecChunkFrames{0};          //!< Emit chunk every N frames (0 = disabled)
        AudioChunkCallback onAudioChunkReady; //!< Per-chunk callback (see AudioChunkCallback contract)
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
    bool handleStreamingGeneration(LLMInferenceRuntime& thinkerRuntime, LLMGenerationRequest& thinkerRequest,
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
     * @brief Capture CUDA graphs for decoding steps (same pattern as LLMInferenceRuntime).
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

    /*!
     * @brief Resolve the language codec token ID for a request (CustomVoice language conditioning)
     *
     * Mirrors the PyTorch reference (modeling_qwen3_tts.py):
     *   1. Empty / "auto" language → -1 (no-language path), unless step 3 overrides.
     *   2. Known language name (case-insensitive lookup in codecLanguageIdMap) → its codec ID.
     *      Unknown names log a warning and fall back to -1.
     *   3. Dialect override: when the resolved language is "auto" or "chinese" and the speaker
     *      is a dialect speaker (spkDialectMap), the dialect's codec ID wins.
     *
     * Always returns -1 when the engine config has no codec_think_id / codec_language_id
     * (e.g. Qwen3-Omni checkpoints or engines exported before language support).
     *
     * @param languageName Request language name ("" = auto)
     * @param speakerName  Request speaker name (for dialect override; "" = default speaker)
     * @return Language codec token ID, or -1 for the no-language 8-row prefill
     */
    int32_t resolveLanguageId(std::string const& languageName, std::string const& speakerName) const;

    //! Speaker names accepted by TalkerGenerationRequest::speakerName.
    std::vector<std::string> getSpeakerNames() const
    {
        std::vector<std::string> names;
        names.reserve(mSpeakerIdMap.size());
        for (auto const& entry : mSpeakerIdMap)
        {
            names.push_back(entry.first);
        }
        return names;
    }

private:
    // ========== Internal Methods ==========

    void initializeTTSEmbeddings(cudaStream_t stream);

    //! @param perBatchContextLengths Optional per-batch context lengths for padded batched prefill.
    //!        If empty, all batches use the full seqLen dimension of inputEmbeds.
    bool executeTalkerPrefillStep(rt::Tensor const& inputEmbeds, rt::Tensor& outputLogits,
        rt::Tensor& outputHiddenStates, cudaStream_t stream, std::vector<int64_t> const& perBatchContextLengths = {});

    //! Run a single Talker vanilla decoding step with the shared ragged decoder ABI.
    //! inputEmbeds shape must be [batch, 1, talkerHiddenSize]; outputLogits is auto-reshaped to [batch, vocab].
    bool executeTalkerDecodingStep(
        rt::Tensor const& inputEmbeds, rt::Tensor& outputLogits, rt::Tensor& outputHiddenStates, cudaStream_t stream);

    //! Run CodePredictor for one frame across all activeBatchSize batch elements in a single engine call.
    //! Batch dim is implicit in the input tensor shapes; bs=1 is just a special case of bs=N where N=1
    //! (same pattern as Talker / spec decode runtime).
    //! @param activeBatchSize          Number of active batches (1..maxBatchSize).
    //! @param codecTokensPerBatch     [activeBS] code_0 from Talker for each batch.
    //! @param talkerLastHiddenBatched [activeBS, talkerHidden] per-batch Talker last hidden.
    //! @param outputCodesPerBatch     [activeBS][mNumCodesPerFrame] generated codes per batch.
    bool runCodePredictorGenerationForFrame(int32_t activeBatchSize, std::vector<int32_t> const& codecTokensPerBatch,
        rt::Tensor const& talkerLastHiddenBatched, SamplingParams const& samplingParams,
        std::vector<std::vector<int32_t>>& outputCodesPerBatch, cudaStream_t stream);

    //! Chain speculative-decoding replacement for the per-frame CP decode loop.
    //!
    //! Entered after the shared prefill, with code_1 already appended and the prefill hidden
    //! in mCodePredictorHiddenStatesBuffer. Produces code_2..code_(mNumRvqLayers) with the
    //! same output distribution as the AR loop: drafts come from reusing lm_head[m+t-1] on
    //! the hidden that produced code_m, and cpSpecProbabilisticAccept emits exact samples
    //! from the CP's own truncated distribution.
    bool runCodePredictorSpecDecodeLoop(int32_t activeBatchSize, SamplingParams const& samplingParams,
        std::vector<std::vector<int32_t>>& outputCodesPerBatch, cudaStream_t stream);

    //! Fill mCpVerifyInput with the n+1 projected codec embeddings of a verify pass.
    //!
    //! Position 0 carries the last finalised code, positions 1..n carry the drafts; position i
    //! reads codec table (m - 1 + i), matching the AR step that feeds E_{step-2}[c_(step-1)].
    bool buildCpVerifyInput(int32_t activeBatchSize, int32_t const* hostM, int32_t n, cudaStream_t stream);

    //! Store the raw codec embeddings of newly finalised codes into mCodecHiddensBuffer.
    //!
    //! computeResidualConnection fills slots 0 and mNumRvqLayers itself and reads slots
    //! 1..mNumRvqLayers-1 from here, so speculative decoding must populate them once the
    //! accept result makes a code final.
    //! @param codeIds   [batchSize, stride] device ids; row b element i holds code (hostM[b] + mOffset + i).
    //! @param counts     Per-batch element count, or nullptr for one element per batch.
    bool storeCodecHiddensForAcceptedCodes(rt::Tensor const& codeIds, int32_t batchSize, int32_t stride,
        int32_t const* hostM, int32_t mOffset, cudaStream_t stream, int32_t const* counts = nullptr);

    //! Fill the mCpRowIdx planes for one round and upload them in a single H2D.
    bool uploadCpRowSelectors(int32_t activeBatchSize, int32_t const* hostM, int32_t n, cudaStream_t stream);

    //! Score \p rows hidden rows, each against the lm_head its selector plane names.
    bool applyRowLmHeads(
        __half const* hiddens, int32_t hiddenPlane, int32_t headPlane, int32_t rows, cudaStream_t stream);

    //! Append \p seqLen positions to the CP KV cache and return every position's hidden.
    //!
    //! Chunked-prefill semantics: unlike executeCodePredictorPrefillStep this does NOT reset
    //! the KV cache, and it does NOT commit the sequence length — the caller commits the
    //! accepted length instead. The engine's own logits output is unused; per-position
    //! logits are produced by applyRowLmHeads from \p outputHiddenStates.
    bool executeCodePredictorVerifyStep(rt::Tensor const& inputsEmbeds, int32_t const* baseLens,
        int32_t activeBatchSize, rt::Tensor& outputHiddenStates, cudaStream_t stream);

    //! Compute residual connection for one batch element.
    //! @param codecHiddensThisBatch  Per-batch view into mCodecHiddensBuffer: [1, mNumCodesPerFrame, talkerH].
    bool computeResidualConnection(rt::Tensor const& codecHiddensThisBatch, std::vector<int32_t> const& codes,
        rt::Tensor const* trailingTextHidden, int32_t generationStep, rt::Tensor& outputResidual, cudaStream_t stream);

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
    //! Per-batch streaming handler — emits codes via callback every chunkFrames as the loop runs.
    //! Empty chunkFrames or null onChunk disables streaming for that batch.
    //! When set, the loop guarantees exactly one isFinal=true callback per such batch on exit.
    struct PerBatchStreamingHandler
    {
        int32_t chunkFrames{0};
        std::function<void(std::vector<std::vector<int32_t>> const& chunkRvqCodes, bool isFinal)> onChunk;
    };

    //! @param prefillSeqLens Per-batch prefill sequence lengths for correct hidden-state extraction
    //!        after batched prefill with padding. Empty for single-batch callers. Mutable: the
    //!        Qwen3.5 chunked re-prefill path grows this in-place after each re-prefill so the
    //!        next iteration's extractTalkerLastHidden sees the updated cumulative seqLen.
    //! @param streamingHandlers Optional per-batch streaming chunk emitters. Empty disables streaming
    //!        globally; otherwise must be sized to activeBatchSize (per-batch entries can still be
    //!        no-ops via chunkFrames==0 / null onChunk).
    bool runTalkerGenerationLoop(std::vector<PerBatchTalkerState>& states, int32_t activeBatchSize, int32_t maxFrames,
        SamplingParams const& talkerSamplingParams, SamplingParams const& predictorSamplingParams,
        float repetitionPenalty, std::vector<rt::Tensor const*> const& trailingTextHiddens, cudaStream_t stream,
        std::vector<int64_t>& prefillSeqLens, std::vector<PerBatchStreamingHandler> const& streamingHandlers = {});

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

    //! Assemble the Qwen3-Next Omni Talker prefill (system + user + assistant parts) into
    //! ``mTalkerInputEmbeds``. Mirrors HF ``_get_talker_{system,user,assistant}_parts``. Kept
    //! separate from ``buildTalkerPrefillFromSegments`` because the row sequence, projection
    //! kernels, and assistant handoff all differ.
    bool buildQwen3OmniNextTalkerPrefill(std::vector<int32_t> const& textTokenIds, rt::Tensor const* prefillHiddenPtr,
        int32_t prefillLen, int32_t speakerId, rt::Tensor& trailingTextHidden, int32_t& trailingCount,
        int64_t& outSeqLen, cudaStream_t stream, std::vector<std::vector<int32_t>> const* promptSpeakerCodes = nullptr,
        std::string const& assistantInstruct = "", std::string const& talkerLanguage = "",
        std::vector<int32_t> const* systemInstructIds = nullptr);

    //! Run one Qwen3-Next Omni chunked re-prefill round: append ``codecEmbedFrames`` codec-sum
    //! rows for the previous call's frames, then up to ``chunkTokensPerCall`` text rows (plus
    //! optional trailing tts_eos), then re-execute the prefill and sample the first codec token
    //! of the new call. Consumes ``mQwen3OmniNextChunkStates[batchIdx]``.
    bool reprefillQwen3OmniNextChunk(int32_t batchIdx, std::vector<std::vector<int32_t>> const& rvqCodes,
        int32_t& outFirstCodecTok, SamplingParams const& talkerSamplingParams, float repetitionPenalty,
        int32_t& numSeenTokens, std::unordered_set<int32_t>& seenTokenSet, cudaStream_t stream);

    //! Fire re-prefill for any batch that has accumulated ``framesPerCall`` new frames. Called
    //! once per iteration of ``runTalkerGenerationLoop``. No-op for legacy Qwen3-Omni engines
    //! (their chunk states stay inactive). Returns false on unrecoverable batch failure.
    bool driveOmniNextChunkReprefills(std::vector<PerBatchTalkerState>& states, int32_t activeBatchSize,
        int32_t globalFrame, SamplingParams const& talkerSamplingParams, float repetitionPenalty, int32_t& unfinished,
        cudaStream_t stream);

    //! Single-batch re-prefill trigger. Returns 0 (no action — inactive or not enough frames),
    //! 1 (re-prefilled and updated ``codecTokenInOut``), -1 (reprefill failed).
    int32_t maybeReprefillOmniNextChunkForBatch(int32_t batchIdx, std::vector<std::vector<int32_t>> const& rvqCodes,
        int32_t& codecTokenInOut, SamplingParams const& talkerSamplingParams, float repetitionPenalty,
        int32_t& numSeenTokens, std::unordered_set<int32_t>& seenTokenSet, cudaStream_t stream);

    //! Append a Thinker-emitted token to the OmniNext chunk stream's ``remainingTextTokens`` so
    //! the next re-prefill picks it up. Used by Thinker-Talker streaming instead of the legacy
    //! per-decode-step trailing-text injection.
    void appendOmniNextChunkStreamToken(int32_t batchIdx, int32_t tokenId);

    //! Mark the OmniNext chunk stream as awaiting a final tts_eos row. Called when the Thinker
    //! signals its last token so the trailing tts_eos is emitted by the next re-prefill.
    void finalizeOmniNextChunkStream(int32_t batchIdx);

    //! Force ``mTalkerLogits[batchIdx, codecEosId] = -INF`` before sampling. Enforces HF's
    //! ``min_new_tokens = chunk_m + 1`` during non-last chunked-streaming calls.
    void suppressTalkerEosLogit(int32_t batchIdx, int32_t batchVocabSize, cudaStream_t stream);
    void trackSeenToken(std::unordered_set<int32_t>& seenSet, int32_t& numSeen, int32_t batchIdx, int32_t token,
        int32_t const* tokenDev, cudaStream_t stream);

    //! Copy one embedding table row into ``dstBase[dstRow]``. Fails on OOB tokenId.
    bool copyEmbedRow(rt::Tensor const& table, int32_t tokenId, __half* dstBase, int64_t dstRow, cudaStream_t stream);

    //! Copy an ``[hiddenSize]`` FP16 tensor into ``dstBase[dstRow]``.
    void copyRawRow(rt::Tensor const& src, __half* dstBase, int64_t dstRow, cudaStream_t stream);

    //! Fill codec pointer/vocab tables once after weights load.
    bool buildCodecEmbedPointerTable(cudaStream_t stream);

    //! Write one per-position speaker-codec sum row via ``invokeSpeakerCodecSum``.
    void sumSpeakerCodecRow(int64_t const* hostCodes, __half* dstBase, int64_t dstRow, cudaStream_t stream);

    //! Project ``prefillHidden[srcRow]`` through the single Linear ``mHiddenProjLinear*`` into
    //! ``dstBase[dstRow]``.
    void projectHiddenRow(
        rt::Tensor const& prefillHidden, int64_t srcRow, __half* dstBase, int64_t dstRow, cudaStream_t stream);

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
        int32_t codecNothinkId{}; //!< Codec no-think control token (2155)
        //! Codec think control token (OmniNext: 4202; Qwen3-TTS CustomVoice: 2154). Used
        //! instead of codecNothinkId when language conditioning is active; sentinel -1 means
        //! the checkpoint has no think token and language conditioning stays disabled.
        int32_t codecThinkId{-1};
        int32_t codecThinkBosId{}; //!< Codec think begin-of-sequence (2156)
        int32_t codecThinkEosId{}; //!< Codec think end-of-sequence (2157)
        int32_t codecPadId{};      //!< Codec padding token (2148)
        int32_t codecBosId{};      //!< Codec begin-of-sequence (2149)
        int32_t codecEosId{};      //!< Codec end-of-sequence

        //! Language name (lower-case) → codec token ID map from config `codec_language_id`
        //! (falls back to `talker_language_id` at export time). Empty for checkpoints without
        //! language conditioning (e.g. the dense Qwen3-Omni checkpoint) — resolveLanguageId then always returns -1.
        std::unordered_map<std::string, int32_t> codecLanguageIdMap;

        //! Speaker name (lower-case) → dialect language name from config `spk_is_dialect`.
        //! Only speakers whose value is a dialect string are present (e.g. eric → sichuan_dialect,
        //! dylan → beijing_dialect); non-dialect speakers (JSON false) are omitted.
        std::unordered_map<std::string, std::string> spkDialectMap;

        //! Checkpoint family from config `tts_model_type`: "custom_voice" (preset speakers),
        //! "voice_design" (no speaker row, instruction-driven), "base" (voice clone), or ""
        //! (Qwen3-Omni / legacy configs).
        std::string ttsModelType;

        // Speaker configuration (read from config)
        int32_t defaultSpeakerId{}; //!< Default speaker ID (e.g., 2301 for f245)

        //! Decoder layer the Talker consumes Thinker hidden_states from. Must match the
        //! Thinker engine's exported hidden_states layer. -1 = unconfigured (TTS-only).
        int32_t acceptHiddenLayer{-1};

        //! Chat-template / placeholder token IDs. Filled from talker config.json;
        //! defaults are Qwen3-Omni values, patched to ``kXxxNext`` by loadTalkerWeights
        //! for OmniNext engines whose config predates the export-side field write.
        int32_t imStartTokenId{talker_constants::kImStartTokenId};
        int32_t assistantRoleId{talker_constants::kAssistantRoleId};
        int32_t userRoleId{talker_constants::kUserRoleId};
        int32_t systemRoleId{talker_constants::kSystemRoleId};
        int32_t audioTokenId{talker_constants::kAudioTokenId};
        int32_t imageTokenId{talker_constants::kImageTokenId};
        int32_t videoTokenId{talker_constants::kVideoTokenId};

        //! <think>/</think> IDs (Qwen3-Omni: 151648/151649; OmniNext: 248068/248069).
        int32_t thinkOpenTokenId{151648};
        int32_t thinkCloseTokenId{151649};

        //! PT _get_talker_user_parts subsamples mm positions to at most this many
        //! via torch.linspace(0,N-1,M).long(). 0 disables.
        int32_t maxThinkerToTalkerMmTokens{16};

        //! HF ``talker_suppressed_tokens`` range start ``[vocab_size - K, vocab_size)``.
        //! K is 1024 for Qwen3-Omni, 3072 for OmniNext; ``loadTalkerWeights`` fills the
        //! final value once the variant is known.
        int32_t talkerSuppressStart{};
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
    bool initializeEngineRunners(std::string const& talkerEngineDir, std::string const& codePredictorEngineDir,
        std::string const& checkpointDir);

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
    //! Friendly-name → internal speaker-name aliases from voice_map.json (keys lowercased).
    std::unordered_map<std::string, std::string> mVoiceAliasMap;
    //! Style/emotion instruction name → token ids (config ``talker_assistant_prompt_id_mapping``).
    std::unordered_map<std::string, std::vector<int32_t>> mAssistantPromptIds;
    //! Language name → codec-side language id (config ``talker_language_id``).
    std::unordered_map<std::string, int32_t> mLanguageIds;

    //! Qwen3-Next Omni speaker_system_prompt_id map (speaker_id → text-token-list). HF emits these
    //! tokens between [im_start, system, nl] and the codec_bos when building the Talker system
    //! prefill section (modeling_qwen3_omni_next.py:_get_talker_system_parts).
    std::unordered_map<int32_t, std::vector<int32_t>> mSpeakerSystemPromptIds;

    // Descriptor-driven prefill assembly (instruction / VoiceDesign / voice-clone layouts).
    // Rows are queued host-side as (srcA, optional srcB) device-pointer pairs and assembled
    // in one invokePrefillRowAssemble launch; per-row math matches the fused
    // invokeAssistantPreamble kernel bit-exactly (see talkerMLPKernelTests).
    std::vector<kernel::PrefillRowDesc> mPrefillRows; //!< Queued row descriptors (reused per prefill)
    rt::Tensor mPrefillDescsHost;                     //!< Pinned host staging for descriptors
    rt::Tensor mPrefillDescsDevice;                   //!< Device descriptor buffer

    //! Talker codec-embedding-table row pointer for a token ID.
    half const* talkerEmbRow(int32_t tokenId) const;
    //! Queue one prefill row: srcA (+ srcB when non-null), both device pointers to [H] rows.
    void pushPrefillRow(half const* srcA, half const* srcB);
    //! Upload queued descriptors and assemble rows into output[0..numRows). Returns numRows.
    int64_t flushPrefillRows(rt::Tensor& output, cudaStream_t stream);

    // Voice clone workspace (loaded per request from voiceClonePromptPath)
    rt::Tensor mVoiceCloneXVector; //!< [talkerH] FP16 GPU x-vector
    rt::Tensor mIclFrameSumBuffer; //!< [maxRefFrames, talkerH] FP16 GPU summed codec embeddings
    rt::Tensor mIclTablePtrsGpu;   //!< [numGroups] device pointer array for sum kernel
    //! Pinned staging for the above; a member because the upload outlives the call that issues it.
    rt::Tensor mIclTablePtrsHost;
    std::vector<int32_t> mIclRefTextIds; //!< Host reference transcript token IDs (assistant-wrapped)

    std::unique_ptr<CloneEncoderRunner> mCloneEncoders; //!< Reference encoders (null unless cloneEncoderDir given)

    //! Run the reference encoders on refAudioPath and fill the clone workspace. Returns false
    //! on error; iclFrames receives the reference frame count (0 = x-vector-only mode).
    bool encodeVoiceCloneReference(
        std::string const& refAudioPath, std::string const& refText, int32_t& iclFrames, cudaStream_t stream);
    int32_t mMaxBatchSize{1}; //!< Maximum batch size from Talker engine config (min of Talker and CodePredictor)

    int32_t mNumRvqLayers{talker_constants::kDefaultNumRvqLayers};
    int32_t mNumCodesPerFrame{talker_constants::kDefaultNumRvqLayers + 1};

    std::unique_ptr<tokenizer::Tokenizer> mTokenizer;           //!< Tokenizer for text-to-token-ID conversion
    std::unique_ptr<chat_template::ChatTemplate> mChatTemplate; //!< Model-owned prompt renderer

    // Talker engine — migrated to EngineExecutor + supporting state
    LLMEngineConfig mTalkerLLMConfig;                  //!< Talker LLM configuration (parsed from config.json)
    std::unique_ptr<EngineExecutor> mTalkerExec;       //!< Talker engine executor
    std::unique_ptr<SharedResources> mTalkerSharedRes; //!< Talker cache managers + RoPE pool + zero buffer
    std::unique_ptr<PipelineIO> mTalkerPipelineIO;     //!< Talker per-step pipeline buffers
    TensorMap mTalkerTensorMap;                        //!< Talker engine binding map (set once, mutated per-step)
    RaggedExecutionBatch mTalkerRaggedBatch;           //!< Reused Talker ragged-metadata scratch
    std::vector<int32_t> mTalkerQueryLengths;          //!< Active Talker query lengths on the host
    std::vector<int32_t> mTalkerPastLengths;           //!< Committed Talker cache lengths on the host

    // CodePredictor engine — migrated to EngineExecutor + supporting state
    LLMEngineConfig mCodePredictorConfig;                     //!< CodePredictor LLM configuration
    std::unique_ptr<EngineExecutor> mCodePredictorExec;       //!< CodePredictor engine executor
    std::unique_ptr<SharedResources> mCodePredictorSharedRes; //!< CodePredictor cache + RoPE + zero buffer
    std::unique_ptr<PipelineIO> mCodePredictorPipelineIO;     //!< CodePredictor per-step pipeline buffers
    TensorMap mCodePredictorTensorMap;                        //!< CodePredictor engine binding map (step-invariant)
    RaggedExecutionBatch mCodePredictorRaggedBatch;           //!< Reused CodePredictor ragged-metadata scratch
    std::vector<int32_t> mCodePredictorQueryLengths;          //!< Active CodePredictor query lengths on the host
    std::vector<int32_t> mCodePredictorPastLengths;           //!< Committed CodePredictor cache lengths on the host

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

    //! Talker family. Filled by ``loadTalkerWeights`` from the engine config's
    //! ``model_type``; enum (not bool) leaves room for a third variant.
    //! Runtime differences: Omni projects text/mm via 2-layer MLPs; OmniNext
    //! uses direct text embed lookup + a single-Linear ``hidden_projection``.
    enum class TalkerVariant
    {
        Omni,
        OmniNext
    };
    TalkerVariant mTalkerVariant{TalkerVariant::Omni};
    //! Whether the talker engine emits hidden_states for the last token only rather than
    //! for the whole sequence. Probed from the engine's declared shape at load time.
    bool mTalkerHiddenIsGathered{false};

    //! Sequence extent the talker hidden-states buffer needs for a prefill of `seqLen`.
    int64_t talkerHiddenSeqDim(int64_t seqLen) const
    {
        return mTalkerHiddenIsGathered ? 1 : seqLen;
    }

    bool isOmniNext() const noexcept
    {
        return mTalkerVariant == TalkerVariant::OmniNext;
    }
    rt::Tensor mHiddenProjLinearWeight; //!< OmniNext: single-Linear weight [talkerHidden, thinkerHidden] FP16
    rt::Tensor mHiddenProjLinearBias;   //!< OmniNext: single-Linear bias [talkerHidden] FP16
    //! OmniNext Talker codec embed table [codecVocab, talkerHidden] (distinct from
    //! mTextEmbeddingTable, which is text-vocab). From codec_embedding.safetensors.
    rt::Tensor mTalkerCodecEmbedTable;
    //! OmniNext per-speaker codec template [maxSpeakerNum, numCodeGroups, speakerEmbedLen] INT64
    //! (padded with -1). From speaker_codec_embeddings.safetensors.
    rt::Tensor mSpeakerCodecEmbeddings;

    //! invokeSpeakerCodecSum workspace (INT8 tensor carries `__half const*[]`).
    rt::Tensor mCodecEmbPtrTable;
    rt::Tensor mCodecEmbVocabSizes;
    rt::Tensor mCodecRowCodes;

    //! OmniNext chunked-streaming per-batch state. HF ``generate_talker`` interleaves
    //! ``chunkTokensPerCall`` text tokens with ``framesPerCall`` codec frames via
    //! repeated KV-reset prefills; pure autoregressive decode past chunk 0 degenerates.
    //! Entries stay inactive for legacy Qwen3-Omni engines.
    struct Qwen3OmniNextChunkStreamState
    {
        bool active{false};
        std::vector<int32_t> remainingTextTokens; //!< Token IDs for chunks[1..].
        bool hasTrailingTtsEos{false};            //!< tts_eos row still owed after all text chunks.
        int32_t chunkTokensPerCall{4};            //!< talker_text_in_chunk_n
        int32_t framesPerCall{4};                 //!< talker_codec_output_chunk_m. HF runs m+1 forwards per
                                                  //!< forced call but its lagged-CP design emits only m frames;
                                                  //!< the (m+1)-th sample is a discarded lookahead. Emitting it
                                                  //!< (the old {5}) rendered one text-starved orphan frame per
                                                  //!< chunk into the audio.
        int32_t codecEmbedFrames{4};              //!< frames fed back as codec_embeds (last frame is lookahead-only)
        int64_t cumulativeSeqLen{0};              //!< Rows written into mTalkerInputEmbeds so far.
        int32_t cursorToken{0};                   //!< Next index in remainingTextTokens.
        int32_t framesSinceLastPrefill{0};
        int32_t firstFrameOfCallIdx{0}; //!< rvqCodes index of the first frame of the current call.
    };
    std::vector<Qwen3OmniNextChunkStreamState> mQwen3OmniNextChunkStates;

    // ========== Embedding Tables ==========
    rt::Tensor mTextEmbeddingTable; //!< Text embedding table [thinkerVocabSize, thinkerHiddenSize] (for standalone TTS)
    rt::Tensor mTalkerEmbeddingTable; //!< Talker LLM embedding table [vocabSize, hiddenSize]
    std::vector<rt::Tensor>
        mCodePredictorEmbeddingTables; //!< CodePredictor embedding tables (mNumRvqLayers) [codebookSize, hiddenSize]

    // CodePredictor lm_heads stacked [mNumRvqLayers, vocabSize, hiddenSize]; the engine
    // gathers the active head by the device lm_head_idx, so bindings stay step-invariant.
    rt::Tensor mCodePredictorLmHeads;
    rt::Tensor mCpLmHeadIdx; //!< Device head index [1] INT32 (0 = prefill, then 1..mNumRvqLayers-1)

    // TTS special token embeddings (initialized from thinker embedding table)
    // Initialized in constructor from Thinker embedding table
    rt::Tensor mTtsPadEmbed; //!< TTS pad embedding [talkerHiddenSize] FP16
    rt::Tensor mTtsBosEmbed; //!< TTS bos embedding [talkerHiddenSize] FP16
    rt::Tensor mTtsEosEmbed; //!< TTS eos embedding [talkerHiddenSize] FP16
    //! Zero row used as the residual addend on the OmniNext decode step (pure
    //! autoregressive, no per-frame text addend). Legacy Qwen3-Omni uses mTtsPadEmbed.
    rt::Tensor mQwen3OmniNextZeroResidualAddend;

    //! Persistent FP32 -INF on device; source for suppressTalkerEosLogit's async D2D copy.
    //! Avoids sourcing an async memcpy from a stack variable.
    rt::Tensor mNegInfConst;

    // ========== Workspace Tensors (allocated at maxBatchSize) ==========
    // Buffers used for per-batch prefill (not batched engine execution, reused per-batch)
    rt::Tensor mThinkerEmbedBuffer; //!< Text embedding output [maxSeqLen, thinkerHiddenSize] FP16
    rt::Tensor mGpuTokenIdsBuffer;  //!< Token IDs upload buffer [1, maxSeqLen] INT32
    rt::Tensor mMLPWorkspace;       //!< MLP intermediate results [maxSeqLen, thinkerHiddenSize] FP16
    rt::Tensor mProjectedBuffer;    //!< Projected tokens [maxSeqLen, talkerHiddenSize] FP16
    rt::Tensor mTalkerInputEmbeds;  //!< Talker input embeddings [maxBS, maxSeqLen, talkerHiddenSize] FP16
    rt::Tensor mSamplingWorkspace;  //!< Workspace for sampling operations

    // Talker LLM workspace (batched)
    rt::Tensor mTalkerLogits;          //!< Talker output logits [maxBS, vocabSize] FP32
    rt::Tensor mTalkerSelectedIndices; //!< Selected token indices [maxBS, 1] INT32
    rt::Tensor mHostSelectedTokenIds;  //!< Host selected tokens [maxBS] INT32
    rt::Tensor mSeenCodecTokensBuf;    //!< Per-batch seen codec tokens [maxBS, maxKVCacheCapacity] INT32
    rt::Tensor mSeenSeedHostScratch;   //!< Pinned host scratch [maxBS] for async H2D seeding of the seen buffer

    // CodePredictor workspace (batch=1 for per-batch CodePredictor calls)
    rt::Tensor mCodePredictorLogits;            //!< CodePredictor output logits [maxBS, codebookSize] FP32
    rt::Tensor mCodePredictorSelectedIndices;   //!< Selected code indices [1, 1] INT32
    rt::Tensor mCodePredictorPrefillInput;      //!< Prefill input [1, 2, cpHidden] FP16
    rt::Tensor mCodePredictorCodecIds;          //!< Codec token IDs [1, 1] INT32
    rt::Tensor mCodePredictorCodecEmbed;        //!< Projected codec embed [1, 1, cpHidden] FP16
    rt::Tensor mRawCodecEmbed;                  //!< Raw codec embed [1, 1, talkerHidden] FP16
    rt::Tensor mSmallToMtpProjectedHidden;      //!< Projected talker hidden [1, cpHidden] FP16
    rt::Tensor mHostSelectedCodeIds;            //!< Host selected codes [maxBS] INT32
    rt::Tensor mHostGenCodeBuf;                 //!< Host pinned buffer for deferred CP gen-loop sample reads
                                                //!< [mNumRvqLayers - 1, maxBS] INT32 — accumulates code_2..code_15
                                                //!< for up to maxBS active batches so we can do one
                                                //!< cudaStreamSynchronize per frame instead of one per step
    rt::Tensor mHostCodePredictorContextLength; //!< Host CodePredictor context length [maxBS] INT32

    // ---- CodePredictor speculative decoding (chain). Allocated only when mCpSpecDecodeK > 0. ----
    //! Verify window the caller asked for, before the capability gates in allocateBuffer().
    int32_t mCpSpecRequested{0};
    bool mCpPrefillOnDecodeProfile{true}; //!< Cleared when the engine's generation profile is seq=1.
    int32_t mCpGenProfileMaxSeq{1};       //!< Widest verify window the CP engine's decode profile accepts
    rt::Tensor mCpTablePtrsGpu;           //!< Device array of the mNumRvqLayers CodePredictor table pointers
    //! Pinned staging for the above; a member because the upload outlives the call that issues it.
    rt::Tensor mCpTablePtrsHost;
    //! One [kCpRowIdxPlanes, maxBS * K] INT32 buffer uploaded once per round holding the draft
    //! and verify row selectors. Per-row rather than base+offset, so a tree-shaped verify
    //! window writes node depths into the same planes without touching the kernels.
    rt::Tensor mCpRowIdx;
    rt::Tensor mHostCpRowIdx;
    //! Scatter selectors for the accepted-code store: plane 0 is the destination row in
    //! mCodecHiddensBuffer (negative drops the row), plane 1 the RVQ depth's table.
    rt::Tensor mCpStoreIdx;
    rt::Tensor mHostCpStoreIdx;
    rt::Tensor mCpRawCodecEmbedRows; //!< [maxBS * K, talkerHidden] staging for the projected TTS layout
    //! Whole-frame uniform pool. A round consumes two contiguous slices, so a single fill
    //! covers the frame; refilled if a frame ever overruns it.
    rt::Tensor mCpUniformPool;
    int64_t mCpUniformPoolSize{0};
    //! Shape the CP executor's TRT bindings currently hold. Verify passes repeat the
    //! same shape for most of a frame, and rebinding costs more than the pass itself.
    int32_t mCpBoundBatch{-1};
    int32_t mCpBoundSeqLen{-1};
    int32_t mCpSpecDecodeK{0};
    rt::Tensor mCpDraftProbs;           //!< Draft distributions [maxBS, K-1, codebookSize] FP32
    rt::Tensor mCpDraftTokenIds;        //!< Sampled drafts [maxBS, K-1] INT32
    rt::Tensor mCpTargetProbs;          //!< Target distributions [maxBS, K, codebookSize] FP32
    rt::Tensor mCpVerifyInput;          //!< Verify pass input embeds [maxBS, K, cpHidden] FP16
    rt::Tensor mCpVerifyHidden;         //!< Verify pass hidden states [maxBS, K, cpHidden] FP16
    rt::Tensor mCpAcceptedTokenIds;     //!< Accepted codes [maxBS, K] INT32
    rt::Tensor mCpAcceptLength;         //!< Accepted count per batch [maxBS] INT32
    rt::Tensor mCpProposalLengths;      //!< Draft slot count per batch [maxBS] INT32
    rt::Tensor mHostCpAcceptedTokenIds; //!< Pinned host mirror of mCpAcceptedTokenIds
    rt::Tensor mHostCpAcceptLength;     //!< Pinned host mirror of mCpAcceptLength
    //! Head output stays FP16: cpSpecTopKTopPProbs is templated on the logit type and reads
    //! it directly, so the small-vocabulary path never materialises an FP32 logit row.
    rt::Tensor mCpLogitsFp16;
    rt::Tensor mCpVerifyCodeIds;     //!< Codes feeding the verify positions [maxBS, K] INT32
    rt::Tensor mHostCpSpecM;         //!< Per-batch index of the last finalised code [maxBS] INT32
    rt::Tensor mHostCpSpecBaseLen;   //!< Per-batch committed CP context length [maxBS] INT32
    rt::Tensor mHostCpProposalLens;  //!< Per-batch draft slot count staged for H2D [maxBS] INT32
    rt::Tensor mCpSpecCtx;           //!< Draft source: hidden that produced each batch's last code [maxBS, cpHidden]
    uint64_t mCpSpecRandomOffset{0}; //!< Monotonic philox offset across all SPD draws
    uint64_t mCpSpecRounds{0};       //!< Verify passes issued, for acceptance accounting
    uint64_t mCpSpecProposed{0};     //!< Draft slots proposed
    uint64_t mCpSpecCodes{0};        //!< Codes committed by the speculative loop

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
     * @param languageId Language codec token ID (-1 = no-language 8-row prefix; >= 0 = 9-row
     *        CustomVoice language-conditioned prefix)
     * @param output Projected talker input embeddings [outputSeqLen, talkerHiddenSize]
     * @param outputSeqLen seqLen + 2 (no language) or seqLen + 3 (with language)
     * @param stream CUDA stream
     * @return True on success, false on failure
     */
    bool projectToTalkerInput(rt::Tensor const& thinkerEmbed, int32_t speakerId, int32_t languageId, rt::Tensor& output,
        int64_t& outputSeqLen, cudaStream_t stream);

    //! Embed token IDs, run MLP projection, and reshape buffers ready for Talker prefill.
    //! Populates mTalkerInputEmbeds and mTalkerHiddenStatesBuffer as side effects.
    //! \param[out] outSeqLen  non-streaming prefill length (seqLen + 2, or seqLen + 3 with language)
    bool prepareTalkerInput(std::vector<int32_t> const& textTokenIds, TalkerGenerationRequest const& request,
        int64_t& outSeqLen, cudaStream_t stream);

    /*!
     * @brief Execute CodePredictor prefill step using CUDA Graph
     *
     * Pure engine wrapper. Batch dim derived from inputsEmbeds.getShape()[0] (1..maxBatchSize).
     * Resets CP KV cache and stages per-batch context lengths.
     *
     * @param inputsEmbeds Codec token embeddings [batch, seqLen, cpHidden] — caller builds the batched buffer.
     * @param lmHeadIdx Must be 0 (prefill always predicts with head 0).
     * @param outputLogits Output logits [batch, codebookSize] (engine output).
     * @param outputHiddenStates Output hidden states for residual / next step.
     * @param stream CUDA stream.
     */
    bool executeCodePredictorPrefillStep(rt::Tensor const& inputsEmbeds, int32_t lmHeadIdx, rt::Tensor& outputLogits,
        rt::Tensor& outputHiddenStates, cudaStream_t stream);

    /*!
     * @brief Set the step-invariant CP decode bindings (stacked lm_heads + device
     *        lm_head_idx + shared logits) and switch to the decode profile. One call
     *        covers the whole per-frame decode loop.
     */
    bool prepareCpDecodeBindings(int32_t activeBatchSize, cudaStream_t stream);

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
