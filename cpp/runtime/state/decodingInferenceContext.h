/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "runtime/llmRuntimeUtils.h"
#include "runtime/streaming.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace trt_edgellm
{
namespace rt
{

class LayerDebugger; // Few-layer-validation debug: per-layer logits/KV dump (runtime/debug/layerDebugger.h)

//! Maximum KV positions that one decoder execution may address beyond its materialized endpoint.
struct DecodingKvHeadroom
{
    int32_t baseExtraTokens{};
    int32_t draftExtraTokens{};
};

/**
 * @brief Pre-allocated flat accumulator for per-step log-probabilities of one batch slot.
 *
 * Stores up to `maxGenerateLength * topK` pairs in a single allocation made at request start,
 * eliminating per-step heap allocations inside the decode loop.
 * `data[step * topK .. (step+1) * topK - 1]` holds the top-K (token_id, log_prob) pairs for step `step`.
 */
struct LogprobsSlot
{
    std::vector<std::pair<int32_t, float>> data; //!< Flat [maxGenerateLength * topK] storage, pre-allocated once
    int32_t numSteps{0};                         //!< Number of steps written so far
};

/*!
 * @brief Batch result data for a single sequence.
 *
 * Encapsulates all data needed to track a batch's execution results, whether
 * the sequence is still active or has already been evicted.
 */
struct BatchResult
{
    std::vector<int32_t> tokenIds;           //!< Generated token IDs
    std::vector<int32_t> rawBatchedInputIds; //!< Original input token IDs
    int32_t generateLength{0};               //!< Number of tokens generated
    int32_t actualIterations{0};             //!< Number of iterations executed
    int32_t effectivePrefillLength{0};       //!< Effective prefill length after system prompt cache reuse
    int32_t prunedPrefillTokens{0};          //!< Prompt tokens removed by visual-token pruning
    //! Per-step top log-probabilities: logprobs[step] = [LogprobEntry, ...], sorted descending.
    //! Populated only when numLogprobs > 0 in the original request.
    std::vector<std::vector<LogprobEntry>> logprobs;
    FinishReason terminalReason{
        FinishReason::kNotFinished}; //!< Why this batch terminated (EOS, length, stop string, cancel, error)
};

/*!
 * @brief Per-request execution context shared by runtime and decoding strategies.
 *
 * Holds request-local sequence metadata, sampling parameters, multimodal
 * embedding references, streaming state, and batch-eviction bookkeeping.
 */
struct DecodingInferenceContext
{
    std::vector<std::string> systemPrompts;               //!< System prompts for each sequence in batch
    std::vector<std::vector<int32_t>> rawBatchedInputIds; //!< Original token IDs before preprocessing
    std::vector<std::vector<int32_t>> tokenIds;           //!< Token IDs for each sequence: [batch_size][seq_length]
    std::vector<int32_t> currentGenerateLengths;          //!< Current generation length for each sequence
    std::vector<int32_t> effectivePrefillLengths;         //!< Prefill length after system prompt cache reuse
    //! Per-slot prompt tokens removed by visual-token pruning (empty when pruning is off —
    //! treat a missing entry as 0).
    std::vector<int32_t> prunedPrefillTokens;
    std::vector<int8_t> finishedStates; //!< Finished state for each sequence

    std::unordered_map<int32_t, BatchResult> completedBatches; //!< Results of completed batches
    std::vector<int32_t> batchIndexMapping;                    //!< Maps current batch index to original index
    std::vector<SlotStreamState> slotStreams;                  //!< Per-slot streaming state
    rt::OptionalInputTensor visualEmbeddings;                  //!< Optional visual embeddings
    rt::OptionalInputTensor audioEmbeddings;                   //!< Optional audio embeddings
    rt::OptionalInputTensors deepstackFeatures;                //!< Optional Deepstack features
    int32_t generationRound{};                                 //!< Current generation round
    int32_t maxGenerateLength{};                               //!< Maximum generation length
    int32_t diffusionMaxDenoisingSteps{0}; //!< Optional DiffusionGemma denoise-step override (0 = runtime default)
    int32_t activeBatchSize{};             //!< Current active batch size
    std::string loraWeightsName{""};       //!< LoRA adapter name used by this request
    cudaStream_t stream{};                 //!< CUDA stream

    float temperature{1.0f}; //!< Temperature for sampling
    float topP{1.0f};        //!< Top-P sampling parameter
    int64_t topK{0};         //!< Top-K sampling parameter
    int32_t numLogprobs{0};  //!< Number of top log-probs to collect per generated token
    //! Per-batch flat logprobs accumulator.  slot.data is pre-allocated
    //! [(maxGenerateLength + draftingStep) * numLogprobs] in spec-decode mode (vanilla: maxGenerateLength)
    //! to accommodate the up-to-(draftingStep+1) tokens accepted per verify step.
    //! slot.data[step*numLogprobs .. (step+1)*numLogprobs-1] holds step's top-K (token_id, log_prob) pairs.
    std::vector<rt::LogprobsSlot> stepLogprobs;

    // Per-slot stop strings; empty list disables stop-string termination for that slot.
    std::vector<std::vector<std::string>> stopStringsPerSlot;

    std::vector<std::unordered_map<int32_t, float>>
        logitBiasPerSlot;          //!< Per-active-slot sparse logit bias maps in output-vocab space
    bool hasLogitBias{false};      //!< True when any active slot has logit bias entries
    bool logitBiasGpuDirty{false}; //!< True when CPU-side bias state must be uploaded to GPU

    bool outputThinkerEmbeddings{false}; //!< Whether to capture hidden states for the Talker pipeline

    //! Hybrid+MTP context-reuse endpoint path is active for this request. Set by the runtime before the folded draft
    //! prefill so MTPDecoder::initializeForGeneration runs the draft prefill pre-publication (default MTP keeps its
    //! decode-round-0 draft prefill when this is false).
    bool hybridMtpEndpointReuse{false};
    //! Guards the speculative draft prefill so it runs exactly once, whether triggered pre-publication (Hybrid+MTP
    //! endpoint reuse) or in decode round 0 (default speculative path).
    bool speculativeDraftPrefillComplete{false};
    //! Hybrid+MTP boundary-replay tail length for the two-chunk prefill (0 = single-chunk). Carried from the request.
    int32_t contextCacheReplayTailLength{0};

    //! Optional per-token callback invoked after each accepted token update.
    std::optional<TokenCallback> onTokenGenerated;
    std::vector<int32_t> callbackEmittedTokenCounts; //!< Per-slot count of tokenIds already sent to callback.

    //! Optional callback used by speculative decoders to stop appending accepted tokens.
    std::function<bool(int32_t, int32_t)> shouldStopAfterAcceptedToken;

    //! Few-layer-validation debug: per-request layer dumper (null unless the
    //! EDGELLM_DUMP_LOGITS_KVCACHE_* env vars are set). Owned here via RAII so it shares the
    //! context's lifetime exactly; see the out-of-line destructor. Also carries optional
    //! teacher-forcing tokens (EDGELLM_FORCE_TOKENS_FILE) applied via LayerDebugger::applyForcedTokens.
    std::unique_ptr<LayerDebugger> layerDebugger;

    /*!
     * @brief Initialize request-local vectors and scalar fields.
     * @param batchSize Active batch size
     * @param maxGenLength Maximum generation length
     * @param visual Optional visual embeddings
     * @param deepstackFeatures Deepstack features for Qwen3-VL
     * @param loraName LoRA weights name used by this request
     * @param cudaStream CUDA stream for operations
     */
    void initialize(int32_t batchSize, int32_t maxGenLength, rt::OptionalInputTensor const& visual,
        rt::OptionalInputTensors const& deepstackFeatures, std::string const& loraName, cudaStream_t cudaStream);

    //! ctor / move ops / dtor are out-of-line (defined in the .cpp) because @ref
    //! layerDebugger is a ``unique_ptr`` to the incomplete type ``LayerDebugger``:
    //! a defaulted special member in the header would instantiate the member's
    //! destructor against the incomplete type in every translation unit (e.g.
    //! unit tests that only forward-declare LayerDebugger). Move ops are declared
    //! because the user-declared destructor otherwise suppresses the implicit
    //! move, and the context is returned by value in places. The struct stays
    //! non-copyable via the unique_ptr member.
    DecodingInferenceContext();
    DecodingInferenceContext(DecodingInferenceContext&&) noexcept;
    DecodingInferenceContext& operator=(DecodingInferenceContext&&) noexcept;
    ~DecodingInferenceContext();
};

} // namespace rt
} // namespace trt_edgellm
