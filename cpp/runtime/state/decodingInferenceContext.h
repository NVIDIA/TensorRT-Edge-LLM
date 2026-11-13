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

#include "common/checkMacros.h"
#include "runtime/decoding/jumpForward.h"
#include "runtime/exec/raggedBatchBuilder.h"
#include "runtime/llmRuntimeUtils.h"
#include "runtime/state/contextCache/blockHash.h"
#include "runtime/state/residentSlotPool.h"
#include "runtime/streaming.h"
#include "sampler/sampling.h"

#include <algorithm>
#include <atomic>
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

//! Relationship between the runtime token list and the model state materialized in KV cache.
enum class DecodingTokenStateContract : uint8_t
{
    //! The token list ends with one sampled token whose KV has not yet been materialized.
    kCommittedPlusLookahead,
    //! Every token in the runtime token list already has materialized KV state.
    kFullyCommitted,
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
    int64_t acceptedDraftTokens{0};          //!< Spec decode: draft tokens accepted by verification, summed over rounds
    int64_t proposedDraftTokens{0}; //!< Spec decode: draft tokens proposed for verification, summed over rounds
    //! Per-step top log-probabilities: logprobs[step] = [LogprobEntry, ...], sorted descending.
    //! Populated only when numLogprobs > 0 in the original request.
    std::vector<std::vector<LogprobEntry>> logprobs;
    FinishReason terminalReason{
        FinishReason::kNotFinished}; //!< Why this batch terminated (EOS, length, stop string, cancel, error)
    bool skipSpecialTokens{true};    //!< The sequence's detokenization policy, for assembling its text
};

//! Prompt tokens a sequence was served from reused state (context cache or saved system prompt) instead of
//! prefilling: the raw prompt less what its prefill executed and what visual-token pruning removed. The prefill
//! metrics and every per-request report count reuse through this one function.
inline int32_t reusedPromptTokens(
    int32_t rawPromptLength, int32_t effectivePrefillLength, int32_t prunedPrefillTokens) noexcept
{
    return rawPromptLength - effectivePrefillLength - prunedPrefillTokens;
}

/*!
 * @brief Per-request execution context shared by runtime and decoding strategies.
 *
 * Holds request-local sequence metadata, sampling parameters, multimodal
 * embedding references, streaming state, and batch-eviction bookkeeping.
 */
//! Everything a newly admitted sequence brings when it joins a batch that is already running.
//!
//! Only what is per-slot belongs here: the prompt, delivery, and the request's own decoding
//! settings (sampling, seed, generate budget, logprobs width). The few settings that remain
//! batch-wide -- the LoRA adapter, thinking mode -- live on the context itself; whether the
//! arriving request agrees with those is the scheduler's admission decision, made before a seed
//! is ever built.
struct SlotSeed
{
    SlotSeed() = default;
    SlotSeed(SlotSeed const&) = default;
    SlotSeed& operator=(SlotSeed const&) = default;
    SlotSeed(SlotSeed&&) noexcept = default;
    SlotSeed& operator=(SlotSeed&&) noexcept = default;

    std::string systemPrompt;

    //! The prompt, already tokenized. Also seeds the slot's running token history, mirroring how a
    //! batch built at request start begins each slot at its prompt.
    std::vector<int32_t> promptTokenIds;

    std::vector<std::string> stopStrings;

    //! Sparse output-vocab bias for this slot; empty for none.
    std::unordered_map<int32_t, float> logitBias;

    //! Token delivery for this slot; null opts out of streaming.
    std::shared_ptr<StreamChannel> channel;

    //! The request's cancellation bit, polled at every step beside the channel's flag; null when
    //! the owner cancels through the channel only.
    std::shared_ptr<std::atomic<bool> const> cancelToken;

    //! This sequence's sampling parameters, exactly as the request specified them.
    RowSamplingParams sampling{};

    //! Sampling seed for this logical request, already resolved by the runtime (the request's own,
    //! else kDefaultSamplingSeed); the batch's per-slot seed table receives it verbatim.
    uint64_t samplingSeed{kDefaultSamplingSeed};

    //! Draw this sequence's sampling uniforms from its seed and absolute position: set for an
    //! explicit seed, and for a lossless speculative decoder whose acceptance replays the draws.
    //! Otherwise the row draws from the shared counter stream keyed on its step and row index.
    bool requestStableSampling{false};

    //! Generate budget for this sequence, already clamped to what the KV pool can hold behind its
    //! prompt; positive.
    int32_t maxGenerateLength{0};

    //! Top log-probabilities to keep per generated token; zero for none. Already clamped to
    //! kMaxLogprobsK by the runtime.
    int32_t numLogprobs{0};

    //! Generate through EOS to the length budget; mirrors LLMGenerationRequest::ignoreEos.
    bool ignoreEos{false};

    //! Mirrors GuidedDecodingParams::jumpForward; only meaningful for a guided sequence.
    bool jumpForward{false};

    //! The index results are filed under in completedBatches. Must be unique among live slots and
    //! results already collected, or two sequences' outputs would land in one bucket.
    int32_t originalIndex{};

    //! Tokens before this offset already have KV state behind them (a context-cache prefix hit),
    //! so the slot's working token history starts here; zero for a cold sequence. The full prompt
    //! still lands in rawBatchedInputIds -- identity, hashing and result assembly need it.
    int32_t prefillStart{};

    //! Set by buildAdmissionIntent when the sequence must run alone in the batch (media under an
    //! active visual-token pruner, which gates on an empty base KV, or, in a compact session, media
    //! whose token layout only its encoder determines). The runtime session admits such
    //! a sequence only into an empty session and refuses company while it is resident; the scheduler
    //! asks RuntimeSession::requiresExclusiveAdmission before deciding to admit, so the request waits
    //! for the session to drain rather than joining and silently skipping the pruner.
    bool requiresExclusiveSession{false};

    //! Non-streaming detokenization policy for this sequence (streaming slots read it from their
    //! channel); mirrors LLMGenerationRequest::skipSpecialTokens.
    bool skipSpecialTokens{true};

    //! Whether the prompt already closes its reasoning block, so a grammar may constrain from the
    //! first generated token (reasoningClosedInPrompt). Only meaningful for a guided sequence.
    bool guidedReasoningEnded{false};

    //! @name Multimodal admission payload.
    //! Encoder outputs staged by the runtime's admission preprocess, consumed by the seated
    //! prefill within the same generation boundary. The tensor references point at the
    //! multimodal runners' output buffers, which stay untouched until the next admission's
    //! preprocess -- at most one admission is pending at a time, so the seated prefill is always
    //! the next consumer. The media identity travels too, so the context cache can tell a media
    //! prefix from a text prefix over the same ids: each image as the content identity the
    //! admission preprocess computed (one entry per image, in prompt order), audio as its raw
    //! buffers, which the cache hashes.
    //! @{
    OptionalInputTensor visualEmbeddings;
    OptionalInputTensor audioEmbeddings;
    OptionalInputTensors deepstackFeatures;
    std::vector<Hash128> imageContentDigests;
    std::vector<audioUtils::AudioData> audioBuffers;
    //! @}
};

struct DecodingInferenceContext
{
    std::vector<std::string> systemPrompts;               //!< System prompts for each sequence in batch
    std::vector<std::vector<int32_t>> rawBatchedInputIds; //!< Original token IDs before preprocessing
    std::vector<std::vector<int32_t>> tokenIds;           //!< Token IDs for each sequence: [batch_size][seq_length]
    std::vector<int32_t> currentGenerateLengths;          //!< Current generation length for each sequence
    std::vector<int32_t> effectivePrefillLengths;         //!< Prefill length after system prompt cache reuse
    std::vector<int32_t> prefillStartLengths;             //!< Persistent frontier captured at prefill invocation entry
    std::vector<int32_t> committedLengths;                //!< Per-sequence persistent state length before next step
    std::vector<RequestId> requestIds;                    //!< Stable identities assigned by the blocking adapter
    std::vector<ResidentRef> residentRefs;                //!< Current resident state rows and epochs
    ResidentSlotPool residentSlots;                       //!< Bounded ownership and epoch validation for resident rows
    StepId nextStepId{1};                                 //!< Monotonic execution-step identity
    ScheduledStep scheduledStep;                          //!< Owned logical-step descriptor snapshot
    std::optional<RaggedBatchBuilder> raggedBatchBuilder; //!< Configured once and reused for ordinary execution
    std::vector<CompletionSequence> completionSequences;  //!< Preallocated completion validation scratch
    RaggedExecutionBatch raggedExecutionBatch;            //!< Reused execution metadata and identity snapshot
    //! Compact (phase-8) builder; lowers into raggedExecutionBatch, so both builders share its reserved storage.
    std::optional<RaggedBatchBuilder> compactBatchBuilder;
    //! Compact-step row lists, reserved by initializeCompactScratch and cleared per step.
    std::vector<int32_t> compactMropeInitSlots;
    std::vector<int32_t> compactDecodeRows;
    std::vector<int32_t> compactSampledRows;
    //! Per-slot prompt tokens removed by visual-token pruning (empty when pruning is off —
    //! treat a missing entry as 0).
    std::vector<int32_t> prunedPrefillTokens;
    std::vector<int8_t> finishedStates;       //!< Finished state for each sequence
    std::vector<int64_t> acceptedDraftTokens; //!< Spec decode: accepted draft tokens summed per slot
    std::vector<int64_t> proposedDraftTokens; //!< Spec decode: proposed draft tokens summed per slot
    //! Decode rounds each slot took part in. Equal to generationRound on the blocking path, where every slot starts
    //! together; a sequence seated into a running batch counts from its own first round.
    std::vector<int32_t> decodeRounds;
    //! Per-slot thinking tracker: 1 once thinking is complete (end marker emitted, or the model
    //! never entered thinking). Lives here, rather than as a `handleRequest` local, so batch
    //! compaction reindexes it together with every other per-slot vector; otherwise an eviction
    //! leaves entry `i` pointing at a different request.
    std::vector<int8_t> thinkingDone;
    //! Per-slot guided-decoding gate: 1 once the reasoning block is over. Deliberately not
    //! `thinkingDone`, which also flips when the first generated token is not a thinking
    //! marker -- that token is answer content, so treating it as "reasoning over" would let
    //! it past the mask and out of the matcher's prefix. Only the end marker opens this one;
    //! it starts open when the request has thinking off.
    std::vector<int8_t> guidedReasoningEnded;
    //! Per slot: 1 when the request asked for jump-forward (GuidedDecodingParams::jumpForward).
    std::vector<int8_t> jumpForwardEnabled;
    //! Per slot: tokens at the end of tokenIds that have no KV state yet, i.e. what the next step feeds. Prefill
    //! and sampling set it to 1; a jump raises it to 1 + the forced tokens; speculative decoders never touch it.
    std::vector<int32_t> pendingTokens;
    //! Per slot: jump-forward counters, reported when the sequence ends.
    std::vector<JumpForwardSlotStats> jumpForwardStats;
    //! Content identity of each image of each sequence, in request order, written by multimodal
    //! preprocessing and read by the encoder embedding cache and the context cache. Either every
    //! image of the request has an entry or the whole table is empty, so a single image whose
    //! identity cannot be computed takes the request out of both caches.
    //! Read while the batch is still the admitted one, so compaction never reindexes it.
    std::vector<std::vector<Hash128>> imageContentDigests;

    std::unordered_map<int32_t, BatchResult> completedBatches; //!< Results of completed batches
    std::vector<int32_t> batchIndexMapping;                    //!< Maps current batch index to original index
    std::vector<SlotStreamState> slotStreams;                  //!< Per-slot streaming state
    rt::OptionalInputTensor visualEmbeddings;                  //!< Optional visual embeddings
    rt::OptionalInputTensor audioEmbeddings;                   //!< Optional audio embeddings
    rt::OptionalInputTensors deepstackFeatures;                //!< Optional Deepstack features
    int32_t generationRound{};                                 //!< Current generation round
    int32_t diffusionMaxDenoisingSteps{0}; //!< Optional DiffusionGemma denoise-step override (0 = runtime default)
    int32_t activeBatchSize{};             //!< Current active batch size
    std::string loraWeightsName{""};       //!< LoRA adapter name used by this request
    cudaStream_t stream{};                 //!< CUDA stream

    //! @name Per-slot decoding settings
    //! Each sequence's own sampling parameters, seed, generate budget (KV-clamped by the runtime),
    //! logprobs width and EOS policy; rows of one batch may differ. The sampler launches once per
    //! run of equal rows, and logprobs are extracted at the widest width and trimmed per slot.
    //! @{
    std::vector<RowSamplingParams> samplingParams;
    std::vector<uint64_t> samplingSeeds;
    std::vector<int8_t> requestStableSampling; //!< 1: uniforms from the slot's seed and position (SlotSeed)
    std::vector<int32_t> maxGenerateLengths;
    std::vector<int32_t> numLogprobsPerSlot;
    std::vector<int8_t> ignoreEos; //!< 1: generate through EOS to the slot's budget
    //! @}
    SpecProposalSampling proposalSampling{SpecProposalSampling::kAuto};
    //! Tokens a step may add beyond a slot's budget (maxAcceptedTokensPerRound - 1), recorded once
    //! by the runtime; each slot's accumulator holds (maxGenerateLength + overshoot) * numLogprobs.
    int32_t logprobsStepOvershoot{0};
    //! Per-slot flat logprobs accumulator; slot.data[step*K .. (step+1)*K-1] holds step's top-K
    //! (token_id, log_prob) pairs, K the slot's own width.
    std::vector<rt::LogprobsSlot> stepLogprobs;

    // Per-slot stop strings; empty list disables stop-string termination for that slot.
    std::vector<std::vector<std::string>> stopStringsPerSlot;

    std::vector<std::unordered_map<int32_t, float>>
        logitBiasPerSlot;          //!< Per-active-slot sparse logit bias maps in output-vocab space
    bool hasLogitBias{false};      //!< True when any active slot has logit bias entries
    bool logitBiasGpuDirty{false}; //!< True when CPU-side bias state must be uploaded to GPU

    //! True when at least one slot compiled a grammar; gates all per-step guided work. The
    //! matchers live in the runtime-owned GuidedDecoder under this same slot numbering.
    bool hasGuidedDecoding{false};
    //! True when at least one slot has jumpForwardEnabled; gates the per-step jump-forward work.
    bool hasJumpForward{false};
    //! Scratch reused every step: slots whose grammar admitted no token at all.
    std::vector<int32_t> guidedUnsatisfiableSlots;
    //! Scratch reused every step: per-slot "leave unconstrained this step" flags, combining
    //! finished slots with slots still inside their thinking block.
    std::vector<int8_t> guidedMaskSuppressedPerSlot;
    //! Mirrors LLMGenerationRequest::enableThinking.
    bool enableThinking{false};

    bool outputThinkerEmbeddings{false}; //!< Whether to capture hidden states for the Talker pipeline

    //! Hybrid+spec context-reuse endpoint path is active for this request. Set by the runtime before the folded draft
    //! prefill so MTPDecoder::initializeForGeneration runs the draft prefill pre-publication (default MTP keeps its
    //! decode-round-0 draft prefill when this is false).
    bool hybridSpecEndpointReuse{false};
    //! Guards the speculative draft prefill so it runs exactly once, whether triggered pre-publication (Hybrid+spec
    //! endpoint reuse) or in decode round 0 (default speculative path).
    bool speculativeDraftPrefillComplete{false};
    //! Hybrid+spec boundary-replay tail length for the two-chunk prefill (0 = single-chunk). Carried from the request.
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

    //! @brief Add one sequence to the batch. Returns the new execution-row index.
    //!
    //! The reverse of eviction, and bound by the same invariant: every per-slot vector grows by
    //! exactly one entry, so the batch stays rectangular and a later compaction moves every slot's
    //! state together. Slots already in flight are not touched -- their tokens, progress and
    //! delivery state stay where they are.
    //!
    //! The batch may be empty: a context initialized for zero sequences grows its first slot here
    //! exactly as it grows its tenth. Nothing per-slot is inherited from a neighbour; the logprobs
    //! accumulator is sized from the seed's own budget and width.
    //!
    //! @param identity Stable request identity and an already-acquired physical resident row.
    //! @throws std::runtime_error if the prompt is empty, the seed carries no generate budget or an
    //!         out-of-range logprobs width or top-k, either identity is invalid or already live, or
    //!         the seed's channel is attached elsewhere. Nothing is modified on any throwing path;
    //!         the caller retains ownership of @p identity.resident.
    int32_t appendSlot(SlotSeed seed, SequenceIdentity identity);

    //! @brief Exchange the transient execution-row view of two logical sequences. Self-inverse.
    //!
    //! This is used to present one admitted sequence as execution row zero for a batch-one prefill.
    //! ResidentRef travels with the logical sequence, so ragged state_indices still name its stable
    //! physical KV/recurrent row; no resident state or page-table row is exchanged.
    //!
    //! @throws std::runtime_error if either slot is out of range. A self-swap is a no-op.
    void swapExecutionRows(int32_t rowA, int32_t rowB);

    //! Restore a previously validated execution-row exchange during stack unwinding. The caller
    //! must pass the same rows accepted by swapExecutionRows before changing activeBatchSize.
    void restoreExecutionRows(int32_t rowA, int32_t rowB) noexcept;

    //! The sampling parameters every slot shares, for decoders that take them per launch
    //! (speculative). Throws when the slots disagree, so a mixed batch fails loudly instead of
    //! running on slot zero's settings; defaults for an empty batch. Budgets and logprobs widths
    //! stay per slot.
    RowSamplingParams uniformSamplingParams() const;

    //! Widest logprobs width any live slot asks for (zero when none): the batch extracts at this
    //! width and each slot keeps its own prefix.
    int32_t maxNumLogprobs() const noexcept;
    //! The same over the execution rows @p rows names (a compact step's sampled rows).
    int32_t maxNumLogprobsForRows(std::vector<int32_t> const& rows) const noexcept;

    //! An accumulator of (maxGenerateLength + logprobsStepOvershoot) * numLogprobs pairs, or none
    //! for width zero.
    rt::LogprobsSlot makeLogprobsAccumulator(int32_t maxGenerateLength, int32_t numLogprobs) const;

    //! Size slot @p slot's accumulator for its own budget and width (a batch built at request start;
    //! appendSlot() sizes an appended slot's before it exists).
    void allocateLogprobsAccumulator(int32_t slot);

    //! The rectangularity that appendSlot() and eviction both preserve, stated once so call sites
    //! can assert it instead of each auditing thirteen vectors. A vector missed by one of the two
    //! paths shows up here as a loud failure rather than as another slot's tokens.
    bool perSlotSizesConsistent() const noexcept;

    //! Generated tokens at which @p row stops with FinishReason::kLength: the sequence's own
    //! KV-clamped budget. @throws std::runtime_error when @p row is outside the batch.
    int32_t generateBudget(int32_t row) const
    {
        ELLM_CHECK(row >= 0 && static_cast<size_t>(row) < maxGenerateLengths.size(),
            "generateBudget: row " + std::to_string(row) + " has no per-slot generate budget.");
        return maxGenerateLengths[static_cast<size_t>(row)];
    }

    //! @brief Check that @p rows names distinct execution rows in [0, activeBatchSize), as the row-set
    //!        sampling helpers require.
    //! @throws std::runtime_error naming the violation otherwise.
    void checkRowSet(std::vector<int32_t> const& rows) const;

    /*!
     * @brief Initialize request-local vectors and scalar fields for a batch of @p batchSize sequences,
     *        each holding one freshly acquired resident row.
     *
     * @p batchSize may be zero: the context then holds no sequence and every later one arrives
     * through appendSlot(). @p residentCapacity bounds the resident rows over the context's life;
     * zero means "exactly the initial batch", so an empty batch needs it explicit.
     *
     * @param batchSize Active batch size
     * @param maxGenLength Maximum generation length
     * @param visual Optional visual embeddings
     * @param deepstackFeatures Deepstack features for Qwen3-VL
     * @param loraName LoRA weights name used by this request
     * @param cudaStream CUDA stream for operations
     * @param residentCapacity Resident rows the slot pool hands out, or 0 for @p batchSize
     */
    void initialize(int32_t batchSize, int32_t maxGenLength, rt::OptionalInputTensor const& visual,
        rt::OptionalInputTensors const& deepstackFeatures, std::string const& loraName, cudaStream_t cudaStream,
        int32_t residentCapacity = 0);

    void initializeRaggedScratch(RaggedEngineContract const& contract);

    //! Configure compactBatchBuilder; @p contract must use the compact backend and the same capacities as the
    //! ordinary contract so either builder can lower into raggedExecutionBatch without reallocation.
    void initializeCompactScratch(RaggedEngineContract const& contract);

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
