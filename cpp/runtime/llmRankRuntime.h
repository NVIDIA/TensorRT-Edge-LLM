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

#include "action/actionKvBatch.h"
#include "action/alpamayo1ActionRunner.h"
#include "common/tensor.h"
#include "multimodal/common/multimodalRunner.h"
#include "profiling/metrics.h"
#include "profiling/timer.h"
#include "runtime/config/deploymentConfig.h"
#include "runtime/config/deploymentOptions.h"
#include "runtime/config/llmEngineConfig.h"
#include "runtime/config/pleRuntimeConfig.h"
#include "runtime/decoding/decoderRegistry.h"
#include "runtime/decoding/guidedDecoder.h"
#include "runtime/decoding/logitBias.h"
#include "runtime/deviceFailure.h"
#include "runtime/exec/engineExecutor.h"
#include "runtime/exec/tensorMap.h"
#include "runtime/features/deepstackBinding.h"
#include "runtime/llmRuntimeUtils.h"
#include "runtime/managedKVCacheRequest.h"
#include "runtime/modelArtifacts.h"
#include "runtime/multiDevice/parallelConfig.h"
#include "runtime/preprocess/embeddingPreprocessor.h"
#include "runtime/preprocess/gemma4EmbeddingPreprocessor.h"
#include "runtime/preprocess/qwen4ExpNgramPreprocessor.h"
#include "runtime/preprocess/stepPreparer.h"
#include "runtime/preprocess/visualTokenPruner.h"
#include "runtime/sessionOptions.h"
#include "runtime/state/compactMediaReservation.h"
#include "runtime/state/contextCache/contextCacheConfig.h"
#include "runtime/state/contextCache/contextCacheMetrics.h"
#include "runtime/state/contextCache/encoderEmbeddingCache.h"
#include "runtime/state/decodingInferenceContext.h"
#include "runtime/state/pipelineIO.h"
#include "runtime/state/sharedResources.h"
#include "runtime/streaming.h"
#include "runtime/visionDataParallel/visionGather.h"
#include "tokenizer/tokenizer.h"
#include <atomic>
#include <bitset>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

namespace trt_edgellm
{
namespace chat_template
{
class ChatTemplate;
}

namespace rt
{

class ContextCacheCoordinator;
class ContextCacheRequest;
class ManagedKVCacheRequest;
class BoundedSwaKVPageManager;
class PleIoBudget;
struct VisionDpPlan;

/*!
 * @brief Internal one-rank LLM execution runtime.
 *
 * Manages inference pipeline for vanilla and speculative decoding modes (EAGLE, MTP, etc.).
 * When constructed without a drafting config, operates as a pure vanilla decoding runtime
 * with zero draft-model memory overhead.
 * Coordinates base model, optional draft model, and multimodal processing (vision + audio).
 *
 * Borrows the shared tokenizer and owns the engine, decoder, KV cache, multimodal runners, sampler, and
 * request execution state for exactly one resolved runtime rank. Public SD/MD APIs
 * should go through LLMInferenceRuntime and RuntimeCoordinator instead.
 *
 * @note This class is not thread-safe. Callers must externally serialize every method invocation and the object
 * lifetime. handleRequest() defensively rejects accidental overlapping calls before mutating runtime state, but that
 * gate does not authorize concurrent use of this object.
 */
class RuntimeSession;
class RankRuntimeSession;

//! What became of an admission attempt.
enum class AdmitDecision : uint8_t
{
    //! Resident; its result arrives through the ordinary harvest.
    kAdmitted,
    //! Transient resource pressure (context-cache pages held by other leases, or a sequence that
    //! must run alone while others are resident). Nothing was modified; retry at a later step.
    kNoCapacity,
    //! The prompt past its cached prefix exceeds the engine input limit; the intent's prefillStart
    //! is that prefix. Nothing was reserved or evicted.
    kOverlongPrefill,
    //! Admission or its prefill failed. The sequence is terminal from birth inside the batch and
    //! its (empty, error) result surfaces through takeCompletedAtOrAbove like any other.
    kFailed,
};

class LLMRankRuntime
{
public:
    //! Releases the runtime's one-request-at-a-time latch, whatever path retires the request or
    //! closes the session that held it. Acquisition is the owner's compare-exchange; this only
    //! guarantees the release.
    struct InProgressGuard
    {
        explicit InProgressGuard(std::atomic<bool>& active) noexcept
            : mActive(active)
        {
        }

        ~InProgressGuard() noexcept
        {
            mActive.store(false, std::memory_order_release);
        }

        InProgressGuard(InProgressGuard const&) = delete;
        InProgressGuard& operator=(InProgressGuard const&) = delete;

        std::atomic<bool>& mActive;
    };

    //! Callback type for broadcasting GPU int32 buffers across parallel ranks.
    //! Rank 0 sends; peer ranks receive.
    using TokenBroadcastFn = std::function<bool(void* buffer, int32_t count, cudaStream_t stream)>;
    //! Gather two FP16 visual shards already placed at their final offsets in the output buffer.
    using VisionGatherFn = rt::VisionGatherFn;

    /*!
     * @brief Construct one rank-local runtime from a fully-resolved parallel mapping.
     * Preferred entry point — carries tensor/context/expert coordinates so future
     * CP/EP support needs no further constructor changes.
     */
    LLMRankRuntime(DeploymentOptions const& options, cudaStream_t stream, ParallelMapping const& mapping,
        tokenizer::Tokenizer& tokenizer, chat_template::ChatTemplate const& chatTemplate,
        std::shared_ptr<PleIoBudget> pleIoBudget = nullptr);

    LLMRankRuntime(ModelArtifacts&& artifacts, DeploymentOptions const& options, cudaStream_t stream,
        ParallelMapping const& mapping, tokenizer::Tokenizer& tokenizer,
        chat_template::ChatTemplate const& chatTemplate, std::shared_ptr<PleIoBudget> pleIoBudget = nullptr);

    //! @brief Destructor
    ~LLMRankRuntime();

    //! @brief Capture CUDA graphs for decoding stages to optimize performance.
    //!
    //! When draft model is present, captures graphs for draft proposal, draft accept token,
    //! base verification, and base vanilla decoding. Without draft model, captures only
    //! vanilla decoding graphs.
    //!
    //! @param stream CUDA stream
    //! @return True if all stage captures succeed, false otherwise
    //! @throws std::runtime_error if a tensor reshape operation fails
    //! @note If capture fails for any stage, the inference can proceed without CUDA graph capture,
    //! but at cost of performance degradation.
    bool captureDecodingCUDAGraph(cudaStream_t stream);

    /*!
     * @brief Handle generation request
     * @param request Generation request with prompts and parameters
     * @param response Output response with generated tokens and text
     * @param stream CUDA stream
     * @return True on success, false on failure
     * @throws std::runtime_error if an LLM or CUDA operation fails
     * @note Calls on the same runtime must be externally serialized. An accidental overlap with another
     * handleRequest() is rejected before runtime or response state is mutated; this is not a general thread-safety
     * guarantee.
     */
    //! One request's generation, advanced a step at a time.
    //!
    //! Exists because the decode loop carries state between steps: a thinking-done flag per slot,
    //! several tokenizer ids, and a stop predicate that closes over them. While those were locals in
    //! one long function, that state and the loop were forced to share a lifetime by construction.
    //! As an object they still share one, but it is now the object's, and a caller can hold it
    //! across steps instead of being obliged to run the loop to the end in a single call.
    //!
    //! The prefill preceding the first step also produces a token on most backbones, so a session is
    //! primed once before it is stepped -- see primeFromPrefill().
    class GenerationSession final
    {
        //! The runtime session drives this through the typed stage methods and reads the context it
        //! wraps; nothing else reaches in.
        friend class RankRuntimeSession;

    public:
        //! @param synchronizeCancellation Run the cross-rank cancellation consensus at every step.
        //!        The blocking path needs it only when the request streams (a consumer can cancel
        //!        the channel); a scheduled session needs it always, since any resident's channel
        //!        may be cancelled and there is no single request whose channels would decide.
        //! @param actionSlots Per original index, whether that sequence carries a trajectory
        //!        request (Alpamayo); empty when none does. Indexed by batchIndexMapping.
        //! @param jumpForwardKvHeadroom The headroom of a slot that runs with jump-forward; equal to @p kvHeadroom
        //!        when no slot does or the strategy does not widen its rounds for a jump.
        GenerationSession(LLMRankRuntime& runtime, DecodingInferenceContext& context, DecodingStrategy& strategy,
            ManagedKVCacheRequest* managedRequest, DecodingKvHeadroom const& kvHeadroom,
            DecodingKvHeadroom const& jumpForwardKvHeadroom, cudaStream_t stream, bool synchronizeCancellation,
            std::vector<bool> actionSlots);

        //! Clears the stop predicate the context holds, which closes over members of this object.
        ~GenerationSession();

        GenerationSession(GenerationSession const&) = delete;
        GenerationSession& operator=(GenerationSession const&) = delete;

        //! @brief Consume the token prefill produced, before any decode step runs.
        //!
        //! Prefill emits a token on every backbone except the diffusion one, and that token has to
        //! travel the same cancel/decode/finalize/emit path a decode step would give it. Skipping
        //! this drops the first token of every request.
        bool primeFromPrefill();

        //! @brief True once every slot has reached a terminal state, or none are left.
        bool finished() const;

        //! @brief Advance every live slot by one decode step.
        bool advance();

        //! @brief Move out the results of finished sequences whose original index is >= @p firstIndex.
        //!
        //! Results are filed under the original index a sequence was seated with; the session
        //! harvests them all (firstIndex 0) at every step it runs.
        std::unordered_map<int32_t, BatchResult> takeCompletedAtOrAbove(int32_t firstIndex);

        //! @brief Turn one finished sequence's result into the caller-facing response, decoding
        //!        included, exactly as the blocking path's own assembly would have.
        LLMGenerationResponse materializeResult(
            BatchResult const& result, std::vector<std::string> const& stopStrings) const;

    private:
        //! The product of admission stage one, and the boundary between "nothing happened" and
        //! "something must be unwound": validation, tokenization, and the encoder preprocess all
        //! land here without touching resident state. Discarding an intent leaves the batch
        //! byte-identical -- the staging buffers its seed references are overwrite-only admission
        //! scratch, reused by the next admission.
        struct AdmissionIntent
        {
            SlotSeed seed;
            RequestId requestId{0};
            std::optional<ResidentRef> resident;
            //! The sequence's context-cache policies, carried into its lease. The lookup policy already
            //! has everything that forces bypass applied (ContextCacheRequest::lookupPolicyFor).
            ContextCacheLookupPolicy lookupPolicy{ContextCacheLookupPolicy::kUseCache};
            ContextCacheCommitPolicy commitPolicy{ContextCacheCommitPolicy::kIncludingGeneratedTokens};
            //! The volatile generation-prompt tail a Hybrid+spec checkpoint stops before
            //! (LLMGenerationRequest::contextCacheReplayTailLength, derived from the chat template when the request
            //! asks for it).
            int32_t replayTailLength{0};
            //! The single-sequence media request whose encoders run only once the reservation holds
            //! (compact admissions); empty when the intent stage already ran them.
            std::optional<LLMGenerationRequest> deferredMedia;
        };

        //! Stage one: validate the request shape and produce the intent (tokenize, or run the
        //! encoder preprocess for media). Throws on requests the batch rejects outright; resident
        //! state is untouched on every path.
        //! With @p deferMediaEncoders (a compact session) a media request's intent carries only the
        //! tokenization, media-token expansion, image digests and MRope; its encoders run in
        //! stageReservedMedia. Media that must run alone under a visual-token pruner is never deferred.
        //! @p strategy is the decoder the batch runs the sequence on, latched by the caller.
        AdmissionIntent buildAdmissionIntent(LLMGenerationRequest const& request, int32_t originalIndex,
            RequestId requestId, bool deferMediaEncoders, DecodingStrategy const& strategy);

        //! Stage two: reserve what the sequence needs before it owns a slot -- the generate-budget
        //! check, the resident identity, under a context cache the page lease (setting the
        //! intent's prefillStart), and the sequence's row in the cache manager's batch. A refusal
        //! leaves the batch untouched; a throw is a device failure after which the caller reports
        //! the session lost. Rollback of a completed reservation is releaseReservation.
        AdmitDecision reserveAdmission(AdmissionIntent& intent);

        //! The media stage of a compact admission, after every refusal: forget what the
        //! reserved slot's previous sequence staged, and for a deferred media intent restore its encoder
        //! outputs from the encoder embedding cache or run the encoders, copy them into the slot's
        //! reservation and write the slot's MRope row for the whole prompt. A throw leaves the
        //! reservation for the caller to release.
        void stageReservedMedia(AdmissionIntent const& intent);

        //! Give back what reserveAdmission took -- the cache manager row, the page lease and the
        //! resident identity -- for an intent the caller refuses after all (its uncached prompt
        //! exceeds the engine's input length, or seating threw). Nothing was resident, so the batch
        //! is as it was.
        void releaseReservation(AdmissionIntent& intent);

        //! Stage three is seating and commit, in two halves (seatSlot, prefillSeated). appendSlot is
        //! the commit point -- before it, failure retracts the lease and nothing was resident; from
        //! it on, every failure files as the new slot's terminal kError result through the ordinary
        //! harvest, so the batch never holds a half-admitted sequence.

        //! Two index spaces meet in a session and are named apart. An execution *row* is a
        //! sequence's position in the context's per-slot vectors; every eviction repacks the rows,
        //! so a row is valid for one step. A resident *slot* (ResidentRef::slot) is stable for the
        //! sequence's life and indexes what the device keeps per sequence: the recurrent and
        //! convolution state pools, the MRoPE rows, the page-table rows, and the caller's ledger.
        //! The two agree only until a non-tail resident leaves; a device-side index that took one
        //! for the other would corrupt another sequence's state without any host-side check
        //! tripping, so `row` and `slot` are never the same name here.

        //! What seatSlot leaves for prefillSeated: the committed seat's execution row plus
        //! everything the seated prefill still needs from the seed. Valid until the next admission
        //! overwrites the admission staging buffers the media references point into -- the seated
        //! prefill must run before another admission's preprocess.
        struct PendingSeat
        {
            int32_t row{-1};
            int32_t prefillStart{0};
            int32_t rawInputLength{0};
            bool hasMedia{false};
            SlotSeed mediaPayload;
            ContextCacheLookupPolicy lookupPolicy{ContextCacheLookupPolicy::kBypass};
            int32_t replayTailLength{0};
        };

        //! The first half of seating: appendSlot (the commit point) plus the slot's MRope cos/sin row.
        //! Throws propagate after retracting a context-cache lease. The runtime session calls this
        //! from admit(); the seated prefill is the sequence's context step.
        PendingSeat seatSlot(AdmissionIntent intent);

        //! The second half: the seated batch-1 prefill, the context-cache ledger finalize, and the
        //! lookahead-token pipeline. On failure the slot is terminal from birth.
        AdmitDecision prefillSeated(PendingSeat pending);

        //! Retire a seated row whose prompt will not be completed: a cancel that landed before its
        //! final prompt chunk ran. Any live row qualifies, including one whose earlier chunks
        //! already ran. The row is already marked finished with the reason; the cache-side
        //! admission is aborted and its channel finished, and the caller's eviction files it.
        void retireSeatUnprefilled(int32_t row);

        //! One sequence of a compact step: execution row @p row runs @p queryLength positions from
        //! its committed length. A context row continues its prompt (firstChunk when none of it ran
        //! yet) and samples only when the chunk reaches the end of the prompt; a decode row runs its
        //! lookahead token and samples.
        struct CompactRow
        {
            int32_t row{-1};
            SequenceWork work{SequenceWork::kContext};
            int32_t queryLength{0};
            bool firstChunk{false};
        };

        //! @brief Run one compact phase-8 step over @p rows (every context row before every decode
        //!        row): one eager forward, sampling only for the rows that produce a
        //!        token, the per-token pipeline for those rows, and the eviction. Rows not named are
        //!        neither read nor written. The caller has validated the rows against the ledger.
        //!        Under a context cache only the decode rows grow pages and advance the
        //!        cache; a non-final chunk advances its admission without publishing, and a final
        //!        chunk publishes the prompt unless its sequence was cancelled while it ran.
        //! @param publishedPrefix Set when a final chunk published its prompt to the context cache.
        //! @param compactRefused Set when the engine refused the phase-8 enqueue with the device healthy (a plugin
        //!        refused phase 8); compact execution is then unavailable for the runtime's life.
        //! @return False when the forward failed or was refused, the context cache refused a
        //!         transition, or the eviction failed.
        //! @throws std::runtime_error when a CUDA call fails.
        bool executeCompactStep(std::vector<CompactRow> const& rows, bool& publishedPrefix, bool& compactRefused);
        //! The compact step's schedule for @p rows, lowered into the context's ragged execution batch. Collects
        //! the resident slots whose M-RoPE row the first chunk of a text prompt initializes, and the decode rows.
        void buildCompactSchedule(std::vector<CompactRow> const& rows, std::vector<int32_t>& mropeInitSlots,
            std::vector<int32_t>& decodeRows);
        //! Under a context cache, move every prompt row's admission on after the forward: a non-final chunk
        //! advances it, a final chunk publishes the prompt (or aborts it when the sequence was cancelled).
        //! @return False when the cache refused a transition.
        bool settlePromptAdmissions(std::vector<CompactRow> const& rows, bool& publishedPrefix);
        //! The per-token pipeline for every row the step sampled: finish decisions, detokenization and stop
        //! matching. @return Whether a decode row was among them.
        bool finishSampledRows(std::vector<CompactRow> const& rows);
        //! Finish execution row @p row with kLength once the benchmark's decode-round cap is reached.
        void applyBenchmarkRoundLimit(int32_t row);
        //! The cache side of a seat that never completes its prefill (the failed and the cancelled
        //! seated admission both end here).
        void abortSeatedAdmission(int32_t row);

        void updateThinkingDoneForToken(int32_t batchIdx, int32_t tokenId);
        //! The per-slot forms of the finish pipeline's three decisions, over the slot's newest
        //! token; the batch forms loop over them. The seated prefill runs them for its one slot, so
        //! its lookahead token is judged by the same code as every decode step's token.
        void updateThinkingDoneForSlot(int32_t batchIdx);
        void updateFinishState(int32_t batchIdx);
        void applyStopMatch(int32_t batchIdx);
        void updateThinkingDone();
        void updateFinishStates();
        void applyStopMatches();
        bool performBatchEvictAndSnapshot();

        //! Whether the sequence at execution row @p batchIdx carries a trajectory request.
        bool isActionSlot(int32_t batchIdx) const noexcept;

        //! Run @p strategy from the next admission on. Only an empty batch changes strategy: the
        //! strategy's per-row state is the residents', so it cannot change under them. A runtime
        //! session selects the strategy for each fill of the batch from the sampling of its first
        //! sequence, as the blocking path selects it for each request.
        void rebindStrategy(DecodingStrategy& strategy);

        LLMRankRuntime& mRuntime;
        DecodingInferenceContext& mContext;
        DecodingStrategy* mStrategy;
        ManagedKVCacheRequest* mManagedRequest;
        DecodingKvHeadroom const& mKvHeadroom;
        DecodingKvHeadroom const& mJumpForwardKvHeadroom;
        cudaStream_t mStream;
        //! Run the cancellation consensus at every step (see the constructor). An admitted slot's
        //! channel lives only on rank zero, and this flag is the one signal every rank computes
        //! identically.
        bool mSynchronizeCancellation;
        //! Per original index: the sequence is a trajectory request. Empty when no sequence is.
        std::vector<bool> mActionSlots;

        int32_t mEndOfChannelId{};
        int32_t mEndOfThinkId{};
        int32_t mStartOfChannelId{};
        int32_t mStartOfThinkId{};
        int32_t mTrajFutureStartId{};
        //! EDGELLM_IGNORE_EOS: every slot generates through EOS (few-layer validation, fixed-output
        //! perf). A slot may also ask for it alone through its request (context.ignoreEos).
        bool mIgnoreEosForAllSlots{};
        bool ignoresEos(int32_t row) const noexcept;
        bool mHasActionRequest{};
    };

    bool handleRequest(LLMGenerationRequest const& request, LLMGenerationResponse& response, cudaStream_t stream,
        bool outputThinkerEmbeddings = false, TokenBroadcastFn tokenBroadcast = nullptr,
        VisionGatherFn visionGather = nullptr, int32_t parallelRank = -1);

    //! Install the rank-local TP collective used by the optional Alpamayo action bridge.
    void setActionKvAllGather(Alpamayo1ActionRunner::KvAllGatherFn kvAllGather)
    {
        if (mActionRunner)
        {
            mActionRunner->setKvAllGather(std::move(kvAllGather));
        }
    }

    //! Everything one request owns between beginGeneration and finishGeneration: the same state
    //! handleRequest used to keep on its stack, packaged so it can be held across the generation
    //! loop. The request must outlive this object; member order is load-bearing (the stream
    //! finalizer references the context and must be destroyed first).
    struct SteppedGeneration
    {
        explicit SteppedGeneration(std::atomic<bool>& activeFlag, Tensor& hostTokenStorage, DeviceFailureState& failure,
            cudaStream_t stream) noexcept
            : guard(activeFlag)
            , mHostTokenStorage(hostTokenStorage)
            , mFailure(failure)
            , mStream(stream)
        {
            context.raggedExecutionBatch.hostTokenIds = std::move(mHostTokenStorage);
        }

        SteppedGeneration(SteppedGeneration const&) = delete;
        SteppedGeneration& operator=(SteppedGeneration const&) = delete;

        ~SteppedGeneration() noexcept;

        InProgressGuard guard;
        DecodingInferenceContext context;
        DecodingStrategy* strategy{nullptr};
        std::optional<ManagedKVCacheRequest> managedKVCacheRequest;
        DecodingKvHeadroom kvHeadroom{};
        //! Headroom of the slots that run with jump-forward; kvHeadroom when none does.
        DecodingKvHeadroom jumpForwardKvHeadroom{};
        std::optional<StreamChannelFinalizer> streamFinalizer;
        bool enableSpecDecode{false};
        bool hasActionRequest{false};
        ActionKvBatchCollector* actionKvCollector{nullptr};
        bool completed{false};

        ManagedKVCacheRequest* managedRequest() noexcept
        {
            return managedKVCacheRequest.has_value() ? &*managedKVCacheRequest : nullptr;
        }

    private:
        Tensor& mHostTokenStorage;
        DeviceFailureState& mFailure;
        cudaStream_t mStream;
    };

    //! Everything handleRequest does before the generation loop: validation, context population,
    //! the context-cache admit, prefill-execution setup, streaming setup, and the request's
    //! prefill itself. Returns null on any refusal (the same conditions that returned false), and
    //! throws where handleRequest threw. @p request must outlive the returned object.
    std::unique_ptr<SteppedGeneration> beginGeneration(LLMGenerationRequest const& request,
        LLMGenerationResponse& response, cudaStream_t stream, bool outputThinkerEmbeddings,
        TokenBroadcastFn tokenBroadcast, VisionGatherFn visionGather, int32_t parallelRank);

    //! Everything handleRequest does after the loop: the drained-batch check, the context-cache
    //! finish, metrics, and response assembly from whatever the loop's harvests left in
    //! completedBatches.
    bool finishGeneration(SteppedGeneration& generation, LLMGenerationRequest const& request,
        LLMGenerationResponse& response, cudaStream_t stream);

    //! Empty while reusable; the first failed CUDA drain otherwise. Safe to read while a request runs.
    std::string unavailableReason() const
    {
        return mDeviceFailure.unavailableReason();
    }

    //! @brief Open the runtime session: one long-lived generation that sequences enter and leave
    //!        one at a time (see RuntimeSession).
    //!
    //! Takes the one-request latch for the session's life, so handleRequest() is refused until the
    //! session is destroyed. Returns null when the runtime is already serving a request, or on the
    //! deployment refusals a request would also have met (an unavailable LoRA reset, a
    //! context-cache request that could not be opened, visual-token pruning under managed reuse).
    //! Deployments whose sequences cannot be seated one at a time (a diffusion backbone, bounded SWA
    //! storage, a context cache without live admission) cannot host a session: check
    //! supportsSeatedAdmission() first.
    //! @p options choose whether the session may run compact steps (see SessionOptions).
    std::unique_ptr<RuntimeSession> openSession(cudaStream_t stream, SessionOptions options = {});

    /*! \brief Set the random seed used when initializing the action diffusion noise trajectory
     *  \param seed Random seed value; has no effect if no action runner is loaded
     */
    void setActionNoiseSeed(int32_t seed) noexcept;

    /*! \brief Enable visual-token pruning for supported VLM prefill execution. */
    void setVisualPrunerConfig(VisualPrunerConfig const& config);

    //! Get LLM prefill stage metrics
    metrics::LLMPrefillMetrics const& getPrefillMetrics() const noexcept
    {
        return mPrefillMetrics;
    }

    //! Get speculative decoding generation stage metrics (only meaningful when draft model is present)
    metrics::SpecDecodeGenerationMetrics const& getSpecDecodeGenerationMetrics() const noexcept
    {
        return mSpecDecodeGenerationMetrics;
    }

    char const* getSpeculativeDecodingStrategyName() const noexcept
    {
        return mDecoderRegistry ? mDecoderRegistry->speculativeDecoderName() : "vanilla";
    }

    char const* getDefaultDecodingStrategyName() const noexcept
    {
        return mDecoderRegistry ? mDecoderRegistry->defaultDecoderName() : "vanilla";
    }

    //! Get vanilla generation stage metrics (only meaningful when no draft model / vanilla path)
    metrics::LLMGenerationMetrics const& getGenerationMetrics() const noexcept
    {
        return mGenerationMetrics;
    }

    //! Get context-cache metrics, or nullopt when the runtime cache is disabled.
    std::optional<ContextCacheMetrics> getContextCacheMetrics() const noexcept;

    //! Get multimodal metrics (returns empty metrics if no multimodal runner)
    metrics::MultimodalMetrics getMultimodalMetrics() const noexcept
    {
        return mVisionRunner ? mVisionRunner->getMultimodalMetrics()
            : mAudioRunner   ? mAudioRunner->getMultimodalMetrics()
                             : metrics::MultimodalMetrics{};
    }

    //! Get the embedding table (for Talker streaming pipeline)
    rt::Tensor const& getEmbeddingTable() const
    {
        return mEmbedding.table;
    }

    //! @brief Get a base model hidden-states buffer for the requested layer index.
    //!
    //! Buffers are owned by the runtime and reused across requests. Layer 0 corresponds to
    //! the post-multimodal input embeddings (backed up before the decode loop reshapes them);
    //! other layer indices correspond to engine-output hidden states (e.g. acceptHiddenLayer
    //! for the Qwen3-Omni Talker, or future MTP layers).
    //!
    //! Lifetime contract:
    //!   - Buffers are sized to {maxRuntimeBatchSize, maxSupportedInputLength, hiddenSize}.
    //!   - Contents are cleared (overwritten) at the start of each handleRequest() call and
    //!     remain valid until the next handleRequest() begins. The buffer is reshaped to
    //!     {activeBatchSize, prefillLength, hiddenSize} for the most recent request — use
    //!     getBaseModelPrefillLength() to query the valid prefill length.
    //!   - The caller is responsible for consuming the data within that window.
    //!
    //! @param layerIdx Layer index. 0 = input embeddings (post-multimodal); other indices are
    //!                 model-specific (e.g. acceptHiddenLayer for Qwen3-Omni Talker).
    //! @return Pointer to the buffer, or nullptr if no buffer is registered for that layer.
    rt::Tensor const* getBaseModelHiddenStates(int32_t layerIdx) const noexcept
    {
        auto it = mHiddenStatesRegistry.find(layerIdx);
        return it != mHiddenStatesRegistry.end() ? it->second : nullptr;
    }

    //! @brief Number of valid prefill tokens in the hidden-states buffers from the most
    //! recent handleRequest() call. Returns 0 if no hidden-states output was requested.
    int32_t getBaseModelPrefillLength() const noexcept
    {
        return mLastPrefillLength;
    }

    //! @brief Per-batch input token IDs from the most recent handleRequest() call.
    //! Cleared at the start of each handleRequest(); valid until the next one begins.
    std::vector<std::vector<int32_t>> const& getBaseModelInputTokenIds() const noexcept
    {
        return mLastInputTokenIds;
    }

    //! @brief Check if draft model is loaded and spec-decode is available
    bool hasDraftModel() const noexcept
    {
        return mDecoderRegistry && mDecoderRegistry->hasSpeculativeDecoder();
    }

    //! @brief Whether this deployment can take boundary admissions via prefillSlotInPlace.
    //!
    //! False for the static refusals prefillSlotInPlace would otherwise throw on mid-request:
    //! diffusion backbones, bounded-SWA deployments, and a context cache without live admission. Probed once at
    //! RequestEngine construction so an unsupported deployment falls back to the blocking path instead of failing
    //! admissions.
    bool supportsSeatedAdmission() const noexcept;

    //! How many of the shared cache managers @p strategy drives: the base one, plus the draft one when the
    //! strategy is speculative. The batch of each of them follows the context's.
    size_t activeCacheManagerCount(DecodingStrategy const& strategy) const noexcept;

    //! Select this runtime's statistics for legacy formatter entry points on the calling thread.
    void selectProfilingTimer() const
    {
        gTimer.select(mProfilingTimer);
    }

    //! @brief The batch dimension the engine was built with: the physical bound on how many
    //! sequences a batch can seat, and therefore on any scheduler's maxBatchSize.
    int32_t maxBatchSize() const noexcept;

private:
    //! The session is the one caller allowed to compose the runtime's request-scoped stages
    //! (validation, admission, seated prefill, eviction) around state it owns for the engine's life.
    friend class RankRuntimeSession;

    //! @brief Prefill one slot of a running batch without recomputing or disturbing the others.
    //!
    //! The sequence is projected to execution row zero for one batch-one prefill. Its ResidentRef
    //! remains attached, so state_indices gathers its stable physical page-table and recurrent rows.
    //! Only transient host bookkeeping and the cache managers' execution lengths are exchanged and
    //! restored; resident state never moves. Inside the projection, after the base prefill,
    //! @p strategy seats the sequence (DecodingStrategy::seatRow) -- a speculative strategy runs its
    //! draft prefill there.
    //!
    //! The execution length projection seeds kvcache_start_index from @p prefillStart. A non-zero
    //! start is a context-cache prefix hit: the
    //! coordinator leased the slot's pages (shared prefix included) and bound its row before this
    //! call, so the pass computes only the tail the cache did not cover.
    //!
    //! A speculative sequence that looks up or publishes its prompt on a Hybrid+spec deployment (@p lookupPolicy not
    //! bypass) runs runHybridSpecPrefill as a batch of one, publishing its checkpoint at the end of the prompt, or
    //! @p replayTailLength tokens before it (the volatile generation-prompt tail; a media prompt's own tail after its
    //! last media run takes its place).
    //!
    //! @throws std::runtime_error if @p strategy cannot seat a row, or if @p prefillStart is non-zero without a
    //!         context cache to have leased the pages behind it.
    bool prefillSlotInPlace(DecodingInferenceContext& context, DecodingStrategy& strategy, int32_t row,
        int32_t prefillStart = 0, SlotSeed const* mediaPayload = nullptr,
        ManagedKVCacheRequest* managedKVCacheRequest = nullptr,
        ContextCacheLookupPolicy lookupPolicy = ContextCacheLookupPolicy::kBypass, int32_t replayTailLength = 0);

    //! Whether this deployment executes compact phase-8 steps.
    struct CompactCapability
    {
        bool compact{false};
        //! Empty when compact is true.
        std::string reason;
    };

    //! How a compact forward ended.
    enum class CompactForwardStatus : uint8_t
    {
        kOk,
        //! Binding the step's shapes and tensors failed; nothing was enqueued.
        kPrepareFailed,
        //! The enqueue failed with no CUDA error pending and the stream healthy: a plugin refused phase 8.
        kEnqueueRefused,
        //! The enqueue failed with a CUDA error pending or the stream in error.
        kDeviceFailed,
    };

    //! Why this deployment cannot run compact steps whatever kernels its layers have, or empty: features compact
    //! steps do not drive, a per-layer embedding cache smaller than one compact step's rows, and engines that cannot
    //! accept a phase-8 shape on both profiles.
    std::string compactPreconditionFailure(DecodingInferenceContext const& context) const;

    //! @brief The deployment's compact capability: compactPreconditionFailure(), then compactExecutionGap() of the
    //!        engine config on this GPU, then whether the engine was built for compact execution
    //!        (LLMEngineConfig::compactExecution). Logs the outcome once.
    CompactCapability resolveCompactCapability(DecodingInferenceContext const& context) const;

    //! @brief Allocate the per-slot media reservation of a compact multimodal deployment on its first
    //!        compact session, or forget every slot's media on a later one.
    //! @throws std::runtime_error naming the reservation's layout and size when it cannot be allocated.
    void prepareCompactMediaReservation();

    //! @brief What a compact session needs beyond the capability: the media reservation and the dedicated compact
    //!        context. When either cannot be allocated, compact execution is recorded as unavailable for the
    //!        runtime's life with a reason naming the failure (one WARNING), the reservation is released, and the
    //!        session serves single-cohort steps.
    //! @return Whether the session executes compact steps.
    //! @throws what the allocation threw when it left @p stream in error.
    bool prepareCompactExecution(cudaStream_t stream);

    //! The batch-one MRope staging buffer the admission preprocess writes, allocated on first use;
    //! empty for a deployment without MRope.
    OptionalOutputTensor admissionMropeStage();

    //! Copy one [maxKVCacheCapacity, rotaryDim] FP32 M-RoPE table from device memory @p source into resident slot
    //! @p slot's row of the resident M-RoPE cache, on @p stream.
    void copyMropeRowToResidentSlot(void const* source, int32_t slot, cudaStream_t stream);

    //! The intent stage of a deferred media admission: image digests, the
    //! encoders' text preprocessing (media-token expansion) and MRope into mAdmissionMropeStage. The
    //! vision encoder does not run. An audio runner's whole preprocess runs here, as it has no
    //! text-only mode; for the runners whose preprocess runs their encoder, the audio encoder runs here too.
    bool preprocessDeferredAdmissionMedia(LLMGenerationRequest const& request, SlotSeed& seed, cudaStream_t stream);

    //! Whether the vision runner can lay out @p request's prompt only by running its encoder
    //! (MultimodalRunner::tokenLayoutNeedsEncoder); such media is never deferred.
    bool mediaLayoutNeedsEncoder(LLMGenerationRequest const& request) const noexcept;

    //! Restore from the encoder embedding cache, or run, the encoders of the request
    //! preprocessDeferredAdmissionMedia just staged, copy their rows from @p prefillStart on into
    //! @p slot's reservation, and copy the staged MRope table into the slot's resident row.
    //! @throws std::runtime_error when an encoder fails or its output does not cover the prompt.
    void encodeDeferredAdmissionMedia(LLMGenerationRequest const& request, std::vector<Hash128> const& imageDigests,
        int32_t slot, std::vector<int32_t> const& promptTokenIds, int32_t prefillStart, cudaStream_t stream);

    //! Map the media positions of a compact step's prompt chunks to reservation rows and upload them
    //! to mMultimodalIndices ([1, T]); decode and text tokens get -1.
    //! @return Whether the step embeds any media row.
    bool uploadCompactMediaIndices(RaggedExecutionBatch const& batch, cudaStream_t stream);

    //! @brief Embed, bind and enqueue the compact step context.raggedExecutionBatch holds (built by
    //!        the compact builder). Leaves the base cache manager's batch at the step's sequence
    //!        count; the caller restores it. Resident M-RoPE rows of @p mropeInitSlots are written
    //!        with the text-only table first. Prompt chunks of staged media sequences embed their
    //!        media rows (and deepstack features) from the slot reservations.
    CompactForwardStatus runCompactForward(
        DecodingInferenceContext& context, std::vector<int32_t> const& mropeInitSlots);

    //! @brief Run one request's decode loop to completion.
    bool runGenerationLoop(DecodingInferenceContext& context, DecodingStrategy& decodingStrategy,
        ManagedKVCacheRequest* managedRequest, LLMGenerationRequest const& request,
        DecodingKvHeadroom const& kvHeadroom, DecodingKvHeadroom const& jumpForwardKvHeadroom, cudaStream_t stream);

    void initializeFromEngineDir(DeploymentOptions const& options, cudaStream_t stream, ParallelMapping const& mapping,
        tokenizer::Tokenizer& tokenizer, chat_template::ChatTemplate const& chatTemplate);

    void initializeCommon(ModelArtifacts&& artifacts, DeploymentOptions const& options, cudaStream_t stream,
        ParallelMapping const& mapping, tokenizer::Tokenizer& tokenizer,
        chat_template::ChatTemplate const& chatTemplate);

    //! @brief Capture a CUDA graph on the base executor for the default (no-adapter)
    //! state, then one additional graph per registered LoRA adapter. Returns the
    //! logical AND of all captures — any single failure flips the aggregate to
    //! false but capture continues for remaining adapters (graceful degrade).
    bool captureBaseGraphWithLoraFanout(InferenceDims const& dims, cudaStream_t stream);

    //! @brief Build the strategy runtime reference bundle after common resources are allocated.
    void buildDecodingRuntimeContext();

    //! Why a guided request does not get the jump-forward it asked for. Each reason warns once per runtime.
    enum class JumpForwardRefusal : uint8_t
    {
        kDiffusionBackbone,
        kVisionBlockAttention,
        kSingleTokenChunk,
        kNonGreedySampling,
        kNoVerifyHeadroom,
        kSpeculativeInFlightBatching,
        kCount
    };

    //! Whether a guided request that asks for jump-forward gets it on `strategy`, the decoder the registry selected
    //! for its request; `nonGreedySampling` says whether the request itself samples non-greedily. The vanilla decoder,
    //! also as a speculative deployment's fallback, needs a context chunk wider than one token and no vision-block
    //! attention. Every speculative decoder needs jump-forward limits (SpecDecodeConfig::maxJumpTokens > 0), and one
    //! that supports lossless sampling also needs a greedy request; the others decode every request greedily. A
    //! speculative decoder grafts only on the blocking path: a session (`inFlightBatching`) reserves the plain KV
    //! headroom and seats rows between rounds, which the grafting rounds do not account for yet.
    bool jumpForwardRequested(GuidedDecodingParams const& params, DecodingStrategy const& strategy,
        bool nonGreedySampling, bool inFlightBatching) const noexcept;

    //! Build the guided decoder's jump-forward tables and, the first time `strategy` is the vanilla decoder, the
    //! execution context its catch-up steps run on; called for every request set up with jump-forward.
    void prepareJumpForwardExecution(DecodingStrategy const& strategy, cudaStream_t stream);
    mutable std::bitset<static_cast<size_t>(JumpForwardRefusal::kCount)> mJumpForwardRefusalsLogged;
    bool mJumpForwardContextPrepared{false};

    std::atomic<bool> mHandleRequestInProgress{false}; //!< Defensive overlap gate, not a thread-safety contract.
    DeviceFailureState mDeviceFailure;
    //! Resolved by the first session that may run compact steps; later sessions reuse it, so a scheduler that
    //! resolved its policy against it sees the same capability on every session it reopens. A failed compact setup
    //! (prepareCompactExecution) or a refused phase-8 step replaces it with an unavailable one, so every later
    //! session is single-cohort.
    std::optional<CompactCapability> mCompactCapability;

    //! Broadcast a sampled int32 token buffer when a parallel request supplies a callback.
    bool broadcastInt32(void* gpuBuffer, int32_t count, cudaStream_t stream);

    //! Broadcast root-rank cancellation decisions at a decode-iteration boundary so all ranks
    //! preserve identical collective and batch-eviction order.
    bool synchronizeCancellationStates(DecodingInferenceContext& context);

    rt::Tensor mSharedExecContextMemory{}; //!< Shared device memory for all execution contexts
    int32_t mMaxRuntimeBatchSize{1};       //!< Maximum runtime batch size

    std::shared_ptr<timer::Timer> mProfilingTimer{std::make_shared<timer::Timer>()};
    RuntimeBenchmarkOptions mBenchmarkOptions{};
    DeploymentConfig mDeployment{};                    //!< Parsed base+draft configs + consolidated strategy settings
    std::unique_ptr<EngineExecutor> mBaseExecutor;     //!< Base model TRT wrapper
    std::unique_ptr<SharedResources> mSharedResources; //!< KV caches / RoPE / LoRA / context memory
    //! Declared after SharedResources so context-cache ownership is released before the physical buffers.
    std::unique_ptr<ContextCacheCoordinator> mContextCache;
    //! Declared after the context cache so SWA work drains before its lease-owned page references are released.
    std::unique_ptr<BoundedSwaKVPageManager> mBoundedSwaKVPageManager;

    //! Admission-scoped MRope staging ([1, maxKVCacheCapacity, rotaryDim]): the admission
    //! preprocess writes here until the batch-one execution copies it to active-row scratch.
    //! The ordinary ragged scatter then publishes it to the admitted ResidentRef. Allocated on the first multimodal
    //! admission of an MRope deployment; other deployments never pay for it.
    Tensor mAdmissionMropeStage;
    //! Per-slot encoder outputs of compact multimodal sessions; null until a compact session opens on
    //! a deployment with an encoder.
    std::unique_ptr<CompactMediaReservation> mCompactMedia;
    std::unique_ptr<PipelineIO> mPipelineIO; //!< Per-pipeline I/O tensors
    //! Scratch [maxSeq, baseOutputHiddenDim] used to shift baseHiddenStates down one row when folding a reused
    //! checkpoint boundary into the draft prefill. Allocated only when futureDependencyTokens > 0.
    rt::Tensor mBoundaryFoldScratch;
    bool mHybridSpecContextReuseDeployment{};
    //! How many future-dependency tokens the speculative mode requires at the checkpoint boundary. MTP and EAGLE use 1
    //! (the successor token drives the draft slot); DFlash/JetSpec/DSpark use 0 (no boundary fold needed).
    int32_t mSpecFutureDependencyTokens{0};
    //! baseHiddenStates' max-sequence rows. The fold writes chunkLength + 1 rows, so it needs one spare row on top of
    //! the chunk it shifts; runHybridSpecPrefill checks the chunk against this bound before reshaping.
    int32_t mBoundaryFoldMaxRows{0};
    TensorMap mBaseTensorMap;                  //!< Base engine binding map
    std::filesystem::path mCheckpointDir;      //!< Provider checkpoint used during startup weight loading
    std::filesystem::path mDraftCheckpointDir; //!< Separate provider draft checkpoint, empty for integrated drafts
    LogitBias mLogitBias;                      //!< Runtime-owned resources that outlive decoding objects borrowing them
    std::unique_ptr<DecodingRuntimeContext> mDecodingRuntimeContext;
    std::unique_ptr<DecoderRegistry> mDecoderRegistry;
    std::unique_ptr<StepPreparer> mStepPreparer;          //!< Per-step sequence preprocessor
    std::unique_ptr<EmbeddingPreprocessor> mEmbeddingPre; //!< Embedding-lookup preprocessor
    std::unique_ptr<VisualTokenPruner> mVisualPruner;     //!< Visual-token pruner (optional)
    std::shared_ptr<PleIoBudget> mPleIoBudget;
    std::unique_ptr<Gemma4EmbeddingPreprocessor> mGemma4Ple;   //!< Gemma4 PLE token-identity preprocessor (optional)
    std::unique_ptr<Qwen4ExpNgramPreprocessor> mQwen4ExpNgram; //!< Qwen4-Exp n-gram row-ID preprocessor (optional)
    //! Base-engine deepstack binding (nullptr when the base engine was built
    //! without deepstack features). Swaps between `io.deepstackEmbeds[i]`
    //! (prefill) and the shared `zeroDeepstackBroadcast` (all other phases).
    std::unique_ptr<DeepstackBinding> mDeepstack;

    std::unique_ptr<MultimodalRunner> mVisionRunner{nullptr};      //!< Vision multimodal runner (optional)
    std::unique_ptr<MultimodalRunner> mAudioRunner{nullptr};       //!< Audio multimodal runner (optional)
    std::unique_ptr<Alpamayo1ActionRunner> mActionRunner{nullptr}; //!< Action/diffusion head runner (optional)
    std::unique_ptr<ActionKvBatchCollector> mActionKvBatchCollector;
    tokenizer::Tokenizer* mTokenizer{nullptr};                 //!< Shared tokenizer owned by RuntimeCoordinator
    chat_template::ChatTemplate const* mChatTemplate{nullptr}; //!< Shared renderer owned by RuntimeCoordinator

    //! Grammar-constrained decoding state; inert unless a request asks for it.
    GuidedDecoder mGuidedDecoder;
    std::unique_ptr<EncoderEmbeddingCache> mEncoderEmbeddingCache; //!< Content-addressed encoder output cache
    std::string mEmptyLoraWeightsName{""};                         //!< Empty LoRA weights name for default case

    // Pre-define key runtime GPU tensors and initialize them during construction.
    // [1] Runtime-local I/O tensors. Embedding table is shared between base and draft models.
    // Core per-pipeline tensors (inputsEmbeds, outputLogits, deepstackEmbeds, baseHiddenStates,
    // draftHiddenStatesIn/Out, contextLengths, mropeCosSin) live on `mPipelineIO`.
    EmbeddingData mEmbedding; //!< Embedding table [vocabSize, hiddenSize] and optional FP8 scales
    rt::Tensor mIdsInput;     //!< Input token IDs (used for embedding lookup)

    // [2] Sampling workspace and output tensors that used across all the sampling operations.
    rt::Tensor mSamplingWorkspace;
    rt::Tensor mSamplingIndices;
    rt::Tensor mSamplingScores;
    rt::Tensor mSamplingRows;          //!< [maxBatch] SamplingRow bytes: each row's parameters and uniform for the step
    rt::Tensor mBaseVocabMappingTable; // Vocab mapping table for base model reduced vocab (empty if not used)

    // [3] Batch eviction support tensors.
    rt::Tensor mDeviceBatchMapping;
    rt::Tensor mDeviceCancellationStates;
    rt::Tensor mHostCancellationStates; //!< Pinned host staging paired with mDeviceCancellationStates.

    // [4] Host pinned memory tensors for optimized CPU-GPU memory transfers
    rt::Tensor mHostRaggedTokenIds;      //!< Runtime-owned token snapshot lent to each request's ragged batch
    rt::Tensor mHostDecoderTokenIds;     //!< Pinned staging for specialized decoder token inputs
    rt::Tensor mHostSelectedTokenIds;    //!< Host pinned memory for selected token IDs from sampling
    rt::Tensor mHostOutputSpaceIds;      //!< Host pinned copy of the sampled indices taken before reduced-vocab remap
    rt::Tensor mHostSamplingRows;        //!< Host pinned staging for mSamplingRows
    rt::Tensor mHostReuseKVCacheLengths; //!< Host pinned memory for reuse KV cache lengths
    // [5] Multimodal support tensors for audio/image token indexing
    rt::Tensor mMultimodalIndices;     //!< GPU [batchSize, seqLen] multimodal embedding indices
    rt::Tensor mHostMultimodalIndices; //!< Host pinned [batchSize, seqLen] staging for CPU-computed indices
    rt::Tensor mImageDigestRows; //!< Host pinned [kMaxImageDigestRows, 16] row scratch for the image content digest

    // [6] Logprobs support tensors. Non-Diffusion paths allocate at construction to preserve existing behavior.
    // DiffusionGemma allocates lazily when a request asks for numLogprobs because its row count is B * canvasLen.
    rt::Tensor mDeviceLogprobsValues;  //!< GPU [logprobsRows, kMaxLogprobsK] top-K log-prob values
    rt::Tensor mDeviceLogprobsIndices; //!< GPU [logprobsRows, kMaxLogprobsK] top-K token indices
    rt::Tensor mHostLogprobsValues;    //!< CPU pinned D2H target for mDeviceLogprobsValues
    rt::Tensor mHostLogprobsIndices;   //!< CPU pinned D2H target for mDeviceLogprobsIndices
    rt::Tensor mGatheredLogits;        //!< GPU [logprobsRows, vocabSize] gathered accepted rows (EAGLE/MTP/DFlash)
    int32_t mLogprobsMaxBatchDim{0};   //!< Max rows fed to extractTopKLogprobs; used only for workspace sizing

    // [8] Base model hidden states portal (Qwen3-Omni audio generation, future MTP).
    //     The actual buffers (engine-output and prefill-embeddings backup) live on
    //     PipelineIO so they can be wired into the engine TensorMap. The registry
    //     below holds non-owning pointers into those buffers, populated per request,
    //     plus the per-request prefill length and the raw input token ids — all are
    //     transient request-scoped state, not pipeline tensors. See
    //     getBaseModelHiddenStates() for the lifetime contract.
    std::unordered_map<int32_t, rt::Tensor const*> mHiddenStatesRegistry; //!< Per-request layer→buffer map
    int32_t mLastPrefillLength{0};                                        //!< Valid prefill length in buffers
    std::vector<std::vector<int32_t>> mLastInputTokenIds;                 //!< Per-batch input token IDs

    ParallelMapping mMapping{};         //!< Fully-resolved parallel coordinates for this rank.
    TokenBroadcastFn mTokenBroadcast{}; //!< Optional sampled-token synchronization callback.
    VisionGatherFn mVisionGather{};     //!< Optional visual-shard synchronization callback.
    int32_t mParallelRank{-1};          //!< Engine parallel rank for the active request (-1 = standalone)

    //! @brief Allocate or grow logprobs tensors/workspace to cover a request that enabled numLogprobs.
    void ensureLogprobsCapacity(int32_t logprobsRows, int32_t topK);

    // Key functions to drive the runtime, defined in a consumer-producer pattern.
    // Consume tokenized IDS as input and produce hidden states for the whole sequence and first generated token.
    //! @throws std::runtime_error if a CUDA error occurs
    bool runBaseModelPrefill(DecodingInferenceContext& context, ManagedKVCacheRequest* managedKVCacheRequest = nullptr,
        bool sampleOutput = true, int32_t admissionSequenceIndex = -1);

    //! Hybrid speculative endpoint-reuse prefill. Publishes at the stable predecessor boundary. When
    //! futureDependencyTokens > 0 (MTP, EAGLE) the reused checkpoint's boundary hidden is folded into the draft prefill
    //! on a cache hit; when 0 (DFlash, JetSpec, DSpark) the fold is skipped and the draft prefill runs directly.
    //! Only reachable when shouldUseHybridSpecEndpointReuse() already established that the cache is live for this
    //! request, so lookup and publication are both enabled here by construction.
    //! With @p seatedRow, the batch is the batch-one projection of a sequence seated at that row of a running batch
    //! (prefillSlotInPlace): its draft prefill seats it (DecodingStrategy::seatRow), and the cache knows it by that
    //! row.
    bool runHybridSpecPrefill(DecodingInferenceContext& context, DecodingStrategy& strategy,
        ManagedKVCacheRequest& managedKVCacheRequest, int32_t seatedRow = -1);
    //! Tokens of @p inputIds after its last media placeholder: where a Hybrid+spec media prompt's checkpoint stops.
    int32_t mediaReplayTailLength(std::vector<int32_t> const& inputIds) const;

    //! Validate request shape/runtime compatibility.
    bool validateRequestConfig(LLMGenerationRequest const& request);

    //! The seed a sequence samples with: the item's own, else the request's, else kDefaultSamplingSeed.
    //! Every rank resolves the same value, so no broadcast is needed.
    static uint64_t resolveSamplingSeed(
        LLMGenerationRequest const& request, LLMGenerationRequest::Request const& item) noexcept;

    //! Whether a sequence draws its sampling uniforms from its seed and absolute position: it gave a
    //! seed (the item's or the request's), or a lossless speculative decoder replays its draws. Other
    //! sequences draw from the shared counter stream keyed on the step and row (decoder_utils).
    static bool usesRequestStableSampling(LLMGenerationRequest const& request,
        LLMGenerationRequest::Request const& item, bool losslessSamplingSpecDecode) noexcept;

    //! The logprobs width a request is served at: clamped to kMaxLogprobsK (with a warning), and a
    //! negative width means disabled, like zero.
    static int32_t clampLogprobsWidth(int32_t requested) noexcept;

    //! The top-k a request is served at: clamped to kMaxSamplingTopK (with a warning), as
    //! TensorRT-LLM does for this kernel family. Negative values pass through (disabled, like zero).
    static int64_t clampSamplingTopK(int64_t requested) noexcept;

    //! The request's sampling parameters as one row's: top-k clamped to the sampler's cap, and a
    //! negative top-k disables it, like zero.
    static RowSamplingParams rowSamplingParamsFor(LLMGenerationRequest const& request)
    {
        return makeRowSamplingParams(request.temperature, clampSamplingTopK(request.topK), request.topP);
    }

    //! Prepare per-request runtime state for models built with multimodal support.
    //! Runs multimodal preprocessing when audio or vision inputs are present.
    //! For text-only requests on MRope-based multimodal models, restores text-only RoPE state
    //! and clears stale multimodal request state.
    //! @brief Stage a joining request's media through the full multimodal preprocess -- encoders,
    //!        encoder-embedding cache, media-token expansion, MRope -- against a temporary batch-1
    //!        context, leaving the resident batch untouched.
    //!
    //! On success the seed carries the expanded prompt tokens and references to the runners'
    //! output embeddings; MRope cos/sin lands in mAdmissionMropeStage for the seating to copy
    //! into the resident cache's new row.
    //! @param request A single-sequence request (the joining head, already chat-formatted).
    bool preprocessAdmissionMedia(LLMGenerationRequest const& request, SlotSeed& seed, cudaStream_t stream);

    //! @param mropeCosSinOverride When set, MRope cos/sin output is written here instead of the
    //!        batch-resident mPipelineIO->mropeCosSin. The admission preprocess uses this: it runs
    //!        mid-generation against a temporary batch-1 context, and writing the resident cache
    //!        would clobber the running batch's rows.
    bool multiModalRuntimePreprocess(LLMGenerationRequest const& request, DecodingInferenceContext& context,
        cudaStream_t stream, OptionalOutputTensor mropeCosSinOverride = std::nullopt);

    //! Content identity of every image of `request`, per sequence and in request order, for
    //! @ref DecodingInferenceContext::imageContentDigests; empty when any one image has no identity.
    std::vector<std::vector<Hash128>> buildImageContentDigests(
        LLMGenerationRequest const& request, cudaStream_t stream);

    //! Bind the named LoRA adapter's weights (the base weights for an empty name) into the base
    //! tensor map. A no-op for deployments built without LoRA support. False when the switch fails.
    bool selectLoraWeights(std::string const& loraWeightsName);

    //! The deployment's media placeholder token ids (image, then audio; each only when the model
    //! has one), in the order the context cache hashes media runs by. Empty for a text-only model.
    std::vector<int32_t> mediaTokenIds() const;

    //! Initialize reusable vision DP metadata before accepting requests.
    void initializeVisionDataParallel();

    //! Encode one contiguous image shard and reconstruct the full visual tensors before TP2 prefill.
    bool runJointVisionDataParallel(LLMGenerationRequest const& request, VisionDpPlan const& plan,
        DecodingInferenceContext& context, std::vector<std::vector<int32_t>>& batchedInputIds, cudaStream_t stream,
        OptionalOutputTensor mropeCosSinOut);

    // Consume batched input ids and produce the padded input ids and input lengths for the prefill step. When
    // contextCachePrefillStarts is set, prefill resumes from each sequence's reused-prefix boundary.
    //! @throws std::runtime_error if the input is malformed
    bool setUpForPrefillExecution(DecodingInferenceContext& context, DecodingStrategy& strategy,
        std::vector<int32_t> const* contextCachePrefillStarts = nullptr);

    // Batch eviction support
    //! @brief Perform batch eviction
    //! @param context Inference context
    //! @return True on success, false on failure
    //! @throws std::runtime_error if a CUDA error occurs
    bool performBatchEvict(
        DecodingInferenceContext& context, DecodingStrategy& strategy, ManagedKVCacheRequest* managedKVCacheRequest);

    // Stage-specific metrics
    metrics::LLMPrefillMetrics mPrefillMetrics;
    metrics::SpecDecodeGenerationMetrics mSpecDecodeGenerationMetrics;
    metrics::LLMGenerationMetrics mGenerationMetrics; //!< Vanilla generation metrics (used when no spec-decode)
};

} // namespace rt
} // namespace trt_edgellm
