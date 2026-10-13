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
#include "common/hashUtils.h"
#include "common/tensor.h"
#include "multimodal/common/multimodalRunner.h"
#include "profiling/metrics.h"
#include "profiling/timer.h"
#include "runtime/config/deploymentConfig.h"
#include "runtime/config/llmEngineConfig.h"
#include "runtime/decoding/decoderRegistry.h"
#include "runtime/decoding/guidedDecoder.h"
#include "runtime/decoding/logitBias.h"
#include "runtime/exec/engineExecutor.h"
#include "runtime/exec/tensorMap.h"
#include "runtime/features/deepstackBinding.h"
#include "runtime/generationBoundary.h"
#include "runtime/llmRuntimeUtils.h"
#include "runtime/managedKVCacheRequest.h"
#include "runtime/modelArtifacts.h"
#include "runtime/multiDevice/parallelConfig.h"
#include "runtime/preprocess/embeddingPreprocessor.h"
#include "runtime/preprocess/gemma4EmbeddingPreprocessor.h"
#include "runtime/preprocess/stepPreparer.h"
#include "runtime/preprocess/visualTokenPruner.h"
#include "runtime/state/contextCache/contextCacheConfig.h"
#include "runtime/state/contextCache/contextCacheMetrics.h"
#include "runtime/state/contextCache/encoderEmbeddingCache.h"
#include "runtime/state/decodingInferenceContext.h"
#include "runtime/state/pipelineIO.h"
#include "runtime/state/sharedResources.h"
#include "runtime/state/systemPromptKVCache.h"
#include "runtime/streaming.h"
#include "tokenizer/tokenizer.h"
#include <atomic>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <tuple>
#include <unordered_map>
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
class RuntimeStepper;
class SteppedRequest;

class LLMRankRuntime
{
public:
    //! Callback type for broadcasting GPU int32 buffers across parallel ranks.
    //! Rank 0 sends; peer ranks receive.
    using TokenBroadcastFn = std::function<bool(void* buffer, int32_t count, cudaStream_t stream)>;

    /*!
     * @brief Construct one rank-local runtime from a fully-resolved parallel mapping.
     * Preferred entry point — carries tensor/context/expert coordinates so future
     * CP/EP support needs no further constructor changes.
     */
    LLMRankRuntime(std::string const& engineDir, std::string const& multimodalEngineDir,
        std::unordered_map<std::string, std::string> const& loraWeightsMap,
        std::optional<SpecDecodeDraftingConfig> const& draftingConfig, cudaStream_t stream,
        ParallelMapping const& mapping, tokenizer::Tokenizer& tokenizer,
        chat_template::ChatTemplate const& chatTemplate, ContextCacheConfig const& contextCacheConfig,
        std::string const& checkpointDir, std::string const& draftCheckpointDir);

    LLMRankRuntime(ModelArtifacts&& artifacts, std::string const& engineDir, std::string const& multimodalEngineDir,
        std::unordered_map<std::string, std::string> const& loraWeightsMap,
        std::optional<SpecDecodeDraftingConfig> const& draftingConfig, cudaStream_t stream,
        ParallelMapping const& mapping, tokenizer::Tokenizer& tokenizer,
        chat_template::ChatTemplate const& chatTemplate, ContextCacheConfig const& contextCacheConfig);

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
    class GenerationSession final : public GenerationBoundary
    {
        //! The stepped control plane's execution facade drives the session through the typed
        //! stage methods and reads the context it wraps; nothing else reaches in.
        friend class RuntimeStepper;
        friend class SteppedRequest;

    public:
        GenerationSession(LLMRankRuntime& runtime, DecodingInferenceContext& context, DecodingStrategy& strategy,
            ManagedKVCacheRequest* managedRequest, LLMGenerationRequest const& request,
            DecodingKvHeadroom const& kvHeadroom, cudaStream_t stream, bool boundarySchedulingActive);

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

        //! @brief Join one new sequence to this running batch and prefill it, without touching the
        //!        sequences in flight.
        //!
        //! Admission plus seating: the context grows a slot (appendSlot), the session's own
        //! per-slot state grows with it, and prefillSlotInPlace runs the new slot's prompt as a
        //! seated batch-1 pass. On a failed prefill the slot is marked terminal with kError so the
        //! next eviction files its result; the batch's other sequences are unaffected either way.
        //!
        //! Whether the arriving request may share this batch at all (sampling parameters, adapter,
        //! step budget) is BatchCompatibility's question, answered before a seed is built.
        //!
        //! @throws std::runtime_error for seeds appendSlot rejects, and for deployments
        //!         prefillSlotInPlace refuses; nothing is modified on those paths.
        AdmitDecision admitSequence(SlotSeed seed) override;

        //! @brief Move out the results of finished sequences whose original index is >= @p firstIndex.
        //!
        //! Sequences admitted mid-flight carry indices above the founding request's range, so this
        //! is how their results leave the batch without appearing in the founding caller's
        //! response. Harvested at every boundary by whoever drives the loop.
        std::unordered_map<int32_t, BatchResult> takeCompletedAtOrAbove(int32_t firstIndex) override;

        AdmitDecision admitRequest(
            LLMGenerationRequest const& request, int32_t originalIndex, RequestId requestId) override;

        LLMGenerationResponse materializeResult(
            BatchResult const& result, std::vector<std::string> const& stopStrings) const override;

        int32_t residentCount() const override
        {
            return mContext.activeBatchSize;
        }

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
        };

        //! Stage one: validate the request shape and produce the intent (tokenize, or run the
        //! encoder preprocess for media). Throws on requests the batch rejects outright; resident
        //! state is untouched on every path.
        AdmissionIntent buildAdmissionIntent(
            LLMGenerationRequest const& request, int32_t originalIndex, RequestId requestId);

        //! Stage two: reserve what the sequence needs before it owns a slot -- the generate-budget
        //! check, and under a context cache the page lease (setting the intent's prefillStart).
        //! A refusal leaves the batch untouched; the lease is the one reservation, and its
        //! rollback (retractSequenceAdmission) belongs to stage three's failure paths.
        AdmitDecision reserveAdmission(AdmissionIntent& intent);

        //! Stage three: seat and commit. appendSlot is the commit point -- before it, failure
        //! retracts the lease and nothing was resident; from it on, every failure files as the
        //! new slot's terminal kError result through the ordinary harvest, so the batch never
        //! holds a half-admitted sequence.
        AdmitDecision seatAdmission(AdmissionIntent intent);

        //! What seatSlot leaves for prefillSeated: the committed slot plus everything the seated
        //! prefill still needs from the seed. Valid until the next admission overwrites the
        //! admission staging buffers the media references point into -- the seated prefill must
        //! run before another admission's preprocess.
        struct PendingSeat
        {
            int32_t slot{-1};
            int32_t prefillStart{0};
            int32_t rawInputLength{0};
            bool hasMedia{false};
            SlotSeed mediaPayload;
        };

        //! The first half of seating: appendSlot (the commit point) plus the slot's MRope cos/sin
        //! row. Throws propagate after retracting a context-cache lease, exactly as seatAdmission
        //! did. The stepped control plane calls this from admit(); the seated prefill becomes the
        //! next prefill tick.
        PendingSeat seatSlot(AdmissionIntent intent);

        //! The second half: the seated batch-1 prefill, the context-cache ledger finalize, and the
        //! lookahead-token pipeline. On failure the slot is terminal from birth, exactly as in the
        //! fused path.
        AdmitDecision prefillSeated(PendingSeat pending);

        void updateThinkingDoneForToken(int32_t batchIdx, int32_t tokenId);
        void updateThinkingDone();
        void updateFinishStates();
        bool performBatchEvictAndSnapshot();

        LLMRankRuntime& mRuntime;
        DecodingInferenceContext& mContext;
        DecodingStrategy& mStrategy;
        ManagedKVCacheRequest* mManagedRequest;
        LLMGenerationRequest const& mRequest;
        DecodingKvHeadroom const& mKvHeadroom;
        cudaStream_t mStream;
        //! True when a boundary hook drives this loop (in-flight batching). The cancellation
        //! consensus must then run regardless of the founder's channels: an admitted slot's
        //! channel lives only on rank zero, and this flag is the one signal every rank computes
        //! identically.
        bool mBoundarySchedulingActive;

        int32_t mEndOfChannelId{};
        int32_t mEndOfThinkId{};
        int32_t mStartOfChannelId{};
        int32_t mStartOfThinkId{};
        int32_t mTrajFutureStartId{};
        bool mIgnoreEos{};
        bool mHasActionRequest{};
    };

    bool handleRequest(LLMGenerationRequest const& request, LLMGenerationResponse& response, cudaStream_t stream,
        bool outputThinkerEmbeddings = false, TokenBroadcastFn tokenBroadcast = nullptr, int32_t parallelRank = -1,
        GenerationBoundaryHook const& boundaryHook = {});

    //! Everything one request owns between beginGeneration and finishGeneration: the same state
    //! handleRequest used to keep on its stack, packaged so a stepped control plane can hold a
    //! request open across ticks. The founding request must outlive this object; member order is
    //! load-bearing (the stream finalizer references the context and must be destroyed first).
    struct SteppedGeneration
    {
        //! Releases the runtime's one-request-at-a-time latch, whatever path retires the request.
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

            std::atomic<bool>& mActive;
        };

        explicit SteppedGeneration(std::atomic<bool>& activeFlag, Tensor& hostTokenStorage) noexcept
            : guard(activeFlag)
            , mHostTokenStorage(hostTokenStorage)
        {
            context.raggedExecutionBatch.hostTokenIds = std::move(mHostTokenStorage);
        }

        ~SteppedGeneration() noexcept
        {
            // Return ownership before destroying the context and releasing the request latch.
            mHostTokenStorage = std::move(context.raggedExecutionBatch.hostTokenIds);
        }

        SteppedGeneration(SteppedGeneration const&) = delete;
        SteppedGeneration& operator=(SteppedGeneration const&) = delete;

        InProgressGuard guard;
        DecodingInferenceContext context;
        DecodingStrategy* strategy{nullptr};
        std::optional<ManagedKVCacheRequest> managedKVCacheRequest;
        DecodingKvHeadroom kvHeadroom{};
        std::optional<StreamChannelFinalizer> streamFinalizer;
        bool enableSpecDecode{false};
        bool hasActionRequest{false};

        ManagedKVCacheRequest* managedRequest() noexcept
        {
            return managedKVCacheRequest.has_value() ? &*managedKVCacheRequest : nullptr;
        }

    private:
        Tensor& mHostTokenStorage;
    };

    //! Everything handleRequest does before the generation loop: validation, context population,
    //! the context-cache admit, prefill-execution setup, streaming setup, and the founding prefill
    //! itself. Returns null on any refusal (the same conditions that returned false), and throws
    //! where handleRequest threw. @p request must outlive the returned object.
    std::unique_ptr<SteppedGeneration> beginGeneration(LLMGenerationRequest const& request,
        LLMGenerationResponse& response, cudaStream_t stream, bool outputThinkerEmbeddings,
        TokenBroadcastFn tokenBroadcast, int32_t parallelRank, RequestId founderRequestId = 0);

    //! Everything handleRequest does after the loop: the drained-batch check, the context-cache
    //! finish, metrics, and response assembly from whatever the loop's harvests left in
    //! completedBatches.
    bool finishGeneration(SteppedGeneration& generation, LLMGenerationRequest const& request,
        LLMGenerationResponse& response, cudaStream_t stream);

    /*!
     * @brief Generate and save system prompt KV cache (public API matching standard runtime signature)
     * @param prompt The system prompt to generate the KVCache
     * @param loraWeightsName The name of the LoRA weights
     * @param stream The CUDA stream used for the generation
     * @return True if the KVCache is generated and saved successfully, false otherwise
     * @throws std::runtime_error if a CUDA operation fails
     */
    bool genAndSaveSystemPromptKVCache(
        std::string const& prompt, std::string const& loraWeightsName, cudaStream_t stream);

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
    //! draft (speculative) engines, diffusion backbones, and bounded-SWA deployments. Probed once at RequestEngine
    //! construction so an unsupported deployment falls back to the blocking path instead of failing admissions.
    bool supportsSeatedAdmission() const noexcept;

    //! @brief The batch dimension the engine was built with: the physical bound on how many
    //! sequences a batch can seat, and therefore on any scheduler's maxBatchSize.
    int32_t maxBatchSize() const noexcept;

private:
    //! @brief Prefill one slot of a running batch without recomputing or disturbing the others.
    //!
    //! The sequence is projected to execution row zero for one batch-one prefill. Its ResidentRef
    //! remains attached, so state_indices gathers its stable physical page-table and recurrent rows.
    //! Only transient host bookkeeping is exchanged and restored; resident state never moves.
    //!
    //! The execution length projection seeds kvcache_start_index from @p prefillStart. A non-zero
    //! start is a context-cache prefix hit: the
    //! coordinator leased the slot's pages (shared prefix included) and bound its row before this
    //! call, so the pass computes only the tail the cache did not cover.
    //!
    //! @throws std::runtime_error if the deployment carries a draft engine (draft-side state has
    //!         no batch-one projection yet) or if @p prefillStart is non-zero without a context cache to have
    //!         leased the pages behind it.
    bool prefillSlotInPlace(DecodingInferenceContext& context, int32_t slot, int32_t prefillStart = 0,
        SlotSeed const* mediaPayload = nullptr);

    //! @brief Run one request's decode loop to completion.
    bool runGenerationLoop(DecodingInferenceContext& context, DecodingStrategy& decodingStrategy,
        ManagedKVCacheRequest* managedRequest, LLMGenerationRequest const& request,
        DecodingKvHeadroom const& kvHeadroom, cudaStream_t stream, GenerationBoundaryHook const& boundaryHook);

    void initializeFromEngineDir(std::string const& engineDir, std::string const& multimodalEngineDir,
        std::unordered_map<std::string, std::string> const& loraWeightsMap,
        std::optional<SpecDecodeDraftingConfig> const& draftingConfig, cudaStream_t stream,
        ParallelMapping const& mapping, tokenizer::Tokenizer& tokenizer,
        chat_template::ChatTemplate const& chatTemplate, ContextCacheConfig const& contextCacheConfig,
        std::string const& checkpointDir, std::string const& draftCheckpointDir);

    void initializeCommon(ModelArtifacts&& artifacts, std::string const& engineDir,
        std::string const& multimodalEngineDir, std::unordered_map<std::string, std::string> const& loraWeightsMap,
        std::optional<SpecDecodeDraftingConfig> const& draftingConfig, cudaStream_t stream,
        ParallelMapping const& mapping, tokenizer::Tokenizer& tokenizer,
        chat_template::ChatTemplate const& chatTemplate, ContextCacheConfig const& contextCacheConfig);

    //! @brief Capture a CUDA graph on the base executor for the default (no-adapter)
    //! state, then one additional graph per registered LoRA adapter. Returns the
    //! logical AND of all captures — any single failure flips the aggregate to
    //! false but capture continues for remaining adapters (graceful degrade).
    bool captureBaseGraphWithLoraFanout(InferenceDims const& dims, cudaStream_t stream);

    //! @brief Build the strategy runtime reference bundle after common resources are allocated.
    void buildDecodingRuntimeContext();

    std::atomic<bool> mHandleRequestInProgress{false}; //!< Defensive overlap gate, not a thread-safety contract.

    //! Broadcast a sampled int32 token buffer when a parallel request supplies a callback.
    bool broadcastInt32(void* gpuBuffer, int32_t count, cudaStream_t stream);

    //! Broadcast root-rank cancellation decisions at a decode-iteration boundary so all ranks
    //! preserve identical collective and batch-eviction order.
    bool synchronizeCancellationStates(DecodingInferenceContext& context);

    rt::Tensor mSharedExecContextMemory{}; //!< Shared device memory for all execution contexts
    int32_t mMaxRuntimeBatchSize{1};       //!< Maximum runtime batch size

    DeploymentConfig mDeployment{};                    //!< Parsed base+draft configs + consolidated strategy settings
    std::unique_ptr<EngineExecutor> mBaseExecutor;     //!< Base model TRT wrapper
    std::unique_ptr<SharedResources> mSharedResources; //!< KV caches / RoPE / LoRA / context memory
    //! Declared after SharedResources so SWA page ownership is released before the physical buffers.
    std::unique_ptr<BoundedSwaKVPageManager> mBoundedSwaKVPageManager;

    //! Admission-scoped MRope staging ([1, maxKVCacheCapacity, rotaryDim]): the admission
    //! preprocess writes here until the batch-one execution copies it to active-row scratch.
    //! The ordinary ragged scatter then publishes it to the admitted ResidentRef. Allocated on the first multimodal
    //! admission of an MRope deployment; other deployments never pay for it.
    Tensor mAdmissionMropeStage;
    //! Declared after SharedResources so context-cache ownership is released before the physical buffers.
    std::unique_ptr<ContextCacheCoordinator> mContextCache;
    std::unique_ptr<PipelineIO> mPipelineIO; //!< Per-pipeline I/O tensors
    //! Scratch [maxSeq, baseOutputHiddenDim] used to shift baseHiddenStates down one row when folding a reused
    //! Hybrid+MTP checkpoint boundary into the draft prefill. Allocated only for Hybrid+MTP deployments.
    rt::Tensor mBoundaryFoldScratch;
    bool mHybridMtpContextReuseDeployment{};
    //! baseHiddenStates' max-sequence rows. The fold writes chunkLength + 1 rows, so it needs one spare row on top of
    //! the chunk it shifts; runHybridMtpPrefill checks the chunk against this bound before reshaping.
    int32_t mBoundaryFoldMaxRows{0};
    TensorMap mBaseTensorMap;                  //!< Base engine binding map
    std::filesystem::path mCheckpointDir;      //!< Provider checkpoint used during startup weight loading
    std::filesystem::path mDraftCheckpointDir; //!< Separate provider draft checkpoint, empty for integrated drafts
    LogitBias mLogitBias;                      //!< Runtime-owned resources that outlive decoding objects borrowing them
    std::unique_ptr<DecodingRuntimeContext> mDecodingRuntimeContext;
    std::unique_ptr<DecoderRegistry> mDecoderRegistry;
    std::unique_ptr<StepPreparer> mStepPreparer;             //!< Per-step sequence preprocessor
    std::unique_ptr<EmbeddingPreprocessor> mEmbeddingPre;    //!< Embedding-lookup preprocessor
    std::unique_ptr<VisualTokenPruner> mVisualPruner;        //!< Visual-token pruner (optional)
    std::unique_ptr<Gemma4EmbeddingPreprocessor> mGemma4Ple; //!< Gemma4 PLE token-identity preprocessor
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
    hash_utils::HashMap<std::tuple<std::string, std::string>, SystemPromptKVCache>
        mSystemPromptKVCacheBase;          //!< System prompt KVCache for base model
    std::string mEmptyLoraWeightsName{""}; //!< Empty LoRA weights name for default case

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
    rt::Tensor mSamplingUniforms;
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
    rt::Tensor mHostSamplingUniforms;    //!< Host pinned request-stable uniforms for prefill sampling
    rt::Tensor mHostReuseKVCacheLengths; //!< Host pinned memory for reuse KV cache lengths
    // [5] Multimodal support tensors for audio/image token indexing
    rt::Tensor mMultimodalIndices;     //!< GPU [batchSize, seqLen] multimodal embedding indices
    rt::Tensor mHostMultimodalIndices; //!< Host pinned [batchSize, seqLen] staging for CPU-computed indices

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
    int32_t mParallelRank{-1};          //!< Engine parallel rank for the active request (-1 = standalone)

    //! @brief Allocate or grow logprobs tensors/workspace to cover a request that enabled numLogprobs.
    void ensureLogprobsCapacity(int32_t logprobsRows, int32_t topK);

    // Key functions to drive the runtime, defined in a consumer-producer pattern.
    // Consume tokenized IDS as input and produce hidden states for the whole sequence and first generated token.
    //! @throws std::runtime_error if a CUDA error occurs
    bool runBaseModelPrefill(DecodingInferenceContext& context, ManagedKVCacheRequest* managedKVCacheRequest = nullptr,
        bool sampleOutput = true);

    //! Hybrid+MTP endpoint-reuse prefill. Mirrors the reference llmInferenceRuntime.cpp::runHybridMtpPrefill: a
    //! two-chunk base prefill publishing at the stable predecessor boundary, folding the reused checkpoint's boundary
    //! hidden into the draft prefill on a cache hit, and driving the coordinator's dedicated MTP publish entrypoint.
    //! Only reachable when shouldUseHybridMtpEndpointReuse() already established that the cache is live for this
    //! request, so lookup and publication are both enabled here by construction.
    bool runHybridMtpPrefill(
        DecodingInferenceContext& context, DecodingStrategy& strategy, ManagedKVCacheRequest& managedKVCacheRequest);

    //! Validate request shape/runtime compatibility.
    bool validateRequestConfig(LLMGenerationRequest const& request);

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

    // Consume system prompt, produce the hash table of system prompt KVCache if kv cache reuse is enabled.
    //! @throws std::runtime_error if a CUDA operation fails
    bool genAndSaveSystemPromptKVCache(DecodingInferenceContext& context, int32_t genAndSaveBatchIdx);

    // Consume batched input ids and the hash table of system prompt KVCache, produce the padded input ids and input
    // lengths. Instantiate the KVCache from the hash table if the system prompt has been cached.
    //! @throws std::runtime_error if system prompt is malformed
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
