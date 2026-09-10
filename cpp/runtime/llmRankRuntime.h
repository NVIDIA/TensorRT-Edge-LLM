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
#include "runtime/decoding/logitBias.h"
#include "runtime/exec/engineExecutor.h"
#include "runtime/exec/tensorMap.h"
#include "runtime/features/deepstackBinding.h"
#include "runtime/llmRuntimeUtils.h"
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
namespace rt
{

class ContextCacheCoordinator;
class ContextCacheRequest;

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
        ParallelMapping const& mapping, tokenizer::Tokenizer& tokenizer, ContextCacheConfig const& contextCacheConfig,
        std::string const& checkpointDir, std::string const& draftCheckpointDir);

    LLMRankRuntime(ModelArtifacts&& artifacts, std::string const& engineDir, std::string const& multimodalEngineDir,
        std::unordered_map<std::string, std::string> const& loraWeightsMap,
        std::optional<SpecDecodeDraftingConfig> const& draftingConfig, cudaStream_t stream,
        ParallelMapping const& mapping, tokenizer::Tokenizer& tokenizer, ContextCacheConfig const& contextCacheConfig);

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
    bool handleRequest(LLMGenerationRequest const& request, LLMGenerationResponse& response, cudaStream_t stream,
        bool outputThinkerEmbeddings = false, TokenBroadcastFn tokenBroadcast = nullptr, int32_t parallelRank = -1);

    /*! \brief Return the input size for an explicit text token-count request. */
    std::vector<int32_t> countPromptTokens(LLMGenerationRequest const& request) const;

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

private:
    void initializeFromEngineDir(std::string const& engineDir, std::string const& multimodalEngineDir,
        std::unordered_map<std::string, std::string> const& loraWeightsMap,
        std::optional<SpecDecodeDraftingConfig> const& draftingConfig, cudaStream_t stream,
        ParallelMapping const& mapping, tokenizer::Tokenizer& tokenizer, ContextCacheConfig const& contextCacheConfig,
        std::string const& checkpointDir, std::string const& draftCheckpointDir);

    void initializeCommon(ModelArtifacts&& artifacts, std::string const& engineDir,
        std::string const& multimodalEngineDir, std::unordered_map<std::string, std::string> const& loraWeightsMap,
        std::optional<SpecDecodeDraftingConfig> const& draftingConfig, cudaStream_t stream,
        ParallelMapping const& mapping, tokenizer::Tokenizer& tokenizer, ContextCacheConfig const& contextCacheConfig);

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
    //! Declared after SharedResources so shutdown and destruction release cache ownership before physical buffers.
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
    tokenizer::Tokenizer* mTokenizer{nullptr};                     //!< Shared tokenizer owned by RuntimeCoordinator
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
    rt::Tensor mBaseVocabMappingTable; // Vocab mapping table for base model reduced vocab (empty if not used)

    // [3] Batch eviction support tensors.
    rt::Tensor mDeviceBatchMapping;
    rt::Tensor mDeviceCancellationStates;
    rt::Tensor mHostCancellationStates; //!< Pinned host staging paired with mDeviceCancellationStates.

    // [4] Host pinned memory tensors for optimized CPU-GPU memory transfers
    rt::Tensor mHostPackedTokenIds;      //!< Host pinned memory for packed token IDs
    rt::Tensor mHostSelectedTokenIds;    //!< Host pinned memory for selected token IDs from sampling
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

    //! @brief Restore recurrent/conv states from a cached system prompt.
    void restoreRecurrentStates(int32_t batchIdx, SystemPromptKVCache const& cachedStates, cudaStream_t stream);

    //! @brief Zero all recurrent/conv states for a given batch index.
    void zeroRecurrentStates(int32_t batchIdx, cudaStream_t stream);

    // Key functions to drive the runtime, defined in a consumer-producer pattern.
    // Consume tokenized IDS as input and produce hidden states for the whole sequence and first generated token.
    //! @throws std::runtime_error if a CUDA error occurs
    bool runBaseModelPrefill(DecodingInferenceContext& context, ContextCacheRequest* contextCacheRequest = nullptr,
        bool sampleOutput = true);

    //! Hybrid+MTP endpoint-reuse prefill. Mirrors the reference llmInferenceRuntime.cpp::runHybridMtpPrefill: a
    //! two-chunk base prefill publishing at the stable predecessor boundary, folding the reused checkpoint's boundary
    //! hidden into the draft prefill on a cache hit, and driving the coordinator's dedicated MTP publish entrypoint.
    //! Only reachable when shouldUseHybridMtpEndpointReuse() already established that the cache is live for this
    //! request, so lookup and publication are both enabled here by construction.
    bool runHybridMtpPrefill(
        DecodingInferenceContext& context, DecodingStrategy& strategy, ContextCacheRequest& contextCacheRequest);

    //! Validate request shape/runtime compatibility.
    bool validateRequestConfig(LLMGenerationRequest const& request);

    //! Prepare per-request runtime state for models built with multimodal support.
    //! Runs multimodal preprocessing when audio or vision inputs are present.
    //! For text-only requests on MRope-based multimodal models, restores text-only RoPE state
    //! and clears stale multimodal request state.
    bool multiModalRuntimePreprocess(
        LLMGenerationRequest const& request, DecodingInferenceContext& context, cudaStream_t stream);

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
    bool performBatchEvict(DecodingInferenceContext& context, DecodingStrategy& strategy,
        std::vector<int8_t>& thinkingDone, ContextCacheRequest* contextCacheRequest);

    // Stage-specific metrics
    metrics::LLMPrefillMetrics mPrefillMetrics;
    metrics::SpecDecodeGenerationMetrics mSpecDecodeGenerationMetrics;
    metrics::LLMGenerationMetrics mGenerationMetrics; //!< Vanilla generation metrics (used when no spec-decode)
};

} // namespace rt
} // namespace trt_edgellm
