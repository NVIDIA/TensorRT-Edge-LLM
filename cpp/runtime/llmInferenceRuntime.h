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

#include "action/alpamayo1ActionRunner.h"
#include "common/hashUtils.h"
#include "common/tensor.h"
#include "multimodal/multimodalRunner.h"
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
#include "runtime/preprocess/embeddingPreprocessor.h"
#include "runtime/preprocess/gemma4EmbeddingPreprocessor.h"
#include "runtime/preprocess/mediaArtifactCache.h"
#include "runtime/preprocess/stepPreparer.h"
#include "runtime/state/contextCache/contextCacheRuntimeAdapter.h"
#include "runtime/state/contextCache/hybridSnapshotStorage.h"
#include "runtime/state/decodingInferenceContext.h"
#include "runtime/state/pipelineIO.h"
#include "runtime/state/sharedResources.h"
#include "runtime/state/systemPromptKVCache.h"
#include "runtime/streaming.h"
#include "tokenizer/tokenizer.h"
#include <memory>
#include <optional>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace trt_edgellm
{
namespace rt
{
/*!
 * @brief Unified LLM inference runtime with optional speculative decoding
 *
 * Manages inference pipeline for vanilla and speculative decoding modes (EAGLE, MTP, etc.).
 * When constructed without a drafting config, operates as a pure vanilla decoding runtime
 * with zero draft-model memory overhead.
 * Coordinates base model, optional draft model, and multimodal processing (vision + audio).
 */
class LLMInferenceRuntime
{
public:
    /*!
     * @brief Construct runtime with speculative decoding
     * @param engineDir Directory containing engine files
     * @param multimodalEngineDir Directory containing multimodal engine files
     * @param loraWeightsMap Map of LoRA weight names to file paths
     * @param draftingConfig Speculative decoding drafting configuration
     * @param stream CUDA stream for operations
     * @param contextCacheConfig Production context-cache policy and snapshot/artifact budgets
     * @throws std::runtime_error if directories do not contain expected data, or runner initialization fails
     */
    LLMInferenceRuntime(std::string const& engineDir, std::string const& multimodalEngineDir,
        std::unordered_map<std::string, std::string> const& loraWeightsMap,
        SpecDecodeDraftingConfig const& draftingConfig, cudaStream_t stream,
        ContextCacheConfig const& contextCacheConfig = {});

    /*!
     * @brief Construct runtime for vanilla-only decoding (no draft model)
     * @param engineDir Directory containing engine files
     * @param multimodalEngineDir Directory containing multimodal engine files
     * @param loraWeightsMap Map of LoRA weight names to file paths
     * @param stream CUDA stream for operations
     * @param contextCacheConfig Production context-cache policy and snapshot/artifact budgets
     * @throws std::runtime_error if directories do not contain expected data, or runner initialization fails
     */
    LLMInferenceRuntime(std::string const& engineDir, std::string const& multimodalEngineDir,
        std::unordered_map<std::string, std::string> const& loraWeightsMap, cudaStream_t stream,
        ContextCacheConfig const& contextCacheConfig = {});

    //! @brief Destructor
    ~LLMInferenceRuntime() noexcept = default;

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
     */
    bool handleRequest(LLMGenerationRequest const& request, LLMGenerationResponse& response, cudaStream_t stream,
        bool outputThinkerEmbeddings = false);

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
    //! @brief Common initialization logic shared between both constructors
    void initializeCommon(std::string const& engineDir, std::string const& multimodalEngineDir,
        std::unordered_map<std::string, std::string> const& loraWeightsMap,
        std::optional<SpecDecodeDraftingConfig> const& draftingConfig, cudaStream_t stream,
        ContextCacheConfig const& contextCacheConfig);

    //! @brief Capture a CUDA graph on the base executor for the default (no-adapter)
    //! state, then one additional graph per registered LoRA adapter. Returns the
    //! logical AND of all captures — any single failure flips the aggregate to
    //! false but capture continues for remaining adapters (graceful degrade).
    bool captureBaseGraphWithLoraFanout(InferenceDims const& dims, cudaStream_t stream);

    //! @brief Build the strategy runtime reference bundle after common resources are allocated.
    void buildDecodingRuntimeContext();

    //! Publish a HALF base-engine logits output through PipelineIO's FP32
    //! sampler contract. No-op for the normal FLOAT-output engine ABI.
    void publishBaseLogits(cudaStream_t stream);

    rt::Tensor mSharedExecContextMemory{}; //!< Shared device memory for all execution contexts
    int32_t mMaxRuntimeBatchSize{1};       //!< Maximum runtime batch size

    DeploymentConfig mDeployment{};                    //!< Parsed base+draft configs + consolidated strategy settings
    std::unique_ptr<EngineExecutor> mBaseExecutor;     //!< Base model TRT wrapper
    std::unique_ptr<SharedResources> mSharedResources; //!< KV caches / RoPE / LoRA / context memory
    std::unique_ptr<PipelineIO> mPipelineIO;           //!< Per-pipeline I/O tensors
    TensorMap mBaseTensorMap;                          //!< Base engine binding map
    bool mConvertBaseLogits{false};                    //!< Base engine emits HALF instead of the common FLOAT ABI
    rt::Tensor mBaseEngineOutputLogits;                //!< Dedicated HALF binding for supported MTP base engines
    LogitBias mLogitBias; //!< Runtime-owned resources that outlive decoding objects borrowing them
    std::unique_ptr<DecodingRuntimeContext> mDecodingRuntimeContext;
    std::unique_ptr<DecoderRegistry> mDecoderRegistry;
    ContextCacheConfig mContextCacheConfig{};
    std::unique_ptr<ContextCacheRuntimeAdapter> mContextCache;
    std::unique_ptr<HybridSnapshotStorage> mHybridSnapshotStorage;
    std::unique_ptr<MediaArtifactCache> mMediaArtifactCache;
    CacheDomainId mContextCacheDomain{};
    std::optional<DraftEngineSignature> mDraftEngineSignature;
    std::optional<RecurrentStateSchemaId> mRecurrentStateSchema;
    std::unique_ptr<StepPreparer> mStepPreparer;             //!< Per-step sequence preprocessor
    std::unique_ptr<EmbeddingPreprocessor> mEmbeddingPre;    //!< Embedding-lookup preprocessor
    std::unique_ptr<Gemma4EmbeddingPreprocessor> mGemma4Ple; //!< Gemma4 PLE token-identity preprocessor
    //! Base-engine deepstack binding (nullptr when the base engine was built
    //! without deepstack features). Swaps between `io.deepstackEmbeds[i]`
    //! (prefill) and the shared `zeroDeepstackBroadcast` (all other phases).
    std::unique_ptr<DeepstackBinding> mDeepstack;

    std::unique_ptr<MultimodalRunner> mVisionRunner{nullptr};      //!< Vision multimodal runner (optional)
    std::unique_ptr<MultimodalRunner> mAudioRunner{nullptr};       //!< Audio multimodal runner (optional)
    std::unique_ptr<Alpamayo1ActionRunner> mActionRunner{nullptr}; //!< Action/diffusion head runner (optional)
    std::unique_ptr<tokenizer::Tokenizer> mTokenizer;              //!< Tokenizer
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
    rt::Tensor mBaseVocabMappingTable; //!< Vocab mapping table for base model reduced vocab (empty if unused)

    // [3] Batch eviction support tensors.
    rt::Tensor mDeviceBatchMapping;

    // [4] Host pinned memory tensors for optimized CPU-GPU memory transfers
    rt::Tensor mHostPackedTokenIds;      //!< Host pinned memory for packed token IDs
    rt::Tensor mHostSelectedTokenIds;    //!< Host pinned memory for selected token IDs from sampling
    rt::Tensor mHostReuseKVCacheLengths; //!< Host pinned memory for reuse KV cache lengths
    //! Host reuse KV cache lengths for the draft (speculative) engine; equals the base lengths except at a Hybrid+MTP
    //! checkpoint boundary, where the draft reuses one fewer token so the boundary slot is recomputed.
    rt::Tensor mHostDraftReuseKVCacheLengths;
    //! Scratch [maxSeq, baseOutputHiddenDim] used to shift base hidden states by one row when folding the Hybrid+MTP
    //! checkpoint boundary into the suffix draft prefill.
    rt::Tensor mBoundaryFoldScratch;

    // [5] Multimodal support tensors for audio/image token indexing
    rt::Tensor mMultimodalIndices;      //!< Multimodal indices tensor [batchSize, seqLen] for audio/image embeddings
    rt::Tensor mVisionArtifactAssembly; //!< Request-local full-row assembly when ViT is skipped
    std::vector<rt::Tensor> mVisionDeepstackArtifactAssembly;
    rt::Tensor mAudioArtifactAssembly; //!< Request-local full-row assembly when audio encoder is skipped

    // [6] Logprobs support tensors (allocated once in the constructor)
    // logprobsRows = B (vanilla) or B * maxAcceptDepth (EAGLE, accepted rows only, not the full verify tree).
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

    //! @brief Allocate logprobs tensors. Called once from the constructor.
    void allocateLogprobsTensors();

    //! @brief Restore recurrent/conv states from a cached system prompt.
    void restoreRecurrentStates(int32_t batchIdx, SystemPromptKVCache const& cachedStates, cudaStream_t stream);

    //! @brief Zero all recurrent/conv states for a given batch index.
    void zeroRecurrentStates(int32_t batchIdx, cudaStream_t stream);

    // Key functions to drive the runtime, defined in a consumer-producer pattern.
    // Consume tokenized IDS as input and produce hidden states for the whole sequence and first generated token.
    //! @throws std::runtime_error if a CUDA error occurs
    bool runBaseModelPrefill(DecodingInferenceContext& context, bool sampleOutput = true);
    //! Split a single-sequence hybrid suffix at configured exact-capture boundaries, then sample only on the final
    //! chunk.
    bool runBaseModelPrefillWithPeriodicHybridCaptures(DecodingInferenceContext& context);
    //! Publish a stable Hybrid-MTP predecessor endpoint, then replay the prompt tail before round zero.
    bool runHybridMtpPrefill(DecodingInferenceContext& context, DecodingStrategy& strategy);

    //! Validate request shape/runtime compatibility against the decoder selected for this request.
    bool validateRequestConfig(LLMGenerationRequest const& request, DecodingStrategyKind selectedStrategy);

    //! Prepare per-request runtime state for models built with multimodal support.
    //! Runs multimodal preprocessing when audio or vision inputs are present.
    //! For text-only requests on MRope-based multimodal models, restores text-only RoPE state
    //! and clears stale multimodal request state.
    bool multiModalRuntimePreprocess(
        LLMGenerationRequest const& request, DecodingInferenceContext& context, cudaStream_t stream);
    //! Resolve cached media artifacts after sequence acquisition, running only encoders still required by the suffix.
    bool resolveMultimodalArtifacts(
        LLMGenerationRequest const& request, DecodingInferenceContext& context, cudaStream_t stream);

    // Consume system prompt, produce the hash table of system prompt KVCache if kv cache reuse is enabled.
    //! @throws std::runtime_error if a CUDA operation fails
    bool genAndSaveSystemPromptKVCache(DecodingInferenceContext& context, int32_t genAndSaveBatchIdx);

    // Consume batched input ids and the hash table of system prompt KVCache, produce the padded input ids and input
    // lengths. Instantiate the KVCache from the hash table if the system prompt has been cached.
    //! @throws std::runtime_error if system prompt is malformed
    bool setUpForPrefillExecution(DecodingInferenceContext& context, DecodingStrategy& strategy);

    //! Grow active vanilla page paths before a decode step crosses a page boundary.
    bool prepareContextCacheForDecode(DecodingInferenceContext& context);

    //! Publish a ready boundary and rebind a final duplicate page when no descendants were computed from it.
    void publishContextCacheBoundary(
        DecodingInferenceContext& context, int32_t slot, PublicationPoint point, int32_t residentStateLength);
    void publishHybridContextCacheBoundary(
        DecodingInferenceContext& context, int32_t slot, PublicationPoint point, int32_t residentStateLength);
    void publishHybridMtpContextCacheBoundary(
        DecodingInferenceContext& context, int32_t slot, PublicationPoint point, int32_t residentStateLength);

    // Batch eviction support
    //! @brief Perform batch eviction
    //! @param context Inference context
    //! @return True on success, false on failure
    //! @throws std::runtime_error if a CUDA error occurs
    bool performBatchEvict(DecodingInferenceContext& context, DecodingStrategy& strategy);

    // Stage-specific metrics
    metrics::LLMPrefillMetrics mPrefillMetrics;
    metrics::SpecDecodeGenerationMetrics mSpecDecodeGenerationMetrics;
    metrics::LLMGenerationMetrics mGenerationMetrics; //!< Vanilla generation metrics (used when no spec-decode)
};

} // namespace rt
} // namespace trt_edgellm
