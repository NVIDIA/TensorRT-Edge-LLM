/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "common/tensor.h"
#include "multimodal/multimodalRunner.h"
#include "profiling/metrics.h"
#include "profiling/timer.h"
#include "runtime/eagleDraftEngineRunner.h"
#include "runtime/llmEngineRunner.h"
#include "runtime/llmRuntimeUtils.h"
#include "tokenizer/tokenizer.h"
#include <unordered_map>
#include <vector>

namespace trt_edgellm
{
namespace
{
/*! \brief Structure to hold cached system prompt and its KV cache
 */
struct SystemPromptKVCache
{
    std::string systemPrompt;                     //!< The system prompt text
    std::vector<tokenizer::Rank> tokenizedPrompt; //!< Tokenized version of the system prompt
    rt::Tensor kvCacheContent;                    //!< Cached KV cache content for the system prompt
};
} // namespace

namespace rt
{
/*!
 * @brief Execution context for speculative decode runtime
 *
 * Holds execution information and intermediate metadata during inference.
 * Supports multi-batch inference with independent sequence tracking.
 */
struct SpecDecodeInferenceContext
{
    std::vector<std::string> systemPrompts;               //!< System prompts for each sequence in batch
    std::vector<std::vector<int32_t>> rawBatchedInputIds; //!< Original token IDs before preprocessing (includes padding
                                                          //!< and removal of reused system IDs)
    std::vector<std::vector<int32_t>> tokenIds;           //!< Token IDs for each sequence: [batch_size][seq_length]
    std::vector<int32_t> currentGenerateLengths;          //!< Current generation length for each sequence: [batch_size]
    std::vector<int32_t> promptLengths; //!< Prompt length (after reuse) for each sequence: [batch_size]
    std::vector<int8_t> finishedStates; //!< Finished state for each sequence: [batch_size] (0=not finished, 1=finished)
    std::vector<int32_t> actualIterations; //!< Actual iterations run for each sequence: [batch_size]
    int32_t packedInputLength; //!< Packed input length for batch processing (max of all sequences, considering engine
                               //!< constraints)

    // Evicted batch results (saved before eviction for final output)
    // Key: original batch index, Value: batch data
    std::unordered_map<int32_t, std::vector<int32_t>> evictedTokenIds; //!< Token IDs of evicted batches
    std::unordered_map<int32_t, int32_t> evictedGenerateLengths;       //!< Generation lengths of evicted batches
    std::unordered_map<int32_t, int32_t> evictedActualIterations;      //!< Iterations of evicted batches
    std::unordered_map<int32_t, std::string> evictedSystemPrompts;     //!< System prompts of evicted batches
    std::unordered_map<int32_t, std::vector<int32_t>> evictedRawBatchedInputIds; //!< Raw input IDs of evicted batches
    std::unordered_map<int32_t, int32_t> evictedPromptLengths;                   //!< Prompt lengths of evicted batches
    std::vector<int32_t> batchIndexMapping;       //!< Maps current batch index to original index
    rt::OptionalInputTensor multimodalEmbeddings; //!< Optional multimodal embeddings
    rt::OptionalInputTensors extraInputTensors;   //!< Extra input tensors (e.g., deepstack features)
    int32_t generationRound;                      //!< Current generation round (shared across all batches)
    int32_t maxGenerateLength;                    //!< Maximum generation length
    int32_t activeBatchSize;                      //!< Current active batch size
    int32_t originalBatchSize;                    //!< Original batch size (before any eviction)
    int32_t genAndSaveSystemCacheIndex; //!< Batch index being processed for generating and saving system prompt KVCache
    cudaStream_t stream;                //!< CUDA stream

    /*!
     * @brief Initialize the context with given parameters
     * @param batchSize Active batch size
     * @param maxGenLength Maximum generation length
     * @param multimodal Optional multimodal embeddings
     * @param extraInputTensors Extra input tensors (e.g., deepstack features)
     * @param cudaStream CUDA stream for operations
     */
    void initialize(int32_t batchSize, int32_t maxGenLength, rt::OptionalInputTensor const& multimodal,
        rt::OptionalInputTensors const& extraInputTensors, cudaStream_t cudaStream);
};

/*!
 * @brief Drafting configuration for Eagle speculative decoding
 *
 * Configuration parameters to drive Eagle spec-decoding.
 */
struct EagleDraftingConfig
{
    int32_t draftingTopK;   //!< Tokens to select from one predecessor for next draft tree level
    int32_t draftingStep;   //!< Number of drafting steps with draft model
    int32_t verifyTreeSize; //!< Number of tokens for base model verification
};

/*!
 * @brief LLM inference runtime with Eagle speculative decoding
 *
 * Manages inference pipeline using Eagle speculative decoding for improved throughput.
 * Coordinates base model, draft model, and multimodal processing.
 */
class LLMInferenceSpecDecodeRuntime
{
public:
    /*!
     * @brief Construct speculative decode runtime
     * @param engineDir Directory containing engine files
     * @param multimodalEngineDir Directory containing multimodal engine files
     * @param draftingConfig Eagle drafting configuration
     * @param stream CUDA stream for operations
     */
    LLMInferenceSpecDecodeRuntime(std::string const& engineDir, std::string const& multimodalEngineDir,
        EagleDraftingConfig const& draftingConfig, cudaStream_t stream);

    //! @brief Destructor
    ~LLMInferenceSpecDecodeRuntime() = default;
    //! @brief Capture CUDA graph for draft proposal
    //! @param stream CUDA stream
    //! @return True on success, false on failure
    bool captureDraftProposalCudaGraph(cudaStream_t stream);

    //! @brief Capture CUDA graph for draft accept decode token
    //! @param stream CUDA stream
    //! @return True on success, false on failure
    bool captureDraftAcceptDecodeTokenCudaGraph(cudaStream_t stream);

    //! @brief Capture CUDA graph for base verification
    //! @param stream CUDA stream
    //! @return True on success, false on failure
    bool captureBaseVerificationCudaGraph(cudaStream_t stream);

    /*!
     * @brief Handle generation request
     * @param request Generation request with prompts and parameters
     * @param response Output response with generated tokens and text
     * @param stream CUDA stream
     * @return True on success, false on failure
     */
    bool handleRequest(LLMGenerationRequest const& request, LLMGenerationResponse& response, cudaStream_t stream);

    //! Get LLM prefill stage metrics
    metrics::LLMPrefillMetrics const& getPrefillMetrics() const
    {
        return mPrefillMetrics;
    }

    //! Get Eagle generation stage metrics
    metrics::EagleGenerationMetrics const& getEagleGenerationMetrics() const
    {
        return mEagleGenerationMetrics;
    }

    //! Get multimodal metrics (returns empty metrics if no multimodal runner)
    metrics::MultimodalMetrics getMultimodalMetrics() const
    {
        return mMultimodalRunner ? mMultimodalRunner->getMultimodalMetrics() : metrics::MultimodalMetrics{};
    }

private:
    int32_t mMaxRuntimeBatchSize{1};                 //!< Maximum runtime batch size
    EagleDraftingConfig mDraftingConfig;             //!< Eagle drafting configuration
    LLMEngineRunnerConfig mBaseEngineConfig;         //!< Base engine configuration
    EagleDraftEngineRunnerConfig mDraftEngineConfig; //!< Draft engine configuration

    std::unique_ptr<LLMEngineRunner> mBaseEngineRunner;                       //!< Base model engine runner
    std::unique_ptr<EagleDraftEngineRunner> mDraftEngineRunner;               //!< Draft model engine runner
    std::unique_ptr<MultimodalRunner> mMultimodalRunner{nullptr};             //!< Multimodal runner (optional)
    std::unique_ptr<tokenizer::Tokenizer> mTokenizer;                         //!< Tokenizer
    std::unordered_map<size_t, SystemPromptKVCache> mSystemPromptKVCacheBase; //!< System prompt KVCache for base model
    std::unordered_map<size_t, SystemPromptKVCache>
        mSystemPromptKVCacheDraft; //!< System prompt KVCache for draft model

    // Pre-define key runtime GPU tensors and initialize them during construction.
    // [1] I/O Tensors to work with base and eagle draft engine.
    rt::Tensor mIdsInput;
    rt::Tensor mContextLengthsInput;
    rt::Tensor mLogitsOutput;
    rt::Tensor mDraftTreeSize;
    rt::Tensor mDraftTreeMask;
    rt::Tensor mBaseHiddenStatesOutput;
    // Distinguish draft hidden states input and output since we cannot easily
    // Perform inplace update for hidden states between drafting steps.
    rt::Tensor mDraftHiddenStatesInput;
    rt::Tensor mDraftHiddenStatesOutput;

    // [2] Sampling workspace and output tensors that used across all the sampling operations.
    rt::Tensor mSamplingWorkspace;
    rt::Tensor mSamplingIndices;
    rt::Tensor mSamplingScores;
    rt::Tensor mBaseVocabMappingTable; // Vocab mapping table for base model reduced vocab (empty if not used)

    // [3] Data structures used during Draft tree constructions.
    // Data tables that store the data structure that can completely describe a multi-layer draft tree.
    rt::Tensor mDraftTokenIdsFullTable;
    rt::Tensor mDraftTokenScoreFullTable;
    rt::Tensor mDraftTokenPredecessorFullTable;
    // Store conversion table (offset) to map from draft-model vocab token id to the original token id.
    // base_id = draft_id + mapping_table[draft_id]
    rt::Tensor mDraftVocabMappingTable;

    rt::Tensor mDraftTreeRootTokenId;
    rt::Tensor mDraftTokenIdsTable;
    rt::Tensor mDraftTokenScoresTable;
    rt::Tensor mDraftTokenIntermediateScores;
    rt::Tensor mDraftTokenIntermediateParents;

    // [4] Data structures that used during base model verification.
    rt::Tensor mAcceptedTokenIds;
    rt::Tensor mAcceptedTokenIndices;
    rt::Tensor mAcceptLength;

    // [5] Batch eviction support tensors.
    rt::Tensor mDeviceBatchMapping;

    // [6] Host pinned memory tensors for optimized CPU-GPU memory transfers
    rt::Tensor mHostPackedTokenIds;      //!< Host pinned memory for packed token IDs
    rt::Tensor mHostSelectedTokenIds;    //!< Host pinned memory for selected token IDs from sampling
    rt::Tensor mHostAcceptLengths;       //!< Host pinned memory for accept lengths from verification
    rt::Tensor mHostAcceptedTokenIds;    //!< Host pinned memory for accepted token IDs
    rt::Tensor mHostReuseKVCacheLengths; //!< Host pinned memory for reuse KV cache lengths

    // Key functions to drive the spec-decode runtime, defined in a consumer-producer pattern.
    // Consume tokenized IDS as input and produce hidden states for the whole sequence and first generated token.
    bool runBaseModelPrefill(SpecDecodeInferenceContext& context);

    // Consume the base model hidden states and input token of the sequence. Produce the draft hidden states and logits
    // for the last token of the sequence.
    bool runDraftModelPrefill(SpecDecodeInferenceContext& context);

    // Consume the draft hidden states and logits for the last token of the sequence. Produce a speculative draft tree
    // that described by a sequence of draft tokens and tree mask that describe the tree structure.
    bool constructDraftTree(SpecDecodeInferenceContext& context);

    // Consume the speulative draft tree, produce selected tokens and corresponding hidden states.
    bool runBaseModelVerification(SpecDecodeInferenceContext& context);

    // Consume the selected tokens and base model hidden state, produce the draft hidden states and logits for the last
    // token of the accepted sequence.
    bool runDraftModelAcceptToken(SpecDecodeInferenceContext& context);

    // Consume system prompt, produce the hash table of system prompt KVCache if kv cache reuse is enabled.
    bool genAndSaveSystemPromptKVCache(SpecDecodeInferenceContext& context);

    // Consume batched input ids and the hash table of system prompt KVCache, produce the padded input ids and input
    // lengths. Instantiate the KVCache from the hash table if the system prompt has been cached.
    bool setUpForPrefillExecution(SpecDecodeInferenceContext& context);

    // Batch eviction support
    //! @brief Perform batch eviction
    //! @param context Inference context
    //! @return True on success, false on failure
    bool performBatchEvict(SpecDecodeInferenceContext& context);

    // Stage-specific metrics
    metrics::LLMPrefillMetrics mPrefillMetrics;
    metrics::EagleGenerationMetrics mEagleGenerationMetrics;
};

} // namespace rt
} // namespace trt_edgellm
