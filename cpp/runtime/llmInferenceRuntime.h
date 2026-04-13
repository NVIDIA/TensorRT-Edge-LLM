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

#include "common/hashUtils.h"
#include "common/safetensorsUtils.h"
#include "multimodal/multimodalRunner.h"
#include "profiling/metrics.h"
#include "profiling/timer.h"
#include "runtime/alpamayoExpertRunner.h"
#include "runtime/llmEngineRunner.h"
#include "runtime/llmRuntimeUtils.h"
#include "tokenizer/tokenizer.h"
#include <memory>
#include <string>
#include <unordered_map>

namespace trt_edgellm
{
namespace rt
{

/*! \brief Structure to hold cached system prompt and its KV cache
 */
struct SystemPromptKVCache
{
    std::string systemPrompt;                     //!< The system prompt text
    std::vector<tokenizer::Rank> tokenizedPrompt; //!< Tokenized version of the system prompt
    rt::Tensor kvCacheContent;                    //!< Cached KV cache content for the system prompt
    std::vector<rt::Tensor> ssmStateContents;     //!< Cached SSM states for Mamba layers
    std::vector<rt::Tensor> convStateContents;    //!< Cached conv states for Mamba layers
};

/*! \brief LLM Inference Runtime for handling generation requests
 */
class LLMInferenceRuntime
{
public:
    /*! \brief Construct an LLM Inference Runtime
     *  \param engineDir Directory containing the LLM engine
     *  \param multimodalEngineDir Directory containing the multimodal engine
     *  \param loraWeightsMap Map of LoRA weights names to their paths
     *  \param stream CUDA stream for initialization
     *  \throws std::runtime_error if loading data from directories fails, or tensor shapes are invalid, or there is an
     * initialization error
     */
    LLMInferenceRuntime(std::string const& engineDir, std::string const& multimodalEngineDir,
        std::unordered_map<std::string, std::string> const& loraWeightsMap, cudaStream_t stream);

    /*! \brief Destructor
     */
    ~LLMInferenceRuntime() noexcept = default;

    /*! \brief Handle an LLM generation request
     *  \param request The generation request containing prompt and generation parameters
     *  \param response The generation response to be filled with output
     *  \param stream CUDA stream for execution
     *  \return True if request was handled successfully, false otherwise
     *  \throws std::runtime_error if a LLM operation or CUDA operation fails
     */
    bool handleRequest(LLMGenerationRequest const& request, LLMGenerationResponse& response, cudaStream_t stream);

    /*! \brief Capture CUDA graph for the decoding step to optimize performance
     *  \param stream CUDA stream for graph capture
     *  \return True if graph was captured successfully, false otherwise
     *  \throws std::runtime_error if a CUDA operation fails
     */
    bool captureDecodingCUDAGraph(cudaStream_t stream);

    /*! \brief Execute the prefill step generation of the KVCache for the prompt and save for later usage
     *
     *  \param prompt The system prompt to generate the KVCache
     *  \param loraWeightsName The name of the LoRA weights
     *  \param stream The CUDA stream used for the generation
     *  \return True if the KVCache is generated and saved successfully, false otherwise
     *  \throws std::runtime_error if a CUDA operation fails
     */
    bool genAndSaveSystemPromptKVCache(
        std::string const& prompt, std::string const& loraWeightsName, cudaStream_t stream);

    /*! \brief Get LLM prefill stage metrics
     *  \return Reference to prefill metrics
     */
    metrics::LLMPrefillMetrics const& getPrefillMetrics() const noexcept
    {
        return mPrefillMetrics;
    }

    /*! \brief Get LLM generation stage metrics
     *  \return Reference to generation metrics
     */
    metrics::LLMGenerationMetrics const& getGenerationMetrics() const noexcept
    {
        return mGenerationMetrics;
    }

    /*! \brief Get multimodal metrics (returns empty metrics if no multimodal runner)
     *  \return Multimodal metrics, or empty metrics if no multimodal runner is available
     */
    metrics::MultimodalMetrics getMultimodalMetrics() const noexcept
    {
        return mVisionRunner ? mVisionRunner->getMultimodalMetrics()
            : mAudioRunner   ? mAudioRunner->getMultimodalMetrics()
                             : metrics::MultimodalMetrics{};
    }

    /*! \brief Enable dumping the KV cache to disk after each prefill+decode.
     *  When non-empty, every successful request in handleRequest() writes the
     *  KV cache (and shape metadata) to <path>.req<idx>.bin / <path>.req<idx>.shape.json.
     */
    void setKVCacheDumpPath(std::string const& path) noexcept { mKVCacheDumpPath = path; }

    /*! \brief Initialize the Expert runner for inline diffusion after VLM decode.
     *  When set, handleRequest() runs Expert diffusion instead of (or in addition to) dumping KV cache.
     */
    void initExpertRunner(std::string const& expertEnginePath, AlpamayoExpertConfig const& config, cudaStream_t stream)
    {
        mExpertRunner = std::make_unique<AlpamayoExpertRunner>(expertEnginePath, config, stream);
    }

private:
    /*! \brief Helper structure to hold token counting results
     */
    struct TokenCountInfo
    {
        int32_t totalReusedTokens{0};   //!< Number of tokens reused from KV cache
        int32_t totalComputedTokens{0}; //!< Number of tokens that need computation
    };

    //! Calculate token counts (reused vs computed) for performance tracking.
    //! \param batchedInputIds Batched input token IDs
    //! \param systemPrompts System prompts for each batch element
    //! \param loraWeightsName Name of the LoRA weights being used
    //! \return TokenCountInfo structure containing reused and computed token counts
    TokenCountInfo calculateTokenCounts(std::vector<std::vector<int32_t>> const& batchedInputIds,
        std::vector<std::string> const& systemPrompts, std::string const& loraWeightsName) const noexcept;

    std::unique_ptr<LLMEngineRunner> mLLMEngineRunner{nullptr}; //!< LLM engine runner instance
    std::unique_ptr<MultimodalRunner> mAudioRunner{nullptr};    //!< Audio runner instance (optional)
    std::unique_ptr<MultimodalRunner> mVisionRunner{nullptr};   //!< Vision runner instance (optional)
    std::unique_ptr<tokenizer::Tokenizer> mTokenizer{nullptr};  //!< Tokenizer instance
    hash_utils::HashMap<std::tuple<std::string, std::string>, SystemPromptKVCache>
        mSystemPromptKVCache{}; //!< Cache of system prompts / LORA weights and their KV caches

    rt::Tensor mEmbeddingTable{};    //!< Shared embedding table [vocabSize, hiddenSize]
    rt::Tensor mSamplingWorkspace{}; //!< Workspace tensor for sampling operations
    rt::Tensor mInputIds{};          //!< Input token IDs tensor
    rt::Tensor mInputsEmbeds{};      //!< Input embeddings tensor [batchSize, seqLen, hiddenSize]
    rt::Tensor mMultimodalIndices{}; //!< Multimodal indices tensor [batchSize, seqLen] for audio/image embeddings
    std::vector<rt::Tensor> mDeepstackEmbeds; //!< Deepstack embeddings tensors for Qwen3-VL (one per feature)
    rt::Tensor mHostPackedInputIds{};         //!< Host tensor for packed input IDs
    rt::Tensor mHostContextLengths{};         //!< Host tensor for context lengths
    rt::Tensor mOutputLogits{};               //!< Output logits tensor
    rt::Tensor mSelectedIndices{};            //!< Selected token indices tensor
    rt::Tensor mHostSelectedTokenIds{};       //!< Host tensor for selected token IDs
    rt::Tensor mHostReuseKVCacheLengths{};    //!< Reuse KV cache lengths for prefill
    rt::Tensor mVocabMappingTable{};          //!< Vocab mapping table for reduced vocab (empty if not used)
    std::string mEmptyLoraWeightsName{""};    //!< Empty LoRA weights name for default case

    LLMEngineRunnerConfig mEngineConfig{}; //!< Engine configuration

    metrics::LLMPrefillMetrics mPrefillMetrics; //!< Stage-specific metrics to store number of tokens in prefill
    metrics::LLMGenerationMetrics
        mGenerationMetrics; //!< Stage-specific metrics to store number of tokens in generation

    //! Examine and validate the generation request.
    //! \param request The generation request to examine
    //! \return True if request is valid, false otherwise
    bool examineRequest(LLMGenerationRequest const& request) noexcept;

    //! Set up tensors and state for prefill execution.
    //! \param batchedInputIds Batched input token IDs
    //! \param systemPrompts System prompts for each batch element
    //! \param loraWeightsName Name of the LoRA weights being used
    //! \param stream CUDA stream for execution
    //! \return True if setup was successful, false otherwise
    //! \throws std::runtime_error if system prompt is malformed, or a CUDA operation fails
    bool setUpForPrefillExecution(std::vector<std::vector<int32_t>> const& batchedInputIds,
        std::vector<std::string> const& systemPrompts, std::string const& loraWeightsName, cudaStream_t stream);

    //! Dump the current prefill KV cache for a single-batch request.
    //! Saves a safetensors file containing:
    //! - kv_cache: [num_layers, 2, num_kv_heads, seq_len, head_dim]
    //! - sequence_length: [1] int32
    //! - input_ids: [seq_len] int32
    bool dumpCurrentPrefillKVCache(std::string const& outputPath, int32_t sequenceLength, cudaStream_t stream);

    std::unique_ptr<AlpamayoExpertRunner> mExpertRunner{nullptr};
    std::string mKVCacheDumpPath{};             //!< Optional path prefix to dump KV cache after decode (Path A)
    int64_t mKVCacheDumpReqIdx{0};              //!< Counter for dump file naming
};
} // namespace rt
} // namespace trt_edgellm
