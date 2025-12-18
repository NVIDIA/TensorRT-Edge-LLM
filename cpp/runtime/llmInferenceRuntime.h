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

#pragma once

#include "multimodal/multimodalRunner.h"
#include "profiling/metrics.h"
#include "profiling/timer.h"
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
     */
    LLMInferenceRuntime(std::string const& engineDir, std::string const& multimodalEngineDir,
        std::unordered_map<std::string, std::string> const& loraWeightsMap, cudaStream_t stream);

    /*! \brief Destructor
     */
    ~LLMInferenceRuntime() = default;

    /*! \brief Handle an LLM generation request
     *  \param request The generation request containing prompt and generation parameters
     *  \param response The generation response to be filled with output
     *  \param stream CUDA stream for execution
     *  \return True if request was handled successfully, false otherwise
     */
    bool handleRequest(LLMGenerationRequest const& request, LLMGenerationResponse& response, cudaStream_t stream);

    /*! \brief Capture CUDA graph for the decoding step to optimize performance
     *  \param stream CUDA stream for graph capture
     *  \return True if graph was captured successfully, false otherwise
     */
    bool captureDecodingCUDAGraph(cudaStream_t stream);

    /*! \brief Execute the prefill step generation of the KVCache for the prompt and save for later usage
     *
     *  \param prompt The system prompt to generate the KVCache
     *  \param loraWeightsName The name of the LoRA weights
     *  \param stream The CUDA stream used for the generation
     *  \return True if the KVCache is generated and saved successfully, false otherwise
     */
    bool genAndSaveSystemPromptKVCache(
        std::string const& prompt, std::string const& loraWeightsName, cudaStream_t stream);

    /*! \brief Get LLM prefill stage metrics
     *  \return Reference to prefill metrics
     */
    metrics::LLMPrefillMetrics const& getPrefillMetrics() const
    {
        return mPrefillMetrics;
    }

    /*! \brief Get LLM generation stage metrics
     *  \return Reference to generation metrics
     */
    metrics::LLMGenerationMetrics const& getGenerationMetrics() const
    {
        return mGenerationMetrics;
    }

    /*! \brief Get multimodal metrics (returns empty metrics if no multimodal runner)
     *  \return Multimodal metrics, or empty metrics if no multimodal runner is available
     */
    metrics::MultimodalMetrics getMultimodalMetrics() const
    {
        return mMultimodalRunner ? mMultimodalRunner->getMultimodalMetrics() : metrics::MultimodalMetrics{};
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
        std::vector<std::string> const& systemPrompts, std::string const& loraWeightsName) const;

    std::unique_ptr<LLMEngineRunner> mLLMEngineRunner{nullptr};   //!< LLM engine runner instance
    std::unique_ptr<MultimodalRunner> mMultimodalRunner{nullptr}; //!< Multimodal runner instance (optional)
    std::unique_ptr<tokenizer::Tokenizer> mTokenizer{nullptr};    //!< Tokenizer instance
    std::unordered_map<size_t, SystemPromptKVCache>
        mSystemPromptKVCache{}; //!< Cache of system prompts and their KV caches

    rt::Tensor mSamplingWorkspace{};       //!< Workspace tensor for sampling operations
    rt::Tensor mInputIds{};                //!< Input token IDs tensor
    rt::Tensor mHostPackedInputIds{};      //!< Host tensor for packed input IDs
    rt::Tensor mHostContextLengths{};      //!< Host tensor for context lengths
    rt::Tensor mOutputLogits{};            //!< Output logits tensor
    rt::Tensor mSelectedIndices{};         //!< Selected token indices tensor
    rt::Tensor mHostSelectedTokenIds{};    //!< Host tensor for selected token IDs
    rt::Tensor mHostReuseKVCacheLengths{}; //!< Reuse KV cache lengths for prefill
    rt::Tensor mVocabMappingTable{};       //!< Vocab mapping table for reduced vocab (empty if not used)
    std::string mEmptyLoraWeightsName{""}; //!< Empty LoRA weights name for default case

    LLMEngineRunnerConfig mEngineConfig{}; //!< Engine configuration

    metrics::LLMPrefillMetrics mPrefillMetrics; //!< Stage-specific metrics to store number of tokens in prefill
    metrics::LLMGenerationMetrics
        mGenerationMetrics; //!< Stage-specific metrics to store number of tokens in generation

    //! Examine and validate the generation request.
    //! \param request The generation request to examine
    //! \return True if request is valid, false otherwise
    bool examineRequest(LLMGenerationRequest const& request);

    //! Set up tensors and state for prefill execution.
    //! \param batchedInputIds Batched input token IDs
    //! \param systemPrompts System prompts for each batch element
    //! \param loraWeightsName Name of the LoRA weights being used
    //! \param stream CUDA stream for execution
    //! \return True if setup was successful, false otherwise
    bool setUpForPrefillExecution(std::vector<std::vector<int32_t>> const& batchedInputIds,
        std::vector<std::string> const& systemPrompts, std::string const& loraWeightsName, cudaStream_t stream);
};
} // namespace rt
} // namespace trt_edgellm
