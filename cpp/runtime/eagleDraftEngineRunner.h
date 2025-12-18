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

#include "common/tensor.h"
#include "runtime/linearKVCache.h"
#include "runtime/llmRuntimeUtils.h"
#include <cstdint>
#include <cuda_runtime.h>
#include <filesystem>
#include <memory>
#include <nlohmann/json.hpp>
#include <optional>
#include <unordered_map>

namespace trt_edgellm
{
namespace rt
{
using Json = nlohmann::json;

/*! \brief Configuration structure for the Eagle Draft Engine Runner
 */
struct EagleDraftEngineRunnerConfig
{
    RopeConfig ropeConfig{};           //!< RoPE configuration
    int32_t numDecoderLayers{};        //!< Number of decoder layers in the draft model
    int32_t numKVHeads{};              //!< Number of key-value heads
    int32_t headDim{};                 //!< Dimension of each attention head
    int32_t rotaryDim{};               //!< Dimension of rotary positional encoding
    int32_t maxSupportedBatchSize{};   //!< Maximum supported batch size
    int32_t maxSupportedInputLength{}; //!< Maximum supported input length
    int32_t maxKVCacheCapacity{};      //!< Maximum KV cache capacity
    int32_t draftModelVocabSize{};     //!< Vocabulary size of the draft model
    int32_t maxDraftTreeSize{};        //!< Maximum size of the draft tree
    int32_t baseModelHiddenDim{};      //!< Hidden dimension of the base model
    int32_t draftModelHiddenDim{};     //!< Hidden dimension of the draft model
    bool isVlm{false};                 //!< Flag indicating if this is a vision-language model
};

// Disable clang-format to explicitly format the class interface documentation.
// clang-format off
/*! \brief Eagle Draft Engine Runner class for speculative decoding
 */
class EagleDraftEngineRunner
{
public:
    /*! \brief Construct an Eagle Draft Engine Runner
     *  \param enginePath Path to the TensorRT engine file
     *  \param configPath Path to the configuration JSON file
     *  \param stream CUDA stream for initialization
     */
    EagleDraftEngineRunner(
        std::filesystem::path const& enginePath, std::filesystem::path const& configPath, cudaStream_t stream);

    /*! \brief Destructor
     */
    ~EagleDraftEngineRunner();

    /*! \brief Get internal RoPE cosine/sine cache tensor for the eagle draft engine
     *  \return Reference to the RoPE cosine/sine cache tensor
     */
    rt::Tensor& getRopeCosSinCacheTensor();
    
    /*! \brief Get internal linear KV cache for the eagle draft engine
     *  \return Reference to the linear KV cache
     */
    rt::LinearKVCache& getLinearKVCache();

    /*! \brief Get the draft engine configuration
     *  \return The draft engine configuration structure
     */
    EagleDraftEngineRunnerConfig getDraftEngineConfig() const;

    /*! \brief API entry to execute prefill step for the eagle draft engine
     * 
     *  By definition, eagle operates on feature level with formulation of f_n = F_proj(f_{n}, token_{n+1}). 
     *  The API will takes hidden states input from base model and token_ids of [1 ~ N] as input, output logits 
     *  and (draft) hidden states for the "last entry" to be used in following draft proposal step. 
     *  Multi-batch is supported - each batch can have different actual sequence length (with padding).
     * 
     *  \param inputIds [GPU, Int32] Input token_ids for the draft model with shape [batch_size, N_padded] denoting the token_ids of [1 ~N]
     *  \param baseModelHiddenStates [GPU, Float16] Hidden states input from base model with shape [batch_size, N_padded, base-Hidden-dim],
     *                               denote hidden states corresponding to token_ids of [1 ~ N-1]
     *  \param draftModelHiddenStates [GPU, Float16] The input [batch_size, N_padded, draft-Hidden-input-dim] is unused in the prefill step,
     *                                but it is required by the engine execution. The input shall be set to all zeros to ensure correctness
     *  \param contextLengths [CPU, Int32] The actual sequence length for each batch with shape [batch_size] (including the +1 token from base prefill)
     *  \param multimodalEmbeddings [GPU] Optional. The multimodal embeddings
     *  \param outputLogits [GPU, Float16] The output logits with shape [batch_size, draft-Vocab-Size]
     *  \param outputHiddenStates [GPU] The output hidden states with shape [batch_size, draft-hidden-dim]
     *  \param stream The CUDA stream to execute the prefill step
     *  \return True if execution was successful, false otherwise
     */
    bool executeEaglePrefillStep(rt::Tensor const& inputIds, rt::Tensor const& baseModelHiddenStates,
        rt::Tensor const& draftModelHiddenStates, rt::Tensor const& contextLengths, rt::OptionalInputTensor multimodalEmbeddings, rt::Tensor& outputLogits,
        rt::Tensor& outputHiddenStates, rt::Tensor const& baseRopeCosSinCache, cudaStream_t stream);

    /*! \brief API entry to execute the draft proposal step for the eagle draft engine
     * 
     *  The API will takes a draft tree of input_token_ids and hidden-states from the draft model. 
     *  DraftTreeMask denote the relationship between the draft tree nodes, draft tree length denote the 
     *  "real" length of the draft tree. To efficiently use cuda graph and reduce implementation complexity, 
     *  the input length will be padded to accommodate the maximum draft tree size.
     * 
     *  \param draftTreeInputIds [GPU, Int32] Input token_ids for the draft model with shape [1, padded-draft-Tree-Size]
     *  \param baseModelHiddenStates [GPU, Float16] The input [1, padded-draft-Tree-Size, base-Hidden-Dim] is unused in the
     *                               draft proposal step, but it is required by the engine execution. The input shall be set to all zeros to ensure correctness
     *  \param draftModelHiddenStates [GPU, Float16] Hidden states input from draft model with shape [1, padded-draft-Tree-Size, draft-Hidden-Dim],
     *                                denote hidden states corresponding to token_ids of [1 ~ draft-Tree-Size]
     *  \param draftTreeLength [GPU, Int32] Denote the "real" length of the draft tree with shape [1]
     *  \param draftTreeMask [GPU, Int32] Denote the relationship between the draft tree nodes with shape [1, padded-draft-Tree-Size, padded-draft-Tree-Size]
     *  \param outputLogits [GPU, Float16] The output logits with shape [topK, draft-Vocab-Size]
     *  \param outputHiddenStates [GPU] The output hidden states with shape [topK, draft-hidden-dim]
     *  \param stream The CUDA stream to execute the draft proposal step
     *  \return True if execution was successful, false otherwise
     * 
     *  \note The API will automatically collect the "last" topK logits and hidden-states counting from the tail of
     *        "real" draft tree size. Caller shall specify the topK parameter through tensor dimension. 
     *        Also this API will NOT "commit" the KVCache during execution.
     */
    bool executeEagleDraftProposalStep(rt::Tensor const& draftTreeInputIds, rt::Tensor const& baseModelHiddenStates,
        rt::Tensor const& draftModelHiddenStates, rt::Tensor const& draftTreeLength, rt::Tensor const& draftTreeMask,
        rt::Tensor& outputLogits, rt::Tensor& outputHiddenStates, cudaStream_t stream);

    /*! \brief API entry for the eagle draft model to accept the "committed" token from the base model
     * 
     *  The functionality is similar to the prefill step where this API will operates based on the previous 
     *  committed KVCache. Output logits and hidden-states will be collected from the last accepted token.
     * 
     *  \param acceptedTokens [GPU, Int32] The accepted tokens with shape [batch_size, N_accepted_padded] where N_accepted_padded is the maximum accepted length across all batches
     *  \param baseModelHiddenStates [GPU, Float16] Hidden states input from base model with shape [batch_size, N_accepted_padded, base-Hidden-Dim]
     *  \param draftModelHiddenStates [GPU, Float16] The input [batch_size, N_accepted_padded, draft-Hidden-Dim] is unused in the accept decode token step,
     *                                but it is required by the engine execution. The input shall be set to all zeros to ensure correctness
     *  \param acceptedTokenNums [GPU, Int32] The actual number of accepted tokens for each batch with shape [batch_size], used to handle variable-length acceptance per sequence
     *  \param outputLogits [GPU, Float16] The output logits with shape [batch_size, draft-Vocab-Size]
     *  \param outputHiddenStates [GPU] The output hidden states with shape [batch_size, draft-hidden-dim]
     *  \param stream The CUDA stream to execute the accept decode token step
     *  \return True if execution was successful, false otherwise
     * 
     *  \note This API will "commit" the KVCache for the accepted tokens.
     */
    bool executeEagleAcceptDecodeTokenStep(rt::Tensor const& acceptedTokens, rt::Tensor const& baseModelHiddenStates,
        rt::Tensor const& draftModelHiddenStates, rt::Tensor const& acceptedTokenNums, rt::Tensor& outputLogits,
        rt::Tensor& outputHiddenStates, cudaStream_t stream);

    /*! \brief API entry to capture the CUDA graph for the draft proposal step
     * 
     *  The API will capture the CUDA graph for the draft proposal step.
     * 
     *  \param draftTreeInputIds [GPU, Int32] Input token_ids for the draft model with shape [1, padded-draft-Tree-Size]
     *  \param baseModelHiddenStates [GPU, Float16] The input [1, padded-draft-Tree-Size, base-Hidden-Dim] is unused in the
     *                               draft proposal step, but it is required by the engine execution. The input shall be set to all zeros to ensure correctness
     *  \param draftModelHiddenStates [GPU, Float16] Hidden states input from draft model with shape [1, padded-draft-Tree-Size, draft-Hidden-Dim],
     *                                denote hidden states corresponding to token_ids of [1 ~ draft-Tree-Size]
     *  \param draftTreeLength [GPU, Int32] Denote the "real" length of the draft tree with shape [1]
     *  \param draftTreeMask [GPU, Int32] Denote the relationship between the draft tree nodes with shape [1, padded-draft-Tree-Size, padded-draft-Tree-Size]
     *  \param outputLogits [GPU, Float16] The output logits with shape [topK, draft-Vocab-Size]
     *  \param outputHiddenStates [GPU] The output hidden states with shape [topK, draft-hidden-dim]
     *  \param stream The CUDA stream to capture the CUDA graph. The API will capture the CUDA graph for the draft proposal step
     *  \return True if the CUDA graph is captured successfully, false otherwise
     */
    bool captureEagleDraftProposalCudaGraph(rt::Tensor const& draftTreeInputIds, rt::Tensor const& baseModelHiddenStates,
        rt::Tensor const& draftModelHiddenStates, rt::Tensor const& draftTreeLength, rt::Tensor const& draftTreeMask,
        rt::Tensor& outputLogits, rt::Tensor& outputHiddenStates, cudaStream_t stream);

    /*! \brief API entry for capturing the CUDA graph for the accept decode token step
     * 
     *  The functionality is similar to the draft proposal step where this API will operates based on the 
     *  previous committed KVCache. Output logits and hidden-states will be collected from the last accepted token.
     * 
     *  \param acceptedTokens [GPU, Int32] The accepted tokens with shape [batch_size, N_accepted_padded] where N_accepted_padded is the maximum accepted length across all batches
     *  \param baseModelHiddenStates [GPU, Float16] Hidden states input from base model with shape [batch_size, N_accepted_padded, base-Hidden-Dim]
     *  \param draftModelHiddenStates [GPU, Float16] The input [batch_size, N_accepted_padded, draft-Hidden-Dim] is unused in the accept decode token step,
     *                                but it is required by the engine execution. The input shall be set to all zeros to ensure correctness
     *  \param acceptedTokenNums [GPU, Int32] The actual number of accepted tokens for each batch with shape [batch_size], used to handle variable-length acceptance per sequence
     *  \param outputLogits [GPU, Float16] The output logits with shape [batch_size, draft-Vocab-Size]
     *  \param outputHiddenStates [GPU] The output hidden states with shape [batch_size, draft-hidden-dim]
     *  \param stream The CUDA stream to capture the CUDA graph. The API will capture the CUDA graph for the accept decode token step
     *  \return True if the CUDA graph is captured successfully, false otherwise
     */
    bool captureEagleAcceptDecodeTokenCudaGraph(rt::Tensor const& acceptedTokens, rt::Tensor const& baseModelHiddenStates,
        rt::Tensor const& draftModelHiddenStates, rt::Tensor const& acceptedTokenNums, rt::Tensor& outputLogits, 
        rt::Tensor& outputHiddenStates, cudaStream_t stream);

private:
    EagleDraftEngineRunnerConfig mConfig{};  //!< Configuration for the Eagle Draft Engine Runner

    std::unique_ptr<nvinfer1::IRuntime> mRuntime;  //!< TensorRT runtime instance
    std::unique_ptr<nvinfer1::ICudaEngine> mEngine;  //!< TensorRT engine instance
    rt::Tensor mExecContextMemory{};                                          //!< Device memory for the execution contexts
    std::unique_ptr<nvinfer1::IExecutionContext> mPrefillExecutionContext;    //!< TensorRT execution context for context phase
    std::unique_ptr<nvinfer1::IExecutionContext> mGenerationExecutionContext; //!< TensorRT execution context for generation phase

    std::unordered_map<size_t, std::pair<cudaGraph_t, cudaGraphExec_t>> mDraftProposalCudaGraphs{};  //!< Map of CUDA graphs for draft proposal step indexed by configuration hash
    std::unordered_map<size_t, std::pair<cudaGraph_t, cudaGraphExec_t>> mAcceptDecodeTokenCudaGraphs{};  //!< Map of CUDA graphs for accept decode token step indexed by configuration hash

    rt::LinearKVCache mLinearKVCache{};  //!< Linear KV cache for storing key-value pairs

    rt::Tensor mPosEncCosSinCache{};  //!< (GPU, Float32) to store the CosSinCache for rotary positional encoding
    rt::Tensor mSelectTokenIndices{};  //!< (GPU, Int64) to store the select token indices that will be outputted from the model
    rt::Tensor mSequenceContextLengths{};  //!< (GPU, Int32) to store the sequence context lengths input that will be used by the TensorRT Engine
    rt::Tensor mDraftTreePositionIds{};  //!< (GPU, Int32) to store the draft tree position ids within the sequence that used by positional encoding
    rt::Tensor mPackedTreeMask{};  //!< (GPU, Int32) to store the packed tree mask to indicate the attention relationship between the draft tree nodes
    rt::Tensor mAcceptedTokenNums{};  //!< (GPU, Int32) to store accepted token numbers for batch sequences
    //! (GPU, Half) to store a GPU buffer as dummy tensor for unused input tensors. TensorRT doesn't
    //! allow binding address to be nullptr.
    rt::Tensor mDummyTensor{};

    //! Initialize the configuration from the JSON file.
    //! \param configJson The JSON configuration object
    //! \return True if initialization was successful, false otherwise
    bool initializeConfigFromJson(Json const& configJson);

    //! Validate the configuration from the engine.
    //! \return True if validation was successful, false otherwise
    bool validateConfigFromEngine();

    //! Bind KV cache to the engine.
    //! \param activeBatchSize The active batch size
    //! \return True if binding was successful, false otherwise
    bool bindKVCacheToEngine(int32_t activeBatchSize);

    //! Validate input parameters for the prefill step.
    //! \param inputIds Input token IDs tensor
    //! \param baseModelHiddenStates Base model hidden states tensor
    //! \param draftModelHiddenStates Draft model hidden states tensor
    //! \param contextLengths Context lengths for each batch (actual lengths)
    //! \param multimodalEmbeddings Optional multimodal embeddings
    //! \param outputLogits Output logits tensor
    //! \param outputHiddenStates Output hidden states tensor
    //! \return True if validation passed, false otherwise
    bool prefillStepInputValidation(rt::Tensor const& inputIds, rt::Tensor const& baseModelHiddenStates,
        rt::Tensor const& draftModelHiddenStates, rt::Tensor const& contextLengths, rt::OptionalInputTensor multimodalEmbeddings, rt::Tensor const& outputLogits, rt::Tensor const& outputHiddenStates);

    //! Validate input parameters for the draft proposal step.
    //! \param draftTreeInputIds Draft tree input IDs tensor
    //! \param baseModelHiddenStates Base model hidden states tensor
    //! \param draftModelHiddenStates Draft model hidden states tensor
    //! \param draftTreeLength Draft tree length tensor
    //! \param draftTreeMask Draft tree mask tensor
    //! \param outputLogits Output logits tensor
    //! \param outputHiddenStates Output hidden states tensor
    //! \return True if validation passed, false otherwise
    bool draftProposalStepInputValidation(rt::Tensor const& draftTreeInputIds, rt::Tensor const& baseModelHiddenStates,
        rt::Tensor const& draftModelHiddenStates, rt::Tensor const& draftTreeLength, rt::Tensor const& draftTreeMask,
        rt::Tensor const& outputLogits, rt::Tensor const& outputHiddenStates);

    //! Validate input parameters for the accept decode token step.
    //! \param acceptedTokens Accepted tokens tensor [batch_size, N_accepted_padded]
    //! \param baseModelHiddenStates Base model hidden states tensor [batch_size, N_accepted_padded, base-Hidden-Dim]
    //! \param draftModelHiddenStates Draft model hidden states tensor [batch_size, N_accepted_padded, draft-Hidden-Dim]
    //! \param acceptedTokenNums Actual number of accepted tokens per batch [batch_size]
    //! \param outputLogits Output logits tensor [batch_size, draft-Vocab-Size]
    //! \param outputHiddenStates Output hidden states tensor [batch_size, draft-hidden-dim]
    //! \return True if validation passed, false otherwise
    bool acceptDecodeTokenStepInputValidation(rt::Tensor const& acceptedTokens, rt::Tensor const& baseModelHiddenStates,
        rt::Tensor const& draftModelHiddenStates, rt::Tensor const& acceptedTokenNums, rt::Tensor const& outputLogits, 
        rt::Tensor const& outputHiddenStates);
};

// clang-format on

} // namespace rt
} // namespace trt_edgellm