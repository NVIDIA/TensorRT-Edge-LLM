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
#include "common/tensor.h"
#include "runtime/linearKVCache.h"
#include "runtime/llmRuntimeUtils.h"

#include <NvInferRuntime.h>
#include <cuda_runtime.h>
#include <filesystem>
#include <nlohmann/json.hpp>
#include <optional>
#include <unordered_map>
#include <vector>

namespace trt_edgellm
{
namespace rt
{
using Json = nlohmann::json;

/*!
 * @brief Configuration structure for LLM engine runner
 *
 * Contains all runtime configuration parameters for the LLM engine.
 */
struct LLMEngineRunnerConfig
{
    RopeConfig ropeConfig{};             //!< Type of rotary positional encoding
    bool useContextDependentRope{false}; //!< Use context-dependent RoPE
    bool enableEagleSpecDecode{false};   //!< Enable Eagle speculative decoding
    bool enableContextEmb{false};        //!< GR00T VLA: engine also outputs context_embs
    bool enableLmHiddenStates{false};    //!< GR00T VLA: engine outputs lm_hidden_states for action_context engine
    bool enablePrefixKVOutputs{false};   //!< VLA: engine outputs prefix_k / prefix_v for action head
    bool hasLogitsOutput{true};          //!< False for prefill-only VLA language engines without lm_head
    bool hasLastTokenIdsInput{true};     //!< False when the exported language engine omits last_token_ids
    bool useTrtNativeOps{false};         //!< Use TensorRT native operations instead of custom plugin
    int32_t numDecoderLayers{};          //!< Number of decoder layers
    int32_t numKVHeads{};                //!< Number of key-value heads
    int32_t headDim{};                   //!< Dimension of each attention head
    int32_t rotaryDim{};                 //!< Rotary embedding dimension
    int32_t hiddenSize{};                //!< Model's hidden dimension
    int32_t maxSupportedBatchSize{};     //!< Maximum supported batch size
    int32_t maxSupportedInputLength{};   //!< Maximum supported input length
    int32_t maxKVCacheCapacity{};        //!< Maximum KV cache capacity
    int32_t vocabSize{};                 //!< Vocabulary size (full vocabulary)
    int32_t reducedVocabSize{0};         //!< Reduced vocabulary size (0 if not using reduced vocab)
    int32_t outputVocabSize{};      //!< Actual output vocabulary size (reducedVocabSize if enabled, else vocabSize)
    int32_t maxSupportedLoraRank{}; //!< Maximum supported LoRA rank
    int32_t outputHiddenDim{};      //!< Output hidden dimension for Eagle speculative decoding (hidden_size * 3)
    int32_t maxVerifyTreeSize{};    //!< Maximum verification tree size for Eagle speculative decoding
    int32_t contextEmbDim{};        //!< context_embs hidden dim for GR00T VLA action head
    std::vector<int64_t> prefixKVOutputShape{}; //!< PI0.5 prefix_k / prefix_v max output shape from engine
    int32_t numDeepstackFeatures{0};            //!< Number of deepstack features for Qwen3-VL and Qwen3-Omni
    int32_t audioTokenId{0};                    //!< Special token ID for audio in Qwen3-Omni
    int32_t imageTokenId{0};                    //!< Special token ID for image in Qwen3-Omni

    // Hybrid model configuration
    int32_t numLinearAttnLayers{0};    //!< Number of recurrent layers (0 for pure attention models)
    int32_t numAttentionLayers{0};     //!< Number of attention layers (equals numDecoderLayers for pure attention)
    int32_t recurrentStateNumHeads{0}; //!< Number of recurrent heads (hv for GDN, mamba_num_heads for Mamba)
    int32_t recurrentStateHeadDim{0};  //!< Dimension of each recurrent head (k for GDN, mamba_head_dim for Mamba)
    int32_t recurrentStateSize{0};     //!< Recurrent state dimension (v for GDN, dstate for Mamba)
    int32_t convDim{0};                //!< Conv1d channel dimension
    int32_t convKernel{0};             //!< Conv1d kernel width
};

//! The class wraps the TensorRT engine built for auto-regressive style decoder model.
//! The LLMEngineRunner define the interface for upper level runtime to execute engine actions to drive
//!     autoregressive decoding with/without speculative decoding for edge inference scenarios. Current design
//!     assume prefill and decoding operations are synchronous so a batched requests need to perform prefill and
//!     decoding at the same time (no continuous batching).
//! The LLMEngineRunner will:
//!     1. Hold TensorRT resources of the LLM engine (TRT IRuntime, CUDA Engine, Execution Contexts).
//!     2. Hold the LinearKVCache resources that support till maxSupportedBatchSize and maxSequenceLength.
//!     3. Hold the Rope CosSinCache tensor required for positional encoding.
class LLMEngineRunner
{
public:
    /*!
     * @brief Construct LLM engine runner
     * @param enginePath Path to TensorRT engine file
     * @param configPath Path to model configuration file
     * @param loraWeightsMap Map of LoRA weight names to file paths
     * @param stream CUDA stream for operations
     * @throws std::runtime_error If engine loading, configuration parsing, or initialization fails, or a CUDA operation
     * fails
     */
    LLMEngineRunner(std::filesystem::path const& enginePath, std::filesystem::path const& configPath,
        std::unordered_map<std::string, std::string> const& loraWeightsMap, cudaStream_t stream);

    //! @brief Destructor
    ~LLMEngineRunner() noexcept;

    /*!
     * @brief Get the required context memory size for this engine
     * @return Required context memory size in bytes
     */
    int64_t getRequiredContextMemorySize() const;

    /*!
     * @brief Set shared context memory for the execution context
     * @param sharedContextMemory Tensor containing the shared device memory (must be on GPU)
     * @return True on success, false if the tensor is too small
     * @note The tensor size must be >= getRequiredContextMemorySize(). Must be called before execution.
     */
    bool setContextMemory(rt::Tensor& sharedContextMemory);

    //! API entry to get the Rope CosSinCache tensor.
    //! The API is useful when the rope cos/sin cache depends on the context which cannot be initialized
    //! in advance when creating the LLMEngineRunner instance.
    rt::Tensor& getRopeCosSinCacheTensor() noexcept;

    //! @brief Get reference to the linear KV cache (also owns recurrent/conv state buffers for hybrid models)
    //! @return Reference to LinearKVCache
    rt::LinearKVCache& getLinearKVCache() noexcept;

    //! @brief Get engine configuration
    //! @return Engine configuration structure
    LLMEngineRunnerConfig getEngineConfig() const noexcept;

    //! @brief Set an extra input tensor for the engine
    //!
    //! This is a temporary API for binding additional input tensors that are not part of
    //! the standard LLM input set.
    //! @note This is not a good design but we put it here temporarily to support TTS inference.
    //! @note The API will be replaced soon with a better design. Please don't follow this schema.
    //!
    //! Example use case: CodePredictor's lm_head_weight input for dynamic lm_head selection.
    //!
    //! @param name The name of the LMHead input weights in the ONNX/TRT model
    //! @param tensor The tensor to bind (must be on GPU, shape must match engine expectation)
    //! @return True if the binding was successful
    //! @note Must be called before executePrefillStep/executeVanillaDecodingStep
    bool setLMHeadWeights(std::string const& name, rt::Tensor const& tensor);

    //! API entry to execute one prefill engine action for a batched request. The API will clear existing KVCache for
    //! last
    //!     batch of requests and perform prefill operations to fill the KVCache and produce the output logits.
    //! Inputs:
    //!     inputsEmbeds [GPU]: The input embeddings for the batch of new requests [batchSize, seqLen, hiddenSize].
    //!     contextLengths [CPU]: The context lengths for each sequence in the batch.
    //!     deepstackEmbeds [GPU]: Optional. Deepstack embeddings for Qwen3-VL (already embedded).
    //!     outputLogits [GPU]: The output logits for the batch of requests.
    //!     outputHiddenStates [GPU]: Optional. The output hidden states for Eagle speculative decoding.
    //!     outputContextEmbeds [GPU]: Optional. The output context_embs for GR00T VLA action head.
    //!     stream: The CUDA stream to execute the prefill step.
    //! Returns:
    //!     True if the prefill step is successful, false otherwise.
    //! @throws std::runtime_error if setting optimization profile fails, or a CUDA operation fails
    bool executePrefillStep(rt::Tensor const& inputsEmbeds, rt::Tensor const& contextLengths,
        rt::OptionalInputTensors deepstackEmbeds, rt::Tensor& outputLogits, rt::OptionalOutputTensor outputHiddenStates,
        cudaStream_t stream, rt::OptionalOutputTensor outputContextEmbeds = std::nullopt,
        rt::OptionalOutputTensor outputPrefixK = std::nullopt, rt::OptionalOutputTensor outputPrefixV = std::nullopt);

    //! API entry to execute one vanilla decoding engine action for a batched request. The API will perform decoding
    //!     operations fill the KVCache of the new generated tokens and produce the output logits. The decoding
    //!     operation shall be performed after the prefill step is completed.
    //! Inputs:
    //!     inputsEmbeds [GPU]: The input embeddings for the batch of new requests [batchSize, 1, hiddenSize].
    //!     stream: The CUDA stream to execute the decoding step.
    //! Outputs:
    //!     outputLogits [GPU]: The output logits for the batch of requests.
    //! Returns:
    //!     True if the decoding step is successful, false otherwise.
    //! @throws std::runtime_error if setting optimization profile fails, or a CUDA operation fails
    bool executeVanillaDecodingStep(rt::Tensor const& inputsEmbeds, rt::Tensor& outputLogits,
        rt::OptionalOutputTensor outputHiddenStates, cudaStream_t stream,
        rt::OptionalOutputTensor outputContextEmbeds = std::nullopt);

    //! API entry to execute eagle base tree decoding step. The API will takes a draft tree of input embeddings.
    //!     baseTreeDecodingMask denote the relationship between the draft tree nodes.
    //! Inputs:
    //!     baseTreeDecodingInputsEmbeds [GPU, Float16]: Input embeddings for the base model with shape [batchSize,
    //!     Tree-Size, hiddenSize]. baseTreeDecodingMask [GPU, Int32]: Denote the relationship between the base tree
    //!     nodes with shape
    //!         [batchSize, Tree-Size, Tree-Size].
    //!     stream: The CUDA stream to execute the base tree decoding step.
    //! Outputs:
    //!     outputLogits [GPU, Float16]: The output logits with shape [batchSize*Tree-Size, base-Vocab-Size].
    //!     outputHiddenStates [GPU]: The output hidden states with shape [batchSize*Tree-Size, base-hidden-dim].
    //! @throws std::runtime_error if setting optimization profile fails, or a CUDA operation fails
    bool executeEagleBaseTreeDecodingStep(rt::Tensor const& baseTreeDecodingInputsEmbeds,
        rt::Tensor const& baseTreeDecodingMask, rt::Tensor& outputLogits, rt::Tensor& outputHiddenStates,
        cudaStream_t stream);

    //! API entry to capture the CUDA graph for the decoding step. If CUDA graph capture is successful, later
    //!     call to executeVanillaDecodingStep() will always launch the captured CUDA graph.
    //! Inputs:
    //!     inputsEmbeds [GPU]: The input embeddings for the batch of new requests [batchSize, 1, hiddenSize].
    //!     outputLogits [GPU]: The output logits for the batch of requests.
    //!     loraWeightsName: The name to the LoRA weights. Empty string if no LoRA weights.
    //!     stream: The CUDA stream to execute the decoding step.
    //! Returns:
    //!     True if the CUDA graph capture is successful, false otherwise.
    //! @throws std::runtime_error if setting optimization profile fails, or a CUDA operation fails
    bool captureVanillaDecodingCudaGraph(rt::Tensor const& inputsEmbeds, rt::Tensor& outputLogits,
        std::string const& loraWeightsName, cudaStream_t stream,
        rt::OptionalOutputTensor outputHiddenStates = std::nullopt,
        rt::OptionalOutputTensor outputContextEmbeds = std::nullopt);

    //! API entry to switch the LoRA weights of the LLM engine.
    //! Inputs:
    //!     loraWeightsName: The name of the LoRA weights.
    //! Returns:
    //!     True if the LoRA weights switch is successful, false otherwise.
    bool switchLoraWeights(std::string const& loraWeightsName);

    //! API entry to get the active LoRA weights name.
    //! Returns:
    //!     The active LoRA weights name.
    std::string getActiveLoraWeightsName() const;

    //! API entry to get the LoRA weights.
    //! Returns:
    //!     The LoRA weights names.
    std::vector<std::string> getAvailableLoraWeights() const;

    //! API entry to capture the CUDA graph for the base model tree decoding step. If CUDA graph capture is successful,
    //! later
    //!     call to executeEagleBaseTreeDecodingStep() will always launch the captured CUDA graph.
    //! Inputs:
    //!     baseTreeDecodingInputsEmbeds [GPU, Float16]: Input embeddings for the base model with shape [batchSize,
    //!     Tree-Size, hiddenSize]. baseTreeDecodingMask [GPU, Int32]: Denote the relationship between the base tree
    //!     nodes with shape
    //!         [batchSize, Tree-Size, Tree-Size].
    //!     outputLogits [GPU, Float16]: The output logits with shape [batchSize*Tree-Size, base-Vocab-Size].
    //!     outputHiddenStates [GPU]: The output hidden states with shape [batchSize*Tree-Size, base-hidden-dim].
    //!     stream: The CUDA stream to capture the CUDA graph. The API will capture the CUDA graph for the base tree
    //!     decoding step.
    //! Returns:
    //!     True if the CUDA graph capture is successful, false otherwise.
    //! @throws std::runtime_error if setting optimization profile fails, or a CUDA operation fails
    bool captureEagleBaseTreeDecodingCudaGraph(rt::Tensor const& baseTreeDecodingInputsEmbeds,
        rt::Tensor const& baseTreeDecodingMask, rt::Tensor& outputLogits, rt::Tensor& outputHiddenStates,
        std::string const& loraWeightsName, cudaStream_t stream);

    //! Key to uniquely identify a captured CUDA graph for the decoding step
    using DecodingGraphKey = std::tuple<int64_t, uintptr_t, uintptr_t, std::string>;

    //! Key to uniquely identify a captured CUDA graph for the base model verification step
    using BaseGraphKey = std::tuple<int64_t, uintptr_t, uintptr_t, uintptr_t, std::string>;

private:
    std::unique_ptr<nvinfer1::IRuntime> mRuntime;                      //!< TensorRT runtime
    std::unique_ptr<nvinfer1::ICudaEngine> mEngine;                    //!< TensorRT engine
    std::unique_ptr<nvinfer1::IExecutionContext> mTRTExecutionContext; //!< Prefill and Generation execution context

    //! Holds the CUDA graph captured for the decoding step. Each CUDA graph is associated with a unique key value
    //! which denote the input/output shapes and other execution properties like LoRA weights.
    hash_utils::HashMap<DecodingGraphKey, std::pair<cudaGraph_t, cudaGraphExec_t>> mCudaGraphs;

    //! Holds the CUDA graph captured for the base model verification step. Each CUDA graph is associated with a unique
    //! key value which denote the input/output shapes and other execution properties.
    hash_utils::HashMap<BaseGraphKey, std::pair<cudaGraph_t, cudaGraphExec_t>> mBaseTreeDecodingCudaGraphs;

    //! Holds the LoRA weights for the LLM engine.
    std::unordered_map<std::string, std::vector<rt::Tensor>> mLoraWeights{};
    std::string mActiveLoraWeightsName{}; //!< Name of currently active LoRA weights

    LLMEngineRunnerConfig mConfig{}; //!< Engine configuration

    //! The Rope CosSinCache tensor that pre-computed prior to engine execution.
    //! The design is to produce better performance and accommodate complex context dependent rope.
    rt::Tensor mPosEncCosSinCache{};

    //! The select token indices tensor is used to select indices from hidden states to pass to
    //! the LM head of LLM model. Enforce to be int64_t to align with ONNX Gather-ND specification.
    rt::Tensor mSelectTokenIndices{};
    rt::Tensor mHostSelectTokenIndices{}; //!< Host tensor for select token indices (pinned memory)

    //! The tensor has different meaning for prefill and decoding phase due to implementation of
    //! the AttentionPlugin. Used as LLM engine input.
    //! For prefill phase, the field denotes the actual content length of input_ids for each sequence.
    //! For decoding phase, this field denotes the cumulative length of the sequence length of prefill
    //!     plus generated tokens (including the length in "current" run).
    rt::Tensor mSequenceContextLengths{};

    //! The LinearKVCache tensor that carried for the LLM model execution.
    //! Also owns recurrent and conv state buffers for hybrid models.
    rt::LinearKVCache mKVCache{};

    //! Dummy input tensor used to reserve space for unused input tensors. We always keep this tensor as zero tensor
    //! because to "void" some computation (ex. use as empty lora weights as if there is no LoRA GEMM).
    rt::Tensor mDummyInputTensor{};

    //! Dummy output tensor used to reserve space for unused output tensors. TRT engines have static I/O, to keep
    //! runtime design clean, we will route unused output tensors to this dummy tensor.
    rt::Tensor mDummyOutputTensor{};

    //! Dummy output tensor for context_embs / lm_hidden_states when the engine exports it but the caller does not need
    //! it.
    rt::Tensor mDummyContextEmbTensor{};

    //! Prefill auxiliary sequence output binding (context_embs or lm_hidden_states).
    std::string mPrefillAuxOutputName{};

    //! The eagle base position ids tensor within the sequence that used by positional encoding.
    rt::Tensor mEagleBasePositionIds{};

    //! The eagle base packed mask tensor to indicate the attention relationship between the base verify nodes.
    rt::Tensor mEagleBasePackedMask{};

    /*!
     * @brief Initialize configuration from JSON file
     * @param configJson JSON configuration object
     * @return True on success, false on failure
     */
    bool initializeConfigFromJson(Json const& configJson) noexcept;

    /*!
     * @brief Validate configuration against engine
     * @return True if valid, false otherwise
     */
    bool validateConfigFromEngine();

    /*!
     * @brief Bind KV cache to engine for prefill and generation of new requests
     * @param activeBatchSize Number of active sequences
     * @return True on success, false on failure
     */
    bool bindKVCacheToEngine(int32_t activeBatchSize);

    //! @brief Validate inputs for prefill step
    bool prefillStepInputValidation(rt::Tensor const& inputsEmbeds, rt::Tensor const& contextLengths,
        rt::Tensor const& outputLogits, rt::OptionalOutputTensor outputHiddenStates,
        rt::OptionalOutputTensor outputContextEmbeds, rt::OptionalInputTensors deepstackEmbeds) noexcept;

    //! @brief Validate inputs for vanilla decoding step
    bool vanillaDecodingStepInputValidation(rt::Tensor const& inputsEmbeds, rt::Tensor const& outputLogits) noexcept;

    //! @brief Prepare inputs for vanilla decoding step (shared between execute and capture)
    bool vanillaDecodingStepPrepareInputs(int32_t activeBatchSize, cudaStream_t stream);

    //! @brief Bind tensors for vanilla decoding step (shared between execute and capture)
    bool vanillaDecodingStepBindTensors(rt::Tensor const& inputsEmbeds, rt::Tensor& outputLogits,
        rt::OptionalOutputTensor outputHiddenStates, rt::OptionalOutputTensor outputContextEmbeds,
        int32_t activeBatchSize);

    //! @brief Validate inputs for Eagle base tree decoding step
    bool eagleBaseTreeDecodingStepInputValidation(rt::Tensor const& baseTreeDecodingInputsEmbeds,
        rt::Tensor const& baseTreeDecodingMask, rt::Tensor const& outputLogits,
        rt::Tensor const& outputHiddenStates) noexcept;

    //! @brief Prepare and bind tensors for Eagle base tree decoding step (shared between execute and capture)
    bool eagleBaseTreeDecodingStepBindTensors(rt::Tensor const& baseTreeDecodingInputsEmbeds, rt::Tensor& outputLogits,
        rt::Tensor& outputHiddenStates, int32_t activeBatchSize);

    bool eagleBaseTreeDecodingStepPrepareInputs(rt::Tensor const& baseTreeDecodingInputsEmbeds,
        rt::Tensor const& baseTreeDecodingMask, int32_t activeBatchSize, cudaStream_t stream);

    //! The Function is used to add a LoRA weights to the LLM engine.
    bool addLoraWeights(std::string const& loraWeightsName, std::string const& loraWeightsPath, cudaStream_t stream);

    /*!
     * @brief Reset LoRA weights to dummy tensors with rank 0
     * @return True on success, false on failure
     */
    bool resetLoraWeights();

    /*!
     * @brief Get maximum dimension required for LoRA weights across all LoRA bindings
     * @return Maximum dimension (k for LoRA A, n for LoRA B), or 0 if no LoRA bindings
     */
    int32_t getMaxLoraWeightsDimension() const;

    /*!
     * @brief Get tensor names of LoRA weights
     * @return Vector of LoRA weight tensor names
     */
    std::vector<std::string> getLoraWeightsTensorNames() const;

    //! @brief Check if LoRA weights are supported
    //! @return True if supported, false otherwise
    bool isLoraWeightsSupported() const noexcept;

    //! @brief Get the KV cache type
    //! @return The KV cache type
    nvinfer1::DataType getKVCacheType() const;

    //! @brief Get the recurrent state dtype from the engine binding (layer 0)
    nvinfer1::DataType getRecurrentStateType() const;

    //! @brief Get the conv state dtype from the engine binding (layer 0)
    nvinfer1::DataType getConvStateType() const;

    //! @brief Validate the KV cache type consistency
    //! @return True if the KV cache type is consistent, false otherwise
    //! @throws std::runtime_error if KV cache has mismatching data type
    bool validateKVCacheType() const;

private:
    /*!
     * @brief Bind KV cache to engine for prefill and generation of new requests (plugin path)
     * @param activeBatchSize Number of active sequences
     * @return True on success, false on failure
     */
    bool bindPluginKVCacheToEngine(int32_t activeBatchSize);

    /*!
     * @brief Bind separate K and V caches to engine for new requests (TRT native path)
     * @param activeBatchSize Number of active sequences
     * @return True on success, false on failure
     */
    bool bindTRTNativeKVCacheToEngine(int32_t activeBatchSize);

    /*!
     * @brief Bind recurrent state buffers for recurrent layers to the engine
     * @param activeBatchSize Number of active sequences
     * @return True on success, false on failure
     */
    bool bindRecurrentStateToEngine(int32_t activeBatchSize);

    /*!
     * @brief Bind conv state tensors to the TensorRT execution context
     *
     * @param activeBatchSize Current batch size to bind
     * @return True on success, false on failure
     */
    bool bindConvStateToEngine(int32_t activeBatchSize);
};

} // namespace rt
} // namespace trt_edgellm
