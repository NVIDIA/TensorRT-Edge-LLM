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

#include "common/tensor.h"
#include "common/trtUtils.h"
#include "multimodal/common/modelTypes.h"
#include "profiling/metrics.h"
#include "runtime/imageUtils.h"
#include "runtime/llmRuntimeUtils.h"
#include "runtime/state/externalWeightManager.h"
#include "tokenizer/tokenizer.h"
#include <cuda_fp16.h>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <variant>
#include <vector>

namespace trt_edgellm
{
namespace rt
{

//! @brief Client input errors, which preprocess catch blocks rethrow: collapsing them to
//!        `return false` leaves the pybind layer nothing to report but a generic 500.
inline bool isCallerActionable(std::exception const& e) noexcept
{
    std::string_view const message{e.what()};
    return message.find("EDGELLM_INPUT_TOO_LONG") != std::string_view::npos
        || message.find("EDGELLM_BAD_MEDIA_COUNT") != std::string_view::npos;
}

/*!
 * @brief Base class for multimodal vision-language model runners
 *
 * Provides interface for vision encoder processing in VLMs.
 * Subclasses implement specific VLM architectures (Qwen-VL, InternVL, etc.).
 */
class MultimodalRunner
{
public:
    //! @brief Default constructor
    MultimodalRunner() noexcept = default;

    /*!
     * @brief Construct multimodal runner
     * @param engineDir Directory containing engine files
     * @param stream CUDA stream for operations
     * @throws std::runtime_error If engine loading or initialization fails
     */
    MultimodalRunner(std::string const& engineDir, cudaStream_t stream);

    //! @brief Virtual destructor
    virtual ~MultimodalRunner() noexcept = default;

    /*!
     * @brief Get the required context memory size for this engine
     * @return Required context memory size in bytes
     * @note Handles both visual and audio engines
     */
    int64_t getRequiredContextMemorySize() const;

    /*!
     * @brief Set shared context memory for the execution context
     * @param sharedContextMemory Tensor containing the shared device memory (must be on GPU)
     * @return True on success, false if the tensor is too small
     * @note The tensor size must be >= getRequiredContextMemorySize(). Must be called before infer().
     * @note Handles both visual and audio engines
     */
    bool setContextMemory(rt::Tensor& sharedContextMemory);

    /*!
     * @brief Create appropriate multimodal runner instance
     *
     * Factory method that detects model type and creates corresponding runner.
     *
     * @param multimodalEngineDir Directory containing multimodal engine files
     * @param llmMaxBatchSize Maximum batch size from LLM engine
     * @param llmMaxPositionEmbeddings Maximum position embeddings from LLM engine
     * @param stream CUDA stream for operations
     * @param checkpointDir HF checkpoint directory used by runtime weight loading
     * @return Unique pointer to created runner
     * @throws std::runtime_error If model type is unknown or runner creation fails
     */
    static std::unique_ptr<MultimodalRunner> create(std::string const& multimodalEngineDir, int32_t llmMaxBatchSize,
        int64_t llmMaxPositionEmbeddings, cudaStream_t stream, std::string const& checkpointDir = "");

    /*!
     * @brief Preprocess request with images and text
     * @param request Generation request with prompts and images
     * @param batchedInputIds Output batched input token IDs
     * @param tokenizer Tokenizer instance
     * @param mropeCosSinOut Output MRope cos/sin cache. Required (`has_value() == true`) only for MRope-based
     *        multimodal runners (QwenViT, Qwen3OmniAudio), which write per-batch 3D position encodings into it.
     *        Pass `std::nullopt` when the base engine uses standard RoPE — runners with standard RoPE
     *        (InternViT, Phi4MMViT) do not read this parameter.
     * @param stream CUDA stream
     * @param imageOnly When true, only run image preprocessing (skip text tokenization and RoPE
     *        generation). Used for benchmarking where only the visual engine inputs need to be set up.
     * @param skipEncoderWork When true, skip GPU pixel operations (resize, normalize, patchify) and
     *        only compute token-length metadata + text tokenization. Used on encoder embedding cache
     *        hits where the cached embedding will be copied directly.
     * @return True on success, false on failure
     */
    virtual bool preprocess(rt::LLMGenerationRequest const& request, std::vector<std::vector<int32_t>>& batchedInputIds,
        tokenizer::Tokenizer const* tokenizer, [[maybe_unused]] rt::OptionalOutputTensor mropeCosSinOut,
        cudaStream_t stream, bool imageOnly = false, bool skipEncoderWork = false) = 0;

    /*!
     * @brief Used for KVCache saving where we need to conduct the tokenization of the system prompt and generate
     * ND-Rope parameters for the system prompt.
     * @details This function may be a no-op for some multimodal runners and only performs nontrivial work for some
     *          derived subclasses.
     * @param systemPrompt System prompt text
     * @param tokenizer Tokenizer instance
     * @param mropeCosSinOut Output MRope cos/sin cache. Required only for MRope-based runners; otherwise
     *        pass `std::nullopt`. See `preprocess` for the full semantics.
     * @param stream CUDA stream
     * @return True on success, false on failure
     */
    virtual bool preprocessSystemPrompt([[maybe_unused]] std::string const& systemPrompt,
        [[maybe_unused]] tokenizer::Tokenizer const* tokenizer,
        [[maybe_unused]] rt::OptionalOutputTensor mropeCosSinOut, [[maybe_unused]] cudaStream_t stream);

    /*!
     * @brief Run multimodal inference
     * @param stream CUDA stream
     * @return True on success, false on failure
     */
    virtual bool infer(cudaStream_t stream) = 0;

    //! @brief Get output embeddings from vision encoder
    //! @return Reference to output embedding tensor
    virtual rt::Tensor& getOutputEmbedding();

    //! @brief Get deepstack features for Qwen3-VL models
    //! @return Optional deepstack features vector (raw features before embedding lookup)
    virtual rt::OptionalInputTensors getDeepstackFeatures();

    /*!
     * @brief Validate and fill configuration from file
     * @param engineDir Path to engine directory
     * @return True on success, false on failure
     */
    virtual bool validateAndFillConfig(std::string const& engineDir) = 0;

    //! @brief Allocate device buffers
    //! @return True on success, false on failure
    virtual bool allocateBuffer(cudaStream_t stream) = 0;

    //! @brief Get model type
    //! @return Model type enum
    virtual multimodal::ModelType getModelType() const noexcept
    {
        return mModelType;
    }

    //! @brief Get multimodal processing metrics
    //! @return Multimodal metrics
    metrics::MultimodalMetrics const& getMultimodalMetrics() const noexcept
    {
        return mMultimodalMetrics;
    }

    //! @brief Per-media-item token lengths from the last preprocess() call.
    //! One entry per image/audio clip, in the order they appear across all batch requests.
    std::vector<int64_t> const& getLastMediaTokenLengths() const noexcept
    {
        return mLastMediaTokenLengths;
    }

    /*!
     * @brief Load this encoder's externalized weights and bind them to its context
     *
     * A no-op when the encoder config lists neither ``external_weight_files``
     * nor ``checkpoint_weight_bindings``, which is the case for engines that
     * carry their weights inline. Call before the first infer(), and before
     * initialize() on runners that have one, since it may enqueue the engine.
     *
     * @param engineDir Directory holding the encoder engine and its config.json
     * @param checkpointDir HF checkpoint directory, empty when not supplied
     * @param stream CUDA stream used during startup weight preparation
     * @throws std::runtime_error If the weights cannot be loaded or do not
     *         match the engine's input signature
     */
    void loadExternalWeights(std::string const& engineDir, std::string const& checkpointDir, cudaStream_t stream);

protected:
    multimodal::ModelType mModelType; //!< Model type identifier
    AuxStreamSet mAuxStreams{};
    std::unique_ptr<nvinfer1::IRuntime> mRuntime;         //!< TensorRT runtime
    std::unique_ptr<nvinfer1::ICudaEngine> mVisualEngine; //!< Visual encoder engine (null for audio-only runners)
    std::unique_ptr<nvinfer1::IExecutionContext> mVisualContext; //!< Visual execution context
    std::unique_ptr<nvinfer1::ICudaEngine> mAudioEngine;        //!< Audio encoder engine (null for visual-only runners)
    std::unique_ptr<nvinfer1::IExecutionContext> mAudioContext; //!< Audio execution context
    rt::Tensor mOutputEmbedding;                                //!< Output embeddings
    metrics::MultimodalMetrics mMultimodalMetrics;              //!< Performance metrics
    std::vector<int64_t> mLastMediaTokenLengths;                //!< Per-item token lengths from last preprocess
    //! Owns the encoder's externalized weights; the context points into them.
    std::unique_ptr<ExternalWeightManager> mExternalWeights;
    bool mExternalWeightsLoaded{false}; //!< Guards the idempotent load
};

} // namespace rt
} // namespace trt_edgellm
