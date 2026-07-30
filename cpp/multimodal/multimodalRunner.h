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
#include "modelTypes.h"
#include "profiling/metrics.h"
#include "runtime/imageUtils.h"
#include "runtime/llmRuntimeUtils.h"
#include "tokenizer/tokenizer.h"
#include <cuda_fp16.h>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

namespace trt_edgellm
{
namespace rt
{

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
     * @return Unique pointer to created runner
     * @throws std::runtime_error If model type is unknown or runner creation fails
     */
    static std::unique_ptr<MultimodalRunner> create(std::string const& multimodalEngineDir, int32_t llmMaxBatchSize,
        int64_t llmMaxPositionEmbeddings, cudaStream_t stream);

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
     * @return True on success, false on failure
     */
    virtual bool preprocess(rt::LLMGenerationRequest const& request, std::vector<std::vector<int32_t>>& batchedInputIds,
        tokenizer::Tokenizer const* tokenizer, [[maybe_unused]] rt::OptionalOutputTensor mropeCosSinOut,
        cudaStream_t stream, bool imageOnly = false) = 0;

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

    /*!\brief Replace the prepared encoder input with the selected artifact subset.
     *
     * The request contains only the selected media buffers, while originalItemIndices identifies their positions in
     * the media list inspected by preprocess(). Vision runners rebuild their packed input from the filtered request;
     * audio runners retain the already-extracted mel tensors at the supplied indices.
     */
    virtual bool prepareArtifactSubset([[maybe_unused]] rt::LLMGenerationRequest const& request,
        [[maybe_unused]] std::vector<size_t> const& originalItemIndices, [[maybe_unused]] cudaStream_t stream)
    {
        return false;
    }

    //! Encoder-output rows for each media buffer retained by the most recent preprocess(), in flattened order.
    //! Empty asks the runtime to use the legacy one-placeholder-run-per-buffer mapping.
    virtual std::vector<int64_t> preparedArtifactRowCounts() const
    {
        return {};
    }

    //! Drop request-local prepared encoder inputs when an artifact hit makes infer() unnecessary.
    virtual void discardPreparedInput() noexcept {}

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
};

} // namespace rt
} // namespace trt_edgellm
