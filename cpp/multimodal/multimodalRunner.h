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
     * @param ropeRotaryCosSinDevice RoPE cache tensor (only used by image / language models)
     * @param stream CUDA stream
     * @return True on success, false on failure
     */
    virtual bool preprocess(rt::LLMGenerationRequest const& request, std::vector<std::vector<int32_t>>& batchedInputIds,
        tokenizer::Tokenizer const* tokenizer, [[maybe_unused]] rt::Tensor& ropeRotaryCosSinDevice, cudaStream_t stream)
        = 0;

    /*!
     * @brief Preprocess using already prepared visual inputs from an external pipeline.
     *
     * This bypasses image decoding / resize / patchification in TRT runtime and instead
     * consumes tensors equivalent to processor outputs such as `pixel_values` and `image_grid_thw`.
     *
     * @param request Generation request with text/messages
     * @param batchedInputIds Output batched input token IDs
     * @param tokenizer Tokenizer instance
     * @param ropeRotaryCosSinDevice RoPE cache tensor
     * @param pixelValues Preprocessed visual input tensor [num_patches, input_dim]
     * @param imageGridTHW Image grid tensor [num_images, 3]
     * @param stream CUDA stream
     * @return True on success, false on failure
     */
    virtual bool preprocessPreparedVisual(rt::LLMGenerationRequest const& request,
        std::vector<std::vector<int32_t>>& batchedInputIds, tokenizer::Tokenizer const* tokenizer,
        [[maybe_unused]] rt::Tensor& ropeRotaryCosSinDevice, [[maybe_unused]] rt::Tensor const& pixelValues,
        [[maybe_unused]] rt::Tensor const& imageGridTHW, cudaStream_t stream);

    /*!
     * @brief Used for KVCache saving where we need to conduct the tokenization of the system prompt and generate
     * ND-Rope parameters for the system prompt.
     * @details This function may be a no-op for some multimodal runners and only performs nontrivial work for some
     *          derived subclasses.
     * @param systemPrompt System prompt text
     * @param tokenizer Tokenizer instance
     * @param ropeRotaryCosSinDevice RoPE cache tensor
     * @param stream CUDA stream
     * @return True on success, false on failure
     */
    virtual bool preprocessSystemPrompt([[maybe_unused]] std::string const& systemPrompt,
        [[maybe_unused]] tokenizer::Tokenizer const* tokenizer, [[maybe_unused]] rt::Tensor& ropeRotaryCosSinDevice,
        [[maybe_unused]] cudaStream_t stream);

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

protected:
    multimodal::ModelType mModelType;                      //!< Model type identifier
    std::unique_ptr<nvinfer1::IRuntime> mRuntime;          //!< TensorRT runtime
    std::unique_ptr<nvinfer1::ICudaEngine> mVisualEngine;  //!< Visual encoder engine
    std::unique_ptr<nvinfer1::IExecutionContext> mContext; //!< Execution context
    rt::Tensor mOutputEmbedding;                           //!< Output embeddings
    metrics::MultimodalMetrics mMultimodalMetrics;         //!< Performance metrics
};

} // namespace rt
} // namespace trt_edgellm
