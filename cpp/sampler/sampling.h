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

#include "common/logger.h"
#include "common/tensor.h"
#include <cstdint>
#include <stdexcept>

namespace trt_edgellm
{

/*! \brief Structure to hold sampling parameters
 */
struct SamplingParams
{
    int32_t batchSize; //!< Number of samples in the batch
    int32_t vocabSize; //!< Size of the vocabulary
    float temperature; //!< Temperature parameter for sampling (higher = more random)
    int32_t topK;      //!< Top-K sampling parameter (0 = disabled)
    float topP;        //!< Top-P (nucleus) sampling parameter (1.0 = disabled)
    bool useTopK;      //!< Flag indicating if top-K sampling is enabled
    bool useTopP;      //!< Flag indicating if top-P sampling is enabled

    /*! \brief Constructor with default values
     *  \param batchSize_ Number of samples in the batch
     *  \param vocabSize_ Size of the vocabulary
     *  \param temperature_ Temperature parameter (default: 1.0f)
     *  \param topK_ Top-K parameter (default: 0, disabled)
     *  \param topP_ Top-P parameter (default: 1.0f, disabled)
     *  \throws std::invalid_argument if neither topK nor topP is set, or if temperature is invalid
     */
    SamplingParams(
        int32_t batchSize_, int32_t vocabSize_, float temperature_ = 1.0f, int32_t topK_ = 0, float topP_ = 1.0f)
        : batchSize(batchSize_)
        , vocabSize(vocabSize_)
        , temperature(temperature_)
        , topK(topK_)
        , topP(topP_)
        , useTopK(topK_ > 0)
        , useTopP(topP_ < 1.0f)
    {
        if (!useTopK && !useTopP)
        {
            throw std::invalid_argument("Either topK or topP must be set");
        }

        if (temperature < 0.0f)
        {
            throw std::invalid_argument("Temperature must be greater than 0.0f");
        }

        if (temperature < 1e-3f)
        {
            if (topK != 1 || topP != 1.0f)
            {
                LOG_WARNING(
                    "Temperature is 0.0f, but topK is not 1 or topP is not 1.0f, this may cause numerical instability. "
                    "Setting topK to 1 and topP to 1.0f");
                topK = 1;
                topP = 1.0f;
                useTopK = true;
                useTopP = false;
            }
        }
    }
};

/*! \brief Forward declaration for internal workspace structure
 */
struct SamplingWorkspace;

/*!
 * \brief Main sampling function for top-K and top-P sampling from logits.
 *
 * Performs token sampling using top-K and/or top-P (nucleus) sampling strategies
 * on the input logits. The function applies temperature scaling and returns the
 * selected token indices for each batch element.
 *
 * \param[in] logits Input logits tensor [GPU, Float] with shape [batch-size, vocab-size]
 * \param[out] selectedIndices Selected token indices [GPU, Int32] with shape [batch-size, 1]
 * \param[in] params Sampling parameters including batch size, vocab size, temperature, top-K, and top-P values
 * \param[in,out] workspace Workspace buffer [GPU, Int8] for intermediate computations
 * \param[in] stream CUDA stream to execute the kernel
 * \param[in] philoxSeed Random seed for sampling (default: 42)
 * \param[in] philoxOffset Random offset for sampling (default: 0)
 */
void topKtopPSamplingFromLogits(rt::Tensor const& logits, rt::Tensor& selectedIndices, SamplingParams const& params,
    rt::Tensor& workspace, cudaStream_t stream, uint64_t philoxSeed = 42, uint64_t philoxOffset = 0);

/*!
 * \brief Select all top-K elements from input tensor.
 *
 * Returns topK indices and raw values from input with no transformations applied.
 * This function identifies the K largest elements in each batch and returns their
 * indices and optionally their values.
 *
 * \param[in] input Input tensor [GPU, Float] with shape [batch-size, vocab-size]
 * \param[out] topKValues Optional top-K values [GPU, Float] with shape [batch-size, top-K]. Can be std::nullopt if
 * values not needed
 * \param[out] topKIndices Top-K indices [GPU, Int32] with shape [batch-size, top-K]
 * \param[in] topK Number of top elements to select
 * \param[in,out] workspace Workspace buffer [GPU, Int8] for intermediate computations
 * \param[in] stream CUDA stream to execute the kernel
 */
void selectAllTopK(rt::Tensor const& input, rt::OptionalOutputTensor topKValues, rt::Tensor& topKIndices, int32_t topK,
    rt::Tensor& workspace, cudaStream_t stream);

/*!
 * \brief Get workspace size required for top-K/top-P sampling (FP32 only).
 *
 * Calculates the amount of GPU memory needed for intermediate computations
 * during the sampling operation. The workspace must be allocated before
 * calling topKtopPSamplingFromLogits().
 *
 * \param[in] batchSize Batch size for sampling
 * \param[in] vocabSize Vocabulary size
 * \param[in] params Sampling parameters
 * \return Required workspace size in bytes
 */
size_t getTopKtopPSamplingWorkspaceSize(int32_t batchSize, int32_t vocabSize, SamplingParams const& params);

/*!
 * \brief Get workspace size required for selectAllTopK operation (FP32 only).
 *
 * Calculates the amount of GPU memory needed for intermediate computations
 * during the top-K selection operation. The workspace must be allocated before
 * calling selectAllTopK().
 *
 * \param[in] batchSize Batch size for selection
 * \param[in] vocabSize Vocabulary size
 * \param[in] topK Number of top elements to select
 * \return Required workspace size in bytes
 */
size_t getSelectAllTopKWorkspaceSize(int32_t batchSize, int32_t vocabSize, int32_t topK);

/*!
 * \brief Map reduced vocabulary IDs to full vocabulary IDs using a lookup table (in-place).
 *
 * Performs in-place mapping from reduced vocabulary space to full vocabulary space
 * using the provided mapping table: vocabIds[i] = vocabMappingTable[vocabIds[i]]
 *
 * The operation is performed in-place, modifying the input tensor directly.
 *
 * \param[in,out] vocabIds Tensor [GPU, Int32] containing reduced vocabulary IDs as input,
 *                         will be overwritten with full vocabulary IDs as output
 * \param[in] vocabMappingTable Lookup table [GPU, Int32] with shape [reduced_vocab_size] mapping reduced IDs to full
 * IDs
 * \param[in] stream CUDA stream to execute the kernel
 */
void mapReducedVocabToFullVocab(rt::Tensor& vocabIds, rt::Tensor const& vocabMappingTable, cudaStream_t stream);

} // namespace trt_edgellm
