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

/*! \brief Maximum number of top log-probabilities that can be requested per generated token.
 *
 *  The cap applies to the per-token K of each individual request; it is not a cumulative
 *  budget across requests or batches. A batch computes at the maximum K requested by its
 *  requests. Input parsing rejects values above this; the runtime clamps as a backstop for
 *  direct API callers. */
inline constexpr int32_t kMaxLogprobsK = 50;

/*!
 * @brief Decide whether sampling parameters require non-greedy decoding
 *
 * Greedy decoding is used for the default tuple `(temperature=1.0, topK<=1, topP=1.0)`
 * as well as near-zero temperatures, which are treated as greedy to avoid unstable softmax.
 *
 * @param temperature Sampling temperature
 * @param topK Top-k sampling parameter
 * @param topP Top-p sampling parameter
 * @return True when top-k / top-p sampling should be used, false for greedy top-1
 */
bool shouldUseNonGreedySampling(float temperature, int64_t topK, float topP) noexcept;

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
 * \throws std::runtime_error If CUDA operations fail
 */
void topKtopPSamplingFromLogits(rt::Tensor const& logits, rt::Tensor& selectedIndices, SamplingParams const& params,
    rt::Tensor& workspace, cudaStream_t stream, uint64_t philoxSeed = 42, uint64_t philoxOffset = 0);

/*!
 * \brief Apply sparse per-batch logit biases in place before sampling.
 *
 * Biases are represented in CSR-style flattened buffers. For each batch row
 * `b`, entries in `[offsets[b], offsets[b + 1])` are added to
 * `logits[b, tokenIds[i]]`. Invalid token IDs are ignored defensively; callers
 * should still validate token IDs before invoking this helper.
 *
 * \param[in,out] logits Logits tensor [GPU, Float] with shape [batch-size, vocab-size]
 * \param[in] tokenIds Flattened biased token IDs [GPU, Int32] with shape [num-biased-tokens]
 * \param[in] biasValues Flattened bias values [GPU, Float] with shape [num-biased-tokens]
 * \param[in] offsets Per-batch CSR offsets [GPU, Int32] with shape [batch-size + 1]
 * \param[in] stream CUDA stream to execute the kernel
 * \throws std::runtime_error If tensor validation or CUDA launch fails
 */
void applyLogitBias(rt::Tensor& logits, rt::Tensor const& tokenIds, rt::Tensor const& biasValues,
    rt::Tensor const& offsets, cudaStream_t stream);

/*!
 * \brief Apply sparse per-slot additive logit biases to repeated logits rows.
 *
 * This variant is used by speculative verification tensors where each active
 * request slot owns multiple contiguous logits rows. Row `r` consumes the CSR
 * bias entries for slot `r / rowsPerSlot`.
 *
 * \param[in,out] logits Logits tensor [GPU, Float] with shape [batch-size * rowsPerSlot, vocab-size]
 * \param[in] tokenIds Flattened biased token IDs [GPU, Int32] with shape [num-biased-tokens]
 * \param[in] biasValues Flattened bias values [GPU, Float] with shape [num-biased-tokens]
 * \param[in] offsets Per-slot CSR offsets [GPU, Int32] with shape [batch-size + 1]
 * \param[in] rowsPerSlot Number of contiguous logits rows for each active request slot
 * \param[in] stream CUDA stream to execute the kernel
 * \throws std::runtime_error If tensor validation or CUDA launch fails
 */
void applyLogitBiasRepeatedRows(rt::Tensor& logits, rt::Tensor const& tokenIds, rt::Tensor const& biasValues,
    rt::Tensor const& offsets, int32_t rowsPerSlot, cudaStream_t stream);

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
 * \throws std::runtime_error If CUDA operations fail
 */
void selectAllTopK(rt::Tensor const& input, rt::OptionalOutputTensor topKValues, rt::Tensor& topKIndices, int32_t topK,
    rt::Tensor& workspace, cudaStream_t stream);

/*!
 * \brief Scatter selected top-k logits into a dense probability tensor.
 *
 * topKValues/topKIndices are [batch-size, top-k] outputs from selectAllTopK().
 * The output probabilities tensor is [batch-size, vocab-size] and is zero
 * outside the selected top-k support.
 */
void topKLogitsToDenseProbabilities(rt::Tensor const& topKValues, rt::Tensor const& topKIndices,
    rt::Tensor& probabilities, int32_t vocabSize, float temperature, cudaStream_t stream);

/*!
 * \brief Select argmax and compute softmax entropy for each input row.
 *
 * Returns the top-1 token ID and entropy for every row in a logits tensor.
 * The logits are temperature-scaled for entropy computation; argmax is
 * unchanged for positive temperatures.
 *
 * \param[in] input Input tensor [GPU, Float/Half/BF16] with shape [rows, vocab-size]
 * \param[out] topIndices Top-1 indices [GPU, Int32] with shape [rows, 1]
 * \param[out] entropy Entropy values [GPU, Float] with shape [rows]
 * \param[in] temperature Softmax temperature
 * \param[in] stream CUDA stream to execute the kernel
 */
void selectArgmaxAndComputeEntropy(
    rt::Tensor const& input, rt::Tensor& topIndices, rt::Tensor& entropy, float temperature, cudaStream_t stream);

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
 * \throws std::runtime_error if topK and topP are both not set
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
size_t getSelectAllTopKWorkspaceSize(int32_t batchSize, int32_t vocabSize, int32_t topK) noexcept;

/*!
 * \brief Get workspace size required for extractTopKLogprobs.
 *
 * The workspace holds the intermediate log-softmax buffer ([batch-size, vocab-size] floats,
 * 256-byte aligned) followed by the selectAllTopK temporary storage. It is NOT a plain
 * selectAllTopK workspace; always size logprobs workspaces with this function.
 *
 * \param[in] batchSize Batch size
 * \param[in] vocabSize Vocabulary size
 * \param[in] topK Number of top log-probabilities to extract
 * \return Required workspace size in bytes
 */
size_t getExtractTopKLogprobsWorkspaceSize(int32_t batchSize, int32_t vocabSize, int32_t topK) noexcept;

/*!
 * \brief Extract top-K log-probabilities from raw logits.
 *
 * Computes numerically-stable log-softmax over the full vocabulary for each batch element,
 * then selects the top-K (token_id, log_prob) pairs sorted by descending probability.
 * No temperature scaling is applied — logprobs match OpenAI convention: log(softmax(logits)).
 *
 * \param[in]  logits          Input logits [GPU, Float] with shape [batch-size, vocab-size]
 * \param[out] logprobsValues  Top-K log-prob values [GPU, Float] with shape [batch-size, top-K]
 * \param[out] logprobsIndices Top-K token indices [GPU, Int32] with shape [batch-size, top-K]
 * \param[in]  topK            Number of top elements to select (must be <= kMaxLogprobsK)
 * \param[in,out] workspace    Workspace buffer [GPU, Int8]; holds the log-softmax buffer and the
 *                             top-K selection temp storage. Size it with
 *                             getExtractTopKLogprobsWorkspaceSize().
 * \param[in]  stream          CUDA stream
 * \throws std::runtime_error If CUDA operations fail
 */
void extractTopKLogprobs(rt::Tensor const& logits, rt::Tensor& logprobsValues, rt::Tensor& logprobsIndices,
    int32_t topK, rt::Tensor& workspace, cudaStream_t stream);

/*!
 * \brief Spec-decode verify: gather accepted logit rows from the full verify-tree logits tensor.
 *
 * For each (batch, depth) pair, copies the logit row at tree position
 * acceptedIndices[batch * maxAcceptDepth + depth] into the output gathered tensor at row
 * batch * maxAcceptDepth + depth.  Enables running log-softmax only on the accepted rows
 * (typically 4–7) rather than the full verify tree (60 rows). Used only by speculative-decode
 * verification (EAGLE / MTP) when collecting per-token logprobs.
 *
 * \param[in]  logits          GPU float32 [batchSize * verifyTreeSize, vocabSize]
 * \param[in]  acceptedIndices GPU int32   [batchSize * maxAcceptDepth] — tree-position indices
 * \param[out] gathered        GPU float32 [batchSize * maxAcceptDepth, vocabSize]
 * \param[in]  batchSize       Active batch size
 * \param[in]  verifyTreeSize  Rows per batch item in logits
 * \param[in]  maxAcceptDepth  draftingStep + 1
 * \param[in]  vocabSize       Vocabulary size
 * \param[in]  stream          CUDA stream
 */
void gatherSpecVerifyAcceptedLogitRows(rt::Tensor const& logits, rt::Tensor const& acceptedIndices,
    rt::Tensor& gathered, int32_t batchSize, int32_t verifyTreeSize, int32_t maxAcceptDepth, int32_t vocabSize,
    cudaStream_t stream);

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
 * \throws std::runtime_error If CUDA operations fail
 */
void mapReducedVocabToFullVocab(rt::Tensor& vocabIds, rt::Tensor const& vocabMappingTable, cudaStream_t stream);

} // namespace trt_edgellm
