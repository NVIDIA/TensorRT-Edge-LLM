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

//! Upper bound on one row's top-k: the top-k kernels' cost and workspace grow with k, so one request
//! must not dictate everyone's launch. The same bound TensorRT-LLM enforces for this kernel family;
//! the runtime clamps requests above it (with a warning) and the sampler rejects them.
inline constexpr int32_t kMaxSamplingTopK = 1024;

//! Sampling parameters of one logits row, as a request carries them: the part of SamplingParams
//! that does not depend on the launch. A plain value, so the runtime stores one per slot and rows
//! that compare equal share a launch. topK <= 0 disables top-k.
struct RowSamplingParams
{
    float temperature{1.0F};
    int32_t topK{0};
    float topP{1.0F};
};

bool operator==(RowSamplingParams const& lhs, RowSamplingParams const& rhs) noexcept;
bool operator!=(RowSamplingParams const& lhs, RowSamplingParams const& rhs) noexcept;

/*! \brief A row's parameters from API values.
 *
 *  The API sends a negative top-k to disable it; it is stored as 0 so rows that mean the same
 *  thing compare equal.
 *  \throws std::invalid_argument if topK does not fit the kernels' int32 top-k
 */
RowSamplingParams makeRowSamplingParams(float temperature, int64_t topK, float topP);

/*! \brief Parameters of one sampling launch: the rows' shared RowSamplingParams plus the launch geometry
 */
struct SamplingParams : RowSamplingParams
{
    int32_t batchSize; //!< Number of samples in the batch
    int32_t vocabSize; //!< Size of the vocabulary
    bool useTopK;      //!< Flag indicating if top-K sampling is enabled
    bool useTopP;      //!< Flag indicating if top-P sampling is enabled

    /*! \brief Constructor with default values
     *  \param batchSize_ Number of samples in the batch
     *  \param vocabSize_ Size of the vocabulary
     *  \param temperature_ Temperature parameter (default: 1.0f)
     *  \param topK_ Top-K parameter (default: 0, disabled)
     *  \param topP_ Top-P parameter (default: 1.0f, disabled)
     *
     *  With top-k disabled and top-p at 1.0 (a temperature alone) the row samples from its whole
     *  temperature-scaled distribution through the top-p path, which retains everything at p = 1.0.
     *  \throws std::invalid_argument if temperature is negative
     */
    SamplingParams(
        int32_t batchSize_, int32_t vocabSize_, float temperature_ = 1.0f, int32_t topK_ = 0, float topP_ = 1.0f)
        : RowSamplingParams{temperature_, topK_, topP_}
        , batchSize(batchSize_)
        , vocabSize(vocabSize_)
        , useTopK(topK_ > 0)
        , useTopP(topP_ < 1.0f || topK_ <= 0)
    {
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

    //! A launch over \p batchSize_ rows that all take \p row.
    SamplingParams(int32_t batchSize_, int32_t vocabSize_, RowSamplingParams const& row)
        : SamplingParams(batchSize_, vocabSize_, row.temperature, row.topK, row.topP)
    {
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
 * Everything else samples, including a temperature alone with top-k disabled and top-p at
 * 1.0, which draws from the whole temperature-scaled distribution (see SamplingParams).
 *
 * @param temperature Sampling temperature
 * @param topK Top-k sampling parameter
 * @param topP Top-p sampling parameter
 * @return True when top-k / top-p sampling should be used, false for greedy top-1
 */
bool shouldUseNonGreedySampling(float temperature, int64_t topK, float topP) noexcept;

//! The same decision for a row's parameters.
bool shouldUseNonGreedySampling(RowSamplingParams const& params) noexcept;

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
 * \param[in] randomSeed Random seed for rows without a supplied uniform (default: 42)
 * \param[in] randomOffset Random offset for rows without a supplied uniform; use a new value per decoding step
 *                         (default: 0)
 * \param[in] rowUniforms Optional GPU FP32 tensor [batch-size]. When supplied,
 *                         each row uses this request-derived uniform instead of
 *                         deriving randomness from its transient batch index.
 * \throws std::runtime_error If CUDA operations fail
 */
void topKtopPSamplingFromLogits(rt::Tensor const& logits, rt::Tensor& selectedIndices, SamplingParams const& params,
    rt::Tensor& workspace, cudaStream_t stream, uint64_t randomSeed = 42, uint64_t randomOffset = 0,
    rt::Tensor const* rowUniforms = nullptr);

/*! \brief One row of a per-row sampling launch: a request's parameters as the kernels read them,
 *  with the uniform the row draws.
 *
 *  makeSamplingRow folds a greedy row into top-k 1, so the top-k kernels serve it; topK == 0 sends a
 *  row to the top-p kernels, which sample its whole temperature-scaled distribution at topP == 1.
 */
struct SamplingRow
{
    float temperature{1.0F};
    int32_t topK{1};
    float topP{1.0F};
    float uniform{0.0F};
};

//! The row the kernels run for \p params, drawing \p uniform in [0, 1).
SamplingRow makeSamplingRow(RowSamplingParams const& params, float uniform) noexcept;

//! Workspace bytes for \ref sampleRows over at most \p maxRows rows: the top-k region at
//! \ref kMaxSamplingTopK and the top-p region, both resident because one launch may use both.
size_t getRowSamplingWorkspaceSize(int32_t maxRows, int32_t vocabSize);

/*!
 * \brief Sample one token per row, each row with its own parameters, in a fixed number of launches.
 *
 * The top-k kernels run once over every row with topK > 0 and the top-p kernels once over the rest;
 * rows that belong to the other path return at the top of each kernel. A row's token is what the
 * whole-batch launch of its parameters and uniform would produce (see topKtopPSamplingFromLogits),
 * whatever else is in the batch and however the rows are ordered.
 *
 * \param[in] logits [GPU, Float] with shape [row-count, vocab-size]
 * \param[out] selectedIndices [GPU, Int32] with shape [row-count, 1]
 * \param[in] hostRows The \p rowCount rows, read on the host to plan the launches: every topK in
 *                     [0, kMaxSamplingTopK] and every temperature non-negative
 * \param[in] deviceRows [GPU, Int8] holding the same rows as \p rowCount SamplingRow, copied on \p stream
 * \param[in,out] workspace [GPU, Int8] sized by \ref getRowSamplingWorkspaceSize
 * \throws std::runtime_error on a contract violation or a CUDA failure
 */
void sampleRows(rt::Tensor const& logits, rt::Tensor& selectedIndices, SamplingRow const* hostRows,
    rt::Tensor const& deviceRows, int32_t rowCount, rt::Tensor& workspace, cudaStream_t stream);

//! Construct the exact top-p distribution in the original vocabulary order.
void topPProbabilitiesFromLogits(rt::Tensor const& logits, rt::Tensor& probabilities, float temperature, float topP,
    rt::Tensor& workspace, cudaStream_t stream);

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
 * \brief Sentinel written to grammar-forbidden logits by \ref applyTokenBitmask.
 *
 * Not -inf: this sampler has no NaN/inf guards, and an all--inf row makes the
 * softmax denominator zero (`exp(-inf - (-FLT_MAX))`), which divides by zero
 * downstream. Not -FLT_MAX either, because logits are scaled by `1/temperature`
 * first and `-FLT_MAX * 10` overflows straight back to -inf. -1e30 stays finite
 * until the inverse temperature exceeds ~3e8, and leaves a fully masked row
 * uniformly random rather than undefined.
 */
constexpr float kMaskedLogitValue = -1e30F;

/*!
 * \brief Zero out grammar-forbidden tokens ahead of sampling.
 *
 * The bitmask is dense and bit-packed, one bit per output-vocabulary entry, with a
 * set bit meaning *allowed*. Its width equals the logits width, so no vocabulary
 * lookup is needed. Row `r` of the bitmask constrains row `r` of the logits. Rows are
 * not slots: each slot owns `rowsPerSlot` consecutive rows, exactly as in
 * \ref applyLogitBiasRepeatedRows.
 *
 * \param[in,out] logits Logits [GPU, Float] with shape [num-rows, vocab-size]
 * \param[in] bitmask Packed mask [GPU, Int32] with shape [>= num-rows, ceil(vocab-size / 32)]
 * \param[in] rowNeedsMask Per-row flag [GPU, Int32], shape [>= num-rows]; rows set to 0 are
 *            skipped entirely, so unconstrained requests in a mixed batch cost nothing
 * \param[in] numRows Number of logits rows to constrain
 * \param[in] stream CUDA stream to execute the kernel
 * \throws std::runtime_error If tensor validation or CUDA launch fails
 */
void applyTokenBitmask(rt::Tensor& logits, rt::Tensor const& bitmask, rt::Tensor const& rowNeedsMask, int32_t numRows,
    cudaStream_t stream);

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

//! Return the workspace required by topPProbabilitiesFromLogits().
size_t getTopPProbabilitiesWorkspaceSize(int32_t rows, int32_t vocabSize);

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
 * \param[in]  maxAcceptDepth  draftSteps + 1
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
