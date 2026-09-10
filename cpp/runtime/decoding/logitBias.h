/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cstdint>
#include <vector>

namespace trt_edgellm
{
namespace rt
{

struct DecodingInferenceContext;
struct LLMGenerationRequest;

/*!
 * @brief Runtime-owned storage for sparse per-request logit biases.
 *
 * GPU tensors and host staging buffers are allocated once and reused across
 * prefill and vanilla decode steps. An empty full-to-reduced map denotes an
 * identity output vocabulary.
 */
struct LogitBias
{
    Tensor tokenIds;                            //!< Flattened biased token IDs on GPU
    Tensor values;                              //!< Flattened additive bias values on GPU
    Tensor offsets;                             //!< Per-slot CSR offsets on GPU
    std::vector<int32_t> hostOffsets;           //!< Reused host staging for CSR offsets
    std::vector<int32_t> hostTokenIds;          //!< Reused host staging for token IDs
    std::vector<float> hostValues;              //!< Reused host staging for bias values
    std::vector<int32_t> fullToReducedVocabMap; //!< Full token ID to reduced output-vocabulary index
};

/*!
 * @brief Allocate reusable logit-bias buffers for a runtime.
 * @param logitBias Storage to initialize
 * @param maxBatchSize Maximum runtime batch size
 * @throws std::runtime_error If maxBatchSize is invalid or tensor allocation fails
 */
void allocateLogitBias(LogitBias& logitBias, int32_t maxBatchSize);

/*!
 * @brief Build the full-to-reduced vocabulary map used when preparing biases.
 * @param logitBias Storage that owns the inverse map
 * @param reducedToFullVocabMap GPU tensor mapping reduced indices to full token IDs
 * @param fullVocabSize Full tokenizer vocabulary size
 * @param reducedVocabSize Reduced output vocabulary size
 * @param stream CUDA stream used for the device-to-host copy
 * @throws std::runtime_error If tensor metadata, vocabulary sizes, token IDs, or CUDA operations are invalid
 */
void setLogitBiasVocabMap(LogitBias& logitBias, Tensor const& reducedToFullVocabMap, int32_t fullVocabSize,
    int32_t reducedVocabSize, cudaStream_t stream);

/*!
 * @brief Report whether any slot in a request contains logit-bias entries.
 * @param request Batched generation request
 * @return True when at least one request slot has a non-empty logit-bias map
 */
bool hasLogitBias(LLMGenerationRequest const& request) noexcept;

/*!
 * @brief Prepare request-local bias maps in the model output vocabulary.
 * @param logitBias Runtime-owned vocabulary mapping
 * @param request Validated batched generation request
 * @param context Request-local inference context to populate
 * @throws std::runtime_error If a full token ID cannot be mapped safely
 */
void prepareLogitBias(
    LogitBias const& logitBias, LLMGenerationRequest const& request, DecodingInferenceContext& context);

/*!
 * @brief Apply request logit biases to output logits before sampling.
 *
 * Dirty request state is flattened and uploaded lazily. Subsequent decode
 * steps reuse the uploaded buffers until batch compaction marks the state dirty.
 *
 * @param logitBias Runtime-owned GPU buffers and host staging
 * @param logits Base-model output logits to update in place
 * @param context Request-local bias maps and dirty state
 * @param stream CUDA stream used for copies and kernel execution
 * @throws std::runtime_error If tensor reshaping, validation, or CUDA operations fail
 */
void applyLogitBias(LogitBias& logitBias, Tensor& logits, DecodingInferenceContext& context, cudaStream_t stream);

/*!
 * @brief Apply request logit biases to repeated per-slot logits rows before sampling or acceptance.
 * @param logitBias Runtime-owned GPU buffers and host staging
 * @param logits Logits to update in place, shaped [activeBatchSize, rowsPerSlot, vocab] or
 * [activeBatchSize * rowsPerSlot, vocab]
 * @param context Request-local bias maps and dirty state
 * @param rowsPerSlot Number of contiguous logits rows owned by each active request slot
 * @param stream CUDA stream used for copies and kernel execution
 * @throws std::runtime_error If tensor reshaping, validation, or CUDA operations fail
 */
void applyLogitBiasRepeatedRows(
    LogitBias& logitBias, Tensor& logits, DecodingInferenceContext& context, int32_t rowsPerSlot, cudaStream_t stream);

} // namespace rt
} // namespace trt_edgellm
