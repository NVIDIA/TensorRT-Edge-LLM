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
#include "runtime/decoding/decodingStrategy.h"
#include "runtime/exec/engineExecutor.h"
#include "runtime/llmRuntimeUtils.h"
#include "tokenizer/tokenizer.h"

#include <cstdint>
#include <filesystem>
#include <memory>

namespace trt_edgellm
{
namespace rt
{
namespace decoder_utils
{

//! @brief Load the draft engine from disk and return an EngineExecutor.
std::unique_ptr<EngineExecutor> loadDraftEngine(
    std::filesystem::path const& engineDir, DeploymentConfig const& deployment);

//! @brief Zero the region @p tensor's current shape covers, not its whole allocation.
//!
//! The spec-decode hidden-state buffers are sized for max_input_len but reshaped down to the few
//! tokens a step binds, so clearing getMemoryCapacity() would cost O(max_input_len) per token. The
//! bound region is a contiguous prefix of the allocation, so shape.volume() is sufficient.
void zeroActiveRegion(Tensor& tensor, cudaStream_t stream);

//! @brief Copy accepted tokens from device buffers into the host-side context token lists.
//! On return, hostAcceptLengths holds the number of tokens actually appended per slot.
void appendAcceptedTokens(DecodingInferenceContext& context, Tensor& hostAcceptLengths, Tensor& hostAcceptedTokenIds,
    Tensor const& deviceAcceptLength, Tensor const& deviceAcceptedTokenIds, int32_t maxAcceptDepth,
    tokenizer::Tokenizer const& tokenizer, cudaStream_t stream);

//! @brief Clamp device accept lengths so multi-token speculative commits never exceed max_generate_length.
void clampAcceptLengthsToRemainingGeneration(
    DecodingInferenceContext& context, Tensor& hostAcceptLengths, Tensor& deviceAcceptLength, cudaStream_t stream);

// Logprobs collection is split into a device-side enqueue and a host-side collect so that
// decoding keeps a single host<->device synchronization point per round: decoders call
// enqueueLogprobsD2H() before their round synchronization (the token / accepted-token D2H
// sync), then one of the collect*FromHost() functions after it. No function below
// synchronizes the stream.

//! @brief Enqueue log-softmax + top-K extraction + async D2H staging for one decode step.
//! Device work only — results land in runtime.logprobs.host* after the caller's round sync.
//! @param inputLogits Row-major logits (GPU): [rows, vocabSize].
//! @param rows        Total rows: activeBatchSize (vanilla / prefill) or
//!                    activeBatchSize * rowsPerBatch (spec decode, gathered if needed).
//! @param runtime     Runtime context providing logprobs/sampling buffers.
//! @param topK        Number of top log-probabilities to extract.
//! @param stream      CUDA stream (not synchronized here).
void enqueueLogprobsD2H(
    Tensor const& inputLogits, int32_t rows, DecodingRuntimeContext& runtime, int32_t topK, cudaStream_t stream);

//! @brief Collect staged logprobs into context.stepLogprobs (vanilla / prefill: one row per slot).
//! Call after the round synchronization that followed enqueueLogprobsD2H().
void collectLogprobsFromHost(
    DecodingRuntimeContext& runtime, DecodingInferenceContext& context, int32_t activeBatchSize, int32_t topK);

//! @brief Collect staged logprobs into context.stepLogprobs (multi-row decode: acceptLen rows per slot).
//! Call after the round synchronization (appendAcceptedTokens) that made hostAcceptLens valid.
//! @param rowsPerBatch Max rows per batch item: maxAcceptDepth for EAGLE/MTP, blockSize for DFlash, or
//!                    canvasLen for DiffusionGemma.
void collectSpecLogprobsFromHost(DecodingRuntimeContext& runtime, DecodingInferenceContext& context,
    int32_t activeBatchSize, int32_t rowsPerBatch, int32_t const* hostAcceptLens, int32_t topK);

} // namespace decoder_utils
} // namespace rt
} // namespace trt_edgellm
