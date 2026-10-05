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
#include "runtime/exec/raggedBatchBuilder.h"
#include "runtime/llmRuntimeUtils.h"
#include "tokenizer/tokenizer.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <vector>

namespace trt_edgellm
{
namespace rt
{
namespace decoder_utils
{

//! @brief Load the draft engine from disk and return an EngineExecutor.
std::unique_ptr<EngineExecutor> loadDraftEngine(
    std::filesystem::path const& engineDir, DeploymentConfig const& deployment);

//! @brief Validate a DIRECT reduced-to-full vocabulary map and convert it in place to OFFSETS.
void directVocabMapToOffsets(std::vector<int32_t>& vocabMap, int32_t baseVocabSize);

//! @brief Append one sampled token per active slot to the host-side token lists.
//!
//! Slots already marked finished are skipped: they are evicted at the end of this step, so a
//! token appended to them is never fed back and only pollutes the output. This mirrors
//! collectLogprobsFromHost, which skips the same slots, keeping tokenIds and stepLogprobs in
//! step with each other.
void appendSampledTokens(DecodingInferenceContext& context, int32_t const* sampledTokenIds, int32_t activeBatchSize);

//! @brief Zero the region @p tensor's current shape covers, not its whole allocation.
//!
//! The spec-decode hidden-state buffers are sized for max_input_len but reshaped down to the few
//! tokens a step binds, so clearing getMemoryCapacity() would cost O(max_input_len) per token. The
//! bound region is a contiguous prefix of the allocation, so shape.volume() is sufficient.
void zeroActiveRegion(Tensor& tensor, cudaStream_t stream);

//! @brief Copy accepted tokens from device buffers into the host-side context token lists.
//! The caller must commit the same accept lengths to persistent base state before calling this function.
//! On return, hostAcceptLengths holds the number of tokens actually appended per slot.
void appendAcceptedTokens(DecodingInferenceContext& context, Tensor& hostAcceptLengths, Tensor& hostAcceptedTokenIds,
    Tensor const& deviceAcceptLength, Tensor const& deviceAcceptedTokenIds, int32_t maxAcceptDepth,
    tokenizer::Tokenizer const& tokenizer, cudaStream_t stream, int32_t proposedDraftsPerRound = 0,
    int32_t const* perSlotProposedDrafts = nullptr);

//! @brief Clamp device accept lengths so multi-token speculative commits never exceed max_generate_length.
//!
//! Runs entirely on @p stream: the per-slot budgets travel as kernel arguments, so the verify round has no D2H
//! copy and no host synchronization before appendAcceptedTokens(). A mid-round host sync drains the stream, and
//! under GPU time-slicing that forfeits a full foreign quantum every round.
void clampAcceptLengthsToRemainingGeneration(
    DecodingInferenceContext const& context, Tensor& deviceAcceptLength, cudaStream_t stream);

// Few-layer numeric validation hooks for a speculative round. Both are no-ops unless the debug
// environment variables are set (see runtime/debug/layerDebugger.h). A speculative round commits a
// variable number of tokens per sequence, so neither can reuse the vanilla path: forcing has to
// trim the acceptance instead of overwriting a single token, and the dump has to pick each
// sequence's own bonus row out of the verify block.

//! @brief Teacher-force this round's acceptance to the golden's tokens.
//!
//! Call after clampAcceptLengthsToRemainingGeneration() and *before* the KV-cache commit, since
//! trimming the acceptance is what keeps a replaced token's stale cache entry out of the commit.
//! @param ownTokens Out: per sequence, the token it would itself have committed at the slot that
//!                  ends up last -- the divergence signal the dump records.
void applyForcedAcceptance(DecodingInferenceContext& context, Tensor& hostAcceptLengths, Tensor& hostAcceptedTokenIds,
    Tensor& deviceAcceptLength, Tensor& deviceAcceptedTokenIds, std::vector<int32_t>& ownTokens, int32_t maxAcceptDepth,
    cudaStream_t stream);

//! @brief Dump one speculative round's committed base state.
//!
//! Call after appendAcceptedTokens(), so the token list and the cache are both final.
//! @param verifyLogits         Base verify logits [activeBatch, verifySize, vocab].
//! @param acceptedTokenIndices Device [activeBatch, maxAcceptDepth] verify rows that were accepted.
//! @param hostAcceptLengths    Host accept lengths, as written back by appendAcceptedTokens().
//! @param ownTokens            From applyForcedAcceptance(); empty when forcing is off.
void dumpSpecRound(DecodingInferenceContext& context, HybridCacheManager& cacheManager, KVPageTable const& pageTable,
    KVPageTable const* swaPageTable, Tensor const& verifyLogits, Tensor const& acceptedTokenIndices,
    Tensor const& hostAcceptLengths, std::vector<int32_t> const& ownTokens, int32_t verifySize, int32_t maxAcceptDepth,
    cudaStream_t stream);

void prepareSpecRaggedBindings(DecodingRuntimeContext& runtime, LLMEngineConfig const& cfg, int32_t kvCacheIndex,
    Tensor const& attentionPositions, Tensor const& committedPastLengths, Tensor const* validCounts,
    Tensor const& selectedTokenIndices, int32_t selectedRows, std::vector<ResidentRef> const* residentRefs,
    int32_t batchSize, int32_t queryWidth, InferenceDims const& dims, cudaStream_t stream);

void prepareSpecPrefillRaggedBindings(DecodingRuntimeContext& runtime, LLMEngineConfig const& cfg, int32_t kvCacheIndex,
    DecodingInferenceContext const& context, int32_t queryWidth, InferenceDims const& dims, cudaStream_t stream);

RaggedExecutionBatch buildSpecPrefillRaggedBatch(DecodingInferenceContext const& context, int32_t queryWidth);

ExecutionPhase contextPrefillPhase(std::vector<int32_t> const& pastLengths, int32_t activeBatchSize);

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
