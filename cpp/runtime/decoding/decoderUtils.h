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
std::unique_ptr<EngineExecutor> loadDraftEngine(std::filesystem::path const& engineDir,
    DeploymentConfig const& deployment, std::shared_ptr<RtxRuntimeCache> runtimeCache = nullptr,
    EngineRefitOptions refit = {});

//! @brief Validate a DIRECT reduced-to-full vocabulary map and convert it in place to OFFSETS.
void directVocabMapToOffsets(std::vector<int32_t>& vocabMap, int32_t baseVocabSize);

//! @brief Append one sampled token per active slot to the host-side token lists.
//!
//! Slots already marked finished are skipped: they are evicted at the end of this step, so a
//! token appended to them is never fed back and only pollutes the output. This mirrors
//! collectLogprobsFromHost, which skips the same slots, keeping tokenIds and stepLogprobs in
//! step with each other. The appended token is the slot's one pending token.
void appendSampledTokens(DecodingInferenceContext& context, int32_t const* sampledTokenIds, int32_t activeBatchSize);

//! @brief Zero the region @p tensor's current shape covers, not its whole allocation.
//!
//! The spec-decode hidden-state buffers are sized for max_input_len but reshaped down to the few
//! tokens a step binds, so clearing getMemoryCapacity() would cost O(max_input_len) per token. The
//! bound region is a contiguous prefix of the allocation, so shape.volume() is sufficient.
void zeroActiveRegion(Tensor& tensor, cudaStream_t stream);

//! @brief The forced nodes of a grafting round's verify tree, for telling forced accepted positions from predictions.
struct ForcedPathInfo
{
    Tensor const* deviceRows; //!< [activeBatch, maxAcceptDepth] verify rows the accept kernel walked.
    Tensor* hostRows;         //!< Pinned [maxBatch, maxAcceptDepth] mirror, filled by appendAcceptedTokens.
    int8_t const* hostForced; //!< [activeBatch, stride] 1 on a forced verify node.
    int32_t stride;           //!< Verify width of the round.
};

//! @brief Whether accepted position @p k of a slot carries a forced token: the step after it continues from a forced
//! node, whose token replaced the prediction at @p k. The last accepted token is always a prediction. Valid once
//! appendAcceptedTokens has filled the host rows.
bool acceptedPositionIsForced(ForcedPathInfo const& info, int32_t slot, int32_t k, int32_t maxAcceptDepth);

//! @brief Copy accepted tokens from device buffers into the host-side context token lists.
//! The caller must commit the same accept lengths to persistent base state before calling this function.
//! On return, hostAcceptLengths holds the number of tokens actually appended per slot.
//! forcedAcceptanceLength includes the target bonus; -1 keeps natural acceptance accounting.
//! proposedDraftsExact is false when proposedDraftsPerRound is a verify-tree capacity rather than valid proposals.
//! forcedPath, when given, has its host rows filled under the same synchronization; the forced tokens on each
//! slot's path are then neither drafts nor bonus.
void appendAcceptedTokens(DecodingInferenceContext& context, Tensor& hostAcceptLengths, Tensor& hostAcceptedTokenIds,
    Tensor const& deviceAcceptLength, Tensor const& deviceAcceptedTokenIds, int32_t maxAcceptDepth,
    tokenizer::Tokenizer const& tokenizer, cudaStream_t stream, int32_t proposedDraftsPerRound = 0,
    int32_t const* perSlotProposedDrafts = nullptr, int32_t forcedAcceptanceLength = -1,
    bool proposedDraftsExact = true, ForcedPathInfo const* forcedPath = nullptr);

//! @brief Return the forced length including the target bonus, or -1 for natural acceptance.
int32_t forcedAcceptanceLength(DecodingRuntimeContext const& runtime, DecodingInferenceContext const& context);

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
//! @param verifyLogits         Base verify logits [activeBatch, verifyTokenCount, vocab].
//! @param acceptedTokenIndices Device [activeBatch, maxAcceptDepth] verify rows that were accepted.
//! @param hostAcceptLengths    Host accept lengths, as written back by appendAcceptedTokens().
//! @param ownTokens            From applyForcedAcceptance(); empty when forcing is off.
void dumpSpecRound(DecodingInferenceContext& context, HybridCacheManager& cacheManager, KVPageTable const& pageTable,
    KVPageTable const* swaPageTable, Tensor const& verifyLogits, Tensor const& acceptedTokenIndices,
    Tensor const& hostAcceptLengths, std::vector<int32_t> const& ownTokens, int32_t verifyTokenCount,
    int32_t maxAcceptDepth, cudaStream_t stream);

void prepareSpecRaggedBindings(DecodingRuntimeContext& runtime, LLMEngineConfig const& cfg, int32_t kvCacheIndex,
    Tensor const& attentionPositions, Tensor const& committedPastLengths, Tensor const* validCounts,
    Tensor const& selectedTokenIndices, int32_t selectedRows, std::vector<ResidentRef> const* residentRefs,
    int32_t batchSize, int32_t queryWidth, InferenceDims const& dims, cudaStream_t stream);

//! @param heldBackTokens Positions at the end of each row's prefill this pass leaves out. A row seated into a
//!        running batch (DecodingStrategy::seatRow) holds its last prompt position back for the next decode step's
//!        accept pass, which feeds it beside every resident's accepted span.
void prepareSpecPrefillRaggedBindings(DecodingRuntimeContext& runtime, LLMEngineConfig const& cfg, int32_t kvCacheIndex,
    DecodingInferenceContext const& context, int32_t queryWidth, InferenceDims const& dims, cudaStream_t stream,
    int32_t heldBackTokens = 0);

RaggedExecutionBatch buildSpecPrefillRaggedBatch(
    DecodingInferenceContext const& context, int32_t queryWidth, int32_t heldBackTokens = 0);

//! @brief Seat an admitted row's accepted span for a two-engine decoder whose draft trails the target by the
//! span the last verification accepted (EAGLE, MTP).
//!
//! The row's draft prefill held its last prompt position back (heldBackTokens = 1), so between steps the row looks
//! like a resident whose last verification accepted one token: the first token the target sampled, fed to the
//! draft beside the target hidden state at the last prompt position. Writes that span at @p executionRow of the
//! decoder's carried tensors, staged through @p hostSeatStaging, a pinned [B, 2] INT32 tensor only seating writes
//! (every seat follows the seated prefill's synchronization, so an earlier seat's copies have run). @p
//! acceptedHiddenStates holds @p hiddenStride hidden rows per execution row, and @p prefillHiddenStates the seated
//! prefill's hidden states, one row per token.
void seatAcceptedSpan(DecodingInferenceContext const& context, int32_t executionRow, Tensor& acceptedTokenIds,
    Tensor& acceptLength, Tensor& hostSeatStaging, Tensor& acceptedHiddenStates, int32_t hiddenStride,
    Tensor const& prefillHiddenStates, cudaStream_t stream);

//! @brief Copy the leading @p elements elements of @p source into @p destination, of the same data type.
void copyLeadingElements(Tensor& destination, Tensor const& source, int64_t elements, cudaStream_t stream);

//! @brief Copy @p batch sequences of @p sourceStride hidden rows each into @p destination at @p destinationStride
//! rows per sequence (destinationStride >= sourceStride); for a decoder whose verification width varies by round
//! and that carries the accepted spans at a fixed stride.
void copyHiddenRowsStrided(Tensor& destination, int32_t destinationStride, Tensor const& source, int32_t sourceStride,
    int32_t batch, int64_t hiddenSize, cudaStream_t stream);

ExecutionPhase contextPrefillPhase(std::vector<int32_t> const& pastLengths, int32_t activeBatchSize);

//! @brief The last @p contextLength tokens a sequence has fed to the engine, oldest first.
//!
//! All of @p tokens but the last have been fed; the last is next round's root. A reused prefix is
//! missing from @p tokens and comes from @p prompt.
//!
//! @param tokens The prompt past any reused prefix, then the output
//! @param prompt The whole prompt
//! @param committedLength Committed KV length, reused prefix included
//! @param contextLength Tokens to return
//! @param[out] tail Shorter than @p contextLength only when fewer have been fed
void consumedTokenTail(std::vector<int32_t> const& tokens, std::vector<int32_t> const& prompt, int32_t committedLength,
    int32_t contextLength, std::vector<int32_t>& tail);

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
//! Call after the round synchronization that followed enqueueLogprobsD2H(). The staged rows are
//! @p topK wide (the batch's widest width); each slot keeps its own width's prefix and a slot with
//! width zero keeps nothing.
void collectLogprobsFromHost(
    DecodingRuntimeContext& runtime, DecodingInferenceContext& context, int32_t activeBatchSize, int32_t topK);

//! @brief Collect staged logprobs into context.stepLogprobs (multi-row decode: acceptLen rows per slot).
//! Call after the round synchronization (appendAcceptedTokens) that made hostAcceptLens valid.
//! Per-slot widths are handled as in collectLogprobsFromHost.
//! @param rowsPerBatch Max rows per batch item: maxAcceptDepth for EAGLE/MTP, blockSize for DFlash, or
//!                    canvasLen for DiffusionGemma.
void collectSpecLogprobsFromHost(DecodingRuntimeContext& runtime, DecodingInferenceContext& context,
    int32_t activeBatchSize, int32_t rowsPerBatch, int32_t const* hostAcceptLens, int32_t topK);

// Row-set variants for steps that sample only some of the active rows (compact steps). Logits /
// staged row i belongs to context row contextRows[i]; contextRows holds distinct rows in
// [0, context.activeBatchSize). For contextRows = {0, ..., activeBatchSize - 1} each one equals
// its all-rows counterpart above. Guided decoding is excluded: guided requests never share a step.

//! @brief appendSampledTokens() over a row set: sampledTokenIds[i] is appended to context row
//! contextRows[i]; finished rows are skipped.
void appendSampledTokensForRows(
    DecodingInferenceContext& context, int32_t const* sampledTokenIds, std::vector<int32_t> const& contextRows);

//! @brief collectLogprobsFromHost() over a row set: staged row i goes to context row contextRows[i];
//! finished rows and rows with width zero are skipped, and each row keeps its own width's prefix.
//! Stage with enqueueLogprobsD2H(logits, contextRows.size(), ...): top-K extraction and the
//! reduced-vocabulary index remap do not depend on which context row a logits row belongs to.
void collectLogprobsFromHostForRows(DecodingRuntimeContext& runtime, DecodingInferenceContext& context,
    std::vector<int32_t> const& contextRows, int32_t topK);

//! @brief The uniform the slot's next token is sampled with. A slot with requestStableSampling set
//! draws from its seed and absolute position (full prompt + generated tokens), so the request draws
//! the same value whichever logits row it lands on. Any other slot draws from the shared counter
//! stream keyed on kDefaultSamplingSeed, the slot's generated-token count and its row, as release
//! 0.11 did for requests without a seed: such a request repeats its draws when it lands on the same
//! row again, and rows of one batch draw different values.
float samplingUniformForSlot(DecodingInferenceContext const& context, size_t slot) noexcept;

//! @brief The sampling row of each selected context row, hostRows[i] for contextRows[i]: the slot's
//! parameters with the uniform of samplingUniformForSlot, as in prefill and vanilla decode.
void fillSamplingRowsForRows(
    DecodingInferenceContext const& context, std::vector<int32_t> const& contextRows, SamplingRow* hostRows);

//! @brief Enqueue sampling of logits [contextRows.size(), vocab] for the selected context rows.
//! Order: logit bias, the per-row sampler over each row's own parameters and the uniform of its
//! stream (sampleNextTokens over a row set), reduced-to-full vocabulary remap, token broadcast,
//! logprobs staging at the widest width any selected row asks for, and the D2H of the sampled ids
//! into runtime.sampling.hostSelectedTokenIds. Does not synchronize @p stream.
//! @return False if the token broadcast failed.
bool enqueueSamplingForRows(DecodingRuntimeContext& runtime, DecodingInferenceContext& context, Tensor& logits,
    std::vector<int32_t> const& contextRows, cudaStream_t stream);

//! @brief After the step's synchronization, append the tokens and collect the logprobs staged by
//! enqueueSamplingForRows() for the same contextRows.
void commitSampledRows(
    DecodingRuntimeContext& runtime, DecodingInferenceContext& context, std::vector<int32_t> const& contextRows);

//! @brief Sample the next token of every active slot, each with its own parameters and the
//! request-stable uniform for the position it fills, so a slot's token never depends on its batch.
//! Builds one SamplingRow per slot in @p hostRows (pinned), copies them into @p deviceRows and runs
//! the per-row sampler; does not synchronize @p stream.
//! @param logits          [activeBatchSize, vocab] GPU float logits, one row per slot.
//! @param selectedIndices GPU int32 output, reshaped here to [activeBatchSize, 1].
void sampleNextTokens(DecodingInferenceContext const& context, int32_t activeBatchSize, Tensor const& logits,
    Tensor& selectedIndices, Tensor& hostRows, Tensor& deviceRows, Tensor& workspace, cudaStream_t stream);

} // namespace decoder_utils
} // namespace rt
} // namespace trt_edgellm
