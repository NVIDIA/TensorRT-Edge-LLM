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
#include "runtime/llmRuntimeUtils.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace trt_edgellm
{

namespace tokenizer
{
class Tokenizer;
}

namespace rt
{

using tokenizer::Tokenizer;

struct DecodingInferenceContext;

/*!
 * @brief Grammar-constrained decoding backed by XGrammar.
 *
 * Owns the tokenizer info, the compiled-grammar cache, and one grammar matcher per active
 * batch slot.
 *
 * Lifetime mirrors the runtime, but the per-slot matchers are request-local and must be
 * reindexed by @ref compactSlots on every batch eviction, in lockstep with the per-slot
 * vectors in DecodingInferenceContext.
 */
class GuidedDecoder
{
public:
    GuidedDecoder();
    ~GuidedDecoder(); //!< Out-of-line: Impl is incomplete here.
    GuidedDecoder(GuidedDecoder const&) = delete;
    GuidedDecoder& operator=(GuidedDecoder const&) = delete;

    /*!
     * @brief Allocate reusable buffers and record the vocabulary geometry.
     *
     * The tokenizer info is built on the first guided request instead: it costs one
     * `idToPiece` call per output-vocabulary entry.
     *
     * @param maxRowsPerSlot    Logits rows each slot can own in one step: 1 for vanilla decode,
     *                          the speculative verify size otherwise. Sizes the bitmask buffers.
     * @param outputVocabSize   Width of the engine's logits, i.e. the reduced vocabulary
     *                          when the engine prunes, else the full vocabulary
     * @param fullVocabSize     Tokenizer vocabulary size; only used to size the full-to-output
     *                          reverse map, which exists only when the engine prunes
     * @param tokenizer         Borrowed; must outlive this object
     * @param reducedToFullVocabMap Output-space index to full token ID; pass an unallocated
     *                          tensor when the engine does not prune
     * @param stream            Used for the one-off device-to-host copy of that map
     */
    void initialize(int32_t maxBatchSize, int32_t maxRowsPerSlot, int32_t outputVocabSize, int32_t fullVocabSize,
        Tokenizer const* tokenizer, Tensor const& reducedToFullVocabMap, cudaStream_t stream);

    /*!
     * @brief Compile a grammar and install a fresh matcher for one slot.
     *
     * Runs before any GPU work for the request, so a failure marks just this slot. This is
     * the only input-driven step here, hence the only one that catches exceptions.
     *
     * @param[out] failReason Set only when the call returns false
     * @return False when the guide could not be compiled; the slot is left unconstrained
     */
    bool prepareSlot(int32_t slot, GuidedDecodingParams const& params, std::string& failReason);

    //! Drop every matcher.
    void reset();

    //! Reindex per-slot state after a batch eviction. `batchMapping[i]` is the new index
    //! of old slot `i`, or -1 when it was evicted. Must run with the same mapping, and at
    //! the same point, as the DecodingInferenceContext vectors.
    void compactSlots(std::vector<int32_t> const& batchMapping);

    bool hasAnyGrammar() const noexcept;
    bool hasGrammar(int32_t slot) const noexcept;

    //! True once the matcher has accepted the stop token. Filling a mask past that point is
    //! a hard error in XGrammar, so callers must gate on this.
    bool isTerminated(int32_t slot) const noexcept;

    /*!
     * @brief Advance a slot's grammar by one accepted token.
     * @param outputSpaceToken Token index in the engine's output vocabulary, i.e. captured
     *        before `mapReducedVocabToFullVocab`
     * @return False when the grammar rejects the token, which the caller turns into kError
     */
    bool advance(int32_t slot, int32_t outputSpaceToken);

    /*!
     * @brief Advance a slot's grammar over every token the step committed.
     *
     * Speculative decoding commits a variable number of tokens per step, so the vanilla
     * one-token \ref advance does not apply. Tokens up to and including the reasoning-end
     * marker are skipped: the marker is a separator, not constrained output, and consuming it
     * is what flips @p reasoningEnded.
     *
     * @param committedFullSpace Tokens committed this step, in the *full* vocabulary
     * @param[in,out] reasoningEnded Latched to 1 by the marker; constrains from the next token on
     * @return False when the grammar rejected a token, which the caller turns into kError
     */
    bool advanceCommitted(int32_t slot, int32_t const* committedFullSpace, int32_t count, int8_t& reasoningEnded);

    /*!
     * @brief Fill and upload one bitmask row per slot, at each slot's current grammar state.
     *
     * Vanilla decode only. Speculative verification needs a different mask per verify row and
     * must call \ref fillMasksForDraftTree: filling `rowsPerSlot` rows from one un-advanced
     * matcher would just repeat the same mask, which is exactly what a grammar must not do.
     *
     * @param maskSuppressedPerSlot Slots to leave unconstrained this step
     * @param[out] unsatisfiableSlots Slots whose mask came out all-zero, i.e. the grammar
     *                          cannot be satisfied in this engine's vocabulary
     */
    void fillMasks(int32_t activeBatchSize, int32_t rowsPerSlot, std::vector<int8_t> const& maskSuppressedPerSlot,
        std::vector<int32_t>& unsatisfiableSlots, cudaStream_t stream);

    /*!
     * @brief Copy this step's draft chains to the host and mark when they are there.
     *
     * The matchers live on the host but the chains are produced on the device, so the mask fill
     * has to wait for them. Recording an event instead of synchronizing the stream is what lets
     * the caller enqueue the verify forward first and then wait, so the GPU is never idle while
     * the host walks the grammar.
     *
     * @param draftChainIds [activeBatchSize, rowsPerSlot] device tensor of full-vocabulary IDs,
     *        node 0 being the token the previous step committed
     */
    void captureDraftChains(
        Tensor const& draftChainIds, int32_t activeBatchSize, int32_t rowsPerSlot, cudaStream_t stream);

    /*!
     * @brief Copy this step's draft tree, topology included, to the host.
     *
     * Same contract as \ref captureDraftChains, plus the shape of the tree. A chain leaves the
     * topology unset and the walk assumes node `i` descends from node `i - 1`.
     *
     * @param nodeTokenIds [activeBatchSize, rowsPerSlot] full-vocabulary IDs, node 0 the root
     * @param parentIds    [activeBatchSize, rowsPerSlot] parent node index; -1 at the root and
     *                     in padding. The builder appends a node only once its parent is in the
     *                     tree, so `parentIds[i] < i` always holds
     * @param validCounts  [activeBatchSize] nodes actually built; the rest are padding. Absent
     *                     where the builder always fills the whole tree, as EAGLE's does
     */
    void captureDraftTree(Tensor const& nodeTokenIds, Tensor const& parentIds, OptionalInputTensor const& validCounts,
        int32_t activeBatchSize, int32_t rowsPerSlot, cudaStream_t stream);

    //! Draft tokens captured by \ref captureDraftChains or \ref captureDraftTree, valid after
    //! \ref waitForDraftTopology. Both geometries share this buffer.
    int32_t const* hostDraftTokens() const;

    //! Tree topology captured by \ref captureDraftTree; null after \ref captureDraftChains.
    int32_t const* hostDraftParentIds() const;
    int32_t const* hostDraftValidCounts() const;

    //! Block until the copy started by \ref captureDraftChains or \ref captureDraftTree has
    //! landed. Long is normal: the wait covers the drafting forwards, during which the verify
    //! forward is already running.
    void waitForDraftTopology();

    /*!
     * @brief Fill and upload one bitmask row per draft-tree node, for every slot.
     *
     * Walks each slot's tree depth-first, feeding the grammar the token on the way down and
     * rewinding it one step on the way back up, so that row `i` carries the mask for the state
     * after the root-to-node-`i` path. The matcher ends where it started. Node 0 is the token the
     * previous step already committed, so it is not fed to the matcher; the caller must have
     * advanced past it.
     *
     * A node whose token the grammar refuses prunes its subtree: those rows keep
     * `rowNeedsMask == 0` and are never applied. They are unreachable, because acceptance is
     * path-based and the refused node's parent row was masked. Sibling branches are unaffected,
     * which is the whole reason the walk is a DFS and not a sweep over row order.
     *
     * @param draftTokensFullSpace [activeBatchSize, rowsPerSlot] node token IDs in the *full*
     *        vocabulary, as the verify tree carries them
     * @param parentIds [activeBatchSize, rowsPerSlot] parent node index, or null for a chain,
     *        which is the degenerate tree with `parentIds[i] == i - 1`. A non-root node may report
     *        -1 when its parent missed the verify selection; it hangs off nothing, so it is
     *        unreachable and left unmasked
     * @param validCounts [activeBatchSize] nodes actually built, or null for a chain, where
     *        every row is a node
     * @param slotSuppressed Slots to leave entirely unconstrained this step
     * @param reasoningEndedPerSlot Per-slot reasoning state at the start of the step. Read only:
     *        the flag is latched when the step's tokens are committed, in \ref advanceCommitted
     * @param[out] unsatisfiableSlots Slots whose row 0 came out all-zero
     */
    void fillMasksForDraftTree(int32_t activeBatchSize, int32_t rowsPerSlot, int32_t const* draftTokensFullSpace,
        int32_t const* parentIds, int32_t const* validCounts, std::vector<int8_t> const& slotSuppressed,
        std::vector<int8_t> const& reasoningEndedPerSlot, std::vector<int32_t>& unsatisfiableSlots,
        cudaStream_t stream);

    //! Apply the uploaded bitmask to `logits`, shaped [activeBatchSize, rowsPerSlot, vocab].
    void applyMask(Tensor& logits, int32_t activeBatchSize, int32_t rowsPerSlot, cudaStream_t stream);

    //! Compiled-grammar cache footprint in bytes.
    int64_t cacheSizeBytes() const;

private:
    //! Keeps XGrammar out of this header, so a version bump or backend swap rebuilds
    //! one translation unit rather than every consumer of llmInferenceRuntime.h.
    struct Impl;
    std::unique_ptr<Impl> mImpl;
};

bool hasGuidedDecoding(LLMGenerationRequest const& request) noexcept;

//! Token IDs that open a reasoning block, or -1 where the model has no such marker.
std::vector<int32_t> reasoningStartMarkers(Tokenizer const& tokenizer);

//! Token IDs that close a reasoning block, or -1 where the model has no such marker. Single
//! source of truth: the grammar gate and the runtime's thinking bookkeeping must agree on
//! which token ends the block, or the constraint starts at the wrong moment.
std::vector<int32_t> reasoningEndMarkers(Tokenizer const& tokenizer);

/*!
 * @brief Whether the prompt leaves the reasoning block closed, i.e. whether guided decoding
 *        may constrain from the very first generated token.
 *
 * The most recent marker wins. A tokenizer carrying none of these markers has no reasoning
 * phase, and is reported as closed.
 *
 * @param startMarkers Ids opening a reasoning block; entries below zero are absent from the
 *                     tokenizer and ignored
 * @param endMarkers   Ids closing one, same convention
 * @param thinkingEnabled Whether the request allows a reasoning phase. When the prompt carries no
 *                        marker, a disabled reasoning phase is closed while an enabled phase may
 *                        still be opened by the model.
 */
bool reasoningClosedInPrompt(std::vector<int32_t> const& promptTokens, std::vector<int32_t> const& startMarkers,
    std::vector<int32_t> const& endMarkers, bool thinkingEnabled) noexcept;

/*!
 * @brief Constrain one step's logits to the grammar-legal tokens.
 *
 * A slot whose grammar admits nothing ends with FinishReason::kError, keeping the text it
 * generated so far. Slots still inside their thinking block are left unconstrained, so a
 * JSON grammar cannot mask away the opening `<think>`.
 *
 * @param rowsPerSlot Logits rows owned by each slot; 1 for vanilla decode
 */
void applyGuidedDecodingMask(GuidedDecoder& decoder, DecodingInferenceContext& context, Tensor& logits,
    int32_t activeBatchSize, int32_t rowsPerSlot, cudaStream_t stream);

/*!
 * @brief Advance each slot's grammar by the token it just produced.
 *
 * Must run before `thinkingDone` is updated for this token; that ordering is what keeps the
 * `</think>` separator itself out of the grammar.
 *
 * @param outputSpaceIds Sampled indices in the engine's output vocabulary, captured before
 *        the reduced-vocabulary remap
 */
void advanceGuidedDecoding(
    GuidedDecoder& decoder, DecodingInferenceContext& context, int32_t const* outputSpaceIds, int32_t activeBatchSize);

/*!
 * @brief Constrain one speculative verification step's logits, one mask per verify row.
 *
 * Waits for the draft tokens captured earlier in the step, walks each slot's draft tree to build
 * its masks, and applies them. The wait sits here rather than next to the copy so that the verify
 * forward is already enqueued and the grammar walk overlaps it.
 *
 * @param rowsPerSlot Verify rows each slot owns, i.e. the deployment's verify size
 */
void applyGuidedDecodingMaskForDraftTree(GuidedDecoder& decoder, DecodingInferenceContext& context, Tensor& logits,
    int32_t activeBatchSize, int32_t rowsPerSlot, cudaStream_t stream);

/*!
 * @brief Advance each slot's grammar over the tokens this speculative step committed.
 *
 * @param hostAcceptedTokenIds [activeBatchSize, maxAcceptDepth] full-vocabulary IDs
 * @param hostAcceptLengths Per-slot count *after* `appendAcceptedTokens` trimmed it at EOS or
 *        max length; using the raw acceptance would advance past tokens that were discarded
 */
void advanceGuidedDecodingForCommitted(GuidedDecoder& decoder, DecodingInferenceContext& context,
    int32_t const* hostAcceptedTokenIds, int32_t const* hostAcceptLengths, int32_t maxAcceptDepth,
    int32_t activeBatchSize);

/*!
 * @brief Reject guides XGrammar would accept but not honour, and oversized ones.
 *
 * Some JSON Schema keywords compile cleanly and then do nothing, so the output would violate
 * the schema while the API claims it cannot. Compilation cannot catch that, which makes this
 * the only line of defence. The blacklist is chosen by measurement against the pinned
 * XGrammar version and must be re-checked when that pin moves.
 *
 * @param[out] failReason Set only when the call returns false
 * @return False when the request must be rejected outright
 */
bool validateGuidedDecodingParams(GuidedDecodingParams const& params, std::string& failReason);

} // namespace rt
} // namespace trt_edgellm
