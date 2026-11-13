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

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace trt_edgellm
{

namespace tokenizer
{
class Tokenizer;
}

namespace rt
{

class GuidedDecoder;
struct DecodingInferenceContext;
struct LogprobsSlot;
struct SpecVerifyInputs;
struct PipelineIO;
struct SpecDecodeConfig;

namespace decoder_utils
{
struct ForcedPathInfo;
}

enum class JumpSkipReason : int32_t
{
    kNone,
    kEmptyForcedString,
    kIncompleteUtf8,    //!< The forced text holds no complete code point.
    kRetokenizeFailed,  //!< The tokenizer could not re-encode the window text, or spelled something else.
    kNotAtWordBoundary, //!< The tokenizer prepends a word marker and the window text does not start with a space.
    kBoundaryConflict,  //!< Re-encoding changed a token that is already emitted or has KV.
    kNoBudget,
    kPrunedToken,          //!< The first new token is outside the engine's output vocabulary.
    kTrailingTokenDropped, //!< Handing back the last token (see JumpForwardPlanner) left nothing.
    kRespellRefused,       //!< Re-encoding would respell the node a chain attaches to; tree nodes are never rewritten.
    kDraftLonger,          //!< Chain mode: the run is shorter than the cut draft nodes the grammar accepts.
    kCount,
};

char const* jumpSkipReasonName(JumpSkipReason reason) noexcept;

//! Per-sequence counters, reported once when the sequence ends.
struct JumpForwardSlotStats
{
    int32_t jumps{0};
    int32_t forcedTokens{0};     //!< Tokens appended without a forward pass.
    int32_t rolledBackTokens{0}; //!< Sampled tokens a re-encoding replaced.
    int32_t chunkSteps{0};       //!< Catch-up chunks or wide verify rounds this sequence took part in.
    std::array<int32_t, static_cast<size_t>(JumpSkipReason::kCount)> skips{};
};

std::string formatJumpForwardStats(JumpForwardSlotStats const& stats);

struct JumpForwardPlan
{
    int32_t rollbackCount{0};            //!< Window tokens to undo before appending the new ones.
    std::vector<int32_t> fullVocabIds;   //!< New tokens, full vocabulary.
    std::vector<int32_t> outputSpaceIds; //!< The same tokens in the engine's output vocabulary.
    JumpSkipReason skipReason{JumpSkipReason::kNone};

    bool empty() const noexcept
    {
        return fullVocabIds.empty();
    }
};

//! Longest prefix of @p text made of complete UTF-8 code points.
std::string_view completeUtf8Prefix(std::string_view text) noexcept;

/*!
 * @brief Turns a grammar-forced string into tokens that can be appended without a forward pass.
 *
 * The forced text is re-encoded together with the window, the most recent tokens of the sequence,
 * so the result is the tokenizer's own split of the joined text rather than the forced text
 * tokenized on its own. Window tokens at the end that the re-encoding changes are rolled back and
 * replaced; a change to any token before them skips the jump, because those are emitted or have KV.
 *
 * A tokenizer that prepends a word marker (`Tokenizer::prependsSpaceMarker`) only reproduces a
 * window that starts at a word boundary, so the caller must widen the window to one; a window text
 * that does not start with a space is skipped.
 *
 * The last planned token is handed back when another vocabulary token extends its piece and the plan does not
 * complete the grammar: the text after it is not known yet, and a canonical tokenization may merge it with what
 * follows.
 *
 * The grammar state is left unchanged.
 */
class JumpForwardPlanner
{
public:
    //! Generated tokens the window may reach back over to find a word boundary.
    static constexpr size_t kMaxWindowTokens{16};

    //! Both references are borrowed and must outlive the planner.
    //! @param maxJumpTokens Tokens one jump may add to a sequence
    JumpForwardPlanner(tokenizer::Tokenizer const& tokenizer, GuidedDecoder const& decoder, int32_t maxJumpTokens);

    /*!
     * @param slot             Decoder slot whose grammar produced @p forced
     * @param forced           Text the grammar forces next, see `GuidedDecoder::forcedContinuation`
     * @param windowFullIds    Most recent tokens of the sequence, full vocabulary
     * @param mutableTailCount Trailing window tokens that may be rolled back: not emitted, no KV
     * @param budgetTokens     Tokens the sequence may still grow by. Rolled-back tokens are replaced
     *                         and do not count
     * @return An empty plan with its skip reason when nothing can be appended
     */
    JumpForwardPlan plan(int32_t slot, std::string const& forced, std::vector<int32_t> const& windowFullIds,
        int32_t mutableTailCount, int32_t budgetTokens) const;

    /*!
     * @brief The window a jump re-encodes: the last token, widened backwards to a word boundary when the
     *        tokenizer prepends a word marker. Only generated tokens are candidates: prompt pieces may be template
     *        markup the tokenizer cannot re-encode, and the widening is bounded to keep the per-step host work small.
     * @param tokens         The sequence, prompt first
     * @param generatedCount Tokens at the end of @p tokens that were generated
     * @param window         Receives the window
     */
    void buildWindow(std::vector<int32_t> const& tokens, int32_t generatedCount, std::vector<int32_t>& window) const;

private:
    tokenizer::Tokenizer const* mTokenizer;
    GuidedDecoder const* mDecoder;
    int32_t mMaxJumpTokens;
    bool mNeedsWordBoundary; //!< The tokenizer prepends a word marker to text not starting with a space.
};

/*!
 * @brief A verify tree on the host at the grafting width: each slot's draft nodes first, then the forced nodes the
 *        tree walk grafts under them. The rows are pinned, so a round uploads the batch with one copy per array.
 *
 * A node's mask row is its ancestors-or-self set and its depth the number of ancestors the tree holds, derived from
 * its parent the way `buildSpecVerificationTree` derives them: a node whose parent is missing hangs off nothing.
 * Nodes follow their parents.
 */
class HostDraftTree
{
public:
    HostDraftTree(int32_t maxBatchSize, int32_t width);

    int32_t width() const noexcept
    {
        return mWidth;
    }

    //! Load one slot's draft tree, `draftWidth` nodes whose parents are lower indices or -1, and pad the rest.
    void loadDraft(int32_t slot, int32_t const* tokens, int32_t const* parents, int32_t draftWidth);

    //! Load one slot's draft chain, `count` nodes each hanging off the one before, and pad the rest.
    void loadChain(int32_t slot, int32_t const* tokens, int32_t count);

    //! Keep a slot's first `count` nodes and turn the rest back into padding; `count` is between 1 and the node count.
    void truncate(int32_t slot, int32_t count);

    /*!
     * @brief Graft `count` forced tokens under node `attach`: a child already carrying the next token is reused,
     *        the rest are inserted. Every node on the chain is flagged forced and labelled `chainId`.
     * @return Nodes inserted
     * @throws std::runtime_error if the chain does not fit the width
     */
    int32_t insertChain(int32_t slot, int32_t attach, int32_t const* tokens, int32_t count, int8_t chainId);

    int32_t count(int32_t slot) const;
    int32_t const* ids(int32_t slot) const;
    int32_t const* parents(int32_t slot) const;
    int32_t const* depths(int32_t slot) const;
    int8_t const* forced(int32_t slot) const;
    //! Per node, the chain it is on, or -1.
    int8_t const* chainOf(int32_t slot) const;
    //! [width, width] ancestors-or-self rows.
    int8_t const* mask(int32_t slot) const;
    //! The forced child of a node, or -1.
    int32_t forcedChildOf(int32_t slot, int32_t node) const;

    //! @name Whole-batch views for the upload: [maxBatchSize, width], [maxBatchSize, width, width], [maxBatchSize].
    //! @{
    Tensor const& idsTensor() const noexcept
    {
        return mIds;
    }

    Tensor const& parentsTensor() const noexcept
    {
        return mParents;
    }

    Tensor const& depthsTensor() const noexcept
    {
        return mDepths;
    }

    Tensor const& forcedTensor() const noexcept
    {
        return mForced;
    }

    Tensor const& maskTensor() const noexcept
    {
        return mMask;
    }

    Tensor const& validCountsTensor() const noexcept
    {
        return mValidCounts;
    }

    //! @}

private:
    size_t nodeOffset(int32_t slot) const;
    //! A null `parents` loads a chain.
    void load(int32_t slot, int32_t const* tokens, int32_t const* parents, int32_t draftWidth);

    int32_t mMaxBatchSize;
    int32_t mWidth;
    Tensor mIds;
    Tensor mParents;
    Tensor mDepths;
    Tensor mForced;
    Tensor mMask;
    Tensor mValidCounts;
    std::vector<int8_t> mChainOf;
    std::vector<int32_t> mForcedChild;
    std::vector<int32_t> mFirstChild;
    std::vector<int32_t> mNextSibling;
};

/*!
 * @brief The decoder-independent part of a jump: the gate, the plan, and the token-list, logprob and statistics
 *        bookkeeping of applying forced tokens to a slot.
 *
 * A plan whose `rollbackCount` is 1 replaces the slot's sampled token, so its tokens are fed to the grammar from
 * the state before that token; otherwise from the current state.
 */
class JumpForwardCoordinator
{
public:
    struct Applied
    {
        int32_t rolledBack{0}; //!< Sampled tokens removed from the token list: 0 or 1.
        int32_t appended{0};   //!< Plan tokens appended to the token list.
    };

    //! Limits of one slot's grafting round.
    struct TreeWalkLimits
    {
        int32_t maxPathLength;       //!< Accepted path capacity of the round.
        int32_t maxJumpTokens;       //!< Tokens one chain may add.
        int32_t remainingGeneration; //!< Tokens the slot may still generate.
    };

    //! Both references are borrowed and must outlive the coordinator.
    //! @param maxJumpTokens Tokens one jump may add to a sequence
    JumpForwardCoordinator(tokenizer::Tokenizer const& tokenizer, GuidedDecoder& decoder, int32_t maxJumpTokens);

    //! Whether a slot takes part in jumps this step: it asked for jump-forward, is unfinished, has left its
    //! reasoning block and has a live grammar. The same gate as the mask.
    bool slotMayJump(DecodingInferenceContext const& context, int32_t slot) const;

    /*!
     * @brief Plan a jump for one slot.
     *
     * Empty without touching the statistics unless the slot may jump (see slotMayJump); empty with the skip counted
     * when nothing can be appended.
     *
     * @param mutableTailCount Trailing tokens of the slot that may be rolled back: not emitted, no KV
     * @param budget           Tokens the slot may still grow by
     */
    JumpForwardPlan plan(DecodingInferenceContext& context, int32_t slot, int32_t mutableTailCount, int32_t budget);

    /*!
     * @brief Apply a plan to a slot: roll back the sampled token when the plan replaces it, feed the plan to the
     *        grammar once, append the tokens it took to the token list with synthetic logprob rows, and count.
     *
     * A replacing run must spell more than the token it replaces; otherwise nothing is applied and the sampled
     * token is re-accepted.
     *
     * @param sampledOutputSpaceId The slot's sampled token in the output vocabulary
     * @return What changed in the token list; both zero when nothing was applied
     */
    Applied applyPlan(
        DecodingInferenceContext& context, int32_t slot, JumpForwardPlan const& plan, int32_t sampledOutputSpaceId);

    /*!
     * @brief Graft forced chains into one slot's draft tree before it is verified.
     *
     * Walks the draft nodes in index order and feeds each node's root path to the grammar; where the grammar then
     * forces text, the text is planned behind the node and grafted under it, reusing draft children that already
     * carry it. Skipped with their subtrees: a node the grammar refuses or terminates at, and a node beside a forced
     * sibling, which acceptance never reaches. A node already on a chain is not planned again. A plan that would
     * respell the node itself is refused and counted. The matcher ends where it started.
     *
     * @param tree The slot's draft tree, loaded; receives the grafted nodes
     * @return Chains grafted
     */
    int32_t planDraftTree(
        DecodingInferenceContext& context, int32_t slot, HostDraftTree& tree, TreeWalkLimits const& limits);

    /*!
     * @brief Graft forced chains into a slot's draft chain before it is verified; the result stays a chain.
     *
     * Feeds the chain to the grammar from the root and plans the text the grammar forces after each node. A plan the
     * draft after the node already spells flags those nodes forced and the walk goes on. The first plan the draft
     * does not spell ends the walk. If it has at least as many tokens as the draft nodes it departs from that the
     * grammar accepts in a row, it keeps the draft nodes that match it and replaces the rest of the draft with its
     * remaining tokens; otherwise the draft stays whole and the plan is counted as draft_longer. A node on a chain is
     * not planned again. The walk ends at a node the grammar refuses or terminates at. A plan that would respell the
     * node itself is refused and counted. The matcher ends where it started.
     *
     * @param tree The slot's draft chain, loaded; receives the grafted nodes
     * @return Chains grafted
     */
    int32_t planDraftChain(
        DecodingInferenceContext& context, int32_t slot, HostDraftTree& tree, TreeWalkLimits const& limits);

    //! Append one row per token: the token at log-probability 0, then negative-id fillers. A forced token has no
    //! logits row and the grammar gives it probability one.
    static void writeSyntheticLogprobRows(LogprobsSlot& logprobs, int32_t topK, int32_t const* tokens, int32_t count);

    //! Rewrite row `step` as a forced token's row.
    static void writeSyntheticLogprobRow(LogprobsSlot& logprobs, int32_t topK, int32_t step, int32_t token);

private:
    //! With the matcher before ids[first]: how many of the cut nodes ids[cutFrom, last) the grammar accepts in a row
    //! after ids[first, cutFrom). The matcher ends where it started.
    int32_t acceptedRunLength(int32_t slot, int32_t const* ids, int32_t first, int32_t cutFrom, int32_t last);

    tokenizer::Tokenizer const* mTokenizer;
    GuidedDecoder* mDecoder;
    JumpForwardPlanner mPlanner;
    std::vector<int32_t> mWindow;  //!< Re-encoding window scratch, reused across slots and steps.
    std::vector<int32_t> mScratch; //!< Walks: the sequence tail the window may reach back over, then a node's path.
    std::vector<int32_t> mPath;    //!< Walks: a node's root path, root excluded.
    std::vector<int8_t> mSkip;     //!< Tree walk: per draft node, whether it and its subtree are out.
};

//! Where a speculative decoder's draft takes forced chains.
enum class JumpGraftGeometry : int32_t
{
    kTree,  //!< Under any draft node; the verify input stays a tree. See JumpForwardCoordinator::planDraftTree.
    kChain, //!< Once along the draft, which stays a chain. See JumpForwardCoordinator::planDraftChain.
};

/*!
 * @brief The grafting layer of a speculative decoder's jump-forward round: grafts forced chains into the drafted
 *        verify input on the host, uploads the result, and accounts the forced tokens on the accepted paths.
 *
 * A grafting round verifies `jumpWidth` rows per slot: the slot's draft, the nodes grafted into it, then padding; a
 * slot without a chain keeps its draft. Its accept step takes the forced flags and node counts held here, and a tree's
 * verify descriptors are prepared from the mask held here.
 *
 * Per round: walk; after kGrafted, upload, verify and accept at the jump width, pass forcedPath to
 * appendAcceptedTokens, collect logprobs, then account.
 */
class SpecJumpGrafter
{
public:
    struct Limits
    {
        int32_t width;         //!< Draft rows per slot.
        int32_t jumpWidth;     //!< Verify rows per slot of a grafting round.
        int32_t pathCapacity;  //!< Accepted path capacity of a grafting round.
        int32_t maxJumpTokens; //!< Tokens one chain may add.
    };

    enum class Walk
    {
        kNoSlot,  //!< No slot may jump; nothing was captured.
        kNoChain, //!< Nothing was grafted; the guided decoder holds the draft as captured.
        kGrafted, //!< Chains were grafted; upload them before verifying.
    };

    //! Both references are borrowed and must outlive the grafter.
    SpecJumpGrafter(tokenizer::Tokenizer const& tokenizer, GuidedDecoder& guidedDecoder, int32_t maxBatchSize,
        JumpGraftGeometry geometry, Limits const& limits);

    /*!
     * @brief Capture the draft through the guided decoder, wait for it, and graft the chains the grammars plan for the
     *        slots that may jump. The GPU idles until the upload, since verification needs the grafted input.
     * @param drafted The draft at `width` rows per slot without forced flags: parents for a tree, and node counts when
     *                the draft is ragged
     */
    Walk walk(DecodingInferenceContext& context, SpecVerifyInputs const& drafted);

    /*!
     * @brief After kGrafted: reshape the decoder's verify views to `jumpWidth` and write the grafted input into them,
     *        and into the forced flags, node counts and, for a tree, the mask held here.
     * @param parents The tree's parent view; null for a chain
     * @param depths  The tree's depth view; null for a chain
     */
    void upload(Tensor& ids, Tensor* parents, Tensor* depths, cudaStream_t stream);

    //! Whether the last walk grafted, so the round verifies `jumpWidth` rows per slot.
    bool wide() const noexcept;

    //! Device [activeBatch, jumpWidth, jumpWidth] int8 ancestors-or-self rows as uploaded. Tree geometry only.
    Tensor& mask();
    Tensor const& mask() const;

    //! Device [activeBatch, jumpWidth] int8 forced flags as uploaded; null unless wide.
    Tensor const* forcedFlags() const noexcept;

    //! Device [activeBatch] int32 node counts as uploaded, for either geometry; null unless wide.
    Tensor const* validCounts() const noexcept;

    //! appendAcceptedTokens' forced path for a wide round, which copies the accepted rows into pinned rows held here.
    //! @param deviceAcceptedRows [activeBatch, pathCapacity] verify rows the accept step walked
    decoder_utils::ForcedPathInfo forcedPath(Tensor const& deviceAcceptedRows);

    /*!
     * @brief After a wide round's appendAcceptedTokens and logprob collection: for each jump-forward slot, count the
     *        round, the forced tokens on its accepted path and the chains they came from, and rewrite each forced
     *        token's logprob row, which holds the verify logits of the node whose prediction the token replaced.
     * @param hostAcceptLengths [activeBatch] tokens appended, as appendAcceptedTokens left them
     * @param hostAcceptedIds   [activeBatch, acceptCapacity] accepted tokens
     */
    void account(DecodingInferenceContext& context, int32_t const* hostAcceptLengths, int32_t const* hostAcceptedIds,
        int32_t acceptCapacity);

    //! The last walk's grafted input.
    HostDraftTree const& hostTree() const noexcept;

private:
    tokenizer::Tokenizer const* mTokenizer;
    GuidedDecoder* mGuidedDecoder;
    JumpGraftGeometry mGeometry;
    Limits mLimits;
    HostDraftTree mHostTree;
    Tensor mMask;
    Tensor mForcedFlags;
    Tensor mValidCounts;
    Tensor mHostAcceptedRows;
    std::optional<JumpForwardCoordinator> mCoordinator;
    //! Per slot, the last walk: chains grafted, chains at the root, rows added to the draft.
    std::vector<int32_t> mGraftedChains;
    std::vector<int32_t> mRootChains;
    std::vector<int32_t> mAddedNodes;
    int32_t mBatchSize{0}; //!< Active batch of the last walk.
    bool mWide{false};
};

/*!
 * @brief A speculative decoder's share of jump-forward: the grafter of a deployment that can jump, whether the current
 *        round grafted, and the steps every grafting decoder performs the same way. Passive: a decoder calls each
 *        helper where its step allows, keeps the rest of its round as it is, and may use the grafter directly.
 */
class JumpGraftSession
{
public:
    using Walk = SpecJumpGrafter::Walk;

    //! Empty until `initialize` finds a deployment that can jump.
    JumpGraftSession() = default;

    //! Builds the grafter when `specConfig.maxJumpTokens > 0`; otherwise no round ever grafts. `plainWidth` is the
    //! draft rows per slot, which a decoder may size below `specConfig.verifyTokenCount`.
    void initialize(tokenizer::Tokenizer const& tokenizer, GuidedDecoder& guidedDecoder, int32_t maxBatchSize,
        JumpGraftGeometry geometry, SpecDecodeConfig const& specConfig, int32_t plainWidth);

    bool canJump() const noexcept
    {
        return mGrafter.has_value();
    }

    //! Whether this request set's slots may jump: it asked for jump-forward on a guided deployment that can jump.
    bool active(DecodingInferenceContext const& context) const noexcept;

    //! Walks the drafted input when active and records whether the round grafted; kNoSlot when inactive.
    Walk walk(DecodingInferenceContext& context, SpecVerifyInputs const& drafted);

    //! Forgets the last walk, for a decoder whose round may skip `walk`.
    void reset() noexcept
    {
        mWide = false;
    }

    //! Whether the current round verifies the grafted input at the jump width.
    bool wide() const noexcept
    {
        return mWide;
    }

    //! The grafter; only on a deployment that can jump.
    SpecJumpGrafter& grafter()
    {
        return *mGrafter;
    }

    SpecJumpGrafter const& grafter() const
    {
        return *mGrafter;
    }

    //! Writes a grafting round's tree descriptors at the jump width from the grafter's mask, over the ones the draft
    //! wrote.
    void prepareGraftedTreeDescriptors(
        PipelineIO& io, Tensor const& kvCacheLengths, int32_t activeBatchSize, cudaStream_t stream) const;

    //! The current round's values: the grafting round's when wide, the plain ones otherwise.
    int32_t verifyWidth(int32_t plain) const noexcept
    {
        return mWide ? mJumpWidth : plain;
    }

    int32_t acceptDepth(int32_t plain) const noexcept
    {
        return mWide ? mPathCapacity : plain;
    }

    Tensor const* validCounts(Tensor const* plain) const noexcept;
    Tensor const* forcedFlags() const noexcept;

    //! appendAcceptedTokens' forced path of a wide round; empty otherwise.
    std::optional<decoder_utils::ForcedPathInfo> forcedPath(Tensor const& deviceAcceptedRows);

    //! After a wide round's appendAcceptedTokens and logprob collection; nothing otherwise.
    void account(DecodingInferenceContext& context, int32_t const* hostAcceptLengths, int32_t const* hostAcceptedIds,
        int32_t acceptCapacity);

    //! For graph capture: adds the grafting round's verify width past the plain one at the front, and its accepted
    //! path past the plain depths when `acceptLengths` is given.
    void addGraphTiers(std::vector<int32_t>& verifyWidths, std::vector<int32_t>* acceptLengths) const;

private:
    std::optional<SpecJumpGrafter> mGrafter;
    int32_t mJumpWidth{0};
    int32_t mPathCapacity{0};
    bool mWide{false};
};

} // namespace rt
} // namespace trt_edgellm
