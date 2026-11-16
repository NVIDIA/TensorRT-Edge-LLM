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

// The host side of jump-forward grafting: the verify tree or chain the walk builds and uploads, and the walk's
// decisions about which draft nodes get a forced chain, on the byte-level BPE vocabulary of the guided-decoding tests.

#include "common/checkMacros.h"
#include "guidedTokenizerFixture.h"
#include "kernels/speculative/speculativeKernels.h"
#include "runtime/decoding/guidedDecoder.h"
#include "runtime/decoding/jumpForward.h"
#include "runtime/state/decodingInferenceContext.h"
#include "scratchDir.h"
#include "testUtils.h"
#include "tokenizer/tokenizer.h"

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

using namespace trt_edgellm;
using namespace guidedFixture;

namespace
{

using Ids = std::vector<int32_t>;
using Flags = std::vector<int8_t>;

//! The BPE fixture with an identity output space, one slot that asked for jump-forward, and a grammar a test installs.
class JumpForwardCoordinatorTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        CUDA_CHECK(cudaStreamCreate(&mStream));
        mDir = makeScratchDir("jumpForwardCoordinator");
        writeBpeTokenizer(mDir);
        ASSERT_TRUE(mTokenizer.loadFromHF(mDir));
        rt::Tensor emptyMap;
        mDecoder.initialize(
            /*maxBatchSize=*/1, /*maxRowsPerSlot=*/1, kBPE_VOCAB_SIZE, kBPE_VOCAB_SIZE, &mTokenizer, emptyMap, mStream);
        mContext.activeBatchSize = 1;
        mContext.jumpForwardEnabled = {1};
        mContext.finishedStates = {0};
        mContext.guidedReasoningEnded = {1};
        mContext.jumpForwardStats = {{}};
        mContext.currentGenerateLengths = {1};
        mContext.batchIndexMapping = {0};
    }

    void TearDown() override
    {
        std::filesystem::remove_all(mDir);
        CUDA_CHECK(cudaStreamDestroy(mStream));
    }

    //! Install an EBNF grammar on the slot and stand it after the root token, as advanceCommitted leaves it.
    void installGrammar(std::string const& ebnf, int32_t rootToken)
    {
        rt::GuidedDecodingParams params;
        params.type = rt::GuideType::kEbnf;
        params.guide = ebnf;
        std::string failReason;
        ASSERT_TRUE(mDecoder.prepareSlot(0, params, failReason)) << failReason;
        ASSERT_TRUE(mDecoder.advance(0, rootToken));
        mContext.tokenIds = {{kOPEN_BRACE, rootToken}};
    }

    int32_t walk(rt::HostDraftTree& tree, int32_t maxPathLength = 32, int32_t remainingGeneration = 64)
    {
        rt::JumpForwardCoordinator coordinator(mTokenizer, mDecoder, /*maxJumpTokens=*/32);
        return coordinator.planDraftTree(mContext, 0, tree, {maxPathLength, /*maxJumpTokens=*/32, remainingGeneration});
    }

    int32_t walkChain(rt::HostDraftTree& tree, int32_t maxPathLength = 32, int32_t remainingGeneration = 64)
    {
        rt::JumpForwardCoordinator coordinator(mTokenizer, mDecoder, /*maxJumpTokens=*/32);
        return coordinator.planDraftChain(
            mContext, 0, tree, {maxPathLength, /*maxJumpTokens=*/32, remainingGeneration});
    }

    int32_t skips(rt::JumpSkipReason reason) const
    {
        return mContext.jumpForwardStats[0].skips[static_cast<size_t>(reason)];
    }

    int32_t totalSkips() const
    {
        int32_t total = 0;
        for (int32_t const count : mContext.jumpForwardStats[0].skips)
        {
            total += count;
        }
        return total;
    }

    static Ids ids(rt::HostDraftTree const& tree)
    {
        return Ids(tree.ids(0), tree.ids(0) + tree.width());
    }

    static Ids nodeIds(rt::HostDraftTree const& tree)
    {
        return Ids(tree.ids(0), tree.ids(0) + tree.count(0));
    }

    static Ids parents(rt::HostDraftTree const& tree)
    {
        return Ids(tree.parents(0), tree.parents(0) + tree.width());
    }

    static Ids depths(rt::HostDraftTree const& tree)
    {
        return Ids(tree.depths(0), tree.depths(0) + tree.width());
    }

    static Flags forced(rt::HostDraftTree const& tree)
    {
        return Flags(tree.forced(0), tree.forced(0) + tree.width());
    }

    static Flags chainOf(rt::HostDraftTree const& tree)
    {
        return Flags(tree.chainOf(0), tree.chainOf(0) + tree.width());
    }

    static Flags maskRow(rt::HostDraftTree const& tree, int32_t node)
    {
        int8_t const* const row = tree.mask(0) + static_cast<size_t>(node) * static_cast<size_t>(tree.width());
        return Flags(row, row + tree.width());
    }

    std::filesystem::path mDir;
    tokenizer::Tokenizer mTokenizer;
    rt::GuidedDecoder mDecoder;
    rt::DecodingInferenceContext mContext;
    cudaStream_t mStream{};
};

// Root 100 with children 10 (node 1) and 11 (node 2); node 1 has child 20 (node 3).
Ids const kDRAFT_IDS{100, 10, 11, 20};
Ids const kDRAFT_PARENTS{-1, 0, 0, 1};

// Given a full draft table and a verify selection whose third node's parent was not selected
// When the host tree loads the tokens and parents the verify-tree kernel built for that selection
// Then its depths and ancestor rows equal the kernel's for every node, the orphan hanging off nothing
TEST_F(JumpForwardCoordinatorTest, HostTreeDerivesDepthsAndMasksAsTheVerifyTreeKernel)
{
    rt::Tensor fullIds({1, 7}, rt::DeviceType::kGPU, trt_edgellm::DataType::kINT32);
    rt::Tensor fullParents({1, 7}, rt::DeviceType::kGPU, trt_edgellm::DataType::kINT32);
    rt::Tensor selected({1, 4}, rt::DeviceType::kGPU, trt_edgellm::DataType::kINT32);
    rt::Tensor treeIds({1, 4}, rt::DeviceType::kGPU, trt_edgellm::DataType::kINT32);
    rt::Tensor treeMask({1, 4, 4}, rt::DeviceType::kGPU, trt_edgellm::DataType::kINT8);
    rt::Tensor treeParents({1, 4}, rt::DeviceType::kGPU, trt_edgellm::DataType::kINT32);
    rt::Tensor treeDepths({1, 4}, rt::DeviceType::kGPU, trt_edgellm::DataType::kINT32);
    // Root 100; children 10, 11; grandchildren 20, 21 under 10 and 22, 23 under 11. Node 11 is not selected.
    copyHostToDevice<int32_t>(fullIds, {100, 10, 11, 20, 21, 22, 23});
    copyHostToDevice<int32_t>(fullParents, {-1, 0, 0, 1, 1, 2, 2});
    copyHostToDevice<int32_t>(selected, {0, 1, 5, 3});
    kernel::buildSpecVerificationTree(
        fullIds, fullParents, selected, treeIds, treeMask, std::ref(treeParents), std::ref(treeDepths), mStream);
    CUDA_CHECK(cudaStreamSynchronize(mStream));
    auto const kernelIds = copyDeviceToHost<int32_t>(treeIds);
    auto const kernelParents = copyDeviceToHost<int32_t>(treeParents);
    auto const kernelDepths = copyDeviceToHost<int32_t>(treeDepths);
    auto const kernelMask = copyDeviceToHost<int8_t>(treeMask);
    ASSERT_EQ(kernelIds, (Ids{100, 10, 22, 20}));
    ASSERT_EQ(kernelParents, (Ids{-1, 0, -1, 1}));
    ASSERT_EQ(kernelDepths, (Ids{0, 1, 0, 2}));

    rt::HostDraftTree tree(1, 4);
    tree.loadDraft(0, kernelIds.data(), kernelParents.data(), 4);

    EXPECT_EQ(depths(tree), kernelDepths);
    EXPECT_EQ(Flags(tree.mask(0), tree.mask(0) + 16), kernelMask);
    EXPECT_EQ(maskRow(tree, 2), (Flags{0, 0, 1, 0})) << "the orphan attends to itself only";
    EXPECT_EQ(maskRow(tree, 3), (Flags{1, 1, 0, 1}));
}

// Given the draft tree 100 -> {10 -> 20, 11} at a width of eight
// When the chain 10, 20, 30 is grafted at the root and the chain 40 under node 11
// Then the draft nodes carrying 10 and 20 are reused and flagged, 30 and 40 are inserted below their parents with the
//      parent's ancestor row plus their own bit, each chain labels its nodes, and the rows past the node count stay
//      padding
TEST_F(JumpForwardCoordinatorTest, InsertChainReusesMatchingChildrenAndInsertsTheRest)
{
    rt::HostDraftTree tree(1, 8);
    tree.loadDraft(0, kDRAFT_IDS.data(), kDRAFT_PARENTS.data(), 4);

    Ids const rootChain{10, 20, 30};
    Ids const sideChain{40};
    EXPECT_EQ(tree.insertChain(0, 0, rootChain.data(), 3, /*chainId=*/0), 1);
    EXPECT_EQ(tree.insertChain(0, 2, sideChain.data(), 1, /*chainId=*/1), 1);

    EXPECT_EQ(tree.count(0), 6);
    EXPECT_EQ(ids(tree), (Ids{100, 10, 11, 20, 30, 40, 0, 0}));
    EXPECT_EQ(parents(tree), (Ids{-1, 0, 0, 1, 3, 2, -1, -1}));
    EXPECT_EQ(depths(tree), (Ids{0, 1, 1, 2, 3, 2, -1, -1}));
    EXPECT_EQ(forced(tree), (Flags{0, 1, 0, 1, 1, 1, 0, 0}));
    EXPECT_EQ(chainOf(tree), (Flags{-1, 0, -1, 0, 0, 1, -1, -1}));
    EXPECT_EQ(maskRow(tree, 4), (Flags{1, 1, 0, 1, 1, 0, 0, 0}));
    EXPECT_EQ(maskRow(tree, 5), (Flags{1, 0, 1, 0, 0, 1, 0, 0}));
    EXPECT_EQ(maskRow(tree, 6), (Flags{0, 0, 0, 0, 0, 0, 0, 0}));
    EXPECT_EQ(tree.forcedChildOf(0, 0), 1);
    EXPECT_EQ(tree.forcedChildOf(0, 1), 3);
    EXPECT_EQ(tree.forcedChildOf(0, 3), 4);
    EXPECT_EQ(tree.forcedChildOf(0, 2), 5);
    EXPECT_EQ(tree.forcedChildOf(0, 4), -1);
}

// Given the grammar "a" ("\"" | "1") ", \"b\":" standing after the root a, and a draft tree whose root has the
//      children " and 1
// When the tree is walked
// Then the quote gets no chain, because the forced text would respell it into ",, and is counted as respell_refused,
//      while the chain , " b ": is grafted under 1
TEST_F(JumpForwardCoordinatorTest, ARespelledNodeIsRefusedAndCountedWhileItsSiblingIsGrafted)
{
    installGrammar(R"(root ::= "a" ("\"" | "1") ", \"b\":")", kA);
    rt::HostDraftTree tree(1, 8);
    Ids const draftIds{kA, kQUOTE, kONE};
    Ids const draftParents{-1, 0, 0};
    tree.loadDraft(0, draftIds.data(), draftParents.data(), 3);

    EXPECT_EQ(walk(tree), 1);

    EXPECT_EQ(skips(rt::JumpSkipReason::kRespellRefused), 1);
    EXPECT_EQ(totalSkips(), 1);
    EXPECT_EQ(tree.count(0), 7);
    EXPECT_EQ(ids(tree), (Ids{kA, kQUOTE, kONE, kCOMMA, kSPACE_QUOTE, kB, kQUOTE_COLON, 0}));
    EXPECT_EQ(parents(tree), (Ids{-1, 0, 0, 2, 3, 4, 5, -1}));
    EXPECT_EQ(forced(tree), (Flags{0, 0, 0, 1, 1, 1, 1, 0}));
    EXPECT_EQ(chainOf(tree), (Flags{-1, -1, -1, 0, 0, 0, 0, -1}));
    EXPECT_EQ(tree.forcedChildOf(0, 1), -1) << "the quote was left alone";
    EXPECT_EQ(mDecoder.forcedContinuation(0), "") << "the walk leaves the matcher after the root";
}

// Given the grammar ":" "\"b\"" standing after the root :, a draft tree of the root and two children the grammar
//      refuses, and a width with room for one more node; then the grammar "a" "1" "2" ", " after the root a and a
//      draft tree of the root alone
// When the first tree is walked, and again with room for two, and the second tree with a path capacity of three
// Then with one row the only token that fits, ", is handed back as extendable and the skip is counted as
//      trailing_dropped with nothing grafted, with two rows the chain " b is grafted, and the second tree's chain is
//      cut to 1 2 so the whole accepted path fits the capacity
TEST_F(JumpForwardCoordinatorTest, AChainIsCutToTheFreeRowsOrThePathCapacityAndAnUngraftableTailIsHandedBack)
{
    installGrammar(R"(root ::= ":" "\"b\"")", kCOLON);
    Ids const draftIds{kCOLON, kONE, kTWO};
    Ids const draftParents{-1, 0, 0};

    rt::HostDraftTree oneRow(1, 4);
    oneRow.loadDraft(0, draftIds.data(), draftParents.data(), 3);
    EXPECT_EQ(walk(oneRow), 0);
    EXPECT_EQ(oneRow.count(0), 3);
    EXPECT_EQ(skips(rt::JumpSkipReason::kTrailingTokenDropped), 1);
    EXPECT_EQ(totalSkips(), 1) << "the refused children are dead, not skips";

    rt::HostDraftTree twoRows(1, 5);
    twoRows.loadDraft(0, draftIds.data(), draftParents.data(), 3);
    EXPECT_EQ(walk(twoRows), 1);
    EXPECT_EQ(twoRows.count(0), 5);
    EXPECT_EQ(ids(twoRows), (Ids{kCOLON, kONE, kTWO, kQUOTE, kB}));
    EXPECT_EQ(parents(twoRows), (Ids{-1, 0, 0, 0, 3}));
    EXPECT_EQ(totalSkips(), 1);

    installGrammar(R"(root ::= "a" "1" "2" ", ")", kA);
    rt::HostDraftTree capped(1, 8);
    Ids const root{kA};
    Ids const rootParent{-1};
    capped.loadDraft(0, root.data(), rootParent.data(), 1);
    EXPECT_EQ(walk(capped, /*maxPathLength=*/3), 1);
    EXPECT_EQ(capped.count(0), 3);
    EXPECT_EQ(ids(capped), (Ids{kA, kONE, kTWO, 0, 0, 0, 0, 0}));
}

// Given the grammar "a" "\":" "b" standing after the root a, and a draft tree whose root has the children ": and ",
//      the quote carrying a child :; then the grammar "a" "1" "b" after the same root and a draft tree whose root
//      has the child 1
// When each tree is walked
// Then the first root's chain ": b reuses the ": child and inserts b, and the quote, a legal sibling the acceptance
//      can never reach, is neither grafted nor counted, nor is the colon below it; the second root's chain 1 b
//      reuses the child and inserts b, and the child, now on the chain, is not planned again
TEST_F(JumpForwardCoordinatorTest, TheWalkSkipsUnreachableSiblingsAndNodesAlreadyOnAChain)
{
    installGrammar(R"(root ::= "a" "\":" "b")", kA);
    rt::HostDraftTree unreachable(1, 8);
    Ids const siblingIds{kA, kQUOTE_COLON, kQUOTE, kCOLON};
    Ids const siblingParents{-1, 0, 0, 2};
    unreachable.loadDraft(0, siblingIds.data(), siblingParents.data(), 4);

    EXPECT_EQ(walk(unreachable), 1);

    EXPECT_EQ(unreachable.count(0), 5);
    EXPECT_EQ(ids(unreachable), (Ids{kA, kQUOTE_COLON, kQUOTE, kCOLON, kB, 0, 0, 0}));
    EXPECT_EQ(parents(unreachable), (Ids{-1, 0, 0, 2, 1, -1, -1, -1}));
    EXPECT_EQ(forced(unreachable), (Flags{0, 1, 0, 0, 1, 0, 0, 0}));
    EXPECT_EQ(totalSkips(), 0);

    installGrammar(R"(root ::= "a" "1" "b")", kA);
    rt::HostDraftTree onChain(1, 6);
    Ids const childIds{kA, kONE};
    Ids const childParents{-1, 0};
    onChain.loadDraft(0, childIds.data(), childParents.data(), 2);

    EXPECT_EQ(walk(onChain), 1);

    EXPECT_EQ(onChain.count(0), 3);
    EXPECT_EQ(ids(onChain), (Ids{kA, kONE, kB, 0, 0, 0}));
    EXPECT_EQ(forced(onChain), (Flags{0, 1, 1, 0, 0, 0}));
    EXPECT_EQ(totalSkips(), 0);
}

// Given the grammar "a" ("1" | "2") "\":" "b" standing after the root a, and the draft chain a, 1, ", :, where the
//      grammar forces ": b after 1 and b after the colon
// When the chain is walked
// Then 1's plan, which the draft spells differently and is as long as the draft after 1, wins: the draft after 1 is
//      cut, ": and b are appended as forced nodes of chain 0, the chain holds depth 1 + 1 + 2 nodes and pads the
//      rest, no node below 1 is planned or counted, and the matcher is back after the root
TEST_F(JumpForwardCoordinatorTest, ChainModeGraftsAtTheFirstForcingNodeAndCutsTheDraftAfterIt)
{
    installGrammar(R"(root ::= "a" ("1" | "2") "\":" "b")", kA);
    rt::HostDraftTree tree(1, 8);
    Ids const draft{kA, kONE, kQUOTE, kCOLON};
    tree.loadChain(0, draft.data(), 4);

    EXPECT_EQ(walkChain(tree), 1);

    EXPECT_EQ(tree.count(0), 4);
    EXPECT_EQ(ids(tree), (Ids{kA, kONE, kQUOTE_COLON, kB, 0, 0, 0, 0}));
    EXPECT_EQ(parents(tree), (Ids{-1, 0, 1, 2, -1, -1, -1, -1}));
    EXPECT_EQ(depths(tree), (Ids{0, 1, 2, 3, -1, -1, -1, -1}));
    EXPECT_EQ(forced(tree), (Flags{0, 0, 1, 1, 0, 0, 0, 0}));
    EXPECT_EQ(chainOf(tree), (Flags{-1, -1, 0, 0, -1, -1, -1, -1}));
    EXPECT_EQ(totalSkips(), 0) << "the quote below 1 would have been refused as a respelling";
    EXPECT_EQ(mDecoder.forcedContinuation(0), "") << "after the root the grammar still chooses between 1 and 2";
}

// Given the grammar "a" ("1" | "2") "\":" "b" standing after the root a, and the draft chain a, 1, ", :, b, which
//      spells the forced ": b after 1 in three tokens
// When the chain is walked
// Then the two-token plan, shorter than the three draft nodes it would cut and the grammar accepts, is not grafted:
//      the chain is untouched, the plan is counted as draft_longer, the walk ends there, and the matcher is back after
//      the root
TEST_F(JumpForwardCoordinatorTest, ChainModeKeepsADraftLongerThanTheForcedRunAndCountsTheSkip)
{
    installGrammar(R"(root ::= "a" ("1" | "2") "\":" "b")", kA);
    rt::HostDraftTree tree(1, 8);
    Ids const draft{kA, kONE, kQUOTE, kCOLON, kB};
    tree.loadChain(0, draft.data(), 5);

    EXPECT_EQ(walkChain(tree), 0);

    EXPECT_EQ(tree.count(0), 5);
    EXPECT_EQ(ids(tree), (Ids{kA, kONE, kQUOTE, kCOLON, kB, 0, 0, 0}));
    EXPECT_EQ(forced(tree), (Flags{0, 0, 0, 0, 0, 0, 0, 0}));
    EXPECT_EQ(skips(rt::JumpSkipReason::kDraftLonger), 1);
    EXPECT_EQ(totalSkips(), 1) << "the quote below 1 would have been refused as a respelling";
    EXPECT_EQ(mDecoder.forcedContinuation(0), "") << "after the root the grammar still chooses between 1 and 2";
}

// Given the grammar "a" ("1" | "2") "\":" "b" standing after the root a, and the draft chain a, 1, ", 2, 2, 2 whose
//      first 2 the grammar refuses
// When the chain is walked
// Then the two-token plan ": b, shorter than the four draft nodes after 1 but not than the one of them the grammar
//      accepts, is grafted: the draft after 1 is cut and ": b appended as chain 0, with no skip counted
TEST_F(JumpForwardCoordinatorTest, ChainModeComparesTheForcedRunOnlyWithTheCutNodesTheGrammarAccepts)
{
    installGrammar(R"(root ::= "a" ("1" | "2") "\":" "b")", kA);
    rt::HostDraftTree tree(1, 8);
    Ids const draft{kA, kONE, kQUOTE, kTWO, kTWO, kTWO};
    tree.loadChain(0, draft.data(), 6);

    EXPECT_EQ(walkChain(tree), 1);

    EXPECT_EQ(ids(tree), (Ids{kA, kONE, kQUOTE_COLON, kB, 0, 0, 0, 0}));
    EXPECT_EQ(chainOf(tree), (Flags{-1, -1, 0, 0, -1, -1, -1, -1}));
    EXPECT_EQ(totalSkips(), 0);
}

// Given the grammar "a" ("1" | "2") ",\",\":b" standing after the root a, which forces the four tokens , ", ": b
//      after 1, and the draft chain a, 1, ,, ", ,, ", :, b whose first node after 1 is the reused ,
// When the chain is walked
// Then the grammar, standing after the reused ,, accepts all five nodes the plan would cut, so the four-token plan is
//      not grafted and is counted as draft_longer, and the matcher is back after the root
TEST_F(JumpForwardCoordinatorTest, ChainModeWeighsTheCutNodesFromTheGrammarStateAfterTheReusedOnes)
{
    installGrammar(R"(root ::= "a" ("1" | "2") ",\",\":b")", kA);
    rt::HostDraftTree tree(1, 12);
    Ids const draft{kA, kONE, kCOMMA, kQUOTE, kCOMMA, kQUOTE, kCOLON, kB};
    tree.loadChain(0, draft.data(), 8);

    EXPECT_EQ(walkChain(tree), 0);

    EXPECT_EQ(nodeIds(tree), draft);
    EXPECT_EQ(skips(rt::JumpSkipReason::kDraftLonger), 1);
    EXPECT_EQ(totalSkips(), 1);
    EXPECT_EQ(mDecoder.forcedContinuation(0), "") << "after the root the grammar still chooses between 1 and 2";
}

// Given the grammar "a" ("1" | "2") "\":" "b" standing after the root a, and the draft chain a, 1, ":, 2, b
// When the chain is walked
// Then the draft node ": spells the first forced token and is kept, flagged forced on chain 0; the draft is cut at the
//      2 that departs from the forced text, and only b is appended
TEST_F(JumpForwardCoordinatorTest, ChainModeReusesDraftNodesThatSpellTheForcedTokens)
{
    installGrammar(R"(root ::= "a" ("1" | "2") "\":" "b")", kA);
    rt::HostDraftTree tree(1, 8);
    Ids const draft{kA, kONE, kQUOTE_COLON, kTWO, kB};
    tree.loadChain(0, draft.data(), 5);

    EXPECT_EQ(walkChain(tree), 1);

    EXPECT_EQ(tree.count(0), 4);
    EXPECT_EQ(ids(tree), (Ids{kA, kONE, kQUOTE_COLON, kB, 0, 0, 0, 0}));
    EXPECT_EQ(forced(tree), (Flags{0, 0, 1, 1, 0, 0, 0, 0}));
    EXPECT_EQ(chainOf(tree), (Flags{-1, -1, 0, 0, -1, -1, -1, -1}));
    EXPECT_EQ(tree.forcedChildOf(0, 1), 2);
}

// Given the grammar "a" ("1" | "2") "\":" "b" ("1" | "2") ",b" standing after the root a, and the draft chain
//      a, 1, ":, b, 2, ",", b, which already spells both runs the grammar forces: ": b after 1 and , b after 2
// When the chain is walked
// Then both runs are flagged forced, as chains 0 and 1, nothing is cut or inserted, the walk reports two chains, and
//      the matcher is back after the root
TEST_F(JumpForwardCoordinatorTest, ChainModeFlagsRunsTheDraftAlreadySpellsAndKeepsTheWholeChain)
{
    installGrammar(R"(root ::= "a" ("1" | "2") "\":" "b" ("1" | "2") ",b")", kA);
    rt::HostDraftTree tree(1, 8);
    Ids const draft{kA, kONE, kQUOTE_COLON, kB, kTWO, kCOMMA, kB};
    tree.loadChain(0, draft.data(), 7);

    EXPECT_EQ(walkChain(tree), 2);

    EXPECT_EQ(tree.count(0), 7);
    EXPECT_EQ(ids(tree), (Ids{kA, kONE, kQUOTE_COLON, kB, kTWO, kCOMMA, kB, 0}));
    EXPECT_EQ(forced(tree), (Flags{0, 0, 1, 1, 0, 1, 1, 0}));
    EXPECT_EQ(chainOf(tree), (Flags{-1, -1, 0, 0, -1, 1, 1, -1}));
    EXPECT_EQ(totalSkips(), 0);
    EXPECT_EQ(mDecoder.forcedContinuation(0), "") << "after the root the grammar still chooses between 1 and 2";
}

// Given the grammar "a" ("1" | "2") "b,2:b," standing after the root a, which forces six tokens after a, 1
// When a draft chain a, 1, ... is walked at a width of six, at a path capacity of five, and with three tokens of
//      generation left
// Then the forced tokens after 1 are cut to the four rows the width leaves, to the three a path of five leaves
//      between 1 and the bonus token, and to the two the slot may still generate after 1
TEST_F(JumpForwardCoordinatorTest, ChainModeCutsTheForcedTokensToTheWidthThePathCapacityAndTheGenerationLeft)
{
    installGrammar(R"(root ::= "a" ("1" | "2") "b,2:b,")", kA);
    Ids const draft{kA, kONE, kTWO, kTWO, kTWO, kTWO};

    rt::HostDraftTree byWidth(1, 6);
    byWidth.loadChain(0, draft.data(), 6);
    EXPECT_EQ(walkChain(byWidth), 1);
    EXPECT_EQ(nodeIds(byWidth), (Ids{kA, kONE, kB, kCOMMA, kTWO, kCOLON}));

    rt::HostDraftTree byCapacity(1, 16);
    byCapacity.loadChain(0, draft.data(), 4);
    EXPECT_EQ(walkChain(byCapacity, /*maxPathLength=*/5), 1);
    EXPECT_EQ(nodeIds(byCapacity), (Ids{kA, kONE, kB, kCOMMA, kTWO}));

    rt::HostDraftTree byGeneration(1, 16);
    byGeneration.loadChain(0, draft.data(), 4);
    EXPECT_EQ(walkChain(byGeneration, /*maxPathLength=*/32, /*remainingGeneration=*/3), 1);
    EXPECT_EQ(nodeIds(byGeneration), (Ids{kA, kONE, kB, kCOMMA}));
}

// Given the grammar "a" ("1" | "2") ("1" | "2") ("1" | "2") standing after the root a, which never forces text
// When the draft chain a, 1, 2, 1 is walked
// Then nothing is grafted, the chain is untouched, and no skip is counted
TEST_F(JumpForwardCoordinatorTest, ChainModeLeavesAChainWithoutForcedTextUntouched)
{
    installGrammar(R"(root ::= "a" ("1" | "2") ("1" | "2") ("1" | "2"))", kA);
    rt::HostDraftTree tree(1, 6);
    Ids const draft{kA, kONE, kTWO, kONE};
    tree.loadChain(0, draft.data(), 4);

    EXPECT_EQ(walkChain(tree), 0);

    EXPECT_EQ(tree.count(0), 4);
    EXPECT_EQ(ids(tree), (Ids{kA, kONE, kTWO, kONE, 0, 0}));
    EXPECT_EQ(parents(tree), (Ids{-1, 0, 1, 2, -1, -1}));
    EXPECT_EQ(forced(tree), (Flags{0, 0, 0, 0, 0, 0}));
    EXPECT_EQ(totalSkips(), 0);
}

// Given the grammar "a" ("1" | "2") "b:," standing after the root a, and the draft chain a, b, 1, 2 whose b the
//      grammar refuses
// When the chain is walked
// Then the walk ends at b and nothing below it is planned, though from the root's state the grammar would accept the
//      1 and force b:, after it; the chain is untouched and no skip is counted
TEST_F(JumpForwardCoordinatorTest, ChainModeEndsTheWalkAtANodeTheGrammarRefuses)
{
    installGrammar(R"(root ::= "a" ("1" | "2") "b:,")", kA);
    rt::HostDraftTree tree(1, 8);
    Ids const draft{kA, kB, kONE, kTWO};
    tree.loadChain(0, draft.data(), 4);

    EXPECT_EQ(walkChain(tree), 0);

    EXPECT_EQ(tree.count(0), 4);
    EXPECT_EQ(ids(tree), (Ids{kA, kB, kONE, kTWO, 0, 0, 0, 0}));
    EXPECT_EQ(forced(tree), (Flags{0, 0, 0, 0, 0, 0, 0, 0}));
    EXPECT_EQ(totalSkips(), 0);
}

// Given the draft tree 100 -> {10 -> 20, 11} with the chain 10, 20, 30 grafted at the root and the chain 40 under 11
// When the tree is truncated to its four draft nodes
// Then the rows of 30 and 40 are padding again, the draft nodes keep their rows and forced flags, the cut parents 20
//      and 11 have no forced child, and a later chain under either carrying the padding id 0 is inserted as a new node
//      instead of matching a cut row
TEST_F(JumpForwardCoordinatorTest, TruncateRestoresPaddingAndUnlinksTheCutNodes)
{
    rt::HostDraftTree tree(1, 8);
    tree.loadDraft(0, kDRAFT_IDS.data(), kDRAFT_PARENTS.data(), 4);
    Ids const rootChain{10, 20, 30};
    Ids const sideChain{40};
    tree.insertChain(0, 0, rootChain.data(), 3, /*chainId=*/0);
    tree.insertChain(0, 2, sideChain.data(), 1, /*chainId=*/1);
    ASSERT_EQ(tree.count(0), 6);

    tree.truncate(0, 4);

    EXPECT_EQ(tree.count(0), 4);
    EXPECT_EQ(ids(tree), (Ids{100, 10, 11, 20, 0, 0, 0, 0}));
    EXPECT_EQ(parents(tree), (Ids{-1, 0, 0, 1, -1, -1, -1, -1}));
    EXPECT_EQ(depths(tree), (Ids{0, 1, 1, 2, -1, -1, -1, -1}));
    EXPECT_EQ(forced(tree), (Flags{0, 1, 0, 1, 0, 0, 0, 0}));
    EXPECT_EQ(chainOf(tree), (Flags{-1, 0, -1, 0, -1, -1, -1, -1}));
    EXPECT_EQ(maskRow(tree, 4), Flags(8, 0));
    EXPECT_EQ(maskRow(tree, 5), Flags(8, 0));
    EXPECT_EQ(tree.forcedChildOf(0, 1), 3);
    EXPECT_EQ(tree.forcedChildOf(0, 3), -1);
    EXPECT_EQ(tree.forcedChildOf(0, 2), -1);

    Ids const paddingId{0};
    EXPECT_EQ(tree.insertChain(0, 3, paddingId.data(), 1, /*chainId=*/2), 1);
    EXPECT_EQ(tree.insertChain(0, 2, paddingId.data(), 1, /*chainId=*/3), 1);
    EXPECT_EQ(parents(tree), (Ids{-1, 0, 0, 1, 3, 2, -1, -1}));
}

} // namespace
