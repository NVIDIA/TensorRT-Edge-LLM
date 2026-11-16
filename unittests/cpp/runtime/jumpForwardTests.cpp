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

#include "common/checkMacros.h"
#include "guidedTokenizerFixture.h"
#include "runtime/decoding/guidedDecoder.h"
#include "runtime/decoding/jumpForward.h"
#include "runtime/llmRuntimeUtils.h"
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

//! The byte-level BPE fixture with an identity output space and no grammar unless a test installs one.
class JumpForwardPlannerTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        CUDA_CHECK(cudaStreamCreate(&mStream));
        mDir = makeScratchDir("jumpForward");
        writeBpeTokenizer(mDir);
        ASSERT_TRUE(mTokenizer.loadFromHF(mDir));
        rt::Tensor emptyMap;
        mDecoder.initialize(
            /*maxBatchSize=*/1, /*maxRowsPerSlot=*/1, kBPE_VOCAB_SIZE, kBPE_VOCAB_SIZE, &mTokenizer, emptyMap, mStream);
    }

    void TearDown() override
    {
        std::filesystem::remove_all(mDir);
        CUDA_CHECK(cudaStreamDestroy(mStream));
    }

    rt::JumpForwardPlan plan(std::string const& forced, Ids const& window, int32_t mutableTailCount = 1,
        int32_t budgetTokens = 32, int32_t maxJumpTokens = 32) const
    {
        rt::JumpForwardPlanner const planner(mTokenizer, mDecoder, maxJumpTokens);
        return planner.plan(/*slot=*/0, forced, window, mutableTailCount, budgetTokens);
    }

    std::filesystem::path mDir;
    tokenizer::Tokenizer mTokenizer;
    rt::GuidedDecoder mDecoder;
    cudaStream_t mStream{};
};

//! Given a forced string that ends inside a code point, when it is planned, then only the complete
//! code points are appended, and a forced string with no complete code point is skipped.
TEST_F(JumpForwardPlannerTest, ForcedTextIsCutAtTheLastCompleteCodePoint)
{
    auto const cjk = plan("\xE4\xB8\xAD\xE6", {kCOLON}); // U+4E2D then the first byte of U+6587
    EXPECT_EQ(cjk.skipReason, rt::JumpSkipReason::kNone);
    EXPECT_EQ(cjk.rollbackCount, 0);
    EXPECT_EQ(cjk.fullVocabIds, Ids{kCJK});

    auto const ascii = plan(",\xE4", {kCOLON});
    EXPECT_EQ(ascii.fullVocabIds, Ids{kCOMMA});

    auto const partial = plan("\xE6", {kCOLON});
    EXPECT_TRUE(partial.empty());
    EXPECT_EQ(partial.skipReason, rt::JumpSkipReason::kIncompleteUtf8);
}

//! Given forced text that spells a stop token, when planned, then the tokens stop before it: a forced string
//! that is nothing but the stop token is skipped as unusable, and text before it is kept.
TEST_F(JumpForwardPlannerTest, AStopTokenInTheForcedTextIsAsUnusableAsAPrunedOne)
{
    auto const onlyStop = plan("<eos>", {kCOLON});
    EXPECT_TRUE(onlyStop.empty());
    EXPECT_EQ(onlyStop.skipReason, rt::JumpSkipReason::kPrunedToken);

    auto const beforeStop = plan(",<eos>", {kCOLON});
    EXPECT_EQ(beforeStop.skipReason, rt::JumpSkipReason::kNone);
    EXPECT_EQ(beforeStop.fullVocabIds, Ids{kCOMMA});
}

//! Given a mutable window token the forced text merges with, when planned, then the token is rolled
//! back and the joined text is respelled the way the tokenizer splits it.
TEST_F(JumpForwardPlannerTest, MergeWithTheWindowRollsBackAndRespells)
{
    auto const merged = plan(", \"b\":", {kQUOTE});
    EXPECT_EQ(merged.skipReason, rt::JumpSkipReason::kNone);
    EXPECT_EQ(merged.rollbackCount, 1);
    EXPECT_EQ(merged.fullVocabIds, (Ids{kQUOTE_COMMA, kSPACE_QUOTE, kB, kQUOTE_COLON}));
    EXPECT_EQ(merged.outputSpaceIds, merged.fullVocabIds) << "identity output space";

    auto const unchanged = plan(",", {kCOLON});
    EXPECT_EQ(unchanged.rollbackCount, 0) << "':' and ',' do not merge";
    EXPECT_EQ(unchanged.fullVocabIds, Ids{kCOMMA});
}

//! Given a window whose read-only part would change under re-encoding, when planned, then the jump
//! is skipped as a boundary conflict rather than rewriting an emitted token.
TEST_F(JumpForwardPlannerTest, MergeIntoAReadOnlyTokenSkips)
{
    auto const emitted = plan(",", {kA, kQUOTE}, /*mutableTailCount=*/0);
    EXPECT_TRUE(emitted.empty());
    EXPECT_EQ(emitted.skipReason, rt::JumpSkipReason::kBoundaryConflict);

    // `"` + `:` re-encode as `":`, so the merge reaches past the mutable colon.
    auto const pastTheTail = plan(" \"b\"", {kQUOTE, kCOLON}, /*mutableTailCount=*/1);
    EXPECT_TRUE(pastTheTail.empty());
    EXPECT_EQ(pastTheTail.skipReason, rt::JumpSkipReason::kBoundaryConflict);

    auto const bothMutable = plan(" \"b\":", {kQUOTE, kCOLON}, /*mutableTailCount=*/2);
    EXPECT_EQ(bothMutable.rollbackCount, 2);
    EXPECT_EQ(bothMutable.fullVocabIds, (Ids{kQUOTE_COLON, kSPACE_QUOTE, kB, kQUOTE_COLON}));
}

//! Given an engine that prunes a token the forced text needs, when planned, then the jump stops in
//! front of that token, and is skipped when it would have been the first one.
TEST_F(JumpForwardPlannerTest, PrunedTokenEndsTheJump)
{
    Ids map;
    for (int32_t id = 0; id < kBPE_VOCAB_SIZE; ++id)
    {
        if (id != kB)
        {
            map.push_back(id);
        }
    }
    rt::Tensor deviceMap({static_cast<int64_t>(map.size())}, rt::DeviceType::kGPU, trt_edgellm::DataType::kINT32);
    copyHostToDevice<int32_t>(deviceMap, map);
    rt::GuidedDecoder pruned;
    pruned.initialize(/*maxBatchSize=*/1, /*maxRowsPerSlot=*/1, static_cast<int32_t>(map.size()), kBPE_VOCAB_SIZE,
        &mTokenizer, deviceMap, mStream);
    rt::JumpForwardPlanner const planner(mTokenizer, pruned, /*maxJumpTokens=*/32);

    auto const cut = planner.plan(0, ", \"b\":", {kQUOTE}, /*mutableTailCount=*/1, /*budgetTokens=*/32);
    EXPECT_EQ(cut.rollbackCount, 1);
    EXPECT_EQ(cut.fullVocabIds, (Ids{kQUOTE_COMMA, kSPACE_QUOTE}));
    EXPECT_EQ(cut.outputSpaceIds, (Ids{kQUOTE_COMMA - 1, kSPACE_QUOTE - 1})) << "ids above 'b' shift down by one";

    auto const first = planner.plan(0, "b", {kCOLON}, /*mutableTailCount=*/1, /*budgetTokens=*/32);
    EXPECT_TRUE(first.empty());
    EXPECT_EQ(first.skipReason, rt::JumpSkipReason::kPrunedToken);
}

//! Given a token budget, when the forced text needs more, then the plan keeps that many new tokens
//! on top of the ones it replaces, the planner's token cap bounds it the same way, and a budget of zero
//! skips.
TEST_F(JumpForwardPlannerTest, BudgetCountsNewTokensNotReplacedOnes)
{
    auto const two = plan(", \"b\":", {kQUOTE}, /*mutableTailCount=*/1, /*budgetTokens=*/2);
    EXPECT_EQ(two.rollbackCount, 1);
    EXPECT_EQ(two.fullVocabIds, (Ids{kQUOTE_COMMA, kSPACE_QUOTE, kB}));

    auto const capped = plan(", \"b\":", {kQUOTE}, /*mutableTailCount=*/1, /*budgetTokens=*/32, /*maxJumpTokens=*/1);
    EXPECT_EQ(capped.fullVocabIds, (Ids{kQUOTE_COMMA, kSPACE_QUOTE}));

    auto const none = plan(", \"b\":", {kQUOTE}, /*mutableTailCount=*/1, /*budgetTokens=*/0);
    EXPECT_TRUE(none.empty());
    EXPECT_EQ(none.skipReason, rt::JumpSkipReason::kNoBudget);
}

//! Given a jump whose last token another vocabulary token extends, when it is planned without a grammar the jump
//! completes, then that token is handed back whether or not it is whitespace; a token nothing extends stays.
TEST_F(JumpForwardPlannerTest, AnExtendableLastTokenIsHandedBack)
{
    EXPECT_EQ(plan(", ", {kQUOTE}).fullVocabIds, Ids{kQUOTE_COMMA}) << "' ' starts ' \"'";
    EXPECT_EQ(plan(",\"", {kCOLON}).fullVocabIds, Ids{kCOMMA}) << "'\"' starts '\",' and '\":'";
    EXPECT_EQ(plan(", \"b\":", {kQUOTE}).fullVocabIds, (Ids{kQUOTE_COMMA, kSPACE_QUOTE, kB, kQUOTE_COLON}))
        << "nothing extends '\":'";
}

//! Given a grammar that is complete at the end of the forced text, when the jump ends in an
//! extendable token, then the token stays, and the grammar state is where it was afterwards.
TEST_F(JumpForwardPlannerTest, AnExtendableLastTokenStaysWhenTheJumpCompletesTheGrammar)
{
    rt::GuidedDecodingParams params;
    params.type = rt::GuideType::kEbnf;
    params.guide = R"(root ::= "\", ")";
    std::string failReason;
    ASSERT_TRUE(mDecoder.prepareSlot(0, params, failReason)) << failReason;
    ASSERT_TRUE(mDecoder.advance(0, kQUOTE));
    std::string const forced = mDecoder.forcedContinuation(0);
    ASSERT_EQ(forced, ", ");

    auto const completing = plan(forced, {kQUOTE});
    EXPECT_EQ(completing.rollbackCount, 1);
    EXPECT_EQ(completing.fullVocabIds, (Ids{kQUOTE_COMMA, kSPACE}));
    EXPECT_EQ(mDecoder.forcedContinuation(0), ", ") << "planning must not move the grammar";
}

//! Given text the vocabulary cannot spell, or no text at all, when planned, then the jump is skipped
//! instead of appending whatever the tokenizer returned.
TEST_F(JumpForwardPlannerTest, UnspellableOrEmptyTextIsSkipped)
{
    auto const unspellable = plan("z", {kCOLON});
    EXPECT_TRUE(unspellable.empty());
    EXPECT_EQ(unspellable.skipReason, rt::JumpSkipReason::kRetokenizeFailed);
    EXPECT_EQ(plan("", {kCOLON}).skipReason, rt::JumpSkipReason::kEmptyForcedString);
}

//! Given a tokenizer that prepends a word marker, when the window does not start at a word
//! boundary, then the jump is skipped, because re-encoding would put a space in front of the
//! window; a window at a word boundary plans normally, and a byte-level tokenizer never needs this.
TEST_F(JumpForwardPlannerTest, MetaspaceWindowMustStartAtAWordBoundary)
{
    EXPECT_FALSE(mTokenizer.prependsSpaceMarker());

    auto const dir = makeScratchDir("jumpForwardMetaspace");
    writeMetaspaceTokenizer(dir);
    tokenizer::Tokenizer metaspace;
    ASSERT_TRUE(metaspace.loadFromHF(dir));
    EXPECT_TRUE(metaspace.prependsSpaceMarker());
    rt::GuidedDecoder decoder;
    rt::Tensor emptyMap;
    decoder.initialize(/*maxBatchSize=*/1, /*maxRowsPerSlot=*/1, kMETASPACE_VOCAB_SIZE, kMETASPACE_VOCAB_SIZE,
        &metaspace, emptyMap, mStream);
    rt::JumpForwardPlanner const planner(metaspace, decoder, /*maxJumpTokens=*/32);

    auto const midWord = planner.plan(0, ",", {kMS_A}, /*mutableTailCount=*/1, /*budgetTokens=*/32);
    EXPECT_TRUE(midWord.empty());
    EXPECT_EQ(midWord.skipReason, rt::JumpSkipReason::kNotAtWordBoundary);

    auto const atBoundary = planner.plan(0, ",", {kMS_MARKER_A}, /*mutableTailCount=*/1, /*budgetTokens=*/32);
    EXPECT_EQ(atBoundary.rollbackCount, 0);
    EXPECT_EQ(atBoundary.fullVocabIds, Ids{kMS_COMMA});
    std::filesystem::remove_all(dir);
}

} // namespace
