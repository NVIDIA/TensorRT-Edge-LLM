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

// The grafting layer a speculative decoder holds for jump-forward, on the byte-level BPE vocabulary of the
// guided-decoding tests: what a grafting round uploads for verification, and what it accounts after acceptance.

#include "guidedTokenizerFixture.h"
#include "runtime/decoding/decoderUtils.h"
#include "runtime/decoding/guidedDecoder.h"
#include "runtime/decoding/jumpForward.h"
#include "runtime/decoding/specDecodeUtils.h"
#include "runtime/state/decodingInferenceContext.h"
#include "scratchDir.h"
#include "testUtils.h"
#include "tokenizer/tokenizer.h"

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <filesystem>
#include <limits>
#include <string>
#include <utility>
#include <vector>

using namespace trt_edgellm;
using namespace guidedFixture;

namespace
{

using Ids = std::vector<int32_t>;
using Flags = std::vector<int8_t>;
using LogprobRows = std::vector<std::pair<int32_t, float>>;

constexpr int32_t kBatch{2};
constexpr int32_t kWidth{4};
constexpr int32_t kJumpWidth{8};
constexpr int32_t kPathCapacity{6};
constexpr int32_t kMaxJumpTokens{4};
//! Fills the decoder's verify views before a round, so rows the upload does not write stand out.
constexpr int32_t kStale{77};

// Slot 0 decodes without jump-forward: root b with children 1 and 2, the 1 carrying a child ,.
Ids const kSLOT0_IDS{kB, kONE, kTWO, kCOMMA};
Ids const kSLOT0_PARENTS{-1, 0, 0, 1};

//! Two slots over the BPE fixture with an identity output space. Slot 1 asked for jump-forward and stands after its
//! root a under the grammar a test installs; slot 0 has no grammar and no jump-forward.
class SpecJumpGrafterTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        CUDA_CHECK(cudaStreamCreate(&mStream));
        mDir = makeScratchDir("specJumpGrafter");
        writeBpeTokenizer(mDir);
        ASSERT_TRUE(mTokenizer.loadFromHF(mDir));
        rt::Tensor emptyMap;
        mDecoder.initialize(kBatch, kJumpWidth, kBPE_VOCAB_SIZE, kBPE_VOCAB_SIZE, &mTokenizer, emptyMap, mStream);
        mContext.initialize(kBatch, /*maxGenLength=*/64, std::nullopt, rt::OptionalInputTensors{}, "", mStream);
        mContext.tokenIds = {{kOPEN_BRACE, kB}, {kOPEN_BRACE, kA}};
        mContext.currentGenerateLengths = {1, 1};
        mContext.committedLengths = {1, 1};
        mContext.guidedReasoningEnded = {1, 1};
        mContext.jumpForwardEnabled = {0, 1};

        mIds = rt::Tensor({kBatch, kJumpWidth}, rt::DeviceType::kGPU, DataType::kINT32);
        mParents = rt::Tensor({kBatch, kJumpWidth}, rt::DeviceType::kGPU, DataType::kINT32);
        mDepths = rt::Tensor({kBatch, kJumpWidth}, rt::DeviceType::kGPU, DataType::kINT32);
        for (rt::Tensor* view : {&mIds, &mParents, &mDepths})
        {
            copyHostToDevice<int32_t>(*view, Ids(kBatch * kJumpWidth, kStale));
            ASSERT_TRUE(view->reshape({kBatch, kWidth}));
        }
    }

    void TearDown() override
    {
        std::filesystem::remove_all(mDir);
        CUDA_CHECK(cudaStreamDestroy(mStream));
    }

    void installGrammar(std::string const& ebnf)
    {
        rt::GuidedDecodingParams params;
        params.type = rt::GuideType::kEbnf;
        params.guide = ebnf;
        std::string failReason;
        ASSERT_TRUE(mDecoder.prepareSlot(1, params, failReason)) << failReason;
        ASSERT_TRUE(mDecoder.advance(1, kA));
    }

    rt::SpecJumpGrafter makeGrafter(rt::JumpGraftGeometry geometry = rt::JumpGraftGeometry::kTree)
    {
        return rt::SpecJumpGrafter(
            mTokenizer, mDecoder, kBatch, geometry, {kWidth, kJumpWidth, kPathCapacity, kMaxJumpTokens});
    }

    //! Walk slot 0's draft beside slot 1's, both laid into the verify views at the draft width.
    rt::SpecJumpGrafter::Walk walk(rt::SpecJumpGrafter& grafter, Ids const& slot1Ids, Ids const& slot1Parents)
    {
        Ids ids = kSLOT0_IDS;
        ids.insert(ids.end(), slot1Ids.begin(), slot1Ids.end());
        Ids parents = kSLOT0_PARENTS;
        parents.insert(parents.end(), slot1Parents.begin(), slot1Parents.end());
        copyHostToDevice<int32_t>(mIds, ids);
        copyHostToDevice<int32_t>(mParents, parents);
        rt::SpecVerifyInputs const drafted{
            mIds, nullptr, &mParents, nullptr, kBatch, kWidth, kPathCapacity, rt::SpecVerifyLayout::kTree};
        return grafter.walk(mContext, drafted);
    }

    //! Walk slot 0's draft beside slot 1's as chains, both laid into the ids view at the draft width.
    rt::SpecJumpGrafter::Walk walkChains(rt::SpecJumpGrafter& grafter, Ids const& slot1Ids)
    {
        Ids ids = kSLOT0_IDS;
        ids.insert(ids.end(), slot1Ids.begin(), slot1Ids.end());
        copyHostToDevice<int32_t>(mIds, ids);
        rt::SpecVerifyInputs const drafted{
            mIds, nullptr, nullptr, nullptr, kBatch, kWidth, kPathCapacity, rt::SpecVerifyLayout::kChain};
        return grafter.walk(mContext, drafted);
    }

    static Flags hostRow(int8_t const* row)
    {
        return Flags(row, row + kJumpWidth);
    }

    //! A flat [.., kJumpWidth] mask as its rows.
    static std::vector<Flags> rows(Flags const& flat)
    {
        std::vector<Flags> split;
        for (size_t begin = 0; begin < flat.size(); begin += kJumpWidth)
        {
            split.emplace_back(flat.begin() + static_cast<std::ptrdiff_t>(begin),
                flat.begin() + static_cast<std::ptrdiff_t>(begin + kJumpWidth));
        }
        return split;
    }

    std::filesystem::path mDir;
    tokenizer::Tokenizer mTokenizer;
    rt::GuidedDecoder mDecoder;
    rt::DecodingInferenceContext mContext;
    cudaStream_t mStream{};
    rt::Tensor mIds;
    rt::Tensor mParents;
    rt::Tensor mDepths;
};

// Given slot 0 not jumping beside slot 1 under the grammar "a" "\":" "b", whose draft root a has the children ": and ",
//      the quote carrying a child :, at a draft width of four and a jump width of eight
// When the walk grafts slot 1's chain ": b and the round is uploaded
// Then the decoder's ids, parents and depths and the grafter's mask, forced flags and node counts hold both slots at
//      the jump width: slot 0 its draft, padded past four nodes, with a count of four, and slot 1 its draft with b
//      inserted under the reused ": and both flagged forced, with a count of five
TEST_F(SpecJumpGrafterTest, UploadWritesEverySlotAtTheJumpWidthAndPadsASlotWithoutAChain)
{
    installGrammar(R"(root ::= "a" "\":" "b")");
    rt::SpecJumpGrafter grafter = makeGrafter();
    ASSERT_EQ(walk(grafter, {kA, kQUOTE_COLON, kQUOTE, kCOLON}, {-1, 0, 0, 2}), rt::SpecJumpGrafter::Walk::kGrafted);

    grafter.upload(mIds, &mParents, &mDepths, mStream);
    CUDA_CHECK(cudaStreamSynchronize(mStream));

    ASSERT_NE(grafter.forcedFlags(), nullptr);
    ASSERT_NE(grafter.validCounts(), nullptr);
    EXPECT_EQ(copyDeviceToHost<int32_t>(mIds),
        (Ids{kB, kONE, kTWO, kCOMMA, 0, 0, 0, 0, kA, kQUOTE_COLON, kQUOTE, kCOLON, kB, 0, 0, 0}));
    EXPECT_EQ(copyDeviceToHost<int32_t>(mParents), (Ids{-1, 0, 0, 1, -1, -1, -1, -1, -1, 0, 0, 2, 1, -1, -1, -1}));
    EXPECT_EQ(copyDeviceToHost<int32_t>(mDepths), (Ids{0, 1, 1, 2, -1, -1, -1, -1, 0, 1, 1, 2, 2, -1, -1, -1}));
    Flags const pad(kJumpWidth, 0);
    EXPECT_EQ(rows(copyDeviceToHost<int8_t>(grafter.mask())),
        (std::vector<Flags>{{1, 0, 0, 0, 0, 0, 0, 0}, {1, 1, 0, 0, 0, 0, 0, 0}, {1, 0, 1, 0, 0, 0, 0, 0},
            {1, 1, 0, 1, 0, 0, 0, 0}, pad, pad, pad, pad, {1, 0, 0, 0, 0, 0, 0, 0}, {1, 1, 0, 0, 0, 0, 0, 0},
            {1, 0, 1, 0, 0, 0, 0, 0}, {1, 0, 1, 1, 0, 0, 0, 0}, {1, 1, 0, 0, 1, 0, 0, 0}, pad, pad, pad}));
    EXPECT_EQ(
        copyDeviceToHost<int8_t>(*grafter.forcedFlags()), (Flags{0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0}));
    EXPECT_EQ(copyDeviceToHost<int32_t>(*grafter.validCounts()), (Ids{4, 5}));
}

// Given slot 0 not jumping beside slot 1 under the grammar "a" ("1" | "2") "b", whose draft chain is a 1 2 2, on a
//      chain grafter at a draft width of four and a jump width of eight
// When the walk grafts b behind 1 and the round is uploaded without tree views
// Then the decoder's ids and the grafter's forced flags and node counts hold both chains at the jump width: slot 0 its
//      draft with a count of four, and slot 1 a 1 b, with b flagged forced, the draft after 1 cut, and a count of three
TEST_F(SpecJumpGrafterTest, AChainUploadCarriesTheCutChainAndItsNodeCount)
{
    installGrammar(R"(root ::= "a" ("1" | "2") "b")");
    rt::SpecJumpGrafter grafter = makeGrafter(rt::JumpGraftGeometry::kChain);
    ASSERT_EQ(walkChains(grafter, {kA, kONE, kTWO, kTWO}), rt::SpecJumpGrafter::Walk::kGrafted);

    grafter.upload(mIds, nullptr, nullptr, mStream);
    CUDA_CHECK(cudaStreamSynchronize(mStream));

    ASSERT_NE(grafter.forcedFlags(), nullptr);
    ASSERT_NE(grafter.validCounts(), nullptr);
    EXPECT_EQ(copyDeviceToHost<int32_t>(mIds), (Ids{kB, kONE, kTWO, kCOMMA, 0, 0, 0, 0, kA, kONE, kB, 0, 0, 0, 0, 0}));
    EXPECT_EQ(
        copyDeviceToHost<int8_t>(*grafter.forcedFlags()), (Flags{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0}));
    EXPECT_EQ(copyDeviceToHost<int32_t>(*grafter.validCounts()), (Ids{4, 3}));
}

// Given slot 1 under the grammar "a" "1" ("1" | "2") "," with the draft chain a 1 2 2, walked into a root chain 1 on
//      the draft's 1 and a chain , inserted under its 2; an accepted path 1 2 , <eos> through those nodes; and two
//      log-probabilities per token, collected from the verify logits
// When the forced path goes through appendAcceptedTokens and the round is accounted
// Then slot 1 counts one grafting round, two forced tokens and two chains, the rows of the forced 1 and , become
//      (token, 0) with fillers while the rows of the predicted 2 and <eos> keep their log-probabilities, and slot 0,
//      without jump-forward, does not count the round
TEST_F(SpecJumpGrafterTest, AccountCountsTheForcedTokensAndChainsOnTheAcceptedPathAndRewritesTheirLogprobRows)
{
    installGrammar(R"(root ::= "a" "1" ("1" | "2") ",")");
    rt::SpecJumpGrafter grafter = makeGrafter();
    ASSERT_EQ(walk(grafter, {kA, kONE, kTWO, kTWO}, {-1, 0, 1, 2}), rt::SpecJumpGrafter::Walk::kGrafted);
    ASSERT_EQ(hostRow(grafter.hostTree().forced(1)), (Flags{0, 1, 0, 0, 1, 0, 0, 0}));
    ASSERT_EQ(hostRow(grafter.hostTree().chainOf(1)), (Flags{-1, 0, -1, -1, 1, -1, -1, -1}));

    // As the accept step leaves them: slot 1's path runs root, 1, 2, then the forced , whose logits give the bonus.
    rt::Tensor acceptLengths({kBatch}, rt::DeviceType::kGPU, DataType::kINT32);
    rt::Tensor acceptedIds({kBatch, kPathCapacity}, rt::DeviceType::kGPU, DataType::kINT32);
    rt::Tensor acceptedRows({kBatch, kPathCapacity}, rt::DeviceType::kGPU, DataType::kINT32);
    copyHostToDevice<int32_t>(acceptLengths, {1, 4});
    copyHostToDevice<int32_t>(acceptedIds, {kONE, -1, -1, -1, -1, -1, kONE, kTWO, kCOMMA, kEOS, -1, -1});
    copyHostToDevice<int32_t>(acceptedRows, {0, -1, -1, -1, -1, -1, 0, 1, 2, 4, -1, -1});
    rt::Tensor hostAcceptLengths({kBatch}, rt::DeviceType::kCPU, DataType::kINT32);
    rt::Tensor hostAcceptedIds({kBatch, kPathCapacity}, rt::DeviceType::kCPU, DataType::kINT32);

    constexpr int32_t kTopK{2};
    float const kNegInf = -std::numeric_limits<float>::infinity();
    LogprobRows const verifyRows{{100, -0.5F}, {101, -1.5F}, {102, -0.25F}, {103, -2.0F}, {104, -0.75F}, {105, -1.0F},
        {106, -0.125F}, {107, -3.0F}};
    mContext.numLogprobsPerSlot.assign(static_cast<size_t>(kBatch), kTopK);
    for (rt::LogprobsSlot& slot : mContext.stepLogprobs)
    {
        slot.data = verifyRows;
        slot.data.resize(kPathCapacity * kTopK, {-1, kNegInf});
        slot.numSteps = 4;
    }

    rt::decoder_utils::ForcedPathInfo const forcedPath = grafter.forcedPath(acceptedRows);
    rt::decoder_utils::appendAcceptedTokens(mContext, hostAcceptLengths, hostAcceptedIds, acceptLengths, acceptedIds,
        kPathCapacity, mTokenizer, mStream, kWidth - 1, nullptr, -1, true, &forcedPath);
    ASSERT_EQ(hostAcceptLengths.dataPointer<int32_t>()[1], 4);
    grafter.account(
        mContext, hostAcceptLengths.dataPointer<int32_t>(), hostAcceptedIds.dataPointer<int32_t>(), kPathCapacity);

    rt::JumpForwardSlotStats const& jumped = mContext.jumpForwardStats[1];
    EXPECT_EQ(jumped.chunkSteps, 1);
    EXPECT_EQ(jumped.forcedTokens, 2);
    EXPECT_EQ(jumped.jumps, 2);
    LogprobRows const written(mContext.stepLogprobs[1].data.begin(), mContext.stepLogprobs[1].data.begin() + 4 * kTopK);
    EXPECT_EQ(written,
        (LogprobRows{{kONE, 0.0F}, {-1, kNegInf}, {102, -0.25F}, {103, -2.0F}, {kCOMMA, 0.0F}, {-1, kNegInf},
            {106, -0.125F}, {107, -3.0F}}));
    EXPECT_EQ(mContext.jumpForwardStats[0].chunkSteps, 0);
}

} // namespace
