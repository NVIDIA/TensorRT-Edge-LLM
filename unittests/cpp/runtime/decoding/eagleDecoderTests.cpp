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

// EAGLE speculative decoding assembled around two substitute engines.

#include "substituteEngine.h"

using namespace trt_edgellm;
using namespace substitute_engine;

namespace
{
// --------------------------------------------------------------------------
// EAGLE speculative decoding.
//
// EAGLE and MTP are separate decoders implementing one DecodingStrategy
// contract, so the expectations below are deliberately the ones already stated
// for MTP. Where the two disagree, one of them is wrong; that is the point of
// stating them twice rather than testing each decoder against itself.
// --------------------------------------------------------------------------

//! EAGLE-3's draft reads a concatenation of three base layers' hidden states, so its `hidden_states_input` binding
//! is three times the base hidden size. `parseDraftEngineConfig` takes this from the config rather than deriving
//! it, because the same field is 1x for MTP.
constexpr int32_t kEagleHiddenStateLayers{3};

Json makeEagleBaseConfig()
{
    Json config = makeSpecBaseConfig("eagle3", kVerifySize);
    // EAGLE-3 conditions its draft on several base layers at once, and the deployment contract requires the base to
    // name them: unique ids inside its own decoder-layer range, as many as the draft's base_model_hidden_size
    // divided by the hidden size. The layer count therefore has to be at least kEagleHiddenStateLayers.
    config["num_hidden_layers"] = kEagleHiddenStateLayers;
    Json layerIds = Json::array();
    for (int32_t layer = 0; layer < kEagleHiddenStateLayers; ++layer)
    {
        layerIds.push_back(layer);
    }
    config["eagle_hidden_state_layers"] = std::move(layerIds);
    return config;
}

Json makeEagleDraftConfig()
{
    Json config = makeSpecDraftConfig("eagle3", kVerifySize);
    config["base_model_hidden_size"] = kEagleHiddenStateLayers * config["hidden_size"].get<int32_t>();
    return config;
}

class EagleAssemblyTest : public SpecAssemblyTest
{
protected:
    //! No d2t.safetensors is written. EAGLE loads one only when the file is present, and leaves the mapping table
    //! zeroed otherwise, so a full-vocab draft needs no sidecar at all.
    std::filesystem::path stageModelDir() override
    {
        return writeSpecModelDir("eagleAssemblyTests", makeEagleBaseConfig(), makeEagleDraftConfig());
    }

    rt::SpecDecodeDraftingConfig drafting() const override
    {
        return makeMtpDrafting();
    }
};

TEST_F(EagleAssemblyTest, AssemblesTwoEnginesWithoutAnyEngineFileOnDisk)
{
    expectAssembledFromArtifactsAlone("eagle");
    // Nor the sidecar EAGLE reads when a reduced-vocab draft has one.
    EXPECT_FALSE(std::filesystem::exists(mModelDir / "d2t.safetensors"));
}

TEST_F(EagleAssemblyTest, RunsTheDraftChainThenOneBaseVerificationPerRound)
{
    // Deliberately the expectation MtpAssemblyTest states, run against a different decoder. EAGLE's draft consumes
    // the target's hidden state exactly as MTP's does, so the round has the same shape; where the two disagree, one
    // of them is wrong.
    expectChainProposalThenOneVerification();
}

// Given a base whose argmax agrees with the draft at every proposed position
// When one round runs
// Then a single verification commits the whole chain
TEST_F(EagleAssemblyTest, AcceptsTheWholeChainInOneVerificationWhenTheBaseAgrees)
{
    using ::testing::_;

    auto baseEngine = makeEngine();
    auto draftEngine = makeEngine();
    auto& base = *baseEngine;

    // Both engines leave the logits zeroed, so the draft proposes kZeroLogitsToken at every position and the base's
    // argmax agrees at every verified position. Full agreement is the case the technique is built for: one
    // verification commits `draftingStep + 1` tokens, which is what DeploymentConfig::maxAcceptedTokensPerRound
    // reports for EAGLE.
    EXPECT_CALL(base, prepare(kPrefillProfile, _, _, _)).Times(1);
    EXPECT_CALL(base, prepare(kDecodeProfile, _, _, _)).Times(1);

    auto artifacts = makeArtifacts(std::move(baseEngine), std::move(draftEngine));
    auto const maxAcceptedPerRound = artifacts.deployment.maxAcceptedTokensPerRound();
    ASSERT_EQ(maxAcceptedPerRound, kDraftingStep + 1);

    auto runtime = makeRuntime(std::move(artifacts));

    auto const request = makeGreedyRequest("a", maxAcceptedPerRound);
    rt::LLMGenerationResponse response;
    ASSERT_TRUE(runtime.handleRequest(request, response, mStream));

    ASSERT_EQ(response.outputIds.size(), 1U);
    EXPECT_EQ(response.outputIds[0].size(), static_cast<size_t>(maxAcceptedPerRound));
}

// Given a base that agrees with the whole chain but a request whose budget is one token short of a full round
// When one round runs
// Then the commit stops at the budget instead of the accepted chain
TEST_F(EagleAssemblyTest, ClampsAFullyAcceptedRoundToTheRemainingGenerationBudget)
{
    using ::testing::_;

    auto baseEngine = makeEngine();
    auto draftEngine = makeEngine();
    auto& base = *baseEngine;

    EXPECT_CALL(base, prepare(kPrefillProfile, _, _, _)).Times(1);
    EXPECT_CALL(base, prepare(kDecodeProfile, _, _, _)).Times(1);

    auto artifacts = makeArtifacts(std::move(baseEngine), std::move(draftEngine));
    auto const maxAcceptedPerRound = artifacts.deployment.maxAcceptedTokensPerRound();
    ASSERT_GT(maxAcceptedPerRound, 1);
    int64_t const maxGenerateLength = maxAcceptedPerRound - 1;

    auto runtime = makeRuntime(std::move(artifacts));

    auto const request = makeGreedyRequest("a", maxGenerateLength);
    rt::LLMGenerationResponse response;
    ASSERT_TRUE(runtime.handleRequest(request, response, mStream));

    ASSERT_EQ(response.outputIds.size(), 1U);
    EXPECT_EQ(response.outputIds[0].size(), static_cast<size_t>(maxGenerateLength));
    EXPECT_EQ(response.finishReasons[0], rt::FinishReason::kLength);
}

// Given a base that disagrees with the draft from the second proposed position onward
// When the same token budget is generated
// Then it takes more than one verification
TEST_F(EagleAssemblyTest, SpendsMoreVerificationRoundsWhenTheBaseRejectsTheProposal)
{
    using ::testing::_;

    constexpr int64_t kMaxGenerateLength{kDraftingStep + 1};

    auto baseEngine = makeEngine();
    auto draftEngine = makeEngine();
    auto& base = *baseEngine;

    // The base disagrees with the draft from the second verified position onward: the draft proposed
    // kZeroLogitsToken everywhere, and this names a different token there. Rejection truncates the round, so the
    // same token budget can no longer be met by a single verification.
    //
    // Asserted as an inequality on purpose. The exact accept length follows from the accept rule, which
    // eagleAcceptTests.cpp already pins against an independent CPU reference; what belongs here is that the decoder
    // spends the rejection rather than silently committing a chain the base did not agree with.
    std::vector<int32_t> rejecting(kVerifySize, kZeroLogitsToken);
    for (size_t i = 1; i < rejecting.size(); ++i)
    {
        rejecting[i] = kZeroLogitsToken - 1;
    }
    ON_CALL(base, execute(_)).WillByDefault(emit(rejecting));

    EXPECT_CALL(base, prepare(kPrefillProfile, _, _, _)).Times(1);
    EXPECT_CALL(base, prepare(kDecodeProfile, _, _, _)).Times(::testing::AtLeast(2));

    auto artifacts = makeArtifacts(std::move(baseEngine), std::move(draftEngine));
    auto runtime = makeRuntime(std::move(artifacts));

    auto const request = makeGreedyRequest("a", kMaxGenerateLength);
    rt::LLMGenerationResponse response;
    ASSERT_TRUE(runtime.handleRequest(request, response, mStream));

    ASSERT_EQ(response.outputIds.size(), 1U);
    EXPECT_EQ(response.outputIds[0].size(), static_cast<size_t>(kMaxGenerateLength));
}

TEST_F(EagleAssemblyTest, CompactsTheBatchWhenOneSlotFinishesAheadOfTheOther)
{
    // EAGLE's per-slot state is a walked chain in the draft cache, plus any pending draft-prefill output.
    expectSurvivorKeepsDecodingAfterCompaction(kVerifySize);
}

} // namespace
