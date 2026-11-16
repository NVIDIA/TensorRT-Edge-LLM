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

#include "fakeEngineExecutor.h"
#include "substituteEngine.h"

#include <algorithm>

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
    //! EAGLE's draft model ends in `F.log_softmax`.
    bool draftEmitsLogProbabilities() const override
    {
        return true;
    }

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
    auto draftEngine = makeLogProbDraftEngine();
    auto& base = *baseEngine;

    // Neither engine names a token, so both peak at kDefaultRowToken and the base's argmax agrees with the draft's
    // proposal at every verified position. The agreement is a real argmax agreement rather than a tie both
    // engines happened to break the same way, which is what it was while an unnamed row was left flat. Full
    // agreement is the case the technique is built for: one verification commits `draftSteps + 1` tokens, which
    // is what DeploymentConfig::maxAcceptedTokensPerRound reports for EAGLE.
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
    auto draftEngine = makeLogProbDraftEngine();
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

TEST_F(EagleAssemblyTest, RefreshesResidentPleRowsForBaseVerification)
{
    using ::testing::_;

    auto baseEngine = makeEngine();
    std::vector<int32_t> verificationRows;
    ON_CALL(*baseEngine, prepare(_, _, _, _))
        .WillByDefault([this, &verificationRows](
                           int32_t profile, rt::InferenceDims const&, rt::TensorMap const& map, cudaStream_t stream) {
            mLogits = map.get(binding_names::kLogits);
            if (profile == kDecodeProfile)
            {
                auto const* rows = map.get("ple_row_map");
                EXPECT_NE(rows, nullptr);
                if (rows == nullptr)
                {
                    return false;
                }
                verificationRows.resize(static_cast<size_t>(rows->getShape().volume()));
                CUDA_CHECK(cudaMemcpyAsync(verificationRows.data(), rows->rawPointer(),
                    verificationRows.size() * sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
                CUDA_CHECK(cudaStreamSynchronize(stream));
            }
            return true;
        });
    auto artifacts = makeArtifacts(std::move(baseEngine), makeEngine());
    auto& config = artifacts.deployment.base;
    config.gemmaPleEnabled = true;
    config.gemmaPleLayerCount = kEagleHiddenStateLayers;
    config.gemmaPleHiddenSize = 1;
    config.plePool = rt::PlePoolConfig{kVocabSize, kEagleHiddenStateLayers, rt::PleTable::DType::kFP16, 1.0};
    config.plePool.source = {"weight", rt::PleTable::DType::kFP16, kVocabSize * kEagleHiddenStateLayers * 2, 1.0,
        {{0, "003c"}}, "ple_table.bin", 0};
    config.plePool.checkpointDir = mModelDir.string();
    std::vector<uint16_t> const table(kVocabSize * kEagleHiddenStateLayers, 0x3C00);
    {
        std::ofstream shard(mModelDir / "ple_table.bin", std::ios::binary);
        shard.write(reinterpret_cast<char const*>(table.data()), table.size() * sizeof(uint16_t));
        ASSERT_TRUE(shard.good());
    }
    auto runtime = makeRuntime(std::move(artifacts));
    auto const request = makeGreedyRequest("aa", kVerifySize);
    rt::LLMGenerationResponse response;
    ASSERT_TRUE(runtime.handleRequest(request, response, mStream));
    EXPECT_THAT(
        verificationRows, ::testing::AllOf(::testing::SizeIs(kVerifySize), ::testing::Each(kDefaultRowToken + 1)));
}

// Given a base that disagrees with the draft from the second proposed position onward
// When the same token budget is generated
// Then it takes more than one verification
TEST_F(EagleAssemblyTest, SpendsMoreVerificationRoundsWhenTheBaseRejectsTheProposal)
{
    using ::testing::_;

    constexpr int64_t kMaxGenerateLength{kDraftingStep + 1};

    auto baseEngine = makeEngine();
    auto draftEngine = makeLogProbDraftEngine();
    auto& base = *baseEngine;

    // The base disagrees with the draft from the second verified position onward: the draft proposed
    // kDefaultRowToken everywhere, and this names a different token there. Rejection truncates the round, so the
    // same token budget can no longer be met by a single verification.
    //
    // Asserted as an inequality on purpose. The exact accept length follows from the accept rule, which
    // specTargetSelectionTests.cpp already pins against an independent CPU reference; what belongs here is that the
    // decoder spends the rejection rather than silently committing a chain the base did not agree with.
    std::vector<int32_t> rejecting(kVerifySize, kDefaultRowToken);
    for (size_t i = 1; i < rejecting.size(); ++i)
    {
        rejecting[i] = kDefaultRowToken - 1;
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

// --------------------------------------------------------------------------
// EAGLE candidate-head drafts: the draft engine emits each row's candidates as sorted full-vocabulary ids with their
// log-probabilities instead of logits, and every tree level takes its children from the first draftTopK of them.
// --------------------------------------------------------------------------

constexpr int32_t kCandidateTopK{2};
constexpr int32_t kCandidateSteps{3};
//! Draft-tree table length 1 + 2 + 2 x 2^2 = 11.
constexpr int32_t kCandidateTreeSize{1 + kCandidateTopK + (kCandidateSteps - 1) * kCandidateTopK * kCandidateTopK};
constexpr int32_t kDraftCandidateCount{8};
//! A verification narrower than the table, so the candidates' log-probs decide which nodes are verified.
constexpr int32_t kCandidateNarrowVerifySize{5};
//! Logit of a nearly flat candidate row's leader: just above the highest background logit (-1.03125, token 83), so
//! it still leads, with a log-prob of about -1.97 instead of a confident row's -0.27.
constexpr float kFlatPeakLogit{-1.0F};

Json makeCandidateEagleBaseConfig()
{
    Json config = makeEagleBaseConfig();
    config["builder_config"]["max_spec_verify_tokens"] = kCandidateTreeSize;
    return config;
}

Json makeCandidateEagleDraftConfig()
{
    Json config = makeEagleDraftConfig();
    config["builder_config"]["max_spec_draft_tokens"] = kCandidateSteps * kCandidateTopK;
    config["eagle3_config"]["candidate_head"]
        = Json{{"type", "low_rank"}, {"rank", 16}, {"num_candidates", kDraftCandidateCount}};
    return config;
}

class EagleCandidateAssemblyTest : public SpecAssemblyTest
{
protected:
    std::filesystem::path stageModelDir() override
    {
        return writeSpecModelDir(
            "eagleCandidateAssemblyTests", makeCandidateEagleBaseConfig(), makeCandidateEagleDraftConfig());
    }

    rt::SpecDecodeDraftingConfig drafting() const override
    {
        rt::SpecDecodeDraftingConfig drafting{};
        drafting.draftTopK = kCandidateTopK;
        drafting.draftingStep = kCandidateSteps;
        drafting.verifyTokens = mVerifyTokens;
        return drafting;
    }

    //! Verification width; a test lowers it below the table length before it assembles the runtime.
    int32_t mVerifyTokens{kCandidateTreeSize};
};

// Given an EAGLE candidate-head runtime whose base computes each output from the token and the committed cache it
// reads (FakeEngineExecutor), a candidate draft, forced acceptance length 1, and two prompts of different lengths
// decoded as one batch
// When the batch runs once with disableSpecDecode on a reference runtime, and speculatively once per verify tree: all
// 11 nodes of the K 2, D 3 draft tree, and its 5 best nodes
// Then every slot returns the same tokens all three ways while each speculative round verified the whole tree: a
// commit that keeps a rejected node's KV entry, or a root row that depends on the tree it is verified with, changes
// what the base reads back
TEST_F(EagleCandidateAssemblyTest, AllRejectRoundsReturnWhatTheVanillaPathReturns)
{
    constexpr int64_t kGenerateLength{8};
    std::vector<std::string> const prompts{makePrompt(5, /*variant=*/0), makePrompt(9, /*variant=*/1)};
    auto const makeBase = [] {
        auto engine = makeFakeEngine(makeCandidateEagleBaseConfig());
        engine->exposeInt32Binding(binding_names::kTreeParentIds);
        engine->exposeInt32Binding(binding_names::kTreeDepths);
        engine->exposeInt32Binding(binding_names::kSpecVerifyLayout);
        return engine;
    };

    // A forced acceptance length is a runtime-wide benchmark option that refuses a vanilla request, so the reference
    // runs on its own runtime around an identical base.
    rt::LLMGenerationResponse vanilla;
    {
        auto runtime = makeRuntime(makeArtifacts(makeBase(), makeCandidateDraftEngine(kDraftCandidateCount)));
        auto request = makeGreedyBatch(prompts, kGenerateLength);
        request.disableSpecDecode = true;
        ASSERT_TRUE(runtime.handleRequest(request, vanilla, mStream));
    }
    ASSERT_EQ(vanilla.outputIds.size(), prompts.size());
    ASSERT_NE(vanilla.outputIds[0], vanilla.outputIds[1]) << "the fixture cannot tell the two slots apart";

    for (int32_t const verifyTokens : {kCandidateTreeSize, kCandidateNarrowVerifySize})
    {
        SCOPED_TRACE(verifyTokens);
        mVerifyTokens = verifyTokens;
        std::unique_ptr<FakeEngineExecutor> baseEngine = makeBase();
        FakeEngineExecutor const& base = *baseEngine;
        rt::DeploymentOptions options{};
        options.engineDir = mModelDir.string();
        options.draftingConfig = drafting();
        options.benchmark.forcedAcceptanceLength = 1;
        rt::LLMInferenceRuntime runtime{
            makeArtifacts(std::move(baseEngine), makeCandidateDraftEngine(kDraftCandidateCount)), options, mStream};
        rt::LLMGenerationResponse speculative;
        ASSERT_TRUE(runtime.handleRequest(makeGreedyBatch(prompts, kGenerateLength), speculative, mStream));

        std::vector<int64_t> verificationTokens;
        for (auto const& round : base.preparedRounds())
        {
            if (round.profileIndex == kDecodeProfile)
            {
                verificationTokens.push_back(round.dims.seqLen);
            }
        }
        EXPECT_THAT(verificationTokens,
            ::testing::AllOf(::testing::SizeIs(kGenerateLength - 1),
                ::testing::Each(static_cast<int64_t>(prompts.size()) * verifyTokens)))
            << "a round verified less than the whole tree, so it rejected less than one";
        ASSERT_EQ(speculative.outputIds.size(), prompts.size());
        for (size_t slot = 0; slot < prompts.size(); ++slot)
        {
            EXPECT_EQ(speculative.outputIds[slot], vanilla.outputIds[slot]) << "slot " << slot;
        }
    }
}

// --------------------------------------------------------------------------
// EAGLE over a hybrid (attention + GDN) base: verification leaves one recurrent and conv snapshot per tree node, and
// the commit replays the accepted path into the persistent state.
// --------------------------------------------------------------------------

constexpr int32_t kHybridConditioningLayers{2};
//! The conv snapshot a verification leaves for verify row r is kConvStashBase + r. It is above kFakeModulus, so no
//! value the fake base computes can be mistaken for it.
constexpr int32_t kConvStashBase{300};

//! The makeHybridMtpBaseConfig() geometry as an EAGLE base: GDN replay needs FP32 state with 128-wide K and V heads,
//! and conv_dim 512 gives one K head for two V heads.
Json makeHybridEagleBaseConfig(int32_t verifyTokens)
{
    Json config = makeSpecBaseConfig("eagle3", verifyTokens);
    config["num_attention_layers"] = 1;
    config["num_linear_attn_layers"] = 1;
    config["recurrent_state_num_heads"] = 2;
    config["recurrent_state_head_dim"] = 128;
    config["recurrent_state_size"] = 128;
    config["conv_dim"] = 512;
    config["conv_kernel"] = 4;
    config["recurrent_state_dtype"] = "fp32";
    config["conv_state_dtype"] = "fp16";
    config["num_hidden_layers"] = kHybridConditioningLayers;
    config["eagle_hidden_state_layers"] = Json::array({0, 1});
    return config;
}

//! A hybrid base that leaves a conv snapshot per verified node, as a tree-verify GDN layer does: after every
//! decode-profile forward it fills each sequence's intermediate conv state at verify row r with kConvStashBase + r.
class ConvStashingBaseEngine : public FakeEngineExecutor
{
public:
    using FakeEngineExecutor::FakeEngineExecutor;

    bool execute(cudaStream_t stream) override
    {
        if (!FakeEngineExecutor::execute(stream))
        {
            return false;
        }
        if (preparedRounds().back().profileIndex != kDecodeProfile)
        {
            return true;
        }
        rt::Tensor* const stash = boundTensor(binding_names::formatIntermediateConvStateName(0));
        if (stash == nullptr || stash->getShape().getNumDims() != 4
            || stash->getDataType() != trt_edgellm::DataType::kHALF)
        {
            ADD_FAILURE() << "verification ran without a [batch, rows, convDim, convKernel] FP16 conv snapshot bound";
            return false;
        }
        auto const shape = stash->getShape();
        int64_t const rowElements = shape[2] * shape[3];
        std::vector<half> host(static_cast<size_t>(shape.volume()));
        for (int64_t row = 0; row < shape[0] * shape[1]; ++row)
        {
            auto const value = __float2half(static_cast<float>(kConvStashBase + row % shape[1]));
            std::fill_n(host.begin() + row * rowElements, rowElements, value);
        }
        CUDA_CHECK(cudaMemcpyAsync(
            stash->rawPointer(), host.data(), host.size() * sizeof(half), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        return true;
    }
};

constexpr int32_t kHybridTopK{8};
constexpr int32_t kHybridSteps{8};
//! The largest tree the limits allow. Its draft table (1 + 8 + 7 x 64 = 457 nodes) spans more than one 256-entry
//! top-K block, so a selection that broke score ties by block could move a deep chain node ahead of the root.
constexpr int32_t kHybridVerifySize{64};

class EagleHybridAssemblyTest : public SpecAssemblyTest
{
protected:
    std::filesystem::path stageModelDir() override
    {
        Json draft = makeSpecDraftConfig("eagle3", kHybridSteps * kHybridTopK);
        draft["base_model_hidden_size"] = kHybridConditioningLayers * draft["hidden_size"].get<int32_t>();
        return writeSpecModelDir("eagleHybridAssemblyTests", makeHybridEagleBaseConfig(kHybridVerifySize), draft);
    }

    rt::SpecDecodeDraftingConfig drafting() const override
    {
        rt::SpecDecodeDraftingConfig drafting{};
        drafting.draftTopK = kHybridTopK;
        drafting.draftingStep = kHybridSteps;
        drafting.verifyTokens = kHybridVerifySize;
        return drafting;
    }
};

// Given a hybrid (attention + GDN) EAGLE base whose verification leaves conv snapshot kConvStashBase + r at verify row
// r, and a dense draft certain of kDefaultRowToken (log-prob 0, every other token -30) that agrees with the base, over
// a K 8, D 8 tree verified on 64 of its 457 nodes, so every node of the agreed chain ties the root's score 0
// When one round verifies and commits, and the next verification starts
// Then the root stays at verify row 0 and the accepted path is the chain at rows 0 to 8, so the sequence's persistent
// conv state holds the snapshot of row 8 and the output is the agreed token throughout: the commit replays the
// accepted path into the recurrent state rather than keeping what verification left in place
TEST_F(EagleHybridAssemblyTest, CommitsTheAcceptedPathIntoTheRecurrentState)
{
    auto stashingBase = std::make_unique<ConvStashingBaseEngine>(
        FakeLayerComposition::fromConfigJson(makeHybridEagleBaseConfig(kHybridVerifySize)));
    stashingBase->forceOutputTokens(agreedLogitsRows(kDefaultRowToken, kMaxBatchSize * kHybridVerifySize));
    stashingBase->exposeInt32Binding(binding_names::kTreeParentIds);
    stashingBase->exposeInt32Binding(binding_names::kTreeDepths);
    stashingBase->exposeInt32Binding(binding_names::kSpecVerifyLayout);
    std::string const convState = binding_names::formatConvStateName(0, /*isPast=*/false);
    stashingBase->recordStatePool(convState);
    stashingBase->recordBinding(binding_names::kStateIndices);
    FakeEngineExecutor const& base = *stashingBase;

    auto draft = makeLogProbDraftEngine();
    ON_CALL(*draft, execute(::testing::_)).WillByDefault([this](cudaStream_t stream) {
        constexpr float kOtherLogProb{-30.0F};
        int64_t const width = mLogits->getShape()[1];
        std::vector<float> host(static_cast<size_t>(mLogits->getShape().volume()), kOtherLogProb);
        for (size_t row = 0; row < host.size() / static_cast<size_t>(width); ++row)
        {
            host[row * static_cast<size_t>(width) + kDefaultRowToken] = 0.0F;
        }
        CUDA_CHECK(cudaMemcpyAsync(
            mLogits->rawPointer(), host.data(), host.size() * sizeof(float), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        return true;
    });
    auto runtime = makeRuntime(makeArtifacts(std::move(stashingBase), std::move(draft)));

    // The prefill token, one fully accepted round of draftSteps + 1, and one token of a second round.
    constexpr int64_t kGenerateLength{1 + (kHybridSteps + 1) + 1};
    rt::LLMGenerationResponse response;
    ASSERT_TRUE(runtime.handleRequest(makeGreedyRequest("a", kGenerateLength), response, mStream));

    ASSERT_EQ(response.outputIds.size(), 1U);
    EXPECT_THAT(
        response.outputIds[0], ::testing::AllOf(::testing::SizeIs(kGenerateLength), ::testing::Each(kDefaultRowToken)));
    ASSERT_GE(base.preparedRounds().size(), 3U) << "prefill, the first verification and the second verification";
    auto const& stateRows = base.preparedBinding(binding_names::kStateIndices);
    ASSERT_FALSE(stateRows[2].empty());
    int32_t const stateRow = stateRows[2][0];
    EXPECT_EQ(base.preparedStatePool(convState)[2].at(static_cast<size_t>(stateRow)), kConvStashBase + kHybridSteps);
}

class EagleCandidateHybridAssemblyTest : public EagleCandidateAssemblyTest
{
protected:
    std::filesystem::path stageModelDir() override
    {
        Json draft = makeCandidateEagleDraftConfig();
        draft["base_model_hidden_size"] = kHybridConditioningLayers * draft["hidden_size"].get<int32_t>();
        return writeSpecModelDir(
            "eagleCandidateHybridAssemblyTests", makeHybridEagleBaseConfig(kCandidateTreeSize), draft);
    }
};

// Given a hybrid (attention + GDN) EAGLE base whose verification leaves conv snapshot kConvStashBase + r at verify row
// r, a candidate-head draft whose root row is confident in 40 (second candidate 83), whose first-level row under 40 is
// nearly flat (60 just ahead of 83) while the row under the root's 83 is confident in 61, a verification of 5 of the
// 11 draft-tree nodes, and a base that answers 40, 60 and 80 at verify rows 0, 1 and 2 and the bonus token
// kAgreedToken at verify row 4
// When one round verifies and commits, and the next verification starts
// Then the base verifies the root, 40, both children of 40 (60, then 83) and 80 under 60, with parent rows -1, 0, 1,
// 1, 2; the round emits 40, 60, 80 and kAgreedToken; and the sequence's persistent conv state holds the snapshot of
// row 4: the cumulative candidate log-probs pick the verified nodes, and the commit replays the accepted path, rows 0,
// 1, 2 and 4, into the recurrent state
TEST_F(EagleCandidateHybridAssemblyTest, CandidateRoundVerifiesAndCommitsTheAcceptedPath)
{
    constexpr int32_t kBonusToken{kAgreedToken};
    auto stashingBase = std::make_unique<ConvStashingBaseEngine>(
        FakeLayerComposition::fromConfigJson(makeHybridEagleBaseConfig(kCandidateTreeSize)));
    // Prefill decodes the tree root; every verification answers 40, 60 and 80 at rows 0 to 2 and the bonus at row 4.
    stashingBase->forceOutputTokensPerRound({{kDefaultRowToken}, {40, 60, 80, kDefaultRowToken, kBonusToken}});
    stashingBase->exposeInt32Binding(binding_names::kTreeParentIds);
    stashingBase->exposeInt32Binding(binding_names::kTreeDepths);
    stashingBase->exposeInt32Binding(binding_names::kSpecVerifyLayout);
    stashingBase->recordBinding(binding_names::kTreeParentIds);
    stashingBase->recordFloatBinding(binding_names::kInputsEmbeds);
    std::string const convState = binding_names::formatConvStateName(0, /*isPast=*/false);
    stashingBase->recordStatePool(convState);
    stashingBase->recordBinding(binding_names::kStateIndices);
    FakeEngineExecutor const& base = *stashingBase;
    mVerifyTokens = kCandidateNarrowVerifySize;
    auto runtime = makeRuntime(makeArtifacts(std::move(stashingBase),
        makeCandidateDraftEngine(
            kDraftCandidateCount, {{40}, {60, 61}, {80, 81}}, {{}, {kFlatPeakLogit, kPeakLogit}, {}})));

    // The prefill token, the round's three accepted tokens and its bonus, and one token of a second round.
    constexpr int64_t kGenerateLength{6};
    rt::LLMGenerationResponse response;
    ASSERT_TRUE(runtime.handleRequest(makeGreedyRequest("a", kGenerateLength), response, mStream));

    ASSERT_EQ(response.outputIds.size(), 1U);
    EXPECT_THAT(response.outputIds[0], ::testing::ElementsAre(kDefaultRowToken, 40, 60, 80, kBonusToken, ::testing::_));
    ASSERT_GE(base.preparedRounds().size(), 3U) << "prefill, the first verification and the second verification";
    // Channel 0 of a token's embedding row is fakeEmbeddingValue(token, 0).
    auto const& embeds = base.preparedFloatBinding(binding_names::kInputsEmbeds)[1];
    size_t const width = embeds.size() / kCandidateNarrowVerifySize;
    std::vector<double> verified;
    std::vector<double> expected;
    for (int32_t const token : {kDefaultRowToken, 40, 60, 83, 80})
    {
        verified.push_back(embeds.at(verified.size() * width));
        expected.push_back(fakeEmbeddingValue(token, 0));
    }
    EXPECT_EQ(verified, expected);
    EXPECT_EQ(base.preparedBinding(binding_names::kTreeParentIds)[1], (std::vector<int32_t>{-1, 0, 1, 1, 2}));
    auto const& stateRows = base.preparedBinding(binding_names::kStateIndices);
    ASSERT_FALSE(stateRows[2].empty());
    int32_t const stateRow = stateRows[2][0];
    EXPECT_EQ(base.preparedStatePool(convState)[2].at(static_cast<size_t>(stateRow)), kConvStashBase + 4);
}

} // namespace
