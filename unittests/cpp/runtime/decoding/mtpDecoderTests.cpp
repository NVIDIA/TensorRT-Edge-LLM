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

// MTP speculative decoding assembled around two substitute engines.

#include "profiling/timer.h"
#include "substituteEngine.h"

using namespace trt_edgellm;
using namespace substitute_engine;

namespace
{

Json makeMtpDraftConfig()
{
    Json config = makeSpecDraftConfig("mtp", kVerifySize);
    // MTP's draft consumes the base hidden state unchanged, so this equals the base hidden size.
    config["base_model_hidden_size"] = config["hidden_size"];
    return config;
}

class MtpAssemblyTest : public SpecAssemblyTest
{
protected:
    //! MTP's draft model ends in `F.log_softmax`.
    bool draftEmitsLogProbabilities() const override
    {
        return true;
    }

    std::filesystem::path stageModelDir() override
    {
        return writeSpecModelDir("mtpAssemblyTests", makeSpecBaseConfig("mtp", kVerifySize), makeMtpDraftConfig());
    }

    rt::SpecDecodeDraftingConfig drafting() const override
    {
        return makeMtpDrafting();
    }
};

TEST_F(MtpAssemblyTest, AssemblesTwoEnginesWithoutAnyEngineFileOnDisk)
{
    expectAssembledFromArtifactsAlone("mtp");
}

TEST_F(MtpAssemblyTest, InheritsResolvedDraftingFromInjectedArtifacts)
{
    auto artifacts = makeArtifacts();
    rt::DeploymentOptions options{};
    options.engineDir = mModelDir.string();
    rt::LLMInferenceRuntime runtime{std::move(artifacts), std::move(options), mStream};
    EXPECT_TRUE(runtime.hasDraftModel());
    EXPECT_STREQ(runtime.getSpeculativeDecodingStrategyName(), "mtp");
    rt::LLMGenerationResponse response;
    ASSERT_TRUE(runtime.handleRequest(makeGreedyRequest("a", kVerifySize), response, mStream));
    EXPECT_EQ(response.outputIds[0].size(), static_cast<size_t>(kVerifySize));
}

TEST_F(MtpAssemblyTest, RejectsConflictingDraftingBeforeConsumingInjectedArtifacts)
{
    auto artifacts = makeArtifacts();
    auto const* executor = artifacts.baseExecutor.get();
    rt::DeploymentOptions options{};
    options.engineDir = mModelDir.string();
    options.draftingConfig = drafting();
    ++*options.draftingConfig->draftTopK;
    std::string reason;
    try
    {
        rt::LLMInferenceRuntime runtime{std::move(artifacts), std::move(options), mStream};
    }
    catch (std::exception const& error)
    {
        reason = error.what();
    }
    EXPECT_THAT(reason, ::testing::HasSubstr("Injected draftingConfig conflicts"));
    EXPECT_EQ(artifacts.baseExecutor.get(), executor);
    EXPECT_NE(artifacts.draftExecutor, nullptr);
}

TEST_F(MtpAssemblyTest, RunsTheDraftChainThenOneBaseVerificationPerRound)
{
    expectChainProposalThenOneVerification();
}

TEST_F(MtpAssemblyTest, CompactsTheBatchWhenOneSlotFinishesAheadOfTheOther)
{
    // MTP's per-slot state is the walked chain in its own draft cache. The other three decoders are held to this
    // too; leaving the mode that has been in production longest out of it would be the odd exception.
    expectSurvivorKeepsDecodingAfterCompaction(kVerifySize);
}

TEST_F(MtpAssemblyTest, ForcedAcceptanceLengthIncludesBonusAndUsesExactRoundBudget)
{
    // Given an MTP proposal and a forced length including its bonus, when the output budget fits exactly one round,
    // then prefill emits one token and the round emits that length, including the AL=1 zero-draft case.
    for (int32_t const length : {1, 2, 4})
    {
        SCOPED_TRACE(length);
        auto artifacts = makeArtifacts();
        rt::DeploymentOptions options{};
        options.engineDir = mModelDir.string();
        options.benchmark = {1, length};
        rt::LLMInferenceRuntime runtime{std::move(artifacts), options, mStream};
        auto request = makeGreedyRequest("a", length + 1);
        request.preTokenizedInputIds = {{42}};
        request.ignoreEos = true;
        rt::LLMGenerationResponse response;
        ASSERT_TRUE(runtime.handleRequest(request, response, mStream));
        ASSERT_EQ(response.outputIds.size(), 1U);
        EXPECT_EQ(response.outputIds[0].size(), static_cast<size_t>(length + 1));
        EXPECT_EQ(response.decodeRoundCounts, std::vector<int32_t>{1});
    }
}

TEST_F(MtpAssemblyTest, RejectsInvalidForcedAcceptanceLength)
{
    // Given a zero or invalid negative forced length, when constructing the runtime,
    // then it rejects the control instead of silently treating it as natural acceptance.
    for (int32_t const length : {0, -2})
    {
        auto artifacts = makeArtifacts();
        rt::DeploymentOptions options{};
        options.engineDir = mModelDir.string();
        options.benchmark.forcedAcceptanceLength = length;
        EXPECT_THROW((rt::LLMInferenceRuntime{std::move(artifacts), options, mStream}), std::runtime_error);
    }
}

TEST_F(MtpAssemblyTest, OneRoundReportsEveryDraftForwardWithItsOwnOrdinal)
{
    // Given a three-step MTP draft and untimed substitute engines, when one full round runs,
    // then draft prefill, both proposal forwards and target verification remain distinct runtime-owned samples.
    auto artifacts = makeArtifacts();
    rt::DeploymentOptions options{};
    options.engineDir = mModelDir.string();
    options.benchmark.maxDecodeRounds = 1;
    rt::LLMInferenceRuntime runtime{std::move(artifacts), options, mStream};
    auto request = makeGreedyRequest("a", 8);
    request.preTokenizedInputIds = {{42}};
    request.ignoreEos = true;
    struct ProfilingReset
    {
        bool previous{getProfilingEnabled()};
        ~ProfilingReset()
        {
            setProfilingEnabled(previous);
        }
    } reset;
    setProfilingEnabled(true);
    rt::LLMGenerationResponse response;
    ASSERT_TRUE(runtime.handleRequest(request, response, mStream));
    using Forward = std::tuple<std::string, std::string, int32_t, int64_t, int64_t>;
    std::vector<Forward> forwards;
    for (auto const& sample : gTimer.getSamples())
    {
        if (sample.forward && sample.round == 0)
        {
            forwards.emplace_back(
                sample.stageId, sample.parentStage, sample.forwardIndex, sample.queryTokens, sample.selectedTokens);
        }
    }
    EXPECT_EQ(forwards,
        (std::vector<Forward>{{"forward/draft", "spec_decode_draft_prefill", 0, 1, 1},
            {"forward/draft", "spec_decode_draft_proposal", 1, kDraftingStep, 1},
            {"forward/draft", "spec_decode_draft_proposal", 2, kDraftingStep, 1},
            {"forward/base", "spec_decode_base_verification", 3, kVerifySize, kVerifySize}}));
}

} // namespace
