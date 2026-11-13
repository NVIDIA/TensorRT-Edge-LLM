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

// A runtime session over a speculative deployment: a sequence admitted into a running batch has its draft state
// seated in its own context step, and from then on every decode step runs one speculative round over all
// residents. What is under test is that seating changes nothing a sequence produces -- its tokens and its
// acceptance are those of the same request decoded alone -- driven against substitute engines whose output is a
// function of each row's token and position only, so any row, position or length mix-up shows as a wrong token.

#include "specSessionSupport.h"

#include "runtime/llmInferenceRuntime.h"
#include "runtime/runtimeSession.h"

#include <memory>
#include <string>
#include <vector>

using namespace trt_edgellm;
using namespace substitute_engine;

namespace
{

//! EAGLE-3's draft reads three base layers' hidden states (see eagleDecoderTests.cpp).
constexpr int32_t kEagleHiddenStateLayers{3};

Json makeEagleBaseConfig()
{
    Json config = makeSpecBaseConfig("eagle3", kVerifySize);
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

Json makeMtpDraftConfig()
{
    Json config = makeSpecDraftConfig("mtp", kVerifySize);
    config["base_model_hidden_size"] = config["hidden_size"];
    return config;
}

//! The two-engine deployments whose draft trails the target by the last accepted span. Their drafts follow the
//! substitute model one position ahead, so the fixture's proposals are accepted over varying lengths.
enum class TwoEngineMode
{
    kEagle,
    kMtp,
};

class SpecSessionTest : public SpecSessionAssemblyTest, public ::testing::WithParamInterface<TwoEngineMode>
{
protected:
    std::filesystem::path stageModelDir() override
    {
        return GetParam() == TwoEngineMode::kEagle
            ? writeSpecModelDir("specSessionEagleTests", makeEagleBaseConfig(), makeEagleDraftConfig())
            : writeSpecModelDir("specSessionMtpTests", makeSpecBaseConfig("mtp", kVerifySize), makeMtpDraftConfig());
    }

    rt::SpecDecodeDraftingConfig drafting() const override
    {
        return makeMtpDrafting();
    }

    //! Both draft models end in `F.log_softmax`.
    bool draftEmitsLogProbabilities() const override
    {
        return true;
    }
};

std::string modeName(::testing::TestParamInfo<TwoEngineMode> const& info)
{
    return info.param == TwoEngineMode::kEagle ? "Eagle" : "Mtp";
}

constexpr int64_t kMaxGenerateLength{kSessionGenerateLength};
std::vector<int32_t> const kResidentPrompt{5, 9, 17, 33, 21};
std::vector<int32_t> const kJoinerPrompt{42, 43, 44};

} // namespace

// Given two prompts whose drafts are accepted over varying lengths
// When the second is seated while the first is mid-generation
// Then each decodes exactly as it does alone (see expectJoinerMatchesSolo)
TEST_P(SpecSessionTest, Seating)
{
    rt::LLMGenerationResponse const alone = decodeBlocking(makeTokenizedRequest(kResidentPrompt, kMaxGenerateLength));
    ASSERT_EQ(alone.specVerifyCounts.size(), 1U);
    ASSERT_LT(alone.specVerifyCounts.front(), kMaxGenerateLength - 1)
        << "no round accepted a draft, so this fixture cannot tell a seated draft state from a broken one";
    expectSeating();
}

// Given a resident decoding speculatively
// When a sequence whose first token ends it joins, and sequences that disabled speculative decoding arrive
// Then the ended sequence leaves in its context step and the resident decodes as alone; an opt-out cannot join the
//      speculative batch (the scheduler's compatibility gate holds it until the session drains), opt-outs batch with
//      each other on the default decoder, and a speculative request cannot join them; only the speculative requests
//      require uniform sampling
TEST_P(SpecSessionTest, DisableSpecDecode)
{
    auto const residentRequest = makeTokenizedRequest(kResidentPrompt, kMaxGenerateLength);
    auto optedOut = makeTokenizedRequest(kJoinerPrompt, kMaxGenerateLength);
    optedOut.disableSpecDecode = true;
    auto otherOptOut = makeTokenizedRequest(kResidentPrompt, kMaxGenerateLength);
    otherOptOut.disableSpecDecode = true;
    auto sampledOptOut = makeTokenizedRequest(kJoinerPrompt, kMaxGenerateLength);
    sampledOptOut.temperature = 0.8F;
    sampledOptOut.topK = 4;
    sampledOptOut.disableSpecDecode = true;
    // Its first token is EOS (see kEndsSequence), so it ends in its context step.
    auto const endsAtOnce = makeTokenizedRequest({7, kEndsSequence}, kMaxGenerateLength);
    rt::LLMGenerationResponse const residentAlone = decodeBlocking(residentRequest);

    rt::LLMInferenceRuntime runtime = makeMarkovRuntime();
    std::unique_ptr<rt::RuntimeSession> session = runtime.openSession(mStream);
    ASSERT_NE(session, nullptr);
    EXPECT_TRUE(session->requiresUniformSampling(residentRequest));
    EXPECT_FALSE(session->requiresUniformSampling(optedOut));
    EXPECT_FALSE(session->requiresUniformSampling(sampledOptOut)) << "the default decoder samples per sequence";
    SpecSessionDriver driver(*session);
    driver.admitAndPrefill(residentRequest, 1);
    driver.decode();
    driver.admitAndPrefill(endsAtOnce, 2);
    ASSERT_EQ(driver.outcomes().count(2), 1U) << "a sequence whose first token ends it leaves in its context step";
    EXPECT_EQ(driver.outcomes().at(2).outputIds.front(), markovContinuation({7, kEndsSequence}, 1));
    EXPECT_ANY_THROW(session->admit(optedOut, 7, nullptr)) << "an opt-out runs on the default decoder";
    driver.drain();
    EXPECT_EQ(driver.outcomes().at(1).outputIds, residentAlone.outputIds);
    EXPECT_EQ(driver.outcomes().at(1).specVerifyCounts, residentAlone.specVerifyCounts);

    driver.admitAndPrefill(optedOut, 3);
    driver.admitAndPrefill(otherOptOut, 4);
    EXPECT_ANY_THROW(session->admit(residentRequest, 8, nullptr)) << "a speculative request waits for the opt-outs";
    driver.drain();
    EXPECT_EQ(driver.outcomes().at(3).outputIds.front(), markovContinuation(kJoinerPrompt, kMaxGenerateLength));
    EXPECT_EQ(driver.outcomes().at(4).outputIds.front(), markovContinuation(kResidentPrompt, kMaxGenerateLength));
    EXPECT_TRUE(driver.outcomes().at(3).specVerifyCounts.empty()) << "the default decoder served the opt-out";
    EXPECT_TRUE(driver.outcomes().at(4).specVerifyCounts.empty()) << "the default decoder served the opt-out";

    driver.admitAndPrefill(sampledOptOut, 5);
    driver.drain();
    EXPECT_TRUE(driver.outcomes().at(5).specVerifyCounts.empty()) << "the default decoder served the sampled opt-out";
    driver.admitAndPrefill(residentRequest, 6);
    driver.drain();
    EXPECT_EQ(driver.outcomes().at(6).outputIds, residentAlone.outputIds);
    EXPECT_EQ(driver.outcomes().at(6).specVerifyCounts, residentAlone.specVerifyCounts);
}

INSTANTIATE_TEST_SUITE_P(
    TwoEngineModes, SpecSessionTest, ::testing::Values(TwoEngineMode::kEagle, TwoEngineMode::kMtp), modeName);

namespace
{

//! The two-engine deployments with room for prompts that span context-cache pages: EAGLE over an attention base,
//! whose cache publishes block-granular endpoints, and MTP over a hybrid base, whose cache publishes exact
//! checkpoints of the recurrent state.
class SpecSessionReuseTest : public SpecSessionTest
{
protected:
    static constexpr int32_t kKvCapacity{3 * rt::kTOKENS_PER_PAGE};

    std::filesystem::path stageModelDir() override
    {
        bool const eagle = GetParam() == TwoEngineMode::kEagle;
        Json base = eagle ? makeEagleBaseConfig() : makeSpecBaseConfig("mtp", kVerifySize);
        Json draft = eagle ? makeEagleDraftConfig() : makeMtpDraftConfig();
        if (!eagle)
        {
            Json const hybrid = makeTinyHybridConfig();
            for (char const* key : {"num_attention_layers", "num_linear_attn_layers", "recurrent_state_num_heads",
                     "recurrent_state_head_dim", "recurrent_state_size", "conv_dim", "conv_kernel",
                     "recurrent_state_dtype", "conv_state_dtype"})
            {
                base[key] = hybrid[key];
            }
            // The verify pass replays GDN state over the accepted prefix, which takes FP32 states with dk == dv == 128
            // and a convolution over the q, k and v heads: one key head here.
            base["recurrent_state_head_dim"] = 128;
            base["recurrent_state_size"] = 128;
            base["recurrent_state_dtype"] = "fp32";
            base["conv_dim"] = 128 * (base["recurrent_state_num_heads"].get<int32_t>() + 2);
        }
        for (Json* config : {&base, &draft})
        {
            (*config)["builder_config"]["max_input_len"] = kKvCapacity - rt::kTOKENS_PER_PAGE / 2;
            (*config)["builder_config"]["max_kv_cache_capacity"] = kKvCapacity;
            // Room for the published prompt's pages beside the residents' working sets.
            (*config)["builder_config"]["max_kv_pool_pages"]
                = 2 * rt::computeMinimumKvPoolPages(kMaxBatchSize, kKvCapacity);
        }
        return writeSpecModelDir(eagle ? "specSessionReuseEagleTests" : "specSessionReuseMtpTests", base, draft);
    }

    void exposeBaseBindings(MockEngineExecutor& engine) override
    {
        using ::testing::Return;
        using ::testing::StrEq;
        for (char const* binding :
            {binding_names::kTreeParentIds, binding_names::kTreeDepths, binding_names::kSpecVerifyLayout})
        {
            ON_CALL(engine, hasIOTensor(StrEq(binding))).WillByDefault(Return(true));
            ON_CALL(engine, getBindingDataType(StrEq(binding))).WillByDefault(Return(trt_edgellm::DataType::kINT32));
        }
    }

    rt::LLMInferenceRuntime makeCachedMarkovRuntime()
    {
        rt::DeploymentOptions options{};
        options.engineDir = mModelDir.string();
        options.draftingConfig = drafting();
        options.contextCacheConfig.enabled = true;
        options.contextCacheConfig.maxRecords = 16;
        options.contextCacheConfig.maxSnapshots = 64;
        return rt::LLMInferenceRuntime{
            makeArtifacts(makeMarkovBase(), makeMarkovEngine(/*draft=*/true)), std::move(options), mStream};
    }
};

} // namespace

// Given a context-reuse speculative deployment, and a resident whose prompt spans two cache pages
// When the resident has published its prompt, and a sequence extending that prompt joins it
// Then the joiner reuses the published prefix (all but the last page on the attention base, the prompt up to its
//      volatile tail on the hybrid base), and each sequence decodes exactly as it does alone without a cache
TEST_P(SpecSessionReuseTest, JoinerReusesPrefix)
{
    std::vector<int32_t> residentPrompt(2 * rt::kTOKENS_PER_PAGE);
    for (size_t position = 0; position < residentPrompt.size(); ++position)
    {
        residentPrompt[position] = 5 + static_cast<int32_t>((position * 7) % 50);
    }
    std::vector<int32_t> joinerPrompt = residentPrompt;
    joinerPrompt.insert(joinerPrompt.end(), {42, 43, 44});
    auto residentRequest = makeTokenizedRequest(residentPrompt, kMaxGenerateLength);
    auto joinerRequest = makeTokenizedRequest(joinerPrompt, kMaxGenerateLength);
    // A hybrid base publishes prefill state only and stops its checkpoint before the volatile tail.
    constexpr int32_t kReplayTail{3};
    for (rt::LLMGenerationRequest* request : {&residentRequest, &joinerRequest})
    {
        request->contextCacheCommitPolicy = rt::ContextCacheCommitPolicy::kPrefillStateOnly;
        request->contextCacheReplayTailLength = kReplayTail;
    }
    rt::LLMGenerationResponse const residentAlone = decodeBlocking(residentRequest);
    rt::LLMGenerationResponse const joinerAlone = decodeBlocking(joinerRequest);

    rt::LLMInferenceRuntime runtime = makeCachedMarkovRuntime();
    ASSERT_TRUE(runtime.supportsSeatedAdmission());
    std::unique_ptr<rt::RuntimeSession> session = runtime.openSession(mStream);
    ASSERT_NE(session, nullptr);
    SpecSessionDriver driver(*session);
    driver.admitAndPrefill(residentRequest, 1);
    EXPECT_EQ(driver.lastPrefillStart(), 0);
    driver.decode();
    driver.admitAndPrefill(joinerRequest, 2);
    EXPECT_EQ(driver.lastPrefillStart(),
        GetParam() == TwoEngineMode::kEagle ? rt::kTOKENS_PER_PAGE
                                            : static_cast<int32_t>(residentPrompt.size()) - kReplayTail);
    driver.drain();

    ASSERT_EQ(driver.outcomes().count(1), 1U);
    ASSERT_EQ(driver.outcomes().count(2), 1U);
    EXPECT_EQ(driver.outcomes().at(1).outputIds, residentAlone.outputIds);
    EXPECT_EQ(driver.outcomes().at(1).specVerifyCounts, residentAlone.specVerifyCounts);
    EXPECT_EQ(driver.outcomes().at(2).outputIds, joinerAlone.outputIds);
    EXPECT_EQ(driver.outcomes().at(2).specVerifyCounts, joinerAlone.specVerifyCounts);
}

// The PREFILL_STATE_ONLY rule of a hybrid base follows the sequence's own lookup policy: a sequence that bypasses the
// cache is admitted under the default commit policy.
TEST_P(SpecSessionReuseTest, PrefillStateOnlyRule)
{
    std::vector<int32_t> prompt(rt::kTOKENS_PER_PAGE + 3);
    for (size_t position = 0; position < prompt.size(); ++position)
    {
        prompt[position] = 5 + static_cast<int32_t>((position * 7) % 50);
    }
    auto bypassing = makeTokenizedRequest(prompt, kMaxGenerateLength);
    bypassing.requests.front().contextCacheLookupPolicy = rt::ContextCacheLookupPolicy::kBypass;
    auto reusing = makeTokenizedRequest(prompt, kMaxGenerateLength);

    rt::LLMInferenceRuntime runtime = makeCachedMarkovRuntime();
    std::unique_ptr<rt::RuntimeSession> session = runtime.openSession(mStream);
    ASSERT_NE(session, nullptr);
    if (GetParam() == TwoEngineMode::kMtp)
    {
        EXPECT_THROW(session->admit(reusing, 1, nullptr), std::invalid_argument);
        EXPECT_TRUE(session->idle());
    }
    SpecSessionDriver driver(*session);
    driver.admitAndPrefill(bypassing, 2);
    EXPECT_EQ(driver.lastPrefillStart(), 0);
    driver.drain();
    EXPECT_EQ(driver.outcomes().count(2), 1U);
}

INSTANTIATE_TEST_SUITE_P(
    TwoEngineModes, SpecSessionReuseTest, ::testing::Values(TwoEngineMode::kEagle, TwoEngineMode::kMtp), modeName);

namespace
{

//! An MTP deployment whose draft engine was built for a smaller batch than its base.
class SmallDraftBatchSessionTest : public SpecSessionAssemblyTest
{
protected:
    std::filesystem::path stageModelDir() override
    {
        Json draft = makeMtpDraftConfig();
        draft["builder_config"]["max_batch_size"] = 1;
        draft["builder_config"]["max_kv_pool_pages"] = rt::computeMinimumKvPoolPages(1, kMtpKvCacheCapacity);
        return writeSpecModelDir("specSessionSmallDraftTests", makeSpecBaseConfig("mtp", kVerifySize), draft);
    }

    rt::SpecDecodeDraftingConfig drafting() const override
    {
        return makeMtpDrafting();
    }

    bool draftEmitsLogProbabilities() const override
    {
        return true;
    }
};

} // namespace

// Given a speculative deployment whose draft engine runs a smaller batch than its base
// When a session is full at the draft's batch
// Then the runtime reports the smaller batch, and a further admission waits instead of overrunning the draft engine
TEST_F(SmallDraftBatchSessionTest, BoundedByDraftBatch)
{
    rt::LLMInferenceRuntime runtime = makeMarkovRuntime();
    EXPECT_EQ(runtime.maxBatchSize(), 1);
    std::unique_ptr<rt::RuntimeSession> session = runtime.openSession(mStream);
    ASSERT_NE(session, nullptr);
    EXPECT_EQ(session->capabilities().maxBatchSize, 1);
    SpecSessionDriver driver(*session);
    driver.admitAndPrefill(makeTokenizedRequest(kResidentPrompt, kMaxGenerateLength), 1);
    rt::AdmissionResult const refused
        = session->admit(makeTokenizedRequest(kJoinerPrompt, kMaxGenerateLength), 2, nullptr);
    EXPECT_EQ(refused.status, rt::AdmissionResult::Status::kNoCapacity);
    driver.drain();
    EXPECT_EQ(driver.outcomes().at(1).outputIds.front(), markovContinuation(kResidentPrompt, kMaxGenerateLength));
}
