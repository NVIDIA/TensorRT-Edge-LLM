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

// DSpark speculative decoding assembled around two substitute engines.

#include "specSessionSupport.h"
#include "substituteEngine.h"

#include <array>
#include <numeric>

using namespace trt_edgellm;
using namespace substitute_engine;

namespace
{
// --------------------------------------------------------------------------
// DSpark speculative decoding.
//
// The one decoder here that genuinely needs a file: loadHeadSidecars() is
// unconditional and a missing tensor is a hard failure, so the test writes a
// real safetensors through the same production API the rest of the system uses.
// --------------------------------------------------------------------------

constexpr int32_t kDSparkBlockSize{4};

//! Rank of the low-rank Markov head. Only the shape matters here; the weights are zeroed.
constexpr int32_t kDSparkMarkovRank{8};

void addDSparkConfig(Json& config)
{
    Json dspark;
    dspark["target_layer_ids"] = Json::array({0, 1});
    dspark["block_size"] = kDSparkBlockSize;
    // The default is a real Qwen id, outside this deployment's 128-token vocabulary.
    dspark["mask_token_id"] = 3;
    dspark["markov_head_type"] = "vanilla";
    dspark["markov_rank"] = kDSparkMarkovRank;
    // Leaving the confidence head off keeps the sidecar to the two required matrices, and is also what lets the
    // scheduler stay off: the decoder refuses a scheduler without confidence tensors.
    dspark["enable_confidence_head"] = false;
    config["dspark_config"] = dspark;
}

Json makeDSparkBaseConfig()
{
    Json config = makeSpecBaseConfig("dspark", kDSparkBlockSize);
    addDSparkConfig(config);
    return config;
}

Json makeDSparkDraftConfig()
{
    Json config = makeSpecDraftConfig("dspark", kDSparkBlockSize);
    addDSparkConfig(config);
    config["base_model_hidden_size"] = 2 * config["hidden_size"].get<int32_t>();
    return config;
}

//! Write the heads sidecar DSpark loads unconditionally.
//!
//! Built with `saveSafetensors`, the same call the rest of the system writes weights with, so the test knows only
//! the tensor names and shapes the decoder validates -- which is the actual contract -- and not the file format.
void writeDSparkHeadsSidecar(std::filesystem::path const& dir, int32_t draftVocabSize, cudaStream_t stream)
{
    std::vector<rt::Tensor> heads;
    for (char const* name : {"markov_w1", "markov_w2"})
    {
        rt::Tensor matrix(
            {draftVocabSize, kDSparkMarkovRank}, rt::DeviceType::kGPU, trt_edgellm::DataType::kHALF, name);
        CUDA_CHECK(cudaMemsetAsync(
            matrix.rawPointer(), 0, static_cast<size_t>(matrix.getShape().volume()) * sizeof(half), stream));
        heads.push_back(std::move(matrix));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    if (!rt::safetensors::saveSafetensors(dir / binding_names::kDSparkHeadsFileName, heads, stream))
    {
        throw std::runtime_error("failed to write the DSpark heads sidecar");
    }
}

void writeReducedDSparkSidecars(
    std::filesystem::path const& dir, int32_t targetVocabSize, int32_t draftVocabSize, cudaStream_t stream)
{
    std::vector<rt::Tensor> heads;
    for (auto const& [name, rows] :
        std::array<std::pair<char const*, int32_t>, 2>{{{"markov_w1", targetVocabSize}, {"markov_w2", draftVocabSize}}})
    {
        rt::Tensor matrix({rows, kDSparkMarkovRank}, rt::DeviceType::kGPU, trt_edgellm::DataType::kHALF, name);
        CUDA_CHECK(cudaMemsetAsync(
            matrix.rawPointer(), 0, static_cast<size_t>(matrix.getShape().volume()) * sizeof(half), stream));
        heads.push_back(std::move(matrix));
    }
    rt::Tensor vocabMap({draftVocabSize}, rt::DeviceType::kGPU, trt_edgellm::DataType::kINT32, "draft_vocab_map");
    std::vector<int32_t> hostMap(static_cast<size_t>(draftVocabSize));
    std::iota(hostMap.begin(), hostMap.end(), 0);
    CUDA_CHECK(cudaMemcpyAsync(
        vocabMap.rawPointer(), hostMap.data(), hostMap.size() * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    std::vector<rt::Tensor> vocabMapTensors;
    vocabMapTensors.push_back(std::move(vocabMap));
    if (!rt::safetensors::saveSafetensors(dir / binding_names::kDSparkHeadsFileName, heads, stream)
        || !rt::safetensors::saveSafetensors(dir / binding_names::kDraftVocabMapFileName, vocabMapTensors, stream))
    {
        throw std::runtime_error("failed to write reduced-vocabulary DSpark sidecars");
    }
}

class DSparkAssemblyTest : public SpecSessionAssemblyTest
{
protected:
    std::filesystem::path stageModelDir() override
    {
        auto const dir = writeSpecModelDir("dsparkAssemblyTests", makeDSparkBaseConfig(), makeDSparkDraftConfig());
        writeDSparkHeadsSidecar(dir, static_cast<int32_t>(kVocabSize), mStream);
        return dir;
    }

    rt::SpecDecodeDraftingConfig drafting() const override
    {
        rt::SpecDecodeDraftingConfig config{};
        config.draftTopK = 1;
        config.proposalTokens = kDSparkBlockSize - 1;
        config.verifyTokens = kDSparkBlockSize;
        return config;
    }
};

TEST_F(DSparkAssemblyTest, AssemblesTwoEnginesAroundTheHeadsSidecar)
{
    expectAssembledFromArtifactsAlone("dspark");
    // The one file this deployment does need. DSpark loads it unconditionally, unlike EAGLE's d2t and DFlash's
    // draft vocab map, both of which are conditional and absent in their tests.
    EXPECT_TRUE(std::filesystem::exists(mModelDir / binding_names::kDSparkHeadsFileName));
}

// Given a heads sidecar whose Markov matrices disagree with the rank the config declares
// When the runtime is assembled
// Then assembly fails naming the tensor and the mismatched dimension
TEST_F(DSparkAssemblyTest, RejectsAHeadsSidecarWhoseMarkovMatricesAreTheWrongShape)
{
    // The load path validates rank, dtype and both dimensions of each matrix. Writing a rank that disagrees with
    // the config proves those checks run rather than being dead weight, and it is the failure a mismatched export
    // would actually produce.
    std::vector<rt::Tensor> heads;
    for (char const* name : {"markov_w1", "markov_w2"})
    {
        rt::Tensor matrix({static_cast<int64_t>(kVocabSize), kDSparkMarkovRank + 1}, rt::DeviceType::kGPU,
            trt_edgellm::DataType::kHALF, name);
        CUDA_CHECK(cudaMemsetAsync(
            matrix.rawPointer(), 0, static_cast<size_t>(matrix.getShape().volume()) * sizeof(half), mStream));
        heads.push_back(std::move(matrix));
    }
    CUDA_CHECK(cudaStreamSynchronize(mStream));
    ASSERT_TRUE(rt::safetensors::saveSafetensors(mModelDir / binding_names::kDSparkHeadsFileName, heads, mStream));

    auto artifacts = makeArtifacts();

    // Asserting the message, not merely that something threw: the assembly path has many ways to fail, and this
    // test is only meaningful if it fails on the column count.
    std::string message;
    try
    {
        auto runtime = makeRuntime(std::move(artifacts));
        FAIL() << "assembly accepted a heads sidecar whose markov matrices are the wrong shape";
    }
    catch (std::exception const& e)
    {
        message = e.what();
    }
    EXPECT_THAT(message, ::testing::HasSubstr("markov_w1"));
    EXPECT_THAT(message, ::testing::HasSubstr("column count mismatch"));
}

TEST_F(DSparkAssemblyTest, CompactsTheBatchWhenOneSlotFinishesAheadOfTheOther)
{
    // The third decoder through the same eviction. DSpark carries per-slot Markov-head state on top of its draft
    // cache, so it has the most to keep in step.
    expectSurvivorKeepsDecodingAfterCompaction(kDSparkBlockSize);
}

// Given a pending DSpark proposal for two requests, one of which finishes during prefill,
// When the surviving request consumes its pending proposal after batch compaction,
// Then its acceptance bounds remain aligned and it accepts the full matching block.
TEST_F(DSparkAssemblyTest, CompactsPendingAcceptanceBoundsAfterPrefillCompletion)
{
    using ::testing::_;
    auto baseEngine = makeEngine();
    auto draftEngine = makeEngine();
    auto& base = *baseEngine;
    auto& draft = *draftEngine;
    ON_CALL(draft, execute(_)).WillByDefault(emit(agreedLogitsRows(kAgreedToken)));
    auto artifacts = makeArtifacts(std::move(baseEngine), std::move(draftEngine));
    int32_t const eosId = static_cast<int32_t>(artifacts.tokenizer->getEosId());
    {
        ::testing::InSequence sequence;
        EXPECT_CALL(base, execute(_)).WillOnce(emit({eosId, kAgreedToken}));
        EXPECT_CALL(base, execute(_)).WillOnce(emit(agreedLogitsRows(kAgreedToken)));
    }
    EXPECT_CALL(draft, execute(_)).Times(1);
    rt::LLMInferenceRuntime runtime{std::move(artifacts), mModelDir.string(), "", {}, drafting(), mStream,
        rt::ContextCacheConfig{/*enabled=*/true, /*maxRecords=*/16}};
    auto request = makeGreedyRequest(kShortPrompt, kDSparkBlockSize);
    request.requests.push_back(makeGreedyRequest(kLongPrompt, kDSparkBlockSize).requests.front());
    rt::LLMGenerationResponse response;
    ASSERT_TRUE(runtime.handleRequest(request, response, mStream));
    ASSERT_EQ(response.outputIds.size(), 2U);
    EXPECT_THAT(response.outputIds[0], ::testing::ElementsAre(eosId));
    EXPECT_EQ(response.finishReasons[0], rt::FinishReason::kEndId);
    EXPECT_THAT(
        response.outputIds[1], ::testing::AllOf(::testing::SizeIs(kDSparkBlockSize), ::testing::Each(kAgreedToken)));
    EXPECT_EQ(response.finishReasons[1], rt::FinishReason::kLength);
}

// Given a runtime that has already compacted a pending greedy chain proposal once,
// When the next request batch again finishes one of its two requests during prefill,
// Then that compaction succeeds too and the survivor accepts the full matching block.
TEST_F(DSparkAssemblyTest, CompactsPendingProposalAgainOnTheNextRequest)
{
    using ::testing::_;
    auto baseEngine = makeEngine();
    auto draftEngine = makeEngine();
    auto& base = *baseEngine;
    auto& draft = *draftEngine;
    ON_CALL(draft, execute(_)).WillByDefault(emit(agreedLogitsRows(kAgreedToken)));
    auto artifacts = makeArtifacts(std::move(baseEngine), std::move(draftEngine));
    int32_t const eosId = static_cast<int32_t>(artifacts.tokenizer->getEosId());
    constexpr int32_t kRequests = 2;
    {
        ::testing::InSequence sequence;
        for (int32_t round = 0; round < kRequests; ++round)
        {
            EXPECT_CALL(base, execute(_)).WillOnce(emit({eosId, kAgreedToken}));
            EXPECT_CALL(base, execute(_)).WillOnce(emit(agreedLogitsRows(kAgreedToken)));
        }
    }
    EXPECT_CALL(draft, execute(_)).Times(kRequests);
    rt::LLMInferenceRuntime runtime{std::move(artifacts), mModelDir.string(), "", {}, drafting(), mStream,
        rt::ContextCacheConfig{/*enabled=*/true, /*maxRecords=*/16}};
    for (int32_t round = 0; round < kRequests; ++round)
    {
        SCOPED_TRACE(::testing::Message() << "request " << round);
        auto request = makeGreedyRequest(kShortPrompt, kDSparkBlockSize);
        request.requests.push_back(makeGreedyRequest(kLongPrompt, kDSparkBlockSize).requests.front());
        rt::LLMGenerationResponse response;
        ASSERT_TRUE(runtime.handleRequest(request, response, mStream));
        ASSERT_EQ(response.outputIds.size(), 2U);
        EXPECT_THAT(response.outputIds[0], ::testing::ElementsAre(eosId));
        EXPECT_THAT(response.outputIds[1],
            ::testing::AllOf(::testing::SizeIs(kDSparkBlockSize), ::testing::Each(kAgreedToken)));
    }
}

TEST_F(DSparkAssemblyTest, ReducedVocabularySamplingFallsBackBeforeDraftExecution)
{
    // Given a reduced-vocabulary DSpark draft, when a request uses stochastic sampling, then vanilla decoding is
    // selected before the draft engine can execute or mutate its cache state.
    constexpr int32_t kReducedVocabSize = static_cast<int32_t>(kVocabSize / 2);
    auto draftConfig = makeDSparkDraftConfig();
    draftConfig["reduced_vocab_size"] = kReducedVocabSize;
    std::ofstream(mModelDir / "draft_config.json") << draftConfig.dump(2);
    writeReducedDSparkSidecars(mModelDir, static_cast<int32_t>(kVocabSize), kReducedVocabSize, mStream);

    using ::testing::_;
    auto baseEngine = makeEngine();
    auto draftEngine = makeEngine();
    EXPECT_CALL(*draftEngine, execute(_)).Times(0);
    auto runtime = makeRuntime(makeArtifacts(std::move(baseEngine), std::move(draftEngine)));

    auto request = makeGreedyRequest("a", 2);
    request.temperature = 1.0F;
    request.topK = 2;
    rt::LLMGenerationResponse response;
    ASSERT_TRUE(runtime.handleRequest(request, response, mStream));
}

// Hybrid DSpark base-engine tree metadata ABI.
// --------------------------------------------------------------------------

constexpr int32_t kDSparkTreeBlockSize{7};
constexpr int32_t kDSparkTreeVerifySize{kDSparkTreeBlockSize + 1};

Json makeHybridDsparkBaseConfig()
{
    Json config = makeTinyVanillaConfig();
    config["spec_decode_type"] = "dspark";
    config["engine_role"] = "base";
    config["num_attention_layers"] = 1;
    config["num_linear_attn_layers"] = 1;
    config["recurrent_state_num_heads"] = 2;
    config["recurrent_state_head_dim"] = 64;
    config["recurrent_state_size"] = 64;
    config["conv_dim"] = 128;
    config["conv_kernel"] = 4;
    config["recurrent_state_dtype"] = "fp16";
    config["conv_state_dtype"] = "fp16";
    config["dspark_config"] = Json{{"block_size", kDSparkTreeBlockSize}, {"mask_token_id", 3},
        {"markov_head_type", "vanilla"}, {"markov_rank", 16}, {"enable_confidence_head", true},
        {"confidence_head_with_markov", true}, {"target_layer_ids", Json::array({0, 1})}};
    config["builder_config"]["spec_base"] = true;
    config["builder_config"]["max_spec_verify_tokens"] = kDSparkTreeVerifySize;
    sizeSpeculativeKvPool(config);
    return config;
}

Json makeHybridDsparkDraftConfig()
{
    Json config = makeTinyVanillaConfig();
    config["num_hidden_layers"] = 1;
    config["spec_decode_type"] = "dspark";
    config["engine_role"] = "draft";
    config["draft_vocab_size"] = kVocabSize;
    config["base_model_hidden_size"] = 256;
    config["dspark_config"] = Json{{"block_size", kDSparkTreeBlockSize}, {"mask_token_id", 3},
        {"markov_head_type", "vanilla"}, {"markov_rank", 16}, {"enable_confidence_head", true},
        {"confidence_head_with_markov", true}, {"target_layer_ids", Json::array({0, 1})}};
    config["builder_config"].erase("spec_base");
    config["builder_config"]["spec_draft"] = true;
    config["builder_config"]["max_spec_draft_tokens"] = kDSparkTreeBlockSize;
    sizeSpeculativeKvPool(config);
    return config;
}

rt::SpecDecodeDraftingConfig makeDsparkDrafting(bool useTree)
{
    rt::SpecDecodeDraftingConfig drafting{};
    drafting.draftTopK = useTree ? 4 : 1;
    drafting.proposalTokens = kDSparkTreeBlockSize;
    drafting.verifyTokens = kDSparkTreeVerifySize;
    return drafting;
}

class DSparkTreeEngineAbiTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        CUDA_CHECK(cudaStreamCreate(&mStream));
        mModelDir = makeScratchDir("dsparkTreeEngineAbiTests");
        std::ofstream(mModelDir / "base_config.json") << makeHybridDsparkBaseConfig().dump(2);
        std::ofstream(mModelDir / "draft_config.json") << makeHybridDsparkDraftConfig().dump(2);
        writeTokenizerFiles(mModelDir);
    }

    void TearDown() override
    {
        CUDA_CHECK(cudaStreamDestroy(mStream));
        std::filesystem::remove_all(mModelDir);
    }

    rt::DeploymentConfig makeDeployment(bool useTree) const
    {
        return rt::createDeploymentConfig(
            mModelDir / "base_config.json", mModelDir / "draft_config.json", makeDsparkDrafting(useTree));
    }

    std::unique_ptr<::testing::NiceMock<MockEngineExecutor>> makeBaseEngine(bool hasParent, bool hasDepth,
        trt_edgellm::DataType parentType = trt_edgellm::DataType::kINT32,
        trt_edgellm::DataType depthType = trt_edgellm::DataType::kINT32)
    {
        using ::testing::_;

        auto engine = std::make_unique<::testing::NiceMock<MockEngineExecutor>>();
        ON_CALL(*engine, hasIOTensor(_)).WillByDefault([hasParent, hasDepth](char const* name) {
            std::string const binding{name};
            return (binding == binding_names::kTreeParentIds && hasParent)
                || (binding == binding_names::kTreeDepths && hasDepth) || binding == binding_names::kSpecVerifyLayout;
        });
        ON_CALL(*engine, getBindingDataType(_)).WillByDefault([parentType, depthType](char const* name) {
            return std::string{name} == binding_names::kTreeParentIds ? parentType : depthType;
        });
        return engine;
    }

    void expectValidationError(
        rt::DeploymentConfig const& deployment, rt::EngineExecutor const& engine, std::string const& expected)
    {
        try
        {
            rt::validateRecurrentVerifyMetadataBindings(deployment, engine);
            FAIL() << "Expected DSpark tree engine ABI validation to fail";
        }
        catch (std::exception const& e)
        {
            EXPECT_THAT(e.what(), ::testing::HasSubstr(expected));
        }
    }

    cudaStream_t mStream{};
    std::filesystem::path mModelDir;
};

TEST_F(DSparkTreeEngineAbiTest, TreeRequiresBothMetadataBindings)
{
    auto const deployment = makeDeployment(/*useTree=*/true);
    auto const engine = makeBaseEngine(/*hasParent=*/false, /*hasDepth=*/false);
    expectValidationError(deployment, *engine, "must expose both INT32 tree metadata bindings");
}

TEST_F(DSparkTreeEngineAbiTest, RejectsEitherIncompleteMetadataPair)
{
    auto const deployment = makeDeployment(/*useTree=*/true);
    auto const parentOnly = makeBaseEngine(/*hasParent=*/true, /*hasDepth=*/false);
    auto const depthOnly = makeBaseEngine(/*hasParent=*/false, /*hasDepth=*/true);

    expectValidationError(deployment, *parentOnly, "must expose both INT32 tree metadata bindings");
    expectValidationError(deployment, *depthOnly, "must expose both INT32 tree metadata bindings");
}

TEST_F(DSparkTreeEngineAbiTest, RejectsNonInt32Metadata)
{
    auto const deployment = makeDeployment(/*useTree=*/true);
    auto const engine = makeBaseEngine(
        /*hasParent=*/true, /*hasDepth=*/true, trt_edgellm::DataType::kINT32, trt_edgellm::DataType::kHALF);
    expectValidationError(deployment, *engine, "tree metadata bindings must be INT32");
}

TEST_F(DSparkTreeEngineAbiTest, AcceptsSameEngineForTreeAndLinear)
{
    auto const treeDeployment = makeDeployment(/*useTree=*/true);
    auto const treeEngine = makeBaseEngine(/*hasParent=*/true, /*hasDepth=*/true);
    EXPECT_NO_THROW(rt::validateRecurrentVerifyMetadataBindings(treeDeployment, *treeEngine));

    auto const linearDeployment = makeDeployment(/*useTree=*/false);
    auto const linearEngine = makeBaseEngine(/*hasParent=*/true, /*hasDepth=*/true);
    EXPECT_NO_THROW(rt::validateRecurrentVerifyMetadataBindings(linearDeployment, *linearEngine));
}

TEST_F(DSparkTreeEngineAbiTest, SameVerifyEngineSupportsLinearRuntime)
{
    auto const deployment = makeDeployment(/*useTree=*/false);
    auto const engine = makeBaseEngine(/*hasParent=*/true, /*hasDepth=*/true);
    // Given a fixed recurrent ABI, when selecting chain at runtime, then the same engine is valid.
    EXPECT_NO_THROW(rt::validateRecurrentVerifyMetadataBindings(deployment, *engine));
}

TEST_F(DSparkTreeEngineAbiTest, InjectedArtifactsRunTheSameValidation)
{
    rt::ModelArtifacts artifacts;
    artifacts.deployment = makeDeployment(/*useTree=*/true);
    artifacts.baseExecutor = makeBaseEngine(/*hasParent=*/false, /*hasDepth=*/false);
    artifacts.tokenizer = std::make_unique<tokenizer::Tokenizer>();
    ASSERT_TRUE(artifacts.tokenizer->loadFromHF(mModelDir.string()));

    try
    {
        rt::LLMInferenceRuntime runtime{std::move(artifacts), mModelDir.string(), /*multimodalEngineDir=*/"", {},
            makeDsparkDrafting(/*useTree=*/true), mStream};
        FAIL() << "Expected injected DSpark artifacts to validate the base engine ABI";
    }
    catch (std::exception const& e)
    {
        EXPECT_THAT(e.what(), ::testing::HasSubstr("must expose both INT32 tree metadata bindings"));
    }
}

// Given a DSpark deployment whose draft and base verify rows genuinely agree on kAgreedToken (a real, unambiguous
// argmax winner, not the all-zero literal tie kDefaultRowToken depends on)
// When one round runs
// Then the draft runs exactly once and the whole block is accepted, the same shape as DFlash's cached-block draft
//
// This used to need three verify rounds to fill the same budget under an all-zero tie-break fixture. Switching to
// a genuine (non-tied) argmax agreement instead of a literal all-vocab tie collapsed that back to one round. That
// is strong evidence the old 3-round behavior was itself a tie-break-kernel collision (two independent top-1
// kernels disagreeing on which index wins an all-zero tie) rather than a genuine property of
// kernel::dsparkGreedyAccept.
TEST_F(DSparkAssemblyTest, SpendsOneDraftForwardPerRoundRatherThanAChain)
{
    using ::testing::_;

    auto baseEngine = makeEngine();
    auto draftEngine = makeEngine();
    auto& base = *baseEngine;
    auto& draft = *draftEngine;
    ON_CALL(base, execute(_)).WillByDefault(emit(agreedLogitsRows(kAgreedToken)));
    ON_CALL(draft, execute(_)).WillByDefault(emit(agreedLogitsRows(kAgreedToken)));

    EXPECT_CALL(draft, prepare(kPrefillProfile, _, _, _)).Times(1);
    EXPECT_CALL(draft, prepare(kDecodeProfile, _, _, _)).Times(0);
    EXPECT_CALL(base, prepare(kPrefillProfile, _, _, _)).Times(1);
    EXPECT_CALL(base, prepare(kDecodeProfile, _, _, _)).Times(1);

    auto artifacts = makeArtifacts(std::move(baseEngine), std::move(draftEngine));
    auto runtime = makeRuntime(std::move(artifacts));

    auto const request = makeGreedyRequest("a", kDSparkBlockSize);
    rt::LLMGenerationResponse response;
    ASSERT_TRUE(runtime.handleRequest(request, response, mStream));

    expectResponseCoversEverySlot(response, 1);
    ASSERT_EQ(response.outputIds.size(), 1U);
    EXPECT_THAT(
        response.outputIds[0], ::testing::AllOf(::testing::SizeIs(kDSparkBlockSize), ::testing::Each(kAgreedToken)))
        << "every draft position genuinely matches the base's own argmax, so the whole block should be accepted";
    EXPECT_EQ(response.finishReasons[0], rt::FinishReason::kLength);
}

// Given a draft config that does not set dspark_config.sample_from_anchor at all (a legacy-style config, the shape
// every DSpark deployment predating that flag has, e.g. Qwen3-8B)
// When the runtime is assembled
// Then the draft block is sized as if sample_from_anchor were true (anchor-inclusive: block length ==
// proposalLen, not proposalLen + 1)
//
// Regression lock for the config-contract concern raised on !1474 (adding Nemotron-3.5's DSpark support): the new
// draft model's block layout is selectable via sample_from_anchor, but older draft models never declared that
// field and were built assuming anchor-inclusive sampling. llmEngineConfig.cpp resolves the unset case to `true`
// specifically to preserve that legacy layout -- if a future change flipped the JSON default, every deployment
// that predates the flag would silently get a block one slot wider than the draft engine it's paired with was
// built for, with no config validation error to catch it.
TEST_F(DSparkAssemblyTest, DraftBlockIsAnchorInclusiveWhenSampleFromAnchorIsUnset)
{
    using ::testing::_;
    using ::testing::AllOf;
    using ::testing::Field;

    auto baseEngine = makeEngine();
    auto draftEngine = makeEngine();
    auto& draft = *draftEngine;

    // proposalLen (= verifyTokenCount - 1, non-tree DSpark) is the anchor-inclusive block width;
    // sample_from_anchor=false would need proposalLen + 1 instead.
    constexpr int64_t kAnchorInclusiveBlockLen{kDSparkBlockSize - 1};
    EXPECT_CALL(draft, prepare(kPrefillProfile, Field(&rt::InferenceDims::seqLen, kAnchorInclusiveBlockLen), _, _));

    auto artifacts = makeArtifacts(std::move(baseEngine), std::move(draftEngine));
    auto runtime = makeRuntime(std::move(artifacts));

    // maxGenerateLength must reach a full round (kDSparkBlockSize): a request satisfied entirely out of prefill's
    // own sampled token never runs decodeStep(), so the draft engine is never touched at all -- same gotcha noted
    // elsewhere in this session's fixtures (e.g. Gemma4MtpAssemblyTest's round-trip test).
    auto const request = makeGreedyRequest("a", kDSparkBlockSize);
    rt::LLMGenerationResponse response;
    ASSERT_TRUE(runtime.handleRequest(request, response, mStream));
}

} // namespace

// Given a DSpark deployment and two prompts
// When the second is seated into a running session while the first is mid-generation, or retired unseated
// Then each decodes exactly as it does alone: the seated draft state moved no token and no round
TEST_F(DSparkAssemblyTest, Seating)
{
    expectSeating();
}
