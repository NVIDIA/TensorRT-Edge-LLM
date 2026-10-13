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

// DFlash speculative decoding assembled around two substitute engines.

#include "substituteEngine.h"

using namespace trt_edgellm;
using namespace substitute_engine;

namespace
{
// --------------------------------------------------------------------------
// DFlash speculative decoding.
//
// The same contract again, but a different proposal shape: one draft forward
// emits a whole block, so createDeploymentConfig requires draftingStep == 1 and
// the chain of draft decode steps EAGLE and MTP run does not exist here. That
// difference is exactly what these tests pin.
// --------------------------------------------------------------------------

//! The draft emits this many tokens per forward. Linear DFlash also uses it as the base verify window, and
//! `createDeploymentConfig` overrides verifySize to match, so the two are one number here.
constexpr int32_t kDFlashBlockSize{4};

//! One forward per block, not one per token. `createDeploymentConfig` rejects anything else for DFlash.
constexpr int32_t kDFlashDraftingStep{1};

void addDFlashConfig(Json& config)
{
    Json dflash;
    // Both engines must name layers that exist in the base; the draft reads their hidden states.
    dflash["target_layer_ids"] = Json::array({0, 1});
    dflash["block_size"] = kDFlashBlockSize;
    // The default is a real Qwen3 id, far outside this deployment's 128-token vocabulary.
    dflash["mask_token_id"] = 3;
    config["dflash_config"] = dflash;
}

Json makeDFlashBaseConfig()
{
    Json config = makeSpecBaseConfig("dflash", kDFlashBlockSize);
    addDFlashConfig(config);
    return config;
}

Json makeDFlashDraftConfig()
{
    Json config = makeSpecDraftConfig("dflash", kDFlashBlockSize);
    addDFlashConfig(config);
    // Concatenated hidden states from the two target layers named above.
    config["base_model_hidden_size"] = 2 * config["hidden_size"].get<int32_t>();
    return config;
}

class DFlashAssemblyTest : public SpecAssemblyTest
{
protected:
    //! No draft vocab map is written. DFlash gates that sidecar on the draft config declaring reduced_vocab_size > 0,
    //! deliberately on the config rather than on file existence, so a full-vocab draft needs no file.
    std::filesystem::path stageModelDir() override
    {
        return writeSpecModelDir("dflashAssemblyTests", makeDFlashBaseConfig(), makeDFlashDraftConfig());
    }

    rt::SpecDecodeDraftingConfig drafting() const override
    {
        rt::SpecDecodeDraftingConfig config{};
        config.draftingTopK = 1;
        config.draftingStep = kDFlashDraftingStep;
        config.verifySize = kDFlashBlockSize;
        config.dflashBlockSize = kDFlashBlockSize;
        return config;
    }
};

TEST_F(DFlashAssemblyTest, AssemblesTwoEnginesWithoutAnyEngineFileOrSidecar)
{
    expectAssembledFromArtifactsAlone("dflash");
}

// Given a DFlash deployment, whose draft emits a whole block per forward
// When a single round runs
// Then the draft runs exactly once, unlike the chain the EAGLE and MTP decoders walk
TEST_F(DFlashAssemblyTest, SpendsOneDraftForwardPerRoundRatherThanAChain)
{
    using ::testing::_;

    auto baseEngine = makeEngine();
    auto draftEngine = makeEngine();
    auto& base = *baseEngine;
    auto& draft = *draftEngine;

    // This is where DFlash parts company with EAGLE and MTP. They spend `draftingStep` draft forwards walking a
    // chain; DFlash spends exactly one, because the draft emits the whole block at once. createDeploymentConfig
    // enforces draftingStep == 1 for precisely this reason, so a second draft forward here would mean the decoder
    // disagrees with the config that admitted it.
    EXPECT_CALL(draft, prepare(kPrefillProfile, _, _, _)).Times(1);
    EXPECT_CALL(draft, prepare(kDecodeProfile, _, _, _)).Times(0);
    EXPECT_CALL(base, prepare(kPrefillProfile, _, _, _)).Times(1);
    EXPECT_CALL(base, prepare(kDecodeProfile, _, _, _)).Times(1);

    auto artifacts = makeArtifacts(std::move(baseEngine), std::move(draftEngine));
    // Linear DFlash verifies one block per round, so the round's ceiling is the block size rather than
    // draftingStep + 1.
    auto const maxAcceptedPerRound = artifacts.deployment.maxAcceptedTokensPerRound();
    ASSERT_EQ(maxAcceptedPerRound, kDFlashBlockSize);

    auto runtime = makeRuntime(std::move(artifacts));

    // Two tokens: the prefill emits the first and any round emits at least one more, so the run is exactly one
    // speculative round whatever that round accepts. How many it accepts is the decoder's business and moves with
    // the verification policy; how many draft forwards it costs is the contract this test is here for.
    auto const request = makeGreedyRequest("a", 2);
    rt::LLMGenerationResponse response;
    ASSERT_TRUE(runtime.handleRequest(request, response, mStream));

    expectResponseCoversEverySlot(response, 1);
    ASSERT_EQ(response.outputIds.size(), 1U);
    EXPECT_FALSE(response.outputIds[0].empty());
}

TEST_F(DFlashAssemblyTest, CompactsTheBatchWhenOneSlotFinishesAheadOfTheOther)
{
    // The same eviction against a decoder that keeps a different shape of draft state: DFlash caches a block of
    // target K/V rather than a walked chain. Whatever the shape, it has to move with the mapping the base cache used.
    expectSurvivorKeepsDecodingAfterCompaction(kDFlashBlockSize);
}

// --------------------------------------------------------------------------
// DFlash2: production decoder assembly, sparse linear acceptance, compaction,
// and decoder-owned CUDA graphs.
// --------------------------------------------------------------------------

constexpr int32_t kDFlash2BlockSize{8};
constexpr int32_t kDFlash2SelectorTopK{16};
constexpr int32_t kDFlash2ProposalToken{10};
constexpr int32_t kDFlash2TargetToken{20};

Json makeDFlash2ConfigSection()
{
    return Json{{"version", 2}, {"block_size", kDFlash2BlockSize}, {"mask_token_id", 3}, {"is_causal", false},
        {"conv_kernel_size", 2}, {"conv_group_size", 16}, {"selector_rank", 256},
        {"selector_top_k", kDFlash2SelectorTopK}, {"supports_probabilistic_sampling", true},
        {"target_layer_ids", Json::array({0, 1, 2, 3, 4})}};
}

void addDFlash2Contract(Json& config)
{
    config["spec_decode_type"] = "dflash";
    config["dflash_config"] = makeDFlash2ConfigSection();
}

Json makeDFlash2BaseConfig()
{
    Json config = makeTinyVanillaConfig();
    config["num_hidden_layers"] = 5;
    addDFlash2Contract(config);
    config["engine_role"] = "base";
    config["builder_config"]["spec_base"] = true;
    config["builder_config"]["max_verify_tree_size"] = kDFlash2BlockSize;
    sizeSpeculativeKvPool(config);
    return config;
}

Json makeDFlash2DraftConfig()
{
    Json config = makeTinyVanillaConfig();
    config["num_hidden_layers"] = 5;
    addDFlash2Contract(config);
    config["engine_role"] = "draft";
    config["base_model_hidden_size"] = 128 * 5;
    config["builder_config"].erase("spec_base");
    config["builder_config"]["max_draft_tree_size"] = kDFlash2BlockSize;
    sizeSpeculativeKvPool(config);
    return config;
}

std::filesystem::path writeDFlash2ModelDir(cudaStream_t stream)
{
    auto const dir = makeScratchDir("dflash2DecoderTests");
    std::ofstream(dir / "base_config.json") << makeDFlash2BaseConfig().dump(2);
    std::ofstream(dir / "draft_config.json") << makeDFlash2DraftConfig().dump(2);
    writeTokenizerFiles(dir);
    std::vector<rt::Tensor> selectorTensors;
    selectorTensors.emplace_back(
        rt::Coords{kVocabSize, 256}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF, "predecessor_codebook");
    selectorTensors.emplace_back(
        rt::Coords{kVocabSize, 256}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF, "successor_codebook");
    for (auto& tensor : selectorTensors)
    {
        CUDA_CHECK(cudaMemsetAsync(tensor.rawPointer(), 0, tensor.getMemoryCapacity(), stream));
    }
    EXPECT_TRUE(
        rt::safetensors::saveSafetensors(dir / binding_names::kDFlash2SelectorFileName, selectorTensors, stream));
    return dir;
}

rt::SpecDecodeDraftingConfig makeDFlash2Drafting()
{
    rt::SpecDecodeDraftingConfig drafting{};
    drafting.draftingTopK = 1;
    drafting.draftingStep = 1;
    drafting.verifySize = kDFlash2BlockSize;
    return drafting;
}

rt::ModelArtifacts makeDFlash2Artifacts(std::filesystem::path const& modelDir,
    std::unique_ptr<rt::EngineExecutor> baseEngine, std::unique_ptr<rt::EngineExecutor> draftEngine,
    cudaStream_t stream)
{
    rt::ModelArtifacts artifacts;
    artifacts.deployment = rt::createDeploymentConfig(
        modelDir / "base_config.json", modelDir / "draft_config.json", makeDFlash2Drafting());
    artifacts.baseExecutor = std::move(baseEngine);
    artifacts.draftExecutor = std::move(draftEngine);

    artifacts.weights.load(modelDir, modelDir / "base_config.json", stream);
    artifacts.weights.validateAgainstEngine(*artifacts.baseExecutor, "base");
    artifacts.draftWeights.load(modelDir, modelDir / "draft_config.json", stream);
    artifacts.draftWeights.validateAgainstEngine(*artifacts.draftExecutor, "draft");

    artifacts.embedding.table = rt::Tensor({artifacts.deployment.base.vocabSize, artifacts.deployment.base.hiddenSize},
        rt::DeviceType::kGPU, nvinfer1::DataType::kHALF, "test::dflash2Embedding");
    CUDA_CHECK(cudaMemsetAsync(artifacts.embedding.table.rawPointer(), 0,
        static_cast<size_t>(artifacts.embedding.table.getShape().volume()) * sizeof(half), stream));

    artifacts.tokenizer = std::make_unique<tokenizer::Tokenizer>();
    if (!artifacts.tokenizer->loadFromHF(modelDir.string()))
    {
        throw std::runtime_error("DFlash2 test tokenizer failed to load");
    }
    return artifacts;
}

//! State deliberately kept outside the mock executor. The runtime owns and destroys the executor, while assertions
//! need to inspect every prepared/captured shape after handleRequest() returns.
struct DFlash2EngineTrace
{
    int32_t profile{-1};
    rt::InferenceDims dims{};
    rt::Tensor* logits{nullptr};
    rt::Tensor* proposalSupportIds{nullptr};
    rt::Tensor* proposalUnaryValues{nullptr};
    rt::Tensor* proposalProjectedHidden{nullptr};
    std::vector<std::pair<int32_t, rt::InferenceDims>> prepares;
    std::vector<rt::InferenceDims> captures;
    int32_t executions{0};
};

void writeDFlash2Proposals(DFlash2EngineTrace const& trace, cudaStream_t stream)
{
    ASSERT_NE(trace.proposalSupportIds, nullptr);
    ASSERT_NE(trace.proposalUnaryValues, nullptr);
    ASSERT_NE(trace.proposalProjectedHidden, nullptr);

    auto const supportElements = static_cast<size_t>(trace.proposalSupportIds->getShape().volume());
    std::vector<int32_t> supportIds(supportElements, kDFlash2ProposalToken + 1);
    std::vector<float> unaryValues(supportElements, 0.0F);
    for (size_t offset = 0; offset < supportElements; offset += kDFlash2SelectorTopK)
    {
        supportIds[offset] = kDFlash2ProposalToken;
        unaryValues[offset] = 1.0F;
    }
    CUDA_CHECK(cudaMemcpyAsync(trace.proposalSupportIds->rawPointer(), supportIds.data(),
        supportIds.size() * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(trace.proposalUnaryValues->rawPointer(), unaryValues.data(),
        unaryValues.size() * sizeof(float), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemsetAsync(
        trace.proposalProjectedHidden->rawPointer(), 0, trace.proposalProjectedHidden->getMemoryCapacity(), stream));
}

std::vector<int32_t> verificationPlan(int32_t acceptanceLength, int32_t replacementToken)
{
    EXPECT_GE(acceptanceLength, 1);
    EXPECT_LE(acceptanceLength, kDFlash2BlockSize);
    std::vector<int32_t> plan(kDFlash2BlockSize, replacementToken);
    std::fill_n(plan.begin(), acceptanceLength - 1, kDFlash2ProposalToken);
    return plan;
}

class DFlash2AssemblyTest : public ::testing::TestWithParam<int32_t>
{
protected:
    void SetUp() override
    {
        CUDA_CHECK(cudaStreamCreate(&mStream));
        mModelDir = writeDFlash2ModelDir(mStream);
    }

    void TearDown() override
    {
        CUDA_CHECK(cudaStreamDestroy(mStream));
        std::filesystem::remove_all(mModelDir);
    }

    std::unique_ptr<::testing::NiceMock<MockEngineExecutor>> makeBaseEngine(bool captureSucceeds = false)
    {
        using ::testing::_;
        using ::testing::Return;

        auto engine = std::make_unique<::testing::NiceMock<MockEngineExecutor>>();
        ON_CALL(*engine, getRequiredContextMemorySize()).WillByDefault(Return(4096));
        ON_CALL(*engine, setContextMemory(_)).WillByDefault(Return(true));
        ON_CALL(*engine, hasIOTensor(_)).WillByDefault(Return(false));
        ON_CALL(*engine, getBindingDataType(_)).WillByDefault(Return(nvinfer1::DataType::kHALF));
        ON_CALL(*engine, prepare(_, _, _, _))
            .WillByDefault(
                [this](int32_t profile, rt::InferenceDims const& dims, rt::TensorMap const& map, cudaStream_t) {
                    mBaseTrace.profile = profile;
                    mBaseTrace.dims = dims;
                    mBaseTrace.prepares.emplace_back(profile, dims);
                    mBaseTrace.logits = map.get(binding_names::kLogits);
                    return true;
                });
        ON_CALL(*engine, execute(_)).WillByDefault([this](cudaStream_t stream) {
            ++mBaseTrace.executions;
            if (mBaseTrace.profile == kPrefillProfile)
            {
                writeLogits(*mBaseTrace.logits, kVocabSize,
                    std::vector<int32_t>(static_cast<size_t>(mBaseTrace.dims.batch), kDFlash2TargetToken), stream);
                return true;
            }
            if (mVerificationPlans.empty())
            {
                ADD_FAILURE() << "DFlash2 base verification executed without a queued logits plan";
                return false;
            }
            auto plan = std::move(mVerificationPlans.front());
            mVerificationPlans.pop_front();
            writeLogits(*mBaseTrace.logits, kVocabSize, plan, stream);
            return true;
        });
        ON_CALL(*engine, captureGraph(_)).WillByDefault([this, captureSucceeds](cudaStream_t) {
            mBaseTrace.captures.push_back(mBaseTrace.dims);
            return captureSucceeds;
        });
        return engine;
    }

    std::unique_ptr<::testing::NiceMock<MockEngineExecutor>> makeDraftEngine(bool captureSucceeds = false)
    {
        using ::testing::_;
        using ::testing::Return;

        auto engine = std::make_unique<::testing::NiceMock<MockEngineExecutor>>();
        ON_CALL(*engine, getRequiredContextMemorySize()).WillByDefault(Return(4096));
        ON_CALL(*engine, setContextMemory(_)).WillByDefault(Return(true));
        ON_CALL(*engine, hasIOTensor(_)).WillByDefault(Return(false));
        ON_CALL(*engine, getBindingDataType(_)).WillByDefault(Return(nvinfer1::DataType::kHALF));
        ON_CALL(*engine, prepare(_, _, _, _))
            .WillByDefault(
                [this](int32_t profile, rt::InferenceDims const& dims, rt::TensorMap const& map, cudaStream_t) {
                    mDraftTrace.profile = profile;
                    mDraftTrace.dims = dims;
                    mDraftTrace.prepares.emplace_back(profile, dims);
                    mDraftTrace.proposalSupportIds = map.get(binding_names::kSpecProposalSupportIds);
                    mDraftTrace.proposalUnaryValues = map.get(binding_names::kSpecProposalUnaryValues);
                    mDraftTrace.proposalProjectedHidden = map.get(binding_names::kSpecProposalProjectedHidden);
                    return true;
                });
        ON_CALL(*engine, execute(_)).WillByDefault([this](cudaStream_t stream) {
            ++mDraftTrace.executions;
            writeDFlash2Proposals(mDraftTrace, stream);
            return true;
        });
        ON_CALL(*engine, captureGraph(_)).WillByDefault([this, captureSucceeds](cudaStream_t) {
            mDraftTrace.captures.push_back(mDraftTrace.dims);
            return captureSucceeds;
        });
        return engine;
    }

    std::unique_ptr<rt::LLMInferenceRuntime> makeRuntime(bool captureSucceeds = false)
    {
        auto artifacts = makeDFlash2Artifacts(
            mModelDir, makeBaseEngine(captureSucceeds), makeDraftEngine(captureSucceeds), mStream);
        return std::make_unique<rt::LLMInferenceRuntime>(std::move(artifacts), mModelDir.string(),
            /*multimodalEngineDir=*/"", std::unordered_map<std::string, std::string>{}, makeDFlash2Drafting(), mStream);
    }

    std::vector<rt::InferenceDims> draftDecodePrepares() const
    {
        std::vector<rt::InferenceDims> result;
        for (auto const& [profile, dims] : mDraftTrace.prepares)
        {
            if (profile == kDecodeProfile)
            {
                result.push_back(dims);
            }
        }
        return result;
    }

    cudaStream_t mStream{};
    std::filesystem::path mModelDir;
    DFlash2EngineTrace mBaseTrace;
    DFlash2EngineTrace mDraftTrace;
    std::deque<std::vector<int32_t>> mVerificationPlans;
};

TEST_P(DFlash2AssemblyTest, PropagatesAcceptanceLengthIntoTheNextDraftDelta)
{
    int32_t const acceptanceLength = GetParam();
    mVerificationPlans.push_back(verificationPlan(acceptanceLength, kDFlash2TargetToken));
    mVerificationPlans.push_back(verificationPlan(/*acceptanceLength=*/1, /*replacementToken=*/1));

    auto runtime = makeRuntime();
    EXPECT_TRUE(runtime->hasDraftModel());
    EXPECT_STREQ(runtime->getSpeculativeDecodingStrategyName(), "dflash");

    auto const request = makeGreedyRequest("a", /*maxGenerateLength=*/20);
    rt::LLMGenerationResponse response;
    ASSERT_TRUE(runtime->handleRequest(request, response, mStream));

    auto const draftDecode = draftDecodePrepares();
    ASSERT_EQ(draftDecode.size(), 1U);
    EXPECT_EQ(draftDecode[0].batch, 1);
    EXPECT_EQ(draftDecode[0].seqLen, kDFlash2BlockSize);
    EXPECT_EQ(draftDecode[0].selectLen, acceptanceLength);
    ASSERT_EQ(response.outputIds.size(), 1U);
    EXPECT_EQ(response.outputIds[0].size(), static_cast<size_t>(acceptanceLength + 2));
    EXPECT_EQ(response.outputIds[0].back(), 1);
    EXPECT_EQ(response.finishReasons[0], rt::FinishReason::kEndId);
}

INSTANTIATE_TEST_SUITE_P(AcceptanceLengths, DFlash2AssemblyTest, ::testing::Values(1, 4, 8));

TEST_F(DFlash2AssemblyTest, ClampsAcceptedBlockToTheRemainingGenerationBudget)
{
    mVerificationPlans.push_back(verificationPlan(/*acceptanceLength=*/8, kDFlash2TargetToken));
    auto runtime = makeRuntime();

    auto const request = makeGreedyRequest("a", /*maxGenerateLength=*/3);
    rt::LLMGenerationResponse response;
    ASSERT_TRUE(runtime->handleRequest(request, response, mStream));

    ASSERT_EQ(response.outputIds.size(), 1U);
    EXPECT_EQ(response.outputIds[0],
        (std::vector<int32_t>{kDFlash2TargetToken, kDFlash2ProposalToken, kDFlash2ProposalToken}));
    EXPECT_EQ(response.finishReasons[0], rt::FinishReason::kLength);
}

TEST_F(DFlash2AssemblyTest, StopsOnEosInsideAnAcceptedBlock)
{
    constexpr int32_t kEosAcceptanceLength{4};
    mVerificationPlans.push_back(verificationPlan(kEosAcceptanceLength, /*replacementToken=*/1));
    auto runtime = makeRuntime();

    auto const request = makeGreedyRequest("a", /*maxGenerateLength=*/20);
    rt::LLMGenerationResponse response;
    ASSERT_TRUE(runtime->handleRequest(request, response, mStream));

    EXPECT_TRUE(draftDecodePrepares().empty());
    ASSERT_EQ(response.outputIds.size(), 1U);
    EXPECT_EQ(response.outputIds[0],
        (std::vector<int32_t>{
            kDFlash2TargetToken, kDFlash2ProposalToken, kDFlash2ProposalToken, kDFlash2ProposalToken, 1}));
    EXPECT_EQ(response.finishReasons[0], rt::FinishReason::kEndId);
}

TEST_F(DFlash2AssemblyTest, BatchCompactionPreservesTheSurvivorsAcceptedState)
{
    std::vector<int32_t> firstBatchPlan(2 * kDFlash2BlockSize, kDFlash2TargetToken);
    // Slot 0 exits immediately. Slot 1 accepts three proposals and one target replacement.
    firstBatchPlan[0] = 1;
    std::fill_n(firstBatchPlan.begin() + kDFlash2BlockSize, 3, kDFlash2ProposalToken);
    mVerificationPlans.push_back(std::move(firstBatchPlan));
    mVerificationPlans.push_back(verificationPlan(/*acceptanceLength=*/1, /*replacementToken=*/1));
    auto runtime = makeRuntime();

    auto request = makeGreedyRequest("a", /*maxGenerateLength=*/20);
    request.requests.push_back(request.requests.front());
    rt::LLMGenerationResponse response;
    ASSERT_TRUE(runtime->handleRequest(request, response, mStream));

    auto const draftDecode = draftDecodePrepares();
    ASSERT_EQ(draftDecode.size(), 1U);
    EXPECT_EQ(draftDecode[0].batch, 1);
    EXPECT_EQ(draftDecode[0].selectLen, 4);
    ASSERT_EQ(response.outputIds.size(), 2U);
    EXPECT_EQ(response.outputIds[0], (std::vector<int32_t>{kDFlash2TargetToken, 1}));
    EXPECT_EQ(response.outputIds[1].size(), 6U);
    EXPECT_EQ(response.outputIds[1].back(), 1);
    EXPECT_EQ(response.finishReasons[0], rt::FinishReason::kEndId);
    EXPECT_EQ(response.finishReasons[1], rt::FinishReason::kEndId);
}

TEST_F(DFlash2AssemblyTest, CapturesOwnedShapeMatrixThenRunsNormally)
{
    auto runtime = makeRuntime(/*captureSucceeds=*/true);
    ASSERT_TRUE(runtime->captureDecodingCUDAGraph(mStream));

    std::set<std::pair<int64_t, int64_t>> draftShapes;
    for (auto const& dims : mDraftTrace.captures)
    {
        EXPECT_EQ(dims.seqLen, dims.batch * kDFlash2BlockSize);
        draftShapes.emplace(dims.batch, dims.selectLen);
    }
    std::set<std::pair<int64_t, int64_t>> expectedDraftShapes;
    for (int64_t batch = 1; batch <= kMaxBatchSize; ++batch)
    {
        for (int64_t delta = 1; delta <= kDFlash2BlockSize; ++delta)
        {
            expectedDraftShapes.emplace(batch, batch * delta);
        }
    }
    EXPECT_EQ(draftShapes, expectedDraftShapes);
    EXPECT_EQ(mDraftTrace.captures.size(), expectedDraftShapes.size());

    ASSERT_EQ(mBaseTrace.captures.size(), static_cast<size_t>(kMaxBatchSize));
    for (int64_t batch = 1; batch <= kMaxBatchSize; ++batch)
    {
        auto const& dims = mBaseTrace.captures[static_cast<size_t>(batch - 1)];
        EXPECT_EQ(dims.batch, batch);
        EXPECT_EQ(dims.seqLen, batch * kDFlash2BlockSize);
        EXPECT_EQ(dims.selectLen, batch * kDFlash2BlockSize);
    }

    // Capture mutates binding shapes and simulated cache lengths. A real request afterwards checks teardown restored
    // both engines to executable state rather than merely reporting successful capture calls.
    mVerificationPlans.push_back(verificationPlan(/*acceptanceLength=*/1, /*replacementToken=*/1));
    auto const request = makeGreedyRequest("a", /*maxGenerateLength=*/20);
    rt::LLMGenerationResponse response;
    ASSERT_TRUE(runtime->handleRequest(request, response, mStream));
    ASSERT_EQ(response.outputIds.size(), 1U);
    EXPECT_EQ(response.outputIds[0], (std::vector<int32_t>{kDFlash2TargetToken, 1}));
    EXPECT_EQ(response.finishReasons[0], rt::FinishReason::kEndId);
    EXPECT_GT(mBaseTrace.executions, 0);
    EXPECT_GT(mDraftTrace.executions, 0);
}

// --------------------------------------------------------------------------

} // namespace
