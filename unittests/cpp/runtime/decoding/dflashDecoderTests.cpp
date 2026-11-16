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

#include "specSessionSupport.h"
#include "substituteEngine.h"

#include "common/executionPhase.h"
#include "common/logger.h"

#include <deque>
#include <functional>
#include <string>
#include <vector>

using namespace trt_edgellm;
using namespace substitute_engine;

namespace
{
// --------------------------------------------------------------------------
// DFlash speculative decoding.
//
// The same contract again, but a different proposal shape: one draft forward
// emits a whole block, so createDeploymentConfig requires draftSteps == 1 and
// the chain of draft decode steps EAGLE and MTP run does not exist here. That
// difference is exactly what these tests pin.
// --------------------------------------------------------------------------

//! The draft emits this many tokens per forward. Linear DFlash also uses it as the base verify window, and
//! `createDeploymentConfig` overrides verifyTokenCount to match, so the two are one number here.
constexpr int32_t kDFlashBlockSize{4};

//! One forward per block, not one per token. `createDeploymentConfig` rejects anything else for DFlash.

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

class DFlashAssemblyTest : public SpecSessionAssemblyTest
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
        config.draftTopK = 1;
        config.proposalTokens = kDFlashBlockSize - 1;
        config.verifyTokens = kDFlashBlockSize;
        return config;
    }
};

TEST_F(DFlashAssemblyTest, AssemblesTwoEnginesWithoutAnyEngineFileOrSidecar)
{
    expectAssembledFromArtifactsAlone("dflash");
}

TEST_F(DFlashAssemblyTest, NonGreedyChainRequestsUseVanilla)
{
    // Given a V1 chain deployment, when sampling is requested, then only the base engine executes.
    auto baseEngine = makeEngine();
    auto draftEngine = makeEngine();
    EXPECT_CALL(*draftEngine, execute(::testing::_)).Times(0);
    EXPECT_CALL(*baseEngine, execute(::testing::_)).Times(::testing::AtLeast(2));
    auto runtime = makeRuntime(makeArtifacts(std::move(baseEngine), std::move(draftEngine)));
    auto request = makeGreedyRequest("a", 3);
    request.temperature = 1.0F;
    request.topK = 2;
    rt::LLMGenerationResponse response;
    ASSERT_TRUE(runtime.handleRequest(request, response, mStream));
    expectResponseCoversEverySlot(response, 1);
}

TEST_F(DFlashAssemblyTest, VerificationDecodesPackedEmbeddingDescriptor)
{
    // Given packed NVFP4 embeddings, when a speculative round verifies tokens, then the base receives decoded rows.
    auto baseEngine = makeEngine();
    EXPECT_CALL(*baseEngine, prepare(kPrefillProfile, ::testing::_, ::testing::_, ::testing::_)).Times(1);
    EXPECT_CALL(*baseEngine, prepare(kDecodeProfile, ::testing::_, ::testing::_, ::testing::_))
        .WillOnce([this](int32_t, rt::InferenceDims const&, rt::TensorMap const& map, cudaStream_t stream) {
            auto const* embeddings = map.get(binding_names::kInputsEmbeds);
            std::vector<half> values(embeddings->getShape().volume());
            CUDA_CHECK(cudaMemcpyAsync(
                values.data(), embeddings->rawPointer(), values.size() * sizeof(half), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            EXPECT_TRUE(std::all_of(values.begin(), values.end(), [](half value) { return float(value) == 3.0F; }));
            mLogits = map.get(binding_names::kLogits);
            return true;
        });
    auto artifacts = makeArtifacts(std::move(baseEngine), makeEngine());
    auto& embedding = artifacts.embedding;
    int64_t const vocab = artifacts.deployment.base.vocabSize;
    int64_t const hidden = artifacts.deployment.base.hiddenSize;
    embedding.table = rt::Tensor({vocab, hidden / 2}, rt::DeviceType::kGPU, trt_edgellm::DataType::kUINT8);
    embedding.tableScalingFactor = rt::Tensor({vocab, hidden / 16}, rt::DeviceType::kGPU, trt_edgellm::DataType::kFP8);
    CUDA_CHECK(cudaMemsetAsync(embedding.table.rawPointer(), 0x22, vocab * hidden / 2, mStream));
    CUDA_CHECK(cudaMemsetAsync(embedding.tableScalingFactor.rawPointer(), 0x38, vocab * hidden / 16, mStream));
    embedding.descriptor.format = kernel::EmbeddingTableFormat::kNVFP4;
    embedding.descriptor.logicalShape = {vocab, hidden};
    embedding.descriptor.packedShape = {vocab, hidden / 2};
    embedding.descriptor.packedDataType = trt_edgellm::DataType::kUINT8;
    embedding.descriptor.groupSize = 16;
    embedding.descriptor.scaleShape = {vocab, hidden / 16};
    embedding.descriptor.scaleDataType = trt_edgellm::DataType::kFP8;
    embedding.descriptor.multiplierGlobalScale = 2.0F;
    embedding.descriptor.modelOutputScale = 1.5F;
    auto runtime = makeRuntime(std::move(artifacts));
    rt::LLMGenerationResponse response;
    ASSERT_TRUE(runtime.handleRequest(makeGreedyRequest("a", 2), response, mStream));
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

    // This is where DFlash parts company with EAGLE and MTP. They spend `draftSteps` draft forwards walking a
    // chain; DFlash spends exactly one, because the draft emits the whole block at once. createDeploymentConfig
    // enforces draftSteps == 1 for precisely this reason, so a second draft forward here would mean the decoder
    // disagrees with the config that admitted it.
    EXPECT_CALL(draft, prepare(kPrefillProfile, _, _, _)).Times(1);
    EXPECT_CALL(draft, prepare(kDecodeProfile, _, _, _)).Times(0);
    EXPECT_CALL(base, prepare(kPrefillProfile, _, _, _)).Times(1);
    EXPECT_CALL(base, prepare(kDecodeProfile, _, _, _)).Times(1);

    auto artifacts = makeArtifacts(std::move(baseEngine), std::move(draftEngine));
    // Linear DFlash verifies one block per round, so the round's ceiling is the block size rather than
    // draftSteps + 1.
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

// Given a DFlash deployment whose draft and base verify rows genuinely agree on kAgreedToken (a real, unambiguous
// argmax winner, not the all-zero literal tie kDefaultRowToken depends on)
// When one round runs
// Then the whole block is accepted -- complementary to SpendsOneDraftForwardPerRoundRatherThanAChain above, which
// deliberately does not claim full acceptance to avoid depending on the tie-break rule
TEST_F(DFlashAssemblyTest, AcceptsTheWholeBlockInOneRoundWhenTheBaseGenuinelyAgrees)
{
    using ::testing::_;

    auto baseEngine = makeEngine();
    auto draftEngine = makeEngine();
    auto& base = *baseEngine;
    auto& draft = *draftEngine;
    ON_CALL(base, execute(_)).WillByDefault(emit(agreedLogitsRows(kAgreedToken)));
    ON_CALL(draft, execute(_)).WillByDefault(emit(agreedLogitsRows(kAgreedToken)));

    auto runtime = makeRuntime(makeArtifacts(std::move(baseEngine), std::move(draftEngine)));

    auto const request = makeGreedyRequest("a", kDFlashBlockSize);
    rt::LLMGenerationResponse response;
    ASSERT_TRUE(runtime.handleRequest(request, response, mStream));

    expectResponseCoversEverySlot(response, 1);
    ASSERT_EQ(response.outputIds.size(), 1U);
    EXPECT_THAT(
        response.outputIds[0], ::testing::AllOf(::testing::SizeIs(kDFlashBlockSize), ::testing::Each(kAgreedToken)))
        << "every draft position genuinely matches the base's own argmax, so the whole block should be accepted";
    EXPECT_EQ(response.finishReasons[0], rt::FinishReason::kLength);
    // The identical-output assertion above can't tell "one round, fully accepted" apart from several
    // reject-and-retry rounds that happen to converge on the same final tokens (every row decodes to the same
    // kAgreedToken either way). specVerifyCounts/specAcceptanceLengths are the production-facing counters that
    // actually distinguish them.
    ASSERT_EQ(response.specVerifyCounts.size(), 1U);
    EXPECT_EQ(response.specVerifyCounts[0], 1) << "the whole block should be accepted in a single verify round";
    ASSERT_EQ(response.specAcceptanceLengths.size(), 1U);
    EXPECT_FLOAT_EQ(response.specAcceptanceLengths[0], static_cast<float>(kDFlashBlockSize));
}

// Given a DFlash V1 deployment whose draft and base agree on every token, and whose base verification writes a
// distinct value into every hidden-state row it produces
// When a round accepts the whole 4-token block and the draft runs its next forward
// Then that forward's target-hidden input holds the 4 accepted base rows in order
TEST_F(DFlashAssemblyTest, DraftReadsTheAcceptedBaseHiddenRows)
{
    using ::testing::_;

    auto baseEngine = makeEngine();
    auto draftEngine = makeEngine();
    int64_t const hidden = makeDFlashDraftConfig()["base_model_hidden_size"].get<int64_t>();
    int32_t baseProfile{-1};
    int64_t baseRows{0};
    rt::Tensor* baseHidden{nullptr};
    int32_t verifyRounds{0};
    int32_t draftProfile{-1};
    rt::Tensor const* draftHidden{nullptr};
    std::vector<std::vector<float>> reads;
    auto const agreed = emit(agreedLogitsRows(kAgreedToken));
    auto const prepare = prepareAction(FakeOutputScale::kLogits);

    ON_CALL(*baseEngine, prepare(_, _, _, _))
        .WillByDefault([&, prepare](int32_t profile, rt::InferenceDims const& dims, rt::TensorMap const& map,
                           cudaStream_t stream) {
            baseProfile = profile;
            baseRows = dims.seqLen;
            baseHidden = map.get(binding_names::kOutputHiddenStates);
            return prepare(profile, dims, map, stream);
        });
    ON_CALL(*baseEngine, execute(_)).WillByDefault([&, agreed](cudaStream_t stream) {
        if (baseProfile == kDecodeProfile)
        {
            EXPECT_NE(baseHidden, nullptr);
            std::vector<half> values(static_cast<size_t>(baseRows * hidden));
            for (int64_t row = 0; row < baseRows; ++row)
            {
                std::fill_n(values.begin() + row * hidden, hidden,
                    __float2half(static_cast<float>(100 * (verifyRounds + 1) + row)));
            }
            CUDA_CHECK(cudaMemcpyAsync(
                baseHidden->rawPointer(), values.data(), values.size() * sizeof(half), cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            ++verifyRounds;
        }
        return agreed(stream);
    });
    ON_CALL(*draftEngine, prepare(_, _, _, _))
        .WillByDefault([&, prepare](int32_t profile, rt::InferenceDims const& dims, rt::TensorMap const& map,
                           cudaStream_t stream) {
            draftProfile = profile;
            draftHidden = map.get(binding_names::kDFlashTargetHiddenConcat);
            return prepare(profile, dims, map, stream);
        });
    ON_CALL(*draftEngine, execute(_)).WillByDefault([&, agreed](cudaStream_t stream) {
        if (draftProfile == kDecodeProfile && draftHidden != nullptr)
        {
            int64_t const rows = draftHidden->getShape().volume() / hidden;
            std::vector<half> values(static_cast<size_t>(rows * hidden));
            CUDA_CHECK(cudaMemcpyAsync(values.data(), draftHidden->rawPointer(), values.size() * sizeof(half),
                cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            std::vector<float> firstColumn(static_cast<size_t>(rows));
            for (int64_t row = 0; row < rows; ++row)
            {
                firstColumn[static_cast<size_t>(row)] = __half2float(values[static_cast<size_t>(row * hidden)]);
            }
            reads.push_back(std::move(firstColumn));
        }
        return agreed(stream);
    });

    auto runtime = makeRuntime(makeArtifacts(std::move(baseEngine), std::move(draftEngine)));
    auto const request = makeGreedyRequest("a", 2 * kDFlashBlockSize + 1);
    rt::LLMGenerationResponse response;
    ASSERT_TRUE(runtime.handleRequest(request, response, mStream));

    ASSERT_FALSE(reads.empty()) << "the draft never ran a decode forward after a verification round";
    ASSERT_EQ(reads.front().size(), static_cast<size_t>(kDFlashBlockSize));
    for (int32_t row = 0; row < kDFlashBlockSize; ++row)
    {
        EXPECT_EQ(reads.front()[static_cast<size_t>(row)], static_cast<float>(100 + row)) << "accepted row " << row;
    }
}

// --------------------------------------------------------------------------
// JetSpec: shares the DFlashDecoder C++ class with DFlash (both are "cached block draft" modes, see
// dflash_utils::isCachedBlockDraftMode), but takes a materially different config path -- spec_decode_type must be
// "jetspec" (not "dflash"), the sidecar config block is jetspec_config (dflash_config is only a fallback name), and
// makeCachedBlockDraftRuntimeConfig() only sets ProposalAttentionPolicy::kCausal (instead of kBidirectional) when
// userMode == SpecDecodeMode::kJetSpec specifically. Causal attention itself is already covered at the config layer
// by DeploymentConfigTest.JetSpecDDTreeRuntimeConfigIsCausal; what these tests cover is the layer on top of that --
// whether the whole runtime actually ends up with a JetSpec decoder, and whether its round-trip call shape matches
// DFlash's (JetSpec and DFlash share decodeStep() byte for byte; only the mask content differs, which a call-shape
// observation does not inspect).
// --------------------------------------------------------------------------

constexpr int32_t kJetSpecBlockSize{4};

void addJetSpecConfig(Json& config)
{
    Json jetspec;
    jetspec["target_layer_ids"] = Json::array({0});
    jetspec["block_size"] = kJetSpecBlockSize;
    // The default (151669) is a real Qwen id, outside this deployment's 128-token vocabulary.
    jetspec["mask_token_id"] = 0;
    jetspec["causal_head"] = true;
    config["jetspec_config"] = jetspec;
}

Json makeJetSpecBaseConfig()
{
    Json config = makeSpecBaseConfig("jetspec", kJetSpecBlockSize);
    addJetSpecConfig(config);
    return config;
}

Json makeJetSpecDraftConfig()
{
    Json config = makeSpecDraftConfig("jetspec", kJetSpecBlockSize);
    addJetSpecConfig(config);
    config["base_model_hidden_size"] = config["hidden_size"];
    return config;
}

class JetSpecAssemblyTest : public SpecAssemblyTest
{
protected:
    std::filesystem::path stageModelDir() override
    {
        return writeSpecModelDir("jetSpecAssemblyTests", makeJetSpecBaseConfig(), makeJetSpecDraftConfig());
    }

    rt::SpecDecodeDraftingConfig drafting() const override
    {
        rt::SpecDecodeDraftingConfig config{};
        config.draftTopK = 1;
        config.proposalTokens = kJetSpecBlockSize - 1;
        return config;
    }
};

// Given a deployment configured with spec_decode_type=jetspec (not dflash) on both base and draft
// When the runtime assembles
// Then the decoder reports its own identity as "jetspec", not "dflash" -- proving DecoderRegistry/DFlashDecoder
// actually route on SpecDecodeMode::kJetSpec rather than silently collapsing to the config-compatible DFlash path
TEST_F(JetSpecAssemblyTest, AssemblesWithJetSpecIdentityNotDFlash)
{
    expectAssembledFromArtifactsAlone("jetspec");
}

TEST_F(JetSpecAssemblyTest, NonGreedyChainRequestsUseVanilla)
{
    // Given a causal V1 chain deployment, when sampling is requested, then only the base engine executes.
    auto baseEngine = makeEngine();
    auto draftEngine = makeEngine();
    EXPECT_CALL(*draftEngine, execute(::testing::_)).Times(0);
    EXPECT_CALL(*baseEngine, execute(::testing::_)).Times(::testing::AtLeast(2));
    auto runtime = makeRuntime(makeArtifacts(std::move(baseEngine), std::move(draftEngine)));
    auto request = makeGreedyRequest("a", 3);
    request.temperature = 1.0F;
    request.topK = 2;
    rt::LLMGenerationResponse response;
    ASSERT_TRUE(runtime.handleRequest(request, response, mStream));
    expectResponseCoversEverySlot(response, 1);
}

// Same happy-path shape as DFlashAssemblyTest.AcceptsTheWholeBlockInOneRoundWhenTheBaseGenuinelyAgrees above:
// JetSpec shares DFlashDecoder's decodeStep() control flow byte for byte, only the proposal attention mask
// content differs, which this call-shape observation does not inspect.
TEST_F(JetSpecAssemblyTest, AcceptsTheWholeBlockInOneRoundWhenTheBaseGenuinelyAgrees)
{
    using ::testing::_;

    auto baseEngine = makeEngine();
    auto draftEngine = makeEngine();
    auto& base = *baseEngine;
    auto& draft = *draftEngine;
    ON_CALL(base, execute(_)).WillByDefault(emit(agreedLogitsRows(kAgreedToken)));
    ON_CALL(draft, execute(_)).WillByDefault(emit(agreedLogitsRows(kAgreedToken)));

    auto runtime = makeRuntime(makeArtifacts(std::move(baseEngine), std::move(draftEngine)));

    auto const request = makeGreedyRequest("a", kJetSpecBlockSize);
    rt::LLMGenerationResponse response;
    ASSERT_TRUE(runtime.handleRequest(request, response, mStream));

    expectResponseCoversEverySlot(response, 1);
    ASSERT_EQ(response.outputIds.size(), 1U);
    EXPECT_THAT(
        response.outputIds[0], ::testing::AllOf(::testing::SizeIs(kJetSpecBlockSize), ::testing::Each(kAgreedToken)))
        << "every draft position genuinely matches the base's own argmax, so the whole block should be accepted";
    EXPECT_EQ(response.finishReasons[0], rt::FinishReason::kLength);
    // See DFlashAssemblyTest's sibling test: the identical-output assertion above can't distinguish one round
    // from several reject-and-retry rounds that converge on the same final tokens.
    ASSERT_EQ(response.specVerifyCounts.size(), 1U);
    EXPECT_EQ(response.specVerifyCounts[0], 1) << "the whole block should be accepted in a single verify round";
    ASSERT_EQ(response.specAcceptanceLengths.size(), 1U);
    EXPECT_FLOAT_EQ(response.specAcceptanceLengths[0], static_cast<float>(kJetSpecBlockSize));
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

Json makeDFlash2BaseConfig(int32_t maxSpecVerifyTokens = kDFlash2BlockSize)
{
    Json config = makeTinyVanillaConfig();
    config["num_hidden_layers"] = 5;
    addDFlash2Contract(config);
    config["engine_role"] = "base";
    config["builder_config"]["spec_base"] = true;
    config["builder_config"]["max_spec_verify_tokens"] = maxSpecVerifyTokens;
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
    config["builder_config"]["max_spec_draft_tokens"] = kDFlash2BlockSize;
    sizeSpeculativeKvPool(config);
    return config;
}

std::filesystem::path writeDFlash2ModelDir(cudaStream_t stream, int32_t maxSpecVerifyTokens = kDFlash2BlockSize)
{
    auto const dir = makeScratchDir("dflash2DecoderTests");
    std::ofstream(dir / "base_config.json") << makeDFlash2BaseConfig(maxSpecVerifyTokens).dump(2);
    std::ofstream(dir / "draft_config.json") << makeDFlash2DraftConfig().dump(2);
    writeTokenizerFiles(dir);
    std::vector<rt::Tensor> selectorTensors;
    selectorTensors.emplace_back(
        rt::Coords{kVocabSize, 256}, rt::DeviceType::kGPU, trt_edgellm::DataType::kHALF, "predecessor_codebook");
    selectorTensors.emplace_back(
        rt::Coords{kVocabSize, 256}, rt::DeviceType::kGPU, trt_edgellm::DataType::kHALF, "successor_codebook");
    for (auto& tensor : selectorTensors)
    {
        CUDA_CHECK(cudaMemsetAsync(tensor.rawPointer(), 0, tensor.getMemoryCapacity(), stream));
    }
    EXPECT_TRUE(
        rt::safetensors::saveSafetensors(dir / binding_names::kDFlash2SelectorFileName, selectorTensors, stream));
    return dir;
}

rt::SpecDecodeDraftingConfig makeDFlash2Drafting(int32_t draftTopK = 1, int32_t verifyTokens = kDFlash2BlockSize)
{
    rt::SpecDecodeDraftingConfig drafting{};
    drafting.draftTopK = draftTopK;
    drafting.proposalTokens = kDFlash2BlockSize - 1;
    drafting.verifyTokens = verifyTokens;
    return drafting;
}

rt::ModelArtifacts makeDFlash2Artifacts(std::filesystem::path const& modelDir,
    std::unique_ptr<rt::EngineExecutor> baseEngine, std::unique_ptr<rt::EngineExecutor> draftEngine,
    rt::SpecDecodeDraftingConfig const& drafting, cudaStream_t stream)
{
    rt::ModelArtifacts artifacts;
    artifacts.deployment
        = rt::createDeploymentConfig(modelDir / "base_config.json", modelDir / "draft_config.json", drafting);
    artifacts.baseExecutor = std::move(baseEngine);
    artifacts.draftExecutor = std::move(draftEngine);

    artifacts.weights.load(modelDir, modelDir / "base_config.json", stream);
    artifacts.weights.validateAgainstEngine(*artifacts.baseExecutor, "base");
    artifacts.draftWeights.load(modelDir, modelDir / "draft_config.json", stream);
    artifacts.draftWeights.validateAgainstEngine(*artifacts.draftExecutor, "draft");

    artifacts.embedding.table = rt::Tensor({artifacts.deployment.base.vocabSize, artifacts.deployment.base.hiddenSize},
        rt::DeviceType::kGPU, trt_edgellm::DataType::kHALF, "test::dflash2Embedding");
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

//! The draft's target-hidden input as EngineExecutor snapshots it: a captured graph replays only when a later forward
//! binds the same address and shape.
struct TargetHiddenBinding
{
    int32_t profile;
    int64_t batch;
    void const* address;
    rt::Coords shape;
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
        mModelDir = writeModelDir();
    }

    void TearDown() override
    {
        CUDA_CHECK(cudaStreamDestroy(mStream));
        // A parameterized test name ("Name/0") nests the scratch directory one level below the temp directory.
        auto const temp = std::filesystem::temp_directory_path();
        auto const relative = mModelDir.lexically_relative(temp);
        bool const underTemp = !relative.empty() && *relative.begin() != ".." && *relative.begin() != ".";
        std::filesystem::remove_all(underTemp ? temp / *relative.begin() : mModelDir);
    }

    virtual std::filesystem::path writeModelDir()
    {
        return writeDFlash2ModelDir(mStream, mDrafting.verifyTokens.value_or(kDFlash2BlockSize));
    }

    virtual void recordDraftBindings(rt::TensorMap const& map)
    {
        mDraftTrace.proposalSupportIds = map.get(binding_names::kSpecProposalSupportIds);
        mDraftTrace.proposalUnaryValues = map.get(binding_names::kSpecProposalUnaryValues);
        mDraftTrace.proposalProjectedHidden = map.get(binding_names::kSpecProposalProjectedHidden);
    }

    virtual void writeDraftOutputs(cudaStream_t stream)
    {
        writeDFlash2Proposals(mDraftTrace, stream);
    }

    virtual void onDraftCapture() {}

    std::unique_ptr<::testing::NiceMock<MockEngineExecutor>> makeBaseEngine(bool captureSucceeds = false)
    {
        using ::testing::_;
        using ::testing::Return;

        auto engine = std::make_unique<::testing::NiceMock<MockEngineExecutor>>();
        ON_CALL(*engine, getRequiredContextMemorySize()).WillByDefault(Return(4096));
        ON_CALL(*engine, setContextMemory(_)).WillByDefault(Return(true));
        ON_CALL(*engine, hasIOTensor(_)).WillByDefault(Return(false));
        ON_CALL(*engine, getBindingDataType(_)).WillByDefault(Return(trt_edgellm::DataType::kHALF));
        ON_CALL(*engine, prepare(_, _, _, _))
            .WillByDefault(
                [this](int32_t profile, rt::InferenceDims const& dims, rt::TensorMap const& map, cudaStream_t) {
                    mBaseTrace.profile = profile;
                    mBaseTrace.dims = dims;
                    mBaseTrace.prepares.emplace_back(profile, dims);
                    mBaseTrace.logits = map.get(binding_names::kLogits);
                    mBaseHidden = map.get(binding_names::kOutputHiddenStates);
                    return true;
                });
        ON_CALL(*engine, execute(_)).WillByDefault([this](cudaStream_t stream) {
            ++mBaseTrace.executions;
            if (mBaseTrace.profile == kPrefillProfile)
            {
                writeLogits(*mBaseTrace.logits, kVocabSize,
                    mPrefillTokens.empty()
                        ? std::vector<int32_t>(static_cast<size_t>(mBaseTrace.dims.batch), kDFlash2TargetToken)
                        : mPrefillTokens,
                    stream);
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
            if (mTagBaseHidden)
            {
                tagBaseHiddenRows(stream);
            }
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
        ON_CALL(*engine, getBindingDataType(_)).WillByDefault(Return(trt_edgellm::DataType::kHALF));
        ON_CALL(*engine, prepare(_, _, _, _))
            .WillByDefault(
                [this](int32_t profile, rt::InferenceDims const& dims, rt::TensorMap const& map, cudaStream_t) {
                    mDraftTrace.profile = profile;
                    mDraftTrace.dims = dims;
                    mDraftTrace.prepares.emplace_back(profile, dims);
                    auto const* targetHidden = map.get(binding_names::kDFlashTargetHiddenConcat);
                    mDraftTargetHidden = targetHidden;
                    mDraftTargetHiddenPrepares.push_back(
                        {profile, dims.batch, targetHidden != nullptr ? targetHidden->rawPointer() : nullptr,
                            targetHidden != nullptr ? targetHidden->getShape() : rt::Coords{}});
                    recordDraftBindings(map);
                    return true;
                });
        ON_CALL(*engine, execute(_)).WillByDefault([this](cudaStream_t stream) {
            ++mDraftTrace.executions;
            if (mTagBaseHidden && mDraftTrace.profile == kDecodeProfile)
            {
                readDraftTargetHidden(stream);
            }
            writeDraftOutputs(stream);
            return true;
        });
        ON_CALL(*engine, captureGraph(_)).WillByDefault([this, captureSucceeds](cudaStream_t) {
            mDraftTrace.captures.push_back(mDraftTrace.dims);
            if (mDraftTargetHiddenPrepares.empty())
            {
                ADD_FAILURE() << "draft graph captured before any prepare bound its inputs";
            }
            else
            {
                mDraftTargetHiddenCaptures.push_back(mDraftTargetHiddenPrepares.back());
            }
            onDraftCapture();
            return captureSucceeds;
        });
        return engine;
    }

    std::unique_ptr<rt::LLMInferenceRuntime> makeRuntime(
        bool captureSucceeds = false, rt::ContextCacheConfig const& contextCache = {})
    {
        auto artifacts = makeDFlash2Artifacts(
            mModelDir, makeBaseEngine(captureSucceeds), makeDraftEngine(captureSucceeds), mDrafting, mStream);
        return std::make_unique<rt::LLMInferenceRuntime>(std::move(artifacts), mModelDir.string(),
            /*multimodalEngineDir=*/"", std::unordered_map<std::string, std::string>{}, mDrafting, mStream,
            contextCache);
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

    std::vector<rt::InferenceDims> baseDecodePrepares() const
    {
        std::vector<rt::InferenceDims> result;
        for (auto const& [profile, dims] : mBaseTrace.prepares)
        {
            if (profile == kDecodeProfile)
            {
                result.push_back(dims);
            }
        }
        return result;
    }

    static int64_t targetHiddenDim()
    {
        return makeDFlash2DraftConfig()["base_model_hidden_size"].get<int64_t>();
    }

    //! Tag of base verification row `row` (sequence-major, kDFlash2BlockSize rows per sequence) in round `round`.
    static float baseRowTag(int32_t round, int64_t row)
    {
        return static_cast<float>(100 * (round + 1) + row);
    }

    //! Fills every column of each base verification row with that row's tag.
    void tagBaseHiddenRows(cudaStream_t stream)
    {
        ASSERT_NE(mBaseHidden, nullptr);
        int64_t const rows = mBaseTrace.dims.seqLen;
        int64_t const hidden = targetHiddenDim();
        ASSERT_LE(rows * hidden * static_cast<int64_t>(sizeof(half)), mBaseHidden->getMemoryCapacity());
        std::vector<half> values(static_cast<size_t>(rows * hidden));
        for (int64_t row = 0; row < rows; ++row)
        {
            std::fill_n(values.begin() + row * hidden, hidden, __float2half(baseRowTag(mVerifyRounds, row)));
        }
        CUDA_CHECK(cudaMemcpyAsync(
            mBaseHidden->rawPointer(), values.data(), values.size() * sizeof(half), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        ++mVerifyRounds;
    }

    //! Records the first column of every row the draft's target-hidden input holds when it executes.
    void readDraftTargetHidden(cudaStream_t stream)
    {
        ASSERT_NE(mDraftTargetHidden, nullptr);
        int64_t const hidden = targetHiddenDim();
        int64_t const rows = mDraftTargetHidden->getShape().volume() / hidden;
        std::vector<half> values(static_cast<size_t>(rows * hidden));
        CUDA_CHECK(cudaMemcpyAsync(values.data(), mDraftTargetHidden->rawPointer(), values.size() * sizeof(half),
            cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        std::vector<float> firstColumn(static_cast<size_t>(rows));
        for (int64_t row = 0; row < rows; ++row)
        {
            firstColumn[static_cast<size_t>(row)] = __half2float(values[static_cast<size_t>(row * hidden)]);
        }
        mDraftTargetHiddenReads.push_back(std::move(firstColumn));
    }

    //! Two sequences verify one round accepting `accepted` tokens each, then stop; the draft forward between the
    //! rounds must read each sequence's first accepted base rows, laid out sequence-major with max(accepted) rows.
    void expectDraftReadsAcceptedBaseRows(std::array<int32_t, 2> const& accepted)
    {
        mTagBaseHidden = true;
        std::vector<int32_t> firstRound = verificationPlan(accepted[0], kDFlash2TargetToken);
        std::vector<int32_t> const second = verificationPlan(accepted[1], kDFlash2TargetToken);
        firstRound.insert(firstRound.end(), second.begin(), second.end());
        mVerificationPlans.push_back(std::move(firstRound));
        std::vector<int32_t> stop = verificationPlan(/*acceptanceLength=*/1, /*replacementToken=*/1);
        std::vector<int32_t> const stopOne = stop;
        stop.insert(stop.end(), stopOne.begin(), stopOne.end());
        mVerificationPlans.push_back(std::move(stop));
        auto runtime = makeRuntime();

        auto request = makeGreedyRequest("a", /*maxGenerateLength=*/20);
        request.requests.push_back(request.requests.front());
        rt::LLMGenerationResponse response;
        ASSERT_TRUE(runtime->handleRequest(request, response, mStream));

        ASSERT_EQ(mDraftTargetHiddenReads.size(), 1U);
        std::vector<float> const& read = mDraftTargetHiddenReads.front();
        int64_t const maxDelta = std::max(accepted[0], accepted[1]);
        ASSERT_EQ(static_cast<int64_t>(read.size()), 2 * maxDelta);
        for (int32_t sequence = 0; sequence < 2; ++sequence)
        {
            for (int32_t row = 0; row < accepted[sequence]; ++row)
            {
                EXPECT_EQ(read[static_cast<size_t>(sequence * maxDelta + row)],
                    baseRowTag(/*round=*/0, static_cast<int64_t>(sequence) * kDFlash2BlockSize + row))
                    << "sequence " << sequence << " accepted row " << row;
            }
        }
    }

    rt::SpecDecodeDraftingConfig mDrafting{makeDFlash2Drafting()};
    cudaStream_t mStream{};
    std::filesystem::path mModelDir;
    bool mTagBaseHidden{false};
    rt::Tensor* mBaseHidden{nullptr};
    rt::Tensor const* mDraftTargetHidden{nullptr};
    int32_t mVerifyRounds{0};
    std::vector<std::vector<float>> mDraftTargetHiddenReads;
    DFlash2EngineTrace mBaseTrace;
    DFlash2EngineTrace mDraftTrace;
    std::vector<TargetHiddenBinding> mDraftTargetHiddenPrepares;
    std::vector<TargetHiddenBinding> mDraftTargetHiddenCaptures;
    std::deque<std::vector<int32_t>> mVerificationPlans;
    //! Per-sequence prefill tokens; empty means every sequence emits kDFlash2TargetToken.
    std::vector<int32_t> mPrefillTokens;
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

// Given a DFlash V2 deployment and a sequence mid-generation in a runtime session
// When a second sequence is seated beside it
// Then both draft from their own carried state in the shared round and finish with their own outputs
TEST_F(DFlash2AssemblyTest, SeatsASequenceIntoARunningSession)
{
    // The resident's first round accepts one proposal and one target replacement; the shared round ends both.
    mVerificationPlans.push_back(verificationPlan(/*acceptanceLength=*/2, kDFlash2TargetToken));
    std::vector<int32_t> sharedRoundPlan = verificationPlan(/*acceptanceLength=*/1, /*replacementToken=*/1);
    sharedRoundPlan.resize(2 * kDFlash2BlockSize, 1);
    mVerificationPlans.push_back(std::move(sharedRoundPlan));
    auto runtime = makeRuntime();
    ASSERT_TRUE(runtime->supportsSeatedAdmission());
    std::unique_ptr<rt::RuntimeSession> session = runtime->openSession(mStream);
    ASSERT_NE(session, nullptr);

    SpecSessionDriver driver(*session);
    ASSERT_NO_FATAL_FAILURE(driver.admitAndPrefill(makeGreedyRequest("a", /*maxGenerateLength=*/20), 1));
    ASSERT_NO_FATAL_FAILURE(driver.decode());
    ASSERT_NO_FATAL_FAILURE(driver.admitAndPrefill(makeGreedyRequest("a", /*maxGenerateLength=*/20), 2));
    ASSERT_NO_FATAL_FAILURE(driver.drain());

    auto const draftDecode = draftDecodePrepares();
    // The resident's own round, then the shared one.
    ASSERT_EQ(draftDecode.size(), 2U);
    EXPECT_EQ(draftDecode[0].batch, 1);
    EXPECT_EQ(draftDecode[1].batch, 2);
    // Both rows pad to the widest delta: the resident's two-token accepted span, beside the joiner's held-back prompt
    // position.
    EXPECT_EQ(draftDecode[1].selectLen, 2 * 2);
    ASSERT_EQ(driver.outcomes().count(1), 1U);
    ASSERT_EQ(driver.outcomes().count(2), 1U);
    EXPECT_EQ(driver.outcomes().at(1).outputIds.front(),
        (std::vector<int32_t>{kDFlash2TargetToken, kDFlash2ProposalToken, kDFlash2TargetToken, 1}));
    EXPECT_EQ(driver.outcomes().at(2).outputIds.front(), (std::vector<int32_t>{kDFlash2TargetToken, 1}));
    EXPECT_EQ(driver.outcomes().at(1).finishReasons.front(), rt::FinishReason::kEndId);
    EXPECT_EQ(driver.outcomes().at(2).finishReasons.front(), rt::FinishReason::kEndId);
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

TEST_F(DFlash2AssemblyTest, FullAcceptRoundReplaysCapturedTargetHiddenBinding)
{
    // Given a DFlash2 selector deployment whose draft CUDA graphs were captured, and a base that accepts the whole
    // 8-token block
    // When the next draft forward consumes all 8 accepted target-hidden rows
    // Then it binds the target hidden states at the device address and shape the batch-1, 8-row graph was captured
    // with, so that graph replays instead of the draft falling back to eager execution
    auto runtime = makeRuntime(/*captureSucceeds=*/true);
    ASSERT_TRUE(runtime->captureDecodingCUDAGraph(mStream));
    mDraftTargetHiddenPrepares.clear();

    mVerificationPlans.push_back(verificationPlan(/*acceptanceLength=*/kDFlash2BlockSize, kDFlash2TargetToken));
    mVerificationPlans.push_back(verificationPlan(/*acceptanceLength=*/1, /*replacementToken=*/1));
    auto const request = makeGreedyRequest("a", /*maxGenerateLength=*/20);
    rt::LLMGenerationResponse response;
    ASSERT_TRUE(runtime->handleRequest(request, response, mStream));

    std::vector<TargetHiddenBinding> decodeBindings;
    std::copy_if(mDraftTargetHiddenPrepares.begin(), mDraftTargetHiddenPrepares.end(),
        std::back_inserter(decodeBindings),
        [](TargetHiddenBinding const& binding) { return binding.profile == kDecodeProfile; });
    ASSERT_EQ(decodeBindings.size(), 1U);
    TargetHiddenBinding const& executed = decodeBindings.front();
    // The draft binds target hidden states ragged, one row per delta token.
    rt::Coords const fullBlockShape{kDFlash2BlockSize, targetHiddenDim()};
    auto const captured = std::find_if(mDraftTargetHiddenCaptures.begin(), mDraftTargetHiddenCaptures.end(),
        [&](TargetHiddenBinding const& binding) { return binding.batch == 1 && binding.shape == fullBlockShape; });
    ASSERT_NE(captured, mDraftTargetHiddenCaptures.end()) << "no batch-1 draft graph was captured for a full block";
    EXPECT_EQ(executed.shape, captured->shape);
    EXPECT_EQ(executed.address, captured->address);
}

TEST_F(DFlash2AssemblyTest, DraftReadsEachSequencesAcceptedBaseHiddenRows)
{
    // Given a DFlash2 selector deployment running two sequences, whose base verification writes a distinct value into
    // every hidden-state row it produces
    // When one verification round accepts the whole 8-token block for both sequences, or 3 and 6 tokens
    // Then the next draft forward's target-hidden input holds, for each sequence, exactly that sequence's accepted
    // base rows in order
    expectDraftReadsAcceptedBaseRows({kDFlash2BlockSize, kDFlash2BlockSize});
    mDraftTargetHiddenReads.clear();
    mVerifyRounds = 0;
    expectDraftReadsAcceptedBaseRows({3, 6});
}

// DFlash2 DDTree: draftTopK = 2 over 16 verify nodes.
//
// With zero selector codebooks every node ranks token 10 (logq log(e/(e+1))) above token 11 (log(1/(e+1))), so
// best-first assembly yields the greedy 10-chain at nodes 1, 2, 3, 4, 6, 9, 13 and the root branch 11 -> 10 -> 10
// at nodes 5 -> 8 -> 12. A target that follows that branch is accepted only through tree verification.
constexpr int32_t kDFlash2TreeVerifySize{16};
constexpr int32_t kDFlash2SecondToken{kDFlash2ProposalToken + 1};

//! Base targets per verify node that accept 11, 10, 10 along the branch at nodes 5 -> 8 -> 12, then the bonus.
std::vector<int32_t> branchVerificationPlan()
{
    std::vector<int32_t> plan(kDFlash2TreeVerifySize, kDFlash2TargetToken);
    plan[0] = kDFlash2SecondToken;
    plan[5] = kDFlash2ProposalToken;
    plan[8] = kDFlash2ProposalToken;
    return plan;
}

class DFlash2TreeAssemblyTest : public DFlash2AssemblyTest
{
protected:
    void SetUp() override
    {
        mDrafting = makeDFlash2Drafting(/*draftTopK=*/2, kDFlash2TreeVerifySize);
        DFlash2AssemblyTest::SetUp();
    }
};

// Given a DFlash2 deployment with draftTopK = 2 and 16 verify nodes whose draft ranks token 10 above 11 everywhere,
// When the base agrees with the draft's second choice at depth 1 and its first choice at depths 2 and 3,
// Then one round accepts 11, 10, 10 and the target's bonus token from a single 16-node base pass, and the next
// draft forward consumes the four accepted rows.
TEST_F(DFlash2TreeAssemblyTest, AcceptsABranchOffTheGreedyChain)
{
    mVerificationPlans.push_back(branchVerificationPlan());
    mVerificationPlans.push_back(std::vector<int32_t>(kDFlash2TreeVerifySize, 1));
    auto runtime = makeRuntime();

    auto const request = makeGreedyRequest("a", /*maxGenerateLength=*/20);
    rt::LLMGenerationResponse response;
    ASSERT_TRUE(runtime->handleRequest(request, response, mStream));

    ASSERT_EQ(response.outputIds.size(), 1U);
    EXPECT_EQ(response.outputIds[0],
        (std::vector<int32_t>{kDFlash2TargetToken, kDFlash2SecondToken, kDFlash2ProposalToken, kDFlash2ProposalToken,
            kDFlash2TargetToken, 1}));
    auto const verifies = baseDecodePrepares();
    ASSERT_EQ(verifies.size(), 2U);
    EXPECT_EQ(verifies[0].seqLen, kDFlash2TreeVerifySize);
    auto const draftDecode = draftDecodePrepares();
    ASSERT_EQ(draftDecode.size(), 1U);
    EXPECT_EQ(draftDecode[0].selectLen, 4);
}

// Given a DFlash2 tree deployment,
// When a request samples (temperature 1, topK 2),
// Then it is decoded by vanilla before the draft engine runs, because a DDTree verifies greedily.
TEST_F(DFlash2TreeAssemblyTest, RoutesSampledRequestsToVanilla)
{
    mVerificationPlans.push_back({1});
    auto runtime = makeRuntime();
    // Two tokens: the prefill's and one decode step, whichever of the two candidates that step samples.
    auto request = makeGreedyRequest("a", /*maxGenerateLength=*/2);
    request.temperature = 1.0F;
    request.topK = 2;
    rt::LLMGenerationResponse response;
    ASSERT_TRUE(runtime->handleRequest(request, response, mStream));
    EXPECT_EQ(mDraftTrace.executions, 0);
    for (auto const& dims : baseDecodePrepares())
    {
        EXPECT_EQ(dims.seqLen, dims.batch) << "a vanilla decode step feeds one token per request";
    }
}

// Given a DFlash2 tree deployment,
// When a greedy request explicitly asks for probabilistic proposals,
// Then the request fails instead of silently verifying a greedy tree.
TEST_F(DFlash2TreeAssemblyTest, RejectsExplicitProbabilisticProposals)
{
    auto runtime = makeRuntime();
    auto request = makeGreedyRequest("a", /*maxGenerateLength=*/5);
    request.proposalSampling = rt::SpecProposalSampling::kProbabilistic;
    rt::LLMGenerationResponse response;
    EXPECT_FALSE(runtime->handleRequest(request, response, mStream));
}

//! The widest jump-forward grafting round: the tree plus a forced chain of one block less its root.
constexpr int32_t kDFlash2TreeJumpVerifySize{kDFlash2TreeVerifySize + kDFlash2BlockSize - 1};

//! A DFlash2 tree deployment whose base verifies up to the widest grafting round.
class DFlash2TreeJumpHeadroomTest : public DFlash2TreeAssemblyTest
{
protected:
    std::filesystem::path writeModelDir() override
    {
        return writeDFlash2ModelDir(mStream, kDFlash2TreeJumpVerifySize);
    }
};

// Given a DFlash2 tree deployment whose base verifies kDFlash2BlockSize - 1 rows more than its 16-node tree,
// When the decoding CUDA graphs are captured,
// Then the base has a graph at the tree width, at the grafting width and at the one-token step of the vanilla decoder
// (which serves sampled requests) for every batch size, and a greedy request still decodes afterwards.
TEST_F(DFlash2TreeJumpHeadroomTest, CapturesTheGraftingWidthBesideTheTreeWidth)
{
    auto runtime = makeRuntime(/*captureSucceeds=*/true);
    ASSERT_TRUE(runtime->captureDecodingCUDAGraph(mStream));

    std::set<std::pair<int64_t, int64_t>> baseShapes;
    for (auto const& dims : mBaseTrace.captures)
    {
        baseShapes.emplace(dims.batch, dims.seqLen);
    }
    std::set<std::pair<int64_t, int64_t>> expectedBaseShapes;
    for (int64_t batch = 1; batch <= kMaxBatchSize; ++batch)
    {
        expectedBaseShapes.emplace(batch, batch);
        expectedBaseShapes.emplace(batch, batch * kDFlash2TreeVerifySize);
        expectedBaseShapes.emplace(batch, batch * kDFlash2TreeJumpVerifySize);
    }
    EXPECT_EQ(baseShapes, expectedBaseShapes);

    mVerificationPlans.push_back(std::vector<int32_t>(kDFlash2TreeVerifySize, 1));
    auto const request = makeGreedyRequest("a", /*maxGenerateLength=*/20);
    rt::LLMGenerationResponse response;
    ASSERT_TRUE(runtime->handleRequest(request, response, mStream));
    ASSERT_EQ(response.outputIds.size(), 1U);
    EXPECT_EQ(response.outputIds[0], (std::vector<int32_t>{kDFlash2TargetToken, 1}));
}

// --------------------------------------------------------------------------
// LiLiCorr: a DFlash V2 draft whose engine emits candidate ids and transition scores; the runtime walks the
// candidate chain instead of running the selector, and needs no selector sidecar.
// --------------------------------------------------------------------------

constexpr int32_t kLiLiCorrTopK{8};
//! Candidate index the fixture's transition scores favour by default; non-zero so a decoder that skips the walk
//! drafts a different token.
constexpr int32_t kLiLiCorrWinner{3};
//! Score margin of the favoured candidate. Large enough that even a sampled walk at temperature 1 picks it and every
//! other candidate's q underflows below the sparse-accept threshold.
constexpr float kLiLiCorrWinnerScore{100.0F};
//! Proposal token and favoured index for a sequence whose proposals must never reach verification in the eviction
//! case; the index differs from kLiLiCorrWinner, so the survivor's drafted candidate has q ~ 0 under this profile.
constexpr int32_t kLiLiCorrDecoyToken{30};
constexpr int32_t kLiLiCorrDecoyWinner{5};

//! Per-row draft output: candidate k of every slot is `token + k - winner`, and the scores favour index `winner`.
struct LiLiCorrRowProposal
{
    int32_t token;
    int32_t winner;
};

Json makeLiLiCorrConfigSection()
{
    Json section = makeDFlash2ConfigSection();
    section.erase("selector_rank");
    section.erase("selector_top_k");
    section["projector_type"] = "lilicorr";
    section["lilicorr_candidate_topk"] = kLiLiCorrTopK;
    return section;
}

std::filesystem::path writeLiLiCorrModelDir()
{
    auto const dir = makeScratchDir("lilicorrDecoderTests");
    Json base = makeDFlash2BaseConfig();
    base["dflash_config"] = makeLiLiCorrConfigSection();
    Json draft = makeDFlash2DraftConfig();
    draft["dflash_config"] = makeLiLiCorrConfigSection();
    std::ofstream(dir / "base_config.json") << base.dump(2);
    std::ofstream(dir / "draft_config.json") << draft.dump(2);
    writeTokenizerFiles(dir);
    return dir;
}

struct LiLiCorrScoresBinding
{
    int64_t batch;
    void const* address;
    rt::Coords shape;
};

struct LiLiCorrTableBinding
{
    void const* address;
    rt::Coords shape;
    trt_edgellm::DataType dtype;
};

class LiLiCorrAssemblyTest : public DFlash2AssemblyTest
{
protected:
    std::filesystem::path writeModelDir() override
    {
        return writeLiLiCorrModelDir();
    }

    void recordDraftBindings(rt::TensorMap const& map) override
    {
        mProposalSupportIds = map.get(binding_names::kSpecProposalSupportIds);
        mTransitionScores = map.get(binding_names::kSpecLiLiCorrTransitionScores);
        rt::Tensor const* table = map.get(binding_names::kLiLiCorrEmbeddingTable);
        mEmbeddingTableBindings.push_back(table != nullptr
                ? LiLiCorrTableBinding{table->rawPointer(), table->getShape(), table->getDataType()}
                : LiLiCorrTableBinding{nullptr, rt::Coords{}, trt_edgellm::DataType::kHALF});
        mSelectorOutputsBound = mSelectorOutputsBound || map.contains(binding_names::kSpecProposalUnaryValues)
            || map.contains(binding_names::kSpecProposalProjectedHidden);
    }

    void writeDraftOutputs(cudaStream_t stream) override
    {
        ASSERT_NE(mProposalSupportIds, nullptr);
        ASSERT_NE(mTransitionScores, nullptr);
        mExecutedScores.push_back(scoresBinding());
        auto const& idsShape = mProposalSupportIds->getShape();
        int64_t const batch = idsShape[0];
        int64_t const slots = idsShape[1];
        std::vector<int32_t> ids(static_cast<size_t>(idsShape.volume()));
        std::vector<float> scores(static_cast<size_t>(mTransitionScores->getShape().volume()));
        for (int64_t b = 0; b < batch; ++b)
        {
            LiLiCorrRowProposal const row = mRowProposals.at(static_cast<size_t>(b));
            for (int64_t s = 0; s < slots; ++s)
            {
                for (int32_t k = 0; k < kLiLiCorrTopK; ++k)
                {
                    ids[static_cast<size_t>((b * slots + s) * kLiLiCorrTopK + k)] = row.token + k - row.winner;
                    for (int32_t prev = 0; prev < kLiLiCorrTopK; ++prev)
                    {
                        scores[static_cast<size_t>(((b * slots + s) * kLiLiCorrTopK + prev) * kLiLiCorrTopK + k)]
                            = k == row.winner ? kLiLiCorrWinnerScore : 0.0F;
                    }
                }
            }
        }
        CUDA_CHECK(cudaMemcpyAsync(mProposalSupportIds->rawPointer(), ids.data(), ids.size() * sizeof(int32_t),
            cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(mTransitionScores->rawPointer(), scores.data(), scores.size() * sizeof(float),
            cudaMemcpyHostToDevice, stream));
    }

    void onDraftCapture() override
    {
        mCapturedScores.push_back(scoresBinding());
    }

    LiLiCorrScoresBinding scoresBinding() const
    {
        if (mTransitionScores == nullptr)
        {
            return {mDraftTrace.dims.batch, nullptr, rt::Coords{}};
        }
        return {mDraftTrace.dims.batch, mTransitionScores->rawPointer(), mTransitionScores->getShape()};
    }

    //! Assembles the runtime after `editEmbedding` adjusts the runtime embedding table; returns the table address.
    std::pair<std::unique_ptr<rt::LLMInferenceRuntime>, void const*> makeRuntimeWithEmbedding(
        std::function<void(rt::EmbeddingData&)> const& editEmbedding, bool captureSucceeds = false)
    {
        auto artifacts = makeDFlash2Artifacts(
            mModelDir, makeBaseEngine(captureSucceeds), makeDraftEngine(captureSucceeds), mDrafting, mStream);
        editEmbedding(artifacts.embedding);
        void const* const address = artifacts.embedding.table.rawPointer();
        auto runtime = std::make_unique<rt::LLMInferenceRuntime>(std::move(artifacts), mModelDir.string(),
            /*multimodalEngineDir=*/"", std::unordered_map<std::string, std::string>{}, mDrafting, mStream,
            rt::ContextCacheConfig{});
        return {std::move(runtime), address};
    }

    rt::Tensor* mProposalSupportIds{nullptr};
    rt::Tensor* mTransitionScores{nullptr};
    std::vector<LiLiCorrTableBinding> mEmbeddingTableBindings;
    bool mSelectorOutputsBound{false};
    std::vector<LiLiCorrRowProposal> mRowProposals{
        {kDFlash2ProposalToken, kLiLiCorrWinner}, {kDFlash2ProposalToken, kLiLiCorrWinner}};
    std::vector<LiLiCorrScoresBinding> mCapturedScores;
    std::vector<LiLiCorrScoresBinding> mExecutedScores;
};

TEST_P(LiLiCorrAssemblyTest, AcceptanceLengthPropagates)
{
    // Given a LiLiCorr deployment with no selector sidecar whose draft scores favour candidate 3 at every slot, and
    // a base whose verification accepts N tokens per round (N - 1 walked proposals plus the bonus token; N = 1, 4, 8)
    // When one greedy request runs to EOS
    // Then the walked candidate tokens are verified and emitted, no selector output is bound to the draft engine, and
    // the next draft forward consumes exactly N target-hidden delta rows
    int32_t const acceptanceLength = GetParam();
    mVerificationPlans.push_back(verificationPlan(acceptanceLength, kDFlash2TargetToken));
    mVerificationPlans.push_back(verificationPlan(/*acceptanceLength=*/1, /*replacementToken=*/1));

    auto runtime = makeRuntime();

    auto const request = makeGreedyRequest("a", /*maxGenerateLength=*/20);
    rt::LLMGenerationResponse response;
    ASSERT_TRUE(runtime->handleRequest(request, response, mStream));

    EXPECT_FALSE(mSelectorOutputsBound);
    auto const draftDecode = draftDecodePrepares();
    ASSERT_EQ(draftDecode.size(), 1U);
    EXPECT_EQ(draftDecode[0].selectLen, acceptanceLength);
    std::vector<int32_t> expected{kDFlash2TargetToken};
    expected.insert(expected.end(), static_cast<size_t>(acceptanceLength - 1), kDFlash2ProposalToken);
    expected.push_back(kDFlash2TargetToken);
    expected.push_back(1);
    ASSERT_EQ(response.outputIds.size(), 1U);
    EXPECT_EQ(response.outputIds[0], expected);
}

INSTANTIATE_TEST_SUITE_P(AcceptanceLengths, LiLiCorrAssemblyTest, ::testing::Values(1, 4, 8));

TEST_F(LiLiCorrAssemblyTest, StopsAtGenerationBudget)
{
    // Given a LiLiCorr deployment whose base agrees with every walked proposal in the 8-token block
    // When the request's generation budget is 3 tokens
    // Then the response holds exactly 3 tokens (the prefill token and the first two walked proposals) and ends on
    // length
    mVerificationPlans.push_back(verificationPlan(/*acceptanceLength=*/8, kDFlash2TargetToken));
    // Ends the request on EOS if a wrong proposal leaves the budget unspent, so the output shows the divergence.
    mVerificationPlans.push_back(verificationPlan(/*acceptanceLength=*/1, /*replacementToken=*/1));
    auto runtime = makeRuntime();

    auto const request = makeGreedyRequest("a", /*maxGenerateLength=*/3);
    rt::LLMGenerationResponse response;
    ASSERT_TRUE(runtime->handleRequest(request, response, mStream));

    ASSERT_EQ(response.outputIds.size(), 1U);
    EXPECT_EQ(response.outputIds[0],
        (std::vector<int32_t>{kDFlash2TargetToken, kDFlash2ProposalToken, kDFlash2ProposalToken}));
    EXPECT_EQ(response.finishReasons[0], rt::FinishReason::kLength);
}

TEST_F(LiLiCorrAssemblyTest, EvictionAfterPrefillKeepsSurvivorProposals)
{
    // Given a two-sequence LiLiCorr batch whose draft prefill gives each sequence different candidates and a different
    // transition-score profile, sampled probabilistically, and whose first sequence emits EOS at prefill
    // When the finished sequence is evicted while the prefill proposal is still pending
    // Then the survivor is verified on its own walked proposals and its own proposal probabilities: three are accepted
    // before EOS, and the evicted sequence's proposals and probabilities never reach verification
    mPrefillTokens = {1, kDFlash2TargetToken};
    mRowProposals = {{kLiLiCorrDecoyToken, kLiLiCorrDecoyWinner}, {kDFlash2ProposalToken, kLiLiCorrWinner}};
    mVerificationPlans.push_back(verificationPlan(/*acceptanceLength=*/4, /*replacementToken=*/1));
    // Ends the survivor on EOS if it verifies the wrong proposals, so the output shows the divergence.
    mVerificationPlans.push_back(verificationPlan(/*acceptanceLength=*/1, /*replacementToken=*/1));
    // Context reuse routes the request through the managed path, which evicts slots that finish during prefill.
    auto runtime = makeRuntime(/*captureSucceeds=*/false, rt::ContextCacheConfig{/*enabled=*/true, /*maxRecords=*/16});

    auto request = makeGreedyRequest("a", /*maxGenerateLength=*/20);
    request.requests.push_back(request.requests.front());
    // A top-1 target keeps verification deterministic, while the walk samples q = softmax(scores) at temperature 1;
    // a survivor verified against the evicted row's q gives its drafted tokens q = 0 and rejects them.
    request.temperature = 1.0F;
    request.topK = 1;
    request.proposalSampling = rt::SpecProposalSampling::kProbabilistic;
    rt::LLMGenerationResponse response;
    ASSERT_TRUE(runtime->handleRequest(request, response, mStream));

    ASSERT_EQ(response.outputIds.size(), 2U);
    EXPECT_EQ(response.outputIds[1],
        (std::vector<int32_t>{
            kDFlash2TargetToken, kDFlash2ProposalToken, kDFlash2ProposalToken, kDFlash2ProposalToken, 1}));
}

TEST_F(LiLiCorrAssemblyTest, GraphCaptureBindsScoresOnce)
{
    // Given a LiLiCorr deployment with draft CUDA graph capture enabled
    // When graphs are captured for every batch size and a request then runs
    // Then every capture and the later draft forwards see one transition-scores buffer at a single device address,
    // shaped [batch, 7, 8, 8] for the batch being captured or run
    auto runtime = makeRuntime(/*captureSucceeds=*/true);
    ASSERT_TRUE(runtime->captureDecodingCUDAGraph(mStream));

    mVerificationPlans.push_back(verificationPlan(/*acceptanceLength=*/1, /*replacementToken=*/1));
    auto const request = makeGreedyRequest("a", /*maxGenerateLength=*/20);
    rt::LLMGenerationResponse response;
    ASSERT_TRUE(runtime->handleRequest(request, response, mStream));

    ASSERT_FALSE(mCapturedScores.empty());
    ASSERT_FALSE(mExecutedScores.empty());
    void const* const address = mCapturedScores.front().address;
    ASSERT_NE(address, nullptr);
    for (auto const& bindings : {mCapturedScores, mExecutedScores})
    {
        for (auto const& binding : bindings)
        {
            EXPECT_EQ(binding.address, address);
            EXPECT_EQ(binding.shape, (rt::Coords{binding.batch, kDFlash2BlockSize - 1, kLiLiCorrTopK, kLiLiCorrTopK}));
        }
    }
    std::set<int64_t> capturedBatches;
    for (auto const& binding : mCapturedScores)
    {
        capturedBatches.insert(binding.batch);
    }
    EXPECT_EQ(capturedBatches, (std::set<int64_t>{1, 2}));
}

TEST_F(LiLiCorrAssemblyTest, UnequalAcceptBatchReplaysCapturedTargetHiddenBinding)
{
    // Given a two-sequence LiLiCorr deployment whose draft CUDA graphs were captured
    // When one verification round accepts 3 tokens for the first sequence and 6 for the second
    // Then the next draft forward binds the target hidden states at the device address and shape the batch-2, 6-row
    // graph was captured with, so that graph replays instead of the draft falling back to eager execution
    auto runtime = makeRuntime(/*captureSucceeds=*/true);
    ASSERT_TRUE(runtime->captureDecodingCUDAGraph(mStream));
    mDraftTargetHiddenPrepares.clear();

    std::vector<int32_t> firstRound = verificationPlan(/*acceptanceLength=*/3, kDFlash2TargetToken);
    std::vector<int32_t> const second = verificationPlan(/*acceptanceLength=*/6, kDFlash2TargetToken);
    firstRound.insert(firstRound.end(), second.begin(), second.end());
    mVerificationPlans.push_back(std::move(firstRound));
    std::vector<int32_t> stop = verificationPlan(/*acceptanceLength=*/1, /*replacementToken=*/1);
    std::vector<int32_t> const stopOne = stop;
    stop.insert(stop.end(), stopOne.begin(), stopOne.end());
    mVerificationPlans.push_back(std::move(stop));
    auto request = makeGreedyRequest("a", /*maxGenerateLength=*/20);
    request.requests.push_back(request.requests.front());
    rt::LLMGenerationResponse response;
    ASSERT_TRUE(runtime->handleRequest(request, response, mStream));

    std::vector<TargetHiddenBinding> decodeBindings;
    std::copy_if(mDraftTargetHiddenPrepares.begin(), mDraftTargetHiddenPrepares.end(),
        std::back_inserter(decodeBindings),
        [](TargetHiddenBinding const& binding) { return binding.profile == kDecodeProfile; });
    ASSERT_EQ(decodeBindings.size(), 1U);
    TargetHiddenBinding const& executed = decodeBindings.front();
    rt::Coords const sixRowBatchShape{2 * 6, targetHiddenDim()};
    auto const captured = std::find_if(mDraftTargetHiddenCaptures.begin(), mDraftTargetHiddenCaptures.end(),
        [&](TargetHiddenBinding const& binding) { return binding.batch == 2 && binding.shape == sixRowBatchShape; });
    ASSERT_NE(captured, mDraftTargetHiddenCaptures.end()) << "no batch-2 draft graph was captured for a 6-row delta";
    EXPECT_EQ(executed.batch, 2);
    EXPECT_EQ(executed.shape, captured->shape);
    EXPECT_EQ(executed.address, captured->address);
}

TEST_F(LiLiCorrAssemblyTest, MidDecodeEvictionKeepsSurvivorAcceptedState)
{
    // Given a two-sequence LiLiCorr batch whose base verification writes a distinct value into every hidden-state row,
    // and whose first verification round ends the first sequence on EOS while the second accepts three walked
    // proposals plus the bonus token
    // When the finished sequence is evicted before the next draft forward
    // Then the survivor's next draft forward runs at batch 1 on exactly its own 4 accepted base rows, and its next
    // walked proposals are verified: three are accepted before EOS
    mTagBaseHidden = true;
    std::vector<int32_t> firstRound = verificationPlan(/*acceptanceLength=*/1, /*replacementToken=*/1);
    std::vector<int32_t> const survivor = verificationPlan(/*acceptanceLength=*/4, kDFlash2TargetToken);
    firstRound.insert(firstRound.end(), survivor.begin(), survivor.end());
    mVerificationPlans.push_back(std::move(firstRound));
    mVerificationPlans.push_back(verificationPlan(/*acceptanceLength=*/4, /*replacementToken=*/1));
    auto runtime = makeRuntime();

    auto request = makeGreedyRequest("a", /*maxGenerateLength=*/20);
    request.requests.push_back(request.requests.front());
    rt::LLMGenerationResponse response;
    ASSERT_TRUE(runtime->handleRequest(request, response, mStream));

    auto const draftDecode = draftDecodePrepares();
    ASSERT_EQ(draftDecode.size(), 1U);
    EXPECT_EQ(draftDecode[0].batch, 1);
    std::vector<float> survivorRows;
    for (int64_t row = 0; row < 4; ++row)
    {
        survivorRows.push_back(baseRowTag(/*round=*/0, kDFlash2BlockSize + row));
    }
    ASSERT_EQ(mDraftTargetHiddenReads.size(), 1U);
    EXPECT_EQ(mDraftTargetHiddenReads.front(), survivorRows);
    ASSERT_EQ(response.outputIds.size(), 2U);
    EXPECT_EQ(response.outputIds[0], (std::vector<int32_t>{kDFlash2TargetToken, 1}));
    EXPECT_EQ(response.outputIds[1],
        (std::vector<int32_t>{kDFlash2TargetToken, kDFlash2ProposalToken, kDFlash2ProposalToken, kDFlash2ProposalToken,
            kDFlash2TargetToken, kDFlash2ProposalToken, kDFlash2ProposalToken, kDFlash2ProposalToken, 1}));
}

TEST_F(LiLiCorrAssemblyTest, DraftGathersFromTheRuntimeEmbeddingTable)
{
    // Given a LiLiCorr deployment whose runtime holds the base FP16 input-embedding table [V, H] and whose draft CUDA
    // graphs are captured
    // When graphs are captured for every batch size and a request then runs
    // Then every draft capture and forward binds lilicorr_embedding_table to that runtime table itself: one device
    // address, shape [V, H], FP16
    auto [runtime, tableAddress] = makeRuntimeWithEmbedding([](rt::EmbeddingData&) {}, /*captureSucceeds=*/true);
    ASSERT_TRUE(runtime->captureDecodingCUDAGraph(mStream));
    mVerificationPlans.push_back(verificationPlan(/*acceptanceLength=*/1, /*replacementToken=*/1));
    auto const request = makeGreedyRequest("a", /*maxGenerateLength=*/20);
    rt::LLMGenerationResponse response;
    ASSERT_TRUE(runtime->handleRequest(request, response, mStream));

    Json const base = makeDFlash2BaseConfig();
    rt::Coords const expectedShape{base["vocab_size"].get<int64_t>(), base["hidden_size"].get<int64_t>()};
    ASSERT_FALSE(mEmbeddingTableBindings.empty());
    for (auto const& binding : mEmbeddingTableBindings)
    {
        EXPECT_EQ(binding.address, tableAddress);
        EXPECT_EQ(binding.shape, expectedShape);
        EXPECT_EQ(binding.dtype, trt_edgellm::DataType::kHALF);
    }
}

TEST_F(LiLiCorrAssemblyTest, RejectsAnEmbeddingTableTheDraftCannotGatherRawRows)
{
    // Given a LiLiCorr deployment whose runtime embedding table is FP8 with per-block scales, or FP16 with an
    // embedding output scale of 2, or FP16 with one row fewer than the base vocabulary, or FP16 resident on the host
    // When the runtime is assembled
    // Then assembly fails naming the draft's lilicorr_embedding_table input, the output scale, the expected shape,
    // or GPU residence
    using ::testing::HasSubstr;
    auto const assemblyError = [this](std::function<void(rt::EmbeddingData&)> const& editEmbedding) -> std::string {
        try
        {
            makeRuntimeWithEmbedding(editEmbedding);
        }
        catch (std::exception const& error)
        {
            return error.what();
        }
        return "<accepted>";
    };

    EXPECT_EQ(assemblyError([](rt::EmbeddingData&) {}), "<accepted>");
    EXPECT_THAT(assemblyError([](rt::EmbeddingData& embedding) {
        auto const shape = embedding.table.getShape();
        embedding.table = rt::Tensor(shape, rt::DeviceType::kGPU, trt_edgellm::DataType::kFP8, "test::fp8Embedding");
        embedding.tableScalingFactor
            = rt::Tensor({shape[0], 1}, rt::DeviceType::kGPU, trt_edgellm::DataType::kFLOAT, "test::fp8EmbeddingScale");
        embedding.descriptor.format = kernel::EmbeddingTableFormat::kFP8;
    }),
        HasSubstr(binding_names::kLiLiCorrEmbeddingTable));
    EXPECT_THAT(assemblyError([](rt::EmbeddingData& embedding) { embedding.descriptor.modelOutputScale = 2.0F; }),
        HasSubstr("output scale"));
    EXPECT_THAT(assemblyError([](rt::EmbeddingData& embedding) {
        auto const shape = embedding.table.getShape();
        embedding.table = rt::Tensor(
            {shape[0] - 1, shape[1]}, rt::DeviceType::kGPU, trt_edgellm::DataType::kHALF, "test::shortEmbedding");
    }),
        HasSubstr("must have shape"));
    EXPECT_THAT(assemblyError([](rt::EmbeddingData& embedding) {
        embedding.table = rt::Tensor(
            embedding.table.getShape(), rt::DeviceType::kCPU, trt_edgellm::DataType::kHALF, "test::hostEmbedding");
    }),
        HasSubstr("reside on the GPU"));
}

// --------------------------------------------------------------------------
// DFlash2 jump-forward. Rejection sampling takes no forced tokens, so a greedy round that grafts a forced chain
// accepts through the greedy chain walk; every other round, and every sampling request, keeps rejection sampling.
// --------------------------------------------------------------------------

//! The tiny tokenizer spends one token per letter: "a" is 0, the other letters start at 3.
constexpr int32_t kEos{1};
constexpr int32_t kE{'e' - 'b' + 3};
constexpr int32_t kH{'h' - 'b' + 3};
constexpr int32_t kL{'l' - 'b' + 3};
constexpr int32_t kO{'o' - 'b' + 3};
constexpr int32_t kS{'s' - 'b' + 3};
constexpr int32_t kW{'w' - 'b' + 3};
constexpr int32_t kX{'x' - 'b' + 3};
constexpr int32_t kY{'y' - 'b' + 3};
constexpr int32_t kZ{'z' - 'b' + 3};

//! The prefill token and two whole blocks.
constexpr int64_t kTwoBlocksLength{1 + 2 * kDFlash2BlockSize};

//! A DFlash2 deployment whose engines play a script: each base forward, prefill or verification, writes the next entry
//! of mBaseForwards and each draft forward the next entry of mDraftForwards, the last entry repeating. Requests run
//! with the logger at DEBUG and both log streams captured.
class DFlash2JumpForwardTest : public ::testing::Test
{
protected:
    //! One logits row: `winner` at the peak and, when set, `tie` at the same peak.
    struct LogitsRow
    {
        int32_t winner;
        int32_t tie{-1};
    };
    //! The rows of one base forward; the last repeats.
    using BaseForward = std::vector<LogitsRow>;
    //! Per proposal position, the last repeating: the candidates the draft scores equally, far above the rest of the
    //! position's support.
    using DraftForward = std::vector<std::vector<int32_t>>;

    struct VerifyRound
    {
        std::vector<int32_t> queryLengths;
        std::vector<int32_t> pastLengths;
    };

    struct Run
    {
        rt::LLMGenerationResponse response;
        std::string warnings; //!< Warnings go to stderr.
        std::string debugLog; //!< Debug and info lines go to stdout.
    };

    void SetUp() override
    {
        mSavedLevel = gLogger.getLevel();
        CUDA_CHECK(cudaStreamCreate(&mStream));
        mModelDir = writeDFlash2ModelDir(mStream);
    }

    void TearDown() override
    {
        CUDA_CHECK(cudaStreamDestroy(mStream));
        std::filesystem::remove_all(mModelDir);
        gLogger.setLevel(mSavedLevel);
    }

    //! A base forward whose row r decodes to winners[r].
    static BaseForward rows(std::vector<int32_t> const& winners)
    {
        BaseForward forward;
        for (int32_t const winner : winners)
        {
            forward.push_back(LogitsRow{winner});
        }
        return forward;
    }

    //! A base forward whose every row ranks `first` and `second` equally.
    static BaseForward tied(int32_t first, int32_t second)
    {
        return {LogitsRow{first, second}};
    }

    //! A draft forward proposing tokens[p] at position p.
    static DraftForward proposals(std::vector<int32_t> const& tokens)
    {
        DraftForward forward;
        for (int32_t const token : tokens)
        {
            forward.push_back({token});
        }
        return forward;
    }

    //! A draft forward that scores `candidates` equally at every position.
    static DraftForward everyPosition(std::vector<int32_t> candidates)
    {
        return DraftForward{std::move(candidates)};
    }

    static rt::LLMGenerationRequest makeGuidedRequest(
        std::string const& regex, int64_t maxGenerateLength, bool jumpForward)
    {
        rt::LLMGenerationRequest request = makeGreedyRequest("a", maxGenerateLength);
        request.requests.front().guidedDecoding = rt::GuidedDecodingParams{rt::GuideType::kRegex, regex, jumpForward};
        return request;
    }

    //! makeGuidedRequest(), sampling at temperature 1 from the two best tokens.
    static rt::LLMGenerationRequest makeSamplingRequest(
        std::string const& regex, int64_t maxGenerateLength, bool jumpForward)
    {
        rt::LLMGenerationRequest request = makeGuidedRequest(regex, maxGenerateLength, jumpForward);
        request.temperature = 1.0F;
        request.topK = 2;
        return request;
    }

    std::unique_ptr<rt::LLMInferenceRuntime> makeRuntime()
    {
        using ::testing::_;
        auto base = makeEngine();
        ON_CALL(*base, prepare(_, _, _, _))
            .WillByDefault(
                [this](int32_t, rt::InferenceDims const& dims, rt::TensorMap const& map, cudaStream_t stream) {
                    mBaseLogits = map.get(binding_names::kLogits);
                    if (dims.executionPhaseLen == static_cast<int64_t>(rt::ExecutionPhase::kSpecTargetVerify))
                    {
                        auto const batch = static_cast<int32_t>(dims.batch);
                        mVerifyRounds.push_back(VerifyRound{readInt32(map, binding_names::kQueryLengths, batch, stream),
                            readInt32(map, binding_names::kPastLengths, batch, stream)});
                    }
                    return true;
                });
        ON_CALL(*base, execute(_)).WillByDefault([this](cudaStream_t stream) {
            writeBaseForward(next(mBaseForwards), stream);
            return true;
        });
        auto draft = makeEngine();
        ON_CALL(*draft, prepare(_, _, _, _))
            .WillByDefault([this](int32_t, rt::InferenceDims const& dims, rt::TensorMap const& map, cudaStream_t) {
                mDraftDeltas.push_back(dims.selectLen / dims.batch);
                mSupportIds = map.get(binding_names::kSpecProposalSupportIds);
                mUnaryValues = map.get(binding_names::kSpecProposalUnaryValues);
                mProjectedHidden = map.get(binding_names::kSpecProposalProjectedHidden);
                return true;
            });
        ON_CALL(*draft, execute(_)).WillByDefault([this](cudaStream_t stream) {
            writeDraftForward(next(mDraftForwards), stream);
            return true;
        });
        return std::make_unique<rt::LLMInferenceRuntime>(
            makeDFlash2Artifacts(mModelDir, std::move(base), std::move(draft), makeDFlash2Drafting(), mStream),
            mModelDir.string(),
            /*multimodalEngineDir=*/"", std::unordered_map<std::string, std::string>{}, makeDFlash2Drafting(), mStream);
    }

    Run run(rt::LLMInferenceRuntime& runtime, rt::LLMGenerationRequest const& request)
    {
        gLogger.setLevel(logger::Severity::kVERBOSE);
        testing::internal::CaptureStdout();
        testing::internal::CaptureStderr();
        Run result;
        bool const handled = runtime.handleRequest(request, result.response, mStream);
        result.debugLog = testing::internal::GetCapturedStdout();
        result.warnings = testing::internal::GetCapturedStderr();
        EXPECT_TRUE(handled) << result.warnings;
        return result;
    }

    std::vector<std::vector<int32_t>> verifyQueryLengths() const
    {
        std::vector<std::vector<int32_t>> lengths;
        for (auto const& round : mVerifyRounds)
        {
            lengths.push_back(round.queryLengths);
        }
        return lengths;
    }

    std::vector<std::vector<int32_t>> verifyPastLengths() const
    {
        std::vector<std::vector<int32_t>> lengths;
        for (auto const& round : mVerifyRounds)
        {
            lengths.push_back(round.pastLengths);
        }
        return lengths;
    }

    static size_t countOf(std::string const& text, std::string const& needle)
    {
        size_t count = 0;
        for (size_t at = text.find(needle); at != std::string::npos; at = text.find(needle, at + needle.size()))
        {
            ++count;
        }
        return count;
    }

    static bool ranWithJumpForward(Run const& run)
    {
        return run.debugLog.find(kRanWithJumpForward) != std::string::npos;
    }

    static constexpr char const* kRefusal{"Jump-forward is not supported on"};
    //! The statistics line a request's completion logs for a slot that ran with jump-forward.
    static constexpr char const* kRanWithJumpForward{"jump-forward: jumps="};

    std::deque<BaseForward> mBaseForwards;
    std::deque<DraftForward> mDraftForwards;
    std::vector<VerifyRound> mVerifyRounds;
    //! The target-hidden rows per slot each draft forward took: the prompt first, then each round's accepted tokens.
    std::vector<int64_t> mDraftDeltas;
    cudaStream_t mStream{};
    std::filesystem::path mModelDir;

private:
    static std::unique_ptr<::testing::NiceMock<MockEngineExecutor>> makeEngine()
    {
        using ::testing::_;
        using ::testing::Return;
        auto engine = std::make_unique<::testing::NiceMock<MockEngineExecutor>>();
        ON_CALL(*engine, getRequiredContextMemorySize()).WillByDefault(Return(4096));
        ON_CALL(*engine, setContextMemory(_)).WillByDefault(Return(true));
        ON_CALL(*engine, hasIOTensor(_)).WillByDefault(Return(false));
        ON_CALL(*engine, getBindingDataType(_)).WillByDefault(Return(trt_edgellm::DataType::kHALF));
        return engine;
    }

    template <typename Forward>
    static Forward next(std::deque<Forward>& script)
    {
        if (script.empty())
        {
            ADD_FAILURE() << "an engine ran past its script";
            return {};
        }
        Forward forward = script.front();
        if (script.size() > 1)
        {
            script.pop_front();
        }
        return forward;
    }

    static std::vector<int32_t> readInt32(
        rt::TensorMap const& map, char const* name, int32_t count, cudaStream_t stream)
    {
        rt::Tensor const* const tensor = map.get(name);
        if (tensor == nullptr || tensor->getShape().volume() < count)
        {
            return {};
        }
        std::vector<int32_t> values(static_cast<size_t>(count));
        CUDA_CHECK(cudaMemcpyAsync(
            values.data(), tensor->rawPointer(), values.size() * sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        return values;
    }

    void writeBaseForward(BaseForward const& forward, cudaStream_t stream)
    {
        ASSERT_NE(mBaseLogits, nullptr);
        ASSERT_FALSE(forward.empty());
        int64_t const rowCount = mBaseLogits->getShape().volume() / kVocabSize;
        std::vector<float> host(static_cast<size_t>(rowCount * kVocabSize));
        for (int64_t row = 0; row < rowCount; ++row)
        {
            LogitsRow const& spec = forward[std::min(static_cast<size_t>(row), forward.size() - 1)];
            for (int64_t token = 0; token < kVocabSize; ++token)
            {
                host[static_cast<size_t>(row * kVocabSize + token)]
                    = token == spec.tie ? kPeakLogit : fakeLogit(token, spec.winner);
            }
        }
        CUDA_CHECK(cudaMemcpyAsync(
            mBaseLogits->rawPointer(), host.data(), host.size() * sizeof(float), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    //! Writes the candidates at the selector score 0 and fills the rest of each support row with distinct non-letter
    //! ids scored low enough that neither the greedy nor the sampling selector picks them.
    void writeDraftForward(DraftForward const& forward, cudaStream_t stream)
    {
        constexpr int32_t kFirstFillerId{64};
        constexpr float kFillerScore{-1.0e9F};
        ASSERT_NE(mSupportIds, nullptr);
        ASSERT_NE(mUnaryValues, nullptr);
        ASSERT_NE(mProjectedHidden, nullptr);
        ASSERT_FALSE(forward.empty());
        auto const& shape = mSupportIds->getShape();
        ASSERT_EQ(shape.getNumDims(), 3);
        int64_t const rowCount = shape[0] * shape[1];
        int64_t const supportSize = shape[2];
        std::vector<int32_t> ids(static_cast<size_t>(rowCount * supportSize));
        std::vector<float> scores(ids.size());
        for (int64_t row = 0; row < rowCount; ++row)
        {
            auto const position = static_cast<size_t>(row % shape[1]);
            std::vector<int32_t> const& candidates = forward[std::min(position, forward.size() - 1)];
            for (int64_t k = 0; k < supportSize; ++k)
            {
                bool const candidate = k < static_cast<int64_t>(candidates.size());
                auto const offset = static_cast<size_t>(row * supportSize + k);
                ids[offset] = candidate ? candidates[static_cast<size_t>(k)] : kFirstFillerId + static_cast<int32_t>(k);
                scores[offset] = candidate ? 0.0F : kFillerScore;
            }
        }
        CUDA_CHECK(cudaMemcpyAsync(
            mSupportIds->rawPointer(), ids.data(), ids.size() * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(
            mUnaryValues->rawPointer(), scores.data(), scores.size() * sizeof(float), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemsetAsync(mProjectedHidden->rawPointer(), 0, mProjectedHidden->getMemoryCapacity(), stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    logger::Severity mSavedLevel{};
    rt::Tensor* mBaseLogits{nullptr};
    rt::Tensor* mSupportIds{nullptr};
    rt::Tensor* mUnaryValues{nullptr};
    rt::Tensor* mProjectedHidden{nullptr};
};

// Given a DFlash2 deployment, the grammar `[a-z][yz]hello[a-z]+`, a draft that proposes y, z, z, z, z, z, z after the
//      root `s`, and a base that predicts `y` at the root and `z` at every other row
// When twelve tokens are generated with jump-forward
// Then the chain `hello` planned behind `y` replaces the draft after it: the first round verifies the root, `y` and
//      the five forced tokens and accepts y h e l l o and the base's `z`, `y` being its one accepted draft of the seven
//      proposed and the forced tokens one jump; the base commits those seven tokens and the next draft forward takes
//      them as its delta, and the next round grafts nothing and accepts up to the length limit
TEST_F(DFlash2JumpForwardTest, AcceptsAGraftedChainThroughTheGreedyWalkAndCommitsIt)
{
    mBaseForwards = {rows({kS}), rows({kY, kZ})};
    mDraftForwards = {proposals({kY, kZ})};
    auto runtime = makeRuntime();

    Run const on = run(*runtime, makeGuidedRequest("[a-z][yz]hello[a-z]+", /*maxGenerateLength=*/12, true));

    ASSERT_EQ(on.response.outputIds.size(), 1U);
    EXPECT_EQ(on.response.outputIds[0], (std::vector<int32_t>{kS, kY, kH, kE, kL, kL, kO, kZ, kY, kZ, kZ, kZ}));
    EXPECT_EQ(countOf(on.warnings, kRefusal), 0U) << on.warnings;
    EXPECT_EQ(verifyQueryLengths(), (std::vector<std::vector<int32_t>>{{2 + 5}, {kDFlash2BlockSize}}))
        << "depth 1 + 1 + five forced tokens; the draft's z's are cut";
    EXPECT_EQ(verifyPastLengths(), (std::vector<std::vector<int32_t>>{{1}, {1 + 7}}));
    EXPECT_EQ(mDraftDeltas, (std::vector<int64_t>{1, 7}));
    EXPECT_NE(on.debugLog.find("generated=7, acceptedDrafts=1, proposedDrafts=7"), std::string::npos) << on.debugLog;
    EXPECT_NE(on.debugLog.find("jump-forward: jumps=1 forced=5 rolledBack=0 chunkSteps=1"), std::string::npos)
        << on.debugLog;
}

// Given a DFlash2 deployment whose draft and base both rank `x` and `w` equally, far above every other token, and the
//      grammar `[xw]+`
// When two requests sampling at temperature 1 from the two best tokens ask for jump-forward
// Then one warning names non-greedy sampling, neither request runs with jump-forward, and both still
//      decode by rejection sampling, which accepts every proposal of a draft whose distribution equals the base's:
//      each request's 17 tokens take two verify rounds
TEST_F(DFlash2JumpForwardTest, RefusesJumpForwardOnceToASamplingRequestThatStillVerifiesLosslessly)
{
    mBaseForwards = {tied(kX, kW)};
    mDraftForwards = {everyPosition({kX, kW})};
    auto runtime = makeRuntime();

    Run const first = run(*runtime, makeSamplingRequest("[xw]+", kTwoBlocksLength, true));
    Run const second = run(*runtime, makeSamplingRequest("[xw]+", kTwoBlocksLength, true));

    EXPECT_EQ(countOf(first.warnings, kRefusal), 1U) << first.warnings;
    EXPECT_NE(first.warnings.find("non-greedy sampling under speculative decoding"), std::string::npos)
        << first.warnings;
    EXPECT_EQ(countOf(second.warnings, kRefusal), 0U) << "a reason warns once per runtime";
    for (Run const* sampled : {&first, &second})
    {
        EXPECT_FALSE(ranWithJumpForward(*sampled));
        EXPECT_EQ(sampled->response.specVerifyCounts, std::vector<int32_t>{2});
    }
}

// Given a DFlash2 deployment that has served a greedy jump-forward request whose only round grafted `hello`
// When a request without jump-forward samples at temperature 1 from `x` and `w`, which the draft and the base rank
//      equally and far above every other token, under the grammar `[xw]+`
// Then it decodes by rejection sampling: each round verifies the plain block and accepts every proposal, so its 17
//      tokens take two verify rounds
TEST_F(DFlash2JumpForwardTest, ASamplingRequestWithoutJumpForwardKeepsRejectionSamplingAfterAGraftingRequest)
{
    mBaseForwards = {rows({kS}), rows({kY, kZ}), tied(kX, kW)};
    mDraftForwards = {proposals({kY, kZ}), everyPosition({kX, kW})};
    auto runtime = makeRuntime();

    Run const grafted = run(*runtime, makeGuidedRequest("[a-z][yz]hello", /*maxGenerateLength=*/10, true));
    mVerifyRounds.clear();
    Run const sampled = run(*runtime, makeSamplingRequest("[xw]+", kTwoBlocksLength, false));

    ASSERT_EQ(grafted.response.outputIds.size(), 1U);
    EXPECT_EQ(grafted.response.outputIds[0], (std::vector<int32_t>{kS, kY, kH, kE, kL, kL, kO, kEos}));
    EXPECT_NE(grafted.debugLog.find("jump-forward: jumps=1 forced=5"), std::string::npos) << grafted.debugLog;
    EXPECT_EQ(sampled.response.specVerifyCounts, std::vector<int32_t>{2});
    EXPECT_EQ(verifyQueryLengths(), (std::vector<std::vector<int32_t>>{{kDFlash2BlockSize}, {kDFlash2BlockSize}}));
}

// --------------------------------------------------------------------------

} // namespace

// Given a DFlash deployment and two prompts
// When the second is seated into a running session while the first is mid-generation, or retired unseated
// Then each decodes exactly as it does alone: the seated draft state moved no token and no round
TEST_F(DFlashAssemblyTest, Seating)
{
    expectSeating();
}
