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

#include "substituteEngine.h"

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
        rt::Tensor matrix({draftVocabSize, kDSparkMarkovRank}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF, name);
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

class DSparkAssemblyTest : public SpecAssemblyTest
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
        config.draftingTopK = 1;
        config.draftingStep = 1;
        config.verifySize = kDSparkBlockSize;
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
            nvinfer1::DataType::kHALF, name);
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
    config["builder_config"]["max_verify_tree_size"] = kDSparkTreeVerifySize;
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
    config["builder_config"]["max_draft_tree_size"] = kDSparkTreeBlockSize;
    sizeSpeculativeKvPool(config);
    return config;
}

rt::SpecDecodeDraftingConfig makeDsparkDrafting(bool useTree)
{
    rt::SpecDecodeDraftingConfig drafting{};
    drafting.draftingTopK = useTree ? 4 : 1;
    drafting.draftingStep = 1;
    drafting.verifySize = kDSparkTreeVerifySize;
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
        nvinfer1::DataType parentType = nvinfer1::DataType::kINT32,
        nvinfer1::DataType depthType = nvinfer1::DataType::kINT32)
    {
        using ::testing::_;

        auto engine = std::make_unique<::testing::NiceMock<MockEngineExecutor>>();
        ON_CALL(*engine, hasIOTensor(_)).WillByDefault([hasParent, hasDepth](char const* name) {
            std::string const binding{name};
            return (binding == binding_names::kTreeParentIds && hasParent)
                || (binding == binding_names::kTreeDepths && hasDepth);
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
            rt::validateDsparkTreeMetadataBindings(deployment, engine);
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
    expectValidationError(deployment, *engine, "Hybrid DSpark DDTree requires a tree-base engine");
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
    auto const engine
        = makeBaseEngine(/*hasParent=*/true, /*hasDepth=*/true, nvinfer1::DataType::kINT32, nvinfer1::DataType::kHALF);
    expectValidationError(deployment, *engine, "tree metadata bindings must be INT32");
}

TEST_F(DSparkTreeEngineAbiTest, AcceptsMatchingTreeAndLinearEngineAbis)
{
    auto const treeDeployment = makeDeployment(/*useTree=*/true);
    auto const treeEngine = makeBaseEngine(/*hasParent=*/true, /*hasDepth=*/true);
    EXPECT_NO_THROW(rt::validateDsparkTreeMetadataBindings(treeDeployment, *treeEngine));

    auto const linearDeployment = makeDeployment(/*useTree=*/false);
    auto const linearEngine = makeBaseEngine(/*hasParent=*/false, /*hasDepth=*/false);
    EXPECT_NO_THROW(rt::validateDsparkTreeMetadataBindings(linearDeployment, *linearEngine));
}

TEST_F(DSparkTreeEngineAbiTest, RejectsTreeEngineForLinearRuntime)
{
    auto const deployment = makeDeployment(/*useTree=*/false);
    auto const engine = makeBaseEngine(/*hasParent=*/true, /*hasDepth=*/true);
    expectValidationError(deployment, *engine, "runtime is configured for linear DSpark");
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
        EXPECT_THAT(e.what(), ::testing::HasSubstr("Hybrid DSpark DDTree requires a tree-base engine"));
    }
}

} // namespace
