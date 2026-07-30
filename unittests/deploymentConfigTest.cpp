/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "runtime/config/deploymentConfig.h"

#include "testUtils.h"
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

using namespace trt_edgellm;
using namespace trt_edgellm::rt;
using Json = nlohmann::json;

namespace
{

//! Base config JSON with optional SpecDecode fields. If `specDecodeMaxVerifyTreeSize`
//! is > 0, the config enables SpecDecode and writes `max_verify_tree_size`.
//!
//! Note: `max_draft_tree_size` is draft-only and is not written on the base
//! side (the builder only emits it when `specDecodeDraft` is set). The
//! `specDecodeMaxDraftTreeSize` parameter is accepted for parallelism with
//! `makeDraftConfig` in call sites but intentionally ignored here.
Json makeBaseConfig(
    int32_t specDecodeMaxVerifyTreeSize = 0, int32_t /*specDecodeMaxDraftTreeSize*/ = 0, int32_t maxBatchSize = 2)
{
    Json config;
    config["num_hidden_layers"] = 12;
    config["num_key_value_heads"] = 4;
    config["head_dim"] = 64;
    config["hidden_size"] = 768;
    config["vocab_size"] = 32000;
    config["kv_cache_dtype"] = "fp16";
    config["spec_decode_type"] = "none";
    config["engine_role"] = "llm";

    Json bc;
    bc["max_batch_size"] = maxBatchSize;
    bc["max_input_len"] = 128;
    bc["max_kv_cache_capacity"] = 256;
    bc["max_lora_rank"] = 0;
    if (specDecodeMaxVerifyTreeSize > 0)
    {
        config["spec_decode_type"] = "eagle3";
        config["engine_role"] = "base";
        bc["spec_base"] = true;
        bc["max_verify_tree_size"] = specDecodeMaxVerifyTreeSize;
    }
    else
    {
        bc["spec_base"] = false;
    }
    config["builder_config"] = bc;
    return config;
}

//! Draft config JSON. Mirrors `parseDraftEngineConfig`'s expected schema.
//!
//! Note: `max_verify_tree_size` is base-only and is not written on the draft
//! side (the builder only emits it when `specDecodeBase` is set). The
//! `maxVerifyTreeSize` parameter is accepted for parallelism with
//! `makeBaseConfig` but intentionally ignored here.
Json makeDraftConfig(int32_t /*maxVerifyTreeSize*/, int32_t maxDraftTreeSize, int32_t maxBatchSize = 2)
{
    Json config;
    config["spec_decode_type"] = "eagle3";
    config["engine_role"] = "draft";
    config["num_hidden_layers"] = 1;
    config["num_key_value_heads"] = 4;
    config["head_dim"] = 64;
    config["hidden_size"] = 768;
    config["draft_vocab_size"] = 32000;
    config["base_model_hidden_size"] = 768 * 3;
    config["kv_cache_dtype"] = "fp16";

    Json bc;
    bc["max_batch_size"] = maxBatchSize;
    bc["max_input_len"] = 128;
    bc["max_kv_cache_capacity"] = 256;
    bc["spec_draft"] = true;
    bc["max_draft_tree_size"] = maxDraftTreeSize;
    config["builder_config"] = bc;
    return config;
}

Json makeMTPBaseConfig(int32_t maxVerifyTreeSize, int32_t maxBatchSize = 2)
{
    Json config = makeBaseConfig(maxVerifyTreeSize, /*maxDraft=*/0, maxBatchSize);
    config["spec_decode_type"] = "mtp";
    config["engine_role"] = "base";
    return config;
}

Json makeMTPDraftConfig(int32_t maxDraftTreeSize, int32_t maxBatchSize = 2)
{
    Json config = makeDraftConfig(/*maxVerify=*/0, maxDraftTreeSize, maxBatchSize);
    config["spec_decode_type"] = "mtp";
    config["engine_role"] = "draft";
    config["base_model_hidden_size"] = config["hidden_size"];
    return config;
}

Json makeGemma4MTPBaseConfig(int32_t maxVerifyTreeSize, int32_t maxBatchSize = 2)
{
    Json config = makeBaseConfig(maxVerifyTreeSize, /*maxDraft=*/0, maxBatchSize);
    config["spec_decode_type"] = "gemma4_mtp";
    config["engine_role"] = "base";
    config["model"] = "gemma4";
    return config;
}

Json makeGemma4MTPDraftConfig(int32_t maxDraftTreeSize, int32_t maxBatchSize = 2)
{
    Json config = makeDraftConfig(/*maxVerify=*/0, maxDraftTreeSize, maxBatchSize);
    config["spec_decode_type"] = "gemma4_mtp";
    config["engine_role"] = "draft";
    config["model"] = "gemma4_assistant";
    config["base_model_hidden_size"] = config["hidden_size"];
    config["assistant_hidden_size"] = config["hidden_size"];
    config["shares_target_kv"] = true;
    config["has_own_kv_cache"] = false;
    config["constant_draft_positions"] = true;
    config["returns_feedback_hidden"] = true;
    config["kv_sharing_map"] = Json::array({Json{{"assistant_layer", 0}, {"target_attention_layer", 0}}});
    return config;
}

Json makeHybridMTPBaseConfig(int32_t maxVerifyTreeSize, int32_t maxBatchSize = 2)
{
    Json config = makeMTPBaseConfig(maxVerifyTreeSize, maxBatchSize);
    config["num_attention_layers"] = 8;
    config["num_linear_attn_layers"] = 4;
    config["recurrent_state_num_heads"] = 4;
    config["recurrent_state_head_dim"] = 128;
    config["recurrent_state_size"] = 128;
    config["conv_dim"] = 1024;
    config["conv_kernel"] = 4;
    config["recurrent_state_dtype"] = "fp32";
    config["conv_state_dtype"] = "fp16";
    return config;
}

Json makeHybridDFlashBaseConfig(int32_t maxVerifyTreeSize, int32_t maxBatchSize = 2)
{
    Json config = makeBaseConfig(maxVerifyTreeSize, /*maxDraft=*/0, maxBatchSize);
    config["spec_decode_type"] = "dflash";
    config["engine_role"] = "base";
    config["num_attention_layers"] = 8;
    config["num_linear_attn_layers"] = 4;
    config["recurrent_state_num_heads"] = 4;
    config["recurrent_state_head_dim"] = 128;
    config["recurrent_state_size"] = 128;
    config["conv_dim"] = 1024;
    config["conv_kernel"] = 4;
    config["recurrent_state_dtype"] = "fp32";
    config["conv_state_dtype"] = "fp16";
    config["dflash_config"]
        = Json{{"block_size", 16}, {"mask_token_id", 248070}, {"target_layer_ids", Json::array({1, 8})}};
    return config;
}

Json makeDenseDFlashBaseConfig(int32_t maxVerifyTreeSize, int32_t maxBatchSize = 2)
{
    Json config = makeBaseConfig(maxVerifyTreeSize, /*maxDraft=*/0, maxBatchSize);
    config["spec_decode_type"] = "dflash";
    config["engine_role"] = "base";
    config["dflash_config"]
        = Json{{"block_size", 16}, {"mask_token_id", 248070}, {"target_layer_ids", Json::array({1, 8})}};
    return config;
}

Json makeDFlashDraftConfig(int32_t maxDraftTreeSize, int32_t maxBatchSize = 2)
{
    Json config = makeDraftConfig(/*maxVerify=*/0, maxDraftTreeSize, maxBatchSize);
    config["spec_decode_type"] = "dflash";
    config["engine_role"] = "draft";
    config["dflash_config"]
        = Json{{"block_size", 16}, {"mask_token_id", 248070}, {"target_layer_ids", Json::array({1, 8})}};
    return config;
}

//! Write a JSON object to a unique temp file and return its path.
std::filesystem::path writeJsonToTempFile(Json const& json, std::string const& suffix)
{
    auto tmpPath = std::filesystem::temp_directory_path() / ("deploymentConfigTest_" + suffix + ".json");
    std::ofstream ofs(tmpPath);
    ofs << json.dump(2);
    ofs.close();
    return tmpPath;
}

} // namespace

class DeploymentConfigTest : public ::testing::Test
{
protected:
    void TearDown() override
    {
        std::filesystem::remove(std::filesystem::temp_directory_path() / "deploymentConfigTest_base.json");
        std::filesystem::remove(std::filesystem::temp_directory_path() / "deploymentConfigTest_draft.json");
    }
};

TEST_F(DeploymentConfigTest, VanillaBundle)
{
    // Base only, no draft, no drafting → succeeds, draft/drafting absent.
    Json const baseJson = makeBaseConfig();
    auto const basePath = writeJsonToTempFile(baseJson, "base");

    DeploymentConfig bundle = createDeploymentConfig(basePath, std::nullopt, std::nullopt);

    EXPECT_EQ(bundle.base.hiddenSize, baseJson["hidden_size"].get<int32_t>());
    EXPECT_FALSE(bundle.base.isSpecDecodeBase);
    EXPECT_FALSE(bundle.draft.has_value());
    EXPECT_FALSE(bundle.specConfig.has_value());
}

TEST_F(DeploymentConfigTest, SpecDecodeBundle)
{
    // Base + draft + drafting with valid values → succeeds, all fields populated.
    Json const baseJson = makeBaseConfig(/*maxVerify=*/16, /*maxDraft=*/16);
    Json const draftJson = makeDraftConfig(/*maxVerify=*/16, /*maxDraft=*/16);
    auto const basePath = writeJsonToTempFile(baseJson, "base");
    auto const draftPath = writeJsonToTempFile(draftJson, "draft");

    SpecDecodeDraftingConfig drafting{};
    drafting.draftingTopK = 4;
    drafting.draftingStep = 4; // draftingStep * draftingTopK = 16 <= 16
    drafting.verifySize = 8;   // 8 <= 16

    DeploymentConfig bundle = createDeploymentConfig(
        basePath, std::optional<std::filesystem::path>{draftPath}, std::optional<SpecDecodeDraftingConfig>{drafting});

    EXPECT_TRUE(bundle.base.isSpecDecodeBase);
    ASSERT_TRUE(bundle.draft.has_value());
    // Draft engines set isSpecDecodeBase=false (they ARE the draft, not the base).
    // Presence of a draft engine is indicated by maxDraftTreeSize > 0.
    EXPECT_FALSE(bundle.draft->isSpecDecodeBase);
    EXPECT_GT(bundle.draft->maxDraftTreeSize, 0);
    ASSERT_TRUE(bundle.specConfig.has_value());

    EXPECT_EQ(bundle.base.maxVerifyTreeSize, 16);
    // `maxDraftTreeSize` is only meaningful on the draft side (see makeBaseConfig).
    EXPECT_EQ(bundle.base.maxDraftTreeSize, 0);
    // `maxVerifyTreeSize` is only meaningful on the base side (see makeDraftConfig).
    EXPECT_EQ(bundle.draft->maxVerifyTreeSize, 0);
    EXPECT_EQ(bundle.draft->maxDraftTreeSize, 16);
    EXPECT_EQ(bundle.specConfig->verifySize, 8);
    EXPECT_EQ(bundle.specConfig->draftingStep, 4);
    EXPECT_EQ(bundle.specConfig->draftingTopK, 4);
}

TEST_F(DeploymentConfigTest, DraftingWithoutDraftThrows)
{
    // Drafting set but draft not set → throws.
    Json const baseJson = makeBaseConfig(/*maxVerify=*/16, /*maxDraft=*/16);
    auto const basePath = writeJsonToTempFile(baseJson, "base");

    SpecDecodeDraftingConfig drafting{};
    drafting.draftingTopK = 4;
    drafting.draftingStep = 4;
    drafting.verifySize = 8;

    EXPECT_THROW(createDeploymentConfig(basePath, std::nullopt, std::optional<SpecDecodeDraftingConfig>{drafting}),
        std::runtime_error);
}

TEST_F(DeploymentConfigTest, DraftingExceedsVerifyCapacityThrows)
{
    // User's verifySize > base.maxVerifyTreeSize → throws.
    Json const baseJson = makeBaseConfig(/*maxVerify=*/8, /*maxDraft=*/16);
    Json const draftJson = makeDraftConfig(/*maxVerify=*/8, /*maxDraft=*/16);
    auto const basePath = writeJsonToTempFile(baseJson, "base");
    auto const draftPath = writeJsonToTempFile(draftJson, "draft");

    SpecDecodeDraftingConfig drafting{};
    drafting.draftingTopK = 2;
    drafting.draftingStep = 2; // 2 * 2 = 4 <= 16 (OK)
    drafting.verifySize = 16;  // 16 > 8 (violation)

    EXPECT_THROW(createDeploymentConfig(basePath, std::optional<std::filesystem::path>{draftPath},
                     std::optional<SpecDecodeDraftingConfig>{drafting}),
        std::runtime_error);
}

TEST_F(DeploymentConfigTest, DraftingExceedsDraftCapacityThrows)
{
    // User's draftingStep * draftingTopK > draft->maxDraftTreeSize → throws.
    Json const baseJson = makeBaseConfig(/*maxVerify=*/16, /*maxDraft=*/8);
    Json const draftJson = makeDraftConfig(/*maxVerify=*/16, /*maxDraft=*/8);
    auto const basePath = writeJsonToTempFile(baseJson, "base");
    auto const draftPath = writeJsonToTempFile(draftJson, "draft");

    SpecDecodeDraftingConfig drafting{};
    drafting.draftingTopK = 4;
    drafting.draftingStep = 4; // 4 * 4 = 16 > 8 (violation)
    drafting.verifySize = 8;

    EXPECT_THROW(createDeploymentConfig(basePath, std::optional<std::filesystem::path>{draftPath},
                     std::optional<SpecDecodeDraftingConfig>{drafting}),
        std::runtime_error);
}

TEST_F(DeploymentConfigTest, ConsistentBundleValidatesOk)
{
    // All fields match → succeeds.
    Json const baseJson = makeBaseConfig(/*maxVerify=*/32, /*maxDraft=*/24);
    Json const draftJson = makeDraftConfig(/*maxVerify=*/32, /*maxDraft=*/24);
    auto const basePath = writeJsonToTempFile(baseJson, "base");
    auto const draftPath = writeJsonToTempFile(draftJson, "draft");

    SpecDecodeDraftingConfig drafting{};
    drafting.draftingTopK = 3;
    drafting.draftingStep = 8; // 3 * 8 = 24 <= 24
    drafting.verifySize = 32;  // 32 <= 32

    DeploymentConfig bundle = createDeploymentConfig(
        basePath, std::optional<std::filesystem::path>{draftPath}, std::optional<SpecDecodeDraftingConfig>{drafting});

    EXPECT_TRUE(bundle.base.isSpecDecodeBase);
    EXPECT_TRUE(bundle.draft.has_value());
    ASSERT_TRUE(bundle.specConfig.has_value());
    EXPECT_EQ(bundle.base.maxVerifyTreeSize, 32);
    EXPECT_EQ(bundle.draft->maxDraftTreeSize, 24);
}

TEST_F(DeploymentConfigTest, MTPTopKOneUsesTreeValidation)
{
    Json const baseJson = makeMTPBaseConfig(/*maxVerify=*/16);
    Json const draftJson = makeMTPDraftConfig(/*maxDraft=*/16);
    auto const basePath = writeJsonToTempFile(baseJson, "base");
    auto const draftPath = writeJsonToTempFile(draftJson, "draft");

    SpecDecodeDraftingConfig drafting{};
    drafting.draftingTopK = 1;
    drafting.draftingStep = 4;
    drafting.verifySize = 5;

    DeploymentConfig bundle = createDeploymentConfig(
        basePath, std::optional<std::filesystem::path>{draftPath}, std::optional<SpecDecodeDraftingConfig>{drafting});

    EXPECT_EQ(bundle.specDecodeMode(), SpecDecodeMode::kMTP);
    ASSERT_TRUE(bundle.specConfig.has_value());
    EXPECT_EQ(bundle.specConfig->draftingTopK, 1);
    EXPECT_EQ(bundle.specConfig->draftingStep, 4);
    EXPECT_EQ(bundle.specConfig->verifySize, 5);
}

TEST_F(DeploymentConfigTest, MTPBranchingTreeValidatesOk)
{
    // Branching-tree verifySize is independent from draftingStep+1.
    Json const baseJson = makeMTPBaseConfig(/*maxVerify=*/16);
    Json const draftJson = makeMTPDraftConfig(/*maxDraft=*/16);
    auto const basePath = writeJsonToTempFile(baseJson, "base");
    auto const draftPath = writeJsonToTempFile(draftJson, "draft");

    SpecDecodeDraftingConfig drafting{};
    drafting.draftingTopK = 2;
    drafting.draftingStep = 4; // chain input 4 <= 16; candidate pool = 1 + 4*2 = 9
    drafting.verifySize = 8;   // 8 <= 9 and 8 <= 16

    DeploymentConfig bundle = createDeploymentConfig(
        basePath, std::optional<std::filesystem::path>{draftPath}, std::optional<SpecDecodeDraftingConfig>{drafting});

    EXPECT_EQ(bundle.specDecodeMode(), SpecDecodeMode::kMTP);
    ASSERT_TRUE(bundle.specConfig.has_value());
    EXPECT_EQ(bundle.specConfig->draftingTopK, 2);
    EXPECT_EQ(bundle.specConfig->draftingStep, 4);
    EXPECT_EQ(bundle.specConfig->verifySize, 8);
}

TEST_F(DeploymentConfigTest, MTPTreeRejectsTopKNotLessThanVerifySize)
{
    Json const baseJson = makeMTPBaseConfig(/*maxVerify=*/16);
    Json const draftJson = makeMTPDraftConfig(/*maxDraft=*/16);
    auto const basePath = writeJsonToTempFile(baseJson, "base");
    auto const draftPath = writeJsonToTempFile(draftJson, "draft");

    SpecDecodeDraftingConfig drafting{};
    drafting.draftingTopK = 4;
    drafting.draftingStep = 3;
    drafting.verifySize = 4; // topK must be < verifySize (root consumes one node)

    EXPECT_THROW(createDeploymentConfig(basePath, std::optional<std::filesystem::path>{draftPath},
                     std::optional<SpecDecodeDraftingConfig>{drafting}),
        std::runtime_error);
}

TEST_F(DeploymentConfigTest, MTPTreeAllowsVerifySizeBeyondTreeCapacity)
{
    // No tree-capacity bound: verifySize=8 exceeds the full 2-ary depth-2 tree
    // capacity (1+2+4=7); the extra node is well-defined padding at runtime,
    // not a configuration error.
    Json const baseJson = makeMTPBaseConfig(/*maxVerify=*/16);
    Json const draftJson = makeMTPDraftConfig(/*maxDraft=*/16);
    auto const basePath = writeJsonToTempFile(baseJson, "base");
    auto const draftPath = writeJsonToTempFile(draftJson, "draft");

    SpecDecodeDraftingConfig drafting{};
    drafting.draftingTopK = 2;
    drafting.draftingStep = 2;
    drafting.verifySize = 8;

    DeploymentConfig bundle = createDeploymentConfig(
        basePath, std::optional<std::filesystem::path>{draftPath}, std::optional<SpecDecodeDraftingConfig>{drafting});

    EXPECT_EQ(bundle.specDecodeMode(), SpecDecodeMode::kMTP);
    ASSERT_TRUE(bundle.specConfig.has_value());
    EXPECT_EQ(bundle.specConfig->verifySize, 8);
}

TEST_F(DeploymentConfigTest, MTPTreeRejectsFanoutAboveLimit)
{
    Json const baseJson = makeMTPBaseConfig(/*maxVerify=*/16);
    Json const draftJson = makeMTPDraftConfig(/*maxDraft=*/16);
    auto const basePath = writeJsonToTempFile(baseJson, "base");
    auto const draftPath = writeJsonToTempFile(draftJson, "draft");

    SpecDecodeDraftingConfig drafting{};
    drafting.draftingTopK = 9; // > kMTPTreeMaxCandidateFanout (8)
    drafting.draftingStep = 2;
    drafting.verifySize = 12;

    EXPECT_THROW(createDeploymentConfig(basePath, std::optional<std::filesystem::path>{draftPath},
                     std::optional<SpecDecodeDraftingConfig>{drafting}),
        std::runtime_error);
}

TEST_F(DeploymentConfigTest, MTPTreeHybridBaseValidatesOk)
{
    // Hybrid (GDN/causal-conv) base + tree drafting is supported;
    // draftingStep+1 stays within the hybrid intermediate-state depth limit (16).
    Json const baseJson = makeHybridMTPBaseConfig(/*maxVerify=*/16);
    Json const draftJson = makeMTPDraftConfig(/*maxDraft=*/16);
    auto const basePath = writeJsonToTempFile(baseJson, "base");
    auto const draftPath = writeJsonToTempFile(draftJson, "draft");

    SpecDecodeDraftingConfig drafting{};
    drafting.draftingTopK = 2;
    drafting.draftingStep = 3;
    drafting.verifySize = 7; // == candidate pool (1 + 3*2)

    DeploymentConfig bundle = createDeploymentConfig(
        basePath, std::optional<std::filesystem::path>{draftPath}, std::optional<SpecDecodeDraftingConfig>{drafting});

    EXPECT_EQ(bundle.specDecodeMode(), SpecDecodeMode::kMTP);
    EXPECT_GT(bundle.base.numLinearAttnLayers, 0);
    ASSERT_TRUE(bundle.specConfig.has_value());
    EXPECT_EQ(bundle.specConfig->draftingTopK, 2);
    EXPECT_EQ(bundle.specConfig->verifySize, 7);
}

TEST_F(DeploymentConfigTest, HybridMTPRejectsEngineVerifyCapacityAboveCompactLimit)
{
    Json const baseJson = makeHybridMTPBaseConfig(/*maxVerify=*/65);
    Json const draftJson = makeMTPDraftConfig(/*maxDraft=*/16);
    auto const basePath = writeJsonToTempFile(baseJson, "base");
    auto const draftPath = writeJsonToTempFile(draftJson, "draft");

    SpecDecodeDraftingConfig drafting{};
    drafting.draftingTopK = 2;
    drafting.draftingStep = 3;
    drafting.verifySize = 7;

    EXPECT_THROW(createDeploymentConfig(basePath, std::optional<std::filesystem::path>{draftPath},
                     std::optional<SpecDecodeDraftingConfig>{drafting}),
        std::runtime_error);
}

TEST_F(DeploymentConfigTest, MTPTreeRejectsDraftStepAboveDepthLimit)
{
    Json const baseJson = makeMTPBaseConfig(/*maxVerify=*/32);
    Json const draftJson = makeMTPDraftConfig(/*maxDraft=*/32);
    auto const basePath = writeJsonToTempFile(baseJson, "base");
    auto const draftPath = writeJsonToTempFile(draftJson, "draft");

    SpecDecodeDraftingConfig drafting{};
    drafting.draftingTopK = 2;
    drafting.draftingStep = 9; // depth = 9 + 1 = 10 > 9 (EAGLE utility kernel limit)
    drafting.verifySize = 12;

    EXPECT_THROW(createDeploymentConfig(basePath, std::optional<std::filesystem::path>{draftPath},
                     std::optional<SpecDecodeDraftingConfig>{drafting}),
        std::runtime_error);
}

TEST_F(DeploymentConfigTest, MTPFanoutOneTreeRejectsVerifySizeNotDraftStepPlusOne)
{
    Json const baseJson = makeMTPBaseConfig(/*maxVerify=*/16);
    Json const draftJson = makeMTPDraftConfig(/*maxDraft=*/16);
    auto const basePath = writeJsonToTempFile(baseJson, "base");
    auto const draftPath = writeJsonToTempFile(draftJson, "draft");

    SpecDecodeDraftingConfig drafting{};
    drafting.draftingTopK = 1;
    drafting.draftingStep = 4;
    drafting.verifySize = 8;

    EXPECT_THROW(createDeploymentConfig(basePath, std::optional<std::filesystem::path>{draftPath},
                     std::optional<SpecDecodeDraftingConfig>{drafting}),
        std::runtime_error);
}

TEST_F(DeploymentConfigTest, MTPRejectsVerifySizeAboveCurrentEagleUtilityKernelLimit)
{
    Json const baseJson = makeMTPBaseConfig(/*maxVerify=*/16);
    Json const draftJson = makeMTPDraftConfig(/*maxDraft=*/16);
    auto const basePath = writeJsonToTempFile(baseJson, "base");
    auto const draftPath = writeJsonToTempFile(draftJson, "draft");

    SpecDecodeDraftingConfig drafting{};
    drafting.draftingTopK = 1;
    drafting.draftingStep = 15;
    drafting.verifySize = 16;

    EXPECT_THROW(createDeploymentConfig(basePath, std::optional<std::filesystem::path>{draftPath},
                     std::optional<SpecDecodeDraftingConfig>{drafting}),
        std::runtime_error);
}

TEST_F(DeploymentConfigTest, Gemma4MTPRejectsMismatchedSharedKVPoolPages)
{
    Json baseJson = makeGemma4MTPBaseConfig(/*maxVerify=*/4);
    Json const draftJson = makeGemma4MTPDraftConfig(/*maxDraft=*/4);
    baseJson["builder_config"]["max_kv_pool_pages"] = 6;
    auto const basePath = writeJsonToTempFile(baseJson, "base");
    auto const draftPath = writeJsonToTempFile(draftJson, "draft");

    EXPECT_THROW(createDeploymentConfig(basePath, std::optional<std::filesystem::path>{draftPath}, std::nullopt),
        std::runtime_error);
}

TEST_F(DeploymentConfigTest, DFlashHybridVerifySize16ValidatesOk)
{
    Json const baseJson = makeHybridDFlashBaseConfig(/*maxVerify=*/16);
    Json const draftJson = makeDFlashDraftConfig(/*maxDraft=*/16);
    auto const basePath = writeJsonToTempFile(baseJson, "base");
    auto const draftPath = writeJsonToTempFile(draftJson, "draft");

    SpecDecodeDraftingConfig drafting{};
    drafting.draftingTopK = 1;
    drafting.draftingStep = 1;
    drafting.verifySize = 16;

    DeploymentConfig bundle = createDeploymentConfig(
        basePath, std::optional<std::filesystem::path>{draftPath}, std::optional<SpecDecodeDraftingConfig>{drafting});

    EXPECT_EQ(bundle.specDecodeMode(), SpecDecodeMode::kDFlash);
    ASSERT_TRUE(bundle.specConfig.has_value());
    EXPECT_EQ(bundle.specConfig->verifySize, 16);
    EXPECT_EQ(bundle.specConfig->dflashBlockSize, 16);
}

TEST_F(DeploymentConfigTest, DFlashDraftPoolPagesAboveFloorThrows)
{
    Json const baseJson = makeDenseDFlashBaseConfig(/*maxVerify=*/16);
    Json draftJson = makeDFlashDraftConfig(/*maxDraft=*/16);
    draftJson["builder_config"]["max_kv_pool_pages"] = 5;
    auto const basePath = writeJsonToTempFile(baseJson, "base");
    auto const draftPath = writeJsonToTempFile(draftJson, "draft");

    SpecDecodeDraftingConfig drafting{};
    drafting.draftingTopK = 1;
    drafting.draftingStep = 1;
    drafting.verifySize = 16;

    EXPECT_THROW(createDeploymentConfig(basePath, std::optional<std::filesystem::path>{draftPath},
                     std::optional<SpecDecodeDraftingConfig>{drafting}),
        std::runtime_error);
}

TEST_F(DeploymentConfigTest, DFlashDDTreeValidatesOk)
{
    Json const baseJson = makeDenseDFlashBaseConfig(/*maxVerify=*/128);
    Json const draftJson = makeDFlashDraftConfig(/*maxDraft=*/16);
    auto const basePath = writeJsonToTempFile(baseJson, "base");
    auto const draftPath = writeJsonToTempFile(draftJson, "draft");

    SpecDecodeDraftingConfig drafting{};
    drafting.draftingTopK = 4;
    drafting.draftingStep = 1;
    drafting.verifySize = 128;
    drafting.dflashBlockSize = 16;

    DeploymentConfig bundle = createDeploymentConfig(
        basePath, std::optional<std::filesystem::path>{draftPath}, std::optional<SpecDecodeDraftingConfig>{drafting});

    EXPECT_EQ(bundle.specDecodeMode(), SpecDecodeMode::kDFlash);
    ASSERT_TRUE(bundle.specConfig.has_value());
    EXPECT_EQ(bundle.specConfig->verifySize, 128);
    EXPECT_EQ(bundle.specConfig->draftingTopK, 4);
    EXPECT_EQ(bundle.specConfig->dflashBlockSize, 16);
}

TEST_F(DeploymentConfigTest, DFlashDDTreeVerifySizeAboveLimitThrows)
{
    Json const baseJson = makeDenseDFlashBaseConfig(/*maxVerify=*/256);
    Json const draftJson = makeDFlashDraftConfig(/*maxDraft=*/16);
    auto const basePath = writeJsonToTempFile(baseJson, "base");
    auto const draftPath = writeJsonToTempFile(draftJson, "draft");

    SpecDecodeDraftingConfig drafting{};
    drafting.draftingTopK = 4;
    drafting.draftingStep = 1;
    drafting.verifySize = 129;
    drafting.dflashBlockSize = 16;

    EXPECT_THROW(createDeploymentConfig(basePath, std::optional<std::filesystem::path>{draftPath},
                     std::optional<SpecDecodeDraftingConfig>{drafting}),
        std::runtime_error);
}

TEST_F(DeploymentConfigTest, DFlashDDTreeBlockSizeOneThrows)
{
    Json const baseJson = makeDenseDFlashBaseConfig(/*maxVerify=*/16);
    Json const draftJson = makeDFlashDraftConfig(/*maxDraft=*/16);
    auto const basePath = writeJsonToTempFile(baseJson, "base");
    auto const draftPath = writeJsonToTempFile(draftJson, "draft");

    SpecDecodeDraftingConfig drafting{};
    drafting.draftingTopK = 2;
    drafting.draftingStep = 1;
    drafting.verifySize = 8;
    drafting.dflashBlockSize = 1;

    EXPECT_THROW(createDeploymentConfig(basePath, std::optional<std::filesystem::path>{draftPath},
                     std::optional<SpecDecodeDraftingConfig>{drafting}),
        std::runtime_error);
}

TEST_F(DeploymentConfigTest, DFlashNegativeBlockSizeThrows)
{
    Json const baseJson = makeDenseDFlashBaseConfig(/*maxVerify=*/16);
    Json const draftJson = makeDFlashDraftConfig(/*maxDraft=*/16);
    auto const basePath = writeJsonToTempFile(baseJson, "base");
    auto const draftPath = writeJsonToTempFile(draftJson, "draft");

    SpecDecodeDraftingConfig drafting{};
    drafting.draftingTopK = 2;
    drafting.draftingStep = 1;
    drafting.verifySize = 8;
    drafting.dflashBlockSize = -1;

    EXPECT_THROW(createDeploymentConfig(basePath, std::optional<std::filesystem::path>{draftPath},
                     std::optional<SpecDecodeDraftingConfig>{drafting}),
        std::runtime_error);
}

TEST_F(DeploymentConfigTest, DFlashBlockSizeAboveDraftCapacityThrows)
{
    Json const baseJson = makeDenseDFlashBaseConfig(/*maxVerify=*/128);
    Json const draftJson = makeDFlashDraftConfig(/*maxDraft=*/16);
    auto const basePath = writeJsonToTempFile(baseJson, "base");
    auto const draftPath = writeJsonToTempFile(draftJson, "draft");

    SpecDecodeDraftingConfig drafting{};
    drafting.draftingTopK = 4;
    drafting.draftingStep = 1;
    drafting.verifySize = 128;
    drafting.dflashBlockSize = 17;

    EXPECT_THROW(createDeploymentConfig(basePath, std::optional<std::filesystem::path>{draftPath},
                     std::optional<SpecDecodeDraftingConfig>{drafting}),
        std::runtime_error);
}

TEST_F(DeploymentConfigTest, DFlashDDTreeBlockSizeAboveIndexedCommitLimitThrows)
{
    Json const baseJson = makeDenseDFlashBaseConfig(/*maxVerify=*/128);
    Json const draftJson = makeDFlashDraftConfig(/*maxDraft=*/32);
    auto const basePath = writeJsonToTempFile(baseJson, "base");
    auto const draftPath = writeJsonToTempFile(draftJson, "draft");

    SpecDecodeDraftingConfig drafting{};
    drafting.draftingTopK = 4;
    drafting.draftingStep = 1;
    drafting.verifySize = 128;
    drafting.dflashBlockSize = 17;

    EXPECT_THROW(createDeploymentConfig(basePath, std::optional<std::filesystem::path>{draftPath},
                     std::optional<SpecDecodeDraftingConfig>{drafting}),
        std::runtime_error);
}

TEST_F(DeploymentConfigTest, DFlashDDTreeLargeBlockWithBoundedVerifySizeValidatesOk)
{
    Json const baseJson = makeDenseDFlashBaseConfig(/*maxVerify=*/16);
    Json const draftJson = makeDFlashDraftConfig(/*maxDraft=*/32);
    auto const basePath = writeJsonToTempFile(baseJson, "base");
    auto const draftPath = writeJsonToTempFile(draftJson, "draft");

    SpecDecodeDraftingConfig drafting{};
    drafting.draftingTopK = 4;
    drafting.draftingStep = 1;
    drafting.verifySize = 16;
    drafting.dflashBlockSize = 32;

    DeploymentConfig bundle = createDeploymentConfig(
        basePath, std::optional<std::filesystem::path>{draftPath}, std::optional<SpecDecodeDraftingConfig>{drafting});

    ASSERT_TRUE(bundle.specConfig.has_value());
    EXPECT_EQ(bundle.specConfig->verifySize, 16);
    EXPECT_EQ(bundle.specConfig->dflashBlockSize, 32);
}

TEST_F(DeploymentConfigTest, DFlashHybridBlockSizeAbove16Throws)
{
    Json const baseJson = makeHybridDFlashBaseConfig(/*maxVerify=*/128);
    Json const draftJson = makeDFlashDraftConfig(/*maxDraft=*/32);
    auto const basePath = writeJsonToTempFile(baseJson, "base");
    auto const draftPath = writeJsonToTempFile(draftJson, "draft");

    SpecDecodeDraftingConfig drafting{};
    drafting.draftingTopK = 4;
    drafting.draftingStep = 1;
    drafting.verifySize = 128;
    drafting.dflashBlockSize = 17;

    EXPECT_THROW(createDeploymentConfig(basePath, std::optional<std::filesystem::path>{draftPath},
                     std::optional<SpecDecodeDraftingConfig>{drafting}),
        std::runtime_error);
}

TEST_F(DeploymentConfigTest, DFlashChainInfersBlockSizeFromEngineConfig)
{
    Json const baseJson = makeDenseDFlashBaseConfig(/*maxVerify=*/32);
    Json const draftJson = makeDFlashDraftConfig(/*maxDraft=*/32);
    auto const basePath = writeJsonToTempFile(baseJson, "base");
    auto const draftPath = writeJsonToTempFile(draftJson, "draft");

    SpecDecodeDraftingConfig drafting{};
    drafting.draftingTopK = 1;
    drafting.draftingStep = 1;
    drafting.verifySize = 17;

    DeploymentConfig bundle = createDeploymentConfig(
        basePath, std::optional<std::filesystem::path>{draftPath}, std::optional<SpecDecodeDraftingConfig>{drafting});

    ASSERT_TRUE(bundle.specConfig.has_value());
    EXPECT_EQ(bundle.specConfig->draftingTopK, 1);
    EXPECT_EQ(bundle.specConfig->verifySize, 16);
    EXPECT_EQ(bundle.specConfig->dflashBlockSize, 16);
}

TEST_F(DeploymentConfigTest, DFlashCandidateTopKGreaterThanOneSelectsDDTree)
{
    Json const baseJson = makeDenseDFlashBaseConfig(/*maxVerify=*/16);
    Json const draftJson = makeDFlashDraftConfig(/*maxDraft=*/16);
    auto const basePath = writeJsonToTempFile(baseJson, "base");
    auto const draftPath = writeJsonToTempFile(draftJson, "draft");

    SpecDecodeDraftingConfig drafting{};
    drafting.draftingTopK = 2;
    drafting.draftingStep = 1;
    drafting.verifySize = 8;

    DeploymentConfig bundle = createDeploymentConfig(
        basePath, std::optional<std::filesystem::path>{draftPath}, std::optional<SpecDecodeDraftingConfig>{drafting});

    ASSERT_TRUE(bundle.specConfig.has_value());
    EXPECT_EQ(bundle.specConfig->draftingTopK, 2);
    EXPECT_EQ(bundle.specConfig->verifySize, 8);
    EXPECT_EQ(bundle.specConfig->dflashBlockSize, 16);
}

TEST_F(DeploymentConfigTest, DFlashRejectsMultiStep)
{
    Json const baseJson = makeDenseDFlashBaseConfig(/*maxVerify=*/16);
    Json const draftJson = makeDFlashDraftConfig(/*maxDraft=*/16);
    auto const basePath = writeJsonToTempFile(baseJson, "base");
    auto const draftPath = writeJsonToTempFile(draftJson, "draft");

    SpecDecodeDraftingConfig drafting{};
    drafting.draftingTopK = 1;
    drafting.draftingStep = 2;
    drafting.verifySize = 8;

    EXPECT_THROW(createDeploymentConfig(basePath, std::optional<std::filesystem::path>{draftPath},
                     std::optional<SpecDecodeDraftingConfig>{drafting}),
        std::runtime_error);
}

TEST_F(DeploymentConfigTest, DFlashDDTreeCandidateTopKAboveLimitThrows)
{
    Json const baseJson = makeDenseDFlashBaseConfig(/*maxVerify=*/16);
    Json const draftJson = makeDFlashDraftConfig(/*maxDraft=*/16);
    auto const basePath = writeJsonToTempFile(baseJson, "base");
    auto const draftPath = writeJsonToTempFile(draftJson, "draft");

    SpecDecodeDraftingConfig drafting{};
    drafting.draftingTopK = 9;
    drafting.draftingStep = 1;
    drafting.verifySize = 16;

    EXPECT_THROW(createDeploymentConfig(basePath, std::optional<std::filesystem::path>{draftPath},
                     std::optional<SpecDecodeDraftingConfig>{drafting}),
        std::runtime_error);
}

TEST_F(DeploymentConfigTest, DFlashTargetLayerOutOfRangeThrows)
{
    Json const baseJson = makeDenseDFlashBaseConfig(/*maxVerify=*/16);
    Json draftJson = makeDFlashDraftConfig(/*maxDraft=*/16);
    draftJson["dflash_config"]["target_layer_ids"] = Json::array({1, 99});
    auto const basePath = writeJsonToTempFile(baseJson, "base");
    auto const draftPath = writeJsonToTempFile(draftJson, "draft");

    SpecDecodeDraftingConfig drafting{};
    drafting.draftingTopK = 1;
    drafting.draftingStep = 1;
    drafting.verifySize = 8;

    EXPECT_THROW(createDeploymentConfig(basePath, std::optional<std::filesystem::path>{draftPath},
                     std::optional<SpecDecodeDraftingConfig>{drafting}),
        std::runtime_error);
}

TEST_F(DeploymentConfigTest, DFlashBaseDraftModeMismatchThrows)
{
    Json const baseJson = makeDenseDFlashBaseConfig(/*maxVerify=*/16);
    Json const draftJson = makeDraftConfig(/*maxVerify=*/0, /*maxDraft=*/16);
    auto const basePath = writeJsonToTempFile(baseJson, "base");
    auto const draftPath = writeJsonToTempFile(draftJson, "draft");

    SpecDecodeDraftingConfig drafting{};
    drafting.draftingTopK = 1;
    drafting.draftingStep = 1;
    drafting.verifySize = 8;

    EXPECT_THROW(createDeploymentConfig(basePath, std::optional<std::filesystem::path>{draftPath},
                     std::optional<SpecDecodeDraftingConfig>{drafting}),
        std::runtime_error);
}

// ===========================================================================
// maxRuntimeBatchSize()
// ===========================================================================

TEST_F(DeploymentConfigTest, MaxRuntimeBatchSizeVanilla)
{
    // Base only: returns base.maxSupportedBatchSize.
    Json const baseJson = makeBaseConfig(/*maxVerify=*/0, /*maxDraft=*/0, /*maxBatch=*/4);
    auto const basePath = writeJsonToTempFile(baseJson, "base");

    DeploymentConfig bundle = createDeploymentConfig(basePath, std::nullopt, std::nullopt);
    EXPECT_EQ(bundle.maxRuntimeBatchSize(), 4);
}

TEST_F(DeploymentConfigTest, MaxRuntimeBatchSizeBaseAndDraftAgree)
{
    // Base and draft both set the same batch → returns the common value.
    Json const baseJson = makeBaseConfig(/*maxVerify=*/16, /*maxDraft=*/16, /*maxBatch=*/3);
    Json const draftJson = makeDraftConfig(/*maxVerify=*/16, /*maxDraft=*/16, /*maxBatch=*/3);
    auto const basePath = writeJsonToTempFile(baseJson, "base");
    auto const draftPath = writeJsonToTempFile(draftJson, "draft");

    DeploymentConfig bundle
        = createDeploymentConfig(basePath, std::optional<std::filesystem::path>{draftPath}, std::nullopt);
    EXPECT_EQ(bundle.maxRuntimeBatchSize(), 3);
}

TEST_F(DeploymentConfigTest, MaxRuntimeBatchSizeMismatchReturnsMin)
{
    // Base and draft disagree on batch → fall back to the smaller of the two.
    // The runtime cannot drive either engine beyond its engine-declared capacity,
    // so the common ceiling (min) is the safe choice; a warning is logged.
    Json const baseJson = makeBaseConfig(/*maxVerify=*/16, /*maxDraft=*/16, /*maxBatch=*/2);
    Json const draftJson = makeDraftConfig(/*maxVerify=*/16, /*maxDraft=*/16, /*maxBatch=*/8);
    auto const basePath = writeJsonToTempFile(baseJson, "base");
    auto const draftPath = writeJsonToTempFile(draftJson, "draft");

    DeploymentConfig bundle
        = createDeploymentConfig(basePath, std::optional<std::filesystem::path>{draftPath}, std::nullopt);
    EXPECT_EQ(bundle.maxRuntimeBatchSize(), 2);
}

// ===========================================================================
// effectiveMaxDraftProposalSize()
// ===========================================================================

TEST_F(DeploymentConfigTest, EffectiveMaxDraftProposalSizeSpecDecode)
{
    // Both engine maxDraftTreeSize (24) and user verifySize (32) contribute.
    // Expected: max(24, 32) = 32.
    Json const baseJson = makeBaseConfig(/*maxVerify=*/32, /*maxDraft=*/24);
    Json const draftJson = makeDraftConfig(/*maxVerify=*/32, /*maxDraft=*/24);
    auto const basePath = writeJsonToTempFile(baseJson, "base");
    auto const draftPath = writeJsonToTempFile(draftJson, "draft");

    SpecDecodeDraftingConfig drafting{};
    drafting.draftingTopK = 3;
    drafting.draftingStep = 8; // 24
    drafting.verifySize = 32;  // 32

    DeploymentConfig bundle = createDeploymentConfig(
        basePath, std::optional<std::filesystem::path>{draftPath}, std::optional<SpecDecodeDraftingConfig>{drafting});
    EXPECT_EQ(bundle.effectiveMaxDraftProposalSize(), 32);
}

TEST_F(DeploymentConfigTest, EffectiveMaxDraftProposalSizeEngineCapacityWins)
{
    // Engine maxDraftTreeSize (16) is larger than verifySize (8).
    // Expected: max(16, 8) = 16.
    Json const baseJson = makeBaseConfig(/*maxVerify=*/16, /*maxDraft=*/16);
    Json const draftJson = makeDraftConfig(/*maxVerify=*/16, /*maxDraft=*/16);
    auto const basePath = writeJsonToTempFile(baseJson, "base");
    auto const draftPath = writeJsonToTempFile(draftJson, "draft");

    SpecDecodeDraftingConfig drafting{};
    drafting.draftingTopK = 4;
    drafting.draftingStep = 4;
    drafting.verifySize = 8;

    DeploymentConfig bundle = createDeploymentConfig(
        basePath, std::optional<std::filesystem::path>{draftPath}, std::optional<SpecDecodeDraftingConfig>{drafting});
    EXPECT_EQ(bundle.effectiveMaxDraftProposalSize(), 16);
}

TEST_F(DeploymentConfigTest, EffectiveMaxDraftProposalSizeNoDraftingThrows)
{
    // Vanilla bundle: drafting not set → throws.
    Json const baseJson = makeBaseConfig();
    auto const basePath = writeJsonToTempFile(baseJson, "base");

    DeploymentConfig bundle = createDeploymentConfig(basePath, std::nullopt, std::nullopt);
    EXPECT_THROW(bundle.effectiveMaxDraftProposalSize(), std::runtime_error);
}
