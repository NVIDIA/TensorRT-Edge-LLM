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

#include "common/pagedKvTypes.h"
#include "runtime/decoding/dflashDecodeUtils.h"
#include "testUtils.h"
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <limits>
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
    bc["max_kv_pool_pages"] = computeMinimumKvPoolPages(maxBatchSize, 256);
    bc["max_lora_rank"] = 0;
    if (specDecodeMaxVerifyTreeSize > 0)
    {
        config["spec_decode_type"] = "eagle3";
        config["engine_role"] = "base";
        config["eagle_hidden_state_layers"] = {0, 5, 11};
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
    config["eagle3_config"] = {{"target_layer_ids", Json::array()}, {"num_target_layers", 3}};
    config["kv_cache_dtype"] = "fp16";

    Json bc;
    bc["max_batch_size"] = maxBatchSize;
    bc["max_input_len"] = 128;
    bc["max_kv_cache_capacity"] = 256;
    bc["max_kv_pool_pages"] = computeMinimumKvPoolPages(maxBatchSize, 256);
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

Json makeHybridMTPBaseConfig(int32_t maxVerifyTreeSize, int32_t maxBatchSize = 2)
{
    Json config = makeMTPBaseConfig(maxVerifyTreeSize, maxBatchSize);
    config["num_attention_layers"] = 8;
    config["num_linear_attn_layers"] = 4;
    config["recurrent_state_num_heads"] = 4;
    config["recurrent_state_head_dim"] = 64;
    config["recurrent_state_size"] = 64;
    config["conv_dim"] = 768;
    config["conv_kernel"] = 4;
    config["recurrent_state_dtype"] = "fp16";
    config["conv_state_dtype"] = "fp16";
    return config;
}

Json makeHybridEagleBaseConfig(int32_t maxVerifyTreeSize, int32_t maxBatchSize = 2)
{
    Json config = makeHybridMTPBaseConfig(maxVerifyTreeSize, maxBatchSize);
    config["spec_decode_type"] = "eagle3";
    return config;
}

Json makeHybridVanillaConfig(int32_t maxBatchSize = 2)
{
    Json config = makeHybridMTPBaseConfig(/*maxVerifyTreeSize=*/4, maxBatchSize);
    config["spec_decode_type"] = "none";
    config["engine_role"] = "llm";
    config["builder_config"]["spec_base"] = false;
    config["builder_config"].erase("max_verify_tree_size");
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
    config["recurrent_state_head_dim"] = 64;
    config["recurrent_state_size"] = 64;
    config["conv_dim"] = 768;
    config["conv_kernel"] = 4;
    config["recurrent_state_dtype"] = "fp16";
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

Json makeDSparkConfigSection(int32_t blockSize)
{
    return Json{{"block_size", blockSize}, {"mask_token_id", 31999}, {"markov_head_type", "vanilla"},
        {"markov_rank", 256}, {"enable_confidence_head", true}, {"confidence_head_with_markov", true},
        {"target_layer_ids", Json::array({1, 8})}};
}

Json makeDSparkBaseConfig(int32_t maxVerifyTreeSize, int32_t blockSize = 7, int32_t maxBatchSize = 2)
{
    Json config = makeBaseConfig(maxVerifyTreeSize, /*maxDraft=*/0, maxBatchSize);
    config["spec_decode_type"] = "dspark";
    config["engine_role"] = "base";
    config["dspark_config"] = makeDSparkConfigSection(blockSize);
    return config;
}

Json makeDSparkDraftConfig(int32_t maxDraftTreeSize, int32_t blockSize = 7, int32_t maxBatchSize = 2)
{
    Json config = makeDraftConfig(/*maxVerify=*/0, maxDraftTreeSize, maxBatchSize);
    config["spec_decode_type"] = "dspark";
    config["engine_role"] = "draft";
    config["dspark_config"] = makeDSparkConfigSection(blockSize);
    return config;
}

Json makeDenseJetSpecBaseConfig(int32_t maxVerifyTreeSize, int32_t maxBatchSize = 2)
{
    Json config = makeBaseConfig(maxVerifyTreeSize, /*maxDraft=*/0, maxBatchSize);
    config["spec_decode_type"] = "jetspec";
    config["engine_role"] = "base";
    config["jetspec_config"] = Json{{"block_size", 16}, {"mask_token_id", 151669}, {"causal_head", true},
        {"target_layer_ids", Json::array({1, 8})}};
    return config;
}

Json makeJetSpecDraftConfig(int32_t maxDraftTreeSize, int32_t maxBatchSize = 2)
{
    Json config = makeDraftConfig(/*maxVerify=*/0, maxDraftTreeSize, maxBatchSize);
    config["spec_decode_type"] = "jetspec";
    config["engine_role"] = "draft";
    config["base_model_hidden_size"] = config["hidden_size"].get<int32_t>() * 2;
    config["jetspec_config"] = Json{{"block_size", 16}, {"mask_token_id", 151669}, {"causal_head", true},
        {"target_layer_ids", Json::array({1, 8})}};
    return config;
}

//! Gemma4-MTP base config: gemma4_mtp spec type, engine_role=base.
Json makeGemma4MTPBaseConfig(int32_t maxBatchSize = 2, int32_t maxKVCacheCapacity = 256)
{
    Json config = makeBaseConfig(/*maxVerify=*/0, /*maxDraft=*/0, maxBatchSize);
    config["spec_decode_type"] = "gemma4_mtp";
    config["engine_role"] = "base";
    config["model"] = "gemma4_text";
    config["builder_config"]["spec_base"] = true;
    config["builder_config"]["max_verify_tree_size"] = 4;
    config["builder_config"]["max_kv_cache_capacity"] = maxKVCacheCapacity;
    config["builder_config"]["max_kv_pool_pages"] = computeMinimumKvPoolPages(maxBatchSize, maxKVCacheCapacity);
    return config;
}

//! Gemma4-MTP assistant (draft) config: shares the target KV, no own cache.
Json makeGemma4MTPDraftConfig(int32_t maxBatchSize = 2, int32_t maxKVCacheCapacity = 256)
{
    Json config;
    config["spec_decode_type"] = "gemma4_mtp";
    config["engine_role"] = "draft";
    config["model"] = "gemma4_assistant";
    config["num_hidden_layers"] = 1;
    config["num_key_value_heads"] = 4;
    config["head_dim"] = 64;
    config["hidden_size"] = 768;
    config["vocab_size"] = 32000;
    config["base_model_hidden_size"] = 768;
    config["kv_cache_dtype"] = "fp16";
    config["shares_target_kv"] = true;
    config["has_own_kv_cache"] = false;
    config["constant_draft_positions"] = true;
    config["returns_feedback_hidden"] = true;
    config["kv_sharing_map"] = Json::array({Json{{"assistant_layer", 0}, {"target_attention_layer", 0}}});

    Json bc;
    bc["max_batch_size"] = maxBatchSize;
    bc["max_input_len"] = 128;
    bc["max_kv_cache_capacity"] = maxKVCacheCapacity;
    bc["max_kv_pool_pages"] = computeMinimumKvPoolPages(maxBatchSize, maxKVCacheCapacity);
    bc["max_lora_rank"] = 0;
    bc["spec_base"] = false;
    bc["max_draft_tree_size"] = 4;
    config["builder_config"] = bc;
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

TEST_F(DeploymentConfigTest, VanillaAndHybridVanillaSupportCrossRequestRetention)
{
    Json vanilla = makeBaseConfig();
    vanilla["builder_config"]["max_kv_pool_pages"] = 9;
    auto const vanillaPath = writeJsonToTempFile(vanilla, "base");
    EXPECT_EQ(createDeploymentConfig(vanillaPath, std::nullopt, std::nullopt).base.kvPoolPages, 9);

    Json hybrid = makeHybridVanillaConfig();
    hybrid["builder_config"]["max_kv_pool_pages"] = 9;
    auto const hybridPath = writeJsonToTempFile(hybrid, "base");
    EXPECT_EQ(createDeploymentConfig(hybridPath, std::nullopt, std::nullopt).base.kvPoolPages, 9);
}

TEST_F(DeploymentConfigTest, NonHybridEagleSupportsBaseAndDraftCrossRequestRetention)
{
    Json base = makeBaseConfig(/*maxVerify=*/8);
    Json draft = makeDraftConfig(/*maxVerify=*/0, /*maxDraft=*/8);
    base["builder_config"]["max_kv_pool_pages"] = 9;
    draft["builder_config"]["max_kv_pool_pages"] = 10;
    auto const basePath = writeJsonToTempFile(base, "base");
    auto const draftPath = writeJsonToTempFile(draft, "draft");

    DeploymentConfig const deployment
        = createDeploymentConfig(basePath, std::optional<std::filesystem::path>{draftPath}, std::nullopt);
    EXPECT_EQ(deployment.base.kvPoolPages, 9);
    ASSERT_TRUE(deployment.draft.has_value());
    EXPECT_EQ(deployment.draft->kvPoolPages, 10);
}

TEST_F(DeploymentConfigTest, EagleTopologyValidationAppliesToAllDeployments)
{
    Json base = makeBaseConfig(/*maxVerify=*/8);
    Json draft = makeDraftConfig(/*maxVerify=*/0, /*maxDraft=*/8);
    auto validate = [&](Json const& candidateBase, Json const& candidateDraft) {
        auto const basePath = writeJsonToTempFile(candidateBase, "base");
        auto const draftPath = writeJsonToTempFile(candidateDraft, "draft");
        return createDeploymentConfig(basePath, std::optional<std::filesystem::path>{draftPath}, std::nullopt);
    };
    EXPECT_NO_THROW(validate(base, draft));

    base["eagle_hidden_state_layers"] = Json::array();
    EXPECT_THROW(validate(base, draft), std::runtime_error);

    base = makeBaseConfig(/*maxVerify=*/8);
    base["eagle_hidden_state_layers"] = {0, 5, 5};
    EXPECT_THROW(validate(base, draft), std::runtime_error);

    base = makeBaseConfig(/*maxVerify=*/8);
    base["eagle_hidden_state_layers"] = {0, 5, 12};
    EXPECT_THROW(validate(base, draft), std::runtime_error);

    base = makeBaseConfig(/*maxVerify=*/8);
    draft["base_model_hidden_size"] = 768 * 2;
    EXPECT_THROW(validate(base, draft), std::runtime_error);
}

TEST_F(DeploymentConfigTest, RetainedKVPoolSupportIsIndependentOfRuntimeReuseForGemma4MTP)
{
    Json mtpBase = makeMTPBaseConfig(/*maxVerifyTreeSize=*/8);
    Json mtpDraft = makeMTPDraftConfig(/*maxDraftTreeSize=*/8);
    mtpBase["builder_config"]["max_kv_pool_pages"] = 9;
    mtpDraft["builder_config"]["max_kv_pool_pages"] = 10;
    auto const mtpBasePath = writeJsonToTempFile(mtpBase, "base");
    auto const mtpDraftPath = writeJsonToTempFile(mtpDraft, "draft");
    EXPECT_THROW(createDeploymentConfig(mtpBasePath, std::optional<std::filesystem::path>{mtpDraftPath}, std::nullopt),
        std::runtime_error);

    Json gemmaBase = makeGemma4MTPBaseConfig();
    Json gemmaDraft = makeGemma4MTPDraftConfig();
    gemmaBase["builder_config"]["max_kv_pool_pages"] = 9;
    gemmaDraft["builder_config"]["max_kv_pool_pages"] = 9;
    auto const gemmaBasePath = writeJsonToTempFile(gemmaBase, "base");
    auto const gemmaDraftPath = writeJsonToTempFile(gemmaDraft, "draft");
    DeploymentConfig const gemma
        = createDeploymentConfig(gemmaBasePath, std::optional<std::filesystem::path>{gemmaDraftPath}, std::nullopt);
    EXPECT_EQ(gemma.base.kvPoolPages, 9);
    ASSERT_TRUE(gemma.draft.has_value());
    EXPECT_EQ(gemma.draft->kvPoolPages, 9);
}

TEST_F(DeploymentConfigTest, RetainedKVPoolSupportIsIndependentOfRuntimeReuseForDFlash)
{
    Json hybridEagleBase = makeHybridEagleBaseConfig(/*maxVerifyTreeSize=*/8);
    Json const eagleDraft = makeDraftConfig(/*maxVerifyTreeSize=*/0, /*maxDraftTreeSize=*/8);
    hybridEagleBase["builder_config"]["max_kv_pool_pages"] = 9;
    auto const hybridEagleBasePath = writeJsonToTempFile(hybridEagleBase, "base");
    auto const eagleDraftPath = writeJsonToTempFile(eagleDraft, "draft");
    EXPECT_THROW(
        createDeploymentConfig(hybridEagleBasePath, std::optional<std::filesystem::path>{eagleDraftPath}, std::nullopt),
        std::runtime_error);
    Json const dflashBase = makeDenseDFlashBaseConfig(/*maxVerifyTreeSize=*/16);
    Json dflashDraft = makeDFlashDraftConfig(/*maxDraftTreeSize=*/16);
    dflashDraft["builder_config"]["max_kv_pool_pages"] = 9;
    auto const dflashBasePath = writeJsonToTempFile(dflashBase, "base");
    auto const dflashDraftPath = writeJsonToTempFile(dflashDraft, "draft");
    DeploymentConfig const dflash
        = createDeploymentConfig(dflashBasePath, std::optional<std::filesystem::path>{dflashDraftPath}, std::nullopt);
    ASSERT_TRUE(dflash.draft.has_value());
    EXPECT_EQ(dflash.draft->kvPoolPages, 9);
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

TEST_F(DeploymentConfigTest, DiffusionBackboneRejectsDraftingConfig)
{
    Json baseJson = makeBaseConfig();
    baseJson["model"] = "diffusion_gemma_text";
    baseJson["engine_role"] = "dllm";
    baseJson["diffusion_unified_conditioning"] = true;
    baseJson["diffusion_config"] = {{"canvas_length", 8}};
    Json const draftJson = makeDraftConfig(/*maxVerify=*/16, /*maxDraft=*/16);
    auto const basePath = writeJsonToTempFile(baseJson, "base");
    auto const draftPath = writeJsonToTempFile(draftJson, "draft");

    SpecDecodeDraftingConfig drafting{};
    drafting.draftingTopK = 4;
    drafting.draftingStep = 4;
    drafting.verifySize = 8;

    EXPECT_THROW(createDeploymentConfig(basePath, std::optional<std::filesystem::path>{draftPath},
                     std::optional<SpecDecodeDraftingConfig>{drafting}),
        std::runtime_error);
}

TEST_F(DeploymentConfigTest, DiffusionBackboneMaxAcceptedTokensUsesCanvasLength)
{
    Json baseJson = makeBaseConfig();
    baseJson["model"] = "diffusion_gemma_text";
    baseJson["engine_role"] = "dllm";
    baseJson["diffusion_unified_conditioning"] = true;
    baseJson["diffusion_config"] = {{"canvas_length", 8}};
    auto const basePath = writeJsonToTempFile(baseJson, "base");

    DeploymentConfig bundle = createDeploymentConfig(basePath, std::nullopt, std::nullopt);

    EXPECT_EQ(bundle.specDecodeMode(), SpecDecodeMode::kNONE);
    EXPECT_EQ(bundle.maxAcceptedTokensPerRound(), 8);
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

TEST_F(DeploymentConfigTest, DraftingProductOverflowIsRejected)
{
    Json const baseJson = makeBaseConfig(/*maxVerify=*/16, /*maxDraft=*/16);
    Json const draftJson = makeDraftConfig(/*maxVerify=*/16, /*maxDraft=*/std::numeric_limits<int32_t>::max());
    auto const basePath = writeJsonToTempFile(baseJson, "base");
    auto const draftPath = writeJsonToTempFile(draftJson, "draft");

    SpecDecodeDraftingConfig drafting{};
    drafting.draftingTopK = 2;
    drafting.draftingStep = std::numeric_limits<int32_t>::max() / 2 + 1;
    drafting.verifySize = 8;

    EXPECT_THROW(createDeploymentConfig(basePath, std::optional<std::filesystem::path>{draftPath},
                     std::optional<SpecDecodeDraftingConfig>{drafting}),
        std::runtime_error);
}

TEST_F(DeploymentConfigTest, DraftingAcceptedDepthOverflowIsRejected)
{
    Json const baseJson = makeBaseConfig(/*maxVerify=*/16, /*maxDraft=*/16);
    Json const draftJson = makeDraftConfig(/*maxVerify=*/16, /*maxDraft=*/std::numeric_limits<int32_t>::max());
    auto const basePath = writeJsonToTempFile(baseJson, "base");
    auto const draftPath = writeJsonToTempFile(draftJson, "draft");

    SpecDecodeDraftingConfig drafting{};
    drafting.draftingTopK = 1;
    drafting.draftingStep = std::numeric_limits<int32_t>::max();
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

TEST_F(DeploymentConfigTest, MTPLinearChainValidatesOk)
{
    Json const baseJson = makeMTPBaseConfig(/*maxVerify=*/9);
    Json const draftJson = makeMTPDraftConfig(/*maxDraft=*/9);
    auto const basePath = writeJsonToTempFile(baseJson, "base");
    auto const draftPath = writeJsonToTempFile(draftJson, "draft");

    SpecDecodeDraftingConfig drafting{};
    drafting.draftingTopK = 1;
    drafting.draftingStep = 8;
    drafting.verifySize = 9;

    DeploymentConfig bundle = createDeploymentConfig(
        basePath, std::optional<std::filesystem::path>{draftPath}, std::optional<SpecDecodeDraftingConfig>{drafting});

    EXPECT_EQ(bundle.specDecodeMode(), SpecDecodeMode::kMTP);
    ASSERT_TRUE(bundle.specConfig.has_value());
    EXPECT_EQ(bundle.specConfig->draftingTopK, 1);
    EXPECT_EQ(bundle.specConfig->draftingStep, 8);
    EXPECT_EQ(bundle.specConfig->verifySize, 9);
}

TEST_F(DeploymentConfigTest, MTPTopKGreaterThanOneSelectsTree)
{
    // draftingTopK > 1 selects tree drafting: verifySize decouples from
    // draftingStep+1 and only needs to fit within the proposal tree.
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

TEST_F(DeploymentConfigTest, MTPTreeRejectsDraftStepAboveDepthLimit)
{
    Json const baseJson = makeMTPBaseConfig(/*maxVerify=*/32);
    Json const draftJson = makeMTPDraftConfig(/*maxDraft=*/32);
    auto const basePath = writeJsonToTempFile(baseJson, "base");
    auto const draftPath = writeJsonToTempFile(draftJson, "draft");

    SpecDecodeDraftingConfig drafting{};
    drafting.draftingTopK = 2;
    drafting.draftingStep = 16; // depth = 16 + 1 = 17 > kMTPMaxAcceptDepthForCurrentEagleUtilityKernels (16)
    drafting.verifySize = 17;

    EXPECT_THROW(createDeploymentConfig(basePath, std::optional<std::filesystem::path>{draftPath},
                     std::optional<SpecDecodeDraftingConfig>{drafting}),
        std::runtime_error);
}

TEST_F(DeploymentConfigTest, MTPRejectsVerifySizeNotDraftStepPlusOne)
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
    // Engine capacities of 32 keep every other constraint satisfied so the depth check is the
    // only violation: chain verifySize = draftingStep + 1 = 17 implies accept depth 17 > 16.
    Json const baseJson = makeMTPBaseConfig(/*maxVerify=*/32);
    Json const draftJson = makeMTPDraftConfig(/*maxDraft=*/32);
    auto const basePath = writeJsonToTempFile(baseJson, "base");
    auto const draftPath = writeJsonToTempFile(draftJson, "draft");

    SpecDecodeDraftingConfig drafting{};
    drafting.draftingTopK = 1;
    drafting.draftingStep = 16;
    drafting.verifySize = 17;

    EXPECT_THROW(createDeploymentConfig(basePath, std::optional<std::filesystem::path>{draftPath},
                     std::optional<SpecDecodeDraftingConfig>{drafting}),
        std::runtime_error);
}

TEST_F(DeploymentConfigTest, MTPAcceptsDepthAtCurrentEagleUtilityKernelLimit)
{
    // Boundary case: accept depth = draftingStep + 1 = 16 is exactly at the kernel limit.
    Json const baseJson = makeMTPBaseConfig(/*maxVerify=*/16);
    Json const draftJson = makeMTPDraftConfig(/*maxDraft=*/16);
    auto const basePath = writeJsonToTempFile(baseJson, "base");
    auto const draftPath = writeJsonToTempFile(draftJson, "draft");

    SpecDecodeDraftingConfig drafting{};
    drafting.draftingTopK = 1;
    drafting.draftingStep = 15;
    drafting.verifySize = 16;

    auto const bundle = createDeploymentConfig(
        basePath, std::optional<std::filesystem::path>{draftPath}, std::optional<SpecDecodeDraftingConfig>{drafting});

    EXPECT_EQ(bundle.specDecodeMode(), SpecDecodeMode::kMTP);
    ASSERT_TRUE(bundle.specConfig.has_value());
    EXPECT_EQ(bundle.specConfig->draftingStep, 15);
    EXPECT_EQ(bundle.specConfig->verifySize, 16);
}

TEST_F(DeploymentConfigTest, DFlashLinearInfersVerifySizeFromBlockSize)
{
    Json const baseJson = makeHybridDFlashBaseConfig(/*maxVerify=*/16);
    Json const draftJson = makeDFlashDraftConfig(/*maxDraft=*/16);
    auto const basePath = writeJsonToTempFile(baseJson, "base");
    auto const draftPath = writeJsonToTempFile(draftJson, "draft");

    SpecDecodeDraftingConfig drafting{};
    drafting.draftingTopK = 1;
    drafting.draftingStep = 1;
    drafting.verifySize = 0;

    DeploymentConfig bundle = createDeploymentConfig(
        basePath, std::optional<std::filesystem::path>{draftPath}, std::optional<SpecDecodeDraftingConfig>{drafting});

    EXPECT_EQ(bundle.specDecodeMode(), SpecDecodeMode::kDFlash);
    ASSERT_TRUE(bundle.specConfig.has_value());
    EXPECT_EQ(bundle.specConfig->verifySize, 16);
    EXPECT_EQ(bundle.specConfig->dflashBlockSize, 16);
}

TEST_F(DeploymentConfigTest, DFlashLinearVerifySizeEqualBlockSizeValidatesOk)
{
    Json const baseJson = makeDenseDFlashBaseConfig(/*maxVerify=*/16);
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

TEST_F(DeploymentConfigTest, DFlashLinearOverridesCallerVerifySizeToBlockSize)
{
    Json const baseJson = makeDenseDFlashBaseConfig(/*maxVerify=*/32);
    Json const draftJson = makeDFlashDraftConfig(/*maxDraft=*/16);
    auto const basePath = writeJsonToTempFile(baseJson, "base");
    auto const draftPath = writeJsonToTempFile(draftJson, "draft");

    SpecDecodeDraftingConfig drafting{};
    drafting.draftingTopK = 1;
    drafting.draftingStep = 1;
    drafting.verifySize = 17;

    DeploymentConfig bundle = createDeploymentConfig(
        basePath, std::optional<std::filesystem::path>{draftPath}, std::optional<SpecDecodeDraftingConfig>{drafting});

    EXPECT_EQ(bundle.specDecodeMode(), SpecDecodeMode::kDFlash);
    ASSERT_TRUE(bundle.specConfig.has_value());
    EXPECT_EQ(bundle.specConfig->verifySize, 16);
    EXPECT_EQ(bundle.specConfig->dflashBlockSize, 16);
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

TEST_F(DeploymentConfigTest, JetSpecDDTreeValidatesOk)
{
    Json const baseJson = makeDenseJetSpecBaseConfig(/*maxVerify=*/128);
    Json const draftJson = makeJetSpecDraftConfig(/*maxDraft=*/16);
    auto const basePath = writeJsonToTempFile(baseJson, "base");
    auto const draftPath = writeJsonToTempFile(draftJson, "draft");

    SpecDecodeDraftingConfig drafting{};
    drafting.draftingTopK = 4;
    drafting.draftingStep = 1;
    drafting.verifySize = 128;
    drafting.dflashBlockSize = 16;

    DeploymentConfig bundle = createDeploymentConfig(
        basePath, std::optional<std::filesystem::path>{draftPath}, std::optional<SpecDecodeDraftingConfig>{drafting});

    EXPECT_EQ(bundle.specDecodeMode(), SpecDecodeMode::kJetSpec);
    ASSERT_TRUE(bundle.specConfig.has_value());
    ASSERT_TRUE(bundle.draft.has_value());
    EXPECT_EQ(bundle.specConfig->verifySize, 128);
    EXPECT_EQ(bundle.specConfig->draftingTopK, 4);
    EXPECT_EQ(bundle.specConfig->dflashBlockSize, 16);
    EXPECT_EQ(bundle.maxAcceptedTokensPerRound(), 16);
    EXPECT_TRUE(bundle.base.specDraftCausalHead);
    EXPECT_TRUE(bundle.draft->specDraftCausalHead);
}

TEST_F(DeploymentConfigTest, JetSpecDDTreeVerifySizeAboveLimitThrows)
{
    Json const baseJson = makeDenseJetSpecBaseConfig(/*maxVerify=*/256);
    Json const draftJson = makeJetSpecDraftConfig(/*maxDraft=*/16);
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

TEST_F(DeploymentConfigTest, DFlashLinearRuntimeConfigIsBidirectional)
{
    Json baseJson = makeDenseDFlashBaseConfig(/*maxVerify=*/16);
    Json draftJson = makeDFlashDraftConfig(/*maxDraft=*/16);
    baseJson["dflash_config"]["mask_token_id"] = 123;
    draftJson["dflash_config"]["mask_token_id"] = 123;
    auto const basePath = writeJsonToTempFile(baseJson, "base");
    auto const draftPath = writeJsonToTempFile(draftJson, "draft");

    SpecDecodeDraftingConfig drafting{};
    drafting.draftingTopK = 1;
    drafting.draftingStep = 1;
    drafting.verifySize = 0;

    DeploymentConfig bundle = createDeploymentConfig(
        basePath, std::optional<std::filesystem::path>{draftPath}, std::optional<SpecDecodeDraftingConfig>{drafting});

    auto const cfg = dflash_utils::makeCachedBlockDraftRuntimeConfig(bundle);
    EXPECT_EQ(cfg.userMode, SpecDecodeMode::kDFlash);
    EXPECT_EQ(cfg.proposalAttention, dflash_utils::ProposalAttentionPolicy::kBidirectional);
    EXPECT_EQ(cfg.treePolicy, dflash_utils::BlockDraftTreePolicy::kLinear);
    EXPECT_EQ(cfg.blockSize, 16);
    EXPECT_EQ(cfg.verifySize, 16);
    EXPECT_EQ(cfg.proposalLen, 15);
    EXPECT_EQ(cfg.candidateTopK, 1);
    EXPECT_EQ(cfg.maskTokenId, 123);
    EXPECT_EQ(cfg.draftHiddenSize, 768);
    EXPECT_EQ(cfg.baseOutputHiddenDim, 2304);
    EXPECT_EQ(cfg.draftVocabSize, 32000);
}

TEST_F(DeploymentConfigTest, JetSpecDDTreeRuntimeConfigIsCausal)
{
    Json baseJson = makeDenseJetSpecBaseConfig(/*maxVerify=*/128);
    Json draftJson = makeJetSpecDraftConfig(/*maxDraft=*/16);
    baseJson["jetspec_config"]["mask_token_id"] = 123;
    draftJson["jetspec_config"]["mask_token_id"] = 123;
    auto const basePath = writeJsonToTempFile(baseJson, "base");
    auto const draftPath = writeJsonToTempFile(draftJson, "draft");

    SpecDecodeDraftingConfig drafting{};
    drafting.draftingTopK = 4;
    drafting.draftingStep = 1;
    drafting.verifySize = 128;
    drafting.dflashBlockSize = 16;

    DeploymentConfig bundle = createDeploymentConfig(
        basePath, std::optional<std::filesystem::path>{draftPath}, std::optional<SpecDecodeDraftingConfig>{drafting});

    auto const cfg = dflash_utils::makeCachedBlockDraftRuntimeConfig(bundle);
    EXPECT_EQ(cfg.userMode, SpecDecodeMode::kJetSpec);
    EXPECT_EQ(cfg.proposalAttention, dflash_utils::ProposalAttentionPolicy::kCausal);
    EXPECT_EQ(cfg.treePolicy, dflash_utils::BlockDraftTreePolicy::kDDTree);
    EXPECT_EQ(cfg.blockSize, 16);
    EXPECT_EQ(cfg.verifySize, 128);
    EXPECT_EQ(cfg.proposalLen, 16);
    EXPECT_EQ(cfg.candidateTopK, 4);
    EXPECT_EQ(cfg.maskTokenId, 123);
    EXPECT_EQ(cfg.draftHiddenSize, 768);
    EXPECT_EQ(cfg.baseOutputHiddenDim, 1536);
    EXPECT_EQ(cfg.draftVocabSize, 32000);
}

TEST_F(DeploymentConfigTest, JetSpecAcceptsOfficialDFlashConfigFallback)
{
    Json baseJson = makeDenseJetSpecBaseConfig(/*maxVerify=*/16);
    baseJson["dflash_config"] = baseJson["jetspec_config"];
    baseJson.erase("jetspec_config");
    Json draftJson = makeJetSpecDraftConfig(/*maxDraft=*/16);
    draftJson["dflash_config"] = draftJson["jetspec_config"];
    draftJson.erase("jetspec_config");
    auto const basePath = writeJsonToTempFile(baseJson, "base");
    auto const draftPath = writeJsonToTempFile(draftJson, "draft");

    SpecDecodeDraftingConfig drafting{};
    drafting.draftingTopK = 1;
    drafting.draftingStep = 1;
    drafting.verifySize = 0;

    DeploymentConfig bundle = createDeploymentConfig(
        basePath, std::optional<std::filesystem::path>{draftPath}, std::optional<SpecDecodeDraftingConfig>{drafting});

    EXPECT_EQ(bundle.specDecodeMode(), SpecDecodeMode::kJetSpec);
    ASSERT_TRUE(bundle.specConfig.has_value());
    ASSERT_TRUE(bundle.draft.has_value());
    EXPECT_EQ(bundle.specConfig->verifySize, 16);
    EXPECT_EQ(bundle.specConfig->dflashBlockSize, 16);
    EXPECT_EQ(bundle.draft->specDraftMaskTokenId, 151669);
    EXPECT_EQ(bundle.draft->specTargetLayerIds.size(), 2U);
}

TEST_F(DeploymentConfigTest, JetSpecRejectsNonCausalProposalHead)
{
    Json const baseJson = makeDenseJetSpecBaseConfig(/*maxVerify=*/16);
    Json draftJson = makeJetSpecDraftConfig(/*maxDraft=*/16);
    draftJson["jetspec_config"]["causal_head"] = false;
    auto const basePath = writeJsonToTempFile(baseJson, "base");
    auto const draftPath = writeJsonToTempFile(draftJson, "draft");

    SpecDecodeDraftingConfig drafting{};
    drafting.draftingTopK = 1;
    drafting.draftingStep = 1;
    drafting.verifySize = 0;

    EXPECT_THROW(createDeploymentConfig(basePath, std::optional<std::filesystem::path>{draftPath},
                     std::optional<SpecDecodeDraftingConfig>{drafting}),
        std::runtime_error);
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
    Json const draftJson = makeDFlashDraftConfig(/*maxDraft=*/64);
    auto const basePath = writeJsonToTempFile(baseJson, "base");
    auto const draftPath = writeJsonToTempFile(draftJson, "draft");

    SpecDecodeDraftingConfig drafting{};
    drafting.draftingTopK = 4;
    drafting.draftingStep = 1;
    drafting.verifySize = 128;
    drafting.dflashBlockSize = 33;

    EXPECT_THROW(createDeploymentConfig(basePath, std::optional<std::filesystem::path>{draftPath},
                     std::optional<SpecDecodeDraftingConfig>{drafting}),
        std::runtime_error);
}

TEST_F(DeploymentConfigTest, DFlashDDTreeOfficialTreeDepth20ValidatesOk)
{
    Json const baseJson = makeDenseDFlashBaseConfig(/*maxVerify=*/128);
    Json const draftJson = makeDFlashDraftConfig(/*maxDraft=*/21);
    auto const basePath = writeJsonToTempFile(baseJson, "base");
    auto const draftPath = writeJsonToTempFile(draftJson, "draft");

    SpecDecodeDraftingConfig drafting{};
    drafting.draftingTopK = 7;
    drafting.draftingStep = 1;
    drafting.verifySize = 128;
    drafting.dflashBlockSize = 21;

    DeploymentConfig bundle = createDeploymentConfig(
        basePath, std::optional<std::filesystem::path>{draftPath}, std::optional<SpecDecodeDraftingConfig>{drafting});

    ASSERT_TRUE(bundle.specConfig.has_value());
    EXPECT_EQ(bundle.specConfig->verifySize, 128);
    EXPECT_EQ(bundle.specConfig->draftingTopK, 7);
    EXPECT_EQ(bundle.specConfig->dflashBlockSize, 21);
    EXPECT_EQ(bundle.maxAcceptedTokensPerRound(), 21);
}

TEST_F(DeploymentConfigTest, JetSpecDDTreeOfficialTreeDepth20ValidatesOk)
{
    Json const baseJson = makeDenseJetSpecBaseConfig(/*maxVerify=*/128);
    Json const draftJson = makeJetSpecDraftConfig(/*maxDraft=*/21);
    auto const basePath = writeJsonToTempFile(baseJson, "base");
    auto const draftPath = writeJsonToTempFile(draftJson, "draft");

    SpecDecodeDraftingConfig drafting{};
    drafting.draftingTopK = 7;
    drafting.draftingStep = 1;
    drafting.verifySize = 128;
    drafting.dflashBlockSize = 21;

    DeploymentConfig bundle = createDeploymentConfig(
        basePath, std::optional<std::filesystem::path>{draftPath}, std::optional<SpecDecodeDraftingConfig>{drafting});

    ASSERT_TRUE(bundle.specConfig.has_value());
    EXPECT_EQ(bundle.specConfig->verifySize, 128);
    EXPECT_EQ(bundle.specConfig->draftingTopK, 7);
    EXPECT_EQ(bundle.specConfig->dflashBlockSize, 21);
    EXPECT_EQ(bundle.maxAcceptedTokensPerRound(), 21);
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
    drafting.verifySize = 0;

    DeploymentConfig bundle = createDeploymentConfig(
        basePath, std::optional<std::filesystem::path>{draftPath}, std::optional<SpecDecodeDraftingConfig>{drafting});

    ASSERT_TRUE(bundle.specConfig.has_value());
    EXPECT_EQ(bundle.specConfig->draftingTopK, 1);
    EXPECT_EQ(bundle.specConfig->verifySize, 16);
    EXPECT_EQ(bundle.specConfig->dflashBlockSize, 16);
}

TEST_F(DeploymentConfigTest, DFlashLinearBlockSizeOneThrows)
{
    Json const baseJson = makeDenseDFlashBaseConfig(/*maxVerify=*/16);
    Json const draftJson = makeDFlashDraftConfig(/*maxDraft=*/16);
    auto const basePath = writeJsonToTempFile(baseJson, "base");
    auto const draftPath = writeJsonToTempFile(draftJson, "draft");

    SpecDecodeDraftingConfig drafting{};
    drafting.draftingTopK = 1;
    drafting.draftingStep = 1;
    drafting.verifySize = 0;
    drafting.dflashBlockSize = 1;

    EXPECT_THROW(createDeploymentConfig(basePath, std::optional<std::filesystem::path>{draftPath},
                     std::optional<SpecDecodeDraftingConfig>{drafting}),
        std::runtime_error);
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

// The assistant's shared-KV pool/page-table profiles are fixed from its own build limits while the
// runtime binds the TARGET's pool, so identical geometry is a hard contract: equal limits validate.
TEST_F(DeploymentConfigTest, Gemma4MTPMatchingPoolGeometryValidatesOk)
{
    auto const basePath = writeJsonToTempFile(makeGemma4MTPBaseConfig(/*maxBatch=*/4, /*maxCap=*/256), "base");
    auto const draftPath = writeJsonToTempFile(makeGemma4MTPDraftConfig(/*maxBatch=*/4, /*maxCap=*/256), "draft");

    EXPECT_NO_THROW(createDeploymentConfig(basePath, std::optional<std::filesystem::path>{draftPath}, std::nullopt));
}

// Differing max batch sizes imply different fixed pool page counts -> must be rejected with the
// geometry error, not a late TensorRT profile failure.
TEST_F(DeploymentConfigTest, Gemma4MTPMismatchedMaxBatchThrows)
{
    auto const basePath = writeJsonToTempFile(makeGemma4MTPBaseConfig(/*maxBatch=*/4, /*maxCap=*/256), "base");
    auto const draftPath = writeJsonToTempFile(makeGemma4MTPDraftConfig(/*maxBatch=*/2, /*maxCap=*/256), "draft");

    EXPECT_THROW(
        {
            try
            {
                createDeploymentConfig(basePath, std::optional<std::filesystem::path>{draftPath}, std::nullopt);
            }
            catch (std::exception const& e)
            {
                EXPECT_NE(std::string(e.what()).find("pool geometry mismatch"), std::string::npos) << e.what();
                throw;
            }
        },
        std::exception);
}

// Differing KV capacities change both the pool page count and the page-table width -> rejected.
TEST_F(DeploymentConfigTest, Gemma4MTPMismatchedKVCapacityThrows)
{
    auto const basePath = writeJsonToTempFile(makeGemma4MTPBaseConfig(/*maxBatch=*/2, /*maxCap=*/512), "base");
    auto const draftPath = writeJsonToTempFile(makeGemma4MTPDraftConfig(/*maxBatch=*/2, /*maxCap=*/256), "draft");

    EXPECT_THROW(
        {
            try
            {
                createDeploymentConfig(basePath, std::optional<std::filesystem::path>{draftPath}, std::nullopt);
            }
            catch (std::exception const& e)
            {
                EXPECT_NE(std::string(e.what()).find("pool geometry mismatch"), std::string::npos) << e.what();
                throw;
            }
        },
        std::exception);
}

TEST_F(DeploymentConfigTest, DSparkChainValidatesOk)
{
    auto const basePath = writeJsonToTempFile(makeDSparkBaseConfig(/*maxVerify=*/8), "base");
    auto const draftPath = writeJsonToTempFile(makeDSparkDraftConfig(/*maxDraft=*/7), "draft");

    SpecDecodeDraftingConfig drafting{};
    drafting.draftingTopK = 1;
    drafting.draftingStep = 1;
    drafting.verifySize = 8;

    DeploymentConfig bundle = createDeploymentConfig(
        basePath, std::optional<std::filesystem::path>{draftPath}, std::optional<SpecDecodeDraftingConfig>{drafting});

    EXPECT_EQ(bundle.specDecodeMode(), SpecDecodeMode::kDSpark);
    ASSERT_TRUE(bundle.specConfig.has_value());
    EXPECT_EQ(bundle.specConfig->verifySize, 8);
    EXPECT_EQ(bundle.specConfig->draftingTopK, 1);
}

TEST_F(DeploymentConfigTest, DSparkCandidateTopKGreaterThanOneSelectsTree)
{
    // Tree decouples verifySize (node budget) from block_size + 1.
    auto const basePath = writeJsonToTempFile(makeDSparkBaseConfig(/*maxVerify=*/16), "base");
    auto const draftPath = writeJsonToTempFile(makeDSparkDraftConfig(/*maxDraft=*/7), "draft");

    SpecDecodeDraftingConfig drafting{};
    drafting.draftingTopK = 4;
    drafting.draftingStep = 1;
    drafting.verifySize = 16;

    DeploymentConfig bundle = createDeploymentConfig(
        basePath, std::optional<std::filesystem::path>{draftPath}, std::optional<SpecDecodeDraftingConfig>{drafting});

    EXPECT_EQ(bundle.specDecodeMode(), SpecDecodeMode::kDSpark);
    ASSERT_TRUE(bundle.specConfig.has_value());
    EXPECT_EQ(bundle.specConfig->verifySize, 16);
    EXPECT_EQ(bundle.specConfig->draftingTopK, 4);
}

TEST_F(DeploymentConfigTest, DSparkTreeRejectsTopKNotLessThanVerifySize)
{
    auto const basePath = writeJsonToTempFile(makeDSparkBaseConfig(/*maxVerify=*/8), "base");
    auto const draftPath = writeJsonToTempFile(makeDSparkDraftConfig(/*maxDraft=*/7), "draft");

    SpecDecodeDraftingConfig drafting{};
    drafting.draftingTopK = 8;
    drafting.draftingStep = 1;
    drafting.verifySize = 8;

    EXPECT_THROW(createDeploymentConfig(basePath, std::optional<std::filesystem::path>{draftPath},
                     std::optional<SpecDecodeDraftingConfig>{drafting}),
        std::runtime_error);
}

TEST_F(DeploymentConfigTest, DSparkTreeRejectsFanoutAboveLimit)
{
    auto const basePath = writeJsonToTempFile(makeDSparkBaseConfig(/*maxVerify=*/32), "base");
    auto const draftPath = writeJsonToTempFile(makeDSparkDraftConfig(/*maxDraft=*/9), "draft");

    SpecDecodeDraftingConfig drafting{};
    drafting.draftingTopK = 9;
    drafting.draftingStep = 1;
    drafting.verifySize = 32;

    EXPECT_THROW(createDeploymentConfig(basePath, std::optional<std::filesystem::path>{draftPath},
                     std::optional<SpecDecodeDraftingConfig>{drafting}),
        std::runtime_error);
}

TEST_F(DeploymentConfigTest, DSparkTreeVerifySizeAboveNodeBudgetThrows)
{
    auto const basePath = writeJsonToTempFile(makeDSparkBaseConfig(/*maxVerify=*/256), "base");
    auto const draftPath = writeJsonToTempFile(makeDSparkDraftConfig(/*maxDraft=*/7), "draft");

    SpecDecodeDraftingConfig drafting{};
    drafting.draftingTopK = 4;
    drafting.draftingStep = 1;
    drafting.verifySize = 129;

    EXPECT_THROW(createDeploymentConfig(basePath, std::optional<std::filesystem::path>{draftPath},
                     std::optional<SpecDecodeDraftingConfig>{drafting}),
        std::runtime_error);
}

TEST_F(DeploymentConfigTest, DSparkTreeAllowsScheduler)
{
    // Tree mode supports confidence scheduling: scheduled depths shrink the DDTree
    // verify budget (dynamic window), so threshold/sps both validate.
    auto const basePath = writeJsonToTempFile(makeDSparkBaseConfig(/*maxVerify=*/16), "base");
    auto const draftPath = writeJsonToTempFile(makeDSparkDraftConfig(/*maxDraft=*/7), "draft");

    SpecDecodeDraftingConfig drafting{};
    drafting.draftingTopK = 4;
    drafting.draftingStep = 1;
    drafting.verifySize = 16;
    drafting.dsparkSchedulerMode = DSparkSchedulerMode::kThreshold;
    drafting.dsparkConfidenceThreshold = 0.5F;

    DeploymentConfig bundle = createDeploymentConfig(
        basePath, std::optional<std::filesystem::path>{draftPath}, std::optional<SpecDecodeDraftingConfig>{drafting});
    ASSERT_TRUE(bundle.specConfig.has_value());
    EXPECT_EQ(bundle.specConfig->dsparkSchedulerMode, DSparkSchedulerMode::kThreshold);
}

TEST_F(DeploymentConfigTest, DSparkTreeRejectsSPS)
{
    // Fixed tree budget has no verify cost for SPS to trade; only threshold applies.
    auto const basePath = writeJsonToTempFile(makeDSparkBaseConfig(/*maxVerify=*/16), "base");
    auto const draftPath = writeJsonToTempFile(makeDSparkDraftConfig(/*maxDraft=*/7), "draft");

    SpecDecodeDraftingConfig drafting{};
    drafting.draftingTopK = 4;
    drafting.draftingStep = 1;
    drafting.verifySize = 16;
    drafting.dsparkSchedulerMode = DSparkSchedulerMode::kSPS;

    EXPECT_THROW(createDeploymentConfig(basePath, std::optional<std::filesystem::path>{draftPath},
                     std::optional<SpecDecodeDraftingConfig>{drafting}),
        std::runtime_error);
}

TEST_F(DeploymentConfigTest, DSparkTreeAcceptedPathAboveCommitLimitThrows)
{
    // block_size 16 -> accepted path min(16 + 1, 32) = 17 > 16.
    auto const basePath = writeJsonToTempFile(makeDSparkBaseConfig(/*maxVerify=*/32, /*blockSize=*/16), "base");
    auto const draftPath = writeJsonToTempFile(makeDSparkDraftConfig(/*maxDraft=*/16, /*blockSize=*/16), "draft");

    SpecDecodeDraftingConfig drafting{};
    drafting.draftingTopK = 4;
    drafting.draftingStep = 1;
    drafting.verifySize = 32;

    EXPECT_THROW(createDeploymentConfig(basePath, std::optional<std::filesystem::path>{draftPath},
                     std::optional<SpecDecodeDraftingConfig>{drafting}),
        std::runtime_error);
}

TEST_F(DeploymentConfigTest, DSparkTreeLargeBlockWithBoundedVerifySizeValidatesOk)
{
    // Same block_size 16, but verifySize 16 caps the accepted path at 16.
    auto const basePath = writeJsonToTempFile(makeDSparkBaseConfig(/*maxVerify=*/16, /*blockSize=*/16), "base");
    auto const draftPath = writeJsonToTempFile(makeDSparkDraftConfig(/*maxDraft=*/16, /*blockSize=*/16), "draft");

    SpecDecodeDraftingConfig drafting{};
    drafting.draftingTopK = 4;
    drafting.draftingStep = 1;
    drafting.verifySize = 16;

    DeploymentConfig bundle = createDeploymentConfig(
        basePath, std::optional<std::filesystem::path>{draftPath}, std::optional<SpecDecodeDraftingConfig>{drafting});

    ASSERT_TRUE(bundle.specConfig.has_value());
    EXPECT_EQ(bundle.specConfig->verifySize, 16);
}

TEST_F(DeploymentConfigTest, DSparkTreeBlockAboveDraftCapacityThrows)
{
    // Tree drafts the full block (7), which must fit the draft engine profile (6).
    auto const basePath = writeJsonToTempFile(makeDSparkBaseConfig(/*maxVerify=*/16), "base");
    auto const draftPath = writeJsonToTempFile(makeDSparkDraftConfig(/*maxDraft=*/6), "draft");

    SpecDecodeDraftingConfig drafting{};
    drafting.draftingTopK = 4;
    drafting.draftingStep = 1;
    drafting.verifySize = 16;

    EXPECT_THROW(createDeploymentConfig(basePath, std::optional<std::filesystem::path>{draftPath},
                     std::optional<SpecDecodeDraftingConfig>{drafting}),
        std::runtime_error);
}

TEST_F(DeploymentConfigTest, DSparkRejectsMultiStep)
{
    auto const basePath = writeJsonToTempFile(makeDSparkBaseConfig(/*maxVerify=*/8), "base");
    auto const draftPath = writeJsonToTempFile(makeDSparkDraftConfig(/*maxDraft=*/7), "draft");

    SpecDecodeDraftingConfig drafting{};
    drafting.draftingTopK = 1;
    drafting.draftingStep = 2;
    drafting.verifySize = 8;

    EXPECT_THROW(createDeploymentConfig(basePath, std::optional<std::filesystem::path>{draftPath},
                     std::optional<SpecDecodeDraftingConfig>{drafting}),
        std::runtime_error);
}

TEST_F(DeploymentConfigTest, DSparkChainVerifySizeAboveUtilityKernelLimitThrows)
{
    auto const basePath = writeJsonToTempFile(makeDSparkBaseConfig(/*maxVerify=*/32, /*blockSize=*/32), "base");
    auto const draftPath = writeJsonToTempFile(makeDSparkDraftConfig(/*maxDraft=*/32, /*blockSize=*/32), "draft");

    SpecDecodeDraftingConfig drafting{};
    drafting.draftingTopK = 1;
    drafting.draftingStep = 1;
    drafting.verifySize = 18;

    EXPECT_THROW(createDeploymentConfig(basePath, std::optional<std::filesystem::path>{draftPath},
                     std::optional<SpecDecodeDraftingConfig>{drafting}),
        std::runtime_error);
}

TEST_F(DeploymentConfigTest, DSparkTreeSurvivalThresholdOfOneThrows)
{
    auto const basePath = writeJsonToTempFile(makeDSparkBaseConfig(/*maxVerify=*/16), "base");
    auto const draftPath = writeJsonToTempFile(makeDSparkDraftConfig(/*maxDraft=*/7), "draft");

    SpecDecodeDraftingConfig drafting{};
    drafting.draftingTopK = 4;
    drafting.draftingStep = 1;
    drafting.verifySize = 16;
    drafting.dsparkSchedulerMode = DSparkSchedulerMode::kThreshold;
    drafting.dsparkConfidenceThreshold = 1.0F;

    EXPECT_THROW(createDeploymentConfig(basePath, std::optional<std::filesystem::path>{draftPath},
                     std::optional<SpecDecodeDraftingConfig>{drafting}),
        std::runtime_error);
}
