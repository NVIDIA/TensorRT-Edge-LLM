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
#include "kernels/gdnKernels/gdnTreeChunkKernels.h"
#include "runtime/decoding/dflashDecodeUtils.h"
#include "testUtils.h"
#include <filesystem>
#include <fstream>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <limits>
#include <nlohmann/json.hpp>
#include <string>

using namespace trt_edgellm;
using namespace trt_edgellm::rt;
using Json = nlohmann::json;

namespace
{

//! Base config JSON with optional SpecDecode fields. If `specDecodeMaxVerifyTreeSize`
//! is > 0, the config enables SpecDecode and writes `max_spec_verify_tokens`.
//!
//! Note: `max_spec_draft_tokens` is draft-only and is not written on the base
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
    bc["ragged_backend"] = "entry_padded_compatibility";
    if (specDecodeMaxVerifyTreeSize > 0)
    {
        config["spec_decode_type"] = "eagle3";
        config["engine_role"] = "base";
        config["eagle_hidden_state_layers"] = {0, 5, 11};
        bc["spec_base"] = true;
        bc["max_spec_verify_tokens"] = specDecodeMaxVerifyTreeSize;
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
//! Note: `max_spec_verify_tokens` is base-only and is not written on the draft
//! side (the builder only emits it when `specDecodeBase` is set). The
//! `maxSpecVerifyTokens` parameter is accepted for parallelism with
//! `makeBaseConfig` but intentionally ignored here.
Json makeDraftConfig(int32_t /*maxSpecVerifyTokens*/, int32_t maxSpecDraftTokens, int32_t maxBatchSize = 2)
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
    bc["max_spec_draft_tokens"] = maxSpecDraftTokens;
    bc["ragged_backend"] = "entry_padded_compatibility";
    config["builder_config"] = bc;
    return config;
}

Json makeMTPBaseConfig(int32_t maxSpecVerifyTokens, int32_t maxBatchSize = 2)
{
    Json config = makeBaseConfig(maxSpecVerifyTokens, /*maxDraft=*/0, maxBatchSize);
    config["spec_decode_type"] = "mtp";
    config["engine_role"] = "base";
    return config;
}

Json makeMTPDraftConfig(int32_t maxSpecDraftTokens, int32_t maxBatchSize = 2)
{
    Json config = makeDraftConfig(/*maxVerify=*/0, maxSpecDraftTokens, maxBatchSize);
    config["spec_decode_type"] = "mtp";
    config["engine_role"] = "draft";
    config["base_model_hidden_size"] = config["hidden_size"];
    return config;
}

void enableMRope(Json& config, float ropeTheta = 1000000.0F)
{
    config["rope_scaling"] = {{"rope_type", "mrope"}, {"mrope_section", Json::array({16, 24, 24})}};
    config["rope_theta"] = ropeTheta;
    config["partial_rotary_factor"] = 1.0F;
}

void enableDualRope(Json& config)
{
    config["sliding_rope_config"] = Json::object();
    config["full_rope_config"] = Json::object();
    config["sliding_rotary_dim"] = config["head_dim"];
    config["full_rotary_dim"] = config["head_dim"];
}

Json makeHybridMTPBaseConfig(int32_t maxSpecVerifyTokens, int32_t maxBatchSize = 2)
{
    Json config = makeMTPBaseConfig(maxSpecVerifyTokens, maxBatchSize);
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

Json makeHybridEagleBaseConfig(int32_t maxSpecVerifyTokens, int32_t maxBatchSize = 2)
{
    Json config = makeHybridMTPBaseConfig(maxSpecVerifyTokens, maxBatchSize);
    config["spec_decode_type"] = "eagle3";
    return config;
}

Json makeHybridVanillaConfig(int32_t maxBatchSize = 2)
{
    Json config = makeHybridMTPBaseConfig(/*maxSpecVerifyTokens=*/4, maxBatchSize);
    config["spec_decode_type"] = "none";
    config["engine_role"] = "llm";
    config["builder_config"]["spec_base"] = false;
    config["builder_config"].erase("max_spec_verify_tokens");
    return config;
}

Json makeHybridDFlashBaseConfig(int32_t maxSpecVerifyTokens, int32_t maxBatchSize = 2)
{
    Json config = makeBaseConfig(maxSpecVerifyTokens, /*maxDraft=*/0, maxBatchSize);
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

Json makeDenseDFlashBaseConfig(int32_t maxSpecVerifyTokens, int32_t maxBatchSize = 2)
{
    Json config = makeBaseConfig(maxSpecVerifyTokens, /*maxDraft=*/0, maxBatchSize);
    config["spec_decode_type"] = "dflash";
    config["engine_role"] = "base";
    config["dflash_config"]
        = Json{{"block_size", 16}, {"mask_token_id", 248070}, {"target_layer_ids", Json::array({1, 8})}};
    return config;
}

Json makeDFlashDraftConfig(int32_t maxSpecDraftTokens, int32_t maxBatchSize = 2)
{
    Json config = makeDraftConfig(/*maxVerify=*/0, maxSpecDraftTokens, maxBatchSize);
    config["spec_decode_type"] = "dflash";
    config["engine_role"] = "draft";
    config["dflash_config"]
        = Json{{"block_size", 16}, {"mask_token_id", 248070}, {"target_layer_ids", Json::array({1, 8})}};
    return config;
}

Json makeDFlash2ConfigSection()
{
    return Json{{"version", 2}, {"block_size", 8}, {"mask_token_id", 248070}, {"is_causal", false},
        {"conv_kernel_size", 2}, {"conv_group_size", 16}, {"selector_rank", 256}, {"selector_top_k", 16},
        {"supports_probabilistic_sampling", true}, {"target_layer_ids", Json::array({1, 3, 5, 8, 11})}};
}

Json makeDFlash2BaseConfig(int32_t maxSpecVerifyTokens, int32_t maxBatchSize = 2)
{
    Json config = makeBaseConfig(maxSpecVerifyTokens, /*maxDraft=*/0, maxBatchSize);
    config["spec_decode_type"] = "dflash";
    config["engine_role"] = "base";
    config["dflash_config"] = makeDFlash2ConfigSection();
    return config;
}

Json makeDFlash2DraftConfig(int32_t maxSpecDraftTokens, int32_t maxBatchSize = 2)
{
    Json config = makeDraftConfig(/*maxVerify=*/0, maxSpecDraftTokens, maxBatchSize);
    config["spec_decode_type"] = "dflash";
    config["engine_role"] = "draft";
    config["num_hidden_layers"] = 5;
    config["base_model_hidden_size"] = config["hidden_size"].get<int32_t>() * 5;
    config["dflash_config"] = makeDFlash2ConfigSection();
    return config;
}

//! The `dflash_config` the LiLiCorr exporter writes for the draft and base runtime configs.
Json makeLiLiCorrConfigSection()
{
    return Json{{"version", 2}, {"target_layer_ids", Json::array({1, 18, 35, 51, 68, 85})}, {"block_size", 8},
        {"mask_token_id", 990}, {"is_causal", false}, {"conv_kernel_size", 2}, {"conv_group_size", 16},
        {"supports_probabilistic_sampling", true}, {"projector_type", "lilicorr"}, {"lilicorr_candidate_topk", 8}};
}

constexpr int32_t kLiLiCorrHiddenSize{4096};

Json makeLiLiCorrBaseConfig(int32_t maxSpecVerifyTokens)
{
    Json config = makeDFlash2BaseConfig(maxSpecVerifyTokens);
    // Target layer 85 must lie inside the base decoder stack.
    config["num_hidden_layers"] = 88;
    config["hidden_size"] = kLiLiCorrHiddenSize;
    config["dflash_config"] = makeLiLiCorrConfigSection();
    return config;
}

Json makeLiLiCorrDraftConfig(int32_t maxSpecDraftTokens)
{
    Json config = makeDFlash2DraftConfig(maxSpecDraftTokens);
    config["hidden_size"] = kLiLiCorrHiddenSize;
    config["base_model_hidden_size"] = 6 * kLiLiCorrHiddenSize;
    config["dflash_config"] = makeLiLiCorrConfigSection();
    return config;
}

//! The message createDeploymentConfig rejects the pair with, or "<accepted>".
std::string deploymentRejection(
    std::filesystem::path const& basePath, std::filesystem::path const& draftPath, SpecDecodeDraftingConfig drafting)
{
    try
    {
        (void) createDeploymentConfig(basePath, std::optional<std::filesystem::path>{draftPath},
            std::optional<SpecDecodeDraftingConfig>{drafting});
    }
    catch (std::runtime_error const& error)
    {
        return error.what();
    }
    return "<accepted>";
}

Json makeDSparkConfigSection(int32_t blockSize)
{
    return Json{{"block_size", blockSize}, {"mask_token_id", 31999}, {"markov_head_type", "vanilla"},
        {"markov_rank", 256}, {"enable_confidence_head", true}, {"confidence_head_with_markov", true},
        {"target_layer_ids", Json::array({1, 8})}};
}

Json makeDSparkBaseConfig(int32_t maxSpecVerifyTokens, int32_t blockSize = 7, int32_t maxBatchSize = 2)
{
    Json config = makeBaseConfig(maxSpecVerifyTokens, /*maxDraft=*/0, maxBatchSize);
    config["spec_decode_type"] = "dspark";
    config["engine_role"] = "base";
    config["dspark_config"] = makeDSparkConfigSection(blockSize);
    return config;
}

Json makeDSparkDraftConfig(int32_t maxSpecDraftTokens, int32_t blockSize = 7, int32_t maxBatchSize = 2)
{
    Json config = makeDraftConfig(/*maxVerify=*/0, maxSpecDraftTokens, maxBatchSize);
    config["spec_decode_type"] = "dspark";
    config["engine_role"] = "draft";
    config["dspark_config"] = makeDSparkConfigSection(blockSize);
    return config;
}

Json makeDenseJetSpecBaseConfig(int32_t maxSpecVerifyTokens, int32_t maxBatchSize = 2)
{
    Json config = makeBaseConfig(maxSpecVerifyTokens, /*maxDraft=*/0, maxBatchSize);
    config["spec_decode_type"] = "jetspec";
    config["engine_role"] = "base";
    config["jetspec_config"] = Json{{"block_size", 16}, {"mask_token_id", 151669}, {"causal_head", true},
        {"target_layer_ids", Json::array({1, 8})}};
    return config;
}

Json makeJetSpecDraftConfig(int32_t maxSpecDraftTokens, int32_t maxBatchSize = 2)
{
    Json config = makeDraftConfig(/*maxVerify=*/0, maxSpecDraftTokens, maxBatchSize);
    config["spec_decode_type"] = "jetspec";
    config["engine_role"] = "draft";
    config["base_model_hidden_size"] = config["hidden_size"].get<int32_t>() * 2;
    config["jetspec_config"] = Json{{"block_size", 16}, {"mask_token_id", 151669}, {"causal_head", true},
        {"target_layer_ids", Json::array({1, 8})}};
    return config;
}

//! Gemma4-MTP base config: gemma4_mtp spec type, engine_role=base.
Json makeGemma4MTPBaseConfig(
    int32_t maxBatchSize = 2, int32_t maxKVCacheCapacity = 256, int32_t maxSpecVerifyTokens = 4)
{
    Json config = makeBaseConfig(/*maxVerify=*/0, /*maxDraft=*/0, maxBatchSize);
    config["spec_decode_type"] = "gemma4_mtp";
    config["engine_role"] = "base";
    config["model"] = "gemma4_text";
    config["builder_config"]["spec_base"] = true;
    config["builder_config"]["max_spec_verify_tokens"] = maxSpecVerifyTokens;
    config["builder_config"]["max_kv_cache_capacity"] = maxKVCacheCapacity;
    config["builder_config"]["max_kv_pool_pages"] = computeMinimumKvPoolPages(maxBatchSize, maxKVCacheCapacity);
    return config;
}

//! Gemma4-MTP assistant (draft) config: shares the target KV, no own cache.
Json makeGemma4MTPDraftConfig(
    int32_t maxBatchSize = 2, int32_t maxKVCacheCapacity = 256, int32_t maxSpecDraftTokens = 4)
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
    bc["max_spec_draft_tokens"] = maxSpecDraftTokens;
    bc["ragged_backend"] = "entry_padded_compatibility";
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

// Given an EAGLE draft whose candidate head emits 8 candidates per row, a base built for 64 verify tokens and a draft
// engine built for 128 draft tokens
// When the tree is K 4, D 3, N 8 (within every limit), or exceeds one limit: an accepted path of 17 tokens (K 2, D 16,
// N 17), a padded tree of 65 nodes (K 5, D 13, N 16), more verify tokens than the tree holds (K 4, D 1, N 6), or
// draftTopK above the candidate count (K 9, D 2, N 8)
// Then the first resolves and each of the others is rejected, in order by the accepted-path limit of 16 tokens, the
// EAGLE tree limit of 64 nodes, the draft tree's capacity of 1 + K + (D - 1) x K^2 nodes and the candidate head's
// num_candidates
TEST_F(DeploymentConfigTest, RejectsCandidateTreesPastTheEagleLimits)
{
    auto const basePath = writeJsonToTempFile(makeBaseConfig(/*maxVerify=*/64), "base");
    Json draft = makeDraftConfig(/*maxVerify=*/0, /*maxDraft=*/128);
    draft["eagle3_config"]["candidate_head"] = {{"type", "low_rank"}, {"rank", 16}, {"num_candidates", 8}};
    auto const draftPath = writeJsonToTempFile(draft, "draft");
    auto const create = [&](int32_t topK, int32_t steps, int32_t verify) {
        SpecDecodeDraftingConfig drafting{};
        drafting.draftTopK = topK;
        drafting.draftingStep = steps;
        drafting.verifyTokens = verify;
        return createDeploymentConfig(basePath, std::optional<std::filesystem::path>{draftPath}, drafting);
    };

    EXPECT_NO_THROW(create(4, 3, 8));
    EXPECT_THROW(create(2, 16, 17), std::runtime_error);
    EXPECT_THROW(create(5, 13, 16), std::runtime_error);
    EXPECT_THROW(create(4, 1, 6), std::runtime_error);
    EXPECT_THROW(create(9, 2, 8), std::runtime_error);
}

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

TEST_F(DeploymentConfigTest, BoundedSwaSelectionFallsBackForUnsupportedRuntimeModes)
{
    DeploymentConfig deployment;
    deployment.base.maxKVCacheCapacity = 256;
    deployment.base.kvPoolPages = 8;
    deployment.base.numSwaPages = 4;
    deployment.base.kvLayerConfigs = {KVLayerConfig{/*numKVHeads=*/4, /*headDim=*/64, /*kvCacheCapacity=*/128}};

    deployment.selectSwaKVCacheMode(/*contextReuseEnabled=*/false);
    EXPECT_TRUE(deployment.base.usesBoundedSwaKVCache());

    deployment.base.kvCacheDtype = trt_edgellm::DataType::kFP8;
    deployment.selectSwaKVCacheMode(/*contextReuseEnabled=*/false);
    EXPECT_TRUE(deployment.base.usesBoundedSwaKVCache());

    deployment.selectSwaKVCacheMode(/*contextReuseEnabled=*/true);
    EXPECT_FALSE(deployment.base.usesBoundedSwaKVCache());

    deployment.base.kvCacheDtype = trt_edgellm::DataType::kHALF;
    deployment.base.specDecodeType = SpecDecodeMode::kEAGLE;
    deployment.selectSwaKVCacheMode(/*contextReuseEnabled=*/false);
    EXPECT_FALSE(deployment.base.usesBoundedSwaKVCache());
}

// Given a vanilla FP16 deployment whose bounded SWA page budget is smaller than its full page budget,
// When the SWA storage mode is selected with context reuse enabled,
// Then full SWA storage is selected, and bounded storage is selected once context reuse is disabled.
TEST_F(DeploymentConfigTest, ContextReuseSelectsFullSwaStorage)
{
    DeploymentConfig deployment;
    deployment.base.maxKVCacheCapacity = 256;
    deployment.base.kvPoolPages = 8;
    deployment.base.numSwaPages = 4;
    deployment.base.kvLayerConfigs = {KVLayerConfig{/*numKVHeads=*/4, /*headDim=*/64, /*kvCacheCapacity=*/128}};

    deployment.selectSwaKVCacheMode(/*contextReuseEnabled=*/true);
    EXPECT_FALSE(deployment.base.usesBoundedSwaKVCache());

    deployment.selectSwaKVCacheMode(/*contextReuseEnabled=*/false);
    EXPECT_TRUE(deployment.base.usesBoundedSwaKVCache());
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
    Json mtpBase = makeMTPBaseConfig(/*maxSpecVerifyTokens=*/8);
    Json mtpDraft = makeMTPDraftConfig(/*maxSpecDraftTokens=*/8);
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
    Json hybridEagleBase = makeHybridEagleBaseConfig(/*maxSpecVerifyTokens=*/8);
    Json const eagleDraft = makeDraftConfig(/*maxSpecVerifyTokens=*/0, /*maxSpecDraftTokens=*/8);
    hybridEagleBase["builder_config"]["max_kv_pool_pages"] = 9;
    auto const hybridEagleBasePath = writeJsonToTempFile(hybridEagleBase, "base");
    auto const eagleDraftPath = writeJsonToTempFile(eagleDraft, "draft");
    DeploymentConfig const hybridEagle = createDeploymentConfig(
        hybridEagleBasePath, std::optional<std::filesystem::path>{eagleDraftPath}, std::nullopt);
    EXPECT_EQ(hybridEagle.base.kvPoolPages, 9);
    Json const dflashBase = makeDenseDFlashBaseConfig(/*maxSpecVerifyTokens=*/16);
    Json dflashDraft = makeDFlashDraftConfig(/*maxSpecDraftTokens=*/16);
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
    drafting.draftTopK = 4;
    drafting.draftingStep = 4; // draftSteps * draftTopK = 16 <= 16
    drafting.verifyTokens = 8; // 8 <= 16

    DeploymentConfig bundle = createDeploymentConfig(
        basePath, std::optional<std::filesystem::path>{draftPath}, std::optional<SpecDecodeDraftingConfig>{drafting});

    EXPECT_TRUE(bundle.base.isSpecDecodeBase);
    ASSERT_TRUE(bundle.draft.has_value());
    // Draft engines set isSpecDecodeBase=false (they ARE the draft, not the base).
    // Presence of a draft engine is indicated by maxSpecDraftTokens > 0.
    EXPECT_FALSE(bundle.draft->isSpecDecodeBase);
    EXPECT_GT(bundle.draft->maxSpecDraftTokens, 0);
    ASSERT_TRUE(bundle.specConfig.has_value());

    EXPECT_EQ(bundle.base.maxSpecVerifyTokens, 16);
    // `maxSpecDraftTokens` is only meaningful on the draft side (see makeBaseConfig).
    EXPECT_EQ(bundle.base.maxSpecDraftTokens, 0);
    // `maxSpecVerifyTokens` is only meaningful on the base side (see makeDraftConfig).
    EXPECT_EQ(bundle.draft->maxSpecVerifyTokens, 0);
    EXPECT_EQ(bundle.draft->maxSpecDraftTokens, 16);
    EXPECT_EQ(bundle.specConfig->verifyTokenCount, 8);
    EXPECT_EQ(bundle.specConfig->draftSteps, 4);
    EXPECT_EQ(bundle.specConfig->draftTopK, 4);
}

TEST_F(DeploymentConfigTest, EagleAllowsProviderVerifyBudget)
{
    Json const baseJson = makeBaseConfig(/*maxVerify=*/64);
    Json const draftJson = makeDraftConfig(/*maxVerify=*/0, /*maxDraft=*/64);
    auto const basePath = writeJsonToTempFile(baseJson, "base");
    auto const draftPath = writeJsonToTempFile(draftJson, "draft");

    SpecDecodeDraftingConfig drafting{};
    drafting.draftTopK = 2;
    drafting.draftingStep = 2;
    // Given an EAGLE provider checkpoint with a larger verification window,
    // when its deployment config is resolved, then the provider-owned window is preserved.
    drafting.verifyTokens = 8;

    DeploymentConfig bundle = createDeploymentConfig(
        basePath, std::optional<std::filesystem::path>{draftPath}, std::optional<SpecDecodeDraftingConfig>{drafting});

    ASSERT_TRUE(bundle.specConfig.has_value());
    EXPECT_EQ(bundle.specConfig->verifyTokenCount, 8);
}

TEST_F(DeploymentConfigTest, EagleRejectsVerifyWindowShorterThanMaximumAcceptDepth)
{
    // Given EAGLE drafting whose maximum accepted path exceeds its verification window,
    // when the deployment config is resolved, then the inconsistent topology is rejected.
    Json const baseJson = makeBaseConfig(/*maxVerify=*/64);
    Json const draftJson = makeDraftConfig(/*maxVerify=*/0, /*maxDraft=*/64);
    auto const basePath = writeJsonToTempFile(baseJson, "base");
    auto const draftPath = writeJsonToTempFile(draftJson, "draft");

    SpecDecodeDraftingConfig drafting{};
    drafting.draftTopK = 2;
    drafting.draftingStep = 8;
    drafting.verifyTokens = 8;

    EXPECT_THROW(createDeploymentConfig(basePath, std::optional<std::filesystem::path>{draftPath}, drafting),
        std::runtime_error);
}

namespace
{
//! An EAGLE bundle from the given engine capacities and drafting shape.
DeploymentConfig makeEagleBundle(
    int32_t maxVerifyTokens, int32_t maxDraftTokens, int32_t draftTopK, int32_t draftingStep, int32_t verifyTokens)
{
    auto const basePath = writeJsonToTempFile(makeBaseConfig(maxVerifyTokens), "base");
    auto const draftPath = writeJsonToTempFile(makeDraftConfig(/*maxVerify=*/0, maxDraftTokens), "draft");
    SpecDecodeDraftingConfig drafting{};
    drafting.draftTopK = draftTopK;
    drafting.draftingStep = draftingStep;
    drafting.verifyTokens = verifyTokens;
    return createDeploymentConfig(
        basePath, std::optional<std::filesystem::path>{draftPath}, std::optional<SpecDecodeDraftingConfig>{drafting});
}

//! An MTP bundle from the given engine capacities and drafting shape, on a dense or a hybrid base.
DeploymentConfig makeMTPBundle(int32_t maxVerifyTokens, int32_t maxDraftTokens, int32_t draftTopK, int32_t draftingStep,
    int32_t verifyTokens, bool hybridBase = false)
{
    auto const basePath = writeJsonToTempFile(
        hybridBase ? makeHybridMTPBaseConfig(maxVerifyTokens) : makeMTPBaseConfig(maxVerifyTokens), "base");
    auto const draftPath = writeJsonToTempFile(makeMTPDraftConfig(maxDraftTokens), "draft");
    SpecDecodeDraftingConfig drafting{};
    drafting.draftTopK = draftTopK;
    drafting.draftingStep = draftingStep;
    drafting.verifyTokens = verifyTokens;
    return createDeploymentConfig(
        basePath, std::optional<std::filesystem::path>{draftPath}, std::optional<SpecDecodeDraftingConfig>{drafting});
}

//! An EAGLE bundle whose plain round obeys the EAGLE tree limits: a draft with an eight-candidate head on a dense
//! base, or a plain draft on a hybrid base. Both engines take 128 tokens.
DeploymentConfig makeTreeLimitedEagleBundle(
    bool hybridBase, int32_t draftTopK, int32_t draftingStep, int32_t verifyTokens)
{
    constexpr int32_t kCapacity{128};
    auto const basePath
        = writeJsonToTempFile(hybridBase ? makeHybridEagleBaseConfig(kCapacity) : makeBaseConfig(kCapacity), "base");
    Json draft = makeDraftConfig(/*maxVerify=*/0, kCapacity);
    if (!hybridBase)
    {
        draft["eagle3_config"]["candidate_head"] = {{"type", "low_rank"}, {"rank", 16}, {"num_candidates", 8}};
    }
    auto const draftPath = writeJsonToTempFile(draft, "draft");
    SpecDecodeDraftingConfig drafting{};
    drafting.draftTopK = draftTopK;
    drafting.draftingStep = draftingStep;
    drafting.verifyTokens = verifyTokens;
    return createDeploymentConfig(
        basePath, std::optional<std::filesystem::path>{draftPath}, std::optional<SpecDecodeDraftingConfig>{drafting});
}
} // namespace

// Given an EAGLE base and an MTP tree base whose max_spec_verify_tokens equal the verify width
// When each deployment is resolved
// Then neither can jump: no jump tokens, and a grafting round has the plain round's verify width and path capacity
TEST_F(DeploymentConfigTest, TreesWithoutVerifyHeadroomCannotJumpForward)
{
    DeploymentConfig const eagle = makeEagleBundle(
        /*maxVerifyTokens=*/8, /*maxDraftTokens=*/16, /*draftTopK=*/2, /*draftingStep=*/3, /*verifyTokens=*/8);
    DeploymentConfig const mtp = makeMTPBundle(
        /*maxVerifyTokens=*/8, /*maxDraftTokens=*/16, /*draftTopK=*/2, /*draftingStep=*/3, /*verifyTokens=*/8);

    ASSERT_TRUE(eagle.specConfig.has_value());
    EXPECT_EQ(eagle.specConfig->maxJumpTokens, 0);
    EXPECT_EQ(eagle.specConfig->jumpVerifyTokenCount, 8);
    EXPECT_EQ(eagle.specConfig->jumpMaxPathLength, 4);
    EXPECT_EQ(eagle.maxAcceptedTokensPerRound(), 4);
    ASSERT_TRUE(mtp.specConfig.has_value());
    EXPECT_EQ(mtp.specConfig->maxJumpTokens, 0);
    EXPECT_EQ(mtp.specConfig->jumpVerifyTokenCount, 8);
    EXPECT_EQ(mtp.specConfig->jumpMaxPathLength, 4);
    EXPECT_EQ(mtp.maxAcceptedTokensPerRound(), 4);
}

// Given an EAGLE base with eight verify tokens of headroom and a draft wide enough for the longer accepted path, and
//      EAGLE deployments capped in turn by the verify headroom, the accepted-path limit, the verify width limit and
//      the draft proposal width
// When each deployment is resolved
// Then the first's jump may add eight tokens, and its grafting round verifies V + 8 nodes and accepts up to nine
//      tokens; every grafting round verifies the plain width plus the jump tokens, accepts the whole of a jump plus
//      the bonus, and never exceeds the commit kernels' path bound of 32 nodes
TEST_F(DeploymentConfigTest, EagleVerifyHeadroomSetsTheJumpLimitsAndEveryCapKeepsTheGraftingRoundInvariants)
{
    DeploymentConfig const bundle = makeEagleBundle(
        /*maxVerifyTokens=*/16, /*maxDraftTokens=*/64, /*draftTopK=*/2, /*draftingStep=*/3, /*verifyTokens=*/8);

    ASSERT_TRUE(bundle.specConfig.has_value());
    EXPECT_EQ(bundle.specConfig->maxJumpTokens, 8);
    EXPECT_EQ(bundle.specConfig->jumpVerifyTokenCount, 16);
    EXPECT_EQ(bundle.specConfig->jumpMaxPathLength, 9);
    EXPECT_EQ(bundle.maxAcceptedTokensPerRound(), 9);

    struct Shape
    {
        int32_t maxVerifyTokens;
        int32_t maxDraftTokens;
    };
    for (Shape const shape : {Shape{8, 16}, Shape{16, 16}, Shape{64, 64}, Shape{200, 200}, Shape{64, 8}})
    {
        DeploymentConfig const capped
            = makeEagleBundle(shape.maxVerifyTokens, shape.maxDraftTokens, /*draftTopK=*/2, /*draftingStep=*/3, 8);
        ASSERT_TRUE(capped.specConfig.has_value());
        auto const& spec = *capped.specConfig;
        EXPECT_EQ(spec.jumpVerifyTokenCount - spec.verifyTokenCount, spec.maxJumpTokens) << shape.maxVerifyTokens;
        EXPECT_GE(spec.jumpMaxPathLength, spec.maxJumpTokens + 1) << shape.maxVerifyTokens;
        EXPECT_LE(spec.jumpMaxPathLength, 32) << shape.maxVerifyTokens;
        EXPECT_LE(spec.jumpVerifyTokenCount, spec.maxVerifyTokens) << shape.maxVerifyTokens;
    }
}

// Given EAGLE deployments whose jump is bounded in turn by the commit kernels' 32-entry accepted path, by the verify
//      kernels' 128-node limit under a base verifying 100 nodes, and by a draft proposal width of eight tokens
// When each deployment is resolved
// Then the jumps add 31, 28 and 7 tokens: the first grafting round verifies V + 31 nodes and accepts up to 32 tokens,
//      the second verifies 128 nodes, and the third accepts a path of eight, which the draft re-reads in one proposal
TEST_F(DeploymentConfigTest, EagleJumpIsCappedByTheAcceptedPathLimitTheVerifyWidthLimitAndTheDraftProposalWidth)
{
    DeploymentConfig const byPath = makeEagleBundle(
        /*maxVerifyTokens=*/256, /*maxDraftTokens=*/64, /*draftTopK=*/2, /*draftingStep=*/3, /*verifyTokens=*/8);
    DeploymentConfig const byVerifyWidth = makeEagleBundle(
        /*maxVerifyTokens=*/256, /*maxDraftTokens=*/64, /*draftTopK=*/8, /*draftingStep=*/3, /*verifyTokens=*/100);
    DeploymentConfig const byProposalWidth = makeEagleBundle(
        /*maxVerifyTokens=*/64, /*maxDraftTokens=*/8, /*draftTopK=*/2, /*draftingStep=*/3, /*verifyTokens=*/8);

    ASSERT_TRUE(byPath.specConfig.has_value());
    EXPECT_EQ(byPath.specConfig->maxJumpTokens, 31);
    EXPECT_EQ(byPath.specConfig->jumpVerifyTokenCount, 39);
    EXPECT_EQ(byPath.specConfig->jumpMaxPathLength, 32);
    ASSERT_TRUE(byVerifyWidth.specConfig.has_value());
    EXPECT_EQ(byVerifyWidth.specConfig->maxJumpTokens, 28);
    EXPECT_EQ(byVerifyWidth.specConfig->jumpVerifyTokenCount, 128);
    EXPECT_EQ(byVerifyWidth.specConfig->jumpMaxPathLength, 29);
    ASSERT_TRUE(byProposalWidth.specConfig.has_value());
    EXPECT_EQ(byProposalWidth.specConfig->maxJumpTokens, 7);
    EXPECT_EQ(byProposalWidth.specConfig->jumpVerifyTokenCount, 15);
    EXPECT_EQ(byProposalWidth.specConfig->jumpMaxPathLength, 8);
    EXPECT_EQ(byProposalWidth.maxAcceptedTokensPerRound(), 8);
}

// Given EAGLE deployments whose plain round obeys the EAGLE tree limits of 64 nodes and 16 accepted tokens (a draft
//      with a candidate head on a dense base, and a plain draft on a hybrid base), engines taking 128 tokens, and
//      top-4 drafting over three steps (V = 8) or top-8 drafting over three steps (V = 60)
// When each deployment is resolved
// Then a grafting round keeps those limits: with V = 8 it accepts up to 16 tokens, so it verifies V + 15 nodes, and
//      with V = 60 it verifies 64 nodes, so a jump adds up to 4 tokens
TEST_F(DeploymentConfigTest, EagleJumpKeepsTheEagleTreeLimitsOfACandidateDraftOrAHybridBase)
{
    for (bool const hybridBase : {false, true})
    {
        SCOPED_TRACE(hybridBase ? "hybrid base" : "candidate-head draft");
        DeploymentConfig const byPath
            = makeTreeLimitedEagleBundle(hybridBase, /*draftTopK=*/4, /*draftingStep=*/3, /*verifyTokens=*/8);
        DeploymentConfig const byNodes
            = makeTreeLimitedEagleBundle(hybridBase, /*draftTopK=*/8, /*draftingStep=*/3, /*verifyTokens=*/60);

        ASSERT_TRUE(byPath.specConfig.has_value());
        EXPECT_EQ(byPath.specConfig->maxJumpTokens, 15);
        EXPECT_EQ(byPath.specConfig->jumpVerifyTokenCount, 23);
        EXPECT_EQ(byPath.specConfig->jumpMaxPathLength, 16);
        ASSERT_TRUE(byNodes.specConfig.has_value());
        EXPECT_EQ(byNodes.specConfig->maxJumpTokens, 4);
        EXPECT_EQ(byNodes.specConfig->jumpVerifyTokenCount, 64);
        EXPECT_EQ(byNodes.specConfig->jumpMaxPathLength, 5);
        for (DeploymentConfig const* bundle : {&byPath, &byNodes})
        {
            EXPECT_LE(bundle->specConfig->jumpVerifyTokenCount, 64);
            EXPECT_LE(bundle->specConfig->jumpMaxPathLength, 16);
        }
    }
}

// Given an MTP tree of top-4 drafting over four steps (V = 60), on a dense base and on a hybrid base that both
//      verify 128 tokens, and a draft taking 32
// When each deployment is resolved
// Then the dense grafting round verifies V + 15 = 75 nodes, while the hybrid one stops at the GDN tree-chunk limit of
//      64 nodes, so its jump adds at most 4 tokens
TEST_F(DeploymentConfigTest, MTPTreeJumpOnAHybridBaseStaysWithinTheTreeChunkNodeLimit)
{
    DeploymentConfig const dense = makeMTPBundle(
        /*maxVerifyTokens=*/128, /*maxDraftTokens=*/32, /*draftTopK=*/4, /*draftingStep=*/4, /*verifyTokens=*/60);
    DeploymentConfig const hybrid = makeMTPBundle(/*maxVerifyTokens=*/128, /*maxDraftTokens=*/32, /*draftTopK=*/4,
        /*draftingStep=*/4, /*verifyTokens=*/60, /*hybridBase=*/true);

    ASSERT_TRUE(dense.specConfig.has_value());
    EXPECT_EQ(dense.specConfig->jumpVerifyTokenCount, 75);
    EXPECT_EQ(dense.specConfig->maxJumpTokens, 15);
    ASSERT_TRUE(hybrid.specConfig.has_value());
    EXPECT_EQ(hybrid.specConfig->jumpVerifyTokenCount, kernel::kGDN_TREE_CHUNK_MAX_NODES);
    EXPECT_EQ(hybrid.specConfig->maxJumpTokens, kernel::kGDN_TREE_CHUNK_MAX_NODES - 60);
    EXPECT_EQ(hybrid.specConfig->jumpMaxPathLength, 5);
}

// Given MTP chains of three draft steps (V = 4) and MTP trees of top-2 drafting over three steps (V = 8), over engines
//      whose verify and draft capacities range from the plain width to 128 tokens
// When each deployment is resolved
// Then every grafting round verifies between V and max_spec_verify_tokens tokens and accepts at most 16; a chain that
//      can jump is one chain no longer than the draft takes, so its round accepts all of it, and a draft that cannot
//      take V tokens leaves the chain plain; a tree widens only into the base's verify headroom, by at most the jump,
//      and does not jump when the draft cannot re-read its path in one proposal
TEST_F(DeploymentConfigTest, MTPJumpLimitsKeepTheGraftingRoundWithinTheEngineAndKernelLimits)
{
    struct Case
    {
        int32_t maxVerifyTokens;
        int32_t maxDraftTokens;
        int32_t jumpVerifyTokenCount;
        int32_t jumpMaxPathLength;
        int32_t maxJumpTokens;
    };
    // Chains: V = 4, so a chain grows to min(max_spec_verify_tokens, 16, max_spec_draft_tokens) tokens.
    for (Case const c : {Case{4, 4, 4, 4, 3}, Case{24, 8, 8, 8, 7}, Case{8, 24, 8, 8, 7}, Case{128, 128, 16, 16, 15},
             Case{8, 3, 4, 4, 0}})
    {
        SCOPED_TRACE("chain " + std::to_string(c.maxVerifyTokens) + "/" + std::to_string(c.maxDraftTokens));
        DeploymentConfig const bundle = makeMTPBundle(c.maxVerifyTokens, c.maxDraftTokens, /*draftTopK=*/1,
            /*draftingStep=*/3, /*verifyTokens=*/4);
        ASSERT_TRUE(bundle.specConfig.has_value());
        auto const& spec = *bundle.specConfig;
        EXPECT_EQ(spec.jumpVerifyTokenCount, c.jumpVerifyTokenCount);
        EXPECT_EQ(spec.jumpMaxPathLength, c.jumpMaxPathLength);
        EXPECT_EQ(spec.maxJumpTokens, c.maxJumpTokens);
        EXPECT_GE(spec.jumpVerifyTokenCount, spec.verifyTokenCount);
        EXPECT_LE(spec.jumpVerifyTokenCount, spec.maxVerifyTokens);
        EXPECT_EQ(spec.jumpMaxPathLength, spec.jumpVerifyTokenCount);
        EXPECT_LE(spec.jumpMaxPathLength, 16);
        if (spec.maxJumpTokens > 0)
        {
            EXPECT_LE(spec.jumpVerifyTokenCount, spec.maxDraftTokens);
            EXPECT_EQ(spec.maxJumpTokens, spec.jumpVerifyTokenCount - 1);
        }
    }
    // Trees: V = 8 and S + 1 = 4. Without verify headroom, or with a draft too narrow for the path, a grafting round
    // is the plain one.
    for (Case const c :
        {Case{8, 16, 8, 4, 0}, Case{128, 128, 23, 16, 15}, Case{128, 4, 11, 4, 3}, Case{128, 3, 8, 4, 0}})
    {
        SCOPED_TRACE("tree " + std::to_string(c.maxVerifyTokens) + "/" + std::to_string(c.maxDraftTokens));
        DeploymentConfig const bundle = makeMTPBundle(c.maxVerifyTokens, c.maxDraftTokens, /*draftTopK=*/2,
            /*draftingStep=*/3, /*verifyTokens=*/8);
        ASSERT_TRUE(bundle.specConfig.has_value());
        auto const& spec = *bundle.specConfig;
        EXPECT_EQ(spec.jumpVerifyTokenCount, c.jumpVerifyTokenCount);
        EXPECT_EQ(spec.jumpMaxPathLength, c.jumpMaxPathLength);
        EXPECT_EQ(spec.maxJumpTokens, c.maxJumpTokens);
        EXPECT_EQ(spec.jumpVerifyTokenCount - spec.verifyTokenCount, spec.maxJumpTokens);
        EXPECT_LE(spec.jumpVerifyTokenCount, spec.maxVerifyTokens);
        EXPECT_LE(spec.jumpMaxPathLength, 16);
        if (spec.maxJumpTokens > 0)
        {
            EXPECT_GE(spec.jumpMaxPathLength, spec.maxJumpTokens + 1);
            EXPECT_LE(spec.jumpMaxPathLength, spec.maxDraftTokens);
        }
    }
}

namespace
{
//! A cached-block bundle over the given engine configs, drafting the largest block the checkpoint and the draft allow,
//! as a linear chain (draftTopK 1) or as a DDTree of `verifyTokens` nodes.
DeploymentConfig makeCachedBlockBundle(Json const& base, Json const& draft, int32_t draftTopK, int32_t verifyTokens)
{
    auto const basePath = writeJsonToTempFile(base, "base");
    auto const draftPath = writeJsonToTempFile(draft, "draft");
    SpecDecodeDraftingConfig drafting{};
    drafting.draftTopK = draftTopK;
    drafting.verifyTokens = verifyTokens;
    return createDeploymentConfig(
        basePath, std::optional<std::filesystem::path>{draftPath}, std::optional<SpecDecodeDraftingConfig>{drafting});
}
} // namespace

// Given a DFlash linear deployment drafting a block of 16 (V = 16) on a base that verifies 32 tokens, and a DFlash2
//      deployment drafting a block of 8 (V = 8) on a base that verifies 8, each with a draft taking its block
// When each deployment is resolved
// Then the verify window stays the drafted block whatever the base's headroom: a grafting round verifies and accepts
//      the block, a jump replaces all but one of its tokens, and a decode round accepts at most the block
TEST_F(DeploymentConfigTest, CachedBlockLinearJumpReplacesDraftTokensWithinTheBlock)
{
    struct Case
    {
        char const* name;
        Json base;
        Json draft;
        int32_t blockSize;
    };
    for (Case const& c : {Case{"DFlash", makeDenseDFlashBaseConfig(/*maxSpecVerifyTokens=*/32),
                              makeDFlashDraftConfig(/*maxSpecDraftTokens=*/16), 16},
             Case{"DFlash2", makeDFlash2BaseConfig(/*maxSpecVerifyTokens=*/8),
                 makeDFlash2DraftConfig(/*maxSpecDraftTokens=*/8), 8}})
    {
        SCOPED_TRACE(c.name);
        DeploymentConfig const bundle
            = makeCachedBlockBundle(c.base, c.draft, /*draftTopK=*/1, /*verifyTokens=*/c.blockSize);

        ASSERT_TRUE(bundle.specConfig.has_value());
        EXPECT_EQ(bundle.specConfig->draftBlockSize, c.blockSize);
        EXPECT_EQ(bundle.specConfig->jumpVerifyTokenCount, c.blockSize);
        EXPECT_EQ(bundle.specConfig->jumpMaxPathLength, c.blockSize);
        EXPECT_EQ(bundle.specConfig->maxJumpTokens, c.blockSize - 1);
        EXPECT_EQ(bundle.maxAcceptedTokensPerRound(), c.blockSize);
    }
}

// Given a DFlash DDTree of V = 16 nodes over a block of 16 on a base that verifies 32 tokens, and a JetSpec DDTree of
//      V = 32 nodes over a block of 16 on a base that verifies 64, both top-4 with a draft taking 16
// When each deployment is resolved
// Then a grafting round accepts at most the one block the next draft forward takes as its delta, 16 tokens: it
//      verifies V + 15 nodes (31 and 47), a jump adds up to 15 tokens, and a decode round accepts at most 16 tokens
TEST_F(DeploymentConfigTest, CachedBlockDDTreeJumpWidensIntoTheVerifyHeadroomWithinOneBlock)
{
    DeploymentConfig const dflash = makeCachedBlockBundle(makeDenseDFlashBaseConfig(/*maxSpecVerifyTokens=*/32),
        makeDFlashDraftConfig(/*maxSpecDraftTokens=*/16), /*draftTopK=*/4, /*verifyTokens=*/16);
    DeploymentConfig const jetspec = makeCachedBlockBundle(makeDenseJetSpecBaseConfig(/*maxSpecVerifyTokens=*/64),
        makeJetSpecDraftConfig(/*maxSpecDraftTokens=*/16), /*draftTopK=*/4, /*verifyTokens=*/32);

    ASSERT_TRUE(dflash.specConfig.has_value());
    EXPECT_EQ(dflash.specConfig->jumpVerifyTokenCount, 31);
    EXPECT_EQ(dflash.specConfig->jumpMaxPathLength, 16);
    EXPECT_EQ(dflash.specConfig->maxJumpTokens, 15);
    EXPECT_EQ(dflash.maxAcceptedTokensPerRound(), 16);
    ASSERT_TRUE(jetspec.specConfig.has_value());
    EXPECT_EQ(jetspec.specConfig->jumpVerifyTokenCount, 47);
    EXPECT_EQ(jetspec.specConfig->jumpMaxPathLength, 16);
    EXPECT_EQ(jetspec.specConfig->maxJumpTokens, 15);
    EXPECT_EQ(jetspec.maxAcceptedTokensPerRound(), 16);
}

// Given DFlash DDTrees of V = 56 nodes over a block of 16, on a dense and on a hybrid base that both verify 128 tokens,
//      with a draft taking 16
// When each deployment is resolved
// Then the dense grafting round verifies V + 15 = 71 nodes, while the hybrid one stops at the GDN tree-chunk limit of
//      64 nodes, so its jump adds at most 8 tokens; both accept at most one block of 16 tokens
TEST_F(DeploymentConfigTest, CachedBlockDDTreeJumpOnAHybridBaseStaysWithinTheTreeChunkNodeLimit)
{
    DeploymentConfig const dense = makeCachedBlockBundle(makeDenseDFlashBaseConfig(/*maxSpecVerifyTokens=*/128),
        makeDFlashDraftConfig(/*maxSpecDraftTokens=*/16), /*draftTopK=*/4, /*verifyTokens=*/56);
    DeploymentConfig const hybrid = makeCachedBlockBundle(makeHybridDFlashBaseConfig(/*maxSpecVerifyTokens=*/128),
        makeDFlashDraftConfig(/*maxSpecDraftTokens=*/16), /*draftTopK=*/4, /*verifyTokens=*/56);

    ASSERT_TRUE(dense.specConfig.has_value());
    EXPECT_EQ(dense.specConfig->jumpVerifyTokenCount, 71);
    EXPECT_EQ(dense.specConfig->maxJumpTokens, 15);
    EXPECT_EQ(dense.specConfig->jumpMaxPathLength, 16);
    ASSERT_TRUE(hybrid.specConfig.has_value());
    EXPECT_EQ(hybrid.specConfig->jumpVerifyTokenCount, kernel::kGDN_TREE_CHUNK_MAX_NODES);
    EXPECT_EQ(hybrid.specConfig->maxJumpTokens, kernel::kGDN_TREE_CHUNK_MAX_NODES - 56);
    EXPECT_EQ(hybrid.specConfig->jumpMaxPathLength, 16);
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
    drafting.draftTopK = 4;
    drafting.draftingStep = 4;
    drafting.verifyTokens = 8;

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
    drafting.draftTopK = 4;
    drafting.draftingStep = 4;
    drafting.verifyTokens = 8;

    EXPECT_THROW(createDeploymentConfig(basePath, std::nullopt, std::optional<SpecDecodeDraftingConfig>{drafting}),
        std::runtime_error);
}

TEST_F(DeploymentConfigTest, DraftingExceedsVerifyCapacityThrows)
{
    // User's verifyTokenCount > base.maxSpecVerifyTokens → throws.
    Json const baseJson = makeBaseConfig(/*maxVerify=*/8, /*maxDraft=*/16);
    Json const draftJson = makeDraftConfig(/*maxVerify=*/8, /*maxDraft=*/16);
    auto const basePath = writeJsonToTempFile(baseJson, "base");
    auto const draftPath = writeJsonToTempFile(draftJson, "draft");

    SpecDecodeDraftingConfig drafting{};
    drafting.draftTopK = 2;
    drafting.draftingStep = 2;  // 2 * 2 = 4 <= 16 (OK)
    drafting.verifyTokens = 16; // 16 > 8 (violation)

    EXPECT_THROW(createDeploymentConfig(basePath, std::optional<std::filesystem::path>{draftPath},
                     std::optional<SpecDecodeDraftingConfig>{drafting}),
        std::runtime_error);
}

TEST_F(DeploymentConfigTest, DraftingExceedsDraftCapacityThrows)
{
    // User's draftSteps * draftTopK > draft->maxSpecDraftTokens → throws.
    Json const baseJson = makeBaseConfig(/*maxVerify=*/16, /*maxDraft=*/8);
    Json const draftJson = makeDraftConfig(/*maxVerify=*/16, /*maxDraft=*/8);
    auto const basePath = writeJsonToTempFile(baseJson, "base");
    auto const draftPath = writeJsonToTempFile(draftJson, "draft");

    SpecDecodeDraftingConfig drafting{};
    drafting.draftTopK = 4;
    drafting.draftingStep = 4; // 4 * 4 = 16 > 8 (violation)
    drafting.verifyTokens = 8;

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
    drafting.draftTopK = 2;
    drafting.draftingStep = std::numeric_limits<int32_t>::max() / 2 + 1;
    drafting.verifyTokens = 8;

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
    drafting.draftTopK = 1;
    drafting.draftingStep = std::numeric_limits<int32_t>::max();
    drafting.verifyTokens = 8;

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
    drafting.draftTopK = 3;
    drafting.draftingStep = 8;  // 3 * 8 = 24 <= 24
    drafting.verifyTokens = 32; // 32 <= 32

    DeploymentConfig bundle = createDeploymentConfig(
        basePath, std::optional<std::filesystem::path>{draftPath}, std::optional<SpecDecodeDraftingConfig>{drafting});

    EXPECT_TRUE(bundle.base.isSpecDecodeBase);
    EXPECT_TRUE(bundle.draft.has_value());
    ASSERT_TRUE(bundle.specConfig.has_value());
    EXPECT_EQ(bundle.base.maxSpecVerifyTokens, 32);
    EXPECT_EQ(bundle.draft->maxSpecDraftTokens, 24);
}

TEST_F(DeploymentConfigTest, MTPLinearChainValidatesOk)
{
    Json const baseJson = makeMTPBaseConfig(/*maxVerify=*/9);
    Json const draftJson = makeMTPDraftConfig(/*maxDraft=*/9);
    auto const basePath = writeJsonToTempFile(baseJson, "base");
    auto const draftPath = writeJsonToTempFile(draftJson, "draft");

    SpecDecodeDraftingConfig drafting{};
    drafting.draftTopK = 1;
    drafting.draftingStep = 8;
    drafting.verifyTokens = 9;

    DeploymentConfig bundle = createDeploymentConfig(
        basePath, std::optional<std::filesystem::path>{draftPath}, std::optional<SpecDecodeDraftingConfig>{drafting});

    EXPECT_EQ(bundle.specDecodeMode(), SpecDecodeMode::kMTP);
    ASSERT_TRUE(bundle.specConfig.has_value());
    EXPECT_EQ(bundle.specConfig->draftTopK, 1);
    EXPECT_EQ(bundle.specConfig->draftSteps, 8);
    EXPECT_EQ(bundle.specConfig->verifyTokenCount, 9);
}

TEST_F(DeploymentConfigTest, MTPChainResolvesProposalAndRootInclusiveVerifyCounts)
{
    // Given an explicit proposal depth with chain selection and omitted Q,
    // when the deployment is resolved, then Q and the accepted path include
    // exactly one verification root while the proposal count does not.
    Json const baseJson = makeMTPBaseConfig(/*maxVerify=*/9);
    Json const draftJson = makeMTPDraftConfig(/*maxDraft=*/9);
    auto const basePath = writeJsonToTempFile(baseJson, "base");
    auto const draftPath = writeJsonToTempFile(draftJson, "draft");

    SpecDecodeDraftingConfig drafting{};
    drafting.draftTopK = 1;
    drafting.draftingStep = 8;

    DeploymentConfig const bundle = createDeploymentConfig(
        basePath, std::optional<std::filesystem::path>{draftPath}, std::optional<SpecDecodeDraftingConfig>{drafting});

    ASSERT_TRUE(bundle.specConfig.has_value());
    EXPECT_EQ(bundle.specConfig->proposalTokenCount, 8);
    EXPECT_EQ(bundle.specConfig->verifyTokenCount, 9);
    EXPECT_EQ(bundle.specConfig->maxPathLength, 9);
    EXPECT_EQ(bundle.specConfig->verifyLayout, SpecVerifyLayout::kChain);
}

TEST_F(DeploymentConfigTest, IterativeDraftingRejectsExplicitProposalTokens)
{
    // Given an iterative MTP deployment with an explicit drafting step, when proposal tokens are also supplied,
    // then the resolver rejects the second control for the same iteration length.
    Json const baseJson = makeMTPBaseConfig(/*maxVerify=*/9);
    Json const draftJson = makeMTPDraftConfig(/*maxDraft=*/9);
    auto const basePath = writeJsonToTempFile(baseJson, "base");
    auto const draftPath = writeJsonToTempFile(draftJson, "draft");

    for (int32_t const proposalTokens : {1, 3})
    {
        SCOPED_TRACE(proposalTokens);
        SpecDecodeDraftingConfig drafting{};
        drafting.draftTopK = 1;
        drafting.draftingStep = 3;
        drafting.proposalTokens = proposalTokens;

        EXPECT_THROW(createDeploymentConfig(basePath, std::optional<std::filesystem::path>{draftPath}, drafting),
            std::runtime_error);
    }
}

TEST_F(DeploymentConfigTest, BlockDraftingOnlyAllowsOneStep)
{
    // Given a block DFlash deployment, when S is omitted or one it is normalized to one, while any other explicit
    // value is rejected before block geometry is resolved.
    Json const baseJson = makeDenseDFlashBaseConfig(/*maxVerify=*/16);
    Json const draftJson = makeDFlashDraftConfig(/*maxDraft=*/16);
    auto const basePath = writeJsonToTempFile(baseJson, "base");
    auto const draftPath = writeJsonToTempFile(draftJson, "draft");

    SpecDecodeDraftingConfig omitted{};
    omitted.draftTopK = 1;
    DeploymentConfig const omittedBundle
        = createDeploymentConfig(basePath, std::optional<std::filesystem::path>{draftPath}, omitted);
    ASSERT_TRUE(omittedBundle.draftingConfig().has_value());
    EXPECT_EQ(omittedBundle.draftingConfig()->draftingStep, 1);

    SpecDecodeDraftingConfig explicitOne = omitted;
    explicitOne.draftingStep = 1;
    DeploymentConfig const explicitBundle
        = createDeploymentConfig(basePath, std::optional<std::filesystem::path>{draftPath}, explicitOne);
    ASSERT_TRUE(explicitBundle.draftingConfig().has_value());
    EXPECT_EQ(explicitBundle.draftingConfig()->draftTopK, omittedBundle.draftingConfig()->draftTopK);
    EXPECT_EQ(explicitBundle.draftingConfig()->draftingStep, omittedBundle.draftingConfig()->draftingStep);
    EXPECT_EQ(explicitBundle.draftingConfig()->proposalTokens, omittedBundle.draftingConfig()->proposalTokens);
    EXPECT_EQ(explicitBundle.draftingConfig()->verifyTokens, omittedBundle.draftingConfig()->verifyTokens);

    for (int32_t const draftingStep : {0, 2})
    {
        SCOPED_TRACE(draftingStep);
        SpecDecodeDraftingConfig invalid = omitted;
        invalid.draftingStep = draftingStep;
        EXPECT_THROW(createDeploymentConfig(basePath, std::optional<std::filesystem::path>{draftPath}, invalid),
            std::runtime_error);
    }
}

TEST_F(DeploymentConfigTest, MTPTopKGreaterThanOneSelectsTree)
{
    // draftTopK > 1 selects tree drafting: verifyTokenCount decouples from
    // draftSteps+1 and only needs to fit within the proposal tree.
    Json const baseJson = makeMTPBaseConfig(/*maxVerify=*/16);
    Json const draftJson = makeMTPDraftConfig(/*maxDraft=*/16);
    auto const basePath = writeJsonToTempFile(baseJson, "base");
    auto const draftPath = writeJsonToTempFile(draftJson, "draft");

    SpecDecodeDraftingConfig drafting{};
    drafting.draftTopK = 2;
    drafting.draftingStep = 4; // chain input 4 <= 16; candidate pool = 1 + 4*2 = 9
    drafting.verifyTokens = 8; // 8 <= 9 and 8 <= 16

    DeploymentConfig bundle = createDeploymentConfig(
        basePath, std::optional<std::filesystem::path>{draftPath}, std::optional<SpecDecodeDraftingConfig>{drafting});

    EXPECT_EQ(bundle.specDecodeMode(), SpecDecodeMode::kMTP);
    ASSERT_TRUE(bundle.specConfig.has_value());
    EXPECT_EQ(bundle.specConfig->draftTopK, 2);
    EXPECT_EQ(bundle.specConfig->draftSteps, 4);
    EXPECT_EQ(bundle.specConfig->verifyTokenCount, 8);
}

TEST_F(DeploymentConfigTest, MTPTreeRejectsReducedDraftVocabMap)
{
    Json const baseJson = makeMTPBaseConfig(/*maxVerify=*/16);
    Json draftJson = makeMTPDraftConfig(/*maxDraft=*/16);
    draftJson["reduced_vocab_size"] = 16000;
    auto const basePath = writeJsonToTempFile(baseJson, "base");
    auto const draftPath = writeJsonToTempFile(draftJson, "draft");

    SpecDecodeDraftingConfig drafting{};
    drafting.draftTopK = 2;
    drafting.draftingStep = 4;
    drafting.verifyTokens = 8;

    try
    {
        static_cast<void>(createDeploymentConfig(basePath, std::optional<std::filesystem::path>{draftPath},
            std::optional<SpecDecodeDraftingConfig>{drafting}));
        FAIL() << "Expected reduced-vocabulary MTP tree drafting to be rejected.";
    }
    catch (std::runtime_error const& error)
    {
        EXPECT_NE(std::string(error.what()).find("use --specDraftTopK 1"), std::string::npos) << error.what();
    }
}

TEST_F(DeploymentConfigTest, MTPTreeRejectsTopKNotLessThanVerifySize)
{
    Json const baseJson = makeMTPBaseConfig(/*maxVerify=*/16);
    Json const draftJson = makeMTPDraftConfig(/*maxDraft=*/16);
    auto const basePath = writeJsonToTempFile(baseJson, "base");
    auto const draftPath = writeJsonToTempFile(draftJson, "draft");

    SpecDecodeDraftingConfig drafting{};
    drafting.draftTopK = 4;
    drafting.draftingStep = 3;
    drafting.verifyTokens = 4; // topK must be < verifyTokenCount (root consumes one node)

    EXPECT_THROW(createDeploymentConfig(basePath, std::optional<std::filesystem::path>{draftPath},
                     std::optional<SpecDecodeDraftingConfig>{drafting}),
        std::runtime_error);
}

TEST_F(DeploymentConfigTest, MTPTreeAllowsVerifySizeBeyondTreeCapacity)
{
    // No tree-capacity bound: verifyTokenCount=8 exceeds the full 2-ary depth-2 tree
    // capacity (1+2+4=7); the extra node is well-defined padding at runtime,
    // not a configuration error.
    Json const baseJson = makeMTPBaseConfig(/*maxVerify=*/16);
    Json const draftJson = makeMTPDraftConfig(/*maxDraft=*/16);
    auto const basePath = writeJsonToTempFile(baseJson, "base");
    auto const draftPath = writeJsonToTempFile(draftJson, "draft");

    SpecDecodeDraftingConfig drafting{};
    drafting.draftTopK = 2;
    drafting.draftingStep = 2;
    drafting.verifyTokens = 8;

    DeploymentConfig bundle = createDeploymentConfig(
        basePath, std::optional<std::filesystem::path>{draftPath}, std::optional<SpecDecodeDraftingConfig>{drafting});

    EXPECT_EQ(bundle.specDecodeMode(), SpecDecodeMode::kMTP);
    ASSERT_TRUE(bundle.specConfig.has_value());
    EXPECT_EQ(bundle.specConfig->verifyTokenCount, 8);
}

TEST_F(DeploymentConfigTest, MTPTreeAcceptedTokensPerRoundAreCappedByVerifySize)
{
    // Given an MTP tree with fewer verify nodes than its chain depth and a base without verify headroom, when bounding
    // a round, then the accepted path cannot exceed the verify nodes.
    auto const basePath = writeJsonToTempFile(makeMTPBaseConfig(/*maxVerify=*/3), "base");
    auto const draftPath = writeJsonToTempFile(makeMTPDraftConfig(/*maxDraft=*/16), "draft");

    SpecDecodeDraftingConfig drafting{};
    drafting.draftTopK = 2;
    drafting.draftingStep = 4;
    drafting.verifyTokens = 3;

    DeploymentConfig bundle = createDeploymentConfig(
        basePath, std::optional<std::filesystem::path>{draftPath}, std::optional<SpecDecodeDraftingConfig>{drafting});

    EXPECT_EQ(bundle.maxAcceptedTokensPerRound(), 3);
}

TEST_F(DeploymentConfigTest, MTPTreeRejectsFanoutAboveLimit)
{
    Json const baseJson = makeMTPBaseConfig(/*maxVerify=*/16);
    Json const draftJson = makeMTPDraftConfig(/*maxDraft=*/16);
    auto const basePath = writeJsonToTempFile(baseJson, "base");
    auto const draftPath = writeJsonToTempFile(draftJson, "draft");

    SpecDecodeDraftingConfig drafting{};
    drafting.draftTopK = 9; // > kMTPTreeMaxCandidateFanout (8)
    drafting.draftingStep = 2;
    drafting.verifyTokens = 12;

    EXPECT_THROW(createDeploymentConfig(basePath, std::optional<std::filesystem::path>{draftPath},
                     std::optional<SpecDecodeDraftingConfig>{drafting}),
        std::runtime_error);
}

TEST_F(DeploymentConfigTest, MTPTreeHybridBaseValidatesOk)
{
    // Hybrid (GDN/causal-conv) base + tree drafting is supported;
    // draftSteps+1 stays within the hybrid intermediate-state depth limit (16).
    Json const baseJson = makeHybridMTPBaseConfig(/*maxVerify=*/16);
    Json const draftJson = makeMTPDraftConfig(/*maxDraft=*/16);
    auto const basePath = writeJsonToTempFile(baseJson, "base");
    auto const draftPath = writeJsonToTempFile(draftJson, "draft");

    SpecDecodeDraftingConfig drafting{};
    drafting.draftTopK = 2;
    drafting.draftingStep = 3;
    drafting.verifyTokens = 7; // == candidate pool (1 + 3*2)

    DeploymentConfig bundle = createDeploymentConfig(
        basePath, std::optional<std::filesystem::path>{draftPath}, std::optional<SpecDecodeDraftingConfig>{drafting});

    EXPECT_EQ(bundle.specDecodeMode(), SpecDecodeMode::kMTP);
    EXPECT_GT(bundle.base.numLinearAttnLayers, 0);
    ASSERT_TRUE(bundle.specConfig.has_value());
    EXPECT_EQ(bundle.specConfig->draftTopK, 2);
    EXPECT_EQ(bundle.specConfig->verifyTokenCount, 7);
}

TEST_F(DeploymentConfigTest, MTPTreeHybridBaseRejectsVerifySizeAboveTransactionalLimit)
{
    Json const baseJson = makeHybridMTPBaseConfig(/*maxVerify=*/128);
    Json const draftJson = makeMTPDraftConfig(/*maxDraft=*/32);
    auto const basePath = writeJsonToTempFile(baseJson, "base");
    auto const draftPath = writeJsonToTempFile(draftJson, "draft");

    SpecDecodeDraftingConfig drafting{};
    drafting.draftTopK = 4;
    drafting.draftingStep = 4;
    drafting.verifyTokens = 65;

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
    drafting.draftTopK = 2;
    drafting.draftingStep = 16; // depth = 16 + 1 = 17 > kMTPMaxAcceptDepthForCurrentEagleUtilityKernels (16)
    drafting.verifyTokens = 17;

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
    drafting.draftTopK = 1;
    drafting.draftingStep = 4;
    drafting.verifyTokens = 8;

    EXPECT_THROW(createDeploymentConfig(basePath, std::optional<std::filesystem::path>{draftPath},
                     std::optional<SpecDecodeDraftingConfig>{drafting}),
        std::runtime_error);
}

TEST_F(DeploymentConfigTest, MTPRejectsVerifySizeAboveCurrentEagleUtilityKernelLimit)
{
    // Engine capacities of 32 keep every other constraint satisfied so the depth check is the
    // only violation: chain verifyTokenCount = draftSteps + 1 = 17 implies accept depth 17 > 16.
    Json const baseJson = makeMTPBaseConfig(/*maxVerify=*/32);
    Json const draftJson = makeMTPDraftConfig(/*maxDraft=*/32);
    auto const basePath = writeJsonToTempFile(baseJson, "base");
    auto const draftPath = writeJsonToTempFile(draftJson, "draft");

    SpecDecodeDraftingConfig drafting{};
    drafting.draftTopK = 1;
    drafting.draftingStep = 16;
    drafting.verifyTokens = 17;

    EXPECT_THROW(createDeploymentConfig(basePath, std::optional<std::filesystem::path>{draftPath},
                     std::optional<SpecDecodeDraftingConfig>{drafting}),
        std::runtime_error);
}

TEST_F(DeploymentConfigTest, MTPAcceptsDepthAtCurrentEagleUtilityKernelLimit)
{
    // Boundary case: accept depth = draftSteps + 1 = 16 is exactly at the kernel limit.
    Json const baseJson = makeMTPBaseConfig(/*maxVerify=*/16);
    Json const draftJson = makeMTPDraftConfig(/*maxDraft=*/16);
    auto const basePath = writeJsonToTempFile(baseJson, "base");
    auto const draftPath = writeJsonToTempFile(draftJson, "draft");

    SpecDecodeDraftingConfig drafting{};
    drafting.draftTopK = 1;
    drafting.draftingStep = 15;
    drafting.verifyTokens = 16;

    auto const bundle = createDeploymentConfig(
        basePath, std::optional<std::filesystem::path>{draftPath}, std::optional<SpecDecodeDraftingConfig>{drafting});

    EXPECT_EQ(bundle.specDecodeMode(), SpecDecodeMode::kMTP);
    ASSERT_TRUE(bundle.specConfig.has_value());
    EXPECT_EQ(bundle.specConfig->draftSteps, 15);
    EXPECT_EQ(bundle.specConfig->verifyTokenCount, 16);
}

TEST_F(DeploymentConfigTest, DFlashLinearInfersVerifySizeFromBlockSize)
{
    Json const baseJson = makeHybridDFlashBaseConfig(/*maxVerify=*/16);
    Json const draftJson = makeDFlashDraftConfig(/*maxDraft=*/16);
    auto const basePath = writeJsonToTempFile(baseJson, "base");
    auto const draftPath = writeJsonToTempFile(draftJson, "draft");

    SpecDecodeDraftingConfig drafting{};
    drafting.draftTopK = 1;

    DeploymentConfig bundle = createDeploymentConfig(
        basePath, std::optional<std::filesystem::path>{draftPath}, std::optional<SpecDecodeDraftingConfig>{drafting});

    EXPECT_EQ(bundle.specDecodeMode(), SpecDecodeMode::kDFlash);
    ASSERT_TRUE(bundle.specConfig.has_value());
    EXPECT_EQ(bundle.specConfig->verifyTokenCount, 16);
    EXPECT_EQ(bundle.specConfig->draftBlockSize, 16);
}

TEST_F(DeploymentConfigTest, CachedBlockDefaultFitsDraftPhysicalCapacity)
{
    // Given a block-16 checkpoint exported with only eight physical draft rows,
    // when P is omitted, then chain and tree use the largest proposal depth the draft can execute.
    for (int32_t const draftTopK : {1, 4})
    {
        Json const baseJson = makeDenseDFlashBaseConfig(/*maxVerify=*/16);
        Json const draftJson = makeDFlashDraftConfig(/*maxDraft=*/8);
        auto const basePath = writeJsonToTempFile(baseJson, "base");
        auto const draftPath = writeJsonToTempFile(draftJson, "draft");

        SpecDecodeDraftingConfig drafting{};
        drafting.draftTopK = draftTopK;
        if (draftTopK > 1)
        {
            drafting.verifyTokens = 16;
        }

        DeploymentConfig const bundle = createDeploymentConfig(basePath, draftPath, drafting);
        ASSERT_TRUE(bundle.specConfig.has_value());
        EXPECT_EQ(bundle.specConfig->proposalTokenCount, 7);
        EXPECT_EQ(bundle.specConfig->draftBlockSize, 8);
        EXPECT_EQ(bundle.specConfig->verifyTokenCount, draftTopK == 1 ? 8 : 16);
    }
}

TEST_F(DeploymentConfigTest, DFlashLinearVerifySizeEqualBlockSizeValidatesOk)
{
    Json const baseJson = makeDenseDFlashBaseConfig(/*maxVerify=*/16);
    Json const draftJson = makeDFlashDraftConfig(/*maxDraft=*/16);
    auto const basePath = writeJsonToTempFile(baseJson, "base");
    auto const draftPath = writeJsonToTempFile(draftJson, "draft");

    SpecDecodeDraftingConfig drafting{};
    drafting.draftTopK = 1;
    drafting.verifyTokens = 16;

    DeploymentConfig bundle = createDeploymentConfig(
        basePath, std::optional<std::filesystem::path>{draftPath}, std::optional<SpecDecodeDraftingConfig>{drafting});

    EXPECT_EQ(bundle.specDecodeMode(), SpecDecodeMode::kDFlash);
    ASSERT_TRUE(bundle.specConfig.has_value());
    EXPECT_EQ(bundle.specConfig->verifyTokenCount, 16);
    EXPECT_EQ(bundle.specConfig->draftBlockSize, 16);
}

TEST_F(DeploymentConfigTest, DFlashLinearRejectsMismatchedVerifyTokenCount)
{
    Json const baseJson = makeDenseDFlashBaseConfig(/*maxVerify=*/32);
    Json const draftJson = makeDFlashDraftConfig(/*maxDraft=*/16);
    auto const basePath = writeJsonToTempFile(baseJson, "base");
    auto const draftPath = writeJsonToTempFile(draftJson, "draft");

    SpecDecodeDraftingConfig drafting{};
    drafting.draftTopK = 1;
    drafting.verifyTokens = 17;

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
    drafting.draftTopK = 4;
    drafting.verifyTokens = 128;
    drafting.proposalTokens = 15;

    DeploymentConfig bundle = createDeploymentConfig(
        basePath, std::optional<std::filesystem::path>{draftPath}, std::optional<SpecDecodeDraftingConfig>{drafting});

    EXPECT_EQ(bundle.specDecodeMode(), SpecDecodeMode::kDFlash);
    ASSERT_TRUE(bundle.specConfig.has_value());
    EXPECT_EQ(bundle.specConfig->verifyTokenCount, 128);
    EXPECT_EQ(bundle.specConfig->draftTopK, 4);
    EXPECT_EQ(bundle.specConfig->draftBlockSize, 16);
}

TEST_F(DeploymentConfigTest, JetSpecDDTreeValidatesOk)
{
    Json const baseJson = makeDenseJetSpecBaseConfig(/*maxVerify=*/128);
    Json const draftJson = makeJetSpecDraftConfig(/*maxDraft=*/16);
    auto const basePath = writeJsonToTempFile(baseJson, "base");
    auto const draftPath = writeJsonToTempFile(draftJson, "draft");

    SpecDecodeDraftingConfig drafting{};
    drafting.draftTopK = 4;
    drafting.verifyTokens = 128;
    drafting.proposalTokens = 15;

    DeploymentConfig bundle = createDeploymentConfig(
        basePath, std::optional<std::filesystem::path>{draftPath}, std::optional<SpecDecodeDraftingConfig>{drafting});

    EXPECT_EQ(bundle.specDecodeMode(), SpecDecodeMode::kJetSpec);
    ASSERT_TRUE(bundle.specConfig.has_value());
    ASSERT_TRUE(bundle.draft.has_value());
    EXPECT_EQ(bundle.specConfig->verifyTokenCount, 128);
    EXPECT_EQ(bundle.specConfig->draftTopK, 4);
    EXPECT_EQ(bundle.specConfig->draftBlockSize, 16);
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
    drafting.draftTopK = 4;
    drafting.verifyTokens = 129;
    drafting.proposalTokens = 15;

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
    drafting.draftTopK = 1;

    DeploymentConfig bundle = createDeploymentConfig(
        basePath, std::optional<std::filesystem::path>{draftPath}, std::optional<SpecDecodeDraftingConfig>{drafting});

    auto const cfg = dflash_utils::makeCachedBlockDraftRuntimeConfig(bundle);
    EXPECT_EQ(cfg.userMode, SpecDecodeMode::kDFlash);
    EXPECT_EQ(cfg.proposalAttention, dflash_utils::ProposalAttentionPolicy::kBidirectional);
    EXPECT_EQ(cfg.treePolicy, dflash_utils::BlockDraftTreePolicy::kLinear);
    EXPECT_EQ(cfg.blockSize, 16);
    EXPECT_EQ(cfg.verifyTokenCount, 16);
    EXPECT_EQ(cfg.proposalLen, 15);
    EXPECT_EQ(cfg.candidateTopK, 1);
    EXPECT_EQ(cfg.maskTokenId, 123);
    EXPECT_EQ(cfg.draftHiddenSize, 768);
    EXPECT_EQ(cfg.baseOutputHiddenDim, 2304);
    EXPECT_EQ(cfg.draftVocabSize, 32000);
}

// Given a base config whose dflash_config.mask_token_id disagrees with the draft's
// When the runtime config is resolved
// Then the draft's value wins, not the base's
//
// Regression lock for the root cause of !1095: the base engine's exported mask_token_id defaults to the exporter's
// placeholder (151669 in production, an id far outside most target models' real vocabulary) whenever the target
// model itself has no native dflash_config -- most non-DFlash-native base models. Reading that placeholder instead
// of the draft's authoritative value fed out-of-vocab embeddings to the draft engine every proposal step, which
// collapsed accept length to ~1 (see !1095's measured before/after). The two DFlashLinearRuntimeConfigIsBidirectional
// -style tests above never catch a base/draft disagreement here because they set both configs' mask_token_id to the
// same value.
TEST_F(DeploymentConfigTest, DFlashRuntimeConfigPrefersDraftMaskTokenIdOverBase)
{
    Json baseJson = makeDenseDFlashBaseConfig(/*maxSpecVerifyTokens=*/16);
    Json draftJson = makeDFlashDraftConfig(/*maxSpecDraftTokens=*/16);
    // The value production actually leaks in this failure mode: the exporter's out-of-vocabulary placeholder.
    baseJson["dflash_config"]["mask_token_id"] = 151669;
    draftJson["dflash_config"]["mask_token_id"] = 123;
    auto const basePath = writeJsonToTempFile(baseJson, "base");
    auto const draftPath = writeJsonToTempFile(draftJson, "draft");

    SpecDecodeDraftingConfig drafting{};
    drafting.draftTopK = 1;

    DeploymentConfig bundle = createDeploymentConfig(
        basePath, std::optional<std::filesystem::path>{draftPath}, std::optional<SpecDecodeDraftingConfig>{drafting});

    auto const cfg = dflash_utils::makeCachedBlockDraftRuntimeConfig(bundle);
    EXPECT_EQ(cfg.maskTokenId, 123) << "the draft's own mask_token_id must win over the base's, however the base's "
                                       "value got there";
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
    drafting.draftTopK = 4;
    drafting.verifyTokens = 128;
    drafting.proposalTokens = 15;

    DeploymentConfig bundle = createDeploymentConfig(
        basePath, std::optional<std::filesystem::path>{draftPath}, std::optional<SpecDecodeDraftingConfig>{drafting});

    auto const cfg = dflash_utils::makeCachedBlockDraftRuntimeConfig(bundle);
    EXPECT_EQ(cfg.userMode, SpecDecodeMode::kJetSpec);
    EXPECT_EQ(cfg.proposalAttention, dflash_utils::ProposalAttentionPolicy::kCausal);
    EXPECT_EQ(cfg.treePolicy, dflash_utils::BlockDraftTreePolicy::kDDTree);
    EXPECT_EQ(cfg.blockSize, 16);
    EXPECT_EQ(cfg.verifyTokenCount, 128);
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
    drafting.draftTopK = 1;

    DeploymentConfig bundle = createDeploymentConfig(
        basePath, std::optional<std::filesystem::path>{draftPath}, std::optional<SpecDecodeDraftingConfig>{drafting});

    EXPECT_EQ(bundle.specDecodeMode(), SpecDecodeMode::kJetSpec);
    ASSERT_TRUE(bundle.specConfig.has_value());
    ASSERT_TRUE(bundle.draft.has_value());
    EXPECT_EQ(bundle.specConfig->verifyTokenCount, 16);
    EXPECT_EQ(bundle.specConfig->draftBlockSize, 16);
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
    drafting.draftTopK = 1;

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
    drafting.draftTopK = 4;
    drafting.verifyTokens = 129;
    drafting.proposalTokens = 15;

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
    drafting.draftTopK = 2;
    drafting.verifyTokens = 8;
    drafting.proposalTokens = 0;

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
    drafting.draftTopK = 2;
    drafting.verifyTokens = 8;
    drafting.proposalTokens = -2;

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
    drafting.draftTopK = 4;
    drafting.verifyTokens = 128;
    drafting.proposalTokens = 16;

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
    drafting.draftTopK = 4;
    drafting.verifyTokens = 128;
    drafting.proposalTokens = 32;

    EXPECT_THROW(createDeploymentConfig(basePath, std::optional<std::filesystem::path>{draftPath},
                     std::optional<SpecDecodeDraftingConfig>{drafting}),
        std::runtime_error);
}

TEST_F(DeploymentConfigTest, DFlashDDTreeRejectsProposalBeyondCheckpointBlock)
{
    Json const baseJson = makeDenseDFlashBaseConfig(/*maxVerify=*/128);
    Json const draftJson = makeDFlashDraftConfig(/*maxDraft=*/21);
    auto const basePath = writeJsonToTempFile(baseJson, "base");
    auto const draftPath = writeJsonToTempFile(draftJson, "draft");

    SpecDecodeDraftingConfig drafting{};
    drafting.draftTopK = 7;
    drafting.verifyTokens = 128;
    drafting.proposalTokens = 20;

    EXPECT_THROW(createDeploymentConfig(basePath, std::optional<std::filesystem::path>{draftPath},
                     std::optional<SpecDecodeDraftingConfig>{drafting}),
        std::runtime_error);
}

TEST_F(DeploymentConfigTest, JetSpecDDTreeRejectsProposalBeyondCheckpointBlock)
{
    Json const baseJson = makeDenseJetSpecBaseConfig(/*maxVerify=*/128);
    Json const draftJson = makeJetSpecDraftConfig(/*maxDraft=*/21);
    auto const basePath = writeJsonToTempFile(baseJson, "base");
    auto const draftPath = writeJsonToTempFile(draftJson, "draft");

    SpecDecodeDraftingConfig drafting{};
    drafting.draftTopK = 7;
    drafting.verifyTokens = 128;
    drafting.proposalTokens = 20;

    EXPECT_THROW(createDeploymentConfig(basePath, std::optional<std::filesystem::path>{draftPath},
                     std::optional<SpecDecodeDraftingConfig>{drafting}),
        std::runtime_error);
}

TEST_F(DeploymentConfigTest, DFlashDDTreeRejectsProposalBeyondCheckpointBlockWithBoundedVerifyCapacity)
{
    Json const baseJson = makeDenseDFlashBaseConfig(/*maxVerify=*/16);
    Json const draftJson = makeDFlashDraftConfig(/*maxDraft=*/32);
    auto const basePath = writeJsonToTempFile(baseJson, "base");
    auto const draftPath = writeJsonToTempFile(draftJson, "draft");

    SpecDecodeDraftingConfig drafting{};
    drafting.draftTopK = 4;
    drafting.verifyTokens = 16;
    drafting.proposalTokens = 31;

    EXPECT_THROW(createDeploymentConfig(basePath, std::optional<std::filesystem::path>{draftPath},
                     std::optional<SpecDecodeDraftingConfig>{drafting}),
        std::runtime_error);
}

TEST_F(DeploymentConfigTest, DFlashHybridBlockSizeAbove16Throws)
{
    Json const baseJson = makeHybridDFlashBaseConfig(/*maxVerify=*/128);
    Json const draftJson = makeDFlashDraftConfig(/*maxDraft=*/32);
    auto const basePath = writeJsonToTempFile(baseJson, "base");
    auto const draftPath = writeJsonToTempFile(draftJson, "draft");

    SpecDecodeDraftingConfig drafting{};
    drafting.draftTopK = 4;
    drafting.verifyTokens = 128;
    drafting.proposalTokens = 16;

    EXPECT_THROW(createDeploymentConfig(basePath, std::optional<std::filesystem::path>{draftPath},
                     std::optional<SpecDecodeDraftingConfig>{drafting}),
        std::runtime_error);
}

TEST_F(DeploymentConfigTest, DFlashHybridTreeRejectsVerifySizeAboveTransactionalLimit)
{
    Json const baseJson = makeHybridDFlashBaseConfig(/*maxVerify=*/128);
    Json const draftJson = makeDFlashDraftConfig(/*maxDraft=*/16);
    auto const basePath = writeJsonToTempFile(baseJson, "base");
    auto const draftPath = writeJsonToTempFile(draftJson, "draft");

    SpecDecodeDraftingConfig drafting{};
    drafting.draftTopK = 4;
    drafting.verifyTokens = 65;
    drafting.proposalTokens = 15;

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
    drafting.draftTopK = 1;

    DeploymentConfig bundle = createDeploymentConfig(
        basePath, std::optional<std::filesystem::path>{draftPath}, std::optional<SpecDecodeDraftingConfig>{drafting});

    ASSERT_TRUE(bundle.specConfig.has_value());
    EXPECT_EQ(bundle.specConfig->draftTopK, 1);
    EXPECT_EQ(bundle.specConfig->verifyTokenCount, 16);
    EXPECT_EQ(bundle.specConfig->draftBlockSize, 16);
}

TEST_F(DeploymentConfigTest, DFlashLinearBlockSizeOneThrows)
{
    Json const baseJson = makeDenseDFlashBaseConfig(/*maxVerify=*/16);
    Json const draftJson = makeDFlashDraftConfig(/*maxDraft=*/16);
    auto const basePath = writeJsonToTempFile(baseJson, "base");
    auto const draftPath = writeJsonToTempFile(draftJson, "draft");

    SpecDecodeDraftingConfig drafting{};
    drafting.draftTopK = 1;
    drafting.proposalTokens = 0;

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
    drafting.draftTopK = 2;
    drafting.verifyTokens = 8;

    DeploymentConfig bundle = createDeploymentConfig(
        basePath, std::optional<std::filesystem::path>{draftPath}, std::optional<SpecDecodeDraftingConfig>{drafting});

    ASSERT_TRUE(bundle.specConfig.has_value());
    EXPECT_EQ(bundle.specConfig->draftTopK, 2);
    EXPECT_EQ(bundle.specConfig->verifyTokenCount, 8);
    EXPECT_EQ(bundle.specConfig->draftBlockSize, 16);
}

TEST_F(DeploymentConfigTest, DFlashRejectsMultiStep)
{
    Json const baseJson = makeDenseDFlashBaseConfig(/*maxVerify=*/16);
    Json const draftJson = makeDFlashDraftConfig(/*maxDraft=*/16);
    auto const basePath = writeJsonToTempFile(baseJson, "base");
    auto const draftPath = writeJsonToTempFile(draftJson, "draft");

    SpecDecodeDraftingConfig drafting{};
    drafting.draftTopK = 1;
    drafting.draftingStep = 2;
    drafting.verifyTokens = 8;

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
    drafting.draftTopK = 9;
    drafting.verifyTokens = 16;

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
    drafting.draftTopK = 1;
    drafting.verifyTokens = 8;

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
    drafting.draftTopK = 1;
    drafting.verifyTokens = 8;

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

TEST_F(DeploymentConfigTest, SpecDecodeMRopeMatchingGeometryValidates)
{
    Json base = makeMTPBaseConfig(/*maxSpecVerifyTokens=*/4);
    Json draft = makeMTPDraftConfig(/*maxSpecDraftTokens=*/4);
    enableMRope(base);
    enableMRope(draft);
    auto const basePath = writeJsonToTempFile(base, "base");
    auto const draftPath = writeJsonToTempFile(draft, "draft");

    DeploymentConfig const deployment
        = createDeploymentConfig(basePath, std::optional<std::filesystem::path>{draftPath}, std::nullopt);
    EXPECT_EQ(deployment.maxRuntimeBatchSize(), 2);
    EXPECT_EQ(deployment.base.recurrentPoolRows, 2);
    ASSERT_TRUE(deployment.draft.has_value());
    EXPECT_EQ(deployment.draft->recurrentPoolRows, 2);
}

TEST_F(DeploymentConfigTest, DFlashAllowsMRopeBaseWithDefaultRopeDraft)
{
    Json base = makeHybridDFlashBaseConfig(/*maxSpecVerifyTokens=*/16);
    Json draft = makeDFlashDraftConfig(/*maxSpecDraftTokens=*/16);
    enableMRope(base);
    auto const basePath = writeJsonToTempFile(base, "base");
    auto const draftPath = writeJsonToTempFile(draft, "draft");

    EXPECT_NO_THROW(createDeploymentConfig(basePath, std::optional<std::filesystem::path>{draftPath}, std::nullopt));
}

TEST_F(DeploymentConfigTest, CachedDraftModeRejectsDualRopeDraft)
{
    Json base = makeDenseDFlashBaseConfig(/*maxSpecVerifyTokens=*/16);
    Json draft = makeDFlashDraftConfig(/*maxSpecDraftTokens=*/16);
    enableDualRope(draft);
    auto const basePath = writeJsonToTempFile(base, "base");
    auto const draftPath = writeJsonToTempFile(draft, "draft");

    EXPECT_THROW(
        {
            try
            {
                createDeploymentConfig(basePath, std::optional<std::filesystem::path>{draftPath}, std::nullopt);
            }
            catch (std::exception const& e)
            {
                EXPECT_NE(std::string(e.what()).find("single RoPE binding"), std::string::npos) << e.what();
                throw;
            }
        },
        std::exception);
}

TEST_F(DeploymentConfigTest, DSparkRejectsDualRopeDraft)
{
    Json base = makeDSparkBaseConfig(/*maxSpecVerifyTokens=*/8);
    Json draft = makeDSparkDraftConfig(/*maxSpecDraftTokens=*/7);
    enableDualRope(draft);
    auto const basePath = writeJsonToTempFile(base, "base");
    auto const draftPath = writeJsonToTempFile(draft, "draft");

    EXPECT_THROW(
        {
            try
            {
                createDeploymentConfig(basePath, std::optional<std::filesystem::path>{draftPath}, std::nullopt);
            }
            catch (std::exception const& e)
            {
                EXPECT_NE(std::string(e.what()).find("single RoPE binding"), std::string::npos) << e.what();
                throw;
            }
        },
        std::exception);
}

TEST_F(DeploymentConfigTest, SpecDecodeMRopeDraftRequiresMRopeBase)
{
    Json base = makeMTPBaseConfig(/*maxSpecVerifyTokens=*/4);
    Json draft = makeMTPDraftConfig(/*maxSpecDraftTokens=*/4);
    enableMRope(draft);
    auto const basePath = writeJsonToTempFile(base, "base");
    auto const draftPath = writeJsonToTempFile(draft, "draft");

    EXPECT_THROW(createDeploymentConfig(basePath, std::optional<std::filesystem::path>{draftPath}, std::nullopt),
        std::runtime_error);
}

TEST_F(DeploymentConfigTest, SpecDecodeMRopeMismatchedGeometryThrows)
{
    Json base = makeMTPBaseConfig(/*maxSpecVerifyTokens=*/4);
    Json draft = makeMTPDraftConfig(/*maxSpecDraftTokens=*/4);
    enableMRope(base);
    enableMRope(draft, /*ropeTheta=*/500000.0F);
    auto const basePath = writeJsonToTempFile(base, "base");
    auto const draftPath = writeJsonToTempFile(draft, "draft");

    EXPECT_THROW(createDeploymentConfig(basePath, std::optional<std::filesystem::path>{draftPath}, std::nullopt),
        std::runtime_error);
}

TEST_F(DeploymentConfigTest, SpecDecodeMRopeMismatchedFrequencyPartitionThrows)
{
    Json base = makeMTPBaseConfig(/*maxSpecVerifyTokens=*/4);
    Json draft = makeMTPDraftConfig(/*maxSpecDraftTokens=*/4);
    enableMRope(base);
    enableMRope(draft);
    draft["rope_scaling"]["mrope_section"] = Json::array({24, 20, 20});
    auto const basePath = writeJsonToTempFile(base, "base");
    auto const draftPath = writeJsonToTempFile(draft, "draft");

    EXPECT_THROW(createDeploymentConfig(basePath, std::optional<std::filesystem::path>{draftPath}, std::nullopt),
        std::runtime_error);
}

// Given an MTP base and draft that share one resident MRoPE table
// When both request the same YaRN scaling, or their YaRN factors differ
// Then the matching pair is accepted and the mismatched pair is rejected
TEST_F(DeploymentConfigTest, SpecDecodeMRopeRequiresMatchingYarn)
{
    Json base = makeMTPBaseConfig(/*maxSpecVerifyTokens=*/4);
    Json draft = makeMTPDraftConfig(/*maxSpecDraftTokens=*/4);
    enableMRope(base);
    enableMRope(draft);
    for (Json* config : {&base, &draft})
    {
        (*config)["rope_scaling"]["rope_type"] = "yarn";
        (*config)["rope_scaling"]["factor"] = 4.0F;
        (*config)["rope_scaling"]["original_max_position_embeddings"] = 32768;
    }
    auto const basePath = writeJsonToTempFile(base, "base");
    EXPECT_NO_THROW(createDeploymentConfig(
        basePath, std::optional<std::filesystem::path>{writeJsonToTempFile(draft, "draft")}, std::nullopt));

    draft["rope_scaling"]["factor"] = 2.0F;
    EXPECT_THROW(createDeploymentConfig(
                     basePath, std::optional<std::filesystem::path>{writeJsonToTempFile(draft, "draft")}, std::nullopt),
        std::runtime_error);
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
    // Both engine maxSpecDraftTokens (24) and user verifyTokenCount (32) contribute.
    // Expected: max(24, 32) = 32.
    Json const baseJson = makeBaseConfig(/*maxVerify=*/32, /*maxDraft=*/24);
    Json const draftJson = makeDraftConfig(/*maxVerify=*/32, /*maxDraft=*/24);
    auto const basePath = writeJsonToTempFile(baseJson, "base");
    auto const draftPath = writeJsonToTempFile(draftJson, "draft");

    SpecDecodeDraftingConfig drafting{};
    drafting.draftTopK = 3;
    drafting.draftingStep = 8;  // 24
    drafting.verifyTokens = 32; // 32

    DeploymentConfig bundle = createDeploymentConfig(
        basePath, std::optional<std::filesystem::path>{draftPath}, std::optional<SpecDecodeDraftingConfig>{drafting});
    EXPECT_EQ(bundle.effectiveMaxDraftProposalSize(), 32);
}

TEST_F(DeploymentConfigTest, EffectiveMaxDraftProposalSizeEngineCapacityWins)
{
    // Engine maxSpecDraftTokens (16) is larger than verifyTokenCount (8).
    // Expected: max(16, 8) = 16.
    Json const baseJson = makeBaseConfig(/*maxVerify=*/16, /*maxDraft=*/16);
    Json const draftJson = makeDraftConfig(/*maxVerify=*/16, /*maxDraft=*/16);
    auto const basePath = writeJsonToTempFile(baseJson, "base");
    auto const draftPath = writeJsonToTempFile(draftJson, "draft");

    SpecDecodeDraftingConfig drafting{};
    drafting.draftTopK = 4;
    drafting.draftingStep = 4;
    drafting.verifyTokens = 8;

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

TEST_F(DeploymentConfigTest, Gemma4MTPTreeValidatesOk)
{
    auto const basePath = writeJsonToTempFile(makeGemma4MTPBaseConfig(), "base");
    auto const draftPath = writeJsonToTempFile(makeGemma4MTPDraftConfig(), "draft");

    SpecDecodeDraftingConfig drafting{};
    drafting.draftTopK = 2;
    drafting.draftingStep = 3;
    drafting.verifyTokens = 4;

    DeploymentConfig const bundle = createDeploymentConfig(
        basePath, std::optional<std::filesystem::path>{draftPath}, std::optional<SpecDecodeDraftingConfig>{drafting});

    EXPECT_EQ(bundle.specDecodeMode(), SpecDecodeMode::kGemma4MTP);
    ASSERT_TRUE(bundle.specConfig.has_value());
    EXPECT_EQ(bundle.specConfig->draftTopK, 2);
    EXPECT_EQ(bundle.specConfig->draftSteps, 3);
    EXPECT_EQ(bundle.specConfig->verifyTokenCount, 4);
}

TEST_F(DeploymentConfigTest, Gemma4MTPLinearChainRejectsMismatchedVerifySize)
{
    auto const basePath = writeJsonToTempFile(makeGemma4MTPBaseConfig(), "base");
    auto const draftPath = writeJsonToTempFile(makeGemma4MTPDraftConfig(), "draft");

    SpecDecodeDraftingConfig drafting{};
    drafting.draftTopK = 1;
    drafting.draftingStep = 2;
    drafting.verifyTokens = 4;

    EXPECT_THROW(createDeploymentConfig(basePath, std::optional<std::filesystem::path>{draftPath},
                     std::optional<SpecDecodeDraftingConfig>{drafting}),
        std::runtime_error);
}
TEST_F(DeploymentConfigTest, Gemma4MTPLongLinearChainRemainsSupported)
{
    auto const basePath
        = writeJsonToTempFile(makeGemma4MTPBaseConfig(/*maxBatch=*/2, /*maxCap=*/256, /*maxVerify=*/17), "base");
    auto const draftPath
        = writeJsonToTempFile(makeGemma4MTPDraftConfig(/*maxBatch=*/2, /*maxCap=*/256, /*maxDraft=*/16), "draft");

    SpecDecodeDraftingConfig drafting{};
    drafting.draftTopK = 1;
    drafting.draftingStep = 16;
    drafting.verifyTokens = 17;

    EXPECT_NO_THROW(createDeploymentConfig(
        basePath, std::optional<std::filesystem::path>{draftPath}, std::optional<SpecDecodeDraftingConfig>{drafting}));
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

namespace
{
//! A Gemma4 MTP bundle whose base verifies `maxVerifyTokens` tokens and whose assistant takes four.
DeploymentConfig makeGemma4MTPBundle(
    int32_t maxVerifyTokens, int32_t draftTopK, int32_t draftingStep, int32_t verifyTokens)
{
    auto const basePath = writeJsonToTempFile(
        makeGemma4MTPBaseConfig(/*maxBatchSize=*/2, /*maxKVCacheCapacity=*/256, maxVerifyTokens), "base");
    auto const draftPath = writeJsonToTempFile(makeGemma4MTPDraftConfig(), "draft");
    SpecDecodeDraftingConfig drafting{};
    drafting.draftTopK = draftTopK;
    drafting.draftingStep = draftingStep;
    drafting.verifyTokens = verifyTokens;
    return createDeploymentConfig(
        basePath, std::optional<std::filesystem::path>{draftPath}, std::optional<SpecDecodeDraftingConfig>{drafting});
}
} // namespace

// Given Gemma4 MTP chains of three draft steps (V = 4) whose assistant takes 4 tokens, on bases that verify 16 and 24
// When each deployment is resolved
// Then forced tokens may replace the chain's draft tokens and extend it to the 16-token depth limit, beyond the
//      assistant's capacity, since the assistant drafts from the last accepted token alone: a grafting round verifies
//      and accepts a 16-token chain, a jump adds up to 15 tokens, and a decode round accepts at most 16 tokens
TEST_F(DeploymentConfigTest, Gemma4MTPChainJumpExtendsTheChainToTheDepthLimit)
{
    for (int32_t const maxVerifyTokens : {16, 24})
    {
        SCOPED_TRACE("max_spec_verify_tokens " + std::to_string(maxVerifyTokens));
        DeploymentConfig const bundle
            = makeGemma4MTPBundle(maxVerifyTokens, /*draftTopK=*/1, /*draftingStep=*/3, /*verifyTokens=*/4);

        ASSERT_TRUE(bundle.specConfig.has_value());
        EXPECT_EQ(bundle.specConfig->jumpVerifyTokenCount, 16);
        EXPECT_EQ(bundle.specConfig->jumpMaxPathLength, 16);
        EXPECT_EQ(bundle.specConfig->maxJumpTokens, 15);
        EXPECT_EQ(bundle.maxAcceptedTokensPerRound(), 16);
    }
}

// Given Gemma4 MTP trees of top-2 drafting over three steps: V = 8 on a base that verifies 24 tokens and on one that
//      verifies 8, and V = 120 on a base that verifies 200
// When each deployment is resolved
// Then the first's grafting round accepts up to 16 tokens, the depth limit, and verifies V + 15 = 23 nodes, so a jump
//      adds up to 15 tokens and a decode round accepts at most 16; the second cannot jump, so a grafting round keeps
//      the plain round's verify width and path capacity; the third verifies the 128-node tree limit, so a jump adds up
//      to 8 tokens
TEST_F(DeploymentConfigTest, Gemma4MTPTreeJumpWidensIntoTheVerifyHeadroomWithinTheDepthAndNodeLimits)
{
    DeploymentConfig const headroom
        = makeGemma4MTPBundle(/*maxVerifyTokens=*/24, /*draftTopK=*/2, /*draftingStep=*/3, /*verifyTokens=*/8);
    DeploymentConfig const noHeadroom
        = makeGemma4MTPBundle(/*maxVerifyTokens=*/8, /*draftTopK=*/2, /*draftingStep=*/3, /*verifyTokens=*/8);
    DeploymentConfig const byNodes
        = makeGemma4MTPBundle(/*maxVerifyTokens=*/200, /*draftTopK=*/2, /*draftingStep=*/3, /*verifyTokens=*/120);

    ASSERT_TRUE(headroom.specConfig.has_value());
    EXPECT_EQ(headroom.specConfig->jumpVerifyTokenCount, 23);
    EXPECT_EQ(headroom.specConfig->jumpMaxPathLength, 16);
    EXPECT_EQ(headroom.specConfig->maxJumpTokens, 15);
    EXPECT_EQ(headroom.maxAcceptedTokensPerRound(), 16);
    ASSERT_TRUE(noHeadroom.specConfig.has_value());
    EXPECT_EQ(noHeadroom.specConfig->maxJumpTokens, 0);
    EXPECT_EQ(noHeadroom.specConfig->jumpVerifyTokenCount, 8);
    EXPECT_EQ(noHeadroom.specConfig->jumpMaxPathLength, 4);
    EXPECT_EQ(noHeadroom.maxAcceptedTokensPerRound(), 4);
    ASSERT_TRUE(byNodes.specConfig.has_value());
    EXPECT_EQ(byNodes.specConfig->jumpVerifyTokenCount, 128);
    EXPECT_EQ(byNodes.specConfig->maxJumpTokens, 8);
    EXPECT_EQ(byNodes.specConfig->jumpMaxPathLength, 9);
}

// Given an exported DSpark model, when a caller selects a W2 backend,
// then deployment construction and reconstruction preserve that policy.
TEST_F(DeploymentConfigTest, DSparkW2BackendSurvivesConfigRoundTrip)
{
    auto const basePath = writeJsonToTempFile(makeDSparkBaseConfig(8), "base");
    auto const draftPath = writeJsonToTempFile(makeDSparkDraftConfig(7), "draft");
    for (auto backend :
        {DSparkW2Backend::kAuto, DSparkW2Backend::kFp16, DSparkW2Backend::kFp8, DSparkW2Backend::kW4A16})
    {
        SpecDecodeDraftingConfig drafting{};
        drafting.draftTopK = 1;
        drafting.draftingStep = 1;
        drafting.verifyTokens = 8;
        drafting.dsparkW2Backend = backend;
        auto const bundle = createDeploymentConfig(basePath, draftPath, drafting);
        ASSERT_TRUE(bundle.specConfig.has_value());
        EXPECT_EQ(bundle.specConfig->dsparkW2Backend, backend);
        ASSERT_TRUE(bundle.draftingConfig().has_value());
        EXPECT_EQ(bundle.draftingConfig()->dsparkW2Backend, backend);
        EXPECT_EQ(parseDSparkW2Backend(dsparkW2BackendName(backend)), backend);
    }
}

// Given packed or dense weights and device capability, when resolving policy,
// then auto falls back safely and explicit W4 never silently quantizes FP16.
TEST(DSparkW2BackendTest, ResolvesPolicyAndRejectsUnsupportedOverrides)
{
    using B = DSparkW2Backend;
    EXPECT_EQ(resolveDSparkW2Backend(B::kAuto, true, true), B::kW4A16);
    EXPECT_EQ(resolveDSparkW2Backend(B::kAuto, true, false), B::kFp16);
    EXPECT_EQ(resolveDSparkW2Backend(B::kAuto, false, true), B::kFp16);
    EXPECT_EQ(resolveDSparkW2Backend(B::kFp16, true, true), B::kFp16);
    EXPECT_EQ(resolveDSparkW2Backend(B::kFp8, true, true), B::kFp8);
    EXPECT_EQ(resolveDSparkW2Backend(B::kFp8, false, false), B::kFp8);
    EXPECT_EQ(resolveDSparkW2Backend(B::kW4A16, true, true), B::kW4A16);
    EXPECT_THROW(resolveDSparkW2Backend(B::kW4A16, false, true), std::exception);
    EXPECT_THROW(resolveDSparkW2Backend(B::kW4A16, true, false), std::exception);
    EXPECT_THROW(resolveDSparkW2Backend(static_cast<B>(99), true, true), std::exception);
    EXPECT_THROW(parseDSparkW2Backend("nvfp4"), std::exception);
    EXPECT_THROW(parseDSparkW2Backend("invalid"), std::exception);
}

TEST_F(DeploymentConfigTest, DSparkChainValidatesOk)
{
    auto const basePath = writeJsonToTempFile(makeDSparkBaseConfig(/*maxVerify=*/8), "base");
    auto const draftPath = writeJsonToTempFile(makeDSparkDraftConfig(/*maxDraft=*/7), "draft");

    SpecDecodeDraftingConfig drafting{};
    drafting.draftTopK = 1;
    drafting.verifyTokens = 8;

    DeploymentConfig bundle = createDeploymentConfig(
        basePath, std::optional<std::filesystem::path>{draftPath}, std::optional<SpecDecodeDraftingConfig>{drafting});

    EXPECT_EQ(bundle.specDecodeMode(), SpecDecodeMode::kDSpark);
    ASSERT_TRUE(bundle.specConfig.has_value());
    EXPECT_EQ(bundle.specConfig->verifyTokenCount, 8);
    EXPECT_EQ(bundle.specConfig->draftTopK, 1);
}

//! A false-anchor DSpark chain can consume every physical draft row without double-counting its mask row.
TEST_F(DeploymentConfigTest, DSparkFalseAnchorChainUsesPhysicalDraftCapacity)
{
    // Given B=7, D=8, V=8 and a false-anchor DSpark draft, when P/Q are omitted
    // or explicitly use the full block, then the mask row is counted exactly once.
    Json baseJson = makeDSparkBaseConfig(/*maxVerify=*/8, /*blockSize=*/7);
    Json draftJson = makeDSparkDraftConfig(/*maxDraft=*/8, /*blockSize=*/7);
    baseJson["dspark_config"]["sample_from_anchor"] = false;
    draftJson["dspark_config"]["sample_from_anchor"] = false;
    auto const basePath = writeJsonToTempFile(baseJson, "base");
    auto const draftPath = writeJsonToTempFile(draftJson, "draft");

    SpecDecodeDraftingConfig drafting{};
    drafting.draftTopK = 1;
    DeploymentConfig const omitted = createDeploymentConfig(
        basePath, std::optional<std::filesystem::path>{draftPath}, std::optional<SpecDecodeDraftingConfig>{drafting});
    ASSERT_TRUE(omitted.specConfig.has_value());
    EXPECT_EQ(omitted.specConfig->proposalTokenCount, 7);
    EXPECT_EQ(omitted.specConfig->verifyTokenCount, 8);
    EXPECT_EQ(omitted.specConfig->draftBlockSize, 8);

    drafting.proposalTokens = 7;
    drafting.verifyTokens = 8;
    DeploymentConfig const explicitConfig = createDeploymentConfig(
        basePath, std::optional<std::filesystem::path>{draftPath}, std::optional<SpecDecodeDraftingConfig>{drafting});
    ASSERT_TRUE(explicitConfig.specConfig.has_value());
    EXPECT_EQ(explicitConfig.specConfig->proposalTokenCount, 7);
    EXPECT_EQ(explicitConfig.specConfig->verifyTokenCount, 8);
    EXPECT_EQ(explicitConfig.specConfig->draftBlockSize, 8);
}

TEST_F(DeploymentConfigTest, DSparkResolvesIndependentlyOmittedBudgets)
{
    // Given false-anchor DSpark geometry, when either budget is omitted,
    // then defaults respect target capacity while explicit full-block P remains valid.
    for (bool const explicitProposal : {false, true})
    {
        Json baseJson = makeDSparkBaseConfig(explicitProposal ? 8 : 6, 7);
        Json draftJson = makeDSparkDraftConfig(8, 7);
        baseJson["dspark_config"]["sample_from_anchor"] = false;
        draftJson["dspark_config"]["sample_from_anchor"] = false;
        auto const basePath = writeJsonToTempFile(baseJson, "base");
        auto const draftPath = writeJsonToTempFile(draftJson, "draft");
        SpecDecodeDraftingConfig drafting{};
        drafting.draftTopK = 1;
        if (explicitProposal)
        {
            drafting.proposalTokens = 7;
        }
        auto const config = createDeploymentConfig(basePath, draftPath, drafting);
        ASSERT_TRUE(config.specConfig.has_value());
        EXPECT_EQ(config.specConfig->proposalTokenCount, explicitProposal ? 7 : 5);
        EXPECT_EQ(config.specConfig->verifyTokenCount, explicitProposal ? 8 : 6);
    }
}

TEST_F(DeploymentConfigTest, RejectsProposalCountWithoutRoomForRoot)
{
    // Given a proposal count at INT32_MAX, when either layout is resolved,
    // then iterative and false-anchor DSpark routes reject it before deriving root-inclusive lengths.
    for (bool const dspark : {false, true})
    {
        Json baseJson = dspark ? makeDSparkBaseConfig(16) : makeBaseConfig(16);
        Json draftJson = dspark ? makeDSparkDraftConfig(8) : makeDraftConfig(16, 8);
        if (dspark)
        {
            baseJson["dspark_config"]["sample_from_anchor"] = false;
            draftJson["dspark_config"]["sample_from_anchor"] = false;
        }
        auto const basePath = writeJsonToTempFile(baseJson, "base");
        auto const draftPath = writeJsonToTempFile(draftJson, "draft");
        for (int32_t const topK : {1, 2})
        {
            SCOPED_TRACE(::testing::Message() << "dspark=" << dspark << ", topK=" << topK);
            SpecDecodeDraftingConfig drafting{};
            if (dspark)
            {
                drafting.proposalTokens = std::numeric_limits<int32_t>::max();
            }
            else
            {
                drafting.draftingStep = std::numeric_limits<int32_t>::max();
            }
            drafting.draftTopK = topK;
            drafting.verifyTokens = 16;
            try
            {
                static_cast<void>(createDeploymentConfig(basePath, draftPath, drafting));
                FAIL() << "Expected proposal/root overflow rejection";
            }
            catch (std::runtime_error const& error)
            {
                EXPECT_NE(std::string(error.what()).find("too large to add the verification root"), std::string::npos)
                    << error.what();
            }
        }
    }
}

TEST_F(DeploymentConfigTest, DSparkCandidateTopKGreaterThanOneSelectsTree)
{
    // Tree decouples verifyTokenCount (node budget) from block_size + 1.
    auto const basePath = writeJsonToTempFile(makeDSparkBaseConfig(/*maxVerify=*/16), "base");
    auto const draftPath = writeJsonToTempFile(makeDSparkDraftConfig(/*maxDraft=*/7), "draft");

    SpecDecodeDraftingConfig drafting{};
    drafting.draftTopK = 4;
    drafting.verifyTokens = 16;

    DeploymentConfig bundle = createDeploymentConfig(
        basePath, std::optional<std::filesystem::path>{draftPath}, std::optional<SpecDecodeDraftingConfig>{drafting});

    EXPECT_EQ(bundle.specDecodeMode(), SpecDecodeMode::kDSpark);
    ASSERT_TRUE(bundle.specConfig.has_value());
    EXPECT_EQ(bundle.specConfig->verifyTokenCount, 16);
    EXPECT_EQ(bundle.specConfig->draftTopK, 4);
}

TEST_F(DeploymentConfigTest, DSparkAcceptedTokensPerRoundFollowAcceptedPath)
{
    // Given DSpark verify capacity larger than one accepted path, when bounding a round, then the tree allows its
    // block path plus bonus and a chain allows its verify input.
    auto const basePath = writeJsonToTempFile(makeDSparkBaseConfig(/*maxVerify=*/16), "base");
    auto const draftPath = writeJsonToTempFile(makeDSparkDraftConfig(/*maxDraft=*/7), "draft");

    SpecDecodeDraftingConfig drafting{};
    drafting.draftTopK = 4;
    drafting.draftingStep = 1;
    drafting.proposalTokens = 7;
    drafting.verifyTokens = 16;
    DeploymentConfig tree = createDeploymentConfig(
        basePath, std::optional<std::filesystem::path>{draftPath}, std::optional<SpecDecodeDraftingConfig>{drafting});
    EXPECT_EQ(tree.maxAcceptedTokensPerRound(), 8);

    drafting.draftTopK = 1;
    drafting.proposalTokens = 3;
    drafting.verifyTokens = 4;
    DeploymentConfig chain = createDeploymentConfig(
        basePath, std::optional<std::filesystem::path>{draftPath}, std::optional<SpecDecodeDraftingConfig>{drafting});
    EXPECT_EQ(chain.maxAcceptedTokensPerRound(), 4);
}

// Given DSpark trees of V = 8 and V = 4 nodes, top-2 over a block of 7, and a DSpark chain over the same block (V = 8),
//      all on a base that verifies 24 tokens with a draft whose proposal profile takes 7 rows
// When each deployment is resolved
// Then each tree's grafting round accepts at most the block and its bonus, 8 tokens, which the next draft forward takes
//      as its delta: the trees verify V + 7 nodes (15 and 11) and a jump adds up to 7 tokens; the chain keeps its
//      8-token verify window, in which a jump replaces up to 7 draft tokens; a round of any of them accepts at most 8
//      tokens, twice the V = 4 tree's plain path
TEST_F(DeploymentConfigTest, DSparkJumpWidensATreeWithinOneBlockAndKeepsTheChainWindow)
{
    auto const basePath = writeJsonToTempFile(makeDSparkBaseConfig(/*maxVerify=*/24), "base");
    auto const draftPath = writeJsonToTempFile(makeDSparkDraftConfig(/*maxDraft=*/7), "draft");
    SpecDecodeDraftingConfig drafting{};
    drafting.draftTopK = 2;
    drafting.proposalTokens = 7;
    drafting.verifyTokens = 8;
    DeploymentConfig const tree = createDeploymentConfig(basePath, draftPath, drafting);
    drafting.verifyTokens = 4;
    DeploymentConfig const smallTree = createDeploymentConfig(basePath, draftPath, drafting);
    drafting.draftTopK = 1;
    drafting.verifyTokens = 8;
    DeploymentConfig const chain = createDeploymentConfig(basePath, draftPath, drafting);

    ASSERT_TRUE(tree.specConfig.has_value());
    EXPECT_EQ(tree.specConfig->jumpVerifyTokenCount, 15);
    EXPECT_EQ(tree.specConfig->jumpMaxPathLength, 8);
    EXPECT_EQ(tree.specConfig->maxJumpTokens, 7);
    EXPECT_EQ(tree.maxAcceptedTokensPerRound(), 8);
    ASSERT_TRUE(smallTree.specConfig.has_value());
    EXPECT_EQ(smallTree.specConfig->maxPathLength, 4);
    EXPECT_EQ(smallTree.specConfig->jumpVerifyTokenCount, 11);
    EXPECT_EQ(smallTree.specConfig->jumpMaxPathLength, 8);
    EXPECT_EQ(smallTree.maxAcceptedTokensPerRound(), 8);
    ASSERT_TRUE(chain.specConfig.has_value());
    EXPECT_EQ(chain.specConfig->jumpVerifyTokenCount, 8);
    EXPECT_EQ(chain.specConfig->jumpMaxPathLength, 8);
    EXPECT_EQ(chain.specConfig->maxJumpTokens, 7);
    EXPECT_EQ(chain.maxAcceptedTokensPerRound(), 8);
}

// Given a DSpark tree of V = 8 nodes over a block of 7 on a hybrid base that verifies 24 tokens
// When the deployment is resolved
// Then it cannot jump: a hybrid tree keeps its V-node verify width, so no jump tokens, and a grafting round keeps the
//      plain round's verify width and path capacity
TEST_F(DeploymentConfigTest, DSparkTreeOnAHybridBaseCannotJumpForward)
{
    Json baseJson = makeDSparkBaseConfig(/*maxVerify=*/24);
    baseJson["num_attention_layers"] = 8;
    baseJson["num_linear_attn_layers"] = 4;
    baseJson["recurrent_state_num_heads"] = 4;
    baseJson["recurrent_state_head_dim"] = 64;
    baseJson["recurrent_state_size"] = 64;
    baseJson["conv_dim"] = 768;
    baseJson["conv_kernel"] = 4;
    baseJson["recurrent_state_dtype"] = "fp16";
    baseJson["conv_state_dtype"] = "fp16";
    auto const basePath = writeJsonToTempFile(baseJson, "base");
    auto const draftPath = writeJsonToTempFile(makeDSparkDraftConfig(/*maxDraft=*/7), "draft");
    SpecDecodeDraftingConfig drafting{};
    drafting.draftTopK = 2;
    drafting.proposalTokens = 7;
    drafting.verifyTokens = 8;

    DeploymentConfig const bundle = createDeploymentConfig(basePath, draftPath, drafting);

    ASSERT_TRUE(bundle.specConfig.has_value());
    EXPECT_EQ(bundle.specConfig->maxJumpTokens, 0);
    EXPECT_EQ(bundle.specConfig->jumpVerifyTokenCount, 8);
    EXPECT_EQ(bundle.specConfig->jumpMaxPathLength, 8);
    EXPECT_EQ(bundle.maxAcceptedTokensPerRound(), 8);
}

TEST_F(DeploymentConfigTest, DSparkTreeRecurrentReplayRequiresExactVerifyProfile)
{
    Json baseJson = makeDSparkBaseConfig(/*maxVerify=*/16);
    baseJson["num_attention_layers"] = 8;
    baseJson["num_linear_attn_layers"] = 4;
    baseJson["recurrent_state_num_heads"] = 4;
    baseJson["recurrent_state_head_dim"] = 64;
    baseJson["recurrent_state_size"] = 64;
    baseJson["conv_dim"] = 768;
    baseJson["conv_kernel"] = 4;
    baseJson["recurrent_state_dtype"] = "fp16";
    baseJson["conv_state_dtype"] = "fp16";
    baseJson["recurrent_spec_verify_mode"] = "replay";
    auto const basePath = writeJsonToTempFile(baseJson, "base");
    auto const draftPath = writeJsonToTempFile(makeDSparkDraftConfig(/*maxDraft=*/7), "draft");

    SpecDecodeDraftingConfig drafting{};
    drafting.draftTopK = 2;
    drafting.verifyTokens = 3;

    try
    {
        static_cast<void>(createDeploymentConfig(basePath, std::optional<std::filesystem::path>{draftPath},
            std::optional<SpecDecodeDraftingConfig>{drafting}));
        FAIL() << "Expected an exact recurrent replay verify-profile error";
    }
    catch (std::runtime_error const& error)
    {
        EXPECT_NE(
            std::string(error.what()).find("tree recurrent-state replay requires verifyTokenCount"), std::string::npos);
        EXPECT_NE(std::string(error.what()).find("maxSpecVerifyTokens"), std::string::npos);
    }

    drafting.verifyTokens = 16;
    EXPECT_NO_THROW(createDeploymentConfig(
        basePath, std::optional<std::filesystem::path>{draftPath}, std::optional<SpecDecodeDraftingConfig>{drafting}));

    drafting.draftTopK = 1;
    drafting.proposalTokens = 3;
    drafting.verifyTokens = 4;
    EXPECT_NO_THROW(createDeploymentConfig(
        basePath, std::optional<std::filesystem::path>{draftPath}, std::optional<SpecDecodeDraftingConfig>{drafting}));

    // Given a hybrid tree profile above the transactional node limit, when deployment is resolved, then reject it
    // early.
    baseJson["builder_config"]["max_spec_verify_tokens"] = kernel::kGDN_TREE_CHUNK_MAX_NODES + 1;
    writeJsonToTempFile(baseJson, "base");
    drafting.draftTopK = 2;
    drafting.proposalTokens = 7;
    drafting.verifyTokens = kernel::kGDN_TREE_CHUNK_MAX_NODES + 1;
    try
    {
        static_cast<void>(createDeploymentConfig(basePath, std::optional<std::filesystem::path>{draftPath},
            std::optional<SpecDecodeDraftingConfig>{drafting}));
        FAIL() << "Expected the hybrid tree node-limit error";
    }
    catch (std::runtime_error const& error)
    {
        EXPECT_NE(std::string(error.what()).find("GDN chunk-kernel node limit"), std::string::npos);
    }
}

TEST_F(DeploymentConfigTest, DSparkCausalTreeRequiresFullBlockDraftProfile)
{
    Json baseJson = makeDSparkBaseConfig(/*maxVerify=*/3, /*blockSize=*/16);
    Json draftJson = makeDSparkDraftConfig(/*maxDraft=*/1, /*blockSize=*/16);
    baseJson["dspark_config"]["causal_head"] = true;
    draftJson["dspark_config"]["causal_head"] = true;
    auto const basePath = writeJsonToTempFile(baseJson, "base");
    auto const draftPath = writeJsonToTempFile(draftJson, "draft");

    SpecDecodeDraftingConfig drafting{};
    drafting.draftTopK = 2;
    drafting.verifyTokens = 3;

    EXPECT_THROW(createDeploymentConfig(basePath, std::optional<std::filesystem::path>{draftPath},
                     std::optional<SpecDecodeDraftingConfig>{drafting}),
        std::runtime_error);
}

TEST_F(DeploymentConfigTest, DSparkCausalTreeAccountsForNonAnchorInputSlot)
{
    Json baseJson = makeDSparkBaseConfig(/*maxVerify=*/3, /*blockSize=*/16);
    Json draftJson = makeDSparkDraftConfig(/*maxDraft=*/15, /*blockSize=*/16);
    baseJson["dspark_config"]["causal_head"] = true;
    draftJson["dspark_config"]["causal_head"] = true;
    baseJson["dspark_config"]["sample_from_anchor"] = false;
    draftJson["dspark_config"]["sample_from_anchor"] = false;
    auto const basePath = writeJsonToTempFile(baseJson, "base");
    auto draftPath = writeJsonToTempFile(draftJson, "draft");

    SpecDecodeDraftingConfig drafting{};
    drafting.draftTopK = 2;
    drafting.verifyTokens = 3;

    EXPECT_THROW(createDeploymentConfig(basePath, std::optional<std::filesystem::path>{draftPath},
                     std::optional<SpecDecodeDraftingConfig>{drafting}),
        std::runtime_error);

    draftJson["builder_config"]["max_spec_draft_tokens"] = 17;
    draftPath = writeJsonToTempFile(draftJson, "draft_with_anchor_slot");
    EXPECT_NO_THROW(createDeploymentConfig(
        basePath, std::optional<std::filesystem::path>{draftPath}, std::optional<SpecDecodeDraftingConfig>{drafting}));
}

TEST_F(DeploymentConfigTest, DSparkTreeRejectsTopKNotLessThanVerifySize)
{
    auto const basePath = writeJsonToTempFile(makeDSparkBaseConfig(/*maxVerify=*/8), "base");
    auto const draftPath = writeJsonToTempFile(makeDSparkDraftConfig(/*maxDraft=*/7), "draft");

    SpecDecodeDraftingConfig drafting{};
    drafting.draftTopK = 8;
    drafting.verifyTokens = 8;

    EXPECT_THROW(createDeploymentConfig(basePath, std::optional<std::filesystem::path>{draftPath},
                     std::optional<SpecDecodeDraftingConfig>{drafting}),
        std::runtime_error);
}

TEST_F(DeploymentConfigTest, DSparkTreeRejectsFanoutAboveLimit)
{
    auto const basePath = writeJsonToTempFile(makeDSparkBaseConfig(/*maxVerify=*/32), "base");
    auto const draftPath = writeJsonToTempFile(makeDSparkDraftConfig(/*maxDraft=*/9), "draft");

    SpecDecodeDraftingConfig drafting{};
    drafting.draftTopK = 9;
    drafting.verifyTokens = 32;

    EXPECT_THROW(createDeploymentConfig(basePath, std::optional<std::filesystem::path>{draftPath},
                     std::optional<SpecDecodeDraftingConfig>{drafting}),
        std::runtime_error);
}

TEST_F(DeploymentConfigTest, DSparkTreeVerifySizeAboveNodeBudgetThrows)
{
    auto const basePath = writeJsonToTempFile(makeDSparkBaseConfig(/*maxVerify=*/256), "base");
    auto const draftPath = writeJsonToTempFile(makeDSparkDraftConfig(/*maxDraft=*/7), "draft");

    SpecDecodeDraftingConfig drafting{};
    drafting.draftTopK = 4;
    drafting.verifyTokens = 129;

    EXPECT_THROW(createDeploymentConfig(basePath, std::optional<std::filesystem::path>{draftPath},
                     std::optional<SpecDecodeDraftingConfig>{drafting}),
        std::runtime_error);
}

TEST_F(DeploymentConfigTest, DSparkTreeAllowsThresholdScheduler)
{
    auto const basePath = writeJsonToTempFile(makeDSparkBaseConfig(/*maxVerify=*/16), "base");
    auto const draftPath = writeJsonToTempFile(makeDSparkDraftConfig(/*maxDraft=*/7), "draft");

    SpecDecodeDraftingConfig drafting{};
    drafting.draftTopK = 4;
    drafting.verifyTokens = 16;
    drafting.dsparkSchedulerMode = DSparkSchedulerMode::kThreshold;
    drafting.dsparkConfidenceThreshold = 0.5F;

    DeploymentConfig bundle = createDeploymentConfig(
        basePath, std::optional<std::filesystem::path>{draftPath}, std::optional<SpecDecodeDraftingConfig>{drafting});
    ASSERT_TRUE(bundle.specConfig.has_value());
    EXPECT_EQ(bundle.specConfig->dsparkSchedulerMode, DSparkSchedulerMode::kThreshold);
}

TEST_F(DeploymentConfigTest, DSparkTreeRejectsSPS)
{
    auto const basePath = writeJsonToTempFile(makeDSparkBaseConfig(/*maxVerify=*/16), "base");
    auto const draftPath = writeJsonToTempFile(makeDSparkDraftConfig(/*maxDraft=*/7), "draft");

    SpecDecodeDraftingConfig drafting{};
    drafting.draftTopK = 4;
    drafting.verifyTokens = 16;
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
    drafting.draftTopK = 4;
    drafting.verifyTokens = 32;

    EXPECT_THROW(createDeploymentConfig(basePath, std::optional<std::filesystem::path>{draftPath},
                     std::optional<SpecDecodeDraftingConfig>{drafting}),
        std::runtime_error);
}

TEST_F(DeploymentConfigTest, DSparkTreeLargeBlockWithBoundedVerifySizeValidatesOk)
{
    // Same block_size 16, but verifyTokenCount 16 caps the accepted path at 16.
    auto const basePath = writeJsonToTempFile(makeDSparkBaseConfig(/*maxVerify=*/16, /*blockSize=*/16), "base");
    auto const draftPath = writeJsonToTempFile(makeDSparkDraftConfig(/*maxDraft=*/16, /*blockSize=*/16), "draft");

    SpecDecodeDraftingConfig drafting{};
    drafting.draftTopK = 4;
    drafting.verifyTokens = 16;

    DeploymentConfig bundle = createDeploymentConfig(
        basePath, std::optional<std::filesystem::path>{draftPath}, std::optional<SpecDecodeDraftingConfig>{drafting});

    ASSERT_TRUE(bundle.specConfig.has_value());
    EXPECT_EQ(bundle.specConfig->verifyTokenCount, 16);
}

TEST_F(DeploymentConfigTest, DSparkTreeBlockAboveDraftCapacityThrows)
{
    // Tree drafts the full block (7), which must fit the draft engine profile (6).
    auto const basePath = writeJsonToTempFile(makeDSparkBaseConfig(/*maxVerify=*/16), "base");
    auto const draftPath = writeJsonToTempFile(makeDSparkDraftConfig(/*maxDraft=*/6), "draft");

    SpecDecodeDraftingConfig drafting{};
    drafting.draftTopK = 4;
    drafting.verifyTokens = 16;

    EXPECT_THROW(createDeploymentConfig(basePath, std::optional<std::filesystem::path>{draftPath},
                     std::optional<SpecDecodeDraftingConfig>{drafting}),
        std::runtime_error);
}

TEST_F(DeploymentConfigTest, DSparkRejectsMultiStep)
{
    auto const basePath = writeJsonToTempFile(makeDSparkBaseConfig(/*maxVerify=*/8), "base");
    auto const draftPath = writeJsonToTempFile(makeDSparkDraftConfig(/*maxDraft=*/7), "draft");

    SpecDecodeDraftingConfig drafting{};
    drafting.draftTopK = 1;
    drafting.draftingStep = 2;
    drafting.verifyTokens = 8;

    EXPECT_THROW(createDeploymentConfig(basePath, std::optional<std::filesystem::path>{draftPath},
                     std::optional<SpecDecodeDraftingConfig>{drafting}),
        std::runtime_error);
}

//! Hybrid DSpark chains cannot exceed the causal-convolution path-width limit in either recurrent verify mode.
TEST_F(DeploymentConfigTest, DSparkHybridChainRejectsPathsAboveCausalConvLimit)
{
    // Given a GDN target, when a chain exceeds causal-conv capacity, then reject it before inference.
    Json base = makeDSparkBaseConfig(/*maxVerify=*/17, /*blockSize=*/16);
    base["num_attention_layers"] = 8;
    base["num_linear_attn_layers"] = 4;
    base["recurrent_state_num_heads"] = 4;
    base["recurrent_state_head_dim"] = 128;
    base["recurrent_state_size"] = 128;
    base["conv_dim"] = 1024;
    base["conv_kernel"] = 4;
    base["recurrent_state_dtype"] = "fp32";
    base["conv_state_dtype"] = "fp16";
    auto const basePath = writeJsonToTempFile(base, "base");
    auto const draftPath = writeJsonToTempFile(makeDSparkDraftConfig(/*maxDraft=*/16, /*blockSize=*/16), "draft");
    SpecDecodeDraftingConfig drafting{};
    drafting.draftTopK = 1;
    drafting.proposalTokens = 16;
    drafting.verifyTokens = 17;
    EXPECT_THROW(createDeploymentConfig(basePath, draftPath, drafting), std::runtime_error);

    // Given the same GDN target, when the path fits exactly, then initialization remains supported.
    drafting.proposalTokens = 15;
    drafting.verifyTokens = 16;
    EXPECT_NO_THROW(createDeploymentConfig(basePath, draftPath, drafting));

    // Given Mamba replay, when Q exceeds causal-conv capacity, then reject it before inference.
    base["recurrent_spec_verify_mode"] = "replay";
    base["recurrent_state_dtype"] = "fp16";
    base["recurrent_state_head_dim"] = 64;
    base["recurrent_state_size"] = 64;
    base["conv_dim"] = 768;
    writeJsonToTempFile(base, "base");
    drafting.proposalTokens = 16;
    drafting.verifyTokens = 17;
    EXPECT_THROW(createDeploymentConfig(basePath, draftPath, drafting), std::runtime_error);

    // Given the same Mamba replay target, when Q fits exactly, then initialization remains supported.
    drafting.proposalTokens = 15;
    drafting.verifyTokens = 16;
    EXPECT_NO_THROW(createDeploymentConfig(basePath, draftPath, drafting));

    // Given an attention-only target, when Q is 17, then the GDN limit does not restrict it.
    writeJsonToTempFile(makeDSparkBaseConfig(/*maxVerify=*/17, /*blockSize=*/16), "base");
    drafting.proposalTokens = 16;
    drafting.verifyTokens = 17;
    EXPECT_NO_THROW(createDeploymentConfig(basePath, draftPath, drafting));
}

TEST_F(DeploymentConfigTest, DSparkChainVerifySizeAboveUtilityKernelLimitThrows)
{
    auto const basePath = writeJsonToTempFile(makeDSparkBaseConfig(/*maxVerify=*/32, /*blockSize=*/32), "base");
    auto const draftPath = writeJsonToTempFile(makeDSparkDraftConfig(/*maxDraft=*/32, /*blockSize=*/32), "draft");

    SpecDecodeDraftingConfig drafting{};
    drafting.draftTopK = 1;
    drafting.verifyTokens = 18;

    EXPECT_THROW(createDeploymentConfig(basePath, std::optional<std::filesystem::path>{draftPath},
                     std::optional<SpecDecodeDraftingConfig>{drafting}),
        std::runtime_error);
}

TEST_F(DeploymentConfigTest, DSparkTreeSurvivalThresholdOfOneThrows)
{
    auto const basePath = writeJsonToTempFile(makeDSparkBaseConfig(/*maxVerify=*/16), "base");
    auto const draftPath = writeJsonToTempFile(makeDSparkDraftConfig(/*maxDraft=*/7), "draft");

    SpecDecodeDraftingConfig drafting{};
    drafting.draftTopK = 4;
    drafting.verifyTokens = 16;
    drafting.dsparkSchedulerMode = DSparkSchedulerMode::kThreshold;
    drafting.dsparkConfidenceThreshold = 1.0F;

    EXPECT_THROW(createDeploymentConfig(basePath, std::optional<std::filesystem::path>{draftPath},
                     std::optional<SpecDecodeDraftingConfig>{drafting}),
        std::runtime_error);
}

TEST_F(DeploymentConfigTest, DFlash2LinearContractValidates)
{
    auto const basePath = writeJsonToTempFile(makeDFlash2BaseConfig(/*maxVerify=*/8), "base");
    auto const draftPath = writeJsonToTempFile(makeDFlash2DraftConfig(/*maxDraft=*/8), "draft");
    SpecDecodeDraftingConfig drafting{};
    drafting.draftTopK = 1;
    drafting.verifyTokens = 8;

    DeploymentConfig bundle = createDeploymentConfig(
        basePath, std::optional<std::filesystem::path>{draftPath}, std::optional<SpecDecodeDraftingConfig>{drafting});
    EXPECT_EQ(bundle.specDecodeMode(), SpecDecodeMode::kDFlash);
    EXPECT_EQ(bundle.specConfig->draftBlockSize, 8);
    EXPECT_EQ(bundle.specConfig->verifyTokenCount, 8);
    EXPECT_EQ(bundle.maxAcceptedTokensPerRound(), 8);
}

TEST_F(DeploymentConfigTest, DFlashRejectsVersionMismatchWithoutDraftingOverrides)
{
    auto const basePath = writeJsonToTempFile(makeDFlash2BaseConfig(/*maxVerify=*/8), "base");
    auto const draftPath = writeJsonToTempFile(makeDFlashDraftConfig(/*maxDraft=*/8), "draft");

    EXPECT_THROW(createDeploymentConfig(basePath, std::optional<std::filesystem::path>{draftPath}, std::nullopt),
        std::runtime_error);
}

TEST_F(DeploymentConfigTest, DFlash2RejectsMismatchedProductionPair)
{
    auto expectInvalid = [&](Json const& draft) {
        auto const basePath = writeJsonToTempFile(makeDFlash2BaseConfig(/*maxVerify=*/8), "base");
        auto const draftPath = writeJsonToTempFile(draft, "draft");
        SpecDecodeDraftingConfig drafting{};
        drafting.draftTopK = 1;
        drafting.verifyTokens = 8;
        EXPECT_THROW(createDeploymentConfig(basePath, std::optional<std::filesystem::path>{draftPath}, drafting),
            std::runtime_error);
    };

    Json targetMismatch = makeDFlash2DraftConfig(/*maxDraft=*/8);
    targetMismatch["dflash_config"]["target_layer_ids"] = Json::array({1, 3, 5, 9, 11});
    expectInvalid(targetMismatch);

    Json layerCountMismatch = makeDFlash2DraftConfig(/*maxDraft=*/8);
    layerCountMismatch["num_hidden_layers"] = 4;
    expectInvalid(layerCountMismatch);

    Json conditioningMismatch = makeDFlash2DraftConfig(/*maxDraft=*/8);
    conditioningMismatch["base_model_hidden_size"] = 768;
    expectInvalid(conditioningMismatch);

    Json hiddenMismatch = makeDFlash2DraftConfig(/*maxDraft=*/8);
    hiddenMismatch["hidden_size"] = 512;
    expectInvalid(hiddenMismatch);

    Json vocabMismatch = makeDFlash2DraftConfig(/*maxDraft=*/8);
    vocabMismatch["draft_vocab_size"] = 31999;
    expectInvalid(vocabMismatch);
}

// Given a DFlash2 pair whose checkpoint selector lists 16 support tokens per row (block_size 8) and a base built for
// 32 verify nodes,
// When the runtime asks for a DDTree with 16 children per node, then with 17,
// Then 16 is accepted as a 32-node tree whose decoder reads the draft's 7 proposal rows with fanout 16, above the
// logit-row DDTree limit of 8, and 17 is rejected before any engine runs, because every child comes from the
// selector support.
TEST_F(DeploymentConfigTest, DFlash2DDTreeFanoutIsBoundedBySelectorSupport)
{
    Json base = makeDFlash2BaseConfig(/*maxVerify=*/32);
    Json draft = makeDFlash2DraftConfig(/*maxDraft=*/8);
    base["dflash_config"]["mask_token_id"] = 3;
    draft["dflash_config"]["mask_token_id"] = 3;
    auto const basePath = writeJsonToTempFile(base, "base");
    auto const draftPath = writeJsonToTempFile(draft, "draft");
    auto const create = [&](int32_t draftTopK) {
        SpecDecodeDraftingConfig drafting{};
        drafting.draftTopK = draftTopK;
        drafting.verifyTokens = 32;
        return createDeploymentConfig(basePath, std::optional<std::filesystem::path>{draftPath},
            std::optional<SpecDecodeDraftingConfig>{drafting});
    };

    DeploymentConfig const bundle = create(16);
    EXPECT_EQ(bundle.specConfig->verifyLayout, SpecVerifyLayout::kTree);
    EXPECT_EQ(bundle.specConfig->verifyTokenCount, 32);
    dflash_utils::CachedBlockDraftRuntimeConfig const runtime = dflash_utils::makeCachedBlockDraftRuntimeConfig(bundle);
    EXPECT_EQ(runtime.treePolicy, dflash_utils::BlockDraftTreePolicy::kDDTree);
    EXPECT_EQ(runtime.proposalLen, 7);
    EXPECT_EQ(runtime.candidateTopK, 16);
    EXPECT_THROW(create(17), std::runtime_error);
}

//! DFlash V2 always executes its checkpoint-owned block width; a smaller runtime proposal request is rejected.
TEST_F(DeploymentConfigTest, DFlash2RejectsRuntimeProposalOverride)
{
    // Given a DFlash V2 checkpoint with block_size=8 and matching engine capacities, when runtime requests P=3,
    // then deployment rejects resizing instead of executing a four-row draft block.
    auto const basePath = writeJsonToTempFile(makeDFlash2BaseConfig(/*maxVerify=*/8), "base");
    auto const draftPath = writeJsonToTempFile(makeDFlash2DraftConfig(/*maxDraft=*/8), "draft");
    SpecDecodeDraftingConfig drafting{};
    drafting.draftTopK = 1;
    drafting.proposalTokens = 3;

    EXPECT_THROW(createDeploymentConfig(basePath, std::optional<std::filesystem::path>{draftPath},
                     std::optional<SpecDecodeDraftingConfig>{drafting}),
        std::runtime_error);
}

TEST_F(DeploymentConfigTest, LiLiCorrPairValidates)
{
    // Given the LiLiCorr base/draft pair the exporter writes: six target layer IDs, a draft base_model_hidden_size
    // of 6 x 4096, block_size 8 and lilicorr_candidate_topk 8,
    // When a chain deployment with draftTopK 1 and verifyTokens 8 is created,
    // Then it is accepted and one round can accept the full eight-token block.
    auto const basePath = writeJsonToTempFile(makeLiLiCorrBaseConfig(/*maxVerify=*/8), "base");
    auto const draftPath = writeJsonToTempFile(makeLiLiCorrDraftConfig(/*maxDraft=*/8), "draft");
    SpecDecodeDraftingConfig drafting{};
    drafting.draftTopK = 1;
    drafting.verifyTokens = 8;

    DeploymentConfig const bundle = createDeploymentConfig(
        basePath, std::optional<std::filesystem::path>{draftPath}, std::optional<SpecDecodeDraftingConfig>{drafting});
    EXPECT_EQ(bundle.maxAcceptedTokensPerRound(), 8);
}

TEST_F(DeploymentConfigTest, LiLiCorrProjectorMismatchRejected)
{
    // Given a LiLiCorr base paired either with a DFlash V2 selector draft or with a LiLiCorr draft whose
    // lilicorr_candidate_topk differs (4 instead of 8),
    // When a chain deployment is created,
    // Then it is rejected with a message naming projector_type or lilicorr_candidate_topk respectively.
    using ::testing::HasSubstr;

    SpecDecodeDraftingConfig drafting{};
    drafting.draftTopK = 1;
    drafting.verifyTokens = 8;
    auto const basePath = writeJsonToTempFile(makeLiLiCorrBaseConfig(/*maxVerify=*/8), "base");

    auto const selectorDraftPath = writeJsonToTempFile(makeDFlash2DraftConfig(/*maxDraft=*/8), "draft");
    EXPECT_THAT(deploymentRejection(basePath, selectorDraftPath, drafting), HasSubstr("projector_type"));

    Json topKMismatch = makeLiLiCorrDraftConfig(/*maxDraft=*/8);
    topKMismatch["dflash_config"]["lilicorr_candidate_topk"] = 4;
    auto const topKMismatchPath = writeJsonToTempFile(topKMismatch, "draft");
    EXPECT_THAT(deploymentRejection(basePath, topKMismatchPath, drafting), HasSubstr("lilicorr_candidate_topk"));
}

TEST_F(DeploymentConfigTest, LiLiCorrRejectsTreeTopK)
{
    // Given a valid LiLiCorr base/draft pair,
    // When a deployment requests DDTree fan-out with draftTopK 2,
    // Then it is rejected with a message requiring draftTopK 1.
    using ::testing::HasSubstr;

    auto const basePath = writeJsonToTempFile(makeLiLiCorrBaseConfig(/*maxVerify=*/8), "base");
    auto const draftPath = writeJsonToTempFile(makeLiLiCorrDraftConfig(/*maxDraft=*/8), "draft");
    SpecDecodeDraftingConfig drafting{};
    drafting.draftTopK = 2;
    drafting.verifyTokens = 8;

    EXPECT_THAT(deploymentRejection(basePath, draftPath, drafting), HasSubstr("draftTopK must be 1"));
}
