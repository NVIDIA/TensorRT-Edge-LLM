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

#include "runtime/config/llmEngineConfig.h"

#include "common/pagedKvTypes.h"
#include "common/specDecodeConfigUtils.h"
#include "testUtils.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <limits>

using namespace trt_edgellm;
using namespace trt_edgellm::rt;
using Json = nlohmann::json;

namespace
{

//! Create a minimal valid config JSON for a base (non-MTP) model.
Json makeMinimalConfig()
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
    bc["max_batch_size"] = 2;
    bc["max_input_len"] = 128;
    bc["max_kv_cache_capacity"] = 256;
    bc["max_kv_pool_pages"] = 4;
    bc["max_lora_rank"] = 0;
    bc["spec_base"] = false;
    bc["ragged_backend"] = "entry_padded_compatibility";
    config["builder_config"] = bc;
    return config;
}

//! Write a JSON object to a temporary file and return the path.
std::filesystem::path writeJsonToTempFile(Json const& json)
{
    auto tmpPath = std::filesystem::temp_directory_path() / "llmEngineConfigTest_config.json";
    std::ofstream ofs(tmpPath);
    ofs << json.dump(2);
    ofs.close();
    return tmpPath;
}

//! Write a raw JSON string to the canonical temp file and return the path.
//! Useful for tests that want to write inline JSON literals directly.
std::filesystem::path writeTempConfig(std::string const& jsonStr)
{
    Json json = Json::parse(jsonStr);
    json["builder_config"]["ragged_backend"] = "entry_padded_compatibility";
    return writeJsonToTempFile(json);
}

//! Extend the minimal config with the fields a hybrid model requires
//! (recurrent / conv dimensions and their dtypes). Used by the hybrid-specific
//! tests that exercise the `numLinearAttnLayers > 0` gating.
Json makeHybridConfig()
{
    Json json = makeMinimalConfig();
    json["num_linear_attn_layers"] = 4;
    json["num_attention_layers"] = 8;
    json["recurrent_state_num_heads"] = 16;
    json["recurrent_state_head_dim"] = 32;
    json["recurrent_state_size"] = 64;
    json["conv_dim"] = 128;
    json["conv_kernel"] = 4;
    json["recurrent_state_dtype"] = "fp16";
    json["conv_state_dtype"] = "fp16";
    return json;
}

} // namespace

class LLMEngineConfigTest : public ::testing::Test
{
protected:
    void TearDown() override
    {
        // Clean up temp file if it exists.
        auto tmpPath = std::filesystem::temp_directory_path() / "llmEngineConfigTest_config.json";
        std::filesystem::remove(tmpPath);
    }
};

TEST(SpecDecodeConfigUtilsTest, DetectsLegacyTopLevelAndBuilderFlags)
{
    constexpr std::array<char const*, 16> kTOP_LEVEL_FLAGS{"eagle_base", "is_eagle3_draft", "mtp_base", "is_mtp_draft",
        "mtp_tree_base", "dflash_base", "dflash_tree_base", "is_dflash_draft", "jetspec_base", "jetspec_tree_base",
        "is_jetspec_draft", "dspark_base", "is_dspark_draft", "gemma4_mtp_base", "gemma4_mtp_draft",
        "shares_target_kv"};
    for (char const* flag : kTOP_LEVEL_FLAGS)
    {
        SCOPED_TRACE(flag);
        EXPECT_TRUE(configRevealsSpecDecode(Json{{flag, true}}));
    }

    EXPECT_TRUE(configRevealsSpecDecode(Json{{"builder_config", Json{{"spec_base", true}}}}));
    EXPECT_TRUE(configRevealsSpecDecode(Json{{"builder_config", Json{{"spec_draft", true}}}}));
    EXPECT_FALSE(configRevealsSpecDecode(Json::object()));
    EXPECT_FALSE(configRevealsSpecDecode(Json{{"builder_config", false}}));
}

TEST_F(LLMEngineConfigTest, ParseMinimalConfig)
{
    Json const json = makeMinimalConfig();
    auto const path = writeJsonToTempFile(json);

    LLMEngineConfig cfg = parseEngineConfig(path);

    EXPECT_EQ(cfg.numDecoderLayers, 12);
    EXPECT_EQ(cfg.numKVHeads, 4);
    EXPECT_EQ(cfg.headDim, 64);
    EXPECT_EQ(cfg.hiddenSize, 768);
    EXPECT_EQ(cfg.vocabSize, 32000);
    EXPECT_EQ(cfg.outputVocabSize, 32000);
    EXPECT_EQ(cfg.reducedVocabSize, 0);
    EXPECT_EQ(cfg.maxSupportedBatchSize, 2);
    EXPECT_EQ(cfg.maxSupportedInputLength, 128);
    EXPECT_EQ(cfg.maxKVCacheCapacity, 256);
    EXPECT_EQ(cfg.kvPoolPages, 4);
    EXPECT_EQ(cfg.numSwaPages, 0);
    EXPECT_EQ(cfg.maxSupportedLoraRank, 0);
    EXPECT_FALSE(cfg.isSpecDecodeBase);
    EXPECT_EQ(cfg.maxVerifyTreeSize, 0);
    EXPECT_EQ(cfg.maxDraftTreeSize, 0);
    EXPECT_EQ(cfg.kvCacheDtype, nvinfer1::DataType::kHALF);
    EXPECT_EQ(cfg.raggedBackend, RaggedBackendKind::kEntryPaddedCompatibility);
    EXPECT_EQ(cfg.maxNumSequences, 2);
    EXPECT_EQ(cfg.maxQueryLength, 128);
    EXPECT_EQ(cfg.maxPhysicalTokens, 256);
    EXPECT_EQ(cfg.recurrentPoolRows, 2);
}

TEST_F(LLMEngineConfigTest, RejectsUnsupportedRaggedBackend)
{
    Json json = makeMinimalConfig();
    json["builder_config"]["ragged_backend"] = "native_compact_ragged";
    EXPECT_THROW(parseEngineConfig(writeJsonToTempFile(json)), std::runtime_error);
}

TEST_F(LLMEngineConfigTest, EveryCommonDecoderRoleRequiresUnifiedContractMetadata)
{
    struct RoleCase
    {
        char const* role;
        char const* specDecodeType;
    };
    std::array<RoleCase, 9> const cases{{
        {"llm", "none"},
        {"base", "eagle3"},
        {"draft", "eagle3"},
        {"base", "mtp"},
        {"draft", "mtp"},
        {"draft", "dflash"},
        {"draft", "dspark"},
        {"draft", "gemma4_mtp"},
        {"dllm", "none"},
    }};

    for (RoleCase const& roleCase : cases)
    {
        SCOPED_TRACE(std::string(roleCase.role) + ":" + roleCase.specDecodeType);
        Json json = makeMinimalConfig();
        json["engine_role"] = roleCase.role;
        json["spec_decode_type"] = roleCase.specDecodeType;
        json["builder_config"].erase("ragged_backend");
        EXPECT_THROW(parseEngineConfig(writeJsonToTempFile(json)), std::runtime_error);
    }
}

TEST_F(LLMEngineConfigTest, RejectsDerivedRaggedPhysicalCapacityOverflow)
{
    Json json = makeMinimalConfig();
    json["builder_config"]["max_batch_size"] = std::numeric_limits<int32_t>::max();
    json["builder_config"]["max_input_len"] = 2;
    json["builder_config"]["max_kv_cache_capacity"] = 2;

    EXPECT_THROW(parseEngineConfig(writeJsonToTempFile(json)), std::runtime_error);
}

TEST_F(LLMEngineConfigTest, DerivesRaggedCapacityFromMaxInputLength)
{
    Json json = makeMinimalConfig();
    json["builder_config"]["max_input_len"] = 129;

    LLMEngineConfig const cfg = parseEngineConfig(writeJsonToTempFile(json));
    EXPECT_EQ(cfg.maxQueryLength, 129);
    EXPECT_EQ(cfg.maxPhysicalTokens, 258);
}

TEST_F(LLMEngineConfigTest, SpecBaseRaggedQueryCapacityCoversVerifyWidth)
{
    Json json = makeMinimalConfig();
    json["spec_decode_type"] = "eagle3";
    json["engine_role"] = "base";
    json["builder_config"]["spec_base"] = true;
    json["builder_config"]["max_verify_tree_size"] = 129;

    LLMEngineConfig const treeDominates = parseEngineConfig(writeJsonToTempFile(json));
    EXPECT_EQ(treeDominates.maxQueryLength, 129);
    EXPECT_EQ(treeDominates.maxPhysicalTokens, 258);

    json["builder_config"]["max_input_len"] = 160;
    json["builder_config"]["max_verify_tree_size"] = 140;
    EXPECT_EQ(parseEngineConfig(writeJsonToTempFile(json)).maxQueryLength, 160);
}

TEST_F(LLMEngineConfigTest, ResolvesIndependentFullAndSwaPoolPagesPerLayer)
{
    LLMEngineConfig config;
    config.maxSupportedBatchSize = 2;
    config.maxKVCacheCapacity = 1024;
    config.kvPoolPages
        = static_cast<int32_t>(computeMinimumKvPoolPages(config.maxSupportedBatchSize, config.maxKVCacheCapacity)) + 7;
    constexpr int32_t kWINDOW_SIZE = 129;
    config.numSwaPages = static_cast<int32_t>(computeMinimumSwaPoolPages(config.maxSupportedBatchSize, kWINDOW_SIZE));
    KVLayerConfig const swaLayer{8, 128, kWINDOW_SIZE};
    config.kvLayerConfigs = {KVLayerConfig{8, 128}, swaLayer};

    EXPECT_EQ(config.getKVPoolPagesForLayer(KVLayerConfig{8, 128}), config.kvPoolPages);
    EXPECT_EQ(config.getKVPoolPagesForLayer(swaLayer), config.numSwaPages);

    EXPECT_TRUE(config.supportsBoundedSwaKVCache());
    EXPECT_TRUE(config.usesBoundedSwaKVCache());
    EXPECT_EQ(config.getSwaKVCacheModeInputLength(), 1);
    EXPECT_EQ(config.getKVPoolPageProfileForLayer(swaLayer),
        (std::array<int32_t, 3>{config.numSwaPages, config.numSwaPages, config.kvPoolPages}));

    config.setSwaKVCacheMode(SwaKVCacheMode::kFull);
    EXPECT_TRUE(config.supportsBoundedSwaKVCache());
    EXPECT_FALSE(config.usesBoundedSwaKVCache());
    EXPECT_EQ(config.getSwaKVCacheModeInputLength(), 0);
    EXPECT_EQ(config.getKVPoolPagesForLayer(swaLayer), config.kvPoolPages);
    EXPECT_EQ(swaLayer.kvCacheCapacity, kWINDOW_SIZE);

    config.maxKVCacheCapacity = 512;
    config.kvPoolPages
        = static_cast<int32_t>(computeMinimumKvPoolPages(config.maxSupportedBatchSize, config.maxKVCacheCapacity));
    ASSERT_GT(config.numSwaPages, config.kvPoolPages);
    EXPECT_EQ(config.getKVPoolPageProfileForLayer(swaLayer),
        (std::array<int32_t, 3>{config.kvPoolPages, config.kvPoolPages, config.numSwaPages}));
}

TEST_F(LLMEngineConfigTest, EveryInferenceRecipeCarriesTheActiveSwaModeLength)
{
    LLMEngineConfig config;
    config.maxSupportedBatchSize = 2;
    config.maxKVCacheCapacity = 1024;
    config.kvPoolPages
        = static_cast<int32_t>(computeMinimumKvPoolPages(config.maxSupportedBatchSize, config.maxKVCacheCapacity));
    config.numSwaPages
        = static_cast<int32_t>(computeMinimumSwaPoolPages(config.maxSupportedBatchSize, /*slidingWindowCapacity=*/129));
    config.kvLayerConfigs = {KVLayerConfig{8, 128, 129}};

    auto expectModeLength = [&](int64_t expected) {
        EXPECT_EQ(config.prefillDims(/*batch=*/2, /*seqLen=*/16, ExecutionPhase::kContextPrefill).swaKVCacheModeLen,
            expected);
        EXPECT_EQ(config.decodeDims(/*batch=*/2).swaKVCacheModeLen, expected);
        EXPECT_EQ(config.denoiseDims(/*batch=*/2, /*canvasLen=*/16).swaKVCacheModeLen, expected);
        EXPECT_EQ(config.diffusionCommitDims(/*batch=*/2, /*commitLen=*/4).swaKVCacheModeLen, expected);
        EXPECT_EQ(config.specVerifyDims(/*batch=*/2, /*verifySize=*/4).swaKVCacheModeLen, expected);
        EXPECT_EQ(config.proposalDims(/*batch=*/2, /*proposalSize=*/4, /*draftTopK=*/1).swaKVCacheModeLen, expected);
        EXPECT_EQ(config.acceptDims(/*batch=*/2, /*acceptLen=*/2).swaKVCacheModeLen, expected);
        EXPECT_EQ(config.resetDims().swaKVCacheModeLen, expected);
    };

    expectModeLength(1);
    config.setSwaKVCacheMode(SwaKVCacheMode::kFull);
    expectModeLength(0);
}

TEST_F(LLMEngineConfigTest, ParseEagleBaseConditioningMetadata)
{
    Json json = makeMinimalConfig();
    json["spec_decode_type"] = "eagle3";
    json["engine_role"] = "base";
    json["eagle_hidden_state_layers"] = {0, 5, 11};
    json["builder_config"]["spec_base"] = true;
    json["builder_config"]["max_verify_tree_size"] = 8;
    auto const path = writeJsonToTempFile(json);

    LLMEngineConfig const config = parseEngineConfig(path);
    EXPECT_EQ(config.specTargetLayerIds, std::vector<int32_t>({0, 5, 11}));
}

TEST_F(LLMEngineConfigTest, ParseDFlash2BaseContract)
{
    Json json = makeMinimalConfig();
    json["spec_decode_type"] = "dflash";
    json["engine_role"] = "base";
    json["dflash_config"] = {
        {"version", 2},
        {"target_layer_ids", {1, 3, 5, 7, 11}},
        {"block_size", 8},
        {"mask_token_id", 248070},
        {"is_causal", false},
        {"conv_kernel_size", 2},
        {"conv_group_size", 16},
        {"selector_rank", 256},
        {"selector_top_k", 16},
        {"selector_file", "custom_selector.safetensors"},
        {"supports_probabilistic_sampling", true},
    };
    json["builder_config"]["spec_base"] = true;
    json["builder_config"]["max_verify_tree_size"] = 8;
    auto const path = writeJsonToTempFile(json);

    auto const cfg = parseEngineConfig(path);
    EXPECT_EQ(cfg.specDecodeType, SpecDecodeMode::kDFlash);
    EXPECT_EQ(cfg.dflashVersion, DFlashVersion::kV2);
    EXPECT_TRUE(isCachedBlockDraftMode(cfg.specDecodeType));
    EXPECT_EQ(cfg.specDraftBlockSize, 8);
    EXPECT_EQ(cfg.specDraftMaskTokenId, 248070);
    EXPECT_FALSE(cfg.specDraftCausalHead);
    EXPECT_EQ(cfg.specSelectorTopK, 16);
    EXPECT_EQ(cfg.specSelectorRank, 256);
    EXPECT_EQ(cfg.dflash2SelectorFile, "custom_selector.safetensors");
    EXPECT_EQ(cfg.specConvKernelSize, 2);
    EXPECT_EQ(cfg.specConvGroupSize, 16);
    EXPECT_TRUE(cfg.specSupportsProbabilistic);
    EXPECT_EQ(cfg.specTargetLayerIds, std::vector<int32_t>({1, 3, 5, 7, 11}));
}

TEST_F(LLMEngineConfigTest, DFlash2RejectsInvalidTargetLayerContract)
{
    auto makeConfig = [] {
        Json json = makeMinimalConfig();
        json["spec_decode_type"] = "dflash";
        json["engine_role"] = "base";
        json["dflash_config"] = {{"version", 2}, {"target_layer_ids", {1, 3, 5, 7, 11}}, {"block_size", 8},
            {"mask_token_id", 248070}, {"is_causal", false}, {"conv_kernel_size", 2}, {"conv_group_size", 16},
            {"selector_rank", 256}, {"selector_top_k", 16}, {"supports_probabilistic_sampling", true}};
        json["builder_config"]["spec_base"] = true;
        json["builder_config"]["max_verify_tree_size"] = 8;
        return json;
    };

    Json wrongCount = makeConfig();
    wrongCount["dflash_config"]["target_layer_ids"] = {1, 3, 5, 7};
    EXPECT_THROW(parseEngineConfig(writeJsonToTempFile(wrongCount)), std::runtime_error);

    Json duplicate = makeConfig();
    duplicate["dflash_config"]["target_layer_ids"] = {1, 3, 5, 5, 11};
    EXPECT_THROW(parseEngineConfig(writeJsonToTempFile(duplicate)), std::runtime_error);
}

TEST_F(LLMEngineConfigTest, LegacyDFlashMissingVersionDefaultsToV1)
{
    Json json = makeMinimalConfig();
    json["spec_decode_type"] = "dflash";
    json["engine_role"] = "base";
    json["dflash_config"] = {{"target_layer_ids", {1, 3, 5, 7, 11}}, {"block_size", 16}, {"mask_token_id", 248070}};
    json["builder_config"]["spec_base"] = true;
    json["builder_config"]["max_verify_tree_size"] = 8;
    auto const path = writeJsonToTempFile(json);

    auto const cfg = parseEngineConfig(path);
    EXPECT_EQ(cfg.dflashVersion, DFlashVersion::kV1);
    EXPECT_FALSE(cfg.specSupportsProbabilistic);
}

TEST_F(LLMEngineConfigTest, MissingKVPoolPagesThrows)
{
    Json json = makeMinimalConfig();
    json["builder_config"].erase("max_kv_pool_pages");
    auto const path = writeJsonToTempFile(json);

    EXPECT_THROW(parseEngineConfig(path), std::runtime_error);
}

TEST_F(LLMEngineConfigTest, KVPoolPagesUsesSerializedValue)
{
    Json json = makeMinimalConfig();
    json["builder_config"]["max_kv_pool_pages"] = 9;
    auto const path = writeJsonToTempFile(json);

    EXPECT_EQ(parseEngineConfig(path).kvPoolPages, 9);
}

TEST_F(LLMEngineConfigTest, KVPoolPagesBelowMinimumActivePagesAreRejected)
{
    Json json = makeMinimalConfig();
    json["builder_config"]["max_kv_pool_pages"] = 3;
    auto const path = writeJsonToTempFile(json);

    EXPECT_THROW(parseEngineConfig(path), std::runtime_error);
}

TEST_F(LLMEngineConfigTest, KVPoolPagesRejectDerivedVIdOverflow)
{
    Json json = makeMinimalConfig();
    json["builder_config"]["max_kv_pool_pages"] = kMAX_KV_POOL_PAGES + 1;
    auto const path = writeJsonToTempFile(json);

    EXPECT_THROW(parseEngineConfig(path), std::runtime_error);
}

TEST_F(LLMEngineConfigTest, MinimumKVPoolPagesRejectNarrowingOverflow)
{
    Json json = makeMinimalConfig();
    json["builder_config"]["max_batch_size"] = std::numeric_limits<int32_t>::max();
    json["builder_config"]["max_input_len"] = kMAX_KV_CACHE_CAPACITY;
    json["builder_config"]["max_kv_cache_capacity"] = kMAX_KV_CACHE_CAPACITY;
    auto const path = writeJsonToTempFile(json);

    EXPECT_THROW(parseEngineConfig(path), std::runtime_error);
}

TEST_F(LLMEngineConfigTest, KVCapacityRejectsPageAlignmentOverflow)
{
    Json json = makeMinimalConfig();
    json["builder_config"]["max_input_len"] = static_cast<int64_t>(kMAX_KV_CACHE_CAPACITY) + 1;
    json["builder_config"]["max_kv_cache_capacity"] = static_cast<int64_t>(kMAX_KV_CACHE_CAPACITY) + 1;
    auto const path = writeJsonToTempFile(json);

    EXPECT_THROW(parseEngineConfig(path), std::runtime_error);
}

TEST_F(LLMEngineConfigTest, RankConfigsApplyOverridesForRequestedRank)
{
    Json json = makeMinimalConfig();
    json["builder_config"]["tp_size"] = 2;
    json["rank_configs"] = Json::array({
        Json{{"rank", 0}, {"config_overrides", Json{{"num_key_value_heads", 2}}}},
        Json{{"rank", 1}, {"config_overrides", Json{{"num_key_value_heads", 1}}}},
    });
    auto const path = writeJsonToTempFile(json);

    LLMEngineConfig const cfg = parseEngineConfig(path, /*rank=*/1, /*expectedWorldSize=*/2);
    EXPECT_EQ(cfg.numKVHeads, 1);
}

TEST_F(LLMEngineConfigTest, RankConfigsRequireUniqueRanks)
{
    Json json = makeMinimalConfig();
    json["builder_config"]["tp_size"] = 2;
    json["rank_configs"] = Json::array({Json{{"rank", 0}}, Json{{"rank", 0}}});
    auto const path = writeJsonToTempFile(json);

    EXPECT_THROW(parseEngineConfig(path, /*rank=*/0, /*expectedWorldSize=*/2), std::runtime_error);
}

TEST_F(LLMEngineConfigTest, RankConfigCountMustMatchTensorParallelSize)
{
    Json json = makeMinimalConfig();
    json["builder_config"]["tp_size"] = 1;
    json["rank_configs"] = Json::array({Json{{"rank", 0}}, Json{{"rank", 1}}});
    auto const path = writeJsonToTempFile(json);

    EXPECT_THROW(parseEngineConfig(path, /*rank=*/0, /*expectedWorldSize=*/2), std::runtime_error);
}

TEST_F(LLMEngineConfigTest, RankConfigCountMustMatchRuntimeWorldSize)
{
    Json json = makeMinimalConfig();
    json["builder_config"]["tp_size"] = 2;
    json["rank_configs"] = Json::array({Json{{"rank", 0}}, Json{{"rank", 1}}});
    auto const path = writeJsonToTempFile(json);

    EXPECT_THROW(parseEngineConfig(path, /*rank=*/0, /*expectedWorldSize=*/1), std::runtime_error);
}

// rank_configs describes rank-local differences and is required only when more than one rank participates.
TEST_F(LLMEngineConfigTest, SingleDeviceConfigLoadsWithoutRankConfigs)
{
    Json json = makeMinimalConfig();
    json["builder_config"]["tp_size"] = 1;
    auto const path = writeJsonToTempFile(json);

    LLMEngineConfig const cfg = parseEngineConfig(path, /*rank=*/0, /*expectedWorldSize=*/1);
    EXPECT_EQ(cfg.numKVHeads, 4);
}

TEST_F(LLMEngineConfigTest, TensorParallelConfigRequiresRankConfigs)
{
    Json json = makeMinimalConfig();
    json["builder_config"]["tp_size"] = 2;
    auto const path = writeJsonToTempFile(json);

    EXPECT_THROW(parseEngineConfig(path, /*rank=*/0, /*expectedWorldSize=*/2), std::runtime_error);
}

TEST_F(LLMEngineConfigTest, ReducedVocabSize)
{
    Json json = makeMinimalConfig();
    json["reduced_vocab_size"] = 16000;
    auto const path = writeJsonToTempFile(json);

    LLMEngineConfig cfg = parseEngineConfig(path);
    EXPECT_EQ(cfg.reducedVocabSize, 16000);
    EXPECT_EQ(cfg.outputVocabSize, 16000);
}

TEST_F(LLMEngineConfigTest, PartialRotaryFactor)
{
    Json json = makeMinimalConfig();
    json["partial_rotary_factor"] = 0.5;
    auto const path = writeJsonToTempFile(json);

    LLMEngineConfig cfg = parseEngineConfig(path);
    // rotaryDim = headDim * partial_rotary_factor = 64 * 0.5 = 32
    EXPECT_EQ(cfg.rotaryDim, 32);
}

TEST_F(LLMEngineConfigTest, DynamicNtkAlphaRescalesRopeBase)
{
    // HunYuan V1 DynamicNTKAlpha: base = rope_theta * alpha^(head_dim / (head_dim - 2)).
    Json json = makeMinimalConfig();
    json["rope_theta"] = 10000.0F;
    json["rope_scaling"] = {{"type", "dynamic"}, {"alpha", 1000.0F}, {"factor", 1.0F}};
    auto const path = writeJsonToTempFile(json);

    LLMEngineConfig cfg = parseEngineConfig(path);
    EXPECT_EQ(cfg.ropeConfig.type, RopeType::kDynamic);
    float const headDim = static_cast<float>(cfg.headDim);
    float const expected = 10000.0F * std::pow(1000.0F, headDim / (headDim - 2.0F));
    EXPECT_NEAR(cfg.ropeConfig.rotaryTheta / expected, 1.0F, 1e-5F);
}

TEST_F(LLMEngineConfigTest, DynamicRopeWithoutAlphaKeepsBase)
{
    Json json = makeMinimalConfig();
    json["rope_theta"] = 10000.0F;
    json["rope_scaling"] = {{"type", "dynamic"}, {"factor", 2.0F}};
    auto const path = writeJsonToTempFile(json);

    LLMEngineConfig cfg = parseEngineConfig(path);
    EXPECT_EQ(cfg.ropeConfig.type, RopeType::kDynamic);
    EXPECT_FLOAT_EQ(cfg.ropeConfig.rotaryTheta, 10000.0F);
}

TEST_F(LLMEngineConfigTest, DynamicNtkAlphaFallsBackToDerivedHeadDim)
{
    // collectRopeConfig may see configs without an explicit head_dim (e.g. per-block
    // RoPE JSON); the alpha rescale then derives it from hidden_size / num_attention_heads.
    Json json;
    json["rope_theta"] = 10000.0F;
    json["rope_scaling"] = {{"type", "dynamic"}, {"alpha", 1000.0F}};
    json["hidden_size"] = 4096;
    json["num_attention_heads"] = 32;
    json["max_position_embeddings"] = 262144;

    RopeConfig const ropeConfig = collectRopeConfig(json);
    EXPECT_EQ(ropeConfig.type, RopeType::kDynamic);
    float const expected = 10000.0F * std::pow(1000.0F, 128.0F / 126.0F);
    EXPECT_NEAR(ropeConfig.rotaryTheta / expected, 1.0F, 1e-5F);
}

TEST_F(LLMEngineConfigTest, HybridModelFields)
{
    Json const json = makeHybridConfig();
    auto const path = writeJsonToTempFile(json);

    LLMEngineConfig cfg = parseEngineConfig(path);
    EXPECT_EQ(cfg.numLinearAttnLayers, 4);
    EXPECT_EQ(cfg.numAttentionLayers, 8);
    EXPECT_EQ(cfg.recurrentStateNumHeads, 16);
    EXPECT_EQ(cfg.recurrentStateHeadDim, 32);
    EXPECT_EQ(cfg.recurrentStateSize, 64);
    EXPECT_EQ(cfg.convDim, 128);
    EXPECT_EQ(cfg.convKernel, 4);
    EXPECT_EQ(cfg.recurrentStateDtype, nvinfer1::DataType::kHALF);
    EXPECT_EQ(cfg.convStateDtype, nvinfer1::DataType::kHALF);
}

TEST_F(LLMEngineConfigTest, HybridMissingRecurrentDtypeThrows)
{
    Json json = makeHybridConfig();
    json.erase("recurrent_state_dtype");
    auto const path = writeJsonToTempFile(json);

    EXPECT_THROW(parseEngineConfig(path), std::runtime_error);
}

TEST_F(LLMEngineConfigTest, HybridMissingConvDtypeThrows)
{
    Json json = makeHybridConfig();
    json.erase("conv_state_dtype");
    auto const path = writeJsonToTempFile(json);

    EXPECT_THROW(parseEngineConfig(path), std::runtime_error);
}

TEST_F(LLMEngineConfigTest, HybridInvalidRecurrentDtypeThrows)
{
    Json json = makeHybridConfig();
    json["recurrent_state_dtype"] = "garbage";
    auto const path = writeJsonToTempFile(json);

    EXPECT_THROW(parseEngineConfig(path), std::runtime_error);
}

TEST_F(LLMEngineConfigTest, MissingOptionalFieldsGetDefaults)
{
    Json const json = makeMinimalConfig();
    auto const path = writeJsonToTempFile(json);

    LLMEngineConfig cfg = parseEngineConfig(path);

    // All optional hybrid fields default to 0.
    EXPECT_EQ(cfg.numLinearAttnLayers, 0);
    EXPECT_EQ(cfg.recurrentStateNumHeads, 0);
    EXPECT_EQ(cfg.recurrentStateHeadDim, 0);
    EXPECT_EQ(cfg.recurrentStateSize, 0);
    EXPECT_EQ(cfg.convDim, 0);
    EXPECT_EQ(cfg.convKernel, 0);
    EXPECT_EQ(cfg.numDeepstackFeatures, 0);
    EXPECT_EQ(cfg.imageTokenId, -1);
    EXPECT_EQ(cfg.audioTokenId, -1);
    EXPECT_NE(cfg.ropeConfig.type, RopeType::kMRope);
    // numAttentionLayers defaults to numDecoderLayers when not specified.
    EXPECT_EQ(cfg.numAttentionLayers, cfg.numDecoderLayers);
}

TEST_F(LLMEngineConfigTest, SpecDecodeMaxProposalSizes)
{
    Json json = makeMinimalConfig();
    json["spec_decode_type"] = "eagle3";
    json["engine_role"] = "base";
    json["builder_config"]["spec_base"] = true;
    json["builder_config"]["max_verify_tree_size"] = 16;
    json["builder_config"]["num_swa_pages"] = 64; // Unused fallback budget must not enable SWA.
    // `max_draft_tree_size` is a draft-engine property and is not written
    // into base_config.json by the builder — intentionally omitted here.
    auto const path = writeJsonToTempFile(json);

    LLMEngineConfig cfg = parseEngineConfig(path);
    EXPECT_TRUE(cfg.isSpecDecodeBase);
    EXPECT_EQ(cfg.maxVerifyTreeSize, 16);
    EXPECT_EQ(cfg.maxDraftTreeSize, 0); // Base side leaves this at the default.
    EXPECT_EQ(cfg.numSwaPages, 64);
    EXPECT_TRUE(std::all_of(cfg.kvLayerConfigs.begin(), cfg.kvLayerConfigs.end(),
        [](KVLayerConfig const& layer) { return layer.kvCacheCapacity == 0; }));
    // baseOutputHiddenDim = hiddenSize * 3 = 768 * 3 = 2304; computed at DeploymentConfig level
    EXPECT_EQ(cfg.hiddenSize * 3, 2304);
}

TEST_F(LLMEngineConfigTest, KVCacheDtypeFP8)
{
    Json json = makeMinimalConfig();
    json["kv_cache_dtype"] = "fp8";
    auto const path = writeJsonToTempFile(json);

    LLMEngineConfig cfg = parseEngineConfig(path);
    EXPECT_EQ(cfg.kvCacheDtype, nvinfer1::DataType::kFP8);
}

TEST_F(LLMEngineConfigTest, MissingKVCacheDtypeThrows)
{
    Json json = makeMinimalConfig();
    json.erase("kv_cache_dtype");
    auto const path = writeJsonToTempFile(json);

    EXPECT_THROW(parseEngineConfig(path), std::runtime_error);
}

TEST_F(LLMEngineConfigTest, InvalidKVCacheDtypeThrows)
{
    Json json = makeMinimalConfig();
    json["kv_cache_dtype"] = "garbage";
    auto const path = writeJsonToTempFile(json);

    EXPECT_THROW(parseEngineConfig(path), std::runtime_error);
}

TEST_F(LLMEngineConfigTest, MissingRequiredFieldThrows)
{
    Json json = makeMinimalConfig();
    json.erase("num_hidden_layers"); // Remove a required field.
    auto const path = writeJsonToTempFile(json);

    EXPECT_THROW(parseEngineConfig(path), std::runtime_error);
}

TEST_F(LLMEngineConfigTest, MissingBuilderConfigThrows)
{
    Json json = makeMinimalConfig();
    json.erase("builder_config");
    auto const path = writeJsonToTempFile(json);

    EXPECT_THROW(parseEngineConfig(path), std::runtime_error);
}

TEST_F(LLMEngineConfigTest, InvalidMaxInputLenThrows)
{
    Json json = makeMinimalConfig();
    // max_input_len > max_kv_cache_capacity is invalid.
    json["builder_config"]["max_input_len"] = 512;
    json["builder_config"]["max_kv_cache_capacity"] = 256;
    auto const path = writeJsonToTempFile(json);

    EXPECT_THROW(parseEngineConfig(path), std::runtime_error);
}

TEST_F(LLMEngineConfigTest, InvalidPositiveFieldThrows)
{
    Json json = makeMinimalConfig();
    json["num_hidden_layers"] = 0; // Must be positive.
    auto const path = writeJsonToTempFile(json);

    EXPECT_THROW(parseEngineConfig(path), std::runtime_error);
}

TEST_F(LLMEngineConfigTest, FileNotFoundThrows)
{
    EXPECT_THROW(parseEngineConfig(std::filesystem::path("/tmp/nonexistent_config_12345.json")), std::runtime_error);
}

TEST_F(LLMEngineConfigTest, MalformedJsonThrows)
{
    auto tmpPath = std::filesystem::temp_directory_path() / "llmEngineConfigTest_config.json";
    std::ofstream ofs(tmpPath);
    ofs << "{ this is not valid json }}}";
    ofs.close();

    EXPECT_THROW(parseEngineConfig(tmpPath), std::runtime_error);
}

TEST_F(LLMEngineConfigTest, FormatEngineConfigDoesNotCrash)
{
    Json const json = makeMinimalConfig();
    auto const path = writeJsonToTempFile(json);

    LLMEngineConfig cfg = parseEngineConfig(path);
    std::string const formatted = formatEngineConfig(cfg);
    EXPECT_FALSE(formatted.empty());
    EXPECT_TRUE(formatted.find("LLMEngineConfig") != std::string::npos);
}

TEST_F(LLMEngineConfigTest, SpecDecodeMissingVerifyTreeSizeThrows)
{
    Json json = makeMinimalConfig();
    json["spec_decode_type"] = "eagle3";
    json["engine_role"] = "base";
    json["builder_config"]["spec_base"] = true;
    // Intentionally omit max_verify_tree_size (the only required specConfig field on the base).
    auto const path = writeJsonToTempFile(json);

    EXPECT_THROW(parseEngineConfig(path), std::runtime_error);
}

TEST_F(LLMEngineConfigTest, DeepstackAndMultimodal)
{
    Json json = makeMinimalConfig();
    json["num_deepstack_features"] = 4;
    json["image_token_id"] = 151655;
    json["audio_token_id"] = 151656;
    auto const path = writeJsonToTempFile(json);

    LLMEngineConfig cfg = parseEngineConfig(path);
    EXPECT_EQ(cfg.numDeepstackFeatures, 4);
    EXPECT_EQ(cfg.imageTokenId, 151655);
    EXPECT_EQ(cfg.audioTokenId, 151656);
}

TEST_F(LLMEngineConfigTest, DiffusionGemmaSamplerConfig)
{
    Json json = makeMinimalConfig();
    json["engine_role"] = "dllm";
    json["decoding_strategy"] = "block_diffusion";
    json["context_mask_selector_enabled"] = true;
    json["diffusion_unified_conditioning"] = true;
    json["self_conditioning_size"] = 256;
    json["diffusion_config"] = {
        {"canvas_length", 8},
        {"max_denoising_steps", 48},
        {"t_max", 0.9},
        {"t_min", 0.3},
        {"entropy_bound", 0.2},
        {"entropy_threshold", 0.01},
        {"stability_window", 3},
    };
    auto const path = writeJsonToTempFile(json);

    LLMEngineConfig cfg = parseEngineConfig(path);
    EXPECT_TRUE(cfg.isDiffusionBackbone);
    EXPECT_TRUE(cfg.contextMaskSelectorEnabled);
    EXPECT_TRUE(cfg.diffusionUnifiedConditioning);
    EXPECT_EQ(cfg.diffusionCanvasLength, 8);
    EXPECT_EQ(cfg.diffusionMaxDenoisingSteps, 48);
    EXPECT_EQ(cfg.diffusionSelfConditioningSize, 256);
    EXPECT_FLOAT_EQ(cfg.diffusionTMax, 0.9F);
    EXPECT_FLOAT_EQ(cfg.diffusionTMin, 0.3F);
    EXPECT_FLOAT_EQ(cfg.diffusionEntropyBound, 0.2F);
    EXPECT_FLOAT_EQ(cfg.diffusionEntropyThreshold, 0.01F);
    EXPECT_EQ(cfg.diffusionStabilityWindow, 3);
}

TEST_F(LLMEngineConfigTest, DiffusionRaggedQueryCapacityCoversCanvasWidth)
{
    Json json = makeMinimalConfig();
    json["engine_role"] = "dllm";
    json["diffusion_unified_conditioning"] = true;
    json["diffusion_config"] = {{"canvas_length", 129}};

    LLMEngineConfig const cfg = parseEngineConfig(writeJsonToTempFile(json));
    EXPECT_EQ(cfg.maxQueryLength, 129);
    EXPECT_EQ(cfg.maxPhysicalTokens, 258);
}

TEST_F(LLMEngineConfigTest, DiffusionGemmaDllmRequiresUnifiedConditioning)
{
    Json json = makeMinimalConfig();
    json["engine_role"] = "dllm";
    json["self_conditioning_size"] = 256;
    json["diffusion_config"] = {
        {"canvas_length", 8},
        {"max_denoising_steps", 48},
    };
    auto const path = writeJsonToTempFile(json);

    EXPECT_THROW(parseEngineConfig(path), std::runtime_error);
}

// ===========================================================================
// Layer-types and kv_layer_configs parsing
// ===========================================================================

TEST_F(LLMEngineConfigTest, ParsesCanonicalLayerTypes)
{
    // Heterogeneous config: 4 layers — 2 attention (different head dims!) + 2 mamba
    auto const tmp = writeTempConfig(R"({
        "num_hidden_layers": 4,
        "num_key_value_heads": 8,
        "head_dim": 64,
        "hidden_size": 512,
        "vocab_size": 32000,
        "kv_cache_dtype": "fp16",
        "layer_types": ["attention", "mamba", "attention", "mamba"],
        "kv_layer_configs": [
            {"num_kv_heads": 8, "head_dim": 64},
            null,
            {"num_kv_heads": 4, "head_dim": 128},
            null
        ],
        "num_linear_attn_layers": 2,
        "num_attention_layers": 2,
        "recurrent_state_dtype": "fp16",
        "conv_state_dtype": "fp16",
        "recurrent_state_num_heads": 4,
        "recurrent_state_head_dim": 64,
        "recurrent_state_size": 128,
        "conv_dim": 128,
        "conv_kernel": 4,
        "builder_config": {
            "max_batch_size": 1,
            "max_input_len": 64,
            "max_kv_cache_capacity": 128,
            "max_kv_pool_pages": 1
        }
    })");

    auto const cfg = parseEngineConfig(tmp);
    ASSERT_EQ(cfg.layerTypes.size(), 4u);
    EXPECT_EQ(cfg.layerTypes[0], rt::HybridCacheManager::LayerType::kAttention);
    EXPECT_EQ(cfg.layerTypes[1], rt::HybridCacheManager::LayerType::kMamba);
    EXPECT_EQ(cfg.layerTypes[2], rt::HybridCacheManager::LayerType::kAttention);
    EXPECT_EQ(cfg.layerTypes[3], rt::HybridCacheManager::LayerType::kMamba);
    ASSERT_EQ(cfg.kvLayerConfigs.size(), 2u);
    EXPECT_EQ(cfg.kvLayerConfigs[0].numKVHeads, 8);
    EXPECT_EQ(cfg.kvLayerConfigs[0].headDim, 64);
    EXPECT_EQ(cfg.kvLayerConfigs[0].kvCacheCapacity, 0);
    EXPECT_EQ(cfg.kvLayerConfigs[1].numKVHeads, 4);
    EXPECT_EQ(cfg.kvLayerConfigs[1].headDim, 128);
    EXPECT_EQ(cfg.kvLayerConfigs[1].kvCacheCapacity, 0);
}

TEST_F(LLMEngineConfigTest, ParsesPerLayerKVCacheCapacityMarkers)
{
    Json json = makeMinimalConfig();
    json["num_hidden_layers"] = 4;
    json["builder_config"]["num_swa_pages"] = 64;
    json["layer_types"] = {"attention", "attention", "attention", "attention"};
    json["kv_layer_configs"] = Json::array({
        Json{{"num_kv_heads", 4}, {"head_dim", 64}},
        Json{{"num_kv_heads", 4}, {"head_dim", 64}, {"kv_cache_capacity", 0}},
        Json{{"num_kv_heads", 4}, {"head_dim", 64}, {"kv_cache_capacity", 128}},
        Json{{"num_kv_heads", 4}, {"head_dim", 64}, {"kv_cache_capacity", 256}},
    });
    auto const path = writeJsonToTempFile(json);

    LLMEngineConfig cfg = parseEngineConfig(path);
    ASSERT_EQ(cfg.kvLayerConfigs.size(), 4u);
    EXPECT_EQ(cfg.numSwaPages, 64);
    EXPECT_EQ(cfg.kvLayerConfigs[0].kvCacheCapacity, 0);   // Missing marker remains compatibility-full.
    EXPECT_EQ(cfg.kvLayerConfigs[1].kvCacheCapacity, 0);   // Explicit zero is also full.
    EXPECT_EQ(cfg.kvLayerConfigs[2].kvCacheCapacity, 128); // Dedicated SWA window.
    EXPECT_EQ(cfg.kvLayerConfigs[3].kvCacheCapacity, 256); // Explicit engine capacity is full.
    EXPECT_TRUE(cfg.usesBoundedSwaKVCache());

    cfg.setSwaKVCacheMode(SwaKVCacheMode::kFull);
    EXPECT_FALSE(cfg.usesBoundedSwaKVCache());
    EXPECT_EQ(cfg.kvLayerConfigs[2].kvCacheCapacity, 128); // Runtime policy never erases serialized capability.
}

TEST_F(LLMEngineConfigTest, RejectsMixedReducedKVCacheCapacities)
{
    Json json = makeMinimalConfig();
    json["num_hidden_layers"] = 2;
    json["builder_config"]["num_swa_pages"] = 64;
    json["layer_types"] = {"attention", "attention"};
    json["kv_layer_configs"] = Json::array({
        Json{{"num_kv_heads", 4}, {"head_dim", 64}, {"kv_cache_capacity", 64}},
        Json{{"num_kv_heads", 4}, {"head_dim", 64}, {"kv_cache_capacity", 128}},
    });
    auto const path = writeJsonToTempFile(json);
    EXPECT_THROW(parseEngineConfig(path), std::runtime_error);
}

TEST_F(LLMEngineConfigTest, RejectsReducedKVCacheCapacityForFp8)
{
    Json json = makeMinimalConfig();
    json["num_hidden_layers"] = 1;
    json["kv_cache_dtype"] = "fp8";
    json["builder_config"]["num_swa_pages"] = 64;
    json["layer_types"] = {"attention"};
    json["kv_layer_configs"] = Json::array({Json{{"num_kv_heads", 4}, {"head_dim", 64}, {"kv_cache_capacity", 128}}});
    auto const path = writeJsonToTempFile(json);
    EXPECT_THROW(parseEngineConfig(path), std::runtime_error);
}

TEST_F(LLMEngineConfigTest, RejectsReducedKVCacheCapacityWhenSpecFlagIsSet)
{
    Json json = makeMinimalConfig();
    json["num_hidden_layers"] = 1;
    json["builder_config"]["spec_base"] = true;
    json["builder_config"]["num_swa_pages"] = 64;
    json["layer_types"] = {"attention"};
    json["kv_layer_configs"] = Json::array({Json{{"num_kv_heads", 4}, {"head_dim", 64}, {"kv_cache_capacity", 128}}});
    auto const path = writeJsonToTempFile(json);
    EXPECT_THROW(parseEngineConfig(path), std::runtime_error);
}

TEST_F(LLMEngineConfigTest, ReducedKVCacheCapacityRequiresConfiguredSwaPages)
{
    Json json = makeMinimalConfig();
    json["num_hidden_layers"] = 1;
    json["layer_types"] = {"attention"};
    json["kv_layer_configs"] = Json::array({Json{{"num_kv_heads", 4}, {"head_dim", 64}, {"kv_cache_capacity", 128}}});

    auto path = writeJsonToTempFile(json);
    EXPECT_THROW(parseEngineConfig(path), std::runtime_error);

    json["builder_config"]["num_swa_pages"]
        = computeMinimumSwaPoolPages(/*maxBatchSize=*/2, /*slidingWindowCapacity=*/128) - 1;
    path = writeJsonToTempFile(json);
    EXPECT_THROW(parseEngineConfig(path), std::runtime_error);
}

TEST_F(LLMEngineConfigTest, Fp8FallbackWithoutMarkerDoesNotEnableReducedPool)
{
    Json json = makeMinimalConfig();
    json["kv_cache_dtype"] = "fp8";
    json["builder_config"]["num_swa_pages"] = 64;
    auto const path = writeJsonToTempFile(json);

    LLMEngineConfig const cfg = parseEngineConfig(path);
    EXPECT_EQ(cfg.numSwaPages, 64);
    ASSERT_FALSE(cfg.kvLayerConfigs.empty());
    EXPECT_TRUE(std::all_of(cfg.kvLayerConfigs.begin(), cfg.kvLayerConfigs.end(),
        [](KVLayerConfig const& layer) { return !isReducedKvCacheCapacity(layer.kvCacheCapacity, 256); }));
}

TEST_F(LLMEngineConfigTest, FallbackBuildsLayerTypesFromScalarsPureAttention)
{
    auto const tmp = writeTempConfig(R"({
        "num_hidden_layers": 3,
        "num_key_value_heads": 8,
        "head_dim": 64,
        "hidden_size": 512,
        "vocab_size": 32000,
        "kv_cache_dtype": "fp16",
        "builder_config": {
            "max_batch_size": 1,
            "max_input_len": 64,
            "max_kv_cache_capacity": 128,
            "max_kv_pool_pages": 1
        }
    })");

    auto const cfg = parseEngineConfig(tmp);
    ASSERT_EQ(cfg.layerTypes.size(), 3u);
    for (auto const& lt : cfg.layerTypes)
        EXPECT_EQ(lt, rt::HybridCacheManager::LayerType::kAttention);
    ASSERT_EQ(cfg.kvLayerConfigs.size(), 3u);
    for (auto const& lc : cfg.kvLayerConfigs)
    {
        EXPECT_EQ(lc.numKVHeads, 8);
        EXPECT_EQ(lc.headDim, 64);
        EXPECT_EQ(lc.kvCacheCapacity, 0);
    }
}

TEST_F(LLMEngineConfigTest, FallbackHybridBuildsAttentionFirstThenMamba)
{
    // 4 total: 2 attention + 2 mamba
    auto const tmp = writeTempConfig(R"({
        "num_hidden_layers": 4,
        "num_key_value_heads": 8,
        "head_dim": 64,
        "hidden_size": 512,
        "vocab_size": 32000,
        "kv_cache_dtype": "fp16",
        "num_attention_layers": 2,
        "num_linear_attn_layers": 2,
        "recurrent_state_dtype": "fp16",
        "conv_state_dtype": "fp16",
        "recurrent_state_num_heads": 4,
        "recurrent_state_head_dim": 64,
        "recurrent_state_size": 128,
        "conv_dim": 128,
        "conv_kernel": 4,
        "builder_config": {
            "max_batch_size": 1,
            "max_input_len": 64,
            "max_kv_cache_capacity": 128,
            "max_kv_pool_pages": 1
        }
    })");

    auto const cfg = parseEngineConfig(tmp);
    ASSERT_EQ(cfg.layerTypes.size(), 4u);
    EXPECT_EQ(cfg.layerTypes[0], rt::HybridCacheManager::LayerType::kAttention);
    EXPECT_EQ(cfg.layerTypes[1], rt::HybridCacheManager::LayerType::kAttention);
    EXPECT_EQ(cfg.layerTypes[2], rt::HybridCacheManager::LayerType::kMamba);
    EXPECT_EQ(cfg.layerTypes[3], rt::HybridCacheManager::LayerType::kMamba);
    EXPECT_EQ(cfg.kvLayerConfigs.size(), 2u); // attention count
}

// ===========================================================================
// InferenceDims recipe methods
//
// The recipes are pure functions of (config, arguments). Tests construct a
// config directly (bypassing JSON parsing) to exercise the math in isolation.
// ===========================================================================

namespace
{

//! Construct a config with just the fields the recipes read. Other fields are
//! left at their defaults; recipes do not touch them.
LLMEngineConfig makeRecipeConfig(int32_t maxKV, bool mrope)
{
    LLMEngineConfig cfg;
    cfg.maxKVCacheCapacity = maxKV;
    cfg.ropeConfig.type = mrope ? RopeType::kMRope : RopeType::kDefault;
    return cfg;
}

} // namespace

TEST(LLMEngineConfigRecipesTest, PrefillDims)
{
    auto const cfg = makeRecipeConfig(/*maxKV=*/4096, /*mrope=*/false);
    auto const d = cfg.prefillDims(/*batch=*/2, /*seqLen=*/128, ExecutionPhase::kContextPrefill);
    EXPECT_EQ(d.batch, 2);
    EXPECT_EQ(d.seqLen, 256);
    EXPECT_EQ(d.kvLen, 4096);
    EXPECT_EQ(d.selectLen, 2);
    EXPECT_EQ(d.attnMaskSeqLen, 256); // token-aligned attention metadata
    EXPECT_EQ(d.ropeBatch, 1);        // non-MRope
    EXPECT_EQ(d.packedMaskLen, 1);
    EXPECT_EQ(d.contextMaskSelectorLen, 0);
    EXPECT_EQ(d.startIndexLen, 0); // plugin-path empty-cache sentinel
    EXPECT_EQ(d.executionPhaseLen, static_cast<int64_t>(ExecutionPhase::kContextPrefill));
}

TEST(LLMEngineConfigRecipesTest, PrefillDimsMRope)
{
    auto const cfg = makeRecipeConfig(/*maxKV=*/4096, /*mrope=*/true);
    auto const d = cfg.prefillDims(/*batch=*/3, /*seqLen=*/65, ExecutionPhase::kContextPrefill);
    EXPECT_EQ(d.ropeBatch, 3);        // MRope → batch
    EXPECT_EQ(d.attnMaskSeqLen, 195); // token-aligned attention metadata
    EXPECT_EQ(d.packedMaskLen, 1);
    EXPECT_EQ(d.contextMaskSelectorLen, 0);
    EXPECT_EQ(d.startIndexLen, 0); // plugin-path empty-cache sentinel
    EXPECT_EQ(d.executionPhaseLen, static_cast<int64_t>(ExecutionPhase::kContextPrefill));
}

TEST(LLMEngineConfigRecipesTest, PrefillDimsChunked)
{
    // Chunked prefill (cache non-empty) uses [batch] startIndexLen.
    auto const cfg = makeRecipeConfig(/*maxKV=*/4096, /*mrope=*/false);
    auto const d = cfg.prefillDims(/*batch=*/2, /*seqLen=*/128, ExecutionPhase::kContextChunk);
    EXPECT_EQ(d.startIndexLen, 2);
    EXPECT_EQ(d.executionPhaseLen, static_cast<int64_t>(ExecutionPhase::kContextChunk));
}

TEST(LLMEngineConfigRecipesTest, DiffusionGemmaInitialPrefillBindsFullKVCapacity)
{
    LLMEngineConfig cfg = makeRecipeConfig(/*maxKV=*/1024, /*mrope=*/false);
    cfg.isDiffusionBackbone = true;
    auto const d = cfg.prefillDims(/*batch=*/1, /*seqLen=*/36, ExecutionPhase::kDiffusionCommit);
    EXPECT_EQ(d.batch, 1);
    EXPECT_EQ(d.seqLen, 36);
    EXPECT_EQ(d.kvLen, 1024);
    EXPECT_EQ(d.selectLen, 1);
    EXPECT_EQ(d.contextMaskSelectorLen, 0);
    EXPECT_EQ(d.startIndexLen, 1);
    EXPECT_EQ(d.executionPhaseLen, static_cast<int64_t>(ExecutionPhase::kDiffusionCommit));
    EXPECT_EQ(d.contextSequenceCount, 0);
    EXPECT_THROW(cfg.prefillDims(/*batch=*/1, /*seqLen=*/36, ExecutionPhase::kContextPrefill), std::runtime_error);
}

TEST(LLMEngineConfigRecipesTest, DiffusionGemmaDenoiseAndCommitDims)
{
    LLMEngineConfig cfg = makeRecipeConfig(/*maxKV=*/1024, /*mrope=*/false);
    auto const denoise = cfg.denoiseDims(/*batch=*/1, /*canvasLen=*/8);
    EXPECT_EQ(denoise.kvLen, 1024);
    EXPECT_EQ(denoise.seqLen, 8);
    EXPECT_EQ(denoise.selectLen, 8);
    EXPECT_EQ(denoise.attnMaskSeqLen, 8);
    EXPECT_EQ(denoise.contextMaskSelectorLen, 1);
    EXPECT_EQ(denoise.startIndexLen, 1);

    auto const denoiseVarlenBatch = cfg.denoiseDims(/*batch=*/2, /*canvasLen=*/8);
    EXPECT_EQ(denoiseVarlenBatch.kvLen, 1024);
    EXPECT_EQ(denoiseVarlenBatch.seqLen, 16);
    EXPECT_EQ(denoiseVarlenBatch.selectLen, 16);
    EXPECT_EQ(denoiseVarlenBatch.attnMaskSeqLen, 16);
    EXPECT_EQ(denoiseVarlenBatch.contextMaskSelectorLen, 2);
    EXPECT_EQ(denoiseVarlenBatch.startIndexLen, 2);

    auto const commit = cfg.diffusionCommitDims(/*batch=*/1, /*commitLen=*/8);
    EXPECT_EQ(commit.kvLen, 1024);
    EXPECT_EQ(commit.seqLen, 8);
    EXPECT_EQ(commit.selectLen, 8);
    EXPECT_EQ(commit.attnMaskSeqLen, 8);
    EXPECT_EQ(commit.contextMaskSelectorLen, 0);
    EXPECT_EQ(commit.startIndexLen, 1);

    auto const commitVarlenBatch = cfg.diffusionCommitDims(/*batch=*/2, /*commitLen=*/8);
    EXPECT_EQ(commitVarlenBatch.kvLen, 1024);
    EXPECT_EQ(commitVarlenBatch.seqLen, 16);
    EXPECT_EQ(commitVarlenBatch.selectLen, 16);
    EXPECT_EQ(commitVarlenBatch.attnMaskSeqLen, 16);
    EXPECT_EQ(commitVarlenBatch.contextMaskSelectorLen, 0);
    EXPECT_EQ(commitVarlenBatch.startIndexLen, 2);
}

TEST(LLMEngineConfigRecipesTest, DecodeDims)
{
    auto const cfg = makeRecipeConfig(/*maxKV=*/2048, /*mrope=*/false);
    auto const d = cfg.decodeDims(/*batch=*/4);
    EXPECT_EQ(d.batch, 4);
    EXPECT_EQ(d.seqLen, 4);
    EXPECT_EQ(d.kvLen, 2048);
    EXPECT_EQ(d.selectLen, 4);
    EXPECT_EQ(d.attnMaskSeqLen, 4);
    EXPECT_EQ(d.ropeBatch, 1);
    EXPECT_EQ(d.packedMaskLen, 1); // explicit 1 in decode
    EXPECT_EQ(d.startIndexLen, 4); // batch
    EXPECT_EQ(d.executionPhaseLen, static_cast<int64_t>(ExecutionPhase::kAutoregressiveDecode));
}

TEST(LLMEngineConfigRecipesTest, DecodeDimsMRope)
{
    auto const cfg = makeRecipeConfig(/*maxKV=*/2048, /*mrope=*/true);
    auto const d = cfg.decodeDims(/*batch=*/4);
    EXPECT_EQ(d.ropeBatch, 4); // MRope → batch
}

TEST(LLMEngineConfigRecipesTest, RaggedPrefillAndDecodeUsePhysicalTokenExtent)
{
    LLMEngineConfig cfg = makeRecipeConfig(/*maxKV=*/4096, /*mrope=*/false);
    auto const prefill = cfg.prefillDims(/*batch=*/3, /*seqLen=*/5, ExecutionPhase::kContextPrefill);
    EXPECT_EQ(prefill.batch, 3);
    EXPECT_EQ(prefill.seqLen, 15);
    EXPECT_EQ(prefill.selectLen, 3);
    EXPECT_EQ(prefill.queryOffsetLen, 4);
    EXPECT_EQ(prefill.executionPhaseLen, static_cast<int64_t>(ExecutionPhase::kContextPrefill));
    EXPECT_EQ(prefill.contextSequenceCount, 3);

    auto const decode = cfg.decodeDims(/*batch=*/3);
    EXPECT_EQ(decode.batch, 3);
    EXPECT_EQ(decode.seqLen, 3);
    EXPECT_EQ(decode.selectLen, 3);
    EXPECT_EQ(decode.queryOffsetLen, 4);
    EXPECT_EQ(decode.executionPhaseLen, static_cast<int64_t>(ExecutionPhase::kAutoregressiveDecode));
    EXPECT_EQ(decode.contextSequenceCount, 0);
}

TEST(LLMEngineConfigRecipesTest, EveryRuntimePhaseHasAnExplicitShapeExtent)
{
    LLMEngineConfig cfg = makeRecipeConfig(/*maxKV=*/4096, /*mrope=*/false);
    EXPECT_EQ(cfg.prefillDims(2, 4, ExecutionPhase::kContextPrefill).executionPhaseLen,
        static_cast<int64_t>(ExecutionPhase::kContextPrefill));
    EXPECT_EQ(cfg.prefillDims(2, 4, ExecutionPhase::kContextChunk).executionPhaseLen,
        static_cast<int64_t>(ExecutionPhase::kContextChunk));
    EXPECT_EQ(cfg.decodeDims(2).executionPhaseLen, static_cast<int64_t>(ExecutionPhase::kAutoregressiveDecode));
    EXPECT_EQ(cfg.proposalDims(2, 8, 4).executionPhaseLen, static_cast<int64_t>(ExecutionPhase::kSpecDraftProposal));
    EXPECT_EQ(cfg.specVerifyDims(2, 8).executionPhaseLen, static_cast<int64_t>(ExecutionPhase::kSpecTargetVerify));
    EXPECT_EQ(cfg.denoiseDims(2, 8).executionPhaseLen, static_cast<int64_t>(ExecutionPhase::kDiffusionDenoise));
    EXPECT_EQ(cfg.diffusionCommitDims(2, 8).executionPhaseLen, static_cast<int64_t>(ExecutionPhase::kDiffusionCommit));
}

TEST(LLMEngineConfigRecipesTest, SpecAndDiffusionRecipesUsePhysicalTokenExtent)
{
    LLMEngineConfig cfg = makeRecipeConfig(/*maxKV=*/4096, /*mrope=*/false);
    EXPECT_EQ(cfg.proposalDims(/*batch=*/3, /*paddedTreeSize=*/8, /*draftTopK=*/4).seqLen, 24);
    EXPECT_EQ(cfg.proposalDims(/*batch=*/3, /*paddedTreeSize=*/8, /*draftTopK=*/4).contextSequenceCount, 0);
    EXPECT_EQ(cfg.specVerifyDims(/*batch=*/3, /*verifySize=*/8).seqLen, 24);
    EXPECT_EQ(cfg.specVerifyDims(/*batch=*/3, /*verifySize=*/8).contextSequenceCount, 0);
    EXPECT_EQ(cfg.denoiseDims(/*batch=*/3, /*canvasLen=*/8).seqLen, 24);
    EXPECT_EQ(cfg.denoiseDims(/*batch=*/3, /*canvasLen=*/8).contextSequenceCount, 0);
    EXPECT_EQ(cfg.diffusionCommitDims(/*batch=*/3, /*commitLen=*/8).seqLen, 24);
    EXPECT_EQ(cfg.diffusionCommitDims(/*batch=*/3, /*commitLen=*/8).contextSequenceCount, 0);
}

TEST(LLMEngineConfigRecipesTest, SpecVerifySelectsEveryPhysicalRow)
{
    auto const cfg = makeRecipeConfig(/*maxKV=*/8192, /*mrope=*/false);
    auto const d = cfg.specVerifyDims(/*batch=*/1, /*verifySize=*/8);
    EXPECT_EQ(d.batch, 1);
    EXPECT_EQ(d.seqLen, 8);
    EXPECT_EQ(d.kvLen, 8192);
    EXPECT_EQ(d.selectLen, 8);
    EXPECT_EQ(d.attnMaskSeqLen, 8); // verifySize — proposal attention shape
    EXPECT_EQ(d.ropeBatch, 1);
    EXPECT_EQ(d.packedMaskLen, 1); // divUp(8, 32) = 1
    EXPECT_EQ(d.startIndexLen, 1); // batch (cache non-empty during verify)
    EXPECT_EQ(d.executionPhaseLen, static_cast<int64_t>(ExecutionPhase::kSpecTargetVerify));
}

TEST(LLMEngineConfigRecipesTest, ShortPrefillUsesContextPhase)
{
    auto const cfg = makeRecipeConfig(/*maxKV=*/8192, /*mrope=*/false);
    for (int32_t seqLen = 2; seqLen <= 16; ++seqLen)
    {
        auto const d = cfg.prefillDims(/*batch=*/2, seqLen, ExecutionPhase::kContextPrefill);
        EXPECT_EQ(d.seqLen, 2 * seqLen);
        EXPECT_EQ(d.executionPhaseLen, static_cast<int64_t>(ExecutionPhase::kContextPrefill));
    }
    auto const verifyDims = cfg.specVerifyDims(/*batch=*/2, /*verifySize=*/16);
    EXPECT_EQ(verifyDims.executionPhaseLen, static_cast<int64_t>(ExecutionPhase::kSpecTargetVerify));
}

TEST(LLMEngineConfigRecipesTest, ProposalDims)
{
    auto const cfg = makeRecipeConfig(/*maxKV=*/4096, /*mrope=*/false);
    auto const d = cfg.proposalDims(/*batch=*/2, /*paddedTreeSize=*/16, /*draftTopK=*/4);
    EXPECT_EQ(d.seqLen, 32);
    EXPECT_EQ(d.selectLen, 8);       // batch * draftTopK token-major selected rows
    EXPECT_EQ(d.attnMaskSeqLen, 32); // flattened physical proposal rows
    EXPECT_EQ(d.packedMaskLen, 1);   // divUp(16, 32) = 1
    EXPECT_EQ(d.startIndexLen, 2);   // batch
    EXPECT_EQ(d.executionPhaseLen, static_cast<int64_t>(ExecutionPhase::kSpecDraftProposal));
}

TEST(LLMEngineConfigRecipesTest, AcceptDims)
{
    auto const cfg = makeRecipeConfig(/*maxKV=*/4096, /*mrope=*/false);
    auto const d = cfg.acceptDims(/*batch=*/3, /*acceptLen=*/33);
    EXPECT_EQ(d.batch, 3);
    EXPECT_EQ(d.seqLen, 99);
    EXPECT_EQ(d.selectLen, 3);
    EXPECT_EQ(d.attnMaskSeqLen, 99); // flattened physical accept rows
    EXPECT_EQ(d.packedMaskLen, 2);   // divUp(33, 32) = 2 — exercises the boundary
    EXPECT_EQ(d.startIndexLen, 3);   // batch
    EXPECT_EQ(d.executionPhaseLen, static_cast<int64_t>(ExecutionPhase::kSpecDraftProposal));
}

TEST(LLMEngineConfigRecipesTest, ResetDimsRopeBatchOneEvenForMRope)
{
    // The resetDims recipe hardcodes ropeBatch=1 even for MRope models, matching
    // pre-migration behavior in both runtimes (reset is a placeholder binding,
    // not a real inference step). Exercise both branches to lock the behavior.
    auto const cfgNonMRope = makeRecipeConfig(/*maxKV=*/4096, /*mrope=*/false);
    auto const cfgMRope = makeRecipeConfig(/*maxKV=*/4096, /*mrope=*/true);

    for (auto const& cfg : {cfgNonMRope, cfgMRope})
    {
        auto const d = cfg.resetDims();
        EXPECT_EQ(d.batch, 1);
        EXPECT_EQ(d.seqLen, 1);
        EXPECT_EQ(d.kvLen, 4096);
        EXPECT_EQ(d.selectLen, 1);
        EXPECT_EQ(d.attnMaskSeqLen, 1);
        EXPECT_EQ(d.ropeBatch, 1); // Always 1, regardless of MRope
        EXPECT_EQ(d.packedMaskLen, 1);
        EXPECT_EQ(d.startIndexLen, 1); // placeholder bind; matches batch=1
        EXPECT_EQ(d.executionPhaseLen, static_cast<int64_t>(ExecutionPhase::kAutoregressiveDecode));
    }
}

// ===========================================================================
// parseDraftEngineConfig
//
// The draft parser shares parseCoreFields with the base parser but applies
// its own rules: numAttentionLayers = numDecoderLayers, vocab from
// `draft_vocab_size`, and `partial_rotary_factor` (Qwen3.5 MTP draft inherits
// the base's rotary fraction, e.g. headDim=256 with factor=0.25 → rotaryDim=64).
// ===========================================================================

namespace
{

//! Minimal valid JSON for a draft engine config.
Json makeMinimalDraftConfig()
{
    Json config = makeMinimalConfig();
    config["spec_decode_type"] = "mtp";
    config["engine_role"] = "draft";
    config["draft_vocab_size"] = 32000;
    config["base_model_hidden_size"] = 768;
    config["builder_config"]["spec_base"] = false;
    config["builder_config"]["spec_draft"] = true;
    config["builder_config"]["max_draft_tree_size"] = 4;
    return config;
}

} // namespace

TEST_F(LLMEngineConfigTest, ParseDraftEngineConfigMinimal)
{
    Json const json = makeMinimalDraftConfig();
    auto const path = writeJsonToTempFile(json);

    LLMEngineConfig cfg = parseDraftEngineConfig(path);

    EXPECT_EQ(cfg.numDecoderLayers, 12);
    EXPECT_EQ(cfg.numAttentionLayers, 12); // = numDecoderLayers (draft is pure-attention)
    EXPECT_EQ(cfg.headDim, 64);
    // No partial_rotary_factor → rotaryDim defaults to headDim.
    EXPECT_EQ(cfg.rotaryDim, 64);
    EXPECT_EQ(cfg.vocabSize, 32000);
    // Draft engines set isSpecDecodeBase=false (they ARE the draft, not the base).
    // Presence of a draft engine is indicated by maxDraftTreeSize > 0.
    EXPECT_FALSE(cfg.isSpecDecodeBase);
    EXPECT_EQ(cfg.maxDraftTreeSize, 4);
    EXPECT_EQ(cfg.baseModelHiddenSize, 768);
}

TEST_F(LLMEngineConfigTest, DraftRaggedQueryCapacityCoversProposalWidth)
{
    Json json = makeMinimalDraftConfig();
    json["builder_config"]["max_draft_tree_size"] = 129;

    LLMEngineConfig const treeDominates = parseDraftEngineConfig(writeJsonToTempFile(json));
    EXPECT_EQ(treeDominates.maxQueryLength, 129);
    EXPECT_EQ(treeDominates.maxPhysicalTokens, 258);

    json["builder_config"]["max_draft_tree_size"] = 140;
    EXPECT_EQ(parseDraftEngineConfig(writeJsonToTempFile(json)).maxQueryLength, 140);
}

TEST_F(LLMEngineConfigTest, DSparkDraftCapacityIncludesNonAnchorInputRow)
{
    Json json = makeMinimalDraftConfig();
    json["spec_decode_type"] = "dspark";
    json["builder_config"]["max_input_len"] = 8;
    json["builder_config"]["max_draft_tree_size"] = 16;
    json["dspark_config"]["sample_from_anchor"] = false;
    json["dspark_config"]["target_layer_ids"] = {0};

    LLMEngineConfig const cfg = parseDraftEngineConfig(writeJsonToTempFile(json));
    EXPECT_FALSE(cfg.dsparkSampleFromAnchor);
    EXPECT_EQ(cfg.maxQueryLength, 17);
    EXPECT_EQ(cfg.maxPhysicalTokens, 34);
}

TEST_F(LLMEngineConfigTest, RejectsDSparkNonAnchorDraftCapacityOverflow)
{
    Json json = makeMinimalDraftConfig();
    json["spec_decode_type"] = "dspark";
    json["builder_config"]["max_batch_size"] = 1;
    json["builder_config"]["max_input_len"] = 1;
    json["builder_config"]["max_draft_tree_size"] = std::numeric_limits<int32_t>::max();
    json["dspark_config"]["sample_from_anchor"] = false;
    json["dspark_config"]["target_layer_ids"] = {0};

    EXPECT_THROW(parseDraftEngineConfig(writeJsonToTempFile(json)), std::runtime_error);
}

TEST_F(LLMEngineConfigTest, ParseDraftEngineConfigMTPBaseModelHiddenSize)
{
    // Regression: createDeploymentConfig used to hardcode
    // `specConfig.baseOutputHiddenDim = base.hiddenSize * 3` (EAGLE-3 convention),
    // which broke MTP — MTP draft expects `base.hiddenSize` (no `* 3`). The
    // value must come from the draft config's `base_model_hidden_size` field.
    Json json = makeMinimalDraftConfig();
    json["base_model_hidden_size"] = 1024; // MTP-style: not multiplied by 3
    auto const path = writeJsonToTempFile(json);

    LLMEngineConfig cfg = parseDraftEngineConfig(path);
    EXPECT_EQ(cfg.baseModelHiddenSize, 1024);
}

TEST_F(LLMEngineConfigTest, ParseDraftEngineConfigPartialRotaryFactor)
{
    // Regression: parseDraftEngineConfig used to hardcode `rotaryDim = headDim`,
    // ignoring `partial_rotary_factor`. For Qwen3.5 MTP that produced
    // rotaryDim=256 instead of 64 → setInputShape mismatch on the draft
    // engine's `rope_rotary_cos_sin` binding ([1, kv, 256] vs expected
    // [1, kv, 64]) and a TRT API usage error during draft prefill.
    Json json = makeMinimalDraftConfig();
    json["head_dim"] = 256;
    json["partial_rotary_factor"] = 0.25;
    auto const path = writeJsonToTempFile(json);

    LLMEngineConfig cfg = parseDraftEngineConfig(path);
    EXPECT_EQ(cfg.headDim, 256);
    EXPECT_EQ(cfg.rotaryDim, 64); // 256 * 0.25
}

TEST_F(LLMEngineConfigTest, ParseDraftEngineConfigDoesNotRequireEagleTargetLayerIds)
{
    Json json = makeMinimalDraftConfig();
    json["spec_decode_type"] = "eagle3";
    json["base_model_hidden_size"] = 2304;
    json["eagle3_config"] = {{"target_layer_ids", Json::array()}, {"num_target_layers", 3}};
    auto const path = writeJsonToTempFile(json);

    auto const config = parseDraftEngineConfig(path);
    EXPECT_TRUE(config.specTargetLayerIds.empty());
}
