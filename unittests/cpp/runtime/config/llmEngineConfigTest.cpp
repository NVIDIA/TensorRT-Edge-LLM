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
#include "action/alpamayo1ActionRunner.h"

#include "common/pagedKvTypes.h"
#include "common/specDecodeConfigUtils.h"
#include "substituteEngine.h"
#include "testUtils.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <limits>
#include <set>
#include <string>

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

//! Geometry of the n-gram pool the tests below share, named so the assertions can refer to it.
constexpr int64_t kNgramRowsPerShard{4176};
constexpr int64_t kNgramRowWidth{160};
constexpr int64_t kNgramHeads{16};

//! Extend the minimal config with Qwen4-Exp's n-gram injection: the six geometry keys, its
//! conv-state dtype, and the sharded FP8 row pool it gathers from. Deliberately not hybrid —
//! the injection is independent of recurrent layers.
Json makeNgramConfig()
{
    Json json = makeMinimalConfig();
    json["ngram_layers"] = Json::array({1});
    json["ngram_embed_dim"] = kNgramRowWidth * kNgramHeads;
    json["ngram_conv_kernel"] = 4;
    json["ngram_conv_dim"] = 10240;
    json["ngram_size"] = 3;
    json["ngram_heads_per_order"] = 8;
    json["ngram_conv_state_dtype"] = "fp16";
    Json shards = Json::array();
    for (int32_t index = 0; index < 4; ++index)
    {
        shards.push_back(
            Json{{"tensor_key", "ngram.shard_" + std::to_string(index) + ".weight"}, {"rows", kNgramRowsPerShard},
                {"bytes", kNgramRowsPerShard * kNgramRowWidth}, {"samples", {{{"offset", 0}, {"data", "0102"}}}}});
    }
    json["ple_pool"] = Json{{"row_count", 4 * kNgramRowsPerShard}, {"row_width", kNgramRowWidth}, {"dtype", "F8_E4M3"},
        {"scale", 0.00019931793212890625}, {"heads", kNgramHeads},
        {"hash_constants",
            {{"layer_multipliers", "ngram.layer_multipliers"}, {"heads_vocab_sizes", "ngram.vocab_sizes"},
                {"heads_offsets", "ngram.offsets"}}},
        {"source",
            {{"dtype", "F8_E4M3"}, {"bytes", 4 * kNgramRowsPerShard * kNgramRowWidth}, {"scale", 1.0},
                {"shards", shards}}}};
    return json;
}

void addPlePool(Json& json, int32_t hiddenSize = 64)
{
    int64_t const rowCount = json["vocab_size"].get<int64_t>();
    int64_t const numLayers = json["num_hidden_layers"].get<int64_t>();
    int64_t const rowWidth = numLayers * hiddenSize;
    int64_t const bytes = rowCount * rowWidth * 2;
    json["gemma_ple_enabled"] = true;
    json["gemma_ple_layer_count"] = numLayers;
    json["gemma_ple_hidden_size"] = hiddenSize;
    json["ple_pool"] = Json{{"row_count", rowCount}, {"row_width", rowWidth}, {"dtype", "F16"}, {"scale", 1.0},
        {"source",
            {{"tensor_key", "model.embed_tokens_per_layer.weight"}, {"dtype", "F16"}, {"bytes", bytes}, {"scale", 1.0},
                {"samples", {{{"offset", 0}, {"data", "003c"}}}}}}};
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
    EXPECT_EQ(cfg.maxSpecVerifyTokens, 0);
    EXPECT_EQ(cfg.maxSpecDraftTokens, 0);
    EXPECT_EQ(cfg.kvCacheDtype, trt_edgellm::DataType::kHALF);
    EXPECT_EQ(cfg.raggedBackend, RaggedBackendKind::kEntryPaddedCompatibility);
    EXPECT_EQ(cfg.maxNumSequences, 2);
    EXPECT_EQ(cfg.maxQueryLength, 128);
    EXPECT_EQ(cfg.maxPhysicalTokens, 256);
    EXPECT_EQ(cfg.recurrentPoolRows, 2);
}

// Given a TP 2 config.json with indexer_head_dim 128 whose rank 1 overrides num_attention_heads 16 with its share of
//       8, and a config.json without num_attention_heads, indexer_head_dim or builder_config.tp_size
// When the first is parsed for rank 1 and the second for a single device
// Then the compact capability check's attention inputs read 8 query heads at TP 2 with a 128-wide QSA indexer, and
//      default to 0 (unknown) query heads at TP 1 without QSA
TEST_F(LLMEngineConfigTest, ParsesTheAttentionShapeOfTheCompactCapabilityCheck)
{
    Json json = makeMinimalConfig();
    json["num_attention_heads"] = 16;
    json["indexer_head_dim"] = 128;
    json["builder_config"]["tp_size"] = 2;
    json["rank_configs"] = Json::array({
        Json{{"rank", 0}, {"config_overrides", Json{{"num_attention_heads", 8}}}},
        Json{{"rank", 1}, {"config_overrides", Json{{"num_attention_heads", 8}}}},
    });
    LLMEngineConfig const rank = parseEngineConfig(writeJsonToTempFile(json), /*rank=*/1, /*expectedWorldSize=*/2);
    EXPECT_EQ(rank.numAttentionHeads, 8);
    EXPECT_EQ(rank.tpSize, 2);
    EXPECT_EQ(rank.indexerHeadDim, 128);

    LLMEngineConfig const legacy = parseEngineConfig(writeJsonToTempFile(makeMinimalConfig()));
    EXPECT_EQ(legacy.numAttentionHeads, 0);
    EXPECT_EQ(legacy.tpSize, 1);
    EXPECT_EQ(legacy.indexerHeadDim, 0);
}

// Given a config.json whose builder_config records compact_execution: true, one without the field, and ones whose
//       field is false, 1, "true" or an object
// When each is parsed
// Then only the first is built for compact execution, and every one loads
TEST_F(LLMEngineConfigTest, ParsesCompactExecution)
{
    Json json = makeMinimalConfig();
    json["builder_config"]["compact_execution"] = true;
    EXPECT_TRUE(parseEngineConfig(writeJsonToTempFile(json)).compactExecution);

    EXPECT_FALSE(parseEngineConfig(writeJsonToTempFile(makeMinimalConfig())).compactExecution);
    for (Json const& value : {Json(false), Json(1), Json("true"), Json{{"version", 1}}})
    {
        json["builder_config"]["compact_execution"] = value;
        EXPECT_FALSE(parseEngineConfig(writeJsonToTempFile(json)).compactExecution) << value.dump();
    }
}

TEST_F(LLMEngineConfigTest, ParsesPerLayerKvCacheScales)
{
    // Given an FP8 runtime config with one K/V dequant scale pair per layer.
    Json json = makeMinimalConfig();
    json["num_hidden_layers"] = 2;
    json["kv_cache_dtype"] = "fp8";
    json["kv_cache_scales"] = {{0.25F, 0.5F}, {1.25F, 1.5F}};

    // When the runtime contract is parsed.
    LLMEngineConfig const cfg = parseEngineConfig(writeJsonToTempFile(json));

    // Then the scale ordering and values are preserved for the action bridge.
    ASSERT_EQ(cfg.kvCacheScales.size(), 2);
    EXPECT_FLOAT_EQ(cfg.kvCacheScales[0][0], 0.25F);
    EXPECT_FLOAT_EQ(cfg.kvCacheScales[0][1], 0.5F);
    EXPECT_FLOAT_EQ(cfg.kvCacheScales[1][0], 1.25F);
    EXPECT_FLOAT_EQ(cfg.kvCacheScales[1][1], 1.5F);
}

TEST_F(LLMEngineConfigTest, RejectsInvalidKvCacheScales)
{
    // Given otherwise-valid configs whose external KV scale contract is malformed.
    std::array<Json, 3> const invalidScales{
        Json{{0.25F, 0.5F}}, Json{{0.25F}, {1.25F, 1.5F}}, Json{{0.25F, 0.5F}, {0.0F, 1.5F}}};

    for (Json const& scales : invalidScales)
    {
        Json json = makeMinimalConfig();
        json["num_hidden_layers"] = 2;
        json["kv_cache_dtype"] = "fp8";
        json["kv_cache_scales"] = scales;

        // When the runtime contract is parsed, then invalid length, shape, and values are rejected.
        EXPECT_THROW(parseEngineConfig(writeJsonToTempFile(json)), std::runtime_error);
    }
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
    json["builder_config"]["max_spec_verify_tokens"] = 129;

    LLMEngineConfig const treeDominates = parseEngineConfig(writeJsonToTempFile(json));
    EXPECT_EQ(treeDominates.maxQueryLength, 129);
    EXPECT_EQ(treeDominates.maxPhysicalTokens, 258);

    json["builder_config"]["max_input_len"] = 160;
    json["builder_config"]["max_spec_verify_tokens"] = 140;
    EXPECT_EQ(parseEngineConfig(writeJsonToTempFile(json)).maxQueryLength, 160);
}

TEST_F(LLMEngineConfigTest, ParsePlePoolMetadata)
{
    Json json = makeMinimalConfig();
    json["gemma_ple_enabled"] = true;
    json["gemma_ple_layer_count"] = 12;
    json["gemma_ple_hidden_size"] = 256;
    json["ple_pool"] = Json{{"row_count", 32000}, {"row_width", 3072}, {"dtype", "F16"}, {"scale", 1.0},
        {"source",
            {{"tensor_key", "model.embed_tokens_per_layer.weight"}, {"dtype", "BF16"}, {"bytes", 32000LL * 3072 * 2},
                {"scale", 16.0}, {"samples", {{{"offset", 0}, {"data", "003c"}}}}}}};
    auto const cfg = parseEngineConfig(writeJsonToTempFile(json));
    EXPECT_TRUE(cfg.gemmaPleEnabled);
    EXPECT_EQ(cfg.plePool.rowCount, 32000);
    EXPECT_EQ(cfg.plePool.rowWidth, 3072);
    EXPECT_EQ(cfg.plePool.dtype, PleTable::DType::kFP16);
    EXPECT_EQ(cfg.plePool.scale, 1.0);

    for (auto const& patch : {Json{{"dtype", "BF16"}}, Json{{"scale", 0.5}}, Json{{"row_width", 3071}},
             Json{{"row_count", 0}}, Json{{"row_count", std::numeric_limits<int32_t>::max()}}})
    {
        SCOPED_TRACE(patch.dump());
        Json invalid = json;
        invalid["ple_pool"].update(patch);
        EXPECT_THROW(parseEngineConfig(writeJsonToTempFile(invalid)), std::exception);
    }
    json["gemma_ple_enabled"] = false;
    EXPECT_THROW(parseEngineConfig(writeJsonToTempFile(json)), std::exception);
}

TEST_F(LLMEngineConfigTest, ParseNgramShardedFp8PlePool)
{
    // Given an n-gram PLE engine whose table is FP8, split across ordered shards and gathered
    // 16 rows per token,
    // When the config is parsed,
    // Then the pool must be accepted with its dequantization scale and head count intact -- the
    // Gemma-only "F16 with unit scale, one row per layer" contract must not apply to it.
    Json const json = makeNgramConfig();

    auto const cfg = parseEngineConfig(writeJsonToTempFile(json));
    EXPECT_FALSE(cfg.gemmaPleEnabled);
    EXPECT_TRUE(cfg.ngramEnabled());
    EXPECT_EQ(cfg.plePool.heads, kNgramHeads);
    EXPECT_EQ(cfg.plePool.dtype, PleTable::DType::kFP8);
    EXPECT_DOUBLE_EQ(cfg.plePool.scale, 0.00019931793212890625);
    EXPECT_EQ(cfg.plePool.rowCount, 4 * kNgramRowsPerShard);
    ASSERT_EQ(cfg.plePool.source.shards.size(), 4U);
    EXPECT_TRUE(cfg.plePool.source.tensorKey.empty());
    EXPECT_EQ(cfg.plePool.source.shards[2].tensorKey, "ngram.shard_2.weight");
    EXPECT_EQ(cfg.plePool.source.shards[0].rows, kNgramRowsPerShard);
    // The values stay in the checkpoint; only their names reach the engine.
    EXPECT_EQ(cfg.plePool.layerMultipliersKey, "ngram.layer_multipliers");
    EXPECT_EQ(cfg.plePool.headVocabSizesKey, "ngram.vocab_sizes");
    EXPECT_EQ(cfg.plePool.headOffsetsKey, "ngram.offsets");

    // Shards that do not cover the pool, a head count that does not fill the embedding, and a
    // missing hash constant each have to be rejected rather than silently truncating the table.
    for (auto const& patch :
        {Json{{"row_count", 4 * kNgramRowsPerShard + 1}}, Json{{"heads", 8}}, Json{{"row_width", kNgramRowWidth + 1}}})
    {
        SCOPED_TRACE(patch.dump());
        Json invalid = json;
        invalid["ple_pool"].update(patch);
        EXPECT_THROW(parseEngineConfig(writeJsonToTempFile(invalid)), std::exception);
    }
    Json missingConstants = json;
    missingConstants["ple_pool"]["hash_constants"].erase("heads_offsets");
    EXPECT_THROW(parseEngineConfig(writeJsonToTempFile(missingConstants)), std::exception);
}

TEST_F(LLMEngineConfigTest, NgramConvStateDtypeDoesNotFollowHybridConvDtype)
{
    // Given a model that is both hybrid and n-gram injected, whose two conv states were exported
    // at different precisions,
    // When the config is parsed,
    // Then each state must keep its own dtype rather than the injection inheriting the mixer's.
    Json json = makeNgramConfig();
    Json const hybrid = makeHybridConfig();
    for (char const* key :
        {"num_linear_attn_layers", "num_attention_layers", "recurrent_state_num_heads", "recurrent_state_head_dim",
            "recurrent_state_size", "conv_dim", "conv_kernel", "recurrent_state_dtype", "conv_state_dtype"})
    {
        json[key] = hybrid.at(key);
    }
    // Neither value may be the FP16 default, or an implementation that ignored one key entirely
    // would still satisfy the assertions below.
    json["conv_state_dtype"] = "fp32";
    json["ngram_conv_state_dtype"] = "bf16";

    auto const cfg = parseEngineConfig(writeJsonToTempFile(json));
    EXPECT_EQ(cfg.convStateDtype, trt_edgellm::DataType::kFLOAT);
    EXPECT_EQ(cfg.ngramConvStateDtype, trt_edgellm::DataType::kBF16);
}

TEST_F(LLMEngineConfigTest, NgramConvStateDtypeMustBePresentAndKnownWhenEnabled)
{
    // Given an n-gram injected config whose conv-state dtype is either omitted or names a dtype the
    // runtime has no mapping for,
    // When the config is parsed,
    // Then both cases must be rejected: the dtype fixes a binding the runtime allocates, so neither
    // defaulting a missing key nor falling back from an unknown one may produce a state the engine
    // cannot accept.
    Json const base = makeNgramConfig();
    Json omitted = base;
    omitted.erase("ngram_conv_state_dtype");
    Json unknown = base;
    unknown["ngram_conv_state_dtype"] = "garbage";
    for (Json const& invalid : {omitted, unknown})
    {
        SCOPED_TRACE(invalid.value("ngram_conv_state_dtype", "<absent>"));
        EXPECT_THROW(parseEngineConfig(writeJsonToTempFile(invalid)), std::runtime_error);
    }
}

TEST_F(LLMEngineConfigTest, NgramConvStateDtypeAbsentWithoutNgramLayers)
{
    // Given a config with no n-gram injection,
    // When the config is parsed,
    // Then the dtype key must not be required -- a model without the feature binds no such state.
    Json json = makeMinimalConfig();

    auto const cfg = parseEngineConfig(writeJsonToTempFile(json));
    EXPECT_FALSE(cfg.ngramEnabled());
    EXPECT_EQ(cfg.ngramConvStateDtype, trt_edgellm::DataType::kHALF);
}

TEST_F(LLMEngineConfigTest, ParsePleCheckpointSource)
{
    Json json = makeMinimalConfig();
    json["gemma_ple_enabled"] = true;
    json["gemma_ple_layer_count"] = 12;
    json["gemma_ple_hidden_size"] = 256;
    json["checkpoint_dir"] = "/models/gemma";
    for (std::string const dtype : {"F16", "BF16", "F32"})
    {
        int64_t const bytes = 32000LL * 3072 * (dtype == "F32" ? 4 : 2);
        json["ple_pool"] = Json{{"row_count", 32000}, {"row_width", 3072}, {"dtype", "F16"}, {"scale", 1.0},
            {"source",
                {{"tensor_key", "model.embed_tokens_per_layer.weight"}, {"dtype", dtype}, {"bytes", bytes},
                    {"scale", 16.0}, {"samples", {{{"offset", 0}, {"data", "003c"}}}}}}};
        auto const cfg = parseEngineConfig(writeJsonToTempFile(json));
        EXPECT_EQ(cfg.plePool.checkpointDir, "/models/gemma");
        EXPECT_EQ(cfg.plePool.source.tensorKey, "model.embed_tokens_per_layer.weight");
        EXPECT_EQ(cfg.plePool.source.dtype,
            dtype == "F16"        ? PleTable::DType::kFP16
                : dtype == "BF16" ? PleTable::DType::kBF16
                                  : PleTable::DType::kFP32);
        EXPECT_EQ(cfg.plePool.source.bytes, bytes);
        EXPECT_EQ(cfg.plePool.source.scale, 16.0);
        ASSERT_EQ(cfg.plePool.source.samples.size(), 1U);
        EXPECT_EQ(cfg.plePool.source.samples[0].offset, 0);
        EXPECT_EQ(cfg.plePool.source.samples[0].data, "003c");
        EXPECT_TRUE(cfg.plePool.source.file.empty());
        json["ple_pool"]["source"]["location"] = Json{{"file", "pytorch_model.bin"}, {"offset", 128}, {"bytes", bytes}};
        auto const indexed = parseEngineConfig(writeJsonToTempFile(json));
        EXPECT_EQ(indexed.plePool.source.file, "pytorch_model.bin");
        EXPECT_EQ(indexed.plePool.source.offset, 128);
    }
}

TEST_F(LLMEngineConfigTest, RejectsInvalidPleSourceContract)
{
    Json json = makeMinimalConfig();
    json["gemma_ple_enabled"] = true;
    json["gemma_ple_layer_count"] = 12;
    json["gemma_ple_hidden_size"] = 256;
    int64_t const bytes = 32000LL * 3072 * 2;
    json["ple_pool"] = Json{{"row_count", 32000}, {"row_width", 3072}, {"dtype", "F16"}, {"scale", 1.0},
        {"source",
            {{"tensor_key", "model.embed_tokens_per_layer.weight"}, {"dtype", "BF16"}, {"bytes", bytes},
                {"scale", 16.0}, {"samples", {{{"offset", 0}, {"data", "003c"}}}}}}};
    for (auto const& patch : {Json{{"tensor_key", ""}}, Json{{"dtype", "F8_E4M3"}}, Json{{"bytes", bytes - 1}},
             Json{{"bytes", static_cast<double>(bytes)}}, Json{{"scale", 0.0}}, Json{{"scale", -1.0}},
             Json{{"samples", Json::array()}}, Json{{"samples", {{{"offset", -1}, {"data", "00"}}}}},
             Json{{"samples", {{{"offset", bytes}, {"data", "00"}}}}},
             Json{{"samples", {{{"offset", 0.5}, {"data", "00"}}}}},
             Json{{"samples", {{{"offset", 0}, {"data", "0"}}}}}, Json{{"samples", {{{"offset", 0}, {"data", "zz"}}}}},
             Json{{"location", {{"file", "../model.bin"}, {"offset", 0}, {"bytes", bytes}}}},
             Json{{"location", {{"file", "/model.bin"}, {"offset", 0}, {"bytes", bytes}}}},
             Json{{"location", {{"file", "model.bin"}, {"offset", -1}, {"bytes", bytes}}}},
             Json{{"location", {{"file", "model.bin"}, {"offset", 0}, {"bytes", bytes - 1}}}}})
    {
        SCOPED_TRACE(patch.dump());
        Json invalid = json;
        invalid["ple_pool"]["source"].update(patch);
        EXPECT_THROW(parseEngineConfig(writeJsonToTempFile(invalid)), std::exception);
    }
    json["ple_pool"]["row_count"] = 32000.5;
    EXPECT_THROW(parseEngineConfig(writeJsonToTempFile(json)), std::exception);
}

TEST_F(LLMEngineConfigTest, RejectsPleWithoutCheckpointSource)
{
    Json json = makeMinimalConfig();
    json["gemma_ple_enabled"] = true;
    json["gemma_ple_layer_count"] = 12;
    json["gemma_ple_hidden_size"] = 256;
    EXPECT_THROW(parseEngineConfig(writeJsonToTempFile(json)), std::runtime_error);
    json["ple_pool"] = Json{{"artifact", "ple/manifest.json"}, {"table_id", "test-table"}, {"row_count", 32000},
        {"row_width", 3072}, {"dtype", "F16"}, {"scale", 1.0}};
    try
    {
        parseEngineConfig(writeJsonToTempFile(json));
        FAIL() << "PLE metadata without a checkpoint source must be rejected";
    }
    catch (std::runtime_error const& error)
    {
        EXPECT_NE(std::string(error.what()).find("re-export and rebuild"), std::string::npos);
    }
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
        EXPECT_EQ(
            config.specVerifyDims(/*batch=*/2, /*verifyTokenCount=*/4, SpecVerifyLayout::kChain).swaKVCacheModeLen,
            expected);
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
    json["builder_config"]["max_spec_verify_tokens"] = 8;
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
    json["builder_config"]["max_spec_verify_tokens"] = 8;
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
        json["builder_config"]["max_spec_verify_tokens"] = 8;
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
    json["builder_config"]["max_spec_verify_tokens"] = 8;
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
    EXPECT_EQ(cfg.recurrentStateDtype, trt_edgellm::DataType::kHALF);
    EXPECT_EQ(cfg.convStateDtype, trt_edgellm::DataType::kHALF);
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
    EXPECT_EQ(cfg.padTokenId, -1);
    EXPECT_EQ(cfg.imageTokenId, -1);
    EXPECT_EQ(cfg.audioTokenId, -1);
    EXPECT_NE(cfg.ropeConfig.type, RopeType::kMRope);
    // numAttentionLayers defaults to numDecoderLayers when not specified.
    EXPECT_EQ(cfg.numAttentionLayers, cfg.numDecoderLayers);
}

// Given an engine config carrying a prefill and a decode skip-softmax table, When the runtime derives its dims for
// every step recipe, Then each recipe binds the prefill table's length on `skip_softmax_prefill_scale` and the decode
// table's length on `skip_softmax_decode_scale`, both flattened in bin order, and only a prefill below every
// skippable length drops the prefill binding to [0].
TEST_F(LLMEngineConfigTest, SkipSoftmaxTablesDriveBothBindingLengthsInEveryRecipe)
{
    Json json = makeMinimalConfig();
    json["skip_softmax_prefill_scale_table"] = Json::array({Json::array({8192, 6000}), Json::array({4096, 3000})});
    json["skip_softmax_decode_scale_table"]
        = Json::array({Json::array({4096, 1228}), Json::array({2048, 614}), Json::array({8192, 2457})});
    auto const path = writeJsonToTempFile(json);

    LLMEngineConfig cfg = parseEngineConfig(path);

    EXPECT_EQ(cfg.flattenedSkipSoftmaxPrefillScaleTable(), (std::vector<int32_t>{4096, 3000, 8192, 6000}));
    EXPECT_EQ(cfg.flattenedSkipSoftmaxDecodeScaleTable(), (std::vector<int32_t>{2048, 614, 4096, 1228, 8192, 2457}));
    for (InferenceDims const& dims : {cfg.prefillDims(1, 8192, ExecutionPhase::kContextPrefill),
             cfg.prefillDims(1, 16, ExecutionPhase::kContextChunk), cfg.decodeDims(1), cfg.denoiseDims(1, 4),
             cfg.diffusionCommitDims(1, 4), cfg.specVerifyDims(1, 4, SpecVerifyLayout::kChain),
             cfg.proposalDims(1, 4, 1), cfg.acceptDims(1, 4), cfg.resetDims(), cfg.compactDims(2, 9, 1, 2),
             cfg.compactDims(2, 2, 0, 2)})
    {
        EXPECT_EQ(dims.skipSoftmaxPrefillScaleLen, 4) << toString(dims);
        EXPECT_EQ(dims.skipSoftmaxDecodeScaleLen, 6) << toString(dims);
    }
    InferenceDims const shortPrefill = cfg.prefillDims(1, 16, ExecutionPhase::kContextPrefill);
    EXPECT_EQ(shortPrefill.skipSoftmaxPrefillScaleLen, 0);
    EXPECT_EQ(shortPrefill.skipSoftmaxDecodeScaleLen, 6);
}

// Given only a prefill table, When dims are derived, Then `skip_softmax_decode_scale` binds at [0] (dense decode)
// while `skip_softmax_prefill_scale` keeps the prefill table on every step.
TEST_F(LLMEngineConfigTest, PrefillOnlySkipSoftmaxTableLeavesDecodeDense)
{
    Json json = makeMinimalConfig();
    json["skip_softmax_prefill_scale_table"] = Json::array({Json::array({0, 6000})});
    auto const path = writeJsonToTempFile(json);

    LLMEngineConfig cfg = parseEngineConfig(path);

    EXPECT_EQ(cfg.decodeDims(1).skipSoftmaxDecodeScaleLen, 0);
    EXPECT_EQ(cfg.decodeDims(1).skipSoftmaxPrefillScaleLen, 2);
    EXPECT_EQ(cfg.prefillDims(1, 8192, ExecutionPhase::kContextPrefill).skipSoftmaxPrefillScaleLen, 2);
    EXPECT_EQ(cfg.prefillDims(1, 8192, ExecutionPhase::kContextPrefill).skipSoftmaxDecodeScaleLen, 0);
}

// Given a config carrying a table under the unphased key `skip_softmax_scale_table`, When it is parsed, Then that
// key is not read: alone the prefill stays dense, next to `skip_softmax_prefill_scale_table` only the latter deploys.
TEST_F(LLMEngineConfigTest, UnphasedSkipSoftmaxScaleTableKeyIsNotRead)
{
    Json json = makeMinimalConfig();
    json["skip_softmax_scale_table"] = Json::array({Json::array({0, 6000})});

    LLMEngineConfig const alone = parseEngineConfig(writeJsonToTempFile(json));
    EXPECT_TRUE(alone.flattenedSkipSoftmaxPrefillScaleTable().empty());
    EXPECT_EQ(alone.prefillDims(1, 8192, ExecutionPhase::kContextPrefill).skipSoftmaxPrefillScaleLen, 0);

    json["skip_softmax_prefill_scale_table"] = Json::array({Json::array({0, 7000})});
    LLMEngineConfig const both = parseEngineConfig(writeJsonToTempFile(json));
    EXPECT_EQ(both.flattenedSkipSoftmaxPrefillScaleTable(), (std::vector<int32_t>{0, 7000}));
}

TEST_F(LLMEngineConfigTest, SpecDecodeMaxProposalSizes)
{
    Json json = makeMinimalConfig();
    json["spec_decode_type"] = "eagle3";
    json["engine_role"] = "base";
    json["builder_config"]["spec_base"] = true;
    json["builder_config"]["max_spec_verify_tokens"] = 16;
    json["builder_config"]["num_swa_pages"] = 64; // Unused fallback budget must not enable SWA.
    // `max_spec_draft_tokens` is a draft-engine property and is not written
    // into base_config.json by the builder — intentionally omitted here.
    auto const path = writeJsonToTempFile(json);

    LLMEngineConfig cfg = parseEngineConfig(path);
    EXPECT_TRUE(cfg.isSpecDecodeBase);
    EXPECT_EQ(cfg.maxSpecVerifyTokens, 16);
    EXPECT_EQ(cfg.maxSpecDraftTokens, 0); // Base side leaves this at the default.
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
    EXPECT_EQ(cfg.kvCacheDtype, trt_edgellm::DataType::kFP8);
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
    // Intentionally omit max_spec_verify_tokens (the only required specConfig field on the base).
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

TEST_F(LLMEngineConfigTest, MultimodalPleRequiresPadToken)
{
    Json json = makeMinimalConfig();
    addPlePool(json);
    json["audio_token_id"] = 151656;
    auto const path = writeJsonToTempFile(json);

    // Given a multimodal PLE config without a pad token, when it is parsed, then validation rejects it.
    EXPECT_THROW(parseEngineConfig(path), std::runtime_error);
}

TEST_F(LLMEngineConfigTest, MultimodalPleParsesPadToken)
{
    Json json = makeMinimalConfig();
    addPlePool(json);
    json["audio_token_id"] = 151656;
    json["pad_token_id"] = 0;
    auto const path = writeJsonToTempFile(json);

    // Given a valid multimodal PLE pad token, when the config is parsed, then its token ID is retained.
    LLMEngineConfig cfg = parseEngineConfig(path);
    EXPECT_EQ(cfg.padTokenId, 0);
}

TEST_F(LLMEngineConfigTest, MultimodalPleRejectsOutOfRangePadToken)
{
    Json json = makeMinimalConfig();
    addPlePool(json);
    json["image_token_id"] = 151655;
    json["pad_token_id"] = json["vocab_size"];
    auto const path = writeJsonToTempFile(json);

    // Given a pad token beyond the PLE table, when the config is parsed, then validation rejects it.
    EXPECT_THROW(parseEngineConfig(path), std::runtime_error);
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

TEST_F(LLMEngineConfigTest, Fp8ReducedKVCacheCapacityPreservesBoundedSwaProfile)
{
    constexpr int32_t kMAX_BATCH_SIZE = 2;
    constexpr int32_t kMAX_KV_CACHE_CAPACITY = 8192;
    constexpr int32_t kSWA_WINDOW_SIZE = 128;

    Json json = makeMinimalConfig();
    json["num_hidden_layers"] = 1;
    json["kv_cache_dtype"] = "fp8";
    json["builder_config"]["max_kv_cache_capacity"] = kMAX_KV_CACHE_CAPACITY;
    json["builder_config"]["max_kv_pool_pages"] = computeMinimumKvPoolPages(kMAX_BATCH_SIZE, kMAX_KV_CACHE_CAPACITY);
    json["builder_config"]["num_swa_pages"] = computeMinimumSwaPoolPages(kMAX_BATCH_SIZE, kSWA_WINDOW_SIZE);
    json["layer_types"] = {"attention"};
    json["kv_layer_configs"]
        = Json::array({Json{{"num_kv_heads", 4}, {"head_dim", 64}, {"kv_cache_capacity", kSWA_WINDOW_SIZE}}});
    auto const path = writeJsonToTempFile(json);

    LLMEngineConfig cfg = parseEngineConfig(path);
    ASSERT_EQ(cfg.kvLayerConfigs.size(), 1U);
    EXPECT_EQ(cfg.kvCacheDtype, trt_edgellm::DataType::kFP8);
    EXPECT_EQ(cfg.numSwaPages, static_cast<int32_t>(computeMinimumSwaPoolPages(kMAX_BATCH_SIZE, kSWA_WINDOW_SIZE)));
    EXPECT_EQ(cfg.kvLayerConfigs[0].kvCacheCapacity, kSWA_WINDOW_SIZE);
    EXPECT_TRUE(cfg.supportsBoundedSwaKVCache());
    EXPECT_TRUE(cfg.usesBoundedSwaKVCache());
    EXPECT_EQ(cfg.getKVPoolPagesForLayer(cfg.kvLayerConfigs[0]), cfg.numSwaPages);
    EXPECT_EQ(cfg.getKVPoolPageProfileForLayer(cfg.kvLayerConfigs[0]),
        (std::array<int32_t, 3>{cfg.numSwaPages, cfg.numSwaPages, cfg.kvPoolPages}));

    cfg.setSwaKVCacheMode(SwaKVCacheMode::kFull);
    EXPECT_TRUE(cfg.supportsBoundedSwaKVCache());
    EXPECT_FALSE(cfg.usesBoundedSwaKVCache());
    EXPECT_EQ(cfg.getKVPoolPagesForLayer(cfg.kvLayerConfigs[0]), cfg.kvPoolPages);
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
// Per-layer LayerSpec table
// ===========================================================================

namespace
{

//! Assert `layer` declares a state of `kind` with exactly the given geometry and precision.
void expectState(LayerSpec const& layer, StateKind kind, Coords const& shape, trt_edgellm::DataType dtype)
{
    StateSpec const* state = layer.findState(kind);
    ASSERT_NE(state, nullptr) << "layer " << layer.absLayerIdx << " declares no " << stateKindName(kind) << " state";
    EXPECT_EQ(state->absLayerIdx, layer.absLayerIdx);
    EXPECT_EQ(state->shape, shape) << stateKindName(kind) << " shape is " << state->shape.formatString()
                                   << ", expected " << shape.formatString();
    EXPECT_EQ(state->dataType, dtype);
}

//! The two states a recurrent layer of `makeHybridConfig()` carries, named so assertions can
//! refer to them without restating the fixture's geometry.
Coords const kHybridRecurrentShape{16, 32, 64};
Coords const kHybridConvShape{128, 4};

//! The n-gram conv state of `makeNgramConfig()`: `ngram_conv_dim` channels over
//! `(ngram_conv_kernel - 1) * ngram_size` steps.
Coords const kNgramConvShape{10240, 9};

} // namespace

TEST_F(LLMEngineConfigTest, AttentionOnlyLayerSpecsCarryKvAndNoState)
{
    // Given a pure-attention engine,
    // When the config is parsed,
    // Then the table must be dense over the decoder stack with every layer attending and carrying
    // no per-slot state -- a spurious state here would make the runtime allocate a pool for a
    // mixer the model does not have.
    auto const cfg = parseEngineConfig(writeJsonToTempFile(makeMinimalConfig()));

    ASSERT_EQ(cfg.layers.size(), 12U);
    for (int32_t index = 0; index < 12; ++index)
    {
        SCOPED_TRACE(index);
        LayerSpec const& layer = cfg.layers[index];
        EXPECT_EQ(layer.absLayerIdx, index);
        ASSERT_TRUE(layer.isAttention());
        EXPECT_FALSE(layer.isRecurrent());
        EXPECT_EQ(layer.kv->numKVHeads, 4);
        EXPECT_EQ(layer.kv->headDim, 64);
        EXPECT_EQ(layer.kvSharingDonor, -1);
        EXPECT_TRUE(layer.states.empty());
    }
}

TEST_F(LLMEngineConfigTest, HybridLayerSpecsPlaceMixerStatesAtAbsoluteLayerIndices)
{
    // Given a hybrid engine whose recurrent layers follow its attention layers,
    // When the config is parsed,
    // Then the recurrent and conv states must land on the absolute indices of the recurrent
    // layers, and the attention layers must consume `kv_layer_configs` in local order -- mixing
    // the two index spaces would attach a state to the wrong layer's bindings.
    auto const cfg = parseEngineConfig(writeJsonToTempFile(makeHybridConfig()));

    ASSERT_EQ(cfg.layers.size(), 12U);
    for (int32_t index = 0; index < 8; ++index)
    {
        SCOPED_TRACE(index);
        LayerSpec const& layer = cfg.layers[index];
        ASSERT_TRUE(layer.isAttention());
        EXPECT_FALSE(layer.isRecurrent());
        EXPECT_EQ(layer.kv->numKVHeads, 4);
        EXPECT_TRUE(layer.states.empty());
    }
    for (int32_t index = 8; index < 12; ++index)
    {
        SCOPED_TRACE(index);
        LayerSpec const& layer = cfg.layers[index];
        EXPECT_EQ(layer.absLayerIdx, index);
        EXPECT_FALSE(layer.isAttention());
        EXPECT_TRUE(layer.isRecurrent());
        ASSERT_EQ(layer.states.size(), 2U);
        expectState(layer, StateKind::kGdn, kHybridRecurrentShape, trt_edgellm::DataType::kHALF);
        expectState(layer, StateKind::kConv, kHybridConvShape, trt_edgellm::DataType::kHALF);
    }
}

TEST_F(LLMEngineConfigTest, InterleavedLayerSpecsMapKvConfigsByLocalAttentionIndex)
{
    // Given a hybrid engine that interleaves attention and recurrent layers, with each attention
    // layer carrying a distinct KV head count,
    // When the config is parsed,
    // Then each attention layer must receive the `kv_layer_configs` entry at its LOCAL attention
    // index, not its absolute layer index. The scalar-broadcast fixtures cannot catch this: they
    // emit every attention layer first, which makes the two index spaces agree.
    Json json = makeHybridConfig();
    Json layerTypes = Json::array();
    Json kvLayerConfigs = Json::array();
    int32_t localAttnIdx = 0;
    for (int32_t index = 0; index < 12; ++index)
    {
        bool const isMamba = index < 8 && index % 2 == 1;
        layerTypes.push_back(isMamba ? "mamba" : "attention");
        if (isMamba)
        {
            kvLayerConfigs.push_back(nullptr);
        }
        else
        {
            kvLayerConfigs.push_back(Json{{"num_kv_heads", 1 + localAttnIdx}, {"head_dim", 64}});
            ++localAttnIdx;
        }
    }
    json["layer_types"] = layerTypes;
    json["kv_layer_configs"] = kvLayerConfigs;

    auto const cfg = parseEngineConfig(writeJsonToTempFile(json));

    ASSERT_EQ(cfg.layers.size(), 12U);
    // Absolute layers 1, 3, 5, 7 are recurrent; the remaining eight attend in local order.
    std::array<int32_t, 12> const expectedKvHeads{1, 0, 2, 0, 3, 0, 4, 0, 5, 6, 7, 8};
    for (int32_t index = 0; index < 12; ++index)
    {
        SCOPED_TRACE(index);
        LayerSpec const& layer = cfg.layers[index];
        EXPECT_EQ(layer.absLayerIdx, index);
        if (expectedKvHeads[index] == 0)
        {
            EXPECT_FALSE(layer.isAttention());
            EXPECT_TRUE(layer.isRecurrent());
            expectState(layer, StateKind::kGdn, kHybridRecurrentShape, trt_edgellm::DataType::kHALF);
            expectState(layer, StateKind::kConv, kHybridConvShape, trt_edgellm::DataType::kHALF);
        }
        else
        {
            ASSERT_TRUE(layer.isAttention());
            EXPECT_EQ(layer.kv->numKVHeads, expectedKvHeads[index]);
            EXPECT_TRUE(layer.states.empty());
        }
    }
}

TEST_F(LLMEngineConfigTest, MixerStateKindComesFromGroupCountNotVerifyMode)
{
    // Given two hybrid engines that differ only in whether they declare B/C groups, the one
    // without them additionally asking for replay-mode spec verification,
    // When each config is parsed,
    // Then the group count alone must name the mixer: groups are a Mamba2 construct, while the
    // verify mode is a commit policy that either mixer may carry. Deriving the kind from the mode
    // would mislabel a GDN engine as Mamba2 and size its state with the wrong geometry.
    Json gdn = makeHybridConfig();
    gdn["recurrent_spec_verify_mode"] = "replay";
    auto const gdnCfg = parseEngineConfig(writeJsonToTempFile(gdn));
    ASSERT_EQ(gdnCfg.layers.size(), 12U);
    EXPECT_TRUE(gdnCfg.recurrentSpecVerifyUsesReplay);
    EXPECT_EQ(gdnCfg.layers[8].findState(StateKind::kMamba2), nullptr);
    expectState(gdnCfg.layers[8], StateKind::kGdn, kHybridRecurrentShape, trt_edgellm::DataType::kHALF);

    Json mamba2 = makeHybridConfig();
    mamba2["recurrent_state_num_groups"] = 2;
    auto const mamba2Cfg = parseEngineConfig(writeJsonToTempFile(mamba2));
    ASSERT_EQ(mamba2Cfg.layers.size(), 12U);
    EXPECT_FALSE(mamba2Cfg.recurrentSpecVerifyUsesReplay);
    EXPECT_EQ(mamba2Cfg.layers[8].findState(StateKind::kGdn), nullptr);
    expectState(mamba2Cfg.layers[8], StateKind::kMamba2, kHybridRecurrentShape, trt_edgellm::DataType::kHALF);
}

TEST_F(LLMEngineConfigTest, NgramLayerSpecCarriesAuxStateWithoutAnyMixerState)
{
    // Given an n-gram injected engine with no recurrent layers at all, whose injection sits on an
    // attention layer and declares its own precision,
    // When the config is parsed,
    // Then that layer must carry the n-gram conv state and nothing else, while remaining an
    // attention layer. The state has to stand alone: an implementation that only reached aux state
    // through a recurrent layer would allocate nothing here.
    Json json = makeNgramConfig();
    json["ngram_conv_state_dtype"] = "bf16";

    auto const cfg = parseEngineConfig(writeJsonToTempFile(json));

    ASSERT_EQ(cfg.numLinearAttnLayers, 0);
    ASSERT_EQ(cfg.layers.size(), 12U);
    LayerSpec const& injected = cfg.layers[1];
    ASSERT_EQ(injected.states.size(), 1U);
    expectState(injected, StateKind::kNgramConv, kNgramConvShape, trt_edgellm::DataType::kBF16);
    EXPECT_TRUE(injected.isAttention());
    EXPECT_FALSE(injected.isRecurrent());
    EXPECT_EQ(injected.findState(StateKind::kGdn), nullptr);
    EXPECT_EQ(injected.findState(StateKind::kConv), nullptr);

    for (int32_t index = 0; index < 12; ++index)
    {
        SCOPED_TRACE(index);
        EXPECT_EQ(cfg.layers[index].states.size(), index == 1 ? 1U : 0U);
    }
}

TEST_F(LLMEngineConfigTest, NgramStateCoexistsWithMixerStateOnARecurrentLayer)
{
    // Given a model that is both hybrid and n-gram injected, with the injection landing on a
    // recurrent layer and the three states exported at three different precisions,
    // When the config is parsed,
    // Then that layer must carry all three, each with its own geometry and dtype -- the aux state
    // must neither displace the mixer's states nor inherit the mixer conv's precision.
    Json json = makeNgramConfig();
    Json const hybrid = makeHybridConfig();
    for (char const* key :
        {"num_linear_attn_layers", "num_attention_layers", "recurrent_state_num_heads", "recurrent_state_head_dim",
            "recurrent_state_size", "conv_dim", "conv_kernel", "recurrent_state_dtype", "conv_state_dtype"})
    {
        json[key] = hybrid.at(key);
    }
    json["ngram_layers"] = Json::array({9});
    json["recurrent_state_dtype"] = "fp32";
    json["conv_state_dtype"] = "fp16";
    json["ngram_conv_state_dtype"] = "bf16";

    auto const cfg = parseEngineConfig(writeJsonToTempFile(json));

    ASSERT_EQ(cfg.layers.size(), 12U);
    LayerSpec const& injected = cfg.layers[9];
    ASSERT_EQ(injected.states.size(), 3U);
    EXPECT_FALSE(injected.isAttention());
    EXPECT_TRUE(injected.isRecurrent());
    expectState(injected, StateKind::kGdn, kHybridRecurrentShape, trt_edgellm::DataType::kFLOAT);
    expectState(injected, StateKind::kConv, kHybridConvShape, trt_edgellm::DataType::kHALF);
    expectState(injected, StateKind::kNgramConv, kNgramConvShape, trt_edgellm::DataType::kBF16);
    // The other recurrent layers keep only the mixer's two states.
    EXPECT_EQ(cfg.layers[8].states.size(), 2U);
}

TEST_F(LLMEngineConfigTest, RejectsNgramLayerOutsideTheLayerTable)
{
    // Given an n-gram injection naming a layer that exists in the decoder stack but not in the
    // parsed layer table, because the engine's attention and recurrent counts do not cover the
    // stack,
    // When the config is parsed,
    // Then it must be rejected. The table is what the runtime iterates, so an entry past its end
    // would silently drop a state the engine still binds.
    Json json = makeNgramConfig();
    json["num_attention_layers"] = 4;
    json["num_linear_attn_layers"] = 2;
    json["recurrent_state_num_heads"] = 16;
    json["recurrent_state_head_dim"] = 32;
    json["recurrent_state_size"] = 64;
    json["conv_dim"] = 128;
    json["conv_kernel"] = 4;
    json["recurrent_state_dtype"] = "fp16";
    json["conv_state_dtype"] = "fp16";
    // Inside the 12-layer stack, so the n-gram range check passes, but past the 6 layers the
    // scalar-broadcast fallback puts in the table.
    json["ngram_layers"] = Json::array({8});

    EXPECT_THROW(parseEngineConfig(writeJsonToTempFile(json)), std::runtime_error);
}

TEST_F(LLMEngineConfigTest, RejectsNgramLayerNamedTwice)
{
    // Given an n-gram injection that names the same layer twice,
    // When the config is parsed,
    // Then it must be rejected. A layer carries at most one state per kind, so a repeated entry
    // would give it two n-gram conv states and make the engine's single binding ambiguous.
    Json json = makeNgramConfig();
    json["ngram_layers"] = Json::array({1, 1});

    EXPECT_THROW(parseEngineConfig(writeJsonToTempFile(json)), std::runtime_error);
}

TEST_F(LLMEngineConfigTest, HeterogeneousAttentionLayerSpecsKeepPerLayerKvGeometry)
{
    // Given a Gemma-4 shaped engine that interleaves windowed and global attention, so head dim
    // and KV capacity both vary per layer,
    // When the config is parsed,
    // Then every layer's own head dim and capacity must survive into its spec. This is the real
    // on-disk shape: a uniform table would size the global layers' caches as windowed ones.
    constexpr int32_t kSWA_WINDOW_SIZE = 128;
    Json json = makeMinimalConfig();
    json["global_head_dim"] = 512;
    json["builder_config"]["num_swa_pages"] = computeMinimumSwaPoolPages(/*maxBatchSize=*/2, kSWA_WINDOW_SIZE);
    Json layerTypes = Json::array();
    Json kvLayerConfigs = Json::array();
    for (int32_t index = 0; index < 12; ++index)
    {
        layerTypes.push_back("attention");
        bool const isGlobal = index % 6 == 5;
        kvLayerConfigs.push_back(isGlobal
                ? Json{{"num_kv_heads", 1}, {"head_dim", 512}}
                : Json{{"num_kv_heads", 8}, {"head_dim", 256}, {"kv_cache_capacity", kSWA_WINDOW_SIZE}});
    }
    json["layer_types"] = layerTypes;
    json["kv_layer_configs"] = kvLayerConfigs;

    auto const cfg = parseEngineConfig(writeJsonToTempFile(json));

    ASSERT_EQ(cfg.layers.size(), 12U);
    for (int32_t index = 0; index < 12; ++index)
    {
        SCOPED_TRACE(index);
        LayerSpec const& layer = cfg.layers[index];
        ASSERT_TRUE(layer.isAttention());
        EXPECT_TRUE(layer.states.empty());
        bool const isGlobal = index % 6 == 5;
        EXPECT_EQ(layer.kv->numKVHeads, isGlobal ? 1 : 8);
        EXPECT_EQ(layer.kv->headDim, isGlobal ? 512 : 256);
        EXPECT_EQ(layer.kv->kvCacheCapacity, isGlobal ? 0 : kSWA_WINDOW_SIZE);
    }
}

TEST_F(LLMEngineConfigTest, KvSharingDonorsLandOnTheDonatingLayersInLocalIndexSpace)
{
    // Given a hybrid engine whose attention layers share KV caches, with the donor array indexed
    // by local attention index,
    // When the config is parsed,
    // Then each attention layer's spec must record its own donor, and the recorded values must
    // stay LOCAL attention indices -- the array is parsed in that space, so reinterpreting them as
    // absolute layer indices would point a sharer at the wrong cache.
    Json json = makeHybridConfig();
    json["layer_types"] = {"attention", "mamba", "attention", "attention", "mamba", "mamba", "mamba", "attention",
        "attention", "attention", "attention", "attention"};
    Json kvLayerConfigs = Json::array();
    for (auto const& layerType : json["layer_types"])
    {
        kvLayerConfigs.push_back(layerType == "mamba" ? Json(nullptr) : Json{{"num_kv_heads", 4}, {"head_dim", 64}});
    }
    json["kv_layer_configs"] = kvLayerConfigs;
    json["num_attention_layers"] = 8;
    // Local attention indices 0..7 sit at absolute layers 0, 2, 3, 7, 8, 9, 10, 11.
    json["kv_sharing_donors"] = {-1, 0, 1, -1, 3, 3, -1, 6};

    auto const cfg = parseEngineConfig(writeJsonToTempFile(json));

    ASSERT_EQ(cfg.layers.size(), 12U);
    // Absolute layer -> donor, with the donor still naming a local attention index. Layer 3's
    // donor of 1 is local attention 1, i.e. absolute layer 2.
    std::array<std::pair<int32_t, int32_t>, 8> const expected{
        {{0, -1}, {2, 0}, {3, 1}, {7, -1}, {8, 3}, {9, 3}, {10, -1}, {11, 6}}};
    for (auto const& [absLayerIdx, donor] : expected)
    {
        SCOPED_TRACE(absLayerIdx);
        LayerSpec const& layer = cfg.layers[absLayerIdx];
        ASSERT_TRUE(layer.isAttention());
        EXPECT_EQ(layer.kvSharingDonor, donor);
    }
    // Recurrent layers own no KV cache, so they never carry a donor.
    for (int32_t absLayerIdx : {1, 4, 5, 6})
    {
        SCOPED_TRACE(absLayerIdx);
        EXPECT_FALSE(cfg.layers[absLayerIdx].isAttention());
        EXPECT_EQ(cfg.layers[absLayerIdx].kvSharingDonor, -1);
    }
}

TEST_F(LLMEngineConfigTest, FormatEngineConfigSummarizesEveryLayersStates)
{
    // Given a hybrid engine,
    // When its config is formatted for the log,
    // Then the summary must name each layer's states, since this line is the only record of what
    // the runtime allocated when a deployment is debugged from logs alone.
    auto const cfg = parseEngineConfig(writeJsonToTempFile(makeHybridConfig()));

    std::string const formatted = formatEngineConfig(cfg);
    EXPECT_NE(formatted.find("layers=["), std::string::npos) << formatted;
    EXPECT_NE(formatted.find("8:+gdn"), std::string::npos) << formatted;
    EXPECT_NE(formatted.find("+conv"), std::string::npos) << formatted;
    EXPECT_NE(formatted.find("0:kv("), std::string::npos) << formatted;
}

// ===========================================================================
// Direct `layers` table parse (dual-read source of truth)
// ===========================================================================

namespace
{

//! One `layers[].states[]` entry: {kind, shape, dtype}.
Json stateJson(std::string const& kind, std::vector<int64_t> const& shape, std::string const& dtype = "fp16")
{
    return Json{{"kind", kind}, {"shape", shape}, {"dtype", dtype}};
}

//! One attention `layers[]` entry carrying a kv block. A donor < 0 is omitted so the entry matches
//! the common "owns its own KV" shape rather than spelling out the default.
Json attentionLayerJson(
    int32_t idx, int32_t numKvHeads, int32_t headDim, int32_t kvCacheCapacity = 0, int32_t kvSharingDonor = -1)
{
    Json kv{{"num_kv_heads", numKvHeads}, {"head_dim", headDim}, {"kv_cache_capacity", kvCacheCapacity}};
    if (kvSharingDonor >= 0)
    {
        kv["kv_sharing_donor"] = kvSharingDonor;
    }
    return Json{{"idx", idx}, {"kv", kv}};
}

//! One recurrent `layers[]` entry: a mixer state (gdn/mamba2 by group count) then its conv state,
//! sized to match `makeHybridConfig()`'s legacy geometry so the two paths can be compared.
Json recurrentLayerJson(int32_t idx, int32_t groupCount = 0)
{
    std::string const mixerKind = groupCount > 0 ? "mamba2" : "gdn";
    return Json{
        {"idx", idx}, {"states", Json::array({stateJson(mixerKind, {16, 32, 64}), stateJson("conv", {128, 4})})}};
}

//! A dense hybrid `layers` table mirroring `makeHybridConfig()`: eight attention layers (0-7) then
//! four recurrent layers (8-11).
Json hybridLayerTable(int32_t groupCount = 0)
{
    Json layers = Json::array();
    for (int32_t idx = 0; idx < 8; ++idx)
    {
        layers.push_back(attentionLayerJson(idx, 4, 64));
    }
    for (int32_t idx = 8; idx < 12; ++idx)
    {
        layers.push_back(recurrentLayerJson(idx, groupCount));
    }
    return layers;
}

//! Assert two configs carry byte-identical per-layer tables and the legacy members derived from them.
void expectSameLayerContract(LLMEngineConfig const& lhs, LLMEngineConfig const& rhs)
{
    ASSERT_EQ(lhs.layers.size(), rhs.layers.size());
    for (size_t idx = 0; idx < lhs.layers.size(); ++idx)
    {
        SCOPED_TRACE(idx);
        LayerSpec const& a = lhs.layers[idx];
        LayerSpec const& b = rhs.layers[idx];
        EXPECT_EQ(a.absLayerIdx, b.absLayerIdx);
        EXPECT_EQ(a.isAttention(), b.isAttention());
        EXPECT_EQ(a.kvSharingDonor, b.kvSharingDonor);
        if (a.isAttention() && b.isAttention())
        {
            EXPECT_EQ(a.kv->numKVHeads, b.kv->numKVHeads);
            EXPECT_EQ(a.kv->headDim, b.kv->headDim);
            EXPECT_EQ(a.kv->kvCacheCapacity, b.kv->kvCacheCapacity);
        }
        ASSERT_EQ(a.states.size(), b.states.size());
        for (size_t s = 0; s < a.states.size(); ++s)
        {
            EXPECT_EQ(a.states[s].kind, b.states[s].kind);
            EXPECT_EQ(a.states[s].shape, b.states[s].shape);
            EXPECT_EQ(a.states[s].dataType, b.states[s].dataType);
        }
    }
    EXPECT_EQ(lhs.layerTypes, rhs.layerTypes);
    EXPECT_EQ(lhs.numAttentionLayers, rhs.numAttentionLayers);
    EXPECT_EQ(lhs.numLinearAttnLayers, rhs.numLinearAttnLayers);
    EXPECT_EQ(lhs.recurrentStateNumHeads, rhs.recurrentStateNumHeads);
    EXPECT_EQ(lhs.recurrentStateHeadDim, rhs.recurrentStateHeadDim);
    EXPECT_EQ(lhs.recurrentStateSize, rhs.recurrentStateSize);
    EXPECT_EQ(lhs.recurrentStateDtype, rhs.recurrentStateDtype);
    EXPECT_EQ(lhs.convDim, rhs.convDim);
    EXPECT_EQ(lhs.convKernel, rhs.convKernel);
    EXPECT_EQ(lhs.convStateDtype, rhs.convStateDtype);
    ASSERT_EQ(lhs.kvLayerConfigs.size(), rhs.kvLayerConfigs.size());
    for (size_t idx = 0; idx < lhs.kvLayerConfigs.size(); ++idx)
    {
        SCOPED_TRACE(idx);
        EXPECT_EQ(lhs.kvLayerConfigs[idx].numKVHeads, rhs.kvLayerConfigs[idx].numKVHeads);
        EXPECT_EQ(lhs.kvLayerConfigs[idx].headDim, rhs.kvLayerConfigs[idx].headDim);
    }
}

} // namespace

TEST_F(LLMEngineConfigTest, LayersTableReproducesTheScalarDerivedTableForAHybridStack)
{
    // Given one hybrid config described by the legacy scalar/array fields and a second described by
    // an explicit `layers` table of the same geometry,
    // When each is parsed,
    // Then both must yield an identical per-layer table and identical legacy members. The dual-read
    // path has to be a drop-in for the derivation, or migrating a model to the table would silently
    // change what the runtime allocates.
    LLMEngineConfig const legacy = parseEngineConfig(writeJsonToTempFile(makeHybridConfig()));

    Json tableConfig = makeMinimalConfig();
    tableConfig["layers"] = hybridLayerTable();
    LLMEngineConfig const fromTable = parseEngineConfig(writeJsonToTempFile(tableConfig));

    expectSameLayerContract(legacy, fromTable);
    EXPECT_EQ(fromTable.numLinearAttnLayers, 4);
    EXPECT_EQ(fromTable.numAttentionLayers, 8);
}

TEST_F(LLMEngineConfigTest, LayersTableOverridesLegacyScalarsWhenBothArePresent)
{
    // Given a config that carries BOTH the legacy hybrid scalars and a `layers` table, with the two
    // disagreeing on the recurrent head dim,
    // When it is parsed,
    // Then the table's geometry must win. This pins the parse-order selector: were the table ignored
    // (or the branch inverted) the stale scalar value would survive into the recurrent state shape.
    Json json = makeHybridConfig();
    json["recurrent_state_head_dim"] = 99; // Conflicts with the table's 32.
    json["layers"] = hybridLayerTable();

    auto const cfg = parseEngineConfig(writeJsonToTempFile(json));

    EXPECT_EQ(cfg.recurrentStateHeadDim, 32);
    expectState(cfg.layers[8], StateKind::kGdn, kHybridRecurrentShape, trt_edgellm::DataType::kHALF);
}

TEST_F(LLMEngineConfigTest, LayersTableClassifiesStandaloneMlpAsStateless)
{
    // Given a hybrid stack that interleaves attention, recurrent, and standalone MLP/MoE layers --
    // the Nemotron-H shape where num_hidden_layers counts every layer but the routing table counts
    // only the stateful ones -- here 3 attention + 2 recurrent + 3 MLP over 8 absolute positions,
    // When the config is parsed,
    // Then the dense table keeps one entry per absolute layer (bare MLP slots included) while the
    // compact routing table restates only the five stateful layers in order. A bare slot that leaked
    // into the routing table would make the hybrid cache manager reserve a pool the engine never binds.
    Json json = makeMinimalConfig();
    json["num_hidden_layers"] = 8;
    json["layers"]
        = Json::array({attentionLayerJson(0, 4, 64), Json{{"idx", 1}}, recurrentLayerJson(2), Json{{"idx", 3}},
            attentionLayerJson(4, 4, 64), Json{{"idx", 5}}, recurrentLayerJson(6), attentionLayerJson(7, 4, 64)});

    auto const cfg = parseEngineConfig(writeJsonToTempFile(json));

    ASSERT_EQ(cfg.layers.size(), 8u);
    for (int32_t const mlpIdx : {1, 3, 5})
    {
        SCOPED_TRACE(mlpIdx);
        LayerSpec const& mlp = cfg.layers[mlpIdx];
        EXPECT_FALSE(mlp.isAttention());
        EXPECT_FALSE(mlp.isRecurrent());
        EXPECT_TRUE(mlp.states.empty());
    }
    ASSERT_EQ(cfg.layerTypes.size(), 5u);
    EXPECT_EQ(cfg.layerTypes[0], rt::HybridCacheManager::LayerType::kAttention);
    EXPECT_EQ(cfg.layerTypes[1], rt::HybridCacheManager::LayerType::kMamba);
    EXPECT_EQ(cfg.layerTypes[2], rt::HybridCacheManager::LayerType::kAttention);
    EXPECT_EQ(cfg.layerTypes[3], rt::HybridCacheManager::LayerType::kMamba);
    EXPECT_EQ(cfg.layerTypes[4], rt::HybridCacheManager::LayerType::kAttention);
    EXPECT_EQ(cfg.numAttentionLayers, 3);
    EXPECT_EQ(cfg.numLinearAttnLayers, 2);
    EXPECT_EQ(cfg.kvLayerConfigs.size(), 3u);
}

TEST_F(LLMEngineConfigTest, LayersTableCarriesPerLayerKvSharingDonor)
{
    // Given a table whose second attention layer shares the first layer's KV cache while the rest
    // own theirs,
    // When it is parsed,
    // Then the sharer's spec must record the donor and the derived donor array must expose it, while
    // the non-sharing layers stay at -1. Dropping the donor would make the sharer allocate a private
    // cache the engine never bound.
    Json json = makeMinimalConfig();
    json["num_hidden_layers"] = 3;
    json["layers"] = Json::array({attentionLayerJson(0, 4, 64), attentionLayerJson(1, 4, 64, 0, /*kvSharingDonor=*/0),
        attentionLayerJson(2, 4, 64)});

    auto const cfg = parseEngineConfig(writeJsonToTempFile(json));

    ASSERT_EQ(cfg.layers.size(), 3u);
    EXPECT_EQ(cfg.layers[0].kvSharingDonor, -1);
    EXPECT_EQ(cfg.layers[1].kvSharingDonor, 0);
    EXPECT_EQ(cfg.layers[2].kvSharingDonor, -1);
    ASSERT_EQ(cfg.kvSharingDonors.size(), 3u);
    EXPECT_EQ(cfg.kvSharingDonors[1], 0);
}

TEST_F(LLMEngineConfigTest, LayersTableAttachesNgramConvAsAuxStateOnAnAttentionLayer)
{
    // Given a table whose layer 1 attends and additionally declares an n-gram conv state,
    // When it is parsed,
    // Then that layer must stay an attention layer, carry the aux state, and surface in `ngramLayers`
    // -- the injection rides on an existing layer, so losing either the state or the index would drop
    // a binding the engine still expects.
    Json json = makeNgramConfig();
    json["num_hidden_layers"] = 3;
    json.erase("ngram_layers");
    Json layers
        = Json::array({attentionLayerJson(0, 4, 64), attentionLayerJson(1, 4, 64), attentionLayerJson(2, 4, 64)});
    layers.at(1)["states"] = Json::array({stateJson("ngram_conv", {10240, 9}, "bf16")});
    json["layers"] = layers;

    auto const cfg = parseEngineConfig(writeJsonToTempFile(json));

    ASSERT_EQ(cfg.layers.size(), 3u);
    LayerSpec const& injected = cfg.layers.at(1);
    EXPECT_TRUE(injected.isAttention());
    EXPECT_FALSE(injected.isRecurrent());
    expectState(injected, StateKind::kNgramConv, kNgramConvShape, trt_edgellm::DataType::kBF16);
    ASSERT_EQ(cfg.ngramLayers.size(), 1u);
    EXPECT_EQ(cfg.ngramLayers.at(0), 1);
}

TEST_F(LLMEngineConfigTest, LayersTableNgramSitesEnableFeatureMetadataAndPoolParsing)
{
    // Given an n-gram engine that has moved its injection sites into the `layers` table and
    // dropped the top-level `ngram_layers`, keeping only the n-gram feature geometry and row pool,
    // When the config is parsed,
    // Then the table's sites must be read before the feature metadata so the shared row pool is
    // actually parsed -- reading the sites afterwards would see no n-gram layers and leave the pool
    // unset with a zero-width state.
    Json json = makeNgramConfig();
    json.erase("ngram_layers");
    Json layers = Json::array();
    for (int32_t idx = 0; idx < 12; ++idx)
    {
        layers.push_back(attentionLayerJson(idx, 4, 64));
    }
    layers.at(1)["states"] = Json::array({stateJson("ngram_conv", {10240, 9})});
    json["layers"] = layers;

    auto const cfg = parseEngineConfig(writeJsonToTempFile(json));

    ASSERT_EQ(cfg.ngramLayers.size(), 1u);
    EXPECT_EQ(cfg.ngramLayers.at(0), 1);
    EXPECT_EQ(cfg.plePool.rowCount, 4 * kNgramRowsPerShard);
    EXPECT_EQ(cfg.plePool.rowWidth, kNgramRowWidth);
    EXPECT_EQ(cfg.plePool.heads, kNgramHeads);
    EXPECT_EQ(cfg.plePool.dtype, PleTable::DType::kFP8);
}

TEST_F(LLMEngineConfigTest, RejectsLayersTableNgramSitesWithoutFeatureMetadata)
{
    // Given a `layers` table whose layer declares an n-gram conv state while the config
    // omits the top-level n-gram feature geometry and row pool,
    // When the config is parsed,
    // Then it must be rejected rather than parsed with the pool unset and a zero-width state -- reading
    // the table's sites first turns the injection on, so the missing feature metadata is caught instead
    // of silently shipping a half-wired engine.
    Json json = makeMinimalConfig();
    json["num_hidden_layers"] = 3;
    Json layers
        = Json::array({attentionLayerJson(0, 4, 64), attentionLayerJson(1, 4, 64), attentionLayerJson(2, 4, 64)});
    layers.at(1)["states"] = Json::array({stateJson("ngram_conv", {10240, 9})});
    json["layers"] = layers;

    EXPECT_THROW(parseEngineConfig(writeJsonToTempFile(json)), std::runtime_error);
}

TEST_F(LLMEngineConfigTest, LayersTableNgramOnlyNonAttentionLayerClaimsNoRecurrentSlot)
{
    // Given a layers table whose middle layer attends to nothing and carries only an
    // n-gram conv state -- the injection riding a bare MLP/MoE layer -- flanked by two attention
    // layers,
    // When the config is parsed,
    // Then that layer must stay out of the recurrent routing: n-gram conv is not a mixer state, so
    // numLinearAttnLayers must stay zero even though the layer is not attention, while the site
    // still surfaces in ngramLayers. Counting it would make the hybrid cache manager reserve a
    // recurrent pool the engine never binds.
    Json json = makeNgramConfig();
    json["num_hidden_layers"] = 3;
    json.erase("ngram_layers");
    Json layers = Json::array(
        {attentionLayerJson(0, 4, 64), Json{{"idx", 1}, {"states", Json::array({stateJson("ngram_conv", {10240, 9})})}},
            attentionLayerJson(2, 4, 64)});
    json["layers"] = layers;

    auto const cfg = parseEngineConfig(writeJsonToTempFile(json));

    ASSERT_EQ(cfg.layers.size(), 3u);
    LayerSpec const& injected = cfg.layers.at(1);
    EXPECT_FALSE(injected.isAttention());
    EXPECT_FALSE(injected.isRecurrent());
    expectState(injected, StateKind::kNgramConv, kNgramConvShape, trt_edgellm::DataType::kHALF);
    EXPECT_EQ(cfg.numLinearAttnLayers, 0);
    EXPECT_EQ(cfg.numAttentionLayers, 2);
    ASSERT_EQ(cfg.ngramLayers.size(), 1u);
    EXPECT_EQ(cfg.ngramLayers.at(0), 1);
}

TEST_F(LLMEngineConfigTest, RejectsLayersTableWhoseLengthDiffersFromDecoderLayers)
{
    // Given a `layers` table with fewer entries than num_hidden_layers,
    // When it is parsed,
    // Then it must be rejected: the table is dense over the decoder stack, so a short table would
    // leave trailing layers without any binding description.
    Json json = makeMinimalConfig();
    json["num_hidden_layers"] = 12;
    json["layers"] = Json::array({attentionLayerJson(0, 4, 64)});

    EXPECT_THROW(parseEngineConfig(writeJsonToTempFile(json)), std::runtime_error);
}

TEST_F(LLMEngineConfigTest, RejectsLayersTableEntryWhoseIdxBreaksAbsoluteOrder)
{
    // Given a two-entry table whose second entry reports an idx that does not match its position,
    // When it is parsed,
    // Then it must be rejected: consumers index the table by absolute layer, so a mislabeled idx
    // would silently attach a layer's state to its neighbor.
    Json json = makeMinimalConfig();
    json["num_hidden_layers"] = 2;
    json["layers"] = Json::array({attentionLayerJson(0, 4, 64), attentionLayerJson(5, 4, 64)});

    EXPECT_THROW(parseEngineConfig(writeJsonToTempFile(json)), std::runtime_error);
}

TEST_F(LLMEngineConfigTest, RejectsLayersTableStateWithUnknownKind)
{
    // Given a table whose recurrent layer declares a state of an unrecognized kind,
    // When it is parsed,
    // Then it must be rejected rather than silently dropped: an unknown kind names a binding the
    // runtime cannot allocate, and swallowing it would ship a half-wired engine.
    Json json = makeMinimalConfig();
    json["num_hidden_layers"] = 2;
    Json badLayer{{"idx", 1}, {"states", Json::array({stateJson("ssm", {16, 32, 64})})}};
    json["layers"] = Json::array({attentionLayerJson(0, 4, 64), badLayer});

    EXPECT_THROW(parseEngineConfig(writeJsonToTempFile(json)), std::runtime_error);
}

TEST_F(LLMEngineConfigTest, RejectsLayersTableLayerDeclaringAKindTwice)
{
    // Given a table whose layer declares two conv states,
    // When it is parsed,
    // Then it must be rejected: a layer carries at most one state per kind, so a repeated kind would
    // make the engine's single conv binding ambiguous.
    Json json = makeMinimalConfig();
    json["num_hidden_layers"] = 2;
    Json badLayer{{"idx", 1}, {"states", Json::array({stateJson("conv", {128, 4}), stateJson("conv", {128, 4})})}};
    json["layers"] = Json::array({attentionLayerJson(0, 4, 64), badLayer});

    EXPECT_THROW(parseEngineConfig(writeJsonToTempFile(json)), std::runtime_error);
}

TEST_F(LLMEngineConfigTest, RejectsLayersTableWhereTwoRecurrentLayersDeclareAKindWithDifferentShapes)
{
    // Given a hybrid table whose recurrent layers all declare the same conv kind, but one layer sizes
    // its conv state differently from the rest,
    // When it is parsed,
    // Then it must be rejected: the legacy derivation projects one conv geometry that the runtime
    // broadcasts to every recurrent layer, so keeping only the last writer would silently allocate the
    // disagreeing layer a buffer that does not match its declared state.
    Json consistent = makeMinimalConfig();
    consistent["layers"] = hybridLayerTable();
    EXPECT_NO_THROW(parseEngineConfig(writeJsonToTempFile(consistent)));

    Json json = makeMinimalConfig();
    Json layers = hybridLayerTable();
    layers[9]["states"][1] = stateJson("conv", {256, 4});
    json["layers"] = layers;

    EXPECT_THROW(parseEngineConfig(writeJsonToTempFile(json)), std::runtime_error);
}

TEST_F(LLMEngineConfigTest, RejectsLayersTableWhereTwoRecurrentLayersDeclareAKindWithDifferentDtypes)
{
    // Given a hybrid table whose recurrent layers all declare the same mixer kind at one shape, but one
    // layer declares it in a different dtype from the rest,
    // When it is parsed,
    // Then it must be rejected: the legacy derivation projects one recurrent-state dtype that the
    // runtime broadcasts to every recurrent layer, so keeping only the last writer would silently
    // allocate the disagreeing layer a buffer of the wrong element type.
    Json consistent = makeMinimalConfig();
    consistent["layers"] = hybridLayerTable();
    EXPECT_NO_THROW(parseEngineConfig(writeJsonToTempFile(consistent)));

    Json json = makeMinimalConfig();
    Json layers = hybridLayerTable();
    layers[9]["states"][0] = stateJson("gdn", {16, 32, 64}, "bf16");
    json["layers"] = layers;

    EXPECT_THROW(parseEngineConfig(writeJsonToTempFile(json)), std::runtime_error);
}

TEST_F(LLMEngineConfigTest, RejectsLayersTableMixerStateWithWrongRank)
{
    // Given a hybrid table whose recurrent layer declares its mixer state with a 2-D shape instead of
    // the 3-D [heads, head_dim, state_size] row,
    // When it is parsed,
    // Then it must be rejected: the runtime reads the mixer slot as three extents, so a 2-D shape
    // would leave one dimension unspecified and mis-size the recurrent buffer.
    Json consistent = makeMinimalConfig();
    consistent["layers"] = hybridLayerTable();
    EXPECT_NO_THROW(parseEngineConfig(writeJsonToTempFile(consistent)));

    Json json = makeMinimalConfig();
    Json layers = hybridLayerTable();
    // Corrupt every recurrent layer's mixer identically so the cross-layer geometry-consistency check
    // stays satisfied and the per-kind rank check is the only guard that can reject this config.
    for (int32_t idx = 8; idx < 12; ++idx)
    {
        layers[idx]["states"][0] = stateJson("gdn", {16, 32});
    }
    json["layers"] = layers;

    EXPECT_THROW(parseEngineConfig(writeJsonToTempFile(json)), std::runtime_error);
}

TEST_F(LLMEngineConfigTest, RejectsLayersTableConvStateWithWrongRank)
{
    // Given a hybrid table whose recurrent layer declares its conv state with a 3-D shape instead of
    // the 2-D [channels, width] row,
    // When it is parsed,
    // Then it must be rejected: a dilated conv slot is two extents, so an extra dimension describes a
    // binding the runtime never allocates.
    Json json = makeMinimalConfig();
    Json layers = hybridLayerTable();
    // Corrupt every recurrent layer's conv identically so the cross-layer geometry-consistency check
    // stays satisfied and the per-kind rank check is the only guard that can reject this config.
    for (int32_t idx = 8; idx < 12; ++idx)
    {
        layers[idx]["states"][1] = stateJson("conv", {128, 4, 1});
    }
    json["layers"] = layers;

    EXPECT_THROW(parseEngineConfig(writeJsonToTempFile(json)), std::runtime_error);
}

TEST_F(LLMEngineConfigTest, RejectsLayersTableStateWithNonPositiveDim)
{
    // Given a hybrid table whose recurrent layer sizes one mixer-state dimension at zero,
    // When it is parsed,
    // Then it must be rejected: every state dimension is a buffer extent the runtime allocates, so a
    // non-positive value would request an empty or invalid allocation.
    Json json = makeMinimalConfig();
    Json layers = hybridLayerTable();
    // Zero the same mixer dimension on every recurrent layer so the cross-layer geometry-consistency
    // check stays satisfied and the non-positive-dim guard is the only thing that can reject this.
    for (int32_t idx = 8; idx < 12; ++idx)
    {
        layers[idx]["states"][0] = stateJson("gdn", {16, 0, 64});
    }
    json["layers"] = layers;

    EXPECT_THROW(parseEngineConfig(writeJsonToTempFile(json)), std::runtime_error);
}

TEST_F(LLMEngineConfigTest, RejectsLayersTableRecurrentLayerMissingConvState)
{
    // Given a hybrid table whose recurrent layer declares a mixer state but no paired conv state,
    // When it is parsed,
    // Then it must be rejected: the runtime allocates and commits a mixer and its conv state together,
    // so a mixer without its conv is a half-wired recurrent layer.
    Json json = makeMinimalConfig();
    Json layers = hybridLayerTable();
    layers[8]["states"] = Json::array({stateJson("gdn", {16, 32, 64})});
    json["layers"] = layers;

    EXPECT_THROW(parseEngineConfig(writeJsonToTempFile(json)), std::runtime_error);
}

TEST_F(LLMEngineConfigTest, RejectsLayersTableConvOnlyRecurrentLayer)
{
    // Given a hybrid table whose recurrent layer declares a conv state but no mixer to pair it with,
    // When it is parsed,
    // Then it must be rejected: a bare conv state names a binding with no mixer feeding it, so the
    // pairing rule holds in both directions.
    Json json = makeMinimalConfig();
    Json layers = hybridLayerTable();
    layers[8]["states"] = Json::array({stateJson("conv", {128, 4})});
    json["layers"] = layers;

    EXPECT_THROW(parseEngineConfig(writeJsonToTempFile(json)), std::runtime_error);
}

TEST_F(LLMEngineConfigTest, RejectsLayersTableLayerDeclaringBothMixerKinds)
{
    // Given a table whose recurrent layer declares both a gdn and a mamba2 mixer state,
    // When it is parsed,
    // Then it must be rejected: a layer carries at most one mixer lineage, so naming both makes the
    // single recurrent binding ambiguous.
    Json json = makeMinimalConfig();
    Json layers = hybridLayerTable();
    layers[8]["states"]
        = Json::array({stateJson("gdn", {16, 32, 64}), stateJson("mamba2", {16, 32, 64}), stateJson("conv", {128, 4})});
    json["layers"] = layers;

    EXPECT_THROW(parseEngineConfig(writeJsonToTempFile(json)), std::runtime_error);
}

TEST_F(LLMEngineConfigTest, RejectsLayersTableMixingGdnAndMamba2AcrossLayers)
{
    // Given a hybrid table whose recurrent layers are internally well-formed but one feeds the mixer
    // group as gdn while another feeds it as mamba2,
    // When it is parsed,
    // Then it must be rejected: the stack is single-lineage, projecting one `recurrentState*` group
    // the runtime broadcasts, so a mixed stack would size one lineage's layers with the other's.
    Json json = makeMinimalConfig();
    Json layers = hybridLayerTable(); // all gdn
    layers[9]["states"][0] = stateJson("mamba2", {16, 32, 64});
    json["layers"] = layers;

    EXPECT_THROW(parseEngineConfig(writeJsonToTempFile(json)), std::runtime_error);
}

TEST_F(LLMEngineConfigTest, RejectsLayersTableNgramConvShapeMismatch)
{
    // Given an n-gram table whose conv state's width does not equal (ngram_conv_kernel-1)*ngram_size,
    // When it is parsed,
    // Then it must be rejected: the dilated-conv history length is fixed by the hash geometry, so a
    // disagreeing width would bind a conv state the kernel cannot drive.
    Json good = makeNgramConfig();
    good.erase("ngram_layers");
    Json goodLayers
        = Json::array({attentionLayerJson(0, 4, 64), attentionLayerJson(1, 4, 64), attentionLayerJson(2, 4, 64)});
    goodLayers.at(1)["states"] = Json::array({stateJson("ngram_conv", {10240, 9})});
    good["num_hidden_layers"] = 3;
    good["layers"] = goodLayers;
    EXPECT_NO_THROW(parseEngineConfig(writeJsonToTempFile(good)));

    Json json = makeNgramConfig();
    json.erase("ngram_layers");
    Json layers
        = Json::array({attentionLayerJson(0, 4, 64), attentionLayerJson(1, 4, 64), attentionLayerJson(2, 4, 64)});
    layers.at(1)["states"] = Json::array({stateJson("ngram_conv", {10240, 12})});
    json["num_hidden_layers"] = 3;
    json["layers"] = layers;

    EXPECT_THROW(parseEngineConfig(writeJsonToTempFile(json)), std::runtime_error);
}

TEST_F(LLMEngineConfigTest, RejectsLayersTableKvCacheCapacityThatIsNotAnInteger)
{
    // Given a table attention layer whose kv_cache_capacity is a floating-point value,
    // When it is parsed,
    // Then it must be rejected rather than coerced: mirroring the legacy `kv_layer_configs` guard, a
    // non-integer capacity would silently truncate to a mis-sized per-layer cache.
    Json json = makeMinimalConfig();
    json["num_hidden_layers"] = 2;
    Json badLayer = attentionLayerJson(1, 4, 64);
    // A fractional value below 1 so a silent truncation would land on 0 — the engine's "full
    // capacity" marker — and parse cleanly through every downstream capacity guard. Only the
    // integer-type check can reject it, so a green mutation here means that check, not another, is
    // doing the work.
    badLayer["kv"]["kv_cache_capacity"] = 0.5;
    json["layers"] = Json::array({attentionLayerJson(0, 4, 64), badLayer});

    EXPECT_THROW(parseEngineConfig(writeJsonToTempFile(json)), std::runtime_error);
}

TEST_F(LLMEngineConfigTest, RejectsLayersTableKvSharingDonorOutOfRange)
{
    // Given a table whose attention layer names a kv_sharing_donor past the attention-layer count,
    // When it is parsed,
    // Then it must be rejected: the donor is a local attention index, so an out-of-range value points
    // at a KV cache no layer owns.
    Json json = makeMinimalConfig();
    json["num_hidden_layers"] = 3;
    json["layers"] = Json::array({attentionLayerJson(0, 4, 64), attentionLayerJson(1, 4, 64, 0, /*kvSharingDonor=*/7),
        attentionLayerJson(2, 4, 64)});

    EXPECT_THROW(parseEngineConfig(writeJsonToTempFile(json)), std::runtime_error);
}

TEST_F(LLMEngineConfigTest, RejectsLayersTableKvSharingDonorPointingAtItself)
{
    // Given a table whose attention layer names itself as its own kv_sharing_donor,
    // When it is parsed,
    // Then it must be rejected: a layer cannot both own and borrow the same cache, so a self-donor is
    // a cycle the runtime cannot resolve.
    Json json = makeMinimalConfig();
    json["num_hidden_layers"] = 2;
    json["layers"] = Json::array({attentionLayerJson(0, 4, 64), attentionLayerJson(1, 4, 64, 0, /*kvSharingDonor=*/1)});

    EXPECT_THROW(parseEngineConfig(writeJsonToTempFile(json)), std::runtime_error);
}

TEST_F(LLMEngineConfigTest, LayersTableWithRecurrentLayerParsesWithoutTopLevelRecurrentStateDtype)
{
    // Given a hybrid table that carries its recurrent/conv dtypes inside its state entries and omits
    // the top-level recurrent_state_dtype / conv_state_dtype scalars a pre-table engine required,
    // When it is parsed,
    // Then it must parse: the table is the source of truth for state dtype, so a table engine must not
    // be forced to also carry the legacy top-level dtype (this is the R9 fix). The derived dtypes come
    // from the state entries.
    Json json = makeMinimalConfig();
    Json layers = hybridLayerTable();
    // Declare the state dtypes as bf16, away from the member's kHALF default, so the derived-dtype
    // assertions below fail if the table is not actually read rather than silently matching the
    // default.
    for (int32_t idx = 8; idx < 12; ++idx)
    {
        layers[idx]["states"][0] = stateJson("gdn", {16, 32, 64}, "bf16");
        layers[idx]["states"][1] = stateJson("conv", {128, 4}, "bf16");
    }
    json["layers"] = layers;
    ASSERT_FALSE(json.contains("recurrent_state_dtype"));
    ASSERT_FALSE(json.contains("conv_state_dtype"));

    LLMEngineConfig cfg;
    EXPECT_NO_THROW(cfg = parseEngineConfig(writeJsonToTempFile(json)));
    EXPECT_EQ(cfg.numLinearAttnLayers, 4);
    EXPECT_EQ(cfg.recurrentStateDtype, trt_edgellm::DataType::kBF16);
    EXPECT_EQ(cfg.convStateDtype, trt_edgellm::DataType::kBF16);
}

TEST_F(LLMEngineConfigTest, NemotronShapedCompactLayerTypesLeaveTheDerivedTableEmpty)
{
    // Given a Nemotron-H-shaped legacy config whose compact layer_types/kv_layer_configs count only
    // the stateful layers (fewer than num_hidden_layers, which counts every layer including the
    // dropped MLP/MoE ones),
    // When it is parsed,
    // Then buildLayerSpecs cannot recover absolute indices from the compact array and must leave
    // cfg.layers empty rather than emit a mis-indexed table; the legacy routing members still populate
    // (M1). A non-empty table here would attach states to the wrong absolute layers.
    Json json = makeMinimalConfig();
    json["num_hidden_layers"] = 4; // Two stateful layers are described; two MLP layers are dropped.
    json["layer_types"] = Json::array({"attention", "mamba"});
    json["kv_layer_configs"] = Json::array({Json{{"num_kv_heads", 4}, {"head_dim", 64}}, Json(nullptr)});
    json["num_linear_attn_layers"] = 1;
    json["num_attention_layers"] = 1;
    json["recurrent_state_num_heads"] = 16;
    json["recurrent_state_head_dim"] = 32;
    json["recurrent_state_size"] = 64;
    json["conv_dim"] = 128;
    json["conv_kernel"] = 4;
    json["recurrent_state_dtype"] = "fp16";
    json["conv_state_dtype"] = "fp16";

    auto const cfg = parseEngineConfig(writeJsonToTempFile(json));

    EXPECT_TRUE(cfg.layers.empty());
    EXPECT_EQ(cfg.layerTypes.size(), 2U);
    EXPECT_EQ(cfg.numAttentionLayers, 1);
    EXPECT_EQ(cfg.numLinearAttnLayers, 1);
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
    EXPECT_EQ(cfg.specVerifyDims(2, 8, SpecVerifyLayout::kChain).executionPhaseLen,
        static_cast<int64_t>(ExecutionPhase::kSpecTargetVerify));
    EXPECT_EQ(cfg.denoiseDims(2, 8).executionPhaseLen, static_cast<int64_t>(ExecutionPhase::kDiffusionDenoise));
    EXPECT_EQ(cfg.diffusionCommitDims(2, 8).executionPhaseLen, static_cast<int64_t>(ExecutionPhase::kDiffusionCommit));
}

TEST(LLMEngineConfigRecipesTest, SpecAndDiffusionRecipesUsePhysicalTokenExtent)
{
    LLMEngineConfig cfg = makeRecipeConfig(/*maxKV=*/4096, /*mrope=*/false);
    EXPECT_EQ(cfg.proposalDims(/*batch=*/3, /*paddedTreeSize=*/8, /*draftTopK=*/4).seqLen, 24);
    EXPECT_EQ(cfg.proposalDims(/*batch=*/3, /*paddedTreeSize=*/8, /*draftTopK=*/4).contextSequenceCount, 0);
    EXPECT_EQ(cfg.specVerifyDims(/*batch=*/3, /*verifyTokenCount=*/8, SpecVerifyLayout::kChain).seqLen, 24);
    EXPECT_EQ(
        cfg.specVerifyDims(/*batch=*/3, /*verifyTokenCount=*/8, SpecVerifyLayout::kChain).contextSequenceCount, 0);
    EXPECT_EQ(cfg.denoiseDims(/*batch=*/3, /*canvasLen=*/8).seqLen, 24);
    EXPECT_EQ(cfg.denoiseDims(/*batch=*/3, /*canvasLen=*/8).contextSequenceCount, 0);
    EXPECT_EQ(cfg.diffusionCommitDims(/*batch=*/3, /*commitLen=*/8).seqLen, 24);
    EXPECT_EQ(cfg.diffusionCommitDims(/*batch=*/3, /*commitLen=*/8).contextSequenceCount, 0);
}

TEST(LLMEngineConfigRecipesTest, SpecVerifySelectsEveryPhysicalRow)
{
    auto const cfg = makeRecipeConfig(/*maxKV=*/8192, /*mrope=*/false);
    auto const d = cfg.specVerifyDims(/*batch=*/1, /*verifyTokenCount=*/8, SpecVerifyLayout::kChain);
    EXPECT_EQ(d.batch, 1);
    EXPECT_EQ(d.seqLen, 8);
    EXPECT_EQ(d.kvLen, 8192);
    EXPECT_EQ(d.selectLen, 8);
    EXPECT_EQ(d.attnMaskSeqLen, 8); // verifyTokenCount — proposal attention shape
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
    auto const verifyDims = cfg.specVerifyDims(/*batch=*/2, /*verifyTokenCount=*/16, SpecVerifyLayout::kChain);
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
    config["builder_config"]["max_spec_draft_tokens"] = 4;
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
    // Presence of a draft engine is indicated by maxSpecDraftTokens > 0.
    EXPECT_FALSE(cfg.isSpecDecodeBase);
    EXPECT_EQ(cfg.maxSpecDraftTokens, 4);
    EXPECT_EQ(cfg.baseModelHiddenSize, 768);
}

TEST_F(LLMEngineConfigTest, DraftEngineBuildsAnAttentionOnlyLayerTable)
{
    // Given a draft engine config,
    // When it is parsed,
    // Then its layer table must be built too and describe a pure-attention stack: the draft shares
    // the base's table machinery, and leaving it empty would make a generic consumer treat the
    // draft as stateless.
    auto const cfg = parseDraftEngineConfig(writeJsonToTempFile(makeMinimalDraftConfig()));

    ASSERT_EQ(cfg.layers.size(), 12U);
    for (int32_t index = 0; index < 12; ++index)
    {
        SCOPED_TRACE(index);
        LayerSpec const& layer = cfg.layers[index];
        EXPECT_EQ(layer.absLayerIdx, index);
        ASSERT_TRUE(layer.isAttention());
        EXPECT_EQ(layer.kv->headDim, 64);
        EXPECT_TRUE(layer.states.empty());
    }
}

TEST_F(LLMEngineConfigTest, DraftRaggedQueryCapacityCoversProposalWidth)
{
    Json json = makeMinimalDraftConfig();
    json["builder_config"]["max_spec_draft_tokens"] = 129;

    LLMEngineConfig const treeDominates = parseDraftEngineConfig(writeJsonToTempFile(json));
    EXPECT_EQ(treeDominates.maxQueryLength, 129);
    EXPECT_EQ(treeDominates.maxPhysicalTokens, 258);

    json["builder_config"]["max_spec_draft_tokens"] = 140;
    EXPECT_EQ(parseDraftEngineConfig(writeJsonToTempFile(json)).maxQueryLength, 140);
}

TEST_F(LLMEngineConfigTest, DSparkDraftCapacityIsPhysicalProfileWidth)
{
    Json json = makeMinimalDraftConfig();
    json["spec_decode_type"] = "dspark";
    json["builder_config"]["max_input_len"] = 8;
    json["builder_config"]["max_spec_draft_tokens"] = 16;
    json["dspark_config"]["sample_from_anchor"] = false;
    json["dspark_config"]["target_layer_ids"] = {0};

    LLMEngineConfig const cfg = parseDraftEngineConfig(writeJsonToTempFile(json));
    EXPECT_FALSE(cfg.dsparkSampleFromAnchor);
    EXPECT_EQ(cfg.maxQueryLength, 16);
    EXPECT_EQ(cfg.maxPhysicalTokens, 32);
}

TEST_F(LLMEngineConfigTest, RejectsDSparkNonAnchorDraftCapacityOverflow)
{
    Json json = makeMinimalDraftConfig();
    json["spec_decode_type"] = "dspark";
    json["builder_config"]["max_batch_size"] = 2;
    json["builder_config"]["max_input_len"] = 1;
    json["builder_config"]["max_spec_draft_tokens"] = std::numeric_limits<int32_t>::max();
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

TEST(AlpamayoActionKVDataTypeTest, MatchingTypesDoNotImposeBackendRestrictions)
{
    // Given matching pool and binding dtypes, when validating sharing, then leave backend support to the plugin.
    for (auto const type : {nvinfer1::DataType::kHALF, nvinfer1::DataType::kFP8, nvinfer1::DataType::kBF16})
    {
        EXPECT_NO_THROW(Alpamayo1ActionRunner::validateKVPoolDataType(type, type, "past_key_values_0"));
    }
}

TEST(AlpamayoActionKVDataTypeTest, Fp8PoolAcceptsInt8ByteTransport)
{
    // Given an FP8 prefix pool, when using an INT8 binding, then accept its unchanged E4M3 bytes.
    EXPECT_NO_THROW(Alpamayo1ActionRunner::validateKVPoolDataType(
        nvinfer1::DataType::kFP8, nvinfer1::DataType::kINT8, "past_key_values_0"));
}

TEST(AlpamayoActionKVDataTypeTest, Int8TransportRejectsNonFp8Pools)
{
    // Given an FP16 prefix pool, when using an INT8 binding, then reject the incompatible encoding.
    EXPECT_THROW(Alpamayo1ActionRunner::validateKVPoolDataType(
                     nvinfer1::DataType::kHALF, nvinfer1::DataType::kINT8, "past_key_values_0"),
        std::runtime_error);
}

TEST(AlpamayoActionKVDataTypeTest, MismatchedTypesAreRejected)
{
    // Given FP16 bindings, when sharing a pool of another dtype, then reject both input and output mismatches.
    for (auto const type : {nvinfer1::DataType::kFP8, nvinfer1::DataType::kBF16})
    {
        for (auto const* name : {"past_key_values_0", "present_key_values_0"})
        {
            EXPECT_THROW(Alpamayo1ActionRunner::validateKVPoolDataType(type, nvinfer1::DataType::kHALF, name),
                std::runtime_error);
        }
    }
}

TEST(AlpamayoActionPageCapacityTest, PoolProfileMustMatchActualTotal)
{
    // Given fixed action profiles, when binding an LLM pool, then require matching MIN, OPT, and MAX page counts.
    EXPECT_NO_THROW(Alpamayo1ActionRunner::validateKVPoolPageCount(34, 34, 34, 34));
    EXPECT_NO_THROW(Alpamayo1ActionRunner::validateKVPoolPageCount(42, 42, 42, 42));
    EXPECT_THROW(Alpamayo1ActionRunner::validateKVPoolPageCount(42, 34, 34, 34), std::runtime_error);
    EXPECT_THROW(Alpamayo1ActionRunner::validateKVPoolPageCount(34, 33, 34, 34), std::runtime_error);
    EXPECT_THROW(Alpamayo1ActionRunner::validateKVPoolPageCount(34, 34, 35, 34), std::runtime_error);
    EXPECT_THROW(Alpamayo1ActionRunner::validateKVPoolPageCount(34, 34, 34, 42), std::runtime_error);
}

namespace
{

//! The `dflash_config` the LiLiCorr exporter writes for the draft and base runtime configs.
Json makeLiLiCorrDFlashConfig()
{
    return Json{{"version", 2}, {"target_layer_ids", {1, 18, 35, 51, 68, 85}}, {"block_size", 8},
        {"mask_token_id", 990}, {"is_causal", false}, {"conv_kernel_size", 2}, {"conv_group_size", 16},
        {"supports_probabilistic_sampling", true}, {"projector_type", "lilicorr"}, {"lilicorr_candidate_topk", 8}};
}

Json makeLiLiCorrDraftConfig()
{
    Json json = makeMinimalDraftConfig();
    json["spec_decode_type"] = "dflash";
    json["num_hidden_layers"] = 5;
    json["base_model_hidden_size"] = 768 * 6;
    json["builder_config"]["max_spec_draft_tokens"] = 8;
    json["dflash_config"] = makeLiLiCorrDFlashConfig();
    return json;
}

Json makeDFlash2SelectorDraftConfig()
{
    Json json = makeMinimalDraftConfig();
    json["spec_decode_type"] = "dflash";
    json["num_hidden_layers"] = 5;
    json["base_model_hidden_size"] = 768 * 5;
    json["builder_config"]["max_spec_draft_tokens"] = 8;
    json["dflash_config"] = {{"version", 2}, {"target_layer_ids", {1, 3, 5, 7, 11}}, {"block_size", 8},
        {"mask_token_id", 248070}, {"is_causal", false}, {"conv_kernel_size", 2}, {"conv_group_size", 16},
        {"selector_rank", 256}, {"selector_top_k", 16}, {"supports_probabilistic_sampling", true}};
    return json;
}

} // namespace

TEST_F(LLMEngineConfigTest, ParseLiLiCorrDraftContract)
{
    // Given the LiLiCorr draft config.json the exporter writes: dflash_config version 2, projector_type "lilicorr",
    // lilicorr_candidate_topk 8, six target layer IDs and no selector_rank/selector_top_k/selector_file,
    // When the draft engine config is parsed,
    // Then it is accepted as a DFlash V2 LiLiCorr draft with candidate top-K 8 and all six target layer IDs.
    LLMEngineConfig const cfg = parseDraftEngineConfig(writeJsonToTempFile(makeLiLiCorrDraftConfig()));

    EXPECT_EQ(cfg.dflashVersion, DFlashVersion::kV2);
    EXPECT_EQ(cfg.dflashProjector, DFlashProjector::kLiLiCorr);
    EXPECT_EQ(cfg.specLiLiCorrTopK, 8);
    EXPECT_EQ(cfg.specTargetLayerIds, std::vector<int32_t>({1, 18, 35, 51, 68, 85}));
}

TEST_F(LLMEngineConfigTest, LiLiCorrRejectsNonPowerOfTwoTopK)
{
    // Given LiLiCorr draft configs that differ only in lilicorr_candidate_topk,
    // When each draft engine config is parsed,
    // Then the powers of two 1, 2, 4, 8 and 16 are accepted and 0, 6 and 32 are rejected.
    for (int32_t const topK : {1, 2, 4, 8, 16})
    {
        Json json = makeLiLiCorrDraftConfig();
        json["dflash_config"]["lilicorr_candidate_topk"] = topK;
        EXPECT_EQ(parseDraftEngineConfig(writeJsonToTempFile(json)).specLiLiCorrTopK, topK);
    }
    for (int32_t const topK : {0, 6, 32})
    {
        Json json = makeLiLiCorrDraftConfig();
        json["dflash_config"]["lilicorr_candidate_topk"] = topK;
        EXPECT_THROW(parseDraftEngineConfig(writeJsonToTempFile(json)), std::runtime_error) << "topK=" << topK;
    }
}

TEST_F(LLMEngineConfigTest, LiLiCorrRejectsSelectorKeys)
{
    // Given the LiLiCorr draft config.json the exporter writes,
    // When it additionally carries selector_rank, selector_top_k or selector_file,
    // Then parsing is rejected naming that key instead of silently ignoring it.
    ASSERT_NO_THROW(parseDraftEngineConfig(writeJsonToTempFile(makeLiLiCorrDraftConfig())));
    for (char const* const key : {"selector_rank", "selector_top_k", "selector_file"})
    {
        Json json = makeLiLiCorrDraftConfig();
        json["dflash_config"][key]
            = key == std::string("selector_file") ? Json("dflash2_selector.safetensors") : Json(16);
        EXPECT_THAT([&json] { parseDraftEngineConfig(writeJsonToTempFile(json)); },
            ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr(key)));
    }
}

TEST_F(LLMEngineConfigTest, ProjectorTypeRequiresDFlashV2)
{
    // Given a DFlash V1 draft config (dflash_config version 1) that parses without a projector_type,
    // When it additionally names projector_type 'lilicorr',
    // Then parsing is rejected because a projector applies only to DFlash V2.
    Json json = makeLiLiCorrDraftConfig();
    json["dflash_config"]["version"] = 1;
    json["dflash_config"].erase("projector_type");
    ASSERT_NO_THROW(parseDraftEngineConfig(writeJsonToTempFile(json)));

    json["dflash_config"]["projector_type"] = "lilicorr";
    EXPECT_THAT([&json] { parseDraftEngineConfig(writeJsonToTempFile(json)); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("version 2")));
}

TEST_F(LLMEngineConfigTest, UnknownProjectorTypeThrows)
{
    // Given a valid DFlash V2 selector draft config that additionally names an unknown projector_type,
    // When the draft engine config is parsed,
    // Then parsing is rejected instead of falling back to the selector projector.
    Json json = makeDFlash2SelectorDraftConfig();
    ASSERT_NO_THROW(parseDraftEngineConfig(writeJsonToTempFile(json)));

    json["dflash_config"]["projector_type"] = "lilicorr_v9";
    EXPECT_THROW(parseDraftEngineConfig(writeJsonToTempFile(json)), std::runtime_error);
}

TEST_F(LLMEngineConfigTest, DominoDraftKeepsItsDominoProjectorType)
{
    // Given the Domino draft config.json the exporters write: spec_decode_type 'domino' and a dflash_config with
    // version 1, projector_type 'domino' and the GRU correction-head contract
    // When the draft engine config is parsed
    // Then it is accepted as a Domino draft on dflash_config version 1 with its correction-head fields, and the DFlash
    // V2 projector stays at its selector default
    Json json = makeMinimalDraftConfig();
    json["spec_decode_type"] = "domino";
    json["dflash_config"] = {{"version", 1}, {"target_layer_ids", {1, 3, 5}}, {"block_size", 16},
        {"mask_token_id", 151}, {"supports_probabilistic_sampling", false}, {"projector_type", "domino"},
        {"gru_hidden_dim", 64}, {"proj_dim", 32}, {"prefix_len", 1}, {"pool_size", 8}, {"shift_label", true},
        {"heads_file", "domino_heads.safetensors"}};

    LLMEngineConfig const cfg = parseDraftEngineConfig(writeJsonToTempFile(json));

    EXPECT_EQ(cfg.specDecodeType, SpecDecodeMode::kDomino);
    EXPECT_EQ(cfg.dflashVersion, DFlashVersion::kV1);
    EXPECT_EQ(cfg.dflashProjector, DFlashProjector::kSelector);
    EXPECT_EQ(cfg.dominoGruHiddenDim, 64);
    EXPECT_EQ(cfg.dominoPoolSize, 8);
}

namespace
{

LLMEngineConfig makeLiLiCorrValidationConfig()
{
    LLMEngineConfig config;
    config.specDecodeType = SpecDecodeMode::kDFlash;
    config.isSpecDecodeBase = false;
    config.dflashVersion = DFlashVersion::kV2;
    config.dflashProjector = DFlashProjector::kLiLiCorr;
    config.specLiLiCorrTopK = 8;
    config.specDraftBlockSize = 8;
    config.vocabSize = 1000;
    config.hiddenSize = 64;
    // No attention layers keeps validation away from paged-KV profile queries that need a real ICudaEngine.
    config.numAttentionLayers = 0;
    return config;
}

std::set<std::string> const& liLiCorrDraftBindings()
{
    static std::set<std::string> const bindings = {
        binding_names::kContextSequenceCountCarrier,
        binding_names::kInputsEmbeds,
        binding_names::kDFlashTargetHiddenConcat,
        binding_names::kRopeCosSin,
        binding_names::kPositions,
        binding_names::kQueryStartOffsets,
        binding_names::kQueryLengths,
        binding_names::kPastLengths,
        binding_names::kAttentionSequenceLengths,
        binding_names::kStateIndices,
        binding_names::kExecutionPhaseMarker,
        binding_names::kKVPageTable,
        binding_names::kDFlashDeltaRopeCosSin,
        binding_names::kDFlashDeltaPositions,
        binding_names::kDFlashDeltaTokenToSequence,
        binding_names::kAttentionMask,
        binding_names::kAttentionPosId,
        binding_names::kSpecProposalSupportIds,
        binding_names::kSpecLiLiCorrTransitionScores,
        binding_names::kLiLiCorrEmbeddingTable,
    };
    return bindings;
}

//! Validates a LiLiCorr draft engine exposing `bindings`; returns the rejection message or "<accepted>".
std::string liLiCorrValidationResult(std::set<std::string> const& bindings,
    trt_edgellm::DataType scoresDtype = trt_edgellm::DataType::kFLOAT,
    trt_edgellm::DataType tableDtype = trt_edgellm::DataType::kHALF,
    nvinfer1::Dims tableShape = nvinfer1::Dims{2, {1000, 64}},
    nvinfer1::Dims supportShape = nvinfer1::Dims{3, {-1, 7, 8}},
    nvinfer1::Dims scoresShape = nvinfer1::Dims{4, {-1, 7, 8, 8}},
    trt_edgellm::DataType supportDtype = trt_edgellm::DataType::kINT32)
{
    using ::testing::_;
    using ::testing::Return;

    ::testing::NiceMock<substitute_engine::MockEngineExecutor> engine;
    ON_CALL(engine, hasIOTensor(_)).WillByDefault([&bindings](char const* name) { return bindings.count(name) > 0; });
    ON_CALL(engine, getBindingDataType(_)).WillByDefault(Return(trt_edgellm::DataType::kINT32));
    ON_CALL(engine, getBindingDataType(::testing::StrEq(binding_names::kSpecLiLiCorrTransitionScores)))
        .WillByDefault(Return(scoresDtype));
    ON_CALL(engine, getBindingDataType(::testing::StrEq(binding_names::kSpecProposalSupportIds)))
        .WillByDefault(Return(supportDtype));
    ON_CALL(engine, getTensorShape(::testing::StrEq(binding_names::kSpecProposalSupportIds)))
        .WillByDefault(Return(supportShape));
    ON_CALL(engine, getTensorShape(::testing::StrEq(binding_names::kSpecLiLiCorrTransitionScores)))
        .WillByDefault(Return(scoresShape));
    if (bindings.count(binding_names::kLiLiCorrEmbeddingTable) > 0)
    {
        ON_CALL(engine, getBindingDataType(::testing::StrEq(binding_names::kLiLiCorrEmbeddingTable)))
            .WillByDefault(Return(tableDtype));
        ON_CALL(engine, getTensorShape(::testing::StrEq(binding_names::kLiLiCorrEmbeddingTable)))
            .WillByDefault(Return(tableShape));
    }
    try
    {
        validateAgainstEngine(makeLiLiCorrValidationConfig(), engine, "draft");
    }
    catch (std::runtime_error const& error)
    {
        return error.what();
    }
    return "<accepted>";
}

} // namespace

TEST_F(LLMEngineConfigTest, LiLiCorrRejectsSelectorOutputs)
{
    // Given a LiLiCorr DFlash V2 draft config and a draft engine exposing the cached-draft inputs plus
    // spec_proposal_support_ids and an FP32 spec_lilicorr_transition_scores,
    // When the engine is validated against the config,
    // Then that binding set is accepted; additionally exposing logits, spec_proposal_unary_values or
    // spec_proposal_projected_hidden is rejected naming that binding; and missing either
    // spec_proposal_support_ids or spec_lilicorr_transition_scores is rejected naming that binding.
    using ::testing::HasSubstr;

    EXPECT_EQ(liLiCorrValidationResult(liLiCorrDraftBindings()), "<accepted>");

    for (char const* const forbidden :
        {binding_names::kLogits, binding_names::kSpecProposalUnaryValues, binding_names::kSpecProposalProjectedHidden})
    {
        std::set<std::string> bindings = liLiCorrDraftBindings();
        bindings.insert(forbidden);
        EXPECT_THAT(liLiCorrValidationResult(bindings), HasSubstr(forbidden)) << "extra binding " << forbidden;
    }

    for (char const* const required :
        {binding_names::kSpecProposalSupportIds, binding_names::kSpecLiLiCorrTransitionScores})
    {
        std::set<std::string> bindings = liLiCorrDraftBindings();
        bindings.erase(required);
        EXPECT_THAT(liLiCorrValidationResult(bindings), HasSubstr(required)) << "missing binding " << required;
    }
}

TEST_F(LLMEngineConfigTest, LiLiCorrRejectsNonFp32TransitionScores)
{
    // Given a LiLiCorr DFlash V2 draft config and a draft engine exposing the complete LiLiCorr binding set,
    // When spec_lilicorr_transition_scores is typed FP16 or INT32 instead of FP32,
    // Then validation rejects the engine naming spec_lilicorr_transition_scores and FP32.
    using ::testing::AllOf;
    using ::testing::HasSubstr;

    for (trt_edgellm::DataType const dtype : {trt_edgellm::DataType::kHALF, trt_edgellm::DataType::kINT32})
    {
        EXPECT_THAT(liLiCorrValidationResult(liLiCorrDraftBindings(), dtype),
            AllOf(HasSubstr(binding_names::kSpecLiLiCorrTransitionScores), HasSubstr("FP32")))
            << "dtype " << static_cast<int32_t>(dtype);
    }
}

TEST_F(LLMEngineConfigTest, LiLiCorrRequiresFp16EmbeddingTableInput)
{
    // Given a LiLiCorr DFlash V2 draft config and a draft engine exposing the complete LiLiCorr binding set,
    // When lilicorr_embedding_table is absent, or typed FP32 or INT8 instead of FP16,
    // Then validation rejects the engine naming lilicorr_embedding_table as missing, or as not FP16.
    using ::testing::AllOf;
    using ::testing::HasSubstr;

    std::set<std::string> bindings = liLiCorrDraftBindings();
    bindings.erase(binding_names::kLiLiCorrEmbeddingTable);
    EXPECT_THAT(liLiCorrValidationResult(bindings),
        AllOf(HasSubstr(binding_names::kLiLiCorrEmbeddingTable), HasSubstr("missing")));

    for (trt_edgellm::DataType const dtype : {trt_edgellm::DataType::kFLOAT, trt_edgellm::DataType::kINT8})
    {
        EXPECT_THAT(liLiCorrValidationResult(liLiCorrDraftBindings(), trt_edgellm::DataType::kFLOAT, dtype),
            AllOf(HasSubstr(binding_names::kLiLiCorrEmbeddingTable), HasSubstr("FP16")))
            << "dtype " << static_cast<int32_t>(dtype);
    }
}

TEST_F(LLMEngineConfigTest, LiLiCorrRequiresEmbeddingTableOfDraftVocabAndHidden)
{
    // Given a LiLiCorr DFlash V2 draft config with vocabulary 1000 and hidden size 64, and a draft engine exposing the
    // complete LiLiCorr binding set with an FP16 lilicorr_embedding_table,
    // When the table's static engine shape is [999, 64], [1000, 32] or [1000, 64, 1] instead of [1000, 64],
    // Then validation rejects the engine naming lilicorr_embedding_table and the expected [1000, 64].
    using ::testing::AllOf;
    using ::testing::HasSubstr;

    for (nvinfer1::Dims const& shape :
        {nvinfer1::Dims{2, {999, 64}}, nvinfer1::Dims{2, {1000, 32}}, nvinfer1::Dims{3, {1000, 64, 1}}})
    {
        EXPECT_THAT(liLiCorrValidationResult(
                        liLiCorrDraftBindings(), trt_edgellm::DataType::kFLOAT, trt_edgellm::DataType::kHALF, shape),
            AllOf(HasSubstr(binding_names::kLiLiCorrEmbeddingTable), HasSubstr("[1000, 64]")))
            << "shape [" << shape.d[0] << ", " << shape.d[1] << "] rank " << shape.nbDims;
    }
}

TEST_F(LLMEngineConfigTest, LiLiCorrRejectsOutputGeometryOtherThanConfig)
{
    // Given a LiLiCorr DFlash V2 draft config with block_size 8 and lilicorr_candidate_topk 8, and a draft engine
    // exposing the complete LiLiCorr binding set,
    // When spec_lilicorr_transition_scores is declared [-1, 7, 4, 4], [-1, 7, 16, 16] or [-1, 6, 8, 8], or
    // spec_proposal_support_ids is declared [-1, 7, 4] or typed FP32,
    // Then validation rejects the engine naming that binding, while [-1, 7, 8, 8] scores with [-1, 7, 8] or
    // [-1, -1, 8] support ids are accepted.
    using ::testing::HasSubstr;
    using trt_edgellm::DataType;
    using nvinfer1::Dims;
    Dims const table{2, {1000, 64}};

    EXPECT_EQ(liLiCorrValidationResult(
                  liLiCorrDraftBindings(), DataType::kFLOAT, DataType::kHALF, table, Dims{3, {-1, -1, 8}}),
        "<accepted>");
    for (Dims const& scores : {Dims{4, {-1, 7, 4, 4}}, Dims{4, {-1, 7, 16, 16}}, Dims{4, {-1, 6, 8, 8}}})
    {
        EXPECT_THAT(liLiCorrValidationResult(
                        liLiCorrDraftBindings(), DataType::kFLOAT, DataType::kHALF, table, Dims{3, {-1, 7, 8}}, scores),
            HasSubstr(binding_names::kSpecLiLiCorrTransitionScores))
            << "scores [" << scores.d[1] << ", " << scores.d[2] << ", " << scores.d[3] << "]";
    }
    EXPECT_THAT(liLiCorrValidationResult(
                    liLiCorrDraftBindings(), DataType::kFLOAT, DataType::kHALF, table, Dims{3, {-1, 7, 4}}),
        HasSubstr(binding_names::kSpecProposalSupportIds));
    EXPECT_THAT(liLiCorrValidationResult(liLiCorrDraftBindings(), DataType::kFLOAT, DataType::kHALF, table,
                    Dims{3, {-1, 7, 8}}, Dims{4, {-1, 7, 8, 8}}, DataType::kFLOAT),
        HasSubstr(binding_names::kSpecProposalSupportIds));
}
