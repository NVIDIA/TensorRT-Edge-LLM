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

#include "testUtils.h"
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

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
    config["sliding_window"] = -1;
    config["spec_decode_type"] = "none";
    config["engine_role"] = "llm";

    Json bc;
    bc["max_batch_size"] = 2;
    bc["max_input_len"] = 128;
    bc["max_kv_cache_capacity"] = 256;
    bc["max_lora_rank"] = 0;
    bc["spec_base"] = false;
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
    return writeJsonToTempFile(Json::parse(jsonStr));
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
    EXPECT_EQ(cfg.slidingWindowSize, std::optional<int32_t>{-1});
    EXPECT_EQ(cfg.maxSupportedLoraRank, 0);
    EXPECT_FALSE(cfg.isSpecDecodeBase);
    EXPECT_EQ(cfg.maxVerifyTreeSize, 0);
    EXPECT_EQ(cfg.maxDraftTreeSize, 0);
    EXPECT_EQ(cfg.kvCacheDtype, nvinfer1::DataType::kHALF);
}

TEST_F(LLMEngineConfigTest, SlidingWindowSizeIsParsed)
{
    Json json = makeMinimalConfig();
    json["sliding_window"] = 4096;
    auto const path = writeJsonToTempFile(json);

    EXPECT_EQ(parseEngineConfig(path).slidingWindowSize, std::optional<int32_t>{4096});
}

TEST_F(LLMEngineConfigTest, MissingSlidingWindowMetadataRemainsUnknown)
{
    Json json = makeMinimalConfig();
    json.erase("sliding_window");
    auto const path = writeJsonToTempFile(json);

    EXPECT_FALSE(parseEngineConfig(path).slidingWindowSize.has_value());
}

TEST_F(LLMEngineConfigTest, InvalidSlidingWindowSizeThrows)
{
    Json json = makeMinimalConfig();
    json["sliding_window"] = 0;
    auto const path = writeJsonToTempFile(json);

    EXPECT_THROW(parseEngineConfig(path), std::runtime_error);
}

TEST_F(LLMEngineConfigTest, KVPoolPagesDefaultsToActiveCapacityFloor)
{
    auto const path = writeJsonToTempFile(makeMinimalConfig());

    EXPECT_EQ(parseEngineConfig(path).kvPoolPages, 4);
}

TEST_F(LLMEngineConfigTest, KVPoolPagesUsesSerializedOverride)
{
    Json json = makeMinimalConfig();
    json["builder_config"]["max_kv_pool_pages"] = 9;
    auto const path = writeJsonToTempFile(json);

    EXPECT_EQ(parseEngineConfig(path).kvPoolPages, 9);
}

TEST_F(LLMEngineConfigTest, KVPoolPagesBelowActiveCapacityFloorThrows)
{
    Json json = makeMinimalConfig();
    json["builder_config"]["max_kv_pool_pages"] = 3;
    auto const path = writeJsonToTempFile(json);

    EXPECT_THROW(parseEngineConfig(path), std::runtime_error);
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
    // `max_draft_tree_size` is a draft-engine property and is not written
    // into base_config.json by the builder — intentionally omitted here.
    auto const path = writeJsonToTempFile(json);

    LLMEngineConfig cfg = parseEngineConfig(path);
    EXPECT_TRUE(cfg.isSpecDecodeBase);
    EXPECT_EQ(cfg.maxVerifyTreeSize, 16);
    EXPECT_EQ(cfg.maxDraftTreeSize, 0); // Base side leaves this at the default.
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
            "max_kv_cache_capacity": 128
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
    EXPECT_EQ(cfg.kvLayerConfigs[1].numKVHeads, 4);
    EXPECT_EQ(cfg.kvLayerConfigs[1].headDim, 128);
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
            "max_kv_cache_capacity": 128
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
            "max_kv_cache_capacity": 128
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
    auto const d = cfg.prefillDims(/*batch=*/2, /*seqLen=*/128, /*kvCacheAllEmpty=*/true);
    EXPECT_EQ(d.batch, 2);
    EXPECT_EQ(d.seqLen, 128);
    EXPECT_EQ(d.kvLen, 4096);
    EXPECT_EQ(d.selectLen, 1);
    EXPECT_EQ(d.attnMaskSeqLen, 1); // dummy attention shape during prefill
    EXPECT_EQ(d.ropeBatch, 1);      // non-MRope
    EXPECT_EQ(d.packedMaskLen, 1);  // pinned to 1 alongside attnMaskSeqLen
    EXPECT_EQ(d.startIndexLen, 0);  // plugin-path empty-cache sentinel
    EXPECT_EQ(d.specVerifyPhaseLen, 0);
}

TEST(LLMEngineConfigRecipesTest, PrefillDimsMRope)
{
    auto const cfg = makeRecipeConfig(/*maxKV=*/4096, /*mrope=*/true);
    auto const d = cfg.prefillDims(/*batch=*/3, /*seqLen=*/65, /*kvCacheAllEmpty=*/true);
    EXPECT_EQ(d.ropeBatch, 3);      // MRope → batch
    EXPECT_EQ(d.attnMaskSeqLen, 1); // dummy attention shape during prefill
    EXPECT_EQ(d.packedMaskLen, 1);  // pinned to 1 alongside attnMaskSeqLen
    EXPECT_EQ(d.startIndexLen, 0);  // plugin-path empty-cache sentinel
    EXPECT_EQ(d.specVerifyPhaseLen, 0);
}

TEST(LLMEngineConfigRecipesTest, PrefillDimsChunked)
{
    // Chunked prefill (cache non-empty) uses [batch] startIndexLen.
    auto const cfg = makeRecipeConfig(/*maxKV=*/4096, /*mrope=*/false);
    auto const d = cfg.prefillDims(/*batch=*/2, /*seqLen=*/128, /*kvCacheAllEmpty=*/false);
    EXPECT_EQ(d.startIndexLen, 2);
    EXPECT_EQ(d.specVerifyPhaseLen, 0);
}

TEST(LLMEngineConfigRecipesTest, DecodeDims)
{
    auto const cfg = makeRecipeConfig(/*maxKV=*/2048, /*mrope=*/false);
    auto const d = cfg.decodeDims(/*batch=*/4);
    EXPECT_EQ(d.batch, 4);
    EXPECT_EQ(d.seqLen, 1);
    EXPECT_EQ(d.kvLen, 2048);
    EXPECT_EQ(d.selectLen, 1);
    EXPECT_EQ(d.attnMaskSeqLen, 1);
    EXPECT_EQ(d.ropeBatch, 1);
    EXPECT_EQ(d.packedMaskLen, 1); // explicit 1 in decode
    EXPECT_EQ(d.startIndexLen, 4); // batch
    EXPECT_EQ(d.specVerifyPhaseLen, 0);
}

TEST(LLMEngineConfigRecipesTest, DecodeDimsMRope)
{
    auto const cfg = makeRecipeConfig(/*maxKV=*/2048, /*mrope=*/true);
    auto const d = cfg.decodeDims(/*batch=*/4);
    EXPECT_EQ(d.ropeBatch, 4); // MRope → batch
}

TEST(LLMEngineConfigRecipesTest, SpecVerifyDimsIsOnlyRecipeWithSelectLenNeq1)
{
    auto const cfg = makeRecipeConfig(/*maxKV=*/8192, /*mrope=*/false);
    auto const d = cfg.specVerifyDims(/*batch=*/1, /*verifySize=*/8);
    EXPECT_EQ(d.batch, 1);
    EXPECT_EQ(d.seqLen, 8);
    EXPECT_EQ(d.kvLen, 8192);
    EXPECT_EQ(d.selectLen, 8);      // verifySize — unique to this recipe
    EXPECT_EQ(d.attnMaskSeqLen, 8); // verifySize — proposal attention shape
    EXPECT_EQ(d.ropeBatch, 1);
    EXPECT_EQ(d.packedMaskLen, 1); // divUp(8, 32) = 1
    EXPECT_EQ(d.startIndexLen, 1); // batch (cache non-empty during verify)
    EXPECT_EQ(d.specVerifyPhaseLen, 1);
}

TEST(LLMEngineConfigRecipesTest, ShortPrefillDoesNotSetSpecVerifyPhaseMarker)
{
    auto const cfg = makeRecipeConfig(/*maxKV=*/8192, /*mrope=*/false);
    for (int32_t seqLen = 2; seqLen <= 16; ++seqLen)
    {
        auto const d = cfg.prefillDims(/*batch=*/2, seqLen, /*kvCacheAllEmpty=*/true);
        EXPECT_EQ(d.seqLen, seqLen);
        EXPECT_EQ(d.specVerifyPhaseLen, 0);
    }
    auto const verifyDims = cfg.specVerifyDims(/*batch=*/2, /*verifySize=*/16);
    EXPECT_EQ(verifyDims.specVerifyPhaseLen, 1);
}

TEST(LLMEngineConfigRecipesTest, ProposalDims)
{
    auto const cfg = makeRecipeConfig(/*maxKV=*/4096, /*mrope=*/false);
    auto const d = cfg.proposalDims(/*batch=*/2, /*paddedTreeSize=*/16, /*draftTopK=*/4);
    EXPECT_EQ(d.seqLen, 16);
    EXPECT_EQ(d.selectLen, 4);       // draftTopK — one per tree branch
    EXPECT_EQ(d.attnMaskSeqLen, 16); // paddedTreeSize — tree attention shape
    EXPECT_EQ(d.packedMaskLen, 1);   // divUp(16, 32) = 1
    EXPECT_EQ(d.startIndexLen, 2);   // batch
    EXPECT_EQ(d.specVerifyPhaseLen, 0);
}

TEST(LLMEngineConfigRecipesTest, AcceptDims)
{
    auto const cfg = makeRecipeConfig(/*maxKV=*/4096, /*mrope=*/false);
    auto const d = cfg.acceptDims(/*batch=*/3, /*acceptLen=*/33);
    EXPECT_EQ(d.batch, 3);
    EXPECT_EQ(d.seqLen, 33);
    EXPECT_EQ(d.selectLen, 1);
    EXPECT_EQ(d.attnMaskSeqLen, 33); // acceptLen — tree attention shape
    EXPECT_EQ(d.packedMaskLen, 2);   // divUp(33, 32) = 2 — exercises the boundary
    EXPECT_EQ(d.startIndexLen, 3);   // batch
    EXPECT_EQ(d.specVerifyPhaseLen, 0);
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
        EXPECT_EQ(d.specVerifyPhaseLen, 0);
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
