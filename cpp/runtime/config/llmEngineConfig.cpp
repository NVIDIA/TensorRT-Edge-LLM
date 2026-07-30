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

#include "common/bindingNames.h"
#include "common/checkMacros.h"
#include "common/cudaUtils.h"
#include "common/logger.h"
#include "common/pagedKvTypes.h"
#include "common/ropeUtils.h"
#include "common/trtUtils.h"
#include "common/version.h"
#include "runtime/exec/engineExecutor.h"

#include <algorithm>
#include <fstream>
#include <nlohmann/json.hpp>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace trt_edgellm
{
namespace rt
{

using Json = nlohmann::json;

namespace
{

//! Helper: read a required field or throw.
template <typename T>
T getRequired(Json const& json, char const* key)
{
    ELLM_CHECK(json.contains(key), std::string("parseEngineConfig: missing required field '") + key + "'");
    return json[key].get<T>();
}

//! Helper: validate that a value is positive.
void requirePositive(int32_t value, char const* fieldName)
{
    ELLM_CHECK(value > 0,
        std::string("parseEngineConfig: invalid ") + fieldName + ": " + std::to_string(value) + " (must be positive)");
}

//! Parse a lowercase dtype token (e.g. from `kv_cache_dtype`, `recurrent_state_dtype`,
//! `conv_state_dtype`) into an `nvinfer1::DataType`.
//!
//! Accepted tokens and their mapping (matches the export pipeline's writer):
//!   "fp16" → kHALF, "fp8" → kFP8, "int8" → kINT8, "bf16" → kBF16, "fp32" → kFLOAT.
//! `fp32` is required for Qwen3.5 GDN recurrent states (the architecture's SSM
//! kernels run in FP32 for numerical stability), so the export writes
//! `recurrent_state_dtype="fp32"` for that model family.
//! Anything else throws with the offending value, the originating field name,
//! and the allowed set.
nvinfer1::DataType parseStateDtype(std::string const& token, char const* fieldName)
{
    if (token == "fp16")
    {
        return nvinfer1::DataType::kHALF;
    }
    if (token == "fp8")
    {
        return nvinfer1::DataType::kFP8;
    }
    if (token == "int8")
    {
        return nvinfer1::DataType::kINT8;
    }
    if (token == "bf16")
    {
        return nvinfer1::DataType::kBF16;
    }
    if (token == "fp32")
    {
        return nvinfer1::DataType::kFLOAT;
    }
    throw std::runtime_error(std::string("parseEngineConfig: invalid ") + fieldName + " '" + token
        + "'. Allowed values: fp16, fp8, int8, bf16, fp32. Re-export the engine with a supported value.");
}

//! Parse a required dtype field. Throws if the field is missing — all current
//! exports (>= 0.7.0) write dtype fields explicitly to config.json.
void parseRequiredStateDtype(Json const& json, char const* key, nvinfer1::DataType& out)
{
    ELLM_CHECK(json.contains(key),
        std::string("parseEngineConfig: config.json missing required field '") + key
            + "'. Re-export the model with the latest llm_export.py to record it.");
    out = parseStateDtype(json[key].get<std::string>(), key);
}

SpecDecodeMode parseSpecDecodeMode(Json const& configJson)
{
    std::string const specDecodeType = configJson.value("spec_decode_type", "none");
    if (specDecodeType == "none")
    {
        return SpecDecodeMode::kNONE;
    }
    if (specDecodeType == "mtp")
    {
        return SpecDecodeMode::kMTP;
    }
    if (specDecodeType == "eagle3")
    {
        return SpecDecodeMode::kEAGLE;
    }
    if (specDecodeType == "dflash")
    {
        return SpecDecodeMode::kDFlash;
    }
    if (specDecodeType == "gemma4_mtp")
    {
        return SpecDecodeMode::kGemma4MTP;
    }
    throw std::runtime_error("parseEngineConfig: invalid spec_decode_type '" + specDecodeType
        + "'. Allowed values: none, mtp, eagle3, dflash, gemma4_mtp.");
}

std::string parseEngineRole(Json const& configJson)
{
    std::string const engineRole = configJson.value("engine_role", "llm");
    if (engineRole == "llm" || engineRole == "base" || engineRole == "draft")
    {
        return engineRole;
    }
    throw std::runtime_error(
        "parseEngineConfig: invalid engine_role '" + engineRole + "'. Allowed values: llm, base, draft.");
}

void validateDFlashTargetLayerIds(
    std::vector<int32_t> const& targetLayerIds, int32_t numDecoderLayers, char const* layerCountOwner)
{
    for (int32_t layerId : targetLayerIds)
    {
        ELLM_CHECK(layerId >= 0 && layerId < numDecoderLayers,
            "parseEngineConfig: DFlash target layer id " + std::to_string(layerId) + " is outside [0, "
                + layerCountOwner + ".num_hidden_layers).");
    }
}

void parseDFlashFields(
    Json const& configJson, LLMEngineConfig& cfg, std::optional<int32_t> targetLayerValidationUpperBound = std::nullopt)
{
    if (cfg.specDecodeType != SpecDecodeMode::kDFlash)
    {
        return;
    }

    Json const empty = Json::object();
    Json const& dflashConfig = configJson.contains("dflash_config") ? configJson["dflash_config"] : empty;

    cfg.dflashBlockSize = dflashConfig.value("block_size", configJson.value("block_size", 16));
    cfg.dflashMaskTokenId = dflashConfig.value("mask_token_id", configJson.value("dflash_mask_token_id", 248070));
    ELLM_CHECK(cfg.dflashBlockSize > 0,
        "parseEngineConfig: invalid DFlash block_size: " + std::to_string(cfg.dflashBlockSize) + " (must be positive)");
    ELLM_CHECK(cfg.dflashMaskTokenId >= 0,
        "parseEngineConfig: invalid DFlash mask_token_id: " + std::to_string(cfg.dflashMaskTokenId)
            + " (must be non-negative)");

    if (dflashConfig.contains("target_layer_ids"))
    {
        ELLM_CHECK(dflashConfig["target_layer_ids"].is_array(),
            "parseEngineConfig: dflash_config.target_layer_ids must be an array");
        for (auto const& id : dflashConfig["target_layer_ids"])
        {
            cfg.dflashTargetLayerIds.push_back(id.get<int32_t>());
        }
    }
    else if (configJson.contains("dflash_target_layer_ids"))
    {
        ELLM_CHECK(configJson["dflash_target_layer_ids"].is_array(),
            "parseEngineConfig: dflash_target_layer_ids must be an array");
        for (auto const& id : configJson["dflash_target_layer_ids"])
        {
            cfg.dflashTargetLayerIds.push_back(id.get<int32_t>());
        }
    }

    if (targetLayerValidationUpperBound.has_value())
    {
        validateDFlashTargetLayerIds(cfg.dflashTargetLayerIds, *targetLayerValidationUpperBound, "base");
    }
}

void parseGemma4MTPFields(Json const& configJson, LLMEngineConfig& cfg)
{
    cfg.modelType = configJson.value("model", configJson.value("model_type", std::string{}));
    cfg.baseModelHiddenSize = configJson.value("base_model_hidden_size", cfg.baseModelHiddenSize);

    if (cfg.specDecodeType != SpecDecodeMode::kGemma4MTP)
    {
        return;
    }

    cfg.sharesTargetKV = configJson.value("shares_target_kv", false);
    cfg.hasOwnKVCache = configJson.value("has_own_kv_cache", true);
    cfg.constantDraftPositions = configJson.value("constant_draft_positions", false);
    cfg.returnsFeedbackHidden = configJson.value("returns_feedback_hidden", false);
    cfg.assistantHiddenSize = configJson.value("assistant_hidden_size", 0);

    if (!configJson.contains("kv_sharing_map"))
    {
        return;
    }

    ELLM_CHECK(
        configJson["kv_sharing_map"].is_array(), "parseEngineConfig: gemma4_mtp kv_sharing_map must be an array.");
    for (auto const& entryJson : configJson["kv_sharing_map"])
    {
        ELLM_CHECK(entryJson.is_object(),
            "parseEngineConfig: gemma4_mtp kv_sharing_map entries must be objects written by export.");
        ELLM_CHECK(entryJson.contains("assistant_layer") && entryJson["assistant_layer"].is_number_integer(),
            "parseEngineConfig: gemma4_mtp kv_sharing_map entry must contain integer assistant_layer.");
        ELLM_CHECK(
            entryJson.contains("target_attention_layer") && entryJson["target_attention_layer"].is_number_integer(),
            "parseEngineConfig: gemma4_mtp kv_sharing_map entry must contain integer target_attention_layer.");

        Gemma4MTPKVSharingEntry entry;
        entry.assistantLayerIdx = entryJson["assistant_layer"].get<int32_t>();
        entry.targetAttentionLayerIdx = entryJson["target_attention_layer"].get<int32_t>();
        cfg.gemma4MTPKVSharingMap.push_back(std::move(entry));
    }
}

bool isDFlashDraftConfig(LLMEngineConfig const& config)
{
    return config.specDecodeType == SpecDecodeMode::kDFlash && !config.isSpecDecodeBase;
}

bool isGemma4MTPDraftConfig(LLMEngineConfig const& config)
{
    return config.specDecodeType == SpecDecodeMode::kGemma4MTP && !config.isSpecDecodeBase;
}

//! Helper: parse explicit sliding/full RoPE config blocks when present.
void parseDualRopeFields(Json const& configJson, LLMEngineConfig& cfg)
{
    if (!configJson.contains("sliding_rope_config") || !configJson["sliding_rope_config"].is_object()
        || !configJson.contains("full_rope_config") || !configJson["full_rope_config"].is_object())
    {
        return;
    }

    auto parseRopeBlock = [&](char const* key, char const* rotaryDimName, RopeConfig& ropeConfig, int32_t& rotaryDim,
                              int32_t headDim) {
        Json ropeJson = configJson.at(key);
        if (!ropeJson.contains("max_position_embeddings") && configJson.contains("max_position_embeddings"))
        {
            ropeJson["max_position_embeddings"] = configJson["max_position_embeddings"];
        }
        // Do not promote original_max_position_embeddings here: it is LongRope-only,
        // and dual RoPE cache binding does not support LongRope.
        ropeConfig = collectRopeConfig(ropeJson);
        if (configJson.contains(rotaryDimName))
        {
            rotaryDim = configJson.at(rotaryDimName).get<int32_t>();
        }
        else
        {
            rotaryDim = static_cast<int32_t>(getRotaryDim(ropeJson, headDim));
        }
        requirePositive(rotaryDim, rotaryDimName);
        ELLM_CHECK(rotaryDim <= headDim,
            std::string("parseEngineConfig: invalid ") + rotaryDimName + ": " + std::to_string(rotaryDim)
                + " exceeds head_dim " + std::to_string(headDim));
        ELLM_CHECK(ropeConfig.type != RopeType::kMRope,
            std::string("parseEngineConfig: dual RoPE does not support context-dependent MRoPE bindings: ") + key);
    };

    cfg.useDualRope = true;
    int32_t const fullHeadDim = configJson.value("global_head_dim", cfg.headDim);
    parseRopeBlock(
        "sliding_rope_config", "sliding_rotary_dim", cfg.slidingRopeConfig, cfg.slidingRotaryDim, cfg.headDim);
    parseRopeBlock("full_rope_config", "full_rotary_dim", cfg.fullRopeConfig, cfg.fullRotaryDim, fullHeadDim);
}

bool engineHasTensorWithPrefix(EngineExecutor const& executor, std::string const& prefix)
{
    int32_t const numIOTensors = executor.getNumIOTensors();
    for (int32_t i = 0; i < numIOTensors; ++i)
    {
        std::string const name = executor.getIOTensorName(i);
        if (name.rfind(prefix, 0) == 0)
        {
            return true;
        }
    }
    return false;
}

//! Fields shared by base and draft engines. Parses top-level model dims and
//! dtypes (kv_cache_dtype lives at top level — it is a property of the
//! exported weights written by the Python export step, not a builder knob;
//! the C++ builder never emits it inside builder_config). Also parses the
//! common builder_config batch/input/kv limits and RoPE configuration.
//! Engine-type-specific fields (reduced vocab, hybrid state, SpecDecode
//! capacities, LoRA) are filled in by the calling parser
//! around this helper.
void parseCoreFields(Json const& configJson, LLMEngineConfig& cfg)
{
    // Top-level model fields.
    cfg.numDecoderLayers = getRequired<int32_t>(configJson, "num_hidden_layers");
    cfg.numKVHeads = getRequired<int32_t>(configJson, "num_key_value_heads");
    cfg.headDim = getRequired<int32_t>(configJson, "head_dim");
    cfg.hiddenSize = getRequired<int32_t>(configJson, "hidden_size");

    // Top-level: kv_cache_dtype. Required — all current exports write this.
    parseRequiredStateDtype(configJson, "kv_cache_dtype", cfg.kvCacheDtype);

    // builder_config: required batch / input / kv limits.
    ELLM_CHECK(configJson.contains("builder_config"), "parseEngineConfig: missing required 'builder_config' section");
    auto const& bc = configJson["builder_config"];
    cfg.maxSupportedBatchSize = getRequired<int32_t>(bc, "max_batch_size");
    cfg.maxSupportedInputLength = getRequired<int32_t>(bc, "max_input_len");
    cfg.maxKVCacheCapacity = getRequired<int32_t>(bc, "max_kv_cache_capacity");
    int32_t const kvPoolFloorPages = computeKvPoolFloorPages(cfg.maxSupportedBatchSize, cfg.maxKVCacheCapacity);
    cfg.kvPoolPages = bc.value("max_kv_pool_pages", kvPoolFloorPages);

    // RoPE configuration (top-level, derived from full config).
    cfg.ropeConfig = collectRopeConfig(configJson);
    if (configJson.contains("sliding_window"))
    {
        cfg.slidingWindowSize = configJson.at("sliding_window").get<int32_t>();
    }

    // Positivity checks on required fields (common to both parsers).
    requirePositive(cfg.numDecoderLayers, "num_hidden_layers");
    requirePositive(cfg.numKVHeads, "num_key_value_heads");
    requirePositive(cfg.headDim, "head_dim");
    requirePositive(cfg.hiddenSize, "hidden_size");
    requirePositive(cfg.maxSupportedBatchSize, "max_batch_size");
    requirePositive(cfg.maxSupportedInputLength, "max_input_len");
    requirePositive(cfg.maxKVCacheCapacity, "max_kv_cache_capacity");
    ELLM_CHECK(!cfg.slidingWindowSize.has_value() || *cfg.slidingWindowSize == -1 || *cfg.slidingWindowSize > 0,
        "parseEngineConfig: sliding_window must be -1 or a positive token count");
    ELLM_CHECK(cfg.kvPoolPages >= kvPoolFloorPages,
        "parseEngineConfig: max_kv_pool_pages (" + std::to_string(cfg.kvPoolPages)
            + ") cannot be smaller than the active-capacity floor (" + std::to_string(kvPoolFloorPages) + ")");
    ELLM_CHECK(cfg.maxSupportedInputLength <= cfg.maxKVCacheCapacity,
        "parseEngineConfig: max_input_len (" + std::to_string(cfg.maxSupportedInputLength)
            + ") cannot be greater than max_kv_cache_capacity (" + std::to_string(cfg.maxKVCacheCapacity) + ")");
}

//! Populate `cfg.layerTypes` and `cfg.kvLayerConfigs` from the canonical
//! `layer_types` + `kv_layer_configs` fields when present, or fall back to
//! scalar-broadcast ordering for legacy engines.
//!
//! Canonical path: `kv_layer_configs` present in config.json.
//!   - Both `layer_types` and `kv_layer_configs` must be present and same length.
//!   - Each "attention" entry appends kAttention to layerTypes and the
//!     corresponding `{num_kv_heads, head_dim}` object to kvLayerConfigs.
//!   - Each "mamba" entry appends kMamba to layerTypes (kvLayerConfigs unchanged).
//!
//! Fallback path: `kv_layer_configs` absent.
//!   - If `numLinearAttnLayers > 0` (hybrid): attention layers first, then mamba.
//!   - Otherwise (pure attention): all layers are kAttention with uniform dims.
void populateLayerTypes(Json const& configJson, LLMEngineConfig& cfg)
{
    if (configJson.contains("kv_layer_configs"))
    {
        ELLM_CHECK(configJson.contains("layer_types"),
            "parseEngineConfig: kv_layer_configs requires layer_types in config.json");
        auto const& layerTypesJson = configJson["layer_types"];
        auto const& kvLayerConfigsJson = configJson["kv_layer_configs"];
        ELLM_CHECK(layerTypesJson.size() == kvLayerConfigsJson.size(),
            "parseEngineConfig: layer_types (" + std::to_string(layerTypesJson.size()) + ") and kv_layer_configs ("
                + std::to_string(kvLayerConfigsJson.size()) + ") must have equal length.");
        int32_t const n = static_cast<int32_t>(layerTypesJson.size());
        for (int32_t i = 0; i < n; ++i)
        {
            std::string const typeStr = layerTypesJson[i].get<std::string>();
            if (typeStr == "attention")
            {
                cfg.layerTypes.push_back(HybridCacheManager::LayerType::kAttention);
                auto const& lc = kvLayerConfigsJson[i];
                ELLM_CHECK(!lc.is_null() && lc.contains("num_kv_heads") && lc.contains("head_dim"),
                    "parseEngineConfig: kv_layer_configs[" + std::to_string(i)
                        + "] missing num_kv_heads/head_dim for attention layer");
                cfg.kvLayerConfigs.push_back({lc["num_kv_heads"].get<int32_t>(), lc["head_dim"].get<int32_t>()});
            }
            else if (typeStr == "mamba")
            {
                // Validate both directions (matches legacy/llmEngineRunner.cpp):
                // mamba entries must be null; a non-null object here would be
                // silently discarded, masking a malformed config.
                ELLM_CHECK(kvLayerConfigsJson[i].is_null(),
                    "parseEngineConfig: kv_layer_configs[" + std::to_string(i) + "] must be null for a mamba layer");
                cfg.layerTypes.push_back(HybridCacheManager::LayerType::kMamba);
            }
            else
            {
                ELLM_CHECK(false, "parseEngineConfig: unknown layer type '" + typeStr + "'");
            }
        }
    }
    else
    {
        if (configJson.contains("layer_types"))
        {
            auto const& layerTypesJson = configJson["layer_types"];
            if (layerTypesJson.is_array() && !layerTypesJson.empty())
            {
                bool const allGemma4AttentionTypes
                    = std::all_of(layerTypesJson.begin(), layerTypesJson.end(), [](Json const& layerTypeJson) {
                          if (!layerTypeJson.is_string())
                          {
                              return false;
                          }
                          std::string const layerType = layerTypeJson.get<std::string>();
                          return layerType == "sliding_attention" || layerType == "full_attention";
                      });
                if (allGemma4AttentionTypes)
                {
                    int32_t const globalHeadDim = configJson.value("global_head_dim", cfg.headDim);
                    requirePositive(globalHeadDim, "global_head_dim");
                    for (auto const& layerTypeJson : layerTypesJson)
                    {
                        std::string const layerType = layerTypeJson.get<std::string>();
                        cfg.layerTypes.push_back(HybridCacheManager::LayerType::kAttention);
                        int32_t const layerHeadDim = (layerType == "full_attention") ? globalHeadDim : cfg.headDim;
                        cfg.kvLayerConfigs.push_back({cfg.numKVHeads, layerHeadDim});
                    }
                    return;
                }
            }
        }

        // Fallback: attention-first then mamba, matching main's
        // llmEngineRunner.cpp scalar-broadcast ordering.
        if (cfg.numLinearAttnLayers > 0)
        {
            for (int32_t i = 0; i < cfg.numAttentionLayers; ++i)
            {
                cfg.layerTypes.push_back(HybridCacheManager::LayerType::kAttention);
                cfg.kvLayerConfigs.push_back({cfg.numKVHeads, cfg.headDim});
            }
            for (int32_t i = 0; i < cfg.numLinearAttnLayers; ++i)
            {
                cfg.layerTypes.push_back(HybridCacheManager::LayerType::kMamba);
            }
        }
        else
        {
            for (int32_t i = 0; i < cfg.numDecoderLayers; ++i)
            {
                cfg.layerTypes.push_back(HybridCacheManager::LayerType::kAttention);
                cfg.kvLayerConfigs.push_back({cfg.numKVHeads, cfg.headDim});
            }
        }
    }
}

} // namespace

LLMEngineConfig parseEngineConfig(std::filesystem::path const& configPath)
{
    LOG_INFO("reading %s", configPath.string().c_str());

    std::ifstream ifs(configPath);
    ELLM_CHECK(ifs.is_open(), "parseEngineConfig: failed to open " + configPath.string());

    Json configJson;
    try
    {
        configJson = Json::parse(ifs);
    }
    catch (Json::parse_error const& e)
    {
        throw std::runtime_error("parseEngineConfig: JSON parse error in " + configPath.string() + ": " + e.what());
    }
    ifs.close();

    // Optional version check.
    std::string const modelVersion = configJson.value(binding_names::kEdgellmVersion, "");
    version::checkVersion(modelVersion);

    LLMEngineConfig cfg;

    cfg.specDecodeType = parseSpecDecodeMode(configJson);
    std::string const engineRole = parseEngineRole(configJson);
    ELLM_CHECK(engineRole != "draft", "parseEngineConfig: use parseDraftEngineConfig for engine_role=draft.");
    cfg.isSpecDecodeBase = (engineRole == "base");
    if (cfg.isSpecDecodeBase)
    {
        ELLM_CHECK(cfg.specDecodeType != SpecDecodeMode::kNONE,
            "parseEngineConfig: engine_role=base requires spec_decode_type to be mtp, eagle3, dflash, or gemma4_mtp.");
    }
    else
    {
        ELLM_CHECK(cfg.specDecodeType == SpecDecodeMode::kNONE,
            "parseEngineConfig: engine_role=llm requires spec_decode_type=none.");
    }

    // Shared core fields (layers, kv heads, head_dim, hidden_size, kv_cache_dtype,
    // batch/input/kv limits, RoPE, common positivity checks).
    parseCoreFields(configJson, cfg);
    parseGemma4MTPFields(configJson, cfg);

    // --- Base-specific: vocab, rotary dim, deepstack / multimodal, hybrid ---
    cfg.vocabSize = getRequired<int32_t>(configJson, "vocab_size");
    cfg.rotaryDim = static_cast<int32_t>(getRotaryDim(configJson, cfg.headDim));

    cfg.reducedVocabSize = configJson.value(binding_names::kReducedVocabSizeKey, 0);
    cfg.outputVocabSize = (cfg.reducedVocabSize > 0) ? cfg.reducedVocabSize : cfg.vocabSize;

    cfg.numDeepstackFeatures = configJson.value("num_deepstack_features", 0);
    cfg.pleEnabled = configJson.value("ple_enabled", false);
    cfg.numPleInputs = configJson.value("num_ple_inputs", 0);
    cfg.pleHiddenSize = configJson.value("ple_hidden_size", 0);
    cfg.audioTokenId = configJson.value("audio_token_id", -1);
    cfg.imageTokenId = configJson.value("image_token_id", -1);
    cfg.useVisionBidirectionalAttention = configJson.value("use_vision_bidirectional_attention", false);
    ELLM_CHECK(!cfg.useVisionBidirectionalAttention || cfg.imageTokenId >= 0,
        "use_vision_bidirectional_attention requires image_token_id in the LLM config.");

    cfg.numLinearAttnLayers = configJson.value("num_linear_attn_layers", 0);
    cfg.numAttentionLayers = configJson.value("num_attention_layers", cfg.numDecoderLayers);
    cfg.recurrentStateNumHeads = configJson.value("recurrent_state_num_heads", 0);
    cfg.recurrentStateHeadDim = configJson.value("recurrent_state_head_dim", 0);
    cfg.recurrentStateSize = configJson.value("recurrent_state_size", 0);
    cfg.convDim = configJson.value("conv_dim", 0);
    cfg.convKernel = configJson.value("conv_kernel", 0);
    parseDFlashFields(configJson, cfg, cfg.numDecoderLayers);

    auto const& bc = configJson["builder_config"];
    cfg.maxSupportedLoraRank = bc.value("max_lora_rank", 0);

    // Recurrent / conv state dtypes are only meaningful for hybrid engines
    // (Mamba / Nemotron-H / GDN). Mirror the Python export gating exactly:
    // present and parsed strictly iff `num_linear_attn_layers > 0`. Same
    // reasoning as kv_cache_dtype above — these are weight-export properties
    // written to the top level by the Python export step.
    if (cfg.numLinearAttnLayers > 0)
    {
        parseRequiredStateDtype(configJson, "recurrent_state_dtype", cfg.recurrentStateDtype);
        parseRequiredStateDtype(configJson, "conv_state_dtype", cfg.convStateDtype);
    }

    // Base-specific positivity checks (beyond parseCoreFields's core set).
    requirePositive(cfg.rotaryDim, "rotary_dim");
    requirePositive(cfg.vocabSize, "vocab_size");
    ELLM_CHECK(cfg.numPleInputs >= 0,
        "parseEngineConfig: invalid num_ple_inputs: " + std::to_string(cfg.numPleInputs) + " (must be non-negative)");
    ELLM_CHECK(cfg.pleHiddenSize >= 0,
        "parseEngineConfig: invalid ple_hidden_size: " + std::to_string(cfg.pleHiddenSize) + " (must be non-negative)");
    if (cfg.pleEnabled)
    {
        requirePositive(cfg.numPleInputs, "num_ple_inputs");
        requirePositive(cfg.pleHiddenSize, "ple_hidden_size");
        ELLM_CHECK(cfg.numPleInputs == cfg.numDecoderLayers,
            "parseEngineConfig: Gemma4 PLE expects num_ple_inputs to match num_hidden_layers; got "
                + std::to_string(cfg.numPleInputs) + " and " + std::to_string(cfg.numDecoderLayers));
    }
    else
    {
        ELLM_CHECK(cfg.numPleInputs == 0 && cfg.pleHiddenSize == 0,
            "parseEngineConfig: ple_enabled is false but num_ple_inputs or ple_hidden_size is non-zero");
    }
    ELLM_CHECK(cfg.maxSupportedLoraRank >= 0,
        "parseEngineConfig: invalid max_lora_rank: " + std::to_string(cfg.maxSupportedLoraRank)
            + " (must be non-negative)");

    // --- SpecDecode base model ---
    // `max_verify_tree_size` is the base engine's own input budget (the
    // seq_len it accepts for proposal verification), so it must be present
    // in the base's builder_config. `max_draft_tree_size` is the draft
    // engine's input budget and lives in draft_config.json; the builder
    // correctly omits it from base_config.json (see llmBuilder.h), so
    // `cfg.maxDraftTreeSize` stays at 0 on the base — consumers read the
    // authoritative value from the draft engine config (or from
    // `DeploymentConfig::specDecode`, which consolidates both sides).
    if (cfg.isSpecDecodeBase)
    {
        cfg.maxVerifyTreeSize = getRequired<int32_t>(bc, "max_verify_tree_size");
        requirePositive(cfg.maxVerifyTreeSize, "max_verify_tree_size");
        if (cfg.specDecodeType == SpecDecodeMode::kGemma4MTP)
        {
            ELLM_CHECK(cfg.modelType == "gemma4" || cfg.modelType == "gemma4_text",
                "parseEngineConfig: gemma4_mtp base config must set model to gemma4 or gemma4_text.");
            ELLM_CHECK(cfg.baseModelHiddenSize == 0 || cfg.baseModelHiddenSize == cfg.hiddenSize,
                "parseEngineConfig: gemma4_mtp base_model_hidden_size must match hidden_size.");
        }
    }

    // Populate per-layer type routing from canonical fields or scalar fallback.
    populateLayerTypes(configJson, cfg);
    parseDualRopeFields(configJson, cfg);

    // KV sharing donors: optional array of per-attention-layer donor indices.
    // Each entry is -1 (owns its own KV) or >= 0 (shares donor's KV cache).
    if (configJson.contains("kv_sharing_donors"))
    {
        auto const& donorsJson = configJson["kv_sharing_donors"];
        int32_t const numAttn = static_cast<int32_t>(cfg.kvLayerConfigs.size());
        ELLM_CHECK(static_cast<int32_t>(donorsJson.size()) == numAttn,
            "parseEngineConfig: kv_sharing_donors length (" + std::to_string(donorsJson.size())
                + ") must equal number of attention layers (" + std::to_string(numAttn) + ")");
        cfg.kvSharingDonors.reserve(numAttn);
        for (int32_t i = 0; i < numAttn; ++i)
        {
            int32_t donor = donorsJson[i].get<int32_t>();
            ELLM_CHECK(donor == -1 || (donor >= 0 && donor < numAttn && donor != i),
                "parseEngineConfig: kv_sharing_donors[" + std::to_string(i) + "] = " + std::to_string(donor)
                    + " is invalid (must be -1 or a different layer in [0, " + std::to_string(numAttn - 1) + "])");
            cfg.kvSharingDonors.push_back(donor);
        }
    }

    // --- EOS token IDs (optional array) ---
    if (configJson.contains("eos_token_id") && configJson["eos_token_id"].is_array())
    {
        for (auto const& id : configJson["eos_token_id"])
        {
            if (id.is_number_integer())
            {
                cfg.eosTokenIds.push_back(id.get<int32_t>());
            }
        }
    }

    LOG_INFO("%s", formatEngineConfig(cfg).c_str());
    return cfg;
}

LLMEngineConfig parseDraftEngineConfig(std::filesystem::path const& configPath)
{
    LOG_INFO("reading %s", configPath.string().c_str());

    std::ifstream ifs(configPath);
    ELLM_CHECK(ifs.is_open(), "parseDraftEngineConfig: failed to open " + configPath.string());

    nlohmann::json configJson;
    try
    {
        configJson = nlohmann::json::parse(ifs);
    }
    catch (nlohmann::json::parse_error const& e)
    {
        throw std::runtime_error(
            "parseDraftEngineConfig: JSON parse error in " + configPath.string() + ": " + e.what());
    }
    ifs.close();

    LLMEngineConfig cfg;

    cfg.specDecodeType = parseSpecDecodeMode(configJson);
    std::string const engineRole = parseEngineRole(configJson);
    ELLM_CHECK(engineRole == "draft", "parseDraftEngineConfig: draft config must set engine_role=draft.");
    ELLM_CHECK(cfg.specDecodeType != SpecDecodeMode::kNONE,
        "parseDraftEngineConfig: engine_role=draft requires spec_decode_type to be mtp, eagle3, dflash, or "
        "gemma4_mtp.");

    // Shared core fields (layers, kv heads, head_dim, hidden_size, kv_cache_dtype,
    // batch/input/kv limits, RoPE, common positivity checks).
    parseCoreFields(configJson, cfg);
    parseGemma4MTPFields(configJson, cfg);

    // --- Draft-specific ---
    cfg.numAttentionLayers = cfg.numDecoderLayers;
    // Match the engine's `rope_rotary_cos_sin` binding shape. Most partial
    // rotary models expose a smaller binding via `partial_rotary_factor`, while
    // proportional RoPE keeps a headDim-sized binding and treats the non-rotated
    // tail as identity.
    cfg.rotaryDim = static_cast<int32_t>(getRotaryDim(configJson, cfg.headDim));
    cfg.vocabSize = configJson.value("draft_vocab_size", configJson.value("vocab_size", 0));
    cfg.reducedVocabSize = configJson.value(binding_names::kReducedVocabSizeKey, 0);
    cfg.outputVocabSize = (cfg.reducedVocabSize > 0) ? cfg.reducedVocabSize : cfg.vocabSize;
    parseDFlashFields(configJson, cfg);

    // Draft engines do not own speculative base verification bindings.
    cfg.isSpecDecodeBase = false;

    auto const& bc = configJson["builder_config"];
    // Symmetric to the base side (see parseEngineConfig): each engine's
    // builder_config carries only its own sequence budget. The base emits
    // `max_verify_tree_size` (its verification budget); the draft emits
    // `max_draft_tree_size` (its proposal/generation budget). So the draft
    // parser only requires `max_draft_tree_size`; `cfg.maxVerifyTreeSize`
    // stays at 0 on the draft — consumers read the consolidated values
    // from `DeploymentConfig::specDecode`.
    cfg.maxDraftTreeSize = getRequired<int32_t>(bc, "max_draft_tree_size");
    requirePositive(cfg.maxDraftTreeSize, "max_draft_tree_size");

    // Hidden dim the draft engine's `hidden_states_input` binding expects. The
    // python export writes this as `base.hidden_size * 3` for EAGLE-3 (multi-layer
    // concat) and `base.hidden_size` for MTP, so the runtime must honor whatever
    // the draft config carries — deriving it from `base.hiddenSize * 3` is wrong
    // for MTP.
    cfg.baseModelHiddenSize = getRequired<int32_t>(configJson, "base_model_hidden_size");
    requirePositive(cfg.baseModelHiddenSize, "base_model_hidden_size");

    if (cfg.specDecodeType == SpecDecodeMode::kGemma4MTP)
    {
        ELLM_CHECK(cfg.modelType == "gemma4_assistant",
            "parseDraftEngineConfig: gemma4_mtp draft config must set model to gemma4_assistant.");
        ELLM_CHECK(cfg.sharesTargetKV, "parseDraftEngineConfig: gemma4_mtp assistant must set shares_target_kv=true.");
        ELLM_CHECK(!cfg.hasOwnKVCache, "parseDraftEngineConfig: gemma4_mtp assistant must set has_own_kv_cache=false.");
        ELLM_CHECK(cfg.constantDraftPositions,
            "parseDraftEngineConfig: gemma4_mtp assistant must set constant_draft_positions=true.");
        ELLM_CHECK(cfg.returnsFeedbackHidden,
            "parseDraftEngineConfig: gemma4_mtp assistant must set returns_feedback_hidden=true.");
        if (cfg.assistantHiddenSize > 0)
        {
            ELLM_CHECK(cfg.assistantHiddenSize == cfg.hiddenSize,
                "parseDraftEngineConfig: assistant_hidden_size must match hidden_size for gemma4_mtp draft.");
        }
    }

    LOG_INFO(
        "parseDraftEngineConfig: hiddenSize=%d vocabSize=%d layers=%d kvHeads=%d headDim=%d "
        "maxBatch=%d maxInputLen=%d maxKVCap=%d maxDraftTreeSize=%d baseModelHiddenSize=%d",
        cfg.hiddenSize, cfg.vocabSize, cfg.numDecoderLayers, cfg.numKVHeads, cfg.headDim, cfg.maxSupportedBatchSize,
        cfg.maxSupportedInputLength, cfg.maxKVCacheCapacity, cfg.maxDraftTreeSize, cfg.baseModelHiddenSize);

    // Populate per-layer type routing from canonical fields or scalar fallback.
    populateLayerTypes(configJson, cfg);
    parseDualRopeFields(configJson, cfg);

    return cfg;
}

std::string formatEngineConfig(LLMEngineConfig const& cfg)
{
    std::ostringstream ss;
    ss << std::boolalpha;
    ss << "LLMEngineConfig{" << " hiddenSize=" << cfg.hiddenSize << " vocabSize=" << cfg.vocabSize
       << " outputVocabSize=" << cfg.outputVocabSize << " numDecoderLayers=" << cfg.numDecoderLayers
       << " numAttentionLayers=" << cfg.numAttentionLayers << " numKVHeads=" << cfg.numKVHeads
       << " headDim=" << cfg.headDim << " rotaryDim=" << cfg.rotaryDim << " maxBatch=" << cfg.maxSupportedBatchSize
       << " maxInputLen=" << cfg.maxSupportedInputLength << " maxKVCapacity=" << cfg.maxKVCacheCapacity
       << " kvPoolPages=" << cfg.kvPoolPages << " slidingWindowSize="
       << (cfg.slidingWindowSize.has_value() ? std::to_string(*cfg.slidingWindowSize) : "unknown")
       << " pleEnabled=" << cfg.pleEnabled << " numPleInputs=" << cfg.numPleInputs
       << " pleHiddenSize=" << cfg.pleHiddenSize << " isSpecDecodeBase=" << cfg.isSpecDecodeBase
       << " specDecodeType=" << static_cast<int>(cfg.specDecodeType) << " loraRank=" << cfg.maxSupportedLoraRank;
    if (cfg.useDualRope)
    {
        ss << " useDualRope=true" << " slidingRotaryDim=" << cfg.slidingRotaryDim
           << " fullRotaryDim=" << cfg.fullRotaryDim;
    }

    if (cfg.numLinearAttnLayers > 0)
    {
        ss << " numLinearAttnLayers=" << cfg.numLinearAttnLayers
           << " recurrentStateDtype=" << static_cast<int>(cfg.recurrentStateDtype)
           << " convStateDtype=" << static_cast<int>(cfg.convStateDtype);
    }
    if (cfg.maxVerifyTreeSize > 0)
    {
        ss << " maxVerifyTreeSize=" << cfg.maxVerifyTreeSize;
    }
    // Only draft engines carry a meaningful `maxDraftTreeSize`; the base side
    // leaves it at 0 (see parseEngineConfig comment), so skip the noise.
    if (cfg.maxDraftTreeSize > 0)
    {
        ss << " maxDraftTreeSize=" << cfg.maxDraftTreeSize;
    }
    if (cfg.specDecodeType == SpecDecodeMode::kDFlash)
    {
        ss << " dflashBlockSize=" << cfg.dflashBlockSize << " dflashMaskTokenId=" << cfg.dflashMaskTokenId
           << " dflashTargetLayerIds=" << cfg.dflashTargetLayerIds.size();
    }
    if (!cfg.eosTokenIds.empty())
    {
        ss << " eosTokenIds=[";
        for (size_t i = 0; i < cfg.eosTokenIds.size(); ++i)
        {
            if (i > 0)
                ss << ",";
            ss << cfg.eosTokenIds[i];
        }
        ss << "]";
    }
    if (cfg.specDecodeType == SpecDecodeMode::kGemma4MTP)
    {
        ss << " modelType=" << cfg.modelType << " sharesTargetKV=" << cfg.sharesTargetKV
           << " hasOwnKVCache=" << cfg.hasOwnKVCache << " assistantHiddenSize=" << cfg.assistantHiddenSize
           << " kvSharingMap=" << cfg.gemma4MTPKVSharingMap.size();
    }
    ss << " }";
    return ss.str();
}

// ---------------------------------------------------------------------------
// InferenceDims recipe methods
//
// Tier-1 #7 (C++20 designated initializers) is DEFERRED: the repo compiles
// with -std=c++17 -Werror, which rejects designated initializers even as a
// GNU extension. Keep the `/*.field=*/value` comment style until the repo's
// CXX_STANDARD is bumped.
// ---------------------------------------------------------------------------

InferenceDims LLMEngineConfig::prefillDims(int64_t batch, int64_t seqLen, bool kvCacheAllEmpty) const
{
    // seqLen drives the inputs_embeds / KV-write length only.
    //
    // attnMaskSeqLen and packedMaskLen are pinned to 1 (NOT derived from seqLen):
    // the SpecDecode base/draft attention plugin treats a `[B, 1, 1]` mask as a
    // signal to use standard causal attention and ignores the buffer contents,
    // which matches the dummy-mask binding from the pre-refactor runtime. A
    // larger attention shape would be interpreted as a proposal-attention mask
    // and read uninitialized buffer bits, producing garbage outputs.
    //
    // startIndexLen=0 is the plugin-path sentinel for "initial prefill of an
    // empty KV cache"; chunked prefill uses [batch].
    int64_t const startIndexLen = kvCacheAllEmpty ? 0 : batch;
    return InferenceDims{
        /*.batch=*/batch,
        /*.seqLen=*/seqLen,
        /*.kvLen=*/maxKVCacheCapacity,
        /*.selectLen=*/1,
        /*.attnMaskSeqLen=*/1,
        /*.ropeBatch=*/(ropeConfig.type == RopeType::kMRope) ? batch : 1,
        /*.packedMaskLen=*/1,
        /*.startIndexLen=*/startIndexLen,
        /*.specVerifyPhaseLen=*/0,
    };
}

InferenceDims LLMEngineConfig::decodeDims(int64_t batch) const
{
    return InferenceDims{
        /*.batch=*/batch,
        /*.seqLen=*/1,
        /*.kvLen=*/maxKVCacheCapacity,
        /*.selectLen=*/1,
        /*.attnMaskSeqLen=*/1,
        /*.ropeBatch=*/(ropeConfig.type == RopeType::kMRope) ? batch : 1,
        /*.packedMaskLen=*/1,
        /*.startIndexLen=*/batch,
        /*.specVerifyPhaseLen=*/0,
    };
}

InferenceDims LLMEngineConfig::specVerifyDims(int64_t batch, int64_t verifySize) const
{
    // verifySize fans out to four fields:
    //   seqLen, selectLen, attnMaskSeqLen (all = verifySize), and packedMaskLen.
    // This is the only recipe where selectLen != 1.
    return InferenceDims{
        /*.batch=*/batch,
        /*.seqLen=*/verifySize,
        /*.kvLen=*/maxKVCacheCapacity,
        /*.selectLen=*/verifySize,
        /*.attnMaskSeqLen=*/verifySize,
        /*.ropeBatch=*/(ropeConfig.type == RopeType::kMRope) ? batch : 1,
        /*.packedMaskLen=*/static_cast<int64_t>(divUp(verifySize, 32)),
        /*.startIndexLen=*/batch,
        /*.specVerifyPhaseLen=*/1,
    };
}

InferenceDims LLMEngineConfig::proposalDims(int64_t batch, int64_t proposalSize, int64_t draftTopK) const
{
    // proposalSize fans out to: seqLen, attnMaskSeqLen, and packedMaskLen.
    // draftTopK is selectLen — the draft proposal selects draftTopK tokens per
    // sequence (matches the draft engine's 3D logits output shape
    // [batch, draftTopK, draftVocabSize] and last_token_ids shape
    // [batch, draftTopK]).
    return InferenceDims{
        /*.batch=*/batch,
        /*.seqLen=*/proposalSize,
        /*.kvLen=*/maxKVCacheCapacity,
        /*.selectLen=*/draftTopK,
        /*.attnMaskSeqLen=*/proposalSize,
        /*.ropeBatch=*/(ropeConfig.type == RopeType::kMRope) ? batch : 1,
        /*.packedMaskLen=*/static_cast<int64_t>(divUp(proposalSize, 32)),
        /*.startIndexLen=*/batch,
        /*.specVerifyPhaseLen=*/0,
    };
}

InferenceDims LLMEngineConfig::acceptDims(int64_t batch, int64_t acceptLen) const
{
    // Precondition: acceptLen >= 1. Caller computes from the SpecDecode accept result in
    // [1, draftingStep+1]; a zero here would indicate a degenerate batch and
    // will be caught by EngineExecutor::prepare's field-completeness check.
    // acceptLen fans out to: seqLen, attnMaskSeqLen, and packedMaskLen.
    return InferenceDims{
        /*.batch=*/batch,
        /*.seqLen=*/acceptLen,
        /*.kvLen=*/maxKVCacheCapacity,
        /*.selectLen=*/1,
        /*.attnMaskSeqLen=*/acceptLen,
        /*.ropeBatch=*/(ropeConfig.type == RopeType::kMRope) ? batch : 1,
        /*.packedMaskLen=*/static_cast<int64_t>(divUp(acceptLen, 32)),
        /*.startIndexLen=*/batch,
        /*.specVerifyPhaseLen=*/0,
    };
}

void validateAgainstEngine(LLMEngineConfig const& config, EngineExecutor const& executor, char const* engineLabel)
{
    if (isDFlashDraftConfig(config))
    {
        // DFlash cached draft engines require KV cache bindings (cached-KV path).
        // Validate required bindings exist and have correct dtype.
        LOG_INFO("DFlash draft engine (%s): validating cached-path bindings.", engineLabel);

        // Required cached-path bindings (fail if missing → old explicit DFlash engine)
        static char const* const kRequiredBindings[] = {
            binding_names::kInputsEmbeds,
            binding_names::kDFlashTargetHiddenConcat,
            binding_names::kLogits,
            binding_names::kContextLengths,
            binding_names::kKVCacheStartIndex,
            binding_names::kDFlashDeltaLengths,
            binding_names::kRopeCosSin,
            binding_names::kAttentionMask,
            binding_names::kAttentionPosId,
        };
        for (auto const* name : kRequiredBindings)
        {
            ELLM_CHECK(executor.hasIOTensor(name),
                std::string("DFlash cached draft engine (") + engineLabel + ") is missing required binding '" + name
                    + "'. This engine may be from the old explicit DFlash path. Re-export and rebuild.");
        }

        // Require KV cache layer 0
        if (config.numAttentionLayers > 0)
        {
            std::string const kvPastName = binding_names::formatKVCacheName(/*layerIdx=*/0, /*isPast=*/true);
            std::string const kvPresentName = binding_names::formatKVCacheName(/*layerIdx=*/0, /*isPast=*/false);
            ELLM_CHECK(executor.hasIOTensor(kvPastName.c_str()),
                std::string("DFlash cached draft engine (") + engineLabel + ") missing KV cache binding '" + kvPastName
                    + "'. Old explicit DFlash engines are not compatible. Re-export and rebuild.");
            ELLM_CHECK(executor.hasIOTensor(kvPresentName.c_str()),
                std::string("DFlash cached draft engine (") + engineLabel + ") missing KV cache binding '"
                    + kvPresentName + "'.");

            auto const engineDtype = executor.getBindingDataType(kvPastName.c_str());
            ELLM_CHECK(engineDtype == config.kvCacheDtype,
                std::string("KV cache dtype mismatch (") + engineLabel + "): config says "
                    + getDataTypeString(config.kvCacheDtype) + ", engine reports " + getDataTypeString(engineDtype)
                    + " for binding '" + kvPastName + "'.");
        }

        return;
    }

    if (isGemma4MTPDraftConfig(config))
    {
        LOG_INFO("Gemma4 MTP draft engine (%s): validating shared-target-KV bindings.", engineLabel);

        static char const* const kRequiredBindings[] = {
            binding_names::kInputsEmbeds,
            binding_names::kBaseModelHiddenStates,
            binding_names::kLogits,
            binding_names::kOutputHiddenStates,
            binding_names::kContextLengths,
        };
        for (auto const* name : kRequiredBindings)
        {
            ELLM_CHECK(executor.hasIOTensor(name),
                std::string("Gemma4 MTP draft engine (") + engineLabel + ") is missing required binding '" + name
                    + "'. Re-export and rebuild the assistant engine.");
        }
        if (config.useDualRope)
        {
            ELLM_CHECK(executor.hasIOTensor(binding_names::kRopeCosSinSliding),
                std::string("Gemma4 MTP draft engine (") + engineLabel + ") is missing required dual-RoPE binding '"
                    + binding_names::kRopeCosSinSliding + "'. Re-export and rebuild the assistant engine.");
            ELLM_CHECK(executor.hasIOTensor(binding_names::kRopeCosSinFull),
                std::string("Gemma4 MTP draft engine (") + engineLabel + ") is missing required dual-RoPE binding '"
                    + binding_names::kRopeCosSinFull + "'. Re-export and rebuild the assistant engine.");
        }
        else
        {
            ELLM_CHECK(executor.hasIOTensor(binding_names::kRopeCosSin),
                std::string("Gemma4 MTP draft engine (") + engineLabel + ") is missing required binding '"
                    + binding_names::kRopeCosSin + "'. Re-export and rebuild the assistant engine.");
        }

        ELLM_CHECK(!executor.hasIOTensor(binding_names::kDraftModelHiddenStates),
            std::string("Gemma4 MTP draft engine (") + engineLabel + ") must not expose binding '"
                + binding_names::kDraftModelHiddenStates + "'.");
        ELLM_CHECK(!executor.hasIOTensor(binding_names::kKVCacheStartIndex),
            std::string("Gemma4 MTP draft engine (") + engineLabel
                + ") must not expose draft-owned kvcache_start_index.");
        ELLM_CHECK(!engineHasTensorWithPrefix(executor, binding_names::kPresentKeyValuesTemplate),
            std::string("Gemma4 MTP draft engine (") + engineLabel + ") must not expose present_key_values_* outputs.");

        for (int32_t assistantLayerIdx = 0; assistantLayerIdx < config.numAttentionLayers; ++assistantLayerIdx)
        {
            std::string const kvPastName = binding_names::formatKVCacheName(assistantLayerIdx, /*isPast=*/true);
            ELLM_CHECK(executor.hasIOTensor(kvPastName.c_str()),
                std::string("Gemma4 MTP draft engine (") + engineLabel + ") missing shared target KV input '"
                    + kvPastName + "'.");
            auto const engineDtype = executor.getBindingDataType(kvPastName.c_str());
            ELLM_CHECK(engineDtype == config.kvCacheDtype,
                std::string("Gemma4 MTP shared KV dtype mismatch (") + engineLabel + "): config says "
                    + getDataTypeString(config.kvCacheDtype) + ", engine reports " + getDataTypeString(engineDtype)
                    + " for binding '" + kvPastName + "'.");
            ELLM_CHECK(assistantLayerIdx < static_cast<int32_t>(config.kvLayerConfigs.size()),
                std::string("Gemma4 MTP draft engine (") + engineLabel
                    + ") config is missing per-layer KV metadata for assistant layer "
                    + std::to_string(assistantLayerIdx) + ".");

            auto const& kvConfig = config.kvLayerConfigs[assistantLayerIdx];
            int32_t const numProfiles = executor.getEngine().getNbOptimizationProfiles();
            for (int32_t profileIdx = 0; profileIdx < numProfiles; ++profileIdx)
            {
                for (auto const selector : {nvinfer1::OptProfileSelector::kMIN, nvinfer1::OptProfileSelector::kOPT,
                         nvinfer1::OptProfileSelector::kMAX})
                {
                    auto const shape = executor.getProfileShape(kvPastName.c_str(), profileIdx, selector);
                    ELLM_CHECK(shape.nbDims == 5,
                        std::string("Gemma4 MTP shared KV binding '") + kvPastName + "' must be rank-5.");
                    // Paged pool binding [2, numPages, kTOKENS_PER_PAGE, numKVHeads, headDim];
                    // numPages (dim 1) is engine-specific.
                    ELLM_CHECK(shape.d[0] == 2 && shape.d[2] == kTOKENS_PER_PAGE && shape.d[3] == kvConfig.numKVHeads
                            && shape.d[4] == kvConfig.headDim,
                        std::string("Gemma4 MTP shared KV profile shape mismatch for binding '") + kvPastName
                            + "': expected static dims [2,*," + std::to_string(kTOKENS_PER_PAGE) + ","
                            + std::to_string(kvConfig.numKVHeads) + "," + std::to_string(kvConfig.headDim) + "].");
                }
            }
        }

        return;
    }

    // KV cache binding: validated on layer 0; all layers share the same dtype.
    // Plugin-path engines use a combined `past_key_values_%d` binding.
    if (config.numAttentionLayers > 0)
    {
        std::string const kvBindingName = binding_names::formatKVCacheName(/*layerIdx=*/0, /*isPast=*/true);
        ELLM_CHECK(executor.hasIOTensor(kvBindingName.c_str()),
            std::string("Missing KV cache binding (") + engineLabel + "): expected '" + kvBindingName + "'.");
        auto const engineDtype = executor.getBindingDataType(kvBindingName.c_str());
        ELLM_CHECK(engineDtype == config.kvCacheDtype,
            std::string("KV cache dtype mismatch (") + engineLabel + "): config says "
                + getDataTypeString(config.kvCacheDtype) + ", engine reports " + getDataTypeString(engineDtype)
                + " for binding '" + kvBindingName + "'. Re-export the engine with matching kv_cache_dtype.");
    }

    // Recurrent / conv state bindings (hybrid models only).
    if (config.numLinearAttnLayers > 0)
    {
        std::string const recBindingName = binding_names::formatRecurrentStateName(/*layerIdx=*/0, /*isPast=*/true);
        ELLM_CHECK(executor.hasIOTensor(recBindingName.c_str()),
            std::string("Missing recurrent-state binding (") + engineLabel + "): expected '" + recBindingName + "'.");
        auto const recEngineDtype = executor.getBindingDataType(recBindingName.c_str());
        ELLM_CHECK(recEngineDtype == config.recurrentStateDtype,
            std::string("Recurrent state dtype mismatch (") + engineLabel + "): config says "
                + getDataTypeString(config.recurrentStateDtype) + ", engine reports "
                + getDataTypeString(recEngineDtype) + " for binding '" + recBindingName
                + "'. Re-export the engine with matching recurrent_state_dtype.");

        std::string const convBindingName = binding_names::formatConvStateName(/*layerIdx=*/0, /*isPast=*/true);
        ELLM_CHECK(executor.hasIOTensor(convBindingName.c_str()),
            std::string("Missing conv-state binding (") + engineLabel + "): expected '" + convBindingName + "'.");
        auto const convEngineDtype = executor.getBindingDataType(convBindingName.c_str());
        ELLM_CHECK(convEngineDtype == config.convStateDtype,
            std::string("Conv state dtype mismatch (") + engineLabel + "): config says "
                + getDataTypeString(config.convStateDtype) + ", engine reports " + getDataTypeString(convEngineDtype)
                + " for binding '" + convBindingName + "'. Re-export the engine with matching conv_state_dtype.");

        if (config.isSpecDecodeBase && config.numLinearAttnLayers > 0
            && (config.specDecodeType == SpecDecodeMode::kMTP || config.specDecodeType == SpecDecodeMode::kDFlash))
        {
            ELLM_CHECK(executor.hasIOTensor(binding_names::kSpecVerifyPhaseMarker),
                std::string("Missing spec-verify phase marker binding (") + engineLabel + "): expected '"
                    + binding_names::kSpecVerifyPhaseMarker + "'. Re-export the hybrid MTP/DFlash base engine.");
            auto const markerEngineDtype = executor.getBindingDataType(binding_names::kSpecVerifyPhaseMarker);
            ELLM_CHECK(markerEngineDtype == nvinfer1::DataType::kINT32,
                std::string("Spec-verify phase marker dtype mismatch (") + engineLabel + "): engine reports "
                    + getDataTypeString(markerEngineDtype) + " for binding '" + binding_names::kSpecVerifyPhaseMarker
                    + "'. Re-export the hybrid MTP/DFlash base engine.");
        }
    }
}

InferenceDims LLMEngineConfig::resetDims() const
{
    // Clean-slate binding for CUDA-graph capture reset. ropeBatch is hardcoded
    // to 1 even for MRope — the reset is a placeholder binding, not an
    // inference step.
    // startIndexLen=1 matches batch=1; the reset is a placeholder bind, not a
    // real initial-prefill (which would need shape [0]).
    return InferenceDims{
        /*.batch=*/1,
        /*.seqLen=*/1,
        /*.kvLen=*/maxKVCacheCapacity,
        /*.selectLen=*/1,
        /*.attnMaskSeqLen=*/1,
        /*.ropeBatch=*/1,
        /*.packedMaskLen=*/1,
        /*.startIndexLen=*/1,
        /*.specVerifyPhaseLen=*/0,
    };
}

} // namespace rt
} // namespace trt_edgellm
