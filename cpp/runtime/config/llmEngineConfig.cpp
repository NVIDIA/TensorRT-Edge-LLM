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

#include <NvInferRuntime.h>

#include "common/bindingNames.h"
#include "common/checkMacros.h"
#include "common/cudaUtils.h"
#include "common/logger.h"
#include "common/pagedKvTypes.h"
#include "common/ropeUtils.h"
#include "common/skipSoftmaxScaleTable.h"
#include "common/specDecodeConfigUtils.h"
#include "common/trtUtils.h"
#include "common/version.h"
#include "runtime/exec/engineExecutor.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <limits>
#include <nlohmann/json.hpp>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace trt_edgellm
{
namespace rt
{

using Json = nlohmann::json;

char const* specDecodeModeName(SpecDecodeMode mode) noexcept
{
    switch (mode)
    {
    case SpecDecodeMode::kNONE: return "none";
    case SpecDecodeMode::kEAGLE: return "eagle3";
    case SpecDecodeMode::kMTP: return "mtp";
    case SpecDecodeMode::kDFlash: return "dflash";
    case SpecDecodeMode::kJetSpec: return "jetspec";
    case SpecDecodeMode::kGemma4MTP: return "gemma4_mtp";
    case SpecDecodeMode::kDSpark: return "dspark";
    case SpecDecodeMode::kDomino: return "domino";
    }
    return "unknown";
}

bool isCachedBlockDraftMode(SpecDecodeMode mode) noexcept
{
    return mode == SpecDecodeMode::kDFlash || mode == SpecDecodeMode::kJetSpec || mode == SpecDecodeMode::kDomino;
}

bool isCachedBlockDraftBase(LLMEngineConfig const& config) noexcept
{
    return config.isSpecDecodeBase && isCachedBlockDraftMode(config.specDecodeType);
}

bool isCachedBlockDraftDraft(LLMEngineConfig const& config) noexcept
{
    return !config.isSpecDecodeBase && isCachedBlockDraftMode(config.specDecodeType);
}

namespace
{

constexpr char kObsoleteGenericTokenOwnerBinding[] = "token_to_sequence";

bool isKVCacheBindingDtypeCompatible(trt_edgellm::DataType configDtype, trt_edgellm::DataType engineDtype)
{
    return engineDtype == configDtype
        || (configDtype == trt_edgellm::DataType::kFP8 && engineDtype == trt_edgellm::DataType::kINT8);
}

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
//! `conv_state_dtype`) into a `trt_edgellm::DataType`.
//!
//! Accepted tokens and their mapping (matches the export pipeline's writer):
//!   "fp16" → kHALF, "fp8" → kFP8, "int8" → kINT8, "bf16" → kBF16, "fp32" → kFLOAT.
//! `fp32` is required for Qwen3.5 GDN recurrent states (the architecture's SSM
//! kernels run in FP32 for numerical stability), so the export writes
//! `recurrent_state_dtype="fp32"` for that model family.
//! Anything else throws with the offending value, the originating field name,
//! and the allowed set.
trt_edgellm::DataType parseStateDtype(std::string const& token, char const* fieldName)
{
    if (token == "fp16")
    {
        return trt_edgellm::DataType::kHALF;
    }
    if (token == "fp8")
    {
        return trt_edgellm::DataType::kFP8;
    }
    if (token == "int8")
    {
        return trt_edgellm::DataType::kINT8;
    }
    if (token == "bf16")
    {
        return trt_edgellm::DataType::kBF16;
    }
    if (token == "fp32")
    {
        return trt_edgellm::DataType::kFLOAT;
    }
    throw std::runtime_error(std::string("parseEngineConfig: invalid ") + fieldName + " '" + token
        + "'. Allowed values: fp16, fp8, int8, bf16, fp32. Re-export the engine with a supported value.");
}

//! Parse a required dtype field. Throws if the field is missing — all current
//! exports (>= 0.7.0) write dtype fields explicitly to config.json.
void parseRequiredStateDtype(Json const& json, char const* key, trt_edgellm::DataType& out)
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
    if (specDecodeType == "jetspec")
    {
        return SpecDecodeMode::kJetSpec;
    }
    if (specDecodeType == "gemma4_mtp")
    {
        return SpecDecodeMode::kGemma4MTP;
    }
    if (specDecodeType == "dspark")
    {
        return SpecDecodeMode::kDSpark;
    }
    if (specDecodeType == "domino")
    {
        return SpecDecodeMode::kDomino;
    }
    throw std::runtime_error("parseEngineConfig: invalid spec_decode_type '" + specDecodeType
        + "'. Allowed values: none, mtp, eagle3, dflash, jetspec, dspark, domino, gemma4_mtp.");
}

std::string parseEngineRole(Json const& configJson)
{
    std::string const engineRole = configJson.value("engine_role", "llm");
    if (engineRole == "llm" || engineRole == "base" || engineRole == "draft" || engineRole == "dllm")
    {
        return engineRole;
    }
    throw std::runtime_error(
        "parseEngineConfig: invalid engine_role '" + engineRole + "'. Allowed values: llm, base, draft, dllm.");
}

void validateSpecTargetLayerIds(std::vector<int32_t> const& targetLayerIds, int32_t numDecoderLayers,
    char const* modeName, char const* layerCountOwner)
{
    for (int32_t layerId : targetLayerIds)
    {
        ELLM_CHECK(layerId >= 0 && layerId < numDecoderLayers,
            "parseEngineConfig: " + std::string(modeName) + " target layer id " + std::to_string(layerId)
                + " is outside [0, " + layerCountOwner + ".num_hidden_layers).");
    }
}

void parseSpecTargetLayerIds(Json const& modeConfig, char const* modeConfigName, LLMEngineConfig& cfg)
{
    if (modeConfig.contains("target_layer_ids"))
    {
        ELLM_CHECK(modeConfig["target_layer_ids"].is_array(),
            "parseEngineConfig: " + std::string(modeConfigName) + ".target_layer_ids must be an array");
        for (auto const& id : modeConfig["target_layer_ids"])
        {
            cfg.specTargetLayerIds.push_back(id.get<int32_t>());
        }
    }
}

//! `dflash_config.projector_type`, or an empty string when absent.
std::string dflashProjectorType(Json const& dflashConfig)
{
    auto const projectorType = dflashConfig.find("projector_type");
    if (projectorType == dflashConfig.end())
    {
        return {};
    }
    ELLM_CHECK(projectorType->is_string(), "parseEngineConfig: dflash_config.projector_type must be a string");
    return projectorType->get<std::string>();
}

DFlashProjector parseDFlashProjector(Json const& dflashConfig)
{
    std::string const projectorType = dflashProjectorType(dflashConfig);
    if (projectorType.empty())
    {
        return DFlashProjector::kSelector;
    }
    ELLM_CHECK(projectorType == binding_names::kDFlashLiLiCorrProjectorType,
        "parseEngineConfig: unsupported DFlash V2 dflash_config.projector_type '" + projectorType
            + "'. Allowed values: '' (selector), '" + binding_names::kDFlashLiLiCorrProjectorType + "'.");
    return DFlashProjector::kLiLiCorr;
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

    int32_t const version = dflashConfig.value("version", 1);
    ELLM_CHECK(version == 1 || version == 2,
        "parseEngineConfig: unsupported dflash_config.version " + std::to_string(version));
    cfg.dflashVersion = static_cast<DFlashVersion>(version);

    cfg.specDraftBlockSize = dflashConfig.value("block_size", 16);
    cfg.specDraftMaskTokenId = dflashConfig.value("mask_token_id", 248070);
    ELLM_CHECK(cfg.specDraftBlockSize > 0,
        "parseEngineConfig: invalid DFlash block_size: " + std::to_string(cfg.specDraftBlockSize)
            + " (must be positive)");
    ELLM_CHECK(cfg.specDraftMaskTokenId >= 0,
        "parseEngineConfig: invalid DFlash mask_token_id: " + std::to_string(cfg.specDraftMaskTokenId)
            + " (must be non-negative)");

    if (cfg.dflashVersion == DFlashVersion::kV2)
    {
        cfg.dflashProjector = parseDFlashProjector(dflashConfig);
        bool const isLiLiCorr = cfg.dflashProjector == DFlashProjector::kLiLiCorr;
        cfg.specDraftCausalHead = getRequired<bool>(dflashConfig, "is_causal");
        cfg.specConvKernelSize = getRequired<int32_t>(dflashConfig, "conv_kernel_size");
        cfg.specConvGroupSize = getRequired<int32_t>(dflashConfig, "conv_group_size");
        if (isLiLiCorr)
        {
            cfg.specLiLiCorrTopK = getRequired<int32_t>(dflashConfig, "lilicorr_candidate_topk");
            for (char const* const selectorKey : {"selector_rank", "selector_top_k", "selector_file"})
            {
                ELLM_CHECK(!dflashConfig.contains(selectorKey),
                    std::string("parseEngineConfig: DFlash V2 projector_type 'lilicorr' must not carry dflash_config.")
                        + selectorKey);
            }
        }
        else
        {
            cfg.specSelectorRank = getRequired<int32_t>(dflashConfig, "selector_rank");
            cfg.specSelectorTopK = getRequired<int32_t>(dflashConfig, "selector_top_k");
            cfg.dflash2SelectorFile
                = dflashConfig.value("selector_file", std::string(binding_names::kDFlash2SelectorFileName));
        }
        cfg.specSupportsProbabilistic = dflashConfig.value("supports_probabilistic_sampling", false);
        ELLM_CHECK(cfg.specDraftBlockSize >= 2 && cfg.specDraftBlockSize <= 16,
            "parseEngineConfig: DFlash V2 block_size must be in [2, 16]");
        ELLM_CHECK(!cfg.specDraftCausalHead, "parseEngineConfig: DFlash V2 requires is_causal=false");
        ELLM_CHECK(cfg.specConvKernelSize == 2, "parseEngineConfig: DFlash V2 conv_kernel_size must be 2");
        ELLM_CHECK(cfg.specConvGroupSize == 16, "parseEngineConfig: DFlash V2 conv_group_size must be 16");
        if (isLiLiCorr)
        {
            int32_t const topK = cfg.specLiLiCorrTopK;
            ELLM_CHECK(topK >= 1 && topK <= binding_names::kLiLiCorrMaxTopK && (topK & (topK - 1)) == 0,
                "parseEngineConfig: DFlash V2 lilicorr_candidate_topk must be one of 1, 2, 4, 8, 16; got "
                    + std::to_string(topK));
        }
        else
        {
            ELLM_CHECK(cfg.specSelectorRank == 256, "parseEngineConfig: DFlash V2 selector_rank must be 256");
            ELLM_CHECK(cfg.specSelectorTopK == 16, "parseEngineConfig: DFlash V2 selector_top_k must be 16");
        }
        ELLM_CHECK(
            cfg.specSupportsProbabilistic, "parseEngineConfig: DFlash V2 requires probabilistic sampling support");
    }
    else
    {
        ELLM_CHECK(dflashProjectorType(dflashConfig).empty(),
            "parseEngineConfig: dflash_config.projector_type requires dflash_config.version 2");
    }

    parseSpecTargetLayerIds(dflashConfig, "dflash_config", cfg);
    ELLM_CHECK(
        !cfg.specTargetLayerIds.empty(), "parseEngineConfig: DFlash requires non-empty dflash_config.target_layer_ids");
    if (cfg.dflashVersion == DFlashVersion::kV2)
    {
        constexpr size_t kDFlashV2ProductionLayers{5};
        ELLM_CHECK(cfg.dflashProjector == DFlashProjector::kLiLiCorr
                || cfg.specTargetLayerIds.size() == kDFlashV2ProductionLayers,
            "parseEngineConfig: DFlash V2 production contract requires exactly five target layer IDs");
        std::vector<int32_t> sortedTargetLayerIds = cfg.specTargetLayerIds;
        std::sort(sortedTargetLayerIds.begin(), sortedTargetLayerIds.end());
        ELLM_CHECK(
            std::adjacent_find(sortedTargetLayerIds.begin(), sortedTargetLayerIds.end()) == sortedTargetLayerIds.end(),
            "parseEngineConfig: DFlash V2 target layer IDs must be unique");
        if (!targetLayerValidationUpperBound.has_value())
        {
            ELLM_CHECK(cfg.numDecoderLayers == static_cast<int32_t>(kDFlashV2ProductionLayers),
                "parseDraftEngineConfig: DFlash V2 production draft requires exactly five decoder layers");
        }
    }
    if (targetLayerValidationUpperBound.has_value())
    {
        validateSpecTargetLayerIds(cfg.specTargetLayerIds, *targetLayerValidationUpperBound, "DFlash", "base");
    }
}

void parseJetSpecFields(
    Json const& configJson, LLMEngineConfig& cfg, std::optional<int32_t> targetLayerValidationUpperBound = std::nullopt)
{
    if (cfg.specDecodeType != SpecDecodeMode::kJetSpec)
    {
        return;
    }

    Json const empty = Json::object();
    Json const& jetSpecConfig = configJson.contains("jetspec_config")
        ? configJson["jetspec_config"]
        : (configJson.contains("dflash_config") ? configJson["dflash_config"] : empty);

    cfg.specDraftBlockSize = jetSpecConfig.value("block_size", configJson.value("block_size", 16));
    cfg.specDraftMaskTokenId = jetSpecConfig.value("mask_token_id", configJson.value("mask_token_id", 151669));
    cfg.specDraftCausalHead = jetSpecConfig.value("causal_head", configJson.value("causal_head", true));
    ELLM_CHECK(cfg.specDraftBlockSize > 0,
        "parseEngineConfig: invalid JetSpec block_size: " + std::to_string(cfg.specDraftBlockSize)
            + " (must be positive)");
    ELLM_CHECK(cfg.specDraftMaskTokenId >= 0,
        "parseEngineConfig: invalid JetSpec mask_token_id: " + std::to_string(cfg.specDraftMaskTokenId)
            + " (must be non-negative)");
    ELLM_CHECK(cfg.specDraftCausalHead,
        "parseEngineConfig: JetSpec requires jetspec_config.causal_head=true for causal proposal attention.");

    char const* configName = configJson.contains("jetspec_config") ? "jetspec_config" : "dflash_config";
    parseSpecTargetLayerIds(jetSpecConfig, configName, cfg);
    ELLM_CHECK(!cfg.specTargetLayerIds.empty(), "parseEngineConfig: JetSpec requires non-empty target_layer_ids");
    if (targetLayerValidationUpperBound.has_value())
    {
        validateSpecTargetLayerIds(cfg.specTargetLayerIds, *targetLayerValidationUpperBound, "JetSpec", "base");
    }
}

//! Checkpoint dtype tokens the PLE pool understands. One table so the accepted set, the enum and
//! the element width cannot drift apart.
struct PleDTypeToken
{
    std::string_view name;
    PleTable::DType dtype;
    int64_t elementBytes;
};

constexpr std::array<PleDTypeToken, 4> kPLE_DTYPES{{
    {"F8_E4M3", PleTable::DType::kFP8, 1},
    {"F16", PleTable::DType::kFP16, 2},
    {"BF16", PleTable::DType::kBF16, 2},
    {"F32", PleTable::DType::kFP32, 4},
}};

PleDTypeToken plePoolDType(std::string const& name)
{
    auto const found = std::find_if(
        kPLE_DTYPES.begin(), kPLE_DTYPES.end(), [&name](PleDTypeToken const& token) { return token.name == name; });
    ELLM_CHECK(found != kPLE_DTYPES.end(), "Unsupported PLE dtype: " + name);
    return *found;
}

PleTable::Source parsePleSource(Json const& metadata, int64_t rowCount, int64_t rowWidth)
{
    ELLM_CHECK(metadata.is_object(), "PLE source metadata must be an object");
    auto const integer = [](Json const& object, char const* field) {
        auto const& value = object.at(field);
        ELLM_CHECK(value.is_number_integer(), std::string("PLE source field must be an integer: ") + field);
        return value.get<int64_t>();
    };

    PleTable::Source source;
    auto const token = plePoolDType(metadata.at("dtype").get<std::string>());
    source.dtype = token.dtype;
    int64_t const elementBytes = token.elementBytes;
    source.bytes = integer(metadata, "bytes");
    ELLM_CHECK(rowWidth <= std::numeric_limits<int64_t>::max() / elementBytes
            && rowCount <= std::numeric_limits<int64_t>::max() / (rowWidth * elementBytes)
            && source.bytes == rowCount * rowWidth * elementBytes,
        "PLE source byte count must match pool rows, row width and source dtype");
    source.scale = metadata.at("scale").get<double>();
    ELLM_CHECK(std::isfinite(source.scale) && source.scale > 0
            && static_cast<double>(static_cast<float>(source.scale)) == source.scale,
        "PLE source scale must be a finite positive float32 value");

    auto const parseSamples = [&integer](Json const& object, int64_t bytes, std::vector<PleTable::Sample>& out) {
        auto const& samples = object.at("samples");
        ELLM_CHECK(samples.is_array() && !samples.empty(), "PLE source requires content identity samples");
        for (auto const& value : samples)
        {
            PleTable::Sample sample{integer(value, "offset"), value.at("data").get<std::string>()};
            ELLM_CHECK(!sample.data.empty() && sample.data.size() % 2 == 0
                    && std::all_of(sample.data.begin(), sample.data.end(),
                        [](char c) { return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'); }),
                "PLE source sample must contain hexadecimal bytes");
            ELLM_CHECK(sample.offset >= 0 && sample.offset <= bytes
                    && sample.data.size() / 2 <= static_cast<uint64_t>(bytes - sample.offset),
                "PLE source sample exceeds tensor byte range");
            out.push_back(std::move(sample));
        }
    };
    auto const parseLocation
        = [&integer](Json const& object, int64_t bytes, std::filesystem::path& file, int64_t& offset) {
              if (!object.contains("location"))
              {
                  return;
              }
              auto const& location = object.at("location");
              file = location.at("file").get<std::string>();
              offset = integer(location, "offset");
              ELLM_CHECK(!file.empty() && !file.has_root_path(), "PLE source location file must be relative");
              for (auto const& part : file)
              {
                  ELLM_CHECK(part != "..", "PLE source location cannot escape the checkpoint directory");
              }
              ELLM_CHECK(offset >= 0 && offset <= std::numeric_limits<int64_t>::max() - bytes
                      && integer(location, "bytes") == bytes,
                  "PLE source location has an invalid tensor byte range");
          };

    // A table split across tensors describes them in "shards"; a single-tensor one names it in
    // "tensor_key".
    if (metadata.contains("shards"))
    {
        ELLM_CHECK(!metadata.contains("tensor_key"), "PLE source must carry either tensor_key or shards, not both");
        auto const& shards = metadata.at("shards");
        ELLM_CHECK(shards.is_array() && !shards.empty(), "PLE source shards must be a non-empty array");
        int64_t rows{};
        int64_t bytes{};
        for (auto const& entry : shards)
        {
            PleTable::ShardSource shard;
            shard.tensorKey = entry.at("tensor_key").get<std::string>();
            ELLM_CHECK(!shard.tensorKey.empty(), "PLE shard tensor_key must not be empty");
            shard.rows = integer(entry, "rows");
            shard.bytes = integer(entry, "bytes");
            ELLM_CHECK(
                shard.rows > 0 && shard.rows <= rowCount - rows && shard.bytes == shard.rows * rowWidth * elementBytes,
                "PLE shard byte count must match its rows, row width and source dtype");
            parseSamples(entry, shard.bytes, shard.samples);
            parseLocation(entry, shard.bytes, shard.file, shard.offset);
            rows += shard.rows;
            bytes += shard.bytes;
            source.shards.push_back(std::move(shard));
        }
        ELLM_CHECK(rows == rowCount && bytes == source.bytes, "PLE shards do not cover the pool geometry exactly");
    }
    else
    {
        source.tensorKey = metadata.at("tensor_key").get<std::string>();
        ELLM_CHECK(!source.tensorKey.empty(), "PLE source tensor_key must not be empty");
        parseSamples(metadata, source.bytes, source.samples);
        parseLocation(metadata, source.bytes, source.file, source.offset);
    }
    return source;
}

//! Parse the Gemma4 per-layer-embedding and Qwen4-Exp n-gram features plus the shared PLE row pool.
//! The n-gram injection sites are read from `cfg.ngramLayers`, which the caller has already
//! populated: from the `layers` table on a table engine (`hasLayerTable`), or from the top-level
//! `ngram_layers` on a pre-table engine. `ngram_conv_dim` and `ngram_conv_state_dtype` describe the
//! per-layer conv state the table already owns, so they are read from the top level only when no
//! table is present.
void parsePleConfig(Json const& configJson, LLMEngineConfig& cfg, bool hasLayerTable)
{
    // Gemma4's per-layer embeddings, one slice per decoder layer.
    cfg.gemmaPleEnabled = configJson.value("gemma_ple_enabled", false);
    cfg.gemmaPleLayerCount = configJson.value("gemma_ple_layer_count", 0);
    cfg.gemmaPleHiddenSize = configJson.value("gemma_ple_hidden_size", 0);
    ELLM_CHECK(cfg.gemmaPleLayerCount >= 0,
        "parseEngineConfig: invalid gemma_ple_layer_count: " + std::to_string(cfg.gemmaPleLayerCount)
            + " (must be non-negative)");
    ELLM_CHECK(cfg.gemmaPleHiddenSize >= 0,
        "parseEngineConfig: invalid gemma_ple_hidden_size: " + std::to_string(cfg.gemmaPleHiddenSize)
            + " (must be non-negative)");
    if (cfg.gemmaPleEnabled)
    {
        requirePositive(cfg.gemmaPleLayerCount, "gemma_ple_layer_count");
        requirePositive(cfg.gemmaPleHiddenSize, "gemma_ple_hidden_size");
        ELLM_CHECK(cfg.gemmaPleLayerCount == cfg.numDecoderLayers,
            "parseEngineConfig: Gemma4 PLE expects gemma_ple_layer_count to match num_hidden_layers; got "
                + std::to_string(cfg.gemmaPleLayerCount) + " and " + std::to_string(cfg.numDecoderLayers));
    }
    else
    {
        ELLM_CHECK(cfg.gemmaPleLayerCount == 0 && cfg.gemmaPleHiddenSize == 0,
            "parseEngineConfig: gemma_ple_enabled is false but gemma_ple_layer_count or gemma_ple_hidden_size is "
            "non-zero");
    }

    // Qwen4-Exp's hashed n-gram injection shares only the row pool. On a table engine
    // `parseLayerSpecs` has already populated the injection sites (`cfg.ngramLayers`) and the
    // per-layer conv-state geometry (`cfg.ngramConvDim`/`cfg.ngramConvStateDtype`) the table owns, so
    // they are not re-read here. A pre-table engine lists those at the top level instead. The hash
    // feature metadata (`ngram_embed_dim`/`ngram_conv_kernel`/`ngram_size`/`ngram_heads_per_order`)
    // is not part of the table and is read here on both paths.
    if (!hasLayerTable)
    {
        cfg.ngramLayers = configJson.value("ngram_layers", std::vector<int32_t>{});
        cfg.ngramConvDim = configJson.value("ngram_conv_dim", 0);
    }
    cfg.ngramEmbedDim = configJson.value("ngram_embed_dim", 0);
    cfg.ngramConvKernel = configJson.value("ngram_conv_kernel", 0);
    cfg.ngramSize = configJson.value("ngram_size", 0);
    cfg.ngramHeadsPerOrder = configJson.value("ngram_heads_per_order", 0);
    if (cfg.ngramEnabled())
    {
        requirePositive(cfg.ngramEmbedDim, "ngram_embed_dim");
        requirePositive(cfg.ngramConvKernel, "ngram_conv_kernel");
        requirePositive(cfg.ngramConvDim, "ngram_conv_dim");
        requirePositive(cfg.ngramHeadsPerOrder, "ngram_heads_per_order");
        // A bigram is the shortest n-gram with a hash head, so anything below 2 describes a table
        // that no head would ever address.
        ELLM_CHECK(cfg.ngramSize >= 2,
            "parseEngineConfig: ngram_size must be at least 2, got " + std::to_string(cfg.ngramSize));
        ELLM_CHECK(cfg.ngramEmbedDim % cfg.ngramHeads() == 0,
            "parseEngineConfig: ngram_embed_dim (" + std::to_string(cfg.ngramEmbedDim) + ") must divide into "
                + std::to_string(cfg.ngramHeads()) + " hash heads");
        for (int32_t layer : cfg.ngramLayers)
        {
            ELLM_CHECK(layer >= 0 && layer < cfg.numDecoderLayers,
                "parseEngineConfig: ngram_layers names layer " + std::to_string(layer) + " outside a "
                    + std::to_string(cfg.numDecoderLayers) + "-layer stack");
        }
        if (hasLayerTable)
        {
            // The dilated-conv state the table carries must match the hash geometry read here: its
            // channel count is `ngram_conv_dim` and its width is `(kernel-1) * ngram_size` steps of
            // history (`ngramConvStateLen`). The table owns the shape, the top level owns the kernel
            // and order, so the cross-check only closes once both are parsed.
            Coords const expected{std::vector<int64_t>{cfg.ngramConvDim, cfg.ngramConvStateLen()}};
            for (LayerSpec const& spec : cfg.layers)
            {
                StateSpec const* ngramConv = spec.findState(StateKind::kNgramConv);
                ELLM_CHECK(ngramConv == nullptr || ngramConv->shape == expected,
                    "parseEngineConfig: layer " + std::to_string(spec.absLayerIdx) + " ngram_conv state shape "
                        + (ngramConv != nullptr ? ngramConv->shape.formatString() : std::string{}) + " must equal "
                        + expected.formatString() + " = [ngram_conv_dim, (ngram_conv_kernel-1)*ngram_size]");
            }
        }
        else
        {
            // A table engine's conv-state dtype rides the table; only a pre-table engine still carries
            // it at the top level.
            parseRequiredStateDtype(configJson, "ngram_conv_state_dtype", cfg.ngramConvStateDtype);
        }
    }
    ELLM_CHECK(cfg.ngramEnabled() || (cfg.ngramEmbedDim == 0 && cfg.ngramSize == 0),
        "parseEngineConfig: n-gram geometry is set but ngram_layers is empty");

    // The row pool both features gather from: the table stays in the checkpoint and is described
    // here, not copied next to the engine.
    ELLM_CHECK(!cfg.usesPlePool() || configJson.contains("ple_pool"),
        "Engines using the PLE row pool require ple_pool metadata; re-export and rebuild them.");
    if (configJson.contains("ple_pool"))
    {
        auto const& pool = configJson.at("ple_pool");
        ELLM_CHECK(
            pool.is_object() && cfg.usesPlePool(), "ple_pool requires an enabled PLE feature and object metadata");
        PlePoolConfig resource;
        ELLM_CHECK(
            pool.contains("source"), "PLE pool requires checkpoint source metadata; re-export and rebuild the engine.");
        ELLM_CHECK(pool.at("row_count").is_number_integer() && pool.at("row_width").is_number_integer(),
            "PLE pool dimensions must be integers");
        resource.rowCount = pool.at("row_count").get<int64_t>();
        resource.rowWidth = pool.at("row_width").get<int64_t>();
        resource.scale = pool.at("scale").get<double>();
        resource.heads = pool.value("heads", 1);
        if (pool.contains("hash_constants"))
        {
            auto const& keys = pool.at("hash_constants");
            resource.layerMultipliersKey = keys.at("layer_multipliers").get<std::string>();
            resource.headVocabSizesKey = keys.at("heads_vocab_sizes").get<std::string>();
            resource.headOffsetsKey = keys.at("heads_offsets").get<std::string>();
            ELLM_CHECK(!resource.layerMultipliersKey.empty() && !resource.headVocabSizesKey.empty()
                    && !resource.headOffsetsKey.empty(),
                "PLE pool hash constant keys must not be empty");
        }

        resource.dtype = plePoolDType(pool.at("dtype").get<std::string>()).dtype;
        ELLM_CHECK(resource.dtype == PleTable::DType::kFP8 || resource.dtype == PleTable::DType::kFP16,
            "PLE pool must hold F8_E4M3 or F16 rows");
        ELLM_CHECK(resource.dtype == PleTable::DType::kFP8 || resource.scale == 1.0,
            "A non-quantized PLE pool requires unit scale");
        ELLM_CHECK(resource.heads >= 1 && resource.rowWidth > 0
                && resource.rowWidth <= std::numeric_limits<int64_t>::max() / resource.heads,
            "PLE pool row width or head count is invalid");

        // One row per head, so the gathered heads concatenate into the width the graph consumes:
        // the stacked per-layer embedding for Gemma4, the n-gram embedding for a multi-head pool.
        int64_t const gatheredWidth = resource.rowWidth * resource.heads;
        int64_t const expectedWidth = resource.heads == 1
            ? static_cast<int64_t>(cfg.gemmaPleLayerCount) * cfg.gemmaPleHiddenSize
            : cfg.ngramEmbedDim;
        ELLM_CHECK(resource.rowCount > 0 && resource.rowCount < std::numeric_limits<int32_t>::max()
                && gatheredWidth == expectedWidth,
            "PLE pool row count or width is invalid");
        resource.source = parsePleSource(pool.at("source"), resource.rowCount, resource.rowWidth);
        resource.checkpointDir = configJson.value("checkpoint_dir", std::string{});
        cfg.plePool = std::move(resource);
    }
}

void parseEagleFields(Json const& configJson, LLMEngineConfig& cfg)
{
    if (cfg.specDecodeType != SpecDecodeMode::kEAGLE)
    {
        return;
    }
    auto const eagleConfig = configJson.find("eagle3_config");
    if (eagleConfig == configJson.end() || !eagleConfig->contains("candidate_head"))
    {
        return;
    }
    Json const& candidateHead = eagleConfig->at("candidate_head");
    ELLM_CHECK(candidateHead.is_object(), "parseDraftEngineConfig: eagle3_config.candidate_head must be an object.");
    std::string const headType = candidateHead.value("type", std::string{});
    ELLM_CHECK(headType == "low_rank",
        "parseDraftEngineConfig: eagle3_config.candidate_head.type must be 'low_rank'; got '" + headType + "'.");
    cfg.draftCandidateCount = getRequired<int32_t>(candidateHead, "num_candidates");
    requirePositive(cfg.draftCandidateCount, "eagle3_config.candidate_head.num_candidates");
    requirePositive(getRequired<int32_t>(candidateHead, "rank"), "eagle3_config.candidate_head.rank");
}

void parseDominoFields(
    Json const& configJson, LLMEngineConfig& cfg, std::optional<int32_t> targetLayerValidationUpperBound = std::nullopt)
{
    if (cfg.specDecodeType != SpecDecodeMode::kDomino)
    {
        return;
    }

    Json const empty = Json::object();
    Json const& dflashConfig = configJson.contains("dflash_config") ? configJson["dflash_config"] : empty;

    std::string const projector = dflashConfig.value("projector_type", "");
    ELLM_CHECK(projector == "domino",
        "parseEngineConfig: spec_decode_type=domino requires dflash_config.projector_type='domino'; got '" + projector
            + "'.");
    ELLM_CHECK(dflashConfig.value("version", 1) == 1, "parseEngineConfig: Domino requires dflash_config.version=1");
    cfg.dflashVersion = DFlashVersion::kV1;

    cfg.specDraftBlockSize = dflashConfig.value("block_size", 16);
    cfg.specDraftMaskTokenId = dflashConfig.value("mask_token_id", 151669);
    cfg.dominoGruHiddenDim = getRequired<int32_t>(dflashConfig, "gru_hidden_dim");
    cfg.dominoProjDim = getRequired<int32_t>(dflashConfig, "proj_dim");
    cfg.dominoPrefixLen = dflashConfig.value("prefix_len", 1);
    cfg.dominoPoolSize = getRequired<int32_t>(dflashConfig, "pool_size");
    cfg.dominoHeadsFile = dflashConfig.value("heads_file", std::string(binding_names::kDominoHeadsFileName));
    ELLM_CHECK(cfg.specDraftBlockSize >= 3 && cfg.specDraftBlockSize <= 16,
        "parseEngineConfig: Domino block_size must be in [3, 16], got " + std::to_string(cfg.specDraftBlockSize));
    ELLM_CHECK(cfg.specDraftMaskTokenId >= 0,
        "parseEngineConfig: invalid Domino mask_token_id: " + std::to_string(cfg.specDraftMaskTokenId));
    ELLM_CHECK(cfg.dominoGruHiddenDim > 0, "parseEngineConfig: Domino gru_hidden_dim must be positive");
    ELLM_CHECK(cfg.dominoProjDim > 0 && cfg.dominoProjDim <= 512,
        "parseEngineConfig: Domino proj_dim must be in (0, 512], got " + std::to_string(cfg.dominoProjDim));
    ELLM_CHECK(cfg.dominoPrefixLen == 1,
        "parseEngineConfig: Domino runtime supports prefix_len=1 only, got " + std::to_string(cfg.dominoPrefixLen));
    ELLM_CHECK(cfg.dominoPoolSize > 0, "parseEngineConfig: Domino pool_size must be positive");

    parseSpecTargetLayerIds(dflashConfig, "dflash_config", cfg);
    ELLM_CHECK(
        !cfg.specTargetLayerIds.empty(), "parseEngineConfig: Domino requires non-empty dflash_config.target_layer_ids");
    if (targetLayerValidationUpperBound.has_value())
    {
        validateSpecTargetLayerIds(cfg.specTargetLayerIds, *targetLayerValidationUpperBound, "Domino", "base");
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

void parseDSparkFields(
    Json const& configJson, LLMEngineConfig& cfg, std::optional<int32_t> targetLayerValidationUpperBound = std::nullopt)
{
    if (cfg.specDecodeType != SpecDecodeMode::kDSpark)
    {
        return;
    }

    Json const empty = Json::object();
    Json const& dsparkConfig = configJson.contains("dspark_config") ? configJson["dspark_config"] : empty;

    cfg.specDraftBlockSize = dsparkConfig.value("block_size", 7);
    cfg.specDraftMaskTokenId = dsparkConfig.value("mask_token_id", 151669);
    cfg.dsparkEnableConfidenceHead = dsparkConfig.value("enable_confidence_head", false);
    cfg.dsparkConfidenceHeadWithMarkov = dsparkConfig.value("confidence_head_with_markov", false);
    cfg.dsparkMarkovHeadType = dsparkConfig.value("markov_head_type", std::string{});
    cfg.dsparkMarkovRank = dsparkConfig.value("markov_rank", 0);
    cfg.dsparkHeadsFile = dsparkConfig.value("heads_file", std::string(binding_names::kDSparkHeadsFileName));
    cfg.dsparkHeadsInfoFile
        = dsparkConfig.value("heads_info_file", std::string(binding_names::kDSparkHeadsInfoFileName));
    cfg.dsparkContiguousQuerySwa = dsparkConfig.value("contiguous_query_swa", false);
    cfg.specDraftCausalHead = dsparkConfig.value("causal_head", false);
    // Selects which draft query slots carry proposals. When true the anchor slot is
    // itself a proposal and the block is block_size wide; when false slot 0 is the
    // bonus token and the block carries one extra mask slot.
    cfg.dsparkSampleFromAnchor = dsparkConfig.value("sample_from_anchor", true);

    ELLM_CHECK(cfg.specDraftBlockSize > 0,
        "parseEngineConfig: invalid DSpark block_size: " + std::to_string(cfg.specDraftBlockSize)
            + " (must be positive)");
    ELLM_CHECK(cfg.specDraftMaskTokenId >= 0,
        "parseEngineConfig: invalid DSpark mask_token_id: " + std::to_string(cfg.specDraftMaskTokenId)
            + " (must be non-negative)");
    ELLM_CHECK(cfg.dsparkMarkovRank >= 0,
        "parseEngineConfig: invalid DSpark markov_rank: " + std::to_string(cfg.dsparkMarkovRank)
            + " (must be non-negative)");

    parseSpecTargetLayerIds(dsparkConfig, "dspark_config", cfg);
    ELLM_CHECK(
        !cfg.specTargetLayerIds.empty(), "parseEngineConfig: DSpark requires non-empty dspark_config.target_layer_ids");
    if (targetLayerValidationUpperBound.has_value())
    {
        validateSpecTargetLayerIds(cfg.specTargetLayerIds, *targetLayerValidationUpperBound, "DSpark", "base");
    }
}

bool isGemma4MTPDraftConfig(LLMEngineConfig const& config)
{
    return config.specDecodeType == SpecDecodeMode::kGemma4MTP && !config.isSpecDecodeBase;
}

bool isDSparkDraftConfig(LLMEngineConfig const& config)
{
    return config.specDecodeType == SpecDecodeMode::kDSpark && !config.isSpecDecodeBase;
}

void validateAndApplyRankConfigOverrides(
    Json& configJson, std::optional<int32_t> rank, std::optional<int32_t> expectedWorldSize, char const* parserName)
{
    std::string const errorPrefix = std::string(parserName) + ": ";
    ELLM_CHECK(!expectedWorldSize.has_value() || *expectedWorldSize > 0,
        errorPrefix + "expected world size must be positive, got " + std::to_string(*expectedWorldSize) + ".");

    std::optional<int32_t> serializedTpSize;
    if (configJson.contains("builder_config") && configJson["builder_config"].is_object()
        && configJson["builder_config"].contains("tp_size"))
    {
        Json const& tpSizeJson = configJson["builder_config"]["tp_size"];
        ELLM_CHECK(tpSizeJson.is_number_integer(), errorPrefix + "builder_config.tp_size must be an integer.");
        serializedTpSize = tpSizeJson.get<int32_t>();
        ELLM_CHECK(*serializedTpSize > 0,
            errorPrefix + "builder_config.tp_size must be positive, got " + std::to_string(*serializedTpSize) + ".");
    }

    if (!configJson.contains("rank_configs"))
    {
        bool const requiresRankConfigs = (expectedWorldSize.has_value() && *expectedWorldSize > 1)
            || (serializedTpSize.has_value() && *serializedTpSize > 1);
        ELLM_CHECK(!requiresRankConfigs, errorPrefix + "multi-device config requires rank_configs.");
        ELLM_CHECK(
            !expectedWorldSize.has_value() || !serializedTpSize.has_value() || *expectedWorldSize == *serializedTpSize,
            errorPrefix + "builder_config.tp_size (" + std::to_string(*serializedTpSize)
                + ") does not match runtime world size (" + std::to_string(*expectedWorldSize) + ").");
        return;
    }

    ELLM_CHECK(configJson["rank_configs"].is_array(), errorPrefix + "rank_configs must be an array when present.");
    ELLM_CHECK(serializedTpSize.has_value(),
        errorPrefix + "config with rank_configs must contain integer builder_config.tp_size.");

    Json const& rankConfigs = configJson["rank_configs"];
    int32_t const rankConfigCount = static_cast<int32_t>(rankConfigs.size());
    ELLM_CHECK(rankConfigCount > 0, errorPrefix + "rank_configs must not be empty.");
    ELLM_CHECK(*serializedTpSize == rankConfigCount,
        errorPrefix + "rank_configs length (" + std::to_string(rankConfigCount)
            + ") does not match builder_config.tp_size (" + std::to_string(*serializedTpSize) + ").");
    ELLM_CHECK(!expectedWorldSize.has_value() || *expectedWorldSize == rankConfigCount,
        errorPrefix + "rank_configs length (" + std::to_string(rankConfigCount)
            + ") does not match runtime world size (" + std::to_string(*expectedWorldSize) + ").");
    ELLM_CHECK(!rank.has_value() || (*rank >= 0 && *rank < rankConfigCount),
        errorPrefix + "requested rank " + std::to_string(*rank) + " is outside [0, " + std::to_string(rankConfigCount)
            + ").");

    std::vector<bool> seenRanks(static_cast<size_t>(rankConfigCount), false);
    Json selectedOverrides = Json::object();
    bool selectedRankFound = !rank.has_value();
    for (auto const& rankConfig : rankConfigs)
    {
        ELLM_CHECK(rankConfig.is_object() && rankConfig.contains("rank") && rankConfig["rank"].is_number_integer(),
            errorPrefix + "each rank_configs entry must be an object with an integer rank field.");
        int32_t const entryRank = rankConfig["rank"].get<int32_t>();
        ELLM_CHECK(entryRank >= 0 && entryRank < rankConfigCount,
            errorPrefix + "rank_configs rank " + std::to_string(entryRank) + " is outside [0, "
                + std::to_string(rankConfigCount) + ").");
        ELLM_CHECK(!seenRanks[static_cast<size_t>(entryRank)],
            errorPrefix + "rank_configs contains duplicate rank " + std::to_string(entryRank) + ".");
        seenRanks[static_cast<size_t>(entryRank)] = true;

        ELLM_CHECK(!rankConfig.contains("config_overrides") || rankConfig["config_overrides"].is_object(),
            errorPrefix + "rank_configs[" + std::to_string(entryRank)
                + "].config_overrides must be an object when present.");
        if (rank.has_value() && entryRank == *rank)
        {
            selectedRankFound = true;
            selectedOverrides = rankConfig.value("config_overrides", Json::object());
        }
    }

    ELLM_CHECK(selectedRankFound,
        errorPrefix + "config contains rank_configs but no entry for rank " + std::to_string(*rank) + ".");
    for (auto it = selectedOverrides.begin(); it != selectedOverrides.end(); ++it)
    {
        configJson[it.key()] = it.value();
    }
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
    cfg.numAttentionHeads = configJson.value("num_attention_heads", 0);
    cfg.indexerHeadDim = configJson.value("indexer_head_dim", 0);
    cfg.hiddenSize = getRequired<int32_t>(configJson, "hidden_size");

    // Top-level: kv_cache_dtype. Required — all current exports write this.
    parseRequiredStateDtype(configJson, "kv_cache_dtype", cfg.kvCacheDtype);
    if (configJson.contains("kv_cache_scales"))
    {
        auto const& scales = configJson.at("kv_cache_scales");
        ELLM_CHECK(scales.is_array(), "parseEngineConfig: kv_cache_scales must be an array");
        cfg.kvCacheScales.reserve(scales.size());
        for (size_t layer = 0; layer < scales.size(); ++layer)
        {
            auto const& pair = scales.at(layer);
            ELLM_CHECK(pair.is_array() && pair.size() == 2,
                "parseEngineConfig: kv_cache_scales entries must be [k_scale, v_scale] pairs");
            float const kScale = pair.at(0).get<float>();
            float const vScale = pair.at(1).get<float>();
            ELLM_CHECK(std::isfinite(kScale) && kScale > 0.0F && std::isfinite(vScale) && vScale > 0.0F,
                "parseEngineConfig: kv_cache_scales entries must be finite and positive");
            cfg.kvCacheScales.push_back({kScale, vScale});
        }
        ELLM_CHECK(cfg.kvCacheScales.size() == static_cast<size_t>(cfg.numDecoderLayers),
            "parseEngineConfig: kv_cache_scales length must match num_hidden_layers");
    }

    // builder_config: required batch / input / kv limits.
    ELLM_CHECK(configJson.contains("builder_config"), "parseEngineConfig: missing required 'builder_config' section");
    auto const& bc = configJson["builder_config"];
    cfg.maxSupportedBatchSize = getRequired<int32_t>(bc, "max_batch_size");
    cfg.maxSupportedInputLength = getRequired<int32_t>(bc, "max_input_len");
    cfg.maxKVCacheCapacity = getRequired<int32_t>(bc, "max_kv_cache_capacity");
    cfg.tpSize = bc.value("tp_size", 1);
    auto const compactExecution = bc.find("compact_execution");
    cfg.compactExecution
        = compactExecution != bc.end() && compactExecution->is_boolean() && compactExecution->get<bool>();
    auto const parseSkipSoftmaxScaleTable = [&configJson](char const* key, std::vector<std::array<int64_t, 2>>& table) {
        if (!configJson.contains(key))
        {
            return;
        }
        for (auto const& row : configJson[key])
        {
            ELLM_CHECK(row.is_array() && row.size() == 2,
                std::string("parseEngineConfig: ") + key + " rows must be [seqlen_bin, S] pairs");
            auto const bin = row.at(0).get<int64_t>();
            auto const scale = row.at(1).get<int64_t>();
            ELLM_CHECK(bin >= 0 && scale >= 0 && bin <= std::numeric_limits<int32_t>::max()
                    && scale <= std::numeric_limits<int32_t>::max(),
                std::string("parseEngineConfig: ") + key + " entries must be non-negative int32");
            table.push_back({bin, scale});
        }
        ELLM_CHECK(2 * table.size() <= static_cast<size_t>(kSKIP_SOFTMAX_SCALE_TABLE_MAX_LEN),
            std::string("parseEngineConfig: ") + key + " has more than "
                + std::to_string(kSKIP_SOFTMAX_SCALE_TABLE_MAX_LEN / 2) + " rows");
        std::sort(table.begin(), table.end());
    };
    parseSkipSoftmaxScaleTable("skip_softmax_prefill_scale_table", cfg.skipSoftmaxPrefillScaleTable);
    parseSkipSoftmaxScaleTable("skip_softmax_decode_scale_table", cfg.skipSoftmaxDecodeScaleTable);
    int64_t const numSwaPages = bc.value("num_swa_pages", static_cast<int64_t>(0));
    ELLM_CHECK(numSwaPages >= 0 && numSwaPages <= std::numeric_limits<int32_t>::max(),
        "parseEngineConfig: invalid num_swa_pages: " + std::to_string(numSwaPages)
            + " (must be a non-negative int32 value)");
    cfg.numSwaPages = static_cast<int32_t>(numSwaPages);

    std::string const backend = getRequired<std::string>(bc, "ragged_backend");
    ELLM_CHECK(backend == "entry_padded_compatibility",
        "parseEngineConfig: unsupported ragged backend '" + backend + "'; rebuild with entry_padded_compatibility.");
    cfg.raggedBackend = RaggedBackendKind::kEntryPaddedCompatibility;

    // RoPE configuration (top-level, derived from full config).
    cfg.ropeConfig = collectRopeConfig(configJson);

    // Positivity checks on required fields (common to both parsers).
    requirePositive(cfg.numDecoderLayers, "num_hidden_layers");
    requirePositive(cfg.numKVHeads, "num_key_value_heads");
    requirePositive(cfg.headDim, "head_dim");
    requirePositive(cfg.hiddenSize, "hidden_size");
    requirePositive(cfg.maxSupportedBatchSize, "max_batch_size");
    requirePositive(cfg.maxSupportedInputLength, "max_input_len");
    requirePositive(cfg.maxKVCacheCapacity, "max_kv_cache_capacity");
    ELLM_CHECK(cfg.maxKVCacheCapacity <= kMAX_KV_CACHE_CAPACITY,
        "parseEngineConfig: max_kv_cache_capacity (" + std::to_string(cfg.maxKVCacheCapacity)
            + ") exceeds the largest value that remains int32 after page alignment ("
            + std::to_string(kMAX_KV_CACHE_CAPACITY) + ")");
    int64_t const minimumActivePages = computeMinimumKvPoolPages(cfg.maxSupportedBatchSize, cfg.maxKVCacheCapacity);
    int64_t const serializedKvPoolPages = getRequired<int64_t>(bc, "max_kv_pool_pages");
    ELLM_CHECK(minimumActivePages <= kMAX_KV_POOL_PAGES,
        "parseEngineConfig: minimum active pages (" + std::to_string(minimumActivePages) + ")"
            + " exceeds the largest int32-addressable paged-KV pool " + std::to_string(kMAX_KV_POOL_PAGES) + ".");
    ELLM_CHECK(serializedKvPoolPages >= minimumActivePages,
        "parseEngineConfig: max_kv_pool_pages (" + std::to_string(serializedKvPoolPages)
            + ") cannot be smaller than the minimum active pages (" + std::to_string(minimumActivePages) + ")");
    ELLM_CHECK(serializedKvPoolPages <= kMAX_KV_POOL_PAGES,
        "parseEngineConfig: max_kv_pool_pages (" + std::to_string(serializedKvPoolPages)
            + ") exceeds the largest int32-addressable paged-KV pool (" + std::to_string(kMAX_KV_POOL_PAGES) + ")");
    cfg.kvPoolPages = static_cast<int32_t>(serializedKvPoolPages);
    ELLM_CHECK(cfg.maxSupportedInputLength <= cfg.maxKVCacheCapacity,
        "parseEngineConfig: max_input_len (" + std::to_string(cfg.maxSupportedInputLength)
            + ") cannot be greater than max_kv_cache_capacity (" + std::to_string(cfg.maxKVCacheCapacity) + ")");
}

void finalizeRaggedCapacities(LLMEngineConfig& cfg, int32_t roleQueryLength, char const* parserName)
{
    requirePositive(roleQueryLength, "role query length");
    cfg.maxNumSequences = cfg.maxSupportedBatchSize;
    cfg.recurrentPoolRows = cfg.maxSupportedBatchSize;
    cfg.maxQueryLength = std::max(cfg.maxSupportedInputLength, roleQueryLength);
    ELLM_CHECK(cfg.maxNumSequences <= std::numeric_limits<int32_t>::max() / cfg.maxQueryLength,
        std::string(parserName) + ": ragged physical-token capacity exceeds int32.");
    cfg.maxPhysicalTokens = cfg.maxNumSequences * cfg.maxQueryLength;
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
                int64_t kvCacheCapacity = 0;
                if (lc.contains("kv_cache_capacity"))
                {
                    ELLM_CHECK(lc["kv_cache_capacity"].is_number_integer(),
                        "parseEngineConfig: kv_layer_configs[" + std::to_string(i)
                            + "].kv_cache_capacity must be an integer");
                    kvCacheCapacity = lc["kv_cache_capacity"].get<int64_t>();
                }
                ELLM_CHECK(kvCacheCapacity >= 0 && kvCacheCapacity <= cfg.maxKVCacheCapacity,
                    "parseEngineConfig: kv_layer_configs[" + std::to_string(i) + "].kv_cache_capacity must be in [0, "
                        + std::to_string(cfg.maxKVCacheCapacity) + "]");
                cfg.kvLayerConfigs.push_back({lc["num_kv_heads"].get<int32_t>(), lc["head_dim"].get<int32_t>(),
                    static_cast<int32_t>(kvCacheCapacity)});
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

void validateKVLayerCapacities(Json const& configJson, LLMEngineConfig const& cfg)
{
    std::optional<int32_t> reducedCapacity;
    for (size_t i = 0; i < cfg.kvLayerConfigs.size(); ++i)
    {
        KVLayerConfig const& layerConfig = cfg.kvLayerConfigs[i];
        ELLM_CHECK(layerConfig.kvCacheCapacity >= 0 && layerConfig.kvCacheCapacity <= cfg.maxKVCacheCapacity,
            "parseEngineConfig: kv_layer_configs[" + std::to_string(i) + "].kv_cache_capacity must be in [0, "
                + std::to_string(cfg.maxKVCacheCapacity) + "]");
        int32_t const resolvedCapacity = resolveKvCacheCapacity(layerConfig.kvCacheCapacity, cfg.maxKVCacheCapacity);
        if (resolvedCapacity == cfg.maxKVCacheCapacity)
        {
            continue;
        }
        ELLM_CHECK(!reducedCapacity.has_value() || *reducedCapacity == resolvedCapacity,
            "parseEngineConfig: all reduced KV layers must use one common non-full kv_cache_capacity");
        reducedCapacity = resolvedCapacity;
    }

    if (!reducedCapacity.has_value())
    {
        return;
    }
    ELLM_CHECK(cfg.specDecodeType == SpecDecodeMode::kNONE && !configRevealsSpecDecode(configJson),
        "parseEngineConfig: reduced SWA KV pools do not support speculative decoding");
    int64_t const minimumSwaPages = computeMinimumSwaPoolPages(cfg.maxSupportedBatchSize, *reducedCapacity);
    ELLM_CHECK(minimumSwaPages <= std::numeric_limits<int32_t>::max() && cfg.numSwaPages >= minimumSwaPages,
        "parseEngineConfig: num_swa_pages must cover every active slot's private SWA pages");
}

//! Derive `cfg.layers` from the already-parsed per-layer arrays and global state geometry. Runs
//! after `kv_sharing_donors`, whose entries it folds in.
//!
//! Every shape here is one slot's row; the pool that holds it prepends its own slot count.
void buildLayerSpecs(LLMEngineConfig& cfg)
{
    // A compact `layer_types` carries one entry per cache-bearing layer rather than one per decoder
    // layer (Nemotron-H drops its mlp/moe layers from the array), so the loop's positional index is
    // not the absolute layer index and the derived table would be mis-indexed. Absolute indices
    // cannot be recovered from the compact array alone, so leave `cfg.layers` empty; the first table
    // consumer rejects an empty table rather than acting on wrong indices.
    if (cfg.layerTypes.size() != static_cast<size_t>(cfg.numDecoderLayers))
    {
        // An n-gram injection names absolute decoder-layer indices, which the compact array cannot
        // resolve, so its states have no table row to attach to and would be silently dropped. The
        // engine still binds them, so reject the combination rather than leaving the table empty.
        ELLM_CHECK(cfg.ngramLayers.empty(),
            "parseEngineConfig: ngram_layers cannot be placed on a compact layer_types stack (" +
                std::to_string(cfg.layerTypes.size()) + " cache-bearing entries for " +
                std::to_string(cfg.numDecoderLayers) + " decoder layers); the absolute injection site "
                "cannot be resolved into the derived layer table");
        cfg.layers.clear();
        return;
    }

    // Mamba2 is the only mixer with B/C groups, so a positive group count names the kind. Commit
    // mode (`recurrent_spec_verify_mode`) is deliberately not consulted: a GDN engine may carry
    // either mode, so it does not discriminate geometry.
    StateKind const mixerKind = cfg.recurrentStateNumGroups > 0 ? StateKind::kMamba2 : StateKind::kGdn;

    // `kvLayerConfigs` and `kvSharingDonors` are indexed below by attention rank rather than by
    // layer, so the arrays must agree on the attention count for that walk to stay in bounds.
    auto const numAttnLayers
        = std::count(cfg.layerTypes.begin(), cfg.layerTypes.end(), HybridCacheManager::LayerType::kAttention);
    ELLM_CHECK(numAttnLayers == static_cast<int64_t>(cfg.kvLayerConfigs.size()),
        "parseEngineConfig: kv_layer_configs supplies " + std::to_string(cfg.kvLayerConfigs.size())
            + " entries but layer_types declares " + std::to_string(numAttnLayers) + " attention layers");

    cfg.layers.clear();
    cfg.layers.reserve(cfg.layerTypes.size());
    int32_t localAttnIdx = 0;
    for (size_t absLayerIdx = 0; absLayerIdx < cfg.layerTypes.size(); ++absLayerIdx)
    {
        LayerSpec spec;
        spec.absLayerIdx = static_cast<int32_t>(absLayerIdx);
        if (cfg.layerTypes.at(absLayerIdx) == HybridCacheManager::LayerType::kAttention)
        {
            spec.kv = cfg.kvLayerConfigs.at(localAttnIdx);
            if (!cfg.kvSharingDonors.empty())
            {
                spec.kvSharingDonor = cfg.kvSharingDonors.at(localAttnIdx);
            }
            ++localAttnIdx;
        }
        else
        {
            spec.states.push_back({mixerKind, spec.absLayerIdx,
                Coords{cfg.recurrentStateNumHeads, cfg.recurrentStateHeadDim, cfg.recurrentStateSize},
                cfg.recurrentStateDtype});
            spec.states.push_back(
                {StateKind::kConv, spec.absLayerIdx, Coords{cfg.convDim, cfg.convKernel}, cfg.convStateDtype});
        }
        cfg.layers.push_back(std::move(spec));
    }

    // The injection may name a layer of any type, so this state attaches independently of the
    // mixer loop above and a layer may end up carrying it and nothing else.
    for (int32_t layer : cfg.ngramLayers)
    {
        ELLM_CHECK(layer >= 0 && layer < static_cast<int32_t>(cfg.layers.size()),
            "parseEngineConfig: ngram_layers names layer " + std::to_string(layer) + " outside the "
                + std::to_string(cfg.layers.size()) + " stateful decoder layers");
        LayerSpec& spec = cfg.layers.at(layer);
        ELLM_CHECK(spec.findState(StateKind::kNgramConv) == nullptr,
            "parseEngineConfig: ngram_layers names layer " + std::to_string(layer) + " twice");
        spec.states.push_back(
            {StateKind::kNgramConv, layer, Coords{cfg.ngramConvDim, cfg.ngramConvStateLen()}, cfg.ngramConvStateDtype});
    }
}

//! Parse a `layers[].states[].kind` token into its `StateKind`. The spelling is the inverse of
//! `stateKindName`, shared with the frontends that emit the schema.
StateKind parseStateKind(std::string const& token)
{
    if (token == "gdn")
    {
        return StateKind::kGdn;
    }
    if (token == "mamba2")
    {
        return StateKind::kMamba2;
    }
    if (token == "conv")
    {
        return StateKind::kConv;
    }
    if (token == "ngram_conv")
    {
        return StateKind::kNgramConv;
    }
    throw std::runtime_error(std::string("parseEngineConfig: invalid layers[].states[].kind '") + token
        + "'. Allowed values: gdn, mamba2, conv, ngram_conv. Re-export the engine.");
}

//! Guard a legacy scalar group against conflicting per-layer geometry. The group holds one shape and
//! dtype that `recurrentStateSpecs` broadcasts to every recurrent layer, so once a kind has been seen
//! any later layer declaring it must match or the earlier layers would be allocated the wrong buffer.
void requireConsistentStateGeometry(std::optional<StateSpec>& seen, StateSpec const& state)
{
    if (seen && (seen->shape != state.shape || seen->dataType != state.dataType))
    {
        throw std::runtime_error("parseEngineConfig: layer " + std::to_string(state.absLayerIdx) + " declares a '"
            + stateKindName(state.kind) + "' state (shape " + state.shape.formatString() + ", dtype "
            + std::to_string(static_cast<int32_t>(state.dataType)) + ") that disagrees with layer "
            + std::to_string(seen->absLayerIdx) + "'s (shape " + seen->shape.formatString() + ", dtype "
            + std::to_string(static_cast<int32_t>(seen->dataType))
            + "); the runtime allocates one buffer geometry per state kind, so every recurrent layer feeding a kind"
            + " must declare the same shape and dtype");
    }
    seen = state;
}

//! Restate the parsed `cfg.layers` table as the legacy scalar/array members the rest of the runtime
//! still reads. This is the inverse of `buildLayerSpecs`: when `layers` is present it is the source
//! of truth and these members are projected back out of it, so no consumer outside the parser has to
//! change in this increment.
//!
//! `recurrent_state_num_groups`, the n-gram feature scalars (`ngram_embed_dim`/`ngram_conv_kernel`/
//! `ngram_size`/`ngram_heads_per_order`), and the spec-verify commit mode stay top-level and are
//! parsed separately; only the per-layer geometry that lives in the table is derived here.
void deriveLegacyLayerMembers(LLMEngineConfig& cfg)
{
    cfg.layerTypes.clear();
    cfg.layerTypes.reserve(cfg.layers.size());
    cfg.kvLayerConfigs.clear();
    cfg.kvSharingDonors.clear();
    cfg.ngramLayers.clear();

    std::vector<int32_t> donors;
    bool anyDonor = false;

    // Each legacy scalar group below holds ONE geometry that `recurrentStateSpecs` broadcasts to
    // every recurrent layer, so capturing only the last writer silently mis-sizes earlier layers
    // whose declared state differs. Until the runtime consumes per-layer geometry, require every
    // layer feeding a group to agree on shape and dtype. kGdn and kMamba2 share one group because
    // they write the same `recurrentState*` scalars.
    std::optional<StateSpec> recurrentSeen;
    std::optional<StateSpec> convSeen;
    std::optional<StateSpec> ngramSeen;

    for (LayerSpec const& spec : cfg.layers)
    {
        // A standalone MLP/MoE layer attends to nothing and carries no recurrent state. It drops out
        // of the compact routing table entirely so the hybrid cache manager, which indexes
        // `layerTypes` as the stateful stack, never reserves a KV or mamba slot for it.
        if (spec.isAttention())
        {
            cfg.layerTypes.push_back(HybridCacheManager::LayerType::kAttention);
            cfg.kvLayerConfigs.push_back(*spec.kv);
            donors.push_back(spec.kvSharingDonor);
            anyDonor = anyDonor || spec.kvSharingDonor >= 0;
        }
        else if (spec.isRecurrent())
        {
            // An n-gram injection riding a bare MLP/MoE layer carries only kNgramConv, which is not a
            // mixer state; it still rides the row pool but claims no recurrent slot, so route on the
            // mixer state rather than on "not attention".
            cfg.layerTypes.push_back(HybridCacheManager::LayerType::kMamba);
        }
        for (StateSpec const& state : spec.states)
        {
            switch (state.kind)
            {
            case StateKind::kGdn:
            case StateKind::kMamba2:
                // The stack is single-lineage: every mixer layer feeds the one `recurrentState*`
                // group, which a Mamba2 vs. GDN layer would size and commit differently. Reject a
                // table that mixes the two kinds across layers (the per-layer check already bars both
                // on one layer).
                ELLM_CHECK(!recurrentSeen || recurrentSeen->kind == state.kind,
                    "parseEngineConfig: layer " + std::to_string(state.absLayerIdx) + " declares a '"
                        + stateKindName(state.kind) + "' mixer but layer " + std::to_string(recurrentSeen->absLayerIdx)
                        + " declared a '" + stateKindName(recurrentSeen->kind)
                        + "'; the recurrent stack is single-lineage (all gdn or all mamba2)");
                requireConsistentStateGeometry(recurrentSeen, state);
                cfg.recurrentStateNumHeads = static_cast<int32_t>(state.shape[0]);
                cfg.recurrentStateHeadDim = static_cast<int32_t>(state.shape[1]);
                cfg.recurrentStateSize = static_cast<int32_t>(state.shape[2]);
                cfg.recurrentStateDtype = state.dataType;
                break;
            case StateKind::kConv:
                requireConsistentStateGeometry(convSeen, state);
                cfg.convDim = static_cast<int32_t>(state.shape[0]);
                cfg.convKernel = static_cast<int32_t>(state.shape[1]);
                cfg.convStateDtype = state.dataType;
                break;
            case StateKind::kNgramConv:
                requireConsistentStateGeometry(ngramSeen, state);
                cfg.ngramLayers.push_back(spec.absLayerIdx);
                cfg.ngramConvDim = static_cast<int32_t>(state.shape[0]);
                cfg.ngramConvStateDtype = state.dataType;
                break;
            }
        }
    }

    cfg.numAttentionLayers = static_cast<int32_t>(cfg.kvLayerConfigs.size());
    cfg.numLinearAttnLayers = static_cast<int32_t>(cfg.layerTypes.size()) - cfg.numAttentionLayers;
    // Keep the donor array empty when no layer shares KV, matching the legacy "absent field" state
    // the `.empty()` consumers test for.
    if (anyDonor)
    {
        cfg.kvSharingDonors = std::move(donors);
    }
}

//! Parse the top-level `layers` array — the single source of truth for per-layer KV and recurrent
//! state geometry when present — into `cfg.layers`, then project the legacy members out of it. Each
//! entry carries its absolute index, an optional `kv` block (present iff the layer attends, with an
//! optional `kv_sharing_donor`), and an ordered `states` list whose order is consumed verbatim.
void parseLayerSpecs(Json const& configJson, LLMEngineConfig& cfg)
{
    auto const& layersJson = configJson.at("layers");
    ELLM_CHECK(layersJson.is_array() && !layersJson.empty(), "parseEngineConfig: 'layers' must be a non-empty array.");
    ELLM_CHECK(static_cast<int32_t>(layersJson.size()) == cfg.numDecoderLayers,
        "parseEngineConfig: 'layers' length (" + std::to_string(layersJson.size()) + ") must equal num_hidden_layers ("
            + std::to_string(cfg.numDecoderLayers) + ")");

    cfg.layers.clear();
    cfg.layers.reserve(layersJson.size());
    for (size_t pos = 0; pos < layersJson.size(); ++pos)
    {
        auto const& entry = layersJson.at(pos);
        LayerSpec spec;
        spec.absLayerIdx = entry.value("idx", static_cast<int32_t>(pos));
        ELLM_CHECK(spec.absLayerIdx == static_cast<int32_t>(pos),
            "parseEngineConfig: layers[" + std::to_string(pos) + "].idx (" + std::to_string(spec.absLayerIdx)
                + ") must equal its position; the table is dense over decoder layers in absolute order");

        if (entry.contains("kv") && !entry.at("kv").is_null())
        {
            auto const& kv = entry.at("kv");
            ELLM_CHECK(kv.contains("num_kv_heads") && kv.contains("head_dim"),
                "parseEngineConfig: layers[" + std::to_string(pos) + "].kv missing num_kv_heads/head_dim");
            int64_t kvCacheCapacity = 0;
            if (kv.contains("kv_cache_capacity"))
            {
                // Mirror the legacy `kv_layer_configs` guard: a non-integer capacity (e.g. a float or
                // string) would otherwise be coerced silently and mis-size the per-layer cache.
                ELLM_CHECK(kv.at("kv_cache_capacity").is_number_integer(),
                    "parseEngineConfig: layers[" + std::to_string(pos) + "].kv.kv_cache_capacity must be an integer");
                kvCacheCapacity = kv.at("kv_cache_capacity").get<int64_t>();
            }
            ELLM_CHECK(kvCacheCapacity >= 0 && kvCacheCapacity <= cfg.maxKVCacheCapacity,
                "parseEngineConfig: layers[" + std::to_string(pos) + "].kv.kv_cache_capacity must be in [0, "
                    + std::to_string(cfg.maxKVCacheCapacity) + "]");
            spec.kv = KVLayerConfig{kv.at("num_kv_heads").get<int32_t>(), kv.at("head_dim").get<int32_t>(),
                static_cast<int32_t>(kvCacheCapacity)};
            spec.kvSharingDonor = kv.value("kv_sharing_donor", -1);
        }

        if (entry.contains("states"))
        {
            auto const& statesJson = entry.at("states");
            ELLM_CHECK(statesJson.is_array(),
                "parseEngineConfig: layers[" + std::to_string(pos) + "].states must be an array");
            for (auto const& stateJson : statesJson)
            {
                ELLM_CHECK(stateJson.contains("kind") && stateJson.contains("shape") && stateJson.contains("dtype"),
                    "parseEngineConfig: layers[" + std::to_string(pos)
                        + "].states[] entries require kind, shape, and dtype");
                StateKind const kind = parseStateKind(stateJson.at("kind").get<std::string>());
                auto const& shapeJson = stateJson.at("shape");
                ELLM_CHECK(shapeJson.is_array() && !shapeJson.empty(),
                    "parseEngineConfig: layers[" + std::to_string(pos) + "].states[].shape must be a non-empty array");
                std::vector<int64_t> dims;
                dims.reserve(shapeJson.size());
                for (auto const& dim : shapeJson)
                {
                    dims.push_back(dim.get<int64_t>());
                }
                // Per-kind rank contract: a mixer (gdn/mamba2) carries a 3-D [heads, head_dim,
                // state_size] slot row; a dilated conv (recurrent conv or ngram conv) carries a 2-D
                // [channels, width] row. Each dim is a buffer extent the runtime allocates, so none
                // may be non-positive.
                int32_t const expectedRank = (kind == StateKind::kGdn || kind == StateKind::kMamba2) ? 3 : 2;
                ELLM_CHECK(static_cast<int32_t>(dims.size()) == expectedRank,
                    "parseEngineConfig: layers[" + std::to_string(pos) + "].states[] kind '" + stateKindName(kind)
                        + "' requires a " + std::to_string(expectedRank) + "-D shape, got "
                        + std::to_string(dims.size()) + "-D");
                for (int64_t dim : dims)
                {
                    ELLM_CHECK(dim > 0,
                        "parseEngineConfig: layers[" + std::to_string(pos) + "].states[] kind '" + stateKindName(kind)
                            + "' has a non-positive dim " + std::to_string(dim));
                }
                trt_edgellm::DataType const dtype
                    = parseStateDtype(stateJson.at("dtype").get<std::string>(), "layers[].states[].dtype");
                ELLM_CHECK(spec.findState(kind) == nullptr,
                    "parseEngineConfig: layers[" + std::to_string(pos) + "] declares kind '" + stateKindName(kind)
                        + "' twice");
                spec.states.push_back({kind, spec.absLayerIdx, Coords{dims}, dtype});
            }
        }

        // A recurrent mixer layer binds its mixer state and its dilated conv state as a pair: the
        // runtime allocates and commits them together, so one without the other is malformed. A layer
        // carries at most one mixer kind (gdn XOR mamba2); the stack is single-lineage. (`kNgramConv`
        // is an injection riding the row pool, not a mixer, so it is exempt from the pairing rule.)
        StateSpec const* gdn = spec.findState(StateKind::kGdn);
        StateSpec const* mamba2 = spec.findState(StateKind::kMamba2);
        StateSpec const* conv = spec.findState(StateKind::kConv);
        ELLM_CHECK(!(gdn != nullptr && mamba2 != nullptr),
            "parseEngineConfig: layers[" + std::to_string(pos)
                + "] declares both gdn and mamba2 mixer states; a layer carries at most one mixer kind");
        bool const hasMixer = (gdn != nullptr || mamba2 != nullptr);
        ELLM_CHECK(hasMixer == (conv != nullptr),
            "parseEngineConfig: layers[" + std::to_string(pos)
                + "] must pair a mixer state (gdn/mamba2) with a conv state; declare neither or both");

        cfg.layers.push_back(std::move(spec));
    }

    // A `kv_sharing_donor` names the LOCAL attention index whose KV cache this layer reuses, so it is
    // only meaningful once the attention layers are counted. Apply the legacy rule (see the
    // `kv_sharing_donors` block in parseEngineConfig): -1 to own its cache, otherwise a different
    // attention layer in range. Recurrent layers carry no KV, so a donor on one is malformed.
    int32_t localAttnIdx = 0;
    for (LayerSpec const& spec : cfg.layers)
    {
        if (spec.isAttention())
        {
            ++localAttnIdx;
        }
    }
    int32_t const numAttn = localAttnIdx;
    localAttnIdx = 0;
    for (LayerSpec const& spec : cfg.layers)
    {
        if (!spec.isAttention())
        {
            continue;
        }
        int32_t const donor = spec.kvSharingDonor;
        ELLM_CHECK(donor == -1 || (donor >= 0 && donor < numAttn && donor != localAttnIdx),
            "parseEngineConfig: layers[" + std::to_string(spec.absLayerIdx) + "].kv.kv_sharing_donor = "
                + std::to_string(donor) + " is invalid (must be -1 or a different attention layer in [0, "
                + std::to_string(numAttn - 1) + "])");
        ++localAttnIdx;
    }

    deriveLegacyLayerMembers(cfg);
}

} // namespace

LLMEngineConfig parseEngineConfig(
    std::filesystem::path const& configPath, std::optional<int32_t> rank, std::optional<int32_t> expectedWorldSize)
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
    validateAndApplyRankConfigOverrides(configJson, rank, expectedWorldSize, "parseEngineConfig");

    // Optional version check.
    std::string const modelVersion = configJson.value(binding_names::kEdgellmVersion, "");
    version::checkVersion(modelVersion);

    LLMEngineConfig cfg;

    cfg.specDecodeType = parseSpecDecodeMode(configJson);
    std::string const engineRole = parseEngineRole(configJson);
    ELLM_CHECK(engineRole != "draft", "parseEngineConfig: use parseDraftEngineConfig for engine_role=draft.");
    cfg.isSpecDecodeBase = (engineRole == "base");
    cfg.isDiffusionBackbone = (engineRole == "dllm");
    if (cfg.isSpecDecodeBase)
    {
        ELLM_CHECK(cfg.specDecodeType != SpecDecodeMode::kNONE,
            "parseEngineConfig: engine_role=base requires spec_decode_type to be mtp, eagle3, dflash, "
            "jetspec, "
            "dspark, domino, or gemma4_mtp.");
    }
    else
    {
        ELLM_CHECK(cfg.specDecodeType == SpecDecodeMode::kNONE,
            "parseEngineConfig: non-speculative engine roles require spec_decode_type=none.");
    }

    if (cfg.isDiffusionBackbone)
    {
        Json const empty = Json::object();
        Json const& diffusionConfig = configJson.contains("diffusion_config") ? configJson["diffusion_config"] : empty;
        cfg.contextMaskSelectorEnabled = configJson.value("context_mask_selector_enabled", true);
        cfg.diffusionUnifiedConditioning = configJson.value("diffusion_unified_conditioning", false);
        cfg.diffusionCanvasLength = diffusionConfig.value("canvas_length", configJson.value("canvas_length", 256));
        cfg.diffusionMaxDenoisingSteps
            = diffusionConfig.value("max_denoising_steps", configJson.value("max_denoising_steps", 48));
        cfg.diffusionSelfConditioningSize = configJson.value("self_conditioning_size", 0);
        cfg.diffusionTMax = diffusionConfig.value("t_max", 0.8F);
        cfg.diffusionTMin = diffusionConfig.value("t_min", 0.4F);
        cfg.diffusionEntropyBound = diffusionConfig.value("entropy_bound", 0.1F);
        cfg.diffusionEntropyThreshold = diffusionConfig.value("entropy_threshold", 0.005F);
        cfg.rmsNormEps = configJson.value("rms_norm_eps", 1.0e-6F);
        cfg.diffusionStabilityWindow = diffusionConfig.value("stability_window", 2);
        requirePositive(cfg.diffusionCanvasLength, "diffusion canvas_length");
        requirePositive(cfg.diffusionMaxDenoisingSteps, "diffusion max_denoising_steps");
        requirePositive(cfg.diffusionStabilityWindow, "diffusion stability_window");
        ELLM_CHECK(cfg.diffusionUnifiedConditioning,
            "parseEngineConfig: DiffusionGemma dllm engines require diffusion_unified_conditioning=true. Re-export the "
            "model with the latest DiffusionGemma exporter and rebuild the engine.");
        ELLM_CHECK(cfg.diffusionTMax > 0.0F && cfg.diffusionTMin > 0.0F,
            "parseEngineConfig: diffusion t_min/t_max must be positive");
        ELLM_CHECK(cfg.diffusionEntropyBound >= 0.0F && cfg.diffusionEntropyThreshold >= 0.0F,
            "parseEngineConfig: diffusion entropy thresholds must be non-negative");
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

    cfg.numDeepstackFeatures = cfg.isDiffusionBackbone ? 0 : configJson.value("num_deepstack_features", 0);

    // The per-layer `layers` table, when present, is the single source of truth for KV and
    // recurrent-state geometry — including the n-gram injection sites and conv-state dtype that
    // `parsePleConfig` consumes next. Parse it up front so those sites are populated before the
    // n-gram feature metadata and shared row pool are validated; a pre-table engine instead lists
    // the sites at the top level, which `parsePleConfig` reads on the no-table path. This is the one
    // fork that selects the table vs. legacy-scalar geometry for the whole parse.
    bool const hasLayerTable = configJson.contains("layers");
    if (hasLayerTable)
    {
        parseLayerSpecs(configJson, cfg);
    }

    parsePleConfig(configJson, cfg, hasLayerTable);
    cfg.audioTokenId = configJson.value("audio_token_id", -1);
    cfg.imageTokenId = configJson.value("image_token_id", -1);
    bool const usesMultimodalPle = cfg.gemmaPleEnabled && (cfg.audioTokenId >= 0 || cfg.imageTokenId >= 0);
    cfg.padTokenId
        = usesMultimodalPle ? getRequired<int32_t>(configJson, "pad_token_id") : configJson.value("pad_token_id", -1);
    cfg.useVisionBidirectionalAttention = configJson.value("use_vision_bidirectional_attention", false);
    ELLM_CHECK(!cfg.useVisionBidirectionalAttention || cfg.imageTokenId >= 0,
        "use_vision_bidirectional_attention requires image_token_id in the LLM config.");

    // `recurrent_state_num_groups` and the spec-verify commit mode are not part of the per-layer
    // table, so they are read from the top level on both paths. The per-layer counts and recurrent/
    // conv geometry below, by contrast, are derived from the table when one is present (see
    // `deriveLegacyLayerMembers`), so a table engine must not re-read the stale legacy scalars.
    cfg.recurrentStateNumGroups = configJson.value("recurrent_state_num_groups", 0);
    // The serialized mode selects Mamba factorized replay storage or GDN chunk storage.
    std::string const recurrentSpecVerifyMode = configJson.value("recurrent_spec_verify_mode", std::string{"snapshot"});
    ELLM_CHECK(recurrentSpecVerifyMode == "replay" || recurrentSpecVerifyMode == "snapshot",
        "parseEngineConfig: invalid recurrent_spec_verify_mode '" + recurrentSpecVerifyMode
            + "'; expected \"replay\" or \"snapshot\".");
    cfg.recurrentSpecVerifyUsesReplay = (recurrentSpecVerifyMode == "replay");
    if (!hasLayerTable)
    {
        cfg.numLinearAttnLayers = configJson.value("num_linear_attn_layers", 0);
        cfg.numAttentionLayers = configJson.value("num_attention_layers", cfg.numDecoderLayers);
        cfg.recurrentStateNumHeads = configJson.value("recurrent_state_num_heads", 0);
        cfg.recurrentStateHeadDim = configJson.value("recurrent_state_head_dim", 0);
        cfg.recurrentStateSize = configJson.value("recurrent_state_size", 0);
        cfg.convDim = configJson.value("conv_dim", 0);
        cfg.convKernel = configJson.value("conv_kernel", 0);
    }
    parseDFlashFields(configJson, cfg, cfg.numDecoderLayers);
    parseJetSpecFields(configJson, cfg, cfg.numDecoderLayers);
    parseDominoFields(configJson, cfg, cfg.numDecoderLayers);
    parseDSparkFields(configJson, cfg, cfg.numDecoderLayers);
    if (cfg.isSpecDecodeBase && cfg.specDecodeType == SpecDecodeMode::kEAGLE
        && configJson.contains("eagle_hidden_state_layers"))
    {
        Json const& layerIds = configJson["eagle_hidden_state_layers"];
        ELLM_CHECK(layerIds.is_array(), "parseEngineConfig: eagle_hidden_state_layers must be an array when present.");
        for (auto const& layerId : layerIds)
        {
            cfg.specTargetLayerIds.push_back(layerId.get<int32_t>());
        }
    }

    auto const& bc = configJson["builder_config"];
    cfg.maxSupportedLoraRank = bc.value("max_lora_rank", 0);

    // Recurrent / conv state dtypes are only meaningful for hybrid engines
    // (Mamba / Nemotron-H / GDN). On a pre-table engine they are weight-export
    // properties written to the top level by the Python export step, present
    // strictly iff the stack has a linear-attention (recurrent) layer. A table
    // engine instead carries each recurrent layer's dtype inside its state
    // entries, which `deriveLegacyLayerMembers` has already projected into these
    // members, so the top-level fields are neither required nor read.
    if (!hasLayerTable && cfg.numLinearAttnLayers > 0)
    {
        parseRequiredStateDtype(configJson, "recurrent_state_dtype", cfg.recurrentStateDtype);
        parseRequiredStateDtype(configJson, "conv_state_dtype", cfg.convStateDtype);
    }

    // Base-specific positivity checks (beyond parseCoreFields's core set).
    requirePositive(cfg.rotaryDim, "rotary_dim");
    requirePositive(cfg.vocabSize, "vocab_size");
    if (usesMultimodalPle)
    {
        ELLM_CHECK(cfg.padTokenId >= 0 && cfg.padTokenId < cfg.plePool.rowCount,
            "parseEngineConfig: pad_token_id must be within the PLE vocabulary for multimodal Gemma4");
    }
    ELLM_CHECK(cfg.maxSupportedLoraRank >= 0,
        "parseEngineConfig: invalid max_lora_rank: " + std::to_string(cfg.maxSupportedLoraRank)
            + " (must be non-negative)");

    // --- SpecDecode base model ---
    // `max_spec_verify_tokens` is the base engine's own input budget (the
    // seq_len it accepts for proposal verification), so it must be present
    // in the base's builder_config. `max_spec_draft_tokens` is the draft
    // engine's input budget and lives in draft_config.json; the builder
    // correctly omits it from base_config.json (see llmBuilder.h), so
    // `cfg.maxSpecDraftTokens` stays at 0 on the base — consumers read the
    // authoritative value from the draft engine config (or from
    // `DeploymentConfig::specDecode`, which consolidates both sides).
    if (cfg.isSpecDecodeBase)
    {
        cfg.maxSpecVerifyTokens = getRequired<int32_t>(bc, "max_spec_verify_tokens");
        requirePositive(cfg.maxSpecVerifyTokens, "max_spec_verify_tokens");
        if (cfg.specDecodeType == SpecDecodeMode::kGemma4MTP)
        {
            ELLM_CHECK(cfg.modelType == "gemma4" || cfg.modelType == "gemma4_text" || cfg.modelType == "gemma4_unified"
                    || cfg.modelType == "gemma4_unified_text",
                "parseEngineConfig: gemma4_mtp base config must identify a Gemma4 target model.");
            ELLM_CHECK(cfg.baseModelHiddenSize == 0 || cfg.baseModelHiddenSize == cfg.hiddenSize,
                "parseEngineConfig: gemma4_mtp base_model_hidden_size must match hidden_size.");
        }
    }

    if (cfg.isDiffusionBackbone)
    {
        finalizeRaggedCapacities(cfg, cfg.diffusionCanvasLength, "parseEngineConfig");
    }
    else
    {
        finalizeRaggedCapacities(cfg, cfg.isSpecDecodeBase ? cfg.maxSpecVerifyTokens : 1, "parseEngineConfig");
    }

    // A table engine already parsed its `layers` table up front (before `parsePleConfig`), projecting
    // the per-layer routing out of it. A pre-table engine derives that routing here from the canonical
    // `layer_types`/`kv_layer_configs` fields or the scalar-broadcast fallback.
    if (!hasLayerTable)
    {
        populateLayerTypes(configJson, cfg);
    }
    validateKVLayerCapacities(configJson, cfg);
    parseDualRopeFields(configJson, cfg);

    if (!hasLayerTable)
    {
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

        buildLayerSpecs(cfg);
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
        "parseDraftEngineConfig: engine_role=draft requires spec_decode_type to be mtp, eagle3, dflash, "
        "jetspec, "
        "dspark, domino, or gemma4_mtp.");

    // Shared core fields (layers, kv heads, head_dim, hidden_size, kv_cache_dtype,
    // batch/input/kv limits, RoPE, common positivity checks).
    parseCoreFields(configJson, cfg);
    parseGemma4MTPFields(configJson, cfg);

    // --- Draft-specific ---
    // A hybrid draft (e.g. Nemotron-H MTP: attention + MoE) has fewer KV-bearing
    // attention layers than total layers, so honor an explicit
    // ``num_attention_layers`` (matches parseEngineConfig); default to the
    // decoder-layer count for single-type drafts that omit it. A table draft
    // instead derives the count from its `layers` table below, so the stale
    // top-level scalar is not read.
    bool const hasLayerTable = configJson.contains("layers");
    if (!hasLayerTable)
    {
        cfg.numAttentionLayers = configJson.value("num_attention_layers", cfg.numDecoderLayers);
    }
    // Match the engine's `rope_rotary_cos_sin` binding shape. Most partial
    // rotary models expose a smaller binding via `partial_rotary_factor`, while
    // proportional RoPE keeps a headDim-sized binding and treats the non-rotated
    // tail as identity.
    cfg.rotaryDim = static_cast<int32_t>(getRotaryDim(configJson, cfg.headDim));
    cfg.vocabSize = configJson.value("draft_vocab_size", configJson.value("vocab_size", 0));
    cfg.reducedVocabSize = configJson.value(binding_names::kReducedVocabSizeKey, 0);
    cfg.outputVocabSize = (cfg.reducedVocabSize > 0) ? cfg.reducedVocabSize : cfg.vocabSize;
    parseDFlashFields(configJson, cfg);
    parseJetSpecFields(configJson, cfg);
    parseEagleFields(configJson, cfg);
    parseDominoFields(configJson, cfg);
    parseDSparkFields(configJson, cfg);

    // Draft engines do not own speculative base verification bindings.
    cfg.isSpecDecodeBase = false;

    auto const& bc = configJson["builder_config"];
    // Symmetric to the base side (see parseEngineConfig): each engine's
    // builder_config carries only its own sequence budget. The base emits
    // `max_spec_verify_tokens` (its verification budget); the draft emits
    // `max_spec_draft_tokens` (its proposal/generation budget). So the draft
    // parser only requires `max_spec_draft_tokens`; `cfg.maxSpecVerifyTokens`
    // stays at 0 on the draft — consumers read the consolidated values
    // from `DeploymentConfig::specDecode`.
    cfg.maxSpecDraftTokens = getRequired<int32_t>(bc, "max_spec_draft_tokens");
    requirePositive(cfg.maxSpecDraftTokens, "max_spec_draft_tokens");
    finalizeRaggedCapacities(cfg, cfg.maxSpecDraftTokens, "parseDraftEngineConfig");

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
        "maxBatch=%d maxInputLen=%d maxKVCap=%d maxSpecDraftTokens=%d baseModelHiddenSize=%d",
        cfg.hiddenSize, cfg.vocabSize, cfg.numDecoderLayers, cfg.numKVHeads, cfg.headDim, cfg.maxSupportedBatchSize,
        cfg.maxSupportedInputLength, cfg.maxKVCacheCapacity, cfg.maxSpecDraftTokens, cfg.baseModelHiddenSize);

    // The per-layer `layers` table, when present, is the single source of truth for KV and
    // recurrent-state geometry (a hybrid draft derives its attention/linear layer counts from it);
    // drafts that predate the table fall back to the scalar-broadcast derivation.
    if (hasLayerTable)
    {
        parseLayerSpecs(configJson, cfg);
    }
    else
    {
        // Populate per-layer type routing from canonical fields or scalar fallback.
        populateLayerTypes(configJson, cfg);
    }
    validateKVLayerCapacities(configJson, cfg);
    parseDualRopeFields(configJson, cfg);
    if (!hasLayerTable)
    {
        buildLayerSpecs(cfg);
    }

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
       << " kvPoolPages=" << cfg.kvPoolPages << " numSwaPages=" << cfg.numSwaPages
       << " swaKVCacheMode=" << (cfg.usesBoundedSwaKVCache() ? "bounded" : "full")
       << " gemmaPleEnabled=" << cfg.gemmaPleEnabled << " gemmaPleLayerCount=" << cfg.gemmaPleLayerCount
       << " gemmaPleHiddenSize=" << cfg.gemmaPleHiddenSize << " padTokenId=" << cfg.padTokenId
       << " isSpecDecodeBase=" << cfg.isSpecDecodeBase << " specDecodeType=" << static_cast<int>(cfg.specDecodeType)
       << " loraRank=" << cfg.maxSupportedLoraRank;
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
    if (cfg.ngramEnabled())
    {
        ss << " ngramLayers=" << cfg.ngramLayers.size()
           << " ngramConvStateDtype=" << static_cast<int>(cfg.ngramConvStateDtype);
    }
    if (cfg.maxSpecVerifyTokens > 0)
    {
        ss << " maxSpecVerifyTokens=" << cfg.maxSpecVerifyTokens;
    }
    // Only draft engines carry a meaningful `maxSpecDraftTokens`; the base side
    // leaves it at 0 (see parseEngineConfig comment), so skip the noise.
    if (cfg.maxSpecDraftTokens > 0)
    {
        ss << " maxSpecDraftTokens=" << cfg.maxSpecDraftTokens;
    }
    if (isCachedBlockDraftMode(cfg.specDecodeType) || cfg.specDecodeType == SpecDecodeMode::kDSpark)
    {
        ss << " specDraftBlockSize=" << cfg.specDraftBlockSize << " specDraftMaskTokenId=" << cfg.specDraftMaskTokenId
           << " specTargetLayerIds=" << cfg.specTargetLayerIds.size();
    }
    if (cfg.specDecodeType == SpecDecodeMode::kJetSpec)
    {
        ss << " specDraftCausalHead=" << cfg.specDraftCausalHead;
    }
    if (cfg.specDecodeType == SpecDecodeMode::kDomino)
    {
        ss << " dominoGruHiddenDim=" << cfg.dominoGruHiddenDim << " dominoProjDim=" << cfg.dominoProjDim
           << " dominoPoolSize=" << cfg.dominoPoolSize;
    }
    if (cfg.specDecodeType == SpecDecodeMode::kDSpark)
    {
        ss << " dsparkMarkovHeadType=" << cfg.dsparkMarkovHeadType << " dsparkMarkovRank=" << cfg.dsparkMarkovRank
           << " dsparkConfidence=" << cfg.dsparkEnableConfidenceHead;
    }
    if (!cfg.eosTokenIds.empty())
    {
        ss << " eosTokenIds=[";
        for (size_t i = 0; i < cfg.eosTokenIds.size(); ++i)
        {
            if (i > 0)
            {
                ss << ",";
            }
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
    if (cfg.isDiffusionBackbone)
    {
        ss << " diffusionCanvasLength=" << cfg.diffusionCanvasLength
           << " diffusionMaxDenoisingSteps=" << cfg.diffusionMaxDenoisingSteps << " diffusionTMin=" << cfg.diffusionTMin
           << " diffusionTMax=" << cfg.diffusionTMax << " diffusionEntropyBound=" << cfg.diffusionEntropyBound
           << " diffusionEntropyThreshold=" << cfg.diffusionEntropyThreshold << " rmsNormEps=" << cfg.rmsNormEps
           << " diffusionStabilityWindow=" << cfg.diffusionStabilityWindow
           << " diffusionUnifiedConditioning=" << cfg.diffusionUnifiedConditioning;
    }
    ss << " layers=[";
    for (size_t i = 0; i < cfg.layers.size(); ++i)
    {
        if (i > 0)
        {
            ss << ",";
        }
        LayerSpec const& layer = cfg.layers[i];
        ss << layer.absLayerIdx << ":";
        if (layer.isAttention())
        {
            ss << "kv(" << layer.kv->numKVHeads << "x" << layer.kv->headDim << ",cap=" << layer.kv->kvCacheCapacity
               << ")";
            if (layer.kvSharingDonor >= 0)
            {
                ss << "<-" << layer.kvSharingDonor;
            }
        }
        for (StateSpec const& state : layer.states)
        {
            ss << "+" << stateKindName(state.kind) << state.shape.formatString();
        }
    }
    ss << "]";
    ss << " }";
    return ss.str();
}

// C++17 requires positional aggregate initialization; field labels keep each
// dimension recipe auditable when InferenceDims changes.

bool LayerSpec::isRecurrent() const noexcept
{
    return std::any_of(
        states.begin(), states.end(), [](StateSpec const& state) { return state.kind != StateKind::kNgramConv; });
}

StateSpec const* LayerSpec::findState(StateKind kind) const
{
    auto const match
        = std::find_if(states.begin(), states.end(), [kind](StateSpec const& state) { return state.kind == kind; });
    return match == states.end() ? nullptr : &*match;
}

bool LLMEngineConfig::supportsBoundedSwaKVCache() const
{
    return std::any_of(kvLayerConfigs.begin(), kvLayerConfigs.end(), [&](KVLayerConfig const& layerConfig) {
        return isReducedKvCacheCapacity(layerConfig.kvCacheCapacity, maxKVCacheCapacity);
    });
}

bool LLMEngineConfig::usesBoundedSwaKVCache() const
{
    return swaKVCacheMode == SwaKVCacheMode::kBounded && supportsBoundedSwaKVCache();
}

void LLMEngineConfig::setSwaKVCacheMode(SwaKVCacheMode mode) noexcept
{
    swaKVCacheMode = mode;
}

int32_t LLMEngineConfig::getSwaKVCacheModeInputLength() const
{
    return usesBoundedSwaKVCache() ? 1 : 0;
}

int32_t LLMEngineConfig::getBoundedKVPoolPagesForLayer(KVLayerConfig const& layerConfig) const
{
    if (!isReducedKvCacheCapacity(layerConfig.kvCacheCapacity, maxKVCacheCapacity))
    {
        return kvPoolPages;
    }
    return numSwaPages;
}

int32_t LLMEngineConfig::getKVPoolPagesForLayer(KVLayerConfig const& layerConfig) const
{
    return usesBoundedSwaKVCache() ? getBoundedKVPoolPagesForLayer(layerConfig) : kvPoolPages;
}

std::array<int32_t, 3> LLMEngineConfig::getKVPoolPageProfileForLayer(KVLayerConfig const& layerConfig) const
{
    if (!isReducedKvCacheCapacity(layerConfig.kvCacheCapacity, maxKVCacheCapacity))
    {
        return {kvPoolPages, kvPoolPages, kvPoolPages};
    }
    int32_t const boundedPages = getBoundedKVPoolPagesForLayer(layerConfig);
    int32_t const smallerPages = std::min(boundedPages, kvPoolPages);
    return {smallerPages, smallerPages, std::max(boundedPages, kvPoolPages)};
}

std::vector<int32_t> LLMEngineConfig::flattenedSkipSoftmaxPrefillScaleTable() const
{
    std::vector<int32_t> flat;
    flat.reserve(2 * skipSoftmaxPrefillScaleTable.size());
    for (auto const& row : skipSoftmaxPrefillScaleTable)
    {
        flat.push_back(static_cast<int32_t>(row[0]));
        flat.push_back(static_cast<int32_t>(row[1]));
    }
    return flat;
}

std::vector<int32_t> LLMEngineConfig::flattenedSkipSoftmaxDecodeScaleTable() const
{
    std::vector<int32_t> flat;
    flat.reserve(2 * skipSoftmaxDecodeScaleTable.size());
    for (auto const& row : skipSoftmaxDecodeScaleTable)
    {
        flat.push_back(static_cast<int32_t>(row[0]));
        flat.push_back(static_cast<int32_t>(row[1]));
    }
    return flat;
}

int64_t LLMEngineConfig::skipSoftmaxPrefillScaleTableLen() const
{
    return static_cast<int64_t>(2 * skipSoftmaxPrefillScaleTable.size());
}

int64_t LLMEngineConfig::skipSoftmaxDecodeScaleTableLen() const
{
    return static_cast<int64_t>(2 * skipSoftmaxDecodeScaleTable.size());
}

int64_t LLMEngineConfig::skipSoftmaxPrefillScaleLen(int64_t seqLen, ExecutionPhase phase) const
{
    if (skipSoftmaxPrefillScaleTable.empty() || phase != ExecutionPhase::kContextPrefill)
    {
        return skipSoftmaxPrefillScaleTableLen();
    }
    // A sequence of length L can skip only if some row has bin <= L and S < L (lambda = S / L < 1).
    int64_t firstSkippableLen = std::numeric_limits<int64_t>::max();
    for (auto const& row : skipSoftmaxPrefillScaleTable)
    {
        firstSkippableLen = std::min(firstSkippableLen, std::max(row[0], row[1] + 1));
    }
    return seqLen < firstSkippableLen ? 0 : skipSoftmaxPrefillScaleTableLen();
}

InferenceDims LLMEngineConfig::prefillDims(int64_t batch, int64_t seqLen, ExecutionPhase phase) const
{
    ELLM_CHECK(isDiffusionBackbone ? phase == ExecutionPhase::kDiffusionCommit
                                   : phase == ExecutionPhase::kContextPrefill || phase == ExecutionPhase::kContextChunk,
        "Prefill dimensions require a context-prefill, context-chunk, or diffusion-commit phase");
    // startIndexLen=0 is the autoregressive plugin-path sentinel for "initial
    // prefill of an empty KV cache"; chunked prefill uses [batch].
    // DiffusionGemma keeps kvcache_start_index materialized and uses
    // context_mask_selector's shape sentinel to switch attention mask modes.
    int64_t const startIndexLen = phase == ExecutionPhase::kContextPrefill ? 0 : batch;
    // Keep KV cache binding shape at physical capacity for CUDA graph stability.
    // Logical work length is carried by query_lengths and past_lengths.
    int64_t const kvLen = maxKVCacheCapacity;
    int64_t const physicalTokens = batch * seqLen;
    int64_t const logits = batch;
    return InferenceDims{
        /*.batch=*/batch,
        /*.seqLen=*/physicalTokens,
        /*.kvLen=*/kvLen,
        /*.selectLen=*/logits,
        /*.attnMaskSeqLen=*/physicalTokens,
        /*.ropeBatch=*/(ropeConfig.type == RopeType::kMRope) ? batch : 1,
        /*.packedMaskLen=*/1,
        /*.contextMaskSelectorLen=*/0,
        /*.startIndexLen=*/startIndexLen,
        /*.executionPhaseLen=*/static_cast<int64_t>(phase),
        /*.specVerifyLayoutLen=*/0,
        /*.skipSoftmaxPrefillScaleLen=*/skipSoftmaxPrefillScaleLen(seqLen, phase),
        /*.skipSoftmaxDecodeScaleLen=*/skipSoftmaxDecodeScaleTableLen(),
        /*.swaKVCacheModeLen=*/getSwaKVCacheModeInputLength(),
        /*.queryOffsetLen=*/batch + 1,
        /*.contextSequenceCount=*/isDiffusionBackbone ? 0 : batch,
    };
}

InferenceDims LLMEngineConfig::decodeDims(int64_t batch) const
{
    return InferenceDims{
        /*.batch=*/batch,
        /*.seqLen=*/batch,
        /*.kvLen=*/maxKVCacheCapacity,
        /*.selectLen=*/batch,
        /*.attnMaskSeqLen=*/batch,
        /*.ropeBatch=*/(ropeConfig.type == RopeType::kMRope) ? batch : 1,
        /*.packedMaskLen=*/1,
        /*.contextMaskSelectorLen=*/0,
        /*.startIndexLen=*/batch,
        /*.executionPhaseLen=*/static_cast<int64_t>(ExecutionPhase::kAutoregressiveDecode),
        /*.specVerifyLayoutLen=*/0,
        /*.skipSoftmaxPrefillScaleLen=*/skipSoftmaxPrefillScaleTableLen(),
        /*.skipSoftmaxDecodeScaleLen=*/skipSoftmaxDecodeScaleTableLen(),
        /*.swaKVCacheModeLen=*/getSwaKVCacheModeInputLength(),
        /*.queryOffsetLen=*/batch + 1,
        /*.contextSequenceCount=*/0,
    };
}

InferenceDims LLMEngineConfig::compactDims(
    int64_t numSequences, int64_t numTokens, int64_t numContextSequences, int64_t numLogits) const
{
    ELLM_CHECK(!isDiffusionBackbone, "Compact dimensions are not defined for diffusion backbones");
    ELLM_CHECK(numSequences > 0 && numContextSequences >= 0 && numContextSequences <= numSequences,
        "Compact dimensions require 0 <= context sequences <= sequences and at least one sequence");
    ELLM_CHECK(numContextSequences > 0 ? numTokens >= numSequences : numTokens == numSequences,
        "Compact dimensions require one token row per decode sequence and at least one per context sequence");
    ELLM_CHECK(numLogits > 0 && numLogits <= numSequences, "Compact dimensions require 1 <= logits rows <= sequences");
    return InferenceDims{
        /*.batch=*/numSequences,
        /*.seqLen=*/numTokens,
        /*.kvLen=*/maxKVCacheCapacity,
        /*.selectLen=*/numLogits,
        /*.attnMaskSeqLen=*/numTokens,
        /*.ropeBatch=*/(ropeConfig.type == RopeType::kMRope) ? numSequences : 1,
        /*.packedMaskLen=*/1,
        /*.contextMaskSelectorLen=*/0,
        /*.startIndexLen=*/numSequences,
        /*.executionPhaseLen=*/static_cast<int64_t>(ExecutionPhase::kMixedPrefillDecode),
        /*.specVerifyLayoutLen=*/0,
        /*.skipSoftmaxPrefillScaleLen=*/skipSoftmaxPrefillScaleTableLen(),
        /*.skipSoftmaxDecodeScaleLen=*/skipSoftmaxDecodeScaleTableLen(),
        /*.swaKVCacheModeLen=*/getSwaKVCacheModeInputLength(),
        /*.queryOffsetLen=*/numSequences + 1,
        /*.contextSequenceCount=*/numContextSequences,
    };
}

InferenceDims LLMEngineConfig::denoiseDims(int64_t batch, int64_t canvasLen) const
{
    int64_t const physicalTokens = batch * canvasLen;
    return InferenceDims{
        /*.batch=*/batch,
        /*.seqLen=*/physicalTokens,
        /*.kvLen=*/maxKVCacheCapacity,
        /*.selectLen=*/physicalTokens,
        /*.attnMaskSeqLen=*/physicalTokens,
        /*.ropeBatch=*/(ropeConfig.type == RopeType::kMRope) ? batch : 1,
        /*.packedMaskLen=*/1,
        /*.contextMaskSelectorLen=*/batch,
        /*.startIndexLen=*/batch,
        /*.executionPhaseLen=*/static_cast<int64_t>(ExecutionPhase::kDiffusionDenoise),
        /*.specVerifyLayoutLen=*/0,
        /*.skipSoftmaxPrefillScaleLen=*/skipSoftmaxPrefillScaleTableLen(),
        /*.skipSoftmaxDecodeScaleLen=*/skipSoftmaxDecodeScaleTableLen(),
        /*.swaKVCacheModeLen=*/getSwaKVCacheModeInputLength(),
        /*.queryOffsetLen=*/batch + 1,
        /*.contextSequenceCount=*/0,
    };
}

InferenceDims LLMEngineConfig::diffusionCommitDims(int64_t batch, int64_t commitLen) const
{
    int64_t const physicalTokens = batch * commitLen;
    return InferenceDims{
        /*.batch=*/batch,
        /*.seqLen=*/physicalTokens,
        /*.kvLen=*/maxKVCacheCapacity,
        /*.selectLen=*/physicalTokens,
        /*.attnMaskSeqLen=*/physicalTokens,
        /*.ropeBatch=*/(ropeConfig.type == RopeType::kMRope) ? batch : 1,
        /*.packedMaskLen=*/1,
        /*.contextMaskSelectorLen=*/0,
        /*.startIndexLen=*/batch,
        /*.executionPhaseLen=*/static_cast<int64_t>(ExecutionPhase::kDiffusionCommit),
        /*.specVerifyLayoutLen=*/0,
        /*.skipSoftmaxPrefillScaleLen=*/skipSoftmaxPrefillScaleTableLen(),
        /*.skipSoftmaxDecodeScaleLen=*/skipSoftmaxDecodeScaleTableLen(),
        /*.swaKVCacheModeLen=*/getSwaKVCacheModeInputLength(),
        /*.queryOffsetLen=*/batch + 1,
        /*.contextSequenceCount=*/0,
    };
}

InferenceDims LLMEngineConfig::specVerifyDims(
    int64_t batch, int64_t verifyTokenCount, SpecVerifyLayout verifyLayout) const
{
    // verifyTokenCount fans out to four fields:
    //   seqLen, selectLen, attnMaskSeqLen (all = verifyTokenCount), and packedMaskLen.
    // This is the only recipe where selectLen != 1.
    return InferenceDims{
        /*.batch=*/batch,
        /*.seqLen=*/batch * verifyTokenCount,
        /*.kvLen=*/maxKVCacheCapacity,
        /*.selectLen=*/batch * verifyTokenCount,
        /*.attnMaskSeqLen=*/batch * verifyTokenCount,
        /*.ropeBatch=*/(ropeConfig.type == RopeType::kMRope) ? batch : 1,
        /*.packedMaskLen=*/static_cast<int64_t>(divUp(verifyTokenCount, 32)),
        /*.contextMaskSelectorLen=*/0,
        /*.startIndexLen=*/batch,
        /*.executionPhaseLen=*/static_cast<int64_t>(ExecutionPhase::kSpecTargetVerify),
        /*.specVerifyLayoutLen=*/specVerifyLayoutExtent(verifyLayout),
        /*.skipSoftmaxPrefillScaleLen=*/skipSoftmaxPrefillScaleTableLen(),
        /*.skipSoftmaxDecodeScaleLen=*/skipSoftmaxDecodeScaleTableLen(),
        /*.swaKVCacheModeLen=*/getSwaKVCacheModeInputLength(),
        /*.queryOffsetLen=*/batch + 1,
        /*.contextSequenceCount=*/0,
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
        /*.seqLen=*/batch * proposalSize,
        /*.kvLen=*/maxKVCacheCapacity,
        /*.selectLen=*/batch * draftTopK,
        /*.attnMaskSeqLen=*/batch * proposalSize,
        /*.ropeBatch=*/(ropeConfig.type == RopeType::kMRope) ? batch : 1,
        /*.packedMaskLen=*/static_cast<int64_t>(divUp(proposalSize, 32)),
        /*.contextMaskSelectorLen=*/0,
        /*.startIndexLen=*/batch,
        /*.executionPhaseLen=*/static_cast<int64_t>(ExecutionPhase::kSpecDraftProposal),
        /*.specVerifyLayoutLen=*/0,
        /*.skipSoftmaxPrefillScaleLen=*/skipSoftmaxPrefillScaleTableLen(),
        /*.skipSoftmaxDecodeScaleLen=*/skipSoftmaxDecodeScaleTableLen(),
        /*.swaKVCacheModeLen=*/getSwaKVCacheModeInputLength(),
        /*.queryOffsetLen=*/batch + 1,
        /*.contextSequenceCount=*/0,
    };
}

InferenceDims LLMEngineConfig::acceptDims(int64_t batch, int64_t acceptLen) const
{
    // Precondition: acceptLen >= 1. Caller computes from the SpecDecode accept result in
    // [1, maxPathLength]; a zero here would indicate a degenerate batch and
    // will be caught by EngineExecutor::prepare's field-completeness check.
    // acceptLen fans out to: seqLen, attnMaskSeqLen, and packedMaskLen.
    return InferenceDims{
        /*.batch=*/batch,
        /*.seqLen=*/batch * acceptLen,
        /*.kvLen=*/maxKVCacheCapacity,
        /*.selectLen=*/batch,
        /*.attnMaskSeqLen=*/batch * acceptLen,
        /*.ropeBatch=*/(ropeConfig.type == RopeType::kMRope) ? batch : 1,
        /*.packedMaskLen=*/static_cast<int64_t>(divUp(acceptLen, 32)),
        /*.contextMaskSelectorLen=*/0,
        /*.startIndexLen=*/batch,
        /*.executionPhaseLen=*/
        static_cast<int64_t>(
            acceptLen == 1 ? ExecutionPhase::kAutoregressiveDecode : ExecutionPhase::kSpecDraftProposal),
        /*.specVerifyLayoutLen=*/0,
        /*.skipSoftmaxPrefillScaleLen=*/skipSoftmaxPrefillScaleTableLen(),
        /*.skipSoftmaxDecodeScaleLen=*/skipSoftmaxDecodeScaleTableLen(),
        /*.swaKVCacheModeLen=*/getSwaKVCacheModeInputLength(),
        /*.queryOffsetLen=*/batch + 1,
        /*.contextSequenceCount=*/0,
    };
}

void validatePagedKVBindings(LLMEngineConfig const& config, EngineExecutor const& executor, char const* engineLabel)
{
    if (config.numAttentionLayers == 0)
    {
        return;
    }

    ELLM_CHECK(static_cast<int32_t>(config.kvLayerConfigs.size()) == config.numAttentionLayers,
        std::string("KV layer metadata mismatch (") + engineLabel + "): expected "
            + std::to_string(config.numAttentionLayers) + " entries, got "
            + std::to_string(config.kvLayerConfigs.size()) + ".");
    int32_t const numProfiles = executor.getEngine().getNbOptimizationProfiles();
    ELLM_CHECK(numProfiles > 0, std::string("Engine has no optimization profiles (") + engineLabel + ").");

    for (int32_t layerIdx = 0; layerIdx < config.numAttentionLayers; ++layerIdx)
    {
        std::string const bindingName = binding_names::formatKVCacheName(layerIdx, /*isPast=*/true);
        ELLM_CHECK(executor.hasIOTensor(bindingName.c_str()),
            std::string("Missing KV cache binding (") + engineLabel + "): expected '" + bindingName + "'.");
        trt_edgellm::DataType const engineDtype = executor.getBindingDataType(bindingName.c_str());
        ELLM_CHECK(isKVCacheBindingDtypeCompatible(config.kvCacheDtype, engineDtype),
            std::string("KV cache dtype mismatch (") + engineLabel + "): config says "
                + getDataTypeString(config.kvCacheDtype) + ", engine reports " + getDataTypeString(engineDtype)
                + " for binding '" + bindingName + "'.");

        KVLayerConfig const& layer = config.kvLayerConfigs[layerIdx];
        std::array<int32_t, 3> const pageProfile = config.getKVPoolPageProfileForLayer(layer);
        for (int32_t profileIdx = 0; profileIdx < numProfiles; ++profileIdx)
        {
            for (nvinfer1::OptProfileSelector const selector : {nvinfer1::OptProfileSelector::kMIN,
                     nvinfer1::OptProfileSelector::kOPT, nvinfer1::OptProfileSelector::kMAX})
            {
                size_t const selectorIndex = selector == nvinfer1::OptProfileSelector::kMIN ? 0U
                    : selector == nvinfer1::OptProfileSelector::kOPT                        ? 1U
                                                                                            : 2U;
                int32_t const expectedPages = pageProfile[selectorIndex];
                nvinfer1::Dims const shape = executor.getProfileShape(bindingName.c_str(), profileIdx, selector);
                ELLM_CHECK(shape.nbDims == 5 && shape.d[0] == 2 && shape.d[1] == expectedPages
                        && shape.d[2] == kTOKENS_PER_PAGE && shape.d[3] == layer.numKVHeads
                        && shape.d[4] == layer.headDim,
                    std::string("Paged KV profile mismatch (") + engineLabel + ") for binding '" + bindingName
                        + "': expected [2," + std::to_string(expectedPages) + "," + std::to_string(kTOKENS_PER_PAGE)
                        + "," + std::to_string(layer.numKVHeads) + "," + std::to_string(layer.headDim) + "].");
            }
        }
    }
}

namespace
{

void validateContextSequenceCountCarrierBinding(EngineExecutor const& executor, char const* engineLabel)
{
    char const* const bindingName = binding_names::kContextSequenceCountCarrier;
    ELLM_CHECK(executor.hasIOTensor(bindingName),
        std::string("Missing ragged context-count carrier (") + engineLabel + "): expected '" + bindingName
            + "'. Re-export the ONNX model and rebuild the engine.");
    ELLM_CHECK(executor.getBindingDataType(bindingName) == trt_edgellm::DataType::kINT32,
        std::string("Ragged context-count carrier (") + engineLabel + ") must have INT32 dtype. Re-export the ONNX "
            "model and rebuild the engine.");
}

//! Validate the current mutable page-table engine ABI.
void validatePageTableBinding(
    LLMEngineConfig const& config, EngineExecutor const& executor, char const* engineLabel, char const* bindingName)
{
    if (config.numAttentionLayers == 0)
    {
        return;
    }

    ELLM_CHECK(executor.hasIOTensor(bindingName),
        std::string("Missing page-table binding (") + engineLabel + "): expected '" + bindingName
            + "' from the current engine toolchain.");
    ELLM_CHECK(executor.getEngine().getTensorIOMode(bindingName) == nvinfer1::TensorIOMode::kINPUT,
        std::string("Page-table binding (") + engineLabel + ") must be an input.");
    ELLM_CHECK(executor.getBindingDataType(bindingName) == trt_edgellm::DataType::kINT32,
        std::string("Page-table binding (") + engineLabel + ") must have INT32 dtype.");

    int32_t const numProfiles = executor.getEngine().getNbOptimizationProfiles();
    ELLM_CHECK(numProfiles > 0, std::string("Engine has no optimization profiles (") + engineLabel + ").");
    int32_t const maxPagesPerSequence = computeMaxPagesPerSeq(config.maxKVCacheCapacity);
    for (int32_t profileIdx = 0; profileIdx < numProfiles; ++profileIdx)
    {
        for (nvinfer1::OptProfileSelector const selector : {nvinfer1::OptProfileSelector::kMIN,
                 nvinfer1::OptProfileSelector::kOPT, nvinfer1::OptProfileSelector::kMAX})
        {
            int32_t const expectedBatch
                = selector == nvinfer1::OptProfileSelector::kMIN ? 1 : config.maxSupportedBatchSize;
            nvinfer1::Dims const shape = executor.getProfileShape(bindingName, profileIdx, selector);
            ELLM_CHECK(shape.nbDims == 3 && shape.d[0] == expectedBatch && shape.d[1] == 2
                    && shape.d[2] == maxPagesPerSequence,
                std::string("Page-table profile mismatch (") + engineLabel + ") for binding '" + bindingName
                    + "': expected [" + std::to_string(expectedBatch) + ",2," + std::to_string(maxPagesPerSequence)
                    + "]. Re-export and rebuild the engine.");
        }
    }
}

void validateSwaModeBinding(LLMEngineConfig const& config, EngineExecutor const& executor, char const* engineLabel)
{
    if (!config.supportsBoundedSwaKVCache())
    {
        return;
    }

    char const* const bindingName = binding_names::kSwaKVCacheMode;
    ELLM_CHECK(executor.hasIOTensor(bindingName),
        std::string("Missing SWA mode binding (") + engineLabel + "): expected '" + bindingName + "'.");
    ELLM_CHECK(executor.getEngine().getTensorIOMode(bindingName) == nvinfer1::TensorIOMode::kINPUT,
        std::string("SWA mode binding (") + engineLabel + ") must be an input.");
    ELLM_CHECK(executor.getBindingDataType(bindingName) == trt_edgellm::DataType::kINT32,
        std::string("SWA mode binding (") + engineLabel + ") must have INT32 dtype.");

    int32_t const numProfiles = executor.getEngine().getNbOptimizationProfiles();
    for (int32_t profileIdx = 0; profileIdx < numProfiles; ++profileIdx)
    {
        for (nvinfer1::OptProfileSelector const selector : {nvinfer1::OptProfileSelector::kMIN,
                 nvinfer1::OptProfileSelector::kOPT, nvinfer1::OptProfileSelector::kMAX})
        {
            int32_t const expectedLength = selector == nvinfer1::OptProfileSelector::kMIN ? 0 : 1;
            nvinfer1::Dims const shape = executor.getProfileShape(bindingName, profileIdx, selector);
            ELLM_CHECK(shape.nbDims == 1 && shape.d[0] == expectedLength,
                std::string("SWA mode profile mismatch (") + engineLabel + ") for binding '" + bindingName
                    + "': expected [" + std::to_string(expectedLength) + "].");
        }
    }
}

//! An EAGLE candidate-head draft replaces `logits` with the two [rows, Kc] support outputs of its candidate head, so an
//! engine built from another export fails at load instead of at its first forward.
void validateEagleCandidateBindings(
    LLMEngineConfig const& config, EngineExecutor const& executor, char const* engineLabel)
{
    std::string const engine = std::string("EAGLE candidate-head draft engine (") + engineLabel + ")";
    std::string const remedy = ". Re-export and rebuild the draft engine.";
    ELLM_CHECK(!executor.hasIOTensor(binding_names::kLogits),
        engine + " must not expose '" + binding_names::kLogits + "', the output of a dense draft head" + remedy);
    for (auto const& [name, dtype] : {std::pair{binding_names::kSpecProposalSupportIds, trt_edgellm::DataType::kINT32},
             std::pair{binding_names::kSpecProposalSupportLogProbs, trt_edgellm::DataType::kFLOAT}})
    {
        ELLM_CHECK(executor.hasIOTensor(name), engine + " is missing required binding '" + name + "'" + remedy);
        nvinfer1::Dims const shape = executor.getTensorShape(name);
        ELLM_CHECK(
            executor.getBindingDataType(name) == dtype && shape.nbDims == 2 && shape.d[1] == config.draftCandidateCount,
            engine + " binding '" + name + "' must be " + getDataTypeString(dtype) + " [rows, "
                + std::to_string(config.draftCandidateCount) + "] to match eagle3_config.candidate_head.num_candidates"
                + remedy);
    }
}

} // namespace

void validateAgainstEngine(LLMEngineConfig const& config, EngineExecutor const& executor, char const* engineLabel)
{
    ELLM_CHECK(!executor.hasIOTensor(kObsoleteGenericTokenOwnerBinding),
        std::string("Engine (") + engineLabel + ") contains the removed MR1 prototype binding '"
            + kObsoleteGenericTokenOwnerBinding + "'. Re-export and rebuild the engine.");
    validateContextSequenceCountCarrierBinding(executor, engineLabel);
    validatePagedKVBindings(config, executor, engineLabel);
    validatePageTableBinding(config, executor, engineLabel, binding_names::kKVPageTable);
    if (config.supportsBoundedSwaKVCache())
    {
        validatePageTableBinding(config, executor, engineLabel, binding_names::kSwaKVPageTable);
        validateSwaModeBinding(config, executor, engineLabel);
    }

    if (isCachedBlockDraftDraft(config))
    {
        char const* modeName = config.specDecodeType == SpecDecodeMode::kJetSpec
            ? "JetSpec"
            : (config.specDecodeType == SpecDecodeMode::kDomino
                      ? "Domino"
                      : (config.dflashVersion == DFlashVersion::kV2 ? "DFlash V2" : "DFlash"));
        // Cached draft engines require KV cache bindings (cached-KV path).
        // Validate required bindings exist and have correct dtype.
        LOG_INFO("%s draft engine (%s): validating cached-path bindings.", modeName, engineLabel);

        // Required unified cached-path bindings.
        static char const* const kRequiredCommonBindings[] = {
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
        };
        for (auto const* name : kRequiredCommonBindings)
        {
            ELLM_CHECK(executor.hasIOTensor(name),
                std::string(modeName) + " cached draft engine (" + engineLabel + ") is missing required binding '"
                    + name + "'. This engine may be from the old explicit cached-draft path. Re-export and rebuild.");
        }

        if (config.specDecodeType == SpecDecodeMode::kDFlash && config.dflashVersion == DFlashVersion::kV2
            && config.dflashProjector == DFlashProjector::kLiLiCorr)
        {
            auto const requireBinding = [&engineLabel](char const* name, bool ok, std::string const& requirement) {
                ELLM_CHECK(ok,
                    std::string("LiLiCorr cached draft engine (") + engineLabel + ") binding '" + name + "' "
                        + requirement + ". Re-export and rebuild the draft engine.");
            };
            auto const requireDtype = [&](char const* name, trt_edgellm::DataType expected, char const* expectedName) {
                auto const actual = executor.getBindingDataType(name);
                requireBinding(name, actual == expected,
                    std::string("must be ") + expectedName + ", engine reports " + getDataTypeString(actual));
            };
            for (auto const* name : {binding_names::kSpecProposalSupportIds,
                     binding_names::kSpecLiLiCorrTransitionScores, binding_names::kLiLiCorrEmbeddingTable})
            {
                requireBinding(name, executor.hasIOTensor(name), "is missing");
            }
            requireDtype(binding_names::kSpecProposalSupportIds, trt_edgellm::DataType::kINT32, "INT32");
            requireDtype(binding_names::kSpecLiLiCorrTransitionScores, trt_edgellm::DataType::kFLOAT, "FP32");
            requireDtype(binding_names::kLiLiCorrEmbeddingTable, trt_edgellm::DataType::kHALF, "FP16");
            nvinfer1::Dims const table = executor.getTensorShape(binding_names::kLiLiCorrEmbeddingTable);
            requireBinding(binding_names::kLiLiCorrEmbeddingTable,
                table.nbDims == 2 && table.d[0] == config.vocabSize && table.d[1] == config.hiddenSize,
                "must be [" + std::to_string(config.vocabSize) + ", " + std::to_string(config.hiddenSize) + "]");
            // Output buffers are sized from the config; TensorRT writes the engine's own static extents into them.
            int64_t const proposalLen = config.specDraftBlockSize - 1;
            int64_t const topK = config.specLiLiCorrTopK;
            std::string const geometry = "] for block_size " + std::to_string(config.specDraftBlockSize)
                + " and lilicorr_candidate_topk " + std::to_string(topK);
            nvinfer1::Dims const support = executor.getTensorShape(binding_names::kSpecProposalSupportIds);
            requireBinding(binding_names::kSpecProposalSupportIds,
                support.nbDims == 3 && (support.d[1] == -1 || support.d[1] == proposalLen) && support.d[2] == topK,
                "must be [-1, " + std::to_string(proposalLen) + " or -1, " + std::to_string(topK) + geometry);
            nvinfer1::Dims const scores = executor.getTensorShape(binding_names::kSpecLiLiCorrTransitionScores);
            requireBinding(binding_names::kSpecLiLiCorrTransitionScores,
                scores.nbDims == 4 && scores.d[1] == proposalLen && scores.d[2] == topK && scores.d[3] == topK,
                "must be [-1, " + std::to_string(proposalLen) + ", " + std::to_string(topK) + ", "
                    + std::to_string(topK) + geometry);
            for (auto const* name : {binding_names::kLogits, binding_names::kSpecProposalUnaryValues,
                     binding_names::kSpecProposalProjectedHidden})
            {
                requireBinding(name, !executor.hasIOTensor(name), "must not be exposed");
            }
        }
        else if (config.specDecodeType == SpecDecodeMode::kDFlash && config.dflashVersion == DFlashVersion::kV2)
        {
            static char const* const kRequiredDFlash2Bindings[] = {
                binding_names::kSpecProposalSupportIds,
                binding_names::kSpecProposalUnaryValues,
                binding_names::kSpecProposalProjectedHidden,
            };
            for (auto const* name : kRequiredDFlash2Bindings)
            {
                ELLM_CHECK(executor.hasIOTensor(name),
                    std::string("DFlash2 cached draft engine (") + engineLabel
                        + ") is missing required sparse-proposal binding '" + name
                        + "'. Re-export and rebuild the draft engine.");
            }
            ELLM_CHECK(!executor.hasIOTensor(binding_names::kLogits),
                std::string("DFlash2 cached draft engine (") + engineLabel
                    + ") must not expose the legacy full-vocabulary logits binding.");
        }
        else if (config.specDecodeType == SpecDecodeMode::kDomino)
        {
            static char const* const kRequiredDominoBindings[] = {
                binding_names::kSpecPoolCandidateIds,
                binding_names::kSpecPoolBaseLogits,
                binding_names::kSpecPoolCodebook,
                binding_names::kSpecProposalZProj,
            };
            for (auto const* name : kRequiredDominoBindings)
            {
                ELLM_CHECK(executor.hasIOTensor(name),
                    std::string("Domino cached draft engine (") + engineLabel
                        + ") is missing required candidate-pool binding '" + name
                        + "'. Re-export and rebuild the draft engine.");
            }
            ELLM_CHECK(!executor.hasIOTensor(binding_names::kLogits),
                std::string("Domino cached draft engine (") + engineLabel
                    + ") must not expose the legacy full-vocabulary logits binding.");
        }
        else
        {
            ELLM_CHECK(executor.hasIOTensor(binding_names::kLogits),
                std::string(modeName) + " cached draft engine (" + engineLabel + ") is missing required binding '"
                    + binding_names::kLogits
                    + "'. This engine may be from the old explicit cached-draft path. Re-export and rebuild.");
        }

        // Require KV cache layer 0
        if (config.numAttentionLayers > 0)
        {
            std::string const kvPastName = binding_names::formatKVCacheName(/*layerIdx=*/0, /*isPast=*/true);
            std::string const kvPresentName = binding_names::formatKVCacheName(/*layerIdx=*/0, /*isPast=*/false);
            ELLM_CHECK(executor.hasIOTensor(kvPastName.c_str()),
                std::string(modeName) + " cached draft engine (" + engineLabel + ") missing KV cache binding '"
                    + kvPastName + "'. Old explicit cached-draft engines are not compatible. Re-export and rebuild.");
            ELLM_CHECK(executor.hasIOTensor(kvPresentName.c_str()),
                std::string(modeName) + " cached draft engine (" + engineLabel + ") missing KV cache binding '"
                    + kvPresentName + "'.");

            auto const engineDtype = executor.getBindingDataType(kvPastName.c_str());
            ELLM_CHECK(isKVCacheBindingDtypeCompatible(config.kvCacheDtype, engineDtype),
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
            binding_names::kPositions,
            binding_names::kQueryStartOffsets,
            binding_names::kQueryLengths,
            binding_names::kPastLengths,
            binding_names::kAttentionSequenceLengths,
            binding_names::kStateIndices,
            binding_names::kExecutionPhaseMarker,
            binding_names::kKVPageTable,
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
            ELLM_CHECK(isKVCacheBindingDtypeCompatible(config.kvCacheDtype, engineDtype),
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
                    ELLM_CHECK(shape.d[0] == 2 && shape.d[1] == config.kvPoolPages && shape.d[2] == kTOKENS_PER_PAGE
                            && shape.d[3] == kvConfig.numKVHeads && shape.d[4] == kvConfig.headDim,
                        std::string("Gemma4 MTP shared KV profile shape mismatch for binding '") + kvPastName
                            + "': expected static dims [2," + std::to_string(config.kvPoolPages) + ","
                            + std::to_string(kTOKENS_PER_PAGE) + "," + std::to_string(kvConfig.numKVHeads) + ","
                            + std::to_string(kvConfig.headDim) + "].");
                }
            }
        }

        return;
    }

    if (isDSparkDraftConfig(config))
    {
        // DSpark cached draft engines reuse the DFlash target-hidden/delta portals
        // and add a DSpark-specific hidden-state output.
        LOG_INFO("DSpark draft engine (%s): validating cached-path bindings.", engineLabel);

        static char const* const kRequiredBindings[] = {
            binding_names::kInputsEmbeds,
            binding_names::kDFlashTargetHiddenConcat,
            binding_names::kLogits,
            binding_names::kDSparkHiddenStates,
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
        };
        for (auto const* name : kRequiredBindings)
        {
            ELLM_CHECK(executor.hasIOTensor(name),
                std::string("DSpark cached draft engine (") + engineLabel + ") is missing required binding '" + name
                    + "'. Re-export and rebuild the DSpark draft engine.");
        }

        if (config.numAttentionLayers > 0)
        {
            std::string const kvPastName = binding_names::formatKVCacheName(/*layerIdx=*/0, /*isPast=*/true);
            std::string const kvPresentName = binding_names::formatKVCacheName(/*layerIdx=*/0, /*isPast=*/false);
            ELLM_CHECK(executor.hasIOTensor(kvPastName.c_str()),
                std::string("DSpark cached draft engine (") + engineLabel + ") missing KV cache binding '" + kvPastName
                    + "'. Re-export and rebuild.");
            ELLM_CHECK(executor.hasIOTensor(kvPresentName.c_str()),
                std::string("DSpark cached draft engine (") + engineLabel + ") missing KV cache binding '"
                    + kvPresentName + "'.");

            auto const engineDtype = executor.getBindingDataType(kvPastName.c_str());
            ELLM_CHECK(isKVCacheBindingDtypeCompatible(config.kvCacheDtype, engineDtype),
                std::string("KV cache dtype mismatch (") + engineLabel + "): config says "
                    + getDataTypeString(config.kvCacheDtype) + ", engine reports " + getDataTypeString(engineDtype)
                    + " for binding '" + kvPastName + "'.");
        }

        return;
    }

    if (config.specDecodeType == SpecDecodeMode::kEAGLE && !config.isSpecDecodeBase && config.draftCandidateCount > 0)
    {
        validateEagleCandidateBindings(config, executor, engineLabel);
    }

    // KV cache binding: validated on layer 0; all layers share the same dtype.
    // Plugin-path engines use a combined `past_key_values_%d` binding.
    if (config.numAttentionLayers > 0)
    {
        std::string const kvBindingName = binding_names::formatKVCacheName(/*layerIdx=*/0, /*isPast=*/true);
        ELLM_CHECK(executor.hasIOTensor(kvBindingName.c_str()),
            std::string("Missing KV cache binding (") + engineLabel + "): expected '" + kvBindingName + "'.");
        auto const engineDtype = executor.getBindingDataType(kvBindingName.c_str());
        ELLM_CHECK(isKVCacheBindingDtypeCompatible(config.kvCacheDtype, engineDtype),
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
            && (config.specDecodeType == SpecDecodeMode::kMTP || config.specDecodeType == SpecDecodeMode::kEAGLE
                || isCachedBlockDraftMode(config.specDecodeType) || config.specDecodeType == SpecDecodeMode::kDSpark))
        {
            ELLM_CHECK(executor.hasIOTensor(binding_names::kExecutionPhaseMarker),
                std::string("Missing spec-verify phase marker binding (") + engineLabel + "): expected '"
                    + binding_names::kExecutionPhaseMarker + "'. Re-export the hybrid speculative base engine.");
            auto const markerEngineDtype = executor.getBindingDataType(binding_names::kExecutionPhaseMarker);
            ELLM_CHECK(markerEngineDtype == trt_edgellm::DataType::kINT32,
                std::string("Spec-verify phase marker dtype mismatch (") + engineLabel + "): engine reports "
                    + getDataTypeString(markerEngineDtype) + " for binding '" + binding_names::kExecutionPhaseMarker
                    + "'. Re-export the hybrid speculative base engine.");
            ELLM_CHECK(executor.hasIOTensor(binding_names::kSpecVerifyLayout),
                std::string("Missing spec-verify layout binding (") + engineLabel + "): expected '"
                    + binding_names::kSpecVerifyLayout + "'. Re-export the hybrid speculative base engine.");
            auto const layoutEngineDtype = executor.getBindingDataType(binding_names::kSpecVerifyLayout);
            ELLM_CHECK(layoutEngineDtype == trt_edgellm::DataType::kINT32,
                std::string("Spec-verify layout dtype mismatch (") + engineLabel + "): engine reports "
                    + getDataTypeString(layoutEngineDtype) + " for binding '" + binding_names::kSpecVerifyLayout
                    + "'. Re-export the hybrid speculative base engine.");
        }
    }

    // The n-gram injection's conv state is bound independently of the recurrent block above: the
    // injected layer need not be recurrent, so this check cannot hang off numLinearAttnLayers.
    if (config.ngramEnabled())
    {
        std::string const ngramBindingName
            = binding_names::formatNgramConvStateName(/*ngramLayerIdx=*/0, /*isPast=*/true);
        ELLM_CHECK(executor.hasIOTensor(ngramBindingName.c_str()),
            std::string("Missing n-gram conv-state binding (") + engineLabel + "): expected '" + ngramBindingName
                + "'.");
        auto const ngramEngineDtype = executor.getBindingDataType(ngramBindingName.c_str());
        ELLM_CHECK(ngramEngineDtype == config.ngramConvStateDtype,
            std::string("N-gram conv state dtype mismatch (") + engineLabel + "): config says "
                + getDataTypeString(config.ngramConvStateDtype) + ", engine reports "
                + getDataTypeString(ngramEngineDtype) + " for binding '" + ngramBindingName
                + "'. Re-export the engine with matching ngram_conv_state_dtype.");
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
        /*.contextMaskSelectorLen=*/0,
        /*.startIndexLen=*/1,
        /*.executionPhaseLen=*/static_cast<int64_t>(ExecutionPhase::kAutoregressiveDecode),
        /*.specVerifyLayoutLen=*/0,
        /*.skipSoftmaxPrefillScaleLen=*/skipSoftmaxPrefillScaleTableLen(),
        /*.skipSoftmaxDecodeScaleLen=*/skipSoftmaxDecodeScaleTableLen(),
        /*.swaKVCacheModeLen=*/getSwaKVCacheModeInputLength(),
        /*.queryOffsetLen=*/2,
        /*.contextSequenceCount=*/0,
    };
}

} // namespace rt
} // namespace trt_edgellm
