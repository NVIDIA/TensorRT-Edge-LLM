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

#include "llmBuilder.h"
#include "builderUtils.h"
#include "common/bindingNames.h"
#include "common/cudaUtils.h"
#include "common/fileUtils.h"
#include "common/logger.h"
#include "common/pagedKvTypes.h"
#include "common/parallelArtifactNames.h"
#include "common/ropeUtils.h"
#include "common/specDecodeConfigUtils.h"
#include "common/trtUtils.h"
#include "common/version.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>
#include <unordered_set>
#include <vector>

using namespace trt_edgellm;

namespace trt_edgellm
{
namespace builder
{
namespace
{

std::string specDecodeType(Json const& config)
{
    return config.value("spec_decode_type", "none");
}

std::string engineRole(Json const& config)
{
    return config.value("engine_role", "llm");
}

bool isSpecDecodeBase(Json const& config, char const* type)
{
    return specDecodeType(config) == type && engineRole(config) == "base";
}

bool isSpecDecodeDraft(Json const& config, char const* type)
{
    return specDecodeType(config) == type && engineRole(config) == "draft";
}

//! DFlash-family draft engines (dflash/dflash2/jetspec share the DFlashDraftModel
//! ONNX; dspark uses the DSpark draft) are the only engines that contain a plugin
//! declaring aliased I/O: DFlashTargetKVCacheUpdatePlugin's present KV pool aliases
//! its past KV pool. No other engine needs the kALIASED_PLUGIN_IO preview.
bool usesAliasedPluginIO(Json const& config)
{
    return isSpecDecodeDraft(config, "dflash") || isSpecDecodeDraft(config, "dflash2")
        || isSpecDecodeDraft(config, "jetspec") || isSpecDecodeDraft(config, "dspark");
}

//! The CodePredictor exports one model name per Qwen3-Omni variant, all sharing this suffix.
bool isCodePredictor(Json const& config)
{
    constexpr std::string_view suffix = "_code_predictor";
    auto const model = config.value("model", std::string{});
    return model.size() >= suffix.size() && model.compare(model.size() - suffix.size(), suffix.size(), suffix) == 0;
}

bool isValidSpecDecodeType(std::string const& type)
{
    return type == "none" || type == "mtp" || type == "eagle3" || type == "dflash" || type == "dflash2"
        || type == "jetspec" || type == "dspark" || type == "gemma4_mtp";
}

bool isValidEngineRole(std::string const& role)
{
    return role == "llm" || role == "base" || role == "draft" || role == "dllm";
}

bool hasInputBinding(nvinfer1::INetworkDefinition const& network, char const* inputName)
{
    std::string_view const target{inputName};
    for (int32_t idx = 0; idx < network.getNbInputs(); ++idx)
    {
        if (std::string_view{network.getInput(idx)->getName()} == target)
        {
            return true;
        }
    }
    return false;
}

std::optional<int64_t> getStaticInputDim(
    nvinfer1::INetworkDefinition const& network, char const* inputName, int32_t axis)
{
    std::string_view const target{inputName};
    for (int32_t idx = 0; idx < network.getNbInputs(); ++idx)
    {
        auto const* input = network.getInput(idx);
        if (std::string_view{input->getName()} != target)
        {
            continue;
        }
        nvinfer1::Dims const dims = input->getDimensions();
        if (axis >= dims.nbDims || dims.d[axis] <= 0)
        {
            return std::nullopt;
        }
        return dims.d[axis];
    }
    return std::nullopt;
}

std::optional<int32_t> getInputRank(nvinfer1::INetworkDefinition const& network, char const* inputName)
{
    for (int32_t idx = 0; idx < network.getNbInputs(); ++idx)
    {
        auto const* input = network.getInput(idx);
        if (std::string_view{input->getName()} == inputName)
        {
            return input->getDimensions().nbDims;
        }
    }
    return std::nullopt;
}

parallel_artifacts::RankArtifactContext makeArtifactContext(LLMBuilderConfig const& config)
{
    return parallel_artifacts::RankArtifactContext{
        static_cast<int32_t>(config.tpSize), static_cast<int32_t>(config.tpRank)};
}

int32_t configRank(LLMBuilderConfig const& config)
{
    return static_cast<int32_t>(config.tpRank);
}

bool validateRankConfigs(Json const& config, LLMBuilderConfig const& builderConfig)
{
    if (builderConfig.tpSize < 1)
    {
        LOG_ERROR("tpSize must be positive, got %lld.", static_cast<long long>(builderConfig.tpSize));
        return false;
    }
    if (builderConfig.tpRank < 0 || builderConfig.tpRank >= builderConfig.tpSize)
    {
        LOG_ERROR("tpRank must be in [0, tpSize), got tpRank=%lld tpSize=%lld.",
            static_cast<long long>(builderConfig.tpRank), static_cast<long long>(builderConfig.tpSize));
        return false;
    }

    if (!config.contains("rank_configs"))
    {
        if (builderConfig.tpSize > 1)
        {
            LOG_ERROR("Multi-device build requires rank_configs, but config.json has none (tpSize=%lld).",
                static_cast<long long>(builderConfig.tpSize));
            return false;
        }
        return true;
    }

    Json const& rankConfigs = config["rank_configs"];
    if (!rankConfigs.is_array())
    {
        LOG_ERROR("rank_configs must be an array when present in config.json");
        return false;
    }
    if (rankConfigs.size() != static_cast<size_t>(builderConfig.tpSize))
    {
        LOG_ERROR("rank_configs length (%zu) must match tpSize (%lld).", rankConfigs.size(),
            static_cast<long long>(builderConfig.tpSize));
        return false;
    }

    std::unordered_set<int64_t> ranks;
    for (auto const& rankConfig : rankConfigs)
    {
        if (!rankConfig.is_object() || !rankConfig.contains("rank") || !rankConfig["rank"].is_number_integer())
        {
            LOG_ERROR("Each rank_configs entry must be an object with an integer rank field.");
            return false;
        }
        int64_t const rank = rankConfig["rank"].get<int64_t>();
        if (rank < 0 || rank >= builderConfig.tpSize)
        {
            LOG_ERROR("rank_configs rank %lld is outside [0, tpSize=%lld).", static_cast<long long>(rank),
                static_cast<long long>(builderConfig.tpSize));
            return false;
        }
        if (!ranks.insert(rank).second)
        {
            LOG_ERROR("rank_configs contains duplicate rank %lld.", static_cast<long long>(rank));
            return false;
        }
        if (rankConfig.contains("config_overrides") && !rankConfig["config_overrides"].is_object())
        {
            LOG_ERROR(
                "rank_configs[%lld].config_overrides must be an object when present.", static_cast<long long>(rank));
            return false;
        }
    }
    return true;
}

bool applyRankConfigOverrides(Json& config, int32_t rank)
{
    if (!config.contains("rank_configs"))
    {
        return true;
    }
    if (!config["rank_configs"].is_array())
    {
        LOG_ERROR("rank_configs must be an array when present in config.json");
        return false;
    }
    for (auto const& rankConfig : config["rank_configs"])
    {
        if (!rankConfig.is_object())
        {
            LOG_ERROR("Each rank_configs entry must be an object.");
            return false;
        }
        if (rankConfig.value("rank", -1) != rank)
        {
            continue;
        }
        Json const overrides = rankConfig.value("config_overrides", Json::object());
        if (!overrides.is_object())
        {
            LOG_ERROR("rank_configs[%d].config_overrides must be an object when present.", rank);
            return false;
        }
        for (auto it = overrides.begin(); it != overrides.end(); ++it)
        {
            config[it.key()] = it.value();
        }
        return true;
    }
    LOG_ERROR("No rank_configs entry found for rank %d.", rank);
    return false;
}

std::string getInputConfigFileName(LLMBuilderConfig const& config)
{
    return parallel_artifacts::configFileName(makeArtifactContext(config));
}

std::string getOutputConfigFileName(LLMBuilderConfig const& config)
{
    if (config.specDraft)
    {
        return "draft_config.json";
    }
    if (config.specBase)
    {
        return "base_config.json";
    }
    return parallel_artifacts::configFileName(makeArtifactContext(config));
}

std::string getEngineFileName(LLMBuilderConfig const& config)
{
    if (config.specDraft)
    {
        return "spec_draft.engine";
    }
    if (config.specBase)
    {
        return "spec_base.engine";
    }
    return parallel_artifacts::engineFileName(makeArtifactContext(config));
}

std::string getOnnxFilePath(std::filesystem::path const& onnxDir, LLMBuilderConfig const& config)
{
    parallel_artifacts::RankArtifactContext const context = makeArtifactContext(config);
    return (onnxDir / parallel_artifacts::onnxFileName(context)).string();
}

} // namespace

LLMBuilder::LLMBuilder(
    std::filesystem::path const& onnxDir, std::filesystem::path const& engineDir, LLMBuilderConfig const& config)
    : mOnnxDir(onnxDir)
    , mEngineDir(engineDir)
    , mBuilderConfig(config)
{
}

bool LLMBuilder::build()
{
    std::string trtVersion = std::to_string(NV_TENSORRT_MAJOR) + "." + std::to_string(NV_TENSORRT_MINOR) + "."
        + std::to_string(NV_TENSORRT_PATCH);
    LOG_INFO("Using TRT_VERSION=%s", trtVersion.c_str());
    std::string const lunowudFlags = applyCompileWorkarounds();
    if (!lunowudFlags.empty())
    {
        LOG_INFO("Using __LUNOWUD=%s", lunowudFlags.c_str());
    }

    // Load plugin library
    auto pluginHandles = loadEdgellmPluginLib();

    // Parse model config
    if (!parseConfig())
    {
        return false;
    }

    int64_t const minimumActivePages
        = rt::computeMinimumKvPoolPages(mBuilderConfig.maxBatchSize, mBuilderConfig.maxKVCacheCapacity);
    int64_t const kvPoolPages = mBuilderConfig.resolvedKVPoolPages();
    bool const hasExtraRetainedPages = kvPoolPages > minimumActivePages;
    std::string const mode = specDecodeType(mModelConfig);
    bool const attentionOnlyReusableSpec = mNumLinearAttnLayers == 0
        && (mode == "eagle3" || mode == "gemma4_mtp" || mode == "dflash" || mode == "dflash2" || mode == "jetspec"
            || mode == "dspark");
    bool const supportsCrossRequestRetention = mNbKVCacheInputs > 0 && (mode == "none" || attentionOnlyReusableSpec);
    if (hasExtraRetainedPages && !supportsCrossRequestRetention)
    {
        LOG_ERROR(
            "maxKVPoolPages=%ld adds extra retained pages for cross-request retention, but the engine configuration "
            "(engine_role=%s, spec_decode_type=%s, num_linear_attn_layers=%d) does not support cross-request "
            "retention. Use maxKVPoolPages=0 (resolved minimum active pages=%ld).",
            kvPoolPages, engineRole(mModelConfig).c_str(), mode.c_str(), mNumLinearAttnLayers, minimumActivePages);
        return false;
    }

    // Create builder and network
    auto [builder, network] = createBuilderAndNetwork();
    if (!builder || !network)
    {
        return false;
    }

    // Determine ONNX file path
    std::string onnxFilePath;
    if (mBuilderConfig.maxLoraRank > 0)
    {
        onnxFilePath = (mOnnxDir / "lora_model.onnx").string();
        LOG_INFO("Parsing LoRA-enabled ONNX model: %s", onnxFilePath.c_str());
    }
    else if (mBuilderConfig.tpSize > 1)
    {
        onnxFilePath = getOnnxFilePath(mOnnxDir, mBuilderConfig);
        LOG_INFO("Parsing rank-local ONNX model: %s", onnxFilePath.c_str());
    }
    else
    {
        onnxFilePath = (mOnnxDir / "model.onnx").string();
        LOG_INFO("Parsing ONNX model: %s", onnxFilePath.c_str());
    }

    // Parse ONNX model
    auto parser = parseOnnxModel(network.get(), onnxFilePath);
    if (!parser)
    {
        return false;
    }

    // Print network information
    LOG_DEBUG("%s", printNetworkInfo(network.get(), "LLM").c_str());

    LOG_DEBUG(
        "ONNX parsing complete. mNbKVCacheInputs=%d, mNumLinearAttnLayers=%d", mNbKVCacheInputs, mNumLinearAttnLayers);

    ELLM_CHECK(hasInputBinding(*network, binding_names::kPositions),
        "LLMBuilder: every common decoder ONNX must expose the unified token-major ABI. Re-export the model.");
    ELLM_CHECK(getInputRank(*network, binding_names::kInputsEmbeds) == 2,
        "LLMBuilder: inputs_embeds must be rank-2 [physical_tokens, hidden_size]. Re-export the model.");

    // Create builder config
    auto config = createBuilderConfig(builder.get(), usesAliasedPluginIO(mModelConfig));
    if (!config)
    {
        return false;
    }

    if (mBuilderConfig.profilingDetailed)
    {
        config->setProfilingVerbosity(nvinfer1::ProfilingVerbosity::kDETAILED);
        LOG_INFO("Profiling verbosity set to DETAILED");
    }

    LOG_DEBUG("Builder config created. Setting up optimization profiles...");

    // Setup optimization profiles
    if (!setupLLMOptimizationProfiles(*builder.get(), *config.get(), *network.get()))
    {
        return false;
    }

    std::error_code errorCode;
    bool const createdEngineDir = std::filesystem::create_directories(mEngineDir, errorCode);
    if (errorCode)
    {
        LOG_ERROR("Failed to create directory %s: %s", mEngineDir.string().c_str(), errorCode.message().c_str());
        return false;
    }
    if (createdEngineDir)
    {
        LOG_INFO("Created directory %s for saving LLM engine.", mEngineDir.string().c_str());
    }

    // Determine engine file name
    std::string const engineFileName = mIsDiffusionBackbone ? "dllm.engine" : getEngineFileName(mBuilderConfig);

    // Build and save engine
    std::string const engineFilePath = (mEngineDir / engineFileName).string();
    if (!buildAndSerializeEngine(builder.get(), network.get(), config.get(), engineFilePath))
    {
        return false;
    }

    // Detect number of deepstack embeds from network (for Qwen3VL models)
    mNumDeepstackFeatures = 0;
    for (int32_t idx = 0; idx < network->getNbInputs(); idx++)
    {
        std::string_view const inputName = network->getInput(idx)->getName();
        if (inputName.find(binding_names::kDeepstackEmbedsTemplate) != std::string_view::npos)
        {
            mNumDeepstackFeatures++;
        }
    }
    if (mNumDeepstackFeatures > 0)
    {
        LOG_INFO("Detected %d deepstack embedding inputs in network (Qwen3VL model)", mNumDeepstackFeatures);
    }

    // The world-level config is shared across ranks. Only rank 0 writes it.
    if ((mBuilderConfig.tpSize == 1 || mBuilderConfig.tpRank == 0) && !copyConfig())
    {
        return false;
    }

    // Shared files are identical across ranks. Only rank 0 copies them to avoid race conditions.
    if (mBuilderConfig.tpRank == 0 || mBuilderConfig.tpSize == 1)
    {
        if (!copyTokenizerFiles())
        {
            return false;
        }

        if (!copyEagleFiles())
        {
            return false;
        }

        if (!copyDSparkFiles())
        {
            return false;
        }

        if (!copyDFlash2Files())
        {
            return false;
        }

        if (!copyVocabMappingFiles())
        {
            return false;
        }

        if (!copyEmbeddingFile())
        {
            return false;
        }
    }

    if (!copyExternalWeightFiles())
    {
        return false;
    }

    return true;
}

bool LLMBuilder::parseConfig()
{
    std::string const jsonPath = (mOnnxDir / getInputConfigFileName(mBuilderConfig)).string();
    if (!loadJsonConfig(jsonPath, mModelConfig))
    {
        return false;
    }
    if (!validateRankConfigs(mModelConfig, mBuilderConfig))
    {
        return false;
    }
    mSharedModelConfig = mModelConfig;

    if (!applyRankConfigOverrides(mModelConfig, configRank(mBuilderConfig)))
    {
        return false;
    }

    // Check model version
    std::string modelVersion = mModelConfig.value(binding_names::kEdgellmVersion, "");
    version::checkVersion(modelVersion);

    std::string const specType = specDecodeType(mModelConfig);
    std::string const role = engineRole(mModelConfig);
    if (!isValidSpecDecodeType(specType))
    {
        LOG_ERROR(
            "Invalid spec_decode_type='%s'. Expected one of: none, mtp, eagle3, dflash, jetspec, dspark, "
            "gemma4_mtp.",
            specType.c_str());
        return false;
    }
    if (!isValidEngineRole(role))
    {
        LOG_ERROR("Invalid engine_role='%s'. Expected one of: llm, base, draft, dllm.", role.c_str());
        return false;
    }
    bool const isSpecRole = role == "base" || role == "draft";
    mIsDiffusionBackbone = role == "dllm";
    if ((role == "llm" || role == "dllm") && specType != "none")
    {
        LOG_ERROR(
            "Invalid config: engine_role='%s' with spec_decode_type='%s'. Non-speculative engines require "
            "spec_decode_type=none.",
            role.c_str(), specType.c_str());
        return false;
    }
    if (isSpecRole && specType == "none")
    {
        LOG_ERROR(
            "Invalid config: engine_role='%s' with spec_decode_type='%s'. Speculative base/draft engines require "
            "a non-none spec_decode_type.",
            role.c_str(), specType.c_str());
        return false;
    }
    bool const nonSpecBuild = !mBuilderConfig.specDraft && !mBuilderConfig.specBase;
    bool const nonSpecRole = role == "llm" || role == "dllm";
    if ((mBuilderConfig.specDraft && role != "draft") || (mBuilderConfig.specBase && role != "base")
        || (nonSpecBuild && !nonSpecRole))
    {
        LOG_ERROR(
            "Build mode does not match config: engine_role='%s' (use --specBase for base, --specDraft for "
            "draft, and neither flag for non-speculative engines).",
            role.c_str());
        return false;
    }
    if (mBuilderConfig.numSwaPages == 0 && mModelConfig.contains("num_swa_pages"))
    {
        mBuilderConfig.numSwaPages = mModelConfig["num_swa_pages"].get<int64_t>();
    }

    mHiddenSize = mModelConfig["hidden_size"].get<int32_t>();
    // MTP draft consumes one target hidden state. Other draft modes may
    // concatenate multiple target layers; prefer the exported contract when it
    // is present and keep the legacy EAGLE3 hidden_size*3 fallback below.
    if (isSpecDecodeDraft(mModelConfig, "mtp"))
    {
        mTargetModelOutputHiddenDim = mHiddenSize;
    }
    else if ((isSpecDecodeDraft(mModelConfig, "eagle3") || isSpecDecodeDraft(mModelConfig, "dflash")
                 || isSpecDecodeDraft(mModelConfig, "dflash2") || isSpecDecodeDraft(mModelConfig, "jetspec")
                 || isSpecDecodeDraft(mModelConfig, "dspark") || isSpecDecodeDraft(mModelConfig, "gemma4_mtp"))
        && mModelConfig.contains("base_model_hidden_size"))
    {
        mTargetModelOutputHiddenDim = mModelConfig["base_model_hidden_size"].get<int32_t>();
    }
    else
    {
        mTargetModelOutputHiddenDim = mHiddenSize * 3;
    }
    mNumKVHeads = mModelConfig["num_key_value_heads"].get<int32_t>();
    auto numAttentionHeads = mModelConfig["num_attention_heads"].get<int32_t>();

    if (mModelConfig.contains("head_dim"))
    {
        mHeadSize = mModelConfig["head_dim"].get<int32_t>();
    }
    else
    {
        mHeadSize = mHiddenSize / numAttentionHeads;
    }

    mNumLinearAttnLayers = mModelConfig.value("num_linear_attn_layers", 0);
    mRecurrentStateNumHeads = mModelConfig.value("recurrent_state_num_heads", 0);
    mRecurrentStateHeadDim = mModelConfig.value("recurrent_state_head_dim", 0);
    mRecurrentStateSize = mModelConfig.value("recurrent_state_size", 0);
    mConvDim = mModelConfig.value("conv_dim", 0);
    mConvKernel = mModelConfig.value("conv_kernel", 0);

    if (mModelConfig.contains("kv_layer_configs") && mModelConfig["kv_layer_configs"].is_array())
    {
        mNbKVCacheInputs = 0;
        for (auto const& layerConfig : mModelConfig["kv_layer_configs"])
        {
            if (layerConfig.is_object())
            {
                ++mNbKVCacheInputs;
            }
        }
    }
    else if (mNumLinearAttnLayers > 0)
    {
        // For hybrid models, only attention layers have KV caches.
        mNbKVCacheInputs = mModelConfig.value("num_attention_layers", mModelConfig["num_hidden_layers"].get<int32_t>());
    }
    else
    {
        mNbKVCacheInputs = mModelConfig["num_hidden_layers"].get<int32_t>();
    }

    mRotaryDim = getRotaryDim(mModelConfig, mHeadSize);
    mSlidingRotaryDim = mRotaryDim;
    mFullRotaryDim = mRotaryDim;
    if (mModelConfig.contains("sliding_rope_config") && mModelConfig["sliding_rope_config"].is_object())
    {
        mSlidingRotaryDim = getRotaryDim(mModelConfig["sliding_rope_config"], mHeadSize);
    }
    if (mModelConfig.contains("full_rope_config") && mModelConfig["full_rope_config"].is_object())
    {
        int64_t const fullHeadDim = mModelConfig.value("global_head_dim", static_cast<int64_t>(mHeadSize));
        mFullRotaryDim = getRotaryDim(mModelConfig["full_rope_config"], fullHeadDim);
    }

    // Build per-layer head size vector for heterogeneous models (e.g. Gemma4).
    // Prefer kv_layer_configs (authoritative per-layer dims) when available;
    // fall back to global_head_dim + layer_types for older exports.
    mPerLayerHeadSize.clear();
    mPerLayerNumKVHeads.clear();
    mPerLayerKVCacheCapacity.clear();
    int64_t globalHeadSize = mModelConfig.value("global_head_dim", static_cast<int64_t>(0));
    if (mBuilderConfig.maxBatchSize <= 0
        || mBuilderConfig.maxBatchSize > static_cast<int64_t>(std::numeric_limits<int32_t>::max())
        || mBuilderConfig.maxKVCacheCapacity <= 0
        || mBuilderConfig.maxKVCacheCapacity > static_cast<int64_t>(std::numeric_limits<int32_t>::max())
        || mBuilderConfig.numSwaPages < 0
        || mBuilderConfig.numSwaPages > static_cast<int64_t>(std::numeric_limits<int32_t>::max()))
    {
        LOG_ERROR(
            "maxBatchSize/maxKVCacheCapacity must be positive int32 values and numSwaPages must be a "
            "non-negative int32 value.");
        return false;
    }
    int32_t const maxKVCacheCapacity = static_cast<int32_t>(mBuilderConfig.maxKVCacheCapacity);

    if (mModelConfig.contains("kv_layer_configs") && mModelConfig["kv_layer_configs"].is_array())
    {
        auto const& kvLayerConfigs = mModelConfig["kv_layer_configs"];
        for (size_t configIdx = 0; configIdx < kvLayerConfigs.size(); ++configIdx)
        {
            auto const& lc = kvLayerConfigs[configIdx];
            if (lc.is_null())
            {
                continue;
            }
            if (!lc.is_object())
            {
                LOG_ERROR("kv_layer_configs[%zu] must be an object or null.", configIdx);
                return false;
            }
            int64_t const layerHeadDim = lc.value("head_dim", mHeadSize);
            mPerLayerHeadSize.push_back(layerHeadDim);
            int64_t const layerNumKVHeads = lc.value("num_kv_heads", mNumKVHeads);
            mPerLayerNumKVHeads.push_back(layerNumKVHeads);

            if (lc.contains("kv_cache_capacity") && !lc["kv_cache_capacity"].is_number_integer())
            {
                LOG_ERROR("kv_layer_configs[%zu].kv_cache_capacity must be an integer.", configIdx);
                return false;
            }
            int64_t const capacity = lc.value("kv_cache_capacity", static_cast<int64_t>(0));
            if (capacity < 0 || capacity > maxKVCacheCapacity)
            {
                LOG_ERROR("kv_layer_configs[%zu].kv_cache_capacity=%ld must be in [0, %d].", configIdx, capacity,
                    maxKVCacheCapacity);
                return false;
            }
            mPerLayerKVCacheCapacity.push_back(static_cast<int32_t>(capacity));
        }
        check::check(static_cast<int32_t>(mPerLayerHeadSize.size()) == mNbKVCacheInputs,
            "kv_layer_configs attention-entry count does not match the expected KV cache input count");
        LOG_INFO("Heterogeneous head sizes from kv_layer_configs: %d layers", mNbKVCacheInputs);
    }
    else
    {
        mPerLayerKVCacheCapacity.assign(static_cast<size_t>(mNbKVCacheInputs), 0);
        if (globalHeadSize > 0 && globalHeadSize != mHeadSize && mModelConfig.contains("layer_types"))
        {
            auto const& layerTypes = mModelConfig["layer_types"];
            for (int i = 0; i < mNbKVCacheInputs; ++i)
            {
                std::string lt = (i < static_cast<int>(layerTypes.size())) ? layerTypes[i].get<std::string>() : "";
                mPerLayerHeadSize.push_back((lt == "full_attention") ? globalHeadSize : mHeadSize);
            }
            LOG_INFO("Heterogeneous head sizes: %d layers with head_dim=%ld, %ld layers with global_head_dim=%ld",
                mNbKVCacheInputs, mHeadSize,
                std::count(mPerLayerHeadSize.begin(), mPerLayerHeadSize.end(), globalHeadSize), globalHeadSize);
        }
    }

    std::optional<int32_t> reducedCapacity;
    for (int32_t const capacity : mPerLayerKVCacheCapacity)
    {
        int32_t const resolvedCapacity = rt::resolveKvCacheCapacity(capacity, maxKVCacheCapacity);
        if (resolvedCapacity == maxKVCacheCapacity)
        {
            continue;
        }
        if (reducedCapacity.has_value() && *reducedCapacity != resolvedCapacity)
        {
            LOG_ERROR("All reduced KV layers must use one common non-full kv_cache_capacity.");
            return false;
        }
        reducedCapacity = resolvedCapacity;
    }
    if (reducedCapacity.has_value())
    {
        if (mModelConfig.value("kv_cache_dtype", "fp16") == "fp8")
        {
            LOG_ERROR("Reduced SWA KV pools do not support FP8 KV cache.");
            return false;
        }
        if (specType != "none" || configRevealsSpecDecode(mModelConfig) || mBuilderConfig.specBase
            || mBuilderConfig.specDraft)
        {
            LOG_ERROR("Reduced SWA KV pools do not support speculative decoding.");
            return false;
        }
        try
        {
            bool const autoSized = mBuilderConfig.numSwaPages == 0;
            int32_t const resolvedSwaPages = mBuilderConfig.resolveNumSwaPages(*reducedCapacity);
            if (autoSized)
            {
                LOG_INFO("Auto-sized numSwaPages=%d for maxBatchSize=%ld and kv_cache_capacity=%d.", resolvedSwaPages,
                    mBuilderConfig.maxBatchSize, *reducedCapacity);
            }
        }
        catch (std::exception const& error)
        {
            LOG_ERROR("Invalid numSwaPages for reduced SWA pool: %s", error.what());
            return false;
        }
    }

    if (mModelConfig.contains("diffusion_config") && mModelConfig["diffusion_config"].is_object())
    {
        mDiffusionCanvasLength = mModelConfig["diffusion_config"].value("canvas_length", static_cast<int64_t>(0));
    }
    mDiffusionCanvasLength = mModelConfig.value("diffusion_canvas_length", mDiffusionCanvasLength);
    if (mIsDiffusionBackbone)
    {
        mBuilderConfig.maxQueryLength
            = std::max({mBuilderConfig.maxQueryLength, mDiffusionCanvasLength, mBuilderConfig.maxInputLen});
    }

    return true;
}

bool LLMBuilder::setupLLMOptimizationProfiles(
    nvinfer1::IBuilder& builder, nvinfer1::IBuilderConfig& config, nvinfer1::INetworkDefinition const& network)
{
    auto* contextProfile = builder.createOptimizationProfile();
    auto* generationProfile = builder.createOptimizationProfile();

    bool result = true;

    if (isSpecDecodeDraft(mModelConfig, "dflash") || isSpecDecodeDraft(mModelConfig, "dflash2")
        || isSpecDecodeDraft(mModelConfig, "jetspec"))
    {
        result &= setupDFlashDraftProfiles(*contextProfile, *generationProfile);
        if (!result)
        {
            LOG_ERROR("Failed to setup DFlash/DFlash2/JetSpec draft optimization profiles");
            return false;
        }
        LOG_DEBUG("%s", printOptimizationProfile(contextProfile, "context_profile", &network).c_str());
        LOG_DEBUG("%s", printOptimizationProfile(generationProfile, "generation_profile", &network).c_str());
        config.addOptimizationProfile(contextProfile);
        config.addOptimizationProfile(generationProfile);
        return true;
    }

    if (isSpecDecodeDraft(mModelConfig, "gemma4_mtp"))
    {
        result &= setupGemma4MTPDraftProfiles(*contextProfile, *generationProfile, network);
        if (!result)
        {
            LOG_ERROR("Failed to setup Gemma4 MTP draft optimization profiles");
            return false;
        }
        LOG_DEBUG("%s", printOptimizationProfile(contextProfile, "context_profile", &network).c_str());
        LOG_DEBUG("%s", printOptimizationProfile(generationProfile, "generation_profile", &network).c_str());
        config.addOptimizationProfile(contextProfile);
        config.addOptimizationProfile(generationProfile);
        return true;
    }

    if (isSpecDecodeDraft(mModelConfig, "dspark"))
    {
        result &= setupDSparkDraftProfiles(*contextProfile, *generationProfile);
        if (!result)
        {
            LOG_ERROR("Failed to setup DSpark draft optimization profiles");
            return false;
        }
        LOG_DEBUG("%s", printOptimizationProfile(contextProfile, "context_profile", &network).c_str());
        LOG_DEBUG("%s", printOptimizationProfile(generationProfile, "generation_profile", &network).c_str());
        config.addOptimizationProfile(contextProfile);
        config.addOptimizationProfile(generationProfile);
        return true;
    }

    // Setup common profiles
    result &= setupCommonProfiles(*contextProfile, *generationProfile, network);
    result &= setupRopeProfiles(*contextProfile, *generationProfile, network);

    // Setup model-specific profiles
    if (mBuilderConfig.specBase || mBuilderConfig.specDraft)
    {
        result &= setupSpecDecodeProfiles(*contextProfile, *generationProfile, network);
    }
    else if (mIsDiffusionBackbone)
    {
        result &= setupDiffusionBackboneProfiles(*contextProfile, *generationProfile, network);
    }
    else
    {
        result &= setupVanillaProfiles(*contextProfile, *generationProfile, network);
    }

    // Setup hybrid state profiles for MTP/DFlash/JetSpec/DSpark base models.
    if (isSpecDecodeBase(mModelConfig, "mtp") || isSpecDecodeBase(mModelConfig, "dflash")
        || isSpecDecodeBase(mModelConfig, "dflash2") || isSpecDecodeBase(mModelConfig, "jetspec")
        || isSpecDecodeBase(mModelConfig, "dspark"))
    {
        result &= setupIntermediateRecurrentStateProfiles(*contextProfile, *generationProfile);
        result &= setupIntermediateConvStateProfiles(*contextProfile, *generationProfile);
        result &= setupLinearAttentionSpecVerifyProfiles(*contextProfile, *generationProfile, network);
    }

    // Setup Gemma4 PLE profiles when ple_token_embeds_* inputs are present.
    result &= setupPleProfiles(*contextProfile, *generationProfile, network);

    // Setup Deepstack profiles for Qwen3VL models
    result &= setupDeepstackProfiles(*contextProfile, *generationProfile, network);

    // Setup lm_head_weight profile for CodePredictor (Qwen3-Omni)
    result &= setupLmHeadWeightProfiles(*contextProfile, *generationProfile, network);

    if (mBuilderConfig.maxLoraRank > 0)
    {
        result &= setupLoraProfiles(*contextProfile, *generationProfile, network);
    }

    if (!result)
    {
        LOG_ERROR("Failed to setup optimization profiles");
        return false;
    }

    LOG_DEBUG("%s", printOptimizationProfile(contextProfile, "context_profile", &network).c_str());
    LOG_DEBUG("%s", printOptimizationProfile(generationProfile, "generation_profile", &network).c_str());

    config.addOptimizationProfile(contextProfile);
    config.addOptimizationProfile(generationProfile);

    return true;
}

bool LLMBuilder::setupCommonProfiles(nvinfer1::IOptimizationProfile& contextProfile,
    nvinfer1::IOptimizationProfile& generationProfile, nvinfer1::INetworkDefinition const& network)
{
    bool result = true;

    // Some model graphs retain context_lengths as an auxiliary input; ragged decoder plugins use the query metadata.
    if (hasInputBinding(network, binding_names::kContextLengths))
    {
        result &= setOptimizationProfile(&contextProfile, binding_names::kContextLengths, createDims({1}),
            createDims({mBuilderConfig.maxBatchSize}), createDims({mBuilderConfig.maxBatchSize}));
        result &= setOptimizationProfile(&generationProfile, binding_names::kContextLengths, createDims({1}),
            createDims({mBuilderConfig.maxBatchSize}), createDims({mBuilderConfig.maxBatchSize}));
    }

    // Autoregressive engines use shape [0] as the initial-prefill empty-KV sentinel.
    // DiffusionGemma keeps kvcache_start_index materialized and uses context_mask_selector as its mask sentinel.
    nvinfer1::Dims const contextKvStartMin = mIsDiffusionBackbone ? createDims({1}) : createDims({0});
    if (hasInputBinding(network, binding_names::kKVCacheStartIndex))
    {
        result &= setOptimizationProfile(&contextProfile, binding_names::kKVCacheStartIndex, contextKvStartMin,
            createDims({mBuilderConfig.maxBatchSize}), createDims({mBuilderConfig.maxBatchSize}));
        result &= setOptimizationProfile(&generationProfile, binding_names::kKVCacheStartIndex, createDims({1}),
            createDims({mBuilderConfig.maxBatchSize}), createDims({mBuilderConfig.maxBatchSize}));
    }

    // kv_page_table: [batch, 2, maxPagesPerSeq] int32. Per-request page table (default
    // identity => bit-equivalent to the non-paged path). Column count is fixed
    // (maxPagesPerSeq = ceil(maxKVCacheCapacity / kTOKENS_PER_PAGE)); batch is dynamic.
    int32_t const maxPagesPerSeq = rt::computeMaxPagesPerSeq(static_cast<int32_t>(mBuilderConfig.maxKVCacheCapacity));
    result &= setOptimizationProfile(&contextProfile, binding_names::kKVPageTable, createDims({1, 2, maxPagesPerSeq}),
        createDims({mBuilderConfig.maxBatchSize, 2, maxPagesPerSeq}),
        createDims({mBuilderConfig.maxBatchSize, 2, maxPagesPerSeq}));
    result &= setOptimizationProfile(&generationProfile, binding_names::kKVPageTable,
        createDims({1, 2, maxPagesPerSeq}), createDims({mBuilderConfig.maxBatchSize, 2, maxPagesPerSeq}),
        createDims({mBuilderConfig.maxBatchSize, 2, maxPagesPerSeq}));
    bool const hasReducedPool
        = std::any_of(mPerLayerKVCacheCapacity.begin(), mPerLayerKVCacheCapacity.end(), [&](int32_t capacity) {
              return rt::isReducedKvCacheCapacity(capacity, static_cast<int32_t>(mBuilderConfig.maxKVCacheCapacity));
          });
    bool const hasSwaPageTableInput = hasInputBinding(network, binding_names::kSwaKVPageTable);
    bool const hasSwaModeInput = hasInputBinding(network, binding_names::kSwaKVCacheMode);
    if (hasReducedPool != hasSwaPageTableInput || hasReducedPool != hasSwaModeInput)
    {
        LOG_ERROR(
            "Reduced kv_cache_capacity markers, swa_kv_page_table, and swa_kv_cache_mode must either all be present "
            "or all be absent.");
        return false;
    }
    if (hasSwaPageTableInput)
    {
        result &= setOptimizationProfile(&contextProfile, binding_names::kSwaKVPageTable,
            createDims({1, 2, maxPagesPerSeq}), createDims({mBuilderConfig.maxBatchSize, 2, maxPagesPerSeq}),
            createDims({mBuilderConfig.maxBatchSize, 2, maxPagesPerSeq}));
        result &= setOptimizationProfile(&generationProfile, binding_names::kSwaKVPageTable,
            createDims({1, 2, maxPagesPerSeq}), createDims({mBuilderConfig.maxBatchSize, 2, maxPagesPerSeq}),
            createDims({mBuilderConfig.maxBatchSize, 2, maxPagesPerSeq}));

        // Shape-only runtime selector: [1] uses bounded SWA storage and [0] uses the full KV pool.
        result &= setOptimizationProfile(
            &contextProfile, binding_names::kSwaKVCacheMode, createDims({0}), createDims({1}), createDims({1}));
        result &= setOptimizationProfile(
            &generationProfile, binding_names::kSwaKVCacheMode, createDims({0}), createDims({1}), createDims({1}));
    }

    // KV cache profiles
    LOG_DEBUG("Setting up KV cache profiles for %d layers...", mNbKVCacheInputs);
    result &= setupKVCacheProfiles(contextProfile, generationProfile);
    LOG_DEBUG("KV cache profiles done. Setting up recurrent state profiles for %d layers...", mNumLinearAttnLayers);

    // Recurrent state profiles for hybrid layers
    result &= setupRecurrentStateProfiles(&contextProfile, &generationProfile);

    LOG_DEBUG("Recurrent state profiles done. Setting up Conv state profiles...");
    // Conv state profiles for recurrent causal conv1d layers
    result &= setupConvStateProfiles(&contextProfile, &generationProfile);
    LOG_DEBUG("Conv state profiles done.");

    // skip_softmax_scale: [S] INT8 shape-only runtime skip-softmax override carrier.
    // dim0 IS the integer scale factor S; a meaningful S satisfies lambda < 1, i.e.
    // S < context length <= maxKVCacheCapacity.
    if (hasInputBinding(network, binding_names::kSkipSoftmaxScale))
    {
        result &= setOptimizationProfile(&contextProfile, binding_names::kSkipSoftmaxScale, createDims({0}),
            createDims({0}), createDims({mBuilderConfig.maxKVCacheCapacity}));
        result &= setOptimizationProfile(&generationProfile, binding_names::kSkipSoftmaxScale, createDims({0}),
            createDims({0}), createDims({mBuilderConfig.maxKVCacheCapacity}));
    }

    return result;
}

bool LLMBuilder::setupRopeProfiles(nvinfer1::IOptimizationProfile& contextProfile,
    nvinfer1::IOptimizationProfile& generationProfile, nvinfer1::INetworkDefinition const& network)
{
    bool result = true;
    auto const [prefill, generation] = tokenAlignedProfileRanges();
    auto setRopeProfile = [&](char const* bindingName, int64_t rotaryDim) {
        if (getInputRank(network, bindingName) != 2)
        {
            LOG_ERROR("RoPE input %s must be rank-2 [physical_tokens, rotary_dim].", bindingName);
            result = false;
            return;
        }
        result &= setOptimizationProfile(&contextProfile, bindingName,
            createDims({prefill.min.physicalTokens, rotaryDim}), createDims({prefill.opt.physicalTokens, rotaryDim}),
            createDims({prefill.max.physicalTokens, rotaryDim}));
        result &= setOptimizationProfile(&generationProfile, bindingName,
            createDims({generation.min.physicalTokens, rotaryDim}),
            createDims({generation.opt.physicalTokens, rotaryDim}),
            createDims({generation.max.physicalTokens, rotaryDim}));
    };

    // RoPE rotary cos/sin inputs: single binding for single-RoPE engines, or
    // explicit sliding/full bindings for mixed-attention engines.
    if (hasInputBinding(network, binding_names::kRopeCosSinSliding))
    {
        setRopeProfile(binding_names::kRopeCosSinSliding, mSlidingRotaryDim);
    }
    if (hasInputBinding(network, binding_names::kRopeCosSinFull))
    {
        setRopeProfile(binding_names::kRopeCosSinFull, mFullRotaryDim);
    }
    if (hasInputBinding(network, binding_names::kRopeCosSin))
    {
        setRopeProfile(binding_names::kRopeCosSin, mRotaryDim);
    }

    return result;
}

std::pair<RaggedProfileRange, RaggedProfileRange> LLMBuilder::tokenAlignedProfileRanges() const
{
    RaggedProfileRange const prefill = mBuilderConfig.raggedPrefillProfileRange();
    if (mBuilderConfig.specBase || mBuilderConfig.specDraft)
    {
        return {
            prefill, mBuilderConfig.raggedMultiTokenGenerationProfileRange(mBuilderConfig.resolvedRoleQueryLength())};
    }
    if (mIsDiffusionBackbone)
    {
        return {
            prefill, mBuilderConfig.raggedMultiTokenGenerationProfileRange(mBuilderConfig.resolvedRoleQueryLength())};
    }
    return {prefill, mBuilderConfig.raggedDecodeProfileRange()};
}

bool LLMBuilder::setupDiffusionBackboneProfiles(nvinfer1::IOptimizationProfile& contextProfile,
    nvinfer1::IOptimizationProfile& generationProfile, nvinfer1::INetworkDefinition const& network)
{
    bool result = true;

    int64_t const maxCanvasLen = mBuilderConfig.resolvedRoleQueryLength();
    int64_t const optPromptLen = std::max<int64_t>(1, mBuilderConfig.maxInputLen / 2);
    int64_t const contextOptTokens = mBuilderConfig.maxBatchSize * optPromptLen;
    int64_t const contextMaxTokens = mBuilderConfig.checkedPhysicalTokens(mBuilderConfig.maxInputLen);
    int64_t const generationTokens = mBuilderConfig.checkedPhysicalTokens(maxCanvasLen);

    // Token-aligned bindings have leading extent T_exec = batch size * canvas length.
    auto setTokenProfile = [&](char const* name, int64_t width = 0) {
        auto dims = [&](int64_t tokens) { return width == 0 ? createDims({tokens}) : createDims({tokens, width}); };
        result
            &= setOptimizationProfile(&contextProfile, name, dims(1), dims(contextOptTokens), dims(contextMaxTokens));
        result &= setOptimizationProfile(
            &generationProfile, name, dims(1), dims(generationTokens), dims(generationTokens));
    };
    // Sequence-aligned bindings have leading extent equal to the active sequence count.
    auto setSequenceProfile = [&](char const* name) {
        result &= setOptimizationProfile(&contextProfile, name, createDims({1}),
            createDims({mBuilderConfig.maxBatchSize}), createDims({mBuilderConfig.maxBatchSize}));
        result &= setOptimizationProfile(&generationProfile, name, createDims({1}),
            createDims({mBuilderConfig.maxBatchSize}), createDims({mBuilderConfig.maxBatchSize}));
    };

    setTokenProfile(binding_names::kInputsEmbeds, mHiddenSize);
    std::array<char const*, 3> const tokenAlignedBindings{
        binding_names::kPositions, binding_names::kSelectTokenIndices, binding_names::kCanvasIds};
    for (char const* name : tokenAlignedBindings)
    {
        if (hasInputBinding(network, name))
        {
            setTokenProfile(name);
        }
    }
    std::array<char const*, 4> const sequenceAlignedBindings{binding_names::kQueryLengths, binding_names::kPastLengths,
        binding_names::kAttentionSequenceLengths, binding_names::kStateIndices};
    for (char const* name : sequenceAlignedBindings)
    {
        setSequenceProfile(name);
    }
    result &= setOptimizationProfile(
        &contextProfile, binding_names::kPhaseIsEncoder, createDims({1}), createDims({1}), createDims({1}));
    result &= setOptimizationProfile(
        &generationProfile, binding_names::kPhaseIsEncoder, createDims({1}), createDims({1}), createDims({1}));
    result &= setOptimizationProfile(&contextProfile, binding_names::kQueryStartOffsets, createDims({2}),
        createDims({mBuilderConfig.maxBatchSize + 1}), createDims({mBuilderConfig.maxBatchSize + 1}));
    result &= setOptimizationProfile(&generationProfile, binding_names::kQueryStartOffsets, createDims({2}),
        createDims({mBuilderConfig.maxBatchSize + 1}), createDims({mBuilderConfig.maxBatchSize + 1}));
    result &= setOptimizationProfile(&contextProfile, binding_names::kContextSequenceCountCarrier, createDims({0}),
        createDims({0}), createDims({0}));
    result &= setOptimizationProfile(&generationProfile, binding_names::kContextSequenceCountCarrier, createDims({0}),
        createDims({0}), createDims({0}));
    result &= setOptimizationProfile(
        &contextProfile, binding_names::kExecutionPhaseMarker, createDims({1}), createDims({7}), createDims({8}));
    result &= setOptimizationProfile(
        &generationProfile, binding_names::kExecutionPhaseMarker, createDims({1}), createDims({6}), createDims({8}));

    if (hasInputBinding(network, binding_names::kContextMaskSelector))
    {
        result &= setOptimizationProfile(&contextProfile, binding_names::kContextMaskSelector, createDims({0}),
            createDims({mBuilderConfig.maxBatchSize}), createDims({mBuilderConfig.maxBatchSize}));
        result &= setOptimizationProfile(&generationProfile, binding_names::kContextMaskSelector, createDims({0}),
            createDims({mBuilderConfig.maxBatchSize}), createDims({mBuilderConfig.maxBatchSize}));
    }

    if (hasInputBinding(network, binding_names::kPrevSelfConditioningEmbeds))
    {
        setTokenProfile(binding_names::kPrevSelfConditioningEmbeds, mHiddenSize);
    }
    if (hasInputBinding(network, binding_names::kNextSelfConditioningEmbeds))
    {
        setTokenProfile(binding_names::kNextSelfConditioningEmbeds, mHiddenSize);
    }

    return result;
}

//! Widest verify window the CodePredictor runtime can request; must stay >= kCpSpecMaxK.
constexpr int64_t kCodePredictorMaxVerifyWindow = 8;

bool LLMBuilder::setupVanillaProfiles(nvinfer1::IOptimizationProfile& contextProfile,
    nvinfer1::IOptimizationProfile& generationProfile, nvinfer1::INetworkDefinition const& network)
{
    bool result = true;
    RaggedProfileRange const prefill = mBuilderConfig.raggedPrefillProfileRange();
    RaggedProfileRange decode = mBuilderConfig.raggedDecodeProfileRange();
    bool const codePredictor = isCodePredictor(mModelConfig);
    if (codePredictor)
    {
        decode.max.physicalTokens = mBuilderConfig.checkedPhysicalTokens(kCodePredictorMaxVerifyWindow);
    }
    auto setTokenProfile = [&](char const* name, int64_t width = 0) {
        auto dims = [&](int64_t tokens) { return width == 0 ? createDims({tokens}) : createDims({tokens, width}); };
        result &= setOptimizationProfile(&contextProfile, name, dims(prefill.min.physicalTokens),
            dims(prefill.opt.physicalTokens), dims(prefill.max.physicalTokens));
        result &= setOptimizationProfile(&generationProfile, name, dims(decode.min.physicalTokens),
            dims(decode.opt.physicalTokens), dims(decode.max.physicalTokens));
    };
    auto setSequenceProfile = [&](char const* name) {
        result &= setOptimizationProfile(&contextProfile, name, createDims({prefill.min.numSequences}),
            createDims({prefill.opt.numSequences}), createDims({prefill.max.numSequences}));
        result &= setOptimizationProfile(&generationProfile, name, createDims({decode.min.numSequences}),
            createDims({decode.opt.numSequences}), createDims({decode.max.numSequences}));
    };

    setTokenProfile(binding_names::kInputsEmbeds, mHiddenSize);
    setTokenProfile(binding_names::kPositions);
    if (hasInputBinding(network, binding_names::kVisionBlockIds))
    {
        setTokenProfile(binding_names::kVisionBlockIds);
    }
    for (char const* name : {binding_names::kQueryLengths, binding_names::kPastLengths,
             binding_names::kAttentionSequenceLengths, binding_names::kStateIndices})
    {
        if (hasInputBinding(network, name))
        {
            setSequenceProfile(name);
        }
    }
    if (hasInputBinding(network, binding_names::kPhaseIsEncoder))
    {
        result &= setOptimizationProfile(
            &contextProfile, binding_names::kPhaseIsEncoder, createDims({1}), createDims({1}), createDims({1}));
        result &= setOptimizationProfile(
            &generationProfile, binding_names::kPhaseIsEncoder, createDims({1}), createDims({1}), createDims({1}));
    }
    result &= setOptimizationProfile(&contextProfile, binding_names::kQueryStartOffsets,
        createDims({prefill.min.queryOffsets}), createDims({prefill.opt.queryOffsets}),
        createDims({prefill.max.queryOffsets}));
    result &= setOptimizationProfile(&generationProfile, binding_names::kQueryStartOffsets,
        createDims({decode.min.queryOffsets}), createDims({decode.opt.queryOffsets}),
        createDims({decode.max.queryOffsets}));
    result &= setOptimizationProfile(&contextProfile, binding_names::kContextSequenceCountCarrier,
        createDims({prefill.min.numSequences}), createDims({prefill.opt.numSequences}),
        createDims({prefill.max.numSequences}));
    result &= setOptimizationProfile(&generationProfile, binding_names::kContextSequenceCountCarrier, createDims({0}),
        createDims({0}), createDims({codePredictor ? decode.max.numSequences : 0}));
    result
        &= setOptimizationProfile(&contextProfile, binding_names::kLogitsIndices, createDims({prefill.min.logitsRows}),
            createDims({prefill.opt.logitsRows}), createDims({prefill.max.logitsRows}));
    result &= setOptimizationProfile(&generationProfile, binding_names::kLogitsIndices,
        createDims({decode.min.logitsRows}), createDims({decode.opt.logitsRows}), createDims({decode.max.logitsRows}));
    result &= setOptimizationProfile(
        &contextProfile, binding_names::kExecutionPhaseMarker, createDims({1}), createDims({1}), createDims({8}));
    result &= setOptimizationProfile(
        &generationProfile, binding_names::kExecutionPhaseMarker, createDims({1}), createDims({3}), createDims({8}));

    return result;
}

int64_t LLMBuilder::effectiveOptTokens(int64_t maxToken) const
{
    return std::max<int64_t>(1, maxToken);
}

bool LLMBuilder::setupSpecDecodeProfiles(nvinfer1::IOptimizationProfile& contextProfile,
    nvinfer1::IOptimizationProfile& generationProfile, nvinfer1::INetworkDefinition const& network)
{
    bool result = true;
    auto const [prefill, generation] = tokenAlignedProfileRanges();
    int64_t const maxTokens = mBuilderConfig.resolvedRoleQueryLength();
    int const optTokens = static_cast<int>(effectiveOptTokens(maxTokens));
    int64_t const contextOptTokens
        = std::max<int64_t>(1, mBuilderConfig.maxBatchSize * std::max<int64_t>(1, mBuilderConfig.maxInputLen / 2));
    int64_t const contextMaxTokens = mBuilderConfig.checkedPhysicalTokens(mBuilderConfig.maxInputLen);
    int64_t const generationOptTokens = std::max<int64_t>(1, mBuilderConfig.maxBatchSize * optTokens);
    int64_t const generationMaxTokens = mBuilderConfig.checkedPhysicalTokens(maxTokens);
    int64_t const contextOptSequences = std::max<int64_t>(1, mBuilderConfig.maxBatchSize);

    auto setTokenProfile = [&](char const* name, int64_t width = 0) {
        auto dims = [&](int64_t tokens) { return width == 0 ? createDims({tokens}) : createDims({tokens, width}); };
        result
            &= setOptimizationProfile(&contextProfile, name, dims(1), dims(contextOptTokens), dims(contextMaxTokens));
        result &= setOptimizationProfile(
            &generationProfile, name, dims(1), dims(generationOptTokens), dims(generationMaxTokens));
    };
    auto setSequenceProfile = [&](char const* name) {
        result &= setOptimizationProfile(&contextProfile, name, createDims({1}), createDims({contextOptSequences}),
            createDims({contextOptSequences}));
        result &= setOptimizationProfile(&generationProfile, name, createDims({1}),
            createDims({mBuilderConfig.maxBatchSize}), createDims({mBuilderConfig.maxBatchSize}));
    };

    setTokenProfile(binding_names::kInputsEmbeds, mHiddenSize);
    for (char const* name : {binding_names::kPositions, binding_names::kAttentionPosId, binding_names::kTreeParentIds,
             binding_names::kTreeDepths})
    {
        if (hasInputBinding(network, name))
        {
            setTokenProfile(name);
        }
    }
    for (char const* name : {binding_names::kQueryLengths, binding_names::kPastLengths,
             binding_names::kAttentionSequenceLengths, binding_names::kStateIndices})
    {
        setSequenceProfile(name);
    }
    result &= setOptimizationProfile(&contextProfile, binding_names::kQueryStartOffsets, createDims({2}),
        createDims({contextOptSequences + 1}), createDims({contextOptSequences + 1}));
    result &= setOptimizationProfile(&generationProfile, binding_names::kQueryStartOffsets, createDims({2}),
        createDims({mBuilderConfig.maxBatchSize + 1}), createDims({mBuilderConfig.maxBatchSize + 1}));
    result &= setOptimizationProfile(&contextProfile, binding_names::kContextSequenceCountCarrier, createDims({1}),
        createDims({contextOptSequences}), createDims({contextOptSequences}));
    result &= setOptimizationProfile(&generationProfile, binding_names::kContextSequenceCountCarrier, createDims({0}),
        createDims({0}), createDims({0}));
    if (hasInputBinding(network, binding_names::kLogitsIndices))
    {
        result &= setOptimizationProfile(&contextProfile, binding_names::kLogitsIndices,
            createDims({prefill.min.logitsRows}), createDims({prefill.opt.logitsRows}),
            createDims({prefill.max.logitsRows}));
        result &= setOptimizationProfile(&generationProfile, binding_names::kLogitsIndices,
            createDims({generation.min.logitsRows}), createDims({generation.opt.logitsRows}),
            createDims({generation.max.logitsRows}));
    }
    if (hasInputBinding(network, binding_names::kValidTreeCounts))
    {
        setSequenceProfile(binding_names::kValidTreeCounts);
    }

    if (mBuilderConfig.specDraft)
    {
        setTokenProfile(binding_names::kDraftModelHiddenStates, mHiddenSize);
        setTokenProfile(binding_names::kBaseModelHiddenStates, mTargetModelOutputHiddenDim);
    }

    int64_t const contextMaskWidth = 1;
    int64_t const generationOptMaskWidth = divUp(optTokens, 32);
    int64_t const generationMaxMaskWidth = divUp(maxTokens, 32);
    result &= setOptimizationProfile(&contextProfile, binding_names::kAttentionMask, createDims({1, contextMaskWidth}),
        createDims({contextOptTokens, contextMaskWidth}), createDims({contextMaxTokens, contextMaskWidth}));
    result &= setOptimizationProfile(&generationProfile, binding_names::kAttentionMask, createDims({1, 1}),
        createDims({generationOptTokens, generationOptMaskWidth}),
        createDims({generationMaxTokens, generationMaxMaskWidth}));
    result &= setOptimizationProfile(
        &contextProfile, binding_names::kExecutionPhaseMarker, createDims({1}), createDims({1}), createDims({8}));
    result &= setOptimizationProfile(&generationProfile, binding_names::kExecutionPhaseMarker, createDims({1}),
        createDims({mBuilderConfig.specDraft ? 4 : 5}), createDims({8}));

    return result;
}

bool LLMBuilder::setupDFlashDraftProfiles(
    nvinfer1::IOptimizationProfile& contextProfile, nvinfer1::IOptimizationProfile& generationProfile)
{
    bool result = true;

    bool const isDFlash2 = isDFlashV2DraftConfig(mModelConfig);
    int64_t const maxDraftTokens = std::max<int64_t>(1, mBuilderConfig.maxDraftTreeSize);
    if (isDFlash2 && (maxDraftTokens < 2 || maxDraftTokens > 16))
    {
        LOG_ERROR("DFlash2 requires maxDraftTreeSize in [2, 16]");
        return false;
    }
    int64_t const optDraftTokens = maxDraftTokens;
    int64_t const maxPrefillTargetHiddenLen = std::max<int64_t>(1, mBuilderConfig.maxInputLen);
    int64_t const optPrefillTargetHiddenLen = std::max<int64_t>(1, maxPrefillTargetHiddenLen / 2);
    // DFlash verifies [anchor] + proposal tokens and can accept the base
    // bonus token, so the next draft round may need one more target-hidden row
    // than the proposal block.
    int64_t const maxDecodeTargetHiddenLen = isDFlash2 ? maxDraftTokens : maxDraftTokens + 1;
    int64_t const optDecodeTargetHiddenLen = maxDecodeTargetHiddenLen;

    int64_t const packedMaskLen = static_cast<int64_t>(divUp(maxDraftTokens, 32));
    int64_t const optPackedMaskLen = static_cast<int64_t>(divUp(optDraftTokens, 32));

    // Profile 0 handles round-0/system-prompt cache update, where target hidden spans the prompt.
    // Profile 1 handles steady-state block proposal, where target hidden delta is bounded by block size.
    auto setupOneProfile = [&](nvinfer1::IOptimizationProfile& profile, int64_t optTargetHiddenLen,
                               int64_t maxTargetHiddenLen) {
        bool ok = true;
        int64_t const minProposalRows = isDFlash2 ? 2 : 1;
        int64_t const optProposalRows = mBuilderConfig.maxBatchSize * optDraftTokens;
        int64_t const maxProposalRows = mBuilderConfig.maxBatchSize * maxDraftTokens;
        int64_t const optDeltaRows = mBuilderConfig.maxBatchSize * optTargetHiddenLen;
        int64_t const maxDeltaRows = mBuilderConfig.maxBatchSize * maxTargetHiddenLen;
        ok &= setOptimizationProfile(&profile, binding_names::kInputsEmbeds, createDims({minProposalRows, mHiddenSize}),
            createDims({optProposalRows, mHiddenSize}), createDims({maxProposalRows, mHiddenSize}));
        ok &= setOptimizationProfile(&profile, binding_names::kDFlashTargetHiddenConcat,
            createDims({1, mTargetModelOutputHiddenDim}), createDims({optDeltaRows, mTargetModelOutputHiddenDim}),
            createDims({maxDeltaRows, mTargetModelOutputHiddenDim}));
        ok &= setOptimizationProfile(&profile, binding_names::kRopeCosSin, createDims({1, mRotaryDim}),
            createDims({optProposalRows, mRotaryDim}), createDims({maxProposalRows, mRotaryDim}));
        ok &= setOptimizationProfile(&profile, binding_names::kDFlashDeltaRopeCosSin, createDims({1, mRotaryDim}),
            createDims({optDeltaRows, mRotaryDim}), createDims({maxDeltaRows, mRotaryDim}));
        for (char const* name : {binding_names::kPositions, binding_names::kAttentionPosId})
        {
            ok &= setOptimizationProfile(
                &profile, name, createDims({1}), createDims({optProposalRows}), createDims({maxProposalRows}));
        }
        for (char const* name : {binding_names::kDFlashDeltaPositions, binding_names::kDFlashDeltaTokenToSequence})
        {
            ok &= setOptimizationProfile(
                &profile, name, createDims({1}), createDims({optDeltaRows}), createDims({maxDeltaRows}));
        }
        for (char const* name : {binding_names::kQueryLengths, binding_names::kPastLengths,
                 binding_names::kAttentionSequenceLengths, binding_names::kStateIndices})
        {
            ok &= setOptimizationProfile(&profile, name, createDims({1}), createDims({mBuilderConfig.maxBatchSize}),
                createDims({mBuilderConfig.maxBatchSize}));
        }
        ok &= setOptimizationProfile(&profile, binding_names::kQueryStartOffsets, createDims({2}),
            createDims({mBuilderConfig.maxBatchSize + 1}), createDims({mBuilderConfig.maxBatchSize + 1}));
        ok &= setOptimizationProfile(
            &profile, binding_names::kContextSequenceCountCarrier, createDims({0}), createDims({0}), createDims({0}));
        ok &= setOptimizationProfile(
            &profile, binding_names::kExecutionPhaseMarker, createDims({1}), createDims({4}), createDims({8}));
        // kv_page_table: [batch, 2, maxPagesPerSeq] int32 for proposal self-attention and target-KV updates.
        int32_t const maxPagesPerSeq
            = rt::computeMaxPagesPerSeq(static_cast<int32_t>(mBuilderConfig.maxKVCacheCapacity));
        ok &= setOptimizationProfile(&profile, binding_names::kKVPageTable, createDims({1, 2, maxPagesPerSeq}),
            createDims({mBuilderConfig.maxBatchSize, 2, maxPagesPerSeq}),
            createDims({mBuilderConfig.maxBatchSize, 2, maxPagesPerSeq}));
        ok &= setOptimizationProfile(&profile, binding_names::kAttentionMask, createDims({minProposalRows, 1}),
            createDims({optProposalRows, optPackedMaskLen}), createDims({maxProposalRows, packedMaskLen}));
        // DFlash draft KV cache uses the paged-pool contract shared with the AttentionPlugin binding:
        // [2, numPages, kTOKENS_PER_PAGE,
        // numKVHeads, headDim] (single fixed numPages value, same as setupKVCacheProfiles).
        int64_t const numPages = mBuilderConfig.resolvedKVPoolPages();
        for (int32_t i = 0; i < mNbKVCacheInputs; ++i)
        {
            int64_t layerHeadSize = (!mPerLayerHeadSize.empty()) ? mPerLayerHeadSize[i] : mHeadSize;
            int64_t layerNumKVHeads = (!mPerLayerNumKVHeads.empty()) ? mPerLayerNumKVHeads[i] : mNumKVHeads;
            std::string pastName = std::string(binding_names::kPastKeyValuesTemplate) + "_" + std::to_string(i);
            std::string presentName = std::string(binding_names::kPresentKeyValuesTemplate) + "_" + std::to_string(i);
            nvinfer1::Dims const kvCacheShape
                = createDims({2, numPages, rt::kTOKENS_PER_PAGE, layerNumKVHeads, layerHeadSize});
            ok &= setOptimizationProfile(&profile, pastName.c_str(), kvCacheShape, kvCacheShape, kvCacheShape);
            ok &= setOptimizationProfile(&profile, presentName.c_str(), kvCacheShape, kvCacheShape, kvCacheShape);
        }
        return ok;
    };

    result &= setupOneProfile(contextProfile, optPrefillTargetHiddenLen, maxPrefillTargetHiddenLen);
    result &= setupOneProfile(generationProfile, optDecodeTargetHiddenLen, maxDecodeTargetHiddenLen);
    return result;
}

bool LLMBuilder::setupGemma4MTPDraftProfiles(nvinfer1::IOptimizationProfile& contextProfile,
    nvinfer1::IOptimizationProfile& generationProfile, nvinfer1::INetworkDefinition const& network)
{
    bool result = true;
    int64_t const baseHiddenSize = mTargetModelOutputHiddenDim;

    auto setupRopeProfile = [&](nvinfer1::IOptimizationProfile& profile, char const* bindingName, int64_t rotaryDim) {
        return setOptimizationProfile(&profile, bindingName, createDims({1, rotaryDim}),
            createDims({mBuilderConfig.maxBatchSize, rotaryDim}), createDims({mBuilderConfig.maxBatchSize, rotaryDim}));
    };

    auto setupOneProfile = [&](nvinfer1::IOptimizationProfile& profile) {
        bool ok = true;
        ok &= setOptimizationProfile(&profile, binding_names::kInputsEmbeds, createDims({1, baseHiddenSize}),
            createDims({mBuilderConfig.maxBatchSize, baseHiddenSize}),
            createDims({mBuilderConfig.maxBatchSize, baseHiddenSize}));
        ok &= setOptimizationProfile(&profile, binding_names::kBaseModelHiddenStates, createDims({1, baseHiddenSize}),
            createDims({mBuilderConfig.maxBatchSize, baseHiddenSize}),
            createDims({mBuilderConfig.maxBatchSize, baseHiddenSize}));

        for (char const* tokenInput : {binding_names::kPositions})
        {
            ok &= setOptimizationProfile(&profile, tokenInput, createDims({1}),
                createDims({mBuilderConfig.maxBatchSize}), createDims({mBuilderConfig.maxBatchSize}));
        }
        for (char const* sequenceInput : {binding_names::kQueryLengths, binding_names::kPastLengths,
                 binding_names::kAttentionSequenceLengths, binding_names::kStateIndices})
        {
            ok &= setOptimizationProfile(&profile, sequenceInput, createDims({1}),
                createDims({mBuilderConfig.maxBatchSize}), createDims({mBuilderConfig.maxBatchSize}));
        }
        ok &= setOptimizationProfile(&profile, binding_names::kQueryStartOffsets, createDims({2}),
            createDims({mBuilderConfig.maxBatchSize + 1}), createDims({mBuilderConfig.maxBatchSize + 1}));
        ok &= setOptimizationProfile(
            &profile, binding_names::kContextSequenceCountCarrier, createDims({0}), createDims({0}), createDims({0}));
        ok &= setOptimizationProfile(
            &profile, binding_names::kExecutionPhaseMarker, createDims({1}), createDims({4}), createDims({8}));

        // kv_page_table: [batch, 2, maxPagesPerSeq] int32 — the TARGET pool's page table
        // (the assistant reads the target's paged KV pool; the runtime binds the target's
        // identity table here).
        int32_t const maxPagesPerSeq
            = rt::computeMaxPagesPerSeq(static_cast<int32_t>(mBuilderConfig.maxKVCacheCapacity));
        ok &= setOptimizationProfile(&profile, binding_names::kKVPageTable, createDims({1, 2, maxPagesPerSeq}),
            createDims({mBuilderConfig.maxBatchSize, 2, maxPagesPerSeq}),
            createDims({mBuilderConfig.maxBatchSize, 2, maxPagesPerSeq}));

        if (auto const rotaryDim = getStaticInputDim(network, binding_names::kRopeCosSinSliding, 1))
        {
            ok &= setupRopeProfile(profile, binding_names::kRopeCosSinSliding, *rotaryDim);
        }
        if (auto const rotaryDim = getStaticInputDim(network, binding_names::kRopeCosSinFull, 1))
        {
            ok &= setupRopeProfile(profile, binding_names::kRopeCosSinFull, *rotaryDim);
        }
        if (auto const rotaryDim = getStaticInputDim(network, binding_names::kRopeCosSin, 1))
        {
            ok &= setupRopeProfile(profile, binding_names::kRopeCosSin, *rotaryDim);
        }

        // KV cache per-layer: the assistant binds the TARGET model's paged pool tensors
        // directly ([2, numPages, kTOKENS_PER_PAGE, numKVHeads, headDim]), so the profile
        // uses the same fixed page count as the target's setupKVCacheProfiles.
        for (int i = 0; i < mNbKVCacheInputs; ++i)
        {
            int64_t const layerHeadSize = (!mPerLayerHeadSize.empty()) ? mPerLayerHeadSize[i] : mHeadSize;
            int64_t const layerNumKVHeads = (!mPerLayerNumKVHeads.empty()) ? mPerLayerNumKVHeads[i] : mNumKVHeads;
            int32_t const layerCapacity
                = mPerLayerKVCacheCapacity.empty() ? 0 : mPerLayerKVCacheCapacity[static_cast<size_t>(i)];
            bool const reduced
                = rt::isReducedKvCacheCapacity(layerCapacity, static_cast<int32_t>(mBuilderConfig.maxKVCacheCapacity));
            int64_t const numPages = reduced ? mBuilderConfig.numSwaPages : mBuilderConfig.resolvedKVPoolPages();
            nvinfer1::Dims const kvCacheShape
                = createDims({2, numPages, rt::kTOKENS_PER_PAGE, layerNumKVHeads, layerHeadSize});
            ok &= setOptimizationProfile(
                &profile, binding_names::formatKVCacheName(i, true).c_str(), kvCacheShape, kvCacheShape, kvCacheShape);
        }

        return ok;
    };

    result &= setupOneProfile(contextProfile);
    result &= setupOneProfile(generationProfile);
    return result;
}

bool LLMBuilder::setupDSparkDraftProfiles(
    nvinfer1::IOptimizationProfile& contextProfile, nvinfer1::IOptimizationProfile& generationProfile)
{
    bool result = true;

    int64_t const maxProposalTokens = std::max<int64_t>(1, mBuilderConfig.maxDraftTreeSize);
    nlohmann::json const emptyDSparkConfig = nlohmann::json::object();
    nlohmann::json const& dsparkConfig
        = mModelConfig.contains("dspark_config") ? mModelConfig["dspark_config"] : emptyDSparkConfig;
    bool const sampleFromAnchor = dsparkConfig.value("sample_from_anchor", true);
    int64_t const maxDraftQueryTokens = maxProposalTokens + (sampleFromAnchor ? 0 : 1);
    int64_t const optDraftQueryTokens = maxDraftQueryTokens;
    int64_t const maxPrefillTargetHiddenLen = std::max<int64_t>(1, mBuilderConfig.maxInputLen);
    int64_t const optPrefillTargetHiddenLen = std::max<int64_t>(1, maxPrefillTargetHiddenLen / 2);
    // DSpark verifies [anchor] + proposal tokens and can accept the base bonus token,
    // so the next draft round may need one more target-hidden row than the proposal block.
    int64_t const maxDecodeTargetHiddenLen = maxProposalTokens + 1;
    int64_t const optDecodeTargetHiddenLen = maxProposalTokens + 1;

    int64_t const packedMaskLen = static_cast<int64_t>(divUp(maxDraftQueryTokens, 32));
    int64_t const optPackedMaskLen = static_cast<int64_t>(divUp(optDraftQueryTokens, 32));

    // Profile 0 handles round-0/system-prompt cache update, where target hidden spans the prompt.
    // Profile 1 handles steady-state block proposal, where target hidden delta is bounded by block size.
    auto setupOneProfile = [&](nvinfer1::IOptimizationProfile& profile, int64_t optTargetHiddenLen,
                               int64_t maxTargetHiddenLen) {
        bool ok = true;
        int64_t const optProposalRows = mBuilderConfig.checkedPhysicalTokens(optDraftQueryTokens);
        int64_t const maxProposalRows = mBuilderConfig.checkedPhysicalTokens(maxDraftQueryTokens);
        int64_t const optDeltaRows = mBuilderConfig.checkedPhysicalTokens(optTargetHiddenLen);
        int64_t const maxDeltaRows = mBuilderConfig.checkedPhysicalTokens(maxTargetHiddenLen);
        ok &= setOptimizationProfile(&profile, binding_names::kInputsEmbeds, createDims({1, mHiddenSize}),
            createDims({optProposalRows, mHiddenSize}), createDims({maxProposalRows, mHiddenSize}));
        ok &= setOptimizationProfile(&profile, binding_names::kDFlashTargetHiddenConcat,
            createDims({1, mTargetModelOutputHiddenDim}), createDims({optDeltaRows, mTargetModelOutputHiddenDim}),
            createDims({maxDeltaRows, mTargetModelOutputHiddenDim}));
        ok &= setOptimizationProfile(&profile, binding_names::kRopeCosSin, createDims({1, mRotaryDim}),
            createDims({optProposalRows, mRotaryDim}), createDims({maxProposalRows, mRotaryDim}));
        ok &= setOptimizationProfile(&profile, binding_names::kDFlashDeltaRopeCosSin, createDims({1, mRotaryDim}),
            createDims({optDeltaRows, mRotaryDim}), createDims({maxDeltaRows, mRotaryDim}));
        for (char const* name : {binding_names::kPositions, binding_names::kAttentionPosId})
        {
            ok &= setOptimizationProfile(
                &profile, name, createDims({1}), createDims({optProposalRows}), createDims({maxProposalRows}));
        }
        for (char const* name : {binding_names::kDFlashDeltaPositions, binding_names::kDFlashDeltaTokenToSequence})
        {
            ok &= setOptimizationProfile(
                &profile, name, createDims({1}), createDims({optDeltaRows}), createDims({maxDeltaRows}));
        }
        for (char const* name : {binding_names::kQueryLengths, binding_names::kPastLengths,
                 binding_names::kAttentionSequenceLengths, binding_names::kStateIndices})
        {
            ok &= setOptimizationProfile(&profile, name, createDims({1}), createDims({mBuilderConfig.maxBatchSize}),
                createDims({mBuilderConfig.maxBatchSize}));
        }
        ok &= setOptimizationProfile(&profile, binding_names::kQueryStartOffsets, createDims({2}),
            createDims({mBuilderConfig.maxBatchSize + 1}), createDims({mBuilderConfig.maxBatchSize + 1}));
        ok &= setOptimizationProfile(
            &profile, binding_names::kContextSequenceCountCarrier, createDims({0}), createDims({0}), createDims({0}));
        ok &= setOptimizationProfile(
            &profile, binding_names::kExecutionPhaseMarker, createDims({1}), createDims({4}), createDims({8}));
        // kv_page_table: [batch, 2, maxPagesPerSeq] int32. Proposal self-attention's page
        // table for proposal self-attention and target-KV updates.
        int32_t const maxPagesPerSeq
            = rt::computeMaxPagesPerSeq(static_cast<int32_t>(mBuilderConfig.maxKVCacheCapacity));
        ok &= setOptimizationProfile(&profile, binding_names::kKVPageTable, createDims({1, 2, maxPagesPerSeq}),
            createDims({mBuilderConfig.maxBatchSize, 2, maxPagesPerSeq}),
            createDims({mBuilderConfig.maxBatchSize, 2, maxPagesPerSeq}));
        ok &= setOptimizationProfile(&profile, binding_names::kAttentionMask, createDims({1, 1}),
            createDims({optProposalRows, optPackedMaskLen}), createDims({maxProposalRows, packedMaskLen}));
        // KV cache per-layer: DSpark's own combined draft cache uses the same paged-pool
        // contract as AttentionPlugin: [2, numPages, kTOKENS_PER_PAGE, numKVHeads, headDim].
        int64_t const numPages = mBuilderConfig.resolvedKVPoolPages();
        for (int32_t i = 0; i < mNbKVCacheInputs; ++i)
        {
            int64_t layerHeadSize = (!mPerLayerHeadSize.empty()) ? mPerLayerHeadSize[i] : mHeadSize;
            int64_t layerNumKVHeads = (!mPerLayerNumKVHeads.empty()) ? mPerLayerNumKVHeads[i] : mNumKVHeads;
            std::string pastName = std::string(binding_names::kPastKeyValuesTemplate) + "_" + std::to_string(i);
            std::string presentName = std::string(binding_names::kPresentKeyValuesTemplate) + "_" + std::to_string(i);
            nvinfer1::Dims const kvCacheShape
                = createDims({2, numPages, rt::kTOKENS_PER_PAGE, layerNumKVHeads, layerHeadSize});
            ok &= setOptimizationProfile(&profile, pastName.c_str(), kvCacheShape, kvCacheShape, kvCacheShape);
            ok &= setOptimizationProfile(&profile, presentName.c_str(), kvCacheShape, kvCacheShape, kvCacheShape);
        }
        return ok;
    };

    result &= setupOneProfile(contextProfile, optPrefillTargetHiddenLen, maxPrefillTargetHiddenLen);
    result &= setupOneProfile(generationProfile, optDecodeTargetHiddenLen, maxDecodeTargetHiddenLen);
    return result;
}

bool LLMBuilder::setupPleProfiles(nvinfer1::IOptimizationProfile& contextProfile,
    nvinfer1::IOptimizationProfile& generationProfile, nvinfer1::INetworkDefinition const& network)
{
    bool result = true;
    bool foundPleInput = false;
    std::string_view const prefix = binding_names::kPleTokenEmbedsTemplate;
    auto const [prefill, generation] = tokenAlignedProfileRanges();

    for (int32_t idx = 0; idx < network.getNbInputs(); ++idx)
    {
        auto const* input = network.getInput(idx);
        char const* const inputName = input->getName();
        std::string_view const inputNameView = inputName;
        if (inputNameView.rfind(prefix, 0) != 0 || inputNameView.size() <= prefix.size()
            || inputNameView[prefix.size()] != '_')
        {
            continue;
        }
        foundPleInput = true;

        auto const inputDims = input->getDimensions();
        if (inputDims.nbDims != 2 || inputDims.d[1] <= 0)
        {
            LOG_ERROR(
                "PLE input %s must be rank-2 [physical_tokens, hidden] with a static hidden dimension.", inputName);
            result = false;
            continue;
        }

        int64_t const pleHiddenSize = inputDims.d[1];
        result &= setOptimizationProfile(&contextProfile, inputName,
            createDims({prefill.min.physicalTokens, pleHiddenSize}),
            createDims({prefill.opt.physicalTokens, pleHiddenSize}),
            createDims({prefill.max.physicalTokens, pleHiddenSize}));
        result &= setOptimizationProfile(&generationProfile, inputName,
            createDims({generation.min.physicalTokens, pleHiddenSize}),
            createDims({generation.opt.physicalTokens, pleHiddenSize}),
            createDims({generation.max.physicalTokens, pleHiddenSize}));
    }

    if (foundPleInput)
    {
        LOG_INFO("Configured optimization profiles for Gemma4 PLE inputs.");
    }
    if (!result)
    {
        LOG_ERROR("Failed to setup optimization profiles at setupPleProfiles().");
    }
    return result;
}

bool LLMBuilder::setupDeepstackProfiles(nvinfer1::IOptimizationProfile& contextProfile,
    nvinfer1::IOptimizationProfile& generationProfile, nvinfer1::INetworkDefinition const& network)
{
    bool result = true;
    auto const [prefill, generation] = tokenAlignedProfileRanges();

    // Dynamically detect all deepstack_embeds inputs in the network
    std::vector<std::string> deepstackInputs;
    for (int32_t idx = 0; idx < network.getNbInputs(); idx++)
    {
        std::string_view const inputName = network.getInput(idx)->getName();
        if (inputName.find(binding_names::kDeepstackEmbedsTemplate) != std::string_view::npos)
        {
            deepstackInputs.emplace_back(inputName);
        }
    }

    // If no deepstack embeds found, return early (not a Qwen3VL model)
    if (deepstackInputs.empty())
    {
        return true;
    }

    LOG_INFO("Detected %zu deepstack embedding inputs", deepstackInputs.size());

    // Deepstack rows use the same physical token address space as inputs_embeds.
    for (auto const& deepstackInputName : deepstackInputs)
    {
        LOG_INFO("Setting up optimization profile for %s", deepstackInputName.c_str());

        if (getInputRank(network, deepstackInputName.c_str()) != 2)
        {
            LOG_ERROR("Deepstack input %s must be rank-2 [physical_tokens, hidden].", deepstackInputName.c_str());
            result = false;
            continue;
        }

        result &= setOptimizationProfile(&contextProfile, deepstackInputName.c_str(),
            createDims({prefill.min.physicalTokens, mHiddenSize}),
            createDims({prefill.opt.physicalTokens, mHiddenSize}),
            createDims({prefill.max.physicalTokens, mHiddenSize}));
        result &= setOptimizationProfile(&generationProfile, deepstackInputName.c_str(),
            createDims({generation.min.physicalTokens, mHiddenSize}),
            createDims({generation.opt.physicalTokens, mHiddenSize}),
            createDims({generation.max.physicalTokens, mHiddenSize}));
    }

    if (!result)
    {
        LOG_ERROR("Failed to setup optimization profiles at setupDeepstackProfiles().");
    }

    return result;
}

bool LLMBuilder::setupLmHeadWeightProfiles(nvinfer1::IOptimizationProfile& contextProfile,
    nvinfer1::IOptimizationProfile& generationProfile, nvinfer1::INetworkDefinition const& network)
{
    bool result = true;

    // Detect if lm_head_weight input exists (gemma4 assistant)
    bool hasLmHeadWeight = false;
    for (int32_t idx = 0; idx < network.getNbInputs(); idx++)
    {
        std::string_view const inputName = network.getInput(idx)->getName();
        if (inputName == binding_names::kLmHeadWeight)
        {
            hasLmHeadWeight = true;
            break;
        }
    }

    // If no lm_head_weight input found, return early (not a CodePredictor model)
    if (!hasLmHeadWeight)
    {
        return true;
    }

    LOG_INFO("Detected lm_head_weight input (CodePredictor model)");

    // lm_head_weight shape: [vocab_size, hidden_size]
    // For CodePredictor: vocab_size=2048 (codebook size), hidden_size=1024
    // This is a fixed-size weight tensor that gets bound at runtime
    int64_t const vocabSize = mModelConfig["vocab_size"].get<int64_t>();
    int64_t const hiddenSize = mHiddenSize;

    // Both context and generation profiles use the same shape since this is a weight tensor
    result &= setOptimizationProfile(&contextProfile, binding_names::kLmHeadWeight, createDims({vocabSize, hiddenSize}),
        createDims({vocabSize, hiddenSize}), createDims({vocabSize, hiddenSize}));
    result &= setOptimizationProfile(&generationProfile, binding_names::kLmHeadWeight,
        createDims({vocabSize, hiddenSize}), createDims({vocabSize, hiddenSize}), createDims({vocabSize, hiddenSize}));

    if (!result)
    {
        LOG_ERROR("Failed to setup optimization profiles for lm_head_weight.");
    }

    return result;
}

bool LLMBuilder::setupLoraProfiles(nvinfer1::IOptimizationProfile& contextProfile,
    nvinfer1::IOptimizationProfile& generationProfile, nvinfer1::INetworkDefinition const& network)
{
    bool result = true;
    if (mBuilderConfig.maxLoraRank == 0)
    {
        LOG_WARNING(
            "Your model has dynamic LoRA, but max LoRA rank is 0. This is equivalent to no LoRA. Please set "
            "--maxLoraRank to a positive value if you want to use LoRA.");
        return true;
    }

    bool findLoraWeights = false;

    for (int i = 0; i < network.getNbInputs(); ++i)
    {
        auto* input = network.getInput(i);
        char const* const inputName = input->getName();
        std::string_view const inputNameView = inputName;

        if (inputNameView.find(binding_names::kLoraAPrefix) != std::string_view::npos)
        {
            if (!findLoraWeights)
            {
                findLoraWeights = true;
            }
            // For lora_A, the shape is [gemm_k, lora_rank]
            auto dims = input->getDimensions();
            if (dims.nbDims == 2)
            {
                int64_t gemm_k = dims.d[0];
                result &= setOptimizationProfile(&contextProfile, inputName, createDims({gemm_k, 0}),    // min shape
                    createDims({gemm_k, mBuilderConfig.maxLoraRank / 2}),                                // opt shape
                    createDims({gemm_k, mBuilderConfig.maxLoraRank}));                                   // max shape
                result &= setOptimizationProfile(&generationProfile, inputName, createDims({gemm_k, 0}), // min shape
                    createDims({gemm_k, mBuilderConfig.maxLoraRank / 2}),                                // opt shape
                    createDims({gemm_k, mBuilderConfig.maxLoraRank}));                                   // max shape
            }
        }
        else if (inputNameView.find(binding_names::kLoraBPrefix) != std::string_view::npos)
        {
            if (!findLoraWeights)
            {
                findLoraWeights = true;
            }
            // For lora_B, the shape is [lora_rank, gemm_n]
            auto dims = input->getDimensions();
            if (dims.nbDims == 2)
            {
                int64_t gemm_n = dims.d[1];
                result &= setOptimizationProfile(&contextProfile, inputName, createDims({0, gemm_n}),    // min shape
                    createDims({mBuilderConfig.maxLoraRank / 2, gemm_n}),                                // opt shape
                    createDims({mBuilderConfig.maxLoraRank, gemm_n}));                                   // max shape
                result &= setOptimizationProfile(&generationProfile, inputName, createDims({0, gemm_n}), // min shape
                    createDims({mBuilderConfig.maxLoraRank / 2, gemm_n}),                                // opt shape
                    createDims({mBuilderConfig.maxLoraRank, gemm_n}));                                   // max shape
            }
        }
    }

    if (!findLoraWeights)
    {
        LOG_ERROR(
            "Failed to find any LoRA weights inputs in the ONNX model. Have you inserted LoRA weights using "
            "tensorrt-edgellm-insert-lora command?");
        return false;
    }

    if (!result)
    {
        LOG_ERROR("Failed to setup optimization profiles at setupLoraProfiles().");
    }

    return result;
}

bool LLMBuilder::setupKVCacheProfiles(
    nvinfer1::IOptimizationProfile& contextProfile, nvinfer1::IOptimizationProfile& generationProfile)
{
    bool result = true;
    // Plugin path: paged pool binding [2, numPages_i, kTOKENS_PER_PAGE, num_kv_heads, head_dim].
    // Full-only layers have a fixed page count. For SWA-capable layers, MIN/OPT use the smaller of
    // bounded and full storage while MAX covers the larger count. This lets short-sequence engines
    // optimize for full storage when bounded transition reservations would consume more memory.
    // "Empty vs non-empty" cache is conveyed by kvcache_start_index's own profile, not by this
    // tensor's shape (the plugin reads numPages from dims.d[1]).
    for (int i = 0; i < mNbKVCacheInputs; ++i)
    {
        // Per-layer dims mirror the runtime registry (kv_layer_configs): head size varies on
        // Gemma4 today; the KV head count is per-layer for the same forward-compat reason.
        int64_t layerHeadSize = (!mPerLayerHeadSize.empty()) ? mPerLayerHeadSize[i] : mHeadSize;
        int64_t layerNumKVHeads = (!mPerLayerNumKVHeads.empty()) ? mPerLayerNumKVHeads[i] : mNumKVHeads;
        int32_t const layerCapacity
            = mPerLayerKVCacheCapacity.empty() ? 0 : mPerLayerKVCacheCapacity[static_cast<size_t>(i)];
        std::array<int64_t, 3> const pageProfile = mBuilderConfig.resolveKVPoolPageProfile(layerCapacity);
        nvinfer1::Dims const minKVCacheShape
            = createDims({2, pageProfile[0], rt::kTOKENS_PER_PAGE, layerNumKVHeads, layerHeadSize});
        nvinfer1::Dims const optKVCacheShape
            = createDims({2, pageProfile[1], rt::kTOKENS_PER_PAGE, layerNumKVHeads, layerHeadSize});
        nvinfer1::Dims const maxKVCacheShape
            = createDims({2, pageProfile[2], rt::kTOKENS_PER_PAGE, layerNumKVHeads, layerHeadSize});

        result &= setOptimizationProfile(&contextProfile, binding_names::formatKVCacheName(i, true).c_str(),
            minKVCacheShape, optKVCacheShape, maxKVCacheShape);
        result &= setOptimizationProfile(&generationProfile, binding_names::formatKVCacheName(i, true).c_str(),
            minKVCacheShape, optKVCacheShape, maxKVCacheShape);
    }

    return result;
}

bool LLMBuilder::setupRecurrentStateProfiles(
    nvinfer1::IOptimizationProfile* const contextProfile, nvinfer1::IOptimizationProfile* const generationProfile)
{
    if (mNumLinearAttnLayers == 0)
    {
        return true;
    }

    bool result = true;

    // Recurrent state shape: [batch, recurrentNumHeads, recurrentHeadDim, recurrentStateSize]
    int64_t const minPoolRows = mBuilderConfig.maxBatchSize;
    nvinfer1::Dims minRecurrentShape
        = createDims({minPoolRows, mRecurrentStateNumHeads, mRecurrentStateHeadDim, mRecurrentStateSize});
    nvinfer1::Dims optRecurrentShape = createDims(
        {mBuilderConfig.maxBatchSize, mRecurrentStateNumHeads, mRecurrentStateHeadDim, mRecurrentStateSize});
    nvinfer1::Dims maxRecurrentShape = createDims(
        {mBuilderConfig.maxBatchSize, mRecurrentStateNumHeads, mRecurrentStateHeadDim, mRecurrentStateSize});

    for (int32_t i = 0; i < mNumLinearAttnLayers; ++i)
    {
        std::string const recurrentStateName = binding_names::formatRecurrentStateName(i, /*isPast=*/true);
        result &= setOptimizationProfile(
            contextProfile, recurrentStateName.c_str(), minRecurrentShape, optRecurrentShape, maxRecurrentShape);
        result &= setOptimizationProfile(
            generationProfile, recurrentStateName.c_str(), minRecurrentShape, optRecurrentShape, maxRecurrentShape);
    }

    LOG_DEBUG("Set up recurrent state optimization profiles for %d recurrent layers", mNumLinearAttnLayers);
    return result;
}

bool LLMBuilder::setupConvStateProfiles(
    nvinfer1::IOptimizationProfile* const contextProfile, nvinfer1::IOptimizationProfile* const generationProfile)
{
    if (mNumLinearAttnLayers == 0 || mConvDim == 0 || mConvKernel == 0)
    {
        return true;
    }

    bool result = true;

    // Conv state shape: [batch, conv_dim, conv_kernel]
    int64_t const minPoolRows = mBuilderConfig.maxBatchSize;
    nvinfer1::Dims minConvShape = createDims({minPoolRows, mConvDim, mConvKernel});
    nvinfer1::Dims optConvShape = createDims({mBuilderConfig.maxBatchSize, mConvDim, mConvKernel});
    nvinfer1::Dims maxConvShape = createDims({mBuilderConfig.maxBatchSize, mConvDim, mConvKernel});

    for (int32_t i = 0; i < mNumLinearAttnLayers; ++i)
    {
        std::string const convStateName = binding_names::formatConvStateName(i, /*isPast=*/true);
        result
            &= setOptimizationProfile(contextProfile, convStateName.c_str(), minConvShape, optConvShape, maxConvShape);
        result &= setOptimizationProfile(
            generationProfile, convStateName.c_str(), minConvShape, optConvShape, maxConvShape);
    }

    LOG_DEBUG("Set up conv state optimization profiles for %d recurrent layers", mNumLinearAttnLayers);
    return result;
}

bool LLMBuilder::setupIntermediateRecurrentStateProfiles(
    nvinfer1::IOptimizationProfile& contextProfile, nvinfer1::IOptimizationProfile& generationProfile)
{
    if (mNumLinearAttnLayers == 0)
    {
        return true;
    }

    bool result = true;

    int64_t const contextOptTokens
        = mBuilderConfig.checkedPhysicalTokens(std::max<int64_t>(1, mBuilderConfig.maxInputLen / 2));
    int64_t const contextMaxTokens = mBuilderConfig.checkedPhysicalTokens(mBuilderConfig.maxInputLen);
    int64_t const generationTokens = mBuilderConfig.checkedPhysicalTokens(mBuilderConfig.resolvedRoleQueryLength());
    nvinfer1::Dims minCtxShape = createDims({1, mRecurrentStateNumHeads, mRecurrentStateHeadDim, mRecurrentStateSize});
    nvinfer1::Dims optCtxShape
        = createDims({contextOptTokens, mRecurrentStateNumHeads, mRecurrentStateHeadDim, mRecurrentStateSize});
    nvinfer1::Dims maxCtxShape
        = createDims({contextMaxTokens, mRecurrentStateNumHeads, mRecurrentStateHeadDim, mRecurrentStateSize});
    nvinfer1::Dims minGenShape = minCtxShape;
    nvinfer1::Dims optGenShape
        = createDims({generationTokens, mRecurrentStateNumHeads, mRecurrentStateHeadDim, mRecurrentStateSize});
    nvinfer1::Dims maxGenShape = optGenShape;

    for (int32_t i = 0; i < mNumLinearAttnLayers; ++i)
    {
        std::string const name = binding_names::formatIntermediateRecurrentStateName(i);
        result &= setOptimizationProfile(&contextProfile, name.c_str(), minCtxShape, optCtxShape, maxCtxShape);
        result &= setOptimizationProfile(&generationProfile, name.c_str(), minGenShape, optGenShape, maxGenShape);
    }

    LOG_DEBUG("Set up intermediate recurrent state profiles for %d recurrent layers (MTP)", mNumLinearAttnLayers);
    return result;
}

bool LLMBuilder::setupIntermediateConvStateProfiles(
    nvinfer1::IOptimizationProfile& contextProfile, nvinfer1::IOptimizationProfile& generationProfile)
{
    if (mNumLinearAttnLayers == 0 || mConvDim == 0 || mConvKernel == 0)
    {
        return true;
    }

    bool result = true;

    int64_t const contextOptTokens
        = mBuilderConfig.checkedPhysicalTokens(std::max<int64_t>(1, mBuilderConfig.maxInputLen / 2));
    int64_t const contextMaxTokens = mBuilderConfig.checkedPhysicalTokens(mBuilderConfig.maxInputLen);
    int64_t const generationTokens = mBuilderConfig.checkedPhysicalTokens(mBuilderConfig.resolvedRoleQueryLength());
    nvinfer1::Dims minCtxShape = createDims({1, mConvDim, mConvKernel});
    nvinfer1::Dims optCtxShape = createDims({contextOptTokens, mConvDim, mConvKernel});
    nvinfer1::Dims maxCtxShape = createDims({contextMaxTokens, mConvDim, mConvKernel});
    nvinfer1::Dims minGenShape = minCtxShape;
    nvinfer1::Dims optGenShape = createDims({generationTokens, mConvDim, mConvKernel});
    nvinfer1::Dims maxGenShape = optGenShape;

    for (int32_t i = 0; i < mNumLinearAttnLayers; ++i)
    {
        std::string const name = binding_names::formatIntermediateConvStateName(i);
        result &= setOptimizationProfile(&contextProfile, name.c_str(), minCtxShape, optCtxShape, maxCtxShape);
        result &= setOptimizationProfile(&generationProfile, name.c_str(), minGenShape, optGenShape, maxGenShape);
    }

    LOG_DEBUG(
        "Set up intermediate conv state profiles for %d recurrent layers (MTP/DFlash/JetSpec)", mNumLinearAttnLayers);
    return result;
}

bool LLMBuilder::setupLinearAttentionSpecVerifyProfiles(nvinfer1::IOptimizationProfile& contextProfile,
    nvinfer1::IOptimizationProfile& generationProfile, nvinfer1::INetworkDefinition const& network)
{
    static_cast<void>(contextProfile);
    static_cast<void>(generationProfile);
    if (mNumLinearAttnLayers == 0)
    {
        return true;
    }

    if (!hasInputBinding(network, binding_names::kExecutionPhaseMarker))
    {
        LOG_ERROR("Hybrid MTP/DFlash/JetSpec base engine is missing input '%s'. Re-export the ONNX model.",
            binding_names::kExecutionPhaseMarker);
        return false;
    }

    LOG_DEBUG("Set up hybrid linear-attention spec-verify profiles for %d recurrent layers", mNumLinearAttnLayers);
    return true;
}

namespace
{
// Speculative base and draft engines share one engineDir, so their external weight
// files (e.g. external_int4_ffn_weights.safetensors) would otherwise overwrite
// each other. Prefix only the draft files; the base keeps the original names
// (matching the standalone non-speculative case), which is enough to avoid the
// collision. copyConfig() and copyExternalWeightFiles() must agree on this name.
std::string externalWeightDstName(std::string const& filename, bool specDraft)
{
    if (specDraft)
    {
        return "draft_" + filename;
    }
    return filename;
}
} // namespace

bool LLMBuilder::copyConfig()
{
    std::string const configFileName = getOutputConfigFileName(mBuilderConfig);

    std::string const targetConfigPath = (mEngineDir / configFileName).string();

    Json configWithBuilder = mSharedModelConfig;
    LLMBuilderConfig outputConfig = mBuilderConfig;
    configWithBuilder["builder_config"] = outputConfig.toJson();

    if (configWithBuilder.contains("rank_configs"))
    {
        if (!configWithBuilder["rank_configs"].is_array())
        {
            LOG_ERROR("rank_configs must be an array when present in config.json");
            return false;
        }
        int32_t const artifactWorldSize
            = std::max<int32_t>(1, static_cast<int32_t>(configWithBuilder["rank_configs"].size()));
        for (auto& rankConfig : configWithBuilder["rank_configs"])
        {
            if (!rankConfig.is_object() || !rankConfig.contains("rank"))
            {
                LOG_ERROR("Each rank_configs entry must be an object with a rank field.");
                return false;
            }
            int32_t const rank = rankConfig["rank"].get<int32_t>();
            parallel_artifacts::RankArtifactContext const context{artifactWorldSize, rank};
            rankConfig["engine"] = parallel_artifacts::engineFileName(context);
        }
    }

    // Keep external weight file references in sync with the names written by
    // copyExternalWeightFiles() (draft files get the "draft_" prefix) so the runtime loads the right file.
    if (configWithBuilder.contains("external_weight_files") && configWithBuilder["external_weight_files"].is_array())
    {
        for (auto& fileEntry : configWithBuilder["external_weight_files"])
        {
            if (fileEntry.is_object() && fileEntry.contains("file") && fileEntry["file"].is_string())
            {
                fileEntry["file"]
                    = externalWeightDstName(fileEntry["file"].get<std::string>(), mBuilderConfig.specDraft);
            }
        }
    }

    // Add detected num_deepstack_features if present (Qwen3VL models)
    configWithBuilder["num_deepstack_features"] = mNumDeepstackFeatures;

    // Emit per-layer KV cache configs for heterogeneous models (e.g. Gemma4).
    // The runtime uses `kv_layer_configs` + normalized `layer_types` ("attention"/"mamba")
    // to allocate per-layer KV cache tensors with the correct head dimensions.
    if (!mPerLayerHeadSize.empty())
    {
        Json kvLayerConfigs = Json::array();
        Json normalizedLayerTypes = Json::array();

        // Preserve the per-position ordering from the input `layer_types` when it is
        // present and complete: attention and mamba layers can interleave (e.g. Qwen3.5
        // GDN hybrids place attention every Nth layer), and emitting "all attention then
        // all mamba" would misplace every layer in the runtime routing table. Attention
        // entries consume the per-layer head sizes in order; mamba entries carry no KV.
        bool const haveInputLayerTypes = mModelConfig.contains("layer_types") && mModelConfig["layer_types"].is_array()
            && mModelConfig["layer_types"].size() == static_cast<size_t>(mNbKVCacheInputs + mNumLinearAttnLayers);
        if (haveInputLayerTypes)
        {
            int attnIdx = 0;
            for (auto const& layerType : mModelConfig["layer_types"])
            {
                // Input layer_types are normalized to "attention" / "mamba" by the export.
                if (layerType.is_string() && layerType.get<std::string>() == "mamba")
                {
                    normalizedLayerTypes.push_back("mamba");
                    kvLayerConfigs.push_back(nullptr);
                }
                else
                {
                    // The total-size guard above does not constrain the attention/mamba split,
                    // so a layer_types inconsistent with the ONNX (more attention entries than
                    // KV cache inputs) would index past mPerLayerHeadSize. Fail clearly instead.
                    check::check(attnIdx < static_cast<int>(mPerLayerHeadSize.size()),
                        "copyConfig: layer_types has more attention layers than the " + std::to_string(mNbKVCacheInputs)
                            + " KV cache input(s); config.json layer_types is inconsistent with the engine.");
                    normalizedLayerTypes.push_back("attention");
                    int64_t const layerNumKVHeads
                        = (!mPerLayerNumKVHeads.empty()) ? mPerLayerNumKVHeads[attnIdx] : mNumKVHeads;
                    Json layerConfig
                        = Json{{"num_kv_heads", layerNumKVHeads}, {"head_dim", mPerLayerHeadSize[attnIdx]}};
                    if (!mPerLayerKVCacheCapacity.empty() && mPerLayerKVCacheCapacity[attnIdx] > 0)
                    {
                        layerConfig["kv_cache_capacity"] = mPerLayerKVCacheCapacity[attnIdx];
                    }
                    kvLayerConfigs.push_back(std::move(layerConfig));
                    ++attnIdx;
                }
            }
        }
        else
        {
            // Legacy fallback (no per-position info): attention layers first, then mamba.
            // Correct for pure-attention and all-attention heterogeneous models (e.g. Gemma4).
            for (int i = 0; i < mNbKVCacheInputs; ++i)
            {
                normalizedLayerTypes.push_back("attention");
                int64_t layerNumKVHeads = (!mPerLayerNumKVHeads.empty()) ? mPerLayerNumKVHeads[i] : mNumKVHeads;
                Json layerConfig = Json{{"num_kv_heads", layerNumKVHeads}, {"head_dim", mPerLayerHeadSize[i]}};
                if (!mPerLayerKVCacheCapacity.empty() && mPerLayerKVCacheCapacity[i] > 0)
                {
                    layerConfig["kv_cache_capacity"] = mPerLayerKVCacheCapacity[i];
                }
                kvLayerConfigs.push_back(std::move(layerConfig));
            }
            for (int i = 0; i < mNumLinearAttnLayers; ++i)
            {
                normalizedLayerTypes.push_back("mamba");
                kvLayerConfigs.push_back(nullptr);
            }
        }
        configWithBuilder["layer_types"] = normalizedLayerTypes;
        configWithBuilder["kv_layer_configs"] = kvLayerConfigs;
    }

    // Write updated config
    std::ofstream targetConfigFile(targetConfigPath);
    if (!targetConfigFile.is_open())
    {
        LOG_ERROR("Failed to open target config file: %s", targetConfigPath.c_str());
        return false;
    }
    targetConfigFile << configWithBuilder.dump(2);
    targetConfigFile.close();

    LOG_INFO("Copied config.json with builder config to %s", targetConfigPath.c_str());
    return true;
}

bool LLMBuilder::copyTokenizerFiles()
{
    // Speculative draft models use the base model tokenizer.
    if (mBuilderConfig.specDraft)
    {
        return true;
    }

    // Models that use embeddings as input (e.g., Talker, CodePredictor) don't need tokenizer
    bool useEmbeddingsInput = mModelConfig.value("use_embeddings_input", false);
    if (useEmbeddingsInput)
    {
        LOG_INFO("Skipping tokenizer files (model uses embeddings input)");
        return true;
    }

    std::vector<std::string> const tokenizerFiles = {"tokenizer_config.json", "tokenizer.json"};
    bool allSuccess = true;

    for (auto const& filename : tokenizerFiles)
    {
        std::string const srcPath = (mOnnxDir / filename).string();
        std::string const dstPath = (mEngineDir / filename).string();

        if (file_io::copyFile(srcPath, dstPath))
        {
            LOG_INFO("Copied tokenizer file: %s", filename.c_str());
        }
        else
        {
            LOG_WARNING("Failed to copy tokenizer file %s", filename.c_str());
            allSuccess = false;
        }
    }

    auto const providerTemplate = mOnnxDir / "chat_template.jinja";
    auto const manualTemplate = mOnnxDir / "chat_template.model";
    auto const processorTemplate = mOnnxDir / "chat_template.processor";
    auto const namedTemplateDir = mOnnxDir / "additional_chat_templates";
    auto removeTemplateArtifact = [&allSuccess](std::filesystem::path const& path) {
        std::error_code error;
        std::filesystem::remove_all(path, error);
        if (error)
        {
            LOG_ERROR("Failed to remove stale chat-template artifact %s: %s", path.c_str(), error.message().c_str());
            allSuccess = false;
        }
    };

    // The only named-template checkpoints in the available test inventory are
    // CohereForAI/aya-23-8B and CohereForAI/aya-23-35B, whose architecture is
    // unsupported. Supported models provide one template containing any tool
    // rendering behavior.
    if (std::filesystem::exists(namedTemplateDir))
    {
        LOG_ERROR("Named provider chat templates are unsupported: %s", namedTemplateDir.c_str());
        return false;
    }

    bool const hasProviderTemplate = std::filesystem::is_regular_file(providerTemplate);
    bool const hasManualTemplate = std::filesystem::is_regular_file(manualTemplate);
    bool const hasProcessorTemplate = std::filesystem::is_regular_file(processorTemplate);
    if (std::filesystem::exists(processorTemplate) && !hasProcessorTemplate)
    {
        LOG_ERROR("Expected %s to be a regular file", processorTemplate.c_str());
        return false;
    }
    if (hasProviderTemplate == hasManualTemplate)
    {
        LOG_ERROR("Expected either provider Jinja or chat_template.model in %s, but not both", mOnnxDir.c_str());
        return false;
    }
    if (hasProcessorTemplate && !hasProviderTemplate)
    {
        LOG_ERROR("%s requires a provider Jinja template", processorTemplate.c_str());
        return false;
    }

    removeTemplateArtifact(mEngineDir / namedTemplateDir.filename());
    if (hasManualTemplate)
    {
        // Native markers are emitted only for Alpamayo-R1 and the Qwen3
        // ASR/TTS families, whose prompt contracts need runtime model data.
        removeTemplateArtifact(mEngineDir / providerTemplate.filename());
        removeTemplateArtifact(mEngineDir / processorTemplate.filename());
        if (!file_io::copyFile(manualTemplate.string(), (mEngineDir / manualTemplate.filename()).string()))
        {
            LOG_ERROR("Failed to copy %s", manualTemplate.c_str());
            allSuccess = false;
        }
    }
    else
    {
        removeTemplateArtifact(mEngineDir / manualTemplate.filename());
        if (!file_io::copyFile(providerTemplate.string(), (mEngineDir / providerTemplate.filename()).string()))
        {
            LOG_ERROR("Failed to copy %s", providerTemplate.c_str());
            allSuccess = false;
        }

        // microsoft/Phi-4-multimodal-instruct and OpenGVLab/InternVL use an
        // explicit raw-message processor before their provider Jinja.
        if (hasProcessorTemplate)
        {
            if (!file_io::copyFile(processorTemplate.string(), (mEngineDir / processorTemplate.filename()).string()))
            {
                LOG_ERROR("Failed to copy %s", processorTemplate.c_str());
                allSuccess = false;
            }
        }
        else
        {
            removeTemplateArtifact(mEngineDir / processorTemplate.filename());
        }
    }

    // Optional (Qwen3-Next Omni TTS): friendly speaker aliases consumed by the runtime.
    if (std::filesystem::exists(mOnnxDir / "voice_map.json"))
    {
        if (file_io::copyFile((mOnnxDir / "voice_map.json").string(), (mEngineDir / "voice_map.json").string()))
        {
            LOG_INFO("Copied voice_map.json");
        }
    }

    return allSuccess;
}

bool LLMBuilder::copyEagleFiles()
{
    // Copy d2t.safetensors for Eagle3 draft models only. MTP/DFlash/JetSpec drafts share vocab with base and have no
    // d2t.
    if (isSpecDecodeDraft(mModelConfig, "eagle3"))
    {
        std::string const d2tPath = (mOnnxDir / "d2t.safetensors").string();
        std::string const targetD2tPath = (mEngineDir / "d2t.safetensors").string();

        if (file_io::copyFile(d2tPath, targetD2tPath))
        {
            LOG_INFO("Copied d2t.safetensors to %s", targetD2tPath.c_str());
        }
        else
        {
            LOG_WARNING("Failed to copy d2t.safetensors to %s", targetD2tPath.c_str());
            return false;
        }
    }

    return true;
}

bool LLMBuilder::copyDSparkFiles()
{
    if (!isSpecDecodeDraft(mModelConfig, "dspark"))
    {
        return true;
    }

    bool allSuccess = true;
    char const* const requiredFiles[] = {
        binding_names::kDSparkHeadsFileName,
        binding_names::kDSparkHeadsInfoFileName,
    };
    for (auto const* filename : requiredFiles)
    {
        std::string const srcPath = (mOnnxDir / filename).string();
        std::string const dstPath = (mEngineDir / filename).string();
        if (file_io::copyFile(srcPath, dstPath))
        {
            LOG_INFO("Copied DSpark sidecar %s to %s", filename, dstPath.c_str());
        }
        else
        {
            LOG_ERROR("Failed to copy DSpark sidecar %s from %s to %s", filename, srcPath.c_str(), dstPath.c_str());
            allSuccess = false;
        }
    }
    return allSuccess;
}

bool LLMBuilder::copyDFlash2Files()
{
    if (!isDFlashV2DraftConfig(mModelConfig))
    {
        return true;
    }

    bool allSuccess = true;
    Json const& dflashConfig = mModelConfig.at("dflash_config");
    std::string const requiredFiles[] = {
        dflashConfig.value("selector_file", std::string(binding_names::kDFlash2SelectorFileName)),
    };
    for (auto const& filename : requiredFiles)
    {
        std::string const srcPath = (mOnnxDir / filename).string();
        std::string const dstPath = (mEngineDir / filename).string();
        if (file_io::copyFile(srcPath, dstPath))
        {
            LOG_INFO("Copied DFlash2 sidecar %s to %s", filename.c_str(), dstPath.c_str());
        }
        else
        {
            LOG_ERROR(
                "Failed to copy DFlash2 sidecar %s from %s to %s", filename.c_str(), srcPath.c_str(), dstPath.c_str());
            allSuccess = false;
        }
    }
    return allSuccess;
}

bool LLMBuilder::copyVocabMappingFiles()
{
    // Copy the vocab map sidecar if reduced vocabulary is used. Base engines
    // consume vocab_map.safetensors; speculative draft engines consume
    // draft_vocab_map.safetensors. Pick the right filename based on the build
    // role so the runtime finds the sidecar in mEngineDir.
    if (mModelConfig.contains(binding_names::kReducedVocabSizeKey)
        && mModelConfig[binding_names::kReducedVocabSizeKey].get<int32_t>() > 0)
    {
        char const* const vocabMapFileName
            = mBuilderConfig.specDraft ? binding_names::kDraftVocabMapFileName : binding_names::kVocabMapFileName;
        std::string const vocabMapPath = (mOnnxDir / vocabMapFileName).string();
        std::string const targetVocabMapPath = (mEngineDir / vocabMapFileName).string();

        if (!file_io::copyFile(vocabMapPath, targetVocabMapPath))
        {
            // The enclosing guard already proved reduced_vocab_size > 0, so the
            // sidecar is required to translate reduced output token IDs. Fail the
            // build instead of letting a broken engine ship.
            LOG_ERROR(
                "%s not found in %s but reduced_vocab_size > 0; the sidecar is "
                "required for the runtime to remap reduced->full vocabulary IDs.",
                vocabMapFileName, mOnnxDir.string().c_str());
            return false;
        }
        LOG_INFO("Copied %s to %s", vocabMapFileName, targetVocabMapPath.c_str());
    }

    return true;
}

bool LLMBuilder::copyEmbeddingFile()
{
    // Speculative draft models use the base model embedding table.
    if (mBuilderConfig.specDraft)
    {
        return true;
    }

    // Check if this is a Talker model (has text_projection.safetensors)
    std::filesystem::path const textProjectionPath = mOnnxDir / "text_projection.safetensors";
    if (std::filesystem::exists(textProjectionPath))
    {
        // Talker: copy embedding + text_projection + hidden_projection (optional, text-only TTS omits it)
        LOG_INFO("Detected Talker model, copying projection files...");

        std::vector<std::string> requiredFiles = {"embedding.safetensors", "text_projection.safetensors"};
        std::vector<std::string> optionalFiles = {"text_embedding.safetensors", "hidden_projection.safetensors"};

        bool allSuccess = true;
        for (auto const& filename : requiredFiles)
        {
            std::string const srcPath = (mOnnxDir / filename).string();
            std::string const dstPath = (mEngineDir / filename).string();

            if (file_io::copyFile(srcPath, dstPath))
            {
                LOG_INFO("Copied %s", filename.c_str());
            }
            else
            {
                LOG_ERROR("Failed to copy %s", filename.c_str());
                allSuccess = false;
            }
        }
        for (auto const& filename : optionalFiles)
        {
            std::string const srcPath = (mOnnxDir / filename).string();
            std::string const dstPath = (mEngineDir / filename).string();

            if (file_io::copyFile(srcPath, dstPath))
            {
                LOG_INFO("Copied %s", filename.c_str());
            }
            else
            {
                LOG_INFO("Optional %s not found, skipping", filename.c_str());
            }
        }

        return allSuccess;
    }

    // Check if this is a CodePredictor model (has codec_embeddings.safetensors)
    std::filesystem::path const codecEmbedPath = mOnnxDir / "codec_embeddings.safetensors";
    if (std::filesystem::exists(codecEmbedPath))
    {
        LOG_INFO("Detected CodePredictor model, copying codec files...");
        std::vector<std::string> cpRequiredFiles = {"codec_embeddings.safetensors", "lm_heads.safetensors"};
        std::vector<std::string> cpOptionalFiles = {"small_to_mtp_projection.safetensors"};
        bool allSuccess = true;
        for (auto const& filename : cpRequiredFiles)
        {
            std::string const srcPath = (mOnnxDir / filename).string();
            std::string const dstPath = (mEngineDir / filename).string();
            if (file_io::copyFile(srcPath, dstPath))
            {
                LOG_INFO("Copied %s", filename.c_str());
            }
            else
            {
                LOG_ERROR("Failed to copy required CodePredictor file: %s", filename.c_str());
                allSuccess = false;
            }
        }
        for (auto const& filename : cpOptionalFiles)
        {
            std::string const srcPath = (mOnnxDir / filename).string();
            std::string const dstPath = (mEngineDir / filename).string();
            if (file_io::copyFile(srcPath, dstPath))
            {
                LOG_INFO("Copied %s", filename.c_str());
            }
            else
            {
                LOG_INFO("Optional %s not found, skipping", filename.c_str());
            }
        }
        return allSuccess;
    }

    // Copy embedding.safetensors for vanilla LLM models
    std::string const embeddingPath = (mOnnxDir / "embedding.safetensors").string();
    std::string const targetEmbeddingPath = (mEngineDir / "embedding.safetensors").string();

    if (file_io::copyFile(embeddingPath, targetEmbeddingPath))
    {
        LOG_INFO("Copied embedding.safetensors to %s", targetEmbeddingPath.c_str());
    }
    else
    {
        LOG_ERROR(
            "Failed to copy embedding.safetensors from %s to %s", embeddingPath.c_str(), targetEmbeddingPath.c_str());
        return false;
    }

    if (mModelConfig.value("ple_enabled", false))
    {
        std::string const plePath = (mOnnxDir / binding_names::kPleEmbeddingFileName).string();
        std::string const targetPlePath = (mEngineDir / binding_names::kPleEmbeddingFileName).string();
        if (file_io::copyFile(plePath, targetPlePath))
        {
            LOG_INFO("Copied %s to %s", binding_names::kPleEmbeddingFileName, targetPlePath.c_str());
        }
        else
        {
            LOG_ERROR("Failed to copy %s from %s to %s", binding_names::kPleEmbeddingFileName, plePath.c_str(),
                targetPlePath.c_str());
            return false;
        }
    }

    return true;
}

bool LLMBuilder::copyExternalWeightFiles()
{
    Json const externalWeightFiles = mModelConfig.value("external_weight_files", Json::array());
    if (!externalWeightFiles.is_array())
    {
        LOG_ERROR("external_weight_files must be an array when present in config.json");
        return false;
    }
    if (externalWeightFiles.empty())
    {
        return true;
    }

    bool allSuccess = true;
    for (auto const& fileEntry : externalWeightFiles)
    {
        if (!fileEntry.is_object() || !fileEntry.contains("file") || !fileEntry["file"].is_string())
        {
            LOG_ERROR("Malformed external weight file entry: %s", fileEntry.dump().c_str());
            return false;
        }
        std::string const filename = fileEntry["file"].get<std::string>();
        std::string const dstFilename = externalWeightDstName(filename, mBuilderConfig.specDraft);
        std::filesystem::path const srcPath = mOnnxDir / filename;
        std::filesystem::path const dstPath = mEngineDir / dstFilename;

        if (file_io::copyFile(srcPath.string(), dstPath.string()))
        {
            LOG_INFO("Copied external weight file: %s -> %s", filename.c_str(), dstFilename.c_str());
        }
        else
        {
            LOG_ERROR("Failed to copy external weight file %s from %s to %s", filename.c_str(),
                srcPath.string().c_str(), dstPath.string().c_str());
            allSuccess = false;
        }
    }
    return allSuccess;
}

} // namespace builder
} // namespace trt_edgellm
