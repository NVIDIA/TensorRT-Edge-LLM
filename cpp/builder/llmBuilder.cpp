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
#include "common/ropeUtils.h"
#include "common/trtUtils.h"
#include "common/version.h"
#include "kernels/gdnKernels/gdnTreeChunkKernels.h"

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

using namespace trt_edgellm;

namespace trt_edgellm
{
namespace builder
{
#if NV_TENSORRT_MAJOR >= 11 || (NV_TENSORRT_MAJOR == 10 && NV_TENSORRT_MINOR >= 13)
namespace
{

void appendLunowudFlag(std::string& flags, std::string const& flag)
{
    if (flags.find(flag) != std::string::npos)
    {
        return;
    }
    if (!flags.empty())
    {
        flags += ' ';
    }
    flags += flag;
}

std::string applyMyelinCompileWorkarounds(int32_t maxBatchSize)
{
    std::string lunowudFlags;
    char const* existingLunowud = std::getenv("__LUNOWUD");
    if (existingLunowud)
    {
        lunowudFlags = existingLunowud;
    }
#if NV_TENSORRT_MAJOR == 10 && (NV_TENSORRT_MINOR == 13 || NV_TENSORRT_MINOR == 14)
    appendLunowudFlag(lunowudFlags, "-peep:match_dual_gemm=off");
#endif
#if NV_TENSORRT_MAJOR >= 11 || (NV_TENSORRT_MAJOR == 10 && NV_TENSORRT_MINOR >= 15)
    appendLunowudFlag(lunowudFlags, "-mlir:autotune:num_threads=1");
    appendLunowudFlag(lunowudFlags, "-mlir:collective:fp4=off");
    appendLunowudFlag(lunowudFlags, "-cask_fusion:async_policy=1");
    if (maxBatchSize == 1)
    {
        appendLunowudFlag(lunowudFlags, "-peep:fc_h_fusion=off");
    }
#endif
    if (existingLunowud || !lunowudFlags.empty())
    {
        setenv("__LUNOWUD", lunowudFlags.c_str(), 1);
    }
    return lunowudFlags;
}

} // namespace
#endif

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

bool isValidSpecDecodeType(std::string const& type)
{
    return type == "none" || type == "mtp" || type == "eagle3" || type == "dflash" || type == "gemma4_mtp";
}

bool isValidEngineRole(std::string const& role)
{
    return role == "llm" || role == "base" || role == "draft";
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
#if NV_TENSORRT_MAJOR >= 11 || NV_TENSORRT_MAJOR == 10 && NV_TENSORRT_MINOR >= 13
    std::string const lunowudFlags = applyMyelinCompileWorkarounds(mBuilderConfig.maxBatchSize);
    if (!lunowudFlags.empty())
    {
        LOG_INFO("Using __LUNOWUD=%s", lunowudFlags.c_str());
    }
#endif

    // Load plugin library
    auto pluginHandles = loadEdgellmPluginLib();

    // Parse model config
    if (!parseConfig())
    {
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

    // DFlash draft's DFlashTargetKVCacheUpdate plugin needs pages_per_slot (capPadded /
    // kTOKENS_PER_PAGE) to split its now-pool-shaped past_key_value binding's numPages back into
    // (maxBatch, cap). That fact is builder-only (mBuilderConfig.maxKVCacheCapacity), unavailable at
    // ONNX-export time, and this plugin's concrete class is intentionally not linked into the
    // builder (plugins are opaque .so modules, loaded via EDGELLM_PLUGIN_PATH — see
    // common/trtUtils.h). Resolve the plugin's exported configuration hook via dlsym on the already
    // -loaded handle instead of adding a new link dependency.
    if (isSpecDecodeDraft(mModelConfig, "dflash"))
    {
        int64_t const floorPages = rt::computeKvPoolFloorPages(
            static_cast<int32_t>(mBuilderConfig.maxBatchSize), static_cast<int32_t>(mBuilderConfig.maxKVCacheCapacity));
        if (mBuilderConfig.resolvedKVPoolPages() != floorPages)
        {
            LOG_ERROR("DFlash draft KV is identity-only and requires maxKVPoolPages=%ld; got %ld.", floorPages,
                mBuilderConfig.resolvedKVPoolPages());
            return false;
        }

        using ConfigurePagesPerSlotFn = bool (*)(nvinfer1::INetworkDefinition*, int32_t);
        auto* configureFn = reinterpret_cast<ConfigurePagesPerSlotFn>(
            dlsym(pluginHandles.get(), "edgellm_dflash_configure_pages_per_slot"));
        if (configureFn == nullptr)
        {
            LOG_ERROR(
                "Failed to resolve edgellm_dflash_configure_pages_per_slot from the plugin library: %s", dlerror());
            return false;
        }
        int32_t const pagesPerSlot = rt::computeMaxPagesPerSeq(static_cast<int32_t>(mBuilderConfig.maxKVCacheCapacity));
        if (!configureFn(network.get(), pagesPerSlot))
        {
            LOG_ERROR("Failed to configure DFlashTargetKVCacheUpdate plugin pages_per_slot=%d", pagesPerSlot);
            return false;
        }
    }

    // Print network information
    LOG_DEBUG("%s", printNetworkInfo(network.get(), "LLM").c_str());

    LOG_DEBUG(
        "ONNX parsing complete. mNbKVCacheInputs=%d, mNumLinearAttnLayers=%d", mNbKVCacheInputs, mNumLinearAttnLayers);

    // Create builder config
    auto config = createBuilderConfig(builder.get());
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

    // Create engine directory
    if (!std::filesystem::exists(mEngineDir))
    {
        if (!std::filesystem::create_directories(mEngineDir))
        {
            LOG_ERROR("Failed to create directory %s", mEngineDir.string().c_str());
            return false;
        }
        LOG_INFO("Created directory %s for saving LLM engine.", mEngineDir.string().c_str());
    }

    // Determine engine file name
    std::string engineFileName;
    if (mBuilderConfig.specDraft)
    {
        engineFileName = "spec_draft.engine";
    }
    else if (mBuilderConfig.specBase)
    {
        engineFileName = "spec_base.engine";
    }
    else
    {
        engineFileName = "llm.engine";
    }

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

    // Copy files and save builder config
    if (!copyConfig())
    {
        return false;
    }

    if (!copyTokenizerFiles())
    {
        return false;
    }

    if (!copyEagleFiles())
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

    if (!copyExternalWeightFiles())
    {
        return false;
    }

    return true;
}

bool LLMBuilder::parseConfig()
{
    std::string const jsonPath = (mOnnxDir / "config.json").string();
    if (!loadJsonConfig(jsonPath, mModelConfig))
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
            "Invalid spec_decode_type='%s'. Expected one of: none, mtp, eagle3, dflash, gemma4_mtp.", specType.c_str());
        return false;
    }
    if (!isValidEngineRole(role))
    {
        LOG_ERROR("Invalid engine_role='%s'. Expected one of: llm, base, draft.", role.c_str());
        return false;
    }
    if ((role == "llm") != (specType == "none"))
    {
        LOG_ERROR(
            "Invalid config: engine_role='%s' with spec_decode_type='%s'. LLM engines require "
            "spec_decode_type=none; speculative base/draft engines require a non-none spec_decode_type.",
            role.c_str(), specType.c_str());
        return false;
    }
    if ((mBuilderConfig.specDraft && role != "draft") || (mBuilderConfig.specBase && role != "base")
        || (!mBuilderConfig.specDraft && !mBuilderConfig.specBase && role != "llm"))
    {
        LOG_ERROR(
            "Build mode does not match config: engine_role='%s' (use --specBase for base, --specDraft for "
            "draft, and neither flag for vanilla LLM).",
            role.c_str());
        return false;
    }

    mHiddenSize = mModelConfig["hidden_size"].get<int32_t>();
    // For MTP draft, base model outputs hidden_size (1x); for EAGLE3 draft, it outputs hidden_size * 3.
    if (isSpecDecodeDraft(mModelConfig, "mtp"))
    {
        mTargetModelOutputHiddenDim = mHiddenSize;
    }
    else if ((isSpecDecodeDraft(mModelConfig, "dflash") || isSpecDecodeDraft(mModelConfig, "gemma4_mtp"))
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

    mNumLinearAttnLayers = mModelConfig.value("num_linear_attn_layers", 0);
    mRecurrentStateNumHeads = mModelConfig.value("recurrent_state_num_heads", 0);
    mRecurrentStateHeadDim = mModelConfig.value("recurrent_state_head_dim", 0);
    mRecurrentStateSize = mModelConfig.value("recurrent_state_size", 0);
    mConvDim = mModelConfig.value("conv_dim", 0);
    mConvKernel = mModelConfig.value("conv_kernel", 0);

    // For hybrid models, only attention layers have KV caches
    if (mNumLinearAttnLayers > 0)
    {
        mNbKVCacheInputs = mModelConfig.value("num_attention_layers", mModelConfig["num_hidden_layers"].get<int32_t>());
    }
    else
    {
        mNbKVCacheInputs = mModelConfig["num_hidden_layers"].get<int32_t>();
    }

    // Build per-layer head size vector for heterogeneous models (e.g. Gemma4).
    // Prefer kv_layer_configs (authoritative per-layer dims) when available;
    // fall back to global_head_dim + layer_types for older exports.
    int64_t globalHeadSize = mModelConfig.value("global_head_dim", static_cast<int64_t>(0));
    if (mModelConfig.contains("kv_layer_configs") && !mModelConfig["kv_layer_configs"].is_null())
    {
        auto const& kvLayerConfigs = mModelConfig["kv_layer_configs"];
        check::check(static_cast<int>(kvLayerConfigs.size()) >= mNbKVCacheInputs,
            "kv_layer_configs has fewer entries than expected KV cache layers");
        for (int i = 0; i < mNbKVCacheInputs; ++i)
        {
            auto const& lc = kvLayerConfigs[i];
            int64_t layerHeadDim
                = (lc.is_null() || !lc.contains("head_dim")) ? mHeadSize : lc["head_dim"].get<int64_t>();
            mPerLayerHeadSize.push_back(layerHeadDim);
            int64_t layerNumKVHeads
                = (lc.is_null() || !lc.contains("num_kv_heads")) ? mNumKVHeads : lc["num_kv_heads"].get<int64_t>();
            mPerLayerNumKVHeads.push_back(layerNumKVHeads);
        }
        LOG_INFO("Heterogeneous head sizes from kv_layer_configs: %d layers", mNbKVCacheInputs);
    }
    else if (globalHeadSize > 0 && globalHeadSize != mHeadSize && mModelConfig.contains("layer_types"))
    {
        auto const& layerTypes = mModelConfig["layer_types"];
        for (int i = 0; i < mNbKVCacheInputs; ++i)
        {
            std::string lt = (i < static_cast<int>(layerTypes.size())) ? layerTypes[i].get<std::string>() : "";
            mPerLayerHeadSize.push_back((lt == "full_attention") ? globalHeadSize : mHeadSize);
        }
        LOG_INFO("Heterogeneous head sizes: %d layers with head_dim=%ld, %ld layers with global_head_dim=%ld",
            mNbKVCacheInputs, mHeadSize, std::count(mPerLayerHeadSize.begin(), mPerLayerHeadSize.end(), globalHeadSize),
            globalHeadSize);
    }

    return true;
}

bool LLMBuilder::setupLLMOptimizationProfiles(
    nvinfer1::IBuilder& builder, nvinfer1::IBuilderConfig& config, nvinfer1::INetworkDefinition const& network)
{
    auto* contextProfile = builder.createOptimizationProfile();
    auto* generationProfile = builder.createOptimizationProfile();

    bool result = true;

    if (isSpecDecodeDraft(mModelConfig, "dflash"))
    {
        result &= setupDFlashDraftProfiles(*contextProfile, *generationProfile);
        if (!result)
        {
            LOG_ERROR("Failed to setup DFlash draft optimization profiles");
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

    // Setup common profiles
    result &= setupCommonProfiles(*contextProfile, *generationProfile);
    result &= setupRopeProfiles(*contextProfile, *generationProfile, network);

    LOG_INFO("Classic LLM opt profiles: prefill seq min/opt/max = 1/%lld/%lld",
        static_cast<long long>(mBuilderConfig.resolvedOptInputLen()),
        static_cast<long long>(mBuilderConfig.maxInputLen));

    // Setup model-specific profiles
    if (mBuilderConfig.specBase || mBuilderConfig.specDraft)
    {
        result &= setupSpecDecodeProfiles(*contextProfile, *generationProfile);
    }
    else
    {
        result &= setupVanillaProfiles(*contextProfile, *generationProfile);
    }

    // Setup hybrid state profiles for MTP/DFlash base models.
    if (isSpecDecodeBase(mModelConfig, "mtp") || isSpecDecodeBase(mModelConfig, "dflash"))
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

bool LLMBuilder::setupCommonProfiles(
    nvinfer1::IOptimizationProfile& contextProfile, nvinfer1::IOptimizationProfile& generationProfile)
{
    bool result = true;

    // Context lengths
    result &= setOptimizationProfile(&contextProfile, binding_names::kContextLengths, createDims({1}),
        createDims({mBuilderConfig.maxBatchSize}), createDims({mBuilderConfig.maxBatchSize}));
    result &= setOptimizationProfile(&generationProfile, binding_names::kContextLengths, createDims({1}),
        createDims({mBuilderConfig.maxBatchSize}), createDims({mBuilderConfig.maxBatchSize}));

    // For KVCacheStartIndex, we use zero shape to indicate the kvcache is empty for all sequences in the batch.
    // This can help distinguish the normal prefill and chunked prefill execution.
    result &= setOptimizationProfile(&contextProfile, binding_names::kKVCacheStartIndex, createDims({0}),
        createDims({mBuilderConfig.maxBatchSize}), createDims({mBuilderConfig.maxBatchSize}));
    result &= setOptimizationProfile(&generationProfile, binding_names::kKVCacheStartIndex, createDims({1}),
        createDims({mBuilderConfig.maxBatchSize}), createDims({mBuilderConfig.maxBatchSize}));

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

    return result;
}

bool LLMBuilder::setupRopeProfiles(nvinfer1::IOptimizationProfile& contextProfile,
    nvinfer1::IOptimizationProfile& generationProfile, nvinfer1::INetworkDefinition const& network)
{
    bool result = true;
    auto setRopeProfile = [&](char const* bindingName, int64_t rotaryDim) {
        result &= setOptimizationProfile(&contextProfile, bindingName,
            createDims({1, mBuilderConfig.maxKVCacheCapacity, rotaryDim}),
            createDims({mBuilderConfig.maxBatchSize, mBuilderConfig.maxKVCacheCapacity, rotaryDim}),
            createDims({mBuilderConfig.maxBatchSize, mBuilderConfig.maxKVCacheCapacity, rotaryDim}));
        result &= setOptimizationProfile(&generationProfile, bindingName,
            createDims({1, mBuilderConfig.maxKVCacheCapacity, rotaryDim}),
            createDims({mBuilderConfig.maxBatchSize, mBuilderConfig.maxKVCacheCapacity, rotaryDim}),
            createDims({mBuilderConfig.maxBatchSize, mBuilderConfig.maxKVCacheCapacity, rotaryDim}));
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

bool LLMBuilder::setupVanillaProfiles(
    nvinfer1::IOptimizationProfile& contextProfile, nvinfer1::IOptimizationProfile& generationProfile)
{
    bool result = true;

    // Input embeddings - always dynamic
    result &= setOptimizationProfile(&contextProfile, binding_names::kInputsEmbeds, createDims({1, 1, mHiddenSize}),
        createDims({mBuilderConfig.maxBatchSize, mBuilderConfig.resolvedOptInputLen(), mHiddenSize}),
        createDims({mBuilderConfig.maxBatchSize, mBuilderConfig.maxInputLen, mHiddenSize}));
    result &= setOptimizationProfile(&generationProfile, binding_names::kInputsEmbeds, createDims({1, 1, mHiddenSize}),
        createDims({mBuilderConfig.maxBatchSize, 1, mHiddenSize}),
        createDims({mBuilderConfig.maxBatchSize, 1, mHiddenSize}));

    if (mModelConfig.value("use_vision_bidirectional_attention", false))
    {
        result &= setOptimizationProfile(&contextProfile, binding_names::kVisionBlockIds, createDims({1, 1}),
            createDims({mBuilderConfig.maxBatchSize, std::max<int64_t>(1, mBuilderConfig.resolvedOptInputLen())}),
            createDims({mBuilderConfig.maxBatchSize, mBuilderConfig.maxInputLen}));
        // Decode ignores block IDs, but the static engine binding remains present.
        result &= setOptimizationProfile(&generationProfile, binding_names::kVisionBlockIds, createDims({1, 1}),
            createDims({mBuilderConfig.maxBatchSize, 1}), createDims({mBuilderConfig.maxBatchSize, 1}));
    }

    // Last token IDs
    result &= setOptimizationProfile(&contextProfile, binding_names::kLastTokenIds, createDims({1, 1}),
        createDims({mBuilderConfig.maxBatchSize, 1}), createDims({mBuilderConfig.maxBatchSize, 1}));
    result &= setOptimizationProfile(&generationProfile, binding_names::kLastTokenIds, createDims({1, 1}),
        createDims({mBuilderConfig.maxBatchSize, 1}), createDims({mBuilderConfig.maxBatchSize, 1}));

    return result;
}

bool LLMBuilder::setupSpecDecodeProfiles(
    nvinfer1::IOptimizationProfile& contextProfile, nvinfer1::IOptimizationProfile& generationProfile)
{
    bool result = true;

    int const maxTokens = mBuilderConfig.specDraft ? mBuilderConfig.maxDraftTreeSize : mBuilderConfig.maxVerifyTreeSize;

    // Input embeddings
    result &= setOptimizationProfile(&contextProfile, binding_names::kInputsEmbeds, createDims({1, 1, mHiddenSize}),
        createDims({mBuilderConfig.maxBatchSize, mBuilderConfig.resolvedOptInputLen(), mHiddenSize}),
        createDims({mBuilderConfig.maxBatchSize, mBuilderConfig.maxInputLen, mHiddenSize}));
    result &= setOptimizationProfile(&generationProfile, binding_names::kInputsEmbeds, createDims({1, 1, mHiddenSize}),
        createDims({mBuilderConfig.maxBatchSize, maxTokens / 2, mHiddenSize}),
        createDims({mBuilderConfig.maxBatchSize, maxTokens, mHiddenSize}));

    // Last token IDs - 2D shape [batch_size, num_selected_tokens]
    result &= setOptimizationProfile(&contextProfile, binding_names::kLastTokenIds, createDims({1, 1}),
        createDims({mBuilderConfig.maxBatchSize, 1}), createDims({mBuilderConfig.maxBatchSize, 1}));
    result &= setOptimizationProfile(&generationProfile, binding_names::kLastTokenIds, createDims({1, 1}),
        createDims({mBuilderConfig.maxBatchSize, maxTokens / 2}), createDims({mBuilderConfig.maxBatchSize, maxTokens}));

    if (mBuilderConfig.specDraft)
    {
        // Hidden states from draft
        result &= setOptimizationProfile(&contextProfile, binding_names::kDraftModelHiddenStates,
            createDims({1, 1, mHiddenSize}),
            createDims({mBuilderConfig.maxBatchSize, mBuilderConfig.resolvedOptInputLen(), mHiddenSize}),
            createDims({mBuilderConfig.maxBatchSize, mBuilderConfig.maxInputLen, mHiddenSize}));
        result &= setOptimizationProfile(&generationProfile, binding_names::kDraftModelHiddenStates,
            createDims({1, 1, mHiddenSize}), createDims({mBuilderConfig.maxBatchSize, maxTokens / 2, mHiddenSize}),
            createDims({mBuilderConfig.maxBatchSize, maxTokens, mHiddenSize}));

        // Hidden states input
        result &= setOptimizationProfile(&contextProfile, binding_names::kBaseModelHiddenStates,
            createDims({1, 1, mTargetModelOutputHiddenDim}),
            createDims({mBuilderConfig.maxBatchSize, mBuilderConfig.resolvedOptInputLen(), mTargetModelOutputHiddenDim}),
            createDims({mBuilderConfig.maxBatchSize, mBuilderConfig.maxInputLen, mTargetModelOutputHiddenDim}));
        result &= setOptimizationProfile(&generationProfile, binding_names::kBaseModelHiddenStates,
            createDims({1, 1, mTargetModelOutputHiddenDim}),
            createDims({mBuilderConfig.maxBatchSize, maxTokens / 2, mTargetModelOutputHiddenDim}),
            createDims({mBuilderConfig.maxBatchSize, maxTokens, mTargetModelOutputHiddenDim}));
    }

    // Attention mask and position ID
    if (mBuilderConfig.specDraft || mBuilderConfig.specBase)
    {
        int32_t const attnMaskAlignSize = 32;
        result &= setOptimizationProfile(&contextProfile, binding_names::kAttentionMask, createDims({1, 1, 1}),
            createDims({mBuilderConfig.maxBatchSize, 1, 1}), createDims({mBuilderConfig.maxBatchSize, 1, 1}));
        result &= setOptimizationProfile(&generationProfile, binding_names::kAttentionMask, createDims({1, 1, 1}),
            createDims({mBuilderConfig.maxBatchSize, maxTokens / 2,
                static_cast<int64_t>(divUp(maxTokens / 2, attnMaskAlignSize) * attnMaskAlignSize)}),
            createDims({mBuilderConfig.maxBatchSize, maxTokens,
                static_cast<int64_t>(divUp(maxTokens, attnMaskAlignSize) * attnMaskAlignSize)}));

        result &= setOptimizationProfile(&contextProfile, binding_names::kAttentionPosId, createDims({1, 1}),
            createDims({mBuilderConfig.maxBatchSize, 1}), createDims({mBuilderConfig.maxBatchSize, 1}));
        result &= setOptimizationProfile(&generationProfile, binding_names::kAttentionPosId, createDims({1, 1}),
            createDims({mBuilderConfig.maxBatchSize, maxTokens / 2}),
            createDims({mBuilderConfig.maxBatchSize, maxTokens}));
    }

    return result;
}

bool LLMBuilder::setupDFlashDraftProfiles(
    nvinfer1::IOptimizationProfile& contextProfile, nvinfer1::IOptimizationProfile& generationProfile)
{
    bool result = true;

    int64_t const maxDraftTokens = std::max<int64_t>(1, mBuilderConfig.maxDraftTreeSize);
    int64_t const optDraftTokens = maxDraftTokens;
    int64_t const maxPrefillTargetHiddenLen = std::max<int64_t>(1, mBuilderConfig.maxInputLen);
    int64_t const optPrefillTargetHiddenLen = mBuilderConfig.resolvedOptInputLen();
    int64_t const maxDecodeTargetHiddenLen = maxDraftTokens;
    int64_t const optDecodeTargetHiddenLen = maxDraftTokens;

    int64_t const packedMaskLen = static_cast<int64_t>(divUp(maxDraftTokens, 32));
    int64_t const optPackedMaskLen = static_cast<int64_t>(divUp(optDraftTokens, 32));

    // Profile 0 handles round-0/system-prompt cache update, where target hidden spans the prompt.
    // Profile 1 handles steady-state block proposal, where target hidden delta is bounded by block size.
    auto setupOneProfile = [&](nvinfer1::IOptimizationProfile& profile, int64_t optTargetHiddenLen,
                               int64_t maxTargetHiddenLen) {
        bool ok = true;
        // inputs_embeds: [batch, block_seq, hiddenSize]
        ok &= setOptimizationProfile(&profile, binding_names::kInputsEmbeds, createDims({1, 1, mHiddenSize}),
            createDims({mBuilderConfig.maxBatchSize, optDraftTokens, mHiddenSize}),
            createDims({mBuilderConfig.maxBatchSize, maxDraftTokens, mHiddenSize}));
        // dflash_target_hidden_concat: [batch, delta_seq, baseOutputHiddenDim]
        ok &= setOptimizationProfile(&profile, binding_names::kDFlashTargetHiddenConcat,
            createDims({1, 1, mTargetModelOutputHiddenDim}),
            createDims({mBuilderConfig.maxBatchSize, optTargetHiddenLen, mTargetModelOutputHiddenDim}),
            createDims({mBuilderConfig.maxBatchSize, maxTargetHiddenLen, mTargetModelOutputHiddenDim}));
        // rope_rotary_cos_sin: [1, kv_capacity, rotaryDim]
        ok &= setOptimizationProfile(&profile, binding_names::kRopeCosSin, createDims({1, 1, mRotaryDim}),
            createDims({1, mBuilderConfig.maxKVCacheCapacity, mRotaryDim}),
            createDims({1, mBuilderConfig.maxKVCacheCapacity, mRotaryDim}));
        // context_lengths: [batch]
        ok &= setOptimizationProfile(&profile, binding_names::kContextLengths, createDims({1}),
            createDims({mBuilderConfig.maxBatchSize}), createDims({mBuilderConfig.maxBatchSize}));
        // kvcache_start_index: [batch]
        ok &= setOptimizationProfile(&profile, binding_names::kKVCacheStartIndex, createDims({1}),
            createDims({mBuilderConfig.maxBatchSize}), createDims({mBuilderConfig.maxBatchSize}));
        // kv_page_table: [batch, 2, maxPagesPerSeq] int32. Proposal self-attention's page
        // table (default identity); the draft's own KV cache above has no page table.
        int32_t const maxPagesPerSeq
            = rt::computeMaxPagesPerSeq(static_cast<int32_t>(mBuilderConfig.maxKVCacheCapacity));
        ok &= setOptimizationProfile(&profile, binding_names::kKVPageTable, createDims({1, 2, maxPagesPerSeq}),
            createDims({mBuilderConfig.maxBatchSize, 2, maxPagesPerSeq}),
            createDims({mBuilderConfig.maxBatchSize, 2, maxPagesPerSeq}));
        // dflash_delta_lengths: [batch]
        ok &= setOptimizationProfile(&profile, binding_names::kDFlashDeltaLengths, createDims({1}),
            createDims({mBuilderConfig.maxBatchSize}), createDims({mBuilderConfig.maxBatchSize}));
        // attention_mask: [batch, block_seq, packed_mask_len]
        ok &= setOptimizationProfile(&profile, binding_names::kAttentionMask, createDims({1, 1, 1}),
            createDims({mBuilderConfig.maxBatchSize, optDraftTokens, optPackedMaskLen}),
            createDims({mBuilderConfig.maxBatchSize, maxDraftTokens, packedMaskLen}));
        // attention_pos_id: [batch, block_seq]
        ok &= setOptimizationProfile(&profile, binding_names::kAttentionPosId, createDims({1, 1}),
            createDims({mBuilderConfig.maxBatchSize, optDraftTokens}),
            createDims({mBuilderConfig.maxBatchSize, maxDraftTokens}));
        // KV cache per-layer: DFlash's own combined draft cache, now unified on the paged-pool
        // contract shared with the AttentionPlugin binding: [2, numPages, kTOKENS_PER_PAGE,
        // numKVHeads, headDim] (single fixed numPages value, same as setupKVCacheProfiles).
        // DFlashTargetKVCacheUpdatePlugin recovers maxBatch/cap at enqueue time from numPages and
        // the pages_per_slot attribute the builder configures via edgellm_dflash_configure_pages_per_slot
        // (see build()); this cache still has no page table of its own.
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
        return setOptimizationProfile(&profile, bindingName,
            createDims({1, mBuilderConfig.maxKVCacheCapacity, rotaryDim}),
            createDims({mBuilderConfig.maxBatchSize, mBuilderConfig.maxKVCacheCapacity, rotaryDim}),
            createDims({mBuilderConfig.maxBatchSize, mBuilderConfig.maxKVCacheCapacity, rotaryDim}));
    };

    auto setupOneProfile = [&](nvinfer1::IOptimizationProfile& profile) {
        bool ok = true;
        ok &= setOptimizationProfile(&profile, binding_names::kInputsEmbeds, createDims({1, 1, baseHiddenSize}),
            createDims({mBuilderConfig.maxBatchSize, 1, baseHiddenSize}),
            createDims({mBuilderConfig.maxBatchSize, 1, baseHiddenSize}));
        ok &= setOptimizationProfile(&profile, binding_names::kBaseModelHiddenStates,
            createDims({1, 1, baseHiddenSize}), createDims({mBuilderConfig.maxBatchSize, 1, baseHiddenSize}),
            createDims({mBuilderConfig.maxBatchSize, 1, baseHiddenSize}));
        ok &= setOptimizationProfile(&profile, binding_names::kContextLengths, createDims({1}),
            createDims({mBuilderConfig.maxBatchSize}), createDims({mBuilderConfig.maxBatchSize}));

        // kv_page_table: [batch, 2, maxPagesPerSeq] int32 — the TARGET pool's page table
        // (the assistant reads the target's paged KV pool; the runtime binds the target's
        // identity table here).
        int32_t const maxPagesPerSeq
            = rt::computeMaxPagesPerSeq(static_cast<int32_t>(mBuilderConfig.maxKVCacheCapacity));
        ok &= setOptimizationProfile(&profile, binding_names::kKVPageTable, createDims({1, 2, maxPagesPerSeq}),
            createDims({mBuilderConfig.maxBatchSize, 2, maxPagesPerSeq}),
            createDims({mBuilderConfig.maxBatchSize, 2, maxPagesPerSeq}));

        if (auto const rotaryDim = getStaticInputDim(network, binding_names::kRopeCosSinSliding, 2))
        {
            ok &= setupRopeProfile(profile, binding_names::kRopeCosSinSliding, *rotaryDim);
        }
        if (auto const rotaryDim = getStaticInputDim(network, binding_names::kRopeCosSinFull, 2))
        {
            ok &= setupRopeProfile(profile, binding_names::kRopeCosSinFull, *rotaryDim);
        }
        if (auto const rotaryDim = getStaticInputDim(network, binding_names::kRopeCosSin, 2))
        {
            ok &= setupRopeProfile(profile, binding_names::kRopeCosSin, *rotaryDim);
        }

        // KV cache per-layer: the assistant binds the TARGET model's paged pool tensors
        // directly ([2, numPages, kTOKENS_PER_PAGE, numKVHeads, headDim]), so the profile
        // uses the same fixed page count as the target's setupKVCacheProfiles.
        int64_t const numPages = mBuilderConfig.resolvedKVPoolPages();
        for (int i = 0; i < mNbKVCacheInputs; ++i)
        {
            int64_t const layerHeadSize = (!mPerLayerHeadSize.empty()) ? mPerLayerHeadSize[i] : mHeadSize;
            int64_t const layerNumKVHeads = (!mPerLayerNumKVHeads.empty()) ? mPerLayerNumKVHeads[i] : mNumKVHeads;
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

bool LLMBuilder::setupPleProfiles(nvinfer1::IOptimizationProfile& contextProfile,
    nvinfer1::IOptimizationProfile& generationProfile, nvinfer1::INetworkDefinition const& network)
{
    bool result = true;
    bool foundPleInput = false;
    std::string_view const prefix = binding_names::kPleTokenEmbedsTemplate;

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
        if (inputDims.nbDims != 3 || inputDims.d[2] <= 0)
        {
            LOG_ERROR("PLE input %s must be rank-3 [batch, seq_len, hidden] with static hidden dimension.", inputName);
            result = false;
            continue;
        }

        int64_t const pleHiddenSize = inputDims.d[2];
        result &= setOptimizationProfile(&contextProfile, inputName, createDims({1, 1, pleHiddenSize}),
            createDims(
                {mBuilderConfig.maxBatchSize, std::max<int64_t>(1, mBuilderConfig.resolvedOptInputLen()), pleHiddenSize}),
            createDims({mBuilderConfig.maxBatchSize, mBuilderConfig.maxInputLen, pleHiddenSize}));

        if (mBuilderConfig.specBase || mBuilderConfig.specDraft)
        {
            int64_t const maxTokens
                = mBuilderConfig.specDraft ? mBuilderConfig.maxDraftTreeSize : mBuilderConfig.maxVerifyTreeSize;
            result &= setOptimizationProfile(&generationProfile, inputName, createDims({1, 1, pleHiddenSize}),
                createDims({mBuilderConfig.maxBatchSize, std::max<int64_t>(1, maxTokens / 2), pleHiddenSize}),
                createDims({mBuilderConfig.maxBatchSize, maxTokens, pleHiddenSize}));
        }
        else
        {
            result &= setOptimizationProfile(&generationProfile, inputName, createDims({1, 1, pleHiddenSize}),
                createDims({mBuilderConfig.maxBatchSize, 1, pleHiddenSize}),
                createDims({mBuilderConfig.maxBatchSize, 1, pleHiddenSize}));
        }
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

    // Setup profiles for all detected deepstack_embeds inputs
    // These have the same shape as inputs_embeds: [batch_size, seq_len, hidden_size]
    for (auto const& deepstackInputName : deepstackInputs)
    {
        LOG_INFO("Setting up optimization profile for %s", deepstackInputName.c_str());

        // Same profile as inputs_embeds
        result &= setOptimizationProfile(&contextProfile, deepstackInputName.c_str(), createDims({1, 1, mHiddenSize}),
            createDims({mBuilderConfig.maxBatchSize, mBuilderConfig.resolvedOptInputLen(), mHiddenSize}),
            createDims({mBuilderConfig.maxBatchSize, mBuilderConfig.maxInputLen, mHiddenSize}));

        if (mBuilderConfig.specBase || mBuilderConfig.specDraft)
        {
            int const maxTokens
                = mBuilderConfig.specDraft ? mBuilderConfig.maxDraftTreeSize : mBuilderConfig.maxVerifyTreeSize;
            result &= setOptimizationProfile(&generationProfile, deepstackInputName.c_str(),
                createDims({1, 1, mHiddenSize}), createDims({mBuilderConfig.maxBatchSize, maxTokens / 2, mHiddenSize}),
                createDims({mBuilderConfig.maxBatchSize, maxTokens, mHiddenSize}));
        }
        else
        {
            result &= setOptimizationProfile(&generationProfile, deepstackInputName.c_str(),
                createDims({1, 1, mHiddenSize}), createDims({mBuilderConfig.maxBatchSize, 1, mHiddenSize}),
                createDims({mBuilderConfig.maxBatchSize, 1, mHiddenSize}));
        }
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

    // Detect if lm_head_weight input exists (CodePredictor model)
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
    // Plugin path: paged pool binding [2, numPages, kTOKENS_PER_PAGE, num_kv_heads, head_dim].
    // numPages is fixed at the engine-authoritative pool count for the life of the engine — it is
    // not resized per inference step. The default is the active-capacity floor; an explicit larger
    // value reserves retention headroom.
    // "Empty vs non-empty" cache is conveyed by kvcache_start_index's own profile, not by this
    // tensor's shape (the plugin reads numPages from dims.d[1], so the binding must always be
    // pool-shaped).
    int64_t const numPages = mBuilderConfig.resolvedKVPoolPages();
    for (int i = 0; i < mNbKVCacheInputs; ++i)
    {
        // Per-layer dims mirror the runtime registry (kv_layer_configs): head size varies on
        // Gemma4 today; the KV head count is per-layer for the same forward-compat reason.
        int64_t layerHeadSize = (!mPerLayerHeadSize.empty()) ? mPerLayerHeadSize[i] : mHeadSize;
        int64_t layerNumKVHeads = (!mPerLayerNumKVHeads.empty()) ? mPerLayerNumKVHeads[i] : mNumKVHeads;
        nvinfer1::Dims kvCacheShape = createDims({2, numPages, rt::kTOKENS_PER_PAGE, layerNumKVHeads, layerHeadSize});

        result &= setOptimizationProfile(&contextProfile, binding_names::formatKVCacheName(i, true).c_str(),
            kvCacheShape, kvCacheShape, kvCacheShape);
        result &= setOptimizationProfile(&generationProfile, binding_names::formatKVCacheName(i, true).c_str(),
            kvCacheShape, kvCacheShape, kvCacheShape);
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
    nvinfer1::Dims minRecurrentShape
        = createDims({1, mRecurrentStateNumHeads, mRecurrentStateHeadDim, mRecurrentStateSize});
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
    nvinfer1::Dims minConvShape = createDims({1, mConvDim, mConvKernel});
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

    if (mBuilderConfig.maxVerifyTreeSize > kernel::kGDN_TREE_CHUNK_MAX_NODES)
    {
        LOG_ERROR("Hybrid speculative base maxVerifyTreeSize=%lld exceeds compact GDN limit %d",
            static_cast<long long>(mBuilderConfig.maxVerifyTreeSize), kernel::kGDN_TREE_CHUNK_MAX_NODES);
        return false;
    }

    int32_t const h = kernel::gdnTreeChunkNumKeyHeads(
        mConvDim, mRecurrentStateNumHeads, mRecurrentStateHeadDim, mRecurrentStateSize);
    if (mRecurrentStateHeadDim != 128 || mRecurrentStateSize != 128 || h <= 0 || mRecurrentStateNumHeads % h != 0)
    {
        LOG_ERROR("Hybrid speculative base has an unsupported GDN contract: convDim=%d hv=%d dk=%d dv=%d derived h=%d",
            mConvDim, mRecurrentStateNumHeads, mRecurrentStateHeadDim, mRecurrentStateSize, h);
        return false;
    }

    bool result = true;
    int64_t const compactElements = kernel::gdnTreeChunkBufferElements(h, mRecurrentStateNumHeads);
    nvinfer1::Dims minShape = createDims({1, compactElements});
    nvinfer1::Dims maxShape = createDims({mBuilderConfig.maxBatchSize, compactElements});

    for (int32_t i = 0; i < mNumLinearAttnLayers; ++i)
    {
        std::string const name = binding_names::formatIntermediateRecurrentStateName(i);
        result &= setOptimizationProfile(&contextProfile, name.c_str(), minShape, maxShape, maxShape);
        result &= setOptimizationProfile(&generationProfile, name.c_str(), minShape, maxShape, maxShape);
    }

    LOG_DEBUG("Set up compact recurrent replay-buffer profiles for %d recurrent layers", mNumLinearAttnLayers);
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

    // Intermediate conv state shape: [batch, seq_len, conv_dim, conv_kernel]
    nvinfer1::Dims minShape = createDims({1, 1, mConvDim, mConvKernel});
    nvinfer1::Dims optCtxShape
        = createDims({mBuilderConfig.maxBatchSize, mBuilderConfig.resolvedOptInputLen(), mConvDim, mConvKernel});
    nvinfer1::Dims maxCtxShape
        = createDims({mBuilderConfig.maxBatchSize, mBuilderConfig.maxInputLen, mConvDim, mConvKernel});
    nvinfer1::Dims optGenShape
        = createDims({mBuilderConfig.maxBatchSize, mBuilderConfig.maxVerifyTreeSize / 2, mConvDim, mConvKernel});
    nvinfer1::Dims maxGenShape
        = createDims({mBuilderConfig.maxBatchSize, mBuilderConfig.maxVerifyTreeSize, mConvDim, mConvKernel});

    for (int32_t i = 0; i < mNumLinearAttnLayers; ++i)
    {
        std::string const name = binding_names::formatIntermediateConvStateName(i);
        result &= setOptimizationProfile(&contextProfile, name.c_str(), minShape, optCtxShape, maxCtxShape);
        result &= setOptimizationProfile(&generationProfile, name.c_str(), minShape, optGenShape, maxGenShape);
    }

    LOG_DEBUG("Set up intermediate conv state profiles for %d recurrent layers (MTP/DFlash)", mNumLinearAttnLayers);
    return result;
}

bool LLMBuilder::setupLinearAttentionSpecVerifyProfiles(nvinfer1::IOptimizationProfile& contextProfile,
    nvinfer1::IOptimizationProfile& generationProfile, nvinfer1::INetworkDefinition const& network)
{
    if (mNumLinearAttnLayers == 0)
    {
        return true;
    }

    if (!hasInputBinding(network, binding_names::kSpecVerifyPhaseMarker))
    {
        LOG_ERROR("Hybrid MTP/DFlash base engine is missing input '%s'. Re-export the ONNX model.",
            binding_names::kSpecVerifyPhaseMarker);
        return false;
    }

    bool result = true;
    result &= setOptimizationProfile(
        &contextProfile, binding_names::kSpecVerifyPhaseMarker, createDims({0}), createDims({0}), createDims({0}));
    result &= setOptimizationProfile(
        &generationProfile, binding_names::kSpecVerifyPhaseMarker, createDims({0}), createDims({1}), createDims({1}));

    auto setOptionalTreeProfile = [&](char const* inputName) {
        if (!hasInputBinding(network, inputName))
        {
            return true;
        }

        int64_t const maxTreeTokens = std::max<int64_t>(1, mBuilderConfig.maxVerifyTreeSize);
        bool ok = true;
        ok &= setOptimizationProfile(&contextProfile, inputName, createDims({1, 1}),
            createDims({mBuilderConfig.maxBatchSize, 1}), createDims({mBuilderConfig.maxBatchSize, maxTreeTokens}));
        ok &= setOptimizationProfile(&generationProfile, inputName, createDims({1, 1}),
            createDims({mBuilderConfig.maxBatchSize, std::max<int64_t>(1, maxTreeTokens / 2)}),
            createDims({mBuilderConfig.maxBatchSize, maxTreeTokens}));
        return ok;
    };

    result &= setOptionalTreeProfile(binding_names::kTreeParentIds);
    result &= setOptionalTreeProfile(binding_names::kTreeDepths);

    LOG_DEBUG("Set up hybrid linear-attention spec-verify profiles for %d recurrent layers", mNumLinearAttnLayers);
    return result;
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
    // Determine config file name based on model type
    std::string configFileName;
    if (mBuilderConfig.specDraft)
    {
        configFileName = "draft_config.json";
    }
    else if (mBuilderConfig.specBase)
    {
        configFileName = "base_config.json";
    }
    else
    {
        configFileName = "config.json";
    }

    std::string const targetConfigPath = (mEngineDir / configFileName).string();

    // Create a copy of mModelConfig and add builder config
    Json configWithBuilder = mModelConfig;
    configWithBuilder["builder_config"] = mBuilderConfig.toJson();

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
                    kvLayerConfigs.push_back(
                        Json{{"num_kv_heads", layerNumKVHeads}, {"head_dim", mPerLayerHeadSize[attnIdx]}});
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
                kvLayerConfigs.push_back(Json{{"num_kv_heads", layerNumKVHeads}, {"head_dim", mPerLayerHeadSize[i]}});
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

    std::vector<std::string> tokenizerFiles
        = {"tokenizer_config.json", "tokenizer.json", "processed_chat_template.json"};
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

    return allSuccess;
}

bool LLMBuilder::copyEagleFiles()
{
    // Copy d2t.safetensors for Eagle3 draft models only. MTP/DFlash drafts share vocab with base and have no d2t.
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

bool LLMBuilder::copyVocabMappingFiles()
{
    // Copy the vocab map sidecar if reduced vocabulary is used. Base engines
    // consume vocab_map.safetensors; DFlash draft engines consume
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
            // sidecar is required. Runtime will refuse to load (DFlashDecoder
            // hard-errors when reducedVocabSize > 0 and the file is missing;
            // base-model runtime falls back to no remap but produces wrong IDs).
            // Fail the build instead of letting a broken engine ship.
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
