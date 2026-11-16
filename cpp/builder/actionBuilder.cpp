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

#include "actionBuilder.h"
#include "builderUtils.h"
#include "common/bindingNames.h"
#include "common/executionPhase.h"
#include "common/logger.h"
#include "common/pagedKvTypes.h"
#include "common/trtUtils.h"
#include "common/version.h"
#include "llmBuilder.h"

using namespace trt_edgellm;

namespace trt_edgellm
{
namespace builder
{

ActionBuilder::ActionBuilder(
    std::filesystem::path const& onnxDir, std::filesystem::path const& engineDir, ActionBuilderConfig const& config)
    : mOnnxDir(onnxDir)
    , mEngineDir(engineDir)
    , mBuilderConfig(config)
{
}

bool ActionBuilder::build()
{
    auto pluginHandles = loadEdgellmPluginLib();

    if (!parseConfig())
    {
        LOG_ERROR("Failed to parse action expert config from %s", mOnnxDir.string().c_str());
        return false;
    }

    auto [builder, network] = createBuilderAndNetwork();
    if (!builder || !network)
    {
        return false;
    }

    std::string onnxPath = (mOnnxDir / "model.onnx").string();
    auto parser = parseOnnxModel(network.get(), onnxPath);
    if (!parser)
    {
        LOG_ERROR("Failed to parse ONNX model from %s", onnxPath.c_str());
        return false;
    }

    LOG_DEBUG("%s", printNetworkInfo(network.get(), "Action").c_str());

    auto config = createBuilderConfig(builder.get());
    if (!config)
    {
        return false;
    }

    if (!setupActionOptimizationProfile(*builder.get(), *config.get(), *network.get()))
    {
        LOG_ERROR("Failed to setup action optimization profile");
        return false;
    }

    if (!std::filesystem::exists(mEngineDir))
    {
        if (!std::filesystem::create_directories(mEngineDir))
        {
            LOG_ERROR("Failed to create directory %s", mEngineDir.string().c_str());
            return false;
        }
        LOG_INFO("Created directory %s for saving Action engine.", mEngineDir.string().c_str());
    }

    std::string engineFilePath = (mEngineDir / "action.engine").string();
    if (!buildAndSerializeEngine(builder.get(), network.get(), config.get(), engineFilePath))
    {
        LOG_ERROR("Failed to build and serialize engine to %s", engineFilePath.c_str());
        return false;
    }

    if (!copyConfig())
    {
        LOG_ERROR("Failed to copy config to engine directory");
        return false;
    }

    return true;
}

bool ActionBuilder::parseConfig()
{
    std::string configPath = (mOnnxDir / "config.json").string();
    if (!loadJsonConfig(configPath, mModelConfig))
    {
        return false;
    }

    // Check model version
    std::string modelVersion = mModelConfig.value(binding_names::kEdgellmVersion, "");
    version::checkVersion(modelVersion);

    // Read model architecture parameters
    if (!mModelConfig.contains("num_hidden_layers"))
    {
        LOG_ERROR("num_hidden_layers not found in config.json");
        return false;
    }
    mNumLayers = mModelConfig["num_hidden_layers"].get<int32_t>();

    if (!mModelConfig.contains("num_key_value_heads"))
    {
        LOG_ERROR("num_key_value_heads not found in config.json");
        return false;
    }
    mNumHeads = mModelConfig["num_key_value_heads"].get<int32_t>();

    if (!mModelConfig.contains("head_dim"))
    {
        LOG_ERROR("head_dim not found in config.json");
        return false;
    }
    mHeadDim = mModelConfig["head_dim"].get<int32_t>();

    if (!mModelConfig.contains("n_diffusion_tokens"))
    {
        LOG_ERROR("n_diffusion_tokens not found in config.json");
        return false;
    }
    mNumDiffusionTokens = mModelConfig["n_diffusion_tokens"].get<int32_t>();

    if (mModelConfig.contains("builder_config"))
    {
        ActionBuilderConfig const exportedConfig = ActionBuilderConfig::fromJson(mModelConfig["builder_config"]);
        if (mBuilderConfig.maxKVCacheCapacity == 0)
        {
            mBuilderConfig.maxKVCacheCapacity = exportedConfig.maxKVCacheCapacity;
        }
    }

    return true;
}

namespace
{

int32_t findInputDim(nvinfer1::INetworkDefinition const& network, std::string const& inputName, int32_t dim)
{
    for (int32_t i = 0; i < network.getNbInputs(); ++i)
    {
        auto const* input = network.getInput(i);
        if (input != nullptr && inputName == input->getName())
        {
            auto const dims = input->getDimensions();
            return dim < dims.nbDims ? static_cast<int32_t>(dims.d[dim]) : 0;
        }
    }
    return 0;
}

} // namespace

bool ActionBuilder::setupActionOptimizationProfile(
    nvinfer1::IBuilder& builder, nvinfer1::IBuilderConfig& config, nvinfer1::INetworkDefinition const& network)
{
    if (findInputDim(network, binding_names::formatKCacheName(0, true), 2) > 0)
    {
        return setupDenseActionOptimizationProfile(builder, config, network);
    }
    return setupPagedActionOptimizationProfile(builder, config, network);
}

bool ActionBuilder::setupDenseActionOptimizationProfile(
    nvinfer1::IBuilder& builder, nvinfer1::IBuilderConfig& config, nvinfer1::INetworkDefinition const& network)
{
    auto* profile = builder.createOptimizationProfile();
    bool result = true;

    // kvcache_start_index: [batch] — one past-KV length per batch row
    result &= setOptimizationProfile(profile, binding_names::kKVCacheStartIndex, createDims({1}),
        createDims({mBuilderConfig.maxBatchSize}), createDims({mBuilderConfig.maxBatchSize}));

    // noise_trajectory: [batch, num_waypoints, 2]
    result &= setOptimizationProfile(profile, binding_names::kNoiseTrajectory, createDims({1, mNumDiffusionTokens, 2}),
        createDims({mBuilderConfig.maxBatchSize, mNumDiffusionTokens, 2}),
        createDims({mBuilderConfig.maxBatchSize, mNumDiffusionTokens, 2}));

    // rope_rotary_cos_sin: [batch, num_diffusion_tokens, head_dim]
    result
        &= setOptimizationProfile(profile, binding_names::kRopeCosSin, createDims({1, mNumDiffusionTokens, mHeadDim}),
            createDims({mBuilderConfig.maxBatchSize, mNumDiffusionTokens, mHeadDim}),
            createDims({mBuilderConfig.maxBatchSize, mNumDiffusionTokens, mHeadDim}));

    // attention_pos_id: [batch, num_diffusion_tokens]
    result &= setOptimizationProfile(profile, binding_names::kActionAttentionPosId,
        createDims({1, mNumDiffusionTokens}), createDims({mBuilderConfig.maxBatchSize, mNumDiffusionTokens}),
        createDims({mBuilderConfig.maxBatchSize, mNumDiffusionTokens}));

    // k_cache, v_cache: [batch, num_heads, max_capacity, head_dim]
    int32_t maxKVCacheCapacity = 0;
    std::string const kCache0Name = binding_names::formatKCacheName(0, true);
    for (int32_t i = 0; i < network.getNbInputs(); ++i)
    {
        auto* input = network.getInput(i);
        if (input->getName() == kCache0Name)
        {
            maxKVCacheCapacity = input->getDimensions().d[2];
        }
    }
    if (maxKVCacheCapacity == 0)
    {
        LOG_ERROR("Cannot infer maxKVCacheCapacity. Do you have proper k_cache and v_cache inputs in ONNX?");
        return false;
    }

    mBuilderConfig.maxKVCacheCapacity = maxKVCacheCapacity;

    // Alpamayo action graphs may use kvcache_start_index as both an execution
    // tensor and a shape tensor. The dimensions above describe its batch
    // extent; shape values cover the live KV-length range.
    nvinfer1::ITensor const* kvCacheStartInput = nullptr;
    for (int32_t i = 0; i < network.getNbInputs(); ++i)
    {
        auto const* input = network.getInput(i);
        if (std::string(input->getName()) == binding_names::kKVCacheStartIndex)
        {
            kvCacheStartInput = input;
            break;
        }
    }
    if (kvCacheStartInput != nullptr && kvCacheStartInput->isShapeTensor())
    {
        if (mBuilderConfig.maxBatchSize != 1)
        {
            LOG_ERROR("A shape-tensor kvcache_start_index currently supports action maxBatchSize=1 only");
            return false;
        }
        int64_t const minKVLength{0};
        int64_t const optKVLength{maxKVCacheCapacity / 2};
        int64_t const maxKVLength{maxKVCacheCapacity};
        result &= profile->setShapeValuesV2(
            binding_names::kKVCacheStartIndex, nvinfer1::OptProfileSelector::kMIN, &minKVLength, 1);
        result &= profile->setShapeValuesV2(
            binding_names::kKVCacheStartIndex, nvinfer1::OptProfileSelector::kOPT, &optKVLength, 1);
        result &= profile->setShapeValuesV2(
            binding_names::kKVCacheStartIndex, nvinfer1::OptProfileSelector::kMAX, &maxKVLength, 1);
    }

    nvinfer1::Dims minCacheDims = createDims({1, mNumHeads, maxKVCacheCapacity, mHeadDim});
    nvinfer1::Dims optCacheDims = createDims({mBuilderConfig.maxBatchSize, mNumHeads, maxKVCacheCapacity, mHeadDim});
    nvinfer1::Dims maxCacheDims = createDims({mBuilderConfig.maxBatchSize, mNumHeads, maxKVCacheCapacity, mHeadDim});

    for (int32_t i = 0; i < mNumLayers; ++i)
    {
        result &= setOptimizationProfile(
            profile, binding_names::formatKCacheName(i, true).c_str(), minCacheDims, optCacheDims, maxCacheDims);
        result &= setOptimizationProfile(
            profile, binding_names::formatVCacheName(i, true).c_str(), minCacheDims, optCacheDims, maxCacheDims);
    }

    if (!result)
    {
        LOG_ERROR("Failed to setup action optimization profile");
        return false;
    }

    LOG_DEBUG("%s", printOptimizationProfile(profile, "action_profile", &network).c_str());
    config.addOptimizationProfile(profile);

    return true;
}

bool ActionBuilder::setupPagedActionOptimizationProfile(
    nvinfer1::IBuilder& builder, nvinfer1::IBuilderConfig& config, nvinfer1::INetworkDefinition const& network)
{
    auto* profile = builder.createOptimizationProfile();
    bool result = true;

    // query_lengths/past_lengths/attention_sequence_lengths: [batch]
    result &= setOptimizationProfile(profile, binding_names::kQueryLengths, createDims({1}),
        createDims({mBuilderConfig.maxBatchSize}), createDims({mBuilderConfig.maxBatchSize}));
    result &= setOptimizationProfile(profile, binding_names::kPastLengths, createDims({1}),
        createDims({mBuilderConfig.maxBatchSize}), createDims({mBuilderConfig.maxBatchSize}));
    result &= setOptimizationProfile(profile, binding_names::kAttentionSequenceLengths, createDims({1}),
        createDims({mBuilderConfig.maxBatchSize}), createDims({mBuilderConfig.maxBatchSize}));
    result &= setOptimizationProfile(profile, binding_names::kQueryStartOffsets, createDims({2}),
        createDims({mBuilderConfig.maxBatchSize + 1}), createDims({mBuilderConfig.maxBatchSize + 1}));
    result &= setOptimizationProfile(profile, binding_names::kContextMaskSelector, createDims({1}),
        createDims({mBuilderConfig.maxBatchSize}), createDims({mBuilderConfig.maxBatchSize}));
    int32_t const denoisePhaseExtent = static_cast<int32_t>(rt::ExecutionPhase::kDiffusionDenoise);
    result &= setOptimizationProfile(profile, binding_names::kExecutionPhaseMarker, createDims({denoisePhaseExtent}),
        createDims({denoisePhaseExtent}), createDims({denoisePhaseExtent}));
    result &= setOptimizationProfile(
        profile, binding_names::kContextSequenceCountCarrier, createDims({0}), createDims({0}), createDims({0}));

    // noise_trajectory: [batch, num_waypoints, 2]
    result &= setOptimizationProfile(profile, binding_names::kNoiseTrajectory, createDims({1, mNumDiffusionTokens, 2}),
        createDims({mBuilderConfig.maxBatchSize, mNumDiffusionTokens, 2}),
        createDims({mBuilderConfig.maxBatchSize, mNumDiffusionTokens, 2}));

    // rope_rotary_cos_sin: [batch, num_diffusion_tokens, head_dim]
    result
        &= setOptimizationProfile(profile, binding_names::kRopeCosSin, createDims({1, mNumDiffusionTokens, mHeadDim}),
            createDims({mBuilderConfig.maxBatchSize, mNumDiffusionTokens, mHeadDim}),
            createDims({mBuilderConfig.maxBatchSize, mNumDiffusionTokens, mHeadDim}));

    std::string const kvPool0Name = binding_names::formatKVCacheName(0, true);
    int32_t const exportedKVPoolPages = findInputDim(network, kvPool0Name, 1);
    int32_t const exportedMaxPrefixPagesPerSeq = findInputDim(network, binding_names::kKVPageTable, 2);
    if (exportedKVPoolPages == 0 || exportedMaxPrefixPagesPerSeq == 0)
    {
        LOG_ERROR("Cannot infer paged action KV geometry. Do you have past_key_values_0 and kv_page_table inputs?");
        return false;
    }

    if (mBuilderConfig.maxKVCacheCapacity <= 0)
    {
        LOG_ERROR("Action builder requires builder_config.max_kv_cache_capacity in the exported config.json.");
        return false;
    }
    ELLM_CHECK(mNumDiffusionTokens > 0 && mNumDiffusionTokens <= rt::kMAX_KV_CACHE_CAPACITY - rt::kTOKENS_PER_PAGE,
        "Action token count exceeds the supported page-aligned capacity.");
    ELLM_CHECK(mBuilderConfig.maxKVCacheCapacity <= rt::kMAX_KV_CACHE_CAPACITY - mNumDiffusionTokens,
        "Action prefix plus action tokens exceed the supported page-aligned capacity.");
    LLMBuilderConfig poolConfig;
    poolConfig.maxBatchSize = mBuilderConfig.maxBatchSize;
    poolConfig.maxKVCacheCapacity = mBuilderConfig.maxKVCacheCapacity;
    poolConfig.maxKVPoolPages = mBuilderConfig.maxKVPoolPages;
    int32_t const prefixPoolPages = static_cast<int32_t>(poolConfig.resolvedKVPoolPages());
    int32_t const maxPrefixPagesPerSeq = rt::computeMaxPagesPerSeq(mBuilderConfig.maxKVCacheCapacity);
    if ((exportedKVPoolPages > 0 && exportedKVPoolPages != prefixPoolPages)
        || (exportedMaxPrefixPagesPerSeq > 0 && exportedMaxPrefixPagesPerSeq != maxPrefixPagesPerSeq))
    {
        LOG_ERROR("Action ONNX fixed KV dimensions do not match builder_config. Re-export with matching capacity.");
        return false;
    }
    mBuilderConfig.maxKVPoolPages = prefixPoolPages;

    nvinfer1::Dims const minPageTableDims = createDims({1, 2, maxPrefixPagesPerSeq});
    nvinfer1::Dims const optPageTableDims = createDims({mBuilderConfig.maxBatchSize, 2, maxPrefixPagesPerSeq});
    nvinfer1::Dims const maxPageTableDims = createDims({mBuilderConfig.maxBatchSize, 2, maxPrefixPagesPerSeq});
    result &= setOptimizationProfile(
        profile, binding_names::kKVPageTable, minPageTableDims, optPageTableDims, maxPageTableDims);

    nvinfer1::Dims const cacheDims = createDims({2, prefixPoolPages, rt::kTOKENS_PER_PAGE, mNumHeads, mHeadDim});

    for (int32_t i = 0; i < mNumLayers; ++i)
    {
        result &= setOptimizationProfile(
            profile, binding_names::formatKVCacheName(i, true).c_str(), cacheDims, cacheDims, cacheDims);
    }

    int32_t const scratchPagesPerSeq = rt::computeMaxPagesPerSeq(mNumDiffusionTokens);
    int64_t const scratchPageCount = mBuilderConfig.maxBatchSize * static_cast<int64_t>(scratchPagesPerSeq);
    ELLM_CHECK(scratchPageCount <= rt::kMAX_KV_POOL_PAGES, "Action scratch page count exceeds supported capacity.");
    int32_t const scratchPages = static_cast<int32_t>(scratchPageCount);
    result &= setOptimizationProfile(profile, binding_names::kActionPageTable, createDims({1, 2, scratchPagesPerSeq}),
        createDims({mBuilderConfig.maxBatchSize, 2, scratchPagesPerSeq}),
        createDims({mBuilderConfig.maxBatchSize, 2, scratchPagesPerSeq}));
    auto const scratchDims = createDims({2, scratchPages, rt::kTOKENS_PER_PAGE, mNumHeads, mHeadDim});
    for (int32_t layer = 0; layer < mNumLayers; ++layer)
    {
        result &= setOptimizationProfile(
            profile, binding_names::formatActionKVCacheName(layer).c_str(), scratchDims, scratchDims, scratchDims);
    }

    if (!result)
    {
        LOG_ERROR("Failed to setup action optimization profile");
        return false;
    }

    LOG_DEBUG("%s", printOptimizationProfile(profile, "action_profile", &network).c_str());
    config.addOptimizationProfile(profile);

    return true;
}

bool ActionBuilder::copyConfig()
{
    if (!saveConfigWithBuilderInfo(mEngineDir, mModelConfig, mBuilderConfig.toJson()))
    {
        LOG_ERROR("Failed to save config to engine directory");
        return false;
    }
    return true;
}

} // namespace builder
} // namespace trt_edgellm
