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
#include "common/logger.h"
#include "common/trtUtils.h"
#include "common/version.h"

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

    return true;
}

bool ActionBuilder::setupActionOptimizationProfile(
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
