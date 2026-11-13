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

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "builder/safety/safetyLLMBuilder.h"

#include "builder/builderUtils.h"
#include "builder/safety/remoteAutoTuningUtils.h"
#include "builder/safety/safetyEngineSerialization.h"
#include "builder/safety/safetyLLMShapeUtils.h"
#include "builder/safety/safetyPluginLoader.h"
#include "common/bindingNames.h"
#include "common/fileUtils.h"
#include "common/logger.h"
#include "common/version.h"

#include <algorithm>
#include <exception>
#include <string>
#include <string_view>
#include <system_error>

namespace trt_edgellm::safety::builder
{
namespace
{

constexpr std::initializer_list<std::string_view> kLLM_SIDECAR_FILES{"embedding.safetensors", "tokenizer.json",
    "tokenizer_config.json", "processed_chat_template.json", "special_tokens_map.json", "generation_config.json",
    "chat_template.jinja", "chat_template.model", "chat_template.processor", binding_names::kVocabMapFileName};

struct ScopedSafetyBuildCompletion
{
    SafetyPluginRegistration& registration;
    ~ScopedSafetyBuildCompletion() noexcept
    {
        registration.completeGpuWork();
    }
};

bool hasDynamicDimension(nvinfer1::Dims const& dimensions) noexcept
{
    for (int32_t index = 0; index < dimensions.nbDims; ++index)
    {
        if (dimensions.d[index] == -1)
        {
            return true;
        }
    }
    return false;
}

} // namespace

SafetyLLMBuilder::SafetyLLMBuilder(
    std::filesystem::path onnxDir, std::filesystem::path engineDir, SafetyLLMBuilderConfig config)
    : mOnnxDir(std::move(onnxDir))
    , mEngineDir(std::move(engineDir))
    , mBuilderConfig(std::move(config))
{
}

bool SafetyLLMBuilder::build() noexcept
{
    try
    {
        if (!validateSafetyAutoTuningConfig(mBuilderConfig.remoteAutoTuningConfig))
        {
            return false;
        }
        if (mBuilderConfig.profiles.empty())
        {
            LOG_ERROR("Safety LLM build requires at least one fixed MSS profile.");
            return false;
        }
        if (!parseConfig())
        {
            return false;
        }

        ScopedTensorRTCpuOnlyMode cpuOnlyMode;
        bool const remoteAutoTuning = !mBuilderConfig.remoteAutoTuningConfig.empty();
        if (remoteAutoTuning && !cpuOnlyMode.enable())
        {
            return false;
        }

        auto pluginRegistration
            = loadSafetyPlugins(mBuilderConfig.safetyPluginLibrary, {{"trt", "AttentionPlugin", "1"}});
        if (!pluginRegistration)
        {
            return false;
        }

        if (!pluginRegistration->initializeAttentionModules(remoteAutoTuning))
        {
            return false;
        }

        auto const buildEngine = [&]() -> bool {
            auto [trtBuilder, network] = trt_edgellm::builder::createBuilderAndNetwork();
            if (!trtBuilder || !network)
            {
                return false;
            }

            std::string const onnxPath = (mOnnxDir / "model.onnx").string();
            auto parser = trt_edgellm::builder::parseOnnxModel(network.get(), onnxPath);
            if (!parser)
            {
                return false;
            }

            auto trtConfig = trt_edgellm::builder::createBuilderConfig(trtBuilder.get());
            if (!trtConfig)
            {
                return false;
            }
            // Complete profiling before TRT destroys its workspace and borrowing plugins.
            ScopedSafetyBuildCompletion completion{*pluginRegistration};
            trtConfig->setEngineCapability(nvinfer1::EngineCapability::kSAFETY);
            if (remoteAutoTuning
                && !trtConfig->setRemoteAutoTuningConfig(mBuilderConfig.remoteAutoTuningConfig.c_str()))
            {
                LOG_ERROR("TensorRT rejected the Safety LLM remote auto-tuning configuration.");
                return false;
            }
            if (mBuilderConfig.profilingDetailed)
            {
                trtConfig->setProfilingVerbosity(nvinfer1::ProfilingVerbosity::kDETAILED);
            }

            if (remoteAutoTuning)
            {
                LOG_INFO("Building a Safety LLM engine with remote auto-tuning.");
            }
            else
            {
                LOG_INFO("Building a Safety LLM engine with local GPU auto-tuning.");
            }
            LOG_DEBUG("%s", trt_edgellm::builder::printNetworkInfo(network.get(), "Safety LLM").c_str());
            if (!setupOptimizationProfiles(*trtBuilder, *trtConfig, *network))
            {
                return false;
            }

            std::error_code directoryError;
            std::filesystem::create_directories(mEngineDir, directoryError);
            if (directoryError)
            {
                LOG_ERROR("Failed to create Safety LLM engine directory '%s': %s", mEngineDir.string().c_str(),
                    directoryError.message().c_str());
                return false;
            }
            return buildAndSerializeSafetyEngine(
                *trtBuilder, *network, *trtConfig, mEngineDir / "llm_safety.engine",
                [this](std::filesystem::path const& directory) { return saveArtifacts(directory); },
                kLLM_SIDECAR_FILES);
        };
        bool const built = buildEngine();
        bool const released = pluginRegistration->finishBuild();
        return built && released;
    }
    catch (std::exception const& exception)
    {
        LOG_ERROR("Safety LLM build failed: %s", exception.what());
    }
    catch (...)
    {
        LOG_ERROR("Safety LLM build failed with an unknown error.");
    }
    return false;
}

bool SafetyLLMBuilder::parseConfig() noexcept
{
    try
    {
        if (!trt_edgellm::builder::loadJsonConfig((mOnnxDir / "config.json").string(), mModelConfig))
        {
            return false;
        }
        version::checkVersion(mModelConfig.value(binding_names::kEdgellmVersion, ""));
        if (mModelConfig.value("trt_native_ops", false))
        {
            LOG_ERROR("The initial Safety LLM builder requires the admitted plugin attention path.");
            return false;
        }
        if (mModelConfig.value("num_linear_attn_layers", 0) != 0)
        {
            LOG_ERROR("The initial Safety LLM builder does not support recurrent-state models.");
            return false;
        }

        mHiddenSize = mModelConfig.at("hidden_size").get<int64_t>();
        mNumKVHeads = mModelConfig.at("num_key_value_heads").get<int64_t>();
        int64_t const numAttentionHeads = mModelConfig.at("num_attention_heads").get<int64_t>();
        if (numAttentionHeads <= 0)
        {
            LOG_ERROR("Safety LLM model configuration contains a non-positive attention-head count.");
            return false;
        }
        mHeadSize = mModelConfig.value("head_dim", mHiddenSize / numAttentionHeads);
        mRotaryDim = mModelConfig.contains("partial_rotary_factor")
            ? static_cast<int64_t>(mModelConfig.at("partial_rotary_factor").get<double>() * mHeadSize)
            : mHeadSize;
        if (mHiddenSize <= 0 || mNumKVHeads <= 0 || mHeadSize <= 0 || mRotaryDim <= 0)
        {
            LOG_ERROR("Safety LLM model configuration contains non-positive dimensions.");
            return false;
        }
        return true;
    }
    catch (std::exception const& exception)
    {
        LOG_ERROR("Failed to parse Safety LLM model configuration: %s", exception.what());
        return false;
    }
}

bool SafetyLLMBuilder::setupOptimizationProfiles(
    nvinfer1::IBuilder& builder, nvinfer1::IBuilderConfig& config, nvinfer1::INetworkDefinition const& network)
{
    for (size_t index = 0; index < mBuilderConfig.profiles.size(); ++index)
    {
        nvinfer1::IOptimizationProfile* const profile = builder.createOptimizationProfile();
        if (profile == nullptr)
        {
            LOG_ERROR("Failed to create Safety LLM MSS profile %zu.", index);
            return false;
        }
        if (!setupStaticProfile(*profile, mBuilderConfig.profiles[index], network))
        {
            LOG_ERROR("Failed to configure Safety LLM MSS profile %zu.", index);
            return false;
        }
        LOG_DEBUG("%s",
            trt_edgellm::builder::printOptimizationProfile(
                profile, "safety_llm_profile_" + std::to_string(index), &network)
                .c_str());
        if (config.addOptimizationProfile(profile) < 0)
        {
            LOG_ERROR("TensorRT rejected Safety LLM MSS profile %zu.", index);
            return false;
        }
    }
    return true;
}

bool SafetyLLMBuilder::setupStaticProfile(nvinfer1::IOptimizationProfile& profile,
    SafetyLLMStaticShapeProfile const& shapeProfile, nvinfer1::INetworkDefinition const& network) const
{
    if (mBuilderConfig.isVlm && shapeProfile.imageTokens <= 0)
    {
        LOG_ERROR("Safety VLM MSS profile requires positive imageTokens.");
        return false;
    }

    for (int32_t inputIndex = 0; inputIndex < network.getNbInputs(); ++inputIndex)
    {
        nvinfer1::ITensor const* const input = network.getInput(inputIndex);
        if (input == nullptr || input->getName() == nullptr)
        {
            LOG_ERROR("Safety LLM network input %d has no name.", inputIndex);
            return false;
        }
        std::string const name{input->getName()};
        auto const exportedShape = input->getDimensions();
        auto const dimensions = getSafetyLLMInputShape(
            name, exportedShape, shapeProfile, mHiddenSize, mNumKVHeads, mHeadSize, mRotaryDim);
        if (!hasDynamicDimension(exportedShape))
        {
            continue;
        }

        if (!trt_edgellm::builder::setOptimizationProfile(&profile, name.c_str(), dimensions, dimensions, dimensions))
        {
            return false;
        }
    }
    return true;
}

bool SafetyLLMBuilder::saveArtifacts(std::filesystem::path const& directory)
{
    if (!trt_edgellm::builder::saveConfigWithBuilderInfo(directory, mModelConfig, mBuilderConfig.toJson()))
    {
        LOG_ERROR("Failed to save the Safety LLM builder configuration.");
        return false;
    }

    for (auto const& filename : kLLM_SIDECAR_FILES)
    {
        std::filesystem::path const source = mOnnxDir / filename;
        std::error_code existsError;
        bool const exists = std::filesystem::exists(source, existsError);
        if (existsError)
        {
            LOG_ERROR("Failed to inspect optional Safety LLM artifact '%s': %s", source.string().c_str(),
                existsError.message().c_str());
            return false;
        }
        if (exists && !file_io::copyFile(source.string(), (directory / filename).string()))
        {
            return false;
        }
    }
    return true;
}

} // namespace trt_edgellm::safety::builder
