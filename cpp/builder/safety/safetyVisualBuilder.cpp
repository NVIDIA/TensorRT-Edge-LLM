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

#include "builder/safety/safetyVisualBuilder.h"

#include "builder/builderUtils.h"
#include "builder/safety/internVLConfig.h"
#include "builder/safety/remoteAutoTuningUtils.h"
#include "builder/safety/safetyEngineSerialization.h"
#include "builder/safety/safetyPluginLoader.h"
#include "builder/safety/safetyVisualShapeUtils.h"
#include "common/bindingNames.h"
#include "common/fileUtils.h"
#include "common/logger.h"
#include "common/version.h"
#include "multimodal/common/modelTypes.h"

#include <cstdlib>
#include <exception>
#include <system_error>
#include <utility>

namespace trt_edgellm::safety::builder
{
namespace
{
constexpr char const* kPREPROCESSOR_CONFIG = "preprocessor_config.json";
} // namespace

SafetyVisualBuilder::SafetyVisualBuilder(
    std::filesystem::path onnxDir, std::filesystem::path engineDir, SafetyVisualBuilderConfig config)
    : mOnnxDir(std::move(onnxDir))
    , mEngineDir(std::move(engineDir))
    , mBuilderConfig(std::move(config))
{
}

bool SafetyVisualBuilder::build() noexcept
{
    try
    {
        if (!validateSafetyAutoTuningConfig(mBuilderConfig.remoteAutoTuningConfig))
        {
            return false;
        }
        if (mBuilderConfig.profiles.empty())
        {
            LOG_ERROR("Safety visual build requires at least one fixed MSS profile.");
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
        std::unique_ptr<SafetyPluginRegistration> pluginRegistration;
        char const* const pluginEnvironment = std::getenv("EDGELLM_SAFETY_PLUGIN_PATH");
        if (!mBuilderConfig.safetyPluginLibrary.empty()
            || (pluginEnvironment != nullptr && pluginEnvironment[0] != '\0'))
        {
            pluginRegistration
                = loadSafetyPlugins(mBuilderConfig.safetyPluginLibrary, {{"trt_edgellm", "ViTAttentionPlugin", "1"}});
            if (!pluginRegistration)
            {
                return false;
            }
        }

        auto [trtBuilder, network] = trt_edgellm::builder::createBuilderAndNetwork();
        if (!trtBuilder || !network)
        {
            return false;
        }
        auto parser = trt_edgellm::builder::parseOnnxModel(network.get(), (mOnnxDir / "model.onnx").string());
        if (!parser)
        {
            return false;
        }
        if (!validateOutputTokenShape(*network))
        {
            return false;
        }
        auto trtConfig = trt_edgellm::builder::createBuilderConfig(trtBuilder.get());
        if (!trtConfig)
        {
            return false;
        }
        trtConfig->setEngineCapability(nvinfer1::EngineCapability::kSAFETY);
        if (remoteAutoTuning && !trtConfig->setRemoteAutoTuningConfig(mBuilderConfig.remoteAutoTuningConfig.c_str()))
        {
            LOG_ERROR("TensorRT rejected the Safety visual remote auto-tuning configuration.");
            return false;
        }
        if (mBuilderConfig.profilingDetailed)
        {
            trtConfig->setProfilingVerbosity(nvinfer1::ProfilingVerbosity::kDETAILED);
        }

        if (remoteAutoTuning)
        {
            LOG_INFO("Building a Safety visual engine with remote auto-tuning.");
        }
        else
        {
            LOG_INFO("Building a Safety visual engine with local GPU auto-tuning.");
        }
        LOG_DEBUG("%s", trt_edgellm::builder::printNetworkInfo(network.get(), "Safety visual").c_str());
        if (!setupOptimizationProfiles(*trtBuilder, *trtConfig, *network))
        {
            return false;
        }

        std::error_code directoryError;
        std::filesystem::create_directories(mEngineDir, directoryError);
        if (directoryError)
        {
            LOG_ERROR("Failed to create Safety visual engine directory '%s': %s", mEngineDir.string().c_str(),
                directoryError.message().c_str());
            return false;
        }
        return buildAndSerializeSafetyEngine(*trtBuilder, *network, *trtConfig, mEngineDir / "visual_safety.engine",
            [this](std::filesystem::path const& directory) { return saveArtifacts(directory); },
            {kPREPROCESSOR_CONFIG});
    }
    catch (std::exception const& exception)
    {
        LOG_ERROR("Safety visual build failed: %s", exception.what());
    }
    catch (...)
    {
        LOG_ERROR("Safety visual build failed with an unknown error.");
    }
    return false;
}

bool SafetyVisualBuilder::parseConfig() noexcept
{
    try
    {
        if (!trt_edgellm::builder::loadJsonConfig((mOnnxDir / "config.json").string(), mModelConfig))
        {
            return false;
        }
        version::checkVersion(mModelConfig.value(binding_names::kEdgellmVersion, ""));
        auto const& visionConfig = mModelConfig.at(trt_edgellm::builder::kVisionConfigKey);
        if (multimodal::stringToModelType(visionConfig.at(trt_edgellm::builder::kModelTypeKey).get<std::string>())
            != multimodal::ModelType::INTERNVL)
        {
            LOG_ERROR("The initial Safety visual builder supports InternVL3 only.");
            return false;
        }
        mNumChannels = visionConfig.at("num_channels").get<int64_t>();
        auto const& imageSize = visionConfig.at("image_size");
        if (!imageSize.is_array() || imageSize.size() != 2)
        {
            LOG_ERROR("Safety visual config requires vision_config.image_size=[height,width].");
            return false;
        }
        mTokensPerTile = getInternVLTokensPerBlock(mModelConfig);
        mImageSizeH = imageSize.at(0).get<int64_t>();
        mImageSizeW = imageSize.at(1).get<int64_t>();
        if (mNumChannels <= 0 || mImageSizeH <= 0 || mImageSizeW <= 0)
        {
            LOG_ERROR("Safety visual model configuration contains non-positive dimensions.");
            return false;
        }
        return true;
    }
    catch (std::exception const& exception)
    {
        LOG_ERROR("Failed to parse Safety visual model configuration: %s", exception.what());
        return false;
    }
}

bool SafetyVisualBuilder::setupOptimizationProfiles(
    nvinfer1::IBuilder& builder, nvinfer1::IBuilderConfig& config, nvinfer1::INetworkDefinition const& network)
{
    for (size_t index = 0; index < mBuilderConfig.profiles.size(); ++index)
    {
        int64_t const imageTokens = mBuilderConfig.profiles[index].imageTokens;
        int64_t tiles{0};
        try
        {
            tiles = getSafetyVisualTileCount(imageTokens, mTokensPerTile);
        }
        catch (std::exception const& exception)
        {
            LOG_ERROR("Invalid Safety visual MSS profile %zu: %s", index, exception.what());
            return false;
        }

        nvinfer1::IOptimizationProfile* const profile = builder.createOptimizationProfile();
        if (profile == nullptr)
        {
            LOG_ERROR("Failed to create Safety visual MSS profile %zu.", index);
            return false;
        }
        nvinfer1::Dims const dimensions
            = trt_edgellm::builder::createDims({tiles, mNumChannels, mImageSizeH, mImageSizeW});
        if (!trt_edgellm::builder::setOptimizationProfile(
                profile, binding_names::kVisualInput, dimensions, dimensions, dimensions))
        {
            return false;
        }
        LOG_DEBUG("%s",
            trt_edgellm::builder::printOptimizationProfile(
                profile, "safety_visual_profile_" + std::to_string(index), &network)
                .c_str());
        if (config.addOptimizationProfile(profile) < 0)
        {
            LOG_ERROR("TensorRT rejected Safety visual MSS profile %zu.", index);
            return false;
        }
    }
    return true;
}

bool SafetyVisualBuilder::validateOutputTokenShape(nvinfer1::INetworkDefinition& network)
{
    if (network.getNbInputs() != 1 || network.getNbOutputs() != 1 || network.getInput(0)->getDimensions().nbDims != 4
        || network.getOutput(0)->getDimensions().nbDims != 2)
    {
        LOG_ERROR("Safety InternVL requires one [tiles,C,H,W] input and one [imageTokens,hidden] output.");
        return false;
    }
    auto* const inputShape = network.addShape(*network.getInput(0));
    auto* const outputShape = network.addShape(*network.getOutput(0));
    if (inputShape == nullptr || outputShape == nullptr)
    {
        LOG_ERROR("Failed to create InternVL input/output shape tensors for token validation.");
        return false;
    }
    auto* const tiles = network.addSlice(*inputShape->getOutput(0), trt_edgellm::builder::createDims({0}),
        trt_edgellm::builder::createDims({1}), trt_edgellm::builder::createDims({1}));
    auto* const tokens = network.addSlice(*outputShape->getOutput(0), trt_edgellm::builder::createDims({0}),
        trt_edgellm::builder::createDims({1}), trt_edgellm::builder::createDims({1}));
    // TensorRT borrows constant weights until serialization; the member outlives the network.
    auto* const perTile
        = network.addConstant(trt_edgellm::builder::createDims({1}), {nvinfer1::DataType::kINT64, &mTokensPerTile, 1});
    if (tiles == nullptr || tokens == nullptr || perTile == nullptr)
    {
        LOG_ERROR("Failed to extract InternVL tile/token dimensions or create the tokens-per-tile constant.");
        return false;
    }
    auto* const expected
        = network.addElementWise(*tiles->getOutput(0), *perTile->getOutput(0), nvinfer1::ElementWiseOperation::kPROD);
    if (expected == nullptr)
    {
        LOG_ERROR("Failed to calculate the expected InternVL output token count.");
        return false;
    }
    auto* const equal = network.addElementWise(
        *tokens->getOutput(0), *expected->getOutput(0), nvinfer1::ElementWiseOperation::kEQUAL);
    if (equal == nullptr
        || network.addAssertion(*equal->getOutput(0), "InternVL ONNX tokens per tile disagree with config.json")
            == nullptr)
    {
        LOG_ERROR("Failed to add the InternVL ONNX/config token-count assertion.");
        return false;
    }
    return true;
}

bool SafetyVisualBuilder::saveArtifacts(std::filesystem::path const& directory)
{
    if (!trt_edgellm::builder::saveConfigWithBuilderInfo(directory, mModelConfig, mBuilderConfig.toJson()))
    {
        LOG_ERROR("Failed to save the Safety visual builder configuration.");
        return false;
    }

    std::filesystem::path const source = mOnnxDir / kPREPROCESSOR_CONFIG;
    std::error_code existsError;
    bool const exists = std::filesystem::exists(source, existsError);
    if (existsError)
    {
        LOG_ERROR("Failed to inspect Safety visual preprocessor config: %s", existsError.message().c_str());
        return false;
    }
    return !exists || file_io::copyFile(source.string(), (directory / source.filename()).string());
}

} // namespace trt_edgellm::safety::builder
