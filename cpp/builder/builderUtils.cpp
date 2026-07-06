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

#include "builderUtils.h"
#include "common/logger.h"
#include "common/trtUtils.h"

#include <NvOnnxParser.h>
#include <fstream>
#include <sstream>
#include <string>

namespace trt_edgellm
{
namespace builder
{

//! Create TensorRT dimensions from a vector of shape values.
//! @param shape Vector of dimension sizes
//! @return TensorRT Dims object with the specified dimensions
nvinfer1::Dims createDims(std::vector<int64_t> const& shape)
{
    if (shape.size() > nvinfer1::Dims::MAX_DIMS)
    {
        LOG_ERROR("Shape size %zu exceeds TensorRT maximum dimensions %d", shape.size(), nvinfer1::Dims::MAX_DIMS);
        throw std::invalid_argument("Shape dimensions exceed MAX_DIMS");
    }
    nvinfer1::Dims dims;
    dims.nbDims = static_cast<int32_t>(shape.size());
    for (int32_t i = 0; i < dims.nbDims; ++i)
    {
        dims.d[i] = shape[i];
    }
    return dims;
}
//! Validate optimization profile dimensions.
//! Ensures that min <= opt <= max for all dimensions and that all profiles have the same number of dimensions.
//! @param minDims Minimum dimensions for the optimization profile
//! @param optDims Optimal dimensions for the optimization profile
//! @param maxDims Maximum dimensions for the optimization profile
//! @return true if dimensions are valid, false otherwise
bool checkOptimizationProfileDims(
    nvinfer1::Dims const& minDims, nvinfer1::Dims const& optDims, nvinfer1::Dims const& maxDims) noexcept
{
    if (minDims.nbDims != optDims.nbDims || optDims.nbDims != maxDims.nbDims)
    {
        LOG_ERROR("Dimension count mismatch: minDims.nbDims=%d, optDims.nbDims=%d, maxDims.nbDims=%d", minDims.nbDims,
            optDims.nbDims, maxDims.nbDims);
        return false;
    }
    for (int32_t i = 0; i < minDims.nbDims; ++i)
    {
        if (minDims.d[i] > optDims.d[i] || optDims.d[i] > maxDims.d[i])
        {
            LOG_ERROR("Condition min <= opt <= max violated at dimension index %d, min = %d, opt = %d, max = %d", i,
                minDims.d[i], optDims.d[i], maxDims.d[i]);
            return false;
        }
    }
    return true;
}

//! Set optimization profile dimensions for a specific input.
//! Validates the dimensions and sets them on the optimization profile.
//! @param profile Optimization profile to configure
//! @param inputName Name of the input tensor
//! @param minDims Minimum dimensions for the input
//! @param optDims Optimal dimensions for the input
//! @param maxDims Maximum dimensions for the input
//! @return true if setting was successful, false otherwise
bool setOptimizationProfile(nvinfer1::IOptimizationProfile* profile, char const* inputName,
    nvinfer1::Dims const& minDims, nvinfer1::Dims const& optDims, nvinfer1::Dims const& maxDims) noexcept
{
    if (!checkOptimizationProfileDims(minDims, optDims, maxDims))
    {
        LOG_INFO("setOptimizationProfile: %s is not valid", inputName);
        return false;
    }
    return profile->setDimensions(inputName, nvinfer1::OptProfileSelector::kMIN, minDims)
        && profile->setDimensions(inputName, nvinfer1::OptProfileSelector::kOPT, optDims)
        && profile->setDimensions(inputName, nvinfer1::OptProfileSelector::kMAX, maxDims);
}
//! Print detailed information about the TensorRT network.
//! Shows input and output tensor names and shapes for debugging purposes.
//! @param network TensorRT network definition to analyze
//! @param prefix Optional prefix for the output string
//! @return Formatted string containing network information
std::string printNetworkInfo(nvinfer1::INetworkDefinition const* network, std::string const& prefix)
{
    std::ostringstream oss;
    std::string title = prefix.empty() ? "Network Information:" : prefix + " Network Information:";
    oss << title << "\n";
    oss << "  Inputs (" << network->getNbInputs() << "):\n";
    for (int32_t i = 0; i < network->getNbInputs(); ++i)
    {
        auto* input = network->getInput(i);
        auto dims = input->getDimensions();
        std::string dimStr = "(";
        for (int32_t j = 0; j < dims.nbDims; ++j)
        {
            if (j > 0)
            {
                dimStr += ", ";
            }
            dimStr += std::to_string(dims.d[j]);
        }
        dimStr += ")";
        oss << "    " << input->getName() << ": " << dimStr << "\n";
    }

    oss << "  Outputs (" << network->getNbOutputs() << "):\n";
    for (int32_t i = 0; i < network->getNbOutputs(); ++i)
    {
        auto* output = network->getOutput(i);
        auto dims = output->getDimensions();
        std::string dimStr = "(";
        for (int32_t j = 0; j < dims.nbDims; ++j)
        {
            if (j > 0)
            {
                dimStr += ", ";
            }
            dimStr += std::to_string(dims.d[j]);
        }
        dimStr += ")";
        oss << "    " << output->getName() << ": " << dimStr << "\n";
    }
    return oss.str();
}

//! Print detailed information about an optimization profile.
//! Shows the min, optimal, and max dimensions for each dynamic input in the profile.
//! @param profile Optimization profile to analyze
//! @param profileName Name of the profile for display purposes
//! @param network TensorRT network definition for input analysis
//! @return Formatted string containing optimization profile information
std::string printOptimizationProfile(nvinfer1::IOptimizationProfile const* profile, std::string const& profileName,
    nvinfer1::INetworkDefinition const* network)
{
    std::ostringstream oss;
    oss << "Optimization Profile: " << profileName << "\n";

    // Print dimensions for each dynamic input in this profile
    for (int j = 0; j < network->getNbInputs(); ++j)
    {
        auto* input = network->getInput(j);
        if (input == nullptr)
        {
            continue;
        }

        char const* inputName = input->getName();
        if (inputName == nullptr)
        {
            continue;
        }

        nvinfer1::Dims const dims = input->getDimensions();
        if (hasDynamicDims(dims))
        {
            auto minDims = profile->getDimensions(inputName, nvinfer1::OptProfileSelector::kMIN);
            auto optDims = profile->getDimensions(inputName, nvinfer1::OptProfileSelector::kOPT);
            auto maxDims = profile->getDimensions(inputName, nvinfer1::OptProfileSelector::kMAX);

            std::string minStr = "(";
            std::string optStr = "(";
            std::string maxStr = "(";

            for (int k = 0; k < minDims.nbDims; ++k)
            {
                if (k > 0)
                {
                    minStr += ", ";
                    optStr += ", ";
                    maxStr += ", ";
                }
                minStr += std::to_string(minDims.d[k]);
                optStr += std::to_string(optDims.d[k]);
                maxStr += std::to_string(maxDims.d[k]);
            }
            minStr += ")";
            optStr += ")";
            maxStr += ")";

            oss << "  " << inputName << ": MIN=" << minStr << ", OPT=" << optStr << ", MAX=" << maxStr << "\n";
        }
        else
        {
            std::string dimStr = "(";
            for (int k = 0; k < dims.nbDims; ++k)
            {
                if (k > 0)
                {
                    dimStr += ", ";
                }
                dimStr += std::to_string(dims.d[k]);
            }
            dimStr += ")";
            oss << "  " << inputName << ": " << dimStr << "\n";
        }
    }
    return oss.str();
}

std::pair<std::unique_ptr<nvinfer1::IBuilder>, std::unique_ptr<nvinfer1::INetworkDefinition>> createBuilderAndNetwork()
{
    // Create builder
    auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(gLogger));
    if (!builder)
    {
        LOG_ERROR("Failed to create TensorRT builder");
        return {nullptr, nullptr};
    }

    // Create network definition with strongly typed flag
    auto const stronglyTyped = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kSTRONGLY_TYPED);
    auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(stronglyTyped));
    if (!network)
    {
        LOG_ERROR("Failed to create TensorRT network definition");
        return {nullptr, nullptr};
    }

    return {std::move(builder), std::move(network)};
}

std::unique_ptr<nvinfer1::IBuilderConfig> createBuilderConfig(nvinfer1::IBuilder* builder)
{
    if (!builder)
    {
        LOG_ERROR("Builder is nullptr");
        return nullptr;
    }

    auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config)
    {
        LOG_ERROR("Failed to create builder config");
        return nullptr;
    }

#if (NV_TENSORRT_MAJOR >= 10 && NV_TENSORRT_MINOR >= 6) || NV_TENSORRT_MAJOR >= 11
    config->setFlag(nvinfer1::BuilderFlag::kMONITOR_MEMORY);
#endif
#if NV_TENSORRT_MAJOR >= 11 || (NV_TENSORRT_MAJOR == 10 && NV_TENSORRT_MINOR >= 3)
    config->setPreviewFeature(nvinfer1::PreviewFeature::kALIASED_PLUGIN_IO_10_03, true);
#endif

    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 16ULL << 30);
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kTACTIC_DRAM, 8ULL << 30);

    return config;
}

std::unique_ptr<nvonnxparser::IParser> parseOnnxModel(
    nvinfer1::INetworkDefinition* network, std::string const& onnxFilePath)
{
    if (!network)
    {
        LOG_ERROR("Network is nullptr");
        return nullptr;
    }

    // Create ONNX parser
    auto parser = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, gLogger));
    if (!parser)
    {
        LOG_ERROR("Failed to create ONNX parser");
        return nullptr;
    }

    // Parse ONNX model
    if (!parser->parseFromFile(onnxFilePath.c_str(), static_cast<int>(gLogger.getLevel())))
    {
        LOG_ERROR("Failed to parse ONNX file: %s", onnxFilePath.c_str());
        return nullptr;
    }

    LOG_DEBUG("Successfully parsed ONNX model: %s", onnxFilePath.c_str());
    return parser;
}

bool buildAndSerializeEngine(nvinfer1::IBuilder* builder, nvinfer1::INetworkDefinition* network,
    nvinfer1::IBuilderConfig* config, std::string const& engineFilePath)
{
    if (!builder || !network || !config)
    {
        LOG_ERROR("Invalid input: builder, network, or config is nullptr");
        return false;
    }

    // Build serialized network
    auto engine = std::unique_ptr<nvinfer1::IHostMemory>(builder->buildSerializedNetwork(*network, *config));
    if (!engine)
    {
        LOG_ERROR("Failed to build serialized engine");
        return false;
    }

    // Write to file
    std::ofstream ofs(engineFilePath, std::ios::out | std::ios::binary);
    if (!ofs)
    {
        LOG_ERROR("Failed to open file for writing: %s", engineFilePath.c_str());
        return false;
    }

    ofs.write(static_cast<char const*>(engine->data()), engine->size());
    ofs.close();

    if (!ofs)
    {
        LOG_ERROR("Failed to write engine to file: %s", engineFilePath.c_str());
        return false;
    }

    LOG_INFO("Engine saved to %s", engineFilePath.c_str());
    return true;
}

bool loadJsonConfig(std::string const& configPath, nlohmann::json& outConfig)
{
    std::ifstream configFileStream(configPath);
    if (!configFileStream.is_open())
    {
        LOG_ERROR("Failed to open config file: %s", configPath.c_str());
        return false;
    }

    try
    {
        outConfig = nlohmann::json::parse(configFileStream);
        configFileStream.close();
        return true;
    }
    catch (nlohmann::json::parse_error const& e)
    {
        LOG_ERROR("Failed to parse config file %s: %s", configPath.c_str(), e.what());
        return false;
    }
}

bool saveConfigWithBuilderInfo(
    std::filesystem::path const& engineDir, nlohmann::json const& modelConfig, nlohmann::json const& builderConfig)
{
    std::string const targetConfigPath = (engineDir / "config.json").string();

    // Create a copy of model config and add builder config
    nlohmann::json configWithBuilder = modelConfig;
    configWithBuilder["builder_config"] = builderConfig;

    // Write updated config
    std::ofstream targetConfigFile(targetConfigPath);
    if (!targetConfigFile.is_open())
    {
        LOG_ERROR("Failed to open target config file: %s", targetConfigPath.c_str());
        return false;
    }

    targetConfigFile << configWithBuilder.dump(2);
    targetConfigFile.close();

    if (!targetConfigFile)
    {
        LOG_ERROR("Failed to write config file: %s", targetConfigPath.c_str());
        return false;
    }

    LOG_INFO("Saved config with builder info to %s", targetConfigPath.c_str());
    return true;
}

} // namespace builder
} // namespace trt_edgellm
