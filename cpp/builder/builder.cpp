/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "builder.h"
#include "common/bindingNames.h"
#include "common/cudaUtils.h"
#include "common/fileUtils.h"
#include "common/logger.h"
#include "common/trtUtils.h"
#include "common/version.h"

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <fstream>
#include <iostream>
#include <sstream>

using namespace trt_edgellm;

namespace trt_edgellm
{
namespace builder
{

namespace
{
//! Utility functions for TensorRT engine building
//! JSON key names used in model configs
constexpr char kVisionConfigKey[] = "vision_config";
constexpr char kModelTypeKey[] = "model_type";
constexpr char kEmbdLayerKey[] = "embd_layer";
constexpr char kImageEmbdLayerKey[] = "image_embd_layer";

//! Create TensorRT dimensions from a vector of shape values.
//! @param shape Vector of dimension sizes
//! @return TensorRT Dims object with the specified dimensions
nvinfer1::Dims createDims(std::vector<int64_t> const& shape)
{
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
    nvinfer1::Dims const& minDims, nvinfer1::Dims const& optDims, nvinfer1::Dims const& maxDims)
{
    if (minDims.nbDims != optDims.nbDims || optDims.nbDims != maxDims.nbDims)
    {
        LOG_ERROR("Dimension count mismatch: minDims.nbDims=%d, optDims.nbDims=%d, maxDims.nbDims=%d", minDims.nbDims,
            optDims.nbDims, maxDims.nbDims);
        return false;
    }
    for (int i = 0; i < minDims.nbDims; ++i)
    {
        if (minDims.d[i] > optDims.d[i] || optDims.d[i] > maxDims.d[i])
        {
            LOG_ERROR("Dimension value mismatch at index %d: min=%d, opt=%d, max=%d", i, minDims.d[i], optDims.d[i],
                maxDims.d[i]);
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
    nvinfer1::Dims const& minDims, nvinfer1::Dims const& optDims, nvinfer1::Dims const& maxDims)
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
std::string printNetworkInfo(nvinfer1::INetworkDefinition const* network, std::string const& prefix = "")
{
    std::ostringstream oss;
    std::string title = prefix.empty() ? "Network Information:" : prefix + " Network Information:";
    oss << title << "\n";
    oss << "  Inputs (" << network->getNbInputs() << "):\n";
    for (int i = 0; i < network->getNbInputs(); ++i)
    {
        auto* input = network->getInput(i);
        auto dims = input->getDimensions();
        std::string dimStr = "(";
        for (int j = 0; j < dims.nbDims; ++j)
        {
            if (j > 0)
                dimStr += ", ";
            dimStr += std::to_string(dims.d[j]);
        }
        dimStr += ")";
        oss << "    " << input->getName() << ": " << dimStr << "\n";
    }

    oss << "  Outputs (" << network->getNbOutputs() << "):\n";
    for (int i = 0; i < network->getNbOutputs(); ++i)
    {
        auto* output = network->getOutput(i);
        auto dims = output->getDimensions();
        std::string dimStr = "(";
        for (int j = 0; j < dims.nbDims; ++j)
        {
            if (j > 0)
                dimStr += ", ";
            dimStr += std::to_string(dims.d[j]);
        }
        dimStr += ")";
        oss << "    " << output->getName() << ": " << dimStr << "\n";
    }
    return oss.str();
}

//! Print detailed information about an optimization profile.
//! Shows the min, optimal, and max dimensions for each input in the profile.
//! @param profile Optimization profile to analyze
//! @param profileName Name of the profile for display purposes
//! @param network TensorRT network definition for input analysis
//! @return Formatted string containing optimization profile information
std::string printOptimizationProfile(nvinfer1::IOptimizationProfile const* profile, std::string const& profileName,
    nvinfer1::INetworkDefinition const* network)
{
    std::ostringstream oss;
    oss << "Optimization Profile: " << profileName << "\n";

    // Print dimensions for each input in this profile
    for (int j = 0; j < network->getNbInputs(); ++j)
    {
        char const* inputName = network->getInput(j)->getName();
        if (inputName != nullptr)
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
    }
    return oss.str();
}
} // namespace

// LLMBuilder implementation

LLMBuilder::LLMBuilder(
    std::filesystem::path const& onnxDir, std::filesystem::path const& engineDir, LLMBuilderConfig const& config)
    : mOnnxDir(onnxDir)
    , mEngineDir(engineDir)
    , mBuilderConfig(config)
{
}

bool LLMBuilder::build()
{
    // Load plugin library
    auto pluginHandles = loadEdgellmPluginLib();

    // Parse model config
    if (!parseConfig())
    {
        return false;
    }

    // Create builder
    auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(gLogger));
    if (!builder)
    {
        LOG_ERROR("Failed to create builder.");
        return false;
    }

    // Create network definition
    auto const stronglyTyped = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kSTRONGLY_TYPED);
    auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(stronglyTyped));
    if (!network)
    {
        LOG_ERROR("Failed to create network.");
        return false;
    }

    // Create ONNX parser
    auto parser = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, gLogger));
    if (!parser)
    {
        LOG_ERROR("Failed to create parser.");
        return false;
    }

    // Parse ONNX model
    std::string onnxFilePath;
    if (mBuilderConfig.maxLoraRank > 0)
    {
        onnxFilePath = mOnnxDir.string() + "/lora_model.onnx";
        LOG_INFO("Parsing LoRA-enabled ONNX model. Please ensure %s exists.", onnxFilePath.c_str());
    }
    else
    {
        onnxFilePath = mOnnxDir.string() + "/model.onnx";
        LOG_INFO("Parsing ONNX model. Please ensure %s exists.", onnxFilePath.c_str());
    }
    if (!parser->parseFromFile(onnxFilePath.c_str(), static_cast<int>(gLogger.getLevel())))
    {
        LOG_ERROR("Failed to parse ONNX file: %s", onnxFilePath.c_str());
        return false;
    }

    // Print network information
    LOG_DEBUG("%s", printNetworkInfo(network.get(), "LLM").c_str());

    // Create builder config
    auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    config->setFlag(nvinfer1::BuilderFlag::kMONITOR_MEMORY);
    if (!config)
    {
        LOG_ERROR("Failed to create builder config.");
        return false;
    }

    // Setup optimization profiles
    if (!setupLLMOptimizationProfiles(builder.get(), config.get(), network.get()))
    {
        return false;
    }

    // Build engine
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
    if (mBuilderConfig.eagleDraft)
    {
        engineFileName = "eagle_draft.engine";
    }
    else if (mBuilderConfig.eagleBase)
    {
        engineFileName = "eagle_base.engine";
    }
    else
    {
        engineFileName = "llm.engine";
    }
    std::string engineFilePath = mEngineDir.string() + "/" + engineFileName;
    auto engine = builder->buildSerializedNetwork(*network, *config);

    if (!engine)
    {
        LOG_ERROR("Failed to build engine.");
        return false;
    }
    std::ofstream ofs(engineFilePath, std::ios::out | std::ios::binary);
    if (!ofs)
    {
        LOG_ERROR("Failed to open file for writing: %s", engineFilePath.c_str());
        return false;
    }
    ofs.write(static_cast<char*>(engine->data()), engine->size());
    ofs.close();
    LOG_INFO("Engine saved to %s", engineFilePath.c_str());

    // Copy files and save builder config
    // Copy config.json with builder config
    if (!copyConfig())
    {
        return false;
    }

    // Copy tokenizer files
    if (!copyTokenizerFiles())
    {
        return false;
    }

    // Copy Eagle-specific files
    if (!copyEagleFiles())
    {
        return false;
    }

    // Copy vocabulary mapping files
    if (!copyVocabMappingFiles())
    {
        return false;
    }

    return true;
}

bool LLMBuilder::parseConfig()
{
    std::string jsonPath = mOnnxDir.string() + "/config.json";
    std::ifstream configFileStream(jsonPath);
    if (!configFileStream.is_open())
    {
        LOG_ERROR("Failed to open config file: %s", jsonPath.c_str());
        return false;
    }

    try
    {
        mModelConfig = Json::parse(configFileStream);
    }
    catch (Json::parse_error const& e)
    {
        LOG_ERROR("Failed to parse config file: %s", e.what());
        return false;
    }

    // Check model version
    std::string modelVersion = mModelConfig.value(binding_names::kEdgellmVersion, "");
    version::checkVersion(modelVersion);

    mHiddenSize = mModelConfig["hidden_size"].get<int32_t>();
    mTargetModelOutputHiddenDim = mHiddenSize * 3;
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

    if (mModelConfig.contains("partial_rotary_factor"))
    {
        mRotaryDim = static_cast<int64_t>(mModelConfig["partial_rotary_factor"].get<float>() * mHeadSize);
    }
    else
    {
        mRotaryDim = mHeadSize;
    }

    mNbKVCacheInputs = mModelConfig["num_hidden_layers"].get<int32_t>();

    return true;
}

bool LLMBuilder::setupLLMOptimizationProfiles(
    nvinfer1::IBuilder* builder, nvinfer1::IBuilderConfig* config, nvinfer1::INetworkDefinition const* network)
{
    auto* contextProfile = builder->createOptimizationProfile();
    auto* generationProfile = builder->createOptimizationProfile();

    bool result = true;

    // Setup common profiles
    result &= setupCommonProfiles(contextProfile, generationProfile);

    // Setup model-specific profiles
    if (mBuilderConfig.eagleBase || mBuilderConfig.eagleDraft)
    {
        result &= setupEagleProfiles(contextProfile, generationProfile);
    }
    else
    {
        result &= setupVanillaProfiles(contextProfile, generationProfile);
    }

    if (mBuilderConfig.isVlm)
    {
        result &= setupVLMProfiles(contextProfile, generationProfile, network);
    }

    if (mBuilderConfig.maxLoraRank > 0)
    {
        result &= setupLoraProfiles(contextProfile, generationProfile, network);
    }

    if (!result)
    {
        LOG_ERROR("Failed to setup optimization profiles");
        return false;
    }

    LOG_DEBUG("%s", printOptimizationProfile(contextProfile, "context_profile", network).c_str());
    LOG_DEBUG("%s", printOptimizationProfile(generationProfile, "generation_profile", network).c_str());

    config->addOptimizationProfile(contextProfile);
    config->addOptimizationProfile(generationProfile);

    return true;
}

bool LLMBuilder::setupCommonProfiles(
    nvinfer1::IOptimizationProfile* const contextProfile, nvinfer1::IOptimizationProfile* const generationProfile)
{
    bool result = true;

    // Context lengths
    result &= setOptimizationProfile(contextProfile, binding_names::kContextLengths, createDims({1}),
        createDims({mBuilderConfig.maxBatchSize}), createDims({mBuilderConfig.maxBatchSize}));
    result &= setOptimizationProfile(generationProfile, binding_names::kContextLengths, createDims({1}),
        createDims({mBuilderConfig.maxBatchSize}), createDims({mBuilderConfig.maxBatchSize}));

    // Rope rotary cos sin
    result &= setOptimizationProfile(contextProfile, binding_names::kRopeCosSin,
        createDims({1, mBuilderConfig.maxKVCacheCapacity, mRotaryDim}),
        createDims({mBuilderConfig.maxBatchSize, mBuilderConfig.maxKVCacheCapacity, mRotaryDim}),
        createDims({mBuilderConfig.maxBatchSize, mBuilderConfig.maxKVCacheCapacity, mRotaryDim}));
    result &= setOptimizationProfile(generationProfile, binding_names::kRopeCosSin,
        createDims({1, mBuilderConfig.maxKVCacheCapacity, mRotaryDim}),
        createDims({mBuilderConfig.maxBatchSize, mBuilderConfig.maxKVCacheCapacity, mRotaryDim}),
        createDims({mBuilderConfig.maxBatchSize, mBuilderConfig.maxKVCacheCapacity, mRotaryDim}));

    // For KVCacheStartIndex, we use zero shape to indicate the kvcache is empty for all sequences in the batch.
    // This can help distinguish the normal prefill and chunked prefill execution.
    result &= setOptimizationProfile(contextProfile, binding_names::kKVCacheStartIndex, createDims({0}),
        createDims({mBuilderConfig.maxBatchSize}), createDims({mBuilderConfig.maxBatchSize}));
    result &= setOptimizationProfile(generationProfile, binding_names::kKVCacheStartIndex, createDims({1}),
        createDims({mBuilderConfig.maxBatchSize}), createDims({mBuilderConfig.maxBatchSize}));

    // KV cache profiles
    result &= setupKVCacheProfiles(contextProfile, generationProfile);

    return result;
}

bool LLMBuilder::setupVanillaProfiles(
    nvinfer1::IOptimizationProfile* const contextProfile, nvinfer1::IOptimizationProfile* const generationProfile)
{
    bool result = true;

    // Input IDs - always dynamic
    result &= setOptimizationProfile(contextProfile, binding_names::kInputIds, createDims({1, 1}),
        createDims({mBuilderConfig.maxBatchSize, mBuilderConfig.maxInputLen / 2}),
        createDims({mBuilderConfig.maxBatchSize, mBuilderConfig.maxInputLen}));
    result &= setOptimizationProfile(generationProfile, binding_names::kInputIds, createDims({1, 1}),
        createDims({mBuilderConfig.maxBatchSize, 1}), createDims({mBuilderConfig.maxBatchSize, 1}));

    // Last token IDs
    result &= setOptimizationProfile(contextProfile, binding_names::kLastTokenIds, createDims({1, 1}),
        createDims({mBuilderConfig.maxBatchSize, 1}), createDims({mBuilderConfig.maxBatchSize, 1}));
    result &= setOptimizationProfile(generationProfile, binding_names::kLastTokenIds, createDims({1, 1}),
        createDims({mBuilderConfig.maxBatchSize, 1}), createDims({mBuilderConfig.maxBatchSize, 1}));

    return result;
}

bool LLMBuilder::setupEagleProfiles(
    nvinfer1::IOptimizationProfile* const contextProfile, nvinfer1::IOptimizationProfile* const generationProfile)
{
    bool result = true;

    int const maxTokens
        = mBuilderConfig.eagleDraft ? mBuilderConfig.maxDraftTreeSize : mBuilderConfig.maxVerifyTreeSize;

    // Input IDs
    result &= setOptimizationProfile(contextProfile, binding_names::kInputIds, createDims({1, 1}),
        createDims({mBuilderConfig.maxBatchSize, mBuilderConfig.maxInputLen / 2}),
        createDims({mBuilderConfig.maxBatchSize, mBuilderConfig.maxInputLen}));
    result &= setOptimizationProfile(generationProfile, binding_names::kInputIds, createDims({1, 1}),
        createDims({mBuilderConfig.maxBatchSize, maxTokens / 2}), createDims({mBuilderConfig.maxBatchSize, maxTokens}));

    // Last token IDs - 2D shape [batch_size, num_selected_tokens]
    result &= setOptimizationProfile(contextProfile, binding_names::kLastTokenIds, createDims({1, 1}),
        createDims({mBuilderConfig.maxBatchSize, 1}), createDims({mBuilderConfig.maxBatchSize, 1}));
    result &= setOptimizationProfile(generationProfile, binding_names::kLastTokenIds, createDims({1, 1}),
        createDims({mBuilderConfig.maxBatchSize, maxTokens / 2}), createDims({mBuilderConfig.maxBatchSize, maxTokens}));

    if (mBuilderConfig.eagleDraft)
    {
        // Hidden states from draft
        result &= setOptimizationProfile(contextProfile, binding_names::kDraftModelHiddenStates,
            createDims({1, 1, mHiddenSize}),
            createDims({mBuilderConfig.maxBatchSize, mBuilderConfig.maxInputLen / 2, mHiddenSize}),
            createDims({mBuilderConfig.maxBatchSize, mBuilderConfig.maxInputLen, mHiddenSize}));
        result &= setOptimizationProfile(generationProfile, binding_names::kDraftModelHiddenStates,
            createDims({1, 1, mHiddenSize}), createDims({mBuilderConfig.maxBatchSize, maxTokens / 2, mHiddenSize}),
            createDims({mBuilderConfig.maxBatchSize, maxTokens, mHiddenSize}));

        // Hidden states input
        result &= setOptimizationProfile(contextProfile, binding_names::kBaseModelHiddenStates,
            createDims({1, 1, mTargetModelOutputHiddenDim}),
            createDims({mBuilderConfig.maxBatchSize, mBuilderConfig.maxInputLen / 2, mTargetModelOutputHiddenDim}),
            createDims({mBuilderConfig.maxBatchSize, mBuilderConfig.maxInputLen, mTargetModelOutputHiddenDim}));
        result &= setOptimizationProfile(generationProfile, binding_names::kBaseModelHiddenStates,
            createDims({1, 1, mTargetModelOutputHiddenDim}),
            createDims({mBuilderConfig.maxBatchSize, maxTokens / 2, mTargetModelOutputHiddenDim}),
            createDims({mBuilderConfig.maxBatchSize, maxTokens, mTargetModelOutputHiddenDim}));
    }

    // Attention mask and position ID
    if (mBuilderConfig.eagleDraft || mBuilderConfig.eagleBase)
    {
        int32_t const attnMaskAlignSize = 32;
        result &= setOptimizationProfile(contextProfile, binding_names::kAttentionMask, createDims({1, 1, 1}),
            createDims({mBuilderConfig.maxBatchSize, 1, 1}), createDims({mBuilderConfig.maxBatchSize, 1, 1}));
        result &= setOptimizationProfile(generationProfile, binding_names::kAttentionMask, createDims({1, 1, 1}),
            createDims({mBuilderConfig.maxBatchSize, maxTokens / 2,
                static_cast<int64_t>(divUp(maxTokens / 2, attnMaskAlignSize) * attnMaskAlignSize)}),
            createDims({mBuilderConfig.maxBatchSize, maxTokens,
                static_cast<int64_t>(divUp(maxTokens, attnMaskAlignSize) * attnMaskAlignSize)}));

        result &= setOptimizationProfile(contextProfile, binding_names::kAttentionPosId, createDims({1, 1}),
            createDims({mBuilderConfig.maxBatchSize, 1}), createDims({mBuilderConfig.maxBatchSize, 1}));
        result &= setOptimizationProfile(generationProfile, binding_names::kAttentionPosId, createDims({1, 1}),
            createDims({mBuilderConfig.maxBatchSize, maxTokens / 2}),
            createDims({mBuilderConfig.maxBatchSize, maxTokens}));
    }

    return result;
}

bool LLMBuilder::setupVLMProfiles(nvinfer1::IOptimizationProfile* contextProfile,
    nvinfer1::IOptimizationProfile* generationProfile, nvinfer1::INetworkDefinition const* network)
{
    bool result = true;

    // Find image_embeds input
    int32_t imageHiddenSize = 0;
    for (int32_t idx = 0; idx < network->getNbInputs(); idx++)
    {
        if (strcmp(network->getInput(idx)->getName(), binding_names::kImageEmbeds) == 0)
        {
            imageHiddenSize = network->getInput(idx)->getDimensions().d[1];
            break;
        }
    }

    if (imageHiddenSize == 0)
    {
        LOG_ERROR("Please add image_embeds as inputs for VLM.");
        return false;
    }

    int64_t optImageTokens = (mBuilderConfig.maxImageTokens + mBuilderConfig.minImageTokens) / 2;

    result &= setOptimizationProfile(contextProfile, binding_names::kImageEmbeds,
        createDims({mBuilderConfig.minImageTokens, imageHiddenSize}), createDims({optImageTokens, imageHiddenSize}),
        createDims({mBuilderConfig.maxImageTokens, imageHiddenSize}));
    result &= setOptimizationProfile(generationProfile, binding_names::kImageEmbeds, createDims({1, imageHiddenSize}),
        createDims({1, imageHiddenSize}), createDims({1, imageHiddenSize}));

    if (mModelConfig["model"].get<std::string>() == "qwen3vltext")
    {
        for (int32_t idx = 0; idx < network->getNbInputs(); idx++)
        {
            std::string const inputName = network->getInput(idx)->getName();
            if (inputName.find(binding_names::kDeepstackFeaturesTemplate) != std::string::npos)
            {
                result &= setOptimizationProfile(contextProfile, inputName.c_str(),
                    createDims({mBuilderConfig.minImageTokens, imageHiddenSize}),
                    createDims({optImageTokens, imageHiddenSize}),
                    createDims({mBuilderConfig.maxImageTokens, imageHiddenSize}));
                result &= setOptimizationProfile(generationProfile, inputName.c_str(), createDims({1, imageHiddenSize}),
                    createDims({1, imageHiddenSize}), createDims({1, imageHiddenSize}));
            }
        }
    }

    if (!result)
    {
        LOG_ERROR("Failed to setup optimization profiles at setupVLMProfiles().");
    }

    return result;
}

bool LLMBuilder::setupLoraProfiles(nvinfer1::IOptimizationProfile* contextProfile,
    nvinfer1::IOptimizationProfile* generationProfile, nvinfer1::INetworkDefinition const* network)
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

    for (int i = 0; i < network->getNbInputs(); ++i)
    {
        auto* input = network->getInput(i);
        std::string inputName = input->getName();

        if (inputName.find(binding_names::kLoraAPrefix) != std::string::npos)
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
                result
                    &= setOptimizationProfile(contextProfile, inputName.c_str(), createDims({gemm_k, 0}), // min shape
                        createDims({gemm_k, mBuilderConfig.maxLoraRank / 2}),                             // opt shape
                        createDims({gemm_k, mBuilderConfig.maxLoraRank}));                                // max shape
                result &= setOptimizationProfile(generationProfile, inputName.c_str(),
                    createDims({gemm_k, 0}),                              // min shape
                    createDims({gemm_k, mBuilderConfig.maxLoraRank / 2}), // opt shape
                    createDims({gemm_k, mBuilderConfig.maxLoraRank}));    // max shape
            }
        }
        else if (inputName.find(binding_names::kLoraBPrefix) != std::string::npos)
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
                result
                    &= setOptimizationProfile(contextProfile, inputName.c_str(), createDims({0, gemm_n}), // min shape
                        createDims({mBuilderConfig.maxLoraRank / 2, gemm_n}),                             // opt shape
                        createDims({mBuilderConfig.maxLoraRank, gemm_n}));                                // max shape
                result &= setOptimizationProfile(generationProfile, inputName.c_str(),
                    createDims({0, gemm_n}),                              // min shape
                    createDims({mBuilderConfig.maxLoraRank / 2, gemm_n}), // opt shape
                    createDims({mBuilderConfig.maxLoraRank, gemm_n}));    // max shape
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
    nvinfer1::IOptimizationProfile* const contextProfile, nvinfer1::IOptimizationProfile* const generationProfile)
{
    bool result = true;
    // KV cache shape is [B, 2, num_kv_heads, 0 to max_kv_cache_capacity, head_dim]
    nvinfer1::Dims minKVCacheShape = createDims({1, 2, mNumKVHeads, 0, mHeadSize});
    nvinfer1::Dims optKVCacheShape
        = createDims({mBuilderConfig.maxBatchSize, 2, mNumKVHeads, mBuilderConfig.maxKVCacheCapacity, mHeadSize});
    nvinfer1::Dims maxKVCacheShape
        = createDims({mBuilderConfig.maxBatchSize, 2, mNumKVHeads, mBuilderConfig.maxKVCacheCapacity, mHeadSize});

    for (int i = 0; i < mNbKVCacheInputs; ++i)
    {
        result &= setOptimizationProfile(contextProfile, binding_names::formatKVCacheName(i, true).c_str(),
            minKVCacheShape, optKVCacheShape, maxKVCacheShape);
        result &= setOptimizationProfile(generationProfile, binding_names::formatKVCacheName(i, true).c_str(),
            minKVCacheShape, optKVCacheShape, maxKVCacheShape);
    }

    return result;
}

bool LLMBuilder::copyConfig()
{
    // Determine config file name based on model type
    std::string targetConfigPath;
    if (mBuilderConfig.eagleDraft)
    {
        targetConfigPath = mEngineDir.string() + "/draft_config.json";
    }
    else if (mBuilderConfig.eagleBase)
    {
        targetConfigPath = mEngineDir.string() + "/base_config.json";
    }
    else
    {
        targetConfigPath = mEngineDir.string() + "/config.json";
    }

    // Create a copy of mModelConfig and add builder config
    Json configWithBuilder = mModelConfig;
    configWithBuilder["builder_config"] = mBuilderConfig.toJson();

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
    // Eagle3 draft model does not need tokenizer files
    if (mBuilderConfig.eagleDraft)
    {
        return true;
    }

    std::vector<std::string> tokenizerFiles
        = {"tokenizer_config.json", "tokenizer.json", "processed_chat_template.json"};
    bool allSuccess = true;

    for (auto const& filename : tokenizerFiles)
    {
        std::string srcPath = mOnnxDir.string() + "/" + filename;
        std::string dstPath = mEngineDir.string() + "/" + filename;

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
    // Copy d2t.safetensors for Eagle3 draft models
    if (mBuilderConfig.eagleDraft)
    {
        std::string d2tPath = mOnnxDir.string() + "/d2t.safetensors";
        std::string targetD2tPath = mEngineDir.string() + "/d2t.safetensors";

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
    // Copy vocab_map.safetensors if reduced vocabulary is used
    if (mModelConfig.contains(binding_names::kReducedVocabSizeKey)
        && mModelConfig[binding_names::kReducedVocabSizeKey].get<int32_t>() > 0)
    {
        std::string vocabMapPath = mOnnxDir.string() + "/" + binding_names::kVocabMapFileName;
        std::string targetVocabMapPath = mEngineDir.string() + "/" + binding_names::kVocabMapFileName;

        if (file_io::copyFile(vocabMapPath, targetVocabMapPath))
        {
            LOG_INFO("Copied %s to %s", binding_names::kVocabMapFileName, targetVocabMapPath.c_str());
        }
        else
        {
            LOG_WARNING("%s not found in %s. This is expected if reduced vocabulary is not used.",
                binding_names::kVocabMapFileName, mOnnxDir.string().c_str());
        }
    }

    return true;
}

// VisualBuilder implementation

VisualBuilder::VisualBuilder(
    std::filesystem::path const& onnxDir, std::filesystem::path const& engineDir, VisualBuilderConfig const& config)
    : mOnnxDir(onnxDir)
    , mEngineDir(engineDir)
    , mBuilderConfig(config)
{
}

bool VisualBuilder::build()
{
    // Load plugin library
    auto pluginHandles = loadEdgellmPluginLib();

    // Parse model config
    if (!parseConfig())
    {
        return false;
    }

    // Create builder
    auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(gLogger));
    if (!builder)
    {
        LOG_ERROR("Failed to create builder.");
        return false;
    }

    // Create network definition
    auto const stronglyTyped = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kSTRONGLY_TYPED);
    auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(stronglyTyped));
    if (!network)
    {
        LOG_ERROR("Failed to create network.");
        return false;
    }

    // Create ONNX parser
    auto parser = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, gLogger));
    if (!parser)
    {
        LOG_ERROR("Failed to create parser.");
        return false;
    }

    // Parse ONNX model
    std::string onnxPath = mOnnxDir.string() + "/model.onnx";
    if (!parser->parseFromFile(onnxPath.c_str(), static_cast<int>(gLogger.getLevel())))
    {
        LOG_ERROR("Failed to parse ONNX file: %s", onnxPath.c_str());
        return false;
    }

    // Print network information
    LOG_DEBUG("%s", printNetworkInfo(network.get(), "Visual").c_str());

    // Create builder config
    auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    config->setFlag(nvinfer1::BuilderFlag::kMONITOR_MEMORY);
    if (!config)
    {
        LOG_ERROR("Failed to create builder config.");
        return false;
    }

    // Setup optimization profile
    if (!setupVisualOptimizationProfile(builder.get(), config.get(), network.get()))
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
        LOG_INFO("Created directory %s for saving Visual engine.", mEngineDir.string().c_str());
        if (mModelType == multimodal::ModelType::PHI4MM)
        {
            // Copy Phi-4MM GN(Grid Newline) projection weights to engine directory for runtime loading
            // GN serves as line separator
            std::string src = mOnnxDir.string() + "/phi4mm_gn_proj.safetensors";
            std::string dst = mEngineDir.string() + "/phi4mm_gn_proj.safetensors";
            if (file_io::copyFile(src, dst))
            {
                LOG_INFO("Copied Phi4MM GN projection weights to %s", dst.c_str());
            }
            else
            {
                LOG_ERROR("Failed to copy Phi4MM GN projection weights to %s", dst.c_str());
                return false;
            }
        }
    }

    // Save engine
    std::string engineFilePath = mEngineDir.string() + "/visual.engine";
    auto engine = builder->buildSerializedNetwork(*network, *config);

    if (!engine)
    {
        LOG_ERROR("Failed to build engine.");
        return false;
    }
    std::ofstream ofs(engineFilePath, std::ios::out | std::ios::binary);
    if (!ofs)
    {
        LOG_ERROR("Failed to open file for writing: %s", engineFilePath.c_str());
        return false;
    }
    ofs.write(static_cast<char*>(engine->data()), engine->size());
    ofs.close();
    LOG_INFO("Engine saved to %s", engineFilePath.c_str());

    // Copy config
    if (!copyConfig())
    {
        return false;
    }

    return true;
}

bool VisualBuilder::parseConfig()
{
    std::string configPath = mOnnxDir.string() + "/config.json";
    std::ifstream configFileStream(configPath);
    if (!configFileStream.is_open())
    {
        LOG_ERROR("Failed to open config file: %s", configPath.c_str());
        return false;
    }

    try
    {
        mModelConfig = Json::parse(configFileStream);
    }
    catch (Json::parse_error const& e)
    {
        LOG_ERROR("Failed to parse config file: %s", e.what());
        return false;
    }

    // Check model version
    std::string modelVersion = mModelConfig.value(binding_names::kEdgellmVersion, "");
    version::checkVersion(modelVersion);

    // Read model type from vision_config.model_type
    std::string modelTypeStr;
    if (mModelConfig.contains(kVisionConfigKey) && mModelConfig[kVisionConfigKey].contains(kModelTypeKey))
    {
        modelTypeStr = mModelConfig[kVisionConfigKey][kModelTypeKey].get<std::string>();
    }
    // For Phi-4MM, the model type is at the top level
    else if (mModelConfig.contains(kModelTypeKey))
    {
        modelTypeStr = mModelConfig[kModelTypeKey].get<std::string>();
    }
    else
    {
        LOG_ERROR(
            "model_type not found in config.json (expected either vision_config.model_type or top-level model_type)");
        return false;
    }

    mModelType = multimodal::stringToModelType(modelTypeStr);

    if (mModelType == multimodal::ModelType::UNKNOWN)
    {
        LOG_ERROR("Unsupported model type: %s", modelTypeStr.c_str());
    }

    if (mModelType == multimodal::ModelType::INTERNVL)
    {
        mNumChannels = mModelConfig[kVisionConfigKey]["num_channels"].get<int64_t>();
        mImageSizeH = mModelConfig[kVisionConfigKey]["image_size"][0].get<int64_t>();
        mImageSizeW = mModelConfig[kVisionConfigKey]["image_size"][1].get<int64_t>();
    }

    if (mModelType == multimodal::ModelType::PHI4MM)
    {
        // Default Phi-4MM vision input
        mNumChannels = 3;
        // Prefer HF config's crop_size if available
        if (mModelConfig.contains(kEmbdLayerKey) && mModelConfig[kEmbdLayerKey].contains(kImageEmbdLayerKey)
            && mModelConfig[kEmbdLayerKey][kImageEmbdLayerKey].contains("crop_size"))
        {
            int64_t const crop = mModelConfig[kEmbdLayerKey][kImageEmbdLayerKey]["crop_size"].get<int64_t>();
            mImageSizeH = crop;
            mImageSizeW = crop;
        }
        else
        {
            LOG_INFO("Phi-4MM crop_size not found in config.json; defaulting to 448x448.");
            mImageSizeH = 448;
            mImageSizeW = 448;
        }
    }

    return true;
}

bool VisualBuilder::setupVisualOptimizationProfile(
    nvinfer1::IBuilder* const builder, nvinfer1::IBuilderConfig* config, nvinfer1::INetworkDefinition const* network)
{
    auto* visualProfile = builder->createOptimizationProfile();
    bool result = true;

    if (mModelType == multimodal::ModelType::QWEN2_VL || mModelType == multimodal::ModelType::QWEN2_5_VL
        || mModelType == multimodal::ModelType::QWEN3_VL)
    {
        result = setupQwenViTProfile(visualProfile, network);
    }
    else if (mModelType == multimodal::ModelType::INTERNVL || mModelType == multimodal::ModelType::PHI4MM)
    {
        result = setupInternPhi4ViTProfile(visualProfile);
    }
    if (!result)
    {
        LOG_ERROR("Failed to setup optimization profile");
        return false;
    }

    LOG_DEBUG("%s", printOptimizationProfile(visualProfile, "visual_profile", network).c_str());

    config->addOptimizationProfile(visualProfile);
    return true;
}

bool VisualBuilder::setupQwenViTProfile(
    nvinfer1::IOptimizationProfile* profile, nvinfer1::INetworkDefinition const* network)
{
    bool result = true;

    // In Qwen-VL, HW is always 4ximageTokens because it equals to spatial_merge_size ** 2.
    int64_t minHW = mBuilderConfig.minImageTokens * 4;
    int64_t maxHW = mBuilderConfig.maxImageTokens * 4;
    int64_t optHW = (mBuilderConfig.minImageTokens + mBuilderConfig.maxImageTokens) / 2 * 4;

    // Infer dimensions from the network
    int64_t inputDim = 0;
    int64_t ropeEmbedSize = 0;

    for (int32_t i = 0; i < network->getNbInputs(); ++i)
    {
        auto* input = network->getInput(i);
        if (strcmp(input->getName(), binding_names::kVisualInput) == 0)
        {
            inputDim = input->getDimensions().d[1];
        }
        else if (strcmp(input->getName(), binding_names::kRotaryPosEmb) == 0)
        {
            ropeEmbedSize = input->getDimensions().d[1];
        }
    }

    if (inputDim == 0)
    {
        LOG_ERROR("Cannot infer inputDim. Do you have proper ONNX input: %s?", binding_names::kVisualInput);
        return false;
    }

    if (ropeEmbedSize == 0)
    {
        LOG_ERROR("Cannot infer ropeEmbedSize. Do you have proper ONNX input: %s?", binding_names::kRotaryPosEmb);
        return false;
    }

    // Base inputs
    result &= setOptimizationProfile(profile, binding_names::kVisualInput, createDims({minHW, inputDim}),
        createDims({optHW, inputDim}), createDims({maxHW, inputDim}));
    result &= setOptimizationProfile(profile, binding_names::kRotaryPosEmb, createDims({minHW, ropeEmbedSize}),
        createDims({optHW, ropeEmbedSize}), createDims({maxHW, ropeEmbedSize}));
    result &= setOptimizationProfile(profile, binding_names::kAttentionMask, createDims({1, minHW, minHW}),
        createDims({1, optHW, optHW}), createDims({1, maxHW, maxHW}));

    // Additional inputs
    if (mModelType == multimodal::ModelType::QWEN2_5_VL)
    {
        result &= setOptimizationProfile(profile, binding_names::kWindowAttentionMask, createDims({1, minHW, minHW}),
            createDims({1, optHW, optHW}), createDims({1, maxHW, maxHW}));
        result &= setOptimizationProfile(profile, binding_names::kWindowIndex, createDims({minHW / 4}),
            createDims({optHW / 4}), createDims({maxHW / 4}));
        result &= setOptimizationProfile(profile, binding_names::kReverseWindowIndex, createDims({minHW / 4}),
            createDims({optHW / 4}), createDims({maxHW / 4}));
    }
    else if (mModelType == multimodal::ModelType::QWEN3_VL)
    {
        result &= setOptimizationProfile(profile, binding_names::kFastPosEmbIdx, createDims({4, minHW}),
            createDims({4, optHW}), createDims({4, maxHW}));
        result &= setOptimizationProfile(profile, binding_names::kFastPosEmbWeight, createDims({4, minHW}),
            createDims({4, optHW}), createDims({4, maxHW}));
    }

    if (!result)
    {
        LOG_ERROR("Failed to setup optimization profile at setupQwenViTProfile().");
    }

    return result;
}

bool VisualBuilder::setupInternPhi4ViTProfile(nvinfer1::IOptimizationProfile* profile)
{
    bool result = true;

    if (mBuilderConfig.minImageTokens % 256 != 0 || mBuilderConfig.maxImageTokens % 256 != 0)
    {
        LOG_ERROR("minImageTokens and maxImageTokens must be divisible by 256 for InternVL/Phi4-MM ViT model.");
        return false;
    }

    int64_t minNumBlocks = mBuilderConfig.minImageTokens / 256;
    int64_t maxNumBlocks = mBuilderConfig.maxImageTokens / 256;
    int64_t optNumBlocks = (minNumBlocks + maxNumBlocks) / 2;

    result &= setOptimizationProfile(profile, binding_names::kVisualInput,
        createDims({minNumBlocks, mNumChannels, mImageSizeH, mImageSizeW}),
        createDims({optNumBlocks, mNumChannels, mImageSizeH, mImageSizeW}),
        createDims({maxNumBlocks, mNumChannels, mImageSizeH, mImageSizeW}));

    return result;
}

bool VisualBuilder::copyConfig()
{
    std::string targetConfigPath = mEngineDir.string() + "/config.json";

    // Create a copy of mModelConfig and add builder config
    Json configWithBuilder = mModelConfig;
    configWithBuilder["builder_config"] = mBuilderConfig.toJson();

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

    // Copy preprocessor config if exists
    std::string preprocessorConfigPath = mOnnxDir.string() + "/preprocessor_config.json";
    if (std::filesystem::exists(preprocessorConfigPath))
    {
        std::string targetPreprocessorConfigPath = mEngineDir.string() + "/preprocessor_config.json";
        std::filesystem::copy(
            preprocessorConfigPath, targetPreprocessorConfigPath, std::filesystem::copy_options::overwrite_existing);
        LOG_INFO("Copied preprocessor config to %s", targetPreprocessorConfigPath.c_str());
    }
    else
    {
        LOG_WARNING("No preprocessor config found.");
    }

    return true;
}

} // namespace builder
} // namespace trt_edgellm
