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

#pragma once

#include "multimodal/common/imageUtils.h"
#include "multimodal/common/modelTypes.h"
#include <NvInfer.h>
#include <filesystem>
#include <nlohmann/json.hpp>
#include <sstream>
#include <string>

using Json = nlohmann::json;

namespace trt_edgellm
{

namespace builder
{

//! Configuration structure for visual model building.
//! Contains parameters needed to configure the TensorRT engine building process
//! for visual encoders used in Vision-Language Models.
struct VisualBuilderConfig
{
    int64_t minImageTokens{4};           //!< Minimum number of image tokens in a batch
    int64_t maxImageTokens{1024};        //!< Maximum number of image tokens in a batch
    int64_t maxImageTokensPerImage{512}; //!< Maximum number of image tokens per image
    bool profilingDetailed{false};       //!< Enable detailed profiling verbosity for layer info extraction
    bool useTrtNativeVitAttn{false};     //!< Use TRT IAttention

    //! Convert configuration to JSON format for serialization.
    //! @return JSON object containing all configuration parameters
    Json toJson() const
    {
        Json json;
        json["min_image_tokens"] = minImageTokens;
        json["max_image_tokens"] = maxImageTokens;
        json["max_image_tokens_per_image"] = maxImageTokensPerImage;
        // The cu_seqlens profile capacity, so the server can budget request
        // media against the real engine limit instead of re-deriving it.
        json["max_cu_seqlen_groups"] = rt::imageUtils::maxCuSeqlenGroups(maxImageTokens);
        json["use_trt_native_vit_attn"] = useTrtNativeVitAttn;
        return json;
    }

    //! Create configuration from JSON format.
    //! @param json JSON object containing configuration parameters
    //! @return VisualBuilderConfig object with parsed parameters
    static VisualBuilderConfig fromJson(Json const& json)
    {
        VisualBuilderConfig config;
        if (json.contains("min_image_tokens"))
        {
            config.minImageTokens = json["min_image_tokens"];
        }
        if (json.contains("max_image_tokens"))
        {
            config.maxImageTokens = json["max_image_tokens"];
        }
        if (json.contains("max_image_tokens_per_image"))
        {
            config.maxImageTokensPerImage = json["max_image_tokens_per_image"];
        }
        if (json.contains("use_trt_native_vit_attn"))
        {
            config.useTrtNativeVitAttn = json["use_trt_native_vit_attn"];
        }
        return config;
    }

    //! Convert configuration to human-readable string format.
    //! @return String representation of the configuration for debugging/logging
    std::string toString() const
    {
        std::ostringstream oss;
        oss << "VisualBuilderConfig:\n";
        oss << "  minImageTokens: " << minImageTokens << "\n";
        oss << "  maxImageTokens: " << maxImageTokens << "\n";
        oss << "  maxImageTokensPerImage: " << maxImageTokensPerImage << "\n";
        return oss.str();
    }
};

//! Builder class for visual encoder TensorRT engines.
//! Handles the complete process of building TensorRT engines from ONNX models
//! for visual encoders used in Vision-Language Models.
class VisualBuilder
{
public:
    //! Constructor for VisualBuilder.
    //! @param onnxDir Directory containing the ONNX model and configuration files
    //! @param engineDir Directory where the built engine and related files will be saved
    //! @param config Configuration object specifying build parameters
    VisualBuilder(std::filesystem::path const& onnxDir, std::filesystem::path const& engineDir,
        VisualBuilderConfig const& config);

    //! Destructor.
    ~VisualBuilder() noexcept = default;

    //! Build the TensorRT engine from the ONNX model.
    //! This method performs the complete build process including:
    //! - Loading and parsing the ONNX model
    //! - Setting up optimization profiles
    //! - Building the TensorRT engine
    //! - Copying necessary files to the engine directory
    //! @return true if build was successful, false otherwise
    bool build();

private:
    std::filesystem::path mOnnxDir;     //!< Directory containing ONNX model files
    std::filesystem::path mEngineDir;   //!< Directory for saving built engine
    VisualBuilderConfig mBuilderConfig; //!< Build configuration
    multimodal::ModelType mModelType;   //!< Model type inferred from config.json

    //! Parse the model configuration from config.json.
    //! Extracts model type and dimensions needed for optimization profile setup.
    //! @return true if parsing was successful, false otherwise
    bool parseConfig();

    //! Set up optimization profile for visual models.
    //! Creates a single optimization profile with appropriate dynamic shapes.
    //! @param builder TensorRT builder object
    //! @param config TensorRT builder config object
    //! @param network TensorRT network definition
    //! @return true if setup was successful, false otherwise
    bool setupVisualOptimizationProfile(
        nvinfer1::IBuilder& builder, nvinfer1::IBuilderConfig& config, nvinfer1::INetworkDefinition const& network);

    //! Set up optimization profile for Qwen ViT models.
    //! Configures inputs for Qwen2-VL and Qwen2.5-VL visual encoders.
    //! @param profile Optimization profile to configure
    //! @param network TensorRT network definition for input analysis
    //! @return true if setup was successful, false otherwise
    bool setupQwenViTProfile(nvinfer1::IOptimizationProfile& profile, nvinfer1::INetworkDefinition const& network);

    //! Set up optimization profile for InternVL or Phi4-MM ViT models.
    //! Configures inputs for InternVL or Phi4-MM visual encoders.
    //! @param profile Optimization profile to configure
    //! @return true if setup was successful, false otherwise
    bool setupInternPhi4ViTProfile(nvinfer1::IOptimizationProfile& profile);

    //! Set up optimization profile for Nemotron-Omni RADIO ViT model.
    //! Configures input for RADIO vision encoder with dynamic tile batching.
    //! @param profile Optimization profile to configure
    //! @return true if setup was successful, false otherwise
    bool setupNemotronOmniViTProfile(nvinfer1::IOptimizationProfile& profile);

    //! Set up optimization profile for Gemma4 ViT model.
    //! Configures ragged patch, position-id, and position-aware pooling inputs.
    //! @param profile Optimization profile to configure
    //! @param network TensorRT network definition for input analysis
    //! @return true if setup was successful, false otherwise
    bool setupGemma4ViTProfile(nvinfer1::IOptimizationProfile& profile, nvinfer1::INetworkDefinition const& network);

    //! Set up the two dynamic packed-patch inputs for encoder-free Gemma4 Unified vision.
    bool setupGemma4UnifiedVisionProfile(
        nvinfer1::IOptimizationProfile& profile, nvinfer1::INetworkDefinition const& network);

    //! Copy and save the model configuration with builder config.
    //! Creates a config.json file in the engine directory with both original model config
    //! and builder configuration parameters.
    //! @return true if copying was successful, false otherwise
    bool copyConfig();

    // Model dimensions extracted from config.json
    int64_t mNumChannels{0};   //!< Number of input channels
    int64_t mImageSizeH{0};    //!< Image height
    int64_t mImageSizeW{0};    //!< Image width
    int64_t mInputDim{0};      //!< Input dimension for Qwen models
    int64_t mRopeEmbedSize{0}; //!< Rotary position embedding size
    Json mModelConfig;         //!< Parsed model configuration
};

} // namespace builder
} // namespace trt_edgellm
