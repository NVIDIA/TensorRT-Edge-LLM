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

#pragma once

#include "multimodal/modelTypes.h"
#include <NvInfer.h>
#include <filesystem>
#include <memory>
#include <nlohmann/json.hpp>
#include <sstream>
#include <string>

using Json = nlohmann::json;

namespace trt_edgellm
{

namespace builder
{

//! Configuration structure for LLM model building.
//! Contains all parameters needed to configure the TensorRT engine building process
//! for Large Language Models, including standard LLMs, Eagle models, and Vision-Language Models.
struct LLMBuilderConfig
{
    int64_t maxInputLen{128};         //!< Maximum input sequence length for the model
    bool isVlm{false};                //!< Whether this is a Vision-Language Model (VLM)
    int64_t minImageTokens{4};        //!< Minimum number of image tokens (VLM only)
    int64_t maxImageTokens{1024};     //!< Maximum number of image tokens (VLM only)
    bool eagleDraft{false};           //!< Whether this is an Eagle draft model
    bool eagleBase{false};            //!< Whether this is an Eagle base model
    int64_t maxBatchSize{4};          //!< Maximum batch size for inference
    int64_t maxLoraRank{0};           //!< Maximum LoRA rank (0 = no LoRA support)
    int64_t maxKVCacheCapacity{4096}; //!< Maximum KV cache capacity (sequence length)
    int64_t maxVerifyTreeSize{60}; //!< Maximum length of input_ids passed into Eagle base model for tree verification
    int64_t maxDraftTreeSize{60};  //!< Maximum length of input_ids passed into Eagle draft model for draft generation

    //! Convert configuration to JSON format for serialization.
    //! @return JSON object containing all configuration parameters
    Json toJson() const
    {
        Json json;
        json["max_input_len"] = maxInputLen;
        json["is_vlm"] = isVlm;
        if (isVlm)
        {
            json["min_image_tokens"] = minImageTokens;
            json["max_image_tokens"] = maxImageTokens;
        }
        json["eagle_draft"] = eagleDraft;
        json["eagle_base"] = eagleBase;
        json["max_batch_size"] = maxBatchSize;
        json["max_lora_rank"] = maxLoraRank;
        json["max_kv_cache_capacity"] = maxKVCacheCapacity;
        // Only include Eagle-specific fields when Eagle is enabled
        if (eagleBase)
        {
            json["max_verify_tree_size"] = maxVerifyTreeSize;
        }
        if (eagleDraft)
        {
            json["max_draft_tree_size"] = maxDraftTreeSize;
        }
        return json;
    }

    //! Create configuration from JSON format.
    //! @param json JSON object containing configuration parameters
    //! @return LLMBuilderConfig object with parsed parameters
    static LLMBuilderConfig fromJson(Json const& json)
    {
        LLMBuilderConfig config;
        if (json.contains("max_input_len"))
        {
            config.maxInputLen = json["max_input_len"];
        }
        if (json.contains("is_vlm"))
        {
            config.isVlm = json["is_vlm"];
        }
        if (json.contains("min_image_tokens"))
        {
            config.minImageTokens = json["min_image_tokens"];
        }
        if (json.contains("max_image_tokens"))
        {
            config.maxImageTokens = json["max_image_tokens"];
        }
        if (json.contains("eagle_draft"))
        {
            config.eagleDraft = json["eagle_draft"];
        }
        if (json.contains("eagle_base"))
        {
            config.eagleBase = json["eagle_base"];
        }
        if (json.contains("max_batch_size"))
        {
            config.maxBatchSize = json["max_batch_size"];
        }
        if (json.contains("max_lora_rank"))
        {
            config.maxLoraRank = json["max_lora_rank"];
        }
        if (json.contains("max_kv_cache_capacity"))
        {
            config.maxKVCacheCapacity = json["max_kv_cache_capacity"];
        }
        if (json.contains("max_verify_tree_size"))
        {
            config.maxVerifyTreeSize = json["max_verify_tree_size"];
        }
        if (json.contains("max_draft_tree_size"))
        {
            config.maxDraftTreeSize = json["max_draft_tree_size"];
        }
        return config;
    }

    //! Convert configuration to human-readable string format.
    //! @return String representation of the configuration for debugging/logging
    std::string toString() const
    {
        std::ostringstream oss;
        oss << "LLMBuilderConfig:\n";
        oss << "  maxInputLen: " << maxInputLen << "\n";
        oss << "  isVlm: " << (isVlm ? "true" : "false") << "\n";
        if (isVlm)
        {
            oss << "  minImageTokens: " << minImageTokens << "\n";
            oss << "  maxImageTokens: " << maxImageTokens << "\n";
        }
        oss << "  eagleDraft: " << (eagleDraft ? "true" : "false") << "\n";
        oss << "  eagleBase: " << (eagleBase ? "true" : "false") << "\n";
        oss << "  maxBatchSize: " << maxBatchSize << "\n";
        oss << "  maxLoraRank: " << maxLoraRank << "\n";
        oss << "  maxKVCacheCapacity: " << maxKVCacheCapacity << "\n";
        // Only show Eagle-specific fields when Eagle is enabled
        if (eagleBase)
        {
            oss << "  maxVerifyTreeSize: " << maxVerifyTreeSize << "\n";
        }
        if (eagleDraft)
        {
            oss << "  maxDraftTreeSize: " << maxDraftTreeSize << "\n";
        }
        return oss.str();
    }
};

//! Configuration structure for visual model building.
//! Contains parameters needed to configure the TensorRT engine building process
//! for visual encoders used in Vision-Language Models.
struct VisualBuilderConfig
{
    int64_t minImageTokens{4};           //!< Minimum number of image tokens in a batch
    int64_t maxImageTokens{1024};        //!< Maximum number of image tokens in a batch
    int64_t maxImageTokensPerImage{512}; //!< Maximum number of image tokens per image, used for preprocessing

    //! Convert configuration to JSON format for serialization.
    //! @return JSON object containing all configuration parameters
    Json toJson() const
    {
        Json json;
        json["min_image_tokens"] = minImageTokens;
        json["max_image_tokens"] = maxImageTokens;
        json["max_image_tokens_per_image"] = maxImageTokensPerImage;
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

//! Builder class for Large Language Model TensorRT engines.
//! Handles the complete process of building TensorRT engines from ONNX models
//! for various types of LLMs including standard models, Eagle models, and VLMs.
class LLMBuilder
{
public:
    //! Constructor for LLMBuilder.
    //! @param onnxDir Directory containing the ONNX model and configuration files
    //! @param engineDir Directory where the built engine and related files will be saved
    //! @param config Configuration object specifying build parameters
    LLMBuilder(
        std::filesystem::path const& onnxDir, std::filesystem::path const& engineDir, LLMBuilderConfig const& config);

    //! Destructor.
    ~LLMBuilder() = default;

    //! Build the TensorRT engine from the ONNX model.
    //! This method performs the complete build process including:
    //! - Loading and parsing the ONNX model
    //! - Setting up optimization profiles
    //! - Building the TensorRT engine
    //! - Copying necessary files to the engine directory
    //! @return true if build was successful, false otherwise
    bool build();

private:
    std::filesystem::path mOnnxDir;   //!< Directory containing ONNX model files
    std::filesystem::path mEngineDir; //!< Directory for saving built engine
    LLMBuilderConfig mBuilderConfig;  //!< Build configuration

    //! Parse the model configuration from config.json.
    //! Extracts model dimensions and parameters needed for optimization profile setup.
    //! @return true if parsing was successful, false otherwise
    bool parseConfig();

    //! Set up optimization profiles for LLM models.
    //! Creates context and generation profiles with appropriate dynamic shapes.
    //! @param builder TensorRT builder object
    //! @param config TensorRT builder config object
    //! @param network TensorRT network definition
    //! @return true if setup was successful, false otherwise
    bool setupLLMOptimizationProfiles(
        nvinfer1::IBuilder* builder, nvinfer1::IBuilderConfig* config, nvinfer1::INetworkDefinition const* network);

    //! Set up common optimization profiles shared by all LLM types.
    //! Configures context lengths, rotary embeddings, and KV cache profiles.
    //! @param contextProfile Optimization profile for context processing
    //! @param generationProfile Optimization profile for generation processing
    //! @return true if setup was successful, false otherwise
    bool setupCommonProfiles(
        nvinfer1::IOptimizationProfile* contextProfile, nvinfer1::IOptimizationProfile* generationProfile);

    //! Set up optimization profiles for vanilla (non-Eagle) LLM models.
    //! Configures input IDs and last token IDs for standard transformer models.
    //! @param contextProfile Optimization profile for context processing
    //! @param generationProfile Optimization profile for generation processing
    //! @return true if setup was successful, false otherwise
    bool setupVanillaProfiles(
        nvinfer1::IOptimizationProfile* contextProfile, nvinfer1::IOptimizationProfile* generationProfile);

    //! Set up optimization profiles for Eagle models.
    //! Configures Eagle-specific inputs like hidden states and attention masks.
    //! @param contextProfile Optimization profile for context processing
    //! @param generationProfile Optimization profile for generation processing
    //! @return true if setup was successful, false otherwise
    bool setupEagleProfiles(
        nvinfer1::IOptimizationProfile* contextProfile, nvinfer1::IOptimizationProfile* generationProfile);

    //! Set up optimization profiles for Vision-Language Models.
    //! Configures image embedding inputs for VLM processing.
    //! @param contextProfile Optimization profile for context processing
    //! @param generationProfile Optimization profile for generation processing
    //! @param network TensorRT network definition for input analysis
    //! @return true if setup was successful, false otherwise
    bool setupVLMProfiles(nvinfer1::IOptimizationProfile* contextProfile,
        nvinfer1::IOptimizationProfile* generationProfile, nvinfer1::INetworkDefinition const* network);

    //! Set up optimization profiles for LoRA-enabled models.
    //! Configures LoRA weight matrices with dynamic rank support.
    //! @param contextProfile Optimization profile for context processing
    //! @param generationProfile Optimization profile for generation processing
    //! @param network TensorRT network definition for LoRA input analysis
    //! @return true if setup was successful, false otherwise
    bool setupLoraProfiles(nvinfer1::IOptimizationProfile* contextProfile,
        nvinfer1::IOptimizationProfile* generationProfile, nvinfer1::INetworkDefinition const* network);

    //! Set up optimization profiles for KV cache tensors.
    //! Configures dynamic shapes for key-value cache inputs across all layers.
    //! @param contextProfile Optimization profile for context processing
    //! @param generationProfile Optimization profile for generation processing
    //! @return true if setup was successful, false otherwise
    bool setupKVCacheProfiles(
        nvinfer1::IOptimizationProfile* contextProfile, nvinfer1::IOptimizationProfile* generationProfile);

    //! Copy and save the model configuration with builder config.
    //! Creates a config.json file in the engine directory with both original model config
    //! and builder configuration parameters.
    //! @return true if copying was successful, false otherwise
    bool copyConfig();

    //! Copy tokenizer files to the engine directory.
    //! Copies tokenizer_config.json and tokenizer.json files needed for inference.
    //! @return true if copying was successful, false otherwise
    bool copyTokenizerFiles();

    //! Copy Eagle-specific files to the engine directory.
    //! Copies d2t.safetensors file for Eagle3 draft models.
    //! @return true if copying was successful, false otherwise
    bool copyEagleFiles();

    //! Copy vocabulary mapping files to the engine directory.
    //! Copies vocab_map.safetensors file if reduced vocabulary is used.
    //! @return true if copying was successful, false otherwise
    bool copyVocabMappingFiles();

    // Model dimensions extracted from config.json
    int64_t mHiddenSize{0};                 //!< Hidden size of the model
    int64_t mNumKVHeads{0};                 //!< Number of key-value heads
    int64_t mHeadSize{0};                   //!< Size of each attention head
    int64_t mRotaryDim{0};                  //!< Dimension for rotary position embeddings
    int32_t mNbKVCacheInputs{0};            //!< Number of KV cache inputs (layers)
    int32_t mTargetModelOutputHiddenDim{0}; //!< Target output hidden dimension
    Json mModelConfig;                      //!< Parsed model configuration
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
    ~VisualBuilder() = default;

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
        nvinfer1::IBuilder* builder, nvinfer1::IBuilderConfig* config, nvinfer1::INetworkDefinition const* network);

    //! Set up optimization profile for Qwen ViT models.
    //! Configures inputs for Qwen2-VL and Qwen2.5-VL visual encoders.
    //! @param profile Optimization profile to configure
    //! @param network TensorRT network definition for input analysis
    //! @return true if setup was successful, false otherwise
    bool setupQwenViTProfile(nvinfer1::IOptimizationProfile* profile, nvinfer1::INetworkDefinition const* network);

    //! Set up optimization profile for InternVL or Phi4-MM ViT models.
    //! Configures inputs for InternVL or Phi4-MM visual encoders.
    //! @param profile Optimization profile to configure
    //! @return true if setup was successful, false otherwise
    bool setupInternPhi4ViTProfile(nvinfer1::IOptimizationProfile* profile);

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
