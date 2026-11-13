/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

//! Configuration structure for action expert model building.
//! Contains parameters needed to build the TensorRT engine from the action expert ONNX
//! (e.g. Alpamayo 1 flow-matching decoder).
struct ActionBuilderConfig
{
    //! Maximum batch size for dynamic batch dimensions.
    int32_t maxBatchSize{4};

    //! Maximum KV cache capacity (must match the LLM engine's maxKVCacheCapacity).
    int32_t maxKVCacheCapacity{0};

    //! Fixed physical K-page count for the pool inputs; zero derives the minimum.
    int32_t maxKVPoolPages{0};

    //! Convert configuration to JSON format for serialization.
    //! @return JSON object containing all configuration parameters
    Json toJson() const
    {
        Json json;
        json["max_batch_size"] = maxBatchSize;
        json["max_kv_cache_capacity"] = maxKVCacheCapacity;
        return json;
    }

    //! Create configuration from JSON format.
    //! @param json JSON object containing configuration parameters
    //! @return ActionBuilderConfig object with parsed parameters
    static ActionBuilderConfig fromJson(Json const& json)
    {
        ActionBuilderConfig config;
        if (json.contains("max_batch_size"))
        {
            config.maxBatchSize = json["max_batch_size"];
        }
        if (json.contains("max_kv_cache_capacity"))
        {
            config.maxKVCacheCapacity = json["max_kv_cache_capacity"];
        }
        return config;
    }

    //! Convert configuration to human-readable string format.
    //! @return String representation of the configuration for debugging/logging
    std::string toString() const
    {
        std::ostringstream oss;
        oss << "ActionBuilderConfig:\n";
        oss << "  maxBatchSize: " << maxBatchSize << "\n";
        oss << "  maxKVCacheCapacity: " << maxKVCacheCapacity << "\n";
        oss << "  maxKVPoolPages: " << maxKVPoolPages << "\n";
        return oss.str();
    }
};

//! Builder class for action expert TensorRT engines.
//! Handles the complete process of building TensorRT engines from ONNX models
//! for action experts used in multimodal models (e.g., Alpamayo 1).
class ActionBuilder
{
public:
    //! Constructor for ActionBuilder.
    //! @param onnxDir Directory containing the ONNX model and configuration files
    //! @param engineDir Directory where the built engine and related files will be saved
    //! @param config Configuration object specifying build parameters
    ActionBuilder(std::filesystem::path const& onnxDir, std::filesystem::path const& engineDir,
        ActionBuilderConfig const& config);

    //! Destructor.
    ~ActionBuilder() noexcept = default;

    //! Build the TensorRT engine from the action expert ONNX.
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
    ActionBuilderConfig mBuilderConfig; //!< Build configuration

    // Model architecture parameters (read from config.json)
    int32_t mNumLayers{0};          //!< Number of transformer layers
    int32_t mNumHeads{0};           //!< Number of key-value heads
    int32_t mHeadDim{0};            //!< Head dimension
    int32_t mNumDiffusionTokens{0}; //!< Number of diffusion tokens (waypoints)

    //! Parse the model configuration from config.json
    //! @return true if parsing was successful, false otherwise
    bool parseConfig();

    //! Set up optimization profile for action expert.
    //! Creates optimization profile with appropriate dynamic shapes for action inputs.
    //! @param builder TensorRT builder object (must not be null)
    //! @param config TensorRT builder config object (must not be null)
    //! @param network TensorRT network definition (must not be null)
    //! @return true if setup was successful, false otherwise
    bool setupActionOptimizationProfile(
        nvinfer1::IBuilder& builder, nvinfer1::IBuilderConfig& config, nvinfer1::INetworkDefinition const& network);

    //! Profile the dense KV inputs used with TP-sharded LLMs.
    bool setupDenseActionOptimizationProfile(
        nvinfer1::IBuilder& builder, nvinfer1::IBuilderConfig& config, nvinfer1::INetworkDefinition const& network);

    //! Profile the shared LLM pool and private action scratch for TP=1.
    bool setupPagedActionOptimizationProfile(
        nvinfer1::IBuilder& builder, nvinfer1::IBuilderConfig& config, nvinfer1::INetworkDefinition const& network);

    //! Copy and save the model configuration with builder config.
    //! Creates a config.json file in the engine directory with both original model config
    //! and builder configuration parameters.
    //! @return true if copying was successful, false otherwise
    bool copyConfig();

    Json mModelConfig; //!< Parsed model configuration (from config.json)
};

} // namespace builder
} // namespace trt_edgellm
