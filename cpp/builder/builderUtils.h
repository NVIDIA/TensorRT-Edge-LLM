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

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <filesystem>
#include <memory>
#include <nlohmann/json.hpp>
#include <string>
#include <utility>
#include <vector>

namespace trt_edgellm
{
namespace builder
{

//! JSON key names used in model configs
constexpr char kVisionConfigKey[] = "vision_config";
constexpr char kModelTypeKey[] = "model_type";
constexpr char kEmbdLayerKey[] = "embd_layer";
constexpr char kImageEmbdLayerKey[] = "image_embd_layer";

//! Create TensorRT dimensions from a vector of shape values.
//! @param shape Vector of dimension sizes
//! @return TensorRT Dims object with the specified dimensions
//! @throws std::invalid_argument if shape size exceeds MAX_DIMS
nvinfer1::Dims createDims(std::vector<int64_t> const& shape);

//! Validate optimization profile dimensions.
//! Ensures that min <= opt <= max for all dimensions and that all profiles have the same number of dimensions.
//! @param minDims Minimum dimensions for the optimization profile
//! @param optDims Optimal dimensions for the optimization profile
//! @param maxDims Maximum dimensions for the optimization profile
//! @return true if dimensions are valid, false otherwise
bool checkOptimizationProfileDims(
    nvinfer1::Dims const& minDims, nvinfer1::Dims const& optDims, nvinfer1::Dims const& maxDims) noexcept;

//! Set optimization profile dimensions for a specific input.
//! Validates the dimensions and sets them on the optimization profile.
//! @param profile Optimization profile to configure
//! @param inputName Name of the input tensor
//! @param minDims Minimum dimensions for the input
//! @param optDims Optimal dimensions for the input
//! @param maxDims Maximum dimensions for the input
//! @return true if setting was successful, false otherwise
bool setOptimizationProfile(nvinfer1::IOptimizationProfile* profile, char const* inputName,
    nvinfer1::Dims const& minDims, nvinfer1::Dims const& optDims, nvinfer1::Dims const& maxDims) noexcept;

//! Print detailed information about the TensorRT network.
//! Shows input and output tensor names and shapes for debugging purposes.
//! @param network TensorRT network definition to analyze
//! @param prefix Optional prefix for the output string
//! @return Formatted string containing network information
std::string printNetworkInfo(nvinfer1::INetworkDefinition const* network, std::string const& prefix = "");

//! Print detailed information about an optimization profile.
//! Shows the min, optimal, and max dimensions for each dynamic input in the profile.
//! @param profile Optimization profile to analyze
//! @param profileName Name of the profile for display purposes
//! @param network TensorRT network definition for input analysis
//! @return Formatted string containing optimization profile information
std::string printOptimizationProfile(nvinfer1::IOptimizationProfile const* profile, std::string const& profileName,
    nvinfer1::INetworkDefinition const* network);

//! Apply TensorRT compile workarounds via the __LUNOWUD environment variable.
//! Must be called before createInferBuilder.
//! @return The applied __LUNOWUD flag string (empty if none)
std::string applyCompileWorkarounds();

//! Create TensorRT builder and network definition with strongly typed flag.
//! @return Pair of builder and network, or {nullptr, nullptr} on failure
std::pair<std::unique_ptr<nvinfer1::IBuilder>, std::unique_ptr<nvinfer1::INetworkDefinition>> createBuilderAndNetwork();

//! Create TensorRT builder config with optimized settings.
//! @param builder TensorRT builder object
//! @param enableAliasedPluginIO Enable the kALIASED_PLUGIN_IO preview. Only the
//!        DFlash/DSpark draft engines declare aliased plugin I/O; every other
//!        plugin returns -1 from getAliasedInput, so leave this off elsewhere.
//! @return Builder config with monitor memory flag enabled (TRT >= 10.6)
std::unique_ptr<nvinfer1::IBuilderConfig> createBuilderConfig(
    nvinfer1::IBuilder* builder, bool enableAliasedPluginIO = false);

//! Parse ONNX model and create parser.
//! @param network TensorRT network definition to populate
//! @param onnxFilePath Path to ONNX model file
//! @return ONNX parser object, or nullptr on failure
std::unique_ptr<nvonnxparser::IParser> parseOnnxModel(
    nvinfer1::INetworkDefinition* network, std::string const& onnxFilePath);

//! Build and serialize TensorRT engine to file.
//! @param builder TensorRT builder object
//! @param network TensorRT network definition
//! @param config TensorRT builder config
//! @param engineFilePath Output path for serialized engine
//! @return true if successful, false otherwise
bool buildAndSerializeEngine(nvinfer1::IBuilder* builder, nvinfer1::INetworkDefinition* network,
    nvinfer1::IBuilderConfig* config, std::string const& engineFilePath);

//! Load and parse JSON config file.
//! @param configPath Path to config.json file
//! @param outConfig Output JSON object
//! @return true if successful, false otherwise
bool loadJsonConfig(std::string const& configPath, nlohmann::json& outConfig);

//! Save model config with builder config to engine directory.
//! @param engineDir Engine directory path
//! @param modelConfig Original model configuration
//! @param builderConfig Builder configuration as JSON
//! @return true if successful, false otherwise
bool saveConfigWithBuilderInfo(
    std::filesystem::path const& engineDir, nlohmann::json const& modelConfig, nlohmann::json const& builderConfig);

//! Log TRT-native attention path selection at engine build time.
void logTrtNativeAttentionPath(char const* component) noexcept;

} // namespace builder
} // namespace trt_edgellm
