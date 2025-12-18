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

#include <NvInferRuntime.h>
#include <string>
#include <vector>

namespace trt_edgellm
{
namespace plugins
{

//! \brief TensorRT plugin for attention operations (context and decode)
//!
//! This plugin implements efficient attention mechanisms including context attention (prefill)
//! and decode attention with KV cache support.
class AttentionPlugin : public nvinfer1::IPluginV2DynamicExt
{
public:
    //! \brief Constructor for attention plugin with configuration parameters
    //! \param[in] name Plugin instance name
    //! \param[in] numQHeads Number of query heads
    //! \param[in] numKVHeads Number of key-value heads
    //! \param[in] headSize Head dimension size
    //! \param[in] supportsSpecDecode Whether to support speculative decoding (Tree attention)
    AttentionPlugin(
        std::string const& name, int32_t numQHeads, int32_t numKVHeads, int32_t headSize, int32_t supportsSpecDecode);

    //! \brief Constructor for deserialization
    //! \param[in] name Plugin instance name
    //! \param[in] data Serialized plugin data
    //! \param[in] length Length of serialized data
    AttentionPlugin(std::string const& name, void const* data, size_t length);

    //! Force to distinguish different instances of the plugin
    AttentionPlugin() = delete;

    AttentionPlugin(AttentionPlugin const&) = delete;

    ~AttentionPlugin() override;

    //! \name IPluginV2DynamicExt Methods
    //! @{

    //! \brief Clone the plugin instance
    //! \return Pointer to cloned plugin
    nvinfer1::IPluginV2DynamicExt* clone() const noexcept override;

    //! \brief Get number of outputs
    //! \return Number of output tensors
    int32_t getNbOutputs() const noexcept override;

    //! \brief Get output data type
    //! \param[in] index Output index
    //! \param[in] inputTypes Array of input data types
    //! \param[in] nbInputs Number of inputs
    //! \return Output data type
    nvinfer1::DataType getOutputDataType(
        int32_t index, nvinfer1::DataType const* inputTypes, int32_t nbInputs) const noexcept override;

    //! \brief Get output dimensions
    //! \param[in] outputIndex Output tensor index
    //! \param[in] inputs Input tensor dimensions
    //! \param[in] nbInputs Number of inputs
    //! \param[in] exprBuilder Expression builder for dimension calculations
    //! \return Output tensor dimensions
    nvinfer1::DimsExprs getOutputDimensions(int32_t outputIndex, nvinfer1::DimsExprs const* inputs, int32_t nbInputs,
        nvinfer1::IExprBuilder& exprBuilder) noexcept override;

    //! \brief Check if format combination is supported
    //! \param[in] pos Position in the input/output tensor list
    //! \param[in] inOut Array of input and output tensor descriptors
    //! \param[in] nbInputs Number of inputs
    //! \param[in] nbOutputs Number of outputs
    //! \return True if format combination is supported
    bool supportsFormatCombination(
        int32_t pos, nvinfer1::PluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept override;

    //! \brief Configure the plugin with input and output tensors
    //! \param[in] in Input tensor descriptors
    //! \param[in] nbInputs Number of inputs
    //! \param[in] out Output tensor descriptors
    //! \param[in] nbOutputs Number of outputs
    void configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in, int32_t nbInputs,
        nvinfer1::DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept override;

    //! \brief Get workspace size required by the plugin
    //! \param[in] inputs Input tensor descriptors
    //! \param[in] nbInputs Number of inputs
    //! \param[in] outputs Output tensor descriptors
    //! \param[in] nbOutputs Number of outputs
    //! \return Workspace size in bytes
    size_t getWorkspaceSize(nvinfer1::PluginTensorDesc const* inputs, int32_t nbInputs,
        nvinfer1::PluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept override;

    //! \brief Execute the plugin
    //! \param[in] inputDesc Input tensor descriptors
    //! \param[in] outputDesc Output tensor descriptors
    //! \param[in] inputs Input tensor data pointers
    //! \param[out] outputs Output tensor data pointers
    //! \param[in] workspace Workspace memory pointer
    //! \param[in] stream CUDA stream for execution
    //! \return 0 on success, non-zero on failure
    int32_t enqueue(nvinfer1::PluginTensorDesc const* inputDesc, nvinfer1::PluginTensorDesc const* outputDesc,
        void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;

    //! \brief Get serialization size
    //! \return Size in bytes required for serialization
    size_t getSerializationSize() const noexcept override;

    //! \brief Serialize the plugin
    //! \param[out] buffer Buffer to write serialized data
    void serialize(void* buffer) const noexcept override;

    //! \brief Get plugin type
    //! \return Plugin type string
    char const* getPluginType() const noexcept override;

    //! \brief Get plugin namespace
    //! \return Plugin namespace string
    char const* getPluginNamespace() const noexcept override;

    //! \brief Set plugin namespace
    //! \param[in] pluginNamespace Namespace to set
    void setPluginNamespace(char const* pluginNamespace) noexcept;

    //! \brief Get plugin version
    //! \return Plugin version string
    char const* getPluginVersion() const noexcept override;

    //! \brief Initialize the plugin
    //! \return 0 on success, non-zero on failure
    int32_t initialize() noexcept override;

    //! \brief Terminate the plugin and release resources
    void terminate() noexcept override;

    //! \brief Destroy the plugin instance
    void destroy() noexcept override;

    //! @}

protected:
    std::string mLayerName; //!< Plugin layer name
    std::string mNamespace; //!< Plugin namespace

    //! Number of query heads (specified by model, runtime constant)
    int32_t mNumQHeads{};
    //! Number of key-value heads (specified by model, runtime constant)
    int32_t mNumKVHeads{};
    //! Number of elements per head (head dimension)
    int32_t mHeadSize{};
    //! Whether to enable tree attention for EAGLE speculative decoding
    int32_t mEnableTreeAttention{};

    //! Datatype of QKV and KV cache. Only supports FP16 as of now.
    nvinfer1::DataType const mDataType{nvinfer1::DataType::kHALF};
    int32_t mSMVersion; //!< CUDA SM version
};

//! \brief Factory class for creating AttentionPlugin instances
class AttentionPluginCreator : public nvinfer1::IPluginCreator
{
public:
    AttentionPluginCreator();

    ~AttentionPluginCreator() override = default;

    //! \brief Get plugin name
    //! \return Plugin name string
    char const* getPluginName() const noexcept override;

    //! \brief Get plugin field collection
    //! \return Pointer to plugin field collection containing all plugin fields
    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override;

    //! \brief Set plugin namespace
    //! \param[in] pluginNamespace Namespace to set
    void setPluginNamespace(char const* pluginNamespace) noexcept;

    //! \brief Get plugin namespace
    //! \return Plugin namespace string
    char const* getPluginNamespace() const noexcept override;

    //! \brief Get plugin version
    //! \return Plugin version string
    char const* getPluginVersion() const noexcept override;

    //! \brief Create a new plugin instance
    //! \param[in] name Plugin instance name
    //! \param[in] fc Plugin field collection containing configuration parameters
    //! \return Pointer to created plugin instance
    nvinfer1::IPluginV2* createPlugin(char const* name, nvinfer1::PluginFieldCollection const* fc) noexcept override;

    //! \brief Deserialize a plugin instance from data
    //! \param[in] name Plugin instance name
    //! \param[in] serialData Serialized plugin data
    //! \param[in] serialLength Length of serialized data in bytes
    //! \return Pointer to deserialized plugin instance
    nvinfer1::IPluginV2* deserializePlugin(
        char const* name, void const* serialData, size_t serialLength) noexcept override;

private:
    static nvinfer1::PluginFieldCollection mFieldCollection;     //!< Plugin field collection for registration
    static std::vector<nvinfer1::PluginField> mPluginAttributes; //!< Plugin attributes/fields
    std::string mNamespace;                                      //!< Plugin namespace
};

} // namespace plugins
} // namespace trt_edgellm