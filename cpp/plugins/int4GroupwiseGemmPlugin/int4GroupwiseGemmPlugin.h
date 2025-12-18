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
/*!
 * @brief TensorRT plugin for INT4 group-wise quantized GEMM
 *
 * Implements efficient INT4 quantized matrix multiplication with group-wise quantization.
 * Used for quantized weight matrix multiplications in LLM inference.
 */
class Int4GroupwiseGemmPlugin : public nvinfer1::IPluginV2DynamicExt
{
public:
    /*!
     * @brief Construct INT4 group-wise GEMM plugin
     * @param name Layer name
     * @param N Output dimension (columns in weight matrix)
     * @param K Input dimension (rows in weight matrix)
     * @param groupSize Quantization group size
     */
    Int4GroupwiseGemmPlugin(std::string const& name, int32_t N, int32_t K, int32_t groupSize);

    /*!
     * @brief Construct from serialized data
     * @param name Layer name
     * @param data Serialized plugin data
     * @param length Size of serialized data
     */
    Int4GroupwiseGemmPlugin(std::string const& name, void const* data, size_t length);

    //! @brief Deleted default constructor
    Int4GroupwiseGemmPlugin() = delete;

    //! @brief Deleted copy constructor
    Int4GroupwiseGemmPlugin(Int4GroupwiseGemmPlugin const&) = delete;

    //! @brief Destructor
    ~Int4GroupwiseGemmPlugin() override;

    // IPluginV2DynamicExt Methods
    //! @brief Clone the plugin for use in another network
    //! @return Cloned plugin instance
    nvinfer1::IPluginV2DynamicExt* clone() const noexcept override;

    //! @brief Get number of output tensors
    //! @return Number of outputs (1)
    int32_t getNbOutputs() const noexcept override;

    //! @brief Get output tensor data type
    //! @param index Output index
    //! @param inputTypes Input data types
    //! @param nbInputs Number of inputs
    //! @return Output data type
    nvinfer1::DataType getOutputDataType(
        int32_t index, nvinfer1::DataType const* inputTypes, int32_t nbInputs) const noexcept override;

    //! @brief Get output tensor dimensions
    //! @param outputIndex Output index
    //! @param inputs Input dimensions
    //! @param nbInputs Number of inputs
    //! @param exprBuilder Expression builder for dynamic shapes
    //! @return Output dimensions
    nvinfer1::DimsExprs getOutputDimensions(int32_t outputIndex, nvinfer1::DimsExprs const* inputs, int32_t nbInputs,
        nvinfer1::IExprBuilder& exprBuilder) noexcept override;

    //! @brief Check if format combination is supported
    //! @param pos Position in input/output array
    //! @param inOut Input and output tensor descriptors
    //! @param nbInputs Number of inputs
    //! @param nbOutputs Number of outputs
    //! @return True if supported
    bool supportsFormatCombination(
        int32_t pos, nvinfer1::PluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept override;

    //! @brief Configure plugin with tensor descriptions
    //! @param in Input tensor descriptors
    //! @param nbInputs Number of inputs
    //! @param out Output tensor descriptors
    //! @param nbOutputs Number of outputs
    void configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in, int32_t nbInputs,
        nvinfer1::DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept override;

    //! @brief Get workspace size required for execution
    //! @param inputs Input tensor descriptors
    //! @param nbInputs Number of inputs
    //! @param outputs Output tensor descriptors
    //! @param nbOutputs Number of outputs
    //! @return Workspace size in bytes
    size_t getWorkspaceSize(nvinfer1::PluginTensorDesc const* inputs, int32_t nbInputs,
        nvinfer1::PluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept override;

    //! @brief Execute the plugin
    //! @param inputDesc Input tensor descriptors
    //! @param outputDesc Output tensor descriptors
    //! @param inputs Input tensor pointers
    //! @param outputs Output tensor pointers
    //! @param workspace Workspace pointer
    //! @param stream CUDA stream
    //! @return 0 on success, non-zero on error
    int32_t enqueue(nvinfer1::PluginTensorDesc const* inputDesc, nvinfer1::PluginTensorDesc const* outputDesc,
        void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;

    //! @brief Get serialization size
    //! @return Size in bytes
    size_t getSerializationSize() const noexcept override;

    //! @brief Serialize plugin state
    //! @param buffer Output buffer
    void serialize(void* buffer) const noexcept override;

    //! @brief Get plugin type name
    //! @return Plugin type string
    char const* getPluginType() const noexcept override;

    //! @brief Get plugin namespace
    //! @return Namespace string
    char const* getPluginNamespace() const noexcept override;

    //! @brief Set plugin namespace
    //! @param pluginNamespace Namespace string
    void setPluginNamespace(char const* pluginNamespace) noexcept;

    //! @brief Get plugin version
    //! @return Version string
    char const* getPluginVersion() const noexcept override;

    //! @brief Initialize plugin resources
    //! @return 0 on success
    int32_t initialize() noexcept override;

    //! @brief Release plugin resources
    void terminate() noexcept override;

    //! @brief Destroy plugin instance
    void destroy() noexcept override;

protected:
    std::string mLayerName; //!< Layer name
    std::string mNamespace; //!< Plugin namespace

    int32_t mGemmN{};     //!< Output dimension (N)
    int32_t mGemmK{};     //!< Input dimension (K)
    int32_t mGroupSize{}; //!< Quantization group size
};

/*!
 * @brief Factory for creating Int4GroupwiseGemmPlugin instances
 *
 * Handles plugin registration and creation in TensorRT.
 */
class Int4GroupwiseGemmPluginCreator : public nvinfer1::IPluginCreator
{
public:
    //! @brief Constructor
    Int4GroupwiseGemmPluginCreator();

    //! @brief Destructor
    ~Int4GroupwiseGemmPluginCreator() override = default;

    //! @brief Get plugin name
    //! @return Plugin name string
    char const* getPluginName() const noexcept override;

    //! @brief Get plugin field names
    //! @return Field collection
    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override;

    //! @brief Set plugin namespace
    //! @param pluginNamespace Namespace string
    void setPluginNamespace(char const* pluginNamespace) noexcept;

    //! @brief Get plugin namespace
    //! @return Namespace string
    char const* getPluginNamespace() const noexcept override;

    //! @brief Get plugin version
    //! @return Version string
    char const* getPluginVersion() const noexcept override;

    //! @brief Create plugin from field collection
    //! @param name Plugin name
    //! @param fc Field collection with parameters
    //! @return Created plugin instance
    nvinfer1::IPluginV2* createPlugin(char const* name, nvinfer1::PluginFieldCollection const* fc) noexcept override;

    //! @brief Deserialize plugin from data
    //! @param name Plugin name
    //! @param serialData Serialized data
    //! @param serialLength Data size
    //! @return Deserialized plugin instance
    nvinfer1::IPluginV2* deserializePlugin(
        char const* name, void const* serialData, size_t serialLength) noexcept override;

private:
    static nvinfer1::PluginFieldCollection mFieldCollection;     //!< Field collection
    static std::vector<nvinfer1::PluginField> mPluginAttributes; //!< Plugin attributes
    std::string mNamespace;                                      //!< Plugin namespace
};

} // namespace plugins
} // namespace trt_edgellm