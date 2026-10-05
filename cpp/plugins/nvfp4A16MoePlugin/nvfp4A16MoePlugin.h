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

#include <NvInferRuntime.h>

#include <cstdint>
#include <string>
#include <vector>

namespace trt_edgellm
{
namespace plugins
{

/*!
 * @brief TensorRT V3 plugin for FP16/BF16 activations and Marlin-packed NVFP4 expert weights.
 *
 * max_routed_rows is the capacity of the padded Marlin routing arrays, not only the number of selected slots.
 * A runtime shape with T tokens requires at most T * top_k + num_experts * (block_size - 1) rows, where block_size
 * is 8 for T <= 8 and 32 otherwise.
 */
class Nvfp4A16MoePlugin : public nvinfer1::IPluginV3,
                          public nvinfer1::IPluginV3OneCore,
                          public nvinfer1::IPluginV3OneBuild,
                          public nvinfer1::IPluginV3OneRuntime
{
public:
    Nvfp4A16MoePlugin(std::string const& name, int32_t numExperts, int32_t topK, int32_t hiddenSize,
        int32_t moeInterSize, int32_t activationType, int32_t nGroup, int32_t topkGroup, int32_t normTopkProb,
        float routedScalingFactor, int32_t routingMode, int32_t maxRoutedRows);

    Nvfp4A16MoePlugin(std::string const& name, nvinfer1::PluginFieldCollection const* fc);

    Nvfp4A16MoePlugin() = delete;
    Nvfp4A16MoePlugin(Nvfp4A16MoePlugin const&) = delete;
    Nvfp4A16MoePlugin& operator=(Nvfp4A16MoePlugin const&) = delete;
    ~Nvfp4A16MoePlugin() noexcept override;

    nvinfer1::IPluginCapability* getCapabilityInterface(nvinfer1::PluginCapabilityType type) noexcept override;
    nvinfer1::IPluginV3* clone() noexcept override;

    char const* getPluginName() const noexcept override;
    char const* getPluginVersion() const noexcept override;
    char const* getPluginNamespace() const noexcept override;
    int32_t getNbOutputs() const noexcept override;

    int32_t getOutputDataTypes(nvinfer1::DataType* outputTypes, int32_t nbOutputs, nvinfer1::DataType const* inputTypes,
        int32_t nbInputs) const noexcept override;

    int32_t getOutputShapes(nvinfer1::DimsExprs const* inputs, int32_t nbInputs, nvinfer1::DimsExprs const* shapeInputs,
        int32_t nbShapeInputs, nvinfer1::DimsExprs* outputs, int32_t nbOutputs,
        nvinfer1::IExprBuilder& exprBuilder) noexcept override;

    bool supportsFormatCombination(int32_t pos, nvinfer1::DynamicPluginTensorDesc const* inOut, int32_t nbInputs,
        int32_t nbOutputs) noexcept override;

    int32_t configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in, int32_t nbInputs,
        nvinfer1::DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept override;

    size_t getWorkspaceSize(nvinfer1::DynamicPluginTensorDesc const* inputs, int32_t nbInputs,
        nvinfer1::DynamicPluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept override;

    int32_t enqueue(nvinfer1::PluginTensorDesc const* inputDesc, nvinfer1::PluginTensorDesc const* outputDesc,
        void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;

    int32_t onShapeChange(nvinfer1::PluginTensorDesc const* in, int32_t nbInputs, nvinfer1::PluginTensorDesc const* out,
        int32_t nbOutputs) noexcept override;

    nvinfer1::IPluginV3* attachToContext(nvinfer1::IPluginResourceContext* context) noexcept override;
    nvinfer1::PluginFieldCollection const* getFieldsToSerialize() noexcept override;

    void setPluginNamespace(char const* pluginNamespace) noexcept;

private:
    void validateAttributes() const;
    bool validateTensorDesc(int32_t pos, nvinfer1::PluginTensorDesc const& desc) const noexcept;

    std::string mLayerName;
    std::string mNamespace;
    int32_t mNumExperts{};
    int32_t mTopK{};
    int32_t mHiddenSize{};
    int32_t mMoeInterSize{};
    int32_t mActivationType{};
    int32_t mNGroup{};
    int32_t mTopkGroup{};
    int32_t mNormTopkProb{};
    float mRoutedScalingFactor{};
    int32_t mRoutingMode{};
    int32_t mMaxRoutedRows{}; //!< Padded Marlin routed-row capacity.

    std::vector<nvinfer1::PluginField> mDataToSerialize;
    nvinfer1::PluginFieldCollection mFCToSerialize{};
};

//! Creator for Nvfp4A16MoePlugin.
class Nvfp4A16MoePluginCreator : public nvinfer1::IPluginCreatorV3One
{
public:
    Nvfp4A16MoePluginCreator();
    ~Nvfp4A16MoePluginCreator() override = default;

    char const* getPluginName() const noexcept override;
    char const* getPluginVersion() const noexcept override;
    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override;
    char const* getPluginNamespace() const noexcept override;
    void setPluginNamespace(char const* pluginNamespace) noexcept;

    nvinfer1::IPluginV3* createPlugin(
        char const* name, nvinfer1::PluginFieldCollection const* fc, nvinfer1::TensorRTPhase phase) noexcept override;

private:
    static nvinfer1::PluginFieldCollection mFieldCollection;
    static std::vector<nvinfer1::PluginField> mPluginAttributes;
    std::string mNamespace;
};

} // namespace plugins
} // namespace trt_edgellm
