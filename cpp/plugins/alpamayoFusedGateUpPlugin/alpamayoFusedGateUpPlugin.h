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

#include <cstddef>
#include <cstdint>
#include <string>

namespace trt_edgellm
{
namespace plugins
{

//! Alpamayo 2B prefill FFN front half on SM110: out = FP8(silu(gate) * up), gate/up = x[M, 2048] @ W^T.
//! Inputs: x (FP8 codes, scale sx), W [12288, 2048] (FP8 codes; rows packed as 24 blocks of
//! [gate 128, up 128, gate 128, up 128], gate scale sg, up scale su), gateDecode = sx * sg and
//! upEncode = sx * su / so (FP32 [1]). Output: FP8 codes [M, 6144] with scale so.
//! M is dynamic; hidden and intermediate dimensions are fixed to the Alpamayo 2B shapes.
class AlpamayoFusedGateUpPlugin : public nvinfer1::IPluginV3,
                                  public nvinfer1::IPluginV3OneCore,
                                  public nvinfer1::IPluginV3OneBuild,
                                  public nvinfer1::IPluginV3OneRuntime
{
public:
    explicit AlpamayoFusedGateUpPlugin(std::string const& name);
    AlpamayoFusedGateUpPlugin() = delete;
    AlpamayoFusedGateUpPlugin(AlpamayoFusedGateUpPlugin const&) = delete;
    ~AlpamayoFusedGateUpPlugin() noexcept override = default;

    nvinfer1::IPluginCapability* getCapabilityInterface(nvinfer1::PluginCapabilityType type) noexcept override;
    nvinfer1::IPluginV3* clone() noexcept override;

    char const* getPluginName() const noexcept override;
    char const* getPluginVersion() const noexcept override;
    char const* getPluginNamespace() const noexcept override;
    void setPluginNamespace(char const* pluginNamespace) noexcept;

    int32_t getNbOutputs() const noexcept override;
    int32_t getOutputDataTypes(nvinfer1::DataType* outputTypes, int32_t nbOutputs, nvinfer1::DataType const* inputTypes,
        int32_t nbInputs) const noexcept override;
    int32_t getOutputShapes(nvinfer1::DimsExprs const* inputs, int32_t nbInputs, nvinfer1::DimsExprs const* shapeInputs,
        int32_t nbShapeInputs, nvinfer1::DimsExprs* outputs, int32_t nbOutputs,
        nvinfer1::IExprBuilder& exprBuilder) noexcept override;
    bool supportsFormatCombination(int32_t pos, nvinfer1::DynamicPluginTensorDesc const* inOut, int32_t nbInputs,
        int32_t nbOutputs) noexcept override;
    int32_t configurePlugin(nvinfer1::DynamicPluginTensorDesc const* inputs, int32_t nbInputs,
        nvinfer1::DynamicPluginTensorDesc const* outputs, int32_t nbOutputs) noexcept override;
    size_t getWorkspaceSize(nvinfer1::DynamicPluginTensorDesc const* inputs, int32_t nbInputs,
        nvinfer1::DynamicPluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept override;

    int32_t enqueue(nvinfer1::PluginTensorDesc const* inputDesc, nvinfer1::PluginTensorDesc const* outputDesc,
        void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;
    int32_t onShapeChange(nvinfer1::PluginTensorDesc const* inputs, int32_t nbInputs,
        nvinfer1::PluginTensorDesc const* outputs, int32_t nbOutputs) noexcept override;
    nvinfer1::IPluginV3* attachToContext(nvinfer1::IPluginResourceContext* context) noexcept override;
    nvinfer1::PluginFieldCollection const* getFieldsToSerialize() noexcept override;

private:
    std::string mLayerName;
    std::string mNamespace;
    nvinfer1::PluginFieldCollection mFieldsToSerialize{};
};

//! Creator for AlpamayoFusedGateUpPlugin version 1 (no fields).
class AlpamayoFusedGateUpPluginCreator : public nvinfer1::IPluginCreatorV3One
{
public:
    AlpamayoFusedGateUpPluginCreator() = default;
    ~AlpamayoFusedGateUpPluginCreator() override = default;

    char const* getPluginName() const noexcept override;
    char const* getPluginVersion() const noexcept override;
    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override;
    char const* getPluginNamespace() const noexcept override;
    void setPluginNamespace(char const* pluginNamespace) noexcept;
    nvinfer1::IPluginV3* createPlugin(char const* name, nvinfer1::PluginFieldCollection const* fields,
        nvinfer1::TensorRTPhase phase) noexcept override;

private:
    nvinfer1::PluginFieldCollection mFieldCollection{};
    std::string mNamespace;
};

} // namespace plugins
} // namespace trt_edgellm
