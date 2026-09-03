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

#include "qkvConcatPlugin.h"
#include "common/logger.h"

#include <cstddef>
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <exception>
#include <limits>

using namespace nvinfer1;

namespace trt_edgellm::plugins
{
namespace
{
constexpr char const* kPLUGIN_NAME{"QkvConcatPlugin"};
constexpr char const* kPLUGIN_VERSION{"1"};
constexpr int32_t kNUM_INPUTS{3};
constexpr int32_t kNUM_OUTPUTS{1};

bool hasPackedShape(Dims const& q, Dims const& k, Dims const& v, Dims const& output)
{
    if (q.nbDims != 3 || k.nbDims != 3 || v.nbDims != 3 || output.nbDims != 3)
    {
        return false;
    }
    if (q.d[0] <= 0 || q.d[1] <= 0 || q.d[2] <= 0 || k.d[2] <= 0 || v.d[2] <= 0)
    {
        return false;
    }
    int64_t const outputWidth = static_cast<int64_t>(q.d[2]) + k.d[2] + v.d[2];
    return q.d[0] == k.d[0] && q.d[0] == v.d[0] && q.d[0] == output.d[0] && q.d[1] == k.d[1] && q.d[1] == v.d[1]
        && q.d[1] == output.d[1] && outputWidth <= std::numeric_limits<int32_t>::max() && output.d[2] == outputWidth;
}

bool isFp16Linear(PluginTensorDesc const& desc)
{
    return desc.type == DataType::kHALF && desc.format == PluginFormat::kLINEAR && desc.dims.nbDims == 3;
}

bool hasConcreteContract(PluginTensorDesc const* inputs, PluginTensorDesc const* output)
{
    if (inputs == nullptr || output == nullptr || !isFp16Linear(output[0]))
    {
        return false;
    }
    for (int32_t inputIdx = 0; inputIdx < kNUM_INPUTS; ++inputIdx)
    {
        if (!isFp16Linear(inputs[inputIdx]))
        {
            return false;
        }
    }
    return hasPackedShape(inputs[0].dims, inputs[1].dims, inputs[2].dims, output[0].dims);
}
} // namespace

REGISTER_TENSORRT_PLUGIN(QkvConcatPluginCreator);

QkvConcatPlugin::QkvConcatPlugin(std::string const& name)
    : mLayerName(name)
{
}

IPluginCapability* QkvConcatPlugin::getCapabilityInterface(PluginCapabilityType type) noexcept
{
    if (type == PluginCapabilityType::kBUILD)
    {
        return static_cast<IPluginV3OneBuild*>(this);
    }
    if (type == PluginCapabilityType::kRUNTIME)
    {
        return static_cast<IPluginV3OneRuntime*>(this);
    }
    return static_cast<IPluginV3OneCore*>(this);
}

IPluginV3* QkvConcatPlugin::clone() noexcept
{
    try
    {
        auto* plugin = new QkvConcatPlugin(mLayerName);
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }
    catch (...)
    {
        return nullptr;
    }
}

char const* QkvConcatPlugin::getPluginName() const noexcept
{
    return kPLUGIN_NAME;
}

char const* QkvConcatPlugin::getPluginVersion() const noexcept
{
    return kPLUGIN_VERSION;
}

char const* QkvConcatPlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

void QkvConcatPlugin::setPluginNamespace(char const* pluginNamespace) noexcept
{
    mNamespace = pluginNamespace != nullptr ? pluginNamespace : "";
}

int32_t QkvConcatPlugin::getNbOutputs() const noexcept
{
    return kNUM_OUTPUTS;
}

int32_t QkvConcatPlugin::getOutputDataTypes(
    DataType* outputTypes, int32_t nbOutputs, DataType const* inputTypes, int32_t nbInputs) const noexcept
{
    if (outputTypes == nullptr || inputTypes == nullptr || nbOutputs != kNUM_OUTPUTS || nbInputs != kNUM_INPUTS
        || inputTypes[0] != DataType::kHALF || inputTypes[1] != inputTypes[0] || inputTypes[2] != inputTypes[0])
    {
        return -1;
    }
    outputTypes[0] = inputTypes[0];
    return 0;
}

int32_t QkvConcatPlugin::getOutputShapes(DimsExprs const* inputs, int32_t nbInputs, DimsExprs const* /* shapeInputs */,
    int32_t /* nbShapeInputs */, DimsExprs* outputs, int32_t nbOutputs, IExprBuilder& exprBuilder) noexcept
{
    if (inputs == nullptr || outputs == nullptr || nbInputs != kNUM_INPUTS || nbOutputs != kNUM_OUTPUTS
        || inputs[0].nbDims != 3 || inputs[1].nbDims != 3 || inputs[2].nbDims != 3 || inputs[0].d[0] == nullptr
        || inputs[0].d[1] == nullptr || inputs[0].d[2] == nullptr || inputs[1].d[2] == nullptr
        || inputs[2].d[2] == nullptr)
    {
        return -1;
    }

    outputs[0].nbDims = 3;
    outputs[0].d[0] = inputs[0].d[0];
    outputs[0].d[1] = inputs[0].d[1];
    auto const* qk = exprBuilder.operation(DimensionOperation::kSUM, *inputs[0].d[2], *inputs[1].d[2]);
    outputs[0].d[2] = exprBuilder.operation(DimensionOperation::kSUM, *qk, *inputs[2].d[2]);
    return 0;
}

bool QkvConcatPlugin::supportsFormatCombination(
    int32_t pos, DynamicPluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
    if (inOut == nullptr || nbInputs != kNUM_INPUTS || nbOutputs != kNUM_OUTPUTS || pos < 0
        || pos >= nbInputs + nbOutputs)
    {
        return false;
    }
    auto const& desc = inOut[pos].desc;
    return desc.type == DataType::kHALF && desc.format == PluginFormat::kLINEAR && desc.dims.nbDims == 3;
}

int32_t QkvConcatPlugin::configurePlugin(
    DynamicPluginTensorDesc const* in, int32_t nbInputs, DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept
{
    if (in == nullptr || out == nullptr || nbInputs != kNUM_INPUTS || nbOutputs != kNUM_OUTPUTS)
    {
        LOG_ERROR("QkvConcatPlugin: expected three inputs and one output");
        return -1;
    }
    for (int32_t inputIdx = 0; inputIdx < kNUM_INPUTS; ++inputIdx)
    {
        if (!isFp16Linear(in[inputIdx].desc))
        {
            LOG_ERROR("QkvConcatPlugin: all inputs must be rank-3 FP16 linear tensors");
            return -1;
        }
    }
    if (!isFp16Linear(out[0].desc) || !hasPackedShape(in[0].min, in[1].min, in[2].min, out[0].min)
        || !hasPackedShape(in[0].opt, in[1].opt, in[2].opt, out[0].opt)
        || !hasPackedShape(in[0].max, in[1].max, in[2].max, out[0].max))
    {
        LOG_ERROR("QkvConcatPlugin: Q/K/V leading dimensions and packed output shape must match for every profile");
        return -1;
    }
    return 0;
}

size_t QkvConcatPlugin::getWorkspaceSize(DynamicPluginTensorDesc const* /* inputs */, int32_t /* nbInputs */,
    DynamicPluginTensorDesc const* /* outputs */, int32_t /* nbOutputs */) const noexcept
{
    return 0;
}

int32_t QkvConcatPlugin::enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc,
    void const* const* inputs, void* const* outputs, void* /* workspace */, cudaStream_t stream) noexcept
{
    if (!hasConcreteContract(inputDesc, outputDesc) || inputs == nullptr || outputs == nullptr || outputs[0] == nullptr
        || inputs[0] == nullptr || inputs[1] == nullptr || inputs[2] == nullptr)
    {
        LOG_ERROR("QkvConcatPlugin: invalid runtime descriptors or null buffers");
        return -1;
    }
    int64_t const rows = static_cast<int64_t>(inputDesc[0].dims.d[0]) * inputDesc[0].dims.d[1];
    int64_t const qWidth = inputDesc[0].dims.d[2];
    int64_t const kWidth = inputDesc[1].dims.d[2];
    int64_t const vWidth = inputDesc[2].dims.d[2];
    int64_t const outputWidth = qWidth + kWidth + vWidth;

    auto* output = static_cast<std::byte*>(outputs[0]);
    int64_t offset = 0;
    for (int32_t i = 0; i < kNUM_INPUTS; ++i)
    {
        int64_t const width = inputDesc[i].dims.d[2];
        cudaError_t const status = cudaMemcpy2DAsync(output + offset * sizeof(__half), outputWidth * sizeof(__half),
            inputs[i], width * sizeof(__half), width * sizeof(__half), rows, cudaMemcpyDeviceToDevice, stream);
        if (status != cudaSuccess)
        {
            return -1;
        }
        offset += width;
    }
    return 0;
}

int32_t QkvConcatPlugin::onShapeChange(
    PluginTensorDesc const* in, int32_t nbInputs, PluginTensorDesc const* out, int32_t nbOutputs) noexcept
{
    if (nbInputs != kNUM_INPUTS || nbOutputs != kNUM_OUTPUTS || !hasConcreteContract(in, out))
    {
        LOG_ERROR("QkvConcatPlugin: invalid runtime Q/K/V or packed output shape");
        return -1;
    }
    return 0;
}

IPluginV3* QkvConcatPlugin::attachToContext(IPluginResourceContext* /* context */) noexcept
{
    return clone();
}

PluginFieldCollection const* QkvConcatPlugin::getFieldsToSerialize() noexcept
{
    return &mFields;
}

char const* QkvConcatPluginCreator::getPluginName() const noexcept
{
    return kPLUGIN_NAME;
}

char const* QkvConcatPluginCreator::getPluginVersion() const noexcept
{
    return kPLUGIN_VERSION;
}

PluginFieldCollection const* QkvConcatPluginCreator::getFieldNames() noexcept
{
    return &mFields;
}

char const* QkvConcatPluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

void QkvConcatPluginCreator::setPluginNamespace(char const* pluginNamespace) noexcept
{
    mNamespace = pluginNamespace != nullptr ? pluginNamespace : "";
}

IPluginV3* QkvConcatPluginCreator::createPlugin(
    char const* name, PluginFieldCollection const* /* fc */, TensorRTPhase /* phase */) noexcept
{
    if (name == nullptr)
    {
        LOG_ERROR("QkvConcatPluginCreator: plugin name must not be null");
        return nullptr;
    }
    try
    {
        auto* plugin = new QkvConcatPlugin(name);
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }
    catch (std::exception const& error)
    {
        LOG_ERROR("QkvConcatPluginCreator: creation failed: %s", error.what());
        return nullptr;
    }
}

} // namespace trt_edgellm::plugins
