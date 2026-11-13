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

#include "q8_0LinearPlugin.h"
#include "common/logger.h"
#include "kernels/q8_0/q8_0Linear.h"

#include <array>
#include <cstdint>
#include <cstring>
#include <exception>
#include <limits>
#include <utility>

using namespace nvinfer1;

namespace trt_edgellm::plugins
{
// The kernel header also declares trt_edgellm::DataType; the plugin interface speaks TensorRT types.
using DataType = nvinfer1::DataType;

namespace
{
constexpr char const* kName{"Q8_0LinearPlugin"};
constexpr char const* kVersion{"1"};
//! Workspace layout recorded in the engine. The plugin version stays "1" because ONNX nodes and the direct
//! builder look creators up by that version; the serialized layout is what protects old engines.
constexpr char const* kACTIVATION_LAYOUT_FIELD{"activation_layout"};
constexpr int32_t kACTIVATION_LAYOUT_ABSENT{-1};

// Inputs are the activation followed by 1..kMaxWeights packed weights (for example Q, K and V); weight w fills the
// next weight.d[0] output columns.
constexpr int32_t kMaxWeights = kernel::kQ8_0_MAX_WEIGHTS;

bool validInputCount(int32_t nbInputs, int32_t nbOutputs)
{
    return nbInputs >= 2 && nbInputs <= 1 + kMaxWeights && nbOutputs == 1;
}

template <typename Desc, typename GetDims>
bool validShape(Desc const* in, int32_t nbInputs, Dims const& output, GetDims dims)
{
    constexpr int32_t blockSize = 32;
    constexpr int32_t blockBytes = 34;
    Dims const& x = dims(in[0]);
    if ((x.nbDims != 2 && x.nbDims != 3) || output.nbDims != x.nbDims)
    {
        return false;
    }
    for (int32_t axis = 0; axis < x.nbDims - 1; ++axis)
    {
        if (x.d[axis] <= 0 || x.d[axis] != output.d[axis])
        {
            return false;
        }
    }
    int64_t columns = 0;
    for (int32_t i = 1; i < nbInputs; ++i)
    {
        Dims const& weight = dims(in[i]);
        if (weight.nbDims != 3 || weight.d[2] != blockBytes || weight.d[0] <= 0 || weight.d[1] <= 0
            || weight.d[1] % 16 != 0 || weight.d[1] > std::numeric_limits<int32_t>::max() / blockSize
            || x.d[x.nbDims - 1] != weight.d[1] * blockSize)
        {
            return false;
        }
        columns += weight.d[0];
    }
    return output.d[output.nbDims - 1] == columns && columns <= std::numeric_limits<int32_t>::max();
}

bool validInputType(DataType type)
{
    return type == DataType::kHALF || type == DataType::kBF16;
}

bool validOutputType(DataType type)
{
    return type == DataType::kHALF || type == DataType::kBF16 || type == DataType::kFLOAT;
}

bool validTypes(PluginTensorDesc const* in, int32_t nbInputs, PluginTensorDesc const& output, DataType outputType)
{
    if (!validInputType(in[0].type) || in[0].format != PluginFormat::kLINEAR || output.type != outputType
        || output.format != PluginFormat::kLINEAR)
    {
        return false;
    }
    for (int32_t i = 1; i < nbInputs; ++i)
    {
        if (in[i].type != DataType::kINT8 || in[i].format != PluginFormat::kLINEAR)
        {
            return false;
        }
    }
    return true;
}

int64_t activationRows(Dims const& x)
{
    int64_t rows = 1;
    for (int32_t axis = 0; axis < x.nbDims - 1; ++axis)
    {
        rows *= x.d[axis];
    }
    return rows;
}
} // namespace

REGISTER_TENSORRT_PLUGIN(Q8_0LinearPluginCreator);

Q8_0LinearPlugin::Q8_0LinearPlugin(std::string name, DataType outputType)
    : mName(std::move(name))
    , mOutputType(static_cast<int32_t>(outputType))
    , mActivationLayout(kernel::kQ8_0_ACTIVATION_LAYOUT_VERSION)
{
    mAttributes.emplace_back("out_dtype", &mOutputType, PluginFieldType::kINT32, 1);
    mAttributes.emplace_back(kACTIVATION_LAYOUT_FIELD, &mActivationLayout, PluginFieldType::kINT32, 1);
    mFields = {static_cast<int32_t>(mAttributes.size()), mAttributes.data()};
}

IPluginCapability* Q8_0LinearPlugin::getCapabilityInterface(PluginCapabilityType type) noexcept
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

IPluginV3* Q8_0LinearPlugin::clone() noexcept
{
    try
    {
        auto* copy = new Q8_0LinearPlugin(mName, static_cast<DataType>(mOutputType));
        copy->mNbInputs = mNbInputs;
        copy->setPluginNamespace(mNamespace.c_str());
        return copy;
    }
    catch (...)
    {
        return nullptr;
    }
}

char const* Q8_0LinearPlugin::getPluginName() const noexcept
{
    return kName;
}

char const* Q8_0LinearPlugin::getPluginVersion() const noexcept
{
    return kVersion;
}

char const* Q8_0LinearPlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

void Q8_0LinearPlugin::setPluginNamespace(char const* pluginNamespace) noexcept
{
    mNamespace = pluginNamespace != nullptr ? pluginNamespace : "";
}

int32_t Q8_0LinearPlugin::getNbOutputs() const noexcept
{
    return 1;
}

int32_t Q8_0LinearPlugin::getOutputDataTypes(
    DataType* outputTypes, int32_t nbOutputs, DataType const* inputTypes, int32_t nbInputs) const noexcept
{
    if (outputTypes == nullptr || inputTypes == nullptr || !validInputCount(nbInputs, nbOutputs)
        || !validInputType(inputTypes[0]))
    {
        return -1;
    }
    for (int32_t i = 1; i < nbInputs; ++i)
    {
        if (inputTypes[i] != DataType::kINT8)
        {
            return -1;
        }
    }
    outputTypes[0] = static_cast<DataType>(mOutputType);
    return 0;
}

int32_t Q8_0LinearPlugin::getOutputShapes(DimsExprs const* inputs, int32_t nbInputs, DimsExprs const* /* shapeInputs */,
    int32_t /* nbShapeInputs */, DimsExprs* outputs, int32_t nbOutputs, IExprBuilder& exprBuilder) noexcept
{
    if (inputs == nullptr || outputs == nullptr || !validInputCount(nbInputs, nbOutputs)
        || (inputs[0].nbDims != 2 && inputs[0].nbDims != 3))
    {
        return -1;
    }
    outputs[0].nbDims = inputs[0].nbDims;
    for (int32_t axis = 0; axis < inputs[0].nbDims - 1; ++axis)
    {
        outputs[0].d[axis] = inputs[0].d[axis];
    }
    IDimensionExpr const* columns = nullptr;
    for (int32_t i = 1; i < nbInputs; ++i)
    {
        if (inputs[i].nbDims != 3)
        {
            return -1;
        }
        columns = columns == nullptr ? inputs[i].d[0]
                                     : exprBuilder.operation(DimensionOperation::kSUM, *columns, *inputs[i].d[0]);
    }
    outputs[0].d[inputs[0].nbDims - 1] = columns;
    return 0;
}

bool Q8_0LinearPlugin::supportsFormatCombination(
    int32_t pos, DynamicPluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
    if (inOut == nullptr || !validInputCount(nbInputs, nbOutputs) || pos < 0 || pos > nbInputs)
    {
        return false;
    }
    auto const& desc = inOut[pos].desc;
    if (desc.format != PluginFormat::kLINEAR)
    {
        return false;
    }
    if (pos == 0)
    {
        return validInputType(desc.type);
    }
    return pos < nbInputs ? desc.type == DataType::kINT8 : desc.type == static_cast<DataType>(mOutputType);
}

int32_t Q8_0LinearPlugin::configurePlugin(
    DynamicPluginTensorDesc const* in, int32_t nbInputs, DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept
{
    if (in == nullptr || out == nullptr || !validInputCount(nbInputs, nbOutputs))
    {
        return -1;
    }
    std::array<PluginTensorDesc, 1 + kMaxWeights> descs{};
    for (int32_t i = 0; i < nbInputs; ++i)
    {
        descs[i] = in[i].desc;
    }
    auto const minDims = [](DynamicPluginTensorDesc const& d) -> Dims const& { return d.min; };
    auto const optDims = [](DynamicPluginTensorDesc const& d) -> Dims const& { return d.opt; };
    auto const maxDims = [](DynamicPluginTensorDesc const& d) -> Dims const& { return d.max; };
    if (!validTypes(descs.data(), nbInputs, out[0].desc, static_cast<DataType>(mOutputType))
        || !validShape(in, nbInputs, out[0].min, minDims) || !validShape(in, nbInputs, out[0].opt, optDims)
        || !validShape(in, nbInputs, out[0].max, maxDims))
    {
        LOG_ERROR("%s: invalid activation type or packed INT8 weight geometry", getPluginName());
        return -1;
    }
    return 0;
}

size_t Q8_0LinearPlugin::getWorkspaceSize(DynamicPluginTensorDesc const* inputs, int32_t nbInputs,
    DynamicPluginTensorDesc const* /* outputs */, int32_t nbOutputs) const noexcept
{
    if (inputs == nullptr || !validInputCount(nbInputs, nbOutputs))
    {
        return 0;
    }
    Dims const& x = inputs[0].max;
    int64_t const rows = activationRows(x);
    if (rows <= 0 || rows > std::numeric_limits<int32_t>::max())
    {
        return 0;
    }
    // One quantized copy of the activation serves every weight.
    return kernel::q8_0WorkspaceSize(static_cast<int32_t>(rows), x.d[x.nbDims - 1]);
}

int32_t Q8_0LinearPlugin::enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc,
    void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    int32_t const nbInputs = mNbInputs;
    auto const dims = [](PluginTensorDesc const& d) -> Dims const& { return d.dims; };
    if (inputDesc == nullptr || outputDesc == nullptr || inputs == nullptr || outputs == nullptr
        || !validInputCount(nbInputs, 1)
        || !validTypes(inputDesc, nbInputs, outputDesc[0], static_cast<DataType>(mOutputType))
        || !validShape(inputDesc, nbInputs, outputDesc[0].dims, dims))
    {
        return -1;
    }
    Dims const& x = inputDesc[0].dims;
    int64_t const rows = activationRows(x);
    if (rows > std::numeric_limits<int32_t>::max())
    {
        return -1;
    }
    std::array<void const*, kMaxWeights> weights{};
    std::array<int32_t, kMaxWeights> outFeatures{};
    for (int32_t w = 0; w + 1 < nbInputs; ++w)
    {
        weights[w] = inputs[w + 1];
        outFeatures[w] = static_cast<int32_t>(inputDesc[w + 1].dims.d[0]);
    }
    return kernel::q8_0Linear(inputs[0], static_cast<trt_edgellm::DataType>(inputDesc[0].type), weights.data(),
               outFeatures.data(), nbInputs - 1, outputs[0], static_cast<trt_edgellm::DataType>(mOutputType), workspace,
               kernel::q8_0WorkspaceSize(static_cast<int32_t>(rows), x.d[x.nbDims - 1]), static_cast<int32_t>(rows),
               x.d[x.nbDims - 1], stream)
            == cudaSuccess
        ? 0
        : -1;
}

int32_t Q8_0LinearPlugin::onShapeChange(
    PluginTensorDesc const* in, int32_t nbInputs, PluginTensorDesc const* out, int32_t nbOutputs) noexcept
{
    auto const dims = [](PluginTensorDesc const& d) -> Dims const& { return d.dims; };
    if (in == nullptr || out == nullptr || !validInputCount(nbInputs, nbOutputs)
        || !validTypes(in, nbInputs, out[0], static_cast<DataType>(mOutputType))
        || !validShape(in, nbInputs, out[0].dims, dims))
    {
        return -1;
    }
    mNbInputs = nbInputs;
    return 0;
}

IPluginV3* Q8_0LinearPlugin::attachToContext(IPluginResourceContext* /* context */) noexcept
{
    return clone();
}

PluginFieldCollection const* Q8_0LinearPlugin::getFieldsToSerialize() noexcept
{
    return &mFields;
}

Q8_0LinearPluginCreator::Q8_0LinearPluginCreator()
{
    mAttributes.emplace_back("out_dtype", nullptr, PluginFieldType::kINT32, 1);
    mFields = {static_cast<int32_t>(mAttributes.size()), mAttributes.data()};
}

char const* Q8_0LinearPluginCreator::getPluginName() const noexcept
{
    return kName;
}

char const* Q8_0LinearPluginCreator::getPluginVersion() const noexcept
{
    return kVersion;
}

PluginFieldCollection const* Q8_0LinearPluginCreator::getFieldNames() noexcept
{
    return &mFields;
}

char const* Q8_0LinearPluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

void Q8_0LinearPluginCreator::setPluginNamespace(char const* pluginNamespace) noexcept
{
    mNamespace = pluginNamespace != nullptr ? pluginNamespace : "";
}

IPluginV3* Q8_0LinearPluginCreator::createPlugin(
    char const* name, PluginFieldCollection const* fc, TensorRTPhase phase) noexcept
{
    if (name == nullptr)
    {
        return nullptr;
    }
    // FP16 output unless the graph asks for BF16 (MoE latent input) or FP32 (logits).
    DataType outputType = DataType::kHALF;
    int32_t layout = kACTIVATION_LAYOUT_ABSENT;
    for (int32_t i = 0; fc != nullptr && i < fc->nbFields; ++i)
    {
        auto const& field = fc->fields[i];
        if (std::string(field.name) == "out_dtype" && field.type == PluginFieldType::kINT32 && field.length == 1
            && field.data != nullptr)
        {
            outputType = static_cast<DataType>(*static_cast<int32_t const*>(field.data));
        }
        else if (std::strcmp(field.name, kACTIVATION_LAYOUT_FIELD) == 0 && field.type == PluginFieldType::kINT32
            && field.length == 1 && field.data != nullptr)
        {
            layout = *static_cast<int32_t const*>(field.data);
        }
    }
    if (!validOutputType(outputType))
    {
        LOG_ERROR("Q8_0LinearPlugin: out_dtype must be FP16, BF16 or FP32");
        return nullptr;
    }
    // IPluginV3 enqueue cannot see the workspace an engine reserved, so a deserialized engine must carry the
    // current activation layout. ONNX nodes and builder calls carry none.
    bool const fromEngine = phase == TensorRTPhase::kRUNTIME;
    if ((fromEngine || layout != kACTIVATION_LAYOUT_ABSENT) && layout != kernel::kQ8_0_ACTIVATION_LAYOUT_VERSION)
    {
        LOG_ERROR("%s: engine built with an older plugin (activation_layout %d, expected %d); rebuild the engine",
            kName, layout, kernel::kQ8_0_ACTIVATION_LAYOUT_VERSION);
        return nullptr;
    }
    try
    {
        auto* plugin = new Q8_0LinearPlugin(name, outputType);
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }
    catch (std::exception const& error)
    {
        LOG_ERROR("Q8_0LinearPlugin creation failed: %s", error.what());
        return nullptr;
    }
}

} // namespace trt_edgellm::plugins
