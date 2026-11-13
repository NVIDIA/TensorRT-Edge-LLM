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

#include "pleGatherPlugin.h"
#include "common/logger.h"
#include "kernels/embeddingKernels/pleGather.h"

#include <limits>
#include <new>

using namespace nvinfer1;

namespace trt_edgellm
{
namespace plugins
{
using nvinfer1::DataType;

namespace
{
constexpr char kNAME[]{"PleGatherPlugin"};
constexpr char kVERSION[]{"1"};

bool isStorageType(DataType type)
{
    return type == DataType::kFP8 || type == DataType::kHALF || type == DataType::kBF16;
}

bool validShape(Dims const& pool, Dims const& map, Dims const& output)
{
    if (pool.nbDims != 2 || (map.nbDims != 2 && map.nbDims != 3) || output.nbDims != map.nbDims + 1 || pool.d[0] <= 0
        || pool.d[0] > std::numeric_limits<int32_t>::max() || pool.d[1] <= 0
        || pool.d[1] > std::numeric_limits<int32_t>::max())
    {
        return false;
    }
    int64_t bytes = pool.d[1] * 2;
    for (int32_t axis = 0; axis < map.nbDims; ++axis)
    {
        if (map.d[axis] <= 0 || map.d[axis] > std::numeric_limits<int64_t>::max() / bytes
            || output.d[axis] != map.d[axis])
        {
            return false;
        }
        bytes *= map.d[axis];
    }
    return output.d[map.nbDims] == pool.d[1];
}

bool validFormat(PluginTensorDesc const& desc, int32_t position)
{
    int32_t const rank = desc.dims.nbDims;
    bool const validRank = position == 0 ? rank == 2 : (rank == position + 1 || rank == position + 2);
    return desc.format == TensorFormat::kLINEAR && validRank
        && (position == 1 ? desc.type == DataType::kINT32 : isStorageType(desc.type));
}

bool validContract(PluginTensorDesc const* inputs, PluginTensorDesc const* outputs)
{
    return inputs != nullptr && outputs != nullptr && validFormat(inputs[0], 0) && validFormat(inputs[1], 1)
        && validFormat(outputs[0], 2) && inputs[0].type == outputs[0].type
        && validShape(inputs[0].dims, inputs[1].dims, outputs[0].dims);
}
} // namespace

REGISTER_TENSORRT_PLUGIN(PleGatherPluginCreator);

IPluginCapability* PleGatherPlugin::getCapabilityInterface(PluginCapabilityType type) noexcept
{
    if (type == PluginCapabilityType::kBUILD)
    {
        return static_cast<IPluginV3OneBuild*>(this);
    }
    if (type == PluginCapabilityType::kRUNTIME)
    {
        return static_cast<IPluginV3OneRuntime*>(this);
    }
    if (type == PluginCapabilityType::kCORE)
    {
        return static_cast<IPluginV3OneCore*>(this);
    }
    return nullptr;
}

IPluginV3* PleGatherPlugin::clone() noexcept
{
    return new (std::nothrow) PleGatherPlugin;
}

char const* PleGatherPlugin::getPluginName() const noexcept
{
    return kNAME;
}

char const* PleGatherPlugin::getPluginVersion() const noexcept
{
    return kVERSION;
}

char const* PleGatherPlugin::getPluginNamespace() const noexcept
{
    return "";
}

int32_t PleGatherPlugin::getNbOutputs() const noexcept
{
    return 1;
}

int32_t PleGatherPlugin::getOutputDataTypes(
    DataType* outputTypes, int32_t nbOutputs, DataType const* inputTypes, int32_t nbInputs) const noexcept
{
    if (outputTypes == nullptr || inputTypes == nullptr || nbOutputs != 1 || nbInputs != 2
        || !isStorageType(inputTypes[0]) || inputTypes[1] != DataType::kINT32)
    {
        return -1;
    }
    outputTypes[0] = inputTypes[0];
    return 0;
}

int32_t PleGatherPlugin::getOutputShapes(DimsExprs const* inputs, int32_t nbInputs, DimsExprs const* /* shapeInputs */,
    int32_t nbShapeInputs, DimsExprs* outputs, int32_t nbOutputs, IExprBuilder& /* exprBuilder */) noexcept
{
    if (inputs == nullptr || outputs == nullptr || nbInputs != 2 || nbOutputs != 1 || nbShapeInputs != 0
        || inputs[0].nbDims != 2 || (inputs[1].nbDims != 2 && inputs[1].nbDims != 3))
    {
        return -1;
    }
    for (int32_t input = 0; input < nbInputs; ++input)
    {
        for (int32_t dim = 0; dim < inputs[input].nbDims; ++dim)
        {
            if (inputs[input].d[dim] == nullptr)
            {
                return -1;
            }
        }
    }
    int32_t const mapRank = inputs[1].nbDims;
    if (!inputs[0].d[1]->isConstant() || inputs[0].d[1]->getConstantValue() <= 0
        || inputs[0].d[1]->getConstantValue() > std::numeric_limits<int32_t>::max()
        || !inputs[1].d[mapRank - 1]->isConstant() || inputs[1].d[mapRank - 1]->getConstantValue() <= 0)
    {
        return -1;
    }
    outputs[0].nbDims = mapRank + 1;
    for (int32_t axis = 0; axis < mapRank; ++axis)
    {
        outputs[0].d[axis] = inputs[1].d[axis];
    }
    outputs[0].d[mapRank] = inputs[0].d[1];
    return 0;
}

bool PleGatherPlugin::supportsFormatCombination(
    int32_t pos, DynamicPluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
    return inOut != nullptr && nbInputs == 2 && nbOutputs == 1 && pos >= 0 && pos < 3
        && validFormat(inOut[pos].desc, pos)
        && (pos != 2
            || (inOut[2].desc.type == inOut[0].desc.type
                && inOut[2].desc.dims.nbDims == inOut[1].desc.dims.nbDims + 1));
}

int32_t PleGatherPlugin::configurePlugin(
    DynamicPluginTensorDesc const* in, int32_t nbInputs, DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept
{
    if (in == nullptr || out == nullptr || nbInputs != 2 || nbOutputs != 1 || !validFormat(in[0].desc, 0)
        || !validFormat(in[1].desc, 1) || !validFormat(out[0].desc, 2) || in[0].desc.type != out[0].desc.type
        || !validShape(in[0].min, in[1].min, out[0].min) || !validShape(in[0].opt, in[1].opt, out[0].opt)
        || !validShape(in[0].max, in[1].max, out[0].max))
    {
        LOG_ERROR("PleGatherPlugin requires a LINEAR FP8/FP16/BF16 [C,W] pool and INT32 [T,H] or [B,S,H] row map.");
        return -1;
    }
    return 0;
}

size_t PleGatherPlugin::getWorkspaceSize(DynamicPluginTensorDesc const* /* inputs */, int32_t /* nbInputs */,
    DynamicPluginTensorDesc const* /* outputs */, int32_t /* nbOutputs */) const noexcept
{
    return 0;
}

int32_t PleGatherPlugin::enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc,
    void const* const* inputs, void* const* outputs, void* /* workspace */, cudaStream_t stream) noexcept
{
    if (!validContract(inputDesc, outputDesc) || inputs == nullptr || outputs == nullptr)
    {
        LOG_ERROR("PleGatherPlugin received invalid runtime descriptors.");
        return -1;
    }
    int64_t rows = 1;
    for (int32_t axis = 0; axis < inputDesc[1].dims.nbDims; ++axis)
    {
        rows *= inputDesc[1].dims.d[axis];
    }
    return kernel::launchPleGather(inputs[0], static_cast<int32_t const*>(inputs[1]), outputs[0], rows,
               inputDesc[0].dims.d[1] * (inputDesc[0].type == DataType::kFP8 ? 1 : 2), stream)
            == cudaSuccess
        ? 0
        : -1;
}

int32_t PleGatherPlugin::onShapeChange(
    PluginTensorDesc const* in, int32_t nbInputs, PluginTensorDesc const* out, int32_t nbOutputs) noexcept
{
    if (nbInputs != 2 || nbOutputs != 1 || !validContract(in, out))
    {
        LOG_ERROR("PleGatherPlugin received invalid runtime shapes or types.");
        return -1;
    }
    return 0;
}

IPluginV3* PleGatherPlugin::attachToContext(IPluginResourceContext* /* context */) noexcept
{
    return clone();
}

PluginFieldCollection const* PleGatherPlugin::getFieldsToSerialize() noexcept
{
    return &mFields;
}

char const* PleGatherPluginCreator::getPluginName() const noexcept
{
    return kNAME;
}

char const* PleGatherPluginCreator::getPluginVersion() const noexcept
{
    return kVERSION;
}

char const* PleGatherPluginCreator::getPluginNamespace() const noexcept
{
    return "";
}

PluginFieldCollection const* PleGatherPluginCreator::getFieldNames() noexcept
{
    return &mFields;
}

IPluginV3* PleGatherPluginCreator::createPlugin(
    char const* /* name */, PluginFieldCollection const* fields, TensorRTPhase /* phase */) noexcept
{
    if (fields != nullptr && fields->nbFields != 0)
    {
        return nullptr;
    }
    return new (std::nothrow) PleGatherPlugin;
}

} // namespace plugins
} // namespace trt_edgellm
