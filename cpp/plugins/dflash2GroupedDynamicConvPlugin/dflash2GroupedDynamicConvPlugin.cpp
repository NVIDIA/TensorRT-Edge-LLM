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

#include "dflash2GroupedDynamicConvPlugin.h"

#include "common/logger.h"
#include "kernels/speculative/dflash2GroupedDynamicConv.h"

#include <limits>
#include <mutex>
#include <stdexcept>

using namespace nvinfer1;

namespace trt_edgellm
{
namespace plugins
{
namespace
{
constexpr char const* kPLUGIN_NAME{"DFlash2GroupedDynamicConvPlugin"};
constexpr char const* kPLUGIN_VERSION{"1"};
constexpr char const* kFIELD_BLOCK{"block_size"};
constexpr char const* kFIELD_KERNEL{"kernel_size"};
constexpr char const* kFIELD_GROUP{"group_size"};
constexpr char const* kFIELD_FUSE_RESIDUAL{"fuse_residual"};
constexpr int32_t kBASE_INPUTS{3};
constexpr int32_t kNB_OUTPUTS{1};
constexpr int32_t kIN_HIDDEN{0};
constexpr int32_t kIN_DELTA{1};
constexpr int32_t kIN_BASE{2};
constexpr int32_t kIN_RESIDUAL{3};

int32_t expectedInputs(bool fuseResidual) noexcept
{
    return kBASE_INPUTS + (fuseResidual ? 1 : 0);
}

bool isActivationType(DataType type) noexcept
{
    return type == DataType::kHALF || type == DataType::kBF16;
}

int32_t getIntField(PluginFieldCollection const* fields, char const* name)
{
    if (fields == nullptr)
    {
        throw std::invalid_argument("DFlash2GroupedDynamicConvPlugin: null fields");
    }
    for (int32_t i = 0; i < fields->nbFields; ++i)
    {
        PluginField const& field = fields->fields[i];
        if (field.name != nullptr && std::string(field.name) == name)
        {
            if (field.type != PluginFieldType::kINT32 || field.length != 1 || field.data == nullptr)
            {
                throw std::invalid_argument(std::string("DFlash2GroupedDynamicConvPlugin: invalid field ") + name);
            }
            return *static_cast<int32_t const*>(field.data);
        }
    }
    throw std::invalid_argument(std::string("DFlash2GroupedDynamicConvPlugin: missing field ") + name);
}

void validateAttributes(int32_t blockSize, int32_t kernelSize, int32_t groupSize)
{
    if (blockSize <= 0 || blockSize > kernel::kDFlash2MaxBlockSize || kernelSize < 1 || kernelSize > 4 || groupSize <= 0
        || groupSize % 2 != 0)
    {
        throw std::invalid_argument(
            "DFlash2GroupedDynamicConvPlugin: require block_size in [1,16], kernel_size in [1,4], even group_size>0");
    }
}

bool dimsEqual(Dims const& lhs, Dims const& rhs) noexcept
{
    if (lhs.nbDims != rhs.nbDims)
    {
        return false;
    }
    for (int32_t i = 0; i < lhs.nbDims; ++i)
    {
        if (lhs.d[i] != rhs.d[i])
        {
            return false;
        }
    }
    return true;
}

bool validateShapes(Dims const& hidden, Dims const& delta, Dims const& base, Dims const* output, int32_t kernelSize,
    int32_t groupSize, bool allowZero) noexcept
{
    if (hidden.nbDims < 2 || hidden.nbDims > 4 || delta.nbDims != hidden.nbDims + 1 || base.nbDims != 2)
    {
        return false;
    }
    int32_t const hiddenSize = hidden.d[hidden.nbDims - 1];
    if (hiddenSize <= 0 || hiddenSize % groupSize != 0 || hiddenSize % 2 != 0 || base.d[0] != kernelSize
        || base.d[1] != hiddenSize || delta.d[delta.nbDims - 2] != kernelSize
        || delta.d[delta.nbDims - 1] != hiddenSize / groupSize)
    {
        return false;
    }
    int64_t tokens{1};
    for (int32_t i = 0; i < hidden.nbDims - 1; ++i)
    {
        if (hidden.d[i] < 0 || (!allowZero && hidden.d[i] == 0) || delta.d[i] != hidden.d[i]
            || (hidden.d[i] != 0 && tokens > std::numeric_limits<int32_t>::max() / hidden.d[i]))
        {
            return false;
        }
        tokens *= hidden.d[i];
    }
    return output == nullptr || dimsEqual(hidden, *output);
}

int32_t flattenedTokens(Dims const& hidden) noexcept
{
    int32_t tokens{1};
    for (int32_t i = 0; i < hidden.nbDims - 1; ++i)
    {
        tokens *= hidden.d[i];
    }
    return tokens;
}

int32_t runtimeBlockSize(Dims const& hidden) noexcept
{
    return hidden.nbDims >= 2 ? hidden.d[hidden.nbDims - 2] : -1;
}

} // namespace

PluginFieldCollection DFlash2GroupedDynamicConvPluginCreator::mFieldCollection{};
std::vector<PluginField> DFlash2GroupedDynamicConvPluginCreator::mPluginAttributes;
REGISTER_TENSORRT_PLUGIN(DFlash2GroupedDynamicConvPluginCreator);

DFlash2GroupedDynamicConvPlugin::DFlash2GroupedDynamicConvPlugin(
    std::string const& name, int32_t blockSize, int32_t kernelSize, int32_t groupSize, bool fuseResidual)
    : mLayerName(name)
    , mBlockSize(blockSize)
    , mKernelSize(kernelSize)
    , mGroupSize(groupSize)
    , mFuseResidual(fuseResidual)
{
    validateAttributes(blockSize, kernelSize, groupSize);
}

DFlash2GroupedDynamicConvPlugin::DFlash2GroupedDynamicConvPlugin(
    std::string const& name, PluginFieldCollection const* fields)
    : DFlash2GroupedDynamicConvPlugin(name, getIntField(fields, kFIELD_BLOCK), getIntField(fields, kFIELD_KERNEL),
          getIntField(fields, kFIELD_GROUP), getIntField(fields, kFIELD_FUSE_RESIDUAL) != 0)
{
}

IPluginCapability* DFlash2GroupedDynamicConvPlugin::getCapabilityInterface(PluginCapabilityType type) noexcept
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

IPluginV3* DFlash2GroupedDynamicConvPlugin::clone() noexcept
{
    try
    {
        auto* plugin
            = new DFlash2GroupedDynamicConvPlugin(mLayerName, mBlockSize, mKernelSize, mGroupSize, mFuseResidual);
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }
    catch (...)
    {
        return nullptr;
    }
}

char const* DFlash2GroupedDynamicConvPlugin::getPluginName() const noexcept
{
    return kPLUGIN_NAME;
}
char const* DFlash2GroupedDynamicConvPlugin::getPluginVersion() const noexcept
{
    return kPLUGIN_VERSION;
}
char const* DFlash2GroupedDynamicConvPlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}
void DFlash2GroupedDynamicConvPlugin::setPluginNamespace(char const* value) noexcept
{
    try
    {
        mNamespace = value == nullptr ? "" : value;
    }
    catch (...)
    {
    }
}

int32_t DFlash2GroupedDynamicConvPlugin::getNbOutputs() const noexcept
{
    return kNB_OUTPUTS;
}

int32_t DFlash2GroupedDynamicConvPlugin::getOutputDataTypes(
    DataType* outputTypes, int32_t nbOutputs, DataType const* inputTypes, int32_t nbInputs) const noexcept
{
    if (outputTypes == nullptr || inputTypes == nullptr || nbInputs != expectedInputs(mFuseResidual)
        || nbOutputs != kNB_OUTPUTS || !isActivationType(inputTypes[0]) || inputTypes[1] != inputTypes[0]
        || inputTypes[2] != inputTypes[0] || (mFuseResidual && inputTypes[kIN_RESIDUAL] != DataType::kFLOAT))
    {
        return -1;
    }
    outputTypes[0] = mFuseResidual ? DataType::kFLOAT : inputTypes[0];
    return 0;
}

int32_t DFlash2GroupedDynamicConvPlugin::getOutputShapes(DimsExprs const* inputs, int32_t nbInputs, DimsExprs const*,
    int32_t, DimsExprs* outputs, int32_t nbOutputs, IExprBuilder&) noexcept
{
    if (inputs == nullptr || outputs == nullptr || nbInputs != expectedInputs(mFuseResidual)
        || nbOutputs != kNB_OUTPUTS)
    {
        return -1;
    }
    outputs[0] = inputs[0];
    return 0;
}

bool DFlash2GroupedDynamicConvPlugin::supportsFormatCombination(
    int32_t pos, DynamicPluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
    if (inOut == nullptr || nbInputs != expectedInputs(mFuseResidual) || nbOutputs != kNB_OUTPUTS || pos < 0
        || pos > expectedInputs(mFuseResidual))
    {
        return false;
    }
    auto const& desc = inOut[pos].desc;
    if (desc.format != TensorFormat::kLINEAR)
    {
        return false;
    }
    if (pos < kBASE_INPUTS)
    {
        return pos == 0 ? isActivationType(desc.type) : desc.type == inOut[0].desc.type;
    }
    if (mFuseResidual)
    {
        return desc.type == DataType::kFLOAT;
    }
    return desc.type == inOut[0].desc.type;
}

int32_t DFlash2GroupedDynamicConvPlugin::configurePlugin(DynamicPluginTensorDesc const* inputs, int32_t nbInputs,
    DynamicPluginTensorDesc const* outputs, int32_t nbOutputs) noexcept
{
    if (inputs == nullptr || outputs == nullptr || nbInputs != expectedInputs(mFuseResidual)
        || nbOutputs != kNB_OUTPUTS)
    {
        return -1;
    }
    for (auto const* dims : {&inputs[0].min, &inputs[0].opt, &inputs[0].max})
    {
        int32_t const offset = dims == &inputs[0].min ? 0 : (dims == &inputs[0].opt ? 1 : 2);
        Dims const& delta = offset == 0 ? inputs[1].min : (offset == 1 ? inputs[1].opt : inputs[1].max);
        Dims const& base = offset == 0 ? inputs[2].min : (offset == 1 ? inputs[2].opt : inputs[2].max);
        Dims const* residual = mFuseResidual
            ? &(offset == 0 ? inputs[kIN_RESIDUAL].min
                            : (offset == 1 ? inputs[kIN_RESIDUAL].opt : inputs[kIN_RESIDUAL].max))
            : nullptr;
        int32_t const blockSize = runtimeBlockSize(*dims);
        // TensorRT bounds dimensions independently, so min/max may overestimate a quotient-derived block axis.
        if (blockSize <= 0 || (offset == 1 && blockSize > mBlockSize)
            || !validateShapes(*dims, delta, base, nullptr, mKernelSize, mGroupSize, false)
            || (residual != nullptr && !dimsEqual(*dims, *residual)))
        {
            LOG_ERROR("DFlash2GroupedDynamicConvPlugin: invalid optimization profile shape");
            return -1;
        }
    }
    return 0;
}

size_t DFlash2GroupedDynamicConvPlugin::getWorkspaceSize(
    DynamicPluginTensorDesc const*, int32_t, DynamicPluginTensorDesc const*, int32_t) const noexcept
{
    return 0;
}

int32_t DFlash2GroupedDynamicConvPlugin::enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc,
    void const* const* inputs, void* const* outputs, void*, cudaStream_t stream) noexcept
{
    if (inputDesc == nullptr || outputDesc == nullptr || inputs == nullptr || outputs == nullptr || inputs[0] == nullptr
        || inputs[1] == nullptr || inputs[2] == nullptr || outputs[0] == nullptr
        || (mFuseResidual
            && (inputs[kIN_RESIDUAL] == nullptr || !dimsEqual(inputDesc[0].dims, inputDesc[kIN_RESIDUAL].dims)))
        || !validateShapes(inputDesc[0].dims, inputDesc[1].dims, inputDesc[2].dims, &outputDesc[0].dims, mKernelSize,
            mGroupSize, true))
    {
        return -1;
    }
    if (flattenedTokens(inputDesc[0].dims) == 0)
    {
        return 0;
    }
    if (runtimeBlockSize(inputDesc[0].dims) <= 0 || runtimeBlockSize(inputDesc[0].dims) > mBlockSize)
    {
        return -1;
    }
    try
    {
        rt::Tensor const hidden(const_cast<void*>(inputs[kIN_HIDDEN]), rt::Coords{inputDesc[kIN_HIDDEN].dims},
            rt::DeviceType::kGPU, inputDesc[kIN_HIDDEN].type);
        rt::Tensor const delta(const_cast<void*>(inputs[kIN_DELTA]), rt::Coords{inputDesc[kIN_DELTA].dims},
            rt::DeviceType::kGPU, inputDesc[kIN_DELTA].type);
        rt::Tensor const base(const_cast<void*>(inputs[kIN_BASE]), rt::Coords{inputDesc[kIN_BASE].dims},
            rt::DeviceType::kGPU, inputDesc[kIN_BASE].type);
        rt::Tensor output(outputs[0], rt::Coords{outputDesc[0].dims}, rt::DeviceType::kGPU, outputDesc[0].type);
        if (mFuseResidual)
        {
            rt::Tensor const residual(const_cast<void*>(inputs[kIN_RESIDUAL]), rt::Coords{inputDesc[kIN_RESIDUAL].dims},
                rt::DeviceType::kGPU, inputDesc[kIN_RESIDUAL].type);
            return kernel::launchDFlash2GroupedDynamicConv(
                       hidden, delta, base, rt::OptionalInputTensor{residual}, output, stream)
                    == cudaSuccess
                ? 0
                : -1;
        }
        return kernel::launchDFlash2GroupedDynamicConv(hidden, delta, base, rt::OptionalInputTensor{}, output, stream)
                == cudaSuccess
            ? 0
            : -1;
    }
    catch (std::exception const& error)
    {
        LOG_ERROR("DFlash2GroupedDynamicConvPlugin enqueue failed: %s", error.what());
        return -1;
    }
}

int32_t DFlash2GroupedDynamicConvPlugin::onShapeChange(
    PluginTensorDesc const* inputs, int32_t nbInputs, PluginTensorDesc const* outputs, int32_t nbOutputs) noexcept
{
    return inputs != nullptr && outputs != nullptr && nbInputs == expectedInputs(mFuseResidual)
            && nbOutputs == kNB_OUTPUTS && runtimeBlockSize(inputs[0].dims) > 0
            && runtimeBlockSize(inputs[0].dims) <= mBlockSize
            && validateShapes(
                inputs[0].dims, inputs[1].dims, inputs[2].dims, &outputs[0].dims, mKernelSize, mGroupSize, true)
            && (!mFuseResidual || dimsEqual(inputs[0].dims, inputs[kIN_RESIDUAL].dims))
        ? 0
        : -1;
}

IPluginV3* DFlash2GroupedDynamicConvPlugin::attachToContext(IPluginResourceContext*) noexcept
{
    return clone();
}

PluginFieldCollection const* DFlash2GroupedDynamicConvPlugin::getFieldsToSerialize() noexcept
{
    mDataToSerialize.clear();
    mDataToSerialize.emplace_back(kFIELD_BLOCK, &mBlockSize, PluginFieldType::kINT32, 1);
    mDataToSerialize.emplace_back(kFIELD_KERNEL, &mKernelSize, PluginFieldType::kINT32, 1);
    mDataToSerialize.emplace_back(kFIELD_GROUP, &mGroupSize, PluginFieldType::kINT32, 1);
    mDataToSerialize.emplace_back(kFIELD_FUSE_RESIDUAL, &mFuseResidual, PluginFieldType::kINT32, 1);
    mFieldsToSerialize.nbFields = static_cast<int32_t>(mDataToSerialize.size());
    mFieldsToSerialize.fields = mDataToSerialize.data();
    return &mFieldsToSerialize;
}

DFlash2GroupedDynamicConvPluginCreator::DFlash2GroupedDynamicConvPluginCreator()
{
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);
    mPluginAttributes = {{kFIELD_BLOCK, nullptr, PluginFieldType::kINT32, 1},
        {kFIELD_KERNEL, nullptr, PluginFieldType::kINT32, 1}, {kFIELD_GROUP, nullptr, PluginFieldType::kINT32, 1},
        {kFIELD_FUSE_RESIDUAL, nullptr, PluginFieldType::kINT32, 1}};
    mFieldCollection.nbFields = static_cast<int32_t>(mPluginAttributes.size());
    mFieldCollection.fields = mPluginAttributes.data();
}

char const* DFlash2GroupedDynamicConvPluginCreator::getPluginName() const noexcept
{
    return kPLUGIN_NAME;
}
char const* DFlash2GroupedDynamicConvPluginCreator::getPluginVersion() const noexcept
{
    return kPLUGIN_VERSION;
}
PluginFieldCollection const* DFlash2GroupedDynamicConvPluginCreator::getFieldNames() noexcept
{
    return &mFieldCollection;
}
char const* DFlash2GroupedDynamicConvPluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}
void DFlash2GroupedDynamicConvPluginCreator::setPluginNamespace(char const* value) noexcept
{
    try
    {
        mNamespace = value == nullptr ? "" : value;
    }
    catch (...)
    {
    }
}

IPluginV3* DFlash2GroupedDynamicConvPluginCreator::createPlugin(
    char const* name, PluginFieldCollection const* fields, TensorRTPhase) noexcept
{
    try
    {
        auto* plugin = new DFlash2GroupedDynamicConvPlugin(name == nullptr ? "" : name, fields);
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }
    catch (std::exception const& error)
    {
        LOG_ERROR("DFlash2GroupedDynamicConvPlugin creation failed: %s", error.what());
        return nullptr;
    }
}

} // namespace plugins
} // namespace trt_edgellm
