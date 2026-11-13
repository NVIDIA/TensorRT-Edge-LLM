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

#include "w2a16GemmPlugin.h"
#include "common/logger.h"
#include "kernels/gemm/w2a16_cutedsl/cuteDslW2A16GemmRunner.h"
#include <NvInfer.h>
#include <algorithm>
#include <memory>
#include <stdexcept>

using namespace nvinfer1;

namespace trt_edgellm
{
namespace plugins
{
namespace
{
constexpr char const* kPLUGIN_NAME = "W2A16GemmPlugin";
constexpr char const* kPLUGIN_VERSION = "1";
constexpr int32_t kMAX_M = 32768;
constexpr int32_t kNB_INPUTS = 4;
constexpr int32_t kOUT_OUTPUT = kNB_INPUTS;

int32_t getTokenCount(Dims const& dims) noexcept
{
    if (dims.nbDims != 2 && dims.nbDims != 3)
    {
        return 0;
    }
    int64_t count = 1;
    for (int32_t i = 0; i < dims.nbDims - 1; ++i)
    {
        if (dims.d[i] <= 0 || dims.d[i] > kMAX_M / count)
        {
            return 0;
        }
        count *= dims.d[i];
    }
    return static_cast<int32_t>(count);
}

bool sameDims(Dims const& a, Dims const& b) noexcept
{
    return a.nbDims == b.nbDims && a.nbDims >= 0 && a.nbDims <= Dims::MAX_DIMS && std::equal(a.d, a.d + a.nbDims, b.d);
}

} // namespace

REGISTER_TENSORRT_PLUGIN(W2A16GemmPluginCreator);

W2A16GemmPlugin::W2A16GemmPlugin(std::string const& name, int32_t gemmN, int32_t gemmK)
    : mLayerName(name)
    , mGemmN(gemmN)
    , mGemmK(gemmK)
{
    validateAttributes();
}

W2A16GemmPlugin::W2A16GemmPlugin(std::string const& name, PluginFieldCollection const* fields)
    : mLayerName(name)
{
    if (fields == nullptr || fields->fields == nullptr || fields->nbFields != 2)
    {
        throw std::invalid_argument("W2A16GemmPlugin: gemm_n and gemm_k are required");
    }
    std::array<bool, 2> fieldsSeen{};
    auto readIntField = [&fieldsSeen](PluginField const& field, int32_t index) {
        if (field.data == nullptr || field.type != PluginFieldType::kINT32 || field.length != 1)
        {
            throw std::invalid_argument("W2A16GemmPlugin: attributes must be scalar INT32 fields");
        }
        if (fieldsSeen[index])
        {
            throw std::invalid_argument("W2A16GemmPlugin: duplicate plugin attribute");
        }
        fieldsSeen[index] = true;
        return *static_cast<int32_t const*>(field.data);
    };
    for (int32_t i = 0; i < fields->nbFields; ++i)
    {
        PluginField const& field = fields->fields[i];
        if (field.name == nullptr)
        {
            throw std::invalid_argument("W2A16GemmPlugin: plugin attribute name must not be null");
        }
        std::string const fieldName(field.name);
        if (fieldName == "gemm_n")
        {
            mGemmN = readIntField(field, 0);
        }
        else if (fieldName == "gemm_k")
        {
            mGemmK = readIntField(field, 1);
        }
        else
        {
            throw std::invalid_argument("W2A16GemmPlugin: unknown plugin attribute " + fieldName);
        }
    }
    validateAttributes();
}

void W2A16GemmPlugin::validateAttributes() const
{
#if defined(CUTE_DSL_W2A16_GEMM_ENABLED)
    if (!CuteDslW2A16GemmRunner::canImplement(1, mGemmN, mGemmK, 121, 48))
    {
        throw std::invalid_argument("W2A16GemmPlugin requires N%8=0, N<=4194240 and K%256=0");
    }
#else
    throw std::runtime_error("W2A16GemmPlugin requires ENABLE_CUTE_DSL=w2a16_gemm");
#endif
}

IPluginCapability* W2A16GemmPlugin::getCapabilityInterface(PluginCapabilityType type) noexcept
{
    if (type == PluginCapabilityType::kBUILD)
    {
        return static_cast<IPluginV3OneBuild*>(this);
    }
    if (type == PluginCapabilityType::kRUNTIME)
    {
        return static_cast<IPluginV3OneRuntime*>(this);
    }
    return type == PluginCapabilityType::kCORE ? static_cast<IPluginV3OneCore*>(this) : nullptr;
}

IPluginV3* W2A16GemmPlugin::clone() noexcept
{
    try
    {
        auto p = std::make_unique<W2A16GemmPlugin>(mLayerName, mGemmN, mGemmK);
        p->mNamespace = mNamespace;
        return p.release();
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("W2A16GemmPlugin clone failed: %s", e.what());
        return nullptr;
    }
}

char const* W2A16GemmPlugin::getPluginName() const noexcept
{
    return kPLUGIN_NAME;
}
char const* W2A16GemmPlugin::getPluginVersion() const noexcept
{
    return kPLUGIN_VERSION;
}
char const* W2A16GemmPlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}
void W2A16GemmPlugin::setPluginNamespace(char const* pluginNamespace) noexcept
{
    try
    {
        mNamespace = pluginNamespace ? pluginNamespace : "";
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("W2A16GemmPlugin namespace failed: %s", e.what());
    }
}
int32_t W2A16GemmPlugin::getNbOutputs() const noexcept
{
    return 1;
}

int32_t W2A16GemmPlugin::getOutputDataTypes(
    DataType* outputTypes, int32_t nbOutputs, DataType const* inputTypes, int32_t nbInputs) const noexcept
{
    if (!outputTypes || !inputTypes || nbInputs != kNB_INPUTS || nbOutputs != 1 || inputTypes[0] != DataType::kBF16
        || inputTypes[1] != DataType::kINT8 || inputTypes[2] != DataType::kHALF || inputTypes[3] != DataType::kBF16)
    {
        return -1;
    }
    outputTypes[0] = DataType::kBF16;
    return 0;
}

int32_t W2A16GemmPlugin::getOutputShapes(DimsExprs const* inputs, int32_t nbInputs, DimsExprs const*,
    int32_t nbShapeInputs, DimsExprs* outputs, int32_t nbOutputs, IExprBuilder& exprBuilder) noexcept
{
    if (!inputs || !outputs || nbInputs != kNB_INPUTS || nbOutputs != 1 || nbShapeInputs != 0
        || (inputs[0].nbDims != 2 && inputs[0].nbDims != 3))
    {
        return -1;
    }
    outputs[0] = inputs[0];
    outputs[0].d[outputs[0].nbDims - 1] = exprBuilder.constant(mGemmN);
    return 0;
}

bool W2A16GemmPlugin::supportsFormatCombination(
    int32_t pos, DynamicPluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
    return inOut != nullptr && nbInputs == kNB_INPUTS && nbOutputs == 1 && pos >= 0 && pos <= kOUT_OUTPUT
        && validateTensorDesc(pos, inOut[pos].desc);
}

bool W2A16GemmPlugin::validateTensorDesc(int32_t pos, PluginTensorDesc const& desc) const noexcept
{
    std::array<DataType, 5> const types{
        DataType::kBF16, DataType::kINT8, DataType::kHALF, DataType::kBF16, DataType::kBF16};
    if (desc.format != TensorFormat::kLINEAR || desc.type != types[pos])
    {
        return false;
    }
    if (pos == 0 || pos == kOUT_OUTPUT)
    {
        if ((desc.dims.nbDims != 2 && desc.dims.nbDims != 3)
            || desc.dims.d[desc.dims.nbDims - 1] != (pos == 0 ? mGemmK : mGemmN))
        {
            return false;
        }
        for (int32_t i = 0; i < desc.dims.nbDims - 1; ++i)
        {
            if (desc.dims.d[i] != -1 && (desc.dims.d[i] <= 0 || desc.dims.d[i] > kMAX_M))
            {
                return false;
            }
        }
        return true;
    }
    int32_t const packing = mGemmN % 128 == 0 ? 128 : 8;
    Dims const expected = pos == 1 ? Dims{4, {mGemmN / packing, mGemmK / 256, packing, 64}}
        : pos == 2                 ? Dims{2, {mGemmK / 256, mGemmN}}
                                   : Dims{2, {256, 8}};
    return sameDims(desc.dims, expected);
}

bool W2A16GemmPlugin::validateDescriptors(
    PluginTensorDesc const* inputs, PluginTensorDesc const* outputs) const noexcept
{
    if (!inputs || !outputs || getTokenCount(inputs[0].dims) == 0)
    {
        return false;
    }
    for (int32_t pos = 0; pos < kNB_INPUTS; ++pos)
    {
        if (!validateTensorDesc(pos, inputs[pos]))
        {
            return false;
        }
    }
    Dims expected = inputs[0].dims;
    expected.d[expected.nbDims - 1] = mGemmN;
    return validateTensorDesc(kOUT_OUTPUT, outputs[0]) && sameDims(expected, outputs[0].dims);
}

int32_t W2A16GemmPlugin::configurePlugin(DynamicPluginTensorDesc const* inputs, int32_t nbInputs,
    DynamicPluginTensorDesc const* outputs, int32_t nbOutputs) noexcept
{
    if (!inputs || !outputs || nbInputs != kNB_INPUTS || nbOutputs != 1)
    {
        return -1;
    }
    for (bool maximum : {false, true})
    {
        std::array<PluginTensorDesc, kNB_INPUTS> in{};
        for (int32_t i = 0; i < kNB_INPUTS; ++i)
        {
            in[i] = inputs[i].desc;
            in[i].dims = maximum ? inputs[i].max : inputs[i].min;
        }
        auto out = outputs[0].desc;
        out.dims = maximum ? outputs[0].max : outputs[0].min;
        if (!validateDescriptors(in.data(), &out))
        {
            return -1;
        }
    }
    return 0;
}

size_t W2A16GemmPlugin::getWorkspaceSize(DynamicPluginTensorDesc const* inputs, int32_t nbInputs,
    DynamicPluginTensorDesc const*, int32_t nbOutputs) const noexcept
{
    size_t bytes = 0;
#if defined(CUTE_DSL_W2A16_GEMM_ENABLED)
    if (inputs && nbInputs == kNB_INPUTS && nbOutputs == 1)
    {
        // Scratch is not monotonic in M: split-K disappears once the grid reaches 48 CTAs.
        int32_t const maxRows = getTokenCount(inputs[0].max);
        for (int32_t m = 1; m <= std::min(maxRows, 48 * 16); ++m)
        {
            bytes = std::max(bytes, CuteDslW2A16GemmRunner::getWorkspaceSize(m, mGemmN, mGemmK));
        }
    }
#endif
    return bytes;
}

int32_t W2A16GemmPlugin::onShapeChange(
    PluginTensorDesc const* inputs, int32_t nbInputs, PluginTensorDesc const* outputs, int32_t nbOutputs) noexcept
{
    if (nbInputs != kNB_INPUTS || nbOutputs != 1 || !validateDescriptors(inputs, outputs))
    {
        return -1;
    }
#if defined(CUTE_DSL_W2A16_GEMM_ENABLED)
    return 0;
#else
    return -1;
#endif
}

int32_t W2A16GemmPlugin::enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc,
    void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
#if defined(CUTE_DSL_W2A16_GEMM_ENABLED)
    if (!inputs || !outputs || !validateDescriptors(inputDesc, outputDesc))
    {
        return -1;
    }
    CuteDslW2A16GemmParams const p{
        getTokenCount(inputDesc[0].dims), mGemmN, mGemmK, inputs[0], inputs[1], inputs[2], inputs[3], outputs[0]};
    if (!CuteDslW2A16GemmRunner::ensureKernelModules(p, stream))
    {
        return -1;
    }
    return CuteDslW2A16GemmRunner{}.run(p, workspace, CuteDslW2A16GemmRunner::getWorkspaceSize(p.m, p.n, p.k), stream);
#else
    return -1;
#endif
}

IPluginV3* W2A16GemmPlugin::attachToContext(IPluginResourceContext*) noexcept
{
    return clone();
}

PluginFieldCollection const* W2A16GemmPlugin::getFieldsToSerialize() noexcept
{
    mDataToSerialize = {PluginField{"gemm_n", &mGemmN, PluginFieldType::kINT32, 1},
        PluginField{"gemm_k", &mGemmK, PluginFieldType::kINT32, 1}};
    mFieldsToSerialize = {2, mDataToSerialize.data()};
    return &mFieldsToSerialize;
}

W2A16GemmPluginCreator::W2A16GemmPluginCreator()
{
    mPluginAttributes = {PluginField{"gemm_n", nullptr, PluginFieldType::kINT32, 1},
        PluginField{"gemm_k", nullptr, PluginFieldType::kINT32, 1}};
    mFieldCollection = {2, mPluginAttributes.data()};
}
char const* W2A16GemmPluginCreator::getPluginName() const noexcept
{
    return kPLUGIN_NAME;
}
char const* W2A16GemmPluginCreator::getPluginVersion() const noexcept
{
    return kPLUGIN_VERSION;
}
char const* W2A16GemmPluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}
PluginFieldCollection const* W2A16GemmPluginCreator::getFieldNames() noexcept
{
    return &mFieldCollection;
}
void W2A16GemmPluginCreator::setPluginNamespace(char const* pluginNamespace) noexcept
{
    try
    {
        mNamespace = pluginNamespace ? pluginNamespace : "";
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("W2A16GemmPluginCreator namespace failed: %s", e.what());
    }
}
IPluginV3* W2A16GemmPluginCreator::createPlugin(
    char const* name, PluginFieldCollection const* fields, TensorRTPhase) noexcept
{
    try
    {
        if (!name)
        {
            return nullptr;
        }
        auto p = std::make_unique<W2A16GemmPlugin>(name, fields);
        p->setPluginNamespace(mNamespace.c_str());
        return p.release();
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("W2A16GemmPlugin creation failed: %s", e.what());
        return nullptr;
    }
}
} // namespace plugins
} // namespace trt_edgellm
