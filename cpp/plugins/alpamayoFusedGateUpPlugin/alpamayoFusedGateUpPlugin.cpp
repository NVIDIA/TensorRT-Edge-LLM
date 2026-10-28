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

#include "alpamayoFusedGateUpPlugin.h"

#include <limits>

#include "common/cudaUtils.h"
#include "common/logger.h"

#include "kernels/alpamayo/alpamayoKernels.h"

using namespace nvinfer1;

namespace trt_edgellm
{
namespace plugins
{
namespace
{

constexpr char const* kPLUGIN_NAME{"AlpamayoFusedGateUpPlugin"};
constexpr char const* kPLUGIN_VERSION{"1"};

constexpr int32_t kNB_INPUTS{4};
constexpr int32_t kNB_OUTPUTS{1};
// The device tile count rounds M up by 255 in signed integer arithmetic.
constexpr int64_t kMAX_TOKENS{std::numeric_limits<int32_t>::max() - 255};
constexpr int64_t kHIDDEN{2048};
constexpr int64_t kGATE_UP{12288};
constexpr int32_t kGRID_N{24};
constexpr int32_t kGRID_SIZE{20};

bool isScalar(Dims const& dims) noexcept
{
    return dims.nbDims == 1 && dims.d[0] == 1;
}

bool shapesSupported(PluginTensorDesc const* in, PluginTensorDesc const* out, bool allowDynamic = false) noexcept
{
    Dims const& x = in[0].dims;
    Dims const& w = in[1].dims;
    Dims const& y = out[0].dims;
    bool const tokensSupported
        = x.nbDims == 2 && ((x.d[0] >= 1 && x.d[0] <= kMAX_TOKENS) || (allowDynamic && x.d[0] == -1));
    return x.nbDims == 2 && tokensSupported && x.d[1] == kHIDDEN && w.nbDims == 2 && w.d[0] == kGATE_UP
        && w.d[1] == kHIDDEN && isScalar(in[2].dims) && isScalar(in[3].dims) && y.nbDims == 2 && y.d[0] == x.d[0]
        && y.d[1] == kGATE_UP / 2;
}

//! The kernel is SM110 (Thor) only.
bool deviceSupported() noexcept
{
    try
    {
        return getSMVersion() == 110;
    }
    catch (...)
    {
        return false;
    }
}

} // namespace

REGISTER_TENSORRT_PLUGIN(AlpamayoFusedGateUpPluginCreator);

AlpamayoFusedGateUpPlugin::AlpamayoFusedGateUpPlugin(std::string const& name)
    : mLayerName(name)
{
}

IPluginCapability* AlpamayoFusedGateUpPlugin::getCapabilityInterface(PluginCapabilityType type) noexcept
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

IPluginV3* AlpamayoFusedGateUpPlugin::clone() noexcept
{
    try
    {
        auto* plugin = new AlpamayoFusedGateUpPlugin(mLayerName);
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }
    catch (std::exception const& error)
    {
        LOG_ERROR("%s: clone failed: %s", kPLUGIN_NAME, error.what());
        return nullptr;
    }
}

char const* AlpamayoFusedGateUpPlugin::getPluginName() const noexcept
{
    return kPLUGIN_NAME;
}

char const* AlpamayoFusedGateUpPlugin::getPluginVersion() const noexcept
{
    return kPLUGIN_VERSION;
}

char const* AlpamayoFusedGateUpPlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

void AlpamayoFusedGateUpPlugin::setPluginNamespace(char const* pluginNamespace) noexcept
{
    mNamespace = pluginNamespace != nullptr ? pluginNamespace : "";
}

int32_t AlpamayoFusedGateUpPlugin::getNbOutputs() const noexcept
{
    return kNB_OUTPUTS;
}

int32_t AlpamayoFusedGateUpPlugin::getOutputDataTypes(
    DataType* outputTypes, int32_t nbOutputs, DataType const* inputTypes, int32_t nbInputs) const noexcept
{
    if (nbInputs != kNB_INPUTS || nbOutputs != kNB_OUTPUTS || inputTypes[0] != DataType::kFP8
        || inputTypes[1] != DataType::kFP8 || inputTypes[2] != DataType::kFLOAT || inputTypes[3] != DataType::kFLOAT)
    {
        return 1;
    }
    outputTypes[0] = DataType::kFP8;
    return 0;
}

int32_t AlpamayoFusedGateUpPlugin::getOutputShapes(DimsExprs const* inputs, int32_t nbInputs, DimsExprs const*,
    int32_t nbShapeInputs, DimsExprs* outputs, int32_t nbOutputs, IExprBuilder& exprBuilder) noexcept
{
    if (nbInputs != kNB_INPUTS || nbOutputs != kNB_OUTPUTS || nbShapeInputs != 0 || inputs[0].nbDims != 2)
    {
        return 1;
    }
    outputs[0].nbDims = 2;
    outputs[0].d[0] = inputs[0].d[0];
    outputs[0].d[1] = exprBuilder.constant(kGATE_UP / 2);
    return 0;
}

bool AlpamayoFusedGateUpPlugin::supportsFormatCombination(
    int32_t pos, DynamicPluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
    if (nbInputs != kNB_INPUTS || nbOutputs != kNB_OUTPUTS || pos < 0 || pos >= kNB_INPUTS + kNB_OUTPUTS)
    {
        return false;
    }
    DataType const expected = (pos == 2 || pos == 3) ? DataType::kFLOAT : DataType::kFP8;
    return inOut[pos].desc.format == TensorFormat::kLINEAR && inOut[pos].desc.type == expected;
}

int32_t AlpamayoFusedGateUpPlugin::configurePlugin(DynamicPluginTensorDesc const* inputs, int32_t nbInputs,
    DynamicPluginTensorDesc const* outputs, int32_t nbOutputs) noexcept
{
    if (nbInputs != kNB_INPUTS || nbOutputs != kNB_OUTPUTS)
    {
        return 1;
    }
    if (!deviceSupported())
    {
        LOG_ERROR("%s: requires an SM110 (Thor) device and a 110a build", kPLUGIN_NAME);
        return 1;
    }
    PluginTensorDesc const in[kNB_INPUTS]{inputs[0].desc, inputs[1].desc, inputs[2].desc, inputs[3].desc};
    if (!shapesSupported(in, &outputs[0].desc, true))
    {
        LOG_ERROR(
            "%s: requires x[M, %ld], W[%ld, %ld] and scalar FP32 scales", kPLUGIN_NAME, kHIDDEN, kGATE_UP, kHIDDEN);
        return 1;
    }
    for (bool const useMax : {false, true})
    {
        PluginTensorDesc bounds[kNB_INPUTS]{in[0], in[1], in[2], in[3]};
        for (int32_t i = 0; i < kNB_INPUTS; ++i)
        {
            bounds[i].dims = useMax ? inputs[i].max : inputs[i].min;
        }
        PluginTensorDesc output = outputs[0].desc;
        output.dims.d[0] = bounds[0].dims.d[0];
        if (!shapesSupported(bounds, &output))
        {
            LOG_ERROR("%s: invalid optimization profile; M must be in [1, %ld]", kPLUGIN_NAME, kMAX_TOKENS);
            return 1;
        }
    }
    return initializeAlpamayoFusedGateUp();
}

size_t AlpamayoFusedGateUpPlugin::getWorkspaceSize(
    DynamicPluginTensorDesc const*, int32_t, DynamicPluginTensorDesc const*, int32_t) const noexcept
{
    return 0;
}

int32_t AlpamayoFusedGateUpPlugin::onShapeChange(
    PluginTensorDesc const* inputs, int32_t nbInputs, PluginTensorDesc const* outputs, int32_t nbOutputs) noexcept
{
    return nbInputs == kNB_INPUTS && nbOutputs == kNB_OUTPUTS && shapesSupported(inputs, outputs) ? 0 : 1;
}

int32_t AlpamayoFusedGateUpPlugin::enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc,
    void const* const* inputs, void* const* outputs, void*, cudaStream_t stream) noexcept
{
    if (!shapesSupported(inputDesc, outputDesc))
    {
        return 1;
    }
    AlpamayoFusedGateUpParams const params{inputs[0], inputs[1], outputs[0], static_cast<float const*>(inputs[2]),
        static_cast<float const*>(inputs[3]), static_cast<int>(inputDesc[0].dims.d[0]), kGATE_UP, kHIDDEN, kGRID_N,
        kGRID_SIZE};
    return launch_alpamayo_fp8_gate_up_swiglu_2cta_m256o256k128_s3_tanh(params, stream);
}

IPluginV3* AlpamayoFusedGateUpPlugin::attachToContext(IPluginResourceContext*) noexcept
{
    if (!deviceSupported() || initializeAlpamayoFusedGateUp() != 0)
    {
        return nullptr;
    }
    return clone();
}

PluginFieldCollection const* AlpamayoFusedGateUpPlugin::getFieldsToSerialize() noexcept
{
    return &mFieldsToSerialize;
}

char const* AlpamayoFusedGateUpPluginCreator::getPluginName() const noexcept
{
    return kPLUGIN_NAME;
}

char const* AlpamayoFusedGateUpPluginCreator::getPluginVersion() const noexcept
{
    return kPLUGIN_VERSION;
}

PluginFieldCollection const* AlpamayoFusedGateUpPluginCreator::getFieldNames() noexcept
{
    return &mFieldCollection;
}

char const* AlpamayoFusedGateUpPluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

void AlpamayoFusedGateUpPluginCreator::setPluginNamespace(char const* pluginNamespace) noexcept
{
    mNamespace = pluginNamespace != nullptr ? pluginNamespace : "";
}

IPluginV3* AlpamayoFusedGateUpPluginCreator::createPlugin(
    char const* name, PluginFieldCollection const*, TensorRTPhase) noexcept
{
    try
    {
        auto* plugin = new AlpamayoFusedGateUpPlugin(name != nullptr ? name : "");
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }
    catch (std::exception const& error)
    {
        LOG_ERROR("%s: createPlugin failed: %s", kPLUGIN_NAME, error.what());
        return nullptr;
    }
}

} // namespace plugins
} // namespace trt_edgellm
