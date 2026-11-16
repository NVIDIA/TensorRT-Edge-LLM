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

#include "sparseAttentionPlugin.h"

#include "common/cudaUtils.h"
#include "common/logger.h"
#include "kernels/sparseAttention/videoSparseAttentionRunner.h"

using namespace nvinfer1;

namespace trt_edgellm
{
namespace plugins
{
using nvinfer1::DataType;

namespace
{
constexpr char const* kPLUGIN_NAME{"SparseAttentionPlugin"};
constexpr char const* kPLUGIN_VERSION{"1"};
constexpr int32_t kIN_SUFFIX_ROWS{3};
constexpr int32_t kNB_INPUTS{4};
constexpr int32_t kNB_OUTPUTS{1};

using Runner = VideoSparseAttentionRunner;

bool isQkvShape(Dims const& dims) noexcept
{
    return dims.nbDims == 3 && dims.d[1] == Runner::kNUM_HEADS && dims.d[2] == Runner::kHEAD_SIZE;
}

//! Q, K, V, and the output share one shape, and the suffix leaves a nonempty prefix.
bool isValidShape(Dims const& q, Dims const& k, Dims const& v, Dims const& output, Dims const& suffix) noexcept
{
    for (Dims const* dims : {&q, &k, &v, &output})
    {
        if (!isQkvShape(*dims) || dims->d[0] != q.d[0])
        {
            return false;
        }
    }
    return suffix.nbDims >= 1 && suffix.d[0] > 0 && suffix.d[0] < q.d[0];
}
} // namespace

#if defined(CUTE_DSL_VIDEO_SPARSE_ATTENTION_ENABLED)
REGISTER_TENSORRT_PLUGIN(SparseAttentionPluginCreator);
#endif

SparseAttentionPlugin::SparseAttentionPlugin(std::string const& name)
    : mLayerName(name)
    , mSmVersion(getSMVersion())
{
}

IPluginCapability* SparseAttentionPlugin::getCapabilityInterface(PluginCapabilityType type) noexcept
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

IPluginV3* SparseAttentionPlugin::clone() noexcept
{
    try
    {
        auto* plugin = new SparseAttentionPlugin(mLayerName);
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }
    catch (...)
    {
        return nullptr;
    }
}

char const* SparseAttentionPlugin::getPluginName() const noexcept
{
    return kPLUGIN_NAME;
}

char const* SparseAttentionPlugin::getPluginVersion() const noexcept
{
    return kPLUGIN_VERSION;
}

char const* SparseAttentionPlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

void SparseAttentionPlugin::setPluginNamespace(char const* pluginNamespace) noexcept
{
    try
    {
        mNamespace = pluginNamespace == nullptr ? "" : pluginNamespace;
    }
    catch (...)
    {
    }
}

int32_t SparseAttentionPlugin::getNbOutputs() const noexcept
{
    return kNB_OUTPUTS;
}

int32_t SparseAttentionPlugin::getOutputDataTypes(
    DataType* outputTypes, int32_t nbOutputs, DataType const* inputTypes, int32_t nbInputs) const noexcept
{
    if (nbInputs != kNB_INPUTS || nbOutputs != kNB_OUTPUTS)
    {
        return -1;
    }
    outputTypes[0] = inputTypes[0];
    return 0;
}

int32_t SparseAttentionPlugin::getOutputShapes(DimsExprs const* inputs, int32_t nbInputs, DimsExprs const*, int32_t,
    DimsExprs* outputs, int32_t nbOutputs, IExprBuilder&) noexcept
{
    if (nbInputs != kNB_INPUTS || nbOutputs != kNB_OUTPUTS)
    {
        return -1;
    }
    outputs[0] = inputs[0];
    return 0;
}

bool SparseAttentionPlugin::supportsFormatCombination(
    int32_t pos, DynamicPluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
    if (nbInputs != kNB_INPUTS || nbOutputs != kNB_OUTPUTS || inOut[pos].desc.format != TensorFormat::kLINEAR)
    {
        return false;
    }
    return pos == kIN_SUFFIX_ROWS || inOut[pos].desc.type == DataType::kBF16;
}

int32_t SparseAttentionPlugin::configurePlugin(
    DynamicPluginTensorDesc const* inputs, int32_t, DynamicPluginTensorDesc const* outputs, int32_t) noexcept
{
    if (!isValidShape(inputs[0].max, inputs[1].max, inputs[2].max, outputs[0].max, inputs[kIN_SUFFIX_ROWS].max))
    {
        LOG_ERROR("%s requires Q/K/V [sequence, %d, %d] and fewer suffix rows than the sequence", kPLUGIN_NAME,
            Runner::kNUM_HEADS, Runner::kHEAD_SIZE);
        return -1;
    }
    return 0;
}

size_t SparseAttentionPlugin::getWorkspaceSize(
    DynamicPluginTensorDesc const* inputs, int32_t, DynamicPluginTensorDesc const*, int32_t) const noexcept
{
    return Runner::getWorkspaceSize(static_cast<int32_t>(inputs[0].max.d[0]));
}

int32_t SparseAttentionPlugin::enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc,
    void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    auto const sequence = static_cast<int32_t>(inputDesc[0].dims.d[0]);
    auto const suffix = static_cast<int32_t>(inputDesc[kIN_SUFFIX_ROWS].dims.d[0]);
    static_cast<void>(outputDesc);
    return Runner::run(
               inputs[0], inputs[1], inputs[2], outputs[0], sequence, sequence - suffix, workspace, mSmVersion, stream)
        ? 0
        : -1;
}

int32_t SparseAttentionPlugin::onShapeChange(
    PluginTensorDesc const* inputs, int32_t, PluginTensorDesc const* outputs, int32_t) noexcept
{
    return isValidShape(inputs[0].dims, inputs[1].dims, inputs[2].dims, outputs[0].dims, inputs[kIN_SUFFIX_ROWS].dims)
        ? 0
        : -1;
}

IPluginV3* SparseAttentionPlugin::attachToContext(IPluginResourceContext*) noexcept
{
    return clone();
}

PluginFieldCollection const* SparseAttentionPlugin::getFieldsToSerialize() noexcept
{
    return &mFieldsToSerialize;
}

char const* SparseAttentionPluginCreator::getPluginName() const noexcept
{
    return kPLUGIN_NAME;
}

char const* SparseAttentionPluginCreator::getPluginVersion() const noexcept
{
    return kPLUGIN_VERSION;
}

PluginFieldCollection const* SparseAttentionPluginCreator::getFieldNames() noexcept
{
    return &mFieldCollection;
}

char const* SparseAttentionPluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

void SparseAttentionPluginCreator::setPluginNamespace(char const* pluginNamespace) noexcept
{
    try
    {
        mNamespace = pluginNamespace == nullptr ? "" : pluginNamespace;
    }
    catch (...)
    {
    }
}

IPluginV3* SparseAttentionPluginCreator::createPlugin(
    char const* name, PluginFieldCollection const*, TensorRTPhase) noexcept
{
    int32_t const smVersion = getSMVersion();
    if (!Runner::canImplement(Runner::kNUM_HEADS, Runner::kHEAD_SIZE, smVersion))
    {
        LOG_ERROR("%s has no kernel for SM %d in this build", kPLUGIN_NAME, smVersion);
        return nullptr;
    }
    try
    {
        auto* plugin = new SparseAttentionPlugin(name == nullptr ? "" : name);
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }
    catch (...)
    {
        return nullptr;
    }
}

} // namespace plugins
} // namespace trt_edgellm
