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

#include "w2a16MoePlugin.h"
#include "common/logger.h"

#ifdef CUTE_DSL_W2A16_MOE_ENABLED
#include "common/cudaUtils.h"
#include "kernels/moe/w2a16_cutedsl/cuteDslW2A16MoeRunner.h"
#endif

#include <algorithm>
#include <array>
#include <cstdint>
#include <exception>
#include <utility>

using namespace nvinfer1;

namespace trt_edgellm::plugins
{
namespace
{
constexpr char const* kName = "W2A16MoePlugin";
constexpr char const* kVersion = "1";
constexpr int32_t kInputs = 9;
constexpr std::array<DataType, kInputs + 1> kTypes
    = {DataType::kBF16, DataType::kINT32, DataType::kINT32, DataType::kINT32, DataType::kINT32, DataType::kFLOAT,
        DataType::kINT8, DataType::kINT32, DataType::kFLOAT, DataType::kBF16};
// TensorRT workspaces may be only 128-byte aligned; the runner requires 256.
constexpr size_t kWorkspaceAlignment = 256;

bool validShape(PluginTensorDesc const* in, PluginTensorDesc const* out, int32_t experts, int32_t interSize)
{
    if (in == nullptr || out == nullptr || in[0].dims.nbDims != 2 || out[0].dims.nbDims != 2 || in[0].dims.d[0] <= 0
        || in[0].dims.d[0] != out[0].dims.d[0] || in[0].dims.d[1] != 1024 || out[0].dims.d[1] != 1024
        || in[7].dims.nbDims != 2 || in[8].dims.nbDims != 2 || in[7].dims.d[0] != in[0].dims.d[0]
        || in[8].dims.d[0] != in[0].dims.d[0] || in[7].dims.d[1] != 9 || in[8].dims.d[1] != 9 || in[5].dims.nbDims != 1
        || in[5].dims.d[0] != experts || in[6].dims.nbDims != 1 || in[6].dims.d[0] != 2048)
    {
        return false;
    }
    for (int32_t i = 0; i <= kInputs; ++i)
    {
        auto const& desc = i == kInputs ? out[0] : in[i];
        if (desc.type != kTypes[i] || desc.format != PluginFormat::kLINEAR)
        {
            return false;
        }
    }
    // Packed IQ2_XXS: exactly two bytes per 256 weights, plus FP16 base per block.
    int64_t const fc1Words = static_cast<int64_t>(experts) * interSize * 1024 / 16;
    int64_t const fc2Words = fc1Words;
    int64_t const scaleWords = static_cast<int64_t>(experts) * interSize * 1024 / 512;
    return in[1].dims.nbDims == 1 && in[1].dims.d[0] == fc1Words && in[2].dims.nbDims == 1
        && in[2].dims.d[0] == fc2Words && in[3].dims.nbDims == 1 && in[3].dims.d[0] == scaleWords
        && in[4].dims.nbDims == 1 && in[4].dims.d[0] == scaleWords;
}
} // namespace

REGISTER_TENSORRT_PLUGIN(W2A16MoePluginCreator);

W2A16MoePlugin::W2A16MoePlugin(std::string name, int32_t numExperts, int32_t moeInterSize)
    : mName(std::move(name))
    , mNumExperts(numExperts)
    , mMoeInterSize(moeInterSize)
{
    mAttributes.emplace_back("num_experts", &mNumExperts, PluginFieldType::kINT32, 1);
    mAttributes.emplace_back("moe_inter_size", &mMoeInterSize, PluginFieldType::kINT32, 1);
    mFields = {static_cast<int32_t>(mAttributes.size()), mAttributes.data()};
}

IPluginCapability* W2A16MoePlugin::getCapabilityInterface(PluginCapabilityType type) noexcept
{
    if (type == PluginCapabilityType::kBUILD)
        return static_cast<IPluginV3OneBuild*>(this);
    if (type == PluginCapabilityType::kRUNTIME)
        return static_cast<IPluginV3OneRuntime*>(this);
    return static_cast<IPluginV3OneCore*>(this);
}

IPluginV3* W2A16MoePlugin::clone() noexcept
{
    try
    {
        auto* plugin = new W2A16MoePlugin(mName, mNumExperts, mMoeInterSize);
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }
    catch (...)
    {
        return nullptr;
    }
}

char const* W2A16MoePlugin::getPluginName() const noexcept
{
    return kName;
}
char const* W2A16MoePlugin::getPluginVersion() const noexcept
{
    return kVersion;
}
char const* W2A16MoePlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}
void W2A16MoePlugin::setPluginNamespace(char const* name) noexcept
{
    mNamespace = name ? name : "";
}
int32_t W2A16MoePlugin::getNbOutputs() const noexcept
{
    return 1;
}

int32_t W2A16MoePlugin::getOutputDataTypes(
    DataType* out, int32_t nbOutputs, DataType const* in, int32_t nbInputs) const noexcept
{
    if (out == nullptr || in == nullptr || nbInputs != kInputs || nbOutputs != 1)
        return -1;
    for (int32_t i = 0; i < kInputs; ++i)
        if (in[i] != kTypes[i])
            return -1;
    out[0] = DataType::kBF16;
    return 0;
}

int32_t W2A16MoePlugin::getOutputShapes(DimsExprs const* in, int32_t nbInputs, DimsExprs const*, int32_t,
    DimsExprs* out, int32_t nbOutputs, IExprBuilder&) noexcept
{
    if (in == nullptr || out == nullptr || nbInputs != kInputs || nbOutputs != 1 || in[0].nbDims != 2)
        return -1;
    out[0] = in[0];
    return 0;
}

bool W2A16MoePlugin::supportsFormatCombination(
    int32_t pos, DynamicPluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
    return inOut != nullptr && nbInputs == kInputs && nbOutputs == 1 && pos >= 0 && pos <= kInputs
        && inOut[pos].desc.type == kTypes[pos] && inOut[pos].desc.format == PluginFormat::kLINEAR;
}

int32_t W2A16MoePlugin::configurePlugin(
    DynamicPluginTensorDesc const* in, int32_t nbInputs, DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept
{
    if (in == nullptr || out == nullptr || nbInputs != kInputs || nbOutputs != 1)
        return -1;
#ifdef CUTE_DSL_W2A16_MOE_ENABLED
    // The runner sizes its workspace per token count and rejects more than kMaxNumTokens rows, so the whole
    // profile range must be implementable for getWorkspaceSize() to cover every enqueue.
    if (in[0].min.d[0] < 1 || in[0].max.d[0] > CuteDslW2A16MoeRunner::kMaxNumTokens)
    {
        LOG_ERROR("W2A16MoePlugin: token profile [%ld, %ld] is outside [1, %d]", static_cast<long>(in[0].min.d[0]),
            static_cast<long>(in[0].max.d[0]), CuteDslW2A16MoeRunner::kMaxNumTokens);
        return -1;
    }
    int32_t const smVersion = getSMVersion();
    int32_t const numSms = getDeviceMultiProcessorCount();
    if (!CuteDslW2A16MoeRunner::isHardwareSupported(smVersion, numSms))
    {
        LOG_ERROR(
            "W2A16MoePlugin: SM%d with %d SMs is not supported by the linked CuTe DSL artifact", smVersion, numSms);
        return -1;
    }
    if (CuteDslW2A16MoeRunner::getWorkspaceSize(in[0].max.d[0], mNumExperts, mMoeInterSize, smVersion, numSms) == 0)
    {
        LOG_ERROR("W2A16MoePlugin: E=%d I=%d is unsupported", mNumExperts, mMoeInterSize);
        return -1;
    }
    std::array<PluginTensorDesc, kInputs> descriptors{};
    for (int32_t i = 0; i < kInputs; ++i)
    {
        descriptors[i] = in[i].desc;
        descriptors[i].dims = in[i].opt;
    }
    PluginTensorDesc output = out[0].desc;
    output.dims = out[0].opt;
    return validShape(descriptors.data(), &output, mNumExperts, mMoeInterSize) ? 0 : -1;
#else
    return -1;
#endif
}

size_t W2A16MoePlugin::getWorkspaceSize(DynamicPluginTensorDesc const* in, int32_t nbInputs,
    DynamicPluginTensorDesc const*, int32_t nbOutputs) const noexcept
{
#ifdef CUTE_DSL_W2A16_MOE_ENABLED
    if (in == nullptr || nbInputs != kInputs || nbOutputs != 1)
        return 0;
    size_t const bytes = CuteDslW2A16MoeRunner::getWorkspaceSize(
        in[0].max.d[0], mNumExperts, mMoeInterSize, getSMVersion(), getDeviceMultiProcessorCount());
    return bytes == 0 ? 0 : bytes + kWorkspaceAlignment;
#else
    return 0;
#endif
}

int32_t W2A16MoePlugin::enqueue(PluginTensorDesc const* inDesc, PluginTensorDesc const* outDesc, void const* const* in,
    void* const* out, void* workspace, cudaStream_t stream) noexcept
{
#ifdef CUTE_DSL_W2A16_MOE_ENABLED
    if (!validShape(inDesc, outDesc, mNumExperts, mMoeInterSize) || in == nullptr || out == nullptr)
    {
        LOG_ERROR("W2A16MoePlugin: invalid runtime tensor shape for E=%d I=%d", mNumExperts, mMoeInterSize);
        return -1;
    }
    CuteDslW2A16MoeParams params{};
    params.numTokens = inDesc[0].dims.d[0];
    params.numExperts = mNumExperts;
    params.moeInterSize = mMoeInterSize;
    params.hiddenStates = in[0];
    params.fc1QWeights = in[1];
    params.fc2QWeights = in[2];
    params.fc1BlocksScale = in[3];
    params.fc2BlocksScale = in[4];
    params.unitScales = static_cast<float const*>(in[5]);
    params.lookupTable = in[6];
    params.topkIds = static_cast<int32_t const*>(in[7]);
    params.topkWeights = static_cast<float const*>(in[8]);
    params.output = out[0];
    int32_t const smVersion = getSMVersion();
    int32_t const numSms = getDeviceMultiProcessorCount();
    auto const bytes
        = CuteDslW2A16MoeRunner::getWorkspaceSize(params.numTokens, mNumExperts, mMoeInterSize, smVersion, numSms);
    if (bytes == 0 || workspace == nullptr)
    {
        LOG_ERROR(
            "W2A16MoePlugin: M=%d is not implementable for E=%d I=%d", params.numTokens, mNumExperts, mMoeInterSize);
        return -1;
    }
    auto const address = reinterpret_cast<uintptr_t>(workspace);
    void* aligned
        = reinterpret_cast<void*>((address + kWorkspaceAlignment - 1) / kWorkspaceAlignment * kWorkspaceAlignment);
    int32_t const status = mRunner.run(params, aligned, bytes, smVersion, numSms, stream);
    if (status != 0)
    {
        LOG_ERROR("W2A16MoePlugin: runner failed for M=%d E=%d I=%d: CUDA status %d (%s); workspace=%p",
            params.numTokens, mNumExperts, mMoeInterSize, status, cudaGetErrorString(static_cast<cudaError_t>(status)),
            workspace);
    }
    return status;
#else
    return -1;
#endif
}

int32_t W2A16MoePlugin::onShapeChange(
    PluginTensorDesc const* in, int32_t nbInputs, PluginTensorDesc const* out, int32_t nbOutputs) noexcept
{
    if (nbInputs != kInputs || nbOutputs != 1 || !validShape(in, out, mNumExperts, mMoeInterSize))
        return -1;
#ifdef CUTE_DSL_W2A16_MOE_ENABLED
    CuteDslW2A16MoeParams params{};
    params.numTokens = in[0].dims.d[0];
    params.numExperts = mNumExperts;
    params.moeInterSize = mMoeInterSize;
    return CuteDslW2A16MoeRunner::ensureKernelModules(params, getSMVersion(), getDeviceMultiProcessorCount(), nullptr)
        ? 0
        : -1;
#else
    return -1;
#endif
}

IPluginV3* W2A16MoePlugin::attachToContext(IPluginResourceContext*) noexcept
{
    return clone();
}
PluginFieldCollection const* W2A16MoePlugin::getFieldsToSerialize() noexcept
{
    return &mFields;
}

W2A16MoePluginCreator::W2A16MoePluginCreator()
{
    mAttributes.emplace_back("num_experts", nullptr, PluginFieldType::kINT32, 1);
    mAttributes.emplace_back("moe_inter_size", nullptr, PluginFieldType::kINT32, 1);
    mFields = {static_cast<int32_t>(mAttributes.size()), mAttributes.data()};
}
char const* W2A16MoePluginCreator::getPluginName() const noexcept
{
    return kName;
}
char const* W2A16MoePluginCreator::getPluginVersion() const noexcept
{
    return kVersion;
}
PluginFieldCollection const* W2A16MoePluginCreator::getFieldNames() noexcept
{
    return &mFields;
}
char const* W2A16MoePluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}
void W2A16MoePluginCreator::setPluginNamespace(char const* name) noexcept
{
    mNamespace = name ? name : "";
}

IPluginV3* W2A16MoePluginCreator::createPlugin(
    char const* name, PluginFieldCollection const* fc, TensorRTPhase) noexcept
{
    if (name == nullptr || fc == nullptr)
        return nullptr;
    int32_t experts = 0, interSize = 0;
    for (int32_t i = 0; i < fc->nbFields; ++i)
    {
        auto const& field = fc->fields[i];
        if (field.data == nullptr || field.type != PluginFieldType::kINT32 || field.length != 1)
            return nullptr;
        auto const value = *static_cast<int32_t const*>(field.data);
        if (std::string(field.name) == "num_experts")
            experts = value;
        else if (std::string(field.name) == "moe_inter_size")
            interSize = value;
        else
            return nullptr;
    }
    if (experts <= 0 || interSize <= 0)
        return nullptr;
    try
    {
        auto* plugin = new W2A16MoePlugin(name, experts, interSize);
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }
    catch (std::exception const& error)
    {
        LOG_ERROR("W2A16MoePluginCreator: %s", error.what());
        return nullptr;
    }
}
} // namespace trt_edgellm::plugins
