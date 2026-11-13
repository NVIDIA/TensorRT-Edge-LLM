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

#include "gemma4AudioAttentionPlugin.h"

#include "common/checkMacros.h"
#include "common/logger.h"
#include "common/tensor.h"
#include "common/trtInterop.h"
#include "kernels/audioAttentionKernels/gemma4AudioAttention.h"
#include "plugins/utils/pluginUtils.h"

#include <cassert>
#include <cstdint>
#include <mutex>
#include <vector>

using namespace nvinfer1;

namespace trt_edgellm
{
namespace plugins
{
using nvinfer1::DataType;

namespace
{
constexpr char const* kPLUGIN_VERSION{"1"};
constexpr char const* kPLUGIN_NAME{"Gemma4AudioAttentionPlugin"};

// Input tensor indices.
constexpr int32_t kIN_QRAW_IDX{0};
constexpr int32_t kIN_KRAW_IDX{1};
constexpr int32_t kIN_V_IDX{2};
constexpr int32_t kIN_GAMMA_IDX{3};
constexpr int32_t kIN_RELKEY_IDX{4};
constexpr int32_t kIN_VALID_IDX{5};
constexpr int32_t kIN_SEQLEN_IDX{6};

constexpr int32_t kNUM_INPUTS{7};
constexpr int32_t kNUM_OUTPUTS{1};
constexpr int32_t kOUT_IDX{0};
} // namespace

// Static class fields initialization.
PluginFieldCollection Gemma4AudioAttentionPluginCreator::mFieldCollection{};
std::vector<PluginField> Gemma4AudioAttentionPluginCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(Gemma4AudioAttentionPluginCreator);

// ---------------------------------------------------------------------------
// Plugin construction
// ---------------------------------------------------------------------------

Gemma4AudioAttentionPlugin::Gemma4AudioAttentionPlugin(
    std::string const& name, int32_t chunkSize, int32_t leftHorizon, int32_t contextSize, float logitCap)
    : mLayerName(name)
    , mChunkSize(chunkSize)
    , mLeftHorizon(leftHorizon)
    , mContextSize(contextSize)
    , mLogitCap(logitCap)
{
}

Gemma4AudioAttentionPlugin::Gemma4AudioAttentionPlugin(std::string const& name, PluginFieldCollection const* fc)
    : mLayerName(name)
{
    mChunkSize = parsePluginScalarField<int32_t>("chunk_size", fc).value_or(12);
    mLeftHorizon = parsePluginScalarField<int32_t>("left_horizon", fc).value_or(12);
    mContextSize = parsePluginScalarField<int32_t>("context_size", fc).value_or(24);
    mLogitCap = parsePluginScalarField<float>("logit_cap", fc).value_or(50.0f);
}

Gemma4AudioAttentionPlugin::~Gemma4AudioAttentionPlugin() = default;

// ---------------------------------------------------------------------------
// IPluginV3
// ---------------------------------------------------------------------------

IPluginCapability* Gemma4AudioAttentionPlugin::getCapabilityInterface(PluginCapabilityType type) noexcept
{
    try
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
    catch (std::exception const&)
    {
        return nullptr;
    }
}

IPluginV3* Gemma4AudioAttentionPlugin::clone() noexcept
{
    try
    {
        auto* p = new Gemma4AudioAttentionPlugin(mLayerName, mChunkSize, mLeftHorizon, mContextSize, mLogitCap);
        p->setPluginNamespace(mNamespace.c_str());
        return p;
    }
    catch (...)
    {
        return nullptr;
    }
}

// ---------------------------------------------------------------------------
// IPluginV3OneCore — metadata
// ---------------------------------------------------------------------------

char const* Gemma4AudioAttentionPlugin::getPluginName() const noexcept
{
    return kPLUGIN_NAME;
}

char const* Gemma4AudioAttentionPlugin::getPluginVersion() const noexcept
{
    return kPLUGIN_VERSION;
}

char const* Gemma4AudioAttentionPlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

void Gemma4AudioAttentionPlugin::setPluginNamespace(char const* pluginNamespace) noexcept
{
    mNamespace = pluginNamespace ? pluginNamespace : "";
}

// ---------------------------------------------------------------------------
// IPluginV3OneBuild — shape / format negotiation
// ---------------------------------------------------------------------------

int32_t Gemma4AudioAttentionPlugin::getNbOutputs() const noexcept
{
    return kNUM_OUTPUTS;
}

int32_t Gemma4AudioAttentionPlugin::getOutputDataTypes(DataType* outputTypes, [[maybe_unused]] int32_t nbOutputs,
    DataType const* inputTypes, [[maybe_unused]] int32_t nbInputs) const noexcept
{
    try
    {
        assert(nbOutputs == kNUM_OUTPUTS);
        // Output follows the Q/K/V element type.
        outputTypes[kOUT_IDX] = inputTypes[kIN_QRAW_IDX];
        return 0;
    }
    catch (std::exception const&)
    {
        return -1;
    }
}

int32_t Gemma4AudioAttentionPlugin::getOutputShapes(DimsExprs const* inputs, [[maybe_unused]] int32_t nbInputs,
    DimsExprs const* /* shapeInputs */, int32_t /* nbShapeInputs */, DimsExprs* outputs,
    [[maybe_unused]] int32_t nbOutputs, IExprBuilder& /* exprBuilder */) noexcept
{
    try
    {
        assert(nbInputs == kNUM_INPUTS);
        assert(nbOutputs == kNUM_OUTPUTS);
        // Output shape == qRaw shape [B, S, H, D].
        outputs[kOUT_IDX] = inputs[kIN_QRAW_IDX];
        return 0;
    }
    catch (std::exception const&)
    {
        return -1;
    }
}

bool Gemma4AudioAttentionPlugin::supportsFormatCombination(
    int32_t pos, DynamicPluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
    if (nbInputs != kNUM_INPUTS || nbOutputs != kNUM_OUTPUTS)
    {
        return false;
    }

    auto const& desc = inOut[pos].desc;
    if (desc.format != TensorFormat::kLINEAR)
    {
        return false;
    }

    // Positions: 0=qRaw, 1=kRaw, 2=v, 3=gamma, 4=relKey, 5=valid, 6=seqLen, 7=out
    switch (pos)
    {
    case kIN_QRAW_IDX:
        // Q: half, bf16, or float; 4D [B, S, H, D]
        return (desc.type == DataType::kHALF || desc.type == DataType::kBF16 || desc.type == DataType::kFLOAT)
            && desc.dims.nbDims == 4;
    case kIN_KRAW_IDX:
    case kIN_V_IDX:
        // K, V must match Q type and shape.
        return desc.type == inOut[kIN_QRAW_IDX].desc.type && desc.dims.nbDims == 4;
    case kIN_GAMMA_IDX:
        // gamma: float, 1D [D]
        return desc.type == DataType::kFLOAT && desc.dims.nbDims == 1;
    case kIN_RELKEY_IDX:
        // relKey: same type as Q, 3D [P, H, D]
        return desc.type == inOut[kIN_QRAW_IDX].desc.type && desc.dims.nbDims == 3;
    case kIN_VALID_IDX:
        // valid: bool, 2D [B, S]
        return desc.type == DataType::kBOOL && desc.dims.nbDims == 2;
    case kIN_SEQLEN_IDX:
        // seqLen carrier: int32, 1D [1]
        return desc.type == DataType::kINT32 && desc.dims.nbDims == 1;
    default:
    {
        // Output: same type as Q, 4D [B, S, H, D]
        int32_t const outPos = pos - nbInputs;
        if (outPos == kOUT_IDX)
        {
            return desc.type == inOut[kIN_QRAW_IDX].desc.type && desc.dims.nbDims == 4;
        }
        return false;
    }
    }
}

int32_t Gemma4AudioAttentionPlugin::configurePlugin([[maybe_unused]] DynamicPluginTensorDesc const* in,
    [[maybe_unused]] int32_t nbInputs, [[maybe_unused]] DynamicPluginTensorDesc const* out,
    [[maybe_unused]] int32_t nbOutputs) noexcept
{
    return 0;
}

size_t Gemma4AudioAttentionPlugin::getWorkspaceSize([[maybe_unused]] DynamicPluginTensorDesc const* inputs,
    [[maybe_unused]] int32_t nbInputs, [[maybe_unused]] DynamicPluginTensorDesc const* outputs,
    [[maybe_unused]] int32_t nbOutputs) const noexcept
{
    // The kernel uses only shared memory; no additional workspace needed.
    return 0;
}

// ---------------------------------------------------------------------------
// IPluginV3OneRuntime — execution
// ---------------------------------------------------------------------------

int32_t Gemma4AudioAttentionPlugin::enqueue(PluginTensorDesc const* inputDesc,
    [[maybe_unused]] PluginTensorDesc const* outputDesc, void const* const* inputs, void* const* outputs,
    [[maybe_unused]] void* workspace, cudaStream_t stream) noexcept
{
    try
    {
        // Extract shapes from qRaw descriptor: [B, S, H, D].
        auto const& qDesc = inputDesc[kIN_QRAW_IDX];
        int32_t const B = static_cast<int32_t>(qDesc.dims.d[0]);
        int32_t const S = static_cast<int32_t>(qDesc.dims.d[1]);
        int32_t const H = static_cast<int32_t>(qDesc.dims.d[2]);
        int32_t const D = static_cast<int32_t>(qDesc.dims.d[3]);

        // relKey shape: [P, H, D]
        int32_t const P = static_cast<int32_t>(inputDesc[kIN_RELKEY_IDX].dims.d[0]);

        // Build rt::Tensor wrappers (non-owning).
        rt::Coords const qkvShape{B, S, H, D};
        rt::Tensor qRaw(const_cast<void*>(inputs[kIN_QRAW_IDX]), qkvShape, rt::DeviceType::kGPU, fromTrt(qDesc.type));
        rt::Tensor kRaw(const_cast<void*>(inputs[kIN_KRAW_IDX]), qkvShape, rt::DeviceType::kGPU, fromTrt(qDesc.type));
        rt::Tensor v(const_cast<void*>(inputs[kIN_V_IDX]), qkvShape, rt::DeviceType::kGPU, fromTrt(qDesc.type));

        rt::Tensor gamma(const_cast<void*>(inputs[kIN_GAMMA_IDX]), rt::Coords{D}, rt::DeviceType::kGPU,
            fromTrt(inputDesc[kIN_GAMMA_IDX].type));
        rt::Tensor relKey(const_cast<void*>(inputs[kIN_RELKEY_IDX]), rt::Coords{P, H, D}, rt::DeviceType::kGPU,
            fromTrt(inputDesc[kIN_RELKEY_IDX].type));
        rt::Tensor valid(const_cast<void*>(inputs[kIN_VALID_IDX]), rt::Coords{B, S}, rt::DeviceType::kGPU,
            fromTrt(inputDesc[kIN_VALID_IDX].type));

        rt::Tensor out(outputs[kOUT_IDX], qkvShape, rt::DeviceType::kGPU, fromTrt(qDesc.type));

        kernel::Gemma4AudioAttentionParams params{};
        params.chunkSize = mChunkSize;
        params.leftHorizon = mLeftHorizon;
        params.contextSize = mContextSize;
        params.logitCap = mLogitCap;

        kernel::gemma4AudioAttentionForward(qRaw, kRaw, v, gamma, relKey, valid, out, params, stream);
        return 0;
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("Gemma4AudioAttentionPlugin enqueue failed: %s", e.what());
        return -1;
    }
}

int32_t Gemma4AudioAttentionPlugin::onShapeChange([[maybe_unused]] PluginTensorDesc const* in,
    [[maybe_unused]] int32_t nbInputs, [[maybe_unused]] PluginTensorDesc const* out,
    [[maybe_unused]] int32_t nbOutputs) noexcept
{
    return 0;
}

IPluginV3* Gemma4AudioAttentionPlugin::attachToContext([[maybe_unused]] IPluginResourceContext* context) noexcept
{
    return clone();
}

PluginFieldCollection const* Gemma4AudioAttentionPlugin::getFieldsToSerialize() noexcept
{
    mDataToSerialize.clear();
    mDataToSerialize.emplace_back("chunk_size", &mChunkSize, PluginFieldType::kINT32, 1);
    mDataToSerialize.emplace_back("left_horizon", &mLeftHorizon, PluginFieldType::kINT32, 1);
    mDataToSerialize.emplace_back("context_size", &mContextSize, PluginFieldType::kINT32, 1);
    mDataToSerialize.emplace_back("logit_cap", &mLogitCap, PluginFieldType::kFLOAT32, 1);
    mFCToSerialize.nbFields = static_cast<int32_t>(mDataToSerialize.size());
    mFCToSerialize.fields = mDataToSerialize.data();
    return &mFCToSerialize;
}

// ---------------------------------------------------------------------------
// Creator
// ---------------------------------------------------------------------------

Gemma4AudioAttentionPluginCreator::Gemma4AudioAttentionPluginCreator()
{
    static std::mutex sMutex;
    std::lock_guard<std::mutex> lock(sMutex);

    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("chunk_size", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("left_horizon", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("context_size", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("logit_cap", nullptr, PluginFieldType::kFLOAT32, 1));
    mFieldCollection.nbFields = mPluginAttributes.size();
    mFieldCollection.fields = mPluginAttributes.data();
}

char const* Gemma4AudioAttentionPluginCreator::getPluginName() const noexcept
{
    return kPLUGIN_NAME;
}

char const* Gemma4AudioAttentionPluginCreator::getPluginVersion() const noexcept
{
    return kPLUGIN_VERSION;
}

PluginFieldCollection const* Gemma4AudioAttentionPluginCreator::getFieldNames() noexcept
{
    return &mFieldCollection;
}

char const* Gemma4AudioAttentionPluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

void Gemma4AudioAttentionPluginCreator::setPluginNamespace(char const* libNamespace) noexcept
{
    mNamespace = libNamespace ? libNamespace : "";
}

IPluginV3* Gemma4AudioAttentionPluginCreator::createPlugin(
    char const* name, PluginFieldCollection const* fc, [[maybe_unused]] TensorRTPhase phase) noexcept
{
    try
    {
        auto* plugin = new Gemma4AudioAttentionPlugin(std::string(name), fc);
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("Failed to create Gemma4AudioAttentionPlugin: %s", e.what());
    }
    return nullptr;
}

} // namespace plugins
} // namespace trt_edgellm
