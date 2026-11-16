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

#include "dualGemmPlugin.h"

#include "common/cudaUtils.h"
#include "common/logger.h"
#include "kernels/dualGemmKernels/cuteDslDualGemmRunner.h"

#include <NvInfer.h>
#include <array>
#include <cstdint>
#include <cuda_runtime.h>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>

using namespace nvinfer1;

namespace trt_edgellm
{
namespace plugins
{
using nvinfer1::DataType;

namespace
{

constexpr char const* kPluginName{"DualGemmPlugin"};
constexpr char const* kPluginVersion{"1"};

constexpr int32_t kInActivation{0};
constexpr int32_t kInPackedWeight{1};
constexpr int32_t kNbInputs{2};
constexpr int32_t kOutOutput{kNbInputs};

constexpr int32_t kActivationSilu{0};

constexpr int32_t kFieldInterSize{0};
constexpr int32_t kFieldHiddenSize{1};
constexpr int32_t kFieldActivation{2};
constexpr int32_t kNbFields{3};

// The activation is [..., K]; every leading dimension is a token dimension.
bool isSupportedActivationRank(int32_t rank) noexcept
{
    return rank >= 2 && rank <= 4;
}

int64_t getTokenCount(Dims const& dims) noexcept
{
    int64_t tokens = 1;
    for (int32_t i = 0; i + 1 < dims.nbDims; ++i)
    {
        tokens *= dims.d[i];
    }
    return tokens;
}

bool sameTokenShape(Dims const& output, Dims const& activation) noexcept
{
    if (output.nbDims != activation.nbDims)
    {
        return false;
    }
    for (int32_t i = 0; i + 1 < output.nbDims; ++i)
    {
        if (output.d[i] != activation.d[i])
        {
            return false;
        }
    }
    return true;
}

} // namespace

DualGemmPlugin::DualGemmPlugin(std::string const& name, int32_t interSize, int32_t hiddenSize, int32_t activation)
    : mLayerName(name)
    , mInterSize(interSize)
    , mHiddenSize(hiddenSize)
    , mActivation(activation)
{
    validateAttributes();
}

DualGemmPlugin::DualGemmPlugin(std::string const& name, PluginFieldCollection const* fc)
    : mLayerName(name)
{
    if (fc == nullptr || fc->fields == nullptr || fc->nbFields <= 0)
    {
        throw std::invalid_argument("DualGemmPlugin: plugin field collection must not be empty");
    }
    std::array<bool, kNbFields> fieldsSeen{};
    auto readIntField = [&fieldsSeen](PluginField const& field, int32_t index) {
        if (field.data == nullptr || field.type != PluginFieldType::kINT32 || field.length != 1)
        {
            throw std::invalid_argument("DualGemmPlugin: attributes must be scalar INT32 fields");
        }
        if (fieldsSeen[index])
        {
            throw std::invalid_argument("DualGemmPlugin: duplicate plugin attribute");
        }
        fieldsSeen[index] = true;
        return *static_cast<int32_t const*>(field.data);
    };
    for (int32_t i = 0; i < fc->nbFields; ++i)
    {
        PluginField const& field = fc->fields[i];
        if (field.name == nullptr)
        {
            throw std::invalid_argument("DualGemmPlugin: plugin attribute name must not be null");
        }
        std::string const fieldName(field.name);
        if (fieldName == "inter_size")
        {
            mInterSize = readIntField(field, kFieldInterSize);
        }
        else if (fieldName == "hidden_size")
        {
            mHiddenSize = readIntField(field, kFieldHiddenSize);
        }
        else if (fieldName == "activation")
        {
            mActivation = readIntField(field, kFieldActivation);
        }
        else
        {
            throw std::invalid_argument("DualGemmPlugin: unknown plugin attribute " + fieldName);
        }
    }
    if (!fieldsSeen[kFieldInterSize] || !fieldsSeen[kFieldHiddenSize] || !fieldsSeen[kFieldActivation])
    {
        throw std::invalid_argument("DualGemmPlugin: inter_size, hidden_size, and activation attributes are required");
    }
    validateAttributes();
}

void DualGemmPlugin::validateAttributes() const
{
    if (mActivation != kActivationSilu)
    {
        throw std::invalid_argument("DualGemmPlugin: activation must be SiLU (0)");
    }
    if (!kernels::dual_gemm_blackwell::isSupportedProblemShape(mInterSize, mHiddenSize))
    {
        throw std::invalid_argument(
            "DualGemmPlugin: inter_size must be a positive multiple of 128 and hidden_size a positive multiple of 64");
    }
}

IPluginCapability* DualGemmPlugin::getCapabilityInterface(PluginCapabilityType type) noexcept
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

IPluginV3* DualGemmPlugin::clone() noexcept
{
    try
    {
        auto plugin = std::make_unique<DualGemmPlugin>(mLayerName, mInterSize, mHiddenSize, mActivation);
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin.release();
    }
    catch (std::exception const& error)
    {
        LOG_ERROR("Failed to clone DualGemmPlugin: %s", error.what());
        return nullptr;
    }
}

char const* DualGemmPlugin::getPluginName() const noexcept
{
    return kPluginName;
}

char const* DualGemmPlugin::getPluginVersion() const noexcept
{
    return kPluginVersion;
}

char const* DualGemmPlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

void DualGemmPlugin::setPluginNamespace(char const* pluginNamespace) noexcept
{
    mNamespace = pluginNamespace == nullptr ? "" : pluginNamespace;
}

int32_t DualGemmPlugin::getNbOutputs() const noexcept
{
    return 1;
}

int32_t DualGemmPlugin::getOutputDataTypes(
    DataType* outputTypes, int32_t nbOutputs, DataType const* inputTypes, int32_t nbInputs) const noexcept
{
    if (outputTypes == nullptr || inputTypes == nullptr || nbOutputs != 1 || nbInputs != kNbInputs
        || inputTypes[kInActivation] != DataType::kHALF)
    {
        return -1;
    }
    outputTypes[0] = DataType::kHALF;
    return 0;
}

int32_t DualGemmPlugin::getOutputShapes(DimsExprs const* inputs, int32_t nbInputs, DimsExprs const* shapeInputs,
    int32_t nbShapeInputs, DimsExprs* outputs, int32_t nbOutputs, IExprBuilder& exprBuilder) noexcept
{
    (void) shapeInputs;
    (void) nbShapeInputs;
    if (inputs == nullptr || outputs == nullptr || nbInputs != kNbInputs || nbOutputs != 1)
    {
        return -1;
    }
    int32_t const rank = inputs[kInActivation].nbDims;
    if (!isSupportedActivationRank(rank))
    {
        return -1;
    }
    outputs[0] = inputs[kInActivation];
    outputs[0].d[rank - 1] = exprBuilder.constant(mInterSize);
    return 0;
}

bool DualGemmPlugin::validateTensorDesc(int32_t pos, PluginTensorDesc const& desc) const noexcept
{
    if (desc.format != TensorFormat::kLINEAR || desc.type != DataType::kHALF)
    {
        return false;
    }
    switch (pos)
    {
    case kInActivation:
        return isSupportedActivationRank(desc.dims.nbDims) && desc.dims.d[desc.dims.nbDims - 1] == mHiddenSize;
    case kInPackedWeight:
        return desc.dims.nbDims == 2 && desc.dims.d[0] == 2 * mInterSize && desc.dims.d[1] == mHiddenSize;
    case kOutOutput:
        return isSupportedActivationRank(desc.dims.nbDims) && desc.dims.d[desc.dims.nbDims - 1] == mInterSize;
    default: return false;
    }
}

bool DualGemmPlugin::supportsFormatCombination(
    int32_t pos, DynamicPluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
    return inOut != nullptr && nbInputs == kNbInputs && nbOutputs == 1 && pos >= 0 && pos <= kOutOutput
        && validateTensorDesc(pos, inOut[pos].desc);
}

int32_t DualGemmPlugin::configurePlugin(
    DynamicPluginTensorDesc const* in, int32_t nbInputs, DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept
{
    try
    {
        if (in == nullptr || out == nullptr || nbInputs != kNbInputs || nbOutputs != 1)
        {
            return -1;
        }
        for (int32_t pos = 0; pos < kNbInputs; ++pos)
        {
            if (!validateTensorDesc(pos, in[pos].desc))
            {
                LOG_ERROR("DualGemmPlugin: invalid input descriptor at position %d", pos);
                return -1;
            }
        }
        if (!validateTensorDesc(kOutOutput, out[0].desc))
        {
            LOG_ERROR("DualGemmPlugin: output must be an FP16 [..., inter_size] tensor");
            return -1;
        }
        int32_t const smVersion = getSMVersion();
        if (!kernels::CuteDslDualGemmRunner::canImplement(smVersion))
        {
            LOG_ERROR("DualGemmPlugin: no CuTe DSL dual GEMM kernels for SM%d in this build", smVersion);
            return -1;
        }
        // Load every AOT variant the optimization profile can reach so that engine execution, which may be
        // captured into a CUDA graph, never loads a module.
        int64_t const profileMaxM = getTokenCount(in[kInActivation].max);
        if (profileMaxM <= 0 || profileMaxM > std::numeric_limits<int32_t>::max())
        {
            LOG_ERROR("DualGemmPlugin: unsupported profile token count");
            return -1;
        }
        for (int32_t const m :
            {1, kernels::dual_gemm_blackwell::kDecodeMaxM + 1, kernels::dual_gemm_blackwell::kPrefillMaxM + 1})
        {
            if (m > profileMaxM)
            {
                break;
            }
            cudaError_t const error = kernels::CuteDslDualGemmRunner::prepare(m, nullptr);
            if (error != cudaSuccess)
            {
                LOG_ERROR(
                    "DualGemmPlugin: loading the dual GEMM module for M=%d failed: %s", m, cudaGetErrorString(error));
                return -1;
            }
        }
        return 0;
    }
    catch (std::exception const& error)
    {
        LOG_ERROR("DualGemmPlugin configurePlugin failed: %s", error.what());
        return -1;
    }
}

size_t DualGemmPlugin::getWorkspaceSize(DynamicPluginTensorDesc const* inputs, int32_t nbInputs,
    DynamicPluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept
{
    (void) inputs;
    (void) nbInputs;
    (void) outputs;
    (void) nbOutputs;
    return 0;
}

int32_t DualGemmPlugin::enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc,
    void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    (void) workspace;
    if (inputDesc == nullptr || outputDesc == nullptr || inputs == nullptr || outputs == nullptr
        || inputs[kInActivation] == nullptr || inputs[kInPackedWeight] == nullptr || outputs[0] == nullptr)
    {
        return -1;
    }
    for (int32_t pos = 0; pos < kNbInputs; ++pos)
    {
        if (!validateTensorDesc(pos, inputDesc[pos]))
        {
            return -1;
        }
    }
    Dims const& activationDims = inputDesc[kInActivation].dims;
    int64_t const m = getTokenCount(activationDims);
    if (!validateTensorDesc(kOutOutput, outputDesc[0]) || !sameTokenShape(outputDesc[0].dims, activationDims) || m < 0
        || m > std::numeric_limits<int32_t>::max())
    {
        LOG_ERROR("DualGemmPlugin: invalid runtime token count or output shape");
        return -1;
    }
    if (m == 0)
    {
        // Empty token batch: nothing to compute.
        return 0;
    }
    kernels::DualGemmBlackwellParams const params{
        inputs[kInActivation], inputs[kInPackedWeight], outputs[0], static_cast<int32_t>(m), mInterSize, mHiddenSize};
    cudaError_t const error = kernels::CuteDslDualGemmRunner::run(params, stream);
    if (error != cudaSuccess)
    {
        LOG_ERROR("DualGemmPlugin: dual GEMM launch failed for M=%lld: %s", static_cast<long long>(m),
            cudaGetErrorString(error));
        return -1;
    }
    return 0;
}

int32_t DualGemmPlugin::onShapeChange(
    PluginTensorDesc const* in, int32_t nbInputs, PluginTensorDesc const* out, int32_t nbOutputs) noexcept
{
    if (in == nullptr || out == nullptr || nbInputs != kNbInputs || nbOutputs != 1)
    {
        return -1;
    }
    for (int32_t pos = 0; pos < kNbInputs; ++pos)
    {
        if (!validateTensorDesc(pos, in[pos]))
        {
            return -1;
        }
    }
    int64_t const m = getTokenCount(in[kInActivation].dims);
    if (!validateTensorDesc(kOutOutput, out[0]) || !sameTokenShape(out[0].dims, in[kInActivation].dims) || m < 0
        || m > std::numeric_limits<int32_t>::max())
    {
        return -1;
    }
    if (m == 0)
    {
        return 0;
    }
    // Runtime engines skip configurePlugin(); make sure the module for this shape is resident before enqueue
    // (this runs outside CUDA graph capture, see the module loader).
    cudaError_t const error = kernels::CuteDslDualGemmRunner::prepare(static_cast<int32_t>(m), nullptr);
    if (error != cudaSuccess)
    {
        LOG_ERROR("DualGemmPlugin: loading the dual GEMM module for M=%lld failed: %s", static_cast<long long>(m),
            cudaGetErrorString(error));
        return -1;
    }
    return 0;
}

IPluginV3* DualGemmPlugin::attachToContext(IPluginResourceContext* context) noexcept
{
    (void) context;
    return clone();
}

PluginFieldCollection const* DualGemmPlugin::getFieldsToSerialize() noexcept
{
    mDataToSerialize.clear();
    mDataToSerialize.emplace_back("inter_size", &mInterSize, PluginFieldType::kINT32, 1);
    mDataToSerialize.emplace_back("hidden_size", &mHiddenSize, PluginFieldType::kINT32, 1);
    mDataToSerialize.emplace_back("activation", &mActivation, PluginFieldType::kINT32, 1);
    mFCToSerialize.nbFields = static_cast<int32_t>(mDataToSerialize.size());
    mFCToSerialize.fields = mDataToSerialize.data();
    return &mFCToSerialize;
}

PluginFieldCollection DualGemmPluginCreator::mFieldCollection{};
std::vector<PluginField> DualGemmPluginCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(DualGemmPluginCreator);

DualGemmPluginCreator::DualGemmPluginCreator()
{
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back("inter_size", nullptr, PluginFieldType::kINT32, 1);
    mPluginAttributes.emplace_back("hidden_size", nullptr, PluginFieldType::kINT32, 1);
    mPluginAttributes.emplace_back("activation", nullptr, PluginFieldType::kINT32, 1);
    mFieldCollection.nbFields = static_cast<int32_t>(mPluginAttributes.size());
    mFieldCollection.fields = mPluginAttributes.data();
}

char const* DualGemmPluginCreator::getPluginName() const noexcept
{
    return kPluginName;
}

char const* DualGemmPluginCreator::getPluginVersion() const noexcept
{
    return kPluginVersion;
}

PluginFieldCollection const* DualGemmPluginCreator::getFieldNames() noexcept
{
    return &mFieldCollection;
}

char const* DualGemmPluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

void DualGemmPluginCreator::setPluginNamespace(char const* pluginNamespace) noexcept
{
    mNamespace = pluginNamespace == nullptr ? "" : pluginNamespace;
}

IPluginV3* DualGemmPluginCreator::createPlugin(
    char const* name, PluginFieldCollection const* fc, TensorRTPhase phase) noexcept
{
    (void) phase;
    try
    {
        auto plugin = std::make_unique<DualGemmPlugin>(name == nullptr ? kPluginName : name, fc);
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin.release();
    }
    catch (std::exception const& error)
    {
        LOG_ERROR("Failed to create DualGemmPlugin: %s", error.what());
        return nullptr;
    }
}

} // namespace plugins
} // namespace trt_edgellm
