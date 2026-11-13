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

#include "nvfp4DualGemmPlugin.h"

#include "common/logger.h"
#include "kernels/dualGemmKernels/cuteDslNvfp4DualGemmRunner.h"

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
namespace
{

constexpr char const* kPluginName{"Nvfp4DualGemmPlugin"};
constexpr char const* kPluginVersion{"1"};

constexpr int32_t kInActivation{0};
constexpr int32_t kInPackedWeight{1};
constexpr int32_t kInActivationScale{2};
constexpr int32_t kInWeightScale{3};
constexpr int32_t kInAlpha{4};
constexpr int32_t kInGlobalScale{5};

constexpr int32_t kActivationSilu{0};

constexpr int32_t kFieldInterSize{0};
constexpr int32_t kFieldHiddenSize{1};
constexpr int32_t kFieldActivation{2};
constexpr int32_t kFieldQuantizedOutput{3};
constexpr int32_t kNbFields{4};

// The packed activation is [..., K/2]; every leading dimension is a token dimension.
bool isSupportedActivationRank(int32_t rank) noexcept
{
    return rank >= 2 && rank <= 4;
}

int64_t getTokenCount(Dims const& dims) noexcept
{
    int64_t tokens = 1;
    for (int32_t i = 0; i + 1 < dims.nbDims; ++i)
    {
        if (dims.d[i] <= 0 || tokens > std::numeric_limits<int32_t>::max() / dims.d[i])
        {
            return -1;
        }
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

Nvfp4DualGemmPlugin::Nvfp4DualGemmPlugin(
    std::string const& name, int32_t interSize, int32_t hiddenSize, int32_t activation, int32_t quantizedOutput)
    : mLayerName(name)
    , mInterSize(interSize)
    , mHiddenSize(hiddenSize)
    , mActivation(activation)
    , mQuantizedOutput(quantizedOutput)
{
    validateAttributes();
}

Nvfp4DualGemmPlugin::Nvfp4DualGemmPlugin(std::string const& name, PluginFieldCollection const* fc)
    : mLayerName(name)
{
    if (fc == nullptr || fc->fields == nullptr || fc->nbFields <= 0)
    {
        throw std::invalid_argument("Nvfp4DualGemmPlugin: plugin field collection must not be empty");
    }
    std::array<bool, kNbFields> fieldsSeen{};
    auto readIntField = [&fieldsSeen](PluginField const& field, int32_t index) {
        if (field.data == nullptr || field.type != PluginFieldType::kINT32 || field.length != 1)
        {
            throw std::invalid_argument("Nvfp4DualGemmPlugin: attributes must be scalar INT32 fields");
        }
        if (fieldsSeen.at(index))
        {
            throw std::invalid_argument("Nvfp4DualGemmPlugin: duplicate plugin attribute");
        }
        fieldsSeen.at(index) = true;
        return *static_cast<int32_t const*>(field.data);
    };
    for (int32_t i = 0; i < fc->nbFields; ++i)
    {
        PluginField const& field = fc->fields[i];
        if (field.name == nullptr)
        {
            throw std::invalid_argument("Nvfp4DualGemmPlugin: plugin attribute name must not be null");
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
        else if (fieldName == "quantized_output")
        {
            mQuantizedOutput = readIntField(field, kFieldQuantizedOutput);
        }
        else
        {
            throw std::invalid_argument("Nvfp4DualGemmPlugin: unknown plugin attribute " + fieldName);
        }
    }
    if (!fieldsSeen.at(kFieldInterSize) || !fieldsSeen.at(kFieldHiddenSize) || !fieldsSeen.at(kFieldActivation))
    {
        throw std::invalid_argument(
            "Nvfp4DualGemmPlugin: inter_size, hidden_size, and activation attributes are required");
    }
    validateAttributes();
}

void Nvfp4DualGemmPlugin::validateAttributes() const
{
    if (mQuantizedOutput != 0 && mQuantizedOutput != 1)
    {
        throw std::invalid_argument("Nvfp4DualGemmPlugin: quantized_output must be 0 or 1");
    }
    if (mActivation != kActivationSilu)
    {
        throw std::invalid_argument("Nvfp4DualGemmPlugin: activation must be SiLU (0)");
    }
    if (mInterSize <= 0 || mHiddenSize <= 0 || mInterSize % 128 != 0 || mHiddenSize % 64 != 0
        || mInterSize > std::numeric_limits<int32_t>::max() / 2)
    {
        throw std::invalid_argument(
            "Nvfp4DualGemmPlugin: inter_size must be a positive multiple of 128 and hidden_size a positive multiple of "
            "64");
    }
}

IPluginCapability* Nvfp4DualGemmPlugin::getCapabilityInterface(PluginCapabilityType type) noexcept
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

IPluginV3* Nvfp4DualGemmPlugin::clone() noexcept
{
    try
    {
        auto plugin
            = std::make_unique<Nvfp4DualGemmPlugin>(mLayerName, mInterSize, mHiddenSize, mActivation, mQuantizedOutput);
        plugin->setPluginNamespace(mNamespace.c_str());
        plugin->mTactic = mTactic;
        plugin->mMaxActiveClusters = mMaxActiveClusters;
        return plugin.release();
    }
    catch (std::exception const& error)
    {
        LOG_ERROR("Failed to copy plugin state: %s", error.what());
        return nullptr;
    }
}

char const* Nvfp4DualGemmPlugin::getPluginName() const noexcept
{
    return kPluginName;
}

char const* Nvfp4DualGemmPlugin::getPluginVersion() const noexcept
{
    return kPluginVersion;
}

char const* Nvfp4DualGemmPlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

void Nvfp4DualGemmPlugin::setPluginNamespace(char const* pluginNamespace) noexcept
{
    mNamespace = pluginNamespace == nullptr ? "" : pluginNamespace;
}

int32_t Nvfp4DualGemmPlugin::getNbOutputs() const noexcept
{
    return mQuantizedOutput ? 2 : 1;
}

int32_t Nvfp4DualGemmPlugin::inputCount() const noexcept
{
    return mQuantizedOutput ? 6 : 5;
}

int32_t Nvfp4DualGemmPlugin::getOutputDataTypes(
    DataType* outputTypes, int32_t nbOutputs, DataType const* inputTypes, int32_t nbInputs) const noexcept
{
    if (outputTypes == nullptr || inputTypes == nullptr || nbOutputs != getNbOutputs() || nbInputs != inputCount()
        || inputTypes[kInActivation] != DataType::kINT8)
    {
        return -1;
    }
    outputTypes[0] = mQuantizedOutput ? DataType::kINT8 : DataType::kHALF;
    if (mQuantizedOutput)
    {
        outputTypes[1] = DataType::kINT8;
    }
    return 0;
}

int32_t Nvfp4DualGemmPlugin::getOutputShapes(DimsExprs const* inputs, int32_t nbInputs, DimsExprs const* shapeInputs,
    int32_t nbShapeInputs, DimsExprs* outputs, int32_t nbOutputs, IExprBuilder& exprBuilder) noexcept
{
    (void) shapeInputs;
    (void) nbShapeInputs;
    if (inputs == nullptr || outputs == nullptr || nbInputs != inputCount() || nbOutputs != getNbOutputs())
    {
        return -1;
    }
    int32_t const rank = inputs[kInActivation].nbDims;
    if (!isSupportedActivationRank(rank))
    {
        return -1;
    }
    outputs[0] = inputs[kInActivation];
    outputs[0].d[rank - 1] = exprBuilder.constant(mQuantizedOutput ? mInterSize / 2 : mInterSize);
    if (mQuantizedOutput)
    {
        auto const* tokens = inputs[kInActivation].d[0];
        for (int32_t i = 1; i + 1 < rank; ++i)
        {
            tokens = exprBuilder.operation(DimensionOperation::kPROD, *tokens, *inputs[kInActivation].d[i]);
        }
        auto const* rows = exprBuilder.operation(DimensionOperation::kCEIL_DIV, *tokens, *exprBuilder.constant(128));
        outputs[1].nbDims = 1;
        outputs[1].d[0] = exprBuilder.operation(
            DimensionOperation::kPROD, *rows, *exprBuilder.constant(static_cast<int64_t>(128) * mInterSize / 16));
    }
    return 0;
}

bool Nvfp4DualGemmPlugin::validateTensorDesc(int32_t pos, PluginTensorDesc const& desc) const noexcept
{
    if (pos >= inputCount())
    {
        if (desc.format != TensorFormat::kLINEAR)
        {
            return false;
        }
        if (pos == inputCount())
        {
            return desc.type == (mQuantizedOutput ? DataType::kINT8 : DataType::kHALF)
                && isSupportedActivationRank(desc.dims.nbDims)
                && desc.dims.d[desc.dims.nbDims - 1] == (mQuantizedOutput ? mInterSize / 2 : mInterSize);
        }
        return mQuantizedOutput && pos == inputCount() + 1 && desc.type == DataType::kINT8 && desc.dims.nbDims == 1
            && (desc.dims.d[0] == -1 || desc.dims.d[0] > 0);
    }
    auto const type = (pos == kInAlpha || pos == kInGlobalScale) ? DataType::kFLOAT : DataType::kINT8;
    if (desc.format != TensorFormat::kLINEAR || desc.type != type)
    {
        return false;
    }
    switch (pos)
    {
    case kInActivation:
        return isSupportedActivationRank(desc.dims.nbDims) && desc.dims.d[desc.dims.nbDims - 1] == mHiddenSize / 2;
    case kInPackedWeight:
        return desc.dims.nbDims == 2 && desc.dims.d[0] == 2 * mInterSize && desc.dims.d[1] == mHiddenSize / 2;
    case kInActivationScale: return desc.dims.nbDims == 1 && (desc.dims.d[0] == -1 || desc.dims.d[0] > 0);
    case kInWeightScale:
        return desc.dims.nbDims == 1 && desc.dims.d[0] == static_cast<int64_t>(2) * mInterSize * (mHiddenSize / 16);
    case kInAlpha: return desc.dims.nbDims == 1 && desc.dims.d[0] == 2;
    case kInGlobalScale: return mQuantizedOutput && desc.dims.nbDims == 1 && desc.dims.d[0] == 1;
    default: return false;
    }
}

bool Nvfp4DualGemmPlugin::supportsFormatCombination(
    int32_t pos, DynamicPluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
    return inOut != nullptr && nbInputs == inputCount() && nbOutputs == getNbOutputs() && pos >= 0
        && pos < inputCount() + getNbOutputs() && validateTensorDesc(pos, inOut[pos].desc);
}

int32_t Nvfp4DualGemmPlugin::configurePlugin(
    DynamicPluginTensorDesc const* in, int32_t nbInputs, DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept
{
    try
    {
        if (in == nullptr || out == nullptr || nbInputs != inputCount() || nbOutputs != getNbOutputs())
        {
            return -1;
        }
        for (int32_t pos = 0; pos < inputCount(); ++pos)
        {
            if (!validateTensorDesc(pos, in[pos].desc))
            {
                LOG_ERROR("Nvfp4DualGemmPlugin: invalid input descriptor at position %d", pos);
                return -1;
            }
        }
        for (int32_t pos = 0; pos < getNbOutputs(); ++pos)
        {
            if (!validateTensorDesc(inputCount() + pos, out[pos].desc))
            {
                return -1;
            }
        }
        int64_t const maxM = getTokenCount(in[kInActivation].max);
        if (maxM <= 0 || maxM > std::numeric_limits<int32_t>::max())
        {
            return -1;
        }
        return 0;
    }
    catch (std::exception const& error)
    {
        LOG_ERROR("Plugin configuration failed: %s", error.what());
        return -1;
    }
}

int32_t Nvfp4DualGemmPlugin::getNbTactics() noexcept
{
    return mQuantizedOutput ? kernels::CuteDslNvfp4DualGemmRunner::kNUM_QUANTIZED_TACTICS : 0;
}

int32_t Nvfp4DualGemmPlugin::getValidTactics(int32_t* tactics, int32_t nbTactics) noexcept
{
    int32_t const count = getNbTactics();
    if (nbTactics < count || (count > 0 && tactics == nullptr))
    {
        return -1;
    }
    for (int32_t i = 0; i < count; ++i)
    {
        tactics[i] = i + kernels::CuteDslNvfp4DualGemmRunner::kFIXED_2X1_TACTIC;
    }
    return 0;
}

int32_t Nvfp4DualGemmPlugin::setTactic(int32_t tactic) noexcept
{
    if (tactic < kernels::CuteDslNvfp4DualGemmRunner::kDEFAULT_TACTIC || tactic > getNbTactics())
    {
        return -1;
    }
    int32_t clusters{};
    auto const error = mRunner.prepare(clusters, mQuantizedOutput != 0, tactic);
    if (error != cudaSuccess)
    {
        LOG_ERROR("Nvfp4DualGemmPlugin: tactic %d initialization failed: %s", tactic, cudaGetErrorString(error));
        return -1;
    }
    mTactic = tactic;
    mMaxActiveClusters = clusters;
    LOG_DEBUG("Nvfp4DualGemmPlugin: selected tactic %d, persistent clusters %d", tactic, clusters);
    return 0;
}

char const* Nvfp4DualGemmPlugin::getTimingCacheID() noexcept
{
    // TRT prefixes the I/O shapes and formats. Version the remaining kernel contract;
    // the selected tactic is deliberately excluded from this build-time identity.
    return mQuantizedOutput ? "nvfp4-dual-v2-quantized" : "nvfp4-dual-v2-fp16";
}

size_t Nvfp4DualGemmPlugin::getWorkspaceSize(DynamicPluginTensorDesc const* inputs, int32_t nbInputs,
    DynamicPluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept
{
    (void) inputs;
    (void) nbInputs;
    (void) outputs;
    (void) nbOutputs;
    return 0;
}

int32_t Nvfp4DualGemmPlugin::enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc,
    void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    (void) workspace;
    if (inputDesc == nullptr || outputDesc == nullptr || inputs == nullptr || outputs == nullptr
        || inputs[kInActivation] == nullptr || inputs[kInPackedWeight] == nullptr || outputs[0] == nullptr)
    {
        return -1;
    }
    for (int32_t pos = 0; pos < inputCount(); ++pos)
    {
        if (!validateTensorDesc(pos, inputDesc[pos]))
        {
            return -1;
        }
    }
    if (mQuantizedOutput && (inputs[kInGlobalScale] == nullptr || outputs[1] == nullptr))
    {
        return -1;
    }
    Dims const& activationDims = inputDesc[kInActivation].dims;
    int64_t const m = getTokenCount(activationDims);
    if (!validateOutputs(outputDesc, activationDims) || m <= 0 || m > std::numeric_limits<int32_t>::max())
    {
        LOG_ERROR("Nvfp4DualGemmPlugin: invalid runtime token count or output shape");
        return -1;
    }
    int64_t const sfBytes = ((m + 127) / 128) * 128 * (mHiddenSize / 16);
    if (inputDesc[kInActivationScale].dims.d[0] != sfBytes)
    {
        return -1;
    }
    kernels::Nvfp4DualGemmParams const params{inputs[kInActivation], inputs[kInPackedWeight],
        inputs[kInActivationScale], inputs[kInWeightScale], inputs[kInAlpha], outputs[0], static_cast<int32_t>(m),
        mInterSize, mHiddenSize, mMaxActiveClusters, mQuantizedOutput ? outputs[1] : nullptr,
        mQuantizedOutput ? static_cast<float const*>(inputs[kInGlobalScale]) : nullptr, mTactic};
    cudaError_t const error = mRunner.run(params, stream);
    if (error != cudaSuccess)
    {
        LOG_ERROR("Nvfp4DualGemmPlugin: dual GEMM launch failed for M=%lld: %s", static_cast<long long>(m),
            cudaGetErrorString(error));
        return -1;
    }
    return 0;
}

bool Nvfp4DualGemmPlugin::validateOutputs(PluginTensorDesc const* outputs, Dims const& activation) const noexcept
{
    if (!validateTensorDesc(inputCount(), outputs[0]) || !sameTokenShape(outputs[0].dims, activation))
    {
        return false;
    }
    int64_t const tokens = getTokenCount(activation);
    return !mQuantizedOutput
        || (validateTensorDesc(inputCount() + 1, outputs[1])
            && outputs[1].dims.d[0] == ((tokens + 127) / 128) * 128 * (mInterSize / 16));
}

int32_t Nvfp4DualGemmPlugin::onShapeChange(
    PluginTensorDesc const* in, int32_t nbInputs, PluginTensorDesc const* out, int32_t nbOutputs) noexcept
{
    if (in == nullptr || out == nullptr || nbInputs != inputCount() || nbOutputs != getNbOutputs())
    {
        return -1;
    }
    for (int32_t pos = 0; pos < inputCount(); ++pos)
    {
        if (!validateTensorDesc(pos, in[pos]))
        {
            return -1;
        }
    }
    int64_t const m = getTokenCount(in[kInActivation].dims);
    if (!validateOutputs(out, in[kInActivation].dims) || m <= 0 || m > std::numeric_limits<int32_t>::max())
    {
        return -1;
    }
    int64_t const sfBytes = ((m + 127) / 128) * 128 * (mHiddenSize / 16);
    if (in[kInActivationScale].dims.d[0] != sfBytes)
    {
        return -1;
    }
    // Build-time profiling also executes plugins without attachToContext().
    cudaError_t const error = mRunner.prepare(mMaxActiveClusters, mQuantizedOutput != 0, mTactic);
    if (error != cudaSuccess)
    {
        LOG_ERROR("Nvfp4DualGemmPlugin: loading the dual GEMM module for M=%lld failed: %s", static_cast<long long>(m),
            cudaGetErrorString(error));
        return -1;
    }
    return 0;
}

IPluginV3* Nvfp4DualGemmPlugin::attachToContext(IPluginResourceContext* context) noexcept
{
    (void) context;
    return clone();
}

PluginFieldCollection const* Nvfp4DualGemmPlugin::getFieldsToSerialize() noexcept
{
    mDataToSerialize.clear();
    mDataToSerialize.emplace_back("inter_size", &mInterSize, PluginFieldType::kINT32, 1);
    mDataToSerialize.emplace_back("hidden_size", &mHiddenSize, PluginFieldType::kINT32, 1);
    mDataToSerialize.emplace_back("activation", &mActivation, PluginFieldType::kINT32, 1);
    mDataToSerialize.emplace_back("quantized_output", &mQuantizedOutput, PluginFieldType::kINT32, 1);
    mFCToSerialize.nbFields = static_cast<int32_t>(mDataToSerialize.size());
    mFCToSerialize.fields = mDataToSerialize.data();
    return &mFCToSerialize;
}

PluginFieldCollection Nvfp4DualGemmPluginCreator::mFieldCollection{};
std::vector<PluginField> Nvfp4DualGemmPluginCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(Nvfp4DualGemmPluginCreator);

Nvfp4DualGemmPluginCreator::Nvfp4DualGemmPluginCreator()
{
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back("inter_size", nullptr, PluginFieldType::kINT32, 1);
    mPluginAttributes.emplace_back("hidden_size", nullptr, PluginFieldType::kINT32, 1);
    mPluginAttributes.emplace_back("activation", nullptr, PluginFieldType::kINT32, 1);
    mPluginAttributes.emplace_back("quantized_output", nullptr, PluginFieldType::kINT32, 1);
    mFieldCollection.nbFields = static_cast<int32_t>(mPluginAttributes.size());
    mFieldCollection.fields = mPluginAttributes.data();
}

char const* Nvfp4DualGemmPluginCreator::getPluginName() const noexcept
{
    return kPluginName;
}

char const* Nvfp4DualGemmPluginCreator::getPluginVersion() const noexcept
{
    return kPluginVersion;
}

PluginFieldCollection const* Nvfp4DualGemmPluginCreator::getFieldNames() noexcept
{
    return &mFieldCollection;
}

char const* Nvfp4DualGemmPluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

void Nvfp4DualGemmPluginCreator::setPluginNamespace(char const* pluginNamespace) noexcept
{
    mNamespace = pluginNamespace == nullptr ? "" : pluginNamespace;
}

IPluginV3* Nvfp4DualGemmPluginCreator::createPlugin(
    char const* name, PluginFieldCollection const* fc, TensorRTPhase phase) noexcept
{
    (void) phase;
    try
    {
        auto plugin = std::make_unique<Nvfp4DualGemmPlugin>(name == nullptr ? kPluginName : name, fc);
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin.release();
    }
    catch (std::exception const& error)
    {
        LOG_ERROR("Failed to create Nvfp4DualGemmPlugin: %s", error.what());
        return nullptr;
    }
}

} // namespace plugins
} // namespace trt_edgellm
