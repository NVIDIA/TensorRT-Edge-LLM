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

#include "rmsNormPlugin.h"

#include "common/cudaUtils.h"
#include "common/logger.h"
#include "plugins/utils/pluginUtils.h"

#if defined(CUTE_DSL_RMSNORM_ENABLED)
#include "kernels/rmsNorm/cuteDslRmsNormRunner.h"
#endif

#include <cmath>
#include <cstdint>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

using namespace nvinfer1;

namespace trt_edgellm
{
namespace plugins
{
namespace
{

constexpr int32_t kIN_X{0};
constexpr int32_t kIN_GAMMA{1};
constexpr int32_t kOUT_Y{2};
constexpr int32_t kNB_INPUTS{2};
constexpr int32_t kNB_OUTPUTS{1};
constexpr int32_t kMIN_RANK{2};
constexpr int32_t kMAX_RANK{4};

constexpr char const* kPLUGIN_NAME{"RmsNormPlugin"};
constexpr char const* kPLUGIN_VERSION{"1"};
constexpr char const* kFIELD_EPS{"rms_norm_eps"};
constexpr char const* kFIELD_WEIGHT_BEFORE_CAST{"weight_before_cast"};

bool isActivationDataType(DataType dataType) noexcept
{
    return dataType == DataType::kHALF || dataType == DataType::kBF16;
}

bool isSupportedHiddenSize(int32_t hiddenSize) noexcept
{
    return hiddenSize == 4096 || hiddenSize == 5120 || hiddenSize == 7168 || hiddenSize == 8192;
}

bool isSupportedOrDynamicHiddenSize(int32_t hiddenSize) noexcept
{
    return hiddenSize == -1 || isSupportedHiddenSize(hiddenSize);
}

bool dimsEqual(Dims const& lhs, Dims const& rhs) noexcept
{
    if (lhs.nbDims != rhs.nbDims)
    {
        return false;
    }
    for (int32_t index = 0; index < lhs.nbDims; ++index)
    {
        if (lhs.d[index] != rhs.d[index])
        {
            return false;
        }
    }
    return true;
}

bool flattenRows(Dims const& xDims, bool allowZero, int32_t& rows) noexcept
{
    if (xDims.nbDims < kMIN_RANK || xDims.nbDims > kMAX_RANK)
    {
        return false;
    }

    int64_t flattened{1};
    for (int32_t index = 0; index < xDims.nbDims - 1; ++index)
    {
        int32_t const dimension = xDims.d[index];
        if (dimension < 0 || (!allowZero && dimension == 0))
        {
            return false;
        }
        if (dimension != 0 && flattened > std::numeric_limits<int32_t>::max() / dimension)
        {
            return false;
        }
        flattened *= dimension;
    }
    rows = static_cast<int32_t>(flattened);
    return true;
}

bool validateConcreteShapes(
    Dims const& xDims, Dims const& gammaDims, Dims const* outputDims, bool allowZero, char const* point) noexcept
{
    int32_t rows{};
    if (!flattenRows(xDims, allowZero, rows))
    {
        LOG_ERROR("RmsNormPlugin: %s x rank must be 2-4 and its flattened row count must fit INT32", point);
        return false;
    }
    int32_t const hiddenSize = xDims.d[xDims.nbDims - 1];
    if (!isSupportedHiddenSize(hiddenSize))
    {
        LOG_ERROR("RmsNormPlugin: %s hidden size must be one of {4096, 5120, 7168, 8192}", point);
        return false;
    }
    if (gammaDims.nbDims != 1 || gammaDims.d[0] != hiddenSize)
    {
        LOG_ERROR("RmsNormPlugin: %s gamma must have shape [H] matching x's last dimension", point);
        return false;
    }
    if (outputDims != nullptr && !dimsEqual(xDims, *outputDims))
    {
        LOG_ERROR("RmsNormPlugin: %s output shape must equal x shape", point);
        return false;
    }
    return true;
}

bool hasCoherentDescriptors(
    PluginTensorDesc const& x, PluginTensorDesc const& gamma, PluginTensorDesc const& output) noexcept
{
    return isActivationDataType(x.type) && gamma.type == x.type && output.type == x.type
        && x.format == TensorFormat::kLINEAR && gamma.format == TensorFormat::kLINEAR
        && output.format == TensorFormat::kLINEAR;
}

bool hasFixedProfileHiddenSize(DynamicPluginTensorDesc const& x, DynamicPluginTensorDesc const& gamma) noexcept
{
    int32_t const hiddenSize = x.min.d[x.min.nbDims - 1];
    return x.opt.d[x.opt.nbDims - 1] == hiddenSize && x.max.d[x.max.nbDims - 1] == hiddenSize
        && gamma.min.d[0] == hiddenSize && gamma.opt.d[0] == hiddenSize && gamma.max.d[0] == hiddenSize;
}

float getRequiredFloatField(PluginFieldCollection const* fields, char const* name)
{
    if (fields == nullptr)
    {
        throw std::invalid_argument("RmsNormPlugin: null PluginFieldCollection");
    }

    bool found{false};
    float value{};
    for (int32_t index = 0; index < fields->nbFields; ++index)
    {
        PluginField const& field = fields->fields[index];
        if (field.name != nullptr && std::string(field.name) == name)
        {
            if (found)
            {
                throw std::invalid_argument(std::string("RmsNormPlugin: duplicate field ") + name);
            }
            if (field.type != PluginFieldType::kFLOAT32 || field.length != 1 || field.data == nullptr)
            {
                throw std::invalid_argument(
                    std::string("RmsNormPlugin: field ") + name + " must be one non-null FLOAT32 scalar");
            }
            value = *static_cast<float const*>(field.data);
            found = true;
        }
    }
    if (!found)
    {
        throw std::invalid_argument(std::string("RmsNormPlugin: missing required field ") + name);
    }
    return value;
}

int32_t getRequiredIntField(PluginFieldCollection const* fields, char const* name)
{
    if (fields == nullptr)
    {
        throw std::invalid_argument("RmsNormPlugin: null PluginFieldCollection");
    }

    bool found{false};
    int32_t value{};
    for (int32_t index = 0; index < fields->nbFields; ++index)
    {
        PluginField const& field = fields->fields[index];
        if (field.name != nullptr && std::string(field.name) == name)
        {
            if (found)
            {
                throw std::invalid_argument(std::string("RmsNormPlugin: duplicate field ") + name);
            }
            if (field.type != PluginFieldType::kINT32 || field.length != 1 || field.data == nullptr)
            {
                throw std::invalid_argument(
                    std::string("RmsNormPlugin: field ") + name + " must be one non-null INT32 scalar");
            }
            value = *static_cast<int32_t const*>(field.data);
            found = true;
        }
    }
    if (!found)
    {
        throw std::invalid_argument(std::string("RmsNormPlugin: missing required field ") + name);
    }
    return value;
}

void validateAttributes(float rmsNormEps, int32_t weightBeforeCast)
{
    if (!std::isfinite(rmsNormEps) || rmsNormEps <= 0.0F)
    {
        throw std::invalid_argument("RmsNormPlugin: rms_norm_eps must be finite and positive");
    }
    if (weightBeforeCast != 0 && weightBeforeCast != 1)
    {
        throw std::invalid_argument("RmsNormPlugin: weight_before_cast must be 0 or 1");
    }
}

} // namespace

PluginFieldCollection RmsNormPluginCreator::mFieldCollection{};
std::vector<PluginField> RmsNormPluginCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(RmsNormPluginCreator);

RmsNormPlugin::RmsNormPlugin(std::string const& name, float rmsNormEps, int32_t weightBeforeCast)
    : mLayerName(name)
    , mRmsNormEps(rmsNormEps)
    , mWeightBeforeCast(weightBeforeCast)
{
    validateAttributes(mRmsNormEps, mWeightBeforeCast);
}

RmsNormPlugin::RmsNormPlugin(std::string const& name, PluginFieldCollection const* fields)
    : RmsNormPlugin(
          name, getRequiredFloatField(fields, kFIELD_EPS), getRequiredIntField(fields, kFIELD_WEIGHT_BEFORE_CAST))
{
}

IPluginCapability* RmsNormPlugin::getCapabilityInterface(PluginCapabilityType type) noexcept
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

IPluginV3* RmsNormPlugin::clone() noexcept
{
    try
    {
        auto* plugin = new RmsNormPlugin(mLayerName, mRmsNormEps, mWeightBeforeCast);
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }
    catch (std::exception const& error)
    {
        LOG_ERROR("RmsNormPlugin clone failed: %s", error.what());
        return nullptr;
    }
}

char const* RmsNormPlugin::getPluginName() const noexcept
{
    return kPLUGIN_NAME;
}

char const* RmsNormPlugin::getPluginVersion() const noexcept
{
    return kPLUGIN_VERSION;
}

char const* RmsNormPlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

void RmsNormPlugin::setPluginNamespace(char const* pluginNamespace) noexcept
{
    try
    {
        mNamespace = pluginNamespace == nullptr ? "" : pluginNamespace;
    }
    catch (std::exception const& error)
    {
        LOG_ERROR("RmsNormPlugin namespace update failed: %s", error.what());
    }
}

int32_t RmsNormPlugin::getNbOutputs() const noexcept
{
    return kNB_OUTPUTS;
}

int32_t RmsNormPlugin::getOutputDataTypes(
    DataType* outputTypes, int32_t nbOutputs, DataType const* inputTypes, int32_t nbInputs) const noexcept
{
    if (outputTypes == nullptr || inputTypes == nullptr || nbInputs != kNB_INPUTS || nbOutputs != kNB_OUTPUTS
        || !isActivationDataType(inputTypes[kIN_X]) || inputTypes[kIN_GAMMA] != inputTypes[kIN_X])
    {
        LOG_ERROR("RmsNormPlugin: getOutputDataTypes expects homogeneous FP16 or BF16 x and gamma inputs");
        return -1;
    }
    outputTypes[0] = inputTypes[kIN_X];
    return 0;
}

int32_t RmsNormPlugin::getOutputShapes(DimsExprs const* inputs, int32_t nbInputs, DimsExprs const* shapeInputs,
    int32_t nbShapeInputs, DimsExprs* outputs, int32_t nbOutputs, IExprBuilder& exprBuilder) noexcept
{
    (void) shapeInputs;
    (void) nbShapeInputs;
    (void) exprBuilder;
    if (inputs == nullptr || outputs == nullptr || nbInputs != kNB_INPUTS || nbOutputs != kNB_OUTPUTS
        || inputs[kIN_X].nbDims < kMIN_RANK || inputs[kIN_X].nbDims > kMAX_RANK)
    {
        LOG_ERROR("RmsNormPlugin: getOutputShapes expects two inputs, one output, and x rank 2-4");
        return -1;
    }
    outputs[0] = inputs[kIN_X];
    return 0;
}

bool RmsNormPlugin::supportsFormatCombination(
    int32_t pos, DynamicPluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
    if (inOut == nullptr || nbInputs != kNB_INPUTS || nbOutputs != kNB_OUTPUTS || pos < 0
        || pos >= nbInputs + nbOutputs)
    {
        return false;
    }

    PluginTensorDesc const& tensor = inOut[pos].desc;
    if (tensor.format != TensorFormat::kLINEAR)
    {
        return false;
    }

    switch (pos)
    {
    case kIN_X:
        return isActivationDataType(tensor.type) && tensor.dims.nbDims >= kMIN_RANK && tensor.dims.nbDims <= kMAX_RANK
            && isSupportedOrDynamicHiddenSize(tensor.dims.d[tensor.dims.nbDims - 1]);
    case kIN_GAMMA:
    {
        Dims const& xDims = inOut[kIN_X].desc.dims;
        int32_t const gammaHiddenSize = tensor.dims.nbDims == 1 ? tensor.dims.d[0] : 0;
        int32_t const xHiddenSize = xDims.nbDims >= kMIN_RANK ? xDims.d[xDims.nbDims - 1] : 0;
        return tensor.type == inOut[kIN_X].desc.type && tensor.dims.nbDims == 1
            && isSupportedOrDynamicHiddenSize(gammaHiddenSize)
            && (gammaHiddenSize == -1 || xHiddenSize == -1 || gammaHiddenSize == xHiddenSize);
    }
    case kOUT_Y:
    {
        Dims const& xDims = inOut[kIN_X].desc.dims;
        return tensor.type == inOut[kIN_X].desc.type && tensor.dims.nbDims == xDims.nbDims;
    }
    default: return false;
    }
}

int32_t RmsNormPlugin::configurePlugin(DynamicPluginTensorDesc const* inputs, int32_t nbInputs,
    DynamicPluginTensorDesc const* outputs, int32_t nbOutputs) noexcept
{
    try
    {
        if (inputs == nullptr || outputs == nullptr || nbInputs != kNB_INPUTS || nbOutputs != kNB_OUTPUTS)
        {
            LOG_ERROR("RmsNormPlugin: configurePlugin expected two inputs and one output");
            return -1;
        }
        if (!hasCoherentDescriptors(inputs[kIN_X].desc, inputs[kIN_GAMMA].desc, outputs[0].desc))
        {
            LOG_ERROR("RmsNormPlugin: x, gamma, and output must have one homogeneous FP16/BF16 linear format");
            return -1;
        }
        if (!validateConcreteShapes(inputs[kIN_X].min, inputs[kIN_GAMMA].min, nullptr, false, "minimum profile")
            || !validateConcreteShapes(inputs[kIN_X].opt, inputs[kIN_GAMMA].opt, nullptr, false, "optimum profile")
            || !validateConcreteShapes(inputs[kIN_X].max, inputs[kIN_GAMMA].max, nullptr, false, "maximum profile"))
        {
            return -1;
        }
        if (!hasFixedProfileHiddenSize(inputs[kIN_X], inputs[kIN_GAMMA]))
        {
            LOG_ERROR("RmsNormPlugin: optimization profile hidden size H must be fixed across x and gamma");
            return -1;
        }

#if defined(CUTE_DSL_RMSNORM_ENABLED)
        int32_t maxRows{};
        static_cast<void>(flattenRows(inputs[kIN_X].max, false, maxRows));
        int32_t const hiddenSize = inputs[kIN_X].max.d[inputs[kIN_X].max.nbDims - 1];
        int32_t const smVersion = getSMVersion();
        if (!CuteDslRmsNormRunner::canImplement(maxRows, hiddenSize, smVersion, inputs[kIN_X].desc.type))
        {
            LOG_ERROR("RmsNormPlugin: no RMSNorm CuTe DSL variant supports rows=%d H=%d dtype=%d on SM%d", maxRows,
                hiddenSize, static_cast<int32_t>(inputs[kIN_X].desc.type), smVersion);
            return -1;
        }
#else
        LOG_ERROR(
            "RmsNormPlugin: rmsnorm CuTe DSL artifacts are not linked. Generate them with "
            "kernelSrcs/build_cutedsl.py --kernels rmsnorm --gpu_arch <sm_NN> --arch <arch>, then rebuild "
            "with -DENABLE_CUTE_DSL=rmsnorm -DCUTE_DSL_ARTIFACT_TAG=<sm_NN>.");
        return -1;
#endif
        return 0;
    }
    catch (std::exception const& error)
    {
        LOG_ERROR("RmsNormPlugin configurePlugin failed: %s", error.what());
    }
    catch (...)
    {
        LOG_ERROR("RmsNormPlugin configurePlugin failed: unknown error");
    }
    return -1;
}

size_t RmsNormPlugin::getWorkspaceSize(DynamicPluginTensorDesc const* inputs, int32_t nbInputs,
    DynamicPluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept
{
    if (inputs == nullptr || outputs == nullptr || nbInputs != kNB_INPUTS || nbOutputs != kNB_OUTPUTS)
    {
        LOG_ERROR("RmsNormPlugin: getWorkspaceSize expected two inputs and one output");
    }
    return 0;
}

int32_t RmsNormPlugin::enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc,
    void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    (void) workspace;
#if !defined(CUTE_DSL_RMSNORM_ENABLED)
    (void) inputDesc;
    (void) outputDesc;
    (void) inputs;
    (void) outputs;
    (void) stream;
    LOG_ERROR("RmsNormPlugin: rmsnorm CuTe DSL artifacts are not linked; rebuild with -DENABLE_CUTE_DSL=rmsnorm");
    return -1;
#else
    try
    {
        if (inputDesc == nullptr || outputDesc == nullptr || inputs == nullptr || outputs == nullptr
            || inputs[kIN_X] == nullptr || inputs[kIN_GAMMA] == nullptr || outputs[0] == nullptr)
        {
            LOG_ERROR("RmsNormPlugin: null descriptor, input, or output at enqueue");
            return -1;
        }
        if (!hasCoherentDescriptors(inputDesc[kIN_X], inputDesc[kIN_GAMMA], outputDesc[0])
            || !validateConcreteShapes(
                inputDesc[kIN_X].dims, inputDesc[kIN_GAMMA].dims, &outputDesc[0].dims, true, "runtime"))
        {
            return -1;
        }

        int32_t rows{};
        static_cast<void>(flattenRows(inputDesc[kIN_X].dims, true, rows));
        if (rows == 0)
        {
            return 0;
        }
        int32_t const hiddenSize = inputDesc[kIN_X].dims.d[inputDesc[kIN_X].dims.nbDims - 1];

        CuteDslRmsNormParams parameters{};
        parameters.input = inputs[kIN_X];
        parameters.gamma = inputs[kIN_GAMMA];
        parameters.output = outputs[0];
        parameters.rows = rows;
        parameters.hiddenSize = hiddenSize;
        parameters.rmsNormEps = mRmsNormEps;
        parameters.weightBeforeCast = mWeightBeforeCast;
        parameters.dataType = inputDesc[kIN_X].type;
        return CuteDslRmsNormRunner::run(parameters, stream);
    }
    catch (std::exception const& error)
    {
        LOG_ERROR("RmsNormPlugin enqueue failed: %s", error.what());
    }
    catch (...)
    {
        LOG_ERROR("RmsNormPlugin enqueue failed: unknown error");
    }
    return -1;
#endif
}

int32_t RmsNormPlugin::onShapeChange(
    PluginTensorDesc const* inputs, int32_t nbInputs, PluginTensorDesc const* outputs, int32_t nbOutputs) noexcept
{
    try
    {
        if (inputs == nullptr || outputs == nullptr || nbInputs != kNB_INPUTS || nbOutputs != kNB_OUTPUTS)
        {
            LOG_ERROR("RmsNormPlugin: onShapeChange expected two inputs and one output");
            return -1;
        }
        if (!hasCoherentDescriptors(inputs[kIN_X], inputs[kIN_GAMMA], outputs[0])
            || !validateConcreteShapes(inputs[kIN_X].dims, inputs[kIN_GAMMA].dims, &outputs[0].dims, true, "runtime"))
        {
            return -1;
        }
#if defined(CUTE_DSL_RMSNORM_ENABLED)
        int32_t rows{};
        static_cast<void>(flattenRows(inputs[kIN_X].dims, true, rows));
        if (rows == 0)
        {
            return 0;
        }
        int32_t const hiddenSize = inputs[kIN_X].dims.d[inputs[kIN_X].dims.nbDims - 1];
        return CuteDslRmsNormRunner::canImplement(rows, hiddenSize, getSMVersion(), inputs[kIN_X].type) ? 0 : -1;
#else
        return -1;
#endif
    }
    catch (std::exception const& error)
    {
        LOG_ERROR("RmsNormPlugin onShapeChange failed: %s", error.what());
    }
    catch (...)
    {
        LOG_ERROR("RmsNormPlugin onShapeChange failed: unknown error");
    }
    return -1;
}

IPluginV3* RmsNormPlugin::attachToContext(IPluginResourceContext* context) noexcept
{
    (void) context;
    return clone();
}

PluginFieldCollection const* RmsNormPlugin::getFieldsToSerialize() noexcept
{
    try
    {
        mDataToSerialize.clear();
        mDataToSerialize.emplace_back(kFIELD_EPS, &mRmsNormEps, PluginFieldType::kFLOAT32, 1);
        mDataToSerialize.emplace_back(kFIELD_WEIGHT_BEFORE_CAST, &mWeightBeforeCast, PluginFieldType::kINT32, 1);
        mFieldsToSerialize.nbFields = static_cast<int32_t>(mDataToSerialize.size());
        mFieldsToSerialize.fields = mDataToSerialize.data();
        return &mFieldsToSerialize;
    }
    catch (std::exception const& error)
    {
        LOG_ERROR("RmsNormPlugin serialization failed: %s", error.what());
        return nullptr;
    }
}

RmsNormPluginCreator::RmsNormPluginCreator()
{
    static std::mutex sMutex;
    std::lock_guard<std::mutex> lock(sMutex);
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(kFIELD_EPS, nullptr, PluginFieldType::kFLOAT32, 1);
    mPluginAttributes.emplace_back(kFIELD_WEIGHT_BEFORE_CAST, nullptr, PluginFieldType::kINT32, 1);
    mFieldCollection.nbFields = static_cast<int32_t>(mPluginAttributes.size());
    mFieldCollection.fields = mPluginAttributes.data();
}

char const* RmsNormPluginCreator::getPluginName() const noexcept
{
    return kPLUGIN_NAME;
}

char const* RmsNormPluginCreator::getPluginVersion() const noexcept
{
    return kPLUGIN_VERSION;
}

PluginFieldCollection const* RmsNormPluginCreator::getFieldNames() noexcept
{
    return &mFieldCollection;
}

char const* RmsNormPluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

void RmsNormPluginCreator::setPluginNamespace(char const* pluginNamespace) noexcept
{
    try
    {
        mNamespace = pluginNamespace == nullptr ? "" : pluginNamespace;
    }
    catch (std::exception const& error)
    {
        LOG_ERROR("RmsNormPluginCreator namespace update failed: %s", error.what());
    }
}

IPluginV3* RmsNormPluginCreator::createPlugin(
    char const* name, PluginFieldCollection const* fields, TensorRTPhase phase) noexcept
{
    (void) phase;
    try
    {
        if (name == nullptr)
        {
            throw std::invalid_argument("RmsNormPlugin: null layer name");
        }
        auto* plugin = new RmsNormPlugin(name, fields);
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }
    catch (std::exception const& error)
    {
        LOG_ERROR("RmsNormPlugin creation failed: %s", error.what());
        return nullptr;
    }
}

} // namespace plugins
} // namespace trt_edgellm
