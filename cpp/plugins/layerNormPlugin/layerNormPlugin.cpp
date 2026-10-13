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

#include "layerNormPlugin.h"

#include "common/cudaUtils.h"
#include "common/logger.h"
#include "plugins/utils/pluginUtils.h"

#if defined(CUTE_DSL_LAYERNORM_ENABLED)
#include "kernels/layerNorm/cuteDslLayerNormRunner.h"
#endif

#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
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
constexpr int32_t kIN_BETA{2};
constexpr int32_t kOUT_Y{3};
constexpr int32_t kNB_INPUTS{3};
constexpr int32_t kNB_OUTPUTS{1};
constexpr int32_t kMIN_RANK{2};
constexpr int32_t kMAX_RANK{4};

constexpr char const* kPLUGIN_NAME{"LayerNormPlugin"};
constexpr char const* kPLUGIN_VERSION{"1"};
constexpr char const* kFIELD_EPSILON{"epsilon"};

bool isActivationDataType(DataType dataType) noexcept
{
    return dataType == DataType::kHALF || dataType == DataType::kBF16;
}

bool isSupportedHiddenSize(int32_t hiddenSize) noexcept
{
    return hiddenSize == 4096 || hiddenSize == 4097 || hiddenSize == 5120 || hiddenSize == 7168 || hiddenSize == 8192;
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

bool dimsKnownCompatible(Dims const& lhs, Dims const& rhs) noexcept
{
    if (lhs.nbDims != rhs.nbDims)
    {
        return false;
    }
    for (int32_t index = 0; index < lhs.nbDims; ++index)
    {
        if (lhs.d[index] != -1 && rhs.d[index] != -1 && lhs.d[index] != rhs.d[index])
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

bool validateSymbolicShapes(
    Dims const& xDims, Dims const& gammaDims, Dims const& betaDims, Dims const& outputDims) noexcept
{
    if (xDims.nbDims < kMIN_RANK || xDims.nbDims > kMAX_RANK)
    {
        LOG_ERROR("LayerNormPlugin: x rank must be 2-4");
        return false;
    }

    int32_t const hiddenSize = xDims.d[xDims.nbDims - 1];
    if (!isSupportedOrDynamicHiddenSize(hiddenSize))
    {
        LOG_ERROR(
            "LayerNormPlugin: x hidden size must be dynamic or one of "
            "{4096, 4097, 5120, 7168, 8192}");
        return false;
    }
    if (gammaDims.nbDims != 1 || !isSupportedOrDynamicHiddenSize(gammaDims.d[0])
        || (hiddenSize != -1 && gammaDims.d[0] != -1 && gammaDims.d[0] != hiddenSize))
    {
        LOG_ERROR("LayerNormPlugin: gamma must have shape [H] compatible with x's last dimension");
        return false;
    }
    if (betaDims.nbDims != 1 || !isSupportedOrDynamicHiddenSize(betaDims.d[0])
        || (hiddenSize != -1 && betaDims.d[0] != -1 && betaDims.d[0] != hiddenSize))
    {
        LOG_ERROR("LayerNormPlugin: beta must have shape [H] compatible with x's last dimension");
        return false;
    }
    if (!dimsKnownCompatible(xDims, outputDims))
    {
        LOG_ERROR("LayerNormPlugin: output shape must be compatible with x shape");
        return false;
    }
    return true;
}

bool validateConcreteShapes(Dims const& xDims, Dims const& gammaDims, Dims const& betaDims, Dims const* outputDims,
    bool allowZero, char const* point) noexcept
{
    int32_t rows{};
    if (!flattenRows(xDims, allowZero, rows))
    {
        LOG_ERROR("LayerNormPlugin: %s x rank must be 2-4 and its flattened row count must fit INT32%s", point,
            allowZero ? "" : " and be positive");
        return false;
    }
    int32_t const hiddenSize = xDims.d[xDims.nbDims - 1];
    if (!isSupportedHiddenSize(hiddenSize))
    {
        LOG_ERROR(
            "LayerNormPlugin: %s hidden size must be one of "
            "{4096, 4097, 5120, 7168, 8192}",
            point);
        return false;
    }
    if (gammaDims.nbDims != 1 || gammaDims.d[0] != hiddenSize)
    {
        LOG_ERROR("LayerNormPlugin: %s gamma must have shape [H] matching x's last dimension", point);
        return false;
    }
    if (betaDims.nbDims != 1 || betaDims.d[0] != hiddenSize)
    {
        LOG_ERROR("LayerNormPlugin: %s beta must have shape [H] matching x's last dimension", point);
        return false;
    }
    if (outputDims != nullptr && !dimsEqual(xDims, *outputDims))
    {
        LOG_ERROR("LayerNormPlugin: %s output shape must equal x shape", point);
        return false;
    }
    return true;
}

bool hasCoherentDescriptors(PluginTensorDesc const& x, PluginTensorDesc const& gamma, PluginTensorDesc const& beta,
    PluginTensorDesc const& output) noexcept
{
    return isActivationDataType(x.type) && gamma.type == x.type && beta.type == x.type && output.type == x.type
        && x.format == TensorFormat::kLINEAR && gamma.format == TensorFormat::kLINEAR
        && beta.format == TensorFormat::kLINEAR && output.format == TensorFormat::kLINEAR;
}

bool profileRanksMatchDescriptor(DynamicPluginTensorDesc const& tensor) noexcept
{
    int32_t const rank = tensor.desc.dims.nbDims;
    return tensor.min.nbDims == rank && tensor.opt.nbDims == rank && tensor.max.nbDims == rank;
}

bool profileBoundsAreOrdered(DynamicPluginTensorDesc const& tensor) noexcept
{
    if (!profileRanksMatchDescriptor(tensor))
    {
        return false;
    }
    for (int32_t index = 0; index < tensor.desc.dims.nbDims; ++index)
    {
        int32_t const declaredDimension = tensor.desc.dims.d[index];
        if (declaredDimension < -1 || tensor.min.d[index] < 0 || tensor.opt.d[index] < tensor.min.d[index]
            || tensor.max.d[index] < tensor.opt.d[index])
        {
            return false;
        }
        if (declaredDimension != -1
            && (tensor.min.d[index] != declaredDimension || tensor.opt.d[index] != declaredDimension
                || tensor.max.d[index] != declaredDimension))
        {
            return false;
        }
    }
    return true;
}

bool validateProfileBounds(DynamicPluginTensorDesc const& tensor, char const* tensorName) noexcept
{
    if (!profileRanksMatchDescriptor(tensor))
    {
        LOG_ERROR("LayerNormPlugin: %s profile ranks must match its descriptor rank", tensorName);
        return false;
    }
    if (!profileBoundsAreOrdered(tensor))
    {
        LOG_ERROR(
            "LayerNormPlugin: %s profile dimensions must satisfy 0 <= min <= opt <= max and match every "
            "fixed descriptor dimension",
            tensorName);
        return false;
    }
    return true;
}

bool hasFixedProfileHiddenSize(DynamicPluginTensorDesc const& x) noexcept
{
    int32_t const lastDimension = x.min.nbDims - 1;
    int32_t const hiddenSize = x.min.d[lastDimension];
    return x.opt.d[lastDimension] == hiddenSize && x.max.d[lastDimension] == hiddenSize;
}

float getRequiredEpsilon(PluginFieldCollection const* fields)
{
    if (fields == nullptr)
    {
        throw std::invalid_argument("LayerNormPlugin: null PluginFieldCollection");
    }
    if (fields->nbFields < 0 || (fields->nbFields > 0 && fields->fields == nullptr))
    {
        throw std::invalid_argument("LayerNormPlugin: invalid PluginFieldCollection storage");
    }

    bool found{false};
    float value{};
    for (int32_t index = 0; index < fields->nbFields; ++index)
    {
        PluginField const& field = fields->fields[index];
        if (field.name == nullptr)
        {
            throw std::invalid_argument("LayerNormPlugin: plugin field name must be non-null");
        }
        if (std::string(field.name) != kFIELD_EPSILON)
        {
            throw std::invalid_argument(std::string("LayerNormPlugin: unexpected field ") + field.name);
        }
        if (found)
        {
            throw std::invalid_argument("LayerNormPlugin: duplicate field epsilon");
        }
        if (field.type != PluginFieldType::kFLOAT32 || field.length != 1 || field.data == nullptr)
        {
            throw std::invalid_argument("LayerNormPlugin: field epsilon must be one non-null FLOAT32 scalar");
        }
        value = *static_cast<float const*>(field.data);
        found = true;
    }
    if (!found)
    {
        throw std::invalid_argument("LayerNormPlugin: missing required field epsilon");
    }
    return value;
}

void validateEpsilon(float epsilon)
{
    if (!std::isfinite(epsilon) || epsilon <= 0.0F)
    {
        throw std::invalid_argument("LayerNormPlugin: epsilon must be finite and positive");
    }
}

} // namespace

PluginFieldCollection LayerNormPluginCreator::mFieldCollection{};
std::vector<PluginField> LayerNormPluginCreator::mPluginAttributes;

#if defined(CUTE_DSL_LAYERNORM_ENABLED)
REGISTER_TENSORRT_PLUGIN(LayerNormPluginCreator);
#endif

LayerNormPlugin::LayerNormPlugin(std::string const& name, float epsilon)
    : mLayerName(name)
    , mEpsilon(epsilon)
{
    validateEpsilon(mEpsilon);
    mDataToSerialize.emplace_back(kFIELD_EPSILON, &mEpsilon, PluginFieldType::kFLOAT32, 1);
    mFieldsToSerialize.nbFields = static_cast<int32_t>(mDataToSerialize.size());
    mFieldsToSerialize.fields = mDataToSerialize.data();
}

LayerNormPlugin::LayerNormPlugin(std::string const& name, PluginFieldCollection const* fields)
    : LayerNormPlugin(name, getRequiredEpsilon(fields))
{
}

IPluginCapability* LayerNormPlugin::getCapabilityInterface(PluginCapabilityType type) noexcept
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
    LOG_ERROR("LayerNormPlugin: unsupported capability type %d", static_cast<int32_t>(type));
    return nullptr;
}

IPluginV3* LayerNormPlugin::clone() noexcept
{
    try
    {
        auto plugin = std::make_unique<LayerNormPlugin>(mLayerName, mEpsilon);
        plugin->mNamespace = mNamespace;
        return plugin.release();
    }
    catch (std::exception const& error)
    {
        LOG_ERROR("LayerNormPlugin clone failed: %s", error.what());
        return nullptr;
    }
}

char const* LayerNormPlugin::getPluginName() const noexcept
{
    return kPLUGIN_NAME;
}

char const* LayerNormPlugin::getPluginVersion() const noexcept
{
    return kPLUGIN_VERSION;
}

char const* LayerNormPlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

void LayerNormPlugin::setPluginNamespace(char const* pluginNamespace) noexcept
{
    try
    {
        mNamespace = pluginNamespace == nullptr ? "" : pluginNamespace;
    }
    catch (std::exception const& error)
    {
        LOG_ERROR("LayerNormPlugin namespace update failed: %s", error.what());
    }
}

int32_t LayerNormPlugin::getNbOutputs() const noexcept
{
    return kNB_OUTPUTS;
}

int32_t LayerNormPlugin::getOutputDataTypes(
    DataType* outputTypes, int32_t nbOutputs, DataType const* inputTypes, int32_t nbInputs) const noexcept
{
    if (outputTypes == nullptr || inputTypes == nullptr || nbInputs != kNB_INPUTS || nbOutputs != kNB_OUTPUTS
        || !isActivationDataType(inputTypes[kIN_X]) || inputTypes[kIN_GAMMA] != inputTypes[kIN_X]
        || inputTypes[kIN_BETA] != inputTypes[kIN_X])
    {
        LOG_ERROR("LayerNormPlugin: getOutputDataTypes expects homogeneous FP16 or BF16 x, gamma, and beta inputs");
        return -1;
    }
    outputTypes[0] = inputTypes[kIN_X];
    return 0;
}

int32_t LayerNormPlugin::getOutputShapes(DimsExprs const* inputs, int32_t nbInputs, DimsExprs const* shapeInputs,
    int32_t nbShapeInputs, DimsExprs* outputs, int32_t nbOutputs, IExprBuilder& exprBuilder) noexcept
{
    (void) shapeInputs;
    (void) exprBuilder;
    if (inputs == nullptr || outputs == nullptr || nbInputs != kNB_INPUTS || nbOutputs != kNB_OUTPUTS
        || nbShapeInputs != 0 || inputs[kIN_X].nbDims < kMIN_RANK || inputs[kIN_X].nbDims > kMAX_RANK
        || inputs[kIN_GAMMA].nbDims != 1 || inputs[kIN_BETA].nbDims != 1)
    {
        LOG_ERROR(
            "LayerNormPlugin: getOutputShapes expects three inputs, no shape inputs, one output, x rank 2-4, "
            "and rank-1 gamma/beta");
        return -1;
    }
    outputs[0] = inputs[kIN_X];
    return 0;
}

bool LayerNormPlugin::supportsFormatCombination(
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
    case kIN_BETA:
    {
        Dims const& xDims = inOut[kIN_X].desc.dims;
        int32_t const affineHiddenSize = tensor.dims.nbDims == 1 ? tensor.dims.d[0] : 0;
        int32_t const xHiddenSize = xDims.nbDims >= kMIN_RANK ? xDims.d[xDims.nbDims - 1] : 0;
        return tensor.type == inOut[kIN_X].desc.type && tensor.dims.nbDims == 1
            && isSupportedOrDynamicHiddenSize(affineHiddenSize)
            && (affineHiddenSize == -1 || xHiddenSize == -1 || affineHiddenSize == xHiddenSize);
    }
    case kOUT_Y:
        return tensor.type == inOut[kIN_X].desc.type && dimsKnownCompatible(inOut[kIN_X].desc.dims, tensor.dims);
    default: return false;
    }
}

int32_t LayerNormPlugin::configurePlugin(DynamicPluginTensorDesc const* inputs, int32_t nbInputs,
    DynamicPluginTensorDesc const* outputs, int32_t nbOutputs) noexcept
{
    try
    {
        if (inputs == nullptr || outputs == nullptr || nbInputs != kNB_INPUTS || nbOutputs != kNB_OUTPUTS)
        {
            LOG_ERROR("LayerNormPlugin: configurePlugin expected three inputs and one output");
            return -1;
        }
        if (!hasCoherentDescriptors(inputs[kIN_X].desc, inputs[kIN_GAMMA].desc, inputs[kIN_BETA].desc, outputs[0].desc))
        {
            LOG_ERROR("LayerNormPlugin: x, gamma, beta, and output must use one homogeneous FP16/BF16 linear format");
            return -1;
        }
        if (!validateSymbolicShapes(
                inputs[kIN_X].desc.dims, inputs[kIN_GAMMA].desc.dims, inputs[kIN_BETA].desc.dims, outputs[0].desc.dims))
        {
            return -1;
        }
        if (!validateProfileBounds(inputs[kIN_X], "x") || !validateProfileBounds(inputs[kIN_GAMMA], "gamma")
            || !validateProfileBounds(inputs[kIN_BETA], "beta") || !validateProfileBounds(outputs[0], "output"))
        {
            return -1;
        }
        if (!validateConcreteShapes(inputs[kIN_X].min, inputs[kIN_GAMMA].min, inputs[kIN_BETA].min, &outputs[0].min,
                true, "minimum profile")
            || !validateConcreteShapes(inputs[kIN_X].opt, inputs[kIN_GAMMA].opt, inputs[kIN_BETA].opt, &outputs[0].opt,
                false, "optimum profile")
            || !validateConcreteShapes(inputs[kIN_X].max, inputs[kIN_GAMMA].max, inputs[kIN_BETA].max, &outputs[0].max,
                false, "maximum profile"))
        {
            return -1;
        }
        if (!hasFixedProfileHiddenSize(inputs[kIN_X]))
        {
            LOG_ERROR("LayerNormPlugin: optimization profile hidden size H must be fixed");
            return -1;
        }

#if defined(CUTE_DSL_LAYERNORM_ENABLED)
        int32_t maxRows{};
        static_cast<void>(flattenRows(inputs[kIN_X].max, false, maxRows));
        int32_t const hiddenSize = inputs[kIN_X].max.d[inputs[kIN_X].max.nbDims - 1];
        int32_t const smVersion = getSMVersion();
        if (!CuteDslLayerNormRunner::canImplement(maxRows, hiddenSize, smVersion, inputs[kIN_X].desc.type))
        {
            LOG_ERROR("LayerNormPlugin: no LayerNorm CuTe DSL variant supports rows=%d H=%d dtype=%d on SM%d", maxRows,
                hiddenSize, static_cast<int32_t>(inputs[kIN_X].desc.type), smVersion);
            return -1;
        }
#else
        LOG_ERROR(
            "LayerNormPlugin: layernorm CuTe DSL artifacts are not linked. Generate them with "
            "kernelSrcs/build_cutedsl.py --kernels layernorm --gpu_arch <sm_NN> --arch <arch>, then rebuild "
            "with -DENABLE_CUTE_DSL=layernorm -DCUTE_DSL_ARTIFACT_TAG=<sm_NN>.");
        return -1;
#endif
        return 0;
    }
    catch (std::exception const& error)
    {
        LOG_ERROR("LayerNormPlugin configurePlugin failed: %s", error.what());
    }
    catch (...)
    {
        LOG_ERROR("LayerNormPlugin configurePlugin failed: unknown error");
    }
    return -1;
}

size_t LayerNormPlugin::getWorkspaceSize(DynamicPluginTensorDesc const* inputs, int32_t nbInputs,
    DynamicPluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept
{
    if (inputs == nullptr || outputs == nullptr || nbInputs != kNB_INPUTS || nbOutputs != kNB_OUTPUTS)
    {
        LOG_ERROR("LayerNormPlugin: getWorkspaceSize expected three inputs and one output");
    }
    return 0;
}

int32_t LayerNormPlugin::enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc,
    void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    (void) workspace;
    try
    {
        if (inputDesc == nullptr || outputDesc == nullptr)
        {
            LOG_ERROR("LayerNormPlugin: input and output descriptors must be non-null at enqueue");
            return -1;
        }
        if (!hasCoherentDescriptors(inputDesc[kIN_X], inputDesc[kIN_GAMMA], inputDesc[kIN_BETA], outputDesc[0]))
        {
            LOG_ERROR(
                "LayerNormPlugin: runtime x, gamma, beta, and output must use one homogeneous FP16/BF16 linear format");
            return -1;
        }
        if (!validateConcreteShapes(inputDesc[kIN_X].dims, inputDesc[kIN_GAMMA].dims, inputDesc[kIN_BETA].dims,
                &outputDesc[0].dims, true, "runtime"))
        {
            return -1;
        }

        int32_t rows{};
        static_cast<void>(flattenRows(inputDesc[kIN_X].dims, true, rows));
        if (rows == 0)
        {
            return 0;
        }
        if (inputs == nullptr || outputs == nullptr || inputs[kIN_X] == nullptr || inputs[kIN_GAMMA] == nullptr
            || inputs[kIN_BETA] == nullptr || outputs[0] == nullptr)
        {
            LOG_ERROR("LayerNormPlugin: nonzero-row enqueue requires non-null x, gamma, beta, and output pointers");
            return -1;
        }

#if defined(CUTE_DSL_LAYERNORM_ENABLED)
        CuteDslLayerNormParams parameters{};
        parameters.input = inputs[kIN_X];
        parameters.gamma = inputs[kIN_GAMMA];
        parameters.beta = inputs[kIN_BETA];
        parameters.output = outputs[0];
        parameters.rows = rows;
        parameters.hiddenSize = inputDesc[kIN_X].dims.d[inputDesc[kIN_X].dims.nbDims - 1];
        parameters.epsilon = mEpsilon;
        parameters.dataType = inputDesc[kIN_X].type;
        return CuteDslLayerNormRunner::run(parameters, stream);
#else
        (void) stream;
        LOG_ERROR(
            "LayerNormPlugin: layernorm CuTe DSL artifacts are not linked; rebuild with "
            "-DENABLE_CUTE_DSL=layernorm");
        return -1;
#endif
    }
    catch (std::exception const& error)
    {
        LOG_ERROR("LayerNormPlugin enqueue failed: %s", error.what());
    }
    catch (...)
    {
        LOG_ERROR("LayerNormPlugin enqueue failed: unknown error");
    }
    return -1;
}

int32_t LayerNormPlugin::onShapeChange(
    PluginTensorDesc const* inputs, int32_t nbInputs, PluginTensorDesc const* outputs, int32_t nbOutputs) noexcept
{
    try
    {
        if (inputs == nullptr || outputs == nullptr || nbInputs != kNB_INPUTS || nbOutputs != kNB_OUTPUTS)
        {
            LOG_ERROR("LayerNormPlugin: onShapeChange expected three inputs and one output");
            return -1;
        }
        if (!hasCoherentDescriptors(inputs[kIN_X], inputs[kIN_GAMMA], inputs[kIN_BETA], outputs[0]))
        {
            LOG_ERROR(
                "LayerNormPlugin: runtime x, gamma, beta, and output must use one homogeneous FP16/BF16 linear format");
            return -1;
        }
        if (!validateConcreteShapes(
                inputs[kIN_X].dims, inputs[kIN_GAMMA].dims, inputs[kIN_BETA].dims, &outputs[0].dims, true, "runtime"))
        {
            return -1;
        }

        int32_t rows{};
        static_cast<void>(flattenRows(inputs[kIN_X].dims, true, rows));
        if (rows == 0)
        {
            return 0;
        }
#if defined(CUTE_DSL_LAYERNORM_ENABLED)
        int32_t const hiddenSize = inputs[kIN_X].dims.d[inputs[kIN_X].dims.nbDims - 1];
        int32_t const smVersion = getSMVersion();
        if (!CuteDslLayerNormRunner::canImplement(rows, hiddenSize, smVersion, inputs[kIN_X].type))
        {
            LOG_ERROR("LayerNormPlugin: no LayerNorm CuTe DSL variant supports runtime rows=%d H=%d dtype=%d on SM%d",
                rows, hiddenSize, static_cast<int32_t>(inputs[kIN_X].type), smVersion);
            return -1;
        }
        return 0;
#else
        LOG_ERROR(
            "LayerNormPlugin: layernorm CuTe DSL artifacts are not linked; rebuild with "
            "-DENABLE_CUTE_DSL=layernorm");
        return -1;
#endif
    }
    catch (std::exception const& error)
    {
        LOG_ERROR("LayerNormPlugin onShapeChange failed: %s", error.what());
    }
    catch (...)
    {
        LOG_ERROR("LayerNormPlugin onShapeChange failed: unknown error");
    }
    return -1;
}

IPluginV3* LayerNormPlugin::attachToContext(IPluginResourceContext* context) noexcept
{
    (void) context;
    return clone();
}

PluginFieldCollection const* LayerNormPlugin::getFieldsToSerialize() noexcept
{
    return &mFieldsToSerialize;
}

LayerNormPluginCreator::LayerNormPluginCreator()
{
    static std::once_flag sOnce;
    std::call_once(sOnce, [] {
        mPluginAttributes.emplace_back(kFIELD_EPSILON, nullptr, PluginFieldType::kFLOAT32, 1);
        mFieldCollection.nbFields = static_cast<int32_t>(mPluginAttributes.size());
        mFieldCollection.fields = mPluginAttributes.data();
    });
}

char const* LayerNormPluginCreator::getPluginName() const noexcept
{
    return kPLUGIN_NAME;
}

char const* LayerNormPluginCreator::getPluginVersion() const noexcept
{
    return kPLUGIN_VERSION;
}

PluginFieldCollection const* LayerNormPluginCreator::getFieldNames() noexcept
{
    return &mFieldCollection;
}

char const* LayerNormPluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

void LayerNormPluginCreator::setPluginNamespace(char const* pluginNamespace) noexcept
{
    try
    {
        mNamespace = pluginNamespace == nullptr ? "" : pluginNamespace;
    }
    catch (std::exception const& error)
    {
        LOG_ERROR("LayerNormPluginCreator namespace update failed: %s", error.what());
    }
}

IPluginV3* LayerNormPluginCreator::createPlugin(
    char const* name, PluginFieldCollection const* fields, TensorRTPhase phase) noexcept
{
    (void) phase;
    try
    {
        if (name == nullptr)
        {
            throw std::invalid_argument("LayerNormPlugin: null layer name");
        }
#if defined(CUTE_DSL_LAYERNORM_ENABLED)
        int32_t const smVersion = getSMVersion();
        if (!CuteDslLayerNormRunner::canImplement(1, 4096, smVersion, DataType::kHALF))
        {
            throw std::invalid_argument("LayerNormPlugin: linked CuTe DSL artifact does not support the active GPU SM");
        }
#endif
        auto plugin = std::make_unique<LayerNormPlugin>(name, fields);
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin.release();
    }
    catch (std::exception const& error)
    {
        LOG_ERROR("LayerNormPlugin creation failed: %s", error.what());
        return nullptr;
    }
}

} // namespace plugins
} // namespace trt_edgellm
