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

#include "fp8BlockGemmPlugin.h"

#include "common/cudaUtils.h"
#include "common/logger.h"
#include "kernels/fp8BlockwiseGemm/cuteDslFp8BlockwiseGemmRunner.h"
#include "kernels/fp8BlockwiseGemm/fp8PerTokenGroupQuantize.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>

using namespace nvinfer1;

namespace trt_edgellm::plugins
{
namespace
{

constexpr char const* kPluginName{"Fp8BlockGemmPlugin"};
constexpr char const* kPluginVersion{"1"};
constexpr int32_t kInputActivation{0};
constexpr int32_t kInputWeight{1};
constexpr int32_t kInputWeightScales{2};
constexpr int32_t kOutput{3};
constexpr int32_t kNbInputs{3};
constexpr int32_t kNbOutputs{1};
constexpr int32_t kFieldGemmN{0};
constexpr int32_t kFieldGemmK{1};
constexpr int32_t kNbFields{2};
constexpr size_t kWorkspaceAlignment{256};

bool checkedMultiply(size_t lhs, size_t rhs, size_t& result) noexcept
{
    if (lhs != 0 && rhs > std::numeric_limits<size_t>::max() / lhs)
    {
        return false;
    }
    result = lhs * rhs;
    return true;
}

bool checkedAlignUp(size_t value, size_t& result) noexcept
{
    if (value > std::numeric_limits<size_t>::max() - (kWorkspaceAlignment - 1))
    {
        return false;
    }
    result = (value + kWorkspaceAlignment - 1) & ~(kWorkspaceAlignment - 1);
    return true;
}

int32_t getTokenCount(Dims const& dims) noexcept
{
    if (dims.nbDims == 2)
    {
        return dims.d[0] > 0 && dims.d[0] <= std::numeric_limits<int32_t>::max() ? static_cast<int32_t>(dims.d[0]) : 0;
    }
    if (dims.nbDims != 3 || dims.d[0] <= 0 || dims.d[1] <= 0
        || dims.d[0] > std::numeric_limits<int32_t>::max() / dims.d[1])
    {
        return 0;
    }
    return static_cast<int32_t>(dims.d[0] * dims.d[1]);
}

bool hasMatchingOutputShape(Dims const& input, Dims const& output, int32_t gemmN) noexcept
{
    if (input.nbDims != output.nbDims || (input.nbDims != 2 && input.nbDims != 3)
        || output.d[output.nbDims - 1] != gemmN)
    {
        return false;
    }
    for (int32_t index = 0; index < input.nbDims - 1; ++index)
    {
        if (input.d[index] != output.d[index])
        {
            return false;
        }
    }
    return true;
}

bool computeWorkspaceLayout(int32_t numTokens, int32_t gemmK, size_t& scaleOffset, size_t& totalBytes) noexcept
{
    if (numTokens <= 0 || gemmK <= 0 || gemmK % kernels::kFp8BlockwiseGroupSize != 0)
    {
        return false;
    }

    size_t activationBytes{0};
    size_t scaleElements{0};
    size_t scaleBytes{0};
    size_t alignedScaleBytes{0};
    if (!checkedMultiply(static_cast<size_t>(numTokens), static_cast<size_t>(gemmK), activationBytes)
        || !checkedAlignUp(activationBytes, scaleOffset)
        || !checkedMultiply(
            static_cast<size_t>(numTokens), static_cast<size_t>(gemmK / kernels::kFp8BlockwiseGroupSize), scaleElements)
        || !checkedMultiply(scaleElements, sizeof(float), scaleBytes) || !checkedAlignUp(scaleBytes, alignedScaleBytes)
        || scaleOffset > std::numeric_limits<size_t>::max() - alignedScaleBytes)
    {
        return false;
    }
    totalBytes = scaleOffset + alignedScaleBytes;
    return true;
}

void validateAttributes(int32_t gemmN, int32_t gemmK)
{
    if (gemmN <= 0 || gemmN % kernels::kFp8BlockwiseGroupSize != 0 || gemmK <= 0
        || gemmK % kernels::kFp8BlockwiseGroupSize != 0)
    {
        throw std::invalid_argument("Fp8BlockGemmPlugin: gemm_n and gemm_k must be positive multiples of 128");
    }
}

} // namespace

PluginFieldCollection Fp8BlockGemmPluginCreator::mFieldCollection{};
std::vector<PluginField> Fp8BlockGemmPluginCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(Fp8BlockGemmPluginCreator);

Fp8BlockGemmPlugin::Fp8BlockGemmPlugin(std::string const& name, int32_t gemmN, int32_t gemmK)
    : mLayerName(name)
    , mGemmN(gemmN)
    , mGemmK(gemmK)
{
    validateAttributes(mGemmN, mGemmK);
}

Fp8BlockGemmPlugin::Fp8BlockGemmPlugin(std::string const& name, PluginFieldCollection const* fields)
    : mLayerName(name)
{
    if (fields == nullptr || fields->fields == nullptr || fields->nbFields <= 0)
    {
        throw std::invalid_argument("Fp8BlockGemmPlugin: plugin field collection must not be empty");
    }

    std::array<bool, kNbFields> fieldsSeen{};
    for (int32_t index = 0; index < fields->nbFields; ++index)
    {
        PluginField const& field = fields->fields[index];
        if (field.name == nullptr || field.data == nullptr || field.type != PluginFieldType::kINT32
            || field.length != 1)
        {
            throw std::invalid_argument("Fp8BlockGemmPlugin: attributes must be named scalar INT32 fields");
        }
        std::string const fieldName(field.name);
        int32_t fieldIndex{-1};
        if (fieldName == "gemm_n")
        {
            fieldIndex = kFieldGemmN;
            mGemmN = *static_cast<int32_t const*>(field.data);
        }
        else if (fieldName == "gemm_k")
        {
            fieldIndex = kFieldGemmK;
            mGemmK = *static_cast<int32_t const*>(field.data);
        }
        else
        {
            throw std::invalid_argument("Fp8BlockGemmPlugin: unknown plugin attribute " + fieldName);
        }
        if (fieldsSeen[fieldIndex])
        {
            throw std::invalid_argument("Fp8BlockGemmPlugin: duplicate plugin attribute " + fieldName);
        }
        fieldsSeen[fieldIndex] = true;
    }
    if (!fieldsSeen[kFieldGemmN] || !fieldsSeen[kFieldGemmK])
    {
        throw std::invalid_argument("Fp8BlockGemmPlugin: gemm_n and gemm_k attributes are required");
    }
    validateAttributes(mGemmN, mGemmK);
}

IPluginCapability* Fp8BlockGemmPlugin::getCapabilityInterface(PluginCapabilityType type) noexcept
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

IPluginV3* Fp8BlockGemmPlugin::clone() noexcept
{
    try
    {
        auto plugin = std::make_unique<Fp8BlockGemmPlugin>(mLayerName, mGemmN, mGemmK);
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin.release();
    }
    catch (std::exception const& error)
    {
        LOG_ERROR("Fp8BlockGemmPlugin clone failed: %s", error.what());
        return nullptr;
    }
}

char const* Fp8BlockGemmPlugin::getPluginName() const noexcept
{
    return kPluginName;
}

char const* Fp8BlockGemmPlugin::getPluginVersion() const noexcept
{
    return kPluginVersion;
}

char const* Fp8BlockGemmPlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

void Fp8BlockGemmPlugin::setPluginNamespace(char const* pluginNamespace) noexcept
{
    try
    {
        mNamespace = pluginNamespace == nullptr ? "" : pluginNamespace;
    }
    catch (std::exception const& error)
    {
        LOG_ERROR("Fp8BlockGemmPlugin namespace update failed: %s", error.what());
    }
}

int32_t Fp8BlockGemmPlugin::getNbOutputs() const noexcept
{
    return kNbOutputs;
}

int32_t Fp8BlockGemmPlugin::getOutputDataTypes(
    DataType* outputTypes, int32_t nbOutputs, DataType const* inputTypes, int32_t nbInputs) const noexcept
{
    if (outputTypes == nullptr || inputTypes == nullptr || nbInputs != kNbInputs || nbOutputs != kNbOutputs
        || inputTypes[kInputActivation] != DataType::kHALF)
    {
        return -1;
    }
    outputTypes[0] = DataType::kHALF;
    return 0;
}

int32_t Fp8BlockGemmPlugin::getOutputShapes(DimsExprs const* inputs, int32_t nbInputs, DimsExprs const* shapeInputs,
    int32_t nbShapeInputs, DimsExprs* outputs, int32_t nbOutputs, IExprBuilder& exprBuilder) noexcept
{
    (void) shapeInputs;
    (void) nbShapeInputs;
    if (inputs == nullptr || outputs == nullptr || nbInputs != kNbInputs || nbOutputs != kNbOutputs
        || (inputs[kInputActivation].nbDims != 2 && inputs[kInputActivation].nbDims != 3))
    {
        return -1;
    }

    outputs[0].nbDims = inputs[kInputActivation].nbDims;
    for (int32_t index = 0; index < outputs[0].nbDims - 1; ++index)
    {
        outputs[0].d[index] = inputs[kInputActivation].d[index];
    }
    outputs[0].d[outputs[0].nbDims - 1] = exprBuilder.constant(mGemmN);
    return 0;
}

bool Fp8BlockGemmPlugin::validateTensorDesc(int32_t pos, PluginTensorDesc const& desc) const noexcept
{
    if (desc.format != TensorFormat::kLINEAR)
    {
        return false;
    }
    switch (pos)
    {
    case kInputActivation:
        return desc.type == DataType::kHALF && (desc.dims.nbDims == 2 || desc.dims.nbDims == 3)
            && desc.dims.d[desc.dims.nbDims - 1] == mGemmK;
    case kInputWeight:
        return desc.type == DataType::kINT8 && desc.dims.nbDims == 2 && desc.dims.d[0] == mGemmN
            && desc.dims.d[1] == mGemmK;
    case kInputWeightScales:
        return desc.type == DataType::kFLOAT && desc.dims.nbDims == 2
            && desc.dims.d[0] == mGemmN / kernels::kFp8BlockwiseGroupSize
            && desc.dims.d[1] == mGemmK / kernels::kFp8BlockwiseGroupSize;
    case kOutput:
        return desc.type == DataType::kHALF && (desc.dims.nbDims == 2 || desc.dims.nbDims == 3)
            && desc.dims.d[desc.dims.nbDims - 1] == mGemmN;
    default: return false;
    }
}

bool Fp8BlockGemmPlugin::supportsFormatCombination(
    int32_t pos, DynamicPluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
    return inOut != nullptr && nbInputs == kNbInputs && nbOutputs == kNbOutputs && pos >= 0
        && pos < nbInputs + nbOutputs && validateTensorDesc(pos, inOut[pos].desc);
}

int32_t Fp8BlockGemmPlugin::configurePlugin(DynamicPluginTensorDesc const* inputs, int32_t nbInputs,
    DynamicPluginTensorDesc const* outputs, int32_t nbOutputs) noexcept
{
    try
    {
        if (inputs == nullptr || outputs == nullptr || nbInputs != kNbInputs || nbOutputs != kNbOutputs)
        {
            return -1;
        }
        for (int32_t pos = 0; pos < kNbInputs; ++pos)
        {
            if (!validateTensorDesc(pos, inputs[pos].desc))
            {
                LOG_ERROR("Fp8BlockGemmPlugin: invalid input descriptor at position %d", pos);
                return -1;
            }
        }
        if (!validateTensorDesc(kOutput, outputs[0].desc))
        {
            LOG_ERROR("Fp8BlockGemmPlugin: invalid output descriptor");
            return -1;
        }

        std::array<Dims, 3> const inputProfiles{
            inputs[kInputActivation].min, inputs[kInputActivation].opt, inputs[kInputActivation].max};
        std::array<Dims, 3> const outputProfiles{outputs[0].min, outputs[0].opt, outputs[0].max};
        int32_t const smVersion = getSMVersion();
        for (size_t index = 0; index < inputProfiles.size(); ++index)
        {
            int32_t const numTokens = getTokenCount(inputProfiles[index]);
            if (inputProfiles[index].d[inputProfiles[index].nbDims - 1] != mGemmK
                || !hasMatchingOutputShape(inputProfiles[index], outputProfiles[index], mGemmN)
                || !kernels::CuteDslFp8BlockwiseGemmRunner::isSupported(smVersion, numTokens, mGemmN, mGemmK))
            {
                LOG_ERROR("Fp8BlockGemmPlugin: invalid or unsupported optimization profile");
                return -1;
            }
        }
        return 0;
    }
    catch (std::exception const& error)
    {
        LOG_ERROR("Fp8BlockGemmPlugin configurePlugin failed: %s", error.what());
        return -1;
    }
}

size_t Fp8BlockGemmPlugin::getWorkspaceSize(DynamicPluginTensorDesc const* inputs, int32_t nbInputs,
    DynamicPluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept
{
    if (inputs == nullptr || outputs == nullptr || nbInputs != kNbInputs || nbOutputs != kNbOutputs)
    {
        return 0;
    }

    int32_t const maxTokens = getTokenCount(inputs[kInputActivation].max);
    size_t scaleOffset{0};
    size_t totalBytes{0};
    if (!computeWorkspaceLayout(maxTokens, mGemmK, scaleOffset, totalBytes))
    {
        LOG_ERROR("Fp8BlockGemmPlugin: workspace size overflow or invalid maximum activation shape");
        return 0;
    }
    return totalBytes;
}

int32_t Fp8BlockGemmPlugin::enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc,
    void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    if (inputDesc == nullptr || outputDesc == nullptr || inputs == nullptr || outputs == nullptr || workspace == nullptr
        || stream == nullptr || reinterpret_cast<uintptr_t>(workspace) % kWorkspaceAlignment != 0)
    {
        return -1;
    }
    for (int32_t pos = 0; pos < kNbInputs; ++pos)
    {
        if (!validateTensorDesc(pos, inputDesc[pos]) || inputs[pos] == nullptr)
        {
            return -1;
        }
    }
    if (!validateTensorDesc(kOutput, outputDesc[0]) || outputs[0] == nullptr
        || !hasMatchingOutputShape(inputDesc[kInputActivation].dims, outputDesc[0].dims, mGemmN))
    {
        return -1;
    }

    int32_t const numTokens = getTokenCount(inputDesc[kInputActivation].dims);
    size_t scaleOffset{0};
    size_t totalBytes{0};
    if (!computeWorkspaceLayout(numTokens, mGemmK, scaleOffset, totalBytes))
    {
        return -1;
    }
    (void) totalBytes;

    cudaError_t error = kernels::CuteDslFp8BlockwiseGemmRunner::prepare(stream);
    if (error != cudaSuccess)
    {
        LOG_ERROR("Fp8BlockGemmPlugin: CuTe DSL module preparation failed: %s", cudaGetErrorString(error));
        return -1;
    }

    auto* const quantizedActivation = reinterpret_cast<__nv_fp8_e4m3*>(workspace);
    auto* const activationScales = reinterpret_cast<float*>(static_cast<uint8_t*>(workspace) + scaleOffset);
    error = kernels::launchFp8PerTokenGroupQuantize(static_cast<half const*>(inputs[kInputActivation]),
        quantizedActivation, activationScales, numTokens, mGemmK, stream);
    if (error != cudaSuccess)
    {
        LOG_ERROR("Fp8BlockGemmPlugin: activation quantization launch failed: %s", cudaGetErrorString(error));
        return -1;
    }

    kernels::CuteDslFp8BlockwiseGemmParams const params{quantizedActivation, inputs[kInputWeight], activationScales,
        static_cast<float const*>(inputs[kInputWeightScales]), outputs[0], numTokens, mGemmN, mGemmK};
    error = kernels::CuteDslFp8BlockwiseGemmRunner::run(params, stream);
    if (error != cudaSuccess)
    {
        LOG_ERROR("Fp8BlockGemmPlugin: blockwise GEMM launch failed: %s", cudaGetErrorString(error));
        return -1;
    }
    return 0;
}

int32_t Fp8BlockGemmPlugin::onShapeChange(
    PluginTensorDesc const* inputs, int32_t nbInputs, PluginTensorDesc const* outputs, int32_t nbOutputs) noexcept
{
    if (inputs == nullptr || outputs == nullptr || nbInputs != kNbInputs || nbOutputs != kNbOutputs)
    {
        return -1;
    }
    for (int32_t pos = 0; pos < kNbInputs; ++pos)
    {
        if (!validateTensorDesc(pos, inputs[pos]))
        {
            return -1;
        }
    }
    return validateTensorDesc(kOutput, outputs[0])
            && hasMatchingOutputShape(inputs[kInputActivation].dims, outputs[0].dims, mGemmN)
        ? 0
        : -1;
}

IPluginV3* Fp8BlockGemmPlugin::attachToContext(IPluginResourceContext* context) noexcept
{
    (void) context;
    return clone();
}

PluginFieldCollection const* Fp8BlockGemmPlugin::getFieldsToSerialize() noexcept
{
    try
    {
        mDataToSerialize.clear();
        mDataToSerialize.emplace_back("gemm_n", &mGemmN, PluginFieldType::kINT32, 1);
        mDataToSerialize.emplace_back("gemm_k", &mGemmK, PluginFieldType::kINT32, 1);
        mFieldsToSerialize.nbFields = static_cast<int32_t>(mDataToSerialize.size());
        mFieldsToSerialize.fields = mDataToSerialize.data();
        return &mFieldsToSerialize;
    }
    catch (std::exception const& error)
    {
        LOG_ERROR("Fp8BlockGemmPlugin serialization failed: %s", error.what());
        return nullptr;
    }
}

Fp8BlockGemmPluginCreator::Fp8BlockGemmPluginCreator()
{
    static std::once_flag once;
    std::call_once(once, [] {
        mPluginAttributes.emplace_back("gemm_n", nullptr, PluginFieldType::kINT32, 1);
        mPluginAttributes.emplace_back("gemm_k", nullptr, PluginFieldType::kINT32, 1);
        mFieldCollection.nbFields = static_cast<int32_t>(mPluginAttributes.size());
        mFieldCollection.fields = mPluginAttributes.data();
    });
}

char const* Fp8BlockGemmPluginCreator::getPluginName() const noexcept
{
    return kPluginName;
}

char const* Fp8BlockGemmPluginCreator::getPluginVersion() const noexcept
{
    return kPluginVersion;
}

PluginFieldCollection const* Fp8BlockGemmPluginCreator::getFieldNames() noexcept
{
    return &mFieldCollection;
}

char const* Fp8BlockGemmPluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

void Fp8BlockGemmPluginCreator::setPluginNamespace(char const* pluginNamespace) noexcept
{
    try
    {
        mNamespace = pluginNamespace == nullptr ? "" : pluginNamespace;
    }
    catch (std::exception const& error)
    {
        LOG_ERROR("Fp8BlockGemmPluginCreator namespace update failed: %s", error.what());
    }
}

IPluginV3* Fp8BlockGemmPluginCreator::createPlugin(
    char const* name, PluginFieldCollection const* fields, TensorRTPhase phase) noexcept
{
    (void) phase;
    try
    {
        auto plugin = std::make_unique<Fp8BlockGemmPlugin>(name == nullptr ? "" : name, fields);
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin.release();
    }
    catch (std::exception const& error)
    {
        LOG_ERROR("Fp8BlockGemmPlugin creation failed: %s", error.what());
        return nullptr;
    }
}

} // namespace trt_edgellm::plugins
