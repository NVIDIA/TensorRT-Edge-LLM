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

#include "nvfp4A16MoePlugin.h"

#include "common/checkMacros.h"
#include "common/cudaUtils.h"
#include "common/logger.h"
#include "common/tensor.h"
#include "kernels/moe/moeActivationKernels.h"
#include "kernels/moe/moeAlignSumKernels.h"
#include "kernels/moe/moeMarlinIndicesKernels.h"
#include "kernels/moe/moeSigmoidGroupTopkKernels.h"
#include "kernels/moe/moeTopkSoftmaxKernels.h"
#include "kernels/moe/moe_marlin/moeMarlin.h"
#include "plugins/utils/pluginUtils.h"
#include "profiling/nvtx_wrapper.h"

#include <NvInfer.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
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

constexpr int32_t kInRouterLogits{0};
constexpr int32_t kInHiddenStates{1};
constexpr int32_t kInFc1QWeights{2};
constexpr int32_t kInFc1BlockScales{3};
constexpr int32_t kInFc1GlobalScales{4};
constexpr int32_t kInFc2QWeights{5};
constexpr int32_t kInFc2BlockScales{6};
constexpr int32_t kInFc2GlobalScales{7};
constexpr int32_t kInExpertScoreBias{8};
constexpr int32_t kOutOutput{9};
constexpr int32_t kNbPluginInputs{9};

constexpr char const* kPluginName{"Nvfp4A16MoePlugin"};
constexpr char const* kPluginVersion{"1"};

constexpr int32_t kActivationSwiGlu{2};
constexpr int32_t kActivationRelu2{4};
constexpr int32_t kRoutingSoftmaxTopk{0};
constexpr int32_t kRoutingSigmoidGroupTopk{1};
constexpr int32_t kNvfp4GroupSize{16};
constexpr int32_t kSupportedNumExperts[]{128, 256, 512};
constexpr int32_t kMaxTopK{32};
constexpr int32_t kDecodeBlockSize{8};
constexpr int32_t kPrefillBlockSize{32};

constexpr int32_t kFieldNumExperts{0};
constexpr int32_t kFieldTopK{1};
constexpr int32_t kFieldHiddenSize{2};
constexpr int32_t kFieldMoeInterSize{3};
constexpr int32_t kFieldActivationType{4};
constexpr int32_t kFieldNGroup{5};
constexpr int32_t kFieldTopkGroup{6};
constexpr int32_t kFieldNormTopkProb{7};
constexpr int32_t kFieldRoutedScalingFactor{8};
constexpr int32_t kFieldRoutingMode{9};
constexpr int32_t kFieldMaxRoutedRows{10};
constexpr int32_t kNbPluginFields{11};

int32_t getFc1OutDim(int32_t moeInterSize, int32_t activationType)
{
    return activationType == kActivationSwiGlu ? 2 * moeInterSize : moeInterSize;
}

int32_t getMoeBlockSize(int64_t seqLen)
{
    return seqLen == 1 ? kDecodeBlockSize : kPrefillBlockSize;
}

bool getConservativePaddedRows(
    int64_t numTokens, int32_t topK, int32_t numExperts, int32_t blockSize, int64_t& paddedRows)
{
    int64_t const padding = static_cast<int64_t>(numExperts) * (blockSize - 1);
    if (numTokens <= 0 || numTokens > (std::numeric_limits<int64_t>::max() - padding) / topK)
    {
        return false;
    }
    paddedRows = numTokens * topK + padding;
    return true;
}

bool isActivationDataType(DataType dataType) noexcept
{
    return dataType == DataType::kHALF || dataType == DataType::kBF16;
}

bool hasCoherentActivationDataTypes(PluginTensorDesc const& hiddenStates, PluginTensorDesc const& fc1GlobalScales,
    PluginTensorDesc const& fc2GlobalScales, PluginTensorDesc const& output) noexcept
{
    DataType const dataType = hiddenStates.type;
    return isActivationDataType(dataType) && fc1GlobalScales.type == dataType && fc2GlobalScales.type == dataType
        && output.type == dataType;
}

size_t computeWorkspaceSize(int32_t maxTokens, int32_t maxRoutedRows, int32_t numExperts, int32_t topK,
    int32_t hiddenSize, int32_t moeInterSize, int32_t activationType, int32_t routingMode,
    DataType activationDataType) noexcept
{
    try
    {
        int32_t device = 0;
        int32_t numSms = 0;
        CUDA_CHECK(cudaGetDevice(&device));
        CUDA_CHECK(cudaDeviceGetAttribute(&numSms, cudaDevAttrMultiProcessorCount, device));

        int64_t const totalSlots = static_cast<int64_t>(maxTokens) * topK;
        int32_t const fc1OutDim = getFc1OutDim(moeInterSize, activationType);
        // Decode uses the smallest block and therefore needs the largest expert-id array for a fixed row cap.
        int64_t const maxPaddedBlocks = divUp(maxRoutedRows, kDecodeBlockSize);
        size_t const softmaxWorkspaceBytes
            = routingMode == kRoutingSoftmaxTopk ? kernel::getMoeTopkSoftmaxWorkspaceSize(maxTokens, numExperts) : 0;

        int64_t marlinWorkspaceElements = 0;
        // A dynamic profile can execute both decode and prefill, so retain the largest workspace across both block
        // sizes and both projections.
        for (int32_t const blockSize : {kDecodeBlockSize, kPrefillBlockSize})
        {
            marlinWorkspaceElements = std::max(marlinWorkspaceElements,
                kernel::getMoeMarlinWorkspaceSize(maxRoutedRows, fc1OutDim, blockSize, numSms));
            marlinWorkspaceElements = std::max(marlinWorkspaceElements,
                kernel::getMoeMarlinWorkspaceSize(maxRoutedRows, hiddenSize, blockSize, numSms));
        }

        size_t size = 0;
        size = accumulateWorkspaceSize(size, rt::Coords{maxTokens, topK}, DataType::kFLOAT);
        size = accumulateWorkspaceSize(size, rt::Coords{maxTokens, topK}, DataType::kINT32);
        if (softmaxWorkspaceBytes > 0)
        {
            size = accumulateWorkspaceSize(
                size, rt::Coords{static_cast<int64_t>(softmaxWorkspaceBytes)}, DataType::kINT8);
        }
        size = accumulateWorkspaceSize(size, rt::Coords{maxRoutedRows}, DataType::kINT32);
        size = accumulateWorkspaceSize(size, rt::Coords{maxPaddedBlocks}, DataType::kINT32);
        size = accumulateWorkspaceSize(size, rt::Coords{1}, DataType::kINT32);
        size = accumulateWorkspaceSize(size, rt::Coords{maxRoutedRows}, DataType::kFLOAT);
        size = accumulateWorkspaceSize(size, rt::Coords{numExperts}, DataType::kINT32);
        size = accumulateWorkspaceSize(size, rt::Coords{numExperts}, DataType::kINT32);
        size = accumulateWorkspaceSize(size, rt::Coords{numExperts, totalSlots}, DataType::kINT32);
        size = accumulateWorkspaceSize(size, rt::Coords{numExperts}, DataType::kINT32);
        size = accumulateWorkspaceSize(size, rt::Coords{totalSlots, fc1OutDim}, activationDataType);
        size = accumulateWorkspaceSize(size, rt::Coords{totalSlots, moeInterSize}, activationDataType);
        size = accumulateWorkspaceSize(size, rt::Coords{totalSlots, hiddenSize}, activationDataType);
        size = accumulateWorkspaceSize(size, rt::Coords{marlinWorkspaceElements}, DataType::kINT32);
        size = accumulateWorkspaceSize(size, rt::Coords{maxRoutedRows}, DataType::kFLOAT);
        return size;
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("Failed to compute Nvfp4A16MoePlugin workspace size: %s", e.what());
        return 0;
    }
}

} // namespace

PluginFieldCollection Nvfp4A16MoePluginCreator::mFieldCollection{};
std::vector<PluginField> Nvfp4A16MoePluginCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(Nvfp4A16MoePluginCreator);

Nvfp4A16MoePlugin::Nvfp4A16MoePlugin(std::string const& name, int32_t numExperts, int32_t topK, int32_t hiddenSize,
    int32_t moeInterSize, int32_t activationType, int32_t nGroup, int32_t topkGroup, int32_t normTopkProb,
    float routedScalingFactor, int32_t routingMode, int32_t maxRoutedRows)
    : mLayerName(name)
    , mNumExperts(numExperts)
    , mTopK(topK)
    , mHiddenSize(hiddenSize)
    , mMoeInterSize(moeInterSize)
    , mActivationType(activationType)
    , mNGroup(nGroup)
    , mTopkGroup(topkGroup)
    , mNormTopkProb(normTopkProb)
    , mRoutedScalingFactor(routedScalingFactor)
    , mRoutingMode(routingMode)
    , mMaxRoutedRows(maxRoutedRows)
{
    validateAttributes();
}

Nvfp4A16MoePlugin::Nvfp4A16MoePlugin(std::string const& name, PluginFieldCollection const* fc)
    : mLayerName(name)
{
    if (fc == nullptr || fc->fields == nullptr || fc->nbFields <= 0)
    {
        throw std::invalid_argument("Nvfp4A16MoePlugin: plugin field collection must not be empty");
    }

    std::array<bool, kNbPluginFields> fieldsSeen{};
    auto readIntField = [&fieldsSeen](PluginField const& field, int32_t fieldIndex) {
        if (field.data == nullptr || field.type != PluginFieldType::kINT32 || field.length != 1)
        {
            throw std::invalid_argument("Nvfp4A16MoePlugin: integer attributes must be scalar INT32 fields");
        }
        if (fieldsSeen[fieldIndex])
        {
            throw std::invalid_argument("Nvfp4A16MoePlugin: duplicate plugin attribute");
        }
        fieldsSeen[fieldIndex] = true;
        return *static_cast<int32_t const*>(field.data);
    };
    auto readFloatField = [&fieldsSeen](PluginField const& field, int32_t fieldIndex) {
        if (field.data == nullptr || field.type != PluginFieldType::kFLOAT32 || field.length != 1)
        {
            throw std::invalid_argument("Nvfp4A16MoePlugin: routed_scaling_factor must be a scalar FLOAT32 field");
        }
        if (fieldsSeen[fieldIndex])
        {
            throw std::invalid_argument("Nvfp4A16MoePlugin: duplicate plugin attribute");
        }
        fieldsSeen[fieldIndex] = true;
        return *static_cast<float const*>(field.data);
    };

    for (int32_t i = 0; i < fc->nbFields; ++i)
    {
        PluginField const& field = fc->fields[i];
        if (field.name == nullptr)
        {
            throw std::invalid_argument("Nvfp4A16MoePlugin: plugin attribute name must not be null");
        }
        std::string const fieldName(field.name);
        if (fieldName == "num_experts")
        {
            mNumExperts = readIntField(field, kFieldNumExperts);
        }
        else if (fieldName == "top_k")
        {
            mTopK = readIntField(field, kFieldTopK);
        }
        else if (fieldName == "hidden_size")
        {
            mHiddenSize = readIntField(field, kFieldHiddenSize);
        }
        else if (fieldName == "moe_inter_size")
        {
            mMoeInterSize = readIntField(field, kFieldMoeInterSize);
        }
        else if (fieldName == "activation_type")
        {
            mActivationType = readIntField(field, kFieldActivationType);
        }
        else if (fieldName == "n_group")
        {
            mNGroup = readIntField(field, kFieldNGroup);
        }
        else if (fieldName == "topk_group")
        {
            mTopkGroup = readIntField(field, kFieldTopkGroup);
        }
        else if (fieldName == "norm_topk_prob")
        {
            mNormTopkProb = readIntField(field, kFieldNormTopkProb);
        }
        else if (fieldName == "routed_scaling_factor")
        {
            mRoutedScalingFactor = readFloatField(field, kFieldRoutedScalingFactor);
        }
        else if (fieldName == "routing_mode")
        {
            mRoutingMode = readIntField(field, kFieldRoutingMode);
        }
        else if (fieldName == "max_routed_rows")
        {
            mMaxRoutedRows = readIntField(field, kFieldMaxRoutedRows);
        }
        else
        {
            throw std::invalid_argument("Nvfp4A16MoePlugin: unknown plugin attribute " + fieldName);
        }
    }

    if (std::any_of(fieldsSeen.begin(), fieldsSeen.end(), [](bool seen) { return !seen; }))
    {
        throw std::invalid_argument("Nvfp4A16MoePlugin: all 11 plugin attributes are required");
    }
    validateAttributes();
}

Nvfp4A16MoePlugin::~Nvfp4A16MoePlugin() noexcept = default;

void Nvfp4A16MoePlugin::validateAttributes() const
{
    if (std::find(std::begin(kSupportedNumExperts), std::end(kSupportedNumExperts), mNumExperts)
        == std::end(kSupportedNumExperts))
    {
        throw std::invalid_argument("Nvfp4A16MoePlugin: num_experts must be one of {128, 256, 512}");
    }
    if (mTopK <= 0 || mTopK > kMaxTopK)
    {
        throw std::invalid_argument("Nvfp4A16MoePlugin: top_k must be in [1, 32]");
    }
    if (mHiddenSize <= 0 || mHiddenSize % 128 != 0)
    {
        throw std::invalid_argument("Nvfp4A16MoePlugin: hidden_size must be positive and divisible by 128");
    }
    if (mMoeInterSize <= 0)
    {
        throw std::invalid_argument("Nvfp4A16MoePlugin: moe_inter_size must be positive");
    }
    if (mActivationType == kActivationSwiGlu)
    {
        if (mMoeInterSize % 64 != 0 || mMoeInterSize > std::numeric_limits<int32_t>::max() / 2)
        {
            throw std::invalid_argument(
                "Nvfp4A16MoePlugin: SwiGLU moe_inter_size must be divisible by 64 and fit the fused FC1 width");
        }
    }
    else if (mActivationType == kActivationRelu2)
    {
        if (mMoeInterSize % 128 != 0)
        {
            throw std::invalid_argument("Nvfp4A16MoePlugin: ReLU2 moe_inter_size must be divisible by 128");
        }
    }
    else
    {
        throw std::invalid_argument("Nvfp4A16MoePlugin: activation_type must be 2 (SwiGLU) or 4 (ReLU2)");
    }
    if (mRoutingMode != kRoutingSoftmaxTopk && mRoutingMode != kRoutingSigmoidGroupTopk)
    {
        throw std::invalid_argument(
            "Nvfp4A16MoePlugin: routing_mode must be 0 (softmax top-k) or 1 (sigmoid group top-k)");
    }
    if (mNormTopkProb != 0 && mNormTopkProb != 1)
    {
        throw std::invalid_argument("Nvfp4A16MoePlugin: norm_topk_prob must be 0 or 1");
    }
    if (mMaxRoutedRows < 0)
    {
        throw std::invalid_argument("Nvfp4A16MoePlugin: max_routed_rows must be non-negative (0 == auto)");
    }
    if (mRoutingMode == kRoutingSigmoidGroupTopk)
    {
        if (!std::isfinite(mRoutedScalingFactor) || mRoutedScalingFactor <= 0.0F)
        {
            throw std::invalid_argument(
                "Nvfp4A16MoePlugin: routed_scaling_factor must be finite and positive for sigmoid group top-k");
        }
        if (mNGroup <= 0 || mNumExperts % mNGroup != 0 || mNumExperts / mNGroup < 2)
        {
            throw std::invalid_argument(
                "Nvfp4A16MoePlugin: n_group must divide num_experts with at least two experts per group");
        }
        if (mTopkGroup <= 0 || mTopkGroup > mNGroup)
        {
            throw std::invalid_argument("Nvfp4A16MoePlugin: topk_group must be in [1, n_group]");
        }
        if (mTopK > mTopkGroup * (mNumExperts / mNGroup))
        {
            throw std::invalid_argument("Nvfp4A16MoePlugin: selected expert groups cannot contain top_k experts");
        }
    }
}

IPluginCapability* Nvfp4A16MoePlugin::getCapabilityInterface(PluginCapabilityType type) noexcept
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

IPluginV3* Nvfp4A16MoePlugin::clone() noexcept
{
    try
    {
        auto* plugin = new Nvfp4A16MoePlugin(mLayerName, mNumExperts, mTopK, mHiddenSize, mMoeInterSize,
            mActivationType, mNGroup, mTopkGroup, mNormTopkProb, mRoutedScalingFactor, mRoutingMode, mMaxRoutedRows);
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("Failed to clone Nvfp4A16MoePlugin: %s", e.what());
        return nullptr;
    }
}

char const* Nvfp4A16MoePlugin::getPluginName() const noexcept
{
    return kPluginName;
}

char const* Nvfp4A16MoePlugin::getPluginVersion() const noexcept
{
    return kPluginVersion;
}

char const* Nvfp4A16MoePlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

void Nvfp4A16MoePlugin::setPluginNamespace(char const* pluginNamespace) noexcept
{
    mNamespace = pluginNamespace == nullptr ? "" : pluginNamespace;
}

int32_t Nvfp4A16MoePlugin::getNbOutputs() const noexcept
{
    return 1;
}

int32_t Nvfp4A16MoePlugin::getOutputDataTypes(
    DataType* outputTypes, int32_t nbOutputs, DataType const* inputTypes, int32_t nbInputs) const noexcept
{
    if (outputTypes == nullptr || inputTypes == nullptr || nbOutputs != 1 || nbInputs != kNbPluginInputs)
    {
        LOG_ERROR("Nvfp4A16MoePlugin: getOutputDataTypes expected %d inputs and 1 output", kNbPluginInputs);
        return -1;
    }
    if (!isActivationDataType(inputTypes[kInHiddenStates]))
    {
        LOG_ERROR("Nvfp4A16MoePlugin: hidden_states must be FP16 or BF16");
        return -1;
    }
    if (inputTypes[kInFc1GlobalScales] != inputTypes[kInHiddenStates]
        || inputTypes[kInFc2GlobalScales] != inputTypes[kInHiddenStates])
    {
        LOG_ERROR("Nvfp4A16MoePlugin: hidden_states and global scales must use the same FP16/BF16 type");
        return -1;
    }
    outputTypes[0] = inputTypes[kInHiddenStates];
    return 0;
}

int32_t Nvfp4A16MoePlugin::getOutputShapes(DimsExprs const* inputs, int32_t nbInputs, DimsExprs const* shapeInputs,
    int32_t nbShapeInputs, DimsExprs* outputs, int32_t nbOutputs, IExprBuilder& exprBuilder) noexcept
{
    if (inputs == nullptr || outputs == nullptr || nbInputs != kNbPluginInputs || nbOutputs != 1)
    {
        LOG_ERROR("Nvfp4A16MoePlugin: getOutputShapes expected %d inputs and 1 output", kNbPluginInputs);
        return -1;
    }
    (void) shapeInputs;
    (void) nbShapeInputs;
    outputs[0].nbDims = 3;
    outputs[0].d[0] = inputs[kInHiddenStates].d[0];
    outputs[0].d[1] = inputs[kInHiddenStates].d[1];
    outputs[0].d[2] = exprBuilder.constant(mHiddenSize);
    return 0;
}

bool Nvfp4A16MoePlugin::validateTensorDesc(int32_t pos, PluginTensorDesc const& desc) const noexcept
{
    if (desc.format != TensorFormat::kLINEAR)
    {
        return false;
    }

    int64_t const fc1OutDim = getFc1OutDim(mMoeInterSize, mActivationType);
    switch (pos)
    {
    case kInRouterLogits:
        return desc.type == DataType::kFLOAT && desc.dims.nbDims == 2 && desc.dims.d[1] == mNumExperts;
    case kInHiddenStates:
        return isActivationDataType(desc.type) && desc.dims.nbDims == 3 && desc.dims.d[2] == mHiddenSize;
    case kInFc1QWeights:
        return desc.type == DataType::kINT8 && desc.dims.nbDims == 3 && desc.dims.d[0] == mNumExperts
            && desc.dims.d[1] == mHiddenSize / kNvfp4GroupSize && desc.dims.d[2] == 8LL * fc1OutDim;
    case kInFc1BlockScales:
        return desc.type == DataType::kINT8 && desc.dims.nbDims == 3 && desc.dims.d[0] == mNumExperts
            && desc.dims.d[1] == mHiddenSize / kNvfp4GroupSize && desc.dims.d[2] == fc1OutDim;
    case kInFc1GlobalScales:
        return isActivationDataType(desc.type) && desc.dims.nbDims == 1 && desc.dims.d[0] == mNumExperts;
    case kInFc2QWeights:
        return desc.type == DataType::kINT8 && desc.dims.nbDims == 3 && desc.dims.d[0] == mNumExperts
            && desc.dims.d[1] == mMoeInterSize / kNvfp4GroupSize && desc.dims.d[2] == 8LL * mHiddenSize;
    case kInFc2BlockScales:
        return desc.type == DataType::kINT8 && desc.dims.nbDims == 3 && desc.dims.d[0] == mNumExperts
            && desc.dims.d[1] == mMoeInterSize / kNvfp4GroupSize && desc.dims.d[2] == mHiddenSize;
    case kInFc2GlobalScales:
        return isActivationDataType(desc.type) && desc.dims.nbDims == 1 && desc.dims.d[0] == mNumExperts;
    case kInExpertScoreBias:
        return desc.type == DataType::kFLOAT && desc.dims.nbDims == 1 && desc.dims.d[0] == mNumExperts;
    case kOutOutput: return isActivationDataType(desc.type) && desc.dims.nbDims == 3 && desc.dims.d[2] == mHiddenSize;
    default: return false;
    }
}

bool Nvfp4A16MoePlugin::supportsFormatCombination(
    int32_t pos, DynamicPluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
    if (inOut == nullptr || nbInputs != kNbPluginInputs || nbOutputs != 1 || pos < 0 || pos > kOutOutput)
    {
        return false;
    }
    if (!validateTensorDesc(pos, inOut[pos].desc))
    {
        return false;
    }
    if (pos == kInFc1GlobalScales || pos == kInFc2GlobalScales || pos == kOutOutput)
    {
        return inOut[pos].desc.type == inOut[kInHiddenStates].desc.type;
    }
    return true;
}

int32_t Nvfp4A16MoePlugin::configurePlugin(
    DynamicPluginTensorDesc const* in, int32_t nbInputs, DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept
{
    try
    {
        if (in == nullptr || out == nullptr || nbInputs != kNbPluginInputs || nbOutputs != 1)
        {
            LOG_ERROR("Nvfp4A16MoePlugin: configurePlugin expected %d inputs and 1 output", kNbPluginInputs);
            return -1;
        }
        for (int32_t pos = 0; pos < kNbPluginInputs; ++pos)
        {
            if (!validateTensorDesc(pos, in[pos].desc))
            {
                LOG_ERROR("Nvfp4A16MoePlugin: invalid input descriptor at position %d", pos);
                return -1;
            }
        }
        if (!validateTensorDesc(kOutOutput, out[0].desc))
        {
            LOG_ERROR("Nvfp4A16MoePlugin: invalid output descriptor");
            return -1;
        }
        if (!hasCoherentActivationDataTypes(
                in[kInHiddenStates].desc, in[kInFc1GlobalScales].desc, in[kInFc2GlobalScales].desc, out[0].desc))
        {
            LOG_ERROR("Nvfp4A16MoePlugin: hidden_states, global scales, and output must use the same FP16/BF16 type");
            return -1;
        }

        auto validateProfileEndpoint = [this](Dims const& hidden, Dims const& router, char const* endpoint) {
            if (hidden.nbDims != 3 || router.nbDims != 2 || hidden.d[0] <= 0 || hidden.d[1] <= 0
                || hidden.d[2] != mHiddenSize || router.d[0] <= 0 || router.d[1] != mNumExperts)
            {
                LOG_ERROR("Nvfp4A16MoePlugin: optimization profile %s dimensions are incomplete or invalid", endpoint);
                return false;
            }
            int64_t const batchSize = hidden.d[0];
            int64_t const seqLen = hidden.d[1];
            if (batchSize > std::numeric_limits<int64_t>::max() / seqLen || router.d[0] != batchSize * seqLen)
            {
                LOG_ERROR("Nvfp4A16MoePlugin: router_logits %s rows must equal hidden_states %s batch*sequence",
                    endpoint, endpoint);
                return false;
            }
            return true;
        };

        Dims const& hiddenMin = in[kInHiddenStates].min;
        Dims const& routerMin = in[kInRouterLogits].min;
        Dims const& hiddenMax = in[kInHiddenStates].max;
        Dims const& routerMax = in[kInRouterLogits].max;
        if (!validateProfileEndpoint(hiddenMin, routerMin, "minimum")
            || !validateProfileEndpoint(hiddenMax, routerMax, "maximum"))
        {
            return -1;
        }
        if (out[0].min.nbDims != 3 || out[0].max.nbDims != 3 || out[0].min.d[0] != hiddenMin.d[0]
            || out[0].min.d[1] != hiddenMin.d[1] || out[0].min.d[2] != mHiddenSize || out[0].max.d[0] != hiddenMax.d[0]
            || out[0].max.d[1] != hiddenMax.d[1] || out[0].max.d[2] != mHiddenSize)
        {
            LOG_ERROR("Nvfp4A16MoePlugin: output profile range must match hidden_states");
            return -1;
        }

        int64_t const maxBatchSize = hiddenMax.d[0];
        int64_t const maxSeqLen = hiddenMax.d[1];
        int64_t const maxTokens = maxBatchSize * maxSeqLen;
        if (maxTokens > std::numeric_limits<int32_t>::max() / mTopK)
        {
            LOG_ERROR("Nvfp4A16MoePlugin: optimization profile routed slots overflow int32");
            return -1;
        }

        int64_t requiredRows = 0;
        int32_t const blockSize = getMoeBlockSize(maxSeqLen);
        if (!getConservativePaddedRows(maxTokens, mTopK, mNumExperts, blockSize, requiredRows))
        {
            LOG_ERROR("Nvfp4A16MoePlugin: optimization profile padded-row count overflows int64");
            return -1;
        }
        if (mMaxRoutedRows == 0)
        {
            // Auto: size the routing/Marlin workspace for the profile maximum.
            if (requiredRows > std::numeric_limits<int32_t>::max())
            {
                LOG_ERROR("Nvfp4A16MoePlugin: auto max_routed_rows overflows int32");
                return -1;
            }
            mMaxRoutedRows = static_cast<int32_t>(requiredRows);
        }
        else if (requiredRows > mMaxRoutedRows)
        {
            LOG_ERROR("Nvfp4A16MoePlugin: max_routed_rows=%d is insufficient for the profile; requires at least %lld",
                mMaxRoutedRows, static_cast<long long>(requiredRows));
            return -1;
        }

        return 0;
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("Nvfp4A16MoePlugin configurePlugin failed: %s", e.what());
        return -1;
    }
}

size_t Nvfp4A16MoePlugin::getWorkspaceSize(DynamicPluginTensorDesc const* inputs, int32_t nbInputs,
    DynamicPluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept
{
    if (inputs == nullptr || outputs == nullptr || nbInputs != kNbPluginInputs || nbOutputs != 1)
    {
        LOG_ERROR("Nvfp4A16MoePlugin: getWorkspaceSize expected %d inputs and 1 output", kNbPluginInputs);
        return 0;
    }
    if (!hasCoherentActivationDataTypes(inputs[kInHiddenStates].desc, inputs[kInFc1GlobalScales].desc,
            inputs[kInFc2GlobalScales].desc, outputs[0].desc))
    {
        LOG_ERROR("Nvfp4A16MoePlugin: workspace tensors require one coherent FP16/BF16 type");
        return 0;
    }
    Dims const& hiddenMax = inputs[kInHiddenStates].max;
    if (hiddenMax.nbDims != 3 || hiddenMax.d[0] <= 0 || hiddenMax.d[1] <= 0)
    {
        LOG_ERROR("Nvfp4A16MoePlugin: invalid profile max shape while computing workspace");
        return 0;
    }
    int64_t const maxBatchSize = hiddenMax.d[0];
    int64_t const maxSeqLen = hiddenMax.d[1];
    if (maxBatchSize > std::numeric_limits<int64_t>::max() / maxSeqLen)
    {
        LOG_ERROR("Nvfp4A16MoePlugin: profile max batch*sequence overflows int64");
        return 0;
    }
    int64_t const maxTokens = maxBatchSize * maxSeqLen;
    if (maxTokens > std::numeric_limits<int32_t>::max() / mTopK)
    {
        LOG_ERROR("Nvfp4A16MoePlugin: profile token count is too large for Marlin indexing");
        return 0;
    }
    int32_t effectiveMaxRoutedRows = mMaxRoutedRows;
    if (effectiveMaxRoutedRows <= 0)
    {
        // configurePlugin normally resolves auto max_routed_rows; recompute here for robustness.
        int64_t requiredRows = 0;
        int32_t const blockSize = getMoeBlockSize(maxSeqLen);
        if (!getConservativePaddedRows(maxTokens, mTopK, mNumExperts, blockSize, requiredRows)
            || requiredRows > std::numeric_limits<int32_t>::max())
        {
            LOG_ERROR("Nvfp4A16MoePlugin: could not resolve auto max_routed_rows for workspace sizing");
            return 0;
        }
        effectiveMaxRoutedRows = static_cast<int32_t>(requiredRows);
    }
    return computeWorkspaceSize(static_cast<int32_t>(maxTokens), effectiveMaxRoutedRows, mNumExperts, mTopK,
        mHiddenSize, mMoeInterSize, mActivationType, mRoutingMode, inputs[kInHiddenStates].desc.type);
}

int32_t Nvfp4A16MoePlugin::enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc,
    void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    try
    {
        using namespace trt_edgellm::kernel;
        using namespace trt_edgellm::rt;

        if (inputDesc == nullptr || outputDesc == nullptr || inputs == nullptr || outputs == nullptr
            || workspace == nullptr || outputs[0] == nullptr)
        {
            LOG_ERROR("Nvfp4A16MoePlugin: enqueue received a null descriptor or buffer");
            return -1;
        }
        for (int32_t i = 0; i < kNbPluginInputs; ++i)
        {
            if (inputs[i] == nullptr || !validateTensorDesc(i, inputDesc[i]))
            {
                LOG_ERROR("Nvfp4A16MoePlugin: invalid runtime input at position %d", i);
                return -1;
            }
        }
        if (!validateTensorDesc(kOutOutput, outputDesc[0]))
        {
            LOG_ERROR("Nvfp4A16MoePlugin: invalid runtime output descriptor");
            return -1;
        }
        if (!hasCoherentActivationDataTypes(inputDesc[kInHiddenStates], inputDesc[kInFc1GlobalScales],
                inputDesc[kInFc2GlobalScales], outputDesc[0]))
        {
            LOG_ERROR("Nvfp4A16MoePlugin: runtime activations, global scales, and output have inconsistent types");
            return -1;
        }
        if (reinterpret_cast<uintptr_t>(inputs[kInFc1QWeights]) % alignof(int32_t) != 0
            || reinterpret_cast<uintptr_t>(inputs[kInFc2QWeights]) % alignof(int32_t) != 0)
        {
            LOG_ERROR("Nvfp4A16MoePlugin: packed weight buffers must be aligned for an INT32 view");
            return -1;
        }

        Dims const& hiddenDims = inputDesc[kInHiddenStates].dims;
        int64_t const batchSize = hiddenDims.d[0];
        int64_t const seqLen = hiddenDims.d[1];
        if (batchSize <= 0 || seqLen <= 0 || batchSize > std::numeric_limits<int64_t>::max() / seqLen)
        {
            LOG_ERROR("Nvfp4A16MoePlugin: runtime batch and sequence dimensions must be positive");
            return -1;
        }
        int64_t const numTokens64 = batchSize * seqLen;
        if (numTokens64 > std::numeric_limits<int32_t>::max() / mTopK)
        {
            LOG_ERROR("Nvfp4A16MoePlugin: runtime routed slots overflow int32");
            return -1;
        }
        int32_t const numTokens = static_cast<int32_t>(numTokens64);
        if (inputDesc[kInRouterLogits].dims.d[0] != numTokens)
        {
            LOG_ERROR("Nvfp4A16MoePlugin: router_logits rows must equal batch*sequence");
            return -1;
        }
        if (outputDesc[0].dims.d[0] != batchSize || outputDesc[0].dims.d[1] != seqLen)
        {
            LOG_ERROR("Nvfp4A16MoePlugin: output batch and sequence dimensions must match hidden_states");
            return -1;
        }

        int32_t const moeBlockSize = getMoeBlockSize(seqLen);
        int64_t requiredRows = 0;
        if (!getConservativePaddedRows(numTokens, mTopK, mNumExperts, moeBlockSize, requiredRows))
        {
            LOG_ERROR("Nvfp4A16MoePlugin: runtime padded-row count overflows int64");
            return -1;
        }
        if (requiredRows > mMaxRoutedRows)
        {
            LOG_ERROR("Nvfp4A16MoePlugin: runtime routed-row capacity requires %lld rows but max_routed_rows is %d",
                static_cast<long long>(requiredRows), mMaxRoutedRows);
            return -1;
        }

        int32_t const totalSlots = numTokens * mTopK;
        int32_t const fc1OutDim = getFc1OutDim(mMoeInterSize, mActivationType);
        int32_t const maxPaddedBlocks = divUp(mMaxRoutedRows, moeBlockSize);
        DataType const activationDataType = inputDesc[kInHiddenStates].type;

        int32_t device = 0;
        int32_t numSms = 0;
        CUDA_CHECK(cudaGetDevice(&device));
        CUDA_CHECK(cudaDeviceGetAttribute(&numSms, cudaDevAttrMultiProcessorCount, device));

        std::byte* workspacePtr = static_cast<std::byte*>(workspace);
        float* topkWeights = static_cast<float*>(
            assignTensorFromWorkspace(workspacePtr, {numTokens, mTopK}, DataType::kFLOAT).rawPointer());
        int32_t* topkIndices = static_cast<int32_t*>(
            assignTensorFromWorkspace(workspacePtr, {numTokens, mTopK}, DataType::kINT32).rawPointer());
        size_t const softmaxWorkspaceBytes
            = mRoutingMode == kRoutingSoftmaxTopk ? getMoeTopkSoftmaxWorkspaceSize(numTokens, mNumExperts) : 0;
        void* softmaxWorkspace = nullptr;
        if (softmaxWorkspaceBytes > 0)
        {
            softmaxWorkspace = assignTensorFromWorkspace(
                workspacePtr, {static_cast<int64_t>(softmaxWorkspaceBytes)}, DataType::kINT8)
                                   .rawPointer();
        }

        int32_t* sortedTokenIds = static_cast<int32_t*>(
            assignTensorFromWorkspace(workspacePtr, {mMaxRoutedRows}, DataType::kINT32).rawPointer());
        int32_t* expertIds = static_cast<int32_t*>(
            assignTensorFromWorkspace(workspacePtr, {maxPaddedBlocks}, DataType::kINT32).rawPointer());
        int32_t* numTokensPostPadded
            = static_cast<int32_t*>(assignTensorFromWorkspace(workspacePtr, {1}, DataType::kINT32).rawPointer());
        float* topkWeightsFlat = static_cast<float*>(
            assignTensorFromWorkspace(workspacePtr, {mMaxRoutedRows}, DataType::kFLOAT).rawPointer());
        int32_t* paddedCounts = static_cast<int32_t*>(
            assignTensorFromWorkspace(workspacePtr, {mNumExperts}, DataType::kINT32).rawPointer());
        int32_t* paddedOffsets = static_cast<int32_t*>(
            assignTensorFromWorkspace(workspacePtr, {mNumExperts}, DataType::kINT32).rawPointer());
        int32_t* slotsByExpert = static_cast<int32_t*>(
            assignTensorFromWorkspace(workspacePtr, {mNumExperts, totalSlots}, DataType::kINT32).rawPointer());
        int32_t* slotsPerExpert = static_cast<int32_t*>(
            assignTensorFromWorkspace(workspacePtr, {mNumExperts}, DataType::kINT32).rawPointer());

        void* fc1Output
            = assignTensorFromWorkspace(workspacePtr, {totalSlots, fc1OutDim}, activationDataType).rawPointer();
        void* activationOutput
            = assignTensorFromWorkspace(workspacePtr, {totalSlots, mMoeInterSize}, activationDataType).rawPointer();
        void* fc2Output
            = assignTensorFromWorkspace(workspacePtr, {totalSlots, mHiddenSize}, activationDataType).rawPointer();

        int64_t const marlinWorkspaceElements
            = std::max(getMoeMarlinWorkspaceSize(mMaxRoutedRows, fc1OutDim, moeBlockSize, numSms),
                getMoeMarlinWorkspaceSize(mMaxRoutedRows, mHiddenSize, moeBlockSize, numSms));
        int32_t* marlinWorkspace = static_cast<int32_t*>(
            assignTensorFromWorkspace(workspacePtr, {marlinWorkspaceElements}, DataType::kINT32).rawPointer());
        float* topkWeightsPadded = static_cast<float*>(
            assignTensorFromWorkspace(workspacePtr, {mMaxRoutedRows}, DataType::kFLOAT).rawPointer());

        Tensor routerLogits(
            const_cast<void*>(inputs[kInRouterLogits]), {numTokens, mNumExperts}, DeviceType::kGPU, DataType::kFLOAT);
        Tensor topkWeightsTensor(topkWeights, {numTokens, mTopK}, DeviceType::kGPU, DataType::kFLOAT);
        Tensor topkIndicesTensor(topkIndices, {numTokens, mTopK}, DeviceType::kGPU, DataType::kINT32);
        Tensor expertScoreBias(
            const_cast<void*>(inputs[kInExpertScoreBias]), {mNumExperts}, DeviceType::kGPU, DataType::kFLOAT);
        OptionalInputTensor expertScoreBiasOpt = expertScoreBias;
        {
            NVTX_SCOPED_RANGE(nvtx_routing, "Nvfp4A16MoePlugin::routing", nvtx_colors::ORANGE);
            if (mRoutingMode == kRoutingSigmoidGroupTopk)
            {
                moeSigmoidGroupTopk(routerLogits, topkWeightsTensor, topkIndicesTensor, mTopK, mNGroup, mTopkGroup,
                    mNormTopkProb != 0, mRoutedScalingFactor, stream, expertScoreBiasOpt);
            }
            else
            {
                moeTopkSoftmax(routerLogits, topkWeightsTensor, topkIndicesTensor, mTopK, softmaxWorkspace,
                    softmaxWorkspaceBytes, stream, mNormTopkProb != 0, 0.0F, expertScoreBiasOpt);
            }
            CUDA_CHECK(cudaGetLastError());

            CUDA_CHECK(cudaMemsetAsync(slotsPerExpert, 0, mNumExperts * sizeof(int32_t), stream));
            launchCountSlotsPerExpertKernel(topkIndices, slotsPerExpert, numTokens, mTopK, mNumExperts, stream);
            CUDA_CHECK(cudaGetLastError());
            launchComputePaddedOffsetsKernel(
                slotsPerExpert, paddedCounts, paddedOffsets, numTokensPostPadded, mNumExperts, moeBlockSize, stream);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemsetAsync(slotsPerExpert, 0, mNumExperts * sizeof(int32_t), stream));
            launchBuildSlotListsKernel(
                topkIndices, slotsByExpert, slotsPerExpert, numTokens, mTopK, mNumExperts, stream);
            CUDA_CHECK(cudaGetLastError());
            launchBuildMarlinIndicesKernel(slotsByExpert, slotsPerExpert, paddedCounts, paddedOffsets, topkWeights,
                sortedTokenIds, topkWeightsFlat, expertIds, numTokens, mTopK, mNumExperts, moeBlockSize, stream);
            CUDA_CHECK(cudaGetLastError());
        }

        Tensor hiddenStates(
            const_cast<void*>(inputs[kInHiddenStates]), {numTokens, mHiddenSize}, DeviceType::kGPU, activationDataType);
        Tensor fc1QWeights(const_cast<void*>(inputs[kInFc1QWeights]),
            {mNumExperts, mHiddenSize / kNvfp4GroupSize, 2 * fc1OutDim}, DeviceType::kGPU, DataType::kINT32);
        Tensor fc1BlockScales(const_cast<void*>(inputs[kInFc1BlockScales]),
            {mNumExperts, mHiddenSize / kNvfp4GroupSize, fc1OutDim}, DeviceType::kGPU, DataType::kINT8);
        Tensor fc1GlobalScales(
            const_cast<void*>(inputs[kInFc1GlobalScales]), {mNumExperts}, DeviceType::kGPU, activationDataType);
        Tensor sortedTokenIdsTensor(sortedTokenIds, {mMaxRoutedRows}, DeviceType::kGPU, DataType::kINT32);
        Tensor expertIdsTensor(expertIds, {maxPaddedBlocks}, DeviceType::kGPU, DataType::kINT32);
        Tensor numTokensPostPaddedTensor(numTokensPostPadded, {1}, DeviceType::kGPU, DataType::kINT32);
        Tensor topkWeightsFlatTensor(topkWeightsFlat, {mMaxRoutedRows}, DeviceType::kGPU, DataType::kFLOAT);
        Tensor marlinWorkspaceTensor(marlinWorkspace, {marlinWorkspaceElements}, DeviceType::kGPU, DataType::kINT32);
        Tensor fc1OutputTensor(fc1Output, {totalSlots, fc1OutDim}, DeviceType::kGPU, activationDataType);

        {
            NVTX_SCOPED_RANGE(nvtx_fc1, "Nvfp4A16MoePlugin::fc1", nvtx_colors::BLUE);
            // FC1 does not apply routing weights; topkWeightsFlat is passed only to preserve the shared Marlin call
            // signature.
            moeNvfp4A16MarlinGemm(hiddenStates, fc1OutputTensor, fc1QWeights, fc1BlockScales, fc1GlobalScales,
                sortedTokenIdsTensor, expertIdsTensor, numTokensPostPaddedTensor, topkWeightsFlatTensor,
                marlinWorkspaceTensor, moeBlockSize, mTopK, false, stream);
            CUDA_CHECK(cudaGetLastError());
        }

        Tensor activationOutputTensor(
            activationOutput, {totalSlots, mMoeInterSize}, DeviceType::kGPU, activationDataType);
        {
            NVTX_SCOPED_RANGE(nvtx_activation, "Nvfp4A16MoePlugin::activation", nvtx_colors::GREEN);
            moeActivation(fc1OutputTensor, activationOutputTensor, totalSlots, mMoeInterSize, mActivationType, stream);
            CUDA_CHECK(cudaGetLastError());
        }

        Tensor fc2QWeights(const_cast<void*>(inputs[kInFc2QWeights]),
            {mNumExperts, mMoeInterSize / kNvfp4GroupSize, 2 * mHiddenSize}, DeviceType::kGPU, DataType::kINT32);
        Tensor fc2BlockScales(const_cast<void*>(inputs[kInFc2BlockScales]),
            {mNumExperts, mMoeInterSize / kNvfp4GroupSize, mHiddenSize}, DeviceType::kGPU, DataType::kINT8);
        Tensor fc2GlobalScales(
            const_cast<void*>(inputs[kInFc2GlobalScales]), {mNumExperts}, DeviceType::kGPU, activationDataType);
        Tensor fc2OutputTensor(fc2Output, {totalSlots, mHiddenSize}, DeviceType::kGPU, activationDataType);

        // FC2 applies routing weights by the original slot ID from sortedTokenIds, so keep weights in [token, topK]
        // order.
        CUDA_CHECK(cudaMemsetAsync(topkWeightsPadded, 0, mMaxRoutedRows * sizeof(float), stream));
        CUDA_CHECK(cudaMemcpyAsync(
            topkWeightsPadded, topkWeights, totalSlots * sizeof(float), cudaMemcpyDeviceToDevice, stream));
        Tensor topkWeightsPaddedTensor(topkWeightsPadded, {mMaxRoutedRows}, DeviceType::kGPU, DataType::kFLOAT);

        {
            NVTX_SCOPED_RANGE(nvtx_fc2, "Nvfp4A16MoePlugin::fc2", nvtx_colors::PURPLE);
            moeNvfp4A16MarlinGemm(activationOutputTensor, fc2OutputTensor, fc2QWeights, fc2BlockScales, fc2GlobalScales,
                sortedTokenIdsTensor, expertIdsTensor, numTokensPostPaddedTensor, topkWeightsPaddedTensor,
                marlinWorkspaceTensor, moeBlockSize, 1, true, stream);
            CUDA_CHECK(cudaGetLastError());
        }

        {
            NVTX_SCOPED_RANGE(nvtx_aggregation, "Nvfp4A16MoePlugin::aggregation", nvtx_colors::MAGENTA);
            if (activationDataType == DataType::kHALF)
            {
                launchAggregateSlotOutputsKernel(fc2Output, outputs[0], numTokens, mTopK, mHiddenSize, stream);
            }
            else
            {
                launchAggregateSlotOutputsBf16Kernel(fc2Output, outputs[0], numTokens, mTopK, mHiddenSize, stream);
            }
            CUDA_CHECK(cudaGetLastError());
        }
        return 0;
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("Nvfp4A16MoePlugin enqueue failed: %s", e.what());
        return -1;
    }
}

int32_t Nvfp4A16MoePlugin::onShapeChange(
    PluginTensorDesc const* in, int32_t nbInputs, PluginTensorDesc const* out, int32_t nbOutputs) noexcept
{
    if (in == nullptr || out == nullptr || nbInputs != kNbPluginInputs || nbOutputs != 1)
    {
        LOG_ERROR("Nvfp4A16MoePlugin: onShapeChange expected %d inputs and 1 output", kNbPluginInputs);
        return -1;
    }
    for (int32_t pos = 0; pos < kNbPluginInputs; ++pos)
    {
        if (!validateTensorDesc(pos, in[pos]))
        {
            LOG_ERROR("Nvfp4A16MoePlugin: invalid shape-change descriptor at input %d", pos);
            return -1;
        }
    }
    if (!validateTensorDesc(kOutOutput, out[0]))
    {
        LOG_ERROR("Nvfp4A16MoePlugin: invalid shape-change output descriptor");
        return -1;
    }
    if (!hasCoherentActivationDataTypes(in[kInHiddenStates], in[kInFc1GlobalScales], in[kInFc2GlobalScales], out[0]))
    {
        LOG_ERROR("Nvfp4A16MoePlugin: shape-change activations, global scales, and output have inconsistent types");
        return -1;
    }

    int64_t const batchSize = in[kInHiddenStates].dims.d[0];
    int64_t const seqLen = in[kInHiddenStates].dims.d[1];
    if (batchSize <= 0 || seqLen <= 0 || batchSize > std::numeric_limits<int64_t>::max() / seqLen)
    {
        LOG_ERROR("Nvfp4A16MoePlugin: invalid runtime batch or sequence dimension");
        return -1;
    }
    int64_t const numTokens = batchSize * seqLen;
    if (in[kInRouterLogits].dims.d[0] != numTokens || out[0].dims.d[0] != batchSize || out[0].dims.d[1] != seqLen)
    {
        LOG_ERROR("Nvfp4A16MoePlugin: runtime router/output shapes do not match hidden_states");
        return -1;
    }

    int64_t requiredRows = 0;
    int32_t const blockSize = getMoeBlockSize(seqLen);
    if (!getConservativePaddedRows(numTokens, mTopK, mNumExperts, blockSize, requiredRows))
    {
        LOG_ERROR("Nvfp4A16MoePlugin: runtime padded-row count overflows int64");
        return -1;
    }
    if (requiredRows > mMaxRoutedRows)
    {
        LOG_ERROR("Nvfp4A16MoePlugin: runtime shape requires %lld routed rows but max_routed_rows is %d",
            static_cast<long long>(requiredRows), mMaxRoutedRows);
        return -1;
    }
    return 0;
}

IPluginV3* Nvfp4A16MoePlugin::attachToContext(IPluginResourceContext* context) noexcept
{
    (void) context;
    return clone();
}

PluginFieldCollection const* Nvfp4A16MoePlugin::getFieldsToSerialize() noexcept
{
    try
    {
        mDataToSerialize.clear();
        mDataToSerialize.emplace_back("num_experts", &mNumExperts, PluginFieldType::kINT32, 1);
        mDataToSerialize.emplace_back("top_k", &mTopK, PluginFieldType::kINT32, 1);
        mDataToSerialize.emplace_back("hidden_size", &mHiddenSize, PluginFieldType::kINT32, 1);
        mDataToSerialize.emplace_back("moe_inter_size", &mMoeInterSize, PluginFieldType::kINT32, 1);
        mDataToSerialize.emplace_back("activation_type", &mActivationType, PluginFieldType::kINT32, 1);
        mDataToSerialize.emplace_back("n_group", &mNGroup, PluginFieldType::kINT32, 1);
        mDataToSerialize.emplace_back("topk_group", &mTopkGroup, PluginFieldType::kINT32, 1);
        mDataToSerialize.emplace_back("norm_topk_prob", &mNormTopkProb, PluginFieldType::kINT32, 1);
        mDataToSerialize.emplace_back("routed_scaling_factor", &mRoutedScalingFactor, PluginFieldType::kFLOAT32, 1);
        mDataToSerialize.emplace_back("routing_mode", &mRoutingMode, PluginFieldType::kINT32, 1);
        mDataToSerialize.emplace_back("max_routed_rows", &mMaxRoutedRows, PluginFieldType::kINT32, 1);
        mFCToSerialize.nbFields = static_cast<int32_t>(mDataToSerialize.size());
        mFCToSerialize.fields = mDataToSerialize.data();
        return &mFCToSerialize;
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("Failed to serialize Nvfp4A16MoePlugin fields: %s", e.what());
        return nullptr;
    }
}

Nvfp4A16MoePluginCreator::Nvfp4A16MoePluginCreator()
{
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);

    mPluginAttributes.clear();
    mPluginAttributes.emplace_back("num_experts", nullptr, PluginFieldType::kINT32, 1);
    mPluginAttributes.emplace_back("top_k", nullptr, PluginFieldType::kINT32, 1);
    mPluginAttributes.emplace_back("hidden_size", nullptr, PluginFieldType::kINT32, 1);
    mPluginAttributes.emplace_back("moe_inter_size", nullptr, PluginFieldType::kINT32, 1);
    mPluginAttributes.emplace_back("activation_type", nullptr, PluginFieldType::kINT32, 1);
    mPluginAttributes.emplace_back("n_group", nullptr, PluginFieldType::kINT32, 1);
    mPluginAttributes.emplace_back("topk_group", nullptr, PluginFieldType::kINT32, 1);
    mPluginAttributes.emplace_back("norm_topk_prob", nullptr, PluginFieldType::kINT32, 1);
    mPluginAttributes.emplace_back("routed_scaling_factor", nullptr, PluginFieldType::kFLOAT32, 1);
    mPluginAttributes.emplace_back("routing_mode", nullptr, PluginFieldType::kINT32, 1);
    mPluginAttributes.emplace_back("max_routed_rows", nullptr, PluginFieldType::kINT32, 1);

    mFieldCollection.nbFields = static_cast<int32_t>(mPluginAttributes.size());
    mFieldCollection.fields = mPluginAttributes.data();
}

char const* Nvfp4A16MoePluginCreator::getPluginName() const noexcept
{
    return kPluginName;
}

char const* Nvfp4A16MoePluginCreator::getPluginVersion() const noexcept
{
    return kPluginVersion;
}

PluginFieldCollection const* Nvfp4A16MoePluginCreator::getFieldNames() noexcept
{
    return &mFieldCollection;
}

char const* Nvfp4A16MoePluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

void Nvfp4A16MoePluginCreator::setPluginNamespace(char const* pluginNamespace) noexcept
{
    mNamespace = pluginNamespace == nullptr ? "" : pluginNamespace;
}

IPluginV3* Nvfp4A16MoePluginCreator::createPlugin(
    char const* name, PluginFieldCollection const* fc, TensorRTPhase phase) noexcept
{
    (void) phase;
    try
    {
        auto* plugin = new Nvfp4A16MoePlugin(name == nullptr ? kPluginName : name, fc);
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("Failed to create Nvfp4A16MoePlugin: %s", e.what());
        return nullptr;
    }
}

} // namespace plugins
} // namespace trt_edgellm
