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

#include "nvfp4A16BlackwellMoePlugin.h"

#include "common/cudaUtils.h"
#include "common/logger.h"
#include "kernels/moe/nvfp4A16BlackwellMoe/nvfp4A16BlackwellMoeDispatchPolicy.h"
#include "kernels/moe/nvfp4A16BlackwellMoe/nvfp4A16BlackwellMoeRunner.h"
#include "kernels/nvfp4A16BlackwellSupport.h"
#include "profiling/nvtx_wrapper.h"

#include <NvInfer.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
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

namespace moe = kernel::nvfp4_a16_blackwell_moe;

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

constexpr char const* kPluginName{"Nvfp4A16BlackwellMoePlugin"};
constexpr char const* kPluginVersion{"1"};

constexpr int32_t kActivationRelu2{4};
constexpr int32_t kRoutingSigmoidGroupTopk{1};
constexpr int32_t kLayoutBlackwellMoeN128K64V1{1};
constexpr int32_t kSupportedNumExperts[]{128, 256, 512};
constexpr int32_t kMaxTopK{32};
constexpr int32_t kNTile{nvfp4_a16_blackwell::kNTile};
constexpr int32_t kKTile{nvfp4_a16_blackwell::kKTile};
constexpr int32_t kPackedBytesPerRowTile{kKTile / 2};
constexpr int32_t kScaleBytesPerRowTile{kKTile / 16};

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
constexpr int32_t kFieldLayout{11};
constexpr int32_t kFieldBackend{12};
constexpr int32_t kNbPluginFields{13};
//! Optional (runtime-phase) field carrying the serialized NVRTC bundle of the
//! layer's CUDA-core kernels; never part of the ONNX node.
constexpr char const* kFieldJitBundle{"moe_jit_bundle"};

int32_t padTo(int32_t const value, int32_t const multiple) noexcept
{
    return (value + multiple - 1) / multiple * multiple;
}

bool getTokenCount(Dims const& hidden, int64_t& numTokens) noexcept
{
    if (hidden.nbDims == 2 && hidden.d[0] > 0)
    {
        numTokens = hidden.d[0];
        return true;
    }
    return false;
}

bool shapesMatch(Dims const& lhs, Dims const& rhs) noexcept
{
    if (lhs.nbDims != rhs.nbDims)
    {
        return false;
    }
    return std::equal(lhs.d, lhs.d + lhs.nbDims, rhs.d);
}

} // namespace

PluginFieldCollection Nvfp4A16BlackwellMoePluginCreator::mFieldCollection{};
std::vector<PluginField> Nvfp4A16BlackwellMoePluginCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(Nvfp4A16BlackwellMoePluginCreator);

Nvfp4A16BlackwellMoePlugin::Nvfp4A16BlackwellMoePlugin(std::string const& name, int32_t numExperts, int32_t topK,
    int32_t hiddenSize, int32_t moeInterSize, int32_t activationType, int32_t nGroup, int32_t topkGroup,
    int32_t normTopkProb, float routedScalingFactor, int32_t routingMode, int32_t maxRoutedRows, int32_t layout,
    int32_t backend)
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
    , mLayout(layout)
    , mBackend(backend)
{
    validateAttributes();
}

Nvfp4A16BlackwellMoePlugin::Nvfp4A16BlackwellMoePlugin(std::string const& name, PluginFieldCollection const* fc)
    : mLayerName(name)
{
    if (fc == nullptr || fc->fields == nullptr || fc->nbFields <= 0)
    {
        throw std::invalid_argument("Nvfp4A16BlackwellMoePlugin: plugin field collection must not be empty");
    }

    std::array<bool, kNbPluginFields> fieldsSeen{};
    auto readIntField = [&fieldsSeen](PluginField const& field, int32_t fieldIndex) {
        if (field.data == nullptr || field.type != PluginFieldType::kINT32 || field.length != 1)
        {
            throw std::invalid_argument("Nvfp4A16BlackwellMoePlugin: integer attributes must be scalar INT32 fields");
        }
        if (fieldsSeen[fieldIndex])
        {
            throw std::invalid_argument("Nvfp4A16BlackwellMoePlugin: duplicate plugin attribute");
        }
        fieldsSeen[fieldIndex] = true;
        return *static_cast<int32_t const*>(field.data);
    };
    auto readFloatField = [&fieldsSeen](PluginField const& field, int32_t fieldIndex) {
        if (field.data == nullptr || field.type != PluginFieldType::kFLOAT32 || field.length != 1)
        {
            throw std::invalid_argument(
                "Nvfp4A16BlackwellMoePlugin: routed_scaling_factor must be a scalar FLOAT32 field");
        }
        if (fieldsSeen[fieldIndex])
        {
            throw std::invalid_argument("Nvfp4A16BlackwellMoePlugin: duplicate plugin attribute");
        }
        fieldsSeen[fieldIndex] = true;
        return *static_cast<float const*>(field.data);
    };

    for (int32_t i = 0; i < fc->nbFields; ++i)
    {
        PluginField const& field = fc->fields[i];
        if (field.name == nullptr)
        {
            throw std::invalid_argument("Nvfp4A16BlackwellMoePlugin: plugin attribute name must not be null");
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
        else if (fieldName == "layout")
        {
            mLayout = readIntField(field, kFieldLayout);
        }
        else if (fieldName == "backend")
        {
            mBackend = readIntField(field, kFieldBackend);
        }
        else if (fieldName == kFieldJitBundle)
        {
            if (!mJitBundle.empty())
            {
                throw std::invalid_argument("Nvfp4A16BlackwellMoePlugin: duplicate moe_jit_bundle attribute");
            }
            if (field.type != PluginFieldType::kCHAR || field.length <= 0 || field.data == nullptr)
            {
                throw std::invalid_argument("Nvfp4A16BlackwellMoePlugin: moe_jit_bundle must be a nonempty CHAR field");
            }
            auto const* bytes = static_cast<uint8_t const*>(field.data);
            mJitBundle.assign(bytes, bytes + field.length);
        }
        else
        {
            throw std::invalid_argument("Nvfp4A16BlackwellMoePlugin: unknown plugin attribute " + fieldName);
        }
    }

    if (std::any_of(fieldsSeen.begin(), fieldsSeen.end(), [](bool seen) { return !seen; }))
    {
        throw std::invalid_argument("Nvfp4A16BlackwellMoePlugin: all 13 plugin attributes are required");
    }
    if (!mJitBundle.empty())
    {
        mJitKernel = deserializeNvfp4A16BlackwellMoeJitKernel(mJitBundle.data(), mJitBundle.size());
    }
    validateAttributes();
}

Nvfp4A16BlackwellMoePlugin::~Nvfp4A16BlackwellMoePlugin() noexcept = default;

int32_t Nvfp4A16BlackwellMoePlugin::interSizePadded() const noexcept
{
    return padTo(mMoeInterSize, kNTile);
}

void Nvfp4A16BlackwellMoePlugin::validateAttributes() const
{
    if (mLayout != kLayoutBlackwellMoeN128K64V1)
    {
        throw std::invalid_argument("Nvfp4A16BlackwellMoePlugin: layout must be BLACKWELL_MOE_N128_K64_V1 (1)");
    }
    if (mBackend != static_cast<int32_t>(moe::Backend::kAuto) && mBackend != static_cast<int32_t>(moe::Backend::kDecode)
        && mBackend != static_cast<int32_t>(moe::Backend::kPrefill))
    {
        throw std::invalid_argument("Nvfp4A16BlackwellMoePlugin: backend must be 0 (auto), 1 (decode) or 2 (prefill)");
    }
    if (std::find(std::begin(kSupportedNumExperts), std::end(kSupportedNumExperts), mNumExperts)
        == std::end(kSupportedNumExperts))
    {
        throw std::invalid_argument("Nvfp4A16BlackwellMoePlugin: num_experts must be one of {128, 256, 512}");
    }
    if (mTopK <= 0 || mTopK > kMaxTopK || mTopK > mNumExperts)
    {
        throw std::invalid_argument("Nvfp4A16BlackwellMoePlugin: top_k must be in [1, 32]");
    }
    if (mHiddenSize <= 0 || mHiddenSize % kNTile != 0)
    {
        throw std::invalid_argument("Nvfp4A16BlackwellMoePlugin: hidden_size must be positive and divisible by 128");
    }
    if (mMoeInterSize <= 0 || mMoeInterSize % kKTile != 0)
    {
        throw std::invalid_argument(
            "Nvfp4A16BlackwellMoePlugin: moe_inter_size must be positive and divisible by 64 (FC2 K tile)");
    }
    if (mActivationType != kActivationRelu2)
    {
        throw std::invalid_argument("Nvfp4A16BlackwellMoePlugin: only activation_type 4 (ReLU2) is supported");
    }
    if (mRoutingMode != kRoutingSigmoidGroupTopk)
    {
        throw std::invalid_argument(
            "Nvfp4A16BlackwellMoePlugin: only routing_mode 1 (sigmoid group top-k) is supported");
    }
    if (mNormTopkProb != 0 && mNormTopkProb != 1)
    {
        throw std::invalid_argument("Nvfp4A16BlackwellMoePlugin: norm_topk_prob must be 0 or 1");
    }
    if (mMaxRoutedRows < 0)
    {
        throw std::invalid_argument("Nvfp4A16BlackwellMoePlugin: max_routed_rows must be non-negative (0 == auto)");
    }
    if (!std::isfinite(mRoutedScalingFactor) || mRoutedScalingFactor <= 0.0F)
    {
        throw std::invalid_argument("Nvfp4A16BlackwellMoePlugin: routed_scaling_factor must be finite and positive");
    }
    if (mNGroup <= 0 || mNumExperts % mNGroup != 0 || mNumExperts / mNGroup < 2)
    {
        throw std::invalid_argument(
            "Nvfp4A16BlackwellMoePlugin: n_group must divide num_experts with at least two experts per group");
    }
    if (mTopkGroup <= 0 || mTopkGroup > mNGroup)
    {
        throw std::invalid_argument("Nvfp4A16BlackwellMoePlugin: topk_group must be in [1, n_group]");
    }
    if (mTopK > mTopkGroup * (mNumExperts / mNGroup))
    {
        throw std::invalid_argument("Nvfp4A16BlackwellMoePlugin: selected expert groups cannot contain top_k experts");
    }
}

IPluginCapability* Nvfp4A16BlackwellMoePlugin::getCapabilityInterface(PluginCapabilityType type) noexcept
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

IPluginV3* Nvfp4A16BlackwellMoePlugin::clone() noexcept
{
    try
    {
        auto plugin = std::make_unique<Nvfp4A16BlackwellMoePlugin>(mLayerName, mNumExperts, mTopK, mHiddenSize,
            mMoeInterSize, mActivationType, mNGroup, mTopkGroup, mNormTopkProb, mRoutedScalingFactor, mRoutingMode,
            mMaxRoutedRows, mLayout, mBackend);
        plugin->mJitKernel = mJitKernel;
        plugin->mJitBundle = mJitBundle;
        if (!plugin->mJitBundle.empty())
        {
            // configurePlugin() compiles and serializes the bundle but does not
            // load it; attachToContext() clones a build-phase or a runtime plugin,
            // and either way the clone loads through the context-keyed registry.
            plugin->loadSerializedJitBundle();
        }
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin.release();
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("Failed to clone Nvfp4A16BlackwellMoePlugin: %s", e.what());
        return nullptr;
    }
}

char const* Nvfp4A16BlackwellMoePlugin::getPluginName() const noexcept
{
    return kPluginName;
}

char const* Nvfp4A16BlackwellMoePlugin::getPluginVersion() const noexcept
{
    return kPluginVersion;
}

char const* Nvfp4A16BlackwellMoePlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

void Nvfp4A16BlackwellMoePlugin::setPluginNamespace(char const* pluginNamespace) noexcept
{
    mNamespace = pluginNamespace == nullptr ? "" : pluginNamespace;
}

int32_t Nvfp4A16BlackwellMoePlugin::getNbOutputs() const noexcept
{
    return 1;
}

int32_t Nvfp4A16BlackwellMoePlugin::getOutputDataTypes(
    DataType* outputTypes, int32_t nbOutputs, DataType const* inputTypes, int32_t nbInputs) const noexcept
{
    if (outputTypes == nullptr || inputTypes == nullptr || nbOutputs != 1 || nbInputs != kNbPluginInputs)
    {
        LOG_ERROR("Nvfp4A16BlackwellMoePlugin: getOutputDataTypes expected %d inputs and 1 output", kNbPluginInputs);
        return -1;
    }
    if (inputTypes[kInHiddenStates] != DataType::kHALF)
    {
        LOG_ERROR("Nvfp4A16BlackwellMoePlugin: hidden_states must be FP16");
        return -1;
    }
    outputTypes[0] = DataType::kHALF;
    return 0;
}

int32_t Nvfp4A16BlackwellMoePlugin::getOutputShapes(DimsExprs const* inputs, int32_t nbInputs,
    DimsExprs const* shapeInputs, int32_t nbShapeInputs, DimsExprs* outputs, int32_t nbOutputs,
    IExprBuilder& exprBuilder) noexcept
{
    if (inputs == nullptr || outputs == nullptr || nbInputs != kNbPluginInputs || nbOutputs != 1)
    {
        LOG_ERROR("Nvfp4A16BlackwellMoePlugin: getOutputShapes expected %d inputs and 1 output", kNbPluginInputs);
        return -1;
    }
    (void) shapeInputs;
    (void) nbShapeInputs;
    if (inputs[kInHiddenStates].nbDims != 2)
    {
        LOG_ERROR("Nvfp4A16BlackwellMoePlugin: hidden_states must have rank 2");
        return -1;
    }
    outputs[0].nbDims = 2;
    outputs[0].d[0] = inputs[kInHiddenStates].d[0];
    outputs[0].d[1] = exprBuilder.constant(mHiddenSize);
    return 0;
}

bool Nvfp4A16BlackwellMoePlugin::validateTensorDesc(int32_t pos, PluginTensorDesc const& desc) const noexcept
{
    if (desc.format != TensorFormat::kLINEAR)
    {
        return false;
    }
    int64_t const fc1NTiles = interSizePadded() / kNTile;
    int64_t const fc1KTiles = mHiddenSize / kKTile;
    int64_t const fc2NTiles = mHiddenSize / kNTile;
    int64_t const fc2KTiles = mMoeInterSize / kKTile;
    auto const weightDims = [&desc](int64_t nTiles, int64_t kTiles, int64_t inner) {
        return desc.type == DataType::kINT8 && desc.dims.nbDims == 5 && desc.dims.d[1] == nTiles
            && desc.dims.d[2] == kTiles && desc.dims.d[3] == kNTile && desc.dims.d[4] == inner;
    };
    switch (pos)
    {
    case kInRouterLogits:
        return desc.type == DataType::kFLOAT && desc.dims.nbDims == 2 && desc.dims.d[1] == mNumExperts;
    case kInHiddenStates: return desc.type == DataType::kHALF && desc.dims.nbDims == 2 && desc.dims.d[1] == mHiddenSize;
    case kInFc1QWeights:
        return desc.dims.d[0] == mNumExperts && weightDims(fc1NTiles, fc1KTiles, kPackedBytesPerRowTile);
    case kInFc1BlockScales:
        return desc.dims.d[0] == mNumExperts && weightDims(fc1NTiles, fc1KTiles, kScaleBytesPerRowTile);
    case kInFc1GlobalScales:
    case kInFc2GlobalScales:
    case kInExpertScoreBias:
        return desc.type == DataType::kFLOAT && desc.dims.nbDims == 1 && desc.dims.d[0] == mNumExperts;
    case kInFc2QWeights:
        return desc.dims.d[0] == mNumExperts && weightDims(fc2NTiles, fc2KTiles, kPackedBytesPerRowTile);
    case kInFc2BlockScales:
        return desc.dims.d[0] == mNumExperts && weightDims(fc2NTiles, fc2KTiles, kScaleBytesPerRowTile);
    case kOutOutput: return desc.type == DataType::kHALF && desc.dims.nbDims == 2 && desc.dims.d[1] == mHiddenSize;
    default: return false;
    }
}

bool Nvfp4A16BlackwellMoePlugin::supportsFormatCombination(
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
    return pos != kOutOutput || shapesMatch(inOut[pos].desc.dims, inOut[kInHiddenStates].desc.dims);
}

namespace
{

//! PDL is on by default for every kernel of this plugin; EDGELLM_ENABLE_PDL=0
//! (read once, before the first enqueue) keeps an explicit production A/B and
//! recovery switch, the same knob as Nvfp4MoePlugin.
bool requestPdl()
{
    static bool const enabled = []() {
        char const* const value = std::getenv("EDGELLM_ENABLE_PDL");
        return value == nullptr || std::strcmp(value, "0") != 0;
    }();
    return enabled;
}

kernel::Nvfp4A16BlackwellMoeParams makeShape(int32_t numTokens, int32_t numExperts, int32_t topK, int32_t hiddenSize,
    int32_t interSize, int32_t interSizePadded, int32_t nGroup, int32_t topkGroup, int32_t normTopkProb,
    float routedScalingFactor, int32_t backend) noexcept
{
    kernel::Nvfp4A16BlackwellMoeParams p{};
    p.dtype = moe::DecodeDtype::kFP16;
    p.backend = static_cast<moe::Backend>(backend);
    p.numTokens = numTokens;
    p.numExperts = numExperts;
    p.topK = topK;
    p.hiddenSize = hiddenSize;
    p.interSize = interSize;
    p.interSizePadded = interSizePadded;
    p.nGroup = nGroup;
    p.topkGroup = topkGroup;
    p.normTopkProb = normTopkProb != 0;
    p.routedScalingFactor = routedScalingFactor;
    return p;
}

} // namespace

int32_t Nvfp4A16BlackwellMoePlugin::configurePlugin(
    DynamicPluginTensorDesc const* in, int32_t nbInputs, DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept
{
    try
    {
        if (in == nullptr || out == nullptr || nbInputs != kNbPluginInputs || nbOutputs != 1)
        {
            LOG_ERROR("Nvfp4A16BlackwellMoePlugin: configurePlugin expected %d inputs and 1 output", kNbPluginInputs);
            return -1;
        }
        for (int32_t pos = 0; pos < kNbPluginInputs; ++pos)
        {
            if (!validateTensorDesc(pos, in[pos].desc))
            {
                LOG_ERROR("Nvfp4A16BlackwellMoePlugin: invalid input descriptor at position %d", pos);
                return -1;
            }
        }
        if (!validateTensorDesc(kOutOutput, out[0].desc))
        {
            LOG_ERROR("Nvfp4A16BlackwellMoePlugin: invalid output descriptor");
            return -1;
        }

        int32_t const smVersion = getSMVersion();
        if (smVersion != nvfp4_a16_blackwell::kTargetSm)
        {
            LOG_ERROR("Nvfp4A16BlackwellMoePlugin: BLACKWELL_MOE_N128_K64_V1 requires SM110, got SM%d", smVersion);
            return -1;
        }

        auto validateProfileEndpoint
            = [this](Dims const& hidden, Dims const& router, char const* endpoint, int64_t& numTokens) {
                  if (!getTokenCount(hidden, numTokens) || hidden.d[hidden.nbDims - 1] != mHiddenSize
                      || router.nbDims != 2 || router.d[0] != numTokens || router.d[1] != mNumExperts)
                  {
                      LOG_ERROR("Nvfp4A16BlackwellMoePlugin: optimization profile %s dimensions are invalid", endpoint);
                      return false;
                  }
                  return true;
              };
        int64_t minTokens{0};
        int64_t optTokens{0};
        int64_t maxTokens{0};
        if (!validateProfileEndpoint(in[kInHiddenStates].min, in[kInRouterLogits].min, "minimum", minTokens)
            || !validateProfileEndpoint(in[kInHiddenStates].opt, in[kInRouterLogits].opt, "optimum", optTokens)
            || !validateProfileEndpoint(in[kInHiddenStates].max, in[kInRouterLogits].max, "maximum", maxTokens))
        {
            return -1;
        }
        Dims const& hiddenMin = in[kInHiddenStates].min;
        Dims const& hiddenMax = in[kInHiddenStates].max;
        if (!shapesMatch(out[0].min, hiddenMin) || !shapesMatch(out[0].opt, in[kInHiddenStates].opt)
            || !shapesMatch(out[0].max, hiddenMax))
        {
            LOG_ERROR("Nvfp4A16BlackwellMoePlugin: output profile range must match hidden_states");
            return -1;
        }
        if (maxTokens > std::numeric_limits<int32_t>::max() / mTopK)
        {
            LOG_ERROR("Nvfp4A16BlackwellMoePlugin: optimization profile routed slots overflow int32");
            return -1;
        }

        kernel::Nvfp4A16BlackwellMoeParams const maxShape
            = makeShape(static_cast<int32_t>(maxTokens), mNumExperts, mTopK, mHiddenSize, mMoeInterSize,
                interSizePadded(), mNGroup, mTopkGroup, mNormTopkProb, mRoutedScalingFactor, mBackend);
        if (!kernel::Nvfp4A16BlackwellMoeRunner::isSupported(smVersion, maxShape))
        {
            LOG_ERROR(
                "Nvfp4A16BlackwellMoePlugin: shape E=%d topK=%d H=%d I=%d maxTokens=%lld is not supported by "
                "the SM110 grouped W4A16 MoE kernels (is the nvfp4_a16_blackwell_moe CuTe DSL group linked?)",
                mNumExperts, mTopK, mHiddenSize, mMoeInterSize, static_cast<long long>(maxTokens));
            return -1;
        }

        int64_t const requiredRows = moe::maxRowsPadded(maxTokens, mTopK, mNumExperts, moe::kLargestTokenTile);
        if (requiredRows > std::numeric_limits<int32_t>::max())
        {
            LOG_ERROR("Nvfp4A16BlackwellMoePlugin: padded routed-row capacity overflows int32");
            return -1;
        }
        if (mMaxRoutedRows == 0)
        {
            mMaxRoutedRows = static_cast<int32_t>(requiredRows);
        }
        else if (requiredRows > mMaxRoutedRows)
        {
            LOG_ERROR("Nvfp4A16BlackwellMoePlugin: max_routed_rows=%d is insufficient for the profile; requires %lld",
                mMaxRoutedRows, static_cast<long long>(requiredRows));
            return -1;
        }
        compileJitBundle();
        return 0;
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("Nvfp4A16BlackwellMoePlugin configurePlugin failed: %s", e.what());
        return -1;
    }
}

size_t Nvfp4A16BlackwellMoePlugin::getWorkspaceSize(DynamicPluginTensorDesc const* inputs, int32_t nbInputs,
    DynamicPluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept
{
    if (inputs == nullptr || outputs == nullptr || nbInputs != kNbPluginInputs || nbOutputs != 1)
    {
        LOG_ERROR("Nvfp4A16BlackwellMoePlugin: getWorkspaceSize expected %d inputs and 1 output", kNbPluginInputs);
        return 0;
    }
    int64_t maxTokens{0};
    auto const& hiddenMax = inputs[kInHiddenStates].max;
    auto const& routerMax = inputs[kInRouterLogits].max;
    if (!getTokenCount(hiddenMax, maxTokens) || hiddenMax.d[hiddenMax.nbDims - 1] != mHiddenSize
        || routerMax.nbDims != 2 || routerMax.d[0] != maxTokens || routerMax.d[1] != mNumExperts
        || !shapesMatch(outputs[0].max, hiddenMax) || maxTokens > std::numeric_limits<int32_t>::max() / mTopK)
    {
        LOG_ERROR("Nvfp4A16BlackwellMoePlugin: invalid profile max shape while computing workspace");
        return 0;
    }
    kernel::Nvfp4A16BlackwellMoeParams const maxShape
        = makeShape(static_cast<int32_t>(maxTokens), mNumExperts, mTopK, mHiddenSize, mMoeInterSize, interSizePadded(),
            mNGroup, mTopkGroup, mNormTopkProb, mRoutedScalingFactor, mBackend);
    return kernel::Nvfp4A16BlackwellMoeRunner::getWorkspaceSize(maxShape);
}

int32_t Nvfp4A16BlackwellMoePlugin::enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc,
    void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    try
    {
        if (inputDesc == nullptr || outputDesc == nullptr || inputs == nullptr || outputs == nullptr
            || outputs[0] == nullptr)
        {
            LOG_ERROR("Nvfp4A16BlackwellMoePlugin: enqueue received null descriptors or buffers");
            return -1;
        }
        int64_t numTokens{0};
        auto const& hiddenDims = inputDesc[kInHiddenStates].dims;
        auto const& routerDims = inputDesc[kInRouterLogits].dims;
        if (!getTokenCount(hiddenDims, numTokens) || hiddenDims.d[hiddenDims.nbDims - 1] != mHiddenSize
            || routerDims.nbDims != 2 || routerDims.d[0] != numTokens || routerDims.d[1] != mNumExperts
            || !shapesMatch(outputDesc[0].dims, hiddenDims) || numTokens > std::numeric_limits<int32_t>::max() / mTopK)
        {
            LOG_ERROR("Nvfp4A16BlackwellMoePlugin: invalid runtime hidden_states shape");
            return -1;
        }
        int64_t const requiredRows = moe::maxRowsPadded(numTokens, mTopK, mNumExperts, moe::kLargestTokenTile);
        if (requiredRows > mMaxRoutedRows)
        {
            LOG_ERROR("Nvfp4A16BlackwellMoePlugin: runtime shape needs %lld padded rows but max_routed_rows is %d",
                static_cast<long long>(requiredRows), mMaxRoutedRows);
            return -1;
        }
        for (int32_t pos = 0; pos < kNbPluginInputs; ++pos)
        {
            if (inputs[pos] == nullptr)
            {
                LOG_ERROR("Nvfp4A16BlackwellMoePlugin: input %d is null", pos);
                return -1;
            }
        }

        NVTX_SCOPED_RANGE(nvtx_moe, "Nvfp4A16BlackwellMoePlugin", nvtx_colors::BLUE);
        kernel::Nvfp4A16BlackwellMoeParams params
            = makeShape(static_cast<int32_t>(numTokens), mNumExperts, mTopK, mHiddenSize, mMoeInterSize,
                interSizePadded(), mNGroup, mTopkGroup, mNormTopkProb, mRoutedScalingFactor, mBackend);
        params.routerLogits = static_cast<float const*>(inputs[kInRouterLogits]);
        params.correctionBias = static_cast<float const*>(inputs[kInExpertScoreBias]);
        params.hiddenStates = inputs[kInHiddenStates];
        params.fc1QWeights = inputs[kInFc1QWeights];
        params.fc1BlockScales = inputs[kInFc1BlockScales];
        params.fc1GlobalScales = static_cast<float const*>(inputs[kInFc1GlobalScales]);
        params.fc2QWeights = inputs[kInFc2QWeights];
        params.fc2BlockScales = inputs[kInFc2BlockScales];
        params.fc2GlobalScales = static_cast<float const*>(inputs[kInFc2GlobalScales]);
        params.output = outputs[0];
        params.enablePdl = requestPdl();
        if (!mJitRunner.isLoaded())
        {
            LOG_ERROR(
                "Nvfp4A16BlackwellMoePlugin: the layer's JIT bundle is not loaded (enqueue before configure/clone)");
            return -1;
        }
        params.jit = &mJitRunner;

        // The workspace was sized for the profile maximum; the runner re-derives
        // the layout for this token count and checks it fits.
        size_t const workspaceSize = kernel::Nvfp4A16BlackwellMoeRunner::getWorkspaceSize(params);
        cudaError_t const error = kernel::Nvfp4A16BlackwellMoeRunner::run(params, workspace, workspaceSize, stream);
        if (error != cudaSuccess)
        {
            LOG_ERROR("Nvfp4A16BlackwellMoePlugin: runner failed: %s", cudaGetErrorString(error));
            return -1;
        }
        return 0;
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("Nvfp4A16BlackwellMoePlugin enqueue failed: %s", e.what());
        return -1;
    }
}

int32_t Nvfp4A16BlackwellMoePlugin::onShapeChange(
    PluginTensorDesc const* in, int32_t nbInputs, PluginTensorDesc const* out, int32_t nbOutputs) noexcept
{
    if (in == nullptr || out == nullptr || nbInputs != kNbPluginInputs || nbOutputs != 1)
    {
        LOG_ERROR("Nvfp4A16BlackwellMoePlugin: onShapeChange expected %d inputs and 1 output", kNbPluginInputs);
        return -1;
    }
    for (int32_t pos = 0; pos < kNbPluginInputs; ++pos)
    {
        if (!validateTensorDesc(pos, in[pos]))
        {
            LOG_ERROR("Nvfp4A16BlackwellMoePlugin: invalid shape-change descriptor at input %d", pos);
            return -1;
        }
    }
    if (!validateTensorDesc(kOutOutput, out[0]))
    {
        LOG_ERROR("Nvfp4A16BlackwellMoePlugin: invalid shape-change output descriptor");
        return -1;
    }
    int64_t numTokens{0};
    auto const& hiddenDims = in[kInHiddenStates].dims;
    auto const& routerDims = in[kInRouterLogits].dims;
    if (!getTokenCount(hiddenDims, numTokens) || hiddenDims.d[hiddenDims.nbDims - 1] != mHiddenSize
        || routerDims.nbDims != 2 || routerDims.d[0] != numTokens || routerDims.d[1] != mNumExperts
        || !shapesMatch(out[0].dims, hiddenDims) || numTokens > std::numeric_limits<int32_t>::max() / mTopK)
    {
        LOG_ERROR("Nvfp4A16BlackwellMoePlugin: runtime router/output shapes do not match hidden_states");
        return -1;
    }
    int64_t const requiredRows = moe::maxRowsPadded(numTokens, mTopK, mNumExperts, moe::kLargestTokenTile);
    if (requiredRows > mMaxRoutedRows)
    {
        LOG_ERROR("Nvfp4A16BlackwellMoePlugin: runtime shape requires %lld padded rows but max_routed_rows is %d",
            static_cast<long long>(requiredRows), mMaxRoutedRows);
        return -1;
    }
    // Warm the exact AOT variants enqueue will dispatch, outside any graph capture.
    kernel::Nvfp4A16BlackwellMoeParams const shape
        = makeShape(static_cast<int32_t>(numTokens), mNumExperts, mTopK, mHiddenSize, mMoeInterSize, interSizePadded(),
            mNGroup, mTopkGroup, mNormTopkProb, mRoutedScalingFactor, mBackend);
    cudaError_t const error = kernel::Nvfp4A16BlackwellMoeRunner::prepare(shape, nullptr);
    if (error != cudaSuccess)
    {
        LOG_ERROR("Nvfp4A16BlackwellMoePlugin: kernel module preparation failed for %lld tokens: %s",
            static_cast<long long>(numTokens), cudaGetErrorString(error));
        return -1;
    }
    return 0;
}

IPluginV3* Nvfp4A16BlackwellMoePlugin::attachToContext(IPluginResourceContext* context) noexcept
{
    (void) context;
    return clone();
}

PluginFieldCollection const* Nvfp4A16BlackwellMoePlugin::getFieldsToSerialize() noexcept
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
        mDataToSerialize.emplace_back("layout", &mLayout, PluginFieldType::kINT32, 1);
        mDataToSerialize.emplace_back("backend", &mBackend, PluginFieldType::kINT32, 1);
        // The bundle exists once configurePlugin has run (compileJitBundle throws
        // otherwise); the runtime creator refuses a plugin without it.
        if (mJitBundle.size() > static_cast<size_t>(std::numeric_limits<int32_t>::max()))
        {
            LOG_ERROR("Nvfp4A16BlackwellMoePlugin: JIT bundle too large to serialize");
            return nullptr;
        }
        if (!mJitBundle.empty())
        {
            mDataToSerialize.emplace_back(
                kFieldJitBundle, mJitBundle.data(), PluginFieldType::kCHAR, static_cast<int32_t>(mJitBundle.size()));
        }
        mFCToSerialize.nbFields = static_cast<int32_t>(mDataToSerialize.size());
        mFCToSerialize.fields = mDataToSerialize.data();
        return &mFCToSerialize;
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("Failed to serialize Nvfp4A16BlackwellMoePlugin fields: %s", e.what());
        return nullptr;
    }
}

void Nvfp4A16BlackwellMoePlugin::compileJitBundle()
{
    kernel::Nvfp4A16BlackwellMoeParams const shape = makeShape(1, mNumExperts, mTopK, mHiddenSize, mMoeInterSize,
        interSizePadded(), mNGroup, mTopkGroup, mNormTopkProb, mRoutedScalingFactor, mBackend);
    Nvfp4A16BlackwellMoeJitKey const key = kernel::makeNvfp4A16BlackwellMoeJitKey(shape);
    if (char const* problem = describeNvfp4A16BlackwellMoeJitKeyProblem(key); problem != nullptr)
    {
        throw std::invalid_argument(
            std::string("Nvfp4A16BlackwellMoePlugin: CUDA-core kernels cannot be built: ") + problem);
    }
    if (!mJitBundle.empty())
    {
        if (!(mJitKernel.key == key))
        {
            throw std::invalid_argument(
                "Nvfp4A16BlackwellMoePlugin: one plugin instance cannot use multiple JIT semantic keys");
        }
    }
    else
    {
        mJitKernel = compileNvfp4A16BlackwellMoeJitKernel(key);
        mJitBundle = serializeNvfp4A16BlackwellMoeJitKernel(mJitKernel);
    }
    // TensorRT executes the build-phase instance (or its clone) for auto-tuning
    // and harnesses drive it directly, so the module is loaded here as well.
    if (!mJitRunner.isLoaded() || !(mJitRunner.getKey() == key))
    {
        mJitRunner.load(mJitKernel);
    }
}

bool Nvfp4A16BlackwellMoePlugin::hasSerializedJitBundle() const noexcept
{
    return !mJitBundle.empty();
}

void Nvfp4A16BlackwellMoePlugin::loadSerializedJitBundle()
{
    if (mJitBundle.empty() || mJitKernel.cubin.empty())
    {
        throw std::invalid_argument(
            "Nvfp4A16BlackwellMoePlugin: runtime requires the moe_jit_bundle serialized at engine build");
    }
    Nvfp4A16BlackwellMoeJitKey const& key = mJitKernel.key;
    if (key.numExperts != mNumExperts || key.topK != mTopK || key.hiddenSize != mHiddenSize
        || key.interSize != mMoeInterSize || key.interSizePadded != interSizePadded()
        || key.layout != kNVFP4_A16_BLACKWELL_MOE_LAYOUT_ABI)
    {
        throw std::invalid_argument("Nvfp4A16BlackwellMoePlugin: serialized JIT key does not match the plugin's shape");
    }
    mJitRunner.load(mJitKernel);
}

Nvfp4A16BlackwellMoePluginCreator::Nvfp4A16BlackwellMoePluginCreator()
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
    mPluginAttributes.emplace_back("layout", nullptr, PluginFieldType::kINT32, 1);
    mPluginAttributes.emplace_back("backend", nullptr, PluginFieldType::kINT32, 1);
    mPluginAttributes.emplace_back(kFieldJitBundle, nullptr, PluginFieldType::kCHAR, 0);

    mFieldCollection.nbFields = static_cast<int32_t>(mPluginAttributes.size());
    mFieldCollection.fields = mPluginAttributes.data();
}

char const* Nvfp4A16BlackwellMoePluginCreator::getPluginName() const noexcept
{
    return kPluginName;
}

char const* Nvfp4A16BlackwellMoePluginCreator::getPluginVersion() const noexcept
{
    return kPluginVersion;
}

PluginFieldCollection const* Nvfp4A16BlackwellMoePluginCreator::getFieldNames() noexcept
{
    return &mFieldCollection;
}

char const* Nvfp4A16BlackwellMoePluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

void Nvfp4A16BlackwellMoePluginCreator::setPluginNamespace(char const* pluginNamespace) noexcept
{
    mNamespace = pluginNamespace == nullptr ? "" : pluginNamespace;
}

IPluginV3* Nvfp4A16BlackwellMoePluginCreator::createPlugin(
    char const* name, PluginFieldCollection const* fc, TensorRTPhase phase) noexcept
{
    try
    {
        auto plugin = std::make_unique<Nvfp4A16BlackwellMoePlugin>(name == nullptr ? kPluginName : name, fc);
        if (phase == TensorRTPhase::kBUILD)
        {
            if (plugin->hasSerializedJitBundle())
            {
                throw std::invalid_argument("Nvfp4A16BlackwellMoePlugin: BUILD phase must not receive moe_jit_bundle");
            }
        }
        else
        {
            // Runtime deserialization: the engine must carry the bundle compiled
            // at build time; the runtime never compiles.
            plugin->loadSerializedJitBundle();
        }
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin.release();
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("Failed to create Nvfp4A16BlackwellMoePlugin: %s", e.what());
        return nullptr;
    }
}

} // namespace plugins
} // namespace trt_edgellm
