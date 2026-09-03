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

#include "vitAttentionPlugin.h"

#include "common/checkMacros.h"
#include "common/cudaUtils.h"
#include "common/logger.h"
#include "common/tensor.h"
#include "kernels/contextAttentionKernels/attentionScaleUtils.h"
#include "kernels/contextAttentionKernels/utilKernels.h"
#include "plugins/utils/pluginUtils.h"

#ifdef CUTE_DSL_FMHA_BLACKWELL_ENABLED
#include "kernels/contextAttentionKernels/cuteDslFMHARunner.h"
#endif

#include "kernels/contextAttentionKernels/cuteDslFMHAV2Runner.h"

#include <cassert>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <vector>

using namespace nvinfer1;

namespace trt_edgellm
{
namespace plugins
{

namespace
{
constexpr char const* kATTENTION_PLUGIN_VERSION{"1"};
constexpr char const* kATTENTION_PLUGIN_NAME{"ViTAttentionPlugin"};

// Define the mapping of input and output indices of the ViTAttentionPlugin.
constexpr int32_t kIN_Q_IDX{0};
constexpr int32_t kIN_K_IDX{1};
constexpr int32_t kIN_V_IDX{2};
constexpr int32_t kIN_CU_SEQLENS_IDX{3};
constexpr int32_t kIN_MAX_SEQLEN_CARRIER_IDX{4};
constexpr int32_t kOUT_ATTENTION_IDX{0};

// Reflect the count of Inputs and Outputs of the ViTAttentionPlugin,
// these definitions shall be consistent.
constexpr int32_t kNUM_REQUIRED_INPUTS{5};
constexpr int32_t kNUM_REQUIRED_OUTPUTS{1};

struct ViTFMHAKernelSelection
{
    ViTFMHABackend backend{ViTFMHABackend::kNONE};
    bool canImplement{false};
};

ViTFMHAKernelSelection selectViTFMHAKernels(int32_t headSize, int32_t smVersion, nvinfer1::DataType dataType)
{
#ifdef CUTE_DSL_FMHA_BLACKWELL_ENABLED
    if (CuteDslFMHARunner::canImplementViT(headSize, smVersion))
    {
        return {ViTFMHABackend::kCUTE_DSL_FMHA_BLACKWELL, true};
    }
#endif

    if (CuteDslFMHAV2Runner::canImplementViT(headSize, smVersion, dataType))
    {
        return {ViTFMHABackend::kCUTE_DSL_FMHA_V2, true};
    }

    return {};
}
} // namespace

// Static class fields initialization
PluginFieldCollection ViTAttentionPluginCreator::mFieldCollection{};
std::vector<PluginField> ViTAttentionPluginCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(ViTAttentionPluginCreator);

ViTAttentionPlugin::ViTAttentionPlugin(std::string const& name, int32_t numHeads, int32_t headSize,
    std::optional<float> attentionScale, std::vector<float> const& qkvScales)
    : mLayerName(name)
    , mNumHeads(numHeads)
    , mHeadSize(headSize)
    , mAttentionScale(resolveAttentionScale(attentionScale, headSize))
    , mQkvScales(qkvScales)
{
    ELLM_CHECK(mQkvScales.size() == 3,
        "ViTAttentionPlugin: qkv_scales has " + std::to_string(mQkvScales.size()) + " elements (expected 3).");
    for (float const scale : mQkvScales)
    {
        ELLM_CHECK(
            scale > 0.F, "ViTAttentionPlugin: qkv_scales entries must be positive, got " + std::to_string(scale) + ".");
    }

    mSMVersion = getSMVersion();
    applyThorSMRenumberWAR(mSMVersion);

    ViTFMHAKernelSelection const fmhaSelection = selectViTFMHAKernels(mHeadSize, mSMVersion, mDataType);
    mFMHABackend = fmhaSelection.backend;

    if (!fmhaSelection.canImplement)
    {
        LOG_ERROR("Cannot implement ViTAttentionPlugin configuration. SM: %d, HeadSize: %d, NumHeads: %d", mSMVersion,
            mHeadSize, mNumHeads);
        throw std::runtime_error("Cannot implement the ViTAttentionPlugin configuration.");
    }
}

ViTAttentionPlugin::ViTAttentionPlugin(std::string const& name, PluginFieldCollection const* fc)
    : mLayerName(name)
    , mNumHeads(parsePluginScalarField<int32_t>("num_heads", fc).value_or(0))
    , mHeadSize(parsePluginScalarField<int32_t>("head_size", fc).value_or(0))
    , mAttentionScale(resolveAttentionScale(parsePluginScalarField<float>("attention_scale", fc), mHeadSize))
{
    ELLM_CHECK(mNumHeads > 0 && mHeadSize > 0, "ViTAttentionPlugin requires both num_heads and head_size > 0.");

    // Parse qkv_scales float array (3 entries: [qScale, kScale, vScale]).
    for (int32_t i = 0; i < fc->nbFields; ++i)
    {
        if (std::string("qkv_scales") == fc->fields[i].name)
        {
            auto const* data = static_cast<float const*>(fc->fields[i].data);
            mQkvScales.assign(data, data + fc->fields[i].length);
            break;
        }
    }
    ELLM_CHECK(mQkvScales.size() == 3,
        "ViTAttentionPlugin: qkv_scales has " + std::to_string(mQkvScales.size()) + " elements (expected 3).");
    for (float const scale : mQkvScales)
    {
        ELLM_CHECK(
            scale > 0.F, "ViTAttentionPlugin: qkv_scales entries must be positive, got " + std::to_string(scale) + ".");
    }

    mSMVersion = getSMVersion();
    applyThorSMRenumberWAR(mSMVersion);

    ViTFMHAKernelSelection const fmhaSelection = selectViTFMHAKernels(mHeadSize, mSMVersion, mDataType);
    mFMHABackend = fmhaSelection.backend;
    if (!fmhaSelection.canImplement)
    {
        LOG_ERROR("Cannot implement ViTAttentionPlugin configuration. SM: %d, HeadSize: %d, NumHeads: %d", mSMVersion,
            mHeadSize, mNumHeads);
        throw std::runtime_error("Cannot implement the ViTAttentionPlugin configuration.");
    }
}

ViTAttentionPlugin::~ViTAttentionPlugin() = default;

// ---------------------------------------------------------------------------
// IPluginV3
// ---------------------------------------------------------------------------

IPluginCapability* ViTAttentionPlugin::getCapabilityInterface(PluginCapabilityType type) noexcept
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
    catch (std::exception const& e)
    {
        return nullptr;
    }
}

IPluginV3* ViTAttentionPlugin::clone() noexcept
{
    try
    {
        auto* p = new ViTAttentionPlugin(mLayerName, mNumHeads, mHeadSize, mAttentionScale, mQkvScales);
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

char const* ViTAttentionPlugin::getPluginName() const noexcept
{
    return kATTENTION_PLUGIN_NAME;
}

char const* ViTAttentionPlugin::getPluginVersion() const noexcept
{
    return kATTENTION_PLUGIN_VERSION;
}

char const* ViTAttentionPlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

void ViTAttentionPlugin::setPluginNamespace(char const* pluginNamespace) noexcept
{
    mNamespace = pluginNamespace ? pluginNamespace : "";
}

// ---------------------------------------------------------------------------
// IPluginV3OneBuild — shape / format
// ---------------------------------------------------------------------------

int32_t ViTAttentionPlugin::getNbOutputs() const noexcept
{
    return kNUM_REQUIRED_OUTPUTS;
}

int32_t ViTAttentionPlugin::getOutputDataTypes(DataType* outputTypes, [[maybe_unused]] int32_t nbOutputs,
    [[maybe_unused]] DataType const* inputTypes, [[maybe_unused]] int32_t nbInputs) const noexcept
{
    try
    {
        assert(nbOutputs == kNUM_REQUIRED_OUTPUTS);
        // Attention output is always FP16; plugin computes FP8 internally but dequants to FP16.
        outputTypes[kOUT_ATTENTION_IDX] = DataType::kHALF;
        return 0;
    }
    catch (std::exception const& e)
    {
        return -1;
    }
}

int32_t ViTAttentionPlugin::getOutputShapes(DimsExprs const* inputs, [[maybe_unused]] int32_t nbInputs,
    DimsExprs const* /* shapeInputs */, int32_t /* nbShapeInputs */, DimsExprs* outputs,
    [[maybe_unused]] int32_t nbOutputs, IExprBuilder& /* exprBuilder */) noexcept
{
    try
    {
        assert(nbInputs == kNUM_REQUIRED_INPUTS);
        assert(nbOutputs == kNUM_REQUIRED_OUTPUTS);
        // Output[0] has the same shape as Q: [total_S, H, D]
        outputs[kOUT_ATTENTION_IDX] = inputs[kIN_Q_IDX];
        return 0;
    }
    catch (std::exception const& e)
    {
        return -1;
    }
}

bool ViTAttentionPlugin::supportsFormatCombination(
    int32_t pos, DynamicPluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
    // Support context/generation phase inputs:
    //      Q tensor (linear FP16) with shape [total_S, H, D]
    //      K tensor (linear FP16) with shape [total_S, H, D]
    //      V tensor (linear FP16) with shape [total_S, H, D]
    //      NOTE: This assumes a head-major layout. The Python export must guarantee this layout.
    //      CuSeqLens tensor (a vector of scalars) with shape [batch_size + 1] and type int32_t.
    //      max_seqlen_carrier tensor with shape [max_seqlen] and type int32_t. Values are ignored.

    // Support context/generation phase outputs:
    //      attention result (linear FP16) with shape [total_S, H, D]
    // Q, K, V, and output all have the same shape [total_S, H, D]
    // FP8 Q/K/V are accepted whenever the configuration has an FP8 kernel; the
    // actual mode is chosen by the graph's tensor dtypes (QuantizeLinear
    // upstream => FP8).
    auto checkQKVO = [this](PluginTensorDesc const& tensorDesc) {
        bool status{true};
        status &= (tensorDesc.type == DataType::kHALF || (supportsFp8Mha() && tensorDesc.type == DataType::kFP8));
        status &= tensorDesc.format == TensorFormat::kLINEAR;
        status &= tensorDesc.dims.nbDims == 3;
        auto const tensorDim = tensorDesc.dims;
        if (status)
        {
            status &= tensorDim.d[1] == mNumHeads;
            status &= tensorDim.d[2] == mHeadSize;
        }
        return status;
    };

    auto checkCuSeqLens = [](PluginTensorDesc const& tensorDesc) {
        bool status{true};
        status &= tensorDesc.type == DataType::kINT32;
        status &= tensorDesc.format == TensorFormat::kLINEAR;
        status &= tensorDesc.dims.nbDims == 1;
        return status;
    };

    auto checkMaxSeqLenCarrier = [](PluginTensorDesc const& tensorDesc) {
        bool status{true};
        status &= tensorDesc.type == DataType::kINT32;
        status &= tensorDesc.format == TensorFormat::kLINEAR;
        status &= tensorDesc.dims.nbDims == 1;
        return status;
    };

    bool const checkNumIOs = nbInputs == kNUM_REQUIRED_INPUTS && nbOutputs == kNUM_REQUIRED_OUTPUTS;
    if (!checkNumIOs)
    {
        LOG_ERROR(
            "Invalid number of inputs or outputs for the ViTAttentionPlugin '%s'. Expected %d inputs and %d outputs, "
            "but "
            "got %d inputs and %d outputs.",
            mLayerName.c_str(), kNUM_REQUIRED_INPUTS, kNUM_REQUIRED_OUTPUTS, nbInputs, nbOutputs);
        return false;
    }

    bool result{true};

    if (pos < nbInputs)
    {
        switch (pos)
        {
        case kIN_Q_IDX: result = checkQKVO(inOut[0].desc); break;
        case kIN_K_IDX: result = checkQKVO(inOut[1].desc) && inOut[1].desc.type == inOut[kIN_Q_IDX].desc.type; break;
        case kIN_V_IDX: result = checkQKVO(inOut[2].desc) && inOut[2].desc.type == inOut[kIN_Q_IDX].desc.type; break;
        case kIN_CU_SEQLENS_IDX: result = checkCuSeqLens(inOut[3].desc); break;
        case kIN_MAX_SEQLEN_CARRIER_IDX: result = checkMaxSeqLenCarrier(inOut[4].desc); break;
        default: break;
        }
    }
    else
    {
        int32_t outPos = pos - nbInputs;
        switch (outPos)
        {
        case kOUT_ATTENTION_IDX:
            // Attention output is always FP16, also for FP8 Q/K/V.
            result = checkQKVO(inOut[pos].desc) && inOut[pos].desc.type == DataType::kHALF;
            break;
        default: break;
        }
    }

    return result;
}

int32_t ViTAttentionPlugin::configurePlugin([[maybe_unused]] DynamicPluginTensorDesc const* in,
    [[maybe_unused]] int32_t nbInputs, [[maybe_unused]] DynamicPluginTensorDesc const* out,
    [[maybe_unused]] int32_t nbOutputs) noexcept
{
    return 0; // No need to configure anything since we will only use the runtime tensor shapes.
}

size_t ViTAttentionPlugin::getWorkspaceSize(DynamicPluginTensorDesc const* inputs, [[maybe_unused]] int32_t nbInputs,
    [[maybe_unused]] DynamicPluginTensorDesc const* outputs, [[maybe_unused]] int32_t nbOutputs) const noexcept
{
    return 0;
}

// ---------------------------------------------------------------------------
// IPluginV3OneRuntime — execution
// ---------------------------------------------------------------------------

int32_t ViTAttentionPlugin::enqueue(PluginTensorDesc const* inputDesc,
    [[maybe_unused]] PluginTensorDesc const* outputDesc, void const* const* inputs, void* const* outputs,
    void* workspace, cudaStream_t stream) noexcept
{
    // Construct non-owned tensor objects from I/O data pointers and shapes.
    // Q, K, V inputs in the graph will be in same shape [total_S, H, D].
    PluginTensorDesc const& qInputDesc = inputDesc[kIN_Q_IDX];
    rt::Coords const qkvCoords{qInputDesc.dims};
    rt::Tensor qInputTensor(const_cast<void*>(inputs[kIN_Q_IDX]), qkvCoords, rt::DeviceType::kGPU, qInputDesc.type);
    rt::Tensor kInputTensor(const_cast<void*>(inputs[kIN_K_IDX]), qkvCoords, rt::DeviceType::kGPU, qInputDesc.type);
    rt::Tensor vInputTensor(const_cast<void*>(inputs[kIN_V_IDX]), qkvCoords, rt::DeviceType::kGPU, qInputDesc.type);

    PluginTensorDesc const& cuSeqLensInputDesc = inputDesc[kIN_CU_SEQLENS_IDX];
    rt::Tensor cuSeqLensTensor(const_cast<void*>(inputs[kIN_CU_SEQLENS_IDX]), rt::Coords{cuSeqLensInputDesc.dims},
        rt::DeviceType::kGPU, cuSeqLensInputDesc.type);

    PluginTensorDesc const& attentionOutputDesc = outputDesc[kOUT_ATTENTION_IDX];
    rt::Tensor attentionOutputTensor(outputs[kOUT_ATTENTION_IDX], rt::Coords{attentionOutputDesc.dims},
        rt::DeviceType::kGPU, attentionOutputDesc.type);

    PluginTensorDesc const& maxSeqLenCarrierDesc = inputDesc[kIN_MAX_SEQLEN_CARRIER_IDX];
    int32_t runtimeMaxSeqLen = static_cast<int32_t>(maxSeqLenCarrierDesc.dims.d[0]);
    int32_t runtimeBatchSize = static_cast<int32_t>(cuSeqLensInputDesc.dims.d[0]) - 1;

#ifdef CUTE_DSL_FMHA_BLACKWELL_ENABLED
    if (mFMHABackend == ViTFMHABackend::kCUTE_DSL_FMHA_BLACKWELL)
    {
        int32_t totalSeqLen = static_cast<int32_t>(qInputDesc.dims.d[0]);
        CuteDslFMHARunner runner(mNumHeads, mNumHeads, mHeadSize);
        if (!runner.preflightViT(stream))
        {
            return -1;
        }
        if (qInputDesc.type == DataType::kFP8)
        {
            // Q/K/V arrive pre-quantized as FP8 via upstream QuantizeLinear nodes.
            return runner.run(inputs[kIN_Q_IDX], inputs[kIN_K_IDX], inputs[kIN_V_IDX],
                       attentionOutputTensor.dataPointer<half>(), cuSeqLensTensor.dataPointer<int32_t>(), totalSeqLen,
                       runtimeMaxSeqLen, runtimeBatchSize, stream, mAttentionScale, /*fp8Input=*/true, mQkvScales[0],
                       mQkvScales[1], mQkvScales[2])
                ? 0
                : -1;
        }
        return runner.run(qInputTensor.dataPointer<half>(), kInputTensor.dataPointer<half>(),
                   vInputTensor.dataPointer<half>(), attentionOutputTensor.dataPointer<half>(),
                   cuSeqLensTensor.dataPointer<int32_t>(), totalSeqLen, runtimeMaxSeqLen, runtimeBatchSize, stream,
                   mAttentionScale)
            ? 0
            : -1;
    }
#endif
    if (mFMHABackend == ViTFMHABackend::kCUTE_DSL_FMHA_V2)
    {
        int32_t const totalSeqLen = static_cast<int32_t>(qInputDesc.dims.d[0]);
        CuteDslFMHAV2Runner runner(mNumHeads, mNumHeads, mHeadSize);
        if (!runner.preflightViT(stream))
        {
            return -1;
        }
        if (!runner.run(qInputTensor.dataPointer<half>(), kInputTensor.dataPointer<half>(),
                vInputTensor.dataPointer<half>(), attentionOutputTensor.dataPointer<half>(),
                cuSeqLensTensor.dataPointer<int32_t>(), totalSeqLen, runtimeMaxSeqLen, runtimeBatchSize, stream,
                mAttentionScale))
        {
            return -1;
        }
        return 0;
    }
    LOG_ERROR("No ViT FMHA backend is available for plugin '%s'.", mLayerName.c_str());
    return -1;
}

int32_t ViTAttentionPlugin::onShapeChange([[maybe_unused]] PluginTensorDesc const* in,
    [[maybe_unused]] int32_t nbInputs, [[maybe_unused]] PluginTensorDesc const* out,
    [[maybe_unused]] int32_t nbOutputs) noexcept
{
    return 0;
}

IPluginV3* ViTAttentionPlugin::attachToContext([[maybe_unused]] IPluginResourceContext* context) noexcept
{
    return clone();
}

PluginFieldCollection const* ViTAttentionPlugin::getFieldsToSerialize() noexcept
{
    mDataToSerialize.clear();
    mDataToSerialize.emplace_back("num_heads", &mNumHeads, PluginFieldType::kINT32, 1);
    mDataToSerialize.emplace_back("head_size", &mHeadSize, PluginFieldType::kINT32, 1);
    mDataToSerialize.emplace_back("attention_scale", &mAttentionScale, PluginFieldType::kFLOAT32, 1);
    mDataToSerialize.emplace_back(
        "qkv_scales", mQkvScales.data(), PluginFieldType::kFLOAT32, static_cast<int32_t>(mQkvScales.size()));
    mFCToSerialize.nbFields = static_cast<int32_t>(mDataToSerialize.size());
    mFCToSerialize.fields = mDataToSerialize.data();
    return &mFCToSerialize;
}

// ---------------------------------------------------------------------------
// Creator
// ---------------------------------------------------------------------------

ViTAttentionPluginCreator::ViTAttentionPluginCreator()
{
    static std::mutex sMutex;
    std::lock_guard<std::mutex> lock(sMutex);

    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("num_heads", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("head_size", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("attention_scale", nullptr, PluginFieldType::kFLOAT32, 0));
    mPluginAttributes.emplace_back(PluginField("qkv_scales", nullptr, PluginFieldType::kFLOAT32, 0));
    mFieldCollection.nbFields = mPluginAttributes.size();
    mFieldCollection.fields = mPluginAttributes.data();
}

char const* ViTAttentionPluginCreator::getPluginName() const noexcept
{
    return kATTENTION_PLUGIN_NAME;
}

PluginFieldCollection const* ViTAttentionPluginCreator::getFieldNames() noexcept
{
    return &mFieldCollection;
}

void ViTAttentionPluginCreator::setPluginNamespace(char const* libNamespace) noexcept
{
    mNamespace = libNamespace ? libNamespace : "";
}

char const* ViTAttentionPluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

char const* ViTAttentionPluginCreator::getPluginVersion() const noexcept
{
    return kATTENTION_PLUGIN_VERSION;
}

IPluginV3* ViTAttentionPluginCreator::createPlugin(
    char const* name, PluginFieldCollection const* fc, [[maybe_unused]] TensorRTPhase phase) noexcept
{
    try
    {
        auto* plugin = new ViTAttentionPlugin(std::string(name), fc);
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("Failed to create ViTAttentionPlugin: %s", e.what());
    }
    return nullptr;
}

} // namespace plugins
} // namespace trt_edgellm
