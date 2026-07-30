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

#include "dflashTargetKVCacheUpdatePlugin.h"
#include "common/logger.h"
#include "common/pagedKvTypes.h"
#include "kernels/speculative/dflashRuntimeKernels.h"
#include "plugins/utils/pluginUtils.h"

#include <NvInfer.h>
#include <cstdint>

using namespace nvinfer1;

namespace trt_edgellm
{
namespace plugins
{

namespace
{
char const* const kPLUGIN_NAME = "DFlashTargetKVCacheUpdate";
char const* const kPLUGIN_VERSION = "1";
constexpr int32_t kNUM_INPUTS{6};
constexpr int32_t kNUM_OUTPUTS{1};

bool checkExpectedIO(char const* where, int32_t nbInputs, int32_t nbOutputs)
{
    if (nbInputs != kNUM_INPUTS || nbOutputs != kNUM_OUTPUTS)
    {
        LOG_ERROR("%s: expected %d inputs and %d output, got %d inputs and %d outputs", where, kNUM_INPUTS,
            kNUM_OUTPUTS, nbInputs, nbOutputs);
        return false;
    }
    return true;
}
} // namespace

// ---------------------------------------------------------------------------
// Plugin
// ---------------------------------------------------------------------------

DFlashTargetKVCacheUpdatePlugin::DFlashTargetKVCacheUpdatePlugin(std::string const& name)
    : mLayerName(name)
{
}

DFlashTargetKVCacheUpdatePlugin::DFlashTargetKVCacheUpdatePlugin(
    std::string const& name, PluginFieldCollection const* fc)
    : mLayerName(name)
{
    // pages_per_slot is absent at ONNX-parse time (export doesn't know maxKVCacheCapacity); the
    // builder sets it via setPagesPerSlot() after parsing, before the network is built. It IS
    // present when this constructor runs during engine deserialization (round-tripped via
    // getFieldsToSerialize()), so parse it here for that path.
    mPagesPerSlot = parsePluginScalarField<int32_t>("pages_per_slot", fc).value_or(0);
}

// IPluginV3
IPluginCapability* DFlashTargetKVCacheUpdatePlugin::getCapabilityInterface(PluginCapabilityType type) noexcept
{
    try
    {
        if (type == PluginCapabilityType::kBUILD)
        {
            return static_cast<IPluginV3OneBuildV2*>(this);
        }
        if (type == PluginCapabilityType::kRUNTIME)
        {
            return static_cast<IPluginV3OneRuntime*>(this);
        }
        if (type == PluginCapabilityType::kCORE)
        {
            return static_cast<IPluginV3OneCore*>(this);
        }
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("DFlashTargetKVCacheUpdatePlugin: getCapabilityInterface exception: %s", e.what());
    }
    return nullptr;
}

IPluginV3* DFlashTargetKVCacheUpdatePlugin::clone() noexcept
{
    try
    {
        auto* plugin = new DFlashTargetKVCacheUpdatePlugin(mLayerName);
        plugin->setPluginNamespace(mNamespace.c_str());
        plugin->setPagesPerSlot(mPagesPerSlot);
        return plugin;
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("DFlashTargetKVCacheUpdatePlugin: clone exception: %s", e.what());
    }
    return nullptr;
}

// IPluginV3OneCore
char const* DFlashTargetKVCacheUpdatePlugin::getPluginName() const noexcept
{
    return kPLUGIN_NAME;
}

char const* DFlashTargetKVCacheUpdatePlugin::getPluginVersion() const noexcept
{
    return kPLUGIN_VERSION;
}

char const* DFlashTargetKVCacheUpdatePlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

// IPluginV3OneBuild
int32_t DFlashTargetKVCacheUpdatePlugin::getNbOutputs() const noexcept
{
    return 1; // present_key_value
}

int32_t DFlashTargetKVCacheUpdatePlugin::getOutputDataTypes(
    DataType* outputTypes, int32_t nbOutputs, DataType const* inputTypes, int32_t nbInputs) const noexcept
{
    // Output dtype = past_key_value dtype (input 2)
    if (!checkExpectedIO("DFlashTargetKVCacheUpdatePlugin::getOutputDataTypes", nbInputs, nbOutputs)
        || outputTypes == nullptr || inputTypes == nullptr)
    {
        return -1;
    }
    outputTypes[0] = inputTypes[kIN_PAST_KV];
    return 0;
}

int32_t DFlashTargetKVCacheUpdatePlugin::getOutputShapes(DimsExprs const* inputs, int32_t nbInputs,
    DimsExprs const* /* shapeInputs */, int32_t /* nbShapeInputs */, DimsExprs* outputs, int32_t nbOutputs,
    IExprBuilder& /* exprBuilder */) noexcept
{
    // Output shape = past_key_value shape
    if (!checkExpectedIO("DFlashTargetKVCacheUpdatePlugin::getOutputShapes", nbInputs, nbOutputs) || inputs == nullptr
        || outputs == nullptr)
    {
        return -1;
    }
    outputs[0] = inputs[kIN_PAST_KV];
    return 0;
}

bool DFlashTargetKVCacheUpdatePlugin::supportsFormatCombination(
    int32_t pos, DynamicPluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
    if (inOut == nullptr || nbInputs != kNUM_INPUTS || nbOutputs != kNUM_OUTPUTS || pos < 0
        || pos >= nbInputs + nbOutputs)
    {
        return false;
    }

    auto const& desc = inOut[pos];
    bool const isLinearFormat = (desc.desc.format == TensorFormat::kLINEAR);

    if (pos == kIN_K_DELTA || pos == kIN_V_DELTA || pos == kIN_PAST_KV)
    {
        // K/V delta and KV cache must be FP16
        return isLinearFormat && desc.desc.type == DataType::kHALF;
    }
    else if (pos == kIN_ROPE_COS_SIN)
    {
        // RoPE cos/sin must be FP32
        return isLinearFormat && desc.desc.type == DataType::kFLOAT;
    }
    else if (pos == kIN_DELTA_START || pos == kIN_DELTA_LENGTHS)
    {
        // delta_start_positions and delta_lengths must be INT32
        return isLinearFormat && desc.desc.type == DataType::kINT32;
    }
    else if (pos == nbInputs + kOUT_PRESENT_KV)
    {
        // Output must match past_key_value
        return isLinearFormat && desc.desc.type == inOut[kIN_PAST_KV].desc.type;
    }
    return false;
}

int32_t DFlashTargetKVCacheUpdatePlugin::configurePlugin(
    DynamicPluginTensorDesc const* in, int32_t nbInputs, DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept
{
    if (!checkExpectedIO("DFlashTargetKVCacheUpdatePlugin::configurePlugin", nbInputs, nbOutputs) || in == nullptr
        || out == nullptr)
    {
        return -1;
    }
    if (in[kIN_K_DELTA].desc.type != DataType::kHALF || in[kIN_V_DELTA].desc.type != DataType::kHALF
        || in[kIN_PAST_KV].desc.type != DataType::kHALF)
    {
        LOG_ERROR("DFlashTargetKVCacheUpdatePlugin: k_delta, v_delta, and past_key_value must be FP16");
        return -1;
    }
    if (in[kIN_ROPE_COS_SIN].desc.type != DataType::kFLOAT)
    {
        LOG_ERROR("DFlashTargetKVCacheUpdatePlugin: rope_cos_sin must be FP32");
        return -1;
    }
    if (in[kIN_DELTA_START].desc.type != DataType::kINT32 || in[kIN_DELTA_LENGTHS].desc.type != DataType::kINT32)
    {
        LOG_ERROR("DFlashTargetKVCacheUpdatePlugin: delta_start_positions and delta_lengths must be INT32");
        return -1;
    }
    if (in[kIN_K_DELTA].desc.dims.nbDims != 4 || in[kIN_V_DELTA].desc.dims.nbDims != 4
        || in[kIN_PAST_KV].desc.dims.nbDims != 5 || in[kIN_ROPE_COS_SIN].desc.dims.nbDims != 3
        || in[kIN_DELTA_START].desc.dims.nbDims != 1 || in[kIN_DELTA_LENGTHS].desc.dims.nbDims != 1)
    {
        LOG_ERROR("DFlashTargetKVCacheUpdatePlugin: invalid input ranks");
        return -1;
    }
    if (out[kOUT_PRESENT_KV].desc.type != in[kIN_PAST_KV].desc.type || out[kOUT_PRESENT_KV].desc.dims.nbDims != 5)
    {
        LOG_ERROR("DFlashTargetKVCacheUpdatePlugin: present_key_value must be 5D and match past_key_value dtype");
        return -1;
    }
    return 0;
}

size_t DFlashTargetKVCacheUpdatePlugin::getWorkspaceSize(DynamicPluginTensorDesc const* /* inputs */,
    int32_t /* nbInputs */, DynamicPluginTensorDesc const* /* outputs */, int32_t /* nbOutputs */) const noexcept
{
    return 0;
}

int32_t DFlashTargetKVCacheUpdatePlugin::getAliasedInput(int32_t outputIndex) noexcept
{
    // present_key_value (output 0) aliases past_key_value (input 2). DFlash
    // draft depends on this in-engine target KV update before subsequent
    // attention layers read the same KV cache.
    if (outputIndex == kOUT_PRESENT_KV)
    {
        return kIN_PAST_KV;
    }
    return -1;
}

// IPluginV3OneRuntime
int32_t DFlashTargetKVCacheUpdatePlugin::enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc,
    void const* const* inputs, void* const* outputs, void* /* workspace */, cudaStream_t stream) noexcept
{
    try
    {
        if (inputDesc == nullptr || outputDesc == nullptr || inputs == nullptr || outputs == nullptr)
        {
            LOG_ERROR("DFlashTargetKVCacheUpdatePlugin::enqueue received null descriptors or pointers");
            return -1;
        }

        // k_delta: [B, L, numKVHeads, headDim]
        auto const& kDeltaDesc = inputDesc[kIN_K_DELTA];
        if (kDeltaDesc.type != DataType::kHALF || kDeltaDesc.dims.nbDims != 4)
        {
            LOG_ERROR("DFlashTargetKVCacheUpdatePlugin: k_delta must be 4D FP16");
            return -1;
        }
        int32_t const batchSize = kDeltaDesc.dims.d[0];
        int32_t const deltaLen = kDeltaDesc.dims.d[1];
        int32_t const numKVHeads = kDeltaDesc.dims.d[2];
        int32_t const headDim = kDeltaDesc.dims.d[3];
        if (batchSize <= 0 || deltaLen <= 0 || numKVHeads <= 0 || headDim <= 0)
        {
            LOG_ERROR("DFlashTargetKVCacheUpdatePlugin: k_delta dimensions must be positive");
            return -1;
        }

        // v_delta must match k_delta shape
        [[maybe_unused]] auto const& vDeltaDesc = inputDesc[kIN_V_DELTA];
        if (vDeltaDesc.type != DataType::kHALF || vDeltaDesc.dims.nbDims != 4 || vDeltaDesc.dims.d[0] != batchSize
            || vDeltaDesc.dims.d[1] != deltaLen || vDeltaDesc.dims.d[2] != numKVHeads
            || vDeltaDesc.dims.d[3] != headDim)
        {
            LOG_ERROR("DFlashTargetKVCacheUpdatePlugin: v_delta must match k_delta shape and dtype");
            return -1;
        }

        // past_key_value is the runtime's KVCacheManager-allocated pool, bound as the same paged-pool
        // contract as the AttentionPlugin binding: [2, numPages, kTOKENS_PER_PAGE, numKVHeads,
        // headDim] (K/V split OUTERMOST; see kvCacheManager.h). numPages = maxBatch * pagesPerSlot,
        // so maxBatch/cap are recovered using the build-time pagesPerSlot fact (see
        // setPagesPerSlot()) rather than read directly off dims 1/2.
        auto const& pastKVDesc = inputDesc[kIN_PAST_KV];
        if (pastKVDesc.type != DataType::kHALF || pastKVDesc.dims.nbDims != 5 || pastKVDesc.dims.d[0] != 2
            || pastKVDesc.dims.d[2] != rt::kTOKENS_PER_PAGE || pastKVDesc.dims.d[3] != numKVHeads
            || pastKVDesc.dims.d[4] != headDim)
        {
            LOG_ERROR(
                "DFlashTargetKVCacheUpdatePlugin: past_key_value must be the paged pool "
                "[2, numPages, %d, numKVHeads, headDim] FP16",
                rt::kTOKENS_PER_PAGE);
            return -1;
        }
        if (mPagesPerSlot <= 0)
        {
            LOG_ERROR(
                "DFlashTargetKVCacheUpdatePlugin: pages_per_slot was never configured (builder bug — "
                "setPagesPerSlot() must run before the engine is built)");
            return -1;
        }
        int32_t const numPages = pastKVDesc.dims.d[1];
        if (numPages % mPagesPerSlot != 0)
        {
            LOG_ERROR(
                "DFlashTargetKVCacheUpdatePlugin: past_key_value numPages (%d) is not a multiple of "
                "pages_per_slot (%d)",
                numPages, mPagesPerSlot);
            return -1;
        }
        int32_t const maxBatch = numPages / mPagesPerSlot;
        int32_t const cap = mPagesPerSlot * rt::kTOKENS_PER_PAGE;
        if (maxBatch < batchSize || cap <= 0)
        {
            LOG_ERROR(
                "DFlashTargetKVCacheUpdatePlugin: past_key_value maxBatch (%d) must be >= active batchSize "
                "(%d), and cap (%d) must be positive",
                maxBatch, batchSize, cap);
            return -1;
        }

        // rope_cos_sin: [cosSinBatch, cosSinSeqLen, rotaryDim]
        auto const& ropeDesc = inputDesc[kIN_ROPE_COS_SIN];
        if (ropeDesc.type != DataType::kFLOAT || ropeDesc.dims.nbDims != 3)
        {
            LOG_ERROR("DFlashTargetKVCacheUpdatePlugin: rope_cos_sin must be 3D FP32");
            return -1;
        }
        int32_t const cosSinBatch = ropeDesc.dims.d[0];
        int32_t const cosSinSeqLen = ropeDesc.dims.d[1];
        int32_t const rotaryDim = ropeDesc.dims.d[2];
        if ((cosSinBatch != 1 && cosSinBatch != batchSize) || rotaryDim <= 0 || rotaryDim > headDim
            || (rotaryDim % 2) != 0)
        {
            LOG_ERROR("DFlashTargetKVCacheUpdatePlugin: invalid rope_cos_sin shape");
            return -1;
        }
        // cap is the KV pool's PADDED capacity; cosSinSeqLen is sized to the real (unpadded) max
        // sequence length, so cosSinSeqLen < cap is expected whenever that length isn't already
        // page-aligned. The only real invariant is cosSinSeqLen <= cap (see checkDFlashRopeCapacity).
        // Throws on violation; caught by this function's enclosing try/catch below.
        kernel::checkDFlashRopeCapacity(cosSinSeqLen, cap);

        auto const& deltaStartDesc = inputDesc[kIN_DELTA_START];
        auto const& deltaLengthsDesc = inputDesc[kIN_DELTA_LENGTHS];
        if (deltaStartDesc.type != DataType::kINT32 || deltaStartDesc.dims.nbDims != 1
            || deltaStartDesc.dims.d[0] != batchSize || deltaLengthsDesc.type != DataType::kINT32
            || deltaLengthsDesc.dims.nbDims != 1 || deltaLengthsDesc.dims.d[0] != batchSize)
        {
            LOG_ERROR("DFlashTargetKVCacheUpdatePlugin: delta_start_positions and delta_lengths must be [B] INT32");
            return -1;
        }

        auto const& presentKVDesc = outputDesc[kOUT_PRESENT_KV];
        if (presentKVDesc.type != pastKVDesc.type || presentKVDesc.dims.nbDims != pastKVDesc.dims.nbDims
            || presentKVDesc.dims.d[0] != pastKVDesc.dims.d[0] || presentKVDesc.dims.d[1] != pastKVDesc.dims.d[1]
            || presentKVDesc.dims.d[2] != pastKVDesc.dims.d[2] || presentKVDesc.dims.d[3] != pastKVDesc.dims.d[3]
            || presentKVDesc.dims.d[4] != pastKVDesc.dims.d[4])
        {
            LOG_ERROR("DFlashTargetKVCacheUpdatePlugin: present_key_value must match past_key_value shape and dtype");
            return -1;
        }
        for (int32_t i = 0; i < kNUM_INPUTS; ++i)
        {
            if (inputs[i] == nullptr)
            {
                LOG_ERROR("DFlashTargetKVCacheUpdatePlugin: input %d is null", i);
                return -1;
            }
        }
        if (outputs[kOUT_PRESENT_KV] == nullptr)
        {
            LOG_ERROR("DFlashTargetKVCacheUpdatePlugin: present_key_value output is null");
            return -1;
        }

        auto const* kDelta = static_cast<half const*>(inputs[kIN_K_DELTA]);
        auto const* vDelta = static_cast<half const*>(inputs[kIN_V_DELTA]);
        auto* kvCache = static_cast<half*>(outputs[kOUT_PRESENT_KV]);
        auto const* cosSinCache = static_cast<float const*>(inputs[kIN_ROPE_COS_SIN]);
        auto const* deltaStartPositions = static_cast<int32_t const*>(inputs[kIN_DELTA_START]);
        auto const* deltaLengths = static_cast<int32_t const*>(inputs[kIN_DELTA_LENGTHS]);

        kernel::launchDFlashTargetKVCacheUpdate(kDelta, vDelta, kvCache, cosSinCache, deltaStartPositions, deltaLengths,
            batchSize, deltaLen, numKVHeads, headDim, rotaryDim, cosSinBatch, cosSinSeqLen, maxBatch, cap, stream);

        return 0;
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("DFlashTargetKVCacheUpdatePlugin::enqueue exception: %s", e.what());
        return -1;
    }
}

int32_t DFlashTargetKVCacheUpdatePlugin::onShapeChange(PluginTensorDesc const* /* in */, int32_t /* nbInputs */,
    PluginTensorDesc const* /* out */, int32_t /* nbOutputs */) noexcept
{
    return 0;
}

IPluginV3* DFlashTargetKVCacheUpdatePlugin::attachToContext([[maybe_unused]] IPluginResourceContext* context) noexcept
{
    return clone();
}

PluginFieldCollection const* DFlashTargetKVCacheUpdatePlugin::getFieldsToSerialize() noexcept
{
    mDataToSerialize.clear();
    mDataToSerialize.emplace_back("pages_per_slot", &mPagesPerSlot, PluginFieldType::kINT32, 1);
    mFCToSerialize.nbFields = static_cast<int32_t>(mDataToSerialize.size());
    mFCToSerialize.fields = mDataToSerialize.data();
    return &mFCToSerialize;
}

void DFlashTargetKVCacheUpdatePlugin::setPluginNamespace(char const* pluginNamespace) noexcept
{
    mNamespace = pluginNamespace;
}

void DFlashTargetKVCacheUpdatePlugin::setPagesPerSlot(int32_t pagesPerSlot) noexcept
{
    mPagesPerSlot = pagesPerSlot;
}

// ---------------------------------------------------------------------------
// Plugin Creator
// ---------------------------------------------------------------------------

PluginFieldCollection DFlashTargetKVCacheUpdatePluginCreator::mFieldCollection{};
std::vector<PluginField> DFlashTargetKVCacheUpdatePluginCreator::mPluginAttributes{};

DFlashTargetKVCacheUpdatePluginCreator::DFlashTargetKVCacheUpdatePluginCreator()
{
    mFieldCollection.nbFields = 0;
    mFieldCollection.fields = nullptr;
}

char const* DFlashTargetKVCacheUpdatePluginCreator::getPluginName() const noexcept
{
    return kPLUGIN_NAME;
}

char const* DFlashTargetKVCacheUpdatePluginCreator::getPluginVersion() const noexcept
{
    return kPLUGIN_VERSION;
}

PluginFieldCollection const* DFlashTargetKVCacheUpdatePluginCreator::getFieldNames() noexcept
{
    return &mFieldCollection;
}

char const* DFlashTargetKVCacheUpdatePluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

void DFlashTargetKVCacheUpdatePluginCreator::setPluginNamespace(char const* pluginNamespace) noexcept
{
    mNamespace = pluginNamespace;
}

IPluginV3* DFlashTargetKVCacheUpdatePluginCreator::createPlugin(
    char const* name, PluginFieldCollection const* fc, [[maybe_unused]] TensorRTPhase phase) noexcept
{
    try
    {
        auto* plugin = new DFlashTargetKVCacheUpdatePlugin(name, fc);
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("DFlashTargetKVCacheUpdatePluginCreator: createPlugin exception: %s", e.what());
    }
    return nullptr;
}

// Register plugin
REGISTER_TENSORRT_PLUGIN(DFlashTargetKVCacheUpdatePluginCreator);

} // namespace plugins
} // namespace trt_edgellm

// ---------------------------------------------------------------------------
// Builder configuration hook (see cpp/common/trtUtils.h::loadEdgellmPluginLib)
// ---------------------------------------------------------------------------
//
// The builder (edgellmBuilder/llm_build) does not link this plugin's concrete class — plugins are
// opaque .so modules loaded via EDGELLM_PLUGIN_PATH, resolved through the generic IPluginV3
// interface only. pages_per_slot (see DFlashTargetKVCacheUpdatePlugin::setPagesPerSlot) is a
// builder-only fact (mBuilderConfig.maxKVCacheCapacity) with nowhere else to enter the ONNX-parsed
// network, so the builder resolves this exported hook via dlsym on the plugin library it already
// dlopen'd, instead of adding a new link dependency. EDGELLM_PLUGIN_EXPORT is required: the plugin
// builds with -fvisibility=hidden, so an unannotated extern "C" symbol would not reach .dynsym.
extern "C" EDGELLM_PLUGIN_EXPORT bool edgellm_dflash_configure_pages_per_slot(
    nvinfer1::INetworkDefinition* network, int32_t pagesPerSlot) noexcept
{
    if (network == nullptr || pagesPerSlot <= 0)
    {
        return false;
    }
    bool foundAny = false;
    for (int32_t i = 0; i < network->getNbLayers(); ++i)
    {
        nvinfer1::ILayer* layer = network->getLayer(i);
        if (layer == nullptr || layer->getType() != nvinfer1::LayerType::kPLUGIN_V3)
        {
            continue;
        }
        auto* pluginLayer = static_cast<nvinfer1::IPluginV3Layer*>(layer);
        auto* ours = dynamic_cast<trt_edgellm::plugins::DFlashTargetKVCacheUpdatePlugin*>(&pluginLayer->getPlugin());
        if (ours == nullptr)
        {
            continue;
        }
        ours->setPagesPerSlot(pagesPerSlot);
        foundAny = true;
    }
    return foundAny;
}
