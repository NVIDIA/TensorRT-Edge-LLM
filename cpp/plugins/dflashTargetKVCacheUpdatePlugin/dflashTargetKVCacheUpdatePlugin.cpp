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
constexpr int32_t kNUM_INPUTS{7};
constexpr int32_t kNUM_OUTPUTS{1};

bool haveSameShape(Dims const& lhs, Dims const& rhs)
{
    if (lhs.nbDims != rhs.nbDims)
    {
        return false;
    }
    for (int32_t dimIdx = 0; dimIdx < lhs.nbDims; ++dimIdx)
    {
        if (lhs.d[dimIdx] != rhs.d[dimIdx])
        {
            return false;
        }
    }
    return true;
}

bool isPagedPoolShape(Dims const& shape, Dims const& kDelta, bool allowUnknownNumPages)
{
    if (shape.nbDims != 5 || kDelta.nbDims != 4)
    {
        return false;
    }
    bool const validNumPages = allowUnknownNumPages ? (shape.d[1] == -1 || shape.d[1] > 0) : shape.d[1] > 0;
    return shape.d[0] == 2 && validNumPages && shape.d[2] == rt::kTOKENS_PER_PAGE && shape.d[3] == kDelta.d[2]
        && shape.d[4] == kDelta.d[3];
}

bool isCompatiblePageTableShape(
    Dims const& pageTable, Dims const& kDelta, Dims const& pastKV, bool allowUnknownDimensions)
{
    if (pageTable.nbDims != 3 || kDelta.nbDims != 4 || pastKV.nbDims != 5 || pageTable.d[1] != 2)
    {
        return false;
    }
    if (allowUnknownDimensions)
    {
        bool const matchingBatch = pageTable.d[0] == -1 || kDelta.d[0] == -1 || pageTable.d[0] == kDelta.d[0];
        bool const validMaxPages = pageTable.d[2] == -1 || pageTable.d[2] > 0;
        bool const maxPagesFitsPool = pageTable.d[2] == -1 || pastKV.d[1] == -1 || pageTable.d[2] <= pastKV.d[1];
        return matchingBatch && validMaxPages && maxPagesFitsPool;
    }
    return pageTable.d[0] == kDelta.d[0] && pageTable.d[2] > 0 && pageTable.d[2] <= pastKV.d[1];
}

bool hasConcretePagedKVContract(Dims const& kDelta, Dims const& vDelta, Dims const& pastKV, Dims const& deltaStart,
    Dims const& deltaLengths, Dims const& pageTable, Dims const& presentKV)
{
    return kDelta.nbDims == 4 && kDelta.d[0] > 0 && kDelta.d[1] > 0 && kDelta.d[2] > 0 && kDelta.d[3] > 0
        && haveSameShape(kDelta, vDelta) && isPagedPoolShape(pastKV, kDelta, false) && deltaStart.nbDims == 1
        && deltaStart.d[0] == kDelta.d[0] && deltaLengths.nbDims == 1 && deltaLengths.d[0] == kDelta.d[0]
        && haveSameShape(pastKV, presentKV) && isCompatiblePageTableShape(pageTable, kDelta, pastKV, false);
}

bool isLinear(PluginTensorDesc const& tensorDesc, DataType dataType)
{
    return tensorDesc.type == dataType && tensorDesc.format == TensorFormat::kLINEAR;
}

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
    static_cast<void>(fc);
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

    if (pos == kIN_K_DELTA || pos == kIN_V_DELTA)
    {
        return isLinearFormat && desc.desc.type == DataType::kHALF;
    }
    if (pos == kIN_PAST_KV)
    {
        return isLinearFormat && desc.desc.type == DataType::kHALF
            && isPagedPoolShape(desc.desc.dims, inOut[kIN_K_DELTA].desc.dims, true);
    }
    else if (pos == kIN_ROPE_COS_SIN)
    {
        // RoPE cos/sin must be FP32
        return isLinearFormat && desc.desc.type == DataType::kFLOAT;
    }
    else if (pos == kIN_DELTA_START || pos == kIN_DELTA_LENGTHS)
    {
        return isLinearFormat && desc.desc.type == DataType::kINT32;
    }
    else if (pos == kIN_KV_PAGE_TABLE)
    {
        return isLinearFormat && desc.desc.type == DataType::kINT32 && desc.desc.dims.nbDims == 3
            && isCompatiblePageTableShape(
                desc.desc.dims, inOut[kIN_K_DELTA].desc.dims, inOut[kIN_PAST_KV].desc.dims, true);
    }
    else if (pos == nbInputs + kOUT_PRESENT_KV)
    {
        return isLinearFormat && desc.desc.type == inOut[kIN_PAST_KV].desc.type
            && isPagedPoolShape(desc.desc.dims, inOut[kIN_K_DELTA].desc.dims, true);
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
    if (!isLinear(in[kIN_K_DELTA].desc, DataType::kHALF) || !isLinear(in[kIN_V_DELTA].desc, DataType::kHALF)
        || !isLinear(in[kIN_PAST_KV].desc, DataType::kHALF) || !isLinear(out[kOUT_PRESENT_KV].desc, DataType::kHALF))
    {
        LOG_ERROR("DFlashTargetKVCacheUpdatePlugin: K/V deltas and KV pool must be linear FP16");
        return -1;
    }
    if (!isLinear(in[kIN_ROPE_COS_SIN].desc, DataType::kFLOAT))
    {
        LOG_ERROR("DFlashTargetKVCacheUpdatePlugin: rope_cos_sin must be linear FP32");
        return -1;
    }
    if (!isLinear(in[kIN_DELTA_START].desc, DataType::kINT32) || !isLinear(in[kIN_DELTA_LENGTHS].desc, DataType::kINT32)
        || !isLinear(in[kIN_KV_PAGE_TABLE].desc, DataType::kINT32))
    {
        LOG_ERROR("DFlashTargetKVCacheUpdatePlugin: index and page-table inputs must be linear INT32");
        return -1;
    }
    if (in[kIN_K_DELTA].desc.dims.nbDims != 4 || in[kIN_V_DELTA].desc.dims.nbDims != 4
        || in[kIN_PAST_KV].desc.dims.nbDims != 5 || in[kIN_ROPE_COS_SIN].desc.dims.nbDims != 3
        || in[kIN_DELTA_START].desc.dims.nbDims != 1 || in[kIN_DELTA_LENGTHS].desc.dims.nbDims != 1
        || in[kIN_KV_PAGE_TABLE].desc.dims.nbDims != 3)
    {
        LOG_ERROR("DFlashTargetKVCacheUpdatePlugin: invalid input ranks");
        return -1;
    }
    bool const validProfiles
        = hasConcretePagedKVContract(in[kIN_K_DELTA].min, in[kIN_V_DELTA].min, in[kIN_PAST_KV].min,
              in[kIN_DELTA_START].min, in[kIN_DELTA_LENGTHS].min, in[kIN_KV_PAGE_TABLE].min, out[kOUT_PRESENT_KV].min)
        && hasConcretePagedKVContract(in[kIN_K_DELTA].opt, in[kIN_V_DELTA].opt, in[kIN_PAST_KV].opt,
            in[kIN_DELTA_START].opt, in[kIN_DELTA_LENGTHS].opt, in[kIN_KV_PAGE_TABLE].opt, out[kOUT_PRESENT_KV].opt)
        && hasConcretePagedKVContract(in[kIN_K_DELTA].max, in[kIN_V_DELTA].max, in[kIN_PAST_KV].max,
            in[kIN_DELTA_START].max, in[kIN_DELTA_LENGTHS].max, in[kIN_KV_PAGE_TABLE].max, out[kOUT_PRESENT_KV].max);
    if (!validProfiles)
    {
        LOG_ERROR(
            "DFlashTargetKVCacheUpdatePlugin: KV pool must be [2, N, %d, H, D], output must match it, and "
            "kv_page_table must be [B, 2, M] with B matching K/V deltas.",
            rt::kTOKENS_PER_PAGE);
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
    // and DSpark draft engines depend on this in-engine target KV update before
    // subsequent attention layers read the same KV cache.
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
        if (!hasConcretePagedKVContract(inputDesc[kIN_K_DELTA].dims, inputDesc[kIN_V_DELTA].dims,
                inputDesc[kIN_PAST_KV].dims, inputDesc[kIN_DELTA_START].dims, inputDesc[kIN_DELTA_LENGTHS].dims,
                inputDesc[kIN_KV_PAGE_TABLE].dims, outputDesc[kOUT_PRESENT_KV].dims))
        {
            LOG_ERROR(
                "DFlashTargetKVCacheUpdatePlugin: KV pool must be [2, N, %d, H, D], output must match it, and "
                "kv_page_table must be [B, 2, M] with B matching K/V deltas.",
                rt::kTOKENS_PER_PAGE);
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

        auto const& pastKVDesc = inputDesc[kIN_PAST_KV];
        if (pastKVDesc.type != DataType::kHALF || pastKVDesc.dims.nbDims != 5 || pastKVDesc.dims.d[0] != 2
            || pastKVDesc.dims.d[1] <= 0 || pastKVDesc.dims.d[2] != rt::kTOKENS_PER_PAGE
            || pastKVDesc.dims.d[3] != numKVHeads || pastKVDesc.dims.d[4] != headDim)
        {
            LOG_ERROR(
                "DFlashTargetKVCacheUpdatePlugin: past_key_value must be the paged pool "
                "[2, numPages, %d, numKVHeads, headDim] FP16",
                rt::kTOKENS_PER_PAGE);
            return -1;
        }
        int32_t const numPages = pastKVDesc.dims.d[1];

        auto const& pageTableDesc = inputDesc[kIN_KV_PAGE_TABLE];
        if (pageTableDesc.type != DataType::kINT32 || pageTableDesc.dims.nbDims != 3
            || pageTableDesc.dims.d[0] != batchSize || pageTableDesc.dims.d[1] != 2 || pageTableDesc.dims.d[2] <= 0)
        {
            LOG_ERROR(
                "DFlashTargetKVCacheUpdatePlugin: kv_page_table must be [B, 2, M] INT32 with B matching K/V deltas");
            return -1;
        }
        int32_t const maxPagesPerSeq = pageTableDesc.dims.d[2];
        int32_t const cap = maxPagesPerSeq * rt::kTOKENS_PER_PAGE;

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
        auto const* pageTable = static_cast<int32_t const*>(inputs[kIN_KV_PAGE_TABLE]);

        kernel::launchDFlashTargetKVCacheUpdate(kDelta, vDelta, kvCache, cosSinCache, deltaStartPositions, deltaLengths,
            pageTable, batchSize, deltaLen, numKVHeads, headDim, rotaryDim, cosSinBatch, cosSinSeqLen, numPages,
            maxPagesPerSeq, stream);

        return 0;
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("DFlashTargetKVCacheUpdatePlugin::enqueue exception: %s", e.what());
        return -1;
    }
}

int32_t DFlashTargetKVCacheUpdatePlugin::onShapeChange(
    PluginTensorDesc const* in, int32_t nbInputs, PluginTensorDesc const* out, int32_t nbOutputs) noexcept
{
    if (!checkExpectedIO("DFlashTargetKVCacheUpdatePlugin::onShapeChange", nbInputs, nbOutputs) || in == nullptr
        || out == nullptr)
    {
        return -1;
    }
    if (!isLinear(in[kIN_K_DELTA], DataType::kHALF) || !isLinear(in[kIN_V_DELTA], DataType::kHALF)
        || !isLinear(in[kIN_PAST_KV], DataType::kHALF) || !isLinear(out[kOUT_PRESENT_KV], DataType::kHALF)
        || !isLinear(in[kIN_ROPE_COS_SIN], DataType::kFLOAT) || !isLinear(in[kIN_DELTA_START], DataType::kINT32)
        || !isLinear(in[kIN_DELTA_LENGTHS], DataType::kINT32) || !isLinear(in[kIN_KV_PAGE_TABLE], DataType::kINT32)
        || !hasConcretePagedKVContract(in[kIN_K_DELTA].dims, in[kIN_V_DELTA].dims, in[kIN_PAST_KV].dims,
            in[kIN_DELTA_START].dims, in[kIN_DELTA_LENGTHS].dims, in[kIN_KV_PAGE_TABLE].dims,
            out[kOUT_PRESENT_KV].dims))
    {
        LOG_ERROR("DFlashTargetKVCacheUpdatePlugin: invalid concrete paged-KV or page-table contract.");
        return -1;
    }
    return 0;
}

IPluginV3* DFlashTargetKVCacheUpdatePlugin::attachToContext([[maybe_unused]] IPluginResourceContext* context) noexcept
{
    return clone();
}

PluginFieldCollection const* DFlashTargetKVCacheUpdatePlugin::getFieldsToSerialize() noexcept
{
    mDataToSerialize.clear();
    mFCToSerialize.nbFields = static_cast<int32_t>(mDataToSerialize.size());
    mFCToSerialize.fields = mDataToSerialize.data();
    return &mFCToSerialize;
}

void DFlashTargetKVCacheUpdatePlugin::setPluginNamespace(char const* pluginNamespace) noexcept
{
    mNamespace = pluginNamespace;
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
