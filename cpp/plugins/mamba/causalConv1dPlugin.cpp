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

#include "causalConv1dPlugin.h"

#include "common/cudaUtils.h"
#include "common/executionPhase.h"
#include "common/logger.h"
#include "common/trtInterop.h"
#include "kernels/compactPhase/compactPhaseSupport.h"
#include "kernels/mamba/causalConv1d.h"
#include "kernels/mamba/selectiveStateUpdate.h"
#include "plugins/utils/pluginUtils.h"
#include "plugins/utils/raggedPluginMetadata.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cuda_fp16.h>
#include <mutex>
#include <optional>
#include <stdexcept>

using namespace nvinfer1;

namespace trt_edgellm
{
namespace plugins
{
using nvinfer1::DataType;

namespace
{
constexpr char const* kCAUSAL_CONV_PLUGIN_VERSION{"1"};
constexpr char const* kCAUSAL_CONV_PLUGIN_NAME{"causal_conv1d"};

constexpr int32_t kIN_X_IDX{0};
constexpr int32_t kIN_WEIGHT_IDX{1};
constexpr int32_t kIN_BIAS_IDX{2};
constexpr int32_t kIN_CONV_STATE_IDX{3};
constexpr int32_t kIN_CONTEXT_LENGTHS_IDX{4};
constexpr int32_t kIN_QUERY_START_OFFSETS_IDX{5};
constexpr int32_t kIN_STATE_INDICES_IDX{6};
constexpr int32_t kIN_EXECUTION_PHASE_MARKER_IDX{7};
constexpr int32_t kIN_CONTEXT_SEQUENCE_COUNT_IDX{8};
constexpr int32_t kIN_TREE_PARENT_IDS_IDX{9};
constexpr int32_t kIN_TREE_DEPTHS_IDX{10};
constexpr int32_t kOUT_IDX{0};
constexpr int32_t kOUT_CONV_STATE_IDX{1};
constexpr int32_t kOUT_INTERMEDIATE_CONV_STATES{2};
constexpr int32_t kNUM_REQUIRED_INPUTS{9};
constexpr int32_t kIN_SPEC_VERIFY_LAYOUT_IDX{11};
constexpr int32_t kNUM_SPEC_VERIFY_OPTIONAL_INPUTS{3};
constexpr int32_t kNUM_REQUIRED_OUTPUTS{2};
constexpr int32_t kNUM_SPEC_VERIFY_OPTIONAL_OUTPUTS{1};

//! Convolution widths the kernels accept; 2/3/4 additionally have specialized instances.
constexpr int32_t kMIN_CONV_WIDTH{2};
constexpr int32_t kMAX_CONV_WIDTH{8};
constexpr int32_t kMAX_DILATION{4};
//! Longest conv state a kernel can hold in registers.
constexpr int32_t kMAX_STATE_LEN{16};

std::optional<int32_t> parsePluginIntField(std::string const& fieldName, PluginFieldCollection const* fc)
{
    for (int32_t i = 0; i < fc->nbFields; ++i)
    {
        PluginField const& pluginField = fc->fields[i];
        if (fieldName != pluginField.name || pluginField.length != 1 || pluginField.data == nullptr)
        {
            continue;
        }
        if (pluginField.type == PluginFieldType::kINT32)
        {
            return *static_cast<int32_t const*>(pluginField.data);
        }
        if (pluginField.type == PluginFieldType::kINT64)
        {
            return static_cast<int32_t>(*static_cast<int64_t const*>(pluginField.data));
        }
    }
    return std::nullopt;
}

//! Phase 8 workspace layout: dense history [numPrefill, dim, historyLen] FP16, then INT32 row indices and
//! hasInitialState, each [numPrefill].
struct CompactWorkspaceLayout
{
    size_t indexOffset;
    size_t initialStateOffset;
    size_t totalBytes;
};

constexpr size_t kCOMPACT_INDEX_ALIGNMENT{16};

size_t alignCompactIndexOffset(size_t bytes)
{
    return (bytes + kCOMPACT_INDEX_ALIGNMENT - 1) / kCOMPACT_INDEX_ALIGNMENT * kCOMPACT_INDEX_ALIGNMENT;
}

CompactWorkspaceLayout compactWorkspaceLayout(int32_t numPrefill, int32_t dim, int32_t historyLen)
{
    size_t const indexOffset
        = alignCompactIndexOffset(static_cast<size_t>(numPrefill) * dim * historyLen * sizeof(half));
    size_t const initialStateOffset
        = indexOffset + alignCompactIndexOffset(static_cast<size_t>(numPrefill) * sizeof(int32_t));
    return {indexOffset, initialStateOffset, initialStateOffset + static_cast<size_t>(numPrefill) * sizeof(int32_t)};
}

} // namespace

PluginFieldCollection CausalConv1dPluginCreator::mFieldCollection{};
std::vector<PluginField> CausalConv1dPluginCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(CausalConv1dPluginCreator);

// ---------------------------------------------------------------------------
// Plugin — construction / destruction
// ---------------------------------------------------------------------------

CausalConv1dPlugin::CausalConv1dPlugin(
    std::string const& name, int32_t stride, int32_t padding, int32_t dilation, int32_t groups, bool useSpecVerifyState)
    : mLayerName(name)
    , mStride(stride)
    , mPadding(padding)
    , mDilation(dilation)
    , mGroups(groups)
    , mSMVersion(getSMVersion())
    , mUseSpecVerifyState(useSpecVerifyState)
{
}

CausalConv1dPlugin::CausalConv1dPlugin(std::string const& name, PluginFieldCollection const* fc)
    : mLayerName(name)
{
    mStride = parsePluginIntField("stride", fc).value_or(1);
    mPadding = parsePluginIntField("padding", fc).value_or(0);
    mDilation = parsePluginIntField("dilation", fc).value_or(1);
    mGroups = parsePluginIntField("groups", fc).value_or(0);
    mSMVersion = getSMVersion();
    auto const verify = parsePluginIntField("use_spec_verify_state", fc);
    ELLM_CHECK(verify.has_value(), "causal_conv1d: missing verify capability; re-export and rebuild artifacts");
    mUseSpecVerifyState = verify.value() != 0;
}

CausalConv1dPlugin::~CausalConv1dPlugin() {}

// ---------------------------------------------------------------------------
// IPluginV3
// ---------------------------------------------------------------------------

IPluginCapability* CausalConv1dPlugin::getCapabilityInterface(PluginCapabilityType type) noexcept
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
        return static_cast<IPluginV3OneCore*>(this);
    }
    catch (std::exception const& e)
    {
        return nullptr;
    }
}

IPluginV3* CausalConv1dPlugin::clone() noexcept
{
    try
    {
        auto* plugin = new CausalConv1dPlugin(mLayerName, mStride, mPadding, mDilation, mGroups, mUseSpecVerifyState);
        plugin->mSMVersion = mSMVersion;
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }
    catch (std::exception const& e)
    {
        return nullptr;
    }
}

// ---------------------------------------------------------------------------
// IPluginV3OneCore — metadata
// ---------------------------------------------------------------------------

char const* CausalConv1dPlugin::getPluginName() const noexcept
{
    return kCAUSAL_CONV_PLUGIN_NAME;
}

char const* CausalConv1dPlugin::getPluginVersion() const noexcept
{
    return kCAUSAL_CONV_PLUGIN_VERSION;
}

char const* CausalConv1dPlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

void CausalConv1dPlugin::setPluginNamespace(char const* pluginNamespace) noexcept
{
    mNamespace = pluginNamespace ? pluginNamespace : "";
}

bool CausalConv1dPlugin::supportsCompactPhase(int32_t dim, int32_t width) const noexcept
{
    return causalConv1dCompactSupport(mSMVersion, dim, width, mDilation, mGroups, mUseSpecVerifyState).supported;
}

std::string CausalConv1dPlugin::describeLayer(int32_t dim, int32_t width) const
{
    std::string shape
        = std::to_string(dim) + " channels, width " + std::to_string(width) + ", dilation " + std::to_string(mDilation);
    if (mGroups != 0 && mGroups != dim)
    {
        shape += ", " + std::to_string(mGroups) + " groups";
    }
    if (mUseSpecVerifyState)
    {
        shape += ", spec-verify state";
    }
    return describePluginLayer("causal_conv1d", mLayerName, shape);
}

// ---------------------------------------------------------------------------
// IPluginV3OneBuild — shape / format
// ---------------------------------------------------------------------------

int32_t CausalConv1dPlugin::getNbOutputs() const noexcept
{
    return kNUM_REQUIRED_OUTPUTS + (mUseSpecVerifyState ? kNUM_SPEC_VERIFY_OPTIONAL_OUTPUTS : 0);
}

int32_t CausalConv1dPlugin::getOutputDataTypes(DataType* outputTypes, [[maybe_unused]] int32_t nbOutputs,
    DataType const* inputTypes, [[maybe_unused]] int32_t nbInputs) const noexcept
{
    try
    {
        [[maybe_unused]] int32_t const expectedNbOutputs
            = kNUM_REQUIRED_OUTPUTS + (mUseSpecVerifyState ? kNUM_SPEC_VERIFY_OPTIONAL_OUTPUTS : 0);
        assert(nbOutputs == expectedNbOutputs);
        outputTypes[kOUT_IDX] = inputTypes[kIN_X_IDX];
        outputTypes[kOUT_CONV_STATE_IDX] = inputTypes[kIN_X_IDX];
        if (mUseSpecVerifyState)
        {
            outputTypes[kOUT_INTERMEDIATE_CONV_STATES] = inputTypes[kIN_X_IDX];
        }
        return 0;
    }
    catch (std::exception const& e)
    {
        return -1;
    }
}

int32_t CausalConv1dPlugin::getOutputShapes(DimsExprs const* inputs, [[maybe_unused]] int32_t nbInputs,
    DimsExprs const* /* shapeInputs */, int32_t /* nbShapeInputs */, DimsExprs* outputs,
    [[maybe_unused]] int32_t nbOutputs, IExprBuilder& exprBuilder) noexcept
{
    try
    {
        [[maybe_unused]] int32_t const expectedNbOutputs
            = kNUM_REQUIRED_OUTPUTS + (mUseSpecVerifyState ? kNUM_SPEC_VERIFY_OPTIONAL_OUTPUTS : 0);
        [[maybe_unused]] int32_t const expectedNbInputs
            = kNUM_REQUIRED_INPUTS + (mUseSpecVerifyState ? kNUM_SPEC_VERIFY_OPTIONAL_INPUTS : 0);
        assert(nbInputs == expectedNbInputs);
        assert(nbOutputs == expectedNbOutputs);
        // Output: same shape as x [batch, seq_len, dim].
        outputs[kOUT_IDX] = inputs[kIN_X_IDX];
        // Conv state output: same shape as conv_state input [batch, dim, kernel].
        outputs[kOUT_CONV_STATE_IDX] = inputs[kIN_CONV_STATE_IDX];
        if (mUseSpecVerifyState)
        {
            auto const* zero = exprBuilder.constant(0);
            auto const* verifyProfile = exprBuilder.operation(
                DimensionOperation::kEQUAL, *inputs[kIN_CONTEXT_SEQUENCE_COUNT_IDX].d[0], *zero);
            outputs[kOUT_INTERMEDIATE_CONV_STATES].nbDims = 3;
            outputs[kOUT_INTERMEDIATE_CONV_STATES].d[0]
                = exprBuilder.operation(DimensionOperation::kPROD, *inputs[kIN_X_IDX].d[0], *verifyProfile);
            outputs[kOUT_INTERMEDIATE_CONV_STATES].d[1] = inputs[kIN_CONV_STATE_IDX].d[1];
            outputs[kOUT_INTERMEDIATE_CONV_STATES].d[2] = inputs[kIN_CONV_STATE_IDX].d[2];
        }
        return 0;
    }
    catch (std::exception const& e)
    {
        return -1;
    }
}

bool CausalConv1dPlugin::supportsFormatCombination(
    int32_t pos, DynamicPluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
    int32_t const expectedNbOutputs
        = kNUM_REQUIRED_OUTPUTS + (mUseSpecVerifyState ? kNUM_SPEC_VERIFY_OPTIONAL_OUTPUTS : 0);
    int32_t const expectedNbInputs
        = kNUM_REQUIRED_INPUTS + (mUseSpecVerifyState ? kNUM_SPEC_VERIFY_OPTIONAL_INPUTS : 0);
    if (nbInputs != expectedNbInputs || nbOutputs != expectedNbOutputs)
        return false;
    auto const& desc = inOut[pos].desc;
    if (desc.format != TensorFormat::kLINEAR)
        return false;
    // Token-major metadata bindings are INT32.
    if (pos == kIN_CONTEXT_LENGTHS_IDX)
        return desc.type == DataType::kINT32;
    if (pos == kIN_QUERY_START_OFFSETS_IDX || pos == kIN_STATE_INDICES_IDX || pos == kIN_EXECUTION_PHASE_MARKER_IDX)
        return desc.type == DataType::kINT32;
    if (pos == kIN_CONTEXT_SEQUENCE_COUNT_IDX)
        return desc.type == DataType::kINT32;
    if (mUseSpecVerifyState
        && (pos == kIN_TREE_PARENT_IDS_IDX || pos == kIN_TREE_DEPTHS_IDX || pos == kIN_SPEC_VERIFY_LAYOUT_IDX))
        return desc.type == DataType::kINT32;
    // Everything else (all inputs + all outputs): FP16
    return desc.type == DataType::kHALF;
}

int32_t CausalConv1dPlugin::configurePlugin(DynamicPluginTensorDesc const* in, int32_t nbInputs,
    [[maybe_unused]] DynamicPluginTensorDesc const* out, [[maybe_unused]] int32_t nbOutputs) noexcept
{
    int32_t const expectedNbInputs
        = kNUM_REQUIRED_INPUTS + (mUseSpecVerifyState ? kNUM_SPEC_VERIFY_OPTIONAL_INPUTS : 0);
    if (nbInputs != expectedNbInputs)
    {
        LOG_ERROR("causal_conv1d: expected %d inputs, got %d", expectedNbInputs, nbInputs);
        return -1;
    }
    if (in[kIN_X_IDX].desc.type != DataType::kHALF)
    {
        LOG_ERROR(
            "causal_conv1d: only FP16 input is supported; got type %d", static_cast<int32_t>(in[kIN_X_IDX].desc.type));
        return -1;
    }
    if (in[kIN_X_IDX].desc.dims.nbDims != 2 || in[kIN_CONTEXT_LENGTHS_IDX].desc.dims.nbDims != 1
        || in[kIN_QUERY_START_OFFSETS_IDX].desc.dims.nbDims != 1 || in[kIN_STATE_INDICES_IDX].desc.dims.nbDims != 1
        || in[kIN_EXECUTION_PHASE_MARKER_IDX].desc.dims.nbDims != 1
        || in[kIN_CONTEXT_SEQUENCE_COUNT_IDX].desc.dims.nbDims != 1
        || in[kIN_CONTEXT_LENGTHS_IDX].desc.type != DataType::kINT32
        || in[kIN_QUERY_START_OFFSETS_IDX].desc.type != DataType::kINT32
        || in[kIN_STATE_INDICES_IDX].desc.type != DataType::kINT32
        || in[kIN_EXECUTION_PHASE_MARKER_IDX].desc.type != DataType::kINT32
        || in[kIN_CONTEXT_SEQUENCE_COUNT_IDX].desc.type != DataType::kINT32)
    {
        LOG_ERROR("causal_conv1d requires rank-2 x and INT32 metadata");
        return -1;
    }
    if (mUseSpecVerifyState
        && (in[kIN_TREE_PARENT_IDS_IDX].desc.type != DataType::kINT32
            || in[kIN_TREE_DEPTHS_IDX].desc.type != DataType::kINT32
            || in[kIN_TREE_PARENT_IDS_IDX].desc.dims.nbDims != 1 || in[kIN_TREE_DEPTHS_IDX].desc.dims.nbDims != 1
            || in[kIN_SPEC_VERIFY_LAYOUT_IDX].desc.type != DataType::kINT32
            || in[kIN_SPEC_VERIFY_LAYOUT_IDX].desc.dims.nbDims != 1))
    {
        LOG_ERROR("causal_conv1d: tree_parent_ids, tree_depths and spec_verify_layout must be 1D INT32");
        return -1;
    }
    if (in[kIN_WEIGHT_IDX].desc.dims.nbDims != 3 || in[kIN_CONV_STATE_IDX].desc.dims.nbDims != 3)
    {
        LOG_ERROR("causal_conv1d requires rank-3 weight [dim, 1, width] and conv_state [rows, dim, state_len]");
        return -1;
    }
    int32_t const width = static_cast<int32_t>(in[kIN_WEIGHT_IDX].desc.dims.d[2]);
    int32_t const stateLen = static_cast<int32_t>(in[kIN_CONV_STATE_IDX].desc.dims.d[2]);
    if (width < kMIN_CONV_WIDTH || width > kMAX_CONV_WIDTH)
    {
        LOG_ERROR("causal_conv1d: kernel width must be in [%d, %d], got %d", kMIN_CONV_WIDTH, kMAX_CONV_WIDTH, width);
        return -1;
    }
    if (mDilation < 1 || mDilation > kMAX_DILATION)
    {
        LOG_ERROR("causal_conv1d: dilation must be in [1, %d], got %d", kMAX_DILATION, mDilation);
        return -1;
    }
    if (stateLen < (width - 1) * mDilation || stateLen > kMAX_STATE_LEN)
    {
        LOG_ERROR("causal_conv1d: conv_state length must be in [(width-1)*dilation, %d] = [%d, %d], got %d",
            kMAX_STATE_LEN, (width - 1) * mDilation, kMAX_STATE_LEN, stateLen);
        return -1;
    }
    if ((width - 1) * mDilation + 1 > kMAX_STATE_LEN)
    {
        LOG_ERROR("causal_conv1d: dilated receptive field (width-1)*dilation+1 must be <= %d, got %d", kMAX_STATE_LEN,
            (width - 1) * mDilation + 1);
        return -1;
    }
    // Prefill left-pads by the full receptive field; anything else desynchronizes the output from
    // the token grid that getOutputShapes promises.
    if (mPadding != (width - 1) * mDilation)
    {
        LOG_ERROR("causal_conv1d: padding must be (width-1)*dilation = %d, got %d", (width - 1) * mDilation, mPadding);
        return -1;
    }
    return 0;
}

size_t CausalConv1dPlugin::getWorkspaceSize(DynamicPluginTensorDesc const* inputs, int32_t nbInputs,
    DynamicPluginTensorDesc const* /* outputs */, int32_t nbOutputs) const noexcept
{
    int32_t const expectedNbInputs
        = kNUM_REQUIRED_INPUTS + (mUseSpecVerifyState ? kNUM_SPEC_VERIFY_OPTIONAL_INPUTS : 0);
    if (inputs == nullptr || nbInputs != expectedNbInputs || nbOutputs != getNbOutputs())
    {
        LOG_ERROR("causal_conv1d: invalid workspace input/output count");
        return 0;
    }
    int32_t const maxBatch = static_cast<int32_t>(inputs[kIN_CONTEXT_LENGTHS_IDX].max.d[0]);
    int32_t const dim = static_cast<int32_t>(inputs[kIN_X_IDX].max.d[1]);
    int32_t const width = static_cast<int32_t>(inputs[kIN_WEIGHT_IDX].max.d[2]);
    int32_t const stateLen = static_cast<int32_t>(inputs[kIN_CONV_STATE_IDX].max.d[2]);
    size_t const paddedBytes = alignTensorSize(static_cast<size_t>(maxBatch) * dim * stateLen * sizeof(half));
    bool const reservesCompact = supportsCompactPhase(dim, width);
    reportCompactWorkspace("causal_conv1d", mLayerName, reservesCompact);
    if (!reservesCompact)
    {
        return paddedBytes;
    }
    return std::max(paddedBytes, compactWorkspaceLayout(maxBatch, dim, width - 1).totalBytes);
}

int32_t CausalConv1dPlugin::getAliasedInput([[maybe_unused]] int32_t outputIndex) noexcept
{
    // Myelin materializes a redundant per-layer copy for a declared read-write
    // alias. The runtime binds both tensors to the same resident state pool.
    return -1;
}

// ---------------------------------------------------------------------------
// IPluginV3OneRuntime — execution
// ---------------------------------------------------------------------------

int32_t CausalConv1dPlugin::enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc,
    void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    PluginTensorDesc const& phaseMarkerDesc = inputDesc[kIN_EXECUTION_PHASE_MARKER_IDX];
    if (phaseMarkerDesc.dims.nbDims == 1
        && phaseMarkerDesc.dims.d[0] == static_cast<int64_t>(rt::ExecutionPhase::kMixedPrefillDecode))
    {
        return enqueueCompact(inputDesc, outputDesc, inputs, outputs, workspace, stream);
    }
    auto const& xDesc = inputDesc[kIN_X_IDX];
    auto const& wDesc = inputDesc[kIN_WEIGHT_IDX];
    auto const& outDesc = outputDesc[kOUT_IDX];

    int32_t const expectedRank = 2;
    if (xDesc.dims.nbDims != expectedRank || wDesc.dims.nbDims != 3 || outDesc.dims.nbDims != expectedRank)
    {
        LOG_ERROR("causal_conv1d expects rank-%d x/output and rank-3 weight.", expectedRank);
        return 1;
    }

    RaggedPluginMetadata ragged{};
    try
    {
        ragged = decodeRaggedPluginMetadata("causal_conv1d", xDesc, inputDesc[kIN_CONTEXT_LENGTHS_IDX],
            inputDesc[kIN_QUERY_START_OFFSETS_IDX], inputDesc[kIN_EXECUTION_PHASE_MARKER_IDX],
            inputDesc[kIN_CONTEXT_SEQUENCE_COUNT_IDX]);
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("%s", e.what());
        return 1;
    }
    int32_t batch = ragged.numSequences;
    int32_t seqLen = 0;
    int32_t const dim = static_cast<int32_t>(xDesc.dims.d[expectedRank - 1]);
    int32_t const stateLen = static_cast<int32_t>(inputDesc[kIN_CONV_STATE_IDX].dims.d[2]);
    int32_t statePoolRows{};
    try
    {
        statePoolRows
            = validateIndexedResidentStateDescriptors("causal_conv1d", batch, inputDesc[kIN_STATE_INDICES_IDX],
                inputDesc[kIN_CONV_STATE_IDX], outputDesc[kOUT_CONV_STATE_IDX], xDesc.type, {dim, stateLen});
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("%s", e.what());
        return 1;
    }
    int32_t const tokenCount = ragged.physicalTokens;
    if (batch <= 0 || tokenCount % batch != 0 || inputDesc[kIN_QUERY_START_OFFSETS_IDX].dims.d[0] != batch + 1
        || outputs[kOUT_CONV_STATE_IDX] != inputs[kIN_CONV_STATE_IDX])
    {
        LOG_ERROR("causal_conv1d: invalid token-major extents or non-aliased resident state");
        return 1;
    }
    seqLen = tokenCount / batch;

    int32_t const groups = mGroups == 0 ? dim : mGroups;
    if (groups != dim)
    {
        LOG_ERROR("causal_conv1d currently supports depthwise conv only: groups=%d, dim=%d", groups, dim);
        return 1;
    }

    void* convStateOut = outputs[kOUT_CONV_STATE_IDX];

    constexpr int32_t kLinearSpecVerifyMaxSeqLen{16};
    rt::ExecutionPhase const phase = ragged.phase;
    bool const verifyActive = phase == rt::ExecutionPhase::kSpecTargetVerify;
    bool const diffusionDenoise = phase == rt::ExecutionPhase::kDiffusionDenoise;
    bool const contextLike = phase == rt::ExecutionPhase::kContextPrefill || phase == rt::ExecutionPhase::kContextChunk
        || diffusionDenoise || phase == rt::ExecutionPhase::kDiffusionCommit;
    bool const proposalActive = phase == rt::ExecutionPhase::kSpecDraftProposal;
    bool const decodeActive = phase == rt::ExecutionPhase::kAutoregressiveDecode;
    if (decodeActive && seqLen != 1)
    {
        LOG_ERROR("causal_conv1d: AUTOREGRESSIVE_DECODE requires one physical token per sequence, got %d", seqLen);
        return 1;
    }
    if (verifyActive && !mUseSpecVerifyState)
    {
        LOG_ERROR("causal_conv1d: SPEC_TARGET_VERIFY requires verify-capable engine metadata");
        return 1;
    }
    std::optional<rt::SpecVerifyLayout> layout;
    try
    {
        if (mUseSpecVerifyState)
        {
            layout = decodeSpecVerifyLayout("causal_conv1d", inputDesc[kIN_SPEC_VERIFY_LAYOUT_IDX], phase);
        }
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("%s", e.what());
        return -1;
    }
    bool const ddtreeActive = layout == rt::SpecVerifyLayout::kTree;
    if (ddtreeActive)
    {
        PluginTensorDesc const& parentDesc = inputDesc[kIN_TREE_PARENT_IDS_IDX];
        PluginTensorDesc const& depthDesc = inputDesc[kIN_TREE_DEPTHS_IDX];
        if (seqLen < 1 || parentDesc.dims.nbDims != 1 || depthDesc.dims.nbDims != 1
            || parentDesc.dims.d[0] != tokenCount || depthDesc.dims.d[0] != tokenCount)
        {
            LOG_ERROR("causal_conv1d: DDTree requires flat tree_parent_ids/tree_depths shape [T_exec=%d]", tokenCount);
            return 1;
        }
    }

    bool const mtpActive = mUseSpecVerifyState && verifyActive && !ddtreeActive;
    if (mtpActive && (seqLen < 1 || seqLen > kLinearSpecVerifyMaxSeqLen))
    {
        LOG_ERROR("causal_conv1d: linear spec-verify kernel supports seqLen in [1, %d], got %d",
            kLinearSpecVerifyMaxSeqLen, seqLen);
        return 1;
    }

    namespace rt = trt_edgellm::rt;
    auto stateIndicesTensor = rt::Tensor{const_cast<void*>(inputs[kIN_STATE_INDICES_IDX]), rt::Coords{batch},
        rt::DeviceType::kGPU, trt_edgellm::DataType::kINT32};
    rt::OptionalInputTensor stateIndicesOpt = std::optional(std::cref(stateIndicesTensor));

    std::optional<rt::Tensor> denoiseStateTensor;
    if (diffusionDenoise)
    {
        if (workspace == nullptr)
        {
            LOG_ERROR("causal_conv1d: DIFFUSION_DENOISE requires transactional state workspace");
            return 1;
        }
        auto residentStateView = rt::Tensor{const_cast<void*>(inputs[kIN_CONV_STATE_IDX]),
            rt::Coords{statePoolRows, dim, 1, stateLen}, rt::DeviceType::kGPU, fromTrt(xDesc.type)};
        auto activeStateView
            = rt::Tensor{workspace, rt::Coords{batch, dim, 1, stateLen}, rt::DeviceType::kGPU, fromTrt(xDesc.type)};
        mamba_ssm::invokeMambaStateGather(residentStateView, activeStateView, stateIndicesTensor, stream);
        denoiseStateTensor.emplace(
            workspace, rt::Coords{batch, dim, stateLen}, rt::DeviceType::kGPU, fromTrt(xDesc.type));
        stateIndicesOpt = std::nullopt;
    }

    if (mtpActive)
    {
        auto mtpStateTensor = rt::Tensor{const_cast<void*>(inputs[kIN_CONV_STATE_IDX]),
            rt::Coords{statePoolRows, dim, stateLen}, rt::DeviceType::kGPU, fromTrt(xDesc.type)};
        auto mtpNewColsTensor = rt::Tensor{const_cast<void*>(inputs[kIN_X_IDX]), rt::Coords{batch, seqLen, dim},
            rt::DeviceType::kGPU, fromTrt(xDesc.type)};
        auto mtpWeightTensor = rt::Tensor{const_cast<void*>(inputs[kIN_WEIGHT_IDX]),
            rt::Coords{wDesc.dims.d[0], wDesc.dims.d[1], wDesc.dims.d[2]}, rt::DeviceType::kGPU, fromTrt(xDesc.type)};
        auto mtpBiasTensor = rt::Tensor{
            const_cast<void*>(inputs[kIN_BIAS_IDX]), rt::Coords{dim}, rt::DeviceType::kGPU, fromTrt(xDesc.type)};
        auto mtpOutTensor
            = rt::Tensor{outputs[kOUT_IDX], rt::Coords{batch, seqLen, dim}, rt::DeviceType::kGPU, fromTrt(xDesc.type)};
        auto mtpIntermTensor = rt::Tensor{outputs[kOUT_INTERMEDIATE_CONV_STATES],
            rt::Coords{batch, seqLen, dim, stateLen}, rt::DeviceType::kGPU, fromTrt(xDesc.type)};

        trt_edgellm::rt::OptionalInputTensor mtpBiasOpt = std::optional(std::cref(mtpBiasTensor));
        bool const parallelChainActive = mamba_ssm::canUseCausalConv1dDecodeChainParallel(
            mSMVersion, batch, seqLen, wDesc.dims.d[2], mDilation, stateLen, mStride);
        if (parallelChainActive)
        {
            mamba_ssm::invokeCausalConv1dDecodeChainParallel(mtpStateTensor, mtpNewColsTensor, mtpWeightTensor,
                mtpBiasOpt, mtpOutTensor, mtpIntermTensor, stateIndicesOpt, stream);
        }
        else
        {
            mamba_ssm::invokeCausalConv1dDecodeMTP(mtpStateTensor, mtpNewColsTensor, mtpWeightTensor, mtpBiasOpt,
                mtpOutTensor, mtpIntermTensor, seqLen, mDilation, stateIndicesOpt, stream);
        }
    }
    else if (ddtreeActive)
    {
        auto treeStateTensor = rt::Tensor{const_cast<void*>(inputs[kIN_CONV_STATE_IDX]),
            rt::Coords{statePoolRows, dim, stateLen}, rt::DeviceType::kGPU, fromTrt(xDesc.type)};
        auto treeNewColsTensor = rt::Tensor{const_cast<void*>(inputs[kIN_X_IDX]), rt::Coords{batch, seqLen, dim},
            rt::DeviceType::kGPU, fromTrt(xDesc.type)};
        auto treeWeightTensor = rt::Tensor{const_cast<void*>(inputs[kIN_WEIGHT_IDX]),
            rt::Coords{wDesc.dims.d[0], wDesc.dims.d[1], wDesc.dims.d[2]}, rt::DeviceType::kGPU, fromTrt(xDesc.type)};
        auto treeBiasTensor = rt::Tensor{
            const_cast<void*>(inputs[kIN_BIAS_IDX]), rt::Coords{dim}, rt::DeviceType::kGPU, fromTrt(xDesc.type)};
        auto treeOutTensor
            = rt::Tensor{outputs[kOUT_IDX], rt::Coords{batch, seqLen, dim}, rt::DeviceType::kGPU, fromTrt(xDesc.type)};
        auto treeStateOutTensor
            = rt::Tensor{convStateOut, rt::Coords{batch, dim, stateLen}, rt::DeviceType::kGPU, fromTrt(xDesc.type)};
        auto treeIntermTensor = rt::Tensor{outputs[kOUT_INTERMEDIATE_CONV_STATES],
            rt::Coords{batch, seqLen, dim, stateLen}, rt::DeviceType::kGPU, fromTrt(xDesc.type)};
        auto treeParentTensor = rt::Tensor{const_cast<void*>(inputs[kIN_TREE_PARENT_IDS_IDX]),
            rt::Coords{batch, seqLen}, rt::DeviceType::kGPU, trt_edgellm::DataType::kINT32};
        auto treeDepthTensor = rt::Tensor{const_cast<void*>(inputs[kIN_TREE_DEPTHS_IDX]), rt::Coords{batch, seqLen},
            rt::DeviceType::kGPU, trt_edgellm::DataType::kINT32};

        trt_edgellm::rt::OptionalInputTensor treeBiasOpt = std::optional(std::cref(treeBiasTensor));
        mamba_ssm::invokeCausalConv1dDecodeDDTree(treeStateTensor, treeNewColsTensor, treeWeightTensor, treeBiasOpt,
            treeOutTensor, treeStateOutTensor, treeIntermTensor, treeParentTensor, treeDepthTensor, mDilation,
            stateIndicesOpt, stream);
    }
    else if (contextLike || (proposalActive && seqLen > 1))
    {
        // Sequential state update for context-like phases and multi-row proposal/commit.
        int32_t const outSeqLen = seqLen;

        auto xTensor = rt::Tensor{const_cast<void*>(inputs[kIN_X_IDX]), rt::Coords{batch, seqLen, dim},
            rt::DeviceType::kGPU, fromTrt(xDesc.type)};
        auto weightTensor = rt::Tensor{const_cast<void*>(inputs[kIN_WEIGHT_IDX]),
            rt::Coords{wDesc.dims.d[0], wDesc.dims.d[1], wDesc.dims.d[2]}, rt::DeviceType::kGPU, fromTrt(xDesc.type)};
        auto biasTensor = rt::Tensor{
            const_cast<void*>(inputs[kIN_BIAS_IDX]), rt::Coords{dim}, rt::DeviceType::kGPU, fromTrt(xDesc.type)};
        auto outTensor = rt::Tensor{
            outputs[kOUT_IDX], rt::Coords{batch, outSeqLen, dim}, rt::DeviceType::kGPU, fromTrt(xDesc.type)};

        auto contextLengthsTensor = rt::Tensor{const_cast<void*>(inputs[kIN_CONTEXT_LENGTHS_IDX]), rt::Coords{batch},
            rt::DeviceType::kGPU, trt_edgellm::DataType::kINT32};
        trt_edgellm::rt::OptionalInputTensor contextLengthsOpt = std::optional(std::cref(contextLengthsTensor));
        auto residentStateTensor = rt::Tensor{
            convStateOut, rt::Coords{statePoolRows, dim, stateLen}, rt::DeviceType::kGPU, fromTrt(xDesc.type)};
        rt::Tensor& activeStateTensor = diffusionDenoise ? denoiseStateTensor.value() : residentStateTensor;
        trt_edgellm::rt::OptionalInputTensor initialStateOpt = std::optional(std::cref(activeStateTensor));

        trt_edgellm::rt::OptionalInputTensor biasOpt = std::optional(std::cref(biasTensor));
        mamba_ssm::invokeCausalConv1d(xTensor, weightTensor, biasOpt, outTensor, mStride, mPadding, mDilation,
            initialStateOpt, contextLengthsOpt, stateIndicesOpt, stream);

        auto captureXTensor = rt::Tensor{const_cast<void*>(inputs[kIN_X_IDX]), rt::Coords{batch, seqLen, dim},
            rt::DeviceType::kGPU, fromTrt(xDesc.type)};
        mamba_ssm::invokeCaptureConvState(
            captureXTensor, initialStateOpt, activeStateTensor, contextLengthsOpt, stateIndicesOpt, stream);
    }
    else
    {
        // Single-row decode or proposal state update.
        enqueueDecodeRows(inputs, outputs, 0, 0, batch, dim, stateLen, statePoolRows, wDesc.dims, xDesc.type, stream);
    }

    return 0;
}

void CausalConv1dPlugin::enqueueDecodeRows(void const* const* inputs, void* const* outputs, int32_t firstRow,
    int32_t firstSequence, int32_t numSequences, int32_t dim, int32_t stateLen, int32_t statePoolRows,
    Dims const& weightDims, DataType type, cudaStream_t stream) const
{
    int64_t const rowOffset = static_cast<int64_t>(firstRow) * dim;
    auto decodeStateTensor = rt::Tensor{
        outputs[kOUT_CONV_STATE_IDX], rt::Coords{statePoolRows, dim, stateLen}, rt::DeviceType::kGPU, fromTrt(type)};
    auto decodeNewColTensor = rt::Tensor{static_cast<half*>(const_cast<void*>(inputs[kIN_X_IDX])) + rowOffset,
        rt::Coords{numSequences, 1, dim}, rt::DeviceType::kGPU, fromTrt(type)};
    auto decodeWeightTensor = rt::Tensor{const_cast<void*>(inputs[kIN_WEIGHT_IDX]),
        rt::Coords{weightDims.d[0], weightDims.d[1], weightDims.d[2]}, rt::DeviceType::kGPU, fromTrt(type)};
    auto decodeBiasTensor
        = rt::Tensor{const_cast<void*>(inputs[kIN_BIAS_IDX]), rt::Coords{dim}, rt::DeviceType::kGPU, fromTrt(type)};
    auto decodeOutTensor = rt::Tensor{static_cast<half*>(outputs[kOUT_IDX]) + rowOffset,
        rt::Coords{numSequences, 1, dim}, rt::DeviceType::kGPU, fromTrt(type)};
    auto decodeStateIndices
        = rt::Tensor{static_cast<int32_t*>(const_cast<void*>(inputs[kIN_STATE_INDICES_IDX])) + firstSequence,
            rt::Coords{numSequences}, rt::DeviceType::kGPU, trt_edgellm::DataType::kINT32};
    rt::OptionalInputTensor const decodeBiasOpt = std::optional(std::cref(decodeBiasTensor));
    rt::OptionalInputTensor const decodeStateIndicesOpt = std::optional(std::cref(decodeStateIndices));
    mamba_ssm::invokeCausalConv1dDecode(decodeStateTensor, decodeNewColTensor, decodeWeightTensor, decodeBiasOpt,
        decodeOutTensor, mDilation, decodeStateIndicesOpt, stream);
}

int32_t CausalConv1dPlugin::enqueueCompact(PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc,
    void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    auto const& xDesc = inputDesc[kIN_X_IDX];
    auto const& wDesc = inputDesc[kIN_WEIGHT_IDX];
    int32_t const dim = static_cast<int32_t>(xDesc.dims.d[1]);
    int32_t const width = static_cast<int32_t>(wDesc.dims.d[2]);
    if (!supportsCompactPhase(dim, width))
    {
        LOG_ERROR(
            "%s: compact execution is not supported for this layer on this GPU.", describeLayer(dim, width).c_str());
        return -1;
    }
    try
    {
        int32_t const stateLen = static_cast<int32_t>(inputDesc[kIN_CONV_STATE_IDX].dims.d[2]);
        RaggedPluginMetadata const ragged
            = decodeRaggedPluginMetadata("causal_conv1d", xDesc, inputDesc[kIN_CONTEXT_LENGTHS_IDX],
                inputDesc[kIN_QUERY_START_OFFSETS_IDX], inputDesc[kIN_EXECUTION_PHASE_MARKER_IDX],
                inputDesc[kIN_CONTEXT_SEQUENCE_COUNT_IDX], RaggedLayoutSupport::kPaddedAndCompact);
        int32_t const statePoolRows = validateIndexedResidentStateDescriptors("causal_conv1d", ragged.numSequences,
            inputDesc[kIN_STATE_INDICES_IDX], inputDesc[kIN_CONV_STATE_IDX], outputDesc[kOUT_CONV_STATE_IDX],
            xDesc.type, {dim, stateLen});
        check::check(workspace != nullptr && outputs[kOUT_CONV_STATE_IDX] == inputs[kIN_CONV_STATE_IDX],
            "causal_conv1d: compact execution requires workspace and an aliased resident state pool.");
        int32_t const numPrefill = ragged.numContextSequences;
        int32_t const numDecode = ragged.numDecodeSequences;
        int32_t const numPrefillTokens = ragged.contextExecutionRows;
        if (numPrefill > 0)
        {
            int32_t const historyLen = width - 1;
            DataType const type = xDesc.type;

            auto residentStates = rt::Tensor{outputs[kOUT_CONV_STATE_IDX], rt::Coords{statePoolRows, dim, stateLen},
                rt::DeviceType::kGPU, fromTrt(type)};
            auto weightTensor = rt::Tensor{const_cast<void*>(inputs[kIN_WEIGHT_IDX]),
                rt::Coords{wDesc.dims.d[0], wDesc.dims.d[1], wDesc.dims.d[2]}, rt::DeviceType::kGPU, fromTrt(type)};
            auto biasTensor = rt::Tensor{
                const_cast<void*>(inputs[kIN_BIAS_IDX]), rt::Coords{dim}, rt::DeviceType::kGPU, fromTrt(type)};
            rt::OptionalInputTensor const biasOpt = std::optional(std::cref(biasTensor));
            auto* const stateIndices = static_cast<int32_t*>(const_cast<void*>(inputs[kIN_STATE_INDICES_IDX]));
            auto* const xRows = static_cast<half*>(const_cast<void*>(inputs[kIN_X_IDX]));
            auto* const outRows = static_cast<half*>(outputs[kOUT_IDX]);

            CompactWorkspaceLayout const layout = compactWorkspaceLayout(numPrefill, dim, historyLen);
            auto* const base = static_cast<char*>(workspace);
            auto historyStates
                = rt::Tensor{base, rt::Coords{numPrefill, dim, historyLen}, rt::DeviceType::kGPU, fromTrt(type)};
            auto historyIndices = rt::Tensor{
                base + layout.indexOffset, rt::Coords{numPrefill}, rt::DeviceType::kGPU, trt_edgellm::DataType::kINT32};
            auto hasInitialState = rt::Tensor{base + layout.initialStateOffset, rt::Coords{numPrefill},
                rt::DeviceType::kGPU, trt_edgellm::DataType::kINT32};
            auto prefillStateIndices
                = rt::Tensor{stateIndices, rt::Coords{numPrefill}, rt::DeviceType::kGPU, trt_edgellm::DataType::kINT32};
            auto prefillOffsets = rt::Tensor{const_cast<void*>(inputs[kIN_QUERY_START_OFFSETS_IDX]),
                rt::Coords{numPrefill + 1}, rt::DeviceType::kGPU, trt_edgellm::DataType::kINT32};
            auto prefillX = rt::Tensor{xRows, rt::Coords{numPrefillTokens, dim}, rt::DeviceType::kGPU, fromTrt(type)};
            auto prefillOut
                = rt::Tensor{outRows, rt::Coords{numPrefillTokens, dim}, rt::DeviceType::kGPU, fromTrt(type)};

            // New sequences start from a zeroed or restored resident state, so every in-range history row is initial
            // state; out-of-range slots run without one and are not written back.
            mamba_ssm::invokeGatherConvStateHistory(
                residentStates, prefillStateIndices, historyStates, historyIndices, hasInitialState, stream);
            mamba_ssm::invokeCausalConv1dPacked(prefillX, prefillOffsets, historyIndices, hasInitialState, weightTensor,
                biasOpt, prefillOut, historyStates, /*padSlotId=*/-1, mStride, mPadding, mDilation, stream);
            mamba_ssm::invokeCapturePackedConvState(
                prefillX, prefillOffsets, prefillStateIndices, residentStates, stream);
        }
        if (numDecode > 0)
        {
            enqueueDecodeRows(inputs, outputs, numPrefillTokens, numPrefill, numDecode, dim, stateLen, statePoolRows,
                wDesc.dims, xDesc.type, stream);
        }
        return 0;
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("%s", e.what());
        return 1;
    }
}

int32_t CausalConv1dPlugin::onShapeChange(
    PluginTensorDesc const* in, int32_t nbInputs, PluginTensorDesc const* out, int32_t nbOutputs) noexcept
{
    int32_t const expectedNbInputs
        = kNUM_REQUIRED_INPUTS + (mUseSpecVerifyState ? kNUM_SPEC_VERIFY_OPTIONAL_INPUTS : 0);
    return in != nullptr && out != nullptr && nbInputs == expectedNbInputs && nbOutputs == getNbOutputs() ? 0 : -1;
}

IPluginV3* CausalConv1dPlugin::attachToContext(IPluginResourceContext* /* context */) noexcept
{
    return clone();
}

// ---------------------------------------------------------------------------
// Serialization
// ---------------------------------------------------------------------------

PluginFieldCollection const* CausalConv1dPlugin::getFieldsToSerialize() noexcept
{
    mDataToSerialize.clear();
    mDataToSerialize.emplace_back("stride", &mStride, PluginFieldType::kINT32, 1);
    mDataToSerialize.emplace_back("padding", &mPadding, PluginFieldType::kINT32, 1);
    mDataToSerialize.emplace_back("dilation", &mDilation, PluginFieldType::kINT32, 1);
    mDataToSerialize.emplace_back("groups", &mGroups, PluginFieldType::kINT32, 1);
    mUseSpecVerifyStateField = mUseSpecVerifyState ? 1 : 0;
    mDataToSerialize.emplace_back("use_spec_verify_state", &mUseSpecVerifyStateField, PluginFieldType::kINT32, 1);
    mDataToSerialize.emplace_back(layerNameField(mLayerName));

    mFCToSerialize.nbFields = mDataToSerialize.size();
    mFCToSerialize.fields = mDataToSerialize.data();
    return &mFCToSerialize;
}

// ---------------------------------------------------------------------------
// Creator
// ---------------------------------------------------------------------------

CausalConv1dPluginCreator::CausalConv1dPluginCreator()
{
    static std::mutex sMutex;
    std::lock_guard<std::mutex> lock(sMutex);
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("stride", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("padding", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("dilation", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("groups", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("use_spec_verify_state", nullptr, PluginFieldType::kINT32, 1));
    mFieldCollection.nbFields = mPluginAttributes.size();
    mFieldCollection.fields = mPluginAttributes.data();
}

char const* CausalConv1dPluginCreator::getPluginName() const noexcept
{
    return kCAUSAL_CONV_PLUGIN_NAME;
}

PluginFieldCollection const* CausalConv1dPluginCreator::getFieldNames() noexcept
{
    return &mFieldCollection;
}

void CausalConv1dPluginCreator::setPluginNamespace(char const* libNamespace) noexcept
{
    mNamespace = libNamespace ? libNamespace : "";
}

char const* CausalConv1dPluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

char const* CausalConv1dPluginCreator::getPluginVersion() const noexcept
{
    return kCAUSAL_CONV_PLUGIN_VERSION;
}

IPluginV3* CausalConv1dPluginCreator::createPlugin(
    char const* name, PluginFieldCollection const* fc, TensorRTPhase /* phase */) noexcept
{
    try
    {
        auto* plugin = new CausalConv1dPlugin(resolvePluginLayerName(name, fc), fc);
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("Failed to create CausalConv1dPlugin: %s", e.what());
    }
    return nullptr;
}

} // namespace plugins
} // namespace trt_edgellm
