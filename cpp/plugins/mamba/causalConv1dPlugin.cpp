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

#include "common/executionPhase.h"
#include "common/logger.h"
#include "kernels/mamba/causalConv1d.h"
#include "kernels/mamba/selectiveStateUpdate.h"
#include "plugins/utils/pluginUtils.h"
#include "plugins/utils/raggedPluginMetadata.h"

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
constexpr int32_t kNUM_DDTREE_OPTIONAL_INPUTS{2};
constexpr int32_t kNUM_REQUIRED_OUTPUTS{2};
constexpr int32_t kNUM_SPEC_VERIFY_OPTIONAL_OUTPUTS{1};

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

} // namespace

PluginFieldCollection CausalConv1dPluginCreator::mFieldCollection{};
std::vector<PluginField> CausalConv1dPluginCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(CausalConv1dPluginCreator);

// ---------------------------------------------------------------------------
// Plugin — construction / destruction
// ---------------------------------------------------------------------------

CausalConv1dPlugin::CausalConv1dPlugin(std::string const& name, int32_t stride, int32_t padding, int32_t dilation,
    int32_t groups, bool useSpecVerifyState, bool useDDTree)
    : mLayerName(name)
    , mStride(stride)
    , mPadding(padding)
    , mDilation(dilation)
    , mGroups(groups)
    , mUseSpecVerifyState(useSpecVerifyState || useDDTree)
    , mUseDDTree(useDDTree)
{
}

CausalConv1dPlugin::CausalConv1dPlugin(std::string const& name, PluginFieldCollection const* fc)
    : mLayerName(name)
{
    mStride = parsePluginIntField("stride", fc).value_or(1);
    mPadding = parsePluginIntField("padding", fc).value_or(0);
    mDilation = parsePluginIntField("dilation", fc).value_or(1);
    mGroups = parsePluginIntField("groups", fc).value_or(0);
    mUseSpecVerifyState = parsePluginIntField("use_mtp", fc).value_or(0) != 0;
    mUseDDTree = parsePluginIntField("use_ddtree", fc).value_or(0) != 0;
    mUseSpecVerifyState = mUseSpecVerifyState || mUseDDTree;
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
        auto* plugin = new CausalConv1dPlugin(
            mLayerName, mStride, mPadding, mDilation, mGroups, mUseSpecVerifyState, mUseDDTree);
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
            = kNUM_REQUIRED_INPUTS + (mUseDDTree ? kNUM_DDTREE_OPTIONAL_INPUTS : 0);
        assert(nbInputs == expectedNbInputs);
        assert(nbOutputs == expectedNbOutputs);
        // Output: same shape as x [batch, seq_len, dim].
        outputs[kOUT_IDX] = inputs[kIN_X_IDX];
        // Conv state output: same shape as conv_state input [batch, dim, kernel].
        outputs[kOUT_CONV_STATE_IDX] = inputs[kIN_CONV_STATE_IDX];
        if (mUseSpecVerifyState)
        {
            outputs[kOUT_INTERMEDIATE_CONV_STATES].nbDims = 3;
            outputs[kOUT_INTERMEDIATE_CONV_STATES].d[0] = inputs[kIN_X_IDX].d[0];
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
    int32_t const expectedNbInputs = kNUM_REQUIRED_INPUTS + (mUseDDTree ? kNUM_DDTREE_OPTIONAL_INPUTS : 0);
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
    if (mUseDDTree && (pos == kIN_TREE_PARENT_IDS_IDX || pos == kIN_TREE_DEPTHS_IDX))
        return desc.type == DataType::kINT32;
    // Everything else (all inputs + all outputs): FP16
    return desc.type == DataType::kHALF;
}

int32_t CausalConv1dPlugin::configurePlugin(DynamicPluginTensorDesc const* in, int32_t nbInputs,
    [[maybe_unused]] DynamicPluginTensorDesc const* out, [[maybe_unused]] int32_t nbOutputs) noexcept
{
    int32_t const expectedNbInputs = kNUM_REQUIRED_INPUTS + (mUseDDTree ? kNUM_DDTREE_OPTIONAL_INPUTS : 0);
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
    if (mUseDDTree
        && (in[kIN_TREE_PARENT_IDS_IDX].desc.type != DataType::kINT32
            || in[kIN_TREE_DEPTHS_IDX].desc.type != DataType::kINT32
            || in[kIN_TREE_PARENT_IDS_IDX].desc.dims.nbDims != 1 || in[kIN_TREE_DEPTHS_IDX].desc.dims.nbDims != 1))
    {
        LOG_ERROR("causal_conv1d: DDTree tree_parent_ids/tree_depths must be 1D INT32");
        return -1;
    }
    return 0;
}

size_t CausalConv1dPlugin::getWorkspaceSize(DynamicPluginTensorDesc const* inputs, int32_t nbInputs,
    DynamicPluginTensorDesc const* /* outputs */, int32_t nbOutputs) const noexcept
{
    int32_t const expectedNbInputs = kNUM_REQUIRED_INPUTS + (mUseDDTree ? kNUM_DDTREE_OPTIONAL_INPUTS : 0);
    if (inputs == nullptr || nbInputs != expectedNbInputs || nbOutputs != getNbOutputs())
    {
        LOG_ERROR("causal_conv1d: invalid workspace input/output count");
        return 0;
    }
    int32_t const maxBatch = static_cast<int32_t>(inputs[kIN_CONTEXT_LENGTHS_IDX].max.d[0]);
    int32_t const dim = static_cast<int32_t>(inputs[kIN_X_IDX].max.d[1]);
    int32_t const width = static_cast<int32_t>(inputs[kIN_WEIGHT_IDX].max.d[2]);
    return alignTensorSize(static_cast<size_t>(maxBatch) * dim * width * sizeof(half));
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
    int32_t const width = static_cast<int32_t>(wDesc.dims.d[2]);
    int32_t statePoolRows{};
    try
    {
        statePoolRows
            = validateIndexedResidentStateDescriptors("causal_conv1d", batch, inputDesc[kIN_STATE_INDICES_IDX],
                inputDesc[kIN_CONV_STATE_IDX], outputDesc[kOUT_CONV_STATE_IDX], xDesc.type, {dim, width});
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

    constexpr int32_t kLinearSpecVerifyMaxSeqLen = 16;
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
    bool const ddtreeActive = mUseDDTree && verifyActive;
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
        rt::DeviceType::kGPU, nvinfer1::DataType::kINT32};
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
            rt::Coords{statePoolRows, dim, 1, width}, rt::DeviceType::kGPU, xDesc.type};
        auto activeStateView
            = rt::Tensor{workspace, rt::Coords{batch, dim, 1, width}, rt::DeviceType::kGPU, xDesc.type};
        mamba_ssm::invokeMambaStateGather(residentStateView, activeStateView, stateIndicesTensor, stream);
        denoiseStateTensor.emplace(workspace, rt::Coords{batch, dim, width}, rt::DeviceType::kGPU, xDesc.type);
        stateIndicesOpt = std::nullopt;
    }

    if (mtpActive)
    {
        auto mtpStateTensor = rt::Tensor{const_cast<void*>(inputs[kIN_CONV_STATE_IDX]),
            rt::Coords{statePoolRows, dim, width}, rt::DeviceType::kGPU, xDesc.type};
        auto mtpNewColsTensor = rt::Tensor{
            const_cast<void*>(inputs[kIN_X_IDX]), rt::Coords{batch, seqLen, dim}, rt::DeviceType::kGPU, xDesc.type};
        auto mtpWeightTensor = rt::Tensor{const_cast<void*>(inputs[kIN_WEIGHT_IDX]),
            rt::Coords{wDesc.dims.d[0], wDesc.dims.d[1], wDesc.dims.d[2]}, rt::DeviceType::kGPU, xDesc.type};
        auto mtpBiasTensor
            = rt::Tensor{const_cast<void*>(inputs[kIN_BIAS_IDX]), rt::Coords{dim}, rt::DeviceType::kGPU, xDesc.type};
        auto mtpOutTensor
            = rt::Tensor{outputs[kOUT_IDX], rt::Coords{batch, seqLen, dim}, rt::DeviceType::kGPU, xDesc.type};
        auto mtpIntermTensor = rt::Tensor{outputs[kOUT_INTERMEDIATE_CONV_STATES], rt::Coords{batch, seqLen, dim, width},
            rt::DeviceType::kGPU, xDesc.type};

        trt_edgellm::rt::OptionalInputTensor mtpBiasOpt = std::optional(std::cref(mtpBiasTensor));
        mamba_ssm::invokeCausalConv1dDecodeMTP(mtpStateTensor, mtpNewColsTensor, mtpWeightTensor, mtpBiasOpt,
            mtpOutTensor, mtpIntermTensor, seqLen, stateIndicesOpt, stream);
    }
    else if (ddtreeActive)
    {
        auto treeStateTensor = rt::Tensor{const_cast<void*>(inputs[kIN_CONV_STATE_IDX]),
            rt::Coords{statePoolRows, dim, width}, rt::DeviceType::kGPU, xDesc.type};
        auto treeNewColsTensor = rt::Tensor{
            const_cast<void*>(inputs[kIN_X_IDX]), rt::Coords{batch, seqLen, dim}, rt::DeviceType::kGPU, xDesc.type};
        auto treeWeightTensor = rt::Tensor{const_cast<void*>(inputs[kIN_WEIGHT_IDX]),
            rt::Coords{wDesc.dims.d[0], wDesc.dims.d[1], wDesc.dims.d[2]}, rt::DeviceType::kGPU, xDesc.type};
        auto treeBiasTensor
            = rt::Tensor{const_cast<void*>(inputs[kIN_BIAS_IDX]), rt::Coords{dim}, rt::DeviceType::kGPU, xDesc.type};
        auto treeOutTensor
            = rt::Tensor{outputs[kOUT_IDX], rt::Coords{batch, seqLen, dim}, rt::DeviceType::kGPU, xDesc.type};
        auto treeStateOutTensor
            = rt::Tensor{convStateOut, rt::Coords{batch, dim, width}, rt::DeviceType::kGPU, xDesc.type};
        auto treeIntermTensor = rt::Tensor{outputs[kOUT_INTERMEDIATE_CONV_STATES],
            rt::Coords{batch, seqLen, dim, width}, rt::DeviceType::kGPU, xDesc.type};
        auto treeParentTensor = rt::Tensor{const_cast<void*>(inputs[kIN_TREE_PARENT_IDS_IDX]),
            rt::Coords{batch, seqLen}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32};
        auto treeDepthTensor = rt::Tensor{const_cast<void*>(inputs[kIN_TREE_DEPTHS_IDX]), rt::Coords{batch, seqLen},
            rt::DeviceType::kGPU, nvinfer1::DataType::kINT32};

        trt_edgellm::rt::OptionalInputTensor treeBiasOpt = std::optional(std::cref(treeBiasTensor));
        mamba_ssm::invokeCausalConv1dDecodeDDTree(treeStateTensor, treeNewColsTensor, treeWeightTensor, treeBiasOpt,
            treeOutTensor, treeStateOutTensor, treeIntermTensor, treeParentTensor, treeDepthTensor, stateIndicesOpt,
            stream);
    }
    else if (contextLike || (proposalActive && seqLen > 1))
    {
        // Sequential state update for context-like phases and multi-row proposal/commit.
        int32_t const outSeqLen = seqLen;

        auto xTensor = rt::Tensor{
            const_cast<void*>(inputs[kIN_X_IDX]), rt::Coords{batch, seqLen, dim}, rt::DeviceType::kGPU, xDesc.type};
        auto weightTensor = rt::Tensor{const_cast<void*>(inputs[kIN_WEIGHT_IDX]),
            rt::Coords{wDesc.dims.d[0], wDesc.dims.d[1], wDesc.dims.d[2]}, rt::DeviceType::kGPU, xDesc.type};
        auto biasTensor
            = rt::Tensor{const_cast<void*>(inputs[kIN_BIAS_IDX]), rt::Coords{dim}, rt::DeviceType::kGPU, xDesc.type};
        auto outTensor
            = rt::Tensor{outputs[kOUT_IDX], rt::Coords{batch, outSeqLen, dim}, rt::DeviceType::kGPU, xDesc.type};

        auto contextLengthsTensor = rt::Tensor{const_cast<void*>(inputs[kIN_CONTEXT_LENGTHS_IDX]), rt::Coords{batch},
            rt::DeviceType::kGPU, nvinfer1::DataType::kINT32};
        trt_edgellm::rt::OptionalInputTensor contextLengthsOpt = std::optional(std::cref(contextLengthsTensor));
        auto residentStateTensor
            = rt::Tensor{convStateOut, rt::Coords{statePoolRows, dim, width}, rt::DeviceType::kGPU, xDesc.type};
        rt::Tensor& activeStateTensor = diffusionDenoise ? denoiseStateTensor.value() : residentStateTensor;
        trt_edgellm::rt::OptionalInputTensor initialStateOpt = std::optional(std::cref(activeStateTensor));

        trt_edgellm::rt::OptionalInputTensor biasOpt = std::optional(std::cref(biasTensor));
        mamba_ssm::invokeCausalConv1d(xTensor, weightTensor, biasOpt, outTensor, mStride, mPadding, mDilation,
            initialStateOpt, contextLengthsOpt, stateIndicesOpt, stream);

        auto captureXTensor = rt::Tensor{
            const_cast<void*>(inputs[kIN_X_IDX]), rt::Coords{batch, seqLen, dim}, rt::DeviceType::kGPU, xDesc.type};
        mamba_ssm::invokeCaptureConvState(
            captureXTensor, initialStateOpt, activeStateTensor, contextLengthsOpt, stateIndicesOpt, stream);
    }
    else
    {
        // Single-row decode or proposal state update.
        auto decodeStateTensor
            = rt::Tensor{convStateOut, rt::Coords{statePoolRows, dim, width}, rt::DeviceType::kGPU, xDesc.type};
        auto decodeNewColTensor = rt::Tensor{
            const_cast<void*>(inputs[kIN_X_IDX]), rt::Coords{batch, 1, dim}, rt::DeviceType::kGPU, xDesc.type};
        auto decodeWeightTensor = rt::Tensor{const_cast<void*>(inputs[kIN_WEIGHT_IDX]),
            rt::Coords{wDesc.dims.d[0], wDesc.dims.d[1], wDesc.dims.d[2]}, rt::DeviceType::kGPU, xDesc.type};
        auto decodeBiasTensor
            = rt::Tensor{const_cast<void*>(inputs[kIN_BIAS_IDX]), rt::Coords{dim}, rt::DeviceType::kGPU, xDesc.type};
        auto decodeOutTensor
            = rt::Tensor{outputs[kOUT_IDX], rt::Coords{batch, 1, dim}, rt::DeviceType::kGPU, xDesc.type};
        trt_edgellm::rt::OptionalInputTensor decodeBiasOpt = std::optional(std::cref(decodeBiasTensor));
        mamba_ssm::invokeCausalConv1dDecode(decodeStateTensor, decodeNewColTensor, decodeWeightTensor, decodeBiasOpt,
            decodeOutTensor, stateIndicesOpt, stream);
    }

    return 0;
}

int32_t CausalConv1dPlugin::onShapeChange(
    PluginTensorDesc const* in, int32_t nbInputs, PluginTensorDesc const* out, int32_t nbOutputs) noexcept
{
    int32_t const expectedNbInputs = kNUM_REQUIRED_INPUTS + (mUseDDTree ? kNUM_DDTREE_OPTIONAL_INPUTS : 0);
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
    mDataToSerialize.emplace_back("use_mtp", &mUseSpecVerifyStateField, PluginFieldType::kINT32, 1);
    mUseDDTreeField = mUseDDTree ? 1 : 0;
    mDataToSerialize.emplace_back("use_ddtree", &mUseDDTreeField, PluginFieldType::kINT32, 1);

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
    mPluginAttributes.emplace_back(PluginField("use_mtp", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("use_ddtree", nullptr, PluginFieldType::kINT32, 1));
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
        auto* plugin = new CausalConv1dPlugin(std::string(name), fc);
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
