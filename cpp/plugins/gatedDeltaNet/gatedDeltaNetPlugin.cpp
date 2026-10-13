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

#include "gatedDeltaNetPlugin.h"

#include "common/cudaUtils.h"
#include "common/executionPhase.h"
#include "common/logger.h"
#include "plugins/utils/pluginUtils.h"
#include "plugins/utils/raggedPluginMetadata.h"
#if defined(CUTE_DSL_GDN_ENABLED) || defined(CUTE_DSL_GDN_BLACKWELL_GEFORCE_ENABLED)
#include "kernels/gdnKernels/cuteDslGDNRunner.h"
#endif
#ifdef CUTE_DSL_GDN_ENABLED
#include "kernels/gdnKernels/gdnKernelUtils.cuh"
#endif

#include "kernels/gdnKernels/gdnTreeChunkKernels.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <stdexcept>

using namespace nvinfer1;

namespace trt_edgellm
{
namespace plugins
{

namespace
{
constexpr char const* kGDN_PLUGIN_VERSION{"1"};
constexpr char const* kGDN_PLUGIN_NAME{"gated_delta_net"};

constexpr int32_t kIN_Q_IDX{0};
constexpr int32_t kIN_K_IDX{1};
constexpr int32_t kIN_V_IDX{2};
constexpr int32_t kIN_A_IDX{3};
constexpr int32_t kIN_B_IDX{4};
constexpr int32_t kIN_A_LOG_IDX{5};
constexpr int32_t kIN_DT_BIAS_IDX{6};
constexpr int32_t kIN_H0_SOURCE_IDX{7};
constexpr int32_t kIN_CONTEXT_LENGTHS_IDX{8};
constexpr int32_t kIN_QUERY_START_OFFSETS_IDX{9};
constexpr int32_t kIN_STATE_INDICES_IDX{10};
constexpr int32_t kIN_EXECUTION_PHASE_MARKER_IDX{11};
constexpr int32_t kIN_CONTEXT_SEQUENCE_COUNT_IDX{12};
constexpr int32_t kIN_TREE_PARENT_IDS_IDX{13};
constexpr int32_t kIN_TREE_DEPTHS_IDX{14};
constexpr int32_t kOUT_O_IDX{0};
constexpr int32_t kOUT_H0_SOURCE_IDX{1};
constexpr int32_t kOUT_INTERMEDIATE_STATES_IDX{2};
constexpr int32_t kNUM_REQUIRED_INPUTS{13};
constexpr int32_t kNUM_DDTREE_OPTIONAL_INPUTS{2};
constexpr int32_t kNUM_REQUIRED_OUTPUTS{2};
constexpr int32_t kNUM_SPEC_VERIFY_OPTIONAL_OUTPUTS{1};

#ifdef CUTE_DSL_GDN_ENABLED
//! PDL is enabled by default for the SM12x GDN prefill path. This intentionally
//! shares EDGELLM_ENABLE_PDL with the existing NVFP4 MoE implementation so one
//! production recovery switch disables both paths. Only the literal value "0"
//! disables PDL; the runner applies the final toolchain, sequence, and SM gates.
bool requestGdnPdl()
{
    static bool const enabled = []() {
        char const* const value = std::getenv("EDGELLM_ENABLE_PDL");
        return value == nullptr || std::strcmp(value, "0") != 0;
    }();
    return enabled;
}
#endif

#ifdef CUTE_DSL_GDN_BLACKWELL_GEFORCE_ENABLED
constexpr size_t blackwellGeforceTensorMapBytes()
{
    return static_cast<size_t>(CuteDslGDNRunner::kBlackwellGeforceMaxSMCount)
        * CuteDslGDNRunner::kBlackwellGeforceTensorMapDescriptorBytes;
}
static_assert(blackwellGeforceTensorMapBytes() % kDEVICE_ALIGNMENT == 0);
#endif

} // namespace

PluginFieldCollection GatedDeltaNetPluginCreator::mFieldCollection{};
std::vector<nvinfer1::PluginField> GatedDeltaNetPluginCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(GatedDeltaNetPluginCreator);

// ---------------------------------------------------------------------------
// Plugin constructor — only this block is compilation-guarded.
// When CUTE_DSL_GDN_ENABLED is not set the constructor throws immediately so
// the object can never be constructed; all other methods are shared.
// ---------------------------------------------------------------------------
#ifdef CUTE_DSL_GDN_ENABLED
GatedDeltaNetPlugin::GatedDeltaNetPlugin(std::string const& name, int32_t kDim, int32_t vDim, bool useSpecVerifyState,
    bool useDDTree, bool useDiffusionState)
    : mLayerName(name)
    , mKDim(kDim)
    , mVDim(vDim)
    , mUseSpecVerifyState(useSpecVerifyState || useDDTree)
    , mUseDDTree(useDDTree)
    , mUseDiffusionState(useDiffusionState)
    , mSMVersion(getSMVersion())
{
    if (!CuteDslGDNRunner::canImplement(mKDim, mVDim, mSMVersion))
    {
        LOG_ERROR(
            "Cannot implement GatedDeltaNetPlugin (CuTe DSL): k_dim=%d v_dim=%d SM=%d. "
            "CuTe DSL GDN is only built for k=v=128 and requires SM>=80 (Ampere+). "
            "Use k_dim=v_dim=128 on a supported GPU, or rebuild without CuTe DSL GDN if applicable.",
            mKDim, mVDim, mSMVersion);
        throw std::runtime_error("Cannot implement the GatedDeltaNetPlugin configuration (CuTe DSL GDN).");
    }
}
#else
GatedDeltaNetPlugin::GatedDeltaNetPlugin(std::string const& name, int32_t kDim, int32_t vDim, bool useSpecVerifyState,
    bool useDDTree, bool useDiffusionState)
    : mLayerName(name)
    , mKDim(kDim)
    , mVDim(vDim)
    , mUseSpecVerifyState(useSpecVerifyState || useDDTree)
    , mUseDDTree(useDDTree)
    , mUseDiffusionState(useDiffusionState)
{
    LOG_ERROR("GatedDeltaNet plugin is not available: build with CUTE_DSL_GDN_ENABLED to enable it.");
    throw std::runtime_error("GatedDeltaNet plugin is not available: build with CUTE_DSL_GDN_ENABLED to enable it.");
}
#endif // CUTE_DSL_GDN_ENABLED

GatedDeltaNetPlugin::GatedDeltaNetPlugin(std::string const& name, PluginFieldCollection const* fc)
    : mLayerName(name)
{
    mKDim = parsePluginScalarField<int32_t>("k_dim", fc).value_or(128);
    mVDim = parsePluginScalarField<int32_t>("v_dim", fc).value_or(128);
    mUseSpecVerifyState = parsePluginScalarField<int32_t>("use_mtp", fc).value_or(0) != 0;
    mUseDDTree = parsePluginScalarField<int32_t>("use_ddtree", fc).value_or(0) != 0;
    mUseDiffusionState = parsePluginScalarField<int32_t>("use_diffusion_state", fc).value_or(0) != 0;
    mUseSpecVerifyState = mUseSpecVerifyState || mUseDDTree;

#ifdef CUTE_DSL_GDN_ENABLED
    mSMVersion = getSMVersion();
#else
    LOG_ERROR("GatedDeltaNet plugin is not available: build with CUTE_DSL_GDN_ENABLED to enable it.");
    throw std::runtime_error("GatedDeltaNet plugin is not available: build with CUTE_DSL_GDN_ENABLED to enable it.");
#endif
}

GatedDeltaNetPlugin::~GatedDeltaNetPlugin() = default;

// ---------------------------------------------------------------------------
// IPluginV3
// ---------------------------------------------------------------------------

IPluginCapability* GatedDeltaNetPlugin::getCapabilityInterface(PluginCapabilityType type) noexcept
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

IPluginV3* GatedDeltaNetPlugin::clone() noexcept
{
    try
    {
        auto* p
            = new GatedDeltaNetPlugin(mLayerName, mKDim, mVDim, mUseSpecVerifyState, mUseDDTree, mUseDiffusionState);
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

char const* GatedDeltaNetPlugin::getPluginName() const noexcept
{
    return kGDN_PLUGIN_NAME;
}

char const* GatedDeltaNetPlugin::getPluginVersion() const noexcept
{
    return kGDN_PLUGIN_VERSION;
}

char const* GatedDeltaNetPlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

void GatedDeltaNetPlugin::setPluginNamespace(char const* pluginNamespace) noexcept
{
    mNamespace = pluginNamespace ? pluginNamespace : "";
}

// ---------------------------------------------------------------------------
// IPluginV3OneBuild — shape / format
// ---------------------------------------------------------------------------

int32_t GatedDeltaNetPlugin::getNbOutputs() const noexcept
{
    return kNUM_REQUIRED_OUTPUTS + (mUseSpecVerifyState ? kNUM_SPEC_VERIFY_OPTIONAL_OUTPUTS : 0);
}

int32_t GatedDeltaNetPlugin::getOutputDataTypes(DataType* outputTypes, [[maybe_unused]] int32_t nbOutputs,
    DataType const* inputTypes, [[maybe_unused]] int32_t nbInputs) const noexcept
{
    try
    {
        [[maybe_unused]] int32_t const expectedNbOutputs
            = kNUM_REQUIRED_OUTPUTS + (mUseSpecVerifyState ? kNUM_SPEC_VERIFY_OPTIONAL_OUTPUTS : 0);
        assert(nbOutputs == expectedNbOutputs);
        outputTypes[kOUT_O_IDX] = inputTypes[kIN_Q_IDX];
        outputTypes[kOUT_H0_SOURCE_IDX] = inputTypes[kIN_H0_SOURCE_IDX];
        if (mUseSpecVerifyState)
        {
            outputTypes[kOUT_INTERMEDIATE_STATES_IDX] = DataType::kFLOAT;
        }
        return 0;
    }
    catch (std::exception const& e)
    {
        return -1;
    }
}

int32_t GatedDeltaNetPlugin::getOutputShapes(DimsExprs const* inputs, [[maybe_unused]] int32_t nbInputs,
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
        // o has same shape as v: [n, seq_len, hv, v]
        outputs[kOUT_O_IDX] = inputs[kIN_V_IDX];
        // h0_out has same shape as h0_source: [n, hv, k, v]
        outputs[kOUT_H0_SOURCE_IDX] = inputs[kIN_H0_SOURCE_IDX];
        if (mUseSpecVerifyState)
        {
            outputs[kOUT_INTERMEDIATE_STATES_IDX].nbDims = 4;
            outputs[kOUT_INTERMEDIATE_STATES_IDX].d[0] = inputs[kIN_Q_IDX].d[0];
            outputs[kOUT_INTERMEDIATE_STATES_IDX].d[1] = inputs[kIN_V_IDX].d[1];
            outputs[kOUT_INTERMEDIATE_STATES_IDX].d[2] = inputs[kIN_Q_IDX].d[2];
            outputs[kOUT_INTERMEDIATE_STATES_IDX].d[3] = inputs[kIN_V_IDX].d[2];
        }
        return 0;
    }
    catch (std::exception const& e)
    {
        return -1;
    }
}

bool GatedDeltaNetPlugin::supportsFormatCombination(
    int32_t pos, DynamicPluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
    int32_t const expectedNbOutputs
        = kNUM_REQUIRED_OUTPUTS + (mUseSpecVerifyState ? kNUM_SPEC_VERIFY_OPTIONAL_OUTPUTS : 0);
    int32_t const expectedNbInputs = kNUM_REQUIRED_INPUTS + (mUseDDTree ? kNUM_DDTREE_OPTIONAL_INPUTS : 0);
    if (nbInputs != expectedNbInputs || nbOutputs != expectedNbOutputs)
        return false;
    if (inOut[pos].desc.format != TensorFormat::kLINEAR)
        return false;
    if (pos == kIN_A_LOG_IDX || pos == kIN_H0_SOURCE_IDX)
        return inOut[pos].desc.type == DataType::kFLOAT;
    if (pos == kIN_CONTEXT_LENGTHS_IDX || pos == kIN_QUERY_START_OFFSETS_IDX || pos == kIN_STATE_INDICES_IDX
        || pos == kIN_EXECUTION_PHASE_MARKER_IDX)
        return inOut[pos].desc.type == DataType::kINT32;
    if (pos == kIN_CONTEXT_SEQUENCE_COUNT_IDX)
        return inOut[pos].desc.type == DataType::kINT32;
    if (mUseDDTree && (pos == kIN_TREE_PARENT_IDS_IDX || pos == kIN_TREE_DEPTHS_IDX))
        return inOut[pos].desc.type == DataType::kINT32;
    // FP32 outputs: h0_out, intermediate_states (when present)
    if (pos == expectedNbInputs + kOUT_H0_SOURCE_IDX)
        return inOut[pos].desc.type == DataType::kFLOAT;
    if (mUseSpecVerifyState && pos == expectedNbInputs + kOUT_INTERMEDIATE_STATES_IDX)
        return inOut[pos].desc.type == DataType::kFLOAT;
    // Everything else: FP16
    return inOut[pos].desc.type == DataType::kHALF;
}

int32_t GatedDeltaNetPlugin::configurePlugin(DynamicPluginTensorDesc const* in, int32_t nbInputs,
    [[maybe_unused]] DynamicPluginTensorDesc const* out, [[maybe_unused]] int32_t nbOutputs) noexcept
{
    int32_t const expectedNbOutputs
        = kNUM_REQUIRED_OUTPUTS + (mUseSpecVerifyState ? kNUM_SPEC_VERIFY_OPTIONAL_OUTPUTS : 0);
    int32_t const expectedNbInputs = kNUM_REQUIRED_INPUTS + (mUseDDTree ? kNUM_DDTREE_OPTIONAL_INPUTS : 0);
    if (nbInputs != expectedNbInputs)
    {
        LOG_ERROR("gated_delta_net: expected %d inputs, got %d", expectedNbInputs, nbInputs);
        return -1;
    }
    if (nbOutputs != expectedNbOutputs)
    {
        LOG_ERROR("gated_delta_net: expected %d outputs, got %d", expectedNbOutputs, nbOutputs);
        return -1;
    }
    if (in[kIN_Q_IDX].desc.type != DataType::kHALF || in[kIN_V_IDX].desc.type != DataType::kHALF)
    {
        LOG_ERROR("gated_delta_net: Q and V must be FP16");
        return -1;
    }
    int32_t const activationRank = 3;
    if (in[kIN_Q_IDX].desc.dims.nbDims != activationRank || in[kIN_V_IDX].desc.dims.nbDims != activationRank)
    {
        LOG_ERROR("gated_delta_net: Q and V must have rank %d", activationRank);
        return -1;
    }
    if (in[kIN_CONTEXT_LENGTHS_IDX].desc.type != DataType::kINT32 || in[kIN_CONTEXT_LENGTHS_IDX].desc.dims.nbDims != 1)
    {
        LOG_ERROR("gated_delta_net: context_lengths must be 1D INT32");
        return -1;
    }
    if (in[kIN_QUERY_START_OFFSETS_IDX].desc.type != DataType::kINT32
        || in[kIN_QUERY_START_OFFSETS_IDX].desc.dims.nbDims != 1
        || in[kIN_STATE_INDICES_IDX].desc.type != DataType::kINT32 || in[kIN_STATE_INDICES_IDX].desc.dims.nbDims != 1)
    {
        LOG_ERROR("gated_delta_net: ragged offsets and state_indices must be 1D INT32");
        return -1;
    }
    if (in[kIN_EXECUTION_PHASE_MARKER_IDX].desc.type != DataType::kINT32
        || in[kIN_EXECUTION_PHASE_MARKER_IDX].desc.dims.nbDims != 1)
    {
        LOG_ERROR("gated_delta_net: execution_phase_marker must be 1D INT32");
        return -1;
    }
    if (in[kIN_CONTEXT_SEQUENCE_COUNT_IDX].desc.type != DataType::kINT32
        || in[kIN_CONTEXT_SEQUENCE_COUNT_IDX].desc.dims.nbDims != 1)
    {
        LOG_ERROR("gated_delta_net: context_sequence_count_carrier must be 1D INT32");
        return -1;
    }
    if (mUseDDTree
        && (in[kIN_TREE_PARENT_IDS_IDX].desc.type != DataType::kINT32
            || in[kIN_TREE_DEPTHS_IDX].desc.type != DataType::kINT32
            || in[kIN_TREE_PARENT_IDS_IDX].desc.dims.nbDims != 1 || in[kIN_TREE_DEPTHS_IDX].desc.dims.nbDims != 1))
    {
        LOG_ERROR("gated_delta_net: DDTree tree_parent_ids/tree_depths must be 1D INT32");
        return -1;
    }
    return 0;
}

size_t GatedDeltaNetPlugin::getWorkspaceSize(DynamicPluginTensorDesc const* inputs, int32_t nbInputs,
    [[maybe_unused]] DynamicPluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept
{
    int32_t const expectedNbInputs = kNUM_REQUIRED_INPUTS + (mUseDDTree ? kNUM_DDTREE_OPTIONAL_INPUTS : 0);
    if (inputs == nullptr || nbInputs != expectedNbInputs || nbOutputs != getNbOutputs())
    {
        LOG_ERROR("gated_delta_net: invalid workspace input/output count");
        return 0;
    }
    size_t total = 0;
    int32_t const maxN = static_cast<int32_t>(inputs[kIN_CONTEXT_LENGTHS_IDX].max.d[0]);
    int32_t const maxHv = static_cast<int32_t>(inputs[kIN_H0_SOURCE_IDX].max.d[1]);
    int32_t const kDim = static_cast<int32_t>(inputs[kIN_H0_SOURCE_IDX].max.d[2]);
    int32_t const vDim = static_cast<int32_t>(inputs[kIN_H0_SOURCE_IDX].max.d[3]);
    size_t const stateBytes = alignTensorSize(static_cast<size_t>(maxN) * maxHv * kDim * vDim * sizeof(float));

    size_t backendWorkspaceBytes = 0;

#ifdef CUTE_DSL_GDN_BLACKWELL_ENABLED
    // cu_seqlens [maxN+1] int32, padded to 128-byte alignment.
    size_t const cuSeqBytes = static_cast<size_t>(maxN + 1) * sizeof(int32_t);
    size_t const cuSeqPadded = alignTensorSize(cuSeqBytes);
    // h0 scratch [maxN, maxHv, kDim, vDim] f32 — separate buffer for Blackwell h0_out.
    size_t const h0ScratchBytes = static_cast<size_t>(maxN) * maxHv * kDim * vDim * sizeof(float);

    backendWorkspaceBytes = std::max(backendWorkspaceBytes, cuSeqPadded + h0ScratchBytes);
#endif

#ifdef CUTE_DSL_GDN_BLACKWELL_GEFORCE_ENABLED
    // Workspace sizes are serialized with the engine, so reserve a fixed
    // architecture-wide upper bound rather than the build GPU's SM count.
    backendWorkspaceBytes = std::max(backendWorkspaceBytes, blackwellGeforceTensorMapBytes());
#endif

    total = backendWorkspaceBytes;
    if (mUseSpecVerifyState && !mUseDDTree)
    {
        total = std::max(total, stateBytes);
    }

    if (mUseDDTree)
    {
        int32_t const maxN = static_cast<int32_t>(inputs[kIN_CONTEXT_LENGTHS_IDX].max.d[0]);
        int32_t const maxTokens = static_cast<int32_t>(inputs[kIN_Q_IDX].max.d[0]);
        int32_t const maxSeqLen = std::max(1, maxTokens / maxN);

        // Chunk-form verify uses ancestor masks. The KS/QS + prep scratch
        // lives in the intermediate-states row tail, NOT here — engine plans
        // serialize this size at build time, so growing it silently overflows
        // on pre-existing engines.
        int32_t const chunkNodes = std::min(maxSeqLen, kernel::kGDN_TREE_CHUNK_MAX_NODES);
        size_t const maskBytes = alignTensorSize(
            static_cast<size_t>(maxN) * chunkNodes * kernel::kGDN_TREE_CHUNK_MASK_WORDS * sizeof(uint32_t));
        total = std::max(total, maskBytes + stateBytes);
    }
    if (mUseDiffusionState)
    {
        total = std::max(total, backendWorkspaceBytes + stateBytes);
    }

    return total;
}

int32_t GatedDeltaNetPlugin::getAliasedInput([[maybe_unused]] int32_t outputIndex) noexcept
{
    // Myelin materializes a redundant per-layer copy for a declared read-write
    // alias. The runtime binds both tensors to the same resident state pool.
    return -1;
}

// ---------------------------------------------------------------------------
// IPluginV3OneRuntime — execution
// ---------------------------------------------------------------------------
#ifdef CUTE_DSL_GDN_ENABLED
int32_t GatedDeltaNetPlugin::enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc,
    void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    int64_t const* qDims = inputDesc[kIN_Q_IDX].dims.d;
    RaggedPluginMetadata ragged{};
    try
    {
        ragged = decodeRaggedPluginMetadata("gated_delta_net", inputDesc[kIN_Q_IDX], inputDesc[kIN_CONTEXT_LENGTHS_IDX],
            inputDesc[kIN_QUERY_START_OFFSETS_IDX], inputDesc[kIN_EXECUTION_PHASE_MARKER_IDX],
            inputDesc[kIN_CONTEXT_SEQUENCE_COUNT_IDX]);
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("%s", e.what());
        return -1;
    }
    int32_t const n = ragged.numSequences;
    int32_t const physicalTokens = ragged.physicalTokens;
    if (n <= 0 || physicalTokens % n != 0)
    {
        LOG_ERROR("gated_delta_net: ragged T_exec=%d must be divisible by N=%d", physicalTokens, n);
        return -1;
    }
    int32_t const seq_len = physicalTokens / n;
    int32_t const h = static_cast<int32_t>(qDims[1]);
    int32_t const k_dim = static_cast<int32_t>(qDims[2]);

    int64_t const* vDims = inputDesc[kIN_V_IDX].dims.d;
    int32_t const hv = static_cast<int32_t>(vDims[1]);
    int32_t const v_dim = static_cast<int32_t>(vDims[2]);
    int32_t statePoolRows{};
    try
    {
        statePoolRows = validateIndexedResidentStateDescriptors("gated_delta_net", n, inputDesc[kIN_STATE_INDICES_IDX],
            inputDesc[kIN_H0_SOURCE_IDX], outputDesc[kOUT_H0_SOURCE_IDX], DataType::kFLOAT, {hv, k_dim, v_dim});
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("%s", e.what());
        return -1;
    }

    constexpr int32_t kLinearSpecVerifyMaxSeqLen = 16;
    rt::ExecutionPhase const phase = ragged.phase;
    bool const verifyActive = phase == rt::ExecutionPhase::kSpecTargetVerify;
    bool const diffusionDenoise = phase == rt::ExecutionPhase::kDiffusionDenoise;
    bool const contextLike = phase == rt::ExecutionPhase::kContextPrefill || phase == rt::ExecutionPhase::kContextChunk
        || diffusionDenoise || phase == rt::ExecutionPhase::kDiffusionCommit;
    bool const proposalActive = phase == rt::ExecutionPhase::kSpecDraftProposal;
    bool const decodeActive = phase == rt::ExecutionPhase::kAutoregressiveDecode;
    if (decodeActive && seq_len != 1)
    {
        LOG_ERROR("gated_delta_net: AUTOREGRESSIVE_DECODE requires one physical token per sequence, got %d", seq_len);
        return -1;
    }
    if ((diffusionDenoise || phase == rt::ExecutionPhase::kDiffusionCommit) && !mUseDiffusionState)
    {
        LOG_ERROR("gated_delta_net: diffusion phase requires diffusion-capable engine metadata");
        return -1;
    }
    if (verifyActive && !mUseSpecVerifyState)
    {
        LOG_ERROR("gated_delta_net: SPEC_TARGET_VERIFY requires verify-capable engine metadata");
        return -1;
    }
    bool const ddtreeActive = mUseDDTree && verifyActive;
    bool const mtpActive = mUseSpecVerifyState && verifyActive && !ddtreeActive;
    if (mtpActive && (seq_len < 1 || seq_len > kLinearSpecVerifyMaxSeqLen))
    {
        LOG_ERROR("gated_delta_net: linear spec-verify kernel supports seq_len in [1, %d], got %d",
            kLinearSpecVerifyMaxSeqLen, seq_len);
        return -1;
    }
    if (ddtreeActive)
    {
        PluginTensorDesc const& parentDesc = inputDesc[kIN_TREE_PARENT_IDS_IDX];
        PluginTensorDesc const& depthDesc = inputDesc[kIN_TREE_DEPTHS_IDX];
        if (seq_len < 1 || parentDesc.dims.nbDims != 1 || depthDesc.dims.nbDims != 1
            || parentDesc.dims.d[0] != physicalTokens || depthDesc.dims.d[0] != physicalTokens)
        {
            LOG_ERROR(
                "gated_delta_net: DDTree requires flat tree_parent_ids/tree_depths shape [T_exec=%d]", physicalTokens);
            return -1;
        }
    }

    GDNParams params{};
    params.seq_len = seq_len;
    params.h = h;
    params.hv = hv;
    params.smVersion = mSMVersion;
    params.use_mtp = mtpActive;
    params.use_prefill = contextLike || (proposalActive && seq_len > 1);
    if (!ddtreeActive && !CuteDslGDNRunner::ensureKernelModules(params, stream))
    {
        LOG_ERROR("gated_delta_net: failed to load the selected CuTe DSL GDN module");
        return -1;
    }

    void* h0Out = outputs[kOUT_H0_SOURCE_IDX];

    void* h0State = const_cast<void*>(inputs[kIN_H0_SOURCE_IDX]);
    if (h0Out != inputs[kIN_H0_SOURCE_IDX])
    {
        LOG_ERROR("gated_delta_net: recurrent state input/output must use the same resident pool address");
        return -1;
    }

    if (ddtreeActive && !kernel::gdnTreeChunkVerifyEnabled(seq_len))
    {
        LOG_ERROR("gated_delta_net: DDTree chunk-form verify supports seq_len <= %d, got %d",
            kernel::kGDN_TREE_CHUNK_MAX_NODES, seq_len);
        return -1;
    }

    size_t backendWorkspaceBytes = 0;
#ifdef CUTE_DSL_GDN_BLACKWELL_ENABLED
    if (mSMVersion == 100 || mSMVersion == 101 || mSMVersion == 110)
    {
        size_t const cuSeqPadded = alignTensorSize(static_cast<size_t>(n + 1) * sizeof(int32_t));
        size_t const h0ScratchBytes = static_cast<size_t>(n) * hv * k_dim * v_dim * sizeof(float);
        backendWorkspaceBytes = cuSeqPadded + h0ScratchBytes;
    }
#endif
#ifdef CUTE_DSL_GDN_BLACKWELL_GEFORCE_ENABLED
    if (mSMVersion == 120 || mSMVersion == 121)
    {
        backendWorkspaceBytes = std::max(backendWorkspaceBytes, blackwellGeforceTensorMapBytes());
    }
#endif

    if (verifyActive || diffusionDenoise)
    {
        if (workspace == nullptr)
        {
            LOG_ERROR("gated_delta_net: transactional execution requires state workspace");
            return -1;
        }
        size_t const maskBytes = ddtreeActive
            ? alignTensorSize(static_cast<size_t>(n) * seq_len * kernel::kGDN_TREE_CHUNK_MASK_WORDS * sizeof(uint32_t))
            : 0;
        size_t const stateOffset = diffusionDenoise ? backendWorkspaceBytes : maskBytes;
        h0State = static_cast<char*>(workspace) + stateOffset;
        launchGdnStateGather(inputs[kIN_H0_SOURCE_IDX], h0State, inputs[kIN_STATE_INDICES_IDX], n, hv, statePoolRows,
            k_dim, v_dim, stream);
    }

    // Stateless chunk-form tree verify. Reads h0 strictly read-only and writes
    // o plus replay stash into the head of the intermediate_states buffer.
    if (ddtreeActive)
    {
        // Workspace: ancestor masks only (reserved by getWorkspaceSize when
        // mUseDDTree). The KS/QS + prep scratch lives in the row tail of the
        // intermediate-states buffer.
        uint32_t* masks = static_cast<uint32_t*>(workspace);
        if (cudaError_t const e
            = kernel::gdnTreeBuildAncestorMasks(static_cast<int32_t const*>(inputs[kIN_TREE_PARENT_IDS_IDX]), masks, n,
                seq_len, /*maxDepth=*/seq_len, stream);
            e != cudaSuccess)
        {
            LOG_ERROR("gated_delta_net: gdnTreeBuildAncestorMasks launch failed: %s", cudaGetErrorString(e));
            return -1;
        }

        // Per-batch stash stride == one intermediate buffer row, so batches
        // land in disjoint, engine-compatible regions.
        size_t const stashBatchStrideBytes
            = static_cast<size_t>(seq_len) * hv * static_cast<size_t>(k_dim) * v_dim * sizeof(float);
        // Standard scaled dot-product attention scale for the chunk-form verify kernel.
        float const qScale = 1.f / std::sqrt(static_cast<float>(k_dim));
        cudaError_t const verifyErr = kernel::gdnTreeVerifyChunk(static_cast<float const*>(h0State),
            static_cast<__half const*>(inputs[kIN_Q_IDX]), static_cast<__half const*>(inputs[kIN_K_IDX]),
            static_cast<__half const*>(inputs[kIN_V_IDX]), static_cast<__half const*>(inputs[kIN_A_IDX]),
            static_cast<__half const*>(inputs[kIN_B_IDX]), static_cast<float const*>(inputs[kIN_A_LOG_IDX]),
            static_cast<__half const*>(inputs[kIN_DT_BIAS_IDX]), masks, static_cast<__half*>(outputs[kOUT_O_IDX]),
            outputs[kOUT_INTERMEDIATE_STATES_IDX], stashBatchStrideBytes, n, seq_len, h, hv, qScale,
            /*useQKL2Norm=*/true, stream);
        if (verifyErr != cudaSuccess)
        {
            LOG_ERROR("gated_delta_net: chunk-form verify launch failed: %s", cudaGetErrorString(verifyErr));
            return -1;
        }
        return 0;
    }

    params.q = const_cast<void*>(inputs[kIN_Q_IDX]);
    params.k = const_cast<void*>(inputs[kIN_K_IDX]);
    params.v = const_cast<void*>(inputs[kIN_V_IDX]);
    params.a = const_cast<void*>(inputs[kIN_A_IDX]);
    params.b = const_cast<void*>(inputs[kIN_B_IDX]);
    params.A_log = const_cast<void*>(inputs[kIN_A_LOG_IDX]);
    params.dt_bias = const_cast<void*>(inputs[kIN_DT_BIAS_IDX]);
    params.h0_source = h0State;
    params.context_lengths = const_cast<void*>(inputs[kIN_CONTEXT_LENGTHS_IDX]);
    params.state_indices = const_cast<void*>(inputs[kIN_STATE_INDICES_IDX]);
    params.o = outputs[kOUT_O_IDX];
    params.n = n;
    params.seq_len = seq_len;
    params.h = h;
    params.hv = hv;
    params.k_dim = k_dim;
    params.v_dim = v_dim;
    params.state_pool_rows = statePoolRows;
    params.smVersion = mSMVersion;
    params.enablePdl = requestGdnPdl();
    if (diffusionDenoise)
    {
        params.state_indices = nullptr;
        params.state_pool_rows = n;
    }

    if (mtpActive)
    {
        // MTP decode: process all seq_len draft tokens with per-step state caching.
        params.use_mtp = true;
        params.intermediate_states = outputs[kOUT_INTERMEDIATE_STATES_IDX];
    }
    else
    {
#ifdef CUTE_DSL_GDN_BLACKWELL_ENABLED
        // Blackwell prefill: carve cu_seqlens and h0 scratch out of the pre-allocated workspace.
        //   workspace layout: [cu_seqlens: (n+1)*int32, pad to 128B] [h0_scratch: n*hv*k*v*f32]
        if (params.use_prefill && (mSMVersion == 100 || mSMVersion == 101 || mSMVersion == 110))
        {
            size_t const cuSeqBytes = static_cast<size_t>(n + 1) * sizeof(int32_t);
            size_t const cuSeqPadded = (cuSeqBytes + 127u) & ~static_cast<size_t>(127u);

            char* bwBase = static_cast<char*>(workspace);
            launchGdnCalCuSeqLens(inputs[kIN_CONTEXT_LENGTHS_IDX], bwBase, n, stream);
            params.cu_seqlens = bwBase;
            params.h0_scratch = bwBase + cuSeqPadded;
        }
#endif
#ifdef CUTE_DSL_GDN_BLACKWELL_GEFORCE_ENABLED
        if (params.use_prefill && (mSMVersion == 120 || mSMVersion == 121))
        {
            if (workspace == nullptr)
            {
                LOG_ERROR("gated_delta_net: Blackwell GeForce prefill requires workspace");
                return -1;
            }
            params.tensormap_scratch = workspace;
        }
#endif
    }

    CuteDslGDNRunner runner;
    int ret = runner.run(params, stream);

    return (ret == 0) ? 0 : -1;
}
#else
int32_t GatedDeltaNetPlugin::enqueue(PluginTensorDesc const* /* inputDesc */, PluginTensorDesc const* /* outputDesc */,
    void const* const* /* inputs */, void* const* /* outputs */, void* /* workspace */,
    cudaStream_t /* stream */) noexcept
{
    // Constructor already threw; this path should be unreachable.
    return -1;
}
#endif // CUTE_DSL_GDN_ENABLED

int32_t GatedDeltaNetPlugin::onShapeChange(
    PluginTensorDesc const* in, int32_t nbInputs, PluginTensorDesc const* out, int32_t nbOutputs) noexcept
{
    int32_t const expectedNbInputs = kNUM_REQUIRED_INPUTS + (mUseDDTree ? kNUM_DDTREE_OPTIONAL_INPUTS : 0);
    return in != nullptr && out != nullptr && nbInputs == expectedNbInputs && nbOutputs == getNbOutputs() ? 0 : -1;
}

IPluginV3* GatedDeltaNetPlugin::attachToContext(IPluginResourceContext* /* context */) noexcept
{
    return clone();
}

// ---------------------------------------------------------------------------
// Serialization
// ---------------------------------------------------------------------------

PluginFieldCollection const* GatedDeltaNetPlugin::getFieldsToSerialize() noexcept
{
    mDataToSerialize.clear();
    mDataToSerialize.emplace_back("k_dim", &mKDim, PluginFieldType::kINT32, 1);
    mDataToSerialize.emplace_back("v_dim", &mVDim, PluginFieldType::kINT32, 1);
    mUseSpecVerifyStateField = mUseSpecVerifyState ? 1 : 0;
    mDataToSerialize.emplace_back("use_mtp", &mUseSpecVerifyStateField, PluginFieldType::kINT32, 1);
    mUseDDTreeField = mUseDDTree ? 1 : 0;
    mDataToSerialize.emplace_back("use_ddtree", &mUseDDTreeField, PluginFieldType::kINT32, 1);
    mUseDiffusionStateField = mUseDiffusionState ? 1 : 0;
    mDataToSerialize.emplace_back("use_diffusion_state", &mUseDiffusionStateField, PluginFieldType::kINT32, 1);

    mFCToSerialize.nbFields = mDataToSerialize.size();
    mFCToSerialize.fields = mDataToSerialize.data();
    return &mFCToSerialize;
}

// ---------------------------------------------------------------------------
// Creator
// ---------------------------------------------------------------------------

GatedDeltaNetPluginCreator::GatedDeltaNetPluginCreator()
{
    static std::mutex sMutex;
    std::lock_guard<std::mutex> lock(sMutex);
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("k_dim", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("v_dim", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("use_mtp", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("use_ddtree", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("use_diffusion_state", nullptr, PluginFieldType::kINT32, 1));
    mFieldCollection.nbFields = static_cast<int32_t>(mPluginAttributes.size());
    mFieldCollection.fields = mPluginAttributes.data();
}

char const* GatedDeltaNetPluginCreator::getPluginName() const noexcept
{
    return kGDN_PLUGIN_NAME;
}

char const* GatedDeltaNetPluginCreator::getPluginVersion() const noexcept
{
    return kGDN_PLUGIN_VERSION;
}

PluginFieldCollection const* GatedDeltaNetPluginCreator::getFieldNames() noexcept
{
    return &mFieldCollection;
}

char const* GatedDeltaNetPluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

void GatedDeltaNetPluginCreator::setPluginNamespace(char const* pluginNamespace) noexcept
{
    mNamespace = pluginNamespace ? pluginNamespace : "";
}

IPluginV3* GatedDeltaNetPluginCreator::createPlugin(
    char const* name, PluginFieldCollection const* fc, TensorRTPhase /* phase */) noexcept
{
    try
    {
        auto* plugin = new GatedDeltaNetPlugin(std::string(name), fc);
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("GatedDeltaNetPluginCreator::createPlugin failed: %s", e.what());
        return nullptr;
    }
}

} // namespace plugins
} // namespace trt_edgellm
