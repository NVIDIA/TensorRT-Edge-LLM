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
#include "common/logger.h"
#include "plugins/utils/pluginUtils.h"
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
constexpr int32_t kIN_SPEC_VERIFY_PHASE_MARKER_IDX{9};
constexpr int32_t kIN_TREE_PARENT_IDS_IDX{10};
constexpr int32_t kIN_TREE_DEPTHS_IDX{11};
constexpr int32_t kOUT_O_IDX{0};
constexpr int32_t kOUT_H0_SOURCE_IDX{1};
constexpr int32_t kOUT_INTERMEDIATE_STATES_IDX{2};
constexpr int32_t kNUM_REQUIRED_INPUTS{9};
constexpr int32_t kNUM_SPEC_VERIFY_OPTIONAL_INPUTS{1};
constexpr int32_t kNUM_DDTREE_OPTIONAL_INPUTS{2};
constexpr int32_t kNUM_REQUIRED_OUTPUTS{2};
constexpr int32_t kNUM_SPEC_VERIFY_OPTIONAL_OUTPUTS{1};

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
GatedDeltaNetPlugin::GatedDeltaNetPlugin(
    std::string const& name, int32_t kDim, int32_t vDim, bool useSpecVerifyState, bool useDDTree)
    : mLayerName(name)
    , mKDim(kDim)
    , mVDim(vDim)
    , mUseSpecVerifyState(useSpecVerifyState || useDDTree)
    , mUseDDTree(useDDTree)
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
GatedDeltaNetPlugin::GatedDeltaNetPlugin(
    std::string const& name, int32_t kDim, int32_t vDim, bool useSpecVerifyState, bool useDDTree)
    : mLayerName(name)
    , mKDim(kDim)
    , mVDim(vDim)
    , mUseSpecVerifyState(useSpecVerifyState || useDDTree)
    , mUseDDTree(useDDTree)
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
        auto* p = new GatedDeltaNetPlugin(mLayerName, mKDim, mVDim, mUseSpecVerifyState, mUseDDTree);
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
        [[maybe_unused]] int32_t const expectedNbInputs = kNUM_REQUIRED_INPUTS
            + (mUseSpecVerifyState ? kNUM_SPEC_VERIFY_OPTIONAL_INPUTS : 0)
            + (mUseDDTree ? kNUM_DDTREE_OPTIONAL_INPUTS : 0);
        assert(nbInputs == expectedNbInputs);
        assert(nbOutputs == expectedNbOutputs);
        // o has same shape as v: [n, seq_len, hv, v]
        outputs[kOUT_O_IDX] = inputs[kIN_V_IDX];
        // h0_out has same shape as h0_source: [n, hv, k, v]
        outputs[kOUT_H0_SOURCE_IDX] = inputs[kIN_H0_SOURCE_IDX];
        if (mUseSpecVerifyState)
        {
            // Only spec-verify produces recurrent checkpoints; normal prefill uses a zero-length marker.
            outputs[kOUT_INTERMEDIATE_STATES_IDX].nbDims = 5;
            outputs[kOUT_INTERMEDIATE_STATES_IDX].d[0] = inputs[kIN_Q_IDX].d[0];
            outputs[kOUT_INTERMEDIATE_STATES_IDX].d[1] = exprBuilder.operation(
                DimensionOperation::kPROD, *inputs[kIN_Q_IDX].d[1], *inputs[kIN_SPEC_VERIFY_PHASE_MARKER_IDX].d[0]);
            outputs[kOUT_INTERMEDIATE_STATES_IDX].d[2] = inputs[kIN_V_IDX].d[2];
            outputs[kOUT_INTERMEDIATE_STATES_IDX].d[3] = inputs[kIN_Q_IDX].d[3];
            outputs[kOUT_INTERMEDIATE_STATES_IDX].d[4] = inputs[kIN_V_IDX].d[3];
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
    int32_t const expectedNbInputs = kNUM_REQUIRED_INPUTS + (mUseSpecVerifyState ? kNUM_SPEC_VERIFY_OPTIONAL_INPUTS : 0)
        + (mUseDDTree ? kNUM_DDTREE_OPTIONAL_INPUTS : 0);
    if (nbInputs != expectedNbInputs || nbOutputs != expectedNbOutputs)
        return false;
    if (inOut[pos].desc.format != TensorFormat::kLINEAR)
        return false;
    if (pos == kIN_A_LOG_IDX || pos == kIN_H0_SOURCE_IDX)
        return inOut[pos].desc.type == DataType::kFLOAT;
    if (pos == kIN_CONTEXT_LENGTHS_IDX)
        return inOut[pos].desc.type == DataType::kINT32;
    if (mUseSpecVerifyState && pos == kIN_SPEC_VERIFY_PHASE_MARKER_IDX)
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
    int32_t const expectedNbInputs = kNUM_REQUIRED_INPUTS + (mUseSpecVerifyState ? kNUM_SPEC_VERIFY_OPTIONAL_INPUTS : 0)
        + (mUseDDTree ? kNUM_DDTREE_OPTIONAL_INPUTS : 0);
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
    if (in[kIN_Q_IDX].desc.dims.nbDims != 4 || in[kIN_V_IDX].desc.dims.nbDims != 4)
    {
        LOG_ERROR("gated_delta_net: Q and V must be 4D");
        return -1;
    }
    if (in[kIN_CONTEXT_LENGTHS_IDX].desc.type != DataType::kINT32 || in[kIN_CONTEXT_LENGTHS_IDX].desc.dims.nbDims != 1)
    {
        LOG_ERROR("gated_delta_net: context_lengths must be 1D INT32");
        return -1;
    }
    if (mUseSpecVerifyState
        && (in[kIN_SPEC_VERIFY_PHASE_MARKER_IDX].desc.type != DataType::kINT32
            || in[kIN_SPEC_VERIFY_PHASE_MARKER_IDX].desc.dims.nbDims != 1))
    {
        LOG_ERROR("gated_delta_net: spec_verify_phase_marker must be 1D INT32");
        return -1;
    }
    if (mUseDDTree
        && (in[kIN_TREE_PARENT_IDS_IDX].desc.type != DataType::kINT32
            || in[kIN_TREE_DEPTHS_IDX].desc.type != DataType::kINT32
            || in[kIN_TREE_PARENT_IDS_IDX].desc.dims.nbDims != 2 || in[kIN_TREE_DEPTHS_IDX].desc.dims.nbDims != 2))
    {
        LOG_ERROR("gated_delta_net: DDTree tree_parent_ids/tree_depths must be 2D INT32");
        return -1;
    }
    return 0;
}

size_t GatedDeltaNetPlugin::getWorkspaceSize([[maybe_unused]] DynamicPluginTensorDesc const* inputs,
    [[maybe_unused]] int32_t nbInputs, [[maybe_unused]] DynamicPluginTensorDesc const* outputs,
    [[maybe_unused]] int32_t nbOutputs) const noexcept
{
    size_t total = 0;

#ifdef CUTE_DSL_GDN_BLACKWELL_ENABLED
    int32_t const maxN = static_cast<int32_t>(inputs[kIN_CONTEXT_LENGTHS_IDX].max.d[0]);
    int32_t const maxHv = static_cast<int32_t>(inputs[kIN_H0_SOURCE_IDX].max.d[1]);
    int32_t const kDim = static_cast<int32_t>(inputs[kIN_H0_SOURCE_IDX].max.d[2]);
    int32_t const vDim = static_cast<int32_t>(inputs[kIN_H0_SOURCE_IDX].max.d[3]);

    // cu_seqlens [maxN+1] int32, padded to 128-byte alignment.
    size_t const cuSeqBytes = static_cast<size_t>(maxN + 1) * sizeof(int32_t);
    size_t const cuSeqPadded = alignTensorSize(cuSeqBytes);
    // h0 scratch [maxN, maxHv, kDim, vDim] f32 — separate buffer for Blackwell h0_out.
    size_t const h0ScratchBytes = static_cast<size_t>(maxN) * maxHv * kDim * vDim * sizeof(float);

    total = cuSeqPadded + h0ScratchBytes;
#endif

#ifdef CUTE_DSL_GDN_BLACKWELL_GEFORCE_ENABLED
    // Workspace sizes are serialized with the engine, so reserve a fixed
    // architecture-wide upper bound rather than the build GPU's SM count.
    size_t const blackwellGeforceTensorMapBytes = static_cast<size_t>(CuteDslGDNRunner::kBlackwellGeforceMaxSMCount)
        * CuteDslGDNRunner::kBlackwellGeforceTensorMapDescriptorBytes;
    total = std::max(total, blackwellGeforceTensorMapBytes);
#endif

    if (mUseDDTree)
    {
        int32_t const maxN = static_cast<int32_t>(inputs[kIN_Q_IDX].max.d[0]);
        int32_t const maxSeqLen = static_cast<int32_t>(inputs[kIN_Q_IDX].max.d[1]);

        // Chunk-form verify uses ancestor masks. The KS/QS + prep scratch
        // lives in the intermediate-states row tail, NOT here — engine plans
        // serialize this size at build time, so growing it silently overflows
        // on pre-existing engines.
        int32_t const chunkNodes = std::min(maxSeqLen, kernel::kGDN_TREE_CHUNK_MAX_NODES);
        size_t const maskBytes = alignTensorSize(
            static_cast<size_t>(maxN) * chunkNodes * kernel::kGDN_TREE_CHUNK_MASK_WORDS * sizeof(uint32_t));
        total = std::max(total, maskBytes);
    }

    return total;
}

int32_t GatedDeltaNetPlugin::getAliasedInput(int32_t outputIndex) noexcept
{
    // WAR: this is not the correct plugin API usage. The
    // plugin updates the recurrent state in place, so the correct return is the
    // recurrent-state input index. We return -1 to drop the alias because
    // declaring it makes Myelin keep a redundant per-layer state copy (the perf
    // regression). In-place read-write still works because the runtime binds the
    // past and present state to the same buffer. TODO: restore the alias
    // declaration once the Myelin issue is fixed.
    return -1;
}

// ---------------------------------------------------------------------------
// IPluginV3OneRuntime — execution
// ---------------------------------------------------------------------------
#ifdef CUTE_DSL_GDN_ENABLED
int32_t GatedDeltaNetPlugin::enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* /* outputDesc */,
    void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    int64_t const* qDims = inputDesc[kIN_Q_IDX].dims.d;
    int32_t const n = static_cast<int32_t>(qDims[0]);
    int32_t const seq_len = static_cast<int32_t>(qDims[1]);
    int32_t const h = static_cast<int32_t>(qDims[2]);
    int32_t const k_dim = static_cast<int32_t>(qDims[3]);

    int64_t const* vDims = inputDesc[kIN_V_IDX].dims.d;
    int32_t const hv = static_cast<int32_t>(vDims[2]);
    int32_t const v_dim = static_cast<int32_t>(vDims[3]);

    constexpr int32_t kLinearSpecVerifyMaxSeqLen = 16;
    // Shape-only phase marker: length 0 is ordinary prefill/decode, length 1 is speculative verify.
    // The marker payload is ignored.
    int32_t const phaseLen
        = mUseSpecVerifyState ? static_cast<int32_t>(inputDesc[kIN_SPEC_VERIFY_PHASE_MARKER_IDX].dims.d[0]) : 0;
    if (phaseLen > 1)
    {
        LOG_ERROR("gated_delta_net: spec_verify_phase_marker length must be 0 or 1, got %d", phaseLen);
        return -1;
    }
    bool const ddtreeActive = mUseDDTree && phaseLen > 0;
    bool const mtpActive = mUseSpecVerifyState && phaseLen > 0 && !ddtreeActive;
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
        if (seq_len < 1 || parentDesc.dims.nbDims != 2 || depthDesc.dims.nbDims != 2 || parentDesc.dims.d[0] != n
            || depthDesc.dims.d[0] != n || parentDesc.dims.d[1] != seq_len || depthDesc.dims.d[1] != seq_len)
        {
            LOG_ERROR(
                "gated_delta_net: DDTree requires tree_parent_ids/tree_depths shape [n=%d, seq_len=%d]; got "
                "parent nbDims=%d [%lld, %lld], depth nbDims=%d [%lld, %lld]",
                n, seq_len, parentDesc.dims.nbDims,
                parentDesc.dims.nbDims > 0 ? static_cast<long long>(parentDesc.dims.d[0]) : -1LL,
                parentDesc.dims.nbDims > 1 ? static_cast<long long>(parentDesc.dims.d[1]) : -1LL, depthDesc.dims.nbDims,
                depthDesc.dims.nbDims > 0 ? static_cast<long long>(depthDesc.dims.d[0]) : -1LL,
                depthDesc.dims.nbDims > 1 ? static_cast<long long>(depthDesc.dims.d[1]) : -1LL);
            return -1;
        }
    }

    GDNParams params{};
    params.seq_len = seq_len;
    params.h = h;
    params.hv = hv;
    params.smVersion = mSMVersion;
    params.use_mtp = mtpActive;
    if (!ddtreeActive && !CuteDslGDNRunner::ensureKernelModules(params, stream))
    {
        LOG_ERROR("gated_delta_net: failed to load the selected CuTe DSL GDN module");
        return -1;
    }

    // h0 is batch-dense [n, hv, k, v]
    size_t const h0Bytes = static_cast<size_t>(n) * hv * static_cast<size_t>(k_dim) * v_dim * sizeof(float);
    void* h0Out = outputs[kOUT_H0_SOURCE_IDX];

    // DDTree verify keeps the committed recurrent state read-only and commits
    // from intermediate_states after accept, so it does not need an h0_out copy.
    void* h0State = ddtreeActive ? const_cast<void*>(inputs[kIN_H0_SOURCE_IDX]) : h0Out;

    // For linear spec verify, the kernel updates h0_source in-place, so we always need the copy
    // (h0Out serves as the working state buffer that the kernel reads/writes).
    // For normal: same logic as before — copy if input != output.
    if (!ddtreeActive && h0Out != inputs[kIN_H0_SOURCE_IDX])
    {
        cudaMemcpyAsync(h0Out, inputs[kIN_H0_SOURCE_IDX], h0Bytes, cudaMemcpyDeviceToDevice, stream);
    }

    if (ddtreeActive && !kernel::gdnTreeChunkVerifyEnabled(seq_len))
    {
        LOG_ERROR("gated_delta_net: DDTree chunk-form verify supports seq_len <= %d, got %d",
            kernel::kGDN_TREE_CHUNK_MAX_NODES, seq_len);
        return -1;
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
        cudaError_t const verifyErr = kernel::gdnTreeVerifyChunk(static_cast<float const*>(inputs[kIN_H0_SOURCE_IDX]),
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
    params.o = outputs[kOUT_O_IDX];
    params.n = n;
    params.seq_len = seq_len;
    params.h = h;
    params.hv = hv;
    params.k_dim = k_dim;
    params.v_dim = v_dim;
    params.smVersion = mSMVersion;

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
        if (seq_len > 1 && (mSMVersion == 100 || mSMVersion == 101 || mSMVersion == 110))
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
        if (seq_len > 1 && (mSMVersion == 120 || mSMVersion == 121))
        {
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

int32_t GatedDeltaNetPlugin::onShapeChange(PluginTensorDesc const* /* in */, int32_t /* nbInputs */,
    PluginTensorDesc const* /* out */, int32_t /* nbOutputs */) noexcept
{
    return 0;
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
