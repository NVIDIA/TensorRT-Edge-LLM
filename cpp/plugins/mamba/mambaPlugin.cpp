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

#include "mambaPlugin.h"

#include "common/cudaUtils.h"
#include "common/logger.h"
#include "kernels/mamba/selectiveStateUpdate.h"
#ifdef CUTE_DSL_SSD_ENABLED
#include "kernels/mamba/cuteDslSSDRunner.h"
#endif
#include "plugins/utils/pluginUtils.h"

#include "common/checkMacros.h"
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cuda_fp16.h>
#include <mutex>
#include <optional>

using namespace nvinfer1;

namespace trt_edgellm
{
namespace plugins
{

namespace
{
constexpr char const* kMAMBA_PLUGIN_VERSION{"1"};
constexpr char const* kMAMBA_PLUGIN_NAME{"update_ssm_state"};

// Input indices – matches the trt_edgellm::update_ssm_state ONNX op.
// x, dt, B, C may carry an optional seq_len dimension (4D instead of 3D).
// When seq_len > 1, the plugin loops over the single-step kernel.
constexpr int32_t kIN_X_IDX{0};                 // [batch, (seq_len,) nheads, dim]
constexpr int32_t kIN_A_IDX{1};                 // [nheads]
constexpr int32_t kIN_B_IDX{2};                 // [batch, (seq_len,) ngroups, dstate]
constexpr int32_t kIN_C_IDX{3};                 // [batch, (seq_len,) ngroups, dstate]
constexpr int32_t kIN_D_IDX{4};                 // [nheads]
constexpr int32_t kIN_DT_IDX{5};                // [batch, (seq_len,) nheads]
constexpr int32_t kIN_DT_BIAS_IDX{6};           // [nheads]
constexpr int32_t kIN_STATE_IDX{7};             // [batch, nheads, dim, dstate]
constexpr int32_t kIN_CONTEXT_LENGTHS_IDX{8};   // [batch]
constexpr int32_t kIN_STATE_START_INDEX_IDX{9}; // [0] for cold prefill, [batch] for restored state
// Optional trailing input, present only in spec-verify mode. Shape-only INT32: length 0 == ordinary
// prefill/decode, length 1 == speculative verify (capture per-token intermediate states).
constexpr int32_t kIN_SPEC_VERIFY_PHASE_MARKER_IDX{10};

// Output indices
constexpr int32_t kOUT_OUTPUT_IDX{0}; // [batch, (seq_len,) nheads, dim]
constexpr int32_t kOUT_STATE_IDX{1};  // [batch, nheads, dim, dstate]
// MTP spec-verify (replay): per-token replay stash (FP32) — see selectiveStateUpdate.h. The runtime
// reconstructs the accepted recurrent state from these after verification instead of scattering a
// per-token full-state snapshot.
constexpr int32_t kOUT_REPLAY_DA_IDX{2}; // [batch, seq_len, nheads]
constexpr int32_t kOUT_REPLAY_U_IDX{3};  // [batch, seq_len, nheads, dim]
constexpr int32_t kOUT_REPLAY_B_IDX{4};  // [batch, seq_len, ngroups, dstate]

// Number of inputs/outputs
// Base I/O = 10 required inputs (through kIN_STATE_START_INDEX_IDX) + 2 required outputs.
// Spec-verify mode adds one optional input (phase marker) and the three replay-stash outputs.
constexpr int32_t kNUM_REQUIRED_INPUTS{10};
constexpr int32_t kNUM_SPEC_VERIFY_OPTIONAL_INPUTS{1};
constexpr int32_t kNUM_REQUIRED_OUTPUTS{2};
constexpr int32_t kNUM_SPEC_VERIFY_OPTIONAL_OUTPUTS{3};

} // namespace

// Static class fields initialization
PluginFieldCollection MambaPluginCreator::mFieldCollection{};
std::vector<PluginField> MambaPluginCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(MambaPluginCreator);

MambaPlugin::MambaPlugin(std::string const& name, int32_t dim, int32_t dstate, int32_t nheads, int32_t ngroups,
    int32_t dtSoftplus, int32_t useSpecVerifyState)
    : mLayerName(name)
    , mDim(dim)
    , mDstate(dstate)
    , mNheads(nheads)
    , mNgroups(ngroups)
    , mDtSoftplus(dtSoftplus)
    , mUseSpecVerifyState(useSpecVerifyState)
{
}

MambaPlugin::~MambaPlugin() {}

int32_t MambaPlugin::numInputs() const noexcept
{
    return kNUM_REQUIRED_INPUTS + (mUseSpecVerifyState ? kNUM_SPEC_VERIFY_OPTIONAL_INPUTS : 0);
}

int32_t MambaPlugin::numOutputs() const noexcept
{
    return kNUM_REQUIRED_OUTPUTS + (mUseSpecVerifyState ? kNUM_SPEC_VERIFY_OPTIONAL_OUTPUTS : 0);
}

IPluginCapability* MambaPlugin::getCapabilityInterface(PluginCapabilityType type) noexcept
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

IPluginV3* MambaPlugin::clone() noexcept
{
    MambaPlugin* plugin
        = new MambaPlugin(mLayerName, mDim, mDstate, mNheads, mNgroups, mDtSoftplus, mUseSpecVerifyState);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

char const* MambaPlugin::getPluginName() const noexcept
{
    return kMAMBA_PLUGIN_NAME;
}

char const* MambaPlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

void MambaPlugin::setPluginNamespace(char const* pluginNamespace) noexcept
{
    mNamespace = std::string(pluginNamespace);
}

char const* MambaPlugin::getPluginVersion() const noexcept
{
    return kMAMBA_PLUGIN_VERSION;
}

int32_t MambaPlugin::getNbOutputs() const noexcept
{
    return numOutputs();
}

int32_t MambaPlugin::getOutputDataTypes(DataType* outputTypes, [[maybe_unused]] int32_t nbOutputs,
    DataType const* inputTypes, [[maybe_unused]] int32_t nbInputs) const noexcept
{
    outputTypes[kOUT_OUTPUT_IDX] = inputTypes[kIN_X_IDX];
    outputTypes[kOUT_STATE_IDX] = inputTypes[kIN_X_IDX];
    if (mUseSpecVerifyState)
    {
        // Replay stash is FP32 so reconstruction reproduces the fp32 forward scan faithfully.
        outputTypes[kOUT_REPLAY_DA_IDX] = DataType::kFLOAT;
        outputTypes[kOUT_REPLAY_U_IDX] = DataType::kFLOAT;
        outputTypes[kOUT_REPLAY_B_IDX] = DataType::kFLOAT;
    }
    return 0;
}

int32_t MambaPlugin::getOutputShapes(DimsExprs const* inputs, [[maybe_unused]] int32_t nbInputs,
    DimsExprs const* /* shapeInputs */, int32_t /* nbShapeInputs */, DimsExprs* outputs,
    [[maybe_unused]] int32_t nbOutputs, IExprBuilder& /* exprBuilder */) noexcept
{
    // Output: same shape as x [batch, (seq_len,) nheads, dim]
    outputs[kOUT_OUTPUT_IDX].nbDims = inputs[kIN_X_IDX].nbDims;
    for (int32_t i = 0; i < outputs[kOUT_OUTPUT_IDX].nbDims; ++i)
    {
        outputs[kOUT_OUTPUT_IDX].d[i] = inputs[kIN_X_IDX].d[i];
    }
    // State output: same shape as state input [batch, nheads, dim, dstate]
    outputs[kOUT_STATE_IDX].nbDims = inputs[kIN_STATE_IDX].nbDims;
    for (int32_t i = 0; i < outputs[kOUT_STATE_IDX].nbDims; ++i)
    {
        outputs[kOUT_STATE_IDX].d[i] = inputs[kIN_STATE_IDX].d[i];
    }
    // Spec-verify replay stash. dA [batch, seq_len, nheads]; u [batch, seq_len, nheads, dim];
    // B [batch, seq_len, ngroups, dstate].
    if (mUseSpecVerifyState)
    {
        outputs[kOUT_REPLAY_DA_IDX].nbDims = 3;
        outputs[kOUT_REPLAY_DA_IDX].d[0] = inputs[kIN_X_IDX].d[0];     // batch
        outputs[kOUT_REPLAY_DA_IDX].d[1] = inputs[kIN_X_IDX].d[1];     // seq_len
        outputs[kOUT_REPLAY_DA_IDX].d[2] = inputs[kIN_STATE_IDX].d[1]; // nheads

        outputs[kOUT_REPLAY_U_IDX].nbDims = 4;
        outputs[kOUT_REPLAY_U_IDX].d[0] = inputs[kIN_X_IDX].d[0];     // batch
        outputs[kOUT_REPLAY_U_IDX].d[1] = inputs[kIN_X_IDX].d[1];     // seq_len
        outputs[kOUT_REPLAY_U_IDX].d[2] = inputs[kIN_STATE_IDX].d[1]; // nheads
        outputs[kOUT_REPLAY_U_IDX].d[3] = inputs[kIN_STATE_IDX].d[2]; // dim

        outputs[kOUT_REPLAY_B_IDX].nbDims = 4;
        outputs[kOUT_REPLAY_B_IDX].d[0] = inputs[kIN_X_IDX].d[0];                            // batch
        outputs[kOUT_REPLAY_B_IDX].d[1] = inputs[kIN_X_IDX].d[1];                            // seq_len
        outputs[kOUT_REPLAY_B_IDX].d[2] = inputs[kIN_B_IDX].d[inputs[kIN_B_IDX].nbDims - 2]; // ngroups
        outputs[kOUT_REPLAY_B_IDX].d[3] = inputs[kIN_STATE_IDX].d[3];                        // dstate
    }
    return 0;
}

bool MambaPlugin::supportsFormatCombination(
    int32_t pos, DynamicPluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
    if (nbOutputs != numOutputs() || nbInputs != numInputs())
        return false;
    auto const& desc = inOut[pos].desc;
    if (desc.format != TensorFormat::kLINEAR)
        return false;
    // Optional spec-verify phase-marker input (INT32).
    if (mUseSpecVerifyState && pos == kIN_SPEC_VERIFY_PHASE_MARKER_IDX)
        return desc.type == DataType::kINT32;
    if (pos >= nbInputs)
    {
        // Token/state outputs follow x's type; the spec-verify replay stash (dA/u/B) is FP32.
        int32_t const outIdx = pos - nbInputs;
        if (mUseSpecVerifyState
            && (outIdx == kOUT_REPLAY_DA_IDX || outIdx == kOUT_REPLAY_U_IDX || outIdx == kOUT_REPLAY_B_IDX))
            return desc.type == DataType::kFLOAT;
        return desc.type == inOut[kIN_X_IDX].desc.type;
    }
    switch (pos)
    {
    case kIN_X_IDX:
    case kIN_B_IDX:
    case kIN_C_IDX:
    case kIN_D_IDX:
    case kIN_DT_IDX:
    case kIN_DT_BIAS_IDX:
    case kIN_STATE_IDX: return desc.type == DataType::kHALF;
    case kIN_A_IDX: return desc.type == DataType::kFLOAT;
    case kIN_CONTEXT_LENGTHS_IDX:
    case kIN_STATE_START_INDEX_IDX: return desc.type == DataType::kINT32;
    default: return false;
    }
}

int32_t MambaPlugin::configurePlugin(DynamicPluginTensorDesc const* in, int32_t nbInputs,
    [[maybe_unused]] DynamicPluginTensorDesc const* out, [[maybe_unused]] int32_t nbOutputs) noexcept
{
    if (nbInputs != numInputs())
    {
        LOG_ERROR("update_ssm_state: expected %d inputs, got %d", numInputs(), nbInputs);
        return -1;
    }

    // Derive dim/dstate/nheads/ngroups from input shapes if not provided as attributes.
    // x: [batch, (seq_len,) nheads, dim]  -> last two dims
    // B: [batch, (seq_len,) ngroups, dstate] -> last two dims
    auto const& xMax = in[kIN_X_IDX].max;
    auto const& bMax = in[kIN_B_IDX].max;

    int32_t const xNDims = xMax.nbDims;
    int32_t const bNDims = bMax.nbDims;

    if (mDim == 0)
    {
        mDim = static_cast<int32_t>(xMax.d[xNDims - 1]);
    }
    if (mNheads == 0)
    {
        mNheads = static_cast<int32_t>(xMax.d[xNDims - 2]);
    }
    if (mDstate == 0)
    {
        mDstate = static_cast<int32_t>(bMax.d[bNDims - 1]);
    }
    if (mNgroups == 0)
    {
        mNgroups = static_cast<int32_t>(bMax.d[bNDims - 2]);
    }
    if (in[kIN_X_IDX].desc.type != DataType::kHALF)
    {
        LOG_ERROR("update_ssm_state: only FP16 input is supported; got type %d",
            static_cast<int32_t>(in[kIN_X_IDX].desc.type));
        return -1;
    }
    return 0;
}

size_t MambaPlugin::getWorkspaceSize([[maybe_unused]] DynamicPluginTensorDesc const* inputs, int32_t /* nbInputs */,
    DynamicPluginTensorDesc const* /* outputs */, int32_t /* nbOutputs */) const noexcept
{
#ifdef CUTE_DSL_SSD_ENABLED
    auto const& xDesc = inputs[kIN_X_IDX];
    if (xDesc.desc.dims.nbDims == 4)
    {
        int32_t const batch = static_cast<int32_t>(xDesc.max.d[0]);
        int32_t const seqLen = static_cast<int32_t>(xDesc.max.d[1]);
        if (seqLen >= 128)
        {
            return trt_edgellm::CuteDslSSDRunner::getWorkspaceSize(batch, seqLen, mNheads, mDim, mDstate, mNgroups);
        }
    }
#endif
    return 0;
}

int32_t MambaPlugin::enqueue(nvinfer1::PluginTensorDesc const* inputDesc, nvinfer1::PluginTensorDesc const* outputDesc,
    void const* const* inputs, void* const* outputs, [[maybe_unused]] void* workspace, cudaStream_t stream) noexcept
{
    auto const& xDesc = inputDesc[kIN_X_IDX];
    size_t const elemSize = sizeof(half);

    int32_t const batch = static_cast<int32_t>(xDesc.dims.d[0]);
    if (inputDesc[kIN_STATE_START_INDEX_IDX].dims.nbDims != 1
        || (inputDesc[kIN_STATE_START_INDEX_IDX].dims.d[0] != 0
            && inputDesc[kIN_STATE_START_INDEX_IDX].dims.d[0] != batch))
    {
        LOG_ERROR("update_ssm_state: state_start_index must have shape [0] or [batch]");
        return -1;
    }

    // Determine seq_len: x is [batch, nheads, dim] (3D) or [batch, seq_len, nheads, dim] (4D)
    bool const hasSeqLen = (xDesc.dims.nbDims == 4);

#ifdef CUTE_DSL_SSD_ENABLED
    if (hasSeqLen)
    {
        int32_t const seqLen = static_cast<int32_t>(xDesc.dims.d[1]);
        int32_t const smVersion = getSMVersion();
        if (trt_edgellm::CuteDslSSDRunner::canImplement(mDim, mDstate, smVersion) && seqLen >= 128)
        {
            trt_edgellm::SSDParams moduleParams{};
            moduleParams.dim = mDim;
            moduleParams.dstate = mDstate;
            moduleParams.smVersion = smVersion;
            moduleParams.has_init_states = inputDesc[kIN_STATE_START_INDEX_IDX].dims.d[0] > 0;
            if (!trt_edgellm::CuteDslSSDRunner::ensureKernelModules(moduleParams, stream))
            {
                LOG_ERROR("Failed to load the selected CuTe DSL SSD kernel module");
                return -1;
            }
        }
    }
#endif

    // Copy input state to output state so the kernel can update in-place across steps.
    void* outputState = outputs[kOUT_STATE_IDX];
    if (outputState != inputs[kIN_STATE_IDX])
    {
        size_t stateSize = static_cast<size_t>(batch) * mNheads * mDim * mDstate * elemSize;
        cudaMemcpyAsync(outputState, inputs[kIN_STATE_IDX], stateSize, cudaMemcpyDeviceToDevice, stream);
    }

    namespace rt = trt_edgellm::rt;

    // Non-owning tensor views for inputs (const_cast is safe: non-owning, read-only inside kernels)
    auto xTensor
        = rt::Tensor{const_cast<void*>(inputs[kIN_X_IDX]), inputDesc[kIN_X_IDX].dims, rt::DeviceType::kGPU, xDesc.type};
    auto aTensor = rt::Tensor{const_cast<void*>(inputs[kIN_A_IDX]), inputDesc[kIN_A_IDX].dims, rt::DeviceType::kGPU,
        inputDesc[kIN_A_IDX].type};
    auto bTensor
        = rt::Tensor{const_cast<void*>(inputs[kIN_B_IDX]), inputDesc[kIN_B_IDX].dims, rt::DeviceType::kGPU, xDesc.type};
    auto cTensor
        = rt::Tensor{const_cast<void*>(inputs[kIN_C_IDX]), inputDesc[kIN_C_IDX].dims, rt::DeviceType::kGPU, xDesc.type};
    auto dtTensor = rt::Tensor{
        const_cast<void*>(inputs[kIN_DT_IDX]), inputDesc[kIN_DT_IDX].dims, rt::DeviceType::kGPU, xDesc.type};
    auto dtBiasTensor = rt::Tensor{
        const_cast<void*>(inputs[kIN_DT_BIAS_IDX]), inputDesc[kIN_DT_BIAS_IDX].dims, rt::DeviceType::kGPU, xDesc.type};

    // Optional D — keep tensor in scope for the duration of the invoke call
    std::optional<rt::Tensor> dTensorOpt;
    if (inputs[kIN_D_IDX])
    {
        dTensorOpt.emplace(
            const_cast<void*>(inputs[kIN_D_IDX]), inputDesc[kIN_D_IDX].dims, rt::DeviceType::kGPU, xDesc.type);
    }
    rt::OptionalInputTensor dOpt = dTensorOpt.has_value() ? std::optional(std::cref(dTensorOpt.value())) : std::nullopt;

    // Output tensors (mutable)
    auto stateTensor = rt::Tensor{outputState, inputDesc[kIN_STATE_IDX].dims, rt::DeviceType::kGPU, xDesc.type};
    auto outTensor
        = rt::Tensor{outputs[kOUT_OUTPUT_IDX], outputDesc[kOUT_OUTPUT_IDX].dims, rt::DeviceType::kGPU, xDesc.type};

    rt::OptionalInputTensor dtBiasOpt = std::optional(std::cref(dtBiasTensor));
    bool const dt_softplus = static_cast<bool>(mDtSoftplus);

    if (hasSeqLen)
    {
        int32_t const seqLen = static_cast<int32_t>(xDesc.dims.d[1]);

        // MTP spec-verify: capture per-token recurrent snapshots only in the verify phase (shape-only
        // marker length 1). The scalar chunked-scan fallback below does the capture, so the CuTe DSL SSD
        // path is disabled when capturing (verify windows are small, well under the 128-token SSD floor).
        bool const captureIntermediates
            = mUseSpecVerifyState && static_cast<int32_t>(inputDesc[kIN_SPEC_VERIFY_PHASE_MARKER_IDX].dims.d[0]) > 0;

#ifdef CUTE_DSL_SSD_ENABLED
        // CuTe DSL path: chunked SSD prefill, requires seq_len >= 128.
        bool usedCuteDsl = false;
        {
            int32_t const smVersion = getSMVersion();
            if (trt_edgellm::CuteDslSSDRunner::canImplement(mDim, mDstate, smVersion) && seqLen >= 128
                && !captureIntermediates)
            {
                trt_edgellm::CuteDslSSDRunner runner;
                trt_edgellm::SSDParams ssdParams{};
                ssdParams.x = const_cast<void*>(inputs[kIN_X_IDX]);
                ssdParams.dt = const_cast<void*>(inputs[kIN_DT_IDX]);
                ssdParams.A = const_cast<void*>(inputs[kIN_A_IDX]);
                ssdParams.B = const_cast<void*>(inputs[kIN_B_IDX]);
                ssdParams.C = const_cast<void*>(inputs[kIN_C_IDX]);
                ssdParams.D = const_cast<void*>(inputs[kIN_D_IDX]);
                ssdParams.dt_bias = const_cast<void*>(inputs[kIN_DT_BIAS_IDX]);
                ssdParams.z = nullptr;
                ssdParams.state = outputState;
                ssdParams.output = outputs[kOUT_OUTPUT_IDX];
                ssdParams.workspace = workspace;
                ssdParams.batch = batch;
                ssdParams.seq_len = seqLen;
                ssdParams.nheads = mNheads;
                ssdParams.dim = mDim;
                ssdParams.dstate = mDstate;
                ssdParams.ngroups = mNgroups;
                ssdParams.smVersion = smVersion;
                ssdParams.dt_softplus = dt_softplus;
                ssdParams.has_D = (inputs[kIN_D_IDX] != nullptr);
                ssdParams.has_z = false;
                ssdParams.context_lengths = inputs[kIN_CONTEXT_LENGTHS_IDX];
                // [0] is the runtime's initial-prefill sentinel. A non-empty start-index binding means at least one
                // sequence carries restored recurrent state; the init-state kernel safely handles zero-state peers.
                ssdParams.has_init_states = inputDesc[kIN_STATE_START_INDEX_IDX].dims.d[0] > 0;
                int const rc = runner.run(ssdParams, stream);
                if (rc != 0)
                {
                    LOG_ERROR("CuTe DSL SSD prefill failed with error %d", rc);
                    return rc;
                }
                usedCuteDsl = true;
            }
        }
        if (!usedCuteDsl)
#endif
        {
            // Only use context_lengths for actual prefill (seqLen > 1).
            // During decode, x is 4D with seqLen=1 but context_lengths holds the
            // cumulative length which would cause an out-of-bounds scan.
            // MTP verify also has seqLen > 1, but it is a decode-like step: the SSM must scan
            // exactly the seqLen new candidate tokens continuing from the committed recurrent
            // state, NOT the cumulative context_lengths. Suppress it whenever we are capturing
            // per-token intermediates (i.e. the spec-verify phase).
            rt::OptionalInputTensor contextLengthsOpt = std::nullopt;
            std::optional<rt::Tensor> clTensorOpt;
            if (seqLen > 1 && !captureIntermediates && inputs[kIN_CONTEXT_LENGTHS_IDX])
            {
                clTensorOpt.emplace(const_cast<void*>(inputs[kIN_CONTEXT_LENGTHS_IDX]), rt::Coords{batch},
                    rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);
                contextLengthsOpt = std::optional(std::cref(clTensorOpt.value()));
            }

            // MTP spec-verify (replay): stash per-token dA/u/B (FP32) during verify; the runtime
            // reconstructs the accepted recurrent state from them afterwards. The committed state is
            // left read-only by the kernel while the stash is active.
            rt::OptionalOutputTensor replayDaOpt = std::nullopt;
            rt::OptionalOutputTensor replayUOpt = std::nullopt;
            rt::OptionalOutputTensor replayBOpt = std::nullopt;
            std::optional<rt::Tensor> replayDaTensorOpt;
            std::optional<rt::Tensor> replayUTensorOpt;
            std::optional<rt::Tensor> replayBTensorOpt;
            if (captureIntermediates && outputs[kOUT_REPLAY_U_IDX] != nullptr)
            {
                replayDaTensorOpt.emplace(outputs[kOUT_REPLAY_DA_IDX], rt::Coords{batch, seqLen, mNheads},
                    rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT);
                replayUTensorOpt.emplace(outputs[kOUT_REPLAY_U_IDX], rt::Coords{batch, seqLen, mNheads, mDim},
                    rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT);
                replayBTensorOpt.emplace(outputs[kOUT_REPLAY_B_IDX], rt::Coords{batch, seqLen, mNgroups, mDstate},
                    rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT);
                replayDaOpt = std::optional(std::ref(replayDaTensorOpt.value()));
                replayUOpt = std::optional(std::ref(replayUTensorOpt.value()));
                replayBOpt = std::optional(std::ref(replayBTensorOpt.value()));
            }

            mamba_ssm::invokeSelectiveStateUpdatePrefill(xTensor, aTensor, bTensor, cTensor, dtTensor, dtBiasOpt, dOpt,
                std::nullopt, stateTensor, outTensor, dt_softplus, contextLengthsOpt, replayDaOpt, replayUOpt,
                replayBOpt, stream);
        }
    }
    else
    {
        mamba_ssm::invokeSelectiveStateUpdate(xTensor, aTensor, bTensor, cTensor, dtTensor, dtBiasOpt, dOpt,
            std::nullopt, stateTensor, outTensor, dt_softplus, stream);
    }

    return 0;
}

int32_t MambaPlugin::onShapeChange(PluginTensorDesc const* /* in */, int32_t /* nbInputs */,
    PluginTensorDesc const* /* out */, int32_t /* nbOutputs */) noexcept
{
    return 0;
}

IPluginV3* MambaPlugin::attachToContext(IPluginResourceContext* /* context */) noexcept
{
    return clone();
}

PluginFieldCollection const* MambaPlugin::getFieldsToSerialize() noexcept
{
    mDataToSerialize.clear();
    mDataToSerialize.emplace_back("dim", &mDim, PluginFieldType::kINT32, 1);
    mDataToSerialize.emplace_back("dstate", &mDstate, PluginFieldType::kINT32, 1);
    mDataToSerialize.emplace_back("nheads", &mNheads, PluginFieldType::kINT32, 1);
    mDataToSerialize.emplace_back("ngroups", &mNgroups, PluginFieldType::kINT32, 1);
    mDataToSerialize.emplace_back("dt_softplus", &mDtSoftplus, PluginFieldType::kINT32, 1);
    mDataToSerialize.emplace_back("use_spec_verify_state", &mUseSpecVerifyState, PluginFieldType::kINT32, 1);
    mFCToSerialize.nbFields = static_cast<int32_t>(mDataToSerialize.size());
    mFCToSerialize.fields = mDataToSerialize.data();
    return &mFCToSerialize;
}

// Plugin Creator implementation.

MambaPluginCreator::MambaPluginCreator()
{
    static std::mutex sMutex;
    std::lock_guard<std::mutex> lock(sMutex);

    // TRT warns for every declared attribute not found in the ONNX node, so
    // we only declare attributes that the trt_edgellm::update_ssm_state ONNX op emits.
    // See createPlugin() for the full attribute breakdown.
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("chunk_size", nullptr, PluginFieldType::kINT32, 0));
    mPluginAttributes.emplace_back(PluginField("time_step_limit", nullptr, PluginFieldType::kFLOAT32, 0));
    mPluginAttributes.emplace_back(PluginField("use_spec_verify_state", nullptr, PluginFieldType::kINT32, 1));
    mFieldCollection.nbFields = mPluginAttributes.size();
    mFieldCollection.fields = mPluginAttributes.data();
}

char const* MambaPluginCreator::getPluginName() const noexcept
{
    return kMAMBA_PLUGIN_NAME;
}

PluginFieldCollection const* MambaPluginCreator::getFieldNames() noexcept
{
    return &mFieldCollection;
}

void MambaPluginCreator::setPluginNamespace(char const* libNamespace) noexcept
{
    mNamespace = libNamespace;
}

char const* MambaPluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

char const* MambaPluginCreator::getPluginVersion() const noexcept
{
    return kMAMBA_PLUGIN_VERSION;
}

IPluginV3* MambaPluginCreator::createPlugin(
    char const* name, PluginFieldCollection const* fc, TensorRTPhase phase) noexcept
{
    try
    {
        // dim, dstate, nheads, ngroups: inferred from input tensor shapes in configurePlugin
        // when not provided (value 0 = "derive from shapes"). At kRUNTIME they come from
        // getFieldsToSerialize and will be non-zero.
        std::optional<int32_t> dim = parsePluginScalarField<int32_t>("dim", fc);
        std::optional<int32_t> dstate = parsePluginScalarField<int32_t>("dstate", fc);
        std::optional<int32_t> nheads = parsePluginScalarField<int32_t>("nheads", fc);
        std::optional<int32_t> ngroups = parsePluginScalarField<int32_t>("ngroups", fc);
        // dt_softplus: apply softplus to dt before discretization (0=off, 1=on).
        //   Default=1 matches Nemotron and most Mamba models.
        std::optional<int32_t> dtSoftplus = parsePluginScalarField<int32_t>("dt_softplus", fc);
        // use_spec_verify_state: emit per-token intermediate recurrent states for MTP verification (0=off).
        std::optional<int32_t> useSpecVerifyState = parsePluginScalarField<int32_t>("use_spec_verify_state", fc);

        if (phase == TensorRTPhase::kBUILD)
        {
            // Accepted but not yet supported (provided by the ONNX node):
            // chunk_size: Mamba2 prefill uses a chunked parallel scan when > 1.
            //   TODO: implement mamba_chunk_scan_combined kernel for chunk_size > 1.
            std::optional<int32_t> chunkSize = parsePluginScalarField<int32_t>("chunk_size", fc);
            ELLM_CHECK(!chunkSize.has_value() || chunkSize.value() <= 1,
                "update_ssm_state: chunk_size > 1 is not supported. "
                "Only single-step kernel with seq_len loop is implemented. "
                "Parallel chunked scan requires a mamba_chunk_scan_combined kernel.");
            // time_step_limit: (0.0, inf) is a no-op. Non-trivial clamping not yet in kernel.
            //   TODO: add dt clamping support to the selectiveStateUpdate kernel.
            for (int32_t i = 0; i < fc->nbFields; ++i)
            {
                PluginField const& f = fc->fields[i];
                if (std::string("time_step_limit") == f.name && f.data != nullptr && f.length >= 2)
                {
                    auto const* limits = static_cast<float const*>(f.data);
                    bool const isNoop = (limits[0] == 0.f && std::isinf(limits[1]) && limits[1] > 0.f);
                    ELLM_CHECK(isNoop,
                        "update_ssm_state: non-trivial time_step_limit is not supported. "
                        "Only the no-op default (0.0, inf) is currently handled. "
                        "Non-trivial dt clamping requires kernel changes.");
                }
            }
        }

        auto* plugin = new MambaPlugin(std::string(name), dim.value_or(0), dstate.value_or(0), nheads.value_or(0),
            ngroups.value_or(0), dtSoftplus.value_or(1), useSpecVerifyState.value_or(0));
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("Failed to create update_ssm_state plugin: %s", e.what());
    }
    return nullptr;
}

} // namespace plugins
} // namespace trt_edgellm
