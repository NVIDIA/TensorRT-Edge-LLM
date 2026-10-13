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

#include "nvfp4MoePlugin.h"

#include "common/checkMacros.h"
#include "common/cudaUtils.h"
#include "common/logger.h"
#include "common/stringUtils.h"
#include "common/tensor.h"
#include "kernels/moe/moePerExpertScaleKernels.h"
#include "kernels/moe/moeSigmoidGroupTopkKernels.h"
#include "kernels/moe/moeTopkSoftmaxKernels.h"
#include "plugins/utils/pluginUtils.h"

#include <NvInfer.h>

#ifdef CUTE_DSL_NVFP4_MOE_ENABLED
#include "kernels/moe/nvfp4_cutedsl/cuteDslNvfp4MoeSm110Runner.h"
#endif

#include <NvInferRuntime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_fp16.h>
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

// Input indices (11 total) and output index — see the plugin doc in the header for a full
// description of each tensor's shape and dtype.
namespace
{
constexpr int32_t kIN_ROUTER_LOGITS{0};
constexpr int32_t kIN_HIDDEN_STATES{1};
constexpr int32_t kIN_FC1_QWEIGHTS{2};
constexpr int32_t kIN_FC1_BLOCKS_SCALE{3};
constexpr int32_t kIN_FC1_ALPHA{4};
constexpr int32_t kIN_FC2_QWEIGHTS{5};
constexpr int32_t kIN_FC2_BLOCKS_SCALE{6};
constexpr int32_t kIN_FC2_ALPHA{7};
constexpr int32_t kIN_INPUT_GLOBAL_SCALE{8};
constexpr int32_t kIN_DOWN_INPUT_SCALE{9};
constexpr int32_t kIN_E_SCORE_CORRECTION_BIAS{10};
constexpr int32_t kOUT_OUTPUT{11};
constexpr int32_t kNbPluginInputs{11};

constexpr char const* kPLUGIN_VERSION{"1"};
constexpr char const* kPLUGIN_NAME{"Nvfp4MoePlugin"};

//! Matches the activation_type attribute encoding (see the header).
constexpr int32_t kACT_IDENTITY{0};
constexpr int32_t kACT_SILU{1};
constexpr int32_t kACT_SWIGLU{2};
constexpr int32_t kACT_GELU{3};
constexpr int32_t kACT_RELU2{4};
constexpr int32_t kACT_GEGLU{5};

constexpr int32_t kBACKEND_AUTO{0};
constexpr int32_t kBACKEND_DECODE{1};
constexpr int32_t kBACKEND_PREFILL{2};

constexpr int32_t kIODT_BF16{0};
constexpr int32_t kIODT_FP16{1};

constexpr int32_t kROUTING_MODE_SOFTMAX_TOPK{0};
constexpr int32_t kROUTING_MODE_SIGMOID_GROUP_TOPK{1};
constexpr int32_t kROUTING_MODE_SOFTMAX_TOPK_POST_SCALE{2};

//! NVFP4 scale-factor group size (kK dim grouping) — block scales have K/sf_vec_size entries.
constexpr int32_t kNvfp4SfVecSize{16};

//! Keep shape inference compilable when the AOT runner is not linked.
#ifdef CUTE_DSL_NVFP4_MOE_ENABLED
constexpr int32_t kCuteDslLevelTileN{CuteDslNvfp4MoeSm110Runner::kLevelTileN};
constexpr int32_t kCuteDslSm110TopK{CuteDslNvfp4MoeSm110Runner::kMaxTopK};
#else
constexpr int32_t kCuteDslLevelTileN{128};
constexpr int32_t kCuteDslSm110TopK{0};
#endif

inline int32_t ceilDivInt(int32_t a, int32_t b)
{
    return (a + b - 1) / b;
}

inline bool isGatedActivation(int32_t activationType)
{
    return activationType == kACT_SWIGLU || activationType == kACT_GEGLU;
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

//! Number of output rows in FC1 given the gated/non-gated activation.
inline int32_t fc1OutDim(int32_t moeInterSize, int32_t activationType)
{
    return isGatedActivation(activationType) ? 2 * moeInterSize : moeInterSize;
}

#ifdef CUTE_DSL_NVFP4_MOE_ENABLED
//! Compute ``maxNumTokens * topK`` saturated to a positive int32 (1 .. INT32_MAX).
//! Used both for inferring the auto-resolved cap and for validating user-supplied
//! \c max_routed_rows against the optimization profile.
int32_t computeProfileMaxRoutedRows(int64_t maxNumTokens, int32_t topK)
{
    if (maxNumTokens <= 0 || topK <= 0)
    {
        return 1;
    }
    if (maxNumTokens > std::numeric_limits<int32_t>::max() / static_cast<int64_t>(topK))
    {
        return std::numeric_limits<int32_t>::max();
    }
    return static_cast<int32_t>(maxNumTokens * static_cast<int64_t>(topK));
}

//! PDL is enabled by default for the NVFP4 MoE CuTe DSL path. Set
//! EDGELLM_ENABLE_PDL=0 before the first enqueue to retain an explicit
//! production A/B and recovery switch. The generated AOT wrapper receives the
//! resolved value per launch; the runner applies its own toolchain/SM gate.
bool requestNvfp4MoePdl()
{
    static bool const enabled = []() {
        char const* const value = std::getenv("EDGELLM_ENABLE_PDL");
        return value == nullptr || std::strcmp(value, "0") != 0;
    }();
    return enabled;
}
#endif // CUTE_DSL_NVFP4_MOE_ENABLED
} // namespace

PluginFieldCollection Nvfp4MoePluginCreator::mFieldCollection{};
std::vector<PluginField> Nvfp4MoePluginCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(Nvfp4MoePluginCreator);

// Constructors / boilerplate

Nvfp4MoePlugin::Nvfp4MoePlugin(std::string const& name, int32_t numExperts, int32_t topK, int32_t hiddenSize,
    int32_t moeInterSize, int32_t activationType, int32_t nGroup, int32_t topkGroup, int32_t normTopkProb,
    float routedScalingFactor, int32_t routingMode, int32_t backend, int32_t maxRoutedRows, int32_t ioDtype)
    : mLayerName(name)
    , mNumExperts(numExperts)
    , mTopK(topK)
    , mHiddenSize(hiddenSize)
    , mMoeInterSize(moeInterSize)
    , mActivationType(activationType)
    , mRoutingMode(routingMode)
    , mNGroup(nGroup)
    , mTopkGroup(topkGroup)
    , mNormTopkProb(normTopkProb)
    , mRoutedScalingFactor(routedScalingFactor)
    , mBackend(backend)
    , mMaxRoutedRows(maxRoutedRows)
    , mIoDtype(ioDtype)
{
}

Nvfp4MoePlugin::Nvfp4MoePlugin(std::string const& name, PluginFieldCollection const* fc)
    : mLayerName(name)
{
    for (int32_t i = 0; i < fc->nbFields; ++i)
    {
        std::string fieldName(fc->fields[i].name);
        if (fieldName == "num_experts")
        {
            mNumExperts = *static_cast<int32_t const*>(fc->fields[i].data);
        }
        else if (fieldName == "top_k")
        {
            mTopK = *static_cast<int32_t const*>(fc->fields[i].data);
        }
        else if (fieldName == "hidden_size")
        {
            mHiddenSize = *static_cast<int32_t const*>(fc->fields[i].data);
        }
        else if (fieldName == "moe_inter_size")
        {
            mMoeInterSize = *static_cast<int32_t const*>(fc->fields[i].data);
        }
        else if (fieldName == "activation_type")
        {
            mActivationType = *static_cast<int32_t const*>(fc->fields[i].data);
        }
        else if (fieldName == "n_group")
        {
            mNGroup = *static_cast<int32_t const*>(fc->fields[i].data);
        }
        else if (fieldName == "topk_group")
        {
            mTopkGroup = *static_cast<int32_t const*>(fc->fields[i].data);
        }
        else if (fieldName == "norm_topk_prob")
        {
            mNormTopkProb = *static_cast<int32_t const*>(fc->fields[i].data);
        }
        else if (fieldName == "routed_scaling_factor")
        {
            mRoutedScalingFactor = *static_cast<float const*>(fc->fields[i].data);
        }
        else if (fieldName == "routing_mode")
        {
            mRoutingMode = *static_cast<int32_t const*>(fc->fields[i].data);
        }
        else if (fieldName == "backend")
        {
            mBackend = *static_cast<int32_t const*>(fc->fields[i].data);
        }
        else if (fieldName == "max_routed_rows")
        {
            mMaxRoutedRows = *static_cast<int32_t const*>(fc->fields[i].data);
        }
        else if (fieldName == "io_dtype")
        {
            mIoDtype = *static_cast<int32_t const*>(fc->fields[i].data);
        }
    }
    if (mNumExperts <= 0 || mTopK <= 0 || mHiddenSize <= 0 || mMoeInterSize <= 0)
    {
        throw std::invalid_argument(
            "Nvfp4MoePlugin: num_experts, top_k, hidden_size, "
            "and moe_inter_size must all be > 0");
    }
    // v1 gating — reject values outside the supported subset early so user sees a clear error
    // instead of a silent wrong-answer or a deep-in-the-stack assertion.
    if (mIoDtype != kIODT_FP16)
    {
        throw std::invalid_argument("Nvfp4MoePlugin: v1 only supports io_dtype=1 (FP16)");
    }
    switch (mActivationType)
    {
    case kACT_IDENTITY:
    case kACT_SILU:
    case kACT_SWIGLU:
    case kACT_GELU:
    case kACT_RELU2:
    case kACT_GEGLU: break;
    default:
        throw std::invalid_argument(
            "Nvfp4MoePlugin: activation_type must be 0 (identity), 1 (silu), "
            "2 (swiglu), 3 (gelu), 4 (relu2), or 5 (geglu)");
    }
    if (mBackend != kBACKEND_AUTO && mBackend != kBACKEND_DECODE && mBackend != kBACKEND_PREFILL)
    {
        throw std::invalid_argument("Nvfp4MoePlugin: backend must be 0 (auto), 1 (decode), or 2 (prefill)");
    }
    if (mRoutingMode != kROUTING_MODE_SOFTMAX_TOPK && mRoutingMode != kROUTING_MODE_SIGMOID_GROUP_TOPK
        && mRoutingMode != kROUTING_MODE_SOFTMAX_TOPK_POST_SCALE)
    {
        throw std::invalid_argument(
            "Nvfp4MoePlugin: routing_mode must be 0 (softmax top-k), 1 (sigmoid group top-k), "
            "or 2 (softmax top-k post-scale)");
    }
    if (mRoutingMode == kROUTING_MODE_SIGMOID_GROUP_TOPK)
    {
        if (mNGroup <= 0 || mNumExperts % mNGroup != 0)
        {
            throw std::invalid_argument("Nvfp4MoePlugin: n_group must be positive and evenly divide num_experts");
        }
        if (mTopkGroup <= 0 || mTopkGroup > mNGroup)
        {
            throw std::invalid_argument("Nvfp4MoePlugin: topk_group must be in [1, n_group]");
        }
    }
}

Nvfp4MoePlugin::~Nvfp4MoePlugin() noexcept = default;

IPluginCapability* Nvfp4MoePlugin::getCapabilityInterface(PluginCapabilityType type) noexcept
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

IPluginV3* Nvfp4MoePlugin::clone() noexcept
{
    try
    {
        auto* p = new Nvfp4MoePlugin(mLayerName, mNumExperts, mTopK, mHiddenSize, mMoeInterSize, mActivationType,
            mNGroup, mTopkGroup, mNormTopkProb, mRoutedScalingFactor, mRoutingMode, mBackend, mMaxRoutedRows, mIoDtype);
        p->setPluginNamespace(mNamespace.c_str());
        return p;
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("Failed to clone Nvfp4MoePlugin: %s", e.what());
        return nullptr;
    }
}

char const* Nvfp4MoePlugin::getPluginName() const noexcept
{
    return kPLUGIN_NAME;
}

char const* Nvfp4MoePlugin::getPluginVersion() const noexcept
{
    return kPLUGIN_VERSION;
}

char const* Nvfp4MoePlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

void Nvfp4MoePlugin::setPluginNamespace(char const* pluginNamespace) noexcept
{
    mNamespace = pluginNamespace ? pluginNamespace : "";
}

int32_t Nvfp4MoePlugin::getNbOutputs() const noexcept
{
    return 1;
}

int32_t Nvfp4MoePlugin::getOutputDataTypes(
    DataType* outputTypes, int32_t nbOutputs, DataType const* inputTypes, int32_t nbInputs) const noexcept
{
    // Always-on contract check (assert() would be a no-op in release builds and
    // could silently write past `outputTypes`).
    if (nbOutputs != 1)
    {
        LOG_ERROR("Nvfp4MoePlugin: getOutputDataTypes expected 1 output, got %d", nbOutputs);
        return -1;
    }
    (void) nbInputs;
    (void) inputTypes;
    outputTypes[0] = DataType::kHALF; // v1 always FP16 out.
    return 0;
}

int32_t Nvfp4MoePlugin::getOutputShapes(DimsExprs const* inputs, int32_t nbInputs, DimsExprs const* shapeInputs,
    int32_t nbShapeInputs, DimsExprs* outputs, int32_t nbOutputs, IExprBuilder& exprBuilder) noexcept
{
    // Always-on contract check (assert() would be a no-op in release builds and
    // could silently dereference past the input/output arrays).
    if (nbInputs != kNbPluginInputs || nbOutputs != 1)
    {
        LOG_ERROR("Nvfp4MoePlugin: getOutputShapes expected %d inputs and 1 output, got %d inputs and %d outputs",
            kNbPluginInputs, nbInputs, nbOutputs);
        return -1;
    }
    (void) shapeInputs;
    (void) nbShapeInputs;
    if (inputs[kIN_HIDDEN_STATES].nbDims != 2)
    {
        LOG_ERROR("Nvfp4MoePlugin: hidden_states must have rank 2");
        return -1;
    }
    outputs[0].nbDims = 2;
    outputs[0].d[0] = inputs[kIN_HIDDEN_STATES].d[0];
    outputs[0].d[1] = exprBuilder.constant(static_cast<int64_t>(mHiddenSize));
    return 0;
}

// Shape / format validation

bool Nvfp4MoePlugin::supportsFormatCombination(
    int32_t pos, DynamicPluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
    // Always-on contract check (assert() would be a no-op in release builds).
    if (nbInputs != kNbPluginInputs || nbOutputs != 1)
    {
        LOG_ERROR(
            "Nvfp4MoePlugin: supportsFormatCombination expected %d inputs and 1 output, got %d inputs and %d "
            "outputs",
            kNbPluginInputs, nbInputs, nbOutputs);
        return false;
    }

    auto const& td = inOut[pos].desc;

    // All inputs/outputs must be LINEAR — no reformat.
    if (td.format != TensorFormat::kLINEAR)
    {
        return false;
    }

    int32_t const n1 = fc1OutDim(mMoeInterSize, mActivationType);

    // supportsFormatCombination is called many times by TensorRT during engine
    // build with different (pos, type, dims) tuples; only one combination is
    // accepted. When something looks off in the engine (e.g. "no supported
    // format combination" errors), the fastest way to diagnose is to enable
    // LOG_DEBUG and inspect why each position was rejected. SFC_REJ wraps the
    // common "log the reason + return false" pattern in one macro so the per-
    // position checks below stay readable.
#define SFC_REJ(reason)                                                                                                \
    do                                                                                                                 \
    {                                                                                                                  \
        LOG_DEBUG("Nvfp4MoePlugin supportsFormatCombination rejected pos %d: %s", pos, reason);                        \
        return false;                                                                                                  \
    } while (0)

    // Expected shape / dtype for each position (shapes at pos == leading dim may be -1 for
    // dynamic at build time, so we only enforce static dims and dtype here).
    switch (pos)
    {
    case kIN_ROUTER_LOGITS:
    {
        if (td.type != DataType::kFLOAT)
            SFC_REJ("ROUTER_LOGITS: type != kFLOAT");
        if (td.dims.nbDims != 2)
            SFC_REJ("ROUTER_LOGITS: nbDims != 2");
        if (td.dims.d[1] != mNumExperts)
            SFC_REJ("ROUTER_LOGITS: d[1] != mNumExperts");
        return true;
    }
    case kIN_HIDDEN_STATES:
    {
        if (td.type != DataType::kHALF)
            SFC_REJ("HIDDEN_STATES: type != kHALF");
        if (td.dims.nbDims != 2)
            SFC_REJ("HIDDEN_STATES: nbDims != 2");
        if (td.dims.d[1] != mHiddenSize)
            SFC_REJ("HIDDEN_STATES: d[1] != mHiddenSize");
        return true;
    }
    case kIN_FC1_QWEIGHTS:
    {
        if (td.type != DataType::kINT8)
            SFC_REJ("FC1_QWEIGHTS: type != kINT8");
        if (td.dims.nbDims != 3)
            SFC_REJ("FC1_QWEIGHTS: nbDims != 3");
        if (td.dims.d[0] != mNumExperts)
            SFC_REJ("FC1_QWEIGHTS: d[0] != mNumExperts");
        if (td.dims.d[1] != n1)
            SFC_REJ("FC1_QWEIGHTS: d[1] != n1");
        if (td.dims.d[2] != mHiddenSize / 2)
            SFC_REJ("FC1_QWEIGHTS: d[2] != mHiddenSize/2");
        return true;
    }
    case kIN_FC1_BLOCKS_SCALE:
    {
        if (td.type != DataType::kINT8)
            SFC_REJ("FC1_BLOCKS_SCALE: type != kINT8");
        if (td.dims.nbDims != 6)
            SFC_REJ("FC1_BLOCKS_SCALE: nbDims != 6");
        int32_t const mTiles = ceilDivInt(n1, 128);
        int32_t const kTiles = ceilDivInt(mHiddenSize / kNvfp4SfVecSize, 4);
        if (td.dims.d[0] != mNumExperts)
            SFC_REJ("FC1_BLOCKS_SCALE: d[0] != mNumExperts");
        if (td.dims.d[1] != mTiles)
            SFC_REJ("FC1_BLOCKS_SCALE: d[1] != mTiles");
        if (td.dims.d[2] != kTiles)
            SFC_REJ("FC1_BLOCKS_SCALE: d[2] != kTiles");
        if (td.dims.d[3] != 32 || td.dims.d[4] != 4 || td.dims.d[5] != 4)
            SFC_REJ("FC1_BLOCKS_SCALE: inner dims != [32,4,4]");
        return true;
    }
    case kIN_FC1_ALPHA:
    {
        if (td.type != DataType::kFLOAT)
            SFC_REJ("FC1_ALPHA: type != kFLOAT");
        if (td.dims.nbDims != 1)
            SFC_REJ("FC1_ALPHA: nbDims != 1");
        if (td.dims.d[0] != mNumExperts)
            SFC_REJ("FC1_ALPHA: d[0] != mNumExperts");
        return true;
    }
    case kIN_FC2_QWEIGHTS:
    {
        if (td.type != DataType::kINT8)
            SFC_REJ("FC2_QWEIGHTS: type != kINT8");
        if (td.dims.nbDims != 3)
            SFC_REJ("FC2_QWEIGHTS: nbDims != 3");
        if (td.dims.d[0] != mNumExperts)
            SFC_REJ("FC2_QWEIGHTS: d[0] != mNumExperts");
        if (td.dims.d[1] != mHiddenSize)
            SFC_REJ("FC2_QWEIGHTS: d[1] != mHiddenSize");
        if (td.dims.d[2] != mMoeInterSize / 2)
            SFC_REJ("FC2_QWEIGHTS: d[2] != mMoeInterSize/2");
        return true;
    }
    case kIN_FC2_BLOCKS_SCALE:
    {
        if (td.type != DataType::kINT8)
            SFC_REJ("FC2_BLOCKS_SCALE: type != kINT8");
        if (td.dims.nbDims != 6)
            SFC_REJ("FC2_BLOCKS_SCALE: nbDims != 6");
        int32_t const mTiles = ceilDivInt(mHiddenSize, 128);
        int32_t const kTiles = ceilDivInt(mMoeInterSize / kNvfp4SfVecSize, 4);
        if (td.dims.d[0] != mNumExperts)
            SFC_REJ("FC2_BLOCKS_SCALE: d[0] != mNumExperts");
        if (td.dims.d[1] != mTiles)
            SFC_REJ("FC2_BLOCKS_SCALE: d[1] != mTiles");
        if (td.dims.d[2] != kTiles)
            SFC_REJ("FC2_BLOCKS_SCALE: d[2] != kTiles");
        if (td.dims.d[3] != 32 || td.dims.d[4] != 4 || td.dims.d[5] != 4)
            SFC_REJ("FC2_BLOCKS_SCALE: inner dims != [32,4,4]");
        return true;
    }
    case kIN_FC2_ALPHA: [[fallthrough]];
    case kIN_INPUT_GLOBAL_SCALE: [[fallthrough]];
    case kIN_DOWN_INPUT_SCALE: [[fallthrough]];
    case kIN_E_SCORE_CORRECTION_BIAS:
    {
        if (td.type != DataType::kFLOAT)
            SFC_REJ("FC2_ALPHA/SCALE group: type != kFLOAT");
        if (td.dims.nbDims != 1)
            SFC_REJ("FC2_ALPHA/SCALE group: nbDims != 1");
        if (td.dims.d[0] != mNumExperts)
            SFC_REJ("FC2_ALPHA/SCALE group: d[0] != mNumExperts");
        return true;
    }
    case kOUT_OUTPUT:
    {
        if (td.type != DataType::kHALF)
            SFC_REJ("OUT_OUTPUT: type != kHALF");
        auto const& hiddenDims = inOut[kIN_HIDDEN_STATES].desc.dims;
        if (td.dims.nbDims != 2 || !shapesMatch(td.dims, hiddenDims))
            SFC_REJ("OUT_OUTPUT: shape != HIDDEN_STATES shape");
        if (td.dims.d[td.dims.nbDims - 1] != mHiddenSize)
            SFC_REJ("OUT_OUTPUT: last dimension != mHiddenSize");
        return true;
    }
    default: SFC_REJ("default: unknown pos");
    }
#undef SFC_REJ
}

int32_t Nvfp4MoePlugin::configurePlugin(
    DynamicPluginTensorDesc const* in, int32_t nbInputs, DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept
{
    if (in == nullptr || out == nullptr || nbInputs != kNbPluginInputs || nbOutputs != 1)
    {
        LOG_ERROR("Nvfp4MoePlugin: configurePlugin expected %d inputs and 1 output", kNbPluginInputs);
        return -1;
    }

    // Invariant: the constructor-supplied mHiddenSize (from the ONNX hidden_size_i
    // attribute / PluginField) is the single source of truth. Verify the network's
    // hidden_states tensor agrees so a wrong attribute fails loudly here instead
    // of silently mismatching the AOT-baked kernel dim downstream.
    auto validateProfileEndpoint
        = [this](Dims const& hidden, Dims const& router, char const* endpoint, int64_t& numTokens) {
              if (!getTokenCount(hidden, numTokens) || hidden.d[hidden.nbDims - 1] != mHiddenSize || router.nbDims != 2
                  || router.d[0] != numTokens || router.d[1] != mNumExperts)
              {
                  LOG_ERROR("Nvfp4MoePlugin: optimization profile %s dimensions are invalid", endpoint);
                  return false;
              }
              return true;
          };
    int64_t minTokens{0};
    int64_t optTokens{0};
    int64_t maxTokens{0};
    if (!validateProfileEndpoint(in[kIN_HIDDEN_STATES].min, in[kIN_ROUTER_LOGITS].min, "minimum", minTokens)
        || !validateProfileEndpoint(in[kIN_HIDDEN_STATES].opt, in[kIN_ROUTER_LOGITS].opt, "optimum", optTokens)
        || !validateProfileEndpoint(in[kIN_HIDDEN_STATES].max, in[kIN_ROUTER_LOGITS].max, "maximum", maxTokens))
    {
        return -1;
    }
    if (!shapesMatch(out[0].min, in[kIN_HIDDEN_STATES].min) || !shapesMatch(out[0].opt, in[kIN_HIDDEN_STATES].opt)
        || !shapesMatch(out[0].max, in[kIN_HIDDEN_STATES].max))
    {
        LOG_ERROR("Nvfp4MoePlugin: output profile range must match hidden_states");
        return -1;
    }
    if (static_cast<int32_t>(in[kIN_E_SCORE_CORRECTION_BIAS].max.d[0]) != mNumExperts)
    {
        LOG_ERROR("Nvfp4MoePlugin: e_score_correction_bias d[0] (%d) must equal num_experts (%d)",
            static_cast<int32_t>(in[kIN_E_SCORE_CORRECTION_BIAS].max.d[0]), mNumExperts);
        return -1;
    }
    if (mRoutingMode == kROUTING_MODE_SIGMOID_GROUP_TOPK)
    {
        if (mNGroup <= 0 || mNumExperts % mNGroup != 0)
        {
            LOG_ERROR("Nvfp4MoePlugin: n_group (%d) must be positive and evenly divide num_experts (%d)", mNGroup,
                mNumExperts);
            return -1;
        }
        if (mTopkGroup <= 0 || mTopkGroup > mNGroup)
        {
            LOG_ERROR("Nvfp4MoePlugin: topk_group (%d) must be in [1, n_group=%d]", mTopkGroup, mNGroup);
            return -1;
        }
    }

#ifdef CUTE_DSL_NVFP4_MOE_ENABLED
    // Defer the (sm version, dtype, activation, backend, bounded hidden size,
    // I / E / top_k divisibility) gate to the split FC1/FC2 runner so the
    // plugin and backend can never disagree about what the linked AOT pack
    // supports.
    int32_t const smVersion = trt_edgellm::getSMVersion();
    bool const canImplement = CuteDslNvfp4MoeSm110Runner::canImplement(
        mHiddenSize, mMoeInterSize, mNumExperts, mTopK, smVersion, mActivationType, mIoDtype, mBackend);
    if (!canImplement)
    {
        LOG_ERROR(
            "Nvfp4MoePlugin: shape tuple (H=%d, I=%d, E=%d, top_k=%d, sm=%d, "
            "act=%d, io=%d, backend=%d) is not supported by the split "
            "FC1/FC2 CuteDSL runner. Requires -DENABLE_CUTE_DSL=nvfp4_moe, "
            "sm in {100, 101, 110}, io_dtype=FP16, activation in {swiglu, relu2, geglu}, "
            "H %% %d == 0, I %% 64 == 0, FC1_N %% %d == 0, E in {128, 256}, 0 < top_k <= %d.",
            mHiddenSize, mMoeInterSize, mNumExperts, mTopK, smVersion, mActivationType, mIoDtype, mBackend, 128,
            kCuteDslLevelTileN, kCuteDslSm110TopK);
        return -1;
    }
#else
    LOG_ERROR(
        "Nvfp4MoePlugin: no CuTeDSL SM100/101/110 MoE backend is linked -- rebuild "
        "with -DENABLE_CUTE_DSL=nvfp4_moe after generating the matching AOT "
        "artifact via kernelSrcs/build_cutedsl.py. For SM12x targets use the "
        "NvFP4MoEPluginGeforce plugin instead.");
    return -1;
#endif

#ifdef CUTE_DSL_NVFP4_MOE_ENABLED
    // Resolve and validate \c max_routed_rows against the optimization profile's max
    // shapes. The profile-derived bound is the largest \c num_tokens * \c top_k
    // any legal runtime shape will request; the runner workspace MUST be sized for
    // it. Reject explicit user values smaller than that bound, and infer/serialize
    // the bound when the user passed 0 so the runtime instance carries a concrete
    // cap (see \c onShapeChange / \c enqueue runtime checks below).
    int64_t const profileMaxTokens = maxTokens;
    if (profileMaxTokens > 0)
    {
        if (mTopK <= 0 || profileMaxTokens > std::numeric_limits<int32_t>::max() / mTopK)
        {
            LOG_ERROR("Nvfp4MoePlugin: optimization profile routed rows exceed INT32 capacity");
            return -1;
        }
        int32_t const profileMaxRoutedRows = computeProfileMaxRoutedRows(profileMaxTokens, mTopK);
        if (mMaxRoutedRows == 0)
        {
            // Auto-resolve from the optimization profile and fall through to be
            // serialized via \c getFieldsToSerialize.
            mMaxRoutedRows = profileMaxRoutedRows;
        }
        else if (mMaxRoutedRows < profileMaxRoutedRows)
        {
            LOG_ERROR(
                "Nvfp4MoePlugin: max_routed_rows (%d) is smaller than the "
                "optimization profile maximum num_tokens*top_k (%lld*%d=%d). The "
                "runner workspace would be undersized when the runtime shape "
                "saturates the profile. Either pass max_routed_rows >= %d or pass "
                "0 to auto-resolve from the profile.",
                mMaxRoutedRows, static_cast<long long>(profileMaxTokens), mTopK, profileMaxRoutedRows,
                profileMaxRoutedRows);
            return -1;
        }
    }
    else if (mMaxRoutedRows <= 0)
    {
        // No profile information and no explicit cap — refuse to silently allocate
        // a 1-row workspace.
        LOG_ERROR(
            "Nvfp4MoePlugin: cannot resolve max_routed_rows. The optimization "
            "profile did not provide hidden_states / router_logits max shapes and the "
            "user did not pass an explicit max_routed_rows.");
        return -1;
    }
#endif

    return 0;
}

// Workspace sizing / enqueue

size_t Nvfp4MoePlugin::getWorkspaceSize(DynamicPluginTensorDesc const* inputs, int32_t nbInputs,
    DynamicPluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept
{
    // Always-on contract check (assert() would be a no-op in release builds).
    // Returning 0 forces the caller to skip the runner workspace allocation,
    // which makes the subsequent enqueue fail loudly instead of dereferencing
    // out-of-bounds workspace pointers.
    if (inputs == nullptr || outputs == nullptr || nbInputs != kNbPluginInputs || nbOutputs != 1)
    {
        LOG_ERROR("Nvfp4MoePlugin: getWorkspaceSize expected %d inputs and 1 output", kNbPluginInputs);
        return 0;
    }

#ifdef CUTE_DSL_NVFP4_MOE_ENABLED
    try
    {
        int64_t maxHiddenTokens{0};
        if (!getTokenCount(inputs[kIN_HIDDEN_STATES].max, maxHiddenTokens) || mTopK <= 0
            || maxHiddenTokens > std::numeric_limits<int32_t>::max() / mTopK)
        {
            LOG_ERROR("Nvfp4MoePlugin: invalid hidden_states profile maximum");
            return 0;
        }
        auto const& maxRouterDims = inputs[kIN_ROUTER_LOGITS].max;
        if (maxRouterDims.nbDims != 2 || maxRouterDims.d[0] != maxHiddenTokens || maxRouterDims.d[1] != mNumExperts
            || !shapesMatch(outputs[0].max, inputs[kIN_HIDDEN_STATES].max))
        {
            LOG_ERROR("Nvfp4MoePlugin: profile maximum router/output shapes do not match hidden_states");
            return 0;
        }
        int64_t const maxRouterTokens = static_cast<int64_t>(maxRouterDims.d[0]);
        int64_t const maxTokens64 = std::max(maxHiddenTokens, maxRouterTokens);
        int32_t const maxTokens = static_cast<int32_t>(
            std::min<int64_t>(maxTokens64, static_cast<int64_t>(std::numeric_limits<int32_t>::max())));

        // mMaxRoutedRows is the resolved profile-or-user cap from configurePlugin()
        // (always > 0 here). It is the upper bound the runner workspace is sized for;
        // onShapeChange / enqueue refuse runtime shapes that would exceed it.
        int32_t const maxRoutedRows
            = mMaxRoutedRows > 0 ? mMaxRoutedRows : computeProfileMaxRoutedRows(maxTokens64, mTopK);

        size_t total = 0;
        // topk softmax scratch (zero when numExperts is a power of 2 <= 256).
        size_t const softmaxWs = trt_edgellm::kernel::getMoeTopkSoftmaxWorkspaceSize(maxTokens, mNumExperts);
        total = accumulateWorkspaceSize(
            total, rt::Coords{static_cast<int64_t>(std::max<size_t>(softmaxWs, 1))}, DataType::kINT8);
        // topk weights / indices produced by moeTopkSoftmax and consumed by the runner.
        total = accumulateWorkspaceSize(total, rt::Coords{maxTokens, mTopK}, DataType::kFLOAT);
        total = accumulateWorkspaceSize(total, rt::Coords{maxTokens, mTopK}, DataType::kINT32);
        size_t const runnerWs = CuteDslNvfp4MoeSm110Runner::getWorkspaceSize(
            maxTokens, maxRoutedRows, mNumExperts, mTopK, mHiddenSize, mMoeInterSize);
        if (runnerWs == 0)
        {
            LOG_ERROR("Nvfp4MoePlugin: SM100/101/110 CuTeDSL backend returned zero workspace");
            return 0;
        }
        total = accumulateWorkspaceSize(
            total, rt::Coords{static_cast<int64_t>(std::max<size_t>(runnerWs, 1))}, DataType::kINT8);
        return total;
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("Failed to compute Nvfp4MoePlugin workspace size: %s", e.what());
        return 0;
    }
#else
    return 0;
#endif
}

int32_t Nvfp4MoePlugin::enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc,
    void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
#ifndef CUTE_DSL_NVFP4_MOE_ENABLED
    (void) inputDesc;
    (void) outputDesc;
    (void) inputs;
    (void) outputs;
    (void) workspace;
    (void) stream;
    LOG_ERROR(
        "Nvfp4MoePlugin: no CuTeDSL SM100/101/110 MoE backend is linked; rebuild "
        "with -DENABLE_CUTE_DSL=nvfp4_moe");
    return -1;
#else
    try
    {
        if (inputDesc == nullptr || outputDesc == nullptr || inputs == nullptr || outputs == nullptr
            || outputs[0] == nullptr)
        {
            LOG_ERROR("Nvfp4MoePlugin: enqueue received null descriptors or buffers");
            return -1;
        }
        PluginTensorDesc const& hiddenDesc = inputDesc[kIN_HIDDEN_STATES];
        int64_t numTokens64{0};
        if (!getTokenCount(hiddenDesc.dims, numTokens64)
            || hiddenDesc.dims.d[hiddenDesc.dims.nbDims - 1] != mHiddenSize)
        {
            LOG_ERROR("Nvfp4MoePlugin: hidden_states must be [T,H] with H=%d", mHiddenSize);
            return -1;
        }
        if (mTopK <= 0 || numTokens64 > std::numeric_limits<int32_t>::max() / mTopK)
        {
            LOG_ERROR("Nvfp4MoePlugin: runtime routed rows exceed INT32 capacity");
            return -1;
        }
        int32_t const numTokens = static_cast<int32_t>(numTokens64);
        auto const& routerDims = inputDesc[kIN_ROUTER_LOGITS].dims;
        if (routerDims.nbDims != 2 || routerDims.d[0] != numTokens || routerDims.d[1] != mNumExperts)
        {
            LOG_ERROR("Nvfp4MoePlugin: router_logits must be [num_tokens=%d, num_experts=%d]", numTokens, mNumExperts);
            return -1;
        }
        if (!shapesMatch(outputDesc[0].dims, hiddenDesc.dims))
        {
            LOG_ERROR("Nvfp4MoePlugin: output shape must match hidden_states");
            return -1;
        }

        // Final-line cap guard: workspace and runner buffers were sized at build
        // time for at most \c mMaxRoutedRows. Refuse a runtime shape that would
        // overflow them. \c onShapeChange has the same check earlier in the
        // lifecycle; this is a defense-in-depth backstop.
        int64_t const runtimeRoutedRows64 = numTokens64 * static_cast<int64_t>(mTopK);
        if (mMaxRoutedRows > 0 && runtimeRoutedRows64 > static_cast<int64_t>(mMaxRoutedRows))
        {
            LOG_ERROR(
                "Nvfp4MoePlugin: runtime num_tokens*top_k (%lld) exceeds the "
                "resolved max_routed_rows cap (%d). Rebuild the engine with a profile "
                "whose max shapes cover this case, or pass a larger explicit "
                "max_routed_rows attribute.",
                static_cast<long long>(runtimeRoutedRows64), mMaxRoutedRows);
            return -1;
        }
        int32_t const maxRoutedRows
            = mMaxRoutedRows > 0 ? mMaxRoutedRows : computeProfileMaxRoutedRows(numTokens64, mTopK);
        CuteDslNvfp4MoeSm110Params moduleParams{};
        moduleParams.activationType = mActivationType;
        if (!CuteDslNvfp4MoeSm110Runner::ensureKernelModules(moduleParams, stream))
        {
            LOG_ERROR("Nvfp4MoePlugin: failed to load the selected SM100/101/110 CuTe DSL AOT modules");
            return -1;
        }

        // ==== Workspace carving (order must match getWorkspaceSize) ====
        size_t const softmaxWs = trt_edgellm::kernel::getMoeTopkSoftmaxWorkspaceSize(numTokens, mNumExperts);
        std::byte* ws = static_cast<std::byte*>(workspace);
        void* const softmaxScratch = assignTensorFromWorkspace(
            ws, rt::Coords{static_cast<int64_t>(std::max<size_t>(softmaxWs, 1))}, DataType::kINT8)
                                         .rawPointer();
        float* const topkWeightsPtr
            = static_cast<float*>(assignTensorFromWorkspace(ws, {numTokens, mTopK}, DataType::kFLOAT).rawPointer());
        int32_t* const topkIdsPtr
            = static_cast<int32_t*>(assignTensorFromWorkspace(ws, {numTokens, mTopK}, DataType::kINT32).rawPointer());
        size_t const runnerWs = CuteDslNvfp4MoeSm110Runner::getWorkspaceSize(
            numTokens, maxRoutedRows, mNumExperts, mTopK, mHiddenSize, mMoeInterSize);
        if (runnerWs == 0)
        {
            LOG_ERROR("Nvfp4MoePlugin: SM100/101/110 CuTeDSL backend returned zero workspace at enqueue");
            return -1;
        }
        void* const runnerScratch = assignTensorFromWorkspace(
            ws, rt::Coords{static_cast<int64_t>(std::max<size_t>(runnerWs, 1))}, DataType::kINT8)
                                        .rawPointer();

        // ==== Step 1: router softmax + top-k + renormalize ====
        rt::Tensor routerLogitsT(const_cast<void*>(inputs[kIN_ROUTER_LOGITS]),
            rt::Coords{inputDesc[kIN_ROUTER_LOGITS].dims}, rt::DeviceType::kGPU, DataType::kFLOAT);
        rt::Tensor topkWeightsT(topkWeightsPtr, {numTokens, mTopK}, rt::DeviceType::kGPU, DataType::kFLOAT);
        rt::Tensor topkIdsT(topkIdsPtr, {numTokens, mTopK}, rt::DeviceType::kGPU, DataType::kINT32);
        if (mRoutingMode == kROUTING_MODE_SIGMOID_GROUP_TOPK)
        {
            rt::Tensor correctionBiasT(const_cast<void*>(inputs[kIN_E_SCORE_CORRECTION_BIAS]), {mNumExperts},
                rt::DeviceType::kGPU, DataType::kFLOAT);
            rt::OptionalInputTensor correctionBiasOpt = correctionBiasT;
            trt_edgellm::kernel::moeSigmoidGroupTopk(routerLogitsT, topkWeightsT, topkIdsT, mTopK, mNGroup, mTopkGroup,
                mNormTopkProb != 0, mRoutedScalingFactor, stream, correctionBiasOpt);
        }
        else if (mRoutingMode == kROUTING_MODE_SOFTMAX_TOPK)
        {
            // Pre-softmax additive bias (DeepSeek, Qwen3).
            rt::Tensor correctionBiasT(const_cast<void*>(inputs[kIN_E_SCORE_CORRECTION_BIAS]), {mNumExperts},
                rt::DeviceType::kGPU, DataType::kFLOAT);
            rt::OptionalInputTensor biasForSoftmax = correctionBiasT;
            trt_edgellm::kernel::moeTopkSoftmax(routerLogitsT, topkWeightsT, topkIdsT, mTopK,
                softmaxWs > 0 ? softmaxScratch : nullptr, softmaxWs, stream, /*renormalize=*/true, 0.0F,
                biasForSoftmax);
        }
        else
        {
            // kSOFTMAX_TOPK_POST_SCALE: softmax (no bias) + post-renorm multiplicative scale (Gemma4).
            rt::OptionalInputTensor biasForSoftmax;
            trt_edgellm::kernel::moeTopkSoftmax(routerLogitsT, topkWeightsT, topkIdsT, mTopK,
                softmaxWs > 0 ? softmaxScratch : nullptr, softmaxWs, stream, /*renormalize=*/true, 0.0F,
                biasForSoftmax);

            float const* scalePtr = static_cast<float const*>(inputs[kIN_E_SCORE_CORRECTION_BIAS]);
            trt_edgellm::kernel::applyPerExpertScale(topkWeightsPtr, topkIdsPtr, scalePtr, numTokens, mTopK, stream);
        }
        CUDA_CHECK(cudaGetLastError());

        (void) outputDesc;

        // ==== Step 2: split FC1/FC2 MoE via CuTeDSL runner ====
        CuteDslNvfp4MoeSm110Params p{};
        p.numTokens = numTokens;
        p.numExperts = mNumExperts;
        p.topK = mTopK;
        p.hiddenSize = mHiddenSize;
        p.moeInterSize = mMoeInterSize;
        p.hiddenStates = inputs[kIN_HIDDEN_STATES];
        p.topkIds = topkIdsPtr;
        p.topkWeights = topkWeightsPtr;
        p.fc1QWeights = inputs[kIN_FC1_QWEIGHTS];
        p.fc1BlocksScale = inputs[kIN_FC1_BLOCKS_SCALE];
        p.fc1Alpha = static_cast<float const*>(inputs[kIN_FC1_ALPHA]);
        p.fc2QWeights = inputs[kIN_FC2_QWEIGHTS];
        p.fc2BlocksScale = inputs[kIN_FC2_BLOCKS_SCALE];
        p.fc2Alpha = static_cast<float const*>(inputs[kIN_FC2_ALPHA]);
        p.inputGlobalScale = static_cast<float const*>(inputs[kIN_INPUT_GLOBAL_SCALE]);
        p.downInputScale = static_cast<float const*>(inputs[kIN_DOWN_INPUT_SCALE]);
        p.output = outputs[0];
        p.activationType = mActivationType;
        p.enablePdl = requestNvfp4MoePdl();

        CuteDslNvfp4MoeSm110Runner runner;
        int32_t const rc = runner.run(p, runnerScratch, stream);
        if (rc != 0)
        {
            LOG_ERROR("Nvfp4MoePlugin: SM100/101/110 runner.run() failed with code %d", rc);
            return rc;
        }
        return 0;
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("Nvfp4MoePlugin enqueue failed: %s", e.what());
        return -1;
    }
#endif // CUTE_DSL_NVFP4_MOE_ENABLED
}

int32_t Nvfp4MoePlugin::onShapeChange(
    PluginTensorDesc const* in, int32_t nbInputs, PluginTensorDesc const* out, int32_t nbOutputs) noexcept
{
    if (in == nullptr || out == nullptr || nbInputs != kNbPluginInputs || nbOutputs != 1)
    {
        LOG_ERROR("Nvfp4MoePlugin: onShapeChange expected %d inputs and 1 output", kNbPluginInputs);
        return -1;
    }
    auto const& hiddenDims = in[kIN_HIDDEN_STATES].dims;
    int64_t numTokens{0};
    if (!getTokenCount(hiddenDims, numTokens) || hiddenDims.d[hiddenDims.nbDims - 1] != mHiddenSize || mTopK <= 0
        || numTokens > std::numeric_limits<int32_t>::max() / mTopK)
    {
        LOG_ERROR("Nvfp4MoePlugin: hidden_states must be [T,H] with H=%d", mHiddenSize);
        return -1;
    }
    auto const& routerDims = in[kIN_ROUTER_LOGITS].dims;
    if (routerDims.nbDims != 2 || routerDims.d[0] != numTokens || routerDims.d[1] != mNumExperts
        || !shapesMatch(out[0].dims, hiddenDims))
    {
        LOG_ERROR("Nvfp4MoePlugin: runtime router/output shapes do not match hidden_states");
        return -1;
    }
    // Runtime cap guard: the workspace was sized at build time for at most
    // \c mMaxRoutedRows. Refuse runtime shapes that would overflow it before
    // any device buffers get carved.
    int64_t const routedRows = numTokens * static_cast<int64_t>(mTopK);
    if (mMaxRoutedRows > 0 && routedRows > static_cast<int64_t>(mMaxRoutedRows))
    {
        LOG_ERROR(
            "Nvfp4MoePlugin::onShapeChange: runtime num_tokens*top_k "
            "(%lld * %d = %lld) exceeds the resolved max_routed_rows cap (%d). "
            "Rebuild the engine with a profile whose max shapes cover this case, or "
            "pass a larger explicit max_routed_rows attribute.",
            static_cast<long long>(numTokens), mTopK, static_cast<long long>(routedRows), mMaxRoutedRows);
        return -1;
    }
    return 0;
}

IPluginV3* Nvfp4MoePlugin::attachToContext(IPluginResourceContext* context) noexcept
{
    auto* plugin = static_cast<Nvfp4MoePlugin*>(clone());
    if (plugin == nullptr)
    {
        return nullptr;
    }
    (void) context;
    return plugin;
}

PluginFieldCollection const* Nvfp4MoePlugin::getFieldsToSerialize() noexcept
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
        mDataToSerialize.emplace_back("backend", &mBackend, PluginFieldType::kINT32, 1);
        mDataToSerialize.emplace_back("max_routed_rows", &mMaxRoutedRows, PluginFieldType::kINT32, 1);
        mDataToSerialize.emplace_back("io_dtype", &mIoDtype, PluginFieldType::kINT32, 1);
        mFCToSerialize.nbFields = mDataToSerialize.size();
        mFCToSerialize.fields = mDataToSerialize.data();
        return &mFCToSerialize;
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("Failed to serialize Nvfp4MoePlugin fields: %s", e.what());
        return nullptr;
    }
}

// Creator

Nvfp4MoePluginCreator::Nvfp4MoePluginCreator()
{
    static std::mutex sMutex;
    std::lock_guard<std::mutex> lock(sMutex);

    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("num_experts", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("top_k", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("hidden_size", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("moe_inter_size", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("activation_type", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("n_group", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("topk_group", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("norm_topk_prob", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("routed_scaling_factor", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("routing_mode", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("backend", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("max_routed_rows", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("io_dtype", nullptr, PluginFieldType::kINT32, 1));

    mFieldCollection.nbFields = mPluginAttributes.size();
    mFieldCollection.fields = mPluginAttributes.data();
}

char const* Nvfp4MoePluginCreator::getPluginName() const noexcept
{
    return kPLUGIN_NAME;
}

char const* Nvfp4MoePluginCreator::getPluginVersion() const noexcept
{
    return kPLUGIN_VERSION;
}

PluginFieldCollection const* Nvfp4MoePluginCreator::getFieldNames() noexcept
{
    return &mFieldCollection;
}

void Nvfp4MoePluginCreator::setPluginNamespace(char const* libNamespace) noexcept
{
    mNamespace = libNamespace ? libNamespace : "";
}

char const* Nvfp4MoePluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

IPluginV3* Nvfp4MoePluginCreator::createPlugin(
    char const* name, PluginFieldCollection const* fc, TensorRTPhase phase) noexcept
{
    (void) phase;
    try
    {
        auto* plugin = new Nvfp4MoePlugin(std::string(name), fc);
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("Failed to create Nvfp4MoePlugin: %s", e.what());
        return nullptr;
    }
}

} // namespace plugins
} // namespace trt_edgellm
