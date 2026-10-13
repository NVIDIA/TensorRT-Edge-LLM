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

#include "int4MoePlugin.h"

#include "common/checkMacros.h"
#include "common/cudaUtils.h"
#include "common/logger.h"
#include "common/stringUtils.h"
#include "common/tensor.h"
#include "kernels/moe/moeActivationKernels.h"
#include "kernels/moe/moeAlignSumKernels.h"
#include "kernels/moe/moeMarlinIndicesKernels.h"
#include "kernels/moe/moeTopkSoftmaxKernels.h"
#include "kernels/moe/moe_marlin/moeMarlin.h"
#include "plugins/utils/pluginUtils.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <limits>
#include <mutex>

using namespace nvinfer1;
namespace trt_edgellm
{
namespace plugins
{

namespace
{
constexpr char const* kINT4_MOE_PLUGIN_VERSION{"1"};
constexpr char const* kINT4_MOE_PLUGIN_NAME{"Int4MoePlugin"};

// Supported plugin-level activation types.
constexpr auto kACTIVATION_SILU = static_cast<ActivationType>(0);
constexpr auto kACTIVATION_GEGLU = static_cast<ActivationType>(5);
constexpr std::array<ActivationType, 2> kSUPPORTED_ACTIVATION_TYPES{kACTIVATION_SILU, kACTIVATION_GEGLU};
constexpr int32_t kSMALL_TOKEN_BLOCK_SIZE{8};
constexpr int32_t kLARGE_TOKEN_BLOCK_SIZE{32};
constexpr int32_t kTOKEN_MAJOR_SMALL_TOKEN_THRESHOLD{8};

bool getTokenCount(Dims const& hidden, int32_t hiddenSize, int64_t& tokenCount) noexcept
{
    if (hidden.nbDims != 2 || hidden.d[1] != hiddenSize || hidden.d[0] <= 0)
    {
        return false;
    }
    tokenCount = hidden.d[0];
    return true;
}

bool hasSameDims(Dims const& lhs, Dims const& rhs) noexcept
{
    if (lhs.nbDims != rhs.nbDims)
    {
        return false;
    }
    for (int32_t index = 0; index < lhs.nbDims; ++index)
    {
        if (lhs.d[index] != rhs.d[index])
        {
            return false;
        }
    }
    return true;
}

//! Selects a Marlin tile by aggregate token count. This is a performance heuristic; both tiles preserve the same
//! routing semantics and do not identify the execution phase.
int32_t getTunedMoeBlockSize(int64_t numTokens) noexcept
{
    return numTokens <= kTOKEN_MAJOR_SMALL_TOKEN_THRESHOLD ? kSMALL_TOKEN_BLOCK_SIZE : kLARGE_TOKEN_BLOCK_SIZE;
}

bool hasValidRoutedRowCapacity(int64_t numTokens, int32_t numExperts, int32_t topK) noexcept
{
    int64_t const padding = static_cast<int64_t>(numExperts) * kLARGE_TOKEN_BLOCK_SIZE;
    return numTokens > 0 && numExperts > 0 && topK > 0 && padding < std::numeric_limits<int32_t>::max()
        && numTokens <= (std::numeric_limits<int32_t>::max() - padding) / topK;
}

// Workspace size for Int4 MoE plugin using accumulateWorkspaceSize (same order as assignTensorFromWorkspace in
// enqueue).
size_t computeInt4MoeWorkspaceSize(
    int64_t numTokens, int32_t numExperts, int32_t topK, int32_t hiddenSize, int32_t moeInterSize) noexcept
{
    try
    {
        if (!hasValidRoutedRowCapacity(numTokens, numExperts, topK))
        {
            return 0;
        }

        int32_t dev = 0;
        int32_t sms = 0;
        CUDA_CHECK(cudaGetDevice(&dev));
        CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));

        int32_t const totalSlots = static_cast<int32_t>(numTokens) * topK;
        int32_t const maxPaddedSlots = totalSlots + numExperts * kLARGE_TOKEN_BLOCK_SIZE;
        int32_t const maxPaddedBlocks = std::max(
            static_cast<int32_t>(divUp(totalSlots + numExperts * kSMALL_TOKEN_BLOCK_SIZE, kSMALL_TOKEN_BLOCK_SIZE)),
            static_cast<int32_t>(divUp(maxPaddedSlots, kLARGE_TOKEN_BLOCK_SIZE)));
        size_t softmaxWorkspaceSizeBytes
            = trt_edgellm::kernel::getMoeTopkSoftmaxWorkspaceSize(static_cast<int32_t>(numTokens), numExperts);
        int64_t marlinWorkspaceSize{};
        for (int32_t const blockSize : {kSMALL_TOKEN_BLOCK_SIZE, kLARGE_TOKEN_BLOCK_SIZE})
        {
            marlinWorkspaceSize = std::max(marlinWorkspaceSize,
                trt_edgellm::kernel::getMoeMarlinWorkspaceSize(maxPaddedSlots, 2 * moeInterSize, blockSize, sms));
            marlinWorkspaceSize = std::max(marlinWorkspaceSize,
                trt_edgellm::kernel::getMoeMarlinWorkspaceSize(maxPaddedSlots, hiddenSize, blockSize, sms));
        }

        size_t size = 0;
        // TopK softmax outputs: selected expert weights and indices per token [numTokens, topK]
        size = accumulateWorkspaceSize(size, rt::Coords{numTokens, topK}, DataType::kFLOAT);
        size = accumulateWorkspaceSize(size, rt::Coords{numTokens, topK}, DataType::kINT32);
        // Optional softmax fallback workspace when numExperts is not a power of 2 (cub/radix sort etc.)
        if (softmaxWorkspaceSizeBytes > 0)
        {
            size = accumulateWorkspaceSize(
                size, rt::Coords{static_cast<int64_t>(softmaxWorkspaceSizeBytes)}, DataType::kINT8);
        }
        // Marlin GEMM indices: token order and expert id per block after padding for coalesced access
        size = accumulateWorkspaceSize(size, rt::Coords{maxPaddedSlots}, DataType::kINT32);
        size = accumulateWorkspaceSize(size, rt::Coords{maxPaddedBlocks}, DataType::kINT32);
        size = accumulateWorkspaceSize(size, rt::Coords{1}, DataType::kINT32);
        size = accumulateWorkspaceSize(size, rt::Coords{maxPaddedSlots}, DataType::kFLOAT);
        // Per-expert padded slot counts/offsets used to build sorted indices
        size = accumulateWorkspaceSize(size, rt::Coords{numExperts}, DataType::kINT32);
        size = accumulateWorkspaceSize(size, rt::Coords{numExperts}, DataType::kINT32);
        // Slot lists per expert and slot count per expert (buildMarlinIndices / countSlotsPerExpert / buildSlotLists)
        size = accumulateWorkspaceSize(
            size, rt::Coords{static_cast<int64_t>(numExperts) * totalSlots}, DataType::kINT32);
        size = accumulateWorkspaceSize(size, rt::Coords{numExperts}, DataType::kINT32);
        // Expert GEMM intermediates: gate-up output, post-activation, down projection output
        size = accumulateWorkspaceSize(size, rt::Coords{totalSlots, 2 * moeInterSize}, DataType::kHALF);
        size = accumulateWorkspaceSize(size, rt::Coords{totalSlots, moeInterSize}, DataType::kHALF);
        size = accumulateWorkspaceSize(size, rt::Coords{totalSlots, hiddenSize}, DataType::kHALF);
        // Marlin W4A16 GEMM workspace (locks + FP32 reduction buffer)
        size = accumulateWorkspaceSize(size, rt::Coords{marlinWorkspaceSize}, DataType::kINT32);
        // Routing weights in padded slot order for down GEMM (mulTopkWeights=true)
        size = accumulateWorkspaceSize(size, rt::Coords{maxPaddedSlots}, DataType::kFLOAT);
        return size;
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("Failed to compute Int4MoePlugin workspace size: %s", e.what());
        return 0;
    }
}
} // namespace

// Static class fields initialization
PluginFieldCollection Int4MoePluginCreator::mFieldCollection{};
std::vector<PluginField> Int4MoePluginCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(Int4MoePluginCreator);

Int4MoePlugin::Int4MoePlugin(std::string const& name, int32_t const numExperts, int32_t const topK,
    int32_t const hiddenSize, int32_t const moeInterSize, ActivationType const activationType,
    int32_t const quantizationGroupSize)
    : mLayerName(name)
    , mNumExperts(numExperts)
    , mTopK(topK)
    , mHiddenSize(hiddenSize)
    , mMoeInterSize(moeInterSize)
    , mActivationType(activationType)
    , mQuantizationGroupSize(quantizationGroupSize)
{
    if (std::find(kSUPPORTED_ACTIVATION_TYPES.begin(), kSUPPORTED_ACTIVATION_TYPES.end(), mActivationType)
        == kSUPPORTED_ACTIVATION_TYPES.end())
    {
        LOG_ERROR("Int4MoePlugin supports SiLU (type 0) and GeGLU (type 5), got type %d",
            static_cast<int32_t>(mActivationType));
    }
}

Int4MoePlugin::Int4MoePlugin(std::string const& name, PluginFieldCollection const* fc)
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
            mActivationType = *static_cast<nvinfer1::ActivationType const*>(fc->fields[i].data);
        }
        else if (fieldName == "quantization_group_size")
        {
            mQuantizationGroupSize = *static_cast<int32_t const*>(fc->fields[i].data);
        }
    }

    if (std::find(kSUPPORTED_ACTIVATION_TYPES.begin(), kSUPPORTED_ACTIVATION_TYPES.end(), mActivationType)
        == kSUPPORTED_ACTIVATION_TYPES.end())
    {
        LOG_ERROR("Int4MoePlugin supports SiLU (type 0) and GeGLU (type 5), got type %d",
            static_cast<int32_t>(mActivationType));
    }
}

Int4MoePlugin::~Int4MoePlugin() noexcept {}

IPluginCapability* Int4MoePlugin::getCapabilityInterface(PluginCapabilityType type) noexcept
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

IPluginV3* Int4MoePlugin::clone() noexcept
{
    try
    {
        auto* plugin = new Int4MoePlugin(
            mLayerName, mNumExperts, mTopK, mHiddenSize, mMoeInterSize, mActivationType, mQuantizationGroupSize);
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("Failed to clone Int4MoePlugin: %s", e.what());
        return nullptr;
    }
}

char const* Int4MoePlugin::getPluginName() const noexcept
{
    return kINT4_MOE_PLUGIN_NAME;
}

char const* Int4MoePlugin::getPluginVersion() const noexcept
{
    return kINT4_MOE_PLUGIN_VERSION;
}

char const* Int4MoePlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

void Int4MoePlugin::setPluginNamespace(char const* pluginNamespace) noexcept
{
    mNamespace = std::string(pluginNamespace);
}

int32_t Int4MoePlugin::getNbOutputs() const noexcept
{
    return 1;
}

int32_t Int4MoePlugin::getOutputDataTypes(DataType* outputTypes, int32_t nbOutputs,
    [[maybe_unused]] DataType const* inputTypes, [[maybe_unused]] int32_t nbInputs) const noexcept
{
    assert(nbOutputs == 1);
    outputTypes[0] = DataType::kHALF;
    return 0;
}

int32_t Int4MoePlugin::getOutputShapes(DimsExprs const* inputs, int32_t nbInputs,
    [[maybe_unused]] DimsExprs const* shapeInputs, [[maybe_unused]] int32_t nbShapeInputs, DimsExprs* outputs,
    int32_t nbOutputs, [[maybe_unused]] IExprBuilder& exprBuilder) noexcept
{
    if (inputs == nullptr || outputs == nullptr || nbInputs != 6 || nbOutputs != 1 || inputs[1].nbDims != 2)
    {
        return -1;
    }
    outputs[0].nbDims = inputs[1].nbDims;
    for (int32_t index = 0; index < outputs[0].nbDims; ++index)
    {
        outputs[0].d[index] = inputs[1].d[index];
    }
    return 0;
}

bool Int4MoePlugin::supportsFormatCombination(
    int32_t pos, DynamicPluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
    auto checkHiddenStates = [this](nvinfer1::PluginTensorDesc const& tensorDesc) {
        bool status{true};
        status &= tensorDesc.type == DataType::kHALF;
        status &= tensorDesc.format == TensorFormat::kLINEAR;
        status &= tensorDesc.dims.nbDims == 2;
        auto const tensorDim = tensorDesc.dims;
        if (status)
        {
            status &= tensorDim.d[1] == mHiddenSize;
        }
        return status;
    };

    // router_logits (numTokens, numExperts) FP32; d[0] may be -1 (dynamic)
    auto checkRouterLogits = [this](nvinfer1::PluginTensorDesc const& tensorDesc) {
        bool status{true};
        status &= tensorDesc.type == DataType::kFLOAT;
        status &= tensorDesc.format == TensorFormat::kLINEAR;
        status &= tensorDesc.dims.nbDims == 2;
        auto const tensorDim = tensorDesc.dims;
        if (status)
        {
            status &= tensorDim.d[1] == mNumExperts;
        }
        return status;
    };

    // Marlin packed format: int8 view of int32-packed weights [E, K//16, 2*N]
    // gate_up: N=2*moeInterSize, so int32 last dim = 2*N = 4*moeInterSize; int8 = 16*moeInterSize
    auto checkFcGateUpQWeights = [this](nvinfer1::PluginTensorDesc const& tensorDesc) {
        bool status{true};
        status &= tensorDesc.type == DataType::kINT8;
        status &= tensorDesc.format == TensorFormat::kLINEAR;
        status &= tensorDesc.dims.nbDims == 3;
        auto const tensorDim = tensorDesc.dims;
        int32_t const expectedLastDim = (2 * 2 * mMoeInterSize) * 4; // 2*N int32 -> int8: 16*moeInterSize
        if (status)
        {
            status &= tensorDim.d[0] == mNumExperts;
            status &= tensorDim.d[1] == mHiddenSize / 16;
            status &= tensorDim.d[2] == expectedLastDim;
        }
        return status;
    };

    // gate_up scales: [E, hiddenSize/quantization_group_size, 2*moeInterSize]
    auto checkFcGateUpScales = [this](nvinfer1::PluginTensorDesc const& tensorDesc) {
        bool status{true};
        status &= tensorDesc.type == DataType::kHALF;
        status &= tensorDesc.format == TensorFormat::kLINEAR;
        status &= tensorDesc.dims.nbDims == 3;
        auto const tensorDim = tensorDesc.dims;
        if (status)
        {
            status &= tensorDim.d[0] == mNumExperts;
            status &= tensorDim.d[1] == mHiddenSize / mQuantizationGroupSize;
            status &= tensorDim.d[2] == 2 * mMoeInterSize; // Fused gate+up
        }
        return status;
    };

    // down: [E, K//16, 2*N*4] for GEMM input K=moeInterSize, output N=hiddenSize
    auto checkFcDownQWeights = [this](nvinfer1::PluginTensorDesc const& tensorDesc) {
        bool status{true};
        status &= tensorDesc.type == DataType::kINT8;
        status &= tensorDesc.format == TensorFormat::kLINEAR;
        status &= tensorDesc.dims.nbDims == 3;
        auto const tensorDim = tensorDesc.dims;
        if (status)
        {
            status &= tensorDim.d[0] == mNumExperts;
            status &= tensorDim.d[1] == mMoeInterSize / 16;
            status &= tensorDim.d[2] == (2 * mHiddenSize) * 4; // int8 view of int32
        }
        return status;
    };

    // down scales: [E, moeInterSize/quantization_group_size, hiddenSize]
    auto checkFcDownScales = [this](nvinfer1::PluginTensorDesc const& tensorDesc) {
        bool status{true};
        status &= tensorDesc.type == DataType::kHALF;
        status &= tensorDesc.format == TensorFormat::kLINEAR;
        status &= tensorDesc.dims.nbDims == 3;
        auto const tensorDim = tensorDesc.dims;
        if (status)
        {
            status &= tensorDim.d[0] == mNumExperts;
            status &= tensorDim.d[1] == mMoeInterSize / mQuantizationGroupSize;
            status &= tensorDim.d[2] == mHiddenSize;
        }
        return status;
    };

    assert(nbInputs == 6 && nbOutputs == 1);
    assert(pos < (nbInputs + nbOutputs));

    auto const& tensorDesc = inOut[pos].desc;

    switch (pos)
    {
    case 0: return checkRouterLogits(tensorDesc);
    case 1: return checkHiddenStates(tensorDesc);
    case 2: return checkFcGateUpQWeights(tensorDesc);
    case 3: return checkFcGateUpScales(tensorDesc);
    case 4: return checkFcDownQWeights(tensorDesc);
    case 5: return checkFcDownScales(tensorDesc);
    case 6: return checkHiddenStates(tensorDesc);
    default: return false;
    }
}

int32_t Int4MoePlugin::configurePlugin(
    DynamicPluginTensorDesc const* in, int32_t nbInputs, DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept
{
    if (in == nullptr || out == nullptr || nbInputs != 6 || nbOutputs != 1)
    {
        return -1;
    }
    auto validateEndpoint = [this](Dims const& hidden, Dims const& router, Dims const& output) {
        int64_t numTokens{};
        return getTokenCount(hidden, mHiddenSize, numTokens) && router.nbDims == 2 && router.d[0] == numTokens
            && router.d[1] == mNumExperts && hasSameDims(output, hidden);
    };
    if (!validateEndpoint(in[1].min, in[0].min, out[0].min) || !validateEndpoint(in[1].opt, in[0].opt, out[0].opt)
        || !validateEndpoint(in[1].max, in[0].max, out[0].max))
    {
        LOG_ERROR("Int4MoePlugin: profile router/output shapes must match rank-2 hidden_states");
        return -1;
    }
    int64_t maxTokens{};
    if (!getTokenCount(in[1].max, mHiddenSize, maxTokens) || !hasValidRoutedRowCapacity(maxTokens, mNumExperts, mTopK))
    {
        LOG_ERROR("Int4MoePlugin: profile maximum exceeds the routed-row index capacity");
        return -1;
    }
    return 0;
}

size_t Int4MoePlugin::getWorkspaceSize(DynamicPluginTensorDesc const* inputs, int32_t nbInputs,
    [[maybe_unused]] DynamicPluginTensorDesc const* outputs, [[maybe_unused]] int32_t nbOutputs) const noexcept
{
    if (inputs == nullptr || outputs == nullptr || nbInputs != 6 || nbOutputs != 1)
    {
        return 0;
    }
    auto const& hiddenStatesMaxDims = inputs[1].max;
    int64_t maxTokens{};
    if (!getTokenCount(hiddenStatesMaxDims, mHiddenSize, maxTokens) || inputs[0].max.nbDims != 2
        || inputs[0].max.d[0] != maxTokens || inputs[0].max.d[1] != mNumExperts
        || !hasSameDims(outputs[0].max, hiddenStatesMaxDims))
    {
        LOG_ERROR("Int4MoePlugin: invalid profile maximum shapes while computing workspace");
        return 0;
    }
    return computeInt4MoeWorkspaceSize(maxTokens, mNumExperts, mTopK, mHiddenSize, mMoeInterSize);
}

int32_t Int4MoePlugin::enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc,
    void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    try
    {
        using namespace trt_edgellm::kernel;
        using namespace trt_edgellm::rt;

        if (inputDesc == nullptr || outputDesc == nullptr || inputs == nullptr || outputs == nullptr
            || workspace == nullptr || outputs[0] == nullptr)
        {
            LOG_ERROR("Int4MoePlugin: enqueue received a null descriptor or buffer");
            return -1;
        }
        for (int32_t index = 0; index < 6; ++index)
        {
            if (inputs[index] == nullptr)
            {
                LOG_ERROR("Int4MoePlugin: null runtime input at position %d", index);
                return -1;
            }
        }

        // inputs[0] = router_logits (numTokens, numExperts), inputs[1] = hidden_states, then expert weights
        PluginTensorDesc const& hiddenStatesDesc = inputDesc[1];
        int64_t numTokens64{};
        if (!getTokenCount(hiddenStatesDesc.dims, mHiddenSize, numTokens64)
            || !hasValidRoutedRowCapacity(numTokens64, mNumExperts, mTopK) || inputDesc[0].dims.nbDims != 2
            || inputDesc[0].dims.d[0] != numTokens64 || inputDesc[0].dims.d[1] != mNumExperts
            || !hasSameDims(outputDesc[0].dims, hiddenStatesDesc.dims))
        {
            LOG_ERROR("Int4MoePlugin: runtime router/output shapes must match rank-2 hidden_states");
            return -1;
        }
        int32_t const numTokens = static_cast<int32_t>(numTokens64);
        int32_t const moeBlockSize = getTunedMoeBlockSize(numTokens64);

        rt::Tensor hiddenStatesTensor(const_cast<void*>(inputs[1]), rt::Coords{hiddenStatesDesc.dims},
            rt::DeviceType::kGPU, hiddenStatesDesc.type);

        // Weights arrive as INT8 (view of Marlin int32-packed); Marlin expects INT32
        rt::Tensor gateUpQWeightsTensor(const_cast<void*>(inputs[2]),
            rt::Coords{mNumExperts, mHiddenSize / 16, 2 * mMoeInterSize}, rt::DeviceType::kGPU, DataType::kINT32);

        rt::Tensor gateUpScalesTensor(
            const_cast<void*>(inputs[3]), rt::Coords{inputDesc[3].dims}, rt::DeviceType::kGPU, inputDesc[3].type);

        rt::Tensor downQWeightsTensor(const_cast<void*>(inputs[4]),
            rt::Coords{mNumExperts, mMoeInterSize / 16, 2 * mHiddenSize}, rt::DeviceType::kGPU, DataType::kINT32);

        rt::Tensor downScalesTensor(
            const_cast<void*>(inputs[5]), rt::Coords{inputDesc[5].dims}, rt::DeviceType::kGPU, inputDesc[5].type);

        rt::Tensor outputTensor(outputs[0], rt::Coords{outputDesc[0].dims}, rt::DeviceType::kGPU, outputDesc[0].type);

        // Get GPU info
        int32_t dev = 0;
        int32_t sms = 0;
        CUDA_CHECK(cudaGetDevice(&dev));
        CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));

        std::byte* alignedWorkspacePtr = static_cast<std::byte*>(workspace);

        // ==================== Workspace Allocation ====================
        int32_t totalSlots = numTokens * mTopK;
        int32_t maxPaddedSlots = totalSlots + mNumExperts * moeBlockSize;
        int32_t maxPaddedBlocks = divUp(maxPaddedSlots, moeBlockSize);

        // Step 1: TopK Softmax outputs (router_logits from inputs[0] FP32, cast done in Python)
        rt::Tensor routerLogitsTensor(const_cast<void*>(inputs[0]), rt::Coords{inputDesc[0].dims}, rt::DeviceType::kGPU,
            nvinfer1::DataType::kFLOAT);
        // Selected expert weights and indices per token [numTokens, topK] for routing and aggregation
        float* topkWeightsPtr = static_cast<float*>(
            assignTensorFromWorkspace(alignedWorkspacePtr, {numTokens, mTopK}, DataType::kFLOAT).rawPointer());
        int32_t* topkIndicesPtr = static_cast<int32_t*>(
            assignTensorFromWorkspace(alignedWorkspacePtr, {numTokens, mTopK}, DataType::kINT32).rawPointer());
        // Optional workspace for MoE top-k softmax when numExperts is not a power of 2 (cub/radix sort fallback)
        size_t softmaxWorkspaceSizeBytes = getMoeTopkSoftmaxWorkspaceSize(numTokens, mNumExperts);
        void* softmaxWorkspacePtr = (softmaxWorkspaceSizeBytes > 0)
            ? assignTensorFromWorkspace(
                  alignedWorkspacePtr, {static_cast<int64_t>(softmaxWorkspaceSizeBytes)}, DataType::kINT8)
                  .rawPointer()
            : nullptr;

        // Step 2: Marlin GEMM indices — token order and expert id per block after padding for coalesced GEMM
        int32_t* sortedTokenIdsPtr = static_cast<int32_t*>(
            assignTensorFromWorkspace(alignedWorkspacePtr, {maxPaddedSlots}, DataType::kINT32).rawPointer());
        int32_t* expertIdsPtr = static_cast<int32_t*>(
            assignTensorFromWorkspace(alignedWorkspacePtr, {maxPaddedBlocks}, DataType::kINT32).rawPointer());
        int32_t* numTokensPostPaddedPtr
            = static_cast<int32_t*>(assignTensorFromWorkspace(alignedWorkspacePtr, {1}, DataType::kINT32).rawPointer());
        float* topkWeightsFlatPtr = static_cast<float*>(
            assignTensorFromWorkspace(alignedWorkspacePtr, {maxPaddedSlots}, DataType::kFLOAT).rawPointer());
        // Per-expert padded slot counts and offsets used to build sorted indices
        int32_t* paddedCountsPtr = static_cast<int32_t*>(
            assignTensorFromWorkspace(alignedWorkspacePtr, {mNumExperts}, DataType::kINT32).rawPointer());
        int32_t* paddedOffsetsPtr = static_cast<int32_t*>(
            assignTensorFromWorkspace(alignedWorkspacePtr, {mNumExperts}, DataType::kINT32).rawPointer());

        // Slot lists per expert and slot count per expert (buildMarlinIndices / countSlotsPerExpert / buildSlotLists)
        int32_t* slotsByExpertWorkspace = static_cast<int32_t*>(assignTensorFromWorkspace(
            alignedWorkspacePtr, {static_cast<int64_t>(mNumExperts) * totalSlots}, DataType::kINT32)
                .rawPointer());
        int32_t* slotsPerExpertWorkspace = static_cast<int32_t*>(
            assignTensorFromWorkspace(alignedWorkspacePtr, {mNumExperts}, DataType::kINT32).rawPointer());

        // Step 3: Expert GEMM intermediates — gate-up output, post–SwiGLU, down projection output
        half* gateUpOutputPtr = static_cast<half*>(
            assignTensorFromWorkspace(alignedWorkspacePtr, {totalSlots, 2 * mMoeInterSize}, DataType::kHALF)
                .rawPointer());
        half* activationOutputPtr = static_cast<half*>(
            assignTensorFromWorkspace(alignedWorkspacePtr, {totalSlots, mMoeInterSize}, DataType::kHALF).rawPointer());
        half* downOutputPtr = static_cast<half*>(
            assignTensorFromWorkspace(alignedWorkspacePtr, {totalSlots, mHiddenSize}, DataType::kHALF).rawPointer());

        // Marlin W4A16 GEMM workspace (locks + FP32 reduction buffer)
        int64_t marlinWorkspaceSize = std::max(
            trt_edgellm::kernel::getMoeMarlinWorkspaceSize(maxPaddedSlots, 2 * mMoeInterSize, moeBlockSize, sms),
            trt_edgellm::kernel::getMoeMarlinWorkspaceSize(maxPaddedSlots, mHiddenSize, moeBlockSize, sms));
        int32_t* marlinWorkspacePtr = static_cast<int32_t*>(
            assignTensorFromWorkspace(alignedWorkspacePtr, {marlinWorkspaceSize}, DataType::kINT32).rawPointer());

        // ==================== Step 1: TopK Softmax on router_logits ====================
        rt::Tensor topkWeightsTensor(
            topkWeightsPtr, {numTokens, mTopK}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT);
        rt::Tensor topkIndicesTensor(
            topkIndicesPtr, {numTokens, mTopK}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);

        moeTopkSoftmax(routerLogitsTensor, topkWeightsTensor, topkIndicesTensor, mTopK, softmaxWorkspacePtr,
            softmaxWorkspaceSizeBytes, stream, true, 0.0f);
        CUDA_CHECK(cudaGetLastError());

        // Reshape hidden_states to 2D for expert GEMMs
        rt::Tensor hiddenStates2D(const_cast<void*>(hiddenStatesTensor.rawPointer()), {numTokens, mHiddenSize},
            rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);

        // ==================== Step 2: Prepare Marlin GEMM Indices ====================
        CUDA_CHECK(cudaMemsetAsync(slotsPerExpertWorkspace, 0, mNumExperts * sizeof(int32_t), stream));
        kernel::launchCountSlotsPerExpertKernel(
            topkIndicesPtr, slotsPerExpertWorkspace, numTokens, mTopK, mNumExperts, stream);
        CUDA_CHECK(cudaGetLastError());

        kernel::launchComputePaddedOffsetsKernel(slotsPerExpertWorkspace, paddedCountsPtr, paddedOffsetsPtr,
            numTokensPostPaddedPtr, mNumExperts, moeBlockSize, stream);
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaMemsetAsync(slotsPerExpertWorkspace, 0, mNumExperts * sizeof(int32_t), stream));
        kernel::launchBuildSlotListsKernel(
            topkIndicesPtr, slotsByExpertWorkspace, slotsPerExpertWorkspace, numTokens, mTopK, mNumExperts, stream);
        CUDA_CHECK(cudaGetLastError());

        rt::Tensor sortedTokenIdsTensor(
            sortedTokenIdsPtr, {maxPaddedSlots}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);
        rt::Tensor expertIdsTensor(expertIdsPtr, {maxPaddedBlocks}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);
        rt::Tensor numTokensPostPaddedTensor(
            numTokensPostPaddedPtr, {1}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);
        rt::Tensor topkWeightsFlatTensor(
            topkWeightsFlatPtr, {maxPaddedSlots}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT);
        rt::Tensor marlinWorkspaceTensor(
            marlinWorkspacePtr, {marlinWorkspaceSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);

        kernel::launchBuildMarlinIndicesKernel(slotsByExpertWorkspace, slotsPerExpertWorkspace, paddedCountsPtr,
            paddedOffsetsPtr, topkWeightsPtr, sortedTokenIdsPtr, topkWeightsFlatPtr, expertIdsPtr, numTokens, mTopK,
            mNumExperts, moeBlockSize, stream);

        // ==================== Step 3: Gate-Up Projection (Marlin INT4 GEMM) ====================
        rt::Tensor gateUpOutputTensor(
            gateUpOutputPtr, {totalSlots, 2 * mMoeInterSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);

        moeAwqW4A16MarlinGemm(hiddenStates2D, gateUpOutputTensor, gateUpQWeightsTensor, gateUpScalesTensor,
            sortedTokenIdsTensor, expertIdsTensor, numTokensPostPaddedTensor, topkWeightsFlatTensor,
            marlinWorkspaceTensor, moeBlockSize, mTopK, false, stream);
        CUDA_CHECK(cudaGetLastError());

        // ==================== Step 4: Gated Activation ====================
        rt::Tensor activationOutputTensor(
            activationOutputPtr, {totalSlots, mMoeInterSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);

        int32_t const kernelActivationType = (mActivationType == kACTIVATION_GEGLU)
            ? trt_edgellm::kernel::MoeActivationType::kMoeGeGlu
            : trt_edgellm::kernel::MoeActivationType::kMoeSwiGlu;
        moeActivation(
            gateUpOutputTensor, activationOutputTensor, totalSlots, mMoeInterSize, kernelActivationType, stream);
        CUDA_CHECK(cudaGetLastError());

        // ==================== Step 5: Down Projection (Marlin INT4 GEMM) ====================
        rt::Tensor downOutputTensor(
            downOutputPtr, {totalSlots, mHiddenSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);

        // Routing weights in padded slot order for down GEMM (mulTopkWeights=true)
        float* topkWeightsPaddedPtr = static_cast<float*>(
            assignTensorFromWorkspace(alignedWorkspacePtr, {maxPaddedSlots}, DataType::kFLOAT).rawPointer());
        cudaMemsetAsync(topkWeightsPaddedPtr, 0, maxPaddedSlots * sizeof(float), stream);
        cudaMemcpyAsync(
            topkWeightsPaddedPtr, topkWeightsPtr, totalSlots * sizeof(float), cudaMemcpyDeviceToDevice, stream);
        rt::Tensor topkWeightsPaddedTensor(
            topkWeightsPaddedPtr, {maxPaddedSlots}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT);

        moeAwqW4A16MarlinGemm(activationOutputTensor, downOutputTensor, downQWeightsTensor, downScalesTensor,
            sortedTokenIdsTensor, expertIdsTensor, numTokensPostPaddedTensor, topkWeightsPaddedTensor,
            marlinWorkspaceTensor, moeBlockSize, 1, true, stream);
        CUDA_CHECK(cudaGetLastError());

        // ==================== Step 6: Aggregate Slots to Tokens ====================
        kernel::launchAggregateSlotOutputsKernel(
            downOutputPtr, outputTensor.rawPointer(), numTokens, mTopK, mHiddenSize, stream);
        return 0;
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("Int4MoePlugin enqueue failed: %s", e.what());
        return -1;
    }
}

int32_t Int4MoePlugin::onShapeChange(
    PluginTensorDesc const* in, int32_t nbInputs, PluginTensorDesc const* out, int32_t nbOutputs) noexcept
{
    if (in == nullptr || out == nullptr || nbInputs != 6 || nbOutputs != 1)
    {
        return -1;
    }
    int64_t numTokens{};
    if (!getTokenCount(in[1].dims, mHiddenSize, numTokens) || !hasValidRoutedRowCapacity(numTokens, mNumExperts, mTopK)
        || in[0].dims.nbDims != 2 || in[0].dims.d[0] != numTokens || in[0].dims.d[1] != mNumExperts
        || !hasSameDims(out[0].dims, in[1].dims))
    {
        LOG_ERROR("Int4MoePlugin: runtime router/output shapes must match rank-2 hidden_states");
        return -1;
    }
    return 0;
}

IPluginV3* Int4MoePlugin::attachToContext([[maybe_unused]] IPluginResourceContext* context) noexcept
{
    return clone();
}

PluginFieldCollection const* Int4MoePlugin::getFieldsToSerialize() noexcept
{
    try
    {
        mDataToSerialize.clear();
        mDataToSerialize.emplace_back("num_experts", &mNumExperts, PluginFieldType::kINT32, 1);
        mDataToSerialize.emplace_back("top_k", &mTopK, PluginFieldType::kINT32, 1);
        mDataToSerialize.emplace_back("hidden_size", &mHiddenSize, PluginFieldType::kINT32, 1);
        mDataToSerialize.emplace_back("moe_inter_size", &mMoeInterSize, PluginFieldType::kINT32, 1);
        mDataToSerialize.emplace_back("activation_type", &mActivationType, PluginFieldType::kINT32, 1);
        mDataToSerialize.emplace_back("quantization_group_size", &mQuantizationGroupSize, PluginFieldType::kINT32, 1);

        mFCToSerialize.nbFields = mDataToSerialize.size();
        mFCToSerialize.fields = mDataToSerialize.data();
        return &mFCToSerialize;
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("Failed to serialize Int4MoePlugin fields: %s", e.what());
        return nullptr;
    }
}

Int4MoePluginCreator::Int4MoePluginCreator()
{
    static std::mutex sMutex;
    std::lock_guard<std::mutex> lock(sMutex);

    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("num_experts", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("top_k", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("hidden_size", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("moe_inter_size", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("activation_type", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("quantization_group_size", nullptr, PluginFieldType::kINT32, 1));

    mFieldCollection.nbFields = mPluginAttributes.size();
    mFieldCollection.fields = mPluginAttributes.data();
}

char const* Int4MoePluginCreator::getPluginName() const noexcept
{
    return kINT4_MOE_PLUGIN_NAME;
}

nvinfer1::PluginFieldCollection const* Int4MoePluginCreator::getFieldNames() noexcept
{
    return &mFieldCollection;
}

void Int4MoePluginCreator::setPluginNamespace(char const* libNamespace) noexcept
{
    mNamespace = libNamespace;
}

char const* Int4MoePluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

char const* Int4MoePluginCreator::getPluginVersion() const noexcept
{
    return kINT4_MOE_PLUGIN_VERSION;
}

IPluginV3* Int4MoePluginCreator::createPlugin(
    char const* name, PluginFieldCollection const* fc, [[maybe_unused]] TensorRTPhase phase) noexcept
{
    try
    {
        Int4MoePlugin* plugin = new Int4MoePlugin(std::string(name), fc);
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("Failed to create Int4MoePlugin: %s", e.what());
        return nullptr;
    }
}

} // namespace plugins
} // namespace trt_edgellm
