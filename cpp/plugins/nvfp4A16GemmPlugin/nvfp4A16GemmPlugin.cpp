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

#include "nvfp4A16GemmPlugin.h"

#include "common/checkMacros.h"
#include "common/cudaUtils.h"
#include "common/logger.h"
#include "common/tensor.h"
#include "kernels/moe/moeMarlinIndicesKernels.h"
#include "kernels/moe/moe_marlin/moeMarlin.h"
#include "plugins/utils/pluginUtils.h"
#include "profiling/nvtx_wrapper.h"

#include <NvInfer.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

using namespace nvinfer1;

namespace trt_edgellm
{
namespace plugins
{

namespace
{

constexpr int32_t kInActivation{0};
constexpr int32_t kInQWeights{1};
constexpr int32_t kInBlockScales{2};
constexpr int32_t kInGlobalScale{3};
constexpr int32_t kOutOutput{4};
constexpr int32_t kNbPluginInputs{4};

constexpr char const* kPluginName{"Nvfp4A16GemmPlugin"};
constexpr char const* kPluginVersion{"1"};

constexpr int32_t kNvfp4GroupSize{16};
constexpr int32_t kDecodeBlockSize{8};
constexpr int32_t kPrefillBlockSize{32};
// The dense plugin only exercises the decode (8) and prefill (32) Marlin block sizes.
constexpr int32_t kMaxBlockSize{kPrefillBlockSize};

constexpr int32_t kFieldGemmN{0};
constexpr int32_t kFieldGemmK{1};
constexpr int32_t kFieldMaxM{2};
constexpr int32_t kNbPluginFields{3};

//! Decode (M == 1) uses the small Marlin block; everything else uses the prefill block.
int32_t getMoeBlockSize(int64_t numTokens)
{
    return numTokens == 1 ? kDecodeBlockSize : kPrefillBlockSize;
}

int64_t getTokenCount(Dims const& dims) noexcept
{
    if (dims.nbDims != 2 && dims.nbDims != 3)
    {
        return 0;
    }
    int64_t tokens = 1;
    for (int32_t index = 0; index < dims.nbDims - 1; ++index)
    {
        if (dims.d[index] <= 0 || tokens > std::numeric_limits<int64_t>::max() / dims.d[index])
        {
            return 0;
        }
        tokens *= dims.d[index];
    }
    return tokens;
}

bool haveMatchingRows(Dims const& input, Dims const& output) noexcept
{
    if (input.nbDims != output.nbDims)
    {
        return false;
    }
    for (int32_t index = 0; index < input.nbDims - 1; ++index)
    {
        if (input.d[index] != output.d[index])
        {
            return false;
        }
    }
    return true;
}

//! Block-aligned row count for a given token count and block size.
int64_t getPaddedRows(int64_t numTokens, int32_t blockSize)
{
    return static_cast<int64_t>(divUp(numTokens, static_cast<int64_t>(blockSize))) * blockSize;
}

//! Worst-case padded-row capacity across both block sizes for a profile with maxM tokens.
//! ceilDiv(maxM, kPrefillBlockSize)*kPrefillBlockSize dominates the decode case (paddedRows == 8 for M == 1).
int64_t getMaxPaddedRows(int64_t maxTokens)
{
    return std::max<int64_t>(getPaddedRows(maxTokens, kMaxBlockSize), kMaxBlockSize);
}

size_t computeWorkspaceSize(int32_t maxTokens, int32_t gemmN) noexcept
{
    try
    {
        int32_t device = 0;
        int32_t numSms = 0;
        CUDA_CHECK(cudaGetDevice(&device));
        CUDA_CHECK(cudaDeviceGetAttribute(&numSms, cudaDevAttrMultiProcessorCount, device));

        int64_t const maxPaddedRows = getMaxPaddedRows(maxTokens);
        // Decode uses the smallest block and therefore needs the largest per-block arrays for a fixed row cap.
        int64_t const maxPaddedBlocks = divUp(maxPaddedRows, static_cast<int64_t>(kDecodeBlockSize));

        int64_t marlinWorkspaceElements = 0;
        for (int32_t const blockSize : {kDecodeBlockSize, kPrefillBlockSize})
        {
            marlinWorkspaceElements = std::max(
                marlinWorkspaceElements, kernel::getMoeMarlinWorkspaceSize(maxPaddedRows, gemmN, blockSize, numSms));
        }

        size_t size = 0;
        size = accumulateWorkspaceSize(size, rt::Coords{maxPaddedRows}, DataType::kINT32);   // sortedTokenIds
        size = accumulateWorkspaceSize(size, rt::Coords{maxPaddedBlocks}, DataType::kINT32); // expertIds
        size = accumulateWorkspaceSize(size, rt::Coords{1}, DataType::kINT32);               // numTokensPostPadded
        size = accumulateWorkspaceSize(size, rt::Coords{maxPaddedRows}, DataType::kFLOAT);   // topkWeights (unused)
        size = accumulateWorkspaceSize(size, rt::Coords{marlinWorkspaceElements}, DataType::kINT32); // marlinWorkspace
        return size;
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("Failed to compute Nvfp4A16GemmPlugin workspace size: %s", e.what());
        return 0;
    }
}

} // namespace

PluginFieldCollection Nvfp4A16GemmPluginCreator::mFieldCollection{};
std::vector<PluginField> Nvfp4A16GemmPluginCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(Nvfp4A16GemmPluginCreator);

Nvfp4A16GemmPlugin::Nvfp4A16GemmPlugin(std::string const& name, int32_t gemmN, int32_t gemmK, int32_t maxM)
    : mLayerName(name)
    , mGemmN(gemmN)
    , mGemmK(gemmK)
    , mMaxM(maxM)
{
    validateAttributes();
}

Nvfp4A16GemmPlugin::Nvfp4A16GemmPlugin(std::string const& name, PluginFieldCollection const* fc)
    : mLayerName(name)
{
    if (fc == nullptr || fc->fields == nullptr || fc->nbFields <= 0)
    {
        throw std::invalid_argument("Nvfp4A16GemmPlugin: plugin field collection must not be empty");
    }

    std::array<bool, kNbPluginFields> fieldsSeen{};
    auto readIntField = [&fieldsSeen](PluginField const& field, int32_t fieldIndex) {
        if (field.data == nullptr || field.type != PluginFieldType::kINT32 || field.length != 1)
        {
            throw std::invalid_argument("Nvfp4A16GemmPlugin: integer attributes must be scalar INT32 fields");
        }
        if (fieldsSeen[fieldIndex])
        {
            throw std::invalid_argument("Nvfp4A16GemmPlugin: duplicate plugin attribute");
        }
        fieldsSeen[fieldIndex] = true;
        return *static_cast<int32_t const*>(field.data);
    };

    for (int32_t i = 0; i < fc->nbFields; ++i)
    {
        PluginField const& field = fc->fields[i];
        if (field.name == nullptr)
        {
            throw std::invalid_argument("Nvfp4A16GemmPlugin: plugin attribute name must not be null");
        }
        std::string const fieldName(field.name);
        if (fieldName == "gemm_n")
        {
            mGemmN = readIntField(field, kFieldGemmN);
        }
        else if (fieldName == "gemm_k")
        {
            mGemmK = readIntField(field, kFieldGemmK);
        }
        else if (fieldName == "max_m")
        {
            mMaxM = readIntField(field, kFieldMaxM);
        }
        else
        {
            throw std::invalid_argument("Nvfp4A16GemmPlugin: unknown plugin attribute " + fieldName);
        }
    }

    // gemm_n and gemm_k are required; max_m is optional (0 == auto: resolve from the optimization profile).
    if (!fieldsSeen[kFieldGemmN] || !fieldsSeen[kFieldGemmK])
    {
        throw std::invalid_argument("Nvfp4A16GemmPlugin: gemm_n and gemm_k attributes are required");
    }
    validateAttributes();
}

Nvfp4A16GemmPlugin::~Nvfp4A16GemmPlugin() noexcept = default;

void Nvfp4A16GemmPlugin::validateAttributes() const
{
    // Marlin instantiates only thread_n = 128/256; N must be a multiple of 128.
    if (mGemmN <= 0 || mGemmN % 128 != 0)
    {
        throw std::invalid_argument("Nvfp4A16GemmPlugin: gemm_n must be positive and divisible by 128");
    }
    // Marlin needs K divisible by thread_k (64) and by the NVFP4 group size (16).
    if (mGemmK <= 0 || mGemmK % 64 != 0 || mGemmK % kNvfp4GroupSize != 0)
    {
        throw std::invalid_argument("Nvfp4A16GemmPlugin: gemm_k must be positive and divisible by 64 (and by 16)");
    }
    if (mMaxM < 0)
    {
        throw std::invalid_argument("Nvfp4A16GemmPlugin: max_m must be non-negative (0 == auto)");
    }
}

IPluginCapability* Nvfp4A16GemmPlugin::getCapabilityInterface(PluginCapabilityType type) noexcept
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

IPluginV3* Nvfp4A16GemmPlugin::clone() noexcept
{
    try
    {
        auto* plugin = new Nvfp4A16GemmPlugin(mLayerName, mGemmN, mGemmK, mMaxM);
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("Failed to clone Nvfp4A16GemmPlugin: %s", e.what());
        return nullptr;
    }
}

char const* Nvfp4A16GemmPlugin::getPluginName() const noexcept
{
    return kPluginName;
}

char const* Nvfp4A16GemmPlugin::getPluginVersion() const noexcept
{
    return kPluginVersion;
}

char const* Nvfp4A16GemmPlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

void Nvfp4A16GemmPlugin::setPluginNamespace(char const* pluginNamespace) noexcept
{
    mNamespace = pluginNamespace == nullptr ? "" : pluginNamespace;
}

int32_t Nvfp4A16GemmPlugin::getNbOutputs() const noexcept
{
    return 1;
}

int32_t Nvfp4A16GemmPlugin::getOutputDataTypes(
    DataType* outputTypes, int32_t nbOutputs, DataType const* inputTypes, int32_t nbInputs) const noexcept
{
    (void) inputTypes;
    if (outputTypes == nullptr || nbOutputs != 1 || nbInputs != kNbPluginInputs)
    {
        LOG_ERROR("Nvfp4A16GemmPlugin: getOutputDataTypes expected %d inputs and 1 output", kNbPluginInputs);
        return -1;
    }
    outputTypes[0] = DataType::kHALF;
    return 0;
}

int32_t Nvfp4A16GemmPlugin::getOutputShapes(DimsExprs const* inputs, int32_t nbInputs, DimsExprs const* shapeInputs,
    int32_t nbShapeInputs, DimsExprs* outputs, int32_t nbOutputs, IExprBuilder& exprBuilder) noexcept
{
    if (inputs == nullptr || outputs == nullptr || nbInputs != kNbPluginInputs || nbOutputs != 1)
    {
        LOG_ERROR("Nvfp4A16GemmPlugin: getOutputShapes expected %d inputs and 1 output", kNbPluginInputs);
        return -1;
    }
    (void) shapeInputs;
    (void) nbShapeInputs;
    int32_t const rank = inputs[kInActivation].nbDims;
    if (rank != 2 && rank != 3)
    {
        return -1;
    }
    outputs[0] = inputs[kInActivation];
    outputs[0].d[rank - 1] = exprBuilder.constant(mGemmN);
    return 0;
}

bool Nvfp4A16GemmPlugin::validateTensorDesc(int32_t pos, PluginTensorDesc const& desc) const noexcept
{
    if (desc.format != TensorFormat::kLINEAR)
    {
        return false;
    }

    switch (pos)
    {
    case kInActivation:
        return desc.type == DataType::kHALF && (desc.dims.nbDims == 2 || desc.dims.nbDims == 3)
            && desc.dims.d[desc.dims.nbDims - 1] == mGemmK;
    case kInQWeights:
        // INT8 view of Marlin-packed E2M1 codes: [1, K/16, 8*N] (8 int8 == 2 int32 per output column).
        return desc.type == DataType::kINT8 && desc.dims.nbDims == 3 && desc.dims.d[0] == 1
            && desc.dims.d[1] == mGemmK / kNvfp4GroupSize && desc.dims.d[2] == 8LL * mGemmN;
    case kInBlockScales:
        // Raw E4M3 bytes in Marlin order: [1, K/16, N].
        return desc.type == DataType::kINT8 && desc.dims.nbDims == 3 && desc.dims.d[0] == 1
            && desc.dims.d[1] == mGemmK / kNvfp4GroupSize && desc.dims.d[2] == mGemmN;
    case kInGlobalScale: return desc.type == DataType::kHALF && desc.dims.nbDims == 1 && desc.dims.d[0] == 1;
    case kOutOutput:
        return desc.type == DataType::kHALF && (desc.dims.nbDims == 2 || desc.dims.nbDims == 3)
            && desc.dims.d[desc.dims.nbDims - 1] == mGemmN;
    default: return false;
    }
}

bool Nvfp4A16GemmPlugin::supportsFormatCombination(
    int32_t pos, DynamicPluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
    if (inOut == nullptr || nbInputs != kNbPluginInputs || nbOutputs != 1 || pos < 0 || pos > kOutOutput)
    {
        return false;
    }
    return validateTensorDesc(pos, inOut[pos].desc);
}

int32_t Nvfp4A16GemmPlugin::configurePlugin(
    DynamicPluginTensorDesc const* in, int32_t nbInputs, DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept
{
    try
    {
        if (in == nullptr || out == nullptr || nbInputs != kNbPluginInputs || nbOutputs != 1)
        {
            LOG_ERROR("Nvfp4A16GemmPlugin: configurePlugin expected %d inputs and 1 output", kNbPluginInputs);
            return -1;
        }
        for (int32_t pos = 0; pos < kNbPluginInputs; ++pos)
        {
            if (!validateTensorDesc(pos, in[pos].desc))
            {
                LOG_ERROR("Nvfp4A16GemmPlugin: invalid input descriptor at position %d", pos);
                return -1;
            }
        }
        if (!validateTensorDesc(kOutOutput, out[0].desc))
        {
            LOG_ERROR("Nvfp4A16GemmPlugin: invalid output descriptor");
            return -1;
        }

        Dims const& actMax = in[kInActivation].max;
        int64_t const maxTokens = getTokenCount(actMax);
        if (maxTokens <= 0 || actMax.d[actMax.nbDims - 1] != mGemmK)
        {
            LOG_ERROR("Nvfp4A16GemmPlugin: optimization profile activation dimensions are incomplete or invalid");
            return -1;
        }
        if (maxTokens > std::numeric_limits<int32_t>::max())
        {
            LOG_ERROR("Nvfp4A16GemmPlugin: profile token count overflows int32");
            return -1;
        }
        if (mMaxM == 0)
        {
            // Auto: size the workspace for the profile's maximum token count.
            mMaxM = static_cast<int32_t>(maxTokens);
        }
        else if (maxTokens > mMaxM)
        {
            LOG_ERROR("Nvfp4A16GemmPlugin: max_m=%d is insufficient for the profile; requires at least %lld", mMaxM,
                static_cast<long long>(maxTokens));
            return -1;
        }
        return 0;
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("Nvfp4A16GemmPlugin configurePlugin failed: %s", e.what());
        return -1;
    }
}

size_t Nvfp4A16GemmPlugin::getWorkspaceSize(DynamicPluginTensorDesc const* inputs, int32_t nbInputs,
    DynamicPluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept
{
    (void) outputs;
    if (inputs == nullptr || nbInputs != kNbPluginInputs || nbOutputs != 1)
    {
        LOG_ERROR("Nvfp4A16GemmPlugin: getWorkspaceSize expected %d inputs and 1 output", kNbPluginInputs);
        return 0;
    }
    int32_t effectiveMaxM = mMaxM;
    if (effectiveMaxM <= 0)
    {
        // configurePlugin normally resolves auto max_m; fall back to the profile max here for robustness.
        Dims const& actMax = inputs[kInActivation].max;
        int64_t const maxTokens = getTokenCount(actMax);
        effectiveMaxM = maxTokens > std::numeric_limits<int32_t>::max() ? 0 : static_cast<int32_t>(maxTokens);
    }
    if (effectiveMaxM <= 0)
    {
        LOG_ERROR("Nvfp4A16GemmPlugin: could not determine max token count for workspace sizing");
        return 0;
    }
    return computeWorkspaceSize(effectiveMaxM, mGemmN);
}

int32_t Nvfp4A16GemmPlugin::enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc,
    void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    try
    {
        using namespace trt_edgellm::kernel;
        using namespace trt_edgellm::rt;

        if (inputDesc == nullptr || outputDesc == nullptr || inputs == nullptr || outputs == nullptr
            || workspace == nullptr || outputs[0] == nullptr)
        {
            LOG_ERROR("Nvfp4A16GemmPlugin: enqueue received a null descriptor or buffer");
            return -1;
        }
        for (int32_t i = 0; i < kNbPluginInputs; ++i)
        {
            if (inputs[i] == nullptr || !validateTensorDesc(i, inputDesc[i]))
            {
                LOG_ERROR("Nvfp4A16GemmPlugin: invalid runtime input at position %d", i);
                return -1;
            }
        }
        if (!validateTensorDesc(kOutOutput, outputDesc[0]))
        {
            LOG_ERROR("Nvfp4A16GemmPlugin: invalid runtime output descriptor");
            return -1;
        }
        if (reinterpret_cast<uintptr_t>(inputs[kInQWeights]) % alignof(int32_t) != 0)
        {
            LOG_ERROR("Nvfp4A16GemmPlugin: packed weight buffer must be aligned for an INT32 view");
            return -1;
        }

        Dims const& actDims = inputDesc[kInActivation].dims;
        int64_t const numTokens64 = getTokenCount(actDims);
        if (numTokens64 <= 0)
        {
            LOG_ERROR("Nvfp4A16GemmPlugin: runtime token dimensions must be positive and not overflow");
            return -1;
        }
        if (numTokens64 > std::numeric_limits<int32_t>::max())
        {
            LOG_ERROR("Nvfp4A16GemmPlugin: runtime token count overflows int32");
            return -1;
        }
        int32_t const numTokens = static_cast<int32_t>(numTokens64);
        if (numTokens > mMaxM)
        {
            LOG_ERROR("Nvfp4A16GemmPlugin: runtime token count %d exceeds max_m=%d", numTokens, mMaxM);
            return -1;
        }
        if (!haveMatchingRows(actDims, outputDesc[0].dims))
        {
            LOG_ERROR("Nvfp4A16GemmPlugin: output leading dimensions must match the activation");
            return -1;
        }

        int32_t const moeBlockSize = getMoeBlockSize(numTokens);
        int64_t const paddedRows = getPaddedRows(numTokens, moeBlockSize);
        int64_t const maxPaddedRows = getMaxPaddedRows(mMaxM);
        int64_t const maxPaddedBlocks = divUp(maxPaddedRows, static_cast<int64_t>(kDecodeBlockSize));

        int32_t device = 0;
        int32_t numSms = 0;
        CUDA_CHECK(cudaGetDevice(&device));
        CUDA_CHECK(cudaDeviceGetAttribute(&numSms, cudaDevAttrMultiProcessorCount, device));

        std::byte* workspacePtr = static_cast<std::byte*>(workspace);
        int32_t* sortedTokenIds = static_cast<int32_t*>(
            assignTensorFromWorkspace(workspacePtr, {maxPaddedRows}, DataType::kINT32).rawPointer());
        int32_t* expertIds = static_cast<int32_t*>(
            assignTensorFromWorkspace(workspacePtr, {maxPaddedBlocks}, DataType::kINT32).rawPointer());
        int32_t* numTokensPostPadded
            = static_cast<int32_t*>(assignTensorFromWorkspace(workspacePtr, {1}, DataType::kINT32).rawPointer());
        float* topkWeights = static_cast<float*>(
            assignTensorFromWorkspace(workspacePtr, {maxPaddedRows}, DataType::kFLOAT).rawPointer());

        int64_t marlinWorkspaceElements = 0;
        for (int32_t const blockSize : {kDecodeBlockSize, kPrefillBlockSize})
        {
            marlinWorkspaceElements = std::max(
                marlinWorkspaceElements, getMoeMarlinWorkspaceSize(maxPaddedRows, mGemmN, blockSize, numSms));
        }
        int32_t* marlinWorkspace = static_cast<int32_t*>(
            assignTensorFromWorkspace(workspacePtr, {marlinWorkspaceElements}, DataType::kINT32).rawPointer());

        {
            NVTX_SCOPED_RANGE(nvtx_indices, "Nvfp4A16GemmPlugin::indices", nvtx_colors::ORANGE);
            launchBuildDenseMarlinIndicesKernel(sortedTokenIds, expertIds, numTokensPostPadded, topkWeights, numTokens,
                static_cast<int32_t>(paddedRows), moeBlockSize, stream);
            CUDA_CHECK(cudaGetLastError());
        }

        Tensor activation(
            const_cast<void*>(inputs[kInActivation]), {numTokens, mGemmK}, DeviceType::kGPU, DataType::kHALF);
        Tensor qWeights(const_cast<void*>(inputs[kInQWeights]), {1, mGemmK / kNvfp4GroupSize, 2 * mGemmN},
            DeviceType::kGPU, DataType::kINT32);
        Tensor blockScales(const_cast<void*>(inputs[kInBlockScales]), {1, mGemmK / kNvfp4GroupSize, mGemmN},
            DeviceType::kGPU, DataType::kINT8);
        Tensor globalScale(const_cast<void*>(inputs[kInGlobalScale]), {1}, DeviceType::kGPU, DataType::kHALF);
        Tensor output(outputs[0], {numTokens, mGemmN}, DeviceType::kGPU, DataType::kHALF);

        Tensor sortedTokenIdsTensor(sortedTokenIds, {maxPaddedRows}, DeviceType::kGPU, DataType::kINT32);
        Tensor expertIdsTensor(expertIds, {maxPaddedBlocks}, DeviceType::kGPU, DataType::kINT32);
        Tensor numTokensPostPaddedTensor(numTokensPostPadded, {1}, DeviceType::kGPU, DataType::kINT32);
        Tensor topkWeightsTensor(topkWeights, {maxPaddedRows}, DeviceType::kGPU, DataType::kFLOAT);
        Tensor marlinWorkspaceTensor(marlinWorkspace, {marlinWorkspaceElements}, DeviceType::kGPU, DataType::kINT32);

        {
            NVTX_SCOPED_RANGE(nvtx_gemm, "Nvfp4A16GemmPlugin::gemm", nvtx_colors::BLUE);
            // Single-expert, top_k = 1, no routing-weight multiply.
            moeNvfp4A16MarlinGemm(activation, output, qWeights, blockScales, globalScale, sortedTokenIdsTensor,
                expertIdsTensor, numTokensPostPaddedTensor, topkWeightsTensor, marlinWorkspaceTensor, moeBlockSize,
                /*topK=*/1, /*mulTopkWeights=*/false, stream);
            CUDA_CHECK(cudaGetLastError());
        }
        return 0;
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("Nvfp4A16GemmPlugin enqueue failed: %s", e.what());
        return -1;
    }
}

int32_t Nvfp4A16GemmPlugin::onShapeChange(
    PluginTensorDesc const* in, int32_t nbInputs, PluginTensorDesc const* out, int32_t nbOutputs) noexcept
{
    if (in == nullptr || out == nullptr || nbInputs != kNbPluginInputs || nbOutputs != 1)
    {
        LOG_ERROR("Nvfp4A16GemmPlugin: onShapeChange expected %d inputs and 1 output", kNbPluginInputs);
        return -1;
    }
    for (int32_t pos = 0; pos < kNbPluginInputs; ++pos)
    {
        if (!validateTensorDesc(pos, in[pos]))
        {
            LOG_ERROR("Nvfp4A16GemmPlugin: invalid shape-change descriptor at input %d", pos);
            return -1;
        }
    }
    if (!validateTensorDesc(kOutOutput, out[0]))
    {
        LOG_ERROR("Nvfp4A16GemmPlugin: invalid shape-change output descriptor");
        return -1;
    }

    int64_t const numTokens = getTokenCount(in[kInActivation].dims);
    if (numTokens <= 0)
    {
        LOG_ERROR("Nvfp4A16GemmPlugin: invalid runtime token dimensions");
        return -1;
    }
    if (!haveMatchingRows(in[kInActivation].dims, out[0].dims))
    {
        LOG_ERROR("Nvfp4A16GemmPlugin: runtime output shape does not match the activation");
        return -1;
    }
    if (numTokens > mMaxM)
    {
        LOG_ERROR(
            "Nvfp4A16GemmPlugin: runtime token count %lld exceeds max_m=%d", static_cast<long long>(numTokens), mMaxM);
        return -1;
    }
    return 0;
}

IPluginV3* Nvfp4A16GemmPlugin::attachToContext([[maybe_unused]] IPluginResourceContext* context) noexcept
{
    return clone();
}

PluginFieldCollection const* Nvfp4A16GemmPlugin::getFieldsToSerialize() noexcept
{
    try
    {
        mDataToSerialize.clear();
        mDataToSerialize.emplace_back("gemm_n", &mGemmN, PluginFieldType::kINT32, 1);
        mDataToSerialize.emplace_back("gemm_k", &mGemmK, PluginFieldType::kINT32, 1);
        mDataToSerialize.emplace_back("max_m", &mMaxM, PluginFieldType::kINT32, 1);
        mFCToSerialize.nbFields = static_cast<int32_t>(mDataToSerialize.size());
        mFCToSerialize.fields = mDataToSerialize.data();
        return &mFCToSerialize;
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("Failed to serialize Nvfp4A16GemmPlugin fields: %s", e.what());
        return nullptr;
    }
}

Nvfp4A16GemmPluginCreator::Nvfp4A16GemmPluginCreator()
{
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);

    mPluginAttributes.clear();
    mPluginAttributes.emplace_back("gemm_n", nullptr, PluginFieldType::kINT32, 1);
    mPluginAttributes.emplace_back("gemm_k", nullptr, PluginFieldType::kINT32, 1);
    mPluginAttributes.emplace_back("max_m", nullptr, PluginFieldType::kINT32, 1);

    mFieldCollection.nbFields = static_cast<int32_t>(mPluginAttributes.size());
    mFieldCollection.fields = mPluginAttributes.data();
}

char const* Nvfp4A16GemmPluginCreator::getPluginName() const noexcept
{
    return kPluginName;
}

char const* Nvfp4A16GemmPluginCreator::getPluginVersion() const noexcept
{
    return kPluginVersion;
}

PluginFieldCollection const* Nvfp4A16GemmPluginCreator::getFieldNames() noexcept
{
    return &mFieldCollection;
}

char const* Nvfp4A16GemmPluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

void Nvfp4A16GemmPluginCreator::setPluginNamespace(char const* pluginNamespace) noexcept
{
    mNamespace = pluginNamespace == nullptr ? "" : pluginNamespace;
}

IPluginV3* Nvfp4A16GemmPluginCreator::createPlugin(
    char const* name, PluginFieldCollection const* fc, TensorRTPhase phase) noexcept
{
    (void) phase;
    try
    {
        auto* plugin = new Nvfp4A16GemmPlugin(name == nullptr ? kPluginName : name, fc);
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("Failed to create Nvfp4A16GemmPlugin: %s", e.what());
        return nullptr;
    }
}

} // namespace plugins
} // namespace trt_edgellm
