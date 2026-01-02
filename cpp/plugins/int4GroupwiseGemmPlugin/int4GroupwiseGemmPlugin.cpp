/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "int4GroupwiseGemmPlugin.h"
#include "kernels/int4GroupwiseGemmKernels/int4GroupwiseGemm.h"
#include "plugins/utils/pluginUtils.h"

#include <cassert>
#include <cuda_fp16.h>
#include <mutex>
#include <optional>

#include <iostream>

using namespace nvinfer1;
namespace trt_edgellm
{
namespace plugins
{

namespace
{
constexpr char const* kINT4_GEMM_PLUGIN_VERSION{"1"};
constexpr char const* kINT4_GEMM_PLUGIN_NAME{"Int4GroupwiseGemmPlugin"};

// Enforce groupsize to be 128, can be further extended to support 64.
constexpr int32_t kGROUP_SIZE{128};

} // namespace

// Static class fields initialization
PluginFieldCollection Int4GroupwiseGemmPluginCreator::mFieldCollection{};
std::vector<PluginField> Int4GroupwiseGemmPluginCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(Int4GroupwiseGemmPluginCreator);

Int4GroupwiseGemmPlugin::Int4GroupwiseGemmPlugin(std::string const& name, int32_t N, int32_t K, int32_t groupSize)
    : mLayerName(name)
    , mGemmN(N)
    , mGemmK(K)
    , mGroupSize(groupSize)
{
}

Int4GroupwiseGemmPlugin::Int4GroupwiseGemmPlugin(std::string const& name, void const* data, size_t length)
    : mLayerName(name)
{
    deserializeValue(&data, &length, &mGemmN);
    deserializeValue(&data, &length, &mGemmK);
    deserializeValue(&data, &length, &mGroupSize);
}

Int4GroupwiseGemmPlugin::~Int4GroupwiseGemmPlugin() {}

IPluginV2DynamicExt* Int4GroupwiseGemmPlugin::clone() const noexcept
{
    Int4GroupwiseGemmPlugin* plugin = new Int4GroupwiseGemmPlugin(mLayerName, mGemmN, mGemmK, mGroupSize);
    return plugin;
}

char const* Int4GroupwiseGemmPlugin::getPluginType() const noexcept
{
    return kINT4_GEMM_PLUGIN_NAME;
}

char const* Int4GroupwiseGemmPlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

void Int4GroupwiseGemmPlugin::setPluginNamespace(char const* pluginNamespace) noexcept
{
    mNamespace = std::string(pluginNamespace);
}

char const* Int4GroupwiseGemmPlugin::getPluginVersion() const noexcept
{
    return kINT4_GEMM_PLUGIN_VERSION;
}

int32_t Int4GroupwiseGemmPlugin::getNbOutputs() const noexcept
{
    return 1;
}

bool Int4GroupwiseGemmPlugin::supportsFormatCombination(
    int32_t pos, nvinfer1::PluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
    // input 0: Fp16 activation tensor, input 1: packed int4 weights in type int8, input2: Fp16 scale values.
    // output 0: Fp16 computed result of the int4-woq gemm
    try
    {
        assert(nbInputs == 3 && nbOutputs == 1);
        assert(pos < (nbInputs + nbOutputs));
        auto const& tensorDesc = inOut[pos];
        bool status{true};

        switch (pos)
        {
        case 0:
        {
            status &= tensorDesc.type == DataType::kHALF;
            status &= tensorDesc.format == TensorFormat::kLINEAR;
            status &= tensorDesc.dims.nbDims == 3;
            status &= tensorDesc.dims.d[2] == mGemmK;
            break;
        }
        case 1:
        {
            // The int4 weights are packed and swizzled into a special layout with int16 [N/4, K].
            // Since TensorRT doesn't have Int16 datatype, we use int8 datatype to store the weights.
            // Therefore the type should be [N/2, K] in int8.
            status &= tensorDesc.type == DataType::kINT8;
            status &= tensorDesc.format == TensorFormat::kLINEAR;
            status &= tensorDesc.dims.nbDims == 2;
            status &= tensorDesc.dims.d[0] == mGemmN / 2;
            status &= tensorDesc.dims.d[1] == mGemmK;
            break;
        }
        case 2:
        {
            // The accepted scale for the kernel should be fp16 with [K/group_size,N]
            status &= tensorDesc.type == DataType::kHALF;
            status &= tensorDesc.format == TensorFormat::kLINEAR;
            status &= tensorDesc.dims.nbDims == 2;
            status &= tensorDesc.dims.d[0] == mGemmK / mGroupSize;
            status &= tensorDesc.dims.d[1] == mGemmN;
            break;
        }
        case 3:
        {
            status &= tensorDesc.type == DataType::kHALF;
            status &= tensorDesc.format == TensorFormat::kLINEAR;
            status &= tensorDesc.dims.nbDims == 3;
            status &= tensorDesc.dims.d[2] == mGemmN;
            break;
        }
        default: break;
        }
        return status;
    }
    catch (std::exception const& e)
    {
    }
    return false;
}

// IPluginV2Ext Methods
DataType Int4GroupwiseGemmPlugin::getOutputDataType([[maybe_unused]] int32_t index,
    [[maybe_unused]] nvinfer1::DataType const* inputTypes, [[maybe_unused]] int32_t nbInputs) const noexcept
{
    return DataType::kHALF;
}

DimsExprs Int4GroupwiseGemmPlugin::getOutputDimensions([[maybe_unused]] int32_t outputIndex,
    nvinfer1::DimsExprs const* inputs, [[maybe_unused]] int32_t nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept
{
    // Output[0] is attention result, has shape [B, S. Hq, D]. Refers to QKV shape [B, S, Hq+Hk+Hv,D]
    DimsExprs output;

    output.nbDims = 3;
    output.d[0] = inputs[0].d[0];
    output.d[1] = inputs[0].d[1];
    output.d[2] = exprBuilder.constant(mGemmN);
    return output;
}

void Int4GroupwiseGemmPlugin::configurePlugin([[maybe_unused]] nvinfer1::DynamicPluginTensorDesc const* in,
    [[maybe_unused]] int32_t nbInputs, [[maybe_unused]] nvinfer1::DynamicPluginTensorDesc const* out,
    [[maybe_unused]] int32_t nbOutputs) noexcept
{
}

// TODO: extend the worksapce calculation to a more generalized form.
size_t Int4GroupwiseGemmPlugin::getWorkspaceSize([[maybe_unused]] nvinfer1::PluginTensorDesc const* inputs,
    [[maybe_unused]] int32_t nbInputs, [[maybe_unused]] nvinfer1::PluginTensorDesc const* outputs,
    [[maybe_unused]] int32_t nbOutputs) const noexcept
{
    return 0;
}

int32_t Int4GroupwiseGemmPlugin::enqueue(nvinfer1::PluginTensorDesc const* inputDesc,
    [[maybe_unused]] nvinfer1::PluginTensorDesc const* outputDesc, void const* const* inputs, void* const* outputs,
    [[maybe_unused]] void* workspace, cudaStream_t stream) noexcept
{
    auto const& inputDesc0 = inputDesc[0];
    int32_t const M = inputDesc0.dims.d[0] * inputDesc0.dims.d[1];

    half* gemmInPtr = reinterpret_cast<half*>(const_cast<void*>(inputs[0]));
    int8_t* weightsInPtr = reinterpret_cast<int8_t*>(const_cast<void*>(inputs[1]));
    half* ScaleInPtr = reinterpret_cast<half*>(const_cast<void*>(inputs[2]));
    half* gemmOutDevicePtr = reinterpret_cast<half*>(outputs[0]);

    if (M <= 6)
    {
        trt_edgellm::kernel::gemv_forward_cuda_new(
            gemmInPtr, weightsInPtr, ScaleInPtr, gemmOutDevicePtr, M, mGemmN, mGemmK, mGroupSize, stream);
    }
    else
    {
        trt_edgellm::kernel::gemm_forward_cuda_new(
            gemmInPtr, weightsInPtr, ScaleInPtr, gemmOutDevicePtr, M, mGemmN, mGemmK, mGroupSize, stream);
    }
    return 0;
}

size_t Int4GroupwiseGemmPlugin::getSerializationSize() const noexcept
{
    return sizeof(mGemmN) + sizeof(mGemmK) + sizeof(mGroupSize);
}

void Int4GroupwiseGemmPlugin::serialize(void* buffer) const noexcept
{
    serializeValue(&buffer, mGemmN);
    serializeValue(&buffer, mGemmK);
    serializeValue(&buffer, mGroupSize);
}

int32_t Int4GroupwiseGemmPlugin::initialize() noexcept
{
    return 0;
}

void Int4GroupwiseGemmPlugin::terminate() noexcept {}

void Int4GroupwiseGemmPlugin::destroy() noexcept
{
    delete this;
}

Int4GroupwiseGemmPluginCreator::Int4GroupwiseGemmPluginCreator()
{
    static std::mutex sMutex;
    std::lock_guard<std::mutex> lock(sMutex);

    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("gemm_n", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("gemm_k", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("group_size", nullptr, PluginFieldType::kINT32, 1));

    mFieldCollection.nbFields = mPluginAttributes.size();
    mFieldCollection.fields = mPluginAttributes.data();
}

char const* Int4GroupwiseGemmPluginCreator::getPluginName() const noexcept
{
    return kINT4_GEMM_PLUGIN_NAME;
}

nvinfer1::PluginFieldCollection const* Int4GroupwiseGemmPluginCreator::getFieldNames() noexcept
{
    return &mFieldCollection;
}

void Int4GroupwiseGemmPluginCreator::setPluginNamespace(char const* libNamespace) noexcept
{
    mNamespace = libNamespace;
}

char const* Int4GroupwiseGemmPluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

char const* Int4GroupwiseGemmPluginCreator::getPluginVersion() const noexcept
{
    return kINT4_GEMM_PLUGIN_VERSION;
}

nvinfer1::IPluginV2* Int4GroupwiseGemmPluginCreator::createPlugin(
    char const* name, nvinfer1::PluginFieldCollection const* fc) noexcept
{
    try
    {
        // Read N, K attributes for the plugin.
        std::optional<int32_t> gemmN = parsePluginScalarField<int32_t>("gemm_n", fc);
        std::optional<int32_t> gemmK = parsePluginScalarField<int32_t>("gemm_k", fc);
        std::optional<int32_t> groupSize = parsePluginScalarField<int32_t>("group_size", fc);

        bool checkRequiredFields = gemmN.has_value() && gemmK.has_value() && groupSize.has_value();
        if (!checkRequiredFields)
        {
            return nullptr;
        }

        Int4GroupwiseGemmPlugin* plugin
            = new Int4GroupwiseGemmPlugin(std::string(name), gemmN.value(), gemmK.value(), groupSize.value());
        return plugin;
    }
    catch (std::exception const& e)
    {
    }
    return nullptr;
}

nvinfer1::IPluginV2* Int4GroupwiseGemmPluginCreator::deserializePlugin(
    char const* name, void const* serialData, size_t serialLength) noexcept
{
    try
    {
        return new Int4GroupwiseGemmPlugin(name, serialData, serialLength);
    }
    catch (std::exception const& e)
    {
    }
    return nullptr;
}

} // namespace plugins
} // namespace trt_edgellm